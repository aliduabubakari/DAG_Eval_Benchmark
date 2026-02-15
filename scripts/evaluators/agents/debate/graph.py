# scripts/evaluators/agents/debate/graph.py

"""
LangGraph-based Debate Architecture.

Flow:
┌──────────┐     ┌────────┐     ┌──────────┐     ┌─────────────┐
│ Proposer │ ──▶ │ Critic │ ──▶ │ Defender │ ──▶ │ Adjudicator │
└──────────┘     └────────┘     └──────────┘     └─────────────┘
     │                                                  │
     │              Issues + Evidence                   │
     └──────────────────────────────────────────────────┘
                    Final Verdicts
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from pathlib import Path

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# Import schemas
import sys
_THIS = Path(__file__).resolve()
SCRIPTS_DIR = _THIS
while SCRIPTS_DIR.name != "scripts" and SCRIPTS_DIR != SCRIPTS_DIR.parent:
    SCRIPTS_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluators.agents.schemas import (
    ProposedIssue,
    ProposerOutput,
    Challenge,
    CriticOutput,
    Defense,
    DefenderOutput,
    AdjudicatorVerdict,
    AdjudicatorOutput,
    GatePredictions,
)

logger = logging.getLogger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class DebateState(TypedDict):
    """State passed through the debate graph."""
    # Input
    code: str
    file_path: str
    
    # Proposer outputs
    proposed_issues: List[Dict[str, Any]]
    initial_gate_predictions: Dict[str, bool]
    analysis_summary: str
    
    # Critic outputs
    challenges: List[Dict[str, Any]]
    accepted_without_challenge: List[str]
    
    # Defender outputs
    defenses: List[Dict[str, Any]]
    
    # Adjudicator outputs
    final_verdicts: List[Dict[str, Any]]
    final_gate_predictions: Dict[str, bool]
    
    # Metadata
    token_usage: Dict[str, int]
    errors: List[str]


def merge_token_usage(current: Dict[str, int], new: Dict[str, int]) -> Dict[str, int]:
    """Merge token usage dicts."""
    return {
        "input_tokens": current.get("input_tokens", 0) + new.get("input_tokens", 0),
        "output_tokens": current.get("output_tokens", 0) + new.get("output_tokens", 0),
    }


# ============================================================================
# AGENT PROMPTS
# ============================================================================

PROPOSER_SYSTEM = """You are a meticulous code reviewer. Your job is to identify ALL potential issues in the code.

For each issue you MUST provide:
1. A unique ID (ISS-001, ISS-002, etc.)
2. Specific evidence from the code (exact line content or code snippet)
3. Clear reasoning for why this is an issue

Be thorough but avoid speculation. Only report issues you can back up with evidence."""

PROPOSER_USER = """Analyze this code and identify all issues.

```python
{code}
```

Respond with JSON matching this schema:
{{
  "issues": [
    {{
      "id": "ISS-001",
      "category": "syntax|import|type_error|security|complexity|style|naming|documentation|error_handling|best_practice|unused|undefined|orchestrator_structure|orchestrator_config",
      "severity": "critical|major|minor|info",
      "message": "Description of the issue",
      "line": <line_number or null>,
      "evidence": "Exact code snippet or line content proving the issue",
      "reasoning": "Why this is an issue"
    }}
  ],
  "gate_predictions": {{
    "syntax_valid": true/false,
    "imports_resolve": true/false,
    "instantiates": true/false,
    "has_structure": true/false
  }},
  "analysis_summary": "Brief summary of code quality"
}}"""

CRITIC_SYSTEM = """You are a skeptical code review critic. Your job is to challenge weak or incorrect issue reports.

For each proposed issue, determine if:
1. The evidence is weak or non-existent
2. It's a false positive (not actually an issue)
3. The severity is wrong
4. The category is wrong
5. Or it's valid and should be accepted

Be rigorous but fair. Don't reject valid issues just to be difficult."""

CRITIC_USER = """Review these proposed issues and challenge any that are weak or incorrect.

## Code:
```python
{code}
```

## Proposed Issues:
{issues_json}

For each issue, provide a challenge OR accept it.

Respond with JSON:
{{
  "challenges": [
    {{
      "issue_id": "ISS-001",
      "challenge_type": "evidence_weak|false_positive|wrong_severity|wrong_category|valid",
      "challenge_reasoning": "Why this issue is being challenged",
      "questions": ["Specific question 1", "Specific question 2"]
    }}
  ],
  "accepted_issues": ["ISS-002", "ISS-003"]
}}"""

DEFENDER_SYSTEM = """You are defending the proposed code issues against challenges.

For each challenge:
1. If the challenge is valid, acknowledge it and withdraw the issue
2. If you can provide stronger evidence, do so
3. If the severity/category should be adjusted, propose the change
4. Stand firm on valid issues with strong evidence

Be honest - withdraw issues that can't be defended."""

DEFENDER_USER = """Defend or withdraw issues based on these challenges.

## Code:
```python
{code}
```

## Original Issues:
{issues_json}

## Challenges:
{challenges_json}

Respond with JSON:
{{
  "defenses": [
    {{
      "issue_id": "ISS-001",
      "defense_successful": true/false,
      "defense_reasoning": "Why the issue stands or should be withdrawn",
      "updated_evidence": "Additional evidence if needed, or null",
      "updated_severity": "critical|major|minor|info or null if unchanged",
      "withdraw": false
    }}
  ]
}}"""

ADJUDICATOR_SYSTEM = """You are the final adjudicator in a code review debate.

Given the original issues, challenges, and defenses, make final verdicts:
1. ACCEPTED: Issue is valid as-is or with modifications
2. REJECTED: Issue is a false positive or lacks evidence
3. MODIFIED: Issue is valid but needs severity/category adjustment

Your verdicts are final. Be fair and evidence-based."""

ADJUDICATOR_USER = """Make final verdicts on these debated issues.

## Code:
```python
{code}
```

## Debate Summary:
### Original Issues:
{issues_json}

### Challenges:
{challenges_json}

### Defenses:
{defenses_json}

### Issues Accepted Without Challenge:
{accepted_ids}

Respond with JSON:
{{
  "verdicts": [
    {{
      "issue_id": "ISS-001",
      "final_verdict": "accepted|rejected|modified",
      "confidence": 0.0-1.0,
      "final_category": "category if modified, else null",
      "final_severity": "severity if modified, else null",
      "final_message": "updated message if modified, else null",
      "reasoning": "Brief explanation of verdict"
    }}
  ],
  "final_gate_predictions": {{
    "syntax_valid": true/false,
    "imports_resolve": true/false,
    "instantiates": true/false,
    "has_structure": true/false
  }}
}}"""


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

class DebateGraphBuilder:
    """Builds the debate graph with injected LLM provider."""
    
    def __init__(self, llm_provider):
        self.provider = llm_provider
    
    def _call_llm(self, system: str, user: str) -> tuple[str, Dict[str, int]]:
        """Call LLM and return response + token usage."""
        response, usage = self.provider.generate_completion(system, user)
        return response, {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
        }
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        t = text.strip()
        if "```json" in t:
            t = t.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in t:
            t = t.split("```", 1)[1].split("```", 1)[0]
        
        # Find JSON object
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start:end + 1]
        
        return json.loads(t)
    
    def proposer_node(self, state: DebateState) -> Dict[str, Any]:
        """Proposer agent: identify issues."""
        try:
            prompt = PROPOSER_USER.format(code=state["code"])
            response, usage = self._call_llm(PROPOSER_SYSTEM, prompt)
            parsed = self._parse_json(response)
            
            return {
                "proposed_issues": parsed.get("issues", []),
                "initial_gate_predictions": parsed.get("gate_predictions", {}),
                "analysis_summary": parsed.get("analysis_summary", ""),
                "token_usage": merge_token_usage(state.get("token_usage", {}), usage),
            }
        except Exception as e:
            logger.error(f"Proposer failed: {e}")
            return {
                "proposed_issues": [],
                "initial_gate_predictions": {},
                "analysis_summary": "",
                "errors": state.get("errors", []) + [f"Proposer: {e}"],
                "token_usage": state.get("token_usage", {}),
            }
    
    def critic_node(self, state: DebateState) -> Dict[str, Any]:
        """Critic agent: challenge weak issues."""
        if not state.get("proposed_issues"):
            return {
                "challenges": [],
                "accepted_without_challenge": [],
            }
        
        try:
            issues_json = json.dumps(state["proposed_issues"], indent=2)
            prompt = CRITIC_USER.format(code=state["code"], issues_json=issues_json)
            response, usage = self._call_llm(CRITIC_SYSTEM, prompt)
            parsed = self._parse_json(response)
            
            return {
                "challenges": parsed.get("challenges", []),
                "accepted_without_challenge": parsed.get("accepted_issues", []),
                "token_usage": merge_token_usage(state.get("token_usage", {}), usage),
            }
        except Exception as e:
            logger.error(f"Critic failed: {e}")
            # On failure, accept all issues without challenge
            return {
                "challenges": [],
                "accepted_without_challenge": [i.get("id") for i in state.get("proposed_issues", []) if i.get("id")],
                "errors": state.get("errors", []) + [f"Critic: {e}"],
                "token_usage": state.get("token_usage", {}),
            }
    
    def defender_node(self, state: DebateState) -> Dict[str, Any]:
        """Defender agent: defend or withdraw issues."""
        challenges = [c for c in state.get("challenges", []) 
                      if c.get("challenge_type") != "valid"]
        
        if not challenges:
            return {"defenses": []}
        
        try:
            issues_json = json.dumps(state["proposed_issues"], indent=2)
            challenges_json = json.dumps(challenges, indent=2)
            prompt = DEFENDER_USER.format(
                code=state["code"],
                issues_json=issues_json,
                challenges_json=challenges_json,
            )
            response, usage = self._call_llm(DEFENDER_SYSTEM, prompt)
            parsed = self._parse_json(response)
            
            return {
                "defenses": parsed.get("defenses", []),
                "token_usage": merge_token_usage(state.get("token_usage", {}), usage),
            }
        except Exception as e:
            logger.error(f"Defender failed: {e}")
            return {
                "defenses": [],
                "errors": state.get("errors", []) + [f"Defender: {e}"],
                "token_usage": state.get("token_usage", {}),
            }
    
    def adjudicator_node(self, state: DebateState) -> Dict[str, Any]:
        """Adjudicator agent: make final verdicts."""
        if not state.get("proposed_issues"):
            return {
                "final_verdicts": [],
                "final_gate_predictions": state.get("initial_gate_predictions", {}),
            }
        
        try:
            issues_json = json.dumps(state["proposed_issues"], indent=2)
            challenges_json = json.dumps(state.get("challenges", []), indent=2)
            defenses_json = json.dumps(state.get("defenses", []), indent=2)
            accepted_ids = json.dumps(state.get("accepted_without_challenge", []))
            
            prompt = ADJUDICATOR_USER.format(
                code=state["code"],
                issues_json=issues_json,
                challenges_json=challenges_json,
                defenses_json=defenses_json,
                accepted_ids=accepted_ids,
            )
            response, usage = self._call_llm(ADJUDICATOR_SYSTEM, prompt)
            parsed = self._parse_json(response)
            
            return {
                "final_verdicts": parsed.get("verdicts", []),
                "final_gate_predictions": parsed.get("final_gate_predictions", 
                                                      state.get("initial_gate_predictions", {})),
                "token_usage": merge_token_usage(state.get("token_usage", {}), usage),
            }
        except Exception as e:
            logger.error(f"Adjudicator failed: {e}")
            # On failure, accept all non-challenged issues
            fallback_verdicts = []
            accepted = set(state.get("accepted_without_challenge", []))
            for issue in state.get("proposed_issues", []):
                issue_id = issue.get("id")
                if issue_id in accepted:
                    fallback_verdicts.append({
                        "issue_id": issue_id,
                        "final_verdict": "accepted",
                        "confidence": 0.7,
                        "reasoning": "Accepted by default (adjudicator error)",
                    })
            
            return {
                "final_verdicts": fallback_verdicts,
                "final_gate_predictions": state.get("initial_gate_predictions", {}),
                "errors": state.get("errors", []) + [f"Adjudicator: {e}"],
                "token_usage": state.get("token_usage", {}),
            }
    
    def build(self) -> StateGraph:
        """Build and compile the debate graph."""
        graph = StateGraph(DebateState)
        
        # Add nodes
        graph.add_node("proposer", self.proposer_node)
        graph.add_node("critic", self.critic_node)
        graph.add_node("defender", self.defender_node)
        graph.add_node("adjudicator", self.adjudicator_node)
        
        # Add edges (linear flow)
        graph.set_entry_point("proposer")
        graph.add_edge("proposer", "critic")
        graph.add_edge("critic", "defender")
        graph.add_edge("defender", "adjudicator")
        graph.add_edge("adjudicator", END)
        
        return graph.compile()