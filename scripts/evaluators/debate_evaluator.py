# scripts/evaluators/evaluations/debate_evaluator.py

#!/usr/bin/env python3
"""
E6: Debate / Critic-Defender Evaluator (Agentic)

Uses a multi-agent debate architecture:
1. Proposer: Identifies issues with evidence
2. Critic: Challenges weak/incorrect issues
3. Defender: Defends or withdraws issues
4. Adjudicator: Makes final verdicts

This reduces hallucinations by requiring evidence and defense.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- bootstrap ---
import sys
_THIS = Path(__file__).resolve()
SCRIPTS_DIR = _THIS
while SCRIPTS_DIR.name != "scripts" and SCRIPTS_DIR != SCRIPTS_DIR.parent:
    SCRIPTS_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# -----------------

from evaluators.output_schema import (
    EvaluatorOutput,
    DetectedIssue,
    GatePrediction,
    EvalCategory,
    EvalSeverity,
)
from evaluators.agents.debate.graph import DebateGraphBuilder, DebateState
from utils.config_loader import load_config, validate_api_keys, get_project_root
from utils.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


def _resolve_llm_config_path(config_path: Optional[str]) -> Path:
    """Resolve LLM config file path."""
    if config_path:
        p = Path(config_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"LLM config not found: {config_path}")

    root = Path(get_project_root())
    for c in [
        root / "config_llm.json",
        root / "config_llm_example.json",
        Path("config_llm.json"),
        Path("config_llm_example.json"),
    ]:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find config_llm.json or config_llm_example.json")


class DebateEvaluator:
    """
    Debate-based evaluator using Critic-Defender architecture.
    
    Key benefits:
    - Reduces hallucinations (issues must survive criticism)
    - Higher precision (false positives get challenged)
    - Evidence-based verdicts
    
    Trade-offs:
    - 4x LLM calls per file (higher cost)
    - Slower than single-LLM
    """
    
    NAME = "debate_critic_defender"
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_alias: Optional[str] = None,
        max_code_chars: int = 12000,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.max_code_chars = int(max_code_chars)
        self.config = config or {}
        
        # Load LLM config
        cfg_path = _resolve_llm_config_path(config_path)
        cfg = load_config(str(cfg_path), default_config=None)
        if not cfg:
            raise ValueError(f"Failed to load LLM config: {cfg_path}")
        if not validate_api_keys(cfg):
            raise ValueError("LLM API key validation failed.")
        
        # Create provider
        self.provider = LLMProvider(cfg, model_key=model_alias)
        self.model_info = self.provider.get_model_info()
        
        # Build debate graph
        builder = DebateGraphBuilder(self.provider)
        self.graph = builder.build()
    
    def evaluate(self, file_path: Path) -> EvaluatorOutput:
        """Run debate-based evaluation."""
        start_time = time.time()
        file_path = Path(file_path)
        
        output = EvaluatorOutput(
            evaluator_name=f"{self.NAME}_{self.model_info.get('model_key', 'default')}",
            file_path=str(file_path),
        )
        output.metadata["llm"] = self.model_info
        output.metadata["architecture"] = "debate_critic_defender"
        
        # Read code
        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            output.issues.append(DetectedIssue(
                category=EvalCategory.SYNTAX.value,
                severity=EvalSeverity.CRITICAL.value,
                message=f"Cannot read file: {e}",
                source="io",
            ))
            output.execution_time_ms = (time.time() - start_time) * 1000
            return output
        
        # Truncate if needed
        if len(code) > self.max_code_chars:
            code = code[:self.max_code_chars] + "\n# ... truncated ...\n"
            output.metadata["truncated"] = True
        
        # Initialize state
        initial_state: DebateState = {
            "code": code,
            "file_path": str(file_path),
            "proposed_issues": [],
            "initial_gate_predictions": {},
            "analysis_summary": "",
            "challenges": [],
            "accepted_without_challenge": [],
            "defenses": [],
            "final_verdicts": [],
            "final_gate_predictions": {},
            "token_usage": {"input_tokens": 0, "output_tokens": 0},
            "errors": [],
        }
        
        # Run the debate graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Extract results
            output.metadata["debate_trace"] = {
                "proposed_count": len(final_state.get("proposed_issues", [])),
                "challenged_count": len([c for c in final_state.get("challenges", []) 
                                        if c.get("challenge_type") != "valid"]),
                "accepted_without_challenge": len(final_state.get("accepted_without_challenge", [])),
                "defenses_count": len(final_state.get("defenses", [])),
                "final_verdicts_count": len(final_state.get("final_verdicts", [])),
                "errors": final_state.get("errors", []),
            }
            
            # Process final verdicts into issues
            output.issues = self._process_verdicts(
                final_state.get("final_verdicts", []),
                final_state.get("proposed_issues", []),
            )
            
            # Gate predictions
            gates = final_state.get("final_gate_predictions", {})
            output.gate_predictions = GatePrediction(
                syntax_valid=bool(gates.get("syntax_valid", True)),
                imports_resolve=bool(gates.get("imports_resolve", True)),
                instantiates=bool(gates.get("instantiates", True)),
                has_structure=bool(gates.get("has_structure", True)),
            )
            
            # Token usage
            usage = final_state.get("token_usage", {})
            output.tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            output.metadata["token_usage"] = usage
            
        except Exception as e:
            logger.exception(f"Debate evaluation failed: {e}")
            output.issues.append(DetectedIssue(
                category=EvalCategory.ERROR_HANDLING.value,
                severity=EvalSeverity.INFO.value,
                message=f"Debate evaluation error: {e}",
                source="debate_error",
            ))
            output.gate_predictions = GatePrediction()
        
        output.execution_time_ms = (time.time() - start_time) * 1000
        return output
    
    def _process_verdicts(
        self,
        verdicts: List[Dict[str, Any]],
        original_issues: List[Dict[str, Any]],
    ) -> List[DetectedIssue]:
        """Convert verdicts to DetectedIssue list."""
        # Build lookup for original issues
        issues_by_id = {i.get("id"): i for i in original_issues if i.get("id")}
        
        result = []
        for verdict in verdicts:
            if not isinstance(verdict, dict):
                continue
            
            final_verdict = verdict.get("final_verdict", "").lower()
            if final_verdict == "rejected":
                continue
            
            issue_id = verdict.get("issue_id")
            original = issues_by_id.get(issue_id, {})
            
            # Use modified values if present, else original
            category = verdict.get("final_category") or original.get("category", "best_practice")
            severity = verdict.get("final_severity") or original.get("severity", "minor")
            message = verdict.get("final_message") or original.get("message", "")
            
            result.append(DetectedIssue(
                category=self._validate_category(category),
                severity=self._validate_severity(severity),
                message=message,
                line=original.get("line"),
                evidence=original.get("evidence"),
                confidence=float(verdict.get("confidence", 0.8)),
                source=f"debate_{final_verdict}",
            ))
        
        return result
    
    def _validate_category(self, cat: Any) -> str:
        """Validate and normalize category."""
        cat_s = str(cat or "").strip().lower()
        valid = {c.value for c in EvalCategory}
        return cat_s if cat_s in valid else EvalCategory.BEST_PRACTICE.value
    
    def _validate_severity(self, sev: Any) -> str:
        """Validate and normalize severity."""
        sev_s = str(sev or "").strip().lower()
        valid = {s.value for s in EvalSeverity}
        return sev_s if sev_s in valid else EvalSeverity.MINOR.value


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Run Debate Evaluator")
    parser.add_argument("file", help="Python file to evaluate")
    parser.add_argument("--config", help="Path to LLM config")
    parser.add_argument("--model", help="Model alias from config")
    args = parser.parse_args()
    
    evaluator = DebateEvaluator(
        config_path=args.config,
        model_alias=args.model,
    )
    
    result = evaluator.evaluate(Path(args.file))
    
    print("\n" + "=" * 60)
    print("DEBATE EVALUATION RESULTS")
    print("=" * 60)
    print(f"File: {result.file_path}")
    print(f"Evaluator: {result.evaluator_name}")
    print(f"Time: {result.execution_time_ms:.0f}ms")
    print(f"Tokens: {result.tokens_used}")
    print(f"\nDebate Trace: {json.dumps(result.metadata.get('debate_trace', {}), indent=2)}")
    print(f"\nGate Predictions: {result.gate_predictions.to_dict()}")
    print(f"\nIssues Found: {len(result.issues)}")
    for i, issue in enumerate(result.issues, 1):
        print(f"\n  [{i}] {issue.severity.upper()} - {issue.category}")
        print(f"      {issue.message}")
        if issue.line:
            print(f"      Line: {issue.line}")
        print(f"      Confidence: {issue.confidence:.2f}")
        print(f"      Source: {issue.source}")