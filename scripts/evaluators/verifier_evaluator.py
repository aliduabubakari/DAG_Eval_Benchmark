# scripts/evaluators/evaluations/verifier_evaluator.py

#!/usr/bin/env python3
"""
E7: Verifier Architecture (Tool-verified LLM)

Flow:
1. LLM proposes issues with VERIFIABLE CLAIMS
2. Deterministic tools verify each claim
3. Only issues with verified claims are kept

This combines LLM semantic understanding with deterministic proof,
yielding high precision.
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
from evaluators.agents.verifier.verification_tools import VerificationEngine, VerificationResult
from utils.config_loader import load_config, validate_api_keys, get_project_root
from utils.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


# ============================================================================
# PROMPTS
# ============================================================================

PROPOSER_SYSTEM = """You are a code analyzer that produces VERIFIABLE claims about code issues.

For each issue you identify, you MUST provide specific, testable claims that can be verified programmatically.

## Claim Types (use exactly these):
- line_exists: Assert specific code exists at a line
- variable_undefined: Assert a variable is used but never defined
- variable_unused: Assert a variable is defined but never used
- import_unresolvable: Assert an import cannot be resolved
- function_undefined: Assert a function is called but not defined
- syntax_error: Assert there is a syntax error
- security_pattern: Assert a security anti-pattern exists (hardcoded_password, hardcoded_secret, eval_usage, sql_injection)
- complexity_threshold: Assert a function exceeds complexity threshold
- import_unused: Assert an import is never used

Your claims will be VERIFIED by deterministic tools. False claims will be rejected.
Be precise and conservative."""

PROPOSER_USER = """Analyze this code and identify issues with VERIFIABLE claims.

```python
{code}
```

For each issue, provide specific claims that can be checked by tools.

Respond with JSON only:
{{
  "issues": [
    {{
      "id": "ISS-001",
      "category": "syntax|import|type_error|security|complexity|style|naming|documentation|error_handling|best_practice|unused|undefined|orchestrator_structure|orchestrator_config",
      "severity": "critical|major|minor|info",
      "message": "Clear description of the issue",
      "line": <line_number or null>,
      "evidence": "The exact code that demonstrates the issue",
      "claims": [
        {{
          "claim_type": "variable_undefined|variable_unused|import_unresolvable|...",
          "target": "specific_name_or_pattern",
          "line": <line_number or null>,
          "context": "additional context if needed"
        }}
      ]
    }}
  ],
  "gate_predictions": {{
    "syntax_valid": true/false,
    "imports_resolve": true/false,
    "instantiates": true/false,
    "has_structure": true/false
  }}
}}

IMPORTANT: Every issue MUST have at least one verifiable claim. Issues without claims will be rejected."""


# ============================================================================
# VERIFIER EVALUATOR
# ============================================================================

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


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences."""
    t = (text or "").strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0]
    return t.strip()


def _extract_json_object(text: str) -> str:
    """Extract JSON object from text."""
    t = _strip_code_fences(text)
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start:end + 1]
    return t


class VerifierEvaluator:
    """
    Tool-verified LLM evaluator.
    
    Key benefits:
    - Very high precision (claims are verified)
    - Combines LLM insight with deterministic proof
    - Transparent: you can see exactly what was verified
    
    Trade-offs:
    - Lower recall (unverifiable issues are dropped)
    - Some semantic issues can't be verified deterministically
    """
    
    NAME = "verifier_tool_verified_llm"
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_alias: Optional[str] = None,
        max_code_chars: int = 15000,
        min_verification_score: float = 0.5,  # Min fraction of claims that must verify
        require_all_claims: bool = False,     # If True, ALL claims must verify
        config: Optional[Dict[str, Any]] = None,
    ):
        self.max_code_chars = int(max_code_chars)
        self.min_verification_score = float(min_verification_score)
        self.require_all_claims = bool(require_all_claims)
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
    
    def evaluate(self, file_path: Path) -> EvaluatorOutput:
        """Run tool-verified evaluation."""
        start_time = time.time()
        file_path = Path(file_path)
        
        output = EvaluatorOutput(
            evaluator_name=f"{self.NAME}_{self.model_info.get('model_key', 'default')}",
            file_path=str(file_path),
        )
        output.metadata["llm"] = self.model_info
        output.metadata["architecture"] = "verifier_tool_verified"
        output.metadata["verification_settings"] = {
            "min_score": self.min_verification_score,
            "require_all": self.require_all_claims,
        }
        
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
        
        original_code = code
        if len(code) > self.max_code_chars:
            code = code[:self.max_code_chars] + "\n# ... truncated ...\n"
            output.metadata["truncated"] = True
        
        # Step 1: Get LLM proposals
        try:
            proposals, token_usage, gates = self._get_proposals(code)
            output.tokens_used = token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0)
            output.metadata["token_usage"] = token_usage
            output.metadata["proposed_count"] = len(proposals)
        except Exception as e:
            logger.exception(f"LLM proposal failed: {e}")
            output.issues.append(DetectedIssue(
                category=EvalCategory.ERROR_HANDLING.value,
                severity=EvalSeverity.INFO.value,
                message=f"LLM proposal error: {e}",
                source="verifier_error",
            ))
            output.gate_predictions = GatePrediction()
            output.execution_time_ms = (time.time() - start_time) * 1000
            return output
        
        # Step 2: Verify each proposal
        engine = VerificationEngine(original_code)  # Use original, not truncated
        verified_issues = []
        verification_details = []
        
        for proposal in proposals:
            issue_id = proposal.get("id", "unknown")
            claims = proposal.get("claims", [])
            
            if not claims:
                # No claims = cannot verify
                verification_details.append({
                    "issue_id": issue_id,
                    "status": "rejected_no_claims",
                    "claims_verified": 0,
                    "claims_total": 0,
                })
                continue
            
            # Verify each claim
            claim_results = []
            verified_count = 0
            
            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                
                result = engine.verify_claim(
                    claim_type=claim.get("claim_type", ""),
                    target=str(claim.get("target", "")),
                    line=claim.get("line"),
                    context=claim.get("context"),
                )
                
                claim_results.append({
                    "claim_type": claim.get("claim_type"),
                    "target": claim.get("target"),
                    "verified": result.verified,
                    "method": result.method,
                    "details": result.details,
                })
                
                if result.verified:
                    verified_count += 1
            
            # Calculate verification score
            verification_score = verified_count / len(claims) if claims else 0.0
            
            # Determine if issue passes
            if self.require_all_claims:
                passes = verified_count == len(claims)
            else:
                passes = verification_score >= self.min_verification_score
            
            verification_details.append({
                "issue_id": issue_id,
                "status": "accepted" if passes else "rejected",
                "claims_verified": verified_count,
                "claims_total": len(claims),
                "verification_score": verification_score,
                "claim_results": claim_results,
            })
            
            if passes:
                verified_issues.append({
                    **proposal,
                    "verification_score": verification_score,
                    "verified_claims": verified_count,
                    "total_claims": len(claims),
                })
        
        # Store verification details
        output.metadata["verification_details"] = verification_details
        output.metadata["verified_count"] = len(verified_issues)
        output.metadata["rejected_count"] = len(proposals) - len(verified_issues)
        
        # Convert to DetectedIssue
        for v in verified_issues:
            output.issues.append(DetectedIssue(
                category=self._validate_category(v.get("category")),
                severity=self._validate_severity(v.get("severity")),
                message=str(v.get("message", "")),
                line=v.get("line"),
                evidence=v.get("evidence"),
                confidence=float(v.get("verification_score", 1.0)),
                source=f"verifier_score_{v.get('verification_score', 1.0):.2f}",
            ))
        
        # Gate predictions
        output.gate_predictions = GatePrediction(
            syntax_valid=bool(gates.get("syntax_valid", True)),
            imports_resolve=bool(gates.get("imports_resolve", True)),
            instantiates=bool(gates.get("instantiates", True)),
            has_structure=bool(gates.get("has_structure", True)),
        )
        
        output.execution_time_ms = (time.time() - start_time) * 1000
        return output
    
    def _get_proposals(self, code: str) -> tuple[List[Dict], Dict[str, int], Dict[str, bool]]:
        """Get issue proposals from LLM."""
        prompt = PROPOSER_USER.format(code=code)
        response, usage = self.provider.generate_completion(PROPOSER_SYSTEM, prompt)
        
        token_usage = {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
        }
        
        # Parse response
        blob = _extract_json_object(response)
        parsed = json.loads(blob)
        
        issues = parsed.get("issues", [])
        gates = parsed.get("gate_predictions", {})
        
        return issues, token_usage, gates
    
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
    
    parser = argparse.ArgumentParser(description="Run Verifier Evaluator")
    parser.add_argument("file", help="Python file to evaluate")
    parser.add_argument("--config", help="Path to LLM config")
    parser.add_argument("--model", help="Model alias from config")
    parser.add_argument("--min-score", type=float, default=0.5,
                        help="Minimum verification score (0.0-1.0)")
    parser.add_argument("--require-all", action="store_true",
                        help="Require all claims to verify")
    args = parser.parse_args()
    
    evaluator = VerifierEvaluator(
        config_path=args.config,
        model_alias=args.model,
        min_verification_score=args.min_score,
        require_all_claims=args.require_all,
    )
    
    result = evaluator.evaluate(Path(args.file))
    
    print("\n" + "=" * 60)
    print("VERIFIER EVALUATION RESULTS")
    print("=" * 60)
    print(f"File: {result.file_path}")
    print(f"Evaluator: {result.evaluator_name}")
    print(f"Time: {result.execution_time_ms:.0f}ms")
    print(f"Tokens: {result.tokens_used}")
    
    print(f"\nProposed: {result.metadata.get('proposed_count', 0)}")
    print(f"Verified: {result.metadata.get('verified_count', 0)}")
    print(f"Rejected: {result.metadata.get('rejected_count', 0)}")
    
    print(f"\nGate Predictions: {result.gate_predictions.to_dict()}")
    
    print(f"\nVerified Issues: {len(result.issues)}")
    for i, issue in enumerate(result.issues, 1):
        print(f"\n  [{i}] {issue.severity.upper()} - {issue.category}")
        print(f"      {issue.message}")
        if issue.line:
            print(f"      Line: {issue.line}")
        print(f"      Confidence: {issue.confidence:.2f}")
        print(f"      Source: {issue.source}")
    
    # Show verification details
    print("\n" + "-" * 60)
    print("VERIFICATION DETAILS")
    print("-" * 60)
    for detail in result.metadata.get("verification_details", []):
        status_icon = "✓" if detail["status"] == "accepted" else "✗"
        print(f"\n{status_icon} {detail['issue_id']}: {detail['status']}")
        print(f"  Claims: {detail['claims_verified']}/{detail['claims_total']}")
        if detail.get("claim_results"):
            for cr in detail["claim_results"]:
                v_icon = "✓" if cr["verified"] else "✗"
                print(f"    {v_icon} {cr['claim_type']}: {cr['target']}")
                print(f"      Method: {cr['method']}, Details: {cr['details']}")