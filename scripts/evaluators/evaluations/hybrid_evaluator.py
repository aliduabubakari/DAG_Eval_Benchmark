#!/usr/bin/env python3
"""
E5: Hybrid Evaluator (deterministic proposer -> LLM judge)

Hardened:
- Judge sees line-numbered code
- Judge output is robustly parsed (strip trailing commas + single repair call)
- Token accounting includes repair call
- Strict evidence sanitization and integer line coercion
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- bootstrap ---
import sys
_THIS = Path(__file__).resolve()
SCRIPTS_DIR = _THIS
while SCRIPTS_DIR.name != "scripts" and SCRIPTS_DIR != SCRIPTS_DIR.parent:
    SCRIPTS_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# -------------

from evaluators.output_schema import (
    EvaluatorOutput,
    DetectedIssue,
    GatePrediction,
    EvalCategory,
    EvalSeverity,
)
from evaluators.deterministic_evaluator import DeterministicEvaluator
from evaluators.evaluations.deterministic_heuristic_evaluator import DeterministicHeuristicEvaluator
from utils.config_loader import load_config, validate_api_keys, get_project_root
from utils.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """You are a code quality judge.

You are given:
1) The code (with line numbers)
2) A list of candidate issues proposed by an automated proposer

Your job:
- Decide which candidates are real vs false positives
- Fix category/severity if needed (use the same taxonomy)
- Provide integer line numbers from the code margin when possible
- Optionally add a few obvious missed issues (be conservative)

## Taxonomy (use exactly these categories):
- syntax
- import
- type_error
- security
- complexity
- style
- naming
- documentation
- error_handling
- best_practice
- unused
- undefined
- orchestrator_structure
- orchestrator_config

Severities:
- critical, major, minor, info

## IMPORTANT: Line Numbers
Code is shown as:
"  12 | actual code ..."
If you can identify a line:
- set "line" to integer 12
- do NOT guess
- else set null
- line must be integer or null (never string)

## IMPORTANT: Evidence Field Safety
Set "evidence": null unless it is a SINGLE LINE snippet that contains:
- no newlines
- no double quotes (")
- no backticks
If unsure, evidence must be null.

## Code (with line numbers):
```python
{numbered_code}
```

## Candidate Issues (JSON list):
{candidates_json}

## Output JSON ONLY:
{{
  "verified_issues": [
    {{
      "original_id": <id from candidates>,
      "is_real": true/false,
      "confidence": 0.0-1.0,
      "category": "<category>",
      "severity": "critical|major|minor|info",
      "message": "<short description>",
      "line": <int|null>,
      "evidence": <string|null>
    }}
  ],
  "additional_issues": [
    {{
      "category": "<category>",
      "severity": "critical|major|minor|info",
      "message": "<short description>",
      "line": <int|null>,
      "evidence": <string|null>,
      "confidence": 0.0-1.0
    }}
  ],
  "gate_predictions": {{
    "syntax_valid": true/false,
    "imports_resolve": true/false,
    "instantiates": true/false,
    "has_structure": true/false
  }}
}}
"""


# =============================================================================
# Helpers
# =============================================================================

def _resolve_llm_config_path(config_path: Optional[str]) -> Path:
    if config_path:
        p = Path(config_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"LLM config not found: {config_path}")

    root = Path(get_project_root())
    for c in [root / "config_llm.json", root / "config_llm_example.json", Path("config_llm.json"), Path("config_llm_example.json")]:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find config_llm.json or config_llm_example.json")


def _add_line_numbers(code: str) -> str:
    lines = code.splitlines()
    return "\n".join([f"{i:4d} | {ln}" for i, ln in enumerate(lines, start=1)])


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0]
    return t.strip()


def _extract_json_object(text: str) -> str:
    t = _strip_code_fences(text)
    s = t.find("{")
    e = t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return t.strip()
    return t[s:e + 1].strip()


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

def _strip_trailing_commas(s: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", s)


def _safe_json_loads(blob: str) -> Dict[str, Any]:
    return json.loads(_strip_trailing_commas(blob))


def _repair_json_with_llm(provider: LLMProvider, bad_blob: str) -> Tuple[str, Dict[str, int]]:
    system = "You repair JSON. Output ONLY a valid JSON object. No markdown, no explanation."
    user = f"""Fix the following content into valid JSON.

Rules:
- Output only ONE JSON object.
- No trailing commas.
- Strings must be escaped.
- "line" must be integer or null (never string).
- If evidence would contain quotes/newlines/backticks or is long, set evidence to null.
- Do not add extra keys.

CONTENT TO FIX:
{bad_blob}
"""
    fixed_text, usage = provider.generate_completion(system, user)
    usage = usage or {}
    return fixed_text, {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
    }


def _is_safe_evidence(e: Any) -> bool:
    if e is None:
        return True
    if not isinstance(e, str):
        return False
    if "\n" in e or "\r" in e:
        return False
    if '"' in e or "`" in e:
        return False
    if len(e) > 180:
        return False
    return True


_LINE_INT_RE = re.compile(r"(\d+)")

def _coerce_line_to_int(line_val: Any) -> Optional[int]:
    if line_val is None:
        return None
    if isinstance(line_val, int):
        return line_val
    if isinstance(line_val, float):
        try:
            return int(line_val)
        except Exception:
            return None
    s = str(line_val).strip()
    if not s:
        return None
    m = _LINE_INT_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# =============================================================================
# Evaluator
# =============================================================================

class HybridEvaluator:
    NAME = "hybrid_propose_deterministic_judge_llm"

    def __init__(
        self,
        config_path: Optional[str] = None,
        judge_model_alias: Optional[str] = None,
        proposer_mode: str = "heuristic",      # "heuristic" (default) | "tools"
        max_code_chars: int = 12000,
        max_candidates: int = 60,
        min_confidence: float = 0.5,
        max_json_repair_attempts: int = 1,
        deterministic_config: Optional[Dict[str, Any]] = None,
    ):
        self.max_code_chars = int(max_code_chars)
        self.max_candidates = int(max_candidates)
        self.min_confidence = float(min_confidence)
        self.max_json_repair_attempts = int(max_json_repair_attempts)

        proposer_mode = (proposer_mode or "heuristic").strip().lower()
        self.proposer_mode = proposer_mode

        if proposer_mode == "tools":
            self.proposer = DeterministicEvaluator(config=deterministic_config or {})
        else:
            self.proposer = DeterministicHeuristicEvaluator(config=deterministic_config or {})

        cfg_path = _resolve_llm_config_path(config_path)
        cfg = load_config(str(cfg_path), default_config=None)
        if not cfg:
            raise ValueError(f"Failed to load LLM config: {cfg_path}")
        if not validate_api_keys(cfg):
            raise ValueError("LLM API key validation failed (set LLM_API_KEY or add to config).")

        self.provider = LLMProvider(cfg, model_key=judge_model_alias)
        self.model_info = self.provider.get_model_info()

    def evaluate(self, file_path: Path) -> EvaluatorOutput:
        start_time = time.time()
        file_path = Path(file_path)

        output = EvaluatorOutput(
            evaluator_name=f"{self.NAME}_{self.proposer_mode}_{self.model_info.get('model_key') or 'default'}",
            file_path=str(file_path),
        )
        output.metadata["llm"] = self.model_info
        output.metadata["proposer_mode"] = self.proposer_mode

        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            output.issues.append(DetectedIssue(
                category=EvalCategory.SYNTAX.value,
                severity=EvalSeverity.CRITICAL.value,
                message=f"Cannot read file: {e}",
                source="io",
            ))
            output.gate_predictions = GatePrediction(syntax_valid=False, imports_resolve=False, instantiates=False, has_structure=False)
            output.execution_time_ms = (time.time() - start_time) * 1000
            return output

        if len(code) > self.max_code_chars:
            code = code[: self.max_code_chars] + "\n# ... truncated ...\n"
            output.metadata["truncated"] = True

        numbered_code = _add_line_numbers(code)

        # Step 1: proposer output
        prop_out = self.proposer.evaluate(file_path)

        candidates = []
        for idx, iss in enumerate((prop_out.issues or [])[: self.max_candidates]):
            # sanitize evidence in candidates to reduce judge JSON risk
            cand_evidence = iss.evidence
            if not _is_safe_evidence(cand_evidence):
                cand_evidence = None

            candidates.append({
                "id": idx,
                "category": iss.category,
                "severity": iss.severity,
                "message": iss.message,
                "line": iss.line,
                "evidence": cand_evidence,
                "source": iss.source,
            })

        candidates_json = json.dumps(candidates, indent=2)

        system_prompt = "You are a strict judge. Output ONLY valid JSON."
        user_prompt = JUDGE_PROMPT.format(numbered_code=numbered_code, candidates_json=candidates_json)

        total_in = 0
        total_out = 0

        try:
            response_text, usage = self.provider.generate_completion(system_prompt, user_prompt)
            output.raw_output = response_text
            usage = usage or {}
            total_in += int(usage.get("input_tokens", 0) or 0)
            total_out += int(usage.get("output_tokens", 0) or 0)

            parsed, repair_meta, repair_usage = self._parse_json_with_repair(response_text)
            total_in += int(repair_usage.get("input_tokens", 0) or 0)
            total_out += int(repair_usage.get("output_tokens", 0) or 0)

            if repair_meta:
                output.metadata["json_repair"] = repair_meta

            # verified issues
            final_issues: List[DetectedIssue] = []

            for v in parsed.get("verified_issues", []) or []:
                if not isinstance(v, dict):
                    continue
                if not bool(v.get("is_real", False)):
                    continue

                conf = float(v.get("confidence", 0.0) or 0.0)
                if conf < self.min_confidence:
                    continue

                evidence = v.get("evidence", None)
                if not _is_safe_evidence(evidence):
                    evidence = None

                final_issues.append(DetectedIssue(
                    category=self._validate_category(v.get("category")),
                    severity=self._validate_severity(v.get("severity")),
                    message=str(v.get("message", "")).strip(),
                    line=_coerce_line_to_int(v.get("line")),
                    evidence=evidence,
                    confidence=conf,
                    source="hybrid_judge_verified",
                ))

            # additional issues
            for a in parsed.get("additional_issues", []) or []:
                if not isinstance(a, dict):
                    continue
                conf = float(a.get("confidence", 0.8) or 0.8)
                if conf < self.min_confidence:
                    continue

                evidence = a.get("evidence", None)
                if not _is_safe_evidence(evidence):
                    evidence = None

                final_issues.append(DetectedIssue(
                    category=self._validate_category(a.get("category")),
                    severity=self._validate_severity(a.get("severity")),
                    message=str(a.get("message", "")).strip(),
                    line=_coerce_line_to_int(a.get("line")),
                    evidence=evidence,
                    confidence=conf,
                    source="hybrid_judge_additional",
                ))

            output.issues = [i for i in final_issues if i.message]

            gates = parsed.get("gate_predictions", {}) or {}
            output.gate_predictions = GatePrediction(
                syntax_valid=bool(gates.get("syntax_valid", True)),
                imports_resolve=bool(gates.get("imports_resolve", True)),
                instantiates=bool(gates.get("instantiates", True)),
                has_structure=bool(gates.get("has_structure", True)),
            )

        except Exception as e:
            logger.exception(f"Hybrid judge failed: {e}; falling back to proposer output")
            output.issues = prop_out.issues
            output.gate_predictions = prop_out.gate_predictions

        output.tokens_used = total_in + total_out
        output.metadata["token_usage_total"] = {"input_tokens": total_in, "output_tokens": total_out}
        output.execution_time_ms = (time.time() - start_time) * 1000
        return output

    # ---------------- Parsing ----------------

    def _parse_json_with_repair(self, response_text: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, int]]:
        blob = _extract_json_object(response_text)
        repair_meta: Optional[Dict[str, Any]] = None
        repair_usage = {"input_tokens": 0, "output_tokens": 0}

        try:
            parsed = _safe_json_loads(blob)
            return parsed, None, repair_usage
        except json.JSONDecodeError as e1:
            if self.max_json_repair_attempts <= 0:
                raise

            repair_meta = {"used": True, "mode": "llm_repair", "first_error": str(e1), "blob_preview": blob[:800]}
            repaired_text, usage = _repair_json_with_llm(self.provider, blob)
            repair_usage = usage
            blob2 = _extract_json_object(repaired_text)
            parsed = _safe_json_loads(blob2)
            repair_meta["repaired_blob_preview"] = blob2[:800]
            return parsed, repair_meta, repair_usage

    # ---------------- Validation ----------------

    def _validate_category(self, cat: Any) -> str:
        cat_s = str(cat or "").strip().lower()
        valid = {c.value for c in EvalCategory}
        if cat_s in valid:
            return cat_s
        return EvalCategory.BEST_PRACTICE.value

    def _validate_severity(self, sev: Any) -> str:
        sev_s = str(sev or "").strip().lower()
        valid = {s.value for s in EvalSeverity}
        if sev_s in valid:
            return sev_s
        mappings = {"high": "critical", "medium": "major", "low": "minor", "warning": "major", "error": "critical"}
        return mappings.get(sev_s, EvalSeverity.MINOR.value)