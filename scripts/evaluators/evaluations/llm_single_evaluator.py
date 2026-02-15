#!/usr/bin/env python3
"""
E2: Single LLM Evaluator (DeepInfra via utils/llm_provider.py)

Hardened:
- Line-numbered prompts; require integer 'line' extracted from margin
- Robust JSON parsing (strip trailing commas + single repair call)
- Token accounting includes repair call
- Strict evidence sanitization (evidence must be null unless single-line & quote-safe)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# --- bootstrap so this module works as script or import ---
import sys
_THIS = Path(__file__).resolve()
SCRIPTS_DIR = _THIS
while SCRIPTS_DIR.name != "scripts" and SCRIPTS_DIR != SCRIPTS_DIR.parent:
    SCRIPTS_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# --------------------------------------------------------

from evaluators.output_schema import (
    EvaluatorOutput,
    DetectedIssue,
    GatePrediction,
    EvalCategory,
    EvalSeverity,
)
from utils.config_loader import load_config, validate_api_keys, get_project_root
from utils.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt
# =============================================================================

EVALUATION_PROMPT = """You are a code quality evaluator for workflow orchestration code (Airflow, Prefect, Dagster).

Analyze the following code and identify issues. Be thorough but precise.

## Issue Categories (use exactly these):
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

## Severity Levels (use exactly these):
- critical
- major
- minor
- info

## IMPORTANT: Line Numbers
The code is shown WITH LINE NUMBERS in the left margin as:
"  12 | actual code ..."
If you report an issue and can identify where it occurs, set "line" to the integer line number from the margin (e.g., 12).
- Do NOT guess line numbers.
- If you cannot identify a line reliably, set "line": null.
- "line" must be an integer or null (never a string).

## IMPORTANT: Evidence Field Safety
To avoid invalid JSON:
- Set "evidence": null unless you can provide a SINGLE-LINE snippet
- The evidence snippet MUST NOT contain newline characters
- The evidence snippet MUST NOT contain double quotes (") or backticks
If unsure, set evidence to null.

## Gate Predictions:
Predict whether this code will:
1. syntax_valid: Parse as valid Python
2. imports_resolve: All imports can be resolved in a typical evaluation environment
3. instantiates: DAG/Flow/Job object will be created/discoverable
4. has_structure: Tasks are defined and connected (or otherwise structurally meaningful)

## Code to Analyze (with line numbers):
```python
{numbered_code}
```

## Output Format (JSON only, no other text):
{{
  "issues": [
    {{
      "category": "<category>",
      "severity": "<severity>",
      "message": "<description>",
      "line": <integer_or_null>,
      "evidence": <single_line_string_or_null>
    }}
  ],
  "gate_predictions": {{
    "syntax_valid": true/false,
    "imports_resolve": true/false,
    "instantiates": true/false,
    "has_structure": true/false
  }}
}}

Be conservative: only report issues you are confident about.
"""


# =============================================================================
# JSON robustness helpers
# =============================================================================

def _resolve_llm_config_path(config_path: Optional[str]) -> Path:
    if config_path:
        p = Path(config_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"LLM config not found: {config_path}")

    root = Path(get_project_root())
    candidates = [
        root / "config_llm.json",
        root / "config_llm_example.json",
        Path("config_llm.json"),
        Path("config_llm_example.json"),
    ]
    for c in candidates:
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
    """
    Robust extraction:
    - remove code fences
    - take substring from first '{' to last '}'
    """
    t = _strip_code_fences(text)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return t.strip()
    return t[start:end + 1].strip()


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

def _strip_trailing_commas(s: str) -> str:
    # Remove trailing commas before } or ]
    return _TRAILING_COMMA_RE.sub(r"\1", s)


def _safe_json_loads(blob: str) -> Dict[str, Any]:
    return json.loads(_strip_trailing_commas(blob))


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
    """
    Accept:
      - int
      - "12"
      - "Line 12"
      - "12 | something"
    Return int or None.
    """
    if line_val is None:
        return None
    if isinstance(line_val, int):
        return line_val
    if isinstance(line_val, float):
        # Sometimes models return 12.0
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


def _repair_json_with_llm(provider: LLMProvider, bad_blob: str) -> Tuple[str, Dict[str, int]]:
    """
    One repair attempt: ask model to output ONLY valid JSON.
    Returns (fixed_text, token_usage).
    """
    system = "You repair JSON. Output ONLY a valid JSON object. No markdown, no explanation."
    user = f"""Fix the following content into valid JSON matching this schema:

{{
  "issues": [
    {{
      "category": "<category>",
      "severity": "<severity>",
      "message": "<description>",
      "line": <integer_or_null>,
      "evidence": <single_line_string_or_null>
    }}
  ],
  "gate_predictions": {{
    "syntax_valid": true/false,
    "imports_resolve": true/false,
    "instantiates": true/false,
    "has_structure": true/false
  }}
}}

Rules:
- Output only ONE JSON object.
- No trailing commas.
- Strings must be properly escaped.
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


# =============================================================================
# Evaluator
# =============================================================================

class LLMSingleEvaluator:
    NAME = "llm_single"

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_alias: Optional[str] = None,      # e.g., "Qwen3-Coder"
        max_code_chars: int = 12000,
        max_json_repair_attempts: int = 1,      # requested: stick to 1
        config: Optional[Dict[str, Any]] = None,
    ):
        self.max_code_chars = int(max_code_chars)
        self.max_json_repair_attempts = int(max_json_repair_attempts)
        self.config = config or {}

        cfg_path = _resolve_llm_config_path(config_path)
        cfg = load_config(str(cfg_path), default_config=None)
        if not cfg:
            raise ValueError(f"Failed to load LLM config: {cfg_path}")

        if not validate_api_keys(cfg):
            raise ValueError("LLM API key validation failed (set LLM_API_KEY or add to config).")

        self.provider = LLMProvider(cfg, model_key=model_alias)
        self.model_info = self.provider.get_model_info()

    def evaluate(self, file_path: Path) -> EvaluatorOutput:
        start_time = time.time()
        file_path = Path(file_path)

        output = EvaluatorOutput(
            evaluator_name=f"{self.NAME}_{self.model_info.get('model_key') or 'default'}",
            file_path=str(file_path),
        )
        output.metadata["llm"] = self.model_info

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

        system_prompt = "You are a precise evaluator. Output ONLY valid JSON."
        user_prompt = EVALUATION_PROMPT.format(numbered_code=numbered_code)

        total_input_tokens = 0
        total_output_tokens = 0

        try:
            response_text, token_usage = self.provider.generate_completion(system_prompt, user_prompt)
            output.raw_output = response_text

            token_usage = token_usage or {}
            total_input_tokens += int(token_usage.get("input_tokens", 0) or 0)
            total_output_tokens += int(token_usage.get("output_tokens", 0) or 0)

            parsed, repair_meta, repair_tokens = self._parse_llm_json_with_repair(response_text)

            total_input_tokens += int(repair_tokens.get("input_tokens", 0) or 0)
            total_output_tokens += int(repair_tokens.get("output_tokens", 0) or 0)
            if repair_meta:
                output.metadata["json_repair"] = repair_meta

            # Postprocess issues
            issues_out: List[DetectedIssue] = []
            for issue_data in parsed.get("issues", []) or []:
                if not isinstance(issue_data, dict):
                    continue

                cat = self._validate_category(issue_data.get("category"))
                sev = self._validate_severity(issue_data.get("severity"))
                msg = str(issue_data.get("message", "")).strip()

                line = _coerce_line_to_int(issue_data.get("line"))
                evidence = issue_data.get("evidence", None)
                if not _is_safe_evidence(evidence):
                    evidence = None

                if msg:
                    issues_out.append(DetectedIssue(
                        category=cat,
                        severity=sev,
                        message=msg,
                        line=line,
                        evidence=evidence,
                        confidence=float(issue_data.get("confidence", 1.0) or 1.0),
                        source="llm_single",
                    ))

            output.issues = issues_out

            gates = parsed.get("gate_predictions", {}) or {}
            output.gate_predictions = GatePrediction(
                syntax_valid=bool(gates.get("syntax_valid", True)),
                imports_resolve=bool(gates.get("imports_resolve", True)),
                instantiates=bool(gates.get("instantiates", True)),
                has_structure=bool(gates.get("has_structure", True)),
            )

        except Exception as e:
            logger.exception(f"LLM evaluation failed: {e}")
            output.issues.append(DetectedIssue(
                category=EvalCategory.ERROR_HANDLING.value,
                severity=EvalSeverity.INFO.value,
                message=f"LLM evaluation error: {type(e).__name__}: {e}",
                source="llm_single_error",
            ))
            output.gate_predictions = GatePrediction()

        output.tokens_used = total_input_tokens + total_output_tokens
        output.metadata["token_usage_total"] = {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens}
        output.execution_time_ms = (time.time() - start_time) * 1000
        return output

    def _parse_llm_json_with_repair(self, response_text: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, int]]:
        """
        Returns: (parsed, repair_meta_or_none, repair_token_usage)
        """
        blob = _extract_json_object(response_text)

        # Save a preview for debugging; avoid storing megabytes.
        repair_meta: Optional[Dict[str, Any]] = None
        repair_tokens: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        try:
            parsed = _safe_json_loads(blob)
            return parsed, None, repair_tokens
        except json.JSONDecodeError as e1:
            if self.max_json_repair_attempts <= 0:
                raise

            # One repair attempt using LLM
            repair_meta = {
                "used": True,
                "mode": "llm_repair",
                "first_error": str(e1),
                "blob_preview": blob[:1000],
            }
            repaired_text, usage = _repair_json_with_llm(self.provider, blob)
            repair_tokens = usage

            blob2 = _extract_json_object(repaired_text)
            parsed = _safe_json_loads(blob2)
            repair_meta["repaired_blob_preview"] = blob2[:1000]
            return parsed, repair_meta, repair_tokens

    def _validate_category(self, cat: Any) -> str:
        cat_s = str(cat or "").strip().lower()
        valid = {c.value for c in EvalCategory}
        if cat_s in valid:
            return cat_s

        mappings = {
            "security_vulnerability": EvalCategory.SECURITY.value,
            "code_style": EvalCategory.STYLE.value,
            "missing_docstring": EvalCategory.DOCUMENTATION.value,
            "orchestrator": EvalCategory.ORCHESTRATOR_STRUCTURE.value,
        }
        return mappings.get(cat_s, EvalCategory.BEST_PRACTICE.value)

    def _validate_severity(self, sev: Any) -> str:
        sev_s = str(sev or "").strip().lower()
        valid = {s.value for s in EvalSeverity}
        if sev_s in valid:
            return sev_s

        mappings = {
            "high": EvalSeverity.CRITICAL.value,
            "medium": EvalSeverity.MAJOR.value,
            "low": EvalSeverity.MINOR.value,
            "warning": EvalSeverity.MAJOR.value,
            "error": EvalSeverity.CRITICAL.value,
        }
        return mappings.get(sev_s, EvalSeverity.MINOR.value)