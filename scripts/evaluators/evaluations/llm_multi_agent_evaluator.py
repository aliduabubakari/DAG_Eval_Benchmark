#!/usr/bin/env python3
"""
E3: Multi-Agent LLM Evaluator (DeepInfra via utils/llm_provider.py)

Hardened + FIXED:
- Avoids Python .format() on prompt strings containing JSON braces (prevents crashes)
- Line-numbered code; require integer line numbers
- Robust JSON parsing + single repair call (token accounting includes repair call)
- Strict evidence=null enforcement if unsafe
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from utils.config_loader import load_config, validate_api_keys, get_project_root
from utils.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt templates (NO .format() used; we use .replace placeholders)
# =============================================================================

COMMON_LINE_RULES = """## IMPORTANT: Line Numbers
The code is shown WITH LINE NUMBERS in the left margin as:
"  12 | actual code ..."
If you report an issue and can identify where it occurs:
- set "line" to the integer line number from the margin (e.g., 12)
- do NOT guess line numbers
- if unknown, set "line": null
- line must be an integer or null (never a string)

## IMPORTANT: Evidence Field Safety
Set "evidence": null unless you can provide a SINGLE-LINE snippet
that contains NO newline characters and NO double quotes (") and NO backticks.
If unsure, evidence must be null.
"""

# Placeholders used: __CODE__ and __RULES__
SECURITY_AGENT_PROMPT = """You are a SECURITY-focused code reviewer.

Focus ONLY on security issues:
- Hardcoded secrets, passwords, API keys
- SQL injection risks
- Command injection risks
- Insecure file operations
- Use of eval/exec
- Insecure deserialization
- subprocess shell=True risks

__RULES__

Code (with line numbers):
```python
__CODE__
```

Output JSON only:
{
  "issues": [
    {"category": "security", "severity": "critical|major|minor|info", "message": "...", "line": null, "evidence": null}
  ]
}
"""

QUALITY_AGENT_PROMPT = """You are a CODE QUALITY reviewer for Python.

Focus on:
- Unused imports/variables
- Missing docstrings
- Naming violations
- High complexity / deeply nested code
- Bare except clauses
- Obvious runtime hazards

__RULES__

Code (with line numbers):
```python
__CODE__
```

Output JSON only:
{
  "issues": [
    {"category": "unused|documentation|naming|complexity|error_handling|style|best_practice|undefined|type_error",
     "severity": "critical|major|minor|info",
     "message": "...",
     "line": null,
     "evidence": null}
  ]
}
"""

ORCHESTRATOR_AGENT_PROMPT = """You are an ORCHESTRATOR expert (Airflow / Prefect / Dagster).

Report orchestrator-specific problems and predict gates:
- Missing DAG/Flow/Job definition
- Missing tasks
- Missing dependencies
- Obvious invalid config patterns

__RULES__

Code (with line numbers):
```python
__CODE__
```

Output JSON only:
{
  "issues": [
    {"category": "orchestrator_structure|orchestrator_config|best_practice",
     "severity": "critical|major|minor|info",
     "message": "...",
     "line": null,
     "evidence": null}
  ],
  "gate_predictions": {
    "syntax_valid": true,
    "imports_resolve": true,
    "instantiates": true,
    "has_structure": true
  }
}
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
    for c in [root / "config_llm.json", root / "config_llm_example.json", Path("config_llm.json"), Path("config_llm_example.json")]:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find config_llm.json or config_llm_example.json")


def _add_line_numbers(code: str) -> str:
    return "\n".join([f"{i:4d} | {ln}" for i, ln in enumerate(code.splitlines(), start=1)])


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


@dataclass
class AgentConfig:
    name: str
    prompt_template: str


# =============================================================================
# Evaluator
# =============================================================================

class LLMMultiAgentEvaluator:
    NAME = "llm_multi_agent"

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_alias: Optional[str] = None,
        max_code_chars: int = 12000,
        parallel: bool = True,
        max_json_repair_attempts: int = 1,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.max_code_chars = int(max_code_chars)
        self.parallel = bool(parallel)
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

        self.agents = [
            AgentConfig("security", SECURITY_AGENT_PROMPT),
            AgentConfig("quality", QUALITY_AGENT_PROMPT),
            AgentConfig("orchestrator", ORCHESTRATOR_AGENT_PROMPT),
        ]

    def evaluate(self, file_path: Path) -> EvaluatorOutput:
        start_time = time.time()
        file_path = Path(file_path)

        output = EvaluatorOutput(
            evaluator_name=f"{self.NAME}_{self.model_info.get('model_key') or 'default'}",
            file_path=str(file_path),
        )
        output.metadata["llm"] = self.model_info
        output.metadata["parallel"] = self.parallel

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

        all_issue_dicts: List[Dict[str, Any]] = []
        gate_predictions: Optional[Dict[str, Any]] = None

        token_total_in = 0
        token_total_out = 0
        per_agent_usage: Dict[str, Dict[str, int]] = {}
        per_agent_repair: Dict[str, Any] = {}

        def build_prompt(template: str) -> str:
            # IMPORTANT: no .format() here
            return template.replace("__RULES__", COMMON_LINE_RULES).replace("__CODE__", numbered_code)

        def run_agent(agent: AgentConfig) -> Tuple[str, Dict[str, Any], Dict[str, int], Optional[Dict[str, Any]]]:
            user_prompt = build_prompt(agent.prompt_template)
            system_prompt = "You are a specialist agent. Output ONLY valid JSON."

            text, usage0 = self.provider.generate_completion(system_prompt, user_prompt)
            usage0 = usage0 or {}

            parsed, repair_meta, repair_usage = self._parse_json_with_repair(text)

            # Attach source to each issue
            for iss in (parsed.get("issues") or []):
                if isinstance(iss, dict) and "source" not in iss:
                    iss["source"] = f"llm_agent_{agent.name}"

            total_usage = {
                "input_tokens": int(usage0.get("input_tokens", 0) or 0) + int(repair_usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage0.get("output_tokens", 0) or 0) + int(repair_usage.get("output_tokens", 0) or 0),
            }
            return agent.name, parsed, total_usage, repair_meta

        try:
            if self.parallel:
                with ThreadPoolExecutor(max_workers=len(self.agents)) as ex:
                    futures = [ex.submit(run_agent, a) for a in self.agents]
                    for fut in as_completed(futures):
                        name, parsed, usage, repair_meta = fut.result()

                        per_agent_usage[name] = usage
                        if repair_meta:
                            per_agent_repair[name] = repair_meta

                        token_total_in += int(usage.get("input_tokens", 0) or 0)
                        token_total_out += int(usage.get("output_tokens", 0) or 0)

                        all_issue_dicts.extend((parsed.get("issues") or []))
                        if name == "orchestrator" and isinstance(parsed.get("gate_predictions"), dict):
                            gate_predictions = parsed["gate_predictions"]
            else:
                for a in self.agents:
                    name, parsed, usage, repair_meta = run_agent(a)

                    per_agent_usage[name] = usage
                    if repair_meta:
                        per_agent_repair[name] = repair_meta

                    token_total_in += int(usage.get("input_tokens", 0) or 0)
                    token_total_out += int(usage.get("output_tokens", 0) or 0)

                    all_issue_dicts.extend((parsed.get("issues") or []))
                    if name == "orchestrator" and isinstance(parsed.get("gate_predictions"), dict):
                        gate_predictions = parsed["gate_predictions"]

        except Exception as e:
            # If anything goes wrong, return a meaningful EvaluatorOutput (no silent zeros)
            output.issues.append(DetectedIssue(
                category=EvalCategory.ERROR_HANDLING.value,
                severity=EvalSeverity.INFO.value,
                message=f"llm_multi_agent failed: {type(e).__name__}: {e}",
                source="llm_multi_agent_error",
            ))
            output.gate_predictions = GatePrediction()
            output.execution_time_ms = (time.time() - start_time) * 1000
            return output

        output.tokens_used = token_total_in + token_total_out
        output.metadata["token_usage_by_agent"] = per_agent_usage
        output.metadata["token_usage_total"] = {"input_tokens": token_total_in, "output_tokens": token_total_out}
        if per_agent_repair:
            output.metadata["json_repair_by_agent"] = per_agent_repair

        output.issues = self._deduplicate_and_sanitize_issues(all_issue_dicts)

        if gate_predictions:
            output.gate_predictions = GatePrediction(
                syntax_valid=bool(gate_predictions.get("syntax_valid", True)),
                imports_resolve=bool(gate_predictions.get("imports_resolve", True)),
                instantiates=bool(gate_predictions.get("instantiates", True)),
                has_structure=bool(gate_predictions.get("has_structure", True)),
            )
        else:
            output.gate_predictions = GatePrediction()

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

    # ---------------- Issue normalization ----------------

    def _deduplicate_and_sanitize_issues(self, issues: List[Dict[str, Any]]) -> List[DetectedIssue]:
        seen = set()
        out: List[DetectedIssue] = []

        for issue in issues:
            if not isinstance(issue, dict):
                continue

            cat = self._validate_category(issue.get("category"))
            sev = self._validate_severity(issue.get("severity"))
            msg = str(issue.get("message", "")).strip()
            line = _coerce_line_to_int(issue.get("line"))
            evidence = issue.get("evidence", None)
            if not _is_safe_evidence(evidence):
                evidence = None
            src = issue.get("source")

            fp = f"{cat}:{line}:{msg[:60]}"
            if fp in seen:
                continue
            seen.add(fp)

            if not msg:
                continue

            out.append(DetectedIssue(
                category=cat,
                severity=sev,
                message=msg,
                line=line,
                evidence=evidence,
                confidence=float(issue.get("confidence", 1.0) or 1.0),
                source=src,
            ))

        return out

    def _validate_category(self, cat: Any) -> str:
        cat_s = str(cat or "").strip().lower()
        valid = {c.value for c in EvalCategory}
        if cat_s in valid:
            return cat_s
        mappings = {
            "security_vulnerability": EvalCategory.SECURITY.value,
            "code_style": EvalCategory.STYLE.value,
            "missing_docstring": EvalCategory.DOCUMENTATION.value,
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