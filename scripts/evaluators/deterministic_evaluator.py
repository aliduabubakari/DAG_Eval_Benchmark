"""
E1: Deterministic Evaluator (baseline)

Principles:
- Runs static tools and reformats output into the common schema.
- Predicts PCT gates using static heuristics (AST + pattern checks).
- Intentionally simple: used as the deterministic baseline.

NOTE:
This deterministic evaluator is *not* the same as your SAT/PCT paper evaluators.
It is a benchmarking baseline for "deterministic tool-based evaluation".
"""

from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger(__name__)


class DeterministicEvaluator:
    """
    Deterministic baseline evaluator.

    Config options (all optional):
      - include_pylint: bool (default True)
      - include_flake8: bool (default True)
      - include_bandit: bool (default True)
      - include_radon: bool (default False)
      - include_orchestrator_heuristics: bool (default False)
      - max_issues_per_tool: int (default 50)
    """

    NAME = "deterministic_tools_v1"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.include_pylint = bool(self.config.get("include_pylint", True))
        self.include_flake8 = bool(self.config.get("include_flake8", True))
        self.include_bandit = bool(self.config.get("include_bandit", True))
        self.include_radon = bool(self.config.get("include_radon", False))
        self.include_orchestrator_heuristics = bool(self.config.get("include_orchestrator_heuristics", False))
        self.max_issues_per_tool = int(self.config.get("max_issues_per_tool", 50))

    def evaluate(self, file_path: Path) -> EvaluatorOutput:
        start_time = time.time()
        file_path = Path(file_path)

        output = EvaluatorOutput(
            evaluator_name=self.NAME,
            file_path=str(file_path),
        )

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

        # Gate predictions (static)
        output.gate_predictions = self._predict_gates(code)

        # Tool-based issues
        tool_meta = {}
        if self.include_pylint:
            issues, meta = self._run_pylint(file_path)
            output.issues.extend(issues)
            tool_meta["pylint"] = meta

        if self.include_flake8:
            issues, meta = self._run_flake8(file_path)
            output.issues.extend(issues)
            tool_meta["flake8"] = meta

        if self.include_bandit:
            issues, meta = self._run_bandit(file_path)
            output.issues.extend(issues)
            tool_meta["bandit"] = meta

        if self.include_radon:
            issues, meta = self._run_radon(file_path)
            output.issues.extend(issues)
            tool_meta["radon"] = meta

        if self.include_orchestrator_heuristics:
            output.issues.extend(self._check_orchestrator_heuristics(code))

        output.metadata["tools"] = tool_meta
        output.execution_time_ms = (time.time() - start_time) * 1000
        return output

    # ---------------------------------------------------------------------
    # Gate prediction (static heuristics)
    # ---------------------------------------------------------------------

    def _predict_gates(self, code: str) -> GatePrediction:
        gates = GatePrediction()

        # Gate: syntax
        try:
            ast.parse(code)
            gates.syntax_valid = True
        except SyntaxError:
            gates.syntax_valid = False
            gates.imports_resolve = False
            gates.instantiates = False
            gates.has_structure = False
            return gates

        # Gate: imports
        gates.imports_resolve = self._imports_resolve_static(code)

        # Gate: instantiates / structure (pattern-based, orchestrator-aware)
        cl = code.lower()

        # Airflow
        if "airflow" in cl and ("from airflow" in cl or "import airflow" in cl):
            has_dag = ("with dag(" in cl) or ("dag(" in cl) or ("@dag" in code)
            has_tasks = bool(re.search(r"\w+Operator\s*\(", code)) or ("@task" in code)
            has_deps = (">>" in code) or ("<<" in code) or ("chain(" in cl)

            gates.instantiates = bool(has_dag)
            # "structure" means tasks exist and are wired (heuristic)
            gates.has_structure = bool(has_tasks and (has_deps or has_tasks))

        # Prefect
        elif "prefect" in cl and ("from prefect" in cl or "import prefect" in cl):
            has_flow = "@flow" in code
            has_tasks = "@task" in code or "def " in code
            gates.instantiates = bool(has_flow)
            gates.has_structure = bool(has_flow and has_tasks)

        # Dagster
        elif "dagster" in cl and ("from dagster" in cl or "import dagster" in cl):
            has_job_or_asset = ("@job" in code) or ("@graph" in code) or ("@asset" in code)
            has_ops_or_assets = ("@op" in code) or ("@asset" in code)
            gates.instantiates = bool(has_job_or_asset)
            gates.has_structure = bool(has_ops_or_assets)

        else:
            # Unknown orchestrator: keep neutral assumptions
            gates.instantiates = True
            gates.has_structure = True

        return gates

    def _imports_resolve_static(self, code: str) -> bool:
        """
        Static import resolvability:
        - parse AST
        - for each top-level import root module, check importlib.util.find_spec
        """
        try:
            tree = ast.parse(code)
        except Exception:
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = (alias.name or "").split(".")[0].strip()
                    if not root:
                        continue
                    if importlib.util.find_spec(root) is None:
                        return False

            elif isinstance(node, ast.ImportFrom):
                # skip relative imports (no module)
                if not node.module:
                    continue
                root = node.module.split(".")[0].strip()
                if not root:
                    continue
                if importlib.util.find_spec(root) is None:
                    return False

        return True

    # ---------------------------------------------------------------------
    # Tool runners
    # ---------------------------------------------------------------------

    def _tool_available(self, name: str) -> bool:
        return shutil.which(name) is not None

    def _run_pylint(self, file_path: Path) -> tuple[List[DetectedIssue], Dict[str, Any]]:
        meta: Dict[str, Any] = {"available": self._tool_available("pylint")}
        if not meta["available"]:
            return [], meta

        issues: List[DetectedIssue] = []
        try:
            res = subprocess.run(
                [
                    "pylint",
                    str(file_path),
                    "--output-format=json",
                    "--reports=n",
                    "--score=n",
                    "--max-line-length=120",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            meta["returncode"] = res.returncode
            raw = (res.stdout or "").strip()
            if not raw:
                return [], meta

            msgs = json.loads(raw)
            meta["raw_count"] = len(msgs)

            for msg in msgs[: self.max_issues_per_tool]:
                ptype = (msg.get("type") or "").lower()
                symbol = msg.get("symbol", "")

                severity = {
                    "fatal": EvalSeverity.CRITICAL,
                    "error": EvalSeverity.CRITICAL,
                    "warning": EvalSeverity.MAJOR,
                    "refactor": EvalSeverity.MINOR,
                    "convention": EvalSeverity.MINOR,
                    "information": EvalSeverity.INFO,
                }.get(ptype, EvalSeverity.MINOR)

                category = self._map_pylint_category(symbol)

                issues.append(DetectedIssue(
                    category=category.value,
                    severity=severity.value,
                    message=f"[{symbol}] {msg.get('message', '')}".strip(),
                    line=msg.get("line"),
                    evidence=None,
                    confidence=1.0,
                    source="pylint",
                ))

        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {e}"

        return issues, meta

    def _run_flake8(self, file_path: Path) -> tuple[List[DetectedIssue], Dict[str, Any]]:
        meta: Dict[str, Any] = {"available": self._tool_available("flake8")}
        if not meta["available"]:
            return [], meta

        issues: List[DetectedIssue] = []
        try:
            res = subprocess.run(
                ["flake8", str(file_path), "--max-line-length=120"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            meta["returncode"] = res.returncode

            out = (res.stdout or "").strip()
            if not out:
                return [], meta

            lines = [l for l in out.splitlines() if l.strip()]
            meta["raw_count"] = len(lines)

            for l in lines[: self.max_issues_per_tool]:
                # Format: path:line:col: code message
                parts = l.split(":", 3)
                if len(parts) < 4:
                    continue
                try:
                    line_no = int(parts[1])
                except Exception:
                    line_no = None
                msg = parts[3].strip()
                code = msg.split()[0] if msg else ""

                severity = EvalSeverity.MINOR
                if code.startswith("F"):
                    severity = EvalSeverity.CRITICAL
                elif code.startswith("E"):
                    severity = EvalSeverity.MAJOR
                elif code.startswith("W"):
                    severity = EvalSeverity.MINOR

                category = self._map_flake8_category(code)

                issues.append(DetectedIssue(
                    category=category.value,
                    severity=severity.value,
                    message=msg,
                    line=line_no,
                    source="flake8",
                ))

        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {e}"

        return issues, meta

    def _run_bandit(self, file_path: Path) -> tuple[List[DetectedIssue], Dict[str, Any]]:
        meta: Dict[str, Any] = {"available": self._tool_available("bandit")}
        if not meta["available"]:
            return [], meta

        issues: List[DetectedIssue] = []
        try:
            res = subprocess.run(
                ["bandit", "-f", "json", "-q", str(file_path)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            meta["returncode"] = res.returncode
            raw = (res.stdout or "").strip()
            if not raw:
                return [], meta

            data = json.loads(raw)
            results = data.get("results", []) or []
            meta["raw_count"] = len(results)

            for r in results[: self.max_issues_per_tool]:
                sev = (r.get("issue_severity") or "").upper()
                severity = {
                    "HIGH": EvalSeverity.CRITICAL,
                    "MEDIUM": EvalSeverity.MAJOR,
                    "LOW": EvalSeverity.MINOR,
                }.get(sev, EvalSeverity.MINOR)

                issues.append(DetectedIssue(
                    category=EvalCategory.SECURITY.value,
                    severity=severity.value,
                    message=r.get("issue_text", ""),
                    line=r.get("line_number"),
                    source="bandit",
                ))

        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {e}"

        return issues, meta

    def _run_radon(self, file_path: Path) -> tuple[List[DetectedIssue], Dict[str, Any]]:
        meta: Dict[str, Any] = {"available": self._tool_available("radon")}
        if not meta["available"]:
            return [], meta

        issues: List[DetectedIssue] = []
        try:
            res = subprocess.run(
                ["radon", "cc", str(file_path), "-j"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            meta["returncode"] = res.returncode

            raw = (res.stdout or "").strip()
            if not raw:
                return [], meta

            data = json.loads(raw)
            items = []
            for _path, entries in (data or {}).items():
                items.extend(entries or [])

            meta["raw_count"] = len(items)

            # Conservative: report only high CC
            for it in items[: self.max_issues_per_tool]:
                cc = int(it.get("complexity", 0) or 0)
                if cc <= 10:
                    continue
                sev = EvalSeverity.MAJOR if cc >= 15 else EvalSeverity.MINOR
                issues.append(DetectedIssue(
                    category=EvalCategory.COMPLEXITY.value,
                    severity=sev.value,
                    message=f"High cyclomatic complexity: {cc} in {it.get('name', '?')}",
                    line=it.get("lineno"),
                    source="radon",
                ))

        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {e}"

        return issues, meta

    # ---------------------------------------------------------------------
    # Mapping helpers
    # ---------------------------------------------------------------------

    def _map_pylint_category(self, symbol: str) -> EvalCategory:
        mappings = {
            "syntax-error": EvalCategory.SYNTAX,
            "import-error": EvalCategory.IMPORT,
            "no-name-in-module": EvalCategory.IMPORT,
            "undefined-variable": EvalCategory.UNDEFINED,
            "unused-import": EvalCategory.UNUSED,
            "unused-variable": EvalCategory.UNUSED,
            "unused-argument": EvalCategory.UNUSED,
            "missing-docstring": EvalCategory.DOCUMENTATION,
            "missing-module-docstring": EvalCategory.DOCUMENTATION,
            "missing-function-docstring": EvalCategory.DOCUMENTATION,
            "invalid-name": EvalCategory.NAMING,
            "line-too-long": EvalCategory.STYLE,
            "bare-except": EvalCategory.ERROR_HANDLING,
            "broad-except": EvalCategory.ERROR_HANDLING,
            "too-many-branches": EvalCategory.COMPLEXITY,
            "too-many-statements": EvalCategory.COMPLEXITY,
        }
        return mappings.get(symbol, EvalCategory.BEST_PRACTICE)

    def _map_flake8_category(self, code: str) -> EvalCategory:
        # A small, pragmatic mapping
        if code.startswith("F401"):
            return EvalCategory.UNUSED
        if code.startswith("F821"):
            return EvalCategory.UNDEFINED
        if code.startswith("E") or code.startswith("W") or code.startswith("F"):
            return EvalCategory.STYLE
        return EvalCategory.STYLE

    # ---------------------------------------------------------------------
    # Optional heuristics (off by default)
    # ---------------------------------------------------------------------

    def _check_orchestrator_heuristics(self, code: str) -> List[DetectedIssue]:
        """
        Optional heuristic checks for orchestrator best practices.
        NOTE: Not part of tool-ensemble SAT ground truth by default.
        """
        issues: List[DetectedIssue] = []
        cl = code.lower()

        if "airflow" in cl and ("from airflow" in cl or "import airflow" in cl):
            if "retries=" not in code:
                issues.append(DetectedIssue(
                    category=EvalCategory.BEST_PRACTICE.value,
                    severity=EvalSeverity.MINOR.value,
                    message="Airflow: no retries= configured",
                    source="heuristic",
                ))
            if "execution_timeout" not in code and "dagrun_timeout" not in code:
                issues.append(DetectedIssue(
                    category=EvalCategory.BEST_PRACTICE.value,
                    severity=EvalSeverity.MINOR.value,
                    message="Airflow: no timeout configured (execution_timeout/dagrun_timeout)",
                    source="heuristic",
                ))

        return issues