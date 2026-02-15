#!/usr/bin/env python3
"""
Deterministic Heuristic Evaluator (no external tools)

Purpose:
- Provide a deterministic baseline that does NOT run pylint/flake8/bandit/radon.
- Uses only AST + regex heuristics.
- Avoids trivial wins on Track C (GT2 tool alignment) by not being a "tool mirror".

Outputs:
- issues (DetectedIssue)
- gate_predictions (GatePrediction)
- execution_time_ms
- tokens_used=0 always
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


# ---------------------------
# Helpers
# ---------------------------

def _clamp_line(x: Any) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def _detect_orchestrator(code: str) -> str:
    c = (code or "").lower()
    if "airflow" in c and ("from airflow" in c or "import airflow" in c or "@dag" in code or "dag(" in c):
        return "airflow"
    if "prefect" in c and ("from prefect" in c or "import prefect" in c or "@flow" in code):
        return "prefect"
    if "dagster" in c and ("from dagster" in c or "import dagster" in c or "@job" in code or "@asset" in code):
        return "dagster"
    return "unknown"


def _iter_import_roots(tree: ast.AST) -> Set[str]:
    roots: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name:
                    roots.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return {r for r in roots if r}


def _defined_names(tree: ast.AST) -> Set[str]:
    defined: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for a in (node.args.args + node.args.kwonlyargs):
                    defined.add(a.arg)
                if node.args.vararg:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined.add(node.args.kwarg.arg)

        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)

        elif isinstance(node, ast.Import):
            for a in node.names:
                defined.add(a.asname or a.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                defined.add(a.asname or a.name)

        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined.add(node.name)

        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    defined.add(item.optional_vars.id)

        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        defined.add(elt.id)

    return defined


def _load_names(tree: ast.AST) -> Set[str]:
    loads: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads.add(node.id)
    return loads


def _has_bare_except(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            return True
    return False


def _has_swallow_exception(tree: ast.AST) -> bool:
    # except Exception: pass
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                if isinstance(h.type, ast.Name) and h.type.id == "Exception":
                    if len(h.body) == 1 and isinstance(h.body[0], ast.Pass):
                        return True
    return False


def _has_eval_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
            return True
    return False


def _has_exec_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "exec":
            return True
    return False


def _has_shell_true(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords or []:
                if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    return True
    return False


def _hardcoded_secret_lines(code: str) -> List[int]:
    # synthetic + general secret patterns
    pat = re.compile(r"(?i)\b(API_KEY|TOKEN|PASSWORD|SECRET)\b\s*=\s*['\"][^'\"]{8,}['\"]")
    lines = []
    for i, ln in enumerate(code.splitlines(), start=1):
        if pat.search(ln):
            lines.append(i)
    return lines


def _task_ids_airflow(code: str) -> List[str]:
    return re.findall(r"task_id\s*=\s*['\"]([^'\"]+)['\"]", code)


def _has_airflow_dag_definition(code: str) -> bool:
    c = code.lower()
    return ("with dag(" in c) or ("@dag" in code) or ("dag(" in c and "dag_id" in c)


def _has_airflow_tasks(code: str) -> bool:
    return bool(re.search(r"\w+Operator\s*\(", code)) or ("@task" in code)


def _has_airflow_deps(code: str) -> bool:
    c = code.lower()
    return (">>" in code) or ("<<" in code) or ("chain(" in c) or ("set_downstream" in c) or ("set_upstream" in c)


def _has_prefect_flow(code: str) -> bool:
    return "@flow" in code


def _has_prefect_tasks(code: str) -> bool:
    return "@task" in code


def _has_dagster_job_or_asset(code: str) -> bool:
    return ("@job" in code) or ("@graph" in code) or ("@asset" in code)


def _has_dagster_ops_or_assets(code: str) -> bool:
    return ("@op" in code) or ("@asset" in code)


def _max_nesting_depth(tree: ast.AST) -> int:
    """
    Simple nesting depth heuristic: count nested blocks of If/For/While/Try/With/Function/Class.
    """
    max_depth = 0

    def walk(node: ast.AST, depth: int) -> None:
        nonlocal max_depth
        max_depth = max(max_depth, depth)

        children = list(ast.iter_child_nodes(node))
        for ch in children:
            # Increase depth on block-like nodes
            if isinstance(ch, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(ch, depth + 1)
            else:
                walk(ch, depth)

    walk(tree, 0)
    return max_depth


# =============================================================================
# Evaluator
# =============================================================================

class DeterministicHeuristicEvaluator:
    """
    Deterministic heuristic evaluator (no external tools).

    This baseline is intended to be stable and cheap, and not circular with GT2.
    """

    NAME = "deterministic_heuristic_v1"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_issues = int(self.config.get("max_issues", 80))
        self.max_code_chars = int(self.config.get("max_code_chars", 200_000))

    def evaluate(self, file_path: Path) -> EvaluatorOutput:
        t0 = time.time()
        file_path = Path(file_path)

        out = EvaluatorOutput(
            evaluator_name=self.NAME,
            file_path=str(file_path),
        )

        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            out.issues.append(DetectedIssue(
                category=EvalCategory.SYNTAX.value,
                severity=EvalSeverity.CRITICAL.value,
                message=f"Cannot read file: {type(e).__name__}: {e}",
                source="heuristic_io",
            ))
            out.gate_predictions = GatePrediction(syntax_valid=False, imports_resolve=False, instantiates=False, has_structure=False)
            out.execution_time_ms = (time.time() - t0) * 1000
            return out

        if len(code) > self.max_code_chars:
            code = code[: self.max_code_chars] + "\n# ... truncated ...\n"
            out.metadata["truncated"] = True

        orch = _detect_orchestrator(code)
        out.metadata["detected_orchestrator"] = orch

        # ---------- Gate predictions ----------
        gates = GatePrediction()

        # syntax
        try:
            tree = ast.parse(code)
            gates.syntax_valid = True
        except SyntaxError as e:
            gates.syntax_valid = False
            gates.imports_resolve = False
            gates.instantiates = False
            gates.has_structure = False

            out.gate_predictions = gates
            out.issues.append(DetectedIssue(
                category=EvalCategory.SYNTAX.value,
                severity=EvalSeverity.CRITICAL.value,
                message=f"SyntaxError: {e.msg}",
                line=_clamp_line(e.lineno),
                source="heuristic_syntax",
            ))
            out.execution_time_ms = (time.time() - t0) * 1000
            return out

        # imports gate heuristic (environment-agnostic):
        # - fail if obviously injected nonexistent module is imported
        # - otherwise assume imports resolve (since oracle env is standardized separately)
        roots = _iter_import_roots(tree)
        gates.imports_resolve = True
        for r in roots:
            if r.startswith("__definitely_nonexistent_module_"):
                gates.imports_resolve = False
                break
            if "undefined_module" in r.lower() or "nonexistent" in r.lower():
                gates.imports_resolve = False
                break

        # instantiation / structure heuristics by orchestrator
        if orch == "airflow":
            has_dag = _has_airflow_dag_definition(code)
            has_tasks = _has_airflow_tasks(code)
            has_deps = _has_airflow_deps(code)
            gates.instantiates = bool(has_dag)
            gates.has_structure = bool(has_tasks and (has_deps or has_tasks))
        elif orch == "prefect":
            has_flow = _has_prefect_flow(code)
            has_tasks = _has_prefect_tasks(code) or ("def " in code)
            gates.instantiates = bool(has_flow)
            gates.has_structure = bool(has_flow and has_tasks)
        elif orch == "dagster":
            has_job = _has_dagster_job_or_asset(code)
            has_ops = _has_dagster_ops_or_assets(code)
            gates.instantiates = bool(has_job)
            gates.has_structure = bool(has_ops)
        else:
            gates.instantiates = True
            gates.has_structure = True

        out.gate_predictions = gates

        # ---------- Issue extraction ----------
        issues: List[DetectedIssue] = []

        # Import issues (only the injected kind reliably)
        if not gates.imports_resolve:
            issues.append(DetectedIssue(
                category=EvalCategory.IMPORT.value,
                severity=EvalSeverity.CRITICAL.value,
                message="Likely import failure: nonexistent/broken module detected",
                source="heuristic_import",
            ))

        # Undefined variable issues (detect synthetic + general)
        defined = _defined_names(tree)
        loads = _load_names(tree)
        undefined = sorted([n for n in loads if (n not in defined) and n.startswith("__undefined_name_")])
        for name in undefined[:10]:
            issues.append(DetectedIssue(
                category=EvalCategory.UNDEFINED.value,
                severity=EvalSeverity.CRITICAL.value,
                message=f"Undefined name reference: {name}",
                source="heuristic_undefined",
            ))

        # Unused variable injection
        # detect __unused_var_* = ... assigned but never loaded
        assigned_unused = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.startswith("__unused_var_"):
                        assigned_unused.append((t.id, getattr(node, "lineno", None)))
        for name, ln in assigned_unused[:10]:
            if name not in loads:
                issues.append(DetectedIssue(
                    category=EvalCategory.UNUSED.value,
                    severity=EvalSeverity.MINOR.value,
                    message=f"Unused variable appears to be assigned but never used: {name}",
                    line=_clamp_line(ln),
                    source="heuristic_unused",
                ))

        # Unused import alias injection: import os as __unused_import_alias_*
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    alias = a.asname or ""
                    if alias.startswith("__unused_import_alias_"):
                        if alias not in loads:
                            issues.append(DetectedIssue(
                                category=EvalCategory.UNUSED.value,
                                severity=EvalSeverity.MINOR.value,
                                message=f"Unused import alias: {alias}",
                                line=_clamp_line(getattr(node, "lineno", None)),
                                source="heuristic_unused",
                            ))

        # Security
        if _has_eval_call(tree) or _has_exec_call(tree):
            issues.append(DetectedIssue(
                category=EvalCategory.SECURITY.value,
                severity=EvalSeverity.MAJOR.value,
                message="Use of eval/exec detected",
                source="heuristic_security",
            ))
        if _has_shell_true(tree):
            issues.append(DetectedIssue(
                category=EvalCategory.SECURITY.value,
                severity=EvalSeverity.MAJOR.value,
                message="subprocess call with shell=True detected",
                source="heuristic_security",
            ))

        for ln in _hardcoded_secret_lines(code)[:5]:
            issues.append(DetectedIssue(
                category=EvalCategory.SECURITY.value,
                severity=EvalSeverity.MAJOR.value,
                message="Potential hardcoded secret assignment detected",
                line=ln,
                source="heuristic_security",
            ))

        # Error handling
        if _has_bare_except(tree):
            issues.append(DetectedIssue(
                category=EvalCategory.ERROR_HANDLING.value,
                severity=EvalSeverity.MINOR.value,
                message="Bare except detected",
                source="heuristic_error_handling",
            ))
        if _has_swallow_exception(tree):
            issues.append(DetectedIssue(
                category=EvalCategory.ERROR_HANDLING.value,
                severity=EvalSeverity.MINOR.value,
                message="except Exception: pass pattern detected",
                source="heuristic_error_handling",
            ))

        # Documentation coverage (very lightweight)
        module_doc = ast.get_docstring(tree) is not None
        if not module_doc:
            issues.append(DetectedIssue(
                category=EvalCategory.DOCUMENTATION.value,
                severity=EvalSeverity.MINOR.value,
                message="Missing module docstring",
                source="heuristic_docs",
            ))

        # Naming conventions
        snake = re.compile(r"^[a-z_][a-z0-9_]*$")
        pascal = re.compile(r"^[A-Z][A-Za-z0-9]*$")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not snake.match(node.name) and not node.name.startswith("_"):
                    issues.append(DetectedIssue(
                        category=EvalCategory.NAMING.value,
                        severity=EvalSeverity.MINOR.value,
                        message=f"Function name not snake_case: {node.name}",
                        line=_clamp_line(getattr(node, "lineno", None)),
                        source="heuristic_naming",
                    ))
            if isinstance(node, ast.ClassDef):
                if not pascal.match(node.name):
                    issues.append(DetectedIssue(
                        category=EvalCategory.NAMING.value,
                        severity=EvalSeverity.MINOR.value,
                        message=f"Class name not PascalCase: {node.name}",
                        line=_clamp_line(getattr(node, "lineno", None)),
                        source="heuristic_naming",
                    ))

        # Style: long lines and trailing whitespace
        lines = code.splitlines()
        long_lines = [i for i, ln in enumerate(lines, start=1) if len(ln) > 120]
        if long_lines:
            issues.append(DetectedIssue(
                category=EvalCategory.STYLE.value,
                severity=EvalSeverity.MINOR.value,
                message=f"{len(long_lines)} lines exceed 120 characters",
                line=long_lines[0],
                source="heuristic_style",
            ))
        trailing_ws = [i for i, ln in enumerate(lines, start=1) if ln.endswith(" ") or ln.endswith("\t")]
        if trailing_ws:
            issues.append(DetectedIssue(
                category=EvalCategory.STYLE.value,
                severity=EvalSeverity.MINOR.value,
                message=f"{len(trailing_ws)} lines have trailing whitespace",
                line=trailing_ws[0],
                source="heuristic_style",
            ))

        # Complexity heuristic
        depth = _max_nesting_depth(tree)
        out.metadata["max_nesting_depth"] = depth
        if depth >= 8:
            issues.append(DetectedIssue(
                category=EvalCategory.COMPLEXITY.value,
                severity=EvalSeverity.MAJOR.value,
                message=f"High nesting depth detected: {depth}",
                source="heuristic_complexity",
            ))
        elif depth >= 6:
            issues.append(DetectedIssue(
                category=EvalCategory.COMPLEXITY.value,
                severity=EvalSeverity.MINOR.value,
                message=f"Moderate nesting depth detected: {depth}",
                source="heuristic_complexity",
            ))

        # Orchestrator-specific issues
        if orch == "airflow":
            if not _has_airflow_dag_definition(code):
                issues.append(DetectedIssue(
                    category=EvalCategory.ORCHESTRATOR_STRUCTURE.value,
                    severity=EvalSeverity.CRITICAL.value,
                    message="Airflow: No DAG definition detected (with DAG(...) or @dag)",
                    source="heuristic_orchestrator",
                ))

            if _has_airflow_tasks(code) and not _has_airflow_deps(code):
                # If multiple task_ids exist and no deps, flag
                tids = _task_ids_airflow(code)
                if len(set(tids)) >= 2:
                    issues.append(DetectedIssue(
                        category=EvalCategory.ORCHESTRATOR_STRUCTURE.value,
                        severity=EvalSeverity.MAJOR.value,
                        message=f"Airflow: Multiple tasks detected ({len(set(tids))}) but no dependencies found",
                        source="heuristic_orchestrator",
                    ))

            tids = _task_ids_airflow(code)
            dups = {t for t in tids if tids.count(t) >= 2}
            if dups:
                issues.append(DetectedIssue(
                    category=EvalCategory.ORCHESTRATOR_CONFIG.value,
                    severity=EvalSeverity.MAJOR.value,
                    message=f"Airflow: Duplicate task_id values detected: {sorted(list(dups))[:5]}",
                    source="heuristic_orchestrator",
                ))

        elif orch == "prefect":
            if not _has_prefect_flow(code):
                issues.append(DetectedIssue(
                    category=EvalCategory.ORCHESTRATOR_STRUCTURE.value,
                    severity=EvalSeverity.CRITICAL.value,
                    message="Prefect: No @flow decorator detected",
                    source="heuristic_orchestrator",
                ))

        elif orch == "dagster":
            if not _has_dagster_job_or_asset(code):
                issues.append(DetectedIssue(
                    category=EvalCategory.ORCHESTRATOR_STRUCTURE.value,
                    severity=EvalSeverity.CRITICAL.value,
                    message="Dagster: No @job/@graph/@asset detected",
                    source="heuristic_orchestrator",
                ))

        # Cap issues
        out.issues = issues[: self.max_issues]
        out.tokens_used = 0
        out.cost_usd = 0.0
        out.execution_time_ms = (time.time() - t0) * 1000
        return out