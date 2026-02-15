#!/usr/bin/env python3
"""
Enhanced static analyzer aligned with paper SAT definition:

SAT dimensions (each in [0,10]):
1) correctness
2) code_quality
3) best_practices
4) maintainability
5) robustness

Key rule:
- Issues are logged for diagnostics but are NOT used as penalty terms.
- Scores are computed from measurable static signals (thresholded mappings).
- SAT is the unweighted mean of the five dimension scores.

CLI:
  python scripts/evaluators/enhanced_static_analyzer.py <path/to/code.py> --print-summary --out results.json
"""

# -------------------------------------------------------------------
# Bootstrap imports so this file works as a standalone CLI
# -------------------------------------------------------------------
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]  # .../scripts
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# -------------------------------------------------------------------

import ast
import json
import os
import re
import subprocess
import tempfile
import logging
from collections import Counter
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

from evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationScore,
    Issue,
    Severity,
    Orchestrator,
)

# Optional imports
try:
    from pylint.lint import Run as PylintRun
    from pylint.reporters import JSONReporter
    PYLINT_AVAILABLE = True
except ImportError:
    PYLINT_AVAILABLE = False

try:
    from radon.complexity import cc_visit
    RADON_AVAILABLE = True
except ImportError:
    RADON_AVAILABLE = False

try:
    from bandit.core import manager as bandit_manager
    from bandit.core import config as bandit_config
    import bandit
    BANDIT_AVAILABLE = True
except ImportError:
    BANDIT_AVAILABLE = False


def clamp10(x: float) -> float:
    return max(0.0, min(10.0, float(x)))


def mean_safe(xs: List[float], default: float = 0.0) -> float:
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else default


class EnhancedStaticAnalyzer(BaseEvaluator):
    """
    Paper-aligned SAT evaluator.

    - Produces 5 dimension scores in [0,10]
    - SAT is the unweighted mean of these dimension scores
    - Issues are logged but do not numerically penalise scores
    """

    EVALUATION_TYPE = "enhanced_static_analysis"

    DIMENSIONS = [
        "correctness",
        "code_quality",
        "best_practices",
        "maintainability",
        "robustness",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.intermediate_yaml: Optional[Dict[str, Any]] = None

    def set_reference(self, intermediate_yaml: Dict[str, Any]):
        """Set intermediate YAML for semantic comparison (optional)."""
        self.intermediate_yaml = intermediate_yaml

    def evaluate(self, file_path: Path) -> EvaluationResult:
        """Run enhanced static analysis and return SAT dimension scores + overall SAT."""
        file_path = Path(file_path)
        self.logger.info(f"Running SAT on: {file_path}")

        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return self._error_result(file_path, f"Failed to read file: {e}")

        orchestrator = self.detect_orchestrator(code)
        self.logger.info(f"Detected orchestrator: {orchestrator.value}")

        # Parse AST once (syntax required for SAT)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return self._syntax_error_result(file_path, e)

        result = EvaluationResult(
            evaluation_type=self.EVALUATION_TYPE,
            file_path=str(file_path),
            orchestrator=orchestrator,
            timestamp=datetime.now().isoformat(),
            metadata={
                "file_size_bytes": len(code.encode("utf-8")),
                "line_count": len(code.splitlines()),
                "tools": {
                    "pylint_available": PYLINT_AVAILABLE,
                    "radon_available": RADON_AVAILABLE,
                    "bandit_available": BANDIT_AVAILABLE,
                },
            }
        )

        # Dimension scores (each in [0,10])
        dim_scores: Dict[str, EvaluationScore] = {}
        dim_scores["correctness"] = self._evaluate_correctness(code, tree, orchestrator)
        dim_scores["code_quality"] = self._evaluate_code_quality(code, tree, file_path)
        dim_scores["best_practices"] = self._evaluate_best_practices(code, tree, orchestrator)
        dim_scores["maintainability"] = self._evaluate_maintainability(code, tree)
        dim_scores["robustness"] = self._evaluate_robustness(code, tree, file_path, orchestrator)

        result.scores.update(dim_scores)

        # SAT aggregation: unweighted mean over the five dimensions
        sat_value = mean_safe([dim_scores[d].raw_score for d in self.DIMENSIONS], default=0.0)
        sat_value = clamp10(sat_value)

        # Store overall SAT explicitly
        result.metadata["SAT"] = sat_value
        result.metadata["SAT_dimensions"] = {d: float(dim_scores[d].raw_score) for d in self.DIMENSIONS}

        # Optional: provide a dedicated top-level score entry
        result.scores["SAT"] = EvaluationScore(
            name="SAT",
            raw_score=sat_value,
            issues=[],
            details={"aggregation": "unweighted_mean", "dimensions": self.DIMENSIONS},
            penalties_applied=0.0,
        )

        return result

    # ---------------------------------------------------------------------
    # Basic error / syntax handling
    # ---------------------------------------------------------------------

    def _error_result(self, file_path: Path, error: str) -> EvaluationResult:
        # SAT is still returned with error payload
        return EvaluationResult(
            evaluation_type=self.EVALUATION_TYPE,
            file_path=str(file_path),
            orchestrator=Orchestrator.UNKNOWN,
            timestamp=datetime.now().isoformat(),
            scores={"SAT": EvaluationScore(name="SAT", raw_score=0.0, error=error, penalties_applied=0.0)},
            metadata={"SAT": 0.0, "error": error},
        )

    def _syntax_error_result(self, file_path: Path, error: SyntaxError) -> EvaluationResult:
        # Syntax failure -> SAT=0 by construction (AST cannot be built)
        crit = Issue(Severity.CRITICAL, "syntax", f"Syntax error: {error.msg}", error.lineno)
        return EvaluationResult(
            evaluation_type=self.EVALUATION_TYPE,
            file_path=str(file_path),
            orchestrator=Orchestrator.UNKNOWN,
            timestamp=datetime.now().isoformat(),
            scores={
                "correctness": EvaluationScore(name="correctness", raw_score=0.0, issues=[crit], penalties_applied=0.0),
                "code_quality": EvaluationScore(name="code_quality", raw_score=0.0, penalties_applied=0.0),
                "best_practices": EvaluationScore(name="best_practices", raw_score=0.0, penalties_applied=0.0),
                "maintainability": EvaluationScore(name="maintainability", raw_score=0.0, penalties_applied=0.0),
                "robustness": EvaluationScore(name="robustness", raw_score=0.0, penalties_applied=0.0),
                "SAT": EvaluationScore(
                    name="SAT",
                    raw_score=0.0,
                    issues=[crit],
                    details={"aggregation": "unweighted_mean"},
                    penalties_applied=0.0
                )
            },
            metadata={"SAT": 0.0, "error": f"Syntax error at line {error.lineno}: {error.msg}"}
        )

    # ---------------------------------------------------------------------
    # DIMENSION 1: correctness (0-10)
    # ---------------------------------------------------------------------

    def _evaluate_correctness(self, code: str, tree: ast.AST, orchestrator: Orchestrator) -> EvaluationScore:
        """
        Correctness is a static proxy for "is this plausibly correct code":
        - Syntax validity (guaranteed here): subscore=10
        - Import sanity: subscore in [0,10]
        - Orchestrator structure completeness: subscore in [0,10]
        - Semantic coverage vs intermediate YAML (if present): subscore in [0,10]
        """
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        syntax_sub = 10.0
        import_sub, import_issues = self._score_import_sanity(code, tree)
        struct_sub, struct_issues = self._score_structure_completeness(code, orchestrator)

        issues.extend(import_issues)
        issues.extend(struct_issues)

        subs = [syntax_sub, import_sub, struct_sub]
        details["syntax_subscore"] = syntax_sub
        details["import_subscore"] = import_sub
        details["structure_subscore"] = struct_sub

        if self.intermediate_yaml:
            sem_sub, sem_issues = self._score_semantic_coverage(code, tree)
            subs.append(sem_sub)
            details["semantic_subscore"] = sem_sub
            issues.extend(sem_issues)
        else:
            details["semantic_subscore"] = None

        score = clamp10(mean_safe(subs, default=0.0))
        return EvaluationScore(name="correctness", raw_score=score, issues=issues, details=details, penalties_applied=0.0)

    def _score_import_sanity(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue]]:
        issues: List[Issue] = []
        star_imports = 0
        total_imports = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                total_imports += len(node.names)
            elif isinstance(node, ast.ImportFrom):
                total_imports += len(node.names)
                for alias in node.names:
                    if alias.name == "*":
                        star_imports += 1

        if star_imports > 0:
            issues.append(Issue(
                Severity.MINOR, "import",
                f"Star import detected ({star_imports}); reduces clarity"
            ))
            score = 6.0
        else:
            score = 10.0

        return clamp10(score), issues

    def _score_structure_completeness(self, code: str, orchestrator: Orchestrator) -> Tuple[float, List[Issue]]:
        issues: List[Issue] = []

        criteria: List[Tuple[str, bool, Severity, str]] = []

        code_lower = code.lower()

        if orchestrator == Orchestrator.AIRFLOW:
            has_dag = ("dag(" in code_lower) or ("with dag" in code_lower) or ("@dag" in code_lower)
            has_tasks = ("operator(" in code_lower) or ("@task" in code)
            has_deps = (">>" in code) or ("<<" in code) or ("chain(" in code_lower)

            criteria = [
                ("has_dag", has_dag, Severity.MAJOR, "No DAG definition detected (DAG(...) or @dag)"),
                ("has_tasks", has_tasks, Severity.MAJOR, "No tasks detected (Operator(...) or @task)"),
                ("has_dependencies", has_deps, Severity.MINOR, "No dependencies detected (>>, <<, or chain(...))"),
            ]

        elif orchestrator == Orchestrator.PREFECT:
            has_flow = "@flow" in code
            has_tasks = "@task" in code
            criteria = [
                ("has_flow", has_flow, Severity.MAJOR, "No @flow decorator detected"),
                ("has_tasks", has_tasks, Severity.MINOR, "No @task decorator detected"),
            ]

        elif orchestrator == Orchestrator.DAGSTER:
            has_job = "@job" in code or "@graph" in code
            has_ops = "@op" in code or "@asset" in code
            criteria = [
                ("has_job_or_graph", has_job, Severity.MAJOR, "No @job/@graph detected"),
                ("has_ops_or_assets", has_ops, Severity.MINOR, "No @op/@asset detected"),
            ]
        else:
            issues.append(Issue(Severity.INFO, "structure", "Unknown orchestrator; completeness score defaulted"))
            return 5.0, issues

        passed = 0
        for _name, ok, sev, msg in criteria:
            if ok:
                passed += 1
            else:
                issues.append(Issue(sev, "structure", msg))

        score = 10.0 * (passed / len(criteria)) if criteria else 0.0
        return clamp10(score), issues

    def _score_semantic_coverage(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue]]:
        issues: List[Issue] = []

        expected_tasks = (self.intermediate_yaml or {}).get("tasks", []) or []
        expected_ids = {t.get("task_id") for t in expected_tasks if isinstance(t, dict)}
        expected_ids = {x for x in expected_ids if x}

        if not expected_ids:
            issues.append(Issue(Severity.INFO, "semantic", "No task_ids in reference; semantic score defaulted"))
            return 5.0, issues

        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in expected_ids:
                found.add(node.name)

        for tid in expected_ids:
            if tid in code:
                found.add(tid)

        coverage = len(found) / len(expected_ids)
        score = 10.0 * coverage

        missing = sorted(list(expected_ids - found))
        if missing:
            issues.append(Issue(
                Severity.MAJOR, "semantic",
                f"Missing task identifiers from reference (showing up to 10): {missing[:10]}"
            ))

        return clamp10(score), issues

    # ---------------------------------------------------------------------
    # DIMENSION 2: code_quality (0-10)
    # ---------------------------------------------------------------------

    def _evaluate_code_quality(self, code: str, tree: ast.AST, file_path: Path) -> EvaluationScore:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        style_sub, style_issues, style_details = self._score_style(file_path, code)
        lint_sub, lint_issues, lint_details = self._score_lint(code)
        doc_sub, doc_issues, doc_details = self._score_doc_coverage(tree)
        naming_sub, naming_issues, naming_details = self._score_naming(tree)

        issues.extend(style_issues + lint_issues + doc_issues + naming_issues)
        details.update({
            "style": style_details,
            "lint": lint_details,
            "docs": doc_details,
            "naming": naming_details,
        })

        score = clamp10(mean_safe([style_sub, lint_sub, doc_sub, naming_sub], default=0.0))
        return EvaluationScore(name="code_quality", raw_score=score, issues=issues, details=details, penalties_applied=0.0)

    def _score_style(self, file_path: Path, code: str) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"tool": "flake8", "available": None}

        # flake8 is optional; if absent -> neutral
        try:
            result = subprocess.run(
                ["flake8", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            details["available"] = (result.returncode == 0)
        except Exception:
            details["available"] = False

        if not details["available"]:
            issues.append(Issue(Severity.INFO, "tool", "flake8 not available; style score defaulted"))
            return 5.0, issues, details

        # Try running flake8 (without relying on flake8-json plugin)
        try:
            res = subprocess.run(
                ["flake8", str(file_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (res.stdout or "") + (res.stderr or "")
            lines = code.splitlines() or ["x"]
            violation_lines = [ln for ln in output.splitlines() if ln.strip()]
            vph = (len(violation_lines) / max(1, len(lines))) * 100.0
            details["violations"] = len(violation_lines)
            details["violations_per_100_lines"] = vph
            details["sample"] = violation_lines[:5]

            if vph == 0:
                score = 10.0
            elif vph <= 1:
                score = 9.0
            elif vph <= 3:
                score = 7.5
            elif vph <= 5:
                score = 6.0
            elif vph <= 10:
                score = 4.0
            else:
                score = 2.0

            for ln in violation_lines[:3]:
                issues.append(Issue(Severity.MINOR, "style", ln, tool="flake8"))

            return clamp10(score), issues, details

        except Exception as e:
            issues.append(Issue(Severity.INFO, "tool", f"flake8 failed: {type(e).__name__}; style score defaulted"))
            details["error"] = str(e)
            return 5.0, issues, details

    def _score_lint(self, code: str) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"tool": "pylint", "available": PYLINT_AVAILABLE}

        if not PYLINT_AVAILABLE:
            issues.append(Issue(Severity.INFO, "tool", "pylint not installed; lint score defaulted"))
            return 5.0, issues, details

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            buf = StringIO()
            reporter = JSONReporter(buf)

            try:
                PylintRun([temp_path], reporter=reporter, exit=False)
            except SystemExit:
                pass

            buf.seek(0)
            messages = []
            if buf.getvalue().strip():
                try:
                    messages = json.loads(buf.getvalue())
                except json.JSONDecodeError:
                    messages = []

            counts = Counter(m.get("type") for m in messages)
            n_err = counts.get("error", 0) + counts.get("fatal", 0)
            n_warn = counts.get("warning", 0)
            n_ref = counts.get("refactor", 0)
            n_conv = counts.get("convention", 0)

            details["message_counts"] = dict(counts)

            if n_err > 0:
                score = 3.0 if n_err <= 2 else 1.5
            elif n_warn > 0:
                score = 7.0 if n_warn <= 2 else 5.5
            else:
                score = 9.0 if n_conv <= 10 else 8.0

            for m in messages[:3]:
                sev = Severity.MAJOR if m.get("type") in ["error", "fatal", "warning"] else Severity.MINOR
                issues.append(Issue(
                    sev,
                    "lint",
                    f"[{m.get('symbol')}] {m.get('message')}",
                    m.get("line"),
                    tool="pylint"
                ))

            return clamp10(score), issues, details

        except Exception as e:
            details["error"] = f"{type(e).__name__}: {e}"
            issues.append(Issue(Severity.INFO, "tool", "pylint failed; lint score defaulted", tool="pylint"))
            return 5.0, issues, details
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def _score_doc_coverage(self, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        module_doc = ast.get_docstring(tree) is not None
        func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        class_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        total = 1 + len(func_nodes) + len(class_nodes)
        documented = (1 if module_doc else 0)

        undocumented_funcs = []
        for f in func_nodes:
            if ast.get_docstring(f):
                documented += 1
            else:
                undocumented_funcs.append(f.name)

        for c in class_nodes:
            if ast.get_docstring(c):
                documented += 1

        cov = documented / total if total > 0 else 0.0
        details["coverage_ratio"] = cov
        details["undocumented_functions"] = undocumented_funcs[:10]

        if cov >= 0.9:
            score = 10.0
        elif cov >= 0.7:
            score = 8.0
        elif cov >= 0.5:
            score = 6.5
        elif cov >= 0.3:
            score = 5.0
        elif cov > 0:
            score = 3.5
        else:
            score = 2.0
            issues.append(Issue(Severity.MINOR, "documentation", "No docstrings found"))

        return clamp10(score), issues, details

    def _score_naming(self, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        snake_case = re.compile(r"^[a-z_][a-z0-9_]*$")
        pascal_case = re.compile(r"^[A-Z][a-zA-Z0-9]*$")

        total = 0
        violations = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total += 1
                if not snake_case.match(node.name) and not node.name.startswith("_"):
                    violations += 1
            elif isinstance(node, ast.ClassDef):
                total += 1
                if not pascal_case.match(node.name):
                    violations += 1

        details["total_named_entities"] = total
        details["naming_violations"] = violations

        if total == 0:
            return 10.0, issues, details

        rate = violations / total
        if rate == 0:
            score = 10.0
        elif rate <= 0.1:
            score = 8.5
        elif rate <= 0.2:
            score = 7.0
        elif rate <= 0.3:
            score = 5.5
        else:
            score = 4.0
            issues.append(Issue(Severity.MINOR, "naming", f"Naming violation rate high: {rate:.2f}"))

        return clamp10(score), issues, details

    # ---------------------------------------------------------------------
    # DIMENSION 3: best_practices (0-10)
    # ---------------------------------------------------------------------

    def _evaluate_best_practices(self, code: str, tree: ast.AST, orchestrator: Orchestrator) -> EvaluationScore:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"orchestrator": orchestrator.value}

        if orchestrator == Orchestrator.AIRFLOW:
            criteria = {
                "uses_default_args": ("default_args" in code),
                "declares_dependencies": any(tok in code for tok in [">>", "<<", "chain("]),
                "externalizes_config": any(tok in code for tok in ["os.getenv", "os.environ", "Variable.get"]),
                "sets_task_id": ("task_id=" in code),
                "sets_retries": ("retries=" in code),
                "sets_timeouts": any(tok in code for tok in ["execution_timeout", "dagrun_timeout", "timeout="]),
            }
        elif orchestrator == Orchestrator.PREFECT:
            criteria = {
                "has_flow": ("@flow" in code),
                "has_task": ("@task" in code),
                "externalizes_config": any(tok in code for tok in ["os.getenv", "os.environ"]),
                "uses_retries": ("retries=" in code),
                "uses_timeout": ("timeout_seconds" in code or "timeout=" in code),
            }
        elif orchestrator == Orchestrator.DAGSTER:
            criteria = {
                "has_job_or_graph": ("@job" in code or "@graph" in code),
                "has_op_or_asset": ("@op" in code or "@asset" in code),
                "uses_config_schema": ("config_schema" in code or "OpConfig" in code),
                "uses_retry_policy": ("RetryPolicy" in code or "retry_policy" in code),
            }
        else:
            criteria = {"unknown_orchestrator": True}
            issues.append(Issue(Severity.INFO, "best_practices", "Unknown orchestrator; best_practices defaulted"))

        details["criteria"] = criteria
        passed = sum(1 for v in criteria.values() if v)
        total = len(criteria) if criteria else 1
        score = 10.0 * (passed / total)

        # log missing criteria as INFO (no penalties)
        for k, ok in criteria.items():
            if not ok:
                issues.append(Issue(Severity.INFO, "best_practices", f"Missing best-practice signal: {k}"))

        return EvaluationScore(name="best_practices", raw_score=clamp10(score), issues=issues, details=details, penalties_applied=0.0)

    # ---------------------------------------------------------------------
    # DIMENSION 4: maintainability (0-10)
    # ---------------------------------------------------------------------

    def _evaluate_maintainability(self, code: str, tree: ast.AST) -> EvaluationScore:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        cplx_sub, cplx_issues, cplx_details = self._score_complexity(code)
        cfg_sub, cfg_issues, cfg_details = self._score_config_externalization(code)
        org_sub, org_issues, org_details = self._score_code_organization(code)

        issues.extend(cplx_issues + cfg_issues + org_issues)
        details["complexity"] = cplx_details
        details["config_externalization"] = cfg_details
        details["organization"] = org_details

        score = clamp10(mean_safe([cplx_sub, cfg_sub, org_sub], default=0.0))
        return EvaluationScore(name="maintainability", raw_score=score, issues=issues, details=details, penalties_applied=0.0)

    def _score_complexity(self, code: str) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"tool": "radon", "available": RADON_AVAILABLE}

        if not RADON_AVAILABLE:
            issues.append(Issue(Severity.INFO, "tool", "radon not installed; complexity defaulted"))
            return 5.0, issues, details

        try:
            results = list(cc_visit(code))
            if not results:
                details["avg_complexity"] = 0.0
                return 10.0, issues, details

            complexities = [r.complexity for r in results]
            avg = sum(complexities) / len(complexities)
            mx = max(complexities)
            details["avg_complexity"] = avg
            details["max_complexity"] = mx
            details["count"] = len(complexities)

            if avg <= 3:
                score = 10.0
            elif avg <= 5:
                score = 8.5
            elif avg <= 8:
                score = 7.0
            elif avg <= 12:
                score = 5.5
            else:
                score = 4.0
                issues.append(Issue(Severity.MAJOR, "complexity", f"High avg complexity: {avg:.1f}"))

            if mx > 15:
                issues.append(Issue(Severity.MAJOR, "complexity", f"Very high function complexity: {mx}"))

            return clamp10(score), issues, details

        except Exception as e:
            details["error"] = f"{type(e).__name__}: {e}"
            issues.append(Issue(Severity.INFO, "tool", "radon failed; complexity defaulted"))
            return 5.0, issues, details

    def _score_config_externalization(self, code: str) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        has_env = ("os.getenv" in code or "os.environ" in code)
        has_literal_secret = bool(re.search(r"(api_key|token|password|secret)\s*=\s*['\"][^'\"]+['\"]", code, re.I))

        details["uses_env_vars"] = has_env
        details["has_literal_secret_pattern"] = has_literal_secret

        if has_literal_secret:
            issues.append(Issue(Severity.MAJOR, "security", "Potential hardcoded secret pattern detected"))

        criteria = {
            "externalizes_config": has_env,
            "no_obvious_literal_secrets": (not has_literal_secret),
        }
        passed = sum(1 for v in criteria.values() if v)
        score = 10.0 * (passed / len(criteria))

        return clamp10(score), issues, {"criteria": criteria, **details}

    def _score_code_organization(self, code: str) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        lines = code.splitlines() or ["x"]
        details: Dict[str, Any] = {}

        long_lines = sum(1 for line in lines if len(line) > 120)
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        ratio_comments = comment_lines / len(lines)

        details["long_lines_over_120"] = long_lines
        details["comment_ratio"] = ratio_comments

        criteria = {
            "reasonable_line_length": (long_lines <= 4),
            "has_some_comments": (ratio_comments >= 0.02),
        }
        passed = sum(1 for v in criteria.values() if v)
        score = 10.0 * (passed / len(criteria))

        for k, ok in criteria.items():
            if not ok:
                issues.append(Issue(Severity.INFO, "organization", f"Organization signal missing: {k}"))

        return clamp10(score), issues, {"criteria": criteria, **details}

    # ---------------------------------------------------------------------
    # DIMENSION 5: robustness (0-10)
    # ---------------------------------------------------------------------

    def _evaluate_robustness(self, code: str, tree: ast.AST, file_path: Path, orchestrator: Orchestrator) -> EvaluationScore:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        err_sub, err_issues, err_details = self._score_error_handling(tree)
        retry_sub, retry_issues, retry_details = self._score_retry_signals(code, orchestrator)
        timeout_sub, timeout_issues, timeout_details = self._score_timeout_signals(code, orchestrator)
        sec_sub, sec_issues, sec_details = self._score_security_bandit(code)

        issues.extend(err_issues + retry_issues + timeout_issues + sec_issues)
        details["error_handling"] = err_details
        details["retries"] = retry_details
        details["timeouts"] = timeout_details
        details["security"] = sec_details

        score = clamp10(mean_safe([err_sub, retry_sub, timeout_sub, sec_sub], default=0.0))
        return EvaluationScore(name="robustness", raw_score=score, issues=issues, details=details, penalties_applied=0.0)

    def _score_error_handling(self, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        try_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
        func_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        details["try_blocks"] = len(try_nodes)
        details["function_defs"] = len(func_nodes)

        if not func_nodes:
            return 7.0, issues, details

        if not try_nodes:
            issues.append(Issue(Severity.INFO, "error_handling", "No try/except found in function bodies"))
            return 6.0, issues, details

        bare = sum(1 for t in try_nodes for h in t.handlers if h.type is None)
        details["bare_except"] = bare
        if bare > 0:
            issues.append(Issue(Severity.MINOR, "error_handling", f"Bare except detected ({bare})"))
            return 6.5, issues, details

        return 8.5, issues, details

    def _score_retry_signals(self, code: str, orchestrator: Orchestrator) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        patterns = {
            Orchestrator.AIRFLOW: ["retries=", "retry_delay=", "retry_exponential_backoff"],
            Orchestrator.PREFECT: ["retries=", "retry_delay_seconds="],
            Orchestrator.DAGSTER: ["retry_policy", "RetryPolicy", "max_retries"],
        }.get(orchestrator, ["retries=", "retry"])

        found = [p for p in patterns if p in code]
        details["found_patterns"] = found

        if not found:
            issues.append(Issue(Severity.INFO, "robustness", "No retry configuration signals detected"))
            return 5.5, issues, details
        if len(found) == 1:
            return 7.0, issues, details
        return 8.5, issues, details

    def _score_timeout_signals(self, code: str, orchestrator: Orchestrator) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        patterns = {
            Orchestrator.AIRFLOW: ["execution_timeout", "dagrun_timeout", "timeout="],
            Orchestrator.PREFECT: ["timeout_seconds=", "timeout="],
            Orchestrator.DAGSTER: ["timeout="],
        }.get(orchestrator, ["timeout"])

        found = [p for p in patterns if p in code]
        details["found_patterns"] = found

        if not found:
            issues.append(Issue(Severity.INFO, "robustness", "No timeout configuration signals detected"))
            return 5.5, issues, details
        return 8.0, issues, details

    def _score_security_bandit(self, code: str) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"tool": "bandit", "available": BANDIT_AVAILABLE}

        if not BANDIT_AVAILABLE:
            issues.append(Issue(Severity.INFO, "tool", "bandit not installed; security defaulted"))
            return 5.0, issues, details

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            mgr = bandit_manager.BanditManager(bandit_config.BanditConfig(), None)
            mgr.discover_files([temp_path], recursive=False)
            mgr.run_tests()
            bandit_issues = mgr.get_issue_list()

            high = sum(1 for i in bandit_issues if i.severity == bandit.HIGH)
            med = sum(1 for i in bandit_issues if i.severity == bandit.MEDIUM)
            low = sum(1 for i in bandit_issues if i.severity == bandit.LOW)

            details["counts"] = {"high": high, "medium": med, "low": low}

            if high > 0:
                score = 3.0
            elif med > 0:
                score = 5.5
            elif low > 0:
                score = 7.5
            else:
                score = 9.0

            for bi in bandit_issues[:3]:
                sev = Severity.CRITICAL if bi.severity == bandit.HIGH else Severity.MAJOR
                issues.append(Issue(
                    sev,
                    "security",
                    f"[{bi.test_id}] {bi.text}",
                    bi.lineno,
                    tool="bandit"
                ))

            return clamp10(score), issues, details

        except Exception as e:
            details["error"] = f"{type(e).__name__}: {e}"
            issues.append(Issue(Severity.INFO, "tool", "bandit failed; security defaulted", tool="bandit"))
            return 5.0, issues, details
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def print_summary(result: EvaluationResult) -> None:
    sat = float(result.metadata.get("SAT", 0.0) or 0.0)
    dims = result.metadata.get("SAT_dimensions", {}) or {}
    orch = result.orchestrator.value

    crit = len([i for i in result.all_issues if i.severity == Severity.CRITICAL])
    maj = len([i for i in result.all_issues if i.severity == Severity.MAJOR])
    minor = len([i for i in result.all_issues if i.severity == Severity.MINOR])

    print("\n" + "=" * 80)
    print("SAT — Enhanced Static Analyzer Summary (penalty-free scoring)")
    print("=" * 80)
    print(f"File:         {result.file_path}")
    print(f"Orchestrator: {orch}")
    print(f"SAT:          {sat:.2f}/10")
    if dims:
        print("Dimensions:")
        for k, v in dims.items():
            print(f"  - {k}: {float(v):.2f}")
    print(f"Issues: total={len(result.all_issues)} critical={crit} major={maj} minor={minor}")
    print("=" * 80)


def build_sat_payload(result: EvaluationResult) -> Dict[str, Any]:
    """
    Build a JSON payload with:
    - full EvaluationResult.to_dict()
    - flattened issues for convenience
    """
    payload = result.to_dict()
    payload["issues"] = [i.to_dict() for i in result.all_issues]
    payload["issue_summary"] = {
        "total": len(result.all_issues),
        "critical": len([i for i in result.all_issues if i.severity == Severity.CRITICAL]),
        "major": len([i for i in result.all_issues if i.severity == Severity.MAJOR]),
        "minor": len([i for i in result.all_issues if i.severity == Severity.MINOR]),
        "info": len([i for i in result.all_issues if i.severity == Severity.INFO]),
    }
    return payload


def default_sidecar_path(code_file: Path) -> Path:
    """
    Default output path when user doesn't specify --out or --out-dir:
      foo.py -> foo.py.sat.json
    """
    return code_file.with_name(code_file.name + ".sat.json")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run SAT (EnhancedStaticAnalyzer) on a Python file.")
    parser.add_argument("file", help="Path to generated workflow Python file")

    # Output options:
    parser.add_argument("--out", default=None, help="Write full JSON result to this exact path")
    parser.add_argument("--out-dir", default=None, help="Write JSON result into this directory (auto filename)")
    parser.add_argument("--stdout", action="store_true", help="Also print JSON to stdout")

    parser.add_argument("--print-summary", action="store_true", help="Print a console summary")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")

    code_path = Path(args.file)
    analyzer = EnhancedStaticAnalyzer(config=None)
    result = analyzer.evaluate(code_path)

    if args.print_summary:
        print_summary(result)

    payload = build_sat_payload(result)

    # Decide output path:
    # 1) --out wins
    # 2) else --out-dir
    # 3) else sidecar next to input file
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"sat_{code_path.stem}_{ts}.json"
    else:
        out_path = default_sidecar_path(code_path)

    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote: {out_path}")

    if args.stdout:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()