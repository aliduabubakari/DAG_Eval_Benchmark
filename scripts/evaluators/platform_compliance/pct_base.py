"""
Penalty-free Platform Compliance Tester (PCT) base class.

Paper-aligned definition:

- Gate checks determine platform validity:
    G(p) = 1 iff all critical gates pass
    If any critical gate fails: PCT(p)=0 and all dimension scores=0.

- Dimension scores are in [0,10] and are NOT reduced by penalties.
  Issues are logged only for diagnostics.

- Aggregation:
    PCT(p) = G(p) * (1/5) * sum_{j in D_comp} c_j(p)

Dimensions:
  1) loadability
  2) structure_validity
  3) configuration_validity
  4) task_validity
  5) executability
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional
import ast
import re

from ..base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationScore,
    GateCheckResult,
    Issue,
    Severity,
    Orchestrator,
)


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _clamp10(x: float) -> float:
    return max(0.0, min(10.0, float(x)))


class BasePlatformComplianceTester(BaseEvaluator):
    """
    Penalty-free, gate-based PCT base class.

    Subclasses must implement:
      - _check_minimum_structure
      - _check_platform_load
      - _check_required_constructs
      - _check_schedule_config
      - _check_default_args
      - _check_task_definitions
      - _check_operator_usage
      - _check_dryrun_capability

    Notes:
    - Issues with Severity.CRITICAL do NOT automatically fail PCT.
      Only GateCheckResult controls pass/fail.
    """

    EVALUATION_TYPE = "platform_compliance"
    ORCHESTRATOR: Orchestrator = Orchestrator.UNKNOWN

    DIMENSIONS = [
        "loadability",
        "structure_validity",
        "configuration_validity",
        "task_validity",
        "executability",
    ]

    # ---------------------------------------------------------------------
    # Main entry point
    # ---------------------------------------------------------------------

    def evaluate(self, file_path: Path) -> EvaluationResult:
        self.logger.info(f"Running PCT (penalty-free) for {self.ORCHESTRATOR.value}: {file_path}")

        try:
            code = Path(file_path).read_text(encoding="utf-8")
        except Exception as e:
            return self._create_error_result(file_path, f"Failed to read file: {e}")

        if not code.strip():
            return self._create_error_result(file_path, "File is empty")

        # Gate 1: syntax (AST parse)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return self._create_syntax_error_result(file_path, e)

        result = EvaluationResult(
            evaluation_type=self.EVALUATION_TYPE,
            file_path=str(file_path),
            orchestrator=self.ORCHESTRATOR,
            timestamp=datetime.now().isoformat(),
            metadata={
                "file_size_bytes": len(code.encode("utf-8")),
                "line_count": len(code.splitlines()),
            },
        )

        # Gate checks define G(p)
        result.gate_checks = self._run_gate_checks(code, tree, file_path)

        if not result.gates_passed:
            self._apply_zero_scores(result)
            result.metadata["PCT"] = 0.0
            result.metadata["platform_gate_passed"] = False
            result.metadata["Passed"] = False  # gate-only passed
            return result

        # Dimension scores (each in [0,10]) — penalty-free
        result.scores["loadability"] = self._evaluate_loadability(code, tree, file_path)
        result.scores["structure_validity"] = self._evaluate_structure_validity(code, tree)
        result.scores["configuration_validity"] = self._evaluate_configuration_validity(code, tree)
        result.scores["task_validity"] = self._evaluate_task_validity(code, tree)
        result.scores["executability"] = self._evaluate_executability(code, tree, file_path)

        # Defensive: enforce penalty-free invariant
        for s in result.scores.values():
            s.penalties_applied = 0.0

        pct = _clamp10(_mean([result.scores[d].raw_score for d in self.DIMENSIONS]))

        result.metadata["PCT"] = pct
        result.metadata["platform_gate_passed"] = True
        result.metadata["Passed"] = True
        result.metadata["PCT_dimensions"] = {d: float(result.scores[d].raw_score) for d in self.DIMENSIONS}

        return result

    # ---------------------------------------------------------------------
    # Error helpers
    # ---------------------------------------------------------------------

    def _create_error_result(self, file_path: Path, error: str) -> EvaluationResult:
        result = EvaluationResult(
            evaluation_type=self.EVALUATION_TYPE,
            file_path=str(file_path),
            orchestrator=self.ORCHESTRATOR,
            timestamp=datetime.now().isoformat(),
            metadata={"error": error},
        )
        result.gate_checks.append(GateCheckResult(
            name="file_readable",
            passed=False,
            message=error,
            is_critical=True,
        ))
        self._apply_zero_scores(result)
        result.metadata["PCT"] = 0.0
        result.metadata["platform_gate_passed"] = False
        result.metadata["Passed"] = False
        return result

    def _create_syntax_error_result(self, file_path: Path, error: SyntaxError) -> EvaluationResult:
        result = EvaluationResult(
            evaluation_type=self.EVALUATION_TYPE,
            file_path=str(file_path),
            orchestrator=self.ORCHESTRATOR,
            timestamp=datetime.now().isoformat(),
            metadata={"error": f"Syntax error at line {error.lineno}: {error.msg}"},
        )
        result.gate_checks.append(GateCheckResult(
            name="syntax_valid",
            passed=False,
            message=f"Syntax error at line {error.lineno}: {error.msg}",
            is_critical=True,
        ))
        self._apply_zero_scores(result)
        result.metadata["PCT"] = 0.0
        result.metadata["platform_gate_passed"] = False
        result.metadata["Passed"] = False
        return result

    def _apply_zero_scores(self, result: EvaluationResult) -> None:
        for dim in self.DIMENSIONS:
            result.scores[dim] = EvaluationScore(
                name=dim,
                raw_score=0.0,
                weight=1.0,
                issues=[],
                details={},
                penalties_applied=0.0,
            )

    # ---------------------------------------------------------------------
    # Gate checks
    # ---------------------------------------------------------------------

    def _run_gate_checks(self, code: str, tree: ast.AST, file_path: Path) -> List[GateCheckResult]:
        gates: List[GateCheckResult] = []

        gates.append(GateCheckResult(
            name="syntax_valid",
            passed=True,
            message="Code has valid Python syntax",
            is_critical=True,
        ))

        has_structure = self._check_minimum_structure(code)
        gates.append(GateCheckResult(
            name="minimum_structure",
            passed=bool(has_structure),
            message="Has minimum pipeline structure" if has_structure else "Missing basic pipeline structure",
            is_critical=True,
        ))

        import_ok, import_msg = self._check_critical_imports(code, tree)
        gates.append(GateCheckResult(
            name="critical_imports",
            passed=bool(import_ok),
            message=import_msg,
            is_critical=True,
        ))

        return gates

    @abstractmethod
    def _check_minimum_structure(self, code: str) -> bool:
        raise NotImplementedError

    def _check_critical_imports(self, code: str, tree: ast.AST) -> Tuple[bool, str]:
        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split(".")[0])

        if not imports:
            return True, "No imports found (might be OK for simple scripts)"

        broken_patterns = ["undefined_module", "nonexistent"]
        for imp in imports:
            if any(p in imp.lower() for p in broken_patterns):
                return False, f"Broken import detected: {imp}"

        return True, f"Found {len(imports)} imports"

    # ---------------------------------------------------------------------
    # Dimension 1: Loadability (0–10)
    # ---------------------------------------------------------------------

    def _evaluate_loadability(self, code: str, tree: ast.AST, file_path: Path) -> EvaluationScore:
        """
        Loadability subcomponents:
          - import resolution (0–3)
          - platform-specific load/discovery (0–4)  [subclass hook]
          - module structure sanity (0–3)
        """
        issues: List[Issue] = []
        details: Dict[str, Any] = {}
        base_score = 0.0

        import_score, import_issues, import_details = self._check_import_resolution(code, tree)
        base_score += import_score
        issues.extend(import_issues)
        details["imports"] = import_details

        platform_score, platform_issues, platform_details = self._check_platform_load(code, file_path)
        base_score += platform_score
        issues.extend(platform_issues)
        details["platform_load"] = platform_details

        structure_score, structure_issues, structure_details = self._check_module_structure(code, tree)
        base_score += structure_score
        issues.extend(structure_issues)
        details["module_structure"] = structure_details

        return EvaluationScore(
            name="loadability",
            raw_score=_clamp10(min(10.0, base_score)),
            weight=1.0,
            issues=issues,
            details=details,
            penalties_applied=0.0,
        )

    def _check_import_resolution(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"total_imports": 0, "standard_lib": 0, "third_party": 0}

        stdlib = {
            "os", "sys", "json", "datetime", "pathlib", "typing", "logging",
            "collections", "itertools", "functools", "tempfile", "subprocess",
            "re", "ast", "time", "copy", "io", "abc", "dataclasses", "enum",
        }

        imports: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        details["total_imports"] = len(imports)
        details["standard_lib"] = len(imports & stdlib)
        details["third_party"] = len(imports - stdlib)

        if not imports:
            return 2.0, issues, details

        if "import *" in code:
            issues.append(Issue(Severity.MINOR, "imports", "Star import detected; reduces clarity"))

        score = 3.0
        orchestrator_imports = {"airflow", "prefect", "dagster"}
        if not (imports & orchestrator_imports):
            issues.append(Issue(Severity.MAJOR, "imports", "No orchestrator imports found"))
            score -= 1.0

        return max(0.0, score), issues, details

    @abstractmethod
    def _check_platform_load(self, code: str, file_path: Path) -> Tuple[float, List[Issue], Dict[str, Any]]:
        raise NotImplementedError

    def _check_module_structure(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {
            "has_functions": False,
            "has_classes": False,
            "function_count": 0,
            "class_count": 0,
            "has_main_guard": False,
        }

        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        details["function_count"] = len(functions)
        details["class_count"] = len(classes)
        details["has_functions"] = len(functions) > 0
        details["has_classes"] = len(classes) > 0

        score = 2.0
        if functions:
            score += 0.5
        if "__name__" in code and "__main__" in code:
            details["has_main_guard"] = True
            score += 0.5

        return min(3.0, score), issues, details

    # ---------------------------------------------------------------------
    # Dimension 2: Structure validity (0–10)
    # ---------------------------------------------------------------------

    def _evaluate_structure_validity(self, code: str, tree: ast.AST) -> EvaluationScore:
        """
        Structure validity subcomponents:
          - required constructs (0–4)  [subclass hook]
          - dependency graph (0–3)
          - orphan tasks (0–3)
        """
        issues: List[Issue] = []
        details: Dict[str, Any] = {}
        base_score = 0.0

        construct_score, construct_issues, construct_details = self._check_required_constructs(code, tree)
        base_score += construct_score
        issues.extend(construct_issues)
        details["constructs"] = construct_details

        graph_score, graph_issues, graph_details = self._check_dependency_graph(code, tree)
        base_score += graph_score
        issues.extend(graph_issues)
        details["dependency_graph"] = graph_details

        orphan_score, orphan_issues, orphan_details = self._check_orphan_tasks(code, tree)
        base_score += orphan_score
        issues.extend(orphan_issues)
        details["orphan_check"] = orphan_details

        return EvaluationScore(
            name="structure_validity",
            raw_score=_clamp10(min(10.0, base_score)),
            weight=1.0,
            issues=issues,
            details=details,
            penalties_applied=0.0,
        )

    @abstractmethod
    def _check_required_constructs(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        raise NotImplementedError

    def _check_dependency_graph(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"has_dependencies": False, "dependency_count": 0}

        dep_patterns = [">>", "<<", "chain(", "set_downstream", "set_upstream"]
        dep_count = sum(code.count(p) for p in dep_patterns)

        details["dependency_count"] = dep_count
        details["has_dependencies"] = dep_count > 0

        task_count = self._count_tasks(code)

        if task_count <= 1:
            return 3.0, issues, details

        if dep_count == 0 and task_count > 1:
            issues.append(Issue(
                Severity.MAJOR, "structure",
                f"Multiple tasks ({task_count}) but no dependencies defined"
            ))
            return 1.0, issues, details

        return 3.0, issues, details

    def _check_orphan_tasks(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"task_count": 0, "orphan_count": 0}

        task_ids = self._extract_task_ids(code)
        details["task_count"] = len(task_ids)

        if len(task_ids) <= 1:
            return 3.0, issues, details

        connected = set()
        for task_id in task_ids:
            patterns = [
                f"{task_id} >>", f">> {task_id}",
                f"{task_id} <<", f"<< {task_id}",
                f"chain({task_id}", f", {task_id})",
                f"({task_id},", f"{task_id}(",
            ]
            if any(p in code for p in patterns):
                connected.add(task_id)

        orphan_count = len(task_ids) - len(connected)
        details["orphan_count"] = orphan_count

        if orphan_count > 0:
            orphan_ratio = orphan_count / len(task_ids)
            if orphan_ratio > 0.5:
                issues.append(Issue(Severity.MAJOR, "structure", f"{orphan_count} orphan tasks detected (>50%)"))
                return 1.0, issues, details
            else:
                issues.append(Issue(Severity.MINOR, "structure", f"{orphan_count} orphan tasks detected"))
                return 2.0, issues, details

        return 3.0, issues, details

    def _extract_task_ids(self, code: str) -> Set[str]:
        task_ids: Set[str] = set()
        patterns = [
            r"task_id\s*=\s*['\"]([^'\"]+)['\"]",
            r"(\w+)\s*=\s*\w+Operator\s*\(",
            r"name\s*=\s*['\"]([^'\"]+)['\"]",
        ]
        for pattern in patterns:
            task_ids.update(re.findall(pattern, code))
        return {t for t in task_ids if t}

    def _count_tasks(self, code: str) -> int:
        return len(self._extract_task_ids(code))

    # ---------------------------------------------------------------------
    # Dimension 3: Configuration validity (0–10)
    # ---------------------------------------------------------------------

    def _evaluate_configuration_validity(self, code: str, tree: ast.AST) -> EvaluationScore:
        """
        Configuration validity subcomponents:
          - schedule configuration (0–2.5) [subclass hook]
          - default args/params (0–2.5)    [subclass hook]
          - connection references (0–2.5)
          - security (0–2.5)
        """
        issues: List[Issue] = []
        details: Dict[str, Any] = {}
        base_score = 0.0

        schedule_score, schedule_issues, schedule_details = self._check_schedule_config(code, tree)
        base_score += schedule_score
        issues.extend(schedule_issues)
        details["schedule"] = schedule_details

        args_score, args_issues, args_details = self._check_default_args(code, tree)
        base_score += args_score
        issues.extend(args_issues)
        details["default_args"] = args_details

        conn_score, conn_issues, conn_details = self._check_connection_config(code, tree)
        base_score += conn_score
        issues.extend(conn_issues)
        details["connections"] = conn_details

        sec_score, sec_issues, sec_details = self._check_security_config(code)
        base_score += sec_score
        issues.extend(sec_issues)
        details["security"] = sec_details

        return EvaluationScore(
            name="configuration_validity",
            raw_score=_clamp10(min(10.0, base_score)),
            weight=1.0,
            issues=issues,
            details=details,
            penalties_applied=0.0,
        )

    @abstractmethod
    def _check_schedule_config(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def _check_default_args(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        raise NotImplementedError

    def _check_connection_config(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"connection_count": 0, "has_connections": False}

        conn_patterns = [
            r"conn_id\s*=\s*['\"]([^'\"]+)['\"]",
            r"connection_id\s*=\s*['\"]([^'\"]+)['\"]",
        ]

        connections: Set[str] = set()
        for pattern in conn_patterns:
            connections.update(re.findall(pattern, code))
        connections = {c for c in connections if c}

        details["connection_count"] = len(connections)
        details["has_connections"] = bool(connections)

        hardcoded_patterns = [
            r"postgresql://\w+:\w+@",
            r"mysql://\w+:\w+@",
            r"mongodb://\w+:\w+@",
            r"redis://:\w+@",
        ]
        for pattern in hardcoded_patterns:
            if re.search(pattern, code):
                issues.append(Issue(Severity.MAJOR, "configuration", "Hardcoded DB connection string detected"))
                return 0.5, issues, details

        if connections:
            return 2.5, issues, details
        return 1.5, issues, details

    def _check_security_config(self, code: str) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"hardcoded_secrets": False, "uses_env_vars": False}

        if "os.getenv" in code or "os.environ" in code:
            details["uses_env_vars"] = True

        secret_patterns = [
            (r"password\s*=\s*['\"][^'\"]{4,}['\"]", "password"),
            (r"api_key\s*=\s*['\"][^'\"]{8,}['\"]", "api_key"),
            (r"token\s*=\s*['\"][^'\"]{8,}['\"]", "token"),
            (r"secret\s*=\s*['\"][^'\"]{4,}['\"]", "secret"),
        ]
        placeholder_indicators = ["{{", "}}", "your_", "placeholder", "xxx", "***", "example", "changeme", "fixme", "todo"]

        for pattern, secret_type in secret_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                match_lower = match.lower() if isinstance(match, str) else str(match).lower()
                if not any(p in match_lower for p in placeholder_indicators):
                    issues.append(Issue(Severity.MAJOR, "security", f"Potential hardcoded {secret_type} detected"))
                    details["hardcoded_secrets"] = True
                    return 0.5, issues, details

        score = 2.0 + (0.5 if details["uses_env_vars"] else 0.0)
        return min(2.5, score), issues, details

    # ---------------------------------------------------------------------
    # Dimension 4: Task validity (0–10)
    # ---------------------------------------------------------------------

    def _evaluate_task_validity(self, code: str, tree: ast.AST) -> EvaluationScore:
        """
        Task validity subcomponents:
          - task definitions (0–4) [subclass hook]
          - operator/decorator usage (0–3) [subclass hook]
          - task parameters (0–3)
        """
        issues: List[Issue] = []
        details: Dict[str, Any] = {}
        base_score = 0.0

        def_score, def_issues, def_details = self._check_task_definitions(code, tree)
        base_score += def_score
        issues.extend(def_issues)
        details["definitions"] = def_details

        op_score, op_issues, op_details = self._check_operator_usage(code, tree)
        base_score += op_score
        issues.extend(op_issues)
        details["operator_usage"] = op_details

        param_score, param_issues, param_details = self._check_task_parameters(code, tree)
        base_score += param_score
        issues.extend(param_issues)
        details["parameters"] = param_details

        return EvaluationScore(
            name="task_validity",
            raw_score=_clamp10(min(10.0, base_score)),
            weight=1.0,
            issues=issues,
            details=details,
            penalties_applied=0.0,
        )

    @abstractmethod
    def _check_task_definitions(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def _check_operator_usage(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        raise NotImplementedError

    def _check_task_parameters(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"has_task_ids": False, "has_retries": False, "has_timeouts": False}

        score = 1.0
        if "task_id=" in code or "name=" in code:
            details["has_task_ids"] = True
            score += 1.0
        else:
            issues.append(Issue(Severity.MINOR, "task", "No explicit task identifiers found"))

        if "retries=" in code or "retry" in code.lower():
            details["has_retries"] = True
            score += 0.5

        if "timeout" in code.lower() or "execution_timeout" in code:
            details["has_timeouts"] = True
            score += 0.5

        return min(3.0, score), issues, details

    # ---------------------------------------------------------------------
    # Dimension 5: Executability (0–10)
    # ---------------------------------------------------------------------

    def _evaluate_executability(self, code: str, tree: ast.AST, file_path: Path) -> EvaluationScore:
        """
        Executability subcomponents:
          - dry-run capability (0–4) [subclass hook]
          - runtime safety heuristics (0–3)
          - external deps signals (0–3)
        """
        issues: List[Issue] = []
        details: Dict[str, Any] = {}
        base_score = 0.0

        dryrun_score, dryrun_issues, dryrun_details = self._check_dryrun_capability(code, file_path)
        base_score += dryrun_score
        issues.extend(dryrun_issues)
        details["dryrun"] = dryrun_details

        runtime_score, runtime_issues, runtime_details = self._check_runtime_safety(code, tree)
        base_score += runtime_score
        issues.extend(runtime_issues)
        details["runtime_safety"] = runtime_details

        ext_score, ext_issues, ext_details = self._check_external_deps(code)
        base_score += ext_score
        issues.extend(ext_issues)
        details["external_deps"] = ext_details

        return EvaluationScore(
            name="executability",
            raw_score=_clamp10(min(10.0, base_score)),
            weight=1.0,
            issues=issues,
            details=details,
            penalties_applied=0.0,
        )

    @abstractmethod
    def _check_dryrun_capability(self, code: str, file_path: Path) -> Tuple[float, List[Issue], Dict[str, Any]]:
        raise NotImplementedError

    def _check_runtime_safety(self, code: str, tree: ast.AST) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {}

        score = 3.0
        bare_except = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                bare_except += 1

        details["bare_except_count"] = bare_except
        if bare_except > 0:
            issues.append(Issue(Severity.MINOR, "runtime", "Bare 'except:' clause found; may hide errors"))
            score = max(0.0, score - 0.25)

        if "/ 0" in code or "/0" in code:
            issues.append(Issue(Severity.MAJOR, "runtime", "Potential division by zero"))
            score = max(0.0, score - 1.0)

        return max(0.0, score), issues, details

    def _check_external_deps(self, code: str) -> Tuple[float, List[Issue], Dict[str, Any]]:
        issues: List[Issue] = []
        details: Dict[str, Any] = {"requires_docker": False, "requires_network": False, "requires_filesystem": False}

        score = 3.0
        cl = code.lower()

        if "docker" in cl or "dockeroperator" in cl:
            details["requires_docker"] = True
        if "http" in cl or "requests" in cl:
            details["requires_network"] = True
        if "/app/data" in code or "host_data_dir" in cl:
            details["requires_filesystem"] = True

        return score, issues, details