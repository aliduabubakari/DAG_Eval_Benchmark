"""
Enhanced base platform compliance tester gate-based, penalty-free scoring (issues logged).
Focuses on structural/functional correctness that can objectively differentiate approaches.
"""

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import ast
import importlib.util
import os
import re
import sys
import tempfile

from evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationScore,
    GateCheckResult,
    Issue,
    Severity,
    Orchestrator,
    DEFAULT_PENALTIES,
)


class BasePlatformComplianceTester(BaseEvaluator):
    """
    Base platform compliance tester with weighted penalties.
    
    Dimensions (all scored 0-10, with penalties applied):
    1. Loadability: Can the code be loaded by Python and the orchestrator?
    2. Structure Validity: Does it have valid orchestrator structure?
    3. Configuration Validity: Are configurations properly set?
    4. Task Validity: Are tasks/ops properly defined?
    5. Executability: Can it actually run (dry-run capability)?
    
    Penalty System:
    - CRITICAL: Automatic gate failure (score = 0 for dimension)
    - MAJOR: -2.0 points per issue
    - MINOR: -0.5 points per issue
    - INFO: No penalty (informational only)
    """
    
    EVALUATION_TYPE = "platform_compliance"
    ORCHESTRATOR: Orchestrator = Orchestrator.UNKNOWN
    
    # Penalty configuration (can be overridden)
    PENALTY_VALUES = {
        Severity.CRITICAL: 0.0,  # Gate failure - handled separately
        Severity.MAJOR: 2.0,
        Severity.MINOR: 0.5,
        Severity.INFO: 0.0,
    }
    
    # Maximum penalties per dimension (prevents negative scores)
    MAX_PENALTY_PER_DIMENSION = 8.0
    
    def evaluate(self, file_path: Path) -> EvaluationResult:
        """Run complete platform compliance evaluation."""
        self.logger.info(f"Running platform compliance for {self.ORCHESTRATOR.value}: {file_path}")
        
        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return self._create_error_result(file_path, f"Failed to read file: {e}")
        
        # Check for empty file
        if not code.strip():
            return self._create_error_result(file_path, "File is empty")
        
        # Parse AST
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
            }
        )
        
        # Run gate checks first
        gate_results = self._run_gate_checks(code, tree, file_path)
        result.gate_checks = gate_results
        
        # If critical gates fail, return early with zero scores
        if not result.gates_passed:
            self._apply_zero_scores(result)
            return result
        
        # Run all 5 dimension evaluations
        result.scores["loadability"] = self._evaluate_loadability(code, tree, file_path)
        result.scores["structure_validity"] = self._evaluate_structure_validity(code, tree)
        result.scores["configuration_validity"] = self._evaluate_configuration_validity(code, tree)
        result.scores["task_validity"] = self._evaluate_task_validity(code, tree)
        result.scores["executability"] = self._evaluate_executability(code, tree, file_path)
        
        # Apply penalties to all scores
        for score in result.scores.values():
            self._apply_penalties_to_score(score)
        
        return result
    
    def _create_error_result(self, file_path: Path, error: str) -> EvaluationResult:
        """Create result for file read error."""
        result = EvaluationResult(
            evaluation_type=self.EVALUATION_TYPE,
            file_path=str(file_path),
            orchestrator=self.ORCHESTRATOR,
            timestamp=datetime.now().isoformat(),
        )
        result.gate_checks.append(GateCheckResult(
            name="file_readable",
            passed=False,
            message=error,
            is_critical=True,
        ))
        self._apply_zero_scores(result)
        return result
    
    def _create_syntax_error_result(self, file_path: Path, error: SyntaxError) -> EvaluationResult:
        """Create result for syntax error - this is a CRITICAL failure."""
        result = EvaluationResult(
            evaluation_type=self.EVALUATION_TYPE,
            file_path=str(file_path),
            orchestrator=self.ORCHESTRATOR,
            timestamp=datetime.now().isoformat(),
        )
        result.gate_checks.append(GateCheckResult(
            name="syntax_valid",
            passed=False,
            message=f"Syntax error at line {error.lineno}: {error.msg}",
            is_critical=True,
        ))
        self._apply_zero_scores(result)
        return result
    
    def _apply_zero_scores(self, result: EvaluationResult):
        """Apply zero scores to all dimensions."""
        dimensions = [
            "loadability", "structure_validity", "configuration_validity",
            "task_validity", "executability"
        ]
        for dim in dimensions:
            result.scores[dim] = EvaluationScore(
                name=dim,
                raw_score=0.0,
                weight=self.get_weight(dim),
            )
    
    def _apply_penalties_to_score(self, score: EvaluationScore):
        """Apply penalties from issues to a score."""
        total_penalty = 0.0
        
        for issue in score.issues:
            if issue.severity == Severity.CRITICAL:
                # Critical issues in a dimension = dimension score becomes 0
                score.raw_score = 0.0
                score.penalties_applied = score.raw_score
                return
            
            penalty = issue.penalty if issue.penalty is not None else self.PENALTY_VALUES.get(issue.severity, 0.0)
            total_penalty += penalty
        
        # Cap penalties
        total_penalty = min(total_penalty, self.MAX_PENALTY_PER_DIMENSION)
        score.penalties_applied = total_penalty
        score.raw_score = max(0.0, score.raw_score - total_penalty)
    
    # ═══════════════════════════════════════════════════════════════════════
    # GATE CHECKS (Must pass to continue)
    # ═══════════════════════════════════════════════════════════════════════
    def _run_gate_checks(
        self, 
        code: str, 
        tree: ast.AST, 
        file_path: Path
    ) -> List[GateCheckResult]:
        """Run critical gate checks."""
        gates = []
        
        # Gate 1: Syntax is valid (already passed if we have AST)
        gates.append(GateCheckResult(
            name="syntax_valid",
            passed=True,
            message="Code has valid Python syntax",
            is_critical=True,
        ))
        
        # Gate 2: Has minimum structure
        has_structure = self._check_minimum_structure(code)
        gates.append(GateCheckResult(
            name="minimum_structure",
            passed=has_structure,
            message="Has minimum pipeline structure" if has_structure else "Missing basic pipeline structure",
            is_critical=True,
        ))
        
        # Gate 3: No critical import errors
        import_ok, import_msg = self._check_critical_imports(code, tree)
        gates.append(GateCheckResult(
            name="critical_imports",
            passed=import_ok,
            message=import_msg,
            is_critical=True,
        ))
        
        return gates
    
    @abstractmethod
    def _check_minimum_structure(self, code: str) -> bool:
        """Check if code has minimum required structure. Override in subclasses."""
        pass
    
    def _check_critical_imports(self, code: str, tree: ast.AST) -> Tuple[bool, str]:
        """Check for critical import issues."""
        # Collect imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
        
        if not imports:
            return True, "No imports found (might be OK for simple scripts)"
        
        # Check for obviously broken imports
        broken_patterns = [
            "undefined_module",
            "nonexistent",
        ]
        
        for imp in imports:
            if any(p in imp.lower() for p in broken_patterns):
                return False, f"Broken import detected: {imp}"
        
        return True, f"Found {len(imports)} imports"
    
    # ═══════════════════════════════════════════════════════════════════════
    # DIMENSION 1: LOADABILITY (0-10)
    # ═══════════════════════════════════════════════════════════════════════
    def _evaluate_loadability(
        self, 
        code: str, 
        tree: ast.AST, 
        file_path: Path
    ) -> EvaluationScore:
        """
        Evaluate loadability - can the code be loaded?
        
        Sub-components:
        - Import resolution (0-3)
        - Platform-specific load (0-4)
        - Module structure (0-3)
        """
        self.logger.info("Evaluating loadability...")
        
        issues = []
        details = {}
        base_score = 0.0
        
        # 1. Import resolution (0-3)
        import_score, import_issues, import_details = self._check_import_resolution(code, tree)
        base_score += import_score
        issues.extend(import_issues)
        details["imports"] = import_details
        
        # 2. Platform-specific load test (0-4)
        platform_score, platform_issues, platform_details = self._check_platform_load(code, file_path)
        base_score += platform_score
        issues.extend(platform_issues)
        details["platform_load"] = platform_details
        
        # 3. Module structure (0-3)
        structure_score, structure_issues, structure_details = self._check_module_structure(code, tree)
        base_score += structure_score
        issues.extend(structure_issues)
        details["module_structure"] = structure_details
        
        return EvaluationScore(
            name="loadability",
            raw_score=min(10.0, base_score),
            weight=self.get_weight("loadability"),
            issues=issues,
            details=details,
        )
    
    def _check_import_resolution(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check if imports can potentially be resolved."""
        issues = []
        details = {"total_imports": 0, "standard_lib": 0, "third_party": 0}
        
        # Standard library modules (common ones)
        stdlib = {
            'os', 'sys', 'json', 'datetime', 'pathlib', 'typing', 'logging',
            'collections', 'itertools', 'functools', 'tempfile', 'subprocess',
            're', 'ast', 'time', 'copy', 'io', 'abc', 'dataclasses', 'enum',
        }
        
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        details["total_imports"] = len(imports)
        details["standard_lib"] = len(imports & stdlib)
        details["third_party"] = len(imports - stdlib)
        
        if not imports:
            return 2.0, issues, details
        
        # Check for star imports (bad practice)
        if "import *" in code:
            issues.append(Issue(
                severity=Severity.MINOR,
                category="imports",
                message="Star import detected - reduces code clarity",
            ))
        
        # Base score
        score = 3.0
        
        # Penalize if no orchestrator imports found
        orchestrator_imports = {'airflow', 'prefect', 'dagster'}
        if not (imports & orchestrator_imports):
            issues.append(Issue(
                severity=Severity.MAJOR,
                category="imports",
                message="No orchestrator imports found",
            ))
            score -= 1.0
        
        return max(0.0, score), issues, details
    
    @abstractmethod
    def _check_platform_load(
        self, 
        code: str, 
        file_path: Path
    ) -> Tuple[float, List[Issue], Dict]:
        """Check if orchestrator can load the code. Override in subclasses."""
        pass
    
    def _check_module_structure(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check module structure validity."""
        issues = []
        details = {
            "has_functions": False,
            "has_classes": False,
            "function_count": 0,
            "class_count": 0,
        }
        
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        
        details["has_functions"] = len(functions) > 0
        details["has_classes"] = len(classes) > 0
        details["function_count"] = len(functions)
        details["class_count"] = len(classes)
        
        score = 2.0  # Base score
        
        if functions:
            score += 0.5
        
        # Check for __name__ == "__main__" pattern (good practice)
        if '__name__' in code and '__main__' in code:
            score += 0.5
            details["has_main_guard"] = True
        
        return min(3.0, score), issues, details
    
    # ═══════════════════════════════════════════════════════════════════════
    # DIMENSION 2: STRUCTURE VALIDITY (0-10)
    # ═══════════════════════════════════════════════════════════════════════
    def _evaluate_structure_validity(
        self, 
        code: str, 
        tree: ast.AST
    ) -> EvaluationScore:
        """
        Evaluate structure validity - does it have valid orchestrator structure?
        
        Sub-components:
        - Required constructs (0-4)
        - Dependency graph (0-3)
        - No orphan tasks (0-3)
        """
        self.logger.info("Evaluating structure validity...")
        
        issues = []
        details = {}
        base_score = 0.0
        
        # 1. Required constructs (0-4)
        construct_score, construct_issues, construct_details = self._check_required_constructs(code, tree)
        base_score += construct_score
        issues.extend(construct_issues)
        details["constructs"] = construct_details
        
        # 2. Dependency graph validity (0-3)
        graph_score, graph_issues, graph_details = self._check_dependency_graph(code, tree)
        base_score += graph_score
        issues.extend(graph_issues)
        details["dependency_graph"] = graph_details
        
        # 3. No orphan tasks (0-3)
        orphan_score, orphan_issues, orphan_details = self._check_orphan_tasks(code, tree)
        base_score += orphan_score
        issues.extend(orphan_issues)
        details["orphan_check"] = orphan_details
        
        return EvaluationScore(
            name="structure_validity",
            raw_score=min(10.0, base_score),
            weight=self.get_weight("structure_validity"),
            issues=issues,
            details=details,
        )
    
    @abstractmethod
    def _check_required_constructs(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check for required orchestrator constructs. Override in subclasses."""
        pass
    
    def _check_dependency_graph(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check dependency graph validity."""
        issues = []
        details = {
            "has_dependencies": False,
            "dependency_count": 0,
        }
        
        # Count dependency patterns (generic)
        dep_patterns = [">>", "<<", "chain(", "set_downstream", "set_upstream"]
        dep_count = sum(code.count(p) for p in dep_patterns)
        
        details["dependency_count"] = dep_count
        details["has_dependencies"] = dep_count > 0
        
        task_count = self._count_tasks(code)
        
        if task_count <= 1:
            # Single task - no dependencies needed
            return 3.0, issues, details
        
        if dep_count == 0 and task_count > 1:
            issues.append(Issue(
                severity=Severity.MAJOR,
                category="structure",
                message=f"Multiple tasks ({task_count}) but no dependencies defined",
            ))
            return 1.0, issues, details
        
        return 3.0, issues, details
    
    def _check_orphan_tasks(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check for orphan (isolated) tasks."""
        issues = []
        details = {"task_count": 0, "orphan_count": 0}
        
        task_ids = self._extract_task_ids(code)
        details["task_count"] = len(task_ids)
        
        if len(task_ids) <= 1:
            return 3.0, issues, details
        
        # Find tasks mentioned in dependencies
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
                issues.append(Issue(
                    severity=Severity.MAJOR,
                    category="structure",
                    message=f"{orphan_count} orphan tasks detected (>50% of tasks)",
                ))
                return 1.0, issues, details
            elif orphan_count > 0:
                issues.append(Issue(
                    severity=Severity.MINOR,
                    category="structure",
                    message=f"{orphan_count} orphan tasks detected",
                ))
                return 2.0, issues, details
        
        return 3.0, issues, details
    
    def _extract_task_ids(self, code: str) -> Set[str]:
        """Extract task IDs from code. Override for better detection."""
        task_ids = set()
        
        # Generic patterns
        patterns = [
            r"task_id\s*=\s*['\"]([^'\"]+)['\"]",
            r"(\w+)\s*=\s*\w+Operator\s*\(",
            r"name\s*=\s*['\"]([^'\"]+)['\"]",
        ]
        
        for pattern in patterns:
            task_ids.update(re.findall(pattern, code))
        
        return task_ids
    
    def _count_tasks(self, code: str) -> int:
        """Count number of tasks in code."""
        return len(self._extract_task_ids(code))
    
    # ═══════════════════════════════════════════════════════════════════════
    # DIMENSION 3: CONFIGURATION VALIDITY (0-10)
    # ═══════════════════════════════════════════════════════════════════════
    def _evaluate_configuration_validity(
        self, 
        code: str, 
        tree: ast.AST
    ) -> EvaluationScore:
        """
        Evaluate configuration validity.
        
        Sub-components:
        - Schedule configuration (0-2.5)
        - Default args/parameters (0-2.5)
        - Connection references (0-2.5)
        - Security (no hardcoded secrets) (0-2.5)
        """
        self.logger.info("Evaluating configuration validity...")
        
        issues = []
        details = {}
        base_score = 0.0
        
        # 1. Schedule configuration (0-2.5)
        schedule_score, schedule_issues, schedule_details = self._check_schedule_config(code, tree)
        base_score += schedule_score
        issues.extend(schedule_issues)
        details["schedule"] = schedule_details
        
        # 2. Default args/parameters (0-2.5)
        args_score, args_issues, args_details = self._check_default_args(code, tree)
        base_score += args_score
        issues.extend(args_issues)
        details["default_args"] = args_details
        
        # 3. Connection references (0-2.5)
        conn_score, conn_issues, conn_details = self._check_connection_config(code, tree)
        base_score += conn_score
        issues.extend(conn_issues)
        details["connections"] = conn_details
        
        # 4. Security - no hardcoded secrets (0-2.5)
        security_score, security_issues, security_details = self._check_security_config(code)
        base_score += security_score
        issues.extend(security_issues)
        details["security"] = security_details
        
        return EvaluationScore(
            name="configuration_validity",
            raw_score=min(10.0, base_score),
            weight=self.get_weight("configuration_validity"),
            issues=issues,
            details=details,
        )
    
    @abstractmethod
    def _check_schedule_config(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check schedule configuration. Override in subclasses."""
        pass
    
    @abstractmethod
    def _check_default_args(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check default arguments. Override in subclasses."""
        pass
    
    def _check_connection_config(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check connection configuration."""
        issues = []
        details = {"connection_count": 0, "has_connections": False}
        
        # Find connection references
        conn_patterns = [
            r"conn_id\s*=\s*['\"]([^'\"]+)['\"]",
            r"connection_id\s*=\s*['\"]([^'\"]+)['\"]",
        ]
        
        connections = set()
        for pattern in conn_patterns:
            connections.update(re.findall(pattern, code))
        
        details["connection_count"] = len(connections)
        details["has_connections"] = len(connections) > 0
        
        # Check for hardcoded connection strings (bad)
        hardcoded_patterns = [
            r"postgresql://\w+:\w+@",
            r"mysql://\w+:\w+@",
            r"mongodb://\w+:\w+@",
            r"redis://:\w+@",
        ]
        
        for pattern in hardcoded_patterns:
            if re.search(pattern, code):
                issues.append(Issue(
                    severity=Severity.MAJOR,
                    category="configuration",
                    message="Hardcoded database connection string detected",
                ))
                return 0.5, issues, details
        
        # Score based on connection usage
        if connections:
            return 2.5, issues, details
        else:
            return 1.5, issues, details  # No connections might be OK
    
    def _check_security_config(self, code: str) -> Tuple[float, List[Issue], Dict]:
        """Check for security issues in configuration."""
        issues = []
        details = {"hardcoded_secrets": False, "uses_env_vars": False}
        
        # Check for environment variable usage (good)
        if "os.getenv" in code or "os.environ" in code:
            details["uses_env_vars"] = True
        
        # Check for hardcoded secrets (bad)
        secret_patterns = [
            (r"password\s*=\s*['\"][^'\"]{4,}['\"]", "password"),
            (r"api_key\s*=\s*['\"][^'\"]{8,}['\"]", "api_key"),
            (r"token\s*=\s*['\"][^'\"]{8,}['\"]", "token"),
            (r"secret\s*=\s*['\"][^'\"]{4,}['\"]", "secret"),
            (r"AWS_SECRET\w*\s*=\s*['\"][^'\"]+['\"]", "AWS secret"),
        ]
        
        # Exclude patterns that are clearly placeholders
        placeholder_indicators = [
            "{{", "}}", "your_", "placeholder", "xxx", "***", "<", ">",
            "example", "changeme", "fixme", "todo", "none", "null",
        ]
        
        for pattern, secret_type in secret_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                # Check if it's a placeholder
                match_lower = match.lower() if isinstance(match, str) else str(match).lower()
                if not any(p in match_lower for p in placeholder_indicators):
                    issues.append(Issue(
                        severity=Severity.MAJOR,
                        category="security",
                        message=f"Potential hardcoded {secret_type} detected",
                    ))
                    details["hardcoded_secrets"] = True
                    return 0.5, issues, details
        
        # Good security practices
        score = 2.0
        if details["uses_env_vars"]:
            score += 0.5
        
        return min(2.5, score), issues, details
    
    # ═══════════════════════════════════════════════════════════════════════
    # DIMENSION 4: TASK VALIDITY (0-10)
    # ═══════════════════════════════════════════════════════════════════════
    def _evaluate_task_validity(
        self, 
        code: str, 
        tree: ast.AST
    ) -> EvaluationScore:
        """
        Evaluate task validity.
        
        Sub-components:
        - Task definitions complete (0-4)
        - Operator/decorator usage (0-3)
        - Task parameters (0-3)
        """
        self.logger.info("Evaluating task validity...")
        
        issues = []
        details = {}
        base_score = 0.0
        
        # 1. Task definitions (0-4)
        def_score, def_issues, def_details = self._check_task_definitions(code, tree)
        base_score += def_score
        issues.extend(def_issues)
        details["definitions"] = def_details
        
        # 2. Operator usage (0-3)
        op_score, op_issues, op_details = self._check_operator_usage(code, tree)
        base_score += op_score
        issues.extend(op_issues)
        details["operator_usage"] = op_details
        
        # 3. Task parameters (0-3)
        param_score, param_issues, param_details = self._check_task_parameters(code, tree)
        base_score += param_score
        issues.extend(param_issues)
        details["parameters"] = param_details
        
        return EvaluationScore(
            name="task_validity",
            raw_score=min(10.0, base_score),
            weight=self.get_weight("task_validity"),
            issues=issues,
            details=details,
        )
    
    @abstractmethod
    def _check_task_definitions(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check task definitions. Override in subclasses."""
        pass
    
    @abstractmethod
    def _check_operator_usage(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check operator/decorator usage. Override in subclasses."""
        pass
    
    def _check_task_parameters(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check task parameter configuration."""
        issues = []
        details = {
            "has_task_ids": False,
            "has_retries": False,
            "has_timeouts": False,
        }
        
        score = 1.0  # Base score
        
        # Check for task identifiers
        if "task_id=" in code or "name=" in code:
            details["has_task_ids"] = True
            score += 1.0
        else:
            issues.append(Issue(
                severity=Severity.MINOR,
                category="task",
                message="No explicit task identifiers found",
            ))
        
        # Check for retry configuration
        if "retries=" in code or "retry" in code.lower():
            details["has_retries"] = True
            score += 0.5
        
        # Check for timeout configuration
        if "timeout" in code.lower() or "execution_timeout" in code:
            details["has_timeouts"] = True
            score += 0.5
        
        return min(3.0, score), issues, details
    
    # ═══════════════════════════════════════════════════════════════════════
    # DIMENSION 5: EXECUTABILITY (0-10)
    # ═══════════════════════════════════════════════════════════════════════
    def _evaluate_executability(
        self, 
        code: str, 
        tree: ast.AST,
        file_path: Path
    ) -> EvaluationScore:
        """
        Evaluate executability.
        
        Sub-components:
        - Dry-run capability (0-4)
        - Runtime error potential (0-3)
        - External dependencies (0-3)
        """
        self.logger.info("Evaluating executability...")
        
        issues = []
        details = {}
        base_score = 0.0
        
        # 1. Dry-run capability (0-4)
        dryrun_score, dryrun_issues, dryrun_details = self._check_dryrun_capability(code, file_path)
        base_score += dryrun_score
        issues.extend(dryrun_issues)
        details["dryrun"] = dryrun_details
        
        # 2. Runtime error potential (0-3)
        runtime_score, runtime_issues, runtime_details = self._check_runtime_safety(code, tree)
        base_score += runtime_score
        issues.extend(runtime_issues)
        details["runtime_safety"] = runtime_details
        
        # 3. External dependencies (0-3)
        ext_score, ext_issues, ext_details = self._check_external_deps(code)
        base_score += ext_score
        issues.extend(ext_issues)
        details["external_deps"] = ext_details
        
        return EvaluationScore(
            name="executability",
            raw_score=min(10.0, base_score),
            weight=self.get_weight("executability"),
            issues=issues,
            details=details,
        )
    
    @abstractmethod
    def _check_dryrun_capability(
        self, 
        code: str, 
        file_path: Path
    ) -> Tuple[float, List[Issue], Dict]:
        """Check dry-run capability. Override in subclasses."""
        pass
    
    def _check_runtime_safety(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check for potential runtime errors."""
        issues = []
        details = {"potential_issues": []}
        
        score = 3.0
        
        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(Issue(
                        severity=Severity.MINOR,
                        category="runtime",
                        message="Bare 'except:' clause found - may hide errors",
                        line=node.lineno,
                    ))
                    score -= 0.25
        
        # Check for obvious issues
        if "/ 0" in code or "/0" in code:
            issues.append(Issue(
                severity=Severity.MAJOR,
                category="runtime",
                message="Potential division by zero",
            ))
            score -= 1.0
        
        return max(0.0, score), issues, details
    
    def _check_external_deps(self, code: str) -> Tuple[float, List[Issue], Dict]:
        """Check external dependency requirements."""
        issues = []
        details = {
            "requires_docker": False,
            "requires_network": False,
            "requires_filesystem": False,
        }
        
        score = 3.0
        
        # Check for Docker
        if "docker" in code.lower() or "DockerOperator" in code:
            details["requires_docker"] = True
        
        # Check for network/HTTP
        if "http" in code.lower() or "requests" in code:
            details["requires_network"] = True
        
        # Check for filesystem
        if "/app/data" in code or "HOST_DATA_DIR" in code:
            details["requires_filesystem"] = True
        
        # These are informational, not penalties
        return score, issues, details