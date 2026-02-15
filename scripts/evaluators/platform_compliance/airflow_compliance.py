"""
Airflow-specific platform compliance tester gate-based, penalty-free scoring (issues logged).
"""

import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluators.base_evaluator import (
    EvaluationScore,
    Issue,
    Severity,
    Orchestrator,
)
from .pct_base import BasePlatformComplianceTester

# --- Robust Airflow detection (prevents import-time crashes) ---
from importlib import metadata as importlib_metadata

AIRFLOW_AVAILABLE = False
AIRFLOW_VERSION = None

try:
    import airflow as airflow_mod  # noqa: F401

    # Prefer importlib.metadata for version (reliable for distributions)
    try:
        AIRFLOW_VERSION = importlib_metadata.version("apache-airflow")
    except importlib_metadata.PackageNotFoundError:
        AIRFLOW_VERSION = getattr(airflow_mod, "__version__", None)

    # If we still don't have a version, this is likely NOT Apache Airflow
    if AIRFLOW_VERSION:
        AIRFLOW_AVAILABLE = True
    else:
        AIRFLOW_AVAILABLE = False

except Exception:
    AIRFLOW_AVAILABLE = False
    AIRFLOW_VERSION = None
# -------------------------------------------------------------

class AirflowComplianceTester(BasePlatformComplianceTester):
    """Airflow-specific compliance testing with weighted penalties."""
    
    ORCHESTRATOR = Orchestrator.AIRFLOW
    
    def _check_minimum_structure(self, code: str) -> bool:
        """Check if code has minimum Airflow structure."""
        # Must have either DAG context or @dag decorator
        has_dag_context = "with DAG(" in code or "with dag(" in code
        has_dag_decorator = "@dag" in code
        
        # Should have task definitions (operators or @task)
        has_operators = "Operator(" in code
        has_task_decorator = "@task" in code
        
        return (has_dag_context or has_dag_decorator) and (has_operators or has_task_decorator)
    
    # ═══════════════════════════════════════════════════════════════════════
    # LOADABILITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_platform_load(
        self, 
        code: str, 
        file_path: Path
    ) -> Tuple[float, List[Issue], Dict]:
        """Check if Airflow can load the DAG."""
        issues = []
        details = {
            "airflow_available": AIRFLOW_AVAILABLE,
            "airflow_version": AIRFLOW_VERSION,
            "module_loadable": False,
            "dags_found": [],
            "tasks_found": [],
        }
        
        if not AIRFLOW_AVAILABLE:
            return 2.0, [Issue(
                severity=Severity.INFO,
                category="platform",
                message="Airflow not installed - cannot verify load",
            )], details
        
        # Write to temp file and try to import
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            spec = importlib.util.spec_from_file_location("airflow_test_module", temp_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["airflow_test_module"] = module
                
                try:
                    spec.loader.exec_module(module)
                    details["module_loadable"] = True
                    
                    # Find DAGs
                    from airflow.models import DAG
                    from airflow.decorators import dag as dag_decorator
                    
                    for attr_name in dir(module):
                        if attr_name.startswith('_'):
                            continue
                        attr = getattr(module, attr_name, None)
                        if attr is None:
                            continue
                        
                        # Check for DAG instances
                        if isinstance(attr, DAG):
                            details["dags_found"].append(attr_name)
                        # Check for TaskFlow DAG functions
                        elif callable(attr) and hasattr(attr, '__wrapped__'):
                            details["dags_found"].append(attr_name)
                    
                    # Count tasks (approximate)
                    task_count = code.count("Operator(") + code.count("@task")
                    details["tasks_found"] = [f"task_{i}" for i in range(task_count)]
                    
                    if details["dags_found"]:
                        return 4.0, issues, details
                    else:
                        issues.append(Issue(
                            severity=Severity.MAJOR,
                            category="platform",
                            message="Module loaded but no DAGs detected",
                        ))
                        return 2.0, issues, details
                
                except ModuleNotFoundError as e:
                    issues.append(Issue(
                        severity=Severity.MAJOR,
                        category="platform",
                        message=f"Missing module: {e.name}",
                    ))
                    return 1.0, issues, details
                except Exception as e:
                    issues.append(Issue(
                        severity=Severity.MAJOR,
                        category="platform",
                        message=f"Load error: {str(e)[:100]}",
                    ))
                    return 1.0, issues, details
                finally:
                    if "airflow_test_module" in sys.modules:
                        del sys.modules["airflow_test_module"]
            else:
                issues.append(Issue(
                    severity=Severity.CRITICAL,
                    category="platform",
                    message="Could not create module spec",
                ))
                return 0.0, issues, details
        
        except Exception as e:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="platform",
                message=f"Exception: {str(e)[:100]}",
            ))
            return 0.0, issues, details
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURE VALIDITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_required_constructs(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check for required Airflow constructs."""
        issues = []
        details = {
            "has_dag_definition": False,
            "has_tasks": False,
            "dag_count": 0,
            "task_count": 0,
            "uses_context_manager": False,
            "uses_taskflow": False,
        }
        
        score = 0.0
        
        # Check for DAG definition (context manager or decorator)
        has_dag_context = "with DAG(" in code or "with dag(" in code
        has_dag_decorator = "@dag" in code
        
        if has_dag_context:
            details["has_dag_definition"] = True
            details["uses_context_manager"] = True
            details["dag_count"] = code.count("with DAG(") + code.count("with dag(")
            score += 2.0
        elif has_dag_decorator:
            details["has_dag_definition"] = True
            details["uses_taskflow"] = True
            details["dag_count"] = code.count("@dag")
            score += 2.0
        else:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="structure",
                message="No DAG definition found (neither 'with DAG()' nor '@dag')",
            ))
        
        # Check for task definitions
        operator_count = len(re.findall(r'\w+Operator\s*\(', code))
        task_decorator_count = code.count("@task")
        
        total_tasks = operator_count + task_decorator_count
        details["task_count"] = total_tasks
        
        if total_tasks > 0:
            details["has_tasks"] = True
            score += 1.5
            
            if task_decorator_count > 0:
                details["uses_taskflow"] = True
        else:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="structure",
                message="No tasks found (no operators or @task decorators)",
            ))
        
        # Check for proper imports
        if "from airflow" in code or "import airflow" in code:
            score += 0.5
        
        return min(4.0, score), issues, details
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION VALIDITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_schedule_config(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Airflow schedule configuration."""
        issues = []
        details = {
            "has_schedule": False,
            "schedule_type": None,
            "has_start_date": False,
            "has_catchup": False,
        }
        
        score = 0.5  # Base score
        
        # Check for schedule_interval
        if "schedule_interval=" in code or "schedule=" in code:
            details["has_schedule"] = True
            
            # Determine schedule type
            if "@daily" in code or "@hourly" in code or "@weekly" in code:
                details["schedule_type"] = "preset"
            elif "timedelta" in code:
                details["schedule_type"] = "timedelta"
            elif re.search(r'["\'][@\d\s\*\-,/]+["\']', code):
                details["schedule_type"] = "cron"
            else:
                details["schedule_type"] = "custom"
            
            score += 1.0
        
        # Check for start_date
        if "start_date=" in code:
            details["has_start_date"] = True
            score += 0.5
        else:
            issues.append(Issue(
                severity=Severity.MINOR,
                category="configuration",
                message="No start_date configured",
            ))
        
        # Check for catchup
        if "catchup=" in code:
            details["has_catchup"] = True
            score += 0.5
        
        return min(2.5, score), issues, details
    
    def _check_default_args(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Airflow default_args configuration."""
        issues = []
        details = {
            "has_default_args": False,
            "has_owner": False,
            "has_retries": False,
            "has_retry_delay": False,
            "has_email_config": False,
        }
        
        score = 0.5  # Base score
        
        # Check for default_args
        if "default_args" in code:
            details["has_default_args"] = True
            score += 0.5
            
            # Check common default_args keys
            if "'owner'" in code or '"owner"' in code:
                details["has_owner"] = True
                score += 0.5
            
            if "'retries'" in code or '"retries"' in code:
                details["has_retries"] = True
                score += 0.5
            
            if "'retry_delay'" in code or '"retry_delay"' in code:
                details["has_retry_delay"] = True
                score += 0.25
            
            if "email" in code.lower():
                details["has_email_config"] = True
                score += 0.25
        else:
            issues.append(Issue(
                severity=Severity.MINOR,
                category="configuration",
                message="No default_args defined",
            ))
        
        return min(2.5, score), issues, details
    
    # ═══════════════════════════════════════════════════════════════════════
    # TASK VALIDITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_task_definitions(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Airflow task definitions."""
        issues = []
        details = {
            "operator_count": 0,
            "taskflow_count": 0,
            "has_task_ids": False,
            "has_python_callable": False,
        }
        
        score = 0.0
        
        # Count operators
        operator_count = len(re.findall(r'\w+Operator\s*\(', code))
        taskflow_count = code.count("@task")
        
        details["operator_count"] = operator_count
        details["taskflow_count"] = taskflow_count
        
        total_tasks = operator_count + taskflow_count
        
        if total_tasks == 0:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="task",
                message="No tasks defined",
            ))
            return 0.0, issues, details
        
        # Give credit for having tasks
        score += min(2.0, total_tasks * 0.5)
        
        # Check for task_id
        task_id_count = code.count("task_id=")
        if task_id_count > 0:
            details["has_task_ids"] = True
            score += 1.0
        else:
            issues.append(Issue(
                severity=Severity.MINOR,
                category="task",
                message="No explicit task_id parameters found",
            ))
        
        # Check for python_callable (for PythonOperator)
        if "python_callable=" in code:
            details["has_python_callable"] = True
            score += 0.5
        
        # Check for proper task naming
        if taskflow_count > 0 and "def " in code:
            score += 0.5
        
        return min(4.0, score), issues, details
    
    def _check_operator_usage(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Airflow operator usage."""
        issues = []
        details = {
            "operator_types": [],
            "has_proper_imports": False,
            "uses_standard_operators": False,
        }
        
        score = 0.0
        
        # Find operator types
        operator_types = re.findall(r'(\w+Operator)\s*\(', code)
        details["operator_types"] = list(set(operator_types))
        
        if operator_types:
            score += 1.0
            
            # Check for standard operators
            standard_ops = {
                'PythonOperator', 'BashOperator', 'DummyOperator', 'EmptyOperator',
                'DockerOperator', 'BranchPythonOperator', 'ShortCircuitOperator'
            }
            if any(op in standard_ops for op in operator_types):
                details["uses_standard_operators"] = True
                score += 0.5
        
        # Check for proper imports
        import_patterns = [
            "from airflow.operators",
            "from airflow.providers",
            "from airflow.sensors",
        ]
        
        if any(p in code for p in import_patterns):
            details["has_proper_imports"] = True
            score += 1.0
        
        # Check for TaskFlow API usage
        if "@task" in code and "from airflow.decorators import task" in code:
            score += 0.5
        
        return min(3.0, score), issues, details
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXECUTABILITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_dryrun_capability(
        self, 
        code: str, 
        file_path: Path
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Airflow dry-run/test capability."""
        issues = []
        details = {
            "airflow_available": AIRFLOW_AVAILABLE,
            "can_test": False,
            "has_test_mode": False,
        }
        
        if not AIRFLOW_AVAILABLE:
            return 2.0, [Issue(
                severity=Severity.INFO,
                category="executability",
                message="Airflow not installed - cannot test execution",
            )], details
        
        # Check for test mode patterns
        test_patterns = [
            "test_mode=True",
            "is_paused_upon_creation=True",
            "dag.test()",
        ]
        
        if any(p in code for p in test_patterns):
            details["has_test_mode"] = True
            details["can_test"] = True
            return 4.0, issues, details
        
        # Check if DAG structure is complete enough to test
        has_dag = "with DAG(" in code or "@dag" in code
        has_tasks = "Operator(" in code or "@task" in code
        has_dependencies = ">>" in code or "<<" in code or "set_downstream" in code
        
        if has_dag and has_tasks:
            details["can_test"] = True
            score = 3.0
            if has_dependencies:
                score += 0.5
            return min(4.0, score), issues, details
        
        return 2.0, issues, details
    
    def _extract_task_ids(self, code: str) -> set:
        """Extract task IDs from Airflow code."""
        task_ids = set()
        
        # Pattern 1: task_id parameter
        pattern1 = r"task_id\s*=\s*['\"]([^'\"]+)['\"]"
        task_ids.update(re.findall(pattern1, code))
        
        # Pattern 2: variable assignment to operator
        pattern2 = r"(\w+)\s*=\s*\w+Operator\s*\("
        task_ids.update(re.findall(pattern2, code))
        
        # Pattern 3: @task decorated functions
        pattern3 = r"@task[^)]*\)?\s*\ndef\s+(\w+)"
        task_ids.update(re.findall(pattern3, code, re.MULTILINE))
        
        return task_ids