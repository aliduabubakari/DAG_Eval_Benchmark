"""
Prefect-specific platform compliance tester gate-based, penalty-free scoring (issues logged).
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

# Check if Prefect is available
try:
    import prefect
    PREFECT_AVAILABLE = True
    PREFECT_VERSION = prefect.__version__
except ImportError:
    PREFECT_AVAILABLE = False
    PREFECT_VERSION = None


class PrefectComplianceTester(BasePlatformComplianceTester):
    """Prefect-specific compliance testing with weighted penalties."""
    
    ORCHESTRATOR = Orchestrator.PREFECT
    
    def _check_minimum_structure(self, code: str) -> bool:
        """Check if code has minimum Prefect structure."""
        # Must have @flow decorator
        has_flow = "@flow" in code
        # Should have either @task or function definitions
        has_tasks = "@task" in code or "def " in code
        
        return has_flow and has_tasks
    
    # ═══════════════════════════════════════════════════════════════════════
    # LOADABILITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_platform_load(
        self, 
        code: str, 
        file_path: Path
    ) -> Tuple[float, List[Issue], Dict]:
        """Check if Prefect can load the flow."""
        issues = []
        details = {
            "prefect_available": PREFECT_AVAILABLE,
            "prefect_version": PREFECT_VERSION,
            "module_loadable": False,
            "flows_found": [],
            "tasks_found": [],
        }
        
        if not PREFECT_AVAILABLE:
            return 2.0, [Issue(
                severity=Severity.INFO,
                category="platform",
                message="Prefect not installed - cannot verify load",
            )], details
        
        # Write to temp file and try to import
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            spec = importlib.util.spec_from_file_location("prefect_test_module", temp_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["prefect_test_module"] = module
                
                try:
                    spec.loader.exec_module(module)
                    details["module_loadable"] = True
                    
                    # Find flows and tasks
                    from prefect import Flow, Task
                    
                    for attr_name in dir(module):
                        if attr_name.startswith('_'):
                            continue
                        attr = getattr(module, attr_name, None)
                        if attr is None:
                            continue
                        
                        # Check for flow
                        if hasattr(attr, '__wrapped__') or callable(attr):
                            # Check if it's decorated with @flow
                            if hasattr(attr, 'fn') or 'flow' in str(type(attr)).lower():
                                details["flows_found"].append(attr_name)
                            elif hasattr(attr, 'is_task') or 'task' in str(type(attr)).lower():
                                details["tasks_found"].append(attr_name)
                    
                    if details["flows_found"]:
                        return 4.0, issues, details
                    else:
                        issues.append(Issue(
                            severity=Severity.MAJOR,
                            category="platform",
                            message="Module loaded but no flows detected",
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
                    if "prefect_test_module" in sys.modules:
                        del sys.modules["prefect_test_module"]
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
        """Check for required Prefect constructs."""
        issues = []
        details = {
            "has_flow": False,
            "has_tasks": False,
            "flow_count": 0,
            "task_count": 0,
        }
        
        score = 0.0
        
        # Check for @flow decorator
        flow_count = code.count("@flow")
        if flow_count > 0:
            details["has_flow"] = True
            details["flow_count"] = flow_count
            score += 2.0
        else:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="structure",
                message="No @flow decorator found",
            ))
        
        # Check for @task decorator
        task_count = code.count("@task")
        if task_count > 0:
            details["has_tasks"] = True
            details["task_count"] = task_count
            score += 1.5
        else:
            # Prefect allows flows without explicit @task
            issues.append(Issue(
                severity=Severity.MINOR,
                category="structure",
                message="No @task decorators found (OK if using inline logic)",
            ))
            score += 0.5
        
        # Check for proper imports
        if "from prefect import" in code or "import prefect" in code:
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
        """Check Prefect schedule configuration."""
        issues = []
        details = {
            "has_schedule": False,
            "has_deployment": False,
            "schedule_type": None,
        }
        
        score = 0.5  # Base score
        
        # Check for schedule patterns
        schedule_patterns = [
            "CronSchedule", "IntervalSchedule", "RRuleSchedule",
            "schedule=", "cron=", "interval=",
        ]
        
        for pattern in schedule_patterns:
            if pattern in code:
                details["has_schedule"] = True
                details["schedule_type"] = pattern.replace("=", "")
                score += 1.0
                break
        
        # Check for deployment configuration
        if "Deployment" in code or "deployment" in code.lower():
            details["has_deployment"] = True
            score += 0.5
        
        # Check for work pool configuration
        if "work_pool" in code or "work_queue" in code:
            score += 0.5
        
        return min(2.5, score), issues, details
    
    def _check_default_args(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Prefect default configuration."""
        issues = []
        details = {
            "has_task_runner": False,
            "has_log_prints": False,
            "has_result_storage": False,
        }
        
        score = 0.5  # Base score
        
        # Check for task runner configuration
        if "task_runner=" in code or "TaskRunner" in code:
            details["has_task_runner"] = True
            score += 0.75
        
        # Check for log_prints
        if "log_prints=" in code:
            details["has_log_prints"] = True
            score += 0.5
        
        # Check for result storage
        if "persist_result" in code or "result_storage" in code:
            details["has_result_storage"] = True
            score += 0.75
        
        return min(2.5, score), issues, details
    
    # ═══════════════════════════════════════════════════════════════════════
    # TASK VALIDITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_task_definitions(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Prefect task definitions."""
        issues = []
        details = {
            "task_count": 0,
            "flow_count": 0,
            "has_docstrings": False,
            "has_names": False,
        }
        
        score = 0.0
        
        # Count tasks and flows
        task_count = code.count("@task")
        flow_count = code.count("@flow")
        
        details["task_count"] = task_count
        details["flow_count"] = flow_count
        
        if flow_count > 0:
            score += 2.0
        else:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="task",
                message="No @flow decorators found",
            ))
            return 0.0, issues, details
        
        if task_count > 0:
            score += 1.0
        
        # Check for task names
        if 'name="' in code or "name='" in code:
            details["has_names"] = True
            score += 0.5
        
        # Check for docstrings
        if '"""' in code or "'''" in code:
            details["has_docstrings"] = True
            score += 0.5
        
        return min(4.0, score), issues, details
    
    def _check_operator_usage(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Prefect decorator usage."""
        issues = []
        details = {
            "uses_flow_decorator": False,
            "uses_task_decorator": False,
            "has_proper_imports": False,
        }
        
        score = 0.0
        
        # Check imports
        if "from prefect import" in code:
            details["has_proper_imports"] = True
            score += 1.0
        
        # Check decorators
        if "@flow" in code:
            details["uses_flow_decorator"] = True
            score += 1.0
        
        if "@task" in code:
            details["uses_task_decorator"] = True
            score += 0.5
        
        # Check for proper decorator configuration
        if "@flow(" in code or "@task(" in code:
            score += 0.5  # Configured decorators
        
        return min(3.0, score), issues, details
    
    # ═══════════════════════════════════════════════════════════════════════
    # EXECUTABILITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_dryrun_capability(
        self, 
        code: str, 
        file_path: Path
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Prefect dry-run/validation capability."""
        issues = []
        details = {
            "prefect_available": PREFECT_AVAILABLE,
            "can_validate": False,
        }
        
        if not PREFECT_AVAILABLE:
            return 2.0, [Issue(
                severity=Severity.INFO,
                category="executability",
                message="Prefect not installed - cannot validate",
            )], details
        
        # Check if flow can be called (basic validation)
        if "@flow" in code and "def " in code:
            details["can_validate"] = True
            return 4.0, issues, details
        
        return 2.0, issues, details
    
    def _extract_task_ids(self, code: str) -> set:
        """Extract task names from Prefect code."""
        task_ids = set()
        
        # Pattern: @task decorated functions
        pattern = r"@task[^)]*\)?\s*\ndef\s+(\w+)"
        task_ids.update(re.findall(pattern, code, re.MULTILINE))
        
        # Pattern: task name parameter
        pattern2 = r'@task\s*\([^)]*name\s*=\s*["\']([^"\']+)["\']'
        task_ids.update(re.findall(pattern2, code))
        
        # Pattern: @flow decorated functions
        pattern3 = r"@flow[^)]*\)?\s*\ndef\s+(\w+)"
        task_ids.update(re.findall(pattern3, code, re.MULTILINE))
        
        return task_ids