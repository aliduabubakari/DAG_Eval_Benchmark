"""
Dagster-specific platform compliance tester gate-based, penalty-free scoring (issues logged).
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

# Check if Dagster is available
try:
    import dagster
    DAGSTER_AVAILABLE = True
    DAGSTER_VERSION = dagster.__version__
except ImportError:
    DAGSTER_AVAILABLE = False
    DAGSTER_VERSION = None


class DagsterComplianceTester(BasePlatformComplianceTester):
    """Dagster-specific compliance testing with weighted penalties."""
    
    ORCHESTRATOR = Orchestrator.DAGSTER
    
    def _check_minimum_structure(self, code: str) -> bool:
        """Check if code has minimum Dagster structure."""
        # Must have @job or @graph
        has_job = "@job" in code or "@graph" in code
        # Should have @op or @asset
        has_ops = "@op" in code or "@asset" in code
        
        return has_job and has_ops
    
    # ═══════════════════════════════════════════════════════════════════════
    # LOADABILITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_platform_load(
        self, 
        code: str, 
        file_path: Path
    ) -> Tuple[float, List[Issue], Dict]:
        """Check if Dagster can load the job."""
        issues = []
        details = {
            "dagster_available": DAGSTER_AVAILABLE,
            "dagster_version": DAGSTER_VERSION,
            "module_loadable": False,
            "jobs_found": [],
            "ops_found": [],
            "assets_found": [],
        }
        
        if not DAGSTER_AVAILABLE:
            return 2.0, [Issue(
                severity=Severity.INFO,
                category="platform",
                message="Dagster not installed - cannot verify load",
            )], details
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            spec = importlib.util.spec_from_file_location("dagster_test_module", temp_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["dagster_test_module"] = module
                
                try:
                    spec.loader.exec_module(module)
                    details["module_loadable"] = True
                    
                    # Find jobs, ops, assets
                    from dagster import JobDefinition, OpDefinition, AssetsDefinition
                    
                    for attr_name in dir(module):
                        if attr_name.startswith('_'):
                            continue
                        attr = getattr(module, attr_name, None)
                        if attr is None:
                            continue
                        
                        try:
                            if isinstance(attr, JobDefinition):
                                details["jobs_found"].append(attr_name)
                            elif isinstance(attr, OpDefinition):
                                details["ops_found"].append(attr_name)
                            elif isinstance(attr, AssetsDefinition):
                                details["assets_found"].append(attr_name)
                        except TypeError:
                            continue
                    
                    if details["jobs_found"]:
                        self.logger.info(f"Found jobs: {details['jobs_found']}")
                        return 4.0, issues, details
                    elif details["assets_found"]:
                        self.logger.info(f"Found assets: {details['assets_found']}")
                        return 3.5, issues, details
                    else:
                        issues.append(Issue(
                            severity=Severity.MAJOR,
                            category="platform",
                            message="Module loaded but no jobs/assets detected",
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
                    if "dagster_test_module" in sys.modules:
                        del sys.modules["dagster_test_module"]
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
        """Check for required Dagster constructs."""
        issues = []
        details = {
            "has_job": False,
            "has_ops": False,
            "has_assets": False,
            "job_count": 0,
            "op_count": 0,
            "asset_count": 0,
        }
        
        score = 0.0
        
        # Check for @job or @graph decorator
        job_count = code.count("@job") + code.count("@graph")
        if job_count > 0:
            details["has_job"] = True
            details["job_count"] = job_count
            score += 2.0
        else:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="structure",
                message="No @job or @graph decorator found",
            ))
        
        # Check for @op decorator
        op_count = code.count("@op")
        if op_count > 0:
            details["has_ops"] = True
            details["op_count"] = op_count
            score += 1.5
        
        # Check for @asset decorator (alternative to @op)
        asset_count = code.count("@asset")
        if asset_count > 0:
            details["has_assets"] = True
            details["asset_count"] = asset_count
            if not details["has_ops"]:
                score += 1.5  # Give credit for assets if no ops
        
        if not details["has_ops"] and not details["has_assets"]:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="structure",
                message="No @op or @asset decorators found",
            ))
        
        # Check for proper imports
        if "from dagster import" in code or "import dagster" in code:
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
        """Check Dagster schedule configuration."""
        issues = []
        details = {
            "has_schedule": False,
            "has_sensor": False,
            "schedule_type": None,
        }
        
        score = 0.5  # Base score
        
        # Check for @schedule decorator
        if "@schedule" in code:
            details["has_schedule"] = True
            details["schedule_type"] = "schedule_decorator"
            score += 1.0
        
        # Check for ScheduleDefinition
        if "ScheduleDefinition" in code:
            details["has_schedule"] = True
            details["schedule_type"] = "ScheduleDefinition"
            score += 1.0
        
        # Check for @sensor decorator
        if "@sensor" in code:
            details["has_sensor"] = True
            score += 0.5
        
        # Check for cron schedule
        if "cron_schedule" in code:
            score += 0.5
        
        return min(2.5, score), issues, details
    
    def _check_default_args(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Dagster default configuration."""
        issues = []
        details = {
            "has_config_schema": False,
            "has_resources": False,
            "has_io_manager": False,
            "has_executor": False,
        }
        
        score = 0.5  # Base score
        
        # Check for config schema
        if "config_schema=" in code or "ConfigurableResource" in code:
            details["has_config_schema"] = True
            score += 0.5
        
        # Check for resource definitions
        if "ResourceDefinition" in code or "@resource" in code or "resources=" in code:
            details["has_resources"] = True
            score += 0.5
        
        # Check for IO manager
        if "io_manager" in code or "IOManager" in code:
            details["has_io_manager"] = True
            score += 0.5
        
        # Check for executor configuration
        if "executor_def=" in code or "Executor" in code:
            details["has_executor"] = True
            score += 0.5
        
        return min(2.5, score), issues, details
    
    # ═══════════════════════════════════════════════════════════════════════
    # TASK VALIDITY
    # ═══════════════════════════════════════════════════════════════════════
    def _check_task_definitions(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Dagster op/asset definitions."""
        issues = []
        details = {
            "op_count": 0,
            "asset_count": 0,
            "job_count": 0,
            "has_in_out": False,
            "has_descriptions": False,
        }
        
        score = 0.0
        
        # Count ops, assets, jobs
        op_count = code.count("@op")
        asset_count = code.count("@asset")
        job_count = code.count("@job") + code.count("@graph")
        
        details["op_count"] = op_count
        details["asset_count"] = asset_count
        details["job_count"] = job_count
        
        if job_count > 0:
            score += 1.5
        else:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="task",
                message="No @job or @graph found",
            ))
            return 0.0, issues, details
        
        if op_count > 0 or asset_count > 0:
            score += 1.5
        else:
            issues.append(Issue(
                severity=Severity.CRITICAL,
                category="task",
                message="No @op or @asset decorators found",
            ))
            return score, issues, details
        
        # Check for In/Out definitions
        if "In(" in code or "Out(" in code:
            details["has_in_out"] = True
            score += 0.5
        
        # Check for descriptions
        if 'description="' in code or "description='" in code:
            details["has_descriptions"] = True
            score += 0.5
        
        return min(4.0, score), issues, details
    
    def _check_operator_usage(
        self, 
        code: str, 
        tree: ast.AST
    ) -> Tuple[float, List[Issue], Dict]:
        """Check Dagster decorator usage."""
        issues = []
        details = {
            "has_proper_imports": False,
            "has_context_usage": False,
            "has_type_hints": False,
        }
        
        score = 0.0
        
        # Check imports
        if "from dagster import" in code:
            details["has_proper_imports"] = True
            score += 1.0
        
        # Check for context usage
        if "context:" in code or "OpExecutionContext" in code:
            details["has_context_usage"] = True
            score += 1.0
        
        # Check for type hints in op signatures
        type_hints = code.count("->") + code.count(": ") - code.count(": #")
        function_count = code.count("def ")
        
        if function_count > 0 and type_hints >= function_count * 0.5:
            details["has_type_hints"] = True
            score += 0.5
        
        # Check for op configuration
        if "@op(" in code:
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
        """Check Dagster execution capability."""
        issues = []
        details = {
            "dagster_available": DAGSTER_AVAILABLE,
            "can_execute_in_process": False,
        }
        
        if not DAGSTER_AVAILABLE:
            return 2.0, [Issue(
                severity=Severity.INFO,
                category="executability",
                message="Dagster not installed - cannot test execution",
            )], details
        
        # Check for execute_in_process pattern
        if "execute_in_process" in code or "@job" in code:
            details["can_execute_in_process"] = True
            return 4.0, issues, details
        
        return 2.0, issues, details
    
    def _extract_task_ids(self, code: str) -> set:
        """Extract op/asset names from Dagster code."""
        task_ids = set()
        
        # Pattern: @op decorated functions
        pattern = r"@op[^)]*\)?\s*\ndef\s+(\w+)"
        task_ids.update(re.findall(pattern, code, re.MULTILINE))
        
        # Pattern: @asset decorated functions
        pattern2 = r"@asset[^)]*\)?\s*\ndef\s+(\w+)"
        task_ids.update(re.findall(pattern2, code, re.MULTILINE))
        
        # Pattern: @job decorated functions
        pattern3 = r"@job[^)]*\)?\s*\ndef\s+(\w+)"
        task_ids.update(re.findall(pattern3, code, re.MULTILINE))
        
        return task_ids