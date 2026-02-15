#!/usr/bin/env python3
"""
PCT Runtime Oracle - Ground truth for platform gates by actually running/inspecting code.

Produces objective booleans:
- gate_syntax: parses as Python
- gate_imports: imports resolve (module-level imports)
- gate_instantiation: DAG/Flow/Job is created/discoverable
- gate_structure: tasks are defined and wired (heuristic per orchestrator)

This is independent ground truth (not using SAT/PCT evaluators).
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Orchestrator(Enum):
    AIRFLOW = "airflow"
    PREFECT = "prefect"
    DAGSTER = "dagster"
    UNKNOWN = "unknown"


@dataclass
class PCTGroundTruth:
    file_path: str
    orchestrator: str
    gate_syntax: bool
    gate_imports: bool
    gate_instantiation: bool
    gate_structure: bool
    gate_execution: bool
    syntax_error: Optional[str] = None
    import_error: Optional[str] = None
    instantiation_error: Optional[str] = None
    structure_error: Optional[str] = None
    execution_error: Optional[str] = None
    dag_id: Optional[str] = None
    task_count: int = 0
    has_dependencies: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def all_gates_pass(self) -> bool:
        return all([self.gate_syntax, self.gate_imports, self.gate_instantiation, self.gate_structure])


def detect_orchestrator(code: str) -> Orchestrator:
    c = (code or "").lower()
    if "airflow" in c and ("from airflow" in c or "import airflow" in c):
        return Orchestrator.AIRFLOW
    if "prefect" in c and ("from prefect" in c or "import prefect" in c):
        return Orchestrator.PREFECT
    if "dagster" in c and ("from dagster" in c or "import dagster" in c):
        return Orchestrator.DAGSTER
    return Orchestrator.UNKNOWN


class PCTRuntimeOracle:
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(self.__class__.__name__)

    def establish_ground_truth(self, file_path: Path) -> PCTGroundTruth:
        input_path = Path(file_path)
        abs_path = input_path.resolve()

        gt = PCTGroundTruth(
            file_path=str(input_path),  # keep original user path (relative is fine)
            orchestrator=self.orchestrator.value,
            gate_syntax=False,
            gate_imports=False,
            gate_instantiation=False,
            gate_structure=False,
            gate_execution=False,
        )

        try:
            code = abs_path.read_text(encoding="utf-8")
        except Exception as e:
            gt.syntax_error = f"Cannot read file: {type(e).__name__}: {e}"
            return gt

        # Gate 1: Syntax
        gt.gate_syntax, gt.syntax_error = self._check_syntax(code)
        if not gt.gate_syntax:
            return gt

        # Gate 2: Imports (subprocess, root-module check; safe-ish)
        gt.gate_imports, gt.import_error = self._check_imports(abs_path, code)
        if not gt.gate_imports:
            return gt

        # Gate 3-4: orchestrator-specific instantiation + structure
        if self.orchestrator == Orchestrator.AIRFLOW:
            return self._check_airflow(abs_path, gt)
        if self.orchestrator == Orchestrator.PREFECT:
            return self._check_prefect(abs_path, gt)
        if self.orchestrator == Orchestrator.DAGSTER:
            return self._check_dagster(abs_path, gt)

        gt.instantiation_error = "Unknown orchestrator"
        return gt

    # ---------------------------------------------------------------------

    def _check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _check_imports(self, file_path: Path, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if imports resolve by attempting to import each top-level import root.
        Runs in a subprocess to avoid polluting parent process.

        NOTE:
        - This checks root modules (e.g., 'airflow', 'pandas') not full dotted paths.
        - Full Airflow provider import errors are caught later by DagBag.
        """
        parent = str(file_path.parent.resolve())
        orch = self.orchestrator.value

        check_script = f"""
import sys, ast, os, tempfile
sys.path.insert(0, {parent!r})
orch = {orch!r}

# Make Airflow import more stable if present
if orch == "airflow":
    tmp = tempfile.mkdtemp(prefix="pct_oracle_airflow_import_")
    os.environ["AIRFLOW_HOME"] = tmp
    os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
    os.environ["AIRFLOW__LOGGING__LOGGING_LEVEL"] = "ERROR"
    os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = "sqlite:///" + tmp + "/airflow.db"
    os.environ["AIRFLOW__DATABASE__LOAD_DEFAULT_CONNECTIONS"] = "False"
    os.environ["AIRFLOW__CORE__UNIT_TEST_MODE"] = "True"

code = sys.stdin.read()
tree = ast.parse(code)
errors = []

for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = (alias.name or "").split(".")[0]
            if not root:
                continue
            try:
                __import__(root)
            except Exception as e:
                errors.append(f"Cannot import {{alias.name}}: {{type(e).__name__}}: {{e}}")
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            root = node.module.split(".")[0]
            if not root:
                continue
            try:
                __import__(root)
            except Exception as e:
                errors.append(f"Cannot import {{node.module}}: {{type(e).__name__}}: {{e}}")

if errors:
    print("IMPORT_ERRORS:" + "|".join(errors[:5]))
    sys.exit(1)

print("OK")
sys.exit(0)
"""
        try:
            res = subprocess.run(
                [sys.executable, "-c", check_script],
                input=code,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(file_path.parent),
            )
            if res.returncode == 0:
                return True, None
            out = (res.stdout or "") + (res.stderr or "")
            if "IMPORT_ERRORS:" in out:
                return False, out.split("IMPORT_ERRORS:", 1)[1].strip()[:2000]
            return False, out.strip()[:2000]
        except subprocess.TimeoutExpired:
            return False, "Import check timed out"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    # ---------------------------------------------------------------------
    # Orchestrator-specific checks
    # ---------------------------------------------------------------------

    def _check_airflow(self, file_path: Path, gt: PCTGroundTruth) -> PCTGroundTruth:
        """
        Airflow-specific: load with DagBag in an isolated environment.

        Critical fix:
        - AIRFLOW_HOME is created OUTSIDE the dag_folder to avoid DagBag walking into logs
          and triggering recursive loop detection (common on macOS due to /var <-> /private/var).
        - DagBag is pointed at the *single file path* to avoid directory walking.
        """
        if shutil.which(sys.executable) is None:
            gt.instantiation_error = "Python executable not found"
            return gt

        try:
            with tempfile.TemporaryDirectory(prefix="pct_oracle_airflow_dagdir_") as dag_td, \
                 tempfile.TemporaryDirectory(prefix="pct_oracle_airflow_home_") as home_td:

                dag_dir = Path(dag_td)
                airflow_home = Path(home_td)
                dag_file = dag_dir / file_path.name
                dag_file.write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")

                check_script = textwrap.dedent(f"""
import json, sys, os

os.environ["AIRFLOW_HOME"] = {str(airflow_home)!r}
os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "False"
os.environ["AIRFLOW__LOGGING__LOGGING_LEVEL"] = "ERROR"
os.environ["AIRFLOW__LOGGING__BASE_LOG_FOLDER"] = {str(airflow_home / "logs")!r}
os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"] = "sqlite:///" + {str(airflow_home)!r} + "/airflow.db"
os.environ["AIRFLOW__DATABASE__LOAD_DEFAULT_CONNECTIONS"] = "False"
os.environ["AIRFLOW__CORE__UNIT_TEST_MODE"] = "True"

# Ensure our temp dag dir is on path for imports
sys.path.insert(0, {str(dag_dir)!r})

try:
    from airflow.models import DagBag

    # IMPORTANT: point DagBag directly at the file to avoid walking directories
    dag_bag = DagBag(
        dag_folder={str(dag_file)!r},
        include_examples=False,
        safe_mode=False,
    )

    if dag_bag.import_errors:
        k = list(dag_bag.import_errors.keys())[0]
        v = dag_bag.import_errors[k]
        print(json.dumps({{"status":"import_error","error": str(v)[:2000], "file": str(k)}}))
        sys.exit(1)

    dags = list(dag_bag.dags.values())
    if not dags:
        print(json.dumps({{"status":"no_dags","error":"No DAGs found"}}))
        sys.exit(1)

    # Prefer DAGs from this file
    target = None
    for d in dags:
        try:
            if str(getattr(d, "fileloc", "")).endswith({file_path.name!r}):
                target = d
                break
        except Exception:
            pass
    if target is None:
        target = dags[0]

    task_count = len(getattr(target, "tasks", []) or [])
    has_deps = any(getattr(t, "downstream_list", []) for t in (getattr(target, "tasks", []) or []))

    print(json.dumps({{
        "status":"ok",
        "dag_id": getattr(target, "dag_id", None),
        "task_count": task_count,
        "has_dependencies": bool(has_deps),
    }}))
    sys.exit(0)

except Exception as e:
    print(json.dumps({{"status":"error","error": f"{{type(e).__name__}}: {{e}}"}}))
    sys.exit(1)
""").lstrip()

                res = subprocess.run(
                    [sys.executable, "-c", check_script],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(dag_dir),
                )

                lines = [ln for ln in (res.stdout or "").splitlines() if ln.strip()]
                if not lines:
                    gt.instantiation_error = (res.stderr or "No output from Airflow check")[:2000]
                    return gt

                try:
                    payload = json.loads(lines[-1])
                except Exception:
                    gt.instantiation_error = f"JSON parse error. stdout={res.stdout[:400]} stderr={res.stderr[:400]}"
                    return gt

                status = payload.get("status")
                if status == "ok":
                    gt.gate_instantiation = True
                    gt.dag_id = payload.get("dag_id")
                    gt.task_count = int(payload.get("task_count", 0) or 0)
                    gt.has_dependencies = bool(payload.get("has_dependencies", False))
                    gt.gate_structure = gt.task_count > 0
                    if not gt.gate_structure:
                        gt.structure_error = "DAG has no tasks"
                elif status == "import_error":
                    gt.gate_imports = False
                    gt.import_error = payload.get("error", "Airflow import_error")
                elif status == "no_dags":
                    gt.gate_instantiation = False
                    gt.instantiation_error = payload.get("error", "No DAGs found")
                else:
                    gt.instantiation_error = payload.get("error", "Unknown Airflow error")

                return gt

        except subprocess.TimeoutExpired:
            gt.instantiation_error = "Airflow DagBag check timed out"
            return gt
        except Exception as e:
            gt.instantiation_error = f"{type(e).__name__}: {e}"
            return gt

    def _check_prefect(self, file_path: Path, gt: PCTGroundTruth) -> PCTGroundTruth:
        """
        Prefect-specific ground truth.
        - gate_instantiation: at least one @flow function exists and is importable
        - gate_structure: at least one @task exists (heuristic) AND flow exists
        """
        parent = str(file_path.parent.resolve())
        abs_file = str(file_path.resolve())

        check_script = f"""
import sys, json, ast, importlib.util
sys.path.insert(0, {parent!r})

code = open({abs_file!r}, "r", encoding="utf-8").read()
tree = ast.parse(code)

flow_funcs = []
task_funcs = []

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        for d in node.decorator_list:
            name = None
            if isinstance(d, ast.Name):
                name = d.id
            elif isinstance(d, ast.Call) and isinstance(d.func, ast.Name):
                name = d.func.id
            if name == "flow":
                flow_funcs.append(node.name)
            if name == "task":
                task_funcs.append(node.name)

try:
    spec = importlib.util.spec_from_file_location("test_module", {abs_file!r})
    if spec is None or spec.loader is None:
        print(json.dumps({{"status":"error","error":"Could not create module spec"}}))
        sys.exit(1)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    flows_found = [fn for fn in flow_funcs if hasattr(mod, fn)]

    print(json.dumps({{
        "status": "ok",
        "flow_count": len(flows_found),
        "task_count": len(task_funcs),
    }}))
    sys.exit(0)

except Exception as e:
    print(json.dumps({{"status":"error","error": f"{{type(e).__name__}}: {{e}}"}}))
    sys.exit(1)
"""
        try:
            res = subprocess.run(
                [sys.executable, "-c", check_script],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(file_path.parent),
            )

            lines = [ln for ln in (res.stdout or "").splitlines() if ln.strip()]
            if not lines:
                gt.instantiation_error = (res.stderr or "No output from Prefect check")[:2000]
                return gt

            payload = json.loads(lines[-1])
            if payload.get("status") == "ok":
                flow_count = int(payload.get("flow_count", 0) or 0)
                task_count = int(payload.get("task_count", 0) or 0)
                gt.task_count = task_count
                gt.gate_instantiation = flow_count > 0
                gt.gate_structure = (flow_count > 0) and (task_count > 0)
                if not gt.gate_instantiation:
                    gt.instantiation_error = "No @flow decorated functions found"
                elif not gt.gate_structure:
                    gt.structure_error = "No @task decorated functions found (structure heuristic failed)"
            else:
                gt.instantiation_error = payload.get("error", "Unknown Prefect error")

            return gt

        except subprocess.TimeoutExpired:
            gt.instantiation_error = "Prefect check timed out"
            return gt
        except Exception as e:
            gt.instantiation_error = f"{type(e).__name__}: {e}"
            return gt

    def _check_dagster(self, file_path: Path, gt: PCTGroundTruth) -> PCTGroundTruth:
        """
        Dagster-specific ground truth.
        - gate_instantiation: job_count>0 OR asset_count>0
        - gate_structure: if job exists, needs ops/assets; if only assets, structure passes
        """
        parent = str(file_path.parent.resolve())
        abs_file = str(file_path.resolve())

        check_script = f"""
import sys, json, ast, importlib.util
sys.path.insert(0, {parent!r})

code = open({abs_file!r}, "r", encoding="utf-8").read()
tree = ast.parse(code)

jobs = []
ops = []
assets = []

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        for d in node.decorator_list:
            name = None
            if isinstance(d, ast.Name):
                name = d.id
            elif isinstance(d, ast.Call) and isinstance(d.func, ast.Name):
                name = d.func.id
            if name in ("job", "graph"):
                jobs.append(node.name)
            elif name == "op":
                ops.append(node.name)
            elif name == "asset":
                assets.append(node.name)

try:
    spec = importlib.util.spec_from_file_location("test_module", {abs_file!r})
    if spec is None or spec.loader is None:
        print(json.dumps({{"status":"error","error":"Could not create module spec"}}))
        sys.exit(1)

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print(json.dumps({{
        "status": "ok",
        "job_count": len(jobs),
        "op_count": len(ops),
        "asset_count": len(assets),
    }}))
    sys.exit(0)

except Exception as e:
    print(json.dumps({{"status":"error","error": f"{{type(e).__name__}}: {{e}}"}}))
    sys.exit(1)
"""
        try:
            res = subprocess.run(
                [sys.executable, "-c", check_script],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(file_path.parent),
            )

            lines = [ln for ln in (res.stdout or "").splitlines() if ln.strip()]
            if not lines:
                gt.instantiation_error = (res.stderr or "No output from Dagster check")[:2000]
                return gt

            payload = json.loads(lines[-1])
            if payload.get("status") == "ok":
                job_count = int(payload.get("job_count", 0) or 0)
                op_count = int(payload.get("op_count", 0) or 0)
                asset_count = int(payload.get("asset_count", 0) or 0)

                gt.task_count = op_count + asset_count
                gt.gate_instantiation = (job_count > 0) or (asset_count > 0)

                if job_count > 0:
                    gt.gate_structure = (op_count > 0) or (asset_count > 0)
                else:
                    gt.gate_structure = (asset_count > 0)

                if not gt.gate_instantiation:
                    gt.instantiation_error = "No @job/@graph and no @asset found"
                elif not gt.gate_structure:
                    gt.structure_error = "Job exists but no ops/assets found"
            else:
                gt.instantiation_error = payload.get("error", "Unknown Dagster error")

            return gt

        except subprocess.TimeoutExpired:
            gt.instantiation_error = "Dagster check timed out"
            return gt
        except Exception as e:
            gt.instantiation_error = f"{type(e).__name__}: {e}"
            return gt


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Establish PCT ground truth via runtime oracle")
    parser.add_argument("file", help="Path to DAG/flow/job file")
    parser.add_argument("--orchestrator", choices=["airflow", "prefect", "dagster", "auto"], default="auto")
    parser.add_argument("--out", default=None, help="Write JSON to this path")
    args = parser.parse_args()

    file_path = Path(args.file)
    code = file_path.read_text(encoding="utf-8")
    orch = detect_orchestrator(code) if args.orchestrator == "auto" else Orchestrator(args.orchestrator)

    oracle = PCTRuntimeOracle(orch)
    gt = oracle.establish_ground_truth(file_path)

    payload = gt.to_dict()
    print(json.dumps(payload, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()