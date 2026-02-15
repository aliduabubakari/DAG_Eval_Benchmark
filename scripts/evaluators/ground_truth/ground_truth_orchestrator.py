#!/usr/bin/env python3
"""
Ground Truth Orchestrator (GT1 + GT2) — no manual venv activation.

Pipeline:
1) (Optional) Setup 3 venvs via env_preflight_check.py --setup --multi
2) Build GT1 mutations (mutation_injection_gpt5.py) [controller python]
3) Bootstrap dependencies (dependency_bootstrap.py):
     - analyze imports from mutation_benchmark/originals
     - write env_specs/*_extra_requirements.txt
     - install into each venv (Airflow with constraints)
     - freeze snapshots
4) Validate GT1 per orchestrator under the correct venv python (validate_gt1_dataset.py)
5) Build GT2 tool ensemble under a chosen tools python (build_gt2_tool_ensemble_batch.py)

This creates a standardized “oracle_extended” environment profile.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def venv_python(venv_path: Path) -> Path:
    return venv_path / ("Scripts/python.exe" if is_windows() else "bin/python")


def run_cmd(
    cmd: List[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    check: bool = True
) -> int:
    cmd_str = " ".join([str(x) for x in cmd])
    print(f"\n[CMD] {cmd_str}")
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed (rc={r.returncode}): {cmd_str}")
    return r.returncode


def require_python(p: Path, name: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{name} python not found at: {p}")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check_tools_available(py: Path) -> None:
    """
    Ensure tools python has pylint/flake8/bandit/radon installed.
    """
    code = "import pylint, flake8, bandit, radon; print('OK')"
    r = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Tools python missing required tool modules.\n"
            f"python={py}\n"
            f"stderr={r.stderr[:500]}\n"
            f"stdout={r.stdout[:500]}"
        )


def merge_validation_reports(reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    combined: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "reports": {k: {"path": v.get("_path"), "counts": v.get("counts")} for k, v in reports.items()},
        "counts": {"ok": 0, "failed": 0, "not_evaluable": 0, "io_error": 0, "schema_error": 0},
        "per_mutation_type": {},
    }

    for orch, rep in reports.items():
        counts = rep.get("counts", {}) or {}
        for k in combined["counts"].keys():
            combined["counts"][k] += int(counts.get(k, 0) or 0)

        pm = rep.get("per_mutation_type", {}) or {}
        for mid, stats in pm.items():
            dst = combined["per_mutation_type"].setdefault(mid, {
                "total": 0, "ok": 0, "failed": 0, "not_evaluable": 0, "io_error": 0, "schema_error": 0
            })
            for k in dst.keys():
                dst[k] += int(stats.get(k, 0) or 0)

    for mid, stats in combined["per_mutation_type"].items():
        total = stats["total"]
        not_eval = stats["not_evaluable"]
        evaluable = total - not_eval
        stats["pass_rate_ok_over_total"] = (stats["ok"] / total) if total else 0.0
        stats["pass_rate_ok_over_evaluable"] = (stats["ok"] / evaluable) if evaluable > 0 else None

    combined["per_mutation_type_table"] = [
        {"mutation_id": mid, **stats}
        for mid, stats in sorted(combined["per_mutation_type"].items(), key=lambda kv: kv[0])
    ]
    return combined


def main():
    ap = argparse.ArgumentParser(description="End-to-end builder for GT1 + GT2 without manual venv activation.")

    parser = ap
    parser.add_argument("--dataset-dir", default="mutation_benchmark", help="Output dataset dir")
    parser.add_argument("--input-dir", default=None, help="Input dir for GT1 generation (must contain airflow/prefect/dagster)")
    parser.add_argument("--catalog", default=None, help="Path to mutation_catalog.json")

    # Azure / GT1 generation args
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--azure-deployment", default="gpt-5-mini")
    parser.add_argument("--api-version", default="2024-12-01-preview")
    parser.add_argument("--api-key", default=None)

    parser.add_argument("--orchestrators", nargs="*", default=["airflow", "prefect", "dagster"])
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--mutations-per-file", type=int, default=1)
    parser.add_argument("--max-spec-tries-per-file", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    # GT2 builder options
    parser.add_argument("--gt2-workers", type=int, default=0)
    parser.add_argument("--gt2-overwrite", action="store_true")

    # Multi-venv paths
    parser.add_argument("--airflow-venv", default=".venv-airflow-eval")
    parser.add_argument("--prefect-venv", default=".venv-prefect-eval")
    parser.add_argument("--dagster-venv", default=".venv-dagster-eval")

    # Versions (used for constraints during dependency bootstrap)
    parser.add_argument("--airflow-version", default="2.8.4")
    parser.add_argument("--prefect-version", default="3.6.9")
    parser.add_argument("--dagster-version", default="1.12.8")
    parser.add_argument("--py-minor", default="3.10")

    # Setup options for env_preflight_check.py
    parser.add_argument("--setup-envs", action="store_true", help="Run env_preflight_check.py --setup --multi")
    parser.add_argument("--strict", action="store_true", help="Pass --strict to env_preflight_check.py")
    parser.add_argument("--python-cmd", default=None, help="Pass-through python cmd to env_preflight_check.py")
    parser.add_argument("--with-mysql-provider", action="store_true")

    # Tools python selection (single stable toolset)
    parser.add_argument("--tools-python", default=None,
                        help="Path to python used for GT2 tools. If omitted, defaults to dagster venv python.")

    # Which steps to run
    parser.add_argument("--build-gt1", action="store_true")
    parser.add_argument("--validate-gt1", action="store_true")
    parser.add_argument("--build-gt2", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run setup + GT1 + deps + validate + GT2")

    # Optional: allow skipping dependency bootstrap (but default is integrated)
    parser.add_argument("--skip-dependency-bootstrap", action="store_true",
                        help="Skip dependency_bootstrap.py step (not recommended).")

    args = parser.parse_args()

    if args.all:
        args.setup_envs = True
        args.build_gt1 = True
        args.validate_gt1 = True
        args.build_gt2 = True

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parents[2]  # .../scripts

    env_preflight = scripts_dir / "evaluators" / "env_preflight_check.py"
    mutation_script = scripts_dir / "evaluators" / "ground_truth" / "mutation_injection_gpt5.py"
    validate_script = scripts_dir / "evaluators" / "ground_truth" / "validate_gt1_dataset.py"
    gt2_script = scripts_dir / "evaluators" / "ground_truth" / "build_gt2_tool_ensemble_batch.py"
    dep_bootstrap_script = scripts_dir / "evaluators" / "ground_truth" / "dependency_bootstrap.py"

    airflow_py = venv_python(Path(args.airflow_venv))
    prefect_py = venv_python(Path(args.prefect_venv))
    dagster_py = venv_python(Path(args.dagster_venv))
    tools_py = Path(args.tools_python) if args.tools_python else dagster_py

    # ------------------------------------------------------------------
    # 1) Setup envs (base orchestrator + base providers + tools)
    # ------------------------------------------------------------------
    if args.setup_envs:
        if not env_preflight.exists():
            raise FileNotFoundError(f"env_preflight_check.py not found: {env_preflight}")

        cmd = [
            sys.executable, str(env_preflight),
            "--setup",
            "--multi",
            "--airflow-venv", args.airflow_venv,
            "--prefect-venv", args.prefect_venv,
            "--dagster-venv", args.dagster_venv,
            "--airflow-version", args.airflow_version,
            "--prefect-version", args.prefect_version,
            "--dagster-version", args.dagster_version,
        ]
        if args.strict:
            cmd.append("--strict")
        if args.python_cmd:
            cmd += ["--python-cmd", args.python_cmd]
        if args.with_mysql_provider:
            cmd.append("--with-mysql-provider")

        run_cmd(cmd, check=True)

    # Require venv python interpreters if we will validate gt1 or bootstrap deps or build gt2
    require_python(airflow_py, "airflow")
    require_python(prefect_py, "prefect")
    require_python(dagster_py, "dagster")

    # ------------------------------------------------------------------
    # 2) Build GT1 mutations (controller python)
    # ------------------------------------------------------------------
    if args.build_gt1:
        if not args.input_dir or not args.catalog:
            raise ValueError("--build-gt1 requires --input-dir and --catalog")
        if not args.azure_endpoint:
            raise ValueError("--build-gt1 requires --azure-endpoint")

        env = dict(os.environ)
        if args.api_key:
            env["AZURE_OPENAI_API_KEY"] = args.api_key

        cmd = [
            sys.executable, str(mutation_script),
            "--input-dir", str(args.input_dir),
            "--output-dir", str(dataset_dir),
            "--catalog", str(args.catalog),
            "--azure-endpoint", str(args.azure_endpoint),
            "--azure-deployment", str(args.azure_deployment),
            "--api-version", str(args.api_version),
            "--max-files", str(args.max_files),
            "--mutations-per-file", str(args.mutations_per_file),
            "--max-spec-tries-per-file", str(args.max_spec_tries_per_file),
        ]
        if args.orchestrators:
            cmd += ["--orchestrators"] + list(args.orchestrators)
        if args.resume:
            cmd.append("--resume")
        if args.verbose:
            cmd.append("--verbose")

        run_cmd(cmd, env=env, check=True)

    # ------------------------------------------------------------------
    # 3) Dependency bootstrap (Option A: heavy/high coverage)
    #     - analyze imports under dataset/originals
    #     - write env_specs/*_extra_requirements.txt
    #     - install into each venv (+ airflow constraints)
    #     - freeze
    # ------------------------------------------------------------------
    if (args.build_gt1 or args.validate_gt1 or args.build_gt2) and (not args.skip_dependency_bootstrap):
        if not dep_bootstrap_script.exists():
            raise FileNotFoundError(
                f"dependency_bootstrap.py not found at {dep_bootstrap_script}. "
                "Create it in scripts/evaluators/ground_truth/dependency_bootstrap.py"
            )

        cmd = [
            sys.executable, str(dep_bootstrap_script),
            "--dataset-dir", str(dataset_dir),
            "--airflow-version", str(args.airflow_version),
            "--py-minor", str(args.py_minor),
            "--airflow-venv", str(args.airflow_venv),
            "--prefect-venv", str(args.prefect_venv),
            "--dagster-venv", str(args.dagster_venv),
            "--install",
            "--freeze",
        ]
        if args.orchestrators:
            cmd += ["--orchestrators"] + list(args.orchestrators)

        run_cmd(cmd, check=True)

        # Optional: run a post-bootstrap check (not setup; just checks)
        cmd_check = [
            sys.executable, str(env_preflight),
            "--multi",
            "--airflow-venv", args.airflow_venv,
            "--prefect-venv", args.prefect_venv,
            "--dagster-venv", args.dagster_venv,
            "--airflow-version", args.airflow_version,
            "--prefect-version", args.prefect_version,
            "--dagster-version", args.dagster_version,
        ]
        if args.strict:
            cmd_check.append("--strict")
        run_cmd(cmd_check, check=True)

    # ------------------------------------------------------------------
    # 4) Validate GT1 (per orchestrator, under correct venv python)
    # ------------------------------------------------------------------
    validation_reports: Dict[str, Dict[str, Any]] = {}
    if args.validate_gt1:
        for orch in args.orchestrators:
            orch = orch.strip().lower()
            if orch not in ("airflow", "prefect", "dagster"):
                continue

            py = airflow_py if orch == "airflow" else prefect_py if orch == "prefect" else dagster_py

            out_path = dataset_dir / f"validation_report_{orch}.json"
            cache_path = dataset_dir / f"oracle_cache_{orch}.json"

            cmd = [
                str(py), str(validate_script),
                "--dataset-dir", str(dataset_dir),
                "--orchestrators", orch,
                "--cache-oracle",
                "--cache-path", str(cache_path),
                "--out", str(out_path),
            ]
            run_cmd(cmd, check=True)

            rep = load_json(out_path)
            rep["_path"] = str(out_path)
            validation_reports[orch] = rep

        if validation_reports:
            combined = merge_validation_reports(validation_reports)
            combined_path = dataset_dir / "validation_report.json"
            combined_path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
            print(f"\n[OK] Wrote combined validation report: {combined_path}")

    # ------------------------------------------------------------------
    # 5) Build GT2 under one stable tools python
    # ------------------------------------------------------------------
    if args.build_gt2:
        require_python(tools_py, "tools")
        check_tools_available(tools_py)

        cmd = [
            str(tools_py), str(gt2_script),
            "--dataset-dir", str(dataset_dir),
        ]
        if args.gt2_workers and args.gt2_workers > 0:
            cmd += ["--workers", str(args.gt2_workers)]
        if args.gt2_overwrite:
            cmd.append("--overwrite")

        run_cmd(cmd, check=True)

    print("\nDONE.")
    print(f"Dataset dir: {dataset_dir}")
    if args.validate_gt1:
        for k, v in validation_reports.items():
            print(f" - validation {k}: {v.get('_path')}")
    if args.build_gt2:
        print("GT2 dir:", dataset_dir / "ground_truth" / "gt2_tool_ensemble")


if __name__ == "__main__":
    main()