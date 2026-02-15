#!/usr/bin/env python3
"""
Dependency Bootstrap for Oracle Environments (Option A - heavy/high coverage)

Pipeline:
1) Run analyze_imports.py over mutation_benchmark/originals
2) Write report to mutation_benchmark/ground_truth/import_analysis_report.json (forced)
3) Map modules -> pip packages (incl. airflow providers)
4) Write per-env requirements files under mutation_benchmark/env_specs/
5) Optionally install into each venv (Airflow uses constraints)
6) Optionally freeze envs (pip freeze) for reproducibility

This is intended to be called from your ground_truth_orchestrator pipeline.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ----------------------------
# Helpers
# ----------------------------

def is_windows() -> bool:
    return platform.system().lower().startswith("win")

def venv_python(venv_path: Path) -> Path:
    return venv_path / ("Scripts/python.exe" if is_windows() else "bin/python")

def sh(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)

def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def read_req_lines(path: Path) -> List[str]:
    pkgs: List[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        pkgs.append(s)
    return pkgs

def airflow_constraints_url(airflow_version: str, py_minor: str = "3.10") -> str:
    # Airflow uses constraints-3.10.txt etc
    return f"https://raw.githubusercontent.com/apache/airflow/constraints-{airflow_version}/constraints-{py_minor}.txt"

def pip_install_requirements(py: Path, req_path: Path, *, constraint: Optional[str] = None) -> None:
    cmd = [str(py), "-m", "pip", "install", "-r", str(req_path)]
    if constraint:
        cmd += ["--constraint", constraint]
    sh(cmd, check=True)

def pip_freeze(py: Path) -> str:
    r = sh([str(py), "-m", "pip", "freeze"], check=True)
    return (r.stdout or "").strip()


# ----------------------------
# Mapping logic
# ----------------------------

# "Heavy/high-coverage" base deps (shared across orchestrators)
HEAVY_BASE = [
    "requests",
    "pandas",
    "redis",
    "psycopg2-binary",
    "snowflake-connector-python",
]

# Import module -> pip package mapping (override analyze_imports defaults)
MODULE_TO_PIP = {
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "yaml": "pyyaml",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",

    # critical DB + snowflake mappings
    "psycopg2": "psycopg2-binary",
    "snowflake": "snowflake-connector-python",
}

# Airflow provider module prefix -> provider dist slug override
# (fallback is first segment after airflow.providers.)
AIRFLOW_PROVIDER_PREFIX_OVERRIDES = {
    "cncf.kubernetes": "cncf-kubernetes",
    "microsoft.azure": "microsoft-azure",
}

def airflow_provider_pkg_from_module(module: str) -> Optional[str]:
    """
    Convert airflow.providers.<provider>... import to apache-airflow-providers-<provider>
    Handles some nested providers.
    """
    if not module.startswith("airflow.providers."):
        return None

    rest = module[len("airflow.providers."):]  # e.g. "snowflake.operators.snowflake"
    parts = rest.split(".")
    if not parts:
        return None

    # Check 2-part override
    if len(parts) >= 2:
        two = f"{parts[0]}.{parts[1]}"
        if two in AIRFLOW_PROVIDER_PREFIX_OVERRIDES:
            slug = AIRFLOW_PROVIDER_PREFIX_OVERRIDES[two]
            return f"apache-airflow-providers-{slug}"

    # Default: first segment
    slug = parts[0]
    return f"apache-airflow-providers-{slug}"


def module_to_pip_pkg(module: str, *, orchestrator: str) -> Optional[str]:
    """
    Map an import module name to a pip package.

    Notes:
    - For Airflow, provider imports are handled separately via airflow_provider_pkg_from_module.
    - Avoid suggesting SQLAlchemy installation for Airflow (must stay pinned by constraints).
    """
    if not module:
        return None

    # Airflow provider mapping
    if orchestrator == "airflow" and module.startswith("airflow.providers."):
        return airflow_provider_pkg_from_module(module)

    root = module.split(".", 1)[0]

    # Skip orchestrator roots
    if root in {"airflow", "prefect", "dagster"}:
        return None

    # Special: never propose installing sqlalchemy in Airflow env
    if orchestrator == "airflow" and root == "sqlalchemy":
        return None

    return MODULE_TO_PIP.get(root, root)


def extract_modules_from_report(report: Dict[str, Any], orchestrator: str) -> Set[str]:
    """
    Extract all non-stdlib imported module strings from the analyze_imports report.
    We do NOT rely only on "top_missing_dependencies" because Airflow providers
    are categorized as "orchestrator".
    """
    orch_block = (report.get("by_orchestrator", {}) or {}).get(orchestrator, {}) or {}
    modules_by_cat = orch_block.get("modules_by_category", {}) or {}

    modules: Set[str] = set()

    # Include common + unknown always
    for cat in ("common", "unknown"):
        for item in modules_by_cat.get(cat, []) or []:
            m = item.get("module")
            if m:
                modules.add(m)

    # For Airflow specifically, also include provider imports (under category orchestrator)
    if orchestrator == "airflow":
        for item in modules_by_cat.get("orchestrator", []) or []:
            m = item.get("module")
            if m and m.startswith("airflow.providers."):
                modules.add(m)

    return modules


def build_requirements_for_orchestrator(report: Dict[str, Any], orchestrator: str) -> List[str]:
    modules = extract_modules_from_report(report, orchestrator)

    pkgs: Set[str] = set()

    # Heavy base set (high coverage)
    for p in HEAVY_BASE:
        # Avoid trying to install psycopg2-binary for Airflow if you prefer not to;
        # but it's usually OK to keep, so we include it.
        pkgs.add(p)

    # Map modules -> packages
    for m in sorted(modules):
        pkg = module_to_pip_pkg(m, orchestrator=orchestrator)
        if pkg:
            pkgs.add(pkg)

    # Airflow: prefer to explicitly include slack/snowflake providers if used
    # (They are provider packages; module mapping already covers it, but keep clarity)
    if orchestrator == "airflow":
        # If any provider appears, mapping will add them; no special casing needed
        pass

    # Clean up weird/unsafe installs
    # Example: "google" on pip is often a meta package and not what users want.
    # In heavy mode, you may still want it. Here we skip "google" root to avoid confusion.
    if "google" in pkgs:
        pkgs.remove("google")

    # Always ensure psycopg2 maps to psycopg2-binary (dedupe)
    if "psycopg2" in pkgs:
        pkgs.remove("psycopg2")
        pkgs.add("psycopg2-binary")

    return sorted(pkgs)


def main():
    ap = argparse.ArgumentParser(description="Bootstrap oracle env dependencies from dataset imports (heavy coverage).")
    ap.add_argument("--dataset-dir", default="mutation_benchmark", help="mutation_benchmark root")
    ap.add_argument("--input-dir", default=None, help="Defaults to <dataset-dir>/originals")
    ap.add_argument("--orchestrators", nargs="*", default=["airflow", "prefect", "dagster"])

    ap.add_argument("--airflow-version", default="2.8.4")
    ap.add_argument("--py-minor", default="3.10", help="Constraint file selector: 3.10, 3.11, etc")

    ap.add_argument("--airflow-venv", default=".venv-airflow-eval")
    ap.add_argument("--prefect-venv", default=".venv-prefect-eval")
    ap.add_argument("--dagster-venv", default=".venv-dagster-eval")

    ap.add_argument("--install", action="store_true", help="Install into venvs after writing requirements")
    ap.add_argument("--freeze", action="store_true", help="Write pip freeze snapshots after install")

    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    input_dir = Path(args.input_dir) if args.input_dir else dataset_dir / "originals"
    gt_dir = dataset_dir / "ground_truth"
    env_specs = dataset_dir / "env_specs"

    analyze_script = Path(__file__).resolve().parent / "analyze_imports.py"
    if not analyze_script.exists():
        raise FileNotFoundError(f"analyze_imports.py not found at: {analyze_script}")

    # 1) Run analyze_imports.py and force output location
    report_path = gt_dir / "import_analysis_report.json"
    cmd = [
        sys.executable, str(analyze_script),
        "--input-dir", str(input_dir),
        "--output", str(report_path),
        "--orchestrators",
    ] + list(args.orchestrators)
    print("[INFO] Running import analysis...")
    sh(cmd, check=True)
    print(f"[OK] Import report written: {report_path}")

    report = load_json(report_path)

    # 2) Build requirements files
    req_paths: Dict[str, Path] = {
        "airflow": env_specs / "airflow_extra_requirements.txt",
        "prefect": env_specs / "prefect_extra_requirements.txt",
        "dagster": env_specs / "dagster_extra_requirements.txt",
    }

    summary: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "input_dir": str(input_dir),
        "report_path": str(report_path),
        "generated_requirements": {},
        "timestamp": datetime.now().isoformat() if "datetime" in globals() else "",
    }

    for orch in args.orchestrators:
        orch = orch.strip().lower()
        if orch not in req_paths:
            continue

        pkgs = build_requirements_for_orchestrator(report, orch)
        header = [
            f"# Auto-generated extra requirements for {orch}",
            f"# Generated from: {report_path}",
            "# Heavy/high-coverage profile: includes pandas/redis/psycopg2-binary/snowflake connector + inferred deps",
            "",
        ]
        content = "\n".join(header + pkgs) + "\n"
        write_text(req_paths[orch], content)

        summary["generated_requirements"][orch] = {
            "path": str(req_paths[orch]),
            "count": len(pkgs),
            "packages": pkgs,
        }
        print(f"[OK] Wrote {orch} extras: {req_paths[orch]} ({len(pkgs)} packages)")

    write_json(env_specs / "dependency_bootstrap_summary.json", summary)
    print(f"[OK] Wrote summary: {env_specs / 'dependency_bootstrap_summary.json'}")

    # 3) Optional install into venvs
    if args.install:
        print("\n[INFO] Installing extras into venvs...")

        airflow_py = venv_python(Path(args.airflow_venv))
        prefect_py = venv_python(Path(args.prefect_venv))
        dagster_py = venv_python(Path(args.dagster_venv))

        constraints = airflow_constraints_url(args.airflow_version, py_minor=args.py_minor)
        print(f"[airflow] constraints: {constraints}")

        # Airflow install with constraints
        if "airflow" in args.orchestrators:
            pip_install_requirements(airflow_py, req_paths["airflow"], constraint=constraints)

        # Prefect/Dagster install without constraints
        if "prefect" in args.orchestrators:
            pip_install_requirements(prefect_py, req_paths["prefect"], constraint=None)

        if "dagster" in args.orchestrators:
            pip_install_requirements(dagster_py, req_paths["dagster"], constraint=None)

        print("[OK] Dependency install finished.")

    # 4) Optional freeze snapshots
    if args.freeze:
        print("\n[INFO] Writing pip freeze snapshots...")
        airflow_py = venv_python(Path(args.airflow_venv))
        prefect_py = venv_python(Path(args.prefect_venv))
        dagster_py = venv_python(Path(args.dagster_venv))

        if "airflow" in args.orchestrators:
            write_text(env_specs / "airflow_freeze.txt", pip_freeze(airflow_py) + "\n")
        if "prefect" in args.orchestrators:
            write_text(env_specs / "prefect_freeze.txt", pip_freeze(prefect_py) + "\n")
        if "dagster" in args.orchestrators:
            write_text(env_specs / "dagster_freeze.txt", pip_freeze(dagster_py) + "\n")

        print(f"[OK] Freeze snapshots written to: {env_specs}")

    print("\nDONE.")


if __name__ == "__main__":
    import sys
    from datetime import datetime
    main()