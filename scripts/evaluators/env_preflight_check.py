#!/usr/bin/env python3
"""
Environment Preflight Check + Auto-Setup (Multi-venv)
====================================================

Why multi-venv?
---------------
Airflow 2.8.4 requires SQLAlchemy 1.4.x (via constraints).
Prefect 3.x typically uses SQLAlchemy 2.x.
Installing Airflow + Prefect in the same venv can break Airflow imports.

This script supports:
- Single-venv legacy mode (airflow only; discouraged for mixed orchestrators)
- Multi-venv mode (recommended):
    * Airflow venv (pinned via constraints)
    * Prefect venv
    * Dagster venv

NEW:
----
Supports installing extra requirements per venv during setup:
  --airflow-extra-req path/to/airflow_extra_requirements.txt
  --prefect-extra-req path/to/prefect_extra_requirements.txt
  --dagster-extra-req path/to/dagster_extra_requirements.txt

Airflow extras are installed with constraints for reproducibility.
"""

from __future__ import annotations

import sys
import json
import argparse
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    installed_version: Optional[str] = None
    required_version: Optional[str] = None
    category: str = "general"
    install_hint: Optional[str] = None


# ----------------------------
# Shell helpers
# ----------------------------

def sh(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=True, text=True)

def exists_exe(p: str) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


# ----------------------------
# Venv helpers
# ----------------------------

def venv_python(venv_path: Path) -> Path:
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"

def venv_pip_cmd(venv_path: Path) -> List[str]:
    return [str(venv_python(venv_path)), "-m", "pip"]

def pip_install(venv_path: Path, pkgs: List[str], *, constraint: Optional[str] = None) -> None:
    cmd = venv_pip_cmd(venv_path) + ["install"] + pkgs
    if constraint:
        cmd += ["--constraint", constraint]
    sh(cmd, check=True)

def pip_install_best_effort(venv_path: Path, pkgs: List[str], *, constraint: Optional[str] = None) -> bool:
    try:
        pip_install(venv_path, pkgs, constraint=constraint)
        return True
    except subprocess.CalledProcessError as e:
        print(f"WARNING: pip install failed: {' '.join(pkgs)}")
        if e.stderr:
            print(e.stderr[:600])
        return False

def pip_upgrade_base(venv_path: Path) -> None:
    pip_install_best_effort(venv_path, ["-U", "pip", "setuptools", "wheel"])

def read_requirements_file(req_path: Optional[str]) -> List[str]:
    """
    Parse a requirements file and return non-comment, non-empty lines.
    Used mainly to decide if there's anything to install.
    """
    if not req_path:
        return []
    p = Path(req_path)
    if not p.exists():
        raise SystemExit(f"ERROR: requirements file not found: {p}")
    pkgs: List[str] = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        pkgs.append(s)
    return pkgs

def pip_install_requirements_file(venv_path: Path, req_path: str, *, constraint: Optional[str] = None) -> None:
    """
    pip install -r <req_path> [--constraint ...]
    """
    cmd = venv_pip_cmd(venv_path) + ["install"]
    if constraint:
        cmd += ["--constraint", constraint]
    cmd += ["-r", req_path]
    sh(cmd, check=True)


# ----------------------------
# Python 3.10.13 discovery (pyenv-aware)
# ----------------------------

def python_version_string(python_cmd: str) -> Optional[str]:
    try:
        r = sh([python_cmd, "--version"], check=False)
        out = (r.stdout or r.stderr or "").strip()
        if r.returncode == 0 and out.startswith("Python "):
            return out.replace("Python ", "")
        return None
    except Exception:
        return None

def is_python_31013(python_cmd: str) -> bool:
    v = python_version_string(python_cmd)
    return v == "3.10.13"

def try_pyenv_prefix(version: str = "3.10.13") -> Optional[str]:
    try:
        r = sh(["pyenv", "prefix", version], check=False)
        if r.returncode != 0:
            return None
        prefix = (r.stdout or "").strip()
        if not prefix:
            return None
        cand = str(Path(prefix) / "bin" / "python")
        if exists_exe(cand) and is_python_31013(cand):
            return cand
        return None
    except Exception:
        return None

def find_python_31013(preferred: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)

    pyenv_bin = try_pyenv_prefix("3.10.13")
    if pyenv_bin:
        return pyenv_bin

    candidates += ["python3.10", "python3", "python"]
    home = Path.home()
    candidates += [
        str(home / ".pyenv" / "shims" / "python3.10"),
        str(home / ".pyenv" / "shims" / "python"),
    ]

    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    for cmd in ordered:
        if is_python_31013(cmd):
            return cmd

    return None

def ensure_python_31013_or_fail(preferred: Optional[str]) -> str:
    resolved = find_python_31013(preferred)
    if not resolved:
        raise SystemExit(
            "ERROR: Could not find Python 3.10.13.\n"
            "Fix options:\n"
            "  - pyenv install 3.10.13\n"
            "  - or pass --python-cmd /path/to/python3.10.13\n"
        )
    return resolved


# ----------------------------
# Create venv
# ----------------------------

def create_venv(venv_path: Path, python_cmd: str) -> None:
    if venv_path.exists():
        return
    print(f"Creating virtual environment at: {venv_path}")
    sh([python_cmd, "-m", "venv", str(venv_path)], check=True)

    vp = str(venv_python(venv_path))
    ver = python_version_string(vp)
    if ver != "3.10.13":
        raise SystemExit(
            f"ERROR: venv python version is {ver}, expected 3.10.13.\n"
            f"venv python: {vp}\n"
        )


# ----------------------------
# Setup install sets
# ----------------------------

DEFAULT_TOOLS = ["pylint", "radon", "bandit", "flake8"]

AIRFLOW_DEFAULT_PROVIDERS = [
    "apache-airflow-providers-ftp",
    "apache-airflow-providers-http",
    "apache-airflow-providers-postgres",
    "apache-airflow-providers-sqlite",
    "apache-airflow-providers-docker",
    "apache-airflow-providers-amazon",
    "apache-airflow-providers-cncf-kubernetes",
]


def setup_airflow_env(
    venv_path: Path,
    python_cmd: str,
    airflow_version: str,
    with_mysql_provider: bool,
    airflow_extra_req: Optional[str] = None,
) -> None:
    create_venv(venv_path, python_cmd)
    pip_upgrade_base(venv_path)

    constraint = f"https://raw.githubusercontent.com/apache/airflow/constraints-{airflow_version}/constraints-3.10.txt"
    print(f"[airflow] Using constraints: {constraint}")

    pip_install(venv_path, [f"apache-airflow=={airflow_version}"], constraint=constraint)
    pip_install(venv_path, AIRFLOW_DEFAULT_PROVIDERS, constraint=constraint)

    if with_mysql_provider:
        pip_install_best_effort(venv_path, ["apache-airflow-providers-mysql"], constraint=constraint)

    # Install static tools in airflow env (do NOT install prefect/dagster here)
    pip_install_best_effort(venv_path, DEFAULT_TOOLS)

    # Install extra requirements if provided
    if airflow_extra_req:
        extra_pkgs = read_requirements_file(airflow_extra_req)
        if extra_pkgs:
            print(f"[airflow] Installing extra requirements from: {airflow_extra_req}")
            pip_install_requirements_file(venv_path, airflow_extra_req, constraint=constraint)


def setup_prefect_env(
    venv_path: Path,
    python_cmd: str,
    prefect_version: str,
    prefect_extra_req: Optional[str] = None,
) -> None:
    create_venv(venv_path, python_cmd)
    pip_upgrade_base(venv_path)

    pip_install(venv_path, [f"prefect=={prefect_version}"])
    pip_install_best_effort(venv_path, DEFAULT_TOOLS)

    if prefect_extra_req:
        extra_pkgs = read_requirements_file(prefect_extra_req)
        if extra_pkgs:
            print(f"[prefect] Installing extra requirements from: {prefect_extra_req}")
            pip_install_requirements_file(venv_path, prefect_extra_req, constraint=None)


def setup_dagster_env(
    venv_path: Path,
    python_cmd: str,
    dagster_version: str,
    dagster_extra_req: Optional[str] = None,
) -> None:
    create_venv(venv_path, python_cmd)
    pip_upgrade_base(venv_path)

    pip_install(venv_path, [f"dagster=={dagster_version}"])
    pip_install_best_effort(venv_path, DEFAULT_TOOLS)

    if dagster_extra_req:
        extra_pkgs = read_requirements_file(dagster_extra_req)
        if extra_pkgs:
            print(f"[dagster] Installing extra requirements from: {dagster_extra_req}")
            pip_install_requirements_file(venv_path, dagster_extra_req, constraint=None)


# ----------------------------
# Checks (inside venv)
# ----------------------------

def venv_pkg_version(venv_path: Path, dist: str) -> Optional[str]:
    py = str(venv_python(venv_path))
    code = (
        "from importlib.metadata import version, PackageNotFoundError\n"
        f"dist={dist!r}\n"
        "try:\n"
        "  print(version(dist))\n"
        "except PackageNotFoundError:\n"
        "  print('')\n"
    )
    r = sh([py, "-c", code], check=True)
    vv = (r.stdout or "").strip()
    return vv or None

def check_dist(venv_path: Path, dist: str, category: str, required: bool, exact: Optional[str] = None) -> CheckResult:
    v = venv_pkg_version(venv_path, dist)
    if v is None:
        return CheckResult(
            name=dist,
            ok=not required,
            detail="NOT INSTALLED" if required else "NOT INSTALLED (optional)",
            category=category
        )
    if exact:
        ok = (v == exact)
        return CheckResult(
            name=dist,
            ok=ok,
            detail=f"installed={v} required={exact}",
            installed_version=v,
            required_version=exact,
            category=category
        )
    return CheckResult(name=dist, ok=True, detail=f"installed={v}", installed_version=v, category=category)

def check_importable(venv_path: Path, module: str, category: str) -> CheckResult:
    py = str(venv_python(venv_path))
    r = sh([py, "-c", f"import {module}; print('OK')"], check=False)
    ok = (r.returncode == 0)
    detail = "import OK" if ok else ((r.stderr or r.stdout or "").strip()[:400])
    return CheckResult(name=f"import:{module}", ok=ok, detail=detail, category=category)

def check_sqlalchemy_compat_for_airflow(venv_path: Path, category: str) -> CheckResult:
    py = str(venv_python(venv_path))
    r = sh([py, "-c", "import sqlalchemy; print(sqlalchemy.__version__)"], check=False)
    ver = (r.stdout or r.stderr or "").strip()
    ok = bool(ver) and ver.startswith("1.4.")
    detail = f"installed={ver} (expected 1.4.x for Airflow 2.8.4)"
    return CheckResult(
        name="sqlalchemy",
        ok=ok,
        detail=detail,
        installed_version=ver if ver else None,
        required_version="1.4.x",
        category=category,
        install_hint="Recreate the Airflow venv using constraints; do not install Prefect into the Airflow venv."
    )

def run_checks_airflow(venv_path: Path, airflow_version: str, strict: bool) -> List[CheckResult]:
    results: List[CheckResult] = []
    py = venv_python(venv_path)
    if not py.exists():
        return [CheckResult(
            name="venv_exists",
            ok=False,
            detail=f"venv python not found at {py}",
            category="airflow/runtime",
            install_hint="Run with --setup --multi to create environments."
        )]

    v = python_version_string(str(py)) or "unknown"
    results.append(CheckResult("venv_python_version", ok=(v == "3.10.13"),
                              detail=f"venv python version={v}", installed_version=v,
                              required_version="3.10.13", category="airflow/runtime"))

    results.append(check_dist(venv_path, "apache-airflow", "airflow/orchestrator", True, exact=airflow_version))
    results.append(check_sqlalchemy_compat_for_airflow(venv_path, "airflow/runtime"))
    results.append(check_importable(venv_path, "airflow", "airflow/runtime"))

    for p in AIRFLOW_DEFAULT_PROVIDERS:
        results.append(check_dist(venv_path, p, "airflow/providers", required=strict))

    for t in DEFAULT_TOOLS:
        results.append(check_dist(venv_path, t, "airflow/tools", required=strict))

    return results

def run_checks_prefect(venv_path: Path, prefect_version: str, strict: bool) -> List[CheckResult]:
    results: List[CheckResult] = []
    py = venv_python(venv_path)
    if not py.exists():
        return [CheckResult(
            name="venv_exists",
            ok=False,
            detail=f"venv python not found at {py}",
            category="prefect/runtime",
            install_hint="Run with --setup --multi to create environments."
        )]

    v = python_version_string(str(py)) or "unknown"
    results.append(CheckResult("venv_python_version", ok=(v == "3.10.13"),
                              detail=f"venv python version={v}", installed_version=v,
                              required_version="3.10.13", category="prefect/runtime"))

    results.append(check_dist(venv_path, "prefect", "prefect/orchestrator", True, exact=prefect_version))
    results.append(check_importable(venv_path, "prefect", "prefect/runtime"))

    for t in DEFAULT_TOOLS:
        results.append(check_dist(venv_path, t, "prefect/tools", required=strict))

    return results

def run_checks_dagster(venv_path: Path, dagster_version: str, strict: bool) -> List[CheckResult]:
    results: List[CheckResult] = []
    py = venv_python(venv_path)
    if not py.exists():
        return [CheckResult(
            name="venv_exists",
            ok=False,
            detail=f"venv python not found at {py}",
            category="dagster/runtime",
            install_hint="Run with --setup --multi to create environments."
        )]

    v = python_version_string(str(py)) or "unknown"
    results.append(CheckResult("venv_python_version", ok=(v == "3.10.13"),
                              detail=f"venv python version={v}", installed_version=v,
                              required_version="3.10.13", category="dagster/runtime"))

    results.append(check_dist(venv_path, "dagster", "dagster/orchestrator", True, exact=dagster_version))
    results.append(check_importable(venv_path, "dagster", "dagster/runtime"))

    for t in DEFAULT_TOOLS:
        results.append(check_dist(venv_path, t, "dagster/tools", required=strict))

    return results


# ----------------------------
# Reporting
# ----------------------------

def print_report(results: List[CheckResult], as_json: bool) -> int:
    failures = [r for r in results if not r.ok]

    if as_json:
        payload = {
            "system": platform.platform(),
            "host_python": sys.version,
            "ok": len(failures) == 0,
            "results": [r.__dict__ for r in results],
            "failures": [r.__dict__ for r in failures],
        }
        print(json.dumps(payload, indent=2))
        return 0 if not failures else 2

    print("\n" + "=" * 88)
    print("ENV PREFLIGHT CHECK (multi-venv)")
    print("=" * 88)
    print(f"System:      {platform.platform()}")
    print(f"Host Python: {sys.version.split()[0]}")
    print("")

    cats: Dict[str, List[CheckResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)

    for cat in sorted(cats.keys()):
        print(f"[{cat}]")
        for r in cats[cat]:
            status = "✓ OK" if r.ok else "✗ FAIL"
            iv = f" ({r.installed_version})" if r.installed_version else ""
            req = f" [required={r.required_version}]" if r.required_version else ""
            print(f"  {status:7} {r.name}{iv}{req} — {r.detail}")
            if not r.ok and r.install_hint:
                print(f"          fix: {r.install_hint}")
        print("")

    if failures:
        print("RESULT: FAIL")
        return 2
    print("RESULT: PASS")
    return 0


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight check + auto setup for evaluation environments.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")

    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--python-cmd", default=None)

    parser.add_argument("--multi", action="store_true", help="Use 3 separate venvs (recommended).")

    # Legacy single-venv flag (airflow only)
    parser.add_argument("--venv-path", default=".venv-airflow-eval", help="Legacy single venv path (airflow only).")

    # Multi-venv paths
    parser.add_argument("--airflow-venv", default=".venv-airflow-eval")
    parser.add_argument("--prefect-venv", default=".venv-prefect-eval")
    parser.add_argument("--dagster-venv", default=".venv-dagster-eval")

    parser.add_argument("--airflow-version", default="2.8.4")
    parser.add_argument("--prefect-version", default="3.6.9")
    parser.add_argument("--dagster-version", default="1.12.8")

    parser.add_argument("--with-mysql-provider", action="store_true")

    # NEW: extra requirements
    parser.add_argument("--airflow-extra-req", default=None, help="Path to extra requirements file for Airflow venv")
    parser.add_argument("--prefect-extra-req", default=None, help="Path to extra requirements file for Prefect venv")
    parser.add_argument("--dagster-extra-req", default=None, help="Path to extra requirements file for Dagster venv")

    args = parser.parse_args()

    if args.setup:
        py310 = ensure_python_31013_or_fail(args.python_cmd)
        print(f"Using Python for venv creation: {py310} (version={python_version_string(py310)})")

        if args.multi:
            setup_airflow_env(
                Path(args.airflow_venv),
                py310,
                args.airflow_version,
                args.with_mysql_provider,
                airflow_extra_req=args.airflow_extra_req,
            )
            setup_prefect_env(
                Path(args.prefect_venv),
                py310,
                args.prefect_version,
                prefect_extra_req=args.prefect_extra_req,
            )
            setup_dagster_env(
                Path(args.dagster_venv),
                py310,
                args.dagster_version,
                dagster_extra_req=args.dagster_extra_req,
            )

            print("\nSetup complete. Activate with:")
            print(f"  source {args.airflow_venv}/bin/activate   # airflow")
            print(f"  source {args.prefect_venv}/bin/activate   # prefect")
            print(f"  source {args.dagster_venv}/bin/activate   # dagster")
        else:
            # Legacy single mode: airflow only
            setup_airflow_env(
                Path(args.venv_path),
                py310,
                args.airflow_version,
                args.with_mysql_provider,
                airflow_extra_req=args.airflow_extra_req,
            )
            print("\nSetup complete. Activate with:")
            print(f"  source {args.venv_path}/bin/activate")

    # Run checks
    results: List[CheckResult] = []
    if args.multi:
        results.extend(run_checks_airflow(Path(args.airflow_venv), args.airflow_version, args.strict))
        results.extend(run_checks_prefect(Path(args.prefect_venv), args.prefect_version, args.strict))
        results.extend(run_checks_dagster(Path(args.dagster_venv), args.dagster_version, args.strict))
    else:
        results.extend(run_checks_airflow(Path(args.venv_path), args.airflow_version, args.strict))

    exit_code = print_report(results, as_json=args.json)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()