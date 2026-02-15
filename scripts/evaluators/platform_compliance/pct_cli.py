#!/usr/bin/env python3
"""
Standalone CLI for penalty-free Platform Compliance Testing (PCT).

Usage:
  python scripts/evaluators/platform_compliance/pct_cli.py path/to/generated.py --print-summary --out-dir eval_results/pct
  python scripts/evaluators/platform_compliance/pct_cli.py path/to/generated.py --orchestrator airflow --out result.json

This emits a full JSON record including:
- gate checks
- 5 dimension scores
- PCT overall score
- issues (critical/major/minor/info)
- error metadata for failure cases
"""

# -------------------- bootstrap --------------------
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]  # .../scripts
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# --------------------------------------------------

import json
import logging
from datetime import datetime
from typing import Optional

from evaluators.base_evaluator import Orchestrator, BaseEvaluator
from evaluators.platform_compliance.airflow_compliance import AirflowComplianceTester
from evaluators.platform_compliance.prefect_compliance import PrefectComplianceTester
from evaluators.platform_compliance.dagster_compliance import DagsterComplianceTester


TESTERS = {
    Orchestrator.AIRFLOW: AirflowComplianceTester,
    Orchestrator.PREFECT: PrefectComplianceTester,
    Orchestrator.DAGSTER: DagsterComplianceTester,
}


def _print_pct_summary(payload: dict) -> None:
    summary_pct = payload.get("metadata", {}).get("PCT", 0.0)
    gates_passed = payload.get("gates_passed", False)
    dims = payload.get("metadata", {}).get("PCT_dimensions", {})

    issues = payload.get("issues", [])
    crit = sum(1 for i in issues if i.get("severity") == "critical")
    maj = sum(1 for i in issues if i.get("severity") == "major")
    minor = sum(1 for i in issues if i.get("severity") == "minor")

    print("\n" + "=" * 80)
    print("PCT — Platform Compliance Summary (penalty-free, gate-based)")
    print("=" * 80)
    print(f"File:         {payload.get('file_path')}")
    print(f"Orchestrator: {payload.get('orchestrator')}")
    print(f"Gates passed: {gates_passed}")
    print(f"PCT:          {float(summary_pct):.2f}/10")
    if dims:
        print("Dimensions:")
        for k, v in dims.items():
            print(f"  - {k}: {float(v):.2f}")
    print(f"Issues: total={len(issues)} critical={crit} major={maj} minor={minor}")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run penalty-free PCT platform compliance evaluation.")
    parser.add_argument("file", help="Path to generated workflow Python file")
    parser.add_argument("--orchestrator", default="auto",
                        choices=["auto", "airflow", "prefect", "dagster"],
                        help="Orchestrator to test; default auto-detect")
    parser.add_argument("--out", default=None, help="Write full JSON result to this path")
    parser.add_argument("--out-dir", default=None, help="Write JSON result into this directory (auto filename)")
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")

    file_path = Path(args.file)
    code = ""
    try:
        code = file_path.read_text(encoding="utf-8")
    except Exception:
        pass

    if args.orchestrator == "auto":
        detector = BaseEvaluator()
        orch = detector.detect_orchestrator(code)
    else:
        orch = Orchestrator(args.orchestrator)

    if orch not in TESTERS:
        # Emit a JSON record consistent with the evaluation schema
        payload = {
            "evaluation_type": "platform_compliance",
            "file_path": str(file_path),
            "orchestrator": orch.value,
            "timestamp": datetime.now().isoformat(),
            "gates_passed": False,
            "gate_checks": [],
            "metadata": {"PCT": 0.0, "error": f"No tester for orchestrator={orch.value}"},
            "scores": {},
            "issues": [],
        }
    else:
        tester = TESTERS[orch](config=None)
        result = tester.evaluate(file_path)
        payload = result.to_dict()

        # Convenience: flatten issues for analysis
        payload["issues"] = [i.to_dict() for i in result.all_issues]

    if args.print_summary:
        _print_pct_summary(payload)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Wrote: {out_path}")
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = file_path.stem
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"pct_{orch.value}_{stem}_{ts}.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Wrote: {out_path}")
        return

    # default: stdout JSON
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()