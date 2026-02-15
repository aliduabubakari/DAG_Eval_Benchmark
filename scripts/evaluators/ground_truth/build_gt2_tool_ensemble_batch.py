#!/usr/bin/env python3
"""
Batch builder for GT2 (SAT tool ensemble) dataset.

Default behavior:
- Read mutation_benchmark/originals/<orch>/*.py
- Run ToolEnsembleOracle on each original file
- Write result JSON to:
    mutation_benchmark/ground_truth/gt2_tool_ensemble/<orch>/<file_id>.json

Optional:
- also run on mutated files listed in manifests/mutations.jsonl
  (not recommended for syntax-invalid mutations)

This is supplementary ground truth and should be independent of your evaluators under test.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- bootstrap for standalone run ---
import sys
_THIS = Path(__file__).resolve()
SCRIPTS_DIR = _THIS
while SCRIPTS_DIR.name != "scripts" and SCRIPTS_DIR != SCRIPTS_DIR.parent:
    SCRIPTS_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# -----------------------------------

from evaluators.ground_truth.sat_tool_ensemble import ToolEnsembleOracle


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _build_one(file_path: str, out_path: str) -> Tuple[str, str, Optional[str]]:
    """
    Worker function to build GT2 for one file.

    Returns: (file_path, out_path, error_or_none)
    """
    try:
        oracle = ToolEnsembleOracle()
        gt = oracle.establish_ground_truth(Path(file_path))
        payload = gt.to_dict()
        payload["metadata"] = payload.get("metadata", {})
        payload["metadata"]["gt_version"] = "gt2_tool_ensemble_v1"
        payload["metadata"]["built_at"] = datetime.now().isoformat()

        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return file_path, out_path, None
    except Exception as e:
        return file_path, out_path, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="Build GT2 SAT tool-ensemble ground truth in batch.")
    ap.add_argument("--dataset-dir", default="mutation_benchmark", help="mutation_benchmark root directory")
    ap.add_argument("--out-dir", default=None, help="Override output dir (default dataset-dir/ground_truth/gt2_tool_ensemble)")

    ap.add_argument("--orchestrators", nargs="*", default=["airflow", "prefect", "dagster"])
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files per scope (0 = all)")
    ap.add_argument("--workers", type=int, default=0, help="Parallel workers (0 = cpu_count-1)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    ap.add_argument("--include-mutated", action="store_true", help="Also build GT2 for mutated files from manifests/mutations.jsonl")
    ap.add_argument("--skip-syntax-mutations", action="store_true",
                    help="If include-mutated, skip syntax_* mutations (recommended)")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir) if args.out_dir else dataset_dir / "ground_truth" / "gt2_tool_ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers if args.workers and args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    jobs: List[Tuple[str, str]] = []

    # -------------------------
    # Originals
    # -------------------------
    for orch in args.orchestrators:
        orch = orch.strip().lower()
        orig_dir = dataset_dir / "originals" / orch
        if not orig_dir.exists():
            print(f"[WARN] Missing originals dir: {orig_dir}")
            continue

        files = sorted([p for p in orig_dir.glob("*.py") if p.is_file()])
        if args.limit and args.limit > 0:
            files = files[: args.limit]

        for fp in files:
            file_id = fp.stem  # originals/<orch>/<file_id>.py
            out_path = out_dir / orch / f"{file_id}.json"
            if out_path.exists() and not args.overwrite:
                continue
            jobs.append((str(fp), str(out_path)))

    # -------------------------
    # Mutated (optional)
    # -------------------------
    if args.include_mutated:
        manifest_path = dataset_dir / "manifests" / "mutations.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        rows = read_jsonl(manifest_path)
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]

        for r in rows:
            orch = (r.get("orchestrator") or "").strip().lower()
            if orch not in set([o.lower() for o in args.orchestrators]):
                continue

            mut_id = r.get("mutation_id")
            if args.skip_syntax_mutations and isinstance(mut_id, str) and mut_id.startswith("syntax_"):
                continue

            file_id = r.get("file_id")
            mutated_path = r.get("mutated_path")

            if not (file_id and mutated_path and mut_id):
                continue

            # write under gt2_tool_ensemble_mutated/<orch>/<file_id>/<mutation_id>.json
            out_path = out_dir / "mutated" / orch / file_id / f"{mut_id}.json"
            if out_path.exists() and not args.overwrite:
                continue

            jobs.append((mutated_path, str(out_path)))

    # -------------------------
    # Execute
    # -------------------------
    print(f"GT2 build jobs: {len(jobs)} (workers={workers})")
    if not jobs:
        print("Nothing to do.")
        return

    errors: List[Dict[str, Any]] = []
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_build_one, fp, op) for fp, op in jobs]
        for fut in as_completed(futures):
            fp, op, err = fut.result()
            done += 1
            if err:
                errors.append({"file_path": fp, "out_path": op, "error": err})
            if done % 25 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] errors={len(errors)}")

    summary = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "timestamp": datetime.now().isoformat(),
        "total_jobs": len(jobs),
        "errors": len(errors),
        "error_samples": errors[:20],
    }
    write_json(out_dir / "gt2_build_summary.json", summary)

    if errors:
        write_json(out_dir / "gt2_build_errors.json", errors)

    print(f"Done. Summary: {out_dir / 'gt2_build_summary.json'}")


if __name__ == "__main__":
    main()