#!/usr/bin/env python3
"""
Validate GT1 mutation dataset.

Reads: mutation_benchmark/manifests/mutations.jsonl

For each entry:
  - load original + mutated code
  - run syntax checks (ast.parse)
  - run PCTRuntimeOracle on both
  - verify expected flip for that mutation id (gate-level oracle consistency)

Writes:
  - mutation_benchmark/validation_report.json
    including:
      * per-mutation-type pass rates
      * bad mutation list (to regenerate or drop)
      * overall stats

Notes:
- This validator is about correctness/consistency of GT1 labels + artifact integrity
  and about whether mutations have the expected effect on runtime gates.

- Some mutations are "gate dependent" and require the original to pass earlier gates.
  If the original fails imports (common if providers missing), flip verification becomes
  "not_evaluable" rather than "failed".
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

# --- bootstrap to make imports work when run as standalone ---
import sys
_THIS = Path(__file__).resolve()
SCRIPTS_DIR = _THIS
while SCRIPTS_DIR.name != "scripts" and SCRIPTS_DIR != SCRIPTS_DIR.parent:
    SCRIPTS_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# -----------------------------------------------------------

from evaluators.ground_truth.pct_runtime_oracle import PCTRuntimeOracle, Orchestrator, PCTGroundTruth


# -------------------------
# IO helpers
# -------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Missing jsonl: {path}")
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


def safe_read_text(path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        return path.read_text(encoding="utf-8"), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def syntax_ok(code: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# -------------------------
# Expectation model
# -------------------------

@dataclass(frozen=True)
class ExpectedEffect:
    """
    What effect do we expect from this mutation, in terms of runtime oracle gates?
    """
    kind: str  # flip_syntax | flip_imports | flip_instantiation | flip_structure | flip_dependencies | no_gate_change | unknown
    requires_original: List[str]  # list of gate preconditions required to evaluate flip (syntax/imports/instantiation/structure)
    target_gate: Optional[str] = None  # which gate must flip if evaluable


def expected_effect(mutation_id: str, orchestrator: str) -> ExpectedEffect:
    """
    Map mutation_id -> expected behavior.
    Adjust this mapping as your catalog grows.

    IMPORTANT:
    - Many mutations (security/unused/error_handling) should NOT change runtime gates.
    - Orchestrator structure mutations are intended to flip instantiation or structure.
    """
    mid = (mutation_id or "").strip()

    if mid.startswith("syntax_"):
        return ExpectedEffect(
            kind="flip_syntax",
            requires_original=["gate_syntax"],
            target_gate="gate_syntax",
        )

    if mid.startswith("import_"):
        return ExpectedEffect(
            kind="flip_imports",
            requires_original=["gate_syntax", "gate_imports"],
            target_gate="gate_imports",
        )

    # Airflow structure
    if mid == "airflow_remove_dag_definition":
        return ExpectedEffect(
            kind="flip_instantiation",
            requires_original=["gate_syntax", "gate_imports", "gate_instantiation"],
            target_gate="gate_instantiation",
        )

    if mid == "airflow_remove_tasks":
        return ExpectedEffect(
            kind="flip_structure",
            requires_original=["gate_syntax", "gate_imports", "gate_structure"],
            target_gate="gate_structure",
        )

    if mid == "airflow_remove_dependencies":
        return ExpectedEffect(
            kind="flip_dependencies",
            requires_original=["gate_syntax", "gate_imports"],
            target_gate="has_dependencies",
        )

    # Prefect structure
    if mid in ("prefect_remove_flow_decorator", "prefect_remove_flow"):
        return ExpectedEffect(
            kind="flip_structure",
            requires_original=["gate_syntax", "gate_imports", "gate_structure"],
            target_gate="gate_structure",
        )

    # Dagster structure
    if mid in ("dagster_remove_ops_and_assets", "dagster_remove_job_and_assets"):
        return ExpectedEffect(
            kind="flip_structure",
            requires_original=["gate_syntax", "gate_imports", "gate_structure"],
            target_gate="gate_structure",
        )

    # Everything else should not change runtime gates (it’s SAT-ish)
    return ExpectedEffect(
        kind="no_gate_change",
        requires_original=["gate_syntax"],  # minimal; we can still compare syntax equivalence
        target_gate=None,
    )


def get_gate_value(gt: PCTGroundTruth, name: str) -> Any:
    return getattr(gt, name, None)


# -------------------------
# Oracle caching
# -------------------------

class OracleCache:
    """
    Cache oracle results to avoid repeatedly running expensive checks.
    Keyed by absolute file path + mtime.
    """
    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    def _key(self, file_path: Path) -> str:
        fp = str(file_path.resolve())
        try:
            st = file_path.stat()
            return f"{fp}|mtime={st.st_mtime_ns}|size={st.st_size}"
        except Exception:
            return fp

    def get(self, file_path: Path) -> Optional[Dict[str, Any]]:
        return self._cache.get(self._key(file_path))

    def put(self, file_path: Path, gt_dict: Dict[str, Any]) -> None:
        self._cache[self._key(file_path)] = gt_dict

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, indent=2, default=str), encoding="utf-8")


def run_oracle(file_path: Path, orch: Orchestrator, cache: Optional[OracleCache] = None) -> PCTGroundTruth:
    if cache:
        hit = cache.get(file_path)
        if hit is not None:
            # reconstruct PCTGroundTruth from dict by passing fields directly
            return PCTGroundTruth(**hit)

    oracle = PCTRuntimeOracle(orch)
    gt = oracle.establish_ground_truth(file_path)

    if cache:
        cache.put(file_path, gt.to_dict())

    return gt


# -------------------------
# Validation logic
# -------------------------

def check_preconditions(effect: ExpectedEffect, orig: PCTGroundTruth) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for gate in effect.requires_original:
        v = get_gate_value(orig, gate)
        if v is False:
            missing.append(gate)
    return (len(missing) == 0), missing


def compare_gates_for_no_change(orig: PCTGroundTruth, mut: PCTGroundTruth) -> Tuple[bool, Dict[str, Any]]:
    """
    For SAT-like mutations, we only require that syntax and import outcomes are unchanged.
    (Instantiation/structure may be unreachable if imports fail, depending on orchestrator/env.)
    """
    details: Dict[str, Any] = {}
    ok = True

    for gate in ["gate_syntax", "gate_imports"]:
        ov = get_gate_value(orig, gate)
        mv = get_gate_value(mut, gate)
        details[gate] = {"original": ov, "mutated": mv}
        if ov != mv:
            ok = False

    return ok, details


def validate_expected_flip(effect: ExpectedEffect, orig: PCTGroundTruth, mut: PCTGroundTruth) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Returns:
      status: ok | not_evaluable | failed
      passed: bool
      details: dict
    """
    details: Dict[str, Any] = {"expected": effect.kind, "target_gate": effect.target_gate}

    # If unknown effect rules: treat as "not_evaluable" (forces you to extend mapping)
    if effect.kind == "unknown":
        details["reason"] = "No expectation mapping for mutation_id"
        return "not_evaluable", False, details

    # Preconditions
    pre_ok, missing = check_preconditions(effect, orig)
    details["preconditions_ok"] = pre_ok
    details["missing_preconditions"] = missing

    if not pre_ok:
        details["reason"] = "Original did not satisfy preconditions; cannot evaluate flip"
        return "not_evaluable", False, details

    # Evaluate per kind
    if effect.kind == "flip_syntax":
        details["gate_syntax"] = {"original": orig.gate_syntax, "mutated": mut.gate_syntax}
        passed = (orig.gate_syntax is True) and (mut.gate_syntax is False)
        if not passed:
            details["reason"] = "Expected mutated gate_syntax=False"
        return ("ok" if passed else "failed"), passed, details

    if effect.kind == "flip_imports":
        details["gate_imports"] = {"original": orig.gate_imports, "mutated": mut.gate_imports}
        details["gate_syntax"] = {"original": orig.gate_syntax, "mutated": mut.gate_syntax}

        passed = (orig.gate_imports is True) and (mut.gate_syntax is True) and (mut.gate_imports is False)
        if not passed:
            details["reason"] = "Expected original imports pass and mutated imports fail (syntax still valid)"
        return ("ok" if passed else "failed"), passed, details

    if effect.kind == "flip_instantiation":
        details["gate_instantiation"] = {"original": orig.gate_instantiation, "mutated": mut.gate_instantiation}
        details["gate_imports"] = {"original": orig.gate_imports, "mutated": mut.gate_imports}

        # We usually want imports to remain true. If mutated breaks imports, that’s an unintended extra defect.
        passed = (orig.gate_instantiation is True) and (mut.gate_imports is True) and (mut.gate_instantiation is False)
        if not passed:
            details["reason"] = "Expected mutated instantiation fail but imports still pass"
        return ("ok" if passed else "failed"), passed, details

    if effect.kind == "flip_structure":
        details["gate_structure"] = {"original": orig.gate_structure, "mutated": mut.gate_structure}
        details["gate_imports"] = {"original": orig.gate_imports, "mutated": mut.gate_imports}

        passed = (orig.gate_structure is True) and (mut.gate_imports is True) and (mut.gate_structure is False)
        if not passed:
            details["reason"] = "Expected mutated structure fail but imports still pass"
        return ("ok" if passed else "failed"), passed, details

    if effect.kind == "flip_dependencies":
        # This uses metadata fields (has_dependencies/task_count) rather than gate_structure
        details["has_dependencies"] = {"original": orig.has_dependencies, "mutated": mut.has_dependencies}
        details["task_count"] = {"original": orig.task_count, "mutated": mut.task_count}

        # Only evaluable if original actually had >1 task and dependencies
        if not (orig.task_count and orig.task_count > 1 and orig.has_dependencies is True):
            details["reason"] = "Original does not have >1 task with dependencies; cannot evaluate dependency flip"
            return "not_evaluable", False, details

        passed = (mut.has_dependencies is False)
        if not passed:
            details["reason"] = "Expected mutated has_dependencies=False"
        return ("ok" if passed else "failed"), passed, details

    if effect.kind == "no_gate_change":
        passed, comp_details = compare_gates_for_no_change(orig, mut)
        details.update({"gate_comparison": comp_details})
        if not passed:
            details["reason"] = "Expected no change in syntax/import gates"
        return ("ok" if passed else "failed"), passed, details

    details["reason"] = f"Unhandled effect kind: {effect.kind}"
    return "not_evaluable", False, details


# -------------------------
# Report aggregation
# -------------------------

def ensure_orchestrator_enum(orch_str: str) -> Orchestrator:
    v = (orch_str or "").strip().lower()
    if v == "airflow":
        return Orchestrator.AIRFLOW
    if v == "prefect":
        return Orchestrator.PREFECT
    if v == "dagster":
        return Orchestrator.DAGSTER
    return Orchestrator.UNKNOWN


def main():
    ap = argparse.ArgumentParser(description="Validate GT1 mutation dataset with PCT runtime oracle.")
    ap.add_argument("--dataset-dir", default="mutation_benchmark", help="Path to mutation benchmark root")
    ap.add_argument("--manifest", default=None, help="Override path to manifests/mutations.jsonl")
    ap.add_argument("--out", default=None, help="Override output report path (default validation_report.json in dataset-dir)")

    ap.add_argument("--limit", type=int, default=0, help="Limit number of entries (0 = all)")
    ap.add_argument("--orchestrators", nargs="*", default=None, help="Filter orchestrators (airflow/prefect/dagster)")
    ap.add_argument("--mutation-ids", nargs="*", default=None, help="Filter mutation ids")

    ap.add_argument("--cache-oracle", action="store_true", help="Cache oracle results to speed up repeated runs")
    ap.add_argument("--cache-path", default=None, help="Path to oracle cache JSON (default dataset-dir/oracle_cache.json)")

    ap.add_argument("--fail-fast", action="store_true", help="Stop on first bad mutation")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_path = Path(args.manifest) if args.manifest else dataset_dir / "manifests" / "mutations.jsonl"
    out_path = Path(args.out) if args.out else dataset_dir / "validation_report.json"

    rows = read_jsonl(manifest_path)

    orch_filter = set([o.strip().lower() for o in (args.orchestrators or [])]) if args.orchestrators else None
    mutid_filter = set(args.mutation_ids) if args.mutation_ids else None

    if orch_filter:
        rows = [r for r in rows if (r.get("orchestrator") or "").lower() in orch_filter]
    if mutid_filter:
        rows = [r for r in rows if r.get("mutation_id") in mutid_filter]

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    cache = None
    if args.cache_oracle:
        cache_path = Path(args.cache_path) if args.cache_path else dataset_dir / "oracle_cache.json"
        cache = OracleCache(cache_path=cache_path)

    report: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "timestamp": datetime.now().isoformat(),
        "total_entries": len(rows),
        "counts": {
            "ok": 0,
            "failed": 0,
            "not_evaluable": 0,
            "io_error": 0,
            "schema_error": 0,
        },
        "per_mutation_type": {},
        "bad_mutations": [],
        "samples": [],
    }

    def bump_mut(mutation_id: str, key: str):
        pm = report["per_mutation_type"].setdefault(mutation_id, {
            "total": 0,
            "ok": 0,
            "failed": 0,
            "not_evaluable": 0,
            "io_error": 0,
            "schema_error": 0,
            "pass_rate_ok_over_total": 0.0,
            "pass_rate_ok_over_evaluable": None,
        })
        pm["total"] += 1
        pm[key] += 1

    for idx, r in enumerate(rows, start=1):
        # Basic schema extraction
        file_id = r.get("file_id")
        orch_str = r.get("orchestrator")
        mutation_id = r.get("mutation_id")
        original_path = r.get("original_path")
        mutated_path = r.get("mutated_path")
        gt1_path = r.get("gt1_path")

        if not (file_id and orch_str and mutation_id and original_path and mutated_path and gt1_path):
            report["counts"]["schema_error"] += 1
            bump_mut(str(mutation_id or "unknown"), "schema_error")
            report["bad_mutations"].append({
                "reason": "schema_error_missing_fields",
                "row": r,
            })
            if args.fail_fast:
                break
            continue

        orch_enum = ensure_orchestrator_enum(orch_str)

        original_p = Path(original_path)
        mutated_p = Path(mutated_path)

        # IO checks
        orig_code, orig_err = safe_read_text(original_p)
        mut_code, mut_err = safe_read_text(mutated_p)

        if orig_err or mut_err or (orig_code is None) or (mut_code is None):
            report["counts"]["io_error"] += 1
            bump_mut(mutation_id, "io_error")
            report["bad_mutations"].append({
                "file_id": file_id,
                "orchestrator": orch_str,
                "mutation_id": mutation_id,
                "reason": "io_error",
                "original_path": str(original_p),
                "mutated_path": str(mutated_p),
                "original_error": orig_err,
                "mutated_error": mut_err,
                "gt1_path": gt1_path,
            })
            if args.fail_fast:
                break
            continue

        # Syntax checks (fast)
        orig_syntax_ok, orig_syntax_err = syntax_ok(orig_code)
        mut_syntax_ok, mut_syntax_err = syntax_ok(mut_code)

        # Oracle checks (slow)
        try:
            orig_oracle = run_oracle(original_p, orch_enum, cache=cache)
            mut_oracle = run_oracle(mutated_p, orch_enum, cache=cache)
        except Exception as e:
            report["counts"]["failed"] += 1
            bump_mut(mutation_id, "failed")
            report["bad_mutations"].append({
                "file_id": file_id,
                "orchestrator": orch_str,
                "mutation_id": mutation_id,
                "reason": f"oracle_exception_{type(e).__name__}",
                "error": str(e),
                "original_path": str(original_p),
                "mutated_path": str(mutated_p),
                "gt1_path": gt1_path,
                "syntax": {
                    "orig_ok": orig_syntax_ok,
                    "orig_err": orig_syntax_err,
                    "mut_ok": mut_syntax_ok,
                    "mut_err": mut_syntax_err,
                },
            })
            if args.fail_fast:
                break
            continue

        # Consistency check: AST syntax vs oracle syntax
        syntax_consistent = (orig_syntax_ok == bool(orig_oracle.gate_syntax)) and (mut_syntax_ok == bool(mut_oracle.gate_syntax))

        effect = expected_effect(mutation_id, orch_str)
        status, passed, flip_details = validate_expected_flip(effect, orig_oracle, mut_oracle)

        # Additional "one-defect sanity": for instantiation/structure flips, ensure imports don't change
        # This is already enforced in validate_expected_flip for flip_instantiation/flip_structure.

        # Compose entry report
        entry = {
            "file_id": file_id,
            "orchestrator": orch_str,
            "mutation_id": mutation_id,
            "paths": {
                "original": str(original_p),
                "mutated": str(mutated_p),
                "gt1_path": gt1_path,
            },
            "syntax_check": {
                "original_ok": orig_syntax_ok,
                "original_error": orig_syntax_err,
                "mutated_ok": mut_syntax_ok,
                "mutated_error": mut_syntax_err,
                "consistent_with_oracle": syntax_consistent,
            },
            "oracle_original": orig_oracle.to_dict(),
            "oracle_mutated": mut_oracle.to_dict(),
            "expected_flip": flip_details,
            "validation_status": status,   # ok | failed | not_evaluable
            "passed": bool(passed),
        }

        # Aggregate counts
        if status == "ok" and passed:
            report["counts"]["ok"] += 1
            bump_mut(mutation_id, "ok")
        elif status == "not_evaluable":
            report["counts"]["not_evaluable"] += 1
            bump_mut(mutation_id, "not_evaluable")
        else:
            report["counts"]["failed"] += 1
            bump_mut(mutation_id, "failed")
            report["bad_mutations"].append({
                "file_id": file_id,
                "orchestrator": orch_str,
                "mutation_id": mutation_id,
                "reason": flip_details.get("reason", "expected_flip_failed"),
                "original_path": str(original_p),
                "mutated_path": str(mutated_p),
                "gt1_path": gt1_path,
                "expected_flip": flip_details,
                "oracle_original": orig_oracle.to_dict(),
                "oracle_mutated": mut_oracle.to_dict(),
                "syntax_check": entry["syntax_check"],
            })
            if args.fail_fast:
                report["samples"].append(entry)
                break

        # Keep limited samples to avoid huge report
        if len(report["samples"]) < 25:
            report["samples"].append(entry)

        if idx % 50 == 0:
            print(f"[{idx}/{len(rows)}] ok={report['counts']['ok']} failed={report['counts']['failed']} not_evaluable={report['counts']['not_evaluable']}")

    # Compute pass-rate table
    for mid, stats in report["per_mutation_type"].items():
        total = stats["total"]
        ok = stats["ok"]
        not_eval = stats["not_evaluable"]
        evaluable = total - not_eval

        stats["pass_rate_ok_over_total"] = (ok / total) if total else 0.0
        stats["pass_rate_ok_over_evaluable"] = (ok / evaluable) if evaluable > 0 else None

    report["per_mutation_type_table"] = [
        {
            "mutation_id": mid,
            **stats
        }
        for mid, stats in sorted(report["per_mutation_type"].items(), key=lambda kv: kv[0])
    ]

    # Save cache + report
    if cache:
        cache.save()

    write_json(out_path, report)
    print(f"Wrote validation report: {out_path}")


if __name__ == "__main__":
    main()