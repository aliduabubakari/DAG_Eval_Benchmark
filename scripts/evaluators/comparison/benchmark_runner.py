#!/usr/bin/env python3
"""
Multi-Track Benchmark Runner (Originals + Mutated)

Tracks (methodology-first):
  Track A (PRIMARY): PCT Gate Prediction
    - ground truth: stored runtime oracle outputs (sidecars or oracle_cache_*.json)
    - metrics: per-gate accuracy, overall gate accuracy, all-gates-correct accuracy

  Track B (PRIMARY for issues): GT1 Injected Defect Detection
    - ground truth: GT1 mutation labels (mutation category + injected_line)
    - metrics:
        * injection_recall (did evaluator report an issue of injected category?)
        * localization_rate (did evaluator report it within ±K lines?)
        * per-mutation-id recall table

  Track C (SECONDARY): GT2 Tool Alignment
    - ground truth: GT2 tool ensemble JSON (loaded from disk)
    - metrics: precision/recall/F1 vs canonicalized tool issues (dedup by category+line)
      includes per-category PRF

Requirements:
- Must have:
    mutation_benchmark/manifests/originals.jsonl
    mutation_benchmark/manifests/mutations.jsonl
    mutation_benchmark/ground_truth/gt2_tool_ensemble/<orch>/<file_id>.json
    mutation_benchmark/ground_truth/gt1_mutations/<orch>/<file_id>/<mutation_id>.json
- Must have stored oracle GT:
    Prefer: mutation_benchmark/ground_truth/pct_runtime/... (if you later add)
    Fallback (current): mutation_benchmark/oracle_cache_<orch>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# --- bootstrap ---
import sys as _sys
_THIS = Path(__file__).resolve()
SCRIPTS_DIR = _THIS
while SCRIPTS_DIR.name != "scripts" and SCRIPTS_DIR != SCRIPTS_DIR.parent:
    SCRIPTS_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(SCRIPTS_DIR))
# -------------

from evaluators.output_schema import EvaluatorOutput, DetectedIssue, GatePrediction, EvalCategory, EvalSeverity

logger = logging.getLogger(__name__)


# =============================================================================
# Utility helpers
# =============================================================================

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


def safe_mean(xs: List[float]) -> float:
    xs = [float(x) for x in xs if x is not None]
    return float(statistics.mean(xs)) if xs else 0.0


def safe_std(xs: List[float]) -> float:
    xs = [float(x) for x in xs if x is not None]
    if len(xs) <= 1:
        return 0.0
    # sample std dev
    return float(statistics.stdev(xs))


def clamp_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def resolve_path(p: str, dataset_dir: Path) -> Path:
    """
    Resolve file paths stored in manifests:
    - If already exists as-is: use it.
    - Else try dataset_dir / p
    - Else try dataset_dir.parent / p
    """
    pp = Path(p)
    if pp.exists():
        return pp
    if not pp.is_absolute():
        p2 = dataset_dir / pp
        if p2.exists():
            return p2
        p3 = dataset_dir.parent / pp
        if p3.exists():
            return p3
    return pp


def normalize_category(cat: Any) -> str:
    return str(cat or "").strip().lower()


def categories_related(c1: str, c2: str) -> bool:
    """
    Related-category groups for matching tolerance.
    Used for GT2 matching only (secondary track).
    """
    groups = [
        {"style", "naming", "documentation"},
        {"unused", "undefined"},
        {"error_handling", "best_practice"},
    ]
    for g in groups:
        if c1 in g and c2 in g:
            return True
    return False


def compute_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    tp = int(tp); fp = int(fp); fn = int(fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# =============================================================================
# Oracle GT loader (stored, no recomputation)
# =============================================================================

class OracleStore:
    """
    Loads stored PCT runtime oracle ground truth from disk.

    Preferred (future):
      dataset_dir/ground_truth/pct_runtime/originals/<orch>/<file_id>.json
      dataset_dir/ground_truth/pct_runtime/mutated/<orch>/<file_id>/<mutation_id>.json

    Current fallback:
      dataset_dir/oracle_cache_<orch>.json produced by validate_gt1_dataset.py

    This store never runs the oracle; it only reads.
    """

    def __init__(
        self,
        dataset_dir: Path,
        pct_gt_dir: Optional[Path] = None,
        oracle_cache_paths: Optional[Dict[str, Path]] = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.pct_gt_dir = Path(pct_gt_dir) if pct_gt_dir else None

        # path -> gt dict
        self._by_path: Dict[str, Dict[str, Any]] = {}

        # Load cache fallbacks
        oracle_cache_paths = oracle_cache_paths or {}
        for orch, cache_path in oracle_cache_paths.items():
            if cache_path and cache_path.exists():
                self._load_oracle_cache(cache_path)

    def _load_oracle_cache(self, cache_path: Path) -> None:
        """
        oracle_cache_*.json format: dict[key] = PCTGroundTruth.to_dict()
        Key includes absolute path + mtime/size, but value contains file_path.
        We index by resolved file_path string.
        """
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed reading oracle cache {cache_path}: {e}")
            return

        if not isinstance(data, dict):
            return

        loaded = 0
        for _k, v in data.items():
            if not isinstance(v, dict):
                continue
            fp = v.get("file_path")
            if not fp:
                continue
            # index by multiple keys
            p = Path(fp)
            self._by_path[str(p)] = v
            try:
                self._by_path[str(p.resolve())] = v
            except Exception:
                pass
            loaded += 1

        logger.info(f"Loaded oracle cache {cache_path}: {loaded} entries")

    def get_for_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Return oracle GT dict for given file_path, if available.
        """
        p = Path(file_path)
        if str(p) in self._by_path:
            return self._by_path[str(p)]
        try:
            rp = str(p.resolve())
            if rp in self._by_path:
                return self._by_path[rp]
        except Exception:
            pass

        # Optional stable gt dir lookup (if you later add stable sidecars)
        if self.pct_gt_dir:
            # We cannot infer file_id from path reliably here without manifest context,
            # so stable sidecar lookup should be done by callers via get_for_ids().
            pass

        return None

    def get_for_ids(
        self,
        *,
        kind: str,              # "original" or "mutated"
        orchestrator: str,
        file_id: str,
        mutation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Stable sidecar lookup if pct_gt_dir exists.
        If not found, returns None (caller may fallback to get_for_file()).
        """
        if not self.pct_gt_dir:
            return None

        orch = orchestrator.strip().lower()
        if kind == "original":
            p = self.pct_gt_dir / "originals" / orch / f"{file_id}.json"
        else:
            if not mutation_id:
                return None
            p = self.pct_gt_dir / "mutated" / orch / file_id / f"{mutation_id}.json"

        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None


# =============================================================================
# GT2 loader (tool ensemble) + canonicalization
# =============================================================================

class GT2Store:
    def __init__(self, gt2_dir: Path):
        self.gt2_dir = Path(gt2_dir)
        self._cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def load_issues(self, orchestrator: str, file_id: str) -> Optional[List[Dict[str, Any]]]:
        key = (orchestrator.lower(), file_id)
        if key in self._cache:
            return self._cache[key]

        p = self.gt2_dir / orchestrator.lower() / f"{file_id}.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

        issues = data.get("issues", []) or []
        if not isinstance(issues, list):
            issues = []

        # canonicalize + dedupe by (category,line)
        canon = self.canonicalize_gt2_issues(issues)
        self._cache[key] = canon
        return canon

    def canonicalize_gt2_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for it in issues:
            if not isinstance(it, dict):
                continue
            cat = normalize_category(it.get("category"))
            line = it.get("line")
            line_i = None
            try:
                line_i = int(line) if line is not None else None
            except Exception:
                line_i = None

            # Primary key: (category, line)
            # Fallback if line missing: include message prefix to avoid collapsing everything.
            if line_i is None:
                msg = str(it.get("message", "")).strip().lower()
                key = (cat, None, msg[:80])
            else:
                key = (cat, line_i)

            if key in seen:
                continue
            seen.add(key)

            out.append({
                "category": cat,
                "line": line_i,
                "message": it.get("message"),
                "severity": normalize_category(it.get("severity")),
                "tool": it.get("tool"),
                "code": it.get("code"),
            })
        return out


# =============================================================================
# GT1 loader (mutation labels)
# =============================================================================

class GT1Store:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def load_gt1(self, gt1_path: Path) -> Optional[Dict[str, Any]]:
        p = Path(gt1_path)
        if str(p) in self._cache:
            return self._cache[str(p)]
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        self._cache[str(p)] = d
        return d


# =============================================================================
# Metric accumulators
# =============================================================================

@dataclass
class GateMetricCounts:
    total_items: int = 0
    total_gates: int = 0
    correct_gates: int = 0
    correct_all_gates_items: int = 0
    per_gate_total: Dict[str, int] = field(default_factory=lambda: {"syntax_valid": 0, "imports_resolve": 0, "instantiates": 0, "has_structure": 0})
    per_gate_correct: Dict[str, int] = field(default_factory=lambda: {"syntax_valid": 0, "imports_resolve": 0, "instantiates": 0, "has_structure": 0})

    def add_item(self, gt: Dict[str, Any], pred: GatePrediction) -> None:
        mapping = [
            ("syntax_valid", bool(gt.get("gate_syntax"))),
            ("imports_resolve", bool(gt.get("gate_imports"))),
            ("instantiates", bool(gt.get("gate_instantiation"))),
            ("has_structure", bool(gt.get("gate_structure"))),
        ]

        all_ok = True
        self.total_items += 1

        for gate_name, actual in mapping:
            self.per_gate_total[gate_name] += 1
            self.total_gates += 1

            predicted = bool(getattr(pred, gate_name, True))
            ok = (predicted == actual)
            if ok:
                self.per_gate_correct[gate_name] += 1
                self.correct_gates += 1
            else:
                all_ok = False

        if all_ok:
            self.correct_all_gates_items += 1

    def summary(self) -> Dict[str, Any]:
        overall_gate_accuracy = self.correct_gates / self.total_gates if self.total_gates else 0.0
        all_gates_accuracy = self.correct_all_gates_items / self.total_items if self.total_items else 0.0

        per_gate_acc = {}
        for g in self.per_gate_total.keys():
            tot = self.per_gate_total[g]
            cor = self.per_gate_correct[g]
            per_gate_acc[g] = (cor / tot) if tot else None

        return {
            "items_scored": self.total_items,
            "overall_gate_accuracy": float(overall_gate_accuracy),
            "all_gates_correct_accuracy": float(all_gates_accuracy),
            "per_gate_accuracy": per_gate_acc,
            "per_gate_counts": {
                g: {"correct": self.per_gate_correct[g], "total": self.per_gate_total[g]}
                for g in self.per_gate_total.keys()
            },
        }


@dataclass
class GT1MetricCounts:
    total_mutations: int = 0
    detected: int = 0
    localized: int = 0
    detected_and_line_present: int = 0  # how often evaluator provided line for matching issue
    per_mutation_id: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_item(
        self,
        *,
        mutation_id: str,
        injected_category: str,
        injected_line: Optional[int],
        predicted_issues: List[DetectedIssue],
        line_tolerance: int,
    ) -> None:
        self.total_mutations += 1
        injected_category = normalize_category(injected_category)

        # exact category match for GT1 recall (primary)
        matching = [iss for iss in predicted_issues if normalize_category(iss.category) == injected_category]

        detected = bool(matching)
        localized = False

        if detected:
            self.detected += 1

            if injected_line is not None:
                # Localization: any matching issue with line within tolerance
                for iss in matching:
                    if iss.line is None:
                        continue
                    try:
                        if abs(int(iss.line) - int(injected_line)) <= int(line_tolerance):
                            localized = True
                            break
                    except Exception:
                        continue

            if any(iss.line is not None for iss in matching):
                self.detected_and_line_present += 1

        if localized:
            self.localized += 1

        # per mutation id table
        pm = self.per_mutation_id.setdefault(mutation_id, {"total": 0, "detected": 0, "localized": 0})
        pm["total"] += 1
        pm["detected"] += 1 if detected else 0
        pm["localized"] += 1 if localized else 0

    def summary(self) -> Dict[str, Any]:
        recall = self.detected / self.total_mutations if self.total_mutations else 0.0
        localization_rate = self.localized / self.total_mutations if self.total_mutations else 0.0
        localization_given_detected = self.localized / self.detected if self.detected else 0.0

        per_mut = {}
        for mid, st in self.per_mutation_id.items():
            tot = st["total"]
            det = st["detected"]
            loc = st["localized"]
            per_mut[mid] = {
                "total": tot,
                "detected": det,
                "localized": loc,
                "recall": (det / tot) if tot else None,
                "localization_rate": (loc / tot) if tot else None,
            }

        return {
            "mutations_scored": self.total_mutations,
            "injection_recall": float(recall),
            "localization_rate": float(localization_rate),
            "localization_given_detected": float(localization_given_detected),
            "detected_with_line_present": int(self.detected_and_line_present),
            "per_mutation_id": per_mut,
        }


@dataclass
class GT2MetricCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_category: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add_counts(self, cat: str, tp: int, fp: int, fn: int) -> None:
        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        cat = normalize_category(cat)
        pc = self.per_category.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
        pc["tp"] += int(tp)
        pc["fp"] += int(fp)
        pc["fn"] += int(fn)

    def summary(self) -> Dict[str, Any]:
        overall = compute_prf(self.tp, self.fp, self.fn)
        per_cat = {c: {**v, **compute_prf(v["tp"], v["fp"], v["fn"])} for c, v in self.per_category.items()}
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "overall": overall,
            "per_category": per_cat,
        }


# =============================================================================
# Matching logic for GT2
# =============================================================================

def dedupe_detected_issues(issues: List[DetectedIssue]) -> List[DetectedIssue]:
    """
    Dedupe predicted issues by (category,line,msg_prefix).
    """
    seen = set()
    out: List[DetectedIssue] = []
    for iss in issues:
        cat = normalize_category(iss.category)
        line = iss.line
        line_i = None
        try:
            line_i = int(line) if line is not None else None
        except Exception:
            line_i = None
        msg = (iss.message or "").strip().lower()
        key = (cat, line_i, msg[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(iss)
    return out


def match_gt2(
    detected: List[DetectedIssue],
    gt: List[Dict[str, Any]],
    *,
    line_tolerance: int = 2,
    scored_categories: Optional[Set[str]] = None,
    strict_line: bool = True,
) -> Tuple[int, int, int, Dict[str, Dict[str, int]]]:
    """
    Greedy matching between detected issues and gt2 issues.

    Returns:
      tp, fp, fn, per_category_counts
    """
    scored_categories = scored_categories or set()

    # filter categories
    det = [d for d in detected if normalize_category(d.category) in scored_categories]
    gt_ = [g for g in gt if normalize_category(g.get("category")) in scored_categories]

    det = dedupe_detected_issues(det)

    gt_matched = set()
    det_matched = set()

    for di, d in enumerate(det):
        dcat = normalize_category(d.category)
        dline = d.line
        dline_i = None
        try:
            dline_i = int(dline) if dline is not None else None
        except Exception:
            dline_i = None

        for gi, g in enumerate(gt_):
            if gi in gt_matched:
                continue
            gcat = normalize_category(g.get("category"))
            gline_i = g.get("line")

            cat_ok = (dcat == gcat) or categories_related(dcat, gcat)

            if not cat_ok:
                continue

            if strict_line:
                # Require lines (or tolerate missing gt line)
                if gline_i is None or dline_i is None:
                    # If either is missing, treat as non-match in strict mode
                    continue
                if abs(int(dline_i) - int(gline_i)) > int(line_tolerance):
                    continue
            else:
                # relaxed: ignore line constraints
                pass

            # matched
            gt_matched.add(gi)
            det_matched.add(di)
            break

    tp = len(gt_matched)
    fp = len(det) - len(det_matched)
    fn = len(gt_) - len(gt_matched)

    # per-category breakdown: approximate by counting matched gt categories as tp and unmatched as fn,
    # and fp by detected category.
    per_cat: Dict[str, Dict[str, int]] = {}
    for gi, g in enumerate(gt_):
        c = normalize_category(g.get("category"))
        pc = per_cat.setdefault(c, {"tp": 0, "fp": 0, "fn": 0})
        if gi in gt_matched:
            pc["tp"] += 1
        else:
            pc["fn"] += 1

    for di, d in enumerate(det):
        c = normalize_category(d.category)
        pc = per_cat.setdefault(c, {"tp": 0, "fp": 0, "fn": 0})
        if di not in det_matched:
            pc["fp"] += 1

    return tp, fp, fn, per_cat


# =============================================================================
# Main benchmark runner
# =============================================================================

class MultiTrackBenchmarkRunner:
    """
    Runs evaluators on:
      - originals (Track A + Track C)
      - mutated   (Track A + Track B)
    in a single end-to-end benchmark execution.

    All GT is loaded from disk.
    """

    DEFAULT_GT2_SCORED_CATEGORIES = {
        "syntax", "import", "type_error", "security", "complexity",
        "style", "naming", "documentation", "error_handling",
        "best_practice", "unused", "undefined",
    }

    def __init__(
        self,
        dataset_dir: Path,
        evaluators: List[Any],
        *,
        gt2_dir: Optional[Path] = None,
        pct_gt_dir: Optional[Path] = None,
        oracle_cache_paths: Optional[Dict[str, Path]] = None,
        runs_per_file: int = 1,
        line_tolerance_gt2: int = 2,
        line_tolerance_gt1: int = 2,
        limit_originals: int = 0,
        limit_mutations: int = 0,
        orchestrators: Optional[List[str]] = None,
        store_item_results: bool = False,
        max_item_results: int = 5000,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.evaluators = evaluators
        self.runs_per_file = int(runs_per_file)

        self.line_tolerance_gt2 = int(line_tolerance_gt2)
        self.line_tolerance_gt1 = int(line_tolerance_gt1)

        self.limit_originals = int(limit_originals)
        self.limit_mutations = int(limit_mutations)

        self.orchestrators_filter = set([o.strip().lower() for o in orchestrators]) if orchestrators else None

        self.store_item_results = bool(store_item_results)
        self.max_item_results = int(max_item_results)

        self.gt1 = GT1Store()
        self.gt2 = GT2Store(gt2_dir if gt2_dir else (self.dataset_dir / "ground_truth" / "gt2_tool_ensemble"))

        self.oracle = OracleStore(
            dataset_dir=self.dataset_dir,
            pct_gt_dir=pct_gt_dir,
            oracle_cache_paths=oracle_cache_paths or {},
        )

        self.gt2_scored_categories = set(self.DEFAULT_GT2_SCORED_CATEGORIES)

        # load manifests
        self.original_rows = self._load_original_manifest()
        self.mutation_rows = self._load_mutation_manifest()

    def _load_original_manifest(self) -> List[Dict[str, Any]]:
        p = self.dataset_dir / "manifests" / "originals.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing originals manifest: {p}")

        rows = read_jsonl(p)
        out = []
        for r in rows:
            orch = normalize_category(r.get("orchestrator"))
            if self.orchestrators_filter and orch not in self.orchestrators_filter:
                continue
            out.append(r)

        if self.limit_originals > 0:
            out = out[: self.limit_originals]

        return out

    def _load_mutation_manifest(self) -> List[Dict[str, Any]]:
        p = self.dataset_dir / "manifests" / "mutations.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing mutations manifest: {p}")

        rows = read_jsonl(p)
        out = []
        for r in rows:
            orch = normalize_category(r.get("orchestrator"))
            if self.orchestrators_filter and orch not in self.orchestrators_filter:
                continue
            out.append(r)

        if self.limit_mutations > 0:
            out = out[: self.limit_mutations]

        return out

    # ---------------------------------------------------------------------

    def run(self, output_dir: Path) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_dir / f"benchmark_multitrack_{timestamp}.json"

        # per evaluator: list of per-run summaries
        per_eval_run_summaries: Dict[str, List[Dict[str, Any]]] = {}
        per_eval_item_results: Dict[str, List[Dict[str, Any]]] = {}

        for run_idx in range(1, self.runs_per_file + 1):
            logger.info(f"=== RUN {run_idx}/{self.runs_per_file} ===")

            for evaluator in self.evaluators:
                ev_name = getattr(evaluator, "NAME", evaluator.__class__.__name__)
                per_eval_run_summaries.setdefault(ev_name, [])
                per_eval_item_results.setdefault(ev_name, [])

                logger.info(f"[run={run_idx}] Evaluator: {ev_name}")

                run_summary, item_results = self._run_one_evaluator_one_run(evaluator=evaluator, run_idx=run_idx)

                per_eval_run_summaries[ev_name].append(run_summary)

                if self.store_item_results:
                    # guard size
                    if len(per_eval_item_results[ev_name]) < self.max_item_results:
                        per_eval_item_results[ev_name].extend(item_results[: max(0, self.max_item_results - len(per_eval_item_results[ev_name]))])

        # Aggregate mean/std across runs (stability)
        leaderboard = self._aggregate_across_runs(per_eval_run_summaries)

        payload: Dict[str, Any] = {
            "timestamp": timestamp,
            "dataset_dir": str(self.dataset_dir),
            "runs_per_file": self.runs_per_file,
            "counts": {
                "originals": len(self.original_rows),
                "mutations": len(self.mutation_rows),
                "total_items": len(self.original_rows) + len(self.mutation_rows),
            },
            "config": {
                "line_tolerance_gt2": self.line_tolerance_gt2,
                "line_tolerance_gt1": self.line_tolerance_gt1,
                "gt2_scored_categories": sorted(list(self.gt2_scored_categories)),
                "orchestrators_filter": sorted(list(self.orchestrators_filter)) if self.orchestrators_filter else None,
            },
            "leaderboard": leaderboard,
            "per_run": per_eval_run_summaries,
        }

        if self.store_item_results:
            payload["item_results"] = per_eval_item_results

        write_json(result_path, payload)
        logger.info(f"Wrote benchmark results: {result_path}")
        return payload

    # ---------------------------------------------------------------------

    def _run_one_evaluator_one_run(self, evaluator: Any, run_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        ev_name = getattr(evaluator, "NAME", evaluator.__class__.__name__)

        gate_counts = GateMetricCounts()
        gt1_counts = GT1MetricCounts()
        gt2_counts_strict = GT2MetricCounts()
        gt2_counts_relaxed = GT2MetricCounts()

        item_results: List[Dict[str, Any]] = []

        total_tokens = 0
        total_time_ms = 0.0
        total_eval_calls = 0

        missing_oracle = 0
        missing_gt2 = 0
        missing_gt1 = 0

        # ---- originals: Track A + Track C ----
        for r in self.original_rows:
            file_id = r.get("file_id")
            orch = normalize_category(r.get("orchestrator"))
            original_path = resolve_path(r.get("original_path"), self.dataset_dir)

            output = self._safe_eval(evaluator, original_path)
            total_eval_calls += 1
            total_tokens += int(output.tokens_used or 0)
            total_time_ms += float(output.execution_time_ms or 0.0)

            # Gate GT
            oracle_gt = self._load_oracle_gt(kind="original", orch=orch, file_id=file_id, mutation_id=None, file_path=original_path)
            if oracle_gt is not None:
                pred_gates = output.gate_predictions or GatePrediction()
                gate_counts.add_item(oracle_gt, pred_gates)
            else:
                missing_oracle += 1

            # GT2 tool alignment (secondary)
            gt2_issues = self.gt2.load_issues(orch, file_id)
            if gt2_issues is not None:
                tp, fp, fn, per_cat = match_gt2(
                    output.issues or [],
                    gt2_issues,
                    line_tolerance=self.line_tolerance_gt2,
                    scored_categories=self.gt2_scored_categories,
                    strict_line=True,
                )
                gt2_counts_strict.tp += tp
                gt2_counts_strict.fp += fp
                gt2_counts_strict.fn += fn
                for c, counts in per_cat.items():
                    gt2_counts_strict.add_counts(c, counts["tp"], counts["fp"], counts["fn"])

                # relaxed (line ignored)
                tp2, fp2, fn2, per_cat2 = match_gt2(
                    output.issues or [],
                    gt2_issues,
                    line_tolerance=self.line_tolerance_gt2,
                    scored_categories=self.gt2_scored_categories,
                    strict_line=False,
                )
                gt2_counts_relaxed.tp += tp2
                gt2_counts_relaxed.fp += fp2
                gt2_counts_relaxed.fn += fn2
                for c, counts in per_cat2.items():
                    gt2_counts_relaxed.add_counts(c, counts["tp"], counts["fp"], counts["fn"])
            else:
                missing_gt2 += 1

            if self.store_item_results:
                item_results.append({
                    "run_idx": run_idx,
                    "evaluator": ev_name,
                    "item_kind": "original",
                    "orchestrator": orch,
                    "file_id": file_id,
                    "mutation_id": None,
                    "path": str(original_path),
                    "tokens_used": int(output.tokens_used or 0),
                    "execution_time_ms": float(output.execution_time_ms or 0.0),
                    "gate_predictions": output.gate_predictions.to_dict() if output.gate_predictions else None,
                })

        # ---- mutated: Track A + Track B ----
        for r in self.mutation_rows:
            file_id = r.get("file_id")
            orch = normalize_category(r.get("orchestrator"))
            mutation_id = r.get("mutation_id")
            mutated_path = resolve_path(r.get("mutated_path"), self.dataset_dir)
            gt1_path = resolve_path(r.get("gt1_path"), self.dataset_dir)

            output = self._safe_eval(evaluator, mutated_path)
            total_eval_calls += 1
            total_tokens += int(output.tokens_used or 0)
            total_time_ms += float(output.execution_time_ms or 0.0)

            # Gate GT
            oracle_gt = self._load_oracle_gt(kind="mutated", orch=orch, file_id=file_id, mutation_id=mutation_id, file_path=mutated_path)
            if oracle_gt is not None:
                pred_gates = output.gate_predictions or GatePrediction()
                gate_counts.add_item(oracle_gt, pred_gates)
            else:
                missing_oracle += 1

            # GT1 injected defect detection (primary issue track)
            gt1 = self.gt1.load_gt1(gt1_path)
            if gt1 and isinstance(gt1, dict) and gt1.get("status") == "ok":
                injected_category = normalize_category(((gt1.get("mutation") or {}).get("category")))
                injected_line = gt1.get("injected_line")
                injected_line_i = None
                try:
                    injected_line_i = int(injected_line) if injected_line is not None else None
                except Exception:
                    injected_line_i = None

                gt1_counts.add_item(
                    mutation_id=str(mutation_id),
                    injected_category=injected_category,
                    injected_line=injected_line_i,
                    predicted_issues=output.issues or [],
                    line_tolerance=self.line_tolerance_gt1,
                )
            else:
                missing_gt1 += 1

            if self.store_item_results:
                item_results.append({
                    "run_idx": run_idx,
                    "evaluator": ev_name,
                    "item_kind": "mutated",
                    "orchestrator": orch,
                    "file_id": file_id,
                    "mutation_id": mutation_id,
                    "path": str(mutated_path),
                    "gt1_path": str(gt1_path),
                    "tokens_used": int(output.tokens_used or 0),
                    "execution_time_ms": float(output.execution_time_ms or 0.0),
                    "gate_predictions": output.gate_predictions.to_dict() if output.gate_predictions else None,
                })

        # Build run summary
        gate_summary = gate_counts.summary()
        gt1_summary = gt1_counts.summary()
        gt2_summary_strict = gt2_counts_strict.summary()
        gt2_summary_relaxed = gt2_counts_relaxed.summary()

        avg_tokens = (total_tokens / total_eval_calls) if total_eval_calls else 0.0
        avg_time = (total_time_ms / total_eval_calls) if total_eval_calls else 0.0

        run_summary = {
            "run_idx": run_idx,
            "evaluator": ev_name,
            "counts": {
                "eval_calls": total_eval_calls,
                "missing_oracle": missing_oracle,
                "missing_gt1": missing_gt1,
                "missing_gt2": missing_gt2,
            },
            "cost": {
                "total_tokens": int(total_tokens),
                "avg_tokens_per_item": float(avg_tokens),
                "total_execution_time_ms": float(total_time_ms),
                "avg_execution_time_ms_per_item": float(avg_time),
            },
            "track_A_pct_gates": gate_summary,
            "track_B_gt1_injection": gt1_summary,
            "track_C_gt2_tool_alignment": {
                "strict": gt2_summary_strict,
                "relaxed": gt2_summary_relaxed,
            },
        }

        return run_summary, item_results

    def _safe_eval(self, evaluator: Any, path: Path) -> EvaluatorOutput:
        """
        Robust evaluate wrapper. Never returns None.
        """
        try:
            out = evaluator.evaluate(Path(path))
            if not isinstance(out, EvaluatorOutput):
                # Some user evaluators may return dict; handle minimally
                eo = EvaluatorOutput(evaluator_name=getattr(evaluator, "NAME", "unknown"), file_path=str(path))
                return eo
            return out
        except Exception as e:
            # Return an error-shaped EvaluatorOutput
            eo = EvaluatorOutput(evaluator_name=getattr(evaluator, "NAME", "unknown"), file_path=str(path))
            eo.issues.append(DetectedIssue(
                category=EvalCategory.ERROR_HANDLING.value,
                severity=EvalSeverity.INFO.value,
                message=f"Evaluator crashed: {type(e).__name__}: {e}",
                line=None,
                confidence=1.0,
                source="benchmark_runner",
            ))
            eo.gate_predictions = GatePrediction()
            return eo

    def _load_oracle_gt(self, *, kind: str, orch: str, file_id: str, mutation_id: Optional[str], file_path: Path) -> Optional[Dict[str, Any]]:
        # Prefer stable sidecar (if exists)
        stable = self.oracle.get_for_ids(kind=kind, orchestrator=orch, file_id=file_id, mutation_id=mutation_id)
        if stable is not None:
            return stable
        # fallback to oracle cache by file path
        return self.oracle.get_for_file(file_path)

    # ---------------------------------------------------------------------

    def _aggregate_across_runs(self, per_eval_run_summaries: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compute mean/std across runs for each evaluator.
        """
        leaderboard: Dict[str, Any] = {}

        for ev_name, runs in per_eval_run_summaries.items():
            # Extract time/tokens
            tokens_means = [r.get("cost", {}).get("avg_tokens_per_item", 0.0) for r in runs]
            time_means = [r.get("cost", {}).get("avg_execution_time_ms_per_item", 0.0) for r in runs]

            # Track A gate accuracy
            gate_acc = [r.get("track_A_pct_gates", {}).get("overall_gate_accuracy", 0.0) for r in runs]
            all_gates_acc = [r.get("track_A_pct_gates", {}).get("all_gates_correct_accuracy", 0.0) for r in runs]

            # Track B injection recall + localization
            inj_recall = [r.get("track_B_gt1_injection", {}).get("injection_recall", 0.0) for r in runs]
            inj_loc = [r.get("track_B_gt1_injection", {}).get("localization_rate", 0.0) for r in runs]

            # Track C strict F1 (secondary)
            gt2_f1_strict = [r.get("track_C_gt2_tool_alignment", {}).get("strict", {}).get("overall", {}).get("f1", 0.0) for r in runs]
            gt2_f1_relaxed = [r.get("track_C_gt2_tool_alignment", {}).get("relaxed", {}).get("overall", {}).get("f1", 0.0) for r in runs]

            # Stability = std devs of key metrics
            stability = {
                "gate_accuracy_std": safe_std(gate_acc),
                "all_gates_correct_std": safe_std(all_gates_acc),
                "gt1_injection_recall_std": safe_std(inj_recall),
                "gt1_localization_std": safe_std(inj_loc),
                "gt2_f1_strict_std": safe_std(gt2_f1_strict),
                "tokens_std": safe_std(tokens_means),
                "latency_ms_std": safe_std(time_means),
            }

            leaderboard[ev_name] = {
                "runs": len(runs),

                "track_A_pct_gates": {
                    "overall_gate_accuracy_mean": safe_mean(gate_acc),
                    "overall_gate_accuracy_std": safe_std(gate_acc),
                    "all_gates_correct_accuracy_mean": safe_mean(all_gates_acc),
                    "all_gates_correct_accuracy_std": safe_std(all_gates_acc),
                    # per-gate accuracy averaged across runs
                    "per_gate_accuracy_mean": self._mean_per_gate_accuracy(runs),
                },

                "track_B_gt1_injection": {
                    "injection_recall_mean": safe_mean(inj_recall),
                    "injection_recall_std": safe_std(inj_recall),
                    "localization_rate_mean": safe_mean(inj_loc),
                    "localization_rate_std": safe_std(inj_loc),
                    # merge per-mutation-id tables (from run 1 only, as structure is same)
                    "per_mutation_id": runs[0].get("track_B_gt1_injection", {}).get("per_mutation_id", {}) if runs else {},
                },

                "track_C_gt2_tool_alignment": {
                    "f1_strict_mean": safe_mean(gt2_f1_strict),
                    "f1_strict_std": safe_std(gt2_f1_strict),
                    "f1_relaxed_mean": safe_mean(gt2_f1_relaxed),
                    "f1_relaxed_std": safe_std(gt2_f1_relaxed),
                    # include per-category (from run 1)
                    "per_category_strict": runs[0].get("track_C_gt2_tool_alignment", {}).get("strict", {}).get("per_category", {}) if runs else {},
                },

                "cost": {
                    "avg_tokens_per_item_mean": safe_mean(tokens_means),
                    "avg_tokens_per_item_std": safe_std(tokens_means),
                    "avg_latency_ms_per_item_mean": safe_mean(time_means),
                    "avg_latency_ms_per_item_std": safe_std(time_means),
                },

                "stability": stability,

                # Convenience: Pareto vector (you can plot later)
                "pareto": {
                    "primary_gate_accuracy": safe_mean(gate_acc),
                    "primary_gt1_recall": safe_mean(inj_recall),
                    "secondary_gt2_f1_strict": safe_mean(gt2_f1_strict),
                    "avg_tokens_per_item": safe_mean(tokens_means),
                    "avg_latency_ms_per_item": safe_mean(time_means),
                },
            }

        return leaderboard

    def _mean_per_gate_accuracy(self, runs: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """
        Average per-gate accuracy across runs (Track A).
        """
        gates = ["syntax_valid", "imports_resolve", "instantiates", "has_structure"]
        vals: Dict[str, List[float]] = {g: [] for g in gates}

        for r in runs:
            per_gate = (r.get("track_A_pct_gates", {}) or {}).get("per_gate_accuracy", {}) or {}
            for g in gates:
                v = per_gate.get(g)
                if v is None:
                    continue
                vals[g].append(float(v))

        return {g: (safe_mean(vals[g]) if vals[g] else None) for g in gates}


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run multi-track benchmark (originals + mutated) with stored GT.")
    parser.add_argument("--dataset-dir", default="mutation_benchmark")
    parser.add_argument("--output-dir", default="benchmark_results")

    # Updated evaluator choices:
    # - deterministic_tools: tool-based deterministic evaluator (upper-bound for tool alignment, Track C)
    # - deterministic_heuristic: AST/regex-only deterministic evaluator (non-circular baseline)
    # Keep "deterministic" as an alias for deterministic_tools for backward compatibility.
    parser.add_argument(
        "--evaluators",
        nargs="+",
        default=["deterministic_heuristic", "deterministic_tools"],
        choices=[
            "deterministic",              # alias -> deterministic_tools
            "deterministic_tools",
            "deterministic_heuristic",
            "llm_single",
            "llm_multi",
            "hybrid",
        ],
        help="Which evaluator systems to run"
    )

    parser.add_argument("--config-llm", default=None, help="LLM config path (for llm_* and hybrid)")
    parser.add_argument("--model-alias", default=None, help="DeepInfra model key (e.g., Qwen3-Coder)")

    parser.add_argument("--runs-per-file", type=int, default=1, help="Repeat each evaluator run for stability stats")
    parser.add_argument("--limit-originals", type=int, default=0)
    parser.add_argument("--limit-mutations", type=int, default=0)
    parser.add_argument("--orchestrators", nargs="*", default=None, help="Filter orchestrators: airflow prefect dagster")

    parser.add_argument("--gt2-dir", default=None, help="Override GT2 dir (default dataset-dir/ground_truth/gt2_tool_ensemble)")
    parser.add_argument("--pct-gt-dir", default=None, help="Optional stable PCT GT dir (default none; uses oracle_cache_*.json)")

    parser.add_argument("--oracle-cache-airflow", default=None)
    parser.add_argument("--oracle-cache-prefect", default=None)
    parser.add_argument("--oracle-cache-dagster", default=None)

    parser.add_argument("--line-tolerance-gt2", type=int, default=2)
    parser.add_argument("--line-tolerance-gt1", type=int, default=2)

    parser.add_argument("--store-item-results", action="store_true", help="Store per-item results (can be large)")
    parser.add_argument("--max-item-results", type=int, default=5000)

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")

    dataset_dir = Path(args.dataset_dir)

    gt2_dir = Path(args.gt2_dir) if args.gt2_dir else (dataset_dir / "ground_truth" / "gt2_tool_ensemble")
    pct_gt_dir = Path(args.pct_gt_dir) if args.pct_gt_dir else None

    # Oracle caches: default to dataset_dir/oracle_cache_<orch>.json if present
    oracle_cache_paths = {
        "airflow": Path(args.oracle_cache_airflow) if args.oracle_cache_airflow else (dataset_dir / "oracle_cache_airflow.json"),
        "prefect": Path(args.oracle_cache_prefect) if args.oracle_cache_prefect else (dataset_dir / "oracle_cache_prefect.json"),
        "dagster": Path(args.oracle_cache_dagster) if args.oracle_cache_dagster else (dataset_dir / "oracle_cache_dagster.json"),
    }
    oracle_cache_paths = {k: v for k, v in oracle_cache_paths.items() if v and v.exists()}

    # Normalize evaluator selection: deterministic -> deterministic_tools
    requested = [(e.strip().lower()) for e in (args.evaluators or [])]
    requested = ["deterministic_tools" if e == "deterministic" else e for e in requested]

    # Build evaluator instances
    evaluators: List[Any] = []

    if "deterministic_tools" in requested:
        from evaluators.deterministic_evaluator import DeterministicEvaluator
        evaluators.append(DeterministicEvaluator())

    if "deterministic_heuristic" in requested:
        from evaluators.evaluations.deterministic_heuristic_evaluator import DeterministicHeuristicEvaluator
        evaluators.append(DeterministicHeuristicEvaluator())

    if "llm_single" in requested:
        from evaluators.evaluations.llm_single_evaluator import LLMSingleEvaluator
        evaluators.append(LLMSingleEvaluator(config_path=args.config_llm, model_alias=args.model_alias))

    if "llm_multi" in requested:
        from evaluators.evaluations.llm_multi_agent_evaluator import LLMMultiAgentEvaluator
        evaluators.append(LLMMultiAgentEvaluator(config_path=args.config_llm, model_alias=args.model_alias))

    if "hybrid" in requested:
        from evaluators.evaluations.hybrid_evaluator import HybridEvaluator
        evaluators.append(HybridEvaluator(config_path=args.config_llm, judge_model_alias=args.model_alias))

    runner = MultiTrackBenchmarkRunner(
        dataset_dir=dataset_dir,
        evaluators=evaluators,
        gt2_dir=gt2_dir,
        pct_gt_dir=pct_gt_dir,
        oracle_cache_paths=oracle_cache_paths,
        runs_per_file=args.runs_per_file,
        line_tolerance_gt2=args.line_tolerance_gt2,
        line_tolerance_gt1=args.line_tolerance_gt1,
        limit_originals=args.limit_originals,
        limit_mutations=args.limit_mutations,
        orchestrators=args.orchestrators,
        store_item_results=args.store_item_results,
        max_item_results=args.max_item_results,
    )

    payload = runner.run(Path(args.output_dir))

    # Print compact leaderboard view
    print("\n" + "=" * 80)
    print("MULTI-TRACK LEADERBOARD (mean across runs)")
    print("=" * 80)
    for ev, row in payload.get("leaderboard", {}).items():
        a = row.get("track_A_pct_gates", {})
        b = row.get("track_B_gt1_injection", {})
        c = row.get("track_C_gt2_tool_alignment", {})
        cost = row.get("cost", {})
        stab = row.get("stability", {})

        print(f"\nEvaluator: {ev}")
        print(f"  Track A Gate Acc:      {a.get('overall_gate_accuracy_mean', 0.0):.4f} (std={a.get('overall_gate_accuracy_std', 0.0):.4f})")
        print(f"  Track B GT1 Recall:    {b.get('injection_recall_mean', 0.0):.4f} (std={b.get('injection_recall_std', 0.0):.4f})")
        print(f"  Track B Localization:  {b.get('localization_rate_mean', 0.0):.4f} (std={b.get('localization_rate_std', 0.0):.4f})")
        print(f"  Track C GT2 F1 strict: {c.get('f1_strict_mean', 0.0):.4f} (std={c.get('f1_strict_std', 0.0):.4f})")
        print(f"  Tokens/item:           {cost.get('avg_tokens_per_item_mean', 0.0):.2f} (std={cost.get('avg_tokens_per_item_std', 0.0):.2f})")
        print(f"  Latency ms/item:       {cost.get('avg_latency_ms_per_item_mean', 0.0):.2f} (std={cost.get('avg_latency_ms_per_item_std', 0.0):.2f})")
        print(f"  Stability (gate std):  {stab.get('gate_accuracy_std', 0.0):.4f}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()