#!/usr/bin/env python3
"""
Unified evaluator combining SAT + penalty-free PCT (+ optional Semantic Fidelity report).

Paper scoring (unchanged):
- SAT: from EnhancedStaticAnalyzer
- PCT: from penalty-free platform compliance testers (pct_base)
- Combined score S_code:
      S_code = alpha*SAT + (1-alpha)*PCT
  gated by:
      - platform gate must pass
      - yaml_valid must be True if explicitly provided

Semantic fidelity (reported metric; NOT used in Combined score):
- Uses SemanticAnalyzer (ROUGE + BERTScore) to compute:
    oracle_semantic_fidelity, variant_semantic_fidelity
- Included in output as `semantic_analysis` + summary fields.
"""

# -------------------------------------------------------------------
# Bootstrap imports so this file works as a standalone CLI
# -------------------------------------------------------------------
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]  # .../scripts
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# -------------------------------------------------------------------

from datetime import datetime
from typing import Any, Dict, Optional, List
import json
import logging

import yaml

from evaluators.base_evaluator import (
    EvaluationResult,
    Orchestrator,
)
from evaluators.enhanced_static_analyzer import EnhancedStaticAnalyzer
from evaluators.platform_compliance.airflow_compliance import AirflowComplianceTester
from evaluators.platform_compliance.prefect_compliance import PrefectComplianceTester
from evaluators.platform_compliance.dagster_compliance import DagsterComplianceTester
from evaluators.semantic_analyzer import SemanticAnalyzer


def _mean(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _clamp10(x: float) -> float:
    return max(0.0, min(10.0, float(x)))


def _default_sidecar_path(code_file: Path) -> Path:
    return code_file.with_name(code_file.name + ".unified.json")


def _default_semantic_sidecar_path(code_file: Path) -> Path:
    return code_file.with_name(code_file.name + ".semantic.json")


def _load_generation_metadata(code_file: Path) -> Optional[Dict[str, Any]]:
    meta_path = code_file.parent / "generation_metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _parse_orchestrator(value: Any) -> Optional[Orchestrator]:
    if isinstance(value, str):
        v = value.strip().lower()
        try:
            return Orchestrator(v)
        except Exception:
            return None
    return None


def _flatten_issues(result: Optional[EvaluationResult]) -> List[Dict[str, Any]]:
    if result is None:
        return []
    return [i.to_dict() for i in result.all_issues]


def _issue_summary(issues: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "total": len(issues),
        "critical": sum(1 for i in issues if i.get("severity") == "critical"),
        "major": sum(1 for i in issues if i.get("severity") == "major"),
        "minor": sum(1 for i in issues if i.get("severity") == "minor"),
        "info": sum(1 for i in issues if i.get("severity") == "info"),
    }


class UnifiedEvaluator:
    COMPLIANCE_TESTERS = {
        Orchestrator.AIRFLOW: AirflowComplianceTester,
        Orchestrator.PREFECT: PrefectComplianceTester,
        Orchestrator.DAGSTER: DagsterComplianceTester,
    }

    SAT_DIMS = ["correctness", "code_quality", "best_practices", "maintainability", "robustness"]
    PCT_DIMS = ["loadability", "structure_validity", "configuration_validity", "task_validity", "executability"]

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        intermediate_yaml: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5,
        yaml_valid: Optional[bool] = None,
        semantic_mode: str = "auto",  # auto|compute|load|off
        pipeline_specs_dir: Path = Path("pipeline_specs"),
        pipeline_variants_dir: Path = Path("pipeline_variants"),
        semantic_reference_mode: str = "both",
        bert_model: str = "microsoft/codebert-base",
        bert_device: Optional[str] = None,
        semantic_max_items: int = 30,
    ):
        self.config = config or {}
        self.intermediate_yaml = intermediate_yaml
        self.alpha = float(alpha)
        self.yaml_valid = yaml_valid

        self.semantic_mode = semantic_mode
        self.pipeline_specs_dir = Path(pipeline_specs_dir)
        self.pipeline_variants_dir = Path(pipeline_variants_dir)
        self.semantic_reference_mode = semantic_reference_mode
        self.bert_model = bert_model
        self.bert_device = bert_device
        self.semantic_max_items = int(semantic_max_items)

        self.logger = logging.getLogger(self.__class__.__name__)

        self.static_analyzer = EnhancedStaticAnalyzer(self.config)
        if intermediate_yaml:
            self.static_analyzer.set_reference(intermediate_yaml)

        self._semantic_analyzer: Optional[SemanticAnalyzer] = None

    def _get_semantic_analyzer(self) -> SemanticAnalyzer:
        if self._semantic_analyzer is None:
            self._semantic_analyzer = SemanticAnalyzer(
                pipeline_specs_dir=self.pipeline_specs_dir,
                pipeline_variants_dir=self.pipeline_variants_dir,
                reference_mode=self.semantic_reference_mode,
                bert_model=self.bert_model,
                device=self.bert_device,
                max_items=self.semantic_max_items,
            )
        return self._semantic_analyzer

    def set_reference(self, intermediate_yaml: Dict[str, Any]):
        self.intermediate_yaml = intermediate_yaml
        self.static_analyzer.set_reference(intermediate_yaml)

    def set_yaml_valid(self, yaml_valid: bool):
        self.yaml_valid = bool(yaml_valid)

    def load_reference_from_file(self, yaml_path: Path):
        with open(yaml_path, "r") as f:
            self.intermediate_yaml = yaml.safe_load(f)
        self.static_analyzer.set_reference(self.intermediate_yaml)

    # ---------------------------------------------------------------------
    # Core evaluation
    # ---------------------------------------------------------------------

    def evaluate(self, file_path: Path, orchestrator: Optional[Orchestrator] = None) -> Dict[str, Any]:
        """
        IMPORTANT: must ALWAYS return a dict.
        """
        file_path = Path(file_path)
        self.logger.info(f"Running unified evaluation on: {file_path}")

        try:
            # SAT
            static_result = self.static_analyzer.evaluate(file_path)

            # Orchestrator resolution (target vs detected)
            detected_orchestrator = static_result.orchestrator
            gen_meta = _load_generation_metadata(file_path)

            target_orchestrator = orchestrator
            orch_source = "explicit_argument"

            if target_orchestrator is None or target_orchestrator == Orchestrator.UNKNOWN:
                from_meta = _parse_orchestrator((gen_meta or {}).get("orchestrator"))
                if from_meta is not None:
                    target_orchestrator = from_meta
                    orch_source = "generation_metadata"
                else:
                    target_orchestrator = detected_orchestrator
                    orch_source = "static_detection"

            # PCT (against target orchestrator)
            compliance_result: Optional[EvaluationResult] = None
            if target_orchestrator in self.COMPLIANCE_TESTERS:
                tester_class = self.COMPLIANCE_TESTERS[target_orchestrator]
                tester = tester_class(self.config)
                try:
                    compliance_result = tester.evaluate(file_path)
                except Exception as e:
                    self.logger.exception(f"PCT evaluation failed for target_orchestrator={target_orchestrator}: {e}")
                    compliance_result = None
            else:
                self.logger.warning(
                    f"No compliance tester available for target_orchestrator={target_orchestrator}. "
                    "Compliance will be treated as failed gate."
                )

            # Semantic (reported metric)
            semantic_payload: Optional[Dict[str, Any]] = None
            semantic_issue_summary: Dict[str, int] = _issue_summary([])
            semantic_summary_fields: Dict[str, Any] = {
                "semantic_fidelity_oracle": None,
                "semantic_fidelity_variant": None,
            }

            semantic_mode = (self.semantic_mode or "off").lower().strip()
            if semantic_mode != "off":
                semantic_payload = self._run_semantic(file_path, mode=semantic_mode)

                if semantic_payload is not None and isinstance(semantic_payload, dict):
                    semantic_issue_summary = semantic_payload.get("issue_summary", _issue_summary([])) or _issue_summary([])
                    meta = semantic_payload.get("metadata", {}) or {}
                    oracle = meta.get("oracle", {}) or {}
                    variant = meta.get("variant", {}) or {}

                    semantic_summary_fields["semantic_fidelity_oracle"] = oracle.get("semantic_fidelity")
                    if isinstance(variant, dict) and variant.get("available"):
                        semantic_summary_fields["semantic_fidelity_variant"] = variant.get("semantic_fidelity")

            combined_payload = self._build_combined_result(
                file_path=file_path,
                orchestrator=target_orchestrator,
                static_result=static_result,
                compliance_result=compliance_result,
                semantic_payload=semantic_payload,
                semantic_issue_summary=semantic_issue_summary,
                semantic_summary_fields=semantic_summary_fields,
            )

            combined_payload.setdefault("metadata", {})
            combined_payload["metadata"]["evaluation_context"] = {
                "target_orchestrator": target_orchestrator.value if target_orchestrator else "unknown",
                "detected_orchestrator": detected_orchestrator.value if detected_orchestrator else "unknown",
                "orchestrator_source": orch_source,
            }

            # ✅ CRITICAL FIX: return the payload (your old file forgot this)
            return combined_payload

        except Exception as e:
            # Never return None; provide a minimal, valid unified payload
            self.logger.exception(f"UnifiedEvaluator crashed: {e}")
            return {
                "file_path": str(file_path),
                "orchestrator": "unknown",
                "evaluation_timestamp": datetime.now().isoformat(),
                "alpha": float(self.alpha),
                "yaml_valid": True if self.yaml_valid is None else bool(self.yaml_valid),
                "static_analysis": None,
                "platform_compliance": None,
                "semantic_analysis": None,
                "summary": {
                    "static_score": None,
                    "compliance_score": None,
                    "combined_score": None,
                    "platform_gate_passed": False,
                    "passed": False,
                    "issues": {"total": 0, "critical": 0, "major": 0, "minor": 0, "info": 0},
                    "semantic_fidelity_oracle": None,
                    "semantic_fidelity_variant": None,
                    "semantic_issues": {"total": 0, "critical": 0, "major": 0, "minor": 0, "info": 0},
                },
                "error": {"stage": "unified_evaluator", "message": str(e)},
            }

    # ---------------------------------------------------------------------
    # Semantic runner (auto/load/compute)
    # ---------------------------------------------------------------------

    def _run_semantic(self, file_path: Path, mode: str) -> Optional[Dict[str, Any]]:
        sidecar = _default_semantic_sidecar_path(file_path)

        if mode == "load":
            if sidecar.exists():
                try:
                    return json.loads(sidecar.read_text(encoding="utf-8"))
                except Exception as e:
                    issues = [{"severity": "major", "category": "io", "message": str(e), "line": None, "tool": None, "details": {}}]
                    return {
                        "evaluation_type": "semantic_fidelity",
                        "file_path": str(file_path),
                        "orchestrator": "unknown",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {"error": f"Failed to load semantic sidecar: {e}"},
                        "gates_passed": False,
                        "gate_checks": [],
                        "scores": {},
                        "issues": issues,
                        "issue_summary": _issue_summary(issues),
                    }
            issues = [{"severity": "major", "category": "reference", "message": "Missing semantic sidecar", "line": None, "tool": None, "details": {}}]
            return {
                "evaluation_type": "semantic_fidelity",
                "file_path": str(file_path),
                "orchestrator": "unknown",
                "timestamp": datetime.now().isoformat(),
                "metadata": {"error": f"Semantic sidecar not found: {sidecar}"},
                "gates_passed": False,
                "gate_checks": [],
                "scores": {},
                "issues": issues,
                "issue_summary": _issue_summary(issues),
            }

        if mode == "auto" and sidecar.exists():
            try:
                return json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception:
                pass  # fall through to compute

        analyzer = self._get_semantic_analyzer()
        sem_result = analyzer.evaluate(file_path)
        sem_payload = sem_result.to_dict()
        sem_payload["issues"] = [i.to_dict() for i in sem_result.all_issues]
        sem_payload["issue_summary"] = _issue_summary(sem_payload["issues"])
        return sem_payload

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _get_sat(self, static_result: EvaluationResult) -> float:
        sat = static_result.metadata.get("SAT")
        if sat is not None:
            return _clamp10(float(sat))
        vals = [static_result.scores[d].raw_score for d in self.SAT_DIMS if d in static_result.scores]
        return _clamp10(_mean(vals))

    def _get_pct(self, compliance_result: Optional[EvaluationResult]) -> float:
        if compliance_result is None or not compliance_result.gates_passed:
            return 0.0
        pct = compliance_result.metadata.get("PCT")
        if pct is not None:
            return _clamp10(float(pct))
        vals = [compliance_result.scores[d].raw_score for d in self.PCT_DIMS if d in compliance_result.scores]
        return _clamp10(_mean(vals))

    def _format_eval_result(self, result: Optional[EvaluationResult], kind: str) -> Dict[str, Any]:
        if result is None:
            return {
                "note": f"{kind} not executed (no result)",
                "evaluation_type": kind,
                "file_path": None,
                "orchestrator": Orchestrator.UNKNOWN.value,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"error": f"{kind} not executed (no result)"},
                "gates_passed": False,
                "gate_checks": [],
                "scores": {},
                "issues": [],
                "issue_summary": _issue_summary([]),
            }

        payload = result.to_dict()
        flat = _flatten_issues(result)
        payload["issues"] = flat
        payload["issue_summary"] = _issue_summary(flat)

        if kind.upper() == "SAT":
            overall = result.metadata.get("SAT", None)
            if overall is None:
                overall = _mean([result.scores[d].raw_score for d in self.SAT_DIMS if d in result.scores])
            payload["overall_score"] = _clamp10(float(overall))
        elif kind.upper() == "PCT":
            overall = 0.0 if not result.gates_passed else float(result.metadata.get("PCT", 0.0))
            payload["overall_score"] = _clamp10(float(overall))

        return payload

    def _build_combined_result(
        self,
        file_path: Path,
        orchestrator: Orchestrator,
        static_result: EvaluationResult,
        compliance_result: Optional[EvaluationResult],
        semantic_payload: Optional[Dict[str, Any]],
        semantic_issue_summary: Dict[str, int],
        semantic_summary_fields: Dict[str, Any],
    ) -> Dict[str, Any]:
        sat_value = self._get_sat(static_result)
        pct_value = self._get_pct(compliance_result)

        platform_gate = bool(compliance_result.gates_passed) if compliance_result is not None else False
        yaml_gate_ok = True if self.yaml_valid is None else bool(self.yaml_valid)

        if (not yaml_gate_ok) or (not platform_gate):
            combined_score = 0.0
        else:
            combined_score = self.alpha * sat_value + (1.0 - self.alpha) * pct_value
        combined_score = _clamp10(combined_score)

        passed = platform_gate

        sat_issues = _flatten_issues(static_result)
        pct_issues = _flatten_issues(compliance_result) if compliance_result is not None else []
        paper_issues = sat_issues + pct_issues

        return {
            "file_path": str(file_path),
            "orchestrator": orchestrator.value,
            "evaluation_timestamp": datetime.now().isoformat(),
            "alpha": float(self.alpha),
            "yaml_valid": yaml_gate_ok,

            "static_analysis": self._format_eval_result(static_result, kind="SAT"),
            "platform_compliance": self._format_eval_result(compliance_result, kind="PCT"),
            "semantic_analysis": semantic_payload,

            "summary": {
                "static_score": round(float(sat_value), 4),
                "compliance_score": round(float(pct_value), 4),
                "combined_score": round(float(combined_score), 4),
                "platform_gate_passed": platform_gate,
                "passed": passed,
                "issues": _issue_summary(paper_issues),
                "semantic_fidelity_oracle": semantic_summary_fields.get("semantic_fidelity_oracle"),
                "semantic_fidelity_variant": semantic_summary_fields.get("semantic_fidelity_variant"),
                "semantic_issues": semantic_issue_summary,
            },
        }

    def print_summary(self, unified_payload: Dict[str, Any]) -> None:
        s = unified_payload.get("summary", {}) or {}
        issues = s.get("issues", {}) or {}
        sem_issues = s.get("semantic_issues", {}) or {}

        print("\n" + "=" * 80)
        print("UNIFIED EVALUATION SUMMARY")
        print("Paper scoring: SAT + PCT only (semantic fidelity reported separately)")
        print("=" * 80)
        print(f"File:         {unified_payload.get('file_path')}")
        print(f"Orchestrator: {unified_payload.get('orchestrator')}")
        print(f"Alpha:        {unified_payload.get('alpha')}")
        print(f"YAML valid:   {unified_payload.get('yaml_valid')}")

        print("\nPAPER SCORES")
        print(f"  SAT:    {s.get('static_score', 0.0):.2f}/10")
        print(f"  PCT:    {s.get('compliance_score', 0.0):.2f}/10")
        print(f"  S_code: {s.get('combined_score', 0.0):.2f}/10")

        print("\nGATES")
        print(f"  Platform gate passed: {s.get('platform_gate_passed')}")
        print(f"  Passed (gate-only):   {s.get('passed')}")

        print("\nSEMANTIC FIDELITY (reported; not in S_code)")
        print(f"  Oracle semantic fidelity:  {s.get('semantic_fidelity_oracle')}")
        print(f"  Variant semantic fidelity: {s.get('semantic_fidelity_variant')}")

        print("\nISSUES (SAT+PCT, penalty-free)")
        print(f"  total={issues.get('total', 0)} "
              f"critical={issues.get('critical', 0)} "
              f"major={issues.get('major', 0)} "
              f"minor={issues.get('minor', 0)} "
              f"info={issues.get('info', 0)}")

        print("\nSEMANTIC ISSUES (diagnostics)")
        print(f"  total={sem_issues.get('total', 0)} "
              f"critical={sem_issues.get('critical', 0)} "
              f"major={sem_issues.get('major', 0)} "
              f"minor={sem_issues.get('minor', 0)} "
              f"info={sem_issues.get('info', 0)}")
        print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run unified evaluation (SAT + PCT + optional semantic fidelity).")
    parser.add_argument("file", help="Path to generated workflow Python file")

    parser.add_argument("--orchestrator", default="auto", choices=["auto", "airflow", "prefect", "dagster"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--yaml-valid", default="none", choices=["true", "false", "none"])
    parser.add_argument("--reference-yaml", default=None)

    parser.add_argument("--semantic-mode", default="auto", choices=["auto", "compute", "load", "off"])
    parser.add_argument("--pipeline-specs-dir", default="pipeline_specs")
    parser.add_argument("--pipeline-variants-dir", default="pipeline_variants")
    parser.add_argument("--semantic-reference-mode", default="both", choices=["oracle", "variant", "both"])
    parser.add_argument("--bert-model", default="microsoft/codebert-base")
    parser.add_argument("--bert-device", default=None)
    parser.add_argument("--semantic-max-items", type=int, default=30)

    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")

    file_path = Path(args.file)

    if args.yaml_valid == "none":
        yaml_valid = None
    elif args.yaml_valid == "true":
        yaml_valid = True
    else:
        yaml_valid = False

    ue = UnifiedEvaluator(
        alpha=args.alpha,
        yaml_valid=yaml_valid,
        semantic_mode=args.semantic_mode,
        pipeline_specs_dir=Path(args.pipeline_specs_dir),
        pipeline_variants_dir=Path(args.pipeline_variants_dir),
        semantic_reference_mode=args.semantic_reference_mode,
        bert_model=args.bert_model,
        bert_device=args.bert_device,
        semantic_max_items=args.semantic_max_items,
    )

    if args.reference_yaml:
        ue.load_reference_from_file(Path(args.reference_yaml))

    orch = None if args.orchestrator == "auto" else Orchestrator(args.orchestrator)
    unified_payload = ue.evaluate(file_path, orchestrator=orch)

    if args.print_summary:
        ue.print_summary(unified_payload)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        orch_label = (orch.value if orch else unified_payload.get("orchestrator", "auto"))
        out_path = out_dir / f"unified_{orch_label}_{file_path.stem}_{ts}.json"
    else:
        out_path = _default_sidecar_path(file_path)

    out_path.write_text(json.dumps(unified_payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote: {out_path}")

    if args.stdout:
        print(json.dumps(unified_payload, indent=2, default=str))


if __name__ == "__main__":
    main()