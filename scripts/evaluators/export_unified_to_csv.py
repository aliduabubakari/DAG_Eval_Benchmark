#!/usr/bin/env python3
"""
Enhanced Export Unified Evaluator JSON results to CSV.

This version captures ALL score details, gate checks, and granular metrics
from the unified evaluation JSON format.

Usage:
  python export_unified_to_csv_enhanced.py \
    --input eval_results/unified \
    --out out/unified_summary.csv \
    --issues-out out/unified_issues.csv \
    --gates-out out/unified_gates.csv \
    --scores-out out/unified_scores.csv
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional


# ----------------------------
# Helpers
# ----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Safe nested dict getter: get(obj, "a.b.c", default)"""
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    return cur


def find_unified_jsons(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.unified.json"))


def load_generation_metadata_for_code(code_path: Path) -> Optional[Dict[str, Any]]:
    """Load generation_metadata.json from same directory as code file"""
    meta_path = code_path.parent / "generation_metadata.json"
    if meta_path.exists():
        try:
            return read_json(meta_path)
        except Exception:
            return None
    return None


def normalize_run_context(unified: Dict[str, Any], gen_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract stable run identifiers"""
    code_path = Path(unified.get("file_path", ""))
    
    sem_ctx = get(unified, "semantic_analysis.metadata.run_context", {}) or {}
    pipeline_id = sem_ctx.get("pipeline_id")
    variant_stem = sem_ctx.get("variant_stem")
    class_id = sem_ctx.get("class_id")
    class_name = sem_ctx.get("class_name")
    provider = sem_ctx.get("provider")
    model = sem_ctx.get("model")
    generation_mode = sem_ctx.get("generation_mode")
    prompt_sha256 = sem_ctx.get("prompt_sha256")
    
    if gen_meta:
        pipeline_id = pipeline_id or gen_meta.get("pipeline_id")
        variant_stem = variant_stem or gen_meta.get("variant_stem")
        class_id = class_id or gen_meta.get("class_id")
        class_name = class_name or gen_meta.get("class_name")
        
        mi = gen_meta.get("model_info") or {}
        provider = provider or mi.get("provider")
        model = model or (mi.get("model_name") or mi.get("model_key"))
        
        generation_mode = generation_mode or gen_meta.get("run_type") or gen_meta.get("generation_mode")
        prompt_sha256 = prompt_sha256 or gen_meta.get("prompt_sha256")
    
    # Fallback parse from path
    if not pipeline_id:
        try:
            pipeline_id = code_path.parent.parent.name
        except Exception:
            pipeline_id = None
    if not variant_stem:
        try:
            variant_stem = code_path.parent.name
        except Exception:
            variant_stem = None
    if not class_id and isinstance(variant_stem, str) and variant_stem.startswith("C"):
        class_id = variant_stem.split("_", 1)[0]
    if not class_name and isinstance(variant_stem, str) and "_" in variant_stem:
        class_name = variant_stem.split("_", 1)[1]
    
    return {
        "pipeline_id": pipeline_id,
        "variant_stem": variant_stem,
        "class_id": class_id,
        "class_name": class_name,
        "provider": provider,
        "model": model,
        "generation_mode": generation_mode,
        "prompt_sha256": prompt_sha256,
    }


def summarize_token_usage(gen_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract token usage from generation metadata"""
    out = {
        "gen_input_tokens": None,
        "gen_output_tokens": None,
        "gen_total_tokens": None,
        "gen_reasoning_tokens": None,
        "gen_calls": None,
    }
    if not gen_meta:
        return out
    
    tu = gen_meta.get("token_usage")
    if not isinstance(tu, dict):
        return out
    
    # react format
    if "input_tokens_total" in tu:
        out["gen_input_tokens"] = as_int(tu.get("input_tokens_total"))
        out["gen_output_tokens"] = as_int(tu.get("output_tokens_total"))
        out["gen_reasoning_tokens"] = as_int(tu.get("reasoning_tokens_total"))
        total = 0
        for k in ["input_tokens_total", "output_tokens_total", "reasoning_tokens_total"]:
            v = as_int(tu.get(k))
            if v:
                total += v
        out["gen_total_tokens"] = total if total > 0 else None
        out["gen_calls"] = len(tu.get("calls") or []) if isinstance(tu.get("calls"), list) else None
        return out
    
    # standard format
    inp = as_int(tu.get("input_tokens"))
    outp = as_int(tu.get("output_tokens"))
    tot = as_int(tu.get("total_tokens"))
    if tot is None and inp is not None and outp is not None:
        tot = inp + outp
    
    out["gen_input_tokens"] = inp
    out["gen_output_tokens"] = outp
    out["gen_total_tokens"] = tot
    return out


# ----------------------------
# Row extraction - ENHANCED
# ----------------------------

def extract_summary_row(unified: Dict[str, Any], unified_path: Path, load_gen_meta: bool) -> Dict[str, Any]:
    """Extract comprehensive summary including all score details"""
    code_path = Path(unified.get("file_path", ""))
    
    gen_meta = load_generation_metadata_for_code(code_path) if (load_gen_meta and code_path.exists()) else None
    ctx = normalize_run_context(unified, gen_meta)
    tok = summarize_token_usage(gen_meta)
    
    # SAT data
    sat_meta = get(unified, "static_analysis.metadata", {}) or {}
    sat_dims = sat_meta.get("SAT_dimensions", {}) if isinstance(sat_meta.get("SAT_dimensions"), dict) else {}
    sat_issue_summary = get(unified, "static_analysis.issue_summary", {}) or {}
    sat_scores = get(unified, "static_analysis.scores", {}) or {}
    
    # PCT data
    pct_meta = get(unified, "platform_compliance.metadata", {}) or {}
    pct_dims = pct_meta.get("PCT_dimensions", {}) if isinstance(pct_meta.get("PCT_dimensions"), dict) else {}
    pct_issue_summary = get(unified, "platform_compliance.issue_summary", {}) or {}
    pct_scores = get(unified, "platform_compliance.scores", {}) or {}
    
    # Semantic data
    sem_meta = get(unified, "semantic_analysis.metadata", {}) or {}
    sem_oracle = sem_meta.get("oracle", {}) or {}
    sem_variant = sem_meta.get("variant", {}) or {}
    sem_issue_summary = get(unified, "semantic_analysis.issue_summary", {}) or {}
    
    # Candidate extraction
    cand_sum = sem_meta.get("candidate_extraction_summary", {}) if isinstance(sem_meta.get("candidate_extraction_summary"), dict) else {}
    token_counts = cand_sum.get("token_counts", {}) if isinstance(cand_sum.get("token_counts"), dict) else {}
    
    # Gate passes
    sat_gates_passed = get(unified, "static_analysis.gates_passed")
    pct_gates_passed = get(unified, "platform_compliance.gates_passed")
    sem_gates_passed = get(unified, "semantic_analysis.gates_passed")
    
    row: Dict[str, Any] = {
        # === IDENTIFICATION ===
        "unified_json_path": str(unified_path),
        "code_file_path": str(code_path),
        "evaluation_timestamp": unified.get("evaluation_timestamp"),
        "orchestrator": unified.get("orchestrator"),
        "alpha": as_float(unified.get("alpha")),
        "yaml_valid": unified.get("yaml_valid"),
        
        # === RUN CONTEXT ===
        "pipeline_id": ctx.get("pipeline_id"),
        "variant_stem": ctx.get("variant_stem"),
        "class_id": ctx.get("class_id"),
        "class_name": ctx.get("class_name"),
        "provider": ctx.get("provider"),
        "model": ctx.get("model"),
        "generation_mode": ctx.get("generation_mode"),
        "prompt_sha256": ctx.get("prompt_sha256"),
        
        # === GENERATION METADATA (TOKENS) ===
        **tok,
        
        # === UNIFIED SUMMARY ===
        "passed": get(unified, "summary.passed"),
        "platform_gate_passed": get(unified, "summary.platform_gate_passed"),
        "static_score": as_float(get(unified, "summary.static_score")),
        "compliance_score": as_float(get(unified, "summary.compliance_score")),
        "combined_score": as_float(get(unified, "summary.combined_score")),
        
        # === GATES PASSED ===
        "sat_gates_passed": sat_gates_passed,
        "pct_gates_passed": pct_gates_passed,
        "sem_gates_passed": sem_gates_passed,
        
        # === ISSUE TOTALS (SUMMARY LEVEL) ===
        "issues_total": as_int(get(unified, "summary.issues.total")),
        "issues_critical": as_int(get(unified, "summary.issues.critical")),
        "issues_major": as_int(get(unified, "summary.issues.major")),
        "issues_minor": as_int(get(unified, "summary.issues.minor")),
        "issues_info": as_int(get(unified, "summary.issues.info")),
        
        # === SAT SCORES - TOP LEVEL ===
        "sat_score": as_float(sat_meta.get("SAT")),
        "sat_correctness": as_float(sat_dims.get("correctness")),
        "sat_code_quality": as_float(sat_dims.get("code_quality")),
        "sat_best_practices": as_float(sat_dims.get("best_practices")),
        "sat_maintainability": as_float(sat_dims.get("maintainability")),
        "sat_robustness": as_float(sat_dims.get("robustness")),
        
        # === SAT SCORES - DETAILED ===
        # Correctness details
        "sat_correctness_raw": as_float(get(sat_scores, "correctness.raw_score")),
        "sat_correctness_weight": as_float(get(sat_scores, "correctness.weight")),
        "sat_correctness_penalties": as_float(get(sat_scores, "correctness.penalties_applied")),
        "sat_correctness_syntax": as_float(get(sat_scores, "correctness.details.syntax_subscore")),
        "sat_correctness_import": as_float(get(sat_scores, "correctness.details.import_subscore")),
        "sat_correctness_structure": as_float(get(sat_scores, "correctness.details.structure_subscore")),
        "sat_correctness_semantic": as_float(get(sat_scores, "correctness.details.semantic_subscore")),
        
        # Code quality details
        "sat_code_quality_raw": as_float(get(sat_scores, "code_quality.raw_score")),
        "sat_code_quality_weight": as_float(get(sat_scores, "code_quality.weight")),
        "sat_code_quality_penalties": as_float(get(sat_scores, "code_quality.penalties_applied")),
        
        # Best practices details
        "sat_best_practices_raw": as_float(get(sat_scores, "best_practices.raw_score")),
        "sat_best_practices_weight": as_float(get(sat_scores, "best_practices.weight")),
        "sat_best_practices_penalties": as_float(get(sat_scores, "best_practices.penalties_applied")),
        "sat_bp_uses_default_args": get(sat_scores, "best_practices.details.criteria.uses_default_args"),
        "sat_bp_declares_dependencies": get(sat_scores, "best_practices.details.criteria.declares_dependencies"),
        "sat_bp_externalizes_config": get(sat_scores, "best_practices.details.criteria.externalizes_config"),
        "sat_bp_sets_task_id": get(sat_scores, "best_practices.details.criteria.sets_task_id"),
        "sat_bp_sets_retries": get(sat_scores, "best_practices.details.criteria.sets_retries"),
        "sat_bp_sets_timeouts": get(sat_scores, "best_practices.details.criteria.sets_timeouts"),
        
        # Maintainability details
        "sat_maintainability_raw": as_float(get(sat_scores, "maintainability.raw_score")),
        "sat_maintainability_weight": as_float(get(sat_scores, "maintainability.weight")),
        "sat_maintainability_penalties": as_float(get(sat_scores, "maintainability.penalties_applied")),
        "sat_maint_avg_complexity": as_float(get(sat_scores, "maintainability.details.complexity.avg_complexity")),
        "sat_maint_uses_env_vars": get(sat_scores, "maintainability.details.config_externalization.uses_env_vars"),
        "sat_maint_has_literal_secrets": get(sat_scores, "maintainability.details.config_externalization.has_literal_secret_pattern"),
        "sat_maint_comment_ratio": as_float(get(sat_scores, "maintainability.details.organization.comment_ratio")),
        
        # Robustness details
        "sat_robustness_raw": as_float(get(sat_scores, "robustness.raw_score")),
        "sat_robustness_weight": as_float(get(sat_scores, "robustness.weight")),
        "sat_robustness_penalties": as_float(get(sat_scores, "robustness.penalties_applied")),
        "sat_rob_try_blocks": as_int(get(sat_scores, "robustness.details.error_handling.try_blocks")),
        "sat_rob_function_defs": as_int(get(sat_scores, "robustness.details.error_handling.function_defs")),
        "sat_rob_security_high": as_int(get(sat_scores, "robustness.details.security.counts.high")),
        "sat_rob_security_medium": as_int(get(sat_scores, "robustness.details.security.counts.medium")),
        "sat_rob_security_low": as_int(get(sat_scores, "robustness.details.security.counts.low")),
        
        # SAT issues
        "sat_issues_total": as_int(sat_issue_summary.get("total")),
        "sat_issues_critical": as_int(sat_issue_summary.get("critical")),
        "sat_issues_major": as_int(sat_issue_summary.get("major")),
        "sat_issues_minor": as_int(sat_issue_summary.get("minor")),
        "sat_issues_info": as_int(sat_issue_summary.get("info")),
        
        # SAT metadata
        "code_size_bytes": as_int(sat_meta.get("file_size_bytes")),
        "code_line_count": as_int(sat_meta.get("line_count")),
        "sat_tool_pylint": get(sat_meta, "tools.pylint_available"),
        "sat_tool_radon": get(sat_meta, "tools.radon_available"),
        "sat_tool_bandit": get(sat_meta, "tools.bandit_available"),
        
        # === PCT SCORES - TOP LEVEL ===
        "pct_score": as_float(pct_meta.get("PCT")),
        "pct_loadability": as_float(pct_dims.get("loadability")),
        "pct_structure_validity": as_float(pct_dims.get("structure_validity")),
        "pct_configuration_validity": as_float(pct_dims.get("configuration_validity")),
        "pct_task_validity": as_float(pct_dims.get("task_validity")),
        "pct_executability": as_float(pct_dims.get("executability")),
        
        # === PCT SCORES - DETAILED ===
        # Loadability details
        "pct_loadability_raw": as_float(get(pct_scores, "loadability.raw_score")),
        "pct_loadability_weight": as_float(get(pct_scores, "loadability.weight")),
        "pct_loadability_penalties": as_float(get(pct_scores, "loadability.penalties_applied")),
        "pct_load_total_imports": as_int(get(pct_scores, "loadability.details.imports.total_imports")),
        "pct_load_standard_lib": as_int(get(pct_scores, "loadability.details.imports.standard_lib")),
        "pct_load_third_party": as_int(get(pct_scores, "loadability.details.imports.third_party")),
        "pct_load_module_loadable": get(pct_scores, "loadability.details.platform_load.module_loadable"),
        "pct_load_airflow_available": get(pct_scores, "loadability.details.platform_load.airflow_available"),
        "pct_load_airflow_version": get(pct_scores, "loadability.details.platform_load.airflow_version"),
        
        # Structure validity details
        "pct_structure_raw": as_float(get(pct_scores, "structure_validity.raw_score")),
        "pct_structure_weight": as_float(get(pct_scores, "structure_validity.weight")),
        "pct_structure_penalties": as_float(get(pct_scores, "structure_validity.penalties_applied")),
        "pct_struct_has_dag": get(pct_scores, "structure_validity.details.constructs.has_dag_definition"),
        "pct_struct_has_tasks": get(pct_scores, "structure_validity.details.constructs.has_tasks"),
        "pct_struct_dag_count": as_int(get(pct_scores, "structure_validity.details.constructs.dag_count")),
        "pct_struct_task_count": as_int(get(pct_scores, "structure_validity.details.constructs.task_count")),
        "pct_struct_uses_context_mgr": get(pct_scores, "structure_validity.details.constructs.uses_context_manager"),
        "pct_struct_uses_taskflow": get(pct_scores, "structure_validity.details.constructs.uses_taskflow"),
        "pct_struct_has_dependencies": get(pct_scores, "structure_validity.details.dependency_graph.has_dependencies"),
        "pct_struct_dependency_count": as_int(get(pct_scores, "structure_validity.details.dependency_graph.dependency_count")),
        "pct_struct_orphan_count": as_int(get(pct_scores, "structure_validity.details.orphan_check.orphan_count")),
        
        # Configuration validity details
        "pct_config_raw": as_float(get(pct_scores, "configuration_validity.raw_score")),
        "pct_config_weight": as_float(get(pct_scores, "configuration_validity.weight")),
        "pct_config_penalties": as_float(get(pct_scores, "configuration_validity.penalties_applied")),
        "pct_cfg_has_schedule": get(pct_scores, "configuration_validity.details.schedule.has_schedule"),
        "pct_cfg_schedule_type": get(pct_scores, "configuration_validity.details.schedule.schedule_type"),
        "pct_cfg_has_start_date": get(pct_scores, "configuration_validity.details.schedule.has_start_date"),
        "pct_cfg_has_catchup": get(pct_scores, "configuration_validity.details.schedule.has_catchup"),
        "pct_cfg_has_default_args": get(pct_scores, "configuration_validity.details.default_args.has_default_args"),
        "pct_cfg_has_owner": get(pct_scores, "configuration_validity.details.default_args.has_owner"),
        "pct_cfg_has_retries": get(pct_scores, "configuration_validity.details.default_args.has_retries"),
        "pct_cfg_has_retry_delay": get(pct_scores, "configuration_validity.details.default_args.has_retry_delay"),
        "pct_cfg_hardcoded_secrets": get(pct_scores, "configuration_validity.details.security.hardcoded_secrets"),
        
        # Task validity details
        "pct_task_raw": as_float(get(pct_scores, "task_validity.raw_score")),
        "pct_task_weight": as_float(get(pct_scores, "task_validity.weight")),
        "pct_task_penalties": as_float(get(pct_scores, "task_validity.penalties_applied")),
        "pct_task_operator_count": as_int(get(pct_scores, "task_validity.details.definitions.operator_count")),
        "pct_task_taskflow_count": as_int(get(pct_scores, "task_validity.details.definitions.taskflow_count")),
        "pct_task_has_task_ids": get(pct_scores, "task_validity.details.definitions.has_task_ids"),
        
        # Executability details
        "pct_exec_raw": as_float(get(pct_scores, "executability.raw_score")),
        "pct_exec_weight": as_float(get(pct_scores, "executability.weight")),
        "pct_exec_penalties": as_float(get(pct_scores, "executability.penalties_applied")),
        "pct_exec_can_test": get(pct_scores, "executability.details.dryrun.can_test"),
        "pct_exec_requires_docker": get(pct_scores, "executability.details.external_deps.requires_docker"),
        "pct_exec_requires_network": get(pct_scores, "executability.details.external_deps.requires_network"),
        
        # PCT issues
        "pct_issues_total": as_int(pct_issue_summary.get("total")),
        "pct_issues_critical": as_int(pct_issue_summary.get("critical")),
        "pct_issues_major": as_int(pct_issue_summary.get("major")),
        "pct_issues_minor": as_int(pct_issue_summary.get("minor")),
        "pct_issues_info": as_int(pct_issue_summary.get("info")),
        
        # === SEMANTIC SUMMARY ===
        "semantic_oracle": as_float(get(unified, "summary.semantic_fidelity_oracle")),
        "semantic_variant": as_float(get(unified, "summary.semantic_fidelity_variant")),
        
        # Semantic extraction counts
        "sem_steps_count": as_int(cand_sum.get("steps_count")),
        "sem_edges_count": as_int(cand_sum.get("edges_count")),
        "sem_token_urls": as_int(token_counts.get("urls")),
        "sem_token_files": as_int(token_counts.get("files")),
        "sem_token_images": as_int(token_counts.get("images")),
        "sem_token_env_vars": as_int(token_counts.get("env_vars")),
        "sem_token_connections": as_int(token_counts.get("connections")),
        "sem_token_retries": as_int(token_counts.get("retries")),
        "sem_token_retry_delays": as_int(token_counts.get("retry_delays")),
        "sem_token_timeouts": as_int(token_counts.get("timeouts")),
        
        # Semantic oracle metrics
        "sem_oracle_fidelity_structural": as_float(sem_oracle.get("fidelity_structural")),
        "sem_oracle_fidelity_full": as_float(sem_oracle.get("fidelity_full")),
        "sem_oracle_semantic_fidelity": as_float(sem_oracle.get("semantic_fidelity")),
        "sem_oracle_rouge1_f1_struct": as_float(get(sem_oracle, "struct_metrics.rouge1_f1")),
        "sem_oracle_rougeL_f1_struct": as_float(get(sem_oracle, "struct_metrics.rougeL_f1")),
        "sem_oracle_bert_p_struct": as_float(get(sem_oracle, "struct_metrics.bert_p")),
        "sem_oracle_bert_r_struct": as_float(get(sem_oracle, "struct_metrics.bert_r")),
        "sem_oracle_bert_f1_struct": as_float(get(sem_oracle, "struct_metrics.bert_f1")),
        "sem_oracle_rouge1_f1_full": as_float(get(sem_oracle, "full_metrics.rouge1_f1")),
        "sem_oracle_rougeL_f1_full": as_float(get(sem_oracle, "full_metrics.rougeL_f1")),
        "sem_oracle_bert_p_full": as_float(get(sem_oracle, "full_metrics.bert_p")),
        "sem_oracle_bert_r_full": as_float(get(sem_oracle, "full_metrics.bert_r")),
        "sem_oracle_bert_f1_full": as_float(get(sem_oracle, "full_metrics.bert_f1")),
        
        # Semantic issues
        "sem_issues_total": as_int(sem_issue_summary.get("total")),
        "sem_issues_critical": as_int(sem_issue_summary.get("critical")),
        "sem_issues_major": as_int(sem_issue_summary.get("major")),
        "sem_issues_minor": as_int(sem_issue_summary.get("minor")),
        "sem_issues_info": as_int(sem_issue_summary.get("info")),
    }
    
    # Variant metrics (if available)
    if isinstance(sem_variant, dict) and sem_variant.get("available"):
        row.update({
            "sem_variant_fidelity_structural": as_float(sem_variant.get("fidelity_structural")),
            "sem_variant_fidelity_full": as_float(sem_variant.get("fidelity_full")),
            "sem_variant_semantic_fidelity": as_float(sem_variant.get("semantic_fidelity")),
            "sem_variant_rouge1_f1_struct": as_float(get(sem_variant, "struct_metrics.rouge1_f1")),
            "sem_variant_rougeL_f1_struct": as_float(get(sem_variant, "struct_metrics.rougeL_f1")),
            "sem_variant_bert_f1_struct": as_float(get(sem_variant, "struct_metrics.bert_f1")),
            "sem_variant_rouge1_f1_full": as_float(get(sem_variant, "full_metrics.rouge1_f1")),
            "sem_variant_rougeL_f1_full": as_float(get(sem_variant, "full_metrics.rougeL_f1")),
            "sem_variant_bert_f1_full": as_float(get(sem_variant, "full_metrics.bert_f1")),
        })
    
    return row


def extract_gate_rows(unified: Dict[str, Any], unified_path: Path, load_gen_meta: bool) -> List[Dict[str, Any]]:
    """Extract all gate check results"""
    code_path = Path(unified.get("file_path", ""))
    gen_meta = load_generation_metadata_for_code(code_path) if (load_gen_meta and code_path.exists()) else None
    ctx = normalize_run_context(unified, gen_meta)
    
    base = {
        "unified_json_path": str(unified_path),
        "code_file_path": str(code_path),
        "evaluation_timestamp": unified.get("evaluation_timestamp"),
        "orchestrator": unified.get("orchestrator"),
        "pipeline_id": ctx.get("pipeline_id"),
        "variant_stem": ctx.get("variant_stem"),
        "class_id": ctx.get("class_id"),
        "class_name": ctx.get("class_name"),
        "provider": ctx.get("provider"),
        "model": ctx.get("model"),
        "generation_mode": ctx.get("generation_mode"),
    }
    
    rows: List[Dict[str, Any]] = []
    
    # SAT gates
    sat_gates = get(unified, "static_analysis.gate_checks", []) or []
    for gate in sat_gates:
        if isinstance(gate, dict):
            rows.append({
                **base,
                "gate_source": "SAT",
                "gate_name": gate.get("name"),
                "gate_passed": gate.get("passed"),
                "gate_message": gate.get("message"),
                "gate_details": str(gate.get("details", ""))
            })

    # PCT gates
    pct_gates = get(unified, "platform_compliance.gate_checks", []) or []
    for gate in pct_gates:
        if isinstance(gate, dict):
            rows.append({
                **base,
                "gate_source": "PCT",
                "gate_name": gate.get("name"),
                "gate_passed": gate.get("passed"),
                "gate_message": gate.get("message"),
                "gate_details": str(gate.get("details", ""))
            })

    # SEM gates
    sem_gates = get(unified, "semantic_analysis.gate_checks", []) or []
    for gate in sem_gates:
        if isinstance(gate, dict):
            rows.append({
                **base,
                "gate_source": "SEM",
                "gate_name": gate.get("name"),
                "gate_passed": gate.get("passed"),
                "gate_message": gate.get("message"),
                "gate_details": str(gate.get("details", ""))
            })

    return rows


def extract_score_rows(unified: Dict[str, Any], unified_path: Path, load_gen_meta: bool) -> List[Dict[str, Any]]:
    """Extract granular scores into a normalized format (one row per metric)"""
    code_path = Path(unified.get("file_path", ""))
    gen_meta = load_generation_metadata_for_code(code_path) if (load_gen_meta and code_path.exists()) else None
    ctx = normalize_run_context(unified, gen_meta)
    
    base = {
        "unified_json_path": str(unified_path),
        "code_file_path": str(code_path),
        "pipeline_id": ctx.get("pipeline_id"),
        "variant_stem": ctx.get("variant_stem"),
        "model": ctx.get("model"),
    }
    
    rows: List[Dict[str, Any]] = []

    def process_score_dict(score_dict: Dict[str, Any], source: str):
        for metric_name, data in score_dict.items():
            if not isinstance(data, dict): continue
            rows.append({
                **base,
                "score_source": source,
                "metric_name": metric_name,
                "raw_score": as_float(data.get("raw_score")),
                "weight": as_float(data.get("weight")),
                "penalties": as_float(data.get("penalties_applied")),
                "error": data.get("error"),
                "details": str(data.get("details", ""))
            })

    process_score_dict(get(unified, "static_analysis.scores", {}), "SAT")
    process_score_dict(get(unified, "platform_compliance.scores", {}), "PCT")
    
    return rows


def extract_issue_rows(unified: Dict[str, Any], unified_path: Path, load_gen_meta: bool) -> List[Dict[str, Any]]:
    """Produce one row per issue with run identifiers"""
    code_path = Path(unified.get("file_path", ""))
    gen_meta = load_generation_metadata_for_code(code_path) if (load_gen_meta and code_path.exists()) else None
    ctx = normalize_run_context(unified, gen_meta)

    base = {
        "unified_json_path": str(unified_path),
        "code_file_path": str(code_path),
        "evaluation_timestamp": unified.get("evaluation_timestamp"),
        "orchestrator": unified.get("orchestrator"),
        "pipeline_id": ctx.get("pipeline_id"),
        "variant_stem": ctx.get("variant_stem"),
        "class_id": ctx.get("class_id"),
        "class_name": ctx.get("class_name"),
        "provider": ctx.get("provider"),
        "model": ctx.get("model"),
        "generation_mode": ctx.get("generation_mode"),
    }

    out_rows: List[Dict[str, Any]] = []

    def add_issues(issue_list: Any, source: str):
        if not isinstance(issue_list, list): return
        for i in issue_list:
            if not isinstance(i, dict): continue
            out_rows.append({
                **base,
                "issue_source": source,
                "severity": i.get("severity"),
                "category": i.get("category"),
                "message": i.get("message"),
                "line": i.get("line"),
                "tool": i.get("tool"),
            })

    add_issues(get(unified, "static_analysis.issues", []), "SAT")
    add_issues(get(unified, "platform_compliance.issues", []), "PCT")
    add_issues(get(unified, "semantic_analysis.issues", []), "SEM")

    return out_rows


# ----------------------------
# CSV Writing Logic
# ----------------------------

def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not out_path: return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    priority = [
        "pipeline_id", "class_id", "class_name", "variant_stem", "orchestrator",
        "provider", "model", "generation_mode", "score_source", "gate_source", "issue_source",
        "metric_name", "gate_name", "evaluation_timestamp"
    ]
    all_keys = sorted({k for r in rows for k in r.keys()})
    cols = [k for k in priority if k in all_keys]
    cols.extend([k for k in all_keys if k not in cols])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------
# Main Execution
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Enhanced Unified Evaluator Export")
    parser.add_argument("--input", required=True, help="Path to .unified.json or directory")
    parser.add_argument("--out", required=True, help="Summary CSV output")
    parser.add_argument("--issues-out", help="Issues CSV output")
    parser.add_argument("--gates-out", help="Gate checks CSV output")
    parser.add_argument("--scores-out", help="Detailed scores CSV output")
    parser.add_argument("--no-generation-metadata", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    unified_files = find_unified_jsons(input_path)
    if not unified_files:
        print(f"No files found for {input_path}")
        return

    load_gen_meta = not args.no_generation_metadata
    
    summaries, issues, gates, scores = [], [], [], []

    for p in unified_files:
        try:
            data = read_json(p)
            summaries.append(extract_summary_row(data, p, load_gen_meta))
            if args.issues_out:
                issues.extend(extract_issue_rows(data, p, load_gen_meta))
            if args.gates_out:
                gates.extend(extract_gate_rows(data, p, load_gen_meta))
            if args.scores_out:
                scores.extend(extract_score_rows(data, p, load_gen_meta))
        except Exception as e:
            print(f"Error processing {p}: {e}")

    write_csv(summaries, Path(args.out))
    if args.issues_out: write_csv(issues, Path(args.issues_out))
    if args.gates_out: write_csv(gates, Path(args.gates_out))
    if args.scores_out: write_csv(scores, Path(args.scores_out))

    print(f"Export complete. Summary rows: {len(summaries)}")


if __name__ == "__main__":
    main()