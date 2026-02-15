#!/usr/bin/env python3
"""
Semantic Fidelity Analyzer (Spec ↔ Code) using ROUGE + BERTScore
===============================================================

Goal:
- Measure semantic fidelity between PipelineSpec JSON reference and generated workflow code.
- Supports BOTH references:
  (1) Oracle spec: pipeline_specs/<pipeline_id>.pipelinespec.json
  (2) Variant spec: pipeline_variants/<pipeline_id>/specs/<Ck>.json

Metrics:
- ROUGE-1 F1, ROUGE-L F1
- BERTScore (P, R, F1), default model: microsoft/codebert-base

Scores (0..10):
- fidelity_structural: compares structural signatures (steps/mechanisms/edges)
- fidelity_full: compares full signatures (adds artifacts/params/env/schedule/etc.)
- semantic_fidelity: 0.7*structural + 0.3*full

Notes:
- We do NOT compare raw code to raw JSON.
- We compare "semantic signature texts" derived deterministically from:
  - PipelineSpec JSON
  - Generated code (orchestrator-aware AST+regex extraction)

Dependencies:
  pip install rouge-score bert-score torch transformers

CLI:
  python scripts/evaluators/semantic_analyzer.py path/to/generated.py \
    --pipeline-specs-dir pipeline_specs \
    --pipeline-variants-dir pipeline_variants \
    --reference-mode both \
    --print-summary
"""

# -------------------------------------------------------------------
# Bootstrap: allow running as standalone CLI
# -------------------------------------------------------------------
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]  # .../scripts
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# -------------------------------------------------------------------

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
import ast
import json
import logging
import re

from transformers import AutoConfig

from evaluators.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationScore,
    GateCheckResult,
    Issue,
    Severity,
    Orchestrator,
)

# ----------------------------
# Optional third-party metrics
# ----------------------------

try:
    from rouge_score import rouge_scorer  # type: ignore
    _HAS_ROUGE = True
except Exception:
    rouge_scorer = None
    _HAS_ROUGE = False

try:
    import torch  # type: ignore
    from bert_score import BERTScorer  # type: ignore
    _HAS_BERTSCORE = True
except Exception:
    torch = None
    BERTScorer = None
    _HAS_BERTSCORE = False


# ----------------------------
# Utilities
# ----------------------------

def _clamp10(x: float) -> float:
    return max(0.0, min(10.0, float(x)))

def _safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""

def _sorted_unique_str(xs: List[str]) -> List[str]:
    return sorted({x for x in xs if isinstance(x, str) and x.strip()})

def _mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else 0.0


# ----------------------------
# Run-context discovery (pipeline_id, class_id)
# ----------------------------

def _try_load_generation_metadata(code_file: Path) -> Optional[Dict[str, Any]]:
    meta_path = code_file.parent / "generation_metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _infer_context_from_paths(code_file: Path) -> Dict[str, Optional[str]]:
    pipeline_id = None
    variant_stem = None
    class_id = None

    try:
        variant_stem = code_file.parent.name
        pipeline_id = code_file.parent.parent.name
        if variant_stem and re.match(r"^C\d+\b", variant_stem):
            class_id = variant_stem.split("_", 1)[0]
    except Exception:
        pass

    return {"pipeline_id": pipeline_id, "variant_stem": variant_stem, "class_id": class_id}

def resolve_run_context(code_file: Path) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "pipeline_id": None,
        "variant_stem": None,
        "class_id": None,
        "class_name": None,
        "provider": None,
        "model": None,
        "generation_mode": None,
        "prompt_sha256": None,
    }

    meta = _try_load_generation_metadata(code_file)
    if meta:
        ctx["pipeline_id"] = meta.get("pipeline_id")
        ctx["variant_stem"] = meta.get("variant_stem")
        ctx["class_id"] = meta.get("class_id")
        ctx["class_name"] = meta.get("class_name")

        mi = meta.get("model_info") or {}
        ctx["provider"] = mi.get("provider")
        ctx["model"] = mi.get("model_name") or mi.get("model_key")

        ctx["generation_mode"] = meta.get("run_type") or meta.get("generation_mode")
        ctx["prompt_sha256"] = meta.get("prompt_sha256")
        return ctx

    infer = _infer_context_from_paths(code_file)
    ctx.update(infer)
    if ctx.get("class_id") and ctx.get("variant_stem"):
        parts = ctx["variant_stem"].split("_", 1)
        if len(parts) == 2:
            ctx["class_name"] = parts[1]
    return ctx


# ----------------------------
# Reference loading
# ----------------------------

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def find_oracle_spec(pipeline_specs_dir: Path, pipeline_id: str) -> Optional[Path]:
    p = pipeline_specs_dir / f"{pipeline_id}.pipelinespec.json"
    return p if p.exists() else None

def find_variant_spec(pipeline_variants_dir: Path, pipeline_id: str, class_id: str) -> Optional[Path]:
    p = pipeline_variants_dir / pipeline_id / "specs" / f"{class_id}.json"
    if p.exists():
        return p
    if class_id == "C9":
        p2 = pipeline_variants_dir / pipeline_id / "specs" / "C0.json"
        if p2.exists():
            return p2
    return None


# ----------------------------
# Signature generation (PipelineSpec JSON → text)
# ----------------------------

@dataclass
class SignatureTexts:
    structural: str
    full: str


def spec_to_signatures(spec: Dict[str, Any], *, max_items: int = 30) -> SignatureTexts:
    summary = spec.get("summary") or {}
    schedule = spec.get("schedule") or {}
    cf = spec.get("control_flow") or {}
    da = spec.get("data_artifacts") or {}

    pattern = _safe_str(summary.get("topology_pattern") or "unknown")
    schedule_type = _safe_str(schedule.get("schedule_type") or "unknown")

    steps = [s for s in (spec.get("steps") or []) if isinstance(s, dict)]
    steps_sorted = sorted(steps, key=lambda x: _safe_str(x.get("step_id") or x.get("name") or ""))

    edges = [e for e in (cf.get("edges") or []) if isinstance(e, dict)]
    edges_sorted = sorted(edges, key=lambda x: (_safe_str(x.get("from")), _safe_str(x.get("to"))))

    # --- structural ---
    struct_lines: List[str] = []
    struct_lines.append(f"PATTERN: {pattern}")
    struct_lines.append(f"SCHEDULE_TYPE: {schedule_type}")

    for st in steps_sorted[:max_items]:
        sid = _safe_str(st.get("step_id") or st.get("name") or "step")
        mech = _safe_str(st.get("mechanism") or "unknown")
        obj = _safe_str(st.get("objective") or "")[:120]
        if obj:
            struct_lines.append(f"STEP: {sid} | MECH: {mech} | OBJ: {obj}")
        else:
            struct_lines.append(f"STEP: {sid} | MECH: {mech}")

    for e in edges_sorted[:max_items * 2]:
        fr = _safe_str(e.get("from"))
        to = _safe_str(e.get("to"))
        if fr and to:
            struct_lines.append(f"EDGE: {fr} -> {to}")

    for g in (cf.get("gates") or [])[:max_items]:
        if isinstance(g, dict):
            struct_lines.append(f"GATE: {g.get('type')} | STEP: {g.get('step')} | WAITS_FOR: {g.get('waits_for')}")

    for b in (cf.get("branch_points") or [])[:max_items]:
        if not isinstance(b, dict):
            continue
        step = b.get("step")
        merge = b.get("merge_step")
        struct_lines.append(f"BRANCH: {step} | MERGE: {merge}")
        for br in (b.get("branches") or [])[:max_items]:
            if isinstance(br, dict):
                struct_lines.append(f"  IF: {br.get('condition')} THEN: {br.get('next_steps')}")

    structural_text = "\n".join([ln for ln in struct_lines if ln and str(ln).strip()])

    # --- full ---
    full_lines: List[str] = []
    full_lines.extend(struct_lines)

    full_lines.append(f"SCHEDULE_EXPR: {schedule.get('schedule_expression')}")
    full_lines.append(f"START_DATE: {schedule.get('start_date')}")
    full_lines.append(f"CATCHUP_BACKFILL: {schedule.get('catchup_backfill')}")

    for sysobj in (spec.get("external_systems") or [])[:max_items]:
        if not isinstance(sysobj, dict):
            continue
        stype = _safe_str(sysobj.get("type") or "other")
        name = _safe_str(sysobj.get("name") or "")
        ident = _safe_str(sysobj.get("identifier") or "")
        full_lines.append(f"SYSTEM: {stype} | NAME: {name} | ID: {ident}")

    def add_artifacts(key: str, prefix: str):
        items = da.get(key) or []
        for it in items[:max_items]:
            if isinstance(it, dict) and it.get("identifier"):
                full_lines.append(f"{prefix}: {_safe_str(it.get('identifier'))}")

    add_artifacts("files", "FILE")
    add_artifacts("tables", "TABLE")
    add_artifacts("buckets", "BUCKET")
    add_artifacts("topics", "TOPIC")

    env_vars: List[str] = []
    mounts: List[str] = []
    networks: List[str] = []

    for st in steps_sorted:
        env = st.get("env_infra") or {}
        env_vars.extend([x for x in (env.get("env_vars") or []) if isinstance(x, str)])
        mounts.extend([x for x in (env.get("mounts") or []) if isinstance(x, str)])
        net = env.get("network")
        if isinstance(net, str) and net.strip():
            networks.append(net.strip())

    for v in _sorted_unique_str(env_vars)[:max_items]:
        full_lines.append(f"ENV_VAR: {v}")
    for m in _sorted_unique_str(mounts)[:max_items]:
        full_lines.append(f"MOUNT: {m}")
    for n in _sorted_unique_str(networks)[:max_items]:
        full_lines.append(f"NETWORK: {n}")

    for st in steps_sorted[:max_items]:
        sid = _safe_str(st.get("step_id") or st.get("name") or "step")
        for kv in (st.get("parameters") or [])[:max_items]:
            if isinstance(kv, dict) and kv.get("key"):
                full_lines.append(f"PARAM: {sid} | {kv.get('key')}={kv.get('value')}")
        fh = st.get("failure_handling") or {}
        if isinstance(fh, dict):
            if fh.get("retries") is not None:
                full_lines.append(f"RETRIES: {sid} | {fh.get('retries')}")
            if fh.get("retry_delay") is not None:
                full_lines.append(f"RETRY_DELAY: {sid} | {fh.get('retry_delay')}")
            if fh.get("timeout") is not None:
                full_lines.append(f"TIMEOUT: {sid} | {fh.get('timeout')}")
            for a in (fh.get("alerts") or [])[:max_items]:
                full_lines.append(f"ALERT: {sid} | {a}")
            for idem in (fh.get("idempotency") or [])[:max_items]:
                full_lines.append(f"IDEMPOTENCY: {sid} | {idem}")

    full_text = "\n".join([ln for ln in full_lines if ln and str(ln).strip()])
    return SignatureTexts(structural=structural_text, full=full_text)


# ----------------------------
# Candidate extraction (Generated code → features)
# ----------------------------

# Regexes for generic token extraction (fallback)
_URL_RE = re.compile(r"https?://[^\s'\"),>]+", re.IGNORECASE)
_FILE_RE = re.compile(r"(/[^ \n\t'\"()]+)", re.IGNORECASE)

# Broadened "image=" extraction (prefer AST, but regex helps for non-AST patterns)
_IMAGE_KW_RE = re.compile(r"\bimage\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)

_ENV_GETENV_RE = re.compile(r"os\.getenv\(\s*['\"]([A-Za-z0-9_]+)['\"]\s*\)")
_ENV_OSENV_RE = re.compile(r"os\.environ\[\s*['\"]([A-Za-z0-9_]+)['\"]\s*\]")
_ENV_OSENVGET_RE = re.compile(r"os\.environ\.get\(\s*['\"]([A-Za-z0-9_]+)['\"]\s*\)")

_CONN_RE = re.compile(r"(?:conn_id|connection_id|postgres_conn_id|http_conn_id|gcp_conn_id)\s*=\s*['\"]([^'\"]+)['\"]")


def _call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None

def _decorator_name(dec: ast.AST) -> Optional[str]:
    if isinstance(dec, ast.Name):
        return dec.id
    if isinstance(dec, ast.Attribute):
        return dec.attr
    if isinstance(dec, ast.Call):
        return _decorator_name(dec.func)
    return None

def _extract_refs(node: ast.AST) -> List[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        base = _extract_refs(node.value)
        if base:
            return [f"{base[0]}.{node.attr}"]
        return [node.attr]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: List[str] = []
        for elt in node.elts:
            out.extend(_extract_refs(elt))
        return out
    return []

def _map_airflow_operator_to_mechanism(op: str) -> str:
    o = (op or "").lower()
    if "dockeroperator" in o:
        return "container_run"
    if "kubernetes" in o or "kubernetespodoperator" in o:
        return "kubernetes_job"
    if "bashoperator" in o:
        return "shell_command"
    if "emailoperator" in o:
        return "notification"
    if "httpoperator" in o or "simplehttpoperator" in o or "graphql" in o:
        return "http_request"
    if "sensor" in o:
        return "wait_poll"
    if "postgresoperator" in o or "hiveoperator" in o or "sql" in o:
        return "sql_query"
    if "spark" in o or "databricks" in o:
        return "spark_job"
    if "triggerdagrun" in o:
        return "external_workflow_trigger"
    return "python_callable"


# ---- AST literal utilities (deterministic) ----

def _const_to_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return str(node.value)
    return None

def _list_of_consts_to_str(node: ast.AST, max_items: int = 60) -> Optional[str]:
    if isinstance(node, (ast.List, ast.Tuple)):
        parts = []
        for elt in node.elts[:max_items]:
            s = _const_to_str(elt)
            if s is None:
                continue
            parts.append(s)
        if parts:
            return " ".join(parts)
    return None

def _dict_keys_to_str(node: ast.AST, max_items: int = 60) -> Optional[str]:
    if isinstance(node, ast.Dict):
        keys = []
        for k in node.keys[:max_items]:
            ks = _const_to_str(k)
            if ks is not None:
                keys.append(ks)
        keys = [k for k in keys if k]
        if keys:
            return ",".join(sorted(set(keys)))
    return None

def _ast_to_compact_str(node: ast.AST, max_len: int = 220) -> Optional[str]:
    """
    Conservative fallback: if we can unparse, truncate. Used only if constant extraction fails.
    """
    try:
        if hasattr(ast, "unparse"):
            s = ast.unparse(node)
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) > max_len:
                s = s[:max_len] + "...[trunc]"
            return s
    except Exception:
        pass
    return None


def extract_airflow_features(code: str, tree: ast.AST) -> Dict[str, Any]:
    """
    Airflow extraction:
    - steps: task_id/operator/mechanism + selected keyword params
    - edges: >>, <<, chain(), set_upstream/downstream
    - schedule: schedule_interval/schedule + catchup
    """
    tasks: List[Dict[str, Any]] = []
    var_to_task_id: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            fname = _call_name(call.func)
            if not fname:
                continue
            if not (fname.endswith("Operator") or fname.endswith("Sensor") or "Operator" in fname or "Sensor" in fname):
                continue

            var_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            task_id: Optional[str] = None
            kw_map: Dict[str, str] = {}

            for kw in call.keywords or []:
                if not kw.arg:
                    continue

                if kw.arg == "task_id":
                    v = _const_to_str(kw.value)
                    if v:
                        task_id = v
                    continue

                # curated scalar params
                if kw.arg in {
                    "image", "network_mode", "namespace", "cluster_context", "name",
                    "bash_command", "sql", "endpoint", "filepath", "file_pattern",
                    "poke_interval", "timeout", "retries", "retry_delay",
                    "docker_url", "api_version",
                    "trigger_dag_id", "external_dag_id", "external_task_id",
                }:
                    v = _const_to_str(kw.value)
                    if v is None:
                        v = _ast_to_compact_str(kw.value)
                    if v is not None:
                        kw_map[kw.arg] = v
                    continue

                # list-like params
                if kw.arg in {"command", "cmds", "arguments"}:
                    v = _list_of_consts_to_str(kw.value)
                    if v is None:
                        v = _ast_to_compact_str(kw.value)
                    if v is not None:
                        kw_map[kw.arg] = v
                    continue

                # dict-like params: capture keys
                if kw.arg in {"environment", "env_vars"}:
                    v = _dict_keys_to_str(kw.value)
                    if v is None:
                        v = _ast_to_compact_str(kw.value)
                    if v is not None:
                        kw_map[kw.arg] = v
                    continue

                # mounts/volumes often list-like
                if kw.arg in {"mounts", "volumes"}:
                    v = _list_of_consts_to_str(kw.value)
                    if v is None:
                        v = _ast_to_compact_str(kw.value)
                    if v is not None:
                        kw_map[kw.arg] = v
                    continue

                # conn ids
                if kw.arg.endswith("_conn_id") or kw.arg in {"conn_id", "connection_id"}:
                    v = _const_to_str(kw.value)
                    if v is None:
                        v = _ast_to_compact_str(kw.value)
                    if v is not None:
                        kw_map[kw.arg] = v
                    continue

            step_id = task_id or (var_names[0] if var_names else fname)
            if var_names and task_id:
                var_to_task_id[var_names[0]] = task_id

            tasks.append({
                "step_id": step_id,
                "operator": fname,
                "mechanism": _map_airflow_operator_to_mechanism(fname),
                "params": kw_map,
            })

    # dependencies
    edges: Set[Tuple[str, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
            left = _extract_refs(node.left)
            right = _extract_refs(node.right)
            for l in left:
                for r in right:
                    edges.add((l, r))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.LShift):
            left = _extract_refs(node.left)
            right = _extract_refs(node.right)
            for l in left:
                for r in right:
                    edges.add((r, l))
        elif isinstance(node, ast.Call):
            fn = _call_name(node.func)
            if fn == "chain":
                flat: List[str] = []
                for a in node.args:
                    flat.extend(_extract_refs(a))
                for i in range(len(flat) - 1):
                    edges.add((flat[i], flat[i + 1]))

            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in ("set_downstream", "set_upstream"):
                    left_refs = _extract_refs(node.func.value)
                    right_refs = _extract_refs(node.args[0]) if node.args else []
                    if attr == "set_downstream":
                        for l in left_refs:
                            for r in right_refs:
                                edges.add((l, r))
                    else:
                        for l in left_refs:
                            for r in right_refs:
                                edges.add((r, l))

    def resolve(name: str) -> str:
        base = name.split(".", 1)[0]
        if base in var_to_task_id:
            mapped = var_to_task_id[base]
            if "." in name:
                return mapped + "." + name.split(".", 1)[1]
            return mapped
        return name

    edges_resolved = [{"from": resolve(fr), "to": resolve(to)} for (fr, to) in sorted(edges)]

    # schedule extraction
    schedule_expr = None
    m = re.search(r"schedule_interval\s*=\s*([^,\n)]+)", code)
    if m:
        schedule_expr = m.group(1).strip()
    if schedule_expr is None:
        m2 = re.search(r"schedule\s*=\s*([^,\n)]+)", code)
        if m2:
            schedule_expr = m2.group(1).strip()

    catchup = None
    m3 = re.search(r"catchup\s*=\s*(True|False)", code)
    if m3:
        catchup = (m3.group(1) == "True")

    schedule_type = "unknown"
    if schedule_expr is None or schedule_expr in ("None", "null"):
        schedule_type = "manual"
    elif "timedelta" in (schedule_expr or ""):
        schedule_type = "interval"
    else:
        schedule_type = "cron"

    return {
        "orchestrator": "airflow",
        "steps": tasks,
        "edges": edges_resolved,
        "schedule_type": schedule_type,
        "schedule_expression": schedule_expr,
        "catchup_backfill": catchup,
    }


# ---- Prefect / Dagster heuristic extraction ----

def infer_mechanism_from_function_body(func: ast.FunctionDef) -> str:
    src = ast.unparse(func) if hasattr(ast, "unparse") else ""
    low = src.lower()

    if "dockercontainer" in low or "prefect_docker" in low or "docker" in low:
        return "container_run"
    if "requests.get" in low or "requests.post" in low or "httpx" in low:
        return "http_request"
    if "subprocess.run" in low or "bash" in low or "shell" in low:
        return "shell_command"
    if "select " in low or "insert " in low or "update " in low or "from " in low:
        return "sql_query"
    if "sleep" in low and ("while" in low or "for " in low):
        return "wait_poll"
    if "smtp" in low or "sendmail" in low or "email" in low:
        return "notification"
    return "python_callable"


def extract_step_params_from_function_body(func: ast.FunctionDef) -> Dict[str, str]:
    """
    Deterministic extraction of some key literals from function bodies for Prefect/Dagster.
    Not perfect, but general and stable:
      - requests URLs
      - docker image=...
      - subprocess commands
      - obvious SQL strings
    """
    params: Dict[str, str] = {}

    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            fn = _call_name(node.func)

            # requests.get(url) / requests.post(url)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                base = node.func.value.id.lower()
                attr = node.func.attr.lower()
                if base in {"requests", "httpx"} and attr in {"get", "post", "put", "delete"}:
                    if node.args:
                        u = _const_to_str(node.args[0])
                        if u:
                            params.setdefault("url", u)

            # DockerContainer(image=...)
            for kw in node.keywords or []:
                if not kw.arg:
                    continue
                if kw.arg in {"image", "network_mode", "namespace", "cluster_context"}:
                    v = _const_to_str(kw.value) or _ast_to_compact_str(kw.value)
                    if v:
                        params.setdefault(kw.arg, v)
                if kw.arg in {"command", "cmds", "arguments"}:
                    v = _list_of_consts_to_str(kw.value) or _ast_to_compact_str(kw.value)
                    if v:
                        params.setdefault(kw.arg, v)

        # detect SQL-ish string constants
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value.strip()
            sl = s.lower()
            if len(s) >= 20 and ("select " in sl or "insert " in sl or "update " in sl or "create table" in sl):
                params.setdefault("sql", s[:220] + ("...[trunc]" if len(s) > 220 else ""))

    # env var reads within function
    src = ast.unparse(func) if hasattr(ast, "unparse") else ""
    envs = _sorted_unique_str(
        _ENV_GETENV_RE.findall(src) +
        _ENV_OSENV_RE.findall(src) +
        _ENV_OSENVGET_RE.findall(src)
    )
    if envs:
        params.setdefault("env_vars", ",".join(envs[:30]))

    return params


def _is_task_call(call: ast.Call, task_names: Set[str]) -> Optional[str]:
    if isinstance(call.func, ast.Name) and call.func.id in task_names:
        return call.func.id
    if isinstance(call.func, ast.Attribute) and call.func.attr == "submit":
        if isinstance(call.func.value, ast.Name) and call.func.value.id in task_names:
            return call.func.value.id
    return None

def _extract_var_names(node: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and n.attr == "result" and isinstance(n.value, ast.Name):
            out.add(n.value.id)
    return out


def extract_prefect_features(code: str, tree: ast.AST) -> Dict[str, Any]:
    task_funcs: Dict[str, ast.FunctionDef] = {}
    flow_func: Optional[ast.FunctionDef] = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            decs = [_decorator_name(d) for d in node.decorator_list]
            if "task" in decs:
                task_funcs[node.name] = node
            if "flow" in decs and flow_func is None:
                flow_func = node

    task_names = set(task_funcs.keys())
    steps = []
    for name, fn in sorted(task_funcs.items(), key=lambda kv: kv[0]):
        mech = infer_mechanism_from_function_body(fn)
        params = extract_step_params_from_function_body(fn)
        steps.append({"step_id": name, "mechanism": mech, "operator": "prefect_task", "params": params})

    edges: Set[Tuple[str, str]] = set()
    if flow_func:
        var_to_task: Dict[str, str] = {}

        for stmt in flow_func.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                called = _is_task_call(stmt.value, task_names)
                if called:
                    deps: Set[str] = set()
                    for arg in stmt.value.args:
                        deps |= _extract_var_names(arg)
                    for kw in stmt.value.keywords or []:
                        if kw.value:
                            deps |= _extract_var_names(kw.value)

                    for d in deps:
                        if d in var_to_task:
                            edges.add((var_to_task[d], called))

                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            var_to_task[tgt.id] = called

            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                called = _is_task_call(stmt.value, task_names)
                if called:
                    deps: Set[str] = set()
                    for arg in stmt.value.args:
                        deps |= _extract_var_names(arg)
                    for kw in stmt.value.keywords or []:
                        if kw.value:
                            deps |= _extract_var_names(kw.value)
                    for d in deps:
                        if d in var_to_task:
                            edges.add((var_to_task[d], called))

    return {
        "orchestrator": "prefect",
        "steps": steps,
        "edges": [{"from": a, "to": b} for (a, b) in sorted(edges)],
        "schedule_type": "unknown",
        "schedule_expression": None,
        "catchup_backfill": None,
    }


def extract_dagster_features(code: str, tree: ast.AST) -> Dict[str, Any]:
    op_funcs: Dict[str, ast.FunctionDef] = {}
    job_func: Optional[ast.FunctionDef] = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            decs = [_decorator_name(d) for d in node.decorator_list]
            if "op" in decs or "asset" in decs:
                op_funcs[node.name] = node
            if "job" in decs and job_func is None:
                job_func = node

    op_names = set(op_funcs.keys())
    steps = []
    for name, fn in sorted(op_funcs.items(), key=lambda kv: kv[0]):
        mech = infer_mechanism_from_function_body(fn)
        params = extract_step_params_from_function_body(fn)
        steps.append({"step_id": name, "mechanism": mech, "operator": "dagster_op", "params": params})

    edges: Set[Tuple[str, str]] = set()
    if job_func:
        var_to_op: Dict[str, str] = {}

        for stmt in job_func.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id in op_names:
                    called = stmt.value.func.id

                    deps: Set[str] = set()
                    for arg in stmt.value.args:
                        deps |= _extract_var_names(arg)
                    for kw in stmt.value.keywords or []:
                        if kw.value:
                            deps |= _extract_var_names(kw.value)
                    for d in deps:
                        if d in var_to_op:
                            edges.add((var_to_op[d], called))

                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            var_to_op[tgt.id] = called

            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if isinstance(stmt.value.func, ast.Name) and stmt.value.func.id in op_names:
                    called = stmt.value.func.id
                    deps: Set[str] = set()
                    for arg in stmt.value.args:
                        deps |= _extract_var_names(arg)
                    for kw in stmt.value.keywords or []:
                        if kw.value:
                            deps |= _extract_var_names(kw.value)
                    for d in deps:
                        if d in var_to_op:
                            edges.add((var_to_op[d], called))

    return {
        "orchestrator": "dagster",
        "steps": steps,
        "edges": [{"from": a, "to": b} for (a, b) in sorted(edges)],
        "schedule_type": "unknown",
        "schedule_expression": None,
        "catchup_backfill": None,
    }


def extract_generic_tokens(code: str) -> Dict[str, Any]:
    urls = _sorted_unique_str(_URL_RE.findall(code))
    files = _sorted_unique_str([p for p in _FILE_RE.findall(code) if "/" in p and len(p) >= 4])

    images = _sorted_unique_str(_IMAGE_KW_RE.findall(code))
    envs = _sorted_unique_str(
        _ENV_GETENV_RE.findall(code) +
        _ENV_OSENV_RE.findall(code) +
        _ENV_OSENVGET_RE.findall(code)
    )
    conns = _sorted_unique_str(_CONN_RE.findall(code))

    retries = _sorted_unique_str(re.findall(r"retries\s*=\s*([0-9]+)", code))
    retry_delays = _sorted_unique_str(re.findall(r"retry_delay\s*=\s*([^\n,)]+)", code))
    timeouts = _sorted_unique_str(re.findall(r"(?:timeout|execution_timeout)\s*=\s*([^\n,)]+)", code))

    return {
        "urls": urls[:60],
        "files": files[:60],
        "images": images[:60],
        "env_vars": envs[:60],
        "connections": conns[:60],
        "retries": retries[:40],
        "retry_delays": retry_delays[:40],
        "timeouts": timeouts[:40],
    }


# ----------------------------
# Candidate signatures (features → signature text)
# ----------------------------

def candidate_to_signatures(features: Dict[str, Any], tokens: Dict[str, Any], *, max_items: int = 30) -> SignatureTexts:
    pattern = "unknown"
    schedule_type = _safe_str(features.get("schedule_type") or "unknown")
    schedule_expr = features.get("schedule_expression")

    steps = [s for s in (features.get("steps") or []) if isinstance(s, dict)]
    edges = [e for e in (features.get("edges") or []) if isinstance(e, dict)]

    steps_sorted = sorted(steps, key=lambda x: _safe_str(x.get("step_id") or ""))
    edges_sorted = sorted(edges, key=lambda x: (_safe_str(x.get("from")), _safe_str(x.get("to"))))

    struct_lines: List[str] = []
    struct_lines.append(f"PATTERN: {pattern}")
    struct_lines.append(f"SCHEDULE_TYPE: {schedule_type}")

    for st in steps_sorted[:max_items]:
        sid = _safe_str(st.get("step_id") or "step")
        mech = _safe_str(st.get("mechanism") or "unknown")
        obj = sid.replace("_", " ")[:120]
        struct_lines.append(f"STEP: {sid} | MECH: {mech} | OBJ: {obj}")

    for e in edges_sorted[:max_items * 2]:
        fr = _safe_str(e.get("from"))
        to = _safe_str(e.get("to"))
        if fr and to:
            struct_lines.append(f"EDGE: {fr} -> {to}")

    structural_text = "\n".join([ln for ln in struct_lines if ln and str(ln).strip()])

    full_lines: List[str] = []
    full_lines.extend(struct_lines)
    full_lines.append(f"SCHEDULE_EXPR: {schedule_expr}")
    full_lines.append("START_DATE: unknown")
    full_lines.append(f"CATCHUP_BACKFILL: {features.get('catchup_backfill')}")

    # Tokens (generic)
    for u in (tokens.get("urls") or [])[:max_items]:
        full_lines.append(f"URL: {u}")
    for img in (tokens.get("images") or [])[:max_items]:
        full_lines.append(f"IMAGE: {img}")
    for f in (tokens.get("files") or [])[:max_items]:
        full_lines.append(f"FILE: {f}")
    for ev in (tokens.get("env_vars") or [])[:max_items]:
        full_lines.append(f"ENV_VAR: {ev}")
    for c in (tokens.get("connections") or [])[:max_items]:
        full_lines.append(f"CONNECTION: {c}")

    for r in (tokens.get("retries") or [])[:max_items]:
        full_lines.append(f"RETRIES: {r}")
    for rd in (tokens.get("retry_delays") or [])[:max_items]:
        full_lines.append(f"RETRY_DELAY: {rd}")
    for t in (tokens.get("timeouts") or [])[:max_items]:
        full_lines.append(f"TIMEOUT: {t}")

    # NEW: include deterministic step params (critical for Docker/K8s/HTTP/SQL)
    for st in steps_sorted[:max_items]:
        sid = _safe_str(st.get("step_id") or "step")
        params = st.get("params") or {}
        if isinstance(params, dict):
            for k in sorted(params.keys())[:max_items]:
                v = params.get(k)
                if v is None:
                    continue
                v_str = _safe_str(v)
                if not v_str:
                    continue
                if len(v_str) > 220:
                    v_str = v_str[:220] + "...[trunc]"
                full_lines.append(f"PARAM: {sid} | {k}={v_str}")

    full_text = "\n".join([ln for ln in full_lines if ln and str(ln).strip()])
    return SignatureTexts(structural=structural_text, full=full_text)


# ----------------------------
# Metric computation
# ----------------------------

@dataclass
class SimilarityMetrics:
    rouge1_f1: Optional[float]
    rougeL_f1: Optional[float]
    bert_p: Optional[float]
    bert_r: Optional[float]
    bert_f1: Optional[float]


class SemanticScorers:
    def __init__(self, *, bert_model: str, device: Optional[str] = None, use_rouge: bool = True, use_bert: bool = True):
        self.bert_model = bert_model
        self.use_rouge = use_rouge
        self.use_bert = use_bert

        self.rouge = None
        if use_rouge and _HAS_ROUGE:
            self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

        self.bert = None
        self.device = device

        if use_bert and _HAS_BERTSCORE:
            dev = device
            if dev is None:
                # prefer MPS if available; else CUDA; else CPU
                try:
                    if torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        dev = "mps"
                    else:
                        dev = "cuda" if torch and torch.cuda.is_available() else "cpu"
                except Exception:
                    dev = "cpu"

            # infer num_layers for models not in bert_score's internal mapping
            num_layers = None
            try:
                cfg = AutoConfig.from_pretrained(bert_model)
                num_layers = getattr(cfg, "num_hidden_layers", None)
            except Exception:
                num_layers = None

            if num_layers is None and bert_model == "microsoft/codebert-base":
                num_layers = 12

            self.bert = BERTScorer(
                model_type=bert_model,
                num_layers=num_layers,
                lang="en",
                rescale_with_baseline=False,
                device=dev,
            )
            self.device = dev

    def rouge_scores(self, cand: str, ref: str) -> Tuple[Optional[float], Optional[float]]:
        if not self.rouge:
            return None, None
        s = self.rouge.score(ref, cand)
        return float(s["rouge1"].fmeasure), float(s["rougeL"].fmeasure)

    def bert_scores_batch(self, cands: List[str], refs: List[str]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
        if not self.bert:
            n = len(cands)
            return [None] * n, [None] * n, [None] * n

        P, R, F = self.bert.score(cands, refs)
        return [float(x) for x in P.tolist()], [float(x) for x in R.tolist()], [float(x) for x in F.tolist()]


def combine_fidelity_0_10(rougeL_f1: Optional[float], bert_f1: Optional[float], *, mode: str) -> float:
    """
    Combine ROUGE-L and BERTScore into a 0..10 score.

    mode:
      - "structural": emphasize semantic overlap (BERT heavier)
      - "full": emphasize lexical overlap (ROUGE heavier)

    If bert is missing, uses ROUGE only; if ROUGE missing, uses bert only.
    """
    if rougeL_f1 is None and bert_f1 is None:
        return 0.0

    if mode == "full":
        w_rouge, w_bert = 0.7, 0.3
    else:
        w_rouge, w_bert = 0.4, 0.6

    if bert_f1 is None:
        return _clamp10(10.0 * float(rougeL_f1 or 0.0))
    if rougeL_f1 is None:
        return _clamp10(10.0 * float(bert_f1 or 0.0))

    return _clamp10(10.0 * (w_rouge * float(rougeL_f1) + w_bert * float(bert_f1)))


# ----------------------------
# Semantic analyzer evaluator
# ----------------------------

class SemanticAnalyzer(BaseEvaluator):
    EVALUATION_TYPE = "semantic_fidelity"

    def __init__(
        self,
        pipeline_specs_dir: Path = Path("pipeline_specs"),
        pipeline_variants_dir: Path = Path("pipeline_variants"),
        reference_mode: str = "both",  # oracle|variant|both
        bert_model: str = "microsoft/codebert-base",
        device: Optional[str] = None,
        max_items: int = 30,
    ):
        super().__init__(config=None)
        self.pipeline_specs_dir = Path(pipeline_specs_dir)
        self.pipeline_variants_dir = Path(pipeline_variants_dir)
        self.reference_mode = reference_mode
        self.bert_model = bert_model
        self.device = device
        self.max_items = int(max_items)

        self.scorers = SemanticScorers(
            bert_model=bert_model,
            device=device,
            use_rouge=True,
            use_bert=True,
        )

    def evaluate(self, file_path: Path) -> EvaluationResult:
        file_path = Path(file_path)
        self.logger.info(f"Running SemanticAnalyzer on {file_path}")

        result = EvaluationResult(
            evaluation_type=self.EVALUATION_TYPE,
            file_path=str(file_path),
            orchestrator=Orchestrator.UNKNOWN,
            timestamp=datetime.now().isoformat(),
            metadata={},
        )

        # Gate 1: file readable
        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            result.gate_checks.append(GateCheckResult(
                name="file_readable",
                passed=False,
                message=f"Failed to read file: {e}",
                is_critical=True,
            ))
            return self._finalize_failure(result)

        if not code.strip():
            result.gate_checks.append(GateCheckResult(
                name="file_nonempty",
                passed=False,
                message="File is empty",
                is_critical=True,
            ))
            return self._finalize_failure(result)

        # Gate 2: syntax valid
        try:
            tree = ast.parse(code)
            result.gate_checks.append(GateCheckResult(
                name="syntax_valid",
                passed=True,
                message="Python syntax valid",
                is_critical=True,
            ))
        except SyntaxError as e:
            result.gate_checks.append(GateCheckResult(
                name="syntax_valid",
                passed=False,
                message=f"SyntaxError line {e.lineno}: {e.msg}",
                is_critical=True,
            ))
            return self._finalize_failure(result)

        # Detect orchestrator
        orch = self.detect_orchestrator(code)
        result.orchestrator = orch
        result.metadata["detected_orchestrator"] = orch.value
        result.metadata["file_size_bytes"] = len(code.encode("utf-8"))
        result.metadata["line_count"] = len(code.splitlines())

        # Resolve run context
        ctx = resolve_run_context(file_path)
        pipeline_id = ctx.get("pipeline_id")
        class_id = ctx.get("class_id")
        result.metadata["run_context"] = ctx

        # Oracle reference (critical if needed)
        oracle_path = None
        oracle_spec = None
        need_oracle = self.reference_mode in ("oracle", "both")

        if need_oracle:
            if pipeline_id:
                oracle_path = find_oracle_spec(self.pipeline_specs_dir, pipeline_id)

            if oracle_path is None:
                result.gate_checks.append(GateCheckResult(
                    name="oracle_reference_found",
                    passed=False,
                    message=f"Oracle PipelineSpec not found for pipeline_id={pipeline_id}",
                    is_critical=True,
                ))
                return self._finalize_failure(result)

            oracle_spec = load_json(oracle_path)
            result.gate_checks.append(GateCheckResult(
                name="oracle_reference_found",
                passed=True,
                message=f"Oracle reference loaded: {oracle_path}",
                is_critical=True,
            ))
            result.metadata["oracle_reference_path"] = str(oracle_path)

        # Variant reference (non-critical)
        variant_path = None
        variant_spec = None
        need_variant = self.reference_mode in ("variant", "both")

        if need_variant:
            if pipeline_id and class_id:
                variant_path = find_variant_spec(self.pipeline_variants_dir, pipeline_id, class_id)
            if variant_path is not None:
                variant_spec = load_json(variant_path)
                result.gate_checks.append(GateCheckResult(
                    name="variant_reference_found",
                    passed=True,
                    message=f"Variant reference loaded: {variant_path}",
                    is_critical=False,
                ))
                result.metadata["variant_reference_path"] = str(variant_path)
            else:
                result.gate_checks.append(GateCheckResult(
                    name="variant_reference_found",
                    passed=False,
                    message=f"Variant spec not found for pipeline_id={pipeline_id} class_id={class_id}",
                    is_critical=False,
                ))

        # Candidate extraction
        cand_features = self._extract_candidate_features(code, tree, orch)
        cand_tokens = extract_generic_tokens(code)
        cand_sigs = candidate_to_signatures(cand_features, cand_tokens, max_items=self.max_items)

        # summarize candidate extraction
        param_count = 0
        for st in cand_features.get("steps") or []:
            if isinstance(st, dict) and isinstance(st.get("params"), dict):
                param_count += len(st["params"])

        result.metadata["candidate_extraction_summary"] = {
            "steps_count": len(cand_features.get("steps") or []),
            "edges_count": len(cand_features.get("edges") or []),
            "step_param_pairs_extracted": param_count,
            "token_counts": {k: len(v or []) if isinstance(v, list) else None for k, v in cand_tokens.items()},
        }

        # Prepare metric pairs
        pairs: List[Tuple[str, str, str]] = []
        if oracle_spec is not None:
            oracle_sigs = spec_to_signatures(oracle_spec, max_items=self.max_items)
            pairs.append(("oracle_structural", cand_sigs.structural, oracle_sigs.structural))
            pairs.append(("oracle_full", cand_sigs.full, oracle_sigs.full))
        if variant_spec is not None:
            variant_sigs = spec_to_signatures(variant_spec, max_items=self.max_items)
            pairs.append(("variant_structural", cand_sigs.structural, variant_sigs.structural))
            pairs.append(("variant_full", cand_sigs.full, variant_sigs.full))

        rouge_by_key: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        for key, cand_text, ref_text in pairs:
            r1, rl = self.scorers.rouge_scores(cand_text, ref_text)
            rouge_by_key[key] = (r1, rl)

        cand_texts = [p[1] for p in pairs]
        ref_texts = [p[2] for p in pairs]
        bert_p, bert_r, bert_f = self.scorers.bert_scores_batch(cand_texts, ref_texts)

        global_metric_issues: List[Issue] = []
        if not _HAS_ROUGE:
            global_metric_issues.append(Issue(
                severity=Severity.MAJOR,
                category="metric",
                message="rouge_score not installed; ROUGE metrics unavailable (pip install rouge-score)"
            ))
        if not _HAS_BERTSCORE:
            global_metric_issues.append(Issue(
                severity=Severity.MAJOR,
                category="metric",
                message="bert_score/torch not installed; BERTScore unavailable (pip install bert-score torch transformers)"
            ))

        result.metadata["metric_config"] = {
            "reference_mode": self.reference_mode,
            "rouge_available": bool(_HAS_ROUGE),
            "bertscore_available": bool(_HAS_BERTSCORE),
            "bert_model": self.bert_model,
            "device": self.scorers.device,
            "max_items": self.max_items,
            "combine_weights": {
                "structural": {"rougeL": 0.4, "bert": 0.6},
                "full": {"rougeL": 0.7, "bert": 0.3},
            },
            "overall_weights": {"structural": 0.7, "full": 0.3},
        }

        @dataclass
        class _Pair:
            key: str
            metrics: SimilarityMetrics

        key_to_index = {pairs[i][0]: i for i in range(len(pairs))}

        def pair_metrics(key: str) -> SimilarityMetrics:
            idx = key_to_index[key]
            r1, rl = rouge_by_key.get(key, (None, None))
            return SimilarityMetrics(
                rouge1_f1=r1,
                rougeL_f1=rl,
                bert_p=bert_p[idx] if idx < len(bert_p) else None,
                bert_r=bert_r[idx] if idx < len(bert_r) else None,
                bert_f1=bert_f[idx] if idx < len(bert_f) else None,
            )

        def add_score(name: str, raw: float, details: Dict[str, Any], issues: List[Issue]) -> None:
            result.scores[name] = EvaluationScore(
                name=name,
                raw_score=_clamp10(raw),
                weight=1.0,
                issues=issues,
                details=details,
                penalties_applied=0.0,
            )

        # ORACLE scoring
        if oracle_spec is not None:
            m_struct = pair_metrics("oracle_structural")
            m_full = pair_metrics("oracle_full")

            f_struct = combine_fidelity_0_10(m_struct.rougeL_f1, m_struct.bert_f1, mode="structural")
            f_full = combine_fidelity_0_10(m_full.rougeL_f1, m_full.bert_f1, mode="full")
            f_all = _clamp10(0.7 * f_struct + 0.3 * f_full)

            add_score("oracle_structural", f_struct, {"metrics": m_struct.__dict__}, list(global_metric_issues))
            add_score("oracle_full", f_full, {"metrics": m_full.__dict__}, list(global_metric_issues))
            add_score("oracle_semantic_fidelity", f_all, {"aggregation": "0.7_structural + 0.3_full"}, list(global_metric_issues))

            result.metadata["oracle"] = {
                "fidelity_structural": f_struct,
                "fidelity_full": f_full,
                "semantic_fidelity": f_all,
                "struct_metrics": m_struct.__dict__,
                "full_metrics": m_full.__dict__,
            }

        # VARIANT scoring
        if need_variant:
            if variant_spec is None:
                missing_issue = Issue(
                    severity=Severity.MAJOR,
                    category="reference",
                    message=f"Variant spec not found for pipeline_id={pipeline_id} class_id={class_id}"
                )
                add_score("variant_structural", 0.0, {"error": "missing_variant_spec"}, [missing_issue] + global_metric_issues)
                add_score("variant_full", 0.0, {"error": "missing_variant_spec"}, [missing_issue] + global_metric_issues)
                add_score("variant_semantic_fidelity", 0.0, {"error": "missing_variant_spec"}, [missing_issue] + global_metric_issues)
                result.metadata["variant"] = {"available": False, "error": "missing_variant_spec"}
            else:
                m_struct = pair_metrics("variant_structural")
                m_full = pair_metrics("variant_full")

                f_struct = combine_fidelity_0_10(m_struct.rougeL_f1, m_struct.bert_f1, mode="structural")
                f_full = combine_fidelity_0_10(m_full.rougeL_f1, m_full.bert_f1, mode="full")
                f_all = _clamp10(0.7 * f_struct + 0.3 * f_full)

                add_score("variant_structural", f_struct, {"metrics": m_struct.__dict__}, list(global_metric_issues))
                add_score("variant_full", f_full, {"metrics": m_full.__dict__}, list(global_metric_issues))
                add_score("variant_semantic_fidelity", f_all, {"aggregation": "0.7_structural + 0.3_full"}, list(global_metric_issues))

                result.metadata["variant"] = {
                    "available": True,
                    "fidelity_structural": f_struct,
                    "fidelity_full": f_full,
                    "semantic_fidelity": f_all,
                    "struct_metrics": m_struct.__dict__,
                    "full_metrics": m_full.__dict__,
                }

        # Convenience top-level
        if "oracle_semantic_fidelity" in result.scores:
            result.metadata["semantic_fidelity"] = float(result.scores["oracle_semantic_fidelity"].raw_score)
        else:
            result.metadata["semantic_fidelity"] = 0.0

        return result

    def _finalize_failure(self, result: EvaluationResult) -> EvaluationResult:
        result.metadata["semantic_fidelity"] = 0.0
        return result

    def _extract_candidate_features(self, code: str, tree: ast.AST, orch: Orchestrator) -> Dict[str, Any]:
        if orch == Orchestrator.AIRFLOW:
            return extract_airflow_features(code, tree)
        if orch == Orchestrator.PREFECT:
            return extract_prefect_features(code, tree)
        if orch == Orchestrator.DAGSTER:
            return extract_dagster_features(code, tree)
        return {
            "orchestrator": "unknown",
            "steps": [],
            "edges": [],
            "schedule_type": "unknown",
            "schedule_expression": None,
            "catchup_backfill": None,
        }


# ----------------------------
# CLI payload helpers
# ----------------------------

def build_payload(result: EvaluationResult) -> Dict[str, Any]:
    payload = result.to_dict()
    payload["issues"] = [i.to_dict() for i in result.all_issues]
    payload["issue_summary"] = {
        "total": len(payload["issues"]),
        "critical": sum(1 for i in payload["issues"] if i.get("severity") == "critical"),
        "major": sum(1 for i in payload["issues"] if i.get("severity") == "major"),
        "minor": sum(1 for i in payload["issues"] if i.get("severity") == "minor"),
        "info": sum(1 for i in payload["issues"] if i.get("severity") == "info"),
    }
    return payload

def default_sidecar_path(code_file: Path) -> Path:
    return code_file.with_name(code_file.name + ".semantic.json")

def print_summary(payload: Dict[str, Any]) -> None:
    meta = payload.get("metadata", {}) or {}
    ctx = meta.get("run_context", {}) or {}
    oracle = meta.get("oracle", {}) or {}
    variant = meta.get("variant", {}) or {}

    print("\n" + "=" * 80)
    print("SEMANTIC ANALYZER SUMMARY (ROUGE + BERTScore)")
    print("=" * 80)
    print(f"File:         {payload.get('file_path')}")
    print(f"Orchestrator: {payload.get('orchestrator')}")
    print(f"Pipeline:     {ctx.get('pipeline_id')}")
    print(f"Variant:      {ctx.get('variant_stem')}  (class={ctx.get('class_id')})")
    mc = meta.get("metric_config", {}) or {}
    print(f"BERT model:   {mc.get('bert_model')}")
    print(f"Device:       {mc.get('device')}")
    print(f"Max items:    {mc.get('max_items')}")
    print(f"Params extracted: {meta.get('candidate_extraction_summary', {}).get('step_param_pairs_extracted')}")

    if oracle:
        print("\nORACLE REFERENCE")
        print(f"  structural: {oracle.get('fidelity_structural', 0.0):.2f}/10")
        print(f"  full:       {oracle.get('fidelity_full', 0.0):.2f}/10")
        print(f"  overall:    {oracle.get('semantic_fidelity', 0.0):.2f}/10")

    if isinstance(variant, dict) and variant.get("available"):
        print("\nVARIANT REFERENCE")
        print(f"  structural: {variant.get('fidelity_structural', 0.0):.2f}/10")
        print(f"  full:       {variant.get('fidelity_full', 0.0):.2f}/10")
        print(f"  overall:    {variant.get('semantic_fidelity', 0.0):.2f}/10")
    else:
        print("\nVARIANT REFERENCE")
        print(f"  available:  {variant.get('available', False)}  error={variant.get('error')}")

    isumm = payload.get("issue_summary", {}) or {}
    print("\nISSUES")
    print(f"  total={isumm.get('total', 0)} critical={isumm.get('critical', 0)} "
          f"major={isumm.get('major', 0)} minor={isumm.get('minor', 0)} info={isumm.get('info', 0)}")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic fidelity evaluator: PipelineSpec ↔ Generated code (ROUGE + BERTScore).")
    parser.add_argument("file", help="Path to generated workflow Python file")

    parser.add_argument("--pipeline-specs-dir", default="pipeline_specs")
    parser.add_argument("--pipeline-variants-dir", default="pipeline_variants")
    parser.add_argument("--reference-mode", default="both", choices=["oracle", "variant", "both"])
    parser.add_argument("--bert-model", default="microsoft/codebert-base")
    parser.add_argument("--device", default=None, help="Force device: cpu|cuda|mps (optional)")
    parser.add_argument("--max-items", type=int, default=30)

    parser.add_argument("--out", default=None, help="Write JSON to this exact path")
    parser.add_argument("--out-dir", default=None, help="Write JSON to this directory (auto filename)")
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--print-summary", action="store_true")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")

    code_file = Path(args.file)
    analyzer = SemanticAnalyzer(
        pipeline_specs_dir=Path(args.pipeline_specs_dir),
        pipeline_variants_dir=Path(args.pipeline_variants_dir),
        reference_mode=args.reference_mode,
        bert_model=args.bert_model,
        device=args.device,
        max_items=args.max_items,
    )

    result = analyzer.evaluate(code_file)
    payload = build_payload(result)

    if args.print_summary:
        print_summary(payload)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    elif args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"semantic_{payload.get('orchestrator')}_{code_file.stem}_{ts}.json"
    else:
        out_path = default_sidecar_path(code_file)

    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote: {out_path}")

    if args.stdout:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()