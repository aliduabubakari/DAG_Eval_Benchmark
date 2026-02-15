#!/usr/bin/env python3
"""
GT1 Mutation Injection using Azure OpenAI - Version 2 (Edit-Operation Based)

Key improvements over v1:
- Uses edit operations (patch-based) instead of full-file output
- Dramatically reduced token cost
- Guaranteed full-file mutations
- Exact injected_line tracking
- True ReAct loop with validation feedback

Input dir must contain subdirs: airflow/, prefect/, dagster/
For each original file, produce mutated file(s) + GT1 JSON labels
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def sha256_12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> str:
    """Extract JSON object from LLM response, handling markdown code blocks."""
    t = (text or "").strip()
    if "```json" in t:
        t = t.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in t:
        t = t.split("```", 1)[1].split("```", 1)[0]
    t = t.strip()
    s = t.find("{")
    e = t.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return t
    return t[s:e + 1]


def clamp_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def add_line_numbers(code: str) -> str:
    """Add line numbers to code for the prompt."""
    lines = code.splitlines()
    numbered = []
    for i, line in enumerate(lines, start=1):
        numbered.append(f"{i:4d} | {line}")
    return "\n".join(numbered)


# -----------------------------------------------------------------------------
# Mutation catalog
# -----------------------------------------------------------------------------

@dataclass
class MutationSpec:
    id: str
    category: str
    severity: str
    orchestrators: List[str]
    description: str
    validation: Dict[str, Any]


def load_catalog(path: Path) -> List[MutationSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    muts = []
    for m in (data.get("mutations") or []):
        muts.append(MutationSpec(
            id=m["id"],
            category=m["category"],
            severity=m["severity"],
            orchestrators=list(m.get("orchestrators", [])),
            description=m.get("description", ""),
            validation=dict(m.get("validation", {})),
        ))
    return muts


# -----------------------------------------------------------------------------
# Dynamic sentinel tokens per file
# -----------------------------------------------------------------------------

def make_injection_constants(file_id: str) -> Dict[str, str]:
    """Generate unique sentinel tokens for this file to enable objective validation."""
    return {
        "NONEXISTENT_MODULE_NAME": f"__definitely_nonexistent_module_{file_id}__",
        "UNDEFINED_NAME": f"__undefined_name_{file_id}__",
        "UNUSED_IMPORT_ALIAS": f"__unused_import_alias_{file_id}__",
        "UNUSED_VAR_NAME": f"__unused_var_{file_id}__",
        "HARDCODED_SECRET_VALUE": f"sk_live_{file_id}_synthetic_secret_ABC123XYZ",
        "EVAL_SENTINEL": f"__eval_sentinel_{file_id}__",
        "SHELL_SENTINEL": f"__shell_sentinel_{file_id}__",
        "DUP_TASK_ID": f"dup_task_{file_id}",
    }


# -----------------------------------------------------------------------------
# Edit Operation Types and Application
# -----------------------------------------------------------------------------

@dataclass
class EditOperation:
    """Represents a single edit operation to apply to source code."""
    op: str  # replace_line, delete_line, insert_after, insert_before, replace_lines, delete_lines
    line_number: int  # 1-indexed
    end_line_number: Optional[int] = None  # For range operations (inclusive)
    new_content: Optional[str] = None  # Content to insert/replace with (can be multi-line)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EditOperation":
        return cls(
            op=d.get("op", ""),
            line_number=clamp_int(d.get("line_number"), 0),
            end_line_number=clamp_int(d.get("end_line_number"), 0) if d.get("end_line_number") else None,
            new_content=d.get("new_content"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "line_number": self.line_number,
            "end_line_number": self.end_line_number,
            "new_content": self.new_content,
        }


def apply_edit(original_code: str, edit: EditOperation) -> Tuple[str, Optional[int], str]:
    """
    Apply an edit operation to the original code.
    
    Returns:
        (mutated_code, injected_line_number, error_message)
        If error_message is non-empty, the edit failed.
    """
    lines = original_code.splitlines()
    n = len(lines)
    
    # Validate line number (1-indexed)
    ln = edit.line_number
    if ln < 1:
        return original_code, None, f"Invalid line_number {ln}: must be >= 1"
    
    # Handle each operation type
    op = edit.op.lower().strip()
    
    if op == "replace_line":
        if ln > n:
            return original_code, None, f"line_number {ln} exceeds file length {n}"
        if edit.new_content is None:
            return original_code, None, "replace_line requires new_content"
        
        new_lines = edit.new_content.splitlines() if edit.new_content else [""]
        result = lines[:ln-1] + new_lines + lines[ln:]
        return "\n".join(result), ln, ""
    
    elif op == "delete_line":
        if ln > n:
            return original_code, None, f"line_number {ln} exceeds file length {n}"
        
        result = lines[:ln-1] + lines[ln:]
        return "\n".join(result), ln, ""
    
    elif op == "insert_after":
        if ln > n:
            return original_code, None, f"line_number {ln} exceeds file length {n}"
        if edit.new_content is None:
            return original_code, None, "insert_after requires new_content"
        
        new_lines = edit.new_content.splitlines()
        result = lines[:ln] + new_lines + lines[ln:]
        return "\n".join(result), ln + 1, ""  # injected line is the first new line
    
    elif op == "insert_before":
        if ln > n + 1:  # Can insert before line n+1 (i.e., at end)
            return original_code, None, f"line_number {ln} exceeds file length + 1"
        if edit.new_content is None:
            return original_code, None, "insert_before requires new_content"
        
        new_lines = edit.new_content.splitlines()
        result = lines[:ln-1] + new_lines + lines[ln-1:]
        return "\n".join(result), ln, ""  # injected line is where we inserted
    
    elif op == "replace_lines":
        end_ln = edit.end_line_number or ln
        if ln > n or end_ln > n:
            return original_code, None, f"line range {ln}-{end_ln} exceeds file length {n}"
        if end_ln < ln:
            return original_code, None, f"end_line_number {end_ln} < line_number {ln}"
        if edit.new_content is None:
            return original_code, None, "replace_lines requires new_content"
        
        new_lines = edit.new_content.splitlines() if edit.new_content else []
        result = lines[:ln-1] + new_lines + lines[end_ln:]
        return "\n".join(result), ln, ""
    
    elif op == "delete_lines":
        end_ln = edit.end_line_number or ln
        if ln > n or end_ln > n:
            return original_code, None, f"line range {ln}-{end_ln} exceeds file length {n}"
        if end_ln < ln:
            return original_code, None, f"end_line_number {end_ln} < line_number {ln}"
        
        result = lines[:ln-1] + lines[end_ln:]
        return "\n".join(result), ln, ""
    
    elif op == "append_line":
        # Special case: append at end of file
        if edit.new_content is None:
            return original_code, None, "append_line requires new_content"
        
        new_lines = edit.new_content.splitlines()
        result = lines + new_lines
        return "\n".join(result), n + 1, ""
    
    elif op == "prepend_line":
        # Special case: insert at beginning
        if edit.new_content is None:
            return original_code, None, "prepend_line requires new_content"
        
        new_lines = edit.new_content.splitlines()
        result = new_lines + lines
        return "\n".join(result), 1, ""
    
    else:
        return original_code, None, f"Unknown operation: {op}"


# -----------------------------------------------------------------------------
# Deterministic validators (objective checks)
# -----------------------------------------------------------------------------

def is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def airflow_has_dag_definition(code: str) -> bool:
    cl = code.lower()
    return ("with dag(" in cl) or ("dag(" in cl and "=" in cl) or ("@dag" in code)


def airflow_has_tasks(code: str) -> bool:
    return bool(re.search(r"\w+Operator\s*\(", code)) or ("@task" in code)


def airflow_has_dependencies(code: str) -> bool:
    return (">>" in code) or ("<<" in code) or bool(re.search(r"chain\s*\(", code.lower()))


def prefect_has_flow(code: str) -> bool:
    return "@flow" in code


def dagster_has_job_or_asset(code: str) -> bool:
    return ("@job" in code) or ("@graph" in code) or ("@asset" in code)


def dagster_has_op_or_asset(code: str) -> bool:
    return ("@op" in code) or ("@asset" in code)

def _module_level_defines_name(tree: ast.AST, name: str) -> bool:
    """
    True if the module (top-level) defines name via:
      - assignment: name = ...
      - annotated assignment: name: T = ...
      - with-as: with ... as name:
    """
    if not isinstance(tree, ast.Module):
        return False

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return True
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
                return True
        elif isinstance(stmt, ast.With):
            for item in stmt.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name) and item.optional_vars.id == name:
                    return True

    return False


def _module_level_uses_name(tree: ast.AST, name: str) -> bool:
    """
    True if module-level (top-level, not inside functions/classes) uses name in Load context.
    This is what causes NameError at import time if name is missing.
    """
    if not isinstance(tree, ast.Module):
        return False

    for stmt in tree.body:
        # ignore nested scopes
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name:
                return True

    return False

def ast_defined_names(tree: ast.AST) -> set:
    """Collect all names that are defined (assigned, function defs, imports, etc.)."""
    defined = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)

        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
            for a in node.args.args + node.args.kwonlyargs:
                defined.add(a.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)

        elif isinstance(node, ast.AsyncFunctionDef):
            defined.add(node.name)
            for a in node.args.args + node.args.kwonlyargs:
                defined.add(a.arg)
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)

        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)

        # Comprehension variables
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in node.generators:
                if isinstance(gen.target, ast.Name):
                    defined.add(gen.target.id)

        # For loop variables
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        defined.add(elt.id)

        # With statement variables
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars and isinstance(item.optional_vars, ast.Name):
                    defined.add(item.optional_vars.id)

        # Exception handler variables
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                defined.add(node.name)

    return defined


def has_bare_except(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            return True
    return False


def has_swallow_exception(code: str) -> bool:
    """Detect pattern: except Exception: pass"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                if h.type is None:
                    continue
                if isinstance(h.type, ast.Name) and h.type.id == "Exception":
                    if len(h.body) == 1 and isinstance(h.body[0], ast.Pass):
                        return True
    return False


def has_eval_call(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "eval":
                return True
    return False


def has_subprocess_shell_true(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords or []:
                if kw.arg == "shell":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        return True
    return False


def extract_airflow_task_ids(code: str) -> List[str]:
    return re.findall(r"task_id\s*=\s*['\"]([^'\"]+)['\"]", code)


def find_line_containing(code: str, token: str) -> Optional[int]:
    if not token:
        return None
    for i, ln in enumerate(code.splitlines(), start=1):
        if token in ln:
            return i
    return None


def validate_mutation(
    orchestrator: str,
    spec: MutationSpec,
    original_code: str,
    mutated_code: str,
    constants: Dict[str, str],
    max_line_delta: int = 20,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Validate that the mutation was applied correctly.
    
    Returns:
        (success, details_dict, error_message)
    """
    details: Dict[str, Any] = {}

    # Check not identical
    if mutated_code.strip() == original_code.strip():
        return False, {"same_as_original": True}, "Mutated code is identical to original."

    # Line count delta check (critical for patch-based approach)
    orig_lines = len(original_code.splitlines())
    mut_lines = len(mutated_code.splitlines())
    line_delta = abs(mut_lines - orig_lines)
    details["original_line_count"] = orig_lines
    details["mutated_line_count"] = mut_lines
    details["line_delta"] = line_delta
    
    if line_delta > max_line_delta:
        return False, details, f"Line count changed too much: {orig_lines} -> {mut_lines} (delta={line_delta}, max={max_line_delta})"

    # Get validation type
    v = spec.validation or {}
    vtype = (v.get("type") or "").strip()
    token_key = v.get("token_key")
    token_val = constants.get(token_key, "") if token_key else ""

    # For non-syntax_invalid mutations, ensure syntax stays valid
    if vtype != "syntax_invalid":
        if not is_syntax_valid(mutated_code):
            return False, details, "Mutation unexpectedly made code syntactically invalid."

    # Validate based on type
    if vtype == "syntax_invalid":
        ok = not is_syntax_valid(mutated_code)
        details["syntax_invalid"] = ok
        return ok, details, ("Expected syntax to be invalid but it still parses." if not ok else "")

    if vtype == "contains_token":
        ok = bool(token_val) and (token_val in mutated_code)
        details["token_key"] = token_key
        details["token_found"] = ok
        return ok, details, (f"Expected token '{token_val}' not found in mutated code." if not ok else "")

    if vtype == "undefined_name_load":
        if not token_val:
            return False, details, "Missing token_val for undefined_name_load"
        try:
            tree = ast.parse(mutated_code)
        except SyntaxError:
            return False, details, "Unexpected syntax error in undefined_name_load validation"

        defined = ast_defined_names(tree)
        used_as_load = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == token_val:
                used_as_load = True
                break

        ok = used_as_load and (token_val not in defined)
        details["used_as_load"] = used_as_load
        details["defined_somewhere"] = (token_val in defined)
        return ok, details, ("Undefined name token not used as load or it was defined somewhere." if not ok else "")

    if vtype == "unused_import_alias_once":
        if not token_val:
            return False, details, "Missing token_val for unused_import_alias_once"
        count = mutated_code.count(token_val)
        details["alias_occurrences"] = count
        ok = (count == 1)
        return ok, details, (f"Unused import alias appears {count} times (should be exactly 1)." if not ok else "")

    if vtype == "unused_variable_once":
        if not token_val:
            return False, details, "Missing token_val for unused_variable_once"
        count = mutated_code.count(token_val)
        details["var_occurrences"] = count
        ok = (count == 1)
        return ok, details, (f"Unused variable appears {count} times (should be exactly 1)." if not ok else "")

    if vtype == "ast_bare_except":
        ok = has_bare_except(mutated_code)
        details["bare_except_found"] = ok
        return ok, details, ("Expected bare 'except:' handler but none found." if not ok else "")

    if vtype == "ast_swallow_exception":
        ok = has_swallow_exception(mutated_code)
        details["swallow_exception_found"] = ok
        return ok, details, ("Expected 'except Exception: pass' pattern but not found." if not ok else "")

    if vtype == "ast_contains_eval":
        has_eval = has_eval_call(mutated_code)
        has_sentinel = (token_val in mutated_code) if token_val else True
        ok = has_eval and has_sentinel
        details["has_eval_call"] = has_eval
        details["has_sentinel"] = has_sentinel
        return ok, details, ("Expected eval() call with sentinel but not found." if not ok else "")

    if vtype == "ast_subprocess_shell_true":
        has_shell = has_subprocess_shell_true(mutated_code)
        has_sentinel = (token_val in mutated_code) if token_val else True
        ok = has_shell and has_sentinel
        details["has_shell_true"] = has_shell
        details["has_sentinel"] = has_sentinel
        return ok, details, ("Expected subprocess call with shell=True and sentinel but not found." if not ok else "")

    # Orchestrator-specific validations
    if vtype == "airflow_no_dag_definition":
        if orchestrator != "airflow":
            return False, details, "airflow_no_dag_definition applied to non-airflow file"

        ok = not airflow_has_dag_definition(mutated_code)
        details["has_dag_definition_after"] = airflow_has_dag_definition(mutated_code)

        if not ok:
            return False, details, "Expected DAG definition to be removed but it's still present."

        # Extra guard: prevent common failure mode:
        # - DAG definition removed
        # - but tasks still use dag=dag at module level and `dag` is no longer defined
        # => NameError during import => flips imports gate instead of instantiation gate
        try:
            tree = ast.parse(mutated_code)
            dag_used = _module_level_uses_name(tree, "dag")
            dag_defined = _module_level_defines_name(tree, "dag")

            details["dag_used_module_level"] = dag_used
            details["dag_defined_module_level"] = dag_defined

            if dag_used and not dag_defined:
                return False, details, (
                    "airflow_remove_dag_definition left module-level references to 'dag' "
                    "but did not define 'dag'. Fix by either removing dag=dag usages or adding `dag = None` at module level."
                )
        except Exception as e:
            # syntax should already be valid here, but keep safe
            details["dag_guard_exception"] = f"{type(e).__name__}: {e}"

        return True, details, ""

    if vtype == "airflow_no_tasks":
        if orchestrator != "airflow":
            return False, details, "airflow_no_tasks applied to non-airflow file"
        ok = not airflow_has_tasks(mutated_code)
        details["has_tasks_after"] = airflow_has_tasks(mutated_code)
        return ok, details, ("Expected tasks to be removed but they're still present." if not ok else "")

    if vtype == "airflow_no_dependencies":
        if orchestrator != "airflow":
            return False, details, "airflow_no_dependencies applied to non-airflow file"
        if not airflow_has_dependencies(original_code):
            return False, details, "Original had no dependencies; mutation should have been skipped"
        ok = not airflow_has_dependencies(mutated_code)
        details["has_dependencies_after"] = airflow_has_dependencies(mutated_code)
        return ok, details, ("Expected dependencies (>>, <<, chain) to be removed but they're still present." if not ok else "")

    if vtype == "airflow_duplicate_task_id":
        if orchestrator != "airflow":
            return False, details, "airflow_duplicate_task_id applied to non-airflow file"
        if not token_val:
            return False, details, "Missing DUP_TASK_ID token"
        tids = extract_airflow_task_ids(mutated_code)
        dup_count = sum(1 for t in tids if t == token_val)
        details["dup_task_id_occurrences"] = dup_count
        details["all_task_ids"] = tids
        ok = dup_count >= 2
        return ok, details, (f"Expected task_id='{token_val}' to appear >= 2 times but found {dup_count}." if not ok else "")

    if vtype == "prefect_no_flow":
        if orchestrator != "prefect":
            return False, details, "prefect_no_flow applied to non-prefect file"
        ok = not prefect_has_flow(mutated_code)
        details["has_flow_after"] = prefect_has_flow(mutated_code)
        return ok, details, ("Expected @flow decorator to be removed but it's still present." if not ok else "")

    if vtype == "dagster_no_job_or_asset":
        if orchestrator != "dagster":
            return False, details, "dagster_no_job_or_asset applied to non-dagster file"
        ok = not dagster_has_job_or_asset(mutated_code)
        details["has_job_or_asset_after"] = dagster_has_job_or_asset(mutated_code)
        return ok, details, ("Expected @job/@graph/@asset to be removed but still present." if not ok else "")

    if vtype == "dagster_no_op_or_asset":
        if orchestrator != "dagster":
            return False, details, "dagster_no_op_or_asset applied to non-dagster file"
        if not dagster_has_op_or_asset(original_code):
            return False, details, "Original had no ops/assets; mutation should have been skipped"
        ok = not dagster_has_op_or_asset(mutated_code)
        details["has_op_or_asset_after"] = dagster_has_op_or_asset(mutated_code)
        return ok, details, ("Expected @op/@asset to be removed but still present." if not ok else "")

    return False, details, f"Unknown validation type: {vtype}"


def precondition_skip(orchestrator: str, spec: MutationSpec, original_code: str) -> Tuple[bool, str]:
    """
    Check if mutation should be skipped due to preconditions not being met.
    Returns (should_skip, reason).
    """
    vtype = (spec.validation or {}).get("type", "")

    if vtype == "airflow_no_dag_definition":
        if not airflow_has_dag_definition(original_code):
            return True, "Original already has no DAG definition to remove."
            
    if vtype == "airflow_no_tasks":
        if not airflow_has_tasks(original_code):
            return True, "Original already has no tasks to remove."
            
    if vtype == "airflow_no_dependencies":
        if not airflow_has_dependencies(original_code):
            return True, "Original has no dependencies (>>, <<, chain) to remove."
            
    if vtype == "airflow_duplicate_task_id":
        if len(extract_airflow_task_ids(original_code)) == 0:
            return True, "Original has no task_id= assignments to duplicate."
            
    if vtype == "prefect_no_flow":
        if not prefect_has_flow(original_code):
            return True, "Original has no @flow decorator to remove."
            
    if vtype == "dagster_no_job_or_asset":
        if not dagster_has_job_or_asset(original_code):
            return True, "Original has no @job/@graph/@asset to remove."
            
    if vtype == "dagster_no_op_or_asset":
        if not dagster_has_op_or_asset(original_code):
            return True, "Original has no @op/@asset to remove."

    return False, ""


# -----------------------------------------------------------------------------
# Azure OpenAI chat client
# -----------------------------------------------------------------------------

class AzureChatClient:
    def __init__(
        self,
        azure_endpoint: str,
        deployment: str,
        api_version: str,
        api_key: str,
        timeout_s: int = 120,
        max_retries: int = 6,
        rate_limit_delay: float = 0.0,
    ):
        self.azure_endpoint = azure_endpoint.rstrip("/")
        self.deployment = deployment
        self.api_version = api_version
        self.api_key = api_key
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)
        self.rate_limit_delay = float(rate_limit_delay)

        self.url = (
            f"{self.azure_endpoint}/openai/deployments/{self.deployment}/chat/completions"
            f"?api-version={self.api_version}"
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_completion_tokens: int,
        temperature: float = 1.0
    ) -> Tuple[str, Dict[str, Any]]:
        """Make a chat completion request to Azure OpenAI."""
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)

        payload = {
            "messages": messages,
            "max_completion_tokens": int(max_completion_tokens),
            "temperature": float(temperature),
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout_s)
                if r.status_code == 200:
                    data = r.json()
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {}) or {}
                    return content, usage

                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                    sleep_s = min(60, 2 ** attempt)
                    time.sleep(sleep_s)
                    continue

                raise RuntimeError(f"Azure OpenAI error HTTP {r.status_code}: {r.text[:1000]}")

            except requests.exceptions.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                sleep_s = min(60, 2 ** attempt)
                time.sleep(sleep_s)

        raise RuntimeError(f"Azure OpenAI chat failed after {self.max_retries} retries. Last error: {last_err}")


# -----------------------------------------------------------------------------
# Prompt Templates for Edit-Based Mutations
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a code mutation generator for a benchmarking dataset.
You output ONLY valid JSON with a single edit operation to introduce exactly one defect.
No explanations, no markdown, just the JSON object."""

MUTATION_PROMPT_TEMPLATE = """# Task
Introduce EXACTLY ONE defect into the code below using a single edit operation.

## Mutation to introduce
- **mutation_id**: {mutation_id}
- **category**: {category}  
- **severity**: {severity}
- **orchestrator**: {orchestrator}
- **description**: {description}

## Sentinel constants (use EXACTLY as shown when the mutation requires them)
- NONEXISTENT_MODULE_NAME: `{NONEXISTENT_MODULE_NAME}`
- UNDEFINED_NAME: `{UNDEFINED_NAME}`
- UNUSED_IMPORT_ALIAS: `{UNUSED_IMPORT_ALIAS}`
- UNUSED_VAR_NAME: `{UNUSED_VAR_NAME}`
- HARDCODED_SECRET_VALUE: `{HARDCODED_SECRET_VALUE}`
- EVAL_SENTINEL: `{EVAL_SENTINEL}`
- SHELL_SENTINEL: `{SHELL_SENTINEL}`
- DUP_TASK_ID: `{DUP_TASK_ID}`

## Specific instructions by mutation type
- **syntax_missing_colon**: Remove the `:` from a `def`, `if`, `for`, `with`, `try`, or `class` line
- **syntax_unbalanced_paren**: Add an extra `(` or remove a `)` 
- **syntax_unterminated_string**: Remove a closing quote from a string
- **import_nonexistent_module**: Add line: `import {NONEXISTENT_MODULE_NAME}`
- **import_nonexistent_from**: Add line: `from {NONEXISTENT_MODULE_NAME} import something`
- **undefined_name_reference**: Add a line that uses `{UNDEFINED_NAME}` without defining it (e.g., `print({UNDEFINED_NAME})`)
- **unused_import_alias**: Add line: `import os as {UNUSED_IMPORT_ALIAS}` (do NOT use the alias anywhere)
- **unused_variable_assignment**: Add line: `{UNUSED_VAR_NAME} = 42` (do NOT use this variable anywhere)
- **security_hardcoded_secret**: Add line: `API_KEY = "{HARDCODED_SECRET_VALUE}"`
- **security_eval_call**: Add line inside a function: `result = eval("'{EVAL_SENTINEL}'")`
- **security_subprocess_shell_true**: Add: `subprocess.run("echo {SHELL_SENTINEL}", shell=True)`
- **error_handling_bare_except**: Change an `except SomeError:` to just `except:`
- **error_handling_swallow_exception**: Add: `try: pass\\nexcept Exception: pass`
- **airflow_duplicate_task_id**: Change a task's task_id to `"{DUP_TASK_ID}"` and change another task's task_id to the same value
- **airflow_remove_dag_definition**: Remove the DAG(...) instantiation OR DAG context manager OR @dag decorator usage so no DAG definition remains.
  IMPORTANT: The module MUST still import. Do not leave references like `dag=dag` unless you also define `dag` at module level (e.g., `dag = None`).
  The goal is: imports_resolve should remain True, but instantiation/structure should fail because no DAG is discovered.
- **airflow_remove_tasks**: Delete all task definitions (Operators and @task functions)
- **airflow_remove_dependencies**: Delete all `>>` and `<<` dependency lines
- **prefect_remove_flow_decorator**: Delete or comment out the `@flow` decorator line
- **dagster_remove_job_and_assets**: Delete `@job`, `@graph`, and `@asset` decorators
- **dagster_remove_ops_and_assets**: Delete all `@op` and `@asset` decorators

## Available edit operations
- `replace_line`: Replace line N with new content
- `delete_line`: Delete line N  
- `insert_after`: Insert new content after line N
- `insert_before`: Insert new content before line N
- `replace_lines`: Replace lines N through M (inclusive) with new content
- `delete_lines`: Delete lines N through M (inclusive)
- `append_line`: Append new content at end of file (line_number ignored)
- `prepend_line`: Insert new content at start of file (line_number ignored)

## Output JSON schema
```json
{{
  "status": "ok",
  "edit": {{
    "op": "<operation_name>",
    "line_number": <1-indexed line number>,
    "end_line_number": <optional, for range operations>,
    "new_content": "<the new line(s) to insert, can include \\n for multi-line>"
  }},
  "mutation_notes": "<brief description of what you changed>"
}}
```

If the mutation cannot be applied (e.g., the code structure doesn't support it), return:
```json
{{
  "status": "skip",
  "skip_reason": "<why it cannot be applied>"
}}
```

## Original code (with line numbers)
```python
{numbered_code}
```

## Important rules
1. Make the SMALLEST possible edit
2. Do NOT change anything else
3. Line numbers are 1-indexed
4. For multi-line new_content, use actual newlines or \\n
5. Output ONLY the JSON object, no other text"""


FEEDBACK_TEMPLATE = """## Previous attempt FAILED validation

**Error**: {error}

**Details**: {details}

Please try again with a corrected edit operation.
Remember:
- Line numbers are 1-indexed
- The edit must produce valid Python (unless it's a syntax mutation)
- Use the exact sentinel values provided
- Output only the JSON object"""


# -----------------------------------------------------------------------------
# GPT Mutator (Edit-Based)
# -----------------------------------------------------------------------------

class GPTMutator:
    def __init__(
        self,
        client: AzureChatClient,
        max_completion_tokens: int = 6000,
        temperature: float = 1.0,
    ):
        self.client = client
        self.max_completion_tokens = int(max_completion_tokens)
        self.temperature = float(temperature)

    def generate_mutation(
        self,
        orchestrator: str,
        spec: MutationSpec,
        original_code: str,
        constants: Dict[str, str],
        max_attempts: int = 3,
        max_line_delta: int = 20,
    ) -> Tuple[Optional[str], Optional[int], Dict[str, Any]]:
        """
        Generate a mutation using edit operations with ReAct-style retry loop.
        
        Returns:
            (mutated_code, injected_line, metadata)
            mutated_code is None if generation failed.
        """
        meta: Dict[str, Any] = {
            "deployment": self.client.deployment,
            "api_version": self.client.api_version,
            "attempts": [],
            "total_attempts": 0,
            "status": "failed",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        numbered_code = add_line_numbers(original_code)
        
        base_prompt = MUTATION_PROMPT_TEMPLATE.format(
            mutation_id=spec.id,
            category=spec.category,
            severity=spec.severity,
            orchestrator=orchestrator,
            description=spec.description,
            numbered_code=numbered_code,
            **constants,
        )

        conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": base_prompt},
        ]

        for attempt in range(1, max_attempts + 1):
            meta["total_attempts"] = attempt
            attempt_info: Dict[str, Any] = {"attempt": attempt}

            try:
                # Make LLM call
                text, usage = self.client.chat(
                    messages=conversation_history,
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=self.temperature,
                )
                
                # Accumulate usage
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    meta["usage"][k] = meta["usage"].get(k, 0) + clamp_int(usage.get(k, 0))

                attempt_info["raw_response"] = text[:2000] if text else ""

                # Parse JSON
                json_str = extract_json_object(text)
                parsed = json.loads(json_str)
                attempt_info["parsed_status"] = parsed.get("status")

            except json.JSONDecodeError as e:
                attempt_info["error"] = f"JSON parse error: {e}"
                meta["attempts"].append(attempt_info)
                
                # Add feedback for retry
                conversation_history.append({"role": "assistant", "content": text if text else ""})
                conversation_history.append({
                    "role": "user",
                    "content": f"Your response was not valid JSON. Error: {e}\nPlease output only a valid JSON object."
                })
                continue
                
            except Exception as e:
                attempt_info["error"] = f"LLM call error: {type(e).__name__}: {e}"
                meta["attempts"].append(attempt_info)
                continue

            # Handle skip response
            if parsed.get("status") == "skip":
                meta["status"] = "skip"
                meta["skip_reason"] = parsed.get("skip_reason", "Unknown")
                meta["attempts"].append(attempt_info)
                return None, None, meta

            # Parse edit operation
            edit_dict = parsed.get("edit")
            if not edit_dict or not isinstance(edit_dict, dict):
                attempt_info["error"] = "Missing or invalid 'edit' field in response"
                meta["attempts"].append(attempt_info)
                
                conversation_history.append({"role": "assistant", "content": text})
                conversation_history.append({
                    "role": "user",
                    "content": FEEDBACK_TEMPLATE.format(
                        error="Missing 'edit' field",
                        details="Response must include an 'edit' object with 'op', 'line_number', and optionally 'new_content'."
                    )
                })
                continue

            try:
                edit = EditOperation.from_dict(edit_dict)
            except Exception as e:
                attempt_info["error"] = f"Failed to parse edit operation: {e}"
                meta["attempts"].append(attempt_info)
                
                conversation_history.append({"role": "assistant", "content": text})
                conversation_history.append({
                    "role": "user",
                    "content": FEEDBACK_TEMPLATE.format(
                        error=f"Invalid edit operation: {e}",
                        details="Ensure 'op' is a valid operation and 'line_number' is an integer."
                    )
                })
                continue

            attempt_info["edit_operation"] = edit.to_dict()
            attempt_info["mutation_notes"] = parsed.get("mutation_notes", "")

            # Apply edit
            mutated_code, injected_line, apply_error = apply_edit(original_code, edit)
            
            if apply_error:
                attempt_info["apply_error"] = apply_error
                meta["attempts"].append(attempt_info)
                
                conversation_history.append({"role": "assistant", "content": text})
                conversation_history.append({
                    "role": "user",
                    "content": FEEDBACK_TEMPLATE.format(
                        error=f"Edit application failed: {apply_error}",
                        details=f"The original file has {len(original_code.splitlines())} lines. Ensure line_number is valid."
                    )
                })
                continue

            # Validate mutation
            ok, val_details, val_error = validate_mutation(
                orchestrator=orchestrator,
                spec=spec,
                original_code=original_code,
                mutated_code=mutated_code,
                constants=constants,
                max_line_delta=max_line_delta,
            )

            attempt_info["validation_passed"] = ok
            attempt_info["validation_details"] = val_details

            if not ok:
                attempt_info["validation_error"] = val_error
                meta["attempts"].append(attempt_info)
                
                conversation_history.append({"role": "assistant", "content": text})
                conversation_history.append({
                    "role": "user",
                    "content": FEEDBACK_TEMPLATE.format(
                        error=val_error,
                        details=json.dumps(val_details, indent=2)
                    )
                })
                continue

            # Success!
            meta["status"] = "ok"
            meta["mutation_notes"] = parsed.get("mutation_notes", "")
            meta["final_edit"] = edit.to_dict()
            meta["attempts"].append(attempt_info)
            
            return mutated_code, injected_line, meta

        # All attempts failed
        meta["status"] = "failed"
        meta["error"] = "All attempts failed validation"
        return None, None, meta


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def select_files(files: List[Path], max_files: int, seed: int) -> List[Path]:
    """Randomly select up to max_files from the list."""
    if max_files <= 0:
        return files
    rnd = random.Random(seed)
    files = list(files)
    rnd.shuffle(files)
    return files[:max_files]


def main():
    ap = argparse.ArgumentParser(
        description="GT1 Mutation Injection using Azure OpenAI (Edit-Operation Based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input/Output
    ap.add_argument("--input-dir", required=True, 
                    help="Directory containing airflow/, prefect/, dagster/ subdirs")
    ap.add_argument("--output-dir", required=True,
                    help="Output mutation_benchmark directory")
    ap.add_argument("--catalog", required=True,
                    help="Path to mutation_catalog.json")

    # Scope control
    ap.add_argument("--orchestrators", nargs="+", default=["airflow", "prefect", "dagster"],
                    help="Which orchestrators to process")
    ap.add_argument("--max-files", type=int, default=0,
                    help="Max files per orchestrator (0 = all)")
    ap.add_argument("--mutations-per-file", type=int, default=1,
                    help="Target number of successful mutations per original file")
    ap.add_argument("--max-spec-tries-per-file", type=int, default=8,
                    help="Max mutation specs to try per file before moving on")

    # Azure OpenAI
    ap.add_argument("--azure-endpoint", required=True,
                    help="Azure OpenAI endpoint (e.g., https://openai-key5.openai.azure.com)")
    ap.add_argument("--azure-deployment", default="gpt-5-mini",
                    help="Deployment name")
    ap.add_argument("--api-version", default="2024-12-01-preview",
                    help="API version")
    ap.add_argument("--api-key", default=None,
                    help="API key (or set AZURE_OPENAI_API_KEY env var)")

    # Generation params
    ap.add_argument("--max-completion-tokens", type=int, default=6000,
                    help="Max completion tokens per LLM call")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="LLM temperature")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="Max retry attempts per mutation spec")
    ap.add_argument("--max-line-delta", type=int, default=20,
                    help="Max allowed change in line count")
    ap.add_argument("--rate-limit-delay", type=float, default=0.3,
                    help="Delay between LLM calls (seconds)")
    
    # Other
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for file selection")
    ap.add_argument("--resume", action="store_true",
                    help="Skip already-generated mutations")
    ap.add_argument("--verbose", action="store_true",
                    help="Print progress details")

    args = ap.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load catalog
    specs = load_catalog(Path(args.catalog))
    print(f"Loaded {len(specs)} mutation specs from catalog")

    # Get API key
    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Use --api-key or set AZURE_OPENAI_API_KEY env var.")

    # Initialize client and mutator
    client = AzureChatClient(
        azure_endpoint=args.azure_endpoint,
        deployment=args.azure_deployment,
        api_version=args.api_version,
        api_key=api_key,
        rate_limit_delay=args.rate_limit_delay,
    )
    
    mutator = GPTMutator(
        client=client,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )

    # Manifest paths
    manifest_mutations = out_dir / "manifests" / "mutations.jsonl"
    manifest_originals = out_dir / "manifests" / "originals.jsonl"

    # Stats
    stats = {
        "total_files": 0,
        "total_mutations_attempted": 0,
        "total_mutations_succeeded": 0,
        "total_mutations_failed": 0,
        "total_mutations_skipped": 0,
        "by_orchestrator": {},
    }

    for orch in args.orchestrators:
        orch = orch.strip().lower()
        orch_in = input_dir / orch
        
        if not orch_in.exists():
            print(f"[WARN] Orchestrator directory not found: {orch_in}")
            continue

        # Find Python files
        files = sorted([p for p in orch_in.rglob("*.py") if p.is_file()])
        files = select_files(files, args.max_files, seed=args.seed)

        # Filter to syntactically valid files
        valid_files = []
        for p in files:
            try:
                code = read_text(p)
                if code.strip() and is_syntax_valid(code):
                    valid_files.append(p)
            except Exception:
                continue

        print(f"[{orch}] Found {len(files)} files, {len(valid_files)} syntactically valid")

        # Get applicable mutations for this orchestrator
        applicable = [s for s in specs if orch in s.orchestrators]
        if not applicable:
            print(f"[WARN] No catalog mutations apply to orchestrator={orch}")
            continue

        orch_stats = {
            "files": len(valid_files),
            "mutations_attempted": 0,
            "mutations_succeeded": 0,
            "mutations_failed": 0,
            "mutations_skipped": 0,
        }

        rnd = random.Random(args.seed)

        for file_idx, p in enumerate(valid_files):
            original_code = read_text(p)
            file_id = sha256_12(original_code)
            original_name = p.name
            constants = make_injection_constants(file_id)

            if args.verbose:
                print(f"  [{file_idx+1}/{len(valid_files)}] {file_id} ({original_name[:50]}...)")

            # Save original
            original_out = out_dir / "originals" / orch / f"{file_id}.py"
            if not original_out.exists():
                write_text(original_out, original_code)
                append_jsonl(manifest_originals, {
                    "file_id": file_id,
                    "orchestrator": orch,
                    "source_path": str(p),
                    "original_path": str(original_out),
                    "original_name": original_name,
                    "line_count": len(original_code.splitlines()),
                })

            # Track successes for this file
            successes = 0
            tries = 0

            # Shuffle specs for coverage diversity
            shuffled_specs = list(applicable)
            rnd.shuffle(shuffled_specs)

            for spec in shuffled_specs:
                if successes >= args.mutations_per_file:
                    break
                if tries >= args.max_spec_tries_per_file:
                    break
                tries += 1

                mutated_path = out_dir / "mutated" / orch / file_id / f"{spec.id}.py"
                gt1_path = out_dir / "ground_truth" / "gt1_mutations" / orch / file_id / f"{spec.id}.json"

                # Resume check
                if args.resume and mutated_path.exists() and gt1_path.exists():
                    successes += 1
                    continue

                orch_stats["mutations_attempted"] += 1

                # Precondition check
                skip, reason = precondition_skip(orch, spec, original_code)
                if skip:
                    orch_stats["mutations_skipped"] += 1
                    write_json(gt1_path, {
                        "status": "skip_precondition",
                        "skip_reason": reason,
                        "file_id": file_id,
                        "orchestrator": orch,
                        "source_path": str(p),
                        "original_path": str(original_out),
                        "mutation_id": spec.id,
                        "category": spec.category,
                        "severity": spec.severity,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    continue

                # Generate mutation
                mutated_code, injected_line, gen_meta = mutator.generate_mutation(
                    orchestrator=orch,
                    spec=spec,
                    original_code=original_code,
                    constants=constants,
                    max_attempts=args.max_attempts,
                    max_line_delta=args.max_line_delta,
                )

                if mutated_code is None:
                    # Failed or skipped
                    if gen_meta.get("status") == "skip":
                        orch_stats["mutations_skipped"] += 1
                    else:
                        orch_stats["mutations_failed"] += 1
                    
                    write_json(gt1_path, {
                        "status": gen_meta.get("status", "failed"),
                        "skip_reason": gen_meta.get("skip_reason"),
                        "error": gen_meta.get("error"),
                        "file_id": file_id,
                        "orchestrator": orch,
                        "source_path": str(p),
                        "original_path": str(original_out),
                        "mutation_id": spec.id,
                        "category": spec.category,
                        "severity": spec.severity,
                        "constants": constants,
                        "generator": gen_meta,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    continue

                # Success!
                orch_stats["mutations_succeeded"] += 1
                successes += 1

                # Ensure trailing newline
                if not mutated_code.endswith("\n"):
                    mutated_code += "\n"

                write_text(mutated_path, mutated_code)

                gt1 = {
                    "status": "ok",
                    "file_id": file_id,
                    "orchestrator": orch,
                    "source_path": str(p),
                    "original_path": str(original_out),
                    "mutated_path": str(mutated_path),
                    "mutation": {
                        "id": spec.id,
                        "category": spec.category,
                        "severity": spec.severity,
                        "description": spec.description,
                    },
                    "constants": constants,
                    "injected_line": injected_line,
                    "edit_operation": gen_meta.get("final_edit"),
                    "mutation_notes": gen_meta.get("mutation_notes"),
                    "validation": {
                        "passed": True,
                        "line_count_original": len(original_code.splitlines()),
                        "line_count_mutated": len(mutated_code.splitlines()),
                    },
                    "generator": {
                        "deployment": gen_meta.get("deployment"),
                        "api_version": gen_meta.get("api_version"),
                        "total_attempts": gen_meta.get("total_attempts"),
                        "usage": gen_meta.get("usage"),
                    },
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                write_json(gt1_path, gt1)

                append_jsonl(manifest_mutations, {
                    "file_id": file_id,
                    "orchestrator": orch,
                    "mutation_id": spec.id,
                    "category": spec.category,
                    "severity": spec.severity,
                    "original_path": str(original_out),
                    "mutated_path": str(mutated_path),
                    "gt1_path": str(gt1_path),
                    "injected_line": injected_line,
                    "usage": gen_meta.get("usage", {}),
                })

                if args.verbose:
                    print(f"    ✓ {spec.id} (line {injected_line})")

        stats["by_orchestrator"][orch] = orch_stats
        stats["total_files"] += orch_stats["files"]
        stats["total_mutations_attempted"] += orch_stats["mutations_attempted"]
        stats["total_mutations_succeeded"] += orch_stats["mutations_succeeded"]
        stats["total_mutations_failed"] += orch_stats["mutations_failed"]
        stats["total_mutations_skipped"] += orch_stats["mutations_skipped"]

        print(f"[{orch}] Completed: {orch_stats['mutations_succeeded']} succeeded, "
              f"{orch_stats['mutations_failed']} failed, {orch_stats['mutations_skipped']} skipped")

    # Write summary stats
    stats_path = out_dir / "generation_stats.json"
    write_json(stats_path, stats)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total files processed: {stats['total_files']}")
    print(f"Total mutations attempted: {stats['total_mutations_attempted']}")
    print(f"Total mutations succeeded: {stats['total_mutations_succeeded']}")
    print(f"Total mutations failed: {stats['total_mutations_failed']}")
    print(f"Total mutations skipped: {stats['total_mutations_skipped']}")
    if stats['total_mutations_attempted'] > 0:
        success_rate = stats['total_mutations_succeeded'] / stats['total_mutations_attempted'] * 100
        print(f"Success rate: {success_rate:.1f}%")
    print(f"\nOutputs written to: {out_dir}")
    print(f"Stats written to: {stats_path}")


if __name__ == "__main__":
    main()