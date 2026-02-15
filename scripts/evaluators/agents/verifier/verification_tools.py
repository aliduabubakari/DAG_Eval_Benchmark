# scripts/evaluators/agents/verifier/verification_tools.py

"""
Verification tools for the Verifier architecture.

These tools deterministically verify claims made by the LLM.
"""

from __future__ import annotations

import ast
import re
import importlib.util
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying a single claim."""
    verified: bool
    method: str
    details: Optional[str] = None


class ASTAnalyzer:
    """AST-based code analysis for verification."""
    
    def __init__(self, code: str):
        self.code = code
        self.lines = code.split('\n')
        self.tree: Optional[ast.AST] = None
        self.parse_error: Optional[str] = None
        
        try:
            self.tree = ast.parse(code)
        except SyntaxError as e:
            self.parse_error = f"Line {e.lineno}: {e.msg}"
    
    def get_defined_names(self) -> Set[str]:
        """Get all defined variable/function/class names."""
        if not self.tree:
            return set()
        
        defined = set()
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                defined.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
        
        return defined
    
    def get_used_names(self) -> Set[str]:
        """Get all used (loaded) names."""
        if not self.tree:
            return set()
        
        used = set()
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.add(node.id)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    used.add(node.func.id)
        
        return used
    
    def get_imports(self) -> List[Dict[str, Any]]:
        """Get all imports with details."""
        if not self.tree:
            return []
        
        imports = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imports.append({
                        "type": "from",
                        "module": node.module or "",
                        "name": alias.name,
                        "alias": alias.asname,
                        "line": node.lineno,
                    })
        
        return imports
    
    def get_undefined_names(self) -> Set[str]:
        """Get names used but not defined (potential undefined variables)."""
        if not self.tree:
            return set()
        
        defined = self.get_defined_names()
        used = self.get_used_names()
        
        # Add builtins
        builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
        
        # Common globals
        common = {"True", "False", "None", "__name__", "__file__", "__doc__"}
        
        return used - defined - builtins - common
    
    def get_unused_imports(self) -> List[Dict[str, Any]]:
        """Get imports that are never used."""
        if not self.tree:
            return []
        
        imports = self.get_imports()
        used = self.get_used_names()
        
        unused = []
        for imp in imports:
            name = imp.get("alias") or imp.get("name") or imp.get("module", "").split('.')[0]
            if name and name not in used:
                unused.append(imp)
        
        return unused
    
    def get_unused_variables(self) -> List[Tuple[str, int]]:
        """Get variables defined but never used."""
        if not self.tree:
            return []
        
        # Track definitions with line numbers
        definitions: Dict[str, int] = {}
        used: Set[str] = set()
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    if node.id not in definitions:
                        definitions[node.id] = node.lineno
                elif isinstance(node.ctx, ast.Load):
                    used.add(node.id)
        
        unused = []
        for name, line in definitions.items():
            if name not in used and not name.startswith('_'):
                unused.append((name, line))
        
        return unused


class VerificationEngine:
    """
    Engine for verifying LLM claims about code.
    
    Supports verification of:
    - Line existence
    - Undefined variables
    - Unused variables/imports
    - Import resolution
    - Syntax errors
    - Complexity thresholds
    - Security patterns
    """
    
    # Known security anti-patterns (regex)
    SECURITY_PATTERNS = {
        "hardcoded_password": [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'passwd\s*=\s*["\'][^"\']+["\']',
            r'pwd\s*=\s*["\'][^"\']+["\']',
        ],
        "hardcoded_secret": [
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'apikey\s*=\s*["\'][^"\']+["\']',
        ],
        "eval_usage": [
            r'\beval\s*\(',
            r'\bexec\s*\(',
        ],
        "sql_injection": [
            r'execute\s*\(\s*["\'].*%s',
            r'execute\s*\(\s*f["\']',
            r'\.format\s*\(.*\)\s*\)',
        ],
    }
    
    def __init__(self, code: str):
        self.code = code
        self.lines = code.split('\n')
        self.analyzer = ASTAnalyzer(code)
    
    def verify_claim(self, claim_type: str, target: str, line: Optional[int] = None, 
                     context: Optional[str] = None) -> VerificationResult:
        """
        Verify a single claim.
        
        Args:
            claim_type: Type of claim to verify
            target: The specific target (variable name, line content, etc.)
            line: Line number if applicable
            context: Additional context
        
        Returns:
            VerificationResult with verified status and details
        """
        verifiers = {
            "line_exists": self._verify_line_exists,
            "variable_undefined": self._verify_variable_undefined,
            "variable_unused": self._verify_variable_unused,
            "import_unresolvable": self._verify_import_unresolvable,
            "function_undefined": self._verify_function_undefined,
            "syntax_error": self._verify_syntax_error,
            "security_pattern": self._verify_security_pattern,
            "complexity_threshold": self._verify_complexity_threshold,
            "import_unused": self._verify_import_unused,
        }
        
        verifier = verifiers.get(claim_type)
        if not verifier:
            return VerificationResult(
                verified=False,
                method="unknown_claim_type",
                details=f"Unknown claim type: {claim_type}",
            )
        
        return verifier(target, line, context)
    
    def _verify_line_exists(self, target: str, line: Optional[int], 
                            context: Optional[str]) -> VerificationResult:
        """Verify that a specific line exists in the code."""
        if line is None:
            # Search for target content anywhere
            for i, l in enumerate(self.lines, 1):
                if target.strip() in l:
                    return VerificationResult(
                        verified=True,
                        method="content_search",
                        details=f"Found at line {i}",
                    )
            return VerificationResult(
                verified=False,
                method="content_search",
                details="Content not found in code",
            )
        
        # Check specific line
        if 1 <= line <= len(self.lines):
            actual_line = self.lines[line - 1]
            # Fuzzy match: target should be substring or similar
            if target.strip() in actual_line or actual_line.strip() in target:
                return VerificationResult(
                    verified=True,
                    method="line_check",
                    details=f"Line {line} matches",
                )
            # Check nearby lines (±2)
            for offset in [-2, -1, 1, 2]:
                nearby = line + offset
                if 1 <= nearby <= len(self.lines):
                    if target.strip() in self.lines[nearby - 1]:
                        return VerificationResult(
                            verified=True,
                            method="nearby_line_check",
                            details=f"Found at line {nearby} (claimed {line})",
                        )
        
        return VerificationResult(
            verified=False,
            method="line_check",
            details=f"Line {line} does not contain expected content",
        )
    
    def _verify_variable_undefined(self, target: str, line: Optional[int],
                                   context: Optional[str]) -> VerificationResult:
        """Verify that a variable is used but not defined."""
        undefined = self.analyzer.get_undefined_names()
        
        if target in undefined:
            return VerificationResult(
                verified=True,
                method="ast_analysis",
                details=f"'{target}' is used but not defined",
            )
        
        return VerificationResult(
            verified=False,
            method="ast_analysis",
            details=f"'{target}' is either defined or not used",
        )
    
    def _verify_variable_unused(self, target: str, line: Optional[int],
                                context: Optional[str]) -> VerificationResult:
        """Verify that a variable is defined but never used."""
        unused = self.analyzer.get_unused_variables()
        unused_names = {name for name, _ in unused}
        
        if target in unused_names:
            return VerificationResult(
                verified=True,
                method="ast_analysis",
                details=f"'{target}' is defined but never used",
            )
        
        return VerificationResult(
            verified=False,
            method="ast_analysis",
            details=f"'{target}' is either used or not defined",
        )
    
    def _verify_import_unresolvable(self, target: str, line: Optional[int],
                                    context: Optional[str]) -> VerificationResult:
        """Verify that an import cannot be resolved."""
        # Extract module name
        module = target.split('.')[0] if '.' in target else target
        
        # Skip orchestrator imports (assume available)
        skip_modules = {'airflow', 'prefect', 'dagster', 'pendulum', 'datetime'}
        if module in skip_modules:
            return VerificationResult(
                verified=False,
                method="skip_list",
                details=f"'{module}' is assumed available",
            )
        
        # Try to find the module
        try:
            spec = importlib.util.find_spec(module)
            if spec is None:
                return VerificationResult(
                    verified=True,
                    method="importlib.find_spec",
                    details=f"Module '{module}' not found",
                )
            return VerificationResult(
                verified=False,
                method="importlib.find_spec",
                details=f"Module '{module}' is available",
            )
        except (ModuleNotFoundError, ImportError):
            return VerificationResult(
                verified=True,
                method="importlib.find_spec",
                details=f"Module '{module}' cannot be imported",
            )
        except Exception as e:
            return VerificationResult(
                verified=False,
                method="importlib.find_spec",
                details=f"Error checking module: {e}",
            )
    
    def _verify_function_undefined(self, target: str, line: Optional[int],
                                   context: Optional[str]) -> VerificationResult:
        """Verify that a function is called but not defined."""
        # Similar to variable undefined
        return self._verify_variable_undefined(target, line, context)
    
    def _verify_syntax_error(self, target: str, line: Optional[int],
                             context: Optional[str]) -> VerificationResult:
        """Verify that there is a syntax error."""
        if self.analyzer.parse_error:
            if line and str(line) in self.analyzer.parse_error:
                return VerificationResult(
                    verified=True,
                    method="ast_parse",
                    details=self.analyzer.parse_error,
                )
            # Syntax error exists but maybe different line
            return VerificationResult(
                verified=True,
                method="ast_parse",
                details=f"Syntax error found: {self.analyzer.parse_error}",
            )
        
        return VerificationResult(
            verified=False,
            method="ast_parse",
            details="No syntax errors found",
        )
    
    def _verify_security_pattern(self, target: str, line: Optional[int],
                                 context: Optional[str]) -> VerificationResult:
        """Verify that a security anti-pattern exists."""
        patterns = self.SECURITY_PATTERNS.get(target.lower(), [])
        
        if not patterns:
            # Generic search for the target as pattern
            patterns = [re.escape(target)]
        
        for pattern in patterns:
            try:
                matches = list(re.finditer(pattern, self.code, re.IGNORECASE))
                if matches:
                    match_lines = []
                    for m in matches:
                        line_no = self.code[:m.start()].count('\n') + 1
                        match_lines.append(line_no)
                    return VerificationResult(
                        verified=True,
                        method="regex_pattern",
                        details=f"Pattern found at lines: {match_lines}",
                    )
            except re.error:
                continue
        
        return VerificationResult(
            verified=False,
            method="regex_pattern",
            details=f"Security pattern '{target}' not found",
        )
    
    def _verify_complexity_threshold(self, target: str, line: Optional[int],
                                     context: Optional[str]) -> VerificationResult:
        """Verify complexity threshold is exceeded."""
        if not self.analyzer.tree:
            return VerificationResult(
                verified=False,
                method="ast_analysis",
                details="Cannot analyze (syntax error)",
            )
        
        # Simple complexity: count branches
        for node in ast.walk(self.analyzer.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == target or target == "*":
                    branches = sum(1 for n in ast.walk(node) 
                                   if isinstance(n, (ast.If, ast.For, ast.While, 
                                                     ast.Try, ast.With, ast.Match)))
                    if branches > 10:  # Threshold
                        return VerificationResult(
                            verified=True,
                            method="branch_count",
                            details=f"Function '{node.name}' has {branches} branches",
                        )
        
        return VerificationResult(
            verified=False,
            method="branch_count",
            details="No high-complexity functions found",
        )
    
    def _verify_import_unused(self, target: str, line: Optional[int],
                              context: Optional[str]) -> VerificationResult:
        """Verify that an import is unused."""
        unused = self.analyzer.get_unused_imports()
        
        for imp in unused:
            name = imp.get("alias") or imp.get("name") or imp.get("module", "").split('.')[0]
            if name == target:
                return VerificationResult(
                    verified=True,
                    method="ast_analysis",
                    details=f"Import '{target}' is unused (line {imp.get('line')})",
                )
        
        return VerificationResult(
            verified=False,
            method="ast_analysis",
            details=f"Import '{target}' is used or not present",
        )