#!/usr/bin/env python3
"""
SAT Ground Truth via Tool Ensemble.

Runs multiple independent static tools and unions their detected issues:
- pylint
- flake8
- bandit
- radon

The union is used as ground truth for issue-detection benchmarking.

Key properties:
- Tool invocation uses `sys.executable -m <tool>` so it runs inside the current
  interpreter environment. This removes reliance on PATH / venv activation.
- Category mapping improved (e.g., trailing-whitespace -> style).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class IssueCategory(str, Enum):
    SYNTAX = "syntax"
    IMPORT = "import"
    TYPE_ERROR = "type_error"
    SECURITY = "security"
    COMPLEXITY = "complexity"
    STYLE = "style"
    NAMING = "naming"
    DOCUMENTATION = "documentation"
    ERROR_HANDLING = "error_handling"
    BEST_PRACTICE = "best_practice"
    UNUSED = "unused"
    UNDEFINED = "undefined"


@dataclass
class GroundTruthIssue:
    category: str
    severity: str
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    tool: str = ""
    code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        # Keep tool/code in fingerprint to reduce over-deduping across tools
        return hashlib.md5(
            f"{self.category}:{self.line}:{self.code}:{self.message[:80]}".encode("utf-8", errors="ignore")
        ).hexdigest()[:12]


@dataclass
class SATGroundTruth:
    file_path: str
    issues: List[GroundTruthIssue] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)
    counts_by_category: Dict[str, int] = field(default_factory=dict)
    counts_by_severity: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["issues"] = [i.to_dict() for i in self.issues]
        return d


class ToolEnsembleOracle:
    """
    Build GT2 issues by running tools inside the current interpreter environment.
    """

    PYLINT_SEVERITY_MAP = {
        "fatal": IssueSeverity.CRITICAL,
        "error": IssueSeverity.CRITICAL,
        "warning": IssueSeverity.MAJOR,
        "refactor": IssueSeverity.MINOR,
        "convention": IssueSeverity.MINOR,
        "information": IssueSeverity.INFO,
    }

    # Improved mapping: trailing whitespace and wrong import order -> STYLE
    PYLINT_CATEGORY_MAP = {
        # imports
        "import-error": IssueCategory.IMPORT,
        "no-name-in-module": IssueCategory.IMPORT,

        # syntax / undefined / unused
        "syntax-error": IssueCategory.SYNTAX,
        "undefined-variable": IssueCategory.UNDEFINED,
        "unused-import": IssueCategory.UNUSED,
        "unused-variable": IssueCategory.UNUSED,
        "unused-argument": IssueCategory.UNUSED,

        # docs / naming
        "missing-docstring": IssueCategory.DOCUMENTATION,
        "missing-module-docstring": IssueCategory.DOCUMENTATION,
        "missing-function-docstring": IssueCategory.DOCUMENTATION,
        "missing-class-docstring": IssueCategory.DOCUMENTATION,
        "invalid-name": IssueCategory.NAMING,

        # style
        "line-too-long": IssueCategory.STYLE,
        "trailing-whitespace": IssueCategory.STYLE,
        "wrong-import-order": IssueCategory.STYLE,
        "bad-indentation": IssueCategory.STYLE,

        # error handling
        "bare-except": IssueCategory.ERROR_HANDLING,
        "broad-except": IssueCategory.ERROR_HANDLING,

        # complexity
        "too-many-branches": IssueCategory.COMPLEXITY,
        "too-many-statements": IssueCategory.COMPLEXITY,

        # misc best practices
        "fixme": IssueCategory.BEST_PRACTICE,
        "pointless-statement": IssueCategory.BEST_PRACTICE,
        "missing-timeout": IssueCategory.BEST_PRACTICE,
        "unspecified-encoding": IssueCategory.BEST_PRACTICE,
        "raise-missing-from": IssueCategory.BEST_PRACTICE,
        "broad-exception-raised": IssueCategory.BEST_PRACTICE,
        "no-else-return": IssueCategory.BEST_PRACTICE,
        "no-value-for-parameter": IssueCategory.BEST_PRACTICE,
    }

    BANDIT_SEVERITY_MAP = {
        "HIGH": IssueSeverity.CRITICAL,
        "MEDIUM": IssueSeverity.MAJOR,
        "LOW": IssueSeverity.MINOR,
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def establish_ground_truth(self, file_path: Path) -> SATGroundTruth:
        file_path = Path(file_path)
        gt = SATGroundTruth(file_path=str(file_path))

        gt.tool_results["pylint"] = self._run_pylint(file_path, gt)
        gt.tool_results["flake8"] = self._run_flake8(file_path, gt)
        gt.tool_results["bandit"] = self._run_bandit(file_path, gt)
        gt.tool_results["radon"] = self._run_radon(file_path, gt)

        gt.issues = self._deduplicate_issues(gt.issues)

        # Counts
        for issue in gt.issues:
            gt.counts_by_category[issue.category] = gt.counts_by_category.get(issue.category, 0) + 1
            gt.counts_by_severity[issue.severity] = gt.counts_by_severity.get(issue.severity, 0) + 1

        return gt

    # ---------------------------------------------------------------------

    def _module_available(self, module_name: str) -> bool:
        try:
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False

    def _run_pylint(self, file_path: Path, gt: SATGroundTruth) -> Dict[str, Any]:
        info: Dict[str, Any] = {"available": self._module_available("pylint"), "raw_count": 0}
        if not info["available"]:
            info["error"] = "pylint not installed in this environment"
            return info

        try:
            # IMPORTANT: use sys.executable -m pylint (no PATH dependency)
            # Optional: disable import errors to make GT2 less environment-dependent.
            cmd = [
                sys.executable, "-m", "pylint",
                str(file_path),
                "--output-format=json",
                "--reports=n",
                "--score=n",
                "--max-line-length=120",
                # If you want GT2 to be stable without orchestrator libs installed in tools env:
                "--disable=import-error,no-name-in-module",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            raw = (res.stdout or "").strip()
            if not raw:
                return info

            msgs = json.loads(raw)
            info["raw_count"] = len(msgs)

            for msg in msgs:
                symbol = msg.get("symbol", "")
                msg_type = (msg.get("type") or "convention").lower()

                severity = self.PYLINT_SEVERITY_MAP.get(msg_type, IssueSeverity.MINOR)
                category = self.PYLINT_CATEGORY_MAP.get(symbol, IssueCategory.BEST_PRACTICE)

                gt.issues.append(GroundTruthIssue(
                    category=category.value,
                    severity=severity.value,
                    message=msg.get("message", ""),
                    line=msg.get("line"),
                    column=msg.get("column"),
                    tool="pylint",
                    code=symbol,
                ))

        except subprocess.TimeoutExpired:
            info["error"] = "timeout"
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"

        return info

    def _run_flake8(self, file_path: Path, gt: SATGroundTruth) -> Dict[str, Any]:
        info: Dict[str, Any] = {"available": self._module_available("flake8"), "raw_count": 0}
        if not info["available"]:
            info["error"] = "flake8 not installed in this environment"
            return info

        try:
            cmd = [sys.executable, "-m", "flake8", str(file_path), "--max-line-length=120"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            out = (res.stdout or "").strip()
            if not out:
                return info

            lines = [l for l in out.splitlines() if l.strip()]
            info["raw_count"] = len(lines)

            for line in lines:
                # format: path:row:col: CODE message
                parts = line.split(":", 3)
                if len(parts) < 4:
                    continue

                try:
                    row = int(parts[1])
                except Exception:
                    row = None
                try:
                    col = int(parts[2])
                except Exception:
                    col = None

                rest = parts[3].strip()
                code = rest.split()[0] if rest else ""
                message = rest

                # Severity heuristic
                # - E999 / E9xx: syntax errors -> critical
                # - F821 undefined name -> critical (runtime error)
                # - Other F*: major (but not necessarily "critical")
                if code.startswith("E9") or code == "E999":
                    severity = IssueSeverity.CRITICAL
                elif code == "F821":
                    severity = IssueSeverity.CRITICAL
                elif code.startswith("F"):
                    severity = IssueSeverity.MAJOR
                elif code.startswith("E"):
                    severity = IssueSeverity.MAJOR
                elif code.startswith("W"):
                    severity = IssueSeverity.MINOR
                else:
                    severity = IssueSeverity.MINOR

                # Category heuristic
                if code.startswith("F401"):
                    category = IssueCategory.UNUSED
                elif code.startswith("F821"):
                    category = IssueCategory.UNDEFINED
                else:
                    category = IssueCategory.STYLE

                gt.issues.append(GroundTruthIssue(
                    category=category.value,
                    severity=severity.value,
                    message=message,
                    line=row,
                    column=col,
                    tool="flake8",
                    code=code,
                ))

        except subprocess.TimeoutExpired:
            info["error"] = "timeout"
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"

        return info

    def _run_bandit(self, file_path: Path, gt: SATGroundTruth) -> Dict[str, Any]:
        info: Dict[str, Any] = {"available": self._module_available("bandit"), "raw_count": 0}
        if not info["available"]:
            info["error"] = "bandit not installed in this environment"
            return info

        try:
            cmd = [sys.executable, "-m", "bandit", "-f", "json", "-q", str(file_path)]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            raw = (res.stdout or "").strip()
            if not raw:
                return info

            data = json.loads(raw)
            results = data.get("results", []) or []
            info["raw_count"] = len(results)

            for r in results:
                bandit_sev = (r.get("issue_severity") or "LOW").upper()
                severity = self.BANDIT_SEVERITY_MAP.get(bandit_sev, IssueSeverity.MINOR)

                gt.issues.append(GroundTruthIssue(
                    category=IssueCategory.SECURITY.value,
                    severity=severity.value,
                    message=r.get("issue_text", ""),
                    line=r.get("line_number"),
                    tool="bandit",
                    code=r.get("test_id", ""),
                ))

        except subprocess.TimeoutExpired:
            info["error"] = "timeout"
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"

        return info

    def _run_radon(self, file_path: Path, gt: SATGroundTruth) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "available": self._module_available("radon"),
            "avg_complexity": None,
            "max_complexity": None
        }
        if not info["available"]:
            info["error"] = "radon not installed in this environment"
            return info

        try:
            cmd = [sys.executable, "-m", "radon", "cc", str(file_path), "-j"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            raw = (res.stdout or "").strip()
            if not raw:
                return info

            data = json.loads(raw)
            all_cc = []

            for _p, items in (data or {}).items():
                for it in items or []:
                    cc = int(it.get("complexity", 0) or 0)
                    all_cc.append(cc)

                    if cc > 10:
                        severity = IssueSeverity.MAJOR if cc > 15 else IssueSeverity.MINOR
                        gt.issues.append(GroundTruthIssue(
                            category=IssueCategory.COMPLEXITY.value,
                            severity=severity.value,
                            message=f"High cyclomatic complexity: {cc} in {it.get('name', '?')}",
                            line=it.get("lineno"),
                            tool="radon",
                            code=f"CC{cc}",
                        ))

            if all_cc:
                info["avg_complexity"] = sum(all_cc) / len(all_cc)
                info["max_complexity"] = max(all_cc)

        except subprocess.TimeoutExpired:
            info["error"] = "timeout"
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"

        return info

    # ---------------------------------------------------------------------

    def _deduplicate_issues(self, issues: List[GroundTruthIssue]) -> List[GroundTruthIssue]:
        seen: Set[str] = set()
        out: List[GroundTruthIssue] = []
        for i in issues:
            fp = i.fingerprint
            if fp in seen:
                continue
            seen.add(fp)
            out.append(i)
        return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Establish SAT ground truth via tool ensemble")
    parser.add_argument("file", help="Path to Python file")
    parser.add_argument("--out", default=None, help="Write JSON to this path")
    args = parser.parse_args()

    oracle = ToolEnsembleOracle()
    gt = oracle.establish_ground_truth(Path(args.file))
    payload = gt.to_dict()

    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
