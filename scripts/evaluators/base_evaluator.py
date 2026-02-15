"""
Core evaluator primitives used by both SAT and PCT.

Design goals:
- Issues are diagnostic only (Severity.CRITICAL/MAJOR/MINOR/INFO)
- Gates are the only mechanism that can define pass/fail in PCT
- No penalties are applied here (post-hoc penalties can be computed later)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import re


class Orchestrator(str, Enum):
    AIRFLOW = "airflow"
    PREFECT = "prefect"
    DAGSTER = "dagster"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


@dataclass
class Issue:
    severity: Severity
    category: str
    message: str
    line: Optional[int] = None
    tool: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "line": self.line,
            "tool": self.tool,
            "details": self.details,
        }


@dataclass
class GateCheckResult:
    name: str
    passed: bool
    message: str
    is_critical: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "message": self.message,
            "is_critical": bool(self.is_critical),
            "details": self.details,
        }


@dataclass
class EvaluationScore:
    name: str
    raw_score: float
    weight: float = 1.0
    issues: List[Issue] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    # penalty-free invariant: evaluators should leave this as 0.0
    penalties_applied: float = 0.0

    # optional error text (mostly for tool failures)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "raw_score": float(self.raw_score),
            "weight": float(self.weight),
            "penalties_applied": float(self.penalties_applied),
            "error": self.error,
            "details": self.details,
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class EvaluationResult:
    evaluation_type: str
    file_path: str
    orchestrator: Orchestrator
    timestamp: str
    scores: Dict[str, EvaluationScore] = field(default_factory=dict)
    gate_checks: List[GateCheckResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def gates_passed(self) -> bool:
        """
        Gate passing means: all critical gate checks passed.
        If there are no gate checks, treat as passed (SAT generally doesn't gate).
        """
        critical = [g for g in self.gate_checks if g.is_critical]
        return all(g.passed for g in critical) if critical else True

    @property
    def all_issues(self) -> List[Issue]:
        issues: List[Issue] = []
        for s in self.scores.values():
            issues.extend(s.issues or [])
        return issues

    @property
    def critical_issues(self) -> List[Issue]:
        return [i for i in self.all_issues if i.severity == Severity.CRITICAL]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_type": self.evaluation_type,
            "file_path": self.file_path,
            "orchestrator": self.orchestrator.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "gates_passed": self.gates_passed,
            "gate_checks": [g.to_dict() for g in self.gate_checks],
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
        }


class BaseEvaluator:
    """
    Base evaluator with:
    - config storage
    - logger
    - orchestrator detection helper
    """
    EVALUATION_TYPE = "base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def detect_orchestrator(self, code: str) -> Orchestrator:
        c = (code or "").lower()
        if "airflow" in c and ("dag(" in c or "from airflow" in c or "@dag" in c):
            return Orchestrator.AIRFLOW
        if "prefect" in c and ("@flow" in code or "from prefect" in c):
            return Orchestrator.PREFECT
        if "dagster" in c and ("@job" in code or "@op" in code or "from dagster" in c):
            return Orchestrator.DAGSTER
        return Orchestrator.UNKNOWN