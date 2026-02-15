"""
Common output schema for all evaluators (deterministic + LLM-based).

This schema is used by BenchmarkRunner for apples-to-apples comparison:
- issue detection metrics (precision/recall/F1) vs SAT tool-ensemble ground truth
- gate prediction accuracy vs PCT runtime oracle ground truth
- cost/time reporting (tokens, ms, optional $)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class EvalSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class EvalCategory(str, Enum):
    """
    Fixed taxonomy.
    All evaluators should map their findings into these categories.
    """
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
    ORCHESTRATOR_STRUCTURE = "orchestrator_structure"
    ORCHESTRATOR_CONFIG = "orchestrator_config"


@dataclass
class DetectedIssue:
    """A single issue detected by an evaluator."""
    category: str  # should be one of EvalCategory values
    severity: str  # should be one of EvalSeverity values
    message: str
    line: Optional[int] = None
    evidence: Optional[str] = None  # optional snippet
    confidence: float = 1.0         # LLM-only (deterministic default 1.0)
    source: Optional[str] = None    # e.g., "pylint", "bandit", "llm_agent_security"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GatePrediction:
    """
    PCT-style gate predictions.
    These are compared against PCTRuntimeOracle ground truth.
    """
    syntax_valid: bool = True
    imports_resolve: bool = True
    instantiates: bool = True
    has_structure: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluatorOutput:
    """
    Standard output emitted by each evaluator.
    BenchmarkRunner consumes this.
    """
    evaluator_name: str
    file_path: str

    issues: List[DetectedIssue] = field(default_factory=list)
    gate_predictions: Optional[GatePrediction] = None

    execution_time_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0

    # Useful for debugging / traceability
    raw_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["issues"] = [i.to_dict() for i in self.issues]
        d["gate_predictions"] = self.gate_predictions.to_dict() if self.gate_predictions else None
        return d

    @property
    def issue_counts(self) -> Dict[str, int]:
        counts = {"total": len(self.issues)}
        for sev in EvalSeverity:
            counts[sev.value] = sum(1 for i in self.issues if i.severity == sev.value)
        return counts