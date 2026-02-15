# scripts/evaluators/agents/schemas.py

"""
Pydantic schemas for agentic evaluators.

These provide:
1. Structured LLM outputs (less parsing errors)
2. Validation
3. Type safety
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class SeverityLevel(str, Enum):
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
    ORCHESTRATOR_STRUCTURE = "orchestrator_structure"
    ORCHESTRATOR_CONFIG = "orchestrator_config"


# ============================================================================
# DEBATE ARCHITECTURE SCHEMAS
# ============================================================================

class ProposedIssue(BaseModel):
    """Issue proposed by the Proposer agent."""
    id: str = Field(..., description="Unique issue ID (e.g., 'ISS-001')")
    category: IssueCategory
    severity: SeverityLevel
    message: str = Field(..., description="Clear description of the issue")
    line: Optional[int] = Field(None, description="Line number where issue occurs")
    evidence: str = Field(..., description="Code snippet or specific evidence supporting this issue")
    reasoning: str = Field(..., description="Why this is an issue")

    class Config:
        use_enum_values = True


class ProposerOutput(BaseModel):
    """Output from the Proposer agent."""
    issues: List[ProposedIssue] = Field(default_factory=list)
    gate_predictions: "GatePredictions"
    analysis_summary: str = Field(..., description="Brief summary of code analysis")


class Challenge(BaseModel):
    """A challenge from the Critic agent."""
    issue_id: str = Field(..., description="ID of the issue being challenged")
    challenge_type: Literal["evidence_weak", "false_positive", "wrong_severity", "wrong_category", "valid"]
    challenge_reasoning: str = Field(..., description="Why this issue is being challenged")
    questions: List[str] = Field(default_factory=list, description="Specific questions for the defender")


class CriticOutput(BaseModel):
    """Output from the Critic agent."""
    challenges: List[Challenge] = Field(default_factory=list)
    accepted_issues: List[str] = Field(default_factory=list, description="Issue IDs accepted without challenge")


class Defense(BaseModel):
    """Defense from the Defender agent."""
    issue_id: str
    defense_successful: bool = Field(..., description="Whether the defense holds up")
    defense_reasoning: str
    updated_evidence: Optional[str] = Field(None, description="Additional evidence if needed")
    updated_severity: Optional[SeverityLevel] = None
    withdraw: bool = Field(False, description="Whether to withdraw this issue")


class DefenderOutput(BaseModel):
    """Output from the Defender agent."""
    defenses: List[Defense] = Field(default_factory=list)


class AdjudicatorVerdict(BaseModel):
    """Verdict on a single issue."""
    issue_id: str
    final_verdict: Literal["accepted", "rejected", "modified"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    final_category: Optional[IssueCategory] = None
    final_severity: Optional[SeverityLevel] = None
    final_message: Optional[str] = None
    reasoning: str


class AdjudicatorOutput(BaseModel):
    """Output from the Adjudicator agent."""
    verdicts: List[AdjudicatorVerdict] = Field(default_factory=list)
    final_gate_predictions: "GatePredictions"


# ============================================================================
# VERIFIER ARCHITECTURE SCHEMAS
# ============================================================================

class VerifiableClaim(BaseModel):
    """A claim that can be verified by tools."""
    claim_type: Literal[
        "line_exists",           # The cited line exists in the code
        "variable_undefined",    # Variable X is not defined before use
        "variable_unused",       # Variable X is defined but never used
        "import_unresolvable",   # Import X cannot be resolved
        "function_undefined",    # Function X is called but not defined
        "syntax_error",          # There is a syntax error at line X
        "security_pattern",      # Code matches security anti-pattern
        "complexity_threshold",  # Function exceeds complexity threshold
    ]
    target: str = Field(..., description="The specific target (variable name, line content, import path, etc.)")
    line: Optional[int] = None
    context: Optional[str] = Field(None, description="Additional context for verification")


class VerifiableIssue(BaseModel):
    """Issue with verifiable claims."""
    id: str
    category: IssueCategory
    severity: SeverityLevel
    message: str
    line: Optional[int] = None
    evidence: str
    claims: List[VerifiableClaim] = Field(..., description="Specific claims that can be verified")

    class Config:
        use_enum_values = True


class VerifierProposerOutput(BaseModel):
    """Output from the Verifier's proposer."""
    issues: List[VerifiableIssue] = Field(default_factory=list)
    gate_predictions: "GatePredictions"


class VerificationResult(BaseModel):
    """Result of verifying a single claim."""
    claim_type: str
    target: str
    verified: bool
    verification_method: str = Field(..., description="How the claim was verified")
    details: Optional[str] = None


class IssueVerificationResult(BaseModel):
    """Verification results for an issue."""
    issue_id: str
    all_claims_verified: bool
    claim_results: List[VerificationResult]
    verification_score: float = Field(..., ge=0.0, le=1.0, description="Proportion of claims verified")


# ============================================================================
# SHARED SCHEMAS
# ============================================================================

class GatePredictions(BaseModel):
    """Gate predictions for PCT."""
    syntax_valid: bool = True
    imports_resolve: bool = True
    instantiates: bool = True
    has_structure: bool = True


# Update forward references
ProposerOutput.model_rebuild()
AdjudicatorOutput.model_rebuild()
VerifierProposerOutput.model_rebuild()