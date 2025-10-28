"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class CompanyMapping(BaseModel):
    """Represents a mapping from original name to canonical name."""
    original_name: str = Field(..., description="Original company name from input")
    canonical_name: str = Field(..., description="Standardized canonical name")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    group_id: int = Field(..., description="ID of the group this company belongs to")
    alternatives: List[str] = Field(default_factory=list, description="Other names in the same group")
    llm_reviewed: bool = Field(default=False, description="Whether this match was LLM-assessed")
    llm_decision: Optional[str] = Field(None, description="LLM decision: 'same'|'different'|'unknown'")


class AuditLogEntry(BaseModel):
    """Single entry in the audit log."""
    timestamp: str = Field(..., description="ISO timestamp of the operation")
    original_name: str = Field(..., description="Original company name")
    canonical_name: str = Field(..., description="Assigned canonical name")
    confidence_score: float = Field(..., description="Matching confidence")
    group_id: int = Field(..., description="Group ID")
    reasoning: str = Field(..., description="Explanation of why names were matched")


class AuditLog(BaseModel):
    """Complete audit log for a processing session."""
    filename: str = Field(..., description="Name of processed file")
    processed_at: str = Field(..., description="ISO timestamp of processing")
    total_names: int = Field(..., description="Total number of unique names processed")
    total_groups: int = Field(..., description="Number of groups created")
    entries: List[AuditLogEntry] = Field(..., description="Detailed log entries")


class ThresholdInfo(BaseModel):
    """Information about the thresholding method used."""
    method: str = Field(..., description="Threshold method: 'fixed' or 'adaptive_gmm'")
    t_low: Optional[float] = Field(None, description="Lower threshold (reject below this)")
    s_90: Optional[float] = Field(None, description="90% confidence threshold (promotion eligibility)")
    t_high: Optional[float] = Field(None, description="High threshold (auto-accept above this)")
    fixed_threshold: Optional[float] = Field(None, description="Fixed threshold value if method='fixed'")
    fallback_reason: Optional[str] = Field(None, description="Reason for fallback to fixed threshold")


class GMMMetadata(BaseModel):
    """Metadata from Gaussian Mixture Model fitting."""
    cluster_means: List[float] = Field(..., description="Mean of each cluster")
    cluster_variances: List[float] = Field(..., description="Variance of each cluster")
    cluster_weights: List[float] = Field(..., description="Weight (proportion) of each cluster")
    total_pairs_analyzed: int = Field(..., description="Number of pairs used for GMM fitting")
    high_cluster_index: Optional[int] = Field(None, description="Index of the 'same company' cluster")


class GuardrailStats(BaseModel):
    """Statistics about LLM guardrails."""
    total_assessments: int = Field(..., description="Total LLM assessments made")
    guardrails_triggered: int = Field(..., description="Number of times guardrails triggered")
    unknown_responses: int = Field(..., description="Number of 'unknown' responses")
    same_decisions: int = Field(..., description="Number of 'same' decisions")
    different_decisions: int = Field(..., description="Number of 'different' decisions")
    avg_confidence: float = Field(..., description="Average LLM confidence across all assessments")
    low_confidence_converted: int = Field(default=0, description="Decisions downgraded due to low confidence")


class LLMBorderlineMetadata(BaseModel):
    """Metadata from LLM borderline assessment with guardrails."""
    enabled: bool = Field(..., description="Whether LLM assessment was enabled")
    total_borderline_pairs: int = Field(..., description="Total pairs in borderline range")
    llm_assessments_made: int = Field(..., description="Number of LLM API calls made")
    cache_hits: int = Field(..., description="Number of cached responses reused")
    adjustments_applied: int = Field(..., description="Number of similarity scores adjusted")
    distance_range: tuple = Field(..., description="Distance range for borderline (low, high)")
    llm_provider: str = Field(..., description="LLM provider (e.g., 'openai')")
    llm_model: str = Field(..., description="LLM model used (e.g., 'gpt-4o-mini')")
    adjustment_strength: float = Field(..., description="Adjustment strength parameter")
    min_confidence_threshold: float = Field(..., description="Minimum confidence threshold")
    guardrail_stats: GuardrailStats = Field(..., description="Guardrail statistics")
    api_cost_estimate: float = Field(..., description="Estimated API cost in USD")


class ProcessingResult(BaseModel):
    """Result of processing a CSV file."""
    mappings: List[CompanyMapping] = Field(..., description="Company name mappings")
    audit_log: AuditLog = Field(..., description="Detailed audit log")
    summary: dict = Field(..., description="Summary statistics")
    gmm_metadata: Optional[GMMMetadata] = Field(None, description="GMM metadata if adaptive thresholding used")
