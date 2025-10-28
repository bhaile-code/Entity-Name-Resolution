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


class ProcessingResult(BaseModel):
    """Result of processing a CSV file."""
    mappings: List[CompanyMapping] = Field(..., description="Company name mappings")
    audit_log: AuditLog = Field(..., description="Detailed audit log")
    summary: dict = Field(..., description="Summary statistics")
    gmm_metadata: Optional[GMMMetadata] = Field(None, description="GMM metadata if adaptive thresholding used")
