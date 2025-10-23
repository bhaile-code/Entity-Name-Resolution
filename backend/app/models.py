"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional
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


class ProcessingResult(BaseModel):
    """Result of processing a CSV file."""
    mappings: List[CompanyMapping] = Field(..., description="Company name mappings")
    audit_log: AuditLog = Field(..., description="Detailed audit log")
    summary: dict = Field(..., description="Summary statistics")
