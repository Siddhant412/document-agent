from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class UrlsMixin(BaseModel):
    status_url: str
    events_url: str
    result_url: str


class JobCreatedResponse(UrlsMixin):
    job_id: UUID
    status: str


class BatchChildJob(BaseModel):
    job_id: UUID
    input_index: int
    filename: str
    status: str


class BatchCreatedResponse(UrlsMixin):
    batch_id: UUID
    status: str
    child_jobs: List[BatchChildJob]


class JobStatusResponse(BaseModel):
    job_id: UUID
    batch_id: Optional[UUID] = None
    input_index: Optional[int] = None
    status: str
    filename: str
    content_type: Optional[str] = None
    detected_type: Optional[str] = None
    sha256: str
    size_bytes: int
    stage: str
    percent: int
    attempt_count: int
    max_attempts: int
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    result_url: str
    events_url: str


class BatchStatusResponse(BaseModel):
    batch_id: UUID
    status: str
    percent: int
    batch_name: Optional[str] = None
    total_files: int
    succeeded_count: int
    failed_count: int
    cancelled_count: int
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    jobs: List[JobStatusResponse]
    result_url: str
    events_url: str


class JobResultResponse(BaseModel):
    job_id: UUID
    status: str
    markdown_url: str
    asset_id: UUID
    markdown: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchManifestFile(BaseModel):
    input_index: Optional[int]
    filename: str
    job_id: UUID
    status: str
    markdown_url: Optional[str] = None
    markdown_filename: Optional[str] = None
    asset_count: int = 0
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class BatchResultResponse(BaseModel):
    batch_id: UUID
    status: str
    manifest_url: Optional[str] = None
    archive_url: Optional[str] = None
    files: List[BatchManifestFile]
