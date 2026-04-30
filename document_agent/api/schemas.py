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
    library_item_id: Optional[UUID] = None
    job_id: UUID
    status: str
    library_url: Optional[str] = None


class BatchChildJob(BaseModel):
    library_item_id: Optional[UUID] = None
    job_id: UUID
    input_index: int
    filename: str
    status: str


class BatchCreatedResponse(UrlsMixin):
    batch_id: UUID
    status: str
    child_jobs: List[BatchChildJob]


class JobStatusResponse(BaseModel):
    library_item_id: Optional[UUID] = None
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
    library_item_id: Optional[UUID] = None
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


class LibraryItemResponse(BaseModel):
    library_item_id: UUID
    current_job_id: Optional[UUID] = None
    batch_id: Optional[UUID] = None
    input_index: Optional[int] = None
    filename: str
    content_type: Optional[str] = None
    detected_type: Optional[str] = None
    sha256: str
    size_bytes: int
    status: str
    stage: str
    percent: int
    preview_status: str
    created_at: datetime
    updated_at: datetime
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    original_url: str
    preview_url: str
    markdown_url: str
    events_url: Optional[str] = None
    has_markdown: bool = False
    has_preview: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LibraryListResponse(BaseModel):
    items: List[LibraryItemResponse]
    limit: int
    offset: int


class LibraryMarkdownResponse(BaseModel):
    library_item_id: UUID
    job_id: Optional[UUID] = None
    status: str
    markdown_url: Optional[str] = None
    asset_id: Optional[UUID] = None
    markdown: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReprocessResponse(UrlsMixin):
    library_item_id: UUID
    job_id: UUID
    status: str
    library_url: str


class DeleteLibraryItemResponse(BaseModel):
    library_item_id: UUID
    deleted: bool
    deleted_assets: int


# ---------------------------------------------------------------------------
# Observability schemas
# ---------------------------------------------------------------------------

class ObservabilityStatsResponse(BaseModel):
    total_jobs: int
    jobs_by_status: Dict[str, int]
    success_rate_pct: Optional[float] = None
    avg_duration_seconds: Optional[float] = None
    p95_duration_seconds: Optional[float] = None
    total_batches: int
    active_jobs: int
    throughput_by_hour: List[Dict[str, Any]]
    jobs_by_type: List[Dict[str, Any]]
    health: Dict[str, str]


class ObsEventRow(BaseModel):
    id: int
    library_item_id: Optional[UUID] = None
    batch_id: Optional[UUID] = None
    job_id: Optional[UUID] = None
    event_type: str
    stage: Optional[str] = None
    percent: Optional[int] = None
    message: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ObservabilityEventsResponse(BaseModel):
    events: List[ObsEventRow]
    has_more: bool
    next_before_id: Optional[int] = None


class ObsErrorItem(BaseModel):
    job_id: UUID
    library_item_id: Optional[UUID] = None
    filename: str
    detected_type: Optional[str] = None
    error_code: str
    error_message: Optional[str] = None
    failed_at: Optional[datetime] = None
    attempt_count: int


class ObsErrorCodeCount(BaseModel):
    error_code: str
    count: int


class ObservabilityErrorsResponse(BaseModel):
    errors: List[ObsErrorItem]
    error_code_counts: List[ObsErrorCodeCount]
    total_failed: int


class ObsLogRecord(BaseModel):
    seq: int
    ts: str
    level: str
    logger: str
    message: str


class ObservabilityLogsResponse(BaseModel):
    logs: List[ObsLogRecord]
    max_seq: int
    buffer_capacity: int
    buffer_used: int
