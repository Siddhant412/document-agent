from __future__ import annotations

from prometheus_client import Counter, Histogram

JOBS_CREATED = Counter(
    "document_agent_jobs_created_total",
    "Document conversion jobs created.",
    ("submission_type", "detected_type"),
)

BATCHES_CREATED = Counter(
    "document_agent_batches_created_total",
    "Document conversion batches created.",
)

JOBS_COMPLETED = Counter(
    "document_agent_jobs_completed_total",
    "Document conversion jobs that reached a terminal worker outcome.",
    ("status", "detected_type", "error_code"),
)

CONVERSION_DURATION_SECONDS = Histogram(
    "document_agent_conversion_duration_seconds",
    "Wall-clock conversion duration per job.",
    ("status", "detected_type"),
    buckets=(0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
)

OCR_REQUESTS = Counter(
    "document_agent_ocr_requests_total",
    "OCR provider requests.",
    ("status", "model", "error_code"),
)

OCR_REQUEST_DURATION_SECONDS = Histogram(
    "document_agent_ocr_request_duration_seconds",
    "OCR provider request duration.",
    ("status", "model"),
    buckets=(0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 240),
)

ASSETS_UPLOADED = Counter(
    "document_agent_assets_uploaded_total",
    "Assets uploaded to object storage.",
    ("role",),
)
