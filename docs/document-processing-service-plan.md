# Containerized Document Processing Service Plan

## Summary

Build `document-agent` as a production-oriented Python service that converts supported documents into AI-readable Markdown.

This document is the implementation source of truth. Implementation must follow this plan unless the plan is explicitly updated and reviewed first.

Plan governance:

- Treat this file as authoritative for architecture, API shape, storage behavior, job/batch semantics, and production constraints.
- Do not drift from this plan during implementation.
- If implementation reveals that the plan is incomplete, wrong, or impractical, pause implementation and ask for approval before changing the plan.
- Only update this plan after user approval.
- After the updated plan is reviewed, continue implementation against the updated source of truth.

Decisions locked:

- Primary interface: FastAPI job API plus CLI using the same pipeline.
- Execution model: always async jobs.
- Batch model: first-class batch submissions; each uploaded file becomes an independent child job under one `batch_id`.
- Worker model: separate worker container using Postgres-backed job leases.
- Storage: MinIO stores completed Markdown and extracted binary assets.
- Original uploads: temporary staging only, deleted after processing by default.
- Image assets: never persisted locally; extracted images are uploaded to MinIO and referenced in Markdown through HTTP URLs.
- Security v1: no auth, but keep middleware/config seam for API key auth later.
- Code reuse: extract/refactor needed `improved_ocr_agent` and `ia_phase1` pieces into clean packages inside `document-agent`.
- OCR stack: hybrid engines. Use deterministic converters where possible, VLM OCR where needed.

## Architecture

Create a Python repo with these services:

- `api`: FastAPI app for uploads, job status, SSE, and result retrieval.
- `worker`: long-running process that claims pending jobs from Postgres and performs conversion.
- `postgres`: batch/job metadata, status, file metadata, asset metadata, errors, and event log.
- `minio`: Markdown output and extracted assets.
- Optional OCR endpoint: OpenAI-compatible VLM OCR server, configured externally.

Container layout:

- `docker-compose.yml` for local production-like dev: `api`, `worker`, `postgres`, `minio`.
- One shared application image for `api` and `worker`, different commands.
- Docker image includes system tools: LibreOffice headless, Pandoc, Poppler, Tesseract fallback, libheif, ImageMagick policy as needed, PyMuPDF deps.

Python package layout:

- `document_agent/api`: FastAPI routes, SSE, schemas.
- `document_agent/worker`: job runner, leasing, retries.
- `document_agent/batches`: batch creation, aggregation, manifests, archive generation.
- `document_agent/converters`: file-type routing and converters.
- `document_agent/ocr`: extracted VLM OCR adapter from `improved_ocr_agent`.
- `document_agent/pdf`: extracted PDF-to-Markdown logic from `improved_ocr_agent` plus required `ia_phase1` modules.
- `document_agent/storage`: MinIO client, URL generation, object metadata.
- `document_agent/db`: Postgres models/repositories/migrations.
- `document_agent/cli`: Typer or Click CLI.

## Public API

Implement these endpoints:

- `POST /v1/jobs`
  - Multipart upload field: `file`.
  - Optional fields: `filename`, `content_type`, `metadata_json`.
  - Optional `Idempotency-Key` header prevents duplicate jobs when clients retry upload requests.
  - Streams upload to private temporary object storage, creates a Postgres job, returns immediately.
  - Response: `job_id`, `status`, `status_url`, `events_url`, `result_url`.

- `GET /v1/jobs/{job_id}`
  - Returns status: `queued`, `running`, `succeeded`, `failed`, `cancelled`.
  - Includes progress fields: `stage`, `percent`, timestamps, retry count, error summary if failed.

- `GET /v1/jobs/{job_id}/events`
  - SSE stream.
  - Emits durable job events from Postgres polling in v1.
  - Event types: `queued`, `started`, `progress`, `asset_uploaded`, `succeeded`, `failed`.

- `GET /v1/jobs/{job_id}/result`
  - If succeeded, returns metadata plus Markdown URL.
  - Supports `?include_markdown=true` to inline Markdown text.
  - Returns `409` if job not complete, `404` if unknown.

- `GET /v1/assets/{asset_id}`
  - Stable HTTP proxy to MinIO object.
  - Used in Markdown image links.
  - No auth in v1; add API key middleware later without changing Markdown URL shape.

- `DELETE /v1/jobs/{job_id}`
  - Marks queued/running job cancelled when possible.
  - Does not delete completed MinIO result unless explicit cleanup policy is later added.

- `POST /v1/batches`
  - Multipart upload fields: repeated `files`.
  - Optional fields: `metadata_json`, `batch_name`.
  - Optional `Idempotency-Key` header prevents duplicate batches when clients retry upload requests.
  - Streams every file to private temporary object storage before job creation is committed.
  - Creates one `document_batches` row and one `document_jobs` row per file.
  - Returns `batch_id`, aggregate status, child job IDs, `status_url`, `events_url`, and `result_url`.

- `GET /v1/batches/{batch_id}`
  - Returns aggregate status and per-file job status.
  - Batch status is `queued`, `running`, `succeeded`, `partial_failed`, `failed`, or `cancelled`.
  - One failed child job must not fail completed child jobs.

- `GET /v1/batches/{batch_id}/events`
  - SSE stream that emits both batch-level and child-job events.
  - Clients can render aggregate progress and per-file progress from the same stream.

- `GET /v1/batches/{batch_id}/result`
  - If all terminal, returns a batch manifest with every file's Markdown URL or error.
  - Supports `?archive=true` to return or create a ZIP asset containing all successful `.md` files plus `manifest.json`.
  - Returns `409` while any child job is still queued/running.

- `DELETE /v1/batches/{batch_id}`
  - Cancels queued/running child jobs.
  - Completed child results remain available unless a future cleanup policy deletes them.

CLI commands:

- `document-agent convert INPUT --output out.md`
  - Runs the same pipeline synchronously for local use.
  - For local CLI without MinIO, support `--asset-url-mode local` only for dev tests.
- `document-agent submit INPUT`
  - Calls the API and prints job ID.
- `document-agent batch INPUT... --output-dir ./markdown_outputs`
  - Accepts multiple files and/or directories.
  - Submits through `/v1/batches` by default.
  - Downloads one Markdown file per successful input plus `manifest.json`.
- `document-agent watch JOB_ID`
  - Connects to SSE and prints progress.
- `document-agent result JOB_ID --output out.md`
  - Fetches completed Markdown.

## Batch Processing

Batch processing is part of v1, not a future feature.

Behavior:

- A batch is a grouping and aggregation primitive; actual conversion work still happens per file.
- Every input file receives its own `job_id`, temp directory, retry state, result Markdown asset, and extracted asset set.
- Workers claim child jobs independently, so mixed file types can process in parallel up to configured worker capacity.
- Batch progress is computed from child jobs, weighted equally per file in v1.
- Batch terminal status:
  - `succeeded`: every child job succeeded.
  - `partial_failed`: at least one child succeeded and at least one child failed/cancelled.
  - `failed`: every child failed/cancelled.
  - `cancelled`: cancellation was requested before any child succeeded.
- Batch result shape must always include every submitted file, preserving input order.
- Batch ZIP output is optional and generated only when requested or explicitly configured.
- Batch creation is failure-safe: if staging any file fails, delete already staged objects and do not commit a partially created batch.
- If a retry uses the same `Idempotency-Key`, return the original batch instead of creating duplicate child jobs.

Batch output manifest:

- `batch_id`
- `created_at`, `finished_at`
- `total_files`, `succeeded_count`, `failed_count`, `cancelled_count`
- Per file:
  - `input_index`, `filename`, `job_id`, `status`
  - `markdown_url` when succeeded
  - `asset_count`
  - `error_code`, `error_message` when failed

Local CLI batch output:

```text
markdown_outputs/
  manifest.json
  lecture-notes.md
  scanned-page.md
  assignment.md
  paper.md
```

Filename collision handling:

- Preserve original basename when safe.
- Normalize unsafe characters.
- If names collide, append `-2`, `-3`, etc.
- The manifest records the original filename and final Markdown filename.

## Data Model

Use Postgres tables:

- `document_batches`
  - `id`, `status`, `batch_name`, `idempotency_key`, `total_files`, `succeeded_count`, `failed_count`, `cancelled_count`
  - `result_archive_asset_id`
  - `created_at`, `started_at`, `finished_at`, `cancelled_at`
  - `metadata_json`

- `document_jobs`
  - `id`, `batch_id`, `input_index`, `idempotency_key`, `status`, `filename`, `content_type`, `detected_type`, `sha256`, `size_bytes`
  - `source_bucket`, `source_object_key`, `source_deleted_at`
  - `stage`, `percent`, `attempt_count`, `max_attempts`
  - `created_at`, `started_at`, `finished_at`, `cancelled_at`
  - `result_markdown_asset_id`
  - `error_code`, `error_message`, `error_details_json`

- `document_assets`
  - `id`, `batch_id`, `job_id`, `role`
  - Roles: `markdown_result`, `embedded_image`, `equation_image`, `diagnostic_manifest`, `batch_manifest`, `batch_archive`
  - `bucket`, `object_key`, `mime_type`, `size_bytes`, `sha256`
  - `public_url`, `created_at`

- `job_events`
  - `id`, `batch_id`, `job_id`, `event_type`, `stage`, `percent`, `message`, `payload_json`, `created_at`

Worker leasing:

- Workers claim jobs with `SELECT ... FOR UPDATE SKIP LOCKED`.
- Each claimed job gets `lease_owner`, `lease_expires_at`.
- Worker periodically extends lease.
- Expired running jobs return to queued until `max_attempts` is exceeded.
- Batch aggregate counters/status are updated transactionally when a child job reaches a terminal state.

## Conversion Pipeline

Input detection:

- Use MIME sniffing from file bytes, not only extension.
- Enforce size/page limits through config.
- Supported v1: `pdf`, `jpg`, `jpeg`, `png`, `heic`, `txt`, `doc`, `docx`.
- Unsupported files fail with `UNSUPPORTED_FILE_TYPE`.

Routing:

- `pdf`: use extracted `improved_ocr_agent` PDF pipeline.
- `jpg/jpeg/png/heic`: normalize image in temp memory/disk, run OCR, produce Markdown.
- `txt`: normalize encoding, preserve structure, emit Markdown text.
- `doc/docx`: convert to PDF or HTML/Markdown using LibreOffice/Pandoc, then normalize to Markdown.
- If Office conversion produces embedded images, upload images to MinIO and replace references with HTTP asset URLs.

PDF strategy:

- Port the current `HybridPDFExtractor` behavior.
- Preserve page routing:
  - OCR pages use VLM OCR.
  - non-OCR scholarly/structured pages use extracted `ia_phase1` Markdown exporter.
  - text-heavy pages use native text path.
- Refactor local asset output so extracted images/equation images are uploaded to MinIO before final Markdown is persisted.
- Rewrite every Markdown image reference to `/v1/assets/{asset_id}` or configured absolute public base URL.

Image input strategy:

- Convert HEIC using `pillow-heif` or ImageMagick/libheif.
- Normalize orientation using EXIF.
- Convert image bytes to PNG in a temp working area.
- Send to existing VLM OCR adapter.
- Output Markdown contains OCR text only unless the model emits figure placeholders; for source image documents, do not persist the source image.

Office strategy:

- `.docx`: prefer Pandoc to Markdown when reliable.
- `.doc`: use LibreOffice headless to convert to `.docx` or PDF, then process.
- Embedded media extracted by Pandoc/LibreOffice must be uploaded to MinIO and removed from local temp storage.
- Tables should be emitted inline as Markdown/HTML tables, not image links, unless they are embedded screenshots.

Markdown output rules:

- Include YAML frontmatter with job ID, source filename, detected type, generated timestamp, converter version, and asset count.
- Use stable headings and page markers for multi-page documents.
- Use Markdown image syntax for assets: `![alt text](https://.../v1/assets/{asset_id})`.
- Do not embed base64 images.
- Do not reference local paths.
- Preserve tables as Markdown/HTML where possible.
- Preserve equations as LaTeX where possible; upload fallback equation images only if needed.

## Storage Rules

MinIO object keys:

- `jobs/{job_id}/result/document.md`
- `jobs/{job_id}/assets/images/{asset_id}.{ext}`
- `jobs/{job_id}/assets/equations/{asset_id}.{ext}`
- `jobs/{job_id}/manifests/conversion.json`
- `batches/{batch_id}/manifest.json`
- `batches/{batch_id}/archive/results.zip`

Local storage:

- Use per-job temp directory.
- Delete temp directory in `finally` after success/failure.
- Never store extracted images outside temp.
- No original upload retention by default.

Temporary source staging:

- API uploads source files to private staging keys so separate worker containers can process them after the request returns.
- Staging keys use `staging/jobs/{job_id}/source/{safe_filename}`.
- Workers download the source into a per-job temp directory, process it, then delete the staging object after terminal success/failure/cancellation.
- A cleanup task deletes orphaned staging objects older than the configured TTL.
- Source image inputs are staging-only; they are not exposed as assets and are not retained as durable outputs.

HTTP asset URLs:

- Default Markdown URL form: `{PUBLIC_BASE_URL}/v1/assets/{asset_id}`.
- API proxies MinIO object bytes with correct `Content-Type`.
- Later auth can be added at route/middleware level.

## Reliability

Job lifecycle:

- `queued -> running -> succeeded`
- `queued/running -> failed`
- `queued/running -> cancelled`

Batch lifecycle:

- `queued -> running -> succeeded`
- `queued/running -> partial_failed`
- `queued/running -> failed`
- `queued/running -> cancelled`
- Batch status is derived from child job terminal states, not manually guessed by workers.

Retries:

- Retry transient OCR, MinIO, and converter failures.
- Do not retry deterministic validation failures like unsupported type, too large, encrypted PDF without password, corrupt file.
- Store full error details in DB, return safe summary through API.

Timeout handling:

- FastAPI never processes documents in request handlers.
- Upload request only stages the source object and creates job metadata.
- SSE is informational; disconnecting does not affect processing.
- Result retrieval is independent from SSE.

Resource controls:

- Configurable max upload size.
- Configurable max files per batch.
- Configurable max total batch size.
- Configurable max PDF pages.
- Configurable OCR concurrency per worker.
- Configurable per-job timeout.
- Configurable per-batch wall-clock timeout for queued/running children.
- Worker process handles graceful shutdown by releasing or expiring leases.

Cleanup controls:

- Configurable staging-object TTL.
- Configurable completed-result retention policy, default retain until explicit deletion.
- Startup or scheduled cleanup reconciles terminal jobs with undeleted staging objects.

Observability:

- Structured JSON logs with `batch_id`, `job_id`, `stage`, `filename`, `duration_ms`.
- `/healthz`: process health.
- `/readyz`: DB and MinIO connectivity.
- `/metrics`: Prometheus counters/histograms for jobs, failures, OCR latency, conversion duration, asset uploads.

## Future Redis/SSE Extension

Do not require Redis in v1.

Design event publishing behind an interface:

- v1 publisher writes `job_events` rows.
- SSE endpoint polls Postgres.
- Future Redis publisher writes to Redis pub/sub and still persists important events to Postgres.
- Future distributed workers can move from Postgres leases to Redis/RQ/Celery without changing public API.

## Testing Plan

Unit tests:

- File type detection for all supported extensions and MIME mismatches.
- Markdown URL rewriting: no local paths, no base64 images.
- MinIO storage client with mocked MinIO.
- Job state transitions and retry classification.
- Temporary source staging upload, worker download, and cleanup.
- Batch status aggregation and per-file manifest generation.
- Batch filename collision handling.
- Idempotent job/batch create requests.
- Cleanup when batch staging partially fails.
- Worker leasing and expired lease recovery.
- Image normalization including HEIC fixture if available.
- TXT conversion with encoding detection.
- DOC/DOCX conversion adapter with mocked subprocess.

Integration tests:

- FastAPI upload returns job immediately.
- FastAPI batch upload returns batch ID and child job IDs immediately.
- Worker can process a job from a staged source object after the API request has ended.
- Worker processes a TXT file to Markdown.
- Worker processes an image file through mocked OCR.
- Worker processes a PDF through a small fixture and uploads assets.
- Result endpoint returns Markdown URL and inline Markdown.
- Batch result returns one Markdown URL per successful file and one error per failed file.
- Batch archive contains all successful Markdown files and `manifest.json`.
- Asset endpoint streams MinIO object.
- SSE emits queued/running/succeeded events.
- Batch SSE emits child job progress and aggregate progress.
- Temp directory is deleted after success and failure.

Container tests:

- Build image successfully.
- `docker compose up` starts API, worker, Postgres, MinIO.
- `/readyz` passes after dependencies are ready.
- LibreOffice, Pandoc, Poppler, and HEIC support are available inside the container.

Acceptance scenarios:

- Upload `.txt`, get Markdown result with no asset links.
- Upload `.png`, OCR result is Markdown, source image is not persisted.
- Upload `.heic`, service normalizes and OCRs it.
- Upload `.pdf` with figures, extracted figures are in MinIO and Markdown contains HTTP asset URLs.
- Upload `.docx` with embedded image, image is in MinIO and Markdown contains HTTP asset URL.
- Upload a mixed batch of `.pdf`, `.docx`, `.png`, `.heic`, and `.txt`; each successful file produces its own Markdown result.
- Upload a batch where one file is unsupported; supported files still complete and the batch status becomes `partial_failed`.
- Upload unsupported file, job fails with `UNSUPPORTED_FILE_TYPE`.
- Disconnect SSE during processing, job still completes and result is retrievable.
- Kill worker mid-job, another worker retries after lease expiry.

## Assumptions

- v1 has no API authentication because that was selected, but route structure keeps an auth seam.
- If deployed without auth in v1, the service must run behind a private network, gateway, or other external access control.
- MinIO is required for production deployments.
- Postgres is required for production job metadata and worker leases.
- Redis is intentionally deferred.
- Original uploaded files are temporary and deleted after conversion.
- Source files may exist briefly as private staging objects so async workers can process them; they are not durable retained originals.
- Extracted image assets are stored only in MinIO, never durably on local disk.
- Existing Instructor Assistant code will be extracted and refactored, not imported from the app repo at runtime.
- OCR uses an external OpenAI-compatible VLM endpoint by config; deterministic converters handle non-OCR text/Office paths where possible.
- Batch processing does not concatenate documents by default; it produces one Markdown result per input file plus a manifest/archive.
