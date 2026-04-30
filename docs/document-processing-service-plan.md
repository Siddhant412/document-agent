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
- Storage: MinIO stores durable original uploads, generated previews, completed Markdown, and extracted binary assets.
- Original uploads: retained durably in MinIO as private library source objects, deleted only by explicit library deletion or configured original-retention policy.
- File library: every upload creates a durable library item; processing is queued in the background and updates that item with preview/result status.
- Image assets: never persisted locally; extracted images are uploaded to MinIO and referenced in Markdown through HTTP URLs. Source image inputs are retained only as original library files and are not embedded back into Markdown.
- Security v1: no auth, but keep middleware/config seam for API key auth later.
- Code reuse: extract/refactor needed `improved_ocr_agent` and `ia_phase1` pieces into clean packages inside `document-agent`.
- OCR stack: hybrid engines. Use deterministic converters where possible, VLM OCR where needed.

## Architecture

Create a Python repo with these services:

- `api`: FastAPI app for uploads, file library, previews, job status, SSE, and result retrieval.
- `worker`: long-running process that claims pending jobs from Postgres and performs conversion.
- `postgres`: library, batch/job metadata, status, file metadata, asset metadata, errors, and event log.
- `minio`: durable original uploads, generated previews, Markdown output, and extracted assets.
- Optional OCR endpoint: OpenAI-compatible VLM OCR server, configured externally.

Container layout:

- `docker-compose.yml` for local production-like dev: `api`, `worker`, `postgres`, `minio`.
- One shared application image for `api` and `worker`, different commands.
- Docker image includes system tools: LibreOffice headless, Pandoc, Poppler, Tesseract fallback, libheif, ImageMagick policy as needed, PyMuPDF deps.

Python package layout:

- `document_agent/api`: FastAPI routes, SSE, schemas.
- `document_agent/worker`: job runner, leasing, retries.
- `document_agent/batches`: batch creation, aggregation, manifests, archive generation.
- `document_agent/library`: durable upload records, file library queries, deletion/reprocessing policies.
- `document_agent/previews`: browser preview generation for Office, HEIC, PDFs, images, and text.
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
  - Streams upload to a private durable MinIO original object, creates one `document_library_items` row and one `document_jobs` row in a single transaction, returns immediately.
  - Response: `library_item_id`, `job_id`, `status`, `library_url`, `status_url`, `events_url`, `result_url`.

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
  - Does not delete the durable original file or completed MinIO results. Library deletion/retention policy owns stored objects.

- `POST /v1/batches`
  - Multipart upload fields: repeated `files`.
  - Optional fields: `metadata_json`, `batch_name`.
  - Optional `Idempotency-Key` header prevents duplicate batches when clients retry upload requests.
  - Streams every file to private durable MinIO original storage before job creation is committed.
  - Creates one `document_batches` row plus one `document_library_items` row and one `document_jobs` row per file.
  - Returns `batch_id`, aggregate status, child `library_item_id`/`job_id` pairs, `status_url`, `events_url`, and `result_url`.

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
  - Completed child originals/results remain available unless explicit library deletion or configured retention deletes them.

File library endpoints:

- `GET /v1/library`
  - Paginated, sortable, filterable list of uploaded files.
  - Filters: `status`, `detected_type`, `batch_id`, creation time range, filename query.
  - Returns each item with `library_item_id`, current `job_id`, `filename`, `content_type`, `detected_type`, `size_bytes`, `status`, `stage`, `percent`, `created_at`, `updated_at`, and preview/result URL fields when available.

- `GET /v1/library/{library_item_id}`
  - Returns durable file metadata, current processing status, active/latest job metadata, preview metadata, Markdown result metadata, extracted asset counts, and error summary if failed.

- `GET /v1/library/{library_item_id}/original`
  - Streams the durable original upload from MinIO through the API with correct `Content-Type` and safe `Content-Disposition`.
  - Supports `Range` requests where practical for browser PDF/video-like viewers and large file previews.

- `GET /v1/library/{library_item_id}/preview`
  - Streams or redirects to the best browser-safe preview.
  - Uses the original object directly for browser-compatible images, PDFs, and text when safe.
  - Uses generated preview assets for HEIC, DOC, DOCX, or any format that needs normalization.
  - Returns `409` while preview generation is pending and `404` only when the library item does not exist.

- `GET /v1/library/{library_item_id}/markdown`
  - Returns completed Markdown metadata and supports `?include_markdown=true`.
  - Equivalent to following the current job result, but stable for UI code keyed by library item.

- `POST /v1/library/{library_item_id}/reprocess`
  - Creates a new job from the retained original file without requiring the user to upload again.
  - Keeps previous successful results unless explicit replacement policy is configured.

- `DELETE /v1/library/{library_item_id}`
  - Cancels queued/running work for that item.
  - Deletes the durable original, generated previews, Markdown result, extracted assets, and related asset rows according to the configured deletion policy.
  - Keeps event/job metadata as tombstoned audit records unless hard-delete mode is explicitly configured.

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

## File Library UI

Build a production web UI for the file library in addition to the API and CLI.

Frontend architecture:

- Use a small React + TypeScript + Vite app under `document_agent/ui` or `frontend`, built into static assets and served by FastAPI under `/app` in production.
- The UI must use only the public API endpoints described in this plan; it must not reach into the database or MinIO directly.
- Keep API client code typed and centralized so SSE, polling fallback, upload progress, and error handling are consistent.
- The UI must support deployments where API auth is added later by reading an API key/token from the same future auth seam instead of hard-coding credentials.

Primary screen:

- First screen is the actual file library workspace, not a marketing/landing page.
- Full viewport application layout.
- Left column uses approximately one fourth of the screen width and contains:
  - Upload button/drop zone supporting single and multiple mixed files.
  - Search and filters for status/type.
  - Uploaded file list with filename, detected type, size, status, progress, and error indicator.
  - Batch grouping when files were uploaded together.
- Remaining three fourths of the screen is split into two columns:
  - Left detail column: selected file preview.
  - Right detail column: generated Markdown result.
- The layout must remain usable on smaller screens by collapsing to tabs or stacked panes, while preserving the same information architecture.

Upload UX:

- Users can drag and drop or select multiple supported files in one action.
- Uploading files immediately creates durable library items and queues background processing jobs.
- The list updates as each upload is accepted, queued, running, succeeded, failed, or cancelled.
- Upload progress and processing progress are separate states; a file can finish uploading but still be queued for conversion.
- Batch uploads preserve input order and show per-file outcomes. One failed file must not hide successful files.
- Re-upload retries should use idempotency where available to avoid duplicate library entries after network retries.

Preview UX:

- The selected file preview must render all supported input types:
  - `jpg`, `jpeg`, `png`: render original image through the original-file endpoint.
  - `heic`: render generated browser-safe PNG/JPEG preview.
  - `pdf`: render original PDF through the original-file endpoint when browser-compatible; fall back to generated page previews if needed.
  - `txt`: render text preview with encoding-normalized content.
  - `doc`, `docx`: render generated PDF or HTML preview produced from the retained original.
- Preview assets are stored in MinIO and streamed/proxied by the API. No preview image or converted preview file is durably stored on local disk.
- Preview pane must handle pending preview, failed preview, unsupported preview, large file, and deleted file states clearly.

Markdown result UX:

- Markdown pane shows queued/running state until conversion succeeds.
- On success, fetch Markdown through `/v1/library/{library_item_id}/markdown?include_markdown=true`.
- Render Markdown in a readable preview mode and provide a raw Markdown view/copy/download affordance.
- Markdown image links must point to API asset URLs, not local paths or base64 payloads.
- If processing fails, show safe error code/message and keep the original preview available for inspection/reprocess.

Realtime behavior:

- Use SSE for active selected item and active batches where available.
- Fall back to polling `/v1/library` and `/v1/library/{library_item_id}` when SSE disconnects or the browser/network blocks streaming.
- SSE disconnects must not cancel processing.
- UI state must reconcile from durable API state after page refresh.

Deletion and reprocessing UX:

- Delete action removes the library item and all configured stored objects through the API, after user confirmation.
- Cancel action cancels queued/running conversion without deleting the durable original.
- Reprocess action creates a new job from the retained original file and updates the library item with latest job status.

## Batch Processing

Batch processing is part of v1, not a future feature.

Behavior:

- A batch is a grouping and aggregation primitive; actual conversion work still happens per file.
- Every input file receives its own `library_item_id`, `job_id`, temp directory, retry state, durable original object, result Markdown asset, generated preview assets, and extracted asset set.
- Workers claim child jobs independently, so mixed file types can process in parallel up to configured worker capacity.
- Batch progress is computed from child jobs, weighted equally per file in v1.
- Batch terminal status:
  - `succeeded`: every child job succeeded.
  - `partial_failed`: at least one child succeeded and at least one child failed/cancelled.
  - `failed`: every child failed/cancelled.
  - `cancelled`: cancellation was requested before any child succeeded.
- Batch result shape must always include every submitted file, preserving input order.
- Batch ZIP output is optional and generated only when requested or explicitly configured.
- Batch creation is failure-safe: if durable original storage for any file fails, delete already stored original objects and do not commit a partially created batch.
- If a retry uses the same `Idempotency-Key`, return the original batch instead of creating duplicate child jobs.

Batch output manifest:

- `batch_id`
- `created_at`, `finished_at`
- `total_files`, `succeeded_count`, `failed_count`, `cancelled_count`
- Per file:
  - `input_index`, `filename`, `job_id`, `status`
  - `library_item_id`
  - `markdown_url` when succeeded
  - `preview_url` when available
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

- `document_library_items`
  - `id`, `batch_id`, `current_job_id`, `input_index`
  - `original_filename`, `display_filename`, `content_type`, `detected_type`, `sha256`, `size_bytes`
  - `status`, `stage`, `percent`
  - `original_asset_id`, `preview_asset_id`, `thumbnail_asset_id`, `latest_markdown_asset_id`
  - `created_at`, `updated_at`, `uploaded_at`, `processed_at`, `deleted_at`
  - `error_code`, `error_message`, `error_details_json`
  - `metadata_json`
  - Purpose: durable file-library row that survives individual job retries/reprocessing and owns the retained original upload.

- `document_jobs`
  - `id`, `library_item_id`, `batch_id`, `input_index`, `idempotency_key`, `status`, `filename`, `content_type`, `detected_type`, `sha256`, `size_bytes`
  - `source_asset_id`
  - `stage`, `percent`, `attempt_count`, `max_attempts`
  - `created_at`, `started_at`, `finished_at`, `cancelled_at`
  - `result_markdown_asset_id`
  - `error_code`, `error_message`, `error_details_json`
  - Purpose: one processing attempt stream for a library item. Reprocessing creates a new job and updates `document_library_items.current_job_id`.

- `document_assets`
  - `id`, `library_item_id`, `batch_id`, `job_id`, `role`
  - Roles: `original_file`, `markdown_result`, `embedded_image`, `equation_image`, `preview_pdf`, `preview_image`, `preview_text`, `thumbnail`, `diagnostic_manifest`, `batch_manifest`, `batch_archive`
  - `bucket`, `object_key`, `mime_type`, `size_bytes`, `sha256`
  - `public_url`, `created_at`
  - Original files use role `original_file` and are private by default; access is through library API endpoints.

- `job_events`
  - `id`, `library_item_id`, `batch_id`, `job_id`, `event_type`, `stage`, `percent`, `message`, `payload_json`, `created_at`

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
- Output Markdown contains OCR text only unless the model emits figure placeholders.
- Source image originals are retained as durable `original_file` assets for library preview/reprocessing, but they are not re-added to Markdown as embedded document assets.

Office strategy:

- `.docx`: prefer Pandoc to Markdown when reliable.
- `.doc`: use LibreOffice headless to convert to `.docx` or PDF, then process.
- Embedded media extracted by Pandoc/LibreOffice must be uploaded to MinIO and removed from local temp storage.
- Tables should be emitted inline as Markdown/HTML tables, not image links, unless they are embedded screenshots.

Preview generation strategy:

- Generate previews from the retained original file independently from Markdown conversion when the original is not browser-safe.
- Use original files directly for browser-safe preview types where possible: PDF, JPEG, PNG, and normalized text.
- Generate PNG/JPEG previews for HEIC.
- Generate PDF or HTML previews for DOC/DOCX through LibreOffice/Pandoc.
- Store preview artifacts in MinIO with roles `preview_pdf`, `preview_image`, `preview_text`, or `thumbnail`.
- Preview generation failures must not automatically fail Markdown conversion, and Markdown conversion failures must not remove the original file preview.

Markdown output rules:

- Include YAML frontmatter with library item ID, job ID, source filename, detected type, generated timestamp, converter version, and asset count.
- Use stable headings and page markers for multi-page documents.
- Use Markdown image syntax for assets: `![alt text](https://.../v1/assets/{asset_id})`.
- Do not embed base64 images.
- Do not reference local paths.
- Preserve tables as Markdown/HTML where possible.
- Preserve equations as LaTeX where possible; upload fallback equation images only if needed.

## Storage Rules

MinIO object keys:

- `library/{library_item_id}/original/{safe_filename}`
- `library/{library_item_id}/previews/preview.pdf`
- `library/{library_item_id}/previews/preview.html`
- `library/{library_item_id}/previews/page-{page_number}.png`
- `library/{library_item_id}/previews/thumbnail.png`
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
- Never store original uploads durably on local disk.
- Local temp copies of originals, previews, converted intermediates, and extracted assets must be deleted after the operation finishes.

Durable source storage:

- API uploads source files to private durable MinIO keys under `library/{library_item_id}/original/`.
- Each original upload creates a `document_assets` row with role `original_file` and is referenced by `document_library_items.original_asset_id` and `document_jobs.source_asset_id`.
- Workers download the retained original into a per-job temp directory, process it, and delete only the local temp copy.
- Original objects remain available for preview, download, and reprocessing after conversion completes.
- Source image inputs are exposed only through library original/preview endpoints; they are not treated as extracted image assets and are not inserted into Markdown unless they are explicitly part of another document.
- Cleanup tasks reconcile orphaned uploaded originals that were stored but never committed to a library item because an upload transaction failed.

Deletion and retention:

- Default policy retains original uploads, generated previews, Markdown results, extracted assets, manifests, and batch archives until explicit deletion.
- `DELETE /v1/library/{library_item_id}` deletes the durable original and all item-scoped generated objects according to configured policy.
- Completed-result retention may delete Markdown/results/previews/extracted assets after a configured TTL, but must not delete original uploads unless original retention is explicitly enabled.
- Original-retention policy is opt-in and must be configured separately from result retention.

HTTP asset URLs:

- Default Markdown URL form: `{PUBLIC_BASE_URL}/v1/assets/{asset_id}`.
- API proxies MinIO object bytes with correct `Content-Type`.
- Original and preview files use library endpoints, not Markdown asset URLs, unless they are extracted document assets that belong in Markdown.
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
- Upload request only stores the durable original object, creates library/job metadata, and returns.
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

- Configurable orphaned-upload TTL for objects stored before a transaction failed or before DB metadata was committed.
- Configurable completed-result retention policy, default retain until explicit deletion.
- Configurable original-file retention policy, default retain until explicit library deletion.
- Startup or scheduled cleanup reconciles object storage with library/assets metadata and removes only confirmed orphaned objects.

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
- Durable original upload asset creation, worker download from original asset, and orphaned-upload cleanup.
- Library item status updates across upload, queue, running, success, failure, cancel, delete, and reprocess.
- Preview routing for browser-safe originals versus generated preview assets.
- Batch status aggregation and per-file manifest generation.
- Batch filename collision handling.
- Idempotent job/batch create requests.
- Cleanup when batch durable-original storage partially fails.
- Worker leasing and expired lease recovery.
- Image normalization including HEIC fixture if available.
- TXT conversion with encoding detection.
- DOC/DOCX conversion adapter with mocked subprocess.

Integration tests:

- FastAPI upload returns job immediately.
- FastAPI batch upload returns batch ID and child job IDs immediately.
- Worker can process a job from a retained original object after the API request has ended.
- Library list endpoint shows uploaded items and current statuses.
- Library detail endpoint returns original, preview, current job, and Markdown result metadata.
- Original endpoint streams durable uploaded files with correct `Content-Type`.
- Preview endpoint streams original-compatible files and generated preview assets for HEIC/DOC/DOCX.
- Worker processes a TXT file to Markdown.
- Worker processes an image file through mocked OCR.
- Worker processes a PDF through a small fixture and uploads assets.
- Result endpoint returns Markdown URL and inline Markdown.
- Library Markdown endpoint returns the latest successful Markdown for the selected item.
- Reprocess endpoint creates a new job from the retained original without another upload.
- Library delete endpoint removes/cancels the item and deletes configured MinIO objects.
- Batch result returns one Markdown URL per successful file and one error per failed file.
- Batch archive contains all successful Markdown files and `manifest.json`.
- Asset endpoint streams MinIO object.
- SSE emits queued/running/succeeded events.
- Batch SSE emits child job progress and aggregate progress.
- Temp directory is deleted after success and failure.
- UI upload flow creates library items, queues background processing, updates the left file list, renders file preview, and renders Markdown result.
- UI recovers current library/job state after browser refresh.

Container tests:

- Build image successfully.
- `docker compose up` starts API, worker, Postgres, MinIO.
- `/readyz` passes after dependencies are ready.
- LibreOffice, Pandoc, Poppler, and HEIC support are available inside the container.

Acceptance scenarios:

- Upload `.txt`, get Markdown result with no asset links.
- Upload `.png`, original image is retained as a private library original for preview/reprocess, OCR result is Markdown, and Markdown does not embed the source image unless the OCR output explicitly describes content.
- Upload `.heic`, service normalizes and OCRs it.
- Upload `.pdf` with figures, extracted figures are in MinIO and Markdown contains HTTP asset URLs.
- Upload `.docx` with embedded image, image is in MinIO and Markdown contains HTTP asset URL.
- Upload a mixed batch of `.pdf`, `.docx`, `.png`, `.heic`, and `.txt`; each successful file produces its own Markdown result.
- Upload multiple mixed files through the UI; every file appears in the library list, gets queued independently, and shows its own preview and Markdown output.
- Select each supported file type in the UI and see a browser-safe preview backed by MinIO original/preview assets.
- Upload a batch where one file is unsupported; supported files still complete and the batch status becomes `partial_failed`.
- Upload unsupported file, job fails with `UNSUPPORTED_FILE_TYPE`.
- Disconnect SSE during processing, job still completes and result is retrievable.
- Kill worker mid-job, another worker retries after lease expiry.
- Reprocess an existing library item without re-uploading the original.
- Delete a library item and verify original, preview, Markdown, and extracted assets are removed according to policy while local disk remains clean.

## Assumptions

- v1 has no API authentication because that was selected, but route structure keeps an auth seam.
- If deployed without auth in v1, the service must run behind a private network, gateway, or other external access control.
- MinIO is required for production deployments.
- Postgres is required for production job metadata and worker leases.
- Redis is intentionally deferred.
- Original uploaded files are durable private MinIO objects owned by file-library items.
- Source files are retained after conversion for preview and reprocessing unless explicit library deletion or configured original-retention policy removes them.
- Durable original storage does not change the rule that local temp copies are deleted after processing.
- Extracted image assets are stored only in MinIO, never durably on local disk.
- Existing Instructor Assistant code will be extracted and refactored, not imported from the app repo at runtime.
- OCR uses an external OpenAI-compatible VLM endpoint by config; deterministic converters handle non-OCR text/Office paths where possible.
- Batch processing does not concatenate documents by default; it produces one Markdown result per input file plus a manifest/archive.
