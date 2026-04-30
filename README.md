# document-agent

Containerized document processing service that converts supported files into AI-readable Markdown.

The implementation source of truth is [docs/document-processing-service-plan.md](docs/document-processing-service-plan.md).

## What It Supports

Input formats:

- `pdf`
- `jpg`, `jpeg`, `png`, `heic`
- `doc`, `docx`
- `txt`

Outputs:

- One Markdown result per input file.
- Original uploaded files are retained as private MinIO library objects for preview and reprocessing.
- Browser-safe previews are available through the file library UI and `/v1/library/{library_item_id}/preview`.
- Extracted document assets, such as figures from PDFs or Office files, are stored in MinIO.
- Markdown links extracted assets through HTTP URLs like `http://localhost:8080/v1/assets/{asset_id}`.
- Source image inputs are OCRed into Markdown and retained only as private original library files, not embedded back into Markdown.

<p align="center">
  <br>
  <img src="assets/ui_home.png" alt="Home UI Screenshot" width="700">
  <br>
  <em>Document Library UI</em>
</p>

<br>

<p align="center">
  <img src="assets/ui_obs.png" alt="Observability UI Screenshot" width="700">
  <br>
  <em>Observability UI</em>
  <br>
</p>

Runtime services:

- `api`: FastAPI upload/library/status/SSE/result API and web UI on `http://localhost:8080`.
- `worker`: async document processor and worker metrics on `http://localhost:8081/metrics`.
- `postgres`: job, batch, lease, asset, and event metadata.
- `minio`: Markdown result and extracted asset object storage.

## Prerequisites

- Docker Desktop or Docker Engine with Compose.
- `curl`.
- `python3` or `python3.11` on the host for JSON parsing in the examples.
- An OpenAI-compatible OCR endpoint for image/scanned PDF OCR.

No local Postgres, MinIO, LibreOffice, Pandoc, or Python virtualenv is required for normal Docker usage. Those run inside containers.

## Configure

From the `document-agent` directory:

```bash
cp .env.example .env
```

`docker-compose.yml` loads `.env.example` first and `.env` second, so local secrets and provider overrides in `.env` take precedence without being committed.

Set OCR values in `.env` for real image and scanned PDF OCR:

```dotenv
OCR_SERVER_URL=https://your-openai-compatible-provider/v1
OCR_API_KEY=your-provider-api-key
OCR_MODEL=your-ocr-model-id
```

`OCR_SERVER_URL` can be either a provider base URL or the full `/chat/completions` URL. For DashScope compatible mode this is typically:

```dotenv
OCR_SERVER_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
OCR_API_KEY=your-dashscope-api-key
OCR_MODEL=qwen-vl-ocr
```

Optional settings:

```dotenv
API_KEY=
MAX_UPLOAD_BYTES=104857600
MAX_BATCH_FILES=100
MAX_BATCH_BYTES=1073741824
MAX_PDF_PAGES=500
OCR_MAX_CONCURRENT_REQUESTS=2
COMPLETED_RESULT_RETENTION_SECONDS=0
ORIGINAL_FILE_RETENTION_SECONDS=0
```

`COMPLETED_RESULT_RETENTION_SECONDS=0` keeps completed Markdown/assets until explicit deletion. Set a positive value only when old completed results should be deleted automatically by the worker.
`ORIGINAL_FILE_RETENTION_SECONDS=0` keeps uploaded originals until explicit library deletion.

## Start

```bash
docker compose up -d --build
```

Check readiness:

```bash
curl -fsS http://localhost:8080/readyz
```

Expected response:

```json
{"status":"ready"}
```

Useful service URLs:

- File library UI: `http://localhost:8080/app`
- Observability UI: `http://localhost:8080/app/observability`
- API health: `http://localhost:8080/healthz`
- API readiness: `http://localhost:8080/readyz`
- API docs: `http://localhost:8080/docs`
- API metrics: `http://localhost:8080/metrics`
- Worker metrics: `http://localhost:8081/metrics`
- MinIO console: `http://localhost:9001`

Local MinIO credentials from `docker-compose.yml`:

```text
username: minioadmin
password: minioadmin
```

## Use The Web UI

Open:

```text
http://localhost:8080/app
```

The UI supports single and multi-file upload, background queue status, a file library list, original/converted previews, Markdown preview/raw views, Markdown download, delete, and reprocess from the retained original file.

If `API_KEY` is set, open the settings button in the UI and enter the same key. Static UI files remain public, but API calls still require the configured key.

## Observability UI

Open in a new tab from the library UI (click the **Observability** button in the toolbar), or go directly to:

```text
http://localhost:8080/app/observability
```

The page has four tabs:

**Overview** — high-level system snapshot:
- Stat cards: total jobs, success rate, active jobs, failed jobs, average conversion duration.
- Bar chart: stacked succeeded/failed job counts per hour over the last 24 hours.
- Donut chart: job distribution by detected file type.
- Health badges: API, Database, and Worker status.

**Events** — global feed from the `job_events` table:
- Filter by event type (`queued`, `started`, `progress`, `succeeded`, `failed`, `asset_uploaded`) and free-text search on message content.
- Newest events first. Click **Load older events** for cursor-based pagination.
- When auto-refresh is on, new events are prepended with a highlighted banner.

**Logs** — in-process API log ring buffer (last 2000 records):
- Filter by log level and free-text search.
- Dark monospace viewer with auto-tail when scrolled to the bottom.
- Polls only new records (`since_seq`) on each refresh cycle so the viewer appends incrementally.
- These are API process logs only. Worker logs are in a separate process; use `docker compose logs -f worker` or the Events tab (which shows what the worker did for each job) to trace worker activity.

**Metrics** — charts derived from the same data as Overview:
- Area chart: throughput trend over the last 24 hours.
- Donut chart: current job status breakdown.
- Bar chart: error rate (%) per hour.
- Horizontal bar chart: top error codes by frequency.
- Duration stats panel: average and p95 conversion time, total batches, active leases.

**Auto-refresh** cycles through Off / 5 s / 15 s / 30 s. Click the button in the top-right corner to change the interval.

The observability endpoints are backed by four REST routes:

```text
GET /v1/observability/stats
GET /v1/observability/events
GET /v1/observability/errors
GET /v1/observability/logs
```

All four are accessible without auth when `API_KEY` is not configured, consistent with the rest of the API.

## Convert One File

Use any supported file path. These examples assume you are already inside the `document-agent` directory. If the `test_data` files are not present in your checkout, replace those paths with your own local files.

```bash
FILE="./test_data/testfile.txt"

response=$(curl -fsS \
  -F "file=@${FILE}" \
  http://localhost:8080/v1/jobs)

echo "$response"

job_id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["job_id"])' "$response")
library_item_id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["library_item_id"])' "$response")
echo "$job_id"
echo "$library_item_id"
```

Poll until terminal:

```bash
while true; do
  status_json=$(curl -fsS "http://localhost:8080/v1/jobs/${job_id}")
  echo "$status_json"
  status=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["status"])' <<< "$status_json")
  case "$status" in
    succeeded|failed|cancelled) break ;;
  esac
  sleep 2
done
```

Fetch result metadata and inline Markdown:

```bash
curl -fsS "http://localhost:8080/v1/jobs/${job_id}/result?include_markdown=true"
```

Fetch the same result through the stable library item endpoint:

```bash
curl -fsS "http://localhost:8080/v1/library/${library_item_id}/markdown?include_markdown=true"
```

Save only the Markdown to a local file:

```bash
curl -fsS "http://localhost:8080/v1/jobs/${job_id}/result?include_markdown=true" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["markdown"])' \
  > output.md
```

If `API_KEY` is set in `.env`, include the configured header on protected API endpoints:

```bash
curl -fsS -H "X-API-Key: ${API_KEY}" "http://localhost:8080/v1/jobs/${job_id}"
```

## Convert A Batch

Submit multiple files with repeated `files` fields:

```bash
response=$(curl -fsS \
  -F "files=@./test_data/testfile.txt" \
  -F "files=@./test_data/IMG_6040.jpg" \
  -F "files=@./test_data/test.docx" \
  http://localhost:8080/v1/batches)

echo "$response"

batch_id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["batch_id"])' "$response")
echo "$batch_id"
```

Poll batch status:

```bash
while true; do
  status_json=$(curl -fsS "http://localhost:8080/v1/batches/${batch_id}")
  echo "$status_json"
  status=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["status"])' <<< "$status_json")
  case "$status" in
    succeeded|partial_failed|failed|cancelled) break ;;
  esac
  sleep 2
done
```

Fetch the batch result manifest:

```bash
curl -fsS "http://localhost:8080/v1/batches/${batch_id}/result"
```

Request a ZIP archive with all successful Markdown files and `manifest.json`:

```bash
curl -fsS "http://localhost:8080/v1/batches/${batch_id}/result?archive=true"
```

A mixed batch can finish as `partial_failed`. Successful child jobs still get Markdown URLs, and failed child jobs include `error_code` and `error_message`.

## Watch SSE Events

Job events:

```bash
curl -N "http://localhost:8080/v1/jobs/${job_id}/events"
```

Batch events:

```bash
curl -N "http://localhost:8080/v1/batches/${batch_id}/events"
```

Disconnecting SSE does not cancel processing. Fetch status/result later with the normal endpoints.

## Use The CLI

The CLI is installed inside the Docker image. Run help through the API container:

```bash
docker compose exec api document-agent --help
```

Submit through the API from a one-off container with your repository mounted:

```bash
docker compose run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e API_BASE_URL=http://api:8080 \
  api document-agent submit ./test_data/testfile.txt
```

Run a batch through the API:

```bash
docker compose run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  -e API_BASE_URL=http://api:8080 \
  api document-agent batch ./test_data/testfile.txt ./test_data/IMG_6040.jpg --wait
```

Synchronous local conversion is mainly for development:

```bash
docker compose run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  api document-agent convert ./test_data/testfile.txt --output ./markdown_outputs/testfile.md
```

## Operational Commands

View logs:

```bash
docker compose logs -f api worker
```

Rebuild after code changes:

```bash
docker compose up -d --build api worker
```

Stop services:

```bash
docker compose down
```

Stop services and delete local Postgres/MinIO volumes:

```bash
docker compose down -v
```

## Tests

Unit tests do not require Docker:

```bash
python3.11 -m pytest -q
```

Real Postgres/MinIO integration tests are opt-in. Run them inside the Compose network:

```bash
docker compose run --rm -v "$PWD:/workspace" -w /workspace \
  -e DOCUMENT_AGENT_RUN_INTEGRATION=1 \
  -e DOCUMENT_AGENT_INTEGRATION_DATABASE_URL=postgresql://document_agent:document_agent@postgres:5432/document_agent \
  -e DOCUMENT_AGENT_INTEGRATION_MINIO_ENDPOINT=minio:9000 \
  api sh -lc 'python -m pip install -q pytest && python -m pytest tests/integration -q'
```

Set `DOCUMENT_AGENT_RUN_LARGE_FIXTURE=1` to include `test_data/researchpaper1.pdf` when that local fixture is present:

```bash
docker compose run --rm -v "$PWD:/workspace" -w /workspace \
  -e DOCUMENT_AGENT_RUN_INTEGRATION=1 \
  -e DOCUMENT_AGENT_RUN_LARGE_FIXTURE=1 \
  -e DOCUMENT_AGENT_INTEGRATION_DATABASE_URL=postgresql://document_agent:document_agent@postgres:5432/document_agent \
  -e DOCUMENT_AGENT_INTEGRATION_MINIO_ENDPOINT=minio:9000 \
  api sh -lc 'python -m pip install -q pytest && python -m pytest tests/integration/test_api_worker_integration.py::test_large_research_pdf_fixture_processes_to_markdown -q'
```

## Troubleshooting

`curl: (26) Failed to open/read local data from file/application`

- The file path passed to `-F "file=@..."` is wrong from your current shell directory.
- If you are inside `document-agent`, use `./test_data/file.ext`, not `document-agent/test_data/file.ext`.

Job stays `running` at OCR stage:

- Check worker logs with `docker compose logs -f worker`.
- Confirm `OCR_SERVER_URL`, `OCR_API_KEY`, and `OCR_MODEL` are set in `.env`.
- Check worker OCR metrics at `http://localhost:8081/metrics`.

Office conversion fails:

- Rebuild the image so LibreOffice and Pandoc are installed: `docker compose up -d --build api worker`.
- Check worker logs for `OFFICE_CONVERSION_FAILED` or `OFFICE_CONVERTER_NOT_AVAILABLE`.

Host integration tests hit the wrong Postgres:

- Some machines have a local Postgres on `localhost:5432`.
- Run integration tests inside the Compose network as shown above so `postgres:5432` resolves to the container.

Need a clean local environment:

```bash
docker compose down -v
docker compose up -d --build
```
