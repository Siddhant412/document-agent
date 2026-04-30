# document-agent

Containerized document processing service that converts supported files into AI-readable Markdown.

The implementation source of truth is [docs/document-processing-service-plan.md](docs/document-processing-service-plan.md).

## Local Services

```bash
cp .env.example .env
docker compose up --build
```

`docker-compose.yml` loads `.env.example` first and `.env` second, so local secrets and provider overrides in `.env` take precedence without being committed.

Required OCR settings in `.env` for real image/PDF OCR:

```dotenv
OCR_SERVER_URL=https://your-openai-compatible-provider/v1
OCR_API_KEY=your-provider-api-key
OCR_MODEL=your-ocr-model-id
```

`OCR_SERVER_URL` can be either the provider base URL or the full `/chat/completions` URL. The service sends OCR requests through an OpenAI-compatible chat completions payload with image input.

API:

- `POST /v1/jobs` for one file.
- `POST /v1/batches` for multiple files.
- `GET /v1/jobs/{job_id}/events` and `GET /v1/batches/{batch_id}/events` for SSE.
- `GET /v1/jobs/{job_id}/result` and `GET /v1/batches/{batch_id}/result` for outputs.
- `GET /metrics` on port `8080` for API Prometheus metrics.
- `GET /metrics` on port `8081` for worker Prometheus metrics, including OCR and conversion latency.

CLI:

```bash
document-agent submit ./sample.pdf
document-agent batch ./a.pdf ./b.png --output-dir ./markdown_outputs
document-agent result JOB_ID --output out.md
document-agent convert ./sample.txt --output out.md
```

## End-to-End Checks

Start the stack:

```bash
docker compose up -d --build
curl -fsS http://localhost:8080/readyz
```

Submit one file:

```bash
FILE="./test_data/test.docx"
response=$(curl -fsS -F "file=@${FILE}" http://localhost:8080/v1/jobs)
job_id=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["job_id"])' "$response")
curl -fsS "http://localhost:8080/v1/jobs/${job_id}"
curl -fsS "http://localhost:8080/v1/jobs/${job_id}/result?include_markdown=true"
```

Submit a mixed batch:

```bash
curl -fsS \
  -F "files=@./test_data/test.docx" \
  -F "files=@./test_data/test.pdf" \
  http://localhost:8080/v1/batches
```

Use absolute paths if your shell is already inside `document-agent`, for example `FILE="./test_data/test.docx"` instead of `FILE="document-agent/test_data/test.docx"`.
