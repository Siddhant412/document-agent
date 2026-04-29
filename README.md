# document-agent

Containerized document processing service that converts supported files into AI-readable Markdown.

The implementation source of truth is [docs/document-processing-service-plan.md](docs/document-processing-service-plan.md).

## Local Services

```bash
cp .env.example .env
docker compose up --build
```

API:

- `POST /v1/jobs` for one file.
- `POST /v1/batches` for multiple files.
- `GET /v1/jobs/{job_id}/events` and `GET /v1/batches/{batch_id}/events` for SSE.
- `GET /v1/jobs/{job_id}/result` and `GET /v1/batches/{batch_id}/result` for outputs.

CLI:

```bash
document-agent submit ./sample.pdf
document-agent batch ./a.pdf ./b.png --output-dir ./markdown_outputs
document-agent result JOB_ID --output out.md
document-agent convert ./sample.txt --output out.md
```
