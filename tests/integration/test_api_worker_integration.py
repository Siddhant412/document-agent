from __future__ import annotations

import os
import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import document_agent.api.app as app_module
import document_agent.api.routes as routes_module
from document_agent.api.sse import events_response
from document_agent.config import Settings, get_settings
from document_agent.db.connection import close_pool, init_db
from document_agent.db.repository import Repository
from document_agent.storage import ObjectStore
from document_agent.worker.runner import Worker


pytestmark = pytest.mark.skipif(
    os.getenv("DOCUMENT_AGENT_RUN_INTEGRATION") != "1",
    reason="Set DOCUMENT_AGENT_RUN_INTEGRATION=1 to run real Postgres/MinIO tests.",
)


@pytest.fixture()
def integration_stack(monkeypatch):
    settings = Settings(
        _env_file=None,
        database_url=os.getenv(
            "DOCUMENT_AGENT_INTEGRATION_DATABASE_URL",
            "postgresql://document_agent:document_agent@localhost:5432/document_agent",
        ),
        minio_endpoint=os.getenv("DOCUMENT_AGENT_INTEGRATION_MINIO_ENDPOINT", "localhost:9000"),
        minio_access_key=os.getenv("DOCUMENT_AGENT_INTEGRATION_MINIO_ACCESS_KEY", "minioadmin"),
        minio_secret_key=os.getenv("DOCUMENT_AGENT_INTEGRATION_MINIO_SECRET_KEY", "minioadmin"),
        minio_secure=False,
        minio_bucket=f"document-agent-it-{uuid4().hex}",
        minio_auto_create_bucket=True,
        public_base_url="http://testserver",
        api_base_url="http://testserver",
        worker_metrics_port=0,
        worker_poll_interval_seconds=0.01,
        worker_lease_seconds=1,
        worker_lease_heartbeat_seconds=60,
        completed_result_retention_seconds=0,
        ocr_server_url=None,
        pdf_use_vendor_extractor=False,
    )
    close_pool()
    get_settings.cache_clear()
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "close_pool", lambda: None)

    init_db(settings)
    repository = Repository(settings)
    object_store = ObjectStore(settings)
    object_store.ensure_bucket()
    app = app_module.create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[routes_module.get_settings] = lambda: settings
    app.dependency_overrides[routes_module.get_repository] = lambda: repository
    app.dependency_overrides[routes_module.get_object_store] = lambda: object_store

    try:
        yield SimpleNamespace(
            app=app,
            settings=settings,
            repository=repository,
            object_store=object_store,
        )
    finally:
        app.dependency_overrides.clear()
        close_pool()
        get_settings.cache_clear()


def test_fastapi_upload_worker_processes_retained_original_with_real_postgres_minio(
    integration_stack,
) -> None:
    with TestClient(integration_stack.app) as client:
        response = client.post(
            "/v1/jobs",
            files={"file": ("notes.txt", b"# Notes\n\nReal integration text.", "text/plain")},
        )
        assert response.status_code == 202
        job_id = UUID(response.json()["job_id"])
        library_item_id = UUID(response.json()["library_item_id"])

        _process_job(integration_stack, job_id=job_id, worker_id="integration-worker")

        status_response = client.get(f"/v1/jobs/{job_id}")
        assert status_response.json()["status"] == "succeeded"

        result_response = client.get(f"/v1/jobs/{job_id}/result", params={"include_markdown": "true"})
        assert result_response.status_code == 200
        assert "Real integration text." in result_response.json()["markdown"]

        job = integration_stack.repository.get_job(job_id)
        assert job["source_deleted_at"] is None

        original_response = client.get(f"/v1/library/{library_item_id}/original")
        assert original_response.status_code == 200
        assert b"Real integration text." in original_response.content


def test_sse_disconnect_does_not_cancel_real_job(integration_stack) -> None:
    with TestClient(integration_stack.app) as client:
        response = client.post(
            "/v1/jobs",
            files={"file": ("disconnect.txt", b"SSE disconnect should not cancel.", "text/plain")},
        )
        assert response.status_code == 202
        job_id = UUID(response.json()["job_id"])

        sse_response = events_response(
            request=_DisconnectAfterFirstPollRequest(),
            repository=integration_stack.repository,
            job_id=job_id,
            poll_seconds=0,
        )
        chunks = asyncio.run(_collect_sse_chunks(sse_response.body_iterator, limit=1))
        assert '"stage": "queued"' in chunks[0]

        _process_job(integration_stack, job_id=job_id, worker_id="sse-worker")

        result_response = client.get(f"/v1/jobs/{job_id}/result", params={"include_markdown": "true"})
        assert result_response.status_code == 200
        assert "SSE disconnect should not cancel." in result_response.json()["markdown"]


def test_batch_sse_emits_batch_and_child_events_with_real_postgres_minio(integration_stack) -> None:
    with TestClient(integration_stack.app) as client:
        response = client.post(
            "/v1/batches",
            files=[
                ("files", ("one.txt", b"One", "text/plain")),
                ("files", ("two.txt", b"Two", "text/plain")),
            ],
        )
        assert response.status_code == 202
        batch_id = UUID(response.json()["batch_id"])

        child_job_ids = [UUID(item["job_id"]) for item in response.json()["child_jobs"]]
        _process_job(integration_stack, job_id=child_job_ids[0], worker_id="batch-worker-1")
        _process_job(integration_stack, job_id=child_job_ids[1], worker_id="batch-worker-2")

        sse_response = events_response(
            request=_DisconnectAfterFirstPollRequest(),
            repository=integration_stack.repository,
            batch_id=batch_id,
            poll_seconds=0,
        )
        event_payloads = asyncio.run(_collect_sse_chunks(sse_response.body_iterator, limit=20))

    assert any('"job_id": null' in payload and '"stage": "succeeded"' in payload for payload in event_payloads)
    assert any('"job_id": "' in payload and '"stage": "convert"' in payload for payload in event_payloads)


def test_expired_lease_allows_another_worker_to_claim_same_job(integration_stack) -> None:
    with TestClient(integration_stack.app) as client:
        response = client.post(
            "/v1/jobs",
            files={"file": ("lease.txt", b"Lease retry text.", "text/plain")},
        )
        assert response.status_code == 202
        job_id = UUID(response.json()["job_id"])

    first_claim = _claim_job_by_id(integration_stack, job_id=job_id, worker_id="worker-one", lease_seconds=1)
    assert UUID(str(first_claim["id"])) == job_id
    assert int(first_claim["attempt_count"]) == 1

    time.sleep(1.2)
    second_claim = _claim_job_by_id(integration_stack, job_id=job_id, worker_id="worker-two", lease_seconds=30)
    assert UUID(str(second_claim["id"])) == job_id
    assert int(second_claim["attempt_count"]) == 2
    assert second_claim["lease_owner"] == "worker-two"

    integration_stack.repository.cancel_job(job_id)
    integration_stack.object_store.delete_object(
        bucket=second_claim["source_bucket"],
        object_key=second_claim["source_object_key"],
    )
    integration_stack.repository.mark_source_deleted(job_id)


def test_completed_result_retention_removes_real_minio_assets(integration_stack) -> None:
    integration_stack.settings.completed_result_retention_seconds = 1
    with TestClient(integration_stack.app) as client:
        response = client.post(
            "/v1/jobs",
            files={"file": ("retention.txt", b"Expiring text.", "text/plain")},
        )
        assert response.status_code == 202
        job_id = UUID(response.json()["job_id"])
        _process_job(integration_stack, job_id=job_id, worker_id="retention-worker")
        result = client.get(f"/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        asset_id = UUID(result.json()["asset_id"])

        with integration_stack.repository.pool.connection() as conn:
            conn.execute(
                "UPDATE document_jobs SET finished_at = now() - interval '5 seconds' WHERE id = %s",
                (str(job_id),),
            )
            conn.execute(
                "UPDATE document_assets SET created_at = now() - interval '5 seconds' WHERE job_id = %s",
                (str(job_id),),
            )
            conn.commit()

        Worker(
            settings=integration_stack.settings,
            repository=integration_stack.repository,
            object_store=integration_stack.object_store,
        )._cleanup_expired_completed_results()

        assert integration_stack.repository.get_asset(asset_id) is None
        expired = client.get(f"/v1/jobs/{job_id}/result")
        assert expired.status_code == 410


@pytest.mark.skipif(
    os.getenv("DOCUMENT_AGENT_RUN_LARGE_FIXTURE") != "1",
    reason="Set DOCUMENT_AGENT_RUN_LARGE_FIXTURE=1 to run local large PDF fixture coverage.",
)
def test_large_research_pdf_fixture_processes_to_markdown(integration_stack) -> None:
    pdf_path = Path(os.getenv("DOCUMENT_AGENT_RESEARCH_PDF_PATH", "test_data/researchpaper1.pdf"))
    if not pdf_path.exists():
        pytest.skip(f"Large PDF fixture not found: {pdf_path}")

    with TestClient(integration_stack.app) as client:
        with pdf_path.open("rb") as handle:
            response = client.post(
                "/v1/jobs",
                files={"file": (pdf_path.name, handle, "application/pdf")},
            )
        assert response.status_code == 202
        job_id = UUID(response.json()["job_id"])
        _process_job(integration_stack, job_id=job_id, worker_id="large-pdf-worker")
        result_response = client.get(f"/v1/jobs/{job_id}/result", params={"include_markdown": "true"})
        assert result_response.status_code == 200
        markdown = result_response.json()["markdown"]
        assert "detected_type: pdf" in markdown
        assert len(markdown) > 500


def _process_job(integration_stack, *, job_id: UUID, worker_id: str) -> None:
    job = _claim_job_by_id(integration_stack, job_id=job_id, worker_id=worker_id, lease_seconds=30)
    Worker(
        settings=integration_stack.settings,
        repository=integration_stack.repository,
        object_store=integration_stack.object_store,
    ).process_job(job)


def _claim_job_by_id(integration_stack, *, job_id: UUID, worker_id: str, lease_seconds: int) -> dict:
    with integration_stack.repository.pool.connection() as conn:
        with conn.transaction():
            row = conn.execute(
                """
                UPDATE document_jobs
                SET status = 'running',
                    stage = 'starting',
                    percent = GREATEST(percent, 1),
                    started_at = COALESCE(started_at, now()),
                    lease_owner = %s,
                    lease_expires_at = now() + (%s || ' seconds')::interval,
                    attempt_count = attempt_count + 1
                WHERE id = %s
                  AND (
                    status = 'queued'
                    OR (
                      status = 'running'
                      AND lease_expires_at < now()
                      AND attempt_count < max_attempts
                    )
                  )
                RETURNING *
                """,
                (worker_id, int(lease_seconds), str(job_id)),
            ).fetchone()
            assert row is not None
            return dict(row)


async def _collect_sse_chunks(iterator, *, limit: int) -> list[str]:
    chunks = []
    async for chunk in iterator:
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        if "data: " in text:
            chunks.append(text)
        if len(chunks) >= limit or any('"job_id": null' in item and '"stage": "succeeded"' in item for item in chunks):
            break
    return chunks


class _DisconnectAfterFirstPollRequest:
    def __init__(self) -> None:
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > 1
