from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from fastapi import HTTPException

from document_agent.api.routes import get_job_result
from document_agent.worker.runner import Worker


def test_completed_result_cleanup_deletes_objects_and_forgets_assets() -> None:
    asset_id = uuid4()
    repository = _RetentionRepository(
        [
            {
                "id": asset_id,
                "bucket": "bucket",
                "object_key": "jobs/job-1/result/document.md",
            }
        ]
    )
    object_store = _RetentionObjectStore()
    worker = Worker(
        settings=SimpleNamespace(completed_result_retention_seconds=60, worker_id=None),
        repository=repository,  # type: ignore[arg-type]
        object_store=object_store,  # type: ignore[arg-type]
    )

    worker._cleanup_expired_completed_results()

    assert object_store.deleted == [("bucket", "jobs/job-1/result/document.md")]
    assert repository.forgotten == [asset_id]


def test_job_result_returns_gone_after_retention_removes_markdown() -> None:
    repository = _ExpiredResultRepository()

    try:
        get_job_result(
            job_id=repository.job_id,
            include_markdown=False,
            repository=repository,  # type: ignore[arg-type]
            object_store=object(),  # type: ignore[arg-type]
        )
    except HTTPException as exc:
        assert exc.status_code == 410
        assert exc.detail == "Markdown result expired."
    else:
        raise AssertionError("Expected HTTPException")


class _RetentionRepository:
    def __init__(self, assets: list[dict]) -> None:
        self.assets = assets
        self.forgotten = []

    def list_expired_completed_assets(self, *, retention_seconds: int, limit: int):
        assert retention_seconds == 60
        assert limit == 500
        return self.assets

    def forget_assets(self, asset_ids):
        self.forgotten.extend(asset_ids)
        return len(asset_ids)


class _RetentionObjectStore:
    def __init__(self) -> None:
        self.deleted = []

    def delete_object(self, *, bucket: str, object_key: str) -> None:
        self.deleted.append((bucket, object_key))


class _ExpiredResultRepository:
    def __init__(self) -> None:
        self.job_id = uuid4()

    def get_job(self, job_id):
        assert job_id == self.job_id
        return {
            "id": self.job_id,
            "status": "succeeded",
            "result_markdown_asset_id": None,
        }
