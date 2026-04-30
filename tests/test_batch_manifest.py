from __future__ import annotations

from types import SimpleNamespace
from uuid import UUID, uuid4

from document_agent.batches.manifest import build_batch_archive, build_batch_manifest


def test_batch_manifest_preserves_order_and_counts_only_document_assets() -> None:
    batch_id = uuid4()
    success_job_id = uuid4()
    failed_job_id = uuid4()
    markdown_asset_id = uuid4()
    repository = _ManifestRepository(
        batch_id=batch_id,
        assets={
            markdown_asset_id: {
                "id": markdown_asset_id,
                "bucket": "bucket",
                "object_key": "jobs/one/result/document.md",
                "public_url": "http://api/v1/assets/markdown",
                "role": "markdown_result",
            }
        },
        job_assets={
            success_job_id: [
                {"role": "markdown_result"},
                {"role": "diagnostic_manifest"},
                {"role": "embedded_image"},
            ],
            failed_job_id: [{"role": "diagnostic_manifest"}],
        },
    )

    manifest = build_batch_manifest(
        batch={
            "id": batch_id,
            "status": "partial_failed",
            "total_files": 2,
            "succeeded_count": 1,
            "failed_count": 1,
            "cancelled_count": 0,
        },
        jobs=[
            {
                "id": success_job_id,
                "input_index": 0,
                "filename": "same.pdf",
                "status": "succeeded",
                "result_markdown_asset_id": markdown_asset_id,
            },
            {
                "id": failed_job_id,
                "input_index": 1,
                "filename": "same.docx",
                "status": "failed",
                "error_code": "UNSUPPORTED_FILE_TYPE",
                "error_message": "Unsupported.",
            },
        ],
        repository=repository,  # type: ignore[arg-type]
    )

    assert [item["filename"] for item in manifest["files"]] == ["same.pdf", "same.docx"]
    assert manifest["files"][0]["markdown_url"] == "http://api/v1/assets/markdown"
    assert manifest["files"][0]["markdown_filename"] == "same.md"
    assert manifest["files"][0]["asset_count"] == 1
    assert manifest["files"][1]["markdown_filename"] is None
    assert manifest["files"][1]["asset_count"] == 0


def test_batch_archive_contains_manifest_and_successful_markdown_only() -> None:
    batch_id = uuid4()
    success_job_id = uuid4()
    failed_job_id = uuid4()
    markdown_asset_id = uuid4()
    repository = _ArchiveRepository(
        batch_id=batch_id,
        markdown_asset_id=markdown_asset_id,
        markdown=b"# Converted\n",
    )
    object_store = _ArchiveObjectStore(markdown=b"# Converted\n")
    manifest = {
        "batch_id": batch_id,
        "files": [
            {
                "filename": "converted.pdf",
                "job_id": success_job_id,
                "status": "succeeded",
                "markdown_filename": "converted.md",
            },
            {
                "filename": "bad.bin",
                "job_id": failed_job_id,
                "status": "failed",
                "markdown_filename": None,
            },
        ],
    }

    asset = build_batch_archive(
        batch_id=batch_id,
        manifest=manifest,
        jobs=[
            {"id": success_job_id, "result_markdown_asset_id": markdown_asset_id},
            {"id": failed_job_id, "result_markdown_asset_id": None},
        ],
        repository=repository,  # type: ignore[arg-type]
        object_store=object_store,  # type: ignore[arg-type]
    )

    assert asset["role"] == "batch_archive"
    names = object_store.archive_names()
    assert names == ["converted.md", "manifest.json"]
    assert repository.archive_asset_id == asset["id"]


class _ManifestRepository:
    def __init__(self, *, batch_id: UUID, assets: dict[UUID, dict], job_assets: dict[UUID, list[dict]]) -> None:
        self.batch_id = batch_id
        self.assets = assets
        self.job_assets = job_assets

    def get_asset(self, asset_id: UUID):
        return self.assets.get(asset_id)

    def list_job_assets(self, job_id: UUID):
        return self.job_assets.get(job_id, [])


class _ArchiveRepository:
    def __init__(self, *, batch_id: UUID, markdown_asset_id: UUID, markdown: bytes) -> None:
        self.batch_id = batch_id
        self.markdown_asset_id = markdown_asset_id
        self.markdown = markdown
        self.archive_asset_id = None
        self.assets = {
            markdown_asset_id: {
                "id": markdown_asset_id,
                "bucket": "bucket",
                "object_key": "jobs/one/result/document.md",
                "role": "markdown_result",
            }
        }

    def get_batch(self, batch_id: UUID):
        return {"id": batch_id, "result_archive_asset_id": self.archive_asset_id}

    def get_asset(self, asset_id: UUID):
        return self.assets.get(asset_id)

    def get_asset_by_object_key(self, *, bucket: str, object_key: str):
        return None

    def create_asset(self, **kwargs):
        row = {"id": kwargs["asset_id"], **kwargs}
        self.assets[row["id"]] = row
        return row

    def update_batch_archive_asset(self, *, batch_id: UUID, asset_id: UUID) -> None:
        self.archive_asset_id = asset_id


class _ArchiveObjectStore:
    bucket = "bucket"

    def __init__(self, *, markdown: bytes) -> None:
        self.markdown = markdown
        self.uploaded = b""

    def batch_archive_key(self, *, batch_id: str) -> str:
        return f"batches/{batch_id}/archive/results.zip"

    def read_object_bytes(self, *, bucket: str, object_key: str) -> bytes:
        return self.markdown

    def upload_bytes(self, *, data: bytes, object_key: str, mime_type: str):
        self.uploaded = data
        return SimpleNamespace(
            bucket=self.bucket,
            object_key=object_key,
            mime_type=mime_type,
            size_bytes=len(data),
            sha256="sha",
        )

    def public_url_for_asset(self, asset_id: str) -> str:
        return f"http://api/v1/assets/{asset_id}"

    def archive_names(self) -> list[str]:
        from io import BytesIO
        from zipfile import ZipFile

        with ZipFile(BytesIO(self.uploaded), "r") as zip_file:
            return sorted(zip_file.namelist())
