from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from document_agent.api.uploads import stage_upload, staged_jobs_payload
from document_agent.converters.pipeline import ConversionPipeline
from document_agent.errors import DocumentAgentError


class FakeUpload:
    def __init__(self, *, filename: str, content_type: str | None, data: bytes) -> None:
        self.filename = filename
        self.content_type = content_type
        self._data = data
        self._offset = 0
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._data):
            return b""
        if size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    async def close(self) -> None:
        self.closed = True


class FakeObjectStore:
    bucket = "test-bucket"

    def original_key(self, *, library_item_id: str, filename: str) -> str:
        return f"library/{library_item_id}/original/{filename}"

    def staging_key(self, *, job_id: str, filename: str) -> str:
        return f"staging/jobs/{job_id}/source/{filename}"

    def upload_file(self, *, path: Path, object_key: str, mime_type: str):
        data = path.read_bytes()
        return SimpleNamespace(
            bucket=self.bucket,
            object_key=object_key,
            mime_type=mime_type,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )


def test_stage_upload_preserves_unsupported_file_as_job_metadata() -> None:
    upload = FakeUpload(
        filename="archive.bin",
        content_type="application/octet-stream",
        data=b"\x00\x01\x02\x03",
    )
    settings = SimpleNamespace(upload_spool_chunk_bytes=2, max_upload_bytes=1024)

    staged = asyncio.run(
        stage_upload(
            upload=upload,  # type: ignore[arg-type]
            object_store=FakeObjectStore(),  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
        )
    )

    assert staged.detected_type == "unsupported"
    assert staged.metadata["detection_error_code"] == "UNSUPPORTED_FILE_TYPE"
    assert staged.source_object_key == f"library/{staged.library_item_id}/original/archive.bin"
    assert upload.closed is True
    assert staged_jobs_payload([staged])[0]["library_item_id"] == staged.library_item_id
    assert staged_jobs_payload([staged])[0]["metadata"]["detection_error_code"] == "UNSUPPORTED_FILE_TYPE"


def test_stage_upload_uses_filename_and_content_type_overrides() -> None:
    upload = FakeUpload(filename="upload.bin", content_type="application/octet-stream", data=b"hello")
    settings = SimpleNamespace(upload_spool_chunk_bytes=8, max_upload_bytes=1024)

    staged = asyncio.run(
        stage_upload(
            upload=upload,  # type: ignore[arg-type]
            object_store=FakeObjectStore(),  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            filename_override="../notes.txt",
            content_type_override="text/plain",
        )
    )

    assert staged.filename == "notes.txt"
    assert staged.content_type == "text/plain"
    assert staged.detected_type == "txt"
    assert staged.source_object_key == f"library/{staged.library_item_id}/original/notes.txt"


def test_stage_upload_rejects_when_batch_remaining_size_is_exhausted() -> None:
    upload = FakeUpload(filename="notes.txt", content_type="text/plain", data=b"hello")
    settings = SimpleNamespace(upload_spool_chunk_bytes=8, max_upload_bytes=1024)

    with pytest.raises(Exception) as exc_info:
        asyncio.run(
            stage_upload(
                upload=upload,  # type: ignore[arg-type]
                object_store=FakeObjectStore(),  # type: ignore[arg-type]
                settings=settings,  # type: ignore[arg-type]
                max_bytes=0,
            )
        )

    assert getattr(exc_info.value, "status_code", None) == 413
    assert upload.closed is True


def test_pipeline_turns_unsupported_detection_metadata_into_non_retryable_failure(tmp_path: Path) -> None:
    pipeline = ConversionPipeline(
        repository=object(),  # type: ignore[arg-type]
        object_store=object(),  # type: ignore[arg-type]
        settings=SimpleNamespace(),  # type: ignore[arg-type]
    )
    source = tmp_path / "unsupported.bin"
    source.write_bytes(b"\x00\x01")

    with pytest.raises(DocumentAgentError) as exc_info:
        pipeline.convert(
            job_id=uuid4(),
            batch_id=None,
            input_index=None,
            source_path=source,
            filename="unsupported.bin",
            detected_type="unsupported",
            content_type="application/octet-stream",
            temp_dir=tmp_path,
            metadata={
                "detection_error_code": "UNSUPPORTED_FILE_TYPE",
                "detection_error_message": "Unsupported file type for 'unsupported.bin'.",
                "detection_error_details": {"suffix": "bin"},
            },
        )

    assert exc_info.value.code == "UNSUPPORTED_FILE_TYPE"
    assert exc_info.value.retryable is False
    assert exc_info.value.details == {"suffix": "bin"}
