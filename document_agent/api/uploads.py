from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from document_agent.config import Settings
from document_agent.converters.detect import detect_file_type
from document_agent.errors import DocumentAgentError
from document_agent.storage import ObjectStore
from document_agent.utils import safe_filename


@dataclass
class StagedUpload:
    job_id: UUID
    input_index: Optional[int]
    filename: str
    content_type: Optional[str]
    detected_type: str
    sha256: str
    size_bytes: int
    source_bucket: str
    source_object_key: str
    metadata: Dict[str, Any]


def parse_metadata_json(metadata_json: Optional[str]) -> Dict[str, Any]:
    if not metadata_json:
        return {}
    try:
        value = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"metadata_json is not valid JSON: {exc.msg}",
        ) from exc
    if not isinstance(value, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata_json must decode to an object.",
        )
    return value


async def stage_upload(
    *,
    upload: UploadFile,
    object_store: ObjectStore,
    settings: Settings,
    input_index: Optional[int] = None,
    filename_override: Optional[str] = None,
    content_type_override: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_bytes: Optional[int] = None,
) -> StagedUpload:
    filename = safe_filename(
        filename_override or upload.filename or f"upload-{input_index or 0}",
        default="document",
    )
    job_id = uuid4()
    content_type = content_type_override or upload.content_type
    max_allowed = max_bytes if max_bytes is not None else settings.max_upload_bytes
    if max_allowed <= 0:
        await upload.close()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Upload exceeds configured batch size limit.",
        )
    digest = hashlib.sha256()
    size = 0
    tmp_path: Optional[Path] = None
    item_metadata = dict(metadata or {})
    try:
        with tempfile.NamedTemporaryFile(prefix="document-agent-upload-", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await upload.read(settings.upload_spool_chunk_bytes)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_allowed:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Upload exceeds configured limit of {max_allowed} bytes.",
                    )
                digest.update(chunk)
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
        try:
            detected_type = detect_file_type(tmp_path, filename, content_type)
        except DocumentAgentError as exc:
            detected_type = "unsupported"
            item_metadata.update(
                {
                    "detection_error_code": exc.code,
                    "detection_error_message": exc.message,
                    "detection_error_details": exc.details or {},
                }
            )

        object_key = object_store.staging_key(job_id=str(job_id), filename=filename)
        info = await run_in_threadpool(
            object_store.upload_file,
            path=tmp_path,
            object_key=object_key,
            mime_type=content_type or "application/octet-stream",
        )
        return StagedUpload(
            job_id=job_id,
            input_index=input_index,
            filename=filename,
            content_type=content_type,
            detected_type=detected_type,
            sha256=digest.hexdigest() or info.sha256,
            size_bytes=size,
            source_bucket=info.bucket,
            source_object_key=info.object_key,
            metadata=item_metadata,
        )
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
        await upload.close()


def staged_jobs_payload(items: List[StagedUpload]) -> List[Dict[str, Any]]:
    return [
        {
            "job_id": item.job_id,
            "input_index": item.input_index,
            "filename": item.filename,
            "content_type": item.content_type,
            "detected_type": item.detected_type,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
            "source_bucket": item.source_bucket,
            "source_object_key": item.source_object_key,
            "metadata": item.metadata,
        }
        for item in items
    ]
