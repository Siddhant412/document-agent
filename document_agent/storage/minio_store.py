from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

from minio import Minio

from document_agent.config import Settings, get_settings
from document_agent.utils import safe_filename, sha256_file


@dataclass
class ObjectInfo:
    bucket: str
    object_key: str
    mime_type: str
    size_bytes: int
    sha256: str
    public_url: str


class ObjectStore:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.bucket = self.settings.minio_bucket
        self.client = Minio(
            self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            secure=self.settings.minio_secure,
            region=self.settings.minio_region,
        )

    def ensure_bucket(self) -> None:
        if self.client.bucket_exists(self.bucket):
            return
        if not self.settings.minio_auto_create_bucket:
            raise RuntimeError(f"MinIO bucket does not exist: {self.bucket}")
        self.client.make_bucket(self.bucket)

    def public_url_for_asset(self, asset_id: str) -> str:
        return f"{self.settings.public_base_url.rstrip('/')}/v1/assets/{asset_id}"

    def staging_key(self, *, job_id: str, filename: str) -> str:
        return f"staging/jobs/{job_id}/source/{safe_filename(filename)}"

    def markdown_key(self, *, job_id: str) -> str:
        return f"jobs/{job_id}/result/document.md"

    def job_asset_key(self, *, job_id: str, role: str, asset_id: str, filename: str) -> str:
        ext = Path(filename).suffix
        folder = {
            "embedded_image": "images",
            "equation_image": "equations",
            "diagnostic_manifest": "diagnostics",
        }.get(role, safe_filename(role, default="asset"))
        safe_ext = ext if ext else ""
        return f"jobs/{job_id}/assets/{folder}/{asset_id}{safe_ext}"

    def conversion_manifest_key(self, *, job_id: str) -> str:
        return f"jobs/{job_id}/manifests/conversion.json"

    def batch_manifest_key(self, *, batch_id: str) -> str:
        return f"batches/{batch_id}/manifest.json"

    def batch_archive_key(self, *, batch_id: str) -> str:
        return f"batches/{batch_id}/archive/results.zip"

    def upload_file(
        self,
        *,
        path: Path,
        object_key: str,
        mime_type: Optional[str] = None,
    ) -> ObjectInfo:
        self.ensure_bucket()
        resolved = path.expanduser().resolve()
        sha256, size_bytes = sha256_file(resolved)
        resolved_mime = mime_type or mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        with resolved.open("rb") as handle:
            self.client.put_object(
                self.bucket,
                object_key,
                handle,
                length=size_bytes,
                content_type=resolved_mime,
            )
        return ObjectInfo(
            bucket=self.bucket,
            object_key=object_key,
            mime_type=resolved_mime,
            size_bytes=size_bytes,
            sha256=sha256,
            public_url="",
        )

    def upload_bytes(
        self,
        *,
        data: bytes,
        object_key: str,
        mime_type: str,
    ) -> ObjectInfo:
        from io import BytesIO

        self.ensure_bucket()
        digest = hashlib.sha256(data).hexdigest()
        self.client.put_object(
            self.bucket,
            object_key,
            BytesIO(data),
            length=len(data),
            content_type=mime_type,
        )
        return ObjectInfo(
            bucket=self.bucket,
            object_key=object_key,
            mime_type=mime_type,
            size_bytes=len(data),
            sha256=digest,
            public_url="",
        )

    def download_to_path(self, *, bucket: str, object_key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        response = self.client.get_object(bucket, object_key)
        try:
            with path.open("wb") as handle:
                for chunk in response.stream(1024 * 1024):
                    handle.write(chunk)
        finally:
            response.close()
            response.release_conn()

    def delete_object(self, *, bucket: str, object_key: str) -> None:
        self.client.remove_object(bucket, object_key)

    def iter_staging_object_keys_older_than(self, *, ttl_seconds: int) -> Iterator[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(0, int(ttl_seconds)))
        for item in self.client.list_objects(self.bucket, prefix="staging/jobs/", recursive=True):
            last_modified = item.last_modified
            if last_modified is None:
                continue
            if last_modified.tzinfo is None:
                last_modified = last_modified.replace(tzinfo=timezone.utc)
            if last_modified < cutoff and item.object_name:
                yield item.object_name

    def iter_object(self, *, bucket: str, object_key: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
        response = self.client.get_object(bucket, object_key)
        try:
            for chunk in response.stream(chunk_size):
                yield chunk
        finally:
            response.close()
            response.release_conn()

    def read_object_bytes(self, *, bucket: str, object_key: str) -> bytes:
        response = self.client.get_object(bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()
