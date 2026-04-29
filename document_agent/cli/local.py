from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID

from document_agent.storage import ObjectInfo
from document_agent.utils import safe_filename, sha256_file


class LocalRepository:
    def update_progress(
        self,
        *,
        job_id: UUID,
        stage: str,
        percent: int,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None

    def create_asset(
        self,
        *,
        asset_id: Optional[UUID] = None,
        batch_id: Optional[UUID],
        job_id: Optional[UUID],
        role: str,
        bucket: str,
        object_key: str,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        public_url: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "id": asset_id,
            "batch_id": batch_id,
            "job_id": job_id,
            "role": role,
            "bucket": bucket,
            "object_key": object_key,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "public_url": public_url,
            "metadata_json": metadata or {},
        }


class LocalObjectStore:
    def __init__(self, asset_dir: Path) -> None:
        self.asset_dir = asset_dir
        self.bucket = "local"
        self._asset_urls: Dict[str, str] = {}

    def public_url_for_asset(self, asset_id: str) -> str:
        return self._asset_urls.get(asset_id, f"assets/{asset_id}")

    def job_asset_key(self, *, job_id: str, role: str, asset_id: str, filename: str) -> str:
        ext = Path(filename).suffix
        folder = {
            "embedded_image": "images",
            "equation_image": "equations",
            "diagnostic_manifest": "diagnostics",
        }.get(role, safe_filename(role, default="asset"))
        key = f"{folder}/{asset_id}{ext}"
        self._asset_urls[asset_id] = f"{self.asset_dir.name}/{key}"
        return key

    def upload_file(self, *, path: Path, object_key: str, mime_type: Optional[str] = None) -> ObjectInfo:
        destination = self.asset_dir / object_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        digest, size = sha256_file(destination)
        return ObjectInfo(
            bucket=self.bucket,
            object_key=object_key,
            mime_type=mime_type or "application/octet-stream",
            size_bytes=size,
            sha256=digest,
            public_url="",
        )
