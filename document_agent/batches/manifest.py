from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict, List
from uuid import UUID, uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from document_agent.assets import count_document_assets
from document_agent.db.repository import Repository
from document_agent.storage import ObjectStore
from document_agent.utils import unique_names


def build_batch_manifest(
    *,
    batch: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    repository: Repository,
) -> Dict[str, Any]:
    markdown_names = unique_names([str(job["filename"]) for job in jobs])
    files = []
    for index, job in enumerate(jobs):
        markdown_url = None
        if job["status"] == "succeeded" and job.get("result_markdown_asset_id"):
            asset = repository.get_asset(UUID(str(job["result_markdown_asset_id"])))
            markdown_url = asset["public_url"] if asset else None
        assets = repository.list_job_assets(UUID(str(job["id"])))
        files.append(
            {
                "input_index": job.get("input_index"),
                "filename": job["filename"],
                "library_item_id": job.get("library_item_id"),
                "job_id": job["id"],
                "status": job["status"],
                "markdown_url": markdown_url,
                "markdown_filename": markdown_names[index] if job["status"] == "succeeded" else None,
                "asset_count": count_document_assets(assets),
                "error_code": job.get("error_code"),
                "error_message": job.get("error_message"),
            }
        )
    return {
        "batch_id": batch["id"],
        "created_at": batch.get("created_at"),
        "finished_at": batch.get("finished_at"),
        "status": batch["status"],
        "total_files": int(batch["total_files"]),
        "succeeded_count": int(batch["succeeded_count"]),
        "failed_count": int(batch["failed_count"]),
        "cancelled_count": int(batch["cancelled_count"]),
        "files": files,
    }


def ensure_batch_manifest_asset(
    *,
    batch_id: UUID,
    manifest: Dict[str, Any],
    repository: Repository,
    object_store: ObjectStore,
) -> Dict[str, Any]:
    object_key = object_store.batch_manifest_key(batch_id=str(batch_id))
    existing = repository.get_asset_by_object_key(bucket=object_store.bucket, object_key=object_key)
    if existing:
        return existing
    asset_id = uuid4()
    data = json.dumps(manifest, default=str, indent=2, ensure_ascii=False).encode("utf-8")
    info = object_store.upload_bytes(data=data, object_key=object_key, mime_type="application/json")
    return repository.create_asset(
        asset_id=asset_id,
        batch_id=batch_id,
        job_id=None,
        role="batch_manifest",
        bucket=info.bucket,
        object_key=info.object_key,
        mime_type=info.mime_type,
        size_bytes=info.size_bytes,
        sha256=info.sha256,
        public_url=object_store.public_url_for_asset(str(asset_id)),
        metadata={"kind": "batch_manifest"},
    )


def build_batch_archive(
    *,
    batch_id: UUID,
    manifest: Dict[str, Any],
    jobs: List[Dict[str, Any]],
    repository: Repository,
    object_store: ObjectStore,
) -> Dict[str, Any]:
    batch = repository.get_batch(batch_id)
    if batch and batch.get("result_archive_asset_id"):
        existing = repository.get_asset(UUID(str(batch["result_archive_asset_id"])))
        if existing:
            return existing

    job_by_id = {str(job["id"]): job for job in jobs}
    archive = BytesIO()
    with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
        zip_file.writestr(
            "manifest.json",
            json.dumps(manifest, default=str, indent=2, ensure_ascii=False),
        )
        for file_item in manifest["files"]:
            if file_item["status"] != "succeeded" or not file_item.get("markdown_filename"):
                continue
            job = job_by_id[str(file_item["job_id"])]
            asset_id = job.get("result_markdown_asset_id")
            if not asset_id:
                continue
            asset = repository.get_asset(UUID(str(asset_id)))
            if not asset:
                continue
            data = object_store.read_object_bytes(bucket=asset["bucket"], object_key=asset["object_key"])
            zip_file.writestr(file_item["markdown_filename"], data)

    asset_id = uuid4()
    object_key = object_store.batch_archive_key(batch_id=str(batch_id))
    existing = repository.get_asset_by_object_key(bucket=object_store.bucket, object_key=object_key)
    if existing:
        repository.update_batch_archive_asset(batch_id=batch_id, asset_id=UUID(str(existing["id"])))
        return existing
    info = object_store.upload_bytes(
        data=archive.getvalue(),
        object_key=object_key,
        mime_type="application/zip",
    )
    row = repository.create_asset(
        asset_id=asset_id,
        batch_id=batch_id,
        job_id=None,
        role="batch_archive",
        bucket=info.bucket,
        object_key=info.object_key,
        mime_type=info.mime_type,
        size_bytes=info.size_bytes,
        sha256=info.sha256,
        public_url=object_store.public_url_for_asset(str(asset_id)),
        metadata={"kind": "batch_archive"},
    )
    repository.update_batch_archive_asset(batch_id=batch_id, asset_id=UUID(str(row["id"])))
    return row
