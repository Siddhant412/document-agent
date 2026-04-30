from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response, StreamingResponse

from document_agent.api.schemas import (
    BatchCreatedResponse,
    BatchChildJob,
    BatchResultResponse,
    BatchStatusResponse,
    JobCreatedResponse,
    JobResultResponse,
    JobStatusResponse,
)
from document_agent.api.sse import events_response
from document_agent.api.uploads import StagedUpload, parse_metadata_json, stage_upload, staged_jobs_payload
from document_agent.assets import count_document_assets
from document_agent.batches.manifest import (
    build_batch_archive,
    build_batch_manifest,
    ensure_batch_manifest_asset,
)
from document_agent.config import Settings, get_settings
from document_agent.db.repository import Repository
from document_agent.metrics import BATCHES_CREATED, JOBS_CREATED
from document_agent.status import TERMINAL_BATCH_STATUSES, batch_percent_from_jobs
from document_agent.storage import ObjectStore

router = APIRouter()


def get_repository() -> Repository:
    return Repository()


def get_object_store() -> ObjectStore:
    return ObjectStore()


def _base(settings: Settings) -> str:
    return settings.public_base_url.rstrip("/")


def _job_urls(job_id: UUID, settings: Settings) -> Dict[str, str]:
    base = _base(settings)
    return {
        "status_url": f"{base}/v1/jobs/{job_id}",
        "events_url": f"{base}/v1/jobs/{job_id}/events",
        "result_url": f"{base}/v1/jobs/{job_id}/result",
    }


def _batch_urls(batch_id: UUID, settings: Settings) -> Dict[str, str]:
    base = _base(settings)
    return {
        "status_url": f"{base}/v1/batches/{batch_id}",
        "events_url": f"{base}/v1/batches/{batch_id}/events",
        "result_url": f"{base}/v1/batches/{batch_id}/result",
    }


def _job_status(row: Dict[str, Any], settings: Settings) -> JobStatusResponse:
    job_id = UUID(str(row["id"]))
    return JobStatusResponse(
        job_id=job_id,
        batch_id=UUID(str(row["batch_id"])) if row.get("batch_id") else None,
        input_index=row.get("input_index"),
        status=row["status"],
        filename=row["filename"],
        content_type=row.get("content_type"),
        detected_type=row.get("detected_type"),
        sha256=row["sha256"],
        size_bytes=int(row["size_bytes"]),
        stage=row["stage"],
        percent=int(row["percent"]),
        attempt_count=int(row["attempt_count"]),
        max_attempts=int(row["max_attempts"]),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        cancelled_at=row.get("cancelled_at"),
        error_code=row.get("error_code"),
        error_message=row.get("error_message"),
        result_url=_job_urls(job_id, settings)["result_url"],
        events_url=_job_urls(job_id, settings)["events_url"],
    )


@router.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(
    repository: Repository = Depends(get_repository),
    object_store: ObjectStore = Depends(get_object_store),
) -> Dict[str, str]:
    with repository.pool.connection() as conn:
        conn.execute("SELECT 1").fetchone()
    object_store.ensure_bucket()
    return {"status": "ready"}


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/v1/jobs", response_model=JobCreatedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    file: UploadFile = File(...),
    filename: Optional[str] = Form(default=None),
    content_type: Optional[str] = Form(default=None),
    metadata_json: Optional[str] = Form(default=None),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
    object_store: ObjectStore = Depends(get_object_store),
) -> JobCreatedResponse:
    if idempotency_key:
        existing = await run_in_threadpool(repository.get_job_by_idempotency_key, idempotency_key)
        if existing:
            job_id = UUID(str(existing["id"]))
            return JobCreatedResponse(job_id=job_id, status=existing["status"], **_job_urls(job_id, settings))

    metadata = parse_metadata_json(metadata_json)
    staged = await stage_upload(
        upload=file,
        object_store=object_store,
        settings=settings,
        filename_override=filename,
        content_type_override=content_type,
        metadata=metadata,
    )
    try:
        row = await run_in_threadpool(
            repository.create_job,
            job_id=staged.job_id,
            filename=staged.filename,
            content_type=staged.content_type,
            detected_type=staged.detected_type,
            sha256=staged.sha256,
            size_bytes=staged.size_bytes,
            source_bucket=staged.source_bucket,
            source_object_key=staged.source_object_key,
            idempotency_key=idempotency_key,
            metadata=metadata,
        )
    except Exception:
        await run_in_threadpool(
            object_store.delete_object,
            bucket=staged.source_bucket,
            object_key=staged.source_object_key,
        )
        raise
    job_id = UUID(str(row["id"]))
    JOBS_CREATED.labels(submission_type="single", detected_type=str(row["detected_type"])).inc()
    return JobCreatedResponse(job_id=job_id, status=row["status"], **_job_urls(job_id, settings))


@router.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(
    job_id: UUID,
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
) -> JobStatusResponse:
    row = repository.get_job(job_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return _job_status(row, settings)


@router.get("/v1/jobs/{job_id}/events")
def get_job_events(
    request: Request,
    job_id: UUID,
    after_id: int = Query(default=0, ge=0),
    repository: Repository = Depends(get_repository),
) -> StreamingResponse:
    if not repository.get_job(job_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return events_response(request=request, repository=repository, job_id=job_id, after_id=after_id)


@router.get("/v1/jobs/{job_id}/result", response_model=JobResultResponse)
def get_job_result(
    job_id: UUID,
    include_markdown: bool = False,
    repository: Repository = Depends(get_repository),
    object_store: ObjectStore = Depends(get_object_store),
) -> JobResultResponse:
    row = repository.get_job(job_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    if row["status"] != "succeeded":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job is not complete.")
    asset_id = row.get("result_markdown_asset_id")
    if not asset_id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Missing Markdown asset.")
    asset = repository.get_asset(UUID(str(asset_id)))
    if not asset:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Markdown asset not found.")
    markdown = None
    if include_markdown:
        data = object_store.read_object_bytes(bucket=asset["bucket"], object_key=asset["object_key"])
        markdown = data.decode("utf-8", errors="replace")
    assets = repository.list_job_assets(job_id)
    return JobResultResponse(
        job_id=job_id,
        status=row["status"],
        markdown_url=asset["public_url"],
        asset_id=UUID(str(asset["id"])),
        markdown=markdown,
        metadata={
            "filename": row["filename"],
            "asset_count": count_document_assets(assets),
        },
    )


@router.delete("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def delete_job(
    job_id: UUID,
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
) -> JobStatusResponse:
    row = repository.cancel_job(job_id) or repository.get_job(job_id)
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return _job_status(row, settings)


@router.get("/v1/assets/{asset_id}")
def get_asset(
    asset_id: UUID,
    repository: Repository = Depends(get_repository),
    object_store: ObjectStore = Depends(get_object_store),
) -> StreamingResponse:
    asset = repository.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found.")
    return StreamingResponse(
        object_store.iter_object(bucket=asset["bucket"], object_key=asset["object_key"]),
        media_type=asset["mime_type"],
        headers={"Content-Length": str(asset["size_bytes"])},
    )


@router.post("/v1/batches", response_model=BatchCreatedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_batch(
    files: List[UploadFile] = File(...),
    metadata_json: Optional[str] = Form(default=None),
    batch_name: Optional[str] = Form(default=None),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
    object_store: ObjectStore = Depends(get_object_store),
) -> BatchCreatedResponse:
    if idempotency_key:
        existing = await run_in_threadpool(repository.get_batch_by_idempotency_key, idempotency_key)
        if existing:
            batch_id = UUID(str(existing["id"]))
            jobs = await run_in_threadpool(repository.list_batch_jobs, batch_id)
            return BatchCreatedResponse(
                batch_id=batch_id,
                status=existing["status"],
                child_jobs=[
                    {
                        "job_id": job["id"],
                        "input_index": int(job["input_index"]),
                        "filename": job["filename"],
                        "status": job["status"],
                    }
                    for job in jobs
                ],
                **_batch_urls(batch_id, settings),
            )

    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch exceeds configured limit of {settings.max_batch_files} files.",
        )
    metadata = parse_metadata_json(metadata_json)
    staged: List[StagedUpload] = []
    total_size = 0
    try:
        for index, upload in enumerate(files):
            remaining = settings.max_batch_bytes - total_size
            item = await stage_upload(
                upload=upload,
                object_store=object_store,
                settings=settings,
                input_index=index,
                metadata=metadata,
                max_bytes=min(settings.max_upload_bytes, remaining),
            )
            total_size += item.size_bytes
            if total_size > settings.max_batch_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Batch exceeds configured limit of {settings.max_batch_bytes} bytes.",
                )
            staged.append(item)
        batch_id = uuid4()
        row = await run_in_threadpool(
            repository.create_batch,
            batch_id=batch_id,
            batch_name=batch_name,
            idempotency_key=idempotency_key,
            jobs=staged_jobs_payload(staged),
            metadata=metadata,
        )
    except Exception:
        for item in staged:
            await run_in_threadpool(
                object_store.delete_object,
                bucket=item.source_bucket,
                object_key=item.source_object_key,
            )
        raise

    batch_uuid = UUID(str(row["id"]))
    BATCHES_CREATED.inc()
    for item in staged:
        JOBS_CREATED.labels(submission_type="batch", detected_type=item.detected_type).inc()
    return BatchCreatedResponse(
        batch_id=batch_uuid,
        status=row["status"],
        child_jobs=[
            BatchChildJob(
                job_id=item.job_id,
                input_index=int(item.input_index or 0),
                filename=item.filename,
                status="queued",
            )
            for item in staged
        ],
        **_batch_urls(batch_uuid, settings),
    )


@router.get("/v1/batches/{batch_id}", response_model=BatchStatusResponse)
def get_batch(
    batch_id: UUID,
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
) -> BatchStatusResponse:
    batch = repository.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found.")
    jobs = repository.list_batch_jobs(batch_id)
    return BatchStatusResponse(
        batch_id=batch_id,
        status=batch["status"],
        percent=batch_percent_from_jobs(jobs),
        batch_name=batch.get("batch_name"),
        total_files=int(batch["total_files"]),
        succeeded_count=int(batch["succeeded_count"]),
        failed_count=int(batch["failed_count"]),
        cancelled_count=int(batch["cancelled_count"]),
        created_at=batch["created_at"],
        started_at=batch.get("started_at"),
        finished_at=batch.get("finished_at"),
        cancelled_at=batch.get("cancelled_at"),
        jobs=[_job_status(job, settings) for job in jobs],
        result_url=_batch_urls(batch_id, settings)["result_url"],
        events_url=_batch_urls(batch_id, settings)["events_url"],
    )


@router.get("/v1/batches/{batch_id}/events")
def get_batch_events(
    request: Request,
    batch_id: UUID,
    after_id: int = Query(default=0, ge=0),
    repository: Repository = Depends(get_repository),
) -> StreamingResponse:
    if not repository.get_batch(batch_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found.")
    return events_response(request=request, repository=repository, batch_id=batch_id, after_id=after_id)


@router.get("/v1/batches/{batch_id}/result", response_model=BatchResultResponse)
def get_batch_result(
    batch_id: UUID,
    archive: bool = False,
    repository: Repository = Depends(get_repository),
    object_store: ObjectStore = Depends(get_object_store),
) -> BatchResultResponse:
    batch = repository.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found.")
    if batch["status"] not in TERMINAL_BATCH_STATUSES:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Batch is not complete.")
    jobs = repository.list_batch_jobs(batch_id)
    manifest = build_batch_manifest(batch=batch, jobs=jobs, repository=repository)
    manifest_asset = ensure_batch_manifest_asset(
        batch_id=batch_id,
        manifest=manifest,
        repository=repository,
        object_store=object_store,
    )
    archive_url = None
    if archive:
        archive_asset = build_batch_archive(
            batch_id=batch_id,
            manifest=manifest,
            jobs=jobs,
            repository=repository,
            object_store=object_store,
        )
        archive_url = archive_asset["public_url"]
    return BatchResultResponse(
        batch_id=batch_id,
        status=batch["status"],
        manifest_url=manifest_asset["public_url"],
        archive_url=archive_url,
        files=manifest["files"],
    )


@router.delete("/v1/batches/{batch_id}", response_model=BatchStatusResponse)
def delete_batch(
    batch_id: UUID,
    settings: Settings = Depends(get_settings),
    repository: Repository = Depends(get_repository),
) -> BatchStatusResponse:
    batch = repository.cancel_batch(batch_id) or repository.get_batch(batch_id)
    if not batch:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Batch not found.")
    jobs = repository.list_batch_jobs(batch_id)
    return BatchStatusResponse(
        batch_id=batch_id,
        status=batch["status"],
        percent=batch_percent_from_jobs(jobs),
        batch_name=batch.get("batch_name"),
        total_files=int(batch["total_files"]),
        succeeded_count=int(batch["succeeded_count"]),
        failed_count=int(batch["failed_count"]),
        cancelled_count=int(batch["cancelled_count"]),
        created_at=batch["created_at"],
        started_at=batch.get("started_at"),
        finished_at=batch.get("finished_at"),
        cancelled_at=batch.get("cancelled_at"),
        jobs=[_job_status(job, settings) for job in jobs],
        result_url=_batch_urls(batch_id, settings)["result_url"],
        events_url=_batch_urls(batch_id, settings)["events_url"],
    )
