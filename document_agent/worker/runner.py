from __future__ import annotations

import json
import logging
import signal
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from document_agent.config import Settings, get_settings
from document_agent.converters.pipeline import ConversionPipeline
from document_agent.db.connection import init_db
from document_agent.db.repository import Repository
from document_agent.errors import error_from_exception
from document_agent.logging_config import configure_logging
from document_agent.metrics import CONVERSION_DURATION_SECONDS, JOBS_COMPLETED
from document_agent.status import TERMINAL_JOB_STATUSES
from document_agent.storage import ObjectStore
from document_agent.utils import safe_filename

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        repository: Optional[Repository] = None,
        object_store: Optional[ObjectStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = repository or Repository(self.settings)
        self.object_store = object_store or ObjectStore(self.settings)
        self.worker_id = self.settings.worker_id or f"worker-{uuid4()}"
        self.pipeline = ConversionPipeline(
            repository=self.repository,
            object_store=self.object_store,
            settings=self.settings,
        )
        self._stop = threading.Event()
        self._last_cleanup = 0.0

    def run_forever(self) -> None:
        configure_logging(self.settings.log_level)
        init_db(self.settings)
        self.object_store.ensure_bucket()
        logger.info("worker_start worker_id=%s", self.worker_id)
        while not self._stop.is_set():
            self._maintenance_if_due()
            job = self.repository.claim_next_job(
                worker_id=self.worker_id,
                lease_seconds=self.settings.worker_lease_seconds,
            )
            if not job:
                self._stop.wait(self.settings.worker_poll_interval_seconds)
                continue
            self.process_job(job)
        logger.info("worker_stop worker_id=%s", self.worker_id)

    def stop(self) -> None:
        self._stop.set()

    def process_job(self, job: Dict[str, Any]) -> None:
        job_id = UUID(str(job["id"]))
        started_monotonic = time.monotonic()
        detected_type = str(job.get("detected_type") or "unknown")
        temp_dir = Path(tempfile.mkdtemp(prefix=f"document-agent-{job_id}-"))
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(job_id, heartbeat_stop),
            daemon=True,
        )
        heartbeat.start()
        terminal = False
        try:
            source_path = temp_dir / safe_filename(str(job["filename"]), default="source")
            self.repository.update_progress(
                job_id=job_id,
                stage="download_source",
                percent=5,
                message="Downloading staged source object.",
            )
            self.object_store.download_to_path(
                bucket=job["source_bucket"],
                object_key=job["source_object_key"],
                path=source_path,
            )
            self.repository.update_progress(
                job_id=job_id,
                stage="convert",
                percent=15,
                message="Starting document conversion.",
            )
            result = self.pipeline.convert(
                job_id=job_id,
                batch_id=UUID(str(job["batch_id"])) if job.get("batch_id") else None,
                input_index=job.get("input_index"),
                source_path=source_path,
                filename=job["filename"],
                detected_type=job["detected_type"],
                content_type=job.get("content_type"),
                temp_dir=temp_dir,
                metadata=job.get("metadata_json") or {},
            )
            markdown_asset_id = self._persist_markdown(job=job, markdown=result.markdown)
            self._persist_conversion_manifest(job=job, result_metadata=result.metadata)
            terminal = self.repository.mark_job_succeeded(
                job_id=job_id,
                markdown_asset_id=markdown_asset_id,
            )
            if terminal:
                JOBS_COMPLETED.labels(
                    status="succeeded",
                    detected_type=detected_type,
                    error_code="",
                ).inc()
                CONVERSION_DURATION_SECONDS.labels(
                    status="succeeded",
                    detected_type=detected_type,
                ).observe(time.monotonic() - started_monotonic)
            if not terminal:
                current = self.repository.get_job(job_id)
                terminal = bool(current and current["status"] in TERMINAL_JOB_STATUSES)
        except Exception as exc:
            error = error_from_exception(exc)
            logger.exception("job_failed job_id=%s code=%s", job_id, error.code)
            self.repository.mark_job_failed(
                job_id=job_id,
                code=error.code,
                message=error.message,
                details=error.details,
                retryable=error.retryable,
            )
            current = self.repository.get_job(job_id)
            terminal = bool(current and current["status"] in TERMINAL_JOB_STATUSES)
            if terminal:
                JOBS_COMPLETED.labels(
                    status=str(current["status"]),
                    detected_type=detected_type,
                    error_code=error.code,
                ).inc()
                CONVERSION_DURATION_SECONDS.labels(
                    status=str(current["status"]),
                    detected_type=detected_type,
                ).observe(time.monotonic() - started_monotonic)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=2)
            if terminal:
                self._delete_staged_source(job)
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _heartbeat(self, job_id: UUID, stop: threading.Event) -> None:
        interval = max(1.0, float(self.settings.worker_lease_heartbeat_seconds))
        while not stop.wait(interval):
            ok = self.repository.extend_lease(
                job_id=job_id,
                worker_id=self.worker_id,
                lease_seconds=self.settings.worker_lease_seconds,
            )
            if not ok:
                logger.warning("lease_heartbeat_failed job_id=%s worker_id=%s", job_id, self.worker_id)
                return

    def _persist_markdown(self, *, job: Dict[str, Any], markdown: str) -> UUID:
        job_id = UUID(str(job["id"]))
        self.repository.update_progress(
            job_id=job_id,
            stage="upload_markdown",
            percent=90,
            message="Uploading Markdown result.",
        )
        asset_id = uuid4()
        object_key = self.object_store.markdown_key(job_id=str(job_id))
        existing = self.repository.get_asset_by_object_key(
            bucket=self.object_store.bucket,
            object_key=object_key,
        )
        if existing:
            return UUID(str(existing["id"]))
        info = self.object_store.upload_bytes(
            data=markdown.encode("utf-8"),
            object_key=object_key,
            mime_type="text/markdown; charset=utf-8",
        )
        row = self.repository.create_asset(
            asset_id=asset_id,
            batch_id=UUID(str(job["batch_id"])) if job.get("batch_id") else None,
            job_id=job_id,
            role="markdown_result",
            bucket=info.bucket,
            object_key=info.object_key,
            mime_type=info.mime_type,
            size_bytes=info.size_bytes,
            sha256=info.sha256,
            public_url=self.object_store.public_url_for_asset(str(asset_id)),
            metadata={"filename": job["filename"]},
        )
        return UUID(str(row["id"]))

    def _persist_conversion_manifest(self, *, job: Dict[str, Any], result_metadata: Dict[str, Any]) -> None:
        job_id = UUID(str(job["id"]))
        payload = {
            "job_id": str(job_id),
            "batch_id": str(job["batch_id"]) if job.get("batch_id") else None,
            "filename": job["filename"],
            "detected_type": job.get("detected_type"),
            "metadata": result_metadata,
        }
        asset_id = uuid4()
        object_key = self.object_store.conversion_manifest_key(job_id=str(job_id))
        if self.repository.get_asset_by_object_key(bucket=self.object_store.bucket, object_key=object_key):
            return
        info = self.object_store.upload_bytes(
            data=json.dumps(payload, default=str, indent=2, ensure_ascii=False).encode("utf-8"),
            object_key=object_key,
            mime_type="application/json",
        )
        self.repository.create_asset(
            asset_id=asset_id,
            batch_id=UUID(str(job["batch_id"])) if job.get("batch_id") else None,
            job_id=job_id,
            role="diagnostic_manifest",
            bucket=info.bucket,
            object_key=info.object_key,
            mime_type=info.mime_type,
            size_bytes=info.size_bytes,
            sha256=info.sha256,
            public_url=self.object_store.public_url_for_asset(str(asset_id)),
            metadata={"kind": "conversion_manifest"},
        )

    def _delete_staged_source(self, job: Dict[str, Any]) -> None:
        job_id = UUID(str(job["id"]))
        if job.get("source_deleted_at"):
            return
        try:
            self.object_store.delete_object(
                bucket=job["source_bucket"],
                object_key=job["source_object_key"],
            )
            self.repository.mark_source_deleted(job_id)
        except Exception:
            logger.exception("staging_delete_failed job_id=%s", job_id)

    def _maintenance_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup < 60:
            return
        self._last_cleanup = now
        for job in self.repository.fail_expired_running_jobs(
            timeout_seconds=self.settings.job_timeout_seconds,
            limit=100,
        ):
            self._delete_staged_source(job)
        self.repository.cancel_expired_batches(
            timeout_seconds=self.settings.batch_timeout_seconds,
            limit=50,
        )
        for job in self.repository.list_terminal_jobs_with_sources(limit=100):
            self._delete_staged_source(job)
        self._cleanup_orphaned_staging_objects()

    def _cleanup_orphaned_staging_objects(self) -> None:
        try:
            object_keys = self.object_store.iter_staging_object_keys_older_than(
                ttl_seconds=self.settings.staging_ttl_seconds,
            )
            for object_key in object_keys:
                if self.repository.has_undeleted_source_object(
                    bucket=self.object_store.bucket,
                    object_key=object_key,
                ):
                    continue
                self.object_store.delete_object(bucket=self.object_store.bucket, object_key=object_key)
        except Exception:
            logger.exception("staging_orphan_cleanup_failed")


def run_worker() -> None:
    worker = Worker()
    signal.signal(signal.SIGTERM, lambda *_: worker.stop())
    signal.signal(signal.SIGINT, lambda *_: worker.stop())
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        worker.stop()
        time.sleep(0.2)
