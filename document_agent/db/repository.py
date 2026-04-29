from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from document_agent.config import Settings, get_settings
from document_agent.db.connection import get_pool
from document_agent.metrics import ASSETS_UPLOADED
from document_agent.status import batch_status_from_counts


def _json_payload(value: Optional[Dict[str, Any]]) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


class Repository:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.pool = get_pool(self.settings)

    def get_job_by_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM document_jobs
                WHERE idempotency_key = %s AND batch_id IS NULL
                LIMIT 1
                """,
                (idempotency_key,),
            ).fetchone()
            return dict(row) if row else None

    def get_batch_by_idempotency_key(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM document_batches WHERE idempotency_key = %s LIMIT 1",
                (idempotency_key,),
            ).fetchone()
            return dict(row) if row else None

    def create_job(
        self,
        *,
        job_id: UUID,
        filename: str,
        content_type: Optional[str],
        detected_type: str,
        sha256: str,
        size_bytes: int,
        source_bucket: str,
        source_object_key: str,
        idempotency_key: Optional[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self.pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    INSERT INTO document_jobs(
                      id, idempotency_key, filename, content_type, detected_type, sha256,
                      size_bytes, source_bucket, source_object_key, max_attempts, metadata_json
                    ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    RETURNING *
                    """,
                    (
                        str(job_id),
                        idempotency_key,
                        filename,
                        content_type,
                        detected_type,
                        sha256,
                        int(size_bytes),
                        source_bucket,
                        source_object_key,
                        self.settings.job_max_attempts,
                        _json_payload(metadata),
                    ),
                ).fetchone()
                self.add_event_in_conn(
                    conn,
                    batch_id=None,
                    job_id=job_id,
                    event_type="queued",
                    stage="queued",
                    percent=0,
                    message="Job queued.",
                )
                return dict(row)

    def create_batch(
        self,
        *,
        batch_id: UUID,
        batch_name: Optional[str],
        idempotency_key: Optional[str],
        jobs: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self.pool.connection() as conn:
            with conn.transaction():
                batch = conn.execute(
                    """
                    INSERT INTO document_batches(
                      id, batch_name, idempotency_key, total_files, metadata_json
                    ) VALUES(%s,%s,%s,%s,%s::jsonb)
                    RETURNING *
                    """,
                    (
                        str(batch_id),
                        batch_name,
                        idempotency_key,
                        len(jobs),
                        _json_payload(metadata),
                    ),
                ).fetchone()
                for item in jobs:
                    conn.execute(
                        """
                        INSERT INTO document_jobs(
                          id, batch_id, input_index, filename, content_type, detected_type,
                          sha256, size_bytes, source_bucket, source_object_key, max_attempts,
                          metadata_json
                        ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        """,
                        (
                            str(item["job_id"]),
                            str(batch_id),
                            int(item["input_index"]),
                            item["filename"],
                            item.get("content_type"),
                            item["detected_type"],
                            item["sha256"],
                            int(item["size_bytes"]),
                            item["source_bucket"],
                            item["source_object_key"],
                            self.settings.job_max_attempts,
                            _json_payload(item.get("metadata")),
                        ),
                    )
                    self.add_event_in_conn(
                        conn,
                        batch_id=batch_id,
                        job_id=item["job_id"],
                        event_type="queued",
                        stage="queued",
                        percent=0,
                        message="Batch child job queued.",
                        payload={"input_index": item["input_index"], "filename": item["filename"]},
                    )
                self.add_event_in_conn(
                    conn,
                    batch_id=batch_id,
                    job_id=None,
                    event_type="queued",
                    stage="queued",
                    percent=0,
                    message="Batch queued.",
                    payload={"total_files": len(jobs)},
                )
                return dict(batch)

    def get_job(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM document_jobs WHERE id = %s", (str(job_id),)).fetchone()
            return dict(row) if row else None

    def get_batch(self, batch_id: UUID) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM document_batches WHERE id = %s", (str(batch_id),)).fetchone()
            return dict(row) if row else None

    def list_batch_jobs(self, batch_id: UUID) -> List[Dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM document_jobs WHERE batch_id = %s ORDER BY input_index ASC, created_at ASC",
                (str(batch_id),),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_terminal_jobs_with_sources(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM document_jobs
                WHERE status IN ('succeeded', 'failed', 'cancelled')
                  AND source_deleted_at IS NULL
                ORDER BY finished_at ASC NULLS LAST, cancelled_at ASC NULLS LAST
                LIMIT %s
                """,
                (int(limit),),
            ).fetchall()
            return [dict(row) for row in rows]

    def has_undeleted_source_object(self, *, bucket: str, object_key: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM document_jobs
                WHERE source_bucket = %s
                  AND source_object_key = %s
                  AND source_deleted_at IS NULL
                LIMIT 1
                """,
                (bucket, object_key),
            ).fetchone()
            return bool(row)

    def claim_next_job(self, *, worker_id: str, lease_seconds: int) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    WITH candidate AS (
                      SELECT id
                      FROM document_jobs
                      WHERE
                        status = 'queued'
                        OR (
                          status = 'running'
                          AND lease_expires_at < now()
                          AND attempt_count < max_attempts
                        )
                      ORDER BY created_at ASC
                      FOR UPDATE SKIP LOCKED
                      LIMIT 1
                    )
                    UPDATE document_jobs AS j
                    SET
                      status = 'running',
                      stage = 'starting',
                      percent = GREATEST(percent, 1),
                      started_at = COALESCE(started_at, now()),
                      lease_owner = %s,
                      lease_expires_at = now() + (%s || ' seconds')::interval,
                      attempt_count = attempt_count + 1
                    FROM candidate
                    WHERE j.id = candidate.id
                    RETURNING j.*
                    """,
                    (worker_id, int(lease_seconds)),
                ).fetchone()
                if not row:
                    return None
                job = dict(row)
                if job.get("batch_id"):
                    conn.execute(
                        """
                        UPDATE document_batches
                        SET status = CASE WHEN status = 'queued' THEN 'running' ELSE status END,
                            started_at = COALESCE(started_at, now())
                        WHERE id = %s
                        """,
                        (str(job["batch_id"]),),
                    )
                self.add_event_in_conn(
                    conn,
                    batch_id=job.get("batch_id"),
                    job_id=job["id"],
                    event_type="started",
                    stage="starting",
                    percent=int(job.get("percent") or 1),
                    message="Job claimed by worker.",
                    payload={"worker_id": worker_id, "attempt_count": job.get("attempt_count")},
                )
                return job

    def extend_lease(self, *, job_id: UUID, worker_id: str, lease_seconds: int) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                """
                UPDATE document_jobs
                SET lease_expires_at = now() + (%s || ' seconds')::interval
                WHERE id = %s AND status = 'running' AND lease_owner = %s
                RETURNING id
                """,
                (int(lease_seconds), str(job_id), worker_id),
            ).fetchone()
            conn.commit()
            return bool(row)

    def fail_expired_running_jobs(self, *, timeout_seconds: int, limit: int = 100) -> List[Dict[str, Any]]:
        if timeout_seconds <= 0:
            return []
        with self.pool.connection() as conn:
            with conn.transaction():
                rows = conn.execute(
                    """
                    WITH expired AS (
                      SELECT id
                      FROM document_jobs
                      WHERE status = 'running'
                        AND started_at IS NOT NULL
                        AND started_at < now() - (%s || ' seconds')::interval
                      ORDER BY started_at ASC
                      FOR UPDATE SKIP LOCKED
                      LIMIT %s
                    )
                    UPDATE document_jobs AS j
                    SET status = 'failed',
                        stage = 'failed',
                        finished_at = now(),
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        error_code = 'JOB_TIMEOUT',
                        error_message = 'Job exceeded configured processing timeout.',
                        error_details_json = jsonb_build_object('timeout_seconds', %s)
                    FROM expired
                    WHERE j.id = expired.id
                    RETURNING j.*
                    """,
                    (int(timeout_seconds), int(limit), int(timeout_seconds)),
                ).fetchall()
                jobs = [dict(row) for row in rows]
                for job in jobs:
                    self.add_event_in_conn(
                        conn,
                        batch_id=job.get("batch_id"),
                        job_id=job["id"],
                        event_type="failed",
                        stage="failed",
                        percent=int(job.get("percent") or 0),
                        message="Job exceeded configured processing timeout.",
                        payload={"error_code": "JOB_TIMEOUT", "timeout_seconds": timeout_seconds},
                    )
                    self.refresh_batch_in_conn(conn, job.get("batch_id"))
                return jobs

    def cancel_expired_batches(self, *, timeout_seconds: int, limit: int = 50) -> List[Dict[str, Any]]:
        if timeout_seconds <= 0:
            return []
        with self.pool.connection() as conn:
            with conn.transaction():
                rows = conn.execute(
                    """
                    SELECT *
                    FROM document_batches
                    WHERE status IN ('queued', 'running')
                      AND COALESCE(started_at, created_at) < now() - (%s || ' seconds')::interval
                    ORDER BY COALESCE(started_at, created_at) ASC
                    FOR UPDATE SKIP LOCKED
                    LIMIT %s
                    """,
                    (int(timeout_seconds), int(limit)),
                ).fetchall()
                batches = [dict(row) for row in rows]
                for batch in batches:
                    batch_id = batch["id"]
                    cancelled_jobs = conn.execute(
                        """
                        UPDATE document_jobs
                        SET status = 'cancelled',
                            stage = 'cancelled',
                            cancelled_at = now(),
                            finished_at = COALESCE(finished_at, now()),
                            lease_owner = NULL,
                            lease_expires_at = NULL,
                            error_code = COALESCE(error_code, 'BATCH_TIMEOUT'),
                            error_message = COALESCE(
                              error_message,
                              'Batch exceeded configured wall-clock timeout.'
                            )
                        WHERE batch_id = %s
                          AND status IN ('queued', 'running')
                        RETURNING *
                        """,
                        (str(batch_id),),
                    ).fetchall()
                    for job in cancelled_jobs:
                        self.add_event_in_conn(
                            conn,
                            batch_id=batch_id,
                            job_id=job["id"],
                            event_type="failed",
                            stage="cancelled",
                            percent=int(job.get("percent") or 0),
                            message="Batch exceeded configured wall-clock timeout.",
                            payload={"error_code": "BATCH_TIMEOUT", "timeout_seconds": timeout_seconds},
                        )
                    self.add_event_in_conn(
                        conn,
                        batch_id=batch_id,
                        job_id=None,
                        event_type="failed",
                        stage="cancelled",
                        percent=None,
                        message="Batch exceeded configured wall-clock timeout.",
                        payload={"error_code": "BATCH_TIMEOUT", "timeout_seconds": timeout_seconds},
                    )
                    self.refresh_batch_in_conn(conn, batch_id, cancellation_requested=True)
                return batches

    def update_progress(
        self,
        *,
        job_id: UUID,
        stage: str,
        percent: int,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        percent = max(0, min(100, int(percent)))
        with self.pool.connection() as conn:
            with conn.transaction():
                job = conn.execute(
                    """
                    UPDATE document_jobs
                    SET stage = %s, percent = %s
                    WHERE id = %s
                    RETURNING batch_id
                    """,
                    (stage, percent, str(job_id)),
                ).fetchone()
                if not job:
                    return
                self.add_event_in_conn(
                    conn,
                    batch_id=job.get("batch_id"),
                    job_id=job_id,
                    event_type="progress",
                    stage=stage,
                    percent=percent,
                    message=message,
                    payload=payload,
                )

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
        with self.pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    INSERT INTO document_assets(
                      id, batch_id, job_id, role, bucket, object_key, mime_type, size_bytes,
                      sha256, public_url, metadata_json
                    ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    RETURNING *
                    """,
                    (
                        str(asset_id or uuid4()),
                        str(batch_id) if batch_id else None,
                        str(job_id) if job_id else None,
                        role,
                        bucket,
                        object_key,
                        mime_type,
                        int(size_bytes),
                        sha256,
                        public_url,
                        _json_payload(metadata),
                    ),
                ).fetchone()
                ASSETS_UPLOADED.labels(role=role).inc()
                if batch_id or job_id:
                    self.add_event_in_conn(
                        conn,
                        batch_id=batch_id,
                        job_id=job_id,
                        event_type="asset_uploaded",
                        stage="asset_uploaded",
                        percent=None,
                        message=f"Uploaded {role} asset.",
                        payload={
                            "asset_id": str(row["id"]),
                            "role": role,
                            "public_url": public_url,
                            "size_bytes": int(size_bytes),
                        },
                    )
                return dict(row)

    def get_asset(self, asset_id: UUID) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM document_assets WHERE id = %s", (str(asset_id),)).fetchone()
            return dict(row) if row else None

    def get_asset_by_object_key(self, *, bucket: str, object_key: str) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM document_assets WHERE bucket = %s AND object_key = %s",
                (bucket, object_key),
            ).fetchone()
            return dict(row) if row else None

    def list_job_assets(self, job_id: UUID) -> List[Dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM document_assets WHERE job_id = %s ORDER BY created_at ASC",
                (str(job_id),),
            ).fetchall()
            return [dict(row) for row in rows]

    def mark_job_succeeded(self, *, job_id: UUID, markdown_asset_id: UUID) -> bool:
        with self.pool.connection() as conn:
            with conn.transaction():
                job = conn.execute(
                    """
                    UPDATE document_jobs
                    SET status = 'succeeded', stage = 'succeeded', percent = 100,
                        finished_at = now(), result_markdown_asset_id = %s,
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE id = %s AND status = 'running'
                    RETURNING *
                    """,
                    (str(markdown_asset_id), str(job_id)),
                ).fetchone()
                if not job:
                    return False
                self.add_event_in_conn(
                    conn,
                    batch_id=job.get("batch_id"),
                    job_id=job_id,
                    event_type="succeeded",
                    stage="succeeded",
                    percent=100,
                    message="Job succeeded.",
                )
                self.refresh_batch_in_conn(conn, job.get("batch_id"))
                return True

    def mark_job_failed(
        self,
        *,
        job_id: UUID,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False,
    ) -> None:
        with self.pool.connection() as conn:
            with conn.transaction():
                current = conn.execute("SELECT * FROM document_jobs WHERE id = %s", (str(job_id),)).fetchone()
                if not current:
                    return
                if str(current["status"]) == "cancelled":
                    return
                if retryable and int(current["attempt_count"]) < int(current["max_attempts"]):
                    job = conn.execute(
                        """
                        UPDATE document_jobs
                        SET status = 'queued', stage = 'retry_queued', percent = 0,
                            lease_owner = NULL, lease_expires_at = NULL,
                            error_code = %s, error_message = %s, error_details_json = %s::jsonb
                        WHERE id = %s
                        RETURNING *
                        """,
                        (code, message, _json_payload(details), str(job_id)),
                    ).fetchone()
                    self.add_event_in_conn(
                        conn,
                        batch_id=job.get("batch_id"),
                        job_id=job_id,
                        event_type="progress",
                        stage="retry_queued",
                        percent=0,
                        message="Job queued for retry.",
                        payload={"error_code": code},
                    )
                    return
                job = conn.execute(
                    """
                    UPDATE document_jobs
                    SET status = 'failed', stage = 'failed', finished_at = now(),
                        lease_owner = NULL, lease_expires_at = NULL,
                        error_code = %s, error_message = %s, error_details_json = %s::jsonb
                    WHERE id = %s
                    RETURNING *
                    """,
                    (code, message, _json_payload(details), str(job_id)),
                ).fetchone()
                self.add_event_in_conn(
                    conn,
                    batch_id=job.get("batch_id"),
                    job_id=job_id,
                    event_type="failed",
                    stage="failed",
                    percent=int(job.get("percent") or 0),
                    message=message,
                    payload={"error_code": code},
                )
                self.refresh_batch_in_conn(conn, job.get("batch_id"))

    def mark_source_deleted(self, job_id: UUID) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE document_jobs SET source_deleted_at = now() WHERE id = %s",
                (str(job_id),),
            )
            conn.commit()

    def cancel_job(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            with conn.transaction():
                job = conn.execute(
                    """
                    UPDATE document_jobs
                    SET status = 'cancelled', stage = 'cancelled', cancelled_at = now(),
                        finished_at = COALESCE(finished_at, now()),
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE id = %s AND status IN ('queued', 'running')
                    RETURNING *
                    """,
                    (str(job_id),),
                ).fetchone()
                if not job:
                    return None
                self.add_event_in_conn(
                    conn,
                    batch_id=job.get("batch_id"),
                    job_id=job_id,
                    event_type="failed",
                    stage="cancelled",
                    percent=int(job.get("percent") or 0),
                    message="Job cancelled.",
                )
                self.refresh_batch_in_conn(conn, job.get("batch_id"))
                return dict(job)

    def cancel_batch(self, batch_id: UUID) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            with conn.transaction():
                batch = conn.execute(
                    """
                    UPDATE document_batches
                    SET status = 'cancelled', cancelled_at = now()
                    WHERE id = %s AND status IN ('queued', 'running')
                    RETURNING *
                    """,
                    (str(batch_id),),
                ).fetchone()
                if not batch:
                    return None
                conn.execute(
                    """
                    UPDATE document_jobs
                    SET status = 'cancelled', stage = 'cancelled', cancelled_at = now(),
                        finished_at = COALESCE(finished_at, now()),
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE batch_id = %s AND status IN ('queued', 'running')
                    """,
                    (str(batch_id),),
                )
                self.add_event_in_conn(
                    conn,
                    batch_id=batch_id,
                    job_id=None,
                    event_type="failed",
                    stage="cancelled",
                    percent=None,
                    message="Batch cancelled.",
                )
                self.refresh_batch_in_conn(conn, batch_id, cancellation_requested=True)
                return dict(batch)

    def get_events(
        self,
        *,
        job_id: Optional[UUID] = None,
        batch_id: Optional[UUID] = None,
        after_id: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if not job_id and not batch_id:
            return []
        clauses = ["id > %s"]
        params: List[Any] = [int(after_id)]
        if job_id:
            clauses.append("job_id = %s")
            params.append(str(job_id))
        if batch_id:
            clauses.append("batch_id = %s")
            params.append(str(batch_id))
        params.append(int(limit))
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM job_events
                WHERE {' AND '.join(clauses)}
                ORDER BY id ASC
                LIMIT %s
                """,
                tuple(params),
            ).fetchall()
            return [dict(row) for row in rows]

    def add_event_in_conn(
        self,
        conn: Any,
        *,
        batch_id: Optional[UUID],
        job_id: Optional[UUID],
        event_type: str,
        stage: Optional[str],
        percent: Optional[int],
        message: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO job_events(batch_id, job_id, event_type, stage, percent, message, payload_json)
            VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (
                str(batch_id) if batch_id else None,
                str(job_id) if job_id else None,
                event_type,
                stage,
                percent,
                message,
                _json_payload(payload),
            ),
        )

    def refresh_batch_in_conn(
        self,
        conn: Any,
        batch_id: Optional[UUID],
        *,
        cancellation_requested: bool = False,
    ) -> None:
        if not batch_id:
            return
        rows = conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM document_jobs
            WHERE batch_id = %s
            GROUP BY status
            """,
            (str(batch_id),),
        ).fetchall()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        total_row = conn.execute(
            "SELECT total_files FROM document_batches WHERE id = %s",
            (str(batch_id),),
        ).fetchone()
        if not total_row:
            return
        total = int(total_row["total_files"])
        succeeded = counts.get("succeeded", 0)
        failed = counts.get("failed", 0)
        cancelled = counts.get("cancelled", 0)
        status = batch_status_from_counts(
            total_files=total,
            succeeded_count=succeeded,
            failed_count=failed,
            cancelled_count=cancelled,
            cancellation_requested=cancellation_requested,
        )
        terminal = succeeded + failed + cancelled
        conn.execute(
            """
            UPDATE document_batches
            SET status = %s,
                succeeded_count = %s,
                failed_count = %s,
                cancelled_count = %s,
                finished_at = CASE WHEN %s >= total_files THEN COALESCE(finished_at, now()) ELSE finished_at END
            WHERE id = %s
            """,
            (status, succeeded, failed, cancelled, terminal, str(batch_id)),
        )
        if terminal >= total:
            self.add_event_in_conn(
                conn,
                batch_id=batch_id,
                job_id=None,
                event_type="succeeded" if status == "succeeded" else "failed",
                stage=status,
                percent=100,
                message=f"Batch reached terminal status: {status}.",
                payload={
                    "total_files": total,
                    "succeeded_count": succeeded,
                    "failed_count": failed,
                    "cancelled_count": cancelled,
                },
            )

    def update_batch_archive_asset(self, *, batch_id: UUID, asset_id: UUID) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE document_batches SET result_archive_asset_id = %s WHERE id = %s",
                (str(asset_id), str(batch_id)),
            )
            conn.commit()
