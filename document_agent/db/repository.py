from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from document_agent.config import Settings, get_settings
from document_agent.db.connection import get_pool
from document_agent.metrics import ASSETS_UPLOADED
from document_agent.search.models import SearchHit
from document_agent.search.text import clean_headline
from document_agent.status import batch_status_from_counts


def _json_payload(value: Optional[Dict[str, Any]]) -> str:
    return json.dumps(value or {}, ensure_ascii=False)


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"


class Repository:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.pool = get_pool(self.settings)

    def _library_original_url(self, library_item_id: UUID) -> str:
        return f"{self.settings.public_base_url.rstrip('/')}/v1/library/{library_item_id}/original"

    def _library_preview_url(self, library_item_id: UUID) -> str:
        return f"{self.settings.public_base_url.rstrip('/')}/v1/library/{library_item_id}/preview"

    def _library_markdown_url(self, library_item_id: UUID) -> str:
        return f"{self.settings.public_base_url.rstrip('/')}/v1/library/{library_item_id}/markdown"

    def _initial_preview_status(self, detected_type: str) -> str:
        return "not_required" if detected_type in {"jpg", "jpeg", "png", "pdf", "txt"} else "pending"

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
        library_item_id: Optional[UUID] = None,
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
        library_item_id = library_item_id or uuid4()
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO document_library_items(
                      id, original_filename, display_filename, content_type, detected_type,
                      sha256, size_bytes, status, stage, percent, preview_status, metadata_json
                    ) VALUES(%s,%s,%s,%s,%s,%s,%s,'queued','queued',0,%s,%s::jsonb)
                    """,
                    (
                        str(library_item_id),
                        filename,
                        filename,
                        content_type,
                        detected_type,
                        sha256,
                        int(size_bytes),
                        self._initial_preview_status(detected_type),
                        _json_payload(metadata),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO document_jobs(
                      id, library_item_id, idempotency_key, filename, content_type, detected_type, sha256,
                      size_bytes, source_bucket, source_object_key, max_attempts, metadata_json
                    ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    """,
                    (
                        str(job_id),
                        str(library_item_id),
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
                )
                original_asset_id = uuid4()
                conn.execute(
                    """
                    INSERT INTO document_assets(
                      id, library_item_id, job_id, role, bucket, object_key, mime_type,
                      size_bytes, sha256, public_url, metadata_json
                    ) VALUES(%s,%s,%s,'original_file',%s,%s,%s,%s,%s,%s,%s::jsonb)
                    """,
                    (
                        str(original_asset_id),
                        str(library_item_id),
                        str(job_id),
                        source_bucket,
                        source_object_key,
                        content_type or "application/octet-stream",
                        int(size_bytes),
                        sha256,
                        self._library_original_url(library_item_id),
                        _json_payload({"filename": filename}),
                    ),
                )
                row = conn.execute(
                    """
                    UPDATE document_jobs
                    SET source_asset_id = %s
                    WHERE id = %s
                    RETURNING *
                    """,
                    (str(original_asset_id), str(job_id)),
                ).fetchone()
                conn.execute(
                    """
                    UPDATE document_library_items
                    SET current_job_id = %s, original_asset_id = %s, updated_at = now()
                    WHERE id = %s
                    """,
                    (str(job_id), str(original_asset_id), str(library_item_id)),
                )
                self.add_event_in_conn(
                    conn,
                    library_item_id=library_item_id,
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
                    library_item_id = item.get("library_item_id") or uuid4()
                    conn.execute(
                        """
                        INSERT INTO document_library_items(
                          id, batch_id, input_index, original_filename, display_filename,
                          content_type, detected_type, sha256, size_bytes, status, stage,
                          percent, preview_status, metadata_json
                        ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued','queued',0,%s,%s::jsonb)
                        """,
                        (
                            str(library_item_id),
                            str(batch_id),
                            int(item["input_index"]),
                            item["filename"],
                            item["filename"],
                            item.get("content_type"),
                            item["detected_type"],
                            item["sha256"],
                            int(item["size_bytes"]),
                            self._initial_preview_status(item["detected_type"]),
                            _json_payload(item.get("metadata")),
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO document_jobs(
                          id, library_item_id, batch_id, input_index, filename, content_type,
                          detected_type, sha256, size_bytes, source_bucket, source_object_key,
                          max_attempts, metadata_json
                        ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                        """,
                        (
                            str(item["job_id"]),
                            str(library_item_id),
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
                    original_asset_id = uuid4()
                    conn.execute(
                        """
                        INSERT INTO document_assets(
                          id, library_item_id, batch_id, job_id, role, bucket, object_key,
                          mime_type, size_bytes, sha256, public_url, metadata_json
                        ) VALUES(%s,%s,%s,%s,'original_file',%s,%s,%s,%s,%s,%s,%s::jsonb)
                        """,
                        (
                            str(original_asset_id),
                            str(library_item_id),
                            str(batch_id),
                            str(item["job_id"]),
                            item["source_bucket"],
                            item["source_object_key"],
                            item.get("content_type") or "application/octet-stream",
                            int(item["size_bytes"]),
                            item["sha256"],
                            self._library_original_url(library_item_id),
                            _json_payload({"filename": item["filename"]}),
                        ),
                    )
                    conn.execute(
                        """
                        UPDATE document_jobs
                        SET source_asset_id = %s
                        WHERE id = %s
                        """,
                        (str(original_asset_id), str(item["job_id"])),
                    )
                    conn.execute(
                        """
                        UPDATE document_library_items
                        SET current_job_id = %s, original_asset_id = %s, updated_at = now()
                        WHERE id = %s
                        """,
                        (str(item["job_id"]), str(original_asset_id), str(library_item_id)),
                    )
                    self.add_event_in_conn(
                        conn,
                        library_item_id=library_item_id,
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
                    library_item_id=None,
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

    def list_library_items(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None,
        detected_type: Optional[str] = None,
        batch_id: Optional[UUID] = None,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses = ["deleted_at IS NULL"]
        params: List[Any] = []
        if status_filter:
            clauses.append("status = %s")
            params.append(status_filter)
        if detected_type:
            clauses.append("detected_type = %s")
            params.append(detected_type)
        if batch_id:
            clauses.append("batch_id = %s")
            params.append(str(batch_id))
        if query:
            clauses.append("display_filename ILIKE %s")
            params.append(f"%{query}%")
        params.extend([int(limit), int(offset)])
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM document_library_items
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                tuple(params),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_library_item(self, library_item_id: UUID) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM document_library_items
                WHERE id = %s AND deleted_at IS NULL
                """,
                (str(library_item_id),),
            ).fetchone()
            return dict(row) if row else None

    def get_library_item_by_job(self, job_id: UUID) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            row = conn.execute(
                """
                SELECT li.*
                FROM document_library_items AS li
                JOIN document_jobs AS j ON j.library_item_id = li.id
                WHERE j.id = %s AND li.deleted_at IS NULL
                """,
                (str(job_id),),
            ).fetchone()
            return dict(row) if row else None

    def list_library_assets(self, library_item_id: UUID) -> List[Dict[str, Any]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM document_assets
                WHERE library_item_id = %s
                ORDER BY created_at ASC
                """,
                (str(library_item_id),),
            ).fetchall()
            return [dict(row) for row in rows]

    def upsert_search_entry(
        self,
        *,
        library_item_id: UUID,
        job_id: UUID,
        asset_id: UUID,
        filename: str,
        detected_type: Optional[str],
        content: str,
    ) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO document_search_entries(
                  library_item_id, job_id, asset_id, filename, detected_type, content
                ) VALUES(%s,%s,%s,%s,%s,%s)
                ON CONFLICT (library_item_id) DO UPDATE
                SET job_id = EXCLUDED.job_id,
                    asset_id = EXCLUDED.asset_id,
                    filename = EXCLUDED.filename,
                    detected_type = EXCLUDED.detected_type,
                    content = EXCLUDED.content,
                    updated_at = now()
                """,
                (
                    str(library_item_id),
                    str(job_id),
                    str(asset_id),
                    filename,
                    detected_type,
                    content,
                ),
            )
            conn.commit()

    def delete_search_entry(self, library_item_id: UUID) -> None:
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    "DELETE FROM document_search_entries WHERE library_item_id = %s",
                    (str(library_item_id),),
                )
                conn.execute(
                    "DELETE FROM document_search_chunks WHERE library_item_id = %s",
                    (str(library_item_id),),
                )
            conn.commit()

    def replace_search_chunks(
        self,
        *,
        library_item_id: UUID,
        job_id: UUID,
        asset_id: UUID,
        filename: str,
        detected_type: Optional[str],
        chunks: list[tuple[int, str, list[float]]],
    ) -> None:
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    "DELETE FROM document_search_chunks WHERE library_item_id = %s",
                    (str(library_item_id),),
                )
                if not chunks:
                    return
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO document_search_chunks(
                          library_item_id, job_id, asset_id, chunk_index, filename,
                          detected_type, content, embedding
                        ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s::vector)
                        """,
                        [
                            (
                                str(library_item_id),
                                str(job_id),
                                str(asset_id),
                                int(index),
                                filename,
                                detected_type,
                                content,
                                _vector_literal(embedding),
                            )
                            for index, content, embedding in chunks
                        ],
                    )

    def list_markdown_assets_for_search_reindex(
        self,
        *,
        limit: int = 500,
        only_missing: bool = True,
    ) -> List[Dict[str, Any]]:
        missing_clause = "AND se.library_item_id IS NULL" if only_missing else ""
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                  li.id AS library_item_id,
                  li.current_job_id AS job_id,
                  li.display_filename AS filename,
                  li.detected_type,
                  a.id AS asset_id,
                  a.bucket,
                  a.object_key
                FROM document_library_items AS li
                JOIN document_assets AS a ON a.id = li.latest_markdown_asset_id
                LEFT JOIN document_search_entries AS se ON se.library_item_id = li.id
                WHERE li.deleted_at IS NULL
                  AND li.status = 'succeeded'
                  AND li.latest_markdown_asset_id IS NOT NULL
                  {missing_clause}
                ORDER BY li.processed_at DESC NULLS LAST, li.updated_at DESC
                LIMIT %s
                """,
                (int(limit),),
            ).fetchall()
            return [dict(row) for row in rows]

    def search_documents(
        self,
        *,
        query: str,
        limit: int = 20,
        offset: int = 0,
        detected_type: Optional[str] = None,
        library_item_id: Optional[UUID] = None,
    ) -> tuple[List[SearchHit], int]:
        clauses = [
            "li.deleted_at IS NULL",
            "li.status = 'succeeded'",
            "(q.query @@ se.search_vector OR se.content ILIKE %s OR se.filename ILIKE %s)",
        ]
        pattern = f"%{query}%"
        params: List[Any] = [query, pattern, pattern]
        if detected_type:
            clauses.append("se.detected_type = %s")
            params.append(detected_type)
        if library_item_id:
            clauses.append("se.library_item_id = %s")
            params.append(str(library_item_id))

        where_sql = " AND ".join(clauses)
        count_params = tuple(params)
        select_params = tuple(params + [int(limit), int(offset)])
        with self.pool.connection() as conn:
            total_row = conn.execute(
                f"""
                WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS query)
                SELECT count(*) AS total
                FROM document_search_entries AS se
                JOIN document_library_items AS li ON li.id = se.library_item_id
                CROSS JOIN q
                WHERE {where_sql}
                """,
                count_params,
            ).fetchone()
            rows = conn.execute(
                f"""
                WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS query)
                SELECT
                  se.library_item_id,
                  se.job_id,
                  se.asset_id,
                  se.filename,
                  se.detected_type,
                  se.content,
                  li.processed_at,
                  ts_rank_cd(se.search_vector, q.query) AS score,
                  ts_headline(
                    'simple',
                    se.content,
                    q.query,
                    'StartSel=<mark>, StopSel=</mark>, MaxWords=35, MinWords=12, MaxFragments=3, FragmentDelimiter= ... '
                  ) AS headline
                FROM document_search_entries AS se
                JOIN document_library_items AS li ON li.id = se.library_item_id
                CROSS JOIN q
                WHERE {where_sql}
                ORDER BY
                  (q.query @@ se.search_vector) DESC,
                  ts_rank_cd(se.search_vector, q.query) DESC,
                  li.processed_at DESC NULLS LAST,
                  se.updated_at DESC
                LIMIT %s OFFSET %s
                """,
                select_params,
            ).fetchall()

        hits = [
            SearchHit(
                library_item_id=UUID(str(row["library_item_id"])),
                job_id=UUID(str(row["job_id"])),
                asset_id=UUID(str(row["asset_id"])),
                filename=row["filename"],
                detected_type=row.get("detected_type"),
                score=float(row.get("score") or 0.0),
                snippet=clean_headline(row.get("headline") or "", row.get("content") or "", query),
                markdown_url=self._library_markdown_url(UUID(str(row["library_item_id"]))),
                preview_url=self._library_preview_url(UUID(str(row["library_item_id"]))),
                processed_at=row.get("processed_at"),
            )
            for row in rows
        ]
        return hits, int(total_row["total"] if total_row else 0)

    def search_chunks_keyword(
        self,
        *,
        query: str,
        limit: int = 40,
        detected_type: Optional[str] = None,
        library_item_id: Optional[UUID] = None,
    ) -> List[SearchHit]:
        clauses = [
            "li.deleted_at IS NULL",
            "li.status = 'succeeded'",
            "(q.query @@ sc.search_vector OR sc.content ILIKE %s OR sc.filename ILIKE %s)",
        ]
        pattern = f"%{query}%"
        params: List[Any] = [query, pattern, pattern]
        if detected_type:
            clauses.append("sc.detected_type = %s")
            params.append(detected_type)
        if library_item_id:
            clauses.append("sc.library_item_id = %s")
            params.append(str(library_item_id))
        params.append(int(limit))
        where_sql = " AND ".join(clauses)
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                WITH q AS (SELECT websearch_to_tsquery('simple', %s) AS query)
                SELECT
                  sc.library_item_id,
                  sc.job_id,
                  sc.asset_id,
                  sc.chunk_index,
                  sc.filename,
                  sc.detected_type,
                  sc.content,
                  li.processed_at,
                  ts_rank_cd(sc.search_vector, q.query) AS score,
                  ts_headline(
                    'simple',
                    sc.content,
                    q.query,
                    'StartSel=<mark>, StopSel=</mark>, MaxWords=35, MinWords=12, MaxFragments=3, FragmentDelimiter= ... '
                  ) AS headline
                FROM document_search_chunks AS sc
                JOIN document_library_items AS li ON li.id = sc.library_item_id
                CROSS JOIN q
                WHERE {where_sql}
                ORDER BY
                  (q.query @@ sc.search_vector) DESC,
                  ts_rank_cd(sc.search_vector, q.query) DESC,
                  li.processed_at DESC NULLS LAST
                LIMIT %s
                """,
                tuple(params),
            ).fetchall()
        return [
            self._search_hit_from_row(
                row,
                query=query,
                score=float(row.get("score") or 0.0),
                keyword_score=float(row.get("score") or 0.0),
                semantic_score=0.0,
            )
            for row in rows
        ]

    def search_chunks_semantic(
        self,
        *,
        query: str,
        embedding: list[float],
        limit: int = 40,
        detected_type: Optional[str] = None,
        library_item_id: Optional[UUID] = None,
    ) -> List[SearchHit]:
        clauses = [
            "li.deleted_at IS NULL",
            "li.status = 'succeeded'",
            "sc.embedding IS NOT NULL",
        ]
        vector = _vector_literal(embedding)
        filter_params: List[Any] = []
        if detected_type:
            clauses.append("sc.detected_type = %s")
            filter_params.append(detected_type)
        if library_item_id:
            clauses.append("sc.library_item_id = %s")
            filter_params.append(str(library_item_id))
        where_sql = " AND ".join(clauses)
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                  sc.library_item_id,
                  sc.job_id,
                  sc.asset_id,
                  sc.chunk_index,
                  sc.filename,
                  sc.detected_type,
                  sc.content,
                  li.processed_at,
                  1 - (sc.embedding <=> %s::vector) AS similarity,
                  NULL::text AS headline
                FROM document_search_chunks AS sc
                JOIN document_library_items AS li ON li.id = sc.library_item_id
                WHERE {where_sql}
                ORDER BY sc.embedding <=> %s::vector
                LIMIT %s
                """,
                tuple([vector, *filter_params, vector, int(limit)]),
            ).fetchall()
        return [
            self._search_hit_from_row(
                row,
                query=query,
                score=float(row.get("similarity") or 0.0),
                keyword_score=0.0,
                semantic_score=float(row.get("similarity") or 0.0),
            )
            for row in rows
        ]

    def _search_hit_from_row(
        self,
        row: Dict[str, Any],
        *,
        query: str,
        score: float,
        keyword_score: float,
        semantic_score: float,
    ) -> SearchHit:
        return SearchHit(
            library_item_id=UUID(str(row["library_item_id"])),
            job_id=UUID(str(row["job_id"])),
            asset_id=UUID(str(row["asset_id"])),
            filename=row["filename"],
            detected_type=row.get("detected_type"),
            score=score,
            keyword_score=keyword_score,
            semantic_score=semantic_score,
            chunk_index=int(row["chunk_index"]) if row.get("chunk_index") is not None else None,
            snippet=clean_headline(row.get("headline") or "", row.get("content") or "", query),
            markdown_url=self._library_markdown_url(UUID(str(row["library_item_id"]))),
            preview_url=self._library_preview_url(UUID(str(row["library_item_id"]))),
            processed_at=row.get("processed_at"),
        )

    def mark_library_preview_failed(self, *, library_item_id: UUID, message: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """
                UPDATE document_library_items
                SET preview_status = 'failed',
                    updated_at = now(),
                    metadata_json = metadata_json || jsonb_build_object('preview_error', %s)
                WHERE id = %s
                """,
                (message, str(library_item_id)),
            )
            conn.commit()

    def delete_library_item_records(self, library_item_id: UUID) -> int:
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE document_jobs
                    SET status = CASE WHEN status IN ('queued', 'running') THEN 'cancelled' ELSE status END,
                        stage = CASE WHEN status IN ('queued', 'running') THEN 'cancelled' ELSE stage END,
                        cancelled_at = CASE
                          WHEN status IN ('queued', 'running') THEN COALESCE(cancelled_at, now())
                          ELSE cancelled_at
                        END,
                        finished_at = CASE
                          WHEN status IN ('queued', 'running') THEN COALESCE(finished_at, now())
                          ELSE finished_at
                        END,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        result_markdown_asset_id = NULL
                    WHERE library_item_id = %s
                    """,
                    (str(library_item_id),),
                )
                conn.execute(
                    """
                    UPDATE document_library_items
                    SET deleted_at = COALESCE(deleted_at, now()),
                        status = 'deleted',
                        stage = 'deleted',
                        updated_at = now(),
                        original_asset_id = NULL,
                        preview_asset_id = NULL,
                        thumbnail_asset_id = NULL,
                        latest_markdown_asset_id = NULL
                    WHERE id = %s
                    """,
                    (str(library_item_id),),
                )
                conn.execute(
                    "DELETE FROM document_search_entries WHERE library_item_id = %s",
                    (str(library_item_id),),
                )
                conn.execute(
                    "DELETE FROM document_search_chunks WHERE library_item_id = %s",
                    (str(library_item_id),),
                )
                rows = conn.execute(
                    "DELETE FROM document_assets WHERE library_item_id = %s RETURNING id",
                    (str(library_item_id),),
                ).fetchall()
                return len(rows)

    def create_reprocess_job(self, library_item_id: UUID) -> Optional[Dict[str, Any]]:
        with self.pool.connection() as conn:
            with conn.transaction():
                item = conn.execute(
                    """
                    SELECT *
                    FROM document_library_items
                    WHERE id = %s AND deleted_at IS NULL
                    FOR UPDATE
                    """,
                    (str(library_item_id),),
                ).fetchone()
                if not item:
                    return None
                original = conn.execute(
                    "SELECT * FROM document_assets WHERE id = %s",
                    (str(item["original_asset_id"]),),
                ).fetchone()
                if not original:
                    return None
                job_id = uuid4()
                row = conn.execute(
                    """
                    INSERT INTO document_jobs(
                      id, library_item_id, batch_id, input_index, filename, content_type,
                      detected_type, sha256, size_bytes, source_bucket, source_object_key,
                      source_asset_id, max_attempts, metadata_json
                    ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    RETURNING *
                    """,
                    (
                        str(job_id),
                        str(library_item_id),
                        str(item["batch_id"]) if item.get("batch_id") else None,
                        item.get("input_index"),
                        item["display_filename"],
                        item.get("content_type"),
                        item.get("detected_type"),
                        item["sha256"],
                        int(item["size_bytes"]),
                        original["bucket"],
                        original["object_key"],
                        str(original["id"]),
                        self.settings.job_max_attempts,
                        _json_payload(item.get("metadata_json")),
                    ),
                ).fetchone()
                conn.execute(
                    """
                    UPDATE document_library_items
                    SET current_job_id = %s,
                        status = 'queued',
                        stage = 'queued',
                        percent = 0,
                        processed_at = NULL,
                        updated_at = now(),
                        error_code = NULL,
                        error_message = NULL,
                        error_details_json = '{}'::jsonb
                    WHERE id = %s
                    """,
                    (str(job_id), str(library_item_id)),
                )
                self.add_event_in_conn(
                    conn,
                    library_item_id=library_item_id,
                    batch_id=item.get("batch_id"),
                    job_id=job_id,
                    event_type="queued",
                    stage="queued",
                    percent=0,
                    message="Library item queued for reprocessing.",
                )
                return dict(row)

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

    def list_expired_completed_assets(
        self,
        *,
        retention_seconds: int,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        if retention_seconds <= 0:
            return []
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT a.*
                FROM document_assets AS a
                LEFT JOIN document_jobs AS j ON j.id = a.job_id
                LEFT JOIN document_batches AS b ON b.id = a.batch_id
                WHERE a.role IN (
                    'markdown_result',
                    'embedded_image',
                    'equation_image',
                    'preview_pdf',
                    'preview_image',
                    'preview_text',
                    'thumbnail',
                    'diagnostic_manifest',
                    'batch_manifest',
                    'batch_archive'
                )
                  AND COALESCE(j.finished_at, b.finished_at, a.created_at)
                    < now() - (%s || ' seconds')::interval
                  AND (
                    j.status IN ('succeeded', 'failed', 'cancelled')
                    OR b.status IN ('succeeded', 'partial_failed', 'failed', 'cancelled')
                  )
                ORDER BY a.created_at ASC
                LIMIT %s
                """,
                (int(retention_seconds), int(limit)),
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
                if job.get("library_item_id"):
                    conn.execute(
                        """
                        UPDATE document_library_items
                        SET status = 'running',
                            stage = 'starting',
                            percent = GREATEST(percent, 1),
                            updated_at = now()
                        WHERE id = %s
                        """,
                        (str(job["library_item_id"]),),
                    )
                self.add_event_in_conn(
                    conn,
                    library_item_id=job.get("library_item_id"),
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
                    if job.get("library_item_id"):
                        conn.execute(
                            """
                            UPDATE document_library_items
                            SET status = 'failed',
                                stage = 'failed',
                                percent = document_jobs.percent,
                                processed_at = now(),
                                updated_at = now(),
                                error_code = 'JOB_TIMEOUT',
                                error_message = 'Job exceeded configured processing timeout.',
                                error_details_json = jsonb_build_object('timeout_seconds', %s)
                            FROM document_jobs
                            WHERE document_library_items.id = %s
                              AND document_jobs.id = %s
                            """,
                            (int(timeout_seconds), str(job["library_item_id"]), str(job["id"])),
                        )
                    self.add_event_in_conn(
                        conn,
                        library_item_id=job.get("library_item_id"),
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
                        if job.get("library_item_id"):
                            conn.execute(
                                """
                                UPDATE document_library_items
                                SET status = 'cancelled',
                                    stage = 'cancelled',
                                    percent = %s,
                                    processed_at = now(),
                                    updated_at = now(),
                                    error_code = COALESCE(error_code, 'BATCH_TIMEOUT'),
                                    error_message = COALESCE(
                                      error_message,
                                      'Batch exceeded configured wall-clock timeout.'
                                    )
                                WHERE id = %s
                                """,
                                (int(job.get("percent") or 0), str(job["library_item_id"])),
                            )
                        self.add_event_in_conn(
                            conn,
                            library_item_id=job.get("library_item_id"),
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
                        library_item_id=None,
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
                    RETURNING batch_id, library_item_id
                    """,
                    (stage, percent, str(job_id)),
                ).fetchone()
                if not job:
                    return
                if job.get("library_item_id"):
                    conn.execute(
                        """
                        UPDATE document_library_items
                        SET stage = %s, percent = %s, updated_at = now()
                        WHERE id = %s
                          AND current_job_id = %s
                          AND status IN ('queued', 'running')
                        """,
                        (stage, percent, str(job["library_item_id"]), str(job_id)),
                    )
                self.add_event_in_conn(
                    conn,
                    library_item_id=job.get("library_item_id"),
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
        library_item_id: Optional[UUID] = None,
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
                      id, library_item_id, batch_id, job_id, role, bucket, object_key, mime_type,
                      size_bytes, sha256, public_url, metadata_json
                    ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    RETURNING *
                    """,
                    (
                        str(asset_id or uuid4()),
                        str(library_item_id) if library_item_id else None,
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
                if library_item_id and role in {"preview_pdf", "preview_image", "preview_text"}:
                    conn.execute(
                        """
                        UPDATE document_library_items
                        SET preview_asset_id = %s, preview_status = 'succeeded', updated_at = now()
                        WHERE id = %s
                        """,
                        (str(row["id"]), str(library_item_id)),
                    )
                if library_item_id and role == "thumbnail":
                    conn.execute(
                        """
                        UPDATE document_library_items
                        SET thumbnail_asset_id = %s, updated_at = now()
                        WHERE id = %s
                        """,
                        (str(row["id"]), str(library_item_id)),
                    )
                if library_item_id and role == "markdown_result":
                    conn.execute(
                        """
                        UPDATE document_library_items
                        SET latest_markdown_asset_id = %s, updated_at = now()
                        WHERE id = %s
                        """,
                        (str(row["id"]), str(library_item_id)),
                    )
                if batch_id or job_id:
                    self.add_event_in_conn(
                        conn,
                        library_item_id=library_item_id,
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
                if job.get("library_item_id"):
                    conn.execute(
                        """
                        UPDATE document_library_items
                        SET status = 'succeeded',
                            stage = 'succeeded',
                            percent = 100,
                            latest_markdown_asset_id = %s,
                            processed_at = now(),
                            updated_at = now(),
                            error_code = NULL,
                            error_message = NULL,
                            error_details_json = '{}'::jsonb
                        WHERE id = %s
                          AND current_job_id = %s
                        """,
                        (str(markdown_asset_id), str(job["library_item_id"]), str(job_id)),
                    )
                self.add_event_in_conn(
                    conn,
                    library_item_id=job.get("library_item_id"),
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
                    if job.get("library_item_id"):
                        conn.execute(
                            """
                            UPDATE document_library_items
                            SET status = 'queued',
                                stage = 'retry_queued',
                                percent = 0,
                                updated_at = now(),
                                error_code = %s,
                                error_message = %s,
                                error_details_json = %s::jsonb
                            WHERE id = %s
                              AND current_job_id = %s
                            """,
                            (
                                code,
                                message,
                                _json_payload(details),
                                str(job["library_item_id"]),
                                str(job_id),
                            ),
                        )
                    self.add_event_in_conn(
                        conn,
                        library_item_id=job.get("library_item_id"),
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
                if job.get("library_item_id"):
                    conn.execute(
                        """
                        UPDATE document_library_items
                        SET status = 'failed',
                            stage = 'failed',
                            processed_at = now(),
                            updated_at = now(),
                            error_code = %s,
                            error_message = %s,
                            error_details_json = %s::jsonb
                        WHERE id = %s
                          AND current_job_id = %s
                        """,
                        (
                            code,
                            message,
                            _json_payload(details),
                            str(job["library_item_id"]),
                            str(job_id),
                        ),
                    )
                self.add_event_in_conn(
                    conn,
                    library_item_id=job.get("library_item_id"),
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
                if job.get("library_item_id"):
                    conn.execute(
                        """
                        UPDATE document_library_items
                        SET status = 'cancelled',
                            stage = 'cancelled',
                            percent = %s,
                            processed_at = now(),
                            updated_at = now()
                        WHERE id = %s
                          AND current_job_id = %s
                        """,
                        (int(job.get("percent") or 0), str(job["library_item_id"]), str(job_id)),
                    )
                self.add_event_in_conn(
                    conn,
                    library_item_id=job.get("library_item_id"),
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
                cancelled_jobs = conn.execute(
                    """
                    UPDATE document_jobs
                    SET status = 'cancelled', stage = 'cancelled', cancelled_at = now(),
                        finished_at = COALESCE(finished_at, now()),
                        lease_owner = NULL, lease_expires_at = NULL
                    WHERE batch_id = %s AND status IN ('queued', 'running')
                    RETURNING *
                    """,
                    (str(batch_id),),
                ).fetchall()
                for job in cancelled_jobs:
                    if job.get("library_item_id"):
                        conn.execute(
                            """
                            UPDATE document_library_items
                            SET status = 'cancelled',
                                stage = 'cancelled',
                                percent = %s,
                                processed_at = now(),
                                updated_at = now()
                            WHERE id = %s
                              AND current_job_id = %s
                            """,
                            (
                                int(job.get("percent") or 0),
                                str(job["library_item_id"]),
                                str(job["id"]),
                            ),
                        )
                self.add_event_in_conn(
                    conn,
                    library_item_id=None,
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
        library_item_id: Optional[UUID] = None,
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
            INSERT INTO job_events(
              library_item_id, batch_id, job_id, event_type, stage, percent, message, payload_json
            )
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (
                str(library_item_id) if library_item_id else None,
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

    # ------------------------------------------------------------------
    # Observability queries
    # ------------------------------------------------------------------

    def get_observability_stats(self) -> Dict[str, Any]:
        with self.pool.connection() as conn:
            status_rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM document_jobs GROUP BY status"
            ).fetchall()
            duration_row = conn.execute(
                """
                SELECT
                  AVG(EXTRACT(EPOCH FROM (finished_at - started_at))) AS avg_seconds,
                  PERCENTILE_CONT(0.95) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (finished_at - started_at))
                  ) AS p95_seconds
                FROM document_jobs
                WHERE status = 'succeeded'
                  AND started_at IS NOT NULL
                  AND finished_at IS NOT NULL
                """
            ).fetchone()
            throughput_rows = conn.execute(
                """
                SELECT date_trunc('hour', finished_at) AS hour, status, COUNT(*) AS count
                FROM document_jobs
                WHERE status IN ('succeeded', 'failed')
                  AND finished_at >= now() - interval '24 hours'
                GROUP BY 1, 2 ORDER BY 1 ASC
                """
            ).fetchall()
            type_rows = conn.execute(
                "SELECT detected_type, COUNT(*) AS count FROM document_jobs GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall()
            batch_row = conn.execute("SELECT COUNT(*) AS total FROM document_batches").fetchone()
            lease_row = conn.execute(
                "SELECT COUNT(*) AS active FROM document_jobs WHERE status = 'running' AND lease_expires_at > now()"
            ).fetchone()
        return {
            "status_counts": [dict(r) for r in status_rows],
            "duration": dict(duration_row) if duration_row else {},
            "throughput_by_hour": [dict(r) for r in throughput_rows],
            "jobs_by_type": [dict(r) for r in type_rows],
            "total_batches": int(batch_row["total"]) if batch_row else 0,
            "active_leases": int(lease_row["active"]) if lease_row else 0,
        }

    def get_global_events(
        self,
        *,
        limit: int = 50,
        before_id: Optional[int] = None,
        since_id: Optional[int] = None,
        event_type: Optional[str] = None,
        q: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if since_id is not None:
            clauses: List[str] = ["id > %s"]
            params: List[Any] = [int(since_id)]
            if event_type:
                clauses.append("event_type = %s")
                params.append(event_type)
            if q:
                clauses.append("message ILIKE %s")
                params.append(f"%{q}%")
            params.append(int(limit))
            with self.pool.connection() as conn:
                rows = conn.execute(
                    f"SELECT * FROM job_events WHERE {' AND '.join(clauses)} ORDER BY id ASC LIMIT %s",
                    tuple(params),
                ).fetchall()
            return [dict(r) for r in rows]

        clauses = []
        params = []
        if before_id is not None:
            clauses.append("id < %s")
            params.append(int(before_id))
        if event_type:
            clauses.append("event_type = %s")
            params.append(event_type)
        if q:
            clauses.append("message ILIKE %s")
            params.append(f"%{q}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit) + 1)
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM job_events {where} ORDER BY id DESC LIMIT %s",
                tuple(params),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_errors(
        self,
        *,
        limit: int = 20,
        error_code: Optional[str] = None,
        q: Optional[str] = None,
    ) -> Dict[str, Any]:
        clauses = ["status IN ('failed', 'cancelled')", "error_code IS NOT NULL"]
        params: List[Any] = []
        if error_code:
            clauses.append("error_code = %s")
            params.append(error_code)
        if q:
            clauses.append("error_message ILIKE %s")
            params.append(f"%{q}%")
        where = " AND ".join(clauses)
        params.append(int(limit))
        with self.pool.connection() as conn:
            error_rows = conn.execute(
                f"""
                SELECT id AS job_id, library_item_id, filename, detected_type,
                       error_code, error_message,
                       COALESCE(finished_at, cancelled_at) AS failed_at, attempt_count
                FROM document_jobs
                WHERE {where}
                ORDER BY failed_at DESC NULLS LAST
                LIMIT %s
                """,
                tuple(params),
            ).fetchall()
            dist_rows = conn.execute(
                """
                SELECT error_code, COUNT(*) AS count
                FROM document_jobs
                WHERE status IN ('failed', 'cancelled') AND error_code IS NOT NULL
                GROUP BY 1 ORDER BY 2 DESC
                """
            ).fetchall()
            total_row = conn.execute(
                "SELECT COUNT(*) AS n FROM document_jobs WHERE status IN ('failed', 'cancelled')"
            ).fetchone()
        return {
            "errors": [dict(r) for r in error_rows],
            "error_code_counts": [dict(r) for r in dist_rows],
            "total_failed": int(total_row["n"]) if total_row else 0,
        }

    def forget_assets(self, asset_ids: List[UUID]) -> int:
        if not asset_ids:
            return 0
        ids = [str(asset_id) for asset_id in asset_ids]
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    """
                    UPDATE document_jobs
                    SET result_markdown_asset_id = NULL
                    WHERE result_markdown_asset_id = ANY(%s::uuid[])
                    """,
                    (ids,),
                )
                conn.execute(
                    """
                    UPDATE document_batches
                    SET result_archive_asset_id = NULL
                    WHERE result_archive_asset_id = ANY(%s::uuid[])
                    """,
                    (ids,),
                )
                conn.execute(
                    """
                    UPDATE document_library_items
                    SET latest_markdown_asset_id = CASE
                          WHEN latest_markdown_asset_id = ANY(%s::uuid[]) THEN NULL
                          ELSE latest_markdown_asset_id
                        END,
                        preview_asset_id = CASE
                          WHEN preview_asset_id = ANY(%s::uuid[]) THEN NULL
                          ELSE preview_asset_id
                        END,
                        thumbnail_asset_id = CASE
                          WHEN thumbnail_asset_id = ANY(%s::uuid[]) THEN NULL
                          ELSE thumbnail_asset_id
                        END,
                        updated_at = now()
                    WHERE latest_markdown_asset_id = ANY(%s::uuid[])
                       OR preview_asset_id = ANY(%s::uuid[])
                       OR thumbnail_asset_id = ANY(%s::uuid[])
                    """,
                    (ids, ids, ids, ids, ids, ids),
                )
                rows = conn.execute(
                    "DELETE FROM document_assets WHERE id = ANY(%s::uuid[]) RETURNING id",
                    (ids,),
                ).fetchall()
                return len(rows)
