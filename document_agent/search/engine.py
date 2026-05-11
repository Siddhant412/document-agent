from __future__ import annotations

from typing import Optional
from uuid import UUID

from document_agent.db.repository import Repository
from document_agent.search.chunking import chunk_markdown
from document_agent.search.embeddings import get_embedding_service
from document_agent.search.models import SearchHit, SearchQuery, SearchResponse
from document_agent.search.text import normalize_query
from document_agent.storage import ObjectStore


class DocumentSearchEngine:
    def __init__(
        self,
        repository: Optional[Repository] = None,
        object_store: Optional[ObjectStore] = None,
    ) -> None:
        self.repository = repository or Repository()
        self.object_store = object_store or ObjectStore()

    def search(self, query: SearchQuery) -> SearchResponse:
        normalized = normalize_query(query.query)
        if not normalized:
            return SearchResponse(query=query.query, hits=[], limit=query.limit, offset=query.offset, total=0)
        mode = query.mode if query.mode in {"keyword", "semantic", "hybrid"} else "hybrid"
        retrieve_limit = max(query.limit + query.offset, query.limit * 4, 40)
        if mode == "keyword":
            ranked = self.repository.search_chunks_keyword(
                query=normalized,
                limit=retrieve_limit,
                detected_type=query.detected_type,
                library_item_id=query.library_item_id,
            )
        elif mode == "semantic":
            ranked = self._semantic_search(normalized, query, retrieve_limit)
        else:
            keyword_hits = self.repository.search_chunks_keyword(
                query=normalized,
                limit=retrieve_limit,
                detected_type=query.detected_type,
                library_item_id=query.library_item_id,
            )
            semantic_hits = self._semantic_search(normalized, query, retrieve_limit)
            ranked = reciprocal_rank_fusion(
                keyword_hits,
                semantic_hits,
                alpha=float(self.repository.settings.hybrid_search_alpha),
            )
        hits = ranked[query.offset : query.offset + query.limit]
        return SearchResponse(
            query=normalized,
            hits=hits,
            limit=query.limit,
            offset=query.offset,
            total=len(ranked),
        )

    def index_markdown(
        self,
        *,
        library_item_id: UUID,
        job_id: UUID,
        asset_id: UUID,
        filename: str,
        detected_type: Optional[str],
        markdown: str,
    ) -> None:
        self.repository.upsert_search_entry(
            library_item_id=library_item_id,
            job_id=job_id,
            asset_id=asset_id,
            filename=filename,
            detected_type=detected_type,
            content=markdown,
        )
        if not self.repository.settings.search_semantic_enabled:
            return
        chunks = chunk_markdown(
            markdown,
            max_chars=self.repository.settings.search_chunk_chars,
            overlap=self.repository.settings.search_chunk_overlap,
        )
        embedder = get_embedding_service(self.repository.settings)
        embeddings = embedder.embed_texts([chunk.text for chunk in chunks])
        self.repository.replace_search_chunks(
            library_item_id=library_item_id,
            job_id=job_id,
            asset_id=asset_id,
            filename=filename,
            detected_type=detected_type,
            chunks=[
                (chunk.index, chunk.text, embeddings[index])
                for index, chunk in enumerate(chunks)
            ],
        )

    def reindex_existing_markdown(self, *, limit: int = 500, only_missing: bool = True) -> tuple[int, int]:
        rows = self.repository.list_markdown_assets_for_search_reindex(
            limit=limit,
            only_missing=only_missing,
        )
        indexed = 0
        skipped = 0
        for row in rows:
            try:
                data = self.object_store.read_object_bytes(
                    bucket=row["bucket"],
                    object_key=row["object_key"],
                )
                markdown = data.decode("utf-8", errors="replace")
                self.index_markdown(
                    library_item_id=UUID(str(row["library_item_id"])),
                    job_id=UUID(str(row["job_id"])),
                    asset_id=UUID(str(row["asset_id"])),
                    filename=row["filename"],
                    detected_type=row.get("detected_type"),
                    markdown=markdown,
                )
                indexed += 1
            except Exception:
                skipped += 1
        return indexed, skipped

    def _semantic_search(self, normalized: str, query: SearchQuery, retrieve_limit: int) -> list[SearchHit]:
        if not self.repository.settings.search_semantic_enabled:
            return []
        embedder = get_embedding_service(self.repository.settings)
        embedding = embedder.embed_query(normalized)
        return self.repository.search_chunks_semantic(
            query=normalized,
            embedding=embedding,
            limit=retrieve_limit,
            detected_type=query.detected_type,
            library_item_id=query.library_item_id,
        )


def reciprocal_rank_fusion(
    keyword_hits: list[SearchHit],
    semantic_hits: list[SearchHit],
    *,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> list[SearchHit]:
    alpha = min(1.0, max(0.0, alpha))
    merged: dict[tuple[UUID, Optional[int]], SearchHit] = {}
    scores: dict[tuple[UUID, Optional[int]], float] = {}
    keyword_scores: dict[tuple[UUID, Optional[int]], float] = {}
    semantic_scores: dict[tuple[UUID, Optional[int]], float] = {}

    for rank, hit in enumerate(keyword_hits, 1):
        key = (hit.library_item_id, hit.chunk_index)
        merged.setdefault(key, hit)
        keyword_scores[key] = max(keyword_scores.get(key, 0.0), hit.keyword_score or hit.score)
        scores[key] = scores.get(key, 0.0) + (1.0 - alpha) / (rrf_k + rank)

    for rank, hit in enumerate(semantic_hits, 1):
        key = (hit.library_item_id, hit.chunk_index)
        merged.setdefault(key, hit)
        semantic_scores[key] = max(semantic_scores.get(key, 0.0), hit.semantic_score or hit.score)
        scores[key] = scores.get(key, 0.0) + alpha / (rrf_k + rank)

    ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [
        SearchHit(
            library_item_id=merged[key].library_item_id,
            job_id=merged[key].job_id,
            asset_id=merged[key].asset_id,
            filename=merged[key].filename,
            detected_type=merged[key].detected_type,
            score=scores[key],
            keyword_score=keyword_scores.get(key, 0.0),
            semantic_score=semantic_scores.get(key, 0.0),
            chunk_index=merged[key].chunk_index,
            snippet=merged[key].snippet,
            markdown_url=merged[key].markdown_url,
            preview_url=merged[key].preview_url,
            processed_at=merged[key].processed_at,
        )
        for key in ranked
    ]
