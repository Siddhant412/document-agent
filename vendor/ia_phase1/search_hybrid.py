from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Protocol


class SimilarityStore(Protocol):
    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        paper_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        ...


def _parse_json_field(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return None


async def full_text_search(
    query: str,
    pool: Any,
    k: int = 10,
    paper_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    PostgreSQL full-text search on `text_blocks` via tsvector/plainto_tsquery.
    """
    sql = """
        SELECT
            tb.id,
            tb.paper_id,
            tb.page_no,
            tb.block_index,
            tb.text,
            tb.bbox,
            tb.metadata,
            p.title as paper_title,
            p.source_url,
            ts_rank(to_tsvector('english', tb.text), plainto_tsquery('english', $1)) as rank
        FROM text_blocks tb
        JOIN papers p ON tb.paper_id = p.id
        WHERE to_tsvector('english', tb.text) @@ plainto_tsquery('english', $1)
    """
    params: List[Any] = [query]
    param_idx = 2

    if paper_ids:
        sql += f" AND tb.paper_id = ANY(${param_idx})"
        params.append(paper_ids)
        param_idx += 1

    sql += f" ORDER BY rank DESC LIMIT ${param_idx}"
    params.append(k)

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    results: List[Dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "paper_id": row["paper_id"],
                "page_no": row["page_no"],
                "block_index": row["block_index"],
                "text": row["text"],
                "bbox": _parse_json_field(row["bbox"]),
                "metadata": _parse_json_field(row["metadata"]),
                "paper_title": row["paper_title"],
                "source_url": row["source_url"],
                "score": float(row["rank"]),
                "source": "fts",
            }
        )
    return results


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    fts_results: List[Dict[str, Any]],
    k: int = 10,
    alpha: float = 0.5,
    k_constant: int = 60,
) -> List[Dict[str, Any]]:
    """
    Combine vector and FTS result lists using reciprocal rank fusion (RRF).
    """
    scores: Dict[Any, float] = {}
    result_map: Dict[Any, Dict[str, Any]] = {}

    for rank, result in enumerate(vector_results, start=1):
        block_id = result["id"]
        rrf_score = alpha / (k_constant + rank)
        if block_id not in scores:
            scores[block_id] = 0.0
            result_map[block_id] = result.copy()
            result_map[block_id]["sources"] = []
        scores[block_id] += rrf_score
        result_map[block_id]["sources"].append(
            {
                "type": "vector",
                "rank": rank,
                "score": result.get("similarity", 0),
            }
        )

    for rank, result in enumerate(fts_results, start=1):
        block_id = result["id"]
        rrf_score = (1 - alpha) / (k_constant + rank)
        if block_id not in scores:
            scores[block_id] = 0.0
            result_map[block_id] = result.copy()
            result_map[block_id]["sources"] = []
        scores[block_id] += rrf_score
        result_map[block_id]["sources"].append(
            {
                "type": "fts",
                "rank": rank,
                "score": result.get("score", 0),
            }
        )

    sorted_ids = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    final_results: List[Dict[str, Any]] = []
    for block_id in sorted_ids[:k]:
        item = result_map[block_id]
        item["hybrid_score"] = scores[block_id]
        final_results.append(item)
    return final_results


async def hybrid_search(
    query: str,
    pgvector_store: SimilarityStore,
    pool: Any,
    k: int = 10,
    paper_ids: Optional[List[int]] = None,
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    retrieve_k = k * 2
    vector_results = await pgvector_store.similarity_search(
        query,
        k=retrieve_k,
        paper_ids=paper_ids,
    )
    fts_results = await full_text_search(
        query,
        pool,
        k=retrieve_k,
        paper_ids=paper_ids,
    )
    return reciprocal_rank_fusion(
        vector_results,
        fts_results,
        k=k,
        alpha=alpha,
    )


async def search_with_reranking(
    query: str,
    pgvector_store: SimilarityStore,
    pool: Any,
    k: int = 10,
    paper_ids: Optional[List[int]] = None,
    alpha: float = 0.5,
    rerank_top_k: int = 50,
) -> List[Dict[str, Any]]:
    candidates = await hybrid_search(
        query,
        pgvector_store,
        pool,
        k=rerank_top_k,
        paper_ids=paper_ids,
        alpha=alpha,
    )
    return candidates[:k]


__all__ = [
    "full_text_search",
    "reciprocal_rank_fusion",
    "hybrid_search",
    "search_with_reranking",
]
