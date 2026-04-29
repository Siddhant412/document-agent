from __future__ import annotations

from contextlib import AbstractContextManager
import re
from typing import Any, Callable, Dict, List, Literal, Optional


SearchType = Literal["keyword", "embedding", "hybrid"]
ConnectionFactory = Callable[[], AbstractContextManager[Any]]

_conn_factory: Optional[ConnectionFactory] = None
_SHORT_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def configure_connection_factory(factory: ConnectionFactory) -> None:
    global _conn_factory
    _conn_factory = factory


def _resolve_conn_factory(explicit: Optional[ConnectionFactory]) -> ConnectionFactory:
    factory = explicit or _conn_factory
    if factory is None:
        raise RuntimeError(
            "No SQLite connection factory configured. "
            "Pass get_conn_fn=... or call configure_connection_factory(...)."
        )
    return factory


def _fts_query(query: str) -> str:
    return f'"{query}"' if " " in query or "-" in query else query


def _should_try_boundary_fallback(query: str) -> bool:
    tokens = _SHORT_TOKEN_RE.findall(query or "")
    if len(tokens) != 1:
        return False
    token = tokens[0]
    if len(token) <= 4:
        return True
    return len(token) <= 8 and any(ch.isupper() for ch in token)


def _boundary_fallback_terms(query: str) -> List[str]:
    tokens = _SHORT_TOKEN_RE.findall(query or "")
    if len(tokens) != 1:
        return []
    token = tokens[0].lower()
    terms = [token]
    if len(token) >= 2 and not token.endswith("s"):
        terms.append(f"{token}s")
    deduped: List[str] = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return deduped


def _normalized_text_sql(column: str) -> str:
    expr = f"lower(coalesce({column}, ''))"
    expr = f"replace(replace(replace({expr}, char(10), ' '), char(13), ' '), char(9), ' ')"
    for char in (".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "\"", "'", "`", "/", "\\", "|", "-", "_"):
        escaped = char.replace("'", "''")
        expr = f"replace({expr}, '{escaped}', ' ')"
    return f"' ' || {expr} || ' '"


def _search_sections_boundary_fallback(
    conn: Any,
    *,
    query: str,
    paper_ids: Optional[List[int]],
    limit: int,
) -> List[Dict[str, Any]]:
    terms = _boundary_fallback_terms(query)
    if not terms:
        return []

    normalized_text = _normalized_text_sql("s.text")
    like_clause = " OR ".join(f"{normalized_text} LIKE ?" for _ in terms)
    params: List[Any] = [f"% {term} %" for term in terms]

    if paper_ids:
        placeholders = ",".join("?" * len(paper_ids))
        sql = f"""
            SELECT s.id, s.paper_id, s.page_no, s.text,
                   p.title as paper_title, p.source_url,
                   0 as rank
            FROM sections s
            JOIN papers p ON s.paper_id = p.id
            WHERE ({like_clause}) AND s.paper_id IN ({placeholders})
            ORDER BY s.page_no, s.id
            LIMIT ?
        """
        params.extend(paper_ids)
    else:
        sql = f"""
            SELECT s.id, s.paper_id, s.page_no, s.text,
                   p.title as paper_title, p.source_url,
                   0 as rank
            FROM sections s
            JOIN papers p ON s.paper_id = p.id
            WHERE ({like_clause})
            ORDER BY s.page_no, s.id
            LIMIT ?
        """
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def search_papers(
    query: str,
    search_type: SearchType = "keyword",
    limit: int = 20,
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> List[Dict[str, Any]]:
    if search_type == "embedding":
        search_type = "keyword"

    if search_type not in {"keyword", "hybrid"}:
        return []

    conn_factory = _resolve_conn_factory(get_conn_fn)
    with conn_factory() as conn:
        fts_query = _fts_query(query)
        try:
            rows = conn.execute(
                """
                SELECT p.id, p.title, p.source_url, p.pdf_path, p.rag_status,
                       p.rag_error, p.rag_updated_at, p.created_at,
                       pf.rank
                FROM papers p
                JOIN papers_fts pf ON p.id = pf.rowid
                WHERE papers_fts MATCH ?
                ORDER BY pf.rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            rows = conn.execute(
                """
                SELECT id, title, source_url, pdf_path, rag_status,
                       rag_error, rag_updated_at, created_at,
                       0 as rank
                FROM papers
                WHERE title LIKE ? OR source_url LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            ).fetchall()
            return [dict(row) for row in rows]


def search_sections(
    query: str,
    search_type: SearchType = "keyword",
    paper_ids: Optional[List[int]] = None,
    limit: int = 50,
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> List[Dict[str, Any]]:
    if search_type == "embedding":
        return []
    if search_type not in {"keyword", "hybrid"}:
        return []

    conn_factory = _resolve_conn_factory(get_conn_fn)
    with conn_factory() as conn:
        fts_query = _fts_query(query)
        try:
            if paper_ids:
                placeholders = ",".join("?" * len(paper_ids))
                sql = f"""
                    SELECT s.id, s.paper_id, s.page_no, s.text,
                           p.title as paper_title, p.source_url,
                           sf.rank
                    FROM sections s
                    JOIN sections_fts sf ON s.id = sf.rowid
                    JOIN papers p ON s.paper_id = p.id
                    WHERE sections_fts MATCH ? AND s.paper_id IN ({placeholders})
                    ORDER BY sf.rank
                    LIMIT ?
                """
                params = [fts_query, *paper_ids, limit]
            else:
                sql = """
                    SELECT s.id, s.paper_id, s.page_no, s.text,
                           p.title as paper_title, p.source_url,
                           sf.rank
                    FROM sections s
                    JOIN sections_fts sf ON s.id = sf.rowid
                    JOIN papers p ON s.paper_id = p.id
                    WHERE sections_fts MATCH ?
                    ORDER BY sf.rank
                    LIMIT ?
                """
                params = [fts_query, limit]
            rows = conn.execute(sql, params).fetchall()
            if rows:
                return [dict(row) for row in rows]
            if _should_try_boundary_fallback(query):
                return _search_sections_boundary_fallback(
                    conn,
                    query=query,
                    paper_ids=paper_ids,
                    limit=limit,
                )
            return []
        except Exception:
            if _should_try_boundary_fallback(query):
                rows = _search_sections_boundary_fallback(
                    conn,
                    query=query,
                    paper_ids=paper_ids,
                    limit=limit,
                )
                if rows:
                    return rows
            if paper_ids:
                placeholders = ",".join("?" * len(paper_ids))
                sql = f"""
                    SELECT s.id, s.paper_id, s.page_no, s.text,
                           p.title as paper_title, p.source_url,
                           0 as rank
                    FROM sections s
                    JOIN papers p ON s.paper_id = p.id
                    WHERE s.text LIKE ? AND s.paper_id IN ({placeholders})
                    ORDER BY s.page_no
                    LIMIT ?
                """
                params = [f"%{query}%", *paper_ids, limit]
            else:
                sql = """
                    SELECT s.id, s.paper_id, s.page_no, s.text,
                           p.title as paper_title, p.source_url,
                           0 as rank
                    FROM sections s
                    JOIN papers p ON s.paper_id = p.id
                    WHERE s.text LIKE ?
                    ORDER BY s.page_no
                    LIMIT ?
                """
                params = [f"%{query}%", limit]
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]


def search_notes(
    query: str,
    search_type: SearchType = "keyword",
    paper_ids: Optional[List[int]] = None,
    limit: int = 50,
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> List[Dict[str, Any]]:
    if search_type == "embedding":
        search_type = "keyword"
    if search_type not in {"keyword", "hybrid"}:
        return []

    conn_factory = _resolve_conn_factory(get_conn_fn)
    with conn_factory() as conn:
        fts_query = _fts_query(query)
        try:
            if paper_ids:
                placeholders = ",".join("?" * len(paper_ids))
                sql = f"""
                    SELECT n.id, n.paper_id, n.title, n.body, n.tags_json, n.created_at,
                           p.title as paper_title,
                           nf.rank
                    FROM notes n
                    JOIN notes_fts nf ON n.id = nf.rowid
                    LEFT JOIN papers p ON n.paper_id = p.id
                    WHERE notes_fts MATCH ? AND (n.paper_id IN ({placeholders}) OR n.paper_id IS NULL)
                    ORDER BY nf.rank
                    LIMIT ?
                """
                params = [fts_query, *paper_ids, limit]
            else:
                sql = """
                    SELECT n.id, n.paper_id, n.title, n.body, n.tags_json, n.created_at,
                           p.title as paper_title,
                           nf.rank
                    FROM notes n
                    JOIN notes_fts nf ON n.id = nf.rowid
                    LEFT JOIN papers p ON n.paper_id = p.id
                    WHERE notes_fts MATCH ?
                    ORDER BY nf.rank
                    LIMIT ?
                """
                params = [fts_query, limit]
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            if paper_ids:
                placeholders = ",".join("?" * len(paper_ids))
                sql = f"""
                    SELECT n.id, n.paper_id, n.title, n.body, n.tags_json, n.created_at,
                           p.title as paper_title,
                           0 as rank
                    FROM notes n
                    LEFT JOIN papers p ON n.paper_id = p.id
                    WHERE (n.title LIKE ? OR n.body LIKE ?)
                          AND (n.paper_id IN ({placeholders}) OR n.paper_id IS NULL)
                    ORDER BY n.created_at DESC
                    LIMIT ?
                """
                params = [f"%{query}%", f"%{query}%", *paper_ids, limit]
            else:
                sql = """
                    SELECT n.id, n.paper_id, n.title, n.body, n.tags_json, n.created_at,
                           p.title as paper_title,
                           0 as rank
                    FROM notes n
                    LEFT JOIN papers p ON n.paper_id = p.id
                    WHERE n.title LIKE ? OR n.body LIKE ?
                    ORDER BY n.created_at DESC
                    LIMIT ?
                """
                params = [f"%{query}%", f"%{query}%", limit]
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]


def search_summaries(
    query: str,
    search_type: SearchType = "keyword",
    paper_ids: Optional[List[int]] = None,
    limit: int = 50,
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> List[Dict[str, Any]]:
    if search_type == "embedding":
        search_type = "keyword"
    if search_type not in {"keyword", "hybrid"}:
        return []

    conn_factory = _resolve_conn_factory(get_conn_fn)
    with conn_factory() as conn:
        fts_query = _fts_query(query)
        try:
            if paper_ids:
                placeholders = ",".join("?" * len(paper_ids))
                sql = f"""
                    SELECT s.id, s.paper_id, s.title, s.content, s.agent, s.style,
                           s.word_count, s.is_edited, s.metadata_json, s.created_at, s.updated_at,
                           p.title as paper_title,
                           sf.rank
                    FROM summaries s
                    JOIN summaries_fts sf ON s.id = sf.rowid
                    JOIN papers p ON s.paper_id = p.id
                    WHERE summaries_fts MATCH ? AND s.paper_id IN ({placeholders})
                    ORDER BY sf.rank
                    LIMIT ?
                """
                params = [fts_query, *paper_ids, limit]
            else:
                sql = """
                    SELECT s.id, s.paper_id, s.title, s.content, s.agent, s.style,
                           s.word_count, s.is_edited, s.metadata_json, s.created_at, s.updated_at,
                           p.title as paper_title,
                           sf.rank
                    FROM summaries s
                    JOIN summaries_fts sf ON s.id = sf.rowid
                    JOIN papers p ON s.paper_id = p.id
                    WHERE summaries_fts MATCH ?
                    ORDER BY sf.rank
                    LIMIT ?
                """
                params = [fts_query, limit]
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            if paper_ids:
                placeholders = ",".join("?" * len(paper_ids))
                sql = f"""
                    SELECT s.id, s.paper_id, s.title, s.content, s.agent, s.style,
                           s.word_count, s.is_edited, s.metadata_json, s.created_at, s.updated_at,
                           p.title as paper_title,
                           0 as rank
                    FROM summaries s
                    JOIN papers p ON s.paper_id = p.id
                    WHERE (s.title LIKE ? OR s.content LIKE ?) AND s.paper_id IN ({placeholders})
                    ORDER BY s.created_at DESC
                    LIMIT ?
                """
                params = [f"%{query}%", f"%{query}%", *paper_ids, limit]
            else:
                sql = """
                    SELECT s.id, s.paper_id, s.title, s.content, s.agent, s.style,
                           s.word_count, s.is_edited, s.metadata_json, s.created_at, s.updated_at,
                           p.title as paper_title,
                           0 as rank
                    FROM summaries s
                    JOIN papers p ON s.paper_id = p.id
                    WHERE s.title LIKE ? OR s.content LIKE ?
                    ORDER BY s.created_at DESC
                    LIMIT ?
                """
                params = [f"%{query}%", f"%{query}%", limit]
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]


def search_all(
    query: str,
    search_type: SearchType = "keyword",
    paper_ids: Optional[List[int]] = None,
    limit_per_category: int = 10,
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> Dict[str, Any]:
    return {
        "papers": search_papers(query, search_type, limit_per_category, get_conn_fn=get_conn_fn),
        "sections": search_sections(query, search_type, paper_ids, limit_per_category, get_conn_fn=get_conn_fn),
        "notes": search_notes(query, search_type, paper_ids, limit_per_category, get_conn_fn=get_conn_fn),
        "summaries": search_summaries(query, search_type, paper_ids, limit_per_category, get_conn_fn=get_conn_fn),
    }


__all__ = [
    "SearchType",
    "configure_connection_factory",
    "search_papers",
    "search_sections",
    "search_notes",
    "search_summaries",
    "search_all",
]
