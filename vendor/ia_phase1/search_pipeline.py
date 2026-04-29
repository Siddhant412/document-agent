from __future__ import annotations

import json
import logging
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional

from .search_context import query_tokens, sentence_focus_features

_REFERENCE_HEADING_RE = re.compile(r"^\s*(references|bibliography)\b", re.I)
_REFERENCE_SIGNAL_RE = re.compile(
    r"(?:\[[0-9]{1,3}\])|(?:\b(?:19|20)\d{2}\b)|(?:\b(?:proc\.?|proceedings|conference|journal|arxiv|doi)\b)",
    re.I,
)
_FRONT_MATTER_SIGNAL_RE = re.compile(
    r"(?:\babstract\b)|(?:@)|(?:\buniversity\b)|(?:\bdepartment\b)|(?:\bcorrespond(?:ing|ence)\b)|(?:\bemail\b)",
    re.I,
)
_SEARCH_STOPWORDS = {
    "about",
    "above",
    "across",
    "after",
    "against",
    "also",
    "and",
    "are",
    "based",
    "before",
    "below",
    "between",
    "can",
    "could",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "here",
    "into",
    "less",
    "more",
    "not",
    "other",
    "over",
    "per",
    "should",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "throughout",
    "toward",
    "towards",
    "under",
    "use",
    "used",
    "using",
    "via",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "within",
    "without",
    "would",
}
_TITLE_ONLY_RESCUE_MIN_SCORE = 0.30
_METHOD_QUERY_TERMS = {
    "approach",
    "architecture",
    "cache",
    "curvature",
    "defense",
    "design",
    "framework",
    "mechanism",
    "method",
    "model",
    "objective",
    "pipeline",
    "reuse",
    "router",
    "training",
}
_RESULT_QUERY_TERMS = {
    "accuracy",
    "benchmark",
    "evaluation",
    "f1",
    "gain",
    "gains",
    "improvement",
    "improvements",
    "latency",
    "performance",
    "result",
    "results",
    "speedup",
    "throughput",
}
_ANALYSIS_QUERY_TERMS = {
    "analysis",
    "challenge",
    "discussion",
    "failure",
    "failures",
    "insight",
    "limitation",
    "limitations",
    "reasoning",
    "tradeoff",
    "understanding",
}

ConnectionFactory = Callable[[], Any]
KeywordSectionHitsFn = Callable[..., List[Dict[str, Any]]]
SemanticSectionHitsFn = Callable[..., List[Dict[str, Any]]]

_CONNECTION_FACTORY: Optional[ConnectionFactory] = None
logger = logging.getLogger(__name__)


def _search_trace_enabled() -> bool:
    raw = os.getenv("SEARCH_TRACE_ENABLED", "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _search_trace_limit() -> int:
    try:
        return max(1, int(os.getenv("SEARCH_TRACE_MAX_HITS", "5")))
    except (TypeError, ValueError):
        return 5


def _search_trace_text_chars() -> int:
    try:
        return max(40, int(os.getenv("SEARCH_TRACE_TEXT_CHARS", "160")))
    except (TypeError, ValueError):
        return 160


def _truncate_for_trace(value: Any, *, max_chars: Optional[int] = None) -> str:
    text = " ".join(str(value or "").split())
    limit = max_chars if max_chars is not None else _search_trace_text_chars()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _summarize_hit_for_trace(hit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": safe_int(hit.get("id")),
        "paper_id": safe_int(hit.get("paper_id")),
        "page_no": safe_int(hit.get("page_no")),
        "section": str(hit.get("match_section_canonical") or ""),
        "bucket": str(hit.get("search_bucket") or ""),
        "match_score": round(safe_float(hit.get("match_score")), 4),
        "keyword_score": round(safe_float(hit.get("keyword_score")), 4),
        "semantic_score": round(safe_float(hit.get("semantic_score")), 4),
        "semantic_raw_score": round(safe_float(hit.get("semantic_raw_score")), 4),
        "block_match_score": round(safe_float(hit.get("block_match_score")), 4),
        "localization_score": round(safe_float(hit.get("localization_score")), 4),
        "lex_hits": safe_int(hit.get("lex_hits")),
        "content_hits": safe_int(hit.get("content_hits")),
        "rare_hits": safe_int(hit.get("rare_hits")),
        "weighted_hit_ratio": round(safe_float(hit.get("weighted_hit_ratio")), 4),
        "exact_phrase": bool(hit.get("exact_phrase")),
        "title_only_match": bool(hit.get("title_only_match")),
        "snippet": _truncate_for_trace(hit.get("match_text") or hit.get("source_text") or hit.get("text")),
    }


def _summarize_hits_for_trace(hits: List[Dict[str, Any]], *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    max_items = limit if limit is not None else _search_trace_limit()
    return [_summarize_hit_for_trace(hit) for hit in hits[:max_items]]


def _summarize_paper_for_trace(paper_id: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    support_hits = list(meta.get("support_hits") or [])
    return {
        "paper_id": int(paper_id),
        "score": round(safe_float(meta.get("score")), 4),
        "title_bonus": round(safe_float(meta.get("title_bonus")), 4),
        "title_only_match": bool(meta.get("title_only_match")),
        "best_hit": _summarize_hit_for_trace(meta.get("best_hit") or {}),
        "support_hits": _summarize_hits_for_trace(support_hits),
    }


def _trace_search(event: str, **payload: Any) -> None:
    if not _search_trace_enabled():
        return
    record = {"event": event, **payload}
    try:
        rendered = json.dumps(record, ensure_ascii=False, sort_keys=True)
    except TypeError:
        rendered = json.dumps({"event": event, "payload_repr": repr(payload)}, ensure_ascii=False, sort_keys=True)
    logger.info("[search-trace] %s", rendered)


def configure_connection_factory(factory: ConnectionFactory) -> None:
    global _CONNECTION_FACTORY
    _CONNECTION_FACTORY = factory


def _get_conn() -> Any:
    if _CONNECTION_FACTORY is None:
        raise RuntimeError(
            "ia_phase1.search_pipeline connection factory is not configured. "
            "Call configure_connection_factory(...) before using connection-backed helpers."
        )
    return _CONNECTION_FACTORY()


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _row_first_value(row: Any) -> Any:
    if row is None:
        return None
    try:
        return row[0]
    except Exception:
        pass
    if isinstance(row, dict):
        for value in row.values():
            return value
        return None
    keys = getattr(row, "keys", None)
    if callable(keys):
        row_keys = list(keys())
        if row_keys:
            try:
                return row[row_keys[0]]
            except Exception:
                return None
    return None


def rrf_score(rank_index: int, k: int = 20) -> float:
    return 1.0 / float(k + rank_index + 1)


def token_overlap(tokens: List[str], text: str) -> int:
    if not tokens or not text:
        return 0
    haystack = text.lower()
    return sum(1 for token in tokens if token and token in haystack)


def infer_search_section_bucket(
    text: str,
    *,
    page_no: Optional[int] = None,
    section_canonical: Optional[str] = None,
) -> str:
    canonical = str(section_canonical or "").strip().lower()
    if canonical == "references":
        return "references"
    if canonical == "front_matter":
        return "front_matter"
    if canonical in {"acknowledgements", "acknowledgments", "acknowledgement", "acknowledgment"}:
        return "front_matter"

    text = (text or "").replace("\x00", " ").strip()
    if not text:
        return "body"

    sample = text[:1200]
    if _REFERENCE_HEADING_RE.search(sample[:120]):
        return "references"

    reference_signals = len(_REFERENCE_SIGNAL_RE.findall(sample))
    if reference_signals >= 8:
        return "references"

    if page_no == 1:
        front_signals = len(_FRONT_MATTER_SIGNAL_RE.findall(sample[:600]))
        if front_signals >= 2:
            return "front_matter"

    return "body"


def section_bucket_multiplier(bucket: str) -> float:
    normalized = str(bucket or "body").strip().lower()
    if normalized == "references":
        return 0.18
    if normalized == "front_matter":
        return 0.60
    return 1.0


def min_lex_hits_for_query(token_count: int) -> int:
    return 1 if token_count <= 2 else 2


def content_query_tokens(query: str) -> List[str]:
    return [token for token in query_tokens(query) if token not in _SEARCH_STOPWORDS]


def query_token_stats(query: str, *, get_conn_fn: Optional[ConnectionFactory] = None) -> Dict[str, Any]:
    content_tokens = content_query_tokens(query)
    if not content_tokens:
        return {
            "content_tokens": [],
            "rare_tokens": [],
            "rare_limit": 0,
            "token_stats": {},
            "total_sections": 0,
        }

    conn_factory = get_conn_fn or _get_conn
    with conn_factory() as conn:
        total_sections_row = conn.execute("SELECT COUNT(*) FROM sections").fetchone()
        total_sections = int(_row_first_value(total_sections_row) or 0) if total_sections_row else 0
        rare_limit = max(3, int(total_sections * 0.06)) if total_sections > 0 else 3
        token_stats: Dict[str, Dict[str, float]] = {}
        rare_tokens: List[str] = []
        for token in content_tokens:
            try:
                df_row = conn.execute(
                    "SELECT COUNT(*) FROM sections_fts WHERE sections_fts MATCH ?",
                    (token,),
                ).fetchone()
                df = int(_row_first_value(df_row) or 0) if df_row else 0
            except Exception:
                like = f"%{token}%"
                df_row = conn.execute(
                    "SELECT COUNT(*) FROM sections WHERE lower(text) LIKE ?",
                    (like,),
                ).fetchone()
                df = int(_row_first_value(df_row) or 0) if df_row else 0
            weight = math.log((total_sections + 1.0) / (df + 1.0)) + 1.0 if total_sections > 0 else 1.0
            token_stats[token] = {"df": float(df), "weight": weight}
            if df <= rare_limit:
                rare_tokens.append(token)

    return {
        "content_tokens": content_tokens,
        "rare_tokens": rare_tokens,
        "rare_limit": rare_limit,
        "token_stats": token_stats,
        "total_sections": total_sections,
    }


def content_token_hits(tokens: List[str], text: str) -> int:
    if not tokens or not text:
        return 0
    haystack = text.lower()
    return sum(1 for token in tokens if token in haystack)


def weighted_token_coverage(token_stats: Dict[str, Dict[str, float]], text: str) -> float:
    if not token_stats or not text:
        return 0.0
    haystack = text.lower()
    total_weight = sum(float(meta.get("weight") or 0.0) for meta in token_stats.values())
    if total_weight <= 0.0:
        return 0.0
    matched_weight = sum(
        float(meta.get("weight") or 0.0)
        for token, meta in token_stats.items()
        if token in haystack
    )
    return matched_weight / total_weight


def is_low_salience_section(section_canonical: Optional[str]) -> bool:
    canonical = str(section_canonical or "").strip().lower()
    return canonical in {
        "acknowledgements",
        "acknowledgments",
        "acknowledgement",
        "acknowledgment",
        "experimental_protocol",
        "supplementary",
        "supplementary_material",
        "appendix",
        "appendices",
    }


def infer_localization_query_profile(query: str) -> Dict[str, bool]:
    tokens = query_tokens(query)
    token_set = set(tokens)
    method_like = bool(token_set & _METHOD_QUERY_TERMS)
    result_like = bool(token_set & _RESULT_QUERY_TERMS)
    analysis_like = bool(token_set & _ANALYSIS_QUERY_TERMS)
    benchmark_like = "benchmark" in token_set or "dataset" in token_set or "evaluation" in token_set
    overview_like = not (method_like or result_like or analysis_like or benchmark_like)
    return {
        "method_like": method_like,
        "result_like": result_like,
        "analysis_like": analysis_like,
        "benchmark_like": benchmark_like,
        "overview_like": overview_like,
    }


def infer_localization_section_role(section_canonical: Optional[str]) -> str:
    canonical = str(section_canonical or "").strip().lower()
    if not canonical:
        return "body"
    if is_low_salience_section(canonical):
        return "low_salience"
    if canonical == "abstract" or canonical.startswith("abstract"):
        return "abstract"
    if canonical == "front_matter":
        return "front_matter"
    if any(token in canonical for token in ("introduction", "background", "related_work", "preliminar")):
        return "intro"
    if any(token in canonical for token in ("conclusion", "discussion", "limitation")):
        return "conclusion"
    if any(token in canonical for token in ("analysis", "ablation", "experiment", "result", "evaluation", "benchmark")):
        return "evaluation"
    if any(token in canonical for token in ("method", "approach", "framework", "architecture", "training", "implementation", "pipeline", "model", "algorithm", "setup")):
        return "method"
    return "body"


def annotate_hit_query_support(hit: Dict[str, Any], *, query_stats: Dict[str, Any]) -> Dict[str, Any]:
    content_tokens = list(query_stats.get("content_tokens") or [])
    rare_tokens = list(query_stats.get("rare_tokens") or [])
    token_stats = dict(query_stats.get("token_stats") or {})
    query = str(query_stats.get("query") or "")
    source_text = str(
        hit.get("source_text")
        or hit.get("match_text")
        or hit.get("text")
        or ""
    )
    hit["content_hits"] = content_token_hits(content_tokens, source_text)
    hit["content_token_count"] = len(content_tokens)
    hit["rare_hits"] = content_token_hits(rare_tokens, source_text)
    hit["rare_token_count"] = len(rare_tokens)
    hit["weighted_hit_ratio"] = weighted_token_coverage(token_stats, source_text)
    sentence_focus = sentence_focus_features(query, content_tokens, source_text)
    hit["sentence_hits"] = int(sentence_focus.get("sentence_hits", 0))
    hit["sentence_ratio"] = safe_float(sentence_focus.get("sentence_ratio"))
    hit["sentence_compactness"] = safe_float(sentence_focus.get("sentence_compactness"))
    hit["sentence_exact"] = bool(sentence_focus.get("sentence_exact"))
    hit["sentence_explanatory"] = bool(sentence_focus.get("sentence_explanatory"))
    hit["sentence_len"] = safe_int(sentence_focus.get("sentence_len"))
    return hit


def localization_score_for_hit(
    query: str,
    hit: Dict[str, Any],
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> float:
    stats = query_token_stats(query, get_conn_fn=get_conn_fn)
    stats["query"] = query
    profile = infer_localization_query_profile(query)
    annotated = annotate_hit_query_support(dict(hit), query_stats=stats)
    return _localization_score_for_annotated_hit(annotated, profile=profile)


def _localization_score_for_annotated_hit(hit: Dict[str, Any], *, profile: Dict[str, bool]) -> float:
    score = safe_float(hit.get("match_score"))
    score += min(safe_int(hit.get("content_hits")), 4) * 0.015
    score += min(safe_int(hit.get("lex_hits")), 4) * 0.008
    score += min(safe_float(hit.get("block_match_score")), 12.0) * 0.002
    score += min(safe_int(hit.get("sentence_hits")), 4) * 0.010
    score += min(1.0, safe_float(hit.get("sentence_compactness"))) * 0.055

    exact_phrase = bool(hit.get("exact_phrase"))
    if exact_phrase:
        score += 0.08

    sentence_exact = bool(hit.get("sentence_exact"))
    sentence_explanatory = bool(hit.get("sentence_explanatory"))
    sentence_ratio = safe_float(hit.get("sentence_ratio"))
    sentence_len = safe_int(hit.get("sentence_len"))
    if sentence_exact:
        score += 0.035
    if sentence_explanatory and sentence_ratio >= 0.75:
        score += 0.05
    if sentence_len and sentence_len <= 180 and sentence_ratio >= 0.75:
        score += 0.03

    bucket = str(hit.get("search_bucket") or "body").strip().lower()
    if bucket == "references":
        score -= 1.0
    elif bucket == "front_matter":
        score -= 0.10

    role = infer_localization_section_role(hit.get("match_section_canonical"))
    if role == "abstract":
        if profile.get("overview_like"):
            score -= 0.01
            if sentence_ratio >= 0.75:
                score += 0.03
            if sentence_explanatory and sentence_len and sentence_len <= 180:
                score += 0.03
            if exact_phrase:
                score += 0.02
        else:
            score -= 0.05
            if exact_phrase:
                score += 0.02
    elif role == "intro":
        if profile.get("overview_like") and sentence_ratio >= 0.75:
            score -= 0.005
        else:
            score -= 0.03
    elif role == "conclusion":
        score -= 0.05
        if profile.get("analysis_like") or profile.get("result_like"):
            score += 0.015
    elif role == "low_salience":
        score -= 0.10
    elif role == "method":
        if profile.get("method_like"):
            score += 0.08
        else:
            score += 0.01
    elif role == "evaluation":
        if profile.get("result_like") or profile.get("benchmark_like"):
            score += 0.09
        elif profile.get("analysis_like"):
            score += 0.04
        else:
            score += 0.02

    if safe_float(hit.get("weighted_hit_ratio")) >= 0.85:
        score += 0.02
    if safe_int(hit.get("rare_hits")) > 0:
        score += 0.015
    return score


def _rerank_section_hits_for_localization_with_stats(
    hits: List[Dict[str, Any]],
    *,
    query_stats: Dict[str, Any],
    profile: Dict[str, bool],
) -> List[Dict[str, Any]]:
    reranked: List[Dict[str, Any]] = []
    for hit in hits:
        annotated = annotate_hit_query_support(dict(hit), query_stats=query_stats)
        annotated["localization_score"] = _localization_score_for_annotated_hit(annotated, profile=profile)
        reranked.append(annotated)
    reranked.sort(
        key=lambda item: (
            safe_float(item.get("localization_score")),
            bool(item.get("exact_phrase")),
            safe_int(item.get("content_hits")),
            safe_int(item.get("lex_hits")),
            safe_float(item.get("match_score")),
        ),
        reverse=True,
    )
    return reranked


def rerank_section_hits_for_localization(
    query: str,
    hits: List[Dict[str, Any]],
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> List[Dict[str, Any]]:
    stats = query_token_stats(query, get_conn_fn=get_conn_fn)
    stats["query"] = query
    profile = infer_localization_query_profile(query)
    return _rerank_section_hits_for_localization_with_stats(hits, query_stats=stats, profile=profile)


def search_paper_sections_for_localization(
    query: str,
    search_type: str,
    paper_id: int,
    *,
    keyword_section_hits_fn: KeywordSectionHitsFn,
    semantic_section_hits_fn: SemanticSectionHitsFn,
    include_text: bool,
    max_chars: Optional[int],
    limit: int = 100,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> List[Dict[str, Any]]:
    hits = search_section_hits_unified(
        query,
        search_type,
        keyword_section_hits_fn=keyword_section_hits_fn,
        semantic_section_hits_fn=semantic_section_hits_fn,
        paper_ids=[paper_id],
        include_text=include_text,
        max_chars=max_chars,
        limit=limit,
    )
    hits = filter_section_hits_for_query(query, hits, get_conn_fn=get_conn_fn)
    reranked = rerank_section_hits_for_localization(query, hits, get_conn_fn=get_conn_fn)
    _trace_search(
        "localization_hits",
        query=query,
        search_type=search_type,
        paper_id=int(paper_id),
        total=len(reranked),
        top_hits=_summarize_hits_for_trace(reranked),
    )
    return reranked


def section_passes_search_gate(
    hit: Dict[str, Any],
    *,
    token_count: int,
    content_tokens: List[str],
) -> bool:
    bucket = str(hit.get("search_bucket") or "body").strip().lower()
    exact_phrase = bool(hit.get("exact_phrase"))
    lex_hits = safe_int(hit.get("lex_hits"))
    keyword_score = safe_float(hit.get("keyword_score"))
    semantic_raw = safe_float(hit.get("semantic_raw_score"))
    block_match_score = safe_float(hit.get("block_match_score"))
    match_score = safe_float(hit.get("match_score"))
    content_hits = safe_int(hit.get("content_hits"))
    content_token_count = max(0, safe_int(hit.get("content_token_count")))
    rare_hits = safe_int(hit.get("rare_hits"))
    rare_token_count = max(0, safe_int(hit.get("rare_token_count")))
    weighted_hit_ratio = safe_float(hit.get("weighted_hit_ratio"))
    min_hits = min_lex_hits_for_query(token_count)
    min_content_hits = 1 if content_token_count <= 2 else 2
    min_rare_hits = 1 if rare_token_count <= 2 else 2

    if bucket == "references":
        return False

    if exact_phrase:
        return True

    if content_token_count == 0 and lex_hits >= min_hits:
        return True

    if bucket == "front_matter":
        return keyword_score >= 0.12 and content_hits >= 1

    if bucket != "body":
        return False

    if content_hits >= min_content_hits and lex_hits >= min_hits:
        if rare_token_count == 0:
            return True
        if rare_hits >= min_rare_hits:
            return True
        if weighted_hit_ratio >= 0.78 and match_score >= 0.10:
            return True
        return False

    if rare_token_count > 0 and rare_hits == 0:
        return False

    if is_low_salience_section(hit.get("match_section_canonical")) and not exact_phrase:
        return False

    if content_hits >= 1 and weighted_hit_ratio < 0.52:
        return False

    if content_hits >= 1 and match_score >= 0.085 and (
        rare_token_count == 0 or rare_hits >= 1
    ) and (
        lex_hits >= max(1, min_hits - 1)
        or block_match_score >= 4.5
        or keyword_score >= 0.05
        or semantic_raw >= 0.012
    ):
        return True

    return False


def filter_section_hits_for_query(
    query: str,
    hits: List[Dict[str, Any]],
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> List[Dict[str, Any]]:
    token_count = len(query_tokens(query))
    stats = query_token_stats(query, get_conn_fn=get_conn_fn)
    filtered: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for hit in hits:
        annotated = annotate_hit_query_support(hit, query_stats=stats)
        if section_passes_search_gate(
            annotated,
            token_count=token_count,
            content_tokens=list(stats.get("content_tokens") or []),
        ):
            filtered.append(annotated)
        else:
            dropped.append(annotated)
    _trace_search(
        "section_hits_filtered",
        query=query,
        input_count=len(hits),
        kept_count=len(filtered),
        dropped_count=len(dropped),
        content_tokens=list(stats.get("content_tokens") or []),
        rare_tokens=list(stats.get("rare_tokens") or []),
        kept_hits=_summarize_hits_for_trace(filtered),
        dropped_hits=_summarize_hits_for_trace(dropped),
    )
    return filtered


def paper_passes_search_gate(
    query: str,
    paper_meta: Dict[str, Any],
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> bool:
    tokens = query_tokens(query)
    stats = query_token_stats(query, get_conn_fn=get_conn_fn)
    content_tokens = list(stats.get("content_tokens") or [])
    rare_tokens = list(stats.get("rare_tokens") or [])
    token_count = len(tokens)
    min_hits = min_lex_hits_for_query(token_count)
    min_content_hits = 1 if len(content_tokens) <= 2 else 2
    min_rare_hits = 1 if len(rare_tokens) <= 2 else 2
    title_bonus = safe_float(paper_meta.get("title_bonus"))
    support_hits = list(paper_meta.get("support_hits") or [])
    support_hits = [
        annotate_hit_query_support(hit, query_stats=stats)
        for hit in support_hits
    ]
    support_hits = [
        hit
        for hit in support_hits
        if section_passes_search_gate(hit, token_count=token_count, content_tokens=content_tokens)
    ]

    if not support_hits:
        return title_bonus >= _TITLE_ONLY_RESCUE_MIN_SCORE

    body_hits = [hit for hit in support_hits if str(hit.get("search_bucket") or "body").strip().lower() == "body"]
    if body_hits:
        best_body = body_hits[0]
        corroborating_body_hits = [
            hit
            for hit in body_hits[1:]
            if not is_low_salience_section(hit.get("match_section_canonical"))
        ]
        best_low_salience = is_low_salience_section(best_body.get("match_section_canonical"))

        if best_low_salience and title_bonus < 0.10 and not corroborating_body_hits:
            return False

        if bool(best_body.get("exact_phrase")):
            return True
        if (
            safe_int(best_body.get("content_hits")) >= min_content_hits
            and safe_int(best_body.get("lex_hits")) >= min_hits
            and (
                len(rare_tokens) == 0
                or safe_int(best_body.get("rare_hits")) >= min_rare_hits
                or safe_float(best_body.get("weighted_hit_ratio")) >= 0.82
            )
        ):
            return True
        if (
            safe_int(best_body.get("content_hits")) >= 1
            and safe_float(best_body.get("match_score")) >= 0.085
            and (
                len(rare_tokens) == 0
                or safe_int(best_body.get("rare_hits")) >= 1
                or safe_float(best_body.get("weighted_hit_ratio")) >= 0.82
            )
            and (
                title_bonus >= 0.06
                or safe_float(best_body.get("block_match_score")) >= 4.5
                or safe_float(best_body.get("semantic_raw_score")) >= 0.012
            )
        ):
            return True

    best_hit = support_hits[0]
    if title_bonus >= 0.10 and (
        bool(best_hit.get("exact_phrase"))
        or safe_int(best_hit.get("content_hits")) >= min_content_hits
        or (
            safe_int(best_hit.get("content_hits")) >= 1
            and safe_float(best_hit.get("match_score")) >= 0.085
        )
    ):
        return True

    return False


def filter_aggregated_papers_for_query(
    query: str,
    aggregated: Dict[int, Dict[str, Any]],
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> Dict[int, Dict[str, Any]]:
    kept = {
        pid: meta
        for pid, meta in aggregated.items()
        if paper_passes_search_gate(query, meta, get_conn_fn=get_conn_fn)
    }
    rejected_ids = [pid for pid in aggregated.keys() if pid not in kept]
    _trace_search(
        "paper_hits_filtered",
        query=query,
        input_count=len(aggregated),
        kept_count=len(kept),
        dropped_count=len(rejected_ids),
        kept_papers=[
            _summarize_paper_for_trace(pid, meta)
            for pid, meta in sorted(kept.items(), key=lambda item: safe_float(item[1].get("score")), reverse=True)[: _search_trace_limit()]
        ],
        dropped_paper_ids=rejected_ids[: _search_trace_limit()],
    )
    return kept


def paper_title_bonus_lookup(
    query: str,
    limit: int = 100,
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> Dict[int, float]:
    stats = query_token_stats(query, get_conn_fn=get_conn_fn)
    tokens = query_tokens(query)
    content_tokens = list(stats.get("content_tokens") or [])
    rare_tokens = list(stats.get("rare_tokens") or [])
    query_l = (query or "").strip().lower()
    bonuses: Dict[int, float] = {}
    conn_factory = get_conn_fn or _get_conn
    with conn_factory() as conn:
        rows = conn.execute(
            "SELECT id, title, source_url FROM papers ORDER BY id DESC LIMIT ?",
            (max(limit * 10, 1000),),
        ).fetchall()
    for row in rows:
        pid = row["id"]
        if pid is None:
            continue
        title = str(row["title"] or "")
        source_url = str(row["source_url"] or "")
        haystack = f"{title} {source_url}".lower()
        overlap = token_overlap(tokens, haystack)
        content_overlap = token_overlap(content_tokens, haystack)
        rare_overlap = token_overlap(rare_tokens, haystack)
        if not (query_l and query_l in haystack) and content_overlap == 0 and rare_overlap == 0 and overlap < 2:
            continue
        score = 0.0
        if query_l and query_l in haystack:
            score += 0.18
        score += min(overlap, 5) * 0.015
        score += min(content_overlap, 5) * 0.05
        score += rare_overlap * 0.07
        if content_tokens:
            score += min(content_overlap / float(len(content_tokens)), 1.0) * 0.10
        if rare_tokens:
            score += min(rare_overlap / float(len(rare_tokens)), 1.0) * 0.10
        if score <= 0.0:
            continue
        bonuses[int(pid)] = score
    return bonuses


def inject_title_only_candidates(
    aggregated: Dict[int, Dict[str, Any]],
    title_bonus_by_id: Dict[int, float],
    *,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> Dict[int, Dict[str, Any]]:
    if not title_bonus_by_id:
        return aggregated

    rescued_ids = [
        int(pid)
        for pid, bonus in title_bonus_by_id.items()
        if float(bonus or 0.0) >= _TITLE_ONLY_RESCUE_MIN_SCORE and int(pid) not in aggregated
    ]
    if not rescued_ids:
        return aggregated

    conn_factory = get_conn_fn or _get_conn
    placeholders = ",".join("?" for _ in rescued_ids)
    with conn_factory() as conn:
        rows = conn.execute(
            f"""
            SELECT s.id, s.paper_id, s.page_no, s.text
            FROM sections s
            INNER JOIN (
                SELECT paper_id, MIN(page_no) AS page_no
                FROM sections
                WHERE paper_id IN ({placeholders})
                GROUP BY paper_id
            ) first_page
            ON s.paper_id = first_page.paper_id AND s.page_no = first_page.page_no
            ORDER BY s.paper_id, s.id
            """,
            tuple(rescued_ids),
        ).fetchall()

    fallback_hits: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        pid = int(row["paper_id"])
        if pid in fallback_hits:
            continue
        fallback_hits[pid] = {
            "id": int(row["id"]),
            "paper_id": pid,
            "page_no": int(row["page_no"]),
            "match_score": float(title_bonus_by_id.get(pid, 0.0)),
            "keyword_score": 0.0,
            "semantic_score": 0.0,
            "semantic_raw_score": 0.0,
            "block_match_score": 0.0,
            "lex_hits": 0,
            "exact_phrase": False,
            "search_bucket": infer_search_section_bucket(str(row["text"] or ""), page_no=int(row["page_no"])),
            "source_text": str(row["text"] or ""),
            "title_only_match": True,
            "match_text": None,
            "match_section_canonical": None,
        }

    for pid in rescued_ids:
        title_bonus = float(title_bonus_by_id.get(pid, 0.0))
        best_hit = fallback_hits.get(pid)
        aggregated[pid] = {
            "score": title_bonus,
            "best_hit": best_hit or {},
            "support_hits": [best_hit] if best_hit else [],
            "title_bonus": title_bonus,
            "title_only_match": True,
        }
    _trace_search(
        "title_only_rescues",
        rescued_ids=rescued_ids[: _search_trace_limit()],
        total_rescued=len(rescued_ids),
    )
    return aggregated


def merge_section_hits(
    keyword_hits: List[Dict[str, Any]],
    semantic_hits: List[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}

    def upsert(hit: Dict[str, Any]) -> None:
        section_id = int(hit["id"])
        existing = merged.get(section_id)
        if existing is None:
            merged[section_id] = dict(hit)
            return

        existing["keyword_score"] = max(float(existing.get("keyword_score") or 0.0), float(hit.get("keyword_score") or 0.0))
        existing["semantic_score"] = max(float(existing.get("semantic_score") or 0.0), float(hit.get("semantic_score") or 0.0))
        existing["semantic_raw_score"] = max(float(existing.get("semantic_raw_score") or 0.0), float(hit.get("semantic_raw_score") or 0.0))
        existing["block_match_score"] = max(float(existing.get("block_match_score") or 0.0), float(hit.get("block_match_score") or 0.0))
        existing["lex_hits"] = max(int(existing.get("lex_hits") or 0), int(hit.get("lex_hits") or 0))
        existing["exact_phrase"] = bool(existing.get("exact_phrase") or hit.get("exact_phrase"))
        if hit.get("match_bbox") is not None:
            existing["match_bbox"] = hit.get("match_bbox")
        if hit.get("match_block_index") is not None:
            existing["match_block_index"] = hit.get("match_block_index")
        if hit.get("match_section_canonical"):
            existing["match_section_canonical"] = hit.get("match_section_canonical")
        hit_bucket = str(hit.get("search_bucket") or "").strip().lower()
        existing_bucket = str(existing.get("search_bucket") or "").strip().lower()
        if hit_bucket:
            if not existing_bucket:
                existing["search_bucket"] = hit_bucket
            elif section_bucket_multiplier(hit_bucket) < section_bucket_multiplier(existing_bucket):
                existing["search_bucket"] = hit_bucket
        if hit.get("match_text") and not existing.get("match_text"):
            existing["match_text"] = hit.get("match_text")
        if hit.get("source_text") and not existing.get("source_text"):
            existing["source_text"] = hit.get("source_text")
        if hit.get("text") and not existing.get("text"):
            existing["text"] = hit.get("text")

    for hit in keyword_hits:
        upsert(hit)
    for hit in semantic_hits:
        upsert(hit)

    merged_hits: List[Dict[str, Any]] = []
    for entry in merged.values():
        keyword_score = float(entry.get("keyword_score") or 0.0)
        semantic_score = float(entry.get("semantic_score") or 0.0)
        combined = keyword_score + semantic_score
        if keyword_score > 0.0 and semantic_score > 0.0:
            combined += 0.05
        if bool(entry.get("exact_phrase")):
            combined += 0.05
        combined += min(int(entry.get("lex_hits") or 0), 4) * 0.01
        combined *= section_bucket_multiplier(str(entry.get("search_bucket") or "body"))
        entry["match_score"] = combined
        merged_hits.append(entry)

    merged_hits.sort(key=lambda item: item.get("match_score", 0.0), reverse=True)
    return merged_hits[:limit]


def search_section_hits_unified(
    query: str,
    search_type: str,
    *,
    keyword_section_hits_fn: KeywordSectionHitsFn,
    semantic_section_hits_fn: SemanticSectionHitsFn,
    paper_ids: Optional[List[int]] = None,
    include_text: bool,
    max_chars: Optional[int],
    limit: int = 100,
) -> List[Dict[str, Any]]:
    st = search_type or "keyword"
    if st not in {"keyword", "embedding", "hybrid"}:
        st = "keyword"

    keyword_hits: List[Dict[str, Any]] = []
    semantic_hits: List[Dict[str, Any]] = []
    if st in {"keyword", "hybrid"}:
        keyword_hits = keyword_section_hits_fn(
            query,
            paper_ids,
            include_text=include_text,
            max_chars=max_chars,
            limit=limit,
        )
    if st in {"embedding", "hybrid"}:
        semantic_hits = semantic_section_hits_fn(
            query,
            st if st != "hybrid" else "hybrid",
            paper_ids,
            include_text=include_text,
            max_chars=max_chars,
            limit=limit,
        )
    result: List[Dict[str, Any]]
    if st == "keyword":
        result = keyword_hits[:limit]
    elif st == "embedding":
        result = semantic_hits[:limit]
    else:
        result = merge_section_hits(keyword_hits, semantic_hits, limit=limit)
    _trace_search(
        "section_hits_unified",
        query=query,
        search_type=st,
        paper_ids=[int(pid) for pid in (paper_ids or [])],
        include_text=bool(include_text),
        limit=int(limit),
        keyword_count=len(keyword_hits),
        semantic_count=len(semantic_hits),
        result_count=len(result),
        keyword_hits=_summarize_hits_for_trace(keyword_hits),
        semantic_hits=_summarize_hits_for_trace(semantic_hits),
        result_hits=_summarize_hits_for_trace(result),
    )
    return result


def aggregate_section_hits_to_papers(
    section_hits: List[Dict[str, Any]],
    title_bonus_by_id: Optional[Dict[int, float]] = None,
    *,
    query: Optional[str] = None,
    get_conn_fn: Optional[ConnectionFactory] = None,
) -> Dict[int, Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for hit in section_hits:
        pid = int(hit["paper_id"])
        grouped.setdefault(pid, []).append(hit)

    aggregated: Dict[int, Dict[str, Any]] = {}
    for pid, hits in grouped.items():
        ranked = sorted(hits, key=lambda item: item.get("match_score", 0.0), reverse=True)
        weights = (1.0, 0.6, 0.35)
        score = 0.0
        support_slice = ranked[: len(weights)]
        buckets = [str(hit.get("search_bucket") or "body") for hit in support_slice]
        for idx, hit in enumerate(support_slice):
            score += weights[idx] * float(hit.get("match_score") or 0.0)
        unique_pages = len({int(hit.get("page_no") or 0) for hit in ranked[: len(weights)] if hit.get("page_no")})
        if unique_pages > 1:
            score += min(unique_pages - 1, 2) * 0.02
        if support_slice:
            if all(bucket == "references" for bucket in buckets):
                score *= 0.12
            elif buckets[0] == "references" and "body" not in buckets:
                score *= 0.22
            elif all(bucket == "front_matter" for bucket in buckets):
                score *= 0.50
            elif buckets[0] == "front_matter" and "body" not in buckets:
                score *= 0.70
        title_bonus = float((title_bonus_by_id or {}).get(pid, 0.0))
        score += title_bonus
        aggregated[pid] = {
            "score": score,
            "best_hit": ranked[0],
            "ranking_best_hit": ranked[0],
            "support_hits": ranked[:3],
            "title_bonus": title_bonus,
        }
    _trace_search(
        "paper_hits_aggregated",
        query=query,
        input_section_count=len(section_hits),
        paper_count=len(aggregated),
        top_papers=[
            _summarize_paper_for_trace(pid, meta)
            for pid, meta in sorted(aggregated.items(), key=lambda item: safe_float(item[1].get("score")), reverse=True)[: _search_trace_limit()]
        ],
    )
    return aggregated


__all__ = [
    "configure_connection_factory",
    "rrf_score",
    "token_overlap",
    "infer_search_section_bucket",
    "section_bucket_multiplier",
    "query_token_stats",
    "infer_localization_query_profile",
    "infer_localization_section_role",
    "paper_title_bonus_lookup",
    "annotate_hit_query_support",
    "localization_score_for_hit",
    "rerank_section_hits_for_localization",
    "search_paper_sections_for_localization",
    "section_passes_search_gate",
    "filter_section_hits_for_query",
    "paper_passes_search_gate",
    "filter_aggregated_papers_for_query",
    "inject_title_only_candidates",
    "merge_section_hits",
    "search_section_hits_unified",
    "aggregate_section_hits_to_papers",
]
