from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")


def normalize_query(query: str) -> str:
    return _WS_RE.sub(" ", query).strip()


def plain_snippet(text: str, query: str, *, radius: int = 160) -> str:
    normalized_text = _WS_RE.sub(" ", text).strip()
    if not normalized_text:
        return ""
    query = normalize_query(query)
    if not query:
        return normalized_text[: radius * 2].strip()
    match = re.search(re.escape(query), normalized_text, flags=re.IGNORECASE)
    if not match:
        return normalized_text[: radius * 2].strip()
    start = max(0, match.start() - radius)
    end = min(len(normalized_text), match.end() + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(normalized_text) else ""
    fragment = normalized_text[start:end].strip()
    highlighted = re.sub(
        re.escape(query),
        lambda found: f"<mark>{found.group(0)}</mark>",
        fragment,
        flags=re.IGNORECASE,
    )
    return f"{prefix}{highlighted}{suffix}"


def clean_headline(headline: str, fallback_text: str, query: str) -> str:
    headline = _WS_RE.sub(" ", headline or "").strip()
    # ts_headline returns an empty-ish fragment for pure ILIKE fallback matches.
    if headline and _TAG_RE.sub("", headline).strip():
        return headline
    return plain_snippet(fallback_text, query)
