from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_DASH_NORMALIZE_RE = re.compile(r"[‐‑‒–—―-]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_EXPLANATORY_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|be|performs?|uses?|builds?|consists?|means?|refers?|denotes?|aligns?|maps?|learns?|predicts?|shows?|provides?)\b",
    re.I,
)


def query_tokens(query: str) -> List[str]:
    if not query:
        return []
    tokens = [t.lower() for t in _WORD_RE.findall(query) if len(t) > 2]
    return list(dict.fromkeys(tokens))


def lexical_hits(tokens: List[str], text: str) -> int:
    if not tokens or not text:
        return 0
    text_lower = text.lower()
    return sum(1 for token in tokens if token in text_lower)


def _normalize_loose_text(value: str) -> str:
    if not value:
        return ""
    text = value.lower()
    text = _DASH_NORMALIZE_RE.sub("-", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def _ordered_token_coverage(tokens: List[str], normalized_text: str) -> float:
    if not tokens or not normalized_text:
        return 0.0
    pos = 0
    matched = 0
    for token in tokens:
        idx = normalized_text.find(token, pos)
        if idx == -1:
            continue
        matched += 1
        pos = idx + len(token)
    return matched / max(1, len(tokens))


def _base_query_match_features(query: str, tokens: List[str], text: str) -> Dict[str, Any]:
    token_hits = lexical_hits(tokens, text)
    if not text:
        return {
            "score": 0.0,
            "lex_hits": token_hits,
            "exact_phrase": False,
            "normalized_text": "",
            "normalized_query": "",
        }

    normalized_text = _normalize_loose_text(text)
    normalized_query = _normalize_loose_text(query)
    exact_phrase = bool(normalized_query and normalized_query in normalized_text)

    token_ratio = (token_hits / len(tokens)) if tokens else 0.0
    ordered_ratio = _ordered_token_coverage(tokens, normalized_text)

    bigram_hits = 0
    if len(tokens) >= 2:
        for idx in range(len(tokens) - 1):
            if f"{tokens[idx]} {tokens[idx + 1]}" in normalized_text:
                bigram_hits += 1

    score = (token_hits * 1.25) + (token_ratio * 1.1) + (ordered_ratio * 1.4) + (bigram_hits * 0.8)
    if exact_phrase:
        score += 8.0
    elif normalized_query and len(normalized_query) >= 20 and normalized_text:
        short_text = normalized_text[: max(260, len(normalized_query) * 2)]
        similarity = SequenceMatcher(None, normalized_query, short_text).ratio()
        if similarity >= 0.55:
            score += (similarity - 0.5) * 2.0

    return {
        "score": score,
        "lex_hits": token_hits,
        "exact_phrase": exact_phrase,
        "normalized_text": normalized_text,
        "normalized_query": normalized_query,
    }


def _best_token_window(tokens: List[str], normalized_text: str) -> Dict[str, float]:
    if not tokens or not normalized_text:
        return {"matched": 0, "window_tokens": 0.0, "window_chars": 0.0}

    occurrences: List[tuple[int, int]] = []
    for token in dict.fromkeys(tokens):
        start = normalized_text.find(token)
        if start == -1:
            continue
        occurrences.append((start, start + len(token)))
    if not occurrences:
        return {"matched": 0, "window_tokens": 0.0, "window_chars": 0.0}

    occurrences.sort()
    token_count = max(1, len(tokens))
    best = {
        "matched": 0,
        "window_tokens": 0.0,
        "window_chars": 0.0,
    }
    for left in range(len(occurrences)):
        seen = 0
        for right in range(left, len(occurrences)):
            seen += 1
            start = occurrences[left][0]
            end = occurrences[right][1]
            window_chars = max(1, end - start)
            token_density = seen / max(1, (right - left + 1))
            window_token_score = (seen / token_count) * token_density
            window_char_score = seen / window_chars
            if (
                seen > best["matched"]
                or (seen == best["matched"] and window_token_score > best["window_tokens"])
                or (
                    seen == best["matched"]
                    and window_token_score == best["window_tokens"]
                    and window_char_score > best["window_chars"]
                )
            ):
                best = {
                    "matched": seen,
                    "window_tokens": window_token_score,
                    "window_chars": window_char_score,
                }
    return best


def _sentence_focus_features(query: str, tokens: List[str], text: str) -> Dict[str, Any]:
    if not text:
        return {
            "sentence_hits": 0,
            "sentence_ratio": 0.0,
            "sentence_exact": False,
            "sentence_compactness": 0.0,
            "sentence_explanatory": False,
            "sentence_len": 0,
        }

    sentences = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(text) if segment.strip()]
    if not sentences:
        sentences = [text.strip()]

    best = {
        "sentence_hits": 0,
        "sentence_ratio": 0.0,
        "sentence_exact": False,
        "sentence_compactness": 0.0,
        "sentence_explanatory": False,
        "sentence_len": 0,
        "_score": -1.0,
    }
    token_count = max(1, len(tokens))
    for sentence in sentences:
        match_eval = _base_query_match_features(query, tokens, sentence)
        hits = int(match_eval.get("lex_hits", 0))
        ratio = hits / token_count
        normalized_sentence = str(match_eval.get("normalized_text") or "")
        window = _best_token_window(tokens, normalized_sentence)
        compactness = min(1.0, float(window["window_tokens"]) * 0.85 + float(window["window_chars"]) * 8.0)
        explanatory = bool(_EXPLANATORY_VERB_RE.search(sentence))
        sentence_score = float(match_eval.get("score", 0.0))
        sentence_score += compactness * 1.4
        if explanatory and hits >= max(2, token_count - 1):
            sentence_score += 0.7
        if len(sentence) <= 180 and hits >= max(2, token_count - 1):
            sentence_score += 0.45
        if (
            sentence_score > best["_score"]
            or (
                sentence_score == best["_score"]
                and bool(match_eval.get("exact_phrase", False))
                and not best["sentence_exact"]
            )
            or (
                sentence_score == best["_score"]
                and bool(match_eval.get("exact_phrase", False)) == best["sentence_exact"]
                and hits > best["sentence_hits"]
            )
        ):
            best = {
                "sentence_hits": hits,
                "sentence_ratio": ratio,
                "sentence_exact": bool(match_eval.get("exact_phrase", False)),
                "sentence_compactness": compactness,
                "sentence_explanatory": explanatory,
                "sentence_len": len(sentence),
                "_score": sentence_score,
            }

    best.pop("_score", None)
    return best


def sentence_focus_features(query: str, tokens: List[str], text: str) -> Dict[str, Any]:
    return _sentence_focus_features(query, tokens, text)


def _query_match_score(query: str, tokens: List[str], text: str) -> Dict[str, Any]:
    base = _base_query_match_features(query, tokens, text)
    score = float(base.get("score", 0.0))

    sentence_focus = _sentence_focus_features(query, tokens, text)
    sentence_ratio = float(sentence_focus.get("sentence_ratio", 0.0))
    sentence_compactness = float(sentence_focus.get("sentence_compactness", 0.0))
    sentence_hits = int(sentence_focus.get("sentence_hits", 0))
    sentence_exact = bool(sentence_focus.get("sentence_exact", False))
    sentence_explanatory = bool(sentence_focus.get("sentence_explanatory", False))
    sentence_len = int(sentence_focus.get("sentence_len", 0))

    score += sentence_ratio * 0.8
    score += sentence_compactness * 1.6
    if sentence_exact:
        score += 1.2
    if sentence_explanatory and sentence_hits >= max(2, len(tokens) - 1):
        score += 1.0
    if sentence_len and sentence_len <= 180 and sentence_hits >= max(2, len(tokens) - 1):
        score += 0.6

    return {
        "score": score,
        "lex_hits": int(base.get("lex_hits", 0)),
        "exact_phrase": bool(base.get("exact_phrase", False)),
        "sentence_hits": sentence_hits,
        "sentence_ratio": sentence_ratio,
        "sentence_compactness": sentence_compactness,
        "sentence_exact": sentence_exact,
        "sentence_explanatory": sentence_explanatory,
        "sentence_len": sentence_len,
    }


def _coalesce_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _bbox_from_payload(value: Any) -> Optional[Dict[str, float]]:
    if isinstance(value, dict):
        try:
            x0 = float(value["x0"])
            y0 = float(value["y0"])
            x1 = float(value["x1"])
            y1 = float(value["y1"])
        except (KeyError, TypeError, ValueError):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            x0 = float(value[0])
            y0 = float(value[1])
            x1 = float(value[2])
            y1 = float(value[3])
        except (TypeError, ValueError):
            return None
        if x1 <= x0 or y1 <= y0:
            return None
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    return None


def _bbox_union(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not a:
        return b
    if not b:
        return a
    return {
        "x0": min(float(a["x0"]), float(b["x0"])),
        "y0": min(float(a["y0"]), float(b["y0"])),
        "x1": max(float(a["x1"]), float(b["x1"])),
        "y1": max(float(a["y1"]), float(b["y1"])),
    }


def _line_text_segments(line: Dict[str, Any]) -> List[Dict[str, Any]]:
    spans = line.get("spans")
    if not isinstance(spans, list):
        return []
    segments: List[Dict[str, Any]] = []
    pieces: List[str] = []
    for span in spans:
        text = str(span.get("text") or "")
        if not text.strip():
            continue
        if pieces:
            pieces.append(" ")
        start = len("".join(pieces))
        pieces.append(text)
        end = len("".join(pieces))
        bbox = _bbox_from_payload(span.get("bbox"))
        if not bbox:
            continue
        segments.append({"start": start, "end": end, "bbox": bbox, "text": text})
    return segments


def _slice_segment_bbox(segment: Dict[str, Any], start: int, end: int) -> Optional[Dict[str, float]]:
    bbox = segment.get("bbox")
    text = str(segment.get("text") or "")
    if not bbox or not text:
        return bbox
    seg_start = int(segment.get("start") or 0)
    seg_end = int(segment.get("end") or seg_start)
    if seg_end <= seg_start:
        return bbox
    local_start = max(0, start - seg_start)
    local_end = min(len(text), end - seg_start)
    if local_end <= local_start:
        return None
    width = max(0.0, float(bbox["x1"]) - float(bbox["x0"]))
    char_count = max(1, len(text))
    x0 = float(bbox["x0"]) + width * (local_start / char_count)
    x1 = float(bbox["x0"]) + width * (local_end / char_count)
    if x1 <= x0:
        return bbox
    return {
        "x0": x0,
        "y0": float(bbox["y0"]),
        "x1": x1,
        "y1": float(bbox["y1"]),
    }


def _phrase_bbox_from_line(line: Dict[str, Any], query: str, tokens: List[str]) -> Optional[Dict[str, float]]:
    line_text = str(line.get("text") or "")
    if not line_text:
        return _bbox_from_payload(line.get("bbox"))

    line_bbox = _bbox_from_payload(line.get("bbox"))
    segments = _line_text_segments(line)
    lowered = line_text.lower()
    query_l = query.strip().lower()

    if query_l and len(query_l) > 1:
        idx = lowered.find(query_l)
        if idx != -1 and segments:
            end = idx + len(query_l)
            phrase_bbox: Optional[Dict[str, float]] = None
            for segment in segments:
                if int(segment["end"]) <= idx or int(segment["start"]) >= end:
                    continue
                phrase_bbox = _bbox_union(phrase_bbox, _slice_segment_bbox(segment, idx, end))
            if phrase_bbox:
                return phrase_bbox

    if tokens and segments:
        token_bbox: Optional[Dict[str, float]] = None
        search_from = 0
        for token in tokens:
            idx = lowered.find(token, search_from)
            if idx == -1:
                idx = lowered.find(token)
            if idx == -1:
                continue
            end = idx + len(token)
            for segment in segments:
                if int(segment["end"]) <= idx or int(segment["start"]) >= end:
                    continue
                token_bbox = _bbox_union(token_bbox, _slice_segment_bbox(segment, idx, end))
            search_from = end
        if token_bbox:
            return token_bbox

    return line_bbox


def _phrase_bbox_for_block(block: Dict[str, Any], query: str, tokens: List[str]) -> Optional[Dict[str, float]]:
    block_meta = block.get("metadata")
    lines = block_meta.get("lines") if isinstance(block_meta, dict) else None
    if not isinstance(lines, list) or not lines:
        return _bbox_from_payload(block.get("bbox"))

    best_line_bbox: Optional[Dict[str, float]] = None
    best_line_score = -1.0
    best_line_exact = False
    best_line_hits = -1
    for line in lines:
        line_text = str(line.get("text") or "")
        match_eval = _query_match_score(query, tokens, line_text)
        score = float(match_eval.get("score", 0.0))
        hits = int(match_eval.get("lex_hits", 0))
        exact = bool(match_eval.get("exact_phrase", False))
        if (
            score > best_line_score
            or (score == best_line_score and exact and not best_line_exact)
            or (score == best_line_score and exact == best_line_exact and hits > best_line_hits)
        ):
            best_line_bbox = _phrase_bbox_from_line(line, query, tokens)
            best_line_score = score
            best_line_exact = exact
            best_line_hits = hits

    return best_line_bbox or _bbox_from_payload(block.get("bbox"))


def pgvector_score(row: Dict[str, Any]) -> float:
    for key in ("hybrid_score", "similarity", "score"):
        val = row.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return 0.0


def select_block_for_query(row: Dict[str, Any], tokens: List[str], query: str = "") -> Dict[str, Any]:
    metadata = row.get("metadata")
    blocks = metadata.get("blocks") if isinstance(metadata, dict) else None
    if not blocks:
        section_canonical = ""
        if isinstance(metadata, dict):
            section_canonical = str(
                metadata.get("section_primary")
                or metadata.get("section_canonical")
                or ""
            ).strip()
        fallback_eval = _query_match_score(query, tokens, row.get("text") or "")
        return {
            "page_no": row.get("page_no"),
            "block_index": row.get("block_index"),
            "bbox": row.get("bbox"),
            "text": row.get("text") or "",
            "lex_hits": lexical_hits(tokens, row.get("text") or ""),
            "match_score": fallback_eval.get("score", 0.0),
            "exact_phrase": fallback_eval.get("exact_phrase", False),
            "section_canonical": section_canonical,
        }

    best_block: Optional[Dict[str, Any]] = None
    best_match_score = -1.0
    best_hits = -1
    best_len = -1
    best_exact_phrase = False
    for block in blocks:
        text = block.get("text") or ""
        match_eval = _query_match_score(query, tokens, text)
        match_score = float(match_eval.get("score", 0.0))
        hits = int(match_eval.get("lex_hits", 0))
        exact_phrase = bool(match_eval.get("exact_phrase", False))
        if (
            match_score > best_match_score
            or (match_score == best_match_score and exact_phrase and not best_exact_phrase)
            or (match_score == best_match_score and exact_phrase == best_exact_phrase and hits > best_hits)
            or (
                match_score == best_match_score
                and hits == best_hits
                and len(text) > best_len
            )
        ):
            best_block = block
            best_match_score = match_score
            best_hits = hits
            best_len = len(text)
            best_exact_phrase = exact_phrase

    if not best_block:
        section_canonical = ""
        if isinstance(metadata, dict):
            section_canonical = str(
                metadata.get("section_primary")
                or metadata.get("section_canonical")
                or ""
            ).strip()
        fallback_eval = _query_match_score(query, tokens, row.get("text") or "")
        return {
            "page_no": row.get("page_no"),
            "block_index": row.get("block_index"),
            "bbox": row.get("bbox"),
            "text": row.get("text") or "",
            "lex_hits": lexical_hits(tokens, row.get("text") or ""),
            "match_score": fallback_eval.get("score", 0.0),
            "exact_phrase": fallback_eval.get("exact_phrase", False),
            "section_canonical": section_canonical,
        }

    block_meta = best_block.get("metadata") if isinstance(best_block, dict) else None
    section_canonical = ""
    if isinstance(block_meta, dict):
        section_canonical = str(
            block_meta.get("section_canonical")
            or block_meta.get("section_primary")
            or ""
        ).strip()
    if not section_canonical and isinstance(metadata, dict):
        section_canonical = str(
            metadata.get("section_primary")
            or metadata.get("section_canonical")
            or ""
        ).strip()

    return {
        "page_no": _coalesce_not_none(best_block.get("page_no"), row.get("page_no")),
        "block_index": _coalesce_not_none(best_block.get("block_index"), row.get("block_index")),
        "bbox": _coalesce_not_none(
            _phrase_bbox_for_block(best_block, query, tokens),
            best_block.get("bbox"),
            row.get("bbox"),
        ),
        "text": best_block.get("text") or row.get("text") or "",
        "lex_hits": best_hits,
        "match_score": best_match_score,
        "exact_phrase": best_exact_phrase,
        "section_canonical": section_canonical,
    }


def build_match_snippet(query: str, tokens: List[str], text: str, max_len: int = 240) -> str:
    if not text:
        return ""
    clean = " ".join(text.replace("\x00", "").split())
    if not clean:
        return ""
    if not tokens:
        return clean[:max_len]
    lower = clean.lower()
    target_tokens = [t for t in tokens if t in lower]
    if target_tokens:
        words = [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(clean)]
        target_set = set(target_tokens)
        needed = len(target_set)
        counts: Dict[str, int] = {}
        have = 0
        best_window: Optional[tuple[int, int]] = None
        left = 0
        for right, (token, _start, end) in enumerate(words):
            if token in target_set:
                counts[token] = counts.get(token, 0) + 1
                if counts[token] == 1:
                    have += 1
            while have == needed and left <= right:
                window_start = words[left][1]
                window_end = end
                if best_window is None or (window_end - window_start) < (best_window[1] - best_window[0]):
                    best_window = (window_start, window_end)
                left_token = words[left][0]
                if left_token in target_set:
                    counts[left_token] -= 1
                    if counts[left_token] == 0:
                        have -= 1
                left += 1
        if best_window:
            pad = 12
            start = max(0, best_window[0] - pad)
            end = min(len(clean), best_window[1] + pad)
            snippet = clean[start:end]
            if len(snippet) <= max_len:
                return snippet
            clean = snippet
            lower = clean.lower()

    tokens_sorted = sorted(tokens, key=len, reverse=True)
    anchor = -1
    for token in tokens_sorted:
        idx = lower.find(token)
        if idx != -1:
            anchor = idx
            break
    if anchor == -1:
        return clean[:max_len]
    target_len = min(max_len, max(60, len(query) * 2))
    start = max(0, anchor - target_len // 4)
    if start > 0:
        prior_space = clean.rfind(" ", 0, start)
        if prior_space != -1:
            start = prior_space + 1
    end = min(len(clean), start + target_len)
    if end < len(clean):
        next_space = clean.find(" ", end)
        if next_space != -1:
            end = next_space
    return clean[start:end]


__all__ = [
    "query_tokens",
    "lexical_hits",
    "pgvector_score",
    "select_block_for_query",
    "build_match_snippet",
]
