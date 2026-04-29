from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .models import MarkdownRenderAudit

_HEADING_RE = re.compile(r"^(#{2,6})\s+(.*?)\s*$")
_CORE_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "related works",
    "method",
    "methods",
    "methodology",
    "experiment",
    "experiments",
    "results",
    "conclusion",
    "conclusions",
    "discussion",
    "discussions",
    "appendix",
    "references",
}


def audit_rendered_markdown(
    markdown: str,
    *,
    metadata: Dict[str, Any],
    blocks: Optional[List[Dict[str, Any]]] = None,
) -> MarkdownRenderAudit:
    headings: List[tuple[int, str, str]] = []
    for line in str(markdown or "").splitlines():
        match = _HEADING_RE.match(line.strip())
        if not match:
            continue
        title = str(match.group(2) or "").strip()
        if not title:
            continue
        headings.append((len(match.group(1)), title, _normalize_heading(title)))

    counts = Counter(normalized for _, _, normalized in headings if normalized)
    suspicious = [title for _, title, normalized in headings if _looks_like_suspicious_heading(title, normalized=normalized, metadata=metadata)]
    issues: List[str] = []

    page_count = _safe_int(metadata.get("page_count"), 0)
    total_headings = len(headings)
    max_reasonable_headings = max(24, page_count * 2 + 6) if page_count > 0 else 24
    if total_headings > max_reasonable_headings:
        issues.append(f"too many headings ({total_headings}) for page count {page_count or 'unknown'}")

    for normalized, count in sorted(counts.items()):
        if not normalized:
            continue
        if normalized in _CORE_HEADINGS and count > 1:
            issues.append(f"duplicate core heading '{normalized}' ({count})")
        elif count > 3:
            issues.append(f"heading repeated excessively '{normalized}' ({count})")

    if suspicious:
        issues.append(f"suspicious headings detected ({len(suspicious)})")

    missing_prose_runs = _detect_consecutive_heading_runs_without_prose(markdown, blocks=blocks)
    if missing_prose_runs:
        issues.append(f"consecutive heading runs missing prose ({len(missing_prose_runs)})")

    return MarkdownRenderAudit(
        conservative_recommended=bool(issues),
        issue_count=len(issues),
        issues=issues,
        suspicious_headings=suspicious[:12],
        heading_counts=dict(sorted(counts.items())),
        total_headings=total_headings,
    )


def _normalize_heading(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^\d+(?:\.\d+)*[\)\.]?\s+", "", text)
    text = re.sub(r"^[a-z](?:\.\d+)*[\)\.]?\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_suspicious_heading(title: str, *, normalized: str, metadata: Dict[str, Any]) -> bool:
    compact = " ".join(str(title or "").split()).strip()
    lowered = compact.lower()
    word_count = len(re.findall(r"[A-Za-z]{2,}", compact))
    title_norm = _normalize_heading(str(metadata.get("title") or ""))

    if not compact:
        return True
    if "{" in compact or "}" in compact:
        return True
    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        return True
    if title_norm and normalized == title_norm:
        return True
    if re.search(r"\\[A-Za-z]+", compact):
        return True
    if compact.startswith(("Figure ", "Table ", "Fig. ", "Eq. ")):
        return True
    if _looks_like_equationish_heading(compact):
        return True
    if word_count > 14:
        return True
    if _looks_like_sentenceish_prose(compact) and word_count > 6:
        return True
    return False


def _looks_like_equationish_heading(text: str) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return False
    if any(marker in compact for marker in ("$$", "\\begin", "\\end")):
        return True
    if re.search(r"[=<>^_]|\\[A-Za-z]+|\{|\}", compact):
        return True
    math_tokens = len(re.findall(r"[=+\-*/^_<>∥∑∫\\()\[\]{}]", compact))
    alpha_words = len(re.findall(r"[A-Za-z]{2,}", compact))
    return math_tokens >= 3 and alpha_words >= 2


def _looks_like_sentenceish_prose(text: str) -> bool:
    lowered = str(text or "").lower()
    if re.search(r"[.!?]\s", text):
        return True
    stopwords = re.findall(r"\b(the|and|that|with|from|this|these|our|their|which|while|using|without|into|through|because|however|although|where|when)\b", lowered)
    return len(stopwords) >= 3


def _detect_consecutive_heading_runs_without_prose(
    markdown: str,
    *,
    blocks: Optional[List[Dict[str, Any]]],
) -> List[str]:
    if not blocks:
        return []

    lines = str(markdown or "").splitlines()
    heading_runs: List[List[tuple[int, str, str]]] = []
    current_run: List[tuple[int, str, str]] = []
    for line in lines:
        stripped = line.strip()
        match = _HEADING_RE.match(stripped)
        if match:
            title = str(match.group(2) or "").strip()
            normalized = _normalize_heading(title)
            level = len(match.group(1))
            if current_run and current_run[-1][0] != level:
                if len(current_run) >= 2:
                    heading_runs.append(list(current_run))
                current_run = []
            current_run.append((level, title, normalized))
            continue
        if stripped:
            if len(current_run) >= 2:
                heading_runs.append(list(current_run))
            current_run = []
            continue
        continue

    if current_run and len(current_run) >= 2:
        heading_runs.append(list(current_run))

    suspicious_runs: List[str] = []
    for run in heading_runs:
        if len(run) < 2:
            continue
        if len({level for level, _, _ in run}) != 1:
            continue
        if run[0][0] < 3:
            continue
        if _blocks_show_intervening_prose_for_headings(run, blocks=blocks):
            suspicious_runs.append(" -> ".join(title for _, title, _ in run))
    return suspicious_runs[:8]


def _blocks_show_intervening_prose_for_headings(
    heading_run: List[tuple[int, str, str]],
    *,
    blocks: List[Dict[str, Any]],
) -> bool:
    heading_matches: List[Dict[str, Any]] = []
    used_indexes: set[int] = set()
    for _, _, normalized in heading_run:
        match = _find_heading_block(normalized, blocks=blocks, used_indexes=used_indexes)
        if match is None:
            return False
        heading_matches.append(match)
        used_indexes.add(match["index"])

    pages = {item["page_no"] for item in heading_matches}
    if len(pages) != 1:
        return False
    page_no = next(iter(pages))
    y0 = min(item["y0"] for item in heading_matches)
    y1 = max(item["y1"] for item in heading_matches)
    if y1 <= y0:
        return False

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if _safe_int(block.get("page_no"), 0) != page_no:
            continue
        text = " ".join(str(block.get("text") or "").split()).strip()
        if not text or len(text) < 60:
            continue
        if _normalize_heading(text) in {item["normalized"] for item in heading_matches}:
            continue
        bbox = block.get("bbox") if isinstance(block.get("bbox"), dict) else {}
        block_y0 = _safe_float(bbox.get("y0"), -1.0)
        block_y1 = _safe_float(bbox.get("y1"), -1.0)
        if block_y0 < y0 - 1.0 or block_y1 > y1 + 1.0:
            continue
        if _looks_like_prose_block(text):
            return True
    return False


def _find_heading_block(
    normalized_heading: str,
    *,
    blocks: List[Dict[str, Any]],
    used_indexes: set[int],
) -> Optional[Dict[str, Any]]:
    for idx, block in enumerate(blocks):
        if idx in used_indexes or not isinstance(block, dict):
            continue
        text = " ".join(str(block.get("text") or "").split()).strip()
        if not text:
            continue
        normalized_block = _normalize_heading(text)
        if normalized_block != normalized_heading:
            continue
        bbox = block.get("bbox") if isinstance(block.get("bbox"), dict) else {}
        return {
            "index": idx,
            "page_no": _safe_int(block.get("page_no"), 0),
            "y0": _safe_float(bbox.get("y0"), 0.0),
            "y1": _safe_float(bbox.get("y1"), 0.0),
            "normalized": normalized_block,
        }
    return None


def _looks_like_prose_block(text: str) -> bool:
    if _looks_like_equationish_heading(text):
        return False
    return _looks_like_sentenceish_prose(text)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
