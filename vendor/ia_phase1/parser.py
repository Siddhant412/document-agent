from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qs, urlparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import httpx
import pymupdf
from bs4 import BeautifulSoup
from pypdf import PdfReader

USER_AGENT = "ia-phase1-parser/0.1 (+https://example.local)"
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _default_pdf_dir() -> Path:
    configured = (Path.cwd() / ".ia_phase1_data" / "pdfs")
    configured.mkdir(parents=True, exist_ok=True)
    return configured


def _safe_filename(seed: str) -> str:
    h = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"{h}.pdf"


def _sanitize_extracted_text(value: Any) -> str:
    text = str(value or "").replace("\x00", "")
    return _SURROGATE_RE.sub("\uFFFD", text)


def _looks_like_pdf_bytes(data: bytes) -> bool:
    head = data[:1024].lstrip()
    return b"%PDF-" in head


def describe_google_drive_source(value: str) -> Optional[Dict[str, str]]:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
    except Exception:
        return None
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path_parts = [part for part in (parsed.path or "").split("/") if part]
    query = parse_qs(parsed.query or "")

    def _build_doc_export(kind: str, file_id: str) -> Dict[str, str]:
        if kind == "document":
            return {
                "source_kind": "google_doc_export",
                "file_id": file_id,
                "download_url": f"https://docs.google.com/document/d/{file_id}/export?format=pdf",
            }
        if kind == "spreadsheets":
            return {
                "source_kind": "google_sheet_export",
                "file_id": file_id,
                "download_url": f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=pdf",
            }
        if kind == "presentation":
            return {
                "source_kind": "google_slide_export",
                "file_id": file_id,
                "download_url": f"https://docs.google.com/presentation/d/{file_id}/export/pdf",
            }
        return {}

    if host == "docs.google.com" and len(path_parts) >= 3 and path_parts[1] == "d":
        kind = path_parts[0]
        file_id = path_parts[2]
        if kind in {"document", "spreadsheets", "presentation"} and file_id:
            return _build_doc_export(kind, file_id)

    if host == "drive.google.com":
        file_id = ""
        if len(path_parts) >= 3 and path_parts[0] == "file" and path_parts[1] == "d":
            file_id = path_parts[2]
        elif path_parts[:1] in (["open"], ["uc"]):
            file_id = (query.get("id") or [""])[0]
        elif not path_parts:
            file_id = (query.get("id") or [""])[0]
        if file_id:
            return {
                "source_kind": "google_drive_file",
                "file_id": file_id,
                "download_url": f"https://drive.google.com/uc?export=download&id={file_id}",
            }

    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _span_x0(span: Dict[str, Any]) -> float:
    bbox = span.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 1:
        return _safe_float(bbox[0])
    origin = span.get("origin")
    if isinstance(origin, (list, tuple)) and len(origin) >= 1:
        return _safe_float(origin[0])
    return 0.0


def _span_x1(span: Dict[str, Any]) -> float:
    bbox = span.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 3:
        return _safe_float(bbox[2], _span_x0(span))
    return _span_x0(span)


def _bbox_payload(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {
            "x0": _safe_float(value.get("x0")),
            "y0": _safe_float(value.get("y0")),
            "x1": _safe_float(value.get("x1")),
            "y1": _safe_float(value.get("y1")),
        }
    if isinstance(value, (list, tuple)):
        return {
            "x0": _safe_float(value[0]) if len(value) > 0 else 0.0,
            "y0": _safe_float(value[1]) if len(value) > 1 else 0.0,
            "x1": _safe_float(value[2]) if len(value) > 2 else 0.0,
            "y1": _safe_float(value[3]) if len(value) > 3 else 0.0,
        }
    return {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}


def _bbox_width(bbox: Dict[str, float]) -> float:
    return max(0.0, float(bbox.get("x1", 0.0)) - float(bbox.get("x0", 0.0)))


def _bbox_height(bbox: Dict[str, float]) -> float:
    return max(0.0, float(bbox.get("y1", 0.0)) - float(bbox.get("y0", 0.0)))


def _bbox_center_x(bbox: Dict[str, float]) -> float:
    return (float(bbox.get("x0", 0.0)) + float(bbox.get("x1", 0.0))) * 0.5


def _bbox_centered_near_page_middle(bbox: Dict[str, float], *, page_width: float) -> bool:
    if page_width <= 0.0:
        return False
    center_x = _bbox_center_x(bbox)
    return abs(center_x - (page_width * 0.5)) <= page_width * 0.18


def _block_line_count(block: Dict[str, Any]) -> int:
    return len(block.get("lines") or [])


def _block_text_span_count(block: Dict[str, Any]) -> int:
    count = 0
    for line in block.get("lines") or []:
        for span in line.get("spans") or []:
            text = _sanitize_extracted_text(span.get("text"))
            if text.strip():
                count += 1
    return count


def _block_text_preview(block: Dict[str, Any]) -> str:
    parts: List[str] = []
    for line in block.get("lines") or []:
        joined = _join_line_spans(line.get("spans") or [])
        if joined:
            parts.append(joined)
    return " ".join(parts).strip()


def _is_margin_note_block(
    block: Dict[str, Any],
    *,
    page_width: float,
    page_height: float,
) -> bool:
    bbox = _bbox_payload(block.get("bbox", [0, 0, 0, 0]))
    width = _bbox_width(bbox)
    height = _bbox_height(bbox)
    if width <= 0.0 or height <= 0.0 or page_width <= 0.0 or page_height <= 0.0:
        return False
    width_rel = width / page_width
    height_rel = height / page_height
    x0_rel = float(bbox["x0"]) / page_width
    x1_rel = float(bbox["x1"]) / page_width
    line_count = _block_line_count(block)
    span_count = _block_text_span_count(block)

    if width_rel <= 0.075 and (x0_rel <= 0.06 or x1_rel >= 0.94):
        return True
    if width_rel <= 0.12 and height_rel >= 0.25 and (x0_rel <= 0.08 or x1_rel >= 0.92):
        return True
    if width_rel <= 0.1 and line_count <= 2 and span_count <= 3 and (x0_rel <= 0.08 or x1_rel >= 0.92):
        return True
    return False


def _is_full_width_block(block: Dict[str, Any], *, page_width: float) -> bool:
    bbox = _bbox_payload(block.get("bbox", [0, 0, 0, 0]))
    if page_width <= 0.0:
        return True
    width_rel = _bbox_width(bbox) / page_width
    x0_rel = float(bbox["x0"]) / page_width
    x1_rel = float(bbox["x1"]) / page_width
    if width_rel >= 0.68:
        return True
    return x0_rel <= 0.16 and x1_rel >= 0.84


def _raw_block_line_count(block: Dict[str, Any]) -> int:
    lines = block.get("lines")
    if not isinstance(lines, list):
        return 0
    return sum(1 for line in lines if _join_line_spans(line.get("spans", [])))


def _raw_block_text(block: Dict[str, Any]) -> str:
    lines = block.get("lines")
    if not isinstance(lines, list):
        return ""
    parts: List[str] = []
    for line in lines:
        line_text = _join_line_spans(line.get("spans", []))
        if line_text:
            parts.append(line_text)
    return _sanitize_extracted_text("\n".join(parts).strip())


def _raw_block_max_font_size(block: Dict[str, Any]) -> float:
    max_size = 0.0
    lines = block.get("lines")
    if not isinstance(lines, list):
        return max_size
    for line in lines:
        spans = line.get("spans")
        if not isinstance(spans, list):
            continue
        for span in spans:
            try:
                max_size = max(max_size, float(span.get("size", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
    return max_size


_HEADING_BLOCK_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+){0,3}|[A-Z]|[IVXLCDM]+)\.?\s+[A-Za-z]",
    re.IGNORECASE,
)
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdm]{1,8})\s*$", re.IGNORECASE)
_STOPWORD_RE = re.compile(
    r"\b(the|and|that|with|from|this|these|our|their|which|while|using|without|into|through|because|however|although|where|when|during|under|across|between|provides|introduces|leverages|approach|method|model|policy)\b",
    re.IGNORECASE,
)


def _looks_like_page_number_block(
    block: Dict[str, Any],
    *,
    bbox: Dict[str, float],
    page_width: float,
    page_height: float,
) -> bool:
    text = _raw_block_text(block).strip()
    if not text or not _PAGE_NUMBER_RE.fullmatch(text):
        return False
    if page_width <= 0.0 or page_height <= 0.0:
        return False
    width_rel = _bbox_width(bbox) / page_width
    height_rel = _bbox_height(bbox) / page_height
    y0_rel = float(bbox["y0"]) / page_height
    centered = abs(_bbox_center_x(bbox) - (page_width * 0.5)) <= page_width * 0.08
    return width_rel <= 0.08 and height_rel <= 0.04 and y0_rel >= 0.82 and centered


def _looks_like_standalone_subsection_heading(block: Dict[str, Any], *, bbox: Dict[str, float]) -> bool:
    text = " ".join(_raw_block_text(block).split()).strip()
    if not text:
        return False
    if not _HEADING_BLOCK_RE.match(text):
        return False
    if len(text) > 120:
        return False
    if _raw_block_line_count(block) > 2:
        return False
    return _bbox_height(bbox) <= 32.0


def _looks_like_tiny_symbol_or_math_fragment(block: Dict[str, Any]) -> bool:
    text = " ".join(_raw_block_text(block).split()).strip()
    if not text:
        return False
    if len(text) > 24:
        return False
    alpha_words = len(re.findall(r"[A-Za-z]{2,}", text))
    digit_count = len(re.findall(r"\d", text))
    math_chars = len(re.findall(r"[=+\-*/^_<>∥∑∫πθαβγλμστωφψρ()\[\]{}|]", text))
    if alpha_words == 0 and (digit_count > 0 or math_chars > 0):
        return True
    return math_chars >= 3 and alpha_words <= 1


def _looks_like_equation_only_block(block: Dict[str, Any]) -> bool:
    text = _raw_block_text(block)
    compact = " ".join(text.split()).strip()
    if not compact:
        return False
    line_count = _raw_block_line_count(block)
    stopwords = len(_STOPWORD_RE.findall(compact))
    alpha_words = len(re.findall(r"[A-Za-z]{2,}", compact))
    math_chars = len(re.findall(r"[=+\-*/^_<>∥∑∫πθαβγλμστωφψρ()\[\]{}|]", compact))
    equation_num = bool(re.search(r"\(\d+\)\s*$", compact))
    if line_count <= 3 and equation_num and stopwords == 0:
        return True
    if stopwords == 0 and math_chars >= max(4, alpha_words * 2) and line_count <= 4:
        return True
    return False


def _looks_like_paragraph_body_block(
    block: Dict[str, Any],
    *,
    bbox: Dict[str, float],
    page_width: float,
    page_height: float,
) -> bool:
    compact = " ".join(_raw_block_text(block).split()).strip()
    if len(compact) < 70:
        return False
    if _raw_block_line_count(block) < 2:
        return False
    if _looks_like_page_number_block(block, bbox=bbox, page_width=page_width, page_height=page_height):
        return False
    if _looks_like_equation_only_block(block):
        return False
    if _looks_like_standalone_subsection_heading(block, bbox=bbox):
        return False
    if _looks_like_tiny_symbol_or_math_fragment(block):
        return False
    stopwords = len(_STOPWORD_RE.findall(compact))
    sentence_punct = bool(re.search(r"[.!?;:]\s", compact))
    return stopwords >= 2 or sentence_punct


def _looks_like_short_visual_label(block: Dict[str, Any], *, bbox: Dict[str, float], page_width: float) -> bool:
    compact = " ".join(_raw_block_text(block).split()).strip()
    if not compact or len(compact) > 48:
        return False
    if _raw_block_line_count(block) > 3:
        return False
    width_rel = _bbox_width(bbox) / max(page_width, 1.0)
    word_count = len(re.findall(r"[A-Za-z]{2,}", compact))
    stopwords = len(_STOPWORD_RE.findall(compact))
    if width_rel > 0.34:
        return False
    if stopwords >= 2:
        return False
    return word_count <= 6


def _page_prefers_single_column(
    core: List[Tuple[int, Dict[str, Any], Dict[str, float], bool, bool]],
    *,
    page_width: float,
    page_height: float,
) -> bool:
    full_width_prose = [
        item
        for item in core
        if item[4]
        and _looks_like_paragraph_body_block(item[1], bbox=item[2], page_width=page_width, page_height=page_height)
    ]
    narrow_prose = [
        item
        for item in core
        if not item[4]
        and _looks_like_paragraph_body_block(item[1], bbox=item[2], page_width=page_width, page_height=page_height)
    ]
    if len(full_width_prose) < 2:
        return False
    full_chars = sum(len(_raw_block_text(item[1])) for item in full_width_prose)
    narrow_chars = sum(len(_raw_block_text(item[1])) for item in narrow_prose)
    return len(full_width_prose) >= len(narrow_prose) or full_chars >= max(240, int(narrow_chars * 1.35))


def _looks_like_first_page_preamble_anchor(
    block: Dict[str, Any],
    *,
    bbox: Dict[str, float],
    page_width: float,
    page_height: float,
) -> bool:
    if page_width <= 0.0 or page_height <= 0.0:
        return False

    text = _raw_block_text(block)
    if not text:
        return False
    compact = " ".join(text.split()).strip()
    lowered = compact.lower()
    line_count = _raw_block_line_count(block)
    max_font = _raw_block_max_font_size(block)
    width_rel = _bbox_width(bbox) / page_width
    y0_rel = float(bbox["y0"]) / page_height
    centered = _bbox_centered_near_page_middle(bbox, page_width=page_width)

    if y0_rel > 0.72:
        return False
    if max_font <= 6.0 and line_count <= 2:
        return False

    if y0_rel <= 0.25 and max_font >= 13.0 and width_rel >= 0.28:
        return True

    if lowered.startswith("abstract") or lowered.startswith("keywords"):
        return y0_rel <= 0.68

    if centered and y0_rel <= 0.42 and line_count <= 6 and width_rel >= 0.30 and max_font >= 8.0:
        return True

    if centered and y0_rel <= 0.62 and line_count >= 4 and len(compact) >= 160 and width_rel >= 0.42 and max_font >= 8.0:
        return True

    if y0_rel <= 0.50 and width_rel >= 0.35 and re.search(r"\b(project page|keywords|university|institute|laborator(?:y|ies)|school|department|college|author|corresponding)\b", lowered):
        return True

    return False


def _first_page_preamble_cutoff(
    core: List[Tuple[int, Dict[str, Any], Dict[str, float], bool, bool]],
    *,
    page_width: float,
    page_height: float,
    page_no: int,
) -> Optional[float]:
    if page_no != 1 or not core or page_width <= 0.0 or page_height <= 0.0:
        return None

    anchors: List[float] = []
    has_title_anchor = False
    for _, block, bbox, _, _ in core:
        if not _looks_like_first_page_preamble_anchor(block, bbox=bbox, page_width=page_width, page_height=page_height):
            continue
        anchors.append(float(bbox["y1"]))
        if (
            float(bbox["y0"]) / page_height <= 0.25
            and _raw_block_max_font_size(block) >= 13.0
            and (_bbox_width(bbox) / page_width) >= 0.28
        ):
            has_title_anchor = True

    if not anchors or not has_title_anchor:
        return None
    return min(page_height * 0.74, max(anchors) + 16.0)


def _order_text_blocks_for_page(
    text_blocks: List[Dict[str, Any]],
    *,
    page_width: float,
    page_height: float,
    page_no: int,
) -> List[Tuple[Dict[str, Any], str, str]]:
    if not text_blocks:
        return []

    annotated: List[Tuple[int, Dict[str, Any], Dict[str, float], bool, bool]] = []
    for original_index, block in enumerate(text_blocks):
        bbox = _bbox_payload(block.get("bbox", [0, 0, 0, 0]))
        is_margin = _is_margin_note_block(block, page_width=page_width, page_height=page_height)
        is_full = _is_full_width_block(block, page_width=page_width)
        annotated.append((original_index, block, bbox, is_margin, is_full))

    core = [item for item in annotated if not item[3]]
    margin = [item for item in annotated if item[3]]

    mid_x = page_width * 0.5
    left_candidates = [
        item
        for item in core
        if (
            not item[4]
            and _bbox_center_x(item[2]) <= page_width * 0.45
            and _bbox_width(item[2]) <= page_width * 0.62
            and _looks_like_paragraph_body_block(
                item[1],
                bbox=item[2],
                page_width=page_width,
                page_height=page_height,
            )
        )
    ]
    right_candidates = [
        item
        for item in core
        if (
            not item[4]
            and _bbox_center_x(item[2]) >= page_width * 0.55
            and _bbox_width(item[2]) <= page_width * 0.62
            and _looks_like_paragraph_body_block(
                item[1],
                bbox=item[2],
                page_width=page_width,
                page_height=page_height,
            )
        )
    ]
    has_two_columns = len(left_candidates) >= 2 and len(right_candidates) >= 2
    if has_two_columns and _page_prefers_single_column(core, page_width=page_width, page_height=page_height):
        has_two_columns = False

    if not has_two_columns:
        ordered = sorted(core, key=lambda item: (float(item[2]["y0"]), float(item[2]["x0"]), item[0]))
        ordered.extend(sorted(margin, key=lambda item: (float(item[2]["y0"]), float(item[2]["x0"]), item[0])))
        results: List[Tuple[Dict[str, Any], str, str]] = []
        for _, block, bbox, is_margin, is_full in ordered:
            role = "margin_note" if is_margin else ("full_width" if is_full else "single_column")
            column_hint = "margin" if is_margin else "single"
            results.append((block, role, column_hint))
        return results

    column_top = min(float(item[2]["y0"]) for item in left_candidates + right_candidates)
    column_bottom = max(float(item[2]["y1"]) for item in left_candidates + right_candidates)
    first_page_preamble_cutoff = _first_page_preamble_cutoff(
        core,
        page_width=page_width,
        page_height=page_height,
        page_no=page_no,
    )

    preamble: List[Tuple[int, Dict[str, Any], Dict[str, float], bool, bool]] = []
    left: List[Tuple[int, Dict[str, Any], Dict[str, float], bool, bool]] = []
    right: List[Tuple[int, Dict[str, Any], Dict[str, float], bool, bool]] = []
    in_flow_full: List[Tuple[int, Dict[str, Any], Dict[str, float], bool, bool]] = []
    postamble: List[Tuple[int, Dict[str, Any], Dict[str, float], bool, bool]] = []

    for item in core:
        original_index, block, bbox, _, is_full = item
        center_x = _bbox_center_x(bbox)
        y0 = float(bbox["y0"])
        is_centered_preamble = (
            y0 < column_top - 8.0
            and _bbox_centered_near_page_middle(bbox, page_width=page_width)
            and _bbox_width(bbox) >= page_width * 0.18
        )
        is_centered_postamble = (
            y0 > column_bottom + 8.0
            and _bbox_centered_near_page_middle(bbox, page_width=page_width)
            and _bbox_width(bbox) >= page_width * 0.18
        )
        if first_page_preamble_cutoff is not None and y0 <= first_page_preamble_cutoff:
            preamble.append(item)
            continue
        if is_centered_preamble:
            preamble.append(item)
            continue
        if is_centered_postamble:
            postamble.append(item)
            continue
        if is_full:
            if y0 < column_top - 8.0:
                preamble.append(item)
            elif y0 > column_bottom + 8.0:
                postamble.append(item)
            elif y0 <= column_top + 18.0:
                preamble.append(item)
            else:
                in_flow_full.append(item)
            continue
        if center_x <= mid_x:
            left.append(item)
        else:
            right.append(item)

    ordered = list(sorted(preamble, key=lambda item: (float(item[2]["y0"]), float(item[2]["x0"]), item[0])))
    left_sorted = sorted(left, key=lambda item: (float(item[2]["y0"]), float(item[2]["x0"]), item[0]))
    right_sorted = sorted(right, key=lambda item: (float(item[2]["y0"]), float(item[2]["x0"]), item[0]))
    in_flow_full_sorted = sorted(in_flow_full, key=lambda item: (float(item[2]["y0"]), float(item[2]["x0"]), item[0]))

    left_idx = 0
    right_idx = 0
    for full_item in in_flow_full_sorted:
        full_y0 = float(full_item[2]["y0"])
        while left_idx < len(left_sorted) and float(left_sorted[left_idx][2]["y0"]) < full_y0:
            ordered.append(left_sorted[left_idx])
            left_idx += 1
        while right_idx < len(right_sorted) and float(right_sorted[right_idx][2]["y0"]) < full_y0:
            ordered.append(right_sorted[right_idx])
            right_idx += 1
        ordered.append(full_item)

    ordered.extend(left_sorted[left_idx:])
    ordered.extend(right_sorted[right_idx:])
    ordered.extend(sorted(postamble, key=lambda item: (float(item[2]["y0"]), float(item[2]["x0"]), item[0])))
    ordered.extend(sorted(margin, key=lambda item: (float(item[2]["y0"]), float(item[2]["x0"]), item[0])))

    results: List[Tuple[Dict[str, Any], str, str]] = []
    for _, block, bbox, is_margin, is_full in ordered:
        if is_margin:
            role = "margin_note"
            column_hint = "margin"
        elif is_full:
            role = "full_width"
            column_hint = "full"
        elif _bbox_center_x(bbox) <= mid_x:
            role = "column_block"
            column_hint = "left"
        else:
            role = "column_block"
            column_hint = "right"
        results.append((block, role, column_hint))
    return results


def _should_insert_span_space(previous: str, current: str, x_gap: float) -> bool:
    if not previous or not current:
        return False
    prev_ch = previous[-1]
    next_ch = current[0]
    if prev_ch.isspace() or next_ch.isspace():
        return False
    if next_ch in ".,;:!?)]}":
        return False
    if prev_ch in "([{/":
        return False
    if x_gap > 1.5:
        return True
    return prev_ch.isalnum() and next_ch.isalnum()


def _join_line_spans(spans: List[Dict[str, Any]]) -> str:
    ordered = sorted(spans, key=_span_x0)
    pieces: List[str] = []
    prev_text = ""
    prev_x1: Optional[float] = None
    for span in ordered:
        span_text = _sanitize_extracted_text(span.get("text"))
        if not span_text.strip():
            continue
        x0 = _span_x0(span)
        if pieces and prev_x1 is not None and _should_insert_span_space(prev_text, span_text, x0 - prev_x1):
            pieces.append(" ")
        pieces.append(span_text)
        prev_text = span_text
        prev_x1 = _span_x1(span)
    if not pieces:
        return ""
    return re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip()


async def _fetch_text(url: str) -> str:
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=30,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def _download_pdf(url: str, out_path: Path) -> None:
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=60,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.content
        if not _looks_like_pdf_bytes(data):
            raise RuntimeError("Resolved source did not return a PDF. Ensure the document is shared/exportable as PDF.")
        out_path.write_bytes(data)


def _guess_title_from_pdf(pdf_path: Path) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata or {}
        title = (metadata.title or "").strip()
        if title:
            return title
        first = reader.pages[0].extract_text() or ""
        first = first.strip().split("\n", 1)[0][:160]
        return first or pdf_path.name
    except Exception:
        return pdf_path.name


async def resolve_any_to_pdf(input_str: str, output_dir: Optional[Path] = None) -> Tuple[str, Path]:
    """
    Resolve DOI, landing page URL, arXiv URL, or direct PDF URL into a local PDF.

    Returns:
        (title, local_pdf_path)
    """
    pdf_dir = (output_dir or _default_pdf_dir()).expanduser().resolve()
    pdf_dir.mkdir(parents=True, exist_ok=True)

    value = input_str.strip()
    if not value:
        raise ValueError("input_str cannot be empty")

    google_source = describe_google_drive_source(value)
    if google_source:
        pdf_url = google_source["download_url"]
        out = pdf_dir / _safe_filename(f"{google_source['file_id']}.pdf")
        await _download_pdf(pdf_url, out)
        return _guess_title_from_pdf(out), out

    # DOI shortcut
    if re.match(r"^10\.\d{4,9}/", value):
        landing_url = f"https://doi.org/{value}"
        html = await _fetch_text(landing_url)
    else:
        if value.lower().endswith(".pdf"):
            pdf_url = value
            out = pdf_dir / _safe_filename(pdf_url)
            await _download_pdf(pdf_url, out)
            return _guess_title_from_pdf(out), out
        html = await _fetch_text(value)

    soup = BeautifulSoup(html, "html.parser")
    pdf_url: Optional[str] = None
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and meta.get("content", "").strip():
        pdf_url = meta["content"].strip()

    if not pdf_url:
        link = soup.find("a", href=re.compile(r"\.pdf($|\?)", re.I))
        if link:
            href = str(link.get("href") or "").strip()
            if href.startswith("//"):
                href = "https:" + href
            pdf_url = href

    if not pdf_url:
        raise RuntimeError("Could not locate a PDF link on the source page.")

    out = pdf_dir / _safe_filename(pdf_url)
    await _download_pdf(pdf_url, out)
    return _guess_title_from_pdf(out), out


def extract_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Simple page-level text extraction using pypdf.
    """
    pages: List[Tuple[int, str]] = []
    reader = PdfReader(str(pdf_path))
    for idx, page in enumerate(reader.pages):
        text = _sanitize_extracted_text(page.extract_text())
        pages.append((idx + 1, text))
    return pages


def _normalize_page_allowlist(page_allowlist: Optional[Sequence[int]]) -> Optional[Set[int]]:
    if not page_allowlist:
        return None
    allowed: Set[int] = set()
    for value in page_allowlist:
        try:
            page_no = int(value)
        except (TypeError, ValueError):
            continue
        if page_no > 0:
            allowed.add(page_no)
    return allowed or None


def extract_text_blocks(
    pdf_path: Path,
    page_allowlist: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Block-level extraction using PyMuPDF with geometry + text style metadata.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    allowed_pages = _normalize_page_allowlist(page_allowlist)
    doc = pymupdf.open(str(pdf_path))
    blocks: List[Dict[str, Any]] = []
    try:
        for page_num in range(len(doc)):
            display_page_num = page_num + 1
            if allowed_pages is not None and display_page_num not in allowed_pages:
                continue
            page = doc[page_num]
            page_dict = page.get_text("dict", sort=False)
            text_blocks = [b for b in page_dict.get("blocks", []) if b.get("type") == 0]
            ordered_blocks = _order_text_blocks_for_page(
                text_blocks,
                page_width=float(page.rect.width or 0.0),
                page_height=float(page.rect.height or 0.0),
                page_no=display_page_num,
            )

            block_idx = 0
            for block, layout_role, column_hint in ordered_blocks:
                lines = block.get("lines", [])
                text_lines: List[str] = []
                line_payloads: List[Dict[str, Any]] = []
                span_sizes: List[float] = []
                span_fonts: List[str] = []
                bold_spans = 0
                total_spans = 0

                for line in lines:
                    spans = line.get("spans", [])
                    line_spans: List[Dict[str, Any]] = []
                    for span in spans:
                        span_text = _sanitize_extracted_text(span.get("text"))
                        if not span_text.strip():
                            continue
                        size = span.get("size")
                        try:
                            span_sizes.append(float(size))
                        except (TypeError, ValueError):
                            pass
                        font_name = str(span.get("font") or "")
                        if font_name:
                            span_fonts.append(font_name)
                        total_spans += 1
                        if "bold" in font_name.lower():
                            bold_spans += 1
                        line_spans.append(
                            {
                                "text": span_text,
                                "bbox": _bbox_payload(span.get("bbox")),
                            }
                        )
                    line_text = _join_line_spans(spans)
                    if line_text:
                        text_lines.append(line_text)
                        line_payloads.append(
                            {
                                "text": line_text,
                                "bbox": _bbox_payload(line.get("bbox")),
                                "spans": line_spans,
                            }
                        )

                text = _sanitize_extracted_text("\n".join(text_lines).strip())
                if not text:
                    continue

                bbox = _bbox_payload(block.get("bbox", [0, 0, 0, 0]))
                first_line = text_lines[0].strip() if text_lines else text.splitlines()[0].strip()
                max_font = max(span_sizes) if span_sizes else 0.0
                avg_font = (sum(span_sizes) / len(span_sizes)) if span_sizes else 0.0
                min_font = min(span_sizes) if span_sizes else 0.0
                bold_ratio = (bold_spans / total_spans) if total_spans else 0.0

                blocks.append(
                    {
                        "page_no": display_page_num,
                        "block_index": block_idx,
                        "text": text,
                        "bbox": bbox,
                        "metadata": {
                            "first_line": first_line,
                            "line_count": len(text_lines),
                            "char_count": len(text),
                            "max_font_size": round(max_font, 3),
                            "avg_font_size": round(avg_font, 3),
                            "min_font_size": round(min_font, 3),
                            "bold_ratio": round(bold_ratio, 3),
                            "font_names": sorted(set(span_fonts))[:6],
                            "layout_role": layout_role,
                            "column_hint": column_hint,
                            "lines": line_payloads,
                        },
                    }
                )
                block_idx += 1
    finally:
        doc.close()

    return blocks
