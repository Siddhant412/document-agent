from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pymupdf

logger = logging.getLogger(__name__)

_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.)\s*\d+\b", re.IGNORECASE)
_EXPLICIT_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?P<label>figure|fig\.?)\s*(?P<number>\d+[A-Za-z]?)\s*(?:[:.\-])\s*(?P<body>.+?)\s*$",
    re.IGNORECASE,
)
_FIGURE_REF_RE = re.compile(r"\b(?:figure|fig\.?)\s*(?P<number>\d+[A-Za-z]?)\b", re.IGNORECASE)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox_from_rect(rect: Any) -> Optional[Dict[str, float]]:
    if rect is None:
        return None
    try:
        return {
            "x0": float(rect.x0),
            "y0": float(rect.y0),
            "x1": float(rect.x1),
            "y1": float(rect.y1),
        }
    except Exception:
        return None


def _bbox_from_payload(payload: Any) -> Optional[Dict[str, float]]:
    if not isinstance(payload, dict):
        return None
    x0 = _safe_float(payload.get("x0"), float("nan"))
    y0 = _safe_float(payload.get("y0"), float("nan"))
    x1 = _safe_float(payload.get("x1"), float("nan"))
    y1 = _safe_float(payload.get("y1"), float("nan"))
    if any(v != v for v in (x0, y0, x1, y1)):  # NaN check
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _bbox_from_tuple(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, (tuple, list)) or len(value) < 4:
        return None
    x0 = _safe_float(value[0], float("nan"))
    y0 = _safe_float(value[1], float("nan"))
    x1 = _safe_float(value[2], float("nan"))
    y1 = _safe_float(value[3], float("nan"))
    if any(v != v for v in (x0, y0, x1, y1)):
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _rect_area(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    width = max(0.0, bbox["x1"] - bbox["x0"])
    height = max(0.0, bbox["y1"] - bbox["y0"])
    return width * height


def _rect_overlap(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    ix0 = max(a["x0"], b["x0"])
    iy0 = max(a["y0"], b["y0"])
    ix1 = min(a["x1"], b["x1"])
    iy1 = min(a["y1"], b["y1"])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _center_y(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return (bbox["y0"] + bbox["y1"]) * 0.5


def _bbox_width(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, float(bbox["x1"]) - float(bbox["x0"]))


def _bbox_height(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, float(bbox["y1"]) - float(bbox["y0"]))


def _bbox_center_x(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return (float(bbox["x0"]) + float(bbox["x1"])) * 0.5


def _rect_iou(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    overlap = _rect_overlap(a, b)
    if overlap <= 0:
        return 0.0
    union = _rect_area(a) + _rect_area(b) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def _bbox_intersection(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not a or not b:
        return None
    ix0 = max(float(a["x0"]), float(b["x0"]))
    iy0 = max(float(a["y0"]), float(b["y0"]))
    ix1 = min(float(a["x1"]), float(b["x1"]))
    iy1 = min(float(a["y1"]), float(b["y1"]))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return {"x0": ix0, "y0": iy0, "x1": ix1, "y1": iy1}


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


def _clip_bbox_to_bounds(
    bbox: Optional[Dict[str, float]],
    bounds: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if not bbox or not bounds:
        return None
    return _bbox_intersection(bbox, bounds)


def _bbox_matches_any_iou(
    bbox: Optional[Dict[str, float]],
    candidates: Iterable[Optional[Dict[str, float]]],
    threshold: float,
) -> bool:
    if not bbox:
        return False
    for candidate in candidates:
        if _rect_iou(bbox, candidate) >= threshold:
            return True
    return False


def _page_bounds(page: Any) -> Dict[str, float]:
    rect = page.rect
    return {
        "x0": float(rect.x0),
        "y0": float(rect.y0),
        "x1": float(rect.x1),
        "y1": float(rect.y1),
    }


def _line_text(line: Dict[str, Any]) -> str:
    spans = line.get("spans") or []
    text = "".join(str(span.get("text") or "") for span in spans)
    return " ".join(text.split()).strip()


def _parse_figure_caption(text: str) -> Optional[Dict[str, str]]:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return None
    match = _EXPLICIT_FIGURE_CAPTION_RE.match(compact)
    if not match:
        return None
    label = str(match.group("label") or "Figure").strip()
    number = str(match.group("number") or "").strip()
    body = str(match.group("body") or "").strip()
    if not number or not body:
        return None
    return {
        "label": label.rstrip(".").capitalize(),
        "number": number,
        "body": body,
        "text": compact,
    }


def _figure_number_sort_key(value: str) -> Tuple[int, str]:
    compact = str(value or "").strip()
    match = re.match(r"^(?P<num>\d+)(?P<suffix>[A-Za-z]?)$", compact)
    if not match:
        return (10**9, compact.lower())
    return (_safe_int(match.group("num"), 10**9), str(match.group("suffix") or "").lower())


def _extract_figure_mentions_from_blocks(blocks: Iterable[Dict[str, Any]]) -> List[str]:
    numbers: set[str] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "")
        if not text:
            continue
        for match in _FIGURE_REF_RE.finditer(text):
            number = str(match.group("number") or "").strip()
            if number:
                numbers.add(number)
    return sorted(numbers, key=_figure_number_sort_key)


def _extract_figure_captions_from_page(page: Any) -> List[Dict[str, Any]]:
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return []
    captions: List[Dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_bbox = _bbox_from_tuple(block.get("bbox"))
        first_line_bbox: Optional[Dict[str, float]] = None
        line_texts: List[str] = []
        for line in block.get("lines") or []:
            line_bbox = _bbox_from_tuple(line.get("bbox"))
            text = _line_text(line)
            if not text:
                continue
            if first_line_bbox is None and line_bbox is not None:
                first_line_bbox = line_bbox
            line_texts.append(text)
        if not line_texts:
            continue
        full_text = " ".join(line_texts).strip()
        parsed = _parse_figure_caption(full_text)
        if not parsed:
            continue
        anchor_bbox = first_line_bbox or block_bbox
        if not anchor_bbox:
            continue
        captions.append(
            {
                "text": parsed["text"][:500],
                "bbox": anchor_bbox,
                "block_bbox": block_bbox or anchor_bbox,
                "figure_label": parsed["label"],
                "figure_number": parsed["number"],
                "figure_body": parsed["body"][:460],
            }
        )
    captions.sort(key=lambda item: (float(item["bbox"]["y0"]), float(item["bbox"]["x0"])))
    return captions


def _collect_vector_drawing_bboxes(page: Any) -> List[Dict[str, float]]:
    min_side = max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_DRAWING_SIDE_PT", "0.8"), 0.8))
    drawings = page.get_drawings() or []
    results: List[Dict[str, float]] = []
    for drawing in drawings:
        bbox = _bbox_from_rect(drawing.get("rect"))
        if not bbox:
            continue
        width = _bbox_width(bbox)
        height = _bbox_height(bbox)
        if width < min_side and height < min_side:
            continue
        results.append(bbox)
    return results


def _bbox_x_overlap_ratio(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    ix0 = max(float(a["x0"]), float(b["x0"]))
    ix1 = min(float(a["x1"]), float(b["x1"]))
    overlap = max(0.0, ix1 - ix0)
    base = max(1e-6, min(_bbox_width(a), _bbox_width(b)))
    return overlap / base


def _bbox_x_reference_coverage(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    ix0 = max(float(a["x0"]), float(b["x0"]))
    ix1 = min(float(a["x1"]), float(b["x1"]))
    overlap = max(0.0, ix1 - ix0)
    return overlap / max(1e-6, _bbox_width(b))


def _bbox_y_overlap_ratio(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    iy0 = max(float(a["y0"]), float(b["y0"]))
    iy1 = min(float(a["y1"]), float(b["y1"]))
    overlap = max(0.0, iy1 - iy0)
    base = max(1e-6, min(_bbox_height(a), _bbox_height(b)))
    return overlap / base


def _vertical_gap(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return float("inf")
    if float(a["y1"]) < float(b["y0"]):
        return float(b["y0"]) - float(a["y1"])
    if float(b["y1"]) < float(a["y0"]):
        return float(a["y0"]) - float(b["y1"])
    return 0.0


def _horizontal_gap(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return float("inf")
    if float(a["x1"]) < float(b["x0"]):
        return float(b["x0"]) - float(a["x1"])
    if float(b["x1"]) < float(a["x0"]):
        return float(a["x0"]) - float(b["x1"])
    return 0.0


def _extract_page_text_boxes(page: Any) -> List[Dict[str, Any]]:
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return []
    items: List[Dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        block_bbox = _bbox_from_tuple(block.get("bbox"))
        for line in block.get("lines") or []:
            line_bbox = _bbox_from_tuple(line.get("bbox")) or block_bbox
            text = _line_text(line)
            if not text or not line_bbox:
                continue
            items.append({"text": text[:320], "bbox": line_bbox})
    items.sort(key=lambda item: (float(item["bbox"]["y0"]), float(item["bbox"]["x0"])))
    return items


def _normalize_caption_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"^\s*(figure|fig\.?|table|tab\.?)\s*\d+[a-z]?\s*[:.]?\s*", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return normalized


def _text_is_caption_like(text: str, caption_text: str) -> bool:
    if _FIGURE_CAPTION_RE.search(str(text or "")):
        return True
    normalized_text = _normalize_caption_text(text)
    normalized_caption = _normalize_caption_text(caption_text)
    if not normalized_text or not normalized_caption:
        return False
    return normalized_text.startswith(normalized_caption) or normalized_caption.startswith(normalized_text)


def _text_is_label_like(text: str) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return False
    words = compact.split()
    if len(words) > 8 or len(compact) > 48:
        return False
    if compact.endswith(".") and len(words) > 3:
        return False
    return True


def _text_is_horizontal_label_candidate(
    text: str,
    *,
    bbox: Optional[Dict[str, float]],
    raw_bbox: Optional[Dict[str, float]],
) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact or not bbox or not raw_bbox:
        return False
    if not _text_is_label_like(compact):
        return False

    # Sideways growth is where adjacent-column prose leaks in on two-column pages.
    # Be much stricter here than for top/bottom title or caption-adjacent labels.
    words = compact.split()
    if len(words) > 5:
        return False
    if "," in compact or ";" in compact:
        return False
    if compact.endswith(":") and len(words) > 2:
        return False
    if compact.endswith(".") and compact.lower() != "vs.":
        return False

    raw_width = max(1.0, _bbox_width(raw_bbox))
    text_width = _bbox_width(bbox)
    max_horizontal_label_width = max(
        72.0,
        _safe_float(os.getenv("FIGURE_VECTOR_MAX_HORIZONTAL_LABEL_WIDTH_PT", "96"), 96.0),
        raw_width * _safe_float(os.getenv("FIGURE_VECTOR_MAX_HORIZONTAL_LABEL_WIDTH_RATIO", "0.52"), 0.52),
    )
    if text_width > max_horizontal_label_width:
        return False

    return True


def _text_is_prose_like(text: str) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return False
    words = compact.split()
    if len(words) >= 10:
        return True
    if len(compact) >= 70:
        return True
    if len(words) >= 6 and any(p in compact for p in [".", ";", ",", ":"]):
        return True
    return False


def _text_is_tabular_like(text: str) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return False
    tokens = compact.split()
    if len(tokens) < 4:
        return False
    numeric_like = 0
    alpha_like = 0
    for token in tokens:
        if re.search(r"\d", token) or any(ch in token for ch in ["×", "%", "/", "+", "-", "="]):
            numeric_like += 1
        if re.search(r"[A-Za-z]", token):
            alpha_like += 1
    if numeric_like < max(3, len(tokens) // 2):
        return False
    return alpha_like <= max(3, len(tokens) // 2)


def _text_is_heading_like(text: str) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return False
    if re.match(r"^(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)+)\s+\S", compact):
        return True
    return compact.lower() in {
        "abstract",
        "introduction",
        "related work",
        "related works",
        "method",
        "methods",
        "experiments",
        "results",
        "discussion",
        "conclusion",
        "references",
        "appendix",
    }


def _text_is_section_inference_noise(text: str) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return True
    if _text_is_heading_like(compact):
        return False
    if _FIGURE_CAPTION_RE.search(compact):
        return True
    if re.match(r"^\(?[a-z]\)\s+", compact, re.IGNORECASE):
        return True
    if _text_is_prose_like(compact):
        return False
    if _text_is_label_like(compact):
        return True
    words = compact.split()
    return len(words) <= 4 and len(compact) <= 48


def _group_text_rows(text_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    row_gap = max(1.0, _safe_float(os.getenv("FIGURE_VECTOR_TEXT_ROW_GAP_PT", "4"), 4.0))
    row_y_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_TEXT_ROW_MIN_Y_OVERLAP", "0.45"), 0.45)))

    rows: List[Dict[str, Any]] = []
    for item in sorted(
        (
            {
                "text": str(entry.get("text") or "").strip(),
                "bbox": _bbox_from_payload(entry.get("bbox")),
            }
            for entry in text_boxes
        ),
        key=lambda entry: (
            float(entry["bbox"]["y0"]) if entry.get("bbox") else float("inf"),
            float(entry["bbox"]["x0"]) if entry.get("bbox") else float("inf"),
        ),
    ):
        text = item.get("text") or ""
        bbox = item.get("bbox")
        if not text or not bbox:
            continue
        if rows:
            prev_bbox = rows[-1]["bbox"]
            same_row = _bbox_y_overlap_ratio(prev_bbox, bbox) >= row_y_overlap or _vertical_gap(prev_bbox, bbox) <= row_gap
            if same_row:
                rows[-1]["items"].append(item)
                rows[-1]["bbox"] = _bbox_union(prev_bbox, bbox) or prev_bbox
                continue
        rows.append({"bbox": dict(bbox), "items": [item]})

    grouped: List[Dict[str, Any]] = []
    for row in rows:
        items = sorted(row["items"], key=lambda entry: float(entry["bbox"]["x0"]))
        grouped.append(
            {
                "bbox": row["bbox"],
                "text": " ".join(entry["text"] for entry in items if entry["text"]).strip(),
                "count": len(items),
            }
        )
    return grouped


def _cluster_vector_bboxes(bboxes: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    if not bboxes:
        return []
    x_gap_max = max(4.0, _safe_float(os.getenv("FIGURE_VECTOR_CLUSTER_X_GAP_PT", "18"), 18.0))
    y_gap_max = max(4.0, _safe_float(os.getenv("FIGURE_VECTOR_CLUSTER_Y_GAP_PT", "16"), 16.0))
    min_x_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_CLUSTER_MIN_X_OVERLAP", "0.08"), 0.08)))
    min_y_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_CLUSTER_MIN_Y_OVERLAP", "0.08"), 0.08)))

    def connected(a: Dict[str, float], b: Dict[str, float]) -> bool:
        if _rect_iou(a, b) > 0:
            return True
        if _bbox_x_overlap_ratio(a, b) >= min_x_overlap and _vertical_gap(a, b) <= y_gap_max:
            return True
        if _bbox_y_overlap_ratio(a, b) >= min_y_overlap and _horizontal_gap(a, b) <= x_gap_max:
            return True
        return False

    remaining = list(bboxes)
    clusters: List[List[Dict[str, float]]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            next_remaining: List[Dict[str, float]] = []
            for item in remaining:
                if any(connected(item, member) for member in cluster):
                    cluster.append(item)
                    changed = True
                else:
                    next_remaining.append(item)
            remaining = next_remaining
        clusters.append(cluster)
    return clusters


def _cluster_embedded_bboxes(bboxes: List[Dict[str, float]]) -> List[List[Dict[str, float]]]:
    if not bboxes:
        return []
    x_gap_max = max(2.0, _safe_float(os.getenv("FIGURE_EMBEDDED_CLUSTER_X_GAP_PT", "14"), 14.0))
    y_gap_max = max(2.0, _safe_float(os.getenv("FIGURE_EMBEDDED_CLUSTER_Y_GAP_PT", "18"), 18.0))
    min_x_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_EMBEDDED_CLUSTER_MIN_X_OVERLAP", "0.05"), 0.05)))
    min_y_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_EMBEDDED_CLUSTER_MIN_Y_OVERLAP", "0.05"), 0.05)))

    def connected(a: Dict[str, float], b: Dict[str, float]) -> bool:
        if _rect_iou(a, b) > 0:
            return True
        if _bbox_x_overlap_ratio(a, b) >= min_x_overlap and _vertical_gap(a, b) <= y_gap_max:
            return True
        if _bbox_y_overlap_ratio(a, b) >= min_y_overlap and _horizontal_gap(a, b) <= x_gap_max:
            return True
        return False

    remaining = list(bboxes)
    clusters: List[List[Dict[str, float]]] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            next_remaining: List[Dict[str, float]] = []
            for item in remaining:
                if any(connected(item, member) for member in cluster):
                    cluster.append(item)
                    changed = True
                else:
                    next_remaining.append(item)
            remaining = next_remaining
        clusters.append(cluster)
    return clusters


def _assign_captions_to_regions(
    *,
    regions: List[Dict[str, Any]],
    captions: List[Dict[str, Any]],
) -> None:
    max_gap = max(24.0, _safe_float(os.getenv("FIGURE_CAPTION_REGION_MAX_GAP_PT", "180"), 180.0))
    min_x_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_CAPTION_REGION_MIN_X_OVERLAP", "0.12"), 0.12)))
    assigned_region_indices: set[int] = set()
    assigned_caption_indices: set[int] = set()

    def _apply_caption(region: Dict[str, Any], caption: Dict[str, Any]) -> None:
        region["figure_caption"] = str(caption.get("text") or "").strip() or None
        region["figure_label"] = str(caption.get("figure_label") or "Figure").strip() or "Figure"
        region["figure_number"] = str(caption.get("figure_number") or "").strip() or None
        region["figure_body"] = str(caption.get("figure_body") or "").strip() or None
        region["caption_bbox"] = _bbox_from_payload(caption.get("bbox"))
        region["caption_block_bbox"] = _bbox_from_payload(caption.get("block_bbox")) or region.get("caption_bbox")

    for caption_idx, caption in enumerate(captions):
        caption_bbox = caption.get("block_bbox") if isinstance(caption.get("block_bbox"), dict) else caption.get("bbox")
        if not isinstance(caption_bbox, dict):
            continue
        best_idx: Optional[int] = None
        best_score = -10**9
        for idx, region in enumerate(regions):
            if idx in assigned_region_indices:
                continue
            bbox = region.get("bbox")
            if not isinstance(bbox, dict):
                continue
            x_overlap = _bbox_x_overlap_ratio(bbox, caption_bbox)
            if x_overlap < min_x_overlap:
                continue
            gap = _vertical_gap(bbox, caption_bbox)
            if gap > max_gap:
                continue
            above_caption = float(bbox["y1"]) <= float(caption_bbox["y0"]) + 6.0
            below_caption = float(bbox["y0"]) >= float(caption_bbox["y1"]) - 6.0
            if not (above_caption or below_caption):
                continue
            score = _score_figure_region_for_caption(
                bbox=bbox,
                caption_bbox=caption_bbox,
                kind="embedded",
                density_count=_safe_int(region.get("tile_count"), 1),
            )
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            continue
        region = regions[best_idx]
        _apply_caption(region, caption)
        assigned_region_indices.add(best_idx)
        assigned_caption_indices.add(caption_idx)

    if len(assigned_region_indices) == len(regions) or len(assigned_caption_indices) == len(captions):
        return

    relaxed_gap = max_gap * 1.6
    relaxed_min_x_overlap = min_x_overlap * 0.2
    unassigned_regions = [idx for idx in range(len(regions)) if idx not in assigned_region_indices]
    for caption_idx, caption in enumerate(captions):
        if caption_idx in assigned_caption_indices:
            continue
        caption_bbox = caption.get("block_bbox") if isinstance(caption.get("block_bbox"), dict) else caption.get("bbox")
        if not isinstance(caption_bbox, dict):
            continue
        best_idx = None
        best_score = -10**9
        for idx in unassigned_regions:
            region = regions[idx]
            bbox = region.get("bbox")
            if not isinstance(bbox, dict):
                continue
            x_overlap = _bbox_x_overlap_ratio(bbox, caption_bbox)
            gap = _vertical_gap(bbox, caption_bbox)
            if gap > relaxed_gap:
                continue
            above_caption = float(bbox["y1"]) <= float(caption_bbox["y0"]) + 10.0
            below_caption = float(bbox["y0"]) >= float(caption_bbox["y1"]) - 10.0
            if not (above_caption or below_caption):
                continue
            if x_overlap < relaxed_min_x_overlap and len(unassigned_regions) > 1:
                continue
            score = _score_figure_region_for_caption(
                bbox=bbox,
                caption_bbox=caption_bbox,
                kind="embedded",
                density_count=_safe_int(region.get("tile_count"), 1),
            )
            if len(unassigned_regions) == 1:
                score += 0.35
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            continue
        _apply_caption(regions[best_idx], caption)
        assigned_caption_indices.add(caption_idx)
        assigned_region_indices.add(best_idx)
        unassigned_regions = [idx for idx in unassigned_regions if idx != best_idx]


def _captions_share_column_band(
    current_caption_bbox: Optional[Dict[str, float]],
    other_caption_bbox: Optional[Dict[str, float]],
    *,
    page_bounds: Dict[str, float],
) -> bool:
    if not current_caption_bbox or not other_caption_bbox:
        return False
    page_width = max(1e-6, _bbox_width(page_bounds))
    current_is_wide = (_bbox_width(current_caption_bbox) / page_width) >= 0.72
    other_is_wide = (_bbox_width(other_caption_bbox) / page_width) >= 0.72
    if current_is_wide or other_is_wide:
        return True
    if _bbox_x_overlap_ratio(current_caption_bbox, other_caption_bbox) >= 0.18:
        return True
    center_delta = abs(_bbox_center_x(current_caption_bbox) - _bbox_center_x(other_caption_bbox))
    return center_delta <= page_width * 0.18


def _score_figure_region_for_caption(
    *,
    bbox: Optional[Dict[str, float]],
    caption_bbox: Optional[Dict[str, float]],
    kind: str,
    density_count: int = 0,
    base_score: float = 0.0,
) -> float:
    if not bbox or not caption_bbox:
        return -10**9
    gap = _vertical_gap(bbox, caption_bbox)
    x_overlap = _bbox_x_overlap_ratio(bbox, caption_bbox)
    caption_coverage = _bbox_x_reference_coverage(bbox, caption_bbox)
    overlap_ratio = _rect_overlap(bbox, caption_bbox) / max(_rect_area(caption_bbox), 1e-6)
    area_bonus = min(4.5, _rect_area(bbox) / 30000.0)
    above_caption = float(bbox["y1"]) <= float(caption_bbox["y0"]) + 10.0
    below_caption = float(bbox["y0"]) >= float(caption_bbox["y1"]) - 10.0
    orientation_bonus = 0.9 if above_caption else (0.15 if below_caption else -0.2)
    width_ratio = min(_bbox_width(bbox), _bbox_width(caption_bbox)) / max(
        1e-6,
        max(_bbox_width(bbox), _bbox_width(caption_bbox)),
    )
    base_weight = 0.55 if kind == "vector" else 1.0

    if kind == "embedded":
        density_bonus = min(2.5, max(0, density_count) * 0.14)
    else:
        density_bonus = min(1.8, max(0, density_count) * 0.05)

    fragment_penalty = 0.0
    if kind == "vector" and caption_coverage < 0.38 and width_ratio < 0.42:
        fragment_penalty = ((0.38 - caption_coverage) * 6.0) + ((0.42 - width_ratio) * 4.0)

    return (
        (base_score * base_weight)
        + area_bonus
        + density_bonus
        + (x_overlap * 1.8)
        + (caption_coverage * 1.8)
        + (overlap_ratio * 0.6)
        - (gap / 150.0)
        + orientation_bonus
        - fragment_penalty
    )


def _relative_position_to_caption(
    bbox: Optional[Dict[str, float]],
    caption_bbox: Optional[Dict[str, float]],
) -> str:
    if not bbox or not caption_bbox:
        return "unknown"
    if float(bbox["y1"]) <= float(caption_bbox["y0"]) + 10.0:
        return "above"
    if float(bbox["y0"]) >= float(caption_bbox["y1"]) - 10.0:
        return "below"
    return "overlap"


def _can_merge_embedded_regions_for_caption(
    *,
    base_bbox: Optional[Dict[str, float]],
    other_bbox: Optional[Dict[str, float]],
    caption_bbox: Optional[Dict[str, float]],
    page_bounds: Dict[str, float],
) -> bool:
    if not base_bbox or not other_bbox or not caption_bbox:
        return False
    base_position = _relative_position_to_caption(base_bbox, caption_bbox)
    other_position = _relative_position_to_caption(other_bbox, caption_bbox)
    if base_position != other_position:
        return False
    if base_position == "overlap":
        return False

    caption_width = _bbox_width(caption_bbox)
    x_pad = max(24.0, caption_width * 0.35)
    other_center_x = _bbox_center_x(other_bbox)
    if (
        _bbox_x_reference_coverage(other_bbox, caption_bbox) < 0.08
        and not (float(caption_bbox["x0"]) - x_pad <= other_center_x <= float(caption_bbox["x1"]) + x_pad)
    ):
        return False

    horizontally_adjacent = _bbox_y_overlap_ratio(base_bbox, other_bbox) >= 0.38 and _horizontal_gap(base_bbox, other_bbox) <= max(
        18.0,
        caption_width * 0.22,
    )
    vertically_adjacent = _bbox_x_overlap_ratio(base_bbox, other_bbox) >= 0.28 and _vertical_gap(base_bbox, other_bbox) <= 28.0
    overlapping = _rect_iou(base_bbox, other_bbox) > 0.0
    if not (horizontally_adjacent or vertically_adjacent or overlapping):
        return False

    union_bbox = _bbox_union(base_bbox, other_bbox)
    if not union_bbox:
        return False
    union_bbox = _clip_bbox_to_bounds(union_bbox, page_bounds) or union_bbox
    if _relative_position_to_caption(union_bbox, caption_bbox) != base_position:
        return False
    return True


def _merge_caption_adjacent_embedded_regions(
    *,
    regions: List[Dict[str, Any]],
    captions: List[Dict[str, Any]],
    page_bounds: Dict[str, float],
) -> None:
    min_gain = max(0.2, _safe_float(os.getenv("FIGURE_EMBEDDED_MERGE_MIN_SCORE_GAIN", "0.35"), 0.35))
    idx = 0
    while idx < len(regions):
        region = regions[idx]
        caption_bbox = region.get("caption_block_bbox") if isinstance(region.get("caption_block_bbox"), dict) else region.get("caption_bbox")
        bbox = region.get("bbox") if isinstance(region.get("bbox"), dict) else None
        if not bbox or not isinstance(caption_bbox, dict):
            idx += 1
            continue
        if not (
            str(region.get("figure_number") or "").strip()
            or str(region.get("figure_caption") or "").strip()
            or str(region.get("figure_body") or "").strip()
        ):
            idx += 1
            continue

        changed = False
        base_score = _score_figure_region_for_caption(
            bbox=bbox,
            caption_bbox=caption_bbox,
            kind="embedded",
            density_count=_safe_int(region.get("tile_count"), 1),
        )
        best_choice: Optional[Tuple[int, Dict[str, float], int, float]] = None
        for other_idx, other in enumerate(regions):
            if other_idx == idx:
                continue
            if (
                str(other.get("figure_number") or "").strip()
                or str(other.get("figure_caption") or "").strip()
                or str(other.get("figure_body") or "").strip()
            ):
                continue
            other_bbox = other.get("bbox") if isinstance(other.get("bbox"), dict) else None
            if not _can_merge_embedded_regions_for_caption(
                base_bbox=bbox,
                other_bbox=other_bbox,
                caption_bbox=caption_bbox,
                page_bounds=page_bounds,
            ):
                continue
            merged_bbox = _bbox_union(bbox, other_bbox)
            if not merged_bbox:
                continue
            trimmed_bbox = _trim_embedded_region_away_from_captions(
                region_bbox=merged_bbox,
                captions=captions,
                page_bounds=page_bounds,
            )
            merged_bbox = trimmed_bbox or merged_bbox
            if not _is_meaningful_render_rect(merged_bbox):
                continue
            merged_tiles = _safe_int(region.get("tile_count"), 1) + _safe_int(other.get("tile_count"), 1)
            merged_score = _score_figure_region_for_caption(
                bbox=merged_bbox,
                caption_bbox=caption_bbox,
                kind="embedded",
                density_count=merged_tiles,
            )
            if merged_score < base_score + min_gain:
                continue
            if not best_choice or merged_score > best_choice[3]:
                best_choice = (other_idx, merged_bbox, merged_tiles, merged_score)

        if not best_choice:
            idx += 1
            continue

        other_idx, merged_bbox, merged_tiles, _ = best_choice
        region["bbox"] = merged_bbox
        region["tile_count"] = merged_tiles
        regions.pop(other_idx)
        if other_idx < idx:
            idx -= 1
        changed = True
        if not changed:
            idx += 1


def _resolve_page_figure_candidates(page_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    passthrough: List[Dict[str, Any]] = []
    for record in page_candidates:
        figure_number = str(record.get("figure_number") or "").strip()
        if not figure_number:
            passthrough.append(record)
            continue
        groups.setdefault(figure_number, []).append(record)

    kept: List[Dict[str, Any]] = list(passthrough)
    for figure_number, records in groups.items():
        if len(records) == 1:
            kept.extend(records)
            continue
        winner = max(records, key=lambda item: float(item.get("_candidate_score") or -10**9))
        kept.append(winner)
        for loser in records:
            if loser is winner:
                continue
            image_path = str(loser.get("image_path") or "").strip()
            if image_path:
                try:
                    Path(image_path).unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to remove superseded figure file %s", image_path)

    kept.sort(
        key=lambda item: (
            _safe_int(item.get("page_no"), 0),
            float((item.get("bbox") or {}).get("y0") or 0.0),
            float((item.get("bbox") or {}).get("x0") or 0.0),
            str(item.get("file_name") or ""),
        )
    )
    return kept


def _trim_embedded_region_away_from_captions(
    *,
    region_bbox: Optional[Dict[str, float]],
    captions: List[Dict[str, Any]],
    page_bounds: Dict[str, float],
) -> Optional[Dict[str, float]]:
    bbox = dict(region_bbox or {})
    if not bbox:
        return None

    trim_pad = max(1.0, _safe_float(os.getenv("FIGURE_EMBEDDED_CAPTION_TRIM_PAD_PT", "4"), 4.0))
    edge_band = max(12.0, _safe_float(os.getenv("FIGURE_EMBEDDED_CAPTION_EDGE_BAND_PT", "56"), 56.0))
    min_x_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_EMBEDDED_CAPTION_MIN_X_OVERLAP", "0.18"), 0.18)))

    for caption in captions:
        caption_bbox = caption.get("block_bbox") if isinstance(caption.get("block_bbox"), dict) else caption.get("bbox")
        if not isinstance(caption_bbox, dict):
            continue
        if _bbox_x_overlap_ratio(bbox, caption_bbox) < min_x_overlap:
            continue
        overlap = _rect_overlap(bbox, caption_bbox)
        if overlap <= 0:
            continue

        region_height = max(1.0, _bbox_height(bbox))
        caption_center_y = _center_y(caption_bbox)
        bottom_edge_hit = (
            float(caption_bbox["y0"]) >= float(bbox["y1"]) - edge_band
            or caption_center_y >= float(bbox["y0"]) + region_height * 0.7
        )
        top_edge_hit = (
            float(caption_bbox["y1"]) <= float(bbox["y0"]) + edge_band
            or caption_center_y <= float(bbox["y0"]) + region_height * 0.3
        )

        if bottom_edge_hit:
            bbox["y1"] = min(float(bbox["y1"]), float(caption_bbox["y0"]) - trim_pad)
        elif top_edge_hit:
            bbox["y0"] = max(float(bbox["y0"]), float(caption_bbox["y1"]) + trim_pad)

    bbox = _clip_bbox_to_bounds(bbox, page_bounds) or bbox
    if not _is_meaningful_render_rect(bbox):
        return None
    return bbox


def _recover_unassigned_embedded_captions(
    *,
    regions: List[Dict[str, Any]],
    captions: List[Dict[str, Any]],
) -> None:
    assigned_keys = {
        str(region.get("figure_number") or "").strip()
        for region in regions
        if str(region.get("figure_number") or "").strip()
    }
    unresolved_caption_indices = [
        idx
        for idx, caption in enumerate(captions)
        if str(caption.get("figure_number") or "").strip() not in assigned_keys
    ]
    unresolved_region_indices = [
        idx
        for idx, region in enumerate(regions)
        if not str(region.get("figure_number") or "").strip()
    ]
    if not unresolved_caption_indices or not unresolved_region_indices:
        return

    def _apply_caption(region: Dict[str, Any], caption: Dict[str, Any]) -> None:
        region["figure_caption"] = str(caption.get("text") or "").strip() or None
        region["figure_label"] = str(caption.get("figure_label") or "Figure").strip() or "Figure"
        region["figure_number"] = str(caption.get("figure_number") or "").strip() or None
        region["figure_body"] = str(caption.get("figure_body") or "").strip() or None
        region["caption_bbox"] = _bbox_from_payload(caption.get("bbox"))
        region["caption_block_bbox"] = _bbox_from_payload(caption.get("block_bbox")) or region.get("caption_bbox")

    max_gap = max(32.0, _safe_float(os.getenv("FIGURE_EMBEDDED_RECOVERY_MAX_GAP_PT", "220"), 220.0))
    min_x_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_EMBEDDED_RECOVERY_MIN_X_OVERLAP", "0.08"), 0.08)))

    for caption_idx in unresolved_caption_indices:
        caption = captions[caption_idx]
        caption_bbox = caption.get("block_bbox") if isinstance(caption.get("block_bbox"), dict) else caption.get("bbox")
        if not isinstance(caption_bbox, dict):
            continue

        best_idx: Optional[int] = None
        best_score = -10**9
        for region_idx in unresolved_region_indices:
            region = regions[region_idx]
            bbox = region.get("bbox")
            if not isinstance(bbox, dict):
                continue
            x_overlap = _bbox_x_overlap_ratio(bbox, caption_bbox)
            center_in_x = float(caption_bbox["x0"]) <= _bbox_center_x(bbox) <= float(caption_bbox["x1"]) or (
                float(bbox["x0"]) - 18.0 <= _bbox_center_x(caption_bbox) <= float(bbox["x1"]) + 18.0
            )
            if x_overlap < min_x_overlap and not center_in_x:
                continue

            gap = _vertical_gap(bbox, caption_bbox)
            overlap = _rect_overlap(bbox, caption_bbox)
            caption_area = max(_rect_area(caption_bbox), 1e-6)
            overlap_ratio = overlap / caption_area
            region_height = max(1.0, _bbox_height(bbox))
            caption_center_y = _center_y(caption_bbox)
            edge_hit = (
                caption_center_y >= float(bbox["y0"]) + region_height * 0.68
                or caption_center_y <= float(bbox["y0"]) + region_height * 0.32
            )
            if overlap_ratio <= 0 and gap > max_gap:
                continue
            if overlap_ratio > 0 and not edge_hit:
                continue

            area_bonus = min(2.0, _rect_area(bbox) / 80000.0)
            score = overlap_ratio * 4.0 + x_overlap * 2.5 - (gap / 180.0) + area_bonus + (0.3 if center_in_x else 0.0)
            if score > best_score:
                best_idx = region_idx
                best_score = score

        if best_idx is None:
            continue

        _apply_caption(regions[best_idx], caption)
        unresolved_region_indices = [idx for idx in unresolved_region_indices if idx != best_idx]
        if not unresolved_region_indices:
            break


def _build_embedded_regions(page: Any, doc: Any) -> List[Dict[str, Any]]:
    placements: List[Dict[str, Any]] = []
    seen_bboxes: List[Dict[str, float]] = []
    for image_info in page.get_images(full=True):
        xref = _safe_int(image_info[0], -1)
        if xref <= 0:
            continue
        try:
            payload = doc.extract_image(xref)
        except Exception:
            continue
        width = _safe_int(payload.get("width"), 0)
        height = _safe_int(payload.get("height"), 0)
        if width > 0 and height > 0 and not _is_meaningful_image(width, height):
            continue
        for rect in page.get_image_rects(xref) or []:
            bbox = _bbox_from_rect(rect)
            if not _is_meaningful_render_rect(bbox):
                continue
            if _bbox_matches_any_iou(bbox, seen_bboxes, 0.995):
                continue
            placements.append({"bbox": bbox})
            seen_bboxes.append(bbox)

    regions: List[Dict[str, Any]] = []
    for cluster in _cluster_embedded_bboxes([item["bbox"] for item in placements]):
        bbox = _region_from_cluster(cluster)
        if not bbox or not _is_meaningful_render_rect(bbox):
            continue
        regions.append(
            {
                "bbox": bbox,
                "tile_count": len(cluster),
            }
        )
    regions.sort(key=lambda item: (float(item["bbox"]["y0"]), float(item["bbox"]["x0"])))
    return regions


def _region_from_cluster(cluster: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    region_bbox: Optional[Dict[str, float]] = None
    for item in cluster:
        region_bbox = _bbox_union(region_bbox, item)
    return region_bbox


def _refine_vector_region_with_text(
    *,
    raw_bbox: Dict[str, float],
    expanded_bbox: Dict[str, float],
    window_bbox: Dict[str, float],
    text_boxes: List[Dict[str, Any]],
    caption_text: str,
) -> Dict[str, float]:
    label_gap = max(2.0, _safe_float(os.getenv("FIGURE_VECTOR_LABEL_MAX_GAP_PT", "18"), 18.0))
    trim_pad = max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_TEXT_TRIM_PAD_PT", "3"), 3.0))
    edge_band = max(4.0, _safe_float(os.getenv("FIGURE_VECTOR_TEXT_EDGE_BAND_PT", "18"), 18.0))
    top_noise_scan = max(edge_band + 8.0, _safe_float(os.getenv("FIGURE_VECTOR_TOP_NOISE_SCAN_PT", "72"), 72.0))
    top_noise_gap = max(4.0, _safe_float(os.getenv("FIGURE_VECTOR_TOP_NOISE_MIN_GAP_PT", "10"), 10.0))

    grown_raw = dict(raw_bbox)
    for item in text_boxes:
        text = str(item.get("text") or "").strip()
        bbox = _bbox_from_payload(item.get("bbox"))
        if not text or not bbox:
            continue
        if not _bbox_intersection(bbox, window_bbox):
            continue
        if _text_is_caption_like(text, caption_text):
            continue
        if not _text_is_label_like(text):
            continue
        if _rect_overlap(grown_raw, bbox) > 0:
            continue
        near_vertical = _bbox_x_overlap_ratio(grown_raw, bbox) >= 0.2 and _vertical_gap(grown_raw, bbox) <= label_gap
        near_horizontal = _bbox_y_overlap_ratio(grown_raw, bbox) >= 0.2 and _horizontal_gap(grown_raw, bbox) <= label_gap
        if near_vertical:
            grown_raw = _bbox_union(grown_raw, bbox) or grown_raw
            continue
        if near_horizontal and _text_is_horizontal_label_candidate(text, bbox=bbox, raw_bbox=grown_raw):
            grown_raw = _bbox_union(grown_raw, bbox) or grown_raw

    refined = _clip_bbox_to_bounds(expanded_bbox, window_bbox) or dict(grown_raw)
    refined = _bbox_union(grown_raw, refined) or grown_raw
    refined = _clip_bbox_to_bounds(refined, window_bbox) or grown_raw

    for item in text_boxes:
        text = str(item.get("text") or "").strip()
        bbox = _bbox_from_payload(item.get("bbox"))
        if not text or not bbox:
            continue
        overlap_refined = _rect_overlap(refined, bbox)
        if overlap_refined <= 0:
            continue
        overlap_with_raw = _rect_overlap(grown_raw, bbox)
        block_area = max(_rect_area(bbox), 1e-6)
        raw_ratio = overlap_with_raw / block_area
        if raw_ratio >= 0.2:
            continue
        if not (_text_is_caption_like(text, caption_text) or _text_is_prose_like(text)):
            continue

        if float(bbox["y0"]) >= float(grown_raw["y1"]) - 1.0:
            refined["y1"] = min(float(refined["y1"]), float(bbox["y0"]) - trim_pad)
        elif float(bbox["y1"]) <= float(grown_raw["y0"]) + 1.0:
            refined["y0"] = max(float(refined["y0"]), float(bbox["y1"]) + trim_pad)
        elif float(bbox["x0"]) >= float(grown_raw["x1"]) - 1.0:
            refined["x1"] = min(float(refined["x1"]), float(bbox["x0"]) - trim_pad)
        elif float(bbox["x1"]) <= float(grown_raw["x0"]) + 1.0:
            refined["x0"] = max(float(refined["x0"]), float(bbox["x1"]) + trim_pad)

        refined = _clip_bbox_to_bounds(refined, window_bbox) or grown_raw

    # Trim stray prose rows that still graze the top/bottom edge of the raw figure box.
    for item in text_boxes:
        text = str(item.get("text") or "").strip()
        bbox = _bbox_from_payload(item.get("bbox"))
        if not text or not bbox or not _text_is_prose_like(text):
            continue
        block_area = max(_rect_area(bbox), 1e-6)
        raw_overlap_ratio = _rect_overlap(grown_raw, bbox) / block_area
        if raw_overlap_ratio >= 0.6:
            continue
        if _bbox_x_overlap_ratio(grown_raw, bbox) < 0.35:
            continue

        if float(bbox["y0"]) < float(grown_raw["y0"]) and float(bbox["y1"]) > float(grown_raw["y0"]) and float(bbox["y1"]) <= float(grown_raw["y0"]) + edge_band:
            refined["y0"] = max(float(refined["y0"]), float(bbox["y1"]) + trim_pad)
        elif float(bbox["y1"]) > float(grown_raw["y1"]) and float(bbox["y0"]) < float(grown_raw["y1"]) and float(bbox["y0"]) >= float(grown_raw["y1"]) - edge_band:
            refined["y1"] = min(float(refined["y1"]), float(bbox["y0"]) - trim_pad)

    top_rows: List[Dict[str, Any]] = []
    for row in _group_text_rows(text_boxes):
        row_bbox = _bbox_from_payload(row.get("bbox"))
        row_text = str(row.get("text") or "").strip()
        if not row_bbox or not row_text:
            continue
        if _bbox_x_overlap_ratio(grown_raw, row_bbox) < 0.35:
            continue
        if _bbox_width(row_bbox) < _bbox_width(grown_raw) * 0.3:
            continue
        if float(row_bbox["y1"]) < float(grown_raw["y0"]) - edge_band:
            continue
        if float(row_bbox["y0"]) > float(grown_raw["y0"]) + top_noise_scan:
            continue
        top_rows.append({"bbox": row_bbox, "text": row_text})

    last_noise_row: Optional[Dict[str, Any]] = None
    first_anchor_row: Optional[Dict[str, Any]] = None
    for row in sorted(top_rows, key=lambda item: (float(item["bbox"]["y0"]), float(item["bbox"]["x0"]))):
        text = row["text"]
        if _text_is_caption_like(text, caption_text) or _text_is_tabular_like(text):
            last_noise_row = row
            continue
        first_anchor_row = row
        break

    if last_noise_row and first_anchor_row:
        gap = float(first_anchor_row["bbox"]["y0"]) - float(last_noise_row["bbox"]["y1"])
        if gap >= top_noise_gap:
            refined["y0"] = max(float(refined["y0"]), float(last_noise_row["bbox"]["y1"]) + trim_pad)

    refined = _clip_bbox_to_bounds(refined, window_bbox) or grown_raw

    return refined


def _build_vector_region_from_caption(
    *,
    caption: Dict[str, Any],
    caption_index: int,
    captions: List[Dict[str, Any]],
    drawing_bboxes: List[Dict[str, float]],
    text_boxes: List[Dict[str, Any]],
    page_bounds: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    if not drawing_bboxes:
        return None

    caption_bbox = caption.get("bbox")
    if not isinstance(caption_bbox, dict):
        return None
    caption_block_bbox = caption.get("block_bbox") if isinstance(caption.get("block_bbox"), dict) else caption_bbox

    max_above = max(24.0, _safe_float(os.getenv("FIGURE_VECTOR_MAX_ABOVE_PT", "380"), 380.0))
    max_below = max(24.0, _safe_float(os.getenv("FIGURE_VECTOR_MAX_BELOW_PT", "240"), 240.0))
    min_window_height = max(16.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_WINDOW_HEIGHT_PT", "36"), 36.0))
    min_drawings = max(1, _safe_int(os.getenv("FIGURE_VECTOR_MIN_DRAWING_COUNT", "6"), 6))
    min_region_area = max(1.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_REGION_AREA_PT", "5000"), 5000.0))
    min_region_side = max(4.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_REGION_SIDE_PT", "40"), 40.0))
    margin = max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_MARGIN_PT", "6"), 6.0))
    column_pad = max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_COLUMN_X_PAD_PT", "28"), 28.0))
    min_x_overlap = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_MIN_X_OVERLAP_RATIO", "0.05"), 0.05)))

    page_width = max(1e-6, _bbox_width(page_bounds))
    caption_width = _bbox_width(caption_bbox)
    caption_is_wide = (caption_width / page_width) >= 0.72
    x0_gate = float(caption_bbox["x0"]) - column_pad
    x1_gate = float(caption_bbox["x1"]) + column_pad

    prev_caption = (
        captions[caption_index - 1].get("block_bbox")
        if caption_index > 0 and isinstance(captions[caption_index - 1].get("block_bbox"), dict)
        else captions[caption_index - 1]["bbox"]
        if caption_index > 0
        else None
    )
    next_caption = (
        captions[caption_index + 1].get("block_bbox")
        if caption_index + 1 < len(captions) and isinstance(captions[caption_index + 1].get("block_bbox"), dict)
        else captions[caption_index + 1]["bbox"]
        if caption_index + 1 < len(captions)
        else None
    )

    windows: List[Tuple[str, Dict[str, float]]] = []
    above_bottom = float(caption_block_bbox["y0"]) - 2.0
    above_top = max(float(page_bounds["y0"]) + 1.0, above_bottom - max_above)
    if isinstance(prev_caption, dict) and _captions_share_column_band(
        caption_block_bbox,
        prev_caption,
        page_bounds=page_bounds,
    ):
        above_top = max(above_top, float(prev_caption["y1"]) + 2.0)
    if above_bottom - above_top >= min_window_height:
        windows.append(("above", {"x0": page_bounds["x0"], "y0": above_top, "x1": page_bounds["x1"], "y1": above_bottom}))

    below_top = float(caption_block_bbox["y1"]) + 2.0
    below_bottom = min(float(page_bounds["y1"]) - 1.0, below_top + max_below)
    if isinstance(next_caption, dict) and _captions_share_column_band(
        caption_block_bbox,
        next_caption,
        page_bounds=page_bounds,
    ):
        below_bottom = min(below_bottom, float(next_caption["y0"]) - 2.0)
    if below_bottom - below_top >= min_window_height:
        windows.append(("below", {"x0": page_bounds["x0"], "y0": below_top, "x1": page_bounds["x1"], "y1": below_bottom}))

    best: Optional[Dict[str, Any]] = None
    for orientation, window_bbox in windows:
        selected: List[Dict[str, float]] = []
        for drawing_bbox in drawing_bboxes:
            clipped = _bbox_intersection(drawing_bbox, window_bbox)
            if not clipped:
                continue
            x_ok = caption_is_wide
            if not x_ok:
                if _bbox_x_overlap_ratio(clipped, caption_bbox) >= min_x_overlap:
                    x_ok = True
                else:
                    cx = _bbox_center_x(clipped)
                    x_ok = (cx >= x0_gate and cx <= x1_gate)
            if not x_ok:
                continue
            selected.append(clipped)
        if len(selected) < min_drawings:
            continue

        for cluster in _cluster_vector_bboxes(selected):
            raw_region_bbox = _region_from_cluster(cluster)
            if not raw_region_bbox:
                continue

            region_bbox = _clip_bbox_to_bounds(
                {
                    "x0": float(raw_region_bbox["x0"]) - margin,
                    "y0": float(raw_region_bbox["y0"]) - margin,
                    "x1": float(raw_region_bbox["x1"]) + margin,
                    "y1": float(raw_region_bbox["y1"]) + margin,
                },
                window_bbox,
            )
            if not region_bbox:
                continue
            region_bbox = _refine_vector_region_with_text(
                raw_bbox=raw_region_bbox,
                expanded_bbox=region_bbox,
                window_bbox=window_bbox,
                text_boxes=text_boxes,
                caption_text=str(caption.get("text") or ""),
            )
            if _rect_area(region_bbox) < min_region_area:
                continue
            if _bbox_width(region_bbox) < min_region_side or _bbox_height(region_bbox) < min_region_side:
                continue

            if orientation == "above":
                distance_penalty = max(0.0, float(caption_block_bbox["y0"]) - float(region_bbox["y1"]))
                orientation_bonus = 0.3
            else:
                distance_penalty = max(0.0, float(region_bbox["y0"]) - float(caption_block_bbox["y1"]))
                orientation_bonus = 0.0
            score = len(cluster) * 0.25 + (_rect_area(region_bbox) / 30000.0) - (distance_penalty / 140.0) + orientation_bonus
            if not best or score > float(best.get("score") or -10**9):
                best = {
                    "bbox": region_bbox,
                    "score": score,
                    "orientation": orientation,
                    "drawing_count": len(cluster),
                }

    return best


def _figure_root() -> Path:
    configured = os.getenv("FIGURE_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    root = (Path.cwd() / ".ia_phase1_data" / "figures").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _paper_dir(paper_id: int) -> Path:
    return _figure_root() / str(int(paper_id))


def _manifest_path(paper_id: int) -> Path:
    return _paper_dir(paper_id) / "manifest.json"


def _is_meaningful_image(width: int, height: int) -> bool:
    min_area = _safe_int(os.getenv("FIGURE_MIN_PIXEL_AREA", "4096"), 4096)
    min_side = _safe_int(os.getenv("FIGURE_MIN_SIDE_PX", "24"), 24)
    if width < min_side or height < min_side:
        return False
    return (width * height) >= max(1, min_area)


def _is_meaningful_render_rect(bbox: Optional[Dict[str, float]]) -> bool:
    if not bbox:
        return True
    width = max(0.0, float(bbox["x1"]) - float(bbox["x0"]))
    height = max(0.0, float(bbox["y1"]) - float(bbox["y0"]))
    min_side = _safe_float(os.getenv("FIGURE_MIN_RENDER_SIDE_PT", "18"), 18.0)
    min_area = _safe_float(os.getenv("FIGURE_MIN_RENDER_AREA", "800"), 800.0)
    if width < min_side or height < min_side:
        return False
    return (width * height) >= max(1.0, min_area)


def _prepare_page_blocks(blocks: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    page_map: Dict[int, List[Dict[str, Any]]] = {}
    for block in blocks:
        page_no = _safe_int(block.get("page_no"), 0)
        if page_no <= 0:
            continue
        metadata = block.get("metadata")
        block_meta = metadata if isinstance(metadata, dict) else {}
        section_canonical = str(block_meta.get("section_canonical") or "").strip() or "other"
        section_title = str(block_meta.get("section_title") or "").strip() or "Document Body"
        section_source = str(block_meta.get("section_source") or "").strip() or "fallback"
        section_confidence = _safe_float(block_meta.get("section_confidence"), 0.35)
        page_map.setdefault(page_no, []).append(
            {
                "page_no": page_no,
                "block_index": _safe_int(block.get("block_index"), 0),
                "bbox": _bbox_from_payload(block.get("bbox")),
                "section_canonical": section_canonical,
                "section_title": section_title,
                "section_source": section_source,
                "section_confidence": section_confidence,
            }
        )
    for page_no, page_blocks in page_map.items():
        page_blocks.sort(key=lambda item: item.get("block_index", 0))
        page_map[page_no] = page_blocks
    return page_map


def _infer_section_for_image(
    page_blocks: List[Dict[str, Any]],
    image_bbox: Optional[Dict[str, float]],
    *,
    caption_bbox: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    if not page_blocks:
        return {
            "section_canonical": "other",
            "section_title": "Document Body",
            "section_source": "fallback",
            "section_confidence": 0.25,
        }

    best_block: Optional[Dict[str, Any]] = None
    best_score = -1.0
    image_area = max(_rect_area(image_bbox), 1e-6)
    anchor_bbox = _bbox_union(image_bbox, caption_bbox) if caption_bbox else image_bbox

    for block in page_blocks:
        block_bbox = block.get("bbox")
        block_text = str(block.get("text") or "").strip()
        block_is_noise = _text_is_section_inference_noise(block_text)
        block_is_narrative = _text_is_prose_like(block_text) or _text_is_heading_like(block_text)
        overlap = _rect_overlap(image_bbox, block_bbox)
        score = 0.0
        if overlap > 0:
            block_area = max(_rect_area(block_bbox), 1e-6)
            overlap_ratio_image = overlap / image_area
            overlap_ratio_block = overlap / block_area
            score = (overlap_ratio_image * 0.8) + (overlap_ratio_block * 0.2)
            if block_is_noise:
                score *= 0.18
        elif image_bbox and block_bbox:
            # If no overlap, prefer nearest block vertically on the same page.
            distance = abs(_center_y(anchor_bbox) - _center_y(block_bbox))
            score = 0.05 / (1.0 + distance)
        else:
            score = 0.01

        if caption_bbox and block_bbox:
            x_coverage = _bbox_x_reference_coverage(block_bbox, caption_bbox)
            caption_gap = _vertical_gap(block_bbox, caption_bbox)
            if x_coverage >= 0.15 and caption_gap <= 220.0:
                score += (x_coverage * 0.45)
                if block_is_narrative:
                    score += 0.12
                if float(block_bbox["y0"]) >= float(caption_bbox["y1"]) - 6.0:
                    score += 0.18
                elif float(block_bbox["y1"]) <= float(caption_bbox["y0"]) + 6.0:
                    score += 0.05
            elif block_is_noise:
                score *= 0.6

        if score > best_score:
            best_score = score
            best_block = block

    if not best_block:
        return {
            "section_canonical": "other",
            "section_title": "Document Body",
            "section_source": "fallback",
            "section_confidence": 0.25,
        }

    inferred_confidence = max(
        0.25,
        min(
            0.98,
            _safe_float(best_block.get("section_confidence"), 0.35) * (0.8 + max(0.0, best_score)),
        ),
    )
    return {
        "section_canonical": str(best_block.get("section_canonical") or "other"),
        "section_title": str(best_block.get("section_title") or "Document Body"),
        "section_source": str(best_block.get("section_source") or "fallback"),
        "section_confidence": round(inferred_confidence, 3),
    }


def extract_and_store_paper_figures(
    pdf_path: Path,
    paper_id: int,
    blocks: Iterable[Dict[str, Any]],
    page_allowlist: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Extract embedded PDF figures for a paper and store a manifest mapped to sections.

    Returns a summary with extracted image count and manifest path.
    """
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    allowed_pages: Optional[Set[int]] = None
    if page_allowlist:
        allowed_pages = set()
        for value in page_allowlist:
            try:
                page_no = int(value)
            except (TypeError, ValueError):
                continue
            if page_no > 0:
                allowed_pages.add(page_no)
        if not allowed_pages:
            allowed_pages = None

    page_blocks = _prepare_page_blocks(blocks)
    output_dir = _paper_dir(paper_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale image files from previous extractions.
    for stale in output_dir.glob("page_*_img_*.*"):
        try:
            stale.unlink()
        except Exception:
            logger.warning("Failed to remove stale figure file %s", stale)
    for stale in output_dir.glob("page_*_vec_*.*"):
        try:
            stale.unlink()
        except Exception:
            logger.warning("Failed to remove stale figure file %s", stale)

    records: List[Dict[str, Any]] = []
    vector_enabled = os.getenv("FIGURE_VECTOR_ENABLED", "true").strip().lower() in {"1", "true", "yes"}
    vector_render_scale = max(0.5, _safe_float(os.getenv("FIGURE_VECTOR_RENDER_SCALE", "2.0"), 2.0))
    embedded_render_scale = max(0.5, _safe_float(os.getenv("FIGURE_EMBEDDED_RENDER_SCALE", str(vector_render_scale)), vector_render_scale))
    vector_dedup_iou = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_DEDUP_IOU", "0.82"), 0.82)))
    vector_skip_embedded_iou = min(1.0, max(0.0, _safe_float(os.getenv("FIGURE_VECTOR_EMBEDDED_IOU_SKIP", "0.88"), 0.88)))
    keep_captionless_embedded = os.getenv("FIGURE_KEEP_CAPTIONLESS_EMBEDDED", "").strip().lower() in {"1", "true", "yes"}
    dropped_captionless_embedded = 0
    detected_caption_numbers: set[str] = set()
    extracted_figure_numbers: set[str] = set()
    mentioned_figure_numbers = set(_extract_figure_mentions_from_blocks(blocks))

    doc = pymupdf.open(str(pdf_path))
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_no = page_index + 1
            if allowed_pages is not None and page_no not in allowed_pages:
                continue
            page_candidates: List[Dict[str, Any]] = []
            page_vector_bboxes: List[Optional[Dict[str, float]]] = []

            captions = _extract_figure_captions_from_page(page)
            for caption in captions:
                number = str(caption.get("figure_number") or "").strip()
                if number:
                    detected_caption_numbers.add(number)
            page_bounds = _page_bounds(page)

            if vector_enabled:
                drawing_bboxes = _collect_vector_drawing_bboxes(page)
                text_boxes = _extract_page_text_boxes(page)
                vec_idx = 0
                for caption_idx, caption in enumerate(captions):
                    candidate = _build_vector_region_from_caption(
                        caption=caption,
                        caption_index=caption_idx,
                        captions=captions,
                        drawing_bboxes=drawing_bboxes,
                        text_boxes=text_boxes,
                        page_bounds=page_bounds,
                    )
                    if not candidate:
                        continue
                    region_bbox = candidate.get("bbox")
                    if not _is_meaningful_render_rect(region_bbox):
                        continue
                    if _bbox_matches_any_iou(region_bbox, page_vector_bboxes, vector_dedup_iou):
                        continue

                    clip_rect = pymupdf.Rect(
                        float(region_bbox["x0"]),
                        float(region_bbox["y0"]),
                        float(region_bbox["x1"]),
                        float(region_bbox["y1"]),
                    )
                    try:
                        pix = page.get_pixmap(
                            matrix=pymupdf.Matrix(vector_render_scale, vector_render_scale),
                            clip=clip_rect,
                            alpha=False,
                        )
                    except Exception:
                        continue

                    vec_idx += 1
                    file_name = f"page_{page_no:03d}_vec_{vec_idx:03d}.png"
                    image_path = output_dir / file_name
                    try:
                        pix.save(str(image_path))
                    except Exception:
                        continue

                    section = _infer_section_for_image(
                        page_blocks.get(page_no, []),
                        region_bbox,
                        caption_bbox=caption.get("block_bbox") if isinstance(caption.get("block_bbox"), dict) else caption.get("bbox"),
                    )
                    caption_text = str(caption.get("text") or "").strip() or None
                    record = {
                        "id": len(records) + 1,
                        "paper_id": int(paper_id),
                        "page_no": page_no,
                        "file_name": file_name,
                        "image_path": str(image_path),
                        "url": f"/api/papers/{paper_id}/figures/{file_name}",
                        "width": int(getattr(pix, "width", 0)) or None,
                        "height": int(getattr(pix, "height", 0)) or None,
                        "bbox": region_bbox,
                        "section_canonical": section["section_canonical"],
                        "section_title": section["section_title"],
                        "section_source": section["section_source"],
                        "section_confidence": section["section_confidence"],
                        "figure_type": "vector",
                        "figure_caption": caption_text,
                        "figure_number": str(caption.get("figure_number") or "").strip() or None,
                        "figure_body": str(caption.get("figure_body") or "").strip() or None,
                        "figure_label": str(caption.get("figure_label") or "Figure").strip() or "Figure",
                        "vector_orientation": candidate.get("orientation"),
                        "vector_drawing_count": int(candidate.get("drawing_count") or 0),
                        "_candidate_score": _score_figure_region_for_caption(
                            bbox=region_bbox,
                            caption_bbox=caption.get("block_bbox") if isinstance(caption.get("block_bbox"), dict) else caption.get("bbox"),
                            kind="vector",
                            density_count=int(candidate.get("drawing_count") or 0),
                            base_score=float(candidate.get("score") or 0.0),
                        ),
                    }
                    page_candidates.append(record)
                    page_vector_bboxes.append(region_bbox)

            embedded_regions = _build_embedded_regions(page, doc)
            for region in embedded_regions:
                refined_bbox = _trim_embedded_region_away_from_captions(
                    region_bbox=region.get("bbox") if isinstance(region.get("bbox"), dict) else None,
                    captions=captions,
                    page_bounds=page_bounds,
                )
                if refined_bbox:
                    region["bbox"] = refined_bbox
            embedded_regions = [
                region
                for region in embedded_regions
                if _is_meaningful_render_rect(region.get("bbox") if isinstance(region.get("bbox"), dict) else None)
            ]
            _assign_captions_to_regions(regions=embedded_regions, captions=captions)
            _recover_unassigned_embedded_captions(regions=embedded_regions, captions=captions)
            _merge_caption_adjacent_embedded_regions(
                regions=embedded_regions,
                captions=captions,
                page_bounds=page_bounds,
            )
            img_idx = 0
            for region in embedded_regions:
                image_bbox = region.get("bbox")
                if not _is_meaningful_render_rect(image_bbox):
                    continue
                region_caption = str(region.get("figure_caption") or "").strip()
                region_number = str(region.get("figure_number") or "").strip()
                if not (region_caption or region_number or str(region.get("figure_body") or "").strip()):
                    if not keep_captionless_embedded:
                        dropped_captionless_embedded += 1
                        continue

                clip_rect = pymupdf.Rect(
                    float(image_bbox["x0"]),
                    float(image_bbox["y0"]),
                    float(image_bbox["x1"]),
                    float(image_bbox["y1"]),
                )
                try:
                    pix = page.get_pixmap(
                        matrix=pymupdf.Matrix(embedded_render_scale, embedded_render_scale),
                        clip=clip_rect,
                        alpha=False,
                    )
                except Exception:
                    continue

                img_idx += 1
                file_name = f"page_{page_no:03d}_img_{img_idx:03d}.png"
                image_path = output_dir / file_name
                try:
                    pix.save(str(image_path))
                except Exception:
                    continue

                section = _infer_section_for_image(
                    page_blocks.get(page_no, []),
                    image_bbox,
                    caption_bbox=region.get("caption_block_bbox") if isinstance(region.get("caption_block_bbox"), dict) else region.get("caption_bbox"),
                )
                record = {
                    "id": len(records) + 1,
                    "paper_id": int(paper_id),
                    "page_no": page_no,
                    "file_name": file_name,
                    "image_path": str(image_path),
                    "url": f"/api/papers/{paper_id}/figures/{file_name}",
                    "width": int(getattr(pix, "width", 0)) or None,
                    "height": int(getattr(pix, "height", 0)) or None,
                    "bbox": image_bbox,
                    "section_canonical": section["section_canonical"],
                    "section_title": section["section_title"],
                    "section_source": section["section_source"],
                    "section_confidence": section["section_confidence"],
                    "figure_type": "embedded",
                    "figure_caption": str(region.get("figure_caption") or "").strip() or None,
                    "figure_number": str(region.get("figure_number") or "").strip() or None,
                    "figure_body": str(region.get("figure_body") or "").strip() or None,
                    "figure_label": str(region.get("figure_label") or "Figure").strip() or "Figure",
                    "embedded_tile_count": int(region.get("tile_count") or 1),
                    "_candidate_score": _score_figure_region_for_caption(
                        bbox=image_bbox,
                        caption_bbox=region.get("caption_block_bbox") if isinstance(region.get("caption_block_bbox"), dict) else region.get("caption_bbox"),
                        kind="embedded",
                        density_count=int(region.get("tile_count") or 1),
                    ),
                }
                page_candidates.append(record)

            resolved_page_records = _resolve_page_figure_candidates(page_candidates)
            for record in resolved_page_records:
                record.pop("_candidate_score", None)
                record["id"] = len(records) + 1
                records.append(record)
                figure_number = str(record.get("figure_number") or "").strip()
                if figure_number:
                    extracted_figure_numbers.add(figure_number)
    finally:
        doc.close()

    missing_caption_numbers = sorted(detected_caption_numbers - extracted_figure_numbers, key=_figure_number_sort_key)
    missing_mentioned_numbers = sorted(mentioned_figure_numbers - extracted_figure_numbers, key=_figure_number_sort_key)
    extracted_numbers_sorted = sorted(extracted_figure_numbers, key=_figure_number_sort_key)
    detected_caption_numbers_sorted = sorted(detected_caption_numbers, key=_figure_number_sort_key)
    mentioned_numbers_sorted = sorted(mentioned_figure_numbers, key=_figure_number_sort_key)

    manifest = {
        "paper_id": int(paper_id),
        "source_pdf": str(pdf_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_images": len(records),
        "images": records,
        "audit": {
            "detected_caption_numbers": detected_caption_numbers_sorted,
            "mentioned_figure_numbers": mentioned_numbers_sorted,
            "extracted_figure_numbers": extracted_numbers_sorted,
            "missing_detected_caption_numbers": missing_caption_numbers,
            "missing_mentioned_numbers": missing_mentioned_numbers,
            "captionless_embedded_dropped": int(dropped_captionless_embedded),
        },
    }
    manifest_path = _manifest_path(paper_id)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    if missing_caption_numbers:
        logger.warning(
            "Figure extraction missing caption-backed figure numbers for paper %s: %s",
            paper_id,
            ", ".join(missing_caption_numbers),
        )

    return {
        "paper_id": int(paper_id),
        "num_images": len(records),
        "manifest_path": str(manifest_path),
    }


def load_paper_figure_manifest(paper_id: int) -> Dict[str, Any]:
    path = _manifest_path(paper_id)
    if not path.exists():
        return {"paper_id": int(paper_id), "num_images": 0, "images": []}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read figure manifest for paper %s: %s", paper_id, exc)
        return {"paper_id": int(paper_id), "num_images": 0, "images": []}
    if not isinstance(payload, dict):
        return {"paper_id": int(paper_id), "num_images": 0, "images": []}
    images = payload.get("images")
    if not isinstance(images, list):
        images = []
    payload["images"] = images
    payload["num_images"] = _safe_int(payload.get("num_images"), len(images))
    payload["paper_id"] = _safe_int(payload.get("paper_id"), int(paper_id))
    return payload


def resolve_figure_file(paper_id: int, file_name: str) -> Path:
    safe_name = Path(file_name).name
    if safe_name != file_name:
        raise ValueError("Invalid figure file name.")
    path = _paper_dir(paper_id) / safe_name
    return path
