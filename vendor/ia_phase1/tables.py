from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pymupdf

logger = logging.getLogger(__name__)

_TABLE_CAPTION_RE = re.compile(r"^\s*(table|tab\.)\s*\d+\b", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.)\s*\d+\b", re.IGNORECASE)
_BASE64_LIKE_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")


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


def _bbox_from_payload(payload: Any) -> Optional[Dict[str, float]]:
    if not isinstance(payload, dict):
        return None
    x0 = _safe_float(payload.get("x0"), float("nan"))
    y0 = _safe_float(payload.get("y0"), float("nan"))
    x1 = _safe_float(payload.get("x1"), float("nan"))
    y1 = _safe_float(payload.get("y1"), float("nan"))
    if any(v != v for v in (x0, y0, x1, y1)):
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _bbox_from_tuple(value: Any) -> Optional[Dict[str, float]]:
    if not isinstance(value, (tuple, list)) or len(value) < 4:
        return None
    return {
        "x0": _safe_float(value[0]),
        "y0": _safe_float(value[1]),
        "x1": _safe_float(value[2]),
        "y1": _safe_float(value[3]),
    }


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


def _vertical_gap(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return float("inf")
    if float(a["y1"]) < float(b["y0"]):
        return float(b["y0"]) - float(a["y1"])
    if float(b["y1"]) < float(a["y0"]):
        return float(a["y0"]) - float(b["y1"])
    return 0.0


def _center_y(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return (bbox["y0"] + bbox["y1"]) * 0.5


def _bbox_width(bbox: Optional[Dict[str, float]]) -> float:
    if not bbox:
        return 0.0
    return max(0.0, float(bbox["x1"]) - float(bbox["x0"]))


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


def _x_overlap_ratio(a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]) -> float:
    if not a or not b:
        return 0.0
    ix0 = max(float(a["x0"]), float(b["x0"]))
    ix1 = min(float(a["x1"]), float(b["x1"]))
    overlap = max(0.0, ix1 - ix0)
    base = max(1e-6, min(_bbox_width(a), _bbox_width(b)))
    return overlap / base


def _merge_table_candidates(
    candidates: List[Dict[str, Any]],
    *,
    gap_limit: Optional[float] = None,
    overlap_min: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if len(candidates) <= 1:
        return candidates

    if gap_limit is None:
        gap_limit = max(0.0, _safe_float(os.getenv("TABLE_MERGE_VERTICAL_GAP_PT", "14"), 14.0))
    if overlap_min is None:
        overlap_min = min(1.0, max(0.0, _safe_float(os.getenv("TABLE_MERGE_X_OVERLAP_RATIO", "0.9"), 0.9)))

    ordered = sorted(candidates, key=lambda item: float(item["bbox"]["y0"]) if item.get("bbox") else 0.0)
    merged: List[Dict[str, Any]] = []
    for candidate in ordered:
        if not merged:
            merged.append(candidate)
            continue

        prev = merged[-1]
        prev_bbox = prev.get("bbox")
        cur_bbox = candidate.get("bbox")
        prev_cols = int(prev.get("n_cols") or 0)
        cur_cols = int(candidate.get("n_cols") or 0)

        can_merge = False
        if prev_bbox and cur_bbox and prev_cols > 0 and cur_cols > 0:
            y_gap = float(cur_bbox["y0"]) - float(prev_bbox["y1"])
            x_overlap = _x_overlap_ratio(prev_bbox, cur_bbox)
            rows_small = int(prev.get("raw_row_count") or 0) <= 2 or int(candidate.get("raw_row_count") or 0) <= 2
            if y_gap >= -2.0 and y_gap <= gap_limit and x_overlap >= overlap_min and prev_cols == cur_cols and rows_small:
                can_merge = True

        if can_merge:
            prev["matrix"].extend(candidate.get("matrix") or [])
            prev["bbox"] = _bbox_union(prev_bbox, cur_bbox)
            prev["raw_row_count"] = int(prev.get("raw_row_count") or 0) + int(candidate.get("raw_row_count") or 0)
            prev["table_obj"] = None
            prev["merged_fragments"] = int(prev.get("merged_fragments") or 1) + int(candidate.get("merged_fragments") or 1)
        else:
            merged.append(candidate)
    return merged


def _bbox_matches_any_iou(
    bbox: Optional[Dict[str, float]],
    existing: Iterable[Optional[Dict[str, float]]],
    threshold: float,
) -> bool:
    if not bbox:
        return False
    for other in existing:
        if _rect_iou(bbox, other) >= threshold:
            return True
    return False


def _extract_page_text_blocks(page: Any) -> List[Dict[str, Any]]:
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return []
    items: List[Dict[str, Any]] = []
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = _bbox_from_tuple(block.get("bbox"))
        if not bbox:
            continue
        text = _extract_block_text(block)
        if not text:
            continue
        items.append({"bbox": bbox, "text": text})
    items.sort(key=lambda item: (float(item["bbox"]["y0"]), float(item["bbox"]["x0"])))
    return items


def _collect_caption_blocks(page: Any, pattern: re.Pattern[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in _extract_page_text_blocks(page):
        text = str(item.get("text") or "")
        if not pattern.search(text):
            continue
        if pattern is _TABLE_CAPTION_RE and not _looks_like_explicit_table_caption(text):
            continue
        items.append(item)
    return items


def _find_nearest_caption_block(
    caption_blocks: List[Dict[str, Any]],
    target_bbox: Optional[Dict[str, float]],
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if not caption_blocks or not target_bbox:
        return None, None

    target_top = float(target_bbox["y0"])
    target_bottom = float(target_bbox["y1"])
    best_idx: Optional[int] = None
    best_block: Optional[Dict[str, Any]] = None
    best_score = float("inf")

    for idx, block in enumerate(caption_blocks):
        bbox = block.get("bbox")
        if not bbox:
            continue
        score = _caption_block_match_score(bbox, target_bbox)
        if score < best_score:
            best_score = score
            best_idx = idx
            best_block = block
    return best_idx, best_block


def _caption_block_match_score(
    caption_bbox: Optional[Dict[str, float]],
    target_bbox: Optional[Dict[str, float]],
) -> float:
    if not caption_bbox or not target_bbox:
        return float("inf")

    target_top = float(target_bbox["y0"])
    target_bottom = float(target_bbox["y1"])
    block_top = float(caption_bbox["y0"])
    block_bottom = float(caption_bbox["y1"])
    x_overlap = _x_overlap_ratio(caption_bbox, target_bbox)
    target_center_x = (float(target_bbox["x0"]) + float(target_bbox["x1"])) * 0.5
    block_center_x = (float(caption_bbox["x0"]) + float(caption_bbox["x1"])) * 0.5
    center_distance = abs(target_center_x - block_center_x)

    if block_bottom <= target_top + 4.0:
        distance = target_top - block_bottom
        penalty = 0.0
    elif block_top >= target_bottom - 4.0:
        distance = block_top - target_bottom
        penalty = 12.0
    else:
        distance = 0.0
        penalty = 20.0

    caption_width = max(1e-6, float(caption_bbox["x1"]) - float(caption_bbox["x0"]))
    target_width = max(1e-6, float(target_bbox["x1"]) - float(target_bbox["x0"]))
    width_ratio = target_width / caption_width

    score = distance + penalty
    score += (1.0 - x_overlap) * 90.0
    score += center_distance / 10.0
    if width_ratio < 0.58 and center_distance > caption_width * 0.18:
        score += 55.0
    return score


def _resolve_candidate_caption_binding(
    *,
    caption_blocks: List[Dict[str, Any]],
    candidate_bbox: Optional[Dict[str, float]],
    detection_strategy: str,
    seed_caption_index: Any,
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    nearest_index, nearest_block = _find_nearest_caption_block(caption_blocks, candidate_bbox)
    if not (
        detection_strategy == "text_caption_fallback"
        and isinstance(seed_caption_index, int)
        and 0 <= seed_caption_index < len(caption_blocks)
    ):
        return nearest_index, nearest_block

    seed_block = caption_blocks[seed_caption_index]
    seed_bbox = seed_block.get("bbox")
    seed_score = _caption_block_match_score(seed_bbox, candidate_bbox)
    nearest_score = _caption_block_match_score(
        nearest_block.get("bbox") if nearest_block else None,
        candidate_bbox,
    )

    if nearest_block is None:
        return seed_caption_index, seed_block
    if nearest_index == seed_caption_index:
        return seed_caption_index, seed_block
    if nearest_score + 18.0 < seed_score:
        return nearest_index, nearest_block
    return seed_caption_index, seed_block


def _looks_like_sentenceish_prose(text: str) -> bool:
    lowered = str(text or "").lower()
    if re.search(r"[.!?]\s", text):
        return True
    stopwords = re.findall(r"\b(the|and|that|with|from|this|these|our|their|which|while|using|without|into|through|because|however|although|especially)\b", lowered)
    return len(stopwords) >= 3


def _looks_like_section_boundary_text(text: str) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return False
    if _TABLE_CAPTION_RE.search(compact) or _FIGURE_CAPTION_RE.search(compact):
        return True
    return bool(re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z]", compact))


def _bbox_to_rect_tuple(bbox: Optional[Dict[str, float]]) -> Optional[Tuple[float, float, float, float]]:
    if not bbox:
        return None
    return (
        float(bbox["x0"]),
        float(bbox["y0"]),
        float(bbox["x1"]),
        float(bbox["y1"]),
    )


def _build_candidates_from_table_objects(
    table_objects: Iterable[Any],
    *,
    min_area: float,
    min_cols: int,
    detection_strategy: str,
    seed_caption: Optional[str] = None,
    seed_caption_id: Optional[int] = None,
    seed_caption_bbox: Optional[Dict[str, float]] = None,
    clip_bbox: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for table_obj in table_objects:
        bbox = _bbox_from_tuple(getattr(table_obj, "bbox", None))
        if _rect_area(bbox) < min_area:
            continue
        try:
            raw_rows = table_obj.extract()
        except Exception:
            raw_rows = []
        matrix = _normalize_matrix(raw_rows)
        if not matrix:
            continue
        n_cols_raw = len(matrix[0]) if matrix else 0
        if n_cols_raw < min_cols:
            continue
        candidates.append(
            {
                "bbox": bbox,
                "matrix": matrix,
                "table_obj": table_obj,
                "n_cols": n_cols_raw,
                "raw_row_count": len(matrix),
                "merged_fragments": 1,
                "detection_strategy": detection_strategy,
                "seed_caption": seed_caption,
                "seed_caption_id": seed_caption_id,
                "seed_caption_bbox": seed_caption_bbox,
                "clip_bbox": clip_bbox,
            }
        )
    return candidates


def _page_bbox(page: Any) -> Dict[str, float]:
    rect = page.rect
    return {
        "x0": float(rect.x0),
        "y0": float(rect.y0),
        "x1": float(rect.x1),
        "y1": float(rect.y1),
    }


def _passes_text_fallback_constraints(
    table_bbox: Optional[Dict[str, float]],
    page_bbox: Optional[Dict[str, float]],
    caption_bbox: Optional[Dict[str, float]],
) -> bool:
    if not table_bbox or not page_bbox or not caption_bbox:
        return False

    max_caption_gap = max(0.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_CAPTION_GAP_PT", "110"), 110.0))
    if _vertical_gap(table_bbox, caption_bbox) > max_caption_gap:
        return False

    min_x_overlap = min(1.0, max(0.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MIN_X_OVERLAP_RATIO", "0.15"), 0.15)))
    if _x_overlap_ratio(table_bbox, caption_bbox) < min_x_overlap:
        return False

    page_area = max(1.0, _rect_area(page_bbox))
    max_page_area_ratio = min(1.0, max(0.01, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MAX_PAGE_AREA_RATIO", "0.65"), 0.65)))
    if (_rect_area(table_bbox) / page_area) > max_page_area_ratio:
        return False

    page_width = max(1e-6, _bbox_width(page_bbox))
    min_width_ratio = min(1.0, max(0.01, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MIN_WIDTH_RATIO", "0.20"), 0.20)))
    if (_bbox_width(table_bbox) / page_width) < min_width_ratio:
        return False

    return True


def _score_text_fallback_candidate(
    candidate: Dict[str, Any],
    *,
    caption_bbox: Dict[str, float],
    page_bbox: Dict[str, float],
) -> float:
    bbox = candidate.get("bbox")
    if not isinstance(bbox, dict):
        return -10**9
    gap = _vertical_gap(bbox, caption_bbox)
    x_overlap = _x_overlap_ratio(bbox, caption_bbox)
    page_area = max(1.0, _rect_area(page_bbox))
    area_ratio = _rect_area(bbox) / page_area
    row_count = int(candidate.get("raw_row_count") or 0)
    col_count = int(candidate.get("n_cols") or 0)
    strategy = str(candidate.get("detection_strategy") or "")
    is_above_caption = float(bbox["y1"]) <= float(caption_bbox["y0"]) + 4.0

    score = 0.0
    score += x_overlap * 3.0
    score -= gap / 180.0
    score -= max(0.0, area_ratio - 0.35) * 2.2
    score += min(1.0, row_count / 18.0) * 0.35
    score += min(1.0, col_count / 10.0) * 0.25
    if strategy == "caption_guided_native":
        score += 0.3
    if is_above_caption:
        score += 0.2
    return score


def _build_caption_guided_text_candidates(
    page: Any,
    caption_blocks: List[Dict[str, Any]],
    *,
    min_area: float,
    min_cols: int,
    caption_indices: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    if not caption_blocks:
        return []

    x_margin = max(0.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_X_MARGIN_PT", "8"), 8.0))
    max_above = max(24.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MAX_ABOVE_PT", "320"), 320.0))
    max_below = max(24.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MAX_BELOW_PT", "260"), 260.0))
    min_clip_height = max(20.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MIN_CLIP_HEIGHT_PT", "48"), 48.0))
    max_candidates_per_caption = max(1, _safe_int(os.getenv("TABLE_TEXT_FALLBACK_MAX_CANDIDATES_PER_CAPTION", "1"), 1))
    caption_native_merge_gap = max(
        12.0,
        _safe_float(os.getenv("TABLE_CAPTION_GUIDED_NATIVE_MERGE_GAP_PT", "28"), 28.0),
    )
    caption_native_overlap = min(
        1.0,
        max(0.0, _safe_float(os.getenv("TABLE_CAPTION_GUIDED_NATIVE_MERGE_X_OVERLAP_RATIO", "0.82"), 0.82)),
    )

    page_bounds = _page_bbox(page)
    clip_x0 = min(page_bounds["x1"], max(page_bounds["x0"], page_bounds["x0"] + x_margin))
    clip_x1 = max(page_bounds["x0"], min(page_bounds["x1"], page_bounds["x1"] - x_margin))
    if clip_x1 <= clip_x0:
        clip_x0 = page_bounds["x0"]
        clip_x1 = page_bounds["x1"]

    all_candidates: List[Dict[str, Any]] = []
    for idx, caption in enumerate(caption_blocks):
        if caption_indices is not None and idx not in caption_indices:
            continue
        caption_bbox = caption.get("bbox")
        caption_text = str(caption.get("text") or "")
        if not caption_bbox:
            continue

        prev_caption_bbox = caption_blocks[idx - 1].get("bbox") if idx > 0 else None
        next_caption_bbox = caption_blocks[idx + 1].get("bbox") if idx + 1 < len(caption_blocks) else None

        clip_bboxes: List[Dict[str, float]] = []

        above_bottom = float(caption_bbox["y0"]) - 2.0
        above_floor = float(page_bounds["y0"]) + 2.0
        same_column_prev = False
        if prev_caption_bbox:
            above_floor = max(above_floor, float(prev_caption_bbox["y1"]) + 2.0)
            same_column_prev = _x_overlap_ratio(prev_caption_bbox, caption_bbox) >= 0.6
        above_top = max(above_floor, above_bottom - max_above)
        allow_above_clip = not (
            prev_caption_bbox
            and same_column_prev
            and (float(caption_bbox["y0"]) - float(prev_caption_bbox["y1"])) <= (max_above + 24.0)
        )
        if allow_above_clip and above_bottom - above_top >= min_clip_height:
            clip_bboxes.append({"x0": clip_x0, "y0": above_top, "x1": clip_x1, "y1": above_bottom})

        below_top = float(caption_bbox["y1"]) + 2.0
        below_ceiling = float(page_bounds["y1"]) - 2.0
        if next_caption_bbox:
            below_ceiling = min(below_ceiling, float(next_caption_bbox["y0"]) - 2.0)
        below_bottom = min(below_ceiling, below_top + max_below)
        if below_bottom - below_top >= min_clip_height:
            clip_bboxes.append({"x0": clip_x0, "y0": below_top, "x1": clip_x1, "y1": below_bottom})

        caption_candidates: List[Dict[str, Any]] = []
        for clip_bbox in clip_bboxes:
            clip_rect = _bbox_to_rect_tuple(clip_bbox)
            if not clip_rect:
                continue
            try:
                native_finder = page.find_tables(strategy="lines", clip=clip_rect)
            except Exception:
                native_finder = None
            native_tables = getattr(native_finder, "tables", None) or []
            if native_tables:
                native_candidates = _build_candidates_from_table_objects(
                    native_tables,
                    min_area=min_area,
                    min_cols=min_cols,
                    detection_strategy="caption_guided_native",
                    seed_caption=caption_text,
                    seed_caption_id=idx,
                    seed_caption_bbox=caption_bbox,
                    clip_bbox=clip_bbox,
                )
                caption_candidates.extend(
                    _merge_table_candidates(
                        native_candidates,
                        gap_limit=caption_native_merge_gap,
                        overlap_min=caption_native_overlap,
                    )
                )
            try:
                finder = page.find_tables(strategy="text", clip=clip_rect)
            except Exception:
                continue
            text_tables = getattr(finder, "tables", None) or []
            if not text_tables:
                continue
            caption_candidates.extend(
                _build_candidates_from_table_objects(
                    text_tables,
                    min_area=min_area,
                    min_cols=min_cols,
                    detection_strategy="text_caption_fallback",
                    seed_caption=caption_text,
                    seed_caption_id=idx,
                    seed_caption_bbox=caption_bbox,
                    clip_bbox=clip_bbox,
                )
            )

        if not caption_candidates:
            continue
        caption_candidates = _merge_table_candidates(caption_candidates)
        caption_candidates.sort(
            key=lambda item: _score_text_fallback_candidate(
                item,
                caption_bbox=caption_bbox,
                page_bbox=page_bounds,
            ),
            reverse=True,
        )
        all_candidates.extend(caption_candidates[:max_candidates_per_caption])

    return _merge_table_candidates(all_candidates)


def _default_table_dir() -> Path:
    root = (Path.cwd() / ".ia_phase1_data" / "tables").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _table_root() -> Path:
    configured = os.getenv("TABLE_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _default_table_dir()


def _paper_dir(paper_id: int) -> Path:
    return _table_root() / str(int(paper_id))


def _manifest_path(paper_id: int) -> Path:
    return _paper_dir(paper_id) / "manifest.json"


def _table_enabled() -> bool:
    raw = os.getenv("TABLE_EXTRACTION_ENABLED", "true").strip().lower()
    return raw in {"1", "true", "yes"}


def _table_text_fallback_enabled() -> bool:
    # Keep the permissive text-based fallback explicitly opt-in. The native
    # PyMuPDF detector is materially less likely to convert nearby prose into
    # fake tables, which is the failure mode we want to avoid by default.
    raw = os.getenv("TABLE_TEXT_FALLBACK_ENABLED", "false").strip().lower()
    return raw in {"1", "true", "yes"}


def _table_auto_text_fallback_enabled() -> bool:
    # Auto fallback is intentionally narrower than TABLE_TEXT_FALLBACK_ENABLED:
    # it only fires for caption-backed regions that native detection did not
    # resolve, and it applies a stricter quality gate before accepting the
    # reconstructed candidate.
    raw = os.getenv("TABLE_AUTO_TEXT_FALLBACK_ENABLED", "true").strip().lower()
    return raw in {"1", "true", "yes"}


def _prepare_page_blocks(blocks: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    page_map: Dict[int, List[Dict[str, Any]]] = {}
    for block in blocks:
        page_no = _safe_int(block.get("page_no"), 0)
        if page_no <= 0:
            continue
        block_meta_raw = block.get("metadata")
        block_meta = block_meta_raw if isinstance(block_meta_raw, dict) else {}
        page_map.setdefault(page_no, []).append(
            {
                "page_no": page_no,
                "block_index": _safe_int(block.get("block_index"), 0),
                "bbox": _bbox_from_payload(block.get("bbox")),
                "section_canonical": str(block_meta.get("section_canonical") or "other"),
                "section_title": str(block_meta.get("section_title") or "Document Body"),
                "section_source": str(block_meta.get("section_source") or "fallback"),
                "section_confidence": _safe_float(block_meta.get("section_confidence"), 0.35),
            }
        )
    for page_no, page_blocks in page_map.items():
        page_blocks.sort(key=lambda item: item.get("block_index", 0))
        page_map[page_no] = page_blocks
    return page_map


def _infer_section_for_table(
    page_blocks: List[Dict[str, Any]],
    table_bbox: Optional[Dict[str, float]],
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
    table_area = max(_rect_area(table_bbox), 1e-6)

    for block in page_blocks:
        block_bbox = block.get("bbox")
        overlap = _rect_overlap(table_bbox, block_bbox)
        score = 0.0
        if overlap > 0:
            block_area = max(_rect_area(block_bbox), 1e-6)
            score = (overlap / table_area) * 0.8 + (overlap / block_area) * 0.2
        elif table_bbox and block_bbox:
            distance = abs(_center_y(table_bbox) - _center_y(block_bbox))
            score = 0.04 / (1.0 + distance)
        else:
            score = 0.01
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

    confidence = max(
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
        "section_confidence": round(confidence, 3),
    }


def _clean_cell(value: Any, *, keep_newlines: bool = False) -> str:
    text = str(value or "").replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if keep_newlines:
        lines = [" ".join(line.split()).strip() for line in text.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines).strip()
    text = " ".join(text.split())
    return text.strip()


def _normalize_matrix(raw_rows: Any) -> List[List[str]]:
    if not isinstance(raw_rows, list):
        return []
    matrix: List[List[str]] = []
    max_cols = 0
    for row in raw_rows:
        if not isinstance(row, (list, tuple)):
            continue
        cleaned = [_clean_cell(cell, keep_newlines=True) for cell in row]
        if not any(cleaned):
            continue
        matrix.append(cleaned)
        max_cols = max(max_cols, len(cleaned))

    if not matrix or max_cols <= 0:
        return []

    for row in matrix:
        if len(row) < max_cols:
            row.extend([""] * (max_cols - len(row)))
        elif len(row) > max_cols:
            del row[max_cols:]
    return matrix


def _pick_headers_and_rows(matrix: List[List[str]], table_obj: Any) -> Tuple[List[str], List[List[str]]]:
    header_names: List[str] = []
    header_obj = getattr(table_obj, "header", None)
    header_candidates = getattr(header_obj, "names", None) if header_obj is not None else None
    if isinstance(header_candidates, (list, tuple)):
        header_names = [_clean_cell(item) for item in header_candidates]
        if not any(header_names):
            header_names = []

    if not matrix:
        return header_names, []

    n_cols = len(matrix[0])
    if header_names and len(header_names) < n_cols:
        header_names.extend([""] * (n_cols - len(header_names)))
    if header_names and len(header_names) > n_cols:
        header_names = header_names[:n_cols]

    if header_names:
        body_rows = matrix
        if matrix:
            first_row = [_clean_cell(cell) for cell in matrix[0][:n_cols]]
            padded_header = header_names[:n_cols] + [""] * max(0, n_cols - len(header_names))
            if first_row == padded_header:
                body_rows = matrix[1:]
        return _refine_headers_and_rows(header_names, body_rows)

    first = matrix[0]
    if len(matrix) >= 2 and sum(1 for cell in first if cell) >= max(1, len(first) // 2):
        return _refine_headers_and_rows(first, matrix[1:])

    generated = [f"col_{idx + 1}" for idx in range(n_cols)]
    return _refine_headers_and_rows(generated, matrix)


def _refine_headers_and_rows(headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    clean_headers = [_clean_cell(cell) for cell in headers]
    clean_rows = [[_clean_cell(cell, keep_newlines=True) for cell in row[: len(clean_headers)]] for row in rows]
    clean_rows = [row + [""] * max(0, len(clean_headers) - len(row)) for row in clean_rows]
    clean_rows = [row[: len(clean_headers)] for row in clean_rows if any(_clean_cell(cell) for cell in row)]
    clean_headers, clean_rows = _merge_header_continuation_row(clean_headers, clean_rows)
    clean_headers, clean_rows = _merge_continuation_columns(clean_headers, clean_rows)
    clean_headers, clean_rows = _repair_leading_text_fragment_columns(clean_headers, clean_rows)
    clean_headers, clean_rows = _merge_sparse_pre_numeric_text_columns(clean_headers, clean_rows)
    clean_headers, clean_rows = _drop_empty_columns(clean_headers, clean_rows)
    clean_rows = _trim_prose_tail(clean_rows)
    clean_rows = _trim_chart_scaffold_tail(clean_headers, clean_rows)
    return clean_headers, clean_rows


def _merge_header_continuation_row(headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    if not headers or not rows:
        return headers, rows
    row = rows[0]
    nonempty = [cell for cell in row if _clean_cell(cell)]
    if not nonempty:
        return headers, rows[1:]
    unit_like = 0
    marker_like = 0
    for cell in nonempty:
        compact = _clean_cell(cell)
        if len(compact) <= 10 and re.fullmatch(r"[\(\)\[\]%A-Za-z0-9˚/.\- ]+", compact):
            unit_like += 1
        if any(marker in compact for marker in ("(", ")", "%", "˚", "/")):
            marker_like += 1
    if marker_like < max(1, len(nonempty) // 2):
        return headers, rows
    if unit_like < max(2, len(nonempty) - 1):
        return headers, rows

    merged_headers: List[str] = []
    for header, suffix in zip(headers, row):
        header_text = _clean_cell(header)
        suffix_text = _clean_cell(suffix)
        if suffix_text:
            merged_headers.append(" ".join(part for part in [header_text, suffix_text] if part).strip())
        else:
            merged_headers.append(header_text)
    return merged_headers, rows[1:]


def _merge_continuation_columns(headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    if len(headers) <= 1:
        return headers, rows

    headers = list(headers)
    rows = [list(row) for row in rows]
    idx = 1
    while idx < len(headers):
        header = _clean_cell(headers[idx])
        prev_header = _clean_cell(headers[idx - 1])
        column_cells = [_clean_cell(row[idx]) for row in rows if idx < len(row) and _clean_cell(row[idx])]
        if header or not prev_header or not column_cells:
            idx += 1
            continue

        textish = 0
        numericish = 0
        for cell in column_cells:
            if re.fullmatch(r"[-–—+]?[\d.xX×%/]+", cell):
                numericish += 1
            elif re.search(r"[A-Za-z]", cell):
                textish += 1
        if textish < max(1, numericish):
            idx += 1
            continue

        for row in rows:
            if idx >= len(row):
                continue
            left = _clean_cell(row[idx - 1], keep_newlines=True)
            right = _clean_cell(row[idx], keep_newlines=True)
            if right:
                row[idx - 1] = " ".join(part for part in [left, right] if part).strip()
            del row[idx]
        del headers[idx]
    return headers, rows


def _join_fragment_parts(left: str, right: str) -> str:
    left_text = _clean_cell(left, keep_newlines=True)
    right_text = _clean_cell(right, keep_newlines=True)
    if not left_text:
        return right_text
    if not right_text:
        return left_text
    if right_text[:1] in {"-", ")", "]", "%", "/", ".", ",", ":"}:
        return f"{left_text}{right_text}"
    if left_text.endswith(("(", "[", "{", "-", "/")):
        return f"{left_text}{right_text}"
    if right_text[:1].islower():
        return f"{left_text}{right_text}"
    if len(left_text) <= 3 and right_text[:1].isalpha():
        return f"{left_text}{right_text}"
    if re.search(r"[A-Za-z]$", left_text) and re.fullmatch(r"[A-Z]{2,}", right_text):
        return f"{left_text}{right_text}"
    return f"{left_text} {right_text}".strip()


def _column_numericish_ratio(rows: List[List[str]], idx: int) -> float:
    total = 0
    numericish = 0
    for row in rows:
        if idx >= len(row):
            continue
        cell = _clean_cell(row[idx])
        if not cell:
            continue
        total += 1
        if re.search(r"\d", cell) or cell in {"–", "-", "OOM"}:
            numericish += 1
    return (numericish / total) if total else 0.0


def _looks_like_fragment_pair(left: str, right: str) -> bool:
    left_text = _clean_cell(left)
    right_text = _clean_cell(right)
    if not left_text or not right_text:
        return False
    if re.search(r"\d", left_text) or re.search(r"\d", right_text):
        return False
    if right_text[:1] in {"-", ")", "]"}:
        return True
    if right_text[:1].islower():
        return True
    if len(left_text) <= 3 and right_text[:1].isalpha():
        return True
    if re.search(r"[A-Za-z]$", left_text) and re.fullmatch(r"[A-Z]{2,}", right_text):
        return True
    return False


def _repair_leading_text_fragment_columns(headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    if len(headers) <= 2 or not rows:
        return headers, rows

    headers = list(headers)
    rows = [list(row) for row in rows]

    numeric_boundary = len(headers)
    for idx in range(len(headers)):
        if _column_numericish_ratio(rows, idx) >= 0.55:
            numeric_boundary = idx
            break
    if numeric_boundary <= 1:
        return headers, rows

    idx = 1
    while idx < min(numeric_boundary, len(headers)):
        left_numeric = _column_numericish_ratio(rows, idx - 1)
        right_numeric = _column_numericish_ratio(rows, idx)
        if max(left_numeric, right_numeric) >= 0.25:
            idx += 1
            continue

        left_header = _clean_cell(headers[idx - 1])
        right_header = _clean_cell(headers[idx])
        next_header = _clean_cell(headers[idx + 1]) if idx + 1 < len(headers) else ""
        if (
            len(left_header) >= 6
            and len(right_header) <= 2
            and next_header[:1].islower()
        ):
            idx += 1
            continue

        pair_rows = 0
        continuation_hits = 0
        right_nonempty = 0
        for row in rows:
            left_cell = _clean_cell(row[idx - 1]) if idx - 1 < len(row) else ""
            right_cell = _clean_cell(row[idx]) if idx < len(row) else ""
            if right_cell:
                right_nonempty += 1
            if left_cell and right_cell:
                pair_rows += 1
                if _looks_like_fragment_pair(left_cell, right_cell):
                    continuation_hits += 1

        header_like = _looks_like_fragment_pair(headers[idx - 1], headers[idx])
        sparse_right = right_nonempty <= max(2, len(rows) // 3)
        should_merge = continuation_hits >= 2 or (continuation_hits >= 1 and sparse_right) or (header_like and continuation_hits >= 1)
        if not should_merge:
            idx += 1
            continue

        headers[idx - 1] = _join_fragment_parts(headers[idx - 1], headers[idx])
        del headers[idx]
        for row in rows:
            left = row[idx - 1] if idx - 1 < len(row) else ""
            right = row[idx] if idx < len(row) else ""
            row[idx - 1] = _join_fragment_parts(left, right)
            del row[idx]
        numeric_boundary -= 1
    return headers, rows


def _merge_sparse_pre_numeric_text_columns(headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    if len(headers) <= 2 or not rows:
        return headers, rows

    headers = list(headers)
    rows = [list(row) for row in rows]

    numeric_boundary = len(headers)
    for idx in range(len(headers)):
        if _column_numericish_ratio(rows, idx) >= 0.55:
            numeric_boundary = idx
            break
    if numeric_boundary <= 1:
        return headers, rows

    idx = 1
    while idx < min(numeric_boundary, len(headers)):
        if idx + 1 >= len(headers):
            break
        current_values = [_clean_cell(row[idx]) for row in rows if idx < len(row) and _clean_cell(row[idx])]
        next_numeric_ratio = _column_numericish_ratio(rows, idx + 1)
        if next_numeric_ratio < 0.45 or not current_values:
            idx += 1
            continue

        current_header = _clean_cell(headers[idx])
        next_header = _clean_cell(headers[idx + 1])
        sparse_current = len(current_values) <= max(3, len(rows) // 3)
        fragment_like = (
            len(current_header) <= 2
            or all(len(value) <= 20 and not re.search(r"\d", value) for value in current_values)
        )
        if not (sparse_current and fragment_like):
            idx += 1
            continue

        prev_header = _clean_cell(headers[idx - 1])
        if current_header and len(current_header) <= 2 and next_header[:1].islower():
            headers[idx + 1] = _join_fragment_parts(current_header, next_header)
        if not (current_header and len(current_header) <= 2 and prev_header):
            headers[idx - 1] = _join_fragment_parts(headers[idx - 1], headers[idx])
        del headers[idx]

        for row in rows:
            left = _clean_cell(row[idx - 1], keep_newlines=True) if idx - 1 < len(row) else ""
            current = _clean_cell(row[idx], keep_newlines=True) if idx < len(row) else ""
            if current:
                row[idx - 1] = _join_fragment_parts(left, current)
            elif idx - 1 < len(row):
                row[idx - 1] = left
            del row[idx]
        numeric_boundary -= 1
    return headers, rows


def _drop_empty_columns(headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
    if not headers:
        return headers, rows
    keep_indices: List[int] = []
    for idx, header in enumerate(headers):
        if _clean_cell(header):
            keep_indices.append(idx)
            continue
        if any(idx < len(row) and _clean_cell(row[idx]) for row in rows):
            keep_indices.append(idx)
    if len(keep_indices) == len(headers):
        return headers, rows
    compact_headers = [headers[idx] for idx in keep_indices]
    compact_rows = [[row[idx] if idx < len(row) else "" for idx in keep_indices] for row in rows]
    return compact_headers, compact_rows


def _trim_prose_tail(rows: List[List[str]]) -> List[List[str]]:
    trimmed: List[List[str]] = []
    for row in rows:
        joined = " ".join(_clean_cell(cell) for cell in row if _clean_cell(cell)).strip()
        if not joined:
            continue
        tokens = re.findall(r"[A-Za-z0-9_]+", joined)
        numeric_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
        if len(trimmed) >= 2 and _looks_like_sentenceish_prose(joined) and numeric_tokens < max(2, len(tokens) // 4):
            break
        trimmed.append(row)
    while len(trimmed) >= 2 and _looks_like_fragmented_prose_tail(trimmed[-1], trimmed[:-1]):
        trimmed.pop()
    return trimmed


def _row_numericish_count(row: List[str]) -> int:
    total = 0
    for cell in row:
        compact = _clean_cell(cell)
        if compact and (re.search(r"\d", compact) or compact in {"–", "-", "OOM"}):
            total += 1
    return total


def _row_blank_ratio(row: List[str]) -> float:
    if not row:
        return 1.0
    blanks = sum(1 for cell in row if not _clean_cell(cell))
    return blanks / max(1, len(row))


def _looks_like_chart_scaffold_row(row: List[str]) -> bool:
    joined = " ".join(_clean_cell(cell) for cell in row if _clean_cell(cell)).strip()
    if not joined:
        return False
    lowered = joined.lower()
    if any(marker in lowered for marker in ["training step", "train loss", "eval loss", "metric value", "loss curve", "step"]):
        return True

    nonempty = [_clean_cell(cell) for cell in row if _clean_cell(cell)]
    if len(nonempty) < 2:
        return False
    numericish = _row_numericish_count(row)
    shortish = sum(1 for cell in nonempty if len(cell) <= 18)
    fragmentish = sum(1 for cell in nonempty if len(cell) <= 12 and re.search(r"[A-Za-z]", cell))
    return (
        _row_blank_ratio(row) >= 0.4
        and numericish <= max(2, len(nonempty) // 3)
        and shortish >= max(2, len(nonempty) - 1)
        and fragmentish >= max(1, len(nonempty) // 2)
    )


def _tail_rows_look_like_chart_scaffold(rows: List[List[str]]) -> bool:
    if len(rows) < 2:
        return False
    scaffold_hits = 0
    for row in rows[:4]:
        if _looks_like_chart_scaffold_row(row):
            scaffold_hits += 1
    required_hits = 2 if len(rows[:4]) >= 2 else 1
    return scaffold_hits >= required_hits


def _trim_chart_scaffold_tail(headers: List[str], rows: List[List[str]]) -> List[List[str]]:
    if len(rows) < 4:
        return rows

    trimmed: List[List[str]] = []
    data_rows_seen = 0
    for idx, row in enumerate(rows):
        if _row_numericish_count(row) >= max(2, len(headers) // 4):
            data_rows_seen += 1
        remaining = rows[idx:]
        row_nonempty = [_clean_cell(cell) for cell in row if _clean_cell(cell)]
        leading_sparse_marker = (
            data_rows_seen >= 2
            and len(row_nonempty) == 1
            and len(row_nonempty[0]) <= 24
            and _row_blank_ratio(row) >= 0.5
        )
        if data_rows_seen >= 2 and (_looks_like_chart_scaffold_row(row) or leading_sparse_marker) and _tail_rows_look_like_chart_scaffold(remaining[1:] if leading_sparse_marker else remaining):
            break
        trimmed.append(row)
    return trimmed or rows


def _looks_like_fragmented_prose_tail(row: List[str], prior_rows: List[List[str]]) -> bool:
    nonempty = [_clean_cell(cell) for cell in row if _clean_cell(cell)]
    if len(nonempty) < 4:
        return False
    joined = " ".join(nonempty)
    tokens = re.findall(r"[A-Za-z0-9_]+", joined)
    if not tokens:
        return False
    numeric_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    if numeric_tokens > 0:
        return False
    if _looks_like_sentenceish_prose(joined):
        return True

    lowerish_cells = sum(1 for cell in nonempty if cell[:1].islower())
    short_fragment_cells = sum(1 for cell in nonempty if len(cell) <= 18)
    wordish = sum(1 for token in tokens if token.isalpha())
    prev_numericish = 0
    prev_total = 0
    for prior in prior_rows[-3:]:
        for cell in prior:
            compact = _clean_cell(cell)
            if not compact:
                continue
            prev_total += 1
            if re.search(r"\d", compact) or compact in {"–", "-", "OOM"}:
                prev_numericish += 1

    return (
        wordish >= max(4, len(tokens) // 2)
        and short_fragment_cells >= max(3, len(nonempty) - 1)
        and lowerish_cells >= max(2, len(nonempty) // 2)
        and prev_numericish >= max(2, prev_total // 4)
    )


def _looks_like_headerish_row(row: List[str]) -> bool:
    nonempty = [_clean_cell(cell) for cell in row if _clean_cell(cell)]
    if len(nonempty) < 2:
        return False
    if _looks_like_sentenceish_prose(" ".join(nonempty)):
        return False
    shortish = sum(1 for cell in nonempty if len(cell) <= 24)
    data_like = sum(
        1
        for cell in nonempty
        if re.search(r"(?:\d+\.\d+|[%×]|OOM\b|\bms\b|\bs\b)", cell, flags=re.IGNORECASE)
    )
    return shortish >= max(2, len(nonempty) - 1) and data_like <= max(1, len(nonempty) // 3)


def _headers_look_fragmented(headers: List[str]) -> bool:
    nonempty = [_clean_cell(cell) for cell in headers if _clean_cell(cell)]
    if not nonempty:
        return True
    blank_ratio = sum(1 for cell in headers if not _clean_cell(cell)) / max(1, len(headers))
    short_alpha = sum(
        1
        for cell in nonempty
        if len(cell) <= 10 and re.search(r"[A-Za-z]", cell) and not re.search(r"[%×()]", cell)
    )
    return blank_ratio >= 0.2 or short_alpha >= max(2, len(nonempty) // 2)


def _collapse_header_matrix(header_rows: List[List[str]], *, target_cols: int) -> List[str]:
    if not header_rows or target_cols <= 0:
        return []
    normalized: List[List[str]] = []
    for row in header_rows:
        clean = [_clean_cell(cell) for cell in row[:target_cols]]
        if len(clean) < target_cols:
            clean.extend([""] * (target_cols - len(clean)))
        normalized.append(clean[:target_cols])

    propagated: List[List[str]] = []
    for row in normalized:
        current = ""
        expanded: List[str] = []
        for cell in row:
            if cell:
                current = cell
                expanded.append(cell)
            else:
                expanded.append(current)
        propagated.append(expanded)

    headers: List[str] = []
    for col_idx in range(target_cols):
        parts: List[str] = []
        for row in propagated:
            part = _clean_cell(row[col_idx])
            if not part:
                continue
            if not parts or parts[-1] != part:
                parts.append(part)
        headers.append(" ".join(parts).strip())
    return headers


def _split_cell_for_extra_column(cell: str) -> Tuple[str, str]:
    compact = _clean_cell(cell, keep_newlines=True)
    if not compact:
        return "", ""
    if "\n" in compact:
        lines = [part.strip() for part in compact.split("\n") if part.strip()]
        left_parts: List[str] = []
        right_parts: List[str] = []
        for line in lines:
            left, right = _split_cell_for_extra_column(line)
            left_parts.append(left)
            right_parts.append(right)
        return "\n".join(part for part in left_parts if part), "\n".join(part for part in right_parts if part)

    parts = compact.split()
    if len(parts) <= 1:
        return compact, ""
    return parts[0], " ".join(parts[1:])


def _expand_rows_for_target_cols(rows: List[List[str]], *, target_cols: int) -> Optional[List[List[str]]]:
    if not rows:
        return None
    current_cols = max(len(row) for row in rows)
    if current_cols >= target_cols or target_cols != current_cols + 1:
        return None

    best_idx: Optional[int] = None
    best_score = -1.0
    for idx in range(current_cols):
        splitable = 0
        nonempty = 0
        for row in rows:
            compact = _clean_cell(row[idx], keep_newlines=True) if idx < len(row) else ""
            if not compact:
                continue
            nonempty += 1
            if len(compact.replace("\n", " ").split()) >= 2:
                splitable += 1
        if nonempty == 0:
            continue
        score = splitable / nonempty
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None or best_score < 0.7:
        return None

    expanded: List[List[str]] = []
    for row in rows:
        padded = list(row[:current_cols]) + [""] * max(0, current_cols - len(row))
        left, right = _split_cell_for_extra_column(padded[best_idx])
        new_row = padded[:best_idx] + [left, right] + padded[best_idx + 1 :]
        if len(new_row) < target_cols:
            new_row.extend([""] * (target_cols - len(new_row)))
        expanded.append(new_row[:target_cols])
    return expanded


def _build_auxiliary_header_candidates(page: Any, *, min_area: float, min_cols: int) -> List[Dict[str, Any]]:
    try:
        finder = page.find_tables(strategy="lines")
    except Exception:
        return []
    candidates = _build_candidates_from_table_objects(
        getattr(finder, "tables", None) or [],
        min_area=min_area,
        min_cols=min_cols,
        detection_strategy="pymupdf_lines_aux",
    )
    headerish: List[Dict[str, Any]] = []
    for candidate in candidates:
        matrix = candidate.get("matrix") or []
        if not matrix:
            continue
        if int(candidate.get("raw_row_count") or 0) > 3:
            continue
        if any(_looks_like_headerish_row(row) for row in matrix):
            headerish.append(candidate)
    return headerish


def _repair_headers_from_auxiliary_band(
    headers: List[str],
    rows: List[List[str]],
    *,
    candidate_bbox: Optional[Dict[str, float]],
    auxiliary_header_candidates: List[Dict[str, Any]],
) -> Tuple[List[str], List[List[str]]]:
    if not headers or not rows or not candidate_bbox or not auxiliary_header_candidates:
        return headers, rows
    if not (_headers_look_fragmented(headers) or _looks_like_headerish_row(rows[0])):
        return headers, rows

    best_candidate: Optional[Dict[str, Any]] = None
    best_score = -10**9
    for candidate in auxiliary_header_candidates:
        bbox = candidate.get("bbox")
        matrix = candidate.get("matrix") or []
        if not bbox or not matrix:
            continue
        gap = float(candidate_bbox["y0"]) - float(bbox["y1"])
        if gap > 42.0 or gap < -32.0:
            continue
        if float(bbox["y0"]) > float(candidate_bbox["y0"]) + 16.0:
            continue
        x_overlap = _x_overlap_ratio(candidate_bbox, bbox)
        if x_overlap < 0.8:
            continue
        score = x_overlap * 3.0 - abs(gap) / 16.0 + min(2.0, float(candidate.get("raw_row_count") or 0))
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None:
        return headers, rows

    best_matrix = best_candidate.get("matrix") or []
    target_cols = max(len(best_matrix[0]) if best_matrix else 0, len(headers))
    working_rows = [list(row) for row in rows]
    if working_rows and len(working_rows[0]) < target_cols:
        expanded = _expand_rows_for_target_cols(working_rows, target_cols=target_cols)
        if expanded is not None:
            working_rows = expanded

    header_rows = [list(row) for row in best_matrix]
    if working_rows and len(working_rows[0]) == target_cols and _looks_like_headerish_row(working_rows[0]):
        header_rows.append(working_rows[0])
        working_rows = working_rows[1:]

    repaired_headers = _collapse_header_matrix(header_rows, target_cols=target_cols)
    if len(repaired_headers) != target_cols or not any(_clean_cell(cell) for cell in repaired_headers):
        return headers, rows
    if _headers_look_fragmented(repaired_headers) and not _headers_look_fragmented(headers):
        return headers, rows
    return repaired_headers, working_rows


def _to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    if not headers:
        return ""

    def _escape(cell: str) -> str:
        return (cell or "").replace("|", "\\|").replace("\n", "<br/>")

    header_row = "| " + " | ".join(_escape(cell) for cell in headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = [
        "| " + " | ".join(_escape(cell) for cell in row[:len(headers)]) + " |"
        for row in rows
    ]
    return "\n".join([header_row, divider, *body_rows]).strip()


def _to_csv_text(headers: List[str], rows: List[List[str]]) -> str:
    if not headers:
        return ""
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    for row in rows:
        normalized = row[:len(headers)] + [""] * max(0, len(headers) - len(row))
        writer.writerow(normalized[:len(headers)])
    return buffer.getvalue().strip()


def _render_table_markdown(table_obj: Any, headers: List[str], rows: List[List[str]]) -> str:
    if table_obj is not None:
        try:
            native = (table_obj.to_markdown() or "").strip()
        except Exception:
            native = ""
        if native:
            return native
    return _to_markdown(headers, rows)


def _build_native_page_candidates(
    page: Any,
    *,
    min_area: float,
    min_cols: int,
) -> List[Dict[str, Any]]:
    strategies = [
        ("pymupdf_lines_strict", "lines_strict"),
        ("pymupdf_native", "lines"),
    ]
    strict_candidates: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []
    for label, strategy in strategies:
        try:
            finder = page.find_tables(strategy=strategy)
        except Exception:
            continue
        candidates = _build_candidates_from_table_objects(
            getattr(finder, "tables", None) or [],
            min_area=min_area,
            min_cols=min_cols,
            detection_strategy=label,
        )
        if strategy == "lines_strict":
            strict_candidates = candidates
        else:
            fallback_candidates = candidates
    if not strict_candidates:
        return fallback_candidates
    if not fallback_candidates:
        return strict_candidates

    strict_bboxes = [candidate.get("bbox") for candidate in strict_candidates]
    combined = list(strict_candidates)
    for candidate in fallback_candidates:
        if _bbox_matches_any_iou(candidate.get("bbox"), strict_bboxes, 0.55):
            continue
        combined.append(candidate)
    return _merge_table_candidates(combined)


def _table_caption_key(caption: str) -> str:
    compact = " ".join(str(caption or "").split()).strip()
    if not compact:
        return ""
    match = re.match(r"^(table\s+\d+[A-Za-z]?)\b", compact, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return re.sub(r"[^a-z0-9]+", " ", compact.lower()).strip()[:120]


def _looks_like_explicit_table_caption(text: str) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return False
    if re.match(r"^table\s+\d+[A-Za-z]?\s*[:.]\s+\S", compact, flags=re.IGNORECASE):
        return True
    match = re.match(r"^(table\s+\d+[A-Za-z]?)\s+([A-Za-z]+)\b", compact, flags=re.IGNORECASE)
    if not match:
        return False
    next_word = str(match.group(2) or "").lower()
    if next_word in {"shows", "show", "presents", "present", "summarizes", "summarize", "reports", "report", "lists", "list", "compares", "compare"}:
        return False
    return len(compact.split()) <= 18


def _table_record_quality(record: Dict[str, Any]) -> float:
    headers = [str(cell or "") for cell in (record.get("headers") or [])]
    rows = record.get("rows") or []
    strategy = str(record.get("detection_strategy") or "")

    nonempty_headers = sum(1 for cell in headers if _clean_cell(cell))
    blank_headers = sum(1 for cell in headers if not _clean_cell(cell))
    generated_headers = sum(1 for cell in headers if re.fullmatch(r"col_\d+", _clean_cell(cell)))
    bad_headers = sum(
        1
        for cell in headers
        if _clean_cell(cell)
        and (
            re.fullmatch(r"\d+", _clean_cell(cell))
            or (len(_clean_cell(cell)) <= 3 and re.search(r"[A-Za-z]", _clean_cell(cell)))
        )
    )
    row_nonempty = [sum(1 for cell in row if _clean_cell(cell)) for row in rows]
    avg_nonempty = (sum(row_nonempty) / len(row_nonempty)) if row_nonempty else 0.0
    cell_texts = [_clean_cell(cell) for row in rows for cell in row if _clean_cell(cell)]
    numericish = sum(1 for cell in cell_texts if re.search(r"\d", cell) or cell in {"–", "-", "OOM"})
    numeric_ratio = (numericish / len(cell_texts)) if cell_texts else 0.0

    score = 0.0
    score += nonempty_headers * 1.8
    score += min(len(rows), 16) * 0.45
    score += avg_nonempty * 0.6
    score += numeric_ratio * 2.0
    score -= blank_headers * 1.3
    score -= generated_headers * 1.5
    score -= bad_headers * 1.7
    if strategy.startswith("pymupdf_text_reconstructed"):
        score += 0.35
    return score


def _append_or_replace_table_record(
    table_records: List[Dict[str, Any]],
    record: Dict[str, Any],
) -> None:
    caption_key = _table_caption_key(str(record.get("caption") or ""))
    if not caption_key:
        table_records.append(record)
        return

    for idx, existing in enumerate(table_records):
        if int(existing.get("page_no") or 0) != int(record.get("page_no") or 0):
            continue
        existing_key = _table_caption_key(str(existing.get("caption") or ""))
        if existing_key != caption_key:
            continue
        if _table_record_quality(record) > _table_record_quality(existing):
            table_records[idx] = record
        return
    table_records.append(record)


def _candidate_needs_text_reconstruction(
    *,
    detection_strategy: str,
    headers: List[str],
    rows: List[List[str]],
    caption_block: Optional[Dict[str, Any]],
) -> bool:
    if caption_block is None:
        return False
    if detection_strategy == "text_caption_fallback":
        return False
    if len(rows) < 2:
        return True
    blank_header_ratio = sum(1 for cell in headers if not _clean_cell(cell)) / max(1, len(headers))
    if blank_header_ratio >= 0.3:
        return True
    if any("\n" in str(cell or "") for cell in headers):
        return True
    return False


def _score_text_reconstruction_candidate(
    candidate: Dict[str, Any],
    *,
    native_bbox: Dict[str, float],
    expected_cols: int,
) -> float:
    bbox = candidate.get("bbox")
    if not isinstance(bbox, dict):
        return -10**9
    native_area = max(_rect_area(native_bbox), 1e-6)
    overlap = _rect_overlap(native_bbox, bbox) / native_area
    x_overlap = _x_overlap_ratio(native_bbox, bbox)
    raw_rows = int(candidate.get("raw_row_count") or 0)
    cols = int(candidate.get("n_cols") or 0)
    score = overlap * 2.5 + x_overlap * 2.0 + min(1.0, raw_rows / 12.0)
    score -= abs(cols - expected_cols) * 0.2
    return score


def _reconstruct_candidate_from_text(
    page: Any,
    candidate: Dict[str, Any],
    *,
    caption_blocks: List[Dict[str, Any]],
    caption_index: Optional[int],
    caption_block: Optional[Dict[str, Any]],
    page_bbox: Dict[str, float],
    min_area: float,
    min_cols: int,
) -> Optional[Dict[str, Any]]:
    native_bbox = candidate.get("bbox")
    if not isinstance(native_bbox, dict) or caption_block is None:
        return None

    bbox = native_bbox
    max_below = max(60.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_MAX_BELOW_PT", "260"), 260.0))
    x_margin = max(2.0, _safe_float(os.getenv("TABLE_TEXT_FALLBACK_X_MARGIN_PT", "8"), 8.0))
    next_caption_bbox = None
    if caption_index is not None and caption_index + 1 < len(caption_blocks):
        next_caption_bbox = caption_blocks[caption_index + 1].get("bbox")

    page_text_blocks = _extract_page_text_blocks(page)
    clip_bottom = float(bbox["y1"]) + max_below
    for block in page_text_blocks:
        block_bbox = block.get("bbox")
        if not block_bbox:
            continue
        if _x_overlap_ratio(block_bbox, bbox) < 0.35:
            continue
        if float(block_bbox["y0"]) <= float(bbox["y1"]) + 4.0:
            continue
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        if _looks_like_section_boundary_text(text):
            clip_bottom = min(clip_bottom, float(block_bbox["y0"]) - 4.0)
            break
        if _looks_like_sentenceish_prose(text) and len(text) >= 80:
            clip_bottom = min(clip_bottom, float(block_bbox["y0"]) - 4.0)
            break

    clip_bbox = {
        "x0": max(page_bbox["x0"] + 2.0, float(bbox["x0"]) - x_margin),
        "y0": max(page_bbox["y0"] + 2.0, float(bbox["y0"]) - 4.0),
        "x1": min(page_bbox["x1"] - 2.0, float(bbox["x1"]) + x_margin),
        "y1": min(
            float(page_bbox["y1"]) - 2.0,
            float(next_caption_bbox["y0"]) - 4.0 if next_caption_bbox else clip_bottom,
        ),
    }
    if clip_bbox["y1"] - clip_bbox["y0"] < 40.0:
        return None

    clip_rect = _bbox_to_rect_tuple(clip_bbox)
    if not clip_rect:
        return None

    try:
        finder = page.find_tables(strategy="text", clip=clip_rect)
    except Exception:
        return None

    reconstructed = _build_candidates_from_table_objects(
        getattr(finder, "tables", None) or [],
        min_area=min_area,
        min_cols=min_cols,
        detection_strategy="pymupdf_text_reconstructed",
        seed_caption=str(caption_block.get("text") or "").strip(),
        seed_caption_id=caption_index,
        seed_caption_bbox=caption_block.get("bbox"),
        clip_bbox=clip_bbox,
    )
    if not reconstructed:
        return None

    expected_cols = max(1, int(candidate.get("n_cols") or 0))
    reconstructed.sort(
        key=lambda item: _score_text_reconstruction_candidate(item, native_bbox=native_bbox, expected_cols=expected_cols),
        reverse=True,
    )
    best = reconstructed[0]
    if int(best.get("raw_row_count") or 0) <= int(candidate.get("raw_row_count") or 0):
        return None
    return best


def _candidate_is_native_enough_for_caption(
    candidate: Dict[str, Any],
    *,
    caption_bbox: Optional[Dict[str, float]],
    page_bbox: Dict[str, float],
    min_rows: int,
) -> bool:
    bbox = candidate.get("bbox")
    if not _passes_text_fallback_constraints(bbox, page_bbox, caption_bbox):
        return False
    matrix = candidate.get("matrix") or []
    if not matrix:
        return False
    raw_row_count = len(matrix)
    if raw_row_count < min_rows:
        return False
    headers, rows = _pick_headers_and_rows(matrix, candidate.get("table_obj"))
    return len(rows) >= min_rows or raw_row_count >= min_rows + 1


def _caption_indices_needing_auto_text_fallback(
    page_candidates: List[Dict[str, Any]],
    caption_blocks: List[Dict[str, Any]],
    *,
    page_bbox: Dict[str, float],
    min_rows: int,
) -> set[int]:
    needed: set[int] = set()
    for idx, caption_block in enumerate(caption_blocks):
        caption_text = str(caption_block.get("text") or "")
        if not _looks_like_explicit_table_caption(caption_text):
            continue
        caption_bbox = caption_block.get("bbox")
        if not caption_bbox:
            continue
        has_native_support = any(
            _candidate_is_native_enough_for_caption(
                candidate,
                caption_bbox=caption_bbox,
                page_bbox=page_bbox,
                min_rows=min_rows,
            )
            for candidate in page_candidates
        )
        if not has_native_support:
            needed.add(idx)
    return needed


def _passes_auto_text_fallback_quality(candidate: Dict[str, Any]) -> bool:
    matrix = candidate.get("matrix") or []
    if not matrix:
        return False
    headers, rows = _pick_headers_and_rows(matrix, candidate.get("table_obj"))
    if len(headers) < 2 or len(rows) < 2:
        return False

    cells = _flatten_matrix_cells(matrix)
    if not cells:
        return False

    joined = " ".join(cells)
    tokens = re.findall(r"[A-Za-z0-9_]+", joined)
    numeric_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    numeric_ratio = (numeric_tokens / len(tokens)) if tokens else 0.0

    signatures = [_row_signature(row) for row in matrix]
    signatures = [sig for sig in signatures if sig]
    duplicate_row_ratio = 1.0 - (len(set(signatures)) / len(signatures)) if signatures else 0.0

    avg_cell_chars = sum(len(cell) for cell in cells) / max(1, len(cells))
    if numeric_ratio >= 0.08:
        return True
    if duplicate_row_ratio >= 0.35:
        return False
    if avg_cell_chars >= 16.0 and _looks_like_sentenceish_prose(joined):
        return False
    return False


def _missing_explicit_table_caption_indices(
    table_caption_blocks: List[Dict[str, Any]],
    table_records: List[Dict[str, Any]],
    *,
    page_no: int,
) -> set[int]:
    resolved_keys = {
        _table_caption_key(str(record.get("caption") or ""))
        for record in table_records
        if int(record.get("page_no") or 0) == int(page_no)
    }
    missing: set[int] = set()
    for idx, caption_block in enumerate(table_caption_blocks):
        caption_text = str(caption_block.get("text") or "").strip()
        if not _looks_like_explicit_table_caption(caption_text):
            continue
        caption_key = _table_caption_key(caption_text)
        if caption_key and caption_key not in resolved_keys:
            missing.add(idx)
    return missing


def _materialize_table_record(
    *,
    page: Any,
    page_no: int,
    paper_id: int,
    candidate: Dict[str, Any],
    page_kept_bboxes: List[Optional[Dict[str, float]]],
    table_caption_blocks: List[Dict[str, Any]],
    figure_caption_blocks: List[Dict[str, Any]],
    auxiliary_header_candidates: List[Dict[str, Any]],
    page_bounds: Dict[str, float],
    page_blocks_for_page: List[Dict[str, Any]],
    min_area: float,
    min_cols: int,
    min_rows: int,
    dedup_iou_threshold: float,
) -> Optional[Dict[str, Any]]:
    bbox = candidate.get("bbox")
    matrix = candidate.get("matrix") or []
    table_obj = candidate.get("table_obj")
    detection_strategy = str(candidate.get("detection_strategy") or "pymupdf_native")
    if not matrix:
        return None

    if _bbox_matches_any_iou(bbox, page_kept_bboxes, dedup_iou_threshold):
        return None

    headers, rows = _pick_headers_and_rows(matrix, table_obj)
    n_cols = len(headers) if headers else len(matrix[0])
    matrix_row_count = len(matrix)
    row_count = len(rows)
    if n_cols < min_cols:
        return None

    seed_caption_text = str(candidate.get("seed_caption") or "").strip()
    seed_caption_bbox = candidate.get("seed_caption_bbox")
    if detection_strategy == "text_caption_fallback":
        if not _passes_text_fallback_constraints(bbox, page_bounds, seed_caption_bbox):
            logger.info(
                "Skipping text fallback candidate for paper %s page %s (%sx%s): failed local caption constraints",
                paper_id,
                page_no,
                row_count,
                n_cols,
            )
            return None

    seed_caption_index = candidate.get("seed_caption_id")
    caption_index, caption_block = _resolve_candidate_caption_binding(
        caption_blocks=table_caption_blocks,
        candidate_bbox=bbox,
        detection_strategy=detection_strategy,
        seed_caption_index=seed_caption_index,
    )
    caption = str(caption_block.get("text") or "").strip() if caption_block else _find_table_caption(page, bbox)
    if detection_strategy == "text_caption_fallback" and not caption and seed_caption_text:
        caption = seed_caption_text[:260]
    if detection_strategy == "text_caption_fallback" and not (caption and _TABLE_CAPTION_RE.search(caption)):
        logger.info(
            "Skipping text fallback candidate for paper %s page %s (%sx%s): no nearby table caption",
            paper_id,
            page_no,
            row_count,
            n_cols,
        )
        return None

    if _candidate_needs_text_reconstruction(
        detection_strategy=detection_strategy,
        headers=headers,
        rows=rows,
        caption_block=caption_block,
    ):
        reconstructed = _reconstruct_candidate_from_text(
            page,
            candidate,
            caption_blocks=table_caption_blocks,
            caption_index=caption_index,
            caption_block=caption_block,
            page_bbox=page_bounds,
            min_area=min_area,
            min_cols=min_cols,
        )
        if reconstructed is not None:
            candidate = reconstructed
            bbox = candidate.get("bbox")
            matrix = candidate.get("matrix") or []
            table_obj = candidate.get("table_obj")
            detection_strategy = str(candidate.get("detection_strategy") or detection_strategy)
            headers, rows = _pick_headers_and_rows(matrix, table_obj)
            n_cols = len(headers) if headers else len(matrix[0])
            matrix_row_count = len(matrix)
            row_count = len(rows)
            caption_index, caption_block = _resolve_candidate_caption_binding(
                caption_blocks=table_caption_blocks,
                candidate_bbox=bbox,
                detection_strategy=detection_strategy,
                seed_caption_index=candidate.get("seed_caption_id"),
            )
            caption = str(caption_block.get("text") or "").strip() if caption_block else _find_table_caption(page, bbox)
            if detection_strategy == "text_caption_fallback" and not caption and seed_caption_text:
                caption = seed_caption_text[:260]

    headers, rows = _repair_headers_from_auxiliary_band(
        headers,
        rows,
        candidate_bbox=bbox,
        auxiliary_header_candidates=auxiliary_header_candidates,
    )
    row_count = len(rows)
    n_cols = len(headers)
    if n_cols < min_cols:
        return None
    if matrix_row_count < min_rows and row_count < min_rows:
        return None
    if row_count <= 0:
        return None

    figure_caption_index, figure_caption_block = _find_nearest_caption_block(figure_caption_blocks, bbox)
    _ = figure_caption_index
    figure_caption = str(figure_caption_block.get("text") or "").strip() if figure_caption_block else None
    table_caption_gap = _vertical_gap(bbox, caption_block.get("bbox") if caption_block else None)
    figure_caption_gap = _vertical_gap(bbox, figure_caption_block.get("bbox") if figure_caption_block else None)

    is_false_positive, fp_reasons = _looks_like_false_positive_table(
        matrix,
        n_cols=n_cols,
        row_count=row_count,
        table_caption=caption,
        figure_caption=figure_caption,
        table_caption_gap=table_caption_gap,
        figure_caption_gap=figure_caption_gap,
    )
    if is_false_positive:
        logger.info(
            "Skipping table candidate for paper %s page %s (%sx%s, fragments=%s, strategy=%s): %s",
            paper_id,
            page_no,
            row_count,
            n_cols,
            int(candidate.get("merged_fragments") or 1),
            detection_strategy,
            ", ".join(fp_reasons),
        )
        return None

    section = _infer_section_for_table(page_blocks_for_page, bbox)
    markdown = _render_table_markdown(
        None if detection_strategy.startswith("pymupdf_text_") else table_obj,
        headers,
        rows,
    )
    csv_text = _to_csv_text(headers, rows)

    page_kept_bboxes.append(bbox)
    return {
        "paper_id": int(paper_id),
        "page_no": page_no,
        "bbox": bbox,
        "caption": caption,
        "n_rows": len(rows),
        "n_cols": n_cols,
        "headers": headers,
        "rows": rows,
        "markdown": markdown,
        "csv_text": csv_text,
        "section_canonical": section["section_canonical"],
        "section_title": section["section_title"],
        "section_source": section["section_source"],
        "section_confidence": section["section_confidence"],
        "figure_caption_nearby": figure_caption,
        "merged_fragments": int(candidate.get("merged_fragments") or 1),
        "detection_strategy": detection_strategy,
    }


def _extract_block_text(block: Dict[str, Any]) -> str:
    lines = block.get("lines") or []
    parts: List[str] = []
    for line in lines:
        spans = line.get("spans") or []
        seg = "".join(str(span.get("text") or "") for span in spans)
        seg = " ".join(seg.split()).strip()
        if seg:
            parts.append(seg)
    return " ".join(parts).strip()


def _find_caption_with_pattern(
    page: Any,
    target_bbox: Optional[Dict[str, float]],
    pattern: re.Pattern[str],
    *,
    prefer_above: bool = True,
    max_distance_pt: float = 120.0,
    fallback_nearby: bool = False,
) -> Optional[str]:
    if not target_bbox:
        return None
    try:
        page_dict = page.get_text("dict")
    except Exception:
        return None

    target_top = float(target_bbox["y0"])
    target_bottom = float(target_bbox["y1"])
    candidates: List[Tuple[float, str]] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        bbox = _bbox_from_tuple(block.get("bbox"))
        if not bbox:
            continue
        text = _extract_block_text(block)
        if not text:
            continue

        block_top = float(bbox["y0"])
        block_bottom = float(bbox["y1"])
        side: Optional[str] = None
        distance = 0.0
        if block_bottom <= target_top + 2.0:
            distance = target_top - block_bottom
            side = "above"
        elif block_top >= target_bottom - 2.0:
            distance = block_top - target_bottom
            side = "below"
        if side is None:
            continue
        if distance > max_distance_pt:
            continue

        if pattern.search(text):
            penalty = 0.0
            if prefer_above and side == "below":
                penalty = 20.0
            elif (not prefer_above) and side == "above":
                penalty = 20.0
            candidates.append((distance + penalty, text[:260]))
            continue

        if fallback_nearby and side == "above" and distance <= 24.0:
            candidates.append((distance + 60.0, text[:180]))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _find_table_caption(page: Any, table_bbox: Optional[Dict[str, float]]) -> Optional[str]:
    max_distance = _safe_float(os.getenv("TABLE_CAPTION_MAX_DISTANCE_PT", "120"), 120.0)
    return _find_caption_with_pattern(
        page,
        table_bbox,
        _TABLE_CAPTION_RE,
        prefer_above=True,
        max_distance_pt=max_distance,
        fallback_nearby=True,
    )


def _find_figure_caption(page: Any, table_bbox: Optional[Dict[str, float]]) -> Optional[str]:
    max_distance = _safe_float(os.getenv("TABLE_CAPTION_MAX_DISTANCE_PT", "120"), 120.0)
    return _find_caption_with_pattern(
        page,
        table_bbox,
        _FIGURE_CAPTION_RE,
        prefer_above=False,
        max_distance_pt=max_distance,
        fallback_nearby=False,
    )


def _flatten_matrix_cells(matrix: List[List[str]]) -> List[str]:
    cells: List[str] = []
    for row in matrix:
        for cell in row:
            text = _clean_cell(cell)
            if text:
                cells.append(text)
    return cells


def _row_signature(row: List[str]) -> str:
    return "|".join(_clean_cell(cell).lower() for cell in row).strip()


def _normalized_caption_body(text: str) -> str:
    compact = " ".join(str(text or "").split()).strip().lower()
    compact = re.sub(r"^(table|tab\.?)\s*\d+[a-z]?\s*[:.]?\s*", "", compact)
    compact = re.sub(r"[^a-z0-9]+", " ", compact)
    return re.sub(r"\s+", " ", compact).strip()


def _looks_like_false_positive_table(
    matrix: List[List[str]],
    *,
    n_cols: int,
    row_count: int,
    table_caption: Optional[str],
    figure_caption: Optional[str],
    table_caption_gap: Optional[float] = None,
    figure_caption_gap: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    has_table_caption = bool(table_caption and _TABLE_CAPTION_RE.search(table_caption))
    has_figure_caption = bool(figure_caption and _FIGURE_CAPTION_RE.search(figure_caption))
    if has_figure_caption and not has_table_caption:
        reasons.append("nearby_figure_caption")
        return True, reasons

    cells = _flatten_matrix_cells(matrix)
    if not cells:
        return True, ["empty_cells"]

    joined = " ".join(cells)
    tokens = re.findall(r"[A-Za-z0-9_]+", joined)
    numeric_tokens = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    numeric_ratio = (numeric_tokens / len(tokens)) if tokens else 0.0

    avg_cell_chars = sum(len(cell) for cell in cells) / max(1, len(cells))
    long_cell_ratio = sum(1 for cell in cells if len(cell) >= 40) / max(1, len(cells))
    duplicate_row_ratio = 0.0
    if matrix:
        signatures = [_row_signature(row) for row in matrix]
        signatures = [sig for sig in signatures if sig]
        if signatures:
            unique_count = len(set(signatures))
            duplicate_row_ratio = 1.0 - (unique_count / len(signatures))

    if _BASE64_LIKE_RE.search(joined) and not has_table_caption:
        reasons.append("base64_like_noise")
    if "figure " in joined.lower():
        reasons.append("figure_like_content")
    normalized_caption = _normalized_caption_body(table_caption or "")
    normalized_joined = _normalized_caption_body(joined)
    if normalized_caption and normalized_joined:
        caption_tokens = [token for token in normalized_caption.split() if len(token) >= 4]
        caption_token_hits = sum(1 for token in caption_tokens if token in normalized_joined)
        if caption_token_hits >= max(2, len(caption_tokens) // 2):
            reasons.append("caption_text_leaked_into_rows")
    if n_cols <= 1 and row_count >= 3:
        reasons.append("collapsed_single_column")
    if row_count <= 2 and n_cols >= 8 and not has_table_caption:
        reasons.append("very_wide_shallow_without_table_caption")
    if row_count <= 1 and avg_cell_chars > 18.0 and not has_table_caption:
        reasons.append("shallow_without_table_caption")
    if avg_cell_chars > 30.0 and numeric_ratio < 0.05 and not has_table_caption:
        reasons.append("prose_like_cells_without_table_caption")
    if long_cell_ratio > 0.45 and numeric_ratio < 0.08 and not has_table_caption:
        reasons.append("too_many_long_cells_low_numeric")
    if duplicate_row_ratio > 0.45 and n_cols >= 6 and not has_table_caption:
        reasons.append("duplicate_rows_pattern")
    if (
        has_figure_caption
        and figure_caption_gap is not None
        and table_caption_gap is not None
        and figure_caption_gap + 48.0 < table_caption_gap
        and (n_cols <= 2 or row_count <= 2 or long_cell_ratio > 0.25)
    ):
        reasons.append("figure_caption_closer_than_table_caption")
        if n_cols <= 2 or row_count <= 2:
            return True, reasons

    min_false_positive_signals = max(1, _safe_int(os.getenv("TABLE_FALSE_POSITIVE_MIN_SIGNALS", "2"), 2))
    if len(reasons) >= min_false_positive_signals:
        return True, reasons
    return False, []


def extract_and_store_paper_tables(
    pdf_path: Path,
    paper_id: int,
    blocks: Iterable[Dict[str, Any]],
    page_allowlist: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Extract structured tables and store a per-paper manifest.
    """
    if not _table_enabled():
        return {"paper_id": int(paper_id), "num_tables": 0, "manifest_path": None, "tables": []}

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

    for stale in output_dir.glob("table_*.json"):
        try:
            stale.unlink()
        except Exception:
            logger.warning("Failed to remove stale table file %s", stale)

    table_records: List[Dict[str, Any]] = []
    min_rows = max(1, _safe_int(os.getenv("TABLE_MIN_ROWS", "2"), 2))
    min_cols = max(1, _safe_int(os.getenv("TABLE_MIN_COLS", "2"), 2))
    min_area = max(1.0, _safe_float(os.getenv("TABLE_MIN_AREA_PT", "1600"), 1600.0))
    text_fallback_enabled = _table_text_fallback_enabled()
    auto_text_fallback_enabled = _table_auto_text_fallback_enabled()
    dedup_iou_threshold = min(1.0, max(0.0, _safe_float(os.getenv("TABLE_DEDUP_IOU_THRESHOLD", "0.80"), 0.80)))

    doc = pymupdf.open(str(pdf_path))
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_no = page_index + 1
            if allowed_pages is not None and page_no not in allowed_pages:
                continue
            if not hasattr(page, "find_tables"):
                logger.warning("PyMuPDF build does not support page.find_tables(); skipping table extraction.")
                break

            page_bounds = _page_bbox(page)
            page_kept_bboxes: List[Optional[Dict[str, float]]] = []
            page_candidates: List[Dict[str, Any]] = []

            table_caption_blocks = _collect_caption_blocks(page, _TABLE_CAPTION_RE)
            figure_caption_blocks = _collect_caption_blocks(page, _FIGURE_CAPTION_RE)
            auxiliary_header_candidates = _build_auxiliary_header_candidates(
                page,
                min_area=min_area,
                min_cols=min_cols,
            )
            try:
                page_candidates.extend(
                    _build_native_page_candidates(
                        page,
                        min_area=min_area,
                        min_cols=min_cols,
                    )
                )
            except Exception as exc:
                logger.warning("Table detection failed for paper %s page %s: %s", paper_id, page_no, exc)
                continue

            fallback_caption_indices: Optional[set[int]] = None
            if table_caption_blocks:
                if text_fallback_enabled:
                    fallback_caption_indices = set(range(len(table_caption_blocks)))
                elif auto_text_fallback_enabled:
                    fallback_caption_indices = _caption_indices_needing_auto_text_fallback(
                        page_candidates,
                        table_caption_blocks,
                        page_bbox=page_bounds,
                        min_rows=min_rows,
                    )

            if fallback_caption_indices:
                fallback_candidates = _build_caption_guided_text_candidates(
                    page,
                    table_caption_blocks,
                    min_area=min_area,
                    min_cols=min_cols,
                    caption_indices=fallback_caption_indices,
                )
                if not text_fallback_enabled:
                    fallback_candidates = [
                        candidate
                        for candidate in fallback_candidates
                        if _passes_auto_text_fallback_quality(candidate)
                    ]
                page_candidates.extend(fallback_candidates)

            for candidate in _merge_table_candidates(page_candidates):
                record = _materialize_table_record(
                    page=page,
                    page_no=page_no,
                    paper_id=paper_id,
                    candidate=candidate,
                    page_kept_bboxes=page_kept_bboxes,
                    table_caption_blocks=table_caption_blocks,
                    figure_caption_blocks=figure_caption_blocks,
                    auxiliary_header_candidates=auxiliary_header_candidates,
                    page_bounds=page_bounds,
                    page_blocks_for_page=page_blocks.get(page_no, []),
                    min_area=min_area,
                    min_cols=min_cols,
                    min_rows=min_rows,
                    dedup_iou_threshold=dedup_iou_threshold,
                )
                if record is not None:
                    _append_or_replace_table_record(table_records, record)

            missing_caption_indices = _missing_explicit_table_caption_indices(
                table_caption_blocks,
                table_records,
                page_no=page_no,
            )
            if missing_caption_indices:
                recovery_candidates = _build_caption_guided_text_candidates(
                    page,
                    table_caption_blocks,
                    min_area=min_area,
                    min_cols=min_cols,
                    caption_indices=missing_caption_indices,
                )
                if not text_fallback_enabled:
                    recovery_candidates = [
                        candidate
                        for candidate in recovery_candidates
                        if _passes_auto_text_fallback_quality(candidate)
                    ]
                for candidate in _merge_table_candidates(recovery_candidates):
                    record = _materialize_table_record(
                        page=page,
                        page_no=page_no,
                        paper_id=paper_id,
                        candidate=candidate,
                        page_kept_bboxes=page_kept_bboxes,
                        table_caption_blocks=table_caption_blocks,
                        figure_caption_blocks=figure_caption_blocks,
                        auxiliary_header_candidates=auxiliary_header_candidates,
                        page_bounds=page_bounds,
                        page_blocks_for_page=page_blocks.get(page_no, []),
                        min_area=min_area,
                        min_cols=min_cols,
                        min_rows=min_rows,
                        dedup_iou_threshold=dedup_iou_threshold,
                    )
                    if record is not None:
                        _append_or_replace_table_record(table_records, record)
    finally:
        doc.close()

    for idx, record in enumerate(table_records, start=1):
        record["id"] = idx
        record["json_file"] = f"table_{idx:04d}.json"
        table_path = output_dir / record["json_file"]
        with table_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)

    manifest = {
        "paper_id": int(paper_id),
        "source_pdf": str(pdf_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_tables": len(table_records),
        "tables": table_records,
    }
    manifest_path = _manifest_path(paper_id)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return {
        "paper_id": int(paper_id),
        "num_tables": len(table_records),
        "manifest_path": str(manifest_path),
        "tables": table_records,
    }


def _next_indices_by_page(text_blocks: Iterable[Dict[str, Any]]) -> Dict[int, int]:
    max_by_page: Dict[int, int] = {}
    for block in text_blocks:
        page_no = _safe_int(block.get("page_no"), 0)
        if page_no <= 0:
            continue
        idx = _safe_int(block.get("block_index"), 0)
        max_by_page[page_no] = max(max_by_page.get(page_no, -1), idx)
    next_by_page: Dict[int, int] = {}
    for page_no, max_idx in max_by_page.items():
        next_by_page[page_no] = max_idx + 1000
    return next_by_page


def table_records_to_chunks(
    tables: Iterable[Dict[str, Any]],
    text_blocks: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert structured tables into chunk records for pgvector insertion.
    """
    records = [item for item in tables if isinstance(item, dict)]
    if not records:
        return []

    max_rows = max(1, _safe_int(os.getenv("TABLE_MAX_ROWS_PER_CHUNK", "12"), 12))
    max_chars = max(500, _safe_int(os.getenv("TABLE_CHUNK_MAX_CHARS", "2200"), 2200))
    next_by_page = _next_indices_by_page(text_blocks)
    chunks: List[Dict[str, Any]] = []

    for table in records:
        page_no = _safe_int(table.get("page_no"), 1)
        headers_raw = table.get("headers")
        headers = [str(cell) for cell in headers_raw] if isinstance(headers_raw, list) else []
        rows_raw = table.get("rows")
        rows = rows_raw if isinstance(rows_raw, list) else []
        rows = [row for row in rows if isinstance(row, list)]
        if not headers:
            if rows:
                headers = [f"col_{idx + 1}" for idx in range(len(rows[0]))]
            else:
                continue

        table_id = _safe_int(table.get("id"), 0)
        caption = str(table.get("caption") or f"Table {table_id}").strip()
        section_canonical = str(table.get("section_canonical") or "other").strip() or "other"
        section_title = str(table.get("section_title") or "Document Body").strip() or "Document Body"
        section_source = str(table.get("section_source") or "fallback").strip() or "fallback"
        section_confidence = _safe_float(table.get("section_confidence"), 0.35)
        table_bbox = _bbox_from_payload(table.get("bbox"))

        if page_no not in next_by_page:
            next_by_page[page_no] = 1000

        if not rows:
            rows = [[]]

        row_ptr = 0
        while row_ptr < len(rows):
            if rows == [[]]:
                window = []
                consumed = 1
            else:
                window = rows[row_ptr:row_ptr + max_rows]
                consumed = max(1, len(window))

            row_start = row_ptr + 1 if rows != [[]] else 0
            row_end = row_ptr + consumed if rows != [[]] else 0
            markdown = _to_markdown(headers, window)
            text_parts = [
                caption,
                f"Section: {section_title}",
                markdown,
            ]
            chunk_text = "\n\n".join(part for part in text_parts if part).strip()

            while len(chunk_text) > max_chars and consumed > 1:
                consumed -= 1
                window = window[:consumed]
                row_end = row_ptr + consumed
                markdown = _to_markdown(headers, window)
                chunk_text = "\n\n".join(part for part in [caption, f"Section: {section_title}", markdown] if part).strip()

            block_index = next_by_page[page_no]
            next_by_page[page_no] = block_index + 1

            source_block = {
                "page_no": page_no,
                "block_index": block_index,
                "text": chunk_text,
                "bbox": table_bbox,
                "metadata": {
                    "section_title": section_title,
                    "section_canonical": section_canonical,
                    "section_source": section_source,
                    "section_confidence": round(section_confidence, 3),
                },
            }

            metadata = {
                "chunk_type": "table_rows",
                "content_type": "table",
                "table_id": table_id,
                "table_caption": caption,
                "table_page_no": page_no,
                "table_row_start": row_start,
                "table_row_end": row_end,
                "table_total_rows": _safe_int(table.get("n_rows"), len(rows)),
                "table_total_cols": _safe_int(table.get("n_cols"), len(headers)),
                "table_columns": headers,
                "table_chunk_index": max(0, row_start - 1) // max_rows if row_start > 0 else 0,
                "section_primary": section_canonical,
                "section_all": [section_canonical],
                "section_titles": [section_title],
                "section_source": section_source,
                "section_confidence": round(section_confidence, 3),
                "spans_multiple_sections": False,
                "blocks": [source_block],
            }

            chunks.append(
                {
                    "text": chunk_text,
                    "page_no": page_no,
                    "block_index": block_index,
                    "bbox": table_bbox,
                    "metadata": metadata,
                }
            )

            row_ptr += consumed
            if rows == [[]]:
                break

    return chunks


def load_paper_table_manifest(paper_id: int) -> Dict[str, Any]:
    path = _manifest_path(paper_id)
    if not path.exists():
        return {"paper_id": int(paper_id), "num_tables": 0, "tables": []}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Failed to read table manifest for paper %s: %s", paper_id, exc)
        return {"paper_id": int(paper_id), "num_tables": 0, "tables": []}
    if not isinstance(payload, dict):
        return {"paper_id": int(paper_id), "num_tables": 0, "tables": []}
    tables = payload.get("tables")
    if not isinstance(tables, list):
        tables = []
    payload["tables"] = tables
    payload["num_tables"] = _safe_int(payload.get("num_tables"), len(tables))
    payload["paper_id"] = _safe_int(payload.get("paper_id"), int(paper_id))
    return payload


def resolve_table_file(paper_id: int, file_name: str) -> Path:
    candidate = str(file_name or "").strip()
    if not candidate:
        raise ValueError("Table file name is required.")
    if "/" in candidate or "\\" in candidate or ".." in candidate:
        raise ValueError("Invalid table file name.")
    return _paper_dir(paper_id) / candidate
