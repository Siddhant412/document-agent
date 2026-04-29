from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Sequence


NonOcrHandler = Literal["ia_phase1", "native"]
DocumentHandler = Literal["ia_phase1", "native", "mixed"]


ACADEMIC_HEADING_PATTERNS = (
    r"abstract",
    r"introduction",
    r"related\s+work",
    r"background",
    r"method(?:s|ology)?",
    r"approach",
    r"experiments?",
    r"evaluation",
    r"results?",
    r"discussion",
    r"conclusion(?:s)?",
    r"references",
    r"bibliography",
    r"appendix",
    r"preliminar(?:y|ies)",
)

REPORT_HEADING_PATTERNS = (
    r"section\s+\d+",
    r"part\s+\d+",
    r"findings?",
    r"recommendations?",
    r"background",
    r"incident",
    r"case\s+review",
    r"policy",
    r"procedures?",
    r"summary",
    r"memorandum",
    r"manual",
    r"proposal",
    r"report",
)

AFFILIATION_PATTERNS = (
    r"university",
    r"institute",
    r"college",
    r"school",
    r"department",
    r"laboratory",
    r"\barxiv\b",
    r"@",
)


_ACADEMIC_HEADING_RE = re.compile(r"^(?:" + "|".join(ACADEMIC_HEADING_PATTERNS) + r")$", re.IGNORECASE)
_REPORT_HEADING_RE = re.compile(r"^(?:" + "|".join(REPORT_HEADING_PATTERNS) + r")$", re.IGNORECASE)
_SECTION_OR_PART_RE = re.compile(r"^(?:section|part)\s+\d+\b", re.IGNORECASE)
_SCHOLARLY_NUMBERED_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+[A-Za-z]")
_ROMAN_NUMBERED_HEADING_RE = re.compile(r"^\s*[IVXLCDM]+\.?\s+[A-Z][A-Za-z]", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^(?:figure|fig\.?)\s+[a-z0-9ivxlcdm]+(?:[\.:]|\s)", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^(?:table|tab\.?)\s+[a-z0-9ivxlcdm]+(?:[\.:]|\s)", re.IGNORECASE)
_CITATION_RE = re.compile(
    r"\[[0-9]{1,3}(?:\s*,\s*[0-9]{1,3})*\]|\b[A-Z][A-Za-z\-]+ et al\.\b|\([A-Z][A-Za-z\-]+ et al\.,?\s*20\d{2}\)",
    re.IGNORECASE,
)
_MATH_LINE_RE = re.compile(r"(?:=|≤|≥|∑|Σ|λ|μ|θ|β|→|<-|->)")
_PAGE_TOKEN_RE = re.compile(r"\bpage\s+\d+\b", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d+")


@dataclass(slots=True)
class PageContentProfile:
    page_num: int
    ia_score: int
    native_score: int
    signals: Dict[str, Any] = field(default_factory=dict)
    handler: NonOcrHandler = "ia_phase1"


@dataclass(slots=True)
class NonOcrRun:
    handler: NonOcrHandler
    page_numbers: List[int] = field(default_factory=list)


@dataclass(slots=True)
class DocumentContentProfile:
    document_handler: DocumentHandler
    page_profiles: List[PageContentProfile] = field(default_factory=list)
    runs: List[NonOcrRun] = field(default_factory=list)
    scores: Dict[str, Any] = field(default_factory=dict)


def _clean_heading_text(value: str) -> str:
    compact = " ".join(str(value or "").split()).strip().lower()
    compact = re.sub(r"^[0-9ivxlcdm]+(?:\.[0-9]+)*[\.\)]?\s+", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"[^a-z0-9\s]+", " ", compact)
    return " ".join(compact.split())


def _normalize_repeat_line(value: str) -> str:
    compact = " ".join(str(value or "").split()).strip().lower()
    compact = _PAGE_TOKEN_RE.sub("page", compact)
    compact = _DIGIT_RE.sub("#", compact)
    compact = re.sub(r"[^a-z0-9\s]+", " ", compact)
    return " ".join(compact.split())


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _count_matches(lines: Sequence[str], pattern: re.Pattern[str]) -> int:
    return sum(1 for line in lines if pattern.match(_clean_heading_text(line)))


def _repeated_line_pages(page_infos: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for page in page_infos:
        seen: set[str] = set()
        for raw_line in page.get(key, []) or []:
            normalized = _normalize_repeat_line(str(raw_line or ""))
            if len(normalized) < 6 or normalized in seen:
                continue
            seen.add(normalized)
            counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _build_page_profile(page_info: Mapping[str, Any], *, repeated_headers: Dict[str, int], repeated_footers: Dict[str, int]) -> PageContentProfile:
    page_num = _safe_int(page_info.get("page_num"))
    heading_lines = [str(line or "") for line in page_info.get("heading_lines", []) or []]
    top_lines = [str(line or "") for line in page_info.get("top_lines", []) or []]
    bottom_lines = [str(line or "") for line in page_info.get("bottom_lines", []) or []]

    citation_count = _safe_int(page_info.get("citation_count"))
    figure_caption_count = _safe_int(page_info.get("figure_caption_count"))
    table_caption_count = _safe_int(page_info.get("table_caption_count"))
    equation_line_count = _safe_int(page_info.get("equation_line_count"))
    author_affiliation_hits = _safe_int(page_info.get("author_affiliation_hits"))
    image_count = _safe_int(page_info.get("image_count"))
    table_count = _safe_int(page_info.get("table_count"))
    drawing_count = _safe_int(page_info.get("drawing_count"))
    text_len = _safe_int(page_info.get("text_len"))
    two_columns = bool(page_info.get("two_columns"))
    paragraph_like_count = _safe_int(page_info.get("paragraph_like_count"))

    academic_heading_hits = _count_matches(heading_lines, _ACADEMIC_HEADING_RE)
    report_heading_hits = _count_matches(heading_lines, _REPORT_HEADING_RE)
    section_heading_hits = sum(1 for line in heading_lines if _SECTION_OR_PART_RE.match(_clean_heading_text(line)))
    part_heading_hits = sum(1 for line in heading_lines if _clean_heading_text(line).startswith("part "))
    scholarly_numbered_heading_hits = sum(1 for line in heading_lines if _SCHOLARLY_NUMBERED_HEADING_RE.match(str(line or "")))
    roman_numbered_heading_hits = sum(1 for line in heading_lines if _ROMAN_NUMBERED_HEADING_RE.match(str(line or "")))
    has_abstract = any(_clean_heading_text(line) == "abstract" for line in heading_lines)
    has_references = any(_clean_heading_text(line) in {"references", "bibliography"} for line in heading_lines)
    has_visual_caption = figure_caption_count + table_caption_count > 0
    repeated_header_hits = sum(1 for line in top_lines if repeated_headers.get(_normalize_repeat_line(line), 0) >= 2)
    repeated_footer_hits = sum(1 for line in bottom_lines if repeated_footers.get(_normalize_repeat_line(line), 0) >= 2)

    scholarly_score = 0
    structured_visual_score = 0
    report_score = 0

    if has_abstract:
        scholarly_score += 3
    if has_references:
        scholarly_score += 3
    if citation_count >= 2:
        scholarly_score += 2
    if academic_heading_hits >= 1:
        scholarly_score += min(academic_heading_hits, 3)
    if scholarly_numbered_heading_hits >= 1:
        scholarly_score += min(scholarly_numbered_heading_hits, 2)
    if roman_numbered_heading_hits >= 1:
        scholarly_score += 1
    if page_num == 1 and author_affiliation_hits > 0:
        scholarly_score += 2
    if equation_line_count >= 2:
        scholarly_score += 1
    if two_columns:
        scholarly_score += 1

    if has_visual_caption:
        structured_visual_score += 2
    if image_count + table_count > 0:
        structured_visual_score += 1
    if section_heading_hits >= 1 and has_visual_caption:
        structured_visual_score += 1
    if drawing_count > 8 and has_visual_caption:
        structured_visual_score += 1

    if section_heading_hits + part_heading_hits >= 1:
        report_score += 2
    if report_heading_hits >= 1:
        report_score += 2
    if repeated_header_hits + repeated_footer_hits >= 1 and academic_heading_hits == 0 and scholarly_numbered_heading_hits == 0 and equation_line_count == 0:
        report_score += 1
    if paragraph_like_count >= 4 and citation_count == 0 and not has_visual_caption and academic_heading_hits == 0 and scholarly_numbered_heading_hits == 0 and equation_line_count == 0:
        report_score += 2
    if not has_abstract and not has_references and academic_heading_hits == 0 and scholarly_numbered_heading_hits == 0 and author_affiliation_hits == 0:
        report_score += 1
    if text_len >= 400 and image_count == 0 and table_count == 0 and not two_columns and equation_line_count == 0 and scholarly_numbered_heading_hits == 0:
        report_score += 1

    ia_score = scholarly_score + structured_visual_score
    native_score = report_score
    handler: NonOcrHandler = "ia_phase1" if ia_score >= native_score else "native"

    return PageContentProfile(
        page_num=page_num,
        ia_score=ia_score,
        native_score=native_score,
        handler=handler,
        signals={
            "academic_heading_hits": academic_heading_hits,
            "report_heading_hits": report_heading_hits,
            "section_heading_hits": section_heading_hits,
            "part_heading_hits": part_heading_hits,
            "scholarly_numbered_heading_hits": scholarly_numbered_heading_hits,
            "roman_numbered_heading_hits": roman_numbered_heading_hits,
            "citation_count": citation_count,
            "figure_caption_count": figure_caption_count,
            "table_caption_count": table_caption_count,
            "equation_line_count": equation_line_count,
            "author_affiliation_hits": author_affiliation_hits,
            "repeated_header_hits": repeated_header_hits,
            "repeated_footer_hits": repeated_footer_hits,
            "paragraph_like_count": paragraph_like_count,
            "has_abstract": has_abstract,
            "has_references": has_references,
            "two_columns": two_columns,
            "scholarly_score": scholarly_score,
            "structured_visual_score": structured_visual_score,
            "report_score": report_score,
        },
    )


def _build_runs(page_profiles: Sequence[PageContentProfile]) -> List[NonOcrRun]:
    runs: List[NonOcrRun] = []
    for profile in page_profiles:
        if runs and runs[-1].handler == profile.handler:
            runs[-1].page_numbers.append(profile.page_num)
            continue
        runs.append(NonOcrRun(handler=profile.handler, page_numbers=[profile.page_num]))
    return runs


def _smooth_isolated_page_runs(page_profiles: Sequence[PageContentProfile], *, total_pages: int) -> List[PageContentProfile]:
    smoothed = [
        PageContentProfile(
            page_num=profile.page_num,
            ia_score=profile.ia_score,
            native_score=profile.native_score,
            signals=dict(profile.signals),
            handler=profile.handler,
        )
        for profile in page_profiles
    ]
    if len(smoothed) < 2:
        return smoothed

    profiles_by_page = {profile.page_num: profile for profile in smoothed}
    runs = _build_runs(smoothed)
    replacements: Dict[int, NonOcrHandler] = {}
    for run_index, run in enumerate(runs):
        if len(run.page_numbers) != 1:
            continue
        page_num = run.page_numbers[0]
        left_run = runs[run_index - 1] if run_index > 0 else None
        right_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
        left_is_ocr = page_num > 1 and (page_num - 1) not in profiles_by_page
        right_is_ocr = page_num < total_pages and (page_num + 1) not in profiles_by_page

        replacement: NonOcrHandler | None = None
        # Only smooth truly isolated middle runs. Edge single-page runs are left
        # alone because they are often legitimate boundary transitions.
        if left_run and right_run and left_run.handler == right_run.handler:
            replacement = left_run.handler
        elif left_is_ocr and right_is_ocr:
            replacement = None
        elif left_run and right_run and not left_is_ocr and not right_is_ocr:
            replacement = left_run.handler if len(left_run.page_numbers) >= len(right_run.page_numbers) else right_run.handler

        if replacement is not None:
            replacements[page_num] = replacement

    for page_num, handler in replacements.items():
        profiles_by_page[page_num].handler = handler

    return [profiles_by_page[profile.page_num] for profile in smoothed]


def build_document_content_profile(page_infos: Sequence[Mapping[str, Any]], *, total_pages: int) -> DocumentContentProfile:
    ordered_pages = sorted(
        (
            page
            for page in page_infos
            if _safe_int(page.get("page_num")) > 0
        ),
        key=lambda page: _safe_int(page.get("page_num")),
    )
    if not ordered_pages:
        return DocumentContentProfile(document_handler="native")

    repeated_headers = _repeated_line_pages(ordered_pages, "top_lines")
    repeated_footers = _repeated_line_pages(ordered_pages, "bottom_lines")
    page_profiles = [
        _build_page_profile(page, repeated_headers=repeated_headers, repeated_footers=repeated_footers)
        for page in ordered_pages
    ]

    scholarly_score = sum(_safe_int(profile.signals.get("scholarly_score")) for profile in page_profiles)
    structured_visual_score = sum(_safe_int(profile.signals.get("structured_visual_score")) for profile in page_profiles)
    report_score = sum(_safe_int(profile.signals.get("report_score")) for profile in page_profiles)
    total_citations = sum(_safe_int(profile.signals.get("citation_count")) for profile in page_profiles)
    total_visual_captions = sum(
        _safe_int(profile.signals.get("figure_caption_count")) + _safe_int(profile.signals.get("table_caption_count"))
        for profile in page_profiles
    )
    total_numbered_scholarly_headings = sum(_safe_int(profile.signals.get("scholarly_numbered_heading_hits")) for profile in page_profiles)
    total_roman_headings = sum(_safe_int(profile.signals.get("roman_numbered_heading_hits")) for profile in page_profiles)
    total_academic_headings = sum(_safe_int(profile.signals.get("academic_heading_hits")) for profile in page_profiles)
    total_equation_lines = sum(_safe_int(profile.signals.get("equation_line_count")) for profile in page_profiles)
    affiliation_pages = sum(1 for profile in page_profiles if _safe_int(profile.signals.get("author_affiliation_hits")) > 0)
    total_section_headings = sum(_safe_int(profile.signals.get("section_heading_hits")) for profile in page_profiles)
    total_report_headings = sum(_safe_int(profile.signals.get("report_heading_hits")) for profile in page_profiles)
    prose_dominant_pages = sum(
        1
        for profile in page_profiles
        if _safe_int(profile.signals.get("paragraph_like_count")) >= 4
        and _safe_int(profile.signals.get("citation_count")) == 0
        and _safe_int(profile.signals.get("figure_caption_count")) == 0
        and _safe_int(profile.signals.get("table_caption_count")) == 0
    )
    repeated_footer_pages = sum(1 for profile in page_profiles if _safe_int(profile.signals.get("repeated_footer_hits")) > 0)
    repeated_header_pages = sum(1 for profile in page_profiles if _safe_int(profile.signals.get("repeated_header_hits")) > 0)

    if any(profile.signals.get("has_abstract") for profile in page_profiles):
        scholarly_score += 3
    if any(profile.signals.get("has_references") for profile in page_profiles):
        scholarly_score += 3
    if total_citations >= 4:
        scholarly_score += 2
    if total_academic_headings >= 3:
        scholarly_score += 2
    if total_numbered_scholarly_headings >= 3:
        scholarly_score += 4
    elif total_numbered_scholarly_headings >= 1:
        scholarly_score += 2
    if total_roman_headings >= 2:
        scholarly_score += 2
    if affiliation_pages >= 1:
        scholarly_score += 2
    if affiliation_pages >= 1 and (total_academic_headings >= 1 or total_numbered_scholarly_headings >= 1):
        scholarly_score += 2
    if total_equation_lines >= 8:
        scholarly_score += 2
    if total_visual_captions >= 2:
        structured_visual_score += 2
    if total_visual_captions >= 4:
        structured_visual_score += 2
    if total_section_headings >= 2 and total_visual_captions >= 2:
        structured_visual_score += 2

    if total_section_headings >= 3:
        report_score += 3
    if total_report_headings >= 3:
        report_score += 2
    if repeated_footer_pages + repeated_header_pages >= 2 and total_academic_headings == 0 and total_numbered_scholarly_headings == 0 and total_equation_lines == 0:
        report_score += 2
    if total_citations == 0 and not any(profile.signals.get("has_abstract") for profile in page_profiles) and not any(profile.signals.get("has_references") for profile in page_profiles) and affiliation_pages == 0 and total_numbered_scholarly_headings == 0 and total_academic_headings == 0:
        report_score += 2
    if prose_dominant_pages >= max(1, len(page_profiles) // 2) and total_visual_captions == 0 and total_equation_lines == 0 and total_numbered_scholarly_headings == 0 and total_academic_headings == 0:
        report_score += 2
    if total_visual_captions == 0 and total_equation_lines == 0 and total_numbered_scholarly_headings == 0 and total_academic_headings == 0 and affiliation_pages == 0:
        report_score += 1

    ia_score = scholarly_score + structured_visual_score
    native_score = report_score

    if ia_score >= native_score + 3:
        for profile in page_profiles:
            profile.handler = "ia_phase1"
        document_handler: DocumentHandler = "ia_phase1"
    elif native_score >= ia_score + 3:
        for profile in page_profiles:
            profile.handler = "native"
        document_handler = "native"
    else:
        page_profiles = _smooth_isolated_page_runs(page_profiles, total_pages=total_pages)
        document_handler = "mixed"

    runs = _build_runs(page_profiles)
    return DocumentContentProfile(
        document_handler=document_handler,
        page_profiles=page_profiles,
        runs=runs,
        scores={
            "ia_score": ia_score,
            "native_score": native_score,
            "scholarly_score": scholarly_score,
            "structured_visual_score": structured_visual_score,
            "report_score": report_score,
            "repeated_header_pages": repeated_header_pages,
            "repeated_footer_pages": repeated_footer_pages,
            "prose_dominant_pages": prose_dominant_pages,
            "total_citations": total_citations,
            "total_visual_captions": total_visual_captions,
            "total_academic_headings": total_academic_headings,
            "total_numbered_scholarly_headings": total_numbered_scholarly_headings,
            "total_roman_headings": total_roman_headings,
            "total_equation_lines": total_equation_lines,
            "affiliation_pages": affiliation_pages,
        },
    )
