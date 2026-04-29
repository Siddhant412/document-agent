"""
Section extraction utilities for PDF ingestion.

Fallback chain:
1) arXiv source parsing (if source URL resolves to an arXiv ID)
2) GROBID TEI parsing (if GROBID_URL is configured)
3) Heuristic heading extraction from PDF text blocks

The output is used to annotate each extracted PDF block with section metadata
before chunking and embedding.
"""
from __future__ import annotations

import io
import logging
import os
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import pymupdf
from pypdf import PdfReader

logger = logging.getLogger(__name__)


SECTION_PATTERNS: List[Tuple[str, List[str]]] = [
    ("abstract", [r"\babstract\b"]),
    ("introduction", [r"\bintroduction\b", r"\boverview\b"]),
    ("problem_definition", [r"\bproblem definition\b", r"\bproblem statement\b"]),
    ("notation", [r"\bnotation(s)?\b"]),
    ("preliminaries", [r"\bpreliminar(?:y|ies)\b", r"\bbackground\b"]),
    ("related_work", [r"\brelated work(s)?\b", r"\bliterature review\b"]),
    ("methodology", [r"\bmethod(s|ology)?\b", r"\bapproach\b", r"\bproblem formulation\b"]),
    ("construction", [r"\bconstruction\b"]),
    ("datasets", [r"\bdataset(s)?\b"]),
    ("implementation_details", [r"\bimplementation detail(s)?\b", r"\btraining detail(s)?\b"]),
    ("achievability", [r"\bachievab(?:ility|le)\b"]),
    ("capacity_bound", [r"\bcapacity\b.*\bbound\b", r"\bupper bound\b", r"\blower bound\b"]),
    ("experiments", [r"\bexperiment(s)?\b", r"\bevaluation\b"]),
    ("results", [r"\bresult(s)?\b", r"\bfindings\b", r"\bperformance\b"]),
    ("discussion", [r"\bdiscussion\b", r"\bimplications\b"]),
    ("numerical_applications", [r"\bnumerical application(s)?\b", r"\bnumerical result(s)?\b", r"\bsimulation(s)?\b"]),
    ("conclusion", [r"\bconclusion(s)?\b", r"\bfuture work\b"]),
    ("limitations", [r"\blimitation(s)?\b"]),
    ("ethical_consideration", [r"\bethical consideration(s)?\b", r"\bethics\b"]),
    ("prompt_templates", [r"\bprompt template(s)?\b"]),
    ("appendix", [r"\bappendix\b", r"\bsupplementary\b"]),
    ("acknowledgements", [r"\backnowledg(e)?ments?\b"]),
    ("references", [r"\breferences?\b", r"\bbibliography\b"]),
]


_ARXIV_ID_RE = re.compile(
    r"(?:(?:https?://)?arxiv\.org/(?:abs|pdf|e-print)/)?"
    r"(?P<id>(?:\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(?:v\d+)?)",
    re.IGNORECASE,
)
_LATEX_SECTION_RE = re.compile(
    r"\\(?P<kind>section|subsection|subsubsection)\*?\{(?P<title>[^{}]{1,220})\}",
    re.IGNORECASE | re.MULTILINE,
)
_LATEX_ABSTRACT_RE = re.compile(
    r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}",
    re.IGNORECASE | re.DOTALL,
)
_HEADING_NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?:[ivxlcdm]+\.?|[a-z]\)|\d+(?:\.\d+){0,3}\.?)\s+",
    re.IGNORECASE,
)
_ROMAN_HEADING_RE = re.compile(r"^\s*(?P<num>[IVXLCDM]+)\.\s+(?P<rest>.+)$")
_NUMERIC_HEADING_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+){0,3})\.?\s+(?P<rest>.+)$")
_HEADING_NOISE_RE = re.compile(r"^(table|fig\.?|figure|algorithm|lemma|theorem)\b", re.IGNORECASE)
_STANDALONE_HEADING_MARKER_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+){0,3}|[IVXLCDM]+)\.?\s*$",
    re.IGNORECASE,
)
_HEADING_CONNECTOR_WORDS = {"and", "&", "to", "for", "of", "in", "on", "with", "without", "via"}
_CORE_SECTION_CANONICALS = {
    "abstract",
    "introduction",
    "problem_definition",
    "notation",
    "related_work",
    "methodology",
    "achievability",
    "capacity_bound",
    "experiments",
    "results",
    "discussion",
    "numerical_applications",
    "conclusion",
    "appendix",
    "acknowledgements",
    "references",
}


@dataclass
class HeadingCandidate:
    title: str
    level: int
    source: str
    confidence: float
    block_hint: Optional[int] = None
    page_hint: Optional[int] = None


@dataclass
class SectionSpan:
    index: int
    title: str
    canonical: str
    level: int
    source: str
    confidence: float
    start_idx: int
    end_idx: int
    start_page: int
    end_page: int


KNOWN_CANONICALS = {canonical for canonical, _ in SECTION_PATTERNS} | {"front_matter", "other"}


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value).strip().lower()
    value = _HEADING_NUMBER_PREFIX_RE.sub("", value)
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _clean_heading_title(value: str) -> str:
    if not value:
        return ""
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\s+", " ", value).strip()
    value = _HEADING_NUMBER_PREFIX_RE.sub("", value)
    return value.strip(" .:-")


def canonicalize_heading(raw_title: str) -> str:
    normalized = _normalize_text(raw_title)
    if not normalized:
        return "other"
    for canonical, patterns in SECTION_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                return canonical
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not slug:
        return "other"
    return "_".join(slug.split("_")[:10])


def _extract_arxiv_id(source_url: Optional[str]) -> Optional[str]:
    if not source_url:
        return None
    value = source_url.strip()
    if not value:
        return None
    match = _ARXIV_ID_RE.search(value)
    if not match:
        return None
    arxiv_id = match.group("id")
    if arxiv_id.lower().endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]
    return arxiv_id


def _extract_arxiv_id_from_pdf_metadata(pdf_path: Path) -> Optional[str]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return None

    metadata = reader.metadata or {}
    candidates = [
        metadata.get("/arXivID"),
        metadata.get("arXivID"),
        metadata.get("/DOI"),
        metadata.get("doi"),
        metadata.get("/Subject"),
        metadata.get("subject"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        match = _ARXIV_ID_RE.search(str(candidate))
        if not match:
            continue
        arxiv_id = match.group("id")
        if arxiv_id.lower().endswith(".pdf"):
            arxiv_id = arxiv_id[:-4]
        return arxiv_id
    return None


def _strip_latex_comments(content: str) -> str:
    cleaned_lines: List[str] = []
    for line in content.splitlines():
        # keep escaped percent signs, strip comments
        stripped = re.sub(r"(?<!\\)%.*$", "", line)
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def _pick_main_tex(tex_candidates: List[Tuple[str, str]]) -> Optional[str]:
    best_content: Optional[str] = None
    best_score = -1
    for _, content in tex_candidates:
        score = 0
        if "\\begin{document}" in content:
            score += 10
        score += len(_LATEX_SECTION_RE.findall(content))
        if _LATEX_ABSTRACT_RE.search(content):
            score += 3
        if score > best_score:
            best_score = score
            best_content = content
    return best_content


def _parse_latex_headings(tex_content: str) -> List[HeadingCandidate]:
    cleaned = _strip_latex_comments(tex_content)
    headings: List[HeadingCandidate] = []
    include_subsections = os.getenv("ARXIV_INCLUDE_SUBSECTIONS", "0").strip().lower() in {"1", "true", "yes"}

    abstract_match = _LATEX_ABSTRACT_RE.search(cleaned)
    if abstract_match:
        headings.append(
            HeadingCandidate(
                title="Abstract",
                level=1,
                source="arxiv_source",
                confidence=0.95,
            )
        )

    level_map = {"section": 1, "subsection": 2, "subsubsection": 3}
    for match in _LATEX_SECTION_RE.finditer(cleaned):
        raw_title = _clean_heading_title(match.group("title") or "")
        if not raw_title:
            continue
        kind = (match.group("kind") or "section").lower()
        if kind != "section" and not include_subsections:
            continue
        headings.append(
            HeadingCandidate(
                title=raw_title,
                level=level_map.get(kind, 1),
                source="arxiv_source",
                confidence=0.92 if kind == "section" else 0.85,
            )
        )

    if re.search(r"\\begin\{thebibliography\}|\\bibliography\s*\{", cleaned, re.IGNORECASE):
        headings.append(
            HeadingCandidate(
                title="References",
                level=1,
                source="arxiv_source",
                confidence=0.9,
            )
        )

    return _dedupe_headings(headings)


def _extract_headings_from_arxiv_source(
    source_url: Optional[str],
    pdf_path: Optional[Path] = None,
) -> List[HeadingCandidate]:
    arxiv_id = _extract_arxiv_id(source_url)
    if not arxiv_id and pdf_path is not None:
        arxiv_id = _extract_arxiv_id_from_pdf_metadata(pdf_path)
    if not arxiv_id:
        return []

    source_endpoint = f"https://arxiv.org/e-print/{arxiv_id}"
    timeout = int(os.getenv("ARXIV_SOURCE_TIMEOUT", "20"))
    try:
        response = requests.get(source_endpoint, timeout=timeout)
        if response.status_code != 200 or not response.content:
            return []
    except Exception as exc:
        logger.warning("arXiv source download failed for %s: %s", arxiv_id, exc)
        return []

    tex_candidates: List[Tuple[str, str]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                if not member.name.lower().endswith(".tex"):
                    continue
                file_obj = archive.extractfile(member)
                if not file_obj:
                    continue
                try:
                    content = file_obj.read().decode("utf-8", errors="ignore")
                except Exception:
                    continue
                tex_candidates.append((member.name, content))
    except Exception as exc:
        logger.warning("Failed to parse arXiv source tarball for %s: %s", arxiv_id, exc)
        return []

    if not tex_candidates:
        return []
    main_tex = _pick_main_tex(tex_candidates)
    if not main_tex:
        return []
    return _parse_latex_headings(main_tex)


def _extract_headings_from_pdf_toc(pdf_path: Path) -> List[HeadingCandidate]:
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        logger.warning("Failed to open PDF for TOC extraction %s: %s", pdf_path, exc)
        return []

    try:
        toc = doc.get_toc()
    except Exception as exc:
        logger.warning("Failed to read PDF TOC for %s: %s", pdf_path, exc)
        doc.close()
        return []

    doc.close()

    if not toc:
        return []

    has_level_one = any(int(item[0]) == 1 for item in toc if len(item) >= 3)
    allowed_levels = {1} if has_level_one else {1, 2}
    headings: List[HeadingCandidate] = []
    seen = set()

    for item in toc:
        if len(item) < 3:
            continue
        try:
            level = int(item[0])
            title = _clean_heading_title(str(item[1] or ""))
            page_no = int(item[2])
        except Exception:
            continue

        if level not in allowed_levels:
            continue
        if page_no < 1:
            continue
        if not _is_reasonable_heading_title(title):
            continue

        normalized = _normalize_text(title)
        key = (normalized, page_no)
        if not normalized or key in seen:
            continue
        seen.add(key)

        headings.append(
            HeadingCandidate(
                title=title,
                level=min(max(level, 1), 3),
                source="pdf_toc",
                confidence=0.97 if level == 1 else 0.9,
                page_hint=page_no,
            )
        )

    return _dedupe_headings(headings)


def _xml_text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split()).strip()


def _extract_headings_with_grobid(pdf_path: Path) -> List[HeadingCandidate]:
    grobid_url = os.getenv("GROBID_URL", "").strip()
    if not grobid_url:
        return []

    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    timeout = int(os.getenv("GROBID_TIMEOUT", "45"))
    try:
        with pdf_path.open("rb") as handle:
            response = requests.post(
                endpoint,
                files={"input": (pdf_path.name, handle, "application/pdf")},
                data={
                    "consolidateHeader": "0",
                    "consolidateCitations": "0",
                    "includeRawCitations": "0",
                },
                timeout=timeout,
            )
        response.raise_for_status()
        xml_text = response.text
        if not xml_text.strip():
            return []
    except Exception as exc:
        logger.warning("GROBID extraction failed for %s: %s", pdf_path, exc)
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("GROBID returned invalid XML for %s: %s", pdf_path, exc)
        return []

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    headings: List[HeadingCandidate] = []
    include_subsections = os.getenv("GROBID_INCLUDE_SUBSECTIONS", "0").strip().lower() in {"1", "true", "yes"}

    abstract_node = root.find(".//tei:abstract", ns)
    if abstract_node is not None:
        headings.append(
            HeadingCandidate(
                title="Abstract",
                level=1,
                source="grobid",
                confidence=0.88,
            )
        )

    for div in root.findall(".//tei:div", ns):
        head = div.find("./tei:head", ns)
        title = _clean_heading_title(_xml_text(head))
        if not title:
            continue
        n_attr = (head.get("n") if head is not None else None) or ""
        level = 1
        if "." in n_attr:
            level = min(3, n_attr.count(".") + 1)
        if level > 1 and not include_subsections:
            continue
        headings.append(
            HeadingCandidate(
                title=title,
                level=level,
                source="grobid",
                confidence=0.84 if level == 1 else 0.78,
            )
        )

    return _dedupe_headings(headings)


def _is_upper_token(token: str) -> bool:
    cleaned = re.sub(r"[^A-Za-z0-9&()/-]+", "", token)
    if not cleaned:
        return False
    letters = [ch for ch in cleaned if ch.isalpha()]
    if not letters:
        return False
    uppercase = sum(1 for ch in letters if ch.isupper())
    return uppercase / len(letters) >= 0.8


def _extract_roman_heading_title(text: str) -> str:
    match = _ROMAN_HEADING_RE.match(text.strip())
    if not match:
        return ""
    rest = match.group("rest").strip()
    if not rest:
        return ""
    tokens = rest.split()
    title_tokens: List[str] = []
    for token in tokens[:24]:
        if _is_upper_token(token):
            title_tokens.append(token)
            continue
        break
    if not title_tokens:
        return ""
    return _clean_heading_title(" ".join(title_tokens))


def _extract_numeric_heading_title(text: str) -> str:
    match = _NUMERIC_HEADING_RE.match(text.strip())
    if not match:
        return ""
    rest = match.group("rest").strip()
    if not rest:
        return ""
    tokens = rest.split()
    title_tokens: List[str] = []
    for token in tokens[:20]:
        cleaned = re.sub(r"[^A-Za-z0-9&()/-]+", "", token)
        if not cleaned:
            continue
        if cleaned.lower() in _HEADING_CONNECTOR_WORDS and title_tokens:
            title_tokens.append(token)
            continue
        if cleaned[0].islower() and len(title_tokens) >= 2:
            break
        title_tokens.append(token)
    if not title_tokens:
        return ""
    return _clean_heading_title(" ".join(title_tokens))


def _is_standalone_heading_marker(text: str) -> bool:
    return bool(_STANDALONE_HEADING_MARKER_RE.match(text or ""))


def _looks_like_heading_phrase(text: str) -> bool:
    line = " ".join((text or "").split())
    if not line:
        return False
    if line.endswith("."):
        return False
    if line.count(",") > 1:
        return False
    tokens = line.split()
    if len(tokens) == 0 or len(tokens) > 12:
        return False
    if _HEADING_NOISE_RE.match(line.lower()):
        return False
    title_case_tokens = sum(1 for token in tokens if token[:1].isupper() or _is_upper_token(token))
    return title_case_tokens >= max(2, len(tokens) - 2)


def _looks_like_sentenceish_prose(text: str) -> bool:
    lowered = str(text or "").lower()
    if re.search(r"[.!?]\s", str(text or "")):
        return True
    stopwords = re.findall(
        r"\b(the|and|that|with|from|this|these|our|their|which|while|using|without|into|through|because|however|although)\b",
        lowered,
    )
    return len(stopwords) >= 3


def _is_reasonable_heading_title(title: str) -> bool:
    cleaned = _clean_heading_title(title)
    normalized = _normalize_text(cleaned)
    if not normalized:
        return False
    lowered = cleaned.lower()
    if any(marker in lowered for marker in ["http://", "https://", "www.", "github", "openreview", "arxiv.org", "doi.org"]):
        return False
    if _HEADING_NOISE_RE.match(normalized):
        return False
    tokens = normalized.split()
    if not tokens:
        return False
    if tokens[0] in {"our", "this", "these", "those", "see"} and canonicalize_heading(cleaned) not in _CORE_SECTION_CANONICALS:
        return False
    if len(tokens) == 1 and len(tokens[0]) < 4:
        return False
    if sum(len(token) for token in tokens) < 4:
        return False
    short_tokens = sum(1 for token in tokens if len(token) <= 1)
    if len(tokens) > 1 and short_tokens >= len(tokens) // 2:
        return False
    alnum = re.findall(r"[A-Za-z0-9]", cleaned)
    digits = sum(1 for ch in alnum if ch.isdigit())
    if alnum and digits / len(alnum) >= 0.28 and canonicalize_heading(cleaned) not in _CORE_SECTION_CANONICALS:
        return False
    return True


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _block_bbox(block: Dict[str, Any]) -> Tuple[float, float, float, float]:
    bbox = block.get("bbox")
    if isinstance(bbox, dict):
        return (
            _safe_float(bbox.get("x0"), 0.0),
            _safe_float(bbox.get("y0"), 0.0),
            _safe_float(bbox.get("x1"), 0.0),
            _safe_float(bbox.get("y1"), 0.0),
        )
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (
            _safe_float(bbox[0], 0.0),
            _safe_float(bbox[1], 0.0),
            _safe_float(bbox[2], 0.0),
            _safe_float(bbox[3], 0.0),
        )
    return (0.0, 0.0, 0.0, 0.0)


def _page_geometries(blocks: List[Dict[str, Any]]) -> Dict[int, Tuple[float, float, float, float]]:
    page_box: Dict[int, List[float]] = {}
    for block in blocks:
        page_no = int(block.get("page_no") or 1)
        x0, y0, x1, y1 = _block_bbox(block)
        if page_no not in page_box:
            page_box[page_no] = [x0, y0, x1, y1]
            continue
        agg = page_box[page_no]
        agg[0] = min(agg[0], x0)
        agg[1] = min(agg[1], y0)
        agg[2] = max(agg[2], x1)
        agg[3] = max(agg[3], y1)
    return {page: (vals[0], vals[1], vals[2], vals[3]) for page, vals in page_box.items()}


def _block_body_like(block: Dict[str, Any], body_font: float) -> bool:
    metadata = block.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}
    char_count = int(_safe_float(meta.get("char_count"), 0))
    line_count = int(_safe_float(meta.get("line_count"), 0))
    avg_font = _safe_float(meta.get("avg_font_size"), 0.0)
    if char_count < 75 or line_count < 2:
        return False
    if body_font <= 0:
        return True
    return avg_font <= (body_font + 0.8)


def _has_nearby_body_block(
    blocks: List[Dict[str, Any]],
    idx: int,
    page_no: int,
    body_font: float,
    lookahead: int = 3,
) -> bool:
    for offset in range(1, lookahead + 1):
        probe = idx + offset
        if probe >= len(blocks):
            break
        candidate = blocks[probe]
        candidate_page = int(candidate.get("page_no") or page_no)
        if candidate_page != page_no:
            break
        if _block_body_like(candidate, body_font):
            return True
    return False


def _is_short_display_heading(candidate_title: str) -> bool:
    canonical = canonicalize_heading(candidate_title)
    if canonical in _CORE_SECTION_CANONICALS:
        return False
    tokens = candidate_title.split()
    if len(tokens) > 4:
        return False
    return True


def _passes_heuristic_layout_filter(
    candidate_title: str,
    detection_kind: str,
    block: Dict[str, Any],
    idx: int,
    blocks: List[Dict[str, Any]],
    body_font: float,
    page_geom: Dict[int, Tuple[float, float, float, float]],
    page_heading_density: Dict[int, int],
) -> bool:
    canonical = canonicalize_heading(candidate_title)
    structural_kind = detection_kind in {
        "abstract",
        "references",
        "numeric",
        "roman",
        "standalone_marker",
    }
    if structural_kind:
        return True

    page_no = int(block.get("page_no") or 1)
    x0, y0, x1, _ = _block_bbox(block)
    g = page_geom.get(page_no, (x0, y0, max(x1, x0 + 1.0), max(y0 + 1.0, y0 + 1.0)))
    width = max(1.0, g[2] - g[0])
    height = max(1.0, g[3] - g[1])
    x0_rel = (x0 - g[0]) / width
    y0_rel = (y0 - g[1]) / height
    width_rel = max(0.0, (x1 - x0) / width)
    near_top = y0_rel <= 0.26
    left_anchor = x0_rel <= 0.25
    has_nearby_body = _has_nearby_body_block(blocks, idx=idx, page_no=page_no, body_font=body_font, lookahead=3)
    short_display = _is_short_display_heading(candidate_title)
    dense_page = page_heading_density.get(page_no, 0) >= 5
    is_core_canonical = canonical in _CORE_SECTION_CANONICALS

    # Figure labels are usually short, narrow, and not followed by body text.
    if short_display and dense_page and not has_nearby_body:
        return False
    if short_display and width_rel < 0.2 and not has_nearby_body:
        return False

    # Unnumbered visual labels (common in figure diagrams) are often all-caps/style-only
    # titles floating in page interiors. Require stronger structural cues for these.
    if detection_kind in {"all_caps", "style_guess"} and not is_core_canonical:
        if not (near_top and left_anchor):
            return False
        if not has_nearby_body:
            return False
        if short_display and width_rel < 0.35:
            return False

    # Non-structural headings should usually be top/left anchored or followed by body text.
    if not has_nearby_body and not (near_top and left_anchor):
        return False

    return True


def _extract_heuristic_headings(blocks: List[Dict[str, Any]]) -> List[HeadingCandidate]:
    body_font_candidates: List[float] = []
    for block in blocks:
        metadata = block.get("metadata")
        if not isinstance(metadata, dict):
            continue
        char_count = int(_safe_float(metadata.get("char_count"), 0))
        line_count = int(_safe_float(metadata.get("line_count"), 0))
        avg_font = _safe_float(metadata.get("avg_font_size"), 0.0)
        if char_count >= 80 and line_count >= 2 and avg_font > 0:
            body_font_candidates.append(avg_font)

    body_font = 0.0
    if body_font_candidates:
        sorted_fonts = sorted(body_font_candidates)
        body_font = sorted_fonts[len(sorted_fonts) // 2]

    page_geom = _page_geometries(blocks)
    page_heading_density: Dict[int, int] = {}
    for block in blocks:
        page_no = int(block.get("page_no") or 1)
        line = _block_first_line(block)
        if not line:
            continue
        if _extract_numeric_heading_title(line) or _extract_roman_heading_title(line):
            page_heading_density[page_no] = page_heading_density.get(page_no, 0) + 1
            continue
        if _looks_like_heading_phrase(line):
            page_heading_density[page_no] = page_heading_density.get(page_no, 0) + 1

    headings: List[HeadingCandidate] = []
    for idx, block in enumerate(blocks):
        raw_text = str(block.get("text") or "").replace("\x00", "").strip()
        if not raw_text:
            continue
        raw_lines = [" ".join(line.split()) for line in raw_text.splitlines() if line.strip()]
        page_no = int(block.get("page_no") or 1)
        metadata = block.get("metadata")
        first_line_raw = ""
        if isinstance(metadata, dict):
            first_line_raw = str(metadata.get("first_line") or "")
        if not first_line_raw:
            first_line_raw = raw_text.splitlines()[0]
        line = " ".join(first_line_raw.split())
        if not line:
            continue

        next_line = ""
        next_meta: Dict[str, Any] = {}
        if idx + 1 < len(blocks):
            next_block = blocks[idx + 1]
            next_page_no = int(next_block.get("page_no") or page_no)
            if next_page_no == page_no:
                raw_next_text = str(next_block.get("text") or "").replace("\x00", "").strip()
                if raw_next_text:
                    candidate = next_block.get("metadata")
                    if isinstance(candidate, dict):
                        next_meta = candidate
                        next_line = str(candidate.get("first_line") or "").strip()
                    if not next_line:
                        next_line = raw_next_text.splitlines()[0].strip()
                    next_line = " ".join(next_line.split())

        candidate_title = ""
        detection_kind = ""
        line_l = line.lower()
        if line_l.startswith("abstract"):
            candidate_title = "Abstract"
            detection_kind = "abstract"
        elif line_l.startswith("references"):
            candidate_title = "References"
            detection_kind = "references"
        else:
            numeric_title = _extract_numeric_heading_title(line)
            if numeric_title:
                candidate_title = numeric_title
                detection_kind = "numeric"
            roman_title = _extract_roman_heading_title(line) if not candidate_title else ""
            if roman_title:
                candidate_title = roman_title
                detection_kind = "roman"
            else:
                words = line.split()
                if 1 <= len(words) <= 12 and all(_is_upper_token(token) for token in words):
                    candidate_title = _clean_heading_title(line)
                    detection_kind = "all_caps"

        if not candidate_title and _is_standalone_heading_marker(line):
            inline_heading_line = raw_lines[1] if len(raw_lines) >= 2 else ""
            if _looks_like_heading_phrase(inline_heading_line):
                candidate_title = _clean_heading_title(inline_heading_line)
                detection_kind = "standalone_marker"
            elif _looks_like_heading_phrase(next_line):
                next_line_count = int(_safe_float(next_meta.get("line_count"), 1)) if next_meta else 1
                next_max_font = _safe_float(next_meta.get("max_font_size"), 0.0) if next_meta else 0.0
                next_bold_ratio = _safe_float(next_meta.get("bold_ratio"), 0.0) if next_meta else 0.0
                next_font_prominent = body_font > 0 and next_max_font >= (body_font + 0.6)
                if next_line_count <= 2 and (next_font_prominent or next_bold_ratio >= 0.25):
                    candidate_title = _clean_heading_title(next_line)
                    detection_kind = "standalone_marker"

        if not candidate_title and isinstance(metadata, dict):
            max_font = _safe_float(metadata.get("max_font_size"), 0.0)
            bold_ratio = _safe_float(metadata.get("bold_ratio"), 0.0)
            line_count = int(_safe_float(metadata.get("line_count"), 1))
            words = line.split()
            title_case_words = sum(1 for token in words if token[:1].isupper())
            looks_title_case = len(words) <= 12 and title_case_words >= max(2, len(words) - 1)
            font_prominent = body_font > 0 and max_font >= (body_font + 1.0)
            if (
                looks_title_case
                and line_count <= 2
                and len(line) <= 120
                and (font_prominent or bold_ratio >= 0.45)
                and not line_l.startswith(("figure", "table", "arxiv:", "http://", "https://"))
            ):
                candidate_title = _clean_heading_title(line)
                detection_kind = "style_guess"

        if not candidate_title:
            continue
        if not _is_reasonable_heading_title(candidate_title):
            continue
        if not _passes_heuristic_layout_filter(
            candidate_title=candidate_title,
            detection_kind=detection_kind or "unknown",
            block=block,
            idx=idx,
            blocks=blocks,
            body_font=body_font,
            page_geom=page_geom,
            page_heading_density=page_heading_density,
        ):
            continue

        headings.append(
            HeadingCandidate(
                title=candidate_title,
                level=1,
                source="heuristic",
                confidence=0.72,
                block_hint=idx,
                page_hint=page_no,
            )
        )
    return _dedupe_headings(headings)


def _dedupe_headings(headings: Iterable[HeadingCandidate]) -> List[HeadingCandidate]:
    deduped: List[HeadingCandidate] = []
    seen = set()
    for heading in headings:
        normalized = _normalize_text(heading.title)
        key = (normalized, heading.level)
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(heading)
    return deduped


def _seed_heading_page_hints(
    headings: List[HeadingCandidate],
    reference_headings: List[HeadingCandidate],
) -> List[HeadingCandidate]:
    if not headings:
        return []
    if not reference_headings:
        return headings

    title_page: Dict[str, int] = {}
    canonical_page: Dict[str, int] = {}
    for ref in reference_headings:
        if ref.page_hint is None:
            continue
        page_hint = int(ref.page_hint)
        if page_hint <= 0:
            continue
        norm_title = _normalize_text(ref.title)
        if norm_title:
            title_page.setdefault(norm_title, page_hint)
        canonical = canonicalize_heading(ref.title)
        if canonical:
            canonical_page.setdefault(canonical, page_hint)

    enriched: List[HeadingCandidate] = []
    for heading in headings:
        page_hint = heading.page_hint
        if page_hint is None:
            norm_title = _normalize_text(heading.title)
            canonical = canonicalize_heading(heading.title)
            inferred = title_page.get(norm_title) or canonical_page.get(canonical)
            if inferred and inferred > 0:
                page_hint = inferred
        enriched.append(
            HeadingCandidate(
                title=heading.title,
                level=heading.level,
                source=heading.source,
                confidence=heading.confidence,
                block_hint=heading.block_hint,
                page_hint=page_hint,
            )
        )
    return enriched


def _fill_missing_heading_page_hints(
    headings: List[HeadingCandidate],
    total_pages: int,
) -> List[HeadingCandidate]:
    if not headings:
        return []

    page_cap = max(1, int(total_pages or 1))
    normalized_hints: List[Optional[int]] = []
    for heading in headings:
        page_hint = heading.page_hint
        if page_hint is None:
            normalized_hints.append(None)
            continue
        try:
            page_val = int(page_hint)
        except (TypeError, ValueError):
            normalized_hints.append(None)
            continue
        if page_val <= 0:
            normalized_hints.append(None)
            continue
        normalized_hints.append(min(page_cap, page_val))

    n = len(normalized_hints)
    prev_known: List[Optional[int]] = [None] * n
    next_known: List[Optional[int]] = [None] * n

    last: Optional[int] = None
    for idx in range(n):
        prev_known[idx] = last
        if normalized_hints[idx] is not None:
            last = normalized_hints[idx]

    last = None
    for idx in range(n - 1, -1, -1):
        next_known[idx] = last
        if normalized_hints[idx] is not None:
            last = normalized_hints[idx]

    for idx in range(n):
        if normalized_hints[idx] is not None:
            continue
        prev_page = prev_known[idx]
        next_page = next_known[idx]
        inferred: Optional[int] = None
        if prev_page is not None and next_page is not None:
            if next_page <= prev_page:
                inferred = prev_page
            elif (next_page - prev_page) <= 1:
                inferred = next_page
            else:
                inferred = prev_page + 1
        elif prev_page is not None:
            inferred = prev_page
        elif next_page is not None:
            inferred = max(1, next_page - 1)
        normalized_hints[idx] = inferred

    # Keep inferred hints monotonic with source heading order.
    running_hint = 1
    for idx in range(n):
        hint = normalized_hints[idx]
        if hint is None:
            continue
        if hint < running_hint:
            hint = running_hint
        hint = min(page_cap, hint)
        normalized_hints[idx] = hint
        running_hint = hint

    enriched: List[HeadingCandidate] = []
    for idx, heading in enumerate(headings):
        enriched.append(
            HeadingCandidate(
                title=heading.title,
                level=heading.level,
                source=heading.source,
                confidence=heading.confidence,
                block_hint=heading.block_hint,
                page_hint=normalized_hints[idx],
            )
        )
    return enriched


def _heading_match_score(heading_norm: str, block_norm: str) -> float:
    if not heading_norm or not block_norm:
        return 0.0
    if heading_norm in block_norm:
        if block_norm.startswith(heading_norm):
            return 1.0
        # Heading phrase appearing mid-line is usually a body mention, not a heading anchor.
        heading_tokens = heading_norm.split()
        token_hits = 0
        if heading_tokens:
            token_hits = sum(1 for token in heading_tokens if token in block_norm)
        return max(0.22, (token_hits / max(1, len(heading_tokens))) * 0.55)
    # Compare with the beginning of the block where headings usually appear.
    short_block = " ".join(block_norm.split()[:20])
    score = SequenceMatcher(None, heading_norm, short_block).ratio()
    heading_tokens = heading_norm.split()
    if heading_tokens:
        token_hits = sum(1 for token in heading_tokens if token in short_block)
        score = max(score, token_hits / len(heading_tokens))
    return score


def _is_heading_prefix_match(heading_norm: str, first_line_norm: str) -> bool:
    if not heading_norm or not first_line_norm:
        return False
    if first_line_norm == heading_norm:
        return True
    if first_line_norm.startswith(heading_norm + " "):
        return True
    return False


def _block_first_line(block: Dict[str, Any]) -> str:
    metadata = block.get("metadata")
    if isinstance(metadata, dict):
        first_line = str(metadata.get("first_line") or "").strip()
        if first_line:
            return " ".join(first_line.split())
    text = str(block.get("text") or "").replace("\x00", "").strip()
    if not text:
        return ""
    return " ".join(text.splitlines()[0].split())


def _estimate_body_font(blocks: List[Dict[str, Any]]) -> float:
    body_font_candidates: List[float] = []
    for block in blocks:
        metadata = block.get("metadata")
        if not isinstance(metadata, dict):
            continue
        char_count = int(_safe_float(metadata.get("char_count"), 0))
        line_count = int(_safe_float(metadata.get("line_count"), 0))
        avg_font = _safe_float(metadata.get("avg_font_size"), 0.0)
        if char_count >= 80 and line_count >= 2 and avg_font > 0:
            body_font_candidates.append(avg_font)
    if not body_font_candidates:
        return 0.0
    sorted_fonts = sorted(body_font_candidates)
    return sorted_fonts[len(sorted_fonts) // 2]


def _heading_level_from_line(line: str) -> int:
    compact = " ".join(str(line or "").split()).strip()
    if not compact:
        return 1
    match = _NUMERIC_HEADING_RE.match(compact)
    if match:
        return min(3, match.group("num").count(".") + 1)
    match = _ROMAN_HEADING_RE.match(compact)
    if match:
        return 1
    return 1


def _extract_local_heading_anchors(blocks: List[Dict[str, Any]]) -> List[HeadingCandidate]:
    if not blocks:
        return []

    body_font = _estimate_body_font(blocks)
    anchors: List[HeadingCandidate] = []
    seen: set[Tuple[int, str]] = set()

    for idx, block in enumerate(blocks):
        raw_text = str(block.get("text") or "").replace("\x00", "").strip()
        if not raw_text:
            continue
        line = _block_first_line(block)
        raw_lines = [" ".join(part.split()) for part in raw_text.splitlines() if " ".join(part.split())]
        if not line:
            continue
        compact_line = " ".join(line.split()).strip()
        if not compact_line:
            continue

        metadata = block.get("metadata")
        meta = metadata if isinstance(metadata, dict) else {}
        line_count = int(_safe_float(meta.get("line_count"), 1))
        max_font = _safe_float(meta.get("max_font_size"), 0.0)
        bold_ratio = _safe_float(meta.get("bold_ratio"), 0.0)
        char_count = int(_safe_float(meta.get("char_count"), len(raw_text)))

        candidate_title = ""
        level = 1
        lowered = compact_line.lower()

        if lowered.startswith("abstract"):
            candidate_title = "Abstract"
        elif lowered.startswith("references"):
            candidate_title = "References"
        else:
            numeric_title = _extract_numeric_heading_title(compact_line)
            if numeric_title:
                candidate_title = numeric_title
                level = _heading_level_from_line(compact_line)
            else:
                roman_title = _extract_roman_heading_title(compact_line)
                if roman_title:
                    candidate_title = roman_title
                    level = 1
                else:
                    if _is_standalone_heading_marker(compact_line):
                        inline_heading_line = raw_lines[1] if len(raw_lines) >= 2 else ""
                        if (
                            inline_heading_line
                            and not _looks_like_sentenceish_prose(inline_heading_line)
                            and (_looks_like_heading_phrase(inline_heading_line) or _is_reasonable_heading_title(inline_heading_line))
                        ):
                            candidate_title = _clean_heading_title(inline_heading_line)
                            level = 1
                        elif idx + 1 < len(blocks):
                            next_line = _block_first_line(blocks[idx + 1])
                            if (
                                next_line
                                and not _looks_like_sentenceish_prose(next_line)
                                and (_looks_like_heading_phrase(next_line) or _is_reasonable_heading_title(next_line))
                            ):
                                candidate_title = _clean_heading_title(next_line)
                                level = 1
                    if candidate_title:
                        pass
                    else:
                        words = compact_line.split()
                        title_case_words = sum(1 for token in words if token[:1].isupper() or _is_upper_token(token))
                        font_prominent = body_font > 0 and max_font >= (body_font + 0.9)
                        if (
                            1 <= len(words) <= 12
                            and title_case_words >= max(2, len(words) - 1)
                            and line_count <= 2
                            and char_count <= 140
                            and (font_prominent or bold_ratio >= 0.4)
                            and not lowered.startswith(("figure", "table", "algorithm", "arxiv:", "http://", "https://"))
                        ):
                            candidate_title = _clean_heading_title(compact_line)
                            level = 1

        candidate_title = _clean_heading_title(candidate_title)
        if not candidate_title:
            continue
        if not _is_reasonable_heading_title(candidate_title):
            continue
        title_lower = candidate_title.lower()
        if title_lower.startswith(("we ", "our ", "this ", "these ", "those ")):
            continue
        if _looks_like_sentenceish_prose(candidate_title):
            continue

        key = (idx, _normalize_text(candidate_title))
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            HeadingCandidate(
                title=candidate_title,
                level=level,
                source="local_anchor",
                confidence=0.86,
                block_hint=idx,
                page_hint=int(block.get("page_no") or 1),
            )
        )

    return anchors


def _local_anchor_match_score(
    heading: HeadingCandidate,
    anchor: HeadingCandidate,
) -> float:
    heading_norm = _normalize_text(heading.title)
    anchor_norm = _normalize_text(anchor.title)
    if not heading_norm or not anchor_norm:
        return 0.0

    score = SequenceMatcher(None, heading_norm, anchor_norm).ratio()
    heading_canonical = canonicalize_heading(heading.title)
    anchor_canonical = canonicalize_heading(anchor.title)
    if heading_canonical == anchor_canonical:
        score += 0.35
    if heading_norm == anchor_norm:
        score += 0.45
    elif heading_norm in anchor_norm or anchor_norm in heading_norm:
        score += 0.18

    if heading.page_hint is not None and anchor.page_hint is not None:
        page_dist = abs(int(heading.page_hint) - int(anchor.page_hint))
        if page_dist == 0:
            score += 0.2
        elif page_dist == 1:
            score += 0.1
        elif page_dist >= 3:
            score -= min(0.25, 0.05 * (page_dist - 2))

    if heading.level == anchor.level:
        score += 0.06
    elif heading.level == 1 and anchor.level == 2:
        score -= 0.08

    return score


def _find_local_anchor_for_heading(
    heading: HeadingCandidate,
    anchors: List[HeadingCandidate],
    *,
    search_start: int,
) -> Optional[Tuple[int, float]]:
    best_idx = -1
    best_score = -1.0
    for anchor in anchors:
        block_hint = anchor.block_hint
        if block_hint is None or block_hint < search_start:
            continue
        score = _local_anchor_match_score(heading, anchor)
        if score > best_score:
            best_score = score
            best_idx = block_hint
    if best_idx < 0 or best_score < 1.02:
        return None
    return best_idx, best_score


def _block_heading_shape_score(block: Dict[str, Any], body_font: float, first_line: str) -> float:
    metadata = block.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}
    line_count = int(_safe_float(meta.get("line_count"), 1))
    max_font = _safe_float(meta.get("max_font_size"), 0.0)
    bold_ratio = _safe_float(meta.get("bold_ratio"), 0.0)

    score = 0.0
    if line_count <= 2:
        score += 0.08
    if _looks_like_heading_phrase(first_line):
        score += 0.13
    if _extract_numeric_heading_title(first_line) or _extract_roman_heading_title(first_line):
        score += 0.1
    if body_font > 0 and max_font >= (body_font + 0.6):
        score += 0.1
    if bold_ratio >= 0.25:
        score += min(0.12, bold_ratio * 0.2)
    if _HEADING_NOISE_RE.match(first_line.lower()):
        score -= 0.25
    return score


def _heading_block_score(
    heading: HeadingCandidate,
    heading_norm: str,
    block: Dict[str, Any],
    block_norm: str,
    first_line: str,
    first_line_norm: str,
    body_font: float,
) -> float:
    if not heading_norm:
        return 0.0

    # Prefer heading-like first lines over arbitrary body text mentions.
    score = _heading_match_score(heading_norm, first_line_norm)
    if _is_heading_prefix_match(heading_norm, first_line_norm):
        score = max(score, 1.0)
    elif heading_norm in first_line_norm:
        score = max(score, 0.35)

    if score < 0.9:
        short_block = " ".join(block_norm.split()[:20])
        score = max(score, _heading_match_score(heading_norm, short_block) * 0.82)

    heading_tokens = heading_norm.split()
    if heading_tokens and first_line_norm:
        token_hits = sum(1 for token in heading_tokens if token in first_line_norm)
        score = max(score, token_hits / len(heading_tokens))

    shape_bonus = _block_heading_shape_score(block, body_font=body_font, first_line=first_line)
    score += shape_bonus

    page_hint = heading.page_hint
    if page_hint is not None:
        page_no = int(block.get("page_no") or 1)
        page_dist = abs(page_no - int(page_hint))
        if page_dist == 0:
            score += 0.12
        elif page_dist == 1:
            score += 0.04
        elif page_dist >= 3:
            score -= min(0.25, 0.04 * (page_dist - 1))

    if _is_standalone_heading_marker(first_line):
        score -= 0.22

    if heading.source == "pdf_toc":
        score += 0.04
    elif heading.source == "arxiv_source":
        score += 0.02
    elif heading.source == "grobid":
        score += 0.01

    return score


def _heading_min_score(source: str) -> float:
    if source == "pdf_toc":
        return 0.48
    if source == "grobid":
        return 0.58
    if source == "arxiv_source":
        return 0.56
    if source == "heuristic":
        return 0.44
    return 0.6


def _align_headings_to_spans(
    headings: List[HeadingCandidate],
    blocks: List[Dict[str, Any]],
) -> List[SectionSpan]:
    if not headings or not blocks:
        return []

    normalized_blocks = [_normalize_text(str(block.get("text") or "")[:320]) for block in blocks]
    first_lines = [_block_first_line(block) for block in blocks]
    normalized_first_lines = [_normalize_text(line) for line in first_lines]
    body_font = _estimate_body_font(blocks)
    local_anchors = _extract_local_heading_anchors(blocks)

    page_ranges: Dict[int, Tuple[int, int]] = {}
    for idx, block in enumerate(blocks):
        page_no = int(block.get("page_no") or 1)
        if page_no in page_ranges:
            first_idx, _ = page_ranges[page_no]
            page_ranges[page_no] = (first_idx, idx)
        else:
            page_ranges[page_no] = (idx, idx)

    max_page = max(page_ranges.keys()) if page_ranges else 1

    matched: List[Tuple[HeadingCandidate, int, float]] = []
    search_start = 0

    for heading_idx, heading in enumerate(headings):
        heading_norm = _normalize_text(heading.title)
        if not heading_norm:
            continue
        best_idx = -1
        best_score = -1.0

        if heading.block_hint is not None and search_start <= heading.block_hint < len(blocks):
            best_idx = heading.block_hint
            best_score = 1.3
        else:
            anchor_match = _find_local_anchor_for_heading(heading, local_anchors, search_start=search_start)
            if anchor_match is not None:
                best_idx, best_score = anchor_match

            scan_start = search_start
            scan_end = len(blocks) - 1
            if heading.page_hint is not None and heading.page_hint in page_ranges:
                page_start, _ = page_ranges[heading.page_hint]
                scan_start = max(scan_start, page_start)

                end_page = min(max_page, heading.page_hint + 2)
                end_idx = None
                for page in range(end_page, heading.page_hint - 1, -1):
                    if page in page_ranges:
                        _, end_candidate = page_ranges[page]
                        end_idx = end_candidate
                        break
                if end_idx is not None:
                    scan_end = max(scan_start, end_idx)
            elif heading.page_hint is None:
                # Avoid drifting unhinted headings to late-page body mentions by
                # bounding search to the neighborhood before the next hinted heading.
                next_hint: Optional[int] = None
                for probe in headings[heading_idx + 1 :]:
                    if probe.page_hint is None:
                        continue
                    candidate = int(probe.page_hint)
                    if candidate in page_ranges:
                        next_hint = candidate
                        break
                if next_hint is not None:
                    bounded_end_page = min(max_page, next_hint + 1)
                    end_idx = None
                    for page in range(bounded_end_page, 0, -1):
                        if page in page_ranges:
                            _, end_candidate = page_ranges[page]
                            end_idx = end_candidate
                            break
                    if end_idx is not None:
                        scan_end = min(scan_end, end_idx)

            if scan_start < len(blocks):
                scan_end = min(scan_end, len(blocks) - 1)
                if scan_end >= scan_start:
                    for idx in range(scan_start, scan_end + 1):
                        score = _heading_block_score(
                            heading=heading,
                            heading_norm=heading_norm,
                            block=blocks[idx],
                            block_norm=normalized_blocks[idx],
                            first_line=first_lines[idx],
                            first_line_norm=normalized_first_lines[idx],
                            body_font=body_font,
                        )
                        if score > best_score:
                            best_score = score
                            best_idx = idx

        min_score = _heading_min_score(heading.source)

        if (
            (best_idx < 0 or best_score < min_score)
            and heading.page_hint is not None
            and heading.page_hint in page_ranges
        ):
            hint_page = int(heading.page_hint)
            fallback_start = max(search_start, page_ranges[hint_page][0])
            fallback_end = page_ranges[hint_page][1]
            if hint_page + 1 in page_ranges:
                # Some renderers place heading text at the very end of previous page.
                fallback_end = max(fallback_end, page_ranges[hint_page + 1][0])

            fallback_idx = -1
            fallback_score = -1.0
            for idx in range(fallback_start, min(len(blocks), fallback_end + 1)):
                score = _heading_block_score(
                    heading=heading,
                    heading_norm=heading_norm,
                    block=blocks[idx],
                    block_norm=normalized_blocks[idx],
                    first_line=first_lines[idx],
                    first_line_norm=normalized_first_lines[idx],
                    body_font=body_font,
                )
                if score > fallback_score:
                    fallback_score = score
                    fallback_idx = idx

            relaxed_min = min_score - 0.14
            if fallback_idx >= 0 and fallback_score >= relaxed_min:
                best_idx = fallback_idx
                best_score = fallback_score

        if best_idx >= 0 and best_score >= min_score:
            matched.append((heading, best_idx, best_score))
            search_start = max(search_start, best_idx + 1)

    if not matched:
        return []

    # Remove duplicates that resolve to the same block index.
    deduped_matches: List[Tuple[HeadingCandidate, int, float]] = []
    for item in matched:
        if deduped_matches and item[1] == deduped_matches[-1][1]:
            if item[2] > deduped_matches[-1][2]:
                deduped_matches[-1] = item
            continue
        deduped_matches.append(item)

    spans: List[SectionSpan] = []
    first_idx = deduped_matches[0][1]
    if first_idx > 0:
        first_block = blocks[0]
        before_block = blocks[first_idx - 1]
        spans.append(
            SectionSpan(
                index=0,
                title="Front Matter",
                canonical="front_matter",
                level=1,
                source="fallback",
                confidence=0.4,
                start_idx=0,
                end_idx=first_idx - 1,
                start_page=int(first_block.get("page_no") or 1),
                end_page=int(before_block.get("page_no") or first_block.get("page_no") or 1),
            )
        )

    for i, (heading, start_idx, score) in enumerate(deduped_matches):
        end_idx = deduped_matches[i + 1][1] - 1 if i + 1 < len(deduped_matches) else len(blocks) - 1
        if end_idx < start_idx:
            continue
        start_block = blocks[start_idx]
        end_block = blocks[end_idx]
        span_confidence = round((heading.confidence * 0.7) + (score * 0.3), 3)
        spans.append(
            SectionSpan(
                index=len(spans),
                title=heading.title,
                canonical=canonicalize_heading(heading.title),
                level=heading.level,
                source=heading.source,
                confidence=span_confidence,
                start_idx=start_idx,
                end_idx=end_idx,
                start_page=int(start_block.get("page_no") or 1),
                end_page=int(end_block.get("page_no") or start_block.get("page_no") or 1),
            )
        )

    return spans


def _apply_spans_to_blocks(blocks: List[Dict[str, Any]], spans: List[SectionSpan]) -> None:
    if not spans:
        fallback_span = SectionSpan(
            index=0,
            title="Document Body",
            canonical="other",
            level=1,
            source="fallback",
            confidence=0.35,
            start_idx=0,
            end_idx=max(0, len(blocks) - 1),
            start_page=int((blocks[0].get("page_no") if blocks else 1) or 1),
            end_page=int((blocks[-1].get("page_no") if blocks else 1) or 1),
        )
        spans = [fallback_span]

    for span in spans:
        for idx in range(span.start_idx, span.end_idx + 1):
            if idx < 0 or idx >= len(blocks):
                continue
            block = blocks[idx]
            metadata = block.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
                block["metadata"] = metadata
            metadata["section_title"] = span.title
            metadata["section_canonical"] = span.canonical
            metadata["section_level"] = span.level
            metadata["section_source"] = span.source
            metadata["section_confidence"] = span.confidence
            metadata["section_index"] = span.index


def _count_non_front_spans(span_items: List[SectionSpan]) -> int:
    return sum(1 for item in span_items if item.canonical != "front_matter")


def _extract_document_title(pdf_path: Path, blocks: List[Dict[str, Any]]) -> str:
    try:
        reader = PdfReader(str(pdf_path))
        metadata = reader.metadata or {}
        title = str(metadata.get("/Title") or metadata.get("Title") or "").strip()
        if title:
            return _normalize_text(title)
    except Exception:
        pass

    if blocks:
        first_meta = blocks[0].get("metadata")
        if isinstance(first_meta, dict):
            first_line = str(first_meta.get("first_line") or "").strip()
            if first_line:
                return _normalize_text(first_line)
        first_text = str(blocks[0].get("text") or "").splitlines()
        if first_text:
            return _normalize_text(first_text[0])
    return ""


def _strategy_score(
    source_name: str,
    heading_items: List[HeadingCandidate],
    span_items: List[SectionSpan],
    total_pages: int,
    document_title_norm: str,
) -> float:
    matched_non_front = _count_non_front_spans(span_items)
    if matched_non_front <= 0:
        return -1.0

    non_front_spans = [item for item in span_items if item.canonical != "front_matter"]
    coverage = matched_non_front / max(1, len(heading_items))
    confidences = [item.confidence for item in non_front_spans]
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    core_count = sum(1 for item in non_front_spans if item.canonical in _CORE_SECTION_CANONICALS)
    core_ratio = (core_count / len(non_front_spans)) if non_front_spans else 0.0
    source_bonus = {
        "pdf_toc": 1.25,
        "arxiv_source": 1.1,
        "grobid": 1.0,
        "heuristic": 0.7,
    }.get(source_name, 0.6)
    # Saturate the span-count contribution so noisy strategies don't win just by over-segmenting.
    span_count_term = min(matched_non_front, 8) * 0.25
    score = source_bonus + span_count_term + (coverage * 1.5) + (avg_confidence * 0.5)

    unique_canonicals = {item.canonical for item in non_front_spans}
    diversity = len(unique_canonicals) / max(1, matched_non_front)
    score += diversity * 0.4
    score += core_ratio * 0.9

    # Structured sources should match a reasonable fraction of the provided headings.
    if heading_items and source_name in {"pdf_toc", "arxiv_source", "grobid"}:
        expected_ratio = {"pdf_toc": 0.55, "arxiv_source": 0.6, "grobid": 0.55}.get(source_name, 0.55)
        expected_non_front = max(1, int(round(len(heading_items) * expected_ratio)))
        if matched_non_front < expected_non_front:
            deficit = expected_non_front - matched_non_front
            score -= min(3.8, deficit * 0.75)

    if total_pages > 0 and non_front_spans:
        span_lengths = [max(1, item.end_page - item.start_page + 1) for item in non_front_spans]
        longest_ratio = max(span_lengths) / max(1, total_pages)
        if longest_ratio > 0.6:
            score -= (longest_ratio - 0.6) * 2.4

        abstract_span = next((item for item in non_front_spans if item.canonical == "abstract"), None)
        if abstract_span:
            abstract_pages = max(1, abstract_span.end_page - abstract_span.start_page + 1)
            if abstract_pages > 2:
                score -= min(1.2, 0.45 * (abstract_pages - 2))
            if total_pages >= 6:
                abstract_ratio = abstract_pages / max(1, total_pages)
                if abstract_ratio > 0.22:
                    score -= min(5.0, (abstract_ratio - 0.22) * 11.5)

        intro_span = next((item for item in non_front_spans if item.canonical == "introduction"), None)
        if intro_span and total_pages >= 8:
            late_intro_threshold = max(3, int(total_pages * 0.35))
            if intro_span.start_page > late_intro_threshold:
                score -= min(3.0, (intro_span.start_page - late_intro_threshold) * 0.55)

    if source_name == "pdf_toc":
        if len(heading_items) <= 2:
            score -= 1.2
        elif len(heading_items) <= 3:
            score -= 0.8

        if document_title_norm:
            title_like_headings = 0
            for heading in heading_items:
                heading_norm = _normalize_text(heading.title)
                if not heading_norm:
                    continue
                similarity = SequenceMatcher(None, heading_norm, document_title_norm).ratio()
                if similarity >= 0.84:
                    title_like_headings += 1
            if title_like_headings:
                score -= min(3.2, title_like_headings * 1.6)

        title_slug_like = sum(
            1
            for span in non_front_spans
            if span.canonical not in KNOWN_CANONICALS and len(span.canonical.split("_")) >= 6
        )
        if title_slug_like:
            score -= min(2.8, title_slug_like * 1.4)

    if source_name == "heuristic":
        if core_ratio < 0.55:
            score -= min(2.4, (0.55 - core_ratio) * 4.0)
        if len(heading_items) > 18:
            score -= min(4.5, 1.0 + (len(heading_items) - 18) * 0.35)
        elif len(heading_items) > 12:
            score -= (len(heading_items) - 12) * 0.15

        page_counts: Dict[int, int] = {}
        for heading in heading_items:
            page_no = int(heading.page_hint or 0)
            if page_no <= 0:
                continue
            page_counts[page_no] = page_counts.get(page_no, 0) + 1
        if page_counts:
            max_on_page = max(page_counts.values())
            if max_on_page >= 6:
                score -= min(2.4, 0.35 * (max_on_page - 5))
            concentration = max_on_page / max(1, len(heading_items))
            if concentration >= 0.55 and len(heading_items) >= 8:
                score -= min(1.6, (concentration - 0.5) * 4.0)

        reference_like = sum(1 for span in non_front_spans if span.canonical == "references")
        if reference_like >= 3:
            score -= min(1.8, (reference_like - 2) * 0.4)

        generic_long_slugs = sum(
            1 for span in non_front_spans
            if span.canonical not in KNOWN_CANONICALS and len(span.canonical.split("_")) >= 5
        )
        if generic_long_slugs:
            score -= min(3.2, generic_long_slugs * 0.22)

    if source_name == "arxiv_source" and non_front_spans and total_pages >= 8:
        first_span = min(non_front_spans, key=lambda item: item.start_page)
        has_early_core = any(
            item.canonical in {"abstract", "introduction", "related_work", "methodology", "experiments", "results"}
            for item in non_front_spans
        )
        if first_span.start_page > 3:
            score -= min(6.0, (first_span.start_page - 3) * 0.6)
        if not has_early_core:
            score -= 3.0

    return score


def annotate_blocks_with_sections(
    blocks: List[Dict[str, Any]],
    pdf_path: Path,
    source_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Annotate block metadata with canonical section information.

    Returns a report dictionary for ingestion diagnostics.
    """
    if not blocks:
        return {
            "strategy": "none",
            "candidate_headings": 0,
            "matched_headings": 0,
            "sections": [],
        }

    strategy = "heuristic"
    headings: List[HeadingCandidate] = []
    spans: List[SectionSpan] = []
    total_pages = max(int(block.get("page_no") or 1) for block in blocks)

    heuristic_headings = _extract_heuristic_headings(blocks)
    heuristic_headings = _fill_missing_heading_page_hints(heuristic_headings, total_pages=total_pages)
    arxiv_headings = _fill_missing_heading_page_hints(
        _seed_heading_page_hints(
            _extract_headings_from_arxiv_source(source_url, pdf_path=pdf_path),
            heuristic_headings,
        ),
        total_pages=total_pages,
    )
    grobid_headings = _fill_missing_heading_page_hints(
        _seed_heading_page_hints(
            _extract_headings_with_grobid(pdf_path),
            heuristic_headings,
        ),
        total_pages=total_pages,
    )

    # Prefer local PDF outline if it already gives strong section coverage.
    toc_headings = _extract_headings_from_pdf_toc(pdf_path)
    if toc_headings:
        toc_canonicals = {canonicalize_heading(item.title) for item in toc_headings}
        abstract_heading = next(
            (item for item in heuristic_headings if canonicalize_heading(item.title) == "abstract"),
            None,
        )
        references_heading = next(
            (item for item in heuristic_headings if canonicalize_heading(item.title) == "references"),
            None,
        )
        if abstract_heading and "abstract" not in toc_canonicals:
            toc_headings = [abstract_heading, *toc_headings]
        if references_heading and "references" not in toc_canonicals:
            toc_headings = [*toc_headings, references_heading]
        toc_headings = _dedupe_headings(toc_headings)

    document_title_norm = _extract_document_title(pdf_path, blocks)

    candidates: List[Tuple[str, List[HeadingCandidate], List[SectionSpan], float]] = []
    strategy_inputs = [
        ("pdf_toc", toc_headings),
        ("arxiv_source", arxiv_headings),
        ("grobid", grobid_headings),
        ("heuristic", heuristic_headings),
    ]

    for source_name, source_headings in strategy_inputs:
        if not source_headings:
            continue
        source_spans = _align_headings_to_spans(source_headings, blocks)
        score = _strategy_score(
            source_name,
            source_headings,
            source_spans,
            total_pages=total_pages,
            document_title_norm=document_title_norm,
        )
        candidates.append((source_name, source_headings, source_spans, score))

    if candidates:
        candidates.sort(key=lambda item: item[3], reverse=True)
        strategy, headings, spans, _ = candidates[0]
    else:
        headings = heuristic_headings
        spans = _align_headings_to_spans(headings, blocks)
        strategy = "heuristic"

    _apply_spans_to_blocks(blocks, spans)

    report_sections = [
        {
            "index": span.index,
            "title": span.title,
            "canonical": span.canonical,
            "level": span.level,
            "source": span.source,
            "confidence": span.confidence,
            "start_page": span.start_page,
            "end_page": span.end_page,
            "start_block_index": span.start_idx,
            "end_block_index": span.end_idx,
        }
        for span in spans
    ]

    return {
        "strategy": strategy,
        "candidate_headings": len(headings),
        "matched_headings": len(spans),
        "sections": report_sections,
    }
