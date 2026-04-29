from __future__ import annotations

import math
import re
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .parser import extract_text_blocks
from .sectioning import annotate_blocks_with_sections

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")
_STOPWORDS = {
    "about",
    "above",
    "after",
    "against",
    "also",
    "among",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "can",
    "could",
    "does",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "into",
    "its",
    "more",
    "most",
    "much",
    "must",
    "not",
    "our",
    "out",
    "over",
    "should",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "using",
    "very",
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
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_URL_RE = re.compile(r"https?://|www\.", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_CITATION_RE = re.compile(r"\[[0-9,\-\s]{1,20}\]|\bet al\.\b|\((?:19|20)\d{2}[a-z]?\)", re.I)
_FIGURE_TABLE_REF_RE = re.compile(r"\b(?:figure|fig\.?|table|tab\.?)\s+\d+[A-Za-z]?\b", re.I)
_CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.?|table|tab\.?|algorithm)\s+\d+[A-Za-z]?\b", re.I)
_SECTION_NUMBER_RE = re.compile(r"^\s*(?:\d+(?:\.\d+){0,3}|[A-Z]\.\d+)\s+")
_EQUATION_NOISE_RE = re.compile(r"[=+*/^_<>≤≥≈≠∈∑∫√λμσθΔΩαβγ⊙∥]{2,}")
_PANEL_LABEL_RE = re.compile(r"^\s*\([a-z0-9]\)\s+", re.I)
_SINGLE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 /+_-]{0,32}$")
_ARXIV_RE = re.compile(r"\barxiv:\S+", re.I)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")
_REFERENCE_ENTRY_RE = re.compile(r"^\s*\[\d+\]\s+")
_CODE_LINE_RE = re.compile(r"(?:\btorch\.|\bdef\b|\breturn\b|=\s*[^ ]|\[[A-Za-z_][A-Za-z0-9_ ,]*\]|_[A-Za-z0-9_]+)")
_BIBLIOGRAPHY_SIGNAL_RE = re.compile(
    r"\b(?:proceedings|conference|journal|volume|pages\b|arxiv|doi|cvpr|iccv|eccv|iclr|neurips|aaai|acl|emnlp)\b",
    re.I,
)
_BULLET_GLYPH_RE = re.compile(r"[■●▪□▢◆◇•]")
_EXPLANATORY_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|means|shows?|demonstrates?|uses?|builds?|improves?|achieves?|introduces?|proposes?|evaluates?|finds?)\b",
    re.I,
)
_PROMPT_LIKE_RE = re.compile(r"^\s*you are\s+", re.I)
_SECTION_TITLE_HINT_RE = re.compile(r"\b(?:qualitative|example|examples|visualization|additional results|supplementary)\b", re.I)
_TABLE_HEADER_TOKEN_RE = re.compile(r"^(?:[A-Z][A-Za-z0-9.+/-]*|[A-Z]{2,}[A-Za-z0-9.+/-]*|\d+(?:x\d+)?|[✓✗⋆†‡%↑↓×-]+)$")
_ABBREVIATION_SENTINELS = (
    ("et al.", "et al<prd>"),
    ("Fig.", "Fig<prd>"),
    ("Figs.", "Figs<prd>"),
    ("Sec.", "Sec<prd>"),
    ("Secs.", "Secs<prd>"),
    ("Eq.", "Eq<prd>"),
    ("Eqs.", "Eqs<prd>"),
    ("Tab.", "Tab<prd>"),
    ("Tabs.", "Tabs<prd>"),
    ("No.", "No<prd>"),
    ("vs.", "vs<prd>"),
    ("e.g.", "e<prd>g<prd>"),
    ("i.e.", "i<prd>e<prd>"),
)
_CUE_TERMS: Dict[str, Tuple[str, ...]] = {
    "abstract": ("propose", "introduce", "show", "results", "demonstrate"),
    "introduction": ("motivation", "problem", "challenge", "goal", "we"),
    "related_work": ("prior", "previous", "existing", "related", "compared"),
    "methodology": ("propose", "method", "framework", "approach", "architecture", "objective"),
    "experiments": ("evaluate", "benchmark", "dataset", "results", "compare"),
    "results": ("outperform", "improve", "results", "achieve", "gain"),
    "discussion": ("suggest", "indicate", "analysis", "limitation", "insight"),
    "conclusion": ("conclude", "summary", "future", "limitation", "work"),
}
_DEFAULT_SKIP_CANONICALS = {"front_matter", "references", "acknowledgements"}


@dataclass(slots=True)
class SectionOverviewConfig:
    include_front_matter: bool = False
    include_references: bool = False
    include_acknowledgements: bool = False
    include_appendix: bool = False
    min_sentences_per_section: int = 1
    max_sentences_per_section: int = 4
    min_words_per_section: int = 75
    max_words_per_section: int = 220
    sentence_similarity_threshold: float = 0.72
    max_sentence_chars: int = 360


@dataclass(slots=True)
class SectionOverviewItem:
    section_title: str
    section_canonical: str
    section_level: int
    page_start: int
    page_end: int
    block_count: int
    word_count: int
    summary_paragraph: str
    source_sentences: List[str] = field(default_factory=list)


@dataclass(slots=True)
class SectionOverviewResult:
    source_pdf: Path
    title: str
    sections: List[SectionOverviewItem] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def section_count(self) -> int:
        return len(self.sections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_pdf": str(self.source_pdf),
            "title": self.title,
            "section_count": self.section_count,
            "metadata": dict(self.metadata),
            "sections": [asdict(item) for item in self.sections],
        }


def build_section_overview(
    pdf_path: str | Path,
    *,
    blocks: Optional[List[Dict[str, Any]]] = None,
    source_url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[SectionOverviewConfig] = None,
) -> SectionOverviewResult:
    config = config or SectionOverviewConfig()
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    working_blocks = deepcopy(blocks) if blocks is not None else extract_text_blocks(pdf_path)
    if not _blocks_have_section_metadata(working_blocks):
        annotate_blocks_with_sections(blocks=working_blocks, pdf_path=pdf_path, source_url=source_url)

    title = _resolve_overview_title(working_blocks, metadata=metadata, fallback=pdf_path.stem)
    section_runs = _collect_section_runs(working_blocks, config=config)
    section_items = [_summarize_section_run(run, config=config, document_title=title) for run in section_runs]
    section_items = [item for item in section_items if item is not None]

    result_metadata: Dict[str, str] = {}
    if source_url:
        result_metadata["source_url"] = str(source_url)
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            result_metadata[str(key)] = str(value)

    return SectionOverviewResult(
        source_pdf=pdf_path,
        title=title,
        sections=section_items,
        metadata=result_metadata,
    )


def render_section_overview_markdown(result: SectionOverviewResult) -> str:
    lines: List[str] = [f"# {result.title}", ""]
    for item in result.sections:
        heading = "#" * max(2, min(6, int(item.section_level) + 1))
        lines.append(f"{heading} {item.section_title}")
        lines.append("")
        lines.append(item.summary_paragraph)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _resolve_overview_title(
    blocks: Sequence[Dict[str, Any]],
    *,
    metadata: Optional[Dict[str, Any]],
    fallback: str,
) -> str:
    if isinstance(metadata, dict):
        title = str(metadata.get("title") or "").strip()
        if title:
            return title
    for block in blocks:
        text = _normalize_inline_whitespace(block.get("text") or "")
        if not text:
            continue
        canonical = _block_section_canonical(block)
        if canonical != "front_matter":
            continue
        if _looks_like_front_matter_title(text):
            return text
    return str(fallback or "Document").strip() or "Document"


def _blocks_have_section_metadata(blocks: Sequence[Dict[str, Any]]) -> bool:
    for block in blocks:
        metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
        if metadata.get("section_canonical") and metadata.get("section_title"):
            return True
    return False


def _looks_like_front_matter_title(text: str) -> bool:
    if not text or len(text) < 12 or len(text) > 220:
        return False
    lower = text.lower()
    if lower in {"abstract", "front matter"}:
        return False
    if _URL_RE.search(text) or _EMAIL_RE.search(text):
        return False
    if lower.startswith("figure ") or lower.startswith("table "):
        return False
    words = _WORD_RE.findall(text)
    return len(words) >= 3


def _collect_section_runs(
    blocks: Sequence[Dict[str, Any]],
    *,
    config: SectionOverviewConfig,
) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for block in blocks:
        if not isinstance(block, dict):
            continue
        canonical = _block_section_canonical(block)
        title = _block_section_title(block)
        level = _block_section_level(block)
        if not _should_include_section(canonical, config=config):
            current = None
            continue
        key = (canonical, title, level)
        if current and current["key"] == key:
            current["blocks"].append(block)
            current["page_end"] = max(current["page_end"], _safe_int(block.get("page_no"), current["page_end"]))
        else:
            current = {
                "key": key,
                "section_canonical": canonical,
                "section_title": title,
                "section_level": level,
                "blocks": [block],
                "page_start": _safe_int(block.get("page_no"), 0),
                "page_end": _safe_int(block.get("page_no"), 0),
            }
            runs.append(current)
    return runs


def _should_include_section(canonical: str, *, config: SectionOverviewConfig) -> bool:
    if canonical == "front_matter":
        return bool(config.include_front_matter)
    if canonical == "references":
        return bool(config.include_references)
    if canonical == "acknowledgements":
        return bool(config.include_acknowledgements)
    if canonical == "appendix":
        return bool(config.include_appendix)
    return canonical not in _DEFAULT_SKIP_CANONICALS


def _summarize_section_run(
    run: Dict[str, Any],
    *,
    config: SectionOverviewConfig,
    document_title: str,
) -> Optional[SectionOverviewItem]:
    blocks = list(run.get("blocks") or [])
    cleaned_blocks: List[str] = []
    caption_fallback_texts: List[str] = []
    section_title = str(run.get("section_title") or "")
    section_canonical = str(run.get("section_canonical") or "other")
    for block in blocks:
        cleaned = _clean_block_text_for_overview(block, section_title=section_title, document_title=document_title)
        if cleaned:
            cleaned_blocks.append(cleaned)
        caption_fallback = _caption_to_overview_text(str(block.get("text") or ""))
        if caption_fallback:
            caption_fallback_texts.append(caption_fallback)

    section_text = "\n\n".join(cleaned_blocks).strip()
    if not section_text and not caption_fallback_texts:
        return None

    sentences = _split_sentences(section_text) if section_text else []
    summary_sentences: List[str] = []
    if sentences:
        summary_sentences = _select_section_sentences(
            sentences,
            section_text=section_text,
            section_canonical=section_canonical,
            config=config,
        )
    if (
        (not summary_sentences or len(_WORD_RE.findall(" ".join(summary_sentences))) < max(24, config.min_words_per_section // 3))
        and _section_prefers_visual_caption_fallback(section_title=section_title, section_canonical=section_canonical)
        and caption_fallback_texts
    ):
        caption_summary = _build_caption_fallback_summary(
            caption_fallback_texts,
            section_canonical=section_canonical,
            config=config,
        )
        if caption_summary:
            summary_sentences = caption_summary
    if not summary_sentences:
        return None
    summary_paragraph = " ".join(summary_sentences).strip()
    summary_paragraph = _trim_leading_tableish_prefix(summary_paragraph)
    summary_paragraph = _trim_trailing_tableish_suffix(summary_paragraph)
    if not summary_paragraph:
        return None

    return SectionOverviewItem(
        section_title=section_title or "Document Body",
        section_canonical=section_canonical,
        section_level=_safe_int(run.get("section_level"), 1),
        page_start=_safe_int(run.get("page_start"), 0),
        page_end=_safe_int(run.get("page_end"), 0),
        block_count=len(blocks),
        word_count=len(_WORD_RE.findall(section_text)),
        summary_paragraph=summary_paragraph,
        source_sentences=summary_sentences,
    )


def _clean_block_text_for_overview(block: Dict[str, Any], *, section_title: str, document_title: str) -> str:
    raw_text = str(block.get("text") or "").strip()
    if not raw_text:
        return ""
    compact = " ".join(raw_text.split()).strip()
    if not compact:
        return ""
    if _looks_like_reference_entry_block(raw_text):
        return ""
    if compact.lower() == section_title.strip().lower():
        return ""
    if _looks_like_section_boundary_block(compact, section_title=section_title):
        return ""
    if _CAPTION_RE.match(compact):
        return ""
    if _PANEL_LABEL_RE.match(compact):
        return ""
    if (_URL_RE.search(compact) or _EMAIL_RE.search(compact)) and len(compact) <= 240:
        return ""
    if _looks_like_noise_line(compact):
        return ""

    normalized = _normalize_block_text(raw_text)
    normalized = _strip_section_title_prefix(normalized, section_title=section_title)
    normalized = _strip_document_title_prefix(normalized, document_title=document_title)
    normalized = _trim_leading_tableish_prefix(normalized)
    compact_normalized = " ".join(normalized.split()).strip()
    if not compact_normalized:
        return ""
    if _looks_like_noise_line(compact_normalized):
        return ""
    return normalized


def _normalize_block_text(raw_text: str) -> str:
    if "\n" not in raw_text:
        return _normalize_inline_whitespace(raw_text)

    paragraphs: List[str] = []
    for chunk in re.split(r"\n\s*\n", raw_text):
        lines = [_normalize_inline_whitespace(line) for line in chunk.splitlines() if _normalize_inline_whitespace(line)]
        if not lines:
            continue
        merged = lines[0]
        for next_line in lines[1:]:
            if _should_dehyphenate_line_break(merged, next_line):
                merged = merged.rstrip()[:-1] + next_line.lstrip()
            else:
                merged = f"{merged.rstrip()} {next_line.lstrip()}".strip()
        paragraphs.append(merged.strip())
    return "\n\n".join(paragraphs).strip()


def _normalize_inline_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _should_dehyphenate_line_break(current: str, next_line: str) -> bool:
    current = str(current or "").rstrip()
    next_line = str(next_line or "").lstrip()
    if len(current) < 2 or not next_line:
        return False
    if current[-1] not in "-‐‑‒–":
        return False
    if not current[-2].isalpha():
        return False
    return next_line[0].isalpha() and next_line[0].islower()


def _looks_like_sentenceish_prose(text: str) -> bool:
    words = _WORD_RE.findall(text)
    if len(words) < 6:
        return False
    if re.search(r"[.!?;:]", text):
        return True
    return len(words) >= 10


def _looks_like_noise_line(text: str) -> bool:
    compact = " ".join(str(text or "").split()).strip()
    if not compact:
        return True
    if _PAGE_NUMBER_RE.match(compact):
        return True
    if _ARXIV_RE.search(compact):
        return True
    if _SECTION_NUMBER_RE.match(compact) and len(compact) <= 20:
        return True
    if compact.startswith(("Correspondence:", "Corresponding author", "Preprint.", "Acknowledgment.", "Acknowledgements.", "Acknowledgment:", "Acknowledgements:")):
        return True
    if _CODE_LINE_RE.search(compact) and not _looks_like_sentenceish_prose(compact):
        return True
    if _looks_like_table_scaffold_text(compact):
        return True
    if _looks_like_visual_label_text(compact):
        return True
    if len(compact) <= 36 and _SINGLE_LABEL_RE.match(compact) and not _looks_like_sentenceish_prose(compact):
        return True
    if _EQUATION_NOISE_RE.search(compact) and not _looks_like_sentenceish_prose(compact):
        return True
    if _FIGURE_TABLE_REF_RE.search(compact) and len(compact) <= 48 and not _looks_like_sentenceish_prose(compact):
        return True
    return False


def _split_sentences(text: str) -> List[str]:
    protected = _protect_sentence_boundaries(text)
    sentences = [
        _restore_sentence_boundaries(_normalize_inline_whitespace(segment))
        for segment in _SENTENCE_SPLIT_RE.split(protected)
        if _normalize_inline_whitespace(segment)
    ]
    merged: List[str] = []
    for sentence in sentences:
        if merged and len(sentence) <= 24 and not sentence.endswith((".", "!", "?")):
            merged[-1] = f"{merged[-1]} {sentence}".strip()
        else:
            merged.append(sentence)
    return [sentence for sentence in merged if _looks_usable_sentence(sentence)]


def _looks_usable_sentence(sentence: str) -> bool:
    if len(sentence) < 35:
        return False
    if len(sentence) > 420:
        return False
    if _looks_like_sentence_fragment(sentence):
        return False
    if len(_WORD_RE.findall(sentence)) < 6:
        return False
    if _BULLET_GLYPH_RE.search(sentence):
        return False
    if _CAPTION_RE.match(sentence):
        return False
    if _looks_like_table_scaffold_text(sentence):
        return False
    if _looks_like_visual_label_text(sentence):
        return False
    if _looks_like_reference_entry_block(sentence):
        return False
    if _looks_like_bibliography_sentence(sentence):
        return False
    if _CODE_LINE_RE.search(sentence):
        return False
    if sentence.lower().startswith("where ") and re.search(r"[=()∥×+\-/*^_]", sentence):
        return False
    return True


def _select_section_sentences(
    sentences: Sequence[str],
    *,
    section_text: str,
    section_canonical: str,
    config: SectionOverviewConfig,
) -> List[str]:
    section_words = len(_WORD_RE.findall(section_text))
    target_sentences = _target_sentence_count(section_words, config=config)
    target_words = _target_word_budget(section_words, config=config)
    tf = Counter(_sentence_tokens(section_text))

    scored: List[Tuple[float, int, str, List[str]]] = []
    total_sentences = max(1, len(sentences))
    for idx, sentence in enumerate(sentences):
        if len(sentence) > config.max_sentence_chars:
            continue
        tokens = _sentence_tokens(sentence)
        if not tokens:
            continue
        score = _sentence_score(
            sentence,
            tokens=tokens,
            idx=idx,
            total_sentences=total_sentences,
            section_canonical=section_canonical,
            term_frequencies=tf,
        )
        scored.append((score, idx, sentence, tokens))

    if not scored:
        fallback = _normalize_inline_whitespace(section_text)
        return [_trim_to_word_budget(fallback, target_words)] if fallback else []

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected: List[Tuple[int, str, List[str]]] = []
    selected_word_count = 0
    hard_sentence_limit = max(target_sentences, int(config.max_sentences_per_section))
    related_work_mode = section_canonical == "related_work"
    for score, idx, sentence, tokens in scored:
        if len(selected) >= hard_sentence_limit:
            break
        if related_work_mode and _looks_like_related_work_result_sentence(sentence) and not _looks_like_prior_work_sentence(sentence):
            if any(not _looks_like_related_work_result_sentence(other_sentence) for _, _, other_sentence, _ in scored):
                continue
        if any(_token_similarity(tokens, existing_tokens) >= config.sentence_similarity_threshold for _, _, existing_tokens in selected):
            continue
        candidate_words = len(_WORD_RE.findall(sentence))
        if selected and selected_word_count + candidate_words > config.max_words_per_section:
            continue
        selected.append((idx, sentence, tokens))
        selected_word_count += candidate_words
        if len(selected) >= target_sentences and selected_word_count >= target_words:
            break

    if not selected:
        selected.append((scored[0][1], scored[0][2], scored[0][3]))

    selected.sort(key=lambda item: item[0])
    summary = " ".join(sentence for _, sentence, _ in selected).strip()
    if len(_WORD_RE.findall(summary)) > config.max_words_per_section:
        summary = _trim_to_word_budget(summary, config.max_words_per_section)
    return [summary] if len(selected) == 1 else [sentence for _, sentence, _ in selected]


def _target_sentence_count(section_words: int, *, config: SectionOverviewConfig) -> int:
    if section_words < 140:
        target = 1
    elif section_words < 420:
        target = 2
    elif section_words < 1050:
        target = 3
    else:
        target = 4
    return max(config.min_sentences_per_section, min(config.max_sentences_per_section, target))


def _target_word_budget(section_words: int, *, config: SectionOverviewConfig) -> int:
    scaled = int(round(section_words * 0.16))
    return max(config.min_words_per_section, min(config.max_words_per_section, scaled))


def _sentence_tokens(text: str) -> List[str]:
    tokens = [token.lower() for token in _WORD_RE.findall(text)]
    return [token for token in tokens if len(token) > 2 and token not in _STOPWORDS]


def _sentence_score(
    sentence: str,
    *,
    tokens: Sequence[str],
    idx: int,
    total_sentences: int,
    section_canonical: str,
    term_frequencies: Counter[str],
) -> float:
    unique_tokens = list(dict.fromkeys(tokens))
    coverage = sum(math.log1p(term_frequencies.get(token, 0)) for token in unique_tokens) / max(1.0, len(unique_tokens))
    position_score = max(0.0, 1.2 - (idx / max(1, total_sentences - 1)) * 0.9)
    cue_terms = _CUE_TERMS.get(section_canonical, ())
    cue_hits = sum(1 for term in cue_terms if term in sentence.lower())
    explanatory = 0.6 if _EXPLANATORY_VERB_RE.search(sentence) else 0.0
    score = (coverage * 1.2) + position_score + (cue_hits * 0.7) + explanatory
    lower_sentence = sentence.lower()

    if idx == 0 and section_canonical in {"abstract", "introduction", "related_work", "conclusion"}:
        score += 0.9
    if idx <= 1 and section_canonical in {"methodology", "experiments", "results"}:
        score += 0.4
    if sentence.endswith((".", "!", "?")):
        score += 0.25
    if _PROMPT_LIKE_RE.match(sentence):
        score -= 0.5
    if section_canonical == "related_work":
        if re.search(r"\b(?:prior|previous|existing|recent|related|literature|approach|approaches|method|methods|work|works)\b", lower_sentence):
            score += 0.8
        if re.search(r"\b(?:we|our|results?|ablate|ablation|benchmark|improve|improves|gain|gains|outperform)\b", lower_sentence):
            score -= 1.0

    citation_penalty = len(_CITATION_RE.findall(sentence)) * 0.4
    figure_ref_penalty = 0.8 if _FIGURE_TABLE_REF_RE.search(sentence) and not _looks_like_sentenceish_prose(sentence) else 0.0
    url_penalty = 1.0 if _URL_RE.search(sentence) else 0.0
    bib_penalty = 1.3 if _looks_like_bibliography_sentence(sentence) else 0.0
    code_penalty = 1.5 if _CODE_LINE_RE.search(sentence) else 0.0
    scaffold_penalty = 1.2 if _looks_like_table_scaffold_text(sentence) else 0.0
    label_penalty = 1.2 if _BULLET_GLYPH_RE.search(sentence) else 0.0
    short_penalty = 0.7 if len(sentence) < 55 else 0.0
    long_penalty = 0.5 if len(sentence) > 280 else 0.0
    digit_penalty = 0.5 if sum(ch.isdigit() for ch in sentence) >= 12 and not _looks_like_sentenceish_prose(sentence) else 0.0
    return score - citation_penalty - figure_ref_penalty - url_penalty - bib_penalty - code_penalty - scaffold_penalty - label_penalty - short_penalty - long_penalty - digit_penalty


def _protect_sentence_boundaries(text: str) -> str:
    protected = str(text or "")
    for needle, sentinel in _ABBREVIATION_SENTINELS:
        protected = protected.replace(needle, sentinel)
    return protected


def _restore_sentence_boundaries(text: str) -> str:
    restored = str(text or "")
    for needle, sentinel in _ABBREVIATION_SENTINELS:
        restored = restored.replace(sentinel, needle)
    return restored


def _token_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def _trim_to_word_budget(text: str, max_words: int) -> str:
    words = str(text or "").split()
    if len(words) <= max_words:
        return str(text or "").strip()
    trimmed = " ".join(words[:max_words]).rstrip(",;:")
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "..."
    return trimmed


def _block_section_canonical(block: Dict[str, Any]) -> str:
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    return str(metadata.get("section_canonical") or "other").strip() or "other"


def _block_section_title(block: Dict[str, Any]) -> str:
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    return str(metadata.get("section_title") or "Document Body").strip() or "Document Body"


def _block_section_level(block: Dict[str, Any]) -> int:
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    return _safe_int(metadata.get("section_level"), 1)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _strip_section_title_prefix(text: str, *, section_title: str) -> str:
    compact = _normalize_inline_whitespace(text)
    title = _normalize_inline_whitespace(section_title)
    if not compact or not title:
        return compact
    lowered = compact.lower()
    title_lower = title.lower()
    if lowered == title_lower:
        return ""
    if lowered.startswith(title_lower + " "):
        return compact[len(title):].lstrip(" .:-")
    return compact


def _strip_document_title_prefix(text: str, *, document_title: str) -> str:
    compact = _normalize_inline_whitespace(text)
    title = _normalize_inline_whitespace(document_title)
    if not compact or not title:
        return compact
    lowered = compact.lower()
    title_lower = title.lower()
    if lowered.startswith(title_lower + " "):
        return compact[len(title):].lstrip(" .:-")
    return compact


def _looks_like_section_boundary_block(text: str, *, section_title: str) -> bool:
    compact = _normalize_inline_whitespace(text)
    title = _normalize_inline_whitespace(section_title)
    if not compact:
        return False
    if compact.lower() == title.lower():
        return True
    if _SECTION_NUMBER_RE.match(compact) and compact.lower().endswith(title.lower()):
        return True
    return False


def _looks_like_reference_entry_block(text: str) -> bool:
    compact = _normalize_inline_whitespace(text)
    if not compact:
        return False
    if compact.lower() in {"references", "acknowledgment", "acknowledgements", "acknowledgment.", "acknowledgements."}:
        return True
    if _REFERENCE_ENTRY_RE.match(compact):
        return True
    return False


def _looks_like_table_scaffold_text(text: str) -> bool:
    compact = _normalize_inline_whitespace(text)
    if not compact:
        return False
    if _CAPTION_RE.match(compact):
        return False
    if compact.startswith("Figure "):
        return False
    if len(compact) >= 220 and _looks_like_sentenceish_prose(compact):
        return False
    short_tokens = sum(1 for token in re.findall(r"\S+", compact) if len(token) <= 4)
    numberish_tokens = sum(1 for token in re.findall(r"\S+", compact) if re.search(r"\d|[%↑↓×†⋆✓–—-]", token))
    metric_markers = len(re.findall(r"(FID|IS|rFID|gFID|RMSD|Match|Valid|Unique|Params|Aux\.?|Adv\.?|Token|Method)", compact))
    tabular_header_tokens = sum(1 for token in re.findall(r"\S+", compact) if _TABLE_HEADER_TOKEN_RE.match(token))
    lowered = compact.lower()
    if _looks_like_reference_entry_block(compact):
        return True
    if re.search(r"[✓✗⋆†‡]", compact) and (numberish_tokens >= 1 or short_tokens >= 4):
        return True
    if metric_markers >= 2 and (numberish_tokens >= 2 or short_tokens >= 6):
        return True
    if (
        tabular_header_tokens >= 6
        and numberish_tokens >= 1
        and not _EXPLANATORY_VERB_RE.search(compact)
        and not _looks_like_sentenceish_prose(compact)
    ):
        return True
    if not _looks_like_sentenceish_prose(compact) and numberish_tokens >= 3:
        return True
    if numberish_tokens >= 4 and short_tokens >= 6 and not _looks_like_sentenceish_prose(compact):
        return True
    if (
        len(compact) <= 160
        and not _looks_like_sentenceish_prose(compact)
        and re.search(r"\b(with|without|single-stage|two-stage|frameworks?|tokenizer|encoder|decoder|baseline)\b", lowered)
        and (numberish_tokens >= 1 or short_tokens >= 4)
    ):
        return True
    return False


def _looks_like_visual_label_text(text: str) -> bool:
    compact = _normalize_inline_whitespace(text)
    if not compact:
        return False
    if _looks_like_sentenceish_prose(compact):
        return False
    if len(compact) > 80:
        return False
    if _PANEL_LABEL_RE.match(compact):
        return True
    if re.match(r"^(?:[A-Za-z0-9][A-Za-z0-9 /+_-]{0,30}|[A-Za-z]+(?:\s+[A-Za-z]+){0,4})$", compact):
        return True
    return False


def _looks_like_bibliography_sentence(text: str) -> bool:
    compact = _normalize_inline_whitespace(text)
    if not compact:
        return False
    citation_count = len(_CITATION_RE.findall(compact))
    if citation_count >= 2 and not (_EXPLANATORY_VERB_RE.search(compact) and _looks_like_sentenceish_prose(compact)):
        return True
    if (
        _BIBLIOGRAPHY_SIGNAL_RE.search(compact)
        and len(_WORD_RE.findall(compact)) >= 6
        and not (_EXPLANATORY_VERB_RE.search(compact) and _looks_like_sentenceish_prose(compact))
    ):
        return True
    return False


def _looks_like_sentence_fragment(sentence: str) -> bool:
    compact = _normalize_inline_whitespace(sentence)
    if not compact:
        return True
    stripped = compact.lstrip("\"'([{")
    if not stripped:
        return True
    if stripped[0].islower():
        return True
    if stripped[0] in ",;:":
        return True
    lowered = stripped.lower()
    if re.match(r"^(?:and|but|or|while|which|that|with|without|using|showing|improving|indicating|suggesting|retrieval)\b", lowered):
        return True
    if _looks_like_reference_entry_block(compact):
        return True
    return False


def _section_prefers_visual_caption_fallback(*, section_title: str, section_canonical: str) -> bool:
    if section_canonical == "appendix":
        return True
    return bool(_SECTION_TITLE_HINT_RE.search(_normalize_inline_whitespace(section_title)))


def _caption_to_overview_text(text: str) -> str:
    compact = _normalize_inline_whitespace(text)
    if not compact or not _CAPTION_RE.match(compact):
        return ""
    compact = re.sub(r"^\s*(?:figure|fig\.?|table|tab\.?|algorithm)\s+\d+[A-Za-z]?[.:]?\s*", "", compact, flags=re.I)
    compact = compact.lstrip(" .:-").strip()
    if not compact:
        return ""
    if compact.lower().startswith("we show additional qualitative results"):
        return ""
    if len(_WORD_RE.findall(compact)) < 8:
        return ""
    return compact


def _build_caption_fallback_summary(
    captions: Sequence[str],
    *,
    section_canonical: str,
    config: SectionOverviewConfig,
) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for caption in captions:
        compact = _normalize_inline_whitespace(caption)
        if not compact:
            continue
        key = compact.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(compact)
    if not cleaned:
        return []
    return _select_section_sentences(
        cleaned,
        section_text=" ".join(cleaned),
        section_canonical=section_canonical,
        config=config,
    )


def _trim_leading_tableish_prefix(text: str) -> str:
    compact = _normalize_inline_whitespace(text)
    if not compact:
        return ""
    tokens = compact.split()
    if len(tokens) < 10:
        return compact
    upper_bound = min(len(tokens) - 6, 20)
    for idx in range(3, max(4, upper_bound)):
        prefix = " ".join(tokens[:idx])
        suffix = " ".join(tokens[idx:])
        if not suffix or not suffix[0].isupper():
            continue
        first_token = suffix.split()[0]
        if _TABLE_HEADER_TOKEN_RE.match(first_token) and first_token.lower() not in {"the", "our", "we", "this", "these", "in", "a", "an"}:
            continue
        if not _looks_like_sentenceish_prose(suffix):
            continue
        if not _EXPLANATORY_VERB_RE.search(suffix):
            continue
        prefix_short = sum(1 for token in prefix.split() if len(token) <= 4)
        prefix_numberish = sum(1 for token in prefix.split() if re.search(r"\d|[%↑↓×†⋆✓–—-]", token))
        prefix_headerish = sum(1 for token in prefix.split() if _TABLE_HEADER_TOKEN_RE.match(token))
        if (
            _looks_like_table_scaffold_text(prefix)
            or (prefix_numberish >= 4 and prefix_short >= 4)
            or (prefix_headerish >= 5 and prefix_numberish >= 2)
        ):
            return suffix
    return compact


def _trim_trailing_tableish_suffix(text: str) -> str:
    compact = _normalize_inline_whitespace(text)
    if not compact:
        return ""
    tokens = compact.split()
    if len(tokens) < 10:
        return compact
    lower_bound = max(4, len(tokens) - 20)
    for idx in range(lower_bound, len(tokens) - 2):
        prefix = " ".join(tokens[:idx])
        suffix = " ".join(tokens[idx:])
        if not prefix or not suffix:
            continue
        first_token = suffix.split()[0]
        if first_token.lower() in {"and", "or", "while", "with", "our", "this", "the", "these", "those", "in", "for"}:
            continue
        if not (_TABLE_HEADER_TOKEN_RE.match(first_token) or re.search(r"\d", first_token)):
            continue
        if not _looks_like_sentenceish_prose(prefix):
            continue
        if not _EXPLANATORY_VERB_RE.search(prefix):
            continue
        suffix_short = sum(1 for token in suffix.split() if len(token) <= 4)
        suffix_numberish = sum(1 for token in suffix.split() if re.search(r"\d|[%↑↓×†⋆✓–—-]", token))
        suffix_headerish = sum(1 for token in suffix.split() if _TABLE_HEADER_TOKEN_RE.match(token))
        if (
            _looks_like_table_scaffold_text(suffix)
            or (suffix_numberish >= 4 and suffix_short >= 4)
            or (suffix_headerish >= 5 and suffix_numberish >= 2)
        ):
            return prefix.rstrip(" ,;:")
    return compact


def _looks_like_prior_work_sentence(sentence: str) -> bool:
    return bool(
        re.search(
            r"\b(?:prior|previous|existing|recent|related|literature|baseline|baselines|concurrent|earlier)\b",
            sentence.lower(),
        )
    )


def _looks_like_related_work_result_sentence(sentence: str) -> bool:
    return bool(
        re.search(
            r"\b(?:our|we|results?|ablate|ablation|benchmark|improve|improves|gain|gains|outperform)\b",
            sentence.lower(),
        )
    )
