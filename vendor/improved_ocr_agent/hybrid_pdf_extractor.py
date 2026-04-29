import os
import re
import json
import shutil
import fitz  # PyMuPDF
import pdfplumber
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ia_phase1_bridge import IaPhase1Bridge
from .non_ocr_routing import DocumentContentProfile, build_document_content_profile

class OCRBackendBase:
    """
    Base OCR backend interface.
    """

    def ocr_page_image(self, image_path: str, page_num: int, page_hint: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    def ocr_pdf_page(self, pdf_path: str, page_num: int, page_hint: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError


class PipelineCustomOCRBackend(OCRBackendBase):
    """
    Adapter for BatchAgent.ocr_agent.pipeline_custom.

    It supports:
    - OCRing a saved page image
    - OCRing a PDF page directly

    Example:
        from BatchAgent.ocr_agent.pipeline_custom import make_ocr_args
        args = make_ocr_args(
            server="http://localhost:8002/v1",
            model="allenai/olmOCR-2-7B-1025-FP8",
            workspace="./tmp_ocr",
        )
        backend = PipelineCustomOCRBackend(args)
    """

    def __init__(self, ocr_args):
        self.ocr_args = ocr_args

    def ocr_page_image(self, image_path: str, page_num: int, page_hint: Optional[Dict[str, Any]] = None) -> str:
        from .pipeline_custom import ocr_page_image_sync

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        result = ocr_page_image_sync(
            self.ocr_args,
            image_bytes=image_bytes,
            page_num=page_num,
        )
        return result.response.natural_text or ""

    def ocr_pdf_page(self, pdf_path: str, page_num: int, page_hint: Optional[Dict[str, Any]] = None) -> str:
        from .pipeline_custom import ocr_page_pdf_sync

        result = ocr_page_pdf_sync(
            self.ocr_args,
            pdf_path=pdf_path,
            page_num=page_num,
        )
        return result.response.natural_text or ""


class DummyOCRBackend(OCRBackendBase):
    """
    Fallback OCR backend for testing.
    """

    def ocr_page_image(self, image_path: str, page_num: int, page_hint: Optional[Dict[str, Any]] = None) -> str:
        image_name = os.path.basename(image_path)
        return f"![OCR Page {page_num}]({image_name})"

    def ocr_pdf_page(self, pdf_path: str, page_num: int, page_hint: Optional[Dict[str, Any]] = None) -> str:
        return f"<!-- OCR backend not configured for page {page_num} -->"



class HybridPDFExtractor:
    """
    Adaptive PDF -> Markdown extractor with 3 modes:
      - simple_text
      - hybrid_paper
      - ocr

    Strategy:
      1. Analyze document/page properties
      2. Route each page to one of:
         - simple text parsing
         - hybrid extraction
         - OCR
      3. Save figures/tables as assets and reference them in Markdown

    Output:
      - <pdf_name>.md
      - <pdf_name>_assets/
          - figures/
          - tables/
          - page_images/
    """

    def __init__(
        self,
        pdf_path: str,
        ocr_backend: Optional[OCRBackendBase] = None,
        image_dpi: int = 180,
        ocr_dpi: int = 180,
        use_pdf_page_ocr: bool = False,
    ):
        self.pdf_path = os.path.abspath(pdf_path)
        self.output_dir = os.path.dirname(self.pdf_path)
        self.pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]

        self.assets_dir = os.path.join(self.output_dir, f"{self.pdf_name}_assets")
        self.figure_dir = os.path.join(self.assets_dir, "figures")
        self.table_dir = os.path.join(self.assets_dir, "tables")
        self.equation_dir = os.path.join(self.assets_dir, "equations")
        self.page_image_dir = os.path.join(self.assets_dir, "page_images")

        os.makedirs(self.assets_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)
        os.makedirs(self.equation_dir, exist_ok=True)
        os.makedirs(self.page_image_dir, exist_ok=True)

        self.ocr_backend = ocr_backend or DummyOCRBackend()
        self.image_dpi = image_dpi
        self.ocr_dpi = ocr_dpi
        self.use_pdf_page_ocr = use_pdf_page_ocr
        self.last_routing_info: Dict[str, Any] = {}

    def _reset_asset_dirs(self) -> None:
        for directory in (self.figure_dir, self.table_dir, self.equation_dir, self.page_image_dir):
            os.makedirs(directory, exist_ok=True)
            for name in os.listdir(directory):
                path = os.path.join(directory, name)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except FileNotFoundError:
                    continue

    # -------------------------------------------------------------------------
    # Basic helpers
    # -------------------------------------------------------------------------

    def _rect_center(self, rect: fitz.Rect) -> fitz.Point:
        return fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)


    def _overlaps_any(self, rect: fitz.Rect, rects: List[fitz.Rect], pad: float = 0) -> bool:
        if pad != 0:
            rect = rect + (-pad, -pad, pad, pad)
        for r in rects:
            if rect.intersects(r):
                return True
        return False


    def _bbox_sort_key(self, bbox, page_width: float):
        x0, y0, x1, y1 = bbox
        cx = (x0 + x1) / 2
        col = 0 if cx < page_width * 0.5 else 1
        return (col, y0, x0)


    def _expand_rect_with_text(self, page_fitz: fitz.Page, rect: fitz.Rect, pad: int = 12) -> fitz.Rect:
        """
        Expand a rectangle and snap nearby small text blocks into it.
        Useful for figures/charts/tables so titles, legends, axes, labels aren't cut off.
        """
        expanded = (rect + (-pad, -pad, pad, pad)).intersect(page_fitz.rect)
        text_blocks = [b for b in page_fitz.get_text("dict")["blocks"] if b["type"] == 0]

        changed = True
        while changed:
            changed = False
            for b in text_blocks:
                b_rect = fitz.Rect(b["bbox"])

                if (expanded + (-8, -8, 8, 8)).intersects(b_rect):
                    if b_rect.height < 120 and b_rect.width < page_fitz.rect.width * 0.8:
                        if not expanded.contains(b_rect):
                            expanded = (expanded | b_rect).intersect(page_fitz.rect)
                            changed = True

        return expanded


    def _extract_block_plain_text(self, block: dict) -> str:
        parts = []
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            line_text = self._normalize_ws(line_text)
            if line_text:
                parts.append(line_text)
        return " ".join(parts).strip()


    def _join_caption_blocks(self, caption_blocks: List[dict]) -> str:
        if not caption_blocks:
            return ""

        caption_blocks = sorted(caption_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
        texts = [self._extract_block_plain_text(b) for b in caption_blocks]
        texts = [t for t in texts if t]
        return "\n".join(texts).strip()

    def _find_table_caption_blocks(self, text_blocks: List[dict]) -> List[dict]:
        table_caps = []
        for block in text_blocks:
            text = self._extract_block_plain_text(block)
            if re.match(r"^table\s+\d+\b", text.strip(), flags=re.IGNORECASE):
                table_caps.append(block)
        return table_caps

    def _expand_table_from_caption(
        self,
        page_fitz: fitz.Page,
        caption_block: dict,
        text_blocks: List[dict],
        max_expand_down: float = 500,
    ) -> fitz.Rect:
        """
        Starting from a 'Table N ...' caption block, expand downward to absorb
        the table body beneath it.
        """
        caption_rect = fitz.Rect(caption_block["bbox"])
        expanded = caption_rect

        for block in text_blocks:
            b_rect = fitz.Rect(block["bbox"])
            text = self._extract_block_plain_text(block)
            if not text:
                continue

            # Only consider blocks below caption and roughly aligned horizontally
            if b_rect.y0 < caption_rect.y1:
                continue
            if b_rect.y0 - caption_rect.y1 > max_expand_down:
                continue

            horizontally_aligned = (
                b_rect.x1 >= caption_rect.x0 - 80 and
                b_rect.x0 <= caption_rect.x1 + 300
            )
            if not horizontally_aligned:
                continue

            # absorb moderate-sized text/table rows
            if b_rect.height < 80:
                expanded = expanded | b_rect

        return expanded.intersect(page_fitz.rect)

    def _detect_composite_figure_regions(self, page_fitz: fitz.Page, figure_rects: List[fitz.Rect]) -> List[fitz.Rect]:
        """
        Merge many nearby figure/chart regions into a single composite region
        for pages like benchmark grids or multi-panel figures.
        """
        if not figure_rects:
            return []

        merged = self._merge_close_rects(figure_rects, x_gap=50, y_gap=50)

        composite = []
        for rect in merged:
            area_ratio = rect.get_area() / max(page_fitz.rect.get_area(), 1.0)

            # Large multi-panel region
            if area_ratio > 0.18:
                composite.append(rect)
                continue

            # Wide region that likely contains several subplots
            if rect.width > page_fitz.rect.width * 0.55 and rect.height > page_fitz.rect.height * 0.22:
                composite.append(rect)

        return composite

    def _merge_close_rects(self, rects: List[fitz.Rect], x_gap: float = 30, y_gap: float = 30) -> List[fitz.Rect]:
        rects = [r for r in rects if not r.is_empty]
        changed = True

        while changed:
            changed = False
            merged = []
            while rects:
                current = rects.pop(0)
                current_expanded = current + (-x_gap, -y_gap, x_gap, y_gap)

                merged_this_round = False
                for i, other in enumerate(merged):
                    other_expanded = other + (-x_gap, -y_gap, x_gap, y_gap)
                    if current_expanded.intersects(other_expanded):
                        merged[i] = other | current
                        changed = True
                        merged_this_round = True
                        break

                if not merged_this_round:
                    merged.append(current)

            rects = merged

        return rects

    def _find_caption_blocks_for_rect(
        self,
        page_fitz: fitz.Page,
        target_rect: fitz.Rect,
        text_blocks: List[dict],
        used_block_ids: set,
        kind: str = "figure",
    ) -> List[dict]:
        """
        Find caption/title/note blocks near a figure/table rectangle.

        kind:
        - "figure"
        - "table"
        """
        candidates = []

        for idx, block in enumerate(text_blocks):
            if idx in used_block_ids:
                continue

            b_rect = fitz.Rect(block["bbox"])
            text = self._extract_block_plain_text(block)
            if not text:
                continue

            if b_rect.height > 120 or b_rect.width > page_fitz.rect.width * 0.9:
                continue

            horizontally_aligned = (
                b_rect.x1 >= target_rect.x0 - 40 and
                b_rect.x0 <= target_rect.x1 + 40
            )

            if not horizontally_aligned:
                continue

            above_gap = target_rect.y0 - b_rect.y1
            below_gap = b_rect.y0 - target_rect.y1

            is_near_above = 0 <= above_gap <= 60
            is_near_below = 0 <= below_gap <= 80

            if not (is_near_above or is_near_below):
                continue

            text_lower = text.lower()
            caption_score = 0

            if kind == "figure":
                if re.match(r"^(figure|fig\.?)\s*\d+", text_lower):
                    caption_score += 5
                if is_near_below:
                    caption_score += 2
            elif kind == "table":
                if re.match(r"^table\s*\d+", text_lower):
                    caption_score += 5
                if is_near_above:
                    caption_score += 2

            if re.match(r"^(source|note|notes)\b", text_lower):
                caption_score += 2

            if b_rect.height < 40:
                caption_score += 1

            if len(text) < 250:
                caption_score += 1

            if caption_score >= 3:
                candidates.append((caption_score, idx, block))

        def _dist(item):
            _, _, block = item
            b_rect = fitz.Rect(block["bbox"])
            if b_rect.y1 <= target_rect.y0:
                return abs(target_rect.y0 - b_rect.y1)
            return abs(b_rect.y0 - target_rect.y1)

        candidates.sort(key=lambda x: (-x[0], _dist(x)))
        return [b for _, _, b in candidates[:3]]


    def _looks_like_real_table(self, page_fitz: fitz.Page, table) -> bool:
        """
        Heuristic filter to distinguish real tables from charts/plots
        that pdfplumber may mis-detect as tables.
        """
        rect = fitz.Rect(table.bbox)
        extracted = table.extract()

        if not extracted or not extracted[0]:
            return False

        rows = len(extracted)
        cols = max(len(r) for r in extracted if r) if extracted else 0

        if cols > 12:
            return False

        total_cells = 0
        empty_cells = 0
        numeric_like = 0
        text_like = 0

        for row in extracted:
            for cell in row:
                total_cells += 1
                cell_str = str(cell).strip() if cell is not None else ""
                if not cell_str:
                    empty_cells += 1
                else:
                    if re.fullmatch(r"[\d\.\+\-%]+", cell_str):
                        numeric_like += 1
                    elif re.search(r"[A-Za-z]", cell_str):
                        text_like += 1

        if total_cells == 0:
            return False

        empty_ratio = empty_cells / total_cells
        if empty_ratio > 0.5:
            return False

        non_empty = total_cells - empty_cells
        if non_empty > 0 and text_like / non_empty < 0.1 and numeric_like / non_empty > 0.8:
            return False

        drawing_count = 0
        for d in page_fitz.get_drawings():
            drect = d["rect"]
            if rect.intersects(drect):
                drawing_count += 1

        if drawing_count > 30 and cols > 6:
            return False

        width = rect.width
        height = rect.height

        if width > page_fitz.rect.width * 0.6 and height > page_fitz.rect.height * 0.25 and cols > 8:
            return False

        if rows >= 2 and cols >= 2 and text_like > 0:
            return True

        return True


    def _normalize_ws(self, text: str) -> str:
        return re.sub(r"[ \t]+", " ", text).strip()

    def _relpath_from_output(self, path: str) -> str:
        return os.path.relpath(path, self.output_dir).replace("\\", "/")

    def _get_base_font_size(self, doc: fitz.Document) -> float:
        sizes = []
        for page in doc:
            page_dict = page.get_text("dict")
            for block in page_dict["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            sizes.append(span["size"])
        return statistics.mode([round(s) for s in sizes]) if sizes else 11.0

    def _get_header_prefix(self, span_size: float, base_size: float) -> str:
        ratio = span_size / base_size
        if ratio >= 1.7:
            return "# "
        if ratio >= 1.3:
            return "## "
        if ratio >= 1.1:
            return "### "
        return ""

    def _looks_like_garbled_text(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return True

        weird_ctrl = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\t\r")
        weird_ratio = weird_ctrl / max(len(text), 1)

        replacement_like = text.count("�") / max(len(text), 1)
        alpha_num = sum(1 for ch in text if ch.isalnum())
        alpha_ratio = alpha_num / max(len(text), 1)

        return weird_ratio > 0.02 or replacement_like > 0.01 or alpha_ratio < 0.10

    def _is_heading_like_line(self, text: str, max_size: float, base_size: float, bold_ratio: float = 0.0) -> bool:
        t = text.strip()
        if not t:
            return False

        # common unnumbered section titles
        common_titles = {
            "abstract", "introduction", "background", "related work", "method", "methods",
            "methodology", "approach", "experiments", "results", "discussion",
            "conclusion", "conclusions", "limitations", "references", "appendix",
            "acknowledgments", "acknowledgements"
        }

        t_clean = t.rstrip(".:").strip().lower()

        if t_clean in common_titles:
            return True

        # numbered section titles
        if re.match(r"^(\d+(\.\d+)*|[A-Z]|[IVXLC]+)[\.\)]?\s+[A-Z]", t):
            return True

        # short single-line bold/large text often means a heading
        word_count = len(t.split())
        if word_count <= 10:
            if max_size >= base_size * 1.15:
                return True
            if bold_ratio > 0.6:
                return True

        return False
    
    def _normalize_block_text(self, lines: List[dict], base_size: float) -> str:
        """
        Merge hard-wrapped lines inside a text block while preserving likely headings.
        """
        normalized_lines = []

        for line in lines:
            spans = line.get("spans", [])
            line_text = ""
            max_size = 0.0
            bold_chars = 0
            total_chars = 0

            for span in spans:
                txt = span.get("text", "")
                line_text += txt
                if txt.strip():
                    max_size = max(max_size, span.get("size", 0.0))
                    total_chars += len(txt.strip())
                    font_name = str(span.get("font", "")).lower()
                    if "bold" in font_name:
                        bold_chars += len(txt.strip())

            line_text = re.sub(r"\s+", " ", line_text).strip()
            if not line_text:
                continue

            bold_ratio = (bold_chars / total_chars) if total_chars > 0 else 0.0
            normalized_lines.append({
                "text": line_text,
                "max_size": max_size,
                "bold_ratio": bold_ratio,
                "is_heading": self._is_heading_like_line(line_text, max_size, base_size, bold_ratio),
            })

        if not normalized_lines:
            return ""

        # If this block is basically a heading block, keep line boundaries
        if len(normalized_lines) == 1 and normalized_lines[0]["is_heading"]:
            return normalized_lines[0]["text"]

        merged_parts = []
        current_para = ""

        for i, item in enumerate(normalized_lines):
            text = item["text"]

            if item["is_heading"]:
                if current_para.strip():
                    merged_parts.append(current_para.strip())
                    current_para = ""
                merged_parts.append(text)
                continue

            # start paragraph
            if not current_para:
                current_para = text
                continue

            prev = current_para.rstrip()

            # Hyphenated line break: "train-\ning" -> "training"
            if prev.endswith("-") and text:
                current_para = prev[:-1] + text
                continue

            # If previous line seems sentence-continuing, merge with space
            if prev and not re.search(r"[.:!?]$", prev):
                current_para += " " + text
            else:
                # new paragraph if previous line seems completed
                merged_parts.append(current_para.strip())
                current_para = text

        if current_para.strip():
            merged_parts.append(current_para.strip())

        return "\n\n".join(p for p in merged_parts if p.strip())

    def _extract_line_features(self, text_blocks: List[dict], base_size: float) -> List[Dict[str, Any]]:
        line_features: List[Dict[str, Any]] = []
        for block in text_blocks:
            block_bbox = block.get("bbox") or (0.0, 0.0, 0.0, 0.0)
            for line in block.get("lines", []):
                line_bbox = line.get("bbox") or block_bbox
                line_text = ""
                max_size = 0.0
                bold_chars = 0
                total_chars = 0
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    line_text += txt
                    if txt.strip():
                        max_size = max(max_size, span.get("size", 0.0))
                        total_chars += len(txt.strip())
                        if "bold" in str(span.get("font", "")).lower():
                            bold_chars += len(txt.strip())
                line_text = self._normalize_ws(line_text)
                if not line_text:
                    continue
                bold_ratio = (bold_chars / total_chars) if total_chars > 0 else 0.0
                line_features.append(
                    {
                        "text": line_text,
                        "y0": float(line_bbox[1]),
                        "y1": float(line_bbox[3]),
                        "max_size": max_size,
                        "bold_ratio": bold_ratio,
                        "is_heading": self._is_heading_like_line(line_text, max_size, base_size, bold_ratio),
                    }
                )
        line_features.sort(key=lambda item: (item["y0"], item["text"]))
        return line_features

    def _looks_like_paragraphish_line(self, text: str) -> bool:
        compact = " ".join(str(text or "").split()).strip()
        if len(compact) < 70:
            return False
        if compact.endswith(":"):
            return False
        if re.match(r"^(figure|fig\.?|table|tab\.?)\s+", compact, flags=re.IGNORECASE):
            return False
        if compact.count(" ") < 8:
            return False
        return True

    def _detect_two_columns(self, page_fitz: fitz.Page) -> bool:
        text_blocks = [b for b in page_fitz.get_text("dict")["blocks"] if b["type"] == 0]
        if len(text_blocks) < 4:
            return False

        centers = []
        for b in text_blocks:
            x0, y0, x1, y1 = b["bbox"]
            w = x1 - x0
            h = y1 - y0
            if w > 40 and h > 10:
                centers.append((x0 + x1) / 2)

        if len(centers) < 4:
            return False

        page_mid = page_fitz.rect.width / 2
        left = sum(1 for c in centers if c < page_mid * 0.9)
        right = sum(1 for c in centers if c > page_mid * 1.1)
        return left >= 2 and right >= 2

    # -------------------------------------------------------------------------
    # Page analysis / classification
    # -------------------------------------------------------------------------

    def _analyze_page(self, page_fitz: fitz.Page, page_plumber, *, page_num: int, base_size: float) -> Dict[str, Any]:
        text = page_fitz.get_text("text")
        text_blocks = [b for b in page_fitz.get_text("dict")["blocks"] if b["type"] == 0]
        line_features = self._extract_line_features(text_blocks, base_size)
        image_infos = page_fitz.get_image_info()
        drawings = page_fitz.get_drawings()
        tables = page_plumber.find_tables()

        page_area = page_fitz.rect.width * page_fitz.rect.height

        image_area = 0.0
        for img in image_infos:
            rect = fitz.Rect(img["bbox"])
            image_area += rect.get_area()

        text_area = 0.0
        for b in text_blocks:
            rect = fitz.Rect(b["bbox"])
            text_area += rect.get_area()

        heading_lines = [item["text"] for item in line_features if item["is_heading"]]
        top_lines = [item["text"] for item in line_features[:4]]
        bottom_lines = [item["text"] for item in line_features[-4:]]
        citation_count = len(re.findall(r"\[[0-9]{1,3}(?:\s*,\s*[0-9]{1,3})*\]|\b[A-Z][A-Za-z\-]+ et al\.\b|\([A-Z][A-Za-z\-]+ et al\.,?\s*20\d{2}\)", text, flags=re.IGNORECASE))
        figure_caption_count = sum(
            1
            for item in line_features
            if re.match(r"^(?:figure|fig\.?)\s+[a-z0-9ivxlcdm]+(?:[\.:]|\s)", item["text"], flags=re.IGNORECASE)
        )
        table_caption_count = sum(
            1
            for item in line_features
            if re.match(r"^(?:table|tab\.?)\s+[a-z0-9ivxlcdm]+(?:[\.:]|\s)", item["text"], flags=re.IGNORECASE)
        )
        equation_line_count = sum(1 for item in line_features if re.search(r"(?:=|≤|≥|∑|Σ|λ|μ|θ|β|→|<-|->)", item["text"]))
        paragraph_like_count = sum(1 for item in line_features if self._looks_like_paragraphish_line(item["text"]))
        author_affiliation_hits = 0
        if page_num == 1:
            lower_text = text.lower()
            author_affiliation_hits = sum(
                1
                for token in ("university", "institute", "college", "school", "department", "laboratory", "arxiv", "@")
                if token in lower_text
            )

        return {
            "page_num": page_num,
            "text_len": len(text.strip()),
            "image_count": len(image_infos),
            "drawing_count": len(drawings),
            "table_count": len(tables),
            "image_area_ratio": image_area / max(page_area, 1.0),
            "text_area_ratio": text_area / max(page_area, 1.0),
            "two_columns": self._detect_two_columns(page_fitz),
            "garbled_text": self._looks_like_garbled_text(text),
            "page_width": page_fitz.rect.width,
            "page_height": page_fitz.rect.height,
            "heading_lines": heading_lines,
            "top_lines": top_lines,
            "bottom_lines": bottom_lines,
            "citation_count": citation_count,
            "figure_caption_count": figure_caption_count,
            "table_caption_count": table_caption_count,
            "equation_line_count": equation_line_count,
            "paragraph_like_count": paragraph_like_count,
            "author_affiliation_hits": author_affiliation_hits,
        }

    def _classify_document(self, page_stats: List[Dict[str, Any]]) -> str:
        num_pages = len(page_stats)
        avg_text_len = sum(p["text_len"] for p in page_stats) / max(num_pages, 1)
        avg_image_ratio = sum(p["image_area_ratio"] for p in page_stats) / max(num_pages, 1)
        total_tables = sum(p["table_count"] for p in page_stats)
        two_col_pages = sum(1 for p in page_stats if p["two_columns"])
        garbled_pages = sum(1 for p in page_stats if p["garbled_text"])
        low_text_pages = sum(1 for p in page_stats if p["text_len"] < 80)
        image_heavy_pages = sum(1 for p in page_stats if p["image_area_ratio"] > 0.45)

        if garbled_pages > num_pages * 0.5 or low_text_pages > num_pages * 0.6 or image_heavy_pages > num_pages * 0.5:
            return "ocr"

        if num_pages == 1 and avg_text_len > 500 and total_tables == 0 and avg_image_ratio < 0.1 and two_col_pages == 0:
            return "simple_text"

        if avg_text_len > 200 and (total_tables > 0 or two_col_pages > 0 or avg_image_ratio > 0.08):
            return "hybrid_paper"

        return "hybrid_paper"

    def _classify_page(self, page_stat: Dict[str, Any], doc_mode: str) -> str:
        if page_stat["text_len"] < 80 or page_stat["garbled_text"] or page_stat["image_area_ratio"] > 0.55:
            return "ocr"

        if doc_mode == "simple_text":
            return "simple_text"

        if (
            page_stat["table_count"] > 0
            or page_stat["image_count"] > 0
            or page_stat["drawing_count"] > 8
            or page_stat["two_columns"]
        ):
            return "hybrid_paper"

        if page_stat["text_len"] > 500 and page_stat["image_count"] == 0 and page_stat["table_count"] == 0:
            return "simple_text"

        return doc_mode

    # -------------------------------------------------------------------------
    # Asset saving
    # -------------------------------------------------------------------------

    def _save_clipped_region(self, page_fitz: fitz.Page, clip_rect: fitz.Rect, out_path: str, dpi: int = 180):
        pix = page_fitz.get_pixmap(clip=clip_rect, dpi=dpi)
        pix.save(out_path)

    def _save_full_page_image(self, page_fitz: fitz.Page, page_num: int, dpi: int = 180) -> str:
        page_img_name = f"{self.pdf_name}_page_{page_num}.png"
        page_img_path = os.path.join(self.page_image_dir, page_img_name)
        pix = page_fitz.get_pixmap(dpi=dpi)
        pix.save(page_img_path)
        return page_img_path

    # -------------------------------------------------------------------------
    # Table / figure detection
    # -------------------------------------------------------------------------

    def _table_to_markdown(self, extracted_data: List[List[str]]) -> str:
        if not extracted_data or not extracted_data[0]:
            return ""

        cleaned = []
        for row in extracted_data:
            cleaned.append([(str(cell).replace("\n", " ") if cell else "") for cell in row])

        md = "\n| " + " | ".join(cleaned[0]) + " |\n"
        md += "|" + "|".join(["---"] * len(cleaned[0])) + "|\n"
        for row in cleaned[1:]:
            md += "| " + " | ".join(row) + " |\n"
        return md.strip()

    def _get_figure_zones(self, page_fitz: fitz.Page, table_bboxes: List[fitz.Rect]) -> List[fitz.Rect]:
        zones = [fitz.Rect(img["bbox"]) for img in page_fitz.get_image_info()]

        for d in page_fitz.get_drawings():
            rect = d["rect"]
            if 10 < rect.width < page_fitz.rect.width * 0.95 and 10 < rect.height < page_fitz.rect.height * 0.95:
                if not any(tb.contains(self._rect_center(rect)) for tb in table_bboxes):
                    zones.append(rect)

        margin = 36
        changed = True
        while changed:
            changed = False
            merged_zones = []
            while zones:
                current = zones.pop(0)
                expanded_current = current + (-margin, -margin, margin, margin)

                merged_this_round = False
                for i, mz in enumerate(merged_zones):
                    if (mz + (-margin, -margin, margin, margin)).intersects(expanded_current):
                        merged_zones[i] = mz | current
                        merged_this_round = True
                        changed = True
                        break

                if not merged_this_round:
                    merged_zones.append(current)

            zones = merged_zones

        text_blocks = [b for b in page_fitz.get_text("dict")["blocks"] if b["type"] == 0]
        for i, z in enumerate(zones):
            expanded_z = z
            text_added = True
            while text_added:
                text_added = False
                for b in text_blocks:
                    b_rect = fitz.Rect(b["bbox"])
                    if (expanded_z + (-5, -5, 5, 5)).intersects(b_rect):
                        if b_rect.height < 100 and b_rect.width < 300:
                            if not expanded_z.contains(b_rect):
                                expanded_z = expanded_z | b_rect
                                text_added = True
            zones[i] = expanded_z

        return zones

    # -------------------------------------------------------------------------
    # Extraction modes
    # -------------------------------------------------------------------------

    def _extract_simple_text_page(self, page_fitz: fitz.Page, base_size: float) -> str:
        blocks = [b for b in page_fitz.get_text("dict")["blocks"] if b["type"] == 0]
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        out = []
        for b in blocks:
            block_text = self._normalize_block_text(b.get("lines", []), base_size)
            if not block_text:
                continue

            # heading detection for single-line block
            lines = block_text.splitlines()
            if len(lines) == 1:
                line = lines[0].strip()
                max_size = 0.0
                bold_chars = 0
                total_chars = 0
                for ln in b.get("lines", []):
                    for span in ln.get("spans", []):
                        txt = span.get("text", "")
                        if txt.strip():
                            max_size = max(max_size, span.get("size", 0.0))
                            total_chars += len(txt.strip())
                            if "bold" in str(span.get("font", "")).lower():
                                bold_chars += len(txt.strip())
                bold_ratio = (bold_chars / total_chars) if total_chars > 0 else 0.0

                if self._is_heading_like_line(line, max_size, base_size, bold_ratio):
                    level = 2 if max_size >= base_size * 1.2 else 3
                    out.append(f"{'#' * level} {line.rstrip('.').strip()}")
                    continue

            out.append(block_text)

        return "\n\n".join(out).strip()

    def _extract_page_text_blocks(self, page_fitz: fitz.Page) -> List[dict]:
        page_dict = page_fitz.get_text("dict")
        return [b for b in page_dict["blocks"] if b["type"] == 0]


    def _build_text_item_from_block(self, block: dict, base_size: float) -> Optional[dict]:
        block_text = self._normalize_block_text(block.get("lines", []), base_size)
        if not block_text:
            return None

        lines = [ln.strip() for ln in block_text.splitlines() if ln.strip()]

        if len(lines) == 1:
            line = lines[0]

            max_size = 0.0
            bold_chars = 0
            total_chars = 0
            for ln in block.get("lines", []):
                for span in ln.get("spans", []):
                    txt = span.get("text", "")
                    if txt.strip():
                        max_size = max(max_size, span.get("size", 0.0))
                        total_chars += len(txt.strip())
                        if "bold" in str(span.get("font", "")).lower():
                            bold_chars += len(txt.strip())

            bold_ratio = (bold_chars / total_chars) if total_chars > 0 else 0.0

            if self._is_heading_like_line(line, max_size, base_size, bold_ratio):
                level = 2 if max_size >= base_size * 1.2 else 3
                block_text = f"{'#' * level} {line.rstrip('.').strip()}"

        return {
            "type": "text",
            "bbox": block["bbox"],
            "content": block_text.strip(),
        }


    def _collect_caption_tables(
        self,
        page_fitz: fitz.Page,
        text_blocks: List[dict],
        used_text_block_ids: set,
        page_num: int,
    ) -> Tuple[List[dict], List[fitz.Rect]]:
        layout_items = []
        real_table_bboxes: List[fitz.Rect] = []

        table_caption_blocks = self._find_table_caption_blocks(text_blocks)
        for idx, cap_block in enumerate(table_caption_blocks, start=1):
            table_region = self._expand_table_from_caption(page_fitz, cap_block, text_blocks, max_expand_down=550)

            if table_region.get_area() <= 5000:
                continue

            real_table_bboxes.append(table_region)

            table_img_name = f"{self.pdf_name}_page_{page_num}_table_caption_{idx}.png"
            table_img_path = os.path.join(self.table_dir, table_img_name)
            table_clip_rect = self._expand_rect_with_text(page_fitz, table_region, pad=12)

            self._save_clipped_region(
                page_fitz,
                table_clip_rect.intersect(page_fitz.rect),
                table_img_path,
                dpi=self.image_dpi,
            )

            used_text_block_ids.add(text_blocks.index(cap_block))
            caption_text = self._extract_block_plain_text(cap_block)

            layout_items.append({
                "type": "table",
                "bbox": (table_clip_rect.x0, table_clip_rect.y0, table_clip_rect.x1, table_clip_rect.y1),
                "content": "\n\n".join([
                    f"**{caption_text}**",
                    f"![Table {idx}]({self._relpath_from_output(table_img_path)})",
                ]),
            })

        return layout_items, real_table_bboxes


    def _collect_plumber_tables(
        self,
        page_fitz: fitz.Page,
        page_plumber,
        text_blocks: List[dict],
        used_text_block_ids: set,
        page_num: int,
        existing_table_bboxes: List[fitz.Rect],
    ) -> Tuple[List[dict], List[fitz.Rect], List[fitz.Rect]]:
        layout_items = []
        real_table_bboxes = list(existing_table_bboxes)
        chart_like_bboxes: List[fitz.Rect] = []

        raw_tables = page_plumber.find_tables()
        for idx, table in enumerate(raw_tables, start=1):
            tb_rect = fitz.Rect(table.bbox)

            if self._overlaps_any(tb_rect, real_table_bboxes, pad=10):
                continue

            if self._looks_like_real_table(page_fitz, table):
                real_table_bboxes.append(tb_rect)

                table_md = self._table_to_markdown(table.extract())
                if not table_md:
                    continue

                table_clip_rect = self._expand_rect_with_text(page_fitz, tb_rect, pad=16)
                table_img_name = f"{self.pdf_name}_page_{page_num}_table_{idx}.png"
                table_img_path = os.path.join(self.table_dir, table_img_name)

                self._save_clipped_region(
                    page_fitz,
                    table_clip_rect.intersect(page_fitz.rect),
                    table_img_path,
                    dpi=self.image_dpi,
                )

                caption_blocks = self._find_caption_blocks_for_rect(
                    page_fitz=page_fitz,
                    target_rect=table_clip_rect,
                    text_blocks=text_blocks,
                    used_block_ids=used_text_block_ids,
                    kind="table",
                )
                for block in caption_blocks:
                    used_text_block_ids.add(text_blocks.index(block))

                caption_text = self._join_caption_blocks(caption_blocks)

                content_parts = []
                if caption_text:
                    content_parts.append(f"**{caption_text}**")
                else:
                    content_parts.append(f"**Table {idx} (Page {page_num})**")

                content_parts.append(table_md)
                content_parts.append(f"![Table {idx}]({self._relpath_from_output(table_img_path)})")

                layout_items.append({
                    "type": "table",
                    "bbox": (table_clip_rect.x0, table_clip_rect.y0, table_clip_rect.x1, table_clip_rect.y1),
                    "content": "\n\n".join(content_parts),
                })
            else:
                chart_like_bboxes.append(tb_rect)

        return layout_items, real_table_bboxes, chart_like_bboxes


    def _build_figure_regions(
        self,
        page_fitz: fitz.Page,
        real_table_bboxes: List[fitz.Rect],
        chart_like_bboxes: List[fitz.Rect],
    ) -> List[fitz.Rect]:
        excluded_bboxes = real_table_bboxes + chart_like_bboxes
        figure_zones = self._get_figure_zones(page_fitz, excluded_bboxes)

        filtered_figure_zones = []
        for fz in figure_zones:
            if self._overlaps_any(fz, excluded_bboxes, pad=5):
                continue
            clip_rect = self._expand_rect_with_text(page_fitz, fz, pad=12).intersect(page_fitz.rect)
            if clip_rect.is_empty or clip_rect.get_area() < 1000:
                continue
            filtered_figure_zones.append(clip_rect)

        composite_figure_zones = self._detect_composite_figure_regions(page_fitz, filtered_figure_zones)
        return composite_figure_zones if composite_figure_zones else filtered_figure_zones


    def _collect_figures(
        self,
        page_fitz: fitz.Page,
        text_blocks: List[dict],
        used_text_block_ids: set,
        page_num: int,
        figure_regions: List[fitz.Rect],
    ) -> List[dict]:
        layout_items = []

        for idx, clip_rect in enumerate(figure_regions, start=1):
            fig_name = f"{self.pdf_name}_page_{page_num}_fig_{idx}.png"
            fig_path = os.path.join(self.figure_dir, fig_name)
            self._save_clipped_region(page_fitz, clip_rect, fig_path, dpi=self.image_dpi)

            caption_blocks = self._find_caption_blocks_for_rect(
                page_fitz=page_fitz,
                target_rect=clip_rect,
                text_blocks=text_blocks,
                used_block_ids=used_text_block_ids,
                kind="figure",
            )
            for block in caption_blocks:
                used_text_block_ids.add(text_blocks.index(block))

            caption_text = self._join_caption_blocks(caption_blocks)

            content_parts = []
            if caption_text:
                content_parts.append(caption_text)
            content_parts.append(f"![Figure {idx} on Page {page_num}]({self._relpath_from_output(fig_path)})")

            layout_items.append({
                "type": "image",
                "bbox": (clip_rect.x0, clip_rect.y0, clip_rect.x1, clip_rect.y1),
                "content": "\n\n".join(content_parts),
            })

        return layout_items


    def _collect_body_text(
        self,
        text_blocks: List[dict],
        used_text_block_ids: set,
        real_table_bboxes: List[fitz.Rect],
        chart_like_bboxes: List[fitz.Rect],
        final_figure_zones: List[fitz.Rect],
        base_size: float,
    ) -> List[dict]:
        layout_items = []

        occupied_rects = []
        occupied_rects.extend(real_table_bboxes)
        occupied_rects.extend(chart_like_bboxes)
        occupied_rects.extend(final_figure_zones)

        for idx, block in enumerate(text_blocks):
            if idx in used_text_block_ids:
                continue

            b_rect = fitz.Rect(block["bbox"])
            center = self._rect_center(b_rect)

            is_in_table = any(tb.contains(center) for tb in real_table_bboxes)
            is_in_chart_like = any(cb.contains(center) for cb in chart_like_bboxes)
            is_in_figure = any((fz + (-10, -10, 10, 10)).contains(center) for fz in final_figure_zones)

            if is_in_table or is_in_chart_like or is_in_figure:
                continue

            if self._overlaps_any(b_rect, occupied_rects, pad=2):
                overlap_area = 0.0
                for occ in occupied_rects:
                    inter = b_rect & occ
                    if not inter.is_empty:
                        overlap_area += inter.get_area()
                if overlap_area / max(b_rect.get_area(), 1.0) > 0.25:
                    continue

            item = self._build_text_item_from_block(block, base_size)
            if item and item["content"]:
                layout_items.append(item)

        return layout_items


    def _assemble_layout_items(self, layout_items: List[dict], page_width: float) -> str:
        layout_items.sort(key=lambda item: self._bbox_sort_key(item["bbox"], page_width))
        return "\n\n".join(item["content"] for item in layout_items if item["content"].strip()).strip()

    def _extract_hybrid_page(self, page_fitz: fitz.Page, page_plumber, page_num: int, base_size: float) -> str:
        text_blocks = self._extract_page_text_blocks(page_fitz)
        used_text_block_ids = set()
        layout_items = []

        # 1. caption-driven tables
        caption_table_items, real_table_bboxes = self._collect_caption_tables(
            page_fitz=page_fitz,
            text_blocks=text_blocks,
            used_text_block_ids=used_text_block_ids,
            page_num=page_num,
        )
        layout_items.extend(caption_table_items)

        # 2. pdfplumber tables
        plumber_table_items, real_table_bboxes, chart_like_bboxes = self._collect_plumber_tables(
            page_fitz=page_fitz,
            page_plumber=page_plumber,
            text_blocks=text_blocks,
            used_text_block_ids=used_text_block_ids,
            page_num=page_num,
            existing_table_bboxes=real_table_bboxes,
        )
        layout_items.extend(plumber_table_items)

        # 3. figures
        final_figure_zones = self._build_figure_regions(
            page_fitz=page_fitz,
            real_table_bboxes=real_table_bboxes,
            chart_like_bboxes=chart_like_bboxes,
        )
        figure_items = self._collect_figures(
            page_fitz=page_fitz,
            text_blocks=text_blocks,
            used_text_block_ids=used_text_block_ids,
            page_num=page_num,
            figure_regions=final_figure_zones,
        )
        layout_items.extend(figure_items)

        # 4. body text
        body_items = self._collect_body_text(
            text_blocks=text_blocks,
            used_text_block_ids=used_text_block_ids,
            real_table_bboxes=real_table_bboxes,
            chart_like_bboxes=chart_like_bboxes,
            final_figure_zones=final_figure_zones,
            base_size=base_size,
        )
        layout_items.extend(body_items)

        # 5. assemble
        return self._assemble_layout_items(layout_items, page_fitz.rect.width)


    def _extract_ocr_page(self, page_fitz: fitz.Page, page_num: int, page_hint: Optional[Dict[str, Any]] = None) -> str:
        if self.use_pdf_page_ocr:
            try:
                md = self.ocr_backend.ocr_pdf_page(self.pdf_path, page_num=page_num, page_hint=page_hint)
                return f"\n\n<!-- OCR page {page_num} -->\n\n{md}\n\n"
            except Exception:
                pass

        page_img_path = self._save_full_page_image(page_fitz, page_num, dpi=self.ocr_dpi)
        try:
            md = self.ocr_backend.ocr_page_image(page_img_path, page_num=page_num, page_hint=page_hint)
        except Exception as exc:
            # OCR server unavailable or failed — fall back to empty placeholder
            # so the rest of the document can still be extracted.
            print(f"[HybridPDFExtractor] OCR failed for page {page_num}: {exc}")
            md = f"[OCR unavailable for page {page_num}]"
        return f"\n\n<!-- OCR page {page_num} -->\n\n{md}\n\n"

    def _build_non_ocr_document_profile(
        self,
        *,
        page_numbers: List[int],
        page_stats: List[Dict[str, Any]],
        total_pages: int,
    ) -> DocumentContentProfile:
        page_info_by_num = {
            int(page_stat.get("page_num") or 0): page_stat
            for page_stat in page_stats
            if int(page_stat.get("page_num") or 0) > 0
        }
        routing_pages = [
            page_info_by_num[page_num]
            for page_num in page_numbers
            if page_num in page_info_by_num
        ]
        return build_document_content_profile(routing_pages, total_pages=total_pages)

    def _extract_non_ocr_segments(self, page_numbers: List[int]):
        bridge = IaPhase1Bridge(
            pdf_path=self.pdf_path,
            pdf_name=self.pdf_name,
            assets_root=self.assets_dir,
        )
        result = bridge.extract_non_ocr_pages(page_allowlist=page_numbers)
        missing = sorted(set(page_numbers) - set(result.page_segments))
        if missing:
            raise ValueError(f"ia_phase1 bridge did not return markdown for non-OCR pages: {missing}")
        return result

    def _extract_native_non_ocr_page(
        self,
        *,
        page_fitz: fitz.Page,
        page_plumber,
        page_num: int,
        page_mode: str,
        base_size: float,
    ) -> str:
        if page_mode == "hybrid_paper":
            return self._extract_hybrid_page(page_fitz, page_plumber, page_num, base_size)
        return self._extract_simple_text_page(page_fitz, base_size)

    def _extract_native_non_ocr_runs(
        self,
        *,
        runs,
        page_modes: Dict[int, str],
        doc_fitz: fitz.Document,
        doc_plumber,
        base_size: float,
    ) -> Dict[int, str]:
        segments: Dict[int, str] = {}
        for run in runs:
            if run.handler != "native":
                continue
            for page_num in run.page_numbers:
                page_index = page_num - 1
                segments[page_num] = self._extract_native_non_ocr_page(
                    page_fitz=doc_fitz[page_index],
                    page_plumber=doc_plumber.pages[page_index],
                    page_num=page_num,
                    page_mode=page_modes[page_num],
                    base_size=base_size,
                )
        return segments

    # -------------------------------------------------------------------------
    # Main public APIs
    # -------------------------------------------------------------------------

    def extract_to_markdown(self) -> str:
        self.last_routing_info = {}
        self._reset_asset_dirs()
        doc_fitz = fitz.open(self.pdf_path)
        doc_plumber = pdfplumber.open(self.pdf_path)
        base_size = self._get_base_font_size(doc_fitz)

        page_stats = []
        for i in range(len(doc_fitz)):
            page_stats.append(
                self._analyze_page(
                    doc_fitz[i],
                    doc_plumber.pages[i],
                    page_num=i + 1,
                    base_size=base_size,
                )
            )

        doc_mode = self._classify_document(page_stats)

        page_modes: Dict[int, str] = {}
        non_ocr_pages: List[int] = []
        for page_index, page_stat in enumerate(page_stats):
            display_page_num = page_index + 1
            page_mode = self._classify_page(page_stat, doc_mode)
            page_modes[display_page_num] = page_mode
            if page_mode != "ocr":
                non_ocr_pages.append(display_page_num)

        non_ocr_segments: Dict[int, str] = {}
        document_title = self.pdf_name
        routing_profile = None
        if non_ocr_pages:
            routing_profile = self._build_non_ocr_document_profile(
                page_numbers=non_ocr_pages,
                page_stats=page_stats,
                total_pages=len(doc_fitz),
            )

            native_segments = self._extract_native_non_ocr_runs(
                runs=routing_profile.runs,
                page_modes=page_modes,
                doc_fitz=doc_fitz,
                doc_plumber=doc_plumber,
                base_size=base_size,
            )
            non_ocr_segments.update(native_segments)

            for run in routing_profile.runs:
                if run.handler != "ia_phase1":
                    continue
                bridge_result = self._extract_non_ocr_segments(run.page_numbers)
                non_ocr_segments.update(bridge_result.page_segments)
                if 1 in run.page_numbers and str(bridge_result.document_title or "").strip():
                    document_title = str(bridge_result.document_title).strip()

            self.last_routing_info = {
                "document_handler": routing_profile.document_handler,
                "ia_score": routing_profile.scores.get("ia_score"),
                "native_score": routing_profile.scores.get("native_score"),
                "scores": dict(routing_profile.scores),
                "page_handlers": {
                    int(profile.page_num): str(profile.handler)
                    for profile in routing_profile.page_profiles
                },
                "runs": [
                    {
                        "handler": str(run.handler),
                        "page_numbers": list(run.page_numbers),
                    }
                    for run in routing_profile.runs
                ],
            }
        else:
            self.last_routing_info = {
                "document_handler": "native",
                "ia_score": 0,
                "native_score": 0,
                "scores": {"ia_score": 0, "native_score": 0},
                "page_handlers": {},
                "runs": [],
            }

        full_markdown = [f"# {document_title}\n"]
        full_markdown.append(f"<!-- document_mode: {doc_mode} -->\n")

        for page_index in range(len(doc_fitz)):
            display_page_num = page_index + 1
            page_fitz = doc_fitz[page_index]
            page_stat = page_stats[page_index]
            page_mode = page_modes[display_page_num]

            full_markdown.append(f"\n<!-- page {display_page_num} mode: {page_mode} -->\n")

            if page_mode == "ocr":
                page_md = self._extract_ocr_page(page_fitz, display_page_num, page_hint=page_stat)
            else:
                page_md = non_ocr_segments.get(display_page_num, "")

            full_markdown.append(page_md)
            full_markdown.append("\n---\n")

        doc_fitz.close()
        doc_plumber.close()

        md = "\n".join(full_markdown)
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()

    def save_markdown(self, output_path: Optional[str] = None) -> str:
        md = self.extract_to_markdown()
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{self.pdf_name}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        return output_path

    def extract_to_dict(self) -> Dict[str, Any]:
        md = self.extract_to_markdown()
        return {
            "pdf_path": self.pdf_path,
            "pdf_name": self.pdf_name,
            "assets_dir": self.assets_dir,
            "markdown": md,
            "routing": dict(self.last_routing_info),
        }


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
def test_hybrid_extractor():
    from BatchAgent.ocr_agent.pipeline_custom import make_ocr_args
    from BatchAgent.ocr_agent.hybrid_pdf_extractor import HybridPDFExtractor, PipelineCustomOCRBackend

    args = make_ocr_args(
        server="http://localhost:8002/v1",
        model="allenai/olmOCR-2-7B-1025-FP8",
        workspace="./tmp_ocr",
        embed_page_markers=True,
        save_rendered_pages=False,
        materialize_assets=False,
    )

    backend = PipelineCustomOCRBackend(args)

    extractor = HybridPDFExtractor(
        pdf_path="data/sote/SP25_CMPE272_03.pdf",
        ocr_backend=backend,
        use_pdf_page_ocr=False,   # False: 保存页图再OCR；True: 直接对PDF页OCR
    )

    md_path = extractor.save_markdown()
    print(md_path)


import argparse
import glob


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adaptive PDF to Markdown extractor with simple/hybrid/OCR routing."
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        help="One or more PDF paths, glob patterns, or directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save markdown outputs. Defaults to each PDF's own folder.",
    )
    parser.add_argument(
        "--ocr-server",
        default=None,
        help="OCR server URL, e.g. http://localhost:8002/v1",
    )
    parser.add_argument(
        "--ocr-model",
        default="allenai/olmOCR-2-7B-1025-FP8",
        help="OCR model name for pipeline_custom backend.",
    )
    parser.add_argument(
        "--ocr-workspace",
        default="./tmp_ocr",
        help="Workspace for pipeline_custom OCR backend.",
    )
    parser.add_argument(
        "--use-pdf-page-ocr",
        action="store_true",
        help="Use direct PDF-page OCR instead of saving page image first.",
    )
    parser.add_argument(
        "--image-dpi",
        type=int,
        default=180,
        help="DPI used for cropped figures/tables.",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=180,
        help="DPI used for OCR page image rendering.",
    )
    parser.add_argument(
        "--dummy-ocr",
        action="store_true",
        help="Use dummy OCR backend instead of pipeline_custom.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search directories for PDFs.",
    )
    parser.add_argument(
        "--embed-page-markers",
        action="store_true",
        help="Pass embed_page_markers=True to pipeline_custom OCR args.",
    )
    parser.add_argument(
        "--save-rendered-pages",
        action="store_true",
        help="Pass save_rendered_pages=True to pipeline_custom OCR args.",
    )
    parser.add_argument(
        "--materialize-assets",
        action="store_true",
        help="Pass materialize_assets=True to pipeline_custom OCR args.",
    )
    parser.add_argument(
        "--guided-decoding",
        action="store_true",
        help="Pass guided_decoding=True to pipeline_custom OCR args.",
    )
    parser.add_argument(
        "--target-longest-image-dim",
        type=int,
        default=1288,
        help="Pass target_longest_image_dim to pipeline_custom OCR args.",
    )
    parser.add_argument(
        "--page-max-tokens",
        type=int,
        default=8000,
        help="Pass page_max_tokens to pipeline_custom OCR args.",
    )
    parser.add_argument(
        "--table-format",
        choices=["html", "markdown"],
        default="html",
        help="Pass table_format to pipeline_custom OCR args.",
    )
    return parser


def expand_pdf_inputs(inputs: list[str], recursive: bool = False) -> list[str]:
    pdf_paths: list[str] = []

    for item in inputs:
        if os.path.isdir(item):
            pattern = "**/*.pdf" if recursive else "*.pdf"
            found = glob.glob(os.path.join(item, pattern), recursive=recursive)
            pdf_paths.extend(found)
        else:
            matched = glob.glob(item, recursive=recursive)
            if matched:
                pdf_paths.extend([p for p in matched if os.path.isfile(p)])
            elif os.path.isfile(item):
                pdf_paths.append(item)

    # keep only pdf files, normalize, deduplicate while preserving order
    seen = set()
    final_paths = []
    for p in pdf_paths:
        if not p.lower().endswith(".pdf"):
            continue
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            final_paths.append(ap)

    return final_paths


def make_backend_from_cli(args) -> OCRBackendBase:
    if args.dummy_ocr:
        return DummyOCRBackend()

    if not args.ocr_server:
        raise ValueError("Either --dummy-ocr or --ocr-server must be provided.")

    from .pipeline_custom import make_ocr_args

    ocr_args = make_ocr_args(
        server=args.ocr_server,
        model=args.ocr_model,
        workspace=args.ocr_workspace,
        guided_decoding=args.guided_decoding,
        page_max_tokens=args.page_max_tokens,
        target_longest_image_dim=args.target_longest_image_dim,
        table_format=args.table_format,
        emit_figure_placeholders=True,
        save_rendered_pages=args.save_rendered_pages,
        materialize_assets=args.materialize_assets,
        embed_page_markers=args.embed_page_markers,
    )
    return PipelineCustomOCRBackend(ocr_args)


def run_cli(args) -> int:
    pdf_paths = expand_pdf_inputs(args.pdfs, recursive=args.recursive)

    if not pdf_paths:
        print("No PDF files found.")
        return 1

    backend = make_backend_from_cli(args)

    failed = 0
    for pdf_path in pdf_paths:
        try:
            extractor = HybridPDFExtractor(
                pdf_path=pdf_path,
                ocr_backend=backend,
                image_dpi=args.image_dpi,
                ocr_dpi=args.ocr_dpi,
                use_pdf_page_ocr=args.use_pdf_page_ocr,
            )

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                out_name = f"{Path(pdf_path).stem}.md"
                output_path = os.path.join(args.output_dir, out_name)
            else:
                output_path = None

            saved_path = extractor.save_markdown(output_path=output_path)
            print(f"[OK] {pdf_path} -> {saved_path}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {pdf_path}: {e}")

    return 1 if failed else 0


def cli_main():
    parser = build_argparser()
    args = parser.parse_args()
    raise SystemExit(run_cli(args))


if __name__ == "__main__":
    cli_main()

"""
python -m BatchAgent.ocr_agent.hybrid_pdf_extractor "data/pdftest/CMPE 188 Project Proposal.pdf" \
  --ocr-server http://localhost:8002/v1 \


python -m BatchAgent.ocr_agent.hybrid_pdf_extractor data/pdftest/NVIDIA-Nemotron-3-Super-Technical-Report.pdf \
  --ocr-server http://localhost:8002/v1

python -m BatchAgent.ocr_agent.hybrid_pdf_extractor a.pdf b.pdf c.pdf \
  --ocr-server http://localhost:8002/v1

Whole folder:
python -m BatchAgent.ocr_agent.hybrid_pdf_extractor ./pdfs \
  --ocr-server http://localhost:8002/v1

python -m BatchAgent.ocr_agent.hybrid_pdf_extractor ./pdfs \
  --recursive \
  --ocr-server http://localhost:8002/v1

python -m BatchAgent.ocr_agent.hybrid_pdf_extractor ./pdfs/*.pdf \
  --ocr-server http://localhost:8002/v1 \
  --output-dir ./md_output

python -m BatchAgent.ocr_agent.hybrid_pdf_extractor paper.pdf --dummy-ocr
"""
