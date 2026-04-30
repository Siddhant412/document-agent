from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import fitz

from document_agent.converters.assets import upload_assets_and_rewrite_markdown
from document_agent.converters.base import ConversionContext, ConversionResult, Converter
from document_agent.converters.markdown import with_frontmatter
from document_agent.errors import DocumentAgentError
from document_agent.ocr import OcrClient

try:
    from improved_ocr_agent.hybrid_pdf_extractor import HybridPDFExtractor
except ImportError:
    HybridPDFExtractor = None  # type: ignore[assignment]


class PdfConverter(Converter):
    detected_types = {"pdf"}

    def convert(self, context: ConversionContext) -> ConversionResult:
        page_count = _pdf_page_count(context.source_path)
        if page_count > context.settings.max_pdf_pages:
            raise DocumentAgentError(
                code="PDF_TOO_MANY_PAGES",
                message=f"PDF has {page_count} pages, limit is {context.settings.max_pdf_pages}.",
                retryable=False,
                details={"page_count": page_count},
            )

        if context.settings.pdf_use_vendor_extractor:
            try:
                return self._convert_with_vendor(context, page_count=page_count)
            except DocumentAgentError:
                raise
            except Exception as exc:
                # Keep extraction robust: fall back to plain PyMuPDF text when the vendor path fails.
                context.repository.update_progress(
                    job_id=context.job_id,
                    stage="pdf_fallback",
                    percent=45,
                    message="Vendor PDF extractor failed; falling back to text extraction.",
                    payload={"error": str(exc), "exception_type": exc.__class__.__name__},
                )

        return self._convert_with_pymupdf(context, page_count=page_count)

    def _convert_with_vendor(self, context: ConversionContext, *, page_count: int) -> ConversionResult:
        if HybridPDFExtractor is None:
            raise RuntimeError("improved_ocr_agent is not importable.")

        ocr_backend = None
        if context.settings.ocr_server_url:
            ocr_backend = _DocumentAgentOcrBackend(settings=context.settings)

        context.repository.update_progress(
            job_id=context.job_id,
            stage="pdf_extract",
            percent=35,
            message="Running routed PDF extraction.",
        )
        extractor = HybridPDFExtractor(
            pdf_path=str(context.source_path),
            ocr_backend=ocr_backend,
            use_pdf_page_ocr=False,
        )
        result = extractor.extract_to_dict()
        markdown = str(result.get("markdown") or "")
        if not markdown.strip() or "[OCR unavailable for page" in markdown:
            raise RuntimeError("Vendor PDF extractor returned empty Markdown.")
        assets_dir = Path(str(result.get("assets_dir") or ""))
        markdown, uploaded = upload_assets_and_rewrite_markdown(
            context,
            markdown=markdown,
            assets_root=assets_dir,
        )
        markdown = with_frontmatter(
            markdown,
            job_id=context.job_id,
            library_item_id=context.library_item_id,
            batch_id=context.batch_id,
            filename=context.filename,
            detected_type=context.detected_type,
            asset_count=len(uploaded),
            extra={"page_count": page_count, "pdf_extractor": "improved_ocr_agent"},
        )
        return ConversionResult(
            markdown=markdown,
            detected_type="pdf",
            asset_count=len(uploaded),
            assets=uploaded,
            metadata={"page_count": page_count, "extractor": "improved_ocr_agent"},
        )

    def _convert_with_pymupdf(self, context: ConversionContext, *, page_count: int) -> ConversionResult:
        context.repository.update_progress(
            job_id=context.job_id,
            stage="pdf_text_extract",
            percent=45,
            message="Extracting PDF text with PyMuPDF fallback.",
        )
        lines = []
        with fitz.open(context.source_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                text = (page.get_text("text") or "").strip()
                lines.append(f"<!-- page:{page_index} -->\n\n{text or '_No text extracted from this page._'}")
        markdown = with_frontmatter(
            "\n\n".join(lines),
            job_id=context.job_id,
            library_item_id=context.library_item_id,
            batch_id=context.batch_id,
            filename=context.filename,
            detected_type=context.detected_type,
            asset_count=0,
            extra={"page_count": page_count, "pdf_extractor": "pymupdf_fallback"},
        )
        return ConversionResult(
            markdown=markdown,
            detected_type="pdf",
            asset_count=0,
            metadata={"page_count": page_count, "extractor": "pymupdf_fallback"},
        )


def _pdf_page_count(path: Path) -> int:
    try:
        with fitz.open(path) as doc:
            return int(doc.page_count)
    except Exception as exc:
        raise DocumentAgentError(
            code="CORRUPT_OR_UNREADABLE_PDF",
            message=f"Could not read PDF: {exc}",
            retryable=False,
        ) from exc


class _DocumentAgentOcrBackend:
    def __init__(self, *, settings: Any) -> None:
        self.client = OcrClient(settings)

    def ocr_page_image(
        self,
        image_path: str,
        page_num: int,
        page_hint: Optional[dict[str, Any]] = None,
    ) -> str:
        return self.client.ocr_image_bytes(Path(image_path).read_bytes(), page_num=page_num)

    def ocr_pdf_page(
        self,
        pdf_path: str,
        page_num: int,
        page_hint: Optional[dict[str, Any]] = None,
    ) -> str:
        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            return self.client.ocr_image_bytes(pixmap.tobytes("png"), page_num=page_num)
