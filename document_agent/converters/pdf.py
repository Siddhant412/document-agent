from __future__ import annotations

import sys
from pathlib import Path

import fitz

from document_agent.converters.assets import upload_assets_and_rewrite_markdown
from document_agent.converters.base import ConversionContext, ConversionResult, Converter
from document_agent.converters.markdown import with_frontmatter
from document_agent.errors import DocumentAgentError


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
        _ensure_vendor_import_path()

        from improved_ocr_agent.hybrid_pdf_extractor import (  # type: ignore
            HybridPDFExtractor,
            PipelineCustomOCRBackend,
        )

        ocr_backend = None
        if context.settings.ocr_server_url:
            from improved_ocr_agent.pipeline_custom import make_ocr_args  # type: ignore

            args = make_ocr_args(
                server=context.settings.ocr_server_url,
                model=context.settings.ocr_model,
                workspace=str(context.temp_dir / "ocr_workspace"),
                api_key=context.settings.ocr_api_key,
                page_max_tokens=context.settings.ocr_page_max_tokens,
                target_longest_image_dim=context.settings.ocr_target_longest_image_dim,
                save_rendered_pages=False,
                materialize_assets=False,
            )
            ocr_backend = PipelineCustomOCRBackend(args)

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
        assets_dir = Path(str(result.get("assets_dir") or ""))
        markdown, uploaded = upload_assets_and_rewrite_markdown(
            context,
            markdown=markdown,
            assets_root=assets_dir,
        )
        markdown = with_frontmatter(
            markdown,
            job_id=context.job_id,
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


def _ensure_vendor_import_path() -> None:
    candidates = [
        Path(__file__).resolve().parents[2] / "vendor",
        Path.cwd() / "vendor",
        Path("/app/vendor"),
    ]
    for vendor_root in candidates:
        if (vendor_root / "improved_ocr_agent").is_dir() and str(vendor_root) not in sys.path:
            sys.path.insert(0, str(vendor_root))
            return
