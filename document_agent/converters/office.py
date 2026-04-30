from __future__ import annotations

import shutil
import subprocess

from document_agent.converters.assets import upload_assets_and_rewrite_markdown
from document_agent.converters.base import ConversionContext, ConversionResult, Converter
from document_agent.converters.markdown import rewrite_frontmatter_fields, with_frontmatter
from document_agent.converters.pdf import PdfConverter
from document_agent.errors import DocumentAgentError


class OfficeConverter(Converter):
    detected_types = {"doc", "docx"}

    def convert(self, context: ConversionContext) -> ConversionResult:
        if context.detected_type == "docx" and shutil.which(context.settings.pandoc_bin):
            return self._convert_docx_with_pandoc(context)
        return self._convert_via_pdf(context)

    def _convert_docx_with_pandoc(self, context: ConversionContext) -> ConversionResult:
        media_dir = context.temp_dir / "pandoc_media"
        output_md = context.temp_dir / "pandoc_output.md"
        media_dir.mkdir(parents=True, exist_ok=True)
        command = [
            context.settings.pandoc_bin,
            str(context.source_path),
            "--to",
            "gfm",
            "--extract-media",
            str(media_dir),
            "--output",
            str(output_md),
        ]
        context.repository.update_progress(
            job_id=context.job_id,
            stage="office_pandoc",
            percent=35,
            message="Converting DOCX to Markdown with Pandoc.",
        )
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=300)
        except subprocess.CalledProcessError as exc:
            raise DocumentAgentError(
                code="OFFICE_CONVERSION_FAILED",
                message=(exc.stderr or b"Pandoc conversion failed.").decode("utf-8", errors="replace"),
                retryable=False,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise DocumentAgentError(
                code="OFFICE_CONVERSION_TIMEOUT",
                message="Pandoc conversion timed out.",
                retryable=True,
            ) from exc

        markdown = output_md.read_text(encoding="utf-8", errors="replace")
        markdown, uploaded = upload_assets_and_rewrite_markdown(
            context,
            markdown=markdown,
            assets_root=media_dir,
        )
        markdown = with_frontmatter(
            markdown,
            job_id=context.job_id,
            library_item_id=context.library_item_id,
            batch_id=context.batch_id,
            filename=context.filename,
            detected_type=context.detected_type,
            asset_count=len(uploaded),
            extra={"office_converter": "pandoc"},
        )
        return ConversionResult(
            markdown=markdown,
            detected_type=context.detected_type,
            asset_count=len(uploaded),
            assets=uploaded,
            metadata={"office_converter": "pandoc"},
        )

    def _convert_via_pdf(self, context: ConversionContext) -> ConversionResult:
        if not shutil.which(context.settings.libreoffice_bin):
            raise DocumentAgentError(
                code="OFFICE_CONVERTER_NOT_AVAILABLE",
                message="LibreOffice is required to convert this Office document.",
                retryable=False,
            )
        out_dir = context.temp_dir / "libreoffice"
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            context.settings.libreoffice_bin,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(context.source_path),
        ]
        context.repository.update_progress(
            job_id=context.job_id,
            stage="office_pdf_convert",
            percent=30,
            message="Converting Office document to PDF with LibreOffice.",
        )
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=300)
        except subprocess.CalledProcessError as exc:
            raise DocumentAgentError(
                code="OFFICE_CONVERSION_FAILED",
                message=(exc.stderr or b"LibreOffice conversion failed.").decode("utf-8", errors="replace"),
                retryable=False,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise DocumentAgentError(
                code="OFFICE_CONVERSION_TIMEOUT",
                message="LibreOffice conversion timed out.",
                retryable=True,
            ) from exc
        pdfs = list(out_dir.glob("*.pdf"))
        if not pdfs:
            raise DocumentAgentError(
                code="OFFICE_CONVERSION_FAILED",
                message="LibreOffice did not produce a PDF.",
                retryable=False,
            )
        original_path = context.source_path
        original_type = context.detected_type
        try:
            context.source_path = pdfs[0]
            context.detected_type = "pdf"
            result = PdfConverter().convert(context)
            result.detected_type = original_type
            result.markdown = rewrite_frontmatter_fields(
                result.markdown,
                {
                    "source_filename": context.filename,
                    "detected_type": original_type,
                    "office_converter": "libreoffice_pdf",
                },
            )
            result.metadata["office_converter"] = "libreoffice_pdf"
            return result
        finally:
            context.source_path = original_path
            context.detected_type = original_type
