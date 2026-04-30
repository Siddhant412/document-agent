from __future__ import annotations

from document_agent.converters.base import ConversionContext, ConversionResult, Converter
from document_agent.converters.markdown import with_frontmatter
from document_agent.ocr import OcrClient


class ImageConverter(Converter):
    detected_types = {"jpg", "jpeg", "png", "heic"}

    def convert(self, context: ConversionContext) -> ConversionResult:
        markdown_body = OcrClient(context.settings).ocr_image_bytes(
            context.source_path.read_bytes(),
            page_num=1,
        )
        markdown = with_frontmatter(
            markdown_body or "_OCR returned no text._",
            job_id=context.job_id,
            library_item_id=context.library_item_id,
            batch_id=context.batch_id,
            filename=context.filename,
            detected_type=context.detected_type,
            asset_count=0,
        )
        return ConversionResult(markdown=markdown, detected_type=context.detected_type)
