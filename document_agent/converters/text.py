from __future__ import annotations

from document_agent.converters.base import ConversionContext, ConversionResult, Converter
from document_agent.converters.markdown import with_frontmatter


class TextConverter(Converter):
    detected_types = {"txt"}

    def convert(self, context: ConversionContext) -> ConversionResult:
        raw = context.source_path.read_bytes()
        text = _decode_text(raw)
        body = text.strip() or "_Empty text document._"
        markdown = with_frontmatter(
            body,
            job_id=context.job_id,
            library_item_id=context.library_item_id,
            batch_id=context.batch_id,
            filename=context.filename,
            detected_type=context.detected_type,
            asset_count=0,
        )
        return ConversionResult(markdown=markdown, detected_type="txt")


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
