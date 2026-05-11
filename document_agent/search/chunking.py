from __future__ import annotations

import re
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str


def chunk_markdown(markdown: str, *, max_chars: int = 1200, overlap: int = 180) -> list[TextChunk]:
    text = _WS_RE.sub(" ", markdown.replace("\x00", " ")).strip()
    if not text:
        return []
    max_chars = max(200, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))
    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(TextChunk(index=len(chunks), text=chunk))
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks
