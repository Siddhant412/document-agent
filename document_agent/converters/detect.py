from __future__ import annotations

from pathlib import Path
from typing import Optional

from document_agent.errors import DocumentAgentError

SUPPORTED_TYPES = {"pdf", "jpg", "jpeg", "png", "heic", "txt", "doc", "docx"}


def detect_file_type(path: Path, filename: str, content_type: Optional[str] = None) -> str:
    with path.open("rb") as handle:
        header = handle.read(512)

    lower_name = filename.lower()
    suffix = Path(lower_name).suffix.lstrip(".")
    content_type = (content_type or "").lower().split(";", 1)[0].strip()

    if header.startswith(b"%PDF"):
        return "pdf"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8"):
        return "jpg"
    heif_brands = (b"ftypheic", b"ftypheix", b"ftypheif", b"ftyphevc", b"ftypheim", b"ftypheis", b"ftypmif1")
    if any(brand in header[:32] for brand in heif_brands):
        return "heic"
    if header.startswith(b"PK\x03\x04") and (
        suffix == "docx"
        or content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return "docx"
    if header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1") and (
        suffix == "doc"
        or content_type == "application/msword"
    ):
        return "doc"
    if content_type in {"application/pdf"}:
        return "pdf"
    if content_type in {"image/png"}:
        return "png"
    if content_type in {"image/jpeg", "image/jpg"}:
        return "jpg"
    if content_type in {"image/heic", "image/heif"}:
        return "heic"
    if content_type in {"text/plain", "text/markdown"}:
        return "txt"
    if content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    if content_type == "application/msword":
        return "doc"
    if suffix in SUPPORTED_TYPES:
        return "jpg" if suffix == "jpeg" else suffix

    # Last chance: simple text heuristic for extensionless text uploads.
    try:
        header.decode("utf-8")
        if b"\x00" not in header:
            return "txt"
    except UnicodeDecodeError:
        pass

    raise DocumentAgentError(
        code="UNSUPPORTED_FILE_TYPE",
        message=f"Unsupported file type for {filename!r}.",
        retryable=False,
        details={"content_type": content_type, "suffix": suffix},
    )
