from pathlib import Path

import pytest

from document_agent.converters.detect import detect_file_type
from document_agent.errors import DocumentAgentError


def test_detect_pdf_by_magic_bytes(tmp_path: Path) -> None:
    path = tmp_path / "upload.bin"
    path.write_bytes(b"%PDF-1.7\n")
    assert detect_file_type(path, "anything.bin", None) == "pdf"


def test_detect_png_by_magic_bytes(tmp_path: Path) -> None:
    path = tmp_path / "scan.bin"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    assert detect_file_type(path, "scan.bin", None) == "png"


def test_detect_txt_by_content_type(tmp_path: Path) -> None:
    path = tmp_path / "notes"
    path.write_text("hello", encoding="utf-8")
    assert detect_file_type(path, "notes", "text/plain") == "txt"


def test_detect_docx_by_office_content_type(tmp_path: Path) -> None:
    path = tmp_path / "upload"
    path.write_bytes(b"PK\x03\x04" + b"x" * 32)
    assert (
        detect_file_type(
            path,
            "upload",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        == "docx"
    )


def test_detect_heif_brand(tmp_path: Path) -> None:
    path = tmp_path / "photo.heif"
    path.write_bytes(b"\x00\x00\x00\x18ftypheif" + b"x" * 32)
    assert detect_file_type(path, "photo.heif", None) == "heic"


def test_unsupported_file_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "archive.bin"
    path.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(DocumentAgentError):
        detect_file_type(path, "archive.bin", "application/octet-stream")
