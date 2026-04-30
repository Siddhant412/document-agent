from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from document_agent.converters.base import ConversionContext, ConversionResult
from document_agent.converters.image import ImageConverter
from document_agent.converters.office import OfficeConverter
from document_agent.converters.text import TextConverter
from document_agent.storage import ObjectInfo


def test_text_converter_decodes_utf16(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_bytes("Hello UTF-16".encode("utf-16"))

    result = TextConverter().convert(_context(tmp_path=tmp_path, source_path=source, detected_type="txt"))

    assert "Hello UTF-16" in result.markdown
    assert "detected_type: txt" in result.markdown


def test_image_converter_uses_ocr_without_persisting_source_image(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "scan.png"
    source.write_bytes(b"not-a-real-image")
    repository = _RecordingRepository()

    class FakeOcrClient:
        def __init__(self, settings) -> None:
            self.settings = settings

        def ocr_image_bytes(self, image_bytes: bytes, *, page_num: int = 1) -> str:
            assert image_bytes == b"not-a-real-image"
            assert page_num == 1
            return "# OCR Result\n\nDetected text."

    monkeypatch.setattr("document_agent.converters.image.OcrClient", FakeOcrClient)

    result = ImageConverter().convert(
        _context(
            tmp_path=tmp_path,
            source_path=source,
            detected_type="png",
            repository=repository,
        )
    )

    assert "# OCR Result" in result.markdown
    assert result.asset_count == 0
    assert repository.created_assets == []


def test_docx_pandoc_conversion_uploads_referenced_media(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "report.docx"
    source.write_bytes(b"docx")
    repository = _RecordingRepository()
    object_store = _LocalAssetStore()

    def fake_run(command, *, check: bool, capture_output: bool, timeout: int):
        media_dir = Path(command[command.index("--extract-media") + 1])
        output_md = Path(command[command.index("--output") + 1])
        (media_dir / "media").mkdir(parents=True)
        (media_dir / "media" / "image1.png").write_bytes(b"png")
        output_md.write_text("# Report\n\n![figure](media/image1.png)\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("document_agent.converters.office.shutil.which", lambda _: "/usr/bin/pandoc")
    monkeypatch.setattr("document_agent.converters.office.subprocess.run", fake_run)

    result = OfficeConverter().convert(
        _context(
            tmp_path=tmp_path,
            source_path=source,
            filename="report.docx",
            detected_type="docx",
            repository=repository,
            object_store=object_store,
        )
    )

    assert "http://api/v1/assets/" in result.markdown
    assert "media/image1.png" not in result.markdown
    assert result.asset_count == 1
    assert repository.created_assets[0]["role"] == "embedded_image"


def test_doc_libeoffice_pdf_path_preserves_original_type(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"doc")

    def fake_run(command, *, check: bool, capture_output: bool, timeout: int):
        out_dir = Path(command[command.index("--outdir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "legacy.pdf").write_bytes(b"%PDF-1.7\n")
        return subprocess.CompletedProcess(command, 0)

    def fake_pdf_convert(self, context: ConversionContext) -> ConversionResult:
        assert context.detected_type == "pdf"
        return ConversionResult(
            markdown=(
                "---\n"
                "source_filename: 'legacy.pdf'\n"
                "detected_type: pdf\n"
                "---\n\n"
                "# Converted PDF\n"
            ),
            detected_type="pdf",
            metadata={"page_count": 1},
        )

    monkeypatch.setattr("document_agent.converters.office.shutil.which", lambda _: "/usr/bin/soffice")
    monkeypatch.setattr("document_agent.converters.office.subprocess.run", fake_run)
    monkeypatch.setattr("document_agent.converters.office.PdfConverter.convert", fake_pdf_convert)

    context = _context(tmp_path=tmp_path, source_path=source, filename="legacy.doc", detected_type="doc")
    result = OfficeConverter().convert(context)

    assert "source_filename: 'legacy.doc'" in result.markdown
    assert "detected_type: 'doc'" in result.markdown
    assert "office_converter: 'libreoffice_pdf'" in result.markdown
    assert result.detected_type == "doc"
    assert context.source_path == source
    assert context.detected_type == "doc"


def _context(
    *,
    tmp_path: Path,
    source_path: Path,
    filename: str | None = None,
    detected_type: str,
    repository=None,
    object_store=None,
) -> ConversionContext:
    return ConversionContext(
        job_id=uuid4(),
        batch_id=None,
        input_index=None,
        source_path=source_path,
        filename=filename or source_path.name,
        detected_type=detected_type,
        content_type=None,
        temp_dir=tmp_path,
        repository=repository or _RecordingRepository(),
        object_store=object_store or _LocalAssetStore(),
        settings=SimpleNamespace(
            pandoc_bin="pandoc",
            libreoffice_bin="soffice",
            ocr_server_url="http://ocr",
            ocr_api_key=None,
            ocr_model="model",
            ocr_page_max_tokens=8000,
            ocr_target_longest_image_dim=1288,
            ocr_max_concurrent_requests=1,
        ),
    )


class _RecordingRepository:
    def __init__(self) -> None:
        self.created_assets = []

    def update_progress(self, **kwargs) -> None:
        return None

    def create_asset(self, **kwargs):
        row = {"id": kwargs["asset_id"], **kwargs}
        self.created_assets.append(row)
        return row


class _LocalAssetStore:
    bucket = "bucket"

    def job_asset_key(self, *, job_id: str, role: str, asset_id: str, filename: str) -> str:
        return f"jobs/{job_id}/assets/images/{asset_id}{Path(filename).suffix}"

    def upload_file(self, *, path: Path, object_key: str, mime_type: str | None = None) -> ObjectInfo:
        return ObjectInfo(
            bucket=self.bucket,
            object_key=object_key,
            mime_type=mime_type or "application/octet-stream",
            size_bytes=path.stat().st_size,
            sha256="sha",
            public_url="",
        )

    def public_url_for_asset(self, asset_id: str) -> str:
        return f"http://api/v1/assets/{asset_id}"
