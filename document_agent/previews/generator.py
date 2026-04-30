from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from PIL import Image, ImageOps

from document_agent.config import Settings
from document_agent.db.repository import Repository
from document_agent.storage import ObjectStore

logger = logging.getLogger(__name__)

_BROWSER_SAFE_ORIGINALS = {"jpg", "jpeg", "png", "pdf", "txt"}


def ensure_preview(
    *,
    job: Dict[str, Any],
    source_path: Path,
    temp_dir: Path,
    repository: Repository,
    object_store: ObjectStore,
    settings: Settings,
) -> Optional[Dict[str, Any]]:
    library_item_id = job.get("library_item_id")
    if not library_item_id:
        return None

    detected_type = str(job.get("detected_type") or "").lower()
    if detected_type in _BROWSER_SAFE_ORIGINALS:
        return None

    item_id = UUID(str(library_item_id))
    try:
        if detected_type == "heic":
            return _heic_preview(
                item_id=item_id,
                job=job,
                source_path=source_path,
                temp_dir=temp_dir,
                repository=repository,
                object_store=object_store,
            )
        if detected_type in {"doc", "docx"}:
            return _office_preview(
                item_id=item_id,
                job=job,
                source_path=source_path,
                temp_dir=temp_dir,
                repository=repository,
                object_store=object_store,
                settings=settings,
            )
    except Exception as exc:
        message = f"Preview generation failed: {exc}"
        logger.warning("preview_generation_failed library_item_id=%s error=%s", item_id, exc)
        repository.mark_library_preview_failed(library_item_id=item_id, message=message)
    return None

def _heic_preview(
    *,
    item_id: UUID,
    job: Dict[str, Any],
    source_path: Path,
    temp_dir: Path,
    repository: Repository,
    object_store: ObjectStore,
) -> Dict[str, Any]:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception:
        logger.debug("pillow_heif_register_failed", exc_info=True)

    preview_path = temp_dir / "preview.png"
    with Image.open(source_path) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        normalized.thumbnail((2400, 2400))
        normalized.save(preview_path, format="PNG", optimize=True)

    asset_id = uuid4()
    info = object_store.upload_file(
        path=preview_path,
        object_key=object_store.preview_key(library_item_id=str(item_id), filename="preview.png"),
        mime_type="image/png",
    )
    return repository.create_asset(
        asset_id=asset_id,
        library_item_id=item_id,
        batch_id=UUID(str(job["batch_id"])) if job.get("batch_id") else None,
        job_id=UUID(str(job["id"])),
        role="preview_image",
        bucket=info.bucket,
        object_key=info.object_key,
        mime_type=info.mime_type,
        size_bytes=info.size_bytes,
        sha256=info.sha256,
        public_url=object_store.public_url_for_library_preview(str(item_id)),
        metadata={"filename": "preview.png"},
    )


def _office_preview(
    *,
    item_id: UUID,
    job: Dict[str, Any],
    source_path: Path,
    temp_dir: Path,
    repository: Repository,
    object_store: ObjectStore,
    settings: Settings,
) -> Dict[str, Any]:
    if not shutil.which(settings.libreoffice_bin):
        raise RuntimeError("LibreOffice is not available for Office preview generation.")

    out_dir = temp_dir / "preview-office"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        settings.libreoffice_bin,
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(source_path),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=300)
    pdfs = sorted(out_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError("LibreOffice did not produce a PDF preview.")

    asset_id = uuid4()
    info = object_store.upload_file(
        path=pdfs[0],
        object_key=object_store.preview_key(library_item_id=str(item_id), filename="preview.pdf"),
        mime_type="application/pdf",
    )
    return repository.create_asset(
        asset_id=asset_id,
        library_item_id=item_id,
        batch_id=UUID(str(job["batch_id"])) if job.get("batch_id") else None,
        job_id=UUID(str(job["id"])),
        role="preview_pdf",
        bucket=info.bucket,
        object_key=info.object_key,
        mime_type=info.mime_type,
        size_bytes=info.size_bytes,
        sha256=info.sha256,
        public_url=object_store.public_url_for_library_preview(str(item_id)),
        metadata={"filename": "preview.pdf"},
    )
