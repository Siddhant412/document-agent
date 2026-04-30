from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Dict
from uuid import uuid4

from document_agent.converters.base import ConversionContext, UploadedAsset
from document_agent.utils import safe_filename

_MD_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def upload_asset_file(
    context: ConversionContext,
    *,
    path: Path,
    role: str,
    mime_type: str | None = None,
) -> UploadedAsset:
    asset_id = uuid4()
    resolved_mime = mime_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    object_key = context.object_store.job_asset_key(
        job_id=str(context.job_id),
        role=role,
        asset_id=str(asset_id),
        filename=path.name,
    )
    info = context.object_store.upload_file(
        path=path,
        object_key=object_key,
        mime_type=resolved_mime,
    )
    public_url = context.object_store.public_url_for_asset(str(asset_id))
    row = context.repository.create_asset(
        asset_id=asset_id,
        library_item_id=context.library_item_id,
        batch_id=context.batch_id,
        job_id=context.job_id,
        role=role,
        bucket=info.bucket,
        object_key=info.object_key,
        mime_type=info.mime_type,
        size_bytes=info.size_bytes,
        sha256=info.sha256,
        public_url=public_url,
        metadata={"original_filename": path.name},
    )
    return UploadedAsset(
        asset_id=row["id"],
        role=role,
        original_path=path,
        public_url=public_url,
        mime_type=info.mime_type,
        size_bytes=info.size_bytes,
    )


def upload_assets_and_rewrite_markdown(
    context: ConversionContext,
    *,
    markdown: str,
    assets_root: Path,
) -> tuple[str, list[UploadedAsset]]:
    if not assets_root.exists():
        return markdown, []

    files = [path for path in assets_root.rglob("*") if path.is_file()]
    referenced_paths = _markdown_image_paths(markdown)
    replacements: Dict[str, str] = {}
    uploaded: list[UploadedAsset] = []
    for path in files:
        role = _role_for_path(path)
        aliases = _path_aliases(path=path, assets_root=assets_root)
        if role in {"embedded_image", "equation_image"} and not referenced_paths.intersection(aliases):
            continue
        item = upload_asset_file(context, path=path, role=role)
        uploaded.append(item)
        for alias in aliases:
            replacements[alias] = item.public_url

    rewritten = markdown
    for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        rewritten = rewritten.replace(old, new)
    return rewritten, uploaded


def rewrite_markdown_image_paths(markdown: str, path_to_url: Dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        raw_path = match.group("path").strip()
        replacement = path_to_url.get(raw_path) or path_to_url.get(Path(raw_path).name)
        if not replacement:
            return match.group(0)
        return f"![{match.group('alt')}]({replacement})"

    return _MD_IMAGE_RE.sub(repl, markdown)


def _markdown_image_paths(markdown: str) -> set[str]:
    return {match.group("path").strip() for match in _MD_IMAGE_RE.finditer(markdown)}


def _path_aliases(*, path: Path, assets_root: Path) -> set[str]:
    rel = path.relative_to(assets_root).as_posix()
    return {
        rel,
        f"{assets_root.name}/{rel}",
        path.name,
        path.as_posix(),
    }


def _role_for_path(path: Path) -> str:
    parts = {safe_filename(part).lower() for part in path.parts}
    suffix = path.suffix.lower()
    if "equations" in parts and suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "equation_image"
    if suffix == ".json":
        return "diagnostic_manifest"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        return "embedded_image"
    return "diagnostic_manifest"
