from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..equations import resolve_equation_file
from ..figures import resolve_figure_file
from ..tables import resolve_table_file
from .models import MarkdownExportConfig


BundlePayload = Dict[str, Any]


def prepare_asset_bundle(
    *,
    paper_id: int,
    bundle_dir: Path,
    figure_manifest: Dict[str, Any],
    table_manifest: Dict[str, Any],
    equation_manifest: Dict[str, Any],
    config: MarkdownExportConfig,
) -> BundlePayload:
    assets_dir = bundle_dir / "assets"
    figures_dir = assets_dir / "figures"
    tables_dir = assets_dir / "tables"
    equations_dir = assets_dir / "equations"
    if config.asset_mode == "copy":
        for path in (figures_dir, tables_dir, equations_dir):
            path.mkdir(parents=True, exist_ok=True)

    figures = _prepare_figure_assets(
        paper_id=paper_id,
        bundle_dir=bundle_dir,
        target_dir=figures_dir,
        records=figure_manifest.get("images") if isinstance(figure_manifest.get("images"), list) else [],
        config=config,
    )
    tables = _prepare_table_assets(
        paper_id=paper_id,
        bundle_dir=bundle_dir,
        target_dir=tables_dir,
        records=table_manifest.get("tables") if isinstance(table_manifest.get("tables"), list) else [],
        config=config,
    )
    equations = _prepare_equation_assets(
        paper_id=paper_id,
        bundle_dir=bundle_dir,
        target_dir=equations_dir,
        records=equation_manifest.get("equations") if isinstance(equation_manifest.get("equations"), list) else [],
        config=config,
    )

    return {
        "figures": figures,
        "tables": tables,
        "equations": equations,
        "asset_counts": {
            "figures": len(figures),
            "tables": len(tables),
            "equations": len(equations),
        },
    }


def _prepare_figure_assets(
    *,
    paper_id: int,
    bundle_dir: Path,
    target_dir: Path,
    records: Iterable[Dict[str, Any]],
    config: MarkdownExportConfig,
) -> list[Dict[str, Any]]:
    prepared: list[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file_name") or "").strip()
        if not file_name:
            continue
        source_path = resolve_figure_file(paper_id, file_name)
        if not source_path.exists():
            continue
        markdown_path = _materialize_asset_path(
            source_path=source_path,
            bundle_dir=bundle_dir,
            target_path=target_dir / file_name,
            config=config,
        )
        prepared.append({**item, "markdown_path": markdown_path})
    return prepared


def _prepare_table_assets(
    *,
    paper_id: int,
    bundle_dir: Path,
    target_dir: Path,
    records: Iterable[Dict[str, Any]],
    config: MarkdownExportConfig,
) -> list[Dict[str, Any]]:
    prepared: list[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        json_file = str(item.get("json_file") or "").strip()
        if not json_file:
            continue
        source_path = resolve_table_file(paper_id, json_file)
        if not source_path.exists():
            continue
        markdown_json_path = _materialize_asset_path(
            source_path=source_path,
            bundle_dir=bundle_dir,
            target_path=target_dir / json_file,
            config=config,
        )
        prepared.append({**item, "markdown_json_path": markdown_json_path})
    return prepared


def _prepare_equation_assets(
    *,
    paper_id: int,
    bundle_dir: Path,
    target_dir: Path,
    records: Iterable[Dict[str, Any]],
    config: MarkdownExportConfig,
) -> list[Dict[str, Any]]:
    prepared: list[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file_name") or "").strip()
        json_file = str(item.get("json_file") or "").strip()
        image_markdown_path: Optional[str] = None
        json_markdown_path: Optional[str] = None
        if file_name:
            source_image_path = resolve_equation_file(paper_id, file_name)
            if source_image_path.exists():
                image_markdown_path = _materialize_asset_path(
                    source_path=source_image_path,
                    bundle_dir=bundle_dir,
                    target_path=target_dir / file_name,
                    config=config,
                )
        if json_file:
            source_json_path = resolve_equation_file(paper_id, json_file)
            if source_json_path.exists():
                json_markdown_path = _materialize_asset_path(
                    source_path=source_json_path,
                    bundle_dir=bundle_dir,
                    target_path=target_dir / json_file,
                    config=config,
                )
        if not image_markdown_path and not json_markdown_path:
            continue
        prepared.append(
            {
                **item,
                "markdown_image_path": image_markdown_path,
                "markdown_json_path": json_markdown_path,
                "latex": str(item.get("latex") or "").strip() or None,
                "latex_confidence": item.get("latex_confidence"),
                "latex_source": str(item.get("latex_source") or "").strip() or None,
                "latex_validation_flags": item.get("latex_validation_flags") if isinstance(item.get("latex_validation_flags"), list) else [],
                "render_mode": str(item.get("render_mode") or "").strip() or None,
            }
        )
    return prepared


def _materialize_asset_path(
    *,
    source_path: Path,
    bundle_dir: Path,
    target_path: Path,
    config: MarkdownExportConfig,
) -> str:
    resolved_source = source_path.expanduser().resolve()
    if config.asset_mode == "copy":
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved_source, target_path)
        resolved_target = target_path.expanduser().resolve()
        return _format_path(resolved_target, bundle_dir=bundle_dir, mode=config.asset_path_mode)
    return _format_path(resolved_source, bundle_dir=bundle_dir, mode=config.asset_path_mode)


def _format_path(path: Path, *, bundle_dir: Path, mode: str) -> str:
    resolved = path.expanduser().resolve()
    if mode == "absolute":
        return resolved.as_posix()
    bundle_root = bundle_dir.resolve()
    if _is_relative_to(resolved, bundle_root):
        return resolved.relative_to(bundle_root).as_posix()
    return Path(os.path.relpath(resolved, bundle_root)).as_posix()


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False
