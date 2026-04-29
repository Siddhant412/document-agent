from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from ia_phase1.markdown_export import MarkdownExportConfig, export_pdf_to_markdown


_PAGE_MARKER_RE = re.compile(r"^<!-- page:(?P<page_no>\d+) -->\s*$")


@dataclass(slots=True)
class IaPhase1BridgeResult:
    page_segments: Dict[int, str]
    asset_counts: Dict[str, int]
    render_mode: str
    sectioning_strategy: str
    document_title: str


def _stable_bridge_paper_id(pdf_path: Path) -> int:
    hasher = hashlib.blake2b(digest_size=8)
    with pdf_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    raw = int.from_bytes(hasher.digest(), "big")
    return 600_000_000_000 + (raw % 300_000_000_000)


def _rewrite_asset_paths(markdown: str, *, pdf_name: str) -> str:
    rewritten = str(markdown or "")
    rewritten = rewritten.replace("assets/figures/", f"{pdf_name}_assets/figures/")
    rewritten = rewritten.replace("assets/tables/", f"{pdf_name}_assets/tables/")
    rewritten = rewritten.replace("assets/equations/", f"{pdf_name}_assets/equations/")
    return rewritten


def _rewrite_asset_paths_with_map(markdown: str, *, path_map: Dict[str, str]) -> str:
    rewritten = str(markdown or "")
    for original, replacement in sorted(path_map.items(), key=lambda item: len(item[0]), reverse=True):
        rewritten = rewritten.replace(original, replacement)
    return rewritten


def _split_page_segments(markdown: str) -> Dict[int, str]:
    segments: Dict[int, str] = {}
    current_page: Optional[int] = None
    current_lines: list[str] = []
    preamble: list[str] = []

    def _flush(page_no: Optional[int], lines: list[str]) -> None:
        if page_no is None:
            return
        payload = "\n".join(lines).strip()
        segments[page_no] = payload

    for line in str(markdown or "").splitlines():
        marker = _PAGE_MARKER_RE.match(line.strip())
        if marker:
            if current_page is None:
                preamble = current_lines[:]
            else:
                _flush(current_page, current_lines)
            current_page = int(marker.group("page_no"))
            current_lines = []
            continue
        current_lines.append(line)

    if current_page is not None:
        _flush(current_page, current_lines)

    preamble_text = "\n".join(preamble).strip()
    if preamble_text and segments:
        first_page = min(segments)
        segments[first_page] = (preamble_text + "\n\n" + segments[first_page]).strip()

    return {page_no: text for page_no, text in segments.items() if text}


def _resolve_ia_destination(target_dir: Path, name: str) -> str:
    candidate = name
    destination = target_dir / candidate
    if not destination.exists():
        return candidate

    item_path = Path(name)
    stem = item_path.stem
    suffix = item_path.suffix
    candidate = f"ia_{name}"
    destination = target_dir / candidate
    if not destination.exists():
        return candidate

    counter = 2
    while True:
        candidate = f"ia_{stem}_{counter}{suffix}"
        destination = target_dir / candidate
        if not destination.exists():
            return candidate
        counter += 1


def _copy_tree_contents_with_map(source_dir: Path, target_dir: Path, *, asset_group: str, pdf_name: str) -> Dict[str, str]:
    path_map: Dict[str, str] = {}
    if not source_dir.exists():
        return path_map
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        destination_name = _resolve_ia_destination(target_dir, item.name)
        destination = target_dir / destination_name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)
        path_map[f"assets/{asset_group}/{item.name}"] = f"{pdf_name}_assets/{asset_group}/{destination_name}"
    return path_map


class IaPhase1Bridge:
    def __init__(
        self,
        *,
        pdf_path: str | Path,
        pdf_name: str,
        assets_root: str | Path,
    ) -> None:
        self.pdf_path = Path(pdf_path).expanduser().resolve()
        self.pdf_name = str(pdf_name)
        self.assets_root = Path(assets_root).expanduser().resolve()

    def extract_non_ocr_pages(
        self,
        *,
        page_allowlist: Sequence[int],
    ) -> IaPhase1BridgeResult:
        allowed_pages = []
        seen_pages = set()
        for value in page_allowlist:
            try:
                page_no = int(value)
            except (TypeError, ValueError):
                continue
            if page_no <= 0 or page_no in seen_pages:
                continue
            seen_pages.add(page_no)
            allowed_pages.append(page_no)
        allowed_pages.sort()
        if not allowed_pages:
            return IaPhase1BridgeResult(
                page_segments={},
                asset_counts={"figures": 0, "tables": 0, "equations": 0},
                render_mode="normal",
                sectioning_strategy="none",
                document_title=self.pdf_name,
            )

        paper_id = _stable_bridge_paper_id(self.pdf_path)
        with tempfile.TemporaryDirectory(prefix=f"{self.pdf_name}_ia_phase1_") as temp_dir:
            bundle_dir = Path(temp_dir) / "bundle"
            result = export_pdf_to_markdown(
                self.pdf_path,
                paper_id=paper_id,
                output_dir=bundle_dir,
                page_allowlist=allowed_pages,
                config=MarkdownExportConfig(
                    asset_mode="copy",
                    asset_path_mode="relative",
                    include_frontmatter=False,
                    include_page_markers=True,
                    ensure_assets=True,
                    overwrite=True,
                    quality_audit_enabled=True,
                    conservative_fallback=True,
                ),
            )

            bundle_assets_dir = result.bundle_dir / "assets"
            path_map: Dict[str, str] = {}
            path_map.update(
                _copy_tree_contents_with_map(
                    bundle_assets_dir / "figures",
                    self.assets_root / "figures",
                    asset_group="figures",
                    pdf_name=self.pdf_name,
                )
            )
            path_map.update(
                _copy_tree_contents_with_map(
                    bundle_assets_dir / "tables",
                    self.assets_root / "tables",
                    asset_group="tables",
                    pdf_name=self.pdf_name,
                )
            )
            path_map.update(
                _copy_tree_contents_with_map(
                    bundle_assets_dir / "equations",
                    self.assets_root / "equations",
                    asset_group="equations",
                    pdf_name=self.pdf_name,
                )
            )

            rewritten_markdown = _rewrite_asset_paths_with_map(
                result.markdown,
                path_map=path_map or {
                    f"assets/{group}/": f"{self.pdf_name}_assets/{group}/"
                    for group in ("figures", "tables", "equations")
                },
            )
            page_segments = _split_page_segments(rewritten_markdown)

        return IaPhase1BridgeResult(
            page_segments=page_segments,
            asset_counts=dict(result.asset_counts),
            render_mode=str(result.render_mode or "normal"),
            sectioning_strategy=str(result.sectioning_strategy or "unknown"),
            document_title=str((getattr(result, "metadata", {}) or {}).get("title") or self.pdf_name),
        )
