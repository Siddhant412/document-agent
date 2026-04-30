from __future__ import annotations

from typing import Any, Iterable, Mapping

DOCUMENT_ASSET_ROLES = {"embedded_image", "equation_image"}


def count_document_assets(assets: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for asset in assets if asset.get("role") in DOCUMENT_ASSET_ROLES)
