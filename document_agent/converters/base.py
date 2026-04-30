from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

if TYPE_CHECKING:
    from document_agent.config import Settings
    from document_agent.db.repository import Repository
    from document_agent.storage import ObjectStore


@dataclass
class UploadedAsset:
    asset_id: UUID
    role: str
    original_path: Path
    public_url: str
    mime_type: str
    size_bytes: int


@dataclass
class ConversionContext:
    job_id: UUID
    batch_id: Optional[UUID]
    input_index: Optional[int]
    source_path: Path
    filename: str
    detected_type: str
    content_type: Optional[str]
    temp_dir: Path
    repository: "Repository"
    object_store: "ObjectStore"
    settings: "Settings"
    library_item_id: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionResult:
    markdown: str
    detected_type: str
    asset_count: int = 0
    assets: List[UploadedAsset] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Converter:
    detected_types: set[str] = set()

    def convert(self, context: ConversionContext) -> ConversionResult:
        raise NotImplementedError
