from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID


@dataclass(frozen=True)
class SearchQuery:
    query: str
    limit: int = 20
    offset: int = 0
    detected_type: Optional[str] = None
    library_item_id: Optional[UUID] = None
    mode: str = "hybrid"


@dataclass(frozen=True)
class SearchHit:
    library_item_id: UUID
    job_id: UUID
    asset_id: UUID
    filename: str
    detected_type: Optional[str]
    score: float
    keyword_score: float
    semantic_score: float
    chunk_index: Optional[int]
    snippet: str
    markdown_url: str
    preview_url: str
    processed_at: Optional[datetime]


@dataclass(frozen=True)
class SearchResponse:
    query: str
    hits: list[SearchHit]
    limit: int
    offset: int
    total: int
