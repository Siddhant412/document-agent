from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

MarkdownAssetKind = Literal["figure", "table", "equation"]


@dataclass(slots=True)
class MarkdownParagraphNode:
    text: str
    page_no: int


@dataclass(slots=True)
class MarkdownSubheadingNode:
    title: str
    level: int
    page_no: int


@dataclass(slots=True)
class MarkdownAssetNode:
    kind: MarkdownAssetKind
    record: Dict[str, Any]
    page_no: int


MarkdownSectionElement = Union[MarkdownParagraphNode, MarkdownSubheadingNode, MarkdownAssetNode]


@dataclass(slots=True)
class MarkdownSectionNode:
    canonical: str
    title: str
    level: int
    page_start: int = 0
    page_end: int = 0
    elements: List[MarkdownSectionElement] = field(default_factory=list)


@dataclass(slots=True)
class MarkdownDocumentModel:
    front_matter: List[str] = field(default_factory=list)
    front_matter_pages: List[int] = field(default_factory=list)
    sections: List[MarkdownSectionNode] = field(default_factory=list)
    inferred_title: Optional[str] = None
