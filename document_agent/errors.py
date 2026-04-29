from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DocumentAgentError(Exception):
    code: str
    message: str
    retryable: bool = False
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def error_from_exception(exc: BaseException) -> DocumentAgentError:
    if isinstance(exc, DocumentAgentError):
        return exc
    return DocumentAgentError(
        code="INTERNAL_ERROR",
        message=str(exc) or exc.__class__.__name__,
        retryable=False,
        details={"exception_type": exc.__class__.__name__},
    )

