from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from uuid import UUID

from document_agent.config import Settings, get_settings
from document_agent.converters.base import ConversionContext, ConversionResult, Converter
from document_agent.converters.detect import detect_file_type
from document_agent.converters.image import ImageConverter
from document_agent.converters.office import OfficeConverter
from document_agent.converters.pdf import PdfConverter
from document_agent.converters.text import TextConverter
from document_agent.db.repository import Repository
from document_agent.errors import DocumentAgentError
from document_agent.storage import ObjectStore


class ConversionPipeline:
    def __init__(
        self,
        *,
        repository: Repository,
        object_store: ObjectStore,
        settings: Optional[Settings] = None,
    ) -> None:
        self.repository = repository
        self.object_store = object_store
        self.settings = settings or get_settings()
        self._converters: list[Converter] = [
            TextConverter(),
            ImageConverter(),
            PdfConverter(),
            OfficeConverter(),
        ]

    def detect(self, *, path: Path, filename: str, content_type: Optional[str]) -> str:
        return detect_file_type(path, filename, content_type)

    def convert(
        self,
        *,
        job_id: UUID,
        batch_id: Optional[UUID],
        input_index: Optional[int],
        source_path: Path,
        filename: str,
        detected_type: str,
        content_type: Optional[str],
        temp_dir: Path,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ConversionResult:
        metadata = dict(metadata or {})
        if detected_type == "unsupported":
            raise DocumentAgentError(
                code=str(metadata.get("detection_error_code") or "UNSUPPORTED_FILE_TYPE"),
                message=str(
                    metadata.get("detection_error_message")
                    or f"Unsupported file type for {filename!r}."
                ),
                retryable=False,
                details=metadata.get("detection_error_details")
                if isinstance(metadata.get("detection_error_details"), dict)
                else {"filename": filename},
            )
        converter = self._find_converter(detected_type)
        context = ConversionContext(
            job_id=job_id,
            batch_id=batch_id,
            input_index=input_index,
            source_path=source_path,
            filename=filename,
            detected_type=detected_type,
            content_type=content_type,
            temp_dir=temp_dir,
            repository=self.repository,
            object_store=self.object_store,
            settings=self.settings,
            metadata=metadata,
        )
        return converter.convert(context)

    def _find_converter(self, detected_type: str) -> Converter:
        for converter in self._converters:
            if detected_type in converter.detected_types:
                return converter
        raise DocumentAgentError(
            code="UNSUPPORTED_FILE_TYPE",
            message=f"No converter registered for detected type {detected_type!r}.",
            retryable=False,
        )
