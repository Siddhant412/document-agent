from __future__ import annotations

import threading
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastembed import TextEmbedding

from document_agent.config import Settings, get_settings


class EmbeddingService:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.model = self._load_model()
        self.dimension = self.settings.search_embedding_dimension
        self._lock = threading.RLock()
        self._query_cache: "OrderedDict[str, list[float]]" = OrderedDict()
        self._query_cache_size = 1024

    def _load_model(self) -> TextEmbedding:
        try:
            return TextEmbedding(
                model_name=self.settings.search_embedding_model,
                cache_dir=self.settings.search_embedding_cache_dir,
            )
        except Exception as exc:
            if not _looks_like_partial_cache_error(exc):
                raise
            shutil.rmtree(Path(self.settings.search_embedding_cache_dir), ignore_errors=True)
            return TextEmbedding(
                model_name=self.settings.search_embedding_model,
                cache_dir=self.settings.search_embedding_cache_dir,
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._normalize_embedding(embedding) for embedding in self.model.embed(texts)]

    def embed_query(self, query: str) -> list[float]:
        key = " ".join(query.split())
        with self._lock:
            cached = self._query_cache.get(key)
            if cached is not None:
                self._query_cache.move_to_end(key)
                return list(cached)
        embedding = self._normalize_embedding(next(self.model.query_embed(query)))
        with self._lock:
            self._query_cache[key] = list(embedding)
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)
        return embedding

    def _normalize_embedding(self, embedding: object) -> list[float]:
        vector = [float(value) for value in embedding]
        if len(vector) != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"model returned {len(vector)}, "
                f"SEARCH_EMBEDDING_DIMENSION is {self.dimension}"
            )
        return vector


_embedding_service: Optional[EmbeddingService] = None
_embedding_service_lock = threading.Lock()


def get_embedding_service(settings: Optional[Settings] = None) -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        with _embedding_service_lock:
            if _embedding_service is None:
                _embedding_service = EmbeddingService(settings)
    return _embedding_service


def _looks_like_partial_cache_error(exc: Exception) -> bool:
    message = str(exc)
    return "NO_SUCHFILE" in message or "File doesn't exist" in message
