from __future__ import annotations

import base64
import threading
from io import BytesIO
from typing import Optional

import httpx
from PIL import Image

from document_agent.config import Settings, get_settings
from document_agent.errors import DocumentAgentError


class OcrClient:
    _semaphore_lock = threading.Lock()
    _semaphores: dict[int, threading.BoundedSemaphore] = {}

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def ocr_image_bytes(self, image_bytes: bytes, *, page_num: int = 1) -> str:
        if not self.settings.ocr_server_url:
            raise DocumentAgentError(
                code="OCR_NOT_CONFIGURED",
                message="OCR_SERVER_URL is not configured.",
                retryable=False,
            )

        png_bytes = self._normalize_png(image_bytes)
        image_b64 = base64.b64encode(png_bytes).decode("ascii")
        prompt = (
            "Convert this document image into faithful Markdown for AI agents. "
            "Preserve reading order, headings, tables, math, lists, and visible text. "
            "Do not summarize. Do not invent content. Return Markdown only."
        )
        payload = {
            "model": self.settings.ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": self.settings.ocr_page_max_tokens,
            "temperature": 0.0,
        }
        headers = {}
        if self.settings.ocr_api_key:
            headers["Authorization"] = f"Bearer {self.settings.ocr_api_key}"
        url = f"{self.settings.ocr_server_url.rstrip('/')}/chat/completions"
        try:
            with self._ocr_semaphore():
                response = httpx.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"] or "").strip()
        except httpx.HTTPError as exc:
            raise DocumentAgentError(
                code="OCR_REQUEST_FAILED",
                message=str(exc),
                retryable=True,
                details={"page_num": page_num},
            ) from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise DocumentAgentError(
                code="OCR_BAD_RESPONSE",
                message="OCR endpoint returned an unexpected response shape.",
                retryable=True,
                details={"page_num": page_num},
            ) from exc

    def _ocr_semaphore(self) -> threading.BoundedSemaphore:
        limit = max(1, int(self.settings.ocr_max_concurrent_requests))
        with self._semaphore_lock:
            semaphore = self._semaphores.get(limit)
            if semaphore is None:
                semaphore = threading.BoundedSemaphore(limit)
                self._semaphores[limit] = semaphore
            return semaphore

    def _normalize_png(self, image_bytes: bytes) -> bytes:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except Exception:
            pass
        with Image.open(BytesIO(image_bytes)) as image:
            try:
                from PIL import ImageOps

                image = ImageOps.exif_transpose(image)
            except Exception:
                pass
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            out = BytesIO()
            image.save(out, format="PNG")
            return out.getvalue()
