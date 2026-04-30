from __future__ import annotations

import base64
import re
import threading
import time
from io import BytesIO
from typing import Any, Optional

import httpx
from PIL import Image

from document_agent.config import Settings, get_settings
from document_agent.errors import DocumentAgentError
from document_agent.metrics import OCR_REQUEST_DURATION_SECONDS, OCR_REQUESTS


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
        url = self.chat_completions_url()
        started = time.monotonic()
        try:
            with self._ocr_semaphore():
                response = httpx.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
            content = _strip_outer_markdown_fence(_extract_message_content(data))
            OCR_REQUESTS.labels(
                status="succeeded",
                model=self.settings.ocr_model,
                error_code="",
            ).inc()
            OCR_REQUEST_DURATION_SECONDS.labels(
                status="succeeded",
                model=self.settings.ocr_model,
            ).observe(time.monotonic() - started)
            return content
        except httpx.HTTPError as exc:
            OCR_REQUESTS.labels(
                status="failed",
                model=self.settings.ocr_model,
                error_code="OCR_REQUEST_FAILED",
            ).inc()
            OCR_REQUEST_DURATION_SECONDS.labels(
                status="failed",
                model=self.settings.ocr_model,
            ).observe(time.monotonic() - started)
            details = {"page_num": page_num}
            if isinstance(exc, httpx.HTTPStatusError):
                details["status_code"] = exc.response.status_code
                details["response_preview"] = exc.response.text[:1000]
            raise DocumentAgentError(
                code="OCR_REQUEST_FAILED",
                message=str(exc),
                retryable=True,
                details=details,
            ) from exc
        except (KeyError, IndexError, TypeError) as exc:
            OCR_REQUESTS.labels(
                status="failed",
                model=self.settings.ocr_model,
                error_code="OCR_BAD_RESPONSE",
            ).inc()
            OCR_REQUEST_DURATION_SECONDS.labels(
                status="failed",
                model=self.settings.ocr_model,
            ).observe(time.monotonic() - started)
            raise DocumentAgentError(
                code="OCR_BAD_RESPONSE",
                message="OCR endpoint returned an unexpected response shape.",
                retryable=True,
                details={"page_num": page_num},
            ) from exc

    def chat_completions_url(self) -> str:
        base = (self.settings.ocr_server_url or "").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

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


def _extract_message_content(data: dict[str, Any]) -> str:
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()
    if content is None:
        return ""
    return str(content).strip()


def _strip_outer_markdown_fence(content: str) -> str:
    match = re.fullmatch(r"\s*```(?:markdown|md|text)?\s*\n(?P<body>.*?)\n```\s*", content, re.DOTALL | re.IGNORECASE)
    if not match:
        return content.strip()
    return match.group("body").strip()
