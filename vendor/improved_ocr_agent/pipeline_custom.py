import argparse
import asyncio
import atexit
import base64
import datetime
import hashlib
import json
import logging
import multiprocessing
import os
import random
import re
import shutil
import ssl
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from io import BytesIO
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
from urllib.parse import urlparse
import subprocess

try:
    from lingua import Language, LanguageDetectorBuilder
except ImportError:
    Language = None
    LanguageDetectorBuilder = None

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None
    ClientError = Exception

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

import httpx
import fitz
from PIL import Image
from pypdf import PdfReader
from tqdm import tqdm

from .renderpdf import render_pdf_to_base64png
from .image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png
from .metrics import MetricsKeeper, WorkerTracker
from .anchor import get_anchor_text
from .ocr_prompt import PageResponse, build_custom_yaml_prompt
from .front_matter import FrontMatterParser
from .work_queue import LocalBackend, WorkQueue


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

server_logger = logging.getLogger("vllm")
server_logger.propagate = False

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(console_handler)
server_logger.addHandler(console_handler)

logging.getLogger("pypdf").setLevel(logging.ERROR)

workspace_s3 = boto3.client("s3") if boto3 is not None else None
pdf_s3 = boto3.client("s3") if boto3 is not None else None

metrics = MetricsKeeper(window=60 * 5)
tracker = WorkerTracker()
vllm_queued_requests = None

TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]

pdf_render_max_workers_limit = asyncio.BoundedSemaphore(
    int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", max(1, multiprocessing.cpu_count() - 2))))
)
max_concurrent_requests_limit = asyncio.BoundedSemaphore(1)


class PdfFilter:
    def __init__(
        self,
        languages_to_keep=None,
        apply_form_check=True,
        apply_download_spam_check=True,
        download_spam_threshold=0.004,
    ):
        super().__init__()
        self.language_detector = (
            LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
            if LanguageDetectorBuilder is not None
            else None
        )
        self.languages_to_keep = languages_to_keep if languages_to_keep is not None else [Language.ENGLISH] if Language else [None]
        self.apply_form_check = apply_form_check
        self.apply_download_spam_check = apply_download_spam_check
        self.download_spam_threshold = download_spam_threshold

    def _is_form(self, pdf_reader) -> bool:
        if pdf_reader.get_form_text_fields():
            return True
        return False

    def _is_download_spam(self, base_text: str) -> bool:
        seo_words = {
            "download",
            "pdf",
            "epub",
            "mobi",
            "free",
            "ebook",
            "file",
            "save",
            "casino",
            "viagra",
            "cialis",
            "ciprofloxacin",
        }

        base_text = base_text.strip().lower()
        clean_text = re.sub(r"\W+", " ", base_text)

        word_counts = Counter(clean_text.split())
        total_words = len(clean_text.split())

        if total_words == 0:
            return False

        seo_score = sum(word_counts[word] for word in seo_words if word in word_counts)
        return (seo_score / total_words) > self.download_spam_threshold

    def filter_out_pdf(self, local_pdf_path: str) -> bool:
        try:
            pdf_reader = PdfReader(local_pdf_path)
            if self.apply_form_check and self._is_form(pdf_reader):
                logger.info(f"Filtering out {local_pdf_path} because it's a form")
                return True
        except Exception as e:
            logger.warning(f"Error reading PDF {local_pdf_path}: {e}")
            return True

        pdftotext_result = subprocess.run(
            ["pdftotext", "-f", "1", "-l", "5", local_pdf_path, "-"],
            timeout=60,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if pdftotext_result.returncode != 0:
            logger.warning(f"pdftotext returned {pdftotext_result.returncode} on {local_pdf_path}")
            return True

        base_text = pdftotext_result.stdout.decode("utf-8")
        alpha_count = sum(c.isalpha() for c in base_text)

        if len(base_text) < 200:
            logger.info(f"Keeping {local_pdf_path} because not enough text exists to analyze")
            return False

        if alpha_count / len(base_text) < 0.50:
            logger.info(f"Keeping {local_pdf_path} because text may be OCRed badly")
            return False

        if self.language_detector is not None:
            language = self.language_detector.detect_language_of(base_text)
            if language not in self.languages_to_keep:
                logger.info(f"Filtering out {local_pdf_path} because language was {language}")
                return True

        if self.apply_download_spam_check and self._is_download_spam(base_text):
            logger.info(f"Filtering out {local_pdf_path} because of SEO/download spam")
            return True

        return False


get_pdf_filter = cache(
    lambda: PdfFilter(
        languages_to_keep={Language.ENGLISH, None} if Language is not None else {None},
        apply_download_spam_check=True,
        apply_form_check=True,
    )
)


@dataclass(frozen=True)
class PageResult:
    s3_path: str
    page_num: int
    response: PageResponse
    input_tokens: int
    output_tokens: int
    is_fallback: bool
    is_valid: bool


FIGURE_REF_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<name>page_(?P<x0>\d+)_(?P<y0>\d+)_(?P<x1>\d+)_(?P<y1>\d+)\.png)\)"
)


def make_ocr_args(
    server: str,
    model: str,
    workspace: str = "./workspace",
    api_key: str | None = None,
    guided_decoding: bool = False,
    page_max_tokens: int = 8000,
    target_longest_image_dim: int = 1288,
    table_format: str = "html",
    emit_figure_placeholders: bool = True,
    save_rendered_pages: bool = False,
    materialize_assets: bool = False,
    embed_page_markers: bool = False,
    max_page_retries: int = 8,
    max_page_error_rate: float = 0.004,
    max_concurrent_requests: int = 16,
    workers: int = 1,
):
    return SimpleNamespace(
        server=server,
        model=model,
        workspace=workspace,
        api_key=api_key,
        guided_decoding=guided_decoding,
        page_max_tokens=page_max_tokens,
        target_longest_image_dim=target_longest_image_dim,
        table_format=table_format,
        emit_figure_placeholders=emit_figure_placeholders,
        save_rendered_pages=save_rendered_pages,
        materialize_assets=materialize_assets,
        embed_page_markers=embed_page_markers,
        max_page_retries=max_page_retries,
        max_page_error_rate=max_page_error_rate,
        max_concurrent_requests=max_concurrent_requests,
        workers=workers,
        apply_filter=False,
        markdown=False,
        stats=False,
        max_server_ready_timeout=600,
        workspace_profile=None,
        pdf_profile=None,
        gpu_memory_utilization=None,
        max_model_len=16384,
        tensor_parallel_size=1,
        data_parallel_size=1,
        port=30024,
        disk_logging=None,
    )


def get_markdown_path(workspace: str, source_file: str) -> str:
    if "::" in source_file:
        tarball_path, internal_path = source_file.split("::", 1)
        tarball_basename = os.path.splitext(os.path.basename(tarball_path))[0]
        if tarball_basename.endswith(".tar"):
            tarball_basename = tarball_basename[:-4]
        relative_path = os.path.join(tarball_basename, internal_path)
    elif source_file.startswith("s3://"):
        parsed = urlparse(source_file)
        relative_path = parsed.path.lstrip("/")
    else:
        relative_path = source_file.lstrip("/")

    parts = relative_path.split("/")
    safe_parts = [p for p in parts if p and p != ".."]
    relative_path = "/".join(safe_parts)

    md_filename = os.path.splitext(os.path.basename(relative_path))[0] + ".md"
    dir_path = os.path.dirname(relative_path)
    markdown_dir = os.path.join(workspace, "markdown", dir_path)
    return os.path.join(markdown_dir, md_filename)


def get_rendered_page_dir(workspace: str, source_file: str) -> str:
    md_path = get_markdown_path(workspace, source_file)
    stem = Path(md_path).stem
    return str(Path(md_path).parent / f"{stem}_assets" / "pages")


def get_cropped_image_dir(workspace: str, source_file: str) -> str:
    md_path = get_markdown_path(workspace, source_file)
    stem = Path(md_path).stem
    return str(Path(md_path).parent / f"{stem}_assets" / "images")


def get_rendered_page_path(workspace: str, source_file: str, page_num: int) -> str:
    return str(Path(get_rendered_page_dir(workspace, source_file)) / f"page_{page_num:04d}.png")


def render_pdf_page_to_png_bytes(local_pdf_path: str, page_num: int, target_longest_image_dim: int) -> bytes:
    image_base64 = render_pdf_to_base64png(local_pdf_path, page_num, target_longest_image_dim=target_longest_image_dim)
    return base64.b64decode(image_base64)


async def render_and_optionally_save_page_image(
    args,
    local_pdf_path: str,
    pdf_orig_path: str,
    page_num: int,
    image_rotation: int = 0,
) -> str:
    async with pdf_render_max_workers_limit:
        image_bytes = await asyncio.to_thread(
            render_pdf_page_to_png_bytes,
            local_pdf_path,
            page_num,
            args.target_longest_image_dim,
        )

    with Image.open(BytesIO(image_bytes)) as img:
        if image_rotation != 0:
            if image_rotation == 90:
                transpose = Image.Transpose.ROTATE_90
            elif image_rotation == 180:
                transpose = Image.Transpose.ROTATE_180
            elif image_rotation == 270:
                transpose = Image.Transpose.ROTATE_270
            else:
                raise ValueError("Invalid rotation")
            img = img.transpose(transpose)

        if getattr(args, "save_rendered_pages", False) and not args.workspace.startswith("s3://"):
            out_path = get_rendered_page_path(args.workspace, pdf_orig_path, page_num)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            img.save(out_path, format="PNG")

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


async def build_page_query(local_pdf_path: str, pdf_orig_path: str, page: int, args, image_rotation: int = 0, model_name: str = "olmocr") -> dict:
    max_tokens = args.page_max_tokens
    assert image_rotation in [0, 90, 180, 270]

    image_base64 = await render_and_optionally_save_page_image(
        args=args,
        local_pdf_path=local_pdf_path,
        pdf_orig_path=pdf_orig_path,
        page_num=page,
        image_rotation=image_rotation,
    )

    prompt = build_custom_yaml_prompt(
        emit_figure_placeholders=args.emit_figure_placeholders,
        table_format=args.table_format,
    )

    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


async def build_image_query_from_bytes(
    image_bytes: bytes,
    args,
    model_name: str = "olmocr",
    image_rotation: int = 0,
) -> dict:
    max_tokens = args.page_max_tokens

    with Image.open(BytesIO(image_bytes)) as img:
        if image_rotation != 0:
            if image_rotation == 90:
                transpose = Image.Transpose.ROTATE_90
            elif image_rotation == 180:
                transpose = Image.Transpose.ROTATE_180
            elif image_rotation == 270:
                transpose = Image.Transpose.ROTATE_270
            else:
                raise ValueError("Invalid image_rotation")
            img = img.transpose(transpose)

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = build_custom_yaml_prompt(
        emit_figure_placeholders=args.emit_figure_placeholders,
        table_format=args.table_format,
    )

    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


async def apost(url, json_data, api_key=None):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    if parsed_url.scheme == "https":
        port = parsed_url.port or 443
        use_ssl = True
    else:
        port = parsed_url.port or 80
        use_ssl = False
    path = parsed_url.path or "/"

    writer = None
    try:
        if use_ssl:
            ssl_context = ssl.create_default_context()
            reader, writer = await asyncio.open_connection(host, port, ssl=ssl_context)
        else:
            reader, writer = await asyncio.open_connection(host, port)

        json_payload = json.dumps(json_data)
        headers = [
            f"POST {path} HTTP/1.1",
            f"Host: {host}",
            "Content-Type: application/json",
            f"Content-Length: {len(json_payload)}",
        ]
        if api_key:
            headers.append(f"Authorization: Bearer {api_key}")
        headers.append("Connection: close")

        request = "\r\n".join(headers) + "\r\n\r\n" + json_payload
        writer.write(request.encode())
        await writer.drain()

        status_line = await reader.readline()
        if not status_line:
            raise ConnectionError("No response from server")
        status_parts = status_line.decode().strip().split(" ", 2)
        if len(status_parts) < 2:
            raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
        status_code = int(status_parts[1])

        headers = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode().partition(":")
            headers[key.strip().lower()] = value.strip()

        if "content-length" in headers:
            body_length = int(headers["content-length"])
            response_body = await reader.readexactly(body_length)
        elif headers.get("transfer-encoding", "") == "chunked":
            chunks = []
            while True:
                size_line = await reader.readline()
                chunk_size = int(size_line.strip(), 16)
                if chunk_size == 0:
                    await reader.readline()
                    break
                chunk_data = await reader.readexactly(chunk_size)
                chunks.append(chunk_data)
                await reader.readline()
            response_body = b"".join(chunks)
        elif headers.get("connection", "") == "close":
            response_body = await reader.read()
        else:
            raise ConnectionError("Cannot determine response body length")

        return status_code, response_body
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


async def try_single_page(args, pdf_orig_path: str, pdf_local_path: str, page_num: int, attempt: int, rotation: int) -> PageResult | None:
    completion_url = f"{args.server.rstrip('/')}/chat/completions"
    model_max_context = 16384
    temp_idx = min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
    temperature = TEMPERATURE_BY_ATTEMPT[temp_idx]
    api_key = args.api_key if args.server and hasattr(args, "api_key") else None

    try:
        query = await build_page_query(
            pdf_local_path,
            pdf_orig_path,
            page_num,
            args,
            image_rotation=rotation,
            model_name=args.model,
        )
        query["temperature"] = temperature

        if getattr(args, "guided_decoding", False):
            query["guided_regex"] = (
                r"---\nprimary_language: (?:[a-z]{2}|null)\n"
                r"is_rotation_valid: (?:True|False|true|false)\n"
                r"rotation_correction: (?:0|90|180|270)\n"
                r"is_table: (?:True|False|true|false)\n"
                r"is_diagram: (?:True|False|true|false)\n"
                r"(?:---|---\n[\s\S]+)"
            )

        async with max_concurrent_requests_limit:
            status_code, response_body = await apost(completion_url, json_data=query, api_key=api_key)

        if status_code != 200:
            logger.warning(f"Server returned {status_code} for {pdf_orig_path}-{page_num} attempt {attempt}")
            return None

        base_response_data = json.loads(response_body)

        metrics.add_metrics(
            server_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
            server_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
        )

        is_valid = True
        if base_response_data["usage"]["total_tokens"] > model_max_context:
            is_valid = False
        if base_response_data["choices"][0]["finish_reason"] != "stop":
            is_valid = False

        model_response_markdown = base_response_data["choices"][0]["message"]["content"]
        parser = FrontMatterParser(front_matter_class=PageResponse)
        front_matter, text = parser._extract_front_matter_and_text(model_response_markdown)
        page_response = parser._parse_front_matter(front_matter, text)

        return PageResult(
            pdf_orig_path,
            page_num,
            page_response,
            input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
            output_tokens=base_response_data["usage"].get("completion_tokens", 0),
            is_fallback=False,
            is_valid=is_valid,
        )
    except asyncio.CancelledError:
        raise
    except (ConnectionError, OSError, asyncio.TimeoutError):
        raise
    except Exception as e:
        logger.warning(f"try_single_page failed for {pdf_orig_path}-{page_num} attempt {attempt}: {type(e).__name__}: {e}")
        return None


async def ocr_page_image(
    args,
    image_bytes: bytes,
    model_name: str | None = None,
    page_num: int | None = None,
    image_rotation: int = 0,
    attempt: int = 0,
) -> PageResult:
    completion_url = f"{args.server.rstrip('/')}/chat/completions"
    model_max_context = 16384
    use_model = model_name or args.model
    temp_idx = min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)
    temperature = TEMPERATURE_BY_ATTEMPT[temp_idx]
    api_key = args.api_key if args.server and hasattr(args, "api_key") else None

    query = await build_image_query_from_bytes(
        image_bytes=image_bytes,
        args=args,
        model_name=use_model,
        image_rotation=image_rotation,
    )
    query["temperature"] = temperature

    if getattr(args, "guided_decoding", False):
        query["guided_regex"] = (
            r"---\nprimary_language: (?:[a-z]{2}|null)\n"
            r"is_rotation_valid: (?:True|False|true|false)\n"
            r"rotation_correction: (?:0|90|180|270)\n"
            r"is_table: (?:True|False|true|false)\n"
            r"is_diagram: (?:True|False|true|false)\n"
            r"(?:---|---\n[\s\S]+)"
        )

    async with max_concurrent_requests_limit:
        status_code, response_body = await apost(completion_url, json_data=query, api_key=api_key)

    if status_code != 200:
        raise RuntimeError(f"OCR request failed with status {status_code}: {response_body[:500]!r}")

    base_response_data = json.loads(response_body)

    metrics.add_metrics(
        server_input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
        server_output_tokens=base_response_data["usage"].get("completion_tokens", 0),
    )

    is_valid = True
    if base_response_data["usage"]["total_tokens"] > model_max_context:
        is_valid = False
    if base_response_data["choices"][0]["finish_reason"] != "stop":
        is_valid = False

    model_response_markdown = base_response_data["choices"][0]["message"]["content"]
    parser = FrontMatterParser(front_matter_class=PageResponse)
    front_matter, text = parser._extract_front_matter_and_text(model_response_markdown)
    page_response = parser._parse_front_matter(front_matter, text)

    return PageResult(
        s3_path=f"image_page_{page_num}" if page_num is not None else "image_page",
        page_num=page_num or 1,
        response=page_response,
        input_tokens=base_response_data["usage"].get("prompt_tokens", 0),
        output_tokens=base_response_data["usage"].get("completion_tokens", 0),
        is_fallback=False,
        is_valid=is_valid,
    )


def make_fallback_result(pdf_orig_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    return PageResult(
        pdf_orig_path,
        page_num,
        PageResponse(
            natural_text=get_anchor_text(pdf_local_path, page_num, pdf_engine="pdftotext"),
            primary_language=None,
            is_rotation_valid=True,
            rotation_correction=0,
            is_table=False,
            is_diagram=False,
        ),
        input_tokens=0,
        output_tokens=0,
        is_fallback=True,
        is_valid=True,
    )


async def try_single_page_with_backoff(args, pdf_orig_path: str, pdf_local_path: str, page_num: int, attempt: int, rotation: int) -> PageResult | None:
    max_backoff_attempts = 10
    for backoff_count in range(max_backoff_attempts):
        try:
            return await try_single_page(args, pdf_orig_path, pdf_local_path, page_num, attempt, rotation)
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            sleep_delay = 10 * (2**backoff_count)
            logger.warning(f"Connection error on {pdf_orig_path}-{page_num} attempt {attempt}: {e}. sleeping {sleep_delay}s")
            await asyncio.sleep(sleep_delay)
    logger.error(f"Max backoff attempts reached for {pdf_orig_path}-{page_num}")
    sys.exit(1)


async def process_page(args, worker_id: int, pdf_orig_path: str, pdf_local_path: str, page_num: int) -> PageResult:
    max_retries = args.max_page_retries
    retry_attempts = list(range(1, max_retries))
    cumulative_rotation = 0

    await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "started")
    result = await try_single_page_with_backoff(args, pdf_orig_path, pdf_local_path, page_num, attempt=0, rotation=cumulative_rotation)

    if result is not None and not result.response.is_rotation_valid:
        cumulative_rotation = result.response.rotation_correction % 360

    if result is not None and result.is_valid and result.response.is_rotation_valid:
        metrics.add_metrics(**{"completed_pages": 1, "finished_on_attempt_0": 1})
        await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
        return result

    if result is not None and not result.response.is_rotation_valid:
        for attempt in retry_attempts:
            result = await try_single_page_with_backoff(args, pdf_orig_path, pdf_local_path, page_num, attempt, cumulative_rotation)
            if result is not None and result.is_valid and result.response.is_rotation_valid:
                metrics.add_metrics(**{"completed_pages": 1, f"finished_on_attempt_{attempt}": 1})
                await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
                return result
            if result is not None:
                cumulative_rotation = (cumulative_rotation + result.response.rotation_correction) % 360

        if result is not None and result.is_valid:
            metrics.add_metrics(**{"completed_pages": 1, f"finished_on_attempt_{max_retries}": 1})
            await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
            return result

        metrics.add_metrics(failed_pages=1)
        await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "errored")
        return make_fallback_result(pdf_orig_path, pdf_local_path, page_num)

    for i, attempt in enumerate(retry_attempts):
        result = await try_single_page_with_backoff(args, pdf_orig_path, pdf_local_path, page_num, attempt, rotation=cumulative_rotation)

        if result is not None and result.is_valid and result.response.is_rotation_valid:
            metrics.add_metrics(**{"completed_pages": 1, f"finished_on_attempt_{attempt}": 1})
            await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
            return result

        remaining_attempts = retry_attempts[i + 1 :]
        if remaining_attempts and vllm_queued_requests == 0:
            tasks = [
                asyncio.create_task(try_single_page_with_backoff(args, pdf_orig_path, pdf_local_path, page_num, a, rotation=cumulative_rotation))
                for a in remaining_attempts
            ]
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result is not None and result.is_valid and result.response.is_rotation_valid:
                        for t in tasks:
                            t.cancel()
                        metrics.add_metrics(**{"completed_pages": 1, "finished_on_parallel_retry": 1})
                        await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
                        return result
                except asyncio.CancelledError:
                    continue
            break

    if result is not None and result.is_valid:
        metrics.add_metrics(**{"completed_pages": 1, f"finished_on_attempt_{max_retries}": 1})
        await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "finished")
        return result

    metrics.add_metrics(failed_pages=1)
    await tracker.track_work(worker_id, f"{pdf_orig_path}-{page_num}", "errored")
    return make_fallback_result(pdf_orig_path, pdf_local_path, page_num)


async def ocr_page_pdf(
    args,
    pdf_path: str,
    page_num: int,
    pdf_orig_path: str | None = None,
) -> PageResult:
    pdf_orig_path = pdf_orig_path or pdf_path
    return await process_page(args, worker_id=0, pdf_orig_path=pdf_orig_path, pdf_local_path=pdf_path, page_num=page_num)


async def ocr_region(
    args,
    pdf_path: str,
    page_num: int,
    bbox: tuple[float, float, float, float],
    pdf_orig_path: str | None = None,
    padding: int = 10,
    dpi: int = 180,
) -> PageResult:
    pdf_orig_path = pdf_orig_path or pdf_path
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        rect = fitz.Rect(bbox)
        clip_rect = (rect + (-padding, -padding, padding, padding)).intersect(page.rect)

        if clip_rect.is_empty:
            raise ValueError(f"Empty clip rect for bbox={bbox} on page={page_num}")

        pix = page.get_pixmap(clip=clip_rect, dpi=dpi)
        image_bytes = pix.tobytes("png")

        result = await ocr_page_image(
            args=args,
            image_bytes=image_bytes,
            model_name=args.model,
            page_num=page_num,
        )
        return result
    finally:
        doc.close()


def is_tarball_path(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".tar.gz") or lower.endswith(".tgz")


def build_dolma_document(pdf_orig_path, page_results):
    document_text = ""
    pdf_page_spans = []
    current_char_pos = 0

    for index, page_result in enumerate(page_results):
        content = (page_result.response.natural_text or "") + ("\n" if index < len(page_results) - 1 else "")
        start_pos = current_char_pos
        document_text += content
        current_char_pos = len(document_text)
        pdf_page_spans.append([start_pos, current_char_pos, page_result.page_num])

    if not document_text:
        return None

    metadata = {
        "Source-File": pdf_orig_path,
        "pdf-total-pages": len(page_results),
        "total-input-tokens": sum(page.input_tokens for page in page_results),
        "total-output-tokens": sum(page.output_tokens for page in page_results),
        "total-fallback-pages": sum(page.is_fallback for page in page_results),
    }

    id_ = hashlib.sha1(document_text.encode()).hexdigest()

    return {
        "id": id_,
        "text": document_text,
        "source": "olmocr",
        "added": datetime.datetime.now().strftime("%Y-%m-%d"),
        "created": datetime.datetime.now().strftime("%Y-%m-%d"),
        "metadata": metadata,
        "attributes": {
            "pdf_page_numbers": pdf_page_spans,
            "primary_language": [p.response.primary_language for p in page_results],
            "is_rotation_valid": [p.response.is_rotation_valid for p in page_results],
            "rotation_correction": [p.response.rotation_correction for p in page_results],
            "is_table": [p.response.is_table for p in page_results],
            "is_diagram": [p.response.is_diagram for p in page_results],
        },
    }


def materialize_markdown_assets(markdown_text: str, rendered_page_path: str, crop_output_dir: str) -> str:
    os.makedirs(crop_output_dir, exist_ok=True)

    with Image.open(rendered_page_path) as page_img:
        width, height = page_img.size

        def repl(match: re.Match) -> str:
            alt = match.group("alt")
            x0 = int(match.group("x0"))
            y0 = int(match.group("y0"))
            x1 = int(match.group("x1"))
            y1 = int(match.group("y1"))

            x0c = max(0, min(x0, width))
            y0c = max(0, min(y0, height))
            x1c = max(0, min(x1, width))
            y1c = max(0, min(y1, height))

            if x1c <= x0c or y1c <= y0c:
                return match.group(0)

            filename = match.group("name")
            out_path = os.path.join(crop_output_dir, filename)
            cropped = page_img.crop((x0c, y0c, x1c, y1c))
            cropped.save(out_path, format="PNG")

            rel = os.path.relpath(out_path, start=os.path.dirname(rendered_page_path))
            rel = rel.replace("\\", "/")
            return f"![{alt}]({rel})"

        return FIGURE_REF_RE.sub(repl, markdown_text)


def rewrite_markdown_with_real_assets(args, source_file: str, markdown_text: str) -> str:
    if args.workspace.startswith("s3://"):
        return markdown_text

    crop_dir = get_cropped_image_dir(args.workspace, source_file)
    page_blocks = []
    current_page = None
    buffer = []

    for line in markdown_text.splitlines():
        m = re.match(r"<!-- page (\d+) start -->", line.strip())
        if m:
            if current_page is not None:
                page_blocks.append((current_page, "\n".join(buffer)))
                buffer = []
            current_page = int(m.group(1))
            buffer.append(line)
            continue
        buffer.append(line)

    if current_page is not None:
        page_blocks.append((current_page, "\n".join(buffer)))
    else:
        return markdown_text

    rewritten_blocks = []
    for page_num, block in page_blocks:
        rendered_page = get_rendered_page_path(args.workspace, source_file, page_num)
        if os.path.exists(rendered_page):
            rewritten_blocks.append(materialize_markdown_assets(block, rendered_page, crop_dir))
        else:
            rewritten_blocks.append(block)

    return "\n".join(rewritten_blocks)


async def process_single_pdf(
    args,
    worker_id: int,
    pdf_orig_path: str,
    local_pdf_path: str,
    return_page_results: bool = False,
):
    try:
        try:
            reader = PdfReader(local_pdf_path)
            num_pages = reader.get_num_pages()
        except Exception:
            logger.exception(f"Could not count number of pages for {pdf_orig_path}")
            return None

        if getattr(args, "apply_filter", False) and get_pdf_filter().filter_out_pdf(local_pdf_path):
            logger.info(f"Filtering out pdf {pdf_orig_path}")
            return None

        page_tasks = []
        async with asyncio.TaskGroup() as tg:
            for page_num in range(1, num_pages + 1):
                page_tasks.append(tg.create_task(process_page(args, worker_id, pdf_orig_path, local_pdf_path, page_num)))

        page_results = [task.result() for task in page_tasks]
        assert all(page_result.is_valid for page_result in page_results)

        num_fallback_pages = sum(page_result.is_fallback for page_result in page_results)
        if num_pages > 0 and num_fallback_pages / num_pages > args.max_page_error_rate:
            logger.error(f"Too many fallback pages in {pdf_orig_path}")
            return None

        doc = build_dolma_document(pdf_orig_path, page_results)
        if doc is not None and getattr(args, "embed_page_markers", False):
            page_chunks = []
            for page_result in page_results:
                txt = page_result.response.natural_text or ""
                page_chunks.append(f"<!-- page {page_result.page_num} start -->\n{txt}\n<!-- page {page_result.page_num} end -->")
            doc["text"] = "\n\n".join(page_chunks)

        if return_page_results:
            return {
                "doc": doc,
                "page_results": page_results,
            }
        return doc
    except Exception as e:
        logger.exception(f"Exception in process_single_pdf for {pdf_orig_path}: {e}")
        return None


async def process_pdf(args, worker_id: int, pdf_orig_path: str):
    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf", delete=False) as tf:
        try:
            with open(pdf_orig_path, "rb") as f:
                data = f.read()
            tf.write(data)
            tf.flush()
        except ClientError as ex:
            if hasattr(ex, "response") and ex.response["Error"]["Code"] == "NoSuchKey":
                logger.info(f"S3 File Not found, skipping {pdf_orig_path}")
                return None
            raise

        if is_png(tf.name) or is_jpeg(tf.name):
            logger.info(f"Converting {pdf_orig_path} from image to PDF format...")
            tf.seek(0)
            tf.write(convert_image_to_pdf_bytes(tf.name))
            tf.flush()

    try:
        return await process_single_pdf(args, worker_id, pdf_orig_path, tf.name)
    finally:
        if os.path.exists(tf.name):
            os.unlink(tf.name)


async def ocr_pdf(
    args,
    pdf_path: str,
    pdf_orig_path: str | None = None,
    write_markdown: bool = False,
) -> dict:
    pdf_orig_path = pdf_orig_path or pdf_path

    result = await process_single_pdf(
        args,
        worker_id=0,
        pdf_orig_path=pdf_orig_path,
        local_pdf_path=pdf_path,
        return_page_results=True,
    )

    if result is None:
        return {"doc": None, "markdown": None, "page_results": []}

    doc = result["doc"]
    page_results = result["page_results"]
    markdown_text = doc["text"] if doc is not None else None

    if write_markdown and markdown_text is not None:
        markdown_path = get_markdown_path(args.workspace, pdf_orig_path)
        os.makedirs(os.path.dirname(markdown_path), exist_ok=True)
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

    return {
        "doc": doc,
        "markdown": markdown_text,
        "page_results": page_results,
    }


async def ocr_pdfs(
    args,
    pdf_paths: list[str],
    write_markdown: bool = False,
) -> list[dict]:
    results = []

    async def _run_one(pdf_path: str):
        return await ocr_pdf(args, pdf_path=pdf_path, pdf_orig_path=pdf_path, write_markdown=write_markdown)

    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(_run_one(p)) for p in pdf_paths]

    for t in tasks:
        results.append(t.result())

    return results


def ocr_page_image_sync(args, image_bytes: bytes, model_name: str | None = None, page_num: int | None = None, image_rotation: int = 0, attempt: int = 0) -> PageResult:
    return asyncio.run(
        ocr_page_image(
            args=args,
            image_bytes=image_bytes,
            model_name=model_name,
            page_num=page_num,
            image_rotation=image_rotation,
            attempt=attempt,
        )
    )


def ocr_page_pdf_sync(args, pdf_path: str, page_num: int, pdf_orig_path: str | None = None) -> PageResult:
    return asyncio.run(ocr_page_pdf(args, pdf_path=pdf_path, page_num=page_num, pdf_orig_path=pdf_orig_path))


def ocr_region_sync(args, pdf_path: str, page_num: int, bbox, pdf_orig_path: str | None = None, padding: int = 10, dpi: int = 180) -> PageResult:
    return asyncio.run(
        ocr_region(
            args=args,
            pdf_path=pdf_path,
            page_num=page_num,
            bbox=bbox,
            pdf_orig_path=pdf_orig_path,
            padding=padding,
            dpi=dpi,
        )
    )


def ocr_pdf_sync(args, pdf_path: str, pdf_orig_path: str | None = None, write_markdown: bool = False) -> dict:
    return asyncio.run(ocr_pdf(args, pdf_path=pdf_path, pdf_orig_path=pdf_orig_path, write_markdown=write_markdown))


def ocr_pdfs_sync(args, pdf_paths: list[str], write_markdown: bool = False) -> list[dict]:
    return asyncio.run(ocr_pdfs(args, pdf_paths=pdf_paths, write_markdown=write_markdown))


async def worker(args, work_queue: WorkQueue, worker_id):
    while True:
        work_item = await work_queue.get_work()
        if work_item is None:
            logger.info(f"Worker {worker_id} exiting due to empty queue")
            break

        await tracker.clear_work(worker_id)

        try:
            async with asyncio.TaskGroup() as tg:
                dolma_tasks = []
                for path in work_item.work_paths:
                    dolma_tasks.append(tg.create_task(process_pdf(args, worker_id, path)))

            dolma_docs = []
            for task in dolma_tasks:
                try:
                    result = task.result()
                except Exception:
                    result = None
                if result is None:
                    continue
                if isinstance(result, list):
                    dolma_docs.extend(result)
                else:
                    dolma_docs.append(result)

            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tf:
                for doc in dolma_docs:
                    tf.write(json.dumps(doc))
                    tf.write("\n")
                tf.flush()
                temp_path = tf.name

            try:
                output_final_path = os.path.join(args.workspace, "results", f"output_{work_item.hash}.jsonl")
                os.makedirs(os.path.dirname(output_final_path), exist_ok=True)
                shutil.copyfile(temp_path, output_final_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            if getattr(args, "markdown", False):
                for doc in dolma_docs:
                    source_file = doc["metadata"]["Source-File"]
                    natural_text = doc["text"]

                    if getattr(args, "materialize_assets", False):
                        natural_text = rewrite_markdown_with_real_assets(args, source_file, natural_text)

                    markdown_path = get_markdown_path(args.workspace, source_file)
                    markdown_dir = os.path.dirname(markdown_path)

                    os.makedirs(markdown_dir, exist_ok=True)
                    with open(markdown_path, "w", encoding="utf-8") as md_f:
                        md_f.write(natural_text)

            metrics.add_metrics(
                finished_input_tokens=sum(doc["metadata"]["total-input-tokens"] for doc in dolma_docs),
                finished_output_tokens=sum(doc["metadata"]["total-output-tokens"] for doc in dolma_docs),
            )
            await work_queue.mark_done(work_item)
        except Exception as e:
            logger.exception(f"Exception occurred while processing work_hash {work_item.hash}: {e}")


async def vllm_server_task(model_name_or_path, args, unknown_args=None):
    cmd = [
        "vllm",
        "serve",
        model_name_or_path,
        "--port",
        str(args.port),
        "--disable-log-requests",
        "--uvicorn-log-level",
        "warning",
        "--served-model-name",
        "olmocr",
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--limit-mm-per-prompt",
        '{"video": 0}',
    ]

    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if unknown_args:
        cmd.extend(unknown_args)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )

    def _kill_proc():
        try:
            proc.terminate()
        except Exception:
            pass

    atexit.register(_kill_proc)

    async def process_line(line):
        global vllm_queued_requests
        server_logger.info(line)
        if match := re.search(r"(?:Waiting|Pending):\s*(\d+)", line):
            vllm_queued_requests = int(match.group(1))

    async def read_stream(stream):
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                await process_line(line.decode("utf-8").rstrip())
            except Exception:
                pass

    stdout_task = asyncio.create_task(read_stream(proc.stdout))
    stderr_task = asyncio.create_task(read_stream(proc.stderr))

    try:
        await proc.wait()
    except asyncio.CancelledError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass
        raise

    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)


async def vllm_server_host(model_name_or_path, args, unknown_args=None):
    max_retries = 5
    retry = 0
    while retry < max_retries:
        await vllm_server_task(model_name_or_path, args, unknown_args)
        retry += 1
    logger.error("vLLM server restarted too many times")
    sys.exit(1)


async def vllm_server_ready(args):
    max_attempts = args.max_server_ready_timeout
    url = f"{args.server.rstrip('/')}/models"

    for attempt in range(1, max_attempts + 1):
        try:
            headers = {}
            if getattr(args, "api_key", None):
                headers["Authorization"] = f"Bearer {args.api_key}"
            async with httpx.AsyncClient() as session:
                response = await session.get(url, headers=headers)
                if response.status_code == 200:
                    logger.info("vllm server is ready.")
                    return
        except Exception:
            logger.warning(f"Attempt {attempt}: waiting for vllm server...")
        await asyncio.sleep(1)

    raise Exception("vllm server did not become ready after waiting.")


async def download_model(model_name_or_path: str, max_retries: int = 5):
    for retry in range(max_retries):
        try:
            if os.path.isabs(model_name_or_path) and os.path.isdir(model_name_or_path):
                return model_name_or_path
            else:
                if snapshot_download is None:
                    raise RuntimeError("huggingface_hub is not installed")
                snapshot_download(repo_id=model_name_or_path)
                return model_name_or_path
        except Exception:
            if retry == max_retries - 1:
                raise
            await asyncio.sleep(random.randrange(10, 30) * 2**retry)


async def metrics_reporter(work_queue):
    while True:
        logger.info(f"Queue remaining: {work_queue.size}")
        logger.info("\n" + str(metrics))
        logger.info("\n" + str(await tracker.get_status_table()))
        await asyncio.sleep(10)


async def main():
    parser = argparse.ArgumentParser(description="Custom olmOCR pipeline with real asset materialization.")
    parser.add_argument("workspace")
    parser.add_argument("--pdfs", nargs="*", default=None)
    parser.add_argument("--model", default="allenai/olmOCR-2-7B-1025-FP8")

    parser.add_argument("--workspace_profile", default=None)
    parser.add_argument("--pdf_profile", default=None)
    parser.add_argument("--pages_per_group", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--max_page_retries", type=int, default=8)
    parser.add_argument("--max_page_error_rate", type=float, default=0.004)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max_concurrent_requests", type=int, default=1600)
    parser.add_argument("--max_server_ready_timeout", type=int, default=600)
    parser.add_argument("--apply_filter", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--target_longest_image_dim", type=int, default=1288)
    parser.add_argument("--guided_decoding", action="store_true")
    parser.add_argument("--disk_logging", type=str, nargs="?", const="olmocr-pipeline-debug.log", default=None)

    parser.add_argument("--emit_figure_placeholders", action="store_true")
    parser.add_argument("--materialize_assets", action="store_true")
    parser.add_argument("--save_rendered_pages", action="store_true")
    parser.add_argument("--embed_page_markers", action="store_true")
    parser.add_argument("--table_format", choices=["html", "markdown"], default="html")
    parser.add_argument("--page_max_tokens", type=int, default=8000)

    server_group = parser.add_argument_group("server")
    server_group.add_argument("--server", type=str, default=None)
    server_group.add_argument("--api_key", type=str, default=None)

    vllm_group = parser.add_argument_group("vllm")
    vllm_group.add_argument("--gpu-memory-utilization", type=float)
    vllm_group.add_argument("--max_model_len", type=int, default=16384)
    vllm_group.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    vllm_group.add_argument("--data-parallel-size", "-dp", type=int, default=1)
    vllm_group.add_argument("--port", type=int, default=30024)

    args, unknown_args = parser.parse_known_args()

    if args.disk_logging:
        file_handler = logging.FileHandler(args.disk_logging, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        server_logger.addHandler(file_handler)

    global workspace_s3, pdf_s3, max_concurrent_requests_limit
    max_concurrent_requests_limit = asyncio.BoundedSemaphore(args.max_concurrent_requests)
    use_internal_server = not args.server

    if args.workspace_profile and boto3 is not None:
        workspace_s3 = boto3.Session(profile_name=args.workspace_profile).client("s3")
    if args.pdf_profile and boto3 is not None:
        pdf_s3 = boto3.Session(profile_name=args.pdf_profile).client("s3")

    work_queue = WorkQueue(LocalBackend(args.workspace))

    if args.pdfs:
        pdf_work_paths = set()
        tarball_paths = set()

        for pdf_path in args.pdfs:
            if os.path.exists(pdf_path):
                if is_tarball_path(pdf_path):
                    tarball_paths.add(pdf_path)
                elif pdf_path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                    pdf_work_paths.add(pdf_path)
                elif pdf_path.lower().endswith(".txt"):
                    with open(pdf_path, "r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    tarball_paths.update(p for p in lines if is_tarball_path(p))
                    pdf_work_paths.update(p for p in lines if not is_tarball_path(p))
                else:
                    raise ValueError(f"Unsupported file extension for {pdf_path}")
            else:
                raise ValueError("pdfs argument must be local path")

        if not hasattr(args, "pages_per_group"):
            args.pages_per_group = 500

        if pdf_work_paths:
            sample_size = min(100, len(pdf_work_paths))
            sampled_pdfs = random.sample(list(pdf_work_paths), sample_size)
            page_counts = []

            for pdf in tqdm(sampled_pdfs, desc="Sampling PDFs"):
                try:
                    reader = PdfReader(pdf)
                    page_counts.append(len(reader.pages))
                except Exception as e:
                    logger.warning(f"Failed to read {pdf}: {e}")

            avg_pages_per_pdf = sum(page_counts) / len(page_counts) if page_counts else 10
            items_per_group = max(1, int(args.pages_per_group / avg_pages_per_pdf))
            await work_queue.populate_queue(list(pdf_work_paths), items_per_group)

        if tarball_paths:
            await work_queue.populate_queue(tarball_paths, 1)

    if args.stats:
        logger.info("Stats mode not implemented in custom pipeline.")
        return

    model_name_or_path = None
    qsize = await work_queue.initialize_queue()
    if qsize == 0:
        logger.info("No work to do, exiting")
        return

    vllm_server = None
    if use_internal_server:
        if not args.model:
            raise ValueError("Internal server mode requires --model")
        model_name_or_path = await download_model(args.model)
        args.server = f"http://localhost:{args.port}/v1"
        args.model = "olmocr"
        vllm_server = asyncio.create_task(vllm_server_host(model_name_or_path, args, unknown_args))

    await vllm_server_ready(args)

    metrics_task = asyncio.create_task(metrics_reporter(work_queue))
    worker_tasks = [asyncio.create_task(worker(args, work_queue, worker_id=i)) for i in range(args.workers)]
    await asyncio.gather(*worker_tasks)

    if vllm_server is not None:
        vllm_server.cancel()
    metrics_task.cancel()

    tasks_to_wait = [metrics_task]
    if vllm_server is not None:
        tasks_to_wait.append(vllm_server)
    await asyncio.gather(*tasks_to_wait, return_exceptions=True)


def cli_main():
    return asyncio.run(main())


if __name__ == "__main__":
    cli_main()

"""
from BatchAgent.ocr_agent.pipeline_custom import make_ocr_args, ocr_page_pdf_sync

args = make_ocr_args(
    server="http://localhost:8002/v1",
    model="allenai/olmOCR-2-7B-1025-FP8",
    workspace="./tmp_ocr",
)

result = ocr_page_pdf_sync(args, pdf_path="paper.pdf", page_num=2)
print(result.response.natural_text)


from BatchAgent.ocr_agent.pipeline_custom import make_ocr_args, ocr_region_sync

args = make_ocr_args(
    server="http://localhost:8002/v1",
    model="allenai/olmOCR-2-7B-1025-FP8",
    workspace="./tmp_ocr",
)

bbox = (100, 200, 500, 700)
result = ocr_region_sync(args, pdf_path="paper.pdf", page_num=3, bbox=bbox)
print(result.response.natural_text)

from BatchAgent.ocr_agent.pipeline_custom import make_ocr_args, ocr_pdf_sync

args = make_ocr_args(
    server="http://localhost:8002/v1",
    model="allenai/olmOCR-2-7B-1025-FP8",
    workspace="./tmp_ocr",
    embed_page_markers=True,
)

result = ocr_pdf_sync(args, pdf_path="paper.pdf", write_markdown=True)
print(result["markdown"][:1000])


from BatchAgent.ocr_agent.pipeline_custom import make_ocr_args, ocr_pdfs_sync

args = make_ocr_args(
    server="http://localhost:8002/v1",
    model="allenai/olmOCR-2-7B-1025-FP8",
    workspace="./tmp_ocr",
)

results = ocr_pdfs_sync(args, ["a.pdf", "b.pdf", "c.pdf"], write_markdown=True)
print(len(results))
"""
