from __future__ import annotations

import json
import mimetypes
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, List, Optional
from uuid import UUID, uuid4

import httpx
import typer

from document_agent.cli.local import LocalObjectStore, LocalRepository
from document_agent.config import get_settings
from document_agent.converters.detect import detect_file_type
from document_agent.converters.pipeline import ConversionPipeline
from document_agent.db.connection import init_db
from document_agent.logging_config import configure_logging
from document_agent.status import TERMINAL_BATCH_STATUSES
from document_agent.utils import markdown_filename, unique_names
from document_agent.worker.runner import run_worker

app = typer.Typer(no_args_is_help=True)


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    import uvicorn

    uvicorn.run(
        "document_agent.api.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )


@app.command()
def worker() -> None:
    run_worker()


@app.command("init-db")
def init_database() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db(settings)
    typer.echo("Database initialized.")


@app.command()
def submit(
    input_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    metadata_json: Optional[str] = typer.Option(None, "--metadata-json"),
) -> None:
    settings = get_settings()
    with input_path.open("rb") as handle:
        files = {"file": (input_path.name, handle, mimetypes.guess_type(input_path.name)[0])}
        data = {"metadata_json": metadata_json} if metadata_json else None
        response = httpx.post(
            f"{settings.api_base_url.rstrip('/')}/v1/jobs",
            files=files,
            data=data,
            headers=_api_headers(),
            timeout=60,
        )
    response.raise_for_status()
    typer.echo(json.dumps(response.json(), indent=2))


@app.command()
def watch(job_id: UUID) -> None:
    settings = get_settings()
    url = f"{settings.api_base_url.rstrip('/')}/v1/jobs/{job_id}/events"
    with httpx.stream("GET", url, headers=_api_headers(), timeout=None) as response:
        response.raise_for_status()
        event = {}
        for line in response.iter_lines():
            if not line:
                if event:
                    typer.echo(event.get("data", ""))
                    event = {}
                continue
            if line.startswith(":"):
                continue
            key, _, value = line.partition(":")
            event[key] = value.strip()


@app.command()
def result(
    job_id: UUID,
    output: Path = typer.Option(..., "--output", "-o"),
) -> None:
    settings = get_settings()
    url = f"{settings.api_base_url.rstrip('/')}/v1/jobs/{job_id}/result"
    response = httpx.get(url, params={"include_markdown": "true"}, headers=_api_headers(), timeout=60)
    response.raise_for_status()
    payload = response.json()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload["markdown"], encoding="utf-8")
    typer.echo(str(output))


@app.command()
def batch(
    inputs: List[Path] = typer.Argument(..., exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    wait: bool = typer.Option(False, "--wait"),
    metadata_json: Optional[str] = typer.Option(None, "--metadata-json"),
) -> None:
    settings = get_settings()
    paths = list(_expand_inputs(inputs))
    if not paths:
        raise typer.BadParameter("No input files found.")
    with ExitStack() as stack:
        files = [
            (
                "files",
                (
                    path.name,
                    stack.enter_context(path.open("rb")),
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                ),
            )
            for path in paths
        ]
        data = {"metadata_json": metadata_json} if metadata_json else None
        response = httpx.post(
            f"{settings.api_base_url.rstrip('/')}/v1/batches",
            files=files,
            data=data,
            headers=_api_headers(),
            timeout=120,
        )
    response.raise_for_status()
    payload = response.json()
    typer.echo(json.dumps(payload, indent=2))
    if output_dir:
        _wait_for_batch(settings.api_base_url, payload["batch_id"])
        _download_batch(settings.api_base_url, payload["batch_id"], output_dir)
    elif wait:
        _wait_for_batch(settings.api_base_url, payload["batch_id"])


@app.command()
def convert(
    input_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    output: Path = typer.Option(..., "--output", "-o"),
    asset_url_mode: str = typer.Option("local", "--asset-url-mode"),
) -> None:
    if asset_url_mode != "local":
        raise typer.BadParameter("Only --asset-url-mode local is supported for synchronous CLI conversion.")
    settings = get_settings()
    job_id = uuid4()
    output.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output.parent / f"{output.stem}_assets"
    with tempfile.TemporaryDirectory(prefix="document-agent-cli-") as tmp:
        source_path = Path(tmp) / input_path.name
        source_path.write_bytes(input_path.read_bytes())
        detected_type = detect_file_type(source_path, input_path.name, mimetypes.guess_type(input_path.name)[0])
        pipeline = ConversionPipeline(
            repository=LocalRepository(),  # type: ignore[arg-type]
            object_store=LocalObjectStore(asset_dir),  # type: ignore[arg-type]
            settings=settings,
        )
        result = pipeline.convert(
            job_id=job_id,
            batch_id=None,
            input_index=None,
            source_path=source_path,
            filename=input_path.name,
            detected_type=detected_type,
            content_type=mimetypes.guess_type(input_path.name)[0],
            temp_dir=Path(tmp),
        )
    output.write_text(result.markdown, encoding="utf-8")
    typer.echo(str(output))


def _expand_inputs(inputs: Iterable[Path]) -> Iterable[Path]:
    for item in inputs:
        if item.is_dir():
            for path in sorted(item.rglob("*")):
                if path.is_file():
                    yield path
        elif item.is_file():
            yield item


def _wait_for_batch(api_base_url: str, batch_id: str) -> None:
    url = f"{api_base_url.rstrip('/')}/v1/batches/{batch_id}"
    while True:
        response = httpx.get(url, headers=_api_headers(), timeout=30)
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in TERMINAL_BATCH_STATUSES:
            typer.echo(json.dumps(payload, indent=2))
            return
        time.sleep(2)


def _download_batch(api_base_url: str, batch_id: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    response = httpx.get(
        f"{api_base_url.rstrip('/')}/v1/batches/{batch_id}/result",
        headers=_api_headers(),
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    names = unique_names([file_item["filename"] for file_item in payload["files"]])
    for index, file_item in enumerate(payload["files"]):
        if file_item["status"] != "succeeded" or not file_item.get("markdown_url"):
            continue
        markdown = httpx.get(file_item["markdown_url"], headers=_api_headers(), timeout=60)
        markdown.raise_for_status()
        name = file_item.get("markdown_filename") or names[index] or markdown_filename(file_item["filename"])
        (output_dir / name).write_bytes(markdown.content)


def _api_headers() -> dict[str, str]:
    settings = get_settings()
    if not settings.api_key:
        return {}
    return {settings.api_key_header: settings.api_key}


if __name__ == "__main__":
    app()
