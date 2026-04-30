from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from document_agent.api.sse import events_response


def test_sse_stream_stops_when_client_disconnects() -> None:
    job_id = uuid4()
    response = events_response(
        request=_DisconnectAfterFirstPollRequest(),
        repository=_EventRepository(
            [
                {
                    "id": 1,
                    "event_type": "queued",
                    "batch_id": None,
                    "job_id": job_id,
                    "stage": "queued",
                    "percent": 0,
                    "message": "Job queued.",
                    "payload_json": {},
                    "created_at": "now",
                }
            ]
        ),  # type: ignore[arg-type]
        job_id=job_id,
        poll_seconds=0,
    )

    chunks = asyncio.run(_collect_chunks(response.body_iterator, limit=2))

    assert len(chunks) == 1
    assert "event: queued" in chunks[0]
    assert f'"job_id": "{job_id}"' in chunks[0]


def test_batch_sse_stream_emits_batch_and_child_events() -> None:
    batch_id = uuid4()
    child_job_id = uuid4()
    response = events_response(
        request=_DisconnectAfterFirstPollRequest(),
        repository=_EventRepository(
            [
                {
                    "id": 1,
                    "event_type": "queued",
                    "batch_id": batch_id,
                    "job_id": None,
                    "stage": "queued",
                    "percent": 0,
                    "message": "Batch queued.",
                    "payload_json": {"total_files": 1},
                    "created_at": "now",
                },
                {
                    "id": 2,
                    "event_type": "progress",
                    "batch_id": batch_id,
                    "job_id": child_job_id,
                    "stage": "convert",
                    "percent": 50,
                    "message": "Converting.",
                    "payload_json": {"input_index": 0},
                    "created_at": "now",
                },
            ]
        ),  # type: ignore[arg-type]
        batch_id=batch_id,
        poll_seconds=0,
    )

    chunks = asyncio.run(_collect_chunks(response.body_iterator, limit=3))
    payloads = [_payload_from_sse(chunk) for chunk in chunks]

    assert [payload["job_id"] for payload in payloads] == [None, str(child_job_id)]
    assert payloads[0]["payload"] == {"total_files": 1}
    assert payloads[1]["stage"] == "convert"


async def _collect_chunks(iterator, *, limit: int) -> list[str]:
    chunks = []
    async for chunk in iterator:
        chunks.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
        if len(chunks) >= limit:
            break
    return chunks


def _payload_from_sse(chunk: str) -> dict:
    for line in chunk.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"No data line in chunk: {chunk!r}")


class _DisconnectAfterFirstPollRequest:
    def __init__(self) -> None:
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > 1


class _EventRepository:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def get_events(self, *, job_id=None, batch_id=None, after_id: int = 0, limit: int = 100):
        return [row for row in self.rows if row["id"] > after_id]
