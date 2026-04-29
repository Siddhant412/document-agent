from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional
from uuid import UUID

from fastapi import Request
from starlette.responses import StreamingResponse

from document_agent.db.repository import Repository


def _sse_message(*, event_id: int, event_type: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, default=str, ensure_ascii=False)
    return f"id: {event_id}\nevent: {event_type}\ndata: {payload}\n\n"


def events_response(
    *,
    request: Request,
    repository: Repository,
    job_id: Optional[UUID] = None,
    batch_id: Optional[UUID] = None,
    after_id: int = 0,
    poll_seconds: float = 1.0,
) -> StreamingResponse:
    async def stream() -> Any:
        last_id = after_id
        idle = 0
        while True:
            if await request.is_disconnected():
                break
            rows = repository.get_events(job_id=job_id, batch_id=batch_id, after_id=last_id)
            if rows:
                idle = 0
                for row in rows:
                    last_id = int(row["id"])
                    yield _sse_message(
                        event_id=last_id,
                        event_type=str(row["event_type"]),
                        data={
                            "id": row["id"],
                            "batch_id": row.get("batch_id"),
                            "job_id": row.get("job_id"),
                            "stage": row.get("stage"),
                            "percent": row.get("percent"),
                            "message": row.get("message"),
                            "payload": row.get("payload_json") or {},
                            "created_at": row.get("created_at"),
                        },
                    )
            else:
                idle += 1
                if idle >= 15:
                    idle = 0
                    yield ": keepalive\n\n"
            await asyncio.sleep(poll_seconds)

    return StreamingResponse(stream(), media_type="text/event-stream")

