from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from document_agent.api.schemas import (
    ObservabilityErrorsResponse,
    ObservabilityEventsResponse,
    ObservabilityLogsResponse,
    ObservabilityStatsResponse,
)
from document_agent.db.repository import Repository
from document_agent.logging_config import get_ring_buffer

observability_router = APIRouter(prefix="/v1/observability", tags=["observability"])

TIME_RANGES: Dict[str, Optional[int]] = {
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
    "7d": 7 * 24 * 60 * 60,
    "30d": 30 * 24 * 60 * 60,
    "all": None,
}


def _get_repo() -> Repository:
    from document_agent.db.connection import get_pool
    return Repository(get_pool())


def _range_seconds(time_range: str) -> Optional[int]:
    if time_range not in TIME_RANGES:
        allowed = ", ".join(TIME_RANGES)
        raise HTTPException(status_code=400, detail=f"Invalid time_range. Use one of: {allowed}.")
    return TIME_RANGES[time_range]


def _range_start(time_range: str) -> Optional[dt.datetime]:
    seconds = _range_seconds(time_range)
    if seconds is None:
        return None
    return dt.datetime.now(dt.UTC) - dt.timedelta(seconds=seconds)


@observability_router.get("/stats", response_model=ObservabilityStatsResponse)
def get_stats(
    time_range: str = Query(default="24h", description="One of: 1h, 6h, 24h, 7d, 30d, all"),
    repo: Repository = Depends(_get_repo),
) -> ObservabilityStatsResponse:
    data = repo.get_observability_stats(range_seconds=_range_seconds(time_range))

    jobs_by_status: Dict[str, int] = {row["status"]: int(row["count"]) for row in data["status_counts"]}
    total_jobs = sum(jobs_by_status.values())
    succeeded = jobs_by_status.get("succeeded", 0)
    terminal = succeeded + jobs_by_status.get("failed", 0) + jobs_by_status.get("cancelled", 0)
    success_rate = round(succeeded / terminal * 100, 1) if terminal > 0 else None

    duration = data.get("duration") or {}

    # Pivot: [{hour, status, count}] -> [{hour, succeeded, failed}] sorted by hour
    hourly: Dict[str, Dict] = {}
    for row in data["throughput_by_hour"]:
        h = str(row["hour"])
        if h not in hourly:
            hourly[h] = {"hour": h, "succeeded": 0, "failed": 0}
        hourly[h][row["status"]] = int(row.get("count", 0))
    throughput = list(hourly.values())

    health = {
        "api": "ok",
        "db": "ok",
        "worker": "ok" if data["active_leases"] > 0 else "idle",
    }

    return ObservabilityStatsResponse(
        total_jobs=total_jobs,
        jobs_by_status=jobs_by_status,
        success_rate_pct=success_rate,
        avg_duration_seconds=duration.get("avg_seconds"),
        p95_duration_seconds=duration.get("p95_seconds"),
        total_batches=data["total_batches"],
        active_jobs=data["active_leases"],
        throughput_by_hour=throughput,
        jobs_by_type=[
            {"detected_type": r["detected_type"], "count": int(r["count"])}
            for r in data["jobs_by_type"]
        ],
        health=health,
    )


@observability_router.get("/events", response_model=ObservabilityEventsResponse)
def get_events(
    limit: int = Query(default=50, ge=1, le=200),
    before_id: Optional[int] = Query(default=None),
    since_id: Optional[int] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    time_range: str = Query(default="24h", description="One of: 1h, 6h, 24h, 7d, 30d, all"),
    repo: Repository = Depends(_get_repo),
) -> ObservabilityEventsResponse:
    rows = repo.get_global_events(
        limit=limit,
        before_id=before_id,
        since_id=since_id,
        event_type=event_type,
        q=q,
        range_seconds=_range_seconds(time_range),
    )

    if since_id is not None:
        return ObservabilityEventsResponse(events=rows, has_more=False, next_before_id=None)

    has_more = len(rows) > limit
    display = rows[:limit]
    next_before_id = display[-1]["id"] if has_more and display else None
    return ObservabilityEventsResponse(events=display, has_more=has_more, next_before_id=next_before_id)


@observability_router.get("/errors", response_model=ObservabilityErrorsResponse)
def get_errors(
    limit: int = Query(default=20, ge=1, le=100),
    error_code: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    time_range: str = Query(default="24h", description="One of: 1h, 6h, 24h, 7d, 30d, all"),
    repo: Repository = Depends(_get_repo),
) -> ObservabilityErrorsResponse:
    result = repo.get_recent_errors(
        limit=limit,
        error_code=error_code,
        q=q,
        range_seconds=_range_seconds(time_range),
    )
    return ObservabilityErrorsResponse(**result)


@observability_router.get("/logs", response_model=ObservabilityLogsResponse)
def get_logs(
    limit: int = Query(default=100, ge=1, le=500),
    level: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    since_seq: int = Query(default=0, ge=0),
    time_range: str = Query(default="24h", description="One of: 1h, 6h, 24h, 7d, 30d, all"),
) -> ObservabilityLogsResponse:
    ring = get_ring_buffer()
    records = ring.get_records(
        limit=limit,
        level=level,
        q=q,
        since_seq=since_seq,
        since_ts=_range_start(time_range),
    )
    stats = ring.stats()
    return ObservabilityLogsResponse(
        logs=records,
        max_seq=stats["max_seq"],
        buffer_capacity=stats["buffer_capacity"],
        buffer_used=stats["buffer_used"],
    )
