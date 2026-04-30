from __future__ import annotations

TERMINAL_JOB_STATUSES = {"succeeded", "failed", "cancelled"}
TERMINAL_BATCH_STATUSES = {"succeeded", "partial_failed", "failed", "cancelled"}


def batch_status_from_counts(
    *,
    total_files: int,
    succeeded_count: int,
    failed_count: int,
    cancelled_count: int,
    cancellation_requested: bool = False,
) -> str:
    terminal = succeeded_count + failed_count + cancelled_count
    if total_files <= 0:
        return "failed"
    if terminal < total_files:
        return "running" if terminal > 0 else "queued"
    if succeeded_count == total_files:
        return "succeeded"
    if succeeded_count > 0:
        return "partial_failed"
    if cancellation_requested:
        return "cancelled"
    return "failed"


def batch_percent_from_jobs(jobs: list[dict[str, object]]) -> int:
    if not jobs:
        return 0
    total = 0
    for job in jobs:
        if str(job.get("status") or "") in TERMINAL_JOB_STATUSES:
            total += 100
        else:
            total += max(0, min(100, int(job.get("percent") or 0)))
    return round(total / len(jobs))
