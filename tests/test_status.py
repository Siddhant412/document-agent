from document_agent.status import batch_percent_from_jobs, batch_status_from_counts


def test_batch_status_success() -> None:
    assert (
        batch_status_from_counts(
            total_files=2,
            succeeded_count=2,
            failed_count=0,
            cancelled_count=0,
        )
        == "succeeded"
    )


def test_batch_status_partial_failed() -> None:
    assert (
        batch_status_from_counts(
            total_files=3,
            succeeded_count=1,
            failed_count=2,
            cancelled_count=0,
        )
        == "partial_failed"
    )


def test_batch_status_cancelled_before_success() -> None:
    assert (
        batch_status_from_counts(
            total_files=2,
            succeeded_count=0,
            failed_count=0,
            cancelled_count=2,
            cancellation_requested=True,
        )
        == "cancelled"
    )


def test_batch_percent_weights_child_jobs_equally() -> None:
    assert (
        batch_percent_from_jobs(
            [
                {"status": "succeeded", "percent": 100},
                {"status": "running", "percent": 50},
                {"status": "queued", "percent": 0},
            ]
        )
        == 50
    )


def test_batch_percent_counts_failed_or_cancelled_children_as_complete() -> None:
    assert (
        batch_percent_from_jobs(
            [
                {"status": "failed", "percent": 35},
                {"status": "cancelled", "percent": 5},
            ]
        )
        == 100
    )
