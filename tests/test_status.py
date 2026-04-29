from document_agent.status import batch_status_from_counts


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

