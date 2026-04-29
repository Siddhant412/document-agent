from types import SimpleNamespace

from document_agent.storage.minio_store import ObjectStore


def _store() -> ObjectStore:
    store = ObjectStore.__new__(ObjectStore)
    store.settings = SimpleNamespace(public_base_url="http://api")
    store.bucket = "document-agent"
    return store


def test_job_asset_key_uses_planned_image_folder() -> None:
    assert (
        _store().job_asset_key(
            job_id="job-1",
            role="embedded_image",
            asset_id="asset-1",
            filename="figure.png",
        )
        == "jobs/job-1/assets/images/asset-1.png"
    )


def test_job_asset_key_uses_planned_equation_folder() -> None:
    assert (
        _store().job_asset_key(
            job_id="job-1",
            role="equation_image",
            asset_id="asset-2",
            filename="eq.jpg",
        )
        == "jobs/job-1/assets/equations/asset-2.jpg"
    )
