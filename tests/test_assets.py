from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from document_agent.assets import count_document_assets
from document_agent.converters.assets import rewrite_markdown_image_paths
from document_agent.converters.assets import upload_assets_and_rewrite_markdown
from document_agent.converters.base import ConversionContext
from document_agent.converters.markdown import rewrite_frontmatter_fields
from document_agent.storage import ObjectInfo


def test_rewrite_markdown_image_paths_replaces_local_paths() -> None:
    markdown = "![figure](media/image1.png)\n\n![keep](https://example.com/image.png)"
    rewritten = rewrite_markdown_image_paths(markdown, {"media/image1.png": "http://api/v1/assets/abc"})
    assert "![figure](http://api/v1/assets/abc)" in rewritten
    assert "![keep](https://example.com/image.png)" in rewritten


def test_rewrite_frontmatter_fields_updates_existing_fields() -> None:
    markdown = "---\ndetected_type: pdf\nsource_filename: 'source.pdf'\n---\n\nBody\n"
    rewritten = rewrite_frontmatter_fields(
        markdown,
        {"detected_type": "doc", "source_filename": "source.doc"},
    )
    assert "detected_type: 'doc'" in rewritten
    assert "source_filename: 'source.doc'" in rewritten
    assert rewritten.endswith("\n\nBody\n")


def test_upload_assets_only_persists_referenced_images(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    referenced = assets_root / "figure.png"
    unreferenced = assets_root / "page_0001.png"
    referenced.write_bytes(b"figure")
    unreferenced.write_bytes(b"page")
    context = ConversionContext(
        job_id=uuid4(),
        batch_id=None,
        input_index=None,
        source_path=tmp_path / "source.pdf",
        filename="source.pdf",
        detected_type="pdf",
        content_type="application/pdf",
        temp_dir=tmp_path,
        repository=_FakeRepository(),
        object_store=_FakeObjectStore(),
        settings=SimpleNamespace(),
    )

    markdown, uploaded = upload_assets_and_rewrite_markdown(
        context,
        markdown="![figure](figure.png)",
        assets_root=assets_root,
    )

    assert len(uploaded) == 1
    assert uploaded[0].original_path == referenced
    assert "http://api/assets/" in markdown


def test_count_document_assets_excludes_results_and_diagnostics() -> None:
    assert (
        count_document_assets(
            [
                {"role": "markdown_result"},
                {"role": "diagnostic_manifest"},
                {"role": "batch_manifest"},
                {"role": "embedded_image"},
                {"role": "equation_image"},
            ]
        )
        == 2
    )


class _FakeRepository:
    def create_asset(self, **kwargs):
        return {"id": kwargs["asset_id"], **kwargs}


class _FakeObjectStore:
    bucket = "test"

    def job_asset_key(self, *, job_id: str, role: str, asset_id: str, filename: str) -> str:
        return f"jobs/{job_id}/assets/images/{asset_id}.png"

    def public_url_for_asset(self, asset_id: str) -> str:
        return f"http://api/assets/{asset_id}"

    def upload_file(self, *, path: Path, object_key: str, mime_type: str | None = None) -> ObjectInfo:
        return ObjectInfo(
            bucket=self.bucket,
            object_key=object_key,
            mime_type=mime_type or "image/png",
            size_bytes=path.stat().st_size,
            sha256="sha",
            public_url="",
        )
