from document_agent.converters.assets import rewrite_markdown_image_paths


def test_rewrite_markdown_image_paths_replaces_local_paths() -> None:
    markdown = "![figure](media/image1.png)\n\n![keep](https://example.com/image.png)"
    rewritten = rewrite_markdown_image_paths(markdown, {"media/image1.png": "http://api/v1/assets/abc"})
    assert "![figure](http://api/v1/assets/abc)" in rewritten
    assert "![keep](https://example.com/image.png)" in rewritten
