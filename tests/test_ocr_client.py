from types import SimpleNamespace

from document_agent.ocr.client import OcrClient, _extract_message_content, _strip_outer_markdown_fence


def test_ocr_url_accepts_base_url() -> None:
    settings = SimpleNamespace(
        ocr_server_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ocr_max_concurrent_requests=2,
    )
    client = OcrClient(settings)  # type: ignore[arg-type]
    assert (
        client.chat_completions_url()
        == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    )


def test_ocr_url_accepts_full_chat_completions_url() -> None:
    settings = SimpleNamespace(
        ocr_server_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        ocr_max_concurrent_requests=2,
    )
    client = OcrClient(settings)  # type: ignore[arg-type]
    assert (
        client.chat_completions_url()
        == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    )


def test_extract_message_content_from_string_response() -> None:
    assert _extract_message_content({"choices": [{"message": {"content": "  hello  "}}]}) == "hello"


def test_extract_message_content_from_list_response() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ]
                }
            }
        ]
    }
    assert _extract_message_content(payload) == "hello\nworld"


def test_strip_outer_markdown_fence() -> None:
    content = "```markdown\n# Title\n\nBody\n```"
    assert _strip_outer_markdown_fence(content) == "# Title\n\nBody"


def test_strip_outer_markdown_fence_leaves_inline_fences() -> None:
    content = "# Title\n\n```python\nprint('hello')\n```"
    assert _strip_outer_markdown_fence(content) == content
