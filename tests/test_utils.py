from document_agent.utils import markdown_filename, safe_filename, unique_names


def test_safe_filename_normalizes_unsafe_text() -> None:
    assert safe_filename("../résumé file?.pdf") == "resume_file.pdf"


def test_markdown_filename_replaces_extension() -> None:
    assert markdown_filename("paper.pdf") == "paper.md"


def test_unique_names_adds_numeric_suffixes() -> None:
    assert unique_names(["paper.pdf", "paper.docx", "paper.pdf"]) == [
        "paper.md",
        "paper-2.md",
        "paper-3.md",
    ]

