import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PageResponse:
    primary_language: Optional[str]
    is_rotation_valid: bool
    rotation_correction: int
    is_table: bool
    is_diagram: bool
    natural_text: Optional[str]

    def __post_init__(self):
        if self.rotation_correction not in {0, 90, 180, 270}:
            raise ValueError("rotation_correction must be one of [0, 90, 180, 270].")
        if not isinstance(self.primary_language, (str, type(None))):
            raise TypeError("primary_language must be Optional[str].")
        if not isinstance(self.is_rotation_valid, bool):
            raise TypeError("is_rotation_valid must be bool.")
        if not isinstance(self.rotation_correction, int):
            raise TypeError("rotation_correction must be int.")
        if not isinstance(self.is_table, bool):
            raise TypeError("is_table must be bool.")
        if not isinstance(self.is_diagram, bool):
            raise TypeError("is_diagram must be bool.")
        if not isinstance(self.natural_text, (str, type(None))):
            raise TypeError("natural_text must be Optional[str].")


def build_custom_yaml_prompt(
    page_width: int | None = None,
    page_height: int | None = None,
    emit_figure_placeholders: bool = True,
    table_format: str = "html",
) -> str:
    table_instruction = "Convert tables to HTML." if table_format == "html" else "Convert tables to markdown."

    figure_instruction = ""
    if emit_figure_placeholders:
        figure_instruction = (
            "If there are any figures, charts, plots, diagrams, or image regions that should be preserved, "
            "insert markdown image references using this exact format:\n"
            "![Alt text describing the contents of the figure](page_STARTX_STARTY_ENDX_ENDY.png)\n"
            "Use integer coordinates in image pixel space.\n"
        )

    size_text = ""
    if page_width is not None and page_height is not None:
        size_text = f"Page width: {page_width}. Page height: {page_height}.\n"

    return (
        "Attached is one page of a document that you must process.\n"
        "Return a faithful markdown transcription in natural reading order.\n"
        "Rules:\n"
        "- Preserve headings, paragraphs, lists, captions, references, and footnotes.\n"
        "- Remove repetitive headers and footers when they are clearly boilerplate.\n"
        "- Convert equations and math expressions to LaTeX.\n"
        "- Use \\( \\) for inline math and \\[ \\] for block math.\n"
        "- Prefer LaTeX commands instead of Unicode math symbols.\n"
        f"- {table_instruction}\n"
        "- Do not hallucinate missing content.\n"
        "- If text is unreadable, keep it conservative.\n"
        f"{figure_instruction}"
        f"{size_text}"
        "Return your output as markdown with YAML front matter on top containing exactly these keys:\n"
        "primary_language, is_rotation_valid, rotation_correction, is_table, is_diagram\n"
    )


def extract_raw_text(prompt: str) -> str:
    pattern = r"RAW_TEXT_START\s*\n(.*?)\nRAW_TEXT_END"
    match = re.search(pattern, prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    raise ValueError("Prompt does not contain raw text")