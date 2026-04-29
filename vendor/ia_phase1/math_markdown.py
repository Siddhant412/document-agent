from __future__ import annotations

import re

__all__ = ["normalize_math_delimiters"]


def normalize_math_delimiters(text: str) -> str:
    """
    Normalize common LaTeX math delimiters to a Markdown-friendly form.

    This mirrors the lightweight math postprocessing used in PaperFlow:
    inline `\\(...\\)` becomes `$ ... $`, display `\\[...\\]` becomes
    `$$ ... $$`, and common equation environments are collapsed to `$$`.
    """

    normalized = str(text or "")
    normalized = re.sub(r"\\\[(.*?)\\\]", r"$$ \1 $$", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\\\((.*?)\\\)", r"$ \1 $", normalized, flags=re.DOTALL)

    for env in ["equation", "align", "align*", "gather", "gather*"]:
        normalized = re.sub(
            rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}",
            r"$$ \1 $$",
            normalized,
            flags=re.DOTALL,
        )

    return normalized
