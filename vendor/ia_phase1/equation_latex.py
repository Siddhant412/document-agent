from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

__all__ = [
    "extract_equation_latex",
    "fallback_text_to_latex",
    "validate_equation_latex",
]

_EQUATION_NUMBER_ONLY_RE = re.compile(r"^\(\s*(\d+[A-Za-z]?)\s*\)$")
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_SYMBOL_RE = re.compile(r"[=+*/^_\\{}]|\\[A-Za-z]+")
_CODE_RE = re.compile(r"\b(?:return|def|class|lambda|torch\.|dtype=|shape=)\b")
_SUBSCRIPT_TOKEN_RE = re.compile(r"\b([A-Za-z])\s+(out|in|row|col)\b")
_NORM_SUBSCRIPT_RE = re.compile(r"∥([^∥\n]+)∥\s*([A-Za-z][A-Za-z0-9_]*)")
_NORM_RE = re.compile(r"∥([^∥\n]+)∥")
_UNICODE_REPLACEMENTS = {
    "⊙": r" \\odot ",
    "×": r" \\times ",
    "≤": r" \\leq ",
    "≥": r" \\geq ",
    "≈": r" \\approx ",
    "≠": r" \\neq ",
    "∈": r" \\in ",
    "∉": r" \\notin ",
    "∀": r" \\forall ",
    "∃": r" \\exists ",
    "∑": r" \\sum ",
    "∫": r" \\int ",
    "√": r" \\sqrt{} ",
    "∞": r" \\infty ",
    "′": "'",
    "ℓ": r"\\ell",
    "λ": r"\\lambda",
    "μ": r"\\mu",
    "σ": r"\\sigma",
    "θ": r"\\theta",
    "Δ": r"\\Delta",
    "Ω": r"\\Omega",
    "α": r"\\alpha",
    "β": r"\\beta",
    "γ": r"\\gamma",
    "π": r"\\pi",
}


def _latex_enabled() -> bool:
    raw = os.getenv("EQUATION_LATEX_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _latex_backend() -> str:
    raw = os.getenv("EQUATION_LATEX_BACKEND", "text").strip().lower()
    if raw in {"text"}:
        return raw
    return "text"


def _latex_fallback_enabled() -> bool:
    raw = os.getenv("EQUATION_LATEX_TEXT_FALLBACK_ENABLED", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _compact(text: str) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split()).strip()


def _strip_math_wrappers(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        return stripped[1:-1].strip()
    if stripped.startswith(r"\[") and stripped.endswith(r"\]") and len(stripped) >= 4:
        return stripped[2:-2].strip()
    return stripped


def _balanced_counts(text: str, opener: str, closer: str) -> bool:
    depth = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def validate_equation_latex(latex: str) -> tuple[bool, list[str]]:
    body = _strip_math_wrappers(latex)
    flags: list[str] = []
    if not body:
        return False, ["empty"]
    if len(body) < 3:
        flags.append("too_short")
    if not _balanced_counts(body, "{", "}"):
        flags.append("unbalanced_braces")
    if not _balanced_counts(body, "(", ")"):
        flags.append("unbalanced_parentheses")
    if not _balanced_counts(body, "[", "]"):
        flags.append("unbalanced_brackets")

    compact = _compact(body)
    word_count = len(_WORD_RE.findall(compact))
    symbol_count = len(_SYMBOL_RE.findall(compact))
    if _CODE_RE.search(compact):
        flags.append("code_like")
    if word_count >= 20 and symbol_count < 4:
        flags.append("wordy")
    if symbol_count == 0 and word_count >= 5:
        flags.append("not_math_like")

    return len(flags) == 0, flags


def _replace_unicode_symbols(text: str) -> str:
    updated = text
    updated = _NORM_SUBSCRIPT_RE.sub(lambda m: rf"\\lVert {m.group(1).strip()} \\rVert_{{{m.group(2).strip()}}}", updated)
    updated = _NORM_RE.sub(lambda m: rf"\\lVert {m.group(1).strip()} \\rVert", updated)
    updated = _SUBSCRIPT_TOKEN_RE.sub(lambda m: rf"{m.group(1)}_{{{m.group(2)}}}", updated)
    for source, target in _UNICODE_REPLACEMENTS.items():
        updated = updated.replace(source, target)
    updated = re.sub(r"\bR\s+d_\{(out|in|row|col)\}", lambda m: rf"\\mathbb{{R}}^{{d_{{{m.group(1)}}}}}", updated)
    updated = re.sub(r"\s+", " ", updated)
    return updated.strip()


def _extract_tag(lines: list[str], equation_number: Optional[str]) -> tuple[list[str], Optional[str]]:
    tag = str(equation_number or "").strip() or None
    cleaned = list(lines)
    if cleaned:
        match = _EQUATION_NUMBER_ONLY_RE.fullmatch(cleaned[-1])
        if match:
            tag = tag or str(match.group(1)).strip()
            cleaned = cleaned[:-1]
    return cleaned, tag


def fallback_text_to_latex(text: str, *, equation_number: Optional[str] = None) -> Optional[str]:
    lines = [_compact(line) for line in str(text or "").splitlines() if _compact(line)]
    if not lines:
        return None
    lines, tag = _extract_tag(lines, equation_number)
    if not lines:
        return None

    rendered_lines: list[str] = []
    for line in lines:
        rendered = _replace_unicode_symbols(line)
        if rendered:
            rendered_lines.append(rendered)
    if not rendered_lines:
        return None

    if len(rendered_lines) == 1:
        body = rendered_lines[0]
    else:
        body = "\\begin{aligned}\n" + " \\\\\n".join(rendered_lines) + "\n\\end{aligned}"
    if tag:
        body = f"{body}\n\\tag{{{tag}}}"
    return body.strip()
def _result(
    *,
    latex: Optional[str],
    confidence: float = 0.0,
    source: Optional[str] = None,
    validation_flags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    return {
        "latex": latex.strip() if isinstance(latex, str) and latex.strip() else None,
        "latex_confidence": round(float(confidence or 0.0), 3),
        "latex_source": source,
        "latex_validation_flags": [str(flag) for flag in (validation_flags or []) if str(flag).strip()],
    }


def extract_equation_latex(
    image_path: Optional[Path],
    *,
    fallback_text: str = "",
    equation_number: Optional[str] = None,
) -> Dict[str, Any]:
    del image_path
    if not _latex_enabled():
        return _result(latex=None, source="disabled", validation_flags=["latex_disabled"])

    backend = _latex_backend()
    failures: list[str] = []

    if backend == "text" and _latex_fallback_enabled():
        fallback_latex = fallback_text_to_latex(fallback_text, equation_number=equation_number)
        if fallback_latex:
            valid, flags = validate_equation_latex(fallback_latex)
            if valid:
                return _result(
                    latex=fallback_latex,
                    confidence=0.46,
                    source="text_fallback",
                    validation_flags=flags,
                )
            failures.extend(flags or ["text_fallback_rejected"])
        else:
            failures.append("text_fallback_empty")

    if not failures:
        failures.append("latex_unavailable")
    return _result(latex=None, source="unavailable", validation_flags=failures)
