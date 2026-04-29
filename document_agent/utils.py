from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable, List

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, default: str = "document") -> str:
    raw = Path(name or default).name
    normalized = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    cleaned = _SAFE_NAME_RE.sub("_", normalized)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = re.sub(r"_+(\.[A-Za-z0-9]+)$", r"\1", cleaned).strip("._")
    return cleaned or default


def markdown_filename(name: str, fallback: str = "document") -> str:
    safe = safe_filename(name, default=fallback)
    stem = Path(safe).stem or fallback
    return f"{stem}.md"


def unique_names(names: Iterable[str]) -> List[str]:
    used = {}
    output = []
    for name in names:
        candidate = markdown_filename(name)
        stem = Path(candidate).stem
        suffix = Path(candidate).suffix
        count = used.get(candidate, 0)
        if count:
            while True:
                next_name = f"{stem}-{count + 1}{suffix}"
                if next_name not in used:
                    candidate = next_name
                    break
                count += 1
        used[candidate] = used.get(candidate, 0) + 1
        output.append(candidate)
    return output


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size
