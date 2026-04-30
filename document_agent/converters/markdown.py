from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID


def frontmatter(
    *,
    job_id: UUID,
    batch_id: Optional[UUID],
    filename: str,
    detected_type: str,
    asset_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        "---",
        f"job_id: {job_id}",
    ]
    if batch_id:
        lines.append(f"batch_id: {batch_id}")
    lines.extend(
        [
            f"source_filename: {filename!r}",
            f"detected_type: {detected_type}",
            f"generated_at: {datetime.now(timezone.utc).isoformat()}",
            "converter_version: document-agent/0.1.0",
            f"asset_count: {asset_count}",
        ]
    )
    if extra:
        for key, value in sorted(extra.items()):
            if isinstance(value, (str, int, float, bool)) or value is None:
                lines.append(f"{key}: {value!r}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def with_frontmatter(body: str, **kwargs: Any) -> str:
    return frontmatter(**kwargs) + body.strip() + "\n"


def rewrite_frontmatter_fields(markdown: str, fields: Dict[str, Any]) -> str:
    if not markdown.startswith("---\n"):
        return markdown
    try:
        end = markdown.index("\n---\n", 4)
    except ValueError:
        return markdown
    front = markdown[4:end].splitlines()
    seen: set[str] = set()
    updated = []
    for line in front:
        key = line.split(":", 1)[0].strip()
        if key in fields:
            value = fields[key]
            rendered = repr(value) if isinstance(value, str) else str(value)
            updated.append(f"{key}: {rendered}")
            seen.add(key)
        else:
            updated.append(line)
    for key, value in fields.items():
        if key in seen:
            continue
        rendered = repr(value) if isinstance(value, str) else str(value)
        updated.append(f"{key}: {rendered}")
    return "---\n" + "\n".join(updated) + markdown[end:]
