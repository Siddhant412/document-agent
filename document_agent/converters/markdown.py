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

