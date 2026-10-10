from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def source_revision(root: Path) -> str | None:
    """Return the exact Git commit used for a run, or None outside a checkout."""
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
