from __future__ import annotations

import os
import re
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

def workflow_revision() -> str | None:
    """Return the triggering GitHub Actions commit when running under Actions."""
    value = os.environ.get("GITHUB_SHA", "").strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None
