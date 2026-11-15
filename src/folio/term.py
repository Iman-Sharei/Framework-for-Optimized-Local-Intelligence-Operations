"""Terminal helpers: OSC 8 hyperlinks (clickable in modern terminals)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _links_enabled() -> bool:
    if os.environ.get("FOLIO_NO_HYPERLINKS", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def hyperlink(url: str, label: str | None = None) -> str:
    """Return an OSC 8 hyperlink when stdout is a TTY; otherwise plain label/url."""
    text = label if label is not None else url
    if not url:
        return text
    if not _links_enabled():
        if label is None or label == url:
            return url
        return f"{text} ({url})"
    # OSC 8: ESC ] 8 ; ; URL ST  label ESC ] 8 ; ; ST
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def path_link(path: Path | str, label: str | None = None) -> str:
    p = Path(path).resolve()
    return hyperlink(p.as_uri(), label if label is not None else str(p))


def url_link(url: str, label: str | None = None) -> str:
    return hyperlink(url, label)
