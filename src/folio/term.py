"""Terminal helpers: OSC 8 hyperlinks (clickable in modern terminals)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .config import package_dir

# Block characters for a smooth linear download chart.
_BAR_FILL = "█"
_BAR_EMPTY = "░"


def banner_text() -> str:
    """HyperTerminal-style FOLIO splash (GONET-like @/ block letters)."""
    path = package_dir() / "assets" / "banner.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").rstrip() + "\n"
    return "folio\n"


def print_banner() -> None:
    print(banner_text(), end="")


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


def _fmt_bytes(n: float) -> str:
    units = ("B", "KB", "MB", "GB")
    value = float(max(0, n))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    total = int(seconds)
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class DownloadBar:
    """Attractive single-line linear progress chart for large downloads."""

    def __init__(
        self,
        total: int | None,
        *,
        label: str = "download",
        width: int = 28,
        stream=None,
    ) -> None:
        self.total = int(total) if total and total > 0 else None
        self.label = label
        self.width = max(10, width)
        self.stream = stream or sys.stderr
        self.done = 0
        self._t0 = time.monotonic()
        self._last_draw = 0.0
        self._enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        if os.environ.get("FOLIO_NO_PROGRESS", "").strip().lower() in {"1", "true", "yes"}:
            self._enabled = False

    def update(self, n: int) -> None:
        self.done += max(0, int(n))
        now = time.monotonic()
        # Redraw often enough to feel live; always draw on first bytes.
        if self._enabled and (self.done == n or now - self._last_draw >= 0.08):
            self._draw(finished=False)
            self._last_draw = now

    def close(self) -> None:
        if self._enabled:
            self._draw(finished=True)
            self.stream.write("\n")
            self.stream.flush()
        elif self.done:
            # Non-TTY (CI/logs): one summary line.
            elapsed = max(time.monotonic() - self._t0, 1e-6)
            rate = self.done / elapsed
            total = self.total or self.done
            self.stream.write(
                f"⬇️  {self.label}: {_fmt_bytes(self.done)}"
                f"{'' if self.total is None else f' / {_fmt_bytes(total)}'}"
                f"  ({_fmt_bytes(rate)}/s)\n"
            )
            self.stream.flush()

    def _draw(self, *, finished: bool) -> None:
        elapsed = max(time.monotonic() - self._t0, 1e-6)
        rate = self.done / elapsed
        if self.total:
            frac = min(1.0, self.done / self.total)
            filled = int(round(frac * self.width))
            bar = _BAR_FILL * filled + _BAR_EMPTY * (self.width - filled)
            pct = f"{frac * 100:5.1f}%"
            eta = _fmt_eta((self.total - self.done) / rate if rate > 0 else None)
            line = (
                f"\r⬇️  {self.label}  │{bar}│ {pct}  "
                f"{_fmt_bytes(self.done)}/{_fmt_bytes(self.total)}  "
                f"{_fmt_bytes(rate)}/s  ETA {eta}   "
            )
        else:
            # Unknown size — still show a moving pulse + bytes + speed.
            pos = int((time.monotonic() * 8) % self.width)
            cells = [_BAR_EMPTY] * self.width
            for i in range(max(0, pos - 2), min(self.width, pos + 3)):
                cells[i] = _BAR_FILL
            bar = "".join(cells)
            line = (
                f"\r⬇️  {self.label}  │{bar}│  "
                f"{_fmt_bytes(self.done)}  {_fmt_bytes(rate)}/s   "
            )
        if finished and self.total:
            bar = _BAR_FILL * self.width
            line = (
                f"\r⬇️  {self.label}  │{bar}│ 100.0%  "
                f"{_fmt_bytes(self.done)}/{_fmt_bytes(self.total)}  "
                f"{_fmt_bytes(rate)}/s  done   "
            )
        self.stream.write(line)
        self.stream.flush()
