"""Terminal helpers: OSC 8 hyperlinks + single-line download progress."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from .config import package_dir


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


def _term_cols() -> int:
    try:
        return max(40, shutil.get_terminal_size(fallback=(80, 24)).columns)
    except Exception:
        return 80


def _ansi(enabled: bool, code: str, text: str) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def _visible_len(text: str) -> int:
    """Length ignoring ANSI color sequences (for wrap-safe truncation)."""
    import re

    return len(re.sub(r"\033\[[0-9;]*m", "", text))


def _smooth_bar(frac: float, width: int) -> str:
    """Partial-block progress track (█ + ▏▎▍▌▋▊▉)."""
    frac = min(1.0, max(0.0, frac))
    phases = " ▏▎▍▌▋▊▉█"
    exact = frac * width
    whole = int(exact)
    rem = int(round((exact - whole) * 8))
    rem = min(8, max(0, rem))
    if whole >= width:
        return "█" * width
    if rem >= 8:
        whole += 1
        if whole >= width:
            return "█" * width
        return "█" * whole + "░" * (width - whole)
    if rem <= 0:
        return "█" * whole + "░" * (width - whole)
    return "█" * whole + phases[rem] + "░" * (width - whole - 1)


class DownloadBar:
    """One in-place progress line (never spams newlines).

    Uses CR + erase-line so the terminal keeps a single chart. Visible length is
    capped to the terminal width so wrapping cannot flood the screen.
    """

    def __init__(
        self,
        total: int | None,
        *,
        label: str = "download",
        width: int = 22,
        stream=None,
    ) -> None:
        self.total = int(total) if total and total > 0 else None
        # Short label keeps the chart readable on narrow terminals.
        self.label = "folio-index" if label.startswith("folio-index") else label
        self.width = max(10, width)
        self.stream = stream or sys.stderr
        self.done = 0
        self._t0 = time.monotonic()
        self._last_draw = 0.0
        self._closed = False
        self._enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self._color = self._enabled and os.environ.get("NO_COLOR") is None
        if os.environ.get("FOLIO_NO_PROGRESS", "").strip().lower() in {"1", "true", "yes"}:
            self._enabled = False

    def update(self, n: int) -> None:
        if self._closed:
            return
        self.done += max(0, int(n))
        now = time.monotonic()
        # Throttle redraws; first chunk always paints.
        if self._enabled and (self._last_draw == 0.0 or now - self._last_draw >= 0.15):
            self._draw(finished=False)
            self._last_draw = now

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._enabled:
            self._draw(finished=True)
            self.stream.write("\n")
            self.stream.flush()
            return
        if self.done:
            elapsed = max(time.monotonic() - self._t0, 1e-6)
            rate = self.done / elapsed
            total = self.total or self.done
            self.stream.write(
                f"[download] {self.label}: {_fmt_bytes(self.done)}"
                f"{'' if self.total is None else f' / {_fmt_bytes(total)}'}"
                f" ({_fmt_bytes(rate)}/s)\n"
            )
            self.stream.flush()

    def _render(self, *, finished: bool) -> str:
        elapsed = max(time.monotonic() - self._t0, 1e-6)
        rate = self.done / elapsed
        c = self._color
        if self.total:
            frac = 1.0 if finished else min(1.0, self.done / self.total)
            bar = _smooth_bar(frac, self.width)
            pct = int(round(frac * 100))
            eta = "done" if finished else _fmt_eta((self.total - self.done) / rate if rate > 0 else None)
            # Teal track + bright percent (folio palette).
            bar_col = _ansi(c, "36;1", bar)  # bright cyan
            pct_col = _ansi(c, "32;1" if finished else "36;1", f"{pct:3d}%")
            meta = _ansi(
                c,
                "90",
                f"{_fmt_bytes(self.done)}/{_fmt_bytes(self.total)}  {_fmt_bytes(rate)}/s  ETA {eta}",
            )
            mark = _ansi(c, "32", "✓") if finished else _ansi(c, "36", "↓")
            return f"{mark} {self.label}  {bar_col}  {pct_col}  {meta}"

        # Unknown size — soft moving pulse, still one line.
        pos = int((time.monotonic() * 8) % self.width)
        cells = ["░"] * self.width
        for i in range(max(0, pos - 2), min(self.width, pos + 3)):
            cells[i] = "█"
        bar = _ansi(c, "36;1", "".join(cells))
        meta = _ansi(c, "90", f"{_fmt_bytes(self.done)}  {_fmt_bytes(rate)}/s")
        return f"{_ansi(c, '36', '↓')} {self.label}  {bar}  {meta}"

    def _draw(self, *, finished: bool) -> None:
        # Cap visible width so wrap + CR cannot flood the screen.
        cols = _term_cols()
        text = self._render(finished=finished)
        while _visible_len(text) > cols - 1 and self.width > 8:
            self.width -= 2
            text = self._render(finished=finished)
        if _visible_len(text) > cols - 1:
            # Last resort: strip colors and hard-trim.
            import re

            plain = re.sub(r"\033\[[0-9;]*m", "", text)
            text = plain[: cols - 4] + "..."
        # Erase whole line, then carriage-return overwrite (single chart).
        self.stream.write("\033[2K\r" + text)
        self.stream.flush()
