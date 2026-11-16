from __future__ import annotations

import io
from pathlib import Path

from folio.distribute import _download
from folio.term import DownloadBar, _fmt_bytes, _fmt_eta


def test_fmt_bytes_and_eta():
    assert _fmt_bytes(512) == "512 B"
    assert "MB" in _fmt_bytes(157 * 1024 * 1024)
    assert _fmt_eta(65) == "01:05"
    assert _fmt_eta(None) == "--:--"


def test_download_bar_stays_on_one_line(monkeypatch):
    buf = io.StringIO()
    buf.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr("folio.term._term_cols", lambda: 100)

    bar = DownloadBar(1000, label="pack", width=12, stream=buf)
    bar.update(250)
    bar.update(250)
    bar.update(500)
    mid = buf.getvalue()
    # While updating: clear+CR only — no newline spam.
    assert mid.count("\n") == 0
    assert "\033[2K\r" in mid
    assert "█" in mid or "░" in mid

    bar.close()
    text = buf.getvalue()
    # Exactly one newline — the final close.
    assert text.count("\n") == 1
    assert "100%" in text.replace(" ", "")
    assert "done" in text
    assert "✓" in text


def test_download_bar_line_never_exceeds_terminal_width(monkeypatch):
    from folio.term import _visible_len

    buf = io.StringIO()
    buf.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr("folio.term._term_cols", lambda: 50)

    bar = DownloadBar(10**9, label="folio-index.tar.gz-very-long-name", width=30, stream=buf)
    bar.update(10**8)
    raw = buf.getvalue().split("\r")[-1]
    visible = raw.replace("\033[2K", "")
    assert _visible_len(visible) <= 50


def test_smooth_bar_monotonic():
    from folio.term import _smooth_bar

    a = _smooth_bar(0.0, 10)
    b = _smooth_bar(0.5, 10)
    c = _smooth_bar(1.0, 10)
    assert len(a) == len(b) == len(c) == 10
    assert c == "█" * 10
    assert a.count("█") <= b.count("█") <= c.count("█")


def test_download_writes_file_and_reports_progress(tmp_path: Path, monkeypatch):
    payload = b"x" * (1024 * 64)

    class FakeResp:
        headers = {"Content-Length": str(len(payload))}

        def __enter__(self):
            self._i = 0
            return self

        def __exit__(self, *args):
            return False

        def read(self, n: int) -> bytes:
            if self._i >= len(payload):
                return b""
            chunk = payload[self._i : self._i + n]
            self._i += len(chunk)
            return chunk

    monkeypatch.setattr(
        "folio.distribute.urllib.request.urlopen",
        lambda *a, **k: FakeResp(),
    )
    dest = tmp_path / "pack.tar.gz"
    monkeypatch.setenv("FOLIO_NO_PROGRESS", "1")
    _download("https://example.com/pack.tar.gz", dest, attempts=1)
    assert dest.read_bytes() == payload
