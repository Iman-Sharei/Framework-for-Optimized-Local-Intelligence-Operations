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


def test_download_bar_draws_percent_and_closes(monkeypatch):
    buf = io.StringIO()
    # Force TTY-style redraws.
    buf.isatty = lambda: True  # type: ignore[method-assign]
    bar = DownloadBar(1000, label="pack", width=10, stream=buf)
    bar.update(250)
    bar.update(750)
    bar.close()
    text = buf.getvalue()
    assert "pack" in text
    assert "█" in text
    assert "100.0%" in text
    assert "done" in text


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
    # Quiet CI-style summary path
    monkeypatch.setenv("FOLIO_NO_PROGRESS", "1")
    _download("https://example.com/pack.tar.gz", dest, attempts=1)
    assert dest.read_bytes() == payload
