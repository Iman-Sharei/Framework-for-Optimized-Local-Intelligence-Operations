from __future__ import annotations

from pathlib import Path

from folio.term import hyperlink, path_link, url_link


def test_hyperlink_plain_when_not_tty(monkeypatch):
    monkeypatch.setattr("folio.term._links_enabled", lambda: False)
    assert hyperlink("https://example.com") == "https://example.com"
    assert hyperlink("https://example.com", "Example") == "Example (https://example.com)"


def test_hyperlink_osc8_when_enabled(monkeypatch):
    monkeypatch.setattr("folio.term._links_enabled", lambda: True)
    out = url_link("https://pypi.org/project/folio-cli/", "PyPI")
    assert out.startswith("\033]8;;https://pypi.org/project/folio-cli/\033\\")
    assert "PyPI" in out
    assert out.endswith("\033]8;;\033\\")


def test_path_link_uses_file_uri(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("folio.term._links_enabled", lambda: True)
    p = tmp_path / "SKILL.md"
    p.write_text("x", encoding="utf-8")
    out = path_link(p)
    assert p.resolve().as_uri() in out
    assert str(p.resolve()) in out
