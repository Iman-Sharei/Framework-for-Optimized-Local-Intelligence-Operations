from __future__ import annotations

from folio.cli import main
from folio.term import banner_text


def test_banner_text_contains_folio_block():
    text = banner_text()
    assert "@@@@@" in text
    assert "LOCAL HYBRID RAG" in text


def test_cli_banner(capsys):
    assert main(["banner"]) == 0
    out = capsys.readouterr().out
    assert "@@@@@" in out
    assert "ELECTRONIC KNOWLEDGE RETRIEVAL SERVICE" in out
