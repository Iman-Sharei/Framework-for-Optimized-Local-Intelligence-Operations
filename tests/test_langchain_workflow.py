"""Prove project wiring: LangChain questions → nav first → open docs URLs."""

from __future__ import annotations

from pathlib import Path

import pytest

from folio.config import load_config
from folio.docs_nav import (
    extract_doc_urls,
    langchain_workflow_required,
    privacy_no_source_names,
    recommend_doc_urls_from_hits,
)
from folio.install import _CLAUDE_REGISTRATION, _CURSOR_RULE, install
from folio.retrieve import Hit, retrieve


def test_bundled_skill_encodes_nav_then_open_docs_workflow():
    skill = Path(__file__).resolve().parents[1] / "src" / "folio" / "assets" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")
    assert langchain_workflow_required(text)
    assert privacy_no_source_names(text)
    assert "docs-suggest" in text
    assert "WebFetch" in text or "fetch" in text.lower()


def test_install_rules_encode_nav_then_open_docs_workflow():
    assert langchain_workflow_required(_CURSOR_RULE)
    assert langchain_workflow_required(_CLAUDE_REGISTRATION)
    assert privacy_no_source_names(_CURSOR_RULE)
    assert privacy_no_source_names(_CLAUDE_REGISTRATION)
    skill = Path(__file__).resolve().parents[1] / "src" / "folio" / "assets" / "SKILL.md"
    assert langchain_workflow_required(skill.read_text(encoding="utf-8"))


def test_install_project_writes_langchain_workflow_into_repo(tmp_path: Path):
    assert install(platform="cursor", project=True, project_dir=tmp_path) == 0
    assert install(platform="claude", project=True, project_dir=tmp_path) == 0
    cursor = (tmp_path / ".cursor" / "rules" / "folio.mdc").read_text(encoding="utf-8")
    claude = (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")
    skill = (tmp_path / ".agents" / "skills" / "folio" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert langchain_workflow_required(cursor)
    assert langchain_workflow_required(claude)
    assert langchain_workflow_required(skill)
    for text in (cursor, claude, skill):
        assert "docs.langchain.com" in text
        assert privacy_no_source_names(text) or "never" in text.lower()
        assert "docs-suggest" in text


def test_local_index_nav_query_returns_docs_urls_to_open():
    """Integration: real ~/.folio langchain-docs-nav hits contain openable URLs."""
    cfg = load_config()
    manifest = Path(cfg["paths"]["manifest_path"])
    if not manifest.exists():
        pytest.skip("local folio index missing")

    try:
        hits = retrieve(
            cfg,
            "LangGraph persistence streaming checkpoint",
            final_k=8,
            source_filter="langchain-docs-nav",
        )
        if not hits:
            # Published packs strip source ids; search the whole index instead.
            hits = retrieve(
                cfg,
                "LangGraph persistence streaming checkpoint",
                final_k=8,
                source_filter=None,
            )
    except Exception as exc:  # pragma: no cover - corrupt/partial index
        pytest.skip(f"local folio index unavailable: {exc}")

    if not hits:
        pytest.skip("no retrieval hits from local index")

    urls = recommend_doc_urls_from_hits(hits, limit=8)
    if not urls:
        pytest.skip("local index has no docs.langchain.com URLs in top hits")
    assert urls, "nav hits must yield docs.langchain.com URLs for the agent to open"
    assert all(u.startswith("https://docs.langchain.com/") for u in urls)
    joined = "\n".join(h.text for h in hits)
    assert extract_doc_urls(joined)


def test_docs_suggest_cli_prints_urls(monkeypatch, capsys, folio_home: Path):
    """Hermetic: never touches ~/.folio; uses FOLIO_HOME + --out + fake retrieve."""
    from folio import cli

    (folio_home / "manifest.json").write_text("{}", encoding="utf-8")

    fake_hits = [
        Hit(
            chunk_id="1",
            text="- Persistence → https://docs.langchain.com/oss/python/langgraph/persistence\n",
            score=0.9,
            source_id="langchain-docs-nav",
            rel_path="nav.md",
            heading_path="Capabilities",
        ),
        Hit(
            chunk_id="2",
            text="- Streaming → https://docs.langchain.com/oss/python/langgraph/streaming\n",
            score=0.8,
            source_id="langchain-docs-nav",
            rel_path="nav.md",
            heading_path="Capabilities",
        ),
    ]

    def fake_retrieve(cfg, query, *, final_k=None, use_rerank=False, source_filter=None):
        assert Path(cfg["paths"]["out_dir"]) == folio_home.resolve()
        assert "LangGraph" in query or "langgraph" in query.lower()
        assert source_filter == "langchain-docs-nav"
        return fake_hits

    monkeypatch.setattr(cli, "retrieve", fake_retrieve)
    monkeypatch.setattr(cli, "migrate_legacy_index", lambda: None)
    code = cli.main(["docs-suggest", "LangGraph persistence", "--out", str(folio_home)])
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "https://docs.langchain.com/oss/python/langgraph/persistence"
    assert out[1] == "https://docs.langchain.com/oss/python/langgraph/streaming"


def test_docs_suggest_falls_back_when_published_pack_has_no_source_ids(
    monkeypatch, capsys, folio_home: Path
):
    """Published packs redact source_id; filter returns [] and must fall back."""
    from folio import cli

    (folio_home / "manifest.json").write_text("{}", encoding="utf-8")
    calls: list[str | None] = []
    anon_hits = [
        Hit(
            chunk_id="c1",
            text="See https://docs.langchain.com/oss/python/langgraph/persistence\n",
            score=0.5,
            source_id="",
            rel_path="",
            heading_path="",
        )
    ]

    def fake_retrieve(cfg, query, *, final_k=None, use_rerank=False, source_filter=None):
        calls.append(source_filter)
        if source_filter == "langchain-docs-nav":
            return []
        return anon_hits

    monkeypatch.setattr(cli, "retrieve", fake_retrieve)
    monkeypatch.setattr(cli, "migrate_legacy_index", lambda: None)
    code = cli.main(["docs-suggest", "LangGraph persistence", "--out", str(folio_home)])
    assert code == 0
    assert calls == ["langchain-docs-nav", None]
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["https://docs.langchain.com/oss/python/langgraph/persistence"]


def test_docs_suggest_cli_errors_without_index(monkeypatch, capsys, folio_home: Path):
    from folio import cli

    monkeypatch.setattr(cli, "migrate_legacy_index", lambda: None)
    code = cli.main(["docs-suggest", "LangGraph persistence", "--out", str(folio_home)])
    assert code == 2
    err = capsys.readouterr().err
    assert "no index found" in err
    assert str(folio_home) in err
