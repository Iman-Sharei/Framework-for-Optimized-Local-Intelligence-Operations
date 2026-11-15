from __future__ import annotations

from folio.retrieve import Hit, format_context, rrf_fuse


def test_rrf_prefers_docs_ranked_high_in_both_lists():
    fused = rrf_fuse(
        [
            ["a", "b", "c"],
            ["b", "a", "d"],
        ],
        k=60,
    )
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["c"]
    assert fused["b"] >= fused["a"]  # first in one list + second in other
    assert "d" in fused


def test_rrf_empty_lists():
    assert rrf_fuse([]) == {}
    assert rrf_fuse([[], []]) == {}


def test_format_context_includes_citations_and_respects_budget():
    hits = [
        Hit(
            chunk_id="1",
            text="alpha " * 50,
            score=0.1,
            source_id="s1",
            rel_path="a.md",
            heading_path="H1",
        ),
        Hit(
            chunk_id="2",
            text="beta " * 50,
            score=0.05,
            source_id="s2",
            rel_path="b.md",
            heading_path="H2",
        ),
    ]
    out = format_context(hits, max_chars=10_000)
    assert "file=a.md" in out
    assert "heading=H1" in out
    assert "source=" not in out  # never leak corpus names in human output
    tiny = format_context(hits, max_chars=80)
    assert "file=a.md" in tiny or "[1]" in tiny
    assert "source=" not in tiny
    assert "[1]" in tiny
    # second block dropped when budget is tight
    assert "[2]" not in tiny
