from __future__ import annotations

from folio.retrieve import Hit, format_context, hits_to_json_payload, rrf_fuse


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


def test_json_payload_respects_token_budget():
    hits = [
        Hit(
            chunk_id=str(i),
            text=("word " * 400),
            score=1.0 - i * 0.01,
            source_id=f"s{i}",
            rel_path=f"{i}.md",
            heading_path="H",
        )
        for i in range(8)
    ]
    payload = hits_to_json_payload(hits, token_budget=2000)
    assert payload
    assert len(payload) < 8
    blob = __import__("json").dumps(payload, ensure_ascii=False)
    assert len(blob) <= 2000 * 4 + 50
