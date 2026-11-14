from __future__ import annotations

from pathlib import Path

from folio.chunk import chunk_file, chunk_markdown


def test_chunk_splits_on_headings_and_keeps_path():
    text = """
# Module
Intro paragraph that is long enough to survive min_chars filtering for preamble? Wait.

## Hybrid retrieval
BM25 finds rare tokens. Dense captures semantic neighbors. Together they improve recall.

## RRF
RRF merges ranks without calibrating incompatible score scales between retrievers.
""".strip()
    chunks = chunk_markdown(
        text,
        source_id="demo",
        rel_path="lessons/a.md",
        max_chars=500,
        overlap_chars=20,
        min_chars=30,
    )
    assert chunks
    headings = {c.heading_path for c in chunks}
    assert any("Hybrid retrieval" in h for h in headings)
    assert any("RRF" in h for h in headings)
    assert all(c.source_id == "demo" for c in chunks)
    assert all(c.rel_path == "lessons/a.md" for c in chunks)
    assert all(c.char_count == len(c.text) for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_size_guard_splits_long_section():
    body = "word " * 400  # ~2000 chars
    text = f"# Long\n{body}"
    chunks = chunk_markdown(
        text,
        source_id="demo",
        rel_path="long.md",
        max_chars=300,
        overlap_chars=40,
        min_chars=20,
    )
    assert len(chunks) >= 2
    assert all(c.char_count <= 300 + 40 + 5 for c in chunks)  # allow overlap stitch


def test_chunk_drops_tiny_fragments():
    text = "# T\nx\n\n## Keep\n" + ("enough content here to pass the minimum character threshold. " * 2)
    chunks = chunk_markdown(
        text,
        source_id="demo",
        rel_path="t.md",
        max_chars=400,
        overlap_chars=10,
        min_chars=40,
    )
    assert all(c.char_count >= 40 for c in chunks)


def test_chunk_file_relative_path(tmp_path: Path):
    root = tmp_path / "lessons"
    root.mkdir()
    path = root / "mod" / "x.md"
    path.parent.mkdir()
    path.write_text("# Title\n" + ("Useful sentence about retrieval and ranking. " * 5), encoding="utf-8")
    chunks = chunk_file(
        path,
        source_id="demo",
        sources_root=root,
        max_chars=400,
        overlap_chars=20,
        min_chars=20,
    )
    assert chunks
    assert chunks[0].rel_path == "mod/x.md"
