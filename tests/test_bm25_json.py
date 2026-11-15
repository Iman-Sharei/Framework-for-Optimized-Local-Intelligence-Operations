from __future__ import annotations

from pathlib import Path

from folio.chunk import Chunk
from folio.store import bm25_index_path, build_bm25, load_bm25, save_bm25


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"c{i}",
        text=f"hybrid retrieval BM25 dense RRF chunk {i}",
        source_id="demo",
        rel_path=f"lessons/{i}.md",
        heading_path="H1",
        chunk_index=i,
        content_hash=f"h{i}",
        char_count=40,
    )


def test_save_bm25_writes_json_even_when_config_path_is_pkl(tmp_path: Path):
    legacy = tmp_path / "bm25" / "index.pkl"
    chunks = [_chunk(0), _chunk(1)]
    bm25, ids, out = build_bm25(chunks)
    written = save_bm25(legacy, bm25, ids, out)

    assert written == bm25_index_path(legacy)
    assert written.suffix == ".json"
    assert written.exists()
    assert not legacy.exists()

    payload = load_bm25(legacy)
    assert payload["ids"] == ids
    assert len(payload["texts"]) == 2
