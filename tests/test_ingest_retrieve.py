from __future__ import annotations

from pathlib import Path

import pytest

from folio.ingest import ingest
from folio.retrieve import retrieve
from folio.store import bm25_index_path
from folio.sync import sync_sources


@pytest.mark.integration
def test_ingest_and_hybrid_retrieve_returns_relevant_citations(
    mini_cfg: dict, sample_curriculum_repo: Path
):
    mini_cfg["sources"] = [
        {
            "id": "demo",
            "repo": str(sample_curriculum_repo),
            "sparse_path": "lessons",
            "branch": "main",
        }
    ]
    sync_sources(mini_cfg, force=True)
    manifest = ingest(mini_cfg, rebuild=True)

    assert manifest["files"] >= 3
    assert manifest["chunks"] >= 3
    assert "index_version" in manifest
    assert "cli_version" in manifest
    assert mini_cfg["paths"]["manifest_path"].exists()
    # Ingest writes JSON even if a legacy config still named the path *.pkl.
    assert bm25_index_path(mini_cfg["paths"]["bm25_path"]).exists()

    hits = retrieve(mini_cfg, "hybrid RAG BM25 dense RRF fusion", final_k=3)
    assert hits, "expected at least one hit for a curriculum-aligned query"
    assert any("rag" in h.rel_path.lower() or "RAG" in h.text or "RRF" in h.text for h in hits)
    assert all(h.source_id == "demo" for h in hits)
    assert hits[0].score >= hits[-1].score


@pytest.mark.integration
def test_source_filter_limits_results(mini_cfg: dict, sample_curriculum_repo: Path):
    mini_cfg["sources"] = [
        {
            "id": "demo",
            "repo": str(sample_curriculum_repo),
            "sparse_path": "lessons",
            "branch": "main",
        }
    ]
    sync_sources(mini_cfg, force=True)
    ingest(mini_cfg, rebuild=True)

    none = retrieve(mini_cfg, "agents workflows guardrails", source_filter="other")
    assert none == []
    some = retrieve(mini_cfg, "agents workflows guardrails", source_filter="demo")
    assert some
