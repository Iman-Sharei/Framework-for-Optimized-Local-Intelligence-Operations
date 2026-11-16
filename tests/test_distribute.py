from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
import yaml

from folio.distribute import bootstrap_index, pack_index
from folio.ingest import ingest
from folio.retrieve import retrieve
from folio.store import load_bm25
from folio.sync import sync_sources


@pytest.mark.integration
def test_pack_and_bootstrap_roundtrip(
    mini_cfg: dict, sample_curriculum_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    ingest(mini_cfg, rebuild=True)

    archive = tmp_path / "folio-index-test.tar.gz"
    packed = pack_index(mini_cfg, dest=archive)
    assert packed.exists() and packed.stat().st_size > 1000

    other = tmp_path / "other-home"
    other.mkdir()
    shared = Path.home() / ".folio" / "model-cache"
    if shared.exists():
        (other / "model-cache").symlink_to(shared, target_is_directory=True)

    monkeypatch.setenv("FOLIO_HOME", str(other))
    from folio.config import ensure_data_dirs, load_config, write_default_config

    write_default_config(other)
    cfg2 = load_config(out_dir=other)
    ensure_data_dirs(cfg2)
    bootstrap_index(cfg2, archive=packed, force=True)

    assert (other / "manifest.json").exists()
    assert (other / "chroma").exists()
    assert (other / "bm25" / "index.json").exists()
    assert (other / "config.yaml").exists()

    with tarfile.open(packed, "r:gz") as tar:
        packed_manifest = json.load(tar.extractfile("manifest.json"))
        packed_config = yaml.safe_load(tar.extractfile("config.yaml").read().decode("utf-8"))
        packed_bm25 = json.load(tar.extractfile("bm25/index.json"))
    assert "sources" not in packed_manifest
    assert "file_list" not in packed_manifest
    assert packed_config["sources"] == []
    assert packed_manifest["embedding"]["window_tokens"] == 200
    assert packed_bm25["texts"]
    for meta in packed_bm25["metadatas"]:
        assert "source_id" not in meta
        assert "rel_path" not in meta
        assert "demo" not in json.dumps(meta)
    assert all(not str(chunk_id).startswith("demo") for chunk_id in packed_bm25["ids"])
    local = json.loads(mini_cfg["paths"]["manifest_path"].read_text(encoding="utf-8"))
    assert any(item.get("id") == "demo" for item in local["sources"])
    local_bm25 = load_bm25(mini_cfg["paths"]["bm25_path"])
    assert any(str(chunk_id).startswith("demo") for chunk_id in local_bm25["ids"])

    cfg2 = load_config(out_dir=other)
    hits = retrieve(cfg2, "hybrid retrieval BM25 dense RRF", final_k=3)
    assert hits
    assert all(hit.source_id == "" for hit in hits)
    assert any("RAG" in hit.text or "RRF" in hit.text for hit in hits)


def test_release_url_falls_back_to_bundled_when_empty():
    from folio.distribute import _release_url

    assert _release_url({"release": {"index_url": ""}}).startswith("https://")
    assert "folio-index.tar.gz" in _release_url({"release": {"index_url": ""}})
    custom = "https://example.com/pack.tar.gz"
    assert _release_url({"release": {"index_url": custom}}) == custom


def test_bootstrap_rejects_non_https_url(mini_cfg: dict):
    with pytest.raises(RuntimeError, match="non-https"):
        bootstrap_index(mini_cfg, url="file:///tmp/x.tar.gz", force=True)
