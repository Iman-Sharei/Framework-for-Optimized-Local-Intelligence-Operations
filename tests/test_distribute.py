from __future__ import annotations

from pathlib import Path

import pytest

from folio.distribute import bootstrap_index, pack_index
from folio.ingest import ingest
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
    assert (other / "bm25" / "index.json").exists() or (other / "bm25" / "index.pkl").exists()
    assert (other / "config.yaml").exists()


def test_bootstrap_without_url_errors(mini_cfg: dict):
    mini_cfg["release"] = {"index_url": ""}
    with pytest.raises(RuntimeError, match="No release URL"):
        bootstrap_index(mini_cfg, force=True)
