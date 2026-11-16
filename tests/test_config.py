from __future__ import annotations

from pathlib import Path

import pytest

from folio.config import default_home, load_config, resolve_out_dir, write_default_config


def test_resolve_out_dir_prefers_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    target = tmp_path / "custom"
    monkeypatch.setenv("FOLIO_HOME", str(target))
    assert resolve_out_dir() == target.resolve()


def test_resolve_out_dir_explicit_beats_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("FOLIO_HOME", str(tmp_path / "env"))
    explicit = tmp_path / "explicit"
    assert resolve_out_dir(explicit) == explicit.resolve()


def test_default_home_is_dot_folio():
    assert default_home() == Path.home() / ".folio"


def test_sources_local_file_is_used_and_not_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "sources.local.yaml").write_text(
        "sources:\n  - id: local-only\n    repo: /tmp/notes\n    sparse_path: .\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg = load_config(out_dir=home)
    assert cfg["sources"][0]["id"] == "local-only"
    assert cfg["sources"][0]["repo"] == "/tmp/notes"


def test_load_config_materializes_paths(folio_home: Path):
    write_default_config(folio_home)
    cfg = load_config(out_dir=folio_home)
    assert cfg["paths"]["out_dir"] == folio_home.resolve()
    assert cfg["paths"]["chroma_dir"] == folio_home / "chroma"
    assert cfg["paths"]["bm25_path"] == folio_home / "bm25" / "index.json"
    assert cfg["paths"]["manifest_path"] == folio_home / "manifest.json"
    assert cfg["sources"] == []

# path-view refresh
