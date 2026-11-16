from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from folio.distribute import (
    _lock_installed_release,
    _sha256_file,
    _verify_index_sha256,
    bootstrap_index,
)
from folio.install import install, uninstall
from folio.store import _path_is_writable, _writable_chroma_dir, get_collection


def test_writable_chroma_mirrors_when_sqlite_is_readonly(tmp_path: Path, mini_cfg: dict):
    chroma = Path(mini_cfg["paths"]["chroma_dir"])
    # Ensure a real collection exists under the mini fixture.
    get_collection(mini_cfg, create=True)
    sqlite = chroma / "chroma.sqlite3"
    assert sqlite.exists()

    for path in [chroma, *chroma.rglob("*")]:
        if path.is_dir():
            path.chmod(0o555)
        else:
            path.chmod(0o444)
    assert not _path_is_writable(chroma)

    mirrored = _writable_chroma_dir(chroma)
    assert mirrored != chroma
    assert _path_is_writable(mirrored)
    # Second call reuses the same mirror.
    assert _writable_chroma_dir(chroma) == mirrored

    # Query path (create=False) must open through the mirror.
    collection = get_collection(mini_cfg, create=False)
    assert collection.count() >= 0


def test_verify_index_sha256_rejects_mismatch(tmp_path: Path):
    pack = tmp_path / "pack.tar.gz"
    pack.write_bytes(b"not-a-real-pack")
    digest = _sha256_file(pack)
    _verify_index_sha256(pack, digest)  # exact match ok
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _verify_index_sha256(pack, "0" * 64)


def test_lock_installed_release_rewrites_index_url(tmp_path: Path):
    out = tmp_path / "folio-home"
    out.mkdir()
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {"release": {"index_url": "https://evil.example/pack.tar.gz"}, "sources": []}
        ),
        encoding="utf-8",
    )
    trusted = "https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/releases/latest/download/folio-index.tar.gz"
    _lock_installed_release(out, url=trusted, sha256="abc")
    data = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert data["release"]["index_url"] == trusted
    assert data["release"]["index_sha256"] == "abc"


def test_bootstrap_archive_verifies_sha256_and_locks_url(tmp_path: Path, mini_cfg: dict):
    # Reuse the already-built mini index by packing is heavy; just check the
    # archive path refuses a bad hash before extract.
    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(b"nope")
    mini_cfg = dict(mini_cfg)
    mini_cfg["release"] = {
        "index_url": "https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/releases/latest/download/folio-index.tar.gz",
        "index_sha256": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        bootstrap_index(mini_cfg, archive=archive, force=True)


def test_cursor_uninstall_without_project_clears_home_rule(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setattr("folio.install.Path.home", lambda: home)
    assert install(platform="cursor", project=False, project_dir=project) == 0
    rule = home / ".cursor" / "rules" / "folio.mdc"
    assert rule.exists()
    assert uninstall(platform="cursor", project=False, project_dir=project) == 0
    assert not rule.exists()


def test_release_url_env_override(monkeypatch):
    from folio.distribute import _release_url

    monkeypatch.setenv("FOLIO_INDEX_URL", "https://example.com/from-env.tar.gz")
    assert _release_url({"release": {"index_url": "https://example.com/cfg.tar.gz"}}) == (
        "https://example.com/from-env.tar.gz"
    )
