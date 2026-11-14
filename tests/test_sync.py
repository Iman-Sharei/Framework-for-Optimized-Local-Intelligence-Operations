from __future__ import annotations

from pathlib import Path

from folio.sync import iter_markdown, sync_sources


def test_sparse_sync_only_materializes_lessons(
    mini_cfg: dict, sample_curriculum_repo: Path, folio_home: Path
):
    mini_cfg["sources"] = [
        {
            "id": "demo",
            "repo": str(sample_curriculum_repo),
            "sparse_path": "lessons",
            "branch": "main",
        }
    ]
    roots = sync_sources(mini_cfg, force=True)
    assert len(roots) == 1
    assert roots[0].name == "lessons"
    md = iter_markdown(mini_cfg)
    paths = {p.name for _, p in md}
    assert "01-rag.md" in paths
    assert "02-agents.md" in paths
    assert "mlops.md" in paths
    # README at repo root is outside sparse path
    assert "README.md" not in paths
    assert all(sid == "demo" for sid, _ in md)


def test_full_tree_sync_with_dot_sparse_path(
    mini_cfg: dict, full_tree_repo: Path, folio_home: Path
):
    mini_cfg["sources"] = [
        {
            "id": "zoom",
            "repo": str(full_tree_repo),
            "sparse_path": ".",
            "branch": "main",
        }
    ]
    roots = sync_sources(mini_cfg, force=True)
    assert roots[0] == folio_home / "sources" / "zoom"
    md = iter_markdown(mini_cfg)
    names = sorted(p.name for _, p in md)
    assert names == ["README.md", "README.md"]
    assert len(md) == 2
    # non-md ignored
    assert not any(p.suffix == ".txt" for _, p in md)
