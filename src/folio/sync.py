from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _clone_url(repo: str) -> str:
    """Normalize local paths to file:// so sparse/depth clones behave."""
    path = Path(repo)
    if path.exists():
        return path.resolve().as_uri()
    return repo


def _is_full_tree(sparse: str) -> bool:
    return sparse.strip() in {".", "", "/"}


def _content_root(dest: Path, sparse: str) -> Path:
    if _is_full_tree(sparse):
        return dest
    return dest / sparse


def sync_sources(cfg: dict[str, Any], force: bool = False) -> list[Path]:
    """Sparse-clone / pull each configured GitHub path into data/sources/<id>/."""
    sources_dir: Path = cfg["paths"]["sources_dir"]
    sources_dir.mkdir(parents=True, exist_ok=True)
    synced: list[Path] = []

    for src in cfg["sources"]:
        dest = sources_dir / src["id"]
        sparse = src["sparse_path"]
        branch = src.get("branch", "main")
        repo = _clone_url(src["repo"])
        full = _is_full_tree(sparse)

        if dest.exists() and force:
            shutil.rmtree(dest)

        if dest.exists() and (dest / ".git").exists():
            _run(["git", "fetch", "--depth", "1", "origin", branch], cwd=dest)
            if full:
                # materialize whole tree (sparse "." only keeps cone root)
                _run(["git", "sparse-checkout", "disable"], cwd=dest)
                _run(["git", "checkout", f"origin/{branch}", "--", "."], cwd=dest)
            else:
                _run(["git", "checkout", f"origin/{branch}", "--", sparse], cwd=dest)
                _run(["git", "sparse-checkout", "set", sparse], cwd=dest)
        else:
            if dest.exists():
                shutil.rmtree(dest)
            if full:
                _run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--branch",
                        branch,
                        repo,
                        str(dest),
                    ]
                )
            else:
                dest.mkdir(parents=True, exist_ok=True)
                _run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "--filter=blob:none",
                        "--sparse",
                        "--branch",
                        branch,
                        repo,
                        str(dest),
                    ]
                )
                _run(["git", "sparse-checkout", "set", sparse], cwd=dest)

        content_root = _content_root(dest, sparse)
        if not content_root.exists():
            raise FileNotFoundError(f"Expected sparse path missing: {content_root}")
        synced.append(content_root)
        print(f"[sync] {src['id']} → {content_root}")

    return synced


def iter_markdown(cfg: dict[str, Any]) -> list[tuple[str, Path]]:
    """Return (source_id, absolute md path) for every markdown file under synced roots.

    Also includes any extra directories under sources/ (e.g. local PDFs via ``folio add``).
    """
    sources_dir: Path = cfg["paths"]["sources_dir"]
    out: list[tuple[str, Path]] = []
    seen_ids: set[str] = set()
    for src in cfg["sources"]:
        sid = src["id"]
        seen_ids.add(sid)
        root = _content_root(sources_dir / sid, src["sparse_path"])
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            if path.is_file() and ".git" not in path.parts:
                out.append((sid, path))

    if sources_dir.exists():
        for child in sorted(sources_dir.iterdir()):
            if not child.is_dir() or child.name in seen_ids or child.name.startswith("."):
                continue
            for path in sorted(child.rglob("*.md")):
                if path.is_file() and ".git" not in path.parts:
                    out.append((child.name, path))
    return out
