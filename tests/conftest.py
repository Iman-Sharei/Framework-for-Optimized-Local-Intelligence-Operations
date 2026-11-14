from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

from folio.config import ensure_data_dirs, load_config, write_default_config


@pytest.fixture
def folio_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated FOLIO_HOME so tests never touch ~/.folio."""
    home = tmp_path / "folio-home"
    home.mkdir()
    monkeypatch.setenv("FOLIO_HOME", str(home))
    # Reuse ONNX weights via symlink (avoid copying ~300MB per test / disk quota).
    shared = Path.home() / ".folio" / "model-cache"
    dest = home / "model-cache"
    if shared.exists() and not dest.exists():
        try:
            dest.symlink_to(shared, target_is_directory=True)
        except OSError:
            # Fall back: point CHROMA cache at the shared dir without copying.
            monkeypatch.setenv("CHROMA_CACHE_DIR", str(shared))
            monkeypatch.setenv("XDG_CACHE_HOME", str(shared))
    return home


@pytest.fixture
def mini_cfg(folio_home: Path) -> dict:
    """Config pointing at FOLIO_HOME with tiny retrieval knobs for speed."""
    write_default_config(folio_home)
    cfg = load_config(out_dir=folio_home)
    cfg["sources"] = [
        {
            "id": "demo",
            "repo": "local",
            "sparse_path": "lessons",
            "branch": "main",
        }
    ]
    cfg["chunking"] = {"max_chars": 400, "overlap_chars": 40, "min_chars": 20}
    cfg["embedding"] = {"model": "onnx-minilm-l6-v2", "batch_size": 16}
    cfg["collection"] = {"name": "folio_test"}
    cfg["retrieval"] = {
        "dense_k": 5,
        "bm25_k": 5,
        "final_k": 3,
        "rrf_k": 60,
        "crag_min_best_score": 0.0,  # disable widen in unit-ish tests
        "crag_widen_factor": 2,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "confidence_floor": 0.0,
    }
    cfg["release"] = {"index_url": ""}
    ensure_data_dirs(cfg)
    (folio_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "sources": cfg["sources"],
                "release": cfg["release"],
                "chunking": cfg["chunking"],
                "embedding": cfg["embedding"],
                "collection": cfg["collection"],
                "retrieval": cfg["retrieval"],
            }
        ),
        encoding="utf-8",
    )
    return load_config(out_dir=folio_home)


def _git_init_with_tree(root: Path, files: dict[str, str]) -> Path:
    """Create a tiny git repo with the given relative path → content map."""
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "folio@test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "folio"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    # Ensure branch name is main for clone --branch main
    subprocess.run(["git", "branch", "-M", "main"], cwd=root, check=True)
    return root


@pytest.fixture
def sample_curriculum_repo(tmp_path: Path) -> Path:
    return _git_init_with_tree(
        tmp_path / "curriculum.git",
        {
            "lessons/mod-01/01-rag.md": """
            # Hybrid RAG
            Hybrid retrieval combines BM25 lexical search with dense embeddings.
            Reciprocal Rank Fusion (RRF) merges the two ranked lists.
            CRAG widens recall when the best fused score is weak.
            """,
            "lessons/mod-01/02-agents.md": """
            # Agents vs Workflows
            Prefer a workflow when steps are deterministic.
            Prefer an agent when tool choice must adapt at runtime.
            Guardrails and idempotency matter for write tools.
            """,
            "lessons/mod-02/mlops.md": """
            # MLOps Serving
            Use MLflow for experiment tracking and a model registry.
            Deploy with vLLM or Triton behind Kubernetes for GPU serving.
            """,
            "README.md": "# ignore me outside sparse path\n",
        },
    )


@pytest.fixture
def full_tree_repo(tmp_path: Path) -> Path:
    return _git_init_with_tree(
        tmp_path / "full.git",
        {
            "01-intro/README.md": "# Intro\nMLOps maturity model and course setup.\n" * 3,
            "02-tracking/README.md": "# Tracking\nMLflow tracking server and registry.\n" * 3,
            "notes.txt": "not markdown",
        },
    )
