from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi

from .chunk import Chunk

# Legacy pickle allowlist (BM25Okapi is pure-Python; block __reduce__ gadgets).
_PICKLE_ALLOWED = {
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "tuple"),
    ("builtins", "str"),
    ("builtins", "int"),
    ("builtins", "float"),
    ("builtins", "bool"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("rank_bm25", "BM25Okapi"),
}


class _RestrictedBm25Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _PICKLE_ALLOWED:
            raise pickle.UnpicklingError(f"Refusing to unpickle {module}.{name}")
        return super().find_class(module, name)


def _ensure_local_chroma_cache(model_cache_dir: Path | None = None) -> Path:
    """Keep ONNX model cache under the folio index home (sandbox-friendly).

    Chroma binds ``ONNXMiniLM_L6_V2.DOWNLOAD_PATH`` at import time from Path.home(),
    so we rewrite that class attribute to a folio-local directory.
    """
    cache = (model_cache_dir or Path.home() / ".folio" / "model-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Do NOT rewrite $HOME — that breaks Path.home() for the whole process.
    os.environ["CHROMA_CACHE_DIR"] = str(cache)
    os.environ["XDG_CACHE_HOME"] = str(cache)

    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    download = cache / "onnx_models" / ONNXMiniLM_L6_V2.MODEL_NAME
    download.mkdir(parents=True, exist_ok=True)
    ONNXMiniLM_L6_V2.DOWNLOAD_PATH = download

    real_home = Path.home()
    for candidate in (
        real_home / ".cache/chroma/onnx_models" / ONNXMiniLM_L6_V2.MODEL_NAME,
        real_home / ".folio" / "model-cache" / "onnx_models" / ONNXMiniLM_L6_V2.MODEL_NAME,
        real_home
        / ".folio"
        / "model-cache"
        / "home"
        / ".cache"
        / "chroma"
        / "onnx_models"
        / ONNXMiniLM_L6_V2.MODEL_NAME,
        Path(__file__).resolve().parents[2] / "data/model-cache/onnx_models" / ONNXMiniLM_L6_V2.MODEL_NAME,
    ):
        if (candidate / "onnx" / "model.onnx").exists() and not (
            download / "onnx" / "model.onnx"
        ).exists():
            shutil.copytree(candidate, download, dirs_exist_ok=True)
            break
    return cache


def get_embedding_fn(model_cache_dir: Path | None = None) -> DefaultEmbeddingFunction:
    _ensure_local_chroma_cache(model_cache_dir)
    return DefaultEmbeddingFunction()


def get_collection(cfg: dict[str, Any]) -> Collection:
    cache = Path(cfg["paths"]["model_cache_dir"])
    _ensure_local_chroma_cache(cache)
    client = chromadb.PersistentClient(path=str(cfg["paths"]["chroma_dir"]))
    emb = get_embedding_fn(cache)
    return client.get_or_create_collection(
        name=cfg["collection"]["name"],
        embedding_function=emb,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(cfg: dict[str, Any]) -> Collection:
    cache = Path(cfg["paths"]["model_cache_dir"])
    _ensure_local_chroma_cache(cache)
    client = chromadb.PersistentClient(path=str(cfg["paths"]["chroma_dir"]))
    name = cfg["collection"]["name"]
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return get_collection(cfg)


def upsert_chunks(collection: Collection, chunks: list[Chunk], batch_size: int = 64) -> int:
    if not chunks:
        return 0
    from tqdm import tqdm

    for i in tqdm(range(0, len(chunks), batch_size), desc="embedding+upsert"):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.to_metadata() for c in batch],
        )
    return len(chunks)


def build_bm25(chunks: list[Chunk]) -> tuple[BM25Okapi, list[str], list[Chunk]]:
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized), [c.chunk_id for c in chunks], chunks


def _bm25_json_path(path: Path) -> Path:
    """Prefer index.json alongside (or instead of) legacy index.pkl."""
    if path.suffix == ".json":
        return path
    return path.with_suffix(".json")


def save_bm25(path: Path, bm25: BM25Okapi, ids: list[str], chunks: list[Chunk]) -> None:
    """Persist BM25 corpus as JSON and rebuild the index on load (no pickle)."""
    del bm25  # rebuilt from texts on load
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = _bm25_json_path(path)
    payload = {
        "ids": ids,
        "texts": [c.text for c in chunks],
        "metadatas": [c.to_metadata() for c in chunks],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # Remove legacy pickle so packs don't ship an RCE surface.
    if path.suffix == ".pkl" and path.exists():
        path.unlink()


def load_bm25(path: Path) -> dict[str, Any]:
    json_path = _bm25_json_path(path)

    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        texts: list[str] = data["texts"]
        ids: list[str] = data["ids"]
        metas: list[dict] = data["metadatas"]
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized) if tokenized else BM25Okapi([["_"]])
        return {"bm25": bm25, "ids": ids, "texts": texts, "metadatas": metas}

    if not path.exists():
        raise FileNotFoundError(path)

    # Legacy pickle path: restricted unpickler only (blocks RCE gadgets).
    with path.open("rb") as f:
        payload = _RestrictedBm25Unpickler(f).load()
    if not isinstance(payload, dict) or "bm25" not in payload:
        raise RuntimeError(f"Invalid legacy BM25 pickle at {path}")
    return payload
