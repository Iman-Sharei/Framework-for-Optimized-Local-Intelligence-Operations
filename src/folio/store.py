from __future__ import annotations

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


def save_bm25(path: Path, bm25: BM25Okapi, ids: list[str], chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bm25": bm25,
        "ids": ids,
        "texts": [c.text for c in chunks],
        "metadatas": [c.to_metadata() for c in chunks],
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_bm25(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)
