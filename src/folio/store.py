from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rank_bm25 import BM25Okapi

# Placeholder so an empty document does not make BM25 divide by zero.
_EMPTY_TOKEN = "__folio_empty__"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# MiniLM hard window is 256. Stay under it so the tail of a passage is embedded.
EMBED_WINDOW_TOKENS = 200
EMBED_WINDOW_OVERLAP = 40
EMBED_MODEL_MAX_TOKENS = 256

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


def tokenize_text(text: str) -> list[str]:
    """Lowercase word tokens. Punctuation is not part of the token.

    ``zygomorphic?`` and ``zygomorphic`` match. A blank document becomes one
    placeholder token so BM25Okapi can still be constructed.
    """
    tokens = _TOKEN_RE.findall((text or "").lower())
    return tokens or [_EMPTY_TOKEN]


def _clamp_nonpositive_idf(bm25: BM25Okapi) -> BM25Okapi:
    """rank_bm25's epsilon floor is negative when every idf is negative.

    Matching documents then score below zero and a ``> 0`` filter drops them.
    Pin those idf values to a small positive floor.
    """
    idf = getattr(bm25, "idf", None)
    if not isinstance(idf, dict):
        return bm25
    for term, value in list(idf.items()):
        try:
            if float(value) <= 0:
                idf[term] = 1e-3
        except (TypeError, ValueError):
            continue
    return bm25


def make_bm25(tokenized: list[list[str]]) -> BM25Okapi:
    if not tokenized:
        tokenized = [[_EMPTY_TOKEN]]
    safe = [doc if doc else [_EMPTY_TOKEN] for doc in tokenized]
    return _clamp_nonpositive_idf(BM25Okapi(safe))


def _token_windows(text: str) -> list[str]:
    """Split text into MiniLM windows so the tail is not truncated away."""
    raw = text or ""
    if not raw.strip():
        return [raw]
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    model = ONNXMiniLM_L6_V2()
    tokenizer = model.tokenizer
    tokenizer.no_truncation()
    tokenizer.no_padding()
    try:
        encoded = tokenizer.encode(raw)
        ids = list(encoded.ids)
        offsets = list(encoded.offsets)
        if len(ids) <= EMBED_MODEL_MAX_TOKENS:
            return [raw]
        windows: list[str] = []
        start = 0
        step = max(1, EMBED_WINDOW_TOKENS - EMBED_WINDOW_OVERLAP)
        while start < len(ids):
            end = min(start + EMBED_WINDOW_TOKENS, len(ids))
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            piece = raw[char_start:char_end].strip()
            if piece:
                windows.append(piece)
            if end >= len(ids):
                break
            start += step
        return windows or [raw]
    finally:
        tokenizer.enable_truncation(max_length=256)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)


class TailAwareEmbedding(DefaultEmbeddingFunction):
    """Mean-pool MiniLM windows so tokens past 256 still change the vector."""

    def __call__(self, input):  # noqa: A003 - chromadb embedding signature
        texts = ["" if item is None else str(item) for item in input]
        windows: list[str] = []
        counts: list[int] = []
        for text in texts:
            parts = _token_windows(text)
            counts.append(len(parts))
            windows.extend(parts)
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

        vectors = np.asarray(ONNXMiniLM_L6_V2()(windows), dtype=np.float64)
        pooled: list[np.ndarray] = []
        cursor = 0
        for count in counts:
            block = vectors[cursor : cursor + count]
            cursor += count
            mean = block.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                mean = mean / norm
            pooled.append(mean.astype(np.float32))
        return np.stack(pooled, axis=0)


def get_embedding_fn(model_cache_dir: Path | None = None) -> DefaultEmbeddingFunction:
    _ensure_local_chroma_cache(model_cache_dir)
    return TailAwareEmbedding()


def _chroma_client(chroma_dir: Path) -> chromadb.PersistentClient:
    # Telemetry writes hit the same SQLite file; disable to keep query usable
    # under readonly / sandboxed mounts.
    from chromadb.config import Settings

    settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True,
        persist_directory=str(chroma_dir),
    )
    return chromadb.PersistentClient(path=str(chroma_dir), settings=settings)


def get_collection(cfg: dict[str, Any], *, create: bool = True) -> Collection:
    cache = Path(cfg["paths"]["model_cache_dir"])
    _ensure_local_chroma_cache(cache)
    client = _chroma_client(Path(cfg["paths"]["chroma_dir"]))
    emb = get_embedding_fn(cache)
    name = cfg["collection"]["name"]
    if not create:
        return client.get_collection(name=name, embedding_function=emb)
    return client.get_or_create_collection(
        name=name,
        embedding_function=emb,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(cfg: dict[str, Any]) -> Collection:
    cache = Path(cfg["paths"]["model_cache_dir"])
    _ensure_local_chroma_cache(cache)
    client = _chroma_client(Path(cfg["paths"]["chroma_dir"]))
    name = cfg["collection"]["name"]
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return get_collection(cfg, create=True)


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
    tokenized = [tokenize_text(c.text) for c in chunks]
    return make_bm25(tokenized), [c.chunk_id for c in chunks], chunks


def bm25_index_path(path: Path) -> Path:
    """Canonical on-disk BM25 path (JSON). Accepts legacy ``*.pkl`` configs."""
    if path.suffix == ".json":
        return path
    return path.with_suffix(".json")


def _bm25_json_path(path: Path) -> Path:
    return bm25_index_path(path)


def opaque_chunk_id(old_id: str) -> str:
    """Stable id with no source name in it."""
    return "c" + hashlib.sha256(old_id.encode("utf-8")).hexdigest()[:16]


def redact_chunk_metadata(meta: dict | None, new_id: str, *, keep_headings: bool) -> dict:
    """Drop source ids, file paths, and repo URLs. Optionally keep a heading title."""
    src = meta or {}
    out: dict[str, Any] = {
        "chunk_id": new_id,
        "chunk_index": int(src.get("chunk_index") or 0),
        "char_count": int(src.get("char_count") or 0),
    }
    if keep_headings and src.get("heading_path"):
        out["heading_path"] = str(src["heading_path"])
    return out


def anonymize_bm25_file(path: Path, *, keep_headings: bool) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = list(data.get("ids") or [])
    texts = list(data.get("texts") or [])
    metas = list(data.get("metadatas") or [])
    if len(texts) != len(ids) or len(metas) != len(ids):
        raise RuntimeError(
            f"BM25 lists disagree (ids={len(ids)} texts={len(texts)} metas={len(metas)}); refusing to pack"
        )
    new_ids = [opaque_chunk_id(str(i)) for i in ids]
    new_metas = [
        redact_chunk_metadata(m if isinstance(m, dict) else {}, new_id, keep_headings=keep_headings)
        for m, new_id in zip(metas, new_ids)
    ]
    path.write_text(
        json.dumps({"ids": new_ids, "texts": texts, "metadatas": new_metas}, ensure_ascii=False),
        encoding="utf-8",
    )
    for extra in path.parent.glob("*.pkl"):
        extra.unlink()


def anonymize_chroma_dir(chroma_dir: Path, collection_name: str, *, keep_headings: bool) -> None:
    """Rewrite a copied Chroma directory so ids and metadata do not name sources.

    Embeddings and document text are kept. The text itself can still contain titles.
    """
    client = _chroma_client(chroma_dir)
    collection = client.get_collection(name=collection_name)
    total = int(collection.count())
    if total == 0:
        return
    # Page the read. One unbounded get() can stop early and the pack would
    # silently drop the rest of the vectors.
    page = 256
    offset = 0
    rows: list[tuple[str, str, dict, Any]] = []
    while offset < total:
        data = collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=page,
            offset=offset,
        )
        ids = list(data.get("ids") or [])
        if not ids:
            break
        documents = list(data.get("documents") or [])
        metadatas = list(data.get("metadatas") or [])
        embeddings = data.get("embeddings")
        if embeddings is None or len(documents) != len(ids) or len(metadatas) != len(ids):
            raise RuntimeError("Chroma page was incomplete; refusing to pack without vectors")
        for i, doc_id in enumerate(ids):
            rows.append((str(doc_id), documents[i], metadatas[i] or {}, embeddings[i]))
        offset += len(ids)
    if len(rows) != total:
        raise RuntimeError(f"Chroma get returned {len(rows)} of {total}; refusing to pack")
    space = "cosine"
    try:
        meta = collection.metadata or {}
        space = str(meta.get("hnsw:space") or "cosine")
    except Exception:
        space = "cosine"
    # Close the old database before replacing it. Delete+VACUUM keeps the old
    # ids inside segment blobs, so the published directory is a new database.
    del collection
    del client
    fresh = chroma_dir.parent / f"{chroma_dir.name}-redacted"
    if fresh.exists():
        shutil.rmtree(fresh)
    new_client = _chroma_client(fresh)
    new_collection = new_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": space},
    )
    new_ids = [opaque_chunk_id(row[0]) for row in rows]
    new_metas = [
        redact_chunk_metadata(row[2] if isinstance(row[2], dict) else {}, new_id, keep_headings=keep_headings)
        for row, new_id in zip(rows, new_ids)
    ]
    batch = 64
    for start in range(0, len(new_ids), batch):
        end = start + batch
        new_collection.add(
            ids=new_ids[start:end],
            documents=[row[1] for row in rows[start:end]],
            metadatas=new_metas[start:end],
            embeddings=[row[3] for row in rows[start:end]],
        )
    del new_collection
    del new_client
    shutil.rmtree(chroma_dir)
    fresh.rename(chroma_dir)


def anonymize_index_copy(root: Path, collection_name: str, *, keep_headings: bool) -> None:
    anonymize_bm25_file(root / "bm25" / "index.json", keep_headings=keep_headings)
    anonymize_chroma_dir(root / "chroma", collection_name, keep_headings=keep_headings)


def save_bm25(path: Path, bm25: BM25Okapi, ids: list[str], chunks: list[Chunk]) -> Path:
    """Persist BM25 corpus as JSON and rebuild the index on load (no pickle)."""
    del bm25  # rebuilt from texts on load
    json_path = bm25_index_path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ids": ids,
        "texts": [c.text for c in chunks],
        "metadatas": [c.to_metadata() for c in chunks],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # Remove legacy pickle so packs don't ship an RCE surface.
    pkl_path = path if path.suffix == ".pkl" else path.with_suffix(".pkl")
    if pkl_path.exists():
        pkl_path.unlink()
    return json_path


def load_bm25(path: Path) -> dict[str, Any]:
    json_path = bm25_index_path(path)
    pkl_path = path if path.suffix == ".pkl" else path.with_suffix(".pkl")

    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        texts: list[str] = data["texts"]
        ids: list[str] = data["ids"]
        metas: list[dict] = data["metadatas"]
        tokenized = [tokenize_text(t) for t in texts]
        bm25 = make_bm25(tokenized)
        return {"bm25": bm25, "ids": ids, "texts": texts, "metadatas": metas}

    if pkl_path.exists():
        # Legacy pickle: restricted unpickler only (blocks RCE gadgets).
        with pkl_path.open("rb") as f:
            payload = _RestrictedBm25Unpickler(f).load()
        if not isinstance(payload, dict) or "bm25" not in payload:
            raise RuntimeError(f"Invalid legacy BM25 pickle at {pkl_path}")
        # One-shot migrate to JSON so status/query stay consistent.
        try:
            texts = list(payload.get("texts") or [])
            ids = list(payload.get("ids") or [])
            metas = list(payload.get("metadatas") or [])
            if texts and ids and len(texts) == len(ids):
                json_path.write_text(
                    json.dumps(
                        {"ids": ids, "texts": texts, "metadatas": metas},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                pkl_path.unlink(missing_ok=True)
        except Exception:
            pass
        return payload

    raise FileNotFoundError(json_path if path.suffix == ".json" else path)
