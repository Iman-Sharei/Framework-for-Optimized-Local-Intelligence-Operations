from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm

from . import __version__ as cli_version
from .chunk import Chunk, chunk_file
from .config import ensure_data_dirs
from .store import (
    EMBED_MODEL_MAX_TOKENS,
    EMBED_WINDOW_OVERLAP,
    EMBED_WINDOW_TOKENS,
    build_bm25,
    get_collection,
    reset_collection,
    save_bm25,
    upsert_chunks,
)
from .add_doc import _merge_bm25
from .sync import iter_markdown


def index_version_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y.%m.%d")


def index_build_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    """Embedding and chunking facts recorded in manifest.json."""
    chunk = cfg.get("chunking") or {}
    model = (cfg.get("embedding") or {}).get("model") or "onnx-minilm-l6-v2"

    def _opt_int(key: str) -> int | None:
        value = chunk.get(key)
        if value in (None, ""):
            return None
        return int(value)

    return {
        "embedding_model": model,
        "embedding": {
            "model": model,
            "provider": "chromadb-onnx-minilm-l6-v2",
            "max_tokens": EMBED_MODEL_MAX_TOKENS,
            "window_tokens": EMBED_WINDOW_TOKENS,
            "window_overlap_tokens": EMBED_WINDOW_OVERLAP,
            "pooling": "mean",
        },
        "chunking": {
            "max_chars": int(chunk.get("max_chars", 1200)),
            "overlap_chars": int(chunk.get("overlap_chars", 150)),
            "min_chars": int(chunk.get("min_chars", 80)),
            "max_tokens": _opt_int("max_tokens"),
            "overlap_tokens": _opt_int("overlap_tokens"),
        },
    }


def manifest_source_entry(src: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "id": src.get("id"),
        "repo": src.get("repo"),
        "sparse_path": src.get("sparse_path", "."),
    }
    for key in ("kind", "branch", "recovered_from"):
        if src.get(key):
            entry[key] = src[key]
    return entry


def ingest(cfg: dict[str, Any], *, rebuild: bool = True) -> dict[str, Any]:
    ensure_data_dirs(cfg)
    files = iter_markdown(cfg)
    if not files:
        raise RuntimeError("No markdown files found. Run sync first.")

    chunk_cfg = cfg["chunking"]
    all_chunks: list[Chunk] = []
    sources_dir = cfg["paths"]["sources_dir"]

    for source_id, path in tqdm(files, desc="chunking"):
        chunks = chunk_file(
            path,
            source_id=source_id,
            sources_root=sources_dir / source_id,
            max_chars=int(chunk_cfg["max_chars"]),
            overlap_chars=int(chunk_cfg["overlap_chars"]),
            min_chars=int(chunk_cfg["min_chars"]),
            max_tokens=int(chunk_cfg["max_tokens"]) if chunk_cfg.get("max_tokens") else None,
            overlap_tokens=int(chunk_cfg["overlap_tokens"]) if chunk_cfg.get("overlap_tokens") else None,
        )
        all_chunks.extend(chunks)

    if rebuild:
        collection = reset_collection(cfg)
    else:
        collection = get_collection(cfg, create=True)
    n = upsert_chunks(collection, all_chunks, batch_size=int(cfg["embedding"]["batch_size"]))
    if rebuild:
        bm25, ids, chunks = build_bm25(all_chunks)
        save_bm25(cfg["paths"]["bm25_path"], bm25, ids, chunks)
    else:
        _merge_bm25(cfg, all_chunks)
        n = len(all_chunks)

    index_version = index_version_now()
    manifest = {
        "index_version": index_version,
        "cli_version": cli_version,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "files": len(files),
        "chunks": n,
        "collection": cfg["collection"]["name"],
        **index_build_metadata(cfg),
        "sources": [manifest_source_entry(s) for s in cfg["sources"]],
        "file_list": [
            {"source_id": sid, "path": str(p.relative_to(sources_dir))}
            for sid, p in files
        ],
    }
    cfg["paths"]["manifest_path"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[ingest] index_version={index_version} files={len(files)} "
        f"chunks={n} chroma={cfg['paths']['chroma_dir']}"
    )
    return manifest
