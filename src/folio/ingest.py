from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from tqdm import tqdm

from . import __version__ as cli_version
from .chunk import Chunk, chunk_file
from .config import ensure_data_dirs
from .store import build_bm25, reset_collection, save_bm25, upsert_chunks
from .sync import iter_markdown


def index_version_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y.%m.%d")


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
        )
        all_chunks.extend(chunks)

    collection = reset_collection(cfg) if rebuild else reset_collection(cfg)
    n = upsert_chunks(collection, all_chunks, batch_size=int(cfg["embedding"]["batch_size"]))
    bm25, ids, chunks = build_bm25(all_chunks)
    save_bm25(cfg["paths"]["bm25_path"], bm25, ids, chunks)

    index_version = index_version_now()
    manifest = {
        "index_version": index_version,
        "cli_version": cli_version,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "files": len(files),
        "chunks": n,
        "embedding_model": cfg["embedding"]["model"],
        "collection": cfg["collection"]["name"],
        "sources": [
            {
                "id": s["id"],
                "repo": s["repo"],
                "sparse_path": s["sparse_path"],
            }
            for s in cfg["sources"]
        ],
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
