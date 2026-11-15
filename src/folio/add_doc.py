"""Add a local document (PDF / markdown / text) into the shared folio index."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__ as cli_version
from .chunk import Chunk, chunk_file, chunk_markdown
from .config import ensure_data_dirs
from .store import build_bm25, get_collection, load_bm25, save_bm25, upsert_chunks


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return (s[:64] or "local-doc").rstrip("-")


def _pdf_to_markdown(pdf: Path) -> str:
    """Extract full text via pdftotext; one markdown section per page."""
    if shutil.which("pdftotext") is None:
        raise RuntimeError(
            "pdftotext not found (poppler-utils). Install it to index PDFs, "
            "or convert the PDF to .md/.txt first."
        )
    with tempfile.TemporaryDirectory(prefix="folio-pdf-") as tmp:
        out = Path(tmp) / "doc.txt"
        subprocess.run(
            ["pdftotext", "-layout", str(pdf), str(out)],
            check=True,
            capture_output=True,
        )
        raw = out.read_text(encoding="utf-8", errors="replace")

    pages = raw.split("\f")
    parts: list[str] = [
        f"# {pdf.stem}\n",
        f"_Indexed from `{pdf.name}` · {len(pages)} pages_\n",
    ]
    for i, page in enumerate(pages, start=1):
        body = page.strip()
        if not body:
            continue
        parts.append(f"\n## Page {i}\n\n{body}\n")
    text = "\n".join(parts).strip() + "\n"
    if len(text) < 80:
        raise RuntimeError(f"PDF extracted almost no text: {pdf}")
    return text


def _load_as_markdown(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_to_markdown(path)
    if suffix in {".md", ".markdown", ".txt", ".text"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise RuntimeError(f"Unsupported file type {suffix!r} (use .pdf .md .txt)")


def _merge_bm25(cfg: dict[str, Any], new_chunks: list[Chunk]) -> None:
    """Rebuild BM25 from existing corpus + new chunks (replace same ids)."""
    path = Path(cfg["paths"]["bm25_path"])
    existing_ids: list[str] = []
    existing_texts: list[str] = []
    existing_metas: list[dict] = []
    try:
        payload = load_bm25(path)
        existing_ids = list(payload["ids"])
        existing_texts = list(payload["texts"])
        existing_metas = list(payload["metadatas"])
    except FileNotFoundError:
        pass

    drop = {c.chunk_id for c in new_chunks}
    kept_ids = [i for i in existing_ids if i not in drop]
    kept_texts = [t for i, t in zip(existing_ids, existing_texts) if i not in drop]
    kept_metas = [m for i, m in zip(existing_ids, existing_metas) if i not in drop]

    ids = kept_ids + [c.chunk_id for c in new_chunks]
    texts = kept_texts + [c.text for c in new_chunks]
    metas = kept_metas + [c.to_metadata() for c in new_chunks]

    merged: list[Chunk] = []
    for i, t, m in zip(ids, texts, metas):
        merged.append(
            Chunk(
                chunk_id=i,
                text=t,
                source_id=str(m.get("source_id", "local")),
                rel_path=str(m.get("rel_path", "")),
                heading_path=str(m.get("heading_path", "")),
                chunk_index=int(m.get("chunk_index", 0) or 0),
                content_hash=str(m.get("content_hash", "")),
                char_count=int(m.get("char_count", len(t)) or len(t)),
            )
        )
    bm25, out_ids, out_chunks = build_bm25(merged)
    save_bm25(path, bm25, out_ids, out_chunks)


def _update_manifest(
    cfg: dict[str, Any],
    *,
    source_id: str,
    rel_path: str,
    origin: str,
) -> None:
    manifest_path = Path(cfg["paths"]["manifest_path"])
    data: dict[str, Any] = {}
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))

    sources = list(data.get("sources") or [])
    if not any(s.get("id") == source_id for s in sources):
        sources.append(
            {
                "id": source_id,
                "repo": origin,
                "sparse_path": ".",
                "kind": "local",
            }
        )

    entry_path = f"{source_id}/{rel_path}"
    file_list = [
        f
        for f in (data.get("file_list") or [])
        if not (f.get("source_id") == source_id and f.get("path") == entry_path)
    ]
    file_list.append({"source_id": source_id, "path": entry_path})

    try:
        payload = load_bm25(cfg["paths"]["bm25_path"])
        chunk_count = len(payload["ids"])
    except FileNotFoundError:
        chunk_count = int(data.get("chunks") or 0)

    data.update(
        {
            "cli_version": cli_version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sources": sources,
            "file_list": file_list,
            "files": len(file_list),
            "chunks": chunk_count,
            "index_version": data.get("index_version")
            or datetime.now(timezone.utc).strftime("%Y.%m.%d"),
            "embedding_model": data.get("embedding_model")
            or cfg.get("embedding", {}).get("model", "onnx-minilm-l6-v2"),
            "collection": data.get("collection")
            or cfg.get("collection", {}).get("name", "folio"),
        }
    )
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def add_local_document(
    cfg: dict[str, Any],
    path: Path,
    *,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Convert a local file to markdown under sources/, embed, merge into BM25."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    ensure_data_dirs(cfg)
    sid = source_id or _slug(path.stem)
    sources_dir: Path = cfg["paths"]["sources_dir"]
    dest_dir = sources_dir / sid
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_md = dest_dir / "document.md"

    md = _load_as_markdown(path)
    dest_md.write_text(md, encoding="utf-8")
    (dest_dir / "ORIGIN.txt").write_text(f"{path}\n", encoding="utf-8")

    chunk_cfg = cfg["chunking"]
    chunks = chunk_file(
        dest_md,
        source_id=sid,
        sources_root=dest_dir,
        max_chars=int(chunk_cfg["max_chars"]),
        overlap_chars=int(chunk_cfg["overlap_chars"]),
        min_chars=int(chunk_cfg["min_chars"]),
    )
    if not chunks:
        chunks = chunk_markdown(
            md,
            source_id=sid,
            rel_path="document.md",
            max_chars=int(chunk_cfg["max_chars"]),
            overlap_chars=int(chunk_cfg["overlap_chars"]),
            min_chars=min(40, int(chunk_cfg["min_chars"])),
        )
    if not chunks:
        raise RuntimeError(f"No chunks produced from {path}")

    collection = get_collection(cfg, create=True)
    n = upsert_chunks(
        collection,
        chunks,
        batch_size=int(cfg["embedding"]["batch_size"]),
    )
    _merge_bm25(cfg, chunks)
    _update_manifest(cfg, source_id=sid, rel_path="document.md", origin=str(path))
    print(f"[add] source={sid} file={path.name} chunks={n} → {dest_md}")
    return {"source_id": sid, "chunks": n, "path": str(dest_md)}
