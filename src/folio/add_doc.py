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


def _delete_source_vectors(collection, source_id: str) -> None:
    """Remove every chunk already stored for this source before a re-add."""
    try:
        existing = collection.get(where={"source_id": source_id}, include=[])
    except Exception:
        return
    ids = list(existing.get("ids") or [])
    if ids:
        collection.delete(ids=ids)


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


def _walk_sidebar_links(nodes: list, trail: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    out: list[tuple[tuple[str, ...], str]] = []
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        title = str(node.get("title") or node.get("name") or "").strip()
        ntype = node.get("type")
        url = node.get("url")
        children = node.get("children")
        next_trail = trail + ((title,) if title else ())
        if url and (ntype == "link" or title):
            out.append((next_trail, str(url)))
        if children:
            out.extend(_walk_sidebar_links(children, next_trail))
    return out


def _sidebars_json_to_markdown(path: Path) -> str:
    """Turn a LangChain/Mintlify sidebars.json export into retrieval-friendly markdown.

    Each nav link becomes a searchable line with section trail + URL so assistants
    can recommend which official docs pages to open before coding.
    """
    import json
    from collections import defaultdict

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "pages" not in data:
        raise RuntimeError(f"Not a folio docs-nav sidebars.json: {path}")

    generated = data.get("generated_at") or "unknown"
    source = data.get("source") or path.name
    buckets: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    seen_urls: set[str] = set()

    for page in data.get("pages") or []:
        base = tuple(str(x) for x in (page.get("section_trail") or []) if x)
        lang = str(page.get("language") or "").strip()
        for trail, url in _walk_sidebar_links(page.get("sidebar") or []):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            leaf = trail[-1] if trail else url.rsplit("/", 1)[-1]
            parent = " > ".join([x for x in (base + trail[:-1]) if x]) or "Docs"
            buckets[parent].append((leaf, url, lang))

    parts: list[str] = [
        "# LangChain ecosystem docs navigation map\n",
        f"_Generated: {generated}_\n",
        f"_Source: {source}_\n",
        f"_Unique doc URLs: {len(seen_urls)}_\n",
        "\n## How assistants must use this index\n",
        "When the user asks about LangChain, LangGraph, LangSmith, Deep Agents, "
        "or related framework APIs / implementation:\n",
        "1. `folio query \"<task> LangChain docs\" --source langchain-docs-nav --json`\n",
        "2. From hits, collect the official `https://docs.langchain.com/...` URLs\n",
        "3. Open/WebFetch **those pages** (current docs) before writing code\n",
        "4. Still use other folio curricula for architecture trade-offs\n",
        "\nDo **not** invent outdated LangChain APIs from memory when this nav index "
        "returns matching doc URLs.\n",
    ]

    for parent in sorted(buckets.keys()):
        parts.append(f"\n## {parent}\n")
        for leaf, url, lang in buckets[parent]:
            lang_bit = f" ({lang})" if lang else ""
            # Repeat parent keywords so BM25/dense match section + leaf.
            parts.append(
                f"- **{leaf}**{lang_bit} — docs page for `{leaf}` in `{parent}` → {url}\n"
            )

    text = "\n".join(parts).strip() + "\n"
    if len(seen_urls) < 10:
        raise RuntimeError(f"sidebars.json produced too few links ({len(seen_urls)})")
    return text


def _load_as_markdown(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_to_markdown(path)
    if suffix in {".md", ".markdown", ".txt", ".text"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        return _sidebars_json_to_markdown(path)
    raise RuntimeError(f"Unsupported file type {suffix!r} (use .pdf .md .txt .json)")


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

    # Chunk ids include a content hash, so an edit mints new ids. Drop the whole
    # previous source, not only ids that happen to match the new batch.
    source_ids = {c.source_id for c in new_chunks}
    drop = {c.chunk_id for c in new_chunks}

    def _keep(i: str, m: dict) -> bool:
        if i in drop:
            return False
        if str(m.get("source_id", "")) in source_ids:
            return False
        return True

    kept_ids = [i for i, m in zip(existing_ids, existing_metas) if _keep(i, m or {})]
    kept_texts = [t for i, t, m in zip(existing_ids, existing_texts, existing_metas) if _keep(i, m or {})]
    kept_metas = [m for i, m in zip(existing_ids, existing_metas) if _keep(i, m or {})]

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

    from .ingest import index_build_metadata

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
            "collection": data.get("collection")
            or cfg.get("collection", {}).get("name", "folio"),
            **index_build_metadata(cfg),
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
        max_tokens=int(chunk_cfg["max_tokens"]) if chunk_cfg.get("max_tokens") else None,
        overlap_tokens=int(chunk_cfg["overlap_tokens"]) if chunk_cfg.get("overlap_tokens") else None,
    )
    if not chunks:
        chunks = chunk_markdown(
            md,
            source_id=sid,
            rel_path="document.md",
            max_chars=int(chunk_cfg["max_chars"]),
            overlap_chars=int(chunk_cfg["overlap_chars"]),
            min_chars=min(40, int(chunk_cfg["min_chars"])),
            max_tokens=int(chunk_cfg["max_tokens"]) if chunk_cfg.get("max_tokens") else None,
            overlap_tokens=int(chunk_cfg["overlap_tokens"]) if chunk_cfg.get("overlap_tokens") else None,
        )
    if not chunks:
        raise RuntimeError(f"No chunks produced from {path}")

    collection = get_collection(cfg, create=True)
    _delete_source_vectors(collection, sid)
    n = upsert_chunks(
        collection,
        chunks,
        batch_size=int(cfg["embedding"]["batch_size"]),
    )
    _merge_bm25(cfg, chunks)
    _update_manifest(cfg, source_id=sid, rel_path="document.md", origin=str(path))
    print(f"[add] source={sid} file={path.name} chunks={n} → {dest_md}")
    return {"source_id": sid, "chunks": n, "path": str(dest_md)}
