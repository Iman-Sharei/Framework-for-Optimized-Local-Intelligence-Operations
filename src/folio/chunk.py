from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_id: str
    rel_path: str
    heading_path: str
    chunk_index: int
    content_hash: str
    char_count: int

    def to_metadata(self) -> dict:
        meta = asdict(self)
        # Chroma metadata values must be str/int/float/bool
        meta.pop("text")
        return meta


def _file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _split_oversized(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    buf = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            parts.append(buf)
        if len(para) <= max_chars:
            buf = para
        else:
            start = 0
            while start < len(para):
                end = min(start + max_chars, len(para))
                parts.append(para[start:end])
                if end >= len(para):
                    break
                start = max(0, end - overlap)
            buf = ""
    if buf:
        parts.append(buf)

    if overlap <= 0 or len(parts) <= 1:
        return parts

    # Soft overlap between adjacent oversized splits already handled for long paras;
    # stitch short tails if needed by overlapping previous suffix.
    overlapped: list[str] = [parts[0]]
    for i in range(1, len(parts)):
        prev = parts[i - 1]
        prefix = prev[-overlap:] if len(prev) > overlap else prev
        nxt = parts[i]
        if not nxt.startswith(prefix[:40]):
            overlapped.append(f"{prefix}\n{nxt}".strip())
        else:
            overlapped.append(nxt)
    return overlapped


def chunk_markdown(
    text: str,
    *,
    source_id: str,
    rel_path: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
    min_chars: int = 80,
) -> list[Chunk]:
    """Structure-aware chunking: split on markdown headings, then size-guard."""
    content_hash = _file_hash(text)
    sections: list[tuple[str, str]] = []
    matches = list(HEADING_RE.finditer(text))

    if not matches:
        sections.append(("", text.strip()))
    else:
        # Preface before first heading
        preface = text[: matches[0].start()].strip()
        if preface:
            sections.append(("", preface))

        stack: list[tuple[int, str]] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            heading_path = " > ".join(t for _, t in stack)
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((heading_path, body))

    raw_chunks: list[tuple[str, str]] = []
    for heading_path, body in sections:
        for piece in _split_oversized(body, max_chars, overlap_chars):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            raw_chunks.append((heading_path, piece))

    chunks: list[Chunk] = []
    for idx, (heading_path, piece) in enumerate(raw_chunks):
        digest = hashlib.sha256(
            f"{source_id}:{rel_path}:{idx}:{piece}".encode("utf-8")
        ).hexdigest()[:24]
        chunks.append(
            Chunk(
                chunk_id=f"{source_id}:{digest}",
                text=piece,
                source_id=source_id,
                rel_path=rel_path,
                heading_path=heading_path or "(preamble)",
                chunk_index=idx,
                content_hash=content_hash,
                char_count=len(piece),
            )
        )
    return chunks


def chunk_file(
    path: Path,
    *,
    source_id: str,
    sources_root: Path,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        rel = str(path.relative_to(sources_root))
    except ValueError:
        rel = path.name
    return chunk_markdown(
        text,
        source_id=source_id,
        rel_path=rel,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        min_chars=min_chars,
    )
