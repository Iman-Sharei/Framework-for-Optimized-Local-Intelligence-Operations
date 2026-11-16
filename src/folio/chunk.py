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
    # overlap >= window never advances `start` and the loop does not return.
    max_chars = max(int(max_chars), 1)
    overlap = max(int(overlap), 0)
    if overlap >= max_chars:
        overlap = max_chars - 1
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


_FENCE_RE = re.compile(r"^(?P<mark>`{3,}|~{3,})(?P<info>.*)$", re.MULTILINE)


def _fence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of fenced code blocks. Headings inside them are not structure."""
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    fence_char = ""
    for match in _FENCE_RE.finditer(text):
        mark = match.group("mark")
        if open_at is None:
            open_at = match.start()
            fence_char = mark[0]
            continue
        if mark[0] == fence_char:
            spans.append((open_at, match.end()))
            open_at = None
            fence_char = ""
    if open_at is not None:
        spans.append((open_at, len(text)))
    return spans


def _inside_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in spans)


def _attach_short_pieces(
    pieces: list[tuple[str, str]],
    *,
    min_chars: int,
) -> list[tuple[str, str]]:
    """Keep text under min_chars by merging it into a neighbor instead of dropping it."""
    if not pieces:
        return []
    merged: list[tuple[str, str]] = []
    pending: list[tuple[str, str]] = []

    def _glue(base: str, extra: str) -> str:
        extra = extra.strip()
        if not extra:
            return base
        if not base:
            return extra
        return f"{base.rstrip()}\n\n{extra}"

    for heading, piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) < min_chars:
            pending.append((heading, piece))
            continue
        heading_out = heading
        body = piece
        if pending:
            body = _glue("\n\n".join(p for _, p in pending), piece)
            if not heading_out:
                heading_out = next((h for h, _ in pending if h), heading)
            pending = []
        merged.append((heading_out, body))
    if pending:
        extra = "\n\n".join(p for _, p in pending)
        if merged:
            heading, body = merged[-1]
            merged[-1] = (heading, _glue(body, extra))
        else:
            heading = next((h for h, _ in pending if h), "")
            merged.append((heading, extra.strip()))
    return merged


def chunk_markdown(
    text: str,
    *,
    source_id: str,
    rel_path: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
    min_chars: int = 80,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """Structure-aware chunking: split on markdown headings, then size-guard."""
    content_hash = _file_hash(text)
    # A 200-token MiniLM window is about 3 characters per token on code-like text.
    if max_tokens and max_tokens > 0:
        token_chars = max(40, int(max_tokens) * 3)
        max_chars = min(max_chars, token_chars)
        if overlap_tokens and overlap_tokens > 0:
            overlap_chars = min(overlap_chars, max(0, int(overlap_tokens) * 3))
    sections: list[tuple[str, str]] = []
    fences = _fence_spans(text)
    matches = [m for m in HEADING_RE.finditer(text) if not _inside_span(m.start(), fences)]

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

    sized: list[tuple[str, str]] = []
    for heading_path, body in sections:
        for piece in _split_oversized(body, max_chars, overlap_chars):
            piece = piece.strip()
            if piece:
                sized.append((heading_path, piece))
    raw_chunks = _attach_short_pieces(sized, min_chars=min_chars)

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
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
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
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
