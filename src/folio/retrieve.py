from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .store import get_collection, load_bm25, tokenize_text


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    source_id: str
    rel_path: str
    heading_path: str
    dense_rank: int | None = None
    bm25_rank: int | None = None


def rrf_fuse(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion — merges rankings without score normalisation."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _dense_search(
    cfg: dict[str, Any],
    query: str,
    k: int,
    source_filter: str | None = None,
) -> list[tuple[str, str, dict, float]]:
    # Query path must not create/write collections (sandbox / readonly chroma).
    collection = get_collection(cfg, create=False)
    where = {"source_id": source_filter} if source_filter else None
    if where:
        matched = collection.get(where=where, include=[])
        match_count = len(matched.get("ids") or [])
        if match_count <= 0:
            return []
        n_results = min(k, match_count)
    else:
        n_results = min(k, max(collection.count(), 1))
    query_kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where
    result = collection.query(**query_kwargs)
    ids = result["ids"][0] if result["ids"] else []
    docs = result["documents"][0] if result["documents"] else []
    metas = result["metadatas"][0] if result["metadatas"] else []
    dists = result["distances"][0] if result["distances"] else []
    out: list[tuple[str, str, dict, float]] = []
    for i, doc_id in enumerate(ids):
        # cosine distance → similarity-ish for ranking display
        sim = 1.0 - float(dists[i]) if i < len(dists) else 0.0
        out.append((doc_id, docs[i], metas[i] or {}, sim))
    return out


def _bm25_search(
    cfg: dict[str, Any],
    query: str,
    k: int,
    source_filter: str | None = None,
) -> list[tuple[str, str, dict, float]]:
    payload = load_bm25(cfg["paths"]["bm25_path"])
    bm25 = payload["bm25"]
    ids: list[str] = payload["ids"]
    texts: list[str] = payload["texts"]
    metas: list[dict] = payload["metadatas"]
    if not ids:
        return []
    tokens = [tok for tok in tokenize_text(query) if tok != "__folio_empty__"]
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked_all = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[tuple[str, str, dict, float]] = []
    for i in ranked_all:
        if i >= len(ids) or i >= len(texts) or i >= len(metas):
            continue
        meta = metas[i] or {}
        if source_filter and str(meta.get("source_id", "")) != source_filter:
            continue
        # A non-positive score can still be a real hit when idf is tiny.
        # Zero means the query terms do not occur.
        if float(scores[i]) == 0.0:
            continue
        out.append((ids[i], texts[i], meta, float(scores[i])))
        if len(out) >= k:
            break
    return out


def retrieve(
    cfg: dict[str, Any],
    query: str,
    *,
    final_k: int | None = None,
    use_rerank: bool = False,
    source_filter: str | None = None,
) -> list[Hit]:
    """
    Hybrid path from senior revision (Bench):
      query → BM25 + dense → RRF → CRAG widen → optional cross-encoder → top-k
    """
    rcfg = cfg["retrieval"]
    dense_k = int(rcfg["dense_k"])
    bm25_k = int(rcfg["bm25_k"])
    if final_k is not None and int(final_k) < 1:
        return []
    final_k = int(rcfg["final_k"]) if final_k is None else int(final_k)
    rrf_k = int(rcfg["rrf_k"])

    dense_hits = _dense_search(cfg, query, dense_k, source_filter)
    bm25_hits = _bm25_search(cfg, query, bm25_k, source_filter)

    dense_ids = [h[0] for h in dense_hits]
    bm25_ids = [h[0] for h in bm25_hits]
    fused = rrf_fuse([dense_ids, bm25_ids], k=rrf_k)

    # Widen when the nearest dense neighbor is a weak cosine match.
    # Rank-1 RRF is always 1/(rrf_k+1), which sits above the old 0.012 cutoff,
    # so the decision uses dense similarity (1 - cosine distance) instead.
    # Configs that still store that old RRF cutoff are lifted to a cosine bar.
    threshold = float(rcfg["crag_min_best_score"])
    if 0 < threshold < 0.05:
        threshold = 0.20
    best_sim = max((float(h[3]) for h in dense_hits), default=0.0)
    if best_sim < threshold:
        widen = int(rcfg["crag_widen_factor"])
        dense_hits = _dense_search(cfg, query, dense_k * widen, source_filter)
        bm25_hits = _bm25_search(cfg, query, bm25_k * widen, source_filter)
        fused = rrf_fuse([[h[0] for h in dense_hits], [h[0] for h in bm25_hits]], k=rrf_k)

    by_id: dict[str, tuple[str, dict]] = {}
    dense_rank = {h[0]: i + 1 for i, h in enumerate(dense_hits)}
    bm25_rank = {h[0]: i + 1 for i, h in enumerate(bm25_hits)}
    for doc_id, text, meta, _ in dense_hits + bm25_hits:
        by_id[doc_id] = (text, meta)

    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    hits: list[Hit] = []
    for doc_id, score in ordered:
        if doc_id not in by_id:
            continue
        text, meta = by_id[doc_id]
        sid = str(meta.get("source_id", ""))
        hits.append(
            Hit(
                chunk_id=doc_id,
                text=text,
                score=score,
                source_id=sid,
                rel_path=str(meta.get("rel_path", "")),
                heading_path=str(meta.get("heading_path", "")),
                dense_rank=dense_rank.get(doc_id),
                bm25_rank=bm25_rank.get(doc_id),
            )
        )

    # The floor is an RRF cutoff. Cross-encoder logits are often negative and
    # must not be compared to it.
    floor = float(rcfg.get("confidence_floor", 0.0))
    hits = [h for h in hits if h.score >= floor]
    if use_rerank and hits:
        return _cross_encoder_rerank(cfg, query, hits, final_k * 3)[:final_k]
    return hits[:final_k]


def _cross_encoder_rerank(
    cfg: dict[str, Any], query: str, hits: list[Hit], pool: int
) -> list[Hit]:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError(
            "Rerank requires optional deps: pip install 'folio[rerank]'"
        ) from e

    model_name = cfg["retrieval"]["reranker_model"]
    model = CrossEncoder(model_name)
    pool_hits = hits[:pool]
    pairs = [(query, h.text) for h in pool_hits]
    scores = model.predict(pairs)
    rescored = []
    for h, s in zip(pool_hits, scores):
        rescored.append(
            Hit(
                chunk_id=h.chunk_id,
                text=h.text,
                score=float(s),
                source_id=h.source_id,
                rel_path=h.rel_path,
                heading_path=h.heading_path,
                dense_rank=h.dense_rank,
                bm25_rank=h.bm25_rank,
            )
        )
    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored


def chars_for_token_budget(token_budget: int) -> int:
    """Approx chars for a token cap (same 4 chars/token heuristic as compact CLIs)."""
    budget = int(token_budget)
    if budget <= 0:
        return 0
    return budget * 4


def apply_text_budget(text: str, *, token_budget: int) -> str:
    limit = chars_for_token_budget(token_budget)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    note = f"\n… (truncated to ~{token_budget} tokens; raise --budget for more)\n"
    if len(note) >= limit:
        return text[:limit]
    room = limit - len(note)
    return text[:room] + note


def _json_item(h: Hit, *, text: str, rel_path: str, heading_path: str, chunk_id: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "score": round(h.score, 4),
        "heading_path": heading_path,
        "rel_path": rel_path,
        "source_id": h.source_id,
        "text": text,
    }


def _fit_json_item(h: Hit, remaining: int) -> dict[str, Any] | None:
    """Shrink one hit until its JSON object fits in ``remaining`` characters."""
    if remaining < 2:
        return None
    text = h.text
    rel_path = h.rel_path
    heading = h.heading_path
    chunk_id = h.chunk_id
    while True:
        item = _json_item(h, text=text, rel_path=rel_path, heading_path=heading, chunk_id=chunk_id)
        blob = json.dumps(item, ensure_ascii=False)
        if len(blob) <= remaining:
            return item
        overflow = len(blob) - remaining
        if text:
            text = text[: max(0, len(text) - overflow)]
            continue
        if len(rel_path) > 1:
            rel_path = rel_path[: max(1, len(rel_path) - overflow)]
            continue
        if len(heading) > 1:
            heading = heading[: max(1, len(heading) - overflow)]
            continue
        if len(chunk_id) > 1:
            chunk_id = chunk_id[: max(1, len(chunk_id) - overflow)]
            continue
        return None


def hits_to_json_payload(hits: list[Hit], *, token_budget: int) -> list[dict[str, Any]]:
    """Compact hit list that fits the token budget (accuracy stays in retrieval; output is capped)."""
    limit = chars_for_token_budget(token_budget)
    payload: list[dict[str, Any]] = []
    used = 2  # []
    for h in hits:
        comma = 2 if payload else 0  # ", " between objects in the default encoding
        remaining = limit - used - comma
        item = _fit_json_item(h, remaining)
        if item is None:
            break
        payload.append(item)
        used += comma + json_len(item)
    return payload


def json_len(item: dict[str, Any]) -> int:
    return len(json.dumps(item, ensure_ascii=False))


_TRUNCATION_NOTE = "\n… (truncated)\n"


def format_context(hits: list[Hit], *, max_chars: int = 8000) -> str:
    parts: list[str] = []
    used = 0
    sep = "\n---\n"
    for i, h in enumerate(hits, start=1):
        # Omit source_id from human output so assistants do not echo corpus names.
        header = (
            f"[{i}] file={h.rel_path}\n"
            f"heading={h.heading_path} score={h.score:.4f}\n"
        )
        body = h.text
        block = f"{header}{body}\n"
        extra = len(sep) if parts else 0
        room = max_chars - used - extra
        if room <= 0:
            break
        if len(block) > room:
            note = _TRUNCATION_NOTE if room > len(_TRUNCATION_NOTE) else ""
            parts.append(block[: room - len(note)] + note)
            break
        parts.append(block)
        used += extra + len(block)
    return sep.join(parts)

# folio retrieve surface
