from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .store import get_collection, load_bm25


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


def _dense_search(cfg: dict[str, Any], query: str, k: int) -> list[tuple[str, str, dict, float]]:
    # Query path must not create/write collections (sandbox / readonly chroma).
    collection = get_collection(cfg, create=False)
    result = collection.query(
        query_texts=[query],
        n_results=min(k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
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


def _bm25_search(cfg: dict[str, Any], query: str, k: int) -> list[tuple[str, str, dict, float]]:
    payload = load_bm25(cfg["paths"]["bm25_path"])
    bm25 = payload["bm25"]
    ids: list[str] = payload["ids"]
    texts: list[str] = payload["texts"]
    metas: list[dict] = payload["metadatas"]
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(ids[i], texts[i], metas[i], float(scores[i])) for i in ranked if scores[i] > 0]


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
    final_k = final_k or int(rcfg["final_k"])
    rrf_k = int(rcfg["rrf_k"])

    dense_hits = _dense_search(cfg, query, dense_k)
    bm25_hits = _bm25_search(cfg, query, bm25_k)

    dense_ids = [h[0] for h in dense_hits]
    bm25_ids = [h[0] for h in bm25_hits]
    fused = rrf_fuse([dense_ids, bm25_ids], k=rrf_k)

    # CRAG-style widen when best fusion score is weak
    best = max(fused.values()) if fused else 0.0
    if best < float(rcfg["crag_min_best_score"]):
        widen = int(rcfg["crag_widen_factor"])
        dense_hits = _dense_search(cfg, query, dense_k * widen)
        bm25_hits = _bm25_search(cfg, query, bm25_k * widen)
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
        if source_filter and sid != source_filter:
            continue
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

    if use_rerank and hits:
        hits = _cross_encoder_rerank(cfg, query, hits, final_k * 3)[:final_k]
    else:
        hits = hits[:final_k]

    floor = float(rcfg.get("confidence_floor", 0.0))
    return [h for h in hits if h.score >= floor]


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


def format_context(hits: list[Hit], *, max_chars: int = 12000) -> str:
    parts: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        header = (
            f"[{i}] source={h.source_id} file={h.rel_path}\n"
            f"heading={h.heading_path} score={h.score:.4f}\n"
        )
        body = h.text
        block = f"{header}{body}\n"
        if used + len(block) > max_chars:
            # Always emit at least one (possibly truncated) citation.
            if not parts:
                room = max(max_chars - used - len(header) - 1, 0)
                parts.append(f"{header}{body[:room]}\n")
            break
        parts.append(block)
        used += len(block)
    return "\n---\n".join(parts)
