"""Labeled retrieval checks against the hybrid index the README describes.

These tests build a tiny corpus with the real ONNX embedder and BM25 index.
They print a metric block (RETRIEVAL_METRICS / EMBED_WINDOW) that the review
cites. Contract failures are expected to fail until the scorer or chunker changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from folio.add_doc import add_local_document
from folio.chunk import Chunk, chunk_markdown
from folio.ingest import ingest
from folio.retrieve import _bm25_search, _dense_search, retrieve
from folio.store import (
    build_bm25,
    get_collection,
    get_embedding_fn,
    reset_collection,
    save_bm25,
    upsert_chunks,
)

pytestmark = pytest.mark.integration


def _published_retrieval(cfg: dict) -> dict:
    cfg["retrieval"] = {
        "dense_k": 20,
        "bm25_k": 20,
        "final_k": 8,
        "rrf_k": 60,
        "crag_min_best_score": 0.012,
        "crag_widen_factor": 2,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "confidence_floor": 0.008,
    }
    cfg["chunking"] = {"max_chars": 800, "overlap_chars": 80, "min_chars": 40}
    return cfg


def _doc(title: str, body: str) -> str:
    return f"# {title}\n\n{body.strip()}\n"


GOLD = {
    "fusion.md": _doc(
        "Fusion",
        """
        Reciprocal rank fusion merges a keyword ranking and an embedding ranking
        by adding one over k plus rank. It is used when raw scores from the two
        retrievers live on different scales and must not be compared directly.
        Documents that place well in both lists rise to the top of the fused order.
        """,
    ),
    "agent.md": _doc(
        "Agents",
        """
        Prefer a fixed workflow when every step is known ahead of time. Prefer an
        agent only when the next tool must be chosen from live observations.
        Write tools need idempotency keys so a retry does not apply the same
        side effect twice.
        """,
    ),
    "shadow.md": _doc(
        "Shadow",
        """
        Shadow deployment duplicates live requests to a challenger model while
        users still receive answers from the production model. Promotion happens
        only after that silent comparison looks safe.
        """,
    ),
    "parallel.md": _doc(
        "Parallelism",
        """
        Tensor parallelism shards each weight matrix across accelerators. Pipeline
        parallelism places successive layers on different devices and keeps the
        devices busy with microbatches.
        """,
    ),
    "eval.md": _doc(
        "Evaluation",
        """
        Offline evaluation scores a frozen labeled collection. Online evaluation
        interleaves a new policy with live traffic and reads user outcomes such
        as acceptance and retry rate.
        """,
    ),
    "registry.md": _doc(
        "Registry",
        """
        The model registry stores each artifact URI together with a stage such as
        staging or production and the metrics recorded at promotion time.
        """,
    ),
    "cache.md": _doc(
        "Cache",
        """
        An embedding cache reuses a stored vector when the chunk text hash is
        unchanged and recomputes the vector only after that hash changes.
        """,
    ),
    "budget.md": _doc(
        "Budget",
        """
        A print budget limits how much retrieved text is shown to the assistant.
        The ranker still scores every indexed passage before that cap is applied.
        """,
    ),
    "rare.md": _doc(
        "Rare token",
        """
        The zygomorphic bloom opens at dusk and closes at dawn. Botanists record
        that shape when they catalog irregular flowers in the greenhouse notes.
        """,
    ),
}

DISTRACTORS = {
    "soccer.md": _doc(
        "Soccer",
        "World cup ranking of national teams uses match points. A draw gives one point. This table is not a search ranking and it has no embeddings.",
    ),
    "tomatoes.md": _doc(
        "Tomatoes",
        "Tomato plants in a production garden need even watering. Staging the harvest means picking fruit that has turned red on the vine.",
    ),
    "guitar.md": _doc(
        "Guitar",
        "String gauges change the tone of a guitar. A heavier gauge needs higher tension and can buzz on a low action.",
    ),
    "trains.md": _doc(
        "Trains",
        "A rail pass in Japan covers intercity trains. Seat reservations are optional and the fastest services still need a ticket.",
    ),
    "knitting.md": _doc(
        "Knitting",
        "A gauge swatch measures stitches per inch before a sweater is started. Needle size is adjusted until the swatch matches the pattern.",
    ),
    "saturn.md": _doc(
        "Saturn",
        "Saturn's rings are ice and dust orbiting in a thin plane. The gaps are named and they are not a model registry.",
    ),
    "sourdough.md": _doc(
        "Sourdough",
        "Feed the starter with equal weights of flour and water on a fixed schedule. A hungry starter smells of alcohol and will not rise.",
    ),
    "hurricane.md": _doc(
        "Hurricane",
        "Hurricane categories depend on sustained wind speed. Storm surge, not the category number alone, drives most coastal damage.",
    ),
}

# Looser wording on purpose: these questions avoid the note's distinctive nouns
# so a pure keyword match is not enough.
HARD = [
    (
        "fusion.md",
        "how do I merge two ordered lists from different search methods when I cannot compare their numeric scores?",
    ),
    (
        "agent.md",
        "is it better to hard-code every action, or to decide the following action from what just happened?",
    ),
    (
        "shadow.md",
        "can I send live traffic to a new version without letting customers notice the experiment?",
    ),
    (
        "parallel.md",
        "should accelerator splitting cut each matrix, or cut the stack of layers?",
    ),
    (
        "eval.md",
        "when is a canned dataset the wrong way to judge quality compared with what real people do?",
    ),
    (
        "registry.md",
        "where should I keep versioned model files and the stage they were promoted to?",
    ),
    (
        "cache.md",
        "how do I avoid recomputing a vector for text that is still the same?",
    ),
    (
        "budget.md",
        "how do I keep the assistant prompt small without changing which passages win?",
    ),
]

PARAPHRASE = [
    (
        "fusion.md",
        "how should I blend a keyword hit list with an embedding hit list when their raw scores are not comparable?",
    ),
    (
        "agent.md",
        "should the controller follow a script written in advance or choose the next tool from the latest observation?",
    ),
    (
        "shadow.md",
        "how do I score a challenger model on live requests while users still see the current model?",
    ),
    (
        "parallel.md",
        "how do I split a neural network by matrices versus by layers across accelerators?",
    ),
    (
        "eval.md",
        "when should quality be judged from live user outcomes instead of a frozen labeled set?",
    ),
    (
        "registry.md",
        "where do staged model artifacts and their promotion metrics live?",
    ),
    (
        "cache.md",
        "how can I reuse an embedding vector until the underlying text hash changes?",
    ),
    (
        "budget.md",
        "how do I limit what the assistant reads while still ordering every indexed passage?",
    ),
]


def _index_markdown(cfg: dict, docs: dict[str, str], source_id: str = "corpus") -> None:
    chunk_cfg = cfg["chunking"]
    chunks: list[Chunk] = []
    for rel, text in docs.items():
        chunks.extend(
            chunk_markdown(
                text,
                source_id=source_id,
                rel_path=rel,
                max_chars=int(chunk_cfg["max_chars"]),
                overlap_chars=int(chunk_cfg["overlap_chars"]),
                min_chars=int(chunk_cfg["min_chars"]),
            )
        )
    assert chunks, "corpus produced no chunks"
    collection = reset_collection(cfg)
    upsert_chunks(collection, chunks, batch_size=16)
    bm25, ids, stored = build_bm25(chunks)
    save_bm25(cfg["paths"]["bm25_path"], bm25, ids, stored)


def _rank(rows, rel: str) -> int | None:
    for i, row in enumerate(rows, start=1):
        if isinstance(row, tuple):
            meta = row[2] or {}
            if meta.get("rel_path") == rel:
                return i
        elif getattr(row, "rel_path", None) == rel:
            return i
    return None


def _reciprocal(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


@pytest.fixture
def indexed(mini_cfg: dict):
    cfg = _published_retrieval(mini_cfg)
    _index_markdown(cfg, {**GOLD, **DISTRACTORS})
    return cfg


def _score_queries(cfg: dict, pairs: list[tuple[str, str]]) -> list[dict]:
    rows = []
    for rel, query in pairs:
        hybrid = retrieve(cfg, query, final_k=8)
        dense = _dense_search(cfg, query, k=16)
        lexical = _bm25_search(cfg, query, k=16)
        rows.append(
            {
                "gold": rel,
                "query": query,
                "hybrid": _rank(hybrid, rel),
                "dense": _rank(dense, rel),
                "bm25": _rank(lexical, rel),
                "top": hybrid[0].rel_path if hybrid else None,
            }
        )
    return rows


def _pack_metrics(rows: list[dict]) -> dict:
    def recall(key: str, k: int) -> float:
        return sum(1 for row in rows if row[key] is not None and row[key] <= k) / len(rows)

    def mrr(key: str) -> float:
        return round(sum(_reciprocal(row[key]) for row in rows) / len(rows), 3)

    return {
        "n": len(rows),
        "hybrid_recall_at_1": recall("hybrid", 1),
        "hybrid_recall_at_3": recall("hybrid", 3),
        "hybrid_recall_at_8": recall("hybrid", 8),
        "hybrid_mrr": mrr("hybrid"),
        "dense_recall_at_1": recall("dense", 1),
        "dense_recall_at_3": recall("dense", 3),
        "dense_mrr": mrr("dense"),
        "bm25_recall_at_1": recall("bm25", 1),
        "bm25_recall_at_3": recall("bm25", 3),
        "bm25_mrr": mrr("bm25"),
        "rows": rows,
    }


def test_hybrid_paraphrase_accuracy(indexed, capsys):
    cfg = indexed
    # Control: a copied sentence must be findable. If this misses, the harness is wrong.
    control = retrieve(
        cfg,
        "Reciprocal rank fusion merges a keyword ranking and an embedding ranking",
        final_k=3,
    )
    assert _rank(control, "fusion.md") == 1

    easy = _pack_metrics(_score_queries(cfg, PARAPHRASE))
    hard = _pack_metrics(_score_queries(cfg, HARD))
    bare = _rank(_bm25_search(cfg, "zygomorphic", k=16), "rare.md")
    punct = _rank(_bm25_search(cfg, "zygomorphic?", k=16), "rare.md")
    hybrid_punct = _rank(retrieve(cfg, "zygomorphic?", final_k=8), "rare.md")
    metrics = {
        "corpus": len(GOLD) + len(DISTRACTORS),
        "random_recall_at_3": round(3 / (len(GOLD) + len(DISTRACTORS)), 3),
        "easy_shared_vocabulary": {k: v for k, v in easy.items() if k != "rows"},
        "hard_paraphrase": {k: v for k, v in hard.items() if k != "rows"},
        "easy_rows": easy["rows"],
        "hard_rows": hard["rows"],
        "bm25_exact_rare_token_rank": bare,
        "bm25_punctuated_rare_token_rank": punct,
        "hybrid_punctuated_rare_token_rank": hybrid_punct,
    }
    print("RETRIEVAL_METRICS " + json.dumps(metrics))

    assert bare == 1
    assert hybrid_punct == 1  # dense still finds the term; BM25 does not (see xfail)
    assert easy["hybrid_recall_at_1"] == 1.0
    # Measured on this fixture with the bundled MiniLM. Dense beats hybrid on the
    # loose set because fusion demotes the eval note from dense rank 1 to rank 5.
    # 5/8. Tokenizing punctuation moves one loose question from rank 1 to rank 2.
    assert hard["hybrid_recall_at_1"] == 0.625
    assert hard["hybrid_recall_at_3"] == 0.875
    assert hard["hybrid_recall_at_8"] == 1.0
    assert hard["dense_recall_at_3"] == 1.0
    assert hard["dense_mrr"] > hard["hybrid_mrr"]
    eval_row = next(row for row in hard["rows"] if row["gold"] == "eval.md")
    assert eval_row["dense"] == 1
    assert eval_row["hybrid"] > eval_row["dense"]


def test_bm25_matches_unique_term_with_trailing_punctuation(indexed):
    assert _rank(_bm25_search(indexed, "zygomorphic?", k=16), "rare.md") == 1


def _tokenizer(mini_cfg):
    get_embedding_fn(Path(mini_cfg["paths"]["model_cache_dir"]))
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    tokenizer = ONNXMiniLM_L6_V2().tokenizer
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def _restore_tokenizer(tokenizer) -> None:
    """The ONNX tokenizer is a process-wide singleton. Put the model window back."""
    tokenizer.enable_truncation(max_length=256)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)


def test_default_chunk_and_print_budget_token_counts(mini_cfg, capsys):
    """The README's 2000-token cap is implemented as 4 characters per token."""
    tokenizer = _tokenizer(mini_cfg)
    samples = {
        "technical_prose_1200": (
            "Hybrid retrieval mixes lexical search and dense vectors for serving decisions. " * 40
        )[:1200],
        "short_words_1200": ("the cat sat on the mat and then ran home. " * 80)[:1200],
        "code_1200": ("result = cache.get(user_id, default=None)\n" * 80)[:1200],
        "technical_prose_8000": (
            "Hybrid retrieval mixes lexical search and dense vectors for serving decisions. " * 200
        )[:8000],
        "code_8000": ("result = cache.get(user_id, default=None)\n" * 400)[:8000],
        "word_repeat_8000": ("word " * 2000)[:8000],
    }
    try:
        counts = {name: len(tokenizer.encode(text).ids) for name, text in samples.items()}
    finally:
        _restore_tokenizer(tokenizer)
    print("TOKEN_COUNTS " + json.dumps(counts))
    assert counts["technical_prose_1200"] <= 256
    assert counts["short_words_1200"] > 256
    assert counts["code_1200"] > 256
    # A full default print of code-like text is past the advertised 2000 tokens.
    assert counts["code_8000"] > 2000
    assert counts["technical_prose_8000"] < 2000


def test_chunk_tail_changes_the_dense_vector(mini_cfg, capsys):
    tokenizer = _tokenizer(mini_cfg)
    full = " ".join(f"garden{i}" for i in range(400))
    full += " purplegiraffe singleflight lock stops a cache stampede"
    encoded = tokenizer.encode(full)
    assert len(encoded.ids) > 256
    prefix = full[: encoded.offsets[255][1]]
    hidden = full[encoded.offsets[255][1] :]
    assert "purplegiraffe" in hidden
    _restore_tokenizer(tokenizer)
    embed = get_embedding_fn(Path(mini_cfg["paths"]["model_cache_dir"]))
    va, vb = np.asarray(embed([prefix, full]), dtype=np.float64)
    cosine = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
    print("EMBED_WINDOW " + json.dumps({"cosine_prefix_vs_full": cosine, "hidden_chars": len(hidden)}))
    assert cosine < 0.999


def test_readd_replaces_previous_chunks(mini_cfg):
    cfg = _published_retrieval(mini_cfg)
    cfg["retrieval"]["confidence_floor"] = 0.0
    path = Path(cfg["paths"]["out_dir"]) / "incoming.md"
    path.write_text(
        "# Note\n\n" + "The secret animal is a purplegiraffe in the first draft. " * 4,
        encoding="utf-8",
    )
    add_local_document(cfg, path, source_id="note")
    path.write_text(
        "# Note\n\n" + "The secret animal is a bluepenguin in the revised draft. " * 4,
        encoding="utf-8",
    )
    add_local_document(cfg, path, source_id="note")
    collection = get_collection(cfg, create=False)
    stored = "\n".join(collection.get(include=["documents"])["documents"] or [])
    from folio.store import load_bm25

    lexical = "\n".join(load_bm25(cfg["paths"]["bm25_path"])["texts"])
    assert "bluepenguin" in stored and "bluepenguin" in lexical
    assert "purplegiraffe" not in stored
    assert "purplegiraffe" not in lexical


def test_ingest_rebuild_false_keeps_existing_chunks(mini_cfg):
    cfg = _published_retrieval(mini_cfg)
    sentinel = Chunk(
        chunk_id="sentinel-keep",
        text="sentinel-token-xyz appears only in the pre-existing collection. " * 3,
        source_id="preexisting",
        rel_path="sentinel.md",
        heading_path="S",
        chunk_index=0,
        content_hash="abc",
        char_count=80,
    )
    collection = reset_collection(cfg)
    upsert_chunks(collection, [sentinel], batch_size=8)
    extra = Path(cfg["paths"]["sources_dir"]) / "localnotes"
    extra.mkdir(parents=True, exist_ok=True)
    (extra / "topic.md").write_text(
        "# Topic\n\n" + "A freshly ingested paragraph about feature stores and point in time joins. " * 3,
        encoding="utf-8",
    )
    ingest(cfg, rebuild=False)
    ids = get_collection(cfg, create=False).get()["ids"]
    assert "sentinel-keep" in ids


def test_empty_collection_query_does_not_crash(mini_cfg):
    cfg = _published_retrieval(mini_cfg)
    reset_collection(cfg)
    _dense_search(cfg, "hello", 5)
