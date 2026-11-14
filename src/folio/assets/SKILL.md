---
name: folio
description: "Use for any AI engineering / MLOps / RAG / agent / GPU / AI infra / related AI architecture decision. Queries the local hybrid RAG over curated curricula (~/.folio). Treat those design questions as a folio query first."
---

# /folio

Local hybrid RAG over curated AI-engineer / MLOps / infra markdown curricula.
Persistent vector + BM25 index in `~/.folio/` (or `$FOLIO_HOME`). No API key required.

## Usage

```
/folio                                              # orient + ensure index
/folio bootstrap                                    # download prebuilt index (preferred)
/folio build                                        # sync curated sources + ingest
/folio update                                       # refresh from maintainer Release
/folio update --from-source                         # rebuild from curated git sources
/folio query "<question>"                           # hybrid retrieve (BM25 + dense + RRF + CRAG)
/folio query "<question>" --json                    # machine-readable hits for agents
/folio query "<question>" -k 10                     # more chunks
/folio query "<question>" --rerank                  # optional cross-encoder
/folio status                                       # CLI + index health
/folio install --project --platform cursor          # wire this repo's assistant
```

## What folio is for

Ground AI / MLOps / AI-infra decisions in retrieved citations from curated curricula
(currently senior AI revision guide + AI infra lessons; more domains later).

**When to query:** RAG, embeddings, vector DBs, agents, training/GPU, model optimization,
feature stores, GitOps, SRE/SLO for ML, AI infra, or related senior AI trade-offs.

**When not to:** ordinary product code exploration — use graphify / Read / Grep for that.

## Token rule (strict)

Do **not** WebFetch, clone, or paste the curriculum GitHub repos into chat.
They are already indexed. Only `folio query`. If the index is missing, tell the user
to run `folio bootstrap` or `folio build` — never dump raw curriculum markdown into context.

## What You Must Do When Invoked

### Fast path — index exists

If `~/.folio/manifest.json` exists (or `$FOLIO_HOME/manifest.json`):

```bash
folio query "<question>" --json
```

Ground your answer **only** in returned citations (`source_id`, `rel_path`, `heading_path`, `text`).
If scores are weak / empty, broaden terms and re-query once before inventing architecture.

### No index yet

```bash
folio bootstrap   # or: folio build
folio query "<question>" --json
```

### Install into a project

```bash
folio install --project --platform cursor
folio bootstrap   # once per machine
```

## Rules for answers

1. Prefer patterns that appear in retrieved chunks; cite `source_id` + file + heading.
2. When sources conflict (e.g. hybrid RAG vs RAGless), state the trade-off and pick for *this* constraint.
3. Do not dump entire markdown files when retrieval exists.
4. Never skip retrieval for “quick” AI architecture guesses when the index is present.
