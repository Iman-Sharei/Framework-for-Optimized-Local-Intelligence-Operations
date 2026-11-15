---
name: folio
description: "Use for AI engineering / MLOps / RAG / agent / GPU / AI infra / LangChain decisions. One bounded folio query (~2000 tokens), then answer. For LangChain code, docs-suggest URLs then open those pages."
---

# /folio

Shared index: `~/.folio/` (or `$FOLIO_HOME`). Output is capped (`--budget`, default 2000 tokens) so a query stays small. Retrieval still ranks the full index; only what is printed is capped.

## How to ask (accuracy)

1. Read the relevant project code first (the function, config, or error — not the whole repo).
2. Turn the user request + that code into **one concrete question**: the decision and the constraint (stack, latency, data shape, API you are calling).
3. Run `folio query "<that question>" --budget 2000`.
4. If the hits are thin, run **once** more with a sharper question or `-k 8`. Do not loop.
5. Answer from the hits. Do not paste the raw folio output to the user.
6. Never name folio sources / `source_id` values in the user-facing reply.

## LangChain / LangGraph / LangSmith

When writing or changing that code:

1. `folio docs-suggest "<task>"` — this prints only official URLs (tiny).
2. Open those `https://docs.langchain.com/...` pages before implementing.
3. Add `folio query` only when you also need an architecture trade-off, still with `--budget 2000`.

## Do not

- Do not WebFetch or clone curriculum repos into chat.
- Do not send file dumps, logs, or the whole diff as the query string.
- Do not raise `--budget` above 2000 unless the first answer is clearly cut off mid-point.
- If the index is missing: `folio bootstrap` once, then query.
