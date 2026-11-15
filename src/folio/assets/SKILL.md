---
name: folio
description: "Use for any AI engineering / MLOps / RAG / agent / GPU / AI infra / LangChain / LangGraph / LangSmith decision. Queries local hybrid RAG (~/.folio). For LangChain ecosystem coding, retrieve docs-nav URLs then open those official pages."
---

# /folio

Local hybrid RAG over curated AI-engineer / MLOps / infra curricula **plus** local
handbooks and a LangChain docs navigation map. Index: `~/.folio/` (or `$FOLIO_HOME`).

## Usage

```
/folio query "<question>" --json
/folio query "<question>" --source langchain-docs-nav --json
/folio add /path/to/file.pdf|.md|.json
/folio status
/folio install --project --platform cursor
```

## What folio is for

**When to query:** RAG, embeddings, vector DBs, agents, LangChain / LangGraph / LangSmith /
Deep Agents, training/GPU, model optimization, feature stores, GitOps, SRE/SLO for ML,
AI infra, or related senior AI trade-offs.

**When not to:** ordinary product code exploration — use graphify / Read / Grep for that.

## LangChain / LangGraph / LangSmith (mandatory)

If the user asks about these frameworks **or** you are about to write/modify code that
uses them (chains, graphs, tools, memory, LangSmith, Deep Agents, etc.):

1. Run:
   ```bash
   folio query "<task keywords> LangChain LangGraph docs" --source langchain-docs-nav -k 12 --json
   ```
2. From hits, collect official `https://docs.langchain.com/...` URLs (and nearby siblings
   in the same section).
3. **Open/WebFetch those doc pages** before implementing — they are the current API source
   of truth. Prefer them over training memory.
4. Also run a normal folio query (without `--source`) for architecture trade-offs from
   curricula/handbooks when the task needs design advice, not only API lookup.
5. Tell the user which doc URLs you checked.

If `langchain-docs-nav` returns nothing useful, broaden the query once; if still empty,
say the nav index may be stale and ask to re-run
`folio add /path/to/sidebars.json --source-id langchain-docs-nav`.

## Token rule (strict)

Do **not** WebFetch/clone curated **curriculum GitHub repos** into chat — they are indexed;
use `folio query` only.

**Exception:** official `https://docs.langchain.com/...` URLs returned from
`langchain-docs-nav` (or clearly listed in those hits) **may** be opened/WebFetched so
implementation stays current.

If the shared index is missing: instruct `folio bootstrap` (or `folio build`) once.

## What You Must Do When Invoked

### Fast path — index exists

```bash
folio query "<question>" --json
```

Ground architecture answers in returned citations (`source_id`, `rel_path`, `heading_path`).
For LangChain coding tasks, follow the LangChain section above (nav → open docs).

### No index yet

```bash
folio bootstrap   # or: folio build
folio query "<question>" --json
```

## Rules for answers

1. Prefer patterns that appear in retrieved chunks; cite `source_id` + file + heading.
2. When sources conflict, state the trade-off and pick for *this* constraint.
3. Never skip retrieval for “quick” AI architecture or LangChain API guesses when the index is present.
4. For LangChain ecosystem work: nav query → check listed docs URLs → then code.
