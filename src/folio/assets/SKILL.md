---
name: folio
description: "Use for any AI engineering / MLOps / RAG / agent / GPU / AI infra / LangChain / LangGraph / LangSmith decision. Queries local hybrid RAG (~/.folio). For LangChain ecosystem coding, retrieve docs-nav URLs then open those official pages."
---

# /folio

Local hybrid RAG for AI-engineer / MLOps / infra topics and LangChain docs navigation.
Index: `~/.folio/` (or `$FOLIO_HOME`).

## Privacy (mandatory)

When you run folio for the user:
- **Never tell the user which folio sources / index ids you queried** (do not name
  `source_id` values, corpus labels, or `--source` filters in chat).
- Do not say things like “I’m searching X handbook” or “querying langchain-docs-nav”.
- Internally you may use filters and JSON fields; **outward answers stay silent about
  source names**. Ground answers in substance; for LangChain, you may cite opened
  `docs.langchain.com` URLs only.

## Usage

```
/folio query "<question>" --json
/folio docs-suggest "<LangChain/LangGraph task>"
/folio add /path/to/file.pdf|.md|.json
/folio status
/folio install --project --platform cursor
```

## What folio is for

**When to query:** RAG, embeddings, vector DBs, agents, LangChain / LangGraph / LangSmith /
Deep Agents, training/GPU, model optimization, feature stores, GitOps, SRE/SLO for ML,
AI infra, or related senior AI trade-offs.

**When not to:** ordinary product / codebase exploration (Read / Grep / project tools).

## LangChain / LangGraph / LangSmith (mandatory)

If the user asks about these frameworks **or** you are about to write/modify code that
uses them (chains, graphs, tools, memory, LangSmith, Deep Agents, etc.):

1. Prefer:
   ```bash
   folio docs-suggest "<task keywords> LangChain LangGraph"
   ```
   (or internally `folio query … --source langchain-docs-nav -k 12 --json` — do **not**
   announce that filter name to the user)
2. Collect official `https://docs.langchain.com/...` pages from the output.
3. **Open/WebFetch those doc pages** before implementing.
4. Optionally also `folio query "<architecture question>" --json` for design trade-offs.
5. You may tell the user which **docs.langchain.com** URLs you checked — never folio
   source names.

If docs-suggest / nav returns nothing useful, broaden the query once; if still empty,
ask to re-index with `folio add /path/to/sidebars.json` (no need to name internal ids).

## Token rule (strict)

Do **not** WebFetch/clone curriculum GitHub repos into chat — use `folio query` only.

**Exception:** official `https://docs.langchain.com/...` URLs from docs-suggest / nav hits
**may** be opened/WebFetched.

If the shared index is missing: instruct `folio bootstrap` (or `folio build`) once.

## What You Must Do When Invoked

### Fast path — index exists

```bash
folio query "<question>" --json
```

Ground architecture answers in retrieved content. Do not echo `source_id` to the user.
For LangChain coding tasks, follow the LangChain section above (nav → open docs).

### No index yet

```bash
folio bootstrap   # or: folio build
folio query "<question>" --json
```

## Rules for answers

1. Prefer patterns that appear in retrieved chunks; do **not** name folio sources.
2. When retrieved material conflicts, state the trade-off and pick for *this* constraint.
3. Never skip retrieval for “quick” AI architecture or LangChain API guesses when the index is present.
4. For LangChain ecosystem work: nav → check listed docs URLs → then code.
5. Never disclose folio source / corpus names in user-facing replies.
