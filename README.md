# folio

Local hybrid RAG over curated **AI engineering / MLOps / AI infra** markdown curricula
(graphify-style CLI). Assistants query a **shared machine index** instead of stuffing
GitHub repos into chat — lower token use, grounded citations.

| Source | Path |
|--------|------|
| [02-senior-ai-engineer-study-path](https://github.com/RavSinghChandan/02-senior-ai-engineer-study-path) | `revision-1/` |
| [ai-infra-senior-engineer-learning](https://github.com/ai-infra-curriculum/ai-infra-senior-engineer-learning) | `lessons/` |
| [agentic-ai-developer-learning](https://github.com/ai-engineering-curriculum/agentic-ai-developer-learning) | `lessons/` |
| [agentic-ai-engineer-learning](https://github.com/ai-engineering-curriculum/agentic-ai-engineer-learning) | `lessons/` |
| [senior-agentic-ai-engineer-learning](https://github.com/ai-engineering-curriculum/senior-agentic-ai-engineer-learning) | `lessons/` |
| [applied-ai-engineer-learning](https://github.com/ai-engineering-curriculum/applied-ai-engineer-learning) | `lessons/` |
| [ai-eval-engineer-learning](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning) | `lessons/` |
| [agentic-systems-architect-learning](https://github.com/ai-engineering-curriculum/agentic-systems-architect-learning) | `lessons/` |
| [mlops-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) | repo MD modules |

## Install (once on this machine)

```bash
# from this checkout
cd ~/Desktop/folio   # or clone the GitHub repo
python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e .

# put on PATH
ln -sfn "$(pwd)/.venv/bin/folio" ~/.local/bin/folio
```

Later (when published):

```bash
uv tool install folio          # or folio-cli if the name is taken on PyPI
# or: uv tool install git+https://github.com/<you>/folio
```

## Get the knowledge index

Index lives in **`~/.folio/`** (override with `FOLIO_HOME`). Not inside product repos.

```bash
folio bootstrap                # preferred: prebuilt pack from Release
# or:
folio build                    # sync curated sources + embed (slower)
folio status
```

Refresh when the maintainer publishes new curricula:

```bash
folio update                   # download new Release pack (falls back to build)
folio update --from-source     # force rebuild from git sources
```

Maintainer release flow:

```bash
folio build
folio pack                     # → dist/folio-index-YYYY.MM.DD.tar.gz
# upload tarball to GitHub Release; set release.index_url in config
```

## Use from any project (like graphify)

```bash
cd /path/to/your-project
folio install --project --platform cursor
folio query "hybrid RAG vs RAGless" --json
```

**Token rule:** assistants must **not** WebFetch/clone curriculum repos into chat.
Only `folio query`.

## Architecture

```
curated MD → chunk → ONNX MiniLM
              ↓
   ChromaDB + BM25  →  ~/.folio/
              ↓
query → dense + BM25 → RRF → CRAG → citations (~1–2k tokens, not the whole corpus)
```

## Commands

```bash
folio bootstrap [--url URL] [--archive FILE] [--force]
folio build
folio update [--from-source]
folio pack [--dest PATH]
folio sync
folio query "<q>" [--json] [-k N] [--rerank]
folio status
folio install --project --platform cursor|claude|codex|agents
folio uninstall --project [--purge]
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q                  # all
pytest -q -m "not integration"   # fast (no Chroma embed)
pytest -q -m integration         # sync + ingest + retrieve + pack
```

CI runs both layers on Python 3.11/3.12. Tag `v*` to build a Release index pack via GitHub Actions.

## Maintainer: publish index for `folio bootstrap`

```bash
folio build
folio pack                 # → dist/folio-index-YYYY.MM.DD.tar.gz
# upload as GitHub Release asset, then set in config:
# release.index_url: https://github.com/<you>/folio/releases/latest/download/folio-index.tar.gz
```
