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

PyPI distribution name is **`folio-cli`** (plain `folio` is taken). The CLI command is still **`folio`**.

```bash
uv tool install folio-cli
folio --help
```

From a checkout (editable / unreleased):

```bash
cd ~/Desktop/folio
uv sync                 # creates .venv + uv.lock-backed env
uv run folio status
# or install the CLI globally from this tree:
uv tool install --editable .
```

Git install without waiting for PyPI:

```bash
uv tool install git+https://github.com/Iman-Sharei/folio
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
uv sync
uv run pytest -q                  # all
uv run pytest -q -m "not integration"   # fast (no Chroma embed)
uv run pytest -q -m integration         # sync + ingest + retrieve + pack
```

CI runs both layers on Python 3.11/3.12. Tag `vX.Y.Z` to:
- upload the curricula index pack (`folio-index.tar.gz`) to the GitHub Release
- publish the `folio-cli` wheel/sdist to PyPI (Trusted Publishing; see below)

## Maintainer: publish `folio-cli` to PyPI

One-time setup (API token **or** Trusted Publishing):

```bash
./scripts/publish-pypi-wizard.sh
```

After Trusted Publishing is configured, a version tag is enough:

```bash
git tag -a v0.3.1 -m v0.3.1
git push origin v0.3.1
```

Manual local publish (token in env):

```bash
uv build
UV_PUBLISH_TOKEN=pypi-... uv publish
```

## Maintainer: publish index for `folio bootstrap`

```bash
folio build
folio pack                 # → dist/folio-index-YYYY.MM.DD.tar.gz
# upload as GitHub Release asset, then set in config:
# release.index_url: https://github.com/<you>/folio/releases/latest/download/folio-index.tar.gz
```
