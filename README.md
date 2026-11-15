# folio

Local hybrid RAG for **AI engineering / MLOps / AI infra** questions. Assistants query a
shared index on the machine and get a short, grounded answer instead of pulling large
docs into chat.

## Install

PyPI name is **`folio-cli`**. The command is **`folio`**.

```bash
uv tool install folio-cli
folio --help
```

## Index

The index lives in **`~/.folio/`** (override with `FOLIO_HOME`). It is not stored inside product repos.

```bash
folio bootstrap     # first download of the published pack
folio status
```

`folio bootstrap` does nothing if `~/.folio` already has an index. To replace it:

```bash
folio update
```

`folio update` downloads the latest pack and rewrites folio assistant files that are already in the current project (or in your home directory). It does not install folio into a project that never had it. Upgrade the program itself with `uv tool upgrade folio-cli`.

## Use in a project

folio does **not** detect which assistant you use. If you omit `--platform`, it installs for **Cursor**.

```bash
cd /path/to/your-project
folio install --project
folio query "when should a retrieval step widen the candidate set?"
```

`--project` writes into the repo:

| platform | files |
|---|---|
| `cursor` (default) | `.cursor/rules/folio.mdc`, `.agents/skills/folio/SKILL.md`, `AGENTS.md` |
| `claude` | `.claude/skills/folio/SKILL.md`, `CLAUDE.md` |
| `codex` | `.codex/skills/folio/SKILL.md` |
| `agents` | `.agents/skills/folio/SKILL.md`, `AGENTS.md` |

This repo already includes the Cursor, Claude, and Agents files above. In another project, run `folio install --project` (and `--platform claude` when you need Claude).

Run it once for each assistant you use. For Cursor, the rule file is always written in the project directory. `SKILL.md` and `AGENTS.md` are added only with `--project`. For Claude, Codex, and `agents`, omitting `--project` writes those files under your home directory instead of the repo.

```bash
folio install --project --platform claude
folio install --project --platform codex
```

Assistants must not name folio sources in replies to you, and must not pull curriculum repos into chat. Use `folio query`.

## Commands

```bash
folio bootstrap
folio update
folio status
folio query "<question>" [--json] [-k N] [--budget N]
folio install --project [--platform cursor|claude|codex|agents]
folio uninstall --project [--platform cursor|claude|codex|agents]
```

`folio query` still searches the full index. `--budget` only caps what is printed (default 2000 tokens).

## Tests

```bash
uv sync
uv run pytest -q
```

## Maintainer

Users do not add documents. `folio add` is for publishing a new pack.

```bash
folio add /path/to/file.pdf|.md|.txt|.json
folio pack
cp "$(ls -t dist/folio-index-*.tar.gz | head -1)" dist/folio-index.tar.gz
TAG=$(gh release view --repo Iman-Sharei/folio --json tagName -q .tagName)
gh release upload "$TAG" dist/folio-index.tar.gz --clobber --repo Iman-Sharei/folio
```

Publish the program by tagging `vX.Y.Z`. That uses the GitHub Environment `pypi` and a PyPI Trusted Publisher for `folio-cli`.
