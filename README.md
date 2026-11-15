# 📚 folio

Local hybrid RAG for **AI engineering · MLOps · AI infra** questions. Assistants query a
shared index on your machine and get a short, grounded answer — without dumping large
docs into chat.

[![PyPI](https://img.shields.io/pypi/v/folio-cli.svg)](https://pypi.org/project/folio-cli/)
[![Python](https://img.shields.io/pypi/pyversions/folio-cli.svg)](https://pypi.org/project/folio-cli/)
[![License](https://img.shields.io/github/license/Iman-Sharei/folio.svg)](https://github.com/Iman-Sharei/folio/blob/main/LICENSE)
[![CI](https://github.com/Iman-Sharei/folio/actions/workflows/ci.yml/badge.svg)](https://github.com/Iman-Sharei/folio/actions/workflows/ci.yml)
[![GitHub](https://img.shields.io/badge/GitHub-Iman--Sharei%2Ffolio-181717?logo=github)](https://github.com/Iman-Sharei/folio)

> PyPI package: [`folio-cli`](https://pypi.org/project/folio-cli/) · CLI command: **`folio`** · Index: [`~/.folio`](https://github.com/Iman-Sharei/folio#index) · Releases: [latest](https://github.com/Iman-Sharei/folio/releases/latest)

---

## 🚀 Install

```bash
uv tool install folio-cli
folio --help
```

More: [uv tool docs](https://docs.astral.sh/uv/guides/tools/) · [PyPI folio-cli](https://pypi.org/project/folio-cli/)

Upgrade later with:

```bash
uv tool upgrade folio-cli
```

---

## 🗄️ Index

The index lives in **`~/.folio/`** (override with [`FOLIO_HOME`](https://github.com/Iman-Sharei/folio#index)). It is **not** stored inside product repos.

```bash
folio bootstrap     # first download of the published pack
folio status
```

`folio bootstrap` is a no-op when `~/.folio` already has an index. To replace it:

```bash
folio update
```

Pack download: [`folio-index.tar.gz`](https://github.com/Iman-Sharei/folio/releases/latest/download/folio-index.tar.gz) (from [GitHub Releases](https://github.com/Iman-Sharei/folio/releases)).

`folio update` refreshes the pack **and** rewrites folio assistant files already present in the current project (or your home directory). It does **not** install folio into a project that never had it.

---

## 🧩 Use in a project

folio does **not** auto-detect your assistant. Omit `--platform` → installs for **[Cursor](https://cursor.com/)**.

```bash
cd /path/to/your-project
folio install --project
folio query "when should a retrieval step widen the candidate set?"
```

`--project` writes into the repo:

| Platform | Files |
|---|---|
| [`cursor`](https://cursor.com/) (default) | `.cursor/rules/folio.mdc`, `.agents/skills/folio/SKILL.md`, `AGENTS.md` |
| [`claude`](https://docs.anthropic.com/en/docs/claude-code) | `.claude/skills/folio/SKILL.md`, `CLAUDE.md` |
| [`codex`](https://github.com/openai/codex) | `.codex/skills/folio/SKILL.md` |
| `agents` | `.agents/skills/folio/SKILL.md`, `AGENTS.md` |

This repo already ships the Cursor, Claude, and Agents files. In another project, run `folio install --project` (add `--platform claude` when you need Claude).

```bash
folio install --project --platform claude
folio install --project --platform codex
```

Assistants must **not** name folio sources in replies, and must **not** pull curriculum repos into chat. Use `folio query`.

---

## ⌨️ Commands

```bash
folio bootstrap
folio update
folio status
folio query "<question>" [--json] [-k N] [--budget N]
folio install --project [--platform cursor|claude|codex|agents]
folio uninstall --project [--platform cursor|claude|codex|agents]
```

`folio query` still searches the **full** index. [`--budget`](https://github.com/Iman-Sharei/folio#commands) only caps what is printed (default **2000** tokens).

---

## 🧪 Tests

```bash
uv sync
uv run pytest -q
```

CI: [GitHub Actions](https://github.com/Iman-Sharei/folio/actions/workflows/ci.yml)

---

## 🛠️ Maintainer

Users do **not** add documents. `folio add` is for publishing a new pack.

```bash
folio add /path/to/file.pdf|.md|.txt|.json
folio pack
cp "$(ls -t dist/folio-index-*.tar.gz | head -1)" dist/folio-index.tar.gz
TAG=$(gh release view --repo Iman-Sharei/folio --json tagName -q .tagName)
gh release upload "$TAG" dist/folio-index.tar.gz --clobber --repo Iman-Sharei/folio
```

Publish the program by tagging `vX.Y.Z`. That uses the GitHub Environment [`pypi`](https://github.com/Iman-Sharei/folio/settings/environments) and a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/) for [`folio-cli`](https://pypi.org/project/folio-cli/).

---

## 🔗 Links

- 📦 [PyPI — folio-cli](https://pypi.org/project/folio-cli/)
- 🐙 [GitHub — Iman-Sharei/folio](https://github.com/Iman-Sharei/folio)
- 🏷️ [Releases & index pack](https://github.com/Iman-Sharei/folio/releases)
- 🐛 [Issues](https://github.com/Iman-Sharei/folio/issues)
- ⚖️ [Apache-2.0 license](https://github.com/Iman-Sharei/folio/blob/main/LICENSE)
