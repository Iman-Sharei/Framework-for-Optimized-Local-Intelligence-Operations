<p align="center">
  <img src="docs/assets/folio-banner.jpg" alt="folio — local RAG for AI engineering assistants" width="100%">
</p>

<p align="center">
  <img src="docs/assets/folio-avatar.jpg" alt="folio mascot" width="120">
</p>

<h1 align="center">📚 folio</h1>

<p align="center">
  <strong>Local hybrid RAG</strong> for AI engineering · MLOps · AI infra.<br>
  Your assistant asks a short question → gets a <strong>budget-capped</strong> answer from a shared index on your machine.
</p>

<p align="center">
  <a href="https://pypi.org/project/folio-cli/"><img src="https://img.shields.io/pypi/v/folio-cli.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/folio-cli/"><img src="https://img.shields.io/pypi/pyversions/folio-cli.svg" alt="Python"></a>
  <a href="https://github.com/Iman-Sharei/folio/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Iman-Sharei/folio.svg" alt="License"></a>
  <a href="https://github.com/Iman-Sharei/folio/actions/workflows/ci.yml"><img src="https://github.com/Iman-Sharei/folio/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Iman-Sharei/folio/stargazers"><img src="https://img.shields.io/github/stars/Iman-Sharei/folio?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#-how-it-works-in-3-steps">Why</a> ·
  <a href="#-install">Install</a> ·
  <a href="#-use-in-a-project">Multi-agent</a> ·
  <a href="#-book--skill">Book → skill</a> ·
  <a href="#-commands">Commands</a>
</p>

> 📦 PyPI: [`folio-cli`](https://pypi.org/project/folio-cli/) · CLI: **`folio`** · Index: `~/.folio` · [Releases](https://github.com/Iman-Sharei/folio/releases/latest)

---

## ✨ Why

Dumping a whole curriculum into chat wastes tokens and still hallucinates. folio pays the index cost once, then every answer stays small.

📏 **Default print budget: 2 000 tokens** — retrieval still ranks the full index; only what is printed is capped. That keeps assistant context lean vs pasting long docs.

---

## 🚀 How it works, in 3 steps

1. **Bootstrap** the shared index once — `folio bootstrap`
2. **Wire** your assistant(s) — `folio install --project` (or `--platform all`)
3. **Ask** one concrete question — `folio query "when should retrieval widen the candidate set?" --budget 2000`

Your agent loads only the capped hits, then decides. No curriculum dump. No naming sources in chat.

---

## 📥 Install

```bash
uv tool install folio-cli
folio --help
```

Upgrade later:

```bash
uv tool upgrade folio-cli
```

More: [uv tool docs](https://docs.astral.sh/uv/guides/tools/) · [PyPI folio-cli](https://pypi.org/project/folio-cli/)

---

## 🗄️ Index

The index lives in **`~/.folio/`** (override with `FOLIO_HOME`). It is **not** stored inside product repos.

```bash
folio bootstrap     # first download of the published pack
folio status
```

`folio bootstrap` is a no-op when `~/.folio` already has an index. To replace it:

```bash
folio update
```

Pack: [`folio-index.tar.gz`](https://github.com/Iman-Sharei/folio/releases/latest/download/folio-index.tar.gz)

---

## 🧩 Use in a project

Omit `--platform` → installs for **[Cursor](https://cursor.com/)**.

```bash
cd /path/to/your-project
folio install --project
folio query "when should a retrieval step widen the candidate set?"
```

### 🤝 Multi-agent install

Wire **every** supported assistant in one shot:

```bash
folio install --project --platform all
```

| Platform | Files |
|---|---|
| `cursor` (default) | `.cursor/rules/folio.mdc`, `.agents/skills/folio/SKILL.md`, `AGENTS.md` |
| `claude` | `.claude/skills/folio/SKILL.md`, `CLAUDE.md` |
| `codex` | `.codex/skills/folio/SKILL.md` |
| `agents` | `.agents/skills/folio/SKILL.md`, `AGENTS.md` |
| `all` | cursor + claude + codex + agents |

```bash
folio install --project --platform claude
folio install --project --platform codex
```

Assistants must **not** name folio sources in replies, and must **not** pull curriculum repos into chat. Use `folio query`.

---

## 📖 Book → skill

Turn a book or doc you already own into an **on-demand agent skill** (structured `SKILL.md` + `chapters/`), without stuffing the whole file into context:

```bash
folio skill-from ./my-book.pdf --name my-book --platform all
# or markdown / plain text:
folio skill-from ./runbook.md --platform agents,claude
```

Writes into the project by default (`.agents/skills/<name>/`, plus Claude/Codex when requested). Use `--no-project` to install under your home directory instead.

> ⚖️ Bring files you have the right to use. folio does not ship book content — it only structures **your** copy locally.

To **index** a PDF into the shared RAG pack (maintainer path), use `folio add` instead.

---

## ⌨️ Commands

```bash
folio bootstrap
folio update
folio status
folio query "<question>" [--json] [-k N] [--budget N]
folio install --project [--platform cursor|claude|codex|agents|all]
folio uninstall --project [--platform cursor|claude|codex|agents]
folio skill-from <path> [--name NAME] [--platform agents|claude|codex|all]
folio docs-suggest "<LangChain task>"
```

`folio query` still searches the **full** index. `--budget` only caps what is printed (default **2000** tokens).

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

<p align="center">
  <img src="docs/assets/folio-og-card.jpg" alt="folio social card" width="640">
</p>
