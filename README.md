<pre align="center">
@@@@@@@@@@@   //@@@@@\    @@@          @@@@@@@@@@@   //@@@@@\
@@@          @@/     \@@  @@@              @@@      @@/     \@@
@@@          @@@     @@@  @@@              @@@      @@@     @@@
@@@@@@@@@    @@@     @@@  @@@              @@@      @@@     @@@
@@@          @@@     @@@  @@@              @@@      @@@     @@@
@@@          @@\     /@@  @@@              @@@      @@\     /@@
@@@           \@@@@@//    @@@@@@@@@@@  @@@@@@@@@@@   \@@@@@//

              LOCAL HYBRID RAG FOR AI ENGINEERING
                  MLOps · AI INFRA · LANGCHAIN
                   KNOWLEDGE RETRIEVAL SERVICE
</pre>

<p align="center">
  <img src="docs/assets/folio-banner.jpg" alt="folio — local RAG for AI engineering assistants" width="100%">
</p>

<h1 align="center">📚 folio</h1>

<p align="center">
  <strong>folio</strong> = <em>Framework for Optimized Local Intelligence Operations</em><br>
  <strong>Local hybrid RAG</strong> for AI engineering · MLOps · AI infra.<br>
  Your assistant asks a short question → gets a <strong>budget-capped</strong> answer from a shared index on your machine.
</p>

<p align="center">
  <a href="https://pypi.org/project/folio-cli/"><img src="https://img.shields.io/pypi/v/folio-cli.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/folio-cli/"><img src="https://img.shields.io/pypi/pyversions/folio-cli.svg" alt="Python"></a>
  <a href="https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations.svg" alt="License"></a>
  <a href="https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/actions/workflows/ci.yml"><img src="https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/stargazers"><img src="https://img.shields.io/github/stars/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#-how-it-works-in-3-steps">Why</a> ·
  <a href="#-install">Install</a> ·
  <a href="#-use-in-a-project">Multi-agent</a> ·
  <a href="#-book--skill">Book → skill</a> ·
  <a href="#-commands">Commands</a>
</p>

> 📦 PyPI: [`folio-cli`](https://pypi.org/project/folio-cli/) · CLI: **`folio`** · Index: `~/.folio` · [Releases](https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/releases/latest)

---

## ✨ Why

<img align="right" width="280" src="docs/assets/folio-owl.png" alt="folio owl mascot — standing with a notebook">

Dumping a whole curriculum into chat wastes tokens and still hallucinates. folio pays the index cost once, then every answer stays small.

The usual workarounds don't help:
- 📄 "I'll paste the docs into chat" → context blows up, answers get noisy
- 🧠 "The model already knows this" → it guesses, or invents APIs
- 📝 "I'll keep a huge notes file" → nobody opens it when coding

**folio solves this with a local hybrid index your assistant queries on demand.**

📏 **Default print budget: about 2 000 tokens** — a 4-characters-per-token cap on what is printed. Retrieval still searches the index; only the print is capped.

<br clear="all">

---

## 🚀 How it works, in 3 steps

1. **Bootstrap** the shared index once — `folio bootstrap`
2. **Wire** your assistant(s) — `folio install --project` (or `--platform all`)
3. **Ask** one concrete question — `folio query "when should retrieval widen the candidate set?" --budget 2000`

Your agent loads only the capped hits, then decides. No curriculum dump. No naming sources in chat.

---

## 📥 Install

Current package release: **0.4.9** (`folio-cli` on PyPI).

```bash
uv tool install folio-cli
folio --help
```

Upgrade an existing install, then refresh the local index:

```bash
uv tool upgrade folio-cli
# or: pip install -U folio-cli
# or: pipx upgrade folio-cli
folio update
```

`folio update` downloads the published index again. It does not change the program version. More: [uv tool docs](https://docs.astral.sh/uv/guides/tools/) · [PyPI folio-cli](https://pypi.org/project/folio-cli/)

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

Pack: [`folio-index.tar.gz`](https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/releases/latest/download/folio-index.tar.gz)

### How a query is answered

BM25 and a dense search each return a candidate set (default 20). The lists are merged with reciprocal rank fusion. If the best dense cosine similarity is below **0.20**, both sets are widened. Older configs that still say `0.012` are treated as 0.20, because that number was a rank-fusion cutoff and never meant "weak similarity." `--budget` only limits the printed text (default 2000, about 8000 characters).

```bash
folio query "when should retrieval widen the candidate set?" --budget 2000
folio sources
```

`folio sources` prints the embedding model, window, pooling, and chunk settings recorded in the installed manifest. It does not print source names. A published pack has no source catalog. `folio pack` rewrites ids so file paths and source ids are not in the tarball, and it writes `sources: []` into the packed config. The chunk text itself stays in the database, so the passages are still readable.

Vectors are Chroma's ONNX MiniLM-L6-v2 embeddings (256-token model). Passages longer than 200 tokens are split into windows of 200 with a 40-token overlap, embedded, and mean-pooled. The same chunks are scored with BM25. Default chunk settings: `max_chars` 1200, `overlap_chars` 150, `min_chars` 80, `max_tokens` 200, `overlap_tokens` 40.

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

---

## ⌨️ Commands

```bash
folio bootstrap
folio update
folio status
folio sources
folio query "<question>" [--budget N]
folio install --project [--platform cursor|claude|codex|agents|all]
folio uninstall --project [--platform cursor|claude|codex|agents]
folio skill-from <path> [--name NAME] [--platform agents|claude|codex|all]
folio docs-suggest "<LangChain task>"
```

`folio query` searches the index with BM25 and dense retrieval, then prints at most `--budget` tokens (default **2000**). `folio pack` publishes a redacted tarball; see [docs/maintainer.md](docs/maintainer.md).

---

## 🛠️ Maintainer

Users do **not** add documents. Building and publishing the shared database is a local maintainer workflow: sources stay in `sources.local.yaml` on that machine, and `folio pack` writes a tarball with no source names. See [docs/maintainer.md](docs/maintainer.md).

```bash
folio add /path/to/file.pdf|.md|.txt|.json
folio pack --dest dist/folio-index.tar.gz
```

Tagging `vX.Y.Z` publishes the program to PyPI through the [`pypi`](https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/settings/environments) environment and Trusted Publishing. That tag does **not** rebuild or replace `folio-index.tar.gz`. Upload the tarball you just packed:

```bash
gh release upload vX.Y.Z dist/folio-index.tar.gz --clobber --repo Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations
```

```bash
uv sync && uv run pytest -q
```

CI: [GitHub Actions](https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/actions/workflows/ci.yml)
---

## 🔗 Links

- 📦 [PyPI — folio-cli](https://pypi.org/project/folio-cli/)
- 🐙 [GitHub — Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations](https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations)
- 🏷️ [Releases & index pack](https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/releases)
- 🐛 [Issues](https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/issues)
- ⚖️ [Apache-2.0 license](https://github.com/Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations/blob/main/LICENSE)
