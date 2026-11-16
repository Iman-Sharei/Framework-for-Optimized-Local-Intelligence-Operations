# Maintainer: build and publish the index

Sources stay on your machine. The git repo and the GitHub release tarball do not list them.

`sources.local.yaml` is gitignored. `folio pack` also strips source ids, file paths, and the source catalog from the tarball it writes. Do not upload `~/.folio` itself. That directory still has the real ids, so you can add more documents later.

The chunk text is stored in the database. Anyone who downloads the pack can read those passages. Removing names does not remove the text.

## One-time setup

```bash
uv sync
cp sources.local.yaml.example sources.local.yaml
```

Edit `sources.local.yaml` with the files or git checkouts you want. Paths in that file are local. They are not packed.

```yaml
sources:
  - id: notes
    repo: /absolute/path/to/markdown-or-git-checkout
    sparse_path: .
    branch: main
```

`repo` may be a local directory or a git URL. `sparse_path: .` takes the whole tree. A subdirectory limits what is indexed.

## Fresh database

```bash
export FOLIO_HOME="$HOME/.folio"
folio sync
folio build
folio status
folio sources
```

`folio sync` clones or updates each entry. `folio build` chunks, embeds, and writes BM25. Your local manifest still records the source ids so the next add can replace them. `folio sources` prints the model and chunk settings only.

## Add more later, without rebuilding

```bash
folio add /absolute/path/to/notes.pdf --source-id notes
folio add /absolute/path/to/another.md
```

`folio add` embeds the new file and merges it into the existing Chroma and BM25 indexes. Adding the same `--source-id` again replaces that source's chunks.

## Pack and publish

```bash
folio pack --dest dist/folio-index.tar.gz
```

Optional: `folio pack --keep-headings` keeps heading titles. Source ids and file paths are still removed.

Check the tarball before upload. `manifest.json` and `config.yaml` inside it should have no source list. Chunk metadata should not contain source ids or paths.

Upload only when you mean to replace the release asset. This repo does not do that for you.

```bash
# Replace the asset on an existing tag
gh release upload vX.Y.Z dist/folio-index.tar.gz --clobber --repo Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations

# Optional integrity sidecar (folio bootstrap verifies when present)
sha256sum dist/folio-index.tar.gz | awk '{print $1}' > dist/folio-index.tar.gz.sha256
gh release upload vX.Y.Z dist/folio-index.tar.gz.sha256 --clobber --repo Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations

# A version tag publishes the Python package. It does not build or replace
# folio-index.tar.gz. Create the release, then upload the tarball you packed.
git tag vX.Y.Z
git push origin vX.Y.Z
gh release create vX.Y.Z dist/folio-index.tar.gz --title vX.Y.Z --repo Iman-Sharei/Framework-for-Optimized-Local-Intelligence-Operations
```

Users install that file with `folio bootstrap` or `folio update`. After install, folio rewrites `~/.folio/config.yaml` so `release.index_url` matches the URL that was actually downloaded (pack configs cannot redirect the next update). Override with `--url`, `FOLIO_INDEX_URL`, or `FOLIO_INDEX_SHA256`.

## What a published pack still contains

- Embedding model, window size, pooling, and chunk settings
- Index version, file count, and chunk count
- The chunk text and its vector

It does not contain source ids, file paths, repo URLs, or `sources.local.yaml`. `folio docs-suggest --source` cannot filter by source name on a published pack, because those ids were removed. A normal `folio query` still searches the text.
