#!/usr/bin/env python3
"""Build a publishable folio pack from sources.local.yaml.

The source list is never stored in this script. Copy
sources.local.yaml.example to sources.local.yaml and edit it locally.

This does not upload a GitHub release.

  uv run python scripts/rebuild_published_index.py --dest dist/folio-index.tar.gz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=Path.home() / ".folio")
    parser.add_argument("--dest", type=Path, default=ROOT / "dist" / "folio-index.tar.gz")
    parser.add_argument(
        "--sources",
        type=Path,
        default=ROOT / "sources.local.yaml",
        help="Local source list. Never commit this file.",
    )
    args = parser.parse_args()
    sources = args.sources.expanduser().resolve()
    if not sources.is_file():
        print(
            f"error: {sources} does not exist. Copy sources.local.yaml.example and edit it locally.",
            file=sys.stderr,
        )
        return 2
    home = args.home.expanduser().resolve()
    os.environ["FOLIO_HOME"] = str(home)
    # Keep the local file where load_config will see it, without copying it into the pack.
    home.mkdir(parents=True, exist_ok=True)
    linked = home / "sources.local.yaml"
    if sources != linked.resolve() and not linked.exists():
        linked.write_text(sources.read_text(encoding="utf-8"), encoding="utf-8")

    from folio.config import ensure_data_dirs, load_config, write_default_config
    from folio.distribute import pack_index
    from folio.ingest import ingest
    from folio.sync import sync_sources

    write_default_config(home)
    cfg = load_config(out_dir=home)
    if not cfg.get("sources"):
        print(f"error: {sources} has no sources.", file=sys.stderr)
        return 2
    ensure_data_dirs(cfg)
    sync_sources(cfg, force=False)
    ingest(cfg, rebuild=True)
    pack_index(cfg, dest=args.dest.expanduser().resolve(), keep_headings=False)
    print("[rebuild] local index still has source ids; only the tarball is redacted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
