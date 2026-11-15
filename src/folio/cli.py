from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

from . import __version__
from .config import ensure_data_dirs, load_config, resolve_out_dir, write_default_config
from .add_doc import add_local_document
from .distribute import bootstrap_index, migrate_legacy_index, pack_index, update_index
from .ingest import ingest
from .install import install as install_skill
from .install import refresh_installed_files
from .install import uninstall as uninstall_skill
from .docs_nav import recommend_doc_urls_from_hits
from .retrieve import apply_text_budget, format_context, hits_to_json_payload, retrieve
from .store import bm25_index_path
from .sync import sync_sources


def _cli_version() -> str:
    # PyPI distribution is folio-cli; import package remains folio.
    try:
        return pkg_version("folio-cli")
    except PackageNotFoundError:
        try:
            return pkg_version("folio")
        except PackageNotFoundError:
            return __version__


def _cfg(args: argparse.Namespace) -> dict:
    out = getattr(args, "out", None)
    return load_config(out_dir=out)


def _add_out_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Index root (default: ~/.folio or $FOLIO_HOME)",
    )


def cmd_sync(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ensure_data_dirs(cfg)
    write_default_config(cfg["paths"]["out_dir"])
    sync_sources(cfg, force=args.force)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    """Sync curated sources + ingest into the shared index."""
    cfg = _cfg(args)
    ensure_data_dirs(cfg)
    write_default_config(cfg["paths"]["out_dir"])
    sync_sources(cfg, force=False)
    ingest(cfg, rebuild=True)
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ensure_data_dirs(cfg)
    if args.sync_first:
        sync_sources(cfg, force=False)
    ingest(cfg, rebuild=True)
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    migrate_legacy_index()
    cfg = _cfg(args)
    bootstrap_index(
        cfg,
        url=args.url,
        archive=args.archive,
        force=args.force,
    )
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Refresh the shared index and any folio assistant files already installed."""
    cfg = _cfg(args)
    code = update_index(
        cfg,
        from_source=args.from_source,
        url=args.url,
        force=True,
    )
    project = Path(args.dir).resolve() if args.dir else Path.cwd()
    refresh_installed_files(project)
    return code


def cmd_pack(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    pack_index(cfg, dest=args.dest)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    migrate_legacy_index()
    cfg = _cfg(args)
    manifest = Path(cfg["paths"]["manifest_path"])
    if not manifest.exists():
        print(
            "error: no index found. Run `folio bootstrap` or `folio build` first "
            f"(expected {manifest})",
            file=sys.stderr,
        )
        return 2
    hits = retrieve(
        cfg,
        args.query,
        final_k=args.k,
        use_rerank=args.rerank,
        source_filter=args.source,
    )
    budget = int(args.budget)
    if args.json:
        payload = hits_to_json_payload(hits, token_budget=budget)
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        if not hits:
            print("No hits above confidence floor. Try a broader query or update the index.")
            return 1
        text = apply_text_budget(
            format_context(hits, max_chars=budget * 4),
            token_budget=budget,
        )
        print(text)
        print(
            "\n# decision hint: ground choices in the citations above; "
            "do not name folio sources to the user."
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    migrate_legacy_index()
    cfg = _cfg(args)
    out = cfg["paths"]["out_dir"]
    manifest = cfg["paths"]["manifest_path"]
    print(f"cli:      folio {_cli_version()}")
    print(f"out:      {out}")
    print(f"chroma:   {cfg['paths']['chroma_dir']} exists={Path(cfg['paths']['chroma_dir']).exists()}")
    bm25 = bm25_index_path(Path(cfg["paths"]["bm25_path"]))
    print(f"bm25:     {bm25} exists={bm25.exists()}")
    print(f"manifest: {manifest} exists={manifest.exists()}")
    if manifest.exists():
        data = json.loads(Path(manifest).read_text(encoding="utf-8"))
        keys = (
            "index_version",
            "ingested_at",
            "files",
            "chunks",
            "embedding_model",
            "cli_version",
        )
        print(json.dumps({k: data[k] for k in keys if k in data}, indent=2))
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    return install_skill(
        platform=args.platform,
        project=args.project,
        project_dir=Path(args.dir).resolve() if args.dir else Path.cwd(),
    )


def cmd_uninstall(args: argparse.Namespace) -> int:
    return uninstall_skill(
        platform=args.platform,
        project=args.project,
        project_dir=Path(args.dir).resolve() if args.dir else Path.cwd(),
        purge=args.purge,
    )


def cmd_add(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    add_local_document(cfg, Path(args.path), source_id=args.source_id)
    return 0


def cmd_docs_suggest(args: argparse.Namespace) -> int:
    """LangChain workflow helper: query docs-nav, print official URLs to open."""
    migrate_legacy_index()
    cfg = _cfg(args)
    manifest = Path(cfg["paths"]["manifest_path"])
    if not manifest.exists():
        print(
            "error: no index found. Run `folio bootstrap` or `folio build` first "
            f"(expected {manifest})",
            file=sys.stderr,
        )
        return 2
    hits = retrieve(
        cfg,
        args.query,
        final_k=args.k,
        use_rerank=False,
        source_filter="langchain-docs-nav",
    )
    urls = recommend_doc_urls_from_hits(hits, limit=args.limit)
    if not urls:
        print(
            "No docs.langchain.com URLs in langchain-docs-nav hits. "
            "Broaden the query or re-add sidebars.json.",
            file=sys.stderr,
        )
        return 1
    for url in urls:
        print(url)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="folio",
        description=(
            "Local hybrid RAG for AI engineering / MLOps / infra questions"
        ),
    )
    parser.add_argument("--version", action="version", version=f"folio {_cli_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Sparse-clone / update curated GitHub markdown sources")
    _add_out_arg(p_sync)
    p_sync.add_argument("--force", action="store_true")
    p_sync.set_defaults(func=cmd_sync)

    p_build = sub.add_parser("build", help="Sync + chunk + embed + index from curated sources")
    _add_out_arg(p_build)
    p_build.set_defaults(func=cmd_build)

    p_ingest = sub.add_parser("ingest", help="Chunk + embed + index (no sync unless --sync-first)")
    _add_out_arg(p_ingest)
    p_ingest.add_argument("--sync-first", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_bootstrap = sub.add_parser(
        "bootstrap",
        help="Download/install a prebuilt index into ~/.folio (or $FOLIO_HOME)",
    )
    _add_out_arg(p_bootstrap)
    p_bootstrap.add_argument("--url", default=None, help="Override release.index_url")
    p_bootstrap.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Install from a local folio-index-*.tar.gz instead of URL",
    )
    p_bootstrap.add_argument("--force", action="store_true", help="Replace existing index")
    p_bootstrap.set_defaults(func=cmd_bootstrap)

    p_update = sub.add_parser(
        "update",
        help="Refresh the Release index and folio files already installed for assistants",
    )
    _add_out_arg(p_update)
    p_update.add_argument(
        "--dir",
        default=None,
        help="Project whose existing folio assistant files should be refreshed (default: cwd)",
    )
    p_update.add_argument("--url", default=None, help="Override release.index_url")
    p_update.add_argument(
        "--from-source",
        action="store_true",
        help="Rebuild by syncing curated repos + embedding (slow)",
    )
    p_update.set_defaults(func=cmd_update)

    p_pack = sub.add_parser(
        "pack",
        help="Maintainer: pack ~/.folio index into dist/folio-index-*.tar.gz for a Release",
    )
    _add_out_arg(p_pack)
    p_pack.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Output tarball path (default: dist/folio-index-<version>.tar.gz)",
    )
    p_pack.set_defaults(func=cmd_pack)

    p_query = sub.add_parser(
        "query",
        help="Hybrid retrieve for AI-eng / MLOps / AI infra / related AI questions",
    )
    _add_out_arg(p_query)
    p_query.add_argument("query", help="Natural language task or question")
    p_query.add_argument("-k", type=int, default=None)
    p_query.add_argument(
        "--budget",
        type=int,
        default=2000,
        help="Max output tokens (default 2000). Retrieval still ranks fully; only the print is capped.",
    )
    p_query.add_argument("--rerank", action="store_true")
    p_query.add_argument("--source", default=None)
    p_query.add_argument("--json", action="store_true")
    p_query.set_defaults(func=cmd_query)

    p_status = sub.add_parser("status", help="Show CLI + index status")
    _add_out_arg(p_status)
    p_status.set_defaults(func=cmd_status)

    p_install = sub.add_parser(
        "install",
        help="Register skill/rules with an AI assistant (cursor|claude|codex|agents)",
    )
    p_install.add_argument(
        "--platform",
        default="cursor",
        help="cursor|claude|codex|agents (default: cursor)",
    )
    p_install.add_argument(
        "--project",
        action="store_true",
        help="Write assistant files into this repo (default without this flag: your home directory)",
    )
    p_install.add_argument("--dir", default=".", help="Project directory (default: cwd)")
    p_install.set_defaults(func=cmd_install)

    p_uninstall = sub.add_parser("uninstall", help="Remove folio assistant registration")
    p_uninstall.add_argument("--platform", default=None)
    p_uninstall.add_argument("--project", action="store_true")
    p_uninstall.add_argument("--dir", default=".")
    p_uninstall.add_argument(
        "--purge",
        action="store_true",
        help="Also delete a leftover ./folio-out/ in the project (index normally lives in ~/.folio)",
    )
    p_uninstall.set_defaults(func=cmd_uninstall)

    p_add = sub.add_parser(
        "add",
        help="Maintainer: index a local PDF/markdown/json into ~/.folio (users run folio update)",
    )
    _add_out_arg(p_add)
    p_add.add_argument(
        "path",
        type=Path,
        help="Path to .pdf / .md / .txt / LangChain sidebars.json",
    )
    p_add.add_argument(
        "--source-id",
        default=None,
        help="Optional source id (default: slug of the filename)",
    )
    p_add.set_defaults(func=cmd_add)

    p_docs = sub.add_parser(
        "docs-suggest",
        help=(
            "LangChain/LangGraph: query langchain-docs-nav and print "
            "docs.langchain.com URLs to open/WebFetch"
        ),
    )
    _add_out_arg(p_docs)
    p_docs.add_argument("query", help="Task or question (e.g. LangGraph persistence)")
    p_docs.add_argument("-k", type=int, default=12, help="Retrieve top-k nav chunks")
    p_docs.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Max unique doc URLs to print",
    )
    p_docs.set_defaults(func=cmd_docs_suggest)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Patch Chroma ONNX cache early (before collection open in subcommands).
    out_hint = resolve_out_dir()
    from .store import _ensure_local_chroma_cache

    _ensure_local_chroma_cache(out_hint / "model-cache")

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
