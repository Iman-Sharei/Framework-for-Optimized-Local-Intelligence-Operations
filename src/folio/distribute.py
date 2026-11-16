from __future__ import annotations

import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from .config import ensure_data_dirs, resolve_out_dir, write_default_config
from . import __version__ as cli_version
from .ingest import index_build_metadata, index_version_now, ingest
from .store import anonymize_index_copy
from .sync import sync_sources
from .term import DownloadBar, path_link, url_link

# Files/dirs included in a portable index pack (query-ready without sources).
_PACK_MEMBERS = ("chroma", "bm25", "manifest.json", "config.yaml")
_PACK_TOP = frozenset(_PACK_MEMBERS)


def _release_url(cfg: dict[str, Any]) -> str:
    release = cfg.get("release") or {}
    url = (release.get("index_url") or "").strip()
    if url:
        return url
    # Local ~/.folio/config.yaml may have an empty placeholder; fall back to bundle.
    try:
        from .config import default_config_text
        import yaml

        bundled = yaml.safe_load(default_config_text()) or {}
        return ((bundled.get("release") or {}).get("index_url") or "").strip()
    except Exception:
        return ""


def _require_https_url(url: str) -> str:
    """Bootstrap downloads must be https (blocks file:// and SSRF-ish schemes)."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RuntimeError(
            f"Refusing non-https index URL (scheme={parsed.scheme!r}). "
            "Use https://… or pass a local --archive path."
        )
    return url


def _safe_tar_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Allowlist pack members; reject path traversal, absolute paths, and links."""
    safe: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        name = member.name
        if name.startswith("./"):
            name = name[2:]
        parts = Path(name).parts
        if not parts or parts[0] not in _PACK_TOP:
            continue
        if Path(name).is_absolute() or ".." in parts:
            raise RuntimeError(f"Unsafe tar member path: {member.name!r}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Tar symlinks/hardlinks are not allowed: {member.name!r}")
        # Re-bind normalized relative name for extract.
        member.name = name
        safe.append(member)
    if not safe:
        raise RuntimeError("Invalid index pack: no allowed members found")
    return safe


def public_manifest(data: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Fields a published pack may contain. No source names, paths, or URLs."""
    build = index_build_metadata(cfg)
    return {
        "index_version": data.get("index_version") or index_version_now(),
        "cli_version": cli_version,
        "ingested_at": data.get("ingested_at"),
        "files": data.get("files"),
        "chunks": data.get("chunks"),
        "collection": data.get("collection") or cfg.get("collection", {}).get("name"),
        "embedding_model": build["embedding_model"],
        "embedding": build["embedding"],
        "chunking": build["chunking"],
    }


def public_config(cfg_text: str) -> str:
    """Pack config keeps retrieval knobs and an empty source list."""
    data = yaml.safe_load(cfg_text) or {}
    if not isinstance(data, dict):
        data = {}
    data["sources"] = []
    return yaml.safe_dump(data, sort_keys=False)


def pack_index(cfg: dict[str, Any], *, dest: Path | None = None, keep_headings: bool = False) -> Path:
    """Create a tarball of the shared index for GitHub Release upload.

    The local index is not modified. The tarball is a copy with source ids,
    file paths, and the source catalog removed. Chunk text is still inside it.
    """
    out = Path(cfg["paths"]["out_dir"])
    manifest_path = Path(cfg["paths"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"No index to pack. Run `folio build` first ({manifest_path})")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = data.get("index_version") or index_version_now()
    if dest is None:
        dest = Path.cwd() / "dist" / f"folio-index-{version}.tar.gz"
    dest = dest.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    config_path = out / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    if not config_text:
        from .config import default_config_text

        config_text = default_config_text()

    with tempfile.TemporaryDirectory(prefix="folio-pack-") as tmp:
        stage = Path(tmp) / "stage"
        stage.mkdir()
        chroma_src = out / "chroma"
        bm25_src = out / "bm25"
        if not chroma_src.exists() or not bm25_src.exists():
            raise FileNotFoundError(f"Missing chroma or bm25 under {out}")
        shutil.copytree(chroma_src, stage / "chroma")
        shutil.copytree(bm25_src, stage / "bm25")
        (stage / "manifest.json").write_text(
            json.dumps(public_manifest(data, cfg), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (stage / "config.yaml").write_text(public_config(config_text), encoding="utf-8")
        anonymize_index_copy(
            stage,
            str(cfg.get("collection", {}).get("name") or data.get("collection") or "folio_knowledge"),
            keep_headings=keep_headings,
        )
        with tarfile.open(dest, "w:gz") as tar:
            for name in _PACK_MEMBERS:
                tar.add(stage / name, arcname=name)

    print(f"[pack] wrote {dest} (index_version={version}, sources omitted)")
    return dest


def _extract_pack(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = _safe_tar_members(tar)
        # Prefer PEP 706 data filter when available (3.12+).
        extract_kw: dict[str, Any] = {"path": dest, "members": members}
        if hasattr(tarfile, "data_filter"):
            extract_kw["filter"] = "data"
        tar.extractall(**extract_kw)


def bootstrap_index(
    cfg: dict[str, Any],
    *,
    url: str | None = None,
    archive: Path | None = None,
    force: bool = False,
) -> Path:
    """Install a prebuilt index into FOLIO_HOME (~/.folio)."""
    out = Path(cfg["paths"]["out_dir"])
    manifest = Path(cfg["paths"]["manifest_path"])
    if manifest.exists() and not force:
        print(
            f"✅ [bootstrap] index already present at {path_link(out)} "
            "(use --force to replace)"
        )
        return out

    ensure_data_dirs(cfg)
    write_default_config(out)

    if archive is not None:
        src = archive.expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        print(f"📦 [bootstrap] extracting {path_link(src)} → {path_link(out)}")
        _extract_into_out(src, out)
        print(f"✅ [bootstrap] ready at {path_link(out)}")
        return out

    resolved = (url or _release_url(cfg)).strip()
    if not resolved:
        raise RuntimeError(
            "No release URL configured. Set release.index_url in config, "
            "pass --url, or run `folio build` / `folio update --from-source`."
        )

    resolved = _require_https_url(resolved)

    with tempfile.TemporaryDirectory(prefix="folio-bootstrap-") as tmp:
        tmp_path = Path(tmp) / "folio-index.tar.gz"
        print(f"⬇️  [bootstrap] downloading {url_link(resolved)}")
        _download(resolved, tmp_path)
        _extract_into_out(tmp_path, out)

    print(f"✅ [bootstrap] ready at {path_link(out)}")
    return out


def _download(url: str, dest: Path, *, attempts: int = 4) -> None:
    """Download a release asset with a live linear progress chart.

    GitHub often resets a long transfer once; we retry a few times.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "folio-cli"})
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        bar: DownloadBar | None = None
        try:
            with urllib.request.urlopen(request, timeout=180) as response, dest.open("wb") as handle:
                # Content-Length may be missing on some CDNs; still show a live bar.
                raw_len = response.headers.get("Content-Length")
                total = int(raw_len) if raw_len and raw_len.isdigit() else None
                bar = DownloadBar(total, label="folio-index.tar.gz")
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bar.update(len(chunk))
                bar.close()
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
            if bar is not None:
                # Close the single progress line cleanly before retrying.
                try:
                    bar.close()
                except Exception:
                    pass
            dest.unlink(missing_ok=True)
            if attempt == attempts:
                break
            print(
                f"[bootstrap] download interrupted ({exc}); "
                f"retry {attempt + 1}/{attempts}",
                file=sys.stderr,
            )
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed to download index: {last}") from last


def _replace_tree(src: Path, dest: Path) -> None:
    """Copy ``src`` beside ``dest`` and swap only after the copy succeeds.

    A failed copy leaves the previous index in place.
    """
    if src.is_symlink():
        raise RuntimeError(f"Refusing symlink pack member: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    incoming = dest.parent / f".{dest.name}.incoming"
    if incoming.exists():
        shutil.rmtree(incoming) if incoming.is_dir() else incoming.unlink()
    try:
        if src.is_dir():
            shutil.copytree(src, incoming, symlinks=True, ignore_dangling_symlinks=False)
            for path in incoming.rglob("*"):
                if path.is_symlink():
                    raise RuntimeError(f"Refusing symlink inside pack member {dest.name}: {path}")
        else:
            shutil.copy2(src, incoming, follow_symlinks=False)
    except Exception:
        if incoming.exists():
            shutil.rmtree(incoming, ignore_errors=True) if incoming.is_dir() else incoming.unlink(
                missing_ok=True
            )
        raise
    backup = dest.parent / f".{dest.name}.backup"
    if backup.exists():
        shutil.rmtree(backup) if backup.is_dir() else backup.unlink()
    if dest.exists():
        dest.rename(backup)
    try:
        incoming.rename(dest)
    except Exception:
        # The live name was already moved aside. Put it back if the swap failed.
        if backup.exists() and not dest.exists():
            backup.rename(dest)
        raise
    if backup.exists():
        shutil.rmtree(backup) if backup.is_dir() else backup.unlink()


def _extract_into_out(archive: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="folio-extract-") as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        _extract_pack(archive, staging)
        for name in _PACK_MEMBERS:
            src = staging / name
            if not src.exists():
                raise RuntimeError(f"Invalid index pack: missing {name}")
            _replace_tree(src, out / name)


def update_index(
    cfg: dict[str, Any],
    *,
    from_source: bool = False,
    url: str | None = None,
    force: bool = True,
) -> int:
    """Refresh the shared index: Release pack by default, or rebuild from sources."""
    ensure_data_dirs(cfg)
    write_default_config(cfg["paths"]["out_dir"])

    if from_source:
        print("[update] rebuilding from curated sources…")
        sync_sources(cfg, force=False)
        ingest(cfg, rebuild=True)
        return 0

    try:
        bootstrap_index(cfg, url=url, force=force)
        return 0
    except RuntimeError as e:
        print(f"[update] could not download the published index: {e}", file=sys.stderr)
        return 1


def migrate_legacy_index(legacy: Path | None = None) -> Path | None:
    """One-shot: move Desktop/aieng/aieng-out → ~/.folio if needed."""
    dest = resolve_out_dir()
    if (dest / "manifest.json").exists():
        return dest
    # A non-empty home that is not already an index may hold unrelated files.
    # model-cache is created by the CLI itself on startup, so it does not count.
    if dest.exists() and any(p.name != "model-cache" for p in dest.iterdir()):
        print(
            f"[migrate] {dest} is not empty and has no manifest; "
            "leaving it in place"
        )
        return None

    candidates = []
    if legacy:
        candidates.append(legacy.expanduser())
    candidates.extend(
        [
            Path.home() / "Desktop" / "aieng" / "aieng-out",
            Path.home() / "Desktop" / "folio" / "aieng-out",
            Path.home() / "Desktop" / "folio" / "folio-out",
            Path(__file__).resolve().parents[2] / "aieng-out",
        ]
    )
    for src in candidates:
        if (src / "manifest.json").exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and any(p.name != "model-cache" for p in dest.iterdir()):
                print(
                    f"[migrate] {dest} is not empty and has no manifest; "
                    "leaving it in place"
                )
                return None
            if dest.exists():
                dest.rmdir()
            shutil.move(str(src), str(dest))
            print(f"[migrate] moved {src} → {dest}")
            return dest
    return None
