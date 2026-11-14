from __future__ import annotations

import json
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config import ensure_data_dirs, resolve_out_dir, write_default_config
from .ingest import index_version_now, ingest
from .sync import sync_sources

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


def pack_index(cfg: dict[str, Any], *, dest: Path | None = None) -> Path:
    """Create a tarball of the shared index for GitHub Release upload."""
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

    with tarfile.open(dest, "w:gz") as tar:
        for name in _PACK_MEMBERS:
            path = out / name
            if not path.exists():
                raise FileNotFoundError(f"Missing pack member: {path}")
            tar.add(path, arcname=name)

    print(f"[pack] wrote {dest} (index_version={version})")
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
        print(f"[bootstrap] index already present at {out} (use --force to replace)")
        return out

    ensure_data_dirs(cfg)
    write_default_config(out)

    if archive is not None:
        src = archive.expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        print(f"[bootstrap] extracting {src} → {out}")
        _extract_into_out(src, out)
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
        print(f"[bootstrap] downloading {resolved}")
        try:
            urllib.request.urlretrieve(resolved, tmp_path)  # noqa: S310
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to download index: {e}") from e
        _extract_into_out(tmp_path, out)

    print(f"[bootstrap] ready at {out}")
    return out


def _extract_into_out(archive: Path, out: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="folio-extract-") as tmp:
        staging = Path(tmp) / "staging"
        staging.mkdir()
        _extract_pack(archive, staging)
        for name in _PACK_MEMBERS:
            src = staging / name
            if not src.exists():
                raise RuntimeError(f"Invalid index pack: missing {name}")
            dest = out / name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            if src.is_symlink():
                raise RuntimeError(f"Refusing symlink pack member: {src}")
            if src.is_dir():
                # symlinks=True copies link nodes as links (does not follow into secrets).
                shutil.copytree(src, dest, symlinks=True, ignore_dangling_symlinks=False)
                for p in dest.rglob("*"):
                    if p.is_symlink():
                        raise RuntimeError(f"Refusing symlink inside pack member {name}: {p}")
            else:
                shutil.copy2(src, dest, follow_symlinks=False)


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
        print(f"[update] bootstrap unavailable: {e}", file=sys.stderr)
        print("[update] falling back to --from-source build…", file=sys.stderr)
        sync_sources(cfg, force=False)
        ingest(cfg, rebuild=True)
        return 0


def migrate_legacy_index(legacy: Path | None = None) -> Path | None:
    """One-shot: move Desktop/aieng/aieng-out → ~/.folio if needed."""
    dest = resolve_out_dir()
    if (dest / "manifest.json").exists():
        return dest

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
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(src), str(dest))
            print(f"[migrate] moved {src} → {dest}")
            return dest
    return None
