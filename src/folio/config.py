from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

# Standalone tool checkout (…/Desktop/folio) — used only as a fallback for
# editable installs; the shared index lives under ~/.folio by default.
TOOL_ROOT = Path(__file__).resolve().parents[2]


def default_home() -> Path:
    """Shared index root (~/.folio). Computed at call time (never cache Path.home())."""
    return Path.home() / ".folio"


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def default_config_text() -> str:
    """Load bundled config.yaml (works for editable + installed wheels)."""
    bundled = package_dir() / "assets" / "config.yaml"
    if bundled.exists():
        return bundled.read_text(encoding="utf-8")
    legacy = TOOL_ROOT / "config.yaml"
    if legacy.exists():
        return legacy.read_text(encoding="utf-8")
    try:
        return resources.files("folio.assets").joinpath("config.yaml").read_text(
            encoding="utf-8"
        )
    except Exception as e:  # noqa: BLE001
        raise FileNotFoundError("Bundled folio config.yaml not found") from e


def resolve_out_dir(explicit: Path | None = None) -> Path:
    """Resolve shared knowledge index root.

    Priority:
    1. explicit --out
    2. FOLIO_HOME env
    3. ~/.folio
    """
    if explicit is not None:
        return explicit.expanduser().resolve()
    env = os.environ.get("FOLIO_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return default_home().resolve()


def load_config(
    *,
    out_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    if config_path is not None:
        raw = config_path.read_text(encoding="utf-8")
    else:
        root = resolve_out_dir(out_dir)
        local_cfg = root / "config.yaml"
        raw = local_cfg.read_text(encoding="utf-8") if local_cfg.exists() else default_config_text()

    cfg = yaml.safe_load(raw)
    cfg = copy.deepcopy(cfg)
    root = resolve_out_dir(out_dir)
    paths = cfg.setdefault("paths", {})
    paths["sources_dir"] = root / "sources"
    paths["chroma_dir"] = root / "chroma"
    paths["bm25_path"] = root / "bm25" / "index.pkl"
    paths["manifest_path"] = root / "manifest.json"
    paths["model_cache_dir"] = root / "model-cache"
    paths["out_dir"] = root
    cfg["_out_dir"] = root
    return cfg


def ensure_data_dirs(cfg: dict[str, Any]) -> None:
    paths = cfg["paths"]
    for key in ("sources_dir", "chroma_dir", "model_cache_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    Path(paths["bm25_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["manifest_path"]).parent.mkdir(parents=True, exist_ok=True)


def write_default_config(out_dir: Path | None = None) -> Path:
    root = resolve_out_dir(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "config.yaml"
    if not dest.exists():
        dest.write_text(default_config_text(), encoding="utf-8")
    return dest
