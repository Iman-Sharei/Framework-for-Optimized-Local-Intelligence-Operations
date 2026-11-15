#!/usr/bin/env python3
"""Runtime audit repro for folio security/install hypotheses H1–H5.

Run from repo root with local sources on PYTHONPATH (not the PyPI wheel alone):
  cd ~/Desktop/folio
  PYTHONPATH=src python3 scripts/debug_audit_repro.py
"""
from __future__ import annotations

import io
import os
import pickle
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOG = ROOT / ".cursor" / "debug-d915d2.log"


def _make_traversal_tar(path: Path, marker: Path) -> None:
    """Tar with ../ escape member + required pack members."""
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "manifest.json").write_text('{"index_version":"audit"}', encoding="utf-8")
        (t / "config.yaml").write_text("collection: {name: x}\n", encoding="utf-8")
        (t / "chroma").mkdir()
        (t / "chroma" / "ok").write_text("c", encoding="utf-8")
        (t / "bm25").mkdir()
        (t / "bm25" / "index.pkl").write_bytes(b"not-a-real-pickle")
        with tarfile.open(path, "w:gz") as tar:
            for name in ("chroma", "bm25", "manifest.json", "config.yaml"):
                tar.add(t / name, arcname=name)
            # Path traversal payload: write marker outside staging via ../
            data = b"TRAVERSAL_OK\n"
            info = tarfile.TarInfo(name="../ESCAPED_BY_FOLIO.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


class _Boom:
    def __reduce__(self):
        marker = Path(os.environ["FOLIO_AUDIT_PICKLE_MARKER"])
        return (marker.write_text, ("PICKLE_RCE_OK\n",))


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"[audit] log → {LOG}")

    # --- H3: version lookup ---
    from folio.cli import _cli_version

    v = _cli_version()
    print(f"[H3] _cli_version() → {v!r}")

    with tempfile.TemporaryDirectory(prefix="folio-audit-") as base_s:
        base = Path(base_s)
        out = base / "folio-home"
        out.mkdir()
        os.environ["FOLIO_HOME"] = str(out)

        from folio.config import load_config
        from folio.distribute import _extract_into_out, bootstrap_index
        from folio.store import load_bm25

        # --- H1: tar traversal ---
        evil_tar = base / "evil.tar.gz"
        escape_parent = base / "staging-parent-probe"
        escape_parent.mkdir()
        # extract into a nested staging so ../ lands in escape_parent
        staging = escape_parent / "staging"
        _make_traversal_tar(evil_tar, escape_parent / "ESCAPED_BY_FOLIO.txt")
        print("[H1] extracting malicious tar via _extract_into_out…")
        try:
            _extract_into_out(evil_tar, out)
            print("[H1] _extract_into_out completed")
        except Exception as e:
            print(f"[H1] _extract_into_out raised: {type(e).__name__}: {e}")

        from folio.distribute import _extract_pack

        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        print("[H1] raw _extract_pack into staging…")
        try:
            _extract_pack(evil_tar, staging)
            print("[H1] raw extract completed (traversal members should be skipped)")
        except Exception as e:
            print(f"[H1] raw extract raised: {type(e).__name__}: {e}")
        escaped = escape_parent / "ESCAPED_BY_FOLIO.txt"
        print(
            f"[H1] escaped file exists={escaped.exists()} "
            f"(expect False after fix) path={escaped}"
        )

        # --- H5: symlink member ---
        link_tar = base / "link.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "manifest.json").write_text("{}", encoding="utf-8")
            (t / "config.yaml").write_text("x: 1\n", encoding="utf-8")
            (t / "chroma").mkdir()
            secret = base / "secret.txt"
            secret.write_text("SECRET\n", encoding="utf-8")
            (t / "bm25").mkdir()
            # Create symlink named bm25/index.pkl pointing at secret — via tar symlink member
            with tarfile.open(link_tar, "w:gz") as tar:
                tar.add(t / "chroma", arcname="chroma")
                tar.add(t / "manifest.json", arcname="manifest.json")
                tar.add(t / "config.yaml", arcname="config.yaml")
                # bm25 as directory containing symlink
                info = tarfile.TarInfo(name="bm25")
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
                link = tarfile.TarInfo(name="bm25/index.pkl")
                link.type = tarfile.SYMTYPE
                link.linkname = str(secret)
                tar.addfile(link)
        print("[H5] extracting symlink pack…")
        out2 = base / "folio-home-link"
        out2.mkdir()
        try:
            _extract_into_out(link_tar, out2)
            dest_pkl = out2 / "bm25" / "index.pkl"
            print(
                f"[H5] dest exists={dest_pkl.exists()} is_symlink={dest_pkl.is_symlink() if dest_pkl.exists() else None}"
            )
            if dest_pkl.exists() and not dest_pkl.is_symlink():
                print(f"[H5] copied content={dest_pkl.read_text()!r}")
        except Exception as e:
            print(f"[H5] raised: {type(e).__name__}: {e}")

        # --- H2: pickle RCE ---
        marker = base / "pickle_rce.txt"
        os.environ["FOLIO_AUDIT_PICKLE_MARKER"] = str(marker)
        pkl = out / "bm25" / "index.pkl"
        pkl.parent.mkdir(parents=True, exist_ok=True)
        with pkl.open("wb") as f:
            pickle.dump(_Boom(), f)
        print("[H2] calling load_bm25 on malicious pickle…")
        try:
            load_bm25(pkl)
            print("[H2] load SUCCEEDED (unexpected after fix)")
        except Exception as e:
            print(f"[H2] load raised: {type(e).__name__}: {e}")
        print(
            f"[H2] marker exists={marker.exists()} "
            f"(expect False after fix) content={marker.read_text() if marker.exists() else None!r}"
        )

        # --- H4: file:// URL ---
        # Build a minimal valid pack for URL download path
        good = base / "good.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "manifest.json").write_text('{"ok":1}', encoding="utf-8")
            (t / "config.yaml").write_text("collection:\n  name: x\n", encoding="utf-8")
            (t / "chroma").mkdir()
            (t / "bm25").mkdir()
            (t / "bm25" / "index.pkl").write_bytes(pickle.dumps({"ids": []}))
            with tarfile.open(good, "w:gz") as tar:
                for name in ("chroma", "bm25", "manifest.json", "config.yaml"):
                    tar.add(t / name, arcname=name)
        out3 = base / "folio-home-url"
        os.environ["FOLIO_HOME"] = str(out3)
        cfg = load_config()
        file_url = good.resolve().as_uri()
        print("[H4] bootstrap via file:// …")
        try:
            bootstrap_index(cfg, url=file_url, force=True)
            print("[H4] file:// bootstrap SUCCEEDED (unexpected after fix)")
        except Exception as e:
            print(f"[H4] file:// bootstrap raised: {type(e).__name__}: {e}")

    print("[audit] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# path-view refresh
