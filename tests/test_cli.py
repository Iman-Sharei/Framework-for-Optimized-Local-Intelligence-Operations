from __future__ import annotations

import json
from pathlib import Path

import pytest

from folio.cli import main
from folio.ingest import ingest
from folio.sync import sync_sources


def test_cli_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_cli_status(folio_home: Path, capsys: pytest.CaptureFixture[str]):
    code = main(["status", "--out", str(folio_home)])
    assert code == 0
    out = capsys.readouterr().out
    assert "folio" in out
    assert str(folio_home) in out


def test_cli_sources_requires_manifest(folio_home: Path):
    code = main(["sources", "--out", str(folio_home)])
    assert code == 2


def test_cli_sources_does_not_reveal_identities(folio_home: Path, capsys: pytest.CaptureFixture[str]):
    manifest = {
        "index_version": "2026.11.16",
        "cli_version": "0.4.6",
        "files": 3,
        "chunks": 9,
        "embedding_model": "onnx-minilm-l6-v2",
        "embedding": {
            "model": "onnx-minilm-l6-v2",
            "provider": "chromadb-onnx-minilm-l6-v2",
            "max_tokens": 256,
            "window_tokens": 200,
            "window_overlap_tokens": 40,
            "pooling": "mean",
        },
        "chunking": {"max_chars": 1200, "max_tokens": 200, "overlap_tokens": 40},
        "sources": [
            {
                "id": "secret-corpus",
                "repo": "https://secret.example/internal.git",
                "sparse_path": "lessons",
            }
        ],
        "file_list": [{"source_id": "secret-corpus", "path": "secret-corpus/secret.md"}],
    }
    (folio_home / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    code = main(["sources", "--out", str(folio_home)])
    assert code == 0
    text = capsys.readouterr().out
    assert "secret.example" not in text
    assert "secret-corpus" not in text
    assert "window_tokens=200" in text
    code = main(["sources", "--json", "--out", str(folio_home)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    blob = json.dumps(payload)
    assert "secret" not in blob
    assert payload["source_catalog"] == "omitted"
    assert payload["source_count"] == 1
    assert payload["embedding"]["window_tokens"] == 200


def test_cli_query_requires_index(folio_home: Path):
    code = main(["query", "anything", "--out", str(folio_home)])
    assert code == 2


@pytest.mark.integration
def test_cli_query_json(
    mini_cfg: dict,
    sample_curriculum_repo: Path,
    folio_home: Path,
    capsys: pytest.CaptureFixture[str],
):
    mini_cfg["sources"] = [
        {
            "id": "demo",
            "repo": str(sample_curriculum_repo),
            "sparse_path": "lessons",
            "branch": "main",
        }
    ]
    sync_sources(mini_cfg, force=True)
    ingest(mini_cfg, rebuild=True)
    capsys.readouterr()  # drain sync/ingest stdout noise

    code = main(
        [
            "query",
            "hybrid retrieval BM25 dense",
            "--json",
            "-k",
            "2",
            "--out",
            str(folio_home),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload
    assert {"chunk_id", "score", "source_id", "rel_path", "text"} <= set(payload[0])
