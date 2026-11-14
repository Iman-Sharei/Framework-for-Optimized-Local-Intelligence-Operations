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
