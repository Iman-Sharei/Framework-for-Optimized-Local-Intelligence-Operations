"""Review probes for retrieval, chunking, install, and pack behavior.

These assert the contracts described in REVIEW.md. They do not need the
embedding model.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

from folio.add_doc import _walk_sidebar_links
from folio.chunk import _split_oversized, chunk_markdown
from folio.distribute import _extract_into_out, _extract_pack, migrate_legacy_index
from folio.install import install, uninstall
from folio.retrieve import (
    Hit,
    apply_text_budget,
    chars_for_token_budget,
    format_context,
    hits_to_json_payload,
    retrieve,
    rrf_fuse,
)
from folio.skill_from import skill_from
from folio.store import _bm25_json_path, load_bm25


def _hit(**kwargs) -> Hit:
    base = dict(
        chunk_id="c",
        text="hello",
        score=0.2,
        source_id="corpus",
        rel_path="a.md",
        heading_path="H",
    )
    base.update(kwargs)
    return Hit(**base)


def _retrieval_cfg(**overrides) -> dict:
    retrieval = {
        "dense_k": 20,
        "bm25_k": 20,
        "final_k": 8,
        "rrf_k": 60,
        "crag_min_best_score": 0.012,
        "crag_widen_factor": 2,
        "reranker_model": "fake-cross-encoder",
        "confidence_floor": 0.008,
    }
    retrieval.update(overrides)
    return {"retrieval": retrieval}


def _stub_searches(monkeypatch, dense_rows, bm25_rows, calls: list):
    def dense(cfg, query, k, source_filter=None):
        calls.append(("dense", k, query))
        return dense_rows

    def bm25(cfg, query, k, source_filter=None):
        calls.append(("bm25", k, query))
        return bm25_rows

    monkeypatch.setattr("folio.retrieve._dense_search", dense)
    monkeypatch.setattr("folio.retrieve._bm25_search", bm25)


def _meta(source_id: str = "s", rel: str = "a.md") -> dict:
    return {"source_id": source_id, "rel_path": rel, "heading_path": "H"}


# ---------------------------------------------------------------------------
# Fusion math vs the published thresholds
# ---------------------------------------------------------------------------


def test_default_rrf_best_score_always_beats_crag_and_floor():
    """Any non-empty top-20 list scores above both published thresholds.

    Rank 1 is 1/61 ≈ 0.0164. The weakest single-list rank inside the default
    window (rank 20) is 1/80 = 0.0125. crag_min_best_score is 0.012 and
    confidence_floor is 0.008, so neither can fire on a normal hit list.
    """
    ids = [f"d{i}" for i in range(20)]
    fused = rrf_fuse([ids, []], k=60)
    best = max(fused.values())
    worst = min(fused.values())
    assert best == pytest.approx(1 / 61)
    assert worst == pytest.approx(1 / 80)
    assert best > 0.012
    assert worst > 0.012
    assert worst > 0.008


def test_crag_does_not_widen_when_dense_similarity_is_zero(monkeypatch):
    """Docstring: widen when the best fusion score is weak.

    Dense similarity is computed in _dense_search and then discarded. Fusion
    uses rank only, so a completely off-topic neighbor at rank 1 still looks
    'strong' (1/61 > 0.012) and the widen path never runs.
    """
    calls: list = []
    rows = [
        (f"d{i}", f"unrelated text {i}", _meta(), 0.0) for i in range(5)
    ]
    _stub_searches(monkeypatch, rows, [], calls)
    hits = retrieve(_retrieval_cfg(), "a very specific unseen question")
    dense_ks = [k for kind, k, _ in calls if kind == "dense"]
    assert dense_ks == [20, 40], (
        "weak (zero-similarity) hits should trigger one widened search; "
        f"dense k sequence was {dense_ks}; returned {len(hits)} hits "
        f"with best score {hits[0].score if hits else None}"
    )


def test_confidence_floor_does_not_drop_zero_similarity_hybrid_hits(monkeypatch):
    calls: list = []
    rows = [(f"d{i}", "text", _meta(), 0.0) for i in range(3)]
    _stub_searches(monkeypatch, rows, [], calls)
    hits = retrieve(_retrieval_cfg(), "query")
    assert hits, "floor removed every hit even though retrieval returned neighbors"
    assert all(h.score >= 0.008 for h in hits)
    assert hits[0].score == pytest.approx(1 / 61)


def test_rerank_logits_are_not_discarded_by_rrf_floor(monkeypatch):
    """Cross-encoder scores are logits, often negative for the right passage.

    confidence_floor is calibrated for RRF (~0.01), then applied to those
    logits, so --rerank can return nothing.
    """
    calls: list = []
    rows = [
        ("d0", "best passage", _meta(rel="best.md"), 0.2),
        ("d1", "other passage", _meta(rel="other.md"), 0.1),
        ("d2", "third passage", _meta(rel="third.md"), 0.05),
    ]
    _stub_searches(monkeypatch, rows, [], calls)

    class _FakeEncoder:
        def __init__(self, name):
            self.name = name

        def predict(self, pairs):
            # Best logit is negative, which is normal for this model family.
            return [-0.4, -2.0, -1.1][: len(pairs)]

    import types

    mod = types.ModuleType("sentence_transformers")
    mod.CrossEncoder = _FakeEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)

    hits = retrieve(_retrieval_cfg(), "query", use_rerank=True, final_k=2)
    assert hits, "rerank dropped every hit because logits were below the RRF floor"
    assert hits[0].rel_path == "best.md"
    assert hits[0].score == pytest.approx(-0.4)


def test_negative_final_k_does_not_return_a_tail_slice(monkeypatch):
    calls: list = []
    rows = [(f"d{i}", f"text {i}", _meta(), 0.5) for i in range(4)]
    _stub_searches(monkeypatch, rows, [], calls)
    hits = retrieve(_retrieval_cfg(confidence_floor=0.0), "query", final_k=-1)
    assert hits == []


def test_source_filter_can_see_matches_outside_the_global_candidate_window(monkeypatch):
    """--source and docs-suggest filter after a global top-k, not inside the source."""
    calls: list = []

    def dense(cfg, query, k, source_filter=None):
        calls.append(("dense", k, source_filter))
        if source_filter == "langchain-docs-nav":
            return [("target", "nav page", _meta("langchain-docs-nav", "nav.md"), 0.4)]
        return [("noise", "other page", _meta("other", "o.md"), 0.9)]

    def bm25(cfg, query, k, source_filter=None):
        calls.append(("bm25", k, source_filter))
        if source_filter == "langchain-docs-nav":
            return [("target", "nav page", _meta("langchain-docs-nav", "nav.md"), 2.0)]
        return [("noise", "other page", _meta("other", "o.md"), 5.0)]

    monkeypatch.setattr("folio.retrieve._dense_search", dense)
    monkeypatch.setattr("folio.retrieve._bm25_search", bm25)
    hits = retrieve(
        _retrieval_cfg(dense_k=5, bm25_k=5, final_k=5, crag_min_best_score=0.0),
        "langgraph persistence",
        source_filter="langchain-docs-nav",
    )
    assert any(h.source_id == "langchain-docs-nav" for h in hits), (
        "source filter returned nothing because the nav chunk was outside "
        f"the global window; calls={calls}"
    )


def test_bm25_does_not_drop_documents_that_contain_the_query_term(tmp_path: Path):
    """Terms in every document get a negative idf floor, then scores <= 0 are removed."""
    from folio.store import bm25_index_path

    # Three identical docs: every term has negative idf, so the epsilon floor
    # is negative and get_scores() is negative. _bm25_search keeps only > 0.
    payload = {
        "ids": ["a", "b", "c"],
        "texts": ["alpha beta", "alpha beta", "alpha beta"],
        "metadatas": [_meta() for _ in range(3)],
    }
    path = tmp_path / "index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_bm25(path)
    scores = loaded["bm25"].get_scores(["alpha"])
    assert (scores > 0).any(), (
        f"BM25 assigned non-positive scores {scores.tolist()} to documents that "
        "contain the query term, so retrieval will drop them"
    )


def test_whitespace_only_corpus_does_not_crash_bm25(tmp_path: Path):
    path = tmp_path / "index.json"
    path.write_text(
        json.dumps({"ids": ["a"], "texts": ["   "], "metadatas": [_meta()]}),
        encoding="utf-8",
    )
    loaded = load_bm25(path)
    loaded["bm25"].get_scores(["hello"])  # avgdl == 0 divides by zero today


def test_empty_bm25_catalog_does_not_crash_on_underscore_query(tmp_path: Path):
    """Empty texts are replaced with a dummy '_' document that is not in ids."""
    from folio.retrieve import _bm25_search

    path = tmp_path / "index.json"
    path.write_text(
        json.dumps({"ids": [], "texts": [], "metadatas": []}),
        encoding="utf-8",
    )
    cfg = {"paths": {"bm25_path": path}, "retrieval": {}}
    hits = _bm25_search(cfg, "_", k=5)
    assert hits == []


# ---------------------------------------------------------------------------
# Budget claim: printed output is capped at ~budget tokens (4 chars/token)
# ---------------------------------------------------------------------------


def test_realistic_prose_print_stays_inside_default_character_budget():
    hits = [
        _hit(
            chunk_id=str(i),
            text=("word " * 250),
            score=0.03 - i * 0.001,
            rel_path=f"lessons/mod/{i}.md",
            heading_path="Module > Section",
        )
        for i in range(8)
    ]
    limit = chars_for_token_budget(2000)
    text = apply_text_budget(format_context(hits, max_chars=limit), token_budget=2000)
    assert len(text) <= limit
    payload = hits_to_json_payload(hits, token_budget=2000)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert len(blob) <= limit


def test_format_context_counts_the_whole_block_against_the_budget():
    hits = [
        _hit(chunk_id="1", text="hello", rel_path="x" * 200, heading_path="h" * 80)
    ]
    out = format_context(hits, max_chars=40)
    assert len(out) <= 40, f"header bypassed the cap ({len(out)} chars):\n{out[:180]}"


def test_tiny_token_budget_is_actually_respected():
    text = apply_text_budget("hello world " * 50, token_budget=1)
    limit = chars_for_token_budget(1)
    assert len(text) <= limit, f"printed {len(text)} chars for a 1-token budget: {text!r}"


def test_json_payload_stays_inside_budget_when_metadata_is_long():
    hits = [_hit(chunk_id="id", text="t" * 500, rel_path="r" * 500, heading_path="H")]
    limit = chars_for_token_budget(100)
    payload = hits_to_json_payload(hits, token_budget=100)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    assert len(blob) <= limit, f"json payload is {len(blob)} chars; budget is {limit}"


def test_truncated_context_tells_the_caller_it_was_cut():
    hits = [
        _hit(chunk_id=str(i), text=("alpha " * 80), rel_path=f"{i}.md", heading_path="H")
        for i in range(4)
    ]
    limit = 400
    raw = format_context(hits, max_chars=limit)
    shown = apply_text_budget(raw, token_budget=limit // 4)
    cut = "[2]" not in shown or len(shown) < sum(len(h.text) for h in hits)
    assert not cut or "truncated" in shown, (
        "output dropped citations without the truncation note the helper promises"
    )


def test_separator_between_citations_is_part_of_the_budget():
    hits = [
        _hit(chunk_id=str(i), text="word", rel_path="a.md", heading_path="H")
        for i in range(4)
    ]
    # Exactly the sum of the blocks format_context is willing to accept.
    one = format_context(hits[:1], max_chars=10_000)
    two = format_context(hits[:2], max_chars=10_000)
    block = len(one)
    # two blocks plus the joiner "\n---\n"
    cap = block * 2
    out = format_context(hits, max_chars=cap)
    assert len(out) <= cap, (
        f"joiners were not counted: len={len(out)} cap={cap} one={block} two={len(two)}"
    )


# ---------------------------------------------------------------------------
# Chunking: information loss, fences, and the overlap hang
# ---------------------------------------------------------------------------


def test_short_factual_sections_stay_in_the_index():
    text = textwrap.dedent(
        """
        # FAQ

        ## Reset
        Run folio-reset-token immediately.

        ## Background
        This background section explains capacity planning for the serving cluster
        and why the on-call rotation keeps a written handover. It is long enough
        to pass the default minimum character filter on its own.
        """
    ).strip()
    chunks = chunk_markdown(
        text,
        source_id="s",
        rel_path="faq.md",
        max_chars=1200,
        overlap_chars=150,
        min_chars=80,
    )
    assert any("folio-reset-token" in c.text for c in chunks), (
        "the only copy of the reset instruction was dropped; headings kept: "
        + ", ".join(c.heading_path for c in chunks)
    )


def test_headings_inside_code_fences_are_not_structure():
    lines = [
        "# Ops",
        "",
        "Rotate the production zebra-token carefully after every incident review meeting.",
        "",
        "```python",
        "## fake-heading",
        "print('this is example code, not a section')",
        "value = 'still inside the fence and long enough to pass min_chars easily'",
        "```",
        "",
        "## Backups",
        "This section describes database backup windows and restore drills for the primary cluster.",
    ]
    chunks = chunk_markdown(
        "\n".join(lines),
        source_id="s",
        rel_path="ops.md",
        max_chars=1200,
        overlap_chars=40,
        min_chars=40,
    )
    headings = [c.heading_path for c in chunks]
    assert not any("fake-heading" in h for h in headings), headings


def test_overlap_equal_to_window_finishes():
    """overlap >= max_chars must not spin. The child is killed if it does."""
    script = textwrap.dedent(
        """
        from folio.chunk import _split_oversized
        parts = _split_oversized("abcdefghij" * 5, max_chars=8, overlap=8)
        print(len(parts))
        """
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(Path(__file__).resolve().parents[1]),
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"_split_oversized did not return within 1.5s (overlap >= max_chars): {exc}")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().isdigit()


def test_overlap_just_below_window_still_terminates():
    parts = _split_oversized("abcdefghij" * 4, max_chars=10, overlap=9)
    assert parts
    assert "".join(parts)  # progress happened


def test_chunk_overlap_stitch_can_exceed_max_chars_but_covers_the_paragraph():
    paragraph = " ".join(f"w{i}" for i in range(200))
    parts = _split_oversized(paragraph, max_chars=80, overlap=20)
    assert parts
    # The long-paragraph splitter is allowed to overlap, but every token of the
    # source paragraph should still appear in some part.
    for token in paragraph.split():
        assert any(token in part for part in parts)


# ---------------------------------------------------------------------------
# skill-from structure
# ---------------------------------------------------------------------------


def test_duplicate_chapter_slugs_keep_both_bodies(tmp_path: Path):
    book = tmp_path / "notes.md"
    book.write_text(
        "## Hello World\n\n"
        "Alpha body is long enough to be its own chapter about retrieval budgets.\n\n"
        "## hello-world\n\n"
        "Beta body is long enough to be its own chapter about model registries.\n",
        encoding="utf-8",
    )
    project = tmp_path / "proj"
    project.mkdir()
    assert skill_from(book, project=True, project_dir=project, platforms="agents") == 0
    bodies = [
        p.read_text(encoding="utf-8")
        for p in (project / ".agents" / "skills" / "notes" / "chapters").glob("*.md")
    ]
    blob = "\n".join(bodies)
    assert "Alpha body" in blob and "Beta body" in blob
    assert len(bodies) >= 2


def test_hash_one_chapters_become_separate_files(tmp_path: Path):
    book = tmp_path / "book.md"
    book.write_text(
        "# Retrieval\n\n"
        + ("Chapter one discusses retrieval budgets and fusion in enough detail to stand alone. " * 3)
        + "\n\n# Registry\n\n"
        + ("Chapter two discusses model registries and shadow deployment in enough detail. " * 3),
        encoding="utf-8",
    )
    project = tmp_path / "proj"
    project.mkdir()
    assert skill_from(book, project=True, project_dir=project, platforms="agents") == 0
    files = list((project / ".agents" / "skills" / "book" / "chapters").glob("*.md"))
    assert len(files) >= 2, [p.name for p in files]


def test_sidebar_parent_link_is_kept_when_it_also_has_children():
    nodes = [
        {
            "title": "Parent",
            "url": "https://docs.langchain.com/parent",
            "children": [
                {
                    "type": "link",
                    "title": "Child",
                    "url": "https://docs.langchain.com/child",
                }
            ],
        }
    ]
    urls = [url for _, url in _walk_sidebar_links(nodes)]
    assert "https://docs.langchain.com/parent" in urls
    assert "https://docs.langchain.com/child" in urls


# ---------------------------------------------------------------------------
# Install / migrate
# ---------------------------------------------------------------------------


def test_uninstall_removes_assistant_markdown(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    assert install(platform="cursor", project=True, project_dir=project) == 0
    assert install(platform="claude", project=True, project_dir=project) == 0
    assert uninstall(platform="all", project=True, project_dir=project) == 0
    agents = project / "AGENTS.md"
    claude = project / "CLAUDE.md"
    assert not agents.exists() or "# folio" not in agents.read_text(encoding="utf-8")
    assert not claude.exists() or "# folio" not in claude.read_text(encoding="utf-8")


def test_uninstall_does_remove_skill_files(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    assert install(platform="cursor", project=True, project_dir=project) == 0
    assert uninstall(platform="cursor", project=True, project_dir=project) == 0
    assert not (project / ".cursor" / "rules" / "folio.mdc").exists()
    assert not (project / ".agents" / "skills" / "folio").exists()


def test_cursor_install_without_project_uses_the_home_directory(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setattr("folio.install.Path.home", lambda: home)
    assert install(platform="cursor", project=False, project_dir=project) == 0
    assert (home / ".cursor" / "rules" / "folio.mdc").exists()
    assert not (project / ".cursor" / "rules" / "folio.mdc").exists()


def test_migrate_does_not_delete_unrelated_files_in_the_index_root(tmp_path: Path, monkeypatch):
    dest = tmp_path / "folio-home"
    dest.mkdir()
    (dest / "user-notes.txt").write_text("keep me", encoding="utf-8")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FOLIO_HOME", str(dest))
    migrate_legacy_index(legacy)
    assert (dest / "user-notes.txt").exists()


# ---------------------------------------------------------------------------
# Pack safety that should hold, and replace that is not atomic
# ---------------------------------------------------------------------------


def _pack(path: Path, extra: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for name in ("chroma", "bm25"):
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tar.addfile(info)
        for name, content in (
            ("manifest.json", b"{}\n"),
            ("config.yaml", b"collection:\n  name: x\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        for info, payload in extra:
            tar.addfile(info, io.BytesIO(payload) if payload is not None else None)


def test_tar_parent_traversal_is_not_extracted(tmp_path: Path):
    archive = tmp_path / "evil.tar.gz"
    data = b"PWNED\n"
    info = tarfile.TarInfo("../ESCAPED.txt")
    info.size = len(data)
    _pack(archive, [(info, data)])
    dest = tmp_path / "out"
    dest.mkdir()
    try:
        _extract_pack(archive, dest)
    except RuntimeError:
        pass
    assert not (tmp_path / "ESCAPED.txt").exists()
    assert not (dest / "ESCAPED.txt").exists()


def test_tar_nested_traversal_is_rejected(tmp_path: Path):
    archive = tmp_path / "evil.tar.gz"
    data = b"PWNED\n"
    info = tarfile.TarInfo("chroma/../../ESCAPED.txt")
    info.size = len(data)
    _pack(archive, [(info, data)])
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(RuntimeError):
        _extract_pack(archive, dest)
    assert not (tmp_path / "ESCAPED.txt").exists()


def test_tar_symlink_member_is_rejected(tmp_path: Path):
    archive = tmp_path / "link.tar.gz"
    info = tarfile.TarInfo("bm25/index.json")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    _pack(archive, [(info, None)])
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(RuntimeError):
        _extract_pack(archive, dest)


def test_legacy_pickle_cannot_run_a_gadget(tmp_path: Path):
    marker = tmp_path / "pwned.txt"
    pkl = tmp_path / "index.pkl"

    class _Boom:
        def __reduce__(self):
            return (eval, (f"open({str(marker)!r}, 'w').write('PWNED')",))

    with pkl.open("wb") as handle:
        pickle.dump(_Boom(), handle)
    with pytest.raises(pickle.UnpicklingError):
        load_bm25(pkl)
    assert not marker.exists()


def test_failed_pack_replace_keeps_the_previous_index(tmp_path: Path, monkeypatch):
    archive = tmp_path / "ok.tar.gz"
    _pack(archive, [])
    out = tmp_path / "folio"
    chroma = out / "chroma"
    chroma.mkdir(parents=True)
    (chroma / "marker.txt").write_text("previous-index", encoding="utf-8")
    (out / "bm25").mkdir()
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    (out / "config.yaml").write_text("x: 1\n", encoding="utf-8")

    real_copytree = __import__("folio.distribute", fromlist=["shutil"]).shutil.copytree

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr("folio.distribute.shutil.copytree", boom)
    with pytest.raises(RuntimeError):
        _extract_into_out(archive, out)
    assert (chroma / "marker.txt").exists(), "old chroma was deleted before the new copy succeeded"
    # silence unused in case the patch style changes
    assert real_copytree


def test_bm25_json_helper_matches_public_path(tmp_path: Path):
    """Guard against the private helper drifting from the path users are told about."""
    legacy = tmp_path / "index.pkl"
    assert _bm25_json_path(legacy) == legacy.with_suffix(".json")
