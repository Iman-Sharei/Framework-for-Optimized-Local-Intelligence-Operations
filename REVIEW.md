# Folio review

The bugs below were fixed on `cursor/review-accuracy-bafa` after this audit. The sections are the original findings. The strict xfails that encoded them now pass.



Reviewed `folio-cli` 0.4.6 (local hybrid RAG: Chroma ONNX MiniLM + BM25 + reciprocal rank fusion). Production code was not changed. New tests are in `tests/test_review_logic.py` and `tests/test_review_accuracy.py`. Cases that violate the README or the function's own contract are `pytest.mark.xfail(strict=True)` so the suite stays green and a real fix turns them into failures.

## What the project claims

The README and the packaged skill describe a local index (`~/.folio` or `$FOLIO_HOME`) that an assistant queries on demand:

- `folio query` runs hybrid retrieval (BM25 + dense vectors, fused with RRF, with a CRAG-style widen when the best score is weak) and prints at most a budget, default **2000 tokens**. Retrieval is claimed to rank the **full** index; the budget only caps the print.
- `folio bootstrap` / `folio update` install a published pack. `folio add` / `folio pack` are maintainer tools.
- `folio install` wires Cursor, Claude, Codex, and a generic agents layout. Without `--project`, the help text says files go in the home directory.
- `folio skill-from` turns a local PDF, Markdown, or text file into a skill with a chapter index.
- `folio docs-suggest` prints `docs.langchain.com` URLs from the `langchain-docs-nav` source.
- Assistants are told not to name folio sources. Human query output omits `source_id`.

The retrieval docstring in `src/folio/retrieve.py` states the pipeline as: query → BM25 + dense → RRF → CRAG widen → optional cross-encoder → top-k.

## Setup and existing tests

- Python 3.12.3, `uv sync --locked`, then `uv run pytest`.
- Existing suite before these tests: **44 passed, 1 skipped, 4 integration passed**. The skip is `tests/test_langchain_workflow.py`, which needs a local index that contains `langchain-docs-nav`. This environment does not have the published pack, so that test does not measure docs-suggest quality.
- Existing tests check RRF ordering, a tiny curriculum round-trip, install smoke, and pack path checks. They do not check the published score thresholds, paraphrase quality, the print budget under edge sizes, or re-indexing.

This review's tests, on top of that suite: **15 passed, 25 xfailed**.

## Accuracy verdict

The hybrid path **does retrieve** on a 17-note fixture (9 labeled notes + 8 distractors) built with the same chunker, `BM25Okapi`, and Chroma MiniLM the CLI uses. It is not a score for the published curriculum pack, which was not downloaded.

| Question set | Hybrid R@1 | Hybrid R@3 | Hybrid R@8 | Hybrid MRR | Dense MRR | BM25 MRR |
|---|---:|---:|---:|---:|---:|---:|
| 8 questions that reuse the note's vocabulary | 8/8 | 8/8 | 8/8 | 1.000 | 1.000 | 1.000 |
| 8 looser paraphrases | 6/8 | 7/8 | 8/8 | 0.838 | 0.917 | 0.585 |

Chance recall@3 on 17 notes is about 0.18. A sentence copied from the fusion note was hybrid rank 1.

On the loose set, dense is the stronger channel. The evaluation question ("canned dataset" vs "what real people do") was **dense rank 1 and hybrid rank 5**, and the top hybrid hit was the budget note. RRF adds `1/(60+rank)` from each list. A distractor that places well in both lists outranks the dense winner when that winner is weak in BM25. Fusion helped some questions (the agent note was BM25 rank 4 and hybrid rank 1) and hurt this one.

Two limits on the "full index, 2000 tokens" claim:

- Each channel contributes only `dense_k` / `bm25_k` hits (both **20**). BM25 does score every chunk, then keeps 20. Dense is an ANN query for 20 neighbors. The fused list is that union, not a ranking of every chunk. The widen step that was supposed to grow the window never runs (bug 1).
- The budget is `max(budget, 1) * 4` characters (`chars_for_token_budget`). Ordinary prose stays inside 8000 characters, and 8000 characters of the sample technical prose is **1319** MiniLM wordpieces (under 2000). The same 8000 characters of code-like text is **2669** wordpieces (over 2000). A 1200-character chunk, the default `max_chars`, is 201 tokens of that prose (fits the 256-token MiniLM window), **317** tokens of short function words, and **402** tokens of the sample code (146 tokens never enter the vector). See bug 3.

A unique token `zygomorphic` is BM25 rank 1. The same token with a question mark is BM25 rank **none** and hybrid rank 1, because dense still matches the string. The lexical tokenizer is whitespace `split()` with no punctuation stripping; the dense channel often covers that up.

## Bugs

### 1. High — CRAG widen never runs, and the confidence floor is on the wrong scale

`src/folio/retrieve.py` lines 49–50, 91–97, and 132–133. Defaults in `src/folio/assets/config.yaml` lines 33–37: `rrf_k: 60`, `crag_min_best_score: 0.012`, `confidence_floor: 0.008`, `dense_k` / `bm25_k`: 20.

RRF's best score on any non-empty list is `1/61 ≈ 0.0164`. The worst score inside a top-20 list is `1/80 = 0.0125`. Both are above 0.012 and 0.008. Dense similarity is computed (`1 - distance`) and then discarded; the widen decision uses the rank score.

Repro: stub dense search to return neighbors with similarity 0 and BM25 to return nothing. `retrieve` calls dense once with `k=20` and returns 5 hits whose best score is 0.0164. The widened `k=40` call never happens. With real hits, every hybrid score is ≥ 0.0125, so the floor removes nothing.

`--rerank` then replaces those scores with cross-encoder logits and runs the same floor. A fake encoder that returns `[-0.4, -2.0, -1.1]` (a normal negative logit for the best passage) yields **no hits**.

Fix: decide widening from dense similarity, or set the RRF threshold above `1/(rrf_k+1)` if the intent is "the two lists did not agree." Apply `confidence_floor` only to RRF scores. Do not compare logits to 0.008.

### 2. High — `folio add` keeps the previous chunks

`src/folio/add_doc.py` lines 160–163, chunk ids in `src/folio/chunk.py` lines 125–130.

Chunk ids include a hash of the text. `_merge_bm25` only drops ids that appear in the new batch. Chroma `upsert` does the same. Editing a file therefore leaves the old passages in both indexes.

Repro: `add_local_document` a note that says `purplegiraffe`, then add the same source id again with `bluepenguin`. Both strings are still in the collection documents and in the BM25 texts.

Fix: before upsert, delete every chunk whose `source_id` (and path) matches the document being replaced, in Chroma and in the BM25 JSON.

### 3. High — dense search cannot see the tail of a long chunk

Chroma's MiniLM tokenizer truncates at 256 tokens (`onnx_mini_lm_l6_v2.py` in the installed package). Default `max_chars` is 1200 (`config.yaml` line 18).

Repro: build a string of 400 `gardenN` tokens plus `purplegiraffe` at the end. The embedding of the full string has cosine **1.0** with the embedding of the first 256 tokens. `purplegiraffe` sits in the discarded tail (about 3000 characters in that sample). A 1200-character code-like chunk is 402 tokens, so about 36% of its tokens never affect the vector. BM25 still sees the tail, which is why a rare exact token can survive. A paraphrase that only matches the tail will not.

Fix: size chunks in tokens (around 200, with overlap) or split so every token falls inside the first 256 of some chunk. Clamp `overlap_chars` so the stitched chunk cannot grow past the window (`_split_oversized` can prepend another `overlap` characters after the size split).

### 4. High — replacing an index deletes the old one first

`src/folio/distribute.py` lines 212–221.

`_extract_into_out` `rmtree`s each existing member (`chroma`, `bm25`, …) and then copies the new one. If `copytree` raises, the previous Chroma directory is already gone.

Repro: pack a valid archive, write `chroma/marker.txt`, monkeypatch `shutil.copytree` to raise. After the exception the marker file does not exist.

Fix: extract into a sibling directory and swap it into place (or copy over the new tree, and only delete the old tree after the copy returns).

### 5. Medium — short sections are dropped, and code-fence headings become sections

`src/folio/chunk.py` lines 8 and 119, and the heading scan at lines 92–113.

Default `min_chars` is 80. A FAQ entry `## Reset` / `Run folio-reset-token immediately.` produces no chunk that contains `folio-reset-token`. The long background section is kept. Anything under the cutoff is unretrievable.

The heading regex is not fence-aware. A `## fake-heading` line inside a Python fence becomes a real section (`Ops > fake-heading` in the probe).

Fix: merge a short section into its neighbor instead of dropping it. Track fenced code regions and ignore `#` lines inside them.

### 6. Medium — `--source` and docs-suggest only see the global top 20

`src/folio/retrieve.py` lines 84–89 and 112–113. `cmd_docs_suggest` calls `retrieve(..., source_filter="langchain-docs-nav")`.

The filter runs after fusion of the global dense and BM25 lists. There is no Chroma `where` clause and no BM25 pre-filter. A nav chunk that is relevant inside its source but outside the global top 20 is dropped. The probe recorded calls `[('dense', 5), ('bm25', 5)]` and returned no `langchain-docs-nav` hit, even though a larger `k` would have returned it.

Fix: pass the source into both searches (Chroma `where={"source_id": ...}`, and skip other ids before the BM25 top-k cut), then fuse.

### 7. Medium — `skill-from` loses chapters

`src/folio/skill_from.py` line 23 (split only on `##`) and lines 216–217 (write `chapters/{slug}.md`).

- `## Hello World` and `## hello-world` both slug to `hello-world`. The command reports 2 chapter files. One file remains, and it contains only the second body.
- A book that uses `# Retrieval` and `# Registry` becomes a single `ch01.md`. PDF input is less exposed because `pdftotext` output is turned into `## Page N` headings.

Fix: split on ATX headings of levels 1–2 (or every level), and suffix duplicate slugs (`hello-world-2.md`).

### 8. Medium — install and uninstall do not match the help text

`src/folio/cli.py` lines 361–364, `src/folio/install.py` `uninstall` (lines 314–356) and `_install_one` for `cursor` (the rule is always written under `project_dir`).

- `folio install` without `--project` still writes `.cursor/rules/folio.mdc` into the given project directory. Claude and Codex without `--project` do use the home directory.
- `folio uninstall --platform all --project` removes the Cursor rule and the skill directory. `AGENTS.md` and `CLAUDE.md` still contain the `# folio` section, so the assistant instructions stay.

Fix: when `--project` is absent, write the Cursor rule under `Path.home()` as well. On uninstall, strip the `# folio` section with the same marker logic install uses to insert it.

### 9. Medium — legacy migration deletes unrelated files

`src/folio/distribute.py` lines 270–276.

If `FOLIO_HOME` has no `manifest.json` and a legacy directory does, migration `rmtree`s the destination and moves the legacy tree onto it. A `user-notes.txt` sitting in that destination is deleted. `folio query` and `folio status` call this on the default home.

Fix: require an explicit flag, and refuse to delete a destination that contains anything other than a partial folio index.

### 10. Medium — `overlap >= max_chars` does not return

`src/folio/chunk.py` lines 52–58.

When a paragraph is longer than `max_chars` and `overlap >= max_chars`, `start = end - overlap` does not advance. The loop appends forever. A subprocess running `_split_oversized("abcdefghij" * 5, max_chars=8, overlap=8)` was still running after 1.5s and had to be killed. Default config (1200 / 150) does not hit this. A hand-edited `config.yaml` does, and ingest/`folio add` hang.

Fix: if `overlap >= max_chars`, raise `ValueError`, or clamp overlap to `max_chars - 1` before the loop.

### 11. Medium — BM25 drops matches when scores go negative, and punctuation splits tokens

`src/folio/retrieve.py` line 63, tokenization at line 61 and `src/folio/store.py` line 183.

`rank_bm25` floors negative idf at `epsilon * average_idf`. If every term is in more than half the documents, that floor is itself negative. Three copies of `"alpha beta"` scored the query `alpha` at **-0.486**, and `scores[i] > 0` then dropped every document that contains the term. On a large varied corpus the average idf stays positive and this stays quiet. On a small or repetitive index, lexical search returns nothing.

A whitespace-only text (`"   "`) makes `BM25Okapi` raise `ZeroDivisionError` while computing idf (`rank_bm25.py` line 101, called from `load_bm25`). The chunker strips and drops short blanks, so normal ingest avoids it. A crafted BM25 JSON does not.

`zygomorphic?` does not match the indexed token `zygomorphic`. Hybrid still returned the note, via dense search.

Fix: use one tokenizer for index and query that lowercases and strips punctuation. Keep a hit when the query term occurs, including a non-positive idf floor. Skip empty token lists instead of constructing `BM25Okapi` on them.

### 12. Low — the print budget leaks on small or awkward hits

Ordinary 8-hit prose stays within 8000 characters (covered by a passing test). The holes:

- `format_context` (`retrieve.py` lines 225–229) always emits the header. A 200-character path and an 80-character heading produced **313** characters against `max_chars=40`.
- `apply_text_budget` (lines 178–180) appends a note that is longer than a 1-token budget. `--budget 1` prints a **53**-character note and none of the hit text. `max(token_budget, 1)` also turns zero and negative budgets into 4 characters.
- `hits_to_json_payload` (lines 202–205) keeps the first object when it is the only one, even if it is over the limit. A 500-character `rel_path` produced **833** JSON characters against a 400-character budget.
- The `"\n---\n"` joiners are not added into `used`. Two citations measured 89 characters against a cap of 84.
- Because `format_context` is already called with `budget * 4` (`cli.py` lines 135–138), the truncation note is not added on the normal path. The caller is not told that later hits were removed.

Fix: count headers, joiners, and the note toward the same limit; drop or trim the first record until the encoded output fits; skip the note when it does not fit.

### 13. Low — other contract misses

- `ingest(rebuild=False)` (`src/folio/ingest.py` line 41) calls `reset_collection` on both branches. A sentinel chunk upserted beforehand is gone after `ingest(..., rebuild=False)`. No CLI path passes `False` today. Make the false branch upsert without deleting the collection.
- `retrieve(..., final_k=-1)` uses `hits[:-1]` and returns every stub hit except the last. Reject `final_k < 1`.
- `_walk_sidebar_links` (`src/folio/add_doc.py` lines 66–69) uses `elif children`, so a nav node that has both a URL and children keeps only the children. `https://docs.langchain.com/parent` was dropped; the child URL remained. Record the node URL, then walk children.

## What held up

- RRF math: a document near the top of both lists outscores a document in only one list. With the shipped `k=60`, rank 1 is 0.0164 and rank 20 is 0.0125.
- On the 17-note fixture, shared-vocabulary questions were 8/8 at rank 1 for hybrid, dense, and BM25. A copied sentence was rank 1.
- Querying an empty Chroma collection does not raise.
- Pack extraction rejects `../ESCAPED.txt`, `chroma/../../ESCAPED.txt`, and a symlink member. A pickle gadget whose `__reduce__` calls `eval` raises `UnpicklingError` and does not write the marker file. `bootstrap` rejects a `file://` URL (existing test).
- `uninstall` does remove `.cursor/rules/folio.mdc` and `.agents/skills/folio`.
- Human `format_context` output does not include `source_id`. JSON output does, on purpose.

Residual supply-chain note, not exploited here: the downloaded pack's `config.yaml` is what the next `folio update` reads for `release.index_url`, and the tarball is not checked against a published digest. Pin a hash, and keep using the URL shipped inside the wheel rather than the URL inside the pack.
