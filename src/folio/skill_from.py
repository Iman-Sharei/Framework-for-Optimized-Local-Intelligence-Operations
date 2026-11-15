"""Turn a local book/doc into an on-demand agent skill (SKILL.md + chapters)."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from .add_doc import _load_as_markdown, _slug
from .term import path_link

_CHAPTER_MAX_CHARS = 3500
_SKILL_INDEX_MAX = 40


def _split_chapters(markdown: str, *, stem: str) -> list[tuple[str, str]]:
    """Split markdown into (slug, body) chapters by ## headings, else by size."""
    text = markdown.strip()
    if not text:
        return []

    parts = re.split(r"(?m)^(## .+)$", text)
    chapters: list[tuple[str, str]] = []
    from_headings = False
    if len(parts) >= 3:
        from_headings = True
        # parts[0] = preamble before first ##; then heading, body, heading, body...
        preamble = parts[0].strip()
        i = 1
        while i + 1 < len(parts):
            heading = parts[i].strip()
            body = parts[i + 1].strip()
            title = re.sub(r"^##\s+", "", heading).strip() or f"section-{len(chapters)+1}"
            block = f"{heading}\n\n{body}".strip()
            if preamble and not chapters:
                block = f"{preamble}\n\n{block}".strip()
                preamble = ""
            chapters.append((_slug(title) or f"ch{len(chapters)+1:02d}", block))
            i += 2
        if preamble and not chapters:
            chapters.append((_slug(stem), preamble))
    else:
        # No ## headings — pack into sized slices.
        raw = text
        idx = 0
        while raw:
            idx += 1
            chunk = raw[:_CHAPTER_MAX_CHARS]
            cut = chunk.rfind("\n\n")
            if cut > _CHAPTER_MAX_CHARS // 2:
                chunk = chunk[:cut]
            raw = raw[len(chunk) :].lstrip()
            chapters.append((f"ch{idx:02d}", chunk))

    if from_headings:
        return chapters

    # Size-sliced mode: merge tiny trailing scraps into the previous chapter.
    merged: list[tuple[str, str]] = []
    for slug, body in chapters:
        if merged and len(body) < 200:
            prev_slug, prev_body = merged[-1]
            merged[-1] = (prev_slug, f"{prev_body}\n\n{body}".strip())
        else:
            merged.append((slug, body))
    return merged


def _skill_markdown(
    *,
    name: str,
    title: str,
    origin: str,
    chapters: list[tuple[str, str]],
) -> str:
    lines = [
        "---",
        f"name: {name}",
        (
            f'description: "On-demand notes from {title}. '
            'Load the matching chapters/ file when the user asks about this material."'
        ),
        "---",
        "",
        f"# /{name}",
        "",
        f"Structured skill distilled from `{origin}`.",
        "Do **not** dump every chapter into context — open only the files you need.",
        "",
        "## How to use",
        "",
        f"1. When the user asks about topics from **{title}**, read this `SKILL.md` first.",
        "2. Pick the matching row in the chapter index below.",
        "3. Open that `chapters/…` file and answer from it (quote lightly; prefer decisions).",
        "4. If nothing matches, say the skill does not cover that topic.",
        "",
        "## Chapter index",
        "",
        "| File | Topic |",
        "|---|---|",
    ]
    for slug, body in chapters[:_SKILL_INDEX_MAX]:
        first = next(
            (ln.strip("# ").strip() for ln in body.splitlines() if ln.strip()),
            slug,
        )
        topic = first[:80].replace("|", "/")
        lines.append(f"| `chapters/{slug}.md` | {topic} |")
    if len(chapters) > _SKILL_INDEX_MAX:
        lines.append(f"| … | +{len(chapters) - _SKILL_INDEX_MAX} more under `chapters/` |")
    lines.append("")
    return "\n".join(lines)


def _dest_roots(
    *,
    name: str,
    project: bool,
    project_dir: Path,
    platforms: list[str],
) -> list[Path]:
    roots: list[Path] = []
    base = project_dir if project else Path.home()
    mapping = {
        "agents": base / ".agents" / "skills" / name,
        "claude": base / ".claude" / "skills" / name,
        "codex": base / ".codex" / "skills" / name,
    }
    for p in platforms:
        if p in mapping:
            roots.append(mapping[p])
    # De-dupe while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def skill_from(
    path: Path,
    *,
    name: str | None = None,
    project: bool = True,
    project_dir: Path | None = None,
    platforms: str = "agents",
) -> int:
    """Convert a local PDF/markdown/text file into agent skill folders."""
    path = path.expanduser().resolve()
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    project_dir = (project_dir or Path(".")).resolve()
    skill_name = _slug(name or path.stem)
    if not skill_name:
        print("error: could not derive a skill name", file=sys.stderr)
        return 2

    plat_raw = platforms.lower().strip()
    if plat_raw in {"all", "*"}:
        plat_list = ["agents", "claude", "codex"]
    else:
        allowed = {"agents", "claude", "codex"}
        plat_list = []
        for p in (x.strip() for x in plat_raw.split(",") if x.strip()):
            if p in {"skills", "cursor"}:
                p = "agents"
            if p not in allowed:
                print(
                    f"error: unknown platform '{p}'. "
                    "Use agents|claude|codex|all (comma-separated ok).",
                    file=sys.stderr,
                )
                return 2
            plat_list.append(p)
        plat_list = plat_list or ["agents"]

    try:
        md = _load_as_markdown(path)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    chapters = _split_chapters(md, stem=skill_name)
    if not chapters:
        print("error: no content extracted", file=sys.stderr)
        return 2

    skill_body = _skill_markdown(
        name=skill_name,
        title=path.stem,
        origin=path.name,
        chapters=chapters,
    )

    roots = _dest_roots(
        name=skill_name,
        project=project,
        project_dir=project_dir,
        platforms=plat_list,
    )
    if not roots:
        print("error: no destination platforms resolved", file=sys.stderr)
        return 2

    for root in roots:
        if root.exists():
            shutil.rmtree(root)
        chapters_dir = root / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (root / "SKILL.md").write_text(skill_body, encoding="utf-8")
        for slug, body in chapters:
            (chapters_dir / f"{slug}.md").write_text(body.strip() + "\n", encoding="utf-8")
        print(
            f"✨ skill `{skill_name}` → {path_link(root)} "
            f"({len(chapters)} chapter file(s))"
        )

    print()
    print("📌 Try in your agent:")
    print(f'  ask about a topic from {path.name} — it should open chapters/ on demand')
    print(f"  skill name: /{skill_name}")
    return 0
