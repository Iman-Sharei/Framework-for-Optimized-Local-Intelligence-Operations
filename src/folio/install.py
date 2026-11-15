from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .config import package_dir

TOOL = "folio"
_CURSOR_RULE_PATH = Path(".cursor") / "rules" / "folio.mdc"
_AGENTS_MD_MARKER = "# folio"

_CURSOR_RULE = """\
---
description: folio curated AI / MLOps / infra / LangChain docs-nav RAG
alwaysApply: true
---

This project uses the standalone **folio** tool (shared curricula + docs-nav RAG on the machine).

**MANDATORY: Before deciding or implementing AI engineering / MLOps / RAG / agent / GPU / AI infra / related AI architecture, you MUST run folio first:**
- `folio query "<question>" --json` — hybrid retrieve grounded citations
- `folio status` — check whether the shared index exists (`~/.folio` or `$FOLIO_HOME`)

This applies to YOU and to every subagent you spawn for those topics (any mode: Agent, Plan, Debug, Ask). Do not skip folio because the topic “feels familiar”.

### LangChain / LangGraph / LangSmith / Deep Agents (coding or Q&A)

When the user asks about these **or** you are about to write code with them:
1. `folio query "<task> LangChain LangGraph docs" --source langchain-docs-nav -k 12 --json`
2. Collect official `https://docs.langchain.com/...` URLs from the hits
3. **WebFetch/open those pages** before implementing (current API truth)
4. Optionally also `folio query "<architecture question>" --json` (no `--source`) for curricula/handbook trade-offs
5. Cite which doc URLs you checked

**Token rule:** Do **not** WebFetch/clone curated curriculum GitHub repos — they are indexed; use `folio query` only.
**Exception:** `docs.langchain.com` URLs from `langchain-docs-nav` hits may be opened/WebFetched.

Only skip folio when:
1. The task is ordinary product/codebase work with no AI / MLOps / AI-infra / LangChain decision
2. The shared index is missing — then instruct `folio bootstrap` (or `folio build`) once

After `folio query`, ground choices in returned `source_id` + `rel_path` + `heading_path` citations (and opened LangChain doc URLs when applicable).
"""

_CLAUDE_REGISTRATION = """\
# folio

Standalone tool: local hybrid RAG for AI engineering / MLOps / AI infra / LangChain docs-nav.
Shared index: `~/.folio/` (override with `FOLIO_HOME`).

Before AI-eng / MLOps / RAG / agent / AI-infra work, run:
`folio query "<question>" --json`

For LangChain / LangGraph / LangSmith coding:
`folio query "<task> docs" --source langchain-docs-nav -k 12 --json`
then open the returned `docs.langchain.com` URLs before writing code.

Do **not** fetch curriculum GitHub repos into chat — query the local index only.
Exception: official docs.langchain.com URLs from langchain-docs-nav may be fetched.

Skill: `.claude/skills/folio/SKILL.md` / `.agents/skills/folio/SKILL.md`.
If index missing: `folio bootstrap` or `folio build`. Refresh: `folio update`.
"""

_PLATFORM_SKILL_DST = {
    "claude": Path(".claude") / "skills" / "folio" / "SKILL.md",
    "codex": Path(".codex") / "skills" / "folio" / "SKILL.md",
    "agents": Path(".agents") / "skills" / "folio" / "SKILL.md",
    "cursor": None,
}


def _skill_src() -> Path:
    src = package_dir() / "assets" / "SKILL.md"
    if not src.exists():
        raise FileNotFoundError(f"Missing packaged skill: {src}")
    return src


def _write_skill(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = _skill_src().read_text(encoding="utf-8")
    if dst.exists() and dst.read_text(encoding="utf-8") == text:
        print(f"folio skill already configured at {dst.resolve()} (no change)")
        return
    action = "updated" if dst.exists() else "written"
    dst.write_text(text, encoding="utf-8")
    print(f"folio skill {action} at {dst.resolve()}")


def _cursor_install(project_dir: Path) -> None:
    rule_path = project_dir / _CURSOR_RULE_PATH
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    if rule_path.exists() and rule_path.read_text(encoding="utf-8") == _CURSOR_RULE:
        print(f"folio Cursor rule at {rule_path.resolve()} already configured (no change)")
        return
    action = "updated" if rule_path.exists() else "written"
    rule_path.write_text(_CURSOR_RULE, encoding="utf-8")
    print(f"folio Cursor rule {action} at {rule_path.resolve()}")
    print("Cursor will retrieve-first via folio for AI / MLOps / infra tasks.")
    print("Shared index: ~/.folio — run `folio bootstrap` or `folio build` if missing.")


def _cursor_uninstall(project_dir: Path) -> None:
    rule_path = project_dir / _CURSOR_RULE_PATH
    # Also remove legacy aieng rule if present
    legacy = project_dir / ".cursor" / "rules" / "aieng.mdc"
    removed_any = False
    for path in (rule_path, legacy):
        if path.exists():
            path.unlink()
            print(f"removed {path.resolve()}")
            removed_any = True
    if not removed_any:
        print("No folio Cursor rule found — nothing to do")


def _replace_or_append_section(content: str, marker: str, section: str) -> str:
    lines = content.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == marker:
            start = i
            break
    if start is None:
        # Drop legacy aieng section if present
        content = _strip_marker_section(content, "# aieng")
        body = content.rstrip()
        if body:
            return body + "\n\n" + section.strip() + "\n"
        return section.strip() + "\n"

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("# "):
            end = j
            break
    new_lines = lines[:start] + section.strip().splitlines() + ([""] if end < len(lines) else []) + lines[end:]
    return "\n".join(new_lines).rstrip() + "\n"


def _strip_marker_section(content: str, marker: str) -> str:
    lines = content.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == marker:
            start = i
            break
    if start is None:
        return content
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("# "):
            end = j
            break
    new_lines = lines[:start] + lines[end:]
    return "\n".join(new_lines).rstrip() + ("\n" if new_lines else "")


def _claude_md_register(project_dir: Path | None, user_global: bool) -> None:
    if user_global:
        target = Path.home() / ".claude" / "CLAUDE.md"
    else:
        assert project_dir is not None
        target = project_dir / "CLAUDE.md"
    try:
        existed = target.exists()
        content = target.read_text(encoding="utf-8") if existed else ""
        content = _strip_marker_section(content, "# aieng")
        new_content = _replace_or_append_section(content, _AGENTS_MD_MARKER, _CLAUDE_REGISTRATION)
        if existed and new_content == content:
            print(f"folio already registered in {target} (no change)")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        print(f"folio registered in {target.resolve()}")
    except OSError as exc:
        print(f"skipped CLAUDE.md registration: {exc}", file=sys.stderr)


def _agents_md_register(project_dir: Path) -> None:
    target = project_dir / "AGENTS.md"
    try:
        existed = target.exists()
        content = target.read_text(encoding="utf-8") if existed else ""
        content = _strip_marker_section(content, "# aieng")
        new_content = _replace_or_append_section(content, _AGENTS_MD_MARKER, _CLAUDE_REGISTRATION)
        if existed and new_content == content:
            print(f"folio already registered in {target} (no change)")
            return
        target.write_text(new_content, encoding="utf-8")
        print(f"folio registered in {target.resolve()}")
    except OSError as exc:
        print(f"skipped AGENTS.md registration: {exc}", file=sys.stderr)


def _skill_dst_for(platform: str, *, project: bool, project_dir: Path) -> Path | None:
    rel = _PLATFORM_SKILL_DST.get(platform)
    if rel is None:
        return None
    if project:
        return project_dir / rel
    return Path.home() / rel


def install(
    *,
    platform: str = "cursor",
    project: bool = False,
    project_dir: Path | None = None,
) -> int:
    project_dir = (project_dir or Path(".")).resolve()
    platform = platform.lower().strip()
    if platform == "skills":
        platform = "agents"

    known = {"cursor", "claude", "codex", "agents"}
    if platform not in known:
        print(
            f"error: unknown platform '{platform}'. Choose from: {', '.join(sorted(known))}",
            file=sys.stderr,
        )
        return 2

    if platform == "cursor":
        _cursor_install(project_dir)
        if project:
            _write_skill(project_dir / ".agents" / "skills" / "folio" / "SKILL.md")
            # remove legacy skill dir
            legacy_skill = project_dir / ".agents" / "skills" / "aieng"
            if legacy_skill.exists():
                shutil.rmtree(legacy_skill)
                print(f"removed legacy {legacy_skill}")
            _agents_md_register(project_dir)
        return 0

    dst = _skill_dst_for(platform, project=project, project_dir=project_dir)
    assert dst is not None
    _write_skill(dst)

    if platform == "claude":
        _claude_md_register(project_dir if project else None, user_global=not project)
    if platform == "agents" and project:
        _agents_md_register(project_dir)

    print()
    print("Next: `folio bootstrap` (or `folio build`), then `folio query \"...\"`.")
    return 0


def uninstall(
    *,
    platform: str | None = None,
    project: bool = False,
    project_dir: Path | None = None,
    purge: bool = False,
) -> int:
    project_dir = (project_dir or Path(".")).resolve()
    platforms = (
        [platform.lower()]
        if platform
        else ["cursor", "claude", "codex", "agents"]
    )
    for p in platforms:
        if p == "skills":
            p = "agents"
        if p == "cursor":
            _cursor_uninstall(project_dir)
            for name in ("folio", "aieng"):
                skill = project_dir / ".agents" / "skills" / name
                if skill.exists():
                    shutil.rmtree(skill)
                    print(f"removed {skill}")
            continue
        dst = _skill_dst_for(p, project=project, project_dir=project_dir)
        if dst and dst.exists():
            skill_dir = dst.parent
            shutil.rmtree(skill_dir)
            print(f"removed {skill_dir}")
        elif not project and p in _PLATFORM_SKILL_DST and _PLATFORM_SKILL_DST[p]:
            user_dst = Path.home() / _PLATFORM_SKILL_DST[p]  # type: ignore[operator]
            if user_dst.exists():
                shutil.rmtree(user_dst.parent)
                print(f"removed {user_dst.parent}")

    if purge:
        for name in ("folio-out", "aieng-out"):
            out = project_dir / name
            if out.exists():
                shutil.rmtree(out)
                print(f"purged {out}")
    return 0
