from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .config import package_dir
from .term import path_link, url_link

TOOL = "folio"
_PYPI_URL = "https://pypi.org/project/folio-cli/"
_REPO_URL = "https://github.com/Iman-Sharei/folio"
_RELEASES_URL = "https://github.com/Iman-Sharei/folio/releases/latest"
_README_URL = "https://github.com/Iman-Sharei/folio#readme"
_CURSOR_RULE_PATH = Path(".cursor") / "rules" / "folio.mdc"
_AGENTS_MD_MARKER = "# folio"

_CURSOR_RULE = """\
---
description: folio for AI / MLOps / LangChain decisions
alwaysApply: true
---

Before AI-eng / MLOps / RAG / agents / GPU / AI-infra decisions (you and subagents), run folio. Skip only for ordinary product code.

1. Read the relevant project code, then ask **one concrete question** (the decision + the constraint from that code). Do not paste files into the query.
   `folio query "<decision>" --budget 2000`
2. LangChain / LangGraph / LangSmith code: `folio docs-suggest "<task>"`, then open the printed `docs.langchain.com` URLs before writing code.
3. Never tell the user folio source names. Do not clone curriculum repos into chat. No index yet: `folio bootstrap`.
"""

_CLAUDE_REGISTRATION = """\
# folio

Local hybrid RAG for AI engineering / MLOps / AI infra / LangChain docs (`~/.folio` or `$FOLIO_HOME`).

Before those decisions: read the relevant code, then
`folio query "<one concrete decision + constraint>" --budget 2000`

LangChain / LangGraph / LangSmith code:
`folio docs-suggest "<task>"` then open the printed docs.langchain.com URLs.

Never name folio sources to the user. Do not fetch curriculum repos into chat.
No index: `folio bootstrap`. Skill: `.claude/skills/folio/SKILL.md`.
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
        print(f"✅ folio skill already configured at {path_link(dst)} (no change)")
        return
    action = "updated" if dst.exists() else "written"
    dst.write_text(text, encoding="utf-8")
    print(f"✨ folio skill {action} at {path_link(dst)}")


def _cursor_install(project_dir: Path) -> None:
    rule_path = project_dir / _CURSOR_RULE_PATH
    rule_path.parent.mkdir(parents=True, exist_ok=True)
    if rule_path.exists() and rule_path.read_text(encoding="utf-8") == _CURSOR_RULE:
        print(f"✅ folio Cursor rule at {path_link(rule_path)} already configured (no change)")
        return
    action = "updated" if rule_path.exists() else "written"
    rule_path.write_text(_CURSOR_RULE, encoding="utf-8")
    print(f"✨ folio Cursor rule {action} at {path_link(rule_path)}")
    print("Cursor will retrieve-first via folio for AI / MLOps / infra tasks.")
    print(
        "Shared index: ~/.folio — run `folio bootstrap` or `folio build` if missing. "
        f"Pack: {url_link(_RELEASES_URL, 'GitHub Releases')}."
    )


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
            print(f"✅ folio already registered in {path_link(target)} (no change)")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_content, encoding="utf-8")
        print(f"✨ folio registered in {path_link(target)}")
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
            print(f"✅ folio already registered in {path_link(target)} (no change)")
            return
        target.write_text(new_content, encoding="utf-8")
        print(f"✨ folio registered in {path_link(target)}")
    except OSError as exc:
        print(f"skipped AGENTS.md registration: {exc}", file=sys.stderr)


def _print_install_next_steps() -> None:
    print()
    print("📌 Next steps")
    print(f"  1. folio bootstrap   ← download pack from {url_link(_RELEASES_URL, 'releases')}")
    print('  2. folio query "…"    ← ask a concrete AI / MLOps question')
    print(
        f"  📖 {url_link(_README_URL, 'README')} · "
        f"📦 {url_link(_PYPI_URL, 'PyPI')} · "
        f"🐙 {url_link(_REPO_URL, 'GitHub')}"
    )


def _skill_dst_for(platform: str, *, project: bool, project_dir: Path) -> Path | None:
    rel = _PLATFORM_SKILL_DST.get(platform)
    if rel is None:
        return None
    if project:
        return project_dir / rel
    return Path.home() / rel


def refresh_installed_files(project_dir: Path | None = None) -> int:
    """Rewrite folio assistant files that already exist. Does not install into new projects."""
    project_dir = (project_dir or Path(".")).resolve()
    refreshed = 0

    rule = project_dir / _CURSOR_RULE_PATH
    if rule.exists():
        _cursor_install(project_dir)
        refreshed += 1

    for platform, rel in _PLATFORM_SKILL_DST.items():
        if rel is None:
            continue
        for base in (project_dir, Path.home()):
            dst = base / rel
            if dst.is_file():
                _write_skill(dst)
                refreshed += 1

    for target in (project_dir / "CLAUDE.md", Path.home() / ".claude" / "CLAUDE.md"):
        if target.is_file() and _AGENTS_MD_MARKER in target.read_text(encoding="utf-8"):
            _claude_md_register(
                None if target.parent == Path.home() / ".claude" else project_dir,
                user_global=target.parent == Path.home() / ".claude",
            )
            refreshed += 1

    agents = project_dir / "AGENTS.md"
    if agents.is_file() and _AGENTS_MD_MARKER in agents.read_text(encoding="utf-8"):
        _agents_md_register(project_dir)
        refreshed += 1

    if refreshed:
        print(f"[update] refreshed {refreshed} assistant file(s)")
    else:
        print("[update] no folio assistant files in this project or home directory")
    return refreshed


def _install_one(
    platform: str,
    *,
    project: bool,
    project_dir: Path,
) -> int:
    if platform == "cursor":
        # Without --project the help text installs into the home directory.
        _cursor_install(project_dir if project else Path.home())
        if project:
            _write_skill(project_dir / ".agents" / "skills" / "folio" / "SKILL.md")
            # remove legacy skill dir
            legacy_skill = project_dir / ".agents" / "skills" / "aieng"
            if legacy_skill.exists():
                shutil.rmtree(legacy_skill)
                print(f"removed legacy {path_link(legacy_skill)}")
            _agents_md_register(project_dir)
        return 0

    dst = _skill_dst_for(platform, project=project, project_dir=project_dir)
    assert dst is not None
    _write_skill(dst)

    if platform == "claude":
        _claude_md_register(project_dir if project else None, user_global=not project)
    if platform == "agents" and project:
        _agents_md_register(project_dir)
    return 0


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

    known = {"cursor", "claude", "codex", "agents", "all"}
    if platform not in known:
        print(
            f"error: unknown platform '{platform}'. "
            f"Choose from: cursor, claude, codex, agents, all",
            file=sys.stderr,
        )
        return 2

    targets = (
        ["cursor", "claude", "codex", "agents"]
        if platform == "all"
        else [platform]
    )
    # Avoid double-writing .agents skill when both cursor --project and agents run.
    wrote_agents_skill = False
    for target in targets:
        if target == "agents" and "cursor" in targets and project:
            # cursor --project already writes .agents/skills/folio + AGENTS.md
            wrote_agents_skill = True
            continue
        if target == "agents" and wrote_agents_skill:
            continue
        code = _install_one(target, project=project, project_dir=project_dir)
        if code != 0:
            return code
        if target == "cursor" and project:
            wrote_agents_skill = True

    _print_install_next_steps()
    return 0


def uninstall(
    *,
    platform: str | None = None,
    project: bool = False,
    project_dir: Path | None = None,
    purge: bool = False,
) -> int:
    project_dir = (project_dir or Path(".")).resolve()
    if platform and platform.lower().strip() in {"all", "*"}:
        platforms = ["cursor", "claude", "codex", "agents"]
    elif platform:
        platforms = [platform.lower().strip()]
    else:
        platforms = ["cursor", "claude", "codex", "agents"]
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
    _strip_folio_markdown(project_dir, project=project)
    return 0


def _strip_folio_markdown(project_dir: Path, *, project: bool) -> None:
    """Remove the folio section install wrote into AGENTS.md / CLAUDE.md."""
    targets = [project_dir / "AGENTS.md", project_dir / "CLAUDE.md"]
    if not project:
        targets.append(Path.home() / ".claude" / "CLAUDE.md")
    for path in targets:
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if _AGENTS_MD_MARKER not in content:
            continue
        path.write_text(_strip_marker_section(content, _AGENTS_MD_MARKER), encoding="utf-8")
