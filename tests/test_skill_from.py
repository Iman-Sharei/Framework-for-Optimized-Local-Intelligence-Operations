from __future__ import annotations

from pathlib import Path

from folio.cli import main
from folio.skill_from import skill_from


def test_skill_from_markdown_writes_chapters(tmp_path: Path):
    book = tmp_path / "design-notes.md"
    book.write_text(
        "# Design notes\n\n"
        "## Retrieval\n\nUse hybrid search when recall matters.\n\n"
        "## Rerank\n\nRerank only the top candidates.\n",
        encoding="utf-8",
    )
    project = tmp_path / "proj"
    project.mkdir()

    assert (
        skill_from(
            book,
            name="design-notes",
            project=True,
            project_dir=project,
            platforms="agents,claude",
        )
        == 0
    )

    agents = project / ".agents" / "skills" / "design-notes"
    claude = project / ".claude" / "skills" / "design-notes"
    assert (agents / "SKILL.md").exists()
    assert (claude / "SKILL.md").exists()
    skill = (agents / "SKILL.md").read_text(encoding="utf-8")
    assert "Chapter index" in skill
    assert "chapters/" in skill
    chapter_files = list((agents / "chapters").glob("*.md"))
    assert len(chapter_files) >= 2


def test_skill_from_platform_all(tmp_path: Path):
    book = tmp_path / "runbook.md"
    book.write_text("## Start\n\nBoot the service.\n\n## Stop\n\nDrain traffic.\n", encoding="utf-8")
    project = tmp_path / "proj"
    project.mkdir()
    assert skill_from(book, project=True, project_dir=project, platforms="all") == 0
    assert (project / ".agents" / "skills" / "runbook" / "SKILL.md").exists()
    assert (project / ".claude" / "skills" / "runbook" / "SKILL.md").exists()
    assert (project / ".codex" / "skills" / "runbook" / "SKILL.md").exists()


def test_cli_skill_from(tmp_path: Path):
    book = tmp_path / "notes.md"
    book.write_text("## A\n\nalpha\n\n## B\n\nbeta\n", encoding="utf-8")
    project = tmp_path / "proj"
    project.mkdir()
    code = main(
        [
            "skill-from",
            str(book),
            "--name",
            "notes",
            "--dir",
            str(project),
            "--platform",
            "agents",
        ]
    )
    assert code == 0
    assert (project / ".agents" / "skills" / "notes" / "SKILL.md").exists()
