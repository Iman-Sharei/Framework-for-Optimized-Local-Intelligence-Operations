from __future__ import annotations

from pathlib import Path

from folio.install import install, uninstall


def test_cursor_install_and_uninstall_writes_rule(tmp_path: Path):
    project = tmp_path / "proj"
    project.mkdir()
    assert install(platform="cursor", project=True, project_dir=project) == 0
    rule = project / ".cursor" / "rules" / "folio.mdc"
    assert rule.exists()
    text = rule.read_text(encoding="utf-8")
    assert "folio query" in text
    assert "docs-suggest" in text
    assert "--budget" in text

    assert install(platform="cursor", project=True, project_dir=project) == 0  # idempotent

    skill = project / ".agents" / "skills" / "folio" / "SKILL.md"
    assert skill.exists()
    agents = project / "AGENTS.md"
    assert agents.exists() and "# folio" in agents.read_text(encoding="utf-8")

    assert uninstall(platform="cursor", project=True, project_dir=project) == 0
    assert not rule.exists()
    assert not skill.exists()


def test_refresh_updates_existing_assistant_files_only(tmp_path: Path):
    from folio.install import refresh_installed_files

    project = tmp_path / "proj"
    skill = project / ".agents" / "skills" / "folio" / "SKILL.md"
    rule = project / ".cursor" / "rules" / "folio.mdc"
    skill.parent.mkdir(parents=True)
    rule.parent.mkdir(parents=True)
    skill.write_text("old skill", encoding="utf-8")
    rule.write_text("old rule", encoding="utf-8")

    assert refresh_installed_files(project) >= 2
    assert "folio query" in skill.read_text(encoding="utf-8")
    assert "folio query" in rule.read_text(encoding="utf-8")
    assert not (project / ".claude" / "skills" / "folio" / "SKILL.md").exists()


def test_unknown_platform_errors(tmp_path: Path):
    assert install(platform="nope", project_dir=tmp_path) == 2
