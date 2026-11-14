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
    assert "WebFetch" in text or "must **not**" in text.lower() or "Do **not** WebFetch" in text

    assert install(platform="cursor", project=True, project_dir=project) == 0  # idempotent

    skill = project / ".agents" / "skills" / "folio" / "SKILL.md"
    assert skill.exists()
    agents = project / "AGENTS.md"
    assert agents.exists() and "# folio" in agents.read_text(encoding="utf-8")

    assert uninstall(platform="cursor", project=True, project_dir=project) == 0
    assert not rule.exists()
    assert not skill.exists()


def test_unknown_platform_errors(tmp_path: Path):
    assert install(platform="nope", project_dir=tmp_path) == 2
