"""Skill data model.

Skills are local directories containing a ``SKILL.md`` file and optional
``skill.toml`` manifest. The markdown body is loaded on demand (progressive
disclosure tier-2); the manifest can declare command allowlists consumed by
``run_skill_command``.

Inspired by Anthropic's Claude Skills / agentskills.io progressive disclosure:

- Tier 1: ``name`` + ``description`` (always available, used by ``skills_list``).
- Tier 2: ``SKILL.md`` body (loaded by ``skill_view``).
- Tier 3: ``references/``, ``templates/``, ``scripts/``, ``assets/`` files
  (loaded by ``skill_view`` with ``file_path=...``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


SUPPORTING_DIRS: tuple[str, ...] = ("references", "templates", "scripts", "assets")


@dataclass(frozen=True)
class SkillCommand:
    name: str
    description: str = ""
    command: str | None = None
    argv: tuple[str, ...] = ()
    cwd: str | None = None
    timeout_seconds: int | None = None

    def has_executable(self) -> bool:
        return bool(self.command or self.argv)


@dataclass(frozen=True)
class Skill:
    name: str
    path: Path
    source_path: Path
    description: str = ""
    body: str = ""
    commands: tuple[SkillCommand, ...] = field(default_factory=tuple)
    platforms: tuple[str, ...] = ()
    license: str = ""
    version: str = ""
    prompt_max_chars: int = 6000

    def command_map(self) -> dict[str, SkillCommand]:
        return {cmd.name: cmd for cmd in self.commands}

    def prompt_body(self, *, max_chars: int | None = None) -> str:
        if max_chars is None:
            max_chars = self.prompt_max_chars
        text = self.body.strip()
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n\n[skill truncated]"
        return text

    def supporting_files(self) -> list[Path]:
        """List every regular file under references/templates/scripts/assets.

        Returned paths are relative to ``self.path`` so they double as both
        the on-disk location and the ``file_path`` argument for ``skill_view``.
        """
        out: list[Path] = []
        for sub in SUPPORTING_DIRS:
            sub_dir = self.path / sub
            if not sub_dir.exists() or not sub_dir.is_dir():
                continue
            for f in sorted(sub_dir.rglob("*")):
                if f.is_file() and not f.is_symlink():
                    try:
                        out.append(f.relative_to(self.path))
                    except ValueError:
                        continue
        return out
