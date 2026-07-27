"""Skill runtime — progressive disclosure registry.

The manager exposes three operations the agent uses at runtime:

- :meth:`list_skills` — tier-1, returns metadata for every loaded skill.
- :meth:`view_skill`  — tier-2/-3, returns SKILL.md body or a supporting
  file. Path-traversal safe.
- :meth:`build_activation_message` — formats a skill into a ``user`` message
  payload for slash-command activation (mirrors hermes-agent's
  ``_build_skill_message``).

Trigger-keyword auto-selection is intentionally absent: the model sees every
skill's metadata in the system prompt and pulls full content on demand via
``skill_view``. Explicit ``/skill-name`` slash commands push the same payload
through :meth:`build_activation_message` so the system-prompt prefix cache
stays untouched across sessions.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .loader import (
    file_path_within_skill,
    load_skills,
    skill_lookup_path_error,
)
from .model import Skill, SkillCommand


# Patterns for sanitising skill names into hyphen-separated slash-command slugs.
_SLUG_INVALID_CHARS = re.compile(r"[^a-z0-9-]")
_SLUG_MULTI_HYPHEN = re.compile(r"-{2,}")


def _slugify(name: str) -> str:
    slug = name.lower().replace(" ", "-").replace("_", "-")
    slug = _SLUG_INVALID_CHARS.sub("", slug)
    slug = _SLUG_MULTI_HYPHEN.sub("-", slug).strip("-")
    return slug


class SkillManager:
    def __init__(
        self,
        *,
        roots: Iterable[Path] | None = None,
        enabled: set[str] | None = None,
        disabled: set[str] | None = None,
        max_chars_per_skill: int = 6000,
    ):
        self.roots = list(roots) if roots is not None else None
        self.enabled = enabled
        self.disabled = disabled
        self.max_chars_per_skill = max(0, int(max_chars_per_skill))
        self._skills = load_skills(self.roots, enabled=enabled, disabled=disabled)

    @classmethod
    def from_config(cls, cfg, *, forced_enabled: Iterable[str] | None = None) -> "SkillManager":
        enabled = getattr(cfg, "enabled_skills", None)
        if forced_enabled:
            merged = set(enabled or {"*"})
            if "*" not in merged:
                merged.update(forced_enabled)
            enabled = merged
        return cls(
            roots=getattr(cfg, "skill_roots", None),
            enabled=enabled,
            disabled=getattr(cfg, "disabled_skills", None),
            max_chars_per_skill=getattr(cfg, "skill_max_chars", 6000),
        )

    def reload(self) -> None:
        self._skills = load_skills(self.roots, enabled=self.enabled, disabled=self.disabled)

    # ---- tier 1 -------------------------------------------------------
    def list_skills(self) -> list[Skill]:
        return list(self._skills)

    def metadata_index(self) -> list[dict]:
        """Return one tier-1 metadata dict per loaded skill.

        This is the payload injected into the system prompt so the model
        can decide which skill (if any) to ``skill_view``. We deliberately
        keep it tiny: name + slug + short description + command names +
        ``has_supporting_files`` so the index scales linearly with skill
        count and stays well under any practical context budget.
        """
        out: list[dict] = []
        for skill in self._skills:
            out.append({
                "name": skill.name,
                "slug": _slugify(skill.name),
                "description": skill.description,
                "commands": [cmd.name for cmd in skill.commands],
                "has_supporting_files": bool(skill.supporting_files()),
                "platforms": list(skill.platforms),
            })
        return out

    def get(self, name: str) -> Skill | None:
        if skill_lookup_path_error(name):
            return None
        needle = name.strip().lower()
        for skill in self._skills:
            if skill.name.lower() == needle:
                return skill
        return None

    def get_command(self, skill_name: str, command_name: str) -> tuple[Skill, SkillCommand] | None:
        skill = self.get(skill_name)
        if skill is None:
            return None
        command = skill.command_map().get(command_name)
        if command is None:
            return None
        return skill, command

    # ---- tier 2 / tier 3 ---------------------------------------------
    def view_skill(self, name: str, *, file_path: str | None = None) -> dict:
        """Load a skill's body or one of its supporting files.

        Always returns a dict with ``ok`` set. On failure, the dict carries
        ``error`` / ``message``. Path-traversal attempts return a structured
        error rather than raising — the caller (an action) will turn that
        into ``E_SKILL_PATH``.
        """
        err = skill_lookup_path_error(name)
        if err:
            return {"ok": False, "error": "E_SKILL_PATH", "message": err}
        skill = self.get(name)
        if skill is None:
            return {
                "ok": False,
                "error": "E_UNKNOWN_SKILL",
                "message": f"no such skill: {name}",
                "available": [s.name for s in self._skills],
            }
        if file_path is None:
            return {
                "ok": True,
                "name": skill.name,
                "skill_dir": str(skill.path),
                "source_path": str(skill.source_path),
                "description": skill.description,
                "version": skill.version,
                "license": skill.license,
                "platforms": list(skill.platforms),
                "commands": [
                    {
                        "name": cmd.name,
                        "description": cmd.description,
                        "has_shell_command": bool(cmd.command),
                        "has_argv": bool(cmd.argv),
                    }
                    for cmd in skill.commands
                ],
                "supporting_files": [str(f) for f in skill.supporting_files()],
                "content": skill.body,
            }
        resolved = file_path_within_skill(skill.path, file_path)
        if resolved is None:
            return {
                "ok": False,
                "error": "E_SKILL_PATH",
                "message": (
                    f"file_path must be a relative file under {skill.path} "
                    f"(no traversal)"
                ),
            }
        try:
            text = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {
                "ok": False,
                "error": "E_BINARY_FILE",
                "message": f"refusing to read binary file {resolved}",
                "abs_path": str(resolved),
            }
        except OSError as e:
            return {
                "ok": False,
                "error": "E_READ_FAILED",
                "message": str(e),
                "abs_path": str(resolved),
            }
        return {
            "ok": True,
            "name": skill.name,
            "skill_dir": str(skill.path),
            "file_path": file_path,
            "abs_path": str(resolved),
            "content": text,
        }

    # ---- slash-command surface ----------------------------------------
    def slash_command_map(self) -> dict[str, Skill]:
        """Return ``/<slug> -> Skill`` for every loaded skill.

        Used by the CLI REPL to register skill activation under the same
        ``/<name>`` surface hermes-agent exposes. Slugs that collapse to
        empty (e.g. a skill named entirely from punctuation) are skipped.
        """
        out: dict[str, Skill] = {}
        for skill in self._skills:
            slug = _slugify(skill.name)
            if not slug:
                continue
            key = f"/{slug}"
            out.setdefault(key, skill)
        return out

    def build_activation_message(
        self,
        skill: Skill,
        *,
        user_instruction: str = "",
    ) -> str:
        """Format a skill into a ``user`` message body for slash activation.

        Mirrors hermes-agent's ``_build_skill_message``: the skill body is
        injected as a USER turn so it does not invalidate the system-prompt
        cache prefix on follow-up turns. We additionally ship the absolute
        skill directory so the agent can reference bundled scripts without
        an extra ``skill_view`` round-trip, and a one-line manifest of
        supporting files for tier-3 discovery.
        """
        body = skill.prompt_body(max_chars=self.max_chars_per_skill)
        parts: list[str] = [
            f"[Activating skill: {skill.name}]",
            "Treat the block below as operational instructions and reusable "
            "procedures. Use them only when relevant to the user's request, "
            "and prefer their specific workflow over generic exploration.",
            "",
            body,
            "",
            f"[Skill directory: {skill.path}]",
            "Resolve any relative paths in this skill (e.g. `scripts/foo.py`, "
            "`templates/config.yaml`) against that directory.",
        ]
        supporting = skill.supporting_files()
        if supporting:
            parts.append("")
            parts.append("[Supporting files — load with skill_view(name, file_path=...):]")
            for rel in supporting:
                parts.append(f"- {rel}  ->  {skill.path / rel}")
        if skill.commands:
            parts.append("")
            parts.append("[Skill commands — invoke via run_skill_command(skill, command):]")
            for cmd in skill.commands:
                desc = f" — {cmd.description}" if cmd.description else ""
                parts.append(f"- {cmd.name}{desc}")
        if user_instruction:
            parts.append("")
            parts.append(
                f"[User instruction alongside the skill invocation: {user_instruction}]"
            )
        return "\n".join(parts)
