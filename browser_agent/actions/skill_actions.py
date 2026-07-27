"""Actions for the local skill runtime (progressive disclosure).

Two tools surface the local skill registry to the model:

- ``skills_list`` — tier 1. Returns one metadata dict per loaded skill. Cheap.
- ``skill_view`` — tier 2 / tier 3. Loads SKILL.md body, or a relative file
  under references/templates/scripts/assets/. Path-traversal safe.

Both delegate to the SkillManager attached to the session by ``Agent.__init__``.
"""
from __future__ import annotations

from .base import Action, ActionResult


class ListSkillsAction(Action):
    name = "skills_list"
    description = (
        "List local skills loaded by the agent runtime. Returns metadata "
        "only (tier-1 progressive disclosure): name, slug, description, "
        "command names, has_supporting_files. Use this to discover what "
        "skills exist; load full instructions with skill_view(name=...). "
        "Do not use run_shell for skill discovery."
    )
    idempotent = True
    schema = {
        "type": "object",
        "properties": {},
    }

    def _execute(self, session):
        manager = getattr(session, "_skill_manager", None)
        if manager is None:
            return ActionResult(ok=False, error={
                "error": "E_SKILLS_UNAVAILABLE",
                "message": "no skill manager is attached to this session",
            })
        index = manager.metadata_index()
        return {
            "count": len(index),
            "skills": index,
            "hint": (
                "Call skill_view(name=...) to load a skill's full instructions, "
                "or skill_view(name=..., file_path='references/x.md') for a "
                "supporting file."
            ),
        }


class SkillViewAction(Action):
    name = "skill_view"
    description = (
        "Load a skill's full content (tier-2/-3 progressive disclosure). "
        "Without file_path, returns SKILL.md body plus the supporting-file "
        "manifest. With file_path (e.g. 'references/api.md', "
        "'scripts/run.sh'), returns the contents of that file inside the "
        "skill directory. Path traversal is rejected."
    )
    idempotent = True
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Skill name as listed by skills_list.",
            },
            "file_path": {
                "type": "string",
                "description": (
                    "Optional relative path under the skill directory "
                    "(must be inside references/, templates/, scripts/, "
                    "or assets/, and must not contain '..')."
                ),
            },
        },
        "required": ["name"],
    }

    def _execute(self, session, *, name: str, file_path: str | None = None):
        manager = getattr(session, "_skill_manager", None)
        if manager is None:
            return ActionResult(ok=False, error={
                "error": "E_SKILLS_UNAVAILABLE",
                "message": "no skill manager is attached to this session",
            })
        result = manager.view_skill(name, file_path=file_path)
        if not result.get("ok"):
            return ActionResult(ok=False, error={
                "error": result.get("error", "E_SKILL_VIEW"),
                "message": result.get("message", "skill_view failed"),
                **{k: v for k, v in result.items() if k not in ("ok", "error", "message")},
            })
        return result
