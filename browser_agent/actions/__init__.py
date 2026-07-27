"""Action registry — maps tool name to Action class."""
from __future__ import annotations

from .base import Action, ActionResult
from .interact import (
    BackAction, ClickAction, FillAction, HoverAction, NavigateAction,
    PressKeyAction, ScrollAction, SelectOptionAction, TabsAction,
)
from .observe import (
    BrowserSnapshotAction, ConsoleLogAction, FindElementAction,
    NetworkLogAction, ReadTextAction, ScreenshotAction, WaitForAction,
)
from .shell import RunShellAction, RunSkillCommandAction
from .skill_actions import ListSkillsAction, SkillViewAction
from .todo import TodoWriteAction


_CORE_ACTIONS: list[type[Action]] = [
    BrowserSnapshotAction, FindElementAction, ReadTextAction, ScreenshotAction,
    NavigateAction, ClickAction, FillAction, SelectOptionAction, HoverAction,
    PressKeyAction, ScrollAction, BackAction, TabsAction,
    WaitForAction,
    ConsoleLogAction, NetworkLogAction,
    ListSkillsAction, SkillViewAction,
    TodoWriteAction,
    # ``run_skill_command`` is always exposed: the agent runs the command in
    # the chosen skill's directory so its bundled scripts resolve without a
    # separate opt-in. Same denylist / workspace_roots / env / timeout guard
    # rails as ``run_shell`` still apply.
    RunSkillCommandAction,
]
# Ad-hoc bash. Default OFF — opens a wider execution surface (any cwd under
# workspace_roots, no skill scoping). Enable via ``--allow-shell`` /
# ``BROWSER_AGENT_ENABLE_SHELL=1``.
_SHELL_ACTIONS: list[type[Action]] = [RunShellAction]
_ALL_ACTIONS: list[type[Action]] = _CORE_ACTIONS + _SHELL_ACTIONS

_REGISTRY: dict[str, type[Action]] = {a.name: a for a in _ALL_ACTIONS}

# Single source of truth for "which tools mutate UI state" — derived from
# ``Action.mutates_ui`` on every registered action so adding a new mutating
# action doesn't require touching downstream consumers.
MUTATING_TOOL_NAMES: frozenset[str] = frozenset(
    name for name, action_cls in _REGISTRY.items() if action_cls.mutates_ui
)


def get_action(name: str) -> Action:
    if name not in _REGISTRY:
        raise KeyError(f"unknown action: {name}")
    return _REGISTRY[name]()


def list_tool_schemas(*, include_shell: bool = False) -> list[dict]:
    actions: list[type[Action]] = list(_CORE_ACTIONS)
    if include_shell:
        actions.extend(_SHELL_ACTIONS)
    return [a.to_tool_schema() for a in actions]


def list_action_names() -> list[str]:
    return list(_REGISTRY.keys())


__all__ = ["Action", "ActionResult", "get_action",
           "list_tool_schemas", "list_action_names",
           "MUTATING_TOOL_NAMES"]
