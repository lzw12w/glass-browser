"""Task-list tool — a port of Claude Code's ``TodoWrite``.

Multi-step UI flows (open home → find button → tap → verify → record path)
benefit from an explicit, visible plan the same way a coding session does.
This tool lets the model publish and update a checklist that the CLI and Web
Console render as ticked items.

Design note — the tool is *stateless by contract*: every call carries the
FULL list and REPLACES the previous one wholesale (mirrors Claude Code's
TodoWrite). The canonical current list lives on ``session._todos`` (a plain
list-of-dicts); the tool_result deliberately does NOT echo the full list,
only the summary counts + an ack. Two reasons:

1. The list is already in the assistant-side ``tool_use.input`` block, so
   returning it in ``tool_result`` is a 2x round-trip of the same payload.
2. Under Tier 2 compaction the model summarizes and drops old assistant
   blocks; ``session._todos`` is the durable snapshot the agent loop exposes
   as a request-local user-role reminder, so the plan survives compaction
   without elevating model/user-derived task text into the system prompt.
"""
from __future__ import annotations

from .base import Action, ActionResult

_STATUSES = ("pending", "in_progress", "completed")


class TodoWriteAction(Action):
    name = "todo_write"
    # Pure bookkeeping — it never touches the device UI, so it must NOT feed
    # the knowledge observer or trigger a post-action hierarchy fetch.
    mutates_ui = False
    # Not idempotent: the whole point is to overwrite the prior list, and two
    # identical writes are a legitimate no-op the caller may repeat.
    idempotent = False
    description = (
        "Create and manage a structured task list for the current session. "
        "Use it for multi-step work (roughly 3+ steps or several stages): "
        "publish a plan up front, then keep it current as you go. Each call "
        "sends the ENTIRE list and REPLACES the previous one — include every "
        "task every time, not just the changed ones.\n\n"
        "Each task has three fields:\n"
        "- content: imperative form of the task (\"Tap the purchase button\").\n"
        "- activeForm: present-continuous form shown while it runs "
        "(\"Tapping the purchase button\").\n"
        "- status: pending | in_progress | completed.\n\n"
        "Rules: exactly ONE task should be in_progress at a time; mark a task "
        "completed the moment it is fully done (don't batch); keep a task "
        "in_progress if it is blocked or only partially done. Skip this tool "
        "for a single trivial step — the overhead isn't worth it."
    )
    schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "The complete task list, replacing the current one.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Imperative form of the task.",
                        },
                        "status": {
                            "type": "string",
                            "enum": list(_STATUSES),
                        },
                        "activeForm": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "Present-continuous form shown while the task "
                                "is in progress."
                            ),
                        },
                    },
                    "required": ["content", "status", "activeForm"],
                },
            },
        },
        "required": ["todos"],
    }

    def _execute(self, session, *, todos):
        if not isinstance(todos, list):
            return ActionResult(ok=False, error={
                "error": "E_TODO_INVALID",
                "message": "todos must be a list of task objects",
            })

        cleaned: list[dict] = []
        in_progress = 0
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                return ActionResult(ok=False, error={
                    "error": "E_TODO_INVALID",
                    "message": f"todos[{i}] must be an object",
                })
            content = item.get("content")
            active = item.get("activeForm")
            status = item.get("status")
            if not isinstance(content, str) or not content.strip():
                return ActionResult(ok=False, error={
                    "error": "E_TODO_INVALID",
                    "message": f"todos[{i}].content must be a non-empty string",
                })
            if not isinstance(active, str) or not active.strip():
                return ActionResult(ok=False, error={
                    "error": "E_TODO_INVALID",
                    "message": f"todos[{i}].activeForm must be a non-empty string",
                })
            if not isinstance(status, str) or status not in _STATUSES:
                return ActionResult(ok=False, error={
                    "error": "E_TODO_INVALID",
                    "message": (
                        f"todos[{i}].status must be one of "
                        f"{', '.join(_STATUSES)}"
                    ),
                })
            if status == "in_progress":
                in_progress += 1
            cleaned.append({
                "content": content.strip(),
                "status": status,
                "activeForm": active.strip(),
            })

        # Mirror Claude Code's soft rule. 0 in_progress is fine (an all-done or
        # not-yet-started list); >1 means the model lost track of "the one
        # thing I'm doing now", which is worth rejecting so it re-plans.
        if in_progress > 1:
            return ActionResult(ok=False, error={
                "error": "E_TODO_INVALID",
                "message": (
                    "exactly ONE task may be in_progress at a time; "
                    f"got {in_progress}"
                ),
            })

        completed = sum(1 for t in cleaned if t["status"] == "completed")
        pending = sum(1 for t in cleaned if t["status"] == "pending")

        # Claude Code parity: when every task is completed, clear the list.
        # Without this the "current plan" block would keep advertising a done
        # checklist round after round. An empty list also signals the reminder
        # injection to skip the block entirely.
        all_done = bool(cleaned) and completed == len(cleaned)
        snapshot: list[dict] = [] if all_done else cleaned

        # Store the durable snapshot on the session. Callers (agent loop, web
        # UI, tests) read from here — not from the tool_result — so this is
        # the single source of truth for "current plan".
        setattr(session, "_todos", snapshot)

        # NOTE: we deliberately do NOT include ``todos`` in the returned data.
        # The list is already in ``tool_use.input`` (assistant side) and on
        # ``session._todos`` (loop reads it back into every request). Echoing
        # it here would double the payload for no gain.
        return ActionResult(ok=True, data={
            "ack": "todos updated" if not all_done else "todos updated (all complete — list cleared)",
            "summary": {
                "total": len(cleaned),
                "completed": completed,
                "in_progress": in_progress,
                "pending": pending,
            },
        })


__all__ = ["TodoWriteAction"]
