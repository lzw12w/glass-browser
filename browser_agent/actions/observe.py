"""Read-only observation actions: snapshot, screenshot, logs, wait_for."""
from __future__ import annotations

import time

from .base import Action, ActionResult


class BrowserSnapshotAction(Action):
    name = "browser_snapshot"
    description = (
        "Capture an LLM-friendly snapshot of the current page: a tree of "
        "visible salient elements. Interactive elements carry a `ref` (e.g. "
        "'e12') — pass THAT to click / fill / press_key. Refs are only valid "
        "until the next snapshot or page change. `_meta` carries url, title, "
        "node counts and tab count. Call this before acting on a page you "
        "haven't observed yet, and again after the page changes."
    )
    idempotent = True
    schema = {"type": "object", "properties": {}}

    def _execute(self, session):
        return session.snapshot()


class ScreenshotAction(Action):
    name = "screenshot"
    description = (
        "Save a PNG screenshot of the current page into the run workdir and "
        "return its path. Use for visual evidence or when the user asks to "
        "see the page; the snapshot tree remains the source of truth for "
        "element targeting."
    )
    idempotent = True
    schema = {
        "type": "object",
        "properties": {
            "full_page": {
                "type": "boolean",
                "description": "Capture the full scrollable page instead of the viewport.",
            },
        },
    }

    def _execute(self, session, *, full_page: bool = False):
        path = session.screenshot(full_page=bool(full_page))
        return ActionResult(ok=True, data={"path": str(path)}, artifacts=[str(path)])


class ConsoleLogAction(Action):
    name = "console_logs"
    description = (
        "Return recent browser console messages (buffered since page load). "
        "Optionally filter by level: log | info | warning | error."
    )
    idempotent = True
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "level": {"type": "string", "enum": ["log", "info", "warning", "error", "debug"]},
        },
    }

    def _execute(self, session, *, limit: int = 50, level: str | None = None):
        entries = session.console_logs(limit=limit, level=level)
        return {"count": len(entries), "entries": entries}


class NetworkLogAction(Action):
    name = "network_requests"
    description = (
        "Return recent network responses observed on the active page "
        "(method, url, status, resource_type). Optionally filter by URL "
        "substring."
    )
    idempotent = True
    schema = {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            "url_contains": {"type": "string"},
        },
    }

    def _execute(self, session, *, limit: int = 50, url_contains: str | None = None):
        entries = session.network_requests(limit=limit, url_contains=url_contains)
        return {"count": len(entries), "entries": entries}


class WaitForAction(Action):
    name = "wait_for"
    description = (
        "Wait until text or a CSS selector reaches the desired state "
        "(visible or hidden) on the active page. Prefer this over guessing "
        "with repeated snapshots when a load/transition is in flight. "
        "Returns matched=false instead of erroring when the timeout expires."
    )
    idempotent = True
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Visible text to wait for (substring match)."},
            "selector": {"type": "string", "description": "CSS selector to wait for."},
            "state": {"type": "string", "enum": ["visible", "hidden"], "default": "visible"},
            "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 30000, "default": 5000},
        },
    }

    def _execute(self, session, *, text: str | None = None,
                 selector: str | None = None, state: str = "visible",
                 timeout_ms: int = 5000):
        if not text and not selector:
            return ActionResult(ok=False, error={
                "error": "E_INVALID_ARG",
                "message": "wait_for needs `text` or `selector`",
            })
        page = session.page
        if selector:
            locator = page.locator(selector)
            evidence = f"selector={selector!r}"
        else:
            locator = page.get_by_text(text, exact=False)
            evidence = f"text={text!r}"
        start = time.time()
        try:
            locator.first.wait_for(state=state, timeout=timeout_ms)
            matched = True
        except Exception:
            matched = False
        return {
            "matched": matched,
            "state": state,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
            "evidence": evidence,
        }
