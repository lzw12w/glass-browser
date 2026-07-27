"""UI-mutating actions: navigate, click, fill, select_option, hover,
press_key, scroll, back, tabs."""
from __future__ import annotations

from .base import Action, ActionResult


class NavigateAction(Action):
    name = "navigate"
    description = (
        "Open a URL in the active tab. Returns the final url, title and HTTP "
        "status. Old snapshot refs become invalid — take a fresh "
        "browser_snapshot before interacting with the new page."
    )
    mutates_ui = True
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Absolute URL (https://... or file://...)."},
        },
        "required": ["url"],
    }

    def _execute(self, session, *, url: str):
        result = session.navigate(url)
        # Navigation invalidates every ref from the previous page.
        session._known_refs = set()
        return result


class ClickAction(Action):
    name = "click"
    description = (
        "Click the element identified by `ref` from the latest "
        "browser_snapshot. On E_STALE_REF or E_TARGET_NOT_FOUND, call "
        "browser_snapshot again and retry with a fresh ref. Returns the "
        "clicked element plus whether the URL changed — if it did, snapshot "
        "the new page before further actions."
    )
    mutates_ui = True
    schema = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Element ref from browser_snapshot, e.g. 'e12'."},
        },
        "required": ["ref"],
    }

    def _execute(self, session, *, ref: str):
        element = session.describe_ref(ref)
        page = session.page
        url_before = page.url
        locator = session.resolve_ref(ref)
        locator.click(timeout=5000)
        session.settle()
        url_after = page.url
        data = {
            "element": element,
            "url": url_after,
            "url_changed": url_after != url_before,
        }
        if data["url_changed"]:
            # The old page's refs are gone; force a re-observe.
            session._known_refs = set()
            data["hint"] = "URL changed; take a fresh browser_snapshot before the next interaction"
        return data


class FillAction(Action):
    name = "fill"
    description = (
        "Replace the value of the input/textarea/contenteditable identified "
        "by `ref` with `text` (clears existing content first). Set "
        "submit=true to press Enter afterwards. Focus is handled internally."
    )
    mutates_ui = True
    schema = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Element ref from browser_snapshot."},
            "text": {"type": "string", "description": "The value to type."},
            "submit": {"type": "boolean", "description": "Press Enter after filling."},
        },
        "required": ["ref", "text"],
    }

    def _execute(self, session, *, ref: str, text: str, submit: bool = False):
        element = session.describe_ref(ref)
        locator = session.resolve_ref(ref)
        locator.fill(text, timeout=5000)
        if submit:
            locator.press("Enter")
            session.settle()
        return {
            "element": element,
            "value": text[:200],
            "submitted": bool(submit),
            "url": session.page.url,
        }


class SelectOptionAction(Action):
    name = "select_option"
    description = (
        "Choose an option in a <select> dropdown identified by `ref` from the "
        "latest browser_snapshot. Match by visible `label` or by `value` "
        "(the snapshot lists a select's options with their v/t). Use this "
        "instead of click for native dropdowns."
    )
    mutates_ui = True
    schema = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "The <select> element's ref."},
            "label": {"type": "string", "description": "Visible option text to select."},
            "value": {"type": "string", "description": "Option value attribute to select."},
        },
        "required": ["ref"],
    }

    def _execute(self, session, *, ref: str, label: str | None = None,
                 value: str | None = None):
        if not label and not value:
            return ActionResult(ok=False, error={
                "error": "E_INVALID_ARG",
                "message": "select_option needs `label` or `value`",
            })
        element = session.describe_ref(ref)
        locator = session.resolve_ref(ref)
        if value is not None:
            chosen = locator.select_option(value=value, timeout=5000)
        else:
            chosen = locator.select_option(label=label, timeout=5000)
        session.settle(timeout_ms=2000)
        return {"element": element, "selected": chosen}


class HoverAction(Action):
    name = "hover"
    description = (
        "Move the pointer over the element identified by `ref` — reveals "
        "hover menus / tooltips. Take a fresh browser_snapshot afterwards to "
        "see any content that appeared."
    )
    mutates_ui = True
    schema = {
        "type": "object",
        "properties": {
            "ref": {"type": "string", "description": "Element ref from browser_snapshot."},
        },
        "required": ["ref"],
    }

    def _execute(self, session, *, ref: str):
        element = session.describe_ref(ref)
        session.resolve_ref(ref).hover(timeout=5000)
        session.settle(timeout_ms=1500)
        return {"element": element}


class PressKeyAction(Action):
    name = "press_key"
    description = (
        "Press a keyboard key (Playwright key names: 'Enter', 'Escape', "
        "'Tab', 'ArrowDown', 'Control+a', ...). With `ref`, the key goes to "
        "that element; otherwise to the page."
    )
    mutates_ui = True
    identity_param_keys = ("key",)
    schema = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "ref": {"type": "string", "description": "Optional target element ref."},
        },
        "required": ["key"],
    }

    def _execute(self, session, *, key: str, ref: str | None = None):
        if ref:
            element = session.describe_ref(ref)
            session.resolve_ref(ref).press(key, timeout=5000)
        else:
            element = None
            session.page.keyboard.press(key)
        session.settle(timeout_ms=2000)
        data = {"key": key, "url": session.page.url}
        if element is not None:
            data["element"] = element
        return data


class ScrollAction(Action):
    name = "scroll"
    description = (
        "Scroll the page vertically. direction='down' | 'up'; amount_px "
        "defaults to ~one viewport. Content may lazy-load — take a fresh "
        "browser_snapshot after scrolling to see it."
    )
    mutates_ui = True
    identity_param_keys = ("direction",)
    schema = {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["down", "up"]},
            "amount_px": {"type": "integer", "minimum": 40, "maximum": 20000},
        },
        "required": ["direction"],
    }

    def _execute(self, session, *, direction: str, amount_px: int | None = None):
        page = session.page
        if amount_px is None:
            amount_px = page.evaluate("() => Math.round(window.innerHeight * 0.85)")
        delta = amount_px if direction == "down" else -amount_px
        page.evaluate("(dy) => window.scrollBy(0, dy)", delta)
        position = page.evaluate(
            "() => ({y: Math.round(window.scrollY), max: Math.round("
            "document.documentElement.scrollHeight - window.innerHeight)})"
        )
        return {"direction": direction, "amount_px": amount_px, "position": position}


class BackAction(Action):
    name = "back"
    description = (
        "Go back one entry in the tab's history. Old refs become invalid."
    )
    mutates_ui = True
    schema = {"type": "object", "properties": {}}

    def _execute(self, session):
        page = session.page
        resp = page.go_back(wait_until="load", timeout=10000)
        session._known_refs = set()
        if resp is None:
            return ActionResult(ok=False, error={
                "error": "E_NO_HISTORY",
                "message": "no previous page in history",
            })
        return session.location()


class TabsAction(Action):
    name = "tabs"
    description = (
        "Manage browser tabs. action='list' shows all tabs; 'switch' "
        "activates tabs[index]; 'new' opens a tab (optionally at `url`); "
        "'close' closes tabs[index] (default: the active tab). Switching "
        "invalidates snapshot refs — re-snapshot after."
    )
    mutates_ui = True
    identity_param_keys = ("action",)
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "switch", "new", "close"]},
            "index": {"type": "integer", "minimum": 0},
            "url": {"type": "string", "description": "Only for action='new'."},
        },
        "required": ["action"],
    }

    def _execute(self, session, *, action: str, index: int | None = None,
                 url: str | None = None):
        if action == "list":
            return session.tabs_info()
        if action == "switch":
            if index is None:
                return ActionResult(ok=False, error={
                    "error": "E_INVALID_ARG", "message": "switch needs `index`",
                })
            return session.switch_tab(index)
        if action == "new":
            return session.new_tab(url)
        if action == "close":
            return session.close_tab(index)
        return ActionResult(ok=False, error={
            "error": "E_INVALID_ARG", "message": f"unknown tabs action: {action}",
        })
