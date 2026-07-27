"""BrowserSession — stateful wrapper around a Playwright BrowserContext.

Responsibilities:
- produce LLM-friendly page snapshots with stable interaction refs (e1, e2, ...)
- resolve refs back to live locators for click / fill / press_key
- archive screenshots into the run workdir
- buffer console messages and network responses for inspection tools

Ref contract: every ``snapshot()`` call re-enumerates interactive elements and
stamps them with a ``data-ba-ref`` attribute. Refs from an older snapshot
generation are rejected with ``E_STALE_REF`` so the model is forced to
re-observe instead of acting on a page that may have changed.
"""
from __future__ import annotations

import datetime as _dt
import re
from collections import deque
from pathlib import Path
from typing import Any, Optional

from ..errors import BrowserUnavailable, InvalidArgument, StaleRef, TargetNotFound
from .driver import BrowserDriver

REF_ATTR = "data-ba-ref"

# One page-side pass: clears old refs, walks the visible DOM, stamps
# interactive elements, and returns a flattened salient-node tree. Wrapper
# divs without semantic value are elided so the model sees content density,
# not markup depth.
_SNAPSHOT_JS = """
() => {
  const INTERACTIVE_TAGS = new Set(['a','button','input','select','textarea','summary','option']);
  const INTERACTIVE_ROLES = new Set(['button','link','tab','checkbox','radio','menuitem',
    'combobox','option','switch','searchbox','textbox','slider','spinbutton']);
  const LANDMARK_TAGS = new Set(['nav','main','header','footer','form','aside','section',
    'table','thead','tbody','tr','ul','ol','li','dialog','details','fieldset','label']);
  const SKIP_TAGS = new Set(['script','style','noscript','template','svg','path','meta','link','head']);
  document.querySelectorAll('[data-ba-ref]').forEach(el => el.removeAttribute('data-ba-ref'));
  let counter = 0;
  let totalNodes = 0;
  let interactiveCount = 0;

  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };
  const isInteractive = (el) => {
    const tag = el.tagName.toLowerCase();
    if (INTERACTIVE_TAGS.has(tag)) return true;
    const role = el.getAttribute('role');
    if (role && INTERACTIVE_ROLES.has(role)) return true;
    if (el.hasAttribute('onclick')) return true;
    if (el.isContentEditable) return true;
    const ti = el.getAttribute('tabindex');
    if (ti !== null && parseInt(ti, 10) >= 0) return true;
    return false;
  };
  const ownText = (el) => {
    let t = '';
    for (const child of el.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) t += child.textContent;
    }
    return t.replace(/\\s+/g, ' ').trim();
  };
  const accName = (el) =>
    el.getAttribute('aria-label') || el.getAttribute('alt') ||
    el.getAttribute('title') || el.getAttribute('placeholder') || '';

  const walk = (el) => {
    const out = [];
    if (totalNodes > 3000) return out;
    for (const child of el.children) {
      const tag = child.tagName.toLowerCase();
      if (SKIP_TAGS.has(tag)) continue;
      if (!isVisible(child)) continue;
      totalNodes++;
      const interactive = isInteractive(child);
      const text = ownText(child);
      const name = accName(child);
      const heading = /^h[1-6]$/.test(tag);
      const kids = walk(child);
      const salient = interactive || heading || text.length > 0;
      if (salient) {
        const node = { tag };
        if (interactive) {
          const ref = 'e' + (++counter);
          child.setAttribute('data-ba-ref', ref);
          node.ref = ref;
          interactiveCount++;
          const rect = child.getBoundingClientRect();
          node.box = { x: Math.round(rect.x), y: Math.round(rect.y),
                       width: Math.round(rect.width), height: Math.round(rect.height) };
          if (tag === 'input' || tag === 'textarea' || tag === 'select') {
            const type = child.getAttribute('type');
            if (type) node.type = type;
            if (child.value) node.value = String(child.value).slice(0, 80);
            if (child.checked) node.checked = true;
            if (child.disabled) node.disabled = true;
          }
        }
        const role = child.getAttribute('role');
        if (role) node.role = role;
        if (heading) node.heading = tag;
        if (text) node.text = text.slice(0, 120);
        if (name && name !== text) node.name = String(name).slice(0, 80);
        if (tag === 'a') {
          const href = child.getAttribute('href');
          if (href && !href.startsWith('javascript:')) node.href = href.slice(0, 160);
        }
        if (kids.length) node.children = kids;
        out.push(node);
      } else if (LANDMARK_TAGS.has(tag) && kids.length) {
        out.push({ tag, children: kids });
      } else {
        out.push(...kids);
      }
    }
    return out;
  };

  const tree = walk(document.body);
  return {
    tree,
    meta: {
      url: location.href,
      title: document.title,
      total_nodes: totalNodes,
      interactive_count: interactiveCount,
    },
  };
}
"""


def _safe_filename_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")[:80] or "page"


class BrowserSession:
    def __init__(self, driver: BrowserDriver, *, workdir: Path | None = None):
        self.driver = driver
        self.workdir = workdir
        self._page = None
        # Refs valid for the CURRENT snapshot generation only.
        self._known_refs: set[str] = set()
        self._console: deque[dict] = deque(maxlen=400)
        self._responses: deque[dict] = deque(maxlen=600)
        self._wired_page_ids: set[int] = set()
        # Durable task list written by todo_write; read by the agent loop.
        self._todos: list[dict] = []
        # Wire pages opened later (popups, window.open, new_tab).
        try:
            self.driver.context.on("page", self._wire_page)
        except BrowserUnavailable:
            pass  # tests construct the session before/without a live browser

    # ---- page management -----------------------------------------------
    @property
    def page(self):
        """The active page. Falls back to the context's last page, creating
        one when the context is empty (fresh launch)."""
        if self._page is not None and not self._page.is_closed():
            return self._page
        pages = self.driver.context.pages
        self._page = pages[-1] if pages else self.driver.context.new_page()
        self._wire_page(self._page)
        return self._page

    def _wire_page(self, page) -> None:
        """Attach console/network listeners once per page object."""
        if id(page) in self._wired_page_ids:
            return
        self._wired_page_ids.add(id(page))

        def on_console(msg):
            try:
                self._console.append({
                    "type": msg.type,
                    "text": str(msg.text)[:500],
                    "ts": _dt.datetime.now().isoformat(timespec="seconds"),
                })
            except Exception:
                pass

        def on_response(resp):
            try:
                self._responses.append({
                    "method": resp.request.method,
                    "url": resp.url[:300],
                    "status": resp.status,
                    "resource_type": resp.request.resource_type,
                })
            except Exception:
                pass

        page.on("console", on_console)
        page.on("response", on_response)

    # ---- snapshot / refs -------------------------------------------------
    def snapshot(self) -> dict:
        """Enumerate the visible DOM into an LLM-friendly tree with refs."""
        page = self.page
        try:
            raw = page.evaluate(_SNAPSHOT_JS)
        except Exception as e:
            raise BrowserUnavailable(f"snapshot failed: {e}") from e
        tree = raw.get("tree") if isinstance(raw, dict) else None
        meta = raw.get("meta") if isinstance(raw, dict) else {}
        refs: set[str] = set()

        def collect(nodes):
            for n in nodes or []:
                ref = n.get("ref")
                if ref:
                    refs.add(ref)
                collect(n.get("children"))

        collect(tree)
        self._known_refs = refs
        meta = dict(meta or {})
        meta["tab_count"] = len(self.driver.context.pages)
        return {"tree": tree or [], "_meta": meta}

    def resolve_ref(self, ref: str):
        """Return a live locator for ``ref``; raise structured errors on
        stale/unknown refs so the model knows to re-snapshot."""
        if not isinstance(ref, str) or not re.fullmatch(r"e\d{1,5}", ref or ""):
            raise InvalidArgument(f"malformed ref: {ref!r} (expected e.g. 'e12')")
        if ref not in self._known_refs:
            raise StaleRef(
                f"ref {ref} does not belong to the current snapshot; "
                "call browser_snapshot to re-observe the page"
            )
        locator = self.page.locator(f'[{REF_ATTR}="{ref}"]')
        if locator.count() == 0:
            raise TargetNotFound(
                f"element for ref {ref} is gone (page changed since snapshot); "
                "call browser_snapshot again"
            )
        return locator.first

    def describe_ref(self, ref: str) -> dict:
        """Small element descriptor echoed back in action results."""
        locator = self.resolve_ref(ref)
        try:
            info = locator.evaluate(
                """(el) => {
                    const rect = el.getBoundingClientRect();
                    return {
                        tag: el.tagName.toLowerCase(),
                        role: el.getAttribute('role') || '',
                        name: (el.getAttribute('aria-label') || el.getAttribute('title') ||
                               el.getAttribute('placeholder') || (el.innerText || '').trim()).slice(0, 80),
                        box: { x: Math.round(rect.x), y: Math.round(rect.y),
                               width: Math.round(rect.width), height: Math.round(rect.height) },
                    };
                }"""
            )
        except Exception:
            info = {}
        out = {"ref": ref}
        if isinstance(info, dict):
            for key in ("tag", "role", "name"):
                value = info.get(key)
                if value:
                    out[key] = value
            if isinstance(info.get("box"), dict):
                out["box"] = info["box"]
        return out

    # ---- navigation ------------------------------------------------------
    def navigate(self, url: str, *, timeout_ms: int = 15000) -> dict:
        page = self.page
        # ``load`` over ``networkidle``: SPAs with polling never go idle.
        resp = page.goto(url, wait_until="load", timeout=timeout_ms)
        return {
            "url": page.url,
            "title": page.title(),
            "status": resp.status if resp is not None else None,
        }

    def location(self) -> dict:
        page = self.page
        return {"url": page.url, "title": page.title()}

    # ---- screenshots -------------------------------------------------------
    def screenshot(self, *, full_page: bool = False) -> Path:
        page = self.page
        base = (self.workdir / "screens") if self.workdir else Path.cwd()
        base.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%H%M%S")
        path = base / f"{ts}_{_safe_filename_stem(page.title())}.png"
        page.screenshot(path=str(path), full_page=full_page)
        return path

    # ---- tabs ---------------------------------------------------------------
    def tabs_info(self) -> dict:
        pages = self.driver.context.pages
        active = self.page
        tabs = []
        for i, p in enumerate(pages):
            entry = {"index": i, "url": p.url}
            try:
                entry["title"] = p.title()
            except Exception:
                entry["title"] = ""
            if p is active:
                entry["active"] = True
            tabs.append(entry)
        return {"count": len(pages), "tabs": tabs}

    def switch_tab(self, index: int) -> dict:
        pages = self.driver.context.pages
        if not 0 <= index < len(pages):
            raise InvalidArgument(f"tab index {index} out of range (0..{len(pages) - 1})")
        self._page = pages[index]
        self._wire_page(self._page)
        self._page.bring_to_front()
        # Refs belong to the previously active page's snapshot.
        self._known_refs = set()
        return {"active_index": index, "url": self._page.url, "title": self._page.title()}

    def new_tab(self, url: str | None = None) -> dict:
        page = self.driver.context.new_page()
        self._page = page
        self._wire_page(page)
        self._known_refs = set()
        if url:
            page.goto(url, wait_until="load")
        return {"active_index": len(self.driver.context.pages) - 1,
                "url": page.url, "title": page.title()}

    def close_tab(self, index: int | None = None) -> dict:
        pages = self.driver.context.pages
        if index is None:
            target = self.page
        elif 0 <= index < len(pages):
            target = pages[index]
        else:
            raise InvalidArgument(f"tab index {index} out of range (0..{len(pages) - 1})")
        target.close()
        self._page = None
        self._known_refs = set()
        remaining = self.driver.context.pages
        return {"count": len(remaining)}

    # ---- log buffers ----------------------------------------------------------
    def console_logs(self, *, limit: int = 50, level: str | None = None) -> list[dict]:
        entries = list(self._console)
        if level:
            entries = [e for e in entries if e.get("type") == level]
        return entries[-max(1, min(limit, 200)):]

    def network_requests(self, *, limit: int = 50, url_contains: str | None = None) -> list[dict]:
        entries = list(self._responses)
        if url_contains:
            entries = [e for e in entries if url_contains in e.get("url", "")]
        return entries[-max(1, min(limit, 200)):]

    # ---- teardown ------------------------------------------------------------
    def close(self) -> None:
        self.driver.close()
