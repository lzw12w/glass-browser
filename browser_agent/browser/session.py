"""BrowserSession — stateful wrapper around a Playwright BrowserContext.

Responsibilities:
- produce LLM-friendly page snapshots with stable interaction refs (e1, e2, ...)
- resolve refs back to live locators for click / fill / press_key
- targeted lookups (find_element / read_text) that avoid full-tree dumps
- archive screenshots into the run workdir
- buffer console messages and network responses for inspection tools

Ref contract: every ``snapshot()`` re-enumerates interactive elements and
stamps them with a ``data-ba-ref`` attribute, resetting the ref generation.
Refs from an older generation are rejected with ``E_STALE_REF`` so the model
re-observes instead of acting on a page that may have changed. ``find_element``
adds refs to the CURRENT generation (additive, same page) so a targeted lookup
yields an immediately-usable ref without a full snapshot.

Frames: the snapshot JS traverses open shadow DOM (Playwright locators pierce
it automatically). Cross-document iframes are handled by evaluating the same
JS inside every child frame; each ref is mapped to its owning frame so
``resolve_ref`` targets the right document — this is why refs must resolve via
``self._ref_frames`` rather than a bare ``page.locator``.
"""
from __future__ import annotations

import datetime as _dt
import re
from collections import deque
from pathlib import Path

from ..errors import BrowserUnavailable, InvalidArgument, StaleRef, TargetNotFound
from .driver import BrowserDriver

REF_ATTR = "data-ba-ref"

# One in-frame pass: clears old refs, walks the visible DOM (piercing open
# shadow roots), stamps interactive elements, and returns a flattened
# salient-node tree. Wrapper divs without semantic value are elided so the
# model sees content density, not markup depth. ``box`` is intentionally NOT
# emitted per node (it costs a lot of tokens and the model targets by ref, not
# coordinates); ``describe_ref`` fetches geometry on demand.
_SNAPSHOT_JS = r"""
(startCounter) => {
  const INTERACTIVE_TAGS = new Set(['a','button','input','select','textarea','summary','option']);
  const INTERACTIVE_ROLES = new Set(['button','link','tab','checkbox','radio','menuitem',
    'menuitemcheckbox','menuitemradio','combobox','option','switch','searchbox','textbox',
    'slider','spinbutton']);
  const LANDMARK_TAGS = new Set(['nav','main','header','footer','form','aside','section',
    'table','thead','tbody','tr','ul','ol','li','dialog','details','fieldset','label']);
  const SKIP_TAGS = new Set(['script','style','noscript','template','svg','path','meta',
    'link','head','iframe','frame']);
  let counter = startCounter || 0;
  let totalNodes = 0;

  const clearRoot = (root) => {
    root.querySelectorAll('[data-ba-ref]').forEach(el => el.removeAttribute('data-ba-ref'));
  };
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
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
    return t.replace(/\s+/g, ' ').trim();
  };
  const attr = (el, ...names) => {
    for (const n of names) {
      const v = el.getAttribute(n);
      if (v) return v;
    }
    return '';
  };
  // Read both hyphen and underscore spellings: some pruned-HTML corpora
  // (e.g. Mind2Web cleaned_html) normalize `aria-label` -> `aria_label`.
  const accName = (el) =>
    attr(el, 'aria-label', 'aria_label', 'alt', 'title', 'placeholder');

  // Walk a root (element / shadowRoot / document.body). Returns salient nodes.
  const walk = (root) => {
    const out = [];
    if (totalNodes > 4000) return out;
    for (const child of root.children) {
      const tag = child.tagName ? child.tagName.toLowerCase() : '';
      if (!tag || SKIP_TAGS.has(tag)) continue;
      if (!isVisible(child)) continue;
      totalNodes++;
      const interactive = isInteractive(child);
      const text = ownText(child);
      const name = accName(child);
      const heading = /^h[1-6]$/.test(tag);
      // Descend into light DOM children AND an open shadow root.
      let kids = walk(child);
      if (child.shadowRoot) kids = kids.concat(walk(child.shadowRoot));
      const salient = interactive || heading || text.length > 0;
      if (salient) {
        const node = { tag };
        if (interactive) {
          const ref = 'e' + (++counter);
          child.setAttribute('data-ba-ref', ref);
          node.ref = ref;
          if (tag === 'input' || tag === 'textarea') {
            const type = child.getAttribute('type');
            if (type) node.type = type;
            if (child.value) node.value = String(child.value).slice(0, 80);
            if (child.checked) node.checked = true;
          }
          if (child.disabled) node.disabled = true;
          if (tag === 'select') {
            const opts = [];
            for (const o of child.options || []) {
              opts.push({ v: o.value, t: (o.textContent || '').trim().slice(0, 40),
                          ...(o.selected ? { selected: true } : {}) });
            }
            if (opts.length) node.options = opts.slice(0, 30);
          }
        }
        const role = child.getAttribute('role');
        if (role) node.role = role;
        if (heading) node.heading = tag;
        if (text) node.text = text.slice(0, 200);
        if (name && name !== text) node.name = String(name).slice(0, 80);
        // Icon-only control (no text, no own accessible name): borrow a label
        // from a descendant so the model can tell what it does (search vs
        // calendar vs menu). Fires only for otherwise-anonymous interactive
        // nodes, so the token cost is negligible.
        if (interactive && !node.text && !node.name) {
          const lbl = child.querySelector('[aria-label],[aria_label],[title],img[alt]');
          let hint = lbl ? attr(lbl, 'aria-label', 'aria_label', 'title', 'alt') : '';
          if (!hint) {
            const cls = (child.getAttribute('class') || '').split(/\s+/).find(
              c => /icon|search|menu|close|calendar|cart|nav|arrow|expand|toggle|filter|submit|next|prev/i.test(c));
            if (cls) hint = cls;
          }
          if (!hint) hint = child.getAttribute('id') || '';
          if (hint) node.name = String(hint).slice(0, 60);
        }
        if (tag === 'a') {
          const href = child.getAttribute('href');
          if (href && !href.startsWith('javascript:')) node.href = href.slice(0, 200);
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

  clearRoot(document);
  const tree = walk(document.body || document.documentElement);
  return {
    tree,
    nextCounter: counter,
    meta: {
      url: location.href,
      title: document.title,
      total_nodes: totalNodes,
    },
  };
}
"""


def _safe_filename_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")[:80] or "page"


def _collect_refs(nodes, out: set) -> None:
    for n in nodes or []:
        ref = n.get("ref")
        if ref:
            out.add(ref)
        _collect_refs(n.get("children"), out)


class BrowserSession:
    def __init__(self, driver: BrowserDriver, *, workdir: Path | None = None):
        self.driver = driver
        self.workdir = workdir
        self._page = None
        # Refs valid for the CURRENT snapshot generation only, each mapped to
        # the frame whose document holds it (so resolve_ref targets the right
        # document — page.locator does not pierce iframes).
        self._known_refs: set[str] = set()
        self._ref_frames: dict = {}
        self._ref_counter: int = 0
        self._console: deque[dict] = deque(maxlen=400)
        self._responses: deque[dict] = deque(maxlen=600)
        self._wired_page_ids: set[int] = set()
        # Durable task list written by todo_write; read by the agent loop.
        self._todos: list[dict] = []
        try:
            self.driver.context.on("page", self._wire_page)
        except BrowserUnavailable:
            pass  # tests construct the session before/without a live browser

    # ---- page management -----------------------------------------------
    @property
    def page(self):
        if self._page is not None and not self._page.is_closed():
            return self._page
        pages = self.driver.context.pages
        self._page = pages[-1] if pages else self.driver.context.new_page()
        self._wire_page(self._page)
        return self._page

    def _wire_page(self, page) -> None:
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

    def _reset_refs(self) -> None:
        self._known_refs = set()
        self._ref_frames = {}
        self._ref_counter = 0

    # ---- snapshot / refs -------------------------------------------------
    def snapshot(self) -> dict:
        """Enumerate the visible DOM (all frames, open shadow DOM) into an
        LLM-friendly tree with refs. Resets the ref generation."""
        page = self.page
        frames = list(page.frames)
        counter = 0
        ref_frames: dict = {}
        main_tree: list = []
        main_meta: dict = {}
        child_sections: list = []
        total_nodes = 0
        for frame in frames:
            try:
                raw = frame.evaluate(_SNAPSHOT_JS, counter)
            except Exception:
                # Detached frame, or a rare eval error — skip rather than fail
                # the whole snapshot. Cross-origin frames still evaluate fine
                # (Playwright runs JS inside each frame's own context).
                continue
            counter = raw.get("nextCounter", counter)
            total_nodes += int(raw.get("meta", {}).get("total_nodes", 0) or 0)
            refs: set = set()
            _collect_refs(raw.get("tree"), refs)
            for ref in refs:
                ref_frames[ref] = frame
            if frame is page.main_frame:
                main_tree = raw.get("tree", [])
                main_meta = raw.get("meta", {})
            elif raw.get("tree"):
                child_sections.append({
                    "url": raw.get("meta", {}).get("url", ""),
                    "tree": raw["tree"],
                })

        self._known_refs = set(ref_frames)
        self._ref_frames = ref_frames
        self._ref_counter = counter
        meta = dict(main_meta)
        meta["total_nodes"] = total_nodes
        meta["interactive_count"] = len(ref_frames)
        meta["tab_count"] = len(self.driver.context.pages)
        meta["frame_count"] = len(frames)
        out = {"tree": main_tree, "_meta": meta}
        if child_sections:
            out["frames"] = child_sections
        return out

    def _frame_for(self, ref: str):
        return self._ref_frames.get(ref) or self.page.main_frame

    def resolve_ref(self, ref: str):
        """Return a live locator for ``ref``; raise structured errors on
        stale/unknown refs so the model knows to re-snapshot."""
        if not isinstance(ref, str) or not re.fullmatch(r"e\d{1,6}", ref or ""):
            raise InvalidArgument(f"malformed ref: {ref!r} (expected e.g. 'e12')")
        if ref not in self._known_refs:
            raise StaleRef(
                f"ref {ref} does not belong to the current snapshot; "
                "call browser_snapshot to re-observe the page"
            )
        frame = self._frame_for(ref)
        locator = frame.locator(f'[{REF_ATTR}="{ref}"]')
        if locator.count() == 0:
            raise TargetNotFound(
                f"element for ref {ref} is gone (page changed since snapshot); "
                "call browser_snapshot again"
            )
        return locator.first

    def describe_ref(self, ref: str) -> dict:
        """Small element descriptor (incl. box geometry) echoed back in
        action results."""
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

    # ---- targeted lookups (cheap alternatives to a full snapshot) --------
    def find_elements(self, *, text: str | None = None, selector: str | None = None,
                      role: str | None = None, name: str | None = None,
                      limit: int = 5) -> dict:
        """Locate up to ``limit`` elements by text / CSS selector / ARIA role
        and return them with FRESH refs added to the current generation, so
        the model can act on a match without dumping the whole tree.

        Refs are stamped into the main-frame document; shadow DOM is pierced
        automatically by Playwright locators.
        """
        page = self.page
        if selector:
            locator = page.locator(selector)
        elif role:
            locator = page.get_by_role(role, name=name) if name else page.get_by_role(role)
        elif text:
            locator = page.get_by_text(text, exact=False)
        else:
            raise InvalidArgument("find_element needs text, selector, or role")

        try:
            total = locator.count()
        except Exception as e:
            raise InvalidArgument(f"bad locator: {e}") from e
        take = min(total, max(1, min(limit, 20)))
        results: list[dict] = []
        for i in range(take):
            el = locator.nth(i)
            self._ref_counter += 1
            ref = f"e{self._ref_counter}"
            try:
                info = el.evaluate(
                    """(node, ref) => {
                        node.setAttribute('data-ba-ref', ref);
                        const rect = node.getBoundingClientRect();
                        return {
                            tag: node.tagName.toLowerCase(),
                            role: node.getAttribute('role') || '',
                            text: (node.innerText || node.value || '').trim().slice(0, 120),
                            name: (node.getAttribute('aria-label') || node.getAttribute('title') ||
                                   node.getAttribute('placeholder') || '').slice(0, 80),
                            visible: rect.width > 0 && rect.height > 0,
                        };
                    }""",
                    ref,
                )
            except Exception:
                continue
            self._known_refs.add(ref)
            self._ref_frames[ref] = page.main_frame
            entry = {"ref": ref}
            for key in ("tag", "role", "text", "name"):
                if info.get(key):
                    entry[key] = info[key]
            if info.get("visible") is False:
                entry["visible"] = False
            results.append(entry)
        return {"count": total, "returned": len(results), "results": results}

    def read_text(self, *, selector: str | None = None, ref: str | None = None,
                  limit: int = 20) -> dict:
        """Return visible text of matching elements — read-only, no ref
        stamping. The cheapest way to extract an answer/value from the page."""
        if ref:
            locator = self.resolve_ref(ref)
            texts = [locator.inner_text()]
        elif selector:
            locator = self.page.locator(selector)
            count = locator.count()
            texts = [locator.nth(i).inner_text()
                     for i in range(min(count, max(1, min(limit, 50))))]
        else:
            raise InvalidArgument("read_text needs `selector` or `ref`")
        cleaned = [re.sub(r"\s+\n", "\n", t).strip()[:2000] for t in texts if t and t.strip()]
        return {"count": len(cleaned), "texts": cleaned}

    # ---- navigation ------------------------------------------------------
    def navigate(self, url: str, *, timeout_ms: int = 20000) -> dict:
        page = self.page
        resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        self.settle()
        return {
            "url": page.url,
            "title": page.title(),
            "status": resp.status if resp is not None else None,
        }

    def settle(self, *, timeout_ms: int = 3000) -> None:
        """Give an in-flight navigation/XHR burst a brief chance to finish.
        Never raises — SPAs that keep polling never fully idle and that's OK."""
        try:
            self.page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            pass

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
        self._reset_refs()
        return {"active_index": index, "url": self._page.url, "title": self._page.title()}

    def new_tab(self, url: str | None = None) -> dict:
        page = self.driver.context.new_page()
        self._page = page
        self._wire_page(page)
        self._reset_refs()
        if url:
            page.goto(url, wait_until="domcontentloaded")
            self.settle()
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
        self._reset_refs()
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
