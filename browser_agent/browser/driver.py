"""Playwright sync driver — browser lifecycle management.

Owns the Playwright process, the Browser/BrowserContext pair, and shutdown.
Everything page-level (snapshots, refs, tabs) lives on :class:`BrowserSession`.

Import of ``playwright`` is deferred to construction so that unit tests using
a fake session never need the dependency at import time.
"""
from __future__ import annotations

from ..errors import BrowserUnavailable


class BrowserDriver:
    """Launch a Playwright-managed Chromium, or attach to a running Chrome
    via CDP. Sync API only — the agent loop is synchronous end-to-end.
    """

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._owns_browser = False

    # ---- lifecycle -----------------------------------------------------
    def launch(self, *, headless: bool = False) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise BrowserUnavailable(
                "playwright not installed. `pip install playwright && playwright install chromium`."
            ) from e
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.launch(headless=headless)
        except Exception as e:
            self._playwright.stop()
            self._playwright = None
            raise BrowserUnavailable(f"failed to launch chromium: {e}") from e
        self._context = self._browser.new_context()
        self._owns_browser = True

    def connect_over_cdp(self, cdp_url: str) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise BrowserUnavailable(
                "playwright not installed. `pip install playwright`."
            ) from e
        self._playwright = sync_playwright().start()
        try:
            self._browser = self._playwright.chromium.connect_over_cdp(cdp_url)
        except Exception as e:
            self._playwright.stop()
            self._playwright = None
            raise BrowserUnavailable(f"failed to attach over CDP: {e}") from e
        # Reuse the running browser's default context so the agent sees the
        # user's existing tabs / cookies instead of an empty profile.
        contexts = self._browser.contexts
        self._context = contexts[0] if contexts else self._browser.new_context()
        self._owns_browser = False

    def close(self) -> None:
        """Best-effort teardown. Never raises — called from finally blocks
        and atexit-style paths where a secondary failure only masks the
        original one.
        """
        try:
            if self._browser is not None and self._owns_browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._context = None
        self._playwright = None

    # ---- accessors -----------------------------------------------------
    @property
    def context(self):
        if self._context is None:
            raise BrowserUnavailable("browser is not started; call launch() or connect_over_cdp()")
        return self._context

    @property
    def connected(self) -> bool:
        return self._context is not None
