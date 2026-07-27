"""Real-Playwright smoke test against a local data: URL page.

Requires ``playwright install chromium``. Skipped automatically when the
chromium binary is missing so the kernel test suite stays runnable on
machines without browsers.
"""
from __future__ import annotations

import pytest

from browser_agent.browser import BrowserDriver, BrowserSession
from browser_agent.errors import BrowserAgentError, StaleRef

PAGE = """
<!DOCTYPE html>
<html><head><title>Smoke</title></head>
<body>
  <h1>Smoke Page</h1>
  <input id="name" placeholder="your name">
  <button id="go" onclick="
      document.getElementById('out').textContent =
          'hello ' + document.getElementById('name').value;
  ">Go</button>
  <p id="out"></p>
  <a href="about:blank">away</a>
</body></html>
"""


@pytest.fixture(scope="module")
def session():
    driver = BrowserDriver()
    try:
        driver.launch(headless=True)
    except BrowserAgentError as e:
        pytest.skip(f"chromium unavailable: {e}")
    sess = BrowserSession(driver)
    sess.navigate("data:text/html," + PAGE.replace("\n", ""))
    yield sess
    driver.close()


def _find(nodes, **want):
    """Depth-first search of the snapshot tree for a node matching fields."""
    for n in nodes:
        if all(n.get(k) == v for k, v in want.items()):
            return n
        hit = _find(n.get("children", []), **want)
        if hit:
            return hit
    return None


def test_snapshot_has_refs_and_meta(session):
    snap = session.snapshot()
    meta = snap["_meta"]
    assert meta["title"] == "Smoke"
    assert meta["interactive_count"] >= 3  # input, button, link
    assert meta["tab_count"] >= 1
    assert _find(snap["tree"], tag="h1") is not None
    button = _find(snap["tree"], tag="button")
    assert button and button["ref"].startswith("e")
    assert button["box"]["width"] > 0


def test_fill_and_click_roundtrip(session):
    snap = session.snapshot()
    name_input = _find(snap["tree"], tag="input")
    button = _find(snap["tree"], tag="button")

    session.resolve_ref(name_input["ref"]).fill("world")
    session.resolve_ref(button["ref"]).click()

    snap2 = session.snapshot()
    out = _find(snap2["tree"], text="hello world")
    assert out is not None, snap2


def test_stale_ref_rejected_after_navigation(session):
    snap = session.snapshot()
    button = _find(snap["tree"], tag="button")
    session.navigate("about:blank")
    session._known_refs = set()  # navigate action does this in the real flow
    with pytest.raises(StaleRef):
        session.resolve_ref(button["ref"])
    # restore the page for other tests (module-scoped fixture)
    session.navigate("data:text/html," + PAGE.replace("\n", ""))


def test_describe_ref_reports_element(session):
    snap = session.snapshot()
    button = _find(snap["tree"], tag="button")
    info = session.describe_ref(button["ref"])
    assert info["tag"] == "button"
    assert info["box"]["height"] > 0
