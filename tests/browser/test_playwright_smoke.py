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
  <select id="pick">
    <option value="a">Apple</option>
    <option value="b">Banana</option>
  </select>
  <div id="host"></div>
  <iframe srcdoc="<button id='inner'>InnerBtn</button>"></iframe>
  <script>
    const host = document.getElementById('host');
    const root = host.attachShadow({mode: 'open'});
    root.innerHTML = '<button id=shadowbtn>ShadowBtn</button>';
  </script>
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
    assert meta["interactive_count"] >= 3  # input, button, link, select
    assert meta["tab_count"] >= 1
    assert _find(snap["tree"], tag="h1") is not None
    button = _find(snap["tree"], tag="button", text="Go")
    assert button and button["ref"].startswith("e")
    # box is intentionally NOT in the tree (token saving); describe_ref has it.
    assert "box" not in button
    info = session.describe_ref(button["ref"])
    assert info["box"]["width"] > 0


def test_snapshot_captures_shadow_dom(session):
    snap = session.snapshot()
    shadow_btn = _find(snap["tree"], text="ShadowBtn")
    assert shadow_btn is not None, "shadow DOM button should be in the tree"
    assert shadow_btn.get("ref")
    # Playwright locators pierce open shadow DOM, so the ref resolves.
    assert session.describe_ref(shadow_btn["ref"])["tag"] == "button"


def test_snapshot_captures_iframe(session):
    snap = session.snapshot()
    frames = snap.get("frames") or []
    inner = None
    for fr in frames:
        inner = _find(fr["tree"], text="InnerBtn")
        if inner:
            break
    assert inner is not None, "iframe button should appear under frames"
    # Ref resolves against the owning frame, not the main document.
    session.resolve_ref(inner["ref"]).click()


def test_snapshot_lists_select_options(session):
    snap = session.snapshot()
    sel = _find(snap["tree"], tag="select")
    assert sel is not None and sel.get("options")
    labels = [o["t"] for o in sel["options"]]
    assert "Apple" in labels and "Banana" in labels
    session.resolve_ref(sel["ref"]).select_option(label="Banana")
    assert session.read_text(ref=sel["ref"]) is not None


def test_find_element_returns_usable_ref(session):
    session.snapshot()
    result = session.find_elements(text="Go", limit=3)
    assert result["returned"] >= 1
    ref = result["results"][0]["ref"]
    # Fresh ref joins the current generation and resolves immediately.
    assert session.resolve_ref(ref) is not None


def test_read_text_extracts_without_snapshot(session):
    session.navigate("data:text/html," + PAGE.replace("\n", ""))
    out = session.read_text(selector="h1")
    assert out["count"] == 1
    assert "Smoke Page" in out["texts"][0]


def test_fill_and_click_roundtrip(session):
    snap = session.snapshot()
    name_input = _find(snap["tree"], tag="input")
    button = _find(snap["tree"], tag="button", text="Go")

    session.resolve_ref(name_input["ref"]).fill("world")
    session.resolve_ref(button["ref"]).click()

    snap2 = session.snapshot()
    out = _find(snap2["tree"], text="hello world")
    assert out is not None, snap2


def test_stale_ref_rejected_after_navigation(session):
    snap = session.snapshot()
    button = _find(snap["tree"], tag="button", text="Go")
    session.navigate("about:blank")
    session._reset_refs()  # navigate action does this in the real flow
    with pytest.raises(StaleRef):
        session.resolve_ref(button["ref"])
    # restore the page for other tests (module-scoped fixture)
    session.navigate("data:text/html," + PAGE.replace("\n", ""))


def test_describe_ref_reports_element(session):
    snap = session.snapshot()
    button = _find(snap["tree"], tag="button", text="Go")
    info = session.describe_ref(button["ref"])
    assert info["tag"] == "button"
    assert info["box"]["height"] > 0
