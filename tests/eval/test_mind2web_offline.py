"""Offline Mind2Web harness tests.

Pure-logic tests (streaming parser, normalization, prediction parsing) run
everywhere. The end-to-end scoring tests need a real Chromium (set_content +
snapshot) and skip automatically when it is unavailable, mirroring the
Playwright smoke suite.
"""
from __future__ import annotations

import pytest

from browser_agent.eval import harness as m2h
from browser_agent.eval import mind2web as m2w
from browser_agent.eval.predict import parse_prediction
from browser_agent.eval.mind2web import Mind2WebStep, normalize_task
from browser_agent.errors import BrowserAgentError
from browser_agent.llm.scripted import ScriptedLLM


# ---- pure logic (no browser) ----------------------------------------------
def test_streaming_json_array_stops_at_limit():
    # Three objects; we only want two — the parser must not require the whole
    # array to be well-terminated (it aborts early).
    blob = b'[{"a":1,"s":"}]{"},{"a":2},{"a":3}'  # note: embedded braces in a string
    chunks = [blob[i:i + 4] for i in range(0, len(blob), 4)]  # tiny chunks
    objs = list(m2w._iter_array_objects(iter(chunks), limit=2))
    assert objs == [{"a": 1, "s": "}]{"}, {"a": 2}]


def test_normalize_task_builds_steps_and_skips_bad_ops():
    task = {
        "annotation_id": "t1",
        "confirmed_task": "Book a rental car",
        "action_reprs": ["[link] Cars -> CLICK", "[textbox] Where -> TYPE: NYC",
                         "[button] noop -> HOVER"],
        "actions": [
            {"operation": {"op": "CLICK", "value": ""},
             "cleaned_html": '<a backend_node_id="10">Cars</a>',
             "pos_candidates": [{"backend_node_id": "10", "is_original_target": True}]},
            {"operation": {"op": "TYPE", "value": "NYC"},
             "cleaned_html": '<input backend_node_id="20">',
             "pos_candidates": [{"backend_node_id": "20"}]},
            {"operation": {"op": "HOVER", "value": ""},  # unsupported op -> skipped
             "cleaned_html": '<div backend_node_id="30"></div>',
             "pos_candidates": [{"backend_node_id": "30"}]},
        ],
    }
    steps = normalize_task(task)
    assert [s.op for s in steps] == ["CLICK", "TYPE"]
    assert steps[0].gold_backend_node_id == "10"
    assert steps[1].value == "NYC"
    assert steps[1].prev_action_reprs == ["[link] Cars -> CLICK"]


def test_parse_prediction_is_tolerant():
    good = parse_prediction('{"ref":"e5","action":"click","value":""}')
    assert good.ref == "e5" and good.action == "CLICK" and good.ok
    fenced = parse_prediction('```json\n{"ref":"e2","action":"TYPE","value":"hi"}\n```')
    assert fenced.ref == "e2" and fenced.value == "hi"
    junk = parse_prediction("I cannot help with that")
    assert not junk.ok


def test_token_f1():
    assert m2h._token_f1("new york", "new york") == 1.0
    assert m2h._token_f1("", "") == 1.0
    assert m2h._token_f1("york", "new york") == pytest.approx(2 / 3)
    assert m2h._token_f1("paris", "new york") == 0.0


def test_jsonl_roundtrip(tmp_path):
    steps = [Mind2WebStep("t", "goal", 0, 1, [], "<a>x</a>", "9", "CLICK", "", "r")]
    path = m2w.dump_steps_to_jsonl(steps, tmp_path / "s.jsonl")
    back = m2w.load_steps_from_jsonl(path)
    assert back[0].gold_backend_node_id == "9" and back[0].goal == "goal"


# ---- end-to-end scoring (needs Chromium) ----------------------------------
CLEANED = (
    '<div backend_node_id="1">'
    '<a backend_node_id="10" href="/cars">Rental Cars</a>'
    '<input backend_node_id="20" type="text" placeholder="Pick-up">'
    '<button backend_node_id="30">Search</button>'
    '</div>'
)


@pytest.fixture()
def session():
    from browser_agent.browser import BrowserDriver, BrowserSession
    driver = BrowserDriver()
    try:
        driver.launch(headless=True)
    except BrowserAgentError as e:
        pytest.skip(f"chromium unavailable: {e}")
    yield BrowserSession(driver)
    driver.close()


def _step(op="CLICK", gold="10", value=""):
    return Mind2WebStep("t1", "Find rental cars", 0, 1, [], CLEANED, gold, op, value, "")


def test_coverage_detects_gold_exposed_as_ref(session):
    session.page.set_content(CLEANED, wait_until="domcontentloaded")
    session._reset_refs()
    session.snapshot()
    # anchor #10 is interactive -> stamped with a ref -> covered
    assert m2h._gold_is_covered(session, "10") is True
    # the wrapper div #1 is not interactive -> never stamped
    assert m2h._gold_is_covered(session, "1") is False


def test_correct_click_scores_full(session):
    # anchor is the first interactive element -> ref e1
    llm = ScriptedLLM(['{"ref":"e1","action":"CLICK","value":""}'])
    res = m2h.evaluate_step(session, llm, _step(op="CLICK", gold="10"))
    assert res.covered and res.element_correct and res.op_correct
    assert res.step_success and res.picked_backend_node_id == "10"


def test_wrong_element_fails_but_still_covered(session):
    # pick the button (e3) instead of the gold anchor (#10)
    llm = ScriptedLLM(['{"ref":"e3","action":"CLICK","value":""}'])
    res = m2h.evaluate_step(session, llm, _step(op="CLICK", gold="10"))
    assert res.covered is True          # perception exposed the gold element
    assert res.element_correct is False  # but the model chose wrong
    assert res.step_success is False


def test_hallucinated_ref_is_not_correct(session):
    llm = ScriptedLLM(['{"ref":"e99","action":"CLICK","value":""}'])
    res = m2h.evaluate_step(session, llm, _step(op="CLICK", gold="10"))
    assert res.picked_backend_node_id is None
    assert res.element_correct is False and res.covered is True


def test_type_op_value_f1(session):
    # input is e2; gold value "New York", model types "New York City"
    llm = ScriptedLLM(['{"ref":"e2","action":"TYPE","value":"New York City"}'])
    res = m2h.evaluate_step(session, llm, _step(op="TYPE", gold="20", value="New York"))
    assert res.element_correct and res.op_correct
    assert 0.5 <= res.value_f1 <= 1.0 and res.step_success


def test_run_aggregates_metrics(session):
    llm = ScriptedLLM([
        '{"ref":"e1","action":"CLICK","value":""}',
        '{"ref":"e3","action":"CLICK","value":""}',
    ])
    metrics = m2h.run(session, llm, [_step(gold="10"), _step(gold="10")])
    d = metrics.as_dict()
    assert d["n"] == 2
    assert d["coverage"] == 1.0        # both steps expose the gold ref
    assert d["element_acc"] == 0.5     # one right, one wrong
