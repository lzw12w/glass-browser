"""Offline Mind2Web evaluator.

For each step we load its ``cleaned_html`` into a live (headless) page via
``set_content``, run the REAL snapshot pipeline, then ask the model for a
single action and score it against the gold target.

Metrics (standard Mind2Web offline set, plus a perception diagnostic):
  * coverage      — gold element was exposed as an actionable ref by our
                    snapshot (upper bound on element accuracy; isolates the
                    perception layer from the model's choice).
  * element_acc   — model picked the ref whose backend_node_id == gold.
  * op_f1         — operation correctness; token-level F1 on the typed/selected
                    value for TYPE/SELECT, 1.0 for a correct CLICK.
  * step_sr       — element correct AND operation+value correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..browser.session import BrowserSession
from ..errors import BrowserAgentError
from ..llm.base import LLMClient
from .mind2web import Mind2WebStep
from .predict import PredictedAction, predict_action


@dataclass
class StepResult:
    task_id: str
    step_index: int
    op: str
    gold_backend_node_id: str
    predicted: PredictedAction
    picked_backend_node_id: str | None
    covered: bool          # gold element exposed as a ref by our snapshot
    element_correct: bool
    op_correct: bool       # op string matches
    value_f1: float
    step_success: bool     # element_correct AND op_correct AND value_f1>=0.5
    error: str | None = None


@dataclass
class Metrics:
    n: int = 0
    coverage: float = 0.0
    element_acc: float = 0.0
    op_f1: float = 0.0
    step_sr: float = 0.0
    results: list[StepResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"n": self.n, "coverage": round(self.coverage, 4),
                "element_acc": round(self.element_acc, 4),
                "op_f1": round(self.op_f1, 4), "step_sr": round(self.step_sr, 4)}


def _token_f1(pred: str, gold: str) -> float:
    p = (pred or "").lower().split()
    g = (gold or "").lower().split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = 0
    gold_pool = list(g)
    for tok in p:
        if tok in gold_pool:
            gold_pool.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    prec = common / len(p)
    rec = common / len(g)
    return 2 * prec * rec / (prec + rec)


def _backend_id_of(session: BrowserSession, ref: str) -> str | None:
    """Resolve a predicted ref to the element's Mind2Web backend_node_id
    (for reporting only)."""
    try:
        locator = session.resolve_ref(ref)
    except BrowserAgentError:
        return None
    try:
        return locator.evaluate("(el) => el.getAttribute('backend_node_id')")
    except Exception:
        return None


def _ref_matches_gold(session: BrowserSession, ref: str, gold: str) -> bool:
    """Element accuracy with the standard web-agent relaxation: the model is
    correct if it picked the gold node itself, OR the closest actionable
    ancestor of the gold node (the icon-inside-a-button case — clicking the
    button is the real action), OR a node nested inside the gold target."""
    try:
        locator = session.resolve_ref(ref)
    except BrowserAgentError:
        return False
    try:
        return bool(locator.evaluate(
            """(picked, gold) => {
                const g = document.querySelector('[backend_node_id=\"' + gold + '\"]');
                if (!g) return false;
                if (picked === g) return true;
                if (g.closest('[data-ba-ref]') === picked) return true;  // actionable ancestor
                if (picked.closest('[backend_node_id=\"' + gold + '\"]')) return true;  // inside gold
                return false;
            }""", gold))
    except Exception:
        return False


def _gold_is_covered(session: BrowserSession, gold: str) -> bool:
    """Perception recall: did our snapshot make the gold target actionable —
    either by stamping a ref on it directly OR on its closest ancestor (so a
    click reaches it)? Independent of the model's choice."""
    try:
        return bool(session.page.evaluate(
            """(gold) => {
                const g = document.querySelector('[backend_node_id=\"' + gold + '\"]');
                return !!(g && g.closest('[data-ba-ref]'));
            }""", gold))
    except Exception:
        return False


def evaluate_step(session: BrowserSession, llm: LLMClient, step: Mind2WebStep) -> StepResult:
    page = session.page
    page.set_content(step.cleaned_html, wait_until="domcontentloaded")
    session._reset_refs()  # fresh ref generation for this page
    snapshot = session.snapshot()
    covered = _gold_is_covered(session, step.gold_backend_node_id)

    pred = predict_action(
        llm, goal=step.goal,
        prev_action_reprs=step.prev_action_reprs, snapshot=snapshot,
    )
    picked = _backend_id_of(session, pred.ref) if pred.ref else None
    element_correct = bool(pred.ref) and _ref_matches_gold(session, pred.ref, step.gold_backend_node_id)
    op_correct = pred.action == step.op
    if step.op == "CLICK":
        value_f1 = 1.0 if op_correct else 0.0
    else:
        value_f1 = _token_f1(pred.value, step.value) if op_correct else 0.0
    step_success = element_correct and op_correct and value_f1 >= 0.5

    return StepResult(
        task_id=step.task_id, step_index=step.step_index, op=step.op,
        gold_backend_node_id=step.gold_backend_node_id, predicted=pred,
        picked_backend_node_id=picked, covered=covered,
        element_correct=element_correct, op_correct=op_correct,
        value_f1=value_f1, step_success=step_success,
    )


def run(session: BrowserSession, llm: LLMClient, steps: list[Mind2WebStep],
        *, on_result=None) -> Metrics:
    results: list[StepResult] = []
    for step in steps:
        try:
            res = evaluate_step(session, llm, step)
        except Exception as e:  # never let one bad step kill the whole run
            res = StepResult(
                task_id=step.task_id, step_index=step.step_index, op=step.op,
                gold_backend_node_id=step.gold_backend_node_id,
                predicted=PredictedAction("", "", ""), picked_backend_node_id=None,
                covered=False, element_correct=False, op_correct=False,
                value_f1=0.0, step_success=False, error=f"{type(e).__name__}: {e}",
            )
        results.append(res)
        if on_result is not None:
            on_result(res)

    n = len(results) or 1
    metrics = Metrics(
        n=len(results),
        coverage=sum(r.covered for r in results) / n,
        element_acc=sum(r.element_correct for r in results) / n,
        op_f1=sum(r.value_f1 for r in results) / n,
        step_sr=sum(r.step_success for r in results) / n,
        results=results,
    )
    return metrics
