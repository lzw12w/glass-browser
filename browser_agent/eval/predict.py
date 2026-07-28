"""Single-step action predictor for offline evaluation.

Unlike the interactive agent loop (a multi-turn tool-calling conversation),
offline Mind2Web evaluation is teacher-forced: given the goal, the
ground-truth prior actions, and the current page snapshot, predict exactly ONE
action. We therefore do a single tool-less completion and parse a compact JSON
verdict — no browser mutation, no retries, one LLM call per step (cheap).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..llm.base import AssistantTurn, LLMClient

SYSTEM = """You are a web-navigation action predictor being evaluated offline.

You are given a TASK, the actions already taken, and a SNAPSHOT of the current
page: a tree of visible elements where interactive ones carry a `ref` (e.g.
"e12"). Predict the SINGLE next action that best advances the task.

Respond with ONLY a JSON object, no prose, no code fence:
  {"ref": "<ref from the snapshot>", "action": "CLICK|TYPE|SELECT", "value": "<text for TYPE/SELECT, else empty>"}

Rules:
- `ref` MUST be one of the refs present in the snapshot. Never invent a ref.
- CLICK for links/buttons/checkboxes. TYPE to enter text into an input/textarea
  (put the text in `value`). SELECT to choose a dropdown option (put the option
  label or value in `value`).
- Pick the element that a careful human would act on for THIS step, given the
  actions already done. Output the JSON and nothing else."""


@dataclass
class PredictedAction:
    ref: str
    action: str
    value: str
    raw_text: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.ref) and self.action in {"CLICK", "TYPE", "SELECT"}


def _collect_turn(chunks) -> AssistantTurn:
    turn = None
    for chunk in chunks:
        if chunk.turn_complete is not None:
            turn = chunk.turn_complete
    if turn is None:
        turn = AssistantTurn(text="", tool_calls=[], stop_reason="end_turn")
    return turn


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_prediction(text: str) -> PredictedAction:
    """Extract the JSON verdict from the model's reply. Tolerant of code
    fences / stray prose by grabbing the first balanced-looking object."""
    raw = text or ""
    match = _JSON_RE.search(raw)
    ref = action = value = ""
    if match:
        try:
            data = json.loads(match.group(0))
            ref = str(data.get("ref") or "").strip()
            action = str(data.get("action") or "").strip().upper()
            value = str(data.get("value") or "")
        except (json.JSONDecodeError, AttributeError):
            pass
    return PredictedAction(ref=ref, action=action, value=value, raw_text=raw)


def _render_snapshot(snapshot: dict, *, max_chars: int = 24000) -> str:
    """Compact JSON rendering of the snapshot tree (+ any iframe sections)."""
    payload = {"url": snapshot.get("_meta", {}).get("url", ""),
               "tree": snapshot.get("tree", [])}
    if snapshot.get("frames"):
        payload["frames"] = snapshot["frames"]
    text = json.dumps(payload, ensure_ascii=False)
    return text[:max_chars]


def build_user_text(goal: str, prev_action_reprs: list[str], snapshot: dict) -> str:
    prev = "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(prev_action_reprs)) or "  (none)"
    return (
        f"TASK: {goal}\n\n"
        f"ACTIONS ALREADY TAKEN:\n{prev}\n\n"
        f"CURRENT PAGE SNAPSHOT (JSON):\n{_render_snapshot(snapshot)}\n\n"
        "Predict the next action as JSON."
    )


def predict_action(llm: LLMClient, *, goal: str, prev_action_reprs: list[str],
                   snapshot: dict) -> PredictedAction:
    user_text = build_user_text(goal, prev_action_reprs, snapshot)
    messages = [llm.make_user_message(user_text)]
    turn = _collect_turn(llm.chat_stream(system=SYSTEM, messages=messages, tools=[]))
    return parse_prediction(turn.text)
