"""Offline evaluation harnesses for the browser agent.

Currently: Mind2Web offline (static-HTML, single-step action prediction).
This exercises the REAL perception pipeline (``set_content`` -> snapshot ->
ref selection) without touching a live network, so it is cheap to iterate on
and measures two decoupled things:

  * perception recall  — did our snapshot expose the gold element as an
    actionable ref at all (coverage), and
  * selection accuracy — given the exposed refs, did the model pick the right
    one and the right operation (element accuracy / operation F1 / step SR).
"""
from __future__ import annotations

from .harness import Metrics, StepResult, evaluate_step, run
from .mind2web import (
    Mind2WebStep,
    load_steps_from_jsonl,
    normalize_task,
    stream_tasks_from_url,
    dump_steps_to_jsonl,
)
from .predict import PredictedAction, predict_action

__all__ = [
    "Metrics",
    "StepResult",
    "evaluate_step",
    "run",
    "Mind2WebStep",
    "load_steps_from_jsonl",
    "normalize_task",
    "stream_tasks_from_url",
    "dump_steps_to_jsonl",
    "PredictedAction",
    "predict_action",
]
