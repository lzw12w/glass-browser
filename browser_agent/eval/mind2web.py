"""Mind2Web offline dataset loading + normalization.

The official ``osunlp/Mind2Web`` train files are ~600 MB each (they embed the
full ``raw_html`` of every step, which we don't use). So instead of the
``datasets`` library we stream the JSON array from the HF resolve endpoint and
stop after the first ``limit`` tasks — pulling only a few MB off the wire.

A normalized :class:`Mind2WebStep` keeps just what the offline evaluator needs
(goal, prior action reprs, the pruned ``cleaned_html``, the gold
``backend_node_id`` + operation). These are small and can be cached to JSONL
so tests / repeat runs never hit the network.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

# Public HF resolve endpoint for the (ungated) text Mind2Web train shards.
HF_TRAIN_URL = (
    "https://huggingface.co/datasets/osunlp/Mind2Web/resolve/main/"
    "data/train/train_{shard}.json"
)

VALID_OPS = {"CLICK", "TYPE", "SELECT"}


@dataclass
class Mind2WebStep:
    task_id: str
    goal: str
    step_index: int
    num_steps: int
    prev_action_reprs: list[str]
    cleaned_html: str
    gold_backend_node_id: str
    op: str
    value: str
    gold_repr: str = ""

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Mind2WebStep":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


def _gold_backend_id(action: dict) -> str | None:
    """Pick the target element's backend_node_id from an action's positive
    candidates. Prefer the flagged original target; else the first pos cand."""
    pos = action.get("pos_candidates") or []
    if not pos:
        return None
    for cand in pos:
        if cand.get("is_original_target"):
            bid = cand.get("backend_node_id")
            if bid is not None:
                return str(bid)
    bid = pos[0].get("backend_node_id")
    return str(bid) if bid is not None else None


def normalize_task(task: dict) -> list[Mind2WebStep]:
    """Turn one raw Mind2Web task into a list of single-step examples.

    Steps whose target has no resolvable backend_node_id (rare) are skipped —
    they can't be scored against our snapshot.
    """
    goal = task.get("confirmed_task") or task.get("task") or ""
    task_id = task.get("annotation_id") or task.get("task_id") or ""
    actions = task.get("actions") or []
    reprs = task.get("action_reprs") or [""] * len(actions)
    steps: list[Mind2WebStep] = []
    for i, action in enumerate(actions):
        op_block = action.get("operation") or {}
        op = str(op_block.get("op") or "").upper()
        if op not in VALID_OPS:
            continue
        gold = _gold_backend_id(action)
        cleaned = action.get("cleaned_html")
        if not gold or not cleaned:
            continue
        steps.append(Mind2WebStep(
            task_id=task_id,
            goal=goal,
            step_index=i,
            num_steps=len(actions),
            prev_action_reprs=list(reprs[:i]),
            cleaned_html=cleaned,
            gold_backend_node_id=gold,
            op=op,
            value=str(op_block.get("value") or ""),
            gold_repr=reprs[i] if i < len(reprs) else "",
        ))
    return steps


# ---- streaming JSON-array reader -------------------------------------------
def _iter_array_objects(byte_iter: Iterable[bytes], *, limit: int) -> Iterator[dict]:
    """Yield up to ``limit`` top-level objects from a streamed JSON array.

    A hand-rolled brace/string scanner so we never buffer the whole 600 MB
    file: we keep a persistent cursor into the growing buffer and abort as
    soon as ``limit`` objects have been decoded.
    """
    buf = bytearray()
    i = 0            # persistent scan cursor (never rescans from 0)
    depth = 0
    obj_start = -1
    in_str = False
    esc = False
    started = False
    yielded = 0
    for chunk in byte_iter:
        buf.extend(chunk)
        while i < len(buf):
            c = buf[i]
            if not started:
                if c == 0x5B:  # '['
                    started = True
                i += 1
                continue
            if in_str:
                if esc:
                    esc = False
                elif c == 0x5C:  # backslash
                    esc = True
                elif c == 0x22:  # '"'
                    in_str = False
                i += 1
                continue
            if c == 0x22:  # '"'
                in_str = True
            elif c == 0x7B:  # '{'
                if depth == 0:
                    obj_start = i
                depth += 1
            elif c == 0x7D:  # '}'
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    obj_bytes = bytes(buf[obj_start:i + 1])
                    yield json.loads(obj_bytes.decode("utf-8"))
                    yielded += 1
                    if yielded >= limit:
                        return
                    obj_start = -1
            i += 1


def stream_tasks_from_url(url: str, *, limit: int, timeout: int = 60,
                          chunk_size: int = 1 << 16) -> Iterator[dict]:
    """Stream raw task dicts from a Mind2Web JSON shard URL, stopping after
    ``limit`` tasks (only a few MB are actually read)."""
    req = urllib.request.Request(url, headers={"User-Agent": "browser-agent-eval"})
    resp = urllib.request.urlopen(req, timeout=timeout)

    def _chunks() -> Iterator[bytes]:
        while True:
            data = resp.read(chunk_size)
            if not data:
                return
            yield data

    try:
        yield from _iter_array_objects(_chunks(), limit=limit)
    finally:
        resp.close()


def load_steps_from_hf(*, limit_tasks: int = 5, shard: int = 0,
                       max_steps: int | None = None) -> list[Mind2WebStep]:
    """Stream ``limit_tasks`` tasks off HF and normalize them into steps."""
    url = HF_TRAIN_URL.format(shard=shard)
    steps: list[Mind2WebStep] = []
    for task in stream_tasks_from_url(url, limit=limit_tasks):
        steps.extend(normalize_task(task))
        if max_steps is not None and len(steps) >= max_steps:
            return steps[:max_steps]
    return steps


# ---- JSONL cache (offline / tests) -----------------------------------------
def dump_steps_to_jsonl(steps: list[Mind2WebStep], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for step in steps:
            f.write(json.dumps(step.to_json(), ensure_ascii=False) + "\n")
    return path


def load_steps_from_jsonl(path: str | Path, *, limit: int | None = None) -> list[Mind2WebStep]:
    steps: list[Mind2WebStep] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps.append(Mind2WebStep.from_json(json.loads(line)))
            if limit is not None and len(steps) >= limit:
                break
    return steps
