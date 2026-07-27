"""Single-turn tool-result JSON compaction.

This module is responsible for compressing individual tool-call result payloads
before they are stored in the conversation history.  It is distinct from the
*context* compaction module (compact.py) which operates across turns.

Design principles:
  - Always emit valid JSON.  Never mid-string truncate.
  - Tools whose name is in FULL_DUMP_TOOLS represent a deliberate user
    request for full content (browser_snapshot, find_element, ...).  For these
    we only minify; never degrade.
  - For other tools, if the minified payload exceeds COMPACT_LIMIT bytes
    we emit a structured summary object that preserves enough handles for the
    LLM to drill down (ref, role, name, text).
"""
from __future__ import annotations

import json


# ---- configuration constants -------------------------------------------

COMPACT_LIMIT = 16000
FULL_DUMP_TOOLS = {"browser_snapshot", "find_element"}


# ---- public API --------------------------------------------------------

def compact_payload(payload, tool_name: str | None = None) -> str:
    """Minify and optionally degrade a tool-result payload to fit within token budget.

    Returns a compact JSON string.  For tools in FULL_DUMP_TOOLS only
    minification is applied (no lossy degradation).  For other tools, payloads
    exceeding COMPACT_LIMIT bytes are lossy-summarised.
    """
    s = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    if tool_name in FULL_DUMP_TOOLS:
        # User explicitly asked for full content — minify only, never degrade.
        return s
    if len(s) <= COMPACT_LIMIT:
        return s
    summary = _summarize_oversized(payload, original_len=len(s))
    return json.dumps(summary, ensure_ascii=False, default=str, separators=(",", ":"))


# ---- internal helpers --------------------------------------------------

def _summarize_oversized(payload, *, original_len: int):
    """Lossy structural summary for over-limit payloads.  Always returns valid JSON-able value."""
    reason = f"payload {original_len} bytes exceeds {COMPACT_LIMIT}"

    if isinstance(payload, dict):
        # Detect snapshot-tree-shaped payloads: keep _meta + a 2-level skeleton
        # because the ref handles are the keys to subsequent operations.
        tree = payload.get("tree") if isinstance(payload.get("tree"), dict) else None
        looks_like_tree = (
            tree is not None
            or "_meta" in payload
            or isinstance(payload.get("children"), list)
        )
        if looks_like_tree:
            root = tree if tree is not None else payload
            return {
                "_truncated": True,
                "_reason": reason,
                "_meta": payload.get("_meta"),
                "ok": payload.get("ok"),
                "skeleton": _skeleton(root, depth=2),
                "hint": "tree was too large; call browser_snapshot again to re-observe the page",
            }
        return {
            "_truncated": True,
            "_reason": reason,
            "keys": {k: _describe(v) for k, v in payload.items()},
        }

    if isinstance(payload, list):
        return {
            "_truncated": True,
            "_reason": f"list of {len(payload)} items, {original_len} bytes",
            "head": payload[:5],
            "tail_count": max(0, len(payload) - 5),
        }

    return {
        "_truncated": True,
        "_reason": f"scalar payload {original_len} bytes",
        "preview": str(payload)[:500],
    }


def _skeleton(node, depth: int):
    """Keep only the navigation handles + at most depth levels of children."""
    if not isinstance(node, dict):
        return node
    out = {k: node[k] for k in ("ref", "role", "tag", "name", "text") if k in node}
    children = node.get("children")
    if depth > 0 and isinstance(children, list):
        out["children"] = [_skeleton(c, depth - 1) for c in children[:10]]
        if len(children) > 10:
            out["children_truncated"] = len(children) - 10
    elif isinstance(children, list) and children:
        out["children_count"] = len(children)
    return out


def _describe(v):
    if isinstance(v, str):
        return f"<str len={len(v)}>"
    if isinstance(v, list):
        return f"<list len={len(v)}>"
    if isinstance(v, dict):
        return f"<dict keys={len(v)}>"
    return v


# ---- backwards compatibility -------------------------------------------
# The old name _compact was used by the agent loop and tests. Keep it
# as an alias so existing imports (from ...loop import _compact) can be
# redirected here without breaking callers.
_compact = compact_payload
_COMPACT_LIMIT = COMPACT_LIMIT
_FULL_DUMP_TOOLS = FULL_DUMP_TOOLS
