"""Action framework. Each action declares JSON schema and executes against a session."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..errors import BrowserAgentError


@dataclass
class ActionResult:
    ok: bool
    data: Any = None
    error: dict | None = None
    artifacts: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        d = {
            "ok": self.ok,
            "duration_ms": round(self.duration_ms, 1),
        }
        if self.data is not None: d["data"] = self.data
        if self.error: d["error"] = self.error
        if self.artifacts: d["artifacts"] = self.artifacts
        if self.notes: d["notes"] = self.notes
        return d


class Action:
    name: str = "base"
    description: str = ""
    schema: dict = {"type": "object", "properties": {}}
    idempotent: bool = False

    # D6.6 - action attribution.
    # Subclasses that meaningfully change UI state (tap, scroll, modify, ...)
    # set mutates_ui = True so the base run() informs the knowledge observer
    # BEFORE _execute(); the observer's ring buffer can then attribute the
    # next page commit to this action (instead of recording "unknown").
    # Read-only actions (vc_hierarchy, screenshot, recall_page_context, ...)
    # leave it False to avoid polluting the attribution log.
    mutates_ui: bool = False

    # D6.9 - content-blind edge keys.
    # ``identity_param_keys`` lists which kwargs are STRUCTURAL (direction,
    # mode enums, axis) and thus part of the action's identity for graph
    # dedup. All other kwargs (text, accessibility_id, address, x, y, url,
    # label, ...) are stored verbatim in ``action_params_json`` for human
    # inspection but DO NOT contribute to ``action_params_hash``. This way
    # "tap on cell #1 in the Feed" and "tap on cell #7 in the Feed" collapse
    # into the same edge instead of multiplying into N indistinguishable rows.
    #
    # Default = empty: only the action ``name`` matters. Subclasses that have
    # genuinely orthogonal modes (e.g. ScrollAction.direction) override this.
    identity_param_keys: tuple[str, ...] = ()

    @classmethod
    def _identity_params(cls, kwargs: dict) -> dict:
        """Subset of kwargs participating in the edge unique key.

        Returned dict is JSON-serializable and has only structural values.
        Non-listed keys are silently dropped from the identity view; they are
        still preserved by the caller for the human-readable params_json.
        """
        if not cls.identity_param_keys:
            return {}
        out: dict = {}
        for k in cls.identity_param_keys:
            if k in kwargs and kwargs[k] is not None:
                v = kwargs[k]
                if isinstance(v, (str, int, float, bool)):
                    out[k] = v
                # Anything else (lists, dicts, custom objects) is treated as
                # opaque content and dropped -- keeping identity simple.
        return out

    def _attribute_to_observer(self, session, kwargs: dict) -> None:
        """Push this action onto the observer's ring buffer.

        Centralized here so adding a new mutating action only requires setting
        ``mutates_ui = True`` -- no need to remember to wire each subclass
        individually. Failures NEVER propagate; observer attribution is a
        nice-to-have, not a correctness concern.
        """
        if not self.mutates_ui:
            return
        observer = getattr(session, "_observer", None)
        if observer is None:
            return
        try:
            # Snapshot params shallowly. We drop None values and any
            # non-jsonable objects (callbacks, sessions) so the params stored
            # in the transitions table stay portable. ``identity`` carries
            # ONLY the structural subset that participates in edge dedup --
            # see ``identity_param_keys`` on each Action subclass.
            safe = {k: v for k, v in kwargs.items()
                    if v is not None and isinstance(v, (str, int, float, bool, list, tuple, dict))}
            identity = self._identity_params(kwargs)
            # Pass the structural-identity subset as a side channel: the
            # observer stores it on the ActionEvent without contaminating the
            # human-visible ``params`` dict that callers/tests inspect.
            observer.record_action(self.name, safe, identity=identity)
        except Exception:  # pragma: no cover - safety net
            pass

    def _record_failure_if_attributed(self, session,
                                      kwargs: dict, error: str) -> None:
        """Mirror of _attribute_to_observer for the failure path.

        We pop the most recent action (which we just pushed) and record a
        self-loop failure edge, so the graph captures "this action did NOT
        change page" -- invaluable for the planner's edge weights.
        """
        if not self.mutates_ui:
            return
        observer = getattr(session, "_observer", None)
        if observer is None:
            return
        try:
            evt = observer.action_log.pop_latest()
            if evt is None:
                return
            observer.record_failed_transition(evt, error)
        except Exception:  # pragma: no cover - safety net
            pass

    def run(self, session, **kwargs) -> ActionResult:
        start = time.time()
        # Attribute BEFORE execution so we have an issued_ms timestamp for
        # latency calculation even if the action throws.
        self._attribute_to_observer(session, kwargs)
        try:
            data = self._execute(session, **kwargs)
        except BrowserAgentError as e:
            self._record_failure_if_attributed(session, kwargs, str(e))
            return ActionResult(ok=False, error=e.to_dict(),
                                duration_ms=(time.time() - start) * 1000)
        except Exception as e:
            self._record_failure_if_attributed(session, kwargs, str(e))
            return ActionResult(ok=False,
                                error={"error": "E_UNEXPECTED", "message": str(e)},
                                duration_ms=(time.time() - start) * 1000)
        if isinstance(data, ActionResult):
            data.duration_ms = (time.time() - start) * 1000
            # If the action returned a failed ActionResult (no exception),
            # still record the failure edge.
            if not data.ok:
                self._record_failure_if_attributed(
                    session, kwargs, (data.error or {}).get("message", ""))
            else:
                self._post_action_observe(session)
            return data
        result = ActionResult(ok=True, data=data,
                              duration_ms=(time.time() - start) * 1000)
        self._post_action_observe(session)
        return result

    def _post_action_observe(self, session) -> None:
        """Eagerly drive the knowledge observer after a successful mutating action.

        Without this hook, page transitions caused by tap / back / switch_tab /
        open_url / scroll only get committed when the LLM happens to call
        browser_snapshot on its next turn -- by which point the action may have
        aged past the staleness window, causing the edge to be recorded as
        ``unattributed_stale``. Driving the observer right after the UI mutates
        keeps every transition correctly attributed.

        Read-only actions (mutates_ui = False) skip this entirely so that
        listing the view tree, fetching the VC hierarchy, taking a screenshot
        and similar operations do NOT incur the extra round-trip.
        """
        if not self.mutates_ui:
            return
        hook = getattr(session, "_observe_after_action", None)
        if hook is None:
            return
        try:
            hook(self.name)
        except Exception:  # pragma: no cover - safety net
            pass

    def _execute(self, session, **kwargs) -> Any:
        raise NotImplementedError

    @classmethod
    def to_tool_schema(cls) -> dict:
        return {
            "name": cls.name,
            "description": cls.description.strip(),
            "input_schema": cls.schema,
        }
