"""Transient-error retry for LLM streaming calls.

Retries network/rate-limit/server errors (429, 5xx, connection drops,
timeouts) when establishing or starting an LLM stream. The retry happens
only **before** any chunk has been emitted to the caller — once the
provider has handed us text/tool deltas, replaying would either duplicate
output or drop reasoning state, so the error propagates.

Backoff: exponential with jitter, honoring ``Retry-After`` when the
provider attaches it to the exception.
"""
from __future__ import annotations

import os
import random
import time
from typing import Any, Callable, Iterable, Iterator


# Tunable via env / kwargs. Defaults aim for ~30s total wall time on a
# fully unhealthy backend (1 + 2 + 4 + 8 + 16 ≈ 31s, capped per-attempt).
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 16.0


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

# Error-class name fragments that are always transient regardless of status.
_RETRYABLE_NAME_FRAGMENTS = (
    "RateLimit",
    "Overloaded",
    "APIConnection",
    "APITimeout",
    "Timeout",
    "Connection",
    "ServiceUnavailable",
    "InternalServer",
)


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "status", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        v = getattr(resp, "status_code", None)
        if isinstance(v, int):
            return v
    return None


def is_retryable(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code is not None and code in _RETRYABLE_STATUS:
        return True
    name = type(exc).__name__
    if any(frag in name for frag in _RETRYABLE_NAME_FRAGMENTS):
        return True
    # Last-resort match on message — many SDK gateways wrap rate-limit
    # responses in a generic APIStatusError whose only signal is the text.
    msg = str(exc).lower()
    if "rate limit" in msg or "too many requests" in msg or "429" in msg:
        return True
    if "overloaded" in msg or "service unavailable" in msg:
        return True
    return False


def _retry_after_seconds(exc: BaseException) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _sleep_for(attempt: int, exc: BaseException, *,
               base: float, cap: float) -> float:
    hint = _retry_after_seconds(exc)
    if hint is not None:
        return min(hint, cap * 2)
    delay = min(cap, base * (2 ** attempt))
    # Full jitter — avoids thundering-herd on shared-key gateways.
    return random.uniform(0.0, delay)


class RetryConfig:
    __slots__ = ("max_retries", "base_delay", "max_delay", "on_retry")

    def __init__(
        self,
        max_retries: int | None = None,
        base_delay: float | None = None,
        max_delay: float | None = None,
        on_retry: Callable[[int, BaseException, float], None] | None = None,
    ):
        self.max_retries = (
            max_retries
            if max_retries is not None
            else int(os.environ.get("BROWSER_AGENT_LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES))
        )
        self.base_delay = (
            base_delay
            if base_delay is not None
            else float(os.environ.get("BROWSER_AGENT_LLM_RETRY_BASE", DEFAULT_BASE_DELAY))
        )
        self.max_delay = (
            max_delay
            if max_delay is not None
            else float(os.environ.get("BROWSER_AGENT_LLM_RETRY_CAP", DEFAULT_MAX_DELAY))
        )
        self.on_retry = on_retry


def retry_stream(
    stream_factory: Callable[[], Iterable[Any]],
    config: RetryConfig | None = None,
) -> Iterator[Any]:
    """Iterate a streaming call, retrying transient failures that occur
    before any item has been yielded.

    ``stream_factory`` is invoked once per attempt and must return a fresh
    iterable (e.g. by re-entering the SDK's stream context manager).
    """
    cfg = config or RetryConfig()
    attempt = 0
    while True:
        emitted = False
        try:
            for item in stream_factory():
                emitted = True
                yield item
            return
        except BaseException as exc:
            if emitted:
                raise
            if not is_retryable(exc):
                raise
            if attempt >= cfg.max_retries:
                raise
            wait = _sleep_for(attempt, exc, base=cfg.base_delay, cap=cfg.max_delay)
            if cfg.on_retry is not None:
                try:
                    cfg.on_retry(attempt + 1, exc, wait)
                except Exception:
                    pass
            time.sleep(wait)
            attempt += 1
