"""Typed error taxonomy for browser operations.

Agent / actions catch these to make decisions (retry, fallback, abort).
"""
from __future__ import annotations


class BrowserAgentError(Exception):
    code: str = "E_UNKNOWN"
    retriable: bool = False

    def __init__(self, message: str = "", *, detail: dict | None = None):
        super().__init__(message)
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {
            "error": self.code,
            "message": str(self),
            "retriable": self.retriable,
            "detail": self.detail,
        }


class BrowserUnavailable(BrowserAgentError):
    code = "E_BROWSER_UNAVAILABLE"
    retriable = True


class Timeout(BrowserAgentError):
    code = "E_TIMEOUT"
    retriable = True


class TargetNotFound(BrowserAgentError):
    code = "E_TARGET_NOT_FOUND"


class StaleRef(BrowserAgentError):
    """The ref points at a snapshot generation that is no longer current."""
    code = "E_STALE_REF"


class TargetAmbiguous(BrowserAgentError):
    code = "E_AMBIGUOUS"


class InvalidArgument(BrowserAgentError):
    code = "E_INVALID_ARG"


class NavigationFailed(BrowserAgentError):
    code = "E_NAVIGATION_FAILED"
    retriable = True
