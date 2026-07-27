"""Per-run working directory: screens, traces, audit logs."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path


def make_run_workdir(root: Path | None = None) -> Path:
    root = root or (Path.home() / ".browser-agent" / "runs")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = root / f"run_{ts}"
    (workdir / "screens").mkdir(parents=True, exist_ok=True)
    return workdir


def list_resumable_runs(root: Path | None = None) -> list[Path]:
    """Return run directories under ``root`` that contain a non-empty
    ``messages.jsonl``, newest first.

    Used by ``ba chat --resume`` (no-arg variant) and ``/load`` to surface
    a picker. Directories without ``messages.jsonl`` are silently skipped:
    that includes pre-resume runs and runs whose first turn never
    completed.
    """
    root = root or (Path.home() / ".browser-agent" / "runs")
    if not root.exists():
        return []
    runs: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        msgs = child / "messages.jsonl"
        try:
            if msgs.is_file() and msgs.stat().st_size > 0:
                runs.append(child)
        except OSError:
            continue
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def resolve_resume_workdir(spec: str | None, root: Path | None = None) -> Path:
    """Resolve a ``--resume`` argument to an actual workdir.

    Accepted shapes for ``spec``:
      - None / "" / "latest" : pick the most recent resumable run under ``root``
      - an absolute path     : used as-is
      - a relative path      : resolved against cwd, then against ``root``
      - a bare run name      : looked up under ``root``

    Raises ``FileNotFoundError`` with a human-friendly message when no
    matching workdir contains a ``messages.jsonl``.
    """
    root = root or (Path.home() / ".browser-agent" / "runs")
    if not spec or spec.strip().lower() == "latest":
        runs = list_resumable_runs(root)
        if not runs:
            raise FileNotFoundError(
                f"no resumable runs under {root} (need a messages.jsonl)"
            )
        return runs[0]
    candidate = Path(spec).expanduser()
    if not candidate.is_absolute():
        cwd_candidate = Path.cwd() / candidate
        if cwd_candidate.exists():
            candidate = cwd_candidate
        else:
            candidate = root / spec
    if not (candidate / "messages.jsonl").exists():
        raise FileNotFoundError(
            f"workdir has no messages.jsonl: {candidate}"
        )
    return candidate
