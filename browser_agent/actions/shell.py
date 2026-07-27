"""Controlled local command execution actions."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Action, ActionResult


@dataclass
class ShellPolicy:
    # ``enabled`` gates ad-hoc ``run_shell``. ``run_skill_command`` is always
    # available — it shares the same denylist / cwd / env / timeout guard
    # rails but doesn't need a separate opt-in flag.
    enabled: bool = False
    workspace_roots: tuple[Path, ...] = field(default_factory=tuple)
    timeout_seconds: int = 60
    max_output_chars: int = 12000

    @classmethod
    def disabled(cls) -> "ShellPolicy":
        return cls(enabled=False)

    def normalized_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        for root in self.workspace_roots:
            try:
                p = Path(root).expanduser().resolve()
            except OSError:
                continue
            if p.exists() and p.is_dir():
                roots.append(p)
        return tuple(roots)


# Match a banned program name regardless of how the shell reaches it:
# - bare token (sudo)
# - absolute path (/usr/bin/sudo)
# - inside a command substitution ($(sudo ...), `sudo ...`)
# - after a separator including newline, redirection, process substitution
#   `>(...)` / `<(...)`, command grouping `(...)`, or backtick.
_BANNED_PROGRAMS: tuple[tuple[str, str], ...] = (
    ("sudo", "sudo is not allowed"),
    ("su", "su is not allowed"),
    ("osascript", "GUI scripting is not allowed from run_shell"),
    ("shutdown", "shutdown is not allowed"),
    ("reboot", "reboot is not allowed"),
    ("mkfs", "filesystem formatting is not allowed"),
    ("dd", "raw disk writes are not allowed"),
)


def _banned_program_pattern(name: str) -> re.Pattern[str]:
    # Allow optional leading absolute path (e.g. /usr/bin/sudo) and require
    # word-boundary on both sides. Front context can be start-of-string,
    # whitespace, newline, or any common shell separator/grouping/quoting
    # character. This catches `$(sudo ...)`, `` `sudo ...` ``, `>(sudo ...)`,
    # `(sudo ...)`, `cmd1 ; sudo ...`, `cmd1\nsudo ...`, etc.
    return re.compile(
        rf"(?:^|[\s;&|`(<>])(?:/[^\s;&|`'\"<>()]*?/)?{re.escape(name)}\b"
    )


_DENIED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (_banned_program_pattern(name), message) for name, message in _BANNED_PROGRAMS
) + (
    # Destructive rm: any single recursive/force flag combined with a
    # dangerous target is enough. Covers `rm -r /`, `rm -R ~`, `rm -fr *`,
    # `rm --recursive /home`, `rm --force /etc`, etc.
    (
        re.compile(
            r"\brm\s+(?:"
            r"-[a-zA-Z]*[rRfFdD][a-zA-Z]*"
            r"|--recursive\b"
            r"|--force\b"
            r"|--no-preserve-root\b"
            r")[^\n;&|]*?\s+(?:/|~|\$HOME|\.\.?|\*)"
        ),
        "destructive rm target is not allowed",
    ),
)


def _inside(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _resolve_cwd(policy: ShellPolicy, cwd: str | None) -> tuple[Path | None, dict | None]:
    roots = policy.normalized_roots()
    if not roots:
        return None, {"error": "E_SHELL_POLICY", "message": "no existing shell workspace roots configured"}
    base = roots[0]
    candidate = Path(cwd).expanduser() if cwd else base
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        resolved = candidate.resolve()
    except OSError as e:
        return None, {"error": "E_BAD_CWD", "message": str(e)}
    if not resolved.exists() or not resolved.is_dir():
        return None, {"error": "E_BAD_CWD", "message": f"cwd is not a directory: {resolved}"}
    if not any(_inside(resolved, root) for root in roots):
        return None, {
            "error": "E_CWD_OUTSIDE_WORKSPACE",
            "message": f"cwd must be under one of: {[str(r) for r in roots]}",
            "cwd": str(resolved),
        }
    return resolved, None


def _validate_command(command: str) -> dict | None:
    if not command or not command.strip():
        return {"error": "E_BAD_COMMAND", "message": "command is empty"}
    if len(command) > 4000:
        return {"error": "E_BAD_COMMAND", "message": "command is too long"}
    if "\x00" in command:
        return {"error": "E_BAD_COMMAND", "message": "command contains NUL byte"}
    for pattern, message in _DENIED_PATTERNS:
        if pattern.search(command):
            return {"error": "E_COMMAND_DENIED", "message": message}
    return None


# Environment variables that can subvert command resolution or shell startup
# (PATH hijack, ld preload, custom rc files, IFS tricks, etc). Even with a
# strict key form check, these would let a caller bypass the command denylist.
_DENIED_ENV_KEYS: frozenset[str] = frozenset({
    "PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "BASH_ENV", "ENV",
    "SHELLOPTS", "BASHOPTS", "PROMPT_COMMAND", "IFS", "SHELL", "PS1", "PS4",
    "GIT_SSH_COMMAND", "GIT_EXTERNAL_DIFF",
})


def _clean_env(env: dict[str, Any] | None) -> tuple[dict[str, str] | None, dict | None]:
    if not env:
        return None, None
    out: dict[str, str] = {}
    for key, value in env.items():
        key = str(key)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            return None, {"error": "E_BAD_ENV", "message": f"invalid env key: {key}"}
        if key in _DENIED_ENV_KEYS or key.startswith("LD_") or key.startswith("DYLD_"):
            return None, {"error": "E_BAD_ENV",
                          "message": f"env key not allowed: {key}"}
        if not isinstance(value, (str, int, float, bool)):
            return None, {"error": "E_BAD_ENV", "message": f"env value for {key} must be scalar"}
        out[key] = str(value)
    return out, None


def _truncate(value: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(value) <= max_chars:
        return value, False
    return value[:max_chars].rstrip() + "\n...[truncated]", True


def run_bash_command(
    *,
    command: str,
    cwd: Path,
    timeout_seconds: int,
    env: dict[str, str] | None,
    max_output_chars: int,
) -> ActionResult:
    bash = shutil.which("bash") or "/bin/bash"
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    start = time.time()
    proc = subprocess.Popen(
        # --noprofile / --norc: do not source ~/.bash_profile or ~/.bashrc, so
        # BASH_ENV-style indirect injection has no foothold even if a future
        # caller sneaks one in.
        [bash, "--noprofile", "--norc", "-c", command],
        cwd=str(cwd),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        stdout, stderr = proc.communicate()
    duration_ms = (time.time() - start) * 1000
    stdout, stdout_truncated = _truncate(stdout or "", max_output_chars)
    stderr, stderr_truncated = _truncate(stderr or "", max_output_chars)
    data = {
        "command": command,
        "cwd": str(cwd),
        "exit_code": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "timed_out": timed_out,
    }
    if timed_out:
        return ActionResult(
            ok=False,
            data=data,
            error={"error": "E_TIMEOUT", "message": f"command exceeded {timeout_seconds}s"},
            duration_ms=duration_ms,
        )
    return ActionResult(ok=(proc.returncode == 0), data=data,
                        error=None if proc.returncode == 0 else {
                            "error": "E_EXIT_CODE",
                            "message": f"command exited with {proc.returncode}",
                        },
                        duration_ms=duration_ms)


def _execute_under_policy(
    session,
    *,
    command: str,
    cwd: str | None = None,
    timeout_seconds: int | None = None,
    env: dict[str, Any] | None = None,
    require_shell_opt_in: bool = True,
) -> ActionResult:
    policy = getattr(session, "_shell_policy", ShellPolicy.disabled())
    if require_shell_opt_in and not policy.enabled:
        return ActionResult(ok=False, error={
            "error": "E_SHELL_DISABLED",
            "message": "shell execution is disabled; start chat with --allow-shell or set BROWSER_AGENT_ENABLE_SHELL=1",
        })
    err = _validate_command(command)
    if err:
        return ActionResult(ok=False, error=err)
    resolved_cwd, err = _resolve_cwd(policy, cwd)
    if err:
        return ActionResult(ok=False, error=err)
    extra_env, err = _clean_env(env)
    if err:
        return ActionResult(ok=False, error=err)
    timeout = timeout_seconds if timeout_seconds is not None else policy.timeout_seconds
    timeout = max(1, min(int(timeout), max(1, policy.timeout_seconds)))
    return run_bash_command(
        command=command,
        cwd=resolved_cwd,
        timeout_seconds=timeout,
        env=extra_env,
        max_output_chars=max(1000, policy.max_output_chars),
    )


class RunShellAction(Action):
    name = "run_shell"
    description = (
        "Run a local bash command in a controlled workspace. Use for skills "
        "that need CLI/script execution. Shell is disabled unless explicitly "
        "enabled by the host. CWD must stay under configured workspace roots; "
        "dangerous commands are rejected; output is truncated."
    )
    idempotent = False
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to run."},
            "cwd": {"type": "string", "description": "Optional cwd, relative to the first workspace root or absolute under an allowed root."},
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
            "env": {"type": "object", "description": "Optional scalar environment overrides."},
        },
        "required": ["command"],
    }

    def _execute(self, session, *, command: str, cwd: str | None = None,
                 timeout_seconds: int | None = None, env: dict[str, Any] | None = None):
        return _execute_under_policy(
            session,
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            env=env,
            require_shell_opt_in=True,
        )


class RunSkillCommandAction(Action):
    name = "run_skill_command"
    description = (
        "Run a loaded skill's named skill.toml command, or an ad-hoc bash "
        "command in that skill's directory when no manifest command has that "
        "name. The same denylist / workspace_roots / env / timeout guard rails "
        "as ``run_shell`` still apply, but this tool does NOT require "
        "``--allow-shell``."
    )
    idempotent = False
    schema = {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "Name of a loaded skill (see skills_list).",
            },
            "command": {
                "type": "string",
                "description": (
                    "A command name declared in skill.toml, or a Bash command "
                    "when the selected skill has no command with that name."
                ),
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional extra args appended to an argv-style manifest "
                    "command. Ignored for raw Bash and command-style manifests."
                ),
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Optional cwd override. Defaults to the skill's directory. "
                    "If absolute, must lie under one of the configured "
                    "workspace_roots."
                ),
            },
            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
            "env": {"type": "object", "description": "Optional scalar environment overrides."},
        },
        "required": ["skill", "command"],
    }

    def _execute(self, session, *, skill: str, command: str,
                 args: list[str] | None = None,
                 cwd: str | None = None,
                 timeout_seconds: int | None = None,
                 env: dict[str, Any] | None = None):
        manager = getattr(session, "_skill_manager", None)
        if manager is None:
            return ActionResult(ok=False, error={
                "error": "E_SKILLS_UNAVAILABLE",
                "message": "no skill manager is attached to this session",
            })
        skill_obj = manager.get(skill) if hasattr(manager, "get") else None
        if skill_obj is None:
            available = (
                [s.name for s in manager.list_skills()]
                if hasattr(manager, "list_skills") else []
            )
            return ActionResult(ok=False, error={
                "error": "E_UNKNOWN_SKILL",
                "message": f"unknown skill: {skill}",
                "available": available,
            })
        found = (
            manager.get_command(skill, command)
            if hasattr(manager, "get_command") else None
        )
        if found is not None:
            _, manifest_command = found
            if manifest_command.command:
                shell_command = manifest_command.command
            else:
                argv = list(manifest_command.argv)
                if args:
                    argv.extend(str(arg) for arg in args)
                shell_command = " ".join(shlex.quote(part) for part in argv)
            command_cwd = cwd or manifest_command.cwd or str(skill_obj.path)
            command_timeout = (
                timeout_seconds
                if timeout_seconds is not None
                else manifest_command.timeout_seconds
            )
        else:
            shell_command = command
            command_cwd = cwd or str(skill_obj.path)
            command_timeout = timeout_seconds
        return _execute_under_policy(
            session,
            command=shell_command,
            cwd=command_cwd,
            timeout_seconds=command_timeout,
            env=env,
            require_shell_opt_in=False,
        )
