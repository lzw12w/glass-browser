"""Append-only step recorder. One JSONL line per agent step.

Three files live in the workdir:

- ``trace.jsonl``       — all events (user / assistant / tool_result / session ...).
                          Append-only, human-readable audit log.
- ``llm_context.jsonl`` — the exact context object handed to the LLM each turn
                          (post view-hierarchy elision). Diagnostic only.
- ``messages.jsonl``    — provider-native message-history snapshot used by
                          the ``--resume`` flow. Rewritten in full at the end
                          of every turn so a crash mid-turn at worst loses the
                          in-progress turn (which the LLM can be asked to
                          retry). The first line is a metadata header tagging
                          the provider; subsequent lines each hold one message
                          dict in the provider's native shape.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import stat
import threading
import time
from pathlib import Path
from typing import BinaryIO


# Sentinel kind used in the messages.jsonl header line. Distinct from any
# real provider message kind so a corrupted file is easy to detect on load.
_MESSAGES_HEADER_KIND = "_browser_agent_messages_header"
# Schema version for messages.jsonl — bump if the on-disk shape changes in
# a backwards-incompatible way.
_MESSAGES_SCHEMA_VERSION = 1

_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_MESSAGES_MAX_BYTES = 64 * 1024 * 1024
_MESSAGE_MAX_BYTES = 1024 * 1024
_MESSAGES_MAX_COUNT = 100_000
_MESSAGES_TEMP_PREFIX = ".messages.jsonl.tmp-"
_TEMP_CREATE_ATTEMPTS = 8
_STALE_TEMP_MAX_AGE_SECONDS = 24 * 60 * 60
_STALE_TEMP_SCAN_LIMIT = 64
_STALE_TEMP_REMOVE_LIMIT = 16


class RecorderSecurityError(RuntimeError):
    """The recorder cannot prove that a write stays in its bound workdir."""


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        raise RecorderSecurityError("secure directory-relative I/O is unsupported")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _open_workdir_pair(workdir: Path) -> tuple[int, int, str]:
    """Open a workdir without following a symlink in any path component.

    The returned parent and workdir descriptors stay open for the lifetime of
    a Recorder.  Keeping both lets every later write prove that the original
    directory is still bound to the caller-visible workdir name.
    """

    path = Path(workdir)
    parts = path.parts
    if path.is_absolute():
        components = parts[1:]
        current_fd = os.open(os.sep, _directory_open_flags())
    else:
        components = parts
        current_fd = os.open(".", _directory_open_flags())
    if not components or any(part in {"", ".", ".."} for part in components):
        os.close(current_fd)
        raise RecorderSecurityError("workdir must name a concrete directory")

    try:
        for component in components[:-1]:
            next_fd = os.open(
                component,
                _directory_open_flags(),
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        workdir_name = components[-1]
        workdir_fd = os.open(
            workdir_name,
            _directory_open_flags(),
            dir_fd=current_fd,
        )
        return current_fd, workdir_fd, workdir_name
    except Exception:
        os.close(current_fd)
        raise


def _identity(file_stat: os.stat_result) -> tuple[int, int]:
    return int(file_stat.st_dev), int(file_stat.st_ino)


def _snapshot_identity(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(file_stat.st_dev),
        int(file_stat.st_ino),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
    )


def _assert_owned(file_stat: os.stat_result, *, label: str) -> None:
    get_euid = getattr(os, "geteuid", None)
    if get_euid is not None and file_stat.st_uid != get_euid():
        raise RecorderSecurityError(f"{label} is not owned by the current user")


def _write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short recorder write")
        view = view[written:]


class Recorder:
    def __init__(
        self,
        workdir: Path,
        *,
        expected_workdir_identity: tuple[int, int] | None = None,
        expected_messages_identity: tuple[int, int, int, int, int] | None = None,
    ):
        self.workdir = Path(workdir)
        self._binding_path = (
            self.workdir
            if self.workdir.is_absolute()
            else Path.cwd() / self.workdir
        )
        self.path = self.workdir / "trace.jsonl"
        self.llm_context_path = self.workdir / "llm_context.jsonl"
        self.messages_path = self.workdir / "messages.jsonl"
        self._lock = threading.RLock()
        self._closed = False
        self._parent_fd = -1
        self._workdir_fd = -1
        self._workdir_name = ""
        self._fp: BinaryIO | None = None
        self._llm_context_fp: BinaryIO | None = None

        try:
            try:
                self._parent_fd, self._workdir_fd, self._workdir_name = (
                    _open_workdir_pair(self._binding_path)
                )
            except OSError as exc:
                raise RecorderSecurityError("cannot securely open workdir") from exc
            opened_stat = os.fstat(self._workdir_fd)
            if not stat.S_ISDIR(opened_stat.st_mode):
                raise RecorderSecurityError("workdir is not a directory")
            _assert_owned(opened_stat, label="workdir")
            self._workdir_identity = _identity(opened_stat)
            if (expected_workdir_identity is not None
                    and tuple(expected_workdir_identity) != self._workdir_identity):
                raise RecorderSecurityError("workdir changed after history was read")
            os.fchmod(self._workdir_fd, _PRIVATE_DIRECTORY_MODE)
            if stat.S_IMODE(os.fstat(self._workdir_fd).st_mode) != _PRIVATE_DIRECTORY_MODE:
                raise RecorderSecurityError("workdir permissions are not private")
            self.assert_workdir_binding()

            # Validate every reserved entry before creating either append log,
            # so a pre-positioned symlink/FIFO fails without being followed.
            self._harden_existing_regular(
                "messages.jsonl",
                expected_snapshot_identity=expected_messages_identity,
            )
            for name in ("trace.jsonl", "llm_context.jsonl"):
                self._harden_existing_regular(name)
            self._cleanup_stale_message_temps()
            self._fp = self._open_append_file("trace.jsonl")
            self._llm_context_fp = self._open_append_file("llm_context.jsonl")
        except Exception:
            self.close()
            raise

    @property
    def workdir_identity(self) -> tuple[int, int]:
        return self._workdir_identity

    def assert_workdir_binding(self) -> None:
        """Fail if the original run directory was renamed or replaced."""

        if self._closed or self._parent_fd < 0 or self._workdir_fd < 0:
            raise RecorderSecurityError("recorder is closed")
        try:
            named_stat = os.stat(
                self._workdir_name,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
            opened_stat = os.fstat(self._workdir_fd)
        except OSError as exc:
            raise RecorderSecurityError("workdir binding is no longer available") from exc
        if (not stat.S_ISDIR(named_stat.st_mode)
                or _identity(named_stat) != self._workdir_identity
                or _identity(opened_stat) != self._workdir_identity):
            raise RecorderSecurityError("workdir was replaced after recorder opened it")

        # The retained parent descriptor protects writes even if a higher path
        # component is renamed. Re-open the complete no-symlink chain as well
        # so such an ancestor swap is detected rather than silently writing to
        # a now-detached run directory.
        fresh_parent_fd = -1
        fresh_workdir_fd = -1
        try:
            fresh_parent_fd, fresh_workdir_fd, _ = _open_workdir_pair(
                self._binding_path
            )
            fresh_stat = os.fstat(fresh_workdir_fd)
            if _identity(fresh_stat) != self._workdir_identity:
                raise RecorderSecurityError("workdir path now resolves to another directory")
        except OSError as exc:
            raise RecorderSecurityError("workdir path is no longer securely resolvable") from exc
        finally:
            for descriptor in (fresh_workdir_fd, fresh_parent_fd):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass

    def _assert_regular_stat(self, file_stat: os.stat_result, *, label: str) -> None:
        if not stat.S_ISREG(file_stat.st_mode):
            raise RecorderSecurityError(f"{label} is not a regular file")
        if file_stat.st_nlink != 1:
            raise RecorderSecurityError(f"{label} must not be hard-linked")
        _assert_owned(file_stat, label=label)

    def _assert_named_entry_matches(
        self,
        name: str,
        file_stat: os.stat_result,
    ) -> None:
        try:
            named_stat = os.stat(
                name,
                dir_fd=self._workdir_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise RecorderSecurityError(f"{name} is no longer bound") from exc
        self._assert_regular_stat(named_stat, label=name)
        if _identity(named_stat) != _identity(file_stat):
            raise RecorderSecurityError(f"{name} was replaced")

    def _harden_existing_regular(
        self,
        name: str,
        *,
        expected_snapshot_identity: tuple[int, int, int, int, int] | None = None,
    ) -> os.stat_result | None:
        flags = (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | os.O_CLOEXEC
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            fd = os.open(name, flags, dir_fd=self._workdir_fd)
        except FileNotFoundError:
            if expected_snapshot_identity is not None:
                raise RecorderSecurityError(
                    "messages snapshot changed after history was read"
                )
            return None
        except OSError as exc:
            raise RecorderSecurityError(f"unsafe recorder entry: {name}") from exc
        try:
            opened_stat = os.fstat(fd)
            self._assert_regular_stat(opened_stat, label=name)
            self._assert_named_entry_matches(name, opened_stat)
            if (expected_snapshot_identity is not None
                    and tuple(expected_snapshot_identity)
                    != _snapshot_identity(opened_stat)):
                raise RecorderSecurityError(
                    "messages snapshot changed after history was read"
                )
            os.fchmod(fd, _PRIVATE_FILE_MODE)
            if stat.S_IMODE(os.fstat(fd).st_mode) != _PRIVATE_FILE_MODE:
                raise RecorderSecurityError(f"{name} permissions are not private")
            return opened_stat
        finally:
            os.close(fd)

    def _open_append_file(self, name: str) -> BinaryIO:
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | os.O_NOFOLLOW
            | os.O_CLOEXEC
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            fd = os.open(name, flags, _PRIVATE_FILE_MODE, dir_fd=self._workdir_fd)
        except OSError as exc:
            raise RecorderSecurityError(f"cannot securely open {name}") from exc
        try:
            opened_stat = os.fstat(fd)
            self._assert_regular_stat(opened_stat, label=name)
            self._assert_named_entry_matches(name, opened_stat)
            os.fchmod(fd, _PRIVATE_FILE_MODE)
            if stat.S_IMODE(os.fstat(fd).st_mode) != _PRIVATE_FILE_MODE:
                raise RecorderSecurityError(f"{name} permissions are not private")
            return os.fdopen(fd, "ab", buffering=0, closefd=True)
        except Exception:
            os.close(fd)
            raise

    def _validate_append_binding(self, name: str, fp: BinaryIO) -> None:
        self.assert_workdir_binding()
        opened_stat = os.fstat(fp.fileno())
        self._assert_regular_stat(opened_stat, label=name)
        self._assert_named_entry_matches(name, opened_stat)

    def _cleanup_stale_message_temps(self) -> None:
        """Remove only a small, age-gated set of our own abandoned temps."""

        now = time.time()
        removed = 0
        try:
            entries = os.scandir(self._workdir_fd)
        except OSError:
            return
        with entries:
            for scanned, entry in enumerate(entries, start=1):
                if scanned > _STALE_TEMP_SCAN_LIMIT or removed >= _STALE_TEMP_REMOVE_LIMIT:
                    break
                if not entry.name.startswith(_MESSAGES_TEMP_PREFIX):
                    continue
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                    if (not stat.S_ISREG(entry_stat.st_mode)
                            or entry_stat.st_nlink != 1
                            or now - entry_stat.st_mtime < _STALE_TEMP_MAX_AGE_SECONDS):
                        continue
                    _assert_owned(entry_stat, label=entry.name)
                    os.unlink(entry.name, dir_fd=self._workdir_fd)
                    removed += 1
                except (OSError, RecorderSecurityError):
                    continue

    def log(self, kind: str, payload: dict):
        self._write(self._fp, "trace.jsonl", kind, payload)

    def log_llm_request(self, payload: dict):
        self._write(
            self._llm_context_fp,
            "llm_context.jsonl",
            "llm_request",
            payload,
        )

    def _write(self, fp: BinaryIO | None, name: str, kind: str, payload: dict):
        record = {
            "ts": _dt.datetime.now().isoformat(timespec="milliseconds"),
            "kind": kind,
            **payload,
        }
        raw = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            if self._closed or fp is None:
                raise RecorderSecurityError("recorder is closed")
            self._validate_append_binding(name, fp)
            _write_all(fp.fileno(), raw)

    # ---- messages.jsonl persistence ---------------------------------------
    def save_messages(self, provider_id: str, messages: list[dict]) -> None:
        """Atomically rewrite ``messages.jsonl`` with the current history.

        Snapshot-on-write (vs. append-on-mutate) is intentional: messages are
        mutated in batches per turn, the list is small, and a full rewrite
        keeps the file always coherent — there's no half-written turn to
        reconcile when ``--resume`` reads it back. We write to a sibling
        temp file then ``replace()`` so a crash mid-write never corrupts
        the previous good snapshot.
        """
        if (not isinstance(provider_id, str)
                or not provider_id
                or provider_id != provider_id.strip()
                or len(provider_id) > 64
                or any(ord(char) < 32 or ord(char) == 127 for char in provider_id)):
            raise ValueError("invalid messages provider")
        if len(messages) > _MESSAGES_MAX_COUNT:
            raise ValueError("too many messages to persist")
        header = {
            "kind": _MESSAGES_HEADER_KIND,
            "version": _MESSAGES_SCHEMA_VERSION,
            "provider": provider_id,
            "saved_at": _dt.datetime.now().isoformat(timespec="milliseconds"),
            "count": len(messages),
        }
        with self._lock:
            if self._closed:
                raise RecorderSecurityError("recorder is closed")
            self.assert_workdir_binding()
            self._harden_existing_regular("messages.jsonl")
            self._cleanup_stale_message_temps()

            tmp_name = ""
            tmp_fd = -1
            tmp_stat: os.stat_result | None = None
            try:
                for _ in range(_TEMP_CREATE_ATTEMPTS):
                    candidate = f"{_MESSAGES_TEMP_PREFIX}{secrets.token_hex(16)}"
                    try:
                        tmp_fd = os.open(
                            candidate,
                            os.O_WRONLY
                            | os.O_CREAT
                            | os.O_EXCL
                            | os.O_NOFOLLOW
                            | os.O_CLOEXEC
                            | getattr(os, "O_NONBLOCK", 0),
                            _PRIVATE_FILE_MODE,
                            dir_fd=self._workdir_fd,
                        )
                        tmp_name = candidate
                        break
                    except FileExistsError:
                        continue
                if tmp_fd < 0:
                    raise RecorderSecurityError("cannot allocate a private messages temp file")

                tmp_stat = os.fstat(tmp_fd)
                self._assert_regular_stat(tmp_stat, label=tmp_name)
                self._assert_named_entry_matches(tmp_name, tmp_stat)
                os.fchmod(tmp_fd, _PRIVATE_FILE_MODE)

                total_bytes = 0
                raw_header = (json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8")
                if len(raw_header) > _MESSAGE_MAX_BYTES:
                    raise ValueError("messages header is too large")
                _write_all(tmp_fd, raw_header)
                total_bytes += len(raw_header)
                for msg in messages:
                    raw_message = (
                        json.dumps(msg, ensure_ascii=False, default=str) + "\n"
                    ).encode("utf-8")
                    if len(raw_message) > _MESSAGE_MAX_BYTES:
                        raise ValueError("one persisted message is too large")
                    if total_bytes + len(raw_message) > _MESSAGES_MAX_BYTES:
                        raise ValueError("messages snapshot is too large")
                    _write_all(tmp_fd, raw_message)
                    total_bytes += len(raw_message)
                os.fsync(tmp_fd)
                final_tmp_stat = os.fstat(tmp_fd)
                if (_identity(final_tmp_stat) != _identity(tmp_stat)
                        or final_tmp_stat.st_size != total_bytes
                        or stat.S_IMODE(final_tmp_stat.st_mode) != _PRIVATE_FILE_MODE):
                    raise RecorderSecurityError("messages temp file changed while writing")
                self._assert_named_entry_matches(tmp_name, final_tmp_stat)
                os.close(tmp_fd)
                tmp_fd = -1

                # Revalidate both the directory binding and destination at the
                # commit boundary. os.replace(..., dir_fd=...) then changes one
                # name atomically without ever resolving an external path.
                self.assert_workdir_binding()
                self._harden_existing_regular("messages.jsonl")
                os.replace(
                    tmp_name,
                    "messages.jsonl",
                    src_dir_fd=self._workdir_fd,
                    dst_dir_fd=self._workdir_fd,
                )
                self._assert_named_entry_matches("messages.jsonl", final_tmp_stat)
                os.fsync(self._workdir_fd)
                tmp_name = ""
            finally:
                if tmp_fd >= 0:
                    try:
                        os.close(tmp_fd)
                    except OSError:
                        pass
                if tmp_name:
                    try:
                        os.unlink(tmp_name, dir_fd=self._workdir_fd)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass

    @staticmethod
    def load_messages(workdir: Path) -> tuple[str, list[dict]]:
        """Read ``messages.jsonl`` from a workdir; return (provider_id, messages).

        Raises ``FileNotFoundError`` when the file is missing and
        ``ValueError`` on malformed content (missing header / version mismatch).
        Callers handle both — typically by surfacing a friendly error to the
        user.
        """
        path = Path(workdir) / "messages.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"no messages.jsonl in {workdir}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"messages.jsonl is empty: {path}")
        try:
            header = json.loads(lines[0])
        except json.JSONDecodeError as e:
            raise ValueError(f"bad header line in {path}: {e}") from e
        if header.get("kind") != _MESSAGES_HEADER_KIND:
            raise ValueError(f"missing header in {path}; first line was {header!r}")
        version = header.get("version")
        if version != _MESSAGES_SCHEMA_VERSION:
            raise ValueError(
                f"messages.jsonl schema {version} unsupported (need {_MESSAGES_SCHEMA_VERSION})"
            )
        provider = header.get("provider", "")
        if not provider:
            raise ValueError(f"header missing provider in {path}")
        messages: list[dict] = []
        for ln in lines[1:]:
            if not ln.strip():
                continue
            messages.append(json.loads(ln))
        return provider, messages

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for fp in (self._fp, self._llm_context_fp):
                if fp is None:
                    continue
                try:
                    fp.close()
                except Exception:
                    pass
            self._fp = None
            self._llm_context_fp = None
            for descriptor_name in ("_workdir_fd", "_parent_fd"):
                descriptor = getattr(self, descriptor_name, -1)
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                    setattr(self, descriptor_name, -1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
