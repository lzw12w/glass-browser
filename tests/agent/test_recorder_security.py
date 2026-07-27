from __future__ import annotations

import json
import os
import stat
import threading
import time
from pathlib import Path

import pytest

from browser_agent.trace import Recorder, RecorderSecurityError
from browser_agent.trace import recorder as recorder_module


def _snapshot_bytes(provider: str, messages: list[dict]) -> bytes:
    header = {
        "kind": "_para_messages_header",
        "version": 1,
        "provider": provider,
        "saved_at": "2026-07-17T00:00:00.000",
        "count": len(messages),
    }
    rows = [header, *messages]
    return b"".join(
        json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n"
        for row in rows
    )


@pytest.mark.parametrize("reserved_name", ["trace.jsonl", "llm_context.jsonl"])
def test_recorder_rejects_append_log_symlink_without_touching_target(
    tmp_path: Path,
    reserved_name: str,
):
    workdir = tmp_path / "run_symlink"
    workdir.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("do not touch", encoding="utf-8")
    (workdir / reserved_name).symlink_to(target)

    with pytest.raises(RecorderSecurityError):
        Recorder(workdir)

    assert target.read_text(encoding="utf-8") == "do not touch"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unsupported")
@pytest.mark.parametrize(
    "reserved_name",
    ["trace.jsonl", "llm_context.jsonl", "messages.jsonl"],
)
def test_recorder_rejects_fifo_without_blocking(
    tmp_path: Path,
    reserved_name: str,
):
    workdir = tmp_path / "run_fifo"
    workdir.mkdir()
    os.mkfifo(workdir / reserved_name)

    started = time.perf_counter()
    with pytest.raises(RecorderSecurityError):
        Recorder(workdir)
    assert time.perf_counter() - started < 1.0


def test_recorder_rejects_hard_linked_reserved_file(tmp_path: Path):
    workdir = tmp_path / "run_hardlink"
    workdir.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("do not append", encoding="utf-8")
    os.link(target, workdir / "trace.jsonl")

    with pytest.raises(RecorderSecurityError, match="hard-linked"):
        Recorder(workdir)

    assert target.read_text(encoding="utf-8") == "do not append"


def test_recorder_fails_closed_if_bound_workdir_is_replaced(tmp_path: Path):
    workdir = tmp_path / "run_bound"
    workdir.mkdir()
    recorder = Recorder(workdir)
    moved_workdir = tmp_path / "run_bound_original"
    try:
        workdir.rename(moved_workdir)
        workdir.mkdir()

        with pytest.raises(RecorderSecurityError, match="workdir"):
            recorder.log("user", {"message": "must not escape"})

        assert not (workdir / "trace.jsonl").exists()
        assert (moved_workdir / "trace.jsonl").read_bytes() == b""
    finally:
        recorder.close()


def test_recorder_fails_closed_if_workdir_ancestor_is_replaced(tmp_path: Path):
    outer = tmp_path / "outer"
    workdir = outer / "runs" / "run_nested"
    workdir.mkdir(parents=True)
    recorder = Recorder(workdir)
    displaced_outer = tmp_path / "outer_original"
    try:
        outer.rename(displaced_outer)
        replacement = outer / "runs" / "run_nested"
        replacement.mkdir(parents=True)

        with pytest.raises(RecorderSecurityError, match="workdir"):
            recorder.log("user", {"message": "must stay in original tree"})

        assert not (replacement / "trace.jsonl").exists()
        assert (
            displaced_outer / "runs" / "run_nested" / "trace.jsonl"
        ).read_bytes() == b""
    finally:
        recorder.close()


@pytest.mark.parametrize(
    ("reserved_name", "write_log"),
    [
        ("trace.jsonl", lambda recorder: recorder.log("user", {"message": "x"})),
        (
            "llm_context.jsonl",
            lambda recorder: recorder.log_llm_request({"messages": []}),
        ),
    ],
)
def test_append_log_fails_closed_if_path_is_replaced_after_open(
    tmp_path: Path,
    reserved_name: str,
    write_log,
):
    workdir = tmp_path / f"run_replace_{reserved_name}"
    workdir.mkdir()
    outside = tmp_path / f"outside_{reserved_name}"
    outside.write_text("do not touch", encoding="utf-8")
    recorder = Recorder(workdir)
    original = workdir / f"{reserved_name}.original"
    try:
        (workdir / reserved_name).rename(original)
        (workdir / reserved_name).symlink_to(outside)

        with pytest.raises(RecorderSecurityError):
            write_log(recorder)

        assert outside.read_text(encoding="utf-8") == "do not touch"
        assert original.read_bytes() == b""
    finally:
        recorder.close()


def test_recorder_rejects_symlinked_workdir_component(tmp_path: Path):
    actual = tmp_path / "actual_run"
    actual.mkdir()
    alias = tmp_path / "run_alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(RecorderSecurityError):
        Recorder(alias)

    assert list(actual.iterdir()) == []


@pytest.mark.parametrize("attack_kind", ["symlink", "fifo"])
def test_save_messages_rejects_replaced_destination(
    tmp_path: Path,
    attack_kind: str,
):
    if attack_kind == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFO unsupported")
    workdir = tmp_path / f"run_destination_{attack_kind}"
    workdir.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("outside stays unchanged", encoding="utf-8")
    recorder = Recorder(workdir)
    try:
        destination = workdir / "messages.jsonl"
        if attack_kind == "symlink":
            destination.symlink_to(outside)
        else:
            os.mkfifo(destination)

        started = time.perf_counter()
        with pytest.raises(RecorderSecurityError):
            recorder.save_messages(
                "scripted",
                [{"role": "user", "content": "must not be written"}],
            )
        assert time.perf_counter() - started < 1.0
        assert outside.read_text(encoding="utf-8") == "outside stays unchanged"
        assert not any(
            child.name.startswith(recorder_module._MESSAGES_TEMP_PREFIX)
            for child in workdir.iterdir()
        )
    finally:
        recorder.close()


def test_failed_snapshot_serialization_preserves_previous_atomic_file(tmp_path: Path):
    workdir = tmp_path / "run_failed_snapshot"
    workdir.mkdir()
    recorder = Recorder(workdir)
    try:
        recorder.save_messages(
            "scripted",
            [{"role": "user", "content": "previous good snapshot"}],
        )
        previous = (workdir / "messages.jsonl").read_bytes()
        circular: dict = {"role": "assistant"}
        circular["content"] = circular

        with pytest.raises(ValueError, match="Circular reference"):
            recorder.save_messages("scripted", [circular])

        assert (workdir / "messages.jsonl").read_bytes() == previous
        assert not any(
            child.name.startswith(recorder_module._MESSAGES_TEMP_PREFIX)
            for child in workdir.iterdir()
        )
    finally:
        recorder.close()


def test_snapshot_commit_uses_complete_private_unpredictable_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workdir = tmp_path / "run_atomic_commit"
    workdir.mkdir()
    recorder = Recorder(workdir)
    recorder.save_messages(
        "scripted",
        [{"role": "user", "content": "old"}],
    )
    old_bytes = (workdir / "messages.jsonl").read_bytes()
    real_replace = os.replace
    observed: dict[str, object] = {}

    def inspect_then_replace(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        assert src.startswith(recorder_module._MESSAGES_TEMP_PREFIX)
        assert src != "messages.jsonl.tmp"
        assert dst == "messages.jsonl"
        temp_fd = os.open(src, os.O_RDONLY, dir_fd=src_dir_fd)
        try:
            temp_stat = os.fstat(temp_fd)
            temp_bytes = b""
            while True:
                chunk = os.read(temp_fd, 64 * 1024)
                if not chunk:
                    break
                temp_bytes += chunk
        finally:
            os.close(temp_fd)
        assert stat.S_IMODE(temp_stat.st_mode) == 0o600
        assert (workdir / "messages.jsonl").read_bytes() == old_bytes
        assert b'"count": 2' in temp_bytes
        assert temp_bytes.endswith(b"\n")
        observed["temp_name"] = src
        observed["temp_bytes"] = temp_bytes
        return real_replace(
            src,
            dst,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(recorder_module.os, "replace", inspect_then_replace)
    try:
        recorder.save_messages(
            "scripted",
            [
                {"role": "user", "content": "new"},
                {"role": "assistant", "content": "complete"},
            ],
        )
    finally:
        recorder.close()

    assert observed["temp_name"]
    provider, messages = Recorder.load_messages(workdir)
    assert provider == "scripted"
    assert [message["content"] for message in messages] == ["new", "complete"]
    assert stat.S_IMODE((workdir / "messages.jsonl").stat().st_mode) == 0o600
    assert not any(
        child.name.startswith(recorder_module._MESSAGES_TEMP_PREFIX)
        for child in workdir.iterdir()
    )


def test_reader_never_observes_partially_written_new_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workdir = tmp_path / "run_atomic_visibility"
    workdir.mkdir()
    recorder = Recorder(workdir)
    recorder.save_messages(
        "scripted",
        [{"role": "user", "content": "old"}],
    )
    old_bytes = (workdir / "messages.jsonl").read_bytes()
    write_paused = threading.Event()
    allow_write = threading.Event()
    real_write_all = recorder_module._write_all
    calls = 0

    def pause_during_temp_write(fd: int, raw: bytes):
        nonlocal calls
        calls += 1
        real_write_all(fd, raw)
        if calls == 1:
            write_paused.set()
            assert allow_write.wait(timeout=2)

    monkeypatch.setattr(recorder_module, "_write_all", pause_during_temp_write)
    error: list[BaseException] = []

    def save_new_snapshot():
        try:
            recorder.save_messages(
                "scripted",
                [
                    {"role": "user", "content": "new"},
                    {"role": "assistant", "content": "new reply"},
                ],
            )
        except BaseException as exc:  # surfaced in the assertion thread
            error.append(exc)

    worker = threading.Thread(target=save_new_snapshot)
    try:
        worker.start()
        assert write_paused.wait(timeout=2)
        assert (workdir / "messages.jsonl").read_bytes() == old_bytes
        allow_write.set()
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert error == []
        _, messages = Recorder.load_messages(workdir)
        assert [message["content"] for message in messages] == ["new", "new reply"]
    finally:
        allow_write.set()
        worker.join(timeout=1)
        recorder.close()


def test_recorder_enforces_private_permissions(tmp_path: Path):
    workdir = tmp_path / "run_permissions"
    workdir.mkdir(mode=0o755)
    for name in ("trace.jsonl", "llm_context.jsonl"):
        path = workdir / name
        path.write_text("", encoding="utf-8")
        path.chmod(0o644)

    recorder = Recorder(workdir)
    try:
        recorder.save_messages("scripted", [])
    finally:
        recorder.close()

    assert stat.S_IMODE(workdir.stat().st_mode) == 0o700
    for name in ("trace.jsonl", "llm_context.jsonl", "messages.jsonl"):
        assert stat.S_IMODE((workdir / name).stat().st_mode) == 0o600


def test_stale_temp_cleanup_is_bounded_and_ignores_symlinks(tmp_path: Path):
    workdir = tmp_path / "run_stale_temps"
    workdir.mkdir()
    old_timestamp = time.time() - recorder_module._STALE_TEMP_MAX_AGE_SECONDS - 60
    for index in range(25):
        temp_path = workdir / f"{recorder_module._MESSAGES_TEMP_PREFIX}{index:02d}"
        temp_path.write_text("abandoned", encoding="utf-8")
        os.utime(temp_path, (old_timestamp, old_timestamp))
    outside = tmp_path / "outside-temp"
    outside.write_text("keep", encoding="utf-8")
    symlink = workdir / f"{recorder_module._MESSAGES_TEMP_PREFIX}symlink"
    symlink.symlink_to(outside)

    recorder = Recorder(workdir)
    recorder.close()

    stale_regular = [
        child
        for child in workdir.iterdir()
        if child.name.startswith(recorder_module._MESSAGES_TEMP_PREFIX)
        and not child.is_symlink()
    ]
    assert len(stale_regular) == 25 - recorder_module._STALE_TEMP_REMOVE_LIMIT
    assert symlink.is_symlink()
    assert outside.read_text(encoding="utf-8") == "keep"
