"""Tests for dfthc.testing.watchdog, the memory and time guard of heavy runs (plan Task 0.2)."""

import contextlib
import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import dfthc.testing.watchdog as watchdog
from dfthc.testing.watchdog import (
    KILLED_EXIT_STATUS,
    WatchdogError,
    memory_mb,
    run_watched,
    tree_memory_mb,
)

pytestmark = pytest.mark.subprocess

SRC = Path(__file__).resolve().parents[1] / "src"

# A child that starts a grandchild; the grandchild prints its pid (and then creates the file
# ``ready``, if given), then allocates random bytes in 50 MB steps and sleeps.
TREE = textwrap.dedent(
    """
    import os, subprocess, sys, time
    grandchild = '''
    import os, sys, time
    print(os.getpid(), flush=True)
    if {ready!r}:
        open({ready!r}, "w").close()
    chunks = []
    for _ in range({steps}):
        chunks.append(os.urandom(50 * 2**20))
        time.sleep(0.05)
    print("ready", flush=True)
    time.sleep({sleep})
    '''
    subprocess.run([sys.executable, "-c", grandchild], check=False)
    """
)

# The root forks a child that calls setsid() and keeps the output pipes open; the fork and the
# root's exit are microseconds apart, so no sample sees the child inside the tree.
ESCAPE = textwrap.dedent(
    """
    import os, time
    pid = os.fork()
    if pid == 0:
        os.setsid()
        time.sleep({sleep})
        os._exit(0)
    with open({pid_file!r}, "w") as file:
        file.write(str(pid))
    print(pid, flush=True)
    os._exit(0)
    """
)


def tree_command(steps, sleep=60, ready=None):
    ready = None if ready is None else str(ready)
    return [sys.executable, "-c", TREE.format(steps=steps, sleep=sleep, ready=ready)]


def wait_until_gone(pid, seconds=3.0):
    """True once ``pid`` has exited, False if it still runs after ``seconds``.

    Only ESRCH from ``kill(pid, 0)``, or libproc reporting an exited process (a zombie that its new
    parent has not reaped yet), counts as gone. Any other error propagates, so a process whose
    state cannot be verified fails the test.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if memory_mb(pid) == 0.0:
            return True
        time.sleep(0.05)
    return False


def test_tree_memory_includes_grandchildren():
    proc = subprocess.Popen(tree_command(steps=6), stdout=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        int(proc.stdout.readline())
        assert proc.stdout.readline().strip() == "ready"
        assert memory_mb(proc.pid) < 100
        assert tree_memory_mb(proc.pid) > 280
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def test_kills_the_whole_tree_at_the_memory_cap():
    run = run_watched(tree_command(steps=8), cap_mb=200, timeout_s=60)
    assert run.killed == "memory"
    assert run.peak_mb > watchdog.SAFETY_FRACTION * 200
    assert run.seconds < 30
    assert run.returncode == -9
    assert run.survivors == []
    assert wait_until_gone(int(run.stdout.split()[0])), "the grandchild survived the kill"


def test_kills_at_the_safety_fraction_of_the_cap(monkeypatch):
    assert watchdog.SAFETY_FRACTION * watchdog.MEMORY_CAP_MB == 7680
    monkeypatch.setattr(watchdog, "SAFETY_FRACTION", 0.5)
    run = run_watched(tree_command(steps=8), cap_mb=500, timeout_s=60)
    assert run.killed == "memory"
    assert 250 < run.peak_mb < 500


def test_kills_at_the_time_limit():
    run = run_watched([sys.executable, "-c", "import time; time.sleep(30)"], cap_mb=200,
                      timeout_s=1)
    assert run.killed == "time"
    assert run.seconds < 10


def test_passes_a_command_within_its_limits():
    run = run_watched(tree_command(steps=1, sleep=0.5), cap_mb=400, timeout_s=60)
    assert (run.killed, run.returncode, run.survivors) == (None, 0, [])
    assert "ready" in run.stdout
    assert 40 < run.peak_mb < 400
    assert not run.heavy


def test_samples_a_short_command_at_once():
    run = run_watched(["sleep", "0.02"], timeout_s=10)
    assert (run.killed, run.returncode) == (None, 0)
    assert run.peak_mb > 0


def test_follows_a_descendant_that_leaves_the_tree_and_the_group():
    escaper = textwrap.dedent(
        """
        import os, time
        time.sleep(1.0)
        chunks = [os.urandom(50 * 2**20) for _ in range(6)]
        time.sleep(60)
        """
    )
    root = textwrap.dedent(
        f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", {escaper!r}], start_new_session=True,
                                 stdout=subprocess.DEVNULL)
        print(child.pid, flush=True)
        time.sleep(0.5)
        """
    )
    run = run_watched([sys.executable, "-c", root], cap_mb=200, timeout_s=20)
    assert run.killed == "memory"
    assert run.survivors == []
    assert wait_until_gone(int(run.stdout.split()[0])), "the escaped descendant survived"


def test_kills_processes_spawned_during_the_kill():
    spawner = textwrap.dedent(
        """
        import subprocess, time
        for _ in range(400):
            child = subprocess.Popen(["sleep", "60"], start_new_session=True)
            print(child.pid, flush=True)
            time.sleep(0.01)
        """
    )
    run = run_watched([sys.executable, "-c", spawner], timeout_s=2)
    assert run.killed == "time"
    assert run.survivors == []
    pids = [int(line) for line in run.stdout.split()]
    assert pids, "nothing was spawned before the time limit"
    assert [pid for pid in pids if not wait_until_gone(pid)] == []


class _FailingLibproc:
    """libproc that fails with EPERM for every process but this one once ``ready`` exists."""

    def __init__(self, real, ready):
        self.real, self.ready = real, ready

    def proc_pid_rusage(self, pid, flavor, buffer):
        if pid == os.getpid() or not self.ready.exists():
            return self.real.proc_pid_rusage(pid, flavor, buffer)
        ctypes.set_errno(errno.EPERM)
        return -1

    def proc_listchildpids(self, pid, buffer, size):
        return self.real.proc_listchildpids(pid, buffer, size)


def test_libproc_failure_kills_the_tree(monkeypatch, tmp_path):
    """libproc starts failing once the grandchild has printed its pid."""
    ready = tmp_path / "ready"
    watchdog.check_backend()
    monkeypatch.setattr(watchdog, "_libproc", _FailingLibproc(watchdog._libproc, ready))
    run = run_watched(tree_command(steps=1, sleep=30, ready=ready), cap_mb=400, timeout_s=60)
    assert run.killed.startswith("watchdog error: proc_pid_rusage")
    assert run.seconds < 10
    assert run.returncode == -9
    monkeypatch.undo()
    assert wait_until_gone(int(run.stdout.split()[0])), "the grandchild survived the kill"


class _ChildListingFails:
    """libproc whose child listings fail with ``code`` for every process but this one."""

    def __init__(self, real, code):
        self.real, self.code = real, code

    def proc_pid_rusage(self, pid, flavor, buffer):
        return self.real.proc_pid_rusage(pid, flavor, buffer)

    def proc_listchildpids(self, pid, buffer, size):
        if pid == os.getpid():
            return self.real.proc_listchildpids(pid, buffer, size)
        ctypes.set_errno(self.code)
        return -1


@pytest.mark.parametrize("code, killed", [(errno.ESRCH, None), (errno.EPERM, "watchdog error")])
def test_child_listing_treats_only_esrch_as_gone(monkeypatch, code, killed):
    watchdog.check_backend()
    monkeypatch.setattr(watchdog, "_libproc", _ChildListingFails(watchdog._libproc, code))
    run = run_watched(["sleep", "0.3"], timeout_s=10)
    if killed is None:
        assert (run.killed, run.returncode) == (None, 0)
    else:
        assert run.killed.startswith("watchdog error: proc_listchildpids")


def test_projections_use_the_effective_limit():
    assert watchdog.EFFECTIVE_LIMIT_MB == watchdog.effective_limit_mb() == 7680
    assert watchdog.within_effective_limit(7.5 * 1024)
    assert not watchdog.within_effective_limit(7.8 * 1024)
    assert not watchdog.within_effective_limit(3900, cap_mb=4096)


def test_refuses_to_start_off_macos(monkeypatch, tmp_path):
    marker = tmp_path / "started"
    monkeypatch.setattr(watchdog, "_libproc", None)
    monkeypatch.setattr(watchdog.sys, "platform", "linux")
    with pytest.raises(WatchdogError, match="not supported, so nothing was started"):
        run_watched([sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"], timeout_s=10)
    monkeypatch.undo()
    time.sleep(0.5)
    assert not marker.exists()


def test_refuses_to_start_without_lowered_priority(monkeypatch, tmp_path):
    marker = tmp_path / "started"
    monkeypatch.setenv("PATH", str(tmp_path))  # neither nice nor taskpolicy can be found
    with pytest.raises(WatchdogError, match="cannot lower the priority"):
        run_watched([sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"], timeout_s=10)
    assert not marker.exists()


def test_signal_permission_error_is_a_watchdog_error():
    with pytest.raises(WatchdogError, match="cannot signal 1"):
        watchdog._signal(1, 0)  # launchd; signal 0 only checks permission


def test_cleanup_falls_back_when_a_signal_is_refused(monkeypatch):
    """A refused per-process signal during the kill ends in the fallback, not in an exception."""
    real_kill, refused = os.kill, []

    def kill(pid, sig):
        if sig == signal.SIGSTOP:
            refused.append(pid)
            raise PermissionError(errno.EPERM, "Operation not permitted")
        return real_kill(pid, sig)

    monkeypatch.setattr(watchdog.os, "kill", kill)
    try:
        run = run_watched(["sleep", "30"], timeout_s=0.5)
    finally:
        monkeypatch.undo()
        for pid in refused:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    assert refused
    assert (run.killed, run.returncode) == ("time", -9)
    assert run.pid in run.survivors  # signalled by the fallback, not verified


@pytest.mark.parametrize(
    "timeout_s, kill_seconds, suffix", [(1, 2.0, " (after time)"), (60, 1.0, "")]
)
def test_untraceable_descendant_holding_the_pipes_ends_the_run(
    monkeypatch, tmp_path, timeout_s, kill_seconds, suffix
):
    """An escaped descendant that keeps the output pipes open cannot stall the run.

    With ``timeout_s=1`` the time limit ends the loop and both bounded final reads time out; with
    ``timeout_s=60`` the loop gives up ``_KILL_SECONDS`` after the root exited with nothing left to
    sample. The escaped child sleeps 8 s, longer than either path takes.
    """
    monkeypatch.setattr(watchdog, "_KILL_SECONDS", kill_seconds)
    pid_file = tmp_path / "escaped.pid"
    command = [sys.executable, "-c", ESCAPE.format(sleep=8, pid_file=str(pid_file))]
    start = time.monotonic()
    try:
        run = run_watched(command, timeout_s=timeout_s)
        elapsed = time.monotonic() - start
    finally:
        if pid_file.exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
    assert run.killed == "watchdog error: an untraceable descendant holds the output pipes" + suffix
    assert elapsed < min(timeout_s, kill_seconds) + 2 * kill_seconds + 1
    assert run.stdout.split() == [pid_file.read_text()]


def test_command_runs_at_lowered_priority():
    code = ("import ctypes, os; print(os.getpriority(os.PRIO_PROCESS, 0), "
            "ctypes.CDLL('/usr/lib/libSystem.dylib').qos_class_self())")
    run = run_watched([sys.executable, "-c", code], timeout_s=30)
    niceness, qos_class = map(int, run.stdout.split())
    qos_utility = 0x11
    assert (run.priority == "nice -n 10" and niceness >= 10) or (
        run.priority == "taskpolicy -c utility" and qos_class == qos_utility
    ), (run.priority, niceness, qos_class)


def test_command_line_reports_a_kill():
    result = subprocess.run(
        [sys.executable, "-m", "dfthc.testing.watchdog", "--cap-mb", "200", "--timeout-min", "1",
         "--", *tree_command(steps=8)],
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(SRC)), check=False,
    )
    assert result.returncode == KILLED_EXIT_STATUS, result.stderr
    summary = json.loads(result.stdout.splitlines()[-1].removeprefix("WATCHDOG "))
    assert summary["killed"] == "memory"
    assert summary["peak_mb"] > watchdog.SAFETY_FRACTION * 200
    assert summary["cap_mb"] == 200


def test_command_line_fails_closed_before_starting(monkeypatch, capsys):
    def refuse():
        raise WatchdogError("probe")

    monkeypatch.setattr(watchdog, "check_backend", refuse)
    status = watchdog.main(["--", sys.executable, "-c", "pass"])
    line = capsys.readouterr().out.splitlines()[-1]
    assert line.startswith("WATCHDOG ")
    summary = json.loads(line.removeprefix("WATCHDOG "))
    assert status == KILLED_EXIT_STATUS
    assert summary["killed"] == "watchdog error: probe"
    assert summary["command"] == [sys.executable, "-c", "pass"]
    assert summary["timeout_s"] == 3600
