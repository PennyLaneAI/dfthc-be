"""Run a command under a memory and wall-time watchdog (master plan section 12.9).

The development laptop has 10 cores and 16 GB, and the user works on it. :data:`MEMORY_CAP_MB`
(8 GB) is a sampled limit, not a hard one: macOS enforces no per-process or per-tree memory limit
(``setrlimit`` rejects lowering ``RLIMIT_AS``, ``RLIMIT_RSS`` and ``RLIMIT_DATA``, and a child then
allocates past them). :func:`run_watched` starts a command at lowered priority, samples the memory
of its whole process tree every :data:`POLL_SECONDS`, kills the tree at the first sample above
:data:`EFFECTIVE_LIMIT_MB` (7.5 GB) or when the time limit passes, and returns the sampled peak,
which is the gate metric. The memory reached past 7.5 GB is bounded by one polling interval of
growth: staying below 8 GB needs growth under 10 GB/s, and compiles grow far slower (the K-1
stand-in averaged 11 to 89 MB/s, at most about 0.5 GB/s within one 50 ms interval). Projections
and budgets use :data:`EFFECTIVE_LIMIT_MB`, the limit at which execution actually stops.

Memory is the macOS physical footprint of each process, read through ``libproc``: the agent
sandbox forbids ``ps``, and ``psutil`` is not installed. The footprint equals the resident size
while memory is free; under memory pressure it keeps counting compressed and swapped-out pages,
which the resident size drops.

The watchdog fails closed. It refuses to start anything unless libproc can read this process.
Every libproc error other than ESRCH (the process is gone) ends the run as a watchdog failure,
and the tree is killed. Processes are tracked by pid and start time, so a descendant that leaves
the tree (its parent exited) or the process group (``setsid``) is still measured and killed,
and a reused pid is never touched. The one gap: a descendant that leaves both the tree and the
process group within one polling interval of being created cannot be traced.

Command line, for heavy runs whose peak and wall time go into the ledger::

    PYTHONPATH=src python -m dfthc.testing.watchdog [--cap-mb 8192] [--timeout-min 60] -- CMD ...

It passes the command's output through, then prints one line ``WATCHDOG {json}``, and exits with
the command's status, or with 3 if the watchdog killed it or could not clean up after it.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

MEMORY_CAP_MB = 8 * 1024
"""The sampled memory limit of a process tree on the development laptop."""

SAFETY_FRACTION = 15 / 16
"""The watchdog kills at this fraction of the cap; the remaining 512 MB of an 8 GB cap absorb
growth within one polling interval below 10 GB/s."""

POLL_SECONDS = 0.05

EFFECTIVE_LIMIT_MB = SAFETY_FRACTION * MEMORY_CAP_MB
"""7,680 MB: where execution stops under the default cap, and the limit for projections and
budgets."""

HEAVY_MEMORY_MB = 4 * 1024
"""A process tree whose peak exceeds this is a heavy process."""

HEAVY_SECONDS = 30 * 60
"""A run longer than this is a heavy process."""

KILLED_EXIT_STATUS = 3

_RUSAGE_INFO_V2 = 2
_KILL_SECONDS = 5.0


class WatchdogError(RuntimeError):
    """The watchdog cannot measure or control a process tree."""


class _RusageInfoV2(ctypes.Structure):
    """``struct rusage_info_v2`` of ``<sys/resource.h>``."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
        "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
        "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
        "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
    )]


_libproc = None


def _load():
    global _libproc
    if sys.platform != "darwin":
        raise WatchdogError(f"the watchdog reads memory through macOS libproc; {sys.platform} is "
                            "not supported, so nothing was started")
    if _libproc is None:
        try:
            _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        except OSError as exc:
            raise WatchdogError(f"cannot load libproc: {exc}") from exc
    return _libproc


def check_backend() -> None:
    """Load macOS libproc and read this process through it, or raise :class:`WatchdogError`."""
    if memory_mb(os.getpid()) <= 0:
        raise WatchdogError("libproc reports no memory for this process")


def _rusage(pid: int) -> _RusageInfoV2 | None:
    """The rusage record of ``pid``, or None if no such process exists (ESRCH)."""
    info = _RusageInfoV2()
    ctypes.set_errno(0)
    if _load().proc_pid_rusage(pid, _RUSAGE_INFO_V2, ctypes.byref(info)) == 0:
        return info
    code = ctypes.get_errno()
    if code == errno.ESRCH:
        return None
    raise WatchdogError(f"proc_pid_rusage({pid}) failed: {os.strerror(code)}")


def memory_mb(pid: int) -> float:
    """Physical footprint of a running process in MB; 0.0 once it has exited."""
    info = _rusage(pid)
    return 0.0 if info is None or info.proc_exit_abstime else info.phys_footprint / 2**20


def child_pids(pid: int) -> list[int]:
    capacity = 256
    while True:
        buffer = (ctypes.c_int * capacity)()
        ctypes.set_errno(0)
        count = _load().proc_listchildpids(pid, buffer, ctypes.sizeof(buffer))
        if count < 0:
            code = ctypes.get_errno()
            if code == errno.ESRCH:
                return []
            raise WatchdogError(f"proc_listchildpids({pid}) failed: {os.strerror(code)}")
        if count < capacity:
            return [child for child in buffer[:count] if child > 0]
        capacity *= 4


class _Tree:
    """The processes of one watched command, tracked by pid and start time."""

    def __init__(self, root: int):
        self.root = root
        self.known: dict[int, int] = {}

    def scan(self) -> dict[int, float]:
        """Running processes of the tree and their footprints in MB; tracks new descendants."""
        running, stack, seen = {}, [self.root, *self.known], set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            info = _rusage(pid)
            if info is None or self.known.setdefault(pid, info.proc_start_abstime) != (
                    info.proc_start_abstime):
                self.known.pop(pid, None)  # gone, or the pid now belongs to another process
                continue
            if info.proc_exit_abstime:
                continue
            running[pid] = info.phys_footprint / 2**20
            stack.extend(child_pids(pid))
        return running

    def kill(self) -> list[int]:
        """Stop the tree until no new process appears, kill it, and return any survivors."""
        _signal_group(self.root, signal.SIGSTOP)
        stopped: set[int] = set()
        while new := set(self.scan()) - stopped:
            for pid in new:
                _signal(pid, signal.SIGSTOP)
            stopped |= new
        deadline = time.monotonic() + _KILL_SECONDS
        while True:
            _signal_group(self.root, signal.SIGKILL)
            running = self.scan()
            if not running or time.monotonic() > deadline:
                return sorted(running)
            for pid in running:
                _signal(pid, signal.SIGKILL)
            time.sleep(0.01)

    def kill_without_libproc(self) -> None:
        """Fallback when libproc fails: kill the process group and every tracked pid."""
        _signal_group(self.root, signal.SIGKILL)
        for pid in self.known:
            _signal(pid, signal.SIGKILL)


def _signal(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _signal_group(pgid: int, sig: int) -> None:
    """Signal a process group. Only a first pass: the per-pid signals and the scans decide."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):  # macOS: EPERM for a group of zombies only
        pass


def effective_limit_mb(cap_mb: float = MEMORY_CAP_MB) -> float:
    """The memory at which :func:`run_watched` kills a tree whose cap is ``cap_mb``."""
    return SAFETY_FRACTION * cap_mb


def within_effective_limit(memory: float, cap_mb: float = MEMORY_CAP_MB) -> bool:
    """Whether a measured or projected memory in MB stays below the kill point of ``cap_mb``."""
    return memory <= effective_limit_mb(cap_mb)


def tree_pids(root: int) -> list[int]:
    """``root`` and all its running descendants."""
    return sorted(_Tree(root).scan())


def tree_memory_mb(root: int) -> float:
    """Sum of the physical footprints of ``root`` and its running descendants, in MB."""
    return sum(_Tree(root).scan().values())


def lowered_priority_prefix() -> list[str]:
    """``nice -n 10`` where setpriority() is allowed, else the utility QoS clamp of taskpolicy.

    The agent sandbox forbids setpriority(); ``nice`` then only warns and runs the command
    unchanged. ``taskpolicy -c utility`` works there, and child processes inherit its clamp.
    """
    for prefix in (["nice", "-n", "10"], ["taskpolicy", "-c", "utility"]):
        try:
            probe = subprocess.run([*prefix, "true"], capture_output=True, check=False)
        except OSError:
            continue
        if probe.returncode == 0 and not probe.stderr:
            return prefix
    return []


@dataclass
class WatchedRun:
    """Outcome of :func:`run_watched`.

    ``killed`` is ``"memory"``, ``"time"``, ``"watchdog error: ..."`` or ``None``. ``survivors``
    lists processes of the tree still running after the kill; it is empty unless cleanup failed.
    """

    command: list[str]
    pid: int
    returncode: int | None
    seconds: float
    peak_mb: float
    samples: int
    killed: str | None
    survivors: list[int]
    cap_mb: float
    timeout_s: float
    priority: str
    stdout: str = field(default="", repr=False)
    stderr: str = field(default="", repr=False)

    @property
    def heavy(self) -> bool:
        return self.peak_mb > HEAVY_MEMORY_MB or self.seconds > HEAVY_SECONDS

    def summary(self) -> dict:
        record = asdict(self)
        del record["stdout"], record["stderr"]
        record.update(seconds=round(self.seconds, 1), peak_mb=round(self.peak_mb, 1),
                      heavy=self.heavy)
        return record


def run_watched(command: list[str], *, cap_mb: float = MEMORY_CAP_MB, timeout_s: float,
                capture: bool = True, poll_s: float = POLL_SECONDS, cwd=None,
                env=None) -> WatchedRun:
    """Run ``command`` at lowered priority in a new session, under the watchdog.

    Raises :class:`WatchdogError` before starting anything if libproc is unusable. The tree is
    sampled at once and then every ``poll_s`` seconds; it is killed at the first sample above
    ``effective_limit_mb(cap_mb)``, when the run passes ``timeout_s``, or when libproc fails. Any
    process of the tree still running when the command ends is killed too, however the run ends.
    With ``capture=False`` the output goes to this process's streams.
    """
    check_backend()
    prefix = lowered_priority_prefix()
    pipe = subprocess.PIPE if capture else None
    proc = subprocess.Popen([*prefix, *command], stdout=pipe, stderr=pipe, text=True, cwd=cwd,
                            env=env, start_new_session=True)
    tree = _Tree(proc.pid)
    start, peak, samples, killed, out, err = time.monotonic(), 0.0, 0, None, None, None
    try:
        while True:
            try:
                peak = max(peak, sum(tree.scan().values()))
                samples += 1
            except WatchdogError as exc:
                killed = f"watchdog error: {exc}"
                break
            if not within_effective_limit(peak, cap_mb):
                killed = "memory"
                break
            if time.monotonic() - start > timeout_s:
                killed = "time"
                break
            try:
                out, err = proc.communicate(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        try:
            survivors = tree.kill()
        except WatchdogError as exc:
            tree.kill_without_libproc()
            survivors = []
            killed = killed or f"watchdog error: {exc}"
        if out is None:
            try:
                out, err = proc.communicate(timeout=_KILL_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
    return WatchedRun(command=list(command), pid=proc.pid, returncode=proc.returncode,
                      seconds=time.monotonic() - start, peak_mb=peak, samples=samples,
                      killed=killed, survivors=survivors, cap_mb=cap_mb, timeout_s=timeout_s,
                      priority=" ".join(prefix) or "unchanged", stdout=out or "",
                      stderr=err or "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cap-mb", type=float, default=MEMORY_CAP_MB)
    parser.add_argument("--timeout-min", type=float, default=60.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")
    run = run_watched(command, cap_mb=args.cap_mb, timeout_s=args.timeout_min * 60,
                      capture=False)
    print("WATCHDOG", json.dumps(run.summary()), flush=True)
    if run.killed or run.survivors:
        return KILLED_EXIT_STATUS
    return run.returncode if run.returncode >= 0 else 128 - run.returncode


if __name__ == "__main__":
    sys.exit(main())
