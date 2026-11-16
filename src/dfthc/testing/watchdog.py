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

The watchdog fails closed. It refuses to start anything unless libproc can read this process and
the priority can be lowered. Every libproc error other than ESRCH (the process is gone) ends the
run as a watchdog failure, and the tree is killed; if libproc fails or a signal is refused during
the kill, the process group and every tracked pid get SIGKILL without verification. Processes are
tracked by pid and start time, so a descendant that leaves the tree (its parent exited) or the
process group (``setsid``) is still measured and killed; the start-time check confines the risk
of signalling a reused pid to the few milliseconds between a scan and a signal, and the root's pid
is retired once the root is reaped, so a process that reuses it is never adopted. The one gap: a
descendant that leaves both the tree and the process group within one polling interval of being
created cannot be traced. If such a descendant keeps the output pipes open, the run ends as a
watchdog error 5 s after nothing traceable is left, and every final read of the output is bounded,
so the descendant cannot stall the watchdog.

Command line, for heavy runs whose peak and wall time go into the ledger::

    PYTHONPATH=src python -m dfthc.testing.watchdog [--cap-mb 8192] [--timeout-min 60] -- CMD ...

It passes the command's output through, then prints one line ``WATCHDOG {json}``, and exits with
the command's status, or with 3 if the watchdog killed it or could not clean up after it.
"""

from __future__ import annotations

import argparse
import contextlib
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
_PIPES_HELD = "watchdog error: an untraceable descendant holds the output pipes"


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
        self.root_retired = False

    def scan(self) -> dict[int, float]:
        """Running processes of the tree and their footprints in MB; tracks new descendants. The
        root's pid is retired once the root is gone, so a process that reuses it is not adopted."""
        running, seen = {}, set()
        stack = [*([] if self.root_retired else [self.root]), *self.known]
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            info = _rusage(pid)
            if info is None or self.known.setdefault(pid, info.proc_start_abstime) != (
                    info.proc_start_abstime):
                self.known.pop(pid, None)  # gone, or the pid now belongs to another process
                self.root_retired |= pid == self.root
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

    def kill_without_libproc(self) -> list[int]:
        """Fallback when libproc fails or a signal is refused: SIGKILL the process group and every
        tracked pid, and return the pids signalled. Nothing verifies that they are gone."""
        _signal_group(self.root, signal.SIGKILL)
        for pid in self.known:
            with contextlib.suppress(WatchdogError):
                _signal(pid, signal.SIGKILL)
        return sorted({self.root, *self.known})


def _signal(pid: int, sig: int) -> None:
    """Signal one process: a vanished process is ignored, a refused signal is a WatchdogError."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise WatchdogError(f"cannot signal {pid}: {exc.strerror}") from exc


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
    Raises :class:`WatchdogError` if neither works: like ``scripts/test.sh``, the watchdog never
    runs a command at normal priority.
    """
    for prefix in (["nice", "-n", "10"], ["taskpolicy", "-c", "utility"]):
        try:
            probe = subprocess.run([*prefix, "true"], capture_output=True, check=False)
        except OSError:
            continue
        if probe.returncode == 0 and not probe.stderr:
            return prefix
    raise WatchdogError("cannot lower the priority: neither nice -n 10 nor taskpolicy -c utility "
                        "works, so nothing was started")


@dataclass
class WatchedRun:
    """Outcome of :func:`run_watched`.

    ``killed`` is ``"memory"``, ``"time"``, ``"watchdog error: ..."`` or ``None``. When a
    descendant the watchdog could not trace kept the output pipes open, it is
    ``"watchdog error: an untraceable descendant holds the output pipes"``, followed by
    ``" (after <earlier reason>)"`` if the run had already been stopped for another reason.

    ``survivors`` lists the processes of the tree still running after a verified kill. If libproc
    failed or a signal was refused during the kill, the fallback sends SIGKILL to the process group
    and every tracked pid, and ``survivors`` lists those pids unverified: some may already be gone.
    Either way the cleanup is confirmed only when ``survivors`` is empty.
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

    Raises :class:`WatchdogError` before starting anything if libproc is unusable or the priority
    cannot be lowered. The tree is sampled at once and then every ``poll_s`` seconds; it is killed
    at the first sample above ``effective_limit_mb(cap_mb)``, when the run passes ``timeout_s``,
    when libproc fails, or 5 s after the command exited while its output pipes stay open with no
    traceable process left. Any process of the tree still running when the command ends is killed
    too, however the run ends. With ``capture=False`` the output goes to this process's streams.
    """
    check_backend()
    prefix = lowered_priority_prefix()
    pipe = subprocess.PIPE if capture else None
    proc = subprocess.Popen([*prefix, *command], stdout=pipe, stderr=pipe, text=True, cwd=cwd,
                            env=env, start_new_session=True)
    tree = _Tree(proc.pid)
    start, peak, samples, killed, out, err = time.monotonic(), 0.0, 0, None, None, None
    untraced_since = None
    try:
        while True:
            try:
                running = tree.scan()
                samples += 1
            except WatchdogError as exc:
                killed = f"watchdog error: {exc}"
                break
            peak = max(peak, sum(running.values()))
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
            if proc.poll() is None or running:
                untraced_since = None
            elif untraced_since is None:
                untraced_since = time.monotonic()
            elif time.monotonic() - untraced_since > _KILL_SECONDS:
                killed = _PIPES_HELD
                break
    finally:
        try:
            survivors = tree.kill()
        except WatchdogError as exc:
            survivors = tree.kill_without_libproc()
            killed = killed or f"watchdog error: {exc}"
        if out is None:
            out, err, pipes_held = _final_output(proc, held=killed == _PIPES_HELD)
            if pipes_held and killed != _PIPES_HELD:
                killed = _PIPES_HELD + (f" (after {killed})" if killed else "")
    return WatchedRun(command=list(command), pid=proc.pid, returncode=proc.returncode,
                      seconds=time.monotonic() - start, peak_mb=peak, samples=samples,
                      killed=killed, survivors=survivors, cap_mb=cap_mb, timeout_s=timeout_s,
                      priority=" ".join(prefix), stdout=out or "", stderr=err or "")


def _final_output(proc: subprocess.Popen, *, held: bool = False) -> tuple[str, str, bool]:
    """The rest of the output, read with bounded waits, and whether the pipes stayed open.

    If the pipes are still open after two waits of ``_KILL_SECONDS`` (the command is killed after
    the first), they are closed, and the output read so far is returned. If the run already found
    them ``held``, they are closed after one read of ``POLL_SECONDS``.
    """
    waits = [POLL_SECONDS] if held else [_KILL_SECONDS, _KILL_SECONDS]
    for attempt, wait in enumerate(waits):
        try:
            out, err = proc.communicate(timeout=wait)
            return out, err, False
        except subprocess.TimeoutExpired as exc:
            partial = exc
        if attempt == 0 and not held:
            proc.kill()
    pipes = [stream for stream in (proc.stdout, proc.stderr) if stream is not None]
    for stream in pipes:
        stream.close()
    proc.poll()
    return _decoded(partial.output), _decoded(partial.stderr), bool(pipes)


def _decoded(data: bytes | str | None) -> str:
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cap-mb", type=float, default=MEMORY_CAP_MB)
    parser.add_argument("--timeout-min", type=float, default=60.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")
    timeout_s = args.timeout_min * 60
    try:
        run = run_watched(command, cap_mb=args.cap_mb, timeout_s=timeout_s, capture=False)
    except WatchdogError as exc:
        record = {"command": command, "killed": f"watchdog error: {exc}", "cap_mb": args.cap_mb,
                  "timeout_s": timeout_s}
        print("WATCHDOG", json.dumps(record), flush=True)
        return KILLED_EXIT_STATUS
    print("WATCHDOG", json.dumps(run.summary()), flush=True)
    if run.killed or run.survivors:
        return KILLED_EXIT_STATUS
    return run.returncode if run.returncode >= 0 else 128 - run.returncode


if __name__ == "__main__":
    sys.exit(main())
