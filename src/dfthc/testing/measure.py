# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run a command in a subprocess with a timeout, and measure its wall time and peak memory."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

POLL_SECONDS = 0.05
_MAXRSS_UNIT = 1 if sys.platform == "darwin" else 1024
"""Bytes per unit of ``ru_maxrss``: bytes on macOS, KiB on Linux."""


@dataclass
class MeasuredRun:
    """Outcome of :func:`run_measured`. ``peak_mb`` is ``wait4``'s ``ru_maxrss``: the largest
    resident size of the child and of the descendants it reaped."""

    returncode: int
    seconds: float
    peak_mb: float
    timed_out: bool
    stdout: str = field(default="", repr=False)
    stderr: str = field(default="", repr=False)

    def summary(self) -> dict:
        return {"returncode": self.returncode, "seconds": round(self.seconds, 1),
                "peak_mb": round(self.peak_mb, 1), "timed_out": self.timed_out}


def run_measured(command: list[str], *, timeout_s: float, cwd=None, env=None) -> MeasuredRun:
    """Run ``command`` in a new session until it exits; after ``timeout_s`` seconds its process
    group is killed. The output goes to temporary files, so a full pipe cannot stall it."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        start = time.monotonic()
        proc = subprocess.Popen(command, stdout=out, stderr=err, cwd=cwd, env=env,
                                start_new_session=True)
        timed_out = False
        while not (reaped := os.wait4(proc.pid, os.WNOHANG))[0]:
            if time.monotonic() - start > timeout_s:
                timed_out = True
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
                reaped = os.wait4(proc.pid, 0)
                break
            time.sleep(POLL_SECONDS)
        seconds = time.monotonic() - start
        _, status, usage = reaped
        # wait4 reaped the child; Popen must not wait for its pid again.
        proc.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        return MeasuredRun(proc.returncode, seconds, usage.ru_maxrss * _MAXRSS_UNIT / 2**20,
                           timed_out, out.read().decode(errors="replace"),
                           err.read().decode(errors="replace"))
