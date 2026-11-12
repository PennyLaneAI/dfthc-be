"""Run canary snippets in a fresh Python process (spec section 14 and T7).

Snippets that may abort the process, that hit "no decomposition path", or that depend on
import-time state run through :func:`run_snippet`. Each snippet reports outcomes with the
prelude's ``report(label, fn)``, which prints one line ``<label> OK <json>`` or
``<label> ERROR <ExceptionType> <message lines joined by " | ">``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Protocol

SRC = Path(__file__).resolve().parents[2]


class _HasOutput(Protocol):
    """A finished run with text output: ``CompletedProcess`` or ``WatchedRun``."""

    stdout: str
    stderr: str


PRELUDE = textwrap.dedent(
    """
    import json
    import warnings

    import numpy as np
    import pennylane as qp
    from pennylane.core import Operator2
    from pennylane.typing import Wire

    from dfthc.compile import GATE_SET


    def counts(circuit, level="user"):
        resources = qp.analyze(circuit, level=level)().resources
        return {str(k): str(v) for k, v in resources.counts.items()}


    def report(label, fn):
        try:
            print(label, "OK", json.dumps(fn(), sort_keys=True, default=str), flush=True)
        except Exception as exc:  # noqa: BLE001
            message = " | ".join(line for line in str(exc).splitlines() if line.strip())
            print(label, "ERROR", type(exc).__name__, message, flush=True)
    """
)


_REMOVED_VARIABLES = ("PYTHONWARNINGS", "PYTHONDEVMODE", "XLA_FLAGS")


def snippet_command(
    body: str, *, export_warning_variable: bool = True
) -> tuple[list[str], dict[str, str]]:
    """The command line and the environment that run ``PRELUDE + body`` in a new interpreter.

    The environment is the parent's without the variables that change warning handling or the
    JAX and Catalyst configuration (``PYTHONWARNINGS``, ``PYTHONDEVMODE``, ``XLA_FLAGS``, every
    ``JAX_*`` and ``CATALYST_*``), so that a canary sees the same defaults whatever the caller
    exports. It then sets ``PYTHONPATH``, and the L-18 variable to ``0`` unless
    ``export_warning_variable`` is false. Neither ``scripts/test.sh``, ``tests/conftest.py`` nor
    ``pyproject.toml`` sets any removed variable for the canaries except the L-18 one.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _REMOVED_VARIABLES and not key.startswith(("JAX_", "CATALYST_"))
    }
    env["PYTHONPATH"] = str(SRC)
    if export_warning_variable:
        env["CATALYST_SILENCE_RULE_LOWERING_WARNINGS"] = "0"
    return [sys.executable, "-c", PRELUDE + textwrap.dedent(body)], env


def run_snippet(
    body: str, *, export_warning_variable: bool = True, timeout: float = 900.0
) -> subprocess.CompletedProcess[str]:
    """Run ``PRELUDE + body`` in a new interpreter and return the completed process.

    ``export_warning_variable=False`` leaves CATALYST_SILENCE_RULE_LOWERING_WARNINGS unset (L-18).
    """
    command, env = snippet_command(body, export_warning_variable=export_warning_variable)
    return subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=timeout, check=False
    )


def outcome(result: _HasOutput, label: str) -> str:
    """The text after ``<label> `` on the snippet's report line, e.g. ``OK {...}``."""
    for line in result.stdout.splitlines():
        if line.startswith(label + " "):
            return line[len(label) + 1 :]
    raise AssertionError(
        f"no report line for {label!r}\nstdout:\n{result.stdout}\nstderr tail:\n"
        + "\n".join(result.stderr.splitlines()[-20:])
    )
