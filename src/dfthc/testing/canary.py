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

SRC = Path(__file__).resolve().parents[2]

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


def run_snippet(
    body: str, *, export_warning_variable: bool = True, timeout: float = 900.0
) -> subprocess.CompletedProcess[str]:
    """Run ``PRELUDE + body`` in a new interpreter and return the completed process.

    ``export_warning_variable=False`` unsets CATALYST_SILENCE_RULE_LOWERING_WARNINGS (L-18).
    """
    env = dict(os.environ, PYTHONPATH=str(SRC))
    if export_warning_variable:
        env["CATALYST_SILENCE_RULE_LOWERING_WARNINGS"] = "0"
    else:
        env.pop("CATALYST_SILENCE_RULE_LOWERING_WARNINGS", None)
    code = PRELUDE + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout,
        check=False,
    )


def outcome(result: subprocess.CompletedProcess[str], label: str) -> str:
    """The text after ``<label> `` on the snippet's report line, e.g. ``OK {...}``."""
    for line in result.stdout.splitlines():
        if line.startswith(label + " "):
            return line[len(label) + 1 :]
    raise AssertionError(
        f"no report line for {label!r}\nstdout:\n{result.stdout}\nstderr tail:\n"
        + "\n".join(result.stderr.splitlines()[-20:])
    )
