"""Canary L-21 and its benchmark: each distinct compilable table is a separate rule compile.

The per-commit canary counts rule traces: on 2026-11-11 every distinct table traces its rule
twice, and identical tables in one program share those traces. The nightly benchmark compiles
each case in a fresh process under the watchdog and records wall time (informational), CPU time
(Python plus the Catalyst compiler subprocess, measured across the compile only: the difference of
``getrusage`` for the process and its finished children before and after ``qp.analyze``) and peak
memory (the watchdog's summed physical footprint of the process tree). It fails if CPU time
exceeds the stored baseline by more than a factor of two and more than ``CPU_SLACK_SECONDS``,
if peak memory exceeds twice the baseline, or if the compile cost no longer grows with table
size. Refresh the baseline at every nightly bump, serially:

    DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 \
        scripts/test.sh tests/canary/test_l21_compile_time.py
"""

import json
import os
from collections import Counter
from pathlib import Path

import pennylane as qp
import pytest
from pennylane.core import Operator2

from dfthc.compile import GATE_SET
from dfthc.testing.canary import outcome, snippet_command
from dfthc.testing.watchdog import HEAVY_MEMORY_MB, run_watched

pytestmark = [pytest.mark.canary, pytest.mark.compiled]

BASELINE = Path(__file__).parent / "compile_baseline.json"
CASES = {
    "tables_1_size_64": (1, 64),
    "tables_2_size_64": (2, 64),
    "tables_4_size_64": (4, 64),
    "tables_1_size_1024": (1, 1024),
    "tables_1_size_8640": (1, 8640),
}
CPU_SLACK_SECONDS = 2.0
"""Absolute CPU slack of the baseline check (sub-second cases are noise-dominated when loaded)."""
SIZE_GROWTH_MIN = 5
"""Least compile CPU-time ratio of the 8,640- to the 64-entry case (about 84 on 2026-11-12)."""

_RULE_TRACES: list[tuple[int, ...]] = []


class L21Probe(Operator2):
    compilable_argnames = ("table",)
    wire_argnames = ("wires",)

    def __init__(self, table, wires):
        super().__init__(table, wires)


def _l21_probe_resources(table, wires):
    return {qp.CNOT: sum(bin(value).count("1") for value in table)}


@qp.register_resources(_l21_probe_resources)
def _l21_probe_rule(table, wires):
    _RULE_TRACES.append(table)
    for value in table:
        for bit in range(2):
            if (value >> bit) & 1:
                qp.CNOT([wires[0], wires[1 + bit]])


qp.add_decomps(L21Probe, _l21_probe_rule)


def _rule_traces(tables) -> Counter:
    """How often the probe's rule is traced, per table, compiling one program over ``tables``."""
    _RULE_TRACES.clear()

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("null.qubit", wires=3))
    def circuit():
        for table in tables:
            L21Probe(table, [0, 1, 2])
        return qp.probs(wires=[0])

    qp.analyze(circuit, level="user")()
    return Counter(_RULE_TRACES)


def test_each_distinct_table_is_a_separate_rule_compile():
    """L-21 (catalyst#3173): every distinct table traces the rule, identical tables share it.

    On 2026-11-11 each distinct table traces its rule exactly twice. A different count is a
    change of Catalyst's behaviour: update the L-21 entries of docs/limitations.md and spec
    section 14 before adjusting these numbers.
    """
    first, second = (1, 2, 3), (3, 1, 2)
    note = "Catalyst's rule tracing changed: update the L-21 docs (limitations.md, spec section 14)"
    assert _rule_traces([first]) == {first: 2}, note
    assert _rule_traces([first, second]) == {first: 2, second: 2}, note
    assert _rule_traces([first, first]) == {first: 2}, note


_SNIPPET = """
import resource
import time


class L21Table(Operator2):
    compilable_argnames = ("table",)
    wire_argnames = ("wires",)

    def __init__(self, table, wires):
        super().__init__(table, wires)


def _l21_resources(table, wires):
    return {qp.CNOT: sum(bin(value).count("1") for value in table)}


@qp.register_resources(_l21_resources)
def _l21_rule(table, wires):
    for value in table:
        for bit in range(4):
            if (value >> bit) & 1:
                qp.CNOT([wires[0], wires[1 + bit]])


qp.add_decomps(L21Table, _l21_rule)
rng = np.random.default_rng(21)
tables = [tuple(int(v) for v in rng.integers(0, 16, size=SIZE)) for _ in range(COUNT)]


@qp.qjit(capture=True)
@qp.decompose(gate_set=GATE_SET)
@qp.qnode(qp.device("null.qubit", wires=5))
def circuit():
    for table in tables:
        L21Table(table, [0, 1, 2, 3, 4])
    return qp.probs(wires=[0])


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def measure():
    cpu_before = cpu_seconds()
    start = time.perf_counter()
    qp.analyze(circuit, level="user")()
    wall = time.perf_counter() - start
    return {"seconds": round(wall, 2), "cpu_seconds": round(cpu_seconds() - cpu_before, 2)}


report("MEASURE", measure)
"""


def _measure(count: int, size: int) -> dict:
    """Compile one case in a fresh process under the watchdog; CPU delta and tree peak memory."""
    command, env = snippet_command(f"COUNT = {count}\nSIZE = {size}\n" + _SNIPPET)
    run = run_watched(command, cap_mb=HEAVY_MEMORY_MB, timeout_s=3600, env=env)
    tail = "\n".join(run.stderr.splitlines()[-20:])
    assert run.killed is None and run.returncode == 0 and not run.survivors, (
        run.killed, run.returncode, run.survivors, tail
    )
    values = json.loads(outcome(run, "MEASURE").removeprefix("OK "))
    values["peak_mb"] = round(run.peak_mb, 1)
    return values


def _regressions(measured: dict, baseline: dict) -> list[str]:
    """Describe every case that exceeds its baseline (an empty list means none does).

    A case regresses if its CPU time is more than ``max(2 * ref, ref + CPU_SLACK_SECONDS)`` or
    its peak memory is more than twice the baseline. A measured case that the baseline lacks is
    reported too, with the command that refreshes the baseline: the check fails rather than
    skipping a case silently.
    """
    problems = []
    for case, values in measured.items():
        if case not in baseline:
            problems.append(
                f"{case}: missing from {BASELINE.name}; "
                "refresh it with DFTHC_UPDATE_COMPILE_BASELINE=1"
            )
            continue
        ref = baseline[case]
        cpu_limit = max(2 * ref["cpu_seconds"], ref["cpu_seconds"] + CPU_SLACK_SECONDS)
        if values["cpu_seconds"] > cpu_limit:
            problems.append(
                f"{case}: cpu_seconds {values['cpu_seconds']} > {cpu_limit:.2f} "
                f"(baseline {ref['cpu_seconds']})"
            )
        if values["peak_mb"] > 2 * ref["peak_mb"]:
            problems.append(
                f"{case}: peak_mb {values['peak_mb']} > {2 * ref['peak_mb']:.1f} "
                f"(baseline {ref['peak_mb']})"
            )
    return problems


def test_baseline_regression_rule():
    """The baseline check: twice the baseline, with 2 s of CPU slack, and no silent skips."""

    def check(cpu, peak, ref_cpu=0.43, ref_peak=200.0, case="case"):
        measured = {case: {"cpu_seconds": cpu, "peak_mb": peak}}
        return _regressions(measured, {"case": {"cpu_seconds": ref_cpu, "peak_mb": ref_peak}})

    assert check(2.4, 200.0) == []
    assert len(check(2.5, 200.0)) == 1
    assert check(72.0, 200.0, ref_cpu=36.0) == []
    assert len(check(72.1, 200.0, ref_cpu=36.0)) == 1
    assert check(0.4, 400.0) == []
    assert len(check(0.4, 400.1)) == 1
    both = check(2.5, 400.1)
    assert len(both) == 2 and "cpu_seconds" in both[0] and "peak_mb" in both[1]
    missing = check(0.4, 200.0, case="other")
    assert len(missing) == 1 and "missing from compile_baseline.json" in missing[0]


@pytest.mark.subprocess
@pytest.mark.slow
def test_compile_time_scaling():
    """Benchmark: every case within the baseline rule, and cost that grows with table size."""
    measured = {case: _measure(*CASES[case]) for case in sorted(CASES)}
    if os.environ.get("DFTHC_UPDATE_COMPILE_BASELINE") == "1":
        BASELINE.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
        return
    regressions = _regressions(measured, json.loads(BASELINE.read_text()))
    assert not regressions, "\n".join(regressions)
    small, large = measured["tables_1_size_64"], measured["tables_1_size_8640"]
    growth = large["cpu_seconds"] / small["cpu_seconds"]
    assert growth >= SIZE_GROWTH_MIN, f"compile cost no longer grows with table size: {growth:.1f}x"
