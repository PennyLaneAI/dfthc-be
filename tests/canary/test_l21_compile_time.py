"""Canary L-21 and its benchmark: each distinct compilable table is a separate rule compile.

The per-commit canary counts rule traces: on 2026-11-11 every distinct table traces its rule
twice, and identical tables in one program share those traces. The nightly benchmark compiles
each case in a fresh process and records wall time, CPU time (Python plus the Catalyst compiler
subprocess) and peak memory (the process plus its finished children, an upper bound on the peak
total). It fails if CPU time or memory exceeds twice the stored baseline, or if the compile cost
no longer grows with table size. Refresh the baseline at every nightly bump, serially:

    DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 \
        scripts/test.sh tests/canary/test_l21_compile_time.py
"""

import json
import os
from pathlib import Path

import pennylane as qp
import pytest
from pennylane.core import Operator2

from dfthc.compile import GATE_SET
from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled]

BASELINE = Path(__file__).parent / "compile_baseline.json"
CASES = {
    "tables_1_size_64": (1, 64),
    "tables_2_size_64": (2, 64),
    "tables_4_size_64": (4, 64),
    "tables_1_size_1024": (1, 1024),
    "tables_1_size_8640": (1, 8640),
}
SIZE_GROWTH_MIN = 5
"""Least CPU-time ratio of the 8,640-entry case to the 64-entry case (about 28 on 2026-11-11)."""

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


def _rule_traces(tables) -> int:
    """Number of times the probe's rule is traced while compiling one program over ``tables``."""
    _RULE_TRACES.clear()

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("null.qubit", wires=3))
    def circuit():
        for table in tables:
            L21Probe(table, [0, 1, 2])
        return qp.probs(wires=[0])

    qp.analyze(circuit, level="user")()
    return len(_RULE_TRACES)


def test_each_distinct_table_is_a_separate_rule_compile():
    """L-21 (catalyst#3173): rule compiles grow with the number of distinct tables."""
    first, second = (1, 2, 3), (3, 1, 2)
    single = _rule_traces([first])
    assert single >= 1
    assert _rule_traces([first, second]) == 2 * single
    assert _rule_traces([first, first]) == single


_SNIPPET = """
import resource
import sys
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


def measure():
    start = time.perf_counter()
    qp.analyze(circuit, level="user")()
    wall = time.perf_counter() - start
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if sys.platform == "darwin" else 1024
    cpu = own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime
    return {
        "seconds": round(wall, 2),
        "cpu_seconds": round(cpu, 2),
        "max_rss_mb": round((own.ru_maxrss + children.ru_maxrss) * scale / 2**20, 1),
    }


report("MEASURE", measure)
"""


def _measure(count: int, size: int) -> dict:
    result = run_snippet(f"COUNT = {count}\nSIZE = {size}\n" + _SNIPPET, timeout=3600)
    return json.loads(outcome(result, "MEASURE").removeprefix("OK "))


@pytest.mark.subprocess
@pytest.mark.slow
def test_compile_time_scaling():
    """Benchmark: every case within twice the baseline, and cost that grows with table size."""
    measured = {case: _measure(*CASES[case]) for case in sorted(CASES)}
    if os.environ.get("DFTHC_UPDATE_COMPILE_BASELINE") == "1":
        BASELINE.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
        return
    baseline = json.loads(BASELINE.read_text())
    for case, values in measured.items():
        reference = baseline[case]
        assert values["cpu_seconds"] <= 2 * reference["cpu_seconds"], (case, values, reference)
        assert values["max_rss_mb"] <= 2 * reference["max_rss_mb"], (case, values, reference)
    small, large = measured["tables_1_size_64"], measured["tables_1_size_8640"]
    growth = large["cpu_seconds"] / small["cpu_seconds"]
    assert growth >= SIZE_GROWTH_MIN, f"compile cost no longer grows with table size: {growth:.1f}x"
