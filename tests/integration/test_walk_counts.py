"""Whole-program counts of the walk and the controlled walk (spec 9.22 and 9.24; master plan Task
3.4): each compiles as one program, its T, M and A equal the closed forms exactly at the test
presets (per commit) and at MID_NAPHTHALENE (nightly, each compile in a subprocess under
run_watched, which records its time and memory), and the FeMoco-54 values on the closed forms.

``python tests/integration/test_walk_counts.py PROGRAM`` compiles one MID_NAPHTHALENE program
(``walk`` or ``controlled_walk``) and prints its counts as JSON on a line prefixed ``RESULT``.
"""

import json
import os
import sys
import time
from pathlib import Path

import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import compiled_resources
from dfthc.data.factors import load_factors
from dfthc.ops.walk import ControlledDFTHCWalk, DFTHCWalk
from dfthc.params import FEMOCO54, MID_NAPHTHALENE
from dfthc.testing import walk_sim
from dfthc.testing.watchdog import effective_limit_mb, run_watched

REPO = Path(__file__).resolve().parents[2]
PROGRAMS = {"walk": (DFTHCWalk, False), "controlled_walk": (ControlledDFTHCWalk, True)}
CAP_MB = 4096
"""The MID_NAPHTHALENE compiles' cap: the watchdog kills at 3,840 MB (master plan Task 3.4)."""
PREFIX = "RESULT "


def compile_program(c, program):
    cls, controlled = PROGRAMS[program]

    def qfunc():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        cls(*c.compilable, **c.groups(controlled))

    return compiled_resources(qfunc, c.num_wires)


def assert_counts(resources, p, controlled):
    """T, M and A against costs.walk and costs.measured_and_uncomputes, the plain Toffolis
    against costs.toffoli_only (spec 9.24); compiled_resources has asserted non-symbolic counts
    and gate-set closure."""
    expected = costs.walk(p, controlled)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measured_and_uncomputes == costs.measured_and_uncomputes(p, controlled)
    assert resources.counts["Toffoli"] == costs.toffoli_only(p)


@pytest.mark.compiled
@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_counts_at_the_test_presets(name, program):
    """T = 158 / 160 and 239 / 241, M = 50 and 68, A = 108 / 110 and 178 / 180."""
    c = walk_sim.case(name)
    assert_counts(compile_program(c, program), c.params, PROGRAMS[program][1])


def compile_mid(program: str) -> dict:
    """One MID_NAPHTHALENE compile (subprocess side): its counts and compile time."""
    c = walk_sim.WalkCase.build(MID_NAPHTHALENE, load_factors("naphthalene"))
    start = time.perf_counter()
    resources = compile_program(c, program)
    return {"program": program, "seconds": round(time.perf_counter() - start, 1),
            "toffoli": resources.toffoli, "measurements": resources.measurements,
            "measured_and_uncomputes": resources.measured_and_uncomputes,
            "plain_toffoli": resources.counts["Toffoli"], "num_wires": c.num_wires}


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize("program", PROGRAMS)
def test_counts_at_mid_naphthalene(program):
    """T4 at MID_NAPHTHALENE (nightly), each program compiled once in a subprocess under
    run_watched with a 4 GB cap: T = 1,910 / 1,912, M = 502, A = 1,675 / 1,677. The MID_COMPILE
    line records the watched wall time and peak memory, for comparison with the K-1 stand-in at
    the same sizes (spikes/results/k1_mid10.json and k1d_PX_mid10.json)."""
    command = [sys.executable, str(Path(__file__).resolve()), program]
    env = os.environ | {"PYTHONPATH": str(REPO / "src"),
                        "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "0"}
    run = run_watched(command, cap_mb=CAP_MB, timeout_s=900, cwd=REPO, env=env)
    lines = [line for line in run.stdout.splitlines() if line.startswith(PREFIX)]
    result = json.loads(lines[0][len(PREFIX):]) if len(lines) == 1 else None
    print("MID_COMPILE " + json.dumps({"watchdog": run.summary(), "result": result}))
    assert run.killed is None and not run.survivors and run.returncode == 0, run.stderr[-2000:]
    assert run.peak_mb <= effective_limit_mb(CAP_MB)
    assert result is not None, run.stdout[-2000:]
    controlled = PROGRAMS[program][1]
    expected = costs.walk(MID_NAPHTHALENE, controlled)
    assert (result["toffoli"], result["measurements"], result["measured_and_uncomputes"],
            result["plain_toffoli"]) == (
        expected.toffoli, expected.measurements,
        costs.measured_and_uncomputes(MID_NAPHTHALENE, controlled),
        costs.toffoli_only(MID_NAPHTHALENE))


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 (spec 9.22 and 9.24), without compiling (master plan section 12.9; the FeMoco
    compile is Task 4.5's): T = 9,746 + 44 = 9,790, controlled 9,747 + 45 = 9,792; M = 2,418;
    T_Toffoli = 18 + 20 + 72 + 780 + 216 + 12 = 1,118; A = 8,672 and 8,674; M + A = 11,090 and
    11,092."""
    p = FEMOCO54
    assert [costs.ref_ab(p, c).toffoli for c in (False, True)] == [44, 45]
    assert costs.walk(p, False) == costs.Cost(9790, 2418)
    assert costs.walk(p, True) == costs.Cost(9792, 2418)
    assert [2 * p.w_x, 4 * p.w_b, (p.K1 - 1) * p.d1, 2 * (p.K2 - 1) * p.d2, 4 * p.N, 12] == [
        18, 20, 72, 780, 216, 12]
    assert costs.toffoli_only(p) == 1118
    assert [costs.measured_and_uncomputes(p, c) for c in (False, True)] == [8672, 8674]
    assert [costs.walk(p, c).measurements + costs.measured_and_uncomputes(p, c)
            for c in (False, True)] == [11090, 11092]


if __name__ == "__main__":
    print(PREFIX + json.dumps(compile_mid(sys.argv[1])), flush=True)
