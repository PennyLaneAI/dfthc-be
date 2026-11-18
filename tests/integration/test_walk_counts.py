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

"""Whole-program counts of the walk and the controlled walk (spec 9.22 and 9.24):
each compiles as one program, its T, M and A equal the closed forms exactly at the test
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
from dfthc.ops.walk import DFTHCWalk
from dfthc.params import FEMOCO54, MID_NAPHTHALENE
from dfthc.testing import walk_sim
from dfthc.testing.watchdog import effective_limit_mb, run_watched

REPO = Path(__file__).resolve().parents[2]
PROGRAMS = {"walk": False, "controlled_walk": True}
"""Each program and whether it is the controlled walk, qp.ctrl(DFTHCWalk)."""
CAP_MB = 4096
"""The MID_NAPHTHALENE compiles' cap: the watchdog kills at 3,840 MB (docs/environment.md)."""
PREFIX = "RESULT "


def compile_program(c, program, control_value=1):
    def qfunc():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        walk = DFTHCWalk(*c.compilable, **c.groups())
        if PROGRAMS[program]:
            qp.ctrl(walk, control=c.control, control_values=[control_value])

    return compiled_resources(qfunc, c.num_wires)


def assert_counts(resources, p, controlled):
    """T, M and A against costs.walk and costs.measured_and_uncomputes (spec 9.24), every
    Toffoli a TemporaryAND (no Toffoli); compiled_resources has asserted non-symbolic counts and
    gate-set closure."""
    expected = costs.walk(p, controlled)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measured_and_uncomputes == costs.measured_and_uncomputes(p, controlled)
    assert (resources.counts.get("Toffoli", 0), resources.counts["TemporaryAND"]) == (
        0, expected.toffoli)


@pytest.mark.compiled
@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_counts_at_the_test_presets(name, program):
    """T = A = 158 / 160 and 239 / 241, M = 48 and 62."""
    c = walk_sim.case(name)
    assert_counts(compile_program(c, program), c.params, PROGRAMS[program])


@pytest.mark.compiled
def test_control_value_0_keeps_the_counts():
    """qp.ctrl(DFTHCWalk, control_values=[0]) at H2_STO3G_TEST: the same T, M and A and wires as
    control value 1; its two conditional X gates on the control count as PauliX at their maximum
    branch."""
    c = walk_sim.case("sto3g")
    resources = compile_program(c, "controlled_walk", control_value=0)
    assert_counts(resources, c.params, True)
    assert resources.num_wires == c.num_wires


def compile_mid(program: str) -> dict:
    """One MID_NAPHTHALENE compile (subprocess side): its counts and compile time."""
    c = walk_sim.WalkCase.build(MID_NAPHTHALENE, load_factors("naphthalene"))
    start = time.perf_counter()
    resources = compile_program(c, program)
    return {"program": program, "seconds": round(time.perf_counter() - start, 1),
            "toffoli": resources.toffoli, "measurements": resources.measurements,
            "measured_and_uncomputes": resources.measured_and_uncomputes,
            "plain_toffoli": resources.counts.get("Toffoli", 0), "num_wires": c.num_wires}


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize("program", PROGRAMS)
def test_counts_at_mid_naphthalene(program):
    """T4 at MID_NAPHTHALENE (nightly), each program compiled once in a subprocess under
    run_watched with a 4 GB cap: T = A = 1,910 / 1,912, M = 484, no Toffoli. The MID_COMPILE
    line records the watched wall time and peak memory."""
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
    controlled = PROGRAMS[program]
    expected = costs.walk(MID_NAPHTHALENE, controlled)
    assert (result["toffoli"], result["measurements"], result["measured_and_uncomputes"],
            result["plain_toffoli"]) == (
        expected.toffoli, expected.measurements,
        costs.measured_and_uncomputes(MID_NAPHTHALENE, controlled), 0)


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 (spec 9.22 and 9.24), without compiling (the FeMoco compile is the resource
    report's): T = 9,746 + 44 = 9,790, controlled 9,747 + 45 = 9,792; M = 2,418; A = T, every
    Toffoli a TemporaryAND; M + A = 12,208 and 12,210."""
    p = FEMOCO54
    assert [costs.ref_ab(p, c).toffoli for c in (False, True)] == [44, 45]
    assert costs.walk(p, False) == costs.Cost(9790, 2418)
    assert costs.walk(p, True) == costs.Cost(9792, 2418)
    assert [costs.measured_and_uncomputes(p, c) for c in (False, True)] == [9790, 9792]
    assert [costs.walk(p, c).measurements + costs.measured_and_uncomputes(p, c)
            for c in (False, True)] == [12208, 12210]


if __name__ == "__main__":
    print(PREFIX + json.dumps(compile_mid(sys.argv[1])), flush=True)
