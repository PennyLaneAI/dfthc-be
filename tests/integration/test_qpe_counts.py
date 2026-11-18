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

"""QPE resource counts (spec 11.3): the iterative QPE program compiles at
H2_STO3G_TEST and H2_631G_TEST for t = 1, 2, 3, and its worst-case counts equal
dfthc.costs.iterative_qpe exactly: T = (2^t - 1) T_C(W) + (t^3 - t)/6 and M = (2^t - 1) M_W + t,
with T_C(W) and M_W compiled at the same preset (A likewise: (2^t - 1) A_C(W) + (t^3 - t)/6).
Every compile runs in a subprocess under run_measured, and its QPE_COMPILE line records the
subprocess's wall time and peak memory and the expected CZ and PauliZ totals. The inventory test
of Task 3.4 passes for the program at H2_STO3G_TEST with t = 2. Nothing is compiled at FeMoco-54
shape: there the closed form stands (spec 11.3).

``python tests/integration/test_qpe_counts.py PRESET PROGRAM`` compiles one program at ``sto3g``
or ``631g``: ``walk`` (the controlled walk) or ``qpe1``, ``qpe2``, ``qpe3`` (the QPE program with
1, 2, 3 bits), and prints its counts as JSON on a line prefixed ``RESULT``.
"""

import functools
import json
import os
import sys
import time
from pathlib import Path

import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import compiled_resources
from dfthc.ops.walk import DFTHCWalk
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST
from dfthc.qpe import qpe_program
from dfthc.registers import WalkLayout
from dfthc.testing import walk_sim
from dfthc.testing.measure import run_measured
from dfthc.testing.resources import LEAF_ADJOINTS, TEMPLATE_ADJOINTS, captured_targets, inventory

REPO = Path(__file__).resolve().parents[2]
PRESETS = {"sto3g": H2_STO3G_TEST, "631g": H2_631G_TEST}
PREFIX = "RESULT "
CLIFFORDS = ("CZ", "PauliZ")
"""Recorded, not asserted, as expected totals: the deterministic CZ and Z of InnerSigns and of
the reflections plus the erase fix-ups' hinted conditionals, which qp.analyze counts at their
expectation (spec 7.6)."""


def build(name: str, program: str):
    """``(qfunc, num_wires)``: the controlled walk on walk_sim's case, or qpe_program."""
    c = walk_sim.case(name)
    if program == "walk":
        def qfunc():
            qp.PhaseGradientStatePrep(c.layout.phase_grad)
            qp.ctrl(DFTHCWalk(*c.compilable, **c.groups()), control=c.control)

        return qfunc, c.num_wires
    return qpe_program(c.params, c.tables, int(program.removeprefix("qpe")))


def compile_counts(name: str, program: str) -> dict:
    """One compile (subprocess side): its counts, the expected CZ and PauliZ totals and the
    compile time."""
    qfunc, num_wires = build(name, program)
    start = time.perf_counter()
    resources = compiled_resources(qfunc, num_wires)
    return {"preset": name, "program": program,
            "seconds": round(time.perf_counter() - start, 1), "toffoli": resources.toffoli,
            "measurements": resources.measurements,
            "measured_and_uncomputes": resources.measured_and_uncomputes,
            "expected_clifford_totals": {gate: resources.counts.get(gate, 0)
                                         for gate in CLIFFORDS},
            "num_wires": num_wires}


@functools.cache
def measured(name: str, program: str) -> dict:
    """``compile_counts`` in a subprocess under run_measured; prints the QPE_COMPILE line and
    returns the counts."""
    command = [sys.executable, str(Path(__file__).resolve()), name, program]
    env = os.environ | {"PYTHONPATH": str(REPO / "src"),
                        "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "0"}
    run = run_measured(command, timeout_s=900, cwd=REPO, env=env)
    lines = [line for line in run.stdout.splitlines() if line.startswith(PREFIX)]
    result = json.loads(lines[0][len(PREFIX):]) if len(lines) == 1 else None
    print("QPE_COMPILE " + json.dumps({"run": run.summary(), "result": result}))
    assert not run.timed_out and run.returncode == 0, run.stderr[-2000:]
    assert result is not None, run.stdout[-2000:]
    return result


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize("t", [1, 2, 3])
@pytest.mark.parametrize("name", PRESETS)
def test_qpe_counts(name, t):
    """Worst case, every unhinted correction at its maximum branch (spec 9.24 and 11.3).
    H2_STO3G_TEST (T_C(W) = 160, M_W = 50, A_C(W) = 110): T = 160, 481, 1,124; M = 51, 152,
    353; A = 110, 331, 774. H2_631G_TEST (241, 68, 180): T = 241, 724, 1,691; M = 69, 206, 479;
    A = 180, 541, 1,264. The program has exactly WalkLayout's wires with qpe_bits = t: the
    corrections' 2t - 1 wires are the work pool's (8 and 9 wires)."""
    p = PRESETS[name]
    walk, qpe = measured(name, "walk"), measured(name, f"qpe{t}")
    assert (walk["toffoli"], walk["measurements"], walk["measured_and_uncomputes"]) == (
        costs.walk(p, True).toffoli, costs.walk(p, False).measurements,
        costs.measured_and_uncomputes(p, True))
    corrections = (t**3 - t) // 6
    assert qpe["toffoli"] == (2**t - 1) * walk["toffoli"] + corrections
    assert qpe["measurements"] == (2**t - 1) * walk["measurements"] + t
    assert qpe["measured_and_uncomputes"] == (2**t - 1) * walk["measured_and_uncomputes"] \
        + corrections
    expected = costs.iterative_qpe(p, t)
    assert (qpe["toffoli"], qpe["measurements"]) == (expected.toffoli, expected.measurements)
    assert qpe["num_wires"] == WalkLayout.from_params(p, qpe_bits=t).num_wires


@pytest.mark.compiled
@pytest.mark.slow
def test_inventory_of_the_qpe_program():
    """The inventory test (spec 7.4 rule 6, 12.1 T6) for the QPE program at
    H2_STO3G_TEST with t = 2: the frontend converts both rounds, both walk powers, the controlled
    walk C(DFTHCWalk) (its rule pinned) and the correction, and the modified dfthc and template
    targets are exactly the controlled walk, C(DFTHCBlockEncoding) (where the generic controlled
    candidate of C(DFTHCWalk) stops: its base rule measures) and the walk's leaf and template
    adjoints."""
    c = walk_sim.case("sto3g")
    program, num_wires = qpe_program(c.params, c.tables, 2)
    names = captured_targets(program, num_wires)
    assert {"QPERound", "ControlledWalkPower", "C(DFTHCWalk)",
            "PhaseGradientRotation"} <= set(names)
    assert names.count("QPERound") == 2 and names.count("ControlledWalkPower") == 2
    assert inventory(names) == LEAF_ADJOINTS | TEMPLATE_ADJOINTS | {"C(DFTHCWalk)",
                                                                    "C(DFTHCBlockEncoding)"}


def test_qpe_closed_forms_at_femoco():
    """Spec 11.3: no compile at FeMoco-54 shape; the closed form with T_C(W) = 9,792 and
    M_W = 2,418 (compiled by the report) gives T = 9,792, 29,377, 68,548 and
    M = 2,419, 7,256, 16,929 for t = 1, 2, 3."""
    assert [costs.iterative_qpe(FEMOCO54, t) for t in (1, 2, 3)] == [
        costs.Cost(9792, 2419), costs.Cost(29377, 7256), costs.Cost(68548, 16929)]


if __name__ == "__main__":
    print(PREFIX + json.dumps(compile_counts(sys.argv[1], sys.argv[2])), flush=True)
