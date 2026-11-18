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

"""End-to-end checks E-1 to E-4 (spec 12.4) on the sparse simulator, placed in the tiers of spec
12.6: the nightly tier (``slow``) and the release tier (``release``); none runs per commit, where
the operators' own T3 tests stand in (tests/ops/test_block_encoding.py, tests/ops/test_walk.py).

Every run starts with every ancilla in |0> and ``phase_grad`` in the gradient state the program
prepares, decomposes the production classes on the tape path, and is strict (every erase outcome
has probability 1/2, every AND precondition holds). The expected values come from the encoded
model, ``encoded_block`` (spec 8.5): its ``Lambda~`` (ledger row 1A.5) and our failure signs, not
dfthcbe's (ledger row 2D.2). Schedules (spec 12.5): S0 every erase outcome 0, S1 every outcome 1,
and seeded random outcomes; the certificates of spec 12.5 prove every outcome, so these are smoke
tests. Inputs are system basis states, psi_up's bits first; FIXED_INPUTS are spec 12.6's four.

The shape fails-c3 (``walk_sim.FAILS_C3``) is added to E-1 and E-2: the test presets cannot show
the failed uniform preparation of x_o, dfthcbe's differing failure signs, or C = 3.
"""

import numpy as np
import pennylane as qp
import pytest

from dfthc.ops.walk import DFTHCBlockEncoding, DFTHCWalk
from dfthc.testing import walk_sim

ALL_INPUTS = tuple(range(16))
FIXED_INPUTS = (0b0000, 0b1000, 0b1010, 0b1111)
"""Spec 12.6: |00>|00>, |10>|00>, |10>|10>, |11>|11> in psi_up (x) psi_dn."""
TWO_ELECTRON_631G = tuple(v for v in range(256) if bin(v).count("1") == 2)
"""The 28 two-electron basis states of H2_631G_TEST (N = 4, 8 spin orbitals)."""
ATOL = 1e-10
"""Spec 12.4 and the Task 3.5 brief: every E-check comparison to 1e-10."""

nightly, release = pytest.mark.slow, pytest.mark.release


def columns(c, ops, inputs, kind, seed=0):
    """The projected output of every basis input, as columns, each run with the forced schedule
    ``kind`` of exactly the program's measurements."""
    outcomes = walk_sim.schedule(kind, walk_sim.num_measurements(ops), seed=seed)
    return np.array([walk_sim.project(c, walk_sim.simulate(
        c, ops, walk_sim.input_state(c, walk_sim.basis(c, v)), outcomes))[0]
        for v in inputs]).T


def assert_same(state, expected, where):
    """``walk_sim.assert_same_state``, naming the input and control state of a failure."""
    try:
        walk_sim.assert_same_state(state, expected, atol=ATOL)
    except AssertionError as error:
        raise AssertionError(f"{where}: largest difference {error}") from None


E1 = [pytest.param("sto3g", "S0", ALL_INPUTS, marks=nightly, id="sto3g-S0-all"),
      pytest.param("sto3g", "S1", FIXED_INPUTS, marks=nightly, id="sto3g-S1-fixed"),
      pytest.param("fails-c3", "S0", ALL_INPUTS, marks=nightly, id="fails-c3-S0-all"),
      pytest.param("sto3g", "S1", ALL_INPUTS, marks=release, id="sto3g-S1-all"),
      pytest.param("sto3g", "seeded", FIXED_INPUTS, marks=release, id="sto3g-seeded-fixed")]
E1 += [pytest.param("631g", "S0", (v,), marks=release, id=f"631g-S0-{v:08b}")
       for v in TWO_ELECTRON_631G]


@pytest.mark.parametrize(("name", "kind", "inputs"), E1)
def test_e1_block_encoding(name, kind, inputs):
    """E-1: <0_anc|BE|0_anc> = H~_SA / Lambda~ - I to 1e-10, built column by column from the
    simulated outputs of DFTHCBlockEncoding (nightly: all 16 inputs under S0 and the 4 fixed
    under S1 at H2_STO3G_TEST, all 16 under S0 at fails-c3; release: all 16 under S1, the 4 fixed
    under a seeded schedule, and H2_631G_TEST's 28 two-electron inputs under S0, one per test)."""
    c = walk_sim.case(name)
    ops = walk_sim.program(c, DFTHCBlockEncoding)
    measured = columns(c, ops, inputs, kind, seed=12)
    np.testing.assert_allclose(measured, c.block()[:, list(inputs)], rtol=0, atol=ATOL)


@nightly
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_e2_walk_powers(name, k):
    """E-2: <0|W^k|0> = T_k(H~_SA / Lambda~ - I) for the Chebyshev polynomials T_1(x) = x and
    T_2(x) = 2x^2 - 1, from k DFTHCWalk steps on the 4 fixed inputs under S0 (nightly)."""
    c = walk_sim.case(name)
    x = c.block()
    chebyshev = {1: x, 2: 2 * x @ x - np.eye(len(x))}[k]
    ops = walk_sim.program(c, DFTHCWalk, repeat=k)
    measured = columns(c, ops, FIXED_INPUTS, "S0")
    np.testing.assert_allclose(measured, chebyshev[:, list(FIXED_INPUTS)], rtol=0, atol=ATOL)


E3 = [pytest.param("S0", FIXED_INPUTS, marks=nightly, id="S0-fixed"),
      pytest.param("S1", ALL_INPUTS, marks=release, id="S1-all"),
      pytest.param("seeded", FIXED_INPUTS, marks=release, id="seeded-fixed")]


@pytest.mark.parametrize(("kind", "inputs"), E3)
def test_e3_controlled_walk(kind, inputs):
    """E-3 at H2_STO3G_TEST: qp.ctrl(DFTHCWalk) with the control in |0> returns the whole input,
    in |1> gives DFTHCWalk's whole output, and in |+> their superposition with weight 1/sqrt 2
    each, which detects a global-phase error of either controlled reflection; and, after one
    uncontrolled DFTHCWalk step (index registers of A u B no longer |0>, as between the rounds of
    iterative QPE), the control in |+> gives (|0> W|psi> + |1> W^2|psi>)/sqrt 2, which also
    needs ref_ab controlled (nightly: the 4 fixed inputs under S0; release: all 16 under S1 and
    the 4 fixed under a seeded schedule)."""
    c = walk_sim.case("sto3g")
    once = walk_sim.program(c, DFTHCWalk)
    twice = walk_sim.program(c, DFTHCWalk, repeat=2)
    controlled = walk_sim.program(c, DFTHCWalk, controlled=True)
    then_controlled = walk_sim.operations(
        c, lambda: DFTHCWalk(*c.compilable, **c.groups()),
        lambda: qp.ctrl(DFTHCWalk(*c.compilable, **c.groups()), control=c.control))
    m = walk_sim.num_measurements(once)
    outcomes = walk_sim.schedule(kind, 2 * m, seed=13)
    plus = 2**-0.5
    for v in inputs:
        psi = walk_sim.basis(c, v)
        identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
        stepped = walk_sim.simulate(c, once, walk_sim.input_state(c, psi), outcomes[:m])
        applied = walk_sim.set_control(c, stepped)
        expected = {"0": identity, "1": applied,
                    "+": walk_sim.superpose((plus, identity), (plus, applied))}
        for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", [plus, plus])):
            out = walk_sim.simulate(c, controlled, walk_sim.input_state(c, psi, control),
                                    outcomes[:m])
            assert_same(out, expected[label], f"input {v:04b}, control |{label}>")
        stepped_twice = walk_sim.simulate(c, twice, walk_sim.input_state(c, psi), outcomes)
        out = walk_sim.simulate(c, then_controlled, walk_sim.input_state(c, psi, [plus, plus]),
                                outcomes)
        assert_same(out, walk_sim.superpose(
            (plus, stepped), (plus, walk_sim.set_control(c, stepped_twice))),
            f"input {v:04b}, control |+> after a walk step")


@nightly
@pytest.mark.parametrize("kind", ["alternating", "seeded"])
def test_e4_prepare_then_unprepare(kind):
    """E-4 at H2_STO3G_TEST: U then U-dagger (the block encoding's children without ref_t2)
    returns every ancilla to |0> and the system unchanged, on all 16 inputs, under the
    alternating and a seeded schedule of the five erases (an all-ones or all-zeros schedule sees
    less of the erases' masks; ledger row 2D.2)."""
    c = walk_sim.case("sto3g")
    ops = walk_sim.prepare_unprepare(c)
    outcomes = walk_sim.schedule(kind, walk_sim.num_measurements(ops), seed=14)
    for v in ALL_INPUTS:
        psi = walk_sim.basis(c, v)
        identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
        assert_same(walk_sim.simulate(c, ops, walk_sim.input_state(c, psi), outcomes), identity,
                    f"input {v:04b}")
