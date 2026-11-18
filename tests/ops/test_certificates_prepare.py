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

"""Outcome certificates of spec 12.5 for the composite unprepares (nightly tier).

A prepare followed by its unprepare must return every register to its input for every outcome of
the unprepare's erase. The phase the erase leaves on the read's address and the phase of its
fix-up are both GF(2)-linear in the outcome vector ``xi``, so the residual is linear, and exact
restoration under the zero vector and under every unit vector, M + 1 forced runs, certifies all
2**M outcomes. The premise is checked on the certified tape itself: it has conditionals, and each
is a ``PauliZ`` or ``CZ`` with a GF(2)-linear predicate (``nonlinear_conditionals``); the measured
uncomputes of the comparators and of the uniform preparations are ``Adjoint(TemporaryAND)``
gates, whose preconditions the simulator checks on every term, not conditionals. Strict mode
also asserts that every erase measurement has probability 1/2. Each run must consume exactly M
outcomes and end in exactly the input state times the phase-gradient state.

OuterPrepare -> OuterUnprepare: the pair has no input register and no enable (the walk uses
none), so every register in |0> is its only legal pre-state, and each run starts from it; an
outcome-dependent phase, a wrong amplitude or a leftover ancilla shows as a difference from it
(the uniform preparation's adjoint moves any relative phase on ``x_o`` away from |0>). This fixed
input does not instantiate spec 12.5's general premise of random complex address and control
amplitudes: the read and erase carry that premise through Task 1C.5's QROAMRead -> QROAMErase
certificates (tests/ops/test_certificates_qroam.py), and this certificate composes them with the
pair's other children on its one legal input. The tables are random, the shapes include a
nontrivial uniform preparation (failure branches and its phase), dead positions in the read's
last block and a partly covered fix-up leaf; one shape has a ``phase_grad`` longer than ``s_u``,
whose extra wires must end unchanged.

InnerPrepare -> InnerUnprepare: ``x_o`` is the read-only address, so each run starts from a
random complex amplitude on every value of ``x_o``, SF rows, one-body rows and failed values
alike; any permutation or relative phase of the rows is caught. The tables are random, with a
zero row (so zero erase masks) where the rank column is zero too; the shapes include dead
positions, failed ``b`` and partly covered fix-up leaves.

Prepare -> junk measurement -> unprepare from outcomes (spec 16, Q-1 variant (b)): the same
shapes, with ``H`` and ``qp.measure(q, reset=True)`` on every junk qubit right after the prepare
and the unprepare from those outcomes, its ``extra_onehot`` lent from the start of the measured
junk register. The outcome vector ``xi`` ranges over the junk outcomes, measured first, and the
target outcomes together: M = K d forced outcomes per run, M + 1 runs.
"""

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional

from dfthc.ops.prepare import (
    InnerPrepare,
    InnerUnprepare,
    InnerUnprepareFromOutcomes,
    OuterPrepare,
    OuterUnprepare,
    OuterUnprepareFromOutcomes,
)
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, lg
from dfthc.testing.sparse_sim import (
    ForcedPolicy,
    SparseSimulator,
    SparseState,
    nonlinear_conditionals,
)
from dfthc.testing.tape import decompose_for_simulation

pytestmark = pytest.mark.slow


def layout(sizes):
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def random_input_state(num_wires, inputs, seed=11):
    """A random complex amplitude on every basis value of ``inputs``, every other wire |0>."""
    rng = np.random.default_rng(seed)
    size = 2 ** len(inputs)
    amps = rng.normal(size=size) + 1j * rng.normal(size=size)
    keys = []
    for v in range(size):
        value = sum(((v >> (len(inputs) - 1 - i)) & 1) << (num_wires - 1 - wire)
                    for i, wire in enumerate(inputs))
        keys.append(SparseState.basis(num_wires, value).keys[0])
    return SparseState(num_wires, np.array(keys), amps / np.linalg.norm(amps))


def sorted_terms(state):
    order = np.lexsort(state.keys.T[::-1])
    return state.keys[order], state.amps[order]


def certify(prepare, unprepare, wires, num_wires, inputs, num_measured):
    """Run ``prepare`` then ``unprepare`` (after the phase-gradient state) under the zero and every
    unit schedule, from a random complex state of the ``inputs`` wires."""
    gradient = lambda: qp.PhaseGradientStatePrep(wires=wires["phase_grad"])  # noqa: E731

    def round_trip():
        gradient()
        prepare()
        unprepare()

    operations = decompose_for_simulation(round_trip).operations
    assert any(isinstance(op, Conditional) for op in operations)
    assert not nonlinear_conditionals(operations)
    assert sum(op.name == "MidMeasure" for op in operations) == num_measured
    initial = random_input_state(num_wires, inputs)
    reference = decompose_for_simulation(gradient).operations
    expected_keys, expected_amps = sorted_terms(
        SparseSimulator(num_wires).run(reference, initial_state=initial))
    schedules = [[0] * num_measured] + [
        [int(i == j) for j in range(num_measured)] for i in range(num_measured)]
    for schedule in schedules:
        simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
        keys, amps = sorted_terms(simulator.run(operations, initial_state=initial))
        assert simulator.policy.position == num_measured
        assert np.array_equal(keys, expected_keys), schedule
        assert np.allclose(amps, expected_amps, rtol=0, atol=1e-12), schedule


# (X_o, b_k1, k1, ke1, s_u): both test presets (X_o = 4 and 8, trivial uniform preparations);
# X_o = 5 with ke1 = 2 (the fix-up leaf [4, 8) covers only 4 and 5 of the loaded [0, 6); the
# uniform preparation fails with weight 0.15 and a phase; phase_grad has two wires more than s_u,
# as when Phase 3 passes the whole register); X_o = 6 with k1 = 2 (dead positions 6 and 7, l = 3).
OUTER = {
    "sto3g": (4, 2, 1, 1, 3),
    "631g": (8, 2, 1, 1, 3),
    "x-o-5-ke2": (5, 3, 1, 2, 3),
    "x-o-6-k2": (6, 2, 2, 1, 3),
}
assert OUTER["sto3g"] == tuple(getattr(H2_STO3G_TEST, f) for f in ("X_o", "b_k1", "k1", "ke1",
                                                                     "s_u"))
assert OUTER["631g"] == tuple(getattr(H2_631G_TEST, f) for f in ("X_o", "b_k1", "k1", "ke1",
                                                                  "s_u"))


def random_outer_tables(X_o, b_k1, seed=5):
    rng = np.random.default_rng(seed)
    return (tuple(int(a) for a in rng.integers(0, X_o, X_o)),
            tuple(int(k) for k in rng.integers(0, 2**b_k1, X_o)))


@pytest.mark.parametrize("name", list(OUTER))
def test_outer_prepare_unprepare_certificate(name):
    X_o, b_k1, k1, ke1, s_u = OUTER[name]
    args = (X_o, *random_outer_tables(X_o, b_k1), b_k1, k1, ke1, s_u)
    sizes = OuterPrepare.required_wires(X_o, b_k1, k1, ke1, s_u)
    sizes["phase_grad"] += 2 if name == "x-o-5-ke2" else 0
    wires, num_wires = layout(sizes)
    certify(lambda: OuterPrepare(*args, **wires), lambda: OuterUnprepare(*args, **wires), wires,
            num_wires, inputs=[], num_measured=2**k1 * (lg(X_o) + b_k1))


# (N, R, B, C, b_k2, k2, ke2, s_u): both test presets ((B + 1, k2) = (3, 1), l_b = 3); rows
# [0, 3) of a 3-bit x_o with (B + 1, k2) = (6, 2), row 0 zero (rank 0 as well, so its masks are
# zero) and the last fix-up leaf (x_o in {2, 3}) partly covered; rows [0, 2) with
# (B + 1, k2) = (5, 1): b = 5 is a dead position, and every fix-up leaf covers b in [0, 6) only.
INNER = {
    "sto3g": (2, 2, 2, 1, 2, 1, 1, 3),
    "631g": (4, 2, 2, 2, 2, 1, 1, 3),
    "inner-6-2-zero-row": (2, 3, 5, 1, 2, 2, 1, 3),
    "inner-5-1": (3, 2, 4, 1, 2, 1, 1, 3),
}
assert INNER["sto3g"] == tuple(getattr(H2_STO3G_TEST, f) for f in ("N", "R", "B", "C", "b_k2",
                                                                     "k2", "ke2", "s_u"))
assert INNER["631g"] == tuple(getattr(H2_631G_TEST, f) for f in ("N", "R", "B", "C", "b_k2",
                                                                  "k2", "ke2", "s_u"))


def random_inner_tables(R, B, C, b_k2, seed=5, zero_rows=()):
    """Random alt, keep, sgn and d_sgn of shape (R C, B + 1), zero on the rows ``zero_rows``."""
    rng = np.random.default_rng(seed)
    shape = (R * C, B + 1)
    columns = [rng.integers(0, B + 1, shape), rng.integers(0, 2**b_k2, shape),
               rng.integers(0, 2, shape), rng.integers(0, 2, shape)]
    return tuple(tuple(tuple(0 if x in zero_rows else int(v) for v in row)
                       for x, row in enumerate(column)) for column in columns)


@pytest.mark.parametrize("name", list(INNER))
def test_inner_prepare_unprepare_certificate(name):
    N, R, B, C, b_k2, k2, ke2, s_u = INNER[name]
    zero_rows = (0,) if name.endswith("zero-row") else ()
    args = (N, R, B, C, *random_inner_tables(R, B, C, b_k2, zero_rows=zero_rows), b_k2, k2, ke2,
            s_u)
    wires, num_wires = layout(InnerPrepare.required_wires(N, R, B, C, b_k2, k2, ke2, s_u))
    certify(lambda: InnerPrepare(*args, **wires), lambda: InnerUnprepare(*args, **wires), wires,
            num_wires, inputs=wires["x_o"],
            num_measured=2**k2 * (lg(B + 1) + b_k2 + 2 + lg(R)))


def from_outcomes(prepare, unprepare, args, wires, junk, structure):
    """``(prepare and junk measurement, unprepare from those outcomes)`` as two callables of one
    round trip; the unprepare's ``extra_onehot`` is the start of the measured junk register."""
    sizes = unprepare.required_wires(*structure)
    registers = {name: wires[name] for name in sizes if name != "extra_onehot"}
    registers["extra_onehot"] = wires[junk][:sizes["extra_onehot"]]
    outcomes = []

    def measure():
        prepare(*args, **wires)
        for wire in wires[junk]:
            qp.H(wire)
            outcomes.append(qp.measure(wire, reset=True))

    return measure, lambda: unprepare(outcomes, *args, **registers)


@pytest.mark.parametrize("name", list(OUTER))
def test_outer_unprepare_from_outcomes_certificate(name):
    X_o, b_k1, k1, ke1, s_u = OUTER[name]
    args = (X_o, *random_outer_tables(X_o, b_k1), b_k1, k1, ke1, s_u)
    sizes = OuterPrepare.required_wires(X_o, b_k1, k1, ke1, s_u)
    sizes["phase_grad"] += 2 if name == "x-o-5-ke2" else 0
    wires, num_wires = layout(sizes)
    certify(*from_outcomes(OuterPrepare, OuterUnprepareFromOutcomes, args, wires, "junk_x_o",
                           (X_o, b_k1, k1, ke1, s_u)),
            wires, num_wires, inputs=[], num_measured=2**k1 * (lg(X_o) + b_k1))


@pytest.mark.parametrize("name", list(INNER))
def test_inner_unprepare_from_outcomes_certificate(name):
    N, R, B, C, b_k2, k2, ke2, s_u = INNER[name]
    zero_rows = (0,) if name.endswith("zero-row") else ()
    args = (N, R, B, C, *random_inner_tables(R, B, C, b_k2, zero_rows=zero_rows), b_k2, k2, ke2,
            s_u)
    structure = (N, R, B, C, b_k2, k2, ke2, s_u)
    wires, num_wires = layout(InnerPrepare.required_wires(*structure))
    certify(*from_outcomes(InnerPrepare, InnerUnprepareFromOutcomes, args, wires, "junk_b",
                           structure),
            wires, num_wires, inputs=wires["x_o"],
            num_measured=2**k2 * (lg(B + 1) + b_k2 + 2 + lg(R)))
