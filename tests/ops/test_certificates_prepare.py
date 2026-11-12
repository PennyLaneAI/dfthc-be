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
"""

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional

from dfthc.ops.prepare import OuterPrepare, OuterUnprepare
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
