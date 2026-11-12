"""K-7: adjoints of PennyLane's comparators and of SemiAdder compile and are exact (Task 0.6).

Each test checks the forward result on every branch of a Hadamard superposition against an exact
probability table. It then prepares a non-uniform Ry product state, applies the primitive and its
adjoint, and compares the full state, including the work wires the forward pass leaves dirty,
with the prepared state. The input is non-uniform because a uniform superposition is invariant
under any permutation of the inputs (as the adder's y -> x + y is), so it cannot reveal a
missing or wrong adjoint.
"""

import itertools

import numpy as np
import pennylane as qp
import pytest

from dfthc.compile import FIXED_DECOMPS, GATE_SET

pytestmark = pytest.mark.compiled

# The project gate set has no rotations (R-9), so the Ry input preparation is kept as it is.
# The primitives under test still decompose to the project gate set.
TEST_GATE_SET = {**GATE_SET, "RY": 1}


def _program(num_wires, body, returns):
    @qp.qjit(capture=True, seed=7)
    @qp.decompose(gate_set=TEST_GATE_SET, fixed_decomps=dict(FIXED_DECOMPS) or None)
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return returns()

    return circuit


def _theta(k):
    """Ry angle of the k-th input wire: distinct, and never a multiple of pi/2."""
    return 0.3 + 0.4 * k


def _ry_prepare(wires):
    """Ry product state with distinct angles on `wires`; every other wire stays in |0>."""
    for k, wire in enumerate(wires):
        qp.RY(_theta(k), wire)


def _ry_state(num_wires, wires):
    """The state of `_ry_prepare(wires)` (wire 0 is the most significant bit)."""
    angles = {wire: _theta(k) for k, wire in enumerate(wires)}
    state = np.ones(1, dtype=complex)
    for wire in range(num_wires):
        theta = angles.get(wire, 0.0)
        state = np.kron(state, [np.cos(theta / 2), np.sin(theta / 2)])
    return state


@pytest.mark.parametrize("comparator", ["<=", "<"])
def test_left_quantum_comparator_and_adjoint(comparator):
    x, y, target, work = [0, 1], [2, 3], [4], [5]

    def prepare():
        for wire in x + y:
            qp.H(wire)

    def compare():
        return qp.LeftQuantumComparator(x, y, target, work, comparator=comparator)

    forward = _program(6, lambda: (prepare(), compare()), lambda: qp.probs(wires=x + y + target))
    probs = np.asarray(forward()).reshape(4, 4, 2)
    for a, b in itertools.product(range(4), repeat=2):
        expected = int(a <= b) if comparator == "<=" else int(a < b)
        assert np.isclose(probs[a, b, expected], 1 / 16), (a, b)

    round_trip = _program(
        6, lambda: (_ry_prepare(x + y), compare(), qp.adjoint(compare())), qp.state
    )
    assert np.allclose(np.asarray(round_trip()), _ry_state(6, x + y), atol=1e-10)


@pytest.mark.parametrize("value", [1, 2, 3])
def test_left_classical_comparator_and_adjoint(value):
    x, target, work = [0, 1, 2], [3], [4, 5]

    def prepare():
        for wire in x:
            qp.H(wire)

    def compare():
        return qp.LeftClassicalComparator(x, value=value, target_wire=target, work_wires=work,
                                          comparator="<")

    forward = _program(6, lambda: (prepare(), compare()), lambda: qp.probs(wires=x + target))
    probs = np.asarray(forward()).reshape(8, 2)
    for a in range(8):
        assert np.isclose(probs[a, int(a < value)], 1 / 8), a

    round_trip = _program(6, lambda: (_ry_prepare(x), compare(), qp.adjoint(compare())), qp.state)
    assert np.allclose(np.asarray(round_trip()), _ry_state(6, x), atol=1e-10)


def test_semiadder_and_adjoint():
    x, y, work = [0, 1, 2], [3, 4, 5], [6, 7]

    def prepare():
        for wire in x + y:
            qp.H(wire)

    def add():
        return qp.SemiAdder(x, y, work)

    forward = _program(8, lambda: (prepare(), add()), lambda: qp.probs(wires=x + y))
    probs = np.asarray(forward()).reshape(8, 8)
    for a, b in itertools.product(range(8), repeat=2):
        assert np.isclose(probs[a, (a + b) % 8], 1 / 64), (a, b)

    # Guard against a vacuous round trip: the adder alone must change the Ry input state.
    forward_only = _program(8, lambda: (_ry_prepare(x + y), add()), qp.state)
    assert not np.allclose(np.asarray(forward_only()), _ry_state(8, x + y), atol=1e-10)

    round_trip = _program(8, lambda: (_ry_prepare(x + y), add(), qp.adjoint(add())), qp.state)
    assert np.allclose(np.asarray(round_trip()), _ry_state(8, x + y), atol=1e-10)
