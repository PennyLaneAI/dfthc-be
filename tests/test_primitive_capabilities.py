"""K-7: adjoints of PennyLane's comparators and of SemiAdder compile and are exact (Task 0.6).

Each test prepares a uniform superposition over every input, applies the primitive, checks the
result on every branch, then applies the adjoint and checks that the state, including the work
wires the forward pass leaves dirty, returns exactly to the input superposition.
"""

import itertools

import numpy as np
import pennylane as qp
import pytest

from dfthc.compile import FIXED_DECOMPS, GATE_SET

pytestmark = pytest.mark.compiled

N = 2  # bits per operand


def _program(num_wires, body, returns):
    @qp.qjit(capture=True, seed=7)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS) or None)
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return returns()

    return circuit


def _uniform_state(num_wires, active):
    """Uniform superposition over the `active` wires, every other wire in |0>."""
    state = np.zeros(2**num_wires, dtype=complex)
    for bits in itertools.product([0, 1], repeat=len(active)):
        index = sum(bit << (num_wires - 1 - wire) for bit, wire in zip(bits, active, strict=True))
        state[index] = 1 / np.sqrt(2 ** len(active))
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

    round_trip = _program(6, lambda: (prepare(), compare(), qp.adjoint(compare())), qp.state)
    assert np.allclose(np.asarray(round_trip()), _uniform_state(6, x + y), atol=1e-10)


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

    round_trip = _program(6, lambda: (prepare(), compare(), qp.adjoint(compare())), qp.state)
    assert np.allclose(np.asarray(round_trip()), _uniform_state(6, x), atol=1e-10)


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

    round_trip = _program(8, lambda: (prepare(), add(), qp.adjoint(add())), qp.state)
    assert np.allclose(np.asarray(round_trip()), _uniform_state(8, x + y), atol=1e-10)
