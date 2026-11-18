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

"""K-7: adjoints of PennyLane's comparators and of SemiAdder compile and are exact (Task 0.6).

Each test checks the forward result on every branch of a Hadamard superposition against an exact
probability table. It then prepares a phased stabilizer state, applies the primitive and its
adjoint, and compares the full state, including the work wires the forward pass leaves dirty,
with the prepared state. The input has non-uniform phases because a uniform superposition with
equal phases is invariant under any permutation of the inputs (as the adder's y -> x + y is), so
it cannot reveal a missing or wrong adjoint. It is built from project gates (H, S, CZ) so that
every program compiles to the project gate set unchanged.
"""

import itertools

import numpy as np
import pennylane as qp
import pytest

from dfthc.compile import FIXED_DECOMPS, GATE_SET

pytestmark = pytest.mark.compiled


def _program(num_wires, body, returns):
    @qp.qjit(capture=True, seed=7)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS) or None)
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return returns()

    return circuit


def _phased_prepare(wires):
    """Phased stabilizer state on `wires`; every other wire stays in |0>.

    H on every wire, S on the wires at even position k, CZ on each consecutive pair (k, k + 1).
    """
    for wire in wires:
        qp.H(wire)
    for k, wire in enumerate(wires):
        if k % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


def _phased_state(num_wires, wires):
    """The state of `_phased_prepare(wires)` (wire 0 is the most significant bit)."""
    state = np.zeros(2**num_wires, dtype=complex)
    n = len(wires)
    for z in itertools.product([0, 1], repeat=n):
        index = sum(bit << (num_wires - 1 - wire) for bit, wire in zip(z, wires, strict=True))
        quarter_turns = sum(z[k] for k in range(0, n, 2))
        sign = (-1) ** sum(z[k] * z[k + 1] for k in range(n - 1))
        state[index] = 1j**quarter_turns * sign / np.sqrt(2**n)
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
        assert np.isclose(probs[a, b, expected], 1 / 16, rtol=0, atol=1e-10), (a, b)

    round_trip = _program(
        6, lambda: (_phased_prepare(x + y), compare(), qp.adjoint(compare())), qp.state
    )
    assert np.allclose(np.asarray(round_trip()), _phased_state(6, x + y), rtol=0, atol=1e-10)


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
        assert np.isclose(probs[a, int(a < value)], 1 / 8, rtol=0, atol=1e-10), a

    round_trip = _program(
        6, lambda: (_phased_prepare(x), compare(), qp.adjoint(compare())), qp.state
    )
    assert np.allclose(np.asarray(round_trip()), _phased_state(6, x), rtol=0, atol=1e-10)


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
        assert np.isclose(probs[a, (a + b) % 8], 1 / 64, rtol=0, atol=1e-10), (a, b)

    # Guard against a vacuous round trip: the adder alone must change the phased input state.
    forward_only = _program(8, lambda: (_phased_prepare(x + y), add()), qp.state)
    assert not np.allclose(np.asarray(forward_only()), _phased_state(8, x + y), rtol=0, atol=1e-10)

    round_trip = _program(8, lambda: (_phased_prepare(x + y), add(), qp.adjoint(add())), qp.state)
    assert np.allclose(np.asarray(round_trip()), _phased_state(8, x + y), rtol=0, atol=1e-10)
