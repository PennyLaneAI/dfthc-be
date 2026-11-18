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

"""Sparse simulator core: state, unitary handlers, preconditions (spec section 12.3).

Every handler is compared with PennyLane's default.qubit statevector on random states.
"""

import numpy as np
import pennylane as qp
import pytest

from dfthc.compile import GATE_SET
from dfthc.testing.sparse_sim import SparseSimulator, SparseState, UnsupportedOperation

STATES_PER_HANDLER = 20


def _gate(name, wires, rng):
    """One instance of a gate-set operation on ``wires`` (3 wires for every gate)."""
    a, b, c = wires
    return {
        "PauliX": lambda: qp.X(a), "PauliY": lambda: qp.Y(a), "PauliZ": lambda: qp.Z(a),
        "Hadamard": lambda: qp.H(a), "S": lambda: qp.S(a),
        "Adjoint(S)": lambda: qp.adjoint(qp.S(a)), "CNOT": lambda: qp.CNOT([a, b]),
        "CZ": lambda: qp.CZ([a, b]), "SWAP": lambda: qp.SWAP([a, b]),
        "Toffoli": lambda: qp.Toffoli([a, b, c]),
        "GlobalPhase": lambda: qp.GlobalPhase(rng.uniform(-np.pi, np.pi)),
        "TemporaryAND": lambda: qp.TemporaryAND([a, b, c], control_values=rng.integers(0, 2, 2)),
    }[name]()


def _random_state(rng, num_wires):
    state = rng.standard_normal(2**num_wires) + 1j * rng.standard_normal(2**num_wires)
    return state / np.linalg.norm(state)


def _zero_wire(state, num_wires, wire):
    """Project ``wire`` onto |0> and renormalize."""
    tensor = state.reshape((2,) * num_wires).copy()
    np.moveaxis(tensor, wire, 0)[1] = 0
    return tensor.ravel() / np.linalg.norm(tensor)


def _with_phase_gradient(state, num_wires, register):
    """``state`` (|0> on ``register``) with PennyLane's phase-gradient state on ``register``.

    default.qubit cannot apply PhaseGradientStatePrep after other operations (it has no matrix),
    so its reference is the operator's own ``state_vector()``.
    """
    index = tuple(0 if wire in register else slice(None) for wire in range(num_wires))
    rest = state.reshape((2,) * num_wires)[index]
    prep = np.asarray(qp.PhaseGradientStatePrep(register).state_vector())
    full = np.multiply.outer(rest, prep.reshape((2,) * len(register)))
    order = [wire for wire in range(num_wires) if wire not in register] + list(register)
    return full.transpose(np.argsort(order)).ravel()


def _default_qubit(state, operations, num_wires):
    tape = qp.tape.QuantumScript([qp.StatePrep(state, wires=range(num_wires)), *operations],
                                 [qp.state()])
    return np.asarray(qp.devices.DefaultQubit().execute(tape))


def _sparse(state, operations, num_wires):
    sim = SparseSimulator(num_wires)
    return sim.run(operations, initial_state=SparseState.from_dense(state)).to_dense()


def _embed(dense, wires, num_wires):
    """``dense`` (on ``len(wires)`` wires) placed on ``wires`` of a ``num_wires`` state."""
    small = SparseState.from_dense(dense)
    words = max(1, -(-num_wires // 64))
    state = SparseState(num_wires, np.zeros((len(small.amps), words), dtype=np.uint64),
                        small.amps.copy())
    for position, wire in enumerate(wires):
        state.flip(wire, small.bits(position))
    return state


def _restrict(state, wires):
    """The dense vector on ``wires``; every other wire must be 0 on every term."""
    for wire in set(range(state.num_wires)) - set(wires):
        assert not state.bits(wire).any(), f"wire {wire} was touched"
    index = np.zeros(len(state.amps), dtype=np.int64)
    for wire in wires:
        index = (index << 1) | state.bits(wire)
    vector = np.zeros(2 ** len(wires), dtype=complex)
    np.add.at(vector, index, state.amps)
    return vector


SPREAD = [0, 63, 127, 128, 1, 65, 129, 62, 64]
"""Logical wires 0..8 on 130 wires: data 0..5 straddle both word boundaries, the phase gradient
(6, 7) spans words 2 and 0, and the AND ancilla (8) is the first wire of word 1."""

UNITARY = ["PauliX", "PauliY", "PauliZ", "Hadamard", "S", "Adjoint(S)", "CNOT", "CZ", "SWAP",
           "Toffoli", "GlobalPhase"]


def _random_circuit(rng, data, steps):
    """A random 9-wire state with |0> on wires 6, 7 and 8, and ``steps`` random steps on ``data``:
    a unitary gate-set operation, or (with probability 0.15) an AND compute, CZ and uncompute on
    ancilla wire 8, which every TemporaryAND finds in |0> and every adjoint leaves in |0>."""
    num_wires, ancilla = 9, 8
    state = _random_state(rng, num_wires)
    for wire in (6, 7, ancilla):
        state = _zero_wire(state, num_wires, wire)
    operations = []
    for _ in range(steps):
        if rng.random() < 0.15:
            a, b, c = (int(w) for w in rng.choice(data, 3, replace=False))
            values = rng.integers(0, 2, 2)
            operations += [qp.TemporaryAND([a, b, ancilla], control_values=values),
                           qp.CZ([ancilla, c]),
                           qp.adjoint(qp.TemporaryAND([a, b, ancilla], control_values=values))]
        else:
            name = UNITARY[int(rng.integers(len(UNITARY)))]
            operations.append(_gate(name, [int(w) for w in rng.choice(data, 3, replace=False)],
                                    rng))
    return state, operations


def test_every_gate_set_member_but_mid_measure_has_a_handler():
    """MidMeasure joins with the measurement policies (Task 1B.2)."""
    assert set(GATE_SET) - {"MidMeasure"} <= SparseSimulator(1).supported_operations


@pytest.mark.parametrize("name", UNITARY)
def test_unitary_handler_matches_default_qubit(name):
    rng = np.random.default_rng(100 + UNITARY.index(name))
    for _ in range(STATES_PER_HANDLER):
        num_wires = int(rng.integers(3, 11))
        state = _random_state(rng, num_wires)
        op = _gate(name, [int(w) for w in rng.choice(num_wires, 3, replace=False)], rng)
        assert np.allclose(_sparse(state, [op], num_wires),
                           _default_qubit(state, [op], num_wires), rtol=0, atol=1e-12)


def test_temporary_and_matches_default_qubit_on_its_domain():
    rng = np.random.default_rng(1)
    for _ in range(STATES_PER_HANDLER):
        num_wires = int(rng.integers(3, 11))
        wires = [int(w) for w in rng.choice(num_wires, 3, replace=False)]
        state = _zero_wire(_random_state(rng, num_wires), num_wires, wires[2])
        op = _gate("TemporaryAND", wires, rng)
        assert np.allclose(_sparse(state, [op], num_wires),
                           _default_qubit(state, [op], num_wires), rtol=0, atol=1e-12)


def test_adjoint_temporary_and_matches_default_qubit_on_its_domain():
    rng = np.random.default_rng(2)
    for _ in range(STATES_PER_HANDLER):
        num_wires = int(rng.integers(3, 11))
        wires = [int(w) for w in rng.choice(num_wires, 3, replace=False)]
        values = rng.integers(0, 2, 2)
        before = _zero_wire(_random_state(rng, num_wires), num_wires, wires[2])
        state = _default_qubit(before, [qp.TemporaryAND(wires, control_values=values)],
                               num_wires)
        op = qp.adjoint(qp.TemporaryAND(wires, control_values=values))
        got = _sparse(state, [op], num_wires)
        assert np.allclose(got, _default_qubit(state, [op], num_wires), rtol=0, atol=1e-12)
        assert np.allclose(got, before, rtol=0, atol=1e-12)


def test_phase_gradient_state_prep_matches_pennylane_on_its_domain():
    rng = np.random.default_rng(3)
    for _ in range(STATES_PER_HANDLER):
        num_wires = int(rng.integers(3, 11))
        size = int(rng.integers(1, num_wires))
        register = [int(w) for w in rng.choice(num_wires, size, replace=False)]
        state = _random_state(rng, num_wires)
        for wire in register:
            state = _zero_wire(state, num_wires, wire)
        op = qp.PhaseGradientStatePrep(register)
        assert np.allclose(_sparse(state, [op], num_wires),
                           _with_phase_gradient(state, num_wires, register), rtol=0, atol=1e-12)


def test_random_circuits_mixing_all_handlers_match():
    """Random circuits after a phase gradient on wires 6-7, with AND compute/uncompute pairs on
    ancilla wire 8 (see ``_random_circuit``), match default.qubit to 1e-12 absolute.

    Ten circuits of 40 steps on 9 wires put gates on wires 0..7. Ten circuits of 60 steps put
    gates on wires 0..5 and run on 130 wires with every logical wire moved across the uint64 word
    boundaries (``SPREAD``); there the keys also stay unique and every other wire stays 0.
    """
    num_wires = 9
    rng = np.random.default_rng(4)
    for _ in range(10):
        state, operations = _random_circuit(rng, list(range(8)), 40)
        prep = qp.PhaseGradientStatePrep([6, 7])
        assert np.allclose(_sparse(state, [prep, *operations], num_wires),
                           _default_qubit(_with_phase_gradient(state, num_wires, [6, 7]),
                                          operations, num_wires), rtol=0, atol=1e-12)
    rng = np.random.default_rng(6)
    wire_map = dict(enumerate(SPREAD))
    for _ in range(10):
        state, operations = _random_circuit(rng, list(range(6)), 60)
        mapped = [qp.map_wires(op, wire_map) for op in
                  [qp.PhaseGradientStatePrep([6, 7]), *operations]]
        got = SparseSimulator(130).run(mapped, initial_state=_embed(state, SPREAD, 130))
        assert len(np.unique(got.keys, axis=0)) == len(got.amps)
        expected = _default_qubit(_with_phase_gradient(state, num_wires, [6, 7]), operations,
                                  num_wires)
        assert np.allclose(_restrict(got, SPREAD), expected, rtol=0, atol=1e-12)


def test_global_phase_multiplies_by_exp_minus_i_phi():
    state = SparseSimulator(2).run([qp.H(0), qp.GlobalPhase(0.7)])
    assert np.allclose(state.to_dense(), np.exp(-0.7j) * np.array([1, 0, 1, 0]) / np.sqrt(2),
                       rtol=0, atol=1e-12)


def test_temporary_and_asserts_a_zero_target():
    sim = SparseSimulator(3)
    with pytest.raises(AssertionError, match="target is 1 on key 001"):
        sim.run([qp.X(2), qp.TemporaryAND([0, 1, 2])])


def test_adjoint_temporary_and_asserts_the_target_equals_the_and():
    sim = SparseSimulator(3)
    with pytest.raises(AssertionError, match="target differs from the AND on key 110"):
        sim.run([qp.X(0), qp.X(1), qp.adjoint(qp.TemporaryAND([0, 1, 2]))])
    with pytest.raises(AssertionError, match="on key 011"):
        sim.run([qp.X(1), qp.X(2), qp.adjoint(qp.TemporaryAND([0, 1, 2], control_values=[1, 1]))])


def test_phase_gradient_state_prep_asserts_a_zero_register():
    with pytest.raises(AssertionError, match=r"register is not \|0> on key 0010"):
        SparseSimulator(4).run([qp.X(2), qp.PhaseGradientStatePrep([1, 2, 3])])


def test_other_operations_are_rejected():
    """Operations without a handler, wires out of range and initial states of another width are
    rejected; equal keys in an initial state are merged, so its probabilities are right."""
    with pytest.raises(UnsupportedOperation, match="RX"):
        SparseSimulator(2).run([qp.RX(0.1, 0)])
    with pytest.raises(UnsupportedOperation, match="MultiControlledX"):
        SparseSimulator(4).run([qp.MultiControlledX(wires=[0, 1, 2, 3])])
    with pytest.raises(ValueError, match="wire 5"):
        SparseSimulator(2).run([qp.X(5)])
    with pytest.raises(ValueError, match="initial_state has 2 wires"):
        SparseSimulator(3).run([qp.X(2)], initial_state=SparseState.basis(2, 0))
    with pytest.raises(ValueError, match="initial_state has 4 wires"):
        SparseSimulator(3).run([], initial_state=SparseState.basis(4, 0))
    narrow_keys = SparseState(70, np.zeros((1, 1), dtype=np.uint64), np.ones(1, dtype=complex))
    with pytest.raises(ValueError, match="width 1; the simulator has 70 wires and key width 2"):
        SparseSimulator(70).run([], initial_state=narrow_keys)
    keys = np.array([[0], [1 << 62], [1 << 62]], dtype=np.uint64)  # |00>, |01>, |01>
    amps = np.array([1, 0.5, 0.5], dtype=complex) / np.sqrt(2)
    merged = SparseSimulator(2).run([], initial_state=SparseState(2, keys, amps))
    assert merged.keys.tolist() == [[0], [1 << 62]]
    assert np.allclose(merged.amps, 1 / np.sqrt(2), rtol=0, atol=1e-15)
    assert np.isclose(merged.probability(1, 1), 0.5, rtol=0, atol=1e-12)


def test_sparse_state_basics():
    state = SparseState.basis(5, 0b10010)
    assert np.flatnonzero(state.to_dense()).tolist() == [0b10010]
    assert state.probability(0, 1) == 1.0 and state.probability(1, 1) == 0.0
    assert state.bits(np.int64(3)).tolist() == [True]
    for wire in (-1, 5, 63, 64):  # below 0, num_wires, a padding wire of word 0, word 1
        for check in (state.bits, state.flip, lambda w: state.probability(w, 1)):
            with pytest.raises(ValueError, match=rf"wire {wire} outside \[0, 5\)"):
                check(wire)
    dense = _random_state(np.random.default_rng(5), 4)
    dense[3] = 1e-16
    round_trip = SparseState.from_dense(dense)
    assert len(round_trip.amps) == 15  # the tiny amplitude is pruned
    assert np.array_equal(round_trip.to_dense(), np.where(np.abs(dense) < 1e-14, 0, dense))
    weights = np.abs(round_trip.to_dense()) ** 2
    wire_2_set = [2, 3, 6, 7, 10, 11, 14, 15]
    assert np.isclose(round_trip.probability(2, 1), weights[wire_2_set].sum() / weights.sum(),
                      rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="25 wires"):
        SparseState.basis(26, 0).to_dense()


def test_keys_span_several_words():
    """130 wires use three uint64 words, with wire w at bit 63 - w % 64 of word w // 64.

    X, H, CNOT and SWAP move bits within and across words, and pruning drops a small amplitude.
    Then H H on wire 100 is the identity on terms that differ in word 0 only (wire 1), word 1 only
    (wire 64), word 2 only (wire 129) and words 0 and 1 together (wires 0 and 127): the Hadamard
    merges equal keys, the 16 terms come back with amplitude 1/4, and the cancelled ones are gone
    (a TemporaryAND on wire 100 finds |0> on every term).
    """
    sim = SparseSimulator(130)
    state = sim.run([qp.X(100), qp.H(129), qp.CNOT([129, 3]), qp.SWAP([100, 64])])
    assert state.keys.shape == (2, 3)
    assert np.isclose(state.probability(64, 1), 1.0, rtol=0, atol=1e-12)
    assert state.probability(100, 1) == 0.0
    assert np.isclose(state.probability(3, 1), 0.5, rtol=0, atol=1e-12)
    assert np.isclose(state.probability(129, 1), 0.5, rtol=0, atol=1e-12)
    state.amps[0] = 1e-15
    assert len(state.prune().amps) == 1
    state = sim.run([qp.H(0), qp.H(1), qp.H(64), qp.H(129), qp.CNOT([0, 127]),
                     qp.H(100), qp.H(100), qp.TemporaryAND([1, 64, 100])])
    assert len(state.amps) == 16 and len(np.unique(state.keys, axis=0)) == 16
    assert np.allclose(np.abs(state.amps), 0.25, rtol=0, atol=1e-15)
    assert np.isclose(state.probability(100, 1), 0.25, rtol=0, atol=1e-15)
    value = sum(1 << (129 - wire) for wire in (0, 63, 64, 127, 128, 129))
    assert SparseState.basis(130, value).keys.tolist() == [
        [(1 << 63) | 1, (1 << 63) | 1, (1 << 63) | (1 << 62)]]
