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


UNITARY = ["PauliX", "PauliY", "PauliZ", "Hadamard", "S", "Adjoint(S)", "CNOT", "CZ", "SWAP",
           "Toffoli", "GlobalPhase"]


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
                           _default_qubit(state, [op], num_wires), atol=1e-12)


def test_temporary_and_matches_default_qubit_on_its_domain():
    rng = np.random.default_rng(1)
    for _ in range(STATES_PER_HANDLER):
        num_wires = int(rng.integers(3, 11))
        wires = [int(w) for w in rng.choice(num_wires, 3, replace=False)]
        state = _zero_wire(_random_state(rng, num_wires), num_wires, wires[2])
        op = _gate("TemporaryAND", wires, rng)
        assert np.allclose(_sparse(state, [op], num_wires),
                           _default_qubit(state, [op], num_wires), atol=1e-12)


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
        assert np.allclose(got, _default_qubit(state, [op], num_wires), atol=1e-12)
        assert np.allclose(got, before, atol=1e-12)


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
                           _with_phase_gradient(state, num_wires, register), atol=1e-12)


def test_random_circuits_mixing_all_handlers_match():
    """Random gates on wires 0..7, a phase gradient on wires 6-7 and AND compute/uncompute pairs
    on ancilla wire 8, which every TemporaryAND finds in |0> and every adjoint leaves in |0>."""
    rng = np.random.default_rng(4)
    num_wires, data, ancilla = 9, list(range(8)), 8
    for _ in range(10):
        state = _random_state(rng, num_wires)
        for wire in (6, 7, ancilla):
            state = _zero_wire(state, num_wires, wire)
        operations = []
        for _ in range(40):
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
        prep = qp.PhaseGradientStatePrep([6, 7])
        assert np.allclose(_sparse(state, [prep, *operations], num_wires),
                           _default_qubit(_with_phase_gradient(state, num_wires, [6, 7]),
                                          operations, num_wires), atol=1e-12)


def test_global_phase_multiplies_by_exp_minus_i_phi():
    state = SparseSimulator(2).run([qp.H(0), qp.GlobalPhase(0.7)])
    assert np.allclose(state.to_dense(), np.exp(-0.7j) * np.array([1, 0, 1, 0]) / np.sqrt(2))


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
    with pytest.raises(UnsupportedOperation, match="RX"):
        SparseSimulator(2).run([qp.RX(0.1, 0)])
    with pytest.raises(UnsupportedOperation, match="MultiControlledX"):
        SparseSimulator(4).run([qp.MultiControlledX(wires=[0, 1, 2, 3])])
    with pytest.raises(ValueError, match="wire 5"):
        SparseSimulator(2).run([qp.X(5)])


def test_sparse_state_basics():
    state = SparseState.basis(5, 0b10010)
    assert np.flatnonzero(state.to_dense()).tolist() == [0b10010]
    assert state.probability(0, 1) == 1.0 and state.probability(1, 1) == 0.0
    dense = _random_state(np.random.default_rng(5), 4)
    dense[3] = 1e-16
    round_trip = SparseState.from_dense(dense)
    assert len(round_trip.amps) == 15  # the tiny amplitude is pruned
    assert np.allclose(round_trip.to_dense(), np.where(np.abs(dense) < 1e-14, 0, dense))
    weights = np.abs(round_trip.to_dense()) ** 2
    wire_2_set = [2, 3, 6, 7, 10, 11, 14, 15]
    assert np.isclose(round_trip.probability(2, 1), weights[wire_2_set].sum() / weights.sum())
    with pytest.raises(ValueError, match="25 wires"):
        SparseState.basis(26, 0).to_dense()


def test_keys_span_several_words():
    """130 wires use three uint64 words; wire 0 stays the most significant bit."""
    sim = SparseSimulator(130)
    state = sim.run([qp.X(100), qp.H(129), qp.CNOT([129, 3]), qp.SWAP([100, 64])])
    assert state.keys.shape == (2, 3)
    assert np.isclose(state.probability(64, 1), 1.0) and state.probability(100, 1) == 0.0
    assert np.isclose(state.probability(3, 1), 0.5) and np.isclose(state.probability(129, 1), 0.5)
    state.amps[0] = 1e-15
    assert len(state.prune().amps) == 1
