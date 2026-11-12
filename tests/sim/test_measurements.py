"""Mid-circuit measurements, conditionals and measurement policies (spec section 12.3).

Branch-resolved simulations are compared with default.qubit under deferred measurements
(analytic): the mixture of all branches equals the reduced density matrix of the deferred circuit.
"""

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional

from dfthc.testing.sparse_sim import (
    Branch,
    BranchPolicy,
    ForcedPolicy,
    SeededPolicy,
    SparseSimulator,
    SparseState,
)


def _tape(build):
    with qp.queuing.AnnotatedQueue() as queue:
        build()
    return qp.tape.QuantumScript.from_queue(queue)


def _deferred_density_matrix(operations, num_wires):
    tape = qp.tape.QuantumScript(operations, [qp.density_matrix(wires=range(num_wires))])
    [deferred], _ = qp.defer_measurements(tape)
    return np.asarray(qp.devices.DefaultQubit().execute(deferred))


def _mixture(branches):
    return sum(b.probability * np.outer(v, v.conj())
               for b in branches for v in [b.state.to_dense()])


def _random_circuit(rng, num_wires, num_ops):
    """Gate-set operations, measurements (some with reset) and conditionals on composite
    expressions of earlier outcomes: XOR, sums mod 2 and a single outcome."""
    gates = [lambda w: qp.H(w[0]), lambda w: qp.S(w[0]), lambda w: qp.CNOT(w[:2]),
             lambda w: qp.Toffoli(w), lambda w: qp.Y(w[0]), lambda w: qp.CZ(w[:2])]
    outcomes = []

    def build():
        for wire in range(num_wires):
            qp.H(wire)
        for _ in range(num_ops):
            wires = [int(w) for w in rng.choice(num_wires, 3, replace=False)]
            choice = rng.random()
            if choice < 0.2:
                outcomes.append(qp.measure(wires[0], reset=bool(rng.integers(2))))
            elif choice < 0.4 and len(outcomes) >= 2:
                a, b = (outcomes[int(i)] for i in rng.choice(len(outcomes), 2, replace=False))
                expression = [a ^ b, (a + b) % 2, a][int(rng.integers(3))]
                qp.cond(expression, [qp.X, qp.Z][int(rng.integers(2))])(wires[0])
            else:
                gates[int(rng.integers(len(gates)))](wires)

    return _tape(build).operations


def test_mid_measure_and_conditional_are_supported():
    assert {"MidMeasure", "Conditional"} <= SparseSimulator(1).supported_operations


@pytest.mark.parametrize("seed", range(8))
def test_branches_match_default_qubit_with_deferred_measurements(seed):
    rng = np.random.default_rng(seed)
    num_wires = 5
    operations = _random_circuit(rng, num_wires, 30)
    sim = SparseSimulator(num_wires, policy=BranchPolicy(range(num_wires)), strict=False)
    branches = sim.run_branches(operations)
    assert sum(b.probability for b in branches) == pytest.approx(1.0, abs=1e-12)
    assert all(isinstance(b, Branch) for b in branches)
    assert np.allclose(_mixture(branches), _deferred_density_matrix(operations, num_wires),
                       atol=1e-12)


def test_reset_and_postselect_match_default_qubit():
    for postselect in (0, 1):
        def build(postselect=postselect):
            qp.H(0)
            qp.CNOT([0, 1])
            qp.S(1)
            m = qp.measure(0, reset=True, postselect=postselect)
            m2 = qp.measure(1, reset=False)
            qp.cond(m ^ m2, qp.X)(2)
            qp.cond((m + m2) % 2, qp.H)(3)

        operations = _tape(build).operations
        sim = SparseSimulator(4, policy=BranchPolicy([1]), strict=False)
        branches = sim.run_branches(operations)
        assert len(branches) == 1 and branches[0].record[operations[3]] == postselect
        assert branches[0].state.probability(0, 1) == 0.0  # reset
        assert np.allclose(_mixture(branches), _deferred_density_matrix(operations, 4),
                           atol=1e-12)


def test_postselection_after_branching_conditions_the_branches():
    """Wire 1 = (w0 and w2) or (w3 and w4) passes the postselection with probability 0, 1/2, 1/2
    and 3/4 in the branches (w0, w3) = 00, 01, 10, 11: branch 00 is dropped, and the others carry
    p q / sum(p q) = 2/7, 2/7, 3/7, as default.qubit gives under the same postselection."""
    outcomes = {}

    def build():
        for wire in (0, 2, 3, 4):
            qp.H(wire)
        qp.Toffoli([0, 2, 5])
        qp.Toffoli([3, 4, 6])
        qp.CNOT([5, 1])
        qp.CNOT([6, 1])
        qp.Toffoli([5, 6, 1])
        outcomes["w0"], outcomes["w3"] = qp.measure(0), qp.measure(3)
        qp.measure(1, postselect=1)

    operations = _tape(build).operations
    branches = SparseSimulator(7, policy=BranchPolicy([0, 3]), strict=False).run_branches(
        operations)
    measured = [op for op in operations if op.name == "MidMeasure"][:2]
    probabilities = np.zeros(4)
    for branch in branches:
        probabilities[2 * branch.record[measured[0]] + branch.record[measured[1]]] = (
            branch.probability)
    reference = qp.tape.QuantumScript(operations, [qp.probs(op=[outcomes["w0"], outcomes["w3"]]),
                                                   qp.density_matrix(wires=range(7))])
    [deferred], _ = qp.defer_measurements(reference)
    probs, rho = qp.devices.DefaultQubit().execute(deferred)
    assert np.allclose(probabilities, [0, 2 / 7, 2 / 7, 3 / 7], atol=1e-12)
    assert np.allclose(probabilities, probs, atol=1e-12)
    assert np.allclose(_mixture(branches), rho, atol=1e-12)
    impossible = _tape(lambda: (qp.H(0), qp.measure(0), qp.measure(1, postselect=1))).operations
    with pytest.raises(ValueError, match="no branch can pass the postselection"):
        SparseSimulator(2, policy=BranchPolicy([0]), strict=False).run_branches(impossible)


def test_a_postselected_outcome_of_probability_zero_raises():
    operations = _tape(lambda: qp.measure(0, postselect=1)).operations
    with pytest.raises(ValueError, match="probability 0"):
        SparseSimulator(1, policy=ForcedPolicy([0]), strict=False).run(operations)


def test_forced_schedule_is_honoured_and_recorded():
    def build():
        for wire in range(3):
            qp.H(wire)
            qp.measure(wire, reset=True)

    operations = _tape(build).operations
    policy = ForcedPolicy([1, 0, 1])
    sim = SparseSimulator(3, policy=policy)
    state = sim.run(operations)
    measurements = [op for op in operations if op.name == "MidMeasure"]
    assert [sim.record[m] for m in measurements] == [1, 0, 1]
    assert np.allclose(state.to_dense(), SparseState.basis(3, 0).to_dense())  # all reset
    assert policy.position == 3
    with pytest.raises(ValueError, match="schedule of 3 outcomes is exhausted"):
        sim.run(operations + operations[:2])
    with pytest.raises(ValueError, match="0 or 1"):
        ForcedPolicy([2])


def test_a_forced_outcome_of_probability_zero_raises():
    operations = _tape(lambda: qp.measure(0)).operations
    with pytest.raises(ValueError, match="probability 0"):
        SparseSimulator(1, policy=ForcedPolicy([1]), strict=False).run(operations)


def test_strict_mode_asserts_uniform_erasure_outcomes():
    unbalanced = _tape(lambda: (qp.H(0), qp.S(0), qp.CNOT([0, 1]), qp.measure(1))).operations
    assert SparseSimulator(2, policy=ForcedPolicy([0])).run(unbalanced) is not None
    biased = _tape(lambda: qp.measure(0)).operations
    with pytest.raises(AssertionError, match="P\\(1\\) = 0.0"):
        SparseSimulator(1, policy=ForcedPolicy([0])).run(biased)
    # Branch wires and postselected measurements are exempt.
    assert len(SparseSimulator(1, policy=BranchPolicy([0])).run_branches(biased)) == 1
    postselected = _tape(lambda: qp.measure(0, postselect=0)).operations
    assert SparseSimulator(1, policy=ForcedPolicy([])).run(postselected) is not None


def test_measurements_renormalize_instead_of_compounding_norm_errors():
    """Probabilities are relative to the norm, and each measurement restores norm 1. Dividing by
    1 - P(1) instead doubles a norm error at every measurement (found on the K-1 stand-in walk,
    whose 50 erase measurements then drove the norm down to 0.13)."""
    def build():
        for wire in range(20):
            qp.H(wire)
            qp.measure(wire, reset=True)

    start = SparseState.basis(20, 0)
    start.amps *= np.sqrt(1 - 1e-6)
    state = SparseSimulator(20, policy=ForcedPolicy([0, 1] * 10)).run(_tape(build).operations,
                                                                     initial_state=start)
    assert np.sum(np.abs(state.amps) ** 2) == pytest.approx(1.0, abs=1e-14)


def test_seeded_policy_is_reproducible_and_follows_the_probabilities():
    def build():
        qp.X(0)
        for wire in range(1, 9):
            qp.H(wire)
            qp.measure(wire)
        qp.measure(0)

    operations = _tape(build).operations
    runs = []
    for _ in range(2):
        sim = SparseSimulator(9, policy=SeededPolicy(7), strict=False)
        sim.run(operations)
        runs.append([sim.record[op] for op in operations if op.name == "MidMeasure"])
    assert runs[0] == runs[1] and runs[0][-1] == 1 and 0 < sum(runs[0][:-1]) < 8


def test_branch_policy_delegates_other_wires_and_each_branch_restarts_the_schedule():
    operations = _tape(lambda: (qp.H(0), qp.H(1), qp.measure(0), qp.measure(1))).operations
    policy = BranchPolicy([0], otherwise=ForcedPolicy([1]))
    branches = SparseSimulator(2, policy=policy).run_branches(operations)
    assert sorted(b.record[operations[2]] for b in branches) == [0, 1]
    assert all(b.record[operations[3]] == 1 for b in branches)
    assert [b.probability for b in branches] == pytest.approx([0.5, 0.5])
    assert policy.otherwise.position == 0  # the branches consumed copies
    with pytest.raises(ValueError, match="run_branches"):
        SparseSimulator(2, policy=policy).run(operations)
    with pytest.raises(ValueError, match="no policy"):
        SparseSimulator(2, policy=BranchPolicy([0])).run_branches(operations)
    with pytest.raises(ValueError, match="no policy"):
        SparseSimulator(2).run(operations)


def test_nested_conditionals_dispatch_recursively():
    def build():
        qp.X(0)
        qp.X(1)
        a, b = qp.measure(0), qp.measure(1)
        Conditional(a, Conditional(b, qp.X(2)))
        Conditional(a, Conditional(b ^ a, qp.X(3)))

    operations = _tape(build).operations
    state = SparseSimulator(4, policy=ForcedPolicy([1, 1]), strict=False).run(operations)
    assert np.flatnonzero(state.to_dense()).tolist() == [0b1110]
