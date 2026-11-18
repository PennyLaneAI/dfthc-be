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

"""Mid-circuit measurements, conditionals and measurement policies (spec section 12.3).

Branch-resolved simulations are compared with default.qubit under deferred measurements
(analytic): the mixture of all branches equals the reduced density matrix of the deferred circuit.
"""

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional, MidMeasure

from dfthc.compile import ParityPredicate, mcm_parities
from dfthc.testing.sparse_sim import (
    Branch,
    BranchPolicy,
    ForcedPolicy,
    SeededPolicy,
    SparseSimulator,
    SparseState,
    nonlinear_conditionals,
)


def _tape(build):
    with qp.queuing.AnnotatedQueue() as queue:
        build()
    return qp.tape.QuantumScript.from_queue(queue)


def _one_qubit(probability_one):
    """``sqrt(1 - p)|0> + sqrt(p)|1>``: measuring it gives 1 with probability ``p``."""
    return SparseState.from_dense([np.sqrt(1 - probability_one), np.sqrt(probability_one)])


def _zero_norm_state():
    """A one-wire state without terms: every outcome probability is 0/0 = NaN."""
    return SparseState(1, np.zeros((0, 1), dtype=np.uint64), np.zeros(0, dtype=complex))


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
    """The mixture of all branches equals the deferred density matrix, from a random start state.
    From |0...0>, every branch outcome of these Clifford-plus-Toffoli circuits has probability 1/2
    or 1, so the final normalization would hide a missing outcome weight; a random start makes the
    outcome probabilities generic."""
    rng = np.random.default_rng(seed)
    num_wires = 5
    operations = _random_circuit(rng, num_wires, 30)
    vector = rng.normal(size=2**num_wires) + 1j * rng.normal(size=2**num_wires)
    vector /= np.linalg.norm(vector)
    sim = SparseSimulator(num_wires, policy=BranchPolicy(range(num_wires)), strict=False)
    branches = sim.run_branches(operations, initial_state=SparseState.from_dense(vector))
    assert sum(b.probability for b in branches) == pytest.approx(1.0, rel=0, abs=1e-12)
    assert all(isinstance(b, Branch) for b in branches)
    expected = _deferred_density_matrix(
        [qp.StatePrep(vector, wires=range(num_wires)), *operations], num_wires)
    assert np.allclose(_mixture(branches), expected, rtol=0, atol=1e-12)


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
                           rtol=0, atol=1e-12)


def test_postselection_after_branching_conditions_the_branches():
    """Wire 1 = (w0 and w2) or (w3 and w4) passes the postselection with probability 0, 1/2, 1/2
    and 3/4 in the branches (w0, w3) = 00, 01, 10, 11: branch 00 is dropped, and the others carry
    p q / sum(p q) = 2/7, 2/7, 3/7, as default.qubit gives under the same postselection. Wire 1
    is also a branch wire: its postselection takes precedence over branching."""
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
    branches = SparseSimulator(7, policy=BranchPolicy([0, 1, 3]), strict=False).run_branches(
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
    assert np.allclose(probabilities, [0, 2 / 7, 2 / 7, 3 / 7], rtol=0, atol=1e-12)
    assert np.allclose(probabilities, probs, rtol=0, atol=1e-12)
    assert np.allclose(_mixture(branches), rho, rtol=0, atol=1e-12)
    impossible = _tape(lambda: (qp.H(0), qp.measure(0), qp.measure(1, postselect=1))).operations
    with pytest.raises(ValueError, match="no branch can pass the postselection"):
        SparseSimulator(2, policy=BranchPolicy([0]), strict=False).run_branches(impossible)


def test_a_postselected_outcome_of_probability_zero_raises():
    """A postselected outcome below MEASUREMENT_TOLERANCE (1e-12) raises in ``run`` and drops its
    branch in ``run_branches``; just above the tolerance it is kept and renormalized."""
    operations = _tape(lambda: qp.measure(0, postselect=1)).operations
    with pytest.raises(ValueError, match="probability 0"):
        SparseSimulator(1, policy=ForcedPolicy([0]), strict=False).run(operations)
    below, above = _one_qubit(0.5e-12), _one_qubit(2e-12)
    sim = SparseSimulator(1, strict=False)
    with pytest.raises(ValueError, match="probability 5e-13"):
        sim.run(operations, initial_state=below)
    assert np.allclose(sim.run(operations, initial_state=above).to_dense(), [0, 1], rtol=0,
                       atol=1e-12)
    with pytest.raises(ValueError, match="no branch can pass the postselection"):
        sim.run_branches(operations, initial_state=below)
    [branch] = sim.run_branches(operations, initial_state=above)
    assert branch.probability == 1.0
    assert np.allclose(branch.state.to_dense(), [0, 1], rtol=0, atol=1e-12)


def test_forced_schedule_is_honoured_and_recorded():
    """The schedule's outcomes are applied and recorded, and an exhausted schedule raises. Two
    measurements that compare equal (hand-built, without a ``meas_uid``) would share one record
    entry, so the second raises."""
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
    assert np.array_equal(state.to_dense(), SparseState.basis(3, 0).to_dense())  # all reset
    assert policy.position == 3
    with pytest.raises(ValueError, match="schedule of 3 outcomes is exhausted"):
        sim.run(operations + operations[:2])
    with pytest.raises(ValueError, match="0 or 1"):
        ForcedPolicy([2])
    twins = [qp.H(0), MidMeasure(0, reset=True), qp.H(0), MidMeasure(0, reset=True)]
    with pytest.raises(ValueError, match="already recorded"):
        SparseSimulator(1, policy=ForcedPolicy([0, 0])).run(twins)


def test_a_forced_outcome_of_probability_zero_raises():
    """A forced outcome below MEASUREMENT_TOLERANCE (1e-12) raises, and so does a NaN probability
    (a state of norm 0); in ``run_branches`` a branch outcome below the tolerance is dropped. Just
    above the tolerance, both are kept."""
    operations = _tape(lambda: qp.measure(0)).operations
    with pytest.raises(ValueError, match="probability 0"):
        SparseSimulator(1, policy=ForcedPolicy([1]), strict=False).run(operations)
    with pytest.raises(ValueError, match="probability 5e-13"):
        SparseSimulator(1, policy=ForcedPolicy([1]), strict=False).run(
            operations, initial_state=_one_qubit(0.5e-12))
    state = SparseSimulator(1, policy=ForcedPolicy([1]), strict=False).run(
        operations, initial_state=_one_qubit(2e-12))
    assert np.allclose(state.to_dense(), [0, 1], rtol=0, atol=1e-12)
    with np.errstate(invalid="ignore"), pytest.raises(ValueError, match="probability nan"):
        SparseSimulator(1, policy=ForcedPolicy([0]), strict=False).run(
            operations, initial_state=_zero_norm_state())
    branching = SparseSimulator(1, policy=BranchPolicy([0]), strict=False)
    assert len(branching.run_branches(operations, initial_state=_one_qubit(0.5e-12))) == 1
    kept = branching.run_branches(operations, initial_state=_one_qubit(2e-12))
    assert sorted(b.probability for b in kept) == pytest.approx([2e-12, 1 - 2e-12], rel=0,
                                                                abs=1e-15)


def test_strict_mode_asserts_uniform_erasure_outcomes():
    """Strict mode accepts |P(1) - 1/2| up to MEASUREMENT_TOLERANCE (1e-12) and rejects a larger
    bias or a NaN probability (a state of norm 0). It checks the measurements a BranchPolicy
    delegates to ``otherwise`` and exempts branch and postselected measurements."""
    unbalanced = _tape(lambda: (qp.H(0), qp.S(0), qp.CNOT([0, 1]), qp.measure(1))).operations
    assert SparseSimulator(2, policy=ForcedPolicy([0])).run(unbalanced) is not None
    biased = _tape(lambda: qp.measure(0)).operations
    with pytest.raises(AssertionError, match="P\\(1\\) = 0.0"):
        SparseSimulator(1, policy=ForcedPolicy([0])).run(biased)
    for bias in (0.5e-12, -0.5e-12):
        assert SparseSimulator(1, policy=ForcedPolicy([0])).run(
            biased, initial_state=_one_qubit(0.5 + bias)) is not None
    for bias in (2e-12, -2e-12):
        with pytest.raises(AssertionError, match="strict mode"):
            SparseSimulator(1, policy=ForcedPolicy([0])).run(
                biased, initial_state=_one_qubit(0.5 + bias))
    with np.errstate(invalid="ignore"), pytest.raises(AssertionError, match="P\\(1\\) = nan"):
        SparseSimulator(1, policy=ForcedPolicy([0])).run(biased,
                                                         initial_state=_zero_norm_state())
    # Branch wires and postselected measurements are exempt; delegated measurements are not.
    assert len(SparseSimulator(1, policy=BranchPolicy([0])).run_branches(biased)) == 1
    postselected = _tape(lambda: qp.measure(0, postselect=0)).operations
    assert SparseSimulator(1, policy=ForcedPolicy([])).run(postselected) is not None
    delegated = _tape(lambda: (qp.H(0), qp.measure(0), qp.measure(1))).operations
    with pytest.raises(AssertionError, match="P\\(1\\) = 0.0"):
        SparseSimulator(2, policy=BranchPolicy([0], otherwise=ForcedPolicy([0]))).run_branches(
            delegated)


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
    assert np.sum(np.abs(state.amps) ** 2) == pytest.approx(1.0, rel=0, abs=1e-14)


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
    """Measurements off the branch wires go to ``otherwise``. Each branch path consumes its own
    copy of the policy: both children of a split continue from the parent's position (here 0,
    since the split comes first), and the caller's policy is not advanced. Afterwards the
    simulator's state, record and policy are as before, not aliases of a returned branch."""
    operations = _tape(lambda: (qp.H(0), qp.H(1), qp.measure(0), qp.measure(1))).operations
    policy = BranchPolicy([0], otherwise=ForcedPolicy([1]))
    sim = SparseSimulator(2, policy=policy)
    branches = sim.run_branches(operations)
    assert sorted(b.record[operations[2]] for b in branches) == [0, 1]
    assert all(b.record[operations[3]] == 1 for b in branches)
    assert [b.probability for b in branches] == pytest.approx([0.5, 0.5], rel=0, abs=1e-12)
    assert policy.otherwise.position == 0  # the branches consumed copies
    assert sim.policy is policy and sim.record == {}
    assert all(sim.state is not b.state and sim.record is not b.record for b in branches)
    with pytest.raises(ValueError, match="run_branches"):
        SparseSimulator(2, policy=policy).run(operations)
    with pytest.raises(ValueError, match="no policy"):
        SparseSimulator(2, policy=BranchPolicy([0])).run_branches(operations)
    with pytest.raises(ValueError, match="no policy"):
        SparseSimulator(2).run(operations)


def test_nested_conditionals_dispatch_recursively():
    """Nested conditionals dispatch recursively, and ``qp.cond(a | b, Z)`` applies its phase.
    ``nonlinear_conditionals`` flags the nested conditionals (their bases are not diagonal) and
    the OR (its predicate is not GF(2)-linear): both break the premise of spec 12.5."""
    def build():
        qp.X(0)
        qp.X(1)
        a, b = qp.measure(0), qp.measure(1)
        Conditional(a, Conditional(b, qp.X(2)))
        Conditional(a, Conditional(b ^ a, qp.X(3)))
        qp.cond(a | b, qp.Z)(2)

    operations = _tape(build).operations
    state = SparseSimulator(4, policy=ForcedPolicy([1, 1]), strict=False).run(operations)
    assert np.flatnonzero(state.to_dense()).tolist() == [0b1110]
    assert state.to_dense()[0b1110] == -1
    assert nonlinear_conditionals(operations) == operations[4:]


@pytest.mark.parametrize("predicate", [
    lambda *bits: (sum(bits) % 2) ^ int(all(bits)),
    lambda *bits: sum(bits) % 2,
], ids=["xor-and-all", "xor"])
def test_untagged_predicates_beyond_the_exhaustive_limit_are_flagged(predicate):
    """A 20-outcome predicate that is not a ``ParityPredicate`` cannot be checked exhaustively, so
    ``nonlinear_conditionals`` flags it (fail closed), even the linear XOR. XOR(all) XOR AND(all)
    differs from XOR only on the all-ones vector, which no zero or unit schedule reaches."""
    def build():
        outcomes = [qp.measure(w) for w in range(20)]
        value = qp.ops.MeasurementValue([m.measurements[0] for m in outcomes], predicate)
        Conditional(value, qp.Z(20))

    operations = _tape(build).operations
    assert isinstance(operations[-1], Conditional)
    assert nonlinear_conditionals(operations) == operations[-1:]


@pytest.mark.parametrize("inverted", [False, True], ids=["linear", "affine"])
def test_tape_parities_are_decided_exactly(inverted):
    """``mcm_parities`` over 20 outcomes tags every tape parity as a ``ParityPredicate``, which
    ``nonlinear_conditionals`` decides exactly at any size: parities of plain outcomes are linear,
    and a parity that includes an inverted outcome ``~m`` is affine (f(0) = 1) and is flagged."""
    masks = [2**20 - 1, 0b1011 << 9, 1 << 19]

    def build():
        outcomes = [qp.measure(w) for w in range(20)]
        if inverted:
            outcomes[3] = ~outcomes[3]
        for parity in mcm_parities(outcomes, masks):
            qp.cond(parity, qp.Z)(20)

    conditionals = _tape(build).operations[20:]
    assert len(conditionals) == len(masks)
    assert all(isinstance(op.meas_val.processing_fn, ParityPredicate) for op in conditionals)
    expected = conditionals[:1] if inverted else []
    assert nonlinear_conditionals(conditionals) == expected
