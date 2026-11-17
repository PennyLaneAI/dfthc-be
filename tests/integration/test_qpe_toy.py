"""The toy walk of spec 11.4 and iterative phase estimation on it (spec 11.1): ToyWalk, its
controlled form qp.ctrl(ToyWalk) and ToyControlledWalkPower (T2 to T6, the action against the dense
reference with the control in |0>, |1> and |+>, for both control values), the exact t = 3
distribution on the sparse simulator, the lightning.qubit smoke test, the compiled feed-forward on
one postselected path, earlier_bits of shape (t,) in every round in both modes, the toy QPE
program's counts and its inventory of captured targets."""

import itertools

import numpy as np
import pennylane as qp
import pytest

from dfthc import qpe
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.qpe import phase_from_bits
from dfthc.testing.resources import captured_targets, chosen_rules, inventory, verbose_program
from dfthc.testing.sparse_sim import BranchPolicy, SparseSimulator
from dfthc.testing.tape import decompose_for_simulation
from dfthc.testing.toy_walk import (
    ToyControlledWalkPower,
    ToyWalk,
    _controlled_toy_walk,
    _toy_controlled_walk_power,
    _toy_walk,
    controlled_toy_walk_matrix,
    toy_qpe,
    toy_qpe_wires,
    toy_walk_cost,
    toy_walk_matrix,
    toy_walk_power_cost,
)

T = 3
NUM_WIRES = 3 * T + 2
COS2, SIN2 = np.cos(np.pi / 8) ** 2, np.sin(np.pi / 8) ** 2
EXACT = {1 / 8: COS2 / 2, 3 / 8: SIN2 / 2, 5 / 8: SIN2 / 2, 7 / 8: COS2 / 2}
"""Spec 11.4: phi in {1/8, 7/8} with total probability cos^2(pi/8), {3/8, 5/8} with sin^2(pi/8);
each eigenvalue of H splits equally between the walk phases +-theta."""
PLUS = np.array([1, 1]) / np.sqrt(2)


# --------------------------------------------------------------------------- the toy walks


def test_reference_walk_has_the_spec_eigenphases():
    """Spec 11.4: W is unitary, its block on the ancilla's |0> is H = (Z + X)/2, and its
    eigenphases are -3 pi/4, -pi/4, pi/4, 3 pi/4."""
    walk = toy_walk_matrix()
    assert np.allclose(walk.conj().T @ walk, np.eye(4), rtol=0, atol=1e-12)
    block = walk[np.ix_([0, 2], [0, 2])]
    assert np.allclose(block, np.array([[1, 1], [1, -1]]) / 2, rtol=0, atol=1e-12)
    phases = np.sort(np.angle(np.linalg.eigvals(walk)))
    assert np.allclose(phases, np.pi * np.array([-3, -1, 1, 3]) / 4, rtol=0, atol=1e-12)


def choi_matrix(build, n: int) -> np.ndarray:
    """The matrix of the n-qubit program ``build(data_wires)`` from one compiled lightning run:
    Bell pairs between reference wires 0..n-1 and data wires n..2n-1, then the program on the
    data wires; the output amplitude [reference x, data y] is U[y, x] / sqrt(2**n)."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=2 * n))
    def circuit():
        for wire in range(n):
            qp.H(wire)
            qp.CNOT([wire, n + wire])
        build(list(range(n, 2 * n)))
        return qp.state()

    return np.sqrt(2**n) * np.asarray(circuit()).reshape(2**n, 2**n).T


@pytest.mark.compiled
def test_toy_walk_acts_as_the_reference():
    """T3: the compiled ToyWalk's whole matrix, global phase included, is spec 11.4's W."""
    matrix = choi_matrix(lambda w: ToyWalk(system=w[:1], ancilla=w[1:]), 2)
    assert np.allclose(matrix, toy_walk_matrix(), rtol=0, atol=1e-12)


@pytest.mark.compiled
@pytest.mark.parametrize(("power", "control_value"), [(1, 1), (1, 0), (3, 1)])
def test_controlled_toy_walks_in_three_control_states(power, control_value):
    """T3 of the controlled forms: the compiled matrix of qp.ctrl(ToyWalk) (power 1) and of
    ToyControlledWalkPower(3) is |0><0| (x) I + |1><1| (x) W^p exactly, so control |0> keeps the
    input, control |1> applies W^p and control |+> gives their superposition with the right
    relative phase (spec 3.3: a wrong global phase of a controlled reflection shows only there);
    with control value 0 the two control states swap roles."""
    def build(w):
        if power == 1:
            qp.ctrl(ToyWalk(system=w[1:2], ancilla=w[2:]), control=w[:1],
                    control_values=[control_value])
        else:
            ToyControlledWalkPower(power, control=w[:1], system=w[1:2], ancilla=w[2:])

    matrix = choi_matrix(build, 3)
    walked = np.linalg.matrix_power(toy_walk_matrix(), power)
    if power == 1:
        assert np.allclose(controlled_toy_walk_matrix(), np.kron(np.diag([1, 0]), np.eye(4))
                           + np.kron(np.diag([0, 1]), walked), rtol=0, atol=1e-12)
    rng = np.random.default_rng(power)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    zero, one = (psi, walked @ psi) if control_value else (walked @ psi, psi)
    expected = {"0": np.kron([1, 0], zero), "1": np.kron([0, 1], one),
                "+": (np.kron([1, 0], zero) + np.kron([0, 1], one)) / np.sqrt(2)}
    for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", PLUS)):
        out = matrix @ np.kron(control, psi)
        assert np.allclose(out, expected[label], rtol=0, atol=1e-12), label


def toy_program(name, power=1, control_value=1):
    if name == "ToyWalk":
        return lambda: ToyWalk(system=[0], ancilla=[1])
    if name == "C(ToyWalk)":
        return lambda: qp.ctrl(ToyWalk(system=[1], ancilla=[2]), control=[0],
                               control_values=[control_value])
    return lambda: ToyControlledWalkPower(power, control=[0], system=[1], ancilla=[2])


@pytest.mark.compiled
@pytest.mark.parametrize(("name", "power"), [("ToyWalk", None), ("C(ToyWalk)", None),
                                             ("ToyControlledWalkPower", 1),
                                             ("ToyControlledWalkPower", 4)])
def test_counts(name, power):
    """T4: ToyWalk T = M = A = 0; qp.ctrl(ToyWalk) two plain Toffolis (T = 2, A = 0, M = 0);
    the power p of it T = 2 p, all against toy_walk_cost and toy_walk_power_cost."""
    resources = compiled_resources(toy_program(name, power), 2 if name == "ToyWalk" else 3)
    expected = (toy_walk_cost(False) if name == "ToyWalk" else toy_walk_cost(True)
                if name == "C(ToyWalk)" else toy_walk_power_cost(power))
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.counts.get("Toffoli", 0) == expected.toffoli
    assert resources.measured_and_uncomputes == 0


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing of the toy QPE program and a ToyWalk names the pinned rules of
    the three toy classes, of QPERound and of the corrections' PhaseGradientRotation; T5: at
    level="device" no dfthc operator survives (compiled_resources asserts it)."""
    wires = toy_qpe_wires(T)

    def program():
        toy_qpe(T)
        ToyWalk(system=wires["system"], ancilla=wires["ancilla"])

    qp.analyze(verbose_program(program, NUM_WIRES), level="user")()
    captured = capfd.readouterr()
    assert {"_toy_walk", "_controlled_toy_walk", "_toy_controlled_walk_power", "_qpe_round",
            "_phase_gradient_rotation"} <= set(chosen_rules(captured.out + captured.err))
    compiled_resources(program, NUM_WIRES, level="device")


def test_rules_are_valid():
    """T2 (tape mode) of the three rules, the power at two values, the controlled walk's with
    control value 0 (its conditional X gates on the control then run, as declared); each pinned
    and the only rule under its key (spec 7.4 rules 7 and 8)."""
    for name, rule, powers in (("ToyWalk", _toy_walk, [None]),
                               ("C(ToyWalk)", _controlled_toy_walk, [None]),
                               ("ToyControlledWalkPower", _toy_controlled_walk_power, [1, 3])):
        assert FIXED_DECOMPS[name] is rule
        assert list(qp.list_decomps(name)) == [rule]
        for power in powers:
            with qp.queuing.AnnotatedQueue():
                op = toy_program(name, power, control_value=0)()
            qp.testing.assert_valid_decomp_rule(op, rule, check_matrix=False)


@pytest.mark.parametrize(("build", "match"), [
    (lambda: ToyWalk(system=[0, 1], ancilla=[2]), "'system' has 2 wires; expected 1"),
    (lambda: ToyControlledWalkPower(0, control=[0], system=[1], ancilla=[2]),
     r"power must lie in \[1, None\], got 0"),
], ids=["system", "power"])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_controlled_rule_takes_one_control_wire():
    """The C(ToyWalk) rule raises for two control wires."""
    with qp.queuing.AnnotatedQueue():
        op = qp.ctrl(ToyWalk(system=[2], ancilla=[3]), control=[0, 1])
    with pytest.raises(ValueError, match=r"C\(ToyWalk\) takes one control wire; got 2"):
        _controlled_toy_walk(**op.arguments)


# --------------------------------------------------------------------------- phase estimation


def control_measurements(ops):
    wires = toy_qpe_wires(T)
    measures = [op for op in ops if op.name == "MidMeasure"]
    assert [int(op.wires[0]) for op in measures] == wires["control"] * T
    return measures


def test_exact_distribution_at_t_3():
    """Spec 11.4 on the tape path (strict sparse simulator, BranchPolicy on the control; the toy
    has no erase measurements, so the policy has no `otherwise` and any other measurement would
    raise): phi in {1/8, 7/8} with cos^2(pi/8) and {3/8, 5/8} with sin^2(pi/8), half each, and
    nothing else. Every branch leaves the control and the spare wires in |0>, qpe_grad in
    the gradient state and the toy wires in an eigenvector of W with eigenvalue e^{2 pi i phi}:
    the bits read the eigenphase of W itself (spec 11.1's convention), not of W^dagger."""
    ops = decompose_for_simulation(toy_qpe, T).operations
    measures = control_measurements(ops)
    simulator = SparseSimulator(NUM_WIRES, policy=BranchPolicy(toy_qpe_wires(T)["control"]),
                                strict=True)
    gradient = SparseSimulator(T).run([qp.PhaseGradientStatePrep(range(T))]).to_dense()
    walk, found = toy_walk_matrix(), {}
    for branch in simulator.run_branches(ops):
        phi = phase_from_bits([branch.record[op] for op in measures])
        found[phi] = found.get(phi, 0.0) + branch.probability
        state = branch.state.to_dense().reshape(2, 2**T, 2 ** (2 * T - 1), 4)
        toy = gradient.conj() @ state[0, :, 0, :]
        assert np.allclose(state[0, :, 0, :], np.outer(gradient, toy), rtol=0, atol=1e-12)
        assert np.isclose(np.linalg.norm(toy), 1, rtol=0, atol=1e-12)
        assert np.allclose(walk @ toy, np.exp(2j * np.pi * phi) * toy, rtol=0, atol=1e-12)
    assert sorted(found) == sorted(EXACT)
    for phi, probability in EXACT.items():
        assert np.isclose(found[phi], probability, rtol=0, atol=1e-12), phi
    assert np.isclose(found[1 / 8] + found[7 / 8], COS2, rtol=0, atol=1e-12)
    assert np.isclose(found[3 / 8] + found[5 / 8], SIN2, rtol=0, atol=1e-12)


SHOTS = 2000


@pytest.mark.compiled
def test_lightning_smoke_with_2000_shots():
    """Spec 11.4: the compiled program on lightning.qubit, 2,000 one-shot samples (no
    postselection, no measurement inside qp.cond; L-15): each of the four phases within five
    standard deviations of its exact probability, the two pairs too, and no other phase."""

    @qp.qjit(capture=True, seed=2026)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.set_shots(SHOTS)
    @qp.qnode(qp.device("lightning.qubit", wires=NUM_WIRES), mcm_method="one-shot")
    def circuit():
        return [qp.sample(op=bit) for bit in toy_qpe(T)]

    samples = np.asarray(circuit()).reshape(T, SHOTS)
    phases = [phase_from_bits(samples[:, shot]) for shot in range(SHOTS)]
    values, counts = np.unique(phases, return_counts=True)
    frequency = dict(zip(values.tolist(), (counts / SHOTS).tolist(), strict=True))
    assert set(frequency) <= set(EXACT)
    pairs = {(1 / 8, 7 / 8): COS2, (3 / 8, 5 / 8): SIN2}
    checks = {(phi,): p for phi, p in EXACT.items()} | pairs
    for group, p in checks.items():
        observed = sum(frequency.get(phi, 0.0) for phi in group)
        assert abs(observed - p) <= 5 * np.sqrt(p * (1 - p) / SHOTS), (group, observed, p)


@pytest.mark.compiled
@pytest.mark.parametrize("bits", [(1, 1, 0), (1, 0, 0)], ids=["phi=3/8", "phi=1/8"])
def test_compiled_feed_forward_leaves_an_eigenvector(bits, monkeypatch):
    """The compiled program's feed-forward, which the exact test runs only on the tape path: under
    capture the driver sets earlier_bits with .at[j].set and each correction is a qp.cond on a
    traced bit. With the control's outcomes postselected to ``bits`` = m_2, m_1, m_0 (analytic
    lightning.qubit, spec 12.2), the control and the spare wires end in |0>, qpe_grad in the
    gradient state and the toy wires in an eigenvector of W with eigenvalue e^{2 pi i phi}.
    phi = 3/8 fires all three corrections; phi = 1/8 fires m_2's alone, at widths 2 and 3. A bit
    stored at the wrong entry or never stored, or a correction of the wrong sign or width, leaves
    no such eigenvector; without this test only the sampled smoke test above sees the first."""
    measure, outcomes = qp.measure, itertools.cycle(bits)
    monkeypatch.setattr(qp, "measure", lambda wire, reset=False: measure(
        wire, reset=reset, postselect=next(outcomes)))

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=NUM_WIRES))
    def circuit():
        toy_qpe(T)
        return qp.state()

    state = np.asarray(circuit())
    state = (state / np.linalg.norm(state)).reshape(2, 2**T, 2 ** (2 * T - 1), 4)
    gradient = SparseSimulator(T).run([qp.PhaseGradientStatePrep(range(T))]).to_dense()
    toy = gradient.conj() @ state[0, :, 0, :]
    assert np.allclose(state[0, :, 0, :], np.outer(gradient, toy), rtol=0, atol=1e-12)
    assert np.isclose(np.linalg.norm(toy), 1, rtol=0, atol=1e-12)
    phi = phase_from_bits(bits)
    assert np.allclose(toy_walk_matrix() @ toy, np.exp(2j * np.pi * phi) * toy, rtol=0,
                       atol=1e-12)


@pytest.mark.parametrize("mode", ["tape", pytest.param("capture", marks=pytest.mark.compiled)])
def test_earlier_bits_have_shape_t_in_every_round(mode, monkeypatch):
    """Spec 11.1, master plan Task 4.1: every QPERound of the driver gets earlier_bits of shape
    (t,), round t - 1 included (no bit set yet). In tape mode entry l of round j is False for
    l <= j and, for l > j, the very MeasurementValue that round l's measurement returned; under
    capture it is a boolean array of the same shape, the measured bits set with .at[j].set."""
    rounds, original = [], qpe.QPERound

    def recording(j, num_bits, earlier_bits, *rest):
        rounds.append((j, earlier_bits))
        return original(j, num_bits, earlier_bits, *rest)

    monkeypatch.setattr(qpe, "QPERound", recording)
    if mode == "tape":
        with qp.capture.pause(), qp.queuing.AnnotatedQueue():
            measured = toy_qpe(T)
    else:
        compiled_resources(lambda: toy_qpe(T), NUM_WIRES)
    assert [j for j, _ in rounds] == [2, 1, 0]
    for j, bits in rounds:
        assert tuple(bits.shape) == (T,)
        if mode == "tape":
            assert all(bits[later] is False for later in range(j + 1))
            assert all(bits[later] is measured[T - 1 - later] for later in range(j + 1, T))
        else:
            assert bits.dtype == bool


@pytest.mark.compiled
@pytest.mark.parametrize(("t", "toffoli", "measurements", "uncomputes"),
                         [(1, 2, 1, 0), (2, 3 * 2 + 1, 2, 1), (3, 7 * 2 + 4, 3, 4)])
def test_toy_qpe_counts(t, toffoli, measurements, uncomputes):
    """T4 of the QPE program on the toy walk, worst case (spec 9.24 with T_C(W) = 2, M_W = 0):
    T = (2^t - 1) 2 + (t^3 - t)/6, M = t, and A = (t^3 - t)/6, the corrections' measured AND
    uncomputes, every unhinted correction counted at its maximum branch."""
    assert toffoli == toy_walk_power_cost(2**t - 1).toffoli + (t**3 - t) // 6
    resources = compiled_resources(lambda: toy_qpe(t), 3 * t + 2)
    assert (resources.toffoli, resources.measurements,
            resources.measured_and_uncomputes) == (toffoli, measurements, uncomputes)


@pytest.mark.compiled
def test_inventory_of_the_toy_qpe_program():
    """The inventory (spec 12.1 T6) of the toy QPE program at t = 3: the frontend converts the
    rounds, the walk powers and the controlled toy walk, and no modified dfthc or template target
    but C(ToyWalk), whose rule is pinned (the corrections are forward rotations; the generic
    controlled candidate of C(ToyWalk) requests only controlled PennyLane gates)."""
    names = captured_targets(lambda: toy_qpe(T), NUM_WIRES)
    assert {"QPERound", "ToyControlledWalkPower", "C(ToyWalk)",
            "PhaseGradientRotation"} <= set(names)
    assert inventory(names) == {"C(ToyWalk)"}
