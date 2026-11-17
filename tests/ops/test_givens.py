"""GivensRotation, GivensLadder and GivensLadderDagger (spec 9.14, 9.15): exact actions, the ladder
order, counts."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.angles import ladder_matrix, vector_from_angles
from dfthc.data.fermion import majorana
from dfthc.ops.givens import (
    GivensLadder,
    GivensLadderDagger,
    GivensRotation,
    _givens_ladder,
    _givens_ladder_dagger,
    _givens_rotation,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.testing.resources import chosen_rules, verbose_program


def gradient(b_rot):
    """PhaseGradientStatePrep's state on b_rot wires: sum_y exp(-2 pi i y / 2^b) |y> / sqrt(2^b)."""
    return np.exp(-2j * np.pi * np.arange(2**b_rot) / 2**b_rot) / np.sqrt(2**b_rot)


def layout(num_modes, b_rot):
    """Consecutive wires for the ladder's registers: system, angles, phase_grad, work."""
    wires, start = {}, 0
    for name, size in GivensLadder.required_wires(b_rot, num_modes).items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def load_bits(value, wires):
    """X on every wire whose bit of the runtime integer ``value`` is 1 (big-endian)."""
    for position, wire in enumerate(wires):
        qp.cond(((value >> (len(wires) - 1 - position)) & 1) == 1, qp.X)(wire)


def angle_value(x, b_rot):
    """The angle registers' big-endian integer: register k (rung k, b_rot - 1 wires) holds
    x[k]."""
    value = 0
    for x_k in x:
        value = (value << (b_rot - 1)) | int(x_k)
    return value


def compiled_matrix(apply, num_modes, b_rot):
    """``x -> the matrix on the modes`` from one compiled lightning program. Each column runs one
    basis input with the angle registers holding ``x``, then projects the state onto
    ``|x>|gradient>|0>`` of the angle, phase-gradient and work registers; the columns are unit
    vectors exactly when nothing leaks into those registers (asserted)."""
    wires, num_wires = layout(num_modes, b_rot)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit(value, basis):
        load_bits(basis, wires["system"])
        load_bits(value, wires["angles"])
        qp.PhaseGradientStatePrep(wires["phase_grad"])
        apply(**wires)
        return qp.state()

    def matrix(x):
        value = angle_value(x, b_rot)
        rest = np.kron(np.kron(np.eye(2 ** len(wires["angles"]))[value], gradient(b_rot)),
                       np.eye(2 ** len(wires["work"]))[0])
        columns = [np.asarray(circuit(value, basis)).reshape(2**num_modes, -1) @ rest.conj()
                   for basis in range(2**num_modes)]
        assert np.allclose([np.linalg.norm(c) for c in columns], 1, rtol=0, atol=1e-10)
        return np.column_stack(columns)

    return matrix


def rung(system, angles, phase_grad, work):
    GivensRotation(system[:1], system[1:], angles, phase_grad, work)


def adjoint_rung(system, angles, phase_grad, work):
    qp.adjoint(GivensRotation(system[:1], system[1:], angles, phase_grad, work))


def rung_matrix(x, b_rot):
    """Spec 6.3: [[cos t, -sin t], [sin t, cos t]] on {|01>, |10>}, identity on |00>, |11>."""
    c, s = np.cos(4 * np.pi * x / 2**b_rot), np.sin(4 * np.pi * x / 2**b_rot)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


@pytest.mark.compiled
@pytest.mark.parametrize("b_rot", [2, 3, 4])
def test_rung_matrix_for_every_angle(b_rot):
    """The dense 4 x 4 action on the mode qubits from all four basis inputs, the angle and
    phase-gradient registers projected out, for every x < 2**(b_rot - 1) that the angle register
    holds: the sign of spec 6.3, no global phase and no leakage."""
    matrix = compiled_matrix(rung, 2, b_rot)
    for x in range(2 ** (b_rot - 1)):
        assert np.allclose(matrix([x]), rung_matrix(x, b_rot), rtol=0, atol=1e-10), x
        assert np.allclose(rung_matrix(x, b_rot), ladder_matrix(np.array([x]), b_rot, 2), rtol=0,
                           atol=1e-12)


@pytest.mark.compiled
@pytest.mark.parametrize("b_rot", [3, 4])
def test_auto_derived_rung_adjoint_rotates_back(b_rot):
    """qp.adjoint(GivensRotation(...)), the leaf adjoint GivensLadderDagger uses, is the
    transposed rung (theta -> -theta) for every x."""
    matrix = compiled_matrix(adjoint_rung, 2, b_rot)
    for x in range(2 ** (b_rot - 1)):
        assert np.allclose(matrix([x]), rung_matrix(x, b_rot).T, rtol=0, atol=1e-10), x


# (b_rot, integer angles below 2**(b_rot - 1)) per N. At b_rot = 4 the rungs turn by multiples
# of 45 degrees; N = 4 uses b_rot = 3 (90 degrees) to stay at 15 lightning wires. The angles are
# distinct, and for N >= 3 both the opposite rung order and the reversed register order give
# different unitaries (asserted below), so the test pins both. At N = 3, u has a nonzero odd
# component, so the gamma_0 identity also sees a flipped rung sign (u_j -> (-1)^j u_j).
LADDERS = {2: (4, [3]), 3: (4, [1, 3]), 4: (3, [1, 3, 2])}
B_ROT = 4


def single_register_majorana(j, flavour, num_modes):
    """gamma_{j, flavour} on one spin register: the psi_dn block of dfthc.data.fermion."""
    return majorana(j, flavour, 1, num_modes)[:2**num_modes, :2**num_modes]


def rotates_gamma_0_onto_u(matrix, x, b_rot, num_modes):
    """Spec 6.3: U^dag gamma_{0,f} U == sum_j u_j gamma_{j,f} with u = vector_from_angles."""
    u = vector_from_angles(x, b_rot)
    return all(np.allclose(
        matrix.conj().T @ single_register_majorana(0, f, num_modes) @ matrix,
        sum(u[j] * single_register_majorana(j, f, num_modes) for j in range(num_modes)),
        rtol=0, atol=1e-10) for f in (0, 1))


@pytest.mark.compiled
@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_ladder_equals_ladder_matrix_and_the_opposite_order_fails(num_modes):
    """The compiled ladder equals ladder_matrix (rung N - 2 first in time, rung 0 last) and
    rotates gamma_0 onto u = vector_from_angles(x). For N >= 3 the opposite time order (rung 0
    first) and the reversed register order (register k driving rung N - 2 - k) are different
    unitaries at these angles, so a ladder built either way fails this test; the opposite order
    also fails the gamma_0 identity there."""
    b_rot, x = LADDERS[num_modes]
    x = np.array(x)
    expected = ladder_matrix(x, b_rot, num_modes)
    matrix = compiled_matrix(
        lambda **wires: GivensLadder(b_rot, **wires), num_modes, b_rot)(x)
    assert np.allclose(matrix, expected, rtol=0, atol=1e-10)
    assert rotates_gamma_0_onto_u(matrix, x, b_rot, num_modes)
    rungs = [ladder_matrix(np.where(np.arange(num_modes - 1) == k, x, 0), b_rot, num_modes)
             for k in range(num_modes - 1)]
    opposite = np.eye(2**num_modes)
    for rung_k in rungs:
        opposite = rung_k @ opposite
    if num_modes == 2:
        assert np.allclose(opposite, expected, rtol=0, atol=1e-12)
    else:
        assert np.abs(opposite - expected).max() > 0.1
        assert not rotates_gamma_0_onto_u(opposite, x, b_rot, num_modes)
        assert np.abs(ladder_matrix(x[::-1], b_rot, num_modes) - expected).max() > 0.1


@pytest.mark.compiled
@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_dagger_is_the_inverse_of_ladder_matrix(num_modes):
    """GivensLadderDagger equals ladder_matrix(x)^dag densely."""
    b_rot, x = LADDERS[num_modes]
    x = np.array(x)
    matrix = compiled_matrix(
        lambda **wires: GivensLadderDagger(b_rot, **wires), num_modes, b_rot)(x)
    assert np.allclose(matrix, ladder_matrix(x, b_rot, num_modes).conj().T, rtol=0,
                       atol=1e-10)


def phased_prepare(wires):
    """H on each wire, S on even positions, CZ on consecutive pairs (non-uniform phases)."""
    for wire in wires:
        qp.H(wire)
    for position, wire in enumerate(wires):
        if position % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


@pytest.mark.compiled
def test_dagger_after_the_ladder_restores_the_state():
    """On lightning: the ladder, then the dagger, on a phased system state with loaded angles
    returns the system state exactly, the angles unchanged and the ancillas clean."""
    num_modes, (_, x) = 3, LADDERS[3]
    wires, num_wires = layout(num_modes, B_ROT)

    def prepared():
        phased_prepare(wires["system"])
        load_bits(angle_value(x, B_ROT), wires["angles"])
        qp.PhaseGradientStatePrep(wires["phase_grad"])

    def program(round_trip):
        @qp.qjit(capture=True)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            prepared()
            if round_trip:
                GivensLadder(B_ROT, **wires)
                GivensLadderDagger(B_ROT, **wires)
            return qp.state()

        return np.asarray(circuit())

    assert np.allclose(program(True), program(False), rtol=0, atol=1e-10)


def ladder_program(cls, num_modes, b_rot):
    wires, num_wires = layout(num_modes, b_rot)
    return (lambda: cls(b_rot, **wires)), num_wires


def rung_program(b_rot, adjoint):
    wires, num_wires = layout(2, b_rot)

    def program():
        (adjoint_rung if adjoint else rung)(**wires)

    return program, num_wires


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["rung", "adjoint"])
@pytest.mark.parametrize("p", [H2_STO3G_TEST, MID_NAPHTHALENE], ids=["b4", "b15"])
def test_rung_toffoli_count(p, adjoint):
    """T4: T = 2 (b_rot - 1) for the rung and its auto-derived adjoint; A = T, M = 0."""
    resources = compiled_resources(*rung_program(p.b_rot, adjoint))
    expected = costs.givens_rotation(p)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli


@pytest.mark.compiled
@pytest.mark.parametrize("cls", [GivensLadder, GivensLadderDagger], ids=["ladder", "dagger"])
@pytest.mark.parametrize("p", [H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE, FEMOCO54],
                         ids=["sto3g", "631g", "mid", "femoco-width"])
def test_ladder_toffoli_count(p, cls):
    """T4 at both test presets, MID_NAPHTHALENE and FeMoco-54's widths (N = 54, b_rot = 15: a
    table-free 53-rung ladder on 825 wires, about 1.5 s, which master plan section 7 allows in any
    tier): T = 2 (b_rot - 1)(N - 1) for the ladder and the dagger, 1,484 at FeMoco-54; A = T,
    M = 0."""
    resources = compiled_resources(*ladder_program(cls, p.N, p.b_rot))
    expected = costs.givens_ladder(p)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 on the closed forms: 28 per rung, 1,484 per ladder and per dagger (spec 9.24)."""
    assert costs.givens_rotation(FEMOCO54) == costs.Cost(28)
    assert costs.givens_ladder(FEMOCO54) == costs.Cost(2 * 14 * 53) == costs.Cost(1484)
    assert GivensLadder.required_wires(15, 54) == {"system": 54, "angles": 742,
                                                   "phase_grad": 15, "work": 14}


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names the three pinned rules; T5 at level="device" for the
    ladder, the dagger, a rung and its adjoint (L-20)."""
    wires, num_wires = layout(3, 3)

    def program():
        GivensLadder(3, **wires)
        GivensLadderDagger(3, **wires)
        rung(wires["system"][:2], wires["angles"][:2], wires["phase_grad"], wires["work"])
        adjoint_rung(wires["system"][:2], wires["angles"][:2], wires["phase_grad"],
                     wires["work"])

    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    assert {"_givens_rotation", "_givens_ladder", "_givens_ladder_dagger"} <= set(
        chosen_rules(captured.out + captured.err))
    compiled_resources(program, num_wires=num_wires, level="device")


@pytest.mark.parametrize(("num_modes", "b_rot"), [(2, 2), (4, 4)])
def test_rules_are_valid(num_modes, b_rot):
    """T2 for the rung, the ladder and the dagger at two sizes (tape mode); each pinned rule is
    the only rule registered under its class name (spec 7.4 rule 8)."""
    for cls, rule in ((GivensRotation, _givens_rotation), (GivensLadder, _givens_ladder),
                      (GivensLadderDagger, _givens_ladder_dagger)):
        assert list(qp.list_decomps(cls)) == [rule]
    wires, _ = layout(num_modes, b_rot)
    qp.testing.assert_valid_decomp_rule(
        GivensRotation(wires["system"][:1], wires["system"][1:2], wires["angles"][:b_rot - 1],
                       wires["phase_grad"], wires["work"]), _givens_rotation, check_matrix=False)
    qp.testing.assert_valid_decomp_rule(GivensLadder(b_rot, **wires), _givens_ladder,
                                        check_matrix=False)
    qp.testing.assert_valid_decomp_rule(GivensLadderDagger(b_rot, **wires),
                                        _givens_ladder_dagger, check_matrix=False)


def test_required_wires():
    assert GivensRotation.required_wires(4) == {"q_i": 1, "q_j": 1, "angle": 3, "phase_grad": 4,
                                                "work": 3}
    assert GivensLadder.required_wires(4, 3) == GivensLadderDagger.required_wires(4, 3) == {
        "system": 3, "angles": 6, "phase_grad": 4, "work": 3}


@pytest.mark.parametrize(("build", "match"), [
    (lambda: GivensRotation([0], [1], [], [3], []),
     r"b_rot = len\(phase_grad\) must lie in \[2,"),
    (lambda: GivensRotation([0], [1], [2, 3, 4], [5, 6, 7], [8, 9]),
     "'angle' has 3 wires; expected 2"),
    (lambda: GivensRotation([0, 9], [1], [2], [5, 6], [7]), "'q_i' has 2 wires; expected 1"),
    (lambda: GivensLadder(3, [0], [], [1, 2, 3], [4, 5]), "system needs N >= 2"),
    (lambda: GivensLadder(3, [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]),
     "'angles' has 3 wires; expected 4"),
    (lambda: GivensLadderDagger(1, [0, 1], [2], [3], []), r"b_rot must lie in \[2,"),
    (lambda: GivensLadderDagger(2, [0, 1], [2, 3, 9], [4, 5], [6]),
     "'angles' has 3 wires; expected 1"),
])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()
