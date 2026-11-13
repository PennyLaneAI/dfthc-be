"""MajoranaGate (spec 9.16): the action on every branch, the flavour convention, counts."""

import functools

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.fermion import annihilation, majorana
from dfthc.ops.majorana import MajoranaGate, _majorana_gate
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.sparse_sim import SparseSimulator
from dfthc.testing.tape import decompose_for_simulation

# Wire order of the 7 one-wire registers, as in the operator: wire 0 is the most significant bit
# of a basis index |succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target>.
NAMES = ("succ_x_o", "succ_b", "G0", "G1", "b_neq_B", "flavour", "target")


def reference_matrix():
    """Spec 9.16 step by step, on every basis input: a = succ_x_o AND G0,
    s = succ_x_o AND NOT G0 AND succ_b; (1) -Z on target if s AND b_neq_B; (2) X on target if a;
    (3) Z on target if a AND flavour; (4) Z on flavour if a AND G1."""
    matrix = np.zeros((2**7, 2**7))
    for index in range(2**7):
        succ_x_o, succ_b, g0, g1, b_neq_b, flavour, target = (
            (index >> (6 - i)) & 1 for i in range(7))
        a, s = succ_x_o & g0, succ_x_o & (1 - g0) & succ_b
        sign = -1 if s and b_neq_b and target == 0 else 1
        target ^= a
        sign *= -1 if a and flavour and target else 1
        sign *= -1 if a and g1 and flavour else 1
        matrix[(index & ~1) | target, index] = sign
    return matrix


@functools.cache
def compiled_matrix(adjoint: bool):
    """The compiled gate's 2**7 x 2**7 matrix from one lightning run: each of the seven wires is
    entangled with a reference wire (H, CNOT), so the amplitude of |y>|x>|00>_work is
    U[y, x] / sqrt(2**7); the work wires must return to |00>."""
    system, reference, work = list(range(7)), list(range(7, 14)), [14, 15]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=16))
    def circuit():
        for s, r in zip(system, reference, strict=True):
            qp.H(r)
            qp.CNOT([r, s])
        program(adjoint, system + work)()
        return qp.state()

    state = np.asarray(circuit()).reshape(2**7, 2**7, 4)
    assert np.allclose(state[:, :, 1:], 0, rtol=0, atol=1e-10)
    return state[:, :, 0] * np.sqrt(2**7)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["gate", "adjoint"])
def test_action_on_every_basis_input(adjoint):
    """T3: the compiled matrix (and its auto-derived adjoint) equals spec 9.16's four steps on
    all 128 basis inputs of the seven registers, every branch included, with clean work wires."""
    expected = reference_matrix()
    assert np.allclose(compiled_matrix(adjoint), expected.T if adjoint else expected, rtol=0,
                       atol=1e-10)


@pytest.mark.parametrize("adjoint", [False, True], ids=["gate", "adjoint"])
def test_tape_path_keeps_the_and_contracts(adjoint):
    """The tape path (spec 7.4 rule 9), as the sparse simulator runs it, on all 128 inputs at
    once (as in T3): spec 9.16's four steps with clean work wires. The simulator checks on every
    term that each TemporaryAND target is 0 and each Adjoint(TemporaryAND) target equals its AND;
    lightning runs both as a Toffoli, so T3 and T4 cannot tell a compute from a measured
    uncompute."""
    system, reference, work = list(range(7)), list(range(7, 14)), [14, 15]

    def qfunc():
        for s, r in zip(system, reference, strict=True):
            qp.H(r)
            qp.CNOT([r, s])
        program(adjoint, system + work)()

    state = SparseSimulator(16).run(decompose_for_simulation(qfunc).operations)
    amplitudes = state.to_dense().reshape(2**7, 2**7, 4)
    expected = reference_matrix()
    assert np.allclose(amplitudes[:, :, 1:], 0, rtol=0, atol=1e-10)
    assert np.allclose(amplitudes[:, :, 0] * np.sqrt(2**7), expected.T if adjoint else expected,
                       rtol=0, atol=1e-10)


PLUS = np.array([1, 1]) / np.sqrt(2)


def generator_on_target(succ_x_o, succ_b, g0, g1, b_neq_b):
    """The 2 x 2 action spec 6.2 assigns to psi_dn[0] on one branch, from dfthc.data.fermion
    (N = 1, spin down): i gamma_0 gamma_1 = -Z on a valid SF branch with b != B, a = (X + iY)/2 on
    a valid D1 branch, a^dag on a valid Q1 branch, the identity otherwise."""
    gamma = [majorana(0, f, 1, 1)[:2, :2] for f in (0, 1)]
    lower = annihilation(0, 1, 1)[:2, :2]
    if succ_x_o and g0:
        return lower if g1 == 0 else lower.conj().T
    if succ_x_o and succ_b and b_neq_b:
        return 1j * gamma[0] @ gamma[1]
    return np.eye(2)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["gate", "adjoint"])
def test_flavour_projected_action_is_the_generator(adjoint):
    """The flavour convention (spec 6.2, 9.16): with flavour = sigma_0 projected onto |+> before
    and after, the compiled gate acts on target as -Z (SF, b != B), a (D1), a^dag (Q1) or the
    identity, on each of the 32 control branches; the adjoint gives the adjoint actions."""
    matrix = compiled_matrix(adjoint)
    project = np.kron(PLUS, np.eye(2))  # <+|_flavour (x) I_target, a 2 x 4 matrix
    for control in range(32):
        bits = [(control >> (4 - i)) & 1 for i in range(5)]
        block = matrix[4 * control:4 * control + 4, 4 * control:4 * control + 4]
        expected = generator_on_target(*bits)
        assert np.allclose(project @ block @ project.T,
                           expected.conj().T if adjoint else expected, rtol=0, atol=1e-10), bits


def program(adjoint, wires=tuple(range(9))):
    """Queues the gate (or its adjoint) on the seven one-wire registers and two work wires."""

    def queue():
        op = MajoranaGate(*([w] for w in wires[:7]), work=list(wires[7:]))
        if adjoint:
            qp.adjoint(op)

    return queue


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["gate", "adjoint"])
def test_toffoli_count(adjoint):
    """T4: T = 5 with three Toffoli and two TemporaryAND (A = 2), M = 0; the same for the
    auto-derived adjoint."""
    resources = compiled_resources(program(adjoint), num_wires=9)
    assert (resources.toffoli, resources.measurements) == (costs.majorana_gate().toffoli, 0)
    assert (resources.counts["Toffoli"], resources.counts["TemporaryAND"]) == (3, 2)
    assert resources.measured_and_uncomputes == 2


def test_femoco_count_on_the_closed_form():
    """FeMoco-54 (spec 9.24): 5 Toffolis for the gate and for its adjoint."""
    assert costs.majorana_gate() == costs.Cost(5)


@pytest.mark.compiled
def test_pinned_rule_is_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names the pinned rule; T5 at level="device" for the gate and
    its auto-derived adjoint (L-20)."""

    def both():
        program(False)()
        program(True)()

    qp.analyze(verbose_program(both, num_wires=9), level="user")()
    captured = capfd.readouterr()
    assert "_majorana_gate" in chosen_rules(captured.out + captured.err)
    compiled_resources(both, num_wires=9, level="device")


@pytest.mark.parametrize("wires", [list(range(9)), [8, 3, 5, 0, 7, 1, 6, 2, 4]],
                         ids=["ordered", "permuted"])
def test_rule_is_valid(wires):
    """T2 (tape mode). The gate has one size; the second case relabels the wires. The rule is
    pinned and the only rule registered under the class name (spec 7.4 rules 7 and 8)."""
    assert FIXED_DECOMPS["MajoranaGate"] is _majorana_gate
    assert list(qp.list_decomps(MajoranaGate)) == [_majorana_gate]
    op = MajoranaGate(*([w] for w in wires[:7]), work=wires[7:])
    qp.testing.assert_valid_decomp_rule(op, _majorana_gate, check_matrix=False)


def test_required_wires():
    assert MajoranaGate.required_wires() == {name: 1 for name in NAMES} | {"work": 2}


@pytest.mark.parametrize(("change", "match"), [
    ({"work": [7]}, "'work' has 1 wires; expected 2"),
    ({"target": [6, 9]}, "'target' has 2 wires; expected 1"),
    ({"G0": []}, "'G0' has 0 wires; expected 1"),
])
def test_constructor_validates(change, match):
    args = {name: [i] for i, name in enumerate(NAMES)} | {"work": [7, 8]}
    with pytest.raises(ValueError, match=match):
        MajoranaGate(**(args | change))
