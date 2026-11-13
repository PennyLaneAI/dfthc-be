"""SpinMux and SpinSwap (spec 9.12, 9.13): every basis input, matrices, counts."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.spin import (
    SpinMux,
    SpinSwap,
    _adjoint_spin_mux,
    _adjoint_spin_swap,
    _spin_mux,
    _spin_swap,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.testing.resources import chosen_rules, verbose_program


def basis_outputs(num_wires, apply):
    """``basis -> output basis index`` from one compiled lightning program; asserts that each
    output is one basis state with amplitude exactly 1 (a permutation, no phase)."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit(basis):
        for wire in range(num_wires):
            qp.cond(((basis >> (num_wires - 1 - wire)) & 1) == 1, qp.X)(wire)
        apply()
        return qp.state()

    def output(basis):
        state = np.asarray(circuit(basis))
        index = int(np.argmax(np.abs(state)))
        assert np.allclose(state, np.eye(2**num_wires)[index], rtol=0, atol=1e-12), basis
        return index

    return output


def mux_image(index):
    """Spec 9.12 on |G0, sigma_0, sigma_1, sg>: sg ^= sigma_1 if G0 else sigma_0."""
    g0, s0, s1, sg = ((index >> (3 - i)) & 1 for i in range(4))
    return (g0 << 3) | (s0 << 2) | (s1 << 1) | (sg ^ (s1 if g0 else s0))


def swap_image(index, num_modes):
    """Spec 9.13 on |sg, psi_up, psi_dn>: the two N-bit registers swapped if sg."""
    sg = index >> (2 * num_modes)
    up, dn = (index >> num_modes) & (2**num_modes - 1), index & (2**num_modes - 1)
    return index if not sg else (sg << (2 * num_modes)) | (dn << num_modes) | up


def permutation(images):
    """The matrix with a 1 at (images[b], b)."""
    matrix = np.zeros((len(images), len(images)))
    matrix[images, range(len(images))] = 1
    return matrix


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["mux", "adjoint"])
def test_spin_mux_on_every_basis_input(adjoint):
    """T3 on the compiled path: on each of the 16 basis inputs, sg ^= sigma_{G0} and nothing else
    changes; the pinned adjoint is the same permutation."""

    def apply():
        op = SpinMux([0], [1], [2], [3])
        if adjoint:
            qp.adjoint(op)

    output = basis_outputs(4, apply)
    assert [output(b) for b in range(16)] == [mux_image(b) for b in range(16)]


@pytest.mark.compiled
@pytest.mark.parametrize(("num_modes", "adjoint"), [(2, False), (3, False), (2, True)],
                         ids=["N2", "N3", "N2-adjoint"])
def test_spin_swap_on_every_basis_input(num_modes, adjoint):
    """T3 on the compiled path: on every basis input, psi_up and psi_dn are swapped exactly
    when sg is 1; the pinned adjoint is the same permutation."""
    up = list(range(1, 1 + num_modes))
    dn = list(range(1 + num_modes, 1 + 2 * num_modes))

    def apply():
        op = SpinSwap([0], up, dn)
        if adjoint:
            qp.adjoint(op)

    output = basis_outputs(1 + 2 * num_modes, apply)
    size = 2 ** (1 + 2 * num_modes)
    assert [output(b) for b in range(size)] == [swap_image(b, num_modes) for b in range(size)]


def test_matrices_are_the_spec_permutations():
    """compute_matrix, which T2 compares with the decompositions, is the action of spec 9.12
    and 9.13."""
    assert SpinMux.has_matrix and SpinSwap.has_matrix
    assert np.array_equal(SpinMux.compute_matrix([0], [1], [2], [3]),
                          permutation([mux_image(b) for b in range(16)]))
    for num_modes in (1, 2, 3):
        size = 2 ** (1 + 2 * num_modes)
        assert np.array_equal(
            SpinSwap.compute_matrix([0], list(range(num_modes)), list(range(num_modes))),
            permutation([swap_image(b, num_modes) for b in range(size)]))


@pytest.mark.parametrize("wires", [[0, 1, 2, 3], [3, 0, 2, 1]], ids=["ordered", "permuted"])
def test_spin_mux_rules_are_valid_with_matrices(wires):
    """T2 with check_matrix=True for the base and the pinned adjoint rule (tape mode): the
    declared resources equal the emitted gates and the decomposition's matrix equals
    compute_matrix. SpinMux has one size; the second case relabels the wires. Each pinned rule is
    the only rule registered under its key (spec 7.4 rule 8)."""
    assert list(qp.list_decomps(SpinMux)) == [_spin_mux]
    assert list(qp.list_decomps("Adjoint(SpinMux)")) == [_adjoint_spin_mux]
    op = SpinMux(*([w] for w in wires))
    qp.testing.assert_valid_decomp_rule(op, _spin_mux, check_matrix=True)
    qp.testing.assert_valid_decomp_rule(qp.adjoint(op), _adjoint_spin_mux, check_matrix=True)


@pytest.mark.parametrize("num_modes", [2, 3])
def test_spin_swap_rules_are_valid_with_matrices(num_modes):
    """T2 with check_matrix=True for the base and the pinned adjoint rule at two sizes; each
    pinned rule is the only rule registered under its key (spec 7.4 rule 8)."""
    assert list(qp.list_decomps(SpinSwap)) == [_spin_swap]
    assert list(qp.list_decomps("Adjoint(SpinSwap)")) == [_adjoint_spin_swap]
    op = SpinSwap([0], list(range(1, 1 + num_modes)),
                  list(range(1 + num_modes, 1 + 2 * num_modes)))
    qp.testing.assert_valid_decomp_rule(op, _spin_swap, check_matrix=True)
    qp.testing.assert_valid_decomp_rule(qp.adjoint(op), _adjoint_spin_swap, check_matrix=True)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["mux", "adjoint"])
def test_spin_mux_toffoli_count(adjoint):
    """T4: T = 1, one plain Toffoli (A = 0), M = 0, for the operator and its pinned adjoint."""

    def program():
        op = SpinMux([0], [1], [2], [3])
        if adjoint:
            qp.adjoint(op)

    resources = compiled_resources(program, num_wires=4)
    assert (resources.toffoli, resources.measurements) == (costs.spin_mux().toffoli, 0)
    assert (resources.counts["Toffoli"], resources.measured_and_uncomputes) == (1, 0)


@pytest.mark.compiled
@pytest.mark.parametrize(("p", "adjoint"), [(H2_STO3G_TEST, False), (H2_631G_TEST, False),
                                            (MID_NAPHTHALENE, False), (FEMOCO54, False),
                                            (H2_631G_TEST, True)],
                         ids=["sto3g", "631g", "mid", "femoco-width", "631g-adjoint"])
def test_spin_swap_toffoli_count(p, adjoint):
    """T4 at both test presets, MID_NAPHTHALENE and FeMoco-54's width (table-free, so per
    commit): T = N plain Toffolis from the CSWAPs (A = 0), M = 0; the pinned adjoint the same."""
    up, dn = list(range(1, 1 + p.N)), list(range(1 + p.N, 1 + 2 * p.N))

    def program():
        op = SpinSwap([0], up, dn)
        if adjoint:
            qp.adjoint(op)

    resources = compiled_resources(program, num_wires=1 + 2 * p.N)
    assert (resources.toffoli, resources.measurements) == (costs.spin_swap(p).toffoli, 0)
    assert (resources.counts["Toffoli"], resources.measured_and_uncomputes) == (p.N, 0)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 (spec 9.24): 1 per SpinMux, 54 per SpinSwap; 2 + 108 per InnerSelect."""
    assert costs.spin_mux() == costs.Cost(1)
    assert costs.spin_swap(FEMOCO54) == costs.Cost(54)
    assert 2 * costs.spin_mux() + 2 * costs.spin_swap(FEMOCO54) == costs.Cost(110)


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names the four pinned rules; T5 at level="device" (L-20)."""

    def program():
        SpinMux([0], [1], [2], [3])
        qp.adjoint(SpinMux([0], [1], [2], [3]))
        SpinSwap([3], [4, 5], [6, 7])
        qp.adjoint(SpinSwap([3], [4, 5], [6, 7]))

    qp.analyze(verbose_program(program, num_wires=8), level="user")()
    captured = capfd.readouterr()
    assert {"_spin_mux", "_adjoint_spin_mux", "_spin_swap", "_adjoint_spin_swap"} <= set(
        chosen_rules(captured.out + captured.err))
    compiled_resources(program, num_wires=8, level="device")


def test_required_wires():
    assert SpinMux.required_wires() == {"G0": 1, "sigma_0": 1, "sigma_1": 1, "sg": 1}
    assert SpinSwap.required_wires(54) == {"sg": 1, "psi_up": 54, "psi_dn": 54}


@pytest.mark.parametrize(("build", "match"), [
    (lambda: SpinMux([0, 4], [1], [2], [3]), "'G0' has 2 wires; expected 1"),
    (lambda: SpinMux([0], [1], [2], []), "'sg' has 0 wires; expected 1"),
    (lambda: SpinSwap([0], [1, 2], [3]), "'psi_dn' has 1 wires; expected 2"),
    (lambda: SpinSwap([0], [], []), "N >= 1"),
])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()
