"""RefT2, RefAB, ControlledRefT2 and ControlledRefAB (spec 9.19, 9.20): the exact action with the
global sign, the control in |0>, |1> and |+>, counts, rule selection, validation."""

import math

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.reflections import (
    ControlledRefAB,
    ControlledRefT2,
    RefAB,
    RefT2,
    _adjoint_ref_ab,
    _adjoint_ref_t2,
    _controlled_ref_ab,
    _controlled_ref_t2,
    _ref_ab,
    _ref_t2,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.registers import WalkLayout
from dfthc.testing.resources import chosen_rules, verbose_program

PRESETS = {"sto3g": H2_STO3G_TEST, "631g": H2_631G_TEST, "mid": MID_NAPHTHALENE,
           "femoco-width": FEMOCO54}
# class, its pattern registers in wire order (control and work excluded), the registers with
# pattern value 1, the closed form, whether it is controlled, its rule
CLASSES = {
    "RefT2": (RefT2, ("succ_x_o", "b", "m_b", "sigma_0"), {"succ_x_o"}, costs.ref_t2, False,
              _ref_t2),
    "RefAB": (RefAB, ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0"), set(), costs.ref_ab,
              False, _ref_ab),
    "ControlledRefT2": (ControlledRefT2, ("succ_x_o", "b", "m_b", "sigma_0"), {"succ_x_o"},
                        costs.ref_t2, True, _controlled_ref_t2),
    "ControlledRefAB": (ControlledRefAB, ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0"),
                        set(), costs.ref_ab, True, _controlled_ref_ab),
}


def sizes(kind, p):
    """The class's required_wires for the walk's register widths at p (WalkLayout's)."""
    cls, pattern, *_ = CLASSES[kind]
    widths = WalkLayout.register_sizes(p)
    args = ([widths["b"], widths["m_b"]] if "x_o" not in pattern
            else [widths["x_o"], widths["m_x_o"], widths["b"], widths["m_b"]])
    return cls.required_wires(*args)


def layout(register_sizes):
    wires, start = {}, 0
    for name, size in register_sizes.items():
        wires[name], start = list(range(start, start + size)), start + size
    return wires, start


def phased_prepare(wires):
    """H on each wire, S on even positions, CZ on consecutive pairs (non-uniform phases)."""
    for wire in wires:
        qp.H(wire)
    for position, wire in enumerate(wires):
        if position % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


def phased_amplitudes(n):
    """The state phased_prepare makes on n wires, wire 0 the most significant bit."""
    bits = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    return (1j ** bits[:, ::2].sum(axis=1) * (-1.0) ** (bits[:, :-1] * bits[:, 1:]).sum(axis=1)
            / math.sqrt(2**n))


def reflection_diagonal(pattern_sizes, ones):
    """Diagonal of 2 Pi - I over the pattern registers (wire order, big-endian): +1 on the one
    basis state with the registers in ``ones`` all ones and the others all zeros, -1 elsewhere."""
    target = "".join(("1" if name in ones else "0") * n for name, n in pattern_sizes.items())
    return np.where(np.arange(2 ** len(target)) == int(target, 2), 1.0, -1.0)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["plain", "adjoint"])
@pytest.mark.parametrize("preset", ["sto3g", "631g"])
@pytest.mark.parametrize("kind", ["RefT2", "RefAB"])
def test_exact_action_with_the_global_sign(kind, preset, adjoint):
    """T3 on the compiled path (lightning) at the walk's register widths of both test presets: on
    a phased state of every pattern register (every amplitude nonzero, phases not uniform) the
    output is exactly (2 Pi - I) applied, global sign included, with work back in |0>; the pinned
    adjoint (the reflection itself) gives the same."""
    cls, pattern, ones, *_ = CLASSES[kind]
    register_sizes = sizes(kind, PRESETS[preset])
    wires, num_wires = layout(register_sizes)
    pattern_wires = [wire for name in pattern for wire in wires[name]]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        phased_prepare(pattern_wires)
        op = cls(**wires)
        if adjoint:
            qp.adjoint(op)
        return qp.state()

    psi = phased_amplitudes(len(pattern_wires))
    diagonal = reflection_diagonal({name: register_sizes[name] for name in pattern}, ones)
    expected = np.kron(diagonal * psi, np.eye(2 ** register_sizes["work"])[0])
    assert np.allclose(np.asarray(circuit()), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize("preset", ["sto3g", "631g"])
@pytest.mark.parametrize("kind", ["ControlledRefT2", "ControlledRefAB"])
def test_controlled_action_for_three_control_states(kind, preset):
    """T3 on the compiled path for the controlled classes, one program with the control's state
    as a runtime argument: with the control in |0> the phased pattern state is unchanged, in |1>
    it gets 2 Pi - I, and in |+> the output is (|0> psi + |1> (2 Pi - I) psi) / sqrt 2, which
    detects a wrong global phase on the controlled branch (spec 3.3); work ends in |0>."""
    cls, pattern, ones, *_ = CLASSES[kind]
    register_sizes = sizes(kind, PRESETS[preset])
    wires, num_wires = layout(register_sizes)
    pattern_wires = [wire for name in pattern for wire in wires[name]]
    control = wires["control"][0]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit(mode):
        qp.cond(mode == 1, qp.X)(control)
        qp.cond(mode == 2, qp.H)(control)
        phased_prepare(pattern_wires)
        cls(**wires)
        return qp.state()

    psi = phased_amplitudes(len(pattern_wires))
    reflected = reflection_diagonal({name: register_sizes[name] for name in pattern}, ones) * psi
    clean = np.eye(2 ** register_sizes["work"])[0]
    zero, one = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    expected = {0: np.kron(zero, psi), 1: np.kron(one, reflected),
                2: (np.kron(zero, psi) + np.kron(one, reflected)) / math.sqrt(2)}
    for mode, state in expected.items():
        assert np.allclose(np.asarray(circuit(mode)), np.kron(state, clean), rtol=0,
                           atol=1e-10), mode


COUNT_CASES = [pytest.param(kind, preset, id=f"{kind}-{preset}")
               for kind in CLASSES for preset in PRESETS]


@pytest.mark.compiled
@pytest.mark.parametrize(("kind", "preset"), COUNT_CASES)
def test_counts(kind, preset):
    """T4 compiled at both test presets, MID_NAPHTHALENE and FeMoco-54's register widths (the
    reflections are table-free, so per commit): T = n_w - 2 (controlled n_w - 1) against
    dfthc.costs.ref_t2 and ref_ab, M = 0, one plain Toffoli, so A = T - 1."""
    cls, _, _, cost, controlled, _ = CLASSES[kind]
    p = PRESETS[preset]
    wires, num_wires = layout(sizes(kind, p))
    resources = compiled_resources(lambda: cls(**wires), num_wires)
    assert (resources.toffoli, resources.measurements) == (cost(p, controlled).toffoli, 0)
    assert resources.counts["Toffoli"] == 1
    assert resources.measured_and_uncomputes == resources.toffoli - 1


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 (spec 9.24): RefT2 20 and 21 controlled, RefAB 44 and 45 controlled; the
    controlled RefAB's 44 work wires are the walk's largest need, WalkLayout.work_size."""
    p = FEMOCO54
    assert [costs.ref_t2(p, False), costs.ref_t2(p, True)] == [costs.Cost(20), costs.Cost(21)]
    assert [costs.ref_ab(p, False), costs.ref_ab(p, True)] == [costs.Cost(44), costs.Cost(45)]
    assert [sizes(kind, p)["work"] for kind in CLASSES] == [19, 43, 20, 44]
    assert sizes("ControlledRefAB", p)["work"] == WalkLayout.work_size(p) == 44


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names the six pinned rules and ZeroReflection's, and not
    C(ZeroReflection)'s: the controlled classes request no C(...) key (spec 7.4 rule 6). T5 at
    level="device" for all four classes and both adjoints (L-20)."""
    p = H2_STO3G_TEST

    def program():
        for kind, (cls, *_) in CLASSES.items():
            wires, _ = layout(sizes(kind, p))
            cls(**wires)
            if kind in ("RefT2", "RefAB"):
                qp.adjoint(cls(**wires))

    num_wires = max(layout(sizes(kind, p))[1] for kind in CLASSES)
    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_ref_t2", "_adjoint_ref_t2", "_controlled_ref_t2", "_ref_ab", "_adjoint_ref_ab",
            "_controlled_ref_ab", "_zero_reflection"} <= rules
    assert "_c_zero_reflection" not in rules
    compiled_resources(program, num_wires, level="device")


@pytest.mark.parametrize("preset", ["sto3g", "631g"])
@pytest.mark.parametrize("kind", CLASSES)
def test_rules_are_valid(kind, preset):
    """T2 (tape mode) at two sizes for the four base rules and the two pinned adjoints; each rule
    is the only one registered under its key (spec 7.4 rules 7 and 8)."""
    cls, _, _, _, controlled, rule = CLASSES[kind]
    wires, _ = layout(sizes(kind, PRESETS[preset]))
    op = cls(**wires)
    assert FIXED_DECOMPS[kind] is rule
    assert list(qp.list_decomps(cls)) == [rule]
    qp.testing.assert_valid_decomp_rule(op, rule, check_matrix=False)
    if not controlled:
        adjoint_rule = {"RefT2": _adjoint_ref_t2, "RefAB": _adjoint_ref_ab}[kind]
        assert list(qp.list_decomps(f"Adjoint({kind})")) == [adjoint_rule]
        qp.testing.assert_valid_decomp_rule(qp.adjoint(op), adjoint_rule, check_matrix=False)


def test_required_wires():
    """By hand at H2_STO3G_TEST (w_x = w_b = b_k1 = b_k2 = 2): n_w = 6 and 10."""
    assert RefT2.required_wires(2, 2) == {"succ_x_o": 1, "b": 2, "m_b": 2, "sigma_0": 1,
                                          "work": 3}
    assert ControlledRefT2.required_wires(2, 2) == {"control": 1, "succ_x_o": 1, "b": 2,
                                                    "m_b": 2, "sigma_0": 1, "work": 4}
    assert RefAB.required_wires(2, 2, 2, 2) == {"x_o": 2, "m_x_o": 2, "sigma_1": 1, "b": 2,
                                                "m_b": 2, "sigma_0": 1, "work": 7}
    assert ControlledRefAB.required_wires(2, 2, 2, 2)["work"] == 8


@pytest.mark.parametrize(("build", "match"), [
    (lambda: RefT2([0], [], [1], [2], []), r"RefT2: len\(b\) must lie in \[1, None\]"),
    (lambda: RefT2([0, 9], [1], [2], [3], [4]), "'succ_x_o' has 2 wires; expected 1"),
    (lambda: RefT2([0], [1], [2], [3], []), "'work' has 0 wires; expected 1"),
    (lambda: ControlledRefT2([], [0], [1], [2], [3], [4, 5]), "'control' has 0 wires"),
    (lambda: RefAB([0], [1], [2], [3], [], [4], [5, 6, 7]), r"len\(m_b\) must lie in \[1, None\]"),
    (lambda: ControlledRefAB([9], [0], [1], [2], [3], [4], [5], [6, 7]),
     "'work' has 2 wires; expected 4"),
])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()
