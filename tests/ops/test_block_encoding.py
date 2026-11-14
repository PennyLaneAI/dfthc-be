"""DFTHCBlockEncoding and ControlledDFTHCBlockEncoding (spec 9.21): the block against the encoded
model on the tape path, the three control states, U then U-dagger, the children's order and
registers, counts, rule selection, validation."""

import re
import warnings

import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, compiled_resources
from dfthc.data.factors import load_factors
from dfthc.ops.walk import (
    ControlledDFTHCBlockEncoding,
    DFTHCBlockEncoding,
    _block_encoding,
    _controlled_block_encoding,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.registers import WalkLayout
from dfthc.testing import walk_sim
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program

CLASSES = {"plain": (DFTHCBlockEncoding, _block_encoding),
           "controlled": (ControlledDFTHCBlockEncoding, _controlled_block_encoding)}
case = walk_sim.case
"""The test shapes: ``sto3g``, ``631g`` and ``fails-c3`` (walk_sim.FAILS_C3: the failed uniform
preparation of x_o, dfthcbe's differing failure signs, C = 3)."""


SUPPORT = {"sto3g": (5, 10), "fails-c3": (5, 12)}
"""Two-electron system inputs (psi_up's bits first): |01>|01> and |10>|10> at sto3g, |01>|01>
and |11>|00> at fails-c3 (about 1.1 and 0.9 s each there). Not both electrons of every input in
one spin: there a D1 or Q1 row's a-dagger a and a a-dagger average to the same value over the
spin, which hides the select exchanged with its dagger."""


# --------------------------------------------------------------------------- semantics


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_block_against_the_encoded_model(name):
    """T3 on the tape path (strict sparse simulator: every erase outcome has probability 1/2 and
    every AND precondition holds): from a superposition of two system states with every ancilla
    in |0>, the output's part with every ancilla in |0> and phase_grad in the gradient state is
    (H~_SA / Lambda~ - I) applied to the input (encoded_block, spec 8.5), under the alternating
    and a seeded schedule of the erases. fails-c3 has the failed uniform preparation of x_o (so
    Lambda~ != Lambda) and the dfthcbe failure-sign difference; sto3g is the H2 test preset.
    Outside the block, only the six index registers of A u B are nonzero (the postcondition that
    makes RefAB the walk's reflection), and the output has norm 1."""
    c = case(name)
    ops = walk_sim.program(c, DFTHCBlockEncoding)
    psi = walk_sim.superposition(c, SUPPORT[name], seed=3)
    for kind in ("alternating", "seeded"):
        out = walk_sim.simulate(c, ops, walk_sim.input_state(c, psi),
                                walk_sim.schedule(kind, walk_sim.num_measurements(ops)))
        block = walk_sim.project(c, out)
        assert np.allclose(block[0], c.block() @ psi, rtol=0, atol=1e-12)
        assert np.allclose(block[1], 0, rtol=0, atol=1e-12)
        assert walk_sim.nonzero_registers(c, out) <= set(walk_sim.AB_REGISTERS)
        assert np.isclose(np.linalg.norm(out.amps), 1, rtol=0, atol=1e-12)


def test_controlled_block_encoding_in_three_control_states():
    """T3 of the controlled class at H2_STO3G_TEST: control |0> returns the whole input
    (U-dagger U = I), control |1> gives DFTHCBlockEncoding's whole output, and control |+> their
    superposition with weight 1/sqrt 2 each, which fails if the controlled reflection had a wrong
    global phase (spec 3.3). The controlled reflection is table-free, so one shape suffices."""
    c = case("sto3g")
    psi = walk_sim.basis(c, 10)
    plain = walk_sim.program(c, DFTHCBlockEncoding)
    controlled = walk_sim.program(c, ControlledDFTHCBlockEncoding, controlled=True)
    outcomes = walk_sim.schedule("seeded", walk_sim.num_measurements(plain), seed=4)
    identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    applied = walk_sim.set_control(c, walk_sim.simulate(c, plain, walk_sim.input_state(c, psi),
                                                        outcomes))
    plus = 2**-0.5
    expected = {"0": identity, "1": applied,
                "+": walk_sim.superpose((plus, identity), (plus, applied))}
    for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", [plus, plus])):
        out = walk_sim.simulate(c, controlled, walk_sim.input_state(c, psi, control), outcomes)
        walk_sim.assert_same_state(out, expected[label])
    assert np.allclose(walk_sim.project(c, applied)[1], c.block() @ psi, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_prepare_then_unprepare_is_the_identity(name):
    """The E-4 precursor (spec 12.4): the rule's children without the reflection, U then
    U-dagger, return a superposition of two system states and every ancilla exactly,
    under the alternating and a seeded schedule of the five erases (an all-ones schedule sees only
    each address's mask parity; ledger row 2D.2)."""
    c = case(name)
    ops = walk_sim.prepare_unprepare(c)
    psi = walk_sim.superposition(c, SUPPORT[name], seed=5)
    identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    for kind in ("alternating", "seeded"):
        out = walk_sim.simulate(c, ops, walk_sim.input_state(c, psi),
                                walk_sim.schedule(kind, walk_sim.num_measurements(ops), seed=6))
        walk_sim.assert_same_state(out, identity)


def children(rule, c, controlled):
    """The rule's operators in tape mode."""
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(*c.compilable, **c.groups(controlled))
    return list(queue.queue)


def registers(op):
    op = getattr(op, "base", op)
    return {name: list(getattr(op, name)) for name in op.wire_argnames}


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_children_order_and_registers(name):
    """Spec 9.21's order, every child on the layout's registers of its own names (spec 9.0):
    the work pool's prefix, the whole phase_grad; InnerSigns after InnerPrepare in the first
    round and after InnerSelectDagger in the second; the two rules differ only in the reflection,
    ControlledRefT2 on the control; the two InnerPrepare and InnerUnprepare calls are equal, and
    the selects have equal arguments with fixup_bits = w_b, so each shares one target."""
    c = case(name)
    p, layout = c.params, c.layout
    names = ["OuterPrepare", "InnerPrepare", "InnerSigns", "InnerSelect", "InnerUnprepare",
             "RefT2", "InnerPrepare", "InnerSelectDagger", "InnerSigns", "InnerUnprepare",
             "OuterUnprepare"]
    plain, controlled = children(_block_encoding, c, False), children(
        _controlled_block_encoding, c, True)
    assert [op.name for op in plain] == names
    assert [op.name for op in controlled] == [
        "ControlledRefT2" if n == "RefT2" else n for n in names]
    for op in plain + controlled:
        for register, wires in registers(op).items():
            if register == "work":
                assert wires == list(layout.work)[:len(wires)]
            elif register == "control":
                assert wires == c.control
            else:
                assert wires == list(getattr(layout, register)), (op.name, register)
    for a, b in zip(plain, controlled, strict=True):
        if a.name != "RefT2":
            assert (a.hyperparameters, registers(a)) == (b.hyperparameters, registers(b))
    reflection, controlled_reflection = registers(plain[5]), registers(controlled[5])
    assert controlled_reflection.pop("control") == c.control
    assert len(controlled_reflection.pop("work")) == len(reflection.pop("work")) + 1
    assert controlled_reflection == reflection
    assert plain[1].hyperparameters == plain[6].hyperparameters
    assert plain[4].hyperparameters == plain[9].hyperparameters
    assert plain[3].hyperparameters == plain[7].hyperparameters
    assert plain[3].hyperparameters["fixup_bits"] == p.w_b
    assert plain[0].hyperparameters == plain[10].hyperparameters == {
        "X_o": p.X_o, "alt": c.tables.outer_alt, "keep": c.tables.outer_keep, "b_k1": p.b_k1,
        "k1": p.k1, "ke1": p.ke1, "s_u": p.s_u}


# --------------------------------------------------------------------------- resources


def assert_counts(resources, p, controlled):
    """T4 and the T6 signature: T and M from costs.block_encoding; the plain Toffolis are the
    walk's T_Toffoli less RefAB's one, so A = T - (T_Toffoli - 1) (spec 9.24)."""
    expected = costs.block_encoding(p, controlled)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.counts["Toffoli"] == costs.toffoli_only(p) - 1
    assert resources.measured_and_uncomputes == resources.toffoli - (costs.toffoli_only(p) - 1)


def compile_class(c, kind):
    cls, _ = CLASSES[kind]
    controlled = kind == "controlled"

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        cls(*c.compilable, **c.groups(controlled))

    return compiled_resources(program, c.num_wires)


@pytest.mark.compiled
@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize("name", ["sto3g", "631g", "fails-c3"])
def test_counts(name, kind):
    """T4 compiled, each class alone: T, M, A at both test presets (T = 150 / 151 and 231 / 232,
    M = 50 and 68) and at fails-c3."""
    c = case(name)
    assert_counts(compile_class(c, kind), c.params, kind == "controlled")


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("kind", CLASSES)
def test_counts_at_mid_naphthalene(kind):
    """T4 at MID_NAPHTHALENE (nightly tier): T = 1,874 / 1,875, M = 502, A = 1,640 / 1,641."""
    c = walk_sim.WalkCase.build(MID_NAPHTHALENE, load_factors("naphthalene"))
    assert_counts(compile_class(c, kind), c.params, kind == "controlled")


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 (spec 9.24), without compiling: T = 279 + 2 * 4,725 + 20 = 9,749 (controlled
    9,750), M = 96 + 2 * (416 + 798) = 2,524, A = T - (1,118 - 1) = 8,632 (8,633); the groups are
    143, 455, 799, 108, 15 and 44 wires, 1,564 in all."""
    p = FEMOCO54
    outer = costs.outer_prepare(p) + costs.outer_unprepare(p)
    inner = costs.inner_prepare(p) + costs.inner_select(p) + costs.inner_unprepare(p)
    assert (outer, inner, costs.ref_t2(p, False)) == (costs.Cost(279, 96), costs.Cost(4725, 1214),
                                                      costs.Cost(20))
    assert costs.block_encoding(p, False) == costs.Cost(9749, 2524)
    assert costs.block_encoding(p, True) == costs.Cost(9750, 2524)
    assert costs.toffoli_only(p) == 1118
    assert [costs.block_encoding(p, c).toffoli - (costs.toffoli_only(p) - 1)
            for c in (False, True)] == [8632, 8633]
    sizes = DFTHCBlockEncoding.required_wires(p.as_compilable(), ())
    assert sizes == {"outer": 143, "inner": 455, "select": 799, "system": 108, "phase_grad": 15,
                     "work": 44}
    assert sum(sizes.values()) == WalkLayout.from_params(p).num_wires == 1564


def both_classes():
    c = case("sto3g")

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        DFTHCBlockEncoding(*c.compilable, **c.groups())
        ControlledDFTHCBlockEncoding(*c.compilable, **c.groups(True))

    return program, c.num_wires


@pytest.mark.compiled
def test_pinned_rules_are_chosen(capfd):
    """T6 at H2_STO3G_TEST: the verbose listing of a program holding both classes names both
    pinned rules and the two reflections', and not C(ZeroReflection)'s."""
    program, num_wires = both_classes()
    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_block_encoding", "_controlled_block_encoding", "_ref_t2",
            "_controlled_ref_t2"} <= rules
    assert "_c_zero_reflection" not in rules


@pytest.mark.compiled
def test_nothing_survives_at_device_level():
    """T5 at level="device" (L-20): no dfthc operator survives in a program holding both
    classes."""
    program, num_wires = both_classes()
    compiled_resources(program, num_wires, level="device")


@pytest.mark.compiled
@pytest.mark.parametrize("wrap", ["adjoint", "control"])
def test_no_adjoint_or_controlled_path(wrap):
    """Spec 7.4 rules 5 and 6: nothing adjoints or controls the block encoding (the controlled
    form is ControlledDFTHCBlockEncoding). Wrapped anyway, Catalyst drops the measuring candidates
    with the expected warning (rule 12), and the compile fails loudly."""
    c = case("sto3g")
    op = DFTHCBlockEncoding(*c.compilable, **c.groups())
    key = "Adjoint(DFTHCBlockEncoding)" if wrap == "adjoint" else "C(DFTHCBlockEncoding)"
    program = ((lambda: qp.adjoint(op)) if wrap == "adjoint"
               else (lambda: qp.ctrl(op, control=c.control[0])))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(program, c.num_wires)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the ")]
    named = [message for message in skipped if classify_rule_warning(message) != "noise"]
    assert named and {classify_rule_warning(message) for message in named} == {"expected"}


@pytest.mark.parametrize("name", ["sto3g", "631g", "fails-c3"])
def test_rules_are_valid(name):
    """T2 (tape mode) for both rules at three sizes: the declared children equal the emitted ones
    (compilable arguments and register sizes, the repeated prepares counted twice); each rule is
    pinned and the only one under its class name (spec 7.4 rules 7 and 8)."""
    c = case(name)
    for kind, (cls, rule) in CLASSES.items():
        assert FIXED_DECOMPS[cls.__name__] is rule
        assert list(qp.list_decomps(cls)) == [rule]
        op = cls(*c.compilable, **c.groups(kind == "controlled"))
        qp.testing.assert_valid_decomp_rule(op, rule, check_matrix=False)


def test_required_wires():
    """By hand at H2_STO3G_TEST: the layout's groups (work: RefAB's controlled need, 8)."""
    p = H2_STO3G_TEST
    sizes = {"outer": 16, "inner": 23, "select": 8, "system": 4, "phase_grad": 4, "work": 8}
    assert DFTHCBlockEncoding.required_wires(p.as_compilable(), ()) == sizes
    assert ControlledDFTHCBlockEncoding.required_wires(p.as_compilable(), ()) == {
        "control": 1} | sizes


@pytest.mark.parametrize(("name", "change", "match"), [
    ("sto3g", {"params": 7}, "params and tables must be"),
    ("sto3g", {"params": H2_631G_TEST.as_compilable()}, r"alt must be a table of ints of shape"),
    ("fails-c3", {"params": walk_sim.FAILS_C3.as_compilable()[:-1] + (3,)},
     r"the erase's one-hot over ke2 \+ lg\(B \+ 1\) = 5 bits needs 32 wires"),
    ("sto3g", {"tables": ()}, "WalkTables needs 8 fields; got 0"),
    ("sto3g", {"work": list(range(7))}, "'work' has 7 wires; expected 8"),
    ("sto3g", {"system": [0]}, "'system' has 1 wires; expected 4"),
], ids=["params-type", "params-shape", "onehot", "tables", "work", "system"])
@pytest.mark.parametrize("kind", CLASSES)
def test_constructor_validates(kind, name, change, match):
    """Spec 9.0: a wrong argument type, tables of another shape, a one-hot register that does
    not fit (ke2 = 3 needs 2**(3 + w_b) = 32 wires of the inner erase's 8), missing tables and
    wrong group widths each raise ValueError."""
    c = case(name)
    cls, _ = CLASSES[kind]
    params, tables = c.compilable
    arguments = {"params": params, "tables": tables} | c.groups(kind == "controlled") | change
    with pytest.raises(ValueError, match=match):
        cls(**arguments)


def test_constructor_checks_the_work_pool(monkeypatch):
    """The layout's pool must cover every child of the walk: with WalkLayout.work_size shrunk to
    7 at H2_STO3G_TEST, ControlledRefAB's 8 wires do not fit (spec 10.1)."""
    c = case("sto3g")
    monkeypatch.setattr(WalkLayout, "work_size", staticmethod(lambda params: 7))
    with pytest.raises(ValueError, match=r"too small for \{'ControlledRefAB': 8\}"):
        DFTHCBlockEncoding(*c.compilable, **c.groups())
