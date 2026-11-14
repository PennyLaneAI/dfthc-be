"""DFTHCWalk and ControlledDFTHCWalk (spec 9.22): the walk against the block encoding and the
encoded model on the tape path, the three control states, the children, rule selection, the
inventory of captured targets and its instrument, the failing qp.ctrl(DFTHCWalk), validation."""

import re
import warnings

import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc.compile import FIXED_DECOMPS, compiled_resources
from dfthc.ops.prepare import InnerSigns
from dfthc.ops.walk import (
    ControlledDFTHCBlockEncoding,
    ControlledDFTHCWalk,
    DFTHCBlockEncoding,
    DFTHCWalk,
    _controlled_walk,
    _walk,
)
from dfthc.params import H2_STO3G_TEST
from dfthc.testing import walk_sim
from dfthc.testing.resources import (
    LEAF_ADJOINTS,
    TEMPLATE_ADJOINTS,
    captured_targets,
    classify_rule_warning,
    inventory,
    verbose_program,
)

CLASSES = {"plain": (DFTHCWalk, _walk), "controlled": (ControlledDFTHCWalk, _controlled_walk)}
SUPPORT = {"sto3g": (5, 10), "fails-c3": (5, 12)}
"""Two-electron system inputs (psi_up's bits first), as in test_block_encoding.py."""


# --------------------------------------------------------------------------- semantics


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_walk_is_ref_ab_after_the_block_encoding(name):
    """T3 on the tape path (strict sparse simulator): the walk's whole output equals
    DFTHCBlockEncoding's whole output with RefAB's exact action applied (-1 on every term with a
    nonzero x_o, m_x_o, sigma_1, b, m_b or sigma_0; spec 3.3), so its block is
    H~_SA / Lambda~ - I (spec 12.4, E-2 at k = 1), under a seeded schedule."""
    c = walk_sim.case(name)
    psi = walk_sim.superposition(c, SUPPORT[name], seed=7)
    walk_ops, be_ops = walk_sim.program(c, DFTHCWalk), walk_sim.program(c, DFTHCBlockEncoding)
    outcomes = walk_sim.schedule("seeded", walk_sim.num_measurements(walk_ops), seed=8)
    walked = walk_sim.simulate(c, walk_ops, walk_sim.input_state(c, psi), outcomes)
    encoded = walk_sim.simulate(c, be_ops, walk_sim.input_state(c, psi), outcomes)
    walk_sim.assert_same_state(walked, walk_sim.reflect_ab(c, encoded))
    assert np.allclose(walk_sim.project(c, walked)[0], c.block() @ psi, rtol=0, atol=1e-12)


def test_controlled_walk_in_three_control_states():
    """T3 of the controlled class at H2_STO3G_TEST: control |0> returns the whole input, control
    |1> gives DFTHCWalk's whole output, and control |+> their superposition with weight 1/sqrt 2
    each; a wrong global phase of either controlled reflection fails the last (spec 3.3)."""
    c = walk_sim.case("sto3g")
    psi = walk_sim.basis(c, 10)
    plain = walk_sim.program(c, DFTHCWalk)
    controlled = walk_sim.program(c, ControlledDFTHCWalk, controlled=True)
    outcomes = walk_sim.schedule("alternating", walk_sim.num_measurements(plain))
    identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    applied = walk_sim.set_control(c, walk_sim.simulate(c, plain, walk_sim.input_state(c, psi),
                                                        outcomes))
    plus = 2**-0.5
    expected = {"0": identity, "1": applied,
                "+": walk_sim.superpose((plus, identity), (plus, applied))}
    for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", [plus, plus])):
        out = walk_sim.simulate(c, controlled, walk_sim.input_state(c, psi, control), outcomes)
        walk_sim.assert_same_state(out, expected[label])


def test_controlled_walk_after_a_walk_step():
    """T3 of the controlled class on the walk's own states, as iterative QPE uses it (spec 11.1):
    after one DFTHCWalk step the index registers of A u B are not |0>, and the controlled walk
    with the control in |+> gives (|0> W|psi> + |1> W^2|psi>)/sqrt 2. From ancillas in |0> an
    uncontrolled RefAB would act as the identity on the control-0 branch; here it would not."""
    c = walk_sim.case("sto3g")
    psi = walk_sim.basis(c, 10)
    once = walk_sim.program(c, DFTHCWalk)
    twice = walk_sim.program(c, DFTHCWalk, repeat=2)
    m = walk_sim.num_measurements(once)
    then_controlled = walk_sim.operations(
        c, lambda: DFTHCWalk(*c.compilable, **c.groups()),
        lambda: ControlledDFTHCWalk(*c.compilable, **c.groups(True)))
    outcomes = walk_sim.schedule("seeded", 2 * m, seed=9)
    stepped = walk_sim.simulate(c, once, walk_sim.input_state(c, psi), outcomes[:m])
    stepped_twice = walk_sim.simulate(c, twice, walk_sim.input_state(c, psi), outcomes)
    plus = 2**-0.5
    out = walk_sim.simulate(c, then_controlled, walk_sim.input_state(c, psi, [plus, plus]),
                            outcomes)
    assert walk_sim.nonzero_registers(c, stepped)
    walk_sim.assert_same_state(out, walk_sim.superpose(
        (plus, stepped), (plus, walk_sim.set_control(c, stepped_twice))))


def registers(op):
    return {name: list(getattr(op, name)) for name in op.wire_argnames}


@pytest.mark.parametrize("kind", CLASSES)
def test_children(kind):
    """Spec 9.22: the (controlled) block encoding on the whole groups and pool, then the
    (controlled) RefAB on x_o, m_x_o, sigma_1, b, m_b, sigma_0 and the pool's prefix of exactly
    its need; the control goes to both children and nowhere else."""
    c = walk_sim.case("sto3g")
    controlled = kind == "controlled"
    _, rule = CLASSES[kind]
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(*c.compilable, **c.groups(controlled))
    be, ref = queue.queue
    layout = c.layout
    assert (be.name, ref.name) == (("ControlledDFTHCBlockEncoding", "ControlledRefAB")
                                   if controlled else ("DFTHCBlockEncoding", "RefAB"))
    assert registers(be) == c.groups(controlled)
    assert be.hyperparameters == {"params": c.compilable[0], "tables": c.compilable[1]}
    ab = {name: list(getattr(layout, name)) for name in walk_sim.AB_REGISTERS}
    work = c.params.w_x + c.params.b_k1 + c.params.w_b + c.params.b_k2 - (0 if controlled else 1)
    assert registers(ref) == ({"control": c.control} if controlled else {}) | ab | {
        "work": list(layout.work)[:work]}


# --------------------------------------------------------------------------- rules, inventory


def chosen_by_operator(listing: str) -> dict[str, set[str]]:
    """``Operator: <name>{...}`` and the ``Chosen Rule: __builtin_<rule>`` line after it, as
    ``{name: {rule, ...}}`` (a verbose listing names each operator signature once)."""
    chosen, operator = {}, None
    for line in listing.splitlines():
        line = line.strip()
        if line.startswith("Operator: "):
            operator = line[len("Operator: "):].split("{", 1)[0]
        elif line.startswith("Chosen Rule: __builtin_") and operator is not None:
            chosen.setdefault(operator, set()).add(line[len("Chosen Rule: __builtin_"):])
            operator = None
    return chosen


@pytest.mark.compiled
@pytest.mark.parametrize("kind", CLASSES)
def test_pinned_rules_are_chosen(kind, capfd):
    """T6 at H2_STO3G_TEST: the listing pairs each assembly operator with its pinned rule, the
    reflections with theirs (no C(ZeroReflection)), and Adjoint(MajoranaGate), the select
    dagger's leaf adjoint, with a lowered copy of the base rule _majorana_gate (ledger row
    2E.5)."""
    c = walk_sim.case("sto3g")
    controlled = kind == "controlled"
    cls, _ = CLASSES[kind]

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        cls(*c.compilable, **c.groups(controlled))

    qp.analyze(verbose_program(program, c.num_wires), level="user")()
    captured = capfd.readouterr()
    chosen = chosen_by_operator(captured.out + captured.err)
    prefix = "Controlled" if controlled else ""
    expected = {f"{prefix}DFTHCWalk": "_controlled_walk" if controlled else "_walk",
                f"{prefix}DFTHCBlockEncoding": ("_controlled_block_encoding" if controlled
                                                else "_block_encoding"),
                f"{prefix}RefT2": "_controlled_ref_t2" if controlled else "_ref_t2",
                f"{prefix}RefAB": "_controlled_ref_ab" if controlled else "_ref_ab",
                "InnerSelect": "_inner_select", "InnerSelectDagger": "_inner_select_dagger"}
    assert {name: chosen[name] for name in expected} == {
        name: {rule} for name, rule in expected.items()}
    assert all(re.fullmatch(r"_majorana_gate(_\d+)?", rule)
               for rule in chosen["Adjoint(MajoranaGate)"])
    assert "C(ZeroReflection)" not in chosen


@pytest.mark.compiled
def test_nothing_survives_at_device_level():
    """T5 at level="device" for the H2 walk (L-20): no dfthc operator survives in a program
    holding the walk and the controlled walk."""
    c = walk_sim.case("sto3g")

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        DFTHCWalk(*c.compilable, **c.groups())
        ControlledDFTHCWalk(*c.compilable, **c.groups(True))

    compiled_resources(program, c.num_wires, level="device")


@pytest.mark.compiled
@pytest.mark.parametrize("kind", CLASSES)
def test_inventory_of_captured_targets(kind):
    """The inventory test (spec 7.4 rule 6, 12.1 T6; master plan section 3, item 20): compiling
    the walk or the controlled walk at H2_STO3G_TEST, the frontend converts the program's own
    top-level target (so the hook sees targets) and every Phase 3 class of the program, and no
    Adjoint(...) or C(...) target of a dfthc composite or QROAMRead: the modified targets of dfthc
    operators and of the templates they adjoint are exactly the leaf adjoints and the template
    adjoints."""
    c = walk_sim.case("sto3g")
    controlled = kind == "controlled"
    cls, _ = CLASSES[kind]
    names = captured_targets(lambda: (qp.PhaseGradientStatePrep(c.layout.phase_grad),
                                      cls(*c.compilable, **c.groups(controlled))), c.num_wires)
    prefix = "Controlled" if controlled else ""
    assert {f"{prefix}DFTHCWalk", f"{prefix}DFTHCBlockEncoding", "InnerSelect",
            "InnerSelectDagger", f"{prefix}RefT2", f"{prefix}RefAB"} <= set(names)
    assert names.count(f"{prefix}DFTHCWalk") == 1
    assert inventory(names) == LEAF_ADJOINTS | TEMPLATE_ADJOINTS


@pytest.mark.compiled
def test_inventory_flags_a_stray_modifier():
    """Negative control: one qp.adjoint of a dfthc operator outside the leaves (InnerSigns, whose
    pinned self-adjoint rule compiles) appears as Adjoint(InnerSigns) in the inventory."""
    names = captured_targets(lambda: qp.adjoint(InnerSigns(sgn=[0], d_sgn=[1],
                                                           less_than_b=[2])), 3)
    assert inventory(names) == {"Adjoint(InnerSigns)"}


@pytest.mark.compiled
def test_captured_targets_records_one_compile_and_restores_the_hook():
    """captured_targets wraps Catalyst's private hook for one compile only: a second call of the
    same program gives the same list and another program of the same width its own (nothing
    accumulates or is served from an earlier call), and afterwards, also after a program that
    raises, the module holds the original function again."""
    import catalyst.from_plxpr.qfunc_interpreter as qfunc_interpreter

    original = qfunc_interpreter._convert_decomp_target_spec
    registers = {"sgn": [0], "d_sgn": [1], "less_than_b": [2]}
    adjointed = list(captured_targets(lambda: qp.adjoint(InnerSigns(**registers)), 3))
    assert qfunc_interpreter._convert_decomp_target_spec is original
    assert captured_targets(lambda: qp.adjoint(InnerSigns(**registers)), 3) == adjointed
    plain = captured_targets(lambda: InnerSigns(**registers), 3)
    assert "Adjoint(InnerSigns)" in adjointed and "InnerSigns" in plain
    assert "Adjoint(InnerSigns)" not in plain

    def failing():
        raise RuntimeError("the program fails to trace")

    with pytest.raises(RuntimeError, match="fails to trace"):
        captured_targets(failing, 3)
    assert qfunc_interpreter._convert_decomp_target_spec is original


@pytest.mark.compiled
def test_controlled_walk_has_no_generic_controlled_rule():
    """Spec 7.4 rule 6 and 9.22, the Task G0.1 probe: qp.ctrl(DFTHCWalk(...)) at H2_STO3G_TEST
    fails to compile with a CompileError naming C(DFTHCWalk), after Catalyst's expected
    "Skipped ... mid-circuit measurement" warnings for the erases (rule 12)."""
    c = walk_sim.case("sto3g")
    op = DFTHCWalk(*c.compilable, **c.groups())
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                "Decomposition rule not found for operator 'C(DFTHCWalk)")):
            compiled_resources(lambda: qp.ctrl(op, control=c.control[0]), c.num_wires)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the ")]
    named = [message for message in skipped if classify_rule_warning(message) != "noise"]
    assert named and {classify_rule_warning(message) for message in named} == {"expected"}


@pytest.mark.parametrize("name", ["sto3g", "631g", "fails-c3"])
def test_rules_are_valid(name):
    """T2 (tape mode) for both rules at three sizes; each pinned and the only rule under its
    class name (spec 7.4 rules 7 and 8)."""
    c = walk_sim.case(name)
    for kind, (cls, rule) in CLASSES.items():
        assert FIXED_DECOMPS[cls.__name__] is rule
        assert list(qp.list_decomps(cls)) == [rule]
        op = cls(*c.compilable, **c.groups(kind == "controlled"))
        qp.testing.assert_valid_decomp_rule(op, rule, check_matrix=False)


def test_required_wires():
    """The block encoding's groups: the walk adds no register (RefAB's work is in the pool)."""
    p = H2_STO3G_TEST
    for plain, controlled in ((DFTHCWalk, ControlledDFTHCWalk),
                              (DFTHCBlockEncoding, ControlledDFTHCBlockEncoding)):
        sizes = plain.required_wires(p.as_compilable(), ())
        assert sizes == {"outer": 16, "inner": 23, "select": 8, "system": 4, "phase_grad": 4,
                         "work": 8}
        assert controlled.required_wires(p.as_compilable(), ()) == {"control": 1} | sizes


@pytest.mark.parametrize(("change", "match"), [
    ({"params": 7}, "params and tables must be"),
    ({"tables": ()}, "WalkTables needs 8 fields; got 0"),
    ({"outer": [0]}, "'outer' has 1 wires; expected 16"),
], ids=["params-type", "tables", "outer"])
@pytest.mark.parametrize("kind", CLASSES)
def test_constructor_validates(kind, change, match):
    """Spec 9.0: the walks run the block encoding's checks (test_block_encoding.py has the full
    set) and raise ValueError."""
    c = walk_sim.case("sto3g")
    cls, _ = CLASSES[kind]
    params, tables = c.compilable
    arguments = {"params": params, "tables": tables} | c.groups(kind == "controlled") | change
    with pytest.raises(ValueError, match=match):
        cls(**arguments)


def test_controlled_walk_checks_the_control_width():
    """ControlledDFTHCWalk checks its one-wire control like the groups (spec 9.0; its _validate
    call passes the class's own wire_argnames, so the block encoding's test does not cover it)."""
    c = walk_sim.case("sto3g")
    with pytest.raises(ValueError, match="'control' has 2 wires; expected 1"):
        ControlledDFTHCWalk(*c.compilable,
                            **c.groups(True) | {"control": [*c.control, c.num_wires]})
