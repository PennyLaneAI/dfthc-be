"""DFTHCWalk and its controlled form qp.ctrl(DFTHCWalk) (spec 9.22): the walk against the block
encoding and the encoded model on the tape path, the three control states for both control
values, the children, rule selection, the inventory of captured targets and its instrument,
validation."""

import re

import numpy as np
import pennylane as qp
import pytest

from dfthc.compile import FIXED_DECOMPS, compiled_resources
from dfthc.ops.prepare import InnerSigns
from dfthc.ops.walk import DFTHCBlockEncoding, DFTHCWalk, _controlled_walk, _walk
from dfthc.params import H2_STO3G_TEST
from dfthc.testing import walk_sim
from dfthc.testing.resources import (
    LEAF_ADJOINTS,
    TEMPLATE_ADJOINTS,
    captured_targets,
    inventory,
    verbose_program,
)

KINDS = ("plain", "controlled")
SUPPORT = {"sto3g": (5, 10), "fails-c3": (5, 12)}
"""Two-electron system inputs (psi_up's bits first), as in test_block_encoding.py."""


# --------------------------------------------------------------------------- semantics


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_walk_is_ref_ab_after_the_block_encoding(name):
    """T3 on the tape path (strict sparse simulator): the walk's whole output equals
    DFTHCBlockEncoding's whole output with ref_ab's exact action applied (-1 on every term with a
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


@pytest.mark.parametrize("control_value", [1, 0])
def test_controlled_walk_in_three_control_states(control_value):
    """T3 of qp.ctrl(DFTHCWalk) at H2_STO3G_TEST: the control in the state of the control value
    gives DFTHCWalk's whole output, in the other state it returns the whole input, and in |+>
    their superposition with weight 1/sqrt 2 each; a wrong global phase of either controlled
    reflection fails the last (spec 3.3)."""
    c = walk_sim.case("sto3g")
    psi = walk_sim.basis(c, 10)
    plain = walk_sim.program(c, DFTHCWalk)
    controlled = walk_sim.program(c, DFTHCWalk, controlled=True, control_value=control_value)
    outcomes = walk_sim.schedule("alternating", walk_sim.num_measurements(plain))
    identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    applied = walk_sim.simulate(c, plain, walk_sim.input_state(c, psi), outcomes)
    zero, one = (identity, applied) if control_value else (applied, identity)
    one = walk_sim.set_control(c, one)
    plus = 2**-0.5
    expected = {"0": zero, "1": one, "+": walk_sim.superpose((plus, zero), (plus, one))}
    for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", [plus, plus])):
        out = walk_sim.simulate(c, controlled, walk_sim.input_state(c, psi, control), outcomes)
        walk_sim.assert_same_state(out, expected[label])


def test_controlled_walk_after_a_walk_step():
    """T3 of the controlled class on the walk's own states, as iterative QPE uses it (spec 11.1):
    after one DFTHCWalk step the index registers of A u B are not |0>, and the controlled walk
    with the control in |+> gives (|0> W|psi> + |1> W^2|psi>)/sqrt 2. From ancillas in |0> an
    uncontrolled ref_ab would act as the identity on the control-0 branch; here it would not."""
    c = walk_sim.case("sto3g")
    psi = walk_sim.basis(c, 10)
    once = walk_sim.program(c, DFTHCWalk)
    twice = walk_sim.program(c, DFTHCWalk, repeat=2)
    m = walk_sim.num_measurements(once)
    then_controlled = walk_sim.operations(
        c, lambda: DFTHCWalk(*c.compilable, **c.groups()),
        lambda: qp.ctrl(DFTHCWalk(*c.compilable, **c.groups()), control=c.control))
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


def walk_op(c, controlled, control_value=1, control=None):
    """The walk on the case's groups, or qp.ctrl of it on ``control`` (default the case's)."""
    op = DFTHCWalk(*c.compilable, **c.groups())
    if not controlled:
        return op
    control = c.control if control is None else control
    return qp.ctrl(op, control=control, control_values=[control_value] * len(control))


def queued(c, kind, control_value=1):
    """What the walk's rule, or the C(DFTHCWalk) rule, queues in tape mode."""
    with qp.capture.pause():
        op = walk_op(c, kind == "controlled", control_value)
        with qp.queuing.AnnotatedQueue() as queue:
            if kind == "controlled":
                _controlled_walk(**op.arguments)
            else:
                _walk(**op.arguments)
    return list(queue.queue)


def signature(op):
    return op.name, list(op.wires)


@pytest.mark.parametrize("kind", KINDS)
def test_children(kind):
    """Spec 9.22: the block encoding on the whole groups and pool, then ref_ab: the
    multi-controlled Z on x_o, m_x_o, sigma_1, b, m_b, sigma_0 with the all-zero pattern (X and H
    around a MultiControlledX on the last sigma_0 wire, with the pool's prefix of exactly its
    need), then GlobalPhase(pi). Controlled, the block encoding's own steps with ref_t2 controlled
    (walk_sim.controlled_block_encoding), then ref_ab with the control as the first control (value
    1) and one more work wire, then Z on the control."""
    c = walk_sim.case("sto3g")
    controlled = kind == "controlled"
    ops = queued(c, kind)
    be, reflection = ops[:-6], ops[-6:]
    if controlled:
        with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
            walk_sim.controlled_block_encoding(c)
        assert [signature(op) for op in be] == [signature(op) for op in queue.queue]
    else:
        assert [op.name for op in be] == ["DFTHCBlockEncoding"]
        assert registers(be[0]) == c.groups()
        assert be[0].hyperparameters == {"params": c.compilable[0], "tables": c.compilable[1]}
    layout = c.layout
    assert [op.name for op in reflection] == [
        "PauliX", "Hadamard", "MultiControlledX", "Hadamard", "PauliX",
        "PauliZ" if controlled else "GlobalPhase"]
    ab = [wire for name in walk_sim.AB_REGISTERS for wire in getattr(layout, name)]
    work = c.params.w_x + c.params.b_k1 + c.params.w_b + c.params.b_k2 - (0 if controlled else 1)
    control = c.control if controlled else []
    mcx, sign = reflection[2], reflection[-1]
    assert all(list(op.wires) == ab[-1:] for op in reflection[:2] + reflection[3:5])
    assert list(mcx.wires) == control + ab
    assert list(mcx.control_values) == [True] * controlled + [False] * (len(ab) - 1)
    assert list(mcx.work_wires) == list(layout.work)[:work]
    assert list(sign.wires) == control


def test_control_value_0_flips_the_control_around_the_rule():
    """qp.ctrl(DFTHCWalk, control_values=[0]): the C(DFTHCWalk) rule puts X on the control first
    and last, and in between emits exactly what it emits for control value 1."""
    c = walk_sim.case("sto3g")
    ops = queued(c, "controlled", control_value=0)
    assert [signature(op) for op in (ops[0], ops[-1])] == [("PauliX", c.control)] * 2
    assert [signature(op) for op in ops[1:-1]] == [
        signature(op) for op in queued(c, "controlled")]


def test_more_than_one_control_wire_raises():
    """The C(DFTHCWalk) rule takes exactly one control wire."""
    c = walk_sim.case("sto3g")
    with qp.capture.pause():
        op = walk_op(c, True, control=[c.num_wires, c.num_wires + 1])
        with pytest.raises(ValueError, match="C\\(DFTHCWalk\\) takes one control wire; got 2"):
            _controlled_walk(**op.arguments)


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
@pytest.mark.parametrize("kind", KINDS)
def test_pinned_rules_are_chosen(kind, capfd):
    """T6 at H2_STO3G_TEST: the listing pairs each assembly operator with its pinned rule (the
    controlled walk, C(DFTHCWalk), with _controlled_walk and no block encoding operator), the
    Adjoint(MajoranaGate), the select dagger's leaf adjoint, with a lowered copy of the base rule
    _majorana_gate (ledger row 2E.5)."""
    c = walk_sim.case("sto3g")
    controlled = kind == "controlled"

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        walk_op(c, controlled)

    qp.analyze(verbose_program(program, c.num_wires), level="user")()
    captured = capfd.readouterr()
    chosen = chosen_by_operator(captured.out + captured.err)
    walk = ({"C(DFTHCWalk)": "_controlled_walk"} if controlled
            else {"DFTHCWalk": "_walk", "DFTHCBlockEncoding": "_block_encoding"})
    expected = walk | {"InnerSelect": "_inner_select", "InnerSelectDagger": "_inner_select_dagger"}
    assert ("DFTHCBlockEncoding" in chosen) is not controlled
    assert {name: chosen[name] for name in expected} == {
        name: {rule} for name, rule in expected.items()}
    assert all(re.fullmatch(r"_majorana_gate(_\d+)?", rule)
               for rule in chosen["Adjoint(MajoranaGate)"])


@pytest.mark.compiled
def test_nothing_survives_at_device_level():
    """T5 at level="device" for the H2 walk (L-20): no dfthc operator survives in a program
    holding the walk and the controlled walk."""
    c = walk_sim.case("sto3g")

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        walk_op(c, False)
        walk_op(c, True)

    compiled_resources(program, c.num_wires, level="device")


@pytest.mark.compiled
@pytest.mark.parametrize("kind", KINDS)
def test_inventory_of_captured_targets(kind):
    """The inventory test (spec 7.4 rule 6, 12.1 T6; master plan section 3, item 20): compiling
    the walk or the controlled walk at H2_STO3G_TEST, the frontend converts the program's own
    top-level target (so the hook sees targets) and every Phase 3 class of the program. The
    modified targets of dfthc operators and of the templates they adjoint are exactly the leaf
    adjoints and the template adjoints, and for the controlled walk its own C(DFTHCWalk), whose
    rule is pinned, and C(DFTHCBlockEncoding): Catalyst still builds the generic controlled
    candidate of C(DFTHCWalk), which requests C(DFTHCBlockEncoding), whose base rule measures, so
    the cascade stops there (rule 12)."""
    c = walk_sim.case("sto3g")
    controlled = kind == "controlled"
    names = captured_targets(lambda: (qp.PhaseGradientStatePrep(c.layout.phase_grad),
                                      walk_op(c, controlled)), c.num_wires)
    walk = "C(DFTHCWalk)" if controlled else "DFTHCWalk"
    assert {walk, "InnerSelect", "InnerSelectDagger"} <= set(names)
    assert ("DFTHCBlockEncoding" in names) is not controlled
    stops = {walk, "C(DFTHCBlockEncoding)"} if controlled else set()
    assert inventory(names) == LEAF_ADJOINTS | TEMPLATE_ADJOINTS | stops


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


@pytest.mark.parametrize("name", ["sto3g", "631g", "fails-c3"])
def test_rules_are_valid(name):
    """T2 (tape mode) for both rules at three sizes, the controlled walk's with control value 0
    (its conditional X gates on the control then run, as declared); each pinned and the only rule
    under its key (spec 7.4 rules 7 and 8)."""
    c = walk_sim.case(name)
    for kind, key, rule in (("plain", "DFTHCWalk", _walk),
                            ("controlled", "C(DFTHCWalk)", _controlled_walk)):
        assert FIXED_DECOMPS[key] is rule
        assert list(qp.list_decomps(key)) == [rule]
        with qp.capture.pause():
            op = walk_op(c, kind == "controlled", control_value=0)
        qp.testing.assert_valid_decomp_rule(op, rule, check_matrix=False)


def test_required_wires():
    """The block encoding's groups: the walk adds no register (ref_ab's work is in the pool)."""
    p = H2_STO3G_TEST
    for cls in (DFTHCWalk, DFTHCBlockEncoding):
        assert cls.required_wires(p.as_compilable(), ()) == {
            "outer": 12, "inner": 16, "select": 11, "system": 4, "phase_grad": 4, "work": 8}


@pytest.mark.parametrize(("change", "match"), [
    ({"params": 7}, "params and tables must be"),
    ({"tables": ()}, "WalkTables needs 8 fields; got 0"),
    ({"outer": [0]}, "'outer' has 1 wires; expected 12"),
], ids=["params-type", "tables", "outer"])
def test_constructor_validates(change, match):
    """Spec 9.0: the walk runs the block encoding's checks (test_block_encoding.py has the full
    set) and raises ValueError."""
    c = walk_sim.case("sto3g")
    params, tables = c.compilable
    arguments = {"params": params, "tables": tables} | c.groups() | change
    with pytest.raises(ValueError, match=match):
        DFTHCWalk(**arguments)
