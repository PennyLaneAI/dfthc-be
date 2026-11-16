"""dfthc.qpe (spec 9.23 and 11): ControlledWalkPower (T2 to T6), QPERound (T2, T4, T5, T6),
the walk-power registry and the corrections' wires, the driver's checks, WalkPowerSpec and
qpe_program, and the energy helpers of spec 11.1."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, compiled_resources
from dfthc.data.encoded import encoded_block
from dfthc.data.factors import load_factors, load_integrals, load_summary
from dfthc.data.fermion import sector_projector
from dfthc.data.sos import e_sos_phys, e_sos_walk, frame_shift, h_sa_dense
from dfthc.ops.walk import DFTHCWalk
from dfthc.params import FEMOCO54, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.qpe import (
    WALK_POWERS,
    ControlledWalkPower,
    QPERound,
    WalkPowerSpec,
    _controlled_walk_power,
    _qpe_round,
    energy_from_phase,
    iterative_phase_estimation,
    phase_from_bits,
    physical_energy,
    qpe_program,
    register_walk_power,
    spare_wires_needed,
    walk_power_spec,
)
from dfthc.registers import WalkLayout
from dfthc.testing import walk_sim
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.toy_walk import toy_walk_matrix, toy_walk_power_spec

PLUS = 2**-0.5


def power_call(c, power):
    return lambda: ControlledWalkPower(power, *c.compilable, **c.groups(controlled=True))


def compile_power(c, power, level="user"):
    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        power_call(c, power)()

    return compiled_resources(program, c.num_wires, level=level)


def assert_power_counts(resources, p, power):
    """T, M against costs.controlled_walk_power; A and the plain Toffolis power times the
    controlled walk's (spec 9.23 and 9.24)."""
    expected = costs.controlled_walk_power(p, power)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measured_and_uncomputes == power * costs.measured_and_uncomputes(p, True)
    assert resources.counts["Toffoli"] == power * costs.toffoli_only(p)


# --------------------------------------------------------------------------- ControlledWalkPower


@pytest.mark.compiled
@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_controlled_walk_power_counts(name):
    """T4 at the test presets, power 2 (one loop body): T = 2 T_C(W) = 320 and 484, M = 2 M_W =
    100 and 136, A = 2 A_C(W) = 220 and 362."""
    c = walk_sim.case(name)
    assert_power_counts(compile_power(c, 2), c.params, 2)


@pytest.mark.compiled
@pytest.mark.slow
def test_controlled_walk_power_counts_at_mid_naphthalene():
    """T4 at MID_NAPHTHALENE (nightly), power 2: T = 3,830, M = 1,004, A = 3,360."""
    c = walk_sim.WalkCase.build(MID_NAPHTHALENE, load_factors("naphthalene"))
    assert_power_counts(compile_power(c, 2), c.params, 2)


@pytest.mark.slow
def test_controlled_walk_power_in_three_control_states():
    """T3 on the tape path (strict sparse simulator) at H2_STO3G_TEST, power 2 (nightly: about
    15 s, eight walk steps from a walked state): control |0> returns the whole input, control |1>
    gives two DFTHCWalk steps' whole output, control |+> their superposition; the second step
    starts from the walk's own state (spec 12.4, E-3). Every output has only the index registers
    of A u B nonzero: the walk power returns its work pool, which QPERound's corrections borrow,
    to |0>."""
    c = walk_sim.case("sto3g")
    psi = walk_sim.basis(c, 10)
    plain = walk_sim.program(c, DFTHCWalk, repeat=2)
    powered = walk_sim.operations(c, power_call(c, 2))
    outcomes = walk_sim.schedule("seeded", walk_sim.num_measurements(plain), seed=4)
    identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    applied = walk_sim.set_control(c, walk_sim.simulate(c, plain, walk_sim.input_state(c, psi),
                                                        outcomes))
    expected = {"0": identity, "1": applied,
                "+": walk_sim.superpose((PLUS, identity), (PLUS, applied))}
    for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", [PLUS, PLUS])):
        out = walk_sim.simulate(c, powered, walk_sim.input_state(c, psi, control), outcomes)
        walk_sim.assert_same_state(out, expected[label])
        assert walk_sim.nonzero_registers(c, out) <= set(walk_sim.AB_REGISTERS), label


@pytest.mark.parametrize(("name", "power"), [("sto3g", 1), ("sto3g", 3), ("fails-c3", 2)])
def test_controlled_walk_power_rule_is_valid(name, power):
    """T2 (tape mode; the loop unrolls) at two shapes and three powers; pinned and the only
    rule under its class name."""
    c = walk_sim.case(name)
    assert FIXED_DECOMPS["ControlledWalkPower"] is _controlled_walk_power
    assert list(qp.list_decomps(ControlledWalkPower)) == [_controlled_walk_power]
    op = ControlledWalkPower(power, *c.compilable, **c.groups(controlled=True))
    qp.testing.assert_valid_decomp_rule(op, _controlled_walk_power, check_matrix=False)


@pytest.mark.parametrize(("name", "power"), [("sto3g", 2), ("fails-c3", 1)])
def test_controlled_walk_power_children(name, power):
    """The rule's loop body binds each group to its own register (tape mode, no compile; final
    review m-8): it queues ``power`` ControlledDFTHCWalk operators whose registers are exactly
    the walk power's groups, so a swap of two groups of equal size (system and phase_grad at
    H2_STO3G_TEST) fails here per commit."""
    c = walk_sim.case(name)
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        _controlled_walk_power(power, *c.compilable, **c.groups(controlled=True))
    assert [op.name for op in queue.queue] == ["ControlledDFTHCWalk"] * power
    for op in queue.queue:
        assert {group: list(getattr(op, group)) for group in op.wire_argnames} == c.groups(True)


def test_controlled_walk_power_validates():
    """Spec 9.0: power at least 1; the controlled walk's checks (``ValueError``)."""
    c = walk_sim.case("sto3g")
    with pytest.raises(ValueError, match=r"power must lie in \[1, None\], got 0"):
        ControlledWalkPower(0, *c.compilable, **c.groups(controlled=True))
    with pytest.raises(ValueError, match="'outer' has 1 wires; expected 12"):
        ControlledWalkPower(1, *c.compilable, **(c.groups(controlled=True) | {"outer": [0]}))
    assert ControlledWalkPower.required_wires(5, *c.compilable) == {
        "control": 1, "outer": 12, "inner": 16, "select": 11, "system": 4, "phase_grad": 4,
        "work": 8}


def test_closed_forms_at_femoco():
    """Spec 9.23 and 9.24 at FeMoco-54, without compiling: C(W)^2 is 19,584 Toffolis and 5,048
    measurements; round 0 of a 3-bit estimation adds 1 + 2 correction Toffolis to one step."""
    assert costs.controlled_walk_power(FEMOCO54, 2) == costs.Cost(2 * 9792, 2 * 2524)
    assert costs.qpe_round(FEMOCO54, 0, 3) == costs.Cost(9792 + 1 + 2, 2524)
    assert costs.qpe_round(FEMOCO54, 2, 3) == costs.Cost(4 * 9792, 4 * 2524)


# --------------------------------------------------------------------------- QPERound


def walk_round(layout, p, t, j, earlier_bits):
    spec = walk_power_spec(p, walk_sim.case("sto3g").tables, layout)
    return QPERound(j, t, earlier_bits, spec.op_name, spec.compilable, spec.wire_sizes,
                    layout.qpe_control, layout.qpe_grad, spec.wires)


@pytest.mark.compiled
def test_qpe_round_counts():
    """T4 of round 0 of a 2-bit estimation at H2_STO3G_TEST, worst case (spec 9.24): T =
    T_C(W) + 1 = 161 (the d = 2 correction, unhinted, at its maximum branch), M = M_W = 50 plus
    the measurement that gives m_1, A = A_C(W) + 1. The correction borrows the work pool."""
    p, t = H2_STO3G_TEST, 2
    layout = WalkLayout.from_params(p, qpe_bits=t)
    extra = layout.num_wires

    def program():
        qp.PhaseGradientStatePrep(layout.phase_grad)
        qp.PhaseGradientStatePrep(layout.qpe_grad)
        qp.H(extra)
        m_1 = qp.measure(extra)
        walk_round(layout, p, t, 0, [False, m_1])

    resources = compiled_resources(program, extra + 1)
    expected = costs.qpe_round(p, 0, t)
    assert (resources.toffoli, resources.measurements - 1) == (expected.toffoli,
                                                               expected.measurements)
    assert expected == costs.Cost(161, 50)
    assert resources.measured_and_uncomputes == costs.measured_and_uncomputes(p, True) + 1


def measured(count):
    with qp.queuing.AnnotatedQueue():
        return [qp.measure(wire) for wire in range(100, 100 + count)]


@pytest.mark.parametrize("j", [0, 1, 2])
def test_qpe_round_rule_is_valid_on_the_toy(j):
    """T2 (tape mode) of the three rounds of a 3-bit estimation of the toy walk, the earlier bits
    MeasurementValues (the corrections are conditionals on them); pinned and the only rule."""
    assert FIXED_DECOMPS["QPERound"] is _qpe_round
    assert list(qp.list_decomps(QPERound)) == [_qpe_round]
    bits = [False] * (j + 1) + measured(2 - j)
    spec = toy_walk_power_spec(system=[9], ancilla=[10])
    op = QPERound(j, 3, bits, spec.op_name, spec.compilable, spec.wire_sizes, control=[0],
                  qpe_grad=[1, 2, 3], walk_wires=[9, 10, 4, 5, 6, 7, 8])
    qp.testing.assert_valid_decomp_rule(op, _qpe_round, check_matrix=False)


def test_qpe_round_rule_is_valid_on_the_walk():
    """T2 (tape mode) of round 0 of a 2-bit estimation of the walk at H2_STO3G_TEST."""
    layout = WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=2)
    op = walk_round(layout, H2_STO3G_TEST, 2, 0, [False, *measured(1)])
    qp.testing.assert_valid_decomp_rule(op, _qpe_round, check_matrix=False)


@pytest.mark.compiled
@pytest.mark.slow
def test_pinned_rules_are_chosen(capfd):
    """T6 (nightly: a verbose compile of about 12 s): the listing of the 2-bit QPE program at
    H2_STO3G_TEST names the rounds', the walk powers', the controlled walk's and the
    correction's pinned rules. The per-commit listing is the toy QPE program's
    (tests/integration/test_qpe_toy.py), and the exact counts below are the per-commit count
    signature of these rules."""
    program, num_wires = qpe_program(H2_STO3G_TEST, walk_sim.case("sto3g").tables, 2)
    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    assert {"_qpe_round", "_controlled_walk_power", "_controlled_walk",
            "_phase_gradient_rotation"} <= set(chosen_rules(captured.out + captured.err))


@pytest.mark.compiled
def test_nothing_survives_at_device_level():
    """T5 at level="device" (L-20): no dfthc operator survives in the 2-bit QPE program."""
    program, num_wires = qpe_program(H2_STO3G_TEST, walk_sim.case("sto3g").tables, 2)
    compiled_resources(program, num_wires, level="device")


@pytest.mark.parametrize(("change", "match"), [
    ({"j": 3}, r"j must lie in \[0, 2\], got 3"),
    ({"walk_op": "DFTHCWalk"}, "walk_op must be one of"),
    ({"walk_compilable": []}, "walk_compilable must be a tuple"),
    ({"walk_wire_sizes": (2,), "walk_wires": [9, 10, 4, 5, 6, 7, 8]},
     r"walk_wire_sizes must be ToyControlledWalkPower's required_wires after control, \(1, 1\)"),
    ({"earlier_bits": [False, False]}, r"earlier_bits must have shape \(3,\); got \(2,\)"),
    ({"qpe_grad": [1, 2]}, "'qpe_grad' has 2 wires; expected 3"),
    ({"walk_wires": [9, 10, 4, 5, 6, 7]}, "'walk_wires' has 6 wires; expected 7"),
], ids=["j", "walk_op", "walk_compilable", "walk_wire_sizes", "earlier_bits", "qpe_grad",
        "spare"])
def test_qpe_round_validates(change, match):
    """Spec 9.0: QPERound checks its compilable arguments, the shape (t,) of earlier_bits and
    every register size (``ValueError``); the toy walk lends no scratch group, so walk_wires
    holds its two wires and the corrections' five spare wires."""
    spec = toy_walk_power_spec(system=[9], ancilla=[10])
    arguments = {"j": 0, "num_bits": 3, "earlier_bits": [False] * 3, "walk_op": spec.op_name,
                 "walk_compilable": spec.compilable, "walk_wire_sizes": spec.wire_sizes,
                 "control": [0], "qpe_grad": [1, 2, 3], "walk_wires": [9, 10, 4, 5, 6, 7, 8]
                 } | change
    with pytest.raises(ValueError, match=match):
        QPERound(**arguments)


def malformed_walk_compilable(change):
    """ControlledWalkPower's compilable arguments at H2_STO3G_TEST after ``power``, changed."""
    c = walk_sim.case("sto3g")
    params, tables = c.compilable
    if change == "angle":
        tail = [list(row) for row in tables[6]]
        tail[0][0] = 2 ** (c.params.b_rot - 1)
        return params, (*tables[:6], tuple(map(tuple, tail)), tables[7])
    if change == "fields":
        return params, tables[:7]
    return (params,)


@pytest.mark.parametrize(("op_name", "change", "match"), [
    ("ControlledWalkPower", "angle", r"angles must lie in \[0, 2\*\*\(b_rot - 1\)\)"),
    ("ControlledWalkPower", "fields", "WalkTables needs 8 fields; got 7"),
    ("ControlledWalkPower", "arity", "walk_compilable must hold ControlledWalkPower's compilable "
                                     r"arguments after power, \('params', 'tables'\); got 1"),
    ("ToyControlledWalkPower", "arity", "walk_compilable must hold ToyControlledWalkPower's "
                                        r"compilable arguments after power, \(\); got 1"),
], ids=["tail-angle", "table-fields", "arity", "toy-arity"])
def test_qpe_round_validates_the_walk_powers_arguments(op_name, change, match):
    """Spec 9.0 (final review I-2): QPERound checks the walk power's compilable arguments at
    construction, through the walk power's validate_compilable, before it computes the wire
    sizes; malformed tables raise ValueError there, not when the rule builds the walk power."""
    layout = WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=2)
    if op_name == "ControlledWalkPower":
        spec = walk_power_spec(H2_STO3G_TEST, walk_sim.case("sto3g").tables, layout)
        compilable, wires = malformed_walk_compilable(change), spec.wires
    else:
        spec = toy_walk_power_spec(system=[60], ancilla=[61])
        compilable, wires = (5,), [60, 61, 62, 63, 64]
    with pytest.raises(ValueError, match=match):
        QPERound(0, 2, [False, False], op_name, compilable, spec.wire_sizes, layout.qpe_control,
                 layout.qpe_grad, wires)


def test_register_walk_power_checks_the_class():
    """A walk power registers once, its scratch_argname is None or its last wire group (the
    group QPERound's corrections borrow), and it has a validate_compilable; a class without
    either attribute is refused."""
    assert WALK_POWERS["ControlledWalkPower"].scratch_argname == "work"
    assert WALK_POWERS["ToyControlledWalkPower"].scratch_argname is None
    with pytest.raises(ValueError, match="ControlledWalkPower is already a registered"):
        register_walk_power(ControlledWalkPower)
    wires = {"wire_argnames": ("control", "system", "work")}
    with pytest.raises(ValueError, match=r"scratch_argname must be None or its last wire group "
                                         r"'work'; got 'system'"):
        register_walk_power(type("Middle", (), wires | {"scratch_argname": "system"}))
    with pytest.raises(ValueError, match="got ''"):
        register_walk_power(type("Undeclared", (), wires))
    with pytest.raises(ValueError, match="Unchecked needs a static validate_compilable"):
        register_walk_power(type("Unchecked", (), wires | {"scratch_argname": None}))
    assert not {"Middle", "Undeclared", "Unchecked"} & set(WALK_POWERS)


def test_spare_wires_needed():
    """The corrections' 2t - 1 wires less the walk power's scratch group: the toy lends none
    (1, 3, 5 spare wires for t = 1, 2, 3); ControlledWalkPower lends its work pool, 8 wires at
    H2_STO3G_TEST (spare wires only from t = 5) and 44 at FeMoco-54 (only from t = 23).
    QPERound's walk_wires are the walk's groups plus these."""
    toy = toy_walk_power_spec(system=[0], ancilla=[1])
    assert [spare_wires_needed(toy.op_name, toy.wire_sizes, t) for t in (1, 2, 3)] == [1, 3, 5]
    sto3g = walk_power_spec(H2_STO3G_TEST, walk_sim.case("sto3g").tables,
                            WalkLayout.from_params(H2_STO3G_TEST))
    assert [spare_wires_needed(sto3g.op_name, sto3g.wire_sizes, t) for t in range(1, 7)] == [
        0, 0, 0, 0, 1, 3]
    assert [QPERound.required_wires(0, t, sto3g.op_name, sto3g.compilable, sto3g.wire_sizes)[
        "walk_wires"] for t in (4, 5)] == [55, 56]
    femoco = WalkLayout.from_params(FEMOCO54)
    sizes = tuple(len(getattr(femoco, name)) for name in ControlledWalkPower.wire_argnames[1:])
    assert sizes[-1] == WalkLayout.work_size(FEMOCO54) == 44 and sum(sizes) == 1102
    assert [spare_wires_needed("ControlledWalkPower", sizes, t) for t in (3, 20, 22, 23, 25)] == [
        0, 0, 0, 1, 5]


# --------------------------------------------------------------------------- driver and wiring


@pytest.mark.parametrize(("arguments", "match"), [
    ((0, [1, 2, 3], [4, 5, 6, 7, 8]), r"num_bits must lie in \[1, None\], got 0"),
    ((3, [1, 2], [4, 5, 6, 7, 8]), r"num_bits = 3 exceeds len\(qpe_grad\) = 2"),
    ((3, [1, 2, 3], [4, 5, 6, 7]),
     "spare_wires needs 5 wires for num_bits = 3 with ToyControlledWalkPower; got 4"),
    ((3, [1, 2, 3], [4, 5, 6, 7, 9]), "must be distinct"),
], ids=["zero-bits", "qpe_grad", "spare", "aliased"])
def test_driver_validates(arguments, match):
    """Spec 11.1: the driver raises ValueError if num_bits > len(qpe_grad), for no bits, for too
    few spare wires (the toy's corrections need 2t - 1) and for a wire bound twice (here the
    toy's system wire as a spare wire), before it queues anything."""
    num_bits, qpe_grad, spare = arguments
    with qp.queuing.AnnotatedQueue() as queue, pytest.raises(ValueError, match=match):
        iterative_phase_estimation(lambda: toy_walk_power_spec([9], [10]), [0], num_bits,
                                   qpe_grad, spare)
    assert not queue.queue


def test_walk_power_spec_flattens_the_groups_in_wire_argnames_order():
    """The flat walk_wires of QPERound hold ControlledWalkPower's groups in its wire_argnames
    order (outer, inner, select, system, phase_grad, work), not the layout's order, which starts
    with system; split by wire_sizes they are the layout's groups again."""
    c = walk_sim.case("sto3g")
    layout = WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=2)
    spec = walk_power_spec(c.params, c.tables, layout)
    assert isinstance(spec, WalkPowerSpec) and WALK_POWERS[spec.op_name] is ControlledWalkPower
    assert spec.compilable == c.compilable
    assert spec.wire_sizes == (12, 16, 11, 4, 4, 8)
    wires, start = list(spec.wires), 0
    for name, size in zip(ControlledWalkPower.wire_argnames[1:], spec.wire_sizes, strict=True):
        assert wires[start:start + size] == list(getattr(layout, name)), name
        start += size
    assert wires[:4] != list(layout.system) and start == len(wires) == 55


@pytest.mark.parametrize(("t", "spare"), [(3, []), (5, [61])])
def test_qpe_program_wires(t, spare):
    """qpe_program at H2_STO3G_TEST: the layout with qpe_bits = t (control 55, qpe_grad from
    56), then the spare wires: none at t = 3, whose 5 correction wires fit the 8-wire work pool
    (47 to 54), one (61) at t = 5, whose 9 do not. Every round binds each wire once, the same
    registers in every round: control, qpe_grad and walk_wires (the walk's 55 wires, then the
    spare ones). Round 0's corrections borrow the pool from its first wire on, into the spare
    wire: for d = l + 1, the scratch wires corrections[:d] and the work wires
    corrections[d:2d - 1]."""
    c = walk_sim.case("sto3g")
    program, num_wires = qpe_program(c.params, c.tables, t)
    layout = WalkLayout.from_params(c.params, qpe_bits=t)
    assert num_wires == layout.num_wires + len(spare) == 55 + 1 + t + len(spare)
    spec = walk_power_spec(c.params, c.tables, layout)
    walk_wires = [*spec.wires, *spare]
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        program()
    rounds = [op for op in queue.queue if isinstance(op, QPERound)]
    assert [op.hyperparameters["j"] for op in rounds] == list(reversed(range(t)))
    for op in rounds:
        assert list(op.control) == [55] and list(op.qpe_grad) == list(range(56, 56 + t))
        assert list(op.walk_wires) == walk_wires and sorted(walk_wires[:55]) == list(range(55))
    corrections = [*layout.work, *spare][:2 * t - 1]
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as inner:
        _qpe_round(0, t, [False, *measured(t - 1)], spec.op_name, spec.compilable,
                   spec.wire_sizes, [55], list(range(56, 56 + t)), walk_wires)
    rotations = [op.base for op in inner.queue if isinstance(op, qp.ops.Conditional)]
    assert len(rotations) == t - 1
    for d, rotation in enumerate(rotations, start=2):
        assert list(rotation.target) == [55] and list(rotation.phase_grad) == list(
            range(56, 56 + d))
        assert list(rotation.scratch) == corrections[:d]
        assert list(rotation.work) == corrections[d:2 * d - 1]


# --------------------------------------------------------------------------- energies


def test_phase_from_bits():
    """phi = sum_j m_j 2^-(j+1), the bits in the driver's order m_{t-1}, ..., m_0 (spec 11.1)."""
    assert phase_from_bits([1, 0, 0]) == 1 / 8
    assert phase_from_bits([0, 0, 1]) == 1 / 2
    assert phase_from_bits([1, 1, 1]) == 7 / 8
    assert phase_from_bits([1]) == 1 / 2 and phase_from_bits([0, 1]) == 1 / 2


def test_energy_from_phase_inverts_the_walk_phases():
    """Spec 11.1: for every eigenvalue E of H~_SA at H2_STO3G_TEST, both walk phases +-theta/2 pi
    with cos theta = E / Lambda~ - 1 give E back (shift 1); for the toy walk (shift 0, Lambda 1)
    the phases 1/8, 7/8 give 1/sqrt 2 and 3/8, 5/8 give -1/sqrt 2, the eigenvalues of
    (Z + X)/2 (spec 11.4)."""
    c = walk_sim.case("sto3g")
    h_tilde, lambda_tilde = encoded_block(c.params, c.tables, c.factors)
    for energy in np.linalg.eigvalsh(h_tilde):
        theta = np.arccos(energy / lambda_tilde - 1)
        for phi in (theta / (2 * np.pi), 1 - theta / (2 * np.pi)):
            assert np.isclose(energy_from_phase(phi, lambda_tilde), energy, rtol=0, atol=1e-12)
    toy = np.sort(np.linalg.eigvalsh(toy_walk_matrix()[np.ix_([0, 2], [0, 2])]))
    assert np.allclose(toy, [-2**-0.5, 2**-0.5], rtol=0, atol=1e-12)
    for phi, energy in ((1 / 8, toy[1]), (7 / 8, toy[1]), (3 / 8, toy[0]), (5 / 8, toy[0])):
        assert np.isclose(energy_from_phase(phi, 1.0, shift=0.0), energy, rtol=0, atol=1e-12)


def test_physical_energy_is_the_physical_frame():
    """Spec 3.2.1, 3.4 and 11.1 for h2_sto3g_R2_B2_C1 (exactly factorized): the lowest
    eigenvalue of H_SA on the two-electron sector plus E_SoS^phys is the physical ground energy
    e_upper, and plus core_energy the total; the walk frame's offset would be off by
    Delta_eta = -1.3209547."""
    system = "h2_sto3g_R2_B2_C1"
    factors, summary = load_factors(system), load_summary(system)
    keep = np.flatnonzero(np.diag(sector_projector(factors.N, factors.eta)).real)
    e_sa = np.linalg.eigvalsh(h_sa_dense(factors)[np.ix_(keep, keep)]).min()
    assert np.isclose(physical_energy(e_sa, e_sos_phys(summary)), summary["e_upper"], rtol=0,
                      atol=1e-10)
    core = float(load_integrals(system)["core_energy"])
    assert np.isclose(physical_energy(e_sa, e_sos_phys(summary), core),
                      summary["e_upper"] + core, rtol=0, atol=1e-10)
    walk_frame = e_sa + e_sos_walk(factors)
    assert np.isclose(physical_energy(e_sa, e_sos_phys(summary)) - walk_frame,
                      frame_shift(factors, summary), rtol=0, atol=1e-12)
    assert np.isclose(frame_shift(factors, summary), -1.3209547, rtol=0, atol=1e-7)
