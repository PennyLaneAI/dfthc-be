"""InnerSelect and InnerSelectDagger (spec 9.18): the action against the encoded model on the tape
path, the inverse under forced schedules, the order and wire passing, counts, rule selection."""

import math
import re
import warnings

import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, compiled_resources
from dfthc.data.angles import vector_from_angles
from dfthc.data.encoded import encoded_factors
from dfthc.data.factors import load_factors
from dfthc.data.fermion import annihilation, rotated_majorana
from dfthc.data.tables import WalkTables
from dfthc.ops.angle_data import AngleErase, SFAngleLoad, TailAngleLoad
from dfthc.ops.givens import GivensLadder
from dfthc.ops.select import InnerSelect, InnerSelectDagger, _inner_select, _inner_select_dagger
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator, SparseState
from dfthc.testing.tape import decompose_for_simulation

CLASSES = {"select": (InnerSelect, _inner_select), "dagger": (InnerSelectDagger,
                                                              _inner_select_dagger)}


def vendored(system, p):
    """The preset's angle tables and its encoded factors (spec 8.5)."""
    factors = load_factors(system)
    tables = WalkTables.from_factors(factors, p)
    return tables.tail_angles, tables.sf_angles, encoded_factors(p, tables, factors)


def random_tables(p, seed):
    """Random angle tables of p's shape, every angle below 2**(b_rot - 1)."""
    rng = np.random.default_rng(seed)
    bound = 2 ** (p.b_rot - 1)
    tail = tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.N))
    sf = tuple(tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.B))
               for _ in range(p.R))
    return tail, sf, None


# C = 3: r = x_o // 3 is no bit field of x_o, and the tail rows [6, 9) start off a power of two;
# N = 3 gives the ladder two rungs (their order and the register order matter), N+ = 2 a Q1 row.
RANDOM_C3 = WalkParams(N=3, R=2, B=2, C=3, n_plus=2, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
# R = 1: r has no wires and the SF address is b alone; b_rot = 4 so that the one-hot fits.
RANDOM_R1 = WalkParams(N=2, R=1, B=2, C=2, n_plus=2, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
SHAPES = {
    "sto3g": (H2_STO3G_TEST, *vendored("h2_sto3g_R2_B2_C1", H2_STO3G_TEST)),
    "631g": (H2_631G_TEST, *vendored("h2_631g_R2_B2_C2", H2_631G_TEST)),
    "random-C3": (RANDOM_C3, *random_tables(RANDOM_C3, seed=5)),
    "random-R1": (RANDOM_R1, *random_tables(RANDOM_R1, seed=0)),
}


def compilable(name, ob=None):
    p, tail, sf, _ = SHAPES[name]
    return (tail, sf, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b if ob is None else ob)


def layout(args):
    """Consecutive wires for every register, in the order of required_wires."""
    wires, start = {}, 0
    for name, size in InnerSelect.required_wires(*args).items():
        wires[name], start = list(range(start, start + size)), start + size
    return wires, start


# --------------------------------------------------------------------------- the reference


def directions(name):
    """u~ of every SF entry (r, b) and tail row j: the encoded model's unit vectors at the
    vendored presets (encoded_factors), vector_from_angles of the tables otherwise (spec 6.3)."""
    p, tail, sf, enc = SHAPES[name]
    if enc is not None:
        return ({(r, b): enc.U[r, b] for r in range(p.R) for b in range(p.B)},
                [*enc.u_plus, *enc.u_minus])
    return ({(r, b): vector_from_angles(sf[r][b], p.b_rot) for r in range(p.R)
             for b in range(p.B)}, [vector_from_angles(row, p.b_rot) for row in tail])


def branch_classes(p):
    """(x_o, succ_x_o, sigma_1, b, succ_b, r) of the branches the T3 runs, as the prepares leave
    them: every SF row with every b <= B (both flags set, r = x_o // C), a failed b (succ_b = 0)
    and sigma_1 = 1 on SF rows, every tail row with sigma_1 = 0 and 1, a D1 row with succ_b = 0,
    and failed x_o (succ_x_o = 0) on an SF row, a tail row and, if it exists, x_o = X_o. The
    inner read loads a zero entry at a failed b, so the alias swap leaves (b, r) = (0, 0) there:
    the SF load then writes sf_angles[0][0] and b_neq_B = 1, and only the Majorana gate's succ_b
    control keeps the branch the identity (spec 8.5, ledger row 2D.2). That class sits on the
    last SF row, whose own rank is not 0; a failed b at b = 2**w_b - 1 (nothing loaded) stays."""
    classes = [(x, 1, 0, b, 1, x // p.C) for x in range(p.RC) for b in range(p.B + 1)]
    classes += [(0, 1, 0, 2**p.w_b - 1, 0, 0), (p.RC - 1, 1, 0, 0, 0, 0),
                (p.RC - 1, 1, 1, 0, 1, (p.RC - 1) // p.C)]
    classes += [(p.RC + j, 1, s, 0, 1, 0) for j in range(p.N) for s in (0, 1)]
    classes += [(p.RC, 1, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (p.RC, 0, 1, 1, 1, 0)]
    if p.X_o < 2**p.w_x:
        classes.append((p.X_o, 0, 0, 0, 1, 0))
    return classes


def flavour_actions(name, branch):
    """The select's action on psi_up (x) psi_dn for sigma_0 = 0 and 1, from the encoded model.

    SF with b < B and both flags: i gamma~_0 gamma~_1 on the register sigma_0 selects (psi_dn for
    0, which is spin 1 of dfthc.data.fermion). D1/Q1 with succ_x_o: on the register sigma_1
    selects, with a = sum_j u~_j a_j, D1 gives a + a^dag and a - a^dag, Q1 a + a^dag and
    a^dag - a, whose <+| projection on sigma_0 is a (D1) and a^dag (Q1). The identity elsewhere."""
    p = SHAPES[name][0]
    sf_u, tail_u = directions(name)
    x_o, succ_x_o, sigma_1, b, succ_b, r = branch
    identity = np.eye(4**p.N)
    if not succ_x_o or (x_o < p.RC and not (succ_b and b < p.B)):
        return [identity, identity]
    if x_o < p.RC:
        u = sf_u[r, b]
        return [1j * rotated_majorana(u, 0, 1 - f, p.N) @ rotated_majorana(u, 1, 1 - f, p.N)
                for f in (0, 1)]
    j, spin = x_o - p.RC, 1 - sigma_1
    lower = sum(tail_u[j][k] * annihilation(k, spin, p.N) for k in range(p.N))
    upper = lower.conj().T
    return [lower + upper, lower - upper] if j < p.n_plus else [lower + upper, upper - lower]


def key_values(num_wires, registers, values):
    """Basis keys of the simulator (one 64-bit word for at most 64 wires, wire 0 the top bit)
    with ``values[name]`` (arrays of ints) big-endian in ``registers[name]``; registers not named
    are 0."""
    assert num_wires <= 64
    total = np.zeros(len(next(iter(values.values()))), dtype=np.uint64)
    one = np.uint64(1)
    for name, column in values.items():
        wires, column = registers[name], np.asarray(column, dtype=np.uint64)
        for i, wire in enumerate(wires):
            total |= ((column >> np.uint64(len(wires) - 1 - i)) & one) << np.uint64(63 - wire)
    return total[:, None]


def gradient_state(b_rot):
    """The phase-gradient state the simulator prepares on b_rot wires."""
    return SparseSimulator(b_rot).run([qp.PhaseGradientStatePrep(range(b_rot))]).to_dense()


def with_gradient(wires, keys, amps, b_rot):
    """``keys``, ``amps`` (phase_grad in |0>) times the phase-gradient state on phase_grad."""
    gradient = gradient_state(b_rot)
    g = np.arange(2**b_rot)
    shift = key_values(64, {"g": wires["phase_grad"]}, {"g": g})[:, 0]
    return ((keys[:, 0][:, None] | shift[None, :]).ravel()[:, None],
            (amps[:, None] * gradient[None, :]).ravel())


def assert_same_state(state, keys, amps):
    """The simulated state equals the terms (keys, amps) exactly: every amplitude to 1e-12, every
    other basis state (dirty ancillas, changed index registers) zero."""
    all_keys = np.concatenate([state.keys[:, 0], keys[:, 0]])
    unique, inverse = np.unique(all_keys, return_inverse=True)
    diff = np.concatenate([state.amps, -amps])
    total = (np.bincount(inverse.ravel(), diff.real, len(unique))
             + 1j * np.bincount(inverse.ravel(), diff.imag, len(unique)))
    assert np.allclose(total, 0, rtol=0, atol=1e-12), np.abs(total).max()


def schedules(num, rng):
    """The all-ones, the alternating and a seeded forced schedule of ``num`` outcomes (an
    all-ones schedule sees only each address's mask parity, ledger row 2D.2)."""
    return {"ones": [1] * num, "alternating": [i % 2 for i in range(num)],
            "seeded": rng.integers(0, 2, num).tolist()}


def simulate(operations, num_wires, keys, amps, schedule):
    simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
    state = simulator.run(operations, initial_state=SparseState(num_wires, keys, amps))
    assert simulator.policy.position == len(schedule)
    return state


# --------------------------------------------------------------------------- semantics


@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize("name", ["sto3g", "random-C3",
                                  pytest.param("631g", marks=pytest.mark.slow)])
def test_action_against_the_encoded_model(name, kind):
    """T3 on the tape path (spec 7.4 rule 9), as the sparse simulator runs it: on the branch
    classes of branch_classes, with sigma_0 in |+>, a random system state per class and the
    phase-gradient state on phase_grad, every
    term equals the encoded model's generator piece (flavour_actions; for InnerSelectDagger its
    adjoint), every ancilla is |0> and the index registers are unchanged, under the all-ones, the
    alternating and a seeded forced schedule of the erase (strict mode). The simulator asserts
    every AND precondition, which lightning cannot see (ledger row 2E.5). sto3g is the H2 test
    shape; random-C3 has a ladder of two rungs, a Q1 row and C = 3 (rank selection); 631g (N = 4,
    three D1 rows and a Q1 row, about 200,000 terms) runs in the nightly tier."""
    p = SHAPES[name][0]
    cls, _ = CLASSES[kind]
    args = compilable(name)
    wires, num_wires = layout(args)
    rng = np.random.default_rng(7)
    classes = branch_classes(p)
    dim = 4**p.N
    states = rng.normal(size=(len(classes), dim)) + 1j * rng.normal(size=(len(classes), dim))
    states /= np.linalg.norm(states) * math.sqrt(2)  # the 1 / sqrt 2 of sigma_0 = |+>
    rows = [(k, f, x) for k in range(len(classes)) for f in (0, 1) for x in range(dim)]
    k, f, x = (np.array(column) for column in zip(*rows, strict=True))
    index = np.array(classes)[k]
    values = dict(zip(("x_o", "succ_x_o", "sigma_1", "b", "succ_b", "r"), index.T, strict=True))
    registers = wires | {"system": wires["psi_up"] + wires["psi_dn"]}
    keys = key_values(num_wires, registers, values | {"sigma_0": f, "system": x})
    actions = [flavour_actions(name, branch) for branch in classes]
    if kind == "dagger":
        actions = [[action.conj().T for action in pair] for pair in actions]
    outputs = np.array([[actions[c][s] @ states[c] for s in (0, 1)] for c in range(len(classes))])
    initial = with_gradient(wires, keys, states[k, x], p.b_rot)
    expected = with_gradient(wires, keys, outputs[k, f, x], p.b_rot)

    operations = decompose_for_simulation(lambda: cls(*args, **wires)).operations
    for schedule in schedules((p.N - 1) * (p.b_rot - 1) + 3, rng).values():
        assert_same_state(simulate(operations, num_wires, *initial, schedule), *expected)


@pytest.mark.parametrize(("name", "num_system"), [("sto3g", 16), ("random-C3", 2)])
def test_dagger_inverts_the_select_on_every_branch(name, num_system):
    """InnerSelectDagger after InnerSelect is the identity on every branch, the ones no prepare
    reaches included (failed x_o >= X_o, r != x_o // C, sigma_0 not in |+>):
    from a random complex state over every value of the index registers and sigma_0, times
    num_system random system basis states, the sparse simulator returns that state exactly with
    every ancilla in |0>, under the alternating and a seeded schedule of both erases."""
    p = SHAPES[name][0]
    args = compilable(name)
    wires, num_wires = layout(args)
    rng = np.random.default_rng(11)
    index = ("x_o", "succ_x_o", "sigma_1", "b", "succ_b", "sigma_0", "r")
    sizes = [len(wires[reg]) for reg in index]
    system = rng.choice(4**p.N, num_system, replace=False)
    grid = np.array(np.meshgrid(*[np.arange(2**n) for n in sizes], system,
                                indexing="ij")).reshape(len(sizes) + 1, -1)
    registers = wires | {"system": wires["psi_up"] + wires["psi_dn"]}
    keys = key_values(num_wires, registers, dict(zip((*index, "system"), grid, strict=True)))
    amps = rng.normal(size=len(keys)) + 1j * rng.normal(size=len(keys))
    initial = with_gradient(wires, keys, amps / np.linalg.norm(amps), p.b_rot)

    def program():
        InnerSelect(*args, **wires)
        InnerSelectDagger(*args, **wires)

    operations = decompose_for_simulation(program).operations
    outcomes = schedules(2 * ((p.N - 1) * (p.b_rot - 1) + 3), rng)
    for schedule in (outcomes["alternating"], outcomes["seeded"]):
        assert_same_state(simulate(operations, num_wires, *initial, schedule), *initial)


def children(rule, args, wires):
    """The rule's operators in tape mode, with their registers."""
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(*args, **wires)
    return list(queue.queue)


def registers(op):
    op = getattr(op, "base", op)
    return {name: list(getattr(op, name)) for name in op.wire_argnames}


@pytest.mark.parametrize("name", ["random-C3", "random-R1"])
def test_children_order_and_wires(name):
    """Spec 9.18's order, with each child on exactly its registers (spec 9.0): the loads, the
    spin multiplexer and swap, the ladder, the Majorana gate on psi_dn[0] with flavour sigma_0
    (InnerSelectDagger: Adjoint(MajoranaGate), the instance form), the explicit
    GivensLadderDagger (never an adjointed ladder), the swap, the multiplexer, the erase; each
    child's work is the prefix of exactly its required size. Both selects load first and erase
    last, and their children differ only in the Majorana gate, so they share every other target
    (identical compilable arguments and register sizes)."""
    p, tail, sf, _ = SHAPES[name]
    args = compilable(name)
    w, _ = layout(args)
    tail_work = TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot)["work"]
    sf_work = SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot)["work"]
    erase_work = AngleErase.required_wires(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot,
                                           p.w_b)["work"]
    mux = {"G0": w["G0"], "sigma_0": w["sigma_0"], "sigma_1": w["sigma_1"], "sg": w["sg"]}
    swap = {"sg": w["sg"], "psi_up": w["psi_up"], "psi_dn": w["psi_dn"]}
    ladder = {"system": w["psi_dn"], "angles": w["angles"], "phase_grad": w["phase_grad"],
              "work": w["work"][:p.b_rot - 1]}
    majorana = {"succ_x_o": w["succ_x_o"], "succ_b": w["succ_b"], "G0": w["G0"], "G1": w["G1"],
                "b_neq_B": w["b_neq_B"], "flavour": w["sigma_0"], "target": w["psi_dn"][:1],
                "work": w["work"][:2]}
    expected = [
        ("TailAngleLoad", {"x_o": w["x_o"], "angles": w["angles"], "G0": w["G0"], "G1": w["G1"],
                           "work": w["work"][:tail_work]}),
        ("SFAngleLoad", {"r": w["r"], "b": w["b"], "G0": w["G0"], "angles": w["angles"],
                         "b_neq_B": w["b_neq_B"], "work": w["work"][:sf_work]}),
        ("SpinMux", mux), ("SpinSwap", swap), ("GivensLadder", ladder), ("MajoranaGate", majorana),
        ("GivensLadderDagger", ladder), ("SpinSwap", swap), ("SpinMux", mux),
        ("AngleErase", {"angles": w["angles"], "b_neq_B": w["b_neq_B"], "G0": w["G0"],
                        "G1": w["G1"], "x_o": w["x_o"], "r": w["r"], "b": w["b"],
                        "work": w["work"][:erase_work]}),
    ]
    forward, backward = (children(rule, args, w) for _, rule in CLASSES.values())
    assert [(op.name, registers(op)) for op in forward] == expected
    assert [op.name for op in backward] == [
        "Adjoint(MajoranaGate)" if name == "MajoranaGate" else name for name, _ in expected]
    assert [registers(op) for op in backward] == [regs for _, regs in expected]
    assert [op.hyperparameters for op in forward] == [
        getattr(op, "base", op).hyperparameters for op in backward]
    assert forward[0].hyperparameters == {"tail_angles": tail, "n_plus": p.n_plus, "RC": p.RC,
                                          "X_o": p.X_o, "b_rot": p.b_rot}
    assert forward[-1].hyperparameters == {
        "tail_angles": tail, "sf_angles": sf, "n_plus": p.n_plus, "N": p.N, "R": p.R, "B": p.B,
        "C": p.C, "b_rot": p.b_rot, "sf_onehot_bits": p.w_b}


# --------------------------------------------------------------------------- resources


def expected_cost(p, ob):
    """costs.inner_select with the erase's SF fix-up at one-hot width ob (spec 9.17); the walk's
    ob = w_b gives costs.inner_select itself."""
    def sf(bits):
        return costs.phase_fixup(AngleErase.sf_fixup_high_ranges(p.R, p.B, bits), bits, True)

    walk = costs.inner_select(p)
    return costs.Cost(walk.toffoli + sf(ob).toffoli - sf(p.w_b).toffoli, walk.measurements)


def assert_counts(resources, p, ob):
    """T4 and the T6 signature: T and M exact; the plain Toffolis are the 2N + 5 of the spin
    operators and the Majorana gate, so A = T - (2N + 5); the erase's hinted conditionals count
    their expectation (nothing else in the select emits CZ or Z): R B conditional CZ (ob >= 1) or
    Z (ob = 0) on the SF fix-up and N conditional Z on the tail fix-up, summed per gate type and
    rounded up once (L-17)."""
    expected = expected_cost(p, ob)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.counts["Toffoli"] == 2 * p.N + 5
    assert resources.measured_and_uncomputes == resources.toffoli - (2 * p.N + 5)
    hinted = ({"CZ": math.ceil(p.R * p.B / 2), "PauliZ": math.ceil(p.N / 2)} if ob
              else {"CZ": 0, "PauliZ": math.ceil((p.R * p.B + p.N) / 2)})
    assert {gate: resources.counts.get(gate, 0) for gate in hinted} == hinted


@pytest.mark.compiled
@pytest.mark.parametrize(("name", "ob"), [("sto3g", None), ("631g", None), ("random-C3", 0),
                                          ("random-R1", None)],
                         ids=["sto3g", "631g", "C3-ob0", "R1"])
def test_counts(name, ob):
    """T4 compiled, each class alone: costs.inner_select at both test presets (T = 33 and 65,
    M = 6 and 12), the ob = 0 fallback of the erase's SF fix-up at random-C3, and R = 1."""
    p = SHAPES[name][0]
    args = compilable(name, ob)
    wires, num_wires = layout(args)
    for cls, _ in CLASSES.values():
        assert_counts(compiled_resources(lambda cls=cls: cls(*args, **wires), num_wires), p,
                      p.w_b if ob is None else ob)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("kind", CLASSES)
def test_counts_at_mid_naphthalene(kind):
    """T4 at MID_NAPHTHALENE (naphthalene tables, nightly tier): T = 625, M = 129, A = 600."""
    p = MID_NAPHTHALENE
    tail, sf, _ = vendored("naphthalene", p)
    args = (tail, sf, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    wires, num_wires = layout(args)
    cls, _ = CLASSES[kind]
    assert_counts(compiled_resources(lambda: cls(*args, **wires), num_wires), p, p.w_b)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 (spec 9.24): 61 + 291 + 2 + 108 + 2,968 + 5 + 102 = 3,537 Toffolis and
    M = 745 for each select, A = 3,537 - 113 = 3,424; the select's work is the ladder's 14 wires,
    within the walk's pool of 44."""
    p = FEMOCO54
    parts = [costs.tail_angle_load(p), costs.sf_angle_load(p), 2 * costs.spin_mux(),
             2 * costs.spin_swap(p), 2 * costs.givens_ladder(p), costs.majorana_gate(),
             costs.angle_erase(p)]
    assert [part.toffoli for part in parts] == [61, 291, 2, 108, 2968, 5, 102]
    assert costs.inner_select(p) == costs.Cost(3537, 745)
    assert costs.inner_select(p).toffoli - (2 * p.N + 5) == 3424
    tail = tuple((0,) * 53 for _ in range(54))
    sf = tuple(tuple((0,) * 53 for _ in range(27)) for _ in range(10))
    sizes = InnerSelect.required_wires(tail, sf, 54, 10, 27, 27, 27, 15, 5)
    assert sizes == {"x_o": 9, "succ_x_o": 1, "sigma_1": 1, "b": 5, "succ_b": 1, "sigma_0": 1,
                     "r": 4, "angles": 742, "G0": 1, "G1": 1, "b_neq_B": 1, "sg": 1,
                     "psi_up": 54, "psi_dn": 54, "phase_grad": 15, "work": 14}
    assert GivensLadder.required_wires(15, 54)["work"] == 14


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6 and T5 at H2_STO3G_TEST: the verbose listing names both pinned rules; at
    level="device" no dfthc operator survives in a program holding both selects (L-20)."""
    args = compilable("sto3g")
    wires, num_wires = layout(args)

    def both():
        InnerSelect(*args, **wires)
        InnerSelectDagger(*args, **wires)

    qp.analyze(verbose_program(both, num_wires), level="user")()
    captured = capfd.readouterr()
    assert {"_inner_select", "_inner_select_dagger"} <= set(chosen_rules(captured.out
                                                                         + captured.err))
    compiled_resources(both, num_wires, level="device")


@pytest.mark.compiled
@pytest.mark.parametrize("cls", [InnerSelect, InnerSelectDagger],
                         ids=["select", "dagger"])
@pytest.mark.parametrize("wrap", [qp.adjoint, lambda op: qp.ctrl(op, control=40)],
                         ids=["adjoint", "control"])
def test_no_adjoint_or_controlled_path(wrap, cls):
    """Spec 7.4 rules 5 and 6: the adjoint of the select is the class InnerSelectDagger, and
    neither class is ever adjointed or controlled. Wrapped anyway, Catalyst drops the measuring
    candidates with the expected warning (rule 12), and the compile fails loudly."""
    args = compilable("sto3g")
    wires, _ = layout(args)
    key = f"Adjoint({cls.__name__})" if wrap is qp.adjoint else f"C({cls.__name__})"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(lambda: wrap(cls(*args, **wires)), num_wires=41)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the ")]
    named = [message for message in skipped if classify_rule_warning(message) != "noise"]
    assert named and {classify_rule_warning(message) for message in named} == {"expected"}


@pytest.mark.parametrize(("name", "ob"), [("sto3g", None), ("631g", 1), ("random-C3", 0),
                                          ("random-R1", None)],
                         ids=["sto3g", "631g-ob1", "C3-ob0", "R1"])
def test_rules_are_valid(name, ob):
    """T2 (tape mode) for both rules at four sizes, every branch of the hand-written resources
    (the dagger's adjoint Majorana key, the erase at three one-hot widths, R = 1); each rule is
    pinned and the only one registered under its class name (spec 7.4 rules 7 and 8)."""
    args = compilable(name, ob)
    wires, _ = layout(args)
    for cls, rule in CLASSES.values():
        assert FIXED_DECOMPS[cls.__name__] is rule
        assert list(qp.list_decomps(cls)) == [rule]
        qp.testing.assert_valid_decomp_rule(cls(*args, **wires), rule, check_matrix=False)


def test_required_wires():
    """By hand at H2_STO3G_TEST: work is the largest child need, max(1, 3, 3, 2, 2) = 3 (the SF
    load's w_r + w_b and the ladder's b_rot - 1)."""
    assert InnerSelect.required_wires(*compilable("sto3g")) == {
        "x_o": 2, "succ_x_o": 1, "sigma_1": 1, "b": 2, "succ_b": 1, "sigma_0": 1, "r": 1,
        "angles": 3, "G0": 1, "G1": 1, "b_neq_B": 1, "sg": 1, "psi_up": 2, "psi_dn": 2,
        "phase_grad": 4, "work": 3}
    assert InnerSelectDagger.required_wires(*compilable("random-R1"))["r"] == 0


@pytest.mark.parametrize(("change", "match"), [
    ({"fixup_bits": 3}, r"SF one-hot bits must lie in \[0, 2\]"),
    ({"N": 3}, "tail_angles must have N = 3 rows"),
    ({"tail_angles": ((1,), (8,))}, r"angles must lie in \[0, 2\*\*\(b_rot - 1\)\)"),
    ({"n_plus": 3}, r"n_plus must lie in \[0, 2\]"),
    ({"C": 0}, r"C must lie in \[1, None\]"),
    ({"phase_grad": [0, 1, 2]}, "'phase_grad' has 3 wires; expected 4"),
    ({"psi_dn": [0]}, "'psi_dn' has 1 wires; expected 2"),
    ({"work": [0, 1]}, "'work' has 2 wires; expected 3"),
])
@pytest.mark.parametrize("cls", [InnerSelect, InnerSelectDagger], ids=["select", "dagger"])
def test_constructor_validates(cls, change, match):
    args = dict(zip(InnerSelect.compilable_argnames, compilable("sto3g"), strict=True))
    wires, _ = layout(compilable("sto3g"))
    with pytest.raises(ValueError, match=match):
        cls(**(args | wires | change))
