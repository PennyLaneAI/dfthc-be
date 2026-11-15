"""InnerPrepare, InnerUnprepare and InnerSigns (spec 9.8, 9.9): the sign of every basis input,
the encoded model, failure signs, counts, rule selection, checks."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.alias import alias_probabilities
from dfthc.data.encoded import (
    dfthcbe_failure_signs,
    encoded_factors,
    our_failure_signs,
    uniform_failure_probability,
)
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.sos import inner_weights, normalization
from dfthc.data.tables import WalkTables
from dfthc.ops.prepare import (
    InnerPrepare,
    InnerSigns,
    InnerUnprepare,
    _adjoint_inner_signs,
    _inner_prepare,
    _inner_signs,
    _inner_unprepare,
)
from dfthc.ops.qroam import QROAMErase, QROAMRead
from dfthc.params import (
    FEMOCO54,
    H2_631G_TEST,
    H2_STO3G_TEST,
    MID_NAPHTHALENE,
    WalkParams,
    lg,
    odd_part,
)
from dfthc.registers import WalkLayout
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator
from dfthc.testing.tape import decompose_for_simulation

FAILURE_SIGN_MARGIN = 1e-2
"""dfthcbe's failure signs differ from ours by more than this where they differ at all
(tests/data/test_encoded.py: every nonzero difference is a multiple of 1/6 at (B + 1, k2) =
(5, 1) and of 1/2 at (7, 0))."""


def failed_address_case(B, k2, ke2=1, seed=5):
    """A spec 12.1 T3 shape: x_o of 3 bits with SF rows [0, 2) (R = 2, C = 1), one-body rows 2 to
    4 (N = 3) and failed values 5 to 7; (B + 1, k2) as given; b_k2 = 2 and s_u = 3."""
    params = WalkParams(N=3, R=2, B=B, C=1, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=1,
                        k2=k2, ke1=1, ke2=ke2)
    return params, random_factors(3, 2, B, 1, n_plus=1, seed=seed)


# Both test presets, and the failed-address shapes (B + 1, k2) = (3, 1), (5, 1), (6, 2), (7, 0)
# of spec 12.1 (the (28, 4) shape is for the read alone). At (7, 0) the one-hot over ke2 = 1 and
# all three bits of b would need 16 wires, more than the 8 of target and junk, so ke2 = 0.
CASES = {
    "sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "3-1": lambda: failed_address_case(2, 1),
    "5-1": lambda: failed_address_case(4, 1),
    "6-2": lambda: failed_address_case(5, 2),
    "7-0": lambda: failed_address_case(6, 0, ke2=0),
}


def case(name):
    p, factors = CASES[name]()
    return p, factors, WalkTables.from_factors(factors, p)


def inner_args(p, tables):
    return (p.N, p.R, p.B, p.C, tables.inner_alt, tables.inner_keep, tables.inner_sgn,
            tables.inner_d_sgn, p.b_k2, p.k2, p.ke2, p.s_u)


def structure(p):
    return (p.N, p.R, p.B, p.C, p.b_k2, p.k2, p.ke2, p.s_u)


def layout(sizes):
    """Consecutive wires for every register, in the order of ``sizes``."""
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def uniform_amplitudes(L, s):
    """FlaggedUniformPrep's amplitudes on t in [0, 2**lg(L)), from spec 9.2 with k recomputed
    from the spec formula (H on target, phase on top < l, H on top, phase on top = 0, H on top):
    every t < L carries the success amplitude, every t >= L the failure amplitude."""
    odd, n = odd_part(L), lg(odd_part(L))
    if n == 0:
        return np.full(L, 1 / np.sqrt(L), dtype=complex)
    angle = np.arccos(1 - 2 ** (odd.bit_length() - 1) / odd)
    k = int(np.floor(angle / (2 * np.pi) * 2**s)) % 2**s
    phase = np.exp(2j * np.pi * k / 2**s)
    h = np.ones((1, 1))
    for _ in range(n):
        h = np.kron(h, [[1, 1], [1, -1]]) / np.sqrt(2)
    top = h @ (np.where(np.arange(2**n) < odd, phase, 1) / np.sqrt(2**n))
    top[0] *= phase
    eta = lg(L) - n
    return np.repeat(h @ top, 2**eta) / np.sqrt(2**eta)


def register_values(state, register):
    """The big-endian value of ``register`` on every term of a sparse state."""
    values = np.zeros(len(state.amps), dtype=np.int64)
    for wire in register:
        values = (values << 1) | state.bits(wire).astype(np.int64)
    return values


def terms_without_gradient(state, wires):
    """``(names, {register values: amplitude})`` over every register but ``phase_grad``, with the
    catalysed phase-gradient factor divided out. Asserts that the state is a product with the
    phase-gradient state."""
    grad = wires["phase_grad"]
    names = [name for name in wires if name != "phase_grad"]
    columns = np.stack([register_values(state, wires[name]) for name in names], axis=1)
    phases = np.exp(2j * np.pi * register_values(state, grad) / 2 ** len(grad))
    groups = {}
    for key, amp in zip(map(tuple, columns.tolist()), state.amps * phases, strict=True):
        groups.setdefault(key, []).append(amp * np.sqrt(2 ** len(grad)))
    for key, amps in groups.items():
        assert len(amps) == 2 ** len(grad), key
        assert np.allclose(amps, amps[0], rtol=0, atol=1e-12), key
    return names, {key: amps[0] for key, amps in groups.items()}


@pytest.mark.parametrize("name", list(CASES))
def test_every_basis_input_gets_the_sign_of_its_selected_weight(name):
    """Spec 12.1 T3 items 2 and 3 (sparse simulator). With H on x_o, the uniform preparation of
    b and the Hadamards on m_b, every basis input (x_o, b_0, m_b) occurs once per sigma_0, with
    succ_b = [b_0 <= B]. After the read, keep test, alias swap and InnerSigns, each term decodes
    as the read loaded it, and its amplitude is the uniform preparation's for b_0 (success phase
    and factor included) over sqrt(2**(w_x + b_k2 + 1)), times the sign of the selected weight
    for a valid b_0 on an SF row and +1 otherwise. Summed per row, the signed weights equal the
    encoded model's inner amplitudes (spec 8.5), the failure signs equal ours (+1), and they
    equal dfthcbe's exactly when 2**w_b <= ceil((B + 1) / K2) K2, else differ as documented. At
    5-1 ``phase_grad`` has two wires more than ``s_u``, as Phase 3 passes it, so the amplitudes
    pin which wires the rotations add into. Then InnerPrepare and InnerUnprepare under the
    all-ones outcome schedule (every fix-up applied) and the alternating one (which also sees a
    permutation of the erase masks' bits, to which the all-ones parity is blind) leave ``x_o`` in
    its uniform superposition and every other register in |0> (the certificates prove every
    outcome)."""
    p, factors, tables = case(name)
    sizes = InnerPrepare.required_wires(*structure(p))
    sizes["phase_grad"] += 2 if name == "5-1" else 0
    wires, num_wires = layout(sizes)

    def prepare():
        for wire in wires["x_o"]:
            qp.H(wire)
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        InnerPrepare(*inner_args(p, tables), **wires)

    def qfunc():
        prepare()
        InnerSigns(wires["sgn"], wires["d_sgn"], wires["less_than_b"])

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    names, terms = terms_without_gradient(state, wires)
    shape = InnerPrepare.read_shape(p.N, p.R, p.B, p.C, p.b_k2, p.k2)
    table = InnerPrepare.read_table(*inner_args(p, tables)[:9])
    junk_bits = (p.K2 - 1) * p.d2
    amplitudes = uniform_amplitudes(p.B + 1, p.s_u)
    scale = 1 / np.sqrt(2 ** (p.w_x + p.b_k2 + 1))
    signs = np.sign(inner_weights(factors))
    selected, failed, drawn = np.zeros((p.RC, p.B + 1)), np.zeros(p.RC), set()
    for key, amp in terms.items():
        v = dict(zip(names, key, strict=True))
        x, swapped = v["x_o"], v["less_than_b"]
        b0 = v["alt_b"] if swapped else v["b"]
        valid = x < p.RC and b0 <= p.B
        loaded = QROAMRead.loaded_bits(table, (x << p.w_b) | b0, **shape)
        target, junk = loaded >> junk_bits, loaded & (2**junk_bits - 1)
        alt, keep = target >> (p.b_k2 + 2 + p.w_r), (target >> (2 + p.w_r)) % 2**p.b_k2
        columns = ((target >> (1 + p.w_r)) & 1, (target >> p.w_r) & 1, target % 2**p.w_r)
        assert (v["keep_b"], v["sgn"], v["d_sgn"], v["r"]) == (keep, *columns)
        assert (v["junk_b"], v["work"]) == (junk, 0)
        assert v["r"] == (x // p.C if valid else 0)
        assert swapped == int(keep <= v["m_b"])
        assert (v["b"], v["alt_b"]) == ((alt, b0) if swapped else (b0, alt))
        assert v["succ_b"] == int(b0 <= p.B)
        sign = (amp / (amplitudes[b0] * scale)).real
        assert abs(amp - (signs[x, v["b"]] if valid else 1) * amplitudes[b0] * scale) < 1e-12
        drawn.add((x, b0, v["m_b"], v["sigma_0"]))
        if valid:
            selected[x, v["b"]] += sign * abs(amp) ** 2 * 2**p.w_x
        elif x < p.RC:
            failed[x] += sign * abs(amp) ** 2 * 2**p.w_x
    nonzero = np.count_nonzero(np.abs(amplitudes) > 1e-12)
    assert len(terms) == len(drawn) == 2**p.w_x * nonzero * 2 ** (p.b_k2 + 1)
    fail = uniform_failure_probability(p.B + 1, p.s_u)
    assert fail > 0
    outer = alias_probabilities(tables.outer_alt, tables.outer_keep, p.b_k1)[:p.RC]
    model = (inner_weights(encoded_factors(p, tables, factors))
             / np.sqrt(4 * normalization(factors) * outer)[:, None])
    selected[:, p.B] += failed
    assert np.allclose(selected, model, rtol=0, atol=1e-12)
    assert np.allclose(failed / fail, our_failure_signs(p, tables), rtol=0, atol=1e-12)
    port = dfthcbe_failure_signs(p, tables)
    if 2**p.w_b <= -(-(p.B + 1) // p.K2) * p.K2:
        assert np.allclose(port, failed / fail, rtol=0, atol=1e-12)
    else:
        assert np.abs(port - failed / fail).max() > FAILURE_SIGN_MARGIN

    def round_trip():
        prepare()
        InnerUnprepare(*inner_args(p, tables), **wires)

    operations = decompose_for_simulation(round_trip).operations
    measured, rest = p.K2 * p.d2, (0,) * (len(names) - 1)
    for schedule in ([1] * measured, [i % 2 for i in range(measured)]):
        final = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True).run(
            operations)
        _, restored = terms_without_gradient(final, wires)
        assert names[0] == "x_o" and sorted(restored) == [(x, *rest) for x in range(2**p.w_x)]
        assert all(abs(amp - 2 ** (-p.w_x / 2)) < 1e-12 for amp in restored.values())


def expected_and_uncomputes(p, unprepare):
    """A: every TemporaryAND of the pair is uncomputed by measurement inside it (spec 9.24). The
    prepare uncomputes the uniform preparation's 2(n - 1) + 2(s_u - 1) ANDs and the read's
    iteration ANDs U; the unprepare the comparator's b_k2, the erase fix-up's T and the adjoint
    uniform preparation's 3(n - 1) + 2(s_u - 1) (n = lg(l_b); none of the uniform terms if
    l_b = 1)."""
    n, s = lg(odd_part(p.B + 1)), p.s_u
    widths, live, width, k = InnerPrepare.read_shape(p.N, p.R, p.B, p.C, p.b_k2, p.k2).values()
    if unprepare:
        ob = p.ke2 + p.w_b
        erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width,
                                  ob, False)
        return p.b_k2 + erase.toffoli + (3 * (n - 1) + 2 * (s - 1) if n else 0)
    read = costs.qroam_read(QROAMRead.visited_ranges(widths, live, k), k, width, False)
    return (2 * (n - 1) + 2 * (s - 1) if n else 0) + read.toffoli - (2**k - 1) * width


def _assert_counts(p, tables, unprepare):
    wires, num_wires = layout(InnerPrepare.required_wires(*structure(p)))
    op_class = InnerUnprepare if unprepare else InnerPrepare
    resources = compiled_resources(lambda: op_class(*inner_args(p, tables), **wires),
                                   num_wires=num_wires)
    expected = (costs.inner_unprepare if unprepare else costs.inner_prepare)(p)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measurements == (p.K2 * p.d2 if unprepare else 0)
    assert resources.measured_and_uncomputes == expected_and_uncomputes(p, unprepare)


@pytest.mark.compiled
@pytest.mark.parametrize("unprepare", [False, True], ids=["prepare", "unprepare"])
@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_counts_match_the_closed_forms(name, unprepare):
    """T4 and the T6 signature: T and M equal dfthc.costs.inner_prepare / inner_unprepare, the
    unprepare measures every target and junk qubit (K2 d2), and A is as derived."""
    p, _, tables = case(name)
    _assert_counts(p, tables, unprepare)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("unprepare", [False, True], ids=["prepare", "unprepare"])
def test_counts_at_mid_naphthalene(unprepare):
    """T4 at MID_NAPHTHALENE with the naphthalene tables (nightly tier)."""
    p = MID_NAPHTHALENE
    _assert_counts(p, WalkTables.from_factors(load_factors("naphthalene"), p), unprepare)


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 on the closed forms, nothing compiled: prepare 18 + 933 + 15 + 5 = 971 with
    A = 16 + 543 = 559, unprepare 5 + 196 + 16 = 217 with M = 416 and A = 15 + 196 + 18 = 229;
    the pair's A is its T less its Toffoli-only part 2 w_b + (K2 - 1) d2, as
    dfthc.costs.measured_and_uncomputes counts it. InnerSigns costs 0 (its A = 0 is the compiled
    count of test_inner_signs_counts_rules_and_device_level: it has three wires at every size).
    The operator's own read and erase shapes give the same pieces, and its registers but x_o are
    spec 10.3's 65-wire inner group and the 390 junk wires, which the walk's layout puts on the
    angle wires (spec 10.2)."""
    p = FEMOCO54
    assert costs.inner_prepare(p) == costs.Cost(971)
    assert costs.inner_unprepare(p) == costs.Cost(217, 416)
    assert costs.inner_signs() == costs.Cost(0)
    uncomputes = expected_and_uncomputes(p, False), expected_and_uncomputes(p, True)
    assert uncomputes == (559, 229)
    assert sum(uncomputes) == 971 + 217 - (2 * p.w_b + (p.K2 - 1) * p.d2)
    widths, live, width, k = InnerPrepare.read_shape(p.N, p.R, p.B, p.C, p.b_k2, p.k2).values()
    ob = p.ke2 + p.w_b
    read = costs.qroam_read(QROAMRead.visited_ranges(widths, live, k), k, width, False)
    erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width, ob,
                              False)
    assert (read, erase) == (costs.Cost(933), costs.Cost(196, 416))
    uniform = costs.flagged_uniform_prep(p.B + 1, p.s_u), costs.flagged_uniform_unprep(p.B + 1,
                                                                                       p.s_u)
    assert uniform == (costs.Cost(18), costs.Cost(16)) and (p.b_k2, p.w_b) == (15, 5)
    assert uniform[0].toffoli + read.toffoli + p.b_k2 + p.w_b == 971
    assert p.w_b + erase.toffoli + uniform[1].toffoli == 217
    sizes = InnerPrepare.required_wires(*structure(p))
    assert sizes == {"x_o": 9, "b": 5, "succ_b": 1, "succ_work_b": 2, "sigma_0": 1, "m_b": 15,
                     "less_than_b": 1, "cmp_work_b": 14, "alt_b": 5, "keep_b": 15, "sgn": 1,
                     "d_sgn": 1, "r": 4, "junk_b": 390, "phase_grad": 7, "work": 16}
    layout = WalkLayout.from_params(p)
    assert sizes["junk_b"] == len(layout.junk_b) == 390
    assert (sum(sizes.values()) - sizes["x_o"] - sizes["junk_b"] - sizes["phase_grad"]
            - sizes["work"] == len(layout.inner) == 65)


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the solver names the pinned rules of the three classes and of the read and the erase;
    T5 at level="device" (L-20)."""
    p, _, tables = case("sto3g")
    wires, num_wires = layout(InnerPrepare.required_wires(*structure(p)))

    def program():
        InnerPrepare(*inner_args(p, tables), **wires)
        InnerSigns(wires["sgn"], wires["d_sgn"], wires["less_than_b"])
        InnerUnprepare(*inner_args(p, tables), **wires)

    qp.analyze(verbose_program(program, num_wires=num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_inner_prepare", "_inner_signs", "_inner_unprepare", "_qroam_read",
            "_qroam_erase"} <= rules, rules
    compiled_resources(program, num_wires=num_wires, level="device")


def random_inner_args(N, R, B, C, b_k2, k2, ke2, s_u, seed=7):
    rng = np.random.default_rng(seed)
    shape = (R * C, B + 1)
    columns = [rng.integers(0, B + 1, shape), rng.integers(0, 2**b_k2, shape),
               rng.integers(0, 2, shape), rng.integers(0, 2, shape)]
    tables = [tuple(tuple(int(v) for v in row) for row in column) for column in columns]
    return (N, R, B, C, *tables, b_k2, k2, ke2, s_u)


# T2 shapes: the sto3g shape (l_b = 3, k2 = 1; with two phase_grad wires more than s_u); the
# (7, 0) shape (k2 = 0: no junk; ke2 = 0); and R = 1, B + 1 = 4, b_k2 = 1 (an empty r register,
# l_b = 1: no uniform work, and no comparator work).
RULE_SHAPES = {"sto3g": (2, 2, 2, 1, 2, 1, 1, 3), "7-0": (3, 2, 6, 1, 2, 0, 0, 3),
               "r1-l1-b1": (2, 1, 3, 2, 1, 2, 1, 3)}


@pytest.mark.parametrize("rule", ["prepare", "unprepare"])
@pytest.mark.parametrize("name", list(RULE_SHAPES))
def test_rules_are_valid(name, rule):
    """T2 in every branch of the hand-written resources. At the sto3g shape ``phase_grad`` has two
    wires more than ``s_u``, as when Phase 3 passes the whole register, so the declared resources
    hold only if the rule passes the uniform preparation ``phase_grad[:s_u]``."""
    args = random_inner_args(*RULE_SHAPES[name])
    sizes = InnerPrepare.required_wires(*args[:4], *args[8:])
    sizes["phase_grad"] += 2 if name == "sto3g" else 0
    wires, _ = layout(sizes)
    op_class, rule_fn = ((InnerUnprepare, _inner_unprepare) if rule == "unprepare"
                         else (InnerPrepare, _inner_prepare))
    qp.testing.assert_valid_decomp_rule(op_class(*args, **wires), rule_fn, check_matrix=False)


def test_inner_signs_rules_are_valid():
    """T2 for InnerSigns' base rule and its pinned adjoint rule (tape mode)."""
    op = InnerSigns([0], [1], [2])
    qp.testing.assert_valid_decomp_rule(op, _inner_signs, check_matrix=False)
    qp.testing.assert_valid_decomp_rule(qp.adjoint(op), _adjoint_inner_signs, check_matrix=False)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["plain", "adjoint"])
def test_inner_signs_phase_on_every_basis_input(adjoint):
    """T3 on lightning: on H|000> of (sgn, d_sgn, less_than_b), the phase
    (-1)**(sgn XOR (less_than_b AND d_sgn)) on every basis input; the adjoint is the same."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    def circuit():
        for wire in range(3):
            qp.H(wire)
        op = InnerSigns([0], [1], [2])
        if adjoint:
            qp.adjoint(op)
        return qp.state()

    expected = [(-1) ** (sgn ^ (less_than & d_sgn)) / np.sqrt(8)
                for sgn in (0, 1) for d_sgn in (0, 1) for less_than in (0, 1)]
    assert np.allclose(np.asarray(circuit()), expected, rtol=0, atol=1e-12)


@pytest.mark.compiled
def test_inner_signs_counts_rules_and_device_level(capfd):
    """T4: T = M = A = 0 (dfthc.costs.inner_signs), one PauliZ and one CZ per application; T6:
    the base and the pinned adjoint rule are chosen; T5 at level="device"."""

    def program():
        InnerSigns([0], [1], [2])
        qp.adjoint(InnerSigns([0], [1], [2]))

    resources = compiled_resources(program, num_wires=3)
    expected = costs.inner_signs()
    assert (resources.toffoli, resources.measurements, resources.measured_and_uncomputes) == (
        2 * expected.toffoli, 2 * expected.measurements, 0)
    assert resources.counts == {"CZ": 2, "PauliZ": 2}
    qp.analyze(verbose_program(program, num_wires=3), level="user")()
    captured = capfd.readouterr()
    assert {"_inner_signs", "_adjoint_inner_signs"} <= set(chosen_rules(captured.out
                                                                        + captured.err))
    compiled_resources(program, num_wires=3, level="device")


def test_read_shape_table_and_required_wires():
    assert InnerPrepare.read_shape(2, 2, 2, 1, 2, 1) == {
        "address_widths": (2, 2), "live_ranges": ((0, 2), (0, 3)), "width": 7, "block_bits": 1}
    assert InnerPrepare.read_table(1, 2, 1, 1, ((1, 0), (0, 1)), ((2, 3), (0, 1)),
                                   ((1, 0), (0, 1)), ((1, 1), (0, 0)), 2) == (
        (0, (54,)), (1, (26,)), (2, (1,)), (3, (45,)))
    # C = 3 (FeMoco has C = 27): r = x // C is not a bit field of x_o, unlike at C = 1 and 2
    zeros = ((0, 0),) * 9
    assert InnerPrepare.read_table(1, 3, 1, 3, zeros, zeros, zeros, zeros, 1) == tuple(
        ((x << 1) | b, (x // 3,)) for x in range(9) for b in range(2))
    assert InnerPrepare.required_wires(2, 2, 2, 1, 2, 1, 1, 3) == {
        "x_o": 2, "b": 2, "succ_b": 1, "succ_work_b": 1, "sigma_0": 1, "m_b": 2,
        "less_than_b": 1, "cmp_work_b": 1, "alt_b": 2, "keep_b": 2, "sgn": 1, "d_sgn": 1, "r": 1,
        "junk_b": 7, "phase_grad": 3, "work": 7}
    assert InnerUnprepare.required_wires(2, 2, 2, 1, 2, 1, 1, 3) == InnerPrepare.required_wires(
        2, 2, 2, 1, 2, 1, 1, 3)
    assert InnerSigns.required_wires() == {"sgn": 1, "d_sgn": 1, "less_than_b": 1}


ROWS = ((0, 1, 2), (2, 1, 0))
BITS = ((0, 1, 0), (1, 1, 0))
BASE = {"N": 3, "R": 2, "B": 2, "C": 1, "alt": ROWS, "keep": ((3, 0, 1), (2, 2, 0)),
        "sgn_table": BITS, "d_sgn_table": BITS, "b_k2": 2, "k2": 1, "ke2": 1, "s_u": 3}


@pytest.mark.parametrize(("change", "match"), [
    ({"N": 0}, r"N must lie in \[1, None\]"),
    ({"B": 0}, r"B must lie in \[1, None\]"),
    ({"k2": 3}, r"k2 must lie in \[0, 2\]"),
    ({"ke2": 2}, r"ke2 must lie in \[0, 1\]"),
    ({"k2": 0}, "needs 8 wires, more than the 7 target and junk wires"),
    ({"s_u": 2}, r"s_u must lie in \[3, None\]"),
    ({"alt": ((0, 1, 3), (2, 1, 0))}, r"alt entries must lie in \[0, 3\), got 3"),
    ({"keep": ((4, 0, 1), (2, 2, 0))}, r"keep entries must lie in \[0, 4\), got 4"),
    ({"sgn_table": ((0, 2, 0), (1, 1, 0))}, r"sgn_table entries must lie in \[0, 2\), got 2"),
    ({"d_sgn_table": ((0, 1), (1, 1))}, r"d_sgn_table must be a table of ints of shape \(2, 3\)"),
    ({"junk_b": 6}, "'junk_b' has 6 wires; expected 7"),
    ({"phase_grad": 2}, "'phase_grad' has 2 wires; expected at least 3"),
])
@pytest.mark.parametrize("op_class", [InnerPrepare, InnerUnprepare])
def test_constructors_validate(op_class, change, match):
    sizes = InnerPrepare.required_wires(3, 2, 2, 1, 2, 1, 1, 3)
    sizes |= {name: value for name, value in change.items() if name in sizes}
    wires = {name: list(range(100 * i, 100 * i + size))
             for i, (name, size) in enumerate(sizes.items())}
    args = BASE | {name: value for name, value in change.items() if name in BASE}
    with pytest.raises(ValueError, match=match):
        op_class(**args, **wires)


def test_inner_signs_constructor_validates():
    with pytest.raises(ValueError, match="'d_sgn' has 2 wires; expected 1"):
        InnerSigns([0], [1, 2], [3])
