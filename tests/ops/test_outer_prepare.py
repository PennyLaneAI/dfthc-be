"""OuterPrepare and OuterUnprepare (spec 9.7): the alias state, counts, rule selection, checks."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, PENNYLANE_PINS, compiled_resources
from dfthc.data.alias import alias_probabilities
from dfthc.data.encoded import encoded_block, encoded_factors, uniform_failure_probability
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.sos import normalization
from dfthc.data.tables import WalkTables
from dfthc.ops.prepare import OuterPrepare, OuterUnprepare, _outer_prepare, _outer_unprepare
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

X_O_FAILS = WalkParams(N=2, R=3, B=1, C=1, n_plus=1, b_rot=6, b_k1=3, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
"""X_o = 5 (l = 5): the uniform preparation of x_o fails, with weight 0.15 and a phase (the
shape of tests/data/test_encoded.py)."""
X_O_6_K2 = WalkParams(N=2, R=2, B=1, C=2, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=2, k2=1,
                      ke1=1, ke2=1)
"""X_o = 6 (l = 3) with K1 = 4: dead positions 6 and 7 in the read's last block."""
X_O_7_K0_B1 = WalkParams(N=3, R=2, B=1, C=2, n_plus=2, b_rot=4, b_k1=1, b_k2=2, s_u=3, k1=0,
                         k2=1, ke1=2, ke2=1)
"""X_o = 7 (l = 7) with k1 = 0 (no junk) and b_k1 = 1 (no comparator work wires)."""

CASES = {
    "sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "x-o-5": lambda: (X_O_FAILS, random_factors(2, 3, 1, 1, n_plus=1, seed=3)),
    "x-o-6-k2": lambda: (X_O_6_K2, random_factors(2, 2, 1, 2, n_plus=1, seed=4)),
    "x-o-7-k0-b1": lambda: (X_O_7_K0_B1, random_factors(3, 2, 1, 2, n_plus=2, seed=5)),
}


def case(name):
    p, factors = CASES[name]()
    return p, factors, WalkTables.from_factors(factors, p)


def outer_args(p, tables):
    return (p.X_o, tables.outer_alt, tables.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)


def layout(p, extra_grad=0):
    """Consecutive wires for every register, in the order of ``OuterPrepare.wire_argnames``;
    ``phase_grad`` gets ``extra_grad`` wires more than ``s_u``."""
    sizes = OuterPrepare.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)
    sizes["phase_grad"] += extra_grad
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
def test_prepared_state_is_the_alias_distribution(name):
    """T3 (sparse simulator). Every term decodes to the drawn x_0: keep and junk as the read
    loaded them, less_than = [keep <= m], the alias swap, succ = [x_0 < X_o], work |0>, and the
    amplitude of the uniform preparation over sqrt(2**(b_k1 + 1)), its success phase and factor
    sqrt(1 - P_fail) included; every (x_0, m, sigma_1) occurs. The succ = 1 marginal is
    (1 - P_fail) alias_probabilities, and against the encoded model (spec 8.5) its total is
    Lambda / Lambda~ and its one-body part (1 - P_fail) w~_pm / Lambda. At x-o-5 ``phase_grad``
    has two wires more than ``s_u``, as Phase 3 passes it, so the amplitudes pin which wires the
    rotations add into. Then OuterUnprepare returns every register to |0> with amplitude 1
    under the all-ones outcome schedule (every fix-up applied), the alternating one and a seeded
    one: the all-ones parity sees only each mask's weight, the alternating one also a permutation
    of the erase entries' bits that moves a bit to a position of the other parity, and the seeded
    one the permutations that keep each bit's position parity, such as a bit reversal of an
    odd-width entry (final review m-7). The certificates prove every outcome (nightly)."""
    p, factors, tables = case(name)
    wires, num_wires = layout(p, extra_grad=2 if name == "x-o-5" else 0)

    def qfunc():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        OuterPrepare(*outer_args(p, tables), **wires)

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    names, terms = terms_without_gradient(state, wires)
    shape = OuterPrepare.read_shape(p.X_o, p.b_k1, p.k1)
    table = OuterPrepare.read_table(p.X_o, tables.outer_alt, tables.outer_keep, p.b_k1)
    junk_bits = (p.K1 - 1) * p.d1
    amplitudes = uniform_amplitudes(p.X_o, p.s_u)
    scale = 1 / np.sqrt(2 ** (p.b_k1 + 1))
    marginal, drawn = np.zeros(p.X_o), set()
    for key, amp in terms.items():
        v = dict(zip(names, key, strict=True))
        swapped = v["less_than_x_o"]
        x0 = v["alt_x_o"] if swapped else v["x_o"]
        loaded = QROAMRead.loaded_bits(table, x0, **shape)
        target, junk = loaded >> junk_bits, loaded & (2**junk_bits - 1)
        alt0, keep0 = target >> p.b_k1, target & (2**p.b_k1 - 1)
        assert (v["keep_x_o"], v["junk_x_o"], v["work"]) == (keep0, junk, 0)
        assert swapped == int(keep0 <= v["m_x_o"])
        assert (v["x_o"], v["alt_x_o"]) == ((alt0, x0) if swapped else (x0, alt0))
        assert v["succ_x_o"] == int(x0 < p.X_o)
        assert abs(amp - amplitudes[x0] * scale) < 1e-12, (x0, amp)
        drawn.add((x0, v["m_x_o"], v["sigma_1"]))
        if x0 < p.X_o:
            marginal[v["x_o"]] += abs(amp) ** 2
    nonzero = np.count_nonzero(np.abs(amplitudes) > 1e-12)
    assert len(terms) == len(drawn) == nonzero * 2 ** (p.b_k1 + 1)
    fail = uniform_failure_probability(p.X_o, p.s_u)
    assert np.allclose(marginal,
                       (1 - fail) * alias_probabilities(tables.outer_alt, tables.outer_keep,
                                                        p.b_k1), rtol=0, atol=1e-12)
    lam = normalization(factors)
    _, lambda_tilde = encoded_block(p, tables, factors)
    enc = encoded_factors(p, tables, factors)
    assert abs(marginal.sum() - lam / lambda_tilde) < 1e-12
    assert np.allclose(marginal[p.RC:],
                       (1 - fail) * np.concatenate([enc.w_plus, enc.w_minus]) / lam,
                       rtol=0, atol=1e-12)

    def round_trip():
        qfunc()
        OuterUnprepare(*outer_args(p, tables), **wires)

    operations = decompose_for_simulation(round_trip).operations
    measured, zero = p.K1 * p.d1, (0,) * len(names)
    for schedule in ([1] * measured, [i % 2 for i in range(measured)],
                     np.random.default_rng(3).integers(0, 2, measured).tolist()):
        final = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True).run(
            operations)
        _, restored = terms_without_gradient(final, wires)
        assert list(restored) == [zero] and abs(restored[zero] - 1) < 1e-12, schedule


def _lightning(num_wires, body, returns, seed=None):
    @qp.qjit(capture=True, seed=seed)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps={**FIXED_DECOMPS, **PENNYLANE_PINS})
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return returns()

    return np.asarray(circuit())


@pytest.mark.compiled
def test_compiled_distribution_on_lightning():
    """T3 on the compiled path at H2_STO3G_TEST (19 wires): P(x_o = x, succ_x_o = 1) is the
    alias distribution, and succ_x_o = 0 never occurs (X_o = 4 is a power of two)."""
    p, _, tables = case("sto3g")
    wires, num_wires = layout(p)

    def body():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        OuterPrepare(*outer_args(p, tables), **wires)

    probs = _lightning(num_wires, body,
                       lambda: qp.probs(wires=wires["x_o"] + wires["succ_x_o"]))
    probs = probs.reshape(2 ** lg(p.X_o), 2)
    expected = alias_probabilities(tables.outer_alt, tables.outer_keep, p.b_k1)
    assert np.allclose(probs[:, 1], expected, rtol=0, atol=1e-10)
    assert np.allclose(probs[:, 0], 0, rtol=0, atol=1e-10)


@pytest.mark.compiled
def test_lightning_round_trip_with_two_seeds():
    """Compiled OuterPrepare then OuterUnprepare at H2_STO3G_TEST: every register returns to |0>
    and the phase gradient is unchanged, for two measurement seeds (smoke runs; the certificates
    in test_certificates_prepare.py prove every outcome)."""
    p, _, tables = case("sto3g")
    wires, num_wires = layout(p)
    grad = wires["phase_grad"]

    def body():
        qp.PhaseGradientStatePrep(wires=grad)
        OuterPrepare(*outer_args(p, tables), **wires)
        OuterUnprepare(*outer_args(p, tables), **wires)

    gradient = np.exp(-2j * np.pi * np.arange(2 ** len(grad)) / 2 ** len(grad))
    expected = np.kron(np.kron(np.eye(2 ** grad[0])[0], gradient / np.sqrt(2 ** len(grad))),
                       np.eye(2 ** (num_wires - grad[-1] - 1))[0])
    for seed in range(2):
        state = _lightning(num_wires, body, qp.state, seed=seed)
        assert np.allclose(state, expected, rtol=0, atol=1e-10), seed


def expected_and_uncomputes(p, unprepare):
    """A: every TemporaryAND of the pair is uncomputed by measurement inside it (spec 9.24). The
    prepare uncomputes the uniform preparation's 2(n - 1) + 2(s_u - 1) ANDs and the read's
    select ANDs N; the unprepare the comparator's b_k1, the erase fix-up's T and the adjoint
    uniform preparation's 3(n - 1) + 2(s_u - 1) (n = lg(l_x); none of the uniform terms if
    l_x = 1)."""
    n, s = lg(odd_part(p.X_o)), p.s_u
    shape = OuterPrepare.read_shape(p.X_o, p.b_k1, p.k1)
    widths, live, width, k = shape.values()
    if unprepare:
        erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, p.ke1), k, width,
                                  p.ke1, False)
        return p.b_k1 + erase.toffoli + (3 * (n - 1) + 2 * (s - 1) if n else 0)
    read = costs.qroam_read(*QROAMRead.select_size(widths, live, k, None), k, width)
    return (2 * (n - 1) + 2 * (s - 1) if n else 0) + read.toffoli - (2**k - 1) * width


def _assert_counts(p, tables, unprepare):
    wires, num_wires = layout(p)
    op_class = OuterUnprepare if unprepare else OuterPrepare
    resources = compiled_resources(lambda: op_class(*outer_args(p, tables), **wires),
                                   num_wires=num_wires)
    expected = (costs.outer_unprepare if unprepare else costs.outer_prepare)(p)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measurements == (p.K1 * p.d1 if unprepare else 0)
    assert resources.measured_and_uncomputes == expected_and_uncomputes(p, unprepare)


@pytest.mark.compiled
@pytest.mark.parametrize("unprepare", [False, True], ids=["prepare", "unprepare"])
@pytest.mark.parametrize("name", ["sto3g", "631g", "x-o-5"])
def test_counts_match_the_closed_forms(name, unprepare):
    """T4 and the T6 signature: T and M equal dfthc.costs.outer_prepare / outer_unprepare, the
    unprepare measures every target and junk qubit (K1 d1), and A is as derived."""
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
    """FeMoco-54 on the closed forms, nothing compiled: prepare 30 + 155 + 15 + 9 = 209 with
    A = 24 + 83 = 107, unprepare 9 + 36 + 24 = 69 with M = 96 and A = 15 + 36 + 30 = 81; the pair
    uncomputes all it computes, so its A is its T less its Toffoli-only part 2 w_x + (K1 - 1) d1,
    as dfthc.costs.measured_and_uncomputes counts it. The operator's own read and erase shapes
    give the same pieces, and its registers are spec 10.3's 71-wire outer group and the 72 junk
    wires, which the walk's layout puts on the angle wires (spec 10.2)."""
    p = FEMOCO54
    assert costs.outer_prepare(p) == costs.Cost(209)
    assert costs.outer_unprepare(p) == costs.Cost(69, 96)
    uncomputes = expected_and_uncomputes(p, False), expected_and_uncomputes(p, True)
    assert uncomputes == (107, 81)
    assert sum(uncomputes) == 209 + 69 - (2 * p.w_x + (p.K1 - 1) * p.d1)
    widths, live, width, k = OuterPrepare.read_shape(p.X_o, p.b_k1, p.k1).values()
    read = costs.qroam_read(*QROAMRead.select_size(widths, live, k, None), k, width)
    erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, p.ke1), k, width,
                              p.ke1, False)
    assert (read, erase) == (costs.Cost(155), costs.Cost(36, 96))
    uniform = costs.flagged_uniform_prep(p.X_o, p.s_u), costs.flagged_uniform_unprep(p.X_o, p.s_u)
    assert uniform == (costs.Cost(30), costs.Cost(24)) and (p.b_k1, p.w_x) == (15, 9)
    assert uniform[0].toffoli + read.toffoli + p.b_k1 + p.w_x == 209
    assert p.w_x + erase.toffoli + uniform[1].toffoli == 69
    sizes = OuterPrepare.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)
    assert sizes == {"x_o": 9, "succ_x_o": 1, "succ_work_x_o": 6, "sigma_1": 1, "m_x_o": 15,
                     "less_than_x_o": 1, "cmp_work_x_o": 14, "alt_x_o": 9, "keep_x_o": 15,
                     "junk_x_o": 72, "phase_grad": 7, "work": 20}
    layout = WalkLayout.from_params(p)
    assert sizes["junk_x_o"] == len(layout.junk_x_o) == 72
    assert sum(sizes.values()) - sizes["junk_x_o"] - sizes["phase_grad"] - sizes["work"] == len(
        layout.outer) == 71


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the solver names the pinned rules of both classes and of the read and the erase; T5
    at level="device" (L-20)."""
    p, _, tables = case("x-o-5")
    wires, num_wires = layout(p)

    def program():
        OuterPrepare(*outer_args(p, tables), **wires)
        OuterUnprepare(*outer_args(p, tables), **wires)

    qp.analyze(verbose_program(program, num_wires=num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_outer_prepare", "_outer_unprepare", "_qroam_read", "_qroam_erase"} <= rules, rules
    compiled_resources(program, num_wires=num_wires, level="device")


@pytest.mark.parametrize("rule", ["prepare", "unprepare"])
@pytest.mark.parametrize("name", ["sto3g", "x-o-5", "x-o-7-k0-b1"])
def test_rules_are_valid(name, rule):
    """T2 in every branch of the hand-written resources: l_x = 1 (sto3g; no uniform work), l_x > 1
    (x-o-5), and k1 = 0 with b_k1 = 1 (no junk, no comparator work). At x-o-5 ``phase_grad`` has
    two wires more than ``s_u``, as when Phase 3 passes the whole register, so the declared
    resources hold only if the rule passes the uniform preparation ``phase_grad[:s_u]``."""
    p, _, tables = case(name)
    wires, _ = layout(p, extra_grad=2 if name == "x-o-5" else 0)
    op_class, rule_fn = ((OuterUnprepare, _outer_unprepare) if rule == "unprepare"
                         else (OuterPrepare, _outer_prepare))
    qp.testing.assert_valid_decomp_rule(op_class(*outer_args(p, tables), **wires), rule_fn,
                                        check_matrix=False)


def test_read_shape_and_table_by_hand():
    assert OuterPrepare.read_shape(5, 3, 1) == {"address_widths": (3,), "live_ranges": ((0, 5),),
                                                "width": 6, "block_bits": 1}
    assert OuterPrepare.read_table(5, (1, 0, 4, 2, 3), (7, 0, 5, 1, 2), 3) == (
        (0, (15,)), (1, (0,)), (2, (37,)), (3, (17,)), (4, (26,)))
    assert OuterPrepare.read_table(3, (2, 0, 1), (2**31, 5, 0), 32) == (
        (0, (2**31, 2)), (1, (5, 0)), (2, (0, 1)))
    assert OuterPrepare.required_wires(4, 2, 1, 1, 3) == {
        "x_o": 2, "succ_x_o": 1, "succ_work_x_o": 0, "sigma_1": 1, "m_x_o": 2,
        "less_than_x_o": 1, "cmp_work_x_o": 1, "alt_x_o": 2, "keep_x_o": 2, "junk_x_o": 4,
        "phase_grad": 3, "work": 0}
    assert OuterUnprepare.required_wires(4, 2, 1, 1, 3) == OuterPrepare.required_wires(4, 2, 1, 1,
                                                                                       3)
    # ke1 = 0: the erase's fix-up iterates both address bits, so its work need (1) is the largest
    assert OuterPrepare.required_wires(4, 2, 1, 0, 3)["work"] == 1


BASE = {"X_o": 5, "alt": (1, 0, 4, 2, 3), "keep": (7, 0, 5, 1, 2), "b_k1": 3, "k1": 1, "ke1": 1,
        "s_u": 3}


@pytest.mark.parametrize(("change", "match"), [
    ({"X_o": 1}, r"X_o must lie in \[2, None\]"),
    ({"b_k1": 0}, r"b_k1 must lie in \[1, None\]"),
    ({"k1": 4}, r"k1 must lie in \[0, 3\]"),
    ({"ke1": 4}, r"ke1 must lie in \[0, 3\]"),
    ({"k1": 0, "ke1": 3}, "needs 8 wires, more than the 6 target and junk wires"),
    ({"s_u": 2}, r"s_u must lie in \[3, None\]"),
    ({"alt": (1, 0, 5, 2, 3)}, r"alt entries must lie in \[0, 5\), got 5"),
    ({"keep": (8, 0, 5, 1, 2)}, r"keep entries must lie in \[0, 8\), got 8"),
    ({"alt": (1, 0, 4, 2)}, r"alt must be a table of ints of shape \(5,\)"),
    ({"junk_x_o": 5}, "'junk_x_o' has 5 wires; expected 6"),
    ({"phase_grad": 2}, "'phase_grad' has 2 wires; expected at least 3"),
])
@pytest.mark.parametrize("op_class", [OuterPrepare, OuterUnprepare])
def test_constructors_validate(op_class, change, match):
    sizes = OuterPrepare.required_wires(5, 3, 1, 1, 3)
    sizes |= {name: value for name, value in change.items() if name in sizes}
    wires = {name: list(range(100 * i, 100 * i + size))
             for i, (name, size) in enumerate(sizes.items())}
    args = BASE | {name: value for name, value in change.items() if name in BASE}
    with pytest.raises(ValueError, match=match):
        op_class(**args, **wires)
