"""AngleErase (spec 9.17): counts, rule selection, smoke runs, parity linearity, validation."""

import itertools
import math
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc import costs
from dfthc.compile import (
    FIXED_DECOMPS,
    GATE_SET,
    compiled_resources,
    mcm_parities,
    pack_masks,
    packed_parities,
)
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.angle_data import (
    AngleErase,
    SFAngleLoad,
    TailAngleLoad,
    _angle_erase,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program


def random_tables(p, seed=3):
    """Random angle tables of p's shape, every angle below 2**(b_rot - 1)."""
    rng = np.random.default_rng(seed)
    bound = 2 ** (p.b_rot - 1)
    tail = tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.N))
    sf = tuple(tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.B))
               for _ in range(p.R))
    return tail, sf


def register_value(angles, b_rot):
    """The angle registers' big-endian integer when register k holds angles[k] (spec 6.7), written
    out bit by bit, independently of the module's angle_entry."""
    return int("".join(format(angle, f"0{b_rot}b") for angle in angles), 2)


def vendored_tables(system, p):
    tables = WalkTables.from_factors(load_factors(system), p)
    return tables.tail_angles, tables.sf_angles


RANDOM_R3 = WalkParams(N=3, R=3, B=3, C=1, n_plus=1, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
RANDOM_R1 = WalkParams(N=2, R=1, B=2, C=2, n_plus=2, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
SHAPES = {
    "sto3g": (H2_STO3G_TEST, vendored_tables("h2_sto3g_R2_B2_C1", H2_STO3G_TEST)),
    "631g": (H2_631G_TEST, vendored_tables("h2_631g_R2_B2_C2", H2_631G_TEST)),
    "random-R3": (RANDOM_R3, random_tables(RANDOM_R3)),
    "random-R1": (RANDOM_R1, random_tables(RANDOM_R1)),
}


def layout(p, tail, sf, ob):
    """Wires: x_o, r, b, angles, G0, G1, b_neq_B, then work for the largest of the three."""
    sizes = {"x_o": p.w_x, "r": p.w_r, "b": p.w_b, "angles": (p.N - 1) * p.b_rot, "G0": 1,
             "G1": 1, "b_neq_B": 1}
    works = (TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot)["work"],
             SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot)["work"],
             AngleErase.required_wires(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot,
                                       ob)["work"])
    sizes["work"] = max(works)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start, works


def loads(p, tail, sf, wires, works):
    TailAngleLoad(tail, p.n_plus, p.RC, p.X_o, p.b_rot, wires["x_o"], wires["angles"],
                  wires["G0"], wires["G1"], wires["work"][:works[0]])
    SFAngleLoad(sf, p.R, p.B, p.b_rot, wires["r"], wires["b"], wires["G0"], wires["angles"],
                wires["b_neq_B"], wires["work"][:works[1]])


def make_erase(p, tail, sf, ob, wires, works):
    return AngleErase(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot, ob, wires["angles"],
                      wires["b_neq_B"], wires["G0"], wires["G1"], wires["x_o"], wires["r"],
                      wires["b"], wires["work"][:works[2]])


def expected_cost(p, ob):
    """The erase's closed form for any SF one-hot width ob: the SF fix-up (enabled) over
    sf_fixup_high_ranges plus the tail fix-up (no one-hot, no enable); M = (N - 1) b_rot + 3."""
    sf = costs.phase_fixup(AngleErase.sf_fixup_high_ranges(p.R, p.B, ob), ob, True)
    tail = costs.phase_fixup(AngleErase.tail_fixup_high_ranges(p.N, p.R, p.C), 0, False)
    return costs.Cost(sf.toffoli + tail.toffoli, (p.N - 1) * p.b_rot + 3)


def compiled_erase(p, tail, sf, ob):
    wires, num_wires, works = layout(p, tail, sf, ob)
    return compiled_resources(lambda: make_erase(p, tail, sf, ob, wires, works), num_wires)


def assert_counts(resources, expected):
    """T4 and the T6 signature: T and M exact, A = T (every AND of both fix-ups is uncomputed by
    measurement; the erase has no plain Toffoli)."""
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measured_and_uncomputes == resources.toffoli


@pytest.mark.compiled
@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_counts_at_the_test_presets(name):
    """T4 at H2_STO3G_TEST and H2_631G_TEST (vendored tables, ob = w_b): dfthc.costs.angle_erase,
    M = (N - 1) b_rot + 3 (7 and 15)."""
    p, (tail, sf) = SHAPES[name]
    assert_counts(compiled_erase(p, tail, sf, p.w_b), costs.angle_erase(p))


@pytest.mark.compiled
@pytest.mark.slow
def test_counts_at_mid_naphthalene():
    """T4 at MID_NAPHTHALENE (naphthalene tables, nightly tier): T = 26, M = 138."""
    p = MID_NAPHTHALENE
    assert_counts(compiled_erase(p, *vendored_tables("naphthalene", p), p.w_b),
                  costs.angle_erase(p))


@pytest.mark.compiled
@pytest.mark.parametrize(("name", "ob"), [("random-R3", 0), ("random-R3", 1), ("random-R1", 2)],
                         ids=["R3-ob0", "R3-ob1", "R1-ob2"])
def test_counts_with_other_one_hot_widths(name, ob):
    """T4 off the walk's ob = w_b: the ob = 0 fallback (conditional Z per SF address), a split
    one-hot (ob = 1: r and the high bit of b iterated), and an SF address of b alone (R = 1)."""
    p, (tail, sf) = SHAPES[name]
    assert_counts(compiled_erase(p, tail, sf, ob), expected_cost(p, ob))


def test_fixup_high_ranges_agree_with_the_cost_model():
    """The erase's fix-up ranges are the oracle of dfthc.costs.angle_erase: at ob = w_b it equals
    the two fix-ups' closed forms at the presets, FeMoco-54 included, and on a grid of shapes with
    R from 1 to 6, B from 1 to 8, N from 2 to 5, C from 1 to 3."""
    assert AngleErase.sf_fixup_high_ranges(10, 27, 5) == [(0, 10, 4)]
    assert AngleErase.tail_fixup_high_ranges(54, 10, 27) == [(270, 324, 9)]
    grid = [WalkParams(N=n, R=r, B=b, C=c, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=0, k2=0,
                       ke1=0, ke2=0)
            for n, r, b, c in itertools.product(range(2, 6), range(1, 7), range(1, 9), (1, 2, 3))]
    for p in [FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE, *grid]:
        assert costs.angle_erase(p) == expected_cost(p, p.w_b), p


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 on the closed forms: T = (11 + 30) + 61 = 102, M = 53 * 15 + 3 = 798; work is the
    tail fix-up's 8 wires (the SF fix-up needs 4)."""
    p = FEMOCO54
    assert costs.angle_erase(p) == costs.Cost(11 + 30 + 61, 798) == costs.Cost(102, 798)
    tail = tuple((0,) * 53 for _ in range(54))
    sf = tuple(tuple((0,) * 53 for _ in range(27)) for _ in range(10))
    assert AngleErase.required_wires(tail, sf, 27, 54, 10, 27, 27, 15, 5) == {
        "angles": 795, "b_neq_B": 1, "G0": 1, "G1": 1, "x_o": 9, "r": 4, "b": 5, "work": 8}


@pytest.mark.compiled
def test_erase_is_not_a_second_load_and_its_conditionals_are_hinted(capfd):
    """T6: the chosen rule is the pinned measured erase, M counts every erased qubit, its T
    differs from re-running both loads, and its conditionals count their hinted expectation:
    R B conditional CZs (one per SF address; no mask is zero) and N conditional Zs (one per tail
    row) give ceil(R B / 2) and ceil(N / 2)."""
    p, (tail, sf) = SHAPES["631g"]
    wires, num_wires, works = layout(p, tail, sf, p.w_b)
    resources = compiled_resources(lambda: make_erase(p, tail, sf, p.w_b, wires, works),
                                   num_wires)
    assert resources.measurements == len(wires["angles"]) + 3
    assert resources.toffoli != (costs.tail_angle_load(p) + costs.sf_angle_load(p)).toffoli
    assert resources.counts["CZ"] == math.ceil(p.R * p.B / 2)
    assert resources.counts["PauliZ"] == math.ceil(p.N / 2)
    qp.analyze(verbose_program(lambda: make_erase(p, tail, sf, p.w_b, wires, works), num_wires),
               level="user")()
    captured = capfd.readouterr()
    assert "_angle_erase" in chosen_rules(captured.out + captured.err)


@pytest.mark.compiled
@pytest.mark.parametrize("wrap", [qp.adjoint, lambda op: qp.ctrl(op, control=40)],
                         ids=["adjoint", "control"])
def test_erase_has_no_adjoint_or_controlled_path(wrap):
    """Spec 7.4 rule 5: never adjointed or controlled. Catalyst drops the measuring candidate
    with the expected warning (rule 12), and without another rule the compile fails loudly."""
    p, (tail, sf) = SHAPES["sto3g"]
    wires, _, works = layout(p, tail, sf, p.w_b)
    key = "Adjoint(AngleErase)" if wrap is qp.adjoint else "C(AngleErase)"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(lambda: wrap(make_erase(p, tail, sf, p.w_b, wires, works)),
                               num_wires=41)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the _angle_erase")]
    assert skipped and {classify_rule_warning(m) for m in skipped} == {"expected"}


def phased_prepare(wires):
    """H on each wire, S on even positions, CZ on consecutive pairs (non-uniform phases)."""
    for wire in wires:
        qp.H(wire)
    for position, wire in enumerate(wires):
        if position % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


SEEDS = [0, *(pytest.param(seed, marks=pytest.mark.slow) for seed in range(1, 5))]


@pytest.mark.compiled
@pytest.mark.parametrize("seed", SEEDS)
def test_lightning_smoke_run(seed):
    """Compiled loads then erase at H2_STO3G_TEST (15 wires) restore a phased state of x_o, r and
    b and clear every ancilla, for five measurement seeds (seeds 1 to 4 in the nightly tier;
    the certificates prove every outcome, this runs the compiled path)."""
    p, (tail, sf) = SHAPES["sto3g"]
    wires, num_wires, works = layout(p, tail, sf, p.w_b)
    inputs = wires["x_o"] + wires["r"] + wires["b"]

    def program(round_trip):
        @qp.qjit(capture=True, seed=seed)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            phased_prepare(inputs)
            if round_trip:
                loads(p, tail, sf, wires, works)
                make_erase(p, tail, sf, p.w_b, wires, works)
            return qp.state()

        return np.asarray(circuit())

    assert np.allclose(program(True), program(False), rtol=0, atol=1e-10)


def erase_masks(p, tail, sf):
    """The two fix-ups' masks as the erase builds them: SF over angles ‖ b_neq_B, tail over
    angles ‖ b_neq_B ‖ G0 ‖ G1 (big-endian)."""
    sf_masks = [(register_value(sf[r][b], p.b_rot) << 1) | 1
                for r in range(p.R) for b in range(p.B)]
    tail_masks = [(register_value(row, p.b_rot) << 3) | 2 | int(j >= p.n_plus)
                  for j, row in enumerate(tail)]
    return [(sf_masks, (p.N - 1) * p.b_rot + 1), (tail_masks, (p.N - 1) * p.b_rot + 3)]


def test_no_mask_is_zero_and_parities_are_linear():
    """Spec 12.5's linearity for the erase's masks, which are never zero (the b_neq_B and G0
    bits): the capture core at the FeMoco-54 shapes (270 SF masks over 796 outcomes, 54 tail
    masks over 798, random tables with a zero angle entry in each; classical, nothing compiled)
    and the tape implementation at H2_631G_TEST, both with f(0) = 0 and f(x ^ y) = f(x) ^ f(y)
    for 100 random pairs."""
    p = FEMOCO54
    tail, sf = random_tables(p, seed=8)
    tail = ((0,) * 53,) + tail[1:]
    sf = (((0,) * 53,) + sf[0][1:],) + sf[1:]
    rng = np.random.default_rng(12)
    for masks, num_outcomes in erase_masks(p, tail, sf):
        assert all(masks) and len(masks) in (270, 54)
        packed = pack_masks(masks, num_outcomes)
        assert not np.asarray(packed_parities(jnp.zeros(num_outcomes, dtype=jnp.uint32),
                                              packed)).any()
        for _ in range(100):
            x, y = rng.integers(0, 2, size=(2, num_outcomes))
            fx, fy = (np.asarray(packed_parities(v, packed)) for v in (x, y))
            assert np.array_equal(np.asarray(packed_parities(x ^ y, packed)), fx ^ fy)
    p, (tail, sf) = SHAPES["631g"]
    for masks, num_outcomes in erase_masks(p, tail, sf):
        with qp.queuing.AnnotatedQueue():
            outcomes = [qp.measure(w) for w in range(num_outcomes)]
            parities = mcm_parities(outcomes[::-1], masks)

        def evaluate(bits, outcomes=outcomes, parities=parities):
            record = {o.measurements[0]: int(b) for o, b in zip(outcomes, bits, strict=True)}
            return np.array([int(f.concretize(record)) for f in parities])

        assert not evaluate(np.zeros(num_outcomes, dtype=int)).any()
        for _ in range(100):
            x, y = rng.integers(0, 2, size=(2, num_outcomes))
            assert np.array_equal(evaluate(x ^ y), evaluate(x) ^ evaluate(y))


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    p, (tail, sf) = SHAPES["sto3g"]
    wires, num_wires, works = layout(p, tail, sf, p.w_b)
    compiled_resources(lambda: make_erase(p, tail, sf, p.w_b, wires, works), num_wires,
                       level="device")


@pytest.mark.parametrize(("name", "ob"), [("sto3g", 2), ("random-R3", 1), ("random-R1", 0)])
def test_rule_is_valid_in_tape_mode(name, ob):
    """T2 in tape mode at three sizes: the outcomes and their parities are MeasurementValues,
    and the counting twin declares exactly the emitted gates. The pinned rule is the only rule
    registered under the class name (spec 7.4 rule 8)."""
    assert list(qp.list_decomps(AngleErase)) == [_angle_erase]
    p, (tail, sf) = SHAPES[name]
    wires, _, works = layout(p, tail, sf, ob)
    qp.testing.assert_valid_decomp_rule(make_erase(p, tail, sf, ob, wires, works), _angle_erase,
                                        check_matrix=False)


def erase_args(**changes):
    """Valid AngleErase arguments (N = 2, R = B = 2, C = 1, b_rot = 3, ob = 2), with changes."""
    return {"tail_angles": ((1,), (2,)), "sf_angles": (((1,), (2,)), ((3,), (0,))), "n_plus": 1,
            "N": 2, "R": 2, "B": 2, "C": 1, "b_rot": 3, "sf_onehot_bits": 2,
            "angles": [0, 1, 2], "b_neq_B": [3], "G0": [4], "G1": [5], "x_o": [6, 7], "r": [8],
            "b": [9, 10], "work": [11]} | changes


def test_required_wires():
    args = erase_args()
    assert AngleErase.required_wires(*list(args.values())[:9]) == {
        "angles": 3, "b_neq_B": 1, "G0": 1, "G1": 1, "x_o": 2, "r": 1, "b": 2, "work": 1}
    AngleErase(**args)


@pytest.mark.parametrize(("change", "match"), [
    ({"sf_onehot_bits": 3}, r"sf_onehot_bits must lie in \[0, 2\]"),
    ({"b_rot": 2, "angles": [0, 1], "tail_angles": ((1,), (0,)),
      "sf_angles": (((1,), (0,)), ((0,), (1,)))}, "needs 4 wires, more than the 3 reset angle"),
    ({"N": 3}, "tail_angles must have N = 3 rows"),
    ({"tail_angles": 3}, "tail_angles must have N = 2 rows"),
    ({"sf_angles": (((1,), (4,)), ((3,), (0,)))}, r"angles must lie in \[0, 2\*\*\(b_rot - 1\)"),
    ({"work": []}, "'work' has 0 wires; expected 1"),
])
def test_constructor_validates(change, match):
    with pytest.raises(ValueError, match=match):
        AngleErase(**erase_args(**change))
