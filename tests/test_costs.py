"""dfthc.costs against every row of the spec 9.24 evaluation table at FeMoco-54."""

import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from dfthc import costs, costs_reference
from dfthc.costs import Cost
from dfthc.data.factors import load_factors
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams, lg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
from prototypes import unary_toffolis  # noqa: E402

P = FEMOCO54
H2 = H2_STO3G_TEST
ODD = WalkParams(N=3, R=3, B=3, C=1, n_plus=1, b_rot=5, b_k1=3, b_k2=4, s_u=3,
                 k1=1, k2=1, ke1=1, ke2=1)
"""A second and a third shape, derived by hand from spec 9.24 (derivations in the Task 1A.7
report). ``ODD`` has ``X_o = 6``, ``w_x = 3``, ``w_b = 2``, ``w_r = 2``, ``d1 = 6``, ``d2 = 10``
and ``n_plus != N / 2``."""


def test_unary_iteration_rows():
    assert costs.unary(0, 81, 7, controlled=False) == 84  # outer read
    assert costs.unary_nested((0, 270, 9), (0, 2, 1), controlled=False) == 543  # 273 + 270 * 1
    assert costs.unary(0, 270, 9, controlled=False) == 273
    assert costs.unary(0, 21, 5, controlled=False) == 22  # outer erase high part
    assert costs.unary(0, 68, 7, controlled=False) == 70  # inner erase high part
    assert costs.unary(270, 324, 9, controlled=False) == 61  # tail range
    assert costs.unary(0, 10, 4, True) + 10 * costs.unary(0, 27, 5, True) == 291  # SF
    assert costs.unary(0, 1, 0, controlled=True) == 0
    # A zero-width outer register without control gives the inner root no flag: c2 = 0.
    assert costs.unary_nested((0, 1, 0), (0, 8, 3), controlled=False) == costs.unary(0, 8, 3, False)
    assert costs.unary_nested((0, 1, 0), (0, 8, 3), controlled=True) == costs.unary(0, 8, 3, True)


def test_unary_equals_the_prototype_on_a_grid():
    """200 random (lo, hi, n, c) with n <= 8 against spikes/prototypes.unary_toffolis (Task 0.7)."""
    rng = np.random.default_rng(924)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        lo, hi = sorted(int(v) for v in rng.choice(2**n + 1, 2, replace=False))
        controlled = bool(rng.integers(2))
        assert costs.unary(lo, hi, n, controlled) == unary_toffolis(lo, hi, n, controlled)


def test_truncated_unary_and_onehot():
    assert [costs.truncated_unary(L, c) for L, c in [(81, False), (81, True), (2, False)]] == [
        79, 80, 0]
    assert costs.truncated_unary(1, False) == costs.truncated_unary(1, True) == 0
    assert [costs.truncated_unary(L, c) for L, c in [(2, True), (3, False), (3, True)]] == [1, 1, 2]
    with pytest.raises(ValueError, match="L >= 1"):
        costs.truncated_unary(0, False)
    with pytest.raises(ValueError, match="L >= 1"):
        costs.truncated_unary(0, True)
    assert [costs.onehot(ob) for ob in (0, 1, 2, 4, 7)] == [0, 0, 2, 14, 126]


def test_uniform_preparation_rows():
    assert costs.flagged_uniform_prep(324, 7) == Cost(30)
    assert costs.flagged_uniform_unprep(324, 7) == Cost(24)
    assert costs.flagged_uniform_prep(28, 7) == Cost(18)
    assert costs.flagged_uniform_unprep(28, 7) == Cost(16)
    assert costs.flagged_uniform_prep(8, 7) == costs.flagged_uniform_unprep(8, 7) == Cost(0)
    assert costs.flagged_uniform_prep(5, 3) == Cost(3 * 3 + 6 - 5)


def test_phase_gradient_rotation_row():
    assert costs.phase_gradient_rotation(0, 5) == Cost(0)
    assert costs.phase_gradient_rotation(1, 5) == costs.phase_gradient_rotation(31, 5) == Cost(4)


def test_read_and_erase_rows():
    assert costs.qroam_read([(0, 81, 7)], 2, 24, controlled=False) == Cost(156)
    assert costs.qroam_read([(0, 270, 9), (0, 2, 1)], 4, 26, controlled=False) == Cost(933)
    assert costs.phase_fixup([(0, 21, 5)], 4, controlled=False) == Cost(22 + 14)
    assert costs.phase_fixup([(0, 68, 7)], 7, controlled=False) == Cost(70 + 126)
    assert costs.qroam_erase([(0, 21, 5)], 2, 24, 4, controlled=False) == Cost(36, 96)
    assert costs.qroam_erase([(0, 68, 7)], 4, 26, 7, controlled=False) == Cost(196, 416)


X_O_5 = dataclasses.replace(H2_STO3G_TEST, N=3, n_plus=1, R=1, C=2, k1=2, ke1=0)
"""X_o = 5 with k1 = 2 > ke1 = 0: the outer read's last block [4, 8) holds dead positions 5 to 7."""


def test_outer_erase_covers_the_loaded_set():
    """The outer erase's fixup covers the read's loaded set [0, ceil(X_o / K1) K1), dead positions
    of the last block included (spec 9.5): at X_o = 5, k1 = 2, ke1 = 0 its high range is [0, 8),
    not [0, 5), so T = w_x + U(0, 8, 3, 0) + F^dag(5, 3) = 3 + 6 + 8 and M = K1 d1 = 4 * 5."""
    assert X_O_5.X_o == 5
    assert costs.outer_unprepare(X_O_5) == Cost(3 + 6 + 8, 4 * 5)
    for n, k1, ke1 in itertools.product(range(3, 11), range(4), range(4)):
        p = dataclasses.replace(X_O_5, N=n, k1=k1, ke1=ke1)
        loaded = -(-p.X_o // p.K1) * p.K1
        fixup = costs.unary(0, -(-loaded // 2**ke1), p.w_x - ke1, False) + costs.onehot(ke1)
        expected = p.w_x + fixup + costs.flagged_uniform_unprep(p.X_o, p.s_u).toffoli
        assert costs.outer_unprepare(p).toffoli == expected, (p.X_o, k1, ke1)


def test_block_rows():
    assert (costs.outer_prepare(P), costs.outer_unprepare(P)) == (Cost(210), Cost(69, 96))
    assert (costs.inner_prepare(P), costs.inner_unprepare(P)) == (Cost(971), Cost(217, 416))
    assert (costs.tail_angle_load(P), costs.sf_angle_load(P)) == (Cost(61), Cost(291))
    assert (costs.spin_mux(), costs.spin_swap(P)) == (Cost(1), Cost(54))
    assert (costs.givens_rotation(P), costs.givens_ladder(P)) == (Cost(28), Cost(1484))
    assert costs.majorana_gate() == Cost(5)
    assert costs.angle_erase(P) == Cost(102, 798)
    assert costs.inner_select(P) == Cost(3537, 798)
    # the second and third shape, derived by hand (report)
    assert (costs.outer_prepare(H2), costs.outer_unprepare(H2)) == (Cost(8), Cost(2, 8))
    assert (costs.inner_prepare(H2), costs.inner_unprepare(H2)) == (Cost(21), Cost(14, 14))
    assert (costs.tail_angle_load(H2), costs.sf_angle_load(H2)) == (Cost(1), Cost(5))
    assert (costs.spin_swap(H2), costs.givens_rotation(H2), costs.givens_ladder(H2)) == (
        Cost(2), Cost(6), Cost(6))
    assert costs.angle_erase(H2) == Cost(4, 7)
    assert costs.inner_select(H2) == Cost(33, 7)
    assert (costs.outer_prepare(ODD), costs.outer_unprepare(ODD)) == (Cost(21), Cost(11, 12))
    assert (costs.inner_prepare(ODD), costs.inner_unprepare(ODD)) == (Cost(22), Cost(9, 20))
    assert (costs.tail_angle_load(ODD), costs.sf_angle_load(ODD)) == (Cost(4), Cost(12))
    assert (costs.spin_swap(ODD), costs.givens_rotation(ODD), costs.givens_ladder(ODD)) == (
        Cost(3), Cost(8), Cost(16))
    assert costs.angle_erase(ODD) == Cost(9, 13)
    assert costs.inner_select(ODD) == Cost(70, 13)


def test_reflection_rows():
    assert (costs.ref_t2(P, False), costs.ref_t2(P, True)) == (Cost(20), Cost(21))
    assert (costs.ref_ab(P, False), costs.ref_ab(P, True)) == (Cost(44), Cost(45))
    assert [costs.reflection(n, False).toffoli for n in (1, 2, 3, 6, 46)] == [0, 0, 1, 4, 44]
    assert [costs.reflection(n, True).toffoli for n in (1, 2, 3, 6, 46)] == [0, 1, 2, 5, 45]
    with pytest.raises(ValueError, match="n_w"):
        costs.reflection(0, False)
    for params, t2, ab in ((H2, 4, 8), (ODD, 6, 12)):  # n_w = 6, 10 and 8, 14
        assert (costs.ref_t2(params, False), costs.ref_t2(params, True)) == (
            Cost(t2), Cost(t2 + 1))
        assert (costs.ref_ab(params, False), costs.ref_ab(params, True)) == (
            Cost(ab), Cost(ab + 1))


def test_totals():
    assert costs.block_encoding(P, False) == Cost(9749, 2524)
    assert costs.block_encoding(P, True) == Cost(9750, 2524)
    assert costs.walk(P, False) == Cost(9793, 2524)
    assert costs.walk(P, True) == Cost(9795, 2524)
    assert costs.toffoli_only(P) == 18 + 20 + 72 + 780 + 216 + 12 == 1118
    assert costs.measured_and_uncomputes(P, False) == 8675
    assert costs.measured_and_uncomputes(P, True) == 8677
    assert costs.walk(P, False).measurements + costs.measured_and_uncomputes(P, False) == 11199
    assert costs.walk(P, True).measurements + costs.measured_and_uncomputes(P, True) == 11201
    # H2: T_Toffoli = 4 + 8 + 4 + 14 + 8 + 12; ODD: 6 + 8 + 6 + 20 + 12 + 12 (report)
    assert costs.block_encoding(H2, False) == Cost(150, 50)
    assert costs.block_encoding(H2, True) == Cost(151, 50)
    assert (costs.walk(H2, False), costs.walk(H2, True)) == (Cost(158, 50), Cost(160, 50))
    assert costs.toffoli_only(H2) == 4 + 8 + 4 + 14 + 8 + 12 == 50
    assert [costs.measured_and_uncomputes(H2, c) for c in (False, True)] == [108, 110]
    assert costs.block_encoding(ODD, False) == Cost(240, 78)
    assert costs.block_encoding(ODD, True) == Cost(241, 78)
    assert (costs.walk(ODD, False), costs.walk(ODD, True)) == (Cost(252, 78), Cost(254, 78))
    assert costs.toffoli_only(ODD) == 6 + 8 + 6 + 20 + 12 + 12 == 64
    assert [costs.measured_and_uncomputes(ODD, c) for c in (False, True)] == [188, 190]


def test_application_rows():
    cw, w = costs.walk(P, True), costs.walk(P, False)
    assert costs.controlled_walk_power(P, 4) == Cost(4 * cw.toffoli, 4 * w.measurements)
    assert costs.qpe_round(P, 0, 3) == Cost(cw.toffoli + 1 + 2, w.measurements)
    assert costs.qpe_round(P, 2, 3) == Cost(4 * cw.toffoli, 4 * w.measurements)
    for t in (1, 2, 3, 5):
        expected = Cost((2**t - 1) * cw.toffoli + (t**3 - t) // 6, (2**t - 1) * w.measurements + t)
        assert costs.iterative_qpe(P, t) == expected
        assert sum(costs.qpe_round(P, j, t).toffoli for j in range(t)) == expected.toffoli
    # literal values at the small shapes: round 1 of 3 bits is 2 T_C(W) + 1 (report)
    assert costs.qpe_round(H2, 1, 3) == Cost(2 * 160 + 1, 2 * 50)
    assert costs.iterative_qpe(H2, 3) == Cost(7 * 160 + 4, 7 * 50 + 3) == Cost(1124, 353)
    assert costs.qpe_round(ODD, 1, 3) == Cost(2 * 254 + 1, 2 * 78) == Cost(509, 156)
    assert costs.iterative_qpe(ODD, 2) == Cost(3 * 254 + 1, 3 * 78 + 2) == Cost(763, 236)


def test_cheapest_block_bits_is_a_brute_force_minimizer():
    """Over every (k1, k2, ke1, ke2) whose erase one-hots fit (2**ke1 <= K1 d1 and
    2**(ke2 + w_b) <= K2 d2), also at shapes whose X_o is not a multiple of K1."""
    for params in (P, H2_STO3G_TEST, MID_NAPHTHALENE, X_O_5):
        best = costs.cheapest_block_bits(params)
        grid = [(a, b, c, d) for a, b, c, d in itertools.product(
                    range(params.w_x + 1), range(params.w_b + 1), range(params.w_x + 1),
                    range(lg(params.RC) + 1))
                if 2**c <= 2**a * params.d1 and 2 ** (d + params.w_b) <= 2**b * params.d2]
        brute = min(costs.walk(dataclasses.replace(params, k1=a, k2=b, ke1=c, ke2=d), False).toffoli
                    for a, b, c, d in grid)
        chosen = dataclasses.replace(params, k1=best[0], k2=best[1], ke1=best[2], ke2=best[3])
        assert costs.walk(chosen, False).toffoli == brute
    assert costs.cheapest_block_bits(P) == (2, 4, 4, 2)  # the FEMOCO54 preset's block bits
    assert costs.cheapest_block_bits(MID_NAPHTHALENE) == (1, 2, 3, 1)  # the preset's, dfthcbe's
    # X_O_5 ties (0, 0, 1, 0) with (0, 0, 2, 0) at T = 7 for the outer read and erase: smallest wins
    assert costs.cheapest_block_bits(X_O_5) == (0, 0, 1, 0)


REPORTED_BLOCK_BITS = {
    "h2_sto3g_R2_B2_C1": (0, 0, 1, 0), "h2_631g_R2_B2_C2": (0, 0, 1, 0),
    "h2_631g_R2_B3_C3": (0, 0, 2, 0), "naphthalene": (1, 2, 3, 1), "fe2s2": (1, 3, 3, 1),
    "fe4s4": (1, 3, 4, 1),
}
"""``(k1, k2, ke1, ke2)`` of ``costs_reference.default_params`` for every vendored system, as
recorded before final review m-1 (the report's and the dfthcbe fixture's parameters)."""
PRESET_BLOCK_BITS = [(FEMOCO54, (2, 4, 4, 2)), (H2_STO3G_TEST, (0, 0, 1, 0)),
                     (H2_631G_TEST, (0, 1, 1, 0)), (MID_NAPHTHALENE, (1, 2, 3, 1))]


def test_cheapest_block_bits_fit_the_erases_and_keep_every_reported_default():
    """Final review m-1: the minimizer keeps the erases' one-hots inside their registers,
    2**ke1 <= K1 d1 and 2**(ke2 + w_b) <= K2 d2; at (N, R, B, C) = (10, 1, 16, 1) the free
    choice (0, 0, 2, 0) needed 32 > K2 d2 = 22 wires. The block bits of every vendored system
    and the minimizer's result at every preset are unchanged."""
    p = costs_reference.default_params(10, 1, 16, 1, 5)
    assert 2**p.ke1 <= p.K1 * p.d1 and 2 ** (p.ke2 + p.w_b) <= p.K2 * p.d2
    for system, bits in REPORTED_BLOCK_BITS.items():
        f = load_factors(system)
        p = costs_reference.default_params(f.N, f.R, f.B, f.C, f.n_plus)
        assert (p.k1, p.k2, p.ke1, p.ke2) == bits, system
    for preset, bits in PRESET_BLOCK_BITS:
        assert costs.cheapest_block_bits(preset) == bits, preset


def test_cost_values_are_python_ints():
    for value in dataclasses.astuple(costs.walk(P, True)) + (costs.toffoli_only(P),):
        assert type(value) is int


def test_invalid_ranges_are_rejected():
    with pytest.raises(ValueError, match="range"):
        costs.unary(4, 4, 3, controlled=False)
    with pytest.raises(ValueError, match="range"):
        costs.unary(0, 9, 3, controlled=False)
    with pytest.raises(ValueError, match="one or two"):
        costs.qroam_read([], 1, 2, controlled=False)
