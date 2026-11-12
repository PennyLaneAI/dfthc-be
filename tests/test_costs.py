"""dfthc.costs against every row of the spec 9.24 evaluation table at FeMoco-54."""

import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from dfthc import costs
from dfthc.costs import Cost
from dfthc.params import FEMOCO54, H2_STO3G_TEST, MID_NAPHTHALENE, lg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
from prototypes import unary_toffolis  # noqa: E402

P = FEMOCO54


def test_unary_iteration_rows():
    assert costs.unary(0, 81, 7, controlled=False) == 84  # outer read
    assert costs.unary_nested((0, 270, 9), (0, 2, 1), controlled=False) == 543  # 273 + 270 * 1
    assert costs.unary(0, 270, 9, controlled=False) == 273
    assert costs.unary(0, 21, 5, controlled=False) == 22  # outer erase high part
    assert costs.unary(0, 68, 7, controlled=False) == 70  # inner erase high part
    assert costs.unary(270, 324, 9, controlled=False) == 61  # tail range
    assert costs.unary(0, 10, 4, True) + 10 * costs.unary(0, 27, 5, True) == 291  # SF
    assert costs.unary(0, 1, 0, controlled=True) == 0


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


def test_reflection_rows():
    assert (costs.ref_t2(P, False), costs.ref_t2(P, True)) == (Cost(20), Cost(21))
    assert (costs.ref_ab(P, False), costs.ref_ab(P, True)) == (Cost(44), Cost(45))
    assert [costs.zero_reflection(n, False).toffoli for n in (1, 2, 3, 6, 46)] == [0, 0, 1, 4, 44]
    assert [costs.zero_reflection(n, True).toffoli for n in (1, 2, 3, 6, 46)] == [0, 1, 2, 5, 45]
    with pytest.raises(ValueError, match="n_w"):
        costs.zero_reflection(0, False)


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


def test_application_rows():
    cw, w = costs.walk(P, True), costs.walk(P, False)
    assert costs.controlled_walk_power(P, 4) == Cost(4 * cw.toffoli, 4 * w.measurements)
    assert costs.qpe_round(P, 0, 3) == Cost(cw.toffoli + 1 + 2, w.measurements)
    assert costs.qpe_round(P, 2, 3) == Cost(4 * cw.toffoli, 4 * w.measurements)
    for t in (1, 2, 3, 5):
        expected = Cost((2**t - 1) * cw.toffoli + (t**3 - t) // 6, (2**t - 1) * w.measurements + t)
        assert costs.iterative_qpe(P, t) == expected
        assert sum(costs.qpe_round(P, j, t).toffoli for j in range(t)) == expected.toffoli


def test_cheapest_block_bits_is_a_brute_force_minimizer():
    """Over every (k1, k2, ke1, ke2), also at shapes whose X_o is not a multiple of K1."""
    for params in (P, H2_STO3G_TEST, MID_NAPHTHALENE, X_O_5):
        best = costs.cheapest_block_bits(params)
        grid = itertools.product(range(params.w_x + 1), range(params.w_b + 1),
                                 range(params.w_x + 1), range(lg(params.RC) + 1))
        brute = min(costs.walk(dataclasses.replace(params, k1=a, k2=b, ke1=c, ke2=d), False).toffoli
                    for a, b, c, d in grid)
        chosen = dataclasses.replace(params, k1=best[0], k2=best[1], ke1=best[2], ke2=best[3])
        assert costs.walk(chosen, False).toffoli == brute
    assert costs.cheapest_block_bits(P) == (2, 4, 4, 2)  # the FEMOCO54 preset's block bits
    assert costs.cheapest_block_bits(MID_NAPHTHALENE) == (1, 2, 3, 1)  # the preset's, dfthcbe's


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
