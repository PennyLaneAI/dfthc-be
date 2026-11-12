"""Sum-of-squares quantities and the two offset frames: spec 8.6 checks 1 to 3 (and 3.2.1)."""

import numpy as np
import pytest

from dfthc.data.factors import SYSTEMS, load_factors, load_integrals, load_summary
from dfthc.data.fermion import sector_projector
from dfthc.data.sos import (
    e_sos_phys,
    e_sos_walk,
    frame_shift,
    h_phys_dense,
    h_sa_dense,
    h_walk_dense,
    inner_weights,
    lambda_eff,
    normalization,
    outer_weights,
    walk_integrals,
)

FACTORS_ERROR = {"h2_sto3g_R2_B2_C1": 1e-10, "h2_631g_R2_B2_C2": 3e-2, "h2_631g_R2_B3_C3": 1e-3}
"""dfthcbe's per-instance factorization tolerances (spec 8.6, check 3)."""

LAMBDA_TOLERANCE = {system: 1e-10 for system in SYSTEMS} | {"fe4s4": 2e-10}
"""fe4s4's stored lambda_h1 exceeds sum(w_plus) + sum(w_minus) by 5.96e-9, so Lambda from its
factors is 1.18e-10 (relative) below lambda_sos: 2e-10 leaves a margin of 1.7. Every other system
agrees to 1e-13."""


def _sector(matrix, n, eta):
    keep = np.flatnonzero(np.diag(sector_projector(n, eta)).real)
    return matrix[np.ix_(keep, keep)]


@pytest.mark.parametrize("system", SYSTEMS)
def test_scalars_match_the_summary(system):
    """Check 1: Lambda, lambda_eff and e_sos against summary.json, relative 1e-10."""
    factors, summary = load_factors(system), load_summary(system)
    Lambda, rel = normalization(factors), LAMBDA_TOLERANCE[system]
    assert Lambda == pytest.approx(summary["lambda_sos"], rel=rel)
    assert lambda_eff(Lambda, summary["e_gap"]) == pytest.approx(summary["lambda_eff"], rel=rel)
    assert e_sos_phys(summary) == summary["e_sos"]
    assert e_sos_phys(summary) == pytest.approx(summary["e_upper"] - summary["e_gap"], rel=1e-10)


def test_weights_layout():
    """SF first at x_o = r C + c, then 2 w_+ (D1) and 2 w_- (Q1); inner rows signed, w_B last."""
    f = load_factors("h2_631g_R2_B3_C3")
    outer, inner = outer_weights(f), inner_weights(f)
    assert outer.shape == (f.N + f.R * f.C,) and inner.shape == (f.R * f.C, f.B + 1)
    for r, c in np.ndindex(f.R, f.C):
        row = np.append(f.W[r, :, c], f.w_B[r, c])
        assert np.array_equal(inner[r * f.C + c], row)
        assert outer[r * f.C + c] == pytest.approx(np.abs(row).sum() ** 2 / 2, rel=1e-15)
    assert np.array_equal(outer[f.R * f.C:], 2 * np.concatenate([f.w_plus, f.w_minus]))
    assert normalization(f) == pytest.approx(outer.sum() / 2, rel=1e-15)


def test_walk_offset_formula():
    f = load_factors("h2_631g_R2_B2_C2")
    identity = f.w_B - f.W.sum(axis=1)
    assert e_sos_walk(f) == pytest.approx(-2 * f.w_minus.sum() - (identity**2).sum() / 2,
                                          rel=1e-15)


def test_walk_integrals_have_the_eightfold_symmetry():
    h1e, eri = walk_integrals(load_factors("h2_631g_R2_B3_C3"))
    assert np.allclose(h1e, h1e.T, rtol=0, atol=1e-12)
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        assert np.allclose(eri, eri.transpose(perm), rtol=0, atol=1e-12)


@pytest.mark.parametrize("system", FACTORS_ERROR)
def test_walk_frame_on_the_whole_fock_space(system):
    """Check 2: the generators' H_SA equals H_walk - E_SoS^walk to 1e-10 and is PSD."""
    f = load_factors(system)
    h_sa = h_sa_dense(f)
    assert h_sa.shape == (4**f.N, 4**f.N)
    assert np.allclose(h_sa, h_walk_dense(f) - e_sos_walk(f) * np.eye(4**f.N), rtol=0, atol=1e-10)
    assert np.allclose(h_sa, h_sa.conj().T, rtol=0, atol=1e-12)
    assert np.linalg.eigvalsh(h_sa).min() >= -1e-10


@pytest.mark.parametrize("system", FACTORS_ERROR)
def test_lowest_sector_eigenvalue_is_the_gap(system):
    """Check 3: on the eta-electron sector the ground energy of H_SA is e_gap (both frames)."""
    f, summary = load_factors(system), load_summary(system)
    gap = np.linalg.eigvalsh(_sector(h_sa_dense(f), f.N, f.eta)).min()
    assert gap == pytest.approx(summary["e_gap"], rel=0, abs=FACTORS_ERROR[system])


def test_physical_frame_on_the_sector_for_h2_sto3g():
    """Check 3: H_phys - H_walk = Delta_eta I on the 2-electron sector (spec 3.2.1 values)."""
    system = "h2_sto3g_R2_B2_C1"
    f, summary = load_factors(system), load_summary(system)
    assert round(e_sos_walk(f), 7) == -0.6923157
    assert round(e_sos_phys(summary), 7) == -2.0132705
    assert round(frame_shift(f, summary), 7) == -1.3209547
    difference = _sector(h_phys_dense(load_integrals(system)) - h_walk_dense(f), f.N, f.eta)
    assert np.allclose(difference, frame_shift(f, summary) * np.eye(len(difference)), rtol=0,
                       atol=1e-10)
    ground = np.linalg.eigvalsh(_sector(h_phys_dense(load_integrals(system)), f.N, f.eta)).min()
    assert ground == pytest.approx(summary["e_upper"], abs=1e-10)


def test_lambda_eff_formula():
    assert lambda_eff(2.0, 0.5) == pytest.approx(np.sqrt(0.5 * 3.5), rel=1e-15)
    with pytest.raises(ValueError, match="0 <= e_gap <= 2 Lambda"):
        lambda_eff(1.0, 3.0)
