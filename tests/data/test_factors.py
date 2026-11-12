"""DFTHCFactors, the loaders and random_factors (spec sections 8.1 and 8.2)."""

import dataclasses

import numpy as np
import pytest

from dfthc.data.factors import (
    DATA_DIR,
    SYSTEMS,
    DFTHCFactors,
    load_factors,
    load_integrals,
    load_summary,
    random_factors,
)


def test_data_dir_is_the_vendored_copy():
    assert DATA_DIR.name == "dfthc" and (DATA_DIR / "LICENSE").is_file()
    assert sorted(SYSTEMS) == sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


@pytest.mark.parametrize("system", SYSTEMS)
def test_every_vendored_system_loads_and_validates(system):
    factors = load_factors(system)
    factors.validate()
    summary = load_summary(system)
    assert (factors.R, factors.B, factors.C, factors.N) == (
        summary["R"], summary["B"], summary["C"], summary["nact"])
    assert factors.n_plus + factors.n_minus == factors.N
    assert factors.lambda_h1 == summary["lambda_h1"] and factors.e_upper == summary["e_upper"]
    assert type(factors.eta) is int and factors.eta > 0


def test_h2_sto3g_shapes():
    f = load_factors("h2_sto3g_R2_B2_C1")
    assert (f.U.shape, f.W.shape, f.w_B.shape) == ((2, 2, 2), (2, 2, 1), (2, 1))
    assert (f.w_plus.shape, f.u_plus.shape, f.w_minus.shape, f.u_minus.shape) == (
        (1,), (1, 2), (1,), (1, 2))
    assert (f.h1.shape, f.h_sym.shape, f.h1_shifted.shape) == ((2, 2),) * 3
    assert (f.n_plus, f.n_minus, f.eta) == (1, 1, 2)


def test_load_integrals():
    ints = load_integrals("h2_631g_R2_B2_C2")
    assert set(ints) == {"h1e", "eri", "core_energy"}
    assert ints["h1e"].shape == (4, 4) and ints["eri"].shape == (4, 4, 4, 4)
    assert ints["core_energy"] == pytest.approx(0.7151043390810812, abs=1e-15)
    with pytest.raises(FileNotFoundError):
        load_integrals("fe2s2")  # the large systems ship no integrals (spec 8.1)


def test_factors_are_immutable():
    f = load_factors("h2_sto3g_R2_B2_C1")
    with pytest.raises(ValueError, match="read-only"):
        f.U[0, 0, 0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.eta = 4


def _broken(**changes):
    return dataclasses.replace(load_factors("h2_631g_R2_B2_C2"), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"W": np.ones((2, 3, 2))}, "W has shape"),
        ({"h_sym": np.ones((3, 3))}, "h_sym has shape"),
        ({"U": np.full((2, 2, 4), 0.6)}, "rows of U"),
        ({"u_plus": np.full((3, 4), 0.6)}, "rows of u_plus"),
        ({"w_plus": np.ones(2), "u_plus": np.eye(4)[:2]}, "N\\+ \\+ N- = N"),
        ({"w_minus": -np.ones(1)}, "w_minus must be >= 0"),
        ({"w_plus": 1.0}, "w_plus must be 1D"),
        ({"U": np.ones((2, 4))}, "U must be 3D"),
        ({"u_minus": np.ones(4)}, "u_minus must be 2D"),
        ({"U": np.full((2, 2, 4), np.nan)}, "U must be finite"),
        ({"w_minus": np.array([np.nan])}, "w_minus must be finite"),
        ({"h_sym": np.full((4, 4), np.inf)}, "h_sym must be finite"),
        ({"lambda_h1": np.inf}, "lambda_h1 must be finite"),
        ({"beta_1": np.nan}, "beta_1 must be finite"),
        ({"e_upper": -np.inf}, "e_upper must not be infinite"),
    ],
)
def test_validate_rejects_bad_factors(changes, message):
    with pytest.raises(ValueError, match=message):
        _broken(**changes).validate()


def test_a_nan_e_upper_marks_unknown_and_validates():
    f = random_factors(4, 2, 3, 2, n_plus=2, seed=0)
    assert np.isnan(f.e_upper)
    f.validate()


def test_random_factors_validate_at_femoco_shape():
    f = random_factors(54, 10, 27, 27, n_plus=27, seed=0)
    f.validate()
    assert (f.N, f.R, f.B, f.C, f.n_plus, f.n_minus) == (54, 10, 27, 27, 27, 27)
    split = f.u_plus.T @ np.diag(f.w_plus) @ f.u_plus - f.u_minus.T @ np.diag(f.w_minus) @ f.u_minus
    assert np.allclose(split, f.h1_shifted, atol=1e-12)
    assert f.lambda_h1 == pytest.approx(f.w_plus.sum() + f.w_minus.sum(), rel=1e-12)


def test_random_factors_are_reproducible_and_respect_n_plus():
    a, b = random_factors(4, 2, 3, 2, n_plus=1, seed=7), random_factors(4, 2, 3, 2, 1, seed=7)
    assert all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("U", "W", "w_B", "u_plus"))
    assert (a.n_plus, a.n_minus) == (1, 3)
    assert not np.array_equal(a.W, random_factors(4, 2, 3, 2, n_plus=1, seed=8).W)
    with pytest.raises(ValueError, match="n_plus"):
        random_factors(4, 2, 3, 2, n_plus=5, seed=0)


def test_factors_type_is_exported():
    assert isinstance(load_factors("naphthalene"), DFTHCFactors)
