"""Givens angles: factorization, discretization, reconstruction and the ladder (spec 6.3, 8.4)."""

import numpy as np
import pytest

from dfthc.data.angles import (
    angles_from_vector,
    discretize_angles,
    givens_vector,
    ladder_matrix,
    vector_from_angles,
)
from dfthc.data.fermion import majorana


def _unit(rng, n):
    u = rng.standard_normal(n)
    return u / np.linalg.norm(u)


@pytest.mark.parametrize("n", [2, 4, 10, 54])
def test_continuum_factorization_round_trips(n):
    rng = np.random.default_rng(n)
    for _ in range(20):
        u = _unit(rng, n)
        theta = angles_from_vector(u)
        assert theta.shape == (n - 1,) and theta.min() >= 0 and theta.max() < 2 * np.pi
        assert np.allclose(givens_vector(theta), u, rtol=0, atol=1e-12)


def test_discretization_formula():
    theta = np.array([0.0, 0.1, np.pi, 2 * np.pi - 1e-9, 1.234])
    x = discretize_angles(theta, b_rot=4)
    assert x.tolist() == [round(t / (2 * np.pi) * 8) % 8 for t in theta]
    assert x.dtype.kind == "i" and x.max() < 2**3


@pytest.mark.parametrize("b_rot", [4, 8, 15])
@pytest.mark.parametrize("n", [2, 4, 10, 54])
def test_discretized_reconstruction_error_is_bounded(n, b_rot):
    """||u~ - u|| <= sum_k |delta_k| <= (N - 1) pi / 2**(b - 1), delta_k the wrapped rounding error
    of angle k: each rung is a planar rotation, so its error adds at most linearly."""
    rng = np.random.default_rng(10 * n + b_rot)
    for _ in range(20):
        u = _unit(rng, n)
        theta = angles_from_vector(u)
        x = discretize_angles(theta, b_rot)
        delta = np.angle(np.exp(1j * (theta - 2 * np.pi * x / 2 ** (b_rot - 1))))
        error = np.linalg.norm(vector_from_angles(x, b_rot) - u)
        assert error <= np.abs(delta).sum() + 1e-12
        assert np.abs(delta).sum() <= (n - 1) * np.pi / 2 ** (b_rot - 1) + 1e-12


def test_femoco_bound_value():
    assert 53 * np.pi / 2**14 == pytest.approx(1.0163e-2, rel=0, abs=1e-6)


def test_vector_from_angles_is_givens_vector_of_the_rung_angles():
    x = np.array([1, 5, 7])
    assert np.allclose(vector_from_angles(x, 4), givens_vector(4 * np.pi * x / 16),
                       rtol=0, atol=1e-15)


def test_one_rung_is_the_spec_matrix():
    """On {|01>, |10>} the rung is [[cos, -sin], [sin, cos]] with theta = 4 pi x / 2**b."""
    for x in range(8):
        theta = 4 * np.pi * x / 16
        u = ladder_matrix([x], b_rot=4, N=2)
        block = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        assert np.allclose(u[np.ix_([1, 2], [1, 2])], block, rtol=0, atol=1e-15)
        assert np.allclose(u[np.ix_([0, 3], [0, 3])], np.eye(2), rtol=0, atol=1e-15)
        assert u[0, 1] == u[3, 2] == 0


@pytest.mark.parametrize("b_rot", [4, 8])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_ladder_maps_gamma_0_onto_the_loaded_direction(n, b_rot):
    """U^dag gamma_{0,f} U = sum_j u~_j gamma_{j,f} for u~ = vector_from_angles(x) (spec 6.3)."""
    rng = np.random.default_rng(n + b_rot)
    for _ in range(5):
        x = rng.integers(0, 2 ** (b_rot - 1), n - 1)
        u_tilde = vector_from_angles(x, b_rot)
        ladder = np.kron(ladder_matrix(x, b_rot, n), np.eye(2**n))  # on psi_up
        assert np.allclose(ladder.conj().T @ ladder, np.eye(4**n), rtol=0, atol=1e-12)
        for flavour in (0, 1):
            rotated = ladder.conj().T @ majorana(0, flavour, 0, n) @ ladder
            expected = sum(u_tilde[j] * majorana(j, flavour, 0, n) for j in range(n))
            assert np.allclose(rotated, expected, rtol=0, atol=1e-12)


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="norm 1"):
        angles_from_vector([1.0, 1.0])
    with pytest.raises(ValueError, match="length N >= 2"):
        angles_from_vector([1.0])
    with pytest.raises(ValueError, match="N - 1 = 2 angles"):
        ladder_matrix([1], b_rot=4, N=3)
