# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dense Jordan-Wigner matrices: spec 6.2's representation, checked at N = 3."""

import functools
import itertools
from math import comb

import numpy as np
import pytest

from dfthc.data.fermion import (
    annihilation,
    majorana,
    number,
    occupation_index,
    rotated_majorana,
    sector_projector,
)

N = 3
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
EYE = np.eye(4**N)
MODES = list(itertools.product(range(N), (0, 1)))  # (j, flavour)


def _on_qubit(gate, j, spin):
    """``gate`` on qubit ``j`` of register ``spin`` (psi_up first, qubit 0 most significant)."""
    one = functools.reduce(np.kron, [gate if q == j else np.eye(2) for q in range(N)])
    return np.kron(one, np.eye(2**N)) if spin == 0 else np.kron(np.eye(2**N), one)


def _expected_majorana(j, flavour, spin):
    """Spec 6.2 written out with explicit Kronecker products, independent of ``majorana``.

    ``gamma_{j,0} = Z_0 ... Z_{j-1} X_j`` and ``gamma_{j,1} = Z_0 ... Z_{j-1} Y_j`` on the
    register of ``spin``, tensored with the identity on the other register (``psi_up`` first).
    """
    target = (X, Y)[flavour]
    one_spin = {  # N = 3, qubit 0 first (most significant)
        0: np.kron(np.kron(target, np.eye(2)), np.eye(2)),
        1: np.kron(np.kron(Z, target), np.eye(2)),
        2: np.kron(np.kron(Z, Z), target),
    }[j]
    identity = np.eye(2**N)
    return np.kron(one_spin, identity) if spin == 0 else np.kron(identity, one_spin)


def test_majorana_matches_the_explicit_jordan_wigner_construction():
    for (j, f), s in itertools.product(MODES, (0, 1)):
        assert np.array_equal(majorana(j, f, s, N), _expected_majorana(j, f, s))


@pytest.mark.parametrize("spin", [0, 1])
def test_majoranas_anticommute_within_one_spin_register(spin):
    for (j, f), (k, g) in itertools.product(MODES, repeat=2):
        a, b = majorana(j, f, spin, N), majorana(k, g, spin, N)
        expected = 2 * EYE if (j, f) == (k, g) else 0 * EYE
        assert np.allclose(a @ b + b @ a, expected, rtol=0, atol=1e-12)


def test_majoranas_on_different_spins_commute():
    """Each spin register has its own Jordan-Wigner string, so odd operators commute across."""
    for (j, f), (k, g) in itertools.product(MODES, repeat=2):
        a, b = majorana(j, f, 0, N), majorana(k, g, 1, N)
        assert np.allclose(a @ b, b @ a, rtol=0, atol=1e-12)


@pytest.mark.parametrize("spin", [0, 1])
def test_the_even_identities_the_walk_relies_on(spin):
    for j in range(N):
        z = _on_qubit(Z, j, spin)
        product = 1j * majorana(j, 0, spin, N) @ majorana(j, 1, spin, N)
        assert np.allclose(product, -z, rtol=0, atol=1e-12)
        a = annihilation(j, spin, N)
        assert np.allclose(a.conj().T @ a, number(j, spin, N), rtol=0, atol=1e-12)
        assert np.allclose(number(j, spin, N), (EYE - z) / 2, rtol=0, atol=1e-12)


@pytest.mark.parametrize("spin", [0, 1])
def test_qubit_zero_carries_no_string(spin):
    """The Majorana-gate target psi_dn[0]: gamma_0 = X, gamma_1 = Y, a = (X + iY)/2."""
    assert np.allclose(majorana(0, 0, spin, N), _on_qubit(X, 0, spin), rtol=0, atol=1e-12)
    assert np.allclose(majorana(0, 1, spin, N), _on_qubit(Y, 0, spin), rtol=0, atol=1e-12)
    expected = _on_qubit((X + 1j * Y) / 2, 0, spin)
    assert np.allclose(annihilation(0, spin, N), expected, rtol=0, atol=1e-12)


def test_annihilation_lowers_the_particle_number_by_one():
    total = sum(number(j, s, N) for j in range(N) for s in (0, 1))
    for j, s in itertools.product(range(N), (0, 1)):
        a = annihilation(j, s, N)
        assert np.allclose(total @ a - a @ total, -a, rtol=0, atol=1e-12)
        assert np.linalg.norm(a) > 1


def test_sector_projector_commutes_with_same_spin_bilinears():
    for eta in range(2 * N + 1):
        p = sector_projector(N, eta)
        assert np.allclose(p @ p, p, rtol=0, atol=1e-12)
        assert round(np.trace(p).real) == comb(2 * N, eta)
        for j, k, s in itertools.product(range(N), range(N), (0, 1)):
            hop = annihilation(j, s, N).conj().T @ annihilation(k, s, N)
            assert np.allclose(p @ hop, hop @ p, rtol=0, atol=1e-12)


def test_occupation_index_matches_the_number_operators():
    for up, dn in itertools.product(itertools.product((0, 1), repeat=N), repeat=2):
        index = occupation_index(up, dn, N)
        for j in range(N):
            assert number(j, 0, N)[index, index] == up[j]
            assert number(j, 1, N)[index, index] == dn[j]
    assert occupation_index((1, 0, 0), (0, 0, 1), N) == 0b100001


def test_rotated_majorana_is_the_linear_combination():
    u = np.array([0.6, 0.0, 0.8])
    for f, s in itertools.product((0, 1), (0, 1)):
        expected = 0.6 * _expected_majorana(0, f, s) + 0.8 * _expected_majorana(2, f, s)
        assert np.allclose(rotated_majorana(u, f, s, N), expected, rtol=0, atol=1e-12)
        squared = rotated_majorana(u, f, s, N) @ rotated_majorana(u, f, s, N)
        assert np.allclose(squared, EYE, rtol=0, atol=1e-12)


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="N <= 4"):
        majorana(0, 0, 0, 5)
    with pytest.raises(ValueError, match="flavour"):
        majorana(0, 2, 0, N)
    with pytest.raises(ValueError, match="spin"):
        majorana(0, 0, 2, N)
    with pytest.raises(ValueError, match="orbital"):
        majorana(3, 0, 0, N)
