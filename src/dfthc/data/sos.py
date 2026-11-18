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

"""Sum-of-squares weights, normalization, both offset frames and dense Hamiltonians (spec 3.2).

Two constants relate ``H_SA = sum_alpha O_alpha^dag O_alpha`` to energies and must not be
conflated (spec 3.2.1):

- the walk frame on the whole Fock space: ``H_walk = H_SA + E_SoS^walk``, with ``H_walk`` built
  from :func:`walk_integrals` and ``E_SoS^walk`` from the factors (:func:`e_sos_walk`);
- the physical frame on the ``eta``-electron sector: ``H_phys = H_SA + E_SoS^phys``, with
  ``E_SoS^phys`` the input datum ``summary.json["e_sos"]`` (:func:`e_sos_phys`). This holds
  exactly for exactly factorized instances and up to the factorization error otherwise.

They differ on the sector by the constant :func:`frame_shift`. The dense matrices act on the
``4**N`` Fock space of :mod:`dfthc.data.fermion` (``N <= 4``).
"""

from __future__ import annotations

import numpy as np

from dfthc.data.factors import DFTHCFactors
from dfthc.data.fermion import annihilation


def outer_weights(factors: DFTHCFactors) -> np.ndarray:
    """``(X_o,)`` outer LCU weights (spec 8.4): ``(sum_b |w_b| + |w_B|)**2 / 2`` at
    ``x_o = r C + c``, then ``2 w_+`` (D1) and ``2 w_-`` (Q1); the 2 counts both spins."""
    sf = (np.abs(inner_weights(factors)).sum(axis=1) ** 2) / 2
    return np.concatenate([sf, 2 * factors.w_plus, 2 * factors.w_minus])


def inner_weights(factors: DFTHCFactors) -> np.ndarray:
    """``(R C, B + 1)`` signed inner weights; row ``r C + c`` holds ``W[r, :, c]`` then ``w_B``."""
    rows = np.concatenate([factors.W.transpose(0, 2, 1), factors.w_B[..., None]], axis=-1)
    return rows.reshape(factors.R * factors.C, factors.B + 1)


def normalization(factors: DFTHCFactors) -> float:
    """``Lambda = sum(outer_weights) / 2`` (spec 3.2)."""
    return float(outer_weights(factors).sum() / 2)


def e_sos_walk(factors: DFTHCFactors) -> float:
    """``E_SoS^walk = -2 sum w_- - 1/2 sum_rc (w_B - sum_b w_b)**2`` (dfthcbe's ``walk_e_sos``)."""
    identity = factors.w_B - factors.W.sum(axis=1)
    return float(-2 * factors.w_minus.sum() - (identity**2).sum() / 2)


def e_sos_phys(summary: dict) -> float:
    """``E_SoS^phys``: the input datum ``summary["e_sos"]``, never recomputed from the factors."""
    return float(summary["e_sos"])


def frame_shift(factors: DFTHCFactors, summary: dict) -> float:
    """``Delta_eta = E_SoS^phys - E_SoS^walk``, the frame difference on the eta sector."""
    return e_sos_phys(summary) - e_sos_walk(factors)


def lambda_eff(Lambda: float, e_gap: float) -> float:
    """``lambda_eff = sqrt(E_gap (2 Lambda - E_gap))`` (spec 3.3)."""
    if not 0 <= e_gap <= 2 * Lambda:
        raise ValueError(f"lambda_eff needs 0 <= e_gap <= 2 Lambda; got {e_gap}, {Lambda}")
    return float(np.sqrt(e_gap * (2 * Lambda - e_gap)))


def walk_integrals(factors: DFTHCFactors) -> tuple[np.ndarray, np.ndarray]:
    """``(h1e, eri)`` of ``H_walk = H_SA + E_SoS^walk``, in chemists' order (port of dfthcbe).

    The one-body part is ``h^(1)'`` from the D1/Q1 split plus ``sum_rc (w_B - sum_b w_b) h_rc``,
    with ``h_rc = sum_b w_b u_b u_b^T``; ``eri = sum_rc h_rc (x) h_rc``, and ``h1e`` adds the
    normal-ordering term ``sum_r eri[p, r, r, q] / 2``.
    """
    f = factors
    h1_split = (f.u_plus.T @ np.diag(f.w_plus) @ f.u_plus
                - f.u_minus.T @ np.diag(f.w_minus) @ f.u_minus)
    h_rc = np.einsum("rbp,rbq,rbc->rcpq", f.U, f.U, f.W)
    h1 = h1_split + np.einsum("rc,rcpq->pq", f.w_B - f.W.sum(axis=1), h_rc)
    eri = np.einsum("rcpq,rcst->pqst", h_rc, h_rc)
    return h1 + np.einsum("prrq->pq", eri) / 2, eri


def _hamiltonian_matrix(h1e: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """``sum h1e E + 1/2 sum eri a^dag a^dag a a`` on the Fock space (port of dfthcbe)."""
    n = len(h1e)
    a = [[annihilation(p, spin, n) for spin in (0, 1)] for p in range(n)]
    E = np.array([[sum(a[p][s].conj().T @ a[q][s] for s in (0, 1)) for q in range(n)]
                  for p in range(n)])
    h1 = h1e - np.einsum("prrq->pq", eri) / 2
    two_body = np.einsum("pqrs,pqij->rsij", eri, E)
    return (np.einsum("pq,pqij->ij", h1, E)
            + sum(two_body[r, s] @ E[r, s] for r in range(n) for s in range(n)) / 2)


def h_sa_dense(factors: DFTHCFactors) -> np.ndarray:
    """``H_SA = sum_alpha O_alpha^dag O_alpha`` built directly from the generators of spec 3.2.

    D1: ``sqrt(w_+) a_{u_+, sigma}``; Q1: ``sqrt(w_-) a_{u_-, sigma}^dag``; SF:
    ``(w_B + i/2 sum_{b, sigma} w_b gamma_{u_b sigma 0} gamma_{u_b sigma 1}) / sqrt(2)``.
    Offset-free: the operator the block encoding encodes, in neither energy frame.
    """
    f, n = factors, factors.N
    a = np.array([[annihilation(p, spin, n) for spin in (0, 1)] for p in range(n)])

    def a_u(u, spin):
        return np.tensordot(u, a[:, spin], axes=1)

    generators = [np.sqrt(w) * a_u(u, s) for w, u in zip(f.w_plus, f.u_plus, strict=True)
                  for s in (0, 1)]
    generators += [np.sqrt(w) * a_u(u, s).conj().T
                   for w, u in zip(f.w_minus, f.u_minus, strict=True) for s in (0, 1)]
    eye = np.eye(4**n)
    for r, c in np.ndindex(f.R, f.C):
        pair = 0
        for b in range(f.B):
            for s in (0, 1):
                ann = a_u(f.U[r, b], s)
                gamma0, gamma1 = ann + ann.conj().T, -1j * (ann - ann.conj().T)
                pair = pair + f.W[r, b, c] * 0.5j * gamma0 @ gamma1
        generators.append((f.w_B[r, c] * eye + pair) / np.sqrt(2))
    return sum(o.conj().T @ o for o in generators)


def h_walk_dense(factors: DFTHCFactors) -> np.ndarray:
    """``H_walk``: the Fock-space matrix of :func:`walk_integrals`."""
    return _hamiltonian_matrix(*walk_integrals(factors))


def h_phys_dense(integrals: dict) -> np.ndarray:
    """``H_phys`` from ``integrals.npz`` (``h1e``, ``eri``), without ``core_energy``.

    The matrix spans the whole Fock space, but ``H_phys = H_SA + E_SoS^phys`` and
    ``H_phys - H_walk = Delta_eta`` hold only on the ``eta``-electron sector: exactly for exactly
    factorized instances, up to the factorization error otherwise.
    """
    return _hamiltonian_matrix(integrals["h1e"], integrals["eri"])
