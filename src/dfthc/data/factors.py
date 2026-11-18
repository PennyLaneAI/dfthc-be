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

"""DFTHC factors: data model, loaders, validation and random factors (spec sections 8.1, 8.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "dfthc"
"""The vendored copy of dfthcbe's ``data/dfthc`` (spec 8.1)."""

SYSTEMS = ("h2_sto3g_R2_B2_C1", "h2_631g_R2_B2_C2", "h2_631g_R2_B3_C3", "naphthalene", "fe2s2",
           "fe4s4")
"""The vendored systems; the last two ship no ``integrals.npz``."""

UNIT_TOLERANCE = 1e-10
"""Allowed deviation of a unit vector's norm from 1."""

_ARRAYS = ("U", "W", "w_B", "w_plus", "u_plus", "w_minus", "u_minus", "h1", "h_sym", "h1_shifted")
_SCALARS = ("beta_1", "lambda_h1", "lambda_h2", "e_upper")
_NDIM = {"U": 3, "W": 3, "w_B": 2, "u_plus": 2, "u_minus": 2, "h1": 2, "h_sym": 2,
         "h1_shifted": 2, "w_plus": 1, "w_minus": 1}


@dataclass(frozen=True, eq=False)
class DFTHCFactors:
    """A DFTHC-factorized Hamiltonian, as stored in dfthcbe's ``factors.npz`` (spec 8.2).

    Arrays (float64, read-only):

    - ``U`` ``(R, B, N)``: unit vectors ``u_b^(r)``.
    - ``W`` ``(R, B, C)``: weights ``w_b^(r,c)``; ``w_B`` ``(R, C)``: identity weights.
    - ``w_plus`` ``(N+,)``, ``u_plus`` ``(N+, N)``: D1 eigenvalues and eigenvectors.
    - ``w_minus`` ``(N-,)``, ``u_minus`` ``(N-, N)``: Q1 eigenvalue magnitudes and eigenvectors.
    - ``h1``, ``h_sym``, ``h1_shifted`` ``(N, N)``: one-body term, BLISS shift, shifted term.

    Scalars: ``beta_1``, ``lambda_h1``, ``lambda_h2``, ``e_upper`` (floats) and ``eta``, the
    electron number of the target sector (int). Construction converts types only; call
    :meth:`validate` to check the factors. Instances compare by identity.
    """

    U: np.ndarray
    W: np.ndarray
    w_B: np.ndarray
    w_plus: np.ndarray
    u_plus: np.ndarray
    w_minus: np.ndarray
    u_minus: np.ndarray
    h1: np.ndarray
    h_sym: np.ndarray
    h1_shifted: np.ndarray
    beta_1: float
    lambda_h1: float
    lambda_h2: float
    e_upper: float
    eta: int

    def __post_init__(self):
        for name in _ARRAYS:
            array = np.array(getattr(self, name), dtype=float)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in _SCALARS:
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "eta", int(round(float(self.eta))))

    @property
    def N(self) -> int:
        return self.U.shape[2]

    @property
    def R(self) -> int:
        return self.U.shape[0]

    @property
    def B(self) -> int:
        return self.U.shape[1]

    @property
    def C(self) -> int:
        return self.W.shape[2]

    @property
    def n_plus(self) -> int:
        return self.w_plus.shape[0]

    @property
    def n_minus(self) -> int:
        return self.w_minus.shape[0]

    def validate(self) -> None:
        """Check the factors; raise ``ValueError`` naming the offending field.

        In order: the dimensionality of every array (``U``, ``W`` 3D; ``w_B``, ``u_plus``,
        ``u_minus``, ``h1``, ``h_sym``, ``h1_shifted`` 2D; ``w_plus``, ``w_minus`` 1D); the exact
        shapes implied by ``N, R, B, C, N+, N-``; finite entries in every array and in
        ``beta_1``, ``lambda_h1``, ``lambda_h2`` (``e_upper`` may be NaN, the marker of
        :func:`random_factors` for "no reference energy", but not infinite); unit rows of ``U``,
        ``u_plus`` and ``u_minus`` within :data:`UNIT_TOLERANCE`; ``N+ + N- = N``; and
        ``w_plus, w_minus >= 0``.
        """
        for name, ndim in _NDIM.items():
            if getattr(self, name).ndim != ndim:
                raise ValueError(f"{name} must be {ndim}D; got shape {getattr(self, name).shape}")
        N, R, B, C, n_plus, n_minus = self.N, self.R, self.B, self.C, self.n_plus, self.n_minus
        expected = {"W": (R, B, C), "w_B": (R, C), "w_plus": (n_plus,), "u_plus": (n_plus, N),
                    "w_minus": (n_minus,), "u_minus": (n_minus, N), "h1": (N, N),
                    "h_sym": (N, N), "h1_shifted": (N, N)}
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        for name in _ARRAYS:
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"{name} must be finite; it has NaN or infinite entries")
        for name in _SCALARS:
            value = getattr(self, name)
            if name == "e_upper":
                if np.isinf(value):
                    raise ValueError(f"e_upper must not be infinite; got {value}")
            elif not np.isfinite(value):
                raise ValueError(f"{name} must be finite; got {value}")
        for name in ("U", "u_plus", "u_minus"):
            norms = np.linalg.norm(getattr(self, name), axis=-1)
            if norms.size and np.abs(norms - 1).max() > UNIT_TOLERANCE:
                raise ValueError(f"the rows of {name} must have norm 1 within {UNIT_TOLERANCE}; "
                                 f"got [{norms.min()}, {norms.max()}]")
        if n_plus + n_minus != N:
            raise ValueError(f"N+ + N- = N is violated: {n_plus} + {n_minus} != {N}")
        for name in ("w_plus", "w_minus"):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} must be >= 0; got {getattr(self, name)}")


def load_factors(system: str) -> DFTHCFactors:
    """The validated factors of a vendored system, from ``factors.npz``."""
    with np.load(DATA_DIR / system / "factors.npz") as data:
        factors = DFTHCFactors(**{field.name: data[field.name] for field in fields(DFTHCFactors)})
    factors.validate()
    return factors


def load_summary(system: str) -> dict:
    """``summary.json`` of a vendored system (``lambda_sos``, ``e_sos``, ``e_gap``, ...)."""
    return json.loads((DATA_DIR / system / "summary.json").read_text())


def load_integrals(system: str) -> dict:
    """``h1e`` ``(N, N)``, ``eri`` ``(N, N, N, N)`` and ``core_energy`` from ``integrals.npz``.

    Raises ``FileNotFoundError`` for ``fe2s2`` and ``fe4s4``, which ship no integrals.
    """
    with np.load(DATA_DIR / system / "integrals.npz") as data:
        return {"h1e": np.array(data["h1e"]), "eri": np.array(data["eri"]),
                "core_energy": float(data["core_energy"])}


def random_factors(N: int, R: int, B: int, C: int, n_plus: int, seed: int) -> DFTHCFactors:
    """Valid random factors of the given shape, with exactly ``n_plus`` D1 rows (spec 8.2).

    For resource counting where no data is vendored (FeMoco-54): Toffoli counts depend only on
    shapes and live ranges. Unit vectors are normalized Gaussians; ``W`` and ``w_B`` are
    Gaussian; the one-body split uses a random orthonormal basis with eigenvalue magnitudes in
    ``[0.1, 1)``. ``h1 = h1_shifted``, ``h_sym = 0``, ``beta_1 = 0``, ``lambda_h1`` and
    ``lambda_h2`` follow the data's conventions, ``e_upper`` is NaN and ``eta = N``.
    """
    if not 0 <= n_plus <= N:
        raise ValueError(f"n_plus must be in [0, {N}]; got {n_plus}")
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((R, B, N))
    U /= np.linalg.norm(U, axis=2, keepdims=True)
    W = rng.standard_normal((R, B, C))
    w_B = rng.standard_normal((R, C))
    basis, _ = np.linalg.qr(rng.standard_normal((N, N)))
    magnitudes = rng.uniform(0.1, 1.0, N)
    w_plus, w_minus = magnitudes[:n_plus], magnitudes[n_plus:]
    u_plus, u_minus = basis.T[:n_plus], basis.T[n_plus:]
    h1_shifted = u_plus.T @ np.diag(w_plus) @ u_plus - u_minus.T @ np.diag(w_minus) @ u_minus
    inner = np.concatenate([W.transpose(0, 2, 1), w_B[..., None]], axis=-1)
    return DFTHCFactors(
        U=U, W=W, w_B=w_B, w_plus=w_plus, u_plus=u_plus, w_minus=w_minus, u_minus=u_minus,
        h1=h1_shifted, h_sym=np.zeros((N, N)), h1_shifted=h1_shifted, beta_1=0.0,
        lambda_h1=float(magnitudes.sum()), lambda_h2=float((np.abs(inner).sum(-1) ** 2).sum() / 4),
        e_upper=float("nan"), eta=N,
    )
