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

"""Givens angles: factorization of unit vectors, discretization, reconstruction (spec 6.3, 8.4).

A Givens ladder of ``N - 1`` rungs applies ``U(u)`` with ``U(u)^dag gamma_{0,f} U(u) =
sum_j u_j gamma_{j,f}``. Rung ``k`` acts on modes ``(q_k, q_{k+1})``; on ``{|01>, |10>}`` it is
``[[cos t, -sin t], [sin t, cos t]]``. Angles are stored as ``b_rot``-bit integers
``x = round(t / (2 pi) 2**(b_rot - 1)) mod 2**(b_rot - 1)``, and the rung realizes
``t = 4 pi x / 2**b_rot`` exactly. ``angles_from_vector``, ``discretize_angles`` and
``givens_vector`` (dfthcbe's ``vector_from_angles`` on radians) are ports of dfthcbe.
"""

from __future__ import annotations

import functools

import numpy as np


def angles_from_vector(u) -> np.ndarray:
    """``N - 1`` angles in ``[0, 2 pi)`` with ``givens_vector(angles) == u`` (one-column Givens
    factorization): ``atan2(|u[k+1:]|, u[k])`` for ``k < N - 2``, then ``atan2(u[-1], u[-2])``."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size < 2:
        raise ValueError(f"u must be 1D of length N >= 2; got shape {u.shape}")
    norm = np.linalg.norm(u)
    if abs(norm - 1) > 1e-8:
        raise ValueError(f"u must have norm 1; got {norm}")
    theta = [np.arctan2(np.linalg.norm(u[k + 1:]), u[k]) for k in range(u.size - 2)]
    theta.append(np.arctan2(u[-1], u[-2]))
    return np.array(theta) % (2 * np.pi)


def givens_vector(theta) -> np.ndarray:
    """The unit vector of angles ``theta`` (radians), the inverse of :func:`angles_from_vector`."""
    theta = np.asarray(theta, dtype=float)
    u = np.zeros(theta.size + 1)
    u[0] = 1.0
    for k, phi in enumerate(theta):
        c, s = np.cos(phi), np.sin(phi)
        u[k], u[k + 1] = c * u[k] + s * u[k + 1], s * u[k] - c * u[k + 1]
    return u


def discretize_angles(theta, b_rot: int) -> np.ndarray:
    """The loaded integers ``round(theta / (2 pi) 2**(b_rot - 1)) mod 2**(b_rot - 1)``."""
    period = 2 ** (b_rot - 1)
    return np.round(np.asarray(theta, dtype=float) / (2 * np.pi) * period).astype(int) % period


def vector_from_angles(x, b_rot: int) -> np.ndarray:
    """``u~ = givens_vector(4 pi x / 2**b_rot)``: the direction the loaded integers realize."""
    return givens_vector(4 * np.pi * np.asarray(x, dtype=float) / 2**b_rot)


def _rung(theta: float, k: int, n: int) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    block = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)
    return functools.reduce(np.kron, [np.eye(2)] * k + [block] + [np.eye(2)] * (n - 2 - k))


def ladder_matrix(x, b_rot: int, N: int) -> np.ndarray:
    """The dense ``2**N x 2**N`` ladder ``U`` on one spin register for integer angles ``x``.

    ``U = G_0 G_1 ... G_{N-2}``: rung ``N - 2`` acts first in time and rung 0 last, as in
    dfthcbe's ``GivensRotationLadder``. Then ``U^dag gamma_{0,f} U = sum_j u~_j gamma_{j,f}`` with
    ``u~ = vector_from_angles(x, b_rot)`` (qubit 0 most significant, spec 6.2).
    """
    x = np.asarray(x)
    if x.shape != (N - 1,):
        raise ValueError(f"need N - 1 = {N - 1} angles; got shape {x.shape}")
    theta = 4 * np.pi * x.astype(float) / 2**b_rot
    ladder = np.eye(2**N)
    for k in range(N - 1):
        ladder = ladder @ _rung(theta[k], k, N)
    return ladder
