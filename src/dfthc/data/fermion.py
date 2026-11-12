"""Dense Jordan-Wigner fermion matrices on ``psi_up (x) psi_dn``, for N <= 4 (spec section 6.2).

Each spin register carries its own Jordan-Wigner string: ``gamma_{j,0} = Z_0 ... Z_{j-1} X_j`` and
``gamma_{j,1} = Z_0 ... Z_{j-1} Y_j``, with qubit 0 the most significant and ``|1>`` occupied. The
Fock space is ``psi_up (x) psi_dn`` with ``psi_up`` first, so a matrix has dimension ``4**N``.
Odd operators on different spin registers commute. ``spin`` is 0 for up and 1 for down;
``flavour`` is 0 (``X``) or 1 (``Y``).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np

MAX_ORBITALS = 4

_PAULI = (np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]))
_Z = np.diag([1.0, -1.0]).astype(complex)
_I2 = np.eye(2, dtype=complex)


def _check(N: int, *, j: int = 0, flavour: int = 0, spin: int = 0) -> None:
    if not 1 <= N <= MAX_ORBITALS:
        raise ValueError(f"dense fermion matrices need 1 <= N <= {MAX_ORBITALS}; got N = {N}")
    if not 0 <= j < N:
        raise ValueError(f"orbital {j} outside [0, {N})")
    if flavour not in (0, 1):
        raise ValueError(f"flavour must be 0 or 1; got {flavour}")
    if spin not in (0, 1):
        raise ValueError(f"spin must be 0 (up) or 1 (down); got {spin}")


def _on_register(matrix: np.ndarray, spin: int, N: int) -> np.ndarray:
    eye = np.eye(2**N, dtype=complex)
    return np.kron(matrix, eye) if spin == 0 else np.kron(eye, matrix)


def majorana(j: int, flavour: int, spin: int, N: int) -> np.ndarray:
    """``gamma_{j, flavour, spin}`` on the ``4**N`` Fock space."""
    _check(N, j=j, flavour=flavour, spin=spin)
    factors = [_Z] * j + [_PAULI[flavour]] + [_I2] * (N - 1 - j)
    return _on_register(functools.reduce(np.kron, factors), spin, N)


def annihilation(j: int, spin: int, N: int) -> np.ndarray:
    """``a_{j, spin} = (gamma_{j,0} + i gamma_{j,1}) / 2``."""
    return (majorana(j, 0, spin, N) + 1j * majorana(j, 1, spin, N)) / 2


def number(j: int, spin: int, N: int) -> np.ndarray:
    """``n_{j, spin} = a^dag a = (I - Z_j) / 2`` on register ``spin``."""
    a = annihilation(j, spin, N)
    return a.conj().T @ a


def rotated_majorana(u: Sequence[float], flavour: int, spin: int, N: int) -> np.ndarray:
    """``sum_j u_j gamma_{j, flavour, spin}``, the Majorana along the direction ``u``."""
    u = np.asarray(u, dtype=float)
    if u.shape != (N,):
        raise ValueError(f"u must have shape ({N},); got {u.shape}")
    return sum(u[j] * majorana(j, flavour, spin, N) for j in range(N))


def sector_projector(N: int, eta: int) -> np.ndarray:
    """Diagonal projector onto the Fock states with exactly ``eta`` electrons."""
    _check(N)
    occupations = np.array([bin(index).count("1") for index in range(4**N)])
    return np.diag((occupations == eta).astype(complex))


def occupation_index(up_bits: Sequence[int], dn_bits: Sequence[int], N: int) -> int:
    """Fock basis index of the occupations ``up_bits`` and ``dn_bits`` (qubit 0 first)."""
    _check(N)
    bits = list(up_bits) + list(dn_bits)
    if len(bits) != 2 * N or any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"need {N} occupation bits per spin; got {up_bits} and {dn_bits}")
    return int("".join(str(bit) for bit in bits), 2)
