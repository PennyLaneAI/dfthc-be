"""Alias tables for coherent alias sampling and the distributions they prepare (spec 6.4, 8.4).

``alias_tables`` is dfthcbe's ``alias_table``, which calls Qualtran's
``preprocess_probabilities_for_reversible_sampling`` (from OpenFermion); its two helpers are
ported verbatim, including the floating-point rounding, so the integer tables equal dfthcbe's.
"""

from __future__ import annotations

import math

import numpy as np


def _discretize(weights: list, bits: int) -> list[int]:
    """Numerators over ``L 2**bits``, rounded on the cumulative sums (Qualtran)."""
    bin_count = 2**bits * len(weights)
    cumulative, total = [0.0], 0
    for weight in weights:
        total += weight
        cumulative.append(total)
    rounded = [int(math.floor(c / total * bin_count + 0.5)) for c in cumulative]
    return [b - a for a, b in zip(rounded, rounded[1:], strict=False)]


def _roulette(numerators: list[int]) -> tuple[list[int], list[int]]:
    """Alternates and keep numerators of one discretized distribution (Qualtran)."""
    weights = list(numerators)
    n = len(weights)
    target = sum(weights) // n
    alternates, keep = list(range(n)), [0] * n
    donor = 0
    for _ in range(2):
        for i in range(n):
            if weights[i] >= target:
                continue
            while weights[donor] <= target:
                donor += 1
            weights[donor] -= target - weights[i]
            alternates[i] = donor
            keep[i] = weights[i]
            weights[i] = target
    return alternates, keep


def alias_tables(weights, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """``(alt, keep)`` for one row of non-negative weights, ``keep`` in ``[0, 2**bits)``.

    Index ``l`` survives with probability ``keep[l] / 2**bits`` and is replaced by ``alt[l]``
    otherwise. An all-zero row loads nothing: zeros in both columns (the D1/Q1 inner rows).
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError(f"weights must be a non-empty 1D array; got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError(f"weights must be non-negative; got {w}")
    if bits < 1:
        raise ValueError(f"bits must be >= 1; got {bits}")
    if not w.sum():
        return np.zeros(w.size, dtype=int), np.zeros(w.size, dtype=int)
    alternates, keep = _roulette(_discretize(list(w), bits))
    return np.array(alternates), np.array(keep)


def alias_probabilities(alt, keep, bits: int) -> np.ndarray:
    """The prepared distribution (spec 6.4):
    ``p_l = (keep_l + sum_{j: alt_j = l} (2**bits - keep_j)) / (L 2**bits)``."""
    keep = np.asarray(keep, dtype=float) / 2**bits
    p = keep.copy()
    np.add.at(p, np.asarray(alt), 1 - keep)
    return p / len(keep)


def inner_alias_row(signed_row, bits: int) -> tuple[np.ndarray, ...]:
    """``(alt, keep, sgn, d_sgn)`` of one signed inner row (spec 6.1, Signs).

    ``alt`` and ``keep`` are the alias tables of ``|w|``, ``sgn = [w < 0]`` and
    ``d_sgn = sgn XOR sgn[alt]``, so ``Z(sgn) CZ(less_than, d_sgn)`` gives every branch the sign
    of the index that survives the swap.
    """
    w = np.asarray(signed_row, dtype=float)
    alt, keep = alias_tables(np.abs(w), bits)
    sgn = (w < 0).astype(int)
    return alt, keep, sgn, sgn ^ sgn[alt]
