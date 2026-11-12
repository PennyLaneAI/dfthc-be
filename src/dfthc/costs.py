"""Closed-form costs of our construction: spec section 9.24, implemented verbatim.

``T`` counts ``Toffoli`` plus ``TemporaryAND``; ``M`` counts ``MidCircuitMeasure``, the explicit
erase measurements. Measured AND uncomputes (``Adjoint(TemporaryAND)``) are in neither; their
count ``A`` is :func:`measured_and_uncomputes`. Every T4 test asserts a compiled count against one
of these functions; each docstring names its spec 9.24 row. ``p`` is a
:class:`~dfthc.params.WalkParams`, and ``s = s_u``.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from dfthc.params import WalkParams, lg, odd_part

Range = tuple[int, int, int]
"""``(lo, hi, n)``: the range ``[lo, hi)`` of an ``n``-bit address register."""


@dataclass(frozen=True)
class Cost:
    """Toffoli count ``T`` and explicit measurement count ``M`` of one operator."""

    toffoli: int
    measurements: int = 0

    def __add__(self, other: Cost) -> Cost:
        return Cost(self.toffoli + other.toffoli, self.measurements + other.measurements)

    def __rmul__(self, factor: int) -> Cost:
        return Cost(factor * self.toffoli, factor * self.measurements)


# --- building blocks ---------------------------------------------------------------------------


def unary(lo: int, hi: int, n: int, controlled: bool) -> int:
    """Exact unary iteration over ``[lo, hi)`` of an ``n``-bit register (spec 9.24, "Exact unary
    iteration"): ``sum_{m=1}^{n} (floor((hi-1)/2^m) - floor(lo/2^m) + 1) - [c = 0]``; 0 if n = 0.
    """
    if not 0 <= lo < hi <= 2**n:
        raise ValueError(f"range [{lo}, {hi}) is not a nonempty range of {n}-bit values")
    if n == 0:
        return 0
    nodes = sum((hi - 1) // 2**m - lo // 2**m + 1 for m in range(1, n + 1))
    return nodes - (0 if controlled else 1)


def unary_nested(outer: Range, inner: Range, controlled: bool) -> int:
    """Nested exact iteration (spec 9.24): ``U_2 = U(lo1, hi1, n1, c) + (hi1 - lo1) U(lo2, hi2,
    n2, 1)``, the inner register iterated under every outer leaf."""
    lo1, hi1, n1 = outer
    return unary(lo1, hi1, n1, controlled) + (hi1 - lo1) * unary(*inner, controlled=True)


def truncated_unary(L: int, controlled: bool) -> int:
    """Truncated iteration of Qualtran and dfthcbe over ``[0, L)``, for comparison only (spec
    9.24): ``L + c - 2``."""
    return L + int(controlled) - 2


def onehot(ob: int) -> int:
    """One-hot register over ``ob`` bits (spec 9.24): ``O(ob) = 2**ob - 2`` for ob >= 1, else 0."""
    return 2**ob - 2 if ob >= 1 else 0


def _iteration(ranges: Sequence[Range], controlled: bool) -> int:
    if len(ranges) == 1:
        return unary(*ranges[0], controlled=controlled)
    if len(ranges) == 2:
        return unary_nested(ranges[0], ranges[1], controlled)
    raise ValueError(f"an address has one or two registers; got {len(ranges)}")


# --- L1 primitives -----------------------------------------------------------------------------


def phase_gradient_rotation(k: int, m: int) -> Cost:
    """``PhaseGradientRotation(k, m)`` (spec 9.24 row): ``T = (m - 1) [k != 0]``, ``M = 0``."""
    return Cost((m - 1) * int(k != 0))


def _uniform_bits(L: int) -> int:
    return lg(odd_part(L))


def flagged_uniform_prep(L: int, s: int) -> Cost:
    """``FlaggedUniformPrep(L, s)`` (spec 9.24 row): with ``L = 2**eta l``, ``l`` odd and
    ``n = lg(l)``, ``T = 3 n + 2 s - 5`` if ``l > 1``, else 0."""
    n = _uniform_bits(L)
    return Cost(3 * n + 2 * s - 5 if n else 0)


def flagged_uniform_unprep(L: int, s: int) -> Cost:
    """Adjoint of ``FlaggedUniformPrep(L, s)`` (spec 9.24 row): ``T = 2 n + 2 s - 4`` if ``l > 1``,
    else 0."""
    n = _uniform_bits(L)
    return Cost(2 * n + 2 * s - 4 if n else 0)


def qroam_read(ranges: Sequence[Range], block_bits: int, width: int, controlled: bool) -> Cost:
    """``QROAMRead`` (spec 9.24 row): ``U`` or ``U_2`` over the visited block ranges ``ranges``
    plus ``(2**block_bits - 1) width`` swap-network CSWAPs."""
    return Cost(_iteration(ranges, controlled) + (2**block_bits - 1) * width)


def phase_fixup(high_ranges: Sequence[Range], onehot_bits: int, controlled: bool) -> Cost:
    """``PhaseFixup`` (spec 9.24 row): ``U`` or ``U_2`` over the covered high ranges plus
    ``O(onehot_bits)``."""
    return Cost(_iteration(high_ranges, controlled) + onehot(onehot_bits))


def qroam_erase(ranges: Sequence[Range], block_bits: int, width: int, onehot_bits: int,
                controlled: bool) -> Cost:
    """``QROAMErase`` (spec 9.24 row): ``T`` of the ``PhaseFixup`` over the read's loaded set,
    whose high ranges are ``ranges``; ``M = 2**block_bits width``."""
    fixup = phase_fixup(ranges, onehot_bits, controlled)
    return Cost(fixup.toffoli, 2**block_bits * width)


def zero_reflection(n_w: int, controlled: bool) -> Cost:
    """``ZeroReflection`` on ``n_w`` wires (spec 9.24 row): ``T = max(n_w - 2, 0)``, controlled
    ``n_w - 1``."""
    if n_w < 1:
        raise ValueError(f"ZeroReflection needs n_w >= 1; got {n_w}")
    return Cost(n_w - 1 if controlled else max(n_w - 2, 0))


# --- L2 and L3 blocks ----------------------------------------------------------------------------


def _outer_read_ranges(p: WalkParams) -> list[Range]:
    """Outer read: ``[0, ceil(X_o / K1))`` on ``w_x - k1`` bits (spec 9.24)."""
    return [(0, -(-p.X_o // p.K1), p.w_x - p.k1)]


def _inner_read_ranges(p: WalkParams) -> list[Range]:
    """Inner read: rows ``[0, RC)`` on ``w_x`` bits, then blocks ``[0, ceil((B+1)/K2))`` on
    ``w_b - k2`` bits (spec 9.24)."""
    return [(0, p.RC, p.w_x), (0, -(-(p.B + 1) // p.K2), p.w_b - p.k2)]


def _outer_erase_ranges(p: WalkParams) -> list[Range]:
    """Outer erase: the high part of the outer read's loaded set ``[0, ceil(X_o / K1) K1)``,
    ``[0, ceil(ceil(X_o / K1) K1 / 2**ke1))`` on ``w_x - ke1`` bits (spec 9.24). It equals
    ``ceil(X_o / 2**ke1)`` when ``ke1 >= k1`` or ``K1`` divides ``X_o``."""
    loaded = -(-p.X_o // p.K1) * p.K1
    return [(0, -(-loaded // 2**p.ke1), p.w_x - p.ke1)]


def _inner_erase_ranges(p: WalkParams) -> list[Range]:
    """Inner erase: high part ``[0, ceil(RC / 2**ke2))`` on ``w_x - ke2`` bits (spec 9.24)."""
    return [(0, -(-p.RC // 2**p.ke2), p.w_x - p.ke2)]


def outer_prepare(p: WalkParams) -> Cost:
    """``OuterPrepare`` (spec 9.24 row): ``F(X_o) + T_read,out + b_k1 + w_x``, ``M = 0``."""
    read = qroam_read(_outer_read_ranges(p), p.k1, p.d1, controlled=False)
    return flagged_uniform_prep(p.X_o, p.s_u) + read + Cost(p.b_k1 + p.w_x)


def outer_unprepare(p: WalkParams) -> Cost:
    """``Adjoint(OuterPrepare)`` (spec 9.24 row): ``w_x + T_erase,out + F^dag(X_o)``,
    ``M = K1 d1``."""
    erase = qroam_erase(_outer_erase_ranges(p), p.k1, p.d1, p.ke1, controlled=False)
    return Cost(p.w_x) + erase + flagged_uniform_unprep(p.X_o, p.s_u)


def inner_prepare(p: WalkParams) -> Cost:
    """``InnerPrepare`` (spec 9.24 row): ``F(B+1) + T_read,in + b_k2 + w_b``, ``M = 0``."""
    read = qroam_read(_inner_read_ranges(p), p.k2, p.d2, controlled=False)
    return flagged_uniform_prep(p.B + 1, p.s_u) + read + Cost(p.b_k2 + p.w_b)


def inner_unprepare(p: WalkParams) -> Cost:
    """``Adjoint(InnerPrepare)`` (spec 9.24 row): ``w_b + T_erase,in + F^dag(B+1)``,
    ``M = K2 d2``; the erase one-hot spans the low ``ke2`` bits of ``x_o`` and all of ``b``."""
    erase = qroam_erase(_inner_erase_ranges(p), p.k2, p.d2, p.ke2 + p.w_b, controlled=False)
    return Cost(p.w_b) + erase + flagged_uniform_unprep(p.B + 1, p.s_u)


def tail_angle_load(p: WalkParams) -> Cost:
    """``TailAngleLoad`` (spec 9.24 row): ``U(RC, X_o, w_x, 0)``."""
    return Cost(unary(p.RC, p.X_o, p.w_x, controlled=False))


def sf_angle_load(p: WalkParams) -> Cost:
    """``SFAngleLoad`` (spec 9.24 row): ``U(0, R, w_r, 1) + R U(0, B, w_b, 1)``."""
    return Cost(unary(0, p.R, p.w_r, True) + p.R * unary(0, p.B, p.w_b, True))


def spin_mux() -> Cost:
    """``SpinMux`` (spec 9.24 row): ``T = 1``."""
    return Cost(1)


def spin_swap(p: WalkParams) -> Cost:
    """``SpinSwap`` (spec 9.24 row): ``T = N``."""
    return Cost(p.N)


def givens_rotation(p: WalkParams) -> Cost:
    """``GivensRotation`` and its adjoint (spec 9.24 row): ``T = 2 (b_rot - 1)``."""
    return Cost(2 * (p.b_rot - 1))


def givens_ladder(p: WalkParams) -> Cost:
    """``GivensLadder`` and its adjoint (spec 9.24 row): ``T = 2 (b_rot - 1)(N - 1)``."""
    return Cost(2 * (p.b_rot - 1) * (p.N - 1))


def majorana_gate() -> Cost:
    """``MajoranaGate`` and its adjoint (spec 9.24 row): ``T = 5``."""
    return Cost(5)


def angle_erase(p: WalkParams) -> Cost:
    """``AngleErase`` (spec 9.24 row): ``U(0, R, w_r, 1) + O(w_b) + U(RC, X_o, w_x, 0)``,
    ``M = (N - 1) b_rot + 3``."""
    toffoli = (unary(0, p.R, p.w_r, True) + onehot(p.w_b)
               + unary(p.RC, p.X_o, p.w_x, controlled=False))
    return Cost(toffoli, (p.N - 1) * p.b_rot + 3)


def inner_select(p: WalkParams) -> Cost:
    """``InnerSelect`` and its adjoint (spec 9.24 row): ``T_tail + T_SF + 2 + 2N +
    4 (b_rot - 1)(N - 1) + 5 + T_angle erase``, ``M = (N - 1) b_rot + 3``."""
    return (tail_angle_load(p) + sf_angle_load(p) + 2 * spin_mux() + 2 * spin_swap(p)
            + 2 * givens_ladder(p) + majorana_gate() + angle_erase(p))


def ref_t2(p: WalkParams, controlled: bool) -> Cost:
    """``RefT2`` (spec 9.24 row): ``ZeroReflection`` with ``n_w = w_b + b_k2 + 2``."""
    return zero_reflection(p.w_b + p.b_k2 + 2, controlled)


def ref_ab(p: WalkParams, controlled: bool) -> Cost:
    """``RefAB`` (spec 9.24 row): ``ZeroReflection`` with ``n_w = w_x + b_k1 + w_b + b_k2 + 2``."""
    return zero_reflection(p.w_x + p.b_k1 + p.w_b + p.b_k2 + 2, controlled)


def block_encoding(p: WalkParams, controlled: bool) -> Cost:
    """``DFTHCBlockEncoding`` (spec 9.24 row): ``T_out + T_out^dag + 2 (T_in + T_sel + T_in^dag)
    + T_RefT2`` (``InnerSigns`` costs 0), ``M = K1 d1 + 2 (K2 d2 + (N - 1) b_rot + 3)``; the
    controlled version controls only ``RefT2``."""
    inner = inner_prepare(p) + inner_select(p) + inner_unprepare(p)
    return outer_prepare(p) + outer_unprepare(p) + 2 * inner + ref_t2(p, controlled)


def walk(p: WalkParams, controlled: bool) -> Cost:
    """``DFTHCWalk`` and the controlled walk (spec 9.24 row): ``T_BE + T_RefAB``, controlled
    ``+2``; ``M = M_BE``."""
    return block_encoding(p, controlled) + ref_ab(p, controlled)


# --- L4 and totals -----------------------------------------------------------------------------


def controlled_walk_power(p: WalkParams, power: int) -> Cost:
    """``ControlledWalkPower(p)`` (spec 9.24 row): ``T = p T_C(W)``, ``M = p M_W``."""
    return Cost(power * walk(p, True).toffoli, power * walk(p, False).measurements)


def qpe_round(p: WalkParams, j: int, t: int) -> Cost:
    """``QPERound(j)`` of a ``t``-bit QPE, worst case (spec 9.24 row): ``T = 2**j T_C(W) +
    sum_{l=j+1}^{t-1} (l - j)``, ``M = 2**j M_W``."""
    corrections = sum(bit - j for bit in range(j + 1, t))
    return controlled_walk_power(p, 2**j) + Cost(corrections)


def iterative_qpe(p: WalkParams, t: int) -> Cost:
    """Iterative QPE with ``t`` bits, worst case (spec 9.24 row): ``T = (2**t - 1) T_C(W) +
    (t**3 - t) / 6``, ``M = (2**t - 1) M_W + t``."""
    return controlled_walk_power(p, 2**t - 1) + Cost((t**3 - t) // 6, t)


def toffoli_only(p: WalkParams) -> int:
    """``T_Toffoli`` per walk and controlled walk (spec 9.24, "Measured AND uncomputes"):
    ``2 w_x + 4 w_b + (K1 - 1) d1 + 2 (K2 - 1) d2 + 4 N + 12``."""
    return (2 * p.w_x + 4 * p.w_b + (p.K1 - 1) * p.d1 + 2 * (p.K2 - 1) * p.d2 + 4 * p.N + 12)


def measured_and_uncomputes(p: WalkParams, controlled: bool) -> int:
    """``A = T - T_Toffoli`` per walk or controlled walk (spec 9.24, "Measured AND uncomputes")."""
    return walk(p, controlled).toffoli - toffoli_only(p)


def cheapest_block_bits(p: WalkParams) -> tuple[int, int, int, int]:
    """``(k1, k2, ke1, ke2)`` minimizing ``walk(p).toffoli`` (spec 8.3), smallest on ties.

    The walk cost is a sum of separate terms: the outer read and erase depend on ``(k1, ke1)``
    (the erase covers the read's loaded set), the inner reads on ``k2`` and the inner erases on
    ``ke2``. So ``(k1, ke1)`` is minimized jointly, in lexicographic order, and ``k2`` and ``ke2``
    each on their own.
    """

    def cost(**changes) -> int:
        return walk(dataclasses.replace(p, **changes), False).toffoli

    k1, ke1 = min(itertools.product(range(p.w_x + 1), repeat=2),
                  key=lambda pair: cost(k1=pair[0], ke1=pair[1]))
    k2 = min(range(p.w_b + 1), key=lambda k: cost(k2=k))
    ke2 = min(range(lg(p.RC) + 1), key=lambda k: cost(ke2=k))
    return k1, k2, ke1, ke2
