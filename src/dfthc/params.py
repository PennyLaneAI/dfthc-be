"""Walk parameters: sizes, precisions, block bits and presets (spec sections 3.1 and 8.3)."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass, fields


def lg(value: int) -> int:
    """``ceil(log2(value))`` for a positive integer, computed exactly; ``lg(1) == 0`` (spec 3.1)."""
    if value < 1:
        raise ValueError(f"lg needs a positive integer; got {value}")
    return (value - 1).bit_length()


def odd_part(value: int) -> int:
    """The odd ``l`` of ``value = 2**eta * l`` (spec 9.2)."""
    if value < 1:
        raise ValueError(f"odd_part needs a positive integer; got {value}")
    return value >> ((value & -value).bit_length() - 1)


@dataclass(frozen=True)
class WalkParams:
    """Sizes, precisions and block bits of one DFTHC walk (spec section 8.3).

    Fields, in the order of :meth:`as_compilable`:

    - ``N``: spatial orbitals; each spin register has ``N`` qubits.
    - ``R``, ``B``, ``C``: DFTHC rank, terms per rank, copies per rank.
    - ``n_plus``: positive eigenvalues of the effective one-body matrix (D1 rows); ``N - n_plus``
      Q1 rows follow.
    - ``b_rot``: Givens angle precision: a rung turns by ``4 pi x / 2**b_rot`` with
      ``x < 2**(b_rot - 1)``, held in an angle register of ``b_rot - 1`` wires; also the
      phase-gradient width ``b_grad``.
    - ``b_k1``, ``b_k2``: bits of the outer and inner alias keep values.
    - ``s_u``: bits of the rotations inside a flagged uniform preparation.
    - ``k1``, ``k2``: block bits of the outer and inner QROAM reads.
    - ``ke1``, ``ke2``: one-hot bits of the outer and inner erase fixups.

    Construction raises ``ValueError`` naming the violated constraint. The constraints are those
    of spec 8.3, the select's one-hot fit ``2**w_b <= (N - 1)(b_rot - 1) + 1`` included, plus
    ``0 <= n_plus <= N``. Composite operators take the 13 fields as one compilable
    tuple (``as_compilable``) and rebuild the object with ``from_compilable``.
    """

    N: int
    R: int
    B: int
    C: int
    n_plus: int
    b_rot: int
    b_k1: int
    b_k2: int
    s_u: int
    k1: int
    k2: int
    ke1: int
    ke2: int

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise ValueError(f"{field.name} must be an int; got {value!r}")
            object.__setattr__(self, field.name, int(value))
        self._check(self.N >= 2, "N >= 2")
        self._check(min(self.R, self.B, self.C) >= 1, "R, B, C >= 1")
        self._check(0 <= self.n_plus <= self.N, "0 <= n_plus <= N")
        self._check(self.b_rot >= 3, "b_rot >= 3")
        self._check(min(self.b_k1, self.b_k2) >= 1, "b_k1, b_k2 >= 1")
        self._check(3 <= self.s_u <= self.b_grad, "3 <= s_u <= b_grad")
        self._check(0 <= self.k1 <= self.w_x, "0 <= k1 <= w_x")
        self._check(0 <= self.k2 <= self.w_b, "0 <= k2 <= w_b")
        self._check(0 <= self.ke1 <= self.w_x, "0 <= ke1 <= w_x")
        self._check(0 <= self.ke2 <= lg(self.RC), "0 <= ke2 <= lg(RC)")
        self._check(2**self.w_b <= (self.N - 1) * (self.b_rot - 1) + 1,
                    "2**w_b <= (N - 1)(b_rot - 1) + 1")

    def _check(self, holds: bool, constraint: str) -> None:
        if not holds:
            raise ValueError(f"WalkParams violates {constraint} (spec 8.3): {self}")

    @property
    def RC(self) -> int:
        """SF generators: ``R C``."""
        return self.R * self.C

    @property
    def X_o(self) -> int:
        """Values of the generator index ``x_o``: ``N + R C``."""
        return self.N + self.RC

    @property
    def n_minus(self) -> int:
        """Q1 rows: ``N - n_plus``."""
        return self.N - self.n_plus

    @property
    def w_x(self) -> int:
        """Width of ``x_o``: ``lg(X_o)``."""
        return lg(self.X_o)

    @property
    def w_b(self) -> int:
        """Width of ``b``: ``lg(B + 1)``."""
        return lg(self.B + 1)

    @property
    def w_r(self) -> int:
        """Width of ``r``: ``lg(R)``."""
        return lg(self.R)

    @property
    def d1(self) -> int:
        """Bits of one outer table entry (alt, keep): ``w_x + b_k1``."""
        return self.w_x + self.b_k1

    @property
    def d2(self) -> int:
        """Bits of one inner table entry (alt, keep, sgn, d_sgn, r): ``w_b + b_k2 + 2 + w_r``."""
        return self.w_b + self.b_k2 + 2 + self.w_r

    @property
    def b_grad(self) -> int:
        """Width of the phase-gradient register: ``b_rot``."""
        return self.b_rot

    @property
    def K1(self) -> int:
        """Outer block size ``2**k1``."""
        return 2**self.k1

    @property
    def K2(self) -> int:
        """Inner block size ``2**k2``."""
        return 2**self.k2

    @property
    def M1(self) -> int:
        """Live leaves of the outer read: ``ceil(X_o / K1)``."""
        return -(-self.X_o // self.K1)

    @property
    def M2(self) -> int:
        """Live leaves of the inner read: ``R C ceil((B + 1) / K2)``."""
        return self.RC * -(-(self.B + 1) // self.K2)

    def as_compilable(self) -> tuple[int, ...]:
        """The 13 fields in declaration order, as Python ints."""
        return tuple(getattr(self, field.name) for field in fields(self))

    @classmethod
    def from_compilable(cls, values: Sequence[int]) -> WalkParams:
        """Inverse of :meth:`as_compilable`."""
        values = tuple(values)
        if len(values) != len(fields(cls)):
            raise ValueError(f"WalkParams needs {len(fields(cls))} values; got {len(values)}")
        return cls(*values)


FEMOCO54 = WalkParams(N=54, R=10, B=27, C=27, n_plus=27, b_rot=15, b_k1=15, b_k2=15, s_u=7,
                      k1=2, k2=4, ke1=4, ke2=2)
"""FeMoco-54 with the precisions of the paper's Table III; ``n_plus`` for ``random_factors``."""

H2_STO3G_TEST = WalkParams(N=2, R=2, B=2, C=1, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3,
                           k1=1, k2=1, ke1=1, ke2=1)
"""The ``h2_sto3g_R2_B2_C1`` factors at test precisions."""

H2_631G_TEST = WalkParams(N=4, R=2, B=2, C=2, n_plus=3, b_rot=4, b_k1=2, b_k2=2, s_u=3,
                          k1=1, k2=1, ke1=1, ke2=1)
"""The ``h2_631g_R2_B2_C2`` factors at test precisions."""

MID_NAPHTHALENE = WalkParams(N=10, R=8, B=6, C=5, n_plus=5, b_rot=15, b_k1=15, b_k2=15, s_u=7,
                             k1=1, k2=2, ke1=3, ke2=1)
"""The ``naphthalene`` factors at FeMoco precisions with the block bits dfthcbe chooses for its
naphthalene example: the mid size of compiled T4 (spec 12.1)."""
