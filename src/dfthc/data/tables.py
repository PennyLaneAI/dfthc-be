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

"""``WalkTables``: every table the circuit loads, as compilable nested tuples (spec 8.4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from dfthc.data.alias import alias_tables, inner_alias_row
from dfthc.data.angles import angles_from_vector, discretize_angles
from dfthc.data.factors import DFTHCFactors
from dfthc.data.sos import inner_weights, outer_weights
from dfthc.params import WalkParams

INT_BOUND = 2**31
"""Every table entry is a nonnegative int below this bound (L-24: compilable ints are int64, and
operators compose wide entries as 32-bit words)."""


def _freeze(values):
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, list | tuple):
        return tuple(_freeze(value) for value in values)
    return int(values)


def _ints(values):
    if isinstance(values, tuple):
        for value in values:
            yield from _ints(value)
    else:
        yield values


@dataclass(frozen=True)
class WalkTables:
    """The alias and angle tables of one walk, as nested tuples of ints (frozen, hashable).

    - ``outer_alt``, ``outer_keep`` ``(X_o,)``: outer alias tables, ``b_k1``-bit keep values.
    - ``inner_alt``, ``inner_keep``, ``inner_sgn``, ``inner_d_sgn`` ``(R C, B + 1)``: inner alias
      tables per SF row ``x_o = r C + c``, ``b_k2``-bit keep values, and the sign columns.
    - ``tail_angles`` ``(N, N - 1)``: the ``N+`` D1 rows (``u_plus``), then the Q1 rows.
    - ``sf_angles`` ``(R, B, N - 1)``: the angles of ``U[r, b]``.

    Angles are ``b_rot``-bit integers (spec 6.3). Operators take :meth:`as_compilable`, the eight
    fields in declaration order (a tuple, the compilable type the K-1 stand-in exercised), and
    rebuild the object with :meth:`from_compilable`.
    """

    outer_alt: tuple
    outer_keep: tuple
    inner_alt: tuple
    inner_keep: tuple
    inner_sgn: tuple
    inner_d_sgn: tuple
    tail_angles: tuple
    sf_angles: tuple

    def __post_init__(self):
        for field in fields(self):
            frozen = _freeze(getattr(self, field.name))
            bad = [v for v in _ints(frozen) if not 0 <= v < INT_BOUND]
            if bad:
                raise ValueError(f"{field.name} entries must be nonnegative ints below 2**31; "
                                 f"got {bad[0]}")
            object.__setattr__(self, field.name, frozen)

    @classmethod
    def from_factors(cls, factors: DFTHCFactors, params: WalkParams) -> WalkTables:
        """The tables of ``factors`` at the precisions of ``params`` (sizes must match)."""
        f, p = factors, params
        if (p.N, p.R, p.B, p.C, p.n_plus) != (f.N, f.R, f.B, f.C, f.n_plus):
            raise ValueError(f"params (N, R, B, C, n_plus) = {(p.N, p.R, p.B, p.C, p.n_plus)} do "
                             f"not match the factors {(f.N, f.R, f.B, f.C, f.n_plus)}")
        outer_alt, outer_keep = alias_tables(outer_weights(f), p.b_k1)
        inner = [inner_alias_row(row, p.b_k2) for row in inner_weights(f)]

        def angles(u):
            return discretize_angles(angles_from_vector(u / np.linalg.norm(u)), p.b_rot)

        return cls(
            outer_alt=outer_alt, outer_keep=outer_keep,
            inner_alt=[row[0] for row in inner], inner_keep=[row[1] for row in inner],
            inner_sgn=[row[2] for row in inner], inner_d_sgn=[row[3] for row in inner],
            tail_angles=[angles(u) for u in np.concatenate([f.u_plus, f.u_minus])],
            sf_angles=[[angles(f.U[r, b]) for b in range(f.B)] for r in range(f.R)],
        )

    def as_compilable(self) -> tuple[tuple, ...]:
        """The eight fields in declaration order."""
        return tuple(getattr(self, field.name) for field in fields(self))

    @classmethod
    def from_compilable(cls, values: Sequence) -> WalkTables:
        """Inverse of :meth:`as_compilable`."""
        values = tuple(values)
        if len(values) != len(fields(cls)):
            raise ValueError(f"WalkTables needs {len(fields(cls))} fields; got {len(values)}")
        return cls(*values)
