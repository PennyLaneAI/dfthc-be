"""Constructor checks shared by the dfthc operators.

Spec 9.0: every constructor validates its wire-register lengths against ``required_wires(...)`` and
its compilable arguments against their documented domains, and raises ``ValueError`` with the
expected value. The checks use ``len()`` only, so they accept wire lists, ``Wires``, the arrays
that rules see and the abstract wires of resource keys.
"""

from __future__ import annotations

import numbers
from collections.abc import Collection, Mapping


def check_int(name: str, value, low: int | None = None, high: int | None = None) -> int:
    """Return ``value`` as a Python ``int``.

    Raises ``ValueError`` unless it is an integer (not a ``bool``) in ``[low, high]``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an int, got {value!r}")
    value = int(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} must lie in [{low}, {high}], got {value}")
    return value


def check_wires(op: str, wires: Mapping[str, object], required: Mapping[str, int], *,
                at_least: Collection[str] = ()) -> None:
    """Compare each register's length with ``required``; names in ``at_least`` may be longer."""
    for name, size in required.items():
        actual = len(wires[name])
        if name in at_least and actual < size:
            raise ValueError(f"{op}: register {name!r} has {actual} wires; expected at least "
                             f"{size} (required_wires)")
        if name not in at_least and actual != size:
            raise ValueError(f"{op}: register {name!r} has {actual} wires; expected {size} "
                             "(required_wires)")


def canonical_ints(name: str, value) -> tuple | int:
    """Nested lists or tuples of integers as nested tuples of Python ints.

    Constructors canonicalize their compilable sequences this way, so equal arguments give one
    rule variant whatever sequence type the caller used (L-21).
    """
    if isinstance(value, (list, tuple)):
        return tuple(canonical_ints(name, item) for item in value)
    return check_int(name, value)
