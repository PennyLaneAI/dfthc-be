"""Constructor checks shared by the dfthc operators.

Spec 9.0: every constructor validates its wire-register lengths against ``required_wires(...)`` and
its compilable arguments against their documented domains, and raises ``ValueError`` with the
expected value. The length checks use ``len()`` only, so they accept wire lists, ``Wires``, the
arrays that rules see and the abstract wires of resource keys; wire identity is checked where the
labels are concrete.
"""

from __future__ import annotations

import numbers
from collections.abc import Collection, Mapping

import jax
from pennylane.typing import AbstractWires


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
    """Compare each register's length with ``required``; names in ``at_least`` may be longer.
    Then require distinct labels within and across the registers of concrete wires, since the
    wires of one operator must be distinct qubits; registers of abstract wires (``Wire[n]``
    keys) or of JAX labels (tracers inside rules) are not compared."""
    for name, size in required.items():
        actual = len(wires[name])
        if name in at_least and actual < size:
            raise ValueError(f"{op}: register {name!r} has {actual} wires; expected at least "
                             f"{size} (required_wires)")
        if name not in at_least and actual != size:
            raise ValueError(f"{op}: register {name!r} has {actual} wires; expected {size} "
                             "(required_wires)")
    seen = {}
    for name, register in wires.items():
        for label in _concrete_labels(register):
            if label in seen:
                where = (f"twice in register {name!r}" if seen[label] == name
                         else f"in registers {seen[label]!r} and {name!r}")
                raise ValueError(f"{op}: wire {label!r} appears {where}; the wires of one "
                                 "operator must be distinct")
            seen[label] = name


def _concrete_labels(register) -> list:
    """The labels of a register of concrete wires; none for abstract wires or a register with a
    JAX label (a tracer inside a rule) or an unhashable one."""
    if isinstance(register, AbstractWires):
        return []
    labels = list(register)
    try:
        for label in labels:
            if isinstance(label, jax.Array):
                return []
            hash(label)
    except TypeError:
        return []
    return labels


def canonical_ints(name: str, value) -> tuple | int:
    """Nested lists or tuples of integers as nested tuples of Python ints.

    Constructors canonicalize their compilable sequences this way, so equal arguments give one
    rule variant whatever sequence type the caller used (L-21).
    """
    if isinstance(value, (list, tuple)):
        return tuple(canonical_ints(name, item) for item in value)
    return check_int(name, value)
