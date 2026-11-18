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

"""Exact unary iteration, the helper that operator rules call to emit their AND ladders.

Spec sections 9.3 (address model, exact iteration) and 9.24 (the cost ``U``); user decision Q-4.
It is a Python helper used inside rules, not an ``Operator2``.

**Construction.** The iteration walks the binary tree of a big-endian address register (spec
6.7), most significant bit first. Every visited internal node holds a flag wire that is 1 exactly
when the address prefix matches the node and the control is on, and computes its child flag with
one ``TemporaryAND`` on a work wire:

- two visited children: ``child = flag AND NOT bit`` (control values ``(1, 0)``), then the left
  subtree, ``CNOT(flag, child)`` (now ``flag AND bit``), the right subtree, and the measured
  uncompute ``Adjoint(TemporaryAND)``;
- one visited child: one ``TemporaryAND`` with that child's control value, its subtree, and the
  measured uncompute with the same control values.

Negative controls use ``TemporaryAND``'s ``control_values``, a dynamic argument, so they keep a
single resource key. X conjugation appears in two places: at an uncontrolled root, which has no
AND, the children use the most significant address bit itself as their flag (X-conjugated for the
left child); and for ``control_value=0`` the control wire is X-conjugated around the whole
iteration, which a zero-width address needs (its leaf flag is the control itself) and every width
uses.

The iteration is exact: an address outside ``[lo, hi)`` raises no flag. Qualtran's and dfthcbe's
truncated iteration, which routes some out-of-range addresses onto live leaves, is a later
optimization (spec 4.4, item 8).

**Cost** (spec 9.24): one ``TemporaryAND`` and one ``Adjoint(TemporaryAND)`` per visited internal
node that has a flag, ``U(lo, hi, n, c) = sum_{m=1}^{n} (floor((hi-1)/2^m) - floor(lo/2^m) + 1)
- [c = 0]``, implemented by :func:`dfthc.costs.unary` and :func:`dfthc.costs.unary_nested`.

**Counting.** Every gate goes through an :class:`Emitter`. ``Emitter()`` emits PennyLane
operations; ``Emitter(counting=True)`` only counts them by name, so a rule and its resource
function run the same code, and :func:`as_resources` turns the counts into resource keys.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import pennylane as qp
from pennylane.typing import Wire

from dfthc.compile import resource_key


class Emitter:
    """Emits gates with PennyLane, or only counts them (``counting=True``).

    Subclasses add gates with :meth:`gate`, which keeps emission and counting in one method.
    """

    def __init__(self, counting: bool = False):
        self.counts: Counter[str] | None = Counter() if counting else None

    def gate(self, name: str, build: Callable[[], object]):
        """Run ``build`` (emit), or count one gate called ``name``."""
        if self.counts is None:
            return build()
        self.counts[name] += 1
        return None

    def x(self, wire):
        self.gate("PauliX", lambda: qp.X(wire))

    def cnot(self, control, target):
        self.gate("CNOT", lambda: qp.CNOT([control, target]))

    def and_(self, a, b, target, control_values=(1, 1)):
        self.gate("TemporaryAND",
                  lambda: qp.TemporaryAND([a, b, target], control_values=control_values))

    def unand(self, a, b, target, control_values=(1, 1)):
        self.gate("Adjoint(TemporaryAND)", lambda: qp.adjoint(
            qp.TemporaryAND([a, b, target], control_values=control_values)))


def as_resources(counts: Counter) -> dict:
    """Resource keys for the gate names an :class:`Emitter` counts (zero counts dropped).

    A name outside this table raises ``KeyError``; an emitter subclass that counts a new gate name
    adds its key here.
    """
    keys = {
        "PauliX": lambda: qp.X, "PauliZ": lambda: qp.Z, "Hadamard": lambda: qp.H,
        "CNOT": lambda: qp.CNOT, "CZ": lambda: qp.CZ, "CSWAP": lambda: qp.CSWAP,
        "TemporaryAND": lambda: resource_key(lambda: qp.TemporaryAND(Wire[3])),
        "Adjoint(TemporaryAND)": lambda: resource_key(
            lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))),
        "MidMeasure": lambda: resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True)),
    }
    return {keys[name](): count for name, count in counts.items() if count}


def work_required(widths: Sequence[int], controlled: bool) -> int:
    """Work wires of a nested iteration over registers of ``widths``, outermost first.

    A register of ``n > 0`` bits needs ``n - 1`` wires, plus one if a flag (the control, or an
    outer leaf) arrives at its root; the paths of all registers are live at an innermost leaf.
    """
    total, flagged = 0, controlled
    for n in widths:
        if n:
            total += n - 1 + int(flagged)
            flagged = True
    return total


def iterate(address, lo: int, hi: int, work, leaf: Callable, control=None, control_value: int = 1,
            *, emit: Emitter | None = None) -> None:
    """Call ``leaf(flag, value)`` for every ``value`` in ``[lo, hi)``, in increasing order.

    ``flag`` is a wire that is 1 exactly when the big-endian register ``address`` holds ``value``
    and the optional ``control`` wire holds ``control_value``. It is ``None`` only for a
    zero-width address without control, where the condition always holds. ``work`` needs at
    least ``work_required([len(address)], control is not None)`` wires in ``|0>``; they are
    returned to ``|0>``. A ``leaf`` may use ``flag`` only as a control.
    """
    iterate_nested([(address, lo, hi)], work, lambda flag, values: leaf(flag, values[0]),
                   control, control_value, emit=emit)


def iterate_nested(registers: Sequence, work, leaf: Callable, control=None,
                   control_value: int = 1, *, emit: Emitter | None = None) -> None:
    """Nested exact iteration over ``registers``, a sequence of ``(wires, lo, hi)``.

    The first register is outermost; under each of its leaves the next register is iterated with
    that leaf's flag as its control. ``leaf(flag, values)`` receives the tuple of register values.
    A leaf's ``flag`` is ``None`` only if every register has zero width and there is no control.
    Work: :func:`work_required` of the register widths. Raises ``ValueError`` for no registers,
    an empty or out-of-range range, too few work wires, or a control value other than 0 or 1.
    """
    emit = Emitter() if emit is None else emit
    # Under capture, rules receive registers as traced arrays; indexing them at every tree node
    # would add captured equations, so they are unpacked into lists of wires once.
    registers = [(list(wires), lo, hi) for wires, lo, hi in registers]
    work = list(work)
    if not registers:
        raise ValueError("iterate_nested needs at least one register")
    widths = [len(wires) for wires, _, _ in registers]
    for (_, lo, hi), n in zip(registers, widths, strict=True):
        if not 0 <= lo < hi <= 2**n:
            raise ValueError(f"range [{lo}, {hi}) is not a nonempty range of {n}-bit values")
    if control_value not in (0, 1):
        raise ValueError(f"control_value must be 0 or 1, got {control_value!r}")
    needed = work_required(widths, control is not None)
    if len(work) < needed:
        raise ValueError(f"the iteration needs {needed} work wires, got {len(work)}")
    if control is not None and control_value == 0:
        emit.x(control)
    _nest(emit, registers, 0, control, (), work, leaf)
    if control is not None and control_value == 0:
        emit.x(control)


def _nest(emit, registers, index, flag, values, work, leaf):
    wires, lo, hi = registers[index]
    n = len(wires)
    depth = n - 1 + int(flag is not None) if n else 0

    def at_leaf(leaf_flag, value):
        if index + 1 == len(registers):
            leaf(leaf_flag, values + (value,))
        else:
            _nest(emit, registers, index + 1, leaf_flag, values + (value,), work[depth:], leaf)

    _node(emit, wires, lo, hi, work[:depth], at_leaf, flag, 0, n)


def _node(emit, wires, lo, hi, work, at_leaf, flag, base, height):
    """Visit the subtree of values ``[base, base + 2**height)`` whose flag wire is ``flag``."""
    if height == 0:
        at_leaf(flag, base)
        return
    bit = wires[len(wires) - height]
    half = 1 << (height - 1)
    left = lo < base + half and base < hi
    right = lo < base + 2 * half and base + half < hi
    if flag is None:
        if left:
            emit.x(bit)
            _node(emit, wires, lo, hi, work, at_leaf, bit, base, height - 1)
            emit.x(bit)
        if right:
            _node(emit, wires, lo, hi, work, at_leaf, bit, base + half, height - 1)
        return
    child, rest = work[0], work[1:]
    if left and right:
        emit.and_(flag, bit, child, (1, 0))
        _node(emit, wires, lo, hi, rest, at_leaf, child, base, height - 1)
        emit.cnot(flag, child)
        _node(emit, wires, lo, hi, rest, at_leaf, child, base + half, height - 1)
        emit.unand(flag, bit, child)
    elif left:
        emit.and_(flag, bit, child, (1, 0))
        _node(emit, wires, lo, hi, rest, at_leaf, child, base, height - 1)
        emit.unand(flag, bit, child, (1, 0))
    else:
        emit.and_(flag, bit, child)
        _node(emit, wires, lo, hi, rest, at_leaf, child, base + half, height - 1)
        emit.unand(flag, bit, child)
