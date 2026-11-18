"""Prototype operators shared by the Phase 0 spikes K-1, K-3 and K-10.

Spikes may use placeholder semantics (spec 16, K-1), but the table read and its measured erase
here are exact, because K-3 checks them on lightning.qubit. The one-hot erases of the K-1 arms
(ProtoTableEraseF and ProtoTableEraseFP, Task 0.9c) are exact too and checked by
``k1_whole_program.py --fixup-check``. They are not the production operators: Phase 1
(workstream C) writes those from the spec with full tests.

Table entries are tuples of unsigned 32-bit words, least significant word first. Catalyst lowers
compilable integers as signed 64-bit values, and larger values fail to compile (canary L-24).
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import mcm_parities, register_rule, resource_key
from dfthc.ops._gates import cswap as zeroed_cswap
from dfthc.ops._gates import toffoli_key
from dfthc.testing.resources import classify_rule_warning

WORD = 32


def to_words(value: int, width: int) -> tuple[int, ...]:
    """Split a ``width``-bit integer into 32-bit words, least significant first.

    Raises ``ValueError`` unless ``0 <= value < 2**width``.
    """
    if not 0 <= value < 2**width:
        raise ValueError(f"value {value} does not fit in {width} unsigned bits")
    return tuple((value >> (WORD * w)) & (2**WORD - 1) for w in range(-(-width // WORD)))


def from_words(words: Sequence[int]) -> int:
    return sum(int(word) << (WORD * w) for w, word in enumerate(words))


# ---------------------------------------------------------------------------------------------
# Exact unary iteration (spec 9.3, cost formula of spec 9.24), with a counting twin.
# ---------------------------------------------------------------------------------------------


def _cz(control, target):
    qp.CZ([control, target])


def _z(wire):
    qp.Z(wire)


HINTED_CZ = qp.hint({"branch-prob": 0.5})(_cz)
"""The conditional CZ of a one-hot fixup leaf (spec 9.4 step 2, hinted as spec 7.6 asks)."""
HINTED_Z = qp.hint({"branch-prob": 0.5})(_z)
"""The conditional Z of a fixup leaf without one-hot bits (spec 9.4 fallback)."""


def indexable(values):
    """``values`` as an array a traced loop index can address: a jax array under capture."""
    return jnp.asarray(values) if qp.capture.enabled() else values


class _Emit:
    """Emits gates. Negative controls are X-conjugated so every TemporaryAND key is the same."""

    def x(self, wire):
        qp.X(wire)

    def and_(self, a, b, target):
        qp.TemporaryAND([a, b, target])

    def unand(self, a, b, target):
        qp.adjoint(qp.TemporaryAND([a, b, target]))

    def cnot(self, control, target):
        qp.CNOT([control, target])

    def cond_cz(self, parity, flag, wire):
        qp.cond(parity, HINTED_CZ)(flag, wire)

    def cond_z(self, parity, flag):
        qp.cond(parity, HINTED_Z)(flag)

    def leaf_loop(self, parities, base, positions, flag, onehot):
        """The conditional CZs of one fixup leaf in one static-bound qp.for_loop: iteration t
        uses ``parities[base + t]`` and ``onehot[positions[t]]``."""
        onehot, position_array = indexable(onehot), indexable(positions)

        @qp.for_loop(0, len(positions), 1)
        def leaf(t):
            qp.cond(parities[base + t], HINTED_CZ)(flag, onehot[position_array[t]])

        leaf()


class _Count(_Emit):
    """Counts the gates :class:`_Emit` would emit, by resource-key name (loop bodies as if
    unrolled)."""

    def __init__(self):
        self.counts = Counter()

    def x(self, wire):
        self.counts["PauliX"] += 1

    def and_(self, a, b, target):
        self.counts["TemporaryAND"] += 1

    def unand(self, a, b, target):
        self.counts["Adjoint(TemporaryAND)"] += 1

    def cnot(self, control, target):
        self.counts["CNOT"] += 1

    def cond_cz(self, parity, flag, wire):
        self.counts["CZ"] += 1

    def cond_z(self, parity, flag):
        self.counts["PauliZ"] += 1

    def leaf_loop(self, parities, base, positions, flag, onehot):
        self.counts["CZ"] += len(positions)


def unary_iterate(emit, address, lo, hi, work, leaf, control=None):
    """Call ``leaf(flag, value)`` for every ``value`` in ``[lo, hi)`` (exact iteration).

    ``address`` is big-endian. ``flag`` is a wire that is 1 exactly when the address equals
    ``value`` and the optional ``control`` wire is 1; it is ``None`` only for a zero-width,
    uncontrolled address. ``work`` needs ``len(address)`` wires (one per level below the root).
    """
    if not 0 <= lo < hi <= 2 ** len(address):
        raise ValueError(f"range [{lo}, {hi}) outside {len(address)}-bit addresses")
    _node(emit, address, lo, hi, list(work), leaf, control, 0, len(address))


def _node(emit, address, lo, hi, work, leaf, flag, base, height):
    if height == 0:
        leaf(flag, base)
        return
    bit = address[len(address) - height]
    half = 1 << (height - 1)
    left = base < hi and lo < base + half
    right = base + half < hi and lo < base + 2 * half
    if flag is None:  # uncontrolled root: the address bit itself is the child flag
        if left:
            emit.x(bit)
            _node(emit, address, lo, hi, work, leaf, bit, base, height - 1)
            emit.x(bit)
        if right:
            _node(emit, address, lo, hi, work, leaf, bit, base + half, height - 1)
        return
    child, rest = work[0], work[1:]
    if left and right:
        emit.x(bit)
        emit.and_(flag, bit, child)
        emit.x(bit)
        _node(emit, address, lo, hi, rest, leaf, child, base, height - 1)
        emit.cnot(flag, child)
        _node(emit, address, lo, hi, rest, leaf, child, base + half, height - 1)
        emit.unand(flag, bit, child)
    elif left:
        emit.x(bit)
        emit.and_(flag, bit, child)
        emit.x(bit)
        _node(emit, address, lo, hi, rest, leaf, child, base, height - 1)
        emit.x(bit)
        emit.unand(flag, bit, child)
        emit.x(bit)
    else:
        emit.and_(flag, bit, child)
        _node(emit, address, lo, hi, rest, leaf, child, base + half, height - 1)
        emit.unand(flag, bit, child)


def unary_toffolis(lo: int, hi: int, n: int, controlled: bool) -> int:
    """U(lo, hi, n, c) of spec 9.24."""
    if n == 0:
        return 0
    nodes = sum((hi - 1) // 2**m - lo // 2**m + 1 for m in range(1, n + 1))
    return nodes - (0 if controlled else 1)


def counts_to_resources(counts: Counter) -> dict:
    """Map counted gate names to register_resources keys."""
    keys = {
        "PauliX": qp.X, "PauliZ": qp.Z, "Hadamard": qp.H, "CNOT": qp.CNOT, "CSWAP": qp.CSWAP,
        "CZ": qp.CZ, "ZeroedToffoli": toffoli_key(),
        "TemporaryAND": resource_key(lambda: qp.TemporaryAND(Wire[3])),
        "Adjoint(TemporaryAND)": resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))),
        "MidMeasure": resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True)),
    }
    return {keys[name]: count for name, count in counts.items() if count}


# ---------------------------------------------------------------------------------------------
# QROAM read and measured erase (spec 9.3 and 9.5, plain fixup of spec 9.4 with onehot_bits=0).
# ---------------------------------------------------------------------------------------------


def nested_iterate(emit, registers, work, leaf, control=None):
    """Nested exact iteration over ``registers = ((wires, lo, hi), ...)``, outermost first.

    ``leaf(flag, values)`` receives the tuple of register values.
    """
    def recurse(index, flag, values, pool):
        wires, lo, hi = registers[index]
        if index == len(registers) - 1:
            unary_iterate(emit, wires, lo, hi, pool, lambda f, v: leaf(f, values + (v,)), flag)
        else:
            unary_iterate(emit, wires, lo, hi, pool[: len(wires)],
                          lambda f, v: recurse(index + 1, f, values + (v,), pool[len(wires):]),
                          flag)

    recurse(0, control, (), list(work))


def swap_permutation(block_bits: int, low: int) -> list[int]:
    """Entry index held by register r after the swap-with-zero network, for a low address."""
    held = list(range(2**block_bits))
    for b in reversed(range(block_bits)):
        if (low >> b) & 1:
            for r in range(2**b):
                held[r], held[r + 2**b] = held[r + 2**b], held[r]
    return held


class ProtoTableRead(Operator2):
    """QROAM read: ``target ‖ junk`` receive the requested entry and the rest of its block.

    Entry bit i goes onto wire i of its register (least significant first, unlike the big-endian
    registers of spec 6.7); prototype only.
    """

    compilable_argnames = ("table", "high_ranges", "block_bits", "width")
    wire_argnames = ("high", "low", "target", "junk", "work")

    def __init__(self, table, high_ranges, block_bits, width, high, low, target, junk, work):
        super().__init__(table, high_ranges, block_bits, width, high, low, target, junk, work)


def _split_high(high, high_ranges):
    registers, start = [], 0
    for width, lo, hi in high_ranges:
        registers.append((high[start:start + width], lo, hi))
        start += width
    return registers


def _read_body(emit, table, high_ranges, block_bits, width, high, low, target, junk, work,
               cswap: Callable):
    entries = dict(table)
    registers = [target] + [junk[r * width:(r + 1) * width] for r in range(2**block_bits - 1)]
    widths = [w for w, _, _ in high_ranges]

    def leaf(flag, values):
        prefix = 0
        for value, w in zip(values, widths, strict=True):
            prefix = (prefix << w) | value
        for position in range(2**block_bits):
            entry = from_words(entries.get((prefix << block_bits) | position, (0,)))
            for i in range(width):
                if (entry >> i) & 1:
                    emit.cnot(flag, registers[position][i])

    nested_iterate(emit, _split_high(high, high_ranges), work, leaf)
    for b in reversed(range(block_bits)):
        for r in range(2**b):
            for i in range(width):
                cswap(low[block_bits - 1 - b], registers[r][i], registers[r + 2**b][i])


def _read_counts(table, high_ranges, block_bits, width):
    counter = _Count()
    dummy = list(range(10_000))
    high = dummy[: sum(w for w, _, _ in high_ranges)]
    _read_body(counter, table, high_ranges, block_bits, width, high, dummy[:block_bits],
               dummy[:width], dummy[: (2**block_bits - 1) * width], dummy[:64],
               lambda c, a, b: counter.counts.update({"CNOT": 2, "ZeroedToffoli": 1}))
    return counter.counts


def _read_resources(table, high_ranges, block_bits, width, high, low, target, junk, work):
    return counts_to_resources(_read_counts(table, high_ranges, block_bits, width))


@qp.register_resources(_read_resources)
def _proto_table_read(table, high_ranges, block_bits, width, high, low, target, junk, work):
    _read_body(_Emit(), table, high_ranges, block_bits, width, high, low, target, junk, work,
               lambda c, a, b: zeroed_cswap(c, a, b, work[0]))


register_rule(ProtoTableRead, _proto_table_read)


def loaded_masks(table, high_ranges, block_bits, width):
    """(address, mask) for every loaded address; bit (r * width + i) is bit i of register r.

    Entry bit i goes onto wire i of its register (least significant first, unlike the big-endian
    registers of spec 6.7); prototype only. Raises ``ValueError`` for an entry outside
    ``[0, 2**width)``.
    """
    entries = {a: from_words(e) for a, e in table}
    for address, entry in entries.items():
        if not 0 <= entry < 2**width:
            raise ValueError(f"entry {entry} at address {address} does not fit in {width} bits")
    ranges = [range(lo, hi) for _, lo, hi in high_ranges]
    widths = [w for w, _, _ in high_ranges]
    result = []

    def walk(index, prefix):
        if index == len(ranges):
            for low in range(2**block_bits):
                held = swap_permutation(block_bits, low)
                mask = 0
                for r, entry_index in enumerate(held):
                    mask |= entries.get((prefix << block_bits) | entry_index, 0) << (r * width)
                result.append(((prefix << block_bits) | low, mask))
            return
        for value in ranges[index]:
            walk(index + 1, (prefix << widths[index]) | value)

    walk(0, 0)
    return result


class ProtoTableErase(Operator2):
    """Measured erase of :class:`ProtoTableRead`: X-measure, then a plain phase fixup."""

    compilable_argnames = ("table", "high_ranges", "block_bits", "width")
    wire_argnames = ("high", "low", "target", "junk", "work")

    def __init__(self, table, high_ranges, block_bits, width, high, low, target, junk, work):
        super().__init__(table, high_ranges, block_bits, width, high, low, target, junk, work)


def _erase_body(emit, table, high_ranges, block_bits, width, high, low, target, junk, work,
                measure, conditional_z):
    qubits = list(target) + list(junk)
    outcomes = [measure(q) for q in qubits]
    masks = loaded_masks(table, high_ranges, block_bits, width)
    parities = dict(zip((a for a, _ in masks), mcm_parities(outcomes, [m for _, m in masks]),
                        strict=True))
    registers = _split_high(high, high_ranges) + [(low, 0, 2**block_bits)]
    widths = [w for w, _, _ in high_ranges] + [block_bits]

    def leaf(flag, values):
        address = 0
        for value, w in zip(values, widths, strict=True):
            address = (address << w) | value
        if parities[address] is not False:
            conditional_z(parities[address], flag)

    nested_iterate(emit, registers, work, leaf)


def _erase_counts(table, high_ranges, block_bits, width):
    counter = _Count()
    num_qubits = width * 2**block_bits
    counter.counts["Hadamard"] += num_qubits
    counter.counts["MidMeasure"] += num_qubits
    nonzero = sum(1 for _, m in loaded_masks(table, high_ranges, block_bits, width) if m)
    counter.counts["PauliZ"] += nonzero
    dummy = list(range(10_000))
    registers = [(dummy[:w], lo, hi) for w, lo, hi in high_ranges] + [
        (dummy[:block_bits], 0, 2**block_bits)
    ]
    nested_iterate(counter, registers, dummy[:64], lambda f, v: None)
    return counter.counts


def _erase_resources(table, high_ranges, block_bits, width, high, low, target, junk, work):
    return counts_to_resources(_erase_counts(table, high_ranges, block_bits, width))


def _measure_x(qubit):
    qp.H(qubit)
    return qp.measure(qubit, reset=True)


@qp.register_resources(_erase_resources)
def _proto_table_erase(table, high_ranges, block_bits, width, high, low, target, junk, work):
    _erase_body(_Emit(), table, high_ranges, block_bits, width, high, low, target, junk, work,
                _measure_x, lambda parity, flag: qp.cond(parity, qp.Z)(flag))


register_rule(ProtoTableErase, _proto_table_erase)


def read_toffolis(high_ranges, block_bits, width) -> int:
    """T of ProtoTableRead: nested unary iteration plus the swap network."""
    return _nested_toffolis(high_ranges) + (2**block_bits - 1) * width


def erase_toffolis(high_ranges, block_bits) -> int:
    """T of ProtoTableErase: plain fixup over every loaded address."""
    return _nested_toffolis(list(high_ranges) + [(block_bits, 0, 2**block_bits)])


def _nested_toffolis(ranges) -> int:
    total, leaves = 0, 1
    for index, (w, lo, hi) in enumerate(ranges):
        total += leaves * unary_toffolis(lo, hi, w, controlled=index > 0)
        leaves *= hi - lo
    return total


# ---------------------------------------------------------------------------------------------
# Phase fixup with one-hot bits (spec 9.4 rule, ob >= 1) and its measured table erase, for the
# K-1 arms of Task 0.9c: F unrolls every conditional CZ, FP loops those of each high leaf.
# ---------------------------------------------------------------------------------------------


def parity_array(parities):
    """One indexable array of the entries of :func:`dfthc.compile.mcm_parities`.

    mcm_parities returns a list on purpose (Python ``False`` for a zero mask, so the plain fixup
    can skip it); a looped fixup leaf needs one array instead. Under capture every entry, ``False``
    included, becomes a ``jnp.bool_`` entry of one stacked array; in tape mode an object array keeps
    the ``MeasurementValue``\\ s (and the ``False`` entries).
    """
    if qp.capture.enabled():
        return jnp.stack([jnp.asarray(p, dtype=jnp.bool_) for p in parities])
    array = np.empty(len(parities), dtype=object)
    for i, parity in enumerate(parities):
        array[i] = parity
    return array


def fixup_leaves(covered: Sequence[int], onehot_bits: int) -> list[tuple[int, tuple, int]]:
    """``(h, positions, base)`` per high value ``h`` with a covered address, in canonical order.

    A covered address is ``a = (h << onehot_bits) | j``. The canonical (dense) order is increasing
    ``h``, then increasing low position ``j``, which is increasing ``a``; ``parities[i]`` and mask
    ``i`` use it. ``positions`` holds the covered ``j`` of ``h`` in order, and ``base`` the number
    of covered addresses in earlier leaves, so dense entry ``base + t`` is ``(h, positions[t])``.
    Raises ``ValueError`` unless ``covered`` is strictly increasing and non-empty.
    """
    if not covered or any(b <= a for a, b in zip(covered, covered[1:], strict=False)):
        raise ValueError("the covered addresses must be strictly increasing and non-empty")
    leaves, base = [], 0
    for h, group in itertools.groupby(covered, key=lambda a: a >> onehot_bits):
        positions = tuple(a & (2**onehot_bits - 1) for a in group)
        leaves.append((h, positions, base))
        base += len(positions)
    return leaves


def onehot_tree(emit, low, onehot, uncompute=False):
    """Spec 9.4 step 1 (step 3 with ``uncompute``): ``onehot[j] = [low == j]``.

    ``low`` is big-endian with ob bits and ``onehot`` has ``2**ob`` zeroed wires. The first bit and
    its negation are copied into two flags; on each further bit ``a`` every flag ``f`` splits into
    ``f AND a`` (one TemporaryAND) and ``f AND NOT a`` (one CNOT), so the tree has
    ``O(ob) = 2**ob - 2`` TemporaryANDs. The flag of a k-bit prefix ``p`` sits on
    ``onehot[p << (ob - k)]``. The uncompute runs the steps in reverse with measured
    ``Adjoint(TemporaryAND)``.
    """
    ob = len(low)
    steps = [("cnot", low[0], onehot[2 ** (ob - 1)]), ("cnot", low[0], onehot[0]),
             ("x", onehot[0])]
    for k in range(1, ob):
        stride = 2 ** (ob - k)
        for p in range(2**k):
            flag, split = onehot[p * stride], onehot[p * stride + stride // 2]
            steps += [("and", flag, low[k], split), ("cnot", split, flag)]
    for kind, *wires in reversed(steps) if uncompute else steps:
        if kind == "x":
            emit.x(*wires)
        elif kind == "cnot":
            emit.cnot(*wires)
        elif uncompute:
            emit.unand(*wires)
        else:
            emit.and_(*wires)


def phase_fixup(emit, address, leaves, onehot_bits, onehot, work, parities, control=None,
                looped=False):
    """Spec 9.4 rule: ``(-1)**parities[i]`` on dense covered address ``i``, no phase elsewhere.

    ``address`` is big-endian; ``leaves`` come from :func:`fixup_leaves` and their high values must
    be contiguous. With ``onehot_bits = ob >= 1`` the low ob address bits are one-hot encoded on
    ``onehot`` (``2**ob`` zeroed wires, returned zeroed) by :func:`onehot_tree`, exact unary
    iteration (controlled by ``control`` if given) runs over the high values, and every covered
    address gets one hinted conditional CZ on (leaf flag, ``onehot[j]``); the tree is then
    uncomputed. With ``looped`` (arm FP) the CZs of a leaf run in one static-bound ``qp.for_loop``
    (``emit.leaf_loop``). With ob = 0 (the fallback) each covered address gets one hinted
    conditional Z on its leaf flag and ``onehot`` is empty. ``work`` needs ``len(address) - ob``
    wires. T = U(high range, control) + O(ob); A = T. Raises ``ValueError`` for non-contiguous
    high values, a wrong one-hot size, or an uncontrolled zero-width high part (no leaf flag).
    """
    ob, highs = onehot_bits, [h for h, _, _ in leaves]
    if highs != list(range(highs[0], highs[-1] + 1)):
        raise ValueError("the covered high values must be contiguous")
    if len(onehot) != (2**ob if ob else 0):
        raise ValueError(f"{len(onehot)} one-hot wires for onehot_bits {ob}")
    high, low = address[: len(address) - ob], address[len(address) - ob:]
    by_high = {h: (positions, base) for h, positions, base in leaves}

    def leaf(flag, h):
        if flag is None:
            raise ValueError("an uncontrolled zero-width high part has no leaf flag")
        positions, base = by_high[h]
        if ob == 0:
            emit.cond_z(parities[base], flag)
        elif looped:
            emit.leaf_loop(parities, base, positions, flag, onehot)
        else:
            for t, j in enumerate(positions):
                emit.cond_cz(parities[base + t], flag, onehot[j])

    if ob:
        onehot_tree(emit, low, onehot)
    unary_iterate(emit, high, highs[0], highs[-1] + 1, work, leaf, control)
    if ob:
        onehot_tree(emit, low, onehot, uncompute=True)


class ProtoTableEraseF(Operator2):
    """Measured erase of :class:`ProtoTableRead` with spec 9.4's one-hot fixup (K-1 arm F).

    X-measures ``target ‖ junk``, then :func:`phase_fixup` over the read's loaded set (the covered
    set, :func:`loaded_masks` order) with ``onehot_bits`` low address bits one-hot encoded on the
    first ``2**onehot_bits`` reset wires of ``target ‖ junk``. Every covered address gets one
    hinted conditional CZ, unrolled. Never adjointed or controlled. Prototype only.
    """

    compilable_argnames = ("table", "high_ranges", "block_bits", "width", "onehot_bits")
    wire_argnames = ("high", "low", "target", "junk", "work")

    def __init__(self, table, high_ranges, block_bits, width, onehot_bits, high, low, target, junk,
                 work):
        super().__init__(table, high_ranges, block_bits, width, onehot_bits, high, low, target,
                         junk, work)


class ProtoTableEraseFP(Operator2):
    """:class:`ProtoTableEraseF` whose conditional CZs run in one static-bound ``qp.for_loop``
    per high leaf (K-1 arm FP); the tree and the unary iteration stay unrolled."""

    compilable_argnames = ("table", "high_ranges", "block_bits", "width", "onehot_bits")
    wire_argnames = ("high", "low", "target", "junk", "work")

    def __init__(self, table, high_ranges, block_bits, width, onehot_bits, high, low, target, junk,
                 work):
        super().__init__(table, high_ranges, block_bits, width, onehot_bits, high, low, target,
                         junk, work)


def onehot_erase_body(emit, table, high_ranges, block_bits, width, onehot_bits, high, low, target,
                      junk, work, measure, parities, looped, masks_hook=None, parities_hook=None,
                      leaves_hook=None):
    """The gates of :class:`ProtoTableEraseF` (``looped=False``) or :class:`ProtoTableEraseFP`.

    ``parities(outcomes, masks)`` maps the outcomes and the masks (dense covered order) to an
    indexable parity array. The hooks (identity when ``None``) let a correctness harness inject
    mutants into the masks, the parity array or the leaves; the rules never pass them. Raises
    ``ValueError`` if the fixup needs ``extra_onehot`` wires (``2**onehot_bits > K * width``).
    """
    qubits = list(target) + list(junk)
    if 2**onehot_bits > len(qubits):
        raise ValueError(f"onehot_bits {onehot_bits} needs extra_onehot wires: "
                         f"{2**onehot_bits} > {len(qubits)}")
    outcomes = [measure(q) for q in qubits]
    masks = loaded_masks(table, high_ranges, block_bits, width)
    masks = masks_hook(masks) if masks_hook else masks
    values = parities(outcomes, [m for _, m in masks])
    values = parities_hook(values) if parities_hook else values
    leaves = fixup_leaves([a for a, _ in masks], onehot_bits)
    leaves = leaves_hook(leaves) if leaves_hook else leaves
    phase_fixup(emit, list(high) + list(low), leaves, onehot_bits, qubits[: 2**onehot_bits], work,
                values, looped=looped)


def measured_parities(outcomes, masks):
    """The rules' parity array: :func:`parity_array` of :func:`mcm_parities`."""
    return parity_array(mcm_parities(outcomes, masks))


def onehot_erase_counts(table, high_ranges, block_bits, width, onehot_bits) -> Counter:
    """Gate counts of ProtoTableEraseF and of ProtoTableEraseFP (loop bodies as if unrolled)."""
    counter = _Count()

    def measure(qubit):
        counter.counts.update({"Hadamard": 1, "MidMeasure": 1})

    dummy = list(range(10_000))
    onehot_erase_body(counter, table, high_ranges, block_bits, width, onehot_bits,
                      dummy[: sum(w for w, _, _ in high_ranges)], dummy[:block_bits],
                      dummy[:width], dummy[: (2**block_bits - 1) * width], dummy[:64], measure,
                      lambda outcomes, masks: [None] * len(masks), looped=False)
    return counter.counts


def _onehot_erase_rule(name, looped):
    def resources(table, high_ranges, block_bits, width, onehot_bits, high, low, target, junk,
                  work):
        return counts_to_resources(onehot_erase_counts(table, high_ranges, block_bits, width,
                                                       onehot_bits))

    def rule(table, high_ranges, block_bits, width, onehot_bits, high, low, target, junk, work):
        onehot_erase_body(_Emit(), table, high_ranges, block_bits, width, onehot_bits, high, low,
                          target, junk, work, _measure_x, measured_parities, looped)

    rule.__name__ = name
    return qp.register_resources(resources)(rule)


register_rule(ProtoTableEraseF, _onehot_erase_rule("_proto_table_erase_f", looped=False))
register_rule(ProtoTableEraseFP, _onehot_erase_rule("_proto_table_erase_fp", looped=True))


# ---------------------------------------------------------------------------------------------
# Warning policy (spec 7.4, rule 12).
# ---------------------------------------------------------------------------------------------


def _frozen_names(names: Iterable[str], argument: str) -> frozenset[str]:
    if isinstance(names, str):
        raise TypeError(f"{argument} must be an iterable of operator names, not the string "
                        f"{names!r}")
    return frozenset(names)


def classify_warnings(records, names, *, expected_measuring: Iterable[str] = ()) -> dict:
    """Sort warnings recorded by ``warnings.catch_warnings(record=True)`` for a spike verdict.

    Returns ``{"prototype": [...], "expected": [...], "other": [...], "noise": count}``. A warning
    whose text names an operator in ``names`` goes to ``"prototype"`` (a spike fails on any),
    except a ``RuleLoweringWarning`` that is exactly Catalyst's "Skipped the <rule> decomposition
    rule for <op>: it contains a mid-circuit measurement, which is not supported with adjoint or
    control regions." where <op> is balanced ``Adjoint(...)``/``C(...)`` wrappers around one
    operator of ``expected_measuring``, followed only by attribute dumps, and that operator is the
    only name of ``names`` in the text: it goes to ``"expected"``. A ``RuleLoweringWarning``
    matching PennyLane's known noise only increments ``"noise"``. Every other text goes to
    ``"other"``, for example the ``qp.analyze`` warning "unable to determine the branch".
    ``names`` and ``expected_measuring`` are frozen to sets; a bare ``str`` raises ``TypeError``.
    Classification uses :func:`dfthc.testing.resources.classify_rule_warning`: its ``"expected"``
    shape against ``names`` finds the sole prototype name, and against ``expected_measuring`` it
    confirms that name is a measuring operator; its ``"dfthc"`` and every other ``"expected"``
    become ``"prototype"``.
    """
    names = _frozen_names(names, "names")
    expected_measuring = _frozen_names(expected_measuring, "expected_measuring")
    result = {"prototype": [], "expected": [], "other": [], "noise": 0}
    for record in records:
        message = str(record.message)
        rule_warning = issubclass(record.category, RuleLoweringWarning)
        kind = classify_rule_warning(message, names)
        if kind == "expected" and rule_warning and (
                classify_rule_warning(message, expected_measuring) == "expected"):
            result["expected"].append(message)
        elif kind in ("expected", "dfthc"):
            result["prototype"].append(message)
        elif kind == "noise" and rule_warning:
            result["noise"] += 1
        else:
            result["other"].append(message)
    return result
