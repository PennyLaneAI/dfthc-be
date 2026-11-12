"""Prototype operators shared by the Phase 0 spikes K-1, K-3 and K-10.

Spikes may use placeholder semantics (spec 16, K-1), but the table read and its measured erase
here are exact, because K-3 checks them on lightning.qubit. They are not the production
operators: Phase 1 (workstream C) writes those from the spec with full tests.

Table entries are tuples of unsigned 32-bit words, least significant word first. Catalyst lowers
compilable integers as signed 64-bit values, and larger values fail to compile (canary L-24).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import pennylane as qp
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import mcm_parities, register_rule, resource_key
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


class _Count(_Emit):
    """Counts the gates :class:`_Emit` would emit, by resource-key name."""

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
        "CZ": qp.CZ,
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
               lambda c, a, b: counter.counts.update({"CSWAP": 1}))
    return counter.counts


def _read_resources(table, high_ranges, block_bits, width, high, low, target, junk, work):
    return counts_to_resources(_read_counts(table, high_ranges, block_bits, width))


@qp.register_resources(_read_resources)
def _proto_table_read(table, high_ranges, block_bits, width, high, low, target, junk, work):
    _read_body(_Emit(), table, high_ranges, block_bits, width, high, low, target, junk, work,
               lambda c, a, b: qp.CSWAP([c, a, b]))


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
# Warning policy (spec 7.4, rule 12).
# ---------------------------------------------------------------------------------------------


def classify_warnings(records, names) -> dict:
    """Sort warnings recorded by ``warnings.catch_warnings(record=True)`` for a spike verdict.

    Returns ``{"prototype": [...], "other": [...], "noise": count}``. A warning whose text names
    an operator in ``names`` goes to ``"prototype"`` (a spike fails on any). A
    ``RuleLoweringWarning`` matching PennyLane's known noise only increments ``"noise"``. Every
    other text goes to ``"other"``, for example the ``qp.analyze`` warning "unable to determine
    the branch". Classification uses :func:`dfthc.testing.resources.classify_rule_warning`.
    """
    result = {"prototype": [], "other": [], "noise": 0}
    for record in records:
        message = str(record.message)
        kind = classify_rule_warning(message, names)
        if kind == "noise" and not issubclass(record.category, RuleLoweringWarning):
            kind = "other"
        if kind == "dfthc":
            result["prototype"].append(message)
        elif kind == "noise":
            result["noise"] += 1
        else:
            result["other"].append(message)
    return result
