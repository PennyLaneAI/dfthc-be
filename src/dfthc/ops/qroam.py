"""The paper's QROAM read, the erase phase fixup and the measured erase (spec 9.3 to 9.5).

**Address model** (spec 9.3). The address is one or two big-endian registers (spec 6.7),
``address_widths = (w_1,)`` or ``(w_1, w_2)``, concatenated with the first register most
significant: the integer address is ``(a_1 << w_2) | a_2``. Each register has a range
``(lo, hi)``. A table maps integer addresses to entries of ``width`` bits, stored as tuples of
32-bit words, least significant word first (L-24).

**Bit order.** Every register holds an entry big-endian: wire ``i`` of a ``width``-wire register
holds bit ``width - 1 - i``. ``target ‖ junk`` is read as one big-endian integer of
``2**block_bits * width`` bits (``target`` most significant, then the junk slots in order); that
integer is :meth:`QROAMRead.loaded_bits`. The read's dynamic ``columns`` hold each block before the
swap network in the same wire order: bit ``p width + i`` of a column is the bit for wire ``i`` of
position ``p``'s register, entry bit ``width - 1 - i`` (:meth:`QROAMRead.columns_from_table`
converts a table).
With ``M = 2**block_bits * width``, the qubits of ``target ‖ junk`` whose outcomes the erase
uses, bit ``i`` of ``loaded_bits`` is the wire ``M - 1 - i`` of ``target ‖ junk``, so the erase
passes its outcomes to :func:`dfthc.compile.mcm_parities` in reverse wire order.
:class:`QROAMErase` measures all of them; :func:`emit_erase_from_outcomes` measures the target and
takes the junk's outcomes, in wire order, from a caller that X-measured the junk after the read.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.ops import MeasurementValue
from pennylane.typing import AbstractArray, Bool, Wire

from dfthc.compile import mcm_parities, register_rule, resource_key
from dfthc.ops._gates import cswap, cswap_resources
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.unary import Emitter, as_resources, iterate_nested, work_required

WORD = 32
"""Data bits per table word (spec 8.4, L-24)."""


class QROAMEmitter(Emitter):
    """:class:`~dfthc.ops.unary.Emitter` with the gates of the angle loads, the fixup and the
    erase."""

    def wire_array(self, register):
        """``register`` as an array that a traced loop index can address: a JAX array under
        capture, a list in tape mode and when counting."""
        if self.counts is None and qp.capture.enabled():
            return jnp.asarray(register)
        return list(register)

    def load(self, flag, rows, leaf, register):
        """The loads of unary leaf ``leaf``: one ``qp.for_loop`` over ``register``
        (:meth:`wire_array`); iteration ``i`` applies, conditioned on ``rows[leaf, i]``, a CNOT
        from ``flag`` (an ``X`` for a ``None`` flag) onto ``register[i]``. No conditional
        surrounds the loop. Counting: ``len(register)`` gates, the upper bound."""
        name = "PauliX" if flag is None else "CNOT"
        if self.counts is not None:
            self.counts[name] += len(register)
            return

        @qp.for_loop(0, len(register), 1)
        def bit_loop(i):
            if flag is None:
                qp.cond(rows[leaf, i], qp.X)(register[i])
            else:
                qp.cond(rows[leaf, i], qp.CNOT)([flag, register[i]])

        bit_loop()

    def cond_cz(self, parity, a, b):
        def branch():
            qp.CZ([a, b])

        self.gate("CZ", lambda: qp.cond(parity, qp.hint({"branch-prob": 0.5})(branch))())

    def cond_z(self, parity, wire):
        def branch():
            qp.Z(wire)

        self.gate("PauliZ", lambda: qp.cond(parity, qp.hint({"branch-prob": 0.5})(branch))())

    def measure_x(self, wire):
        """``H`` then ``qp.measure(wire, reset=True)``; returns the outcome (``None`` counting)."""
        self.gate("Hadamard", lambda: qp.H(wire))
        return self.gate("MidMeasure", lambda: qp.measure(wire, reset=True))

    def parities(self, outcomes, masks):
        """:func:`dfthc.compile.mcm_parities`; counting, ``False`` exactly for the zero masks."""
        if self.counts is None:
            return mcm_parities(outcomes, masks)
        return [False if mask == 0 else True for mask in masks]


# --------------------------------------------------------------------------- address helpers


def _check_address(widths, ranges) -> tuple[tuple, tuple]:
    """Canonical ``(address_widths, ranges)``; ``ValueError`` outside the address model."""
    widths = canonical_ints("address_widths", widths)
    ranges = canonical_ints("ranges", ranges)
    if not isinstance(widths, tuple) or len(widths) not in (1, 2):
        raise ValueError(f"an address has one or two registers, got address_widths={widths!r}")
    if any(n < 0 for n in widths) or sum(widths) < 1 or (len(widths) == 2 and widths[0] < 1):
        raise ValueError("address registers need nonnegative widths, at least one wire, and a "
                         f"nonempty first register of two; got {widths}")
    if not isinstance(ranges, tuple) or len(ranges) != len(widths):
        raise ValueError(f"need one (lo, hi) per address register, got {ranges!r}")
    for n, pair in zip(widths, ranges, strict=True):
        if not isinstance(pair, tuple) or len(pair) != 2 or not 0 <= pair[0] < pair[1] <= 2**n:
            raise ValueError(f"range {pair!r} is not a nonempty range of {n}-bit values")
    return widths, ranges


def _check_control_value(control_value):
    if control_value is None:
        return None
    return check_int("control_value", control_value, low=0, high=1)


def _control(control, control_value) -> dict:
    """``iterate_nested``'s control arguments for a 0- or 1-wire ``control`` register."""
    if control_value is None:
        return {"control": None, "control_value": 1}
    return {"control": control[0], "control_value": control_value}


def _high_specs(widths, ranges, low_bits) -> list[tuple[int, int, int, int]]:
    """``(start, n, lo, hi)`` of every register that keeps bits once the low ``low_bits`` bits of
    the concatenated address are taken off; a register split by the boundary keeps its top bits
    and the shifted range. A fully low address gives one zero-width register ``(0, 0, 0, 1)``."""
    starts = list(itertools.accumulate((0, *widths)))
    specs, remaining = [], low_bits
    for index in reversed(range(len(widths))):
        n, (lo, hi) = widths[index], ranges[index]
        if remaining >= n:
            remaining -= n
            continue
        specs.append((starts[index], n - remaining, lo >> remaining,
                      ((hi - 1) >> remaining) + 1))
        remaining = 0
    return list(reversed(specs)) or [(0, 0, 0, 1)]


def _registers(address, specs) -> list:
    return [(address[start:start + n], lo, hi) for start, n, lo, hi in specs]


def _concatenate(values, widths) -> int:
    result = 0
    for value, n in zip(values, widths, strict=True):
        result = (result << n) | value
    return result


def _join(*registers) -> list:
    """The wires of ``registers``, concatenated in order."""
    return [wire for register in registers for wire in register]


def dummy_wires(sizes: dict[str, int]) -> dict[str, range]:
    """Consecutive integer wires for every register: what resource functions count on."""
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = range(start, start + size)
        start += size
    return wires


# --------------------------------------------------------------------------- QROAMRead


def _check_table(table, widths, live_ranges, width) -> tuple:
    """Canonical table: ``(address, words)`` pairs sorted by address, each address live."""
    table = canonical_ints("table", table)
    words = -(-width // WORD)
    seen = set()
    for pair in table:
        if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[1], tuple):
            raise ValueError(f"table entries are (address, words) pairs, got {pair!r}")
        address, entry = pair
        if address in seen:
            raise ValueError(f"address {address} appears twice in the table")
        seen.add(address)
        if not _is_live(address, widths, live_ranges):
            raise ValueError(f"table entry at address {address} lies outside the live ranges "
                             f"{live_ranges}")
        if len(entry) != words or any(not 0 <= word < 2**WORD for word in entry):
            raise ValueError(f"the entry at address {address} must be {words} words in "
                             f"[0, 2**{WORD}), got {entry}")
        if _entry_value(entry) >> width:
            raise ValueError(f"the entry at address {address} does not fit in {width} bits")
    return tuple(sorted(table))


def _register_values(address, widths) -> list[int]:
    values, shift = [], sum(widths)
    for n in widths:
        shift -= n
        values.append((address >> shift) & (2**n - 1))
    return values


def _is_live(address, widths, ranges) -> bool:
    return 0 <= address < 2 ** sum(widths) and all(
        lo <= value < hi for value, (lo, hi) in zip(_register_values(address, widths), ranges,
                                                    strict=True))


def _entry_value(words) -> int:
    return sum(word << (WORD * i) for i, word in enumerate(words))


def _entries(table) -> dict[int, int]:
    return {address: _entry_value(words) for address, words in table}


def _live_bits(widths, live_ranges, width, block_bits) -> np.ndarray:
    """Booleans of the columns' shape: which bits belong to a live address."""
    num_columns, _ = QROAMRead.select_size(widths, live_ranges, block_bits, None)
    live = [[_is_live((column << block_bits) | p, widths, live_ranges)
             for p in range(2**block_bits)] for column in range(num_columns)]
    return np.repeat(np.array(live, dtype=bool), width, axis=1)


def _column_array(columns, widths, live_ranges, width, block_bits):
    """Canonicalize the dynamic ``columns`` once (L-23).

    Concrete columns (nested sequences or arrays) must be a bool array of
    :meth:`QROAMRead.columns_shape` that is zero at every dead position and unvisited column.
    Tracers and abstract arrays (operators rebuilt by Catalyst, resource keys) are checked for
    their shape and dtype only. Raises ``ValueError``.
    """
    shape = QROAMRead.columns_shape(widths, live_ranges, width, block_bits)
    abstract = isinstance(columns, AbstractArray) or qp.math.is_tracer(columns)
    array = columns if abstract else np.asarray(columns)
    if tuple(array.shape) != shape or np.dtype(array.dtype) != np.bool_:
        raise ValueError(f"columns must be a bool array of shape {shape}, got "
                         f"{tuple(array.shape)} of {array.dtype}")
    if abstract:
        return columns
    if np.any(array & ~_live_bits(widths, live_ranges, width, block_bits)):
        raise ValueError("a dead position or an unvisited column of columns holds a nonzero bit")
    return jnp.asarray(array) if qp.capture.enabled() else array


def _swap_order(block_bits, low) -> list[int]:
    """The block position each register ``target, junk slot 0, ...`` holds after the swap
    network for the low address bits ``low``."""
    held = list(range(2**block_bits))
    for b in reversed(range(block_bits)):
        if (low >> b) & 1:
            for r in range(2**b):
                held[r], held[r + 2**b] = held[r + 2**b], held[r]
    return held


def _loaded_contents(entries, widths, live_ranges, width, block_bits, addresses) -> list[int]:
    """``target ‖ junk`` (big-endian) after an enabled read, for each address."""
    loaded = QROAMRead.loaded_ranges(widths, live_ranges, block_bits)
    size = 2**block_bits
    contents = []
    for address in addresses:
        if not _is_live(address, widths, loaded):
            contents.append(0)
            continue
        base, value = address - (address % size), 0
        for position in _swap_order(block_bits, address % size):
            value = (value << width) | entries.get(base | position, 0)
        contents.append(value)
    return contents


class QROAMRead(Operator2):
    r"""The paper's QROAM read: the requested entry into ``target``, the rest of its block into
    ``junk``, for a measured erase: the junk stays until :class:`QROAMErase`, or the caller
    X-measures it early and erases with :func:`emit_erase_from_outcomes`.

    **Action** on every computational address ``a`` (spec 9.3), with :math:`K = 2^k` and
    ``k = block_bits``; a *block* is the set of ``K`` addresses that differ only in the low ``k``
    bits of the last register, and a block is *visited* if it meets the live ranges:

    ============================================== ================ =========================
    case                                           ``target``       ``junk``
    ============================================== ================ =========================
    visited block, live position, enable satisfied :math:`T[a]`     the block's other entries
    visited block, dead position (padding)         0                the block's other entries
    block not visited (e.g. one-body rows,         0                0
    failed :math:`x_o \ge X_o`)
    enable not satisfied                           0                0
    ============================================== ================ =========================

    ``T`` is the table that :meth:`columns_from_table` laid out in ``columns``; entries missing
    from it are zero. ``junk`` slot ``r`` holds the block position ``junk_layout(k, a mod K)[r]``.
    :meth:`loaded_bits` gives ``target ‖ junk`` for every address. The read adds no phase, and
    the address is unchanged.

    **Registers.**

    =========== ================================= ======== ===============================
    name        width                             input    output
    =========== ================================= ======== ===============================
    ``address`` ``sum(address_widths)``           any      unchanged
    ``target``  ``width``                         ``|0>``  the entry
    ``junk``    ``(K - 1) width``                 ``|0>``  the block's other entries
    ``work``    ``max(c - 1, [k > 0])``           ``|0>``  ``|0>``
    ``control`` 0 wires, or 1 (the enable)        any      unchanged
    =========== ================================= ======== ===============================

    ``c`` is the select's control count (:meth:`select_size`): the address bits above the low
    ``k``, plus the enable. The swap network needs one work wire.

    **Arguments.** Dynamic: ``columns``, a bool array of shape :meth:`columns_shape`,
    ``(num_columns, K width)``: row ``h`` is the block whose address bits above the low ``k``
    hold ``h``, its ``K`` positions in order, each entry big-endian (spec 6.7; bit ``i`` of
    position ``p`` is entry bit ``width - 1 - i``, the bit for wire ``p width + i`` of
    ``target ‖ junk``); zero at dead positions and at the rows of unvisited blocks below the
    last visited one. Callers build it from their compilable table with
    :meth:`columns_from_table`. Compilable: ``address_widths`` (one or two widths);
    ``live_ranges`` (one ``(lo, hi)`` per register); ``width`` (int >= 1); ``block_bits``
    (``0 <= k <= address_widths[-1]``); ``control_value`` (``None`` without an enable wire, else
    the enabling value 0 or 1).

    **Pre- and postconditions.** ``target``, ``junk`` and ``work`` start in ``|0>``; ``work``
    ends in ``|0>``. The loaded data is erased by measurement on one of two paths: (a) nothing
    writes ``target ‖ junk`` until :class:`QROAMErase` measures both; or (b) the caller applies
    ``H`` and ``qp.measure(q, reset=True)`` to every ``junk`` qubit after the read and before
    anything else acts on the junk (right after the read, or later: the walk measures right
    after each enclosing prepare, whose comparator and swaps do not touch the junk), keeps the
    outcomes in wire order (slot ``r`` of :meth:`junk_layout` is outcomes ``r width`` to
    ``(r + 1) width - 1``), may then use the junk wires for other registers, and later erases
    the unchanged ``target`` with :func:`emit_erase_from_outcomes` (spec 16, Q-1 variant (b)).
    On both paths, operations in between may act on the address only in ways that keep what the
    read loaded for the current address.

    **Decomposition.** (1) The select: ``qp.QROM(columns, control_wires=control ‖ high,
    target_wires=target ‖ junk, work_wires=work[:c - 1], clean=True)``, ``high`` the address
    wires above the low ``k``. With its ``c - 1`` work wires its depth is 1 and only its
    unary-iteration rule applies, to which :data:`dfthc.compile.PENNYLANE_PINS` pins it. An
    address beyond the last column loads nothing, and the enable, its most significant control
    wire, loads only on 0 (``X``-conjugated for ``control_value = 1``). (2) A swap-with-zero
    network on the low ``k`` address bits, most significant first: for bit ``b``, ``CSWAP``
    register ``r`` with register ``r + 2**b`` for ``r < 2**b``, which moves the selected entry
    into ``target`` with ``(K - 1) width`` exact ``CSWAP``\s, each with the zeroed ``work[0]``
    (:func:`dfthc.ops._gates.cswap`).

    **Adjoint and control.** Never adjointed or controlled: its select has loops and data
    conditionals (spec 7.4 rule 4; L-6), and no program wraps a composite that contains it in an
    adjoint or a control (spec 7.4 rule 6). Its data is erased by :class:`QROAMErase` (the
    baseline unprepares) or by :func:`emit_erase_from_outcomes` (the unprepares from outcomes,
    which the walk calls); use the enable instead of a control.

    **Cost.** :func:`dfthc.costs.qroam_read` with ``select_size(...)``: :math:`T = N(c, C)`,
    PennyLane's unary QROM over the ``C`` columns, plus :math:`(2^k - 1)` ``width``; ``M = 0``;
    ``A = T`` (each swap-network ``CSWAP`` a ``TemporaryAND`` with a measured uncompute on
    ``work[0]``). Where every block from 0 to the last is visited, ``N`` is one fewer than
    exact unary iteration over the blocks (its root merges the two top ANDs into CNOTs). At
    FeMoco-54: outer 83 + 72 = 155, inner 542 + 390 = 932 (its 540 columns of ``x_o ‖ b_top``
    are contiguous). The ``CNOT`` and ``PauliX`` counts are upper bounds: ``QROM`` declares a
    load gate for every bit.

    **Paper and dfthcbe.** Steps 2 and 6; the QROAM of Berry et al. (Quantum 3, 208) with a
    measured erase. The paper X-measures its prepares' junk early, in the prepare
    (arXiv:2502.15882v1 App. B.7.2, steps 2 and 4); the walk measures it right after each
    enclosing prepare, before any angle load, which equals measuring right after the read
    because the comparator and the swaps in between do not touch the junk; dfthcbe keeps it
    until the erase. dfthcbe: the reads in ``OuterPrepare``/``InnerPrepare``; its truncated
    iteration and flat ``r`` target are deferred refinements (spec 9.3, "Known deltas").

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import QROAMRead

        table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
        shape = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
        columns = QROAMRead.columns_from_table(table, **shape)  # shape (3, 4)
        program = lambda: QROAMRead(columns, *shape.values(), None, address=[0, 1, 2],
                                    target=[3, 4], junk=[5, 6], work=[7], control=[])
        compiled_resources(program, num_wires=8).toffoli  # N(2, 3) + 2 = 1 + 2
    """

    dynamic_argnames = ("columns",)
    compilable_argnames = ("address_widths", "live_ranges", "width", "block_bits",
                           "control_value")
    wire_argnames = ("address", "target", "junk", "work", "control")

    def __init__(self, columns, address_widths, live_ranges, width, block_bits, control_value,
                 address, target, junk, work, control):
        address_widths, live_ranges = _check_address(address_widths, live_ranges)
        width = check_int("width", width, low=1)
        block_bits = check_int("block_bits", block_bits, low=0, high=address_widths[-1])
        control_value = _check_control_value(control_value)
        columns = _column_array(columns, address_widths, live_ranges, width, block_bits)
        check_wires("QROAMRead",
                    {"address": address, "target": target, "junk": junk, "work": work,
                     "control": control},
                    self.required_wires(address_widths, live_ranges, width, block_bits,
                                        control_value))
        super().__init__(columns, address_widths, live_ranges, width, block_bits,
                         control_value, address, target, junk, work, control)

    @staticmethod
    def required_wires(address_widths, live_ranges, width, block_bits,
                       control_value) -> dict[str, int]:
        """Exact register sizes."""
        _, num_controls = QROAMRead.select_size(address_widths, live_ranges, block_bits,
                                                control_value)
        return {"address": sum(address_widths), "target": width,
                "junk": (2**block_bits - 1) * width,
                "work": max(num_controls - 1, int(block_bits > 0)),
                "control": int(control_value is not None)}

    @staticmethod
    def select_size(address_widths, live_ranges, block_bits, control_value) -> tuple[int, int]:
        """``(num_columns, num_controls)`` of the select: every block up to the last visited
        one, on the address bits above the low ``block_bits`` plus the enable; the first two
        arguments of :func:`dfthc.costs.qroam_read`."""
        widths, ranges = _check_address(address_widths, live_ranges)
        last = _concatenate([hi - 1 for _, hi in ranges], widths)
        return ((last >> block_bits) + 1,
                sum(widths) - block_bits + int(control_value is not None))

    @staticmethod
    def columns_shape(address_widths, live_ranges, width, block_bits) -> tuple[int, int]:
        """``(num_columns, 2**block_bits * width)``, the shape of ``columns``."""
        num_columns, _ = QROAMRead.select_size(address_widths, live_ranges, block_bits, None)
        return num_columns, 2**block_bits * width

    @staticmethod
    def loaded_ranges(address_widths, live_ranges, block_bits) -> tuple[tuple[int, int], ...]:
        """The loaded set (every address of a visited block) as one ``(lo, hi)`` per register:
        the ``covered`` argument of :class:`PhaseFixup` for this read's erase."""
        widths, ranges = _check_address(address_widths, live_ranges)
        (*outer, (lo, hi)) = ranges
        return (*outer, ((lo >> block_bits) << block_bits,
                         (((hi - 1) >> block_bits) + 1) << block_bits))

    @staticmethod
    def loaded_set(address_widths, live_ranges, block_bits) -> list[int]:
        """Every address of a visited block, in increasing order."""
        widths, _ = _check_address(address_widths, live_ranges)
        loaded = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
        return [_concatenate(values, widths)
                for values in itertools.product(*(range(lo, hi) for lo, hi in loaded))]

    @staticmethod
    def junk_layout(block_bits, low) -> list[int]:
        """The block position held by each junk slot, for low address bits ``low``."""
        return _swap_order(block_bits, low)[1:]

    @staticmethod
    def columns_from_table(table, *, address_widths, live_ranges, width,
                           block_bits) -> np.ndarray:
        """The ``columns`` of ``table`` (``(address, words)`` pairs, every address live, each
        entry below ``2**width`` in 32-bit words, least significant first): each entry
        big-endian at its column and position, zero elsewhere. ``ValueError`` for an invalid
        table."""
        widths, ranges = _check_address(address_widths, live_ranges)
        entries = _entries(_check_table(table, widths, ranges, width))
        columns = np.zeros(QROAMRead.columns_shape(widths, ranges, width, block_bits), dtype=bool)
        for address, value in entries.items():
            start = (address % 2**block_bits) * width
            columns[address >> block_bits, start:start + width] = [
                (value >> (width - 1 - i)) & 1 for i in range(width)]
        return columns

    @staticmethod
    def loaded_bits(table, address, *, address_widths, live_ranges, width, block_bits) -> int:
        """``target ‖ junk`` as one big-endian integer after an enabled read at ``address``;
        0 outside the loaded set."""
        return QROAMRead.loaded_contents(table, [address], address_widths=address_widths,
                                         live_ranges=live_ranges, width=width,
                                         block_bits=block_bits)[0]

    @staticmethod
    def loaded_contents(table, addresses, *, address_widths, live_ranges, width,
                        block_bits) -> list[int]:
        """:meth:`loaded_bits` of each address in ``addresses``, validating the table once."""
        widths, ranges = _check_address(address_widths, live_ranges)
        table = _check_table(table, widths, ranges, width)
        return _loaded_contents(_entries(table), widths, ranges, width, block_bits, addresses)


def _qroam_read_resources(columns, address_widths, live_ranges, width, block_bits,
                          control_value, address, target, junk, work, control):
    num_columns, num_controls = QROAMRead.select_size(address_widths, live_ranges, block_bits,
                                                      control_value)
    size = 2**block_bits * width
    select = resource_key(lambda: qp.QROM(Bool[num_columns, size], Wire[num_controls],
                                          Wire[size], Wire[max(num_controls - 1, 0)], True))
    resources = {select: 1, qp.X: 2 * int(control_value == 1),
                 **cswap_resources((2**block_bits - 1) * width)}
    return {key: count for key, count in resources.items() if count}


@qp.register_resources(_qroam_read_resources)
def _qroam_read(columns, address_widths, live_ranges, width, block_bits, control_value, address,
                target, junk, work, control):
    high = address[:len(address) - block_bits]
    controls = _join(control, high) if control_value is not None else high
    if control_value == 1:
        qp.X(control[0])
    qp.QROM(columns, control_wires=controls, target_wires=_join(target, junk),
            work_wires=work[:max(len(controls) - 1, 0)], clean=True)
    if control_value == 1:
        qp.X(control[0])
    registers = [target] + [junk[r * width:(r + 1) * width] for r in range(2**block_bits - 1)]
    low = address[len(address) - block_bits:]
    for b in reversed(range(block_bits)):
        for r in range(2**b):
            for i in range(width):
                cswap(low[block_bits - 1 - b], registers[r][i], registers[r + 2**b][i],
                      work[0])


register_rule(QROAMRead, _qroam_read)


# --------------------------------------------------------------------------- PhaseFixup


def _parity_array(parities, size):
    """Canonicalize the dynamic ``parities`` once (L-23): a bool array under capture; in tape
    mode an object array of ``MeasurementValue``s, so both modes emit one conditional per covered
    address (K-10). Abstract arrays and tracers pass unchanged."""
    if isinstance(parities, AbstractArray) or qp.math.is_tracer(parities):
        if tuple(parities.shape) != (size,):
            raise ValueError(f"parities must have shape ({size},), got {tuple(parities.shape)}")
        return parities
    values = list(parities)
    if len(values) != size:
        raise ValueError(f"need one parity per covered address ({size}), got {len(values)}")
    if qp.capture.enabled():
        return jnp.stack([jnp.asarray(value, dtype=bool) for value in values])
    for i, value in enumerate(values):
        if not isinstance(value, MeasurementValue):
            raise ValueError(f"in tape mode every parity must be a MeasurementValue; entry {i} "
                             f"is {value!r}")
    return np.asarray(values, dtype=object)


class PhaseFixup(Operator2):
    r"""Multiply each address of the covered set by :math:`(-1)^{f(a)}`, with ``f`` a dynamic
    array; the standalone form of :func:`emit_phase_fixup`.

    **Action.** :math:`|a\rangle \mapsto (-1)^{f(a)}|a\rangle` for ``a`` in the covered set, and
    nothing elsewhere; no other phase (with an enable: only where the enable is satisfied). The
    covered set is the product of the ``covered`` ranges; for an erase it is the read's loaded
    set, :meth:`QROAMRead.loaded_ranges`.

    **Registers.**

    =========== ========================================= ======== ========================
    name        width                                     input    output
    =========== ========================================= ======== ========================
    ``address`` ``sum(address_widths)``                   any      phased
    ``onehot``  ``2**onehot_bits`` (0 if ``onehot_bits``  ``|0>``  ``|0>``
                is 0)
    ``work``    unary-iteration ancillas of the high part ``|0>``  ``|0>``
    ``control`` 0 wires, or 1 (the enable)                any      unchanged
    =========== ========================================= ======== ========================

    **Arguments.** Compilable: ``address_widths`` and ``covered`` (one ``(lo, hi)`` per register,
    the address model of :class:`QROAMRead`), ``onehot_bits`` (ob, ``0 <= ob <=
    sum(address_widths)``), ``control_value`` (``None``, 0 or 1). Dynamic: ``parities``, one
    boolean per covered address in the order of :meth:`address_order` (increasing addresses).
    Under capture it becomes a bool array; in tape mode every entry must be a
    ``MeasurementValue``, so that both modes emit a conditional for every covered address.

    **Pre- and postconditions.** ``onehot`` and ``work`` start and end in ``|0>``.

    **Decomposition** (spec 9.4). (1) The one-hot encoding of the low ob address bits: copy the
    top low bit and its negation into two flags, then split every flag ``f`` on each further bit
    ``a_j`` into ``f AND a_j`` (one ``TemporaryAND``) and ``f AND NOT a_j`` (one CNOT), so that
    ``onehot[j] = [low bits == j]``. (2) Exact unary iteration over the high part, visiting only
    high values with a covered address (:meth:`high_ranges`); at leaf ``h``, for each covered
    ``a = (h << ob) | j``, ``qp.cond(parities[a], CZ)(leaf flag, onehot[j])`` with
    ``qp.hint({"branch-prob": 0.5})``. (3) The measured uncompute of the tree. With
    ``onehot_bits = 0`` it is plain iteration with a conditional ``Z`` on every leaf flag.

    **Adjoint and control.** Never adjointed or controlled; use the enable.

    **Cost.** :func:`dfthc.costs.phase_fixup` with ``high_ranges(...)``: :math:`T = U` (or
    :math:`U_2`) over the covered high ranges, controlled if there is an enable, plus
    :math:`O(ob) = 2^{ob} - 2` (0 for ob = 0); ``M = 0``. At FeMoco-54: outer 22 + 14, inner
    70 + 126.

    **Paper and dfthcbe.** The phase fixup of the measured QROAM erase (Berry et al., Quantum 3,
    208, App. C), with ``CZ`` gates applied directly instead of a parity load into a
    :math:`|-\rangle` ancilla (spec 4.4, item 2). dfthcbe: the erase fixups of its unprepares.

    **Example.**

    .. code-block:: python

        import numpy as np
        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import PhaseFixup

        program = lambda: PhaseFixup(np.ones(6, dtype=bool), (3,), ((0, 6),), 1, None,
                                     address=[0, 1, 2], onehot=[3, 4], work=[5], control=[])
        compiled_resources(program, num_wires=6).toffoli  # U(0, 3, 2, 0) + O(1) = 2
    """

    dynamic_argnames = ("parities",)
    compilable_argnames = ("address_widths", "covered", "onehot_bits", "control_value")
    wire_argnames = ("address", "onehot", "work", "control")

    def __init__(self, parities, address_widths, covered, onehot_bits, control_value, address,
                 onehot, work, control):
        address_widths, covered = _check_address(address_widths, covered)
        onehot_bits = check_int("onehot_bits", onehot_bits, low=0, high=sum(address_widths))
        control_value = _check_control_value(control_value)
        parities = _parity_array(parities, len(self.address_order(address_widths, covered)))
        check_wires("PhaseFixup",
                    {"address": address, "onehot": onehot, "work": work, "control": control},
                    self.required_wires(address_widths, covered, onehot_bits, control_value))
        super().__init__(parities, address_widths, covered, onehot_bits, control_value, address,
                         onehot, work, control)

    @staticmethod
    def required_wires(address_widths, covered, onehot_bits, control_value) -> dict[str, int]:
        """Exact register sizes."""
        ranges = PhaseFixup.high_ranges(address_widths, covered, onehot_bits)
        return {"address": sum(address_widths), "onehot": 2**onehot_bits if onehot_bits else 0,
                "work": work_required([n for _, _, n in ranges], control_value is not None),
                "control": int(control_value is not None)}

    @staticmethod
    def address_order(address_widths, covered) -> list[int]:
        """The covered addresses in the canonical order of ``parities``: increasing."""
        widths, covered = _check_address(address_widths, covered)
        return [_concatenate(values, widths)
                for values in itertools.product(*(range(lo, hi) for lo, hi in covered))]

    @staticmethod
    def high_ranges(address_widths, covered, onehot_bits) -> list[tuple[int, int, int]]:
        """``(lo, hi, n)`` of each iterated high register: the ``high_ranges`` of
        :func:`dfthc.costs.phase_fixup`."""
        widths, covered = _check_address(address_widths, covered)
        return [(lo, hi, n) for _, n, lo, hi in _high_specs(widths, covered, onehot_bits)]


def _onehot_compute(emit, low, onehot):
    size = 2 ** len(low)
    emit.cnot(low[0], onehot[size // 2])
    emit.x(onehot[0])
    emit.cnot(low[0], onehot[0])
    for t in range(1, len(low)):
        step = size >> t
        for s in range(0, size, step):
            emit.and_(onehot[s], low[t], onehot[s + step // 2])
            emit.cnot(onehot[s + step // 2], onehot[s])


def _onehot_uncompute(emit, low, onehot):
    size = 2 ** len(low)
    for t in reversed(range(1, len(low))):
        step = size >> t
        for s in reversed(range(0, size, step)):
            emit.cnot(onehot[s + step // 2], onehot[s])
            emit.unand(onehot[s], low[t], onehot[s + step // 2])
    emit.cnot(low[0], onehot[0])
    emit.x(onehot[0])
    emit.cnot(low[0], onehot[size // 2])


def emit_phase_fixup(parities, *, address, address_widths, covered, onehot, onehot_bits, work,
                     control=None, control_value=1, emit=None) -> None:
    r"""Emit the gates of :class:`PhaseFixup` inside a rule (spec 9.4).

    ``parities`` holds one entry per covered address in :meth:`PhaseFixup.address_order`: a
    predicate for ``qp.cond``, or the Python ``False``, which emits no conditional. Callers decide
    the ``False`` entries from compilable data only (the erase: its zero masks), so both modes
    emit the same gates. ``control`` is the enable wire or ``None``; ``onehot`` has exactly
    ``2**onehot_bits`` wires (none for 0) in ``|0>``; ``work`` has
    ``PhaseFixup.required_wires(...)["work"]`` wires in ``|0>``. ``emit`` is a
    :class:`QROAMEmitter` (default: emitting).
    """
    emit = QROAMEmitter() if emit is None else emit
    widths, covered = _check_address(address_widths, covered)
    if not 0 <= onehot_bits <= sum(widths):
        raise ValueError(f"onehot_bits must be in [0, {sum(widths)}], got {onehot_bits}")
    order = PhaseFixup.address_order(widths, covered)
    if len(parities) != len(order):
        raise ValueError(f"need one parity per covered address ({len(order)}), "
                         f"got {len(parities)}")
    if len(onehot) != (2**onehot_bits if onehot_bits else 0):
        raise ValueError(f"onehot needs {2**onehot_bits if onehot_bits else 0} wires, "
                         f"got {len(onehot)}")
    index = {a: i for i, a in enumerate(order)}
    low = list(address[len(address) - onehot_bits:])
    onehot = list(onehot)
    specs = _high_specs(widths, covered, onehot_bits)
    high_widths = [n for _, n, _, _ in specs]

    def leaf(flag, values):
        block = _concatenate(values, high_widths)
        for j in range(2**onehot_bits):
            i = index.get((block << onehot_bits) | j)
            if i is None or parities[i] is False:
                continue
            if onehot_bits == 0:
                emit.cond_z(parities[i], flag)
            elif flag is None:
                emit.cond_z(parities[i], onehot[j])
            else:
                emit.cond_cz(parities[i], flag, onehot[j])

    if onehot_bits:
        _onehot_compute(emit, low, onehot)
    iterate_nested(_registers(address, specs), work, leaf, control, control_value, emit=emit)
    if onehot_bits:
        _onehot_uncompute(emit, low, onehot)


def _phase_fixup_body(emit, parities, address_widths, covered, onehot_bits, control_value,
                      address, onehot, work, control):
    emit_phase_fixup(parities, address=address, address_widths=address_widths, covered=covered,
                     onehot=onehot, onehot_bits=onehot_bits, work=work, emit=emit,
                     **_control(control, control_value))


def _phase_fixup_resources(parities, address_widths, covered, onehot_bits, control_value,
                           address, onehot, work, control):
    counter = QROAMEmitter(counting=True)
    size = len(PhaseFixup.address_order(address_widths, covered))
    wires = dummy_wires(PhaseFixup.required_wires(address_widths, covered, onehot_bits,
                                                  control_value))
    _phase_fixup_body(counter, [True] * size, address_widths, covered, onehot_bits, control_value,
                      **wires)
    return as_resources(counter.counts)


@qp.register_resources(_phase_fixup_resources)
def _phase_fixup(parities, address_widths, covered, onehot_bits, control_value, address, onehot,
                 work, control):
    size = len(PhaseFixup.address_order(address_widths, covered))
    _phase_fixup_body(QROAMEmitter(), [parities[i] for i in range(size)], address_widths,
                      covered, onehot_bits, control_value, address, onehot, work, control)


register_rule(PhaseFixup, _phase_fixup)


# --------------------------------------------------------------------------- QROAMErase


class QROAMErase(Operator2):
    r"""Erase what the matching :class:`QROAMRead` loaded, by X measurement and a phase fixup.

    **Action.** Returns ``target``, ``junk`` and ``work`` to :math:`|0\rangle` and leaves the
    address state exactly as it was before the read, for every measurement outcome.

    **Registers.**

    ================ =========================================== ============= ==============
    name             width                                       input         output
    ================ =========================================== ============= ==============
    ``address``      ``sum(address_widths)``                     any           unchanged
    ``target``       ``width``                                   the read's    ``|0>``
    ``junk``         ``(K - 1) width``                           the read's    ``|0>``
    ``work``         unary-iteration ancillas of the fixup        ``|0>``       ``|0>``
    ``control``      0 wires, or 1 (the read's enable)           any           unchanged
    ``extra_onehot`` ``max(2**onehot_bits - K width, 0)``        ``|0>``       ``|0>``
    ================ =========================================== ============= ==============

    **Compilable arguments.** The matching read's ``table``, ``address_widths``,
    ``live_ranges``, ``width``, ``block_bits`` and ``control_value``, plus the fixup's
    ``onehot_bits`` (``0 <= ob <= sum(address_widths)``).

    **Pre- and postconditions.** ``target ‖ junk`` hold exactly what the matching read produced
    for the current address (operations in between may act on the address only in ways that
    preserve this). Afterwards ``target``, ``junk``, ``work`` and ``extra_onehot`` are
    :math:`|0\rangle`, and the address state equals its pre-read state exactly. A caller that
    X-measured the junk after the read, before anything else acted on it, erases the target
    with :func:`emit_erase_from_outcomes` instead (spec 16, Q-1 variant (b)).

    **Decomposition** (spec 9.5). (1) ``H`` and ``qp.measure(q, reset=True)`` on every qubit of
    ``target ‖ junk``, giving the outcomes :math:`\xi`. (2) The parities
    :math:`f(a) = \xi \cdot g(a) \bmod 2`, with :math:`g(a) =` :meth:`QROAMRead.loaded_bits`, by
    :func:`dfthc.compile.mcm_parities` from masks built here from the compilable table, for every
    address of the read's loaded set (dead positions included: their junk is not zero). (3)
    :func:`emit_phase_fixup` over the loaded set with the freshly reset ``target ‖ junk`` (and
    ``extra_onehot``) as its one-hot register; a zero mask emits no conditional.

    **Adjoint and control.** Never: its rule measures (spec 7.4, rule 5).

    **Cost.** :func:`dfthc.costs.qroam_erase` with the fixup's ``PhaseFixup.high_ranges`` of
    ``QROAMRead.loaded_ranges``: :math:`T` = the fixup's, :math:`M = 2^k` ``width``. At
    FeMoco-54: outer (ob = 4) T = 22 + 14 = 36, M = 96; inner (ob = 7) T = 70 + 126 = 196,
    M = 416.

    **Paper and dfthcbe.** Steps 21 and 26: the measured erase of Berry et al. (Quantum 3, 208,
    App. C). Its fixups are the phase-fixup terms of PRX Eqs. (B50) (inner) and (B53) (outer),
    whose totals also undo the swaps and the inequality test; in arXiv:2502.15882v1 these are
    Eqs. (123) and (126), and the inner fixup term alone is Eq. (122). dfthcbe: the erases inside
    its unprepares.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import QROAMErase

        table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
        program = lambda: QROAMErase(table, (3,), ((0, 5),), 2, 1, None, 1, address=[0, 1, 2],
                                     target=[3, 4], junk=[5, 6], work=[7], control=[],
                                     extra_onehot=[])
        compiled_resources(program, num_wires=8).measurements  # 4
    """

    compilable_argnames = ("table", "address_widths", "live_ranges", "width", "block_bits",
                           "control_value", "onehot_bits")
    wire_argnames = ("address", "target", "junk", "work", "control", "extra_onehot")

    def __init__(self, table, address_widths, live_ranges, width, block_bits, control_value,
                 onehot_bits, address, target, junk, work, control, extra_onehot):
        address_widths, live_ranges = _check_address(address_widths, live_ranges)
        width = check_int("width", width, low=1)
        block_bits = check_int("block_bits", block_bits, low=0, high=address_widths[-1])
        control_value = _check_control_value(control_value)
        onehot_bits = check_int("onehot_bits", onehot_bits, low=0, high=sum(address_widths))
        table = _check_table(table, address_widths, live_ranges, width)
        check_wires("QROAMErase",
                    {"address": address, "target": target, "junk": junk, "work": work,
                     "control": control, "extra_onehot": extra_onehot},
                    self.required_wires(table, address_widths, live_ranges, width, block_bits,
                                        control_value, onehot_bits))
        super().__init__(table, address_widths, live_ranges, width, block_bits, control_value,
                         onehot_bits, address, target, junk, work, control, extra_onehot)

    @staticmethod
    def required_wires(table, address_widths, live_ranges, width, block_bits, control_value,
                       onehot_bits) -> dict[str, int]:
        """Exact register sizes."""
        covered = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
        fixup = PhaseFixup.required_wires(address_widths, covered, onehot_bits, control_value)
        return {"address": sum(address_widths), "target": width,
                "junk": (2**block_bits - 1) * width, "work": fixup["work"],
                "control": int(control_value is not None),
                "extra_onehot": max(fixup["onehot"] - 2**block_bits * width, 0)}

    @staticmethod
    def fixup_high_ranges(address_widths, live_ranges, block_bits,
                          onehot_bits) -> list[tuple[int, int, int]]:
        """The fixup's iterated ranges: the ``ranges`` of :func:`dfthc.costs.qroam_erase`."""
        covered = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
        return PhaseFixup.high_ranges(address_widths, covered, onehot_bits)


def _fixup_from_outcomes(emit, outcomes, onehot_wires, table, address_widths, live_ranges, width,
                         block_bits, control_value, onehot_bits, address, work, control):
    """Steps 2 and 3 of the erase (spec 9.5): the parities of ``outcomes``, one per qubit of
    ``target ‖ junk`` in wire order, against the loaded bits of every address of the read's loaded
    set (masks built here from the compilable table), then the fixup over that set with the first
    ``2**onehot_bits`` wires of ``onehot_wires``, reset wires, as its one-hot register."""
    covered = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
    order = PhaseFixup.address_order(address_widths, covered)
    masks = _loaded_contents(_entries(table), address_widths, live_ranges, width, block_bits,
                             order)
    parities = emit.parities(outcomes[::-1], masks)
    onehot = list(onehot_wires)[:2**onehot_bits] if onehot_bits else []
    emit_phase_fixup(parities, address=address, address_widths=address_widths, covered=covered,
                     onehot=onehot, onehot_bits=onehot_bits, work=work, emit=emit,
                     **_control(control, control_value))


def _erase_body(emit, table, address_widths, live_ranges, width, block_bits, control_value,
                onehot_bits, address, target, junk, work, control, extra_onehot):
    qubits = list(target) + list(junk)
    outcomes = [emit.measure_x(qubit) for qubit in qubits]
    _fixup_from_outcomes(emit, outcomes, qubits + list(extra_onehot), table, address_widths,
                         live_ranges, width, block_bits, control_value, onehot_bits, address,
                         work, control)


def emit_erase_from_outcomes(junk_outcomes, *, table, address_widths, live_ranges, width,
                             block_bits, onehot_bits, address, target, work, extra_onehot,
                             emit=None) -> None:
    r"""Emit the erase of a :class:`QROAMRead` without an enable whose junk was X-measured
    earlier (spec 16, Q-1 variant (b)), inside a rule.

    ``H`` and ``qp.measure(q, reset=True)`` on every ``target`` qubit, then the parities of the
    concatenated outcomes ``target ‖ junk_outcomes`` and the fixup over the read's loaded set,
    exactly as :class:`QROAMErase` does once it has measured ``target ‖ junk``; the one-hot
    register is the reset ``target ‖ extra_onehot``. ``junk_outcomes`` holds one outcome per junk
    qubit, ``(2**block_bits - 1) width`` of them, in the junk register's wire order: slot ``r`` of
    :meth:`QROAMRead.junk_layout` holds outcomes ``r width`` to ``(r + 1) width - 1``.
    ``extra_onehot`` has ``max(2**onehot_bits - width, 0)`` wires in ``|0>``; ``work`` the fixup's
    ``PhaseFixup.required_wires(...)["work"]``. ``emit`` is a :class:`QROAMEmitter` (default:
    emitting); counting, the outcomes are not read.
    """
    emit = QROAMEmitter() if emit is None else emit
    outcomes = [emit.measure_x(qubit) for qubit in target] + list(junk_outcomes)
    _fixup_from_outcomes(emit, outcomes, list(target) + list(extra_onehot), table,
                         address_widths, live_ranges, width, block_bits, None, onehot_bits,
                         address, work, [])


def _qroam_erase_resources(table, address_widths, live_ranges, width, block_bits, control_value,
                           onehot_bits, address, target, junk, work, control, extra_onehot):
    counter = QROAMEmitter(counting=True)
    wires = dummy_wires(QROAMErase.required_wires(table, address_widths, live_ranges, width,
                                                  block_bits, control_value, onehot_bits))
    _erase_body(counter, table, address_widths, live_ranges, width, block_bits, control_value,
                onehot_bits, **wires)
    return as_resources(counter.counts)


@qp.register_resources(_qroam_erase_resources)
def _qroam_erase(table, address_widths, live_ranges, width, block_bits, control_value,
                 onehot_bits, address, target, junk, work, control, extra_onehot):
    _erase_body(QROAMEmitter(), table, address_widths, live_ranges, width, block_bits,
                control_value, onehot_bits, address, target, junk, work, control, extra_onehot)


register_rule(QROAMErase, _qroam_erase)
