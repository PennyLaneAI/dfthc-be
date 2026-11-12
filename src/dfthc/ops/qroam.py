"""The paper's QROAM read (spec 9.3).

**Address model** (spec 9.3). The address is one or two big-endian registers (spec 6.7),
``address_widths = (w_1,)`` or ``(w_1, w_2)``, concatenated with the first register most
significant: the integer address is ``(a_1 << w_2) | a_2``. Each register has a range
``(lo, hi)``. A table maps integer addresses to entries of ``width`` bits, stored as tuples of
32-bit words, least significant word first (L-24).

**Bit order.** Every register holds an entry big-endian: wire ``i`` of a ``width``-wire register
holds bit ``width - 1 - i``. ``target ‖ junk`` is read as one big-endian integer of
``2**block_bits * width`` bits (``target`` most significant, then the junk slots in order); that
integer is :meth:`QROAMRead.loaded_bits`. The read's dynamic ``leaf_words`` hold the entries in
wire order instead: bit ``s`` of word ``w`` of block position ``p`` is the bit for wire
``32 w + s`` of that position's register, which is entry bit ``width - 1 - (32 w + s)``
(:meth:`QROAMRead.leaf_words_from_table` converts a table).
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import AbstractArray

from dfthc.compile import register_rule
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.unary import Emitter, as_resources, iterate_nested, work_required

WORD = 32
"""Data bits per table word (spec 8.4, L-24)."""

WORD_DTYPE = np.int64
"""dtype of ``leaf_words``: signed, because Catalyst builds no rule for an operator with an
unsigned dynamic argument (L-26); 32 data bits per word keep every word non-negative."""


class QROAMEmitter(Emitter):
    """:class:`~dfthc.ops.unary.Emitter` with the gates of the read."""

    def cswap(self, control, a, b):
        self.gate("CSWAP", lambda: qp.CSWAP([control, a, b]))

    def wire_array(self, target, junk):
        """``target ‖ junk`` as one register that a traced loop index can address: a JAX array
        under capture, a list in tape mode; ``None`` when counting."""
        if self.counts is not None:
            return None
        if not qp.capture.enabled():
            return list(target) + list(junk)
        if len(junk) == 0:
            return jnp.asarray(target)
        return jnp.concatenate([jnp.asarray(target), jnp.asarray(junk)])

    def load(self, flag, leaf_words, leaf, register, positions, width):
        """The loads of unary leaf ``leaf``: one ``qp.for_loop`` over its ``positions * width``
        bits. Iteration ``t`` takes position ``p = t // width`` and bit ``i = t % width`` and
        applies, conditioned on that bit of ``leaf_words``, a CNOT from ``flag`` (an ``X`` for a
        ``None`` flag) onto ``register[p * width + i]``. No conditional surrounds the loop.
        Counting: ``positions * width`` gates, the upper bound."""
        name = "PauliX" if flag is None else "CNOT"
        if self.counts is not None:
            self.counts[name] += positions * width
            return

        @qp.for_loop(0, positions * width, 1)
        def bit_loop(t):
            p, i = t // width, t % width
            bit = (leaf_words[leaf, p, i // WORD] >> (i % WORD)) & 1
            if flag is None:
                qp.cond(bit == 1, qp.X)(register[p * width + i])
            else:
                qp.cond(bit == 1, qp.CNOT)([flag, register[p * width + i]])

        bit_loop()


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


def _dummy(sizes: dict[str, int]) -> dict[str, range]:
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


def _wire_order(value, width) -> int:
    """``value`` with its ``width`` bits reversed: bit ``i`` of the result is the bit for wire
    ``i`` of a big-endian register (spec 6.7)."""
    return int(format(value, f"0{width}b")[::-1], 2)


def _live_positions(widths, live_ranges, block_bits) -> np.ndarray:
    """``(num_leaves, K)`` booleans: which block positions of each leaf hold a live address."""
    blocks = QROAMRead.leaf_blocks(widths, live_ranges, block_bits)
    return np.array([[_is_live((block << block_bits) | p, widths, live_ranges)
                      for p in range(2**block_bits)] for block in blocks], dtype=bool)


def _leaf_word_array(leaf_words, widths, live_ranges, width, block_bits):
    """Canonicalize the dynamic ``leaf_words`` once (L-23).

    Concrete words (Python or NumPy integers, nested sequences or arrays) must be integers in
    ``[0, 2**32)``, booleans and floats rejected, checked on the Python values before the
    ``int64`` cast, so that no word can wrap around before the check; every padding and dead
    position must be zero, and so must every bit beyond ``width`` (wire order). Tracers and
    abstract arrays (operators rebuilt by Catalyst, resource keys) are checked for their shape
    and an integer dtype only. Raises ``ValueError``.
    """
    shape = QROAMRead.leaf_words_shape(widths, live_ranges, width, block_bits)
    if isinstance(leaf_words, AbstractArray) or qp.math.is_tracer(leaf_words):
        if tuple(leaf_words.shape) != shape or not np.issubdtype(np.dtype(leaf_words.dtype),
                                                                 np.integer):
            raise ValueError(f"leaf_words must be an integer array of shape {shape}, got "
                             f"{tuple(leaf_words.shape)} of {leaf_words.dtype}")
        return leaf_words
    values = np.asarray(leaf_words, dtype=object)
    if values.shape != shape:
        raise ValueError(f"leaf_words must have shape {shape}, got {values.shape}")
    for word in values.flat:
        if isinstance(word, bool) or not isinstance(word, (int, np.integer)) or not (
                0 <= word < 2**WORD):
            raise ValueError(f"leaf words must be integers in [0, 2**{WORD}), got {word!r}")
    array = values.astype(WORD_DTYPE)
    if np.any(array[~_live_positions(widths, live_ranges, block_bits)]):
        raise ValueError("a padding or dead position of leaf_words holds a nonzero word")
    spare = WORD * shape[2] - width
    if spare and np.any(array[..., -1] >> (WORD - spare)):
        raise ValueError(f"leaf_words holds bits beyond width {width}")
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
    ``junk``, which stays until the measured erase (:class:`QROAMErase`).

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

    ``T`` is the table that :meth:`leaf_words_from_table` packed into ``leaf_words``; entries
    missing from it are zero. ``junk`` slot ``r`` holds the block position
    ``junk_layout(k, a mod K)[r]``. :meth:`loaded_bits` gives ``target ‖ junk`` for every address.
    The read adds no phase, and the address is unchanged.

    **Registers.**

    =========== ================================= ======== ===============================
    name        width                             input    output
    =========== ================================= ======== ===============================
    ``address`` ``sum(address_widths)``           any      unchanged
    ``target``  ``width``                         ``|0>``  the entry
    ``junk``    ``(K - 1) width``                 ``|0>``  the block's other entries
    ``work``    unary-iteration ancillas          ``|0>``  ``|0>``
    ``control`` 0 wires, or 1 (the enable)        any      unchanged
    =========== ================================= ======== ===============================

    **Arguments.** Dynamic: ``leaf_words``, an ``int64`` array of shape :meth:`leaf_words_shape`,
    ``(num_leaves, K, ceil(width / 32))``: row ``l`` belongs to the ``l``-th unary leaf in
    visiting order (:meth:`leaf_blocks`), position ``p`` to the address
    ``(leaf_blocks[l] << k) | p``; 32 data bits per word in wire order (module docstring); zero
    at padding and dead positions. Callers build it from their compilable table with
    :meth:`leaf_words_from_table`. Compilable: ``address_widths`` (one or two widths);
    ``live_ranges`` (one ``(lo, hi)`` per register); ``width`` (int >= 1); ``block_bits``
    (``0 <= k <= address_widths[-1]``); ``control_value`` (``None`` without an enable wire, else
    the enabling value 0 or 1).

    **Pre- and postconditions.** ``target``, ``junk`` and ``work`` start in ``|0>``; ``work``
    ends in ``|0>``.

    **Decomposition.** (1) Nested exact unary iteration (:mod:`dfthc.ops.unary`) over the visited
    blocks: the first register over its live range, then the top ``w_p - k`` bits of the last
    register over the blocks that meet its live range, rooted at the enable. (2) At each block
    leaf ``l``, one ``qp.for_loop`` over its ``K width`` bits: iteration ``t`` takes position
    ``p = t // width`` and bit ``i = t % width`` and applies
    ``qp.cond(bit == 1, CNOT)(leaf flag, (target ‖ junk)[p width + i])`` with
    ``bit = (leaf_words[l, p, i // 32] >> (i % 32)) & 1`` (an ``X`` instead of the CNOT when the
    whole address is low and there is no enable, so no flag exists). No conditional surrounds a
    loop. (3) A swap-with-zero network on the low ``k`` address bits, most significant first:
    for bit ``b``, ``CSWAP`` register ``r`` with register ``r + 2**b`` for ``r < 2**b``, which
    moves the selected entry into ``target`` with ``(K - 1) width`` exact ``CSWAP``\s.

    **Adjoint and control.** Never adjointed or controlled: its body has loops and data
    conditionals (spec 7.4 rule 4; L-6), and no program wraps a composite that contains it in an
    adjoint or a control (spec 7.4 rule 6). The walk erases with :class:`QROAMErase`; use the
    enable instead of a control.

    **Cost.** :func:`dfthc.costs.qroam_read` with ``ranges = visited_ranges(...)``:
    :math:`T = U` (or :math:`U_2`) over the visited block ranges, controlled if there is an
    enable, plus :math:`(2^k - 1)` ``width``; ``M = 0``; ``A = U``. At FeMoco-54: outer
    84 + 72 = 156, inner 543 + 390 = 933. The rule declares ``num_leaves K width`` load CNOTs,
    or X gates for the flagless leaf (every position and bit, padding included), and is
    registered with ``exact=False``: the CNOT or X count is an upper bound, which ``qp.analyze``
    reports as well, because it counts every unhinted load conditional; T, M and A do not depend
    on the data and are exact.

    **Paper and dfthcbe.** Steps 2 and 6 (and the engine of 10 and 11); the QROAM of Berry et al.
    (Quantum 3, 208) with the junk kept until a measured erase. dfthcbe: the reads in
    ``OuterPrepare``/``InnerPrepare``; its truncated iteration and flat ``r`` target are
    deferred refinements (spec 9.3, "Known deltas").

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import QROAMRead

        table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
        shape = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
        words = QROAMRead.leaf_words_from_table(table, **shape)  # shape (3, 2, 1)
        program = lambda: QROAMRead(words, *shape.values(), None, address=[0, 1, 2],
                                    target=[3, 4], junk=[5, 6], work=[7], control=[])
        compiled_resources(program, num_wires=8).toffoli  # U(0, 3, 2, 0) + 2 = 2 + 2
    """

    dynamic_argnames = ("leaf_words",)
    compilable_argnames = ("address_widths", "live_ranges", "width", "block_bits",
                           "control_value")
    wire_argnames = ("address", "target", "junk", "work", "control")

    def __init__(self, leaf_words, address_widths, live_ranges, width, block_bits, control_value,
                 address, target, junk, work, control):
        address_widths, live_ranges = _check_address(address_widths, live_ranges)
        width = check_int("width", width, low=1)
        block_bits = check_int("block_bits", block_bits, low=0, high=address_widths[-1])
        control_value = _check_control_value(control_value)
        leaf_words = _leaf_word_array(leaf_words, address_widths, live_ranges, width, block_bits)
        check_wires("QROAMRead",
                    {"address": address, "target": target, "junk": junk, "work": work,
                     "control": control},
                    self.required_wires(address_widths, live_ranges, width, block_bits,
                                        control_value))
        super().__init__(leaf_words, address_widths, live_ranges, width, block_bits,
                         control_value, address, target, junk, work, control)

    @staticmethod
    def required_wires(address_widths, live_ranges, width, block_bits,
                       control_value) -> dict[str, int]:
        """Exact register sizes."""
        ranges = QROAMRead.visited_ranges(address_widths, live_ranges, block_bits)
        return {"address": sum(address_widths), "target": width,
                "junk": (2**block_bits - 1) * width,
                "work": work_required([n for _, _, n in ranges], control_value is not None),
                "control": int(control_value is not None)}

    @staticmethod
    def visited_ranges(address_widths, live_ranges, block_bits) -> list[tuple[int, int, int]]:
        """``(lo, hi, n)`` of each iterated register: the ``ranges`` of
        :func:`dfthc.costs.qroam_read`."""
        widths, ranges = _check_address(address_widths, live_ranges)
        return [(lo, hi, n) for _, n, lo, hi in _high_specs(widths, ranges, block_bits)]

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
    def leaf_blocks(address_widths, live_ranges, block_bits) -> list[int]:
        """The high part of every unary leaf (its block), in the order the iteration visits the
        leaves: outermost register first, values increasing. Row ``l`` of ``leaf_words``."""
        widths, ranges = _check_address(address_widths, live_ranges)
        specs = _high_specs(widths, ranges, block_bits)
        return [_concatenate(values, [n for _, n, _, _ in specs])
                for values in itertools.product(*(range(lo, hi) for _, _, lo, hi in specs))]

    @staticmethod
    def leaf_words_shape(address_widths, live_ranges, width, block_bits) -> tuple[int, int, int]:
        """``(num_leaves, K, ceil(width / 32))``, the shape of ``leaf_words``."""
        return (len(QROAMRead.leaf_blocks(address_widths, live_ranges, block_bits)),
                2**block_bits, -(-width // WORD))

    @staticmethod
    def leaf_words_from_table(table, *, address_widths, live_ranges, width,
                              block_bits) -> np.ndarray:
        """The ``leaf_words`` of ``table`` (``(address, words)`` pairs, every address live, each
        entry below ``2**width`` in 32-bit words, least significant first): each entry in wire
        order at its leaf and position, zero at padding and dead positions. ``ValueError`` for an
        invalid table."""
        widths, ranges = _check_address(address_widths, live_ranges)
        entries = _entries(_check_table(table, widths, ranges, width))
        words = np.zeros(QROAMRead.leaf_words_shape(widths, ranges, width, block_bits),
                         dtype=WORD_DTYPE)
        for leaf, block in enumerate(QROAMRead.leaf_blocks(widths, ranges, block_bits)):
            for position in range(2**block_bits):
                stream = _wire_order(entries.get((block << block_bits) | position, 0), width)
                for w in range(words.shape[2]):
                    words[leaf, position, w] = (stream >> (WORD * w)) & (2**WORD - 1)
        return words

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


def _read_body(emit, leaf_words, address_widths, live_ranges, width, block_bits, control_value,
               address, target, junk, work, control):
    registers = [target] + [junk[r * width:(r + 1) * width] for r in range(2**block_bits - 1)]
    register = emit.wire_array(target, junk)
    leaves = itertools.count()

    def leaf(flag, values):
        emit.load(flag, leaf_words, next(leaves), register, 2**block_bits, width)

    iterate_nested(_registers(address, _high_specs(address_widths, live_ranges, block_bits)),
                   work, leaf, emit=emit, **_control(control, control_value))
    low = address[len(address) - block_bits:]
    for b in reversed(range(block_bits)):
        for r in range(2**b):
            for i in range(width):
                emit.cswap(low[block_bits - 1 - b], registers[r][i], registers[r + 2**b][i])


def _qroam_read_resources(leaf_words, address_widths, live_ranges, width, block_bits,
                          control_value, address, target, junk, work, control):
    counter = QROAMEmitter(counting=True)
    wires = _dummy(QROAMRead.required_wires(address_widths, live_ranges, width, block_bits,
                                            control_value))
    _read_body(counter, leaf_words, address_widths, live_ranges, width, block_bits,
               control_value, **wires)
    return as_resources(counter.counts)


@qp.register_resources(_qroam_read_resources, exact=False)
def _qroam_read(leaf_words, address_widths, live_ranges, width, block_bits, control_value,
                address, target, junk, work, control):
    _read_body(QROAMEmitter(), leaf_words, address_widths, live_ranges, width, block_bits,
               control_value, address, target, junk, work, control)


register_rule(QROAMRead, _qroam_read)
