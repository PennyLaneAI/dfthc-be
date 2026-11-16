"""The angle registers of the inner select: loads and measured erase (spec 9.10, 9.11, 9.17).

**The shared angle register** (spec 6.7). ``angles`` is the concatenation of ``N - 1`` registers
of ``b_rot`` wires; register ``k`` drives rung ``k`` of the Givens ladder and holds the angle
integer ``x_k`` big-endian, with ``x_k < 2**(b_rot - 1)`` (spec 6.3). :func:`angle_entry` is the
big-endian integer of the whole register, register 0 most significant.

**Two exclusive loads.** :class:`TailAngleLoad` writes the one-body rows ``x_o`` in
``[RC, X_o)``: their angles, ``G0 = 1`` and ``G1 = [x_o - RC >= N+]``. :class:`SFAngleLoad` then
writes, where ``G0 = 0``, the angles of ``(r, b)`` in ``[0, R) x [0, B)`` and ``b_neq_B = 1``.
Each is a table load by exact unary iteration (:mod:`dfthc.ops.unary`) over its live ranges, the
address ranges and nesting that ``qp.QROM`` does not offer (spec 9.3): at every leaf one
``qp.for_loop`` over the target loads the leaf's entry, one big-endian integer over the target
(``angles ‖ G0 ‖ G1``, ``angles ‖ b_neq_B``), from a bool array of rows that the rule builds from
the compilable angle table. After both loads, tail rows hold tail data on ``G0 = 1``; every
``G0 = 0`` branch, failed ``x_o`` included, holds the SF entry of ``(r, b)`` where it is in
range, and zeros elsewhere (``b = B``, failed ``b``, unvisited ``r``). :class:`AngleErase`
(Task 2E.2) erases what they write.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2

from dfthc.compile import register_rule
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.qroam import (
    PhaseFixup,
    QROAMEmitter,
    _high_specs,
    _registers,
    dummy_wires,
    emit_phase_fixup,
)
from dfthc.ops.unary import as_resources, iterate_nested, work_required
from dfthc.params import lg


def angle_entry(angles, b_rot: int) -> int:
    """The big-endian integer of the angle registers holding ``angles``: register ``k`` holds
    ``angles[k]``, register 0 most significant (spec 6.7)."""
    value = 0
    for angle in angles:
        value = (value << b_rot) | angle
    return value


def _check_angle_table(name: str, table, shape: tuple[int, ...], b_rot: int) -> tuple:
    """``table`` as nested tuples of Python ints of ``shape``, every angle in
    ``[0, 2**(b_rot - 1))`` (spec 6.3); ``ValueError`` otherwise."""
    table = canonical_ints(name, table)

    def check(value, dims):
        if not dims:
            if not isinstance(value, int):
                raise ValueError(f"{name} must be a table of ints of shape {shape}")
            if not 0 <= value < 2 ** (b_rot - 1):
                raise ValueError(f"{name} angles must lie in [0, 2**(b_rot - 1)) = "
                                 f"[0, {2 ** (b_rot - 1)}), got {value}")
            return
        if not isinstance(value, tuple) or len(value) != dims[0]:
            raise ValueError(f"{name} must be a table of ints of shape {shape}")
        for item in value:
            check(item, dims[1:])

    check(table, shape)
    return table


def _join(*registers) -> list:
    """The wires of ``registers``, concatenated in order."""
    return [wire for register in registers for wire in register]


def _rows(entries, width: int) -> np.ndarray:
    """``entries`` as a bool array, one row per entry, big-endian over ``width`` bits (spec
    6.7: bit ``i`` of a row is entry bit ``width - 1 - i``, the bit for target wire ``i``)."""
    return np.array([[(entry >> (width - 1 - i)) & 1 for i in range(width)] for entry in entries],
                    dtype=bool)


def _load_work(shape: dict, controlled: bool) -> int:
    """The work wires of a load over ``shape``'s address: its unary iteration's."""
    return work_required(shape["address_widths"], controlled)


def _load_body(emit, entries, shape, address, target, work, control=None, control_value=1):
    """A table load (module docstring): exact unary iteration over ``shape``'s live ranges,
    leaves in increasing address order, rooted at the enable ``control``; at leaf ``l`` the
    loads of ``entries[l]`` onto ``target`` (:meth:`QROAMEmitter.load`)."""
    width = shape["width"]
    rows = _rows(entries, width)
    if emit.counts is None and qp.capture.enabled():
        rows = jnp.asarray(rows)
    register = emit.wire_array(target)
    leaves = iter(range(len(entries)))

    def leaf(flag, values):
        emit.load(flag, rows, next(leaves), register)

    specs = _high_specs(shape["address_widths"], shape["live_ranges"], 0)
    iterate_nested(_registers(address, specs), work, leaf, control, control_value, emit=emit)


# --------------------------------------------------------------------------- TailAngleLoad


def _tail_entries(tail_angles, n_plus: int, b_rot: int) -> list[int]:
    """Entry of tail row ``j``: ``angles ‖ G0 ‖ G1`` big-endian, ``G0 = 1``, ``G1 = [j >= N+]``."""
    return [(angle_entry(row, b_rot) << 2) | 2 | int(j >= n_plus)
            for j, row in enumerate(tail_angles)]


class TailAngleLoad(Operator2):
    r"""Load the one-body rows' Givens angles and the family flags ``G0``, ``G1`` (step 10).

    **Action.** For ``x_o`` in ``[RC, X_o)``, with ``j = x_o - RC``: the angle registers get
    ``tail_angles[j]`` (XOR), ``G0`` gets 1 and ``G1`` gets :math:`[j \ge N_+]` (D1 rows first,
    then Q1; spec 6.1). Every other ``x_o`` (the SF rows ``x_o < RC`` and the failed
    ``x_o >= X_o``) is untouched. ``x_o`` is unchanged and no phase is added.

    **Registers.** With ``N = len(tail_angles)`` and :math:`w_x = \lg X_o`:

    ========== ======================= ======== =========================================
    name       width                   input    output
    ========== ======================= ======== =========================================
    ``x_o``    :math:`w_x`             any      unchanged
    ``angles`` :math:`(N - 1) b_{rot}` ``|0>``  ``tail_angles[x_o - RC]`` on tail rows
    ``G0``     1                       ``|0>``  :math:`[RC \le x_o < X_o]`
    ``G1``     1                       ``|0>``  :math:`[x_o - RC \ge N_+]` on tail rows
    ``work``   :math:`w_x - 1`         ``|0>``  ``|0>``
    ========== ======================= ======== =========================================

    **Compilable arguments.** ``tail_angles``: ``N x (N - 1)`` ints in
    ``[0, 2**(b_rot - 1))``, the ``N+`` D1 rows first (``WalkTables.tail_angles``), ``N >= 2``;
    ``n_plus`` (``0 <= N+ <= N``); ``RC`` (int >= 1); ``X_o`` (``RC + N``); ``b_rot``
    (int >= 2). ``b_rot`` sizes the angle registers; it is not in the master plan's interface
    but determines the structure (spec 7.4 rules 1 and 3).

    **Pre- and postconditions.** ``angles``, ``G0``, ``G1`` and ``work`` start in ``|0>``;
    ``work`` ends in ``|0>``. After both loads, :class:`AngleErase` erases the shared
    ``angles``, ``b_neq_B``, ``G0`` and ``G1`` registers.

    **Decomposition.** A table load (module docstring) with address ``x_o`` over ``[RC, X_o)``,
    no enable, and target ``angles ‖ G0 ‖ G1`` (:meth:`load_shape`): exact unary iteration
    (:func:`~dfthc.ops.unary.iterate_nested`), and at the leaf of row ``j`` one ``qp.for_loop``
    over the target that applies ``qp.cond(bit, CNOT)(leaf flag, target[i])`` for bit ``i`` of
    the row's entry, from a bool array the rule builds from the compilable ``tail_angles``
    (spec 9.3).

    **Adjoint and control.** Never adjointed or controlled (it has loops; spec 7.4 rules 4 and
    6). :class:`AngleErase` undoes it by measurement.

    **Cost.** :func:`dfthc.costs.tail_angle_load`: :math:`T = U(RC, X_o, w_x, 0)` (``A = T``),
    ``M = 0``. The rule declares a load CNOT for every bit of every row and is registered with
    ``exact=False``: the CNOT count is an upper bound, which ``qp.analyze`` reports as well,
    because it counts every unhinted load conditional. At FeMoco-54: 61, with 54 rows of 797
    bits.

    **Paper and dfthcbe.** Step 10, the D1/Q1 angle load (PRX Eq. (B41)); dfthcbe:
    ``tail_rotation_qrom`` (``dfthcbe/rot_data.py``), which prunes the SF prefix to aligned
    blocks (56 at FeMoco-54) where ours iterates exactly (spec 9.3, "Known deltas").

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.angle_data import TailAngleLoad

        tail = ((1,), (2,))  # N = 2, b_rot = 3
        program = lambda: TailAngleLoad(tail, 1, 2, 4, 3, x_o=[0, 1], angles=[2, 3, 4], G0=[5],
                                        G1=[6], work=[7])
        compiled_resources(program, num_wires=8).toffoli  # U(2, 4, 2, 0) = 1
    """

    compilable_argnames = ("tail_angles", "n_plus", "RC", "X_o", "b_rot")
    wire_argnames = ("x_o", "angles", "G0", "G1", "work")

    def __init__(self, tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
        tail_angles, n_plus, RC, X_o, b_rot = self._check(tail_angles, n_plus, RC, X_o, b_rot)
        check_wires("TailAngleLoad",
                    {"x_o": x_o, "angles": angles, "G0": G0, "G1": G1, "work": work},
                    self.required_wires(tail_angles, n_plus, RC, X_o, b_rot))
        super().__init__(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work)

    @staticmethod
    def _check(tail_angles, n_plus, RC, X_o, b_rot) -> tuple:
        b_rot = check_int("b_rot", b_rot, low=2)
        rows = canonical_ints("tail_angles", tail_angles)
        if not isinstance(rows, tuple):
            raise ValueError("tail_angles must be a table of ints of shape (N, N - 1)")
        num_rows = len(rows)
        if num_rows < 2:
            raise ValueError(f"tail_angles needs N >= 2 rows, got {num_rows}")
        tail_angles = _check_angle_table("tail_angles", rows, (num_rows, num_rows - 1), b_rot)
        n_plus = check_int("n_plus", n_plus, low=0, high=num_rows)
        RC = check_int("RC", RC, low=1)
        X_o = check_int("X_o", X_o, low=RC + num_rows, high=RC + num_rows)
        return tail_angles, n_plus, RC, X_o, b_rot

    @staticmethod
    def load_shape(N: int, RC: int, X_o: int, b_rot: int) -> dict:
        """The load's ``address_widths``, ``live_ranges`` and ``width``."""
        return {"address_widths": (lg(X_o),), "live_ranges": ((RC, X_o),),
                "width": (N - 1) * b_rot + 2}

    @staticmethod
    def required_wires(tail_angles, n_plus, RC, X_o, b_rot) -> dict[str, int]:
        """Exact register sizes."""
        shape = TailAngleLoad.load_shape(len(tail_angles), RC, X_o, b_rot)
        return {"x_o": lg(X_o), "angles": (len(tail_angles) - 1) * b_rot, "G0": 1, "G1": 1,
                "work": _load_work(shape, controlled=False)}


def _tail_body(emit, tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
    _load_body(emit, _tail_entries(tail_angles, n_plus, b_rot),
               TailAngleLoad.load_shape(len(tail_angles), RC, X_o, b_rot), x_o,
               _join(angles, G0, G1), work)


def _tail_angle_load_resources(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
    counter = QROAMEmitter(counting=True)
    wires = dummy_wires(TailAngleLoad.required_wires(tail_angles, n_plus, RC, X_o, b_rot))
    _tail_body(counter, tail_angles, n_plus, RC, X_o, b_rot, **wires)
    return as_resources(counter.counts)


@qp.register_resources(_tail_angle_load_resources, exact=False)
def _tail_angle_load(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
    _tail_body(QROAMEmitter(), tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work)


register_rule(TailAngleLoad, _tail_angle_load)


# --------------------------------------------------------------------------- SFAngleLoad


def _sf_address(R: int, B: int) -> tuple[tuple, tuple]:
    """``(address_widths, ranges)`` of ``r ‖ b`` over ``[0, R) x [0, B)``; only ``b`` if
    ``R = 1`` (then ``r`` has no wires)."""
    if lg(R) == 0:
        return (lg(B + 1),), ((0, B),)
    return (lg(R), lg(B + 1)), ((0, R), (0, B))


def _sf_entries(sf_angles, b_rot: int) -> list[list[int]]:
    """Entry of ``(r, b)``: ``angles ‖ b_neq_B`` big-endian, ``b_neq_B = 1``."""
    return [[(angle_entry(entry, b_rot) << 1) | 1 for entry in plane] for plane in sf_angles]


class SFAngleLoad(Operator2):
    r"""Load the SF Givens angles and ``b_neq_B`` at ``(r, b)``, enabled on ``G0 = 0`` (step 11).

    **Action.** Where ``G0 = 0`` and ``(r, b)`` lies in ``[0, R) x [0, B)``: the angle registers
    get ``sf_angles[r][b]`` (XOR) and ``b_neq_B`` gets 1. Everywhere else (``G0 = 1``, the
    identity term ``b = B``, failed ``b > B``, ``r >= R``) nothing changes. ``r``, ``b`` and
    ``G0`` are unchanged and no phase is added. In the walk it follows :class:`TailAngleLoad`,
    which sets ``G0`` on the one-body rows, so the two loads never write the same branch.

    **Registers.** With :math:`w_r = \lg R` and :math:`w_b = \lg(B + 1)`:

    =========== ======================= ======== =========================================
    name        width                   input    output
    =========== ======================= ======== =========================================
    ``r``       :math:`w_r` (0 if R = 1) any     unchanged
    ``b``       :math:`w_b`             any      unchanged
    ``G0``      1                       any      unchanged (the enable, value 0)
    ``angles``  :math:`(N - 1) b_{rot}` ``|0>``  ``sf_angles[r][b]`` where enabled and in range
    ``b_neq_B`` 1                       ``|0>``  1 where enabled and in range
    ``work``    :math:`w_r + w_b`       ``|0>``  ``|0>``
    =========== ======================= ======== =========================================

    **Compilable arguments.** ``sf_angles``: ``R x B x (N - 1)`` ints in
    ``[0, 2**(b_rot - 1))`` (``WalkTables.sf_angles``), ``N >= 2``; ``R`` and ``B`` (ints >= 1,
    its first two sizes); ``b_rot`` (int >= 2), which sizes the angle registers (not in the
    master plan's interface; spec 7.4 rules 1 and 3).

    **Pre- and postconditions.** On enabled branches ``angles`` and ``b_neq_B`` start in
    ``|0>`` (they hold the tail load's data only where ``G0 = 1``); ``work`` starts and ends in
    ``|0>``. :class:`AngleErase` erases ``angles`` and ``b_neq_B``.

    **Decomposition.** A table load (module docstring) with address ``r ‖ b`` (``r`` outer over
    ``[0, R)``, ``b`` inner over ``[0, B)``; ``b`` alone if ``R = 1``), the enable ``G0`` with
    control value 0 as the iteration's root, and target ``angles ‖ b_neq_B``
    (:meth:`load_shape`): nested exact unary iteration, and at the leaf of ``(r, b)`` one
    ``qp.for_loop`` of conditional CNOTs that loads its entry, from a bool array the rule builds
    from the compilable ``sf_angles`` (spec 9.3).

    **Adjoint and control.** Never adjointed or controlled (it has loops; spec 7.4 rules 4 and
    6).

    **Cost.** :func:`dfthc.costs.sf_angle_load`:
    :math:`T = U(0, R, w_r, 1) + R\,U(0, B, w_b, 1)` (``A = T``), ``M = 0``; the CNOT count is an
    upper bound (``exact=False``, as for :class:`TailAngleLoad`). At FeMoco-54:
    :math:`11 + 10 \cdot 28 = 291`, with 270 rows of 796 bits.

    **Paper and dfthcbe.** Step 11, the SF angle load (PRX Table III, :math:`RB`); dfthcbe:
    ``sf_rotation_qrom`` (``dfthcbe/rot_data.py``), nested ``b`` outer and ``r`` inner with the
    identity row pruned (271 at FeMoco-54); ours nests ``r`` outer and iterates exactly.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.angle_data import SFAngleLoad

        sf = (((1,), (2,)), ((3,), (0,)))  # R = 2, B = 2, N = 2, b_rot = 3
        program = lambda: SFAngleLoad(sf, 2, 2, 3, r=[0], b=[1, 2], G0=[3], angles=[4, 5, 6],
                                      b_neq_B=[7], work=[8, 9, 10])
        compiled_resources(program, num_wires=11).toffoli  # U(0, 2, 1, 1) + 2 U(0, 2, 2, 1) = 5
    """

    compilable_argnames = ("sf_angles", "R", "B", "b_rot")
    wire_argnames = ("r", "b", "G0", "angles", "b_neq_B", "work")

    def __init__(self, sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
        sf_angles, R, B, b_rot = self._check(sf_angles, R, B, b_rot)
        check_wires("SFAngleLoad",
                    {"r": r, "b": b, "G0": G0, "angles": angles, "b_neq_B": b_neq_B,
                     "work": work},
                    self.required_wires(sf_angles, R, B, b_rot))
        super().__init__(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work)

    @staticmethod
    def _check(sf_angles, R, B, b_rot) -> tuple:
        b_rot = check_int("b_rot", b_rot, low=2)
        R, B = check_int("R", R, low=1), check_int("B", B, low=1)
        planes = canonical_ints("sf_angles", sf_angles)
        try:
            num_rungs = len(planes[0][0])
        except (IndexError, TypeError):
            raise ValueError(f"sf_angles must be a table of shape (R, B, N - 1) = ({R}, {B}, "
                             "N - 1)") from None
        if num_rungs < 1:
            raise ValueError("sf_angles needs N - 1 >= 1 angles per entry")
        return _check_angle_table("sf_angles", planes, (R, B, num_rungs), b_rot), R, B, b_rot

    @staticmethod
    def load_shape(N: int, R: int, B: int, b_rot: int) -> dict:
        """The load's ``address_widths``, ``live_ranges`` and ``width``."""
        widths, ranges = _sf_address(R, B)
        return {"address_widths": widths, "live_ranges": ranges, "width": (N - 1) * b_rot + 1}

    @staticmethod
    def required_wires(sf_angles, R, B, b_rot) -> dict[str, int]:
        """Exact register sizes."""
        num_modes = len(sf_angles[0][0]) + 1
        shape = SFAngleLoad.load_shape(num_modes, R, B, b_rot)
        return {"r": lg(R), "b": lg(B + 1), "G0": 1, "angles": (num_modes - 1) * b_rot,
                "b_neq_B": 1, "work": _load_work(shape, controlled=True)}


def _sf_body(emit, sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
    entries = [entry for plane in _sf_entries(sf_angles, b_rot) for entry in plane]
    _load_body(emit, entries, SFAngleLoad.load_shape(len(sf_angles[0][0]) + 1, R, B, b_rot),
               _join(r, b), _join(angles, b_neq_B), work, control=G0[0], control_value=0)


def _sf_angle_load_resources(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
    counter = QROAMEmitter(counting=True)
    wires = dummy_wires(SFAngleLoad.required_wires(sf_angles, R, B, b_rot))
    _sf_body(counter, sf_angles, R, B, b_rot, **wires)
    return as_resources(counter.counts)


@qp.register_resources(_sf_angle_load_resources, exact=False)
def _sf_angle_load(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
    _sf_body(QROAMEmitter(), sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work)


register_rule(SFAngleLoad, _sf_angle_load)


# --------------------------------------------------------------------------- AngleErase


def _sf_fixup(R: int, B: int, onehot_bits: int) -> dict:
    """The SF fix-up's address model: the SF load's live set, enabled on ``G0 = 0``."""
    widths, covered = _sf_address(R, B)
    return {"address_widths": widths, "covered": covered, "onehot_bits": onehot_bits,
            "control_value": 0}


def _tail_fixup(N: int, R: int, C: int) -> dict:
    """The tail fix-up's address model: ``x_o`` over ``[RC, X_o)``, no one-hot, no enable."""
    return {"address_widths": (lg(N + R * C),), "covered": ((R * C, N + R * C),),
            "onehot_bits": 0, "control_value": None}


def check_erase_arguments(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                          sf_onehot_bits) -> tuple:
    """:class:`AngleErase`'s compilable arguments, in its order, as canonical tuples and ints,
    each checked against its domain (spec 9.0; ``ValueError`` otherwise). ``InnerSelect``, which
    passes its own to the erase, checks them here too, so both give the same messages."""
    N = check_int("N", N, low=2)
    C = check_int("C", C, low=1)
    sf_angles, R, B, b_rot = SFAngleLoad._check(sf_angles, R, B, b_rot)
    rows = canonical_ints("tail_angles", tail_angles)
    if not isinstance(rows, tuple) or len(rows) != N or len(sf_angles[0][0]) != N - 1:
        raise ValueError(f"tail_angles must have N = {N} rows and sf_angles N - 1 = {N - 1} "
                         "angles per entry")
    tail_angles, n_plus, _, _, _ = TailAngleLoad._check(rows, n_plus, R * C, N + R * C, b_rot)
    sf_onehot_bits = check_int("the SF one-hot bits", sf_onehot_bits, low=0, high=lg(B + 1))
    if sf_onehot_bits and 2**sf_onehot_bits > (N - 1) * b_rot + 1:
        raise ValueError(f"the SF one-hot over {sf_onehot_bits} bits needs {2**sf_onehot_bits} "
                         f"wires, more than the {(N - 1) * b_rot + 1} reset angle and b_neq_B "
                         "wires")
    return tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits


class AngleErase(Operator2):
    r"""Erase what the two angle loads wrote, by X measurement and two phase fix-ups (step 18).

    **Action.** Returns ``angles``, ``b_neq_B``, ``G0``, ``G1`` and ``work`` to
    :math:`|0\rangle` and leaves the state of ``x_o``, ``r`` and ``b`` exactly as it was before
    :class:`TailAngleLoad` and :class:`SFAngleLoad`, for every measurement outcome.

    **Registers.** With :math:`w_x = \lg X_o`, :math:`w_r = \lg R`, :math:`w_b = \lg(B+1)`:

    =========== ======================= ============== ==============
    name        width                   input          output
    =========== ======================= ============== ==============
    ``angles``  :math:`(N - 1) b_{rot}` the loads'     ``|0>``
    ``b_neq_B`` 1                       the SF load's  ``|0>``
    ``G0``      1                       the tail's     ``|0>``
    ``G1``      1                       the tail's     ``|0>``
    ``x_o``     :math:`w_x`             any            unchanged
    ``r``       :math:`w_r` (0 if R=1)  any            unchanged
    ``b``       :math:`w_b`             any            unchanged
    ``work``    :meth:`required_wires`  ``|0>``        ``|0>``
    =========== ======================= ============== ==============

    **Compilable arguments.** The loads' tables ``tail_angles`` (``N x (N - 1)``) and
    ``sf_angles`` (``R x B x (N - 1)``), ``n_plus``, ``N``, ``R``, ``B``, ``C`` (ints >= 1,
    ``N >= 2``), ``b_rot`` (int >= 2) and ``sf_onehot_bits`` (ob, ``0 <= ob <= w_b``; the walk
    uses :math:`w_b`, spec 9.17), whose one-hot register needs ``2**ob <= (N - 1) b_rot + 1``.

    **Pre- and postconditions.** ``angles``, ``b_neq_B``, ``G0`` and ``G1`` hold exactly what
    the two loads wrote for the current ``x_o``, ``r``, ``b`` (operations in between may act on
    them only in ways that preserve this; in ``InnerSelect`` they are controls). Afterwards
    they and ``work`` are :math:`|0\rangle`, with exact phases on every branch.

    **Decomposition** (spec 9.17; the order matters). With :math:`\xi_a` the outcomes of
    ``angles ‖ b_neq_B`` and :math:`\xi_g` those of ``G0 ‖ G1``:

    1. ``H`` and ``qp.measure(q, reset=True)`` on every qubit of ``angles ‖ b_neq_B``.
    2. SF fix-up: :func:`~dfthc.ops.qroam.emit_phase_fixup` over ``(r, b)`` in
       ``[0, R) x [0, B)``, enabled on ``G0 = 0``, with the parities
       :math:`f(r, b) = \xi_a \cdot (\mathrm{sf\_angles}[r][b] \Vert 1)` and a one-hot over the
       low ob bits of ``b`` on the reset ``angles ‖ b_neq_B`` wires.
    3. ``H`` and ``qp.measure(q, reset=True)`` on ``G0`` and ``G1``.
    4. Tail fix-up: ``emit_phase_fixup`` over ``x_o`` in ``[RC, X_o)``, no one-hot, no enable,
       with :math:`f(x_o) = \xi_a \cdot (\mathrm{tail\_angles}[x_o - RC] \Vert 0) \oplus
       \xi_g \cdot (1, [x_o - RC \ge N_+])`.

    The SF fix-up runs while ``G0`` still separates SF from tail branches: a tail branch's
    ``(r, b)`` may be anything, and its angle phase is paid by the tail fix-up. The parities
    come from :func:`dfthc.compile.mcm_parities` with masks built here from the compilable
    tables, through the loads' own entries; bit ``i`` of a big-endian mask belongs to the
    ``i``-th outcome from the end, so the outcomes are passed reversed. No mask is zero: every SF
    mask has the ``b_neq_B`` bit and every tail mask the ``G0`` bit. Every conditional is a
    hinted ``CZ`` (or ``Z`` for ob = 0, and on the tail) on a GF(2)-linear predicate (spec 12.5).

    **Adjoint and control.** Never: its rule measures (spec 7.4 rule 5).

    **Cost.** :func:`dfthc.costs.angle_erase` for ob = :math:`w_b`:
    :math:`T = U(0, R, w_r, 1) + O(w_b) + U(RC, X_o, w_x, 0)` (``A = T``),
    :math:`M = (N - 1) b_{rot} + 3`. For other ob, the SF term is
    ``costs.phase_fixup(sf_fixup_high_ranges(R, B, ob), ob, True)``. At FeMoco-54:
    :math:`T = (11 + 30) + 61 = 102`, :math:`M = 798`.

    **Paper and dfthcbe.** Step 18; the fix-ups of the paper's (B47) (the ``(b, r)`` table) and
    the :math:`N` term of (B50) (the D1/Q1 angles), PRX App. B, as dfthcbe's README attributes
    them. dfthcbe: ``RotDataErase`` (``dfthcbe/rot_data.py``, 90 at FeMoco-54), whose SF fix-up is
    not enabled (a ``Z`` on ``G0`` removes the tail branches' ``(0, 0)`` phase), whose iterations
    are truncated and pruned, and whose one-hot over ``b`` has ``B + 1`` positions where ours has
    ``2**w_b`` (FeMoco-54: 26 against 30).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.angle_data import AngleErase

        tail, sf = ((1,), (2,)), (((1,), (2,)), ((3,), (0,)))  # N = 2, R = B = 2, C = 1
        program = lambda: AngleErase(tail, sf, 1, 2, 2, 2, 1, 3, 2, angles=[0, 1, 2],
                                     b_neq_B=[3], G0=[4], G1=[5], x_o=[6, 7], r=[8], b=[9, 10],
                                     work=[11])
        compiled_resources(program, num_wires=12).measurements  # (N - 1) b_rot + 3 = 6
    """

    compilable_argnames = ("tail_angles", "sf_angles", "n_plus", "N", "R", "B", "C", "b_rot",
                           "sf_onehot_bits")
    wire_argnames = ("angles", "b_neq_B", "G0", "G1", "x_o", "r", "b", "work")

    def __init__(self, tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                 angles, b_neq_B, G0, G1, x_o, r, b, work):
        (tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
         sf_onehot_bits) = check_erase_arguments(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                                                 sf_onehot_bits)
        check_wires("AngleErase",
                    {"angles": angles, "b_neq_B": b_neq_B, "G0": G0, "G1": G1, "x_o": x_o,
                     "r": r, "b": b, "work": work},
                    self.required_wires(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                                        sf_onehot_bits))
        super().__init__(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                         angles, b_neq_B, G0, G1, x_o, r, b, work)

    @staticmethod
    def required_wires(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                       sf_onehot_bits) -> dict[str, int]:
        """Exact register sizes; ``work`` is the larger of the two fix-ups' needs. The tables and
        ``n_plus`` change no size: ``required_wires`` takes every compilable argument (spec 7.4
        rule 1)."""
        sf, tail = _sf_fixup(R, B, sf_onehot_bits), _tail_fixup(N, R, C)
        work = max(PhaseFixup.required_wires(**sf)["work"],
                   PhaseFixup.required_wires(**tail)["work"])
        return {"angles": (N - 1) * b_rot, "b_neq_B": 1, "G0": 1, "G1": 1,
                "x_o": lg(N + R * C), "r": lg(R), "b": lg(B + 1), "work": work}

    @staticmethod
    def sf_fixup_high_ranges(R, B, sf_onehot_bits) -> list[tuple[int, int, int]]:
        """The SF fix-up's iterated ranges: the ``high_ranges`` of
        :func:`dfthc.costs.phase_fixup` (controlled)."""
        sf = _sf_fixup(R, B, sf_onehot_bits)
        return PhaseFixup.high_ranges(sf["address_widths"], sf["covered"], sf_onehot_bits)

    @staticmethod
    def tail_fixup_high_ranges(N, R, C) -> list[tuple[int, int, int]]:
        """The tail fix-up's iterated range ``[(RC, X_o, w_x)]`` (uncontrolled, no one-hot)."""
        tail = _tail_fixup(N, R, C)
        return PhaseFixup.high_ranges(tail["address_widths"], tail["covered"], 0)


def _erase_body(emit, tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                angles, b_neq_B, G0, G1, x_o, r, b, work):
    data = _join(angles, b_neq_B)
    data_outcomes = [emit.measure_x(qubit) for qubit in data]
    sf, w_b = _sf_fixup(R, B, sf_onehot_bits), lg(B + 1)
    entries = _sf_entries(sf_angles, b_rot)
    masks = [entries[address >> w_b][address & (2**w_b - 1)]  # angles ‖ b_neq_B = 1
             for address in PhaseFixup.address_order(sf["address_widths"], sf["covered"])]
    sf_work = PhaseFixup.required_wires(**sf)["work"]
    emit_phase_fixup(emit.parities(data_outcomes[::-1], masks), address=_join(r, b),
                     address_widths=sf["address_widths"], covered=sf["covered"],
                     onehot=data[:2**sf_onehot_bits] if sf_onehot_bits else [],
                     onehot_bits=sf_onehot_bits, work=list(work)[:sf_work], control=G0[0],
                     control_value=0, emit=emit)
    flag_outcomes = [emit.measure_x(G0[0]), emit.measure_x(G1[0])]
    tail = _tail_fixup(N, R, C)
    masks = [((entry >> 2) << 3) | (entry & 3)  # angles ‖ b_neq_B = 0 ‖ G0 = 1 ‖ G1
             for entry in _tail_entries(tail_angles, n_plus, b_rot)]
    tail_work = PhaseFixup.required_wires(**tail)["work"]
    emit_phase_fixup(emit.parities((data_outcomes + flag_outcomes)[::-1], masks),
                     address=x_o, address_widths=tail["address_widths"],
                     covered=tail["covered"], onehot=[], onehot_bits=0,
                     work=list(work)[:tail_work], emit=emit)


def _angle_erase_resources(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                           angles, b_neq_B, G0, G1, x_o, r, b, work):
    counter = QROAMEmitter(counting=True)
    wires = dummy_wires(AngleErase.required_wires(tail_angles, sf_angles, n_plus, N, R, B, C,
                                                  b_rot, sf_onehot_bits))
    _erase_body(counter, tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                **wires)
    return as_resources(counter.counts)


@qp.register_resources(_angle_erase_resources)
def _angle_erase(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits, angles,
                 b_neq_B, G0, G1, x_o, r, b, work):
    _erase_body(QROAMEmitter(), tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                sf_onehot_bits, angles, b_neq_B, G0, G1, x_o, r, b, work)


register_rule(AngleErase, _angle_erase)
