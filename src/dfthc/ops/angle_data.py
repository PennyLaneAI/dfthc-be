"""The angle registers of the inner select: loads and measured erase (spec 9.10, 9.11, 9.17).

**The shared angle register** (spec 6.7). ``angles`` is the concatenation of ``N - 1`` registers
of ``b_rot`` wires; register ``k`` drives rung ``k`` of the Givens ladder and holds the angle
integer ``x_k`` big-endian, with ``x_k < 2**(b_rot - 1)`` (spec 6.3). :func:`angle_entry` is the
big-endian integer of the whole register, register 0 most significant.

**Two exclusive loads.** :class:`TailAngleLoad` writes the one-body rows ``x_o`` in
``[RC, X_o)``: their angles, ``G0 = 1`` and ``G1 = [x_o - RC >= N+]``. :class:`SFAngleLoad` then
writes, where ``G0 = 0``, the angles of ``(r, b)`` in ``[0, R) x [0, B)`` and ``b_neq_B = 1``.
Each is a :class:`~dfthc.ops.qroam.QROAMRead` without a swap network (``block_bits = 0``) whose
entry is one big-endian integer over its target (``angles ‖ G0 ‖ G1``, ``angles ‖ b_neq_B``) and
whose ``leaf_words`` the rule builds from the compilable angle table (spec 9.3). After both
loads every branch holds one family's data: a tail row's on ``G0 = 1``, an SF entry's on
``G0 = 0`` where ``(r, b)`` is in range, and zeros elsewhere (failed ``x_o``, ``b = B``,
failed ``b``). :class:`AngleErase` (Task 2E.2) erases what they write.
"""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import AbstractArray, Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.qroam import WORD, WORD_DTYPE, QROAMRead
from dfthc.params import lg


def angle_entry(angles, b_rot: int) -> int:
    """The big-endian integer of the angle registers holding ``angles``: register ``k`` holds
    ``angles[k]``, register 0 most significant (spec 6.7)."""
    value = 0
    for angle in angles:
        value = (value << b_rot) | angle
    return value


def _words(value: int, width: int) -> tuple[int, ...]:
    """``value`` as ``ceil(width / 32)`` words, least significant first (L-24)."""
    return tuple((value >> (WORD * w)) & (2**WORD - 1) for w in range(-(-width // WORD)))


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


def _read_key(read: dict, control_value):
    """The resource key of the loads' ``QROAMRead`` for the read shape ``read``."""
    words = AbstractArray(QROAMRead.leaf_words_shape(**read), WORD_DTYPE)
    sizes = QROAMRead.required_wires(**read, control_value=control_value)
    return resource_key(lambda: QROAMRead(words, **read, control_value=control_value,
                                          **{name: Wire[n] for name, n in sizes.items()}))


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
    ``work`` ends in ``|0>``. :class:`AngleErase` erases ``angles``, ``G0`` and ``G1``.

    **Decomposition.** One :class:`~dfthc.ops.qroam.QROAMRead` of :meth:`table` with address
    ``x_o`` over ``[RC, X_o)``, no swap network (``block_bits = 0``), no enable, and target
    ``angles ‖ G0 ‖ G1`` (:meth:`read_shape`). The rule builds the read's ``leaf_words`` from
    the compilable ``tail_angles`` with ``QROAMRead.leaf_words_from_table`` (spec 9.3).

    **Adjoint and control.** Never adjointed or controlled (its read has loops; spec 7.4 rules
    4 and 6). :class:`AngleErase` undoes it by measurement.

    **Cost.** :func:`dfthc.costs.tail_angle_load`: :math:`T = U(RC, X_o, w_x, 0)` (``A = T``),
    ``M = 0``. The read's CNOT count is an upper bound (``exact=False`` in ``QROAMRead``). At
    FeMoco-54: 61, with 54 leaves of 25 words each.

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
    def read_shape(N: int, RC: int, X_o: int, b_rot: int) -> dict:
        """The read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``."""
        return {"address_widths": (lg(X_o),), "live_ranges": ((RC, X_o),),
                "width": (N - 1) * b_rot + 2, "block_bits": 0}

    @staticmethod
    def table(tail_angles, n_plus: int, RC: int, b_rot: int) -> tuple:
        """The read's table: ``(RC + j, words)`` with the entry ``angles ‖ G0 ‖ G1`` of row ``j``
        in 32-bit words, least significant first."""
        width = (len(tail_angles) - 1) * b_rot + 2
        return tuple((RC + j, _words(entry, width))
                     for j, entry in enumerate(_tail_entries(tail_angles, n_plus, b_rot)))

    @staticmethod
    def required_wires(tail_angles, n_plus, RC, X_o, b_rot) -> dict[str, int]:
        """Exact register sizes."""
        read = TailAngleLoad.read_shape(len(tail_angles), RC, X_o, b_rot)
        return {"x_o": lg(X_o), "angles": (len(tail_angles) - 1) * b_rot, "G0": 1, "G1": 1,
                "work": QROAMRead.required_wires(**read, control_value=None)["work"]}


def _tail_angle_load_resources(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
    return {_read_key(TailAngleLoad.read_shape(len(tail_angles), RC, X_o, b_rot), None): 1}


@qp.register_resources(_tail_angle_load_resources)
def _tail_angle_load(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
    read = TailAngleLoad.read_shape(len(tail_angles), RC, X_o, b_rot)
    words = QROAMRead.leaf_words_from_table(TailAngleLoad.table(tail_angles, n_plus, RC, b_rot),
                                            **read)
    QROAMRead(words, **read, control_value=None, address=x_o, target=_join(angles, G0, G1),
              junk=[], work=work, control=[])


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

    **Decomposition.** One :class:`~dfthc.ops.qroam.QROAMRead` of :meth:`table` with address
    ``r ‖ b`` (``r`` outer over ``[0, R)``, ``b`` inner over ``[0, B)``; ``b`` alone if
    ``R = 1``), no swap network, the enable ``G0`` with control value 0, and target
    ``angles ‖ b_neq_B`` (:meth:`read_shape`). The rule builds the read's ``leaf_words`` from
    the compilable ``sf_angles`` (spec 9.3).

    **Adjoint and control.** Never adjointed or controlled (spec 7.4 rules 4 and 6).

    **Cost.** :func:`dfthc.costs.sf_angle_load`:
    :math:`T = U(0, R, w_r, 1) + R\,U(0, B, w_b, 1)` (``A = T``), ``M = 0``; the read's CNOT
    count is an upper bound. At FeMoco-54: :math:`11 + 10 \cdot 28 = 291`, with 270 leaves of
    25 words each.

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
    def read_shape(N: int, R: int, B: int, b_rot: int) -> dict:
        """The read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``."""
        widths, ranges = _sf_address(R, B)
        return {"address_widths": widths, "live_ranges": ranges, "width": (N - 1) * b_rot + 1,
                "block_bits": 0}

    @staticmethod
    def table(sf_angles, B: int, b_rot: int) -> tuple:
        """The read's table: ``((r << w_b) | b, words)`` with the entry ``angles ‖ b_neq_B`` of
        ``(r, b)`` in 32-bit words, least significant first."""
        width = len(sf_angles[0][0]) * b_rot + 1
        return tuple(((r << lg(B + 1)) | b, _words(entry, width))
                     for r, plane in enumerate(_sf_entries(sf_angles, b_rot))
                     for b, entry in enumerate(plane))

    @staticmethod
    def required_wires(sf_angles, R, B, b_rot) -> dict[str, int]:
        """Exact register sizes."""
        num_modes = len(sf_angles[0][0]) + 1
        read = SFAngleLoad.read_shape(num_modes, R, B, b_rot)
        return {"r": lg(R), "b": lg(B + 1), "G0": 1, "angles": (num_modes - 1) * b_rot,
                "b_neq_B": 1, "work": QROAMRead.required_wires(**read, control_value=0)["work"]}


def _sf_angle_load_resources(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
    return {_read_key(SFAngleLoad.read_shape(len(sf_angles[0][0]) + 1, R, B, b_rot), 0): 1}


@qp.register_resources(_sf_angle_load_resources)
def _sf_angle_load(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
    read = SFAngleLoad.read_shape(len(sf_angles[0][0]) + 1, R, B, b_rot)
    words = QROAMRead.leaf_words_from_table(SFAngleLoad.table(sf_angles, B, b_rot), **read)
    QROAMRead(words, **read, control_value=0, address=_join(r, b),
              target=_join(angles, b_neq_B), junk=[], work=work, control=G0)


register_rule(SFAngleLoad, _sf_angle_load)
