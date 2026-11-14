r"""The block encoding of the sum-of-squares Hamiltonian and its controlled class (spec 9.21;
steps 1 to 27).

**The circuit** (spec 3.3 and 3.5). The block encoding is :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U`
with :math:`U` = ``OuterPrepare``, ``InnerPrepare``, ``InnerSigns``, ``InnerSelect``,
``InnerUnprepare`` in time order and :math:`U^\dagger` = ``InnerPrepare``, ``InnerSelectDagger``,
``InnerSigns``, ``InnerUnprepare``, ``OuterUnprepare``; its block on the ancillas in
:math:`|0\rangle` is :math:`\tilde H_{SA}/\tilde\Lambda - I` (spec 8.5). Every adjoint in it is an
explicit class (``InnerUnprepare``, ``InnerSelectDagger``, ``OuterUnprepare``; spec 7.4 rule 6).
The controlled class controls only the reflection: on states whose ancillas start in
:math:`|0\rangle`, :math:`U^\dagger U = I` (spec 3.3). There is no ``C(...)`` rule of either
operator.

**Wires.** Each class takes the groups of :class:`~dfthc.registers.WalkLayout` (``outer``,
``inner``, ``select``, ``system``, ``phase_grad``, ``work``; a controlled class ``control``
first). Its rule splits them once with :meth:`WalkLayout.split_group`, unpacks every register into
a list once, and passes each child the registers its ``wire_argnames`` name, with the prefix of the
``work`` group of exactly the child's need (spec 9.0, wire passing). ``work`` is the walk's shared
pool: every child returns it in :math:`|0\rangle`; the comparators' dirty work registers belong to
the ``outer`` and ``inner`` groups, never to ``work`` (spec 10.2).

**Arguments.** ``params`` is :meth:`WalkParams.as_compilable` and ``tables`` is
:meth:`WalkTables.as_compilable`; the rules rebuild both with ``from_compilable``. One list,
:func:`_children`, gives the rule its children in time order with their compilable arguments and
work needs, and the resource function its declaration, so the two cannot disagree.
"""

from __future__ import annotations

from collections import Counter

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.data.tables import WalkTables
from dfthc.ops._validate import check_wires
from dfthc.ops.prepare import (
    InnerPrepare,
    InnerSigns,
    InnerUnprepare,
    OuterPrepare,
    OuterUnprepare,
    _check_inner,
    _check_outer,
)
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefT2
from dfthc.ops.select import InnerSelect, InnerSelectDagger
from dfthc.ops.select import _check as _check_select
from dfthc.params import WalkParams
from dfthc.registers import GROUPS, WalkLayout

_GROUPS = ("outer", "inner", "select", "system", "phase_grad", "work")
"""The wire arguments of the assembly operators, after ``control`` in the controlled ones."""


def _rebuild(params, tables) -> tuple[WalkParams, WalkTables]:
    return WalkParams.from_compilable(params), WalkTables.from_compilable(tables)


def _children(p: WalkParams, t: WalkTables, controlled: bool) -> list[tuple[type, tuple, int]]:
    """The block encoding's children in time order (spec 9.21), each ``(class, compilable
    arguments, work wires)``; the reflection is ``ControlledRefT2`` if ``controlled``."""
    outer = (p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    inner = (p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn, p.b_k2,
             p.k2, p.ke2, p.s_u)
    select = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    reflection = ControlledRefT2 if controlled else RefT2
    outer_work = OuterPrepare.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)["work"]
    inner_work = InnerPrepare.required_wires(p.N, p.R, p.B, p.C, p.b_k2, p.k2, p.ke2,
                                             p.s_u)["work"]
    select_work = InnerSelect.required_wires(*select)["work"]
    reflection_work = reflection.required_wires(p.w_b, p.b_k2)["work"]
    return [
        (OuterPrepare, outer, outer_work),
        (InnerPrepare, inner, inner_work),
        (InnerSigns, (), 0),
        (InnerSelect, select, select_work),
        (InnerUnprepare, inner, inner_work),
        (reflection, (), reflection_work),
        (InnerPrepare, inner, inner_work),
        (InnerSelectDagger, select, select_work),
        (InnerSigns, (), 0),
        (InnerUnprepare, inner, inner_work),
        (OuterUnprepare, outer, outer_work),
    ]


def _group_sizes(p: WalkParams) -> dict[str, int]:
    """The width of every group: the layout's (spec 10.1)."""
    sizes = WalkLayout.register_sizes(p)
    return {group: sum(sizes[name] for name in GROUPS[group]) for group in _GROUPS}


def _sizes(p: WalkParams) -> dict[str, int]:
    """The width of every register and group a child of an assembly rule can take."""
    return WalkLayout.register_sizes(p) | _group_sizes(p) | {"control": 1}


def _check(params, tables) -> tuple[WalkParams, WalkTables]:
    """``params`` and ``tables`` rebuilt and checked against the children's domains (spec 9.0):
    the tables have the shapes of ``params`` and lie in the children's ranges, the erases'
    one-hot registers fit (``2**ke1 <= K1 d1``, ``2**(ke2 + w_b) <= K2 d2`` and
    ``2**w_b <= (N - 1) b_rot + 1``), and the layout's ``work`` pool covers every child of the walk
    and the controlled walk. ``ValueError`` otherwise."""
    if not isinstance(params, (tuple, list)) or not isinstance(tables, (tuple, list)):
        raise ValueError("params and tables must be WalkParams.as_compilable() and "
                         "WalkTables.as_compilable()")
    p, t = _rebuild(params, tables)
    _check_outer(p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    _check_inner(p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn,
                 p.b_k2, p.k2, p.ke2, p.s_u)
    _check_select(t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    needs = {cls.__name__: work for cls, _, work in _children(p, t, controlled=True)}
    needs["ControlledRefAB"] = ControlledRefAB.required_wires(p.w_x, p.b_k1, p.w_b,
                                                              p.b_k2)["work"]
    pool = WalkLayout.work_size(p)
    over = {name: need for name, need in needs.items() if need > pool}
    if over:
        raise ValueError(f"the work pool of {pool} wires (WalkLayout.work_size) is too small for "
                         f"{over}")
    return p, t


def _validate(name: str, wire_argnames: tuple, params, tables, wires: tuple) -> tuple:
    """The canonical compilable arguments, after checking them and the wire groups."""
    p, t = _check(params, tables)
    required = ({"control": 1} if "control" in wire_argnames else {}) | _group_sizes(p)
    check_wires(name, dict(zip(wire_argnames, wires, strict=True)), required)
    return p.as_compilable(), t.as_compilable()


def _registers(p: WalkParams, outer, inner, select, system, phase_grad, control=()) -> dict:
    """Every group and every register by name, each unpacked into a list once (spec 9.0)."""
    registers = {"outer": list(outer), "inner": list(inner), "select": list(select),
                 "system": list(system), "phase_grad": list(phase_grad),
                 "control": list(control)}
    for group in ("outer", "inner", "select", "system"):
        registers |= WalkLayout.split_group(p, group, registers[group])
    return registers


def _emit(children: list, registers: dict, work) -> None:
    """Apply ``children`` in order, each on the registers its ``wire_argnames`` name and on the
    prefix of ``work`` of exactly its need."""
    work = list(work)
    for cls, args, need in children:
        cls(*args, **{name: work[:need] if name == "work" else registers[name]
                      for name in cls.wire_argnames})


def _resources(children: list, p: WalkParams) -> dict:
    """The declaration of ``children``: each distinct child once, with its count, its compilable
    arguments and exactly the register sizes :func:`_emit` passes (spec 7.4 rule 2)."""
    sizes = _sizes(p)

    def key(cls, args, need):
        return resource_key(lambda: cls(*args, **{
            name: Wire[need if name == "work" else sizes[name]] for name in cls.wire_argnames}))

    return {key(*child): count for child, count in Counter(children).items()}


class DFTHCBlockEncoding(Operator2):
    r"""The DFTHC block encoding :math:`\mathrm{BE} = U^\dagger\,\mathrm{REF}_{T_2}\,U` (steps 1
    to 27): :math:`\langle 0|\mathrm{BE}|0\rangle = \tilde H_{SA}/\tilde\Lambda - I`.

    **Action.** With every group but ``system`` and ``phase_grad`` in :math:`|0\rangle`, the part
    of the output with every ancilla back in :math:`|0\rangle` is
    :math:`(\tilde H_{SA}/\tilde\Lambda - I)|\psi\rangle` for the system input :math:`|\psi\rangle`,
    with :math:`(\tilde H_{SA}, \tilde\Lambda)` = ``encoded_block(params, tables, factors)``
    (spec 8.5): the alias probabilities of the tables, the sign columns, the directions of the
    rounded angles, the identity term ``b = B``, the failed uniform preparation of ``b`` (acting as
    :math:`+I`) and that of ``x_o`` (where :math:`\mathrm{REF}_{T_2}` is :math:`-I`, which gives
    :math:`\tilde\Lambda = \Lambda/(1 - P_{fail}(x_o))`). :math:`\mathrm{BE}` is Hermitian and
    self-inverse. The rest of the output has a nonzero index register of
    :math:`\mathcal{A}\cup\mathcal{B}`. The output is the same for every outcome of the erase
    measurements.

    **Registers.** The groups of :class:`~dfthc.registers.WalkLayout` (spec 10.1), each a
    concatenation of named registers in layout order.

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``outer``      the ``outer`` group (143)      ``|0>``               see the action
    ``inner``      the ``inner`` group (455)      ``|0>``               see the action
    ``select``     ``(N - 1) b_rot + 4`` (799)    ``|0>``               ``|0>``
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** ``params``: :meth:`WalkParams.as_compilable`, the 13 fields; the
    erases' one-hot bits must fit their reset registers (:math:`2^{k_{e1}} \le K_1 d_1`,
    :math:`2^{k_{e2} + w_b} \le K_2 d_2`, :math:`2^{w_b} \le (N - 1) b_{rot} + 1`) and
    ``WalkLayout.work_size`` must cover every child. ``tables``:
    :meth:`WalkTables.as_compilable` of tables with the shapes of ``params``, in the ranges the
    prepares and the select accept. The constructor checks both (``ValueError``).

    **Pre- and postconditions.** The groups are pairwise disjoint (unchecked). ``phase_grad``
    holds the state ``PhaseGradientStatePrep`` prepares and keeps it. The block needs the ancilla
    groups in ``|0>``; in the walk the input is also any state the walk reaches from there, on
    which every erase's precondition holds (spec 3.3). Afterwards every register but the six
    index registers of :math:`\mathcal{A}\cup\mathcal{B}` (``x_o``, ``m_x_o``, ``sigma_1``,
    ``b``, ``m_b``, ``sigma_0``) is back in ``|0>``: the erases reset the loaded data, the
    measured uncomputes clean the comparators, and the adjoint uniform preparations clear the
    success flags. So ``RefAB``'s reflection about those six is the walk's reflection about every
    ancilla.

    **Decomposition** (spec 9.21). The groups are split with ``WalkLayout.split_group``; each
    child gets its registers by name and the prefix of ``work`` of its need: ``OuterPrepare``;
    ``InnerPrepare``, ``InnerSigns``, ``InnerSelect``, ``InnerUnprepare``; ``RefT2``;
    ``InnerPrepare``, ``InnerSelectDagger``, ``InnerSigns``, ``InnerUnprepare``;
    ``OuterUnprepare``. The prepare classes take the whole ``phase_grad`` (they use its top
    ``s_u`` wires), the selects exactly its ``b_rot`` wires; both selects get
    ``fixup_bits = w_b``. ``InnerSigns`` is diagonal and touches no register of the selects, so
    its place on either side of them is exact; this is spec 9.21's.

    **Adjoint and control.** Never adjointed: its rule measures (spec 7.4 rule 5), and the
    operator is its own inverse anyway. Never controlled: the controlled form is the class
    :class:`ControlledDFTHCBlockEncoding`, which controls only ``RefT2``; there is no
    ``C(DFTHCBlockEncoding)`` rule, so ``qp.ctrl`` of this operator does not compile (spec 7.4
    rule 6).

    **Cost.** :func:`dfthc.costs.block_encoding` (uncontrolled):
    :math:`T = T_{out} + T_{out}^\dagger + 2(T_{in} + T_{sel} + T_{in}^\dagger) + T_{RefT2}`,
    :math:`M = K_1 d_1 + 2(K_2 d_2 + (N - 1) b_{rot} + 3)`; ``A = T - (T_Toffoli - 1)`` with
    ``T_Toffoli`` = :func:`dfthc.costs.toffoli_only` (the walk's plain Toffolis less ``RefAB``'s
    one). At FeMoco-54: :math:`279 + 2 \cdot 4725 + 20 = 9749`, ``M = 2524``, ``A = 8632``.
    ``CNOT`` and ``PauliX`` counts are upper bounds (spec 7.6).

    **Paper and dfthcbe.** Steps 1 to 27 of spec 3.5: the sum-of-squares block encoding
    :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U` (PRX 15, 041016; arXiv:2502.15882v1 Eq. (11) and
    Fig. 1), built in App. B: the prepare over :math:`x_o` (B.1, Eq. (87)) and the step of
    oblivious amplitude amplification, whose reflection is controlled on the success of
    :math:`x_o` (B.2, Eq. (88)); the costs are its Table 3 (PRX Table III). dfthcbe:
    ``DFTHCBlockEncoding`` (``dfthcbe/block_encoding.py``), 9,596 Toffolis at FeMoco-54 by its
    call graph, :math:`212 + 2 (913 + 220) + 2 \cdot 3502 + 20 + 94`; ours is +153 (the inner
    prepare +58 and select +35 per round, the inner unprepare -3 per round, the outer prepare -2
    and unprepare -25; spec 13.1). The paper gives 9,997 per walk, 46 of them for
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}`; against the other 9,951 ours is -202: the
    Givens ladders -424 (``rz-addsub``), the 26-bit inner entries against its 20-bit ones +186,
    exact boundaries and one-hot padding +80 (``exact-iter``, ``no-prune``, ``onehot-pad``), the
    free comparator adjoints -46 (``cmp-free``) and the rest +2 (spec 13.1).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import DFTHCBlockEncoding
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p)
        groups = {name: getattr(layout, name) for name in DFTHCBlockEncoding.wire_argnames}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            DFTHCBlockEncoding(p.as_compilable(), t.as_compilable(), **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 150
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = _GROUPS

    def __init__(self, params, tables, outer, inner, select, system, phase_grad, work):
        wires = (outer, inner, select, system, phase_grad, work)
        compilable = _validate("DFTHCBlockEncoding", self.wire_argnames, params, tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """The layout's group sizes (spec 10.1); ``tables`` does not change them."""
        return _group_sizes(WalkParams.from_compilable(params))


def _block_encoding_resources(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_children(p, t, controlled=False), p)


@qp.register_resources(_block_encoding_resources)
def _block_encoding(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    _emit(_children(p, t, controlled=False),
          _registers(p, outer, inner, select, system, phase_grad), work)


register_rule(DFTHCBlockEncoding, _block_encoding)


class ControlledDFTHCBlockEncoding(Operator2):
    r"""The controlled block encoding: the explicit controlled class of
    :class:`DFTHCBlockEncoding` (spec 7.4 rule 6), :math:`U^\dagger\,C(\mathrm{REF}_{T_2})\,U`.

    **Action.** :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle 1|_{control}
    \otimes \mathrm{BE}` on inputs whose ancilla groups are in :math:`|0\rangle`, with
    :math:`\mathrm{BE}` the action of :class:`DFTHCBlockEncoding` for the same arguments: control
    ``|0>`` returns the whole input (:math:`U^\dagger U = I`), control ``|1>`` gives
    :class:`DFTHCBlockEncoding`'s whole output, global sign included. The output is the same for
    every outcome of the erase measurements.

    **Registers.** Those of :class:`DFTHCBlockEncoding` after a one-wire ``control``, which
    stays a control (only phases on it).

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``control``    1                              any                   a control
    ``outer``      the ``outer`` group (143)      ``|0>``               see the action
    ``inner``      the ``inner`` group (455)      ``|0>``               see the action
    ``select``     ``(N - 1) b_rot + 4`` (799)    ``|0>``               ``|0>``
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCBlockEncoding`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCBlockEncoding`, on both control
    branches; ``control`` is disjoint from the groups (unchecked).

    **Decomposition** (spec 9.21). :class:`DFTHCBlockEncoding`'s eleven children with
    ``ControlledRefT2(control, succ_x_o, b, m_b, sigma_0, work)`` in place of ``RefT2``: every
    other child is uncontrolled, the same operator with the same compilable arguments and
    register sizes as in :class:`DFTHCBlockEncoding`, so the measured unprepares are never
    controlled (spec 7.4 rule 5).

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0).

    **Cost.** :func:`dfthc.costs.block_encoding` (controlled): :class:`DFTHCBlockEncoding`'s
    ``T + 1`` (the controlled reflection's extra ``TemporaryAND``), the same ``M``,
    ``A = T - (T_Toffoli - 1)``. At FeMoco-54: ``T = 9750``, ``M = 2524``, ``A = 8633``.

    **Paper and dfthcbe.** The controlled block encoding inside the controlled walk
    :math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,U^\dagger\,
    C(\mathrm{REF}_{T_2})\,U` (spec 3.3). dfthcbe: ``DFTHCBlockEncoding(is_controlled=True)``,
    9,597 Toffolis at FeMoco-54, which also controls only ``ref_t2``.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import ControlledDFTHCBlockEncoding
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p, qpe_bits=1)
        groups = {name: getattr(layout, name)
                  for name in ControlledDFTHCBlockEncoding.wire_argnames[1:]}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            ControlledDFTHCBlockEncoding(p.as_compilable(), t.as_compilable(),
                                         layout.qpe_control, **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 150 + 1 = 151
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = ("control", *_GROUPS)

    def __init__(self, params, tables, control, outer, inner, select, system, phase_grad, work):
        wires = (control, outer, inner, select, system, phase_grad, work)
        compilable = _validate("ControlledDFTHCBlockEncoding", self.wire_argnames, params,
                               tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """``control`` (1 wire), then the layout's group sizes."""
        return {"control": 1} | _group_sizes(WalkParams.from_compilable(params))


def _controlled_block_encoding_resources(params, tables, control, outer, inner, select, system,
                                         phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_children(p, t, controlled=True), p)


@qp.register_resources(_controlled_block_encoding_resources)
def _controlled_block_encoding(params, tables, control, outer, inner, select, system, phase_grad,
                               work):
    p, t = _rebuild(params, tables)
    _emit(_children(p, t, controlled=True),
          _registers(p, outer, inner, select, system, phase_grad, control), work)


register_rule(ControlledDFTHCBlockEncoding, _controlled_block_encoding)
