r"""The block encoding of the sum-of-squares Hamiltonian, the qubitized walk and their controlled
classes (spec 9.21 and 9.22; steps 1 to 28).

**The circuit** (spec 3.3 and 3.5). The block encoding is :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U`
with :math:`U` = ``OuterPrepare``, ``InnerPrepare``, ``InnerSigns``, ``InnerSelect``,
``InnerUnprepareFromOutcomes`` in time order and :math:`U^\dagger` = ``InnerPrepare``,
``InnerSelectDagger``, ``InnerSigns``, ``InnerUnprepareFromOutcomes``,
``OuterUnprepareFromOutcomes``; its block on the ancillas in :math:`|0\rangle` is
:math:`\tilde H_{SA}/\tilde\Lambda - I` (spec 8.5). Every adjoint in it is an explicit class
(``InnerUnprepareFromOutcomes``, ``InnerSelectDagger``, ``OuterUnprepareFromOutcomes``; spec 7.4
rule 6). The walk is :math:`W = \mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}\,\mathrm{BE}`: the block
encoding, then ``RefAB``. The controlled classes control only the reflections: on states whose
ancillas start in :math:`|0\rangle`, :math:`U^\dagger U = I`, so
:math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,U^\dagger\,C(\mathrm{REF}_{T_2})\,U`
(spec 3.3). There is no ``C(...)`` rule of any of these operators.

**Early junk measurement** (spec 16, Q-1 variant (b)). Right after each prepare the block
encoding's rule X-measures the prepare's junk register (``H`` and ``qp.measure(q, reset=True)``
on every qubit) and later passes the outcomes to the matching unprepare from outcomes, which
measures only the read's target and fixes the phases from both outcome vectors. The junk wires
are free from their measurement on, so the layout gives them to the angle registers
(:data:`~dfthc.registers.JUNK`): the angles are loaded only after the junk is measured, and the
unprepares borrow their ``extra_onehot`` wires from the start of the same, by then reset, wires.
T and A are those of the baseline unprepares, and the total M is unchanged: the junk's share
moves into this rule (:func:`dfthc.costs.junk_measurements`).

**Wires.** Each class takes the groups of :class:`~dfthc.registers.WalkLayout` (``outer``,
``inner``, ``select``, ``system``, ``phase_grad``, ``work``; a controlled class ``control``
first). Its rule splits them once with :meth:`WalkLayout.split_group` and
:meth:`WalkLayout.split_junk`, unpacks every register into a list once, and passes each child the
registers its ``wire_argnames`` name, with the prefix of the ``work`` group of exactly the child's
need (spec 9.0, wire passing). ``work`` is the walk's shared pool: every child returns it in
:math:`|0\rangle`; the comparators' dirty work registers belong to the ``outer`` and ``inner``
groups, never to ``work`` (spec 10.2).

**Arguments.** ``params`` is :meth:`WalkParams.as_compilable` and ``tables`` is
:meth:`WalkTables.as_compilable`; the rules rebuild both with ``from_compilable``. One list,
:func:`_children`, gives the rule its steps in time order (the children with their compilable
arguments, work needs and lent one-hot wires, and the junk measurements), and the resource
function its declaration, so the two cannot disagree; :func:`_walk_children` does the same for
the walks.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.data.tables import WalkTables
from dfthc.ops._validate import check_wires
from dfthc.ops.prepare import (
    InnerPrepare,
    InnerSigns,
    InnerUnprepareFromOutcomes,
    OuterPrepare,
    OuterUnprepareFromOutcomes,
    _check_inner,
    _check_outer,
    abstract_junk_outcomes,
)
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefAB, RefT2
from dfthc.ops.select import InnerSelect, InnerSelectDagger
from dfthc.ops.select import _check as _check_select
from dfthc.params import WalkParams
from dfthc.registers import GROUPS, WalkLayout

_GROUPS = ("outer", "inner", "select", "system", "phase_grad", "work")
"""The wire arguments of the assembly operators, after ``control`` in the controlled ones."""

JUNK_MEASUREMENT = "junk measurement"
"""The step of :func:`_children` that X-measures the junk register its ``args`` names."""

FROM_OUTCOMES = {OuterUnprepareFromOutcomes: "junk_x_o", InnerUnprepareFromOutcomes: "junk_b"}
"""Each unprepare from outcomes and the junk register whose outcomes it takes and whose first
wires it borrows as ``extra_onehot``."""


class Step(NamedTuple):
    """One step of an assembly rule: a child with its compilable arguments, the prefix of the
    ``work`` pool it takes and its ``extra_onehot`` wires (for an unprepare from outcomes), or,
    with ``op`` = :data:`JUNK_MEASUREMENT`, the X measurement of the junk register ``args``."""

    op: type | str
    args: tuple | str
    work: int
    extra_onehot: int = 0


def _rebuild(params, tables) -> tuple[WalkParams, WalkTables]:
    return WalkParams.from_compilable(params), WalkTables.from_compilable(tables)


def _children(p: WalkParams, t: WalkTables, controlled: bool) -> list[Step]:
    """The block encoding's steps in time order (spec 9.21; spec 16, Q-1 variant (b)): the
    children, each junk register's X measurement right after its prepare, and the unprepares from
    those outcomes; the reflection is ``ControlledRefT2`` if ``controlled``."""
    outer = (p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    inner = (p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn, p.b_k2,
             p.k2, p.ke2, p.s_u)
    select = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    reflection = ControlledRefT2 if controlled else RefT2
    outer_sizes = OuterUnprepareFromOutcomes.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)
    inner_sizes = InnerUnprepareFromOutcomes.required_wires(p.N, p.R, p.B, p.C, p.b_k2, p.k2,
                                                            p.ke2, p.s_u)
    outer_work, inner_work = outer_sizes["work"], inner_sizes["work"]
    select_work = InnerSelect.required_wires(*select)["work"]
    reflection_work = reflection.required_wires(p.w_b, p.b_k2)["work"]
    return [
        Step(OuterPrepare, outer, outer_work),
        Step(JUNK_MEASUREMENT, "junk_x_o", 0),
        Step(InnerPrepare, inner, inner_work),
        Step(JUNK_MEASUREMENT, "junk_b", 0),
        Step(InnerSigns, (), 0),
        Step(InnerSelect, select, select_work),
        Step(InnerUnprepareFromOutcomes, inner, inner_work, inner_sizes["extra_onehot"]),
        Step(reflection, (), reflection_work),
        Step(InnerPrepare, inner, inner_work),
        Step(JUNK_MEASUREMENT, "junk_b", 0),
        Step(InnerSelectDagger, select, select_work),
        Step(InnerSigns, (), 0),
        Step(InnerUnprepareFromOutcomes, inner, inner_work, inner_sizes["extra_onehot"]),
        Step(OuterUnprepareFromOutcomes, outer, outer_work, outer_sizes["extra_onehot"]),
    ]


def _group_sizes(p: WalkParams) -> dict[str, int]:
    """The width of every group: the layout's (spec 10.1)."""
    sizes = WalkLayout.register_sizes(p)
    return {group: sum(sizes[name] for name in GROUPS[group]) for group in _GROUPS}


def _sizes(p: WalkParams) -> dict[str, int]:
    """The width of every register, junk register and group a step of an assembly rule can
    take."""
    return (WalkLayout.register_sizes(p) | WalkLayout.junk_sizes(p) | _group_sizes(p)
            | {"control": 1})


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
    needs = {step.op.__name__: step.work for step in _children(p, t, controlled=True)
             if step.op != JUNK_MEASUREMENT}
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
    """Every group, every register and both junk registers (aliases of the start of ``select``)
    by name, each unpacked into a list once (spec 9.0)."""
    registers = {"outer": list(outer), "inner": list(inner), "select": list(select),
                 "system": list(system), "phase_grad": list(phase_grad),
                 "control": list(control)}
    for group in ("outer", "inner", "select", "system"):
        registers |= WalkLayout.split_group(p, group, registers[group])
    return registers | WalkLayout.split_junk(p, registers["select"])


def _measure_x(wire):
    qp.H(wire)
    return qp.measure(wire, reset=True)


def _emit(steps: list, registers: dict, work) -> None:
    """Apply ``steps`` in order: each child on the registers its ``wire_argnames`` name and on
    the prefix of ``work`` of exactly its need; each junk measurement on every wire of its
    register, keeping the outcomes; an unprepare from outcomes with the latest outcomes of its
    junk register and that register's first ``extra_onehot`` wires (reset by the measurement)."""
    work, outcomes = list(work), {}
    for step in steps:
        if step.op == JUNK_MEASUREMENT:
            outcomes[step.args] = [_measure_x(wire) for wire in registers[step.args]]
            continue
        wires = {name: work[:step.work] if name == "work" else registers[name]
                 for name in step.op.wire_argnames if name != "extra_onehot"}
        if step.op in FROM_OUTCOMES:
            junk = FROM_OUTCOMES[step.op]
            step.op(outcomes[junk], *step.args, **wires,
                    extra_onehot=registers[junk][:step.extra_onehot])
        else:
            step.op(*step.args, **wires)


def _resources(steps: list, p: WalkParams) -> dict:
    """The declaration of ``steps``: each distinct child once, with its count, its compilable
    arguments and exactly the register sizes :func:`_emit` passes, and the junk measurements'
    ``Hadamard`` and ``MidMeasure`` gates (spec 7.4 rule 2)."""
    sizes = _sizes(p)
    measure = resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True))
    resources = Counter()
    for step, count in Counter(steps).items():
        if step.op == JUNK_MEASUREMENT:
            resources[qp.H] += count * sizes[step.args]
            resources[measure] += count * sizes[step.args]
            continue
        dynamic = ((abstract_junk_outcomes(sizes[FROM_OUTCOMES[step.op]]),)
                   if step.op in FROM_OUTCOMES else ())

        def build(step=step, dynamic=dynamic):
            return step.op(*dynamic, *step.args, **{
                name: Wire[step.work if name == "work" else step.extra_onehot
                           if name == "extra_onehot" else sizes[name]]
                for name in step.op.wire_argnames})

        resources[resource_key(build)] += count
    return dict(resources)


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
    concatenation of named registers in layout order; the junk registers ``junk_x_o`` and
    ``junk_b`` are the first wires of ``select`` (spec 10.2).

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``outer``      the ``outer`` group (71)       ``|0>``               see the action
    ``inner``      the ``inner`` group (65)       ``|0>``               see the action
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
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
    ``b``, ``m_b``, ``sigma_0``) is back in ``|0>``: the junk and target measurements and the
    angle erase reset the loaded data, the measured uncomputes clean the comparators, and the
    adjoint uniform preparations clear the success flags. So ``RefAB``'s reflection about those
    six is the walk's reflection about every ancilla.

    **Decomposition** (spec 9.21; spec 16, Q-1 variant (b)). The groups are split with
    ``WalkLayout.split_group`` and ``WalkLayout.split_junk``; each child gets its registers by
    name and the prefix of ``work`` of its need: ``OuterPrepare``, the X measurement of
    ``junk_x_o``; ``InnerPrepare``, the X measurement of ``junk_b``, ``InnerSigns``,
    ``InnerSelect``, ``InnerUnprepareFromOutcomes``; ``RefT2``; ``InnerPrepare``, the X
    measurement of ``junk_b``, ``InnerSelectDagger``, ``InnerSigns``,
    ``InnerUnprepareFromOutcomes``; ``OuterUnprepareFromOutcomes``. An X measurement is ``H`` and
    ``qp.measure(q, reset=True)`` on every junk wire; each unprepare from outcomes gets the
    outcomes of the latest measurement of its junk register and that register's first
    ``extra_onehot`` wires, reset by then (the select loads and erases the angles on the same
    wires in between). The prepare classes take the whole ``phase_grad`` (they use its top
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
    one). Of ``M``, :math:`(K_1 - 1) d_1 + 2 (K_2 - 1) d_2` are this rule's own junk
    measurements (:func:`dfthc.costs.junk_measurements`) and the unprepares from outcomes
    measure the targets, :math:`d_1 + 2 d_2`. At FeMoco-54:
    :math:`278 + 2 \cdot 4724 + 20 = 9746`, ``M = 2524`` (852 of them junk), ``A = 8629``.
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
    free comparator adjoints -46 (``cmp-free``) and the rest +2 (spec 13.1). As the paper does
    (App. B.7.2, steps 2 and 4, and Table 3, whose inner QROM temporaries share the angle
    qubits), the rule measures each read's junk right after its prepare; dfthcbe keeps the junk
    until the erase.

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
    ``outer``      the ``outer`` group (71)       ``|0>``               see the action
    ``inner``      the ``inner`` group (65)       ``|0>``               see the action
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCBlockEncoding`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCBlockEncoding`, on both control
    branches; ``control`` is disjoint from the groups (unchecked).

    **Decomposition** (spec 9.21). :class:`DFTHCBlockEncoding`'s eleven children and three junk
    measurements with ``ControlledRefT2(control, succ_x_o, b, m_b, sigma_0, work)`` in place of
    ``RefT2``: every other step is uncontrolled, the same operator with the same compilable
    arguments and register sizes as in :class:`DFTHCBlockEncoding`, so the measurements and the
    measured unprepares are never controlled (spec 7.4 rule 5).

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0).

    **Cost.** :func:`dfthc.costs.block_encoding` (controlled): :class:`DFTHCBlockEncoding`'s
    ``T + 1`` (the controlled reflection's extra ``TemporaryAND``), the same ``M``,
    ``A = T - (T_Toffoli - 1)``. At FeMoco-54: ``T = 9747``, ``M = 2524``, ``A = 8630``.

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


def _walk_children(p: WalkParams, t: WalkTables, controlled: bool) -> list[Step]:
    """The walk's children (spec 9.22): the block encoding on the whole groups and pool, then
    ``RefAB``; both controlled if ``controlled``."""
    block_encoding = ControlledDFTHCBlockEncoding if controlled else DFTHCBlockEncoding
    reflection = ControlledRefAB if controlled else RefAB
    return [Step(block_encoding, (p.as_compilable(), t.as_compilable()), WalkLayout.work_size(p)),
            Step(reflection, (), reflection.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)["work"])]


class DFTHCWalk(Operator2):
    r"""The qubitized walk :math:`W = \mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}\,\mathrm{BE}`
    (steps 1 to 28) of the DFTHC block encoding.

    **Action.** :class:`DFTHCBlockEncoding`, then
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}} = 2|0\rangle\langle 0| - I` on
    ``x_o``, ``m_x_o``, ``sigma_1``, ``b``, ``m_b`` and ``sigma_0`` (spec 3.3). With the ancilla
    groups in :math:`|0\rangle`, :math:`\langle 0|W^k|0\rangle = T_k(\tilde H_{SA}/\tilde\Lambda
    - I)` for the Chebyshev polynomials :math:`T_k` (spec 12.4, E-2), and :math:`W` has the
    eigenphases :math:`\pm\arccos(E_j/\tilde\Lambda - 1)` for the eigenvalues :math:`E_j` of
    :math:`\tilde H_{SA}` (spec 3.3, 8.5). The output is the same for every outcome of the erase
    measurements.

    **Registers.** Those of :class:`DFTHCBlockEncoding`, with the same input and output states.

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``outer``      the ``outer`` group (71)       ``|0>``               see the action
    ``inner``      the ``inner`` group (65)       ``|0>``               see the action
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCBlockEncoding`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCBlockEncoding`. After every step only
    the six index registers of :math:`\mathcal{A}\cup\mathcal{B}` may differ from ``|0>``;
    they carry the walk's state to the next step.

    **Decomposition** (spec 9.22). ``DFTHCBlockEncoding(params, tables, outer, inner, select,
    system, phase_grad, work)`` on the whole groups, then ``RefAB(x_o, m_x_o, sigma_1, b, m_b,
    sigma_0, work[:n_w - 3])`` on the registers ``WalkLayout.split_group`` names.

    **Adjoint and control.** Never adjointed: its rule measures. Never controlled: the
    controlled walk is the class :class:`ControlledDFTHCWalk`. There is no ``C(DFTHCWalk)``
    rule, so ``qp.ctrl(DFTHCWalk(...))`` fails to compile with a ``CompileError`` (the erases
    cannot be controlled); a delegating rule would compile but bring back the cascade of
    distributed candidates (spec 7.4 rule 6, Task G0.1).

    **Cost.** :func:`dfthc.costs.walk` (uncontrolled): :math:`T = T_{BE} + T_{RefAB}`,
    :math:`M = M_{BE}`; ``A`` = :func:`dfthc.costs.measured_and_uncomputes` (uncontrolled),
    ``T - T_Toffoli``. At FeMoco-54: :math:`9746 + 44 = 9790`, ``M = 2524``, ``A = 8672``, so
    ``M + A = 11196`` physical measurements (spec 9.24). ``CNOT`` and ``PauliX`` counts are
    upper bounds (spec 7.6).

    **Paper and dfthcbe.** Steps 1 to 28 of spec 3.5: the walk
    :math:`W = \mathrm{Ref}_{\mathrm{a}\mathcal{B}}\,U^\dagger\,\mathrm{Ref}_{\mathcal{B}}\,U`
    with eigenphases :math:`\pm\arccos(E_k/\Lambda - 1)` (PRX 15, 041016; arXiv:2502.15882v1
    Eq. (10), App. A.2 Eqs. (76) and (77), :math:`\mathrm{Ref} = 2|0\rangle\langle 0| - I` as in
    App. A.1), whose reflection covers every ancilla of the linear combination and of the
    qubitization signal (App. B); its Table 3 (PRX Table III) gives 9,997 Toffolis at FeMoco-54,
    46 of them for :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` with the phase-estimation
    control (App. B.7, Eqs. (96) and (128)). Ours is -204: the block encoding's -202 and
    ``RefAB``'s 44 against 46 (spec 13.1). dfthcbe: ``DFTHCWalk``
    (``dfthcbe/block_encoding.py``), the same two children with the same sign (Qualtran's
    ``reflection_around_zero`` with ``global_phase=-1``), :math:`9596 + 45 = 9641` by its call
    graph; ours is +152: the block encoding's +153 and ``RefAB``'s 44 against 45 (dfthcbe's
    uncontrolled reflection adds a phase target).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import DFTHCWalk
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p)
        groups = {name: getattr(layout, name) for name in DFTHCWalk.wire_argnames}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            DFTHCWalk(p.as_compilable(), t.as_compilable(), **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 150 + 8 = 158
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = _GROUPS

    def __init__(self, params, tables, outer, inner, select, system, phase_grad, work):
        wires = (outer, inner, select, system, phase_grad, work)
        compilable = _validate("DFTHCWalk", self.wire_argnames, params, tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """The layout's group sizes (spec 10.1); ``tables`` does not change them."""
        return _group_sizes(WalkParams.from_compilable(params))


def _walk_resources(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_walk_children(p, t, controlled=False), p)


@qp.register_resources(_walk_resources)
def _walk(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    _emit(_walk_children(p, t, controlled=False),
          _registers(p, outer, inner, select, system, phase_grad), work)


register_rule(DFTHCWalk, _walk)


class ControlledDFTHCWalk(Operator2):
    r"""The controlled walk: the explicit controlled class of :class:`DFTHCWalk` (spec 7.4 rule
    6), :math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,U^\dagger\,
    C(\mathrm{REF}_{T_2})\,U`.

    **Action.** :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle 1|_{control}
    \otimes W` on inputs whose ancilla groups are in :math:`|0\rangle` and on every state the walk
    reaches from there, where only the six index registers of :math:`\mathcal{A}\cup\mathcal{B}`
    are nonzero (as between QPE rounds; the prepares and unprepares run on both branches and
    :math:`U^\dagger U = I` there too), with :math:`W` the action of :class:`DFTHCWalk` for the
    same arguments: control ``|0>`` returns the whole input,
    control ``|1>`` gives :class:`DFTHCWalk`'s whole output, global sign included, and control
    ``|+>`` their superposition (spec 12.4, E-3). The output is the same for every outcome of the
    erase measurements.

    **Registers.** Those of :class:`DFTHCWalk` after a one-wire ``control``, which stays a
    control (only phases on it).

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``control``    1                              any                   a control
    ``outer``      the ``outer`` group (71)       ``|0>``               see the action
    ``inner``      the ``inner`` group (65)       ``|0>``               see the action
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCWalk`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCWalk`, on both control branches;
    ``control`` is disjoint from the groups (unchecked).

    **Decomposition** (spec 9.22). ``ControlledDFTHCBlockEncoding(params, tables, control, outer,
    inner, select, system, phase_grad, work)``, then ``ControlledRefAB(control, x_o, m_x_o,
    sigma_1, b, m_b, sigma_0, work[:n_w - 2])``: only the two reflections are controlled; the
    prepares, selects and measured unprepares run uncontrolled (spec 7.4 rule 5).

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0);
    ``ControlledWalkPower`` (spec 9.23) repeats it.

    **Cost.** :func:`dfthc.costs.walk` (controlled): :class:`DFTHCWalk`'s ``T + 2``, the same
    ``M``; ``A`` = :func:`dfthc.costs.measured_and_uncomputes` (controlled). At FeMoco-54:
    ``T = 9792``, ``M = 2524``, ``A = 8674``, ``M + A = 11198``.

    **Paper and dfthcbe.** The controlled walk of spec 3.3, the unit of the paper's phase
    estimation (arXiv:2502.15882v1 Sec. II, after Eq. (11)). The paper counts the phase-estimation
    control on :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` only (App. B.7.2, Eq. (128)),
    which gives controlled even powers of :math:`W` because the block encoding is self-inverse;
    this class also controls :math:`\mathrm{REF}_{T_2}` (+1), so every power works. dfthcbe:
    ``DFTHCWalk(is_controlled=True)``, which also controls only the two reflections,
    :math:`9597 + 45 = 9642` Toffolis at FeMoco-54 by its call graph (its controlled ``ref_ab``
    uses the control as the phase target, so it costs the same as the uncontrolled one); ours is
    +153.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import ControlledDFTHCWalk
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p, qpe_bits=1)
        groups = {name: getattr(layout, name) for name in ControlledDFTHCWalk.wire_argnames[1:]}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            ControlledDFTHCWalk(p.as_compilable(), t.as_compilable(), layout.qpe_control,
                                **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 158 + 2 = 160
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = ("control", *_GROUPS)

    def __init__(self, params, tables, control, outer, inner, select, system, phase_grad, work):
        wires = (control, outer, inner, select, system, phase_grad, work)
        compilable = _validate("ControlledDFTHCWalk", self.wire_argnames, params, tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """``control`` (1 wire), then the layout's group sizes."""
        return {"control": 1} | _group_sizes(WalkParams.from_compilable(params))


def _controlled_walk_resources(params, tables, control, outer, inner, select, system, phase_grad,
                               work):
    p, t = _rebuild(params, tables)
    return _resources(_walk_children(p, t, controlled=True), p)


@qp.register_resources(_controlled_walk_resources)
def _controlled_walk(params, tables, control, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    _emit(_walk_children(p, t, controlled=True),
          _registers(p, outer, inner, select, system, phase_grad, control), work)


register_rule(ControlledDFTHCWalk, _controlled_walk)
