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

r"""The block encoding of the sum-of-squares Hamiltonian, the qubitized walk and the walk's
``C(DFTHCWalk)`` rule (spec 9.21 and 9.22; steps 1 to 28).

**The circuit** (spec 3.3 and 3.5). The block encoding is :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U`
with :math:`U` = ``OuterPrepare``, ``InnerPrepare``, ``InnerSigns``, ``InnerSelect``,
``InnerUnprepareFromOutcomes`` in time order and :math:`U^\dagger` = ``InnerPrepare``,
``InnerSelectDagger``, ``InnerSigns``, ``InnerUnprepareFromOutcomes``,
``OuterUnprepareFromOutcomes``; its block on the ancillas in :math:`|0\rangle` is
:math:`\tilde H_{SA}/\tilde\Lambda - I` (spec 8.5). Every adjoint in it is an explicit class
(``InnerUnprepareFromOutcomes``, ``InnerSelectDagger``, ``OuterUnprepareFromOutcomes``; spec 7.4
rule 6). The walk is :math:`W = \mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}\,\mathrm{BE}`: the block
encoding, then ``ref_ab``. The two reflections are the helpers ``ref_t2`` and ``ref_ab`` of
:mod:`dfthc.ops.reflections` (a multi-controlled Z from PennyLane's ``MultiControlledX``),
called in the rules. The controlled walk ``qp.ctrl(DFTHCWalk(...), control)`` has one pinned
rule, registered for ``"C(DFTHCWalk)"``, which controls only the reflections: on states whose
ancillas start in :math:`|0\rangle`, :math:`U^\dagger U = I`, so
:math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,U^\dagger\,C(\mathrm{REF}_{T_2})\,U`
(spec 3.3). The block encoding has no ``C(...)`` rule; the controlled walk's rule runs the block
encoding's steps itself.

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
``inner``, ``select``, ``system``, ``phase_grad``, ``work``; the controlled walk's rule also its
one control wire). Its rule splits them once with :meth:`WalkLayout.split_group` and
:meth:`WalkLayout.split_junk`, unpacks every register into a list once, and passes each child the
registers its ``wire_argnames`` name, with the prefix of the ``work`` group of exactly the child's
need (spec 9.0, wire passing). ``work`` is the walk's shared pool: every child returns it in
:math:`|0\rangle`; the comparators' dirty work registers belong to the ``outer`` and ``inner``
groups, never to ``work`` (spec 10.2).

**Arguments.** ``params`` is a ``WalkParams`` and ``tables`` a ``WalkTables``, or their
``as_compilable()`` tuples, which the constructors store; the rules rebuild both with
``from_compilable``. One list,
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
from dfthc.ops.reflections import (
    ref_ab,
    ref_ab_resources,
    ref_t2,
    ref_t2_resources,
    reflection_work,
)
from dfthc.ops.select import InnerSelect, InnerSelectDagger
from dfthc.ops.select import _check as _check_select
from dfthc.params import WalkParams
from dfthc.registers import GROUPS, WalkLayout

_GROUPS = ("outer", "inner", "select", "system", "phase_grad", "work")
"""The wire arguments of the assembly operators."""

JUNK_MEASUREMENT = "junk measurement"
"""The step of :func:`_children` that X-measures the junk register its ``args`` names."""

FROM_OUTCOMES = {OuterUnprepareFromOutcomes: "junk_x_o", InnerUnprepareFromOutcomes: "junk_b"}
"""Each unprepare from outcomes and the junk register whose outcomes it takes and whose first
wires it borrows as ``extra_onehot``."""

REFLECTIONS = {ref_t2: (ref_t2_resources, ("succ_x_o", "b", "m_b", "sigma_0")),
               ref_ab: (ref_ab_resources, ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0"))}
"""Each reflection helper, its resource helper and the registers it takes before ``work``."""

LENT_TO_REFLECTIONS = "angles"
"""Each reflection gets the last wire of this register as its ``extra_work``. Every ``angles``
wire is ``|0>`` at both reflections: ``AngleErase`` resets them at the end of each select, the
unprepares from outcomes return the junk wires they borrow in ``|0>``, and the next load is in the
next select (after the junk measurement of the next prepare)."""


class Step(NamedTuple):
    """One step of an assembly rule: a child with its compilable arguments, the prefix of the
    ``work`` pool it takes and its ``extra_onehot`` wires (for an unprepare from outcomes); with
    ``op`` a reflection helper of :data:`REFLECTIONS`, ``args`` is ``(controlled,)``; with
    ``op`` = :data:`JUNK_MEASUREMENT`, the X measurement of the junk register ``args``."""

    op: type | str
    args: tuple | str
    work: int
    extra_onehot: int = 0


def _rebuild(params, tables) -> tuple[WalkParams, WalkTables]:
    return WalkParams.from_compilable(params), WalkTables.from_compilable(tables)


def _reflection_step(helper, p: WalkParams, controlled: bool) -> Step:
    """The step of a reflection helper on the layout's registers, with its work need."""
    sizes = WalkLayout.register_sizes(p)
    n_w = sum(sizes[name] for name in REFLECTIONS[helper][1])
    return Step(helper, (controlled,), reflection_work(n_w, controlled))


def _children(p: WalkParams, t: WalkTables, controlled: bool) -> list[Step]:
    """The block encoding's steps in time order (spec 9.21; spec 16, Q-1 variant (b)): the
    children, each junk register's X measurement right after its prepare, and the unprepares from
    those outcomes; the reflection :func:`~dfthc.ops.reflections.ref_t2` is controlled if
    ``controlled``."""
    outer = (p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    inner = (p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn, p.b_k2,
             p.k2, p.ke2, p.s_u)
    select = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    outer_sizes = OuterUnprepareFromOutcomes.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)
    inner_sizes = InnerUnprepareFromOutcomes.required_wires(p.N, p.R, p.B, p.C, p.b_k2, p.k2,
                                                            p.ke2, p.s_u)
    outer_work, inner_work = outer_sizes["work"], inner_sizes["work"]
    select_work = InnerSelect.required_wires(*select)["work"]
    return [
        Step(OuterPrepare, outer, outer_work),
        Step(JUNK_MEASUREMENT, "junk_x_o", 0),
        Step(InnerPrepare, inner, inner_work),
        Step(JUNK_MEASUREMENT, "junk_b", 0),
        Step(InnerSigns, (), 0),
        Step(InnerSelect, select, select_work),
        Step(InnerUnprepareFromOutcomes, inner, inner_work, inner_sizes["extra_onehot"]),
        _reflection_step(ref_t2, p, controlled),
        Step(InnerPrepare, inner, inner_work),
        Step(JUNK_MEASUREMENT, "junk_b", 0),
        Step(InnerSelectDagger, select, select_work),
        Step(InnerSigns, (), 0),
        Step(InnerUnprepareFromOutcomes, inner, inner_work, inner_sizes["extra_onehot"]),
        Step(OuterUnprepareFromOutcomes, outer, outer_work, outer_sizes["extra_onehot"]),
    ]


def _as_params(params) -> WalkParams:
    return params if isinstance(params, WalkParams) else WalkParams.from_compilable(params)


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
    ``2**w_b <= (N - 1)(b_rot - 1) + 1``), and the layout's ``work`` pool covers every child of
    the walk and the controlled walk. ``WalkParams`` and ``WalkTables`` objects are accepted as
    their ``as_compilable()`` tuples. ``ValueError`` otherwise."""
    if isinstance(params, WalkParams):
        params = params.as_compilable()
    if isinstance(tables, WalkTables):
        tables = tables.as_compilable()
    if not isinstance(params, (tuple, list)) or not isinstance(tables, (tuple, list)):
        raise ValueError("params and tables must be WalkParams and WalkTables or their "
                         "as_compilable() tuples")
    p, t = _rebuild(params, tables)
    _check_outer(p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    _check_inner(p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn,
                 p.b_k2, p.k2, p.ke2, p.s_u)
    _check_select(t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    needs = {step.op.__name__: step.work for step in _children(p, t, controlled=True)
             if step.op != JUNK_MEASUREMENT}
    needs["ref_ab"] = _reflection_step(ref_ab, p, controlled=True).work
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
    the prefix of ``work`` of exactly its need; a reflection helper likewise, on the registers
    :data:`REFLECTIONS` names, controlled by ``registers["control"]`` if its step says so; each
    junk measurement on every wire of its register, keeping the outcomes; an unprepare from
    outcomes with the latest outcomes of its junk register and that register's first
    ``extra_onehot`` wires (reset by the measurement)."""
    work, outcomes = list(work), {}
    for step in steps:
        if step.op == JUNK_MEASUREMENT:
            outcomes[step.args] = [_measure_x(wire) for wire in registers[step.args]]
            continue
        if step.op in REFLECTIONS:
            (controlled,) = step.args
            step.op(*(registers[name] for name in REFLECTIONS[step.op][1]), work[:step.work],
                    control=registers["control"][0] if controlled else None,
                    extra_work=registers[LENT_TO_REFLECTIONS][-1:])
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
    arguments and exactly the register sizes :func:`_emit` passes, a reflection helper's gates
    from its resource helper, and the junk measurements' ``Hadamard`` and ``MidMeasure`` gates
    (spec 7.4 rule 2)."""
    sizes = _sizes(p)
    measure = resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True))
    resources = Counter()
    for step, count in Counter(steps).items():
        if step.op == JUNK_MEASUREMENT:
            resources[qp.H] += count * sizes[step.args]
            resources[measure] += count * sizes[step.args]
            continue
        if step.op in REFLECTIONS:
            gates, names = REFLECTIONS[step.op]
            for gate, number in gates(*(sizes[name] for name in names), *step.args,
                                      extra_work=1).items():
                resources[gate] += count * number
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
    ``select``     ``(N - 1)(b_rot - 1)`` +       ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (746)
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** ``params``: :meth:`WalkParams.as_compilable`, the 13 fields; the
    erases' one-hot bits must fit their reset registers (:math:`2^{k_{e1}} \le K_1 d_1`,
    :math:`2^{k_{e2} + w_b} \le K_2 d_2`, :math:`2^{w_b} \le (N - 1)(b_{rot} - 1) + 1`)
    and ``WalkLayout.work_size`` must cover every child. ``tables``:
    :meth:`WalkTables.as_compilable` of tables with the shapes of ``params``, in the ranges the
    prepares and the select accept. The constructor checks both (``ValueError``).

    **Pre- and postconditions.** The groups are pairwise disjoint (unchecked). ``phase_grad``
    holds the state ``PhaseGradientStatePrep`` prepares and keeps it. The block needs the ancilla
    groups in ``|0>``; in the walk the input is also any state the walk reaches from there, on
    which every erase's precondition holds (spec 3.3). Afterwards every register but the six
    index registers of :math:`\mathcal{A}\cup\mathcal{B}` (``x_o``, ``m_x_o``, ``sigma_1``,
    ``b``, ``m_b``, ``sigma_0``) is back in ``|0>``: the junk and target measurements and the
    angle erase reset the loaded data, the measured uncomputes clean the comparators, and the
    adjoint uniform preparations clear the success flags. So ``ref_ab``'s reflection about those
    six is the walk's reflection about every ancilla.

    **Decomposition** (spec 9.21; spec 16, Q-1 variant (b)). The groups are split with
    ``WalkLayout.split_group`` and ``WalkLayout.split_junk``; each child gets its registers by
    name and the prefix of ``work`` of its need: ``OuterPrepare``, the X measurement of
    ``junk_x_o``; ``InnerPrepare``, the X measurement of ``junk_b``, ``InnerSigns``,
    ``InnerSelect``, ``InnerUnprepareFromOutcomes``; ``ref_t2(succ_x_o, b, m_b, sigma_0,
    work[:n_w - 3], extra_work=angles[-1:])`` (the multi-controlled Z and ``GlobalPhase(pi)``,
    with the last ``angles`` wire lent in ``|0>``, :data:`LENT_TO_REFLECTIONS`;
    :math:`n_w = w_b + b_{k2} + 2`); ``InnerPrepare``, the X measurement of ``junk_b``,
    ``InnerSelectDagger``, ``InnerSigns``, ``InnerUnprepareFromOutcomes``;
    ``OuterUnprepareFromOutcomes``. An X measurement is ``H`` and ``qp.measure(q, reset=True)``
    on every junk wire; each unprepare from outcomes gets the
    outcomes of the latest measurement of its junk register and that register's first
    ``extra_onehot`` wires, reset by then (the select loads and erases the angles on the same
    wires in between). The prepare classes take the whole ``phase_grad`` (they use its top
    ``s_u`` wires), the selects exactly its ``b_rot`` wires; both selects get
    ``fixup_bits = w_b``. ``InnerSigns`` is diagonal and touches no register of the selects, so
    its place on either side of them is exact; this is spec 9.21's.

    **Adjoint and control.** Never adjointed: its rule measures (spec 7.4 rule 5), and the
    operator is its own inverse anyway. Never controlled: there is no ``C(DFTHCBlockEncoding)``
    rule, so ``qp.ctrl`` of this operator does not compile (spec 7.4 rule 6). The controlled walk's
    rule (:class:`DFTHCWalk`) runs these steps with ``ref_t2`` controlled, the controlled block
    encoding :math:`U^\dagger\,C(\mathrm{REF}_{T_2})\,U`: ``ref_t2(..., work[:n_w - 2],
    control=control)`` (the multi-controlled Z over the control and the pattern, then ``Z`` on the
    control), every other step uncontrolled and unchanged.

    **Cost.** :func:`dfthc.costs.block_encoding` (uncontrolled):
    :math:`T = T_{out} + T_{out}^\dagger + 2(T_{in} + T_{sel} + T_{in}^\dagger) + T_{ref\_t2}`,
    :math:`M = K_1 d_1 + 2(K_2 d_2 + (N - 1)(b_{rot} - 1) + 3)`; ``A = T`` (every Toffoli is a
    ``TemporaryAND`` uncomputed by measurement). Of ``M``,
    :math:`(K_1 - 1) d_1 + 2 (K_2 - 1) d_2` are this rule's own junk
    measurements (:func:`dfthc.costs.junk_measurements`) and the unprepares from outcomes
    measure the targets, :math:`d_1 + 2 d_2`. At FeMoco-54:
    :math:`278 + 2 \cdot 4724 + 20 = 9746`, ``M = 2418`` (852 of them junk), ``A = 9746``.
    Controlled (the controlled walk's first 27 steps): ``T + 1`` (the controlled reflection's
    extra ``TemporaryAND``), the same ``M``, ``A = T``; at FeMoco-54 ``T = A = 9747``.
    ``CNOT`` and ``PauliX`` counts are upper bounds (spec 7.6).

    **Paper and dfthcbe.** Steps 1 to 27 of spec 3.5: the sum-of-squares block encoding
    :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U` (PRX 15, 041016; arXiv:2502.15882v1 Eq. (11) and
    Fig. 1), built in App. B: the prepare over :math:`x_o` (B.1, Eq. (87)) and the step of
    oblivious amplitude amplification, whose reflection is controlled on the success of
    :math:`x_o` (B.2, Eq. (88)); the costs are its Table 3 (PRX Table III). dfthcbe:
    ``DFTHCBlockEncoding`` (``dfthcbe/block_encoding.py``), 9,596 Toffolis at FeMoco-54 by its
    call graph, :math:`212 + 2 (913 + 220) + 2 \cdot 3502 + 20 + 94`; ours is +150 (the inner
    reads +120, the angle loads +50 and erases +24, the inner erases +4, the outer read +4, the
    uniform preparations -36, the outer unprepare -12 and the spin multiplexers -4; spec 13.1).
    The paper gives 9,997 per walk, 46 of them for
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}`; against the other 9,951 ours is -205: the
    Givens ladders -424 (``rz-addsub``), the 26-bit inner entries against its 20-bit ones +184,
    exact boundaries and one-hot padding +80 (``exact-iter``, ``no-prune``, ``onehot-pad``), the
    free comparator adjoints -46 (``cmp-free``) and the rest +1 (spec 13.1). As the paper does
    (App. B.7.2, steps 2 and 4, and Table 3, whose inner QROM temporaries share the angle
    qubits), the rule measures each read's junk right after its prepare; dfthcbe keeps the junk
    until the erase. The controlled block encoding is dfthcbe's
    ``DFTHCBlockEncoding(is_controlled=True)``, 9,597 Toffolis at FeMoco-54, which also controls
    only ``ref_t2``.

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
    def required_wires(params, tables=None) -> dict[str, int]:
        """The layout's group sizes (spec 10.1), for ``qp.registers``; ``tables`` does not change
        them."""
        return _group_sizes(_as_params(params))


def _block_encoding_resources(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_children(p, t, controlled=False), p)


@qp.register_resources(_block_encoding_resources)
def _block_encoding(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    _emit(_children(p, t, controlled=False),
          _registers(p, outer, inner, select, system, phase_grad), work)


register_rule(DFTHCBlockEncoding, _block_encoding)


def _walk_children(p: WalkParams, t: WalkTables, controlled: bool) -> list[Step]:
    """The walk's steps (spec 9.22): the block encoding on the whole groups and pool, then
    :func:`~dfthc.ops.reflections.ref_ab`; if ``controlled``, the block encoding's own steps
    with ``ref_t2`` controlled, then ``ref_ab`` controlled."""
    if controlled:
        return [*_children(p, t, controlled=True), _reflection_step(ref_ab, p, controlled=True)]
    return [Step(DFTHCBlockEncoding, (p.as_compilable(), t.as_compilable()),
                 WalkLayout.work_size(p)),
            _reflection_step(ref_ab, p, controlled=False)]


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
    ``select``     ``(N - 1)(b_rot - 1)`` +       ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (746)
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCBlockEncoding`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCBlockEncoding`. After every step only
    the six index registers of :math:`\mathcal{A}\cup\mathcal{B}` may differ from ``|0>``;
    they carry the walk's state to the next step.

    **Decomposition** (spec 9.22). ``DFTHCBlockEncoding(params, tables, outer, inner, select,
    system, phase_grad, work)`` on the whole groups, then ``ref_ab(x_o, m_x_o, sigma_1, b, m_b,
    sigma_0, work[:n_w - 3], extra_work=angles[-1:])`` (the multi-controlled Z and
    ``GlobalPhase(pi)``) on the registers ``WalkLayout.split_group`` names.

    **Adjoint and control.** Never adjointed: its rule measures. Controlled with
    ``qp.ctrl(DFTHCWalk(...), control, control_values=...)``, whose one pinned rule, registered for
    ``"C(DFTHCWalk)"``, gives :math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,
    U^\dagger\,C(\mathrm{REF}_{T_2})\,U` (spec 3.3, 9.22): the block encoding's steps with
    ``ref_t2(..., control=control)`` (see :class:`DFTHCBlockEncoding`), then ``ref_ab(...,
    work[:n_w - 2], control=control)``; the prepares, selects and measured unprepares run
    uncontrolled (spec 7.4 rule 5). Its action is :math:`|0\rangle\langle 0| \otimes I +
    |1\rangle\langle 1| \otimes W` on the control (the control value's state applies
    :math:`W`), global sign included, on inputs whose ancilla groups are in :math:`|0\rangle` and
    on every state the walk reaches from there (spec 12.4, E-3); the control stays a control
    (only phases on it) and must be disjoint from the groups (unchecked). A control value 0 puts
    ``X`` on the control before and after, conditioned on the value, which may be traced. The rule
    takes exactly one control wire (``ValueError`` otherwise) and uses none of ``qp.ctrl``'s
    ``work_wires``: the ``work`` group serves every step. ``ControlledWalkPower`` (spec 9.23)
    repeats the controlled walk.

    **Cost.** :func:`dfthc.costs.walk` (uncontrolled): :math:`T = T_{BE} + T_{ref\_ab}`,
    :math:`M = M_{BE}`; ``A`` = :func:`dfthc.costs.measured_and_uncomputes` (uncontrolled),
    ``A = T``. At FeMoco-54: :math:`9746 + 44 = 9790`, ``M = 2418``, ``A = 9790``, so
    ``M + A = 12208`` physical measurements (spec 9.24). Controlled: :func:`dfthc.costs.walk`
    (controlled), ``T + 2`` (one extra ``TemporaryAND`` per controlled reflection), the same
    ``M``; ``A`` = :func:`dfthc.costs.measured_and_uncomputes` (controlled). At FeMoco-54:
    ``T = 9792``, ``M = 2418``, ``A = 9792``, ``M + A = 12210``. ``CNOT`` and ``PauliX`` counts
    are upper bounds (spec 7.6).

    **Paper and dfthcbe.** Steps 1 to 28 of spec 3.5: the walk
    :math:`W = \mathrm{Ref}_{\mathrm{a}\mathcal{B}}\,U^\dagger\,\mathrm{Ref}_{\mathcal{B}}\,U`
    with eigenphases :math:`\pm\arccos(E_k/\Lambda - 1)` (PRX 15, 041016; arXiv:2502.15882v1
    Eq. (10), App. A.2 Eqs. (76) and (77), :math:`\mathrm{Ref} = 2|0\rangle\langle 0| - I` as in
    App. A.1), whose reflection covers every ancilla of the linear combination and of the
    qubitization signal (App. B); its Table 3 (PRX Table III) gives 9,997 Toffolis at FeMoco-54,
    46 of them for :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` with the phase-estimation
    control (App. B.7, Eqs. (96) and (128)). Ours is -207: the block encoding's -205 and
    ``ref_ab``'s 44 against 46 (spec 13.1). dfthcbe: ``DFTHCWalk``
    (``dfthcbe/block_encoding.py``), the same two children with the same sign (Qualtran's
    ``reflection_around_zero`` with ``global_phase=-1``), :math:`9596 + 45 = 9641` by its call
    graph; ours is +149: the block encoding's +150 and ``ref_ab``'s 44 against 45 (dfthcbe's
    uncontrolled reflection adds a phase target). The controlled walk is the unit of the paper's
    phase estimation (Sec. II, after Eq. (11)), which counts the control on
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` only (App. B.7.2, Eq. (128)) and so
    gives controlled even powers of :math:`W`; the rule also controls
    :math:`\mathrm{REF}_{T_2}` (+1), so every power works. dfthcbe:
    ``DFTHCWalk(is_controlled=True)``, which also controls only the two reflections,
    :math:`9597 + 45 = 9642` by its call graph (its controlled ``ref_ab`` uses the control as the
    phase target, so it costs the same as the uncontrolled one).

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

        def controlled():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            qp.ctrl(DFTHCWalk(p.as_compilable(), t.as_compilable(), **groups),
                    control=layout.num_wires)

        compiled_resources(controlled, num_wires=layout.num_wires + 1).toffoli  # 158 + 2 = 160
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = _GROUPS

    def __init__(self, params, tables, outer, inner, select, system, phase_grad, work):
        wires = (outer, inner, select, system, phase_grad, work)
        compilable = _validate("DFTHCWalk", self.wire_argnames, params, tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables=None) -> dict[str, int]:
        """The layout's group sizes (spec 10.1), for ``qp.registers``; ``tables`` does not change
        them."""
        return _group_sizes(_as_params(params))


def _walk_resources(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_walk_children(p, t, controlled=False), p)


@qp.register_resources(_walk_resources)
def _walk(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    _emit(_walk_children(p, t, controlled=False),
          _registers(p, outer, inner, select, system, phase_grad), work)


register_rule(DFTHCWalk, _walk)


def _controlled_walk_resources(base, control_wires, control_values, work_wires, work_wire_type):
    p, t = _rebuild(base.params, base.tables)
    resources = Counter(_resources(_walk_children(p, t, controlled=True), p))
    resources[qp.X] += 2
    return dict(resources)


@qp.register_resources(_controlled_walk_resources)
def _controlled_walk(base, control_wires, control_values, work_wires, work_wire_type):
    """``work_wires`` and ``work_wire_type`` are unused: ``base.work`` serves every step."""
    if len(control_wires) != 1:
        raise ValueError(f"C(DFTHCWalk) takes one control wire; got {len(control_wires)}")
    p, t = _rebuild(base.params, base.tables)
    control = control_wires[0]
    flip = qp.cond(qp.math.logical_not(control_values[0]), qp.X)
    flip(control)
    _emit(_walk_children(p, t, controlled=True),
          _registers(p, base.outer, base.inner, base.select, base.system, base.phase_grad,
                     [control]), base.work)
    flip(control)


register_rule("C(DFTHCWalk)", _controlled_walk)

