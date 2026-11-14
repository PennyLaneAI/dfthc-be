r"""The inner select of the block encoding and its explicit adjoint (spec 9.18; steps 10 to 18).

**The circuit** (spec 3.5 and 9.18). For every branch of the index registers ``InnerSelect``
loads the branch's Givens angles and family flags (:class:`~dfthc.ops.angle_data.TailAngleLoad`,
then :class:`~dfthc.ops.angle_data.SFAngleLoad`), routes the spin register that ``sg`` selects into
``psi_dn`` (:class:`~dfthc.ops.spin.SpinMux`, :class:`~dfthc.ops.spin.SpinSwap`), applies
:math:`U^\dagger\,\mathrm{Maj}\,U` there (:class:`~dfthc.ops.givens.GivensLadder`,
:class:`~dfthc.ops.majorana.MajoranaGate`, :class:`~dfthc.ops.givens.GivensLadderDagger`), undoes
the routing in reverse order, and erases the loaded data by measurement
(:class:`~dfthc.ops.angle_data.AngleErase`). :class:`InnerSelectDagger` is the same sequence with
``qp.adjoint(MajoranaGate(...))``: it also loads first and erases last, and it inverts
:class:`InnerSelect` on every branch, because the select is block diagonal in the index
registers and only the Majorana gate is not Hermitian.

**Wires.** Both classes take the index registers they read (``x_o``, ``succ_x_o``, ``sigma_1``
of the outer group; ``b``, ``succ_b``, ``sigma_0``, ``r`` of the inner group) and the ``select``
and ``system`` registers by name, in ``WalkLayout`` order, plus ``phase_grad`` and ``work``. The
rule unpacks every register into a list once and passes each child only the registers it needs,
with the prefix of ``work`` it needs (spec 9.0, wire passing). Both rules pass ``AngleErase``,
the loads, the ladders and the spin operators identical compilable arguments and register sizes,
so a program holding both selects captures each of those targets once.
"""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.angle_data import AngleErase, SFAngleLoad, TailAngleLoad
from dfthc.ops.givens import GivensLadder, GivensLadderDagger
from dfthc.ops.majorana import MajoranaGate
from dfthc.ops.spin import SpinMux, SpinSwap
from dfthc.params import lg

_WIRES = ("x_o", "succ_x_o", "sigma_1", "b", "succ_b", "sigma_0", "r", "angles", "G0", "G1",
          "b_neq_B", "sg", "psi_up", "psi_dn", "phase_grad", "work")


def _check(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits) -> tuple:
    """The compilable arguments as canonical tuples and ints, each checked against its domain
    (spec 9.0); ``ValueError`` otherwise. The domains are those of ``AngleErase``."""
    N = check_int("N", N, low=2)
    C = check_int("C", C, low=1)
    sf_angles, R, B, b_rot = SFAngleLoad._check(sf_angles, R, B, b_rot)
    rows = canonical_ints("tail_angles", tail_angles)
    if not isinstance(rows, tuple) or len(rows) != N or len(sf_angles[0][0]) != N - 1:
        raise ValueError(f"tail_angles must have N = {N} rows and sf_angles N - 1 = {N - 1} "
                         "angles per entry")
    tail_angles, n_plus, _, _, _ = TailAngleLoad._check(rows, n_plus, R * C, N + R * C, b_rot)
    fixup_bits = check_int("fixup_bits", fixup_bits, low=0, high=lg(B + 1))
    if fixup_bits and 2**fixup_bits > (N - 1) * b_rot + 1:
        raise ValueError(f"the SF one-hot over fixup_bits = {fixup_bits} bits needs "
                         f"{2**fixup_bits} wires, more than the {(N - 1) * b_rot + 1} reset angle "
                         "and b_neq_B wires")
    return tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits


def _child_sizes(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits) -> dict:
    """``required_wires`` of every child, keyed by child; each child gets a prefix of ``work``
    of exactly its size."""
    return {
        "tail": TailAngleLoad.required_wires(tail_angles, n_plus, R * C, N + R * C, b_rot),
        "sf": SFAngleLoad.required_wires(sf_angles, R, B, b_rot),
        "mux": SpinMux.required_wires(),
        "swap": SpinSwap.required_wires(N),
        "ladder": GivensLadder.required_wires(b_rot, N),
        "majorana": MajoranaGate.required_wires(),
        "erase": AngleErase.required_wires(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                                           fixup_bits),
    }


def _required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits) -> dict:
    children = _child_sizes(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)
    return {"x_o": lg(N + R * C), "succ_x_o": 1, "sigma_1": 1, "b": lg(B + 1), "succ_b": 1,
            "sigma_0": 1, "r": lg(R), "angles": (N - 1) * b_rot, "G0": 1, "G1": 1, "b_neq_B": 1,
            "sg": 1, "psi_up": N, "psi_dn": N, "phase_grad": b_rot,
            "work": max(sizes["work"] for sizes in children.values() if "work" in sizes)}


def _validate(name: str, compilable: tuple, wires: tuple) -> tuple:
    compilable = _check(*compilable)
    check_wires(name, dict(zip(_WIRES, wires, strict=True)), _required_wires(*compilable))
    return compilable


class InnerSelect(Operator2):
    r"""The inner select (steps 10 to 18): on every branch, :math:`U^\dagger\,\mathrm{Maj}\,U`
    with the branch's Givens ladder :math:`U` on the spin register that ``sg`` selects.

    **Action.** Block diagonal in the index registers ``x_o``, ``succ_x_o``, ``sigma_1``, ``b``,
    ``succ_b``, ``r`` and diagonal in ``sigma_0``; every ancilla starts and ends in
    :math:`|0\rangle`. On a basis state of the index registers let
    :math:`G_0 = [RC \le x_o < X_o]`, :math:`G_1 = G_0 \wedge [x_o - RC \ge N_+]`, ``x`` the
    angle integers ``tail_angles[x_o - RC]`` if :math:`G_0`, else ``sf_angles[r][b]`` if
    :math:`(r, b) \in [0, R) \times [0, B)`, else zeros, and
    :math:`\tilde u` = ``vector_from_angles(x, b_rot)`` (spec 6.3). The spin register is
    selected by :math:`\sigma = \sigma_{G_0}`: ``psi_dn`` for :math:`\sigma = 0`, ``psi_up`` for
    :math:`\sigma = 1`. With :math:`U` = ``ladder_matrix(x, b_rot, N)`` on that register and
    ``Maj`` the :class:`~dfthc.ops.majorana.MajoranaGate` on (``succ_x_o``, ``succ_b``,
    :math:`G_0`, :math:`G_1`, ``b_neq_B`` = :math:`\neg G_0 \wedge [(r, b) \in [0, R) \times
    [0, B)]`, flavour ``sigma_0``, target the register's qubit 0), the system and ``sigma_0`` get
    :math:`U^\dagger\,\mathrm{Maj}\,U`. On
    the valid branches of the walk this is a generator piece of spec 3.2, with
    :math:`\tilde\gamma_f = \sum_j \tilde u_j \gamma_{j,f}` on the selected spin register:

    - SF (``x_o = rC + c < RC``, ``b < B``, both success flags set):
      :math:`i\tilde\gamma_0\tilde\gamma_1`, on the register ``sigma_0`` selects;
    - D1 (``RC <= x_o < RC + N+``) and Q1 (the other tail rows), ``succ_x_o`` set: with
      ``sigma_0`` projected onto :math:`|+\rangle` before and after,
      :math:`a_{\tilde u} = (\tilde\gamma_0 + i\tilde\gamma_1)/2` (D1) or
      :math:`a_{\tilde u}^\dagger` (Q1) on the register ``sigma_1`` selects, whatever
      ``succ_b`` (spec 9.16);
    - the identity on ``b = B``, on SF rows with ``succ_b = 0`` and wherever ``succ_x_o = 0``.

    **Registers.** :math:`w_x = \lg(N + RC)`, :math:`w_b = \lg(B + 1)`, :math:`w_r = \lg R`.

    ============== ======================= ============================ =======================
    name           width                   input                        output
    ============== ======================= ============================ =======================
    ``x_o``        :math:`w_x`             any (read only)              unchanged
    ``succ_x_o``   1                       any (read only)              unchanged
    ``sigma_1``    1                       any (read only)              unchanged
    ``b``          :math:`w_b`             any (read only)              unchanged
    ``succ_b``     1                       any (read only)              unchanged
    ``sigma_0``    1                       any                          a control (diagonal)
    ``r``          :math:`w_r` (0 if R=1)  any (read only)              unchanged
    ``angles``     :math:`(N - 1) b_{rot}` ``|0>``                      ``|0>``
    ``G0``         1                       ``|0>``                      ``|0>``
    ``G1``         1                       ``|0>``                      ``|0>``
    ``b_neq_B``    1                       ``|0>``                      ``|0>``
    ``sg``         1                       ``|0>``                      ``|0>``
    ``psi_up``     ``N``                   any                          the action above
    ``psi_dn``     ``N``                   any                          the action above
    ``phase_grad`` :math:`b_{rot}`         phase-gradient state         unchanged (catalysed)
    ``work``       :meth:`required_wires`  ``|0>``                      ``|0>``
    ============== ======================= ============================ =======================

    **Compilable arguments.** ``tail_angles`` (``N x (N - 1)``, D1 rows first) and
    ``sf_angles`` (``R x B x (N - 1)``), ints in ``[0, 2**(b_rot - 1))``
    (``WalkTables.tail_angles`` and ``sf_angles``); ``N >= 2``; ``R``, ``B``, ``C >= 1``;
    ``n_plus`` (``0 <= N+ <= N``); ``b_rot >= 2``; ``fixup_bits``, the SF fix-up's one-hot bits
    of ``AngleErase`` (``0 <= fixup_bits <= w_b`` with
    ``2**fixup_bits <= (N - 1) b_rot + 1``; the walk passes ``w_b``).

    **Pre- and postconditions.** The registers are pairwise disjoint (unchecked); the ancillas
    and ``work`` start in ``|0>`` and end there for every measurement outcome of the erase;
    ``phase_grad`` holds the state ``PhaseGradientStatePrep`` prepares. In the walk the index
    registers hold what ``InnerPrepare`` and ``OuterPrepare`` wrote, and the select writes none
    of them (``sigma_0`` gets only a phase), so ``InnerUnprepare``'s preconditions hold
    afterwards.

    **Decomposition** (spec 9.18), each child on its own registers and a prefix of ``work``:
    ``TailAngleLoad(x_o; angles, G0, G1)``; ``SFAngleLoad(r, b, G0; angles, b_neq_B)``, enabled
    on ``G0 = 0``; ``SpinMux(G0, sigma_0, sigma_1; sg)``; ``SpinSwap(sg; psi_up, psi_dn)``;
    ``GivensLadder(psi_dn, angles, phase_grad)``; ``MajoranaGate(succ_x_o, succ_b, G0, G1,
    b_neq_B, flavour=sigma_0, target=psi_dn[0])``; ``GivensLadderDagger(psi_dn, angles,
    phase_grad)``; ``SpinSwap``; ``SpinMux``; ``AngleErase(angles, b_neq_B, G0, G1; x_o, r, b)``
    with ``sf_onehot_bits = fixup_bits``. Between the loads and the erase, ``angles``, ``G0``,
    ``G1`` and ``b_neq_B`` are only controls or the adders' unchanged addend, so the erase's
    precondition holds.

    **Adjoint and control.** Never adjointed or controlled: its rule contains the measured
    ``AngleErase`` (spec 7.4 rules 5 and 6). The adjoint is the class :class:`InnerSelectDagger`.

    **Cost.** :func:`dfthc.costs.inner_select` for ``fixup_bits = w_b``:
    :math:`T = T_{tail} + T_{SF} + 2 + 2N + 4(b_{rot} - 1)(N - 1) + 5 + T_{angle\,erase}`,
    :math:`M = (N - 1) b_{rot} + 3`; ``A = T - (2N + 5)``, the plain ``Toffoli`` gates being the
    two ``SpinMux`` Toffolis, the ``2N`` ``SpinSwap`` CSWAPs and three in the Majorana gate. At
    FeMoco-54: :math:`61 + 291 + 2 + 108 + 2968 + 5 + 102 = 3537`, ``M = 798``, ``A = 3424``. For
    another ``fixup_bits`` the erase's SF term is ``costs.phase_fixup`` over
    ``AngleErase.sf_fixup_high_ranges``. ``CNOT`` and ``PauliX`` counts are upper bounds (the
    loads' reads, the erase's resets).

    **Paper and dfthcbe.** Steps 10 to 18 (spec 3.5): the paper's inner SELECT,
    :math:`U_{Rot}^\dagger\,\mathrm{Maj}\,U_{Rot}` with its spin swap, data loads and erase (PRX
    App. B; arXiv:2502.15882v1 App. B.6). dfthcbe: ``InnerSelect`` (``dfthcbe/select.py``),
    3,502 Toffolis at FeMoco-54; ours is +35: the tail load +5 and the SF load +20 (exact
    iteration), the erase +12 (one-hot over 32 positions, exact iteration), the spin multiplexer
    -2 (one Toffoli per application where dfthcbe has two ``MultiControlX``). The paper's Table
    III gives 3,710 per select with the tail fix-up its Eq. (B50) charges to the inner erase;
    ours is -173: the Givens ladders -212 (``rz-addsub``), the loads +28 and the erase +11
    (exact iteration), the multiplexer +2, the Majorana gate -2 (spec 13.1).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.select import InnerSelect
        from dfthc.params import H2_STO3G_TEST as p

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        args = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
        wires, start = {}, 0
        for name, size in InnerSelect.required_wires(*args).items():
            wires[name], start = list(range(start, start + size)), start + size

        def program():
            qp.PhaseGradientStatePrep(wires["phase_grad"])
            InnerSelect(*args, **wires)

        compiled_resources(program, num_wires=start).toffoli  # 33
    """

    compilable_argnames = ("tail_angles", "sf_angles", "N", "R", "B", "C", "n_plus", "b_rot",
                           "fixup_bits")
    wire_argnames = _WIRES

    def __init__(self, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work):
        wires = (x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg,
                 psi_up, psi_dn, phase_grad, work)
        compilable = _validate("InnerSelect", (tail_angles, sf_angles, N, R, B, C, n_plus,
                                               b_rot, fixup_bits), wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                       fixup_bits) -> dict[str, int]:
        """Exact register sizes; ``work`` is the largest need among the children."""
        return _required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)


def _select_body(dagger, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work) -> None:
    """Steps 10 to 18, with ``qp.adjoint(MajoranaGate(...))`` if ``dagger``. Every register is
    unpacked into a list once; each child gets lists (spec 9.0)."""
    sizes = _child_sizes(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)
    (x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up, psi_dn,
     phase_grad, work) = (list(register) for register in (
         x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
         psi_dn, phase_grad, work))
    TailAngleLoad(tail_angles, n_plus, R * C, N + R * C, b_rot, x_o, angles, G0, G1,
                  work[:sizes["tail"]["work"]])
    SFAngleLoad(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work[:sizes["sf"]["work"]])
    SpinMux(G0, sigma_0, sigma_1, sg)
    SpinSwap(sg, psi_up, psi_dn)
    ladder_work = work[:sizes["ladder"]["work"]]
    GivensLadder(b_rot, psi_dn, angles, phase_grad, ladder_work)
    majorana = (succ_x_o, succ_b, G0, G1, b_neq_B, sigma_0, psi_dn[:1],
                work[:sizes["majorana"]["work"]])
    if dagger:
        qp.adjoint(MajoranaGate(*majorana))
    else:
        MajoranaGate(*majorana)
    GivensLadderDagger(b_rot, psi_dn, angles, phase_grad, ladder_work)
    SpinSwap(sg, psi_up, psi_dn)
    SpinMux(G0, sigma_0, sigma_1, sg)
    AngleErase(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, fixup_bits, angles, b_neq_B, G0,
               G1, x_o, r, b, work[:sizes["erase"]["work"]])


def _select_resources(dagger, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                      fixup_bits) -> dict:
    """The rule's children, declared by hand (a straight-line rule with composite keys; T2 pins
    it): each key has the child's compilable arguments and exactly its register sizes."""
    sizes = _child_sizes(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)
    wires = {child: {name: Wire[n] for name, n in regs.items()} for child, regs in sizes.items()}

    def majorana():
        op = MajoranaGate(**wires["majorana"])
        return qp.adjoint(op) if dagger else op

    return {
        resource_key(lambda: TailAngleLoad(tail_angles, n_plus, R * C, N + R * C, b_rot,
                                           **wires["tail"])): 1,
        resource_key(lambda: SFAngleLoad(sf_angles, R, B, b_rot, **wires["sf"])): 1,
        resource_key(lambda: SpinMux(**wires["mux"])): 2,
        resource_key(lambda: SpinSwap(**wires["swap"])): 2,
        resource_key(lambda: GivensLadder(b_rot, **wires["ladder"])): 1,
        resource_key(majorana): 1,
        resource_key(lambda: GivensLadderDagger(b_rot, **wires["ladder"])): 1,
        resource_key(lambda: AngleErase(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                                        fixup_bits, **wires["erase"])): 1,
    }


def _inner_select_resources(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits,
                            x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1,
                            b_neq_B, sg, psi_up, psi_dn, phase_grad, work):
    return _select_resources(False, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                             fixup_bits)


@qp.register_resources(_inner_select_resources)
def _inner_select(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o, succ_x_o,
                  sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up, psi_dn,
                  phase_grad, work):
    _select_body(False, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work)


register_rule(InnerSelect, _inner_select)


class InnerSelectDagger(Operator2):
    r"""The adjoint of :class:`InnerSelect`: its explicit adjoint class (spec 7.4 rule 6).

    **Action.** :math:`U^\dagger\,\mathrm{Maj}^\dagger\,U` on every branch, with :math:`U` and
    ``Maj`` as in :class:`InnerSelect` for the same arguments: ``InnerSelectDagger`` after
    ``InnerSelect`` is the identity on every input with the ancillas in :math:`|0\rangle`, for
    every measurement outcome of both erases. On the valid branches of the walk it applies the
    adjoint generator pieces: :math:`i\tilde\gamma_0\tilde\gamma_1` (Hermitian) on SF, and with
    ``sigma_0`` projected onto :math:`|+\rangle`, :math:`a_{\tilde u}^\dagger` on D1 and
    :math:`a_{\tilde u}` on Q1; the identity where :class:`InnerSelect` acts as the identity.

    **Registers.** Those of :class:`InnerSelect`, with the same input and output states.

    ============== ======================= ============================ =======================
    name           width                   input                        output
    ============== ======================= ============================ =======================
    ``x_o``        :math:`w_x`             any (read only)              unchanged
    ``succ_x_o``   1                       any (read only)              unchanged
    ``sigma_1``    1                       any (read only)              unchanged
    ``b``          :math:`w_b`             any (read only)              unchanged
    ``succ_b``     1                       any (read only)              unchanged
    ``sigma_0``    1                       any                          a control (diagonal)
    ``r``          :math:`w_r` (0 if R=1)  any (read only)              unchanged
    ``angles``     :math:`(N - 1) b_{rot}` ``|0>``                      ``|0>``
    ``G0``         1                       ``|0>``                      ``|0>``
    ``G1``         1                       ``|0>``                      ``|0>``
    ``b_neq_B``    1                       ``|0>``                      ``|0>``
    ``sg``         1                       ``|0>``                      ``|0>``
    ``psi_up``     ``N``                   any                          the action above
    ``psi_dn``     ``N``                   any                          the action above
    ``phase_grad`` :math:`b_{rot}`         phase-gradient state         unchanged (catalysed)
    ``work``       :meth:`required_wires`  ``|0>``                      ``|0>``
    ============== ======================= ============================ =======================

    **Compilable arguments.** Those of :class:`InnerSelect`, with the same domains.

    **Pre- and postconditions.** Those of :class:`InnerSelect`. In the walk it runs after the
    second ``InnerPrepare``, before ``InnerSigns`` (spec 9.21).

    **Decomposition** (spec 9.18). The sequence of :class:`InnerSelect` with
    ``qp.adjoint(MajoranaGate(...))``, the auto-derived leaf adjoint, in place of the Majorana
    gate: it loads first and erases last (reversing ``InnerSelect`` would run the measured erase
    backwards). Every other child has the same compilable arguments and register sizes as in
    :class:`InnerSelect`, so both selects share those targets.

    **Adjoint and control.** It is the explicit adjoint class of :class:`InnerSelect`; it is
    never adjointed or controlled itself (its rule measures; spec 7.4 rules 5 and 6).

    **Cost.** :func:`dfthc.costs.inner_select`, as for :class:`InnerSelect`: ``T`` and ``M``
    unchanged, ``A = T - (2N + 5)``. At FeMoco-54: ``T = 3537``, ``M = 798``, ``A = 3424``.

    **Paper and dfthcbe.** :math:`\mathrm{SEL}^\dagger` of the second round of steps 5 to 22
    (spec 3.5); dfthcbe: ``InnerSelect(..., is_adjoint=True)``, ``InnerSelect.adjoint()``, the
    same circuit with ``Maj().adjoint()`` (3,502 at FeMoco-54).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.select import InnerSelectDagger
        from dfthc.params import H2_STO3G_TEST as p

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        args = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
        wires, start = {}, 0
        for name, size in InnerSelectDagger.required_wires(*args).items():
            wires[name], start = list(range(start, start + size)), start + size

        def program():
            qp.PhaseGradientStatePrep(wires["phase_grad"])
            InnerSelectDagger(*args, **wires)

        compiled_resources(program, num_wires=start).measurements  # (N - 1) b_rot + 3 = 7
    """

    compilable_argnames = ("tail_angles", "sf_angles", "N", "R", "B", "C", "n_plus", "b_rot",
                           "fixup_bits")
    wire_argnames = _WIRES

    def __init__(self, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work):
        wires = (x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg,
                 psi_up, psi_dn, phase_grad, work)
        compilable = _validate("InnerSelectDagger", (tail_angles, sf_angles, N, R, B, C, n_plus,
                                                     b_rot, fixup_bits), wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                       fixup_bits) -> dict[str, int]:
        """:meth:`InnerSelect.required_wires`."""
        return _required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)


def _inner_select_dagger_resources(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                                   fixup_bits, x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r,
                                   angles, G0, G1, b_neq_B, sg, psi_up, psi_dn, phase_grad,
                                   work):
    return _select_resources(True, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                             fixup_bits)


@qp.register_resources(_inner_select_dagger_resources)
def _inner_select_dagger(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                         succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg,
                         psi_up, psi_dn, phase_grad, work):
    _select_body(True, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work)


register_rule(InnerSelectDagger, _inner_select_dagger)
