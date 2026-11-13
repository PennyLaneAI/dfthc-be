"""``MajoranaGate``: the generator's one-qubit Majorana action after the Givens ladder (spec 9.16).

After the ladder the generator acts on ``psi_dn[0]`` alone, where the Jordan-Wigner string is
empty, so ``gamma_0 = X`` and ``gamma_1 = Y`` there (spec 6.2). SF branches apply
``i gamma_0 gamma_1 = -Z`` (dfthcbe's sign, not Fig. 4's ``+Z``); D1 and Q1 branches apply
``a = (X + iY)/2`` and ``a^dag = (X - iY)/2`` through the flavour ancilla ``sigma_0``, which arrives
in ``|+>`` from the inner prepare and is projected back onto ``|+>`` by the inner unprepare.
"""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_wires

_REGISTERS = ("succ_x_o", "succ_b", "G0", "G1", "b_neq_B", "flavour", "target", "work")


class MajoranaGate(Operator2):
    r"""The Majorana gate of step 15: ``-Z`` on SF branches, ``a`` and ``a^dag`` on D1 and Q1.

    **Action.** With :math:`a = \mathrm{succ}_{x_o} \wedge G_0` (a valid one-body branch) and
    :math:`s = \mathrm{succ}_{x_o} \wedge \neg G_0 \wedge \mathrm{succ}_b` (a valid SF branch),
    in this order (spec 9.16):

    1. if :math:`s \wedge [b \ne B]`: :math:`-Z` on ``target`` (a phase -1 on ``target = 0``);
    2. if :math:`a`: ``X`` on ``target``;
    3. if :math:`a \wedge` ``flavour``: ``Z`` on ``target``;
    4. if :math:`a \wedge G_1`: ``Z`` on ``flavour``.

    Every other register is unchanged. On ``target = psi_dn[0]``, :math:`\gamma_0 = X` and
    :math:`\gamma_1 = Y` (spec 6.2), so step 1 is :math:`i\gamma_0\gamma_1 = -Z`, the SF pair
    (dfthcbe's sign; the paper's Fig. 4 applies :math:`+Z`, which would flip every :math:`w_b`
    against :math:`w_B`). **Flavour convention:** ``flavour`` is ``sigma_0``, which arrives in
    :math:`|+\rangle` on D1/Q1 branches; with it projected onto :math:`|+\rangle` before and
    after, steps 2 to 4 give :math:`(X + iY)/2 = a` on D1 (:math:`G_1 = 0`) and
    :math:`(X - iY)/2 = a^\dagger` on Q1 (:math:`G_1 = 1`). On SF branches ``flavour`` is the spin
    sum and is untouched. D1/Q1 ignores ``succ_b``; failed :math:`x_o` and failed SF :math:`b`
    branches get the identity.

    **Registers.**

    ============ ======= ================================ ==============================
    name         width   input                            output
    ============ ======= ================================ ==============================
    ``succ_x_o`` 1       any                              unchanged
    ``succ_b``   1       any                              unchanged
    ``G0``       1       any (0: SF, 1: one-body)         unchanged
    ``G1``       1       any (0: D1, 1: Q1)               unchanged
    ``b_neq_B``  1       any (:math:`[b \ne B]` on SF)    unchanged
    ``flavour``  1       any (``sigma_0``)                ``Z`` applied by step 4
    ``target``   1       any (``psi_dn[0]``)              the Majorana action
    ``work``     2       ``|0>``                          ``|0>``
    ============ ======= ================================ ==============================

    **Compilable arguments.** None; every register has a fixed width.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (dfthcbe's ``Maj``). ``work[0] = succ_x_o AND G0`` (``TemporaryAND``);
    ``CNOT(work[0], succ_x_o)``, which leaves :math:`\mathrm{succ}_{x_o} \wedge \neg G_0`;
    ``work[1] = succ_x_o AND succ_b`` (``TemporaryAND``), so ``work`` holds :math:`a` and
    :math:`s`. Step 1 is ``CCZ(work[1], target, b_neq_B)`` with ``target`` as an open control
    (``X``-conjugated); step 2 ``CNOT(work[0], target)``; step 3 ``CCZ(work[0], flavour,
    target)``; step 4 ``CCZ(work[0], G1, flavour)``; each ``CCZ`` is a ``Toffoli`` conjugated by
    ``H`` on its last wire. Then the measured uncompute of ``work[1]``, the ``CNOT`` again and
    the measured uncompute of ``work[0]``.

    **Adjoint and control.** The adjoint is auto-derived (the body has no loops or
    conditionals): it is the :math:`\mathrm{Maj}^\dagger` of ``InnerSelectDagger`` (spec 9.18),
    with ``a`` and ``a^dag`` exchanged on the D1 and Q1 branches. Never controlled.

    **Cost.** :func:`dfthc.costs.majorana_gate`: :math:`T = 5`, two ``TemporaryAND`` and three
    ``Toffoli`` (``A = 2``); ``M = 0``. The adjoint costs the same.

    **Paper and dfthcbe.** Step 15; the Majorana gate of the paper's Fig. 4 (PRX), with the
    success flags combined with :math:`G_0` into two ANDs (7 Toffolis there, 5 here) and the
    SF sign of dfthcbe. dfthcbe: ``Maj`` (``dfthcbe/majorana.py``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.majorana import MajoranaGate

        program = lambda: MajoranaGate([0], [1], [2], [3], [4], [5], [6], work=[7, 8])
        compiled_resources(program, num_wires=9).toffoli  # 5
    """

    wire_argnames = _REGISTERS

    def __init__(self, succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
        check_wires("MajoranaGate",
                    dict(zip(_REGISTERS, (succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target,
                                          work), strict=True)),
                    self.required_wires())
        super().__init__(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work)

    @staticmethod
    def required_wires() -> dict[str, int]:
        """Exact register sizes."""
        return {name: 1 for name in _REGISTERS[:-1]} | {"work": 2}


def _ccz(a, b, target) -> None:
    qp.H(target)
    qp.Toffoli([a, b, target])
    qp.H(target)


def _majorana_gate_resources(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
    return {resource_key(lambda: qp.TemporaryAND(Wire[3])): 2,
            resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))): 2,
            qp.Toffoli: 3, qp.CNOT: 3, qp.H: 6, qp.X: 2}


@qp.register_resources(_majorana_gate_resources)
def _majorana_gate(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
    succ, a, s, t = succ_x_o[0], work[0], work[1], target[0]
    qp.TemporaryAND([succ, G0[0], a])
    qp.CNOT([a, succ])
    qp.TemporaryAND([succ, succ_b[0], s])
    qp.X(t)
    _ccz(s, t, b_neq_B[0])
    qp.X(t)
    qp.CNOT([a, t])
    _ccz(a, flavour[0], t)
    _ccz(a, G1[0], flavour[0])
    qp.adjoint(qp.TemporaryAND([succ, succ_b[0], s]))
    qp.CNOT([a, succ])
    qp.adjoint(qp.TemporaryAND([succ, G0[0], a]))


register_rule(MajoranaGate, _majorana_gate)
