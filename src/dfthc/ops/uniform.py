"""``FlaggedUniformPrep``: a uniform superposition over ``[0, L)`` and its flag (spec 9.2)."""

from __future__ import annotations

import math

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_int, check_wires
from dfthc.ops.phase_gradient import PhaseGradientRotation
from dfthc.params import lg, odd_part


def amplitude_angles(L: int, s: int) -> tuple[int, int]:
    r"""``(k1, k2)``: the ``s``-bit phase-gradient integers of the two amplification rotations.

    With ``L = 2**eta l`` and ``l`` odd, both rotations use dfthcbe's angle
    :math:`\theta = \arccos(1 - 2^{\lfloor \log_2 l \rfloor}/l)`, loaded as
    :math:`k = \lfloor \theta/(2\pi)\, 2^s \rfloor \bmod 2^s` (Qualtran's
    ``QFxp(s, s).to_fixed_width_int`` truncates; the same ``k`` as
    :func:`dfthc.data.encoded.uniform_failure_probability`). So ``k1 == k2``; the pair keeps the
    two rotation steps of spec 9.2 apart. ``(0, 0)`` if ``l = 1``.
    """
    odd = odd_part(L)
    if odd == 1:
        return 0, 0
    angle = math.acos(1 - 2 ** math.floor(math.log2(odd)) / odd)
    k = math.floor(angle / (2 * math.pi) * 2**s) % 2**s
    return k, k


class FlaggedUniformPrep(Operator2):
    r"""Prepare a uniform superposition over ``[0, L)`` and flag success (dfthcbe's
    ``UniformViaPhaseGradient``).

    **Action.** From :math:`|0\rangle_t|0\rangle_{succ}`, prepare a state whose ``succ = 1``
    part is exactly uniform over :math:`t \in [0, L)` and whose ``succ = 0`` part is supported on
    :math:`t \ge L`. The weight of the ``succ = 0`` part is
    :func:`dfthc.data.encoded.uniform_failure_probability` ``(L, s)``. The ``succ = 1``
    amplitudes share one complex phase (not 1 unless ``l = 1``) and have weight
    ``(1 - P_fail) / L`` each; the ``succ = 0`` amplitudes are equal on every ``t >= L``. Only
    these weights enter the walk, because the unprepare is the exact adjoint.

    **Registers.** Write :math:`L = 2^\eta l` with ``l`` odd and :math:`n = \lg l`.

    ============== ============================ ====================== =========================
    name           width                        input                  output
    ============== ============================ ====================== =========================
    ``target``     :math:`\lg L`                ``|0>``                prepared
    ``succ``       1                            ``|0>``                ``[t < L]``
    ``succ_work``  :math:`\max(n - 1, 0)`       ``|0>``                dirty until the adjoint
    ``phase_grad`` at least ``s``               phase-gradient state   unchanged (catalysed)
    ``work``       ``n + 2s - 1`` (0 if l = 1)  ``|0>``                ``|0>``
    ============== ============================ ====================== =========================

    **Compilable arguments.** ``L`` (int, ``L >= 1``) and ``s`` (int, ``s >= 3``): the bits of
    the two rotations, which add into the top ``s`` wires of ``phase_grad``.

    **Pre- and postconditions.** As in the table; ``succ_work`` holds the success comparator's
    intermediate ANDs from the prepare until its adjoint uncomputes them.

    **Decomposition.** If ``l = 1``: ``H`` on every ``target`` wire and ``X(succ)``. Otherwise,
    with ``top = target[:n]`` (spec 9.2, the order of dfthcbe's construction):

    1. ``H`` on every ``target`` wire;
    2. ``LeftClassicalComparator(top, l, "<")`` into ``work[0]``, its work wires ``work[1:n]``;
    3. ``PhaseGradientRotation(k1, s)`` on ``work[0]``, scratch ``work[n:n+s]``, work
       ``work[n+s:]``;
    4. the comparator's adjoint (measured uncompute);
    5. ``H`` on ``top``;
    6. a ``TemporaryAND`` ladder with control values 0 computing ``[top == 0]`` into
       ``work[n-2]``, on ``work[0:n-1]``;
    7. ``PhaseGradientRotation(k2, s)`` on ``work[n-2]``;
    8. the ladder's measured uncompute;
    9. ``H`` on ``top``;
    10. ``LeftClassicalComparator(top, l, "<")`` into ``succ``, its work wires ``succ_work``.

    ``(k1, k2) = amplitude_angles(L, s)``. The low :math:`\eta` qubits keep their Hadamards.

    **Adjoint and control.** The adjoint is auto-derived: the body is unrolled in Python, and the
    comparator adjoints compile (K-7). The adjoint uncomputes ``succ_work``. Never controlled.

    **Cost.** :func:`dfthc.costs.flagged_uniform_prep` and
    :func:`dfthc.costs.flagged_uniform_unprep`: for ``l > 1`` the prepare costs
    :math:`T = 3n + 2s - 5` (two comparators, two rotations, one ladder) and the adjoint
    :math:`T = 2n + 2s - 4` (one comparator, two rotations, one ladder); both 0 if ``l = 1``;
    ``M = 0``. At FeMoco-54: :math:`X_o = 324` gives 30 / 24, ``B + 1 = 28`` gives 18 / 16.

    **Paper and dfthcbe.** Steps 1 and 5 of the circuit (adjoint: 22 and 27); the uniform
    preparation of Babbush et al. (PRX 8, 041015, Fig. 12) with phase-gradient rotations.
    dfthcbe: ``UniformViaPhaseGradient`` (``dfthcbe/prepare.py``).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.uniform import FlaggedUniformPrep

        def program():
            qp.PhaseGradientStatePrep(wires=[4, 5, 6])
            FlaggedUniformPrep(5, 3, target=[0, 1, 2], succ=[3], succ_work=[7, 8],
                               phase_grad=[4, 5, 6], work=list(range(9, 17)))

        compiled_resources(program, num_wires=17).toffoli  # 3 * 3 + 2 * 3 - 5 = 10
    """

    compilable_argnames = ("L", "s")
    wire_argnames = ("target", "succ", "succ_work", "phase_grad", "work")

    def __init__(self, L, s, target, succ, succ_work, phase_grad, work):
        L = check_int("L", L, low=1)
        s = check_int("s", s, low=3)
        check_wires("FlaggedUniformPrep",
                    {"target": target, "succ": succ, "succ_work": succ_work,
                     "phase_grad": phase_grad, "work": work},
                    self.required_wires(L, s), at_least=("phase_grad",))
        super().__init__(L, s, target, succ, succ_work, phase_grad, work)

    @staticmethod
    def required_wires(L: int, s: int) -> dict[str, int]:
        """Register sizes; ``phase_grad`` is a minimum, every other size is exact."""
        n = lg(odd_part(L))
        return {"target": lg(L), "succ": 1, "succ_work": max(n - 1, 0), "phase_grad": s,
                "work": n + 2 * s - 1 if n else 0}


def _comparator(top, odd, target, work):
    return qp.LeftClassicalComparator(top, odd, target, work, "<")


def _rotation(k, s, target, phase_grad, work, n):
    PhaseGradientRotation(k, s, [target], phase_grad[:s], work[n:n + s], work[n + s:n + 2 * s - 1])


def _flagged_uniform_prep_resources(L, s, target, succ, succ_work, phase_grad, work):
    odd, n = odd_part(L), lg(odd_part(L))
    if n == 0:
        return {qp.H: lg(L), qp.X: 1}
    k, _ = amplitude_angles(L, s)
    comparator = resource_key(
        lambda: qp.LeftClassicalComparator(Wire[n], odd, Wire[1], Wire[n - 1], "<"))
    return {
        qp.H: lg(L) + 2 * n,
        comparator: 2,
        resource_key(lambda: qp.adjoint(
            qp.LeftClassicalComparator(Wire[n], odd, Wire[1], Wire[n - 1], "<"))): 1,
        resource_key(lambda: PhaseGradientRotation(
            k, s, Wire[1], Wire[s], Wire[s], Wire[s - 1])): 2,
        resource_key(lambda: qp.TemporaryAND(Wire[3])): n - 1,
        resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))): n - 1,
    }


@qp.register_resources(_flagged_uniform_prep_resources)
def _flagged_uniform_prep(L, s, target, succ, succ_work, phase_grad, work):
    odd, n = odd_part(L), lg(odd_part(L))
    for wire in target:
        qp.H(wire)
    if n == 0:
        qp.X(succ[0])
        return
    k1, k2 = amplitude_angles(L, s)
    top = target[:n]
    _comparator(top, odd, work[0:1], work[1:n])
    _rotation(k1, s, work[0], phase_grad, work, n)
    qp.adjoint(_comparator(top, odd, work[0:1], work[1:n]))
    for wire in top:
        qp.H(wire)
    ladder = [(top[0], top[1], work[0], (0, 0))]
    ladder += [(work[j - 2], top[j], work[j - 1], (1, 0)) for j in range(2, n)]
    for a, b, out, values in ladder:
        qp.TemporaryAND([a, b, out], control_values=values)
    _rotation(k2, s, work[n - 2], phase_grad, work, n)
    for a, b, out, values in reversed(ladder):
        qp.adjoint(qp.TemporaryAND([a, b, out], control_values=values))
    for wire in top:
        qp.H(wire)
    _comparator(top, odd, succ, succ_work)


register_rule(FlaggedUniformPrep, _flagged_uniform_prep)
