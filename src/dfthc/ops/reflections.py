"""Reflections about a basis pattern, exact including the global sign (spec 9.6, 9.19, 9.20)."""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_wires


class ZeroReflection(Operator2):
    r""":math:`2\Pi - I` with :math:`\Pi` the projector onto zeros on ``zero_wires`` and ones on
    ``one_wires``.

    **Action.** :math:`2\Pi - I`, :math:`\Pi = |0\cdots0\rangle\langle 0\cdots0|_{zero} \otimes
    |1\cdots1\rangle\langle 1\cdots1|_{one}`, exactly, including the global sign (spec 6.6).

    **Registers.** With :math:`n_w` = ``len(zero_wires) + len(one_wires)``:

    ============== ================================= ======== =========
    name           width                             input    output
    ============== ================================= ======== =========
    ``zero_wires`` any (pattern value 0)             any      reflected
    ``one_wires``  any (pattern value 1)             any      reflected
    ``work``       at least :math:`\max(n_w - 3, 0)` ``|0>``  ``|0>``
                   (controlled: :math:`\max(n_w - 2,
                   0)`)
    ============== ================================= ======== =========

    **Compilable arguments.** None; the sizes follow from the wires. :math:`n_w \ge 1`
    (``ValueError`` otherwise).

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition.** The last pattern wire is the target. For :math:`n_w \ge 2`: ``X`` on the
    target if its pattern value is 0, ``H`` on the target, ``MultiControlledX`` with the other
    :math:`n_w - 1` wires as controls (their pattern values as control values) and
    :math:`\max(n_w - 3, 0)` zeroed work wires, ``H``, the ``X`` again: that is
    :math:`I - 2\Pi`. Then ``GlobalPhase(pi)``, which multiplies by -1. For :math:`n_w = 1` a
    ``Z`` on the pattern wire (X-conjugated for pattern value 0), because PennyLane rejects a
    one-wire ``MultiControlledX``.

    **Adjoint and control.** ``Adjoint(ZeroReflection)`` is pinned: the operator is
    self-inverse. ``C(ZeroReflection)`` is pinned, for one control wire with control value 1: the
    multi-controlled Z over the control and the pattern, then ``Z`` on the control, which is
    exactly :math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes (2\Pi - I)`; for
    :math:`n_w = 1` a ``CZ``. The controlled rule uses :math:`\max(n_w - 2, 0)` of ``work``
    (``ValueError`` if ``work`` is shorter); generic ``C(GlobalPhase)`` is never used.

    **Cost.** :func:`dfthc.costs.zero_reflection`: :math:`T = \max(n_w - 2, 0)`, controlled
    :math:`n_w - 1` (one ``Toffoli`` and a ``TemporaryAND`` ladder with measured uncompute);
    ``M = 0``.

    **Paper and dfthcbe.** The reflections of steps 23 and 28 (:math:`\mathrm{REF}_{T_2}`,
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}`); dfthcbe: Qualtran's ``Reflection``-based
    ``RefT2`` and ``RefAB``.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import ZeroReflection

        program = lambda: ZeroReflection(zero_wires=[0, 1, 2], one_wires=[3], work=[4])
        compiled_resources(program, num_wires=5).toffoli  # 2
    """

    wire_argnames = ("zero_wires", "one_wires", "work")

    def __init__(self, zero_wires, one_wires, work):
        n_zero, n_one = len(zero_wires), len(one_wires)
        if n_zero + n_one < 1:
            raise ValueError("ZeroReflection needs n_w >= 1 pattern wires, got 0")
        check_wires("ZeroReflection",
                    {"zero_wires": zero_wires, "one_wires": one_wires, "work": work},
                    self.required_wires(n_zero, n_one), at_least=("work",))
        super().__init__(zero_wires, one_wires, work)

    @staticmethod
    def required_wires(n_zero: int, n_one: int, controlled: bool = False) -> dict[str, int]:
        """Register sizes; the controlled version needs one more work wire."""
        n_w = n_zero + n_one
        return {"zero_wires": n_zero, "one_wires": n_one,
                "work": max(n_w - 2, 0) if controlled else max(n_w - 3, 0)}


def _pattern(zero_wires, one_wires):
    """Controls, their values, the target wire and the target's pattern value."""
    wires = list(zero_wires) + list(one_wires)
    values = [0] * len(zero_wires) + [1] * len(one_wires)
    return wires[:-1], values[:-1], wires[-1], values[-1]


def _mcx_key(num_controls, values):
    num_work = max(num_controls - 2, 0)
    return resource_key(lambda: qp.MultiControlledX(
        Wire[num_controls + 1], control_values=list(values),
        work_wires=Wire[num_work] if num_work else None, work_wire_type="zeroed"))


def _mcz(controls, values, target, target_value, work):
    """Multi-controlled Z on ``controls`` (``values``) and ``target`` (``target_value``)."""
    num_work = max(len(controls) - 2, 0)
    if target_value == 0:
        qp.X(target)
    qp.H(target)
    qp.MultiControlledX(wires=list(controls) + [target], control_values=list(values),
                        work_wires=list(work[:num_work]) if num_work else None,
                        work_wire_type="zeroed")
    qp.H(target)
    if target_value == 0:
        qp.X(target)


def _zero_reflection_resources(zero_wires, one_wires, work):
    controls, values, _, last = _pattern(zero_wires, one_wires)
    if not controls:
        return {qp.Z: 1, qp.X: 2 * (last == 0), qp.GlobalPhase: 1}
    return {_mcx_key(len(controls), values): 1, qp.H: 2, qp.X: 2 * (last == 0),
            qp.GlobalPhase: 1}


@qp.register_resources(_zero_reflection_resources)
def _zero_reflection(zero_wires, one_wires, work):
    controls, values, target, last = _pattern(zero_wires, one_wires)
    if controls:
        _mcz(controls, values, target, last, work)
    else:
        if last == 0:
            qp.X(target)
        qp.Z(target)
        if last == 0:
            qp.X(target)
    qp.GlobalPhase(np.pi)


def _adjoint_zero_reflection_resources(base):
    return {resource_key(lambda: ZeroReflection(Wire[len(base.zero_wires)],
                                                Wire[len(base.one_wires)],
                                                Wire[len(base.work)])): 1}


@qp.register_resources(_adjoint_zero_reflection_resources)
def _adjoint_zero_reflection(base):
    ZeroReflection(base.zero_wires, base.one_wires, base.work)


def _c_zero_reflection_resources(base, control_wires, control_values, work_wires,
                                 work_wire_type):
    controls, values, _, last = _pattern(base.zero_wires, base.one_wires)
    if not controls:
        return {qp.CZ: 1, qp.X: 2 * (last == 0), qp.Z: 1}
    return {_mcx_key(len(controls) + 1, [1] + values): 1, qp.H: 2, qp.X: 2 * (last == 0),
            qp.Z: 1}


@qp.register_resources(_c_zero_reflection_resources)
def _c_zero_reflection(base, control_wires, control_values, work_wires, work_wire_type):
    controls, values, target, last = _pattern(base.zero_wires, base.one_wires)
    control = control_wires[0]
    needed = ZeroReflection.required_wires(len(base.zero_wires), len(base.one_wires), True)
    if len(base.work) < needed["work"]:
        raise ValueError(f"C(ZeroReflection) needs {needed['work']} work wires, got "
                         f"{len(base.work)}")
    if controls:
        _mcz([control, *controls], [1, *values], target, last, base.work)
    else:
        if last == 0:
            qp.X(target)
        qp.CZ([control, target])
        if last == 0:
            qp.X(target)
    qp.Z(control)


register_rule(ZeroReflection, _zero_reflection)
register_rule("Adjoint(ZeroReflection)", _adjoint_zero_reflection)
register_rule("C(ZeroReflection)", _c_zero_reflection)
