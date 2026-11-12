"""``PhaseGradientRotation``: a Z rotation by phase-gradient addition (spec section 9.1)."""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_int, check_wires


class PhaseGradientRotation(Operator2):
    r"""Multiply ``|t>`` of one qubit by :math:`e^{2\pi i k t/2^m}`: a phase-gradient addition.

    **Action.** :math:`|t\rangle \mapsto e^{2\pi i k t/2^m}|t\rangle` on ``target``; every other
    register is returned unchanged. So ``k = 2**m - 1`` gives :math:`e^{-2\pi i/2^m}` on
    :math:`|1\rangle`.

    **Registers.**

    ============== ================ ========================== ===========================
    name           width            input                      output
    ============== ================ ========================== ===========================
    ``target``     1                any                        phased by ``k t``
    ``phase_grad`` at least ``m``   phase-gradient state       unchanged (catalysed)
    ``scratch``    ``m``            ``|0>``                    ``|0>``
    ``work``       ``m - 1``        ``|0>``                    ``|0>``
    ============== ================ ========================== ===========================

    ``phase_grad`` is the walk's register (:math:`b_{grad}` wires) or ``qpe_grad``; the rotation
    adds into its first (most significant) ``m`` wires, which hold an ``m``-bit phase gradient.

    **Compilable arguments.** ``k`` (int, ``0 <= k < 2**m``) and ``m`` (int, ``m >= 1``).

    **Pre- and postconditions.** ``phase_grad`` holds the state that ``PhaseGradientStatePrep``
    prepares; ``scratch`` and ``work`` start and end in ``|0>``.

    **Decomposition.** CNOTs from ``target`` onto the ``scratch`` bits where ``k`` has ones
    (``scratch`` is big-endian, so it then holds ``k t``), ``SemiAdder(scratch, phase_grad[:m],
    work)``, and the same CNOTs again. Adding ``k t`` into the top ``m`` qubits multiplies the
    phase gradient by :math:`e^{+2\pi i k t/2^m}` (spec 6.5), and the phase kicks back onto
    ``target``. For ``k = 0`` the rule emits nothing.

    **Adjoint and control.** The adjoint is auto-derived: the body has no loops or conditionals,
    and the adjoint ``SemiAdder`` subtracts (K-7). Never controlled.

    **Cost.** :func:`dfthc.costs.phase_gradient_rotation`: :math:`T = (m - 1)[k \ne 0]`, one
    ``m``-bit ``SemiAdder``; ``M = 0``.

    **Paper and dfthcbe.** The phase-gradient rotation of Sanders et al. (PRX Quantum 1, 020312);
    in the paper it realizes the rotations of steps 1, 5, 22 and 27 inside the uniform
    preparations. dfthcbe uses Qualtran's ``ZPowConstViaPhaseGradient``.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.phase_gradient import PhaseGradientRotation

        def program():
            qp.PhaseGradientStatePrep(wires=[1, 2, 3])
            PhaseGradientRotation(5, 3, target=[0], phase_grad=[1, 2, 3], scratch=[4, 5, 6],
                                  work=[7, 8])

        compiled_resources(program, num_wires=9).toffoli  # 2
    """

    compilable_argnames = ("k", "m")
    wire_argnames = ("target", "phase_grad", "scratch", "work")

    def __init__(self, k, m, target, phase_grad, scratch, work):
        m = check_int("m", m, low=1)
        k = check_int("k", k, low=0, high=2**m - 1)
        check_wires("PhaseGradientRotation",
                    {"target": target, "phase_grad": phase_grad, "scratch": scratch,
                     "work": work},
                    self.required_wires(k, m), at_least=("phase_grad",))
        super().__init__(k, m, target, phase_grad, scratch, work)

    @staticmethod
    def required_wires(k: int, m: int) -> dict[str, int]:
        """Register sizes; ``phase_grad`` is a minimum, every other size is exact."""
        return {"target": 1, "phase_grad": m, "scratch": m, "work": m - 1}


def _fan_out(k, m, target, scratch):
    for j in range(m):
        if (k >> (m - 1 - j)) & 1:
            qp.CNOT([target[0], scratch[j]])


def _phase_gradient_rotation_resources(k, m, target, phase_grad, scratch, work):
    if k == 0:
        return {}
    adder = resource_key(lambda: qp.SemiAdder(Wire[m], Wire[m], Wire[m - 1] if m > 1 else None))
    return {qp.CNOT: 2 * bin(k).count("1"), adder: 1}


@qp.register_resources(_phase_gradient_rotation_resources)
def _phase_gradient_rotation(k, m, target, phase_grad, scratch, work):
    if k == 0:
        return
    _fan_out(k, m, target, scratch)
    qp.SemiAdder(scratch, phase_grad[:m], work if m > 1 else None)
    _fan_out(k, m, target, scratch)


register_rule(PhaseGradientRotation, _phase_gradient_rotation)
