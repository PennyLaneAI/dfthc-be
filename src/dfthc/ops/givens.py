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

"""Givens rotations by phase-gradient addition and the Givens ladders (spec 9.14 and 9.15).

**Rung.** On the mode qubits ``(q_k, q_{k+1})``, big-endian (spec 6.7), a rung fixes ``|00>``
and ``|11>`` and rotates ``{|01>, |10>}`` by ``[[cos t, -sin t], [sin t, cos t]]`` with
``t = 4 pi x / 2**b_rot``, where ``x < 2**(b_rot - 1)`` is the integer in the rung's
``(b_rot - 1)``-wire angle register (spec 6.3).
It is dfthcbe's ``GivensRotationByPhaseGradient``: a Clifford basis change around two ``Rz(t)``,
each an addition of ``x`` into the phase-gradient register or a subtraction, as the qubit says.

**Ladder.** ``GivensLadder`` applies ``U = G_0 G_1 ... G_{N-2}``: rung ``N - 2`` first in time
and rung 0 last, which realizes ``U^dag gamma_{0,f} U = sum_j u_j gamma_{j,f}`` for
``u = vector_from_angles(x, b_rot)``; :func:`dfthc.data.angles.ladder_matrix` is its dense matrix.
The opposite order fails for ``N >= 3``. ``GivensLadderDagger`` runs the inverse rungs in the
opposite time order. Nothing adjoints ``GivensLadder`` itself (spec 7.4 rule 6).
"""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_int, check_wires


class GivensRotation(Operator2):
    r"""One rung of the Givens ladder: a real rotation of ``{|01>, |10>}`` by the loaded angle.

    **Action.** On ``(q_i, q_j)``, with :math:`\theta = 4\pi x/2^{b_{rot}}` and ``x`` the
    big-endian integer in ``angle``, so :math:`x < 2^{b_{rot} - 1}`: :math:`|00\rangle` and
    :math:`|11\rangle` are fixed, :math:`|01\rangle \mapsto \cos\theta|01\rangle +
    \sin\theta|10\rangle` and :math:`|10\rangle \mapsto -\sin\theta|01\rangle +
    \cos\theta|10\rangle`, exactly, with no global phase (spec 6.3). ``angle`` is unchanged;
    ``phase_grad`` is returned unchanged.

    **Registers.** With :math:`b_{rot}` = ``len(phase_grad)``:

    ============== ========================= ====================== ====================
    name           width                     input                  output
    ============== ========================= ====================== ====================
    ``q_i``        1                         any                    rotated
    ``q_j``        1                         any                    rotated
    ``angle``      :math:`b_{rot} - 1`       any (``x``)            unchanged
    ``phase_grad`` :math:`b_{rot}`           phase-gradient state   unchanged (catalysed)
    ``work``       :math:`b_{rot} - 1`       ``|0>``                ``|0>``
    ============== ========================= ====================== ====================

    **Compilable arguments.** None; the sizes follow from the wires. :math:`b_{rot} \ge 2`.

    **Pre- and postconditions.** ``phase_grad`` holds the state that ``PhaseGradientStatePrep``
    prepares on :math:`b_{rot}` wires; ``work`` starts and ends in ``|0>``. ``angle`` has no
    top bit: the stored angles have ``x < 2**(b_rot - 1)`` (spec 6.3).

    **Decomposition** (dfthcbe's ``GivensRotationByPhaseGradient``; spec 9.14), in time order,
    with :math:`\sqrt{X} = HSH` and :math:`\sqrt{X}^\dagger = HS^\dagger H`:

    1. :math:`\sqrt{X}^\dagger` on ``q_i``; ``S``, then :math:`\sqrt{X}^\dagger`, on ``q_j``;
    2. ``CNOT(q_i, q_j)``, then ``H`` on ``q_i``;
    3. :math:`R_z(\theta)` on ``q_i``, then on ``q_j``;
    4. ``H`` on ``q_i``, then ``CNOT(q_i, q_j)``;
    5. :math:`\sqrt{X}` on ``q_i``; :math:`\sqrt{X}`, then :math:`S^\dagger`, on ``q_j``.

    Each :math:`R_z(\theta)` on a qubit ``q`` is Qualtran's ``ControlledAddOrSubtract``:
    ``X(q)``, ``CNOT`` from ``q`` onto every ``phase_grad`` wire, ``SemiAdder(angle, phase_grad,
    work)``, the same CNOTs, ``X(q)``; the adder aligns ``x`` at the least significant bit of
    ``phase_grad``. It maps the gradient's value ``g`` to ``g + x`` for
    ``q = 1`` and to ``g - x`` for ``q = 0``, so ``q = 1`` picks up
    :math:`e^{+2\pi i x/2^{b_{rot}}}` and ``q = 0`` the conjugate (spec 6.5): exactly
    :math:`R_z(\theta)`.

    **Adjoint and control.** The adjoint is auto-derived (:math:`\theta \to -\theta`; the body
    has no loops or conditionals, and the adjoint ``SemiAdder`` subtracts, K-7):
    :class:`GivensLadderDagger` uses it. Never controlled.

    **Cost.** :func:`dfthc.costs.givens_rotation`: :math:`T = 2(b_{rot} - 1)`, two
    ``SemiAdder``\ s into the :math:`b_{rot}`-bit gradient (their cost depends only on its
    width), all ``TemporaryAND`` with measured uncomputes
    (``A = T``); ``M = 0``. The adjoint costs the same. At FeMoco-54: 28, as dfthcbe's; the
    paper counts :math:`b_{rot}` Toffolis per :math:`R_z`, so 30 per rung (PRX Table III;
    ``rz-addsub`` in spec 13.1).

    **Paper and dfthcbe.** One rung of the paper's :math:`U_{Rot}` (PRX Eq. (B20)), steps 14 and
    16; dfthcbe's ``GivensRotationByPhaseGradient`` (``dfthcbe/givens.py``), whose two
    :math:`R_z` are Qualtran's ``RzViaPhaseGradient`` (user decision Q-5).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.givens import GivensRotation

        def program():
            qp.PhaseGradientStatePrep(wires=[4, 5, 6])
            GivensRotation(q_i=[0], q_j=[1], angle=[2, 3], phase_grad=[4, 5, 6], work=[7, 8])

        compiled_resources(program, num_wires=9).toffoli  # 2 (3 - 1) = 4
    """

    wire_argnames = ("q_i", "q_j", "angle", "phase_grad", "work")

    def __init__(self, q_i, q_j, angle, phase_grad, work):
        b_rot = check_int("b_rot = len(phase_grad)", len(phase_grad), low=2)
        check_wires("GivensRotation",
                    {"q_i": q_i, "q_j": q_j, "angle": angle, "phase_grad": phase_grad,
                     "work": work},
                    self.required_wires(b_rot))
        super().__init__(q_i, q_j, angle, phase_grad, work)

    @staticmethod
    def required_wires(b_rot: int) -> dict[str, int]:
        """Exact register sizes."""
        return {"q_i": 1, "q_j": 1, "angle": b_rot - 1, "phase_grad": b_rot, "work": b_rot - 1}


def _sqrt_x(wire, adjoint: bool) -> None:
    qp.H(wire)
    if adjoint:
        qp.adjoint(qp.S(wire))
    else:
        qp.S(wire)
    qp.H(wire)


def _rz(qubit, angle, phase_grad, work) -> None:
    qp.X(qubit)
    for wire in phase_grad:
        qp.CNOT([qubit, wire])
    qp.SemiAdder(angle, phase_grad, work)
    for wire in phase_grad:
        qp.CNOT([qubit, wire])
    qp.X(qubit)


def _givens_rotation_resources(q_i, q_j, angle, phase_grad, work):
    b_rot = len(phase_grad)
    return {
        qp.H: 10, qp.S: 3, resource_key(lambda: qp.adjoint(qp.S(Wire[1]))): 3,
        qp.CNOT: 4 * b_rot + 2, qp.X: 4,
        resource_key(lambda: qp.SemiAdder(Wire[b_rot - 1], Wire[b_rot], Wire[b_rot - 1])): 2,
    }


@qp.register_resources(_givens_rotation_resources)
def _givens_rotation(q_i, q_j, angle, phase_grad, work):
    i, j = q_i[0], q_j[0]
    _sqrt_x(i, adjoint=True)
    qp.S(j)
    _sqrt_x(j, adjoint=True)
    qp.CNOT([i, j])
    qp.H(i)
    _rz(i, angle, phase_grad, work)
    _rz(j, angle, phase_grad, work)
    qp.H(i)
    qp.CNOT([i, j])
    _sqrt_x(i, adjoint=False)
    _sqrt_x(j, adjoint=False)
    qp.adjoint(qp.S(j))


register_rule(GivensRotation, _givens_rotation)


def _ladder_sizes(op: str, b_rot, system) -> tuple[int, int]:
    b_rot = check_int("b_rot", b_rot, low=2)
    num_modes = len(system)
    if num_modes < 2:
        raise ValueError(f"{op}: system needs N >= 2 mode wires, got {num_modes}")
    return b_rot, num_modes


def _ladder_required_wires(b_rot: int, num_modes: int) -> dict[str, int]:
    return {"system": num_modes, "angles": (num_modes - 1) * (b_rot - 1), "phase_grad": b_rot,
            "work": b_rot - 1}


def _rung_key(b_rot: int, adjoint: bool):
    def build():
        rung = GivensRotation(Wire[1], Wire[1], Wire[b_rot - 1], Wire[b_rot], Wire[b_rot - 1])
        return qp.adjoint(rung) if adjoint else rung

    return resource_key(build)


def _rungs(b_rot, system, angles, phase_grad, work, order, adjoint: bool) -> None:
    """The rungs ``k`` of ``order`` on ``(system[k], system[k+1])`` with angle register ``k``.
    Registers are unpacked into lists once, so each rung is passed lists of wires (spec 9.0)."""
    system, angles, width = list(system), list(angles), b_rot - 1
    phase_grad, work = list(phase_grad), list(work)
    for k in order:
        wires = ([system[k]], [system[k + 1]], angles[k * width:(k + 1) * width], phase_grad,
                 work)
        if adjoint:
            qp.adjoint(GivensRotation(*wires))
        else:
            GivensRotation(*wires)


class GivensLadder(Operator2):
    r"""The Givens ladder :math:`U(u)` on one spin register: rung ``N - 2`` first, rung 0 last.

    **Action.** :math:`U = G_0 G_1 \cdots G_{N-2}` on ``system``, with :math:`G_k` the
    :class:`GivensRotation` on modes ``(q_k, q_{k+1})`` driven by angle register ``k``,
    ``angles[k (b_rot - 1):(k + 1)(b_rot - 1)]`` (spec 6.7). Then
    :math:`U^\dagger \gamma_{0,f} U = \sum_j u_j \gamma_{j,f}` for
    ``u = vector_from_angles(x, b_rot)`` (spec 6.3), and :math:`U` equals
    :func:`dfthc.data.angles.ladder_matrix` ``(x, b_rot, N)``. ``angles`` is unchanged.

    **Registers.** With ``N = len(system)``:

    ============== ========================= ====================== ====================
    name           width                     input                  output
    ============== ========================= ====================== ====================
    ``system``     ``N`` (at least 2)        any                    :math:`U` applied
    ``angles``     :math:`(N-1)(b_{rot}-1)`  any                    unchanged
    ``phase_grad`` :math:`b_{rot}`           phase-gradient state   unchanged (catalysed)
    ``work``       :math:`b_{rot} - 1`       ``|0>``                ``|0>``
    ============== ========================= ====================== ====================

    **Compilable arguments.** ``b_rot`` (int, at least 2): the width of ``phase_grad``; one angle
    register has :math:`b_{rot} - 1` wires. ``N`` follows from ``system``.

    **Pre- and postconditions.** As for :class:`GivensRotation`.

    **Decomposition.** The :class:`GivensRotation` rungs in time order ``k = N - 2, ..., 0``,
    unrolled in Python, all sharing ``phase_grad`` and ``work``: the order of dfthcbe's
    ``GivensRotationLadder``, which realizes ``angles_from_vector``. The opposite time order
    gives a different unitary for ``N >= 3``.

    **Adjoint and control.** Never adjointed or controlled. The inverse is the explicit class
    :class:`GivensLadderDagger` (spec 7.4 rule 6); nothing applies ``qp.adjoint`` to this
    operator.

    **Cost.** :func:`dfthc.costs.givens_ladder`: :math:`T = 2(b_{rot} - 1)(N - 1)`, ``A = T``,
    ``M = 0``. At FeMoco-54: 1,484, as dfthcbe's; the paper's Table III gives 1,590
    (:math:`2 b_{rot}` per rung, ``rz-addsub``).

    **Paper and dfthcbe.** The paper's :math:`U_{Rot}` (PRX Eq. (B20)), step 14; dfthcbe's
    ``GivensRotationLadder`` (``dfthcbe/rotations.py``).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.givens import GivensLadder

        def program():
            qp.PhaseGradientStatePrep(wires=[7, 8, 9])
            GivensLadder(3, system=[0, 1, 2], angles=[3, 4, 5, 6], phase_grad=[7, 8, 9],
                         work=[10, 11])

        compiled_resources(program, num_wires=12).toffoli  # 2 (3 - 1)(3 - 1) = 8
    """

    compilable_argnames = ("b_rot",)
    wire_argnames = ("system", "angles", "phase_grad", "work")

    def __init__(self, b_rot, system, angles, phase_grad, work):
        b_rot, num_modes = _ladder_sizes("GivensLadder", b_rot, system)
        check_wires("GivensLadder",
                    {"system": system, "angles": angles, "phase_grad": phase_grad,
                     "work": work},
                    self.required_wires(b_rot, num_modes))
        super().__init__(b_rot, system, angles, phase_grad, work)

    @staticmethod
    def required_wires(b_rot: int, N: int) -> dict[str, int]:
        """Exact register sizes for ``N`` modes."""
        return _ladder_required_wires(b_rot, N)


def _givens_ladder_resources(b_rot, system, angles, phase_grad, work):
    return {_rung_key(b_rot, adjoint=False): len(system) - 1}


@qp.register_resources(_givens_ladder_resources)
def _givens_ladder(b_rot, system, angles, phase_grad, work):
    _rungs(b_rot, system, angles, phase_grad, work, reversed(range(len(system) - 1)),
           adjoint=False)


register_rule(GivensLadder, _givens_ladder)


class GivensLadderDagger(Operator2):
    r"""The inverse of :class:`GivensLadder`: :math:`U^\dagger` on the same registers.

    **Action.** :math:`U^\dagger = G_{N-2}^\dagger \cdots G_1^\dagger G_0^\dagger` on
    ``system``, with :math:`U` the :class:`GivensLadder` of the same arguments: applied after
    the ladder it gives the identity on every input. ``angles`` is unchanged.

    **Registers.** Those of :class:`GivensLadder`:

    ============== ========================= ====================== ====================
    name           width                     input                  output
    ============== ========================= ====================== ====================
    ``system``     ``N`` (at least 2)        any                    :math:`U^\dagger`
                                                                    applied
    ``angles``     :math:`(N-1)(b_{rot}-1)`  any                    unchanged
    ``phase_grad`` :math:`b_{rot}`           phase-gradient state   unchanged (catalysed)
    ``work``       :math:`b_{rot} - 1`       ``|0>``                ``|0>``
    ============== ========================= ====================== ====================

    **Compilable arguments.** ``b_rot``, as for :class:`GivensLadder`.

    **Pre- and postconditions.** As for :class:`GivensLadder`.

    **Decomposition.** The rungs in time order ``k = 0, ..., N - 2`` (the ladder's order
    reversed), each ``qp.adjoint(GivensRotation(...))``, the auto-derived leaf adjoint
    (:math:`\theta \to -\theta`), unrolled in Python (spec 9.15).

    **Adjoint and control.** It is the explicit adjoint class of :class:`GivensLadder` (spec 7.4
    rule 6); it is never adjointed or controlled itself.

    **Cost.** :func:`dfthc.costs.givens_ladder`, the same as the ladder:
    :math:`T = 2(b_{rot} - 1)(N - 1)`, ``A = T``, ``M = 0``. At FeMoco-54: 1,484.

    **Paper and dfthcbe.** :math:`U_{Rot}^\dagger`, step 16; dfthcbe's
    ``GivensRotationLadder(...).adjoint()``.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.givens import GivensLadderDagger

        def program():
            qp.PhaseGradientStatePrep(wires=[7, 8, 9])
            GivensLadderDagger(3, system=[0, 1, 2], angles=[3, 4, 5, 6], phase_grad=[7, 8, 9],
                               work=[10, 11])

        compiled_resources(program, num_wires=12).toffoli  # 8
    """

    compilable_argnames = ("b_rot",)
    wire_argnames = ("system", "angles", "phase_grad", "work")

    def __init__(self, b_rot, system, angles, phase_grad, work):
        b_rot, num_modes = _ladder_sizes("GivensLadderDagger", b_rot, system)
        check_wires("GivensLadderDagger",
                    {"system": system, "angles": angles, "phase_grad": phase_grad,
                     "work": work},
                    self.required_wires(b_rot, num_modes))
        super().__init__(b_rot, system, angles, phase_grad, work)

    @staticmethod
    def required_wires(b_rot: int, N: int) -> dict[str, int]:
        """Exact register sizes for ``N`` modes (those of :class:`GivensLadder`)."""
        return _ladder_required_wires(b_rot, N)


def _givens_ladder_dagger_resources(b_rot, system, angles, phase_grad, work):
    return {_rung_key(b_rot, adjoint=True): len(system) - 1}


@qp.register_resources(_givens_ladder_dagger_resources)
def _givens_ladder_dagger(b_rot, system, angles, phase_grad, work):
    _rungs(b_rot, system, angles, phase_grad, work, range(len(system) - 1), adjoint=True)


register_rule(GivensLadderDagger, _givens_ladder_dagger)
