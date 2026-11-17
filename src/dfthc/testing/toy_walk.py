r"""The toy qubitization walk of :math:`H = (Z + X)/2` for phase-estimation tests (spec 11.4).

One system qubit and one LCU ancilla: PREP is a Hadamard on the ancilla, SELECT is a zero-controlled
``Z`` plus a one-controlled ``X`` on the system, and the walk is
:math:`W = \mathrm{REF}_{anc}\,\mathrm{PREP}^\dagger\,\mathrm{SEL}\,\mathrm{PREP}` with
:math:`\mathrm{REF}_{anc} = 2|0\rangle\langle 0| - I = Z`. Its block on the ancilla's
:math:`|0\rangle` is :math:`H` (:math:`\lambda = 1`), so its eigenphases are
:math:`\pm\arccos(\pm 1/\sqrt 2) = \pm\pi/4, \pm 3\pi/4`, exact 3-bit phases
:math:`\varphi \in \{1/8, 3/8, 5/8, 7/8\}`.

The walk has the DFTHC walk's form :math:`\mathrm{REF}\,U^\dagger R\,U` with :math:`U` = PREP and
:math:`R` = SELECT, itself a reflection (a Hermitian unitary). So the ``C(ToyWalk)`` rule
controls exactly its two reflections, as ``C(DFTHCWalk)`` controls ``ref_ab`` and ``ref_t2``:
:math:`C(W) = C(\mathrm{REF}_{anc})\,\mathrm{PREP}^\dagger\,C(\mathrm{SEL})\,\mathrm{PREP}`, and
with the control in :math:`|0\rangle` the rest is :math:`\mathrm{PREP}^\dagger\mathrm{PREP} = I`.
Controlling the ancilla reflection alone would leave :math:`\mathrm{PREP}^\dagger\,\mathrm{SEL}\,
\mathrm{PREP} \ne I` on the control's :math:`|0\rangle` (the toy has one reflection outside
:math:`U`, the DFTHC walk two). :class:`ToyControlledWalkPower` repeats the controlled walk with a
``qp.for_loop``, as ``ControlledWalkPower`` repeats ``C(DFTHCWalk)``; it is registered as a
walk power of :class:`dfthc.qpe.QPERound`.

The dense references (:func:`toy_walk_matrix`, :func:`controlled_toy_walk_matrix`) are built from
the definition above, never from the circuit. The costs (:func:`toy_walk_cost`) live here because
the toy is test scaffolding, not a spec 9.24 block.
"""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire
from pennylane.wires import Wires

from dfthc.compile import register_rule, resource_key
from dfthc.costs import Cost
from dfthc.ops._validate import check_int, check_wires
from dfthc.qpe import WalkPowerSpec, iterative_phase_estimation, register_walk_power

_I2 = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_P0, _P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)


def toy_walk_matrix() -> np.ndarray:
    r"""The 4 x 4 matrix of :math:`W` on (``system``, ``ancilla``), ``system`` the more
    significant qubit, from spec 11.4's definition."""
    prep = np.kron(_I2, _H)
    select = np.kron(_Z, _P0) + np.kron(_X, _P1)
    reflection = np.kron(_I2, 2 * _P0 - _I2)
    return reflection @ prep.conj().T @ select @ prep


def controlled_toy_walk_matrix() -> np.ndarray:
    r""":math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes W` on (``control``,
    ``system``, ``ancilla``)."""
    return np.kron(_P0, np.eye(4)) + np.kron(_P1, toy_walk_matrix())


def toy_walk_cost(controlled: bool) -> Cost:
    """``ToyWalk``: no Toffoli; ``C(ToyWalk)``: the two Toffolis of the controlled SELECT.
    No measurement either way."""
    return Cost(2 if controlled else 0)


def toy_walk_power_cost(power: int) -> Cost:
    """``ToyControlledWalkPower(power)``: ``power`` controlled toy walks."""
    return power * toy_walk_cost(True)


def _check(name: str, wires: dict) -> None:
    check_wires(name, wires, {wire: 1 for wire in wires})


class ToyWalk(Operator2):
    r"""The toy qubitization walk :math:`W = \mathrm{REF}_{anc}\,\mathrm{PREP}^\dagger\,
    \mathrm{SEL}\,\mathrm{PREP}` of :math:`H = (Z + X)/2` (spec 11.4).

    **Action.** :func:`toy_walk_matrix` on (``system``, ``ancilla``): eigenphases
    :math:`\pm\pi/4` (the eigenvalue :math:`+1/\sqrt 2` of :math:`H`) and :math:`\pm 3\pi/4`
    (:math:`-1/\sqrt 2`); its block on the ancilla's :math:`|0\rangle` is :math:`H`.

    **Registers.**

    ============ ===== ===== ==============
    name         width input output
    ============ ===== ===== ==============
    ``system``   1     any   see the action
    ``ancilla``  1     any   see the action
    ============ ===== ===== ==============

    **Compilable arguments.** None.

    **Pre- and postconditions.** None: it is unitary on every input.

    **Decomposition.** PREP: ``H(ancilla)``. SELECT: ``X(ancilla)``, ``CZ(ancilla, system)``,
    ``X(ancilla)`` (the zero-controlled ``Z``), ``CNOT(ancilla, system)`` (the one-controlled
    ``X``). PREP†: ``H(ancilla)``. :math:`\mathrm{REF}_{anc}`: ``Z(ancilla)``, exactly
    :math:`2|0\rangle\langle 0| - I`.

    **Adjoint and control.** Never adjointed. Controlled with ``qp.ctrl(ToyWalk(...), control,
    control_values=...)``, whose one pinned rule, registered for ``"C(ToyWalk)"``, controls only
    the toy's two reflections, :math:`C(W) = C(\mathrm{REF}_{anc})\,\mathrm{PREP}^\dagger\,
    C(\mathrm{SEL})\,\mathrm{PREP}`: PREP; the controlled SELECT, two ``Toffoli`` s
    (``X(ancilla)``, ``H(system)``, ``Toffoli(control, ancilla, system)``, ``H(system)``,
    ``X(ancilla)``: ``Z`` on the system where the control is 1 and the ancilla 0; then
    ``Toffoli(control, ancilla, system)``: ``X`` where both are 1); PREP†; ``CZ(control,
    ancilla)``. Its action is :func:`controlled_toy_walk_matrix` on (``control``, ``system``,
    ``ancilla``) for control value 1, global phase included; a control value 0 puts ``X`` on the
    control before and after, conditioned on the value, which may be traced. The rule takes exactly
    one control wire (``ValueError`` otherwise) and no work wires. :class:`ToyControlledWalkPower`
    repeats it.

    **Cost.** :func:`toy_walk_cost`: uncontrolled ``T = 0``, controlled ``T = 2`` (two
    ``Toffoli``, ``A = 0``); ``M = 0``.

    **Paper and dfthcbe.** The qubitization walk of Low and Chuang (Quantum 3, 163 (2019)) for a
    two-term LCU; test scaffolding of spec 11.4 with no counterpart in the paper or dfthcbe.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.testing.toy_walk import ToyWalk

        compiled_resources(lambda: ToyWalk(system=[0], ancilla=[1]), num_wires=2).toffoli  # 0
    """

    wire_argnames = ("system", "ancilla")

    def __init__(self, system, ancilla):
        _check("ToyWalk", {"system": system, "ancilla": ancilla})
        super().__init__(system, ancilla)

    @staticmethod
    def required_wires() -> dict[str, int]:
        return {"system": 1, "ancilla": 1}


def _toy_walk_resources(system, ancilla):
    return {qp.H: 2, qp.X: 2, qp.CZ: 1, qp.CNOT: 1, qp.Z: 1}


@qp.register_resources(_toy_walk_resources)
def _toy_walk(system, ancilla):
    qp.H(ancilla)
    qp.X(ancilla)
    qp.CZ([ancilla[0], system[0]])
    qp.X(ancilla)
    qp.CNOT([ancilla[0], system[0]])
    qp.H(ancilla)
    qp.Z(ancilla)


register_rule(ToyWalk, _toy_walk)


def _controlled_toy_walk_resources(base, control_wires, control_values, work_wires,
                                   work_wire_type):
    return {qp.H: 4, qp.X: 4, qp.Toffoli: 2, qp.CZ: 1}


@qp.register_resources(_controlled_toy_walk_resources)
def _controlled_toy_walk(base, control_wires, control_values, work_wires, work_wire_type):
    """``work_wires`` and ``work_wire_type`` are unused: the construction needs no work wire."""
    if len(control_wires) != 1:
        raise ValueError(f"C(ToyWalk) takes one control wire; got {len(control_wires)}")
    control, system, ancilla = control_wires[0], base.system[0], base.ancilla[0]
    flip = qp.cond(qp.math.logical_not(control_values[0]), qp.X)
    flip(control)
    qp.H(ancilla)
    qp.X(ancilla)
    qp.H(system)
    qp.Toffoli([control, ancilla, system])
    qp.H(system)
    qp.X(ancilla)
    qp.Toffoli([control, ancilla, system])
    qp.H(ancilla)
    qp.CZ([control, ancilla])
    flip(control)


register_rule("C(ToyWalk)", _controlled_toy_walk)


class ToyControlledWalkPower(Operator2):
    r"""The controlled toy walk to a power, :math:`C(W^p)`: ``power`` steps of
    ``qp.ctrl(ToyWalk(...), control)`` (the toy twin of ``dfthc.qpe.ControlledWalkPower``, spec
    11.4).

    **Action.** :math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes W^p` on
    (``control``, ``system``, ``ancilla``).

    **Registers.** A one-wire ``control``, then those of :class:`ToyWalk`.

    ============ ===== ===== ==============
    name         width input output
    ============ ===== ===== ==============
    ``control``  1     any   a control
    ``system``   1     any   see the action
    ``ancilla``  1     any   see the action
    ============ ===== ===== ==============

    **Compilable arguments.** ``power`` (int, at least 1).

    **Pre- and postconditions.** None: it is unitary on every input.

    **Decomposition.** One ``qp.for_loop`` with the static bound ``power`` whose body is
    ``qp.ctrl(ToyWalk(system, ancilla), control)``, decomposed by the pinned ``C(ToyWalk)``
    rule.

    **Adjoint and control.** Never adjointed or controlled (spec 7.4 rules 4 and 6); registered
    in :data:`dfthc.qpe.WALK_POWERS` for :class:`dfthc.qpe.QPERound`, with no scratch group
    (``scratch_argname = None``), so the round's corrections take spare wires.

    **Cost.** :func:`toy_walk_power_cost`: ``T = 2 power``, ``M = 0``.

    **Paper and dfthcbe.** Test scaffolding of spec 11.4.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.testing.toy_walk import ToyControlledWalkPower

        program = lambda: ToyControlledWalkPower(4, control=[0], system=[1], ancilla=[2])
        compiled_resources(program, num_wires=3).toffoli  # 8
    """

    compilable_argnames = ("power",)
    wire_argnames = ("control", "system", "ancilla")
    scratch_argname = None

    def __init__(self, power, control, system, ancilla):
        power = check_int("power", power, low=1)
        _check("ToyControlledWalkPower", {"control": control, "system": system,
                                          "ancilla": ancilla})
        super().__init__(power, control, system, ancilla)

    @staticmethod
    def required_wires(power) -> dict[str, int]:
        return {"control": 1, "system": 1, "ancilla": 1}

    @staticmethod
    def validate_compilable(power) -> None:
        """The constructor's check of ``power`` (``ValueError``); ``dfthc.qpe.QPERound`` calls
        it."""
        check_int("power", power, low=1)


def _toy_controlled_walk_power_resources(power, control, system, ancilla):
    step = resource_key(lambda: qp.ctrl(ToyWalk(Wire[1], Wire[1]), control=Wire[1]))
    return {step: power}


@qp.register_resources(_toy_controlled_walk_power_resources)
def _toy_controlled_walk_power(power, control, system, ancilla):
    control, system, ancilla = list(control), list(system), list(ancilla)

    @qp.for_loop(power)
    def step(_):
        qp.ctrl(ToyWalk(system, ancilla), control=control)

    step()


register_rule(ToyControlledWalkPower, _toy_controlled_walk_power)
register_walk_power(ToyControlledWalkPower)


def toy_walk_power_spec(system, ancilla) -> WalkPowerSpec:
    """The :class:`dfthc.qpe.WalkPowerSpec` of :class:`ToyControlledWalkPower` on ``system``
    and ``ancilla`` (no compilable arguments after ``power``)."""
    return WalkPowerSpec("ToyControlledWalkPower", (), (1, 1), Wires.all_wires([system, ancilla]))


def toy_qpe_wires(num_bits: int) -> dict[str, list[int]]:
    """The wires of :func:`toy_qpe`: ``control`` 0, ``qpe_grad`` (``num_bits``), the corrections'
    ``spare_wires`` (``2 num_bits - 1``: the toy walk lends no scratch group), then ``system`` and
    ``ancilla``; ``3 num_bits + 2`` wires in all."""
    t = num_bits
    return {"control": [0], "qpe_grad": list(range(1, t + 1)),
            "spare_wires": list(range(t + 1, 3 * t)), "system": [3 * t], "ancilla": [3 * t + 1]}


def toy_qpe(num_bits: int) -> list:
    """Iterative phase estimation of the toy walk with ``num_bits`` bits on
    :func:`toy_qpe_wires` (system and ancilla start in the caller's state, ``|0>`` by default):
    the measured bits, as :func:`dfthc.qpe.iterative_phase_estimation` returns them."""
    wires = toy_qpe_wires(num_bits)
    return iterative_phase_estimation(
        lambda: toy_walk_power_spec(wires["system"], wires["ancilla"]), wires["control"],
        num_bits, wires["qpe_grad"], wires["spare_wires"])
