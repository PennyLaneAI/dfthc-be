"""Reflections about a basis pattern, exact including the global sign (spec 9.6, 9.19, 9.20)."""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_int, check_wires


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
    self-inverse. ``C(ZeroReflection)`` is pinned, for one control wire with control value 1 only:
    the rule raises ``ValueError`` for more control wires, and for control value 0 where the value
    is concrete (tape mode); under capture ``control_values`` is a dynamic argument of PennyLane's
    controlled operator, which the rule cannot read, so control value 0 is unchecked there and
    outside the domain. The rule applies the
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
    # Under capture control_values is a dynamic argument of ControlledOp2: a tracer, unreadable.
    concrete = not qp.math.is_tracer(control_values)
    if len(control_wires) != 1 or (concrete and not bool(control_values[0])):
        values = f" with values {[bool(v) for v in control_values]}" if concrete else ""
        raise ValueError("C(ZeroReflection) supports exactly one control wire with control value "
                         f"1; got {len(control_wires)} control wires{values}")
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


# --------------------------------------------------------------------------- RefT2 and RefAB


def _widths(op: str, **registers) -> tuple[int, ...]:
    """The lengths of the named pattern registers, each at least 1 (``ValueError`` otherwise)."""
    return tuple(check_int(f"{op}: len({name})", len(register), low=1)
                 for name, register in registers.items())


def _reflection_key(n_zero: int, n_one: int):
    """The ``ZeroReflection`` that ``RefT2`` and ``RefAB`` emit, with exactly its work."""
    work = ZeroReflection.required_wires(n_zero, n_one)["work"]
    return resource_key(lambda: ZeroReflection(Wire[n_zero], Wire[n_one], Wire[work]))


def _controlled_reflection(control, zero_wires, one_wires, work) -> None:
    r""":math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes (2\Pi - I)` for at
    least two pattern wires: the multi-controlled Z over the control and the pattern, then ``Z``
    on the control, as the ``C(ZeroReflection)`` rule builds it (spec 9.6)."""
    controls, values, target, last = _pattern(zero_wires, one_wires)
    _mcz([control, *controls], [1, *values], target, last, work)
    qp.Z(control)


def _controlled_reflection_resources(n_zero: int, n_one: int) -> dict:
    controls, values, _, last = _pattern(range(n_zero), range(n_zero, n_zero + n_one))
    return {_mcx_key(len(controls) + 1, [1, *values]): 1, qp.H: 2, qp.X: 2 * (last == 0),
            qp.Z: 1}


class RefT2(Operator2):
    r""":math:`\mathrm{REF}_{T_2} = 2\Pi_{T_2} - I` (step 23), exact including the global sign.

    **Action.** :math:`\Pi_{T_2} = |0\rangle\langle 0|_{b, m_b, \sigma_0} \otimes
    |1\rangle\langle 1|_{\mathrm{succ}_{x_o}}` (spec 3.3): the phase +1 on the one basis state
    with ``b = 0``, ``m_b = 0``, ``sigma_0 = 0`` and ``succ_x_o = 1``, and -1 on every other, so
    the operator is :math:`-I` wherever the uniform preparation of ``x_o`` failed
    (``succ_x_o = 0``; spec 8.5). Every register is unchanged up to that phase.

    **Registers.** :math:`n_w = w_b + b_{k2} + 2` pattern wires.

    ============ ============================ ======== =========
    name         width                        input    output
    ============ ============================ ======== =========
    ``succ_x_o`` 1                            any      phased
    ``b``        :math:`w_b` (at least 1)     any      phased
    ``m_b``      :math:`b_{k2}` (at least 1)  any      phased
    ``sigma_0``  1                            any      phased
    ``work``     :math:`n_w - 3`              ``|0>``  ``|0>``
    ============ ============================ ======== =========

    **Compilable arguments.** None; :math:`w_b` and :math:`b_{k2}` follow from the wires.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.19). ``ZeroReflection(zero_wires=b ‖ m_b ‖ sigma_0,
    one_wires=succ_x_o, work)``: a ``MultiControlledX`` on ``succ_x_o`` with the other
    :math:`n_w - 1` pattern wires as zero-valued controls, ``H``-conjugated, then
    ``GlobalPhase(pi)``; :math:`n_w = w_b + b_{k2} + 2`.

    **Adjoint and control.** ``Adjoint(RefT2)`` is pinned to the operator itself (it is
    self-inverse; spec 7.5). The controlled form is the class :class:`ControlledRefT2`; nothing
    applies ``qp.ctrl`` to this operator (spec 7.4 rule 6).

    **Cost.** :func:`dfthc.costs.ref_t2` (uncontrolled): :math:`T = n_w - 2 = w_b + b_{k2}`, one
    ``Toffoli`` and :math:`w_b + b_{k2} - 1` ``TemporaryAND`` with measured uncomputes
    (``A = T - 1``); ``M = 0``. At FeMoco-54: 20, as dfthcbe's; the paper's Table III counts 21.

    **Paper and dfthcbe.** Step 23, :math:`\mathrm{REF}_{T_2}` of spec 3.3 (PRX App. B; dfthcbe
    cites the paper's (B11) for its multi-controlled Z). dfthcbe: ``DFTHCBlockEncoding.ref_t2``,
    a Qualtran ``ReflectionUsingPrepare.reflection_around_zero`` on ``b``, ``m_b``, ``sigma_0``
    controlled by ``succ_x_o``, with ``GlobalPhase(exponent=1)`` (``dfthcbe/block_encoding.py``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import RefT2

        program = lambda: RefT2(succ_x_o=[0], b=[1, 2], m_b=[3, 4], sigma_0=[5], work=[6, 7, 8])
        compiled_resources(program, num_wires=9).toffoli  # w_b + b_k2 = 4
    """

    wire_argnames = ("succ_x_o", "b", "m_b", "sigma_0", "work")

    def __init__(self, succ_x_o, b, m_b, sigma_0, work):
        w_b, b_k2 = _widths("RefT2", b=b, m_b=m_b)
        check_wires("RefT2", {"succ_x_o": succ_x_o, "b": b, "m_b": m_b, "sigma_0": sigma_0,
                              "work": work}, self.required_wires(w_b, b_k2))
        super().__init__(succ_x_o, b, m_b, sigma_0, work)

    @staticmethod
    def required_wires(w_b: int, b_k2: int) -> dict[str, int]:
        """Exact register sizes; ``work`` is the reflection's :math:`n_w - 3`."""
        return {"succ_x_o": 1, "b": w_b, "m_b": b_k2, "sigma_0": 1,
                "work": ZeroReflection.required_wires(w_b + b_k2 + 1, 1)["work"]}


def _ref_t2_resources(succ_x_o, b, m_b, sigma_0, work):
    return {_reflection_key(len(b) + len(m_b) + 1, 1): 1}


@qp.register_resources(_ref_t2_resources)
def _ref_t2(succ_x_o, b, m_b, sigma_0, work):
    ZeroReflection([*b, *m_b, *sigma_0], [*succ_x_o], work)


def _adjoint_ref_t2_resources(base):
    sizes = RefT2.required_wires(len(base.b), len(base.m_b))
    return {resource_key(lambda: RefT2(**{name: Wire[n] for name, n in sizes.items()})): 1}


@qp.register_resources(_adjoint_ref_t2_resources)
def _adjoint_ref_t2(base):
    RefT2(base.succ_x_o, base.b, base.m_b, base.sigma_0, base.work)


register_rule(RefT2, _ref_t2)
register_rule("Adjoint(RefT2)", _adjoint_ref_t2)


class ControlledRefT2(Operator2):
    r"""The controlled :math:`\mathrm{REF}_{T_2}` of the controlled block encoding: the explicit
    controlled class of :class:`RefT2` (spec 7.4 rule 6).

    **Action.** Exactly :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle
    1|_{control} \otimes (2\Pi_{T_2} - I)`, with :math:`\Pi_{T_2}` as in :class:`RefT2`: the
    identity where ``control = 0`` and :class:`RefT2`, global sign included, where
    ``control = 1``. The sign matters here: control turns :class:`RefT2`'s global -1 into a
    relative phase (spec 3.3).

    **Registers.** :math:`n_w = w_b + b_{k2} + 2` pattern wires.

    ============ ============================ ======== =========
    name         width                        input    output
    ============ ============================ ======== =========
    ``control``  1                            any      phased
    ``succ_x_o`` 1                            any      phased
    ``b``        :math:`w_b` (at least 1)     any      phased
    ``m_b``      :math:`b_{k2}` (at least 1)  any      phased
    ``sigma_0``  1                            any      phased
    ``work``     :math:`n_w - 2`              ``|0>``  ``|0>``
    ============ ============================ ======== =========

    **Compilable arguments.** None; the sizes follow from the wires.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.19). The ``MultiControlledX`` on ``succ_x_o`` with ``control``
    (value 1) and ``b ‖ m_b ‖ sigma_0`` (value 0) as controls, ``H``-conjugated, which is
    :math:`I - 2|1\rangle\langle 1|_{control} \otimes \Pi_{T_2}`; then ``Z`` on ``control``. It is
    the construction of the ``C(ZeroReflection)`` rule (spec 9.6), emitted directly with that
    rule's helper, so no ``C(...)`` key is requested.

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0).

    **Cost.** :func:`dfthc.costs.ref_t2` (controlled): :math:`T = n_w - 1 = w_b + b_{k2} + 1`,
    one ``Toffoli`` and :math:`w_b + b_{k2}` ``TemporaryAND`` (``A = T - 1``); ``M = 0``. At
    FeMoco-54: 21, as dfthcbe's.

    **Paper and dfthcbe.** The controlled walk's controlled :math:`\mathrm{REF}_{T_2}` (spec
    3.3); dfthcbe: ``DFTHCBlockEncoding.ref_t2`` of ``is_controlled=True``, the reflection's
    ``.controlled()`` with a ``ZGate`` on ``ctrl`` (``dfthcbe/block_encoding.py``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import ControlledRefT2

        program = lambda: ControlledRefT2(control=[0], succ_x_o=[1], b=[2, 3], m_b=[4, 5],
                                          sigma_0=[6], work=[7, 8, 9, 10])
        compiled_resources(program, num_wires=11).toffoli  # w_b + b_k2 + 1 = 5
    """

    wire_argnames = ("control", "succ_x_o", "b", "m_b", "sigma_0", "work")

    def __init__(self, control, succ_x_o, b, m_b, sigma_0, work):
        w_b, b_k2 = _widths("ControlledRefT2", b=b, m_b=m_b)
        check_wires("ControlledRefT2", {"control": control, "succ_x_o": succ_x_o, "b": b,
                                        "m_b": m_b, "sigma_0": sigma_0, "work": work},
                    self.required_wires(w_b, b_k2))
        super().__init__(control, succ_x_o, b, m_b, sigma_0, work)

    @staticmethod
    def required_wires(w_b: int, b_k2: int) -> dict[str, int]:
        """Exact register sizes; ``work`` is the controlled reflection's :math:`n_w - 2`."""
        return {"control": 1, "succ_x_o": 1, "b": w_b, "m_b": b_k2, "sigma_0": 1,
                "work": ZeroReflection.required_wires(w_b + b_k2 + 1, 1, controlled=True)["work"]}


def _controlled_ref_t2_resources(control, succ_x_o, b, m_b, sigma_0, work):
    return _controlled_reflection_resources(len(b) + len(m_b) + 1, 1)


@qp.register_resources(_controlled_ref_t2_resources)
def _controlled_ref_t2(control, succ_x_o, b, m_b, sigma_0, work):
    _controlled_reflection(control[0], [*b, *m_b, *sigma_0], [*succ_x_o], work)


register_rule(ControlledRefT2, _controlled_ref_t2)


class RefAB(Operator2):
    r""":math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}} = 2|0\rangle\langle 0| - I` on the outer
    and inner index registers (step 28), exact including the global sign.

    **Action.** The phase +1 on the one basis state with ``x_o``, ``m_x_o``, ``sigma_1``, ``b``,
    ``m_b`` and ``sigma_0`` all zero, and -1 on every other (spec 3.3:
    :math:`\mathcal{A} = (x_o, m_{x_o}, \sigma_1)`, :math:`\mathcal{B} = (b, m_b, \sigma_0)`).
    Every register is unchanged up to that phase.

    **Registers.** :math:`n_w = w_x + b_{k1} + w_b + b_{k2} + 2` pattern wires.

    =========== ============================ ======== =========
    name        width                        input    output
    =========== ============================ ======== =========
    ``x_o``     :math:`w_x` (at least 1)     any      phased
    ``m_x_o``   :math:`b_{k1}` (at least 1)  any      phased
    ``sigma_1`` 1                            any      phased
    ``b``       :math:`w_b` (at least 1)     any      phased
    ``m_b``     :math:`b_{k2}` (at least 1)  any      phased
    ``sigma_0`` 1                            any      phased
    ``work``    :math:`n_w - 3`              ``|0>``  ``|0>``
    =========== ============================ ======== =========

    **Compilable arguments.** None; the sizes follow from the wires.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.20). ``ZeroReflection(zero_wires=x_o ‖ m_x_o ‖ sigma_1 ‖ b ‖ m_b ‖
    sigma_0, one_wires=[], work)``: a ``MultiControlledX`` on ``sigma_0`` with the other
    :math:`n_w - 1` wires as zero-valued controls, conjugated by ``X`` and ``H``, then
    ``GlobalPhase(pi)``; :math:`n_w = w_x + b_{k1} + w_b + b_{k2} + 2`.

    **Adjoint and control.** ``Adjoint(RefAB)`` is pinned to the operator itself (spec 7.5). The
    controlled form is the class :class:`ControlledRefAB`; nothing applies ``qp.ctrl`` to this
    operator (spec 7.4 rule 6).

    **Cost.** :func:`dfthc.costs.ref_ab` (uncontrolled):
    :math:`T = n_w - 2 = w_x + b_{k1} + w_b + b_{k2}`, one ``Toffoli`` (``A = T - 1``);
    ``M = 0``. At FeMoco-54: 44; dfthcbe's costs 45 and the paper's Table III 46 (spec 13.1,
    reason ``constants``).

    **Paper and dfthcbe.** Step 28, :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}`, which makes
    :math:`W` a qubitization walk (spec 3.3); dfthcbe: ``DFTHCWalk.ref_ab``, Qualtran's
    ``ReflectionUsingPrepare.reflection_around_zero`` on the six registers
    (``dfthcbe/block_encoding.py``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import RefAB

        program = lambda: RefAB(x_o=[0, 1], m_x_o=[2, 3], sigma_1=[4], b=[5, 6], m_b=[7, 8],
                                sigma_0=[9], work=list(range(10, 17)))
        compiled_resources(program, num_wires=17).toffoli  # w_x + b_k1 + w_b + b_k2 = 8
    """

    wire_argnames = ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0", "work")

    def __init__(self, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
        widths = _widths("RefAB", x_o=x_o, m_x_o=m_x_o, b=b, m_b=m_b)
        check_wires("RefAB", {"x_o": x_o, "m_x_o": m_x_o, "sigma_1": sigma_1, "b": b,
                              "m_b": m_b, "sigma_0": sigma_0, "work": work},
                    self.required_wires(*widths))
        super().__init__(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work)

    @staticmethod
    def required_wires(w_x: int, b_k1: int, w_b: int, b_k2: int) -> dict[str, int]:
        """Exact register sizes; ``work`` is the reflection's :math:`n_w - 3`."""
        return {"x_o": w_x, "m_x_o": b_k1, "sigma_1": 1, "b": w_b, "m_b": b_k2, "sigma_0": 1,
                "work": ZeroReflection.required_wires(w_x + b_k1 + w_b + b_k2 + 2, 0)["work"]}


def _ref_ab_resources(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
    return {_reflection_key(len(x_o) + len(m_x_o) + len(b) + len(m_b) + 2, 0): 1}


@qp.register_resources(_ref_ab_resources)
def _ref_ab(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
    ZeroReflection([*x_o, *m_x_o, *sigma_1, *b, *m_b, *sigma_0], [], work)


def _adjoint_ref_ab_resources(base):
    sizes = RefAB.required_wires(len(base.x_o), len(base.m_x_o), len(base.b), len(base.m_b))
    return {resource_key(lambda: RefAB(**{name: Wire[n] for name, n in sizes.items()})): 1}


@qp.register_resources(_adjoint_ref_ab_resources)
def _adjoint_ref_ab(base):
    RefAB(base.x_o, base.m_x_o, base.sigma_1, base.b, base.m_b, base.sigma_0, base.work)


register_rule(RefAB, _ref_ab)
register_rule("Adjoint(RefAB)", _adjoint_ref_ab)


class ControlledRefAB(Operator2):
    r"""The controlled :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` of the controlled walk:
    the explicit controlled class of :class:`RefAB` (spec 7.4 rule 6).

    **Action.** Exactly :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle
    1|_{control} \otimes (2|0\rangle\langle 0|_{\mathcal{A}\cup\mathcal{B}} - I)`, global sign
    included (spec 3.3).

    **Registers.** :math:`n_w = w_x + b_{k1} + w_b + b_{k2} + 2` pattern wires.

    =========== ============================ ======== =========
    name        width                        input    output
    =========== ============================ ======== =========
    ``control`` 1                            any      phased
    ``x_o``     :math:`w_x` (at least 1)     any      phased
    ``m_x_o``   :math:`b_{k1}` (at least 1)  any      phased
    ``sigma_1`` 1                            any      phased
    ``b``       :math:`w_b` (at least 1)     any      phased
    ``m_b``     :math:`b_{k2}` (at least 1)  any      phased
    ``sigma_0`` 1                            any      phased
    ``work``    :math:`n_w - 2`              ``|0>``  ``|0>``
    =========== ============================ ======== =========

    **Compilable arguments.** None; the sizes follow from the wires.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.20). The ``MultiControlledX`` on ``sigma_0`` with ``control``
    (value 1) and the other pattern wires (value 0) as controls, conjugated by ``X`` and ``H``,
    then ``Z`` on ``control``: the ``C(ZeroReflection)`` construction (spec 9.6), emitted with
    that rule's helper, so no ``C(...)`` key is requested.

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0).

    **Cost.** :func:`dfthc.costs.ref_ab` (controlled):
    :math:`T = n_w - 1 = w_x + b_{k1} + w_b + b_{k2} + 1` (``A = T - 1``); ``M = 0``. At
    FeMoco-54: 45; dfthcbe's controlled reflection also costs 45. Its work, 44 wires at
    FeMoco-54, is the walk's largest need, the size of ``WalkLayout.work`` (spec 10.3).

    **Paper and dfthcbe.** The controlled walk's controlled
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` (spec 3.3); dfthcbe: ``DFTHCWalk.ref_ab``
    of ``is_controlled=True`` (``control_val=1``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import ControlledRefAB

        program = lambda: ControlledRefAB(control=[0], x_o=[1, 2], m_x_o=[3, 4], sigma_1=[5],
                                          b=[6, 7], m_b=[8, 9], sigma_0=[10],
                                          work=list(range(11, 19)))
        compiled_resources(program, num_wires=19).toffoli  # w_x + b_k1 + w_b + b_k2 + 1 = 9
    """

    wire_argnames = ("control", "x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0", "work")

    def __init__(self, control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
        widths = _widths("ControlledRefAB", x_o=x_o, m_x_o=m_x_o, b=b, m_b=m_b)
        check_wires("ControlledRefAB", {"control": control, "x_o": x_o, "m_x_o": m_x_o,
                                        "sigma_1": sigma_1, "b": b, "m_b": m_b,
                                        "sigma_0": sigma_0, "work": work},
                    self.required_wires(*widths))
        super().__init__(control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work)

    @staticmethod
    def required_wires(w_x: int, b_k1: int, w_b: int, b_k2: int) -> dict[str, int]:
        """Exact register sizes; ``work`` is the controlled reflection's :math:`n_w - 2`."""
        return {"control": 1, "x_o": w_x, "m_x_o": b_k1, "sigma_1": 1, "b": w_b, "m_b": b_k2,
                "sigma_0": 1,
                "work": ZeroReflection.required_wires(w_x + b_k1 + w_b + b_k2 + 2, 0,
                                                      controlled=True)["work"]}


def _controlled_ref_ab_resources(control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
    return _controlled_reflection_resources(len(x_o) + len(m_x_o) + len(b) + len(m_b) + 2, 0)


@qp.register_resources(_controlled_ref_ab_resources)
def _controlled_ref_ab(control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
    _controlled_reflection(control[0], [*x_o, *m_x_o, *sigma_1, *b, *m_b, *sigma_0], [], work)


register_rule(ControlledRefAB, _controlled_ref_ab)
