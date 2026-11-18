r"""The reflections of the block encoding and the walk, exact including the global sign (spec 6.6,
9.6, 9.19, 9.20), built on PennyLane's ``MultiControlledX`` with static control values.

:func:`ref_t2` is :math:`\mathrm{REF}_{T_2} = 2\Pi_{T_2} - I` (step 23) with
:math:`\Pi_{T_2} = |0\rangle\langle 0|_{b, m_b, \sigma_0} \otimes
|1\rangle\langle 1|_{\mathrm{succ}_{x_o}}` (spec 3.3): -1 on every basis state but the pattern,
so it is :math:`-I` wherever the uniform preparation of ``x_o`` failed (spec 8.5).
:func:`ref_ab` is :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}} = 2|0\rangle\langle 0| - I`
(step 28) on ``x_o``, ``m_x_o``, ``sigma_1``, ``b``, ``m_b`` and ``sigma_0``. With ``control``
each is exactly :math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes (2\Pi - I)`;
control turns the global -1 into a relative phase, so the sign matters (spec 3.3).

The last pattern wire is the target: ``succ_x_o`` (pattern value 1) for :func:`ref_t2`,
``sigma_0`` (pattern value 0) for :func:`ref_ab`. The multi-controlled Z is ``X`` on the target
if its pattern value is 0, ``H``, ``MultiControlledX`` with the other pattern wires as controls
(their pattern values as control values) and :func:`reflection_work` zeroed work wires plus
the caller's ``extra_work`` (with at least two controls), ``H``, the ``X`` again:
:math:`I - 2\Pi`. Uncontrolled, ``GlobalPhase(pi)`` then multiplies by -1;
with one pattern wire the multi-controlled Z is a ``Z`` (PennyLane rejects a one-wire
``MultiControlledX``). Controlled, ``control`` is one more control of value 1, giving
:math:`I - 2|1\rangle\langle 1| \otimes \Pi`, and ``Z(control)`` multiplies the control's ``|1>``
branch by -1. ``work`` and ``extra_work`` return to ``|0>``; ``MultiControlledX`` without work
wires has no path (L-14). With :math:`c` controls PennyLane's rule uses :math:`c - 2` work wires
for its ``TemporaryAND`` ladder and leaves a two-control ``MultiControlledX`` at its core; one
more zeroed wire, ``extra_work`` (the walk lends one, so the pool does not grow), lets that core
decompose to a ``TemporaryAND`` and its measured uncompute. :data:`dfthc.compile.GATE_SET` has no
``Toffoli``, so with :math:`c \ge 2` controls ``extra_work`` is required (``ValueError``
otherwise). ``qp.FlipSign`` is not used: its compile time grows exponentially with the pattern
width, and generic ``qp.ctrl`` of it with ``GlobalPhase`` has no path (L-29). Nothing adjoints
a reflection: both forms are self-inverse.

**Cost.** :func:`dfthc.costs.ref_t2` and :func:`dfthc.costs.ref_ab`, :func:`dfthc.costs.reflection`
with :math:`n_w` pattern wires: :math:`T = \max(n_w - 2, 0)`, controlled :math:`n_w - 1`, all
``TemporaryAND`` with measured uncomputes (``A = T``), ``M = 0``. At
FeMoco-54: ``ref_t2`` 20 (controlled 21), as dfthcbe's, against the paper's 21 (``ref-unif``);
``ref_ab`` 44 (controlled 45) against dfthcbe's 45 and the paper's 46 (``ref-target``).

**Paper and dfthcbe.** Steps 23 and 28 (PRX App. B; dfthcbe cites the paper's (B11) for its
multi-controlled Z). dfthcbe: ``DFTHCBlockEncoding.ref_t2``, a Qualtran
``ReflectionUsingPrepare.reflection_around_zero`` on ``b``, ``m_b``, ``sigma_0`` controlled by
``succ_x_o`` with ``GlobalPhase(exponent=1)``, and ``DFTHCWalk.ref_ab`` on the six registers; their
controlled forms are the reflections' ``.controlled()`` with a ``ZGate`` on the control
(``dfthcbe/block_encoding.py``).
"""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.typing import Wire

from dfthc.compile import resource_key


def reflection_work(n_w: int, controlled: bool) -> int:
    """The zeroed work wires of a reflection about ``n_w`` pattern wires: ``max(n_w - 3, 0)``,
    controlled ``max(n_w - 2, 0)`` (a ``MultiControlledX`` with ``c`` controls takes
    ``max(c - 2, 0)``)."""
    return max(n_w + int(controlled) - 3, 0)


def _reflect(pattern: list[int], wires: list, work, control, extra_work) -> None:
    controls, values, target = list(wires[:-1]), list(pattern[:-1]), wires[-1]
    if control is not None:
        controls, values = [control, *controls], [1, *values]
    if len(controls) >= 2 and not extra_work:
        raise ValueError(f"a reflection with {len(controls)} controls needs one zeroed "
                         "extra_work wire: its MultiControlledX core is a Toffoli otherwise, "
                         "which GATE_SET lacks")
    if pattern[-1] == 0:
        qp.X(target)
    if controls:
        work_wires = [*work[:max(len(controls) - 2, 0)],
                      *(extra_work if len(controls) >= 2 else ())]
        qp.H(target)
        qp.MultiControlledX(wires=[*controls, target], control_values=values,
                            work_wires=work_wires or None, work_wire_type="zeroed")
        qp.H(target)
    else:
        qp.Z(target)
    if pattern[-1] == 0:
        qp.X(target)
    if control is None:
        qp.GlobalPhase(np.pi)
    else:
        qp.Z(control)


def _reflection_resources(pattern: list[int], controlled: bool, extra_work: int) -> dict:
    values = [1] * controlled + list(pattern[:-1])
    resources = {qp.X: 2 * (pattern[-1] == 0), qp.Z if controlled else qp.GlobalPhase: 1}
    if not values:
        resources[qp.Z] = resources.get(qp.Z, 0) + 1
        return resources
    num_work = max(len(values) - 2, 0) + (extra_work if len(values) >= 2 else 0)
    mcx = resource_key(lambda: qp.MultiControlledX(
        Wire[len(values) + 1], control_values=values,
        work_wires=Wire[num_work] if num_work else None, work_wire_type="zeroed"))
    return resources | {mcx: 1, qp.H: 2}


def ref_t2(succ_x_o, b, m_b, sigma_0, work, control=None, extra_work=()) -> None:
    r""":math:`\mathrm{REF}_{T_2}` on ``succ_x_o`` (1 wire), ``b``, ``m_b`` and ``sigma_0`` (1
    wire), controlled by the wire ``control`` if given: the multi-controlled Z on
    ``b ‖ m_b ‖ sigma_0 ‖ succ_x_o`` with the pattern zeros and a one (target ``succ_x_o``), then
    ``GlobalPhase(pi)`` or ``Z(control)``. ``work``: :func:`reflection_work` of
    :math:`n_w = w_b + b_{k2} + 2` zeroed wires; ``extra_work``: at most one more zeroed wire,
    disjoint from the others, which makes the decomposition Toffoli-free; required with at
    least two controls (``ValueError``), since ``GATE_SET`` has no ``Toffoli``."""
    wires = [*b, *m_b, *sigma_0, *succ_x_o]
    _reflect([0] * (len(wires) - len(succ_x_o)) + [1] * len(succ_x_o), wires, work, control,
             extra_work)


def ref_t2_resources(succ_x_o: int, b: int, m_b: int, sigma_0: int, controlled: bool,
                     extra_work: int = 0) -> dict:
    """The gates :func:`ref_t2` emits on registers of these widths with ``extra_work``
    lent wires."""
    return _reflection_resources([0] * (b + m_b + sigma_0) + [1] * succ_x_o, controlled,
                                 extra_work)


def ref_ab(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work, control=None, extra_work=()) -> None:
    r""":math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` on ``x_o``, ``m_x_o``, ``sigma_1``,
    ``b``, ``m_b`` and ``sigma_0``, controlled by the wire ``control`` if given: the
    multi-controlled Z on the six registers with the all-zero pattern (target the last wire of
    ``sigma_0``), then ``GlobalPhase(pi)`` or ``Z(control)``. ``work``: :func:`reflection_work`
    of :math:`n_w = w_x + b_{k1} + w_b + b_{k2} + 2` zeroed wires; ``extra_work`` as for
    :func:`ref_t2`."""
    wires = [*x_o, *m_x_o, *sigma_1, *b, *m_b, *sigma_0]
    _reflect([0] * len(wires), wires, work, control, extra_work)


def ref_ab_resources(x_o: int, m_x_o: int, sigma_1: int, b: int, m_b: int, sigma_0: int,
                     controlled: bool, extra_work: int = 0) -> dict:
    """The gates :func:`ref_ab` emits on registers of these widths with ``extra_work``
    lent wires."""
    return _reflection_resources([0] * (x_o + m_x_o + sigma_1 + b + m_b + sigma_0), controlled,
                                 extra_work)
