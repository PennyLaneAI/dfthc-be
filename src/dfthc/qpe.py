r"""Iterative phase estimation on the controlled walk (spec 9.23 and 11).

**The rounds** (spec 11.1). With :math:`t` = ``num_bits``, rounds run for
:math:`j = t-1, \dots, 0`. Round :math:`j` is one :class:`QPERound`: ``H`` on the control, the
controlled walk to the power :math:`2^j`, a phase correction
:math:`e^{-2\pi i/2^{l-j+1}}` on the control's :math:`|1\rangle` for every earlier bit
:math:`m_l = 1` (:math:`l > j`), ``H``; then the driver measures the control, :math:`m_j`, with
reset. The estimated phase is :math:`\varphi = \sum_j m_j 2^{-(j+1)}`: the bits give the
eigenvalue :math:`e^{2\pi i\varphi}` of the walk (an exact :math:`t`-bit phase is returned with
certainty).

**The walk power.** :class:`ControlledWalkPower` repeats ``ControlledDFTHCWalk`` with a
``qp.for_loop`` of static bound ``power`` (spec 9.23); ``dfthc.testing.toy_walk``'s
``ToyControlledWalkPower`` does the same for the toy walk (spec 11.4). :class:`QPERound` cannot
take the class (an ``Operator2`` takes no Python callable) or nested wire groups (it cannot mix
compilable with hybrid arguments), so it takes the class's name, its compilable arguments after
``power``, its wire groups after ``control`` as one flat register and their sizes: a
:class:`WalkPowerSpec`. Its rule looks the name up in :data:`WALK_POWERS`, where each walk-power
class registers itself with :func:`register_walk_power`.

**The correction wires** (spec 11.1). A round binds every wire once: its corrections' ``2t - 1``
wires are a slice of the round's flat register, taken inside the rule, not a register of their
own. The slice starts at the walk power's scratch group (``scratch_argname``: its last group,
which it takes and returns in ``|0>``; ``work`` for :class:`ControlledWalkPower`) and runs on
into spare wires after the walk's groups where that group is too small
(:func:`spare_wires_needed`); the toy walk has no scratch group, so its corrections take spare
wires only. A second register on the same qubits would not compile: the wire registers of one
operator must be distinct qubits.

**The driver** :func:`iterative_phase_estimation` is a plain quantum function, not an
``Operator2``, because it returns the measured bits (spec 2.1 R-4, 16 Q-3). It passes every
round the earlier bits as one array of shape :math:`(t,)`, ``False`` where no bit is known yet,
so every round has one argument signature (spec 11.1; K-10 part 3 passed in both modes).

**Energies** (spec 3.4 and 11.1): :func:`phase_from_bits`, :func:`energy_from_phase` and
:func:`physical_energy`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import AbstractArray, Wire
from pennylane.wires import Wires

from dfthc.compile import register_rule, resource_key
from dfthc.data.tables import WalkTables
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.phase_gradient import PhaseGradientRotation
from dfthc.ops.walk import ControlledDFTHCWalk, _check, _validate
from dfthc.params import WalkParams
from dfthc.registers import WalkLayout

WALK_POWERS: dict[str, type[Operator2]] = {}
"""The walk-power classes :class:`QPERound` can call, by class name."""


def register_walk_power(cls: type[Operator2]) -> type[Operator2]:
    """Add ``cls`` to :data:`WALK_POWERS` (once). A walk-power class takes ``power`` first, then
    its compilable arguments, then a one-wire ``control`` and its wire groups. Its class
    attribute ``scratch_argname`` names its last wire group if it takes that group in ``|0>`` and
    returns it in ``|0>`` (:class:`QPERound`'s corrections borrow it), and is ``None``
    otherwise. Its static ``validate_compilable(power, *compilable)`` checks ``power`` and the
    compilable arguments as the constructor does, without side effects (``ValueError``):
    :class:`QPERound` calls it at construction."""
    if cls.__name__ in WALK_POWERS:
        raise ValueError(f"{cls.__name__} is already a registered walk power")
    scratch = getattr(cls, "scratch_argname", "")
    if scratch not in (None, cls.wire_argnames[-1]):
        raise ValueError(f"{cls.__name__}.scratch_argname must be None or its last wire group "
                         f"{cls.wire_argnames[-1]!r}; got {scratch!r}")
    if not callable(getattr(cls, "validate_compilable", None)):
        raise ValueError(f"{cls.__name__} needs a static validate_compilable(power, *compilable)")
    WALK_POWERS[cls.__name__] = cls
    return cls


def _walk_power(walk_op: str) -> type[Operator2]:
    if walk_op not in WALK_POWERS:
        raise ValueError(f"walk_op must be one of {sorted(WALK_POWERS)}; got {walk_op!r}")
    return WALK_POWERS[walk_op]


def spare_wires_needed(walk_op: str, walk_wire_sizes: Sequence[int], num_bits: int) -> int:
    """The wires a ``num_bits``-bit estimation of the walk power ``walk_op`` (its groups after
    ``control`` of sizes ``walk_wire_sizes``) needs after the walk's groups: the
    ``2 num_bits - 1`` correction wires less the walk power's scratch group, at least 0.
    ``ControlledWalkPower`` lends its ``work`` pool (``WalkLayout.work_size``: 8 wires at
    ``H2_STO3G_TEST``, 44 at FeMoco-54, so no spare wire up to 4 and 22 bits); the toy walk
    lends none."""
    scratch = walk_wire_sizes[-1] if _walk_power(walk_op).scratch_argname else 0
    return max(0, 2 * num_bits - 1 - scratch)


class WalkPowerSpec(NamedTuple):
    """A walk-power operator as :class:`QPERound` takes it (spec 11.1)."""

    op_name: str
    """The class name in :data:`WALK_POWERS`."""
    compilable: tuple
    """Its compilable arguments after ``power``."""
    wire_sizes: tuple[int, ...]
    """The sizes of its wire groups after ``control``, in ``wire_argnames`` order."""
    wires: Wires
    """Those groups concatenated in the same order."""


class ControlledWalkPower(Operator2):
    r"""The controlled walk to a power, :math:`C(W)^p = C(W^p)` (spec 9.23): ``power`` steps of
    :class:`~dfthc.ops.walk.ControlledDFTHCWalk`.

    **Action.** :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle 1|_{control}
    \otimes W^p` on inputs whose ancilla groups are in :math:`|0\rangle` or in a state the walk
    reaches from there, with :math:`W` the walk of
    :class:`~dfthc.ops.walk.DFTHCWalk` for the same ``params`` and ``tables``: each step is the
    controlled walk, which is correct on the walk's own states (spec 12.4, E-3), so the steps
    compose. The output is the same for every outcome of the erase measurements.

    **Registers.** Those of :class:`~dfthc.ops.walk.ControlledDFTHCWalk`.

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``control``    1                              any                   a control
    ``outer``      the ``outer`` group (71)       walk state            walk state
    ``inner``      the ``inner`` group (65)       walk state            walk state
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    "Walk state": ``|0>`` before the first step; between steps only the six index registers of
    :math:`\mathcal{A}\cup\mathcal{B}` (``x_o``, ``m_x_o``, ``sigma_1`` in ``outer``; ``b``,
    ``m_b``, ``sigma_0`` in ``inner``) may differ from ``|0>``. ``work`` is its scratch group
    (``scratch_argname``): :class:`QPERound`'s corrections borrow it between walk powers.

    **Compilable arguments.** ``power`` (int, at least 1), then ``params`` and ``tables`` as in
    :class:`~dfthc.ops.walk.ControlledDFTHCWalk`, with the same checks (``ValueError``).

    **Pre- and postconditions.** Those of :class:`~dfthc.ops.walk.ControlledDFTHCWalk`, on both
    control branches.

    **Decomposition** (spec 9.23). One ``qp.for_loop`` with the static bound ``power`` whose body
    is ``ControlledDFTHCWalk(params, tables, control, outer, inner, select, system, phase_grad,
    work)``, the groups unpacked into lists once before the loop (spec 9.0).

    **Adjoint and control.** Never adjointed or controlled (spec 7.4 rules 4 and 6): its body
    holds a loop and measured erases. :class:`QPERound` calls it.

    **Cost.** :func:`dfthc.costs.controlled_walk_power`: :math:`T = p\,T_{C(W)}`,
    :math:`M = p\,M_W`; ``A`` = ``p`` times :func:`dfthc.costs.measured_and_uncomputes`
    (controlled). At FeMoco-54 and ``p = 1``: 9,795, 2,524 and 8,677.

    **Paper and dfthcbe.** The walk powers :math:`W^{2^j}` of phase estimation on the
    controlled walk, whose query count the paper estimates (PRX 15, 041016; arXiv:2502.15882v1
    Sec. II, after Eq. (11); spec 3.3 and 3.4). The paper's phase-estimation step controls only
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` (App. B.7.2, Eq. (128)), which gives
    :math:`C(W^p)` only for even ``p`` because the block encoding is self-inverse. Each step here
    also controls :math:`\mathrm{REF}_{T_2}` (spec 9.23), so every ``p`` works, round
    :math:`j = 0` included, at ``p`` Toffolis more than that model: at FeMoco-54 ``9792 p``
    against ``9791 p``, and the paper's Table III step is 9,997. dfthcbe builds no phase
    estimation; its controlled walk is ``DFTHCWalk(is_controlled=True)``.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.qpe import ControlledWalkPower
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p)
        groups = {name: getattr(layout, name) for name in ControlledWalkPower.wire_argnames[1:]}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            ControlledWalkPower(2, p.as_compilable(), t.as_compilable(), [layout.num_wires],
                                **groups)

        compiled_resources(program, num_wires=layout.num_wires + 1).toffoli  # 2 * 160 = 320
    """

    compilable_argnames = ("power", "params", "tables")
    wire_argnames = ControlledDFTHCWalk.wire_argnames
    scratch_argname = "work"

    def __init__(self, power, params, tables, control, outer, inner, select, system, phase_grad,
                 work):
        power = check_int("power", power, low=1)
        wires = (control, outer, inner, select, system, phase_grad, work)
        compilable = _validate("ControlledWalkPower", self.wire_argnames, params, tables, wires)
        super().__init__(power, *compilable, *wires)

    @staticmethod
    def required_wires(power, params, tables) -> dict[str, int]:
        """Those of :class:`~dfthc.ops.walk.ControlledDFTHCWalk`; ``power`` changes none."""
        return ControlledDFTHCWalk.required_wires(params, tables)

    @staticmethod
    def validate_compilable(power, params, tables) -> None:
        """The constructor's checks of ``power``, ``params`` and ``tables``, without wires
        (``ValueError``); :class:`QPERound` calls it."""
        check_int("power", power, low=1)
        _check(params, tables)


def _controlled_walk_power_resources(power, params, tables, control, outer, inner, select,
                                     system, phase_grad, work):
    sizes = ControlledDFTHCWalk.required_wires(params, tables)
    step = resource_key(lambda: ControlledDFTHCWalk(params, tables, **{
        name: Wire[sizes[name]] for name in ControlledDFTHCWalk.wire_argnames}))
    return {step: power}


@qp.register_resources(_controlled_walk_power_resources)
def _controlled_walk_power(power, params, tables, control, outer, inner, select, system,
                           phase_grad, work):
    groups = [list(group) for group in (control, outer, inner, select, system, phase_grad, work)]

    @qp.for_loop(power)
    def step(_):
        ControlledDFTHCWalk(params, tables, *groups)

    step()


register_rule(ControlledWalkPower, _controlled_walk_power)
register_walk_power(ControlledWalkPower)


def walk_power_spec(params: WalkParams, tables: WalkTables, layout: WalkLayout) -> WalkPowerSpec:
    """The :class:`WalkPowerSpec` of :class:`ControlledWalkPower` on ``layout``'s groups, in
    ``wire_argnames`` order (which is not the layout's order: the layout starts with
    ``system``)."""
    names = ControlledWalkPower.wire_argnames[1:]
    compilable = (params.as_compilable(), tables.as_compilable())
    sizes = ControlledWalkPower.required_wires(1, *compilable)
    return WalkPowerSpec("ControlledWalkPower", compilable, tuple(sizes[name] for name in names),
                         Wires.all_wires([getattr(layout, name) for name in names]))


def _bits_array(earlier_bits, num_bits: int):
    """``earlier_bits`` as one array of shape ``(num_bits,)`` (L-23): a boolean ``jax`` array
    under capture, an object array of ``MeasurementValue`` s and ``False`` in tape mode."""
    if isinstance(earlier_bits, AbstractArray):
        bits = earlier_bits
    elif qp.capture.enabled():
        bits = jnp.stack([jnp.asarray(bit, dtype=bool) for bit in earlier_bits])
    else:
        bits = np.empty(len(earlier_bits), dtype=object)
        bits[:] = list(earlier_bits)
    if tuple(bits.shape) != (num_bits,):
        raise ValueError(f"earlier_bits must have shape ({num_bits},); got {tuple(bits.shape)}")
    return bits


class QPERound(Operator2):
    r"""Round :math:`j` of iterative phase estimation, up to the control's measurement (spec
    11.1): ``H``, the walk power :math:`2^j`, the phase corrections of the earlier bits, ``H``.

    **Action.** On the control in :math:`|0\rangle` and a walk eigenstate with eigenvalue
    :math:`e^{2\pi i\varphi}`, it leaves the control in
    :math:`\big((1 + e^{i\omega})|0\rangle + (1 - e^{i\omega})|1\rangle\big)/2` with
    :math:`\omega = 2\pi\big(2^j\varphi - \sum_{l > j} m_l\,2^{-(l-j+1)}\big)`, where
    :math:`m_l` = ``earlier_bits[l]``; measuring it gives :math:`m_j`. For a :math:`t`-bit
    :math:`\varphi` with the earlier bits right, :math:`\omega = \pi m_j` exactly.

    **Registers.**

    ============== ================================== ===================== ==================
    name           width                              input                 output
    ============== ================================== ===================== ==================
    ``control``    1                                  ``|0>``               see the action
    ``qpe_grad``   ``num_bits``                       phase-gradient state  unchanged
    ``walk_wires`` ``sum(walk_wire_sizes) + spare``   see below             see below
    ============== ================================== ===================== ==================

    ``walk_wires`` are the walk power's wire groups after its ``control``, concatenated in its
    ``wire_argnames`` order (the walk's input and output), then ``spare`` =
    :func:`spare_wires_needed` wires in ``|0>``. The corrections use the ``2 num_bits - 1``
    wires of ``walk_wires`` from the first wire of the walk power's scratch group (its last
    group, ``|0>`` whenever corrections run; ``work`` for ``ControlledWalkPower``) on, into the
    spare wires where the group is too small; without a scratch group (the toy walk) they are
    the spare wires. So each wire is bound once (the wire registers of one operator must be
    distinct qubits).

    **Compilable arguments.** ``j`` (int, ``0 <= j < num_bits``), ``num_bits`` (int, at least
    1), ``walk_op`` (a name in :data:`WALK_POWERS`), ``walk_compilable`` (the walk power's
    compilable arguments after ``power``, a tuple, checked by its ``validate_compilable``) and
    ``walk_wire_sizes`` (the walk power's ``required_wires`` after ``control``, in order).
    **Dynamic:** ``earlier_bits``, shape ``(num_bits,)``: ``m_l`` at every ``l > j``, ``False``
    elsewhere; the rule reads only ``l > j``.

    **Pre- and postconditions.** ``qpe_grad`` holds the ``num_bits``-bit phase-gradient state;
    the walk power's scratch group and the spare wires start and end in ``|0>``; the walk
    power's own conditions hold on its groups.

    **Decomposition** (spec 11.1). ``H(control)``; ``WALK_POWERS[walk_op](2**j,
    *walk_compilable, control, *groups)`` with ``groups`` split from ``walk_wires`` by
    ``walk_wire_sizes``; ``corrections`` the ``2 num_bits - 1`` correction wires above; for each
    ``l = j + 1, ..., num_bits - 1`` with ``d = l - j + 1``, ``qp.cond(earlier_bits[l],
    correction)()``, whose branch builds ``PhaseGradientRotation(2**d - 1, d, control,
    qpe_grad[:d], corrections[:d], corrections[d:2 d - 1])``, the phase
    :math:`e^{-2\pi i/2^d}` on :math:`|1\rangle` (the top ``d`` wires of a ``num_bits``-bit
    gradient are a ``d``-bit gradient); ``H(control)``. The corrections are not hinted, so
    ``qp.analyze`` counts each at its maximum branch.

    **Adjoint and control.** Never adjointed or controlled (spec 7.4 rules 4 to 6): it holds the
    walk power's loop, conditionals on measured bits and measured erases.

    **Cost.** :func:`dfthc.costs.qpe_round`, worst case: :math:`T = 2^j T_{C(W)} +
    \sum_{l=j+1}^{t-1}(l - j)`, :math:`M = 2^j M_W` (a ``d``-bit rotation costs ``d - 1``
    Toffolis, spec 9.1). With ``walk_op = "ControlledWalkPower"``.

    **Paper and dfthcbe.** The semiclassical (iterative) phase estimation with feed-forward
    rotations (Griffiths and Niu, PRL 76, 3228 (1996)) on the controlled walk; the paper's
    estimate costs and dfthcbe have no circuit for it. The corrections are phase-gradient
    additions (Sanders et al., PRX Quantum 1, 020312), so no inverse QFT is needed (spec 11.2).

    **Example.**

    .. code-block:: python

        import numpy as np
        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.qpe import QPERound
        from dfthc.testing.toy_walk import toy_walk_power_spec

        spec = toy_walk_power_spec(system=[3], ancilla=[4])

        def program():
            qp.PhaseGradientStatePrep(wires=[1, 2])
            QPERound(0, 2, np.array([False, True]), spec.op_name, spec.compilable,
                     spec.wire_sizes, control=[0], qpe_grad=[1, 2], walk_wires=[3, 4, 5, 6, 7])

        compiled_resources(program, num_wires=8).toffoli  # 2 + 1 = 3
    """

    dynamic_argnames = ("earlier_bits",)
    compilable_argnames = ("j", "num_bits", "walk_op", "walk_compilable", "walk_wire_sizes")
    wire_argnames = ("control", "qpe_grad", "walk_wires")

    def __init__(self, j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes,
                 control, qpe_grad, walk_wires):
        num_bits = check_int("num_bits", num_bits, low=1)
        j = check_int("j", j, low=0, high=num_bits - 1)
        cls = _walk_power(walk_op)
        if not isinstance(walk_compilable, tuple):
            raise ValueError(f"walk_compilable must be a tuple; got {walk_compilable!r}")
        if len(walk_compilable) != len(cls.compilable_argnames) - 1:
            raise ValueError(f"walk_compilable must hold {walk_op}'s compilable arguments after "
                             f"power, {cls.compilable_argnames[1:]}; got {len(walk_compilable)}")
        cls.validate_compilable(2**j, *walk_compilable)
        walk_wire_sizes = canonical_ints("walk_wire_sizes", tuple(walk_wire_sizes))
        required = cls.required_wires(2**j, *walk_compilable)
        groups = tuple(size for name, size in required.items() if name != "control")
        if walk_wire_sizes != groups:
            raise ValueError(f"walk_wire_sizes must be {walk_op}'s required_wires after control, "
                             f"{groups}; got {walk_wire_sizes}")
        bits = _bits_array(earlier_bits, num_bits)
        check_wires("QPERound", {"control": control, "qpe_grad": qpe_grad,
                                 "walk_wires": walk_wires},
                    self.required_wires(j, num_bits, walk_op, walk_compilable, walk_wire_sizes))
        super().__init__(j, num_bits, bits, walk_op, walk_compilable, walk_wire_sizes, control,
                         qpe_grad, walk_wires)

    @staticmethod
    def required_wires(j, num_bits, walk_op, walk_compilable, walk_wire_sizes) -> dict[str, int]:
        """Exact sizes; every round of a ``num_bits``-bit estimation has the same ones."""
        return {"control": 1, "qpe_grad": num_bits,
                "walk_wires": sum(walk_wire_sizes)
                + spare_wires_needed(walk_op, walk_wire_sizes, num_bits)}


def _corrections(j: int, num_bits: int) -> list[int]:
    """The correction widths ``d = l - j + 1`` of round ``j``, for ``l = j + 1 .. num_bits - 1``."""
    return [later - j + 1 for later in range(j + 1, num_bits)]


def _qpe_round_resources(j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes,
                         control, qpe_grad, walk_wires):
    cls = WALK_POWERS[walk_op]
    walk = resource_key(lambda: cls(2**j, *walk_compilable, Wire[1],
                                    *[Wire[size] for size in walk_wire_sizes]))
    resources = {qp.H: 2, walk: 1}
    for d in _corrections(j, num_bits):
        rotation = resource_key(lambda d=d: PhaseGradientRotation(
            2**d - 1, d, Wire[1], Wire[d], Wire[d], Wire[d - 1]))
        resources[rotation] = 1
    return resources


@qp.register_resources(_qpe_round_resources)
def _qpe_round(j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes, control,
               qpe_grad, walk_wires):
    control, qpe_grad, walk_wires = list(control), list(qpe_grad), list(walk_wires)
    groups, start = [], 0
    for size in walk_wire_sizes:
        groups.append(walk_wires[start:start + size])
        start += size
    if WALK_POWERS[walk_op].scratch_argname:
        start -= walk_wire_sizes[-1]
    corrections = walk_wires[start:start + 2 * num_bits - 1]
    qp.H(control)
    WALK_POWERS[walk_op](2**j, *walk_compilable, control, *groups)
    for later, d in zip(range(j + 1, num_bits), _corrections(j, num_bits), strict=True):
        def correction(d=d):
            PhaseGradientRotation(2**d - 1, d, control, qpe_grad[:d], corrections[:d],
                                  corrections[d:2 * d - 1])

        qp.cond(earlier_bits[later], correction)()
    qp.H(control)


register_rule(QPERound, _qpe_round)


def _no_bits(num_bits: int):
    if qp.capture.enabled():
        return jnp.zeros(num_bits, dtype=bool)
    return np.full(num_bits, False, dtype=object)


def _with_bit(bits, j: int, bit):
    if qp.capture.enabled():
        return bits.at[j].set(jnp.asarray(bit, dtype=bool))
    bits = bits.copy()
    bits[j] = bit
    return bits


def iterative_phase_estimation(walk_power_factory: Callable[[], WalkPowerSpec], control,
                               num_bits: int, qpe_grad, spare_wires=()) -> list:
    r"""Iterative phase estimation of the walk power that ``walk_power_factory()`` names (spec
    11.1): the measured bits :math:`m_{t-1}, \dots, m_0` in measurement order, with
    :math:`t` = ``num_bits`` and :math:`\varphi = \sum_j m_j 2^{-(j+1)}` (:func:`phase_from_bits`).

    A plain quantum function (spec 2.1 R-4, 16 Q-3): it prepares the ``num_bits``-bit phase
    gradient on ``qpe_grad[:num_bits]`` (``PhaseGradientStatePrep``), then for
    :math:`j = t-1, \dots, 0` applies :class:`QPERound` with the earlier bits as one array of
    shape :math:`(t,)` and measures ``control`` with reset, so it ends in :math:`|0\rangle`.
    ``control`` (one wire) starts in :math:`|0\rangle`. The corrections borrow the walk power's
    scratch group and, where it is too small, the first :func:`spare_wires_needed` wires of
    ``spare_wires`` (in :math:`|0\rangle`, disjoint from the walk's wires; the rest unused).
    The walk's own registers keep the walk's state between rounds; the caller prepares the
    walk's ``phase_grad``. Returns ``MeasurementValue`` s in tape mode and traced booleans under
    capture. Raises ``ValueError`` if ``num_bits`` is not a positive int, exceeds
    ``len(qpe_grad)``, if ``spare_wires`` is shorter than the corrections need, or if the wires
    the rounds use are not distinct.
    """
    num_bits = check_int("num_bits", num_bits, low=1)
    if num_bits > len(qpe_grad):
        raise ValueError(f"num_bits = {num_bits} exceeds len(qpe_grad) = {len(qpe_grad)}")
    spec = walk_power_factory()
    spare = spare_wires_needed(spec.op_name, spec.wire_sizes, num_bits)
    if len(spare_wires) < spare:
        raise ValueError(f"spare_wires needs {spare} wires for num_bits = {num_bits} with "
                         f"{spec.op_name}; got {len(spare_wires)}")
    control, qpe_grad = list(control), list(qpe_grad)[:num_bits]
    walk_wires = [*spec.wires, *list(spare_wires)[:spare]]
    if len({*control, *qpe_grad, *walk_wires}) != len(control) + num_bits + len(walk_wires):
        raise ValueError("control, qpe_grad, the walk's wires and the spare wires must be "
                         "distinct")
    qp.PhaseGradientStatePrep(qpe_grad)
    earlier, measured = _no_bits(num_bits), []
    for j in reversed(range(num_bits)):
        QPERound(j, num_bits, earlier, spec.op_name, spec.compilable, spec.wire_sizes, control,
                 qpe_grad, walk_wires)
        bit = qp.measure(control[0], reset=True)
        measured.append(bit)
        earlier = _with_bit(earlier, j, bit)
    return measured


def phase_from_bits(bits: Sequence[int]) -> float:
    r""":math:`\varphi = \sum_j m_j 2^{-(j+1)}` from the bits in the driver's order
    :math:`m_{t-1}, \dots, m_0` (spec 11.1)."""
    t = len(bits)
    return sum(int(bit) * 2.0 ** -(t - position) for position, bit in enumerate(bits))


def energy_from_phase(phi: float, Lambda: float, shift: float = 1.0) -> float:
    r"""The eigenvalue :math:`E = \Lambda(\mathrm{shift} + \cos 2\pi\varphi)` that the phase
    :math:`\varphi` of the walk estimates (spec 11.1): with ``Lambda`` the encoded
    :math:`\tilde\Lambda` and ``shift = 1`` the sum-of-squares eigenvalue :math:`E_{SA}` of
    :math:`\tilde H_{SA}` for the DFTHC walk (block :math:`\tilde H_{SA}/\tilde\Lambda - I`),
    ``shift = 0`` for standard qubitization (block :math:`H/\lambda`). The phases
    :math:`\pm\varphi` give the same energy."""
    return float(Lambda * (shift + np.cos(2 * np.pi * phi)))


def physical_energy(E_SA: float, e_sos_phys: float, core_energy: float = 0.0) -> float:
    r""":math:`E_{SA} + E_{SoS}^{phys}` + ``core_energy``: the physical electronic energy of an
    :math:`\eta`-electron eigenstate, plus the core energy where it is given (spec 3.4 and 11.1).
    ``e_sos_phys`` is ``summary.json``'s ``e_sos`` (:func:`dfthc.data.sos.e_sos_phys`), not the
    walk frame's offset."""
    return float(E_SA + e_sos_phys + core_energy)


def qpe_program(params: WalkParams, tables: WalkTables,
                num_bits: int) -> tuple[Callable[[], list], int]:
    """The QPE program of spec 11.3 at ``params`` and its wire count: the walk's phase gradient,
    then :func:`iterative_phase_estimation` of :class:`ControlledWalkPower` on the groups of
    ``WalkLayout.from_params(params, qpe_bits=num_bits)``, its ``qpe_control`` and ``qpe_grad``.
    The corrections borrow the walk's ``work`` pool; only where ``2 num_bits - 1`` exceeds it
    (from 5 bits at ``H2_STO3G_TEST``, from 23 at FeMoco-54) do spare wires follow the
    layout."""
    layout = WalkLayout.from_params(params, qpe_bits=num_bits)
    spec = walk_power_spec(params, tables, layout)
    spare = spare_wires_needed(spec.op_name, spec.wire_sizes, num_bits)
    spare_wires = list(range(layout.num_wires, layout.num_wires + spare))

    def program():
        qp.PhaseGradientStatePrep(layout.phase_grad)
        return iterative_phase_estimation(lambda: spec, layout.qpe_control, num_bits,
                                          layout.qpe_grad, spare_wires)

    return program, layout.num_wires + spare
