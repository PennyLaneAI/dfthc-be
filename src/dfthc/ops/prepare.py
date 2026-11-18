"""The alias-sampling prepares and their measured unprepares (spec 9.7 and 9.8).

**Alias sampling** (spec 6.4). Each prepare samples an index ``i`` over ``[0, L)`` with the
probabilities ``alias_probabilities(alt, keep, b_k)`` of its alias tables, in five steps (dfthcbe's
``OuterPrepare`` and ``InnerPrepare``):

1. ``FlaggedUniformPrep(L, s_u)`` on the index register, its flag ``succ`` and ``succ_work``;
2. ``H`` on the ``b_k`` wires of ``m`` and on the spin wire;
3. ``QROAMRead`` of the alias entry at the address into ``alt ‖ keep ‖ ...`` (the target) and the
   block's other entries (the junk);
4. ``LeftQuantumComparator(keep, m, less_than, cmp_work, "<=")``: ``less_than = [keep <= m]``;
5. one ``CSWAP`` per index wire swaps ``alt`` and the index, controlled by ``less_than``, each
   with the zeroed ``work[0]`` (:func:`dfthc.ops._gates.cswap`).

The unprepare runs the swaps again, the comparator's adjoint (a measured uncompute, which also
cleans ``cmp_work``), ``QROAMErase`` of the read, the Hadamards, and
``qp.adjoint(FlaggedUniformPrep(...))`` on the same index, ``succ`` and ``succ_work`` wires. The
order is forced: the erase needs ``target ‖ junk`` to hold what the read loaded at the current
address (for an unprepare from outcomes, the target: the junk was X-measured right after the
prepare), so the swap is undone first, and the comparator reads ``keep``, so its adjoint comes
before the erase; nothing between a prepare and its unprepare may write ``succ``, ``succ_work`` or
``cmp_work``.

**Tables.** A read entry is one big-endian integer (spec 6.7) of ``width`` bits, stored as 32-bit
words, least significant first (L-24): ``(alt << b_k1) | keep`` for the outer read
(:meth:`OuterPrepare.read_table`). A prepare lays its compilable tables out as the read's dynamic
``columns`` inside its rule (:meth:`dfthc.ops.qroam.QROAMRead.columns_from_table`); its
unprepare passes the same table, in the same entry layout, to
:class:`~dfthc.ops.qroam.QROAMErase`, whose masks are therefore exactly what the read loaded. The
erase's one-hot register is the reset ``target ‖ junk``, so ``2**onehot_bits`` must not exceed
``2**block_bits * width`` (the constructors check it, so ``QROAMErase`` needs no
``extra_onehot``; an unprepare from outcomes, whose junk was measured earlier, takes the wires
beyond its target as ``extra_onehot``).

**Inner read.** Its address is ``x_o ‖ b``, rows ``[0, R C)`` and ``b`` in ``[0, B + 1)`` with the
block bits on ``b``, and its entry ``alt ‖ keep ‖ sgn ‖ d_sgn ‖ r`` with ``r = x_o // C``
(:meth:`InnerPrepare.read_table`). At one-body rows, at ``x_o >= X_o`` and at failed ``b`` the
selected target entry is zero (spec 8.5; the junk may hold other entries of a visited block):
``keep = 0``, so ``less_than = 1``, the swap moves ``alt = 0`` into ``b``, and
``sgn = d_sgn = r = 0``. ``x_o >= X_o`` is an input the walk never presents: ``OuterPrepare``'s
swap maps a failed draw to ``x_o = 0``, an SF row, where the inner signs cancel between the two
rounds and ``REF_T2`` gives ``-I``.

**Signs** (spec 6.1). :class:`InnerSigns` applies ``Z(sgn) CZ(less_than_b, d_sgn)`` after
``InnerPrepare``: with ``sgn = [w_b < 0]`` and ``d_sgn = sgn XOR sgn[alt]`` loaded for the drawn
``b``, every branch carries the sign of the weight that survives the swap, and every branch whose
selected entry is zero carries +1.

**Rules.** The rules are straight-line and their children are composite keys, so they declare
their resources by hand; T2 runs them at a size in every branch.

**Unprepares from outcomes** (spec 16, Q-1 variant (b)). :class:`OuterUnprepareFromOutcomes` and
:class:`InnerUnprepareFromOutcomes` are the unprepares for a caller that X-measured the junk
right after the prepare (the block encoding does, so the junk wires are free for the angle
registers). They take the prepare's wires except the junk, the prepare's compilable arguments and
the junk outcomes as a dynamic array, and run the unprepare with the erase split: the swaps, the
comparator's adjoint, X measurements of the target only, the parities of the target and junk
outcomes together and the fixup (:func:`~dfthc.ops.qroam.emit_erase_from_outcomes`), the
Hadamards and the adjoint uniform preparation. The fixup's one-hot register is the reset target
and ``extra_onehot``, wires the caller lends in ``|0>``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.ops import MeasurementValue
from pennylane.typing import AbstractArray, Bool, Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._gates import cswap, cswap_resources
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.qroam import (
    WORD,
    QROAMEmitter,
    QROAMErase,
    QROAMRead,
    emit_erase_from_outcomes,
)
from dfthc.ops.unary import as_resources
from dfthc.ops.uniform import FlaggedUniformPrep
from dfthc.params import lg

COMPARATOR = "<="
"""The keep test writes ``less_than = [keep <= m]``; a set flag swaps in ``alt`` (spec 6.4)."""


# --------------------------------------------------------------------------- shared skeleton


def _words(value: int, width: int) -> tuple[int, ...]:
    """``value`` as ``ceil(width / 32)`` words, least significant first (L-24)."""
    return tuple((value >> (WORD * w)) & (2**WORD - 1) for w in range(-(-width // WORD)))


def _join(*registers) -> list:
    """The wires of ``registers``, concatenated in order."""
    return [wire for register in registers for wire in register]


def _check_table(name: str, values, shape: tuple[int, ...], bound: int) -> tuple:
    """``values`` as nested tuples of Python ints of ``shape`` (one or two sizes), every entry in
    ``[0, bound)``; ``ValueError`` otherwise."""
    values = canonical_ints(name, values)
    rows = [values] if len(shape) == 1 else values
    if (not isinstance(values, tuple) or len(values) != shape[0]
            or any(not isinstance(row, tuple) or len(row) != shape[-1] for row in rows)
            or any(not isinstance(entry, int) for row in rows for entry in row)):
        raise ValueError(f"{name} must be a table of ints of shape {shape}")
    bad = [entry for row in rows for entry in row if not 0 <= entry < bound]
    if bad:
        raise ValueError(f"{name} entries must lie in [0, {bound}), got {bad[0]}")
    return values


def _check_onehot(name: str, onehot_bits: int, read: dict) -> None:
    """``ValueError`` unless the erase's one-hot register fits in the reset ``target ‖ junk``."""
    wires = 2 ** read["block_bits"] * read["width"]
    if 2**onehot_bits > wires:
        raise ValueError(f"the erase's one-hot over {name} = {onehot_bits} bits needs "
                         f"{2**onehot_bits} wires, more than the {wires} target and junk wires")


def _junk_outcomes(junk_outcomes, size: int):
    """Canonicalize the dynamic ``junk_outcomes`` once (L-23), one outcome per junk qubit: under
    capture a bool array of shape ``(size,)`` (empty for ``size = 0``); in tape mode an object
    array whose every entry is a ``MeasurementValue``, so that both modes emit the same fixup
    conditionals (K-10). Abstract arrays and tracers pass with their shape checked.
    ``ValueError`` otherwise."""
    if isinstance(junk_outcomes, AbstractArray) or qp.math.is_tracer(junk_outcomes):
        if tuple(junk_outcomes.shape) != (size,):
            raise ValueError(f"junk_outcomes must have shape ({size},), got "
                             f"{tuple(junk_outcomes.shape)}")
        return junk_outcomes
    values = list(junk_outcomes)
    if len(values) != size:
        raise ValueError(f"need one outcome per junk qubit ({size}), got {len(values)}")
    if qp.capture.enabled():
        if not values:
            return jnp.zeros((0,), dtype=bool)
        return jnp.stack([jnp.asarray(value, dtype=bool) for value in values])
    for i, value in enumerate(values):
        if not isinstance(value, MeasurementValue):
            raise ValueError(f"in tape mode every junk outcome must be a MeasurementValue; "
                             f"entry {i} is {value!r}")
    return np.asarray(values, dtype=object)


def abstract_junk_outcomes(size: int) -> AbstractArray:
    """The ``junk_outcomes`` of a resource key that declares an unprepare from outcomes: a bool
    array under capture, an object array of ``MeasurementValue``s in tape mode, as
    :func:`_junk_outcomes` canonicalizes them (spec 7.4 rule 9: the declarations' only
    mode-specific code). Call it before :func:`~dfthc.compile.resource_key`, which pauses
    capture."""
    return AbstractArray((size,), bool if qp.capture.enabled() else object)


@dataclass(frozen=True)
class _AliasSampling:
    """The alias-sampling rule skeleton of both prepares and unprepares (module docstring).

    ``L`` values of the sampled index, ``s_u``-bit uniform rotations, ``b_k`` keep bits, the
    read's shape ``read`` (``address_widths``, ``live_ranges``, ``width``, ``block_bits``) and the
    erase's ``onehot_bits``.
    """

    L: int
    s_u: int
    b_k: int
    read: dict
    onehot_bits: int

    def work_sizes(self) -> tuple[int, int, int]:
        """Work wires of the uniform preparation, the read and the erase."""
        return (FlaggedUniformPrep.required_wires(self.L, self.s_u)["work"],
                QROAMRead.required_wires(**self.read, control_value=None)["work"],
                QROAMErase.required_wires((), **self.read, control_value=None,
                                          onehot_bits=self.onehot_bits)["work"])

    def sizes(self) -> dict[str, int]:
        """Sizes of the skeleton's registers; ``work`` is the largest of the three needs and at
        least 1, the swaps' zeroed wire."""
        uniform = FlaggedUniformPrep.required_wires(self.L, self.s_u)
        return {"index": lg(self.L), "succ": 1, "succ_work": uniform["succ_work"],
                "m": self.b_k, "less_than": 1, "cmp_work": self.b_k - 1, "alt": lg(self.L),
                "keep": self.b_k,
                "junk": (2 ** self.read["block_bits"] - 1) * self.read["width"],
                "phase_grad": self.s_u, "work": max(*self.work_sizes(), 1)}

    def extra_onehot(self) -> int:
        """Wires of the erase's one-hot register beyond the reset target, when the junk was
        measured earlier: ``max(2**onehot_bits - width, 0)`` (0 without a one-hot)."""
        return max(2**self.onehot_bits - self.read["width"], 0) if self.onehot_bits else 0

    def prepare(self, table, *, index, succ, succ_work, hadamards, m, less_than, cmp_work, alt,
                keep, address, target, junk, phase_grad, work) -> None:
        """Steps 1 to 5; the read's ``columns`` come from ``table``."""
        uniform, read, _ = self.work_sizes()
        FlaggedUniformPrep(self.L, self.s_u, index, succ, succ_work, phase_grad[:self.s_u],
                           work[:uniform])
        for wire in hadamards:
            qp.H(wire)
        QROAMRead(QROAMRead.columns_from_table(table, **self.read), **self.read,
                  control_value=None, address=address, target=target, junk=junk,
                  work=work[:read], control=[])
        qp.LeftQuantumComparator(keep, m, less_than, cmp_work, COMPARATOR)
        for a, i in zip(alt, index, strict=True):
            cswap(less_than[0], a, i, work[0])

    def unprepare(self, table, *, index, succ, succ_work, hadamards, m, less_than, cmp_work,
                  alt, keep, address, target, junk, phase_grad, work) -> None:
        """Steps 5 to 1 undone, with the measured comparator uncompute and the measured erase."""
        uniform, _, erase = self.work_sizes()
        for a, i in zip(alt, index, strict=True):
            cswap(less_than[0], a, i, work[0])
        qp.adjoint(qp.LeftQuantumComparator(keep, m, less_than, cmp_work, COMPARATOR))
        QROAMErase(table, **self.read, control_value=None, onehot_bits=self.onehot_bits,
                   address=address, target=target, junk=junk, work=work[:erase], control=[],
                   extra_onehot=[])
        for wire in hadamards:
            qp.H(wire)
        qp.adjoint(FlaggedUniformPrep(self.L, self.s_u, index, succ, succ_work,
                                      phase_grad[:self.s_u], work[:uniform]))

    def unprepare_from_outcomes(self, junk_outcomes, table, *, index, succ, succ_work, hadamards,
                                m, less_than, cmp_work, alt, keep, address, target, phase_grad,
                                work, extra_onehot) -> None:
        """:meth:`unprepare` with the erase split (spec 16, Q-1 variant (b)): the caller measured
        the junk; the target is measured here and the fixup reads both outcome vectors."""
        uniform, _, erase = self.work_sizes()
        for a, i in zip(alt, index, strict=True):
            cswap(less_than[0], a, i, work[0])
        qp.adjoint(qp.LeftQuantumComparator(keep, m, less_than, cmp_work, COMPARATOR))
        emit_erase_from_outcomes(junk_outcomes, table=table, **self.read,
                                 onehot_bits=self.onehot_bits, address=address, target=target,
                                 work=work[:erase], extra_onehot=extra_onehot)
        for wire in hadamards:
            qp.H(wire)
        qp.adjoint(FlaggedUniformPrep(self.L, self.s_u, index, succ, succ_work,
                                      phase_grad[:self.s_u], work[:uniform]))

    def _uniform_key(self, adjoint: bool):
        sizes = FlaggedUniformPrep.required_wires(self.L, self.s_u)

        def build():
            op = FlaggedUniformPrep(self.L, self.s_u, Wire[sizes["target"]], Wire[1],
                                    Wire[sizes["succ_work"]], Wire[self.s_u], Wire[sizes["work"]])
            return qp.adjoint(op) if adjoint else op

        return resource_key(build)

    def _comparator_key(self, adjoint: bool):
        n = self.b_k

        def build():
            op = qp.LeftQuantumComparator(Wire[n], Wire[n], Wire[1], Wire[n - 1], COMPARATOR)
            return qp.adjoint(op) if adjoint else op

        return resource_key(build)

    def prepare_resources(self) -> dict:
        columns = Bool[QROAMRead.columns_shape(**self.read)]
        sizes = QROAMRead.required_wires(**self.read, control_value=None)
        read = resource_key(lambda: QROAMRead(columns, **self.read, control_value=None,
                                              **{name: Wire[n] for name, n in sizes.items()}))
        return {self._uniform_key(False): 1, qp.H: self.b_k + 1, read: 1,
                self._comparator_key(False): 1, **cswap_resources(lg(self.L))}

    def unprepare_resources(self, table) -> dict:
        sizes = QROAMErase.required_wires(table, **self.read, control_value=None,
                                          onehot_bits=self.onehot_bits)
        erase = resource_key(lambda: QROAMErase(table, **self.read, control_value=None,
                                                onehot_bits=self.onehot_bits,
                                                **{name: Wire[n] for name, n in sizes.items()}))
        return {**cswap_resources(lg(self.L)), self._comparator_key(True): 1, erase: 1,
                qp.H: self.b_k + 1, self._uniform_key(True): 1}

    def unprepare_from_outcomes_resources(self, table) -> dict:
        """The declaration of :meth:`unprepare_from_outcomes`: the swaps, the comparator's
        adjoint, the Hadamards and the adjoint uniform preparation as keys, and the split erase
        gate by gate, counted by running it."""
        _, _, erase = self.work_sizes()
        counter = QROAMEmitter(counting=True)
        emit_erase_from_outcomes(
            [None] * self.sizes()["junk"], table=table, **self.read, onehot_bits=self.onehot_bits,
            address=range(sum(self.read["address_widths"])), target=range(self.read["width"]),
            work=range(erase), extra_onehot=range(self.extra_onehot()), emit=counter)
        counts = Counter({**cswap_resources(lg(self.L)), self._comparator_key(True): 1,
                          qp.H: self.b_k + 1, self._uniform_key(True): 1})
        counts.update(as_resources(counter.counts))
        return dict(counts)


# --------------------------------------------------------------------------- OuterPrepare


def _check_outer(X_o, alt, keep, b_k1, k1, ke1, s_u) -> tuple:
    X_o = check_int("X_o", X_o, low=2)
    b_k1 = check_int("b_k1", b_k1, low=1)
    k1 = check_int("k1", k1, low=0, high=lg(X_o))
    ke1 = check_int("ke1", ke1, low=0, high=lg(X_o))
    s_u = check_int("s_u", s_u, low=3)
    alt = _check_table("alt", alt, (X_o,), X_o)
    keep = _check_table("keep", keep, (X_o,), 2**b_k1)
    _check_onehot("ke1", ke1, OuterPrepare.read_shape(X_o, b_k1, k1))
    return X_o, alt, keep, b_k1, k1, ke1, s_u


def _outer(X_o, b_k1, k1, ke1, s_u) -> _AliasSampling:
    return _AliasSampling(X_o, s_u, b_k1, OuterPrepare.read_shape(X_o, b_k1, k1), ke1)


def _outer_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                     alt_x_o, keep_x_o, junk_x_o, phase_grad, work) -> dict:
    """The skeleton's registers for the outer prepare: the read's address is ``x_o``."""
    return {"index": x_o, "succ": succ_x_o, "succ_work": succ_work_x_o,
            "hadamards": _join(m_x_o, sigma_1), "m": m_x_o, "less_than": less_than_x_o,
            "cmp_work": cmp_work_x_o, "alt": alt_x_o, "keep": keep_x_o, "address": x_o,
            "target": _join(alt_x_o, keep_x_o), "junk": junk_x_o, "phase_grad": phase_grad,
            "work": work}


_OUTER_WIRES = ("x_o", "succ_x_o", "succ_work_x_o", "sigma_1", "m_x_o", "less_than_x_o",
                "cmp_work_x_o", "alt_x_o", "keep_x_o", "junk_x_o", "phase_grad", "work")


class OuterPrepare(Operator2):
    r"""Prepare the outer index ``x_o`` with the alias distribution of the outer weights
    (steps 1 to 4).

    **Action.** From ``|0>`` on every register, with ``phase_grad`` holding the phase-gradient
    state, let :math:`x_0` be the value the uniform preparation draws on ``x_o``,
    :math:`\tilde p` = ``alias_probabilities(alt, keep, b_k1)`` (spec 6.4) and
    :math:`P_f` = ``uniform_failure_probability(X_o, s_u)``. The output is a sum of basis terms,
    one per :math:`x_0`, value of ``m_x_o`` and value of ``sigma_1``. A term with
    :math:`x_0 < X_o` has ``succ_x_o = 1`` and the amplitude
    :math:`e^{i\varphi}\sqrt{(1 - P_f)/X_o}\,/\sqrt{2^{b_{k1} + 1}}`, where :math:`\varphi` is the
    one phase that :class:`~dfthc.ops.uniform.FlaggedUniformPrep` gives all its ``succ = 1``
    amplitudes (0 when ``X_o`` is a power of two): steps 2 to 5 permute basis states and add no
    phase. So the ``succ_x_o = 1`` part is
    :math:`e^{i\varphi}\sqrt{1 - P_f}\sum_x \sqrt{\tilde p(x)}\,|x\rangle|\mathrm{garb}_x\rangle
    \otimes |+\rangle_{\sigma_1}` with normalized garbage states, and ``x_o`` holds :math:`x` with
    probability :math:`(1 - P_f)\tilde p(x)`. The failure part, of weight :math:`P_f`, comes from
    :math:`x_0 \ge X_o`, where the selected target entry ``alt_x_o ‖ keep_x_o`` is zero (the junk
    may hold other entries of a visited block): ``less_than = 1``, and the swap leaves ``x_o = 0``
    and :math:`x_0` in ``alt_x_o``, with ``succ_x_o = 0``.

    **Registers.** ``w_x = lg(X_o)``, ``l_x`` the odd part of ``X_o``, ``K1 = 2**k1``.

    ================= ============================= ============== ===============================
    name              width                         input          output
    ================= ============================= ============== ===============================
    ``x_o``           ``w_x``                       ``|0>``        the sampled index
    ``succ_x_o``      1                             ``|0>``        ``[x_0 < X_o]``
    ``succ_work_x_o`` ``max(lg(l_x) - 1, 0)``       ``|0>``        dirty until the unprepare
    ``sigma_1``       1                             ``|0>``        ``|+>``
    ``m_x_o``         ``b_k1``                      ``|0>``        uniform over ``[0, 2**b_k1)``
    ``less_than_x_o`` 1                             ``|0>``        ``[keep[x_0] <= m]``
    ``cmp_work_x_o``  ``b_k1 - 1``                  ``|0>``        dirty until the unprepare
    ``alt_x_o``       ``w_x``                       ``|0>``        ``alt[x_0]``, or ``x_0`` if
                                                                   swapped
    ``keep_x_o``      ``b_k1``                      ``|0>``        ``keep[x_0]``
    ``junk_x_o``      ``(K1 - 1)(w_x + b_k1)``      ``|0>``        the read block's other entries
    ``phase_grad``    at least ``s_u``              gradient state unchanged (catalysed)
    ``work``          :meth:`required_wires`        ``|0>``        ``|0>``
    ================= ============================= ============== ===============================

    **Compilable arguments.** ``X_o`` (int >= 2, the values of ``x_o``); ``alt`` (``X_o`` ints in
    ``[0, X_o)``) and ``keep`` (``X_o`` ints in ``[0, 2**b_k1)``), the outer alias tables
    (``WalkTables.outer_alt``, ``outer_keep``); ``b_k1`` (int >= 1); ``k1`` (the read's block
    bits, ``0 <= k1 <= w_x``); ``ke1`` (the erase's one-hot bits, ``0 <= ke1 <= w_x``, with
    ``2**ke1 <= K1 (w_x + b_k1)``; used by :class:`OuterUnprepare`, checked here so that one
    argument tuple serves both); ``s_u`` (int >= 3, the uniform preparation's rotation bits).

    **Pre- and postconditions.** As in the table. Until the unprepare, nothing may write
    ``succ_x_o``, ``succ_work_x_o``, ``less_than_x_o``, ``cmp_work_x_o``, ``m_x_o``, ``alt_x_o``
    or ``keep_x_o``, and ``x_o`` may be used only as a control or changed by operations that are
    undone before the unprepare. ``junk_x_o`` takes one of two paths: (a) nothing writes it until
    :class:`OuterUnprepare`; or (b) the caller applies ``H`` and ``qp.measure(q, reset=True)`` to
    each of its qubits right after the prepare, keeps the outcomes in wire order (the slots of
    :meth:`QROAMRead.junk_layout <dfthc.ops.qroam.QROAMRead.junk_layout>`), may use its wires for
    other registers in between, and calls :class:`OuterUnprepareFromOutcomes` with the outcomes
    (spec 16, Q-1 variant (b); the block encoding does).

    **Decomposition** (spec 9.7). ``FlaggedUniformPrep(X_o, s_u)`` on ``x_o``, ``succ_x_o``,
    ``succ_work_x_o``, ``phase_grad[:s_u]``; ``H`` on ``m_x_o`` and ``sigma_1``; ``QROAMRead`` of
    :meth:`read_table` at ``x_o`` (shape :meth:`read_shape`, no enable) into
    ``alt_x_o ‖ keep_x_o`` and ``junk_x_o``, its ``columns`` built here from the table;
    ``LeftQuantumComparator(keep_x_o, m_x_o, less_than_x_o, cmp_work_x_o, "<=")``; ``w_x``
    ``CSWAP``\s of ``alt_x_o`` and ``x_o`` controlled by ``less_than_x_o``. Each child gets the
    start of ``work`` it needs.

    **Adjoint and control.** Never adjointed or controlled: the read has loops and data
    conditionals (spec 7.4 rules 4 and 6). The inverse is the operator :class:`OuterUnprepare`
    (path (a) above) or :class:`OuterUnprepareFromOutcomes` (path (b)).

    **Cost.** :func:`dfthc.costs.outer_prepare`: :math:`T = F(X_o) + T_{read,out} + b_{k1} +
    w_x` (uniform preparation, read, comparator, swaps), ``M = 0``; ``A`` is the uniform
    preparation's :math:`2(n - 1) + 2(s_u - 1)` (0 if ``l_x = 1``; ``n = lg(l_x)``) plus the
    read's :math:`T_{read,out}` and the swaps' :math:`w_x` (``T`` less the ANDs the unprepare
    uncomputes). At FeMoco-54: :math:`30 + 155 + 15 + 9 = 209`, ``A = 24 + 155 + 9 = 188``. The
    ``CNOT`` and
    ``PauliX`` counts are upper bounds: the read's ``QROM`` declares a load gate for every bit
    (spec 9.3).

    **Paper and dfthcbe.** Steps 1 to 4 of dfthcbe's README circuit, its ``OuterPrepare``
    (``dfthcbe/prepare.py``), with PennyLane's ``LeftQuantumComparator`` in place of
    ``LinearDepthGreaterThan`` and its ``X``; the outer superposition of arXiv:2502.15882v1
    App. B.1 and B.7.2, steps 1 and 2, Eqs. (97) and (100) to (102) (Table 7).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import OuterPrepare

        sizes = OuterPrepare.required_wires(5, 2, 1, 1, 3)  # X_o, b_k1, k1, ke1, s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        program = lambda: OuterPrepare(5, (1, 0, 0, 2, 3), (3, 2, 1, 0, 2), 2, 1, 1, 3, **wires)
        compiled_resources(program, num_wires=start).toffoli  # F 10 + read 1 + 5, + 2 + 3 = 21
    """

    compilable_argnames = ("X_o", "alt", "keep", "b_k1", "k1", "ke1", "s_u")
    wire_argnames = _OUTER_WIRES

    def __init__(self, X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o,
                 sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o,
                 phase_grad, work):
        args = _check_outer(X_o, alt, keep, b_k1, k1, ke1, s_u)
        wires = (x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                 alt_x_o, keep_x_o, junk_x_o, phase_grad, work)
        check_wires("OuterPrepare", dict(zip(_OUTER_WIRES, wires, strict=True)),
                    self.required_wires(args[0], *args[3:]), at_least=("phase_grad",))
        super().__init__(*args, *wires)

    @staticmethod
    def required_wires(X_o, b_k1, k1, ke1, s_u) -> dict[str, int]:
        """Register sizes; ``phase_grad`` is a minimum, every other size is exact. ``work`` is
        the largest need of the uniform preparation, the read and :class:`OuterUnprepare`'s
        erase, and at least 1 (the swaps' zeroed wire), so that both classes take the same
        wires."""
        sizes = _outer(X_o, b_k1, k1, ke1, s_u).sizes()
        return {"x_o": sizes["index"], "succ_x_o": 1, "succ_work_x_o": sizes["succ_work"],
                "sigma_1": 1, "m_x_o": b_k1, "less_than_x_o": 1, "cmp_work_x_o": b_k1 - 1,
                "alt_x_o": sizes["alt"], "keep_x_o": b_k1, "junk_x_o": sizes["junk"],
                "phase_grad": s_u, "work": sizes["work"]}

    @staticmethod
    def read_shape(X_o, b_k1, k1) -> dict:
        """The outer read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``:
        ``x_o`` over ``[0, X_o)`` on ``lg(X_o)`` bits, entries of ``d1 = lg(X_o) + b_k1`` bits."""
        return {"address_widths": (lg(X_o),), "live_ranges": ((0, X_o),),
                "width": lg(X_o) + b_k1, "block_bits": k1}

    @staticmethod
    def read_table(X_o, alt, keep, b_k1) -> tuple:
        """The outer read's table: ``(x, words)`` with entry ``(alt[x] << b_k1) | keep[x]`` for
        every ``x < X_o``, the target ``alt_x_o ‖ keep_x_o`` big-endian."""
        width = lg(X_o) + b_k1
        return tuple((x, _words((alt[x] << b_k1) | keep[x], width)) for x in range(X_o))


def _outer_prepare_resources(X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o,
                             sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o,
                             junk_x_o, phase_grad, work):
    return _outer(X_o, b_k1, k1, ke1, s_u).prepare_resources()


@qp.register_resources(_outer_prepare_resources)
def _outer_prepare(X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o, sigma_1,
                   m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o, phase_grad,
                   work):
    _outer(X_o, b_k1, k1, ke1, s_u).prepare(
        OuterPrepare.read_table(X_o, alt, keep, b_k1),
        **_outer_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                           cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o, phase_grad, work))


register_rule(OuterPrepare, _outer_prepare)


# --------------------------------------------------------------------------- OuterUnprepare


class OuterUnprepare(Operator2):
    r"""The measured unprepare of :class:`OuterPrepare` (steps 24 to 27): its explicit adjoint
    class (spec 7.4 rule 6).

    **Action.** ``OuterUnprepare`` after ``OuterPrepare`` with the same arguments is the
    identity on the prepare's input (every register ``|0>``, ``phase_grad`` unchanged), for every
    measurement outcome; operations in between must respect :class:`OuterPrepare`'s
    postconditions.

    **Registers.** Those of :class:`OuterPrepare`; the input is the prepare's output.

    ================= ============================= ============================== ==============
    name              width                         input                          output
    ================= ============================= ============================== ==============
    ``x_o``           ``w_x``                       the sampled index              ``|0>``
    ``succ_x_o``      1                             ``[x_0 < X_o]``                ``|0>``
    ``succ_work_x_o`` ``max(lg(l_x) - 1, 0)``       the prepare's dirty wires      ``|0>``
    ``sigma_1``       1                             ``|+>``                        ``|0>``
    ``m_x_o``         ``b_k1``                      uniform                        ``|0>``
    ``less_than_x_o`` 1                             ``[keep[x_0] <= m]``           ``|0>``
    ``cmp_work_x_o``  ``b_k1 - 1``                  the prepare's dirty wires      ``|0>``
    ``alt_x_o``       ``w_x``                       ``alt[x_0]``, or ``x_0``       ``|0>``
    ``keep_x_o``      ``b_k1``                      ``keep[x_0]``                  ``|0>``
    ``junk_x_o``      ``(K1 - 1)(w_x + b_k1)``      the read block's other entries ``|0>``
    ``phase_grad``    at least ``s_u``              gradient state                 unchanged
    ``work``          :meth:`required_wires`        ``|0>``                        ``|0>``
    ================= ============================= ============================== ==============

    **Compilable arguments.** Those of :class:`OuterPrepare`, with the same domains; ``ke1`` sets
    the erase's one-hot bits (spec 9.5).

    **Pre- and postconditions.** The registers hold what :class:`OuterPrepare` produced, up to
    operations that use ``x_o`` only as a control or are undone; ``work`` is ``|0>``. Afterwards
    every register but ``phase_grad`` is ``|0>``.

    **Decomposition** (spec 9.7). The ``w_x`` ``CSWAP``\s (self-inverse);
    ``qp.adjoint(LeftQuantumComparator(keep_x_o, m_x_o, less_than_x_o, cmp_work_x_o, "<="))``,
    whose rule is a measured uncompute (it also cleans ``cmp_work_x_o``); ``QROAMErase`` of
    :meth:`OuterPrepare.read_table` with ``onehot_bits = ke1``, no enable and no
    ``extra_onehot``; ``H`` on ``m_x_o`` and ``sigma_1``; ``qp.adjoint(FlaggedUniformPrep(X_o,
    s_u, ...))`` on the prepare's wires, the auto-derived leaf adjoint.

    **Adjoint and control.** Never adjointed or controlled: it measures (spec 7.4 rule 5), and it
    is itself the adjoint class of :class:`OuterPrepare`. The walk's block encoding measures the
    junk right after the prepare and calls the sibling :class:`OuterUnprepareFromOutcomes`
    instead (spec 16, Q-1 variant (b)); this class serves callers that keep the junk.

    **Cost.** :func:`dfthc.costs.outer_unprepare`: :math:`T = w_x + T_{erase,out} +
    F^\dagger(X_o)`, :math:`M = K_1 d_1` (every target and junk qubit); ``A`` is the swaps'
    ``w_x`` plus the comparator's ``b_k1`` plus the erase's ``T`` plus the adjoint uniform
    preparation's
    :math:`3(n - 1) + 2(s_u - 1)` (0 if ``l_x = 1``). At FeMoco-54: :math:`9 + 36 + 24 = 69`,
    ``M = 96``. The ``PauliX`` count is an upper bound: each measurement's reset is counted at
    its maximum (spec 7.6).

    **Paper and dfthcbe.** Steps 24 to 27 of dfthcbe's README circuit, the adjoint of its
    ``OuterPrepare`` with the measured erase; arXiv:2502.15882v1 App. B.7.2, steps 10 and 11,
    Eqs. (126) and (127) (Table 7). Our comparator adjoint costs no Toffoli (the paper's and
    dfthcbe's ``b_k1`` term): the comparator's ``b_k1 - 1`` work wires stay dirty until here,
    the saving the paper notes after Eq. (127).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import OuterUnprepare

        sizes = OuterUnprepare.required_wires(5, 2, 1, 1, 3)  # X_o, b_k1, k1, ke1, s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        program = lambda: OuterUnprepare(5, (1, 0, 0, 2, 3), (3, 2, 1, 0, 2), 2, 1, 1, 3,
                                         **wires)
        compiled_resources(program, num_wires=start).measurements  # K1 d1 = 2 * 5 = 10
    """

    compilable_argnames = ("X_o", "alt", "keep", "b_k1", "k1", "ke1", "s_u")
    wire_argnames = _OUTER_WIRES

    def __init__(self, X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o,
                 sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o,
                 phase_grad, work):
        args = _check_outer(X_o, alt, keep, b_k1, k1, ke1, s_u)
        wires = (x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                 alt_x_o, keep_x_o, junk_x_o, phase_grad, work)
        check_wires("OuterUnprepare", dict(zip(_OUTER_WIRES, wires, strict=True)),
                    self.required_wires(args[0], *args[3:]), at_least=("phase_grad",))
        super().__init__(*args, *wires)

    @staticmethod
    def required_wires(X_o, b_k1, k1, ke1, s_u) -> dict[str, int]:
        """:meth:`OuterPrepare.required_wires`."""
        return OuterPrepare.required_wires(X_o, b_k1, k1, ke1, s_u)


def _outer_unprepare_resources(X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o,
                               succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                               alt_x_o, keep_x_o, junk_x_o, phase_grad, work):
    return _outer(X_o, b_k1, k1, ke1, s_u).unprepare_resources(
        OuterPrepare.read_table(X_o, alt, keep, b_k1))


@qp.register_resources(_outer_unprepare_resources)
def _outer_unprepare(X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o, sigma_1,
                     m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o,
                     phase_grad, work):
    _outer(X_o, b_k1, k1, ke1, s_u).unprepare(
        OuterPrepare.read_table(X_o, alt, keep, b_k1),
        **_outer_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                           cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o, phase_grad, work))


register_rule(OuterUnprepare, _outer_unprepare)


# ----------------------------------------------------------------------- OuterUnprepareFromOutcomes

_OUTER_FROM_OUTCOMES_WIRES = ("x_o", "succ_x_o", "succ_work_x_o", "sigma_1", "m_x_o",
                              "less_than_x_o", "cmp_work_x_o", "alt_x_o", "keep_x_o",
                              "phase_grad", "work", "extra_onehot")


class OuterUnprepareFromOutcomes(Operator2):
    r"""The measured unprepare of :class:`OuterPrepare` (steps 24 to 27) for a caller that
    X-measured ``junk_x_o`` right after the prepare (spec 16, Q-1 variant (b)).

    **Action.** After ``OuterPrepare``, ``H`` and ``qp.measure(q, reset=True)`` on every qubit
    of ``junk_x_o`` (the outcomes :math:`\xi_{junk}`) and operations that respect
    :class:`OuterPrepare`'s postconditions, ``OuterUnprepareFromOutcomes`` with the outcomes
    :math:`\xi_{junk}` and the prepare's compilable arguments is the identity on the prepare's
    input (every register ``|0>``, ``phase_grad`` unchanged), for every outcome of the junk and
    target measurements: :class:`OuterUnprepare` with its erase's junk measurements moved to the
    caller. The junk wires are free from their measurement on; the block encoding gives them to
    the angle registers (spec 10.2).

    **Registers.** Those of :class:`OuterPrepare` without ``junk_x_o``, then ``extra_onehot``.

    ================= ============================= ============================== ==============
    name              width                         input                          output
    ================= ============================= ============================== ==============
    ``x_o``           ``w_x``                       the sampled index              ``|0>``
    ``succ_x_o``      1                             ``[x_0 < X_o]``                ``|0>``
    ``succ_work_x_o`` ``max(lg(l_x) - 1, 0)``       the prepare's dirty wires      ``|0>``
    ``sigma_1``       1                             ``|+>``                        ``|0>``
    ``m_x_o``         ``b_k1``                      uniform                        ``|0>``
    ``less_than_x_o`` 1                             ``[keep[x_0] <= m]``           ``|0>``
    ``cmp_work_x_o``  ``b_k1 - 1``                  the prepare's dirty wires      ``|0>``
    ``alt_x_o``       ``w_x``                       ``alt[x_0]``, or ``x_0``       ``|0>``
    ``keep_x_o``      ``b_k1``                      ``keep[x_0]``                  ``|0>``
    ``phase_grad``    at least ``s_u``              gradient state                 unchanged
    ``work``          :meth:`required_wires`        ``|0>``                        ``|0>``
    ``extra_onehot``  ``max(2**ke1 - d1, 0)``       ``|0>``                        ``|0>``
    ================= ============================= ============================== ==============

    **Arguments.** Dynamic: ``junk_outcomes``, the :math:`(K_1 - 1) d_1` outcomes of the junk
    measurements in the order of the ``junk_x_o`` wires: slot ``r`` of
    :meth:`QROAMRead.junk_layout <dfthc.ops.qroam.QROAMRead.junk_layout>` gives outcomes
    ``r d1`` to ``(r + 1) d1 - 1`` (``d1 = w_x + b_k1``). Under capture it becomes a bool array;
    in tape mode every entry must be a ``MeasurementValue``, so both modes emit the same fixup
    (K-10). Compilable: those of :class:`OuterPrepare`, with the same domains; ``ke1`` sets the
    fixup's one-hot bits (spec 9.5).

    **Pre- and postconditions.** The registers hold what :class:`OuterPrepare` produced, up to
    operations that use ``x_o`` only as a control or are undone; ``junk_x_o`` was X-measured with
    reset right after the prepare, ``junk_outcomes`` are its outcomes in wire order, and its
    wires may serve other registers in between; ``work`` and ``extra_onehot`` are ``|0>``.
    Afterwards every register but ``phase_grad`` is ``|0>``.

    **Decomposition** (spec 16, Q-1 variant (b)). The ``w_x`` ``CSWAP``\s;
    ``qp.adjoint(LeftQuantumComparator(keep_x_o, m_x_o, less_than_x_o, cmp_work_x_o, "<="))``
    (a measured uncompute); ``H`` and ``qp.measure(q, reset=True)`` on every qubit of
    ``alt_x_o ‖ keep_x_o``; the parities :math:`f(a) = \xi \cdot g(a)` of the outcomes
    :math:`\xi` = target ‖ ``junk_outcomes`` with :math:`g(a)` =
    :meth:`QROAMRead.loaded_bits <dfthc.ops.qroam.QROAMRead.loaded_bits>` of
    :meth:`OuterPrepare.read_table` over the read's loaded set (masks built here from the
    compilable tables, never passed as data), by :func:`dfthc.compile.mcm_parities`; the fixup
    over that set with the reset ``alt_x_o ‖ keep_x_o ‖ extra_onehot`` as its one-hot register
    (:func:`dfthc.ops.qroam.emit_erase_from_outcomes`); ``H`` on ``m_x_o`` and ``sigma_1``;
    ``qp.adjoint(FlaggedUniformPrep(X_o, s_u, ...))``, the auto-derived leaf adjoint.

    **Adjoint and control.** Never adjointed or controlled: it measures (spec 7.4 rule 5). It is
    a sibling of :class:`OuterUnprepare`, the same inverse of :class:`OuterPrepare` with another
    signature, not a modified form of a composite (spec 7.4 rule 6).

    **Cost.** :func:`dfthc.costs.outer_unprepare_from_outcomes`: the ``T`` of
    :func:`dfthc.costs.outer_unprepare`, :math:`w_x + T_{erase,out} + F^\dagger(X_o)`, and
    ``M = d1`` (the target qubits; the caller's :math:`(K_1 - 1) d_1` junk measurements make up
    :class:`OuterUnprepare`'s :math:`K_1 d_1`); ``A`` as :class:`OuterUnprepare`'s, the swaps'
    ``w_x`` plus the comparator's ``b_k1`` plus the fixup's ``T`` plus the adjoint uniform
    preparation's
    :math:`3(n - 1) + 2(s_u - 1)`. At FeMoco-54: :math:`9 + 36 + 24 = 69`, ``M = 24`` (and 72
    junk measurements in the block encoding). The ``PauliX`` count is an upper bound (spec 7.6).

    **Paper and dfthcbe.** Steps 24 to 27 of dfthcbe's README circuit with the junk measured
    early, as the paper does: arXiv:2502.15882v1 App. B.7.2, step 2 (":math:`(2^{k_1} - 1)
    b_1` qubits are erased (via an X measurement and a later phase fixup)", after Eq. (104)) and
    steps 10 and 11, Eqs. (126) and (127) (Table 7). The walk measures the junk right after
    step 4, the end of the prepare, before any angle load; the comparator and the swaps of steps
    3 and 4 do not touch it, so this equals measuring right after the read (step 2). dfthcbe's
    adjoint ``OuterPrepare`` measures target and junk together at the erase, as
    :class:`OuterUnprepare` does.

    **Example.**

    .. code-block:: python

        import numpy as np
        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import OuterUnprepareFromOutcomes

        sizes = OuterUnprepareFromOutcomes.required_wires(5, 2, 1, 1, 3)  # X_o b_k1 k1 ke1 s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        junk = np.zeros(OuterUnprepareFromOutcomes.junk_size(5, 2, 1), dtype=bool)
        program = lambda: OuterUnprepareFromOutcomes(junk, 5, (1, 0, 0, 2, 3), (3, 2, 1, 0, 2),
                                                     2, 1, 1, 3, **wires)
        compiled_resources(program, num_wires=start).measurements  # d1 = 5
    """

    dynamic_argnames = ("junk_outcomes",)
    compilable_argnames = ("X_o", "alt", "keep", "b_k1", "k1", "ke1", "s_u")
    wire_argnames = _OUTER_FROM_OUTCOMES_WIRES

    def __init__(self, junk_outcomes, X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o,
                 succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o,
                 phase_grad, work, extra_onehot):
        args = _check_outer(X_o, alt, keep, b_k1, k1, ke1, s_u)
        junk_outcomes = _junk_outcomes(junk_outcomes, self.junk_size(args[0], args[3], args[4]))
        wires = (x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                 alt_x_o, keep_x_o, phase_grad, work, extra_onehot)
        check_wires("OuterUnprepareFromOutcomes",
                    dict(zip(_OUTER_FROM_OUTCOMES_WIRES, wires, strict=True)),
                    self.required_wires(args[0], *args[3:]), at_least=("phase_grad",))
        super().__init__(junk_outcomes, *args, *wires)

    @staticmethod
    def required_wires(X_o, b_k1, k1, ke1, s_u) -> dict[str, int]:
        """:meth:`OuterPrepare.required_wires` without ``junk_x_o``, then ``extra_onehot``:
        ``max(2**ke1 - d1, 0)`` wires (0 for ``ke1 = 0``), the one-hot beyond the reset target."""
        sizes = OuterPrepare.required_wires(X_o, b_k1, k1, ke1, s_u)
        del sizes["junk_x_o"]
        return sizes | {"extra_onehot": _outer(X_o, b_k1, k1, ke1, s_u).extra_onehot()}

    @staticmethod
    def junk_size(X_o, b_k1, k1) -> int:
        """The number of junk outcomes: ``(K1 - 1) d1``, the width of ``junk_x_o``."""
        return (2**k1 - 1) * (lg(X_o) + b_k1)


def _outer_from_outcomes_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                                   cmp_work_x_o, alt_x_o, keep_x_o, phase_grad, work,
                                   extra_onehot) -> dict:
    registers = _outer_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                                 cmp_work_x_o, alt_x_o, keep_x_o, (), phase_grad, work)
    del registers["junk"]
    return registers | {"extra_onehot": extra_onehot}


def _outer_unprepare_from_outcomes_resources(junk_outcomes, X_o, alt, keep, b_k1, k1, ke1, s_u,
                                             x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o,
                                             less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o,
                                             phase_grad, work, extra_onehot):
    return _outer(X_o, b_k1, k1, ke1, s_u).unprepare_from_outcomes_resources(
        OuterPrepare.read_table(X_o, alt, keep, b_k1))


@qp.register_resources(_outer_unprepare_from_outcomes_resources)
def _outer_unprepare_from_outcomes(junk_outcomes, X_o, alt, keep, b_k1, k1, ke1, s_u, x_o,
                                   succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                                   cmp_work_x_o, alt_x_o, keep_x_o, phase_grad, work,
                                   extra_onehot):
    size = OuterUnprepareFromOutcomes.junk_size(X_o, b_k1, k1)
    _outer(X_o, b_k1, k1, ke1, s_u).unprepare_from_outcomes(
        [junk_outcomes[i] for i in range(size)], OuterPrepare.read_table(X_o, alt, keep, b_k1),
        **_outer_from_outcomes_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o,
                                         less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o,
                                         phase_grad, work, extra_onehot))


register_rule(OuterUnprepareFromOutcomes, _outer_unprepare_from_outcomes)


# --------------------------------------------------------------------------- InnerPrepare


def _check_inner(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u) -> tuple:
    N = check_int("N", N, low=1)
    R = check_int("R", R, low=1)
    B = check_int("B", B, low=1)
    C = check_int("C", C, low=1)
    b_k2 = check_int("b_k2", b_k2, low=1)
    k2 = check_int("k2", k2, low=0, high=lg(B + 1))
    ke2 = check_int("ke2", ke2, low=0, high=lg(R * C))
    s_u = check_int("s_u", s_u, low=3)
    shape = (R * C, B + 1)
    alt = _check_table("alt", alt, shape, B + 1)
    keep = _check_table("keep", keep, shape, 2**b_k2)
    sgn_table = _check_table("sgn_table", sgn_table, shape, 2)
    d_sgn_table = _check_table("d_sgn_table", d_sgn_table, shape, 2)
    _check_onehot("ke2 + lg(B + 1)", ke2 + lg(B + 1), InnerPrepare.read_shape(N, R, B, C, b_k2, k2))
    return N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u


def _inner(N, R, B, C, b_k2, k2, ke2, s_u) -> _AliasSampling:
    return _AliasSampling(B + 1, s_u, b_k2, InnerPrepare.read_shape(N, R, B, C, b_k2, k2),
                          ke2 + lg(B + 1))


def _inner_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b,
                     keep_b, sgn, d_sgn, r, junk_b, phase_grad, work) -> dict:
    """The skeleton's registers for the inner prepare: the read's address is ``x_o ‖ b``."""
    return {"index": b, "succ": succ_b, "succ_work": succ_work_b,
            "hadamards": _join(sigma_0, m_b), "m": m_b, "less_than": less_than_b,
            "cmp_work": cmp_work_b, "alt": alt_b, "keep": keep_b, "address": _join(x_o, b),
            "target": _join(alt_b, keep_b, sgn, d_sgn, r), "junk": junk_b,
            "phase_grad": phase_grad, "work": work}


_INNER_WIRES = ("x_o", "b", "succ_b", "succ_work_b", "sigma_0", "m_b", "less_than_b",
                "cmp_work_b", "alt_b", "keep_b", "sgn", "d_sgn", "r", "junk_b", "phase_grad",
                "work")


class InnerPrepare(Operator2):
    r"""Prepare the inner index ``b`` with the alias distribution of row ``x_o`` of the inner
    weights, and load the signs and the rank (steps 5 to 8).

    **Action.** ``x_o`` is a read-only address; every other register starts in ``|0>``, with
    ``phase_grad`` holding the phase-gradient state. Let :math:`b_0` be the value the uniform
    preparation draws on ``b`` and :math:`P_f` = ``uniform_failure_probability(B + 1, s_u)``. The
    output is a sum of basis terms, one per :math:`b_0`, value of ``m_b`` and value of
    ``sigma_0``, each with the amplitude of :math:`b_0` in
    :class:`~dfthc.ops.uniform.FlaggedUniformPrep` over :math:`\sqrt{2^{b_{k2} + 1}}` (one common
    success phase :math:`e^{i\varphi}` on :math:`b_0 \le B`): steps 2 to 5 permute basis states
    and add no phase. On an SF row ``x_o = x < R C`` the ``succ_b = 1`` part is
    :math:`e^{i\varphi}\sqrt{1 - P_f}\sum_b \sqrt{\tilde q_x(b)}\,|b\rangle|\mathrm{garb}_{x,b}
    \rangle \otimes |+\rangle_{\sigma_0}` with :math:`\tilde q_x` =
    ``alias_probabilities(alt[x], keep[x], b_k2)``, and ``sgn``, ``d_sgn`` and ``r = x // C``
    hold the drawn entry's columns. Where the selected target entry is zero (one-body rows
    :math:`R C \le x < N + R C`, :math:`x \ge X_o`, and failed :math:`b_0 > B`, which have
    ``succ_b = 0``; the junk may hold other entries of a visited block): ``keep = 0``, so
    ``less_than_b = 1``, the swap leaves ``b = 0`` and :math:`b_0` in ``alt_b``, and
    ``sgn = d_sgn = r = 0`` (spec 8.5). :class:`InnerSigns` then applies the signs.
    :math:`x \ge X_o` is an input the walk never presents: ``OuterPrepare``'s swap maps a failed
    draw to ``x_o = 0``, an SF row, where the inner signs cancel between the two rounds and
    ``REF_T2`` gives ``-I``.

    **Registers.** ``w_x = lg(N + R C)``, ``w_b = lg(B + 1)``, ``w_r = lg(R)``, ``l_b`` the odd
    part of ``B + 1``, ``K2 = 2**k2``, ``d2 = w_b + b_k2 + 2 + w_r``.

    =============== =========================== ============== =================================
    name            width                       input          output
    =============== =========================== ============== =================================
    ``x_o``         ``w_x``                     any            unchanged
    ``b``           ``w_b``                     ``|0>``        the sampled index
    ``succ_b``      1                           ``|0>``        ``[b_0 <= B]``
    ``succ_work_b`` ``max(lg(l_b) - 1, 0)``     ``|0>``        dirty until the unprepare
    ``sigma_0``     1                           ``|0>``        ``|+>``
    ``m_b``         ``b_k2``                    ``|0>``        uniform over ``[0, 2**b_k2)``
    ``less_than_b`` 1                           ``|0>``        ``[keep <= m_b]``
    ``cmp_work_b``  ``b_k2 - 1``                ``|0>``        dirty until the unprepare
    ``alt_b``       ``w_b``                     ``|0>``        the loaded ``alt``, or ``b_0`` if
                                                               swapped
    ``keep_b``      ``b_k2``                    ``|0>``        the loaded ``keep``
    ``sgn``         1                           ``|0>``        the loaded ``sgn``
    ``d_sgn``       1                           ``|0>``        the loaded ``d_sgn``
    ``r``           ``w_r``                     ``|0>``        the loaded rank, ``x_o // C``
    ``junk_b``      ``(K2 - 1) d2``             ``|0>``        the read block's other entries
    ``phase_grad``  at least ``s_u``            gradient state unchanged (catalysed)
    ``work``        :meth:`required_wires`      ``|0>``        ``|0>``
    =============== =========================== ============== =================================

    **Compilable arguments.** ``N``, ``R``, ``B``, ``C`` (ints >= 1; ``x_o`` takes ``N + R C``
    values); ``alt`` (ints in ``[0, B + 1)``), ``keep`` (ints in ``[0, 2**b_k2)``),
    ``sgn_table`` and ``d_sgn_table`` (bits), each of shape ``(R C, B + 1)``: the inner tables
    ``WalkTables.inner_alt``, ``inner_keep``, ``inner_sgn``, ``inner_d_sgn``, named
    ``sgn_table`` and ``d_sgn_table`` because ``sgn`` and ``d_sgn`` name the wire registers;
    ``b_k2`` (int >= 1); ``k2`` (the read's block bits on ``b``, ``0 <= k2 <= w_b``); ``ke2`` (the
    erase's one-hot bits of ``x_o``, ``0 <= ke2 <= lg(R C)``; the one-hot also spans all ``w_b``
    bits of ``b``, and ``2**(ke2 + w_b) <= K2 d2``; used by :class:`InnerUnprepare`, checked here
    so that one argument tuple serves both); ``s_u`` (int >= 3).

    **Pre- and postconditions.** As in the table. Until the unprepare, nothing may write ``x_o``,
    ``succ_b``, ``succ_work_b``, ``less_than_b``, ``cmp_work_b``, ``m_b``, ``alt_b``, ``keep_b``,
    ``sgn``, ``d_sgn`` or ``r``, and ``b`` may be used only as a control or changed by
    operations that are undone before the unprepare. ``junk_b`` takes one of two paths: (a)
    nothing writes it until :class:`InnerUnprepare`; or (b) the caller applies ``H`` and
    ``qp.measure(q, reset=True)`` to each of its qubits right after the prepare, keeps the
    outcomes in wire order (the slots of
    :meth:`QROAMRead.junk_layout <dfthc.ops.qroam.QROAMRead.junk_layout>`), may use its wires for
    other registers in between, and calls :class:`InnerUnprepareFromOutcomes` with the outcomes
    (spec 16, Q-1 variant (b); the block encoding does, once per inner round).

    **Decomposition** (spec 9.8). ``FlaggedUniformPrep(B + 1, s_u)`` on ``b``, ``succ_b``,
    ``succ_work_b``, ``phase_grad[:s_u]`` (unconditionally, as in dfthcbe); ``H`` on ``sigma_0``
    and ``m_b``; ``QROAMRead`` of :meth:`read_table` at ``x_o ‖ b`` (shape :meth:`read_shape`, no
    enable) into ``alt_b ‖ keep_b ‖ sgn ‖ d_sgn ‖ r`` and ``junk_b``, its ``columns`` built
    here from the tables; ``LeftQuantumComparator(keep_b, m_b, less_than_b, cmp_work_b, "<=")``;
    ``w_b`` ``CSWAP``\s of ``alt_b`` and ``b`` controlled by ``less_than_b``.

    **Adjoint and control.** Never adjointed or controlled: the read has loops and data
    conditionals (spec 7.4 rules 4 and 6). The inverse is the operator :class:`InnerUnprepare`
    (path (a) above) or :class:`InnerUnprepareFromOutcomes` (path (b)).

    **Cost.** :func:`dfthc.costs.inner_prepare`: :math:`T = F(B + 1) + T_{read,in} + b_{k2} +
    w_b`, ``M = 0``; ``A`` is the uniform preparation's :math:`2(n - 1) + 2(s_u - 1)` (0 if
    ``l_b = 1``; ``n = lg(l_b)``) plus the read's :math:`T_{read,in}` and the swaps' ``w_b``.
    At FeMoco-54:
    :math:`18 + 932 + 15 + 5 = 970`, where the read carries ``r`` through its swap network (+60
    against dfthcbe's flat ``r``, spec 9.3). The ``CNOT`` and ``PauliX`` counts are upper bounds:
    the read's ``QROM`` declares a load gate for every bit.

    **Paper and dfthcbe.** Steps 5 to 8 of dfthcbe's README circuit, its ``InnerPrepare``
    (``dfthcbe/prepare.py``), with PennyLane's ``LeftQuantumComparator`` and with ``r`` in every
    entry; the superposition over ``b`` of arXiv:2502.15882v1 App. B.4 and B.7.2, steps 3 and 4,
    Eqs. (105) and (108) to (110) (Table 8).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import InnerPrepare

        sizes = InnerPrepare.required_wires(2, 2, 2, 1, 2, 1, 1, 3)  # N R B C b_k2 k2 ke2 s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        tables = (((2, 0, 0), (1, 1, 0)), ((3, 1, 2), (0, 2, 3)), ((0, 1, 0), (1, 0, 0)),
                  ((1, 1, 0), (1, 1, 1)))
        program = lambda: InnerPrepare(2, 2, 2, 1, *tables, 2, 1, 1, 3, **wires)
        compiled_resources(program, num_wires=start).toffoli  # 7 + (3 + 7) + 2 + 2 = 21
    """

    compilable_argnames = ("N", "R", "B", "C", "alt", "keep", "sgn_table", "d_sgn_table", "b_k2",
                           "k2", "ke2", "s_u")
    wire_argnames = _INNER_WIRES

    def __init__(self, N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b,
                 succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b, sgn,
                 d_sgn, r, junk_b, phase_grad, work):
        args = _check_inner(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u)
        wires = (x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b,
                 keep_b, sgn, d_sgn, r, junk_b, phase_grad, work)
        check_wires("InnerPrepare", dict(zip(_INNER_WIRES, wires, strict=True)),
                    self.required_wires(*args[:4], *args[8:]), at_least=("phase_grad",))
        super().__init__(*args, *wires)

    @staticmethod
    def required_wires(N, R, B, C, b_k2, k2, ke2, s_u) -> dict[str, int]:
        """Register sizes; ``phase_grad`` is a minimum, every other size is exact. ``work`` is
        the largest need of the uniform preparation, the read and :class:`InnerUnprepare`'s
        erase, and at least 1 (the swaps' zeroed wire), so that both classes take the same
        wires."""
        sizes = _inner(N, R, B, C, b_k2, k2, ke2, s_u).sizes()
        return {"x_o": lg(N + R * C), "b": sizes["index"], "succ_b": 1,
                "succ_work_b": sizes["succ_work"], "sigma_0": 1, "m_b": b_k2,
                "less_than_b": 1, "cmp_work_b": b_k2 - 1, "alt_b": sizes["alt"],
                "keep_b": b_k2, "sgn": 1, "d_sgn": 1, "r": lg(R), "junk_b": sizes["junk"],
                "phase_grad": s_u, "work": sizes["work"]}

    @staticmethod
    def read_shape(N, R, B, C, b_k2, k2) -> dict:
        """The inner read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``:
        rows ``x_o`` in ``[0, R C)`` of ``lg(N + R C)`` bits, then ``b`` in ``[0, B + 1)`` of
        ``lg(B + 1)`` bits, entries of ``d2 = lg(B + 1) + b_k2 + 2 + lg(R)`` bits."""
        return {"address_widths": (lg(N + R * C), lg(B + 1)),
                "live_ranges": ((0, R * C), (0, B + 1)),
                "width": lg(B + 1) + b_k2 + 2 + lg(R), "block_bits": k2}

    @staticmethod
    def read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2) -> tuple:
        """The inner read's table: ``(x << lg(B + 1) | b, words)`` for every SF row ``x < R C``
        and every ``b <= B``, with the entry ``alt ‖ keep ‖ sgn ‖ d_sgn ‖ r`` (``r = x // C``)
        big-endian, as the target ``alt_b ‖ keep_b ‖ sgn ‖ d_sgn ‖ r`` holds it."""
        w_b, w_r = lg(B + 1), lg(R)
        width = w_b + b_k2 + 2 + w_r
        table = []
        for x in range(R * C):
            for b in range(B + 1):
                entry = (alt[x][b] << b_k2) | keep[x][b]
                entry = (entry << 2) | (sgn_table[x][b] << 1) | d_sgn_table[x][b]
                table.append(((x << w_b) | b, _words((entry << w_r) | (x // C), width)))
        return tuple(table)


def _inner_prepare_resources(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u,
                             x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                             alt_b, keep_b, sgn, d_sgn, r, junk_b, phase_grad, work):
    return _inner(N, R, B, C, b_k2, k2, ke2, s_u).prepare_resources()


@qp.register_resources(_inner_prepare_resources)
def _inner_prepare(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b,
                   succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b, sgn,
                   d_sgn, r, junk_b, phase_grad, work):
    _inner(N, R, B, C, b_k2, k2, ke2, s_u).prepare(
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2),
        **_inner_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                           alt_b, keep_b, sgn, d_sgn, r, junk_b, phase_grad, work))


register_rule(InnerPrepare, _inner_prepare)


# --------------------------------------------------------------------------- InnerUnprepare


class InnerUnprepare(Operator2):
    r"""The measured unprepare of :class:`InnerPrepare` (steps 19 to 22): its explicit adjoint
    class (spec 7.4 rule 6).

    **Action.** ``InnerUnprepare`` after ``InnerPrepare`` with the same arguments is the
    identity on the prepare's input (``x_o`` unchanged, every other register ``|0>``,
    ``phase_grad`` unchanged), for every measurement outcome; operations in between must respect
    :class:`InnerPrepare`'s postconditions (:class:`InnerSigns` does: it only applies phases).

    **Registers.** Those of :class:`InnerPrepare`; the input is the prepare's output.

    =============== =========================== ============================== =================
    name            width                       input                          output
    =============== =========================== ============================== =================
    ``x_o``         ``w_x``                     any                            unchanged
    ``b``           ``w_b``                     the sampled index              ``|0>``
    ``succ_b``      1                           ``[b_0 <= B]``                 ``|0>``
    ``succ_work_b`` ``max(lg(l_b) - 1, 0)``     the prepare's dirty wires      ``|0>``
    ``sigma_0``     1                           ``|+>``                        ``|0>``
    ``m_b``         ``b_k2``                    uniform                        ``|0>``
    ``less_than_b`` 1                           ``[keep <= m_b]``              ``|0>``
    ``cmp_work_b``  ``b_k2 - 1``                the prepare's dirty wires      ``|0>``
    ``alt_b``       ``w_b``                     the loaded ``alt``, or ``b_0`` ``|0>``
    ``keep_b``      ``b_k2``                    the loaded ``keep``            ``|0>``
    ``sgn``         1                           the loaded ``sgn``             ``|0>``
    ``d_sgn``       1                           the loaded ``d_sgn``           ``|0>``
    ``r``           ``w_r``                     the loaded rank                ``|0>``
    ``junk_b``      ``(K2 - 1) d2``             the read block's other entries ``|0>``
    ``phase_grad``  at least ``s_u``            gradient state                 unchanged
    ``work``        :meth:`required_wires`      ``|0>``                        ``|0>``
    =============== =========================== ============================== =================

    **Compilable arguments.** Those of :class:`InnerPrepare`, with the same domains; ``ke2`` sets
    the erase's one-hot bits, ``ke2 + lg(B + 1)`` (spec 9.5).

    **Pre- and postconditions.** The registers hold what :class:`InnerPrepare` produced, up to
    phases and operations that use ``b`` only as a control or are undone; ``work`` is ``|0>``.
    Afterwards every register but ``x_o`` and ``phase_grad`` is ``|0>``.

    **Decomposition** (spec 9.8). The ``w_b`` ``CSWAP``\s (self-inverse);
    ``qp.adjoint(LeftQuantumComparator(keep_b, m_b, less_than_b, cmp_work_b, "<="))``, whose rule
    is a measured uncompute (it also cleans ``cmp_work_b``); ``QROAMErase`` of
    :meth:`InnerPrepare.read_table` with ``onehot_bits = ke2 + lg(B + 1)`` (the low ``ke2`` bits
    of ``x_o`` and all of ``b``), no enable and no ``extra_onehot``; ``H`` on ``sigma_0`` and
    ``m_b``; ``qp.adjoint(FlaggedUniformPrep(B + 1, s_u, ...))`` on the prepare's wires, the
    auto-derived leaf adjoint.

    **Adjoint and control.** Never adjointed or controlled: it measures (spec 7.4 rule 5), and it
    is itself the adjoint class of :class:`InnerPrepare`. The walk's block encoding measures the
    junk right after the prepare and calls the sibling :class:`InnerUnprepareFromOutcomes`
    instead (spec 16, Q-1 variant (b)); this class serves callers that keep the junk.

    **Cost.** :func:`dfthc.costs.inner_unprepare`: :math:`T = w_b + T_{erase,in} +
    F^\dagger(B + 1)`, :math:`M = K_2 d_2` (every target and junk qubit); ``A`` is the swaps'
    ``w_b`` plus the comparator's ``b_k2`` plus the erase's ``T`` plus the adjoint uniform
    preparation's
    :math:`3(n - 1) + 2(s_u - 1)` (0 if ``l_b = 1``). At FeMoco-54: :math:`5 + 196 + 16 = 217`,
    ``M = 416``. The ``PauliX`` count is an upper bound: each measurement's reset is counted at
    its maximum (spec 7.6).

    **Paper and dfthcbe.** Steps 19 to 22 of dfthcbe's README circuit, the adjoint of its
    ``InnerPrepare`` with the measured erase; arXiv:2502.15882v1 App. B.7.2, steps 8 and 9,
    Eqs. (121) to (123) and (125) (Table 8). Eq. (123)'s ``N`` term, the fix-up of the tail
    rows, is :class:`~dfthc.ops.angle_data.AngleErase`'s here (spec 13.1). Our comparator
    adjoint costs no Toffoli (the paper's and dfthcbe's ``b_k2`` term): the comparator's
    ``b_k2 - 1`` work wires stay dirty until here, the saving the paper notes after Eq. (127).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import InnerUnprepare

        sizes = InnerUnprepare.required_wires(2, 2, 2, 1, 2, 1, 1, 3)  # N R B C b_k2 k2 ke2 s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        tables = (((2, 0, 0), (1, 1, 0)), ((3, 1, 2), (0, 2, 3)), ((0, 1, 0), (1, 0, 0)),
                  ((1, 1, 0), (1, 1, 1)))
        program = lambda: InnerUnprepare(2, 2, 2, 1, *tables, 2, 1, 1, 3, **wires)
        compiled_resources(program, num_wires=start).measurements  # K2 d2 = 2 * 7 = 14
    """

    compilable_argnames = ("N", "R", "B", "C", "alt", "keep", "sgn_table", "d_sgn_table", "b_k2",
                           "k2", "ke2", "s_u")
    wire_argnames = _INNER_WIRES

    def __init__(self, N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b,
                 succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b, sgn,
                 d_sgn, r, junk_b, phase_grad, work):
        args = _check_inner(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u)
        wires = (x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b,
                 keep_b, sgn, d_sgn, r, junk_b, phase_grad, work)
        check_wires("InnerUnprepare", dict(zip(_INNER_WIRES, wires, strict=True)),
                    self.required_wires(*args[:4], *args[8:]), at_least=("phase_grad",))
        super().__init__(*args, *wires)

    @staticmethod
    def required_wires(N, R, B, C, b_k2, k2, ke2, s_u) -> dict[str, int]:
        """:meth:`InnerPrepare.required_wires`."""
        return InnerPrepare.required_wires(N, R, B, C, b_k2, k2, ke2, s_u)


def _inner_unprepare_resources(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2,
                               s_u, x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b,
                               cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, junk_b, phase_grad,
                               work):
    return _inner(N, R, B, C, b_k2, k2, ke2, s_u).unprepare_resources(
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2))


@qp.register_resources(_inner_unprepare_resources)
def _inner_unprepare(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b,
                     succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b,
                     sgn, d_sgn, r, junk_b, phase_grad, work):
    _inner(N, R, B, C, b_k2, k2, ke2, s_u).unprepare(
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2),
        **_inner_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                           alt_b, keep_b, sgn, d_sgn, r, junk_b, phase_grad, work))


register_rule(InnerUnprepare, _inner_unprepare)


# ----------------------------------------------------------------------- InnerUnprepareFromOutcomes

_INNER_FROM_OUTCOMES_WIRES = ("x_o", "b", "succ_b", "succ_work_b", "sigma_0", "m_b",
                              "less_than_b", "cmp_work_b", "alt_b", "keep_b", "sgn", "d_sgn", "r",
                              "phase_grad", "work", "extra_onehot")


class InnerUnprepareFromOutcomes(Operator2):
    r"""The measured unprepare of :class:`InnerPrepare` (steps 19 to 22) for a caller that
    X-measured ``junk_b`` right after the prepare (spec 16, Q-1 variant (b)).

    **Action.** After ``InnerPrepare``, ``H`` and ``qp.measure(q, reset=True)`` on every qubit
    of ``junk_b`` (the outcomes :math:`\xi_{junk}`) and operations that respect
    :class:`InnerPrepare`'s postconditions (:class:`InnerSigns` and the selects do),
    ``InnerUnprepareFromOutcomes`` with the outcomes :math:`\xi_{junk}` and the prepare's
    compilable arguments is the identity on the prepare's input (``x_o`` unchanged, every other
    register ``|0>``, ``phase_grad`` unchanged), for every outcome of the junk and target
    measurements: :class:`InnerUnprepare` with its erase's junk measurements moved to the caller.
    The junk wires are free from their measurement on; the block encoding gives them to the
    angle registers, which the selects load and erase in between (spec 10.2).

    **Registers.** Those of :class:`InnerPrepare` without ``junk_b``, then ``extra_onehot``.

    ================ ================================ ============================== ==========
    name             width                            input                          output
    ================ ================================ ============================== ==========
    ``x_o``          ``w_x``                          any                            unchanged
    ``b``            ``w_b``                          the sampled index              ``|0>``
    ``succ_b``       1                                ``[b_0 <= B]``                 ``|0>``
    ``succ_work_b``  ``max(lg(l_b) - 1, 0)``          the prepare's dirty wires      ``|0>``
    ``sigma_0``      1                                ``|+>``                        ``|0>``
    ``m_b``          ``b_k2``                         uniform                        ``|0>``
    ``less_than_b``  1                                ``[keep <= m_b]``              ``|0>``
    ``cmp_work_b``   ``b_k2 - 1``                     the prepare's dirty wires      ``|0>``
    ``alt_b``        ``w_b``                          the loaded ``alt``, or ``b_0`` ``|0>``
    ``keep_b``       ``b_k2``                         the loaded ``keep``            ``|0>``
    ``sgn``          1                                the loaded ``sgn``             ``|0>``
    ``d_sgn``        1                                the loaded ``d_sgn``           ``|0>``
    ``r``            ``w_r``                          the loaded rank                ``|0>``
    ``phase_grad``   at least ``s_u``                 gradient state                 unchanged
    ``work``         :meth:`required_wires`           ``|0>``                        ``|0>``
    ``extra_onehot`` ``max(2**(ke2 + w_b) - d2, 0)``  ``|0>``                        ``|0>``
    ================ ================================ ============================== ==========

    **Arguments.** Dynamic: ``junk_outcomes``, the :math:`(K_2 - 1) d_2` outcomes of the junk
    measurements in the order of the ``junk_b`` wires: slot ``r`` of
    :meth:`QROAMRead.junk_layout <dfthc.ops.qroam.QROAMRead.junk_layout>` gives outcomes
    ``r d2`` to ``(r + 1) d2 - 1`` (``d2 = w_b + b_k2 + 2 + w_r``). Under capture it becomes a
    bool array; in tape mode every entry must be a ``MeasurementValue``, so both modes emit the
    same fixup (K-10). Compilable: those of :class:`InnerPrepare`, with the same domains; ``ke2``
    sets the fixup's one-hot bits, ``ke2 + lg(B + 1)`` (spec 9.5).

    **Pre- and postconditions.** The registers hold what :class:`InnerPrepare` produced, up to
    phases and operations that use ``b`` only as a control or are undone; ``junk_b`` was
    X-measured with reset right after the prepare, ``junk_outcomes`` are its outcomes in wire
    order, and its wires may serve other registers in between; ``work`` and ``extra_onehot`` are
    ``|0>``. Afterwards every register but ``x_o`` and ``phase_grad`` is ``|0>``.

    **Decomposition** (spec 16, Q-1 variant (b)). The ``w_b`` ``CSWAP``\s;
    ``qp.adjoint(LeftQuantumComparator(keep_b, m_b, less_than_b, cmp_work_b, "<="))`` (a
    measured uncompute); ``H`` and ``qp.measure(q, reset=True)`` on every qubit of
    ``alt_b ‖ keep_b ‖ sgn ‖ d_sgn ‖ r``; the parities :math:`f(a) = \xi \cdot g(a)` of the
    outcomes :math:`\xi` = target ‖ ``junk_outcomes`` with :math:`g(a)` =
    :meth:`QROAMRead.loaded_bits <dfthc.ops.qroam.QROAMRead.loaded_bits>` of
    :meth:`InnerPrepare.read_table` over the read's loaded set (masks built here from the
    compilable tables, never passed as data), by :func:`dfthc.compile.mcm_parities`; the fixup
    over that set (one-hot over the low ``ke2`` bits of ``x_o`` and all of ``b``) with the reset
    target ‖ ``extra_onehot`` as its one-hot register
    (:func:`dfthc.ops.qroam.emit_erase_from_outcomes`); ``H`` on ``sigma_0`` and ``m_b``;
    ``qp.adjoint(FlaggedUniformPrep(B + 1, s_u, ...))``, the auto-derived leaf adjoint.

    **Adjoint and control.** Never adjointed or controlled: it measures (spec 7.4 rule 5). It is
    a sibling of :class:`InnerUnprepare`, the same inverse of :class:`InnerPrepare` with another
    signature, not a modified form of a composite (spec 7.4 rule 6).

    **Cost.** :func:`dfthc.costs.inner_unprepare_from_outcomes`: the ``T`` of
    :func:`dfthc.costs.inner_unprepare`, :math:`w_b + T_{erase,in} + F^\dagger(B + 1)`, and
    ``M = d2`` (the target qubits; the caller's :math:`(K_2 - 1) d_2` junk measurements make up
    :class:`InnerUnprepare`'s :math:`K_2 d_2`); ``A`` as :class:`InnerUnprepare`'s. At
    FeMoco-54: :math:`5 + 196 + 16 = 217`, ``M = 26`` (and 390 junk measurements in the block
    encoding per round), ``extra_onehot`` 102 wires. The ``PauliX`` count is an upper bound
    (spec 7.6).

    **Paper and dfthcbe.** Steps 19 to 22 of dfthcbe's README circuit with the junk measured
    early, as the paper does: its :math:`(2^{k_2} - 1) b_2` QROM output qubits are temporary
    and share the angle qubits (arXiv:2502.15882v1 App. B.7.2, step 4, after Eq. (113), and the
    caption of Table 3), and steps 8 and 9, Eqs. (121) to (123) and (125) (Table 8). The walk
    measures the junk right after step 8, the end of the prepare, before any angle load; the
    comparator and the swaps of steps 7 and 8 do not touch it, so this equals measuring right
    after the read (step 6). dfthcbe's adjoint ``InnerPrepare`` measures target and junk
    together at the erase, as :class:`InnerUnprepare` does.

    **Example.**

    .. code-block:: python

        import numpy as np
        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import InnerUnprepareFromOutcomes

        structure = (2, 2, 2, 1, 2, 1, 1, 3)  # N R B C b_k2 k2 ke2 s_u
        sizes = InnerUnprepareFromOutcomes.required_wires(*structure)
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        tables = (((2, 0, 0), (1, 1, 0)), ((3, 1, 2), (0, 2, 3)), ((0, 1, 0), (1, 0, 0)),
                  ((1, 1, 0), (1, 1, 1)))
        junk = np.zeros(InnerUnprepareFromOutcomes.junk_size(2, 2, 2, 1, 2, 1), dtype=bool)
        program = lambda: InnerUnprepareFromOutcomes(junk, 2, 2, 2, 1, *tables, 2, 1, 1, 3,
                                                     **wires)
        compiled_resources(program, num_wires=start).measurements  # d2 = 7
    """

    dynamic_argnames = ("junk_outcomes",)
    compilable_argnames = ("N", "R", "B", "C", "alt", "keep", "sgn_table", "d_sgn_table", "b_k2",
                           "k2", "ke2", "s_u")
    wire_argnames = _INNER_FROM_OUTCOMES_WIRES

    def __init__(self, junk_outcomes, N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2,
                 ke2, s_u, x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                 alt_b, keep_b, sgn, d_sgn, r, phase_grad, work, extra_onehot):
        args = _check_inner(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u)
        junk_outcomes = _junk_outcomes(junk_outcomes,
                                       self.junk_size(*args[:4], args[8], args[9]))
        wires = (x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b,
                 keep_b, sgn, d_sgn, r, phase_grad, work, extra_onehot)
        check_wires("InnerUnprepareFromOutcomes",
                    dict(zip(_INNER_FROM_OUTCOMES_WIRES, wires, strict=True)),
                    self.required_wires(*args[:4], *args[8:]), at_least=("phase_grad",))
        super().__init__(junk_outcomes, *args, *wires)

    @staticmethod
    def required_wires(N, R, B, C, b_k2, k2, ke2, s_u) -> dict[str, int]:
        """:meth:`InnerPrepare.required_wires` without ``junk_b``, then ``extra_onehot``:
        ``max(2**(ke2 + w_b) - d2, 0)`` wires, the one-hot beyond the reset target."""
        sizes = InnerPrepare.required_wires(N, R, B, C, b_k2, k2, ke2, s_u)
        del sizes["junk_b"]
        return sizes | {"extra_onehot": _inner(N, R, B, C, b_k2, k2, ke2, s_u).extra_onehot()}

    @staticmethod
    def junk_size(N, R, B, C, b_k2, k2) -> int:
        """The number of junk outcomes: ``(K2 - 1) d2``, the width of ``junk_b``."""
        return (2**k2 - 1) * (lg(B + 1) + b_k2 + 2 + lg(R))


def _inner_from_outcomes_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b,
                                   cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, phase_grad, work,
                                   extra_onehot) -> dict:
    registers = _inner_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b,
                                 cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, (), phase_grad, work)
    del registers["junk"]
    return registers | {"extra_onehot": extra_onehot}


def _inner_unprepare_from_outcomes_resources(junk_outcomes, N, R, B, C, alt, keep, sgn_table,
                                             d_sgn_table, b_k2, k2, ke2, s_u, x_o, b, succ_b,
                                             succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                                             alt_b, keep_b, sgn, d_sgn, r, phase_grad, work,
                                             extra_onehot):
    return _inner(N, R, B, C, b_k2, k2, ke2, s_u).unprepare_from_outcomes_resources(
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2))


@qp.register_resources(_inner_unprepare_from_outcomes_resources)
def _inner_unprepare_from_outcomes(junk_outcomes, N, R, B, C, alt, keep, sgn_table, d_sgn_table,
                                   b_k2, k2, ke2, s_u, x_o, b, succ_b, succ_work_b, sigma_0, m_b,
                                   less_than_b, cmp_work_b, alt_b, keep_b, sgn, d_sgn, r,
                                   phase_grad, work, extra_onehot):
    size = InnerUnprepareFromOutcomes.junk_size(N, R, B, C, b_k2, k2)
    _inner(N, R, B, C, b_k2, k2, ke2, s_u).unprepare_from_outcomes(
        [junk_outcomes[i] for i in range(size)],
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2),
        **_inner_from_outcomes_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b,
                                         cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, phase_grad,
                                         work, extra_onehot))


register_rule(InnerUnprepareFromOutcomes, _inner_unprepare_from_outcomes)


# --------------------------------------------------------------------------- InnerSigns


class InnerSigns(Operator2):
    r"""Give every branch of the inner prepare the sign of the weight it selected (step 9).

    **Action.** :math:`Z(\mathrm{sgn})\,CZ(\mathrm{less\_than}_b, \mathrm{d\_sgn})`, the phase
    :math:`(-1)^{\mathrm{sgn} \oplus (\mathrm{less\_than}_b \wedge \mathrm{d\_sgn})}` on every
    basis state. After :class:`InnerPrepare`, ``sgn = [w_{b_0} < 0]`` and
    ``d_sgn = sgn XOR sgn[alt[b_0]]`` of the drawn :math:`b_0`, so the phase is the sign of the
    weight of the ``b`` that survives the swap (spec 6.1); branches whose selected entry is zero
    (failed ``b``, one-body rows and ``x_o >= X_o``) get +1 (spec 8.5). ``x_o >= X_o`` is an input
    the walk never presents: ``OuterPrepare``'s swap maps a failed draw to ``x_o = 0``, an SF
    row, where the inner signs cancel between the two rounds and ``REF_T2`` gives ``-I``.

    **Registers.**

    =============== ===== ===== =========
    name            width input output
    =============== ===== ===== =========
    ``sgn``         1     any   unchanged
    ``d_sgn``       1     any   unchanged
    ``less_than_b`` 1     any   unchanged
    =============== ===== ===== =========

    **Compilable arguments.** None.

    **Pre- and postconditions.** None: it acts on every input as stated (it only applies
    phases).

    **Decomposition** (spec 9.9). ``Z(sgn)``, then ``CZ(less_than_b, d_sgn)``.

    **Adjoint and control.** Self-adjoint: ``Adjoint(InnerSigns)`` is pinned to the operator
    itself (spec 7.5); no walk program requests it. Never controlled.

    **Cost.** :func:`dfthc.costs.inner_signs`: ``T = 0``, ``M = 0``.

    **Paper and dfthcbe.** Step 9 of dfthcbe's README circuit, its ``InnerSigns``
    (``dfthcbe/prepare.py``); the paper applies the signs inside its iteration over ``b``
    (arXiv:2502.15882v1 App. B.5).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import InnerSigns

        compiled_resources(lambda: InnerSigns([0], [1], [2]), num_wires=3).counts
        # {"CZ": 1, "PauliZ": 1}
    """

    wire_argnames = ("sgn", "d_sgn", "less_than_b")

    def __init__(self, sgn, d_sgn, less_than_b):
        check_wires("InnerSigns", {"sgn": sgn, "d_sgn": d_sgn, "less_than_b": less_than_b},
                    self.required_wires())
        super().__init__(sgn, d_sgn, less_than_b)

    @staticmethod
    def required_wires() -> dict[str, int]:
        """Exact register sizes: one wire each."""
        return {"sgn": 1, "d_sgn": 1, "less_than_b": 1}


def _inner_signs_resources(sgn, d_sgn, less_than_b):
    return {qp.Z: 1, qp.CZ: 1}


@qp.register_resources(_inner_signs_resources)
def _inner_signs(sgn, d_sgn, less_than_b):
    qp.Z(sgn[0])
    qp.CZ([less_than_b[0], d_sgn[0]])


def _adjoint_inner_signs_resources(base):
    return {resource_key(lambda: InnerSigns(Wire[1], Wire[1], Wire[1])): 1}


@qp.register_resources(_adjoint_inner_signs_resources)
def _adjoint_inner_signs(base):
    InnerSigns(base.sgn, base.d_sgn, base.less_than_b)


register_rule(InnerSigns, _inner_signs)
register_rule("Adjoint(InnerSigns)", _adjoint_inner_signs)
