"""Gate set, pinned rules and compile helpers shared by every operator and test.

Spec sections 7.2 (pipeline), 7.3 (gate set), 7.4 (rules for operators), 7.5 (pins) and
7.6 (resource accounting).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qp

GATE_SET: dict[str, int] = {
    "PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "Adjoint(S)": 1,
    "CNOT": 1, "CZ": 1, "SWAP": 1, "GlobalPhase": 1, "MidMeasure": 1,
    "Adjoint(TemporaryAND)": 1, "PhaseGradientStatePrep": 1,
    "Toffoli": 1000, "TemporaryAND": 1000,
}
"""Toffoli-level target gate set with solver weights (spec section 7.3)."""

ANALYZE_NAMES: frozenset[str] = frozenset(GATE_SET) - {"MidMeasure"} | {"MidCircuitMeasure"}
"""Names ``qp.analyze`` may report at ``level="user"`` (it names MidMeasure MidCircuitMeasure)."""

FIXED_DECOMPS: dict[str, Callable] = {}
"""Pinned rule for every rule key registered through :func:`register_rule` (spec section 7.5)."""

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_WRAPPERS = frozenset({"Adjoint", "C"})


def register_rule(op: type | str, rule: Callable) -> None:
    """Register ``rule`` as the only rule for a dfthc operator key and pin it.

    ``op`` is a dfthc ``Operator2`` subclass or a symbolic key such as ``"Adjoint(OuterPrepare)"``
    or ``"C(RefT2)"``. Each key gets exactly one rule (spec section 7.4, rule 7). Never call this
    for a PennyLane operator: a pin also applies to nested instances of the same name (L-4).
    """
    key = op if isinstance(op, str) else op.__name__
    if key in FIXED_DECOMPS:
        raise ValueError(f"{key} already has a pinned rule: {FIXED_DECOMPS[key]}")
    qp.add_decomps(op, rule)
    FIXED_DECOMPS[key] = rule


def resource_key(build: Callable[[], qp.operation.Operator]) -> qp.operation.Operator:
    """Build a resource-function key with program capture paused (spec section 7.4, rule 2; L-11).

    Example: ``resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3])))``.
    """
    with qp.capture.pause():
        return build()


@dataclass(frozen=True)
class CompiledResources:
    """Counts of one compiled program (spec section 7.6).

    T, M and A are defined at ``level="user"`` only. At other levels the device pass rewrites
    both ``TemporaryAND`` and ``Adjoint(TemporaryAND)`` into ``Toffoli``, so the three are
    ``None`` there, and only ``counts``, ``num_wires`` and the closure check are meaningful.
    """

    toffoli: int | None
    """T: number of Toffoli plus number of TemporaryAND (``None`` unless ``level="user"``)."""
    measurements: int | None
    """M: number of MidCircuitMeasure, the explicit erase measurements (``None`` unless
    ``level="user"``)."""
    measured_and_uncomputes: int | None
    """A: number of Adjoint(TemporaryAND), the measured AND uncomputes (``None`` unless
    ``level="user"``)."""
    counts: dict[str, int]
    """All gate counts reported by qp.analyze."""
    num_wires: int
    """Wire count reported by qp.analyze."""
    level: str = "user"
    """The ``qp.analyze`` level the counts were taken at."""

    @property
    def physical_measurements(self) -> int:
        """M + A: every explicit measurement and every measured AND uncompute."""
        if self.level != "user":
            raise ValueError(
                f"physical_measurements needs a level='user' record, got level={self.level!r}"
            )
        return self.measurements + self.measured_and_uncomputes


def compiled_resources(
    qfunc: Callable[[], None],
    num_wires: int,
    *,
    device: str = "null.qubit",
    level: str = "user",
    fixed_decomps: dict[str, Callable] | None = None,
) -> CompiledResources:
    """Compile ``qfunc`` with the stack of spec section 7.2 and return exact counts.

    ``qfunc`` takes no arguments and queues operations. The program is compiled once, by
    ``qp.analyze``; the decorator's ahead-of-time binary compile is off (L-27).
    ``fixed_decomps`` replaces :data:`FIXED_DECOMPS` (pass ``{**FIXED_DECOMPS, ...}`` to add
    pins); an empty dict disables pinning. Raises ``AssertionError`` if the counts are symbolic
    or, at ``level="user"``, if an operation outside the gate set remains (T5). At
    ``level="device"`` it instead asserts that no dfthc operator survives, bare or under
    ``C(...)`` and ``Adjoint(...)`` (L-20; :func:`surviving_dfthc_operators`); the record then
    holds only ``counts`` and ``num_wires``, with T, M and A set to ``None``, because the device
    pass rewrites ``TemporaryAND`` and ``Adjoint(TemporaryAND)`` into ``Toffoli``.
    """
    pins = FIXED_DECOMPS if fixed_decomps is None else fixed_decomps
    dev = qp.device(device, wires=num_wires)

    # A function without parameters counts as fully annotated, so qjit would compile it to a binary
    # at decoration, and qp.analyze would then compile it again (L-27): switch the first one off.
    @qp.qjit(capture=True, enable_aot_compilation=False)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(pins) or None)
    @qp.qnode(dev)
    def circuit():
        qfunc()
        return qp.probs(wires=[0])

    resources = qp.analyze(circuit, level=level)().resources
    if resources.is_symbolic:
        raise AssertionError(f"qp.analyze returned symbolic counts: {resources.counts}")
    counts = {str(name): int(value) for name, value in resources.counts.items()}
    if level == "user":
        outside = sorted(set(counts) - ANALYZE_NAMES)
        if outside:
            raise AssertionError(f"operations outside the gate set remain: {outside}")
    else:
        surviving = surviving_dfthc_operators(counts, set(FIXED_DECOMPS) | set(pins))
        if surviving:
            raise AssertionError(f"dfthc operators reached level={level!r}: {surviving}")
    user = level == "user"
    return CompiledResources(
        toffoli=counts.get("Toffoli", 0) + counts.get("TemporaryAND", 0) if user else None,
        measurements=counts.get("MidCircuitMeasure", 0) if user else None,
        measured_and_uncomputes=counts.get("Adjoint(TemporaryAND)", 0) if user else None,
        counts=counts,
        num_wires=int(resources.num_wires),
        level=level,
    )


def dfthc_operator_names(keys: Iterable[str] | None = None) -> frozenset[str]:
    """Operator names inside rule keys, by default those of :data:`FIXED_DECOMPS`.

    ``"C(Adjoint(OuterPrepare))"`` gives ``OuterPrepare``.
    """
    keys = FIXED_DECOMPS if keys is None else keys
    return frozenset(
        token for key in keys for token in _IDENTIFIER.findall(key) if token not in _WRAPPERS
    )


def surviving_dfthc_operators(
    names: Iterable[str], keys: Iterable[str] | None = None
) -> list[str]:
    """The gate ``names`` that contain an operator name of the rule ``keys``, bare or wrapped.

    A name matches if any identifier in it is a dfthc operator name, so ``OuterPrepare``,
    ``C(OuterPrepare)``, ``Adjoint(OuterPrepare)`` and ``Adjoint(C(OuterPrepare))`` match, while
    ``C(PauliX)``, ``Adjoint(TemporaryAND)`` and ``OuterPrepareX`` do not.
    """
    operators = dfthc_operator_names(keys)
    return sorted(name for name in names if operators.intersection(_IDENTIFIER.findall(name)))


def mcm_parities(outcomes: Sequence, masks: Sequence[int]) -> list:
    """Parities ``f(a) = xi . g(a) mod 2`` of measurement outcomes, one per address.

    ``outcomes[i]`` is the result of ``qp.measure`` for outcome bit ``i``: a ``MeasurementValue``
    in tape mode, a traced scalar under capture. ``masks[a]`` is the integer whose bit ``i`` is
    ``g(a)_i``. Returns one entry per mask: ``False`` (a Python bool) when the mask is 0, so callers
    emit no conditional; otherwise a predicate usable in ``qp.cond``. If every mask is 0, including
    the case of no outcomes, every entry is ``False`` in both modes. Tape mode builds one flat
    ``MeasurementValue`` per mask, the sum of its outcome bits mod 2; capture mode computes a
    packed-bit matrix-vector product with ``jax.numpy`` (spec section 7.4, rule 9). The map is
    GF(2)-linear in the outcomes in both modes.
    """
    num_outcomes = len(outcomes)
    for mask in masks:
        if mask < 0 or mask >> num_outcomes:
            raise ValueError(f"mask {mask} has bits beyond the {num_outcomes} outcomes")
    if not any(masks):
        return [False] * len(masks)
    if qp.capture.enabled():
        return _capture_parities(outcomes, masks)
    return [_tape_parity(outcomes, mask) for mask in masks]


def _tape_parity(outcomes: Sequence, mask: int):
    """One flat ``MeasurementValue`` over the chosen outcomes' measurements.

    Each outcome keeps its own ``processing_fn``, so a composite outcome such as ``~m0`` gives the
    right bit. Every chosen outcome must depend on exactly one measurement.
    """
    indices = [i for i in range(len(outcomes)) if (mask >> i) & 1]
    if not indices:
        return False
    for i in indices:
        if len(outcomes[i].measurements) != 1:
            raise ValueError(
                f"outcome {i} depends on {len(outcomes[i].measurements)} measurements; "
                "each outcome must depend on exactly one"
            )
    fns = [outcomes[i].processing_fn for i in indices]
    return qp.ops.MeasurementValue(
        [outcomes[i].measurements[0] for i in indices],
        lambda *bits: sum(fn(bit) for fn, bit in zip(fns, bits, strict=True)) % 2,
    )


def pack_masks(masks: Sequence[int], num_outcomes: int) -> np.ndarray:
    """Pack masks into a ``(len(masks), ceil(num_outcomes / 32))`` uint32 array, low word first."""
    num_words = max(1, -(-num_outcomes // 32))
    packed = np.zeros((len(masks), num_words), dtype=np.uint32)
    for row, mask in enumerate(masks):
        for word in range(num_words):
            packed[row, word] = (mask >> (32 * word)) & 0xFFFFFFFF
    return packed


def packed_parities(outcome_bits, packed_masks: np.ndarray):
    """Parities of a 0/1 outcome vector against packed masks, with jax.numpy (traceable)."""
    num_words = packed_masks.shape[1]
    bits = jnp.asarray(outcome_bits, dtype=jnp.uint32)
    bits = jnp.pad(bits, (0, 32 * num_words - bits.shape[0]))
    shifts = jnp.arange(32, dtype=jnp.uint32)
    words = jnp.sum(bits.reshape(num_words, 32) << shifts, axis=1, dtype=jnp.uint32)
    overlap = jnp.asarray(packed_masks) & words
    return jnp.sum(jax.lax.population_count(overlap), axis=1) & 1


def _capture_parities(outcomes: Sequence, masks: Sequence[int]) -> list:
    bits = jnp.stack([jnp.asarray(value, dtype=jnp.uint32) for value in outcomes])
    parities = packed_parities(bits, pack_masks(masks, len(outcomes)))
    return [parities[row] == 1 if mask else False for row, mask in enumerate(masks)]
