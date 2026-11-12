"""Sparse-state simulator for the gate-level circuits of the walk (spec section 12.3).

The state is a list of computational-basis keys with ``complex128`` amplitudes. A key packs the
wire values into ``uint64`` words: wire ``w`` is bit ``63 - w % 64`` of word ``w // 64``, so wire
0 is the most significant bit, as in PennyLane's statevectors. Amplitudes below
:data:`PRUNE_THRESHOLD` are dropped whenever terms merge.

Wires are the integers ``0 .. num_wires - 1``. Every operation of ``dfthc.compile.GATE_SET`` has
one handler, and so has ``Conditional``; ``TemporaryAND``, ``Adjoint(TemporaryAND)`` and
``PhaseGradientStatePrep`` check their preconditions on every term and raise ``AssertionError``
with the offending key. Any other operation raises :class:`UnsupportedOperation`.

Mid-circuit measurements take their outcome from ``postselect`` if it is set, else from the
simulator's policy: :class:`ForcedPolicy` (a schedule, for the outcome certificates of spec 12.5),
:class:`SeededPolicy` (random) or :class:`BranchPolicy` (both outcomes on listed wires, through
:meth:`SparseSimulator.run_branches`). Each outcome is recorded under its ``MidMeasure`` operator,
which is what ``MeasurementValue.concretize`` looks up.

:func:`nonlinear_conditionals` lists the conditionals of a tape that break the premise of the
outcome certificates (spec 12.5): a diagonal base and a GF(2)-linear predicate.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from pennylane.ops import Conditional

PRUNE_THRESHOLD = 1e-14
MAX_DENSE_WIRES = 25
MEASUREMENT_TOLERANCE = 1e-12
_ONE = np.uint64(1)


class UnsupportedOperation(Exception):
    """An operation without a simulator handler (spec 12.3: the operation contract)."""


def _num_words(num_wires: int) -> int:
    return max(1, -(-num_wires // 64))


def _locate(wire: int) -> tuple[int, np.uint64]:
    """The word and the bit mask of ``wire``."""
    return wire // 64, _ONE << np.uint64(63 - wire % 64)


def _merge(keys: np.ndarray, amps: np.ndarray, threshold: float = PRUNE_THRESHOLD):
    """Sum the amplitudes of equal keys and drop those below ``threshold``."""
    if keys.shape[1] == 1:
        unique, inverse = np.unique(keys[:, 0], return_inverse=True)
        unique = unique[:, None]
    else:
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    summed = (np.bincount(inverse, amps.real, len(unique))
              + 1j * np.bincount(inverse, amps.imag, len(unique)))
    keep = np.abs(summed) >= threshold
    return unique[keep], summed[keep]


@dataclass
class SparseState:
    """Basis keys ``(terms, ceil(num_wires / 64))`` of ``uint64`` and their amplitudes."""

    num_wires: int
    keys: np.ndarray
    amps: np.ndarray

    @classmethod
    def basis(cls, num_wires: int, value: int) -> SparseState:
        """The basis state ``|value>``, big-endian: wire 0 is the most significant bit."""
        if not 0 <= value < 2**num_wires:
            raise ValueError(f"basis value {value} outside [0, 2**{num_wires})")
        keys = np.zeros((1, _num_words(num_wires)), dtype=np.uint64)
        for wire in range(num_wires):
            if (value >> (num_wires - 1 - wire)) & 1:
                word, mask = _locate(wire)
                keys[0, word] |= mask
        return cls(num_wires, keys, np.ones(1, dtype=complex))

    @classmethod
    def from_dense(cls, vector) -> SparseState:
        """The nonzero amplitudes of a statevector of ``2**n`` entries, ``n <= 25``."""
        vector = np.asarray(vector, dtype=complex).ravel()
        num_wires = int(np.log2(len(vector)))
        if 2**num_wires != len(vector) or num_wires > MAX_DENSE_WIRES:
            raise ValueError(f"need 2**n amplitudes with n <= {MAX_DENSE_WIRES}; got {len(vector)}")
        index = np.flatnonzero(np.abs(vector) >= PRUNE_THRESHOLD)
        keys = (index.astype(np.uint64) << np.uint64(64 - num_wires))[:, None]
        return cls(num_wires, keys, vector[index])

    def to_dense(self) -> np.ndarray:
        """The statevector, for at most 25 wires."""
        if self.num_wires > MAX_DENSE_WIRES:
            raise ValueError(f"to_dense supports at most {MAX_DENSE_WIRES} wires; "
                             f"this state has {self.num_wires}")
        vector = np.zeros(2**self.num_wires, dtype=complex)
        index = (self.keys[:, 0] >> np.uint64(64 - self.num_wires)).astype(np.int64)
        np.add.at(vector, index, self.amps)
        return vector

    def copy(self) -> SparseState:
        return SparseState(self.num_wires, self.keys.copy(), self.amps.copy())

    def _checked_locate(self, wire: int) -> tuple[int, np.uint64]:
        """The word and the bit mask of ``wire``, a Python or NumPy integer in the state."""
        if not isinstance(wire, (int, np.integer)) or not 0 <= wire < self.num_wires:
            raise ValueError(f"wire {wire} outside [0, {self.num_wires})")
        return _locate(int(wire))

    def bits(self, wire: int) -> np.ndarray:
        """Boolean value of ``wire`` on every term."""
        word, mask = self._checked_locate(wire)
        return (self.keys[:, word] & mask) != 0

    def flip(self, wire: int, where: np.ndarray | None = None) -> None:
        """Flip ``wire`` on the terms selected by ``where`` (all terms if ``None``)."""
        word, mask = self._checked_locate(wire)
        if where is None:
            self.keys[:, word] ^= mask
        else:
            self.keys[where, word] ^= mask

    def probability(self, wire: int, outcome: int) -> float:
        """Probability that measuring ``wire`` gives ``outcome``, relative to the state's norm."""
        weights = np.abs(self.amps) ** 2
        return float(weights[self.bits(wire) == bool(outcome)].sum() / weights.sum())

    def prune(self, threshold: float = PRUNE_THRESHOLD) -> SparseState:
        """Drop amplitudes below ``threshold`` (in place) and return the state."""
        keep = np.abs(self.amps) >= threshold
        self.keys, self.amps = self.keys[keep], self.amps[keep]
        return self

    def key_string(self, row: int) -> str:
        """The key of term ``row`` as a bit string, wire 0 first."""
        return "".join("1" if self.bits(wire)[row] else "0" for wire in range(self.num_wires))


class ForcedPolicy:
    """Outcomes from ``schedule``, one per measurement in order (spec 12.3, forced policy)."""

    def __init__(self, schedule: Sequence[int]):
        self.schedule = [int(outcome) for outcome in schedule]
        if any(outcome not in (0, 1) for outcome in self.schedule):
            raise ValueError(f"forced outcomes must be 0 or 1; got {list(schedule)}")
        self.position = 0

    def choose(self, op, probability_one: float) -> int:
        if self.position >= len(self.schedule):
            raise ValueError(f"the forced schedule of {len(self.schedule)} outcomes is exhausted "
                             f"at {op}")
        outcome = self.schedule[self.position]
        self.position += 1
        return outcome


class SeededPolicy:
    """Random outcomes with the measured probabilities, from ``seed`` (spec 12.3, seeded)."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def choose(self, op, probability_one: float) -> int:
        return int(self.rng.random() < probability_one)


class BranchPolicy:
    """Both outcomes of every measurement on ``wires``, for QPE control bits (spec 12.3, branch).

    Only :meth:`SparseSimulator.run_branches` can follow both outcomes. Measurements on other
    wires go to ``otherwise`` (for example a :class:`ForcedPolicy` for erasure measurements).
    """

    def __init__(self, wires: Collection, *, otherwise=None):
        self.wires = frozenset(int(wire) for wire in wires)
        self.otherwise = otherwise

    def branches(self, op) -> bool:
        return int(op.wires[0]) in self.wires

    def choose(self, op, probability_one: float) -> int:
        if self.branches(op):
            raise ValueError(f"{op} is on a branch wire: simulate with run_branches")
        if self.otherwise is None:
            raise ValueError(f"no policy chooses the outcome of {op}: it is not on a branch wire")
        return self.otherwise.choose(op, probability_one)


@dataclass
class Branch:
    """One measurement branch: the recorded outcomes, the normalized state and its probability."""

    record: dict
    state: SparseState
    probability: float


class SparseSimulator:
    """Applies gate-set operations, measurements and conditionals to a :class:`SparseState`.

    :meth:`run` starts from ``initial_state`` (default ``|0...0>``; it must have ``num_wires``
    wires, and its equal keys are merged), applies the operations in order and returns the final
    state; ``record`` then maps every ``MidMeasure`` to its outcome.
    :meth:`run_branches` also follows both outcomes of the measurements a :class:`BranchPolicy`
    names. In strict mode every other measurement without ``postselect`` must have probability 1/2
    (within :data:`MEASUREMENT_TOLERANCE`): a precondition check of the measured erasure, not a
    certificate (spec 12.3).
    """

    def __init__(self, num_wires: int, *, policy=None, strict: bool = True):
        self.num_wires = num_wires
        self.policy = policy
        self.strict = strict
        self.state = SparseState.basis(num_wires, 0)
        self.record: dict = {}
        self._handlers = {
            "PauliX": self._controlled_x, "CNOT": self._controlled_x,
            "Toffoli": self._controlled_x, "SWAP": self._swap, "PauliY": self._y,
            "PauliZ": lambda op: self._phase(op, -1), "S": lambda op: self._phase(op, 1j),
            "Adjoint(S)": lambda op: self._phase(op, -1j), "CZ": self._cz,
            "GlobalPhase": self._global_phase, "Hadamard": self._hadamard,
            "TemporaryAND": self._temporary_and,
            "Adjoint(TemporaryAND)": self._adjoint_temporary_and,
            "PhaseGradientStatePrep": self._phase_gradient_state_prep,
            "MidMeasure": self._mid_measure,
        }

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset(self._handlers) | {"Conditional"}

    def _start(self, initial_state: SparseState | None) -> SparseState:
        if initial_state is None:
            return SparseState.basis(self.num_wires, 0)
        words = _num_words(self.num_wires)
        if initial_state.num_wires != self.num_wires or initial_state.keys.shape[1] != words:
            raise ValueError(f"initial_state has {initial_state.num_wires} wires and key "
                             f"width {initial_state.keys.shape[1]}; the simulator has "
                             f"{self.num_wires} wires and key width {words}")
        return SparseState(self.num_wires, *_merge(initial_state.keys, initial_state.amps))

    def run(self, operations: Iterable, initial_state: SparseState | None = None) -> SparseState:
        if isinstance(self.policy, BranchPolicy):
            raise ValueError("a BranchPolicy follows both outcomes: simulate with run_branches")
        self.state, self.record = self._start(initial_state), {}
        for op in operations:
            self.apply(op)
        return self.state

    def run_branches(self, operations: Iterable,
                     initial_state: SparseState | None = None) -> list[Branch]:
        """Every branch of the measurements on branch wires, with its probability.

        Each branch path consumes its own copy of the policy: at a split both children continue
        from the parent's position, so the i-th delegated measurement on every path reads
        ``schedule[i]`` of a forced ``otherwise``, and a :class:`SeededPolicy` replays the same
        draws in both children. Outcomes chosen by ``otherwise`` are not weighted; in strict mode
        each has probability 1/2 in every branch, so the branch probabilities are unaffected.
        A postselected measurement drops the branches that cannot pass it and weights the others
        by their success probability, so the returned probabilities are conditional on every
        postselection succeeding, as with ``default.qubit`` and deferred measurements. Raises
        ``ValueError`` if no branch passes a postselection. It leaves ``state``, ``record`` and
        ``policy`` as they were.
        """
        policy, state, record = self.policy, self.state, self.record
        branches = [(Branch({}, self._start(initial_state), 1.0), copy.deepcopy(policy))]
        try:
            for op in operations:
                following = []
                for branch, self.policy in branches:
                    self.state, self.record = branch.state, branch.record
                    if op.name == "MidMeasure" and op.postselect is not None:
                        (wire,) = self._wires(op)
                        success = branch.state.probability(wire, int(op.postselect))
                        if success >= MEASUREMENT_TOLERANCE:
                            self._project(op, wire, int(op.postselect))
                            branch.probability *= success
                            following.append((branch, self.policy))
                        continue
                    if not self._branches(op):
                        self.apply(op)
                        following.append((branch, self.policy))
                        continue
                    (wire,) = self._wires(op)
                    for outcome in (0, 1):
                        if branch.state.probability(wire, outcome) < MEASUREMENT_TOLERANCE:
                            continue
                        self.state, self.record = branch.state.copy(), dict(branch.record)
                        probability = self._project(op, wire, outcome)
                        following.append((Branch(self.record, self.state,
                                                 branch.probability * probability),
                                          copy.deepcopy(self.policy)))
                if not following:
                    raise ValueError(f"no branch can pass the postselection {op}")
                branches = following
        finally:
            self.policy, self.state, self.record = policy, state, record
        total = sum(branch.probability for branch, _ in branches)
        for branch, _ in branches:
            branch.probability /= total
        return [branch for branch, _ in branches]

    def apply(self, op) -> None:
        if isinstance(op, Conditional):
            if op.meas_val.concretize(self.record):
                self.apply(op.base)
            return
        handler = self._handlers.get(op.name)
        if handler is None:
            raise UnsupportedOperation(f"no simulator handler for {op.name} (spec 12.3): {op}")
        handler(op)

    def _wires(self, op) -> list[int]:
        wires = [int(wire) for wire in op.wires]
        for wire in wires:
            if not 0 <= wire < self.num_wires:
                raise ValueError(f"{op.name} acts on wire {wire} outside [0, {self.num_wires})")
        return wires

    def _all(self, wires: Iterable[int]) -> np.ndarray:
        where = np.ones(len(self.state.amps), dtype=bool)
        for wire in wires:
            where &= self.state.bits(wire)
        return where

    def _controlled_x(self, op) -> None:
        *controls, target = self._wires(op)
        self.state.flip(target, self._all(controls) if controls else None)

    def _swap(self, op) -> None:
        a, b = self._wires(op)
        differ = self.state.bits(a) != self.state.bits(b)
        self.state.flip(a, differ)
        self.state.flip(b, differ)

    def _y(self, op) -> None:
        (target,) = self._wires(op)
        self.state.amps *= np.where(self.state.bits(target), -1j, 1j)
        self.state.flip(target)

    def _phase(self, op, phase: complex) -> None:
        (target,) = self._wires(op)
        self.state.amps[self.state.bits(target)] *= phase

    def _cz(self, op) -> None:
        self.state.amps[self._all(self._wires(op))] *= -1

    def _global_phase(self, op) -> None:
        self._wires(op)
        self.state.amps *= np.exp(-1j * float(op.data[0]))

    def _hadamard(self, op) -> None:
        (target,) = self._wires(op)
        state = self.state
        word, mask = _locate(target)
        bit = state.bits(target)
        low = state.keys.copy()
        low[:, word] &= ~mask
        high = low.copy()
        high[:, word] |= mask
        amps = state.amps / np.sqrt(2)
        state.keys, state.amps = _merge(np.concatenate([low, high]),
                                        np.concatenate([amps, np.where(bit, -amps, amps)]))

    def _and(self, base, controls: list[int]) -> np.ndarray:
        values = [bool(value) for value in np.asarray(base.control_values).ravel()]
        where = np.ones(len(self.state.amps), dtype=bool)
        for wire, value in zip(controls, values, strict=True):
            where &= self.state.bits(wire) == value
        return where

    def _temporary_and(self, op) -> None:
        c1, c2, target = self._wires(op)
        bad = np.flatnonzero(self.state.bits(target))
        if bad.size:
            raise AssertionError(f"TemporaryAND on {op.wires.tolist()}: the target is 1 on key "
                                 f"{self.state.key_string(bad[0])}")
        self.state.flip(target, self._and(op, [c1, c2]))

    def _adjoint_temporary_and(self, op) -> None:
        c1, c2, target = self._wires(op)
        conjunction = self._and(op.base, [c1, c2])
        bad = np.flatnonzero(self.state.bits(target) != conjunction)
        if bad.size:
            raise AssertionError(f"Adjoint(TemporaryAND) on {op.wires.tolist()}: the target "
                                 f"differs from the AND on key {self.state.key_string(bad[0])}")
        self.state.flip(target, conjunction)

    def _phase_gradient_state_prep(self, op) -> None:
        register = self._wires(op)
        state = self.state
        busy = np.zeros(len(state.amps), dtype=bool)
        for wire in register:
            busy |= state.bits(wire)
        bad = np.flatnonzero(busy)
        if bad.size:
            raise AssertionError(f"PhaseGradientStatePrep on {register}: the register is not |0> "
                                 f"on key {state.key_string(bad[0])}")
        size = len(register)
        values = np.arange(2**size)
        keys = np.repeat(state.keys, 2**size, axis=0)
        flat = np.tile(values, len(state.amps))
        for position, wire in enumerate(register):
            word, mask = _locate(wire)
            ones = ((flat >> (size - 1 - position)) & 1).astype(bool)
            keys[ones, word] |= mask
        phases = np.exp(-2j * np.pi * values / 2**size) / np.sqrt(2**size)
        state.keys, state.amps = keys, (state.amps[:, None] * phases[None, :]).ravel()

    def _branches(self, op) -> bool:
        return (op.name == "MidMeasure" and op.postselect is None
                and isinstance(self.policy, BranchPolicy) and self.policy.branches(op))

    def _mid_measure(self, op) -> None:
        (wire,) = self._wires(op)
        probability_one = self.state.probability(wire, 1)
        if op.postselect is not None:
            outcome = int(op.postselect)
        else:
            if self.policy is None:
                raise ValueError(f"no policy chooses the outcome of {op}")
            if self.strict and not abs(probability_one - 0.5) <= MEASUREMENT_TOLERANCE:
                raise AssertionError(f"strict mode: {op} has P(1) = {probability_one}, not 1/2 "
                                     "(spec 12.3)")
            outcome = self.policy.choose(op, probability_one)
        self._project(op, wire, outcome)

    def _project(self, op, wire: int, outcome: int) -> float:
        """Keep the terms with ``wire == outcome``, normalize them, record the outcome and reset
        if asked; return the outcome's probability. Normalizing by the kept weight itself, not by
        ``1 - P(1)``, keeps rounding errors from compounding over many measurements. Measurements
        that compare equal (hand-built ones without a ``meas_uid``) would share one record entry,
        so a second one raises."""
        if op in self.record:
            raise ValueError(f"{op} is already recorded: every measurement needs its own "
                             "meas_uid (qp.measure sets one)")
        state = self.state
        keep = state.bits(wire) == bool(outcome)
        weights = np.abs(state.amps) ** 2
        kept = weights[keep].sum()
        probability = float(kept / weights.sum())
        if not probability >= MEASUREMENT_TOLERANCE:
            raise ValueError(f"{op} cannot give outcome {outcome}: it has probability "
                             f"{probability:.3g}")
        state.keys, state.amps = state.keys[keep], state.amps[keep] / np.sqrt(kept)
        if op.reset and outcome:
            state.flip(wire)
        self.record[op] = outcome
        return probability


def nonlinear_conditionals(operations: Iterable, *, exhaustive_up_to: int = 12,
                           samples: int = 256, seed: int = 0) -> list:
    """The conditionals that break the premise of spec 12.5's GF(2) conclusion.

    The conclusion from the zero and unit outcome vectors to every outcome vector holds only if
    each conditional of the certified block is diagonal (``PauliZ`` or ``CZ``) and its predicate
    is GF(2)-linear in its outcomes: f(0) = 0 and f(x ^ y) = f(x) ^ f(y). The predicate is
    checked on every outcome vector up to ``exhaustive_up_to`` outcomes, else on ``samples``
    random ones against the unit-vector responses.
    """
    rng = np.random.default_rng(seed)
    flagged = []
    for op in operations:
        if not isinstance(op, Conditional):
            continue
        value, size = op.meas_val, len(op.meas_val.measurements)

        def predicate(bits, value=value):
            return bool(value.processing_fn(*bits))

        mask = [predicate(tuple(int(i == j) for j in range(size))) for i in range(size)]
        points = (itertools.product((0, 1), repeat=size) if size <= exhaustive_up_to else
                  (tuple(int(b) for b in rng.integers(0, 2, size)) for _ in range(samples)))
        linear = all(predicate(x) == bool(sum(m & b for m, b in zip(mask, x, strict=True)) % 2)
                     for x in points)
        if op.base.name not in {"PauliZ", "CZ"} or not linear:
            flagged.append(op)
    return flagged
