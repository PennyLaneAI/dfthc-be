"""Sparse-state simulator for the gate-level circuits of the walk (spec section 12.3).

The state is a list of computational-basis keys with ``complex128`` amplitudes. A key packs the
wire values into ``uint64`` words: wire ``w`` is bit ``63 - w % 64`` of word ``w // 64``, so wire
0 is the most significant bit, as in PennyLane's statevectors. Amplitudes below
:data:`PRUNE_THRESHOLD` are dropped whenever terms merge.

Wires are the integers ``0 .. num_wires - 1``. Every operation of ``dfthc.compile.GATE_SET``
except ``MidMeasure`` has one handler; ``TemporaryAND``, ``Adjoint(TemporaryAND)`` and
``PhaseGradientStatePrep`` check their preconditions on every term and raise ``AssertionError``
with the offending key. Any other operation raises :class:`UnsupportedOperation`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

PRUNE_THRESHOLD = 1e-14
MAX_DENSE_WIRES = 25
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

    def bits(self, wire: int) -> np.ndarray:
        """Boolean value of ``wire`` on every term."""
        word, mask = _locate(wire)
        return (self.keys[:, word] & mask) != 0

    def flip(self, wire: int, where: np.ndarray | None = None) -> None:
        """Flip ``wire`` on the terms selected by ``where`` (all terms if ``None``)."""
        word, mask = _locate(wire)
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


class SparseSimulator:
    """Applies gate-set operations to a :class:`SparseState` (spec 12.3).

    ``policy`` (outcomes of mid-circuit measurements) and ``strict`` (assert uniformly random
    erasure measurements) are stored for the measurement handler, which this version lacks.
    :meth:`run` starts from ``initial_state`` (default ``|0...0>``), applies the operations in
    order and returns the final state.
    """

    def __init__(self, num_wires: int, *, policy=None, strict: bool = True):
        self.num_wires = num_wires
        self.policy = policy
        self.strict = strict
        self.state = SparseState.basis(num_wires, 0)
        self._handlers = {
            "PauliX": self._controlled_x, "CNOT": self._controlled_x,
            "Toffoli": self._controlled_x, "SWAP": self._swap, "PauliY": self._y,
            "PauliZ": lambda op: self._phase(op, -1), "S": lambda op: self._phase(op, 1j),
            "Adjoint(S)": lambda op: self._phase(op, -1j), "CZ": self._cz,
            "GlobalPhase": self._global_phase, "Hadamard": self._hadamard,
            "TemporaryAND": self._temporary_and,
            "Adjoint(TemporaryAND)": self._adjoint_temporary_and,
            "PhaseGradientStatePrep": self._phase_gradient_state_prep,
        }

    @property
    def supported_operations(self) -> frozenset[str]:
        return frozenset(self._handlers)

    def run(self, operations: Iterable, initial_state: SparseState | None = None) -> SparseState:
        self.state = (SparseState.basis(self.num_wires, 0) if initial_state is None
                      else initial_state.copy())
        for op in operations:
            self.apply(op)
        return self.state

    def apply(self, op) -> None:
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
