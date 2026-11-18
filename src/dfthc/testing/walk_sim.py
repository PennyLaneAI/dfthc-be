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

r"""Sparse-simulator harness of the block encoding and the walk (spec 12.3 and 12.4).

A :class:`WalkCase` holds one walk's parameters, tables, factors and wires: the ``WalkLayout`` of
the parameters plus one more wire, ``control``, for the controlled programs (the uncontrolled
programs leave it in ``|0>``). A program (:func:`program`, :func:`operations`) prepares the
phase-gradient state on ``phase_grad`` and then applies the assembly operators; it is decomposed
once to gate-set operations on the tape path. :func:`prepare_unprepare` is the block encoding
without its reflection, :math:`U^\dagger U` (spec 12.4, E-4), and
:func:`controlled_block_encoding` the controlled walk's first 27 steps. An input
(:func:`input_state`) is a vector of amplitudes on the system ``psi_up (x) psi_dn``
(:func:`basis`, :func:`superposition`), optionally times a control state, with every other wire
in ``|0>``.

:func:`project` reads the block of an output: for each control value and system basis state, the
amplitude with every other wire in ``|0>`` and ``phase_grad`` contracted with the gradient state,
which is ``<0|BE|0>`` applied to the input (spec 12.4). :func:`assert_same_state` compares two
whole outputs, ancillas included, so a dirty ancilla, a wrong sign outside the block or a wrong
global phase under control shows. The expected blocks come from the encoded model
(:func:`dfthc.data.encoded.encoded_block`), never from the circuit. Every run is strict: each
erase measurement must have probability 1/2, and every AND precondition is asserted (spec 12.3).

:data:`CASES` names the shapes of the assembly tests: the two test presets and
:data:`FAILS_C3`. At most 64 wires make a key one 64-bit word; wider programs (``H2_631G_TEST``:
68 wires) are correct but about 25 times slower per input (``np.unique`` over two-word keys).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pennylane as qp

from dfthc.data.encoded import encoded_block
from dfthc.data.factors import DFTHCFactors, load_factors, random_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.reflections import ref_t2
from dfthc.ops.walk import _children, _emit, _registers
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, WalkParams
from dfthc.registers import GROUPS, WalkLayout
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator, SparseState
from dfthc.testing.tape import decompose_for_simulation

ATOL = 1e-12
"""Absolute tolerance of the state comparisons (docs/verification.md)."""

AB_REGISTERS = ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0")
r"""The registers of :math:`\mathcal{A} \cup \mathcal{B}` that ``ref_ab`` reflects (spec 3.3)."""


@dataclass(frozen=True)
class WalkCase:
    """One walk on the sparse simulator: parameters, tables, factors and wires."""

    params: WalkParams
    tables: WalkTables
    factors: DFTHCFactors
    layout: WalkLayout

    @classmethod
    def build(cls, params: WalkParams, factors: DFTHCFactors) -> WalkCase:
        return cls(params, WalkTables.from_factors(factors, params), factors,
                   WalkLayout.from_params(params))

    @property
    def control(self) -> list[int]:
        """The control wire of the controlled programs, right after the layout."""
        return [self.layout.num_wires]

    @property
    def num_wires(self) -> int:
        return self.layout.num_wires + 1

    @property
    def system(self) -> list[int]:
        """``psi_up`` then ``psi_dn``: the order of the dense fermion matrices (spec 6.2)."""
        return list(self.layout.psi_up) + list(self.layout.psi_dn)

    @property
    def compilable(self) -> tuple[tuple, tuple]:
        """``(params, tables)``, the compilable arguments of the assembly operators."""
        return self.params.as_compilable(), self.tables.as_compilable()

    def groups(self, controlled: bool = False) -> dict[str, list[int]]:
        """The wire groups of an assembly operator, ``control`` first if ``controlled`` (as
        ``ControlledWalkPower`` takes them)."""
        groups = {name: list(wires) for name, wires in self.layout.groups().items()}
        return {"control": self.control} | groups if controlled else groups

    def block(self) -> np.ndarray:
        """The encoded block ``H~_SA / Lambda~ - I`` (spec 8.5)."""
        h_tilde, lambda_tilde = encoded_block(self.params, self.tables, self.factors)
        return h_tilde / lambda_tilde - np.eye(len(h_tilde))


FAILS_C3 = WalkParams(N=2, R=3, B=2, C=3, n_plus=1, b_rot=4, b_k1=1, b_k2=2, s_u=3, k1=0, k2=0,
                      ke1=1, ke2=1)
"""A shape the test presets cannot be, with ``random_factors(2, 3, 2,
3, n_plus=1, seed=5)``: ``X_o = 11``, so the uniform preparation of ``x_o`` fails (P = 0.11),
``REF_T2`` is ``-I`` there and ``Lambda~ = Lambda / (1 - P)``; ``B + 1 = 3`` fails too, and with
``k2 = 0`` the failed ``b = 3`` reads padding in our read but entry ``B`` in dfthcbe's truncated
iteration, so dfthcbe's failure signs differ from ours (spec 8.5); ``R = C = 3``, so
``r = x_o // 3`` is no bit field of ``x_o``; ``b_k1 != b_k2``. With the control wire, 59 wires."""

CASES = {
    "sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "fails-c3": lambda: (FAILS_C3, random_factors(2, 3, 2, 3, n_plus=1, seed=5)),
}
"""The assembly test shapes: ``H2_STO3G_TEST`` (56 wires with the control), ``H2_631G_TEST``
(68) and :data:`FAILS_C3` (59), each with its factors."""


@functools.cache
def case(name: str) -> WalkCase:
    """The :class:`WalkCase` of :data:`CASES` ``[name]``, built once per process."""
    return WalkCase.build(*CASES[name]())


def operations(case: WalkCase, *calls: Callable[[], None]) -> list:
    """The gate-set operations of ``PhaseGradientStatePrep(phase_grad)`` followed by ``calls``."""
    def program():
        qp.PhaseGradientStatePrep(case.layout.phase_grad)
        for call in calls:
            call()

    return list(decompose_for_simulation(program).operations)


def program(case: WalkCase, cls: type, controlled: bool = False, repeat: int = 1,
            control_value: int = 1) -> list:
    """The decomposed program: the phase gradient, then ``cls`` (an assembly class) on the case's
    groups, ``repeat`` times; if ``controlled``, ``qp.ctrl`` of it on the case's control with
    ``control_value``."""
    def call():
        op = cls(*case.compilable, **case.groups())
        if controlled:
            qp.ctrl(op, control=case.control, control_values=[control_value])

    return operations(case, *[call] * repeat)


def controlled_block_encoding(case: WalkCase) -> None:
    r"""The controlled walk's first 27 steps on the case's groups and control, the controlled
    block encoding :math:`U^\dagger\,C(\mathrm{REF}_{T_2})\,U` (``dfthc.ops.walk``'s own list, in
    its order and on its registers); a quantum function for tape mode and capture."""
    g = case.groups()
    registers = _registers(case.params, g["outer"], g["inner"], g["select"], g["system"],
                           g["phase_grad"], case.control)
    _emit(_children(case.params, case.tables, controlled=True), registers, g["work"])


def prepare_unprepare(case: WalkCase) -> list:
    r"""The decomposed :math:`U^\dagger U`: the block encoding's children (``dfthc.ops.walk``'s
    own list, in its order and on its registers) without ``ref_t2`` (spec 12.4, E-4)."""
    children = [child for child in _children(case.params, case.tables, controlled=False)
                if child.op is not ref_t2]
    g = case.groups()
    registers = _registers(case.params, g["outer"], g["inner"], g["select"], g["system"],
                           g["phase_grad"])
    return operations(case, lambda: _emit(children, registers, g["work"]))


def num_measurements(ops: Sequence) -> int:
    """The erase measurements of a decomposed program."""
    return sum(op.name == "MidMeasure" for op in ops)


def schedule(kind: str, num: int, seed: int = 0) -> list[int]:
    """Forced outcomes (spec 12.5): ``"S0"`` every outcome 0, ``"S1"`` every outcome 1,
    ``"alternating"`` 0, 1, 0, ..., ``"seeded"`` random bits from ``seed``."""
    if kind == "S0":
        return [0] * num
    if kind == "S1":
        return [1] * num
    if kind == "alternating":
        return [i % 2 for i in range(num)]
    if kind == "seeded":
        return np.random.default_rng(seed).integers(0, 2, num).tolist()
    raise ValueError(f"unknown schedule {kind!r}")


def _keys(num_wires: int, wires: Sequence[int], values) -> np.ndarray:
    """Basis keys with ``values`` big-endian on ``wires`` and every other wire 0."""
    keys = np.zeros((len(values), max(1, -(-num_wires // 64))), dtype=np.uint64)
    values = np.asarray(values, dtype=np.uint64)
    for position, wire in enumerate(wires):
        bit = (values >> np.uint64(len(wires) - 1 - position)) & np.uint64(1)
        keys[:, wire // 64] |= bit << np.uint64(63 - wire % 64)
    return keys


def input_state(case: WalkCase, system, control=None) -> SparseState:
    """``control (x) system`` on (``control``, ``psi_up``, ``psi_dn``), every other wire in
    ``|0>``; ``control`` has two amplitudes and defaults to ``|0>``."""
    control = np.array([1.0, 0.0]) if control is None else np.asarray(control)
    amps = np.kron(control, np.asarray(system, dtype=complex))
    values = np.flatnonzero(amps)
    keys = _keys(case.num_wires, [*case.control, *case.system], values)
    return SparseState(case.num_wires, keys, amps[values])


def basis(case: WalkCase, value: int) -> np.ndarray:
    """The system basis vector ``|value>``, ``psi_up``'s bits first."""
    vector = np.zeros(4**case.params.N, dtype=complex)
    vector[value] = 1
    return vector


def superposition(case: WalkCase, values: Sequence[int], seed: int) -> np.ndarray:
    """A normalized superposition of the system basis states ``values`` with random complex
    amplitudes from ``seed``."""
    rng = np.random.default_rng(seed)
    vector = np.zeros(4**case.params.N, dtype=complex)
    vector[list(values)] = rng.normal(size=len(values)) + 1j * rng.normal(size=len(values))
    return vector / np.linalg.norm(vector)


def simulate(case: WalkCase, ops: Sequence, initial: SparseState,
             outcomes: Sequence[int]) -> SparseState:
    """Run ``ops`` from ``initial`` in strict mode with the forced ``outcomes``, all of them."""
    simulator = SparseSimulator(case.num_wires, policy=ForcedPolicy(outcomes), strict=True)
    state = simulator.run(ops, initial_state=initial)
    if simulator.policy.position != len(outcomes):
        raise AssertionError(f"the program made {simulator.policy.position} measurements; "
                             f"{len(outcomes)} outcomes were forced")
    return state


def gradient(b_grad: int) -> np.ndarray:
    """The phase-gradient state the simulator prepares on ``b_grad`` wires."""
    return SparseSimulator(b_grad).run([qp.PhaseGradientStatePrep(range(b_grad))]).to_dense()


def _values(state: SparseState, wires: Sequence[int], rows: np.ndarray) -> np.ndarray:
    total = np.zeros(int(rows.sum()), dtype=np.int64)
    for wire in wires:
        total = (total << 1) | state.bits(wire)[rows]
    return total


def project(case: WalkCase, state: SparseState) -> np.ndarray:
    """``(2, 4**N)``: for control 0 and 1 and every system basis state, the amplitude with every
    other wire in ``|0>`` and ``phase_grad`` contracted with the gradient state."""
    kept = {*case.system, *case.layout.phase_grad, *case.control}
    rows = np.ones(len(state.amps), dtype=bool)
    for wire in range(case.num_wires):
        if wire not in kept:
            rows &= ~state.bits(wire)
    grad_wires = list(case.layout.phase_grad)
    weights = np.conj(gradient(len(grad_wires))[_values(state, grad_wires, rows)])
    out = np.zeros((2, 4**case.params.N), dtype=complex)
    np.add.at(out, (_values(state, case.control, rows), _values(state, case.system, rows)),
              state.amps[rows] * weights)
    return out


def nonzero_registers(case: WalkCase, state: SparseState) -> set[str]:
    """The registers of the ``outer``, ``inner``, ``select`` and ``work`` groups that are 1 on
    some wire of some term of ``state``."""
    return {name for group in ("outer", "inner", "select", "work") for name in GROUPS[group]
            if any(state.bits(wire).any() for wire in getattr(case.layout, name))}


def reflect_ab(case: WalkCase, state: SparseState) -> SparseState:
    """``ref_ab``'s exact action on ``state``: -1 on every term with a nonzero
    :data:`AB_REGISTERS` wire (spec 3.3)."""
    nonzero = np.zeros(len(state.amps), dtype=bool)
    for name in AB_REGISTERS:
        for wire in getattr(case.layout, name):
            nonzero |= state.bits(wire)
    return SparseState(state.num_wires, state.keys.copy(), np.where(nonzero, -1, 1) * state.amps)


def set_control(case: WalkCase, state: SparseState) -> SparseState:
    """``state`` with the control wire set to 1 on every term (it must be 0 on every term)."""
    if state.bits(case.control[0]).any():
        raise ValueError("the control wire is not 0 on every term")
    result = state.copy()
    result.flip(case.control[0])
    return result


def superpose(*terms: tuple[complex, SparseState]) -> SparseState:
    """``sum_i c_i |state_i>`` over states of one width, equal keys merged."""
    num_wires = terms[0][1].num_wires
    keys = np.concatenate([state.keys for _, state in terms])
    amps = np.concatenate([coefficient * state.amps for coefficient, state in terms])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    summed = (np.bincount(inverse, amps.real, len(unique))
              + 1j * np.bincount(inverse, amps.imag, len(unique)))
    return SparseState(num_wires, unique, summed)


def assert_same_state(state: SparseState, other: SparseState, atol: float = ATOL) -> None:
    """Two outputs are equal term by term to ``atol``, every basis state of either included."""
    difference = superpose((1, state), (-1, other))
    largest = np.abs(difference.amps).max(initial=0.0)
    assert np.isclose(largest, 0, rtol=0, atol=atol), largest
