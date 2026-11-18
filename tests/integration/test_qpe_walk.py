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

"""Iterative phase estimation of the DFTHC walk itself on the sparse simulator (spec 11.1): the
branch distribution of the control bits equals the exact distribution the encoded model predicts, at
H2_STO3G_TEST (t = 1 nightly, t = 2 release) and at FAILS_C3 (t = 1 and t = 2 nightly), with the
erase measurements forced by the BranchPolicy's `otherwise` and every run strict. The reference
formula is checked against the toy walk per commit."""

import numpy as np
import pytest

from dfthc.qpe import phase_from_bits, qpe_program
from dfthc.registers import WalkLayout
from dfthc.testing import walk_sim
from dfthc.testing.sparse_sim import BranchPolicy, ForcedPolicy, SparseSimulator, SparseState
from dfthc.testing.tape import decompose_for_simulation
from dfthc.testing.toy_walk import toy_walk_matrix


def qpe_distribution(block: np.ndarray, psi: np.ndarray, num_bits: int) -> np.ndarray:
    r"""P(m), m = 2^t phi, of t-bit phase estimation of the qubitization walk whose block on the
    ancillas' |0> is ``block``, from the ancillas in |0> and the system in ``psi``. Each
    eigenvalue a of the block splits its weight equally between the walk phases
    :math:`\pm\arccos(a)/2\pi`, and a phase phi gives m with probability
    :math:`|2^{-t}\sum_{x<2^t} e^{2\pi i x(\varphi - m/2^t)}|^2` (the semiclassical estimation
    has the textbook distribution; Griffiths and Niu 1996)."""
    values, vectors = np.linalg.eigh(block)
    weights = np.abs(vectors.conj().T @ psi) ** 2
    x = np.arange(2**num_bits)
    probabilities = np.zeros(2**num_bits)
    for value, weight in zip(values, weights, strict=True):
        theta = np.arccos(np.clip(value, -1, 1))
        for phi in (theta / (2 * np.pi), -theta / (2 * np.pi)):
            for m in range(2**num_bits):
                amplitude = np.exp(2j * np.pi * x * (phi - m / 2**num_bits)).sum() / 2**num_bits
                probabilities[m] += weight / 2 * abs(amplitude) ** 2
    return probabilities


def test_reference_distribution_reproduces_the_toy():
    """The reference formula on the toy walk's block (H = (Z + X)/2) from |0> gives spec 11.4's
    distribution at t = 3: cos^2(pi/8)/2 at m = 1, 7 and sin^2(pi/8)/2 at m = 3, 5."""
    block = toy_walk_matrix()[np.ix_([0, 2], [0, 2])]
    expected = np.zeros(8)
    expected[[1, 7]] = np.cos(np.pi / 8) ** 2 / 2
    expected[[3, 5]] = np.sin(np.pi / 8) ** 2 / 2
    assert np.allclose(qpe_distribution(block, np.array([1.0, 0.0]), 3), expected, rtol=0,
                       atol=1e-12)


INPUTS = {"sto3g": 10, "fails-c3": 5}
"""Two-electron system inputs, psi_up's bits first: |10>|10> and |01>|01>."""


def gradient_weight(state: SparseState, wires) -> float:
    """The squared norm of ``state``'s projection onto (anything) (x) the phase-gradient state on
    ``wires``: 1 exactly when those wires still hold it."""
    value, rest = np.zeros(len(state.amps), dtype=np.int64), state.copy()
    for wire in wires:
        value = (value << 1) | rest.bits(wire)
        rest.flip(wire, rest.bits(wire))
    _, group = np.unique(rest.keys, axis=0, return_inverse=True)
    terms = np.conj(walk_sim.gradient(len(wires))[value]) * state.amps
    projected = (np.bincount(group.ravel(), terms.real)
                 + 1j * np.bincount(group.ravel(), terms.imag))
    return float(np.sum(np.abs(projected) ** 2))


@pytest.mark.parametrize(("name", "num_bits", "kind"), [
    pytest.param("sto3g", 1, "seeded", marks=pytest.mark.slow),
    pytest.param("fails-c3", 1, "alternating", marks=pytest.mark.slow),
    pytest.param("fails-c3", 1, "seeded", marks=pytest.mark.slow),
    pytest.param("fails-c3", 2, "seeded", marks=pytest.mark.slow),
    pytest.param("sto3g", 2, "seeded", marks=pytest.mark.release),
])
def test_qpe_of_the_walk_is_exact(name, num_bits, kind):
    """The QPE program of qpe_program (ControlledWalkPower in every round, the earlier bits'
    corrections on wires of the walk's work pool, the walk's state carried between rounds) on
    the tape path: the probabilities of the control bits equal qpe_distribution of the encoded
    block H~_SA / Lambda~ - I to 1e-12, and every branch ends with only the index registers of
    A u B nonzero (the work pool back in |0>) and with the walk's phase_grad and qpe_grad in the
    gradient state (a last-round correction that disturbs another register leaves the bits
    unchanged; only this sees it). BranchPolicy follows both outcomes of each control
    measurement; the erase measurements go to its `otherwise`, a forced schedule that every
    branch path reads from its own position, and strict mode asserts each has probability 1/2,
    so the unweighted forced outcomes leave the branch probabilities exact (spec 12.3, 12.5)."""
    c = walk_sim.case(name)
    layout = WalkLayout.from_params(c.params, qpe_bits=num_bits)
    program, num_wires = qpe_program(c.params, c.tables, num_bits)
    ops = decompose_for_simulation(program).operations
    controls = [op for op in ops if op.name == "MidMeasure"
                and int(op.wires[0]) == layout.qpe_control[0]]
    assert len(controls) == num_bits
    erasures = walk_sim.num_measurements(ops) - num_bits
    schedule = walk_sim.schedule(kind, erasures, seed=12)
    assert list(layout.system) == list(range(2 * c.params.N))
    initial = SparseState.basis(num_wires, INPUTS[name] << (num_wires - 2 * c.params.N))
    simulator = SparseSimulator(num_wires, strict=True, policy=BranchPolicy(
        layout.qpe_control, otherwise=ForcedPolicy(schedule)))
    found = np.zeros(2**num_bits)
    for branch in simulator.run_branches(ops, initial_state=initial):
        bits = [branch.record[op] for op in controls]
        found[round(phase_from_bits(bits) * 2**num_bits)] += branch.probability
        assert walk_sim.nonzero_registers(c, branch.state) <= set(walk_sim.AB_REGISTERS)
        for register in (layout.phase_grad, layout.qpe_grad):
            assert np.isclose(gradient_weight(branch.state, register), 1, rtol=0, atol=1e-12)
    psi = walk_sim.basis(c, INPUTS[name])
    assert np.allclose(found, qpe_distribution(c.block(), psi, num_bits), rtol=0, atol=1e-12)
