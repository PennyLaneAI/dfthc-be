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

"""Tape-path decomposition for simulation, and the K-3 round trip on the simulator (spec 12.3)."""

import sys
from pathlib import Path

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional, MidMeasure

from dfthc.compile import GATE_SET
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator, nonlinear_conditionals
from dfthc.testing.tape import decompose_for_simulation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "spikes"))
from k3_mode_agnostic_erase import HIGH, LOW, NUM_WIRES, round_trip  # noqa: E402

NUM_MEASUREMENTS = 4  # target || junk of the K-3 read: WIDTH * 2**BLOCK_BITS


def test_decomposition_reaches_the_gate_set():
    tape = decompose_for_simulation(round_trip)
    names = {op.name for op in tape.operations if not isinstance(op, Conditional)}
    assert names <= set(GATE_SET) and "MidMeasure" in names and "TemporaryAND" in names
    assert sum(isinstance(op, MidMeasure) for op in tape.operations) == NUM_MEASUREMENTS
    assert all(op.base.name in GATE_SET for op in tape.operations if isinstance(op, Conditional))


def test_arguments_are_forwarded_and_measurements_kept():
    def circuit(wire, *, gate):
        gate(wire)
        qp.probs(wires=[wire])

    tape = decompose_for_simulation(circuit, 2, gate=qp.H)
    assert [op.name for op in tape.operations] == ["Hadamard"]
    assert len(tape.measurements) == 1


def test_graph_flag_and_capture_are_restored():
    """The graph flag comes back disabled or enabled, as it was, and capture stays enabled."""
    graph_was_enabled = qp.decomposition.enabled_graph()
    qp.decomposition.disable_graph()
    try:
        decompose_for_simulation(round_trip)
        assert not qp.decomposition.enabled_graph()
        qp.decomposition.enable_graph()
        decompose_for_simulation(round_trip)
        assert qp.decomposition.enabled_graph()
        qp.decomposition.disable_graph()
        qp.capture.enable()
        try:
            tape = decompose_for_simulation(round_trip)
            assert qp.capture.enabled()
        finally:
            qp.capture.disable()
    finally:
        if graph_was_enabled:
            qp.decomposition.enable_graph()
    assert sum(isinstance(op, MidMeasure) for op in tape.operations) == NUM_MEASUREMENTS


@pytest.mark.parametrize("unit", [None, 0, 1, 2, 3])
def test_k3_round_trip_restores_its_state_exactly(unit):
    """Zero schedule and the four unit schedules (spec 12.5 basis runs), strict mode on. Every
    conditional of the tape is diagonal with a GF(2)-linear predicate, which the basis runs'
    conclusion for all outcome vectors needs."""
    schedule = [int(i == unit) for i in range(NUM_MEASUREMENTS)]
    policy = ForcedPolicy(schedule)
    operations = decompose_for_simulation(round_trip).operations
    assert not nonlinear_conditionals(operations)
    state = SparseSimulator(NUM_WIRES, policy=policy).run(operations)
    expected = np.zeros(2**NUM_WIRES, dtype=complex)
    address_bits = len(HIGH + LOW)
    for address in range(2**address_bits):
        expected[address << (NUM_WIRES - address_bits)] = 1 / np.sqrt(2**address_bits)
    assert policy.position == NUM_MEASUREMENTS
    assert np.allclose(state.to_dense(), expected, rtol=0, atol=1e-12)
