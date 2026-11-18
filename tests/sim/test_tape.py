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

"""Tape-path decomposition for simulation (spec 12.3), on a QROAM read and its measured erase.
The round trip's state is certified in ``tests/ops/test_certificates_qroam.py``."""

import pennylane as qp
from pennylane.ops import Conditional, MidMeasure

from dfthc.compile import GATE_SET
from dfthc.ops.qroam import QROAMErase, QROAMRead
from dfthc.testing.tape import decompose_for_simulation

WIDTHS, LIVE, WIDTH, BLOCK_BITS, ONEHOT_BITS = (3,), ((0, 5),), 2, 1, 0
TABLE = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))  # addresses 5 to 7 are dead
NUM_MEASUREMENTS = WIDTH * 2**BLOCK_BITS


def round_trip():
    sizes = QROAMErase.required_wires((), WIDTHS, LIVE, WIDTH, BLOCK_BITS, None, ONEHOT_BITS)
    read_work = QROAMRead.required_wires(WIDTHS, LIVE, WIDTH, BLOCK_BITS, None)["work"]
    sizes["work"] = max(sizes["work"], read_work)
    wires = qp.registers(sizes)
    columns = QROAMRead.columns_from_table(TABLE, address_widths=WIDTHS, live_ranges=LIVE,
                                           width=WIDTH, block_bits=BLOCK_BITS)
    for wire in wires["address"]:
        qp.H(wire)
    QROAMRead(columns, WIDTHS, LIVE, WIDTH, BLOCK_BITS, None, wires["address"], wires["target"],
              wires["junk"], wires["work"][:read_work], wires["control"])
    QROAMErase(TABLE, WIDTHS, LIVE, WIDTH, BLOCK_BITS, None, ONEHOT_BITS, wires["address"],
               wires["target"], wires["junk"], wires["work"], wires["control"],
               wires["extra_onehot"])


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
