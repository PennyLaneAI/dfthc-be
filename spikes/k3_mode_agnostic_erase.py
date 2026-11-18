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

"""Spike K-3: a measured erase whose fixup uses measurement-derived parities runs in both modes.

Pass criteria (spec section 16, K-3), all on a read/erase pair with a dead block position and
unvisited addresses:
1. tape mode: qp.testing.assert_valid_decomp_rule passes for the read and the erase rules;
2. capture: read + erase compiles with exact, non-symbolic counts equal to the closed forms,
   and on lightning.qubit the address superposition is restored exactly for five seeds;
3. tape path: _tape_decompose with the pinned rules, simulated by default.qubit with deferred
   measurements, restores the address state exactly and leaves every ancilla in |0>.
Every warning of the run is recorded and classified, and a warning that names a prototype
operator fails the verdict (spec 7.4, rule 12).
Writes spikes/results/k3.json and exits with status 1 if any criterion fails.

Run: CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src .venv/bin/python spikes/k3_mode_agnostic_erase.py
"""  # noqa: E501

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pennylane as qp
from pennylane.transforms import _tape_decompose

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prototypes import (  # noqa: E402
    ProtoTableErase,
    ProtoTableRead,
    _proto_table_erase,
    _proto_table_read,
    classify_warnings,
    erase_toffolis,
    read_toffolis,
    to_words,
)

from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources  # noqa: E402

WIDTH, BLOCK_BITS = 2, 1
HIGH_RANGES = ((2, 0, 3),)  # blocks 0..2 visited; addresses 6 and 7 are never loaded
TABLE = tuple((a, to_words(v, WIDTH)) for a, v in enumerate([1, 2, 3, 1, 2]))  # 5 is dead
HIGH, LOW, TARGET, JUNK, WORK = [0, 1], [2], [3, 4], [5, 6], [7, 8, 9, 10]
NUM_WIRES = 11
ARGS = (TABLE, HIGH_RANGES, BLOCK_BITS, WIDTH)


def read():
    return ProtoTableRead(*ARGS, HIGH, LOW, TARGET, JUNK, WORK)


def erase():
    return ProtoTableErase(*ARGS, HIGH, LOW, TARGET, JUNK, WORK)


def round_trip():
    for wire in HIGH + LOW:
        qp.H(wire)
    read()
    erase()


def expected_state():
    state = np.zeros(2**NUM_WIRES, dtype=complex)
    for address in range(8):
        state[address << (NUM_WIRES - 3)] = 1 / np.sqrt(8)
    return state


def check_tape_rules():
    qp.decomposition.enable_graph()
    with qp.capture.pause():
        qp.testing.assert_valid_decomp_rule(read(), _proto_table_read, check_matrix=False)
        qp.testing.assert_valid_decomp_rule(erase(), _proto_table_erase, check_matrix=False)
    return True


def check_capture():
    resources = compiled_resources(round_trip, num_wires=NUM_WIRES)
    expected_t = (read_toffolis(HIGH_RANGES, BLOCK_BITS, WIDTH)
                  + erase_toffolis(HIGH_RANGES, BLOCK_BITS))
    counts_ok = (resources.toffoli, resources.measurements) == (expected_t, WIDTH * 2**BLOCK_BITS)
    states_ok = []
    for seed in range(5):
        @qp.qjit(capture=True, seed=seed)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
        @qp.qnode(qp.device("lightning.qubit", wires=NUM_WIRES))
        def circuit():
            round_trip()
            return qp.state()

        states_ok.append(bool(np.allclose(np.asarray(circuit()), expected_state(), atol=1e-10)))
    return {"toffoli": resources.toffoli, "expected_toffoli": expected_t,
            "measurements": resources.measurements, "counts_ok": counts_ok,
            "states_ok": states_ok, "passed": counts_ok and all(states_ok)}


def check_tape_path():
    qp.decomposition.enable_graph()
    with qp.queuing.AnnotatedQueue() as queue:
        round_trip()
        qp.density_matrix(wires=HIGH + LOW)
        qp.probs(wires=TARGET + JUNK + WORK)
    tape = qp.tape.QuantumScript.from_queue(queue)
    [decomposed], _ = _tape_decompose(tape, gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    names = sorted({op.name for op in decomposed.operations})
    [deferred], _ = qp.defer_measurements(decomposed)
    rho, probs = qp.devices.DefaultQubit().execute(deferred)
    plus = np.full(8, 1 / np.sqrt(8))
    address_ok = bool(np.allclose(rho, np.outer(plus, plus), atol=1e-10))
    ancillas_ok = bool(np.isclose(probs[0], 1.0, atol=1e-10))
    return {"operation_names": names, "address_restored": address_ok,
            "ancillas_zero": ancillas_ok, "passed": address_ok and ancillas_ok}


def main():
    results = {}
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        for name, check in (("tape_rules", check_tape_rules), ("capture", check_capture),
                            ("tape_path", check_tape_path)):
            try:
                results[name] = check()
            except Exception as exc:  # noqa: BLE001
                results[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    passed = all(r is True or r.get("passed") for r in results.values())
    results["warnings"] = classify_warnings(
        records, {ProtoTableRead.__name__, ProtoTableErase.__name__})
    passed = passed and not results["warnings"]["prototype"]
    results["verdict"] = "pass" if passed else "fail"
    out = Path(__file__).parent / "results" / "k3.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps(results, indent=2, default=str))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
