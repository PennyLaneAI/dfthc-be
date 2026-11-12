"""Simulator benchmark gate (spec 12.6): the K-1 stand-in walk on the tape path and the sparse
simulator, at the shapes of the two test presets, each run under the watchdog with a 4 GB cap.

The stand-in (``spikes/k1_whole_program.py``, Task 0.9) has the walk's structure and sizes, so it
is the Phase 1 proxy; Task 3.5 measures again on the real walk. Every erase measurement is forced
to 0 and strict mode checks the uniform outcomes. ``python tests/sim/test_benchmark.py SHAPE
INPUT...`` runs one measurement and prints its JSON result on a line prefixed ``RESULT``.
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

from dfthc.testing.watchdog import effective_limit_mb, run_watched

REPO = Path(__file__).resolve().parents[2]
SHAPES = {
    "h2_sto3g": dict(N=2, R=2, B=2, C=1, n_plus=1, b_rot=4, b_k1=2, b_k2=2, k1=1, k2=1),
    "h2_631g": dict(N=4, R=2, B=2, C=2, n_plus=3, b_rot=4, b_k1=2, b_k2=2, k1=1, k2=1),
}
# Workload pin: a change means the stand-in or PennyLane's decompositions changed; re-measure.
WORKLOAD = {"h2_sto3g": dict(num_wires=66, operations=1108, measurements=50),
            "h2_631g": dict(num_wires=82, operations=2123, measurements=68)}
INPUTS = {"h2_sto3g": tuple(range(16)), "h2_631g": (0b1000_1000,)}
"""Basis inputs of psi_up (x) psi_dn, psi_up's bits first: all 16 at H2_STO3G_TEST (one E-1
schedule), and one electron per spin in orbital 0 at H2_631G_TEST."""
BUDGET_S = {"h2_sto3g": 20 * 60, "h2_631g": 10 * 60}
CAP_MB = 4096
PREFIX = "RESULT "


def simulate(shape_name: str, inputs) -> dict:
    """Build and decompose the stand-in walk once, then simulate every input (subprocess side)."""
    import pennylane as qp

    sys.path.insert(0, str(REPO / "spikes"))
    import k1_whole_program as k1

    from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator, SparseState
    from dfthc.testing.tape import decompose_for_simulation

    shape = k1.Shape(**SHAPES[shape_name])
    layout, tables = k1.Layout(shape), k1.random_tables(shape)
    fields = tuple(getattr(shape, name) for name in k1.Shape.__dataclass_fields__)

    def walk():
        qp.PhaseGradientStatePrep(layout.phase_grad)
        k1.ProtoWalk(fields, tables, list(range(layout.num_wires)))

    start = time.perf_counter()
    operations = decompose_for_simulation(walk).operations
    decompose_s = time.perf_counter() - start
    measurements = sum(op.name == "MidMeasure" for op in operations)
    postselected = sum(op.name == "MidMeasure" and op.postselect is not None for op in operations)
    if postselected:
        raise RuntimeError(f"{postselected} postselected measurements are not benchmarked")
    system = layout.psi_up + layout.psi_dn
    seconds, terms = [], []
    for value in inputs:
        index = sum(((value >> (len(system) - 1 - k)) & 1) << (layout.num_wires - 1 - wire)
                    for k, wire in enumerate(system))
        sim = SparseSimulator(layout.num_wires, policy=ForcedPolicy([0] * measurements),
                              strict=True)
        start = time.perf_counter()
        state = sim.run(operations, initial_state=SparseState.basis(layout.num_wires, index))
        seconds.append(round(time.perf_counter() - start, 2))
        if sim.policy.position != measurements:
            raise RuntimeError(f"input {value}: the forced policy consumed "
                               f"{sim.policy.position} of {measurements} outcomes")
        terms.append(len(state.amps))
    return {"shape": shape_name, "num_wires": layout.num_wires, "operations": len(operations),
            "measurements": measurements, "inputs": list(inputs),
            "decompose_s": round(decompose_s, 2), "seconds": seconds, "final_terms": terms}


def run_benchmark(shape_name: str, inputs, timeout_s: float):
    """One measurement in a subprocess under ``run_watched`` (cap 4 GB, killed at 3.75 GB)."""
    command = [sys.executable, str(Path(__file__).resolve()), shape_name,
               *(str(value) for value in inputs)]
    env = os.environ | {"PYTHONPATH": str(REPO / "src"),
                        "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "0"}
    run = run_watched(command, cap_mb=CAP_MB, timeout_s=timeout_s, cwd=REPO, env=env)
    if run.killed is not None or run.survivors or run.returncode != 0:
        return run, None
    lines = [line for line in run.stdout.splitlines() if line.startswith(PREFIX)]
    assert len(lines) == 1, _diagnostics(run, f"expected one RESULT line, got {len(lines)}")
    try:
        return run, json.loads(lines[0][len(PREFIX):])
    except json.JSONDecodeError as error:
        raise AssertionError(_diagnostics(run, f"malformed RESULT line: {error}")) from error


def _diagnostics(run, message: str) -> str:
    return f"{message}\nstdout (tail): {run.stdout[-1000:]}\nstderr (tail): {run.stderr[-2000:]}"


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_benchmark_gate(shape_name):
    run, result = run_benchmark(shape_name, INPUTS[shape_name], BUDGET_S[shape_name])
    print("BENCHMARK " + json.dumps({"watchdog": run.summary(), "result": result}))
    assert run.killed is None and not run.survivors and run.returncode == 0, _diagnostics(
        run, f"killed={run.killed!r} survivors={run.survivors!r} status={run.returncode}")
    assert run.seconds <= BUDGET_S[shape_name]
    assert run.peak_mb <= effective_limit_mb(CAP_MB)
    assert {key: result[key] for key in WORKLOAD[shape_name]} == WORKLOAD[shape_name]
    assert result["shape"] == shape_name
    assert result["inputs"] == list(INPUTS[shape_name])
    assert len(result["seconds"]) == len(INPUTS[shape_name])


if __name__ == "__main__":
    print(PREFIX + json.dumps(simulate(sys.argv[1], [int(value) for value in sys.argv[2:]])))
