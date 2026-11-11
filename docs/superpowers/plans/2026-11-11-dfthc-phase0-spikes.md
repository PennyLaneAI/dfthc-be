# DFTHC Phase 0, Part 3: Feasibility Spikes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the four Phase 0 risks of spec section 16 before any production operator is written: K-7 (adjoints of PennyLane comparators), K-3 (a measured-erase rule in both tape and capture mode), K-10 (measurement-valued arrays as dynamic `Operator2` arguments) and K-1 (compile time and memory of the complete walk and controlled walk: measured at small and mid shapes and projected to FeMoco-54, which is compiled only if the projection fits this machine), and record each result.

**Architecture:** K-7 becomes a permanent capability test. K-3, K-10 and K-1 are scripts under `spikes/` that print and write a JSON verdict to `spikes/results/<id>.json`; they share `spikes/prototypes.py` (exact unary iteration, a QROAM read and its measured erase). Prototypes are not production code: Phase 1 rewrites them from the spec with full tests. Spike scripts are not collected by pytest (`testpaths = ["tests"]`).

**Tech Stack:** as Part 1, plus `lightning.qubit` and `default.qubit` for exact state checks.

This file holds Tasks 0.6 to 0.9 of the master plan. Every script below was dry-run on 2026-11-11. K-7 and K-3 passed in full. K-10 passed at small size; its mid-size compile path ran at 64 and 128 addresses, and an earlier version compiled the mid sizes themselves. The K-1 driver passed at the small shape, and its projection and FeMoco gate ran on canned calibration data; no FeMoco-shaped compile ran in round 3.

Two limits shape Tasks 0.8 and 0.9 (master plan section 12.9): no process tree may exceed 8 GB, and FeMoco-shaped compiles are kept to about six for the whole project. Indicative FeMoco-shape timings from the K-10 dry-run while writing this plan, taken on the development laptop with other compiles running, are quoted, not reproduced:
- one inner-erase-shaped fixup over 8,640 addresses, with parities packed inside the rule, compiled in 199 to 274 s;
- the same fixup with its masks passed as compilable data took 1,517 s.

The K-1 calibration of master plan section 3, item 9, projects the FeMoco stand-in walk to 12 to 18 GB, above the cap. Expect Task 0.9 to end with an escalation at G0 rather than a FeMoco compile.

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the
spec (`docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md`); section numbers refer to it.

**Standing requirements (spec 2.1)**
- R-1: Use the latest nightly PennyLane and Catalyst, installed as in §7.1.
- R-2: Compile with `qp.qjit(capture=True)` and `qp.decompose` (the graph decomposition system). Obtain resources with `qp.analyze`.
- R-3: Decomposition rules are lowered as part of the `qjit` call. The precompiled decomposition-rule bytecode file is not used.
- R-4: Every building block is an `Operator2` with clear documentation. The code is modular. Documented exemption, decided by the user (§16, Q-3): a top-level algorithm that returns classical measurement values, namely the iterative QPE driver of §11, is a plain quantum function. Its quantum content per round remains `Operator2`s (`QPERound`, `ControlledWalkPower`).
- R-5: Resource-aware tests accompany all development.
- R-6: Verifiability, testability and comprehensiveness: other people must be able to understand what was done and how the code works.
- R-7: No workarounds to match the paper's or dfthcbe's resource counts beyond the paper's own constructions. Deltas are documented.
- R-8: Scope: block encoding, walk, controlled walk, and an iterative QPE wrapper on the controlled walk. All are compiled and resource-counted. QPE is simulated only at toy sizes.
- R-9: Gate set: Toffoli level, with rotations done by phase-gradient addition (§7.3).
- R-10: Verification: a purpose-built sparse-state simulator checks ⟨0|BE|0⟩ = H_SA/Λ − I end to end on H2. Correctness for every erase-measurement outcome is certified at block level (§12.5). Per-block checks run on `lightning.qubit`.
- R-11: Git: local branches only. Never push and never modify remotes.

**Toolchain (spec header and 7.1)**
- Toolchain snapshot: PennyLane `0.46.0.dev151`, Catalyst `0.16.0.dev140`, Lightning `0.46.0.dev34` (nightlies of 2026-11-11).
- One canonical environment serves all agents and worktrees: `.venv` in the repository root, CPython 3.12.
- Always use plain `pip`; never `uv pip`.
- Export `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` before Python starts.
- Do not install the package in editable mode. Use pytest's `pythonpath = ["src"]`.
- Reference fixtures come from `.venv-ref`, a separate environment with Qualtran and dfthcbe at commit `930255020cec30172e0a7d17c0071d9476a687d1` (Q-2, decided: yes).

**Gate set and accounting (spec 7.3 and 7.6)**
```python
GATE_SET = {
    "PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "Adjoint(S)": 1,
    "CNOT": 1, "CZ": 1, "SWAP": 1, "GlobalPhase": 1, "MidMeasure": 1,
    "Adjoint(TemporaryAND)": 1, "PhaseGradientStatePrep": 1,
    "Toffoli": 1000, "TemporaryAND": 1000,
}
```
- Toffoli count T: number of `Toffoli` plus number of `TemporaryAND`. M: number of `MidCircuitMeasure`, the explicit erase measurements. A: number of `Adjoint(TemporaryAND)`, the measured AND uncomputes.
- What tests assert: Toffoli and measurement counts are exact and asserted against `dfthc.costs` closed forms. Clifford counts are reported, not asserted.
- Baseline at FeMoco-54 (§9.24): 9,793 Toffolis per walk and 9,795 per controlled walk; M = 2,524; A = 8,675 (controlled 8,677); M + A = 11,199 (controlled 11,201).

**Rules for operators and decomposition rules (spec 7.4, one line each)**
1. No allocation inside rules: no `qp.allocate`, no `work_wires=` in `qp.register_resources`; every ancilla is an explicit wire argument; every operator exposes `required_wires(**compilable_args) -> dict[str, int]`.
2. Resource-function keys are `Operator2` abstract instances built through `dfthc.compile.resource_key`; every emitted operator is declared; counts are Python integers; resource functions never raise.
3. Every argument that determines circuit structure is in `compilable_argnames`; `static_argnames` are not used; `dynamic_argnames` only for runtime data (`PhaseFixup` parities, `QPERound` earlier bits, K-1 remedy tables).
4. Bodies of auto-adjointed operators contain no `qp.for_loop` or `qp.cond`; `qp.cond` only on mid-circuit measurement outcomes in erase operators and the QPE wrapper; `qp.for_loop` only in `ControlledWalkPower`.
5. Measurement-based uncompute lives only in `QROAMErase`, `AngleErase` and explicit `Adjoint(Op)` rules with signature `(base)`; an operator that contains a measurement is never placed under `qp.ctrl`.
6. Controlled versions come only from explicit `C(Op)` rules with signature `(base, control_wires, control_values, work_wires, work_wire_type)`.
7. One base rule per operator; explicit adjoint and controlled rules are pinned (§7.5).
8. Unique class names; a test asserts that PennyLane registers none of our names.
9. Rules run in both capture and tape mode; mode-specific classical code is confined to helpers (`dfthc.compile.mcm_parities`).
10. Exact swaps only (`CSWAP`).
11. No `Subroutine`, and no `qp.pow` on our operators.
12. Tests fail on any `RuleLoweringWarning` that names a dfthc operator, and filter the PennyLane noise of §7.1.

**Rules added by the Phase 0 dry-run (2026-11-11; canaries L-24, L-25)**
- Compilable integers must fit in signed 64 bits; table entries are tuples of 32-bit words, least significant word first.
- Apply adjoints as `qp.adjoint(Op(...))`, never `qp.adjoint(Op)(*positional_args)`: under capture the positional form traces compilable arguments.

**Conventions (spec 6.7)**
- Integers in registers are big-endian: the first wire is the most significant bit.
- Register names follow dfthcbe: `x_o`, `succ_x_o`, `sigma_1`, `m_x_o`, `less_than_x_o`, `alt_x_o`, `keep_x_o`, `junk_x_o`, `b`, `succ_b`, `sigma_0`, `m_b`, `less_than_b`, `alt_b`, `keep_b`, `sgn`, `d_sgn`, `r`, `junk_b`, `angles`, `G0`, `G1`, `b_neq_B`, `sg`, `psi_up`, `psi_dn`, `phase_grad`, `work`.

**Tests (spec 12.6)**
- Markers: `compiled`, `slow`, `release`, `canary`, `femoco`, `subprocess`. `release` implies `slow`.
- `scripts/test.sh` runs 2 xdist workers at lowered priority (`DFTHC_TEST_WORKERS` overrides).
- Per commit: `scripts/test.sh -m "not slow and not release"` (15 min). Nightly: `scripts/test.sh -m "slow and not release"` (4 h), only at phase gates, started by the orchestrator. Release: `scripts/test.sh` (8 h), once, at the end of Phase 5, after the user's go-ahead.

**Load limits (master plan section 12.9; spec 15)**
- The development laptop has 10 cores and 16 GB, and the user works on it. No process tree may exceed 8 GB of memory.
- A heavy process is a run whose process tree exceeds 4 GB, keeps more than 2 cores busy or lasts longer than 30 minutes, or any FeMoco-shaped compile. Only the orchestrator starts heavy processes, one at a time, under the watchdog (`dfthc.testing.watchdog`), and logs each in the ledger.
- During TDD, run only the targeted tests; run the per-commit tier once at commit time. Never start a heavy process yourself.
- The per-commit and nightly tiers contain no FeMoco-shaped compile. FeMoco-54 values are asserted on the closed forms (`dfthc.costs`); the FeMoco-54 walk and controlled walk are compiled once, in Task 4.5.

**Git (spec 15)**
- Local branches and worktrees only. Merges go into local `main`. Never push, never modify remotes. The repo-local git identity (`dwierichs` / `david.wierichs@xanadu.ai`) is shared by all worktrees.
- Each task runs in a git worktree on a branch named `phase<N>/<task-id>` from local `main`; only the orchestrator merges.

---

## File structure

| Path | Task | Responsibility |
| --- | --- | --- |
| `tests/test_primitive_capabilities.py` | 0.6 | K-7: comparators and `SemiAdder`, forward and adjoint, exact on all inputs |
| `spikes/prototypes.py` | 0.7 | exact unary iteration, `ProtoTableRead`, `ProtoTableErase`, cost helpers |
| `spikes/k3_mode_agnostic_erase.py` | 0.7 | K-3 verdict |
| `spikes/k10_measurement_arrays.py` | 0.8 | K-10 verdict |
| `spikes/k1_whole_program.py` | 0.9 | K-1 stand-in walk, driver and FeMoco projection |
| `spikes/results/{k3,k10,k1_small,k1_mid10,k1_mid14,k1_projection}.json` | 0.7 to 0.9 | recorded results |
| `spikes/results/k10_femoco_dryrun.json` | 0.8 | the quoted FeMoco-shape timings of the plan's dry-run |
| `spikes/results/k1_femoco.json` | 0.9 | only if the projection allows the FeMoco compile |

Run spike scripts as `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/<script>.py`, with `VENV` set as in Part 1.

---

### Task 0.6: Spike K-7: comparator and adder adjoints

**Task id:** `0.6` · **Branch:** `phase0/0.6` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 0.2

**Files:**
- Create: `tests/test_primitive_capabilities.py`

**Interfaces:**
- Consumes: `dfthc.compile.GATE_SET`, `FIXED_DECOMPS`.
- Produces: the K-7 verdict for the ledger. On failure, the fallback of spec 16 K-7 (our own comparator operator with an explicit adjoint rule) becomes a Phase 1C task.

**Acceptance tests** (6 tests): `test_left_quantum_comparator_and_adjoint[<=|<]`: on a superposition of all 16 inputs the result bit equals `[x <= y]` (or `<`), and comparator then adjoint restores the input superposition exactly, including the dirty work wire; `test_left_classical_comparator_and_adjoint[1|2|3]`: likewise for `x < value` on 3 bits; `test_semiadder_and_adjoint`: `y -> x + y mod 8` on all 64 inputs, and adder then adjoint is the identity.

**Definition of done:** `6 passed`; K-7 recorded as resolved in the report; one commit.

- [ ] **Step 1: Write the capability tests**

Create `tests/test_primitive_capabilities.py`:

```python
"""K-7: adjoints of PennyLane's comparators and of SemiAdder compile and are exact (Task 0.6).

Each test prepares a uniform superposition over every input, applies the primitive, checks the
result on every branch, then applies the adjoint and checks that the state, including the work
wires the forward pass leaves dirty, returns exactly to the input superposition.
"""

import itertools

import numpy as np
import pennylane as qp
import pytest

from dfthc.compile import FIXED_DECOMPS, GATE_SET

pytestmark = pytest.mark.compiled

N = 2  # bits per operand


def _program(num_wires, body, returns):
    @qp.qjit(capture=True, seed=7)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS) or None)
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return returns()

    return circuit


def _uniform_state(num_wires, active):
    """Uniform superposition over the `active` wires, every other wire in |0>."""
    state = np.zeros(2**num_wires, dtype=complex)
    for bits in itertools.product([0, 1], repeat=len(active)):
        index = sum(bit << (num_wires - 1 - wire) for bit, wire in zip(bits, active, strict=True))
        state[index] = 1 / np.sqrt(2 ** len(active))
    return state


@pytest.mark.parametrize("comparator", ["<=", "<"])
def test_left_quantum_comparator_and_adjoint(comparator):
    x, y, target, work = [0, 1], [2, 3], [4], [5]

    def prepare():
        for wire in x + y:
            qp.H(wire)

    def compare():
        return qp.LeftQuantumComparator(x, y, target, work, comparator=comparator)

    forward = _program(6, lambda: (prepare(), compare()), lambda: qp.probs(wires=x + y + target))
    probs = np.asarray(forward()).reshape(4, 4, 2)
    for a, b in itertools.product(range(4), repeat=2):
        expected = int(a <= b) if comparator == "<=" else int(a < b)
        assert np.isclose(probs[a, b, expected], 1 / 16), (a, b)

    round_trip = _program(6, lambda: (prepare(), compare(), qp.adjoint(compare())), qp.state)
    assert np.allclose(np.asarray(round_trip()), _uniform_state(6, x + y), atol=1e-10)


@pytest.mark.parametrize("value", [1, 2, 3])
def test_left_classical_comparator_and_adjoint(value):
    x, target, work = [0, 1, 2], [3], [4, 5]

    def prepare():
        for wire in x:
            qp.H(wire)

    def compare():
        return qp.LeftClassicalComparator(x, value=value, target_wire=target, work_wires=work,
                                          comparator="<")

    forward = _program(6, lambda: (prepare(), compare()), lambda: qp.probs(wires=x + target))
    probs = np.asarray(forward()).reshape(8, 2)
    for a in range(8):
        assert np.isclose(probs[a, int(a < value)], 1 / 8), a

    round_trip = _program(6, lambda: (prepare(), compare(), qp.adjoint(compare())), qp.state)
    assert np.allclose(np.asarray(round_trip()), _uniform_state(6, x), atol=1e-10)


def test_semiadder_and_adjoint():
    x, y, work = [0, 1, 2], [3, 4, 5], [6, 7]

    def prepare():
        for wire in x + y:
            qp.H(wire)

    def add():
        return qp.SemiAdder(x, y, work)

    forward = _program(8, lambda: (prepare(), add()), lambda: qp.probs(wires=x + y))
    probs = np.asarray(forward()).reshape(8, 8)
    for a, b in itertools.product(range(8), repeat=2):
        assert np.isclose(probs[a, (a + b) % 8], 1 / 64), (a, b)

    round_trip = _program(8, lambda: (prepare(), add(), qp.adjoint(add())), qp.state)
    assert np.allclose(np.asarray(round_trip()), _uniform_state(8, x + y), atol=1e-10)
```

- [ ] **Step 2: Run them**

Run: `scripts/test.sh tests/test_primitive_capabilities.py -q`
Expected: `6 passed`. These pin PennyLane behaviour, so there is no implementation step. If one fails, K-7 is unresolved: report DONE_WITH_CONCERNS with the failing case; the orchestrator adds the fallback task.

- [ ] **Step 3: Commit**

```bash
git add tests/test_primitive_capabilities.py
git commit -m "Add K-7 capability tests for comparator and adder adjoints"
```

---

### Task 0.7: Spike K-3: a mode-agnostic measured-erase rule

**Task id:** `0.7` · **Branch:** `phase0/0.7` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 0.2

**Files:**
- Create: `spikes/prototypes.py`, `spikes/k3_mode_agnostic_erase.py`, `spikes/results/k3.json`

**Interfaces:**
- Consumes: `dfthc.compile.register_rule`, `resource_key`, `mcm_parities`, `compiled_resources`, `GATE_SET`, `FIXED_DECOMPS`.
- Produces (`spikes/prototypes.py`, used by Tasks 0.8 and 0.9):
  - `to_words(value: int, width: int) -> tuple[int, ...]`, `from_words(words) -> int` (32-bit words, least significant first).
  - `unary_iterate(emit, address, lo, hi, work, leaf, control=None)` and `nested_iterate(emit, registers, work, leaf, control=None)` with emitters `_Emit` and `_Count`; `unary_toffolis(lo, hi, n, controlled) -> int` (spec 9.24 U).
  - `ProtoTableRead(table, high_ranges, block_bits, width, high, low, target, junk, work)` and `ProtoTableErase(...)` (same arguments); `table` is a tuple of `(address, words)` pairs; `high_ranges` a tuple of `(width, lo, hi)`.
  - `loaded_masks(table, high_ranges, block_bits, width) -> list[tuple[int, int]]`, `swap_permutation(block_bits, low) -> list[int]`, `read_toffolis(...)`, `erase_toffolis(...)`, `counts_to_resources(counts) -> dict`, `_measure_x(qubit)`.
- Produces the K-3 verdict: `spikes/results/k3.json` with `"verdict": "pass"`.

**Pass criteria (spec 16, K-3):** on a read/erase pair with a dead block position and unvisited addresses:
1. `qp.testing.assert_valid_decomp_rule` passes in tape mode for the read and the erase rules;
2. under capture, the round trip compiles with exact non-symbolic counts equal to the closed forms (T = 9, M = 4), and `lightning.qubit` restores the address superposition exactly for five seeds;
3. on the tape path, `_tape_decompose` with the pinned rules, simulated by `default.qubit` with deferred measurements, restores the address state and leaves all ancillas in |0>.

**Definition of done:** the script exits 0 and prints `"verdict": "pass"`; `spikes/results/k3.json` committed; `ruff check spikes` clean. Fallback if it fails: mode-specific branches inside `mcm_parities`, keeping one operator and one rule (spec 16, K-3); the orchestrator records it and adjusts Task 1C.5.

- [ ] **Step 1: Write the spike harness**

Create `spikes/k3_mode_agnostic_erase.py`:

```python
"""Spike K-3: a measured erase whose fixup uses measurement-derived parities runs in both modes.

Pass criteria (spec section 16, K-3), all on a read/erase pair with a dead block position and
unvisited addresses:
1. tape mode: qp.testing.assert_valid_decomp_rule passes for the read and the erase rules;
2. capture: read + erase compiles with exact, non-symbolic counts equal to the closed forms,
   and on lightning.qubit the address superposition is restored exactly for five seeds;
3. tape path: _tape_decompose with the pinned rules, simulated by default.qubit with deferred
   measurements, restores the address state exactly and leaves every ancilla in |0>.
Writes spikes/results/k3.json and exits with status 1 if any criterion fails.

Run: CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 .venv/bin/python spikes/k3_mode_agnostic_erase.py
"""

import json
import sys
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
    for name, check in (("tape_rules", check_tape_rules), ("capture", check_capture),
                        ("tape_path", check_tape_path)):
        try:
            results[name] = check()
        except Exception as exc:  # noqa: BLE001
            results[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    passed = all(r is True or r.get("passed") for r in results.values())
    results["verdict"] = "pass" if passed else "fail"
    out = Path(__file__).parent / "results" / "k3.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps(results, indent=2, default=str))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/k3_mode_agnostic_erase.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'prototypes'`.

- [ ] **Step 3: Write the prototypes**

Create `spikes/prototypes.py`:

```python
"""Prototype operators shared by the Phase 0 spikes K-1, K-3 and K-10.

Spikes may use placeholder semantics (spec 16, K-1), but the table read and its measured erase
here are exact, because K-3 checks them on lightning.qubit. They are not the production
operators: Phase 1 (workstream C) writes those from the spec with full tests.

Table entries are tuples of unsigned 32-bit words, least significant word first. Catalyst lowers
compilable integers as signed 64-bit values, and larger values fail to compile (canary L-24).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import mcm_parities, register_rule, resource_key

WORD = 32


def to_words(value: int, width: int) -> tuple[int, ...]:
    """Split a ``width``-bit integer into 32-bit words, least significant first."""
    return tuple((value >> (WORD * w)) & (2**WORD - 1) for w in range(-(-width // WORD)))


def from_words(words: Sequence[int]) -> int:
    return sum(int(word) << (WORD * w) for w, word in enumerate(words))


# ---------------------------------------------------------------------------------------------
# Exact unary iteration (spec 9.3, cost formula of spec 9.24), with a counting twin.
# ---------------------------------------------------------------------------------------------


class _Emit:
    """Emits gates. Negative controls are X-conjugated so every TemporaryAND key is the same."""

    def x(self, wire):
        qp.X(wire)

    def and_(self, a, b, target):
        qp.TemporaryAND([a, b, target])

    def unand(self, a, b, target):
        qp.adjoint(qp.TemporaryAND([a, b, target]))

    def cnot(self, control, target):
        qp.CNOT([control, target])


class _Count(_Emit):
    """Counts the gates :class:`_Emit` would emit, by resource-key name."""

    def __init__(self):
        self.counts = Counter()

    def x(self, wire):
        self.counts["PauliX"] += 1

    def and_(self, a, b, target):
        self.counts["TemporaryAND"] += 1

    def unand(self, a, b, target):
        self.counts["Adjoint(TemporaryAND)"] += 1

    def cnot(self, control, target):
        self.counts["CNOT"] += 1


def unary_iterate(emit, address, lo, hi, work, leaf, control=None):
    """Call ``leaf(flag, value)`` for every ``value`` in ``[lo, hi)`` (exact iteration).

    ``address`` is big-endian. ``flag`` is a wire that is 1 exactly when the address equals
    ``value`` and the optional ``control`` wire is 1; it is ``None`` only for a zero-width,
    uncontrolled address. ``work`` needs ``len(address)`` wires (one per level below the root).
    """
    if not 0 <= lo < hi <= 2 ** len(address):
        raise ValueError(f"range [{lo}, {hi}) outside {len(address)}-bit addresses")
    _node(emit, address, lo, hi, list(work), leaf, control, 0, len(address))


def _node(emit, address, lo, hi, work, leaf, flag, base, height):
    if height == 0:
        leaf(flag, base)
        return
    bit = address[len(address) - height]
    half = 1 << (height - 1)
    left = base < hi and lo < base + half
    right = base + half < hi and lo < base + 2 * half
    if flag is None:  # uncontrolled root: the address bit itself is the child flag
        if left:
            emit.x(bit)
            _node(emit, address, lo, hi, work, leaf, bit, base, height - 1)
            emit.x(bit)
        if right:
            _node(emit, address, lo, hi, work, leaf, bit, base + half, height - 1)
        return
    child, rest = work[0], work[1:]
    if left and right:
        emit.x(bit)
        emit.and_(flag, bit, child)
        emit.x(bit)
        _node(emit, address, lo, hi, rest, leaf, child, base, height - 1)
        emit.cnot(flag, child)
        _node(emit, address, lo, hi, rest, leaf, child, base + half, height - 1)
        emit.unand(flag, bit, child)
    elif left:
        emit.x(bit)
        emit.and_(flag, bit, child)
        emit.x(bit)
        _node(emit, address, lo, hi, rest, leaf, child, base, height - 1)
        emit.x(bit)
        emit.unand(flag, bit, child)
        emit.x(bit)
    else:
        emit.and_(flag, bit, child)
        _node(emit, address, lo, hi, rest, leaf, child, base + half, height - 1)
        emit.unand(flag, bit, child)


def unary_toffolis(lo: int, hi: int, n: int, controlled: bool) -> int:
    """U(lo, hi, n, c) of spec 9.24."""
    if n == 0:
        return 0
    nodes = sum((hi - 1) // 2**m - lo // 2**m + 1 for m in range(1, n + 1))
    return nodes - (0 if controlled else 1)


def counts_to_resources(counts: Counter) -> dict:
    """Map counted gate names to register_resources keys."""
    keys = {
        "PauliX": qp.X, "PauliZ": qp.Z, "Hadamard": qp.H, "CNOT": qp.CNOT, "CSWAP": qp.CSWAP,
        "CZ": qp.CZ,
        "TemporaryAND": resource_key(lambda: qp.TemporaryAND(Wire[3])),
        "Adjoint(TemporaryAND)": resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))),
        "MidMeasure": resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True)),
    }
    return {keys[name]: count for name, count in counts.items() if count}


# ---------------------------------------------------------------------------------------------
# QROAM read and measured erase (spec 9.3 and 9.5, plain fixup of spec 9.4 with onehot_bits=0).
# ---------------------------------------------------------------------------------------------


def nested_iterate(emit, registers, work, leaf, control=None):
    """Nested exact iteration over ``registers = ((wires, lo, hi), ...)``, outermost first.

    ``leaf(flag, values)`` receives the tuple of register values.
    """
    def recurse(index, flag, values, pool):
        wires, lo, hi = registers[index]
        if index == len(registers) - 1:
            unary_iterate(emit, wires, lo, hi, pool, lambda f, v: leaf(f, values + (v,)), flag)
        else:
            unary_iterate(emit, wires, lo, hi, pool[: len(wires)],
                          lambda f, v: recurse(index + 1, f, values + (v,), pool[len(wires):]),
                          flag)

    recurse(0, control, (), list(work))


def swap_permutation(block_bits: int, low: int) -> list[int]:
    """Entry index held by register r after the swap-with-zero network, for a low address."""
    held = list(range(2**block_bits))
    for b in reversed(range(block_bits)):
        if (low >> b) & 1:
            for r in range(2**b):
                held[r], held[r + 2**b] = held[r + 2**b], held[r]
    return held


class ProtoTableRead(Operator2):
    """QROAM read: ``target ‖ junk`` receive the requested entry and the rest of its block."""

    compilable_argnames = ("table", "high_ranges", "block_bits", "width")
    wire_argnames = ("high", "low", "target", "junk", "work")

    def __init__(self, table, high_ranges, block_bits, width, high, low, target, junk, work):
        super().__init__(table, high_ranges, block_bits, width, high, low, target, junk, work)


def _split_high(high, high_ranges):
    registers, start = [], 0
    for width, lo, hi in high_ranges:
        registers.append((high[start:start + width], lo, hi))
        start += width
    return registers


def _read_body(emit, table, high_ranges, block_bits, width, high, low, target, junk, work,
               cswap: Callable):
    entries = dict(table)
    registers = [target] + [junk[r * width:(r + 1) * width] for r in range(2**block_bits - 1)]
    widths = [w for w, _, _ in high_ranges]

    def leaf(flag, values):
        prefix = 0
        for value, w in zip(values, widths, strict=True):
            prefix = (prefix << w) | value
        for position in range(2**block_bits):
            entry = from_words(entries.get((prefix << block_bits) | position, (0,)))
            for i in range(width):
                if (entry >> i) & 1:
                    emit.cnot(flag, registers[position][i])

    nested_iterate(emit, _split_high(high, high_ranges), work, leaf)
    for b in reversed(range(block_bits)):
        for r in range(2**b):
            for i in range(width):
                cswap(low[block_bits - 1 - b], registers[r][i], registers[r + 2**b][i])


def _read_counts(table, high_ranges, block_bits, width):
    counter = _Count()
    dummy = list(range(10_000))
    high = dummy[: sum(w for w, _, _ in high_ranges)]
    _read_body(counter, table, high_ranges, block_bits, width, high, dummy[:block_bits],
               dummy[:width], dummy[: (2**block_bits - 1) * width], dummy[:64],
               lambda c, a, b: counter.counts.update({"CSWAP": 1}))
    return counter.counts


def _read_resources(table, high_ranges, block_bits, width, high, low, target, junk, work):
    return counts_to_resources(_read_counts(table, high_ranges, block_bits, width))


@qp.register_resources(_read_resources)
def _proto_table_read(table, high_ranges, block_bits, width, high, low, target, junk, work):
    _read_body(_Emit(), table, high_ranges, block_bits, width, high, low, target, junk, work,
               lambda c, a, b: qp.CSWAP([c, a, b]))


register_rule(ProtoTableRead, _proto_table_read)


def loaded_masks(table, high_ranges, block_bits, width):
    """(address, mask) for every loaded address; bit (r * width + i) is bit i of register r."""
    entries = {a: from_words(e) for a, e in table}
    ranges = [range(lo, hi) for _, lo, hi in high_ranges]
    widths = [w for w, _, _ in high_ranges]
    result = []

    def walk(index, prefix):
        if index == len(ranges):
            for low in range(2**block_bits):
                held = swap_permutation(block_bits, low)
                mask = 0
                for r, entry_index in enumerate(held):
                    mask |= entries.get((prefix << block_bits) | entry_index, 0) << (r * width)
                result.append(((prefix << block_bits) | low, mask))
            return
        for value in ranges[index]:
            walk(index + 1, (prefix << widths[index]) | value)

    walk(0, 0)
    return result


class ProtoTableErase(Operator2):
    """Measured erase of :class:`ProtoTableRead`: X-measure, then a plain phase fixup."""

    compilable_argnames = ("table", "high_ranges", "block_bits", "width")
    wire_argnames = ("high", "low", "target", "junk", "work")

    def __init__(self, table, high_ranges, block_bits, width, high, low, target, junk, work):
        super().__init__(table, high_ranges, block_bits, width, high, low, target, junk, work)


def _erase_body(emit, table, high_ranges, block_bits, width, high, low, target, junk, work,
                measure, conditional_z):
    qubits = list(target) + list(junk)
    outcomes = [measure(q) for q in qubits]
    masks = loaded_masks(table, high_ranges, block_bits, width)
    parities = dict(zip((a for a, _ in masks),
                        mcm_parities(outcomes, [m for _, m in masks]) if outcomes else [],
                        strict=True))
    registers = _split_high(high, high_ranges) + [(low, 0, 2**block_bits)]
    widths = [w for w, _, _ in high_ranges] + [block_bits]

    def leaf(flag, values):
        address = 0
        for value, w in zip(values, widths, strict=True):
            address = (address << w) | value
        if parities[address] is not False:
            conditional_z(parities[address], flag)

    nested_iterate(emit, registers, work, leaf)


def _erase_counts(table, high_ranges, block_bits, width):
    counter = _Count()
    num_qubits = width * 2**block_bits
    counter.counts["Hadamard"] += num_qubits
    counter.counts["MidMeasure"] += num_qubits
    nonzero = sum(1 for _, m in loaded_masks(table, high_ranges, block_bits, width) if m)
    counter.counts["PauliZ"] += nonzero
    dummy = list(range(10_000))
    registers = [(dummy[:w], lo, hi) for w, lo, hi in high_ranges] + [
        (dummy[:block_bits], 0, 2**block_bits)
    ]
    nested_iterate(counter, registers, dummy[:64], lambda f, v: None)
    return counter.counts


def _erase_resources(table, high_ranges, block_bits, width, high, low, target, junk, work):
    return counts_to_resources(_erase_counts(table, high_ranges, block_bits, width))


def _measure_x(qubit):
    qp.H(qubit)
    return qp.measure(qubit, reset=True)


@qp.register_resources(_erase_resources)
def _proto_table_erase(table, high_ranges, block_bits, width, high, low, target, junk, work):
    _erase_body(_Emit(), table, high_ranges, block_bits, width, high, low, target, junk, work,
                _measure_x, lambda parity, flag: qp.cond(parity, qp.Z)(flag))


register_rule(ProtoTableErase, _proto_table_erase)


def read_toffolis(high_ranges, block_bits, width) -> int:
    """T of ProtoTableRead: nested unary iteration plus the swap network."""
    return _nested_toffolis(high_ranges) + (2**block_bits - 1) * width


def erase_toffolis(high_ranges, block_bits) -> int:
    """T of ProtoTableErase: plain fixup over every loaded address."""
    return _nested_toffolis(list(high_ranges) + [(block_bits, 0, 2**block_bits)])


def _nested_toffolis(ranges) -> int:
    total, leaves = 0, 1
    for index, (w, lo, hi) in enumerate(ranges):
        total += leaves * unary_toffolis(lo, hi, w, controlled=index > 0)
        leaves *= hi - lo
    return total
```

- [ ] **Step 4: Run the spike**

Run: `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/k3_mode_agnostic_erase.py`
Expected (about 15 s), the JSON written to `spikes/results/k3.json`:

```json
{
  "tape_rules": true,
  "capture": {"toffoli": 9, "expected_toffoli": 9, "measurements": 4, "counts_ok": true,
              "states_ok": [true, true, true, true, true], "passed": true},
  "tape_path": {"operation_names": ["Adjoint(TemporaryAND)", "CNOT", "Conditional(PauliZ)",
                                    "Hadamard", "MidMeasure", "PauliX", "TemporaryAND", "Toffoli"],
                "address_restored": true, "ancillas_zero": true, "passed": true},
  "verdict": "pass"
}
```

- [ ] **Step 5: Lint and commit**

Run: `"$VENV/bin/ruff" check spikes` · Expected: `All checks passed!`

```bash
git add spikes/prototypes.py spikes/k3_mode_agnostic_erase.py spikes/results/k3.json
git commit -m "Add K-3 spike: measured erase in tape and capture mode"
```

---

### Task 0.8: Spike K-10: measurement-valued arrays as dynamic arguments

**Task id:** `0.8` · **Branch:** `phase0/0.8` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 0.7

**Files:**
- Create: `spikes/k10_measurement_arrays.py`, `spikes/results/k10.json`, `spikes/results/k10_femoco_dryrun.json`

**Interfaces:**
- Consumes: `spikes/prototypes.py` (Task 0.7); `dfthc.compile`.
- Produces: the K-10 verdict, which decides Q-1 variant (b) and the `QPERound` form (spec 11.1). It also produces three prototype operators that later tasks copy from:
  - `ProtoParityFixup(parities, register_ranges, address, work)`, with dynamic `parities`;
  - `ProtoUnprepareFromOutcomes(junk_outcomes, masks, register_ranges, address, target, work)`, with dynamic `junk_outcomes` and `masks` as tuples of 32-bit words;
  - `ProtoRound(j, num_bits, earlier_bits, control)`, with dynamic `earlier_bits` of shape (t,).

  `as_dynamic_array(values)` canonicalizes lists (L-23).

**Pass criteria (spec 16, K-10):** in tape mode and under capture:
1. a parity array derived from measurements;
2. a junk-outcome array passed to an operator shaped like `InnerUnprepareFromOutcomes`;
3. `earlier_bits` of shape (t,), covering both round t-1 (no entry set) and rounds with set entries.

Each part simulates exactly at a small size in both modes. Parts 1 and 2 also compile at two mid sizes with the FeMoco-54 inner-erase widths (416 outcomes, 390 of them junk), over 544 and 1,088 addresses, with non-symbolic counts and 416 measurements, and the growth exponent of each part's compile time between the two sizes is recorded. The FeMoco-54 shape (8,640 addresses) is not compiled: the large-compile policy (master plan section 12.9) forbids repeating it, and its dry-run timings are quoted in `k10_femoco_dryrun.json`. The mid sizes cannot reproduce part 2's FeMoco-size cost: in the round-3 calibration both parts grew about linearly up to 1,088 addresses.

**Definition of done:** the script exits 0 with `verdict: pass`; `spikes/results/k10.json` and `spikes/results/k10_femoco_dryrun.json` committed; the mid-size compile times and growth exponents recorded in the report. If part 3 fails: corrections stay in the QPE driver (spec 11.1 fallback). If part 2 fails: Q-1 keeps only variant (a). The orchestrator records either in the ledger's decision log.

- [ ] **Step 1: Write the spike**

Create `spikes/k10_measurement_arrays.py`:

```python
"""Spike K-10: measurement-valued arrays as dynamic Operator2 arguments, in both modes.

Three arrays cross an operator boundary (spec section 16, K-10):
1. parities: the caller X-measures a read's target and junk, computes the fixup parities with
   mcm_parities and passes them as one dynamic array to ProtoParityFixup;
2. junk outcomes: the caller X-measures only the junk and passes the outcome array to
   ProtoUnprepareFromOutcomes (shaped like InnerUnprepareFromOutcomes, Q-1 variant (b)), which
   measures the target and computes the parities from the concatenated outcomes;
3. earlier_bits: a QPERound-like operator receives a boolean array of shape (t,), with no entry
   set in round j = t-1 and set entries in later rounds.
Each part is simulated at a small size in both modes (lightning.qubit under capture; the tape
path with deferred measurements on default.qubit) and must be exact. Parts 1 and 2 also compile
at two mid sizes with the FeMoco-54 inner-erase widths (416 outcomes, 390 of them junk) over 544
and 1088 addresses, with non-symbolic counts; the growth exponent of each part's compile time
between the two sizes is recorded. Writes spikes/results/k10.json; exit status 1 on failure.

The FeMoco-54 shape (8640 addresses) is not compiled (master plan section 12.9): the timings of the
plan's dry-run are quoted in spikes/results/k10_femoco_dryrun.json.

Run: CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 .venv/bin/python spikes/k10_measurement_arrays.py
(add --small-only to skip the four mid-size compiles, about 80 s together)
"""

import json
import math
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.transforms import _tape_decompose
from pennylane.typing import AbstractArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prototypes import (  # noqa: E402
    ProtoTableRead,
    _Count,
    _Emit,
    _measure_x,
    counts_to_resources,
    from_words,
    loaded_masks,
    nested_iterate,
    to_words,
)

from dfthc.compile import (  # noqa: E402
    FIXED_DECOMPS,
    GATE_SET,
    compiled_resources,
    mcm_parities,
    register_rule,  # noqa: E402
)


def as_dynamic_array(values):
    """Canonicalize a list of measurement values for a dynamic argument (L-23)."""
    if isinstance(values, AbstractArray):
        return values
    if qp.capture.enabled():
        return jnp.stack([jnp.asarray(v, dtype=bool) for v in values])
    return np.asarray(values, dtype=object)


def _fixup_leaves(emit, registers, work, parity_of):
    """Plain fixup: a conditional Z on the flag of every address, in iteration order."""
    widths = [len(w) for w, _, _ in registers]
    index = {"next": 0}

    def leaf(flag, values):
        parity = parity_of(index["next"])
        index["next"] += 1
        if parity is not False:
            emit.cond_z(parity, flag)

    nested_iterate(emit, registers, work, leaf)
    del widths


class _EmitCond(_Emit):
    def cond_z(self, parity, flag):
        qp.cond(parity, qp.Z)(flag)


class _CountCond(_Count):
    def cond_z(self, parity, flag):
        self.counts["PauliZ"] += 1


def _fixup_counts(register_ranges, num_measured):
    counter = _CountCond()
    dummy = list(range(10_000))
    registers = [(dummy[:w], lo, hi) for w, lo, hi in register_ranges]
    _fixup_leaves(counter, registers, dummy[:64], lambda a: True)
    counter.counts["Hadamard"] += num_measured
    counter.counts["MidMeasure"] += num_measured
    return counter.counts


class ProtoParityFixup(Operator2):
    """Part 1: applies (-1)^parities[a] on the a-th address of ``register_ranges``."""

    dynamic_argnames = ("parities",)
    compilable_argnames = ("register_ranges",)
    wire_argnames = ("address", "work")

    def __init__(self, parities, register_ranges, address, work):
        super().__init__(as_dynamic_array(parities), register_ranges, address, work)


def _split(address, register_ranges):
    registers, start = [], 0
    for width, lo, hi in register_ranges:
        registers.append((address[start:start + width], lo, hi))
        start += width
    return registers


def _parity_fixup_resources(parities, register_ranges, address, work):
    return counts_to_resources(_fixup_counts(register_ranges, 0))


@qp.register_resources(_parity_fixup_resources)
def _proto_parity_fixup(parities, register_ranges, address, work):
    _fixup_leaves(_EmitCond(), _split(address, register_ranges), work, lambda a: parities[a])


register_rule(ProtoParityFixup, _proto_parity_fixup)


class ProtoUnprepareFromOutcomes(Operator2):
    """Part 2: X-measures ``target``, then fixes phases from (target ‖ junk_outcomes).

    ``masks`` holds one tuple of 32-bit words per address (compilable integers must fit in
    int64, L-24).
    """

    dynamic_argnames = ("junk_outcomes",)
    compilable_argnames = ("masks", "register_ranges")
    wire_argnames = ("address", "target", "work")

    def __init__(self, junk_outcomes, masks, register_ranges, address, target, work):
        super().__init__(as_dynamic_array(junk_outcomes), masks, register_ranges, address,
                         target, work)


def _unprepare_resources(junk_outcomes, masks, register_ranges, address, target, work):
    counts = _fixup_counts(register_ranges, len(target))
    counts["PauliZ"] = sum(1 for m in masks if from_words(m))
    return counts_to_resources(counts)


@qp.register_resources(_unprepare_resources)
def _proto_unprepare(junk_outcomes, masks, register_ranges, address, target, work):
    target_outcomes = [_measure_x(q) for q in target]
    junk = [junk_outcomes[i] for i in range(junk_outcomes.shape[0])]
    parities = mcm_parities(target_outcomes + junk, [from_words(m) for m in masks])
    _fixup_leaves(_EmitCond(), _split(address, register_ranges), work, lambda a: parities[a])


register_rule(ProtoUnprepareFromOutcomes, _proto_unprepare)


class ProtoRound(Operator2):
    """Part 3: X on ``control`` for round t-1, then X conditioned on every earlier bit l > j."""

    dynamic_argnames = ("earlier_bits",)
    compilable_argnames = ("j", "num_bits")
    wire_argnames = ("control",)

    def __init__(self, j, num_bits, earlier_bits, control):
        super().__init__(j, num_bits, as_dynamic_array(earlier_bits), control)


def _round_resources(j, num_bits, earlier_bits, control):
    return {qp.X: int(j == num_bits - 1) + (num_bits - 1 - j)}


@qp.register_resources(_round_resources)
def _proto_round(j, num_bits, earlier_bits, control):
    if j == num_bits - 1:
        qp.X(control[0])
    for later in range(j + 1, num_bits):
        qp.cond(earlier_bits[later], qp.X)(control[0])


register_rule(ProtoRound, _proto_round)

# --------------------------------------------------------------------------- small instances

WIDTH, BLOCK_BITS = 2, 1
HIGH_RANGES = ((2, 0, 3),)
TABLE = tuple((a, to_words(v, WIDTH)) for a, v in enumerate([1, 2, 3, 1, 2]))
HIGH, LOW, TARGET, JUNK, WORK = [0, 1], [2], [3, 4], [5, 6], [7, 8, 9, 10]
MASK_INTS = tuple(m for _, m in loaded_masks(TABLE, HIGH_RANGES, BLOCK_BITS, WIDTH))
MASKS = tuple(to_words(m, WIDTH * 2**BLOCK_BITS) for m in MASK_INTS)
FIXUP_RANGES = ((2, 0, 3), (1, 0, 2))  # high part, then the low (block) bit


def part1_small():
    for wire in HIGH + LOW:
        qp.H(wire)
    ProtoTableRead(TABLE, HIGH_RANGES, BLOCK_BITS, WIDTH, HIGH, LOW, TARGET, JUNK, WORK)
    outcomes = [_measure_x(q) for q in TARGET + JUNK]
    ProtoParityFixup(mcm_parities(outcomes, MASK_INTS), FIXUP_RANGES, HIGH + LOW, WORK)


def part2_small():
    for wire in HIGH + LOW:
        qp.H(wire)
    ProtoTableRead(TABLE, HIGH_RANGES, BLOCK_BITS, WIDTH, HIGH, LOW, TARGET, JUNK, WORK)
    junk_outcomes = [_measure_x(q) for q in JUNK]
    ProtoUnprepareFromOutcomes(junk_outcomes, MASKS, FIXUP_RANGES, HIGH + LOW, TARGET, WORK)


NUM_BITS, CONTROL, OUT = 4, [0], [1, 2, 3, 4]
EXPECTED_BITS = (0, 0, 1, 1)  # m_3 = 1, m_2 = m_3, m_1 = m_3 ^ m_2, m_0 = m_3 ^ m_2 ^ m_1


def part3_small():
    if qp.capture.enabled():
        bits = jnp.zeros(NUM_BITS, dtype=bool)
    else:
        bits = np.full(NUM_BITS, False, dtype=object)
    for j in reversed(range(NUM_BITS)):
        ProtoRound(j, NUM_BITS, bits, CONTROL)
        outcome = qp.measure(CONTROL[0], reset=True)
        if qp.capture.enabled():
            bits = bits.at[j].set(jnp.asarray(outcome, dtype=bool))
        else:
            bits = bits.copy()
            bits[j] = outcome
        qp.cond(outcome, qp.X)(OUT[j])


def restored_address_checks(qfunc):
    """Exact restoration of |+++> on the address in both modes, ancillas back in |0>."""
    plus = np.full(8, 1 / np.sqrt(8))
    expected = np.zeros(2**11, dtype=complex)
    expected[[a << 8 for a in range(8)]] = 1 / np.sqrt(8)
    capture_ok = []
    for seed in range(3):
        @qp.qjit(capture=True, seed=seed)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
        @qp.qnode(qp.device("lightning.qubit", wires=11))
        def circuit():
            qfunc()
            return qp.state()

        capture_ok.append(bool(np.allclose(np.asarray(circuit()), expected, atol=1e-10)))
    qp.decomposition.enable_graph()
    with qp.queuing.AnnotatedQueue() as queue:
        qfunc()
        qp.density_matrix(wires=HIGH + LOW)
        qp.probs(wires=TARGET + JUNK + WORK)
    [tape], _ = _tape_decompose(qp.tape.QuantumScript.from_queue(queue), gate_set=GATE_SET,
                                fixed_decomps=dict(FIXED_DECOMPS))
    [deferred], _ = qp.defer_measurements(tape)
    rho, probs = qp.devices.DefaultQubit().execute(deferred)
    tape_ok = bool(np.allclose(rho, np.outer(plus, plus), atol=1e-10) and np.isclose(probs[0], 1))
    return {"capture_seeds_ok": capture_ok, "tape_ok": tape_ok,
            "passed": all(capture_ok) and tape_ok}


def round_bits_checks():
    expected = np.zeros(16)
    expected[int("".join(map(str, EXPECTED_BITS)), 2)] = 1.0

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=5))
    def circuit():
        part3_small()
        return qp.probs(wires=OUT)

    capture_ok = bool(np.allclose(np.asarray(circuit()), expected))
    qp.decomposition.enable_graph()
    with qp.queuing.AnnotatedQueue() as queue:
        part3_small()
        qp.probs(wires=OUT)
    [tape], _ = _tape_decompose(qp.tape.QuantumScript.from_queue(queue), gate_set=GATE_SET,
                                fixed_decomps=dict(FIXED_DECOMPS))
    [deferred], _ = qp.defer_measurements(tape)
    tape_ok = bool(np.allclose(qp.devices.DefaultQubit().execute(deferred), expected))
    return {"capture_ok": capture_ok, "tape_ok": tape_ok, "passed": capture_ok and tape_ok}


# ------------------------------------------------- mid sizes with the FeMoco-54 widths

TARGET_BITS, JUNK_BITS = 26, 390                   # d_2 and (K_2 - 1) d_2 at FeMoco-54
SHAPES = {                                         # (rows, row bits, b positions, b bits)
    "mid544": (17, 5, 32, 5),
    "mid1088": (34, 6, 32, 5),                     # an eighth of FeMoco-54's 270 x 32
}


def shape_compile(part, shape):
    rows, row_bits, b_positions, b_bits = SHAPES[shape]
    ranges = ((row_bits, 0, rows), (b_bits, 0, b_positions))
    num_addresses = rows * b_positions
    rng = np.random.default_rng(10)
    num_outcomes = TARGET_BITS + JUNK_BITS
    masks = tuple(int.from_bytes(rng.bytes(52), "little") for _ in range(num_addresses))
    address = list(range(row_bits + b_bits))
    target = list(range(14, 14 + TARGET_BITS))
    junk = list(range(40, 40 + JUNK_BITS))
    work = list(range(430, 445))

    def qfunc():
        if part == 1:
            outcomes = [_measure_x(q) for q in target + junk]
            ProtoParityFixup(mcm_parities(outcomes, masks), ranges, address, work)
        else:
            junk_outcomes = [_measure_x(q) for q in junk]
            mask_words = tuple(to_words(m, num_outcomes) for m in masks)
            ProtoUnprepareFromOutcomes(junk_outcomes, mask_words, ranges, address, target, work)

    start = time.perf_counter()
    resources = compiled_resources(qfunc, num_wires=445)
    seconds = round(time.perf_counter() - start, 1)
    return {"seconds": seconds, "measurements": resources.measurements,
            "conditional_z_upper_bound": resources.counts.get("PauliZ", 0),
            "num_addresses": num_addresses, "passed": resources.measurements == num_outcomes}


def growth_exponents(results):
    """Per part, d log(seconds) / d log(addresses) between the two mid sizes."""
    exponents = {}
    for part in (1, 2):
        a, b = (results.get(f"part{part}_{shape}_compile", {}) for shape in SHAPES)
        if a.get("passed") and b.get("passed"):
            exponents[f"part{part}"] = round(
                math.log(b["seconds"] / a["seconds"])
                / math.log(b["num_addresses"] / a["num_addresses"]), 2)
    return exponents


def main():
    checks = {
        "part1_small": lambda: restored_address_checks(part1_small),
        "part2_small": lambda: restored_address_checks(part2_small),
        "part3_small": round_bits_checks,
    }
    if "--small-only" not in sys.argv:
        for shape in SHAPES:
            for part in (1, 2):
                checks[f"part{part}_{shape}_compile"] = (
                    lambda part=part, shape=shape: shape_compile(part, shape))
    results = {}
    for name, check in checks.items():
        try:
            results[name] = check()
        except Exception as exc:  # noqa: BLE001
            results[name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
        print(name, json.dumps(results[name], default=str), flush=True)
    passed = all(r.get("passed") for r in results.values())
    if "--small-only" not in sys.argv:
        results["compile_time_growth_exponents"] = growth_exponents(results)
        print("compile_time_growth_exponents", json.dumps(results["compile_time_growth_exponents"]))
    results["verdict"] = "pass" if passed else "fail"
    out = Path(__file__).parent / "results" / "k10.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n")
    print("verdict:", results["verdict"])
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the small-size checks**

Run: `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/k10_measurement_arrays.py --small-only`
Expected (about 15 s):

```text
part1_small {"capture_seeds_ok": [true, true, true], "tape_ok": true, "passed": true}
part2_small {"capture_seeds_ok": [true, true, true], "tape_ok": true, "passed": true}
part3_small {"capture_ok": true, "tape_ok": true, "passed": true}
verdict: pass
```

- [ ] **Step 3: Run the full spike, with the mid-size compiles**

Run: `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/k10_measurement_arrays.py`
Expected (about 2 minutes, below 2 GB, so not heavy): the three small parts as above, then

```text
part1_mid544_compile {"seconds": ..., "measurements": 416, "conditional_z_upper_bound": 544, "num_addresses": 544, "passed": true}
part2_mid544_compile {"seconds": ..., "measurements": 416, "conditional_z_upper_bound": 544, "num_addresses": 544, "passed": true}
part1_mid1088_compile {"seconds": ..., "measurements": 416, "conditional_z_upper_bound": 1088, "num_addresses": 1088, "passed": true}
part2_mid1088_compile {"seconds": ..., "measurements": 416, "conditional_z_upper_bound": 1088, "num_addresses": 1088, "passed": true}
compile_time_growth_exponents {"part1": ..., "part2": ...}
verdict: pass
```

The counts are exact and must match; the times vary. In the round-3 calibration the four compiles took 13.2, 11.6, 30.9 and 23.2 s (peak 1.1 and 1.75 GB), growth exponents about 1.2 and 1.0. This round's dry-run ran the same code path at 64 and 128 addresses, where every count was exact; with some measurements dropped, part 1 counted 375 instead of 416 and the verdict was `fail`.

- [ ] **Step 4: Record the FeMoco-shape dry-run as a quote**

Create `spikes/results/k10_femoco_dryrun.json` with exactly this content. It quotes this plan's dry-run; it is not a measurement of this task:

```json
{
  "status": "quoted, not reproduced: a dry-run measurement taken while writing the plan; the large-compile policy (master plan section 12.9) forbids repeating it",
  "source": "docs/superpowers/plans/2026-11-11-dfthc-phase0-spikes.md at commit 994f7aa: the preamble and Task 0.8, Step 3",
  "what": "K-10 parts 1 and 2 compiled at the FeMoco-54 inner-erase shape: 8640 addresses (270 rows x 32 b positions), 416 outcomes, 390 of them junk",
  "conditions": "development laptop (10 cores, 16 GB), other compiles running at the same time; peak memory not recorded",
  "part1_packed_parities_seconds": "199 to 274",
  "part2_masks_as_compilable_data_seconds": 1517,
  "conclusion": "masks are derived inside rules from the table and never passed as compilable data"
}
```

- [ ] **Step 5: Lint and commit**

Run: `"$VENV/bin/ruff" check spikes` · Expected: `All checks passed!`

```bash
git add spikes/k10_measurement_arrays.py spikes/results/k10.json spikes/results/k10_femoco_dryrun.json
git commit -m "Add K-10 spike: measurement arrays as dynamic arguments"
```

---

### Task 0.9: Spike K-1: the whole-program stand-in walk at small and mid shapes, projected to FeMoco-54

**Task id:** `0.9` · **Branch:** `phase0/0.9` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 0.7

**Files:**
- Create: `spikes/k1_whole_program.py`, `spikes/results/k1_small.json`, `spikes/results/k1_mid10.json`, `spikes/results/k1_mid14.json`, `spikes/results/k1_projection.json`; `spikes/results/k1_femoco.json` only if the projection allows the FeMoco compile (Step 5)

**Interfaces:**
- Consumes: `spikes/prototypes.py`; `dfthc.compile.compiled_resources`, `register_rule`, `resource_key`, `mcm_parities`, `GATE_SET`, `FIXED_DECOMPS`; `dfthc.testing.watchdog.run_watched`, `MEMORY_CAP_MB`, `HEAVY_MEMORY_MB` (Task 0.2).
- Produces: the K-1 measurements (wall time, CPU time, peak memory, T, M and A per program and shape) and the FeMoco projection, which gate Phase 1 (G0). Also:
  - the stand-in operator family `Proto*` (structure of spec 9.21 and 9.22, with pinned `Adjoint(...)` and `C(...)` rules);
  - `Layout(shape)`, `random_tables(shape, seed)` and `build(shape_name, program) -> (qfunc, num_wires)` for the named shapes `small`, `mid10`, `mid14` and `femoco`;
  - `stand_in_gates(shape_name, program) -> int`, the emitted gates from the resource functions (no compile), `project(calibration) -> dict` and `femoco_refusal() -> str | None`.

  Task 1B.3 (master plan) builds its benchmark workload the way `build` does, from `Shape`, `Layout`, `random_tables` and `ProtoWalk`, at two shapes:
  - the `H2_STO3G_TEST` shape, `Shape(N=2, R=2, B=2, C=1, n_plus=1, b_rot=4, b_k1=2, b_k2=2, k1=1, k2=1)`. A dry-run compiled the stand-in walk there: 67 wires with the control, T 138 and 140 (controlled), M 50, A 88 and 90;
  - the `H2_631G_TEST` shape, which is `SHAPES["small"]` (`n_plus` only sets sign flags in the tables).

**Pass criteria (spec 16, K-1; master plan section 12.9):**
- the small shape (all four programs) and the mid shapes, `mid10` (N = 10, R = 8, B = 6, C = 5: walk and controlled walk) and `mid14` (N = 14, R = 6, B = 13, C = 8: walk, controlled walk and the diagnostic inner prepare and inner select), each compile as one program within `LIMITS`: 5 minutes at the small shape and 15 minutes at the mid shapes, and 4 GB, so none of them is a heavy process;
- counts are non-symbolic, and the controlled walk costs exactly 2 more Toffolis, 2 more measured uncomputes and the same number of explicit measurements as the walk, at every shape that compiles both. The driver checks this (`controlled_walk_check`) before it sets the verdict;
- every compile runs in its own process under `run_watched`, which samples the physical footprint of its process tree, kills the tree above the limit and records the kill. The sampled peak (`peak_mb`) is the gate metric; `rusage_sum_mb`, the compile process's `ru_maxrss` plus that of its finished children, is recorded as an upper bound (`memory_method` in the JSON);
- `--project` fits the calibration shapes (`PROJECTION_METHOD`) and writes the projected FeMoco walk and controlled walk, with time, sampled peak, the upper bound and the method, to `k1_projection.json`. `femoco_allowed` is true only if both stay within 60 minutes and 8 GB. The projection refuses calibration results of other code (`code_hash()` over the driver, the prototypes and `dfthc.compile`);
- the FeMoco driver (`--shape femoco --all`) refuses, with exit status 2 and without compiling, unless the projection belongs to the current code and allows the run, and unless no FeMoco result exists for the current code.

**Definition of done:** `k1_small.json`, `k1_mid10.json` and `k1_mid14.json` committed with `"verdict": "pass"`, and `k1_projection.json` committed; the report gives the projection with its method and says whether the FeMoco compile may run; `ruff check spikes` clean. Do not run the FeMoco driver yourself.

**Gate rule** (spec 16, K-1; master plan section 12.6):
1. If `femoco_allowed` is true, the orchestrator runs `--shape femoco --all` once, alone, at lowered priority under the watchdog (8 GB cap), as two of the six FeMoco-scale compiles, and commits `k1_femoco.json` on this task's branch.
2. If the projection or that run exceeds 60 minutes or 8 GB, the orchestrator opens Task 0.9b (master plan), which applies remedy 1 in the stand-in and re-measures at the small and mid shapes. Remedy 1 passes the bulk tables of never-adjointed operators (reads, angle loads, erases) as dynamic arrays and iterates with `qp.for_loop` and `qp.cond`.
3. Remedy 2 (packing the parity constants) is already the baseline of `mcm_parities`. Its other half, never passing masks as compilable data, is already met by `ProtoTableErase`.
4. If the projection still exceeds the limits, the orchestrator escalates to the user at G0 with the measurements, the projection and the options: the FeMoco-size compiles on a larger machine, or in an approved idle-time window with a higher cap.

- [ ] **Step 1: Write the stand-in and driver**

Create `spikes/k1_whole_program.py`:

```python
"""Spike K-1: compile a complete FeMoco-54-shaped stand-in walk and controlled walk.

Spec section 16, K-1. The stand-in has the walk's structure: WalkLayout-style registers, the
operator nesting of spec 9.21-9.22 with pinned adjoint and controlled rules, prototype QROAM
reads, measured erases, angle loads and angle erase with random tables of FeMoco shape, and
53-rung Givens ladders built exactly as spec 9.14. Placeholders: the flagged uniform preparation
(Hadamards and an X on the success flag) and the Majorana gate (five Toffoli-class gates).

Usage:
    python spikes/k1_whole_program.py --shape small --program walk   # one compile, JSON line
    python spikes/k1_whole_program.py --shape mid14 --all            # driver: one process per
                                                                     # program, under the watchdog
    python spikes/k1_whole_program.py --project                      # FeMoco projection
The driver compiles SHAPE_PROGRAMS[shape] within LIMITS[shape] (seconds, MB per program) and
writes spikes/results/k1_<shape>.json. Memory is MEMORY_METHOD. The controlled walk must cost
exactly 2 more Toffolis, 2 more measured uncomputes and the same explicit measurements as the walk.
Exit status 1 if a compile fails, is killed or over its limits, or the controlled-walk deltas
differ.

The FeMoco shape is a large-scale compile (master plan section 12.9). --project fits the small and
mid results (PROJECTION_METHOD) and writes k1_projection.json. The FeMoco driver refuses, with exit
status 2 and without compiling, unless that projection belongs to the current code and keeps the
walk and the controlled walk within LIMITS["femoco"], and unless no FeMoco result exists yet for the
current code: it compiles at most once per code version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.decomposition import DecompositionGraph
from pennylane.typing import Wire

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prototypes import (  # noqa: E402
    ProtoTableErase,
    ProtoTableRead,
    _Count,
    _Emit,
    _measure_x,
    counts_to_resources,
    from_words,
    nested_iterate,
    to_words,
)

import dfthc.compile  # noqa: E402
from dfthc.compile import (  # noqa: E402
    FIXED_DECOMPS,
    GATE_SET,
    compiled_resources,
    mcm_parities,
    register_rule,
    resource_key,
)
from dfthc.testing.watchdog import HEAVY_MEMORY_MB, MEMORY_CAP_MB, run_watched  # noqa: E402


def lg(value: int) -> int:
    return max(1, math.ceil(math.log2(value)))


@dataclass(frozen=True)
class Shape:
    N: int
    R: int
    B: int
    C: int
    n_plus: int
    b_rot: int
    b_k1: int
    b_k2: int
    k1: int
    k2: int

    @property
    def X_o(self):
        return self.N + self.R * self.C

    @property
    def RC(self):
        return self.R * self.C

    @property
    def w_x(self):
        return lg(self.X_o)

    @property
    def w_b(self):
        return lg(self.B + 1)

    @property
    def w_r(self):
        return lg(self.R)

    @property
    def d1(self):
        return self.w_x + self.b_k1

    @property
    def d2(self):
        return self.w_b + self.b_k2 + 2 + self.w_r


SHAPES = {
    "femoco": Shape(N=54, R=10, B=27, C=27, n_plus=27, b_rot=15, b_k1=15, b_k2=15, k1=2, k2=4),
    "mid14": Shape(N=14, R=6, B=13, C=8, n_plus=4, b_rot=15, b_k1=15, b_k2=15, k1=2, k2=2),
    "mid10": Shape(N=10, R=8, B=6, C=5, n_plus=5, b_rot=15, b_k1=15, b_k2=15, k1=2, k2=1),
    "small": Shape(N=4, R=2, B=2, C=2, n_plus=2, b_rot=4, b_k1=2, b_k2=2, k1=1, k2=1),
}


class Layout:
    """Stand-in for WalkLayout: consecutive named registers (spec 10.1)."""

    def __init__(self, s: Shape):
        self._next = 0
        reg = self._take
        self.psi_up, self.psi_dn = reg(s.N), reg(s.N)
        self.x_o, self.succ_x_o, self.sigma_1 = reg(s.w_x), reg(1), reg(1)
        self.m_x_o, self.lt_x_o, self.cmp_x_o = reg(s.b_k1), reg(1), reg(s.b_k1 - 1)
        self.alt_x_o, self.keep_x_o = reg(s.w_x), reg(s.b_k1)
        self.junk_x_o = reg((2**s.k1 - 1) * s.d1)
        self.b, self.succ_b, self.sigma_0 = reg(s.w_b), reg(1), reg(1)
        self.m_b, self.lt_b, self.cmp_b = reg(s.b_k2), reg(1), reg(s.b_k2 - 1)
        self.alt_b, self.keep_b, self.sgn, self.d_sgn = reg(s.w_b), reg(s.b_k2), reg(1), reg(1)
        self.r = reg(s.w_r)
        self.junk_b = reg((2**s.k2 - 1) * s.d2)
        self.angles = reg((s.N - 1) * s.b_rot)
        self.G0, self.G1, self.b_neq_B, self.sg = reg(1), reg(1), reg(1), reg(1)
        self.phase_grad = reg(s.b_rot)
        refab = s.w_x + s.b_k1 + s.w_b + s.b_k2 + 2
        self.work = reg(max(refab, s.w_x + s.w_b + 2, s.b_rot) + 2)
        self.num_wires = self._next

    def _take(self, size):
        wires = list(range(self._next, self._next + size))
        self._next += size
        return wires


def random_tables(s: Shape, seed: int = 54):
    """Random tables of the walk's shapes, as compilable tuples of 32-bit words."""
    rng = np.random.default_rng(seed)
    outer = tuple(
        (x, to_words(int(rng.integers(0, s.X_o)) | int(rng.integers(0, 2**s.b_k1)) << s.w_x, s.d1))
        for x in range(s.X_o)
    )
    inner = []
    for row in range(s.RC):
        for b in range(s.B + 1):
            value = int(rng.integers(0, 2**s.d2 // 2**s.w_r)) | (row // s.C) << (s.d2 - s.w_r)
            inner.append(((row << s.w_b) | b, to_words(value, s.d2)))
    angle_bits = (s.N - 1) * s.b_rot

    def angles():
        value = 0
        for k in range(s.N - 1):
            value |= int(rng.integers(0, 2 ** (s.b_rot - 1))) << (k * s.b_rot)
        return value

    tail = tuple(
        (s.RC + j, to_words(angles() | 1 << angle_bits | int(j >= s.n_plus) << (angle_bits + 1),
                            angle_bits + 2))
        for j in range(s.N)
    )
    sf = tuple(
        ((r << s.w_b) | b, to_words(angles() | 1 << angle_bits, angle_bits + 1))
        for r in range(s.R) for b in range(s.B)
    )
    return outer, tuple(inner), tail, sf


def _abstract(op_class, *args, **kwargs):
    return resource_key(lambda: op_class(*args, **kwargs))


# --------------------------------------------------------------------------- small blocks


class ProtoUniformPrep(Operator2):
    """Placeholder for FlaggedUniformPrep: Hadamards on target, X on succ."""

    wire_argnames = ("target", "succ")

    def __init__(self, target, succ):
        super().__init__(target, succ)


@qp.register_resources(lambda target, succ: {qp.H: len(target), qp.X: 1})
def _proto_uniform(target, succ):
    for wire in target:
        qp.H(wire)
    qp.X(succ[0])


register_rule(ProtoUniformPrep, _proto_uniform)


class ProtoSpinMux(Operator2):
    wire_argnames = ("G0", "sigma_0", "sigma_1", "sg")

    def __init__(self, G0, sigma_0, sigma_1, sg):
        super().__init__(G0, sigma_0, sigma_1, sg)


@qp.register_resources({qp.CNOT: 3, qp.Toffoli: 1})
def _proto_spin_mux(G0, sigma_0, sigma_1, sg):
    qp.CNOT([sigma_0[0], sg[0]])
    qp.CNOT([sigma_0[0], sigma_1[0]])
    qp.Toffoli([G0[0], sigma_1[0], sg[0]])
    qp.CNOT([sigma_0[0], sigma_1[0]])


def _self_adjoint_resources(base):
    return {_abstract(type(base), **base.arguments): 1}


def _self_adjoint(base):
    type(base)(**base.arguments)


register_rule(ProtoSpinMux, _proto_spin_mux)
register_rule("Adjoint(ProtoSpinMux)",
              qp.register_resources(_self_adjoint_resources)(_self_adjoint))


class ProtoSpinSwap(Operator2):
    wire_argnames = ("sg", "psi_up", "psi_dn")

    def __init__(self, sg, psi_up, psi_dn):
        super().__init__(sg, psi_up, psi_dn)


@qp.register_resources(lambda sg, psi_up, psi_dn: {qp.CSWAP: len(psi_up)})
def _proto_spin_swap(sg, psi_up, psi_dn):
    for up, dn in zip(psi_up, psi_dn, strict=True):
        qp.CSWAP([sg[0], up, dn])


register_rule(ProtoSpinSwap, _proto_spin_swap)
register_rule("Adjoint(ProtoSpinSwap)",
              qp.register_resources(_self_adjoint_resources)(_self_adjoint))


class ProtoInnerSigns(Operator2):
    wire_argnames = ("sgn", "d_sgn", "less_than_b")

    def __init__(self, sgn, d_sgn, less_than_b):
        super().__init__(sgn, d_sgn, less_than_b)


@qp.register_resources({qp.Z: 1, qp.CZ: 1})
def _proto_inner_signs(sgn, d_sgn, less_than_b):
    qp.Z(sgn[0])
    qp.CZ([less_than_b[0], d_sgn[0]])


register_rule(ProtoInnerSigns, _proto_inner_signs)
register_rule("Adjoint(ProtoInnerSigns)",
              qp.register_resources(_self_adjoint_resources)(_self_adjoint))


class ProtoMajorana(Operator2):
    """Placeholder Majorana gate with five Toffoli-class gates (spec 9.16 cost)."""

    wire_argnames = ("succ_x_o", "succ_b", "G0", "G1", "b_neq_B", "flavour", "target", "work")

    def __init__(self, succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
        super().__init__(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work)


def _majorana_resources(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
    return {resource_key(lambda: qp.TemporaryAND(Wire[3])): 2,
            resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))): 2,
            qp.Toffoli: 3, qp.H: 4, qp.CZ: 1}


@qp.register_resources(_majorana_resources)
def _proto_majorana(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
    a, s = work[0], work[1]
    qp.TemporaryAND([succ_x_o[0], G0[0], a])
    qp.TemporaryAND([succ_x_o[0], succ_b[0], s])
    qp.H(target[0])
    qp.Toffoli([s, b_neq_B[0], target[0]])
    qp.H(target[0])
    qp.Toffoli([a, G1[0], target[0]])
    qp.H(flavour[0])
    qp.Toffoli([a, target[0], flavour[0]])
    qp.H(flavour[0])
    qp.CZ([a, target[0]])
    qp.adjoint(qp.TemporaryAND([succ_x_o[0], succ_b[0], s]))
    qp.adjoint(qp.TemporaryAND([succ_x_o[0], G0[0], a]))


register_rule(ProtoMajorana, _proto_majorana)


# --------------------------------------------------------------------------- Givens (spec 9.14)


class ProtoGivensRotation(Operator2):
    wire_argnames = ("q_i", "q_j", "angle", "phase_grad", "work")

    def __init__(self, q_i, q_j, angle, phase_grad, work):
        super().__init__(q_i, q_j, angle, phase_grad, work)


def _givens_resources(q_i, q_j, angle, phase_grad, work):
    b = len(angle)
    adder = resource_key(lambda: qp.SemiAdder(Wire[b], Wire[b], Wire[b - 1]))
    return {qp.H: 8 + 6, qp.S: 3, resource_key(lambda: qp.adjoint(qp.S(Wire[1]))): 3,
            qp.CNOT: 2 + 4 * b, qp.X: 4, adder: 2}


def _rz(q, angle, phase_grad, work):
    qp.X(q)
    for wire in phase_grad:
        qp.CNOT([q, wire])
    qp.SemiAdder(angle, phase_grad, work)
    for wire in phase_grad:
        qp.CNOT([q, wire])
    qp.X(q)


@qp.register_resources(_givens_resources)
def _proto_givens(q_i, q_j, angle, phase_grad, work):
    i, j = q_i[0], q_j[0]
    qp.H(i), qp.adjoint(qp.S(i)), qp.H(i)
    qp.S(j), qp.H(j), qp.adjoint(qp.S(j)), qp.H(j)
    qp.CNOT([i, j])
    qp.H(i)
    _rz(i, angle, phase_grad, work)
    _rz(j, angle, phase_grad, work)
    qp.H(i)
    qp.CNOT([i, j])
    qp.H(i), qp.S(i), qp.H(i)
    qp.H(j), qp.S(j), qp.H(j), qp.adjoint(qp.S(j))


register_rule(ProtoGivensRotation, _proto_givens)


class ProtoGivensLadder(Operator2):
    compilable_argnames = ("b_rot",)
    wire_argnames = ("system", "angles", "phase_grad", "work")

    def __init__(self, b_rot, system, angles, phase_grad, work):
        super().__init__(b_rot, system, angles, phase_grad, work)


def _ladder_resources(b_rot, system, angles, phase_grad, work):
    rung = _abstract(ProtoGivensRotation, Wire[1], Wire[1], Wire[b_rot], Wire[b_rot],
                     Wire[b_rot - 1])
    return {rung: len(system) - 1}


@qp.register_resources(_ladder_resources)
def _proto_ladder(b_rot, system, angles, phase_grad, work):
    for k in range(len(system) - 1):
        ProtoGivensRotation([system[k]], [system[k + 1]], angles[k * b_rot:(k + 1) * b_rot],
                            phase_grad, work[: b_rot - 1])


register_rule(ProtoGivensLadder, _proto_ladder)


# --------------------------------------------------------------------------- reflections


class ProtoZeroReflection(Operator2):
    """2 Pi - I on zero_wires (pattern 0) and one_wires (pattern 1), spec 9.6."""

    wire_argnames = ("zero_wires", "one_wires", "work")

    def __init__(self, zero_wires, one_wires, work):
        super().__init__(zero_wires, one_wires, work)


def _mcz(controls, values, target, work):
    """Multi-controlled Z with pattern ``values`` on controls, 1 on target.

    Inside a rule, wire registers are arrays: never test them for truth (``work or None``).
    """
    num_work = max(len(controls) - 2, 0)
    qp.H(target)
    qp.MultiControlledX(wires=list(controls) + [target], control_values=list(values),
                        work_wires=list(work[:num_work]) if num_work else None,
                        work_wire_type="zeroed")
    qp.H(target)


def _mcz_key(num_controls, values):
    return resource_key(lambda: qp.MultiControlledX(
        Wire[num_controls + 1], control_values=list(values),
        work_wires=Wire[max(num_controls - 2, 0)] if num_controls > 2 else None,
        work_wire_type="zeroed"))


def _reflection_parts(zero_wires, one_wires):
    pattern = [0] * len(zero_wires) + [1] * len(one_wires)
    wires = list(zero_wires) + list(one_wires)
    return wires[:-1], pattern[:-1], wires[-1], pattern[-1]


def _zero_reflection_resources(zero_wires, one_wires, work):
    controls, values, _, last = _reflection_parts(zero_wires, one_wires)
    return {_mcz_key(len(controls), values): 1, qp.H: 2, qp.X: 2 * (last == 0), qp.GlobalPhase: 1}


@qp.register_resources(_zero_reflection_resources)
def _proto_zero_reflection(zero_wires, one_wires, work):
    controls, values, target, last = _reflection_parts(zero_wires, one_wires)
    if last == 0:
        qp.X(target)
    _mcz(controls, values, target, work)
    if last == 0:
        qp.X(target)
    qp.GlobalPhase(np.pi)


def _c_zero_reflection_resources(base, control_wires, control_values, work_wires, work_wire_type):
    controls, values, _, last = _reflection_parts(base.zero_wires, base.one_wires)
    return {_mcz_key(len(controls) + 1, [1] + list(values)): 1, qp.H: 2, qp.X: 2 * (last == 0),
            qp.Z: 1}


@qp.register_resources(_c_zero_reflection_resources)
def _proto_c_zero_reflection(base, control_wires, control_values, work_wires, work_wire_type):
    controls, values, target, last = _reflection_parts(base.zero_wires, base.one_wires)
    if last == 0:
        qp.X(target)
    _mcz([control_wires[0]] + list(controls), [1] + list(values), target, base.work)
    if last == 0:
        qp.X(target)
    qp.Z(control_wires[0])


register_rule(ProtoZeroReflection, _proto_zero_reflection)
register_rule("Adjoint(ProtoZeroReflection)",
              qp.register_resources(_self_adjoint_resources)(_self_adjoint))
register_rule("C(ProtoZeroReflection)", _proto_c_zero_reflection)


# --------------------------------------------------------------------------- walk assembly


class ProtoWalkPart(Operator2):
    """Base for the stand-in blocks that take the whole layout through one compilable shape."""

    compilable_argnames = ("shape", "tables")
    wire_argnames = ("wires",)

    def __init__(self, shape, tables, wires):
        super().__init__(shape, tables, wires)


def _layout(shape_tuple):
    return Layout(Shape(*shape_tuple))


def _w(wires, register):
    return [wires[i] for i in register]


def _read_args(s, lay, wires, which, tables):
    outer, inner, tail, sf = tables
    if which == "outer":
        return ((outer, ((s.w_x - s.k1, 0, -(-s.X_o // 2**s.k1)),), s.k1, s.d1),
                (_w(wires, lay.x_o[: s.w_x - s.k1]), _w(wires, lay.x_o[s.w_x - s.k1:]),
                 _w(wires, lay.alt_x_o + lay.keep_x_o), _w(wires, lay.junk_x_o),
                 _w(wires, lay.work)))
    return ((inner, ((s.w_x, 0, s.RC), (s.w_b - s.k2, 0, -(-(s.B + 1) // 2**s.k2))), s.k2, s.d2),
            (_w(wires, lay.x_o + lay.b[: s.w_b - s.k2]), _w(wires, lay.b[s.w_b - s.k2:]),
             _w(wires, lay.alt_b + lay.keep_b + lay.sgn + lay.d_sgn + lay.r),
             _w(wires, lay.junk_b), _w(wires, lay.work)))


def _prepare_body(s, lay, wires, which, tables, unprepare):
    static, wire_args = _read_args(s, lay, wires, which, tables)
    if which == "outer":
        index, succ, extra = lay.x_o, lay.succ_x_o, lay.sigma_1
        m, lt, cmp_work, alt, keep = lay.m_x_o, lay.lt_x_o, lay.cmp_x_o, lay.alt_x_o, lay.keep_x_o
    else:
        index, succ, extra = lay.b, lay.succ_b, lay.sigma_0
        m, lt, cmp_work, alt, keep = lay.m_b, lay.lt_b, lay.cmp_b, lay.alt_b, lay.keep_b
    def comparator():
        return qp.LeftQuantumComparator(_w(wires, keep), _w(wires, m), _w(wires, lt),
                                        _w(wires, cmp_work), "<=")

    swaps = [(lt[0], a, x) for a, x in zip(alt, index, strict=True)]
    if not unprepare:
        ProtoUniformPrep(_w(wires, index), _w(wires, succ))
        for wire in m + extra:
            qp.H(wires[wire])
        ProtoTableRead(*static, *wire_args)
        comparator()
        for c, a, x in swaps:
            qp.CSWAP([wires[c], wires[a], wires[x]])
    else:
        for c, a, x in swaps:
            qp.CSWAP([wires[c], wires[a], wires[x]])
        qp.adjoint(comparator())
        ProtoTableErase(*static, *wire_args)
        for wire in m + extra:
            qp.H(wires[wire])
        qp.adjoint(ProtoUniformPrep(_w(wires, index), _w(wires, succ)))


def _prepare_resources(s, lay, which, tables, unprepare):
    static, _ = _read_args(s, lay, list(range(lay.num_wires)), which, tables)
    nh, nl = sum(w for w, _, _ in static[1]), static[2]
    width, junk = static[3], (2**static[2] - 1) * static[3]
    n_index = s.w_x if which == "outer" else s.w_b
    b_k = s.b_k1 if which == "outer" else s.b_k2
    table_op = ProtoTableErase if unprepare else ProtoTableRead
    def maybe_adjoint(op):
        return qp.adjoint(op) if unprepare else op

    cmp_key = resource_key(lambda: maybe_adjoint(qp.LeftQuantumComparator(
        Wire[b_k], Wire[b_k], Wire[1], Wire[b_k - 1], "<=")))
    uniform = resource_key(lambda: maybe_adjoint(ProtoUniformPrep(Wire[n_index], Wire[1])))
    table = _abstract(table_op, *static, Wire[nh], Wire[nl], Wire[width], Wire[junk],
                      Wire[len(lay.work)])
    return {uniform: 1, qp.H: b_k + 1, table: 1, cmp_key: 1, qp.CSWAP: n_index}


def _make_prepare(name, which):
    cls = type(name, (ProtoWalkPart,), {"__doc__": f"Stand-in {which} prepare (spec 9.7/9.8)."})

    def resources(shape, tables, wires):
        return _prepare_resources(Shape(*shape), _layout(shape), which, tables, False)

    def rule(shape, tables, wires):
        _prepare_body(Shape(*shape), _layout(shape), wires, which, tables, False)

    def adjoint_resources(base):
        shape, tables = base.shape, base.tables
        return _prepare_resources(Shape(*shape), _layout(shape), which, tables, True)

    def adjoint_rule(base):
        _prepare_body(Shape(*base.shape), _layout(base.shape), base.wires, which, base.tables, True)

    rule.__name__ = f"_{name.lower()}_rule"
    adjoint_rule.__name__ = f"_{name.lower()}_unprepare"
    register_rule(cls, qp.register_resources(resources)(rule))
    register_rule(f"Adjoint({name})", qp.register_resources(adjoint_resources)(adjoint_rule))
    return cls


ProtoOuterPrepare = _make_prepare("ProtoOuterPrepare", "outer")
ProtoInnerPrepare = _make_prepare("ProtoInnerPrepare", "inner")


class ProtoAngleErase(ProtoWalkPart):
    """Stand-in AngleErase: measure angles and flags, SF fixup on G0 = 0, then tail fixup."""


def _angle_erase_body(emit, s, lay, wires, tables, measure, cond_z):
    _, _, tail, sf = tables
    angle_wires = _w(wires, lay.angles + lay.b_neq_B)
    outcomes = [measure(q) for q in angle_wires]
    sf_masks = [from_words(e) for _, e in sf]
    sf_parities = mcm_parities(outcomes, sf_masks) if outcomes else []
    registers = [(_w(wires, lay.r), 0, s.R), (_w(wires, lay.b), 0, s.B)]
    emit.x(wires[lay.G0[0]])
    index = iter(range(len(sf_masks)))
    nested_iterate(emit, registers, _w(wires, lay.work),
                   lambda flag, v: cond_z(sf_parities[next(index)], flag),
                   control=wires[lay.G0[0]])
    emit.x(wires[lay.G0[0]])
    flag_outcomes = [measure(wires[lay.G0[0]]), measure(wires[lay.G1[0]])]
    angle_bits = (s.N - 1) * s.b_rot
    tail_masks = []
    for _, entry in tail:
        value = from_words(entry)
        tail_masks.append((value & (2**angle_bits - 1)) | (value >> angle_bits) << (angle_bits + 1))
    tail_parities = mcm_parities(outcomes + flag_outcomes, tail_masks)
    index = iter(range(len(tail_masks)))
    nested_iterate(emit, [(_w(wires, lay.x_o), s.RC, s.X_o)], _w(wires, lay.work),
                   lambda flag, v: cond_z(tail_parities[next(index)], flag))


class _AngleEmit(_Emit):
    pass


class _AngleCount(_Count):
    pass


def _angle_erase_resources(shape, tables, wires):
    s, lay = Shape(*shape), _layout(shape)
    counter = _AngleCount()
    measured = len(lay.angles) + 3
    counter.counts["Hadamard"] += measured
    counter.counts["MidMeasure"] += measured

    def count_measure(q):
        return 1

    def count_cond(parity, flag):
        if parity is not False:
            counter.counts["PauliZ"] += 1

    _angle_erase_body(counter, s, lay, list(range(lay.num_wires)), tables, count_measure,
                      count_cond)
    counter.counts["Hadamard"] -= 0
    counter.counts["MidMeasure"] -= 0
    return counts_to_resources(counter.counts)


@qp.register_resources(_angle_erase_resources)
def _proto_angle_erase(shape, tables, wires):
    _angle_erase_body(_AngleEmit(), Shape(*shape), _layout(shape), wires, tables, _measure_x,
                      lambda parity, flag: qp.cond(parity, qp.Z)(flag) if parity is not False
                      else None)


register_rule(ProtoAngleErase, _proto_angle_erase)


class ProtoTailAngleLoad(ProtoWalkPart):
    """Stand-in TailAngleLoad: a read of rows [RC, X_o) without swap network (spec 9.10)."""


class ProtoSFAngleLoad(ProtoWalkPart):
    """Stand-in SFAngleLoad: a read over (r, b), enabled on G0 = 0 (spec 9.11)."""


def _angle_read(s, lay, wires, tables, which):
    _, _, tail, sf = tables
    angle_bits = (s.N - 1) * s.b_rot
    if which == "tail":
        static = (tail, ((s.w_x, s.RC, s.X_o),), 0, angle_bits + 2)
        return static, (_w(wires, lay.x_o), [], _w(wires, lay.angles + lay.G0 + lay.G1), [],
                        _w(wires, lay.work))
    static = (sf, ((s.w_r, 0, s.R), (s.w_b, 0, s.B)), 0, angle_bits + 1)
    return static, (_w(wires, lay.r + lay.b), [], _w(wires, lay.angles + lay.b_neq_B), [],
                    _w(wires, lay.work))


def _make_angle_load(cls, which):
    def resources(shape, tables, wires):
        s, lay = Shape(*shape), _layout(shape)
        static, wire_args = _angle_read(s, lay, list(range(lay.num_wires)), tables, which)
        key = _abstract(ProtoTableRead, *static, *(Wire[len(w)] for w in wire_args))
        return {key: 1, qp.X: 2 * (which == "sf")}

    def rule(shape, tables, wires):
        s, lay = Shape(*shape), _layout(shape)
        static, wire_args = _angle_read(s, lay, wires, tables, which)
        if which == "sf":
            qp.X(wires[lay.G0[0]])
        # The SF load is enabled on G0 = 0; the stand-in read ignores the enable (placeholder).
        ProtoTableRead(*static, *wire_args)
        if which == "sf":
            qp.X(wires[lay.G0[0]])

    rule.__name__ = f"_proto_{which}_angle_load"
    register_rule(cls, qp.register_resources(resources)(rule))


_make_angle_load(ProtoTailAngleLoad, "tail")
_make_angle_load(ProtoSFAngleLoad, "sf")


class ProtoInnerSelect(ProtoWalkPart):
    """Stand-in InnerSelect (spec 9.18); the pinned adjoint uses the adjoint Majorana gate."""


def _select_ops(shape, tables, wires, adjoint_majorana):
    s, lay = Shape(*shape), _layout(shape)
    part = (shape, tables, wires)
    mux = (_w(wires, lay.G0), _w(wires, lay.sigma_0), _w(wires, lay.sigma_1), _w(wires, lay.sg))
    swap = (_w(wires, lay.sg), _w(wires, lay.psi_up), _w(wires, lay.psi_dn))
    ladder = (s.b_rot, _w(wires, lay.psi_dn), _w(wires, lay.angles), _w(wires, lay.phase_grad),
              _w(wires, lay.work))
    majorana = (_w(wires, lay.succ_x_o), _w(wires, lay.succ_b), _w(wires, lay.G0),
                _w(wires, lay.G1), _w(wires, lay.b_neq_B), _w(wires, lay.sigma_0),
                _w(wires, lay.psi_dn[:1]), _w(wires, lay.work[:2]))
    ProtoTailAngleLoad(*part)
    ProtoSFAngleLoad(*part)
    ProtoSpinMux(*mux)
    ProtoSpinSwap(*swap)
    ProtoGivensLadder(*ladder)
    if adjoint_majorana:
        qp.adjoint(ProtoMajorana(*majorana))
    else:
        ProtoMajorana(*majorana)
    qp.adjoint(ProtoGivensLadder(*ladder))
    ProtoSpinSwap(*swap)
    ProtoSpinMux(*mux)
    ProtoAngleErase(*part)


def _select_resources(shape, tables, adjoint_majorana):
    s, lay = Shape(*shape), _layout(shape)
    part_keys = [_abstract(c, shape, tables, Wire[lay.num_wires])
                 for c in (ProtoTailAngleLoad, ProtoSFAngleLoad, ProtoAngleErase)]
    mux = _abstract(ProtoSpinMux, Wire[1], Wire[1], Wire[1], Wire[1])
    swap = _abstract(ProtoSpinSwap, Wire[1], Wire[s.N], Wire[s.N])
    ladder_args = (s.b_rot, Wire[s.N], Wire[len(lay.angles)], Wire[s.b_rot], Wire[len(lay.work)])
    ladder = _abstract(ProtoGivensLadder, *ladder_args)
    ladder_dagger = resource_key(lambda: qp.adjoint(ProtoGivensLadder(*ladder_args)))
    maj_args = (Wire[1],) * 7 + (Wire[2],)
    majorana = resource_key(lambda: qp.adjoint(ProtoMajorana(*maj_args)) if adjoint_majorana
                            else ProtoMajorana(*maj_args))
    return {part_keys[0]: 1, part_keys[1]: 1, part_keys[2]: 1, mux: 2, swap: 2, ladder: 1,
            ladder_dagger: 1, majorana: 1}


register_rule(ProtoInnerSelect, qp.register_resources(
    lambda shape, tables, wires: _select_resources(shape, tables, False))(
    lambda shape, tables, wires: _select_ops(shape, tables, wires, False)))
register_rule("Adjoint(ProtoInnerSelect)", qp.register_resources(
    lambda base: _select_resources(base.shape, base.tables, True))(
    lambda base: _select_ops(base.shape, base.tables, base.wires, True)))


def _refs(shape, wires):
    lay = _layout(shape)
    t2 = (_w(wires, lay.b + lay.m_b + lay.sigma_0), _w(wires, lay.succ_x_o), _w(wires, lay.work))
    ab = (_w(wires, lay.x_o + lay.m_x_o + lay.sigma_1 + lay.b + lay.m_b + lay.sigma_0), [],
          _w(wires, lay.work))
    return t2, ab


class ProtoBlockEncoding(ProtoWalkPart):
    """Stand-in DFTHCBlockEncoding (spec 9.21)."""


class ProtoWalk(ProtoWalkPart):
    """Stand-in DFTHCWalk (spec 9.22)."""


def _be_ops(shape, tables, wires, control=None):
    part = (shape, tables, wires)
    t2, _ = _refs(shape, wires)
    ProtoOuterPrepare(*part)
    ProtoInnerPrepare(*part)
    ProtoInnerSigns(*_signs(shape, wires))
    ProtoInnerSelect(*part)
    qp.adjoint(ProtoInnerPrepare(*part))
    if control is None:
        ProtoZeroReflection(*t2)
    else:
        qp.ctrl(ProtoZeroReflection(*t2), control=control)
    ProtoInnerPrepare(*part)
    qp.adjoint(ProtoInnerSelect(*part))
    ProtoInnerSigns(*_signs(shape, wires))
    qp.adjoint(ProtoInnerPrepare(*part))
    qp.adjoint(ProtoOuterPrepare(*part))


def _signs(shape, wires):
    lay = _layout(shape)
    return _w(wires, lay.sgn), _w(wires, lay.d_sgn), _w(wires, lay.lt_b)


def _be_resources(shape, tables, controlled):
    lay = _layout(shape)
    s = Shape(*shape)
    n = Wire[lay.num_wires]
    keys = {}
    for cls in (ProtoOuterPrepare, ProtoInnerPrepare, ProtoInnerSelect):
        keys[_abstract(cls, shape, tables, n)] = 2 if cls is ProtoInnerPrepare else 1
        keys[resource_key(lambda c=cls: qp.adjoint(c(shape, tables, n)))] = (
            2 if cls is ProtoInnerPrepare else 1)
    keys[_abstract(ProtoInnerSigns, Wire[1], Wire[1], Wire[1])] = 2
    t2_args = (Wire[s.w_b + s.b_k2 + 1], Wire[1], Wire[len(lay.work)])
    ref = (resource_key(lambda: qp.ctrl(ProtoZeroReflection(*t2_args), control=Wire[1]))
           if controlled else _abstract(ProtoZeroReflection, *t2_args))
    keys[ref] = 1
    return keys


def _ab_args(shape):
    s, lay = Shape(*shape), _layout(shape)
    return (Wire[s.w_x + s.b_k1 + s.w_b + s.b_k2 + 2], Wire[0], Wire[len(lay.work)])


register_rule(ProtoBlockEncoding, qp.register_resources(
    lambda shape, tables, wires: _be_resources(shape, tables, False))(
    lambda shape, tables, wires: _be_ops(shape, tables, wires)))


def _c_be_resources(base, control_wires, control_values, work_wires, work_wire_type):
    return _be_resources(base.shape, base.tables, True)


def _c_be(base, control_wires, control_values, work_wires, work_wire_type):
    _be_ops(base.shape, base.tables, base.wires, control=control_wires[0])


register_rule("C(ProtoBlockEncoding)", qp.register_resources(_c_be_resources)(_c_be))


def _walk_resources(shape, tables, wires):
    n = Wire[_layout(shape).num_wires]
    return {_abstract(ProtoBlockEncoding, shape, tables, n): 1,
            _abstract(ProtoZeroReflection, *_ab_args(shape)): 1}


def _walk(shape, tables, wires):
    ProtoBlockEncoding(shape, tables, wires)
    _, ab = _refs(shape, wires)
    ProtoZeroReflection(*ab)


def _c_walk_resources(base, control_wires, control_values, work_wires, work_wire_type):
    n = Wire[_layout(base.shape).num_wires]
    return {resource_key(lambda: qp.ctrl(ProtoBlockEncoding(base.shape, base.tables, n),
                                         control=Wire[1])): 1,
            resource_key(lambda: qp.ctrl(ProtoZeroReflection(*_ab_args(base.shape)),
                                         control=Wire[1])): 1}


def _c_walk(base, control_wires, control_values, work_wires, work_wire_type):
    qp.ctrl(ProtoBlockEncoding(base.shape, base.tables, base.wires), control=control_wires[0])
    _, ab = _refs(base.shape, base.wires)
    qp.ctrl(ProtoZeroReflection(*ab), control=control_wires[0])


register_rule(ProtoWalk, qp.register_resources(_walk_resources)(_walk))
register_rule("C(ProtoWalk)", qp.register_resources(_c_walk_resources)(_c_walk))


# --------------------------------------------------------------------------- driver

PROGRAMS = ("walk", "controlled_walk", "inner_prepare", "inner_select")
SHAPE_PROGRAMS = {
    "small": PROGRAMS,
    "mid10": ("walk", "controlled_walk"),
    "mid14": PROGRAMS,
    "femoco": ("walk", "controlled_walk"),
}
CALIBRATION_SHAPES = ("small", "mid10", "mid14")
LIMITS = {
    "small": (300, HEAVY_MEMORY_MB),
    "mid10": (900, HEAVY_MEMORY_MB),
    "mid14": (900, HEAVY_MEMORY_MB),
    "femoco": (3600, MEMORY_CAP_MB),
}
CONTROL_DELTAS = {"toffoli": 2, "measured_and_uncomputes": 2, "measurements": 0}
MEMORY_METHOD = (
    "peak_mb: sampled peak of the summed physical footprint of the compile's process tree "
    "(dfthc.testing.watchdog), the gate metric; the watchdog kills the tree above the limit. "
    "rusage_sum_mb: ru_maxrss of the compile process plus ru_maxrss of its finished children "
    "(getrusage), an upper bound, because Catalyst runs its compiler and linker as blocking "
    "subprocesses while Python holds the traced program"
)
PROJECTION_METHOD = (
    "size measure: the stand-in program's emitted gates, from its resource functions "
    "(DecompositionGraph.resource_estimate, no compile). Time: power law through the two largest "
    "calibration shapes, exponent at least 1. Memory: least-squares line through all calibration "
    "shapes, slope at least 0, for peak_mb (gate) and rusage_sum_mb (upper bound). The FeMoco "
    "compile may run if, for the walk and the controlled walk, the projected time and peak_mb "
    "are within LIMITS['femoco']"
)
RESULTS = Path(__file__).resolve().parent / "results"
_HASHED_FILES = (Path(__file__).resolve(), Path(__file__).resolve().parent / "prototypes.py",
                 Path(dfthc.compile.__file__).resolve())


def code_hash() -> str:
    """SHA-256 over this driver, the prototypes and dfthc.compile: what a compile depends on."""
    digest = hashlib.sha256()
    for path in _HASHED_FILES:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build(shape_name: str, program: str):
    s = SHAPES[shape_name]
    shape = tuple(getattr(s, f) for f in Shape.__dataclass_fields__)
    lay = Layout(s)
    tables = random_tables(s)
    wires = list(range(lay.num_wires))
    control = lay.num_wires

    def qfunc():
        qp.PhaseGradientStatePrep(lay.phase_grad)
        if program == "walk":
            ProtoWalk(shape, tables, wires)
        elif program == "controlled_walk":
            qp.ctrl(ProtoWalk(shape, tables, wires), control=control)
        elif program == "inner_prepare":
            ProtoInnerPrepare(shape, tables, wires)
            qp.adjoint(ProtoInnerPrepare(shape, tables, wires))
        else:
            ProtoInnerSelect(shape, tables, wires)

    return qfunc, lay.num_wires + 1


def stand_in_gates(shape_name: str, program: str) -> int:
    """Gates the walk or controlled walk emits, from the resource functions alone (no compile)."""
    s = SHAPES[shape_name]
    shape = tuple(getattr(s, f) for f in Shape.__dataclass_fields__)
    lay = Layout(s)
    with qp.queuing.AnnotatedQueue():
        op = ProtoWalk(shape, random_tables(s), list(range(lay.num_wires)))
        if program == "controlled_walk":
            op = qp.ctrl(op, control=lay.num_wires)
    graph = DecompositionGraph(operations=[op], gate_set=GATE_SET, fixed_decomps=FIXED_DECOMPS)
    return int(graph.solve().resource_estimate(op).num_gates)


def compile_one(shape_name: str, program: str) -> dict:
    qfunc, num_wires = build(shape_name, program)
    start = time.perf_counter()
    resources = compiled_resources(qfunc, num_wires=num_wires)
    wall = time.perf_counter() - start
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if sys.platform == "darwin" else 1024
    return {"program": program, "shape": shape_name, "num_wires": num_wires,
            "seconds": round(wall, 1),
            "cpu_seconds": round(own.ru_utime + own.ru_stime + children.ru_utime
                                 + children.ru_stime, 1),
            "rusage_sum_mb": round((own.ru_maxrss + children.ru_maxrss) * scale / 2**20, 1),
            "compiled_gates": sum(resources.counts.values()),
            "toffoli": resources.toffoli, "measurements": resources.measurements,
            "measured_and_uncomputes": resources.measured_and_uncomputes}


def controlled_walk_check(results: dict) -> dict:
    """Compare the controlled walk's counts with the walk's against CONTROL_DELTAS."""
    walk, controlled = results.get("walk", {}), results.get("controlled_walk", {})
    if not all(key in walk and key in controlled for key in CONTROL_DELTAS):
        return {"ok": False, "reason": "the walk or the controlled walk has no counts"}
    deltas = {key: controlled[key] - walk[key] for key in CONTROL_DELTAS}
    return {"ok": deltas == CONTROL_DELTAS, "deltas": deltas, "expected": CONTROL_DELTAS}


def _fit_time(points, x):
    (x1, t1), (x2, t2) = sorted(points)[-2:]
    exponent = max(1.0, math.log(t2 / t1) / math.log(x2 / x1))
    return t2 * (x / x2) ** exponent, exponent


def _fit_memory(points, x):
    xs, ys = np.array([p[0] for p in points], float), np.array([p[1] for p in points], float)
    slope = max(0.0, float(np.polyfit(xs, ys, 1)[0]))
    intercept = float(ys.mean() - slope * xs.mean())
    return intercept + slope * x, slope


def project(calibration: dict[str, dict]) -> dict:
    """Project the FeMoco walk and controlled walk from the calibration results (PROJECTION_METHOD).

    ``calibration`` maps each of CALIBRATION_SHAPES to its k1_<shape>.json content. Raises
    ValueError if a result is missing, failed, or belongs to other code than the current.
    """
    current = code_hash()
    for shape in CALIBRATION_SHAPES:
        record = calibration.get(shape)
        if record is None or record.get("verdict") != "pass":
            raise ValueError(f"no passing k1_{shape}.json: run --shape {shape} --all first")
        if record.get("code_hash") != current:
            raise ValueError(f"k1_{shape}.json belongs to other code: rerun --shape {shape} --all")
    seconds_limit, memory_limit = LIMITS["femoco"]
    programs, reasons = {}, []
    for program in SHAPE_PROGRAMS["femoco"]:
        shapes = [s for s in CALIBRATION_SHAPES if program in calibration[s]]
        gates = {s: stand_in_gates(s, program) for s in [*shapes, "femoco"]}
        seconds, exponent = _fit_time(
            [(gates[s], calibration[s][program]["seconds"]) for s in shapes], gates["femoco"])
        peak, peak_slope = _fit_memory(
            [(gates[s], calibration[s][program]["peak_mb"]) for s in shapes], gates["femoco"])
        upper, _ = _fit_memory(
            [(gates[s], calibration[s][program]["rusage_sum_mb"]) for s in shapes], gates["femoco"])
        programs[program] = {
            "gates": gates, "seconds": round(seconds), "time_exponent": round(exponent, 2),
            "peak_mb": round(peak), "mb_per_1000_gates": round(1000 * peak_slope, 2),
            "rusage_sum_mb_upper_bound": round(upper)}
        if seconds > seconds_limit:
            reasons.append(f"{program}: {seconds / 60:.0f} min > {seconds_limit / 60:.0f} min")
        if peak > memory_limit:
            reasons.append(f"{program}: {peak / 1024:.1f} GB > {memory_limit / 1024:.0f} GB")
    return {"method": PROJECTION_METHOD, "limits": {"seconds": seconds_limit, "mb": memory_limit},
            "programs": programs, "femoco_allowed": not reasons,
            "reason": "; ".join(reasons) or "within the limits", "code_hash": current}


def femoco_refusal() -> str | None:
    """Why the FeMoco compile may not run now, or None if it may."""
    projection_file = RESULTS / "k1_projection.json"
    if not projection_file.exists():
        return "no projection: run the small and mid shapes, then --project"
    projection = json.loads(projection_file.read_text())
    if projection.get("code_hash") != code_hash():
        return "the projection belongs to other code: rerun the mid shapes and --project"
    if not projection.get("femoco_allowed"):
        return f"the projection exceeds the limits ({projection.get('reason')}); escalate at G0"
    femoco_file = RESULTS / "k1_femoco.json"
    if femoco_file.exists() and json.loads(femoco_file.read_text()).get("code_hash") == code_hash():
        return "the FeMoco compile already ran for this code; it runs at most once"
    return None


def drive(shape_name: str) -> int:
    if shape_name == "femoco":
        refusal = femoco_refusal()
        if refusal:
            print("refused:", refusal, flush=True)
            return 2
    seconds_limit, memory_limit = LIMITS[shape_name]
    results, failed = {}, False
    for program in SHAPE_PROGRAMS[shape_name]:
        command = [sys.executable, __file__, "--shape", shape_name, "--program", program]
        run = run_watched(command, cap_mb=memory_limit, timeout_s=seconds_limit + 60)
        lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
        if run.killed or run.returncode != 0 or not lines:
            reason = f"killed: {run.killed}" if run.killed else run.stderr[-3000:]
            results[program] = {"error": reason, **run.summary()}
            failed = True
        else:
            record = json.loads(lines[-1])
            record["peak_mb"] = round(run.peak_mb, 1)
            record["within_budget"] = (record["seconds"] <= seconds_limit
                                       and run.peak_mb <= memory_limit)
            failed |= not record["within_budget"]
            results[program] = record
        print(program, json.dumps(results[program])[:600], flush=True)
    if "controlled_walk" in SHAPE_PROGRAMS[shape_name]:
        results["controlled_walk_check"] = controlled_walk_check(results)
        failed |= not results["controlled_walk_check"]["ok"]
        print("controlled_walk_check", json.dumps(results["controlled_walk_check"]), flush=True)
    results["limits"] = {"seconds": seconds_limit, "mb": memory_limit}
    results["memory_method"] = MEMORY_METHOD
    results["code_hash"] = code_hash()
    results["verdict"] = "fail" if failed else "pass"
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"k1_{shape_name}.json").write_text(json.dumps(results, indent=2) + "\n")
    print("verdict:", results["verdict"])
    return 1 if failed else 0


def write_projection() -> int:
    calibration = {}
    for shape in CALIBRATION_SHAPES:
        path = RESULTS / f"k1_{shape}.json"
        if path.exists():
            calibration[shape] = json.loads(path.read_text())
    projection = project(calibration)
    (RESULTS / "k1_projection.json").write_text(json.dumps(projection, indent=2) + "\n")
    for program, values in projection["programs"].items():
        print(program, json.dumps(values), flush=True)
    print("femoco_allowed:", projection["femoco_allowed"], "-", projection["reason"])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape", choices=sorted(SHAPES))
    parser.add_argument("--program", choices=PROGRAMS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--project", action="store_true")
    args = parser.parse_args()
    if args.project:
        sys.exit(write_projection())
    if args.shape is None:
        parser.error("--shape is required unless --project is given")
    if args.all:
        sys.exit(drive(args.shape))
    if args.shape == "femoco" and (refusal := femoco_refusal()):
        sys.exit(f"refused: {refusal}")
    print(json.dumps(compile_one(args.shape, args.program)), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the driver at the small shape**

Run: `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/k1_whole_program.py --shape small --all`
Expected (about 45 s), as in the dry-run:

```text
walk {"program": "walk", "shape": "small", "num_wires": 83, "seconds": 9.2, ..., "rusage_sum_mb": 635.1, "compiled_gates": 2201, "toffoli": 236, "measurements": 68, "measured_and_uncomputes": 175, "peak_mb": 498.7, "within_budget": true}
controlled_walk {"program": "controlled_walk", ..., "toffoli": 238, "measurements": 68, "measured_and_uncomputes": 177, ..., "within_budget": true}
inner_prepare {"program": "inner_prepare", ..., "toffoli": 35, "measurements": 14, "measured_and_uncomputes": 24, ..., "within_budget": true}
inner_select {"program": "inner_select", ..., "toffoli": 66, "measurements": 15, "measured_and_uncomputes": 53, ..., "within_budget": true}
controlled_walk_check {"ok": true, "deltas": {"toffoli": 2, "measured_and_uncomputes": 2, "measurements": 0}, ...}
verdict: pass
```

The Toffoli and measurement counts are exact and must match; times and memory vary. A wrong controlled rule shows up as `"ok": false` and `verdict: fail` (round-2 dry-run with canned driver input: T + 3 or M + 1 fails).

- [ ] **Step 3: Run the driver at the mid shapes**

Run, one after the other (about 15 minutes together; each compile stays below 4 GB):

```bash
CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/k1_whole_program.py --shape mid10 --all
CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/k1_whole_program.py --shape mid14 --all
```

Expected: one JSON line per program, the `controlled_walk_check` line with `"ok": true`, and `verdict: pass` for each shape. The round-3 calibration measured, with the stand-in and the sampled resident size of the process tree: the `mid10` walk 60 s and 1.6 GB, controlled walk 124 s and 2.3 GB; the `mid14` walk 128 to 146 s and 1.7 to 2.3 GB, inner prepare 105 s and 1.2 GB, inner select 50 s and 1.0 GB. The `mid14` controlled walk was not measured; expect about 5 minutes and 2.5 to 3.5 GB. If a compile is killed at 4 GB, stop and report: the orchestrator decides whether that shape runs as a heavy process.

- [ ] **Step 4: Project to FeMoco-54**

Run: `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/k1_whole_program.py --project`
Expected (a few seconds; nothing is compiled): one line per program with the gate counts per shape (FeMoco walk 627,557 and controlled walk 627,563 from the resource functions), the projected seconds, time exponent, `peak_mb`, MB per 1,000 gates and `rusage_sum_mb_upper_bound`, then `femoco_allowed: True` or `femoco_allowed: False - <reasons>`.

The dry-run fed the projection the round-3 calibration (with the real small run above and an estimated `mid14` controlled walk) through a stub driver: walk 57 minutes and 12.5 GB (17.6 GB with the other `mid14` run), controlled walk 118 minutes and 17.3 GB, so `femoco_allowed: False`, and the FeMoco driver refused with exit status 2 without compiling. Fed an artificial calibration within the limits, it compiled the walk and the controlled walk once, each with the 8 GB cap, and refused a second run for the same code, a projection of other code, and stale mid results. A stubbed watchdog kill was recorded as `"error": "killed: memory"` with `verdict: fail`. The round-2 driver, given the same over-limit data, launched all four FeMoco compiles.

- [ ] **Step 5: Hand over the FeMoco decision**

Do not run `--shape femoco --all`. Report the projection lines and `femoco_allowed`. If it is true, the orchestrator runs the FeMoco driver once, alone, and adds `k1_femoco.json` to this branch (gate rule 1); otherwise gate rules 2 to 4 apply.

- [ ] **Step 6: Lint and commit the results**

Run: `"$VENV/bin/ruff" check spikes` · Expected: `All checks passed!`

```bash
git add spikes/k1_whole_program.py spikes/results/k1_small.json spikes/results/k1_mid10.json \
        spikes/results/k1_mid14.json spikes/results/k1_projection.json
git commit -m "Add K-1 spike: small and mid stand-in compiles and FeMoco projection"
```
