# DFTHC Phase 1, Workstream B: The Sparse-State Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the purpose-built sparse-state simulator of spec section 12.3, which runs the gate-level walk on H2 end to end (E-1 to E-4, Task 3.5) and the outcome certificates of spec 12.5 (Tasks 1C.5, 2D.1, 2D.2, 2E.2). It has handlers for every gate-set operation, measurements and conditionals with forced, seeded and branch policies, and the tape-path front end. The benchmark gate measures it on the K-1 stand-in walk.

**Architecture:** `dfthc.testing.sparse_sim` stores a state as computational-basis keys packed into `uint64` words, wire 0 most significant, next to `complex128` amplitudes. Each operation is one vectorized NumPy update over all terms; Hadamards split terms and merge equal keys. `dfthc.testing.tape.decompose_for_simulation` records a quantum function as a tape and decomposes it with the pinned rules (`GATE_SET`, `FIXED_DECOMPS`, graph enabled). The simulator is used for semantics only; counts always come from `qp.analyze` on the compiled program.

**Tech Stack:** Python 3.12, NumPy, PennyLane (`default.qubit` with deferred measurements as the differential reference), pytest with pytest-xdist; `dfthc.testing.watchdog` for the benchmark.

This file holds Tasks 1B.1 to 1B.3 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 7). Workstream A is in `2026-11-11-dfthc-phase1-data.md`; that file's sections "Expected test counts" and the count table cover both workstreams.

**Code provenance.** Every code block below was executed on 2026-11-12 in a scratch replica of the repository (`.scratch/plan-check/repo-phase1`, not committed), last resynced to local `main` at `3534c30` (Tasks 0.1, 0.2 and 0.5 as merged). The replica also held the spike modules `spikes/prototypes.py` and `spikes/k3_mode_agnostic_erase.py` (Task 0.7) and `spikes/k1_whole_program.py` (Task 0.9) from the Phase 0 plan's dry-run. The last of these carries the counting fix that Task 0.9 needs against the merged Task 0.2 (change 6). pytest ran the way `scripts/test.sh` does (at most 2 workers, `taskpolicy -c utility`); the replica is not a git repository. The Task 1B.1 version of `sparse_sim.py` was checked against Task 1B.1's tests on its own, before Task 1B.2 replaced it.

**Task 1B.3 was dry-run at reduced scale only:** 2 of the 16 `H2_STO3G_TEST` inputs, and the single `H2_631G_TEST` input. Its full acceptance gate, the 16-input schedule, is unverified until the task runs.

---

## Changes from the master plan

Every change below is applied in the master plan (`2026-11-11-dfthc-master-plan.md`, the version assembled with this file). The spec edit of change 3 is an old-to-new text pair in `phase1-doc-edits.md`, which the orchestrator applies on `main`.

1. **Task 1B.1 interfaces, completed:** `SparseSimulator(num_wires, *, policy=None, strict=True)`; `run(operations, initial_state=None)`; `supported_operations`; `SparseState.copy()`, `bits()`, `flip()`, `key_string()`; `probability()` relative to the state's norm; `PRUNE_THRESHOLD` and `MAX_DENSE_WIRES`. Task 1B.1 covers every gate-set member except `MidMeasure`, whose handler comes with the policies in Task 1B.2. *Applied:* section 7, Task 1B.1 interfaces.
2. **Task 1B.1's `PhaseGradientStatePrep` reference** is the operator's own `state_vector()`, because `default.qubit` cannot apply it after other operations (`MatrixUndefinedError`). *Applied:* section 7, Task 1B.1 acceptance tests.
3. **Task 1B.2 interfaces, completed, with two fixes.**
   - The completions: `BranchPolicy(wires, *, otherwise=None)`; each branch consumes its own copy of the policy; `SparseSimulator.record`; `run` refuses a `BranchPolicy`; `MEASUREMENT_TOLERANCE = 1e-12`; `Conditional` dispatched by type, because the operators are named `Conditional(<base>)`.
   - Strict mode exempts postselected measurements as well as branch measurements. It certifies that erase measurements give uniformly random X outcomes, whereas a postselection is a test device: postselecting a returned ancilla onto `|0>` has p = 1. This amends master plan Task 1B.2 and spec 12.3.
   - Fix 1, measurement normalization. The first version divided by `1 - P(1)`, which assumes norm 1, so every measurement doubled any rounding error, and the stand-in's 50 erase measurements drove the norm down to 0.13. Measurements now normalize by the kept weight itself, and a test pins it.
   - Fix 2, postselection after branching (review round 1). An impossible postselection in one branch used to abort every branch, and the survivors were not reweighted. Now `run_branches` drops a branch whose postselection probability is below `MEASUREMENT_TOLERANCE` and weights the others by their success probability. The returned probabilities are `p_b q_b / sum over survivors of p_b' q_b'`, conditional on every postselection succeeding, as `default.qubit` gives with deferred measurements in analytic mode. It raises `ValueError` only if no branch passes. A new differential test covers a postselection whose success probability is 0, 1/2, 1/2 and 3/4 in four branches.

   *Applied:* section 7, Task 1B.2 interfaces and acceptance tests. The spec 12.3 edit is in `phase1-doc-edits.md`.
4. **The K-3 round trip comes from the spike script:** `round_trip`, `HIGH`, `LOW` and `NUM_WIRES` of `spikes/k3_mode_agnostic_erase.py` (Task 0.7), built on `spikes/prototypes.py`. *Applied:* section 7, Task 1B.2 acceptance tests.
5. **Task 1B.3's form.**
   - It is one test parametrized over the two shapes (2 tests, marked `slow` and `subprocess`).
   - The test module is also the measured program, run as a subprocess under `run_watched` with a 4 GB cap, and it prints one `BENCHMARK {json}` line per shape.
   - Strict mode is on, and the `H2_631G_TEST` input is one electron per spin in orbital 0.
   - A run passes only with `killed is None`, no `survivors` and status 0 (review round 1): the merged watchdog can report unverified survivors while `killed` is `None`.

   *Applied:* section 7, Task 1B.3.
6. **Task 1B.3 needs a fix in Task 0.9's stand-in.** The merged Task 0.2 computes tape-mode parities as one flat `MeasurementValue` and reads `outcomes[i].measurements`. The stand-in's resource function `_angle_erase_resources` passes integer placeholders to `mcm_parities`, so on the tape path the decomposition fails with `AttributeError: 'int' object has no attribute 'measurements'`. Task 0.9's compile path is not affected, because capture mode takes the `jax.numpy` branch. The fix counts nonzero masks instead of calling `mcm_parities` while counting; it is in `phase1-doc-edits.md` as an edit of the Phase 0 spikes plan, for the orchestrator to apply before Task 0.9 starts. With it, the small-shape walk still compiles to Task 0.9's acceptance values (T 236, M 68, A 175; 9.5 s, 602 MB). *Applied:* section 7, Task 1B.3 interfaces (the dependency is stated there).

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
- The development laptop has 10 cores and 16 GB, and the user works on it. Each process tree has a sampled 8 GB memory cap: the watchdog kills it at the first sample above 7.5 GB (`EFFECTIVE_LIMIT_MB`), and projections and budgets must stay within 7.5 GB.
- A heavy process is a run whose process tree exceeds 4 GB, keeps more than 2 cores busy or lasts longer than 30 minutes, or any FeMoco-shaped compile. Only the orchestrator starts heavy processes, one at a time, under the watchdog (`dfthc.testing.watchdog`), and logs each in the ledger.
- During TDD, run only the targeted tests; run the per-commit tier once at commit time. Never start a heavy process yourself.
- The per-commit and nightly tiers contain no FeMoco-shaped compile. FeMoco-54 values are asserted on the closed forms (`dfthc.costs`); the FeMoco-54 walk and controlled walk are compiled once, in Task 4.5.

**Git (spec 15)**
- Local branches and worktrees only. Merges go into local `main`. Never push, never modify remotes. The repo-local git identity (`dwierichs` / `david.wierichs@xanadu.ai`) is shared by all worktrees.
- Each task runs in one of two worktree lanes (`.worktrees/lane-a`, `.worktrees/lane-b`) on a branch named `phase<N>/<task-id>` from local `main`; only the orchestrator merges. Never run `git worktree add/remove` or `git clone`.

---

## File structure

| Path | Task | Responsibility |
| --- | --- | --- |
| `src/dfthc/testing/sparse_sim.py` | 1B.1, replaced in 1B.2 | `SparseState`, `SparseSimulator`, handlers; from 1B.2 also policies, `Branch`, measurements, conditionals |
| `tests/sim/test_sparse_sim.py` | 1B.1 | state, unitary handlers against `default.qubit`, preconditions |
| `src/dfthc/testing/tape.py` | 1B.2 | `decompose_for_simulation` |
| `tests/sim/test_measurements.py` | 1B.2 | measurements and conditionals against deferred `default.qubit`, policies |
| `tests/sim/test_tape.py` | 1B.2 | the tape front end and the K-3 round trip |
| `tests/sim/test_benchmark.py` | 1B.3 | the spec 12.6 benchmark gate (slow) |

**Shell conventions.** Run commands from the worktree root of your lane, with `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"`.

**Expected per-commit counts** follow the rule and the table of the Workstream A plan: 181 at G0, plus the per-commit tests of every Phase 1 task merged before your branch was created (1B.1: 23, 1B.2: 27, 1B.3: 0), plus your own.

---

### Task 1B.1: Simulator core and differential tests

**Task id:** `1B.1` · **Branch:** `phase1/1B.1` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** G0

**Files:**
- Create: `src/dfthc/testing/sparse_sim.py`, `tests/sim/test_sparse_sim.py`

**Interfaces:**
- Consumes: `dfthc.compile.GATE_SET` (Task 0.2); PennyLane's `default.qubit` and `PhaseGradientStatePrep.state_vector()` as references.
- Produces, in `dfthc.testing.sparse_sim`:
  - `PRUNE_THRESHOLD = 1e-14`, `MAX_DENSE_WIRES = 25`, `class UnsupportedOperation(Exception)`.
  - `@dataclass class SparseState(num_wires, keys, amps)`: `keys` is `ndarray[uint64] (terms, ceil(num_wires / 64))` with wire `w` at bit `63 - w % 64` of word `w // 64`, and `amps` is `ndarray[complex128]`. It has:
    - constructors `SparseState.basis(num_wires, value)` (big-endian) and `SparseState.from_dense(vector)`;
    - `to_dense()` (at most 25 wires) and `copy()`;
    - `bits(wire) -> ndarray[bool]` and `flip(wire, where=None)`;
    - `probability(wire, outcome) -> float`, relative to the norm;
    - `prune(threshold=1e-14) -> SparseState` and `key_string(row) -> str`.
  - `class SparseSimulator(num_wires, *, policy=None, strict=True)`, with `apply(op)`, `run(operations, initial_state=None) -> SparseState`, the attribute `state`, and `supported_operations -> frozenset[str]`.
  - One handler per spec 12.3 row for every `GATE_SET` member except `MidMeasure` (Task 1B.2). `TemporaryAND` and `Adjoint(TemporaryAND)` honour `control_values` and assert their preconditions on every term; `PhaseGradientStatePrep` asserts that its register is `|0>`. Each raises `AssertionError` with the offending key as a bit string, wire 0 first. Any other operation raises `UnsupportedOperation`, and wires outside `[0, num_wires)` raise `ValueError`.

**Acceptance tests** (23 tests in `tests/sim/test_sparse_sim.py`, about 3 s):
- Every gate-set member except `MidMeasure` has a handler.
- Each unitary handler matches `default.qubit` on 20 random states of 3 to 10 qubits, to 1e-12, on random wires (11 tests). These are `PauliX`, `PauliY`, `PauliZ`, `Hadamard`, `S`, `Adjoint(S)`, `CNOT`, `CZ`, `SWAP`, `Toffoli` and `GlobalPhase`.
- `TemporaryAND` and `Adjoint(TemporaryAND)` match on their domains, with random `control_values`, and the uncompute restores the input.
- `PhaseGradientStatePrep` matches PennyLane's `state_vector()` on random registers.
- Ten random circuits of 40 operations mix all handlers, including AND compute and uncompute pairs, and match.
- `GlobalPhase(phi)` multiplies by `e^{-i phi}`.
- The three precondition violations raise `AssertionError` naming the key.
- `RX`, `MultiControlledX` and a wire out of range are rejected.
- `SparseState` basics; keys spanning three words at 130 wires.

**Definition of done:** `23 passed` in well under 2 minutes; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/sim/test_sparse_sim.py`:

```python
"""Sparse simulator core: state, unitary handlers, preconditions (spec section 12.3).

Every handler is compared with PennyLane's default.qubit statevector on random states.
"""

import numpy as np
import pennylane as qp
import pytest

from dfthc.compile import GATE_SET
from dfthc.testing.sparse_sim import SparseSimulator, SparseState, UnsupportedOperation

STATES_PER_HANDLER = 20


def _gate(name, wires, rng):
    """One instance of a gate-set operation on ``wires`` (3 wires for every gate)."""
    a, b, c = wires
    return {
        "PauliX": lambda: qp.X(a), "PauliY": lambda: qp.Y(a), "PauliZ": lambda: qp.Z(a),
        "Hadamard": lambda: qp.H(a), "S": lambda: qp.S(a),
        "Adjoint(S)": lambda: qp.adjoint(qp.S(a)), "CNOT": lambda: qp.CNOT([a, b]),
        "CZ": lambda: qp.CZ([a, b]), "SWAP": lambda: qp.SWAP([a, b]),
        "Toffoli": lambda: qp.Toffoli([a, b, c]),
        "GlobalPhase": lambda: qp.GlobalPhase(rng.uniform(-np.pi, np.pi)),
        "TemporaryAND": lambda: qp.TemporaryAND([a, b, c], control_values=rng.integers(0, 2, 2)),
    }[name]()


def _random_state(rng, num_wires):
    state = rng.standard_normal(2**num_wires) + 1j * rng.standard_normal(2**num_wires)
    return state / np.linalg.norm(state)


def _zero_wire(state, num_wires, wire):
    """Project ``wire`` onto |0> and renormalize."""
    tensor = state.reshape((2,) * num_wires).copy()
    np.moveaxis(tensor, wire, 0)[1] = 0
    return tensor.ravel() / np.linalg.norm(tensor)


def _with_phase_gradient(state, num_wires, register):
    """``state`` (|0> on ``register``) with PennyLane's phase-gradient state on ``register``.

    default.qubit cannot apply PhaseGradientStatePrep after other operations (it has no matrix),
    so its reference is the operator's own ``state_vector()``.
    """
    index = tuple(0 if wire in register else slice(None) for wire in range(num_wires))
    rest = state.reshape((2,) * num_wires)[index]
    prep = np.asarray(qp.PhaseGradientStatePrep(register).state_vector())
    full = np.multiply.outer(rest, prep.reshape((2,) * len(register)))
    order = [wire for wire in range(num_wires) if wire not in register] + list(register)
    return full.transpose(np.argsort(order)).ravel()


def _default_qubit(state, operations, num_wires):
    tape = qp.tape.QuantumScript([qp.StatePrep(state, wires=range(num_wires)), *operations],
                                 [qp.state()])
    return np.asarray(qp.devices.DefaultQubit().execute(tape))


def _sparse(state, operations, num_wires):
    sim = SparseSimulator(num_wires)
    return sim.run(operations, initial_state=SparseState.from_dense(state)).to_dense()


UNITARY = ["PauliX", "PauliY", "PauliZ", "Hadamard", "S", "Adjoint(S)", "CNOT", "CZ", "SWAP",
           "Toffoli", "GlobalPhase"]


def test_every_gate_set_member_but_mid_measure_has_a_handler():
    """MidMeasure joins with the measurement policies (Task 1B.2)."""
    assert set(GATE_SET) - {"MidMeasure"} <= SparseSimulator(1).supported_operations


@pytest.mark.parametrize("name", UNITARY)
def test_unitary_handler_matches_default_qubit(name):
    rng = np.random.default_rng(100 + UNITARY.index(name))
    for _ in range(STATES_PER_HANDLER):
        num_wires = int(rng.integers(3, 11))
        state = _random_state(rng, num_wires)
        op = _gate(name, [int(w) for w in rng.choice(num_wires, 3, replace=False)], rng)
        assert np.allclose(_sparse(state, [op], num_wires),
                           _default_qubit(state, [op], num_wires), atol=1e-12)


def test_temporary_and_matches_default_qubit_on_its_domain():
    rng = np.random.default_rng(1)
    for _ in range(STATES_PER_HANDLER):
        num_wires = int(rng.integers(3, 11))
        wires = [int(w) for w in rng.choice(num_wires, 3, replace=False)]
        state = _zero_wire(_random_state(rng, num_wires), num_wires, wires[2])
        op = _gate("TemporaryAND", wires, rng)
        assert np.allclose(_sparse(state, [op], num_wires),
                           _default_qubit(state, [op], num_wires), atol=1e-12)


def test_adjoint_temporary_and_matches_default_qubit_on_its_domain():
    rng = np.random.default_rng(2)
    for _ in range(STATES_PER_HANDLER):
        num_wires = int(rng.integers(3, 11))
        wires = [int(w) for w in rng.choice(num_wires, 3, replace=False)]
        values = rng.integers(0, 2, 2)
        before = _zero_wire(_random_state(rng, num_wires), num_wires, wires[2])
        state = _default_qubit(before, [qp.TemporaryAND(wires, control_values=values)],
                               num_wires)
        op = qp.adjoint(qp.TemporaryAND(wires, control_values=values))
        got = _sparse(state, [op], num_wires)
        assert np.allclose(got, _default_qubit(state, [op], num_wires), atol=1e-12)
        assert np.allclose(got, before, atol=1e-12)


def test_phase_gradient_state_prep_matches_pennylane_on_its_domain():
    rng = np.random.default_rng(3)
    for _ in range(STATES_PER_HANDLER):
        num_wires = int(rng.integers(3, 11))
        size = int(rng.integers(1, num_wires))
        register = [int(w) for w in rng.choice(num_wires, size, replace=False)]
        state = _random_state(rng, num_wires)
        for wire in register:
            state = _zero_wire(state, num_wires, wire)
        op = qp.PhaseGradientStatePrep(register)
        assert np.allclose(_sparse(state, [op], num_wires),
                           _with_phase_gradient(state, num_wires, register), atol=1e-12)


def test_random_circuits_mixing_all_handlers_match():
    """Random gates on wires 0..5, a phase gradient on wires 6-7 and AND compute/uncompute pairs
    on ancilla wire 8, which every TemporaryAND finds in |0> and every adjoint leaves in |0>."""
    rng = np.random.default_rng(4)
    num_wires, data, ancilla = 9, list(range(8)), 8
    for _ in range(10):
        state = _random_state(rng, num_wires)
        for wire in (6, 7, ancilla):
            state = _zero_wire(state, num_wires, wire)
        operations = []
        for _ in range(40):
            if rng.random() < 0.15:
                a, b, c = (int(w) for w in rng.choice(data, 3, replace=False))
                values = rng.integers(0, 2, 2)
                operations += [qp.TemporaryAND([a, b, ancilla], control_values=values),
                               qp.CZ([ancilla, c]),
                               qp.adjoint(qp.TemporaryAND([a, b, ancilla], control_values=values))]
            else:
                name = UNITARY[int(rng.integers(len(UNITARY)))]
                operations.append(_gate(name, [int(w) for w in rng.choice(data, 3, replace=False)],
                                        rng))
        prep = qp.PhaseGradientStatePrep([6, 7])
        assert np.allclose(_sparse(state, [prep, *operations], num_wires),
                           _default_qubit(_with_phase_gradient(state, num_wires, [6, 7]),
                                          operations, num_wires), atol=1e-12)


def test_global_phase_multiplies_by_exp_minus_i_phi():
    state = SparseSimulator(2).run([qp.H(0), qp.GlobalPhase(0.7)])
    assert np.allclose(state.to_dense(), np.exp(-0.7j) * np.array([1, 0, 1, 0]) / np.sqrt(2))


def test_temporary_and_asserts_a_zero_target():
    sim = SparseSimulator(3)
    with pytest.raises(AssertionError, match="target is 1 on key 001"):
        sim.run([qp.X(2), qp.TemporaryAND([0, 1, 2])])


def test_adjoint_temporary_and_asserts_the_target_equals_the_and():
    sim = SparseSimulator(3)
    with pytest.raises(AssertionError, match="target differs from the AND on key 110"):
        sim.run([qp.X(0), qp.X(1), qp.adjoint(qp.TemporaryAND([0, 1, 2]))])
    with pytest.raises(AssertionError, match="on key 011"):
        sim.run([qp.X(1), qp.X(2), qp.adjoint(qp.TemporaryAND([0, 1, 2], control_values=[1, 1]))])


def test_phase_gradient_state_prep_asserts_a_zero_register():
    with pytest.raises(AssertionError, match="register is not |0> on key 0010"):
        SparseSimulator(4).run([qp.X(2), qp.PhaseGradientStatePrep([1, 2, 3])])


def test_other_operations_are_rejected():
    with pytest.raises(UnsupportedOperation, match="RX"):
        SparseSimulator(2).run([qp.RX(0.1, 0)])
    with pytest.raises(UnsupportedOperation, match="MultiControlledX"):
        SparseSimulator(4).run([qp.MultiControlledX(wires=[0, 1, 2, 3])])
    with pytest.raises(ValueError, match="wire 5"):
        SparseSimulator(2).run([qp.X(5)])


def test_sparse_state_basics():
    state = SparseState.basis(5, 0b10010)
    assert np.flatnonzero(state.to_dense()).tolist() == [0b10010]
    assert state.probability(0, 1) == 1.0 and state.probability(1, 1) == 0.0
    dense = _random_state(np.random.default_rng(5), 4)
    dense[3] = 1e-16
    round_trip = SparseState.from_dense(dense)
    assert len(round_trip.amps) == 15  # the tiny amplitude is pruned
    assert np.allclose(round_trip.to_dense(), np.where(np.abs(dense) < 1e-14, 0, dense))
    weights = np.abs(round_trip.to_dense()) ** 2
    wire_2_set = [2, 3, 6, 7, 10, 11, 14, 15]
    assert np.isclose(round_trip.probability(2, 1), weights[wire_2_set].sum() / weights.sum())
    with pytest.raises(ValueError, match="25 wires"):
        SparseState.basis(26, 0).to_dense()


def test_keys_span_several_words():
    """130 wires use three uint64 words; wire 0 stays the most significant bit."""
    sim = SparseSimulator(130)
    state = sim.run([qp.X(100), qp.H(129), qp.CNOT([129, 3]), qp.SWAP([100, 64])])
    assert state.keys.shape == (2, 3)
    assert np.isclose(state.probability(64, 1), 1.0) and state.probability(100, 1) == 0.0
    assert np.isclose(state.probability(3, 1), 0.5) and np.isclose(state.probability(129, 1), 0.5)
    state.amps[0] = 1e-15
    assert len(state.prune().amps) == 1
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/sim/test_sparse_sim.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.testing.sparse_sim'`.

- [ ] **Step 3: Write the simulator core**

Create `src/dfthc/testing/sparse_sim.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/sim/test_sparse_sim.py -q`
Expected: `23 passed` (about 3 s).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `204 passed` (181 plus 23).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/testing/sparse_sim.py tests/sim/test_sparse_sim.py
git commit -m "Add the sparse-state simulator core with differential tests"
```

---

### Task 1B.2: Measurements, conditionals, policies and the tape front end

**Task id:** `1B.2` · **Branch:** `phase1/1B.2` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 1B.1

**Files:**
- Modify: `src/dfthc/testing/sparse_sim.py` (replace it with the version below; everything of Task 1B.1 is kept)
- Create: `src/dfthc/testing/tape.py`, `tests/sim/test_measurements.py`, `tests/sim/test_tape.py`

**Interfaces:**
- Consumes:
  - Task 1B.1's simulator;
  - `dfthc.compile.GATE_SET` and `FIXED_DECOMPS` (Task 0.2);
  - `pennylane.transforms._tape_decompose` and `pennylane.ops.Conditional` and `MidMeasure`;
  - `spikes/k3_mode_agnostic_erase.py` (`round_trip`, `HIGH`, `LOW`, `NUM_WIRES`; Task 0.7), imported in the test through `sys.path`.
- Produces, in `dfthc.testing.sparse_sim`:
  - `MEASUREMENT_TOLERANCE = 1e-12`.
  - Policies, each with `choose(op, probability_one) -> int`:
    - `ForcedPolicy(schedule: Sequence[int])`, with the attribute `position`; it raises `ValueError` once the schedule is exhausted;
    - `SeededPolicy(seed: int)`;
    - `BranchPolicy(wires: Collection, *, otherwise=None)`, with `branches(op) -> bool`.
  - `@dataclass class Branch(record: dict, state: SparseState, probability: float)`.
  - `SparseSimulator.run_branches(operations, initial_state=None) -> list[Branch]`, `SparseSimulator.record: dict` (each `MidMeasure` mapped to its outcome), and `supported_operations`, which now includes `MidMeasure` and `Conditional`.
  - `run_branches` follows both outcomes of each measurement on a branch wire; each branch consumes its own copy of the policy. A postselected measurement drops the branches whose success probability is below `MEASUREMENT_TOLERANCE` and weights the others by it, so the returned probabilities are conditional on every postselection succeeding (`p_b q_b / sum p_b' q_b'`, as `default.qubit` with deferred measurements); `ValueError` only if no branch passes. `run` refuses a `BranchPolicy`.
  - The `MidMeasure` handler:
    - the outcome comes from `postselect` if set, else from the policy;
    - strict mode asserts `|P(1) - 1/2| <= 1e-12` for measurements that are neither on branch wires nor postselected: it certifies uniformly random erase outcomes, and a postselection is a test device;
    - it projects, normalizes by the kept weight, records the outcome, and resets the wire if `reset=True` and the outcome is 1;
    - in `run`, an outcome of probability below 1e-12 raises `ValueError`.
  - The `Conditional` handler: `op.meas_val.concretize(record)`; if true, it dispatches `op.base` recursively.
  - `dfthc.testing.tape.decompose_for_simulation(qfunc, *args, **kwargs) -> QuantumScript`. Capture is paused and the graph enabled, then both are restored. It runs `qp.transforms._tape_decompose(tape, gate_set=GATE_SET, fixed_decomps=FIXED_DECOMPS)`.

**Acceptance tests** (27 tests):
- `tests/sim/test_measurements.py` (19):
  - `MidMeasure` and `Conditional` are supported.
  - Eight random circuits with measurements (some with `reset`) and conditionals on `^`, sums mod 2 and single outcomes: the mixture of all branches under a `BranchPolicy` on every wire equals the reduced density matrix of the deferred circuit on `default.qubit` (analytic, 1e-12).
  - `reset` and `postselect` match the deferred, postselected density matrix.
  - Postselection after branching: wire 1 = (w0 ∧ w2) ∨ (w3 ∧ w4) passes the postselection with probability 0, 1/2, 1/2 and 3/4 in the branches of w0 and w3. Branch 00 is dropped, and the others carry 2/7, 2/7 and 3/7, equal to `default.qubit`'s `qp.probs(op=[m0, m3])` and its density matrix under the same postselection. A postselection that no branch passes raises.
  - Postselected and forced outcomes of probability 0 raise.
  - Forced schedules are honoured, recorded and exhausted with an error.
  - Strict mode rejects a biased measurement and exempts branch and postselected ones.
  - Measurements renormalize instead of compounding norm errors.
  - Seeded policies are reproducible and follow the probabilities.
  - `BranchPolicy` delegates other wires, and each branch restarts the schedule.
  - Nested conditionals dispatch recursively.
- `tests/sim/test_tape.py` (8):
  - the decomposition reaches the gate set;
  - arguments are forwarded and measurements kept;
  - the graph flag and capture are restored;
  - the K-3 round trip restores its state exactly under the zero schedule and all four unit schedules, in strict mode (5 tests).

**Definition of done:** `scripts/test.sh tests/sim -q` gives `50 passed` (Task 1B.1's 23 and these 27); the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/sim/test_measurements.py`:

```python
"""Mid-circuit measurements, conditionals and measurement policies (spec section 12.3).

Branch-resolved simulations are compared with default.qubit under deferred measurements
(analytic): the mixture of all branches equals the reduced density matrix of the deferred circuit.
"""

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional

from dfthc.testing.sparse_sim import (
    Branch,
    BranchPolicy,
    ForcedPolicy,
    SeededPolicy,
    SparseSimulator,
    SparseState,
)


def _tape(build):
    with qp.queuing.AnnotatedQueue() as queue:
        build()
    return qp.tape.QuantumScript.from_queue(queue)


def _deferred_density_matrix(operations, num_wires):
    tape = qp.tape.QuantumScript(operations, [qp.density_matrix(wires=range(num_wires))])
    [deferred], _ = qp.defer_measurements(tape)
    return np.asarray(qp.devices.DefaultQubit().execute(deferred))


def _mixture(branches):
    return sum(b.probability * np.outer(v, v.conj())
               for b in branches for v in [b.state.to_dense()])


def _random_circuit(rng, num_wires, num_ops):
    """Gate-set operations, measurements (some with reset) and conditionals on composite
    expressions of earlier outcomes: XOR, sums mod 2 and a single outcome."""
    gates = [lambda w: qp.H(w[0]), lambda w: qp.S(w[0]), lambda w: qp.CNOT(w[:2]),
             lambda w: qp.Toffoli(w), lambda w: qp.Y(w[0]), lambda w: qp.CZ(w[:2])]
    outcomes = []

    def build():
        for wire in range(num_wires):
            qp.H(wire)
        for _ in range(num_ops):
            wires = [int(w) for w in rng.choice(num_wires, 3, replace=False)]
            choice = rng.random()
            if choice < 0.2:
                outcomes.append(qp.measure(wires[0], reset=bool(rng.integers(2))))
            elif choice < 0.4 and len(outcomes) >= 2:
                a, b = (outcomes[int(i)] for i in rng.choice(len(outcomes), 2, replace=False))
                expression = [a ^ b, (a + b) % 2, a][int(rng.integers(3))]
                qp.cond(expression, [qp.X, qp.Z][int(rng.integers(2))])(wires[0])
            else:
                gates[int(rng.integers(len(gates)))](wires)

    return _tape(build).operations


def test_mid_measure_and_conditional_are_supported():
    assert {"MidMeasure", "Conditional"} <= SparseSimulator(1).supported_operations


@pytest.mark.parametrize("seed", range(8))
def test_branches_match_default_qubit_with_deferred_measurements(seed):
    rng = np.random.default_rng(seed)
    num_wires = 5
    operations = _random_circuit(rng, num_wires, 30)
    sim = SparseSimulator(num_wires, policy=BranchPolicy(range(num_wires)), strict=False)
    branches = sim.run_branches(operations)
    assert sum(b.probability for b in branches) == pytest.approx(1.0, abs=1e-12)
    assert all(isinstance(b, Branch) for b in branches)
    assert np.allclose(_mixture(branches), _deferred_density_matrix(operations, num_wires),
                       atol=1e-12)


def test_reset_and_postselect_match_default_qubit():
    for postselect in (0, 1):
        def build(postselect=postselect):
            qp.H(0)
            qp.CNOT([0, 1])
            qp.S(1)
            m = qp.measure(0, reset=True, postselect=postselect)
            m2 = qp.measure(1, reset=False)
            qp.cond(m ^ m2, qp.X)(2)
            qp.cond((m + m2) % 2, qp.H)(3)

        operations = _tape(build).operations
        sim = SparseSimulator(4, policy=BranchPolicy([1]), strict=False)
        branches = sim.run_branches(operations)
        assert len(branches) == 1 and branches[0].record[operations[3]] == postselect
        assert branches[0].state.probability(0, 1) == 0.0  # reset
        assert np.allclose(_mixture(branches), _deferred_density_matrix(operations, 4),
                           atol=1e-12)


def test_postselection_after_branching_conditions_the_branches():
    """Wire 1 = (w0 and w2) or (w3 and w4) passes the postselection with probability 0, 1/2, 1/2
    and 3/4 in the branches (w0, w3) = 00, 01, 10, 11: branch 00 is dropped, and the others carry
    p q / sum(p q) = 2/7, 2/7, 3/7, as default.qubit gives under the same postselection."""
    outcomes = {}

    def build():
        for wire in (0, 2, 3, 4):
            qp.H(wire)
        qp.Toffoli([0, 2, 5])
        qp.Toffoli([3, 4, 6])
        qp.CNOT([5, 1])
        qp.CNOT([6, 1])
        qp.Toffoli([5, 6, 1])
        outcomes["w0"], outcomes["w3"] = qp.measure(0), qp.measure(3)
        qp.measure(1, postselect=1)

    operations = _tape(build).operations
    branches = SparseSimulator(7, policy=BranchPolicy([0, 3]), strict=False).run_branches(
        operations)
    measured = [op for op in operations if op.name == "MidMeasure"][:2]
    probabilities = np.zeros(4)
    for branch in branches:
        probabilities[2 * branch.record[measured[0]] + branch.record[measured[1]]] = (
            branch.probability)
    reference = qp.tape.QuantumScript(operations, [qp.probs(op=[outcomes["w0"], outcomes["w3"]]),
                                                   qp.density_matrix(wires=range(7))])
    [deferred], _ = qp.defer_measurements(reference)
    probs, rho = qp.devices.DefaultQubit().execute(deferred)
    assert np.allclose(probabilities, [0, 2 / 7, 2 / 7, 3 / 7], atol=1e-12)
    assert np.allclose(probabilities, probs, atol=1e-12)
    assert np.allclose(_mixture(branches), rho, atol=1e-12)
    impossible = _tape(lambda: (qp.H(0), qp.measure(0), qp.measure(1, postselect=1))).operations
    with pytest.raises(ValueError, match="no branch can pass the postselection"):
        SparseSimulator(2, policy=BranchPolicy([0]), strict=False).run_branches(impossible)


def test_a_postselected_outcome_of_probability_zero_raises():
    operations = _tape(lambda: qp.measure(0, postselect=1)).operations
    with pytest.raises(ValueError, match="probability 0"):
        SparseSimulator(1, policy=ForcedPolicy([0]), strict=False).run(operations)


def test_forced_schedule_is_honoured_and_recorded():
    def build():
        for wire in range(3):
            qp.H(wire)
            qp.measure(wire, reset=True)

    operations = _tape(build).operations
    policy = ForcedPolicy([1, 0, 1])
    sim = SparseSimulator(3, policy=policy)
    state = sim.run(operations)
    measurements = [op for op in operations if op.name == "MidMeasure"]
    assert [sim.record[m] for m in measurements] == [1, 0, 1]
    assert np.allclose(state.to_dense(), SparseState.basis(3, 0).to_dense())  # all reset
    assert policy.position == 3
    with pytest.raises(ValueError, match="schedule of 3 outcomes is exhausted"):
        sim.run(operations + operations[:2])
    with pytest.raises(ValueError, match="0 or 1"):
        ForcedPolicy([2])


def test_a_forced_outcome_of_probability_zero_raises():
    operations = _tape(lambda: qp.measure(0)).operations
    with pytest.raises(ValueError, match="probability 0"):
        SparseSimulator(1, policy=ForcedPolicy([1]), strict=False).run(operations)


def test_strict_mode_asserts_uniform_erasure_outcomes():
    unbalanced = _tape(lambda: (qp.H(0), qp.S(0), qp.CNOT([0, 1]), qp.measure(1))).operations
    assert SparseSimulator(2, policy=ForcedPolicy([0])).run(unbalanced) is not None
    biased = _tape(lambda: qp.measure(0)).operations
    with pytest.raises(AssertionError, match="P\\(1\\) = 0.0"):
        SparseSimulator(1, policy=ForcedPolicy([0])).run(biased)
    # Branch wires and postselected measurements are exempt.
    assert len(SparseSimulator(1, policy=BranchPolicy([0])).run_branches(biased)) == 1
    postselected = _tape(lambda: qp.measure(0, postselect=0)).operations
    assert SparseSimulator(1, policy=ForcedPolicy([])).run(postselected) is not None


def test_measurements_renormalize_instead_of_compounding_norm_errors():
    """Probabilities are relative to the norm, and each measurement restores norm 1. Dividing by
    1 - P(1) instead doubles a norm error at every measurement (found on the K-1 stand-in walk,
    whose 50 erase measurements then drove the norm down to 0.13)."""
    def build():
        for wire in range(20):
            qp.H(wire)
            qp.measure(wire, reset=True)

    start = SparseState.basis(20, 0)
    start.amps *= np.sqrt(1 - 1e-6)
    state = SparseSimulator(20, policy=ForcedPolicy([0, 1] * 10)).run(_tape(build).operations,
                                                                     initial_state=start)
    assert np.sum(np.abs(state.amps) ** 2) == pytest.approx(1.0, abs=1e-14)


def test_seeded_policy_is_reproducible_and_follows_the_probabilities():
    def build():
        qp.X(0)
        for wire in range(1, 9):
            qp.H(wire)
            qp.measure(wire)
        qp.measure(0)

    operations = _tape(build).operations
    runs = []
    for _ in range(2):
        sim = SparseSimulator(9, policy=SeededPolicy(7), strict=False)
        sim.run(operations)
        runs.append([sim.record[op] for op in operations if op.name == "MidMeasure"])
    assert runs[0] == runs[1] and runs[0][-1] == 1 and 0 < sum(runs[0][:-1]) < 8


def test_branch_policy_delegates_other_wires_and_each_branch_restarts_the_schedule():
    operations = _tape(lambda: (qp.H(0), qp.H(1), qp.measure(0), qp.measure(1))).operations
    policy = BranchPolicy([0], otherwise=ForcedPolicy([1]))
    branches = SparseSimulator(2, policy=policy).run_branches(operations)
    assert sorted(b.record[operations[2]] for b in branches) == [0, 1]
    assert all(b.record[operations[3]] == 1 for b in branches)
    assert [b.probability for b in branches] == pytest.approx([0.5, 0.5])
    assert policy.otherwise.position == 0  # the branches consumed copies
    with pytest.raises(ValueError, match="run_branches"):
        SparseSimulator(2, policy=policy).run(operations)
    with pytest.raises(ValueError, match="no policy"):
        SparseSimulator(2, policy=BranchPolicy([0])).run_branches(operations)
    with pytest.raises(ValueError, match="no policy"):
        SparseSimulator(2).run(operations)


def test_nested_conditionals_dispatch_recursively():
    def build():
        qp.X(0)
        qp.X(1)
        a, b = qp.measure(0), qp.measure(1)
        Conditional(a, Conditional(b, qp.X(2)))
        Conditional(a, Conditional(b ^ a, qp.X(3)))

    operations = _tape(build).operations
    state = SparseSimulator(4, policy=ForcedPolicy([1, 1]), strict=False).run(operations)
    assert np.flatnonzero(state.to_dense()).tolist() == [0b1110]
```

Create `tests/sim/test_tape.py`:

```python
"""Tape-path decomposition for simulation, and the K-3 round trip on the simulator (spec 12.3)."""

import sys
from pathlib import Path

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional, MidMeasure

from dfthc.compile import GATE_SET
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator
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
    graph_was_enabled = qp.decomposition.enabled_graph()
    qp.decomposition.disable_graph()
    try:
        decompose_for_simulation(round_trip)
        assert not qp.decomposition.enabled_graph()
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
    """Zero schedule and the four unit schedules (spec 12.5 basis runs), strict mode on."""
    schedule = [int(i == unit) for i in range(NUM_MEASUREMENTS)]
    policy = ForcedPolicy(schedule)
    state = SparseSimulator(NUM_WIRES, policy=policy).run(
        decompose_for_simulation(round_trip).operations)
    expected = np.zeros(2**NUM_WIRES, dtype=complex)
    address_bits = len(HIGH + LOW)
    for address in range(2**address_bits):
        expected[address << (NUM_WIRES - address_bits)] = 1 / np.sqrt(2**address_bits)
    assert policy.position == NUM_MEASUREMENTS
    assert np.allclose(state.to_dense(), expected, atol=1e-12)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/sim/test_measurements.py tests/sim/test_tape.py -q`
Expected: FAIL at collection, `2 errors`: `ImportError: cannot import name 'ForcedPolicy' from 'dfthc.testing.sparse_sim'` (and `Branch` in the other module).

- [ ] **Step 3: Replace the simulator with the version that measures**

Replace `src/dfthc/testing/sparse_sim.py` with:

```python
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
"""

from __future__ import annotations

import copy
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

    :meth:`run` starts from ``initial_state`` (default ``|0...0>``), applies the operations in
    order and returns the final state; ``record`` then maps every ``MidMeasure`` to its outcome.
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
        return (SparseState.basis(self.num_wires, 0) if initial_state is None
                else initial_state.copy())

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

        Each branch consumes its own copy of the policy, so a forced schedule for the other
        measurements restarts in every branch. A postselected measurement drops the branches
        that cannot pass it and weights the others by their success probability, so the returned
        probabilities are conditional on every postselection succeeding, as with ``default.qubit``
        and deferred measurements. Raises ``ValueError`` if no branch passes a postselection.
        """
        policy = self.policy
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
            self.policy = policy
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
            if self.strict and abs(probability_one - 0.5) > MEASUREMENT_TOLERANCE:
                raise AssertionError(f"strict mode: {op} has P(1) = {probability_one}, not 1/2 "
                                     "(spec 12.3)")
            outcome = self.policy.choose(op, probability_one)
        self._project(op, wire, outcome)

    def _project(self, op, wire: int, outcome: int) -> float:
        """Keep the terms with ``wire == outcome``, normalize them, record the outcome and reset
        if asked; return the outcome's probability. Normalizing by the kept weight itself, not by
        ``1 - P(1)``, keeps rounding errors from compounding over many measurements."""
        state = self.state
        keep = state.bits(wire) == bool(outcome)
        weights = np.abs(state.amps) ** 2
        kept = weights[keep].sum()
        probability = float(kept / weights.sum())
        if probability < MEASUREMENT_TOLERANCE:
            raise ValueError(f"{op} cannot give outcome {outcome}: it has probability "
                             f"{probability:.3g}")
        state.keys, state.amps = state.keys[keep], state.amps[keep] / np.sqrt(kept)
        if op.reset and outcome:
            state.flip(wire)
        self.record[op] = outcome
        return probability
```

- [ ] **Step 4: Write the tape front end**

Create `src/dfthc/testing/tape.py`:

```python
"""Tape-path decomposition of a quantum function for the sparse simulator (spec section 12.3)."""

from __future__ import annotations

from collections.abc import Callable

import pennylane as qp
from pennylane.transforms import _tape_decompose

from dfthc.compile import FIXED_DECOMPS, GATE_SET


def decompose_for_simulation(qfunc: Callable, *args, **kwargs) -> qp.tape.QuantumScript:
    """Record ``qfunc(*args, **kwargs)`` as a tape and decompose it to the gate set.

    Capture is paused while recording, and the graph decomposition system is enabled for
    ``qp.transforms._tape_decompose(tape, gate_set=GATE_SET, fixed_decomps=FIXED_DECOMPS)``;
    both are restored afterwards. The result holds gate-set operations, ``MidMeasure`` and
    ``Conditional`` operations with ``MeasurementValue`` expressions, plus the measurements
    ``qfunc`` queued. Our operators decompose with their pinned rules; PennyLane primitives may
    use rules the compiled path does not, which leaves the semantics unchanged (spec 12.3).
    """
    graph_was_enabled = qp.decomposition.enabled_graph()
    qp.decomposition.enable_graph()
    try:
        with qp.capture.pause():
            with qp.queuing.AnnotatedQueue() as queue:
                qfunc(*args, **kwargs)
            tape = qp.tape.QuantumScript.from_queue(queue)
            [decomposed], _ = _tape_decompose(tape, gate_set=GATE_SET,
                                              fixed_decomps=dict(FIXED_DECOMPS) or None)
    finally:
        if not graph_was_enabled:
            qp.decomposition.disable_graph()
    return decomposed
```

- [ ] **Step 5: Run the tests**

Run: `scripts/test.sh tests/sim -q`
Expected: `50 passed`.

- [ ] **Step 6: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `303 passed` (wave 1-a merged: 181 + 41 + 31 + 23, plus 27).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/dfthc/testing/sparse_sim.py src/dfthc/testing/tape.py \
        tests/sim/test_measurements.py tests/sim/test_tape.py
git commit -m "Add measurements, conditionals, policies and the tape front end"
```

---

### Task 1B.3: Simulator benchmark gate

**Task id:** `1B.3` · **Branch:** `phase1/1B.3` · **Tier:** B (`claude-sonnet-5-5-high`). It runs a defined benchmark and records measurements; if a spec 12.6 mitigation is needed, the orchestrator re-dispatches the task to Tier A with an Opus review. · **Reviewers:** GPT-5.6 · **Depends on:** 1B.2

**Files:**
- Create: `tests/sim/test_benchmark.py` (marked `slow` and `subprocess`)
- The orchestrator writes the `docs/agents/LEDGER.md` row from the report.

**Interfaces:**
- Consumes:
  - `spikes/k1_whole_program.py` (Task 0.9): `Shape`, `Layout`, `random_tables`, `ProtoWalk`, with the counting fix of change 6 (without it, the tape decomposition of the stand-in fails against the merged Task 0.2);
  - `decompose_for_simulation`, `SparseSimulator`, `SparseState`, `ForcedPolicy`;
  - `dfthc.testing.watchdog.run_watched` and `effective_limit_mb` (Task 0.2).
- Produces:
  - `tests/sim/test_benchmark.py::test_benchmark_gate[h2_sto3g|h2_631g]`, which prints one `BENCHMARK {"watchdog": ..., "result": ...}` line per shape;
  - the helpers `simulate(shape_name, inputs) -> dict` and `run_benchmark(shape_name, inputs, timeout_s) -> (WatchedRun, dict | None)`.
  - The stand-in is built at the `H2_STO3G_TEST` shape (`Shape(N=2, R=2, B=2, C=1, n_plus=1, b_rot=4, b_k1=2, b_k2=2, k1=1, k2=1)`) and the `H2_631G_TEST` shape (`N=4, C=2, n_plus=3`, the rest equal). It is decomposed once with `decompose_for_simulation` and simulated in strict mode with `ForcedPolicy` (every erase outcome 0).

**Acceptance tests (spec 12.6 benchmark gate):**
- One full E-1-like schedule, all 16 basis inputs of `psi_up (x) psi_dn` at the `H2_STO3G_TEST` shape, finishes within 20 minutes and below 3.75 GB, so it is not a heavy process.
- One `H2_631G_TEST` input (`|1000>|1000>`) finishes within 10 minutes and below 3.75 GB.
- Each run goes through `run_watched` with a 4 GB cap, which kills it at 3.75 GB and records its peak memory. A run passes only with `killed is None`, no `survivors` and status 0.

**Definition of done:** both cases pass. The two `BENCHMARK` lines, with wall time, peak memory, operation and measurement counts and per-input seconds, are in the report; the orchestrator copies them into the ledger. If a case is over budget, apply the spec 12.6 mitigations in order: batch the inputs through a reference register, vectorize the hot handlers, use a `k1 = k2 = 0` preset for the per-commit and nightly tiers. If it is still over budget, escalate (master plan section 12.6). The stand-in has the walk's structure and sizes, so it is the Phase 1 proxy; Task 3.5 measures again on the real walk.

- [ ] **Step 1: Write the benchmark**

Create `tests/sim/test_benchmark.py`:

```python
"""Simulator benchmark gate (spec 12.6): the K-1 stand-in walk on the tape path and the sparse
simulator, at the shapes of the two test presets, each run under the watchdog with a 4 GB cap.

The stand-in (``spikes/k1_whole_program.py``, Task 0.9) has the walk's structure and sizes, so it
is the Phase 1 proxy; Task 3.5 measures again on the real walk. Every erase measurement is forced
to 0 and strict mode checks the uniform outcomes. ``python tests/sim/test_benchmark.py SHAPE
INPUT...`` runs one measurement and prints its JSON result as the last line.
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
INPUTS = {"h2_sto3g": tuple(range(16)), "h2_631g": (0b1000_1000,)}
"""Basis inputs of psi_up (x) psi_dn, psi_up's bits first: all 16 at H2_STO3G_TEST (one E-1
schedule), and one electron per spin in orbital 0 at H2_631G_TEST."""
BUDGET_S = {"h2_sto3g": 20 * 60, "h2_631g": 10 * 60}
CAP_MB = 4096


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
    system = layout.psi_up + layout.psi_dn
    seconds, terms = [], []
    for value in inputs:
        index = sum(((value >> (len(system) - 1 - k)) & 1) << (layout.num_wires - 1 - wire)
                    for k, wire in enumerate(system))
        sim = SparseSimulator(layout.num_wires, policy=ForcedPolicy([0] * measurements))
        start = time.perf_counter()
        state = sim.run(operations, initial_state=SparseState.basis(layout.num_wires, index))
        seconds.append(round(time.perf_counter() - start, 2))
        terms.append(len(state.amps))
    return {"shape": shape_name, "num_wires": layout.num_wires, "operations": len(operations),
            "measurements": measurements, "decompose_s": round(decompose_s, 2),
            "seconds": seconds, "final_terms": terms}


def run_benchmark(shape_name: str, inputs, timeout_s: float):
    """One measurement in a subprocess under ``run_watched`` (cap 4 GB, killed at 3.75 GB)."""
    command = [sys.executable, str(Path(__file__).resolve()), shape_name,
               *(str(value) for value in inputs)]
    env = os.environ | {"PYTHONPATH": str(REPO / "src"),
                        "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "0"}
    run = run_watched(command, cap_mb=CAP_MB, timeout_s=timeout_s, cwd=REPO, env=env)
    result = json.loads(run.stdout.strip().splitlines()[-1]) if run.returncode == 0 else None
    return run, result


@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_benchmark_gate(shape_name):
    run, result = run_benchmark(shape_name, INPUTS[shape_name], BUDGET_S[shape_name])
    print("BENCHMARK " + json.dumps({"watchdog": run.summary(), "result": result}))
    assert run.killed is None and not run.survivors and run.returncode == 0, run.stderr[-2000:]
    assert run.seconds <= BUDGET_S[shape_name]
    assert run.peak_mb <= effective_limit_mb(CAP_MB)
    assert len(result["seconds"]) == len(INPUTS[shape_name])


if __name__ == "__main__":
    print(json.dumps(simulate(sys.argv[1], [int(value) for value in sys.argv[2:]])))
```

- [ ] **Step 2: Confirm that the per-commit tier deselects it**

Run: `DFTHC_TEST_WORKERS=0 scripts/test.sh -m "not slow and not release" tests/sim -q`
Expected: `50 passed, 2 deselected`.

- [ ] **Step 3: Run the gate**

This run is not heavy: one subprocess at a time, one busy core, under 200 MB, about 2 minutes. Run it alone, targeted, without other test runs of yours in parallel:

Run: `DFTHC_TEST_WORKERS=0 scripts/test.sh -m "slow and not release" tests/sim/test_benchmark.py -q -s`
Expected: `2 passed`, each preceded by a `BENCHMARK` line with `"killed": null` and `"survivors": []`. The dry-run was at reduced scale (see "Code provenance"), against the merged watchdog, on the development laptop with two implementers running:
- `h2_sto3g`: 66 wires, 1,108 operations, 50 measurements, decomposition 0.2 s. Simulation took 4.8 to 5.1 s per input; 2 of the 16 inputs took 11.9 s in total with a 173 MB peak, which projects the full schedule to about 1.5 minutes. The 16-input run itself has not been executed.
- `h2_631g`: 82 wires, 2,123 operations, 68 measurements. The single input took 34.5 s with a 181 MB peak; earlier runs took 27 to 47 s, depending on the machine's load.

- [ ] **Step 4: Check the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: no failures; the count is unchanged by this task (`333 passed` with waves 1-a and 1-b merged).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 5: Commit and report the measurements**

```bash
git add tests/sim/test_benchmark.py
git commit -m "Add the simulator benchmark gate"
```

Put both `BENCHMARK` lines in the report's "Measurements" section.
