# DFTHC Phase 1, Workstream A: Classical Data and the Encoded Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build layer L0 of spec section 5.1: the walk parameters and register layout, the DFTHC factors and dense fermion matrices, the sum-of-squares quantities in both offset frames, the alias and Givens-angle tables, the encoded-Hamiltonian model that is the oracle of the end-to-end checks, the dfthcbe reference fixtures, and the closed-form costs `dfthc.costs` that every later resource test asserts.

**Architecture:** Pure NumPy and SciPy modules under `src/dfthc/` (`params.py`, `registers.py`, `costs.py`) and `src/dfthc/data/` (`factors`, `fermion`, `sos`, `alias`, `angles`, `tables`, `encoded`), plus the fixture loader `src/dfthc/testing/fixtures.py`. Every function that the spec calls a port of dfthcbe is ported line by line from dfthcbe at commit `930255020cec30172e0a7d17c0071d9476a687d1`. While writing this plan the ports were checked against dfthcbe itself: the alias and angle tables are bit-identical at both test presets, `MID_NAPHTHALENE` and FeMoco-54 shape; the encoded factors, failure signs and failure probabilities agree exactly; and `encoded_block` matches dfthcbe's exactly contracted block encoding to 1e-10 (Task 1A.6).

**Tech Stack:** Python 3.12, NumPy, SciPy, PennyLane `Wires`; pytest with pytest-xdist; ruff. Task 1A.6 also uses `.venv-ref` (Qualtran 0.7.0 and dfthcbe).

This file holds Tasks 1A.1 to 1A.7 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 7). Workstream B (Tasks 1B.1 to 1B.3) is in `2026-11-11-dfthc-phase1-simulator.md`.

**Code provenance.** Every code block below was executed on 2026-11-12 in a scratch replica of the repository (`.scratch/plan-check/repo-phase1`, not committed), last resynced to local `main` at `3534c30`. That replica holds Tasks 0.1, 0.2 and 0.5 as merged, including Task 0.1's two extra tests and Task 0.2's 13 post-review changes, plus the spike modules `spikes/prototypes.py` (Task 0.7) and `spikes/k1_whole_program.py` (Task 0.9) from the Phase 0 plan's dry-run. The replica is not a git repository, so the dry-run ran pytest the way `scripts/test.sh` does: same environment, at most 2 workers, `taskpolicy -c utility`. The steps below use `scripts/test.sh` as usual. Task 1A.6's fixtures were generated in a scratch environment built by the recipe of Task 1A.6, Step 7, from the pinned dfthcbe tarball in `.scratch/dfthcbe-ref`. Revision 1 (2026-11-12; changes 9 and 10) changed Task 1A.1's `MID_NAPHTHALENE` preset and its test, and Task 1A.7's `_outer_erase_ranges`, `cheapest_block_bits` and tests; those steps were re-run in the same replica, RED and GREEN (logs in `.scratch/plan-check/stages/1A-rev1/`).

---

## Changes from the master plan

Every change below is applied in the master plan (`2026-11-11-dfthc-master-plan.md`, the version assembled with this file). The spec, HANDOFF and LEDGER edits it implies are old-to-new text pairs in `phase1-doc-edits.md`, which the orchestrator applies on `main`.

1. **Task 1A.4 depends on Task 1A.3, and Workstream A moves one wave later from 1A.4 on.** `WalkTables.from_factors` builds its alias tables from `outer_weights` and `inner_weights`, which live in Task 1A.3's `sos.py` (spec 5.2). Duplicating them in `tables.py` would be duplicated logic. Workstream A still ends in wave 1-e, before Task 1C.5 (wave 1-f), so the critical path is unchanged. *Applied:* section 4 (row 1A.4), section 5 (mermaid and waves 1-b to 1-e) and section 7 (Task 1A.4, "Depends on"); the LEDGER row is in `phase1-doc-edits.md`.
2. **The preset-shape check moves from Task 1A.2 to Task 1A.4,** because Task 1A.2 runs in the same wave as Task 1A.1 and cannot import `dfthc.params`. *Applied:* section 7, the acceptance tests of Tasks 1A.2 and 1A.4.
3. **Task 1A.1 gains helpers and two checks:** `lg(y)` (exact `ceil(log2 y)`, `lg(1) == 0`) and `odd_part(y)`; `WalkParams` also rejects `n_plus` outside `[0, N]` and non-integer fields; `GROUPS`; zero-width registers are empty `Wires`; `split_group` returns slices of the type it receives. *Applied:* section 7, Task 1A.1 interfaces.
4. **Task 1A.2 gains constants and conventions:** `SYSTEMS`, `UNIT_TOLERANCE = 1e-10`, `DFTHCFactors(frozen=True, eq=False)` with read-only arrays, `FileNotFoundError` from `load_integrals` for `fe2s2` and `fe4s4`, the fixed scalars of `random_factors`, `MAX_ORBITALS = 4`. *Applied:* section 7, Task 1A.2 interfaces.
5. **Task 1A.3's check 1 allows relative 2e-10 for `fe4s4`.** The stored `lambda_h1` of `fe4s4` exceeds `sum(w_plus) + sum(w_minus)` by 5.96e-9, so Λ from its factors is 1.18e-10 (relative) below `lambda_sos`; 2e-10 leaves a margin of 1.7. Every other system agrees to 1e-13 and keeps 1e-10. *Applied:* section 7, Task 1A.3 acceptance tests.
6. **The Givens ladder's time order is pinned.** With the rung matrix of spec 6.3, `U^dag gamma_0 U = sum_j u_j gamma_j` holds for `U = G_0 G_1 ... G_{N-2}`: rung `N - 2` acts first in time and rung 0 last, as in dfthcbe's `GivensRotationLadder` (`for k in reversed(range(N - 1))`). The opposite order fails for `N >= 3`, and the K-1 stand-in's placeholder ladder uses it. *Applied:* section 7, Task 1A.4 (`ladder_matrix`) and section 8, Task 2E.4 interfaces.
7. **Task 1A.6's fixtures come from the pinned dfthcbe tarball, generated by the orchestrator.** The recipe of Step 7 downloads the tarball of commit `930255020cec30172e0a7d17c0071d9476a687d1` from `codeload.github.com` into `<main checkout>/.scratch/dfthcbe-ref` if that directory is missing, checks a SHA-256 digest of its sources, and installs a copy into `.venv-ref`. No step needs elevated permission and nothing is cloned. The recipe also avoids the two sandbox failures of a plain install: pip's certificate verification through the proxy, and the jedi file `typeshed/stdlib/venv/__init__.pyi`, which the sandbox refuses because no directory named `venv` may be created in the workspace. The FeMoco counts come from dfthcbe's symbolic call graph at `FEMOCO54`, so no FeMoco table is built, and they reproduce dfthcbe's own FeMoco test values (walk 9,641). The encoded fixture holds `block`, `lambda_tilde` and `h_sa`. *Applied:* section 7, Task 1A.6 ("Files" gains `requirements-ref.lock`; interfaces; "Definition of done"); section 5, wave 1-e note. The HANDOFF 3.4 replacement is in `phase1-doc-edits.md`.
8. **Task 1A.7 gains arithmetic and checks:** `Cost` supports `+` and `int * Cost`; the `ranges` of `qroam_erase` are its fixup's high ranges; invalid ranges and `zero_reflection(0, ...)` raise `ValueError`; `cheapest_block_bits` breaks ties toward the smallest value and returns `FEMOCO54`'s `(2, 4, 4, 2)`. *Applied:* section 7, Task 1A.7 interfaces.

Revision 1 (2026-11-12), from the Phase 1 C plan review (I4) and the orchestrator's decision on `MID_NAPHTHALENE`. The master plan and spec edits of these two changes are old-to-new text pairs in `phase1c-doc-edits.md` (M8 to M10, M12, S3, S7); they are not applied yet.

9. **Task 1A.7's outer erase covers the outer read's loaded set.** The erase fixup must cover every address of a visited block, including dead positions beyond `X_o` whose junk is not zero (spec 9.5, Rule step 2). The read loads `[0, ceil(X_o / K1) K1)`, so `_outer_erase_ranges` uses the high endpoint `ceil(ceil(X_o / K1) K1 / 2**ke1)`; spec 9.24's `ceil(X_o / 2**ke1)` undercounts when `ke1 < k1` and `K1` does not divide `X_o` (at `X_o = 5`, `k1 = 2`, `ke1 = 0`: 8 against 5). No preset is affected. Because the outer erase now depends on `k1`, `cheapest_block_bits` minimizes `(k1, ke1)` jointly; no shape was found on which this changes the result (`.scratch/plan-check/stages/1A-rev1/minimizers.log`). The test `test_outer_erase_covers_the_loaded_set` pins `(5, 2, 0)` and a grid of nonaligned shapes, and the brute-force test also runs at `MID_NAPHTHALENE` (`X_o = 50`) and at `X_o = 5`. Task 1C.5's `QROAMErase.fixup_high_ranges` is the implementation oracle; a Task 1C.5 test cross-checks the two.
10. **`MID_NAPHTHALENE` takes dfthcbe's naphthalene block bits, `k1 = 1`, `k2 = 2`, `ke1 = 3`, `ke2 = 1`** (were `2, 1, 2, 1`): dfthcbe's README example chooses them by its Toffoli minimizers (`.scratch/plan-check/probes/p1c_dfthcbe_naphthalene_blocks.py`: outer read block bits 1, erase 3; inner read `(0, 2)`, erase `(1, 0)`). `dfthc.costs.cheapest_block_bits` gives the same four values, and the brute-force test pins them. Task 3.4's comparison with the K-1 stand-in's `mid10` shape (block bits `k1 = 2`, `k2 = 1`) is then between equal sizes but different block bits.

Additions that need no master-plan edit:
- Task 1A.5 adds a third construction of the block: `(1 - P_fail(x_o)) sum_x p_x avg_sigma 2 A_x^dag A_x - I` built from the tables without `walk_integrals`. It covers a shape whose uniform preparation of `x_o` fails (`X_o = 5`).
- Task 1A.4 adds the exhaustive sign check of spec 6.1: every `(l, m)` branch carries the sign of the index that survives the swap.

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
| `src/dfthc/params.py` | 1A.1 | `WalkParams`, `lg`, `odd_part`, the four presets |
| `src/dfthc/registers.py` | 1A.1 | `WalkLayout`, `GROUPS`: named wires, groups, the work pool, `split_group` |
| `tests/data/test_params.py` | 1A.1 | constraints, derived sizes, presets |
| `tests/test_registers.py` | 1A.1 | group sizes, disjointness, `split_group`, `angle_register` |
| `src/dfthc/data/__init__.py` | 1A.2 | package docstring only (never re-exports) |
| `src/dfthc/data/factors.py` | 1A.2 | `DFTHCFactors`, loaders, validation, `random_factors` |
| `src/dfthc/data/fermion.py` | 1A.2 | dense Jordan-Wigner matrices on `psi_up (x) psi_dn` (N <= 4) |
| `tests/data/test_factors.py`, `tests/data/test_fermion.py` | 1A.2 | loaders, validation, the spec 6.2 representation |
| `src/dfthc/data/sos.py` | 1A.3 | weights, Λ, both offset frames, λ_eff, dense Hamiltonians |
| `tests/data/test_sos.py` | 1A.3 | spec 8.6 checks 1 to 3 |
| `src/dfthc/data/alias.py`, `angles.py`, `tables.py` | 1A.4 | alias tables, Givens angles, `WalkTables` |
| `tests/data/test_alias.py`, `test_angles.py`, `test_tables.py` | 1A.4 | probabilities, factorization bounds, the ladder, table shapes |
| `src/dfthc/data/encoded.py` | 1A.5 | `encoded_block`, `encoded_factors`, failure probabilities and signs |
| `tests/data/test_encoded.py` | 1A.5 | three constructions of the block, convergence, failure signs |
| `scripts/make_dfthcbe_fixtures.py` | 1A.6 | fixture generation in `.venv-ref` |
| `src/dfthc/testing/fixtures.py` | 1A.6 | `FIXTURE_DIR`, `require_fixture`, `load_dfthcbe_fixture` |
| `tests/fixtures/dfthcbe/*` | 1A.6 | the generated fixtures, provenance and `requirements-ref.lock` |
| `tests/data/test_dfthcbe_fixtures.py` | 1A.6 | spec 8.6 check 4 |
| `src/dfthc/costs.py` | 1A.7 | the closed forms of spec 9.24 |
| `tests/test_costs.py` | 1A.7 | every row of the spec 9.24 evaluation table at FeMoco-54 |

Within each wave of master plan section 5 (as amended by change 1), the tasks' file lists are disjoint. Test modules have unique base names, because `tests/` has no `__init__.py` files.

**Shell conventions.** Run commands from the worktree root of your lane. `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` locates the shared environment, as in the Phase 0 plans.

**Expected test counts.** Each task states its own targeted count exactly. The per-commit tier (`scripts/test.sh -m "not slow and not release" -q`) counts 181 tests at G0: 140 measured on `main` at `3534c30` (Tasks 0.1, 0.2 and 0.5, with Task 0.1's two extra tests and Task 0.2's 104 tests) plus the 41 per-commit tests that the Phase 0 plans give Tasks 0.3, 0.4 and 0.6. If those tasks merge with a different count, shift every number below by the difference. On top of that it counts the per-commit tests of every Phase 1 task merged into `main` before your branch was created, plus your own. The table gives those counts. Each task's expected per-commit number assumes that every task of the earlier waves is merged and no Workstream C task is. Add the per-commit counts of any merged Workstream C task from `2026-11-11-dfthc-phase1-primitives.md`. The pass criterion is "no failures, no errors, no warnings summary"; the number is a cross-check.

| Task | Per-commit tests | Slow tests |
| --- | ---: | ---: |
| 1A.1 | 41 | 0 |
| 1A.2 | 31 | 0 |
| 1A.3 | 17 | 0 |
| 1A.4 | 49 | 0 |
| 1A.5 | 15 | 0 |
| 1A.6 | 7 (2 before the fixtures exist, 5 skipped) | 0 |
| 1A.7 | 14 | 0 |
| 1B.1 | 23 | 0 |
| 1B.2 | 27 | 0 |
| 1B.3 | 0 | 2 |

With every task of both plans merged, the per-commit tier gives `405 passed` (dry-run: 364 in the replica, which lacks the 41 per-commit tests of Tasks 0.3, 0.4 and 0.6).

---

### Task 1A.1: `WalkParams`, presets and `WalkLayout`

**Task id:** `1A.1` · **Branch:** `phase1/1A.1` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 (`gpt-5.6-sol-high-fast`) · **Depends on:** G0

**Files:**
- Create: `src/dfthc/params.py`, `src/dfthc/registers.py`, `tests/data/test_params.py`, `tests/test_registers.py`

**Interfaces:**
- Consumes: PennyLane's `pennylane.wires.Wires`; nothing from other Phase 1 tasks.
- Produces:
  - `dfthc.params.lg(value: int) -> int` (exact `ceil(log2 value)`, `lg(1) == 0`, `ValueError` below 1) and `odd_part(value: int) -> int`.
  - `@dataclass(frozen=True) class WalkParams` with fields `N, R, B, C, n_plus, b_rot, b_k1, b_k2, s_u, k1, k2, ke1, ke2: int` (in this order). Derived properties: `RC, X_o, n_minus, w_x, w_b, w_r, d1, d2, b_grad, K1, K2, M1, M2: int`. `__post_init__` raises `ValueError("WalkParams violates <constraint> (spec 8.3): ...")` for each bound of spec 8.3, for `0 <= n_plus <= N`, and for a non-integer field. `as_compilable() -> tuple[int, ...]` returns the 13 fields; `WalkParams.from_compilable(values) -> WalkParams` inverts it.
  - Presets `FEMOCO54`, `H2_STO3G_TEST`, `H2_631G_TEST`, `MID_NAPHTHALENE`, with the values of master plan Task 1A.1.
  - `dfthc.registers.GROUPS: dict[str, tuple[str, ...]]`: `system`, `outer`, `inner`, `select`, `phase_grad`, `work` and their registers in layout order (spec 10.1).
  - `WalkLayout.from_params(params, qpe_bits=0) -> WalkLayout`, with one `Wires` attribute per register (empty for width 0). With `qpe_bits > 0` it also has `qpe_control` (1 wire) and `qpe_grad` (`qpe_bits` wires) after `work`. Group properties `system`, `outer`, `inner`, `select: Wires`, plus `num_wires: int` and `angle_register(k) -> Wires`.
  - Static methods `register_sizes(params) -> dict[str, int]`, `work_size(params) -> int` and `split_group(params, group, wires) -> dict`. `split_group` returns slices of the type it receives and raises `ValueError` for an unknown group or a wrong length.

**Acceptance tests** (41 tests):
- `tests/data/test_params.py` (25): `lg` and `odd_part`; the four presets field by field; derived values at `FEMOCO54` (`X_o = 324`, `w_x = 9`, `w_b = 5`, `w_r = 4`, `d1 = 24`, `d2 = 26`, `M1 = 81`, `M2 = 540`) and at `H2_STO3G_TEST`; 17 constraint violations, each naming its constraint; non-integer fields; the compilable round trip; hashing.
- `tests/test_registers.py` (16):
  - group sizes at `FEMOCO54` (system 108, phase_grad 15, outer 143, inner 455, select 799, work 44) and `num_wires == 1564`;
  - the spec 9.7 and 9.8 register sizes (`succ_work`, `cmp_work`, `junk`);
  - `H2_STO3G_TEST.num_wires == 63` with work 8;
  - the work-size formula at three presets;
  - disjoint registers that cover every wire;
  - `split_group` inverts the layout, also on arrays as rules see them, and rejects a wrong length;
  - `angle_register(k)` is `angles[k b_rot:(k+1) b_rot]`;
  - QPE registers only when requested.

**Definition of done:** `41 passed`; the per-commit tier passes; `ruff check .` clean; the work-size formula documented as provisional (in `WalkLayout`'s docstring; Task 3.3 asserts that every operator's `required_wires()["work"]` fits); one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_params.py`:

```python
"""WalkParams, its derived sizes and the presets (spec sections 3.1 and 8.3)."""

import dataclasses

import pytest

from dfthc.params import (
    FEMOCO54,
    H2_631G_TEST,
    H2_STO3G_TEST,
    MID_NAPHTHALENE,
    WalkParams,
    lg,
    odd_part,
)


def test_lg_is_the_exact_ceiling_of_log2():
    assert [lg(y) for y in (1, 2, 3, 4, 5, 8, 9, 28, 270, 324)] == [0, 1, 2, 2, 3, 3, 4, 5, 9, 9]
    with pytest.raises(ValueError, match="positive"):
        lg(0)


def test_odd_part():
    assert [odd_part(y) for y in (1, 2, 3, 4, 12, 28, 324)] == [1, 1, 3, 1, 3, 7, 81]
    with pytest.raises(ValueError, match="positive"):
        odd_part(0)


def test_presets_hold_the_spec_values():
    assert FEMOCO54.as_compilable() == (54, 10, 27, 27, 27, 15, 15, 15, 7, 2, 4, 4, 2)
    assert H2_STO3G_TEST.as_compilable() == (2, 2, 2, 1, 1, 4, 2, 2, 3, 1, 1, 1, 1)
    assert H2_631G_TEST.as_compilable() == (4, 2, 2, 2, 3, 4, 2, 2, 3, 1, 1, 1, 1)
    assert MID_NAPHTHALENE.as_compilable() == (10, 8, 6, 5, 5, 15, 15, 15, 7, 1, 2, 3, 1)


def test_derived_values_at_femoco():
    p = FEMOCO54
    assert (p.X_o, p.RC, p.n_minus, p.w_x, p.w_b, p.w_r) == (324, 270, 27, 9, 5, 4)
    assert (p.d1, p.d2, p.b_grad, p.K1, p.K2, p.M1, p.M2) == (24, 26, 15, 4, 16, 81, 540)


def test_derived_values_at_h2_sto3g():
    p = H2_STO3G_TEST
    assert (p.X_o, p.RC, p.n_minus, p.w_x, p.w_b, p.w_r) == (4, 2, 1, 2, 2, 1)
    assert (p.d1, p.d2, p.b_grad, p.K1, p.K2, p.M1, p.M2) == (4, 7, 4, 2, 2, 2, 4)


@pytest.mark.parametrize(
    ("changes", "constraint"),
    [
        ({"N": 1, "n_plus": 1}, "N >= 2"),
        ({"R": 0}, "R, B, C >= 1"),
        ({"B": 0}, "R, B, C >= 1"),
        ({"C": 0}, "R, B, C >= 1"),
        ({"n_plus": 3}, "0 <= n_plus <= N"),
        ({"n_plus": -1}, "0 <= n_plus <= N"),
        ({"b_rot": 2}, "b_rot >= 3"),
        ({"b_k1": 0}, "b_k1, b_k2 >= 1"),
        ({"b_k2": 0}, "b_k1, b_k2 >= 1"),
        ({"s_u": 2}, "3 <= s_u <= b_grad"),
        ({"s_u": 5}, "3 <= s_u <= b_grad"),
        ({"k1": 3}, "0 <= k1 <= w_x"),
        ({"k1": -1}, "0 <= k1 <= w_x"),
        ({"k2": 3}, "0 <= k2 <= w_b"),
        ({"ke1": 3}, "0 <= ke1 <= w_x"),
        ({"ke2": 2}, "0 <= ke2 <= lg(RC)"),
        ({"ke2": -1}, "0 <= ke2 <= lg(RC)"),
    ],
)
def test_constraint_violations_raise(changes, constraint):
    with pytest.raises(ValueError, match=constraint.replace("(", r"\(").replace(")", r"\)")):
        dataclasses.replace(H2_STO3G_TEST, **changes)


def test_fields_must_be_integers():
    with pytest.raises(ValueError, match="b_rot must be an int"):
        dataclasses.replace(H2_STO3G_TEST, b_rot=4.0)
    with pytest.raises(ValueError, match="k1 must be an int"):
        dataclasses.replace(H2_STO3G_TEST, k1=True)


def test_compilable_round_trip():
    for preset in (FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE):
        values = preset.as_compilable()
        assert len(values) == 13 and all(type(v) is int for v in values)
        assert WalkParams.from_compilable(values) == preset
    with pytest.raises(ValueError, match="13 values"):
        WalkParams.from_compilable(FEMOCO54.as_compilable()[:-1])


def test_params_are_hashable_and_frozen():
    assert len({FEMOCO54, dataclasses.replace(FEMOCO54), H2_STO3G_TEST}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        FEMOCO54.N = 3
```

Create `tests/test_registers.py`:

```python
"""WalkLayout: named wires, groups, the work pool and the split helper (spec section 10)."""

import jax.numpy as jnp
import pytest
from pennylane.wires import Wires

from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.registers import GROUPS, WalkLayout

PRESETS = [FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE]


def test_femoco_group_sizes():
    layout = WalkLayout.from_params(FEMOCO54)
    sizes = {name: len(getattr(layout, name)) for name in
             ("system", "phase_grad", "outer", "inner", "select", "work")}
    assert sizes == {"system": 108, "phase_grad": 15, "outer": 143, "inner": 455,
                     "select": 799, "work": 44}
    assert layout.num_wires == 1564


def test_femoco_register_sizes_follow_spec_9_7_and_9_8():
    sizes = WalkLayout.register_sizes(FEMOCO54)
    assert (sizes["succ_work_x_o"], sizes["cmp_work_x_o"], sizes["junk_x_o"]) == (6, 14, 72)
    assert (sizes["succ_work_b"], sizes["cmp_work_b"], sizes["junk_b"]) == (2, 14, 390)
    assert (sizes["x_o"], sizes["b"], sizes["r"], sizes["angles"]) == (9, 5, 4, 795)


def test_h2_sto3g_layout():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    assert layout.num_wires == 63
    assert len(layout.work) == 8 == WalkLayout.work_size(H2_STO3G_TEST)
    assert len(layout.succ_work_x_o) == 0  # X_o = 4 has odd part 1: a zero-width register


def test_work_size_formula():
    # max(w_x + b_k1 + w_b + b_k2, b_rot - 1, w_x + w_b,
    #     lg(l_x) + 2 s_u - 1, lg(l_b) + 2 s_u - 1, 2)
    assert WalkLayout.work_size(FEMOCO54) == max(44, 14, 14, 20, 16, 2) == 44
    assert WalkLayout.work_size(H2_STO3G_TEST) == max(8, 3, 4, 5, 7, 2) == 8
    assert WalkLayout.work_size(MID_NAPHTHALENE) == max(39, 14, 9, 18, 16, 2) == 39


@pytest.mark.parametrize("params", PRESETS)
def test_groups_never_overlap_and_cover_every_wire(params):
    layout = WalkLayout.from_params(params, qpe_bits=3)
    registers = [getattr(layout, name) for group in GROUPS.values() for name in group]
    registers += [layout.qpe_control, layout.qpe_grad]
    labels = [wire for wires in registers for wire in wires]
    assert sorted(labels) == list(range(layout.num_wires))
    groups = [layout.system, layout.outer, layout.inner, layout.select, layout.phase_grad,
              layout.work]
    assert sum(len(g) for g in groups) + 1 + 3 == layout.num_wires
    assert len(Wires.all_wires(groups)) == sum(len(g) for g in groups)


@pytest.mark.parametrize("params", PRESETS)
def test_split_group_inverts_the_layout(params):
    layout = WalkLayout.from_params(params)
    for group, names in GROUPS.items():
        wires = getattr(layout, group)
        parts = WalkLayout.split_group(params, group, wires)
        assert list(parts) == list(names)
        assert all(parts[name] == getattr(layout, name) for name in names)


def test_split_group_slices_arrays_as_rules_see_them():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    parts = WalkLayout.split_group(H2_STO3G_TEST, "inner", jnp.asarray(layout.inner.tolist()))
    assert [p.tolist() for p in parts.values()] == [w.tolist() for w in
                                                   (getattr(layout, n) for n in GROUPS["inner"])]


def test_split_group_rejects_a_wrong_length():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    with pytest.raises(ValueError, match="outer needs 16 wires; got 15"):
        WalkLayout.split_group(H2_STO3G_TEST, "outer", layout.outer[:-1])
    with pytest.raises(ValueError, match="unknown group"):
        WalkLayout.split_group(H2_STO3G_TEST, "angles", layout.angles)


def test_angle_register_is_a_slice_of_angles():
    layout = WalkLayout.from_params(FEMOCO54)
    for k in (0, 1, 52):
        assert layout.angle_register(k) == layout.angles[15 * k:15 * (k + 1)]
    with pytest.raises(ValueError, match="rung 53"):
        layout.angle_register(53)


def test_qpe_registers_only_when_requested():
    layout = WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=3)
    assert layout.num_wires == 63 + 1 + 3
    assert layout.qpe_control == Wires([63]) and layout.qpe_grad == Wires([64, 65, 66])
    with pytest.raises(AttributeError):
        _ = WalkLayout.from_params(H2_STO3G_TEST).qpe_control
    with pytest.raises(ValueError, match="qpe_bits"):
        WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=-1)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/data/test_params.py tests/test_registers.py -q`
Expected: FAIL at collection, `2 errors`: `ModuleNotFoundError: No module named 'dfthc.params'`.

- [ ] **Step 3: Write `dfthc.params`**

Create `src/dfthc/params.py`:

```python
"""Walk parameters: sizes, precisions, block bits and presets (spec sections 3.1 and 8.3)."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass, fields


def lg(value: int) -> int:
    """``ceil(log2(value))`` for a positive integer, computed exactly; ``lg(1) == 0`` (spec 3.1)."""
    if value < 1:
        raise ValueError(f"lg needs a positive integer; got {value}")
    return (value - 1).bit_length()


def odd_part(value: int) -> int:
    """The odd ``l`` of ``value = 2**eta * l`` (spec 9.2)."""
    if value < 1:
        raise ValueError(f"odd_part needs a positive integer; got {value}")
    return value >> ((value & -value).bit_length() - 1)


@dataclass(frozen=True)
class WalkParams:
    """Sizes, precisions and block bits of one DFTHC walk (spec section 8.3).

    Fields, in the order of :meth:`as_compilable`:

    - ``N``: spatial orbitals; each spin register has ``N`` qubits.
    - ``R``, ``B``, ``C``: DFTHC rank, terms per rank, copies per rank.
    - ``n_plus``: positive eigenvalues of the effective one-body matrix (D1 rows); ``N - n_plus``
      Q1 rows follow.
    - ``b_rot``: bits per Givens angle; also the phase-gradient width ``b_grad``.
    - ``b_k1``, ``b_k2``: bits of the outer and inner alias keep values.
    - ``s_u``: bits of the rotations inside a flagged uniform preparation.
    - ``k1``, ``k2``: block bits of the outer and inner QROAM reads.
    - ``ke1``, ``ke2``: one-hot bits of the outer and inner erase fixups.

    Construction raises ``ValueError`` naming the violated constraint. The constraints are those
    of spec 8.3 plus ``0 <= n_plus <= N``. Composite operators take the 13 fields as one compilable
    tuple (``as_compilable``) and rebuild the object with ``from_compilable``.
    """

    N: int
    R: int
    B: int
    C: int
    n_plus: int
    b_rot: int
    b_k1: int
    b_k2: int
    s_u: int
    k1: int
    k2: int
    ke1: int
    ke2: int

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, numbers.Integral) or isinstance(value, bool):
                raise ValueError(f"{field.name} must be an int; got {value!r}")
            object.__setattr__(self, field.name, int(value))
        self._check(self.N >= 2, "N >= 2")
        self._check(min(self.R, self.B, self.C) >= 1, "R, B, C >= 1")
        self._check(0 <= self.n_plus <= self.N, "0 <= n_plus <= N")
        self._check(self.b_rot >= 3, "b_rot >= 3")
        self._check(min(self.b_k1, self.b_k2) >= 1, "b_k1, b_k2 >= 1")
        self._check(3 <= self.s_u <= self.b_grad, "3 <= s_u <= b_grad")
        self._check(0 <= self.k1 <= self.w_x, "0 <= k1 <= w_x")
        self._check(0 <= self.k2 <= self.w_b, "0 <= k2 <= w_b")
        self._check(0 <= self.ke1 <= self.w_x, "0 <= ke1 <= w_x")
        self._check(0 <= self.ke2 <= lg(self.RC), "0 <= ke2 <= lg(RC)")

    def _check(self, holds: bool, constraint: str) -> None:
        if not holds:
            raise ValueError(f"WalkParams violates {constraint} (spec 8.3): {self}")

    @property
    def RC(self) -> int:
        """SF generators: ``R C``."""
        return self.R * self.C

    @property
    def X_o(self) -> int:
        """Values of the generator index ``x_o``: ``N + R C``."""
        return self.N + self.RC

    @property
    def n_minus(self) -> int:
        """Q1 rows: ``N - n_plus``."""
        return self.N - self.n_plus

    @property
    def w_x(self) -> int:
        """Width of ``x_o``: ``lg(X_o)``."""
        return lg(self.X_o)

    @property
    def w_b(self) -> int:
        """Width of ``b``: ``lg(B + 1)``."""
        return lg(self.B + 1)

    @property
    def w_r(self) -> int:
        """Width of ``r``: ``lg(R)``."""
        return lg(self.R)

    @property
    def d1(self) -> int:
        """Bits of one outer table entry (alt, keep): ``w_x + b_k1``."""
        return self.w_x + self.b_k1

    @property
    def d2(self) -> int:
        """Bits of one inner table entry (alt, keep, sgn, d_sgn, r): ``w_b + b_k2 + 2 + w_r``."""
        return self.w_b + self.b_k2 + 2 + self.w_r

    @property
    def b_grad(self) -> int:
        """Width of the phase-gradient register: ``b_rot``."""
        return self.b_rot

    @property
    def K1(self) -> int:
        """Outer block size ``2**k1``."""
        return 2**self.k1

    @property
    def K2(self) -> int:
        """Inner block size ``2**k2``."""
        return 2**self.k2

    @property
    def M1(self) -> int:
        """Live leaves of the outer read: ``ceil(X_o / K1)``."""
        return -(-self.X_o // self.K1)

    @property
    def M2(self) -> int:
        """Live leaves of the inner read: ``R C ceil((B + 1) / K2)``."""
        return self.RC * -(-(self.B + 1) // self.K2)

    def as_compilable(self) -> tuple[int, ...]:
        """The 13 fields in declaration order, as Python ints."""
        return tuple(getattr(self, field.name) for field in fields(self))

    @classmethod
    def from_compilable(cls, values: Sequence[int]) -> WalkParams:
        """Inverse of :meth:`as_compilable`."""
        values = tuple(values)
        if len(values) != len(fields(cls)):
            raise ValueError(f"WalkParams needs {len(fields(cls))} values; got {len(values)}")
        return cls(*values)


FEMOCO54 = WalkParams(N=54, R=10, B=27, C=27, n_plus=27, b_rot=15, b_k1=15, b_k2=15, s_u=7,
                      k1=2, k2=4, ke1=4, ke2=2)
"""FeMoco-54 with the precisions of the paper's Table III; ``n_plus`` for ``random_factors``."""

H2_STO3G_TEST = WalkParams(N=2, R=2, B=2, C=1, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3,
                           k1=1, k2=1, ke1=1, ke2=1)
"""The ``h2_sto3g_R2_B2_C1`` factors at test precisions."""

H2_631G_TEST = WalkParams(N=4, R=2, B=2, C=2, n_plus=3, b_rot=4, b_k1=2, b_k2=2, s_u=3,
                          k1=1, k2=1, ke1=1, ke2=1)
"""The ``h2_631g_R2_B2_C2`` factors at test precisions."""

MID_NAPHTHALENE = WalkParams(N=10, R=8, B=6, C=5, n_plus=5, b_rot=15, b_k1=15, b_k2=15, s_u=7,
                             k1=1, k2=2, ke1=3, ke2=1)
"""The ``naphthalene`` factors at FeMoco precisions with the block bits dfthcbe chooses for its
naphthalene example: the mid size of compiled T4 (spec 12.1)."""
```

- [ ] **Step 4: Write `dfthc.registers`**

Create `src/dfthc/registers.py`:

```python
"""Named wires of the walk: ``WalkLayout`` (spec section 10)."""

from __future__ import annotations

from pennylane.wires import Wires

from dfthc.params import WalkParams, lg, odd_part

GROUPS: dict[str, tuple[str, ...]] = {
    "system": ("psi_up", "psi_dn"),
    "outer": ("x_o", "succ_x_o", "succ_work_x_o", "sigma_1", "m_x_o", "less_than_x_o",
              "cmp_work_x_o", "alt_x_o", "keep_x_o", "junk_x_o"),
    "inner": ("b", "succ_b", "succ_work_b", "sigma_0", "m_b", "less_than_b", "cmp_work_b",
              "alt_b", "keep_b", "sgn", "d_sgn", "r", "junk_b"),
    "select": ("angles", "G0", "G1", "b_neq_B", "sg"),
    "phase_grad": ("phase_grad",),
    "work": ("work",),
}
"""Groups of spec 10.1 and their registers, in layout order."""


class WalkLayout:
    """Non-overlapping named wires for every register of the walk (spec 10.1).

    Build it with :meth:`from_params`. Every register of :data:`GROUPS` is an attribute holding
    its ``Wires``; a register of width 0 is an empty ``Wires``. With ``qpe_bits > 0`` the layout
    also has ``qpe_control`` (1 wire) and ``qpe_grad`` (``qpe_bits`` wires) after ``work``.
    Wires are consecutive integers from 0 in group order: ``system``, ``outer``, ``inner``,
    ``select``, ``phase_grad``, ``work``.

    The register sizes follow spec 9.7 and 9.8: ``succ_work_* = max(lg(l) - 1, 0)`` with ``l``
    the odd part of ``X_o`` or ``B + 1``, ``cmp_work_* = b_k - 1``, ``junk_x_o = (K1 - 1) d1``
    and ``junk_b = (K2 - 1) d2``. The work pool size is :meth:`work_size`. That formula is
    provisional: Task 3.3 asserts that every operator's ``required_wires()["work"]`` fits it.
    """

    def __init__(self, params: WalkParams, qpe_bits: int, registers: dict[str, Wires]):
        self.params = params
        self.qpe_bits = qpe_bits
        self._registers = dict(registers)
        for name, wires in registers.items():
            setattr(self, name, wires)

    @classmethod
    def from_params(cls, params: WalkParams, qpe_bits: int = 0) -> WalkLayout:
        if qpe_bits < 0:
            raise ValueError(f"qpe_bits must be >= 0; got {qpe_bits}")
        sizes = cls.register_sizes(params)
        if qpe_bits:
            sizes |= {"qpe_control": 1, "qpe_grad": qpe_bits}
        registers, start = {}, 0
        for name, size in sizes.items():
            registers[name] = Wires(range(start, start + size))
            start += size
        return cls(params, qpe_bits, registers)

    @staticmethod
    def register_sizes(params: WalkParams) -> dict[str, int]:
        """Width of every register of :data:`GROUPS`, in layout order."""
        p = params
        return {
            "psi_up": p.N, "psi_dn": p.N,
            "x_o": p.w_x, "succ_x_o": 1, "succ_work_x_o": max(lg(odd_part(p.X_o)) - 1, 0),
            "sigma_1": 1, "m_x_o": p.b_k1, "less_than_x_o": 1, "cmp_work_x_o": p.b_k1 - 1,
            "alt_x_o": p.w_x, "keep_x_o": p.b_k1, "junk_x_o": (p.K1 - 1) * p.d1,
            "b": p.w_b, "succ_b": 1, "succ_work_b": max(lg(odd_part(p.B + 1)) - 1, 0),
            "sigma_0": 1, "m_b": p.b_k2, "less_than_b": 1, "cmp_work_b": p.b_k2 - 1,
            "alt_b": p.w_b, "keep_b": p.b_k2, "sgn": 1, "d_sgn": 1, "r": p.w_r,
            "junk_b": (p.K2 - 1) * p.d2,
            "angles": (p.N - 1) * p.b_rot, "G0": 1, "G1": 1, "b_neq_B": 1, "sg": 1,
            "phase_grad": p.b_grad,
            "work": WalkLayout.work_size(p),
        }

    @staticmethod
    def work_size(params: WalkParams) -> int:
        """Shared work pool (spec 10.1), provisional:
        ``max(w_x + b_k1 + w_b + b_k2, b_rot - 1, w_x + w_b, lg(l_x) + 2 s_u - 1,
        lg(l_b) + 2 s_u - 1, 2)``, with ``l_x`` and ``l_b`` the odd parts of ``X_o`` and ``B + 1``.
        """
        p = params
        return max(p.w_x + p.b_k1 + p.w_b + p.b_k2, p.b_rot - 1, p.w_x + p.w_b,
                   lg(odd_part(p.X_o)) + 2 * p.s_u - 1, lg(odd_part(p.B + 1)) + 2 * p.s_u - 1, 2)

    @staticmethod
    def split_group(params: WalkParams, group: str, wires) -> dict:
        """Split the wires of one group into its registers: the only way operators do it.

        ``wires`` is a ``Wires`` (outside rules) or an array (inside rules, where wire registers
        are arrays); each register is the slice of the same type. Raises ``ValueError`` for an
        unknown group or a wrong length.
        """
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}; expected one of {list(GROUPS)}")
        sizes = WalkLayout.register_sizes(params)
        total = sum(sizes[name] for name in GROUPS[group])
        if len(wires) != total:
            raise ValueError(f"{group} needs {total} wires; got {len(wires)}")
        parts, start = {}, 0
        for name in GROUPS[group]:
            parts[name] = wires[start:start + sizes[name]]
            start += sizes[name]
        return parts

    def _group(self, group: str) -> Wires:
        return Wires.all_wires([self._registers[name] for name in GROUPS[group]])

    @property
    def system(self) -> Wires:
        return self._group("system")

    @property
    def outer(self) -> Wires:
        return self._group("outer")

    @property
    def inner(self) -> Wires:
        return self._group("inner")

    @property
    def select(self) -> Wires:
        return self._group("select")

    @property
    def num_wires(self) -> int:
        return sum(len(wires) for wires in self._registers.values())

    def angle_register(self, k: int) -> Wires:
        """The ``b_rot`` wires of angle register ``k``, which drives rung ``k`` (spec 6.7)."""
        if not 0 <= k < self.params.N - 1:
            raise ValueError(f"rung {k} outside [0, {self.params.N - 1})")
        b_rot = self.params.b_rot
        return self.angles[k * b_rot:(k + 1) * b_rot]
```

- [ ] **Step 5: Run the tests**

Run: `scripts/test.sh tests/data/test_params.py tests/test_registers.py -q`
Expected: `41 passed`.

- [ ] **Step 6: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `222 passed` (181 at G0 plus 41; see "Expected test counts").

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/dfthc/params.py src/dfthc/registers.py tests/data/test_params.py tests/test_registers.py
git commit -m "Add WalkParams, presets and WalkLayout"
```

---

### Task 1A.2: Factors, loaders and fermion matrices

**Task id:** `1A.2` · **Branch:** `phase1/1A.2` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** G0

**Files:**
- Create: `src/dfthc/data/__init__.py`, `src/dfthc/data/factors.py`, `src/dfthc/data/fermion.py`, `tests/data/test_factors.py`, `tests/data/test_fermion.py`

**Interfaces:**
- Consumes: the vendored data of Task 0.5 (`data/dfthc/<system>/{factors.npz, integrals.npz, summary.json}`).
- Produces:
  - `dfthc.data` (docstring only; only this task edits `__init__.py`).
  - `dfthc.data.factors`: `DATA_DIR`, `SYSTEMS`, `UNIT_TOLERANCE`. `@dataclass(frozen=True, eq=False) class DFTHCFactors` holds array fields `U (R,B,N)`, `W (R,B,C)`, `w_B (R,C)`, `w_plus (N+)`, `u_plus (N+,N)`, `w_minus (N-)`, `u_minus (N-,N)`, `h1`, `h_sym`, `h1_shifted (N,N)` (read-only float64) and scalar fields `beta_1`, `lambda_h1`, `lambda_h2`, `e_upper: float`, `eta: int`. It has properties `N`, `R`, `B`, `C`, `n_plus`, `n_minus`, and `validate()`, which checks shapes, unit rows of `U`, `u_plus` and `u_minus` (1e-10), `N+ + N- = N` and `w+, w- >= 0`, raising `ValueError`.
  - Loaders: `load_factors(system) -> DFTHCFactors` (validated), `load_summary(system) -> dict`, `load_integrals(system) -> dict` with keys `h1e`, `eri`, `core_energy`, and `random_factors(N, R, B, C, n_plus, seed) -> DFTHCFactors`.
  - `dfthc.data.fermion` (dense, `N <= MAX_ORBITALS = 4`, `psi_up (x) psi_dn` ordering, qubit 0 most significant, spin 0 up and 1 down): `majorana(j, flavour, spin, N)`, `annihilation(j, spin, N)`, `number(j, spin, N)`, `rotated_majorana(u, flavour, spin, N)`, `sector_projector(N, eta)` and `occupation_index(up_bits, dn_bits, N) -> int`.

**Acceptance tests** (31 tests):
- `tests/data/test_factors.py` (19): every vendored system loads, validates and matches its summary; shapes at `h2_sto3g_R2_B2_C1`; `load_integrals` (and `FileNotFoundError` for `fe2s2`); immutability; six rejected factor sets; `random_factors` at FeMoco shape with `n_plus = 27`; reproducibility and `n_plus`.
- `tests/data/test_fermion.py` (12), the representation of spec 6.2 at N = 3:
  - Majoranas anticommute within one spin register and commute across;
  - `i gamma_{j,0} gamma_{j,1} = -Z_j` and `a^dag a = n = (I - Z)/2`;
  - qubit 0 carries no string (`gamma_0 = X`, `gamma_1 = Y`, `a = (X + iY)/2`);
  - `annihilation` lowers the particle number by one;
  - `sector_projector` commutes with same-spin bilinears;
  - `occupation_index` and `rotated_majorana`;
  - input checks.

**Definition of done:** `31 passed`; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_factors.py`:

```python
"""DFTHCFactors, the loaders and random_factors (spec sections 8.1 and 8.2)."""

import dataclasses

import numpy as np
import pytest

from dfthc.data.factors import (
    DATA_DIR,
    SYSTEMS,
    DFTHCFactors,
    load_factors,
    load_integrals,
    load_summary,
    random_factors,
)


def test_data_dir_is_the_vendored_copy():
    assert DATA_DIR.name == "dfthc" and (DATA_DIR / "LICENSE").is_file()
    assert sorted(SYSTEMS) == sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


@pytest.mark.parametrize("system", SYSTEMS)
def test_every_vendored_system_loads_and_validates(system):
    factors = load_factors(system)
    factors.validate()
    summary = load_summary(system)
    assert (factors.R, factors.B, factors.C, factors.N) == (
        summary["R"], summary["B"], summary["C"], summary["nact"])
    assert factors.n_plus + factors.n_minus == factors.N
    assert factors.lambda_h1 == summary["lambda_h1"] and factors.e_upper == summary["e_upper"]
    assert type(factors.eta) is int and factors.eta > 0


def test_h2_sto3g_shapes():
    f = load_factors("h2_sto3g_R2_B2_C1")
    assert (f.U.shape, f.W.shape, f.w_B.shape) == ((2, 2, 2), (2, 2, 1), (2, 1))
    assert (f.w_plus.shape, f.u_plus.shape, f.w_minus.shape, f.u_minus.shape) == (
        (1,), (1, 2), (1,), (1, 2))
    assert (f.h1.shape, f.h_sym.shape, f.h1_shifted.shape) == ((2, 2),) * 3
    assert (f.n_plus, f.n_minus, f.eta) == (1, 1, 2)


def test_load_integrals():
    ints = load_integrals("h2_631g_R2_B2_C2")
    assert set(ints) == {"h1e", "eri", "core_energy"}
    assert ints["h1e"].shape == (4, 4) and ints["eri"].shape == (4, 4, 4, 4)
    assert ints["core_energy"] == pytest.approx(0.7151043390810812, abs=1e-15)
    with pytest.raises(FileNotFoundError):
        load_integrals("fe2s2")  # the large systems ship no integrals (spec 8.1)


def test_factors_are_immutable():
    f = load_factors("h2_sto3g_R2_B2_C1")
    with pytest.raises(ValueError, match="read-only"):
        f.U[0, 0, 0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.eta = 4


def _broken(**changes):
    return dataclasses.replace(load_factors("h2_631g_R2_B2_C2"), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"W": np.ones((2, 3, 2))}, "W has shape"),
        ({"h_sym": np.ones((3, 3))}, "h_sym has shape"),
        ({"U": np.full((2, 2, 4), 0.6)}, "rows of U"),
        ({"u_plus": np.full((3, 4), 0.6)}, "rows of u_plus"),
        ({"w_plus": np.ones(2), "u_plus": np.eye(4)[:2]}, "N\\+ \\+ N- = N"),
        ({"w_minus": -np.ones(1)}, "w_minus must be >= 0"),
    ],
)
def test_validate_rejects_bad_factors(changes, message):
    with pytest.raises(ValueError, match=message):
        _broken(**changes).validate()


def test_random_factors_validate_at_femoco_shape():
    f = random_factors(54, 10, 27, 27, n_plus=27, seed=0)
    f.validate()
    assert (f.N, f.R, f.B, f.C, f.n_plus, f.n_minus) == (54, 10, 27, 27, 27, 27)
    split = f.u_plus.T @ np.diag(f.w_plus) @ f.u_plus - f.u_minus.T @ np.diag(f.w_minus) @ f.u_minus
    assert np.allclose(split, f.h1_shifted, atol=1e-12)
    assert f.lambda_h1 == pytest.approx(f.w_plus.sum() + f.w_minus.sum(), rel=1e-12)


def test_random_factors_are_reproducible_and_respect_n_plus():
    a, b = random_factors(4, 2, 3, 2, n_plus=1, seed=7), random_factors(4, 2, 3, 2, 1, seed=7)
    assert all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("U", "W", "w_B", "u_plus"))
    assert (a.n_plus, a.n_minus) == (1, 3)
    assert not np.array_equal(a.W, random_factors(4, 2, 3, 2, n_plus=1, seed=8).W)
    with pytest.raises(ValueError, match="n_plus"):
        random_factors(4, 2, 3, 2, n_plus=5, seed=0)


def test_factors_type_is_exported():
    assert isinstance(load_factors("naphthalene"), DFTHCFactors)
```

Create `tests/data/test_fermion.py`:

```python
"""Dense Jordan-Wigner matrices: spec 6.2's representation, checked at N = 3."""

import functools
import itertools
from math import comb

import numpy as np
import pytest

from dfthc.data.fermion import (
    annihilation,
    majorana,
    number,
    occupation_index,
    rotated_majorana,
    sector_projector,
)

N = 3
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
EYE = np.eye(4**N)
MODES = list(itertools.product(range(N), (0, 1)))  # (j, flavour)


def _on_qubit(gate, j, spin):
    """``gate`` on qubit ``j`` of register ``spin`` (psi_up first, qubit 0 most significant)."""
    one = functools.reduce(np.kron, [gate if q == j else np.eye(2) for q in range(N)])
    return np.kron(one, np.eye(2**N)) if spin == 0 else np.kron(np.eye(2**N), one)


@pytest.mark.parametrize("spin", [0, 1])
def test_majoranas_anticommute_within_one_spin_register(spin):
    for (j, f), (k, g) in itertools.product(MODES, repeat=2):
        a, b = majorana(j, f, spin, N), majorana(k, g, spin, N)
        expected = 2 * EYE if (j, f) == (k, g) else 0 * EYE
        assert np.allclose(a @ b + b @ a, expected, atol=1e-12)


def test_majoranas_on_different_spins_commute():
    """Each spin register has its own Jordan-Wigner string, so odd operators commute across."""
    for (j, f), (k, g) in itertools.product(MODES, repeat=2):
        a, b = majorana(j, f, 0, N), majorana(k, g, 1, N)
        assert np.allclose(a @ b, b @ a, atol=1e-12)


@pytest.mark.parametrize("spin", [0, 1])
def test_the_even_identities_the_walk_relies_on(spin):
    for j in range(N):
        z = _on_qubit(Z, j, spin)
        assert np.allclose(1j * majorana(j, 0, spin, N) @ majorana(j, 1, spin, N), -z)
        a = annihilation(j, spin, N)
        assert np.allclose(a.conj().T @ a, number(j, spin, N))
        assert np.allclose(number(j, spin, N), (EYE - z) / 2)


@pytest.mark.parametrize("spin", [0, 1])
def test_qubit_zero_carries_no_string(spin):
    """The Majorana-gate target psi_dn[0]: gamma_0 = X, gamma_1 = Y, a = (X + iY)/2."""
    assert np.allclose(majorana(0, 0, spin, N), _on_qubit(X, 0, spin))
    assert np.allclose(majorana(0, 1, spin, N), _on_qubit(Y, 0, spin))
    assert np.allclose(annihilation(0, spin, N), _on_qubit((X + 1j * Y) / 2, 0, spin))


def test_annihilation_lowers_the_particle_number_by_one():
    total = sum(number(j, s, N) for j in range(N) for s in (0, 1))
    for j, s in itertools.product(range(N), (0, 1)):
        a = annihilation(j, s, N)
        assert np.allclose(total @ a - a @ total, -a)
        assert np.linalg.norm(a) > 1


def test_sector_projector_commutes_with_same_spin_bilinears():
    for eta in range(2 * N + 1):
        p = sector_projector(N, eta)
        assert np.allclose(p @ p, p) and round(np.trace(p).real) == comb(2 * N, eta)
        for j, k, s in itertools.product(range(N), range(N), (0, 1)):
            hop = annihilation(j, s, N).conj().T @ annihilation(k, s, N)
            assert np.allclose(p @ hop, hop @ p)


def test_occupation_index_matches_the_number_operators():
    for up, dn in itertools.product(itertools.product((0, 1), repeat=N), repeat=2):
        index = occupation_index(up, dn, N)
        for j in range(N):
            assert number(j, 0, N)[index, index] == up[j]
            assert number(j, 1, N)[index, index] == dn[j]
    assert occupation_index((1, 0, 0), (0, 0, 1), N) == 0b100001


def test_rotated_majorana_is_the_linear_combination():
    u = np.array([0.6, 0.0, 0.8])
    for f, s in itertools.product((0, 1), (0, 1)):
        expected = 0.6 * majorana(0, f, s, N) + 0.8 * majorana(2, f, s, N)
        assert np.allclose(rotated_majorana(u, f, s, N), expected)
        assert np.allclose(rotated_majorana(u, f, s, N) @ rotated_majorana(u, f, s, N), EYE)


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="N <= 4"):
        majorana(0, 0, 0, 5)
    with pytest.raises(ValueError, match="flavour"):
        majorana(0, 2, 0, N)
    with pytest.raises(ValueError, match="spin"):
        majorana(0, 0, 2, N)
    with pytest.raises(ValueError, match="orbital"):
        majorana(3, 0, 0, N)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/data/test_factors.py tests/data/test_fermion.py -q`
Expected: FAIL at collection, `2 errors`: `ModuleNotFoundError: No module named 'dfthc.data'`.

- [ ] **Step 3: Write the package and the factors module**

Create `src/dfthc/data/__init__.py`:

```python
"""L0 classical data: factors, sum-of-squares quantities, tables and the encoded model (spec 8)."""
```

Create `src/dfthc/data/factors.py`:

```python
"""DFTHC factors: data model, loaders, validation and random factors (spec sections 8.1, 8.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "dfthc"
"""The vendored copy of dfthcbe's ``data/dfthc`` (spec 8.1)."""

SYSTEMS = ("h2_sto3g_R2_B2_C1", "h2_631g_R2_B2_C2", "h2_631g_R2_B3_C3", "naphthalene", "fe2s2",
           "fe4s4")
"""The vendored systems; the last two ship no ``integrals.npz``."""

UNIT_TOLERANCE = 1e-10
"""Allowed deviation of a unit vector's norm from 1."""

_ARRAYS = ("U", "W", "w_B", "w_plus", "u_plus", "w_minus", "u_minus", "h1", "h_sym", "h1_shifted")
_SCALARS = ("beta_1", "lambda_h1", "lambda_h2", "e_upper")


@dataclass(frozen=True, eq=False)
class DFTHCFactors:
    """A DFTHC-factorized Hamiltonian, as stored in dfthcbe's ``factors.npz`` (spec 8.2).

    Arrays (float64, read-only):

    - ``U`` ``(R, B, N)``: unit vectors ``u_b^(r)``.
    - ``W`` ``(R, B, C)``: weights ``w_b^(r,c)``; ``w_B`` ``(R, C)``: identity weights.
    - ``w_plus`` ``(N+,)``, ``u_plus`` ``(N+, N)``: D1 eigenvalues and eigenvectors.
    - ``w_minus`` ``(N-,)``, ``u_minus`` ``(N-, N)``: Q1 eigenvalue magnitudes and eigenvectors.
    - ``h1``, ``h_sym``, ``h1_shifted`` ``(N, N)``: one-body term, BLISS shift, shifted term.

    Scalars: ``beta_1``, ``lambda_h1``, ``lambda_h2``, ``e_upper`` (floats) and ``eta``, the
    electron number of the target sector (int). Construction converts types only; call
    :meth:`validate` to check the factors. Instances compare by identity.
    """

    U: np.ndarray
    W: np.ndarray
    w_B: np.ndarray
    w_plus: np.ndarray
    u_plus: np.ndarray
    w_minus: np.ndarray
    u_minus: np.ndarray
    h1: np.ndarray
    h_sym: np.ndarray
    h1_shifted: np.ndarray
    beta_1: float
    lambda_h1: float
    lambda_h2: float
    e_upper: float
    eta: int

    def __post_init__(self):
        for name in _ARRAYS:
            array = np.array(getattr(self, name), dtype=float)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in _SCALARS:
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "eta", int(round(float(self.eta))))

    @property
    def N(self) -> int:
        return self.U.shape[2]

    @property
    def R(self) -> int:
        return self.U.shape[0]

    @property
    def B(self) -> int:
        return self.U.shape[1]

    @property
    def C(self) -> int:
        return self.W.shape[2]

    @property
    def n_plus(self) -> int:
        return self.w_plus.shape[0]

    @property
    def n_minus(self) -> int:
        return self.w_minus.shape[0]

    def validate(self) -> None:
        """Check shapes, unit rows, ``N+ + N- = N`` and ``w+, w- >= 0``; raise ``ValueError``."""
        if self.U.ndim != 3 or self.W.ndim != 3:
            raise ValueError(f"U and W must be 3D; got {self.U.shape} and {self.W.shape}")
        N, R, B, C, n_plus, n_minus = self.N, self.R, self.B, self.C, self.n_plus, self.n_minus
        expected = {"W": (R, B, C), "w_B": (R, C), "w_plus": (n_plus,), "u_plus": (n_plus, N),
                    "w_minus": (n_minus,), "u_minus": (n_minus, N), "h1": (N, N),
                    "h_sym": (N, N), "h1_shifted": (N, N)}
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        for name in ("U", "u_plus", "u_minus"):
            norms = np.linalg.norm(getattr(self, name), axis=-1)
            if norms.size and np.abs(norms - 1).max() > UNIT_TOLERANCE:
                raise ValueError(f"the rows of {name} must have norm 1 within {UNIT_TOLERANCE}; "
                                 f"got [{norms.min()}, {norms.max()}]")
        if n_plus + n_minus != N:
            raise ValueError(f"N+ + N- = N is violated: {n_plus} + {n_minus} != {N}")
        for name in ("w_plus", "w_minus"):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} must be >= 0; got {getattr(self, name)}")


def load_factors(system: str) -> DFTHCFactors:
    """The validated factors of a vendored system, from ``factors.npz``."""
    with np.load(DATA_DIR / system / "factors.npz") as data:
        factors = DFTHCFactors(**{field.name: data[field.name] for field in fields(DFTHCFactors)})
    factors.validate()
    return factors


def load_summary(system: str) -> dict:
    """``summary.json`` of a vendored system (``lambda_sos``, ``e_sos``, ``e_gap``, ...)."""
    return json.loads((DATA_DIR / system / "summary.json").read_text())


def load_integrals(system: str) -> dict:
    """``h1e`` ``(N, N)``, ``eri`` ``(N, N, N, N)`` and ``core_energy`` from ``integrals.npz``.

    Raises ``FileNotFoundError`` for ``fe2s2`` and ``fe4s4``, which ship no integrals.
    """
    with np.load(DATA_DIR / system / "integrals.npz") as data:
        return {"h1e": np.array(data["h1e"]), "eri": np.array(data["eri"]),
                "core_energy": float(data["core_energy"])}


def random_factors(N: int, R: int, B: int, C: int, n_plus: int, seed: int) -> DFTHCFactors:
    """Valid random factors of the given shape, with exactly ``n_plus`` D1 rows (spec 8.2).

    For resource counting where no data is vendored (FeMoco-54): Toffoli counts depend only on
    shapes and live ranges. Unit vectors are normalized Gaussians; ``W`` and ``w_B`` are
    Gaussian; the one-body split uses a random orthonormal basis with eigenvalue magnitudes in
    ``[0.1, 1)``. ``h1 = h1_shifted``, ``h_sym = 0``, ``beta_1 = 0``, ``lambda_h1`` and
    ``lambda_h2`` follow the data's conventions, ``e_upper`` is NaN and ``eta = N``.
    """
    if not 0 <= n_plus <= N:
        raise ValueError(f"n_plus must be in [0, {N}]; got {n_plus}")
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((R, B, N))
    U /= np.linalg.norm(U, axis=2, keepdims=True)
    W = rng.standard_normal((R, B, C))
    w_B = rng.standard_normal((R, C))
    basis, _ = np.linalg.qr(rng.standard_normal((N, N)))
    magnitudes = rng.uniform(0.1, 1.0, N)
    w_plus, w_minus = magnitudes[:n_plus], magnitudes[n_plus:]
    u_plus, u_minus = basis.T[:n_plus], basis.T[n_plus:]
    h1_shifted = u_plus.T @ np.diag(w_plus) @ u_plus - u_minus.T @ np.diag(w_minus) @ u_minus
    inner = np.concatenate([W.transpose(0, 2, 1), w_B[..., None]], axis=-1)
    return DFTHCFactors(
        U=U, W=W, w_B=w_B, w_plus=w_plus, u_plus=u_plus, w_minus=w_minus, u_minus=u_minus,
        h1=h1_shifted, h_sym=np.zeros((N, N)), h1_shifted=h1_shifted, beta_1=0.0,
        lambda_h1=float(magnitudes.sum()), lambda_h2=float((np.abs(inner).sum(-1) ** 2).sum() / 4),
        e_upper=float("nan"), eta=N,
    )
```

- [ ] **Step 4: Write the fermion module**

Create `src/dfthc/data/fermion.py`:

```python
"""Dense Jordan-Wigner fermion matrices on ``psi_up (x) psi_dn``, for N <= 4 (spec section 6.2).

Each spin register carries its own Jordan-Wigner string: ``gamma_{j,0} = Z_0 ... Z_{j-1} X_j`` and
``gamma_{j,1} = Z_0 ... Z_{j-1} Y_j``, with qubit 0 the most significant and ``|1>`` occupied. The
Fock space is ``psi_up (x) psi_dn`` with ``psi_up`` first, so a matrix has dimension ``4**N``.
Odd operators on different spin registers commute. ``spin`` is 0 for up and 1 for down;
``flavour`` is 0 (``X``) or 1 (``Y``).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np

MAX_ORBITALS = 4

_PAULI = (np.array([[0, 1], [1, 0]], dtype=complex), np.array([[0, -1j], [1j, 0]]))
_Z = np.diag([1.0, -1.0]).astype(complex)
_I2 = np.eye(2, dtype=complex)


def _check(N: int, *, j: int = 0, flavour: int = 0, spin: int = 0) -> None:
    if not 1 <= N <= MAX_ORBITALS:
        raise ValueError(f"dense fermion matrices need 1 <= N <= {MAX_ORBITALS}; got N = {N}")
    if not 0 <= j < N:
        raise ValueError(f"orbital {j} outside [0, {N})")
    if flavour not in (0, 1):
        raise ValueError(f"flavour must be 0 or 1; got {flavour}")
    if spin not in (0, 1):
        raise ValueError(f"spin must be 0 (up) or 1 (down); got {spin}")


def _on_register(matrix: np.ndarray, spin: int, N: int) -> np.ndarray:
    eye = np.eye(2**N, dtype=complex)
    return np.kron(matrix, eye) if spin == 0 else np.kron(eye, matrix)


def majorana(j: int, flavour: int, spin: int, N: int) -> np.ndarray:
    """``gamma_{j, flavour, spin}`` on the ``4**N`` Fock space."""
    _check(N, j=j, flavour=flavour, spin=spin)
    factors = [_Z] * j + [_PAULI[flavour]] + [_I2] * (N - 1 - j)
    return _on_register(functools.reduce(np.kron, factors), spin, N)


def annihilation(j: int, spin: int, N: int) -> np.ndarray:
    """``a_{j, spin} = (gamma_{j,0} + i gamma_{j,1}) / 2``."""
    return (majorana(j, 0, spin, N) + 1j * majorana(j, 1, spin, N)) / 2


def number(j: int, spin: int, N: int) -> np.ndarray:
    """``n_{j, spin} = a^dag a = (I - Z_j) / 2`` on register ``spin``."""
    a = annihilation(j, spin, N)
    return a.conj().T @ a


def rotated_majorana(u: Sequence[float], flavour: int, spin: int, N: int) -> np.ndarray:
    """``sum_j u_j gamma_{j, flavour, spin}``, the Majorana along the direction ``u``."""
    u = np.asarray(u, dtype=float)
    if u.shape != (N,):
        raise ValueError(f"u must have shape ({N},); got {u.shape}")
    return sum(u[j] * majorana(j, flavour, spin, N) for j in range(N))


def sector_projector(N: int, eta: int) -> np.ndarray:
    """Diagonal projector onto the Fock states with exactly ``eta`` electrons."""
    _check(N)
    occupations = np.array([bin(index).count("1") for index in range(4**N)])
    return np.diag((occupations == eta).astype(complex))


def occupation_index(up_bits: Sequence[int], dn_bits: Sequence[int], N: int) -> int:
    """Fock basis index of the occupations ``up_bits`` and ``dn_bits`` (qubit 0 first)."""
    _check(N)
    bits = list(up_bits) + list(dn_bits)
    if len(bits) != 2 * N or any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"need {N} occupation bits per spin; got {up_bits} and {dn_bits}")
    return int("".join(str(bit) for bit in bits), 2)
```

- [ ] **Step 5: Run the tests**

Run: `scripts/test.sh tests/data/test_factors.py tests/data/test_fermion.py -q`
Expected: `31 passed`.

- [ ] **Step 6: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `212 passed` (181 plus 31).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/dfthc/data tests/data/test_factors.py tests/data/test_fermion.py
git commit -m "Add DFTHC factors, loaders and Jordan-Wigner fermion matrices"
```

---

### Task 1A.3: Sum-of-squares quantities and the two offset frames

**Task id:** `1A.3` · **Branch:** `phase1/1A.3` · **Tier:** A (`inherit`): conflating the walk-frame and physical-frame offsets was a load-bearing finding in the spec reviews · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 1A.2

**Files:**
- Create: `src/dfthc/data/sos.py`, `tests/data/test_sos.py`

**Interfaces:**
- Consumes: `DFTHCFactors`, `load_factors`, `load_summary`, `load_integrals`, `SYSTEMS`; `dfthc.data.fermion.annihilation`, `sector_projector`.
- Produces, in `dfthc.data.sos`:
  - Weights and offsets:
    - `outer_weights(factors) -> ndarray (X_o,)`: SF first at `x_o = r C + c`, then `2 w_+`, then `2 w_-`;
    - `inner_weights(factors) -> ndarray (RC, B+1)`: signed, with `w_B` at `b = B`;
    - `normalization(factors) -> float`, `e_sos_walk(factors) -> float`, `e_sos_phys(summary) -> float`, `frame_shift(factors, summary) -> float`;
    - `lambda_eff(Lambda, e_gap) -> float`, which raises `ValueError` unless `0 <= e_gap <= 2 Lambda`.
  - Dense Hamiltonians: `walk_integrals(factors) -> (h1e, eri)` (port of dfthcbe), and `h_sa_dense(factors)` (directly from the generators of spec 3.2), `h_walk_dense(factors)` and `h_phys_dense(integrals)`, each an `ndarray (4**N, 4**N)`.

**Acceptance tests** (17 tests, spec 8.6 checks 1 to 3):
- Check 1, for all six systems:
  - Λ equals `lambda_sos`, relative 1e-10 (2e-10 for `fe4s4`: measured 1.18e-10, see change 5);
  - λ_eff equals `lambda_eff`;
  - `e_sos == e_upper - e_gap`.
- The weights layout; the walk offset formula; the 8-fold symmetry of `walk_integrals`.
- Check 2, walk frame, for the three H2 systems: `h_sa_dense` equals `h_walk_dense - E_SoS^walk I` to 1e-10, is Hermitian and is PSD.
- Check 3, sector ground energy: the lowest eigenvalue on the eta-electron sector equals `e_gap` within 1e-10, 3e-2 and 1e-3.
- Check 3, physical frame, `h2_sto3g_R2_B2_C1`:
  - `E_SoS^walk = -0.6923157`, `E_SoS^phys = -2.0132705` and `Delta_eta = -1.3209547` (7 decimals);
  - `H_phys - H_walk = Delta_eta I` on the sector to 1e-10;
  - `H_phys`'s sector ground energy is `e_upper`.
- `lambda_eff`'s formula and domain.

**Definition of done:** `17 passed`; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_sos.py`:

```python
"""Sum-of-squares quantities and the two offset frames: spec 8.6 checks 1 to 3 (and 3.2.1)."""

import numpy as np
import pytest

from dfthc.data.factors import SYSTEMS, load_factors, load_integrals, load_summary
from dfthc.data.fermion import sector_projector
from dfthc.data.sos import (
    e_sos_phys,
    e_sos_walk,
    frame_shift,
    h_phys_dense,
    h_sa_dense,
    h_walk_dense,
    inner_weights,
    lambda_eff,
    normalization,
    outer_weights,
    walk_integrals,
)

FACTORS_ERROR = {"h2_sto3g_R2_B2_C1": 1e-10, "h2_631g_R2_B2_C2": 3e-2, "h2_631g_R2_B3_C3": 1e-3}
"""dfthcbe's per-instance factorization tolerances (spec 8.6, check 3)."""

LAMBDA_TOLERANCE = {system: 1e-10 for system in SYSTEMS} | {"fe4s4": 2e-10}
"""fe4s4's stored lambda_h1 exceeds sum(w_plus) + sum(w_minus) by 5.96e-9, so Lambda from its
factors is 1.18e-10 (relative) below lambda_sos: 2e-10 leaves a margin of 1.7. Every other system
agrees to 1e-13."""


def _sector(matrix, n, eta):
    keep = np.flatnonzero(np.diag(sector_projector(n, eta)).real)
    return matrix[np.ix_(keep, keep)]


@pytest.mark.parametrize("system", SYSTEMS)
def test_scalars_match_the_summary(system):
    """Check 1: Lambda, lambda_eff and e_sos against summary.json, relative 1e-10."""
    factors, summary = load_factors(system), load_summary(system)
    Lambda, rel = normalization(factors), LAMBDA_TOLERANCE[system]
    assert Lambda == pytest.approx(summary["lambda_sos"], rel=rel)
    assert lambda_eff(Lambda, summary["e_gap"]) == pytest.approx(summary["lambda_eff"], rel=rel)
    assert e_sos_phys(summary) == summary["e_sos"]
    assert e_sos_phys(summary) == pytest.approx(summary["e_upper"] - summary["e_gap"], rel=1e-10)


def test_weights_layout():
    """SF first at x_o = r C + c, then 2 w_+ (D1) and 2 w_- (Q1); inner rows signed, w_B last."""
    f = load_factors("h2_631g_R2_B3_C3")
    outer, inner = outer_weights(f), inner_weights(f)
    assert outer.shape == (f.N + f.R * f.C,) and inner.shape == (f.R * f.C, f.B + 1)
    for r, c in np.ndindex(f.R, f.C):
        row = np.append(f.W[r, :, c], f.w_B[r, c])
        assert np.array_equal(inner[r * f.C + c], row)
        assert outer[r * f.C + c] == pytest.approx(np.abs(row).sum() ** 2 / 2, rel=1e-15)
    assert np.array_equal(outer[f.R * f.C:], 2 * np.concatenate([f.w_plus, f.w_minus]))
    assert normalization(f) == pytest.approx(outer.sum() / 2, rel=1e-15)


def test_walk_offset_formula():
    f = load_factors("h2_631g_R2_B2_C2")
    identity = f.w_B - f.W.sum(axis=1)
    assert e_sos_walk(f) == pytest.approx(-2 * f.w_minus.sum() - (identity**2).sum() / 2,
                                          rel=1e-15)


def test_walk_integrals_have_the_eightfold_symmetry():
    h1e, eri = walk_integrals(load_factors("h2_631g_R2_B3_C3"))
    assert np.allclose(h1e, h1e.T)
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        assert np.allclose(eri, eri.transpose(perm))


@pytest.mark.parametrize("system", FACTORS_ERROR)
def test_walk_frame_on_the_whole_fock_space(system):
    """Check 2: the generators' H_SA equals H_walk - E_SoS^walk to 1e-10 and is PSD."""
    f = load_factors(system)
    h_sa = h_sa_dense(f)
    assert h_sa.shape == (4**f.N, 4**f.N)
    assert np.allclose(h_sa, h_walk_dense(f) - e_sos_walk(f) * np.eye(4**f.N), atol=1e-10)
    assert np.allclose(h_sa, h_sa.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(h_sa).min() >= -1e-10


@pytest.mark.parametrize("system", FACTORS_ERROR)
def test_lowest_sector_eigenvalue_is_the_gap(system):
    """Check 3: on the eta-electron sector the ground energy of H_SA is e_gap (both frames)."""
    f, summary = load_factors(system), load_summary(system)
    gap = np.linalg.eigvalsh(_sector(h_sa_dense(f), f.N, f.eta)).min()
    assert gap == pytest.approx(summary["e_gap"], rel=0, abs=FACTORS_ERROR[system])


def test_physical_frame_on_the_sector_for_h2_sto3g():
    """Check 3: H_phys - H_walk = Delta_eta I on the 2-electron sector (spec 3.2.1 values)."""
    system = "h2_sto3g_R2_B2_C1"
    f, summary = load_factors(system), load_summary(system)
    assert round(e_sos_walk(f), 7) == -0.6923157
    assert round(e_sos_phys(summary), 7) == -2.0132705
    assert round(frame_shift(f, summary), 7) == -1.3209547
    difference = _sector(h_phys_dense(load_integrals(system)) - h_walk_dense(f), f.N, f.eta)
    assert np.allclose(difference, frame_shift(f, summary) * np.eye(len(difference)), atol=1e-10)
    ground = np.linalg.eigvalsh(_sector(h_phys_dense(load_integrals(system)), f.N, f.eta)).min()
    assert ground == pytest.approx(summary["e_upper"], abs=1e-10)


def test_lambda_eff_formula():
    assert lambda_eff(2.0, 0.5) == pytest.approx(np.sqrt(0.5 * 3.5), rel=1e-15)
    with pytest.raises(ValueError, match="0 <= e_gap <= 2 Lambda"):
        lambda_eff(1.0, 3.0)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/data/test_sos.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.data.sos'`.

- [ ] **Step 3: Write `dfthc.data.sos`**

Create `src/dfthc/data/sos.py`:

```python
"""Sum-of-squares weights, normalization, both offset frames and dense Hamiltonians (spec 3.2).

Two constants relate ``H_SA = sum_alpha O_alpha^dag O_alpha`` to energies and must not be
conflated (spec 3.2.1):

- the walk frame on the whole Fock space: ``H_walk = H_SA + E_SoS^walk``, with ``H_walk`` built
  from :func:`walk_integrals` and ``E_SoS^walk`` from the factors (:func:`e_sos_walk`);
- the physical frame on the ``eta``-electron sector: ``H_phys = H_SA + E_SoS^phys``, with
  ``E_SoS^phys`` the input datum ``summary.json["e_sos"]`` (:func:`e_sos_phys`).

They differ on the sector by the constant :func:`frame_shift`. The dense matrices act on the
``4**N`` Fock space of :mod:`dfthc.data.fermion` (``N <= 4``).
"""

from __future__ import annotations

import numpy as np

from dfthc.data.factors import DFTHCFactors
from dfthc.data.fermion import annihilation


def outer_weights(factors: DFTHCFactors) -> np.ndarray:
    """``(X_o,)`` outer LCU weights (spec 8.4): ``(sum_b |w_b| + |w_B|)**2 / 2`` at
    ``x_o = r C + c``, then ``2 w_+`` (D1) and ``2 w_-`` (Q1); the 2 counts both spins."""
    sf = (np.abs(inner_weights(factors)).sum(axis=1) ** 2) / 2
    return np.concatenate([sf, 2 * factors.w_plus, 2 * factors.w_minus])


def inner_weights(factors: DFTHCFactors) -> np.ndarray:
    """``(R C, B + 1)`` signed inner weights; row ``r C + c`` holds ``W[r, :, c]`` then ``w_B``."""
    rows = np.concatenate([factors.W.transpose(0, 2, 1), factors.w_B[..., None]], axis=-1)
    return rows.reshape(factors.R * factors.C, factors.B + 1)


def normalization(factors: DFTHCFactors) -> float:
    """``Lambda = sum(outer_weights) / 2`` (spec 3.2)."""
    return float(outer_weights(factors).sum() / 2)


def e_sos_walk(factors: DFTHCFactors) -> float:
    """``E_SoS^walk = -2 sum w_- - 1/2 sum_rc (w_B - sum_b w_b)**2`` (dfthcbe's ``walk_e_sos``)."""
    identity = factors.w_B - factors.W.sum(axis=1)
    return float(-2 * factors.w_minus.sum() - (identity**2).sum() / 2)


def e_sos_phys(summary: dict) -> float:
    """``E_SoS^phys``: the input datum ``summary["e_sos"]``, never recomputed from the factors."""
    return float(summary["e_sos"])


def frame_shift(factors: DFTHCFactors, summary: dict) -> float:
    """``Delta_eta = E_SoS^phys - E_SoS^walk``, the frame difference on the eta sector."""
    return e_sos_phys(summary) - e_sos_walk(factors)


def lambda_eff(Lambda: float, e_gap: float) -> float:
    """``lambda_eff = sqrt(E_gap (2 Lambda - E_gap))`` (spec 3.3)."""
    if not 0 <= e_gap <= 2 * Lambda:
        raise ValueError(f"lambda_eff needs 0 <= e_gap <= 2 Lambda; got {e_gap}, {Lambda}")
    return float(np.sqrt(e_gap * (2 * Lambda - e_gap)))


def walk_integrals(factors: DFTHCFactors) -> tuple[np.ndarray, np.ndarray]:
    """``(h1e, eri)`` of ``H_walk = H_SA + E_SoS^walk``, in chemists' order (port of dfthcbe).

    The one-body part is ``h^(1)'`` from the D1/Q1 split plus ``sum_rc (w_B - sum_b w_b) h_rc``,
    with ``h_rc = sum_b w_b u_b u_b^T``; ``eri = sum_rc h_rc (x) h_rc``, and ``h1e`` adds the
    normal-ordering term ``sum_r eri[p, r, r, q] / 2``.
    """
    f = factors
    h1_split = (f.u_plus.T @ np.diag(f.w_plus) @ f.u_plus
                - f.u_minus.T @ np.diag(f.w_minus) @ f.u_minus)
    h_rc = np.einsum("rbp,rbq,rbc->rcpq", f.U, f.U, f.W)
    h1 = h1_split + np.einsum("rc,rcpq->pq", f.w_B - f.W.sum(axis=1), h_rc)
    eri = np.einsum("rcpq,rcst->pqst", h_rc, h_rc)
    return h1 + np.einsum("prrq->pq", eri) / 2, eri


def _hamiltonian_matrix(h1e: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """``sum h1e E + 1/2 sum eri a^dag a^dag a a`` on the Fock space (port of dfthcbe)."""
    n = len(h1e)
    a = [[annihilation(p, spin, n) for spin in (0, 1)] for p in range(n)]
    E = np.array([[sum(a[p][s].conj().T @ a[q][s] for s in (0, 1)) for q in range(n)]
                  for p in range(n)])
    h1 = h1e - np.einsum("prrq->pq", eri) / 2
    two_body = np.einsum("pqrs,pqij->rsij", eri, E)
    return (np.einsum("pq,pqij->ij", h1, E)
            + sum(two_body[r, s] @ E[r, s] for r in range(n) for s in range(n)) / 2)


def h_sa_dense(factors: DFTHCFactors) -> np.ndarray:
    """``H_SA = sum_alpha O_alpha^dag O_alpha`` built directly from the generators of spec 3.2.

    D1: ``sqrt(w_+) a_{u_+, sigma}``; Q1: ``sqrt(w_-) a_{u_-, sigma}^dag``; SF:
    ``(w_B + i/2 sum_{b, sigma} w_b gamma_{u_b sigma 0} gamma_{u_b sigma 1}) / sqrt(2)``.
    """
    f, n = factors, factors.N
    a = np.array([[annihilation(p, spin, n) for spin in (0, 1)] for p in range(n)])

    def a_u(u, spin):
        return np.tensordot(u, a[:, spin], axes=1)

    generators = [np.sqrt(w) * a_u(u, s) for w, u in zip(f.w_plus, f.u_plus, strict=True)
                  for s in (0, 1)]
    generators += [np.sqrt(w) * a_u(u, s).conj().T
                   for w, u in zip(f.w_minus, f.u_minus, strict=True) for s in (0, 1)]
    eye = np.eye(4**n)
    for r, c in np.ndindex(f.R, f.C):
        pair = 0
        for b in range(f.B):
            for s in (0, 1):
                ann = a_u(f.U[r, b], s)
                gamma0, gamma1 = ann + ann.conj().T, -1j * (ann - ann.conj().T)
                pair = pair + f.W[r, b, c] * 0.5j * gamma0 @ gamma1
        generators.append((f.w_B[r, c] * eye + pair) / np.sqrt(2))
    return sum(o.conj().T @ o for o in generators)


def h_walk_dense(factors: DFTHCFactors) -> np.ndarray:
    """``H_walk``: the Fock-space matrix of :func:`walk_integrals`."""
    return _hamiltonian_matrix(*walk_integrals(factors))


def h_phys_dense(integrals: dict) -> np.ndarray:
    """``H_phys`` from ``integrals.npz`` (``h1e``, ``eri``), without ``core_energy``."""
    return _hamiltonian_matrix(integrals["h1e"], integrals["eri"])
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/data/test_sos.py -q`
Expected: `17 passed`.

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `293 passed` (wave 1-a merged: 181 + 41 + 31 + 23, plus 17).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/data/sos.py tests/data/test_sos.py
git commit -m "Add sum-of-squares quantities and both offset frames"
```

---

### Task 1A.4: Alias tables, Givens angles and `WalkTables`

**Task id:** `1A.4` · **Branch:** `phase1/1A.4` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 1A.1, 1A.2, 1A.3 (change 1)

**Files:**
- Create: `src/dfthc/data/alias.py`, `src/dfthc/data/angles.py`, `src/dfthc/data/tables.py`, `tests/data/test_alias.py`, `tests/data/test_angles.py`, `tests/data/test_tables.py`

**Interfaces:**
- Consumes: `WalkParams` and the presets; `DFTHCFactors`, `load_factors`, `random_factors`; `outer_weights`, `inner_weights`; `dfthc.data.fermion.majorana`.
- Produces:
  - `dfthc.data.alias`:
    - `alias_tables(weights, bits) -> (alt, keep)`: dfthcbe's algorithm, which is Qualtran's `preprocess_probabilities_for_reversible_sampling` ported verbatim; an all-zero row gives zeros;
    - `alias_probabilities(alt, keep, bits) -> ndarray`: the formula of spec 6.4;
    - `inner_alias_row(signed_row, bits) -> (alt, keep, sgn, d_sgn)`.
  - `dfthc.data.angles`:
    - `angles_from_vector(u) -> ndarray (N-1,)` in `[0, 2 pi)`;
    - `givens_vector(theta) -> ndarray (N,)`, the continuum inverse;
    - `discretize_angles(theta, b_rot) -> ndarray[int]`;
    - `vector_from_angles(x, b_rot) -> ndarray (N,)`, equal to `givens_vector(4 pi x / 2**b_rot)`;
    - `ladder_matrix(x, b_rot, N) -> ndarray (2**N, 2**N)`, equal to `G_0 G_1 ... G_{N-2}` (change 6).
  - `dfthc.data.tables`:
    - `INT_BOUND = 2**31`;
    - `@dataclass(frozen=True) class WalkTables` with fields `outer_alt`, `outer_keep`, `inner_alt`, `inner_keep`, `inner_sgn`, `inner_d_sgn`, `tail_angles` (N × (N−1)) and `sf_angles` (R × B × (N−1)), all nested tuples of int;
    - `from_factors(factors, params) -> WalkTables`, which raises `ValueError` if `(N, R, B, C, n_plus)` differ or an entry is not in `[0, 2**31)`;
    - `as_compilable() -> tuple[tuple, ...]` and `WalkTables.from_compilable(values) -> WalkTables`.

**Acceptance tests** (49 tests):
- `tests/data/test_alias.py` (12):
  - a hand-computed table (weights 1:2:3:4 at 2 bits give `alt = [2, 3, 3, 3]`, `keep = [2, 3, 3, 0]`);
  - probabilities within `1/(L 2**bits)`, so within `2**-bits`, of the weights at five sizes up to `(324, 15)`;
  - the formula of spec 6.4 equals an exhaustive enumeration of the sampling process with `less_than = [keep <= m]`;
  - all-zero rows; input checks;
  - every `(l, m)` branch of a signed row carries the sign of its surviving index.
- `tests/data/test_angles.py` (27):
  - continuum round trips at N = 2, 4, 10, 54 (1e-12);
  - the discretization formula;
  - the reconstruction bound `||u~ - u|| <= sum |delta_k| <= (N-1) pi / 2**(b-1)` at b = 4, 8, 15 and N = 2, 4, 10, 54, with the FeMoco value 1.016e-2;
  - `vector_from_angles` against `givens_vector`;
  - one rung is the matrix of spec 6.3 for every `x < 8` at `b_rot = 4`;
  - `U^dag gamma_{0,f} U = sum_j u~_j gamma_{j,f}` densely for N = 2, 3, 4 at b = 4 and 8 (1e-12);
  - input checks.
- `tests/data/test_tables.py` (10):
  - the preset sizes equal the factor shapes, including `MID_NAPHTHALENE.n_plus == 5` (moved from Task 1A.2, change 2);
  - the tables are the alias and angle tables of the factors at three presets;
  - hashability and the compilable round trip, with every int in `[0, 2**31)`;
  - FeMoco-54 shapes with `random_factors`: inner (270, 28), tail (54, 53), SF (10, 27, 53);
  - mismatched parameters and 32-bit entries are rejected.

**Definition of done:** `49 passed`; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_alias.py`:

```python
"""Alias tables and their probabilities (spec 6.4 and 8.4)."""

import itertools

import numpy as np
import pytest

from dfthc.data.alias import alias_probabilities, alias_tables, inner_alias_row


def test_a_hand_computed_table():
    """Weights 1:2:3:4 at 2 bits discretize to 2, 3, 5, 6 sixteenths (Qualtran's rounding)."""
    alt, keep = alias_tables([1.0, 2.0, 3.0, 4.0], bits=2)
    assert alt.tolist() == [2, 3, 3, 3] and keep.tolist() == [2, 3, 3, 0]
    assert np.allclose(alias_probabilities(alt, keep, 2), np.array([2, 3, 5, 6]) / 16)


@pytest.mark.parametrize(("length", "bits"), [(3, 1), (5, 2), (28, 4), (81, 7), (324, 15)])
def test_probabilities_reproduce_the_weights(length, bits):
    """Each probability is within 1 / (L 2**bits) of w / sum(w), so within 2**-bits."""
    rng = np.random.default_rng(length)
    weights = rng.random(length) ** 3
    alt, keep = alias_tables(weights, bits)
    assert alt.shape == keep.shape == (length,)
    assert keep.min() >= 0 and keep.max() < 2**bits and 0 <= alt.min() and alt.max() < length
    error = np.abs(alias_probabilities(alt, keep, bits) - weights / weights.sum())
    assert error.max() <= 1 / (length * 2**bits) + 1e-15
    assert error.max() <= 2.0**-bits


@pytest.mark.parametrize(("length", "bits"), [(3, 2), (7, 3), (12, 2)])
def test_formula_equals_the_sampling_process(length, bits):
    """Spec 6.4: draw l uniform over [0, L) and m uniform over [0, 2**bits); the comparator sets
    less_than = [keep <= m], which swaps in alt. Enumerating every (l, m) gives the formula."""
    rng = np.random.default_rng(bits)
    alt, keep = alias_tables(rng.random(length), bits)
    counts = np.zeros(length)
    for index, m in itertools.product(range(length), range(2**bits)):
        less_than = keep[index] <= m
        counts[alt[index] if less_than else index] += 1
    assert np.array_equal(counts / (length * 2**bits), alias_probabilities(alt, keep, bits))


def test_an_all_zero_row_loads_nothing():
    alt, keep = alias_tables(np.zeros(5), bits=3)
    assert alt.tolist() == keep.tolist() == [0] * 5


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="non-negative"):
        alias_tables([1.0, -1.0], bits=2)
    with pytest.raises(ValueError, match="bits"):
        alias_tables([1.0, 1.0], bits=0)


def test_inner_row_signs_follow_the_surviving_index():
    """sgn = [w < 0] and d_sgn = sgn XOR sgn[alt]: Z(sgn) and CZ(less_than, d_sgn) leave every
    branch (l, m) with the sign of the index that survives the swap (spec 6.1)."""
    row = np.array([0.3, -1.2, 0.5, -0.1, 0.9, 0.0, -0.4])
    alt, keep, sgn, d_sgn = inner_alias_row(row, bits=3)
    assert sgn.tolist() == [0, 1, 0, 1, 0, 0, 1]
    assert np.array_equal(d_sgn, sgn ^ sgn[alt])
    assert np.array_equal((alt, keep), alias_tables(np.abs(row), 3))
    for index, m in itertools.product(range(len(row)), range(8)):
        less_than = keep[index] <= m
        survivor = alt[index] if less_than else index
        applied = (-1) ** sgn[index] * (-1) ** (less_than * d_sgn[index])
        assert applied == (-1) ** sgn[survivor]
```

Create `tests/data/test_angles.py`:

```python
"""Givens angles: factorization, discretization, reconstruction and the ladder (spec 6.3, 8.4)."""

import numpy as np
import pytest

from dfthc.data.angles import (
    angles_from_vector,
    discretize_angles,
    givens_vector,
    ladder_matrix,
    vector_from_angles,
)
from dfthc.data.fermion import majorana


def _unit(rng, n):
    u = rng.standard_normal(n)
    return u / np.linalg.norm(u)


@pytest.mark.parametrize("n", [2, 4, 10, 54])
def test_continuum_factorization_round_trips(n):
    rng = np.random.default_rng(n)
    for _ in range(20):
        u = _unit(rng, n)
        theta = angles_from_vector(u)
        assert theta.shape == (n - 1,) and theta.min() >= 0 and theta.max() < 2 * np.pi
        assert np.allclose(givens_vector(theta), u, atol=1e-12)


def test_discretization_formula():
    theta = np.array([0.0, 0.1, np.pi, 2 * np.pi - 1e-9, 1.234])
    x = discretize_angles(theta, b_rot=4)
    assert x.tolist() == [round(t / (2 * np.pi) * 8) % 8 for t in theta]
    assert x.dtype.kind == "i" and x.max() < 2**3


@pytest.mark.parametrize("b_rot", [4, 8, 15])
@pytest.mark.parametrize("n", [2, 4, 10, 54])
def test_discretized_reconstruction_error_is_bounded(n, b_rot):
    """||u~ - u|| <= sum_k |delta_k| <= (N - 1) pi / 2**(b - 1), delta_k the wrapped rounding error
    of angle k: each rung is a planar rotation, so its error adds at most linearly."""
    rng = np.random.default_rng(10 * n + b_rot)
    for _ in range(20):
        u = _unit(rng, n)
        theta = angles_from_vector(u)
        x = discretize_angles(theta, b_rot)
        delta = np.angle(np.exp(1j * (theta - 2 * np.pi * x / 2 ** (b_rot - 1))))
        error = np.linalg.norm(vector_from_angles(x, b_rot) - u)
        assert error <= np.abs(delta).sum() + 1e-12
        assert np.abs(delta).sum() <= (n - 1) * np.pi / 2 ** (b_rot - 1) + 1e-12


def test_femoco_bound_value():
    assert 53 * np.pi / 2**14 == pytest.approx(1.0163e-2, abs=1e-6)


def test_vector_from_angles_is_givens_vector_of_the_rung_angles():
    x = np.array([1, 5, 7])
    assert np.allclose(vector_from_angles(x, 4), givens_vector(4 * np.pi * x / 16), atol=1e-15)


def test_one_rung_is_the_spec_matrix():
    """On {|01>, |10>} the rung is [[cos, -sin], [sin, cos]] with theta = 4 pi x / 2**b."""
    for x in range(8):
        theta = 4 * np.pi * x / 16
        u = ladder_matrix([x], b_rot=4, N=2)
        block = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        assert np.allclose(u[np.ix_([1, 2], [1, 2])], block, atol=1e-15)
        assert np.allclose(u[np.ix_([0, 3], [0, 3])], np.eye(2)) and u[0, 1] == u[3, 2] == 0


@pytest.mark.parametrize("b_rot", [4, 8])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_ladder_maps_gamma_0_onto_the_loaded_direction(n, b_rot):
    """U^dag gamma_{0,f} U = sum_j u~_j gamma_{j,f} for u~ = vector_from_angles(x) (spec 6.3)."""
    rng = np.random.default_rng(n + b_rot)
    for _ in range(5):
        x = rng.integers(0, 2 ** (b_rot - 1), n - 1)
        u_tilde = vector_from_angles(x, b_rot)
        ladder = np.kron(ladder_matrix(x, b_rot, n), np.eye(2**n))  # on psi_up
        assert np.allclose(ladder.conj().T @ ladder, np.eye(4**n), atol=1e-12)
        for flavour in (0, 1):
            rotated = ladder.conj().T @ majorana(0, flavour, 0, n) @ ladder
            expected = sum(u_tilde[j] * majorana(j, flavour, 0, n) for j in range(n))
            assert np.allclose(rotated, expected, atol=1e-12)


def test_inputs_are_checked():
    with pytest.raises(ValueError, match="norm 1"):
        angles_from_vector([1.0, 1.0])
    with pytest.raises(ValueError, match="length N >= 2"):
        angles_from_vector([1.0])
    with pytest.raises(ValueError, match="N - 1 = 2 angles"):
        ladder_matrix([1], b_rot=4, N=3)
```

Create `tests/data/test_tables.py`:

```python
"""WalkTables: everything the circuit loads, as compilable tuples (spec 8.4)."""

import dataclasses

import numpy as np
import pytest

from dfthc.data.alias import alias_tables, inner_alias_row
from dfthc.data.angles import angles_from_vector, discretize_angles, vector_from_angles
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.sos import inner_weights, outer_weights
from dfthc.data.tables import WalkTables
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE

PRESET_SYSTEMS = [(H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"), (H2_631G_TEST, "h2_631g_R2_B2_C2"),
                  (MID_NAPHTHALENE, "naphthalene")]


def _ints(values):
    if isinstance(values, tuple):
        for value in values:
            yield from _ints(value)
    else:
        yield values


@pytest.mark.parametrize(("params", "system"), PRESET_SYSTEMS)
def test_preset_sizes_equal_the_factor_shapes(params, system):
    """Moved here from Task 1A.2, which runs in the same wave as WalkParams (Task 1A.1)."""
    f = load_factors(system)
    assert (params.N, params.R, params.B, params.C, params.n_plus) == (
        f.N, f.R, f.B, f.C, f.n_plus)


@pytest.mark.parametrize(("params", "system"), PRESET_SYSTEMS)
def test_tables_are_the_alias_and_angle_tables_of_the_factors(params, system):
    f = load_factors(system)
    tables = WalkTables.from_factors(f, params)
    alt, keep = alias_tables(outer_weights(f), params.b_k1)
    assert tables.outer_alt == tuple(alt.tolist()) and tables.outer_keep == tuple(keep.tolist())
    rows = [inner_alias_row(row, params.b_k2) for row in inner_weights(f)]
    for column, name in enumerate(("inner_alt", "inner_keep", "inner_sgn", "inner_d_sgn")):
        assert getattr(tables, name) == tuple(tuple(row[column].tolist()) for row in rows)
    tail = np.concatenate([f.u_plus, f.u_minus])
    for j, u in enumerate(tail):
        assert tables.tail_angles[j] == tuple(discretize_angles(angles_from_vector(u),
                                                                params.b_rot).tolist())
    for r, b in np.ndindex(f.R, f.B):
        reconstructed = vector_from_angles(tables.sf_angles[r][b], params.b_rot)
        bound = (f.N - 1) * np.pi / 2 ** (params.b_rot - 1)
        assert np.linalg.norm(reconstructed - f.U[r, b]) <= bound


def test_tables_are_hashable_and_round_trip_through_compilable():
    tables = WalkTables.from_factors(load_factors("h2_631g_R2_B2_C2"), H2_631G_TEST)
    values = tables.as_compilable()
    assert len(values) == 8 and isinstance(values, tuple)
    assert WalkTables.from_compilable(values) == tables
    assert hash(WalkTables.from_compilable(values)) == hash(tables)
    assert all(type(v) is int and 0 <= v < 2**31 for v in _ints(values))
    with pytest.raises(dataclasses.FrozenInstanceError):
        tables.outer_alt = ()


def test_table_shapes_at_femoco():
    tables = WalkTables.from_factors(random_factors(54, 10, 27, 27, n_plus=27, seed=0), FEMOCO54)
    assert np.shape(tables.outer_alt) == np.shape(tables.outer_keep) == (324,)
    for name in ("inner_alt", "inner_keep", "inner_sgn", "inner_d_sgn"):
        assert np.shape(getattr(tables, name)) == (270, 28)
    assert np.shape(tables.tail_angles) == (54, 53)
    assert np.shape(tables.sf_angles) == (10, 27, 53)
    assert max(_ints(tables.as_compilable())) < 2**31


def test_mismatched_params_are_rejected():
    with pytest.raises(ValueError, match="do not match the factors"):
        WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), H2_631G_TEST)


def test_entries_of_31_bits_or_more_are_rejected():
    """Compilable integers stay below 2**31 (L-24): 32-bit keep values are refused."""
    params = dataclasses.replace(H2_STO3G_TEST, b_k1=32)
    with pytest.raises(ValueError, match="below 2\\*\\*31"):
        WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), params)
    with pytest.raises(ValueError, match="below 2\\*\\*31"):
        WalkTables.from_compilable(((2**31,),) + ((),) * 7)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/data/test_alias.py tests/data/test_angles.py tests/data/test_tables.py -q`
Expected: FAIL at collection, `3 errors`: `ModuleNotFoundError: No module named 'dfthc.data.alias'` (and `dfthc.data.angles`).

- [ ] **Step 3: Write the alias module**

Create `src/dfthc/data/alias.py`:

```python
"""Alias tables for coherent alias sampling and the distributions they prepare (spec 6.4, 8.4).

``alias_tables`` is dfthcbe's ``alias_table``, which calls Qualtran's
``preprocess_probabilities_for_reversible_sampling`` (from OpenFermion); its two helpers are
ported verbatim, including the floating-point rounding, so the integer tables equal dfthcbe's.
"""

from __future__ import annotations

import math

import numpy as np


def _discretize(weights: list, bits: int) -> list[int]:
    """Numerators over ``L 2**bits``, rounded on the cumulative sums (Qualtran)."""
    bin_count = 2**bits * len(weights)
    cumulative, total = [0.0], 0
    for weight in weights:
        total += weight
        cumulative.append(total)
    rounded = [int(math.floor(c / total * bin_count + 0.5)) for c in cumulative]
    return [b - a for a, b in zip(rounded, rounded[1:], strict=False)]


def _roulette(numerators: list[int]) -> tuple[list[int], list[int]]:
    """Alternates and keep numerators of one discretized distribution (Qualtran)."""
    weights = list(numerators)
    n = len(weights)
    target = sum(weights) // n
    alternates, keep = list(range(n)), [0] * n
    donor = 0
    for _ in range(2):
        for i in range(n):
            if weights[i] >= target:
                continue
            while weights[donor] <= target:
                donor += 1
            weights[donor] -= target - weights[i]
            alternates[i] = donor
            keep[i] = weights[i]
            weights[i] = target
    return alternates, keep


def alias_tables(weights, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """``(alt, keep)`` for one row of non-negative weights, ``keep`` in ``[0, 2**bits)``.

    Index ``l`` survives with probability ``keep[l] / 2**bits`` and is replaced by ``alt[l]``
    otherwise. An all-zero row loads nothing: zeros in both columns (the D1/Q1 inner rows).
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0:
        raise ValueError(f"weights must be a non-empty 1D array; got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError(f"weights must be non-negative; got {w}")
    if bits < 1:
        raise ValueError(f"bits must be >= 1; got {bits}")
    if not w.sum():
        return np.zeros(w.size, dtype=int), np.zeros(w.size, dtype=int)
    alternates, keep = _roulette(_discretize(list(w), bits))
    return np.array(alternates), np.array(keep)


def alias_probabilities(alt, keep, bits: int) -> np.ndarray:
    """The prepared distribution (spec 6.4):
    ``p_l = (keep_l + sum_{j: alt_j = l} (2**bits - keep_j)) / (L 2**bits)``."""
    keep = np.asarray(keep, dtype=float) / 2**bits
    p = keep.copy()
    np.add.at(p, np.asarray(alt), 1 - keep)
    return p / len(keep)


def inner_alias_row(signed_row, bits: int) -> tuple[np.ndarray, ...]:
    """``(alt, keep, sgn, d_sgn)`` of one signed inner row (spec 6.1, Signs).

    ``alt`` and ``keep`` are the alias tables of ``|w|``, ``sgn = [w < 0]`` and
    ``d_sgn = sgn XOR sgn[alt]``, so ``Z(sgn) CZ(less_than, d_sgn)`` gives every branch the sign
    of the index that survives the swap.
    """
    w = np.asarray(signed_row, dtype=float)
    alt, keep = alias_tables(np.abs(w), bits)
    sgn = (w < 0).astype(int)
    return alt, keep, sgn, sgn ^ sgn[alt]
```

- [ ] **Step 4: Write the angles module**

Create `src/dfthc/data/angles.py`:

```python
"""Givens angles: factorization of unit vectors, discretization, reconstruction (spec 6.3, 8.4).

A Givens ladder of ``N - 1`` rungs applies ``U(u)`` with ``U(u)^dag gamma_{0,f} U(u) =
sum_j u_j gamma_{j,f}``. Rung ``k`` acts on modes ``(q_k, q_{k+1})``; on ``{|01>, |10>}`` it is
``[[cos t, -sin t], [sin t, cos t]]``. Angles are stored as ``b_rot``-bit integers
``x = round(t / (2 pi) 2**(b_rot - 1)) mod 2**(b_rot - 1)``, and the rung realizes
``t = 4 pi x / 2**b_rot`` exactly. ``angles_from_vector``, ``discretize_angles`` and
``givens_vector`` (dfthcbe's ``vector_from_angles`` on radians) are ports of dfthcbe.
"""

from __future__ import annotations

import functools

import numpy as np


def angles_from_vector(u) -> np.ndarray:
    """``N - 1`` angles in ``[0, 2 pi)`` with ``givens_vector(angles) == u`` (one-column Givens
    factorization): ``atan2(|u[k+1:]|, u[k])`` for ``k < N - 2``, then ``atan2(u[-1], u[-2])``."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1 or u.size < 2:
        raise ValueError(f"u must be 1D of length N >= 2; got shape {u.shape}")
    norm = np.linalg.norm(u)
    if abs(norm - 1) > 1e-8:
        raise ValueError(f"u must have norm 1; got {norm}")
    theta = [np.arctan2(np.linalg.norm(u[k + 1:]), u[k]) for k in range(u.size - 2)]
    theta.append(np.arctan2(u[-1], u[-2]))
    return np.array(theta) % (2 * np.pi)


def givens_vector(theta) -> np.ndarray:
    """The unit vector of angles ``theta`` (radians), the inverse of :func:`angles_from_vector`."""
    theta = np.asarray(theta, dtype=float)
    u = np.zeros(theta.size + 1)
    u[0] = 1.0
    for k, phi in enumerate(theta):
        c, s = np.cos(phi), np.sin(phi)
        u[k], u[k + 1] = c * u[k] + s * u[k + 1], s * u[k] - c * u[k + 1]
    return u


def discretize_angles(theta, b_rot: int) -> np.ndarray:
    """The loaded integers ``round(theta / (2 pi) 2**(b_rot - 1)) mod 2**(b_rot - 1)``."""
    period = 2 ** (b_rot - 1)
    return np.round(np.asarray(theta, dtype=float) / (2 * np.pi) * period).astype(int) % period


def vector_from_angles(x, b_rot: int) -> np.ndarray:
    """``u~ = givens_vector(4 pi x / 2**b_rot)``: the direction the loaded integers realize."""
    return givens_vector(4 * np.pi * np.asarray(x, dtype=float) / 2**b_rot)


def _rung(theta: float, k: int, n: int) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    block = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)
    return functools.reduce(np.kron, [np.eye(2)] * k + [block] + [np.eye(2)] * (n - 2 - k))


def ladder_matrix(x, b_rot: int, N: int) -> np.ndarray:
    """The dense ``2**N x 2**N`` ladder ``U`` on one spin register for integer angles ``x``.

    ``U = G_0 G_1 ... G_{N-2}``: rung ``N - 2`` acts first in time and rung 0 last, as in
    dfthcbe's ``GivensRotationLadder``. Then ``U^dag gamma_{0,f} U = sum_j u~_j gamma_{j,f}`` with
    ``u~ = vector_from_angles(x, b_rot)`` (qubit 0 most significant, spec 6.2).
    """
    x = np.asarray(x)
    if x.shape != (N - 1,):
        raise ValueError(f"need N - 1 = {N - 1} angles; got shape {x.shape}")
    theta = 4 * np.pi * x.astype(float) / 2**b_rot
    ladder = np.eye(2**N)
    for k in range(N - 1):
        ladder = ladder @ _rung(theta[k], k, N)
    return ladder
```

- [ ] **Step 5: Write `WalkTables`**

Create `src/dfthc/data/tables.py`:

```python
"""``WalkTables``: every table the circuit loads, as compilable nested tuples (spec 8.4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from dfthc.data.alias import alias_tables, inner_alias_row
from dfthc.data.angles import angles_from_vector, discretize_angles
from dfthc.data.factors import DFTHCFactors
from dfthc.data.sos import inner_weights, outer_weights
from dfthc.params import WalkParams

INT_BOUND = 2**31
"""Every table entry is a nonnegative int below this bound (L-24: compilable ints are int64, and
operators compose wide entries as 32-bit words)."""


def _freeze(values):
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, list | tuple):
        return tuple(_freeze(value) for value in values)
    return int(values)


def _ints(values):
    if isinstance(values, tuple):
        for value in values:
            yield from _ints(value)
    else:
        yield values


@dataclass(frozen=True)
class WalkTables:
    """The alias and angle tables of one walk, as nested tuples of ints (frozen, hashable).

    - ``outer_alt``, ``outer_keep`` ``(X_o,)``: outer alias tables, ``b_k1``-bit keep values.
    - ``inner_alt``, ``inner_keep``, ``inner_sgn``, ``inner_d_sgn`` ``(R C, B + 1)``: inner alias
      tables per SF row ``x_o = r C + c``, ``b_k2``-bit keep values, and the sign columns.
    - ``tail_angles`` ``(N, N - 1)``: the ``N+`` D1 rows (``u_plus``), then the Q1 rows.
    - ``sf_angles`` ``(R, B, N - 1)``: the angles of ``U[r, b]``.

    Angles are ``b_rot``-bit integers (spec 6.3). Operators take :meth:`as_compilable`, the eight
    fields in declaration order (a tuple, the compilable type the K-1 stand-in exercised), and
    rebuild the object with :meth:`from_compilable`.
    """

    outer_alt: tuple
    outer_keep: tuple
    inner_alt: tuple
    inner_keep: tuple
    inner_sgn: tuple
    inner_d_sgn: tuple
    tail_angles: tuple
    sf_angles: tuple

    def __post_init__(self):
        for field in fields(self):
            frozen = _freeze(getattr(self, field.name))
            bad = [v for v in _ints(frozen) if not 0 <= v < INT_BOUND]
            if bad:
                raise ValueError(f"{field.name} entries must be nonnegative ints below 2**31; "
                                 f"got {bad[0]}")
            object.__setattr__(self, field.name, frozen)

    @classmethod
    def from_factors(cls, factors: DFTHCFactors, params: WalkParams) -> WalkTables:
        """The tables of ``factors`` at the precisions of ``params`` (sizes must match)."""
        f, p = factors, params
        if (p.N, p.R, p.B, p.C, p.n_plus) != (f.N, f.R, f.B, f.C, f.n_plus):
            raise ValueError(f"params (N, R, B, C, n_plus) = {(p.N, p.R, p.B, p.C, p.n_plus)} do "
                             f"not match the factors {(f.N, f.R, f.B, f.C, f.n_plus)}")
        outer_alt, outer_keep = alias_tables(outer_weights(f), p.b_k1)
        inner = [inner_alias_row(row, p.b_k2) for row in inner_weights(f)]

        def angles(u):
            return discretize_angles(angles_from_vector(u / np.linalg.norm(u)), p.b_rot)

        return cls(
            outer_alt=outer_alt, outer_keep=outer_keep,
            inner_alt=[row[0] for row in inner], inner_keep=[row[1] for row in inner],
            inner_sgn=[row[2] for row in inner], inner_d_sgn=[row[3] for row in inner],
            tail_angles=[angles(u) for u in np.concatenate([f.u_plus, f.u_minus])],
            sf_angles=[[angles(f.U[r, b]) for b in range(f.B)] for r in range(f.R)],
        )

    def as_compilable(self) -> tuple[tuple, ...]:
        """The eight fields in declaration order."""
        return tuple(getattr(self, field.name) for field in fields(self))

    @classmethod
    def from_compilable(cls, values: Sequence) -> WalkTables:
        """Inverse of :meth:`as_compilable`."""
        values = tuple(values)
        if len(values) != len(fields(cls)):
            raise ValueError(f"WalkTables needs {len(fields(cls))} fields; got {len(values)}")
        return cls(*values)
```

- [ ] **Step 6: Run the tests**

Run: `scripts/test.sh tests/data/test_alias.py tests/data/test_angles.py tests/data/test_tables.py -q`
Expected: `49 passed`.

- [ ] **Step 7: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `383 passed` (waves 1-a and 1-b merged: 181 + 95 + 58, plus 49).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add src/dfthc/data/alias.py src/dfthc/data/angles.py src/dfthc/data/tables.py \
        tests/data/test_alias.py tests/data/test_angles.py tests/data/test_tables.py
git commit -m "Add alias tables, Givens angles and WalkTables"
```

---

### Task 1A.5: Encoded-Hamiltonian model and failure signs

**Task id:** `1A.5` · **Branch:** `phase1/1A.5` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 1A.3, 1A.4

**Files:**
- Create: `src/dfthc/data/encoded.py`, `tests/data/test_encoded.py`

**Interfaces:**
- Consumes:
  - `WalkParams`, `lg`, `odd_part` and the presets;
  - `DFTHCFactors`, `load_factors`, `random_factors`;
  - `normalization`, `e_sos_walk`, `h_walk_dense`, `h_sa_dense`;
  - `alias_probabilities`, `vector_from_angles` and `WalkTables`;
  - `dfthc.data.fermion.annihilation` and `MAX_ORBITALS`.
- Produces, in `dfthc.data.encoded`:
  - `encoded_block(params, tables, factors) -> tuple[ndarray, float]`: `(H~_SA, Lambda~)` with `<0|BE|0> = H~_SA / Lambda~ - I` (spec 8.5), for `N <= 4`. Here `H~_SA = H_walk - E_SoS^walk` of the encoded factors and `Lambda~ = Lambda / (1 - P_fail(x_o))`.
  - `encoded_factors(params, tables, factors) -> DFTHCFactors`, in dfthcbe's convention.
  - `uniform_failure_probability(L, s) -> float`, a port of dfthcbe's `failure_probability`. Qualtran's fixed-point conversion truncates, so the angle is `floor(angle / (2 pi) 2**s)`.
  - `dfthcbe_failure_signs(params, tables) -> ndarray (RC,)`, a port including `_leaf`.
  - `our_failure_signs(params, tables) -> ndarray (RC,)`, all +1 (spec 8.5).

**Acceptance tests** (15 tests):
- The two constructions of `H~_SA` (`H_walk - E_SoS^walk` and the generators, both of the encoded factors) agree to 1e-10 at both test presets.
- A third construction from the circuit's structure equals `H~_SA / Lambda~ - I` to 1e-10. It is `(1 - P_fail(x_o)) sum_x p_x avg_sigma 2 A_x^dag A_x - I`, with `A_x` the failure-weighted inner LCU along the rounded directions. It is checked at both test presets, where `b`'s uniform preparation fails, and at an `X_o = 5` shape, where `x_o`'s fails.
- `H~_SA` approaches `H_SA`: at `b_rot = b_k1 = b_k2 = s_u = 12` the spectral error is below 1e-2; at 4 bits it is above.
- The encoded factors are valid factors of the same shape.
- `uniform_failure_probability` reproduces six values recorded from dfthcbe at the pinned commit (Qualtran 0.7.0) to relative 1e-12, and gives 0 when `L` is a power of two.
- Failure signs, spec 8.5 and spec 12.1 T1 and T3 item 3:
  - ours are +1;
  - the dfthcbe port equals ours when `2**w_b <= ceil((B+1)/2**k2) 2**k2` (cases `(3, 1)`, `(6, 2)`, `(28, 4)`);
  - the port differs as documented at `(5, 1)` and `(7, 0)`, where failed addresses read the identity entry `B`.
- `encoded_block` refuses `N > 4`.

Dry-run cross-check, not part of the suite: on 2026-11-12, in a scratch environment with dfthcbe at the pinned commit, every encoded-factor array and Λ̃ agreed exactly with dfthcbe at the two test presets and the `X_o = 5` shape. `dfthcbe_failure_signs` agreed with dfthcbe's `failure_signs` in all ten cases above, and `uniform_failure_probability` agreed over `L` in [2, 100] and `s` in [3, 15].

**Definition of done:** `15 passed`; the per-commit tier passes; `ruff check .` clean; one commit. This model is the E-1 oracle (risk K-5).

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_encoded.py`:

```python
"""The encoded-Hamiltonian model and the failure signs (spec 8.5; spec 12.1 T1 and T3 item 3)."""

import dataclasses

import numpy as np
import pytest

from dfthc.data.alias import alias_probabilities
from dfthc.data.angles import vector_from_angles
from dfthc.data.encoded import (
    dfthcbe_failure_signs,
    encoded_block,
    encoded_factors,
    our_failure_signs,
    uniform_failure_probability,
)
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.fermion import annihilation
from dfthc.data.sos import h_sa_dense, normalization
from dfthc.data.tables import WalkTables
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, WalkParams

TEST_PRESETS = [(H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"), (H2_631G_TEST, "h2_631g_R2_B2_C2")]
X_O_FAILS = WalkParams(N=2, R=3, B=1, C=1, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
"""X_o = 5, so the uniform preparation of x_o fails too (dfthcbe's "x_o_fails" shape)."""


def _setup(params, factors):
    return params, WalkTables.from_factors(factors, params), factors


def _lcu_block(params, tables, factors):
    """<0|BE|0> from the circuit's structure, without walk_integrals (spec 3.3 and 8.5):
    (1 - P_x) sum_x p_x avg_sigma (2 A_x^dag A_x) - I, with A_SF = a_B + sum_b a_b M_b,
    M_b = (i/2) sum_sigma gamma_0 gamma_1 along u~_b, a_b the failure-weighted signed inner
    probabilities, and A = a_u (D1) or a_u^dag (Q1) on one spin."""
    p, t, n = params, tables, params.N
    ann = np.array([[annihilation(j, s, n) for s in (0, 1)] for j in range(n)])
    eye = np.eye(4**n)

    def a_u(u, spin):
        return np.tensordot(u, ann[:, spin], axes=1)

    def pair(u):
        terms = [a_u(u, s) for s in (0, 1)]
        return sum(0.5j * (a + a.conj().T) @ (-1j * (a - a.conj().T)) for a in terms)

    outer = alias_probabilities(t.outer_alt, t.outer_keep, p.b_k1)
    fail_b = uniform_failure_probability(p.B + 1, p.s_u)
    block = 0
    for x in range(p.X_o):
        if x < p.RC:
            q = alias_probabilities(t.inner_alt[x], t.inner_keep[x], p.b_k2)
            amps = (1 - fail_b) * (1 - 2 * np.array(t.inner_sgn[x])) * q
            amps[p.B] += fail_b * our_failure_signs(p, t)[x]
            u = [vector_from_angles(t.sf_angles[x // p.C][b], p.b_rot) for b in range(p.B)]
            a = amps[p.B] * eye + sum(amps[b] * pair(u[b]) for b in range(p.B))
            block = block + outer[x] * 2 * a.conj().T @ a
        else:
            j = x - p.RC
            u = vector_from_angles(t.tail_angles[j], p.b_rot)
            ops = [a_u(u, s) if j < p.n_plus else a_u(u, s).conj().T for s in (0, 1)]
            block = block + outer[x] * sum(o.conj().T @ o for o in ops)
    return (1 - uniform_failure_probability(p.X_o, p.s_u)) * block - eye


@pytest.mark.parametrize(("params", "system"), TEST_PRESETS)
def test_the_two_constructions_agree(params, system):
    """Spec 8.6 check 2 for the encoded factors: H_walk - E_SoS^walk equals the generators."""
    p, t, f = _setup(params, load_factors(system))
    h_tilde, _ = encoded_block(p, t, f)
    assert np.allclose(h_tilde, h_sa_dense(encoded_factors(p, t, f)), atol=1e-10)


LCU_CASES = {
    "h2_sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "h2_631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "x_o_fails": lambda: (X_O_FAILS, random_factors(2, 3, 1, 1, n_plus=1, seed=3)),
}
FAILS = {"h2_sto3g": (False, True), "h2_631g": (False, True), "x_o_fails": (True, False)}
"""Whether the uniform preparation of x_o and of b fails: B + 1 = 3 fails, X_o = 5 fails."""


@pytest.mark.parametrize("case", list(LCU_CASES))
def test_block_equals_the_lcu_structure(case):
    p, t, f = _setup(*LCU_CASES[case]())
    h_tilde, lambda_tilde = encoded_block(p, t, f)
    failures = [uniform_failure_probability(p.X_o, p.s_u),
                uniform_failure_probability(p.B + 1, p.s_u)]
    assert tuple(failure > 0 for failure in failures) == FAILS[case]
    assert lambda_tilde == pytest.approx(normalization(f) / (1 - failures[0]), rel=1e-14)
    assert np.allclose(h_tilde / lambda_tilde - np.eye(len(h_tilde)), _lcu_block(p, t, f),
                       atol=1e-10)


@pytest.mark.parametrize(("params", "system"), TEST_PRESETS)
def test_encoded_hamiltonian_approaches_h_sa(params, system):
    f = load_factors(system)
    errors = {}
    for bits in (4, 12):
        p = dataclasses.replace(params, b_rot=bits, b_k1=bits, b_k2=bits, s_u=bits)
        h_tilde, _ = encoded_block(p, WalkTables.from_factors(f, p), f)
        errors[bits] = np.linalg.norm(h_tilde - h_sa_dense(f), 2)
    assert errors[12] < 1e-2 < errors[4]


def test_encoded_factors_are_valid_factors_of_the_same_shape():
    p, t, f = _setup(H2_631G_TEST, load_factors("h2_631g_R2_B2_C2"))
    enc = encoded_factors(p, t, f)
    enc.validate()
    assert (enc.N, enc.R, enc.B, enc.C, enc.n_plus) == (f.N, f.R, f.B, f.C, f.n_plus)
    assert enc.eta == f.eta


def test_uniform_failure_probability_reproduces_dfthcbe():
    """Values from dfthcbe's UniformViaPhaseGradient.failure_probability at the pinned commit
    (Qualtran 0.7.0), recorded while writing this plan."""
    recorded = {(3, 3): 0.07858495705504472, (5, 3): 0.1506780981594029,
                (324, 7): 0.0006445392838034328, (28, 7): 0.0007020808150612868,
                (7, 4): 0.029699337423164093, (25, 12): 9.387453336647742e-07}
    for (L, s), value in recorded.items():
        assert uniform_failure_probability(L, s) == pytest.approx(value, rel=1e-12, abs=1e-18)
    assert uniform_failure_probability(4, 3) == uniform_failure_probability(64, 7) == 0.0


def _signs_case(B, k2, seed):
    params = WalkParams(N=2, R=2, B=B, C=1, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=1, k2=k2,
                        ke1=1, ke2=1)
    factors = random_factors(2, 2, B, 1, n_plus=1, seed=seed)
    return params, WalkTables.from_factors(factors, params)


@pytest.mark.parametrize(("B", "k2"), [(2, 1), (5, 2), (27, 4)])
def test_failure_signs_agree_when_the_last_block_covers_the_register(B, k2):
    params, tables = _signs_case(B, k2, seed=11)
    assert 2**params.w_b <= -(-(B + 1) // 2**k2) * 2**k2
    assert np.array_equal(our_failure_signs(params, tables), np.ones(params.RC))
    assert np.allclose(dfthcbe_failure_signs(params, tables), our_failure_signs(params, tables))


@pytest.mark.parametrize(("B", "k2", "reads"), [(4, 1, (1, 1, "B")), (6, 0, ("B",))])
def test_failure_signs_differ_as_documented(B, k2, reads):
    """(B+1, k2) = (5, 1): failed b = 5, 6, 7 read padding, entry 4 (= B) and padding in dfthcbe;
    (7, 0): failed b = 7 reads entry 6 (= B). Ours read padding everywhere: +1."""
    params, tables = _signs_case(B, k2, seed=5)
    kept = np.array(tables.inner_keep)[:, B] / 2**params.b_k2
    sgn, d_sgn = np.array(tables.inner_sgn)[:, B], np.array(tables.inner_d_sgn)[:, B]
    entry_sign = (1 - 2 * sgn) * (kept + (1 - kept) * (1 - 2 * d_sgn))
    expected = np.mean([entry_sign if r == "B" else np.ones(params.RC) for r in reads], axis=0)
    port = dfthcbe_failure_signs(params, tables)
    assert np.allclose(port, expected)
    assert not np.allclose(port, our_failure_signs(params, tables))


def test_encoded_block_needs_dense_sizes():
    f = random_factors(5, 2, 2, 1, n_plus=2, seed=0)
    p = dataclasses.replace(H2_STO3G_TEST, N=5, n_plus=2)
    with pytest.raises(ValueError, match="N <= 4"):
        encoded_block(p, WalkTables.from_factors(f, p), f)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/data/test_encoded.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.data.encoded'`.

- [ ] **Step 3: Write `dfthc.data.encoded`**

Create `src/dfthc/data/encoded.py`:

```python
"""The Hamiltonian and normalization the circuit encodes: the encoded-Hamiltonian model (spec 8.5).

The circuit block-encodes ``<0|BE|0> = H~_SA / Lambda~ - I``. ``H~_SA`` is the sum-of-squares
Hamiltonian of the *encoded factors*: the prepared alias probabilities instead of the weights, the
sign columns, the directions ``u~`` of the rounded angles, the identity term ``b = B``, and the
failed uniform preparation of ``b``. A failed uniform preparation of ``x_o`` makes ``REF_T2`` act
as ``-I`` there, which renormalizes: ``Lambda~ = Lambda / (1 - P_fail(x_o))``. The construction is
dfthcbe's ``encoded_factors``, with our failure signs (all +1, spec 8.5) in place of its
``failure_signs``. Only the walk-frame offset enters ``H~_SA``. This model is the oracle of the
end-to-end check E-1 (spec 12.4).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import hadamard

from dfthc.data.alias import alias_probabilities
from dfthc.data.angles import vector_from_angles
from dfthc.data.factors import DFTHCFactors
from dfthc.data.fermion import MAX_ORBITALS
from dfthc.data.sos import e_sos_walk, h_walk_dense, normalization
from dfthc.data.tables import WalkTables
from dfthc.params import WalkParams, lg, odd_part


def uniform_failure_probability(L: int, s: int) -> float:
    """``P(succ = 0)`` of the flagged uniform preparation over ``L`` values with ``s``-bit
    rotations: a port of dfthcbe's ``UniformViaPhaseGradient.failure_probability``.

    With ``L = 2**eta l``, ``l`` odd and ``n = lg(l)``, one round of amplitude amplification on
    the top ``n`` qubits uses the angle ``arccos(1 - 2**floor(log2 l) / l)``, loaded as
    ``k = floor(angle / (2 pi) 2**s)``: Qualtran's ``QFxp(s, s).to_fixed_width_int`` truncates.
    The failure probability is the weight left on ``t >= l``; it is 0 if ``l = 1``.
    """
    odd = odd_part(L)
    n = lg(odd)
    if not n:
        return 0.0
    angle = np.arccos(1 - 2 ** np.floor(np.log2(odd)) / odd)
    k = int(np.floor(angle / (2 * np.pi) * 2**s)) % 2**s
    phase, size = np.exp(2j * np.pi * k / 2**s), 2**n
    good, h = np.arange(size) < odd, hadamard(size) / np.sqrt(size)
    psi = h @ (np.where(good, phase, 1) / np.sqrt(size))
    psi[0] *= phase
    return float(np.sum(np.abs(h @ psi)[~good] ** 2))


def our_failure_signs(params: WalkParams, tables: WalkTables) -> np.ndarray:
    """The sign every failed ``b`` of an SF row carries: +1, because our exact read loads zeros
    there, so the branch acts as ``+I`` (spec 8.5)."""
    return np.ones(params.RC)


def _leaf(v: int, n: int, length: int) -> int:
    """The leaf that ``v`` reaches in Qualtran's truncated unary iteration over ``[0, length)``
    on ``n`` bits, also for ``v >= length`` (dfthcbe's ``_leaf``)."""
    lo = 0
    for j in reversed(range(n)):
        if lo + 2**j < length and (v >> j) & 1:
            lo += 2**j
    return lo


def dfthcbe_failure_signs(params: WalkParams, tables: WalkTables) -> np.ndarray:
    """dfthcbe's ``InnerPrepare.failure_signs``, ported for comparison (spec 8.5).

    Per SF row, the mean over the failed ``b in [B+1, 2**w_b)`` of the keep-averaged sign of the
    entry that ``b`` reaches through the truncated iteration, or +1 past the table. It equals
    :func:`our_failure_signs` when ``2**w_b <= ceil((B+1) / 2**k2) 2**k2``.
    """
    p = params
    failed = range(p.B + 1, 2**p.w_b)
    if not failed:
        return np.ones(p.RC)
    kept = np.array(tables.inner_keep) / 2**p.b_k2
    sgn, d_sgn = np.array(tables.inner_sgn), np.array(tables.inner_d_sgn)
    signs = (1 - 2 * sgn) * (kept + (1 - kept) * (1 - 2 * d_sgn))
    blocks = -(-(p.B + 1) // 2**p.k2)
    loaded = [_leaf(b >> p.k2, p.w_b - p.k2, blocks) * 2**p.k2 + b % 2**p.k2 for b in failed]
    return np.mean([signs[:, e] if e <= p.B else np.ones(p.RC) for e in loaded], axis=0)


def encoded_factors(params: WalkParams, tables: WalkTables,
                    factors: DFTHCFactors) -> DFTHCFactors:
    """The factors the circuit encodes, in dfthcbe's convention (its ``encoded_factors``).

    With ``Lambda`` of the input factors, ``p`` and ``q`` the prepared outer and inner
    probabilities and ``P_b`` the failure probability of the uniform preparation of ``b``:
    ``a_b = (1 - P_b) sign(w_b) q_b``, ``a_B += P_b sigma_fail`` (``sigma_fail = +1``),
    ``w~_b = a_b sqrt(4 Lambda p_rc)``, ``w~_pm = Lambda p``, and the unit vectors are
    ``vector_from_angles`` of the loaded angles. ``h1``, ``h_sym``, ``beta_1``, ``e_upper`` and
    ``eta`` are carried over; ``h1_shifted``, ``lambda_h1`` and ``lambda_h2`` are recomputed.
    """
    p, t = params, tables
    lam = normalization(factors)
    outer = alias_probabilities(t.outer_alt, t.outer_keep, p.b_k1)
    inner = np.array([alias_probabilities(alt, keep, p.b_k2)
                      for alt, keep in zip(t.inner_alt, t.inner_keep, strict=True)])
    fail = uniform_failure_probability(p.B + 1, p.s_u)
    amps = (1 - fail) * (1 - 2 * np.array(t.inner_sgn)) * inner
    amps[:, p.B] += fail * our_failure_signs(p, t)
    w = (amps * np.sqrt(4 * lam * outer[:p.RC])[:, None]).reshape(p.R, p.C, p.B + 1)
    w_tail = lam * outer[p.RC:]
    u_tail = np.array([vector_from_angles(row, p.b_rot) for row in t.tail_angles])
    U = np.array([[vector_from_angles(row, p.b_rot) for row in plane] for plane in t.sf_angles])
    w_plus, w_minus = w_tail[:p.n_plus], w_tail[p.n_plus:]
    u_plus, u_minus = u_tail[:p.n_plus], u_tail[p.n_plus:]
    W, w_B = w[..., :p.B].transpose(0, 2, 1), w[..., p.B]
    return DFTHCFactors(
        U=U, W=W, w_B=w_B, w_plus=w_plus, u_plus=u_plus, w_minus=w_minus, u_minus=u_minus,
        h1=factors.h1, h_sym=factors.h_sym,
        h1_shifted=u_plus.T @ np.diag(w_plus) @ u_plus - u_minus.T @ np.diag(w_minus) @ u_minus,
        beta_1=factors.beta_1, lambda_h1=float(w_tail.sum()),
        lambda_h2=float((np.abs(w).sum(axis=-1) ** 2).sum() / 4), e_upper=factors.e_upper,
        eta=factors.eta,
    )


def encoded_block(params: WalkParams, tables: WalkTables,
                  factors: DFTHCFactors) -> tuple[np.ndarray, float]:
    """``(H~_SA, Lambda~)`` with ``<0|BE|0> = H~_SA / Lambda~ - I`` (spec 8.5), for ``N <= 4``.

    ``H~_SA = H_walk - E_SoS^walk`` of :func:`encoded_factors` (a dense ``4**N`` matrix), and
    ``Lambda~ = Lambda / (1 - P_fail(x_o))`` with ``Lambda`` of the input factors.
    """
    if params.N > MAX_ORBITALS:
        raise ValueError(f"the dense encoded model needs N <= {MAX_ORBITALS}; got N = {params.N}")
    enc = encoded_factors(params, tables, factors)
    h_tilde = h_walk_dense(enc) - e_sos_walk(enc) * np.eye(4**params.N)
    lambda_tilde = normalization(factors) / (1 - uniform_failure_probability(params.X_o,
                                                                              params.s_u))
    return h_tilde, lambda_tilde
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/data/test_encoded.py -q`
Expected: `15 passed`.

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `398 passed` plus the per-commit tests of Task 1C.1 (waves 1-a to 1-c merged: 181 + 95 + 58 + 49, plus 15).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/data/encoded.py tests/data/test_encoded.py
git commit -m "Add the encoded-Hamiltonian model and failure signs"
```

---

### Task 1A.6: dfthcbe reference fixtures (spec 8.6, check 4)

**Task id:** `1A.6` · **Branch:** `phase1/1A.6` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 1A.4, 1A.5

**Files:**
- Create: `scripts/make_dfthcbe_fixtures.py`, `src/dfthc/testing/fixtures.py`, `tests/data/test_dfthcbe_fixtures.py`
- Create (generated, Step 7): `tests/fixtures/dfthcbe/{provenance.json,tables_h2_sto3g_test.json,tables_h2_631g_test.json,encoded_h2_sto3g_test.npz,femoco_block_toffolis.json,requirements-ref.lock}`

**Interfaces:**
- Consumes:
  - the presets;
  - `load_factors`, `WalkTables` and `encoded_block`;
  - in `.venv-ref` only: dfthcbe at `930255020cec30172e0a7d17c0071d9476a687d1`, installed from its tarball in `<main checkout>/.scratch/dfthcbe-ref`, and Qualtran 0.7.0.
- Produces:
  - `scripts/make_dfthcbe_fixtures.py [--out DIR]` (default `tests/fixtures/dfthcbe`). It imports Qualtran and dfthcbe only inside its functions, so its constants `DFTHCBE_COMMIT`, `FIXTURES`, `FEMOCO_BLOCKS` and `REPO` can be imported in the main environment. It writes these fixtures:
    - the alias and angle tables of both test presets, keyed by the `WalkTables` field names;
    - dfthcbe's `<0|BE|0>` for `H2_STO3G_TEST`, by exact tensor contraction with `exact_erase=True` (`block`), with `lambda_tilde` and dfthcbe's `H_SA` of its `encoded_factors` (`h_sa`);
    - dfthcbe's Toffolis per block at `FEMOCO54` (`k_e1 = 4`, `k_e2 = 2`, `s = 7`), from its symbolic call graph;
    - the provenance: commit, versions and parameters.
  - `dfthc.testing.fixtures` (spec 5.2): `FIXTURE_DIR`, `SKIP_REASON`, `require_fixture(name) -> Path` and `load_dfthcbe_fixture(name) -> dict | np.ndarray`. `require_fixture` calls `pytest.skip` with the reason "dfthcbe fixtures not generated: run scripts/make_dfthcbe_fixtures.py in .venv-ref" if the file is missing. `load_dfthcbe_fixture` returns a dict for `.json` and `.npz` files and an array for `.npy` files.

**Acceptance tests** (7 tests in `tests/data/test_dfthcbe_fixtures.py`):
- `require_fixture` skips with the explicit reason, and loads a present file.
- The script writes exactly the fixtures the tests read, and names the pinned commit.
- Our tables equal the fixture tables exactly (integers), at both test presets.
- `encoded_block` equals the fixtures to 1e-10: dfthcbe's contracted block, its `H_SA`, and `lambda_tilde` (relative 1e-12).
- The FeMoco counts load and add up: the blocks sum to the block encoding, the walk is the block encoding plus `RefAB`, and the controlled walk is the walk plus 1. Task 4.3 compares them with ours.
- The provenance names commit `930255020cec30172e0a7d17c0071d9476a687d1` and the three presets' parameters.

**Definition of done:** before the fixtures exist, 2 passed and 5 skipped with the explicit reason (first commit). After the orchestrator's Step 7, all 7 pass and none is skipped, on the development machine, with the fixtures committed (second commit). The per-commit tier passes and `ruff check .` is clean.

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_dfthcbe_fixtures.py`:

```python
"""Our tables and encoded model against dfthcbe's reference fixtures (spec 8.6, check 4)."""

import dataclasses
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from dfthc.data.encoded import encoded_block
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST
from dfthc.testing import fixtures
from dfthc.testing.fixtures import SKIP_REASON, load_dfthcbe_fixture, require_fixture

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_dfthcbe_fixtures.py"
COMMIT = "930255020cec30172e0a7d17c0071d9476a687d1"
TABLE_FIELDS = [field.name for field in dataclasses.fields(WalkTables)]


def _script():
    """The fixture script's module; it imports Qualtran and dfthcbe only inside its functions."""
    spec = importlib.util.spec_from_file_location("make_dfthcbe_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_fixture_skips_with_the_explicit_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "FIXTURE_DIR", tmp_path)
    with pytest.raises(pytest.skip.Exception, match=re.escape(SKIP_REASON)):
        require_fixture("missing.json")
    (tmp_path / "present.json").write_text('{"a": 1}')
    assert require_fixture("present.json") == tmp_path / "present.json"
    assert load_dfthcbe_fixture("present.json") == {"a": 1}


def test_the_script_writes_the_fixtures_these_tests_read():
    script = _script()
    assert script.FIXTURES == ("tables_h2_sto3g_test.json", "tables_h2_631g_test.json",
                               "encoded_h2_sto3g_test.npz", "femoco_block_toffolis.json",
                               "provenance.json")
    assert script.DFTHCBE_COMMIT == COMMIT
    assert fixtures.FIXTURE_DIR == script.REPO / "tests" / "fixtures" / "dfthcbe"


@pytest.mark.parametrize(("name", "params", "system"), [
    ("tables_h2_sto3g_test.json", H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"),
    ("tables_h2_631g_test.json", H2_631G_TEST, "h2_631g_R2_B2_C2"),
])
def test_our_tables_equal_dfthcbe_exactly(name, params, system):
    fixture = load_dfthcbe_fixture(name)
    assert sorted(fixture) == sorted(TABLE_FIELDS)
    reference = WalkTables.from_compilable([fixture[field] for field in TABLE_FIELDS])
    assert WalkTables.from_factors(load_factors(system), params) == reference


def test_encoded_block_equals_dfthcbe():
    """dfthcbe's <0|BE|0> by exact contraction equals H~_SA / Lambda~ - I to 1e-10."""
    fixture = load_dfthcbe_fixture("encoded_h2_sto3g_test.npz")
    factors = load_factors("h2_sto3g_R2_B2_C1")
    h_tilde, lambda_tilde = encoded_block(
        H2_STO3G_TEST, WalkTables.from_factors(factors, H2_STO3G_TEST), factors)
    assert lambda_tilde == pytest.approx(float(fixture["lambda_tilde"]), rel=1e-12)
    assert np.allclose(h_tilde, fixture["h_sa"], atol=1e-10)
    assert np.allclose(h_tilde / lambda_tilde - np.eye(16), fixture["block"], atol=1e-10)


def test_femoco_block_counts_are_loadable():
    """Task 4.3 compares these with ours; here they only need to load and add up."""
    counts = load_dfthcbe_fixture("femoco_block_toffolis.json")
    assert list(counts) == list(_script().FEMOCO_BLOCKS)
    assert all(type(value) is int and value > 0 for value in counts.values())
    blocks = (counts["outer_prepare"] + counts["outer_unprepare"]
              + 2 * (counts["inner_prepare"] + counts["inner_unprepare"])
              + counts["inner_select"] + counts["inner_select_adjoint"] + counts["ref_t2"])
    assert counts["block_encoding"] == blocks
    assert counts["walk"] == counts["block_encoding"] + counts["ref_ab"]
    assert counts["controlled_walk"] == counts["walk"] + 1


def test_provenance_names_the_commit_and_parameters():
    provenance = load_dfthcbe_fixture("provenance.json")
    assert provenance["dfthcbe_commit"] == COMMIT
    assert {"dfthcbe", "qualtran", "numpy", "python"} <= set(provenance["versions"])
    for name, preset in [("H2_STO3G_TEST", H2_STO3G_TEST), ("H2_631G_TEST", H2_631G_TEST),
                         ("FEMOCO54", FEMOCO54)]:
        assert provenance["parameters"][name] == dataclasses.asdict(preset)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/data/test_dfthcbe_fixtures.py -q`
Expected: FAIL at collection, `1 error`: `ImportError: cannot import name 'fixtures' from 'dfthc.testing'`.

- [ ] **Step 3: Write the fixture loaders**

Create `src/dfthc/testing/fixtures.py` (`src/dfthc/testing/__init__.py` belongs to Task 0.2 and stays unchanged):

```python
"""Loaders for the dfthcbe reference fixtures (spec sections 5.2 and 8.6, check 4).

``scripts/make_dfthcbe_fixtures.py`` writes the fixtures in ``.venv-ref``; tests read them here.
A missing fixture skips the test with :data:`SKIP_REASON`, so checks 1 and 2 of spec 8.6 and the
end-to-end simulation still run where ``.venv-ref`` cannot be built.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "dfthcbe"
SKIP_REASON = "dfthcbe fixtures not generated: run scripts/make_dfthcbe_fixtures.py in .venv-ref"


def require_fixture(name: str) -> Path:
    """The path of fixture ``name``; skips the calling test with :data:`SKIP_REASON` if missing."""
    path = FIXTURE_DIR / name
    if not path.is_file():
        pytest.skip(SKIP_REASON)
    return path


def load_dfthcbe_fixture(name: str) -> dict | np.ndarray:
    """A fixture's content: a dict for ``.json`` and ``.npz`` files, an array for ``.npy``."""
    path = require_fixture(name)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    if path.suffix == ".npz":
        with np.load(path) as data:
            return {key: np.array(data[key]) for key in data.files}
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"unknown fixture format: {name}")
```

- [ ] **Step 4: Write the generation script**

Create `scripts/make_dfthcbe_fixtures.py`:

```python
"""Generate the dfthcbe reference fixtures of spec 8.6, check 4. Runs only in ``.venv-ref``.

Usage, from the repository root (environment recipe: docs/agents/HANDOFF.md, section 3.4):

    .venv-ref/bin/python scripts/make_dfthcbe_fixtures.py [--out tests/fixtures/dfthcbe]

Writes into ``--out``:

- ``tables_h2_sto3g_test.json``, ``tables_h2_631g_test.json``: dfthcbe's alias and angle tables at
  the bits of ``H2_STO3G_TEST`` and ``H2_631G_TEST``, keyed by the ``WalkTables`` field names;
- ``encoded_h2_sto3g_test.npz``: dfthcbe's ``<0|BE|0>`` of ``H2_STO3G_TEST`` by exact tensor
  contraction with ``exact_erase=True`` (``block``), its ``Lambda`` (``lambda_tilde``) and the
  ``H_SA`` of its ``encoded_factors`` (``h_sa``);
- ``femoco_block_toffolis.json``: dfthcbe's Toffolis per block at ``FEMOCO54`` (``k_e1 = 4``,
  ``k_e2 = 2``, ``s = 7``), from its symbolic call graph, so no FeMoco table is built;
- ``provenance.json``: the dfthcbe commit, package versions and parameters.

The heavy imports (Qualtran, dfthcbe) happen inside the functions, so the test suite can import
this module's constants in the main environment.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST  # noqa: E402

DFTHCBE_COMMIT = "930255020cec30172e0a7d17c0071d9476a687d1"
DATA = REPO / "data" / "dfthc"
KEYS = ("U", "W", "w_B", "w_plus", "u_plus", "w_minus", "u_minus")
TABLE_FIXTURES = {
    "tables_h2_sto3g_test.json": (H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"),
    "tables_h2_631g_test.json": (H2_631G_TEST, "h2_631g_R2_B2_C2"),
}
ENCODED_FIXTURE = "encoded_h2_sto3g_test.npz"
FEMOCO_FIXTURE = "femoco_block_toffolis.json"
PROVENANCE = "provenance.json"
FIXTURES = (*TABLE_FIXTURES, ENCODED_FIXTURE, FEMOCO_FIXTURE, PROVENANCE)
FEMOCO_BLOCKS = ("outer_prepare", "outer_unprepare", "inner_prepare", "inner_unprepare",
                 "inner_select", "inner_select_adjoint", "ref_t2", "ref_ab", "block_encoding",
                 "controlled_block_encoding", "walk", "controlled_walk")


def _factors(system: str) -> dict:
    with np.load(DATA / system / "factors.npz") as data:
        return {key: np.array(data[key]) for key in KEYS}


def block_encoding(params, system: str, exact_erase: bool):
    """dfthcbe's ``DFTHCBlockEncoding.from_factors`` at the bits and block sizes of ``params``."""
    from dfthcbe import DFTHCBlockEncoding

    return DFTHCBlockEncoding.from_factors(
        **_factors(system), b_rot=params.b_rot, b_k1=params.b_k1, b_k2=params.b_k2,
        k_1=params.k1, k_2=params.k2, k_e1=params.ke1, k_e2=params.ke2, s=params.s_u,
        exact_erase=exact_erase,
    )


def tables(params, system: str) -> dict:
    be = block_encoding(params, system, exact_erase=False)
    o, i, s = be.outer, be.inner, be.select
    return {"outer_alt": o.alt, "outer_keep": o.keep, "inner_alt": i.alt, "inner_keep": i.keep,
            "inner_sgn": i.sgn, "inner_d_sgn": i.d_sgn, "tail_angles": s.tail_data,
            "sf_angles": s.angle_data}


def signal_block(be) -> np.ndarray:
    """``<0|be|0>`` on the ancillas: a ``4**N`` matrix on ``(psi_up, psi_dn)``.

    A port of ``_signal_block`` in dfthcbe's ``tests/test_block_encoding.py``.
    """
    from dfthcbe.block_encoding import INNER_ANCILLAS, OUTER_ANCILLAS
    from qualtran import BloqBuilder, QAny
    from qualtran.bloqs.basic_gates import IntEffect, IntState
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState

    num_orb, b_rot = be.select.num_orb, be.select.b_rot
    bb = BloqBuilder()
    psi = {name: bb.add_register(name, QAny(num_orb)) for name in ("psi_up", "psi_dn")}
    width = {name: be.signature.get_left(name).total_bits()
             for name in OUTER_ANCILLAS + INNER_ANCILLAS}
    grad = PhaseGradientState(b_rot)
    out = bb.add_d(be, **psi, phase_grad=bb.add(grad),
                   **{name: bb.add(IntState(0, n)) for name, n in width.items()})
    bb.add(grad.adjoint(), phase_grad=out.pop("phase_grad"))
    for name, n in width.items():
        bb.add(IntEffect(0, n), val=out.pop(name))
    return bb.finalize(**out).tensor_contract()


def encoded(params, system: str) -> dict:
    from dfthcbe import hamiltonian_matrix, walk_e_sos, walk_integrals

    be = block_encoding(params, system, exact_erase=True)
    enc = be.encoded_factors()
    h_sa = (hamiltonian_matrix(*walk_integrals(**enc))
            - walk_e_sos(enc["W"], enc["w_B"], enc["w_minus"]) * np.eye(4**params.N))
    with warnings.catch_warnings():
        # exact_erase replaces each measured erase by its read's inverse and says so each time
        warnings.filterwarnings("ignore", message="QROAMCleanHetero", category=UserWarning)
        block = signal_block(be)
    return {"block": block, "lambda_tilde": np.array(be.Lambda), "h_sa": h_sa}


def femoco_toffolis() -> dict[str, int]:
    """Toffolis per block (``Toffoli + CSwap + And + T/4`` of Qualtran's ``QECGatesCost``, as in
    dfthcbe's tests) of the symbolic walk, evaluated at ``FEMOCO54``."""
    import sympy
    from dfthcbe import DFTHCWalk
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    p = FEMOCO54
    symbols = sympy.symbols("N R B C b_rot b_k1 b_k2 k_1 k_2 k_e1 k_e2", positive=True,
                            integer=True)
    walk = DFTHCWalk.symbolic(*symbols, p.s_u)
    values = dict(zip(symbols, (p.N, p.R, p.B, p.C, p.b_rot, p.b_k1, p.b_k2, p.k1, p.k2, p.ke1,
                                p.ke2), strict=True))

    def toffoli(bloq) -> int:
        cost = get_cost_value(bloq, QECGatesCost())
        total = sympy.sympify(cost.toffoli + cost.cswap + cost.and_bloq + cost.t / 4)
        return int(total.subs(values))

    be = walk.block_encoding
    bloqs = (be.outer, be.outer.adjoint(), be.inner, be.inner.adjoint(), be.select,
             be.select.adjoint(), be.ref_t2, walk.ref_ab, be, be.controlled(), walk,
             walk.controlled())
    return {name: toffoli(bloq) for name, bloq in zip(FEMOCO_BLOCKS, bloqs, strict=True)}


def provenance() -> dict:
    versions = {name: importlib.metadata.version(name) for name in ("dfthcbe", "qualtran",
                                                                     "numpy", "sympy")}
    presets = {"H2_STO3G_TEST": H2_STO3G_TEST, "H2_631G_TEST": H2_631G_TEST, "FEMOCO54": FEMOCO54}
    return {
        "dfthcbe_commit": DFTHCBE_COMMIT,
        "versions": versions | {"python": platform.python_version()},
        "parameters": {name: dataclasses.asdict(p) for name, p in presets.items()},
        "systems": {name: system for name, (_, system) in TABLE_FIXTURES.items()}
        | {ENCODED_FIXTURE: "h2_sto3g_R2_B2_C1"},
        "encoded_block": "exact tensor contraction of DFTHCBlockEncoding.from_factors(..., "
                         "exact_erase=True) with every ancilla pinned to |0>",
        "femoco_counts": "DFTHCWalk.symbolic call graph evaluated at FEMOCO54; Toffolis = "
                         "Toffoli + CSwap + And + T/4 of QECGatesCost",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "tests" / "fixtures" / "dfthcbe")
    out = parser.parse_args(argv).out
    out.mkdir(parents=True, exist_ok=True)
    for name, (params, system) in TABLE_FIXTURES.items():
        (out / name).write_text(json.dumps(tables(params, system)) + "\n")
    np.savez(out / ENCODED_FIXTURE, **encoded(H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"))
    (out / FEMOCO_FIXTURE).write_text(json.dumps(femoco_toffolis(), indent=2) + "\n")
    (out / PROVENANCE).write_text(json.dumps(provenance(), indent=2) + "\n")
    for name in FIXTURES:
        print(f"wrote {out / name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests without fixtures**

Run: `scripts/test.sh tests/data/test_dfthcbe_fixtures.py -q -rs`
Expected: `2 passed, 5 skipped`, with the line `SKIPPED [5] src/dfthc/testing/fixtures.py:24: dfthcbe fixtures not generated: run scripts/make_dfthcbe_fixtures.py in .venv-ref`.

- [ ] **Step 6: Run the per-commit tier and the linter, then commit**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: no failures. With waves 1-a to 1-d merged (and Workstream C's tests on top), the count is `400 passed` plus the Workstream C tasks' tests: 181 + 95 + 58 + 49 + 15, plus 2. The 5 skips do not show in the xdist summary.

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

```bash
git add scripts/make_dfthcbe_fixtures.py src/dfthc/testing/fixtures.py \
        tests/data/test_dfthcbe_fixtures.py
git commit -m "Add dfthcbe fixture loaders and the fixture generation script"
```

Then stop and report to the orchestrator: Step 7 is the orchestrator's.

- [ ] **Step 7 (orchestrator): Build `.venv-ref` and generate the fixtures in the lane**

The orchestrator builds `.venv-ref` from the pinned dfthcbe tarball; nothing is cloned and no step needs elevated permission.
- If `<main checkout>/.scratch/dfthcbe-ref` is missing, the script downloads the tarball of commit `930255020cec30172e0a7d17c0071d9476a687d1` from `codeload.github.com` into it.
- It checks a SHA-256 digest of `pyproject.toml` and the package sources.
- It installs a copy, because pip builds inside the directory it installs and would otherwise modify `.scratch/dfthcbe-ref`.

It also avoids the two sandbox failures of a plain install:
- pip fails certificate verification through the sandbox proxy unless it is given the macOS keychain's certificates;
- the sandbox refuses any directory named `venv` inside the workspace, which jedi's `typeshed/stdlib/venv/` hits; jedi comes with IPython in Qualtran's full dependency set.

Export the two locations, then run the script below with `bash`, for example saved as `$LANE/.scratch/make_venv_ref.sh`. As its own process it keeps its `TMPDIR` and pip variables out of your shell. A `TMPDIR` inside a worktree makes a Task 0.1 test fail in the sandbox, because that test creates a `venv` directory.

```bash
export MAIN="<main checkout>"            # the repository root, which holds .venv
export LANE="$MAIN/.worktrees/<lane>"     # the lane of Task 1A.6
bash "$LANE/.scratch/make_venv_ref.sh"
```

```bash
#!/usr/bin/env bash
# Build .venv-ref from the pinned dfthcbe tarball and write the Task 1A.6 fixtures into $LANE.
# Needs MAIN (the repository root, which holds .venv) and LANE (the worktree of Task 1A.6).
# VENV_REF overrides the environment's location (default: $MAIN/.venv-ref).
set -euo pipefail
PIN=930255020cec30172e0a7d17c0071d9476a687d1
REF="$MAIN/.scratch/dfthcbe-ref"
VENV_REF="${VENV_REF:-$MAIN/.venv-ref}"
[ -d "$REF" ] || { mkdir -p "$REF" && curl -sSfL \
    "https://codeload.github.com/Covestro-HPQC/dfthcbe/tar.gz/$PIN" \
    | tar -xz -C "$REF" --strip-components=1; }
# SHA-256 of pyproject.toml and the package sources at $PIN (xargs fails in the sandbox).
test "$(cd "$REF" && shasum -a 256 pyproject.toml $(find dfthcbe -name '*.py' | LC_ALL=C sort) \
    | shasum -a 256 | cut -d' ' -f1)" \
    = 802221370175be502cde773d1c369f02c77a0d167d72ddffc1fb2d5df7df3c27
cd "$LANE"
export PIP_NO_CACHE_DIR=1 TMPDIR="$LANE/.scratch/tmp"
mkdir -p "$TMPDIR"
# pip verifies certificates against the macOS keychain, as curl does through the sandbox proxy.
security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain \
    /Library/Keychains/System.keychain > "$TMPDIR/keychain-certs.pem"
export PIP_CERT="$TMPDIR/keychain-certs.pem"
[ -d "$VENV_REF" ] || python3.12 -m venv "$VENV_REF"
"$VENV_REF/bin/pip" install --upgrade pip
# Qualtran 0.7.0 without its notebook dependencies: jedi (through IPython) ships a file under
# typeshed/stdlib/venv/, and the sandbox refuses any directory named venv in the workspace.
"$VENV_REF/bin/pip" install "attrs>=23.2.0" "cachetools>=5.3" networkx "numpy<3.0,>=1.26" \
    sympy "cirq-core~=1.4" fxpmath galois pydot quimb "typing_extensions>=4.10.0" mpmath \
    protobuf scipy pytest decorator ipython-pygments-lexers matplotlib-inline pexpect \
    prompt_toolkit pygments stack_data traitlets
"$VENV_REF/bin/pip" install --no-deps qualtran==0.7.0 ipython ipywidgets comm \
    widgetsnbextension jupyterlab_widgets
# pip builds inside the directory it installs, so install a copy and keep $REF pristine.
rm -rf "$TMPDIR/dfthcbe-src"
cp -R "$REF" "$TMPDIR/dfthcbe-src"
"$VENV_REF/bin/pip" install --no-deps "$TMPDIR/dfthcbe-src"
mkdir -p tests/fixtures/dfthcbe
"$VENV_REF/bin/pip" freeze > tests/fixtures/dfthcbe/requirements-ref.lock
MPLCONFIGDIR="$TMPDIR" "$VENV_REF/bin/python" scripts/make_dfthcbe_fixtures.py
```

Expected: pip's install lines, then the five lines `wrote <lane>/tests/fixtures/dfthcbe/<name>` for `tables_h2_sto3g_test.json`, `tables_h2_631g_test.json`, `encoded_h2_sto3g_test.npz`, `femoco_block_toffolis.json` and `provenance.json`. The run takes about 30 s once the packages are installed. Its process tree stays far below the 4 GB "heavy" threshold, so it is not a heavy run. The lock lists `qualtran==0.7.0` and `dfthcbe @ file://` followed by the copy's path.

Dry-run on 2026-11-12: the script ran exactly as shown, with `VENV_REF` pointing at a scratch environment, against `.scratch/dfthcbe-ref` (the orchestrator's extraction of the tarball). The five fixtures were identical to the ones the tests below were dry-run with: the JSON files byte for byte, the arrays exactly. The download branch, run separately into an empty directory, reproduced the same tree and passed the digest check. `femoco_block_toffolis.json` must read:

```json
{
  "outer_prepare": 212,
  "outer_unprepare": 94,
  "inner_prepare": 913,
  "inner_unprepare": 220,
  "inner_select": 3502,
  "inner_select_adjoint": 3502,
  "ref_t2": 20,
  "ref_ab": 45,
  "block_encoding": 9596,
  "controlled_block_encoding": 9597,
  "walk": 9641,
  "controlled_walk": 9642
}
```

These are dfthcbe's own FeMoco-54 values (its `test_femoco_cost_by_block`: walk 9,641, controlled walk 9,642).

If `.venv-ref` cannot be built at all, record the reason in the ledger. The fixture tests then keep skipping with the explicit reason, and spec 8.6 checks 1 and 2 and the end-to-end simulation still hold. Ask the user before giving up, because the definition of done expects the fixtures.

- [ ] **Step 8: Run the tests with the fixtures and commit them**

Run: `scripts/test.sh tests/data/test_dfthcbe_fixtures.py -q -rs`
Expected: `7 passed`, nothing skipped.

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: no failures; 5 more passed than in Step 6.

```bash
git add tests/fixtures/dfthcbe
git commit -m "Add dfthcbe reference fixtures"
```

---

### Task 1A.7: `dfthc.costs`, the closed forms of spec 9.24

**Task id:** `1A.7` · **Branch:** `phase1/1A.7` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 1A.1

**Files:**
- Create: `src/dfthc/costs.py`, `tests/test_costs.py`

**Interfaces:**
- Consumes: `WalkParams`, `lg`, `odd_part`, `FEMOCO54`, `H2_STO3G_TEST`; `spikes/prototypes.unary_toffolis` (Task 0.7), imported in the test through `sys.path`. Importing `prototypes` registers its two prototype rules, which is harmless, and the module is cached, so they register once per worker.
- Produces, in `dfthc.costs`:
  - `@dataclass(frozen=True) class Cost(toffoli: int, measurements: int = 0)`, which supports `+` and `int * Cost`.
  - `Range = tuple[int, int, int]` (`(lo, hi, n)`).
  - Building blocks: `unary(lo, hi, n, controlled) -> int` (`ValueError` for an empty or out-of-range range), `unary_nested(outer, inner, controlled) -> int`, `truncated_unary(L, controlled) -> int` and `onehot(ob) -> int`.
  - L1 primitives:
    - `phase_gradient_rotation(k, m)`, `flagged_uniform_prep(L, s)`, `flagged_uniform_unprep(L, s)`;
    - `qroam_read(ranges, block_bits, width, controlled)`, with one or two ranges;
    - `phase_fixup(high_ranges, onehot_bits, controlled)`;
    - `qroam_erase(ranges, block_bits, width, onehot_bits, controlled)`, whose `ranges` are the fixup's high ranges;
    - `zero_reflection(n_w, controlled)`, which raises `ValueError` for `n_w < 1`.
  - L2 and L3 blocks: `outer_prepare(p)`, `outer_unprepare(p)`, `inner_prepare(p)`, `inner_unprepare(p)`, `tail_angle_load(p)`, `sf_angle_load(p)`, `spin_mux()`, `spin_swap(p)`, `givens_rotation(p)`, `givens_ladder(p)`, `majorana_gate()`, `angle_erase(p)`, `inner_select(p)`, `ref_t2(p, controlled)`, `ref_ab(p, controlled)`, `block_encoding(p, controlled)` and `walk(p, controlled)`, each returning `Cost`.
  - L4 and totals: `controlled_walk_power(p, power)`, `qpe_round(p, j, t)`, `iterative_qpe(p, t)`, `toffoli_only(p) -> int` (T_Toffoli) and `measured_and_uncomputes(p, controlled) -> int` (A).
  - `cheapest_block_bits(p) -> tuple[int, int, int, int]` (`k1, k2, ke1, ke2`), smallest on ties; `(k1, ke1)` jointly, because the outer erase covers the outer read's loaded set (change 9).

**Acceptance tests** (14 tests): every row of the spec 9.24 evaluation table at `FEMOCO54`, exactly:
- unary: outer read 84, inner read 543 (273 + 270), erase high parts 22 and 70, tail 61, SF 291;
- uniform preparation: 30 / 24 and 18 / 16;
- reads and erases: 156, 933; 36 with M 96, 196 with M 416;
- prepares: 210 / 69 (M 96) and 971 / 217 (M 416);
- `GivensLadder` 1,484, `AngleErase` 102 (M 798), `InnerSelect` 3,537;
- `RefT2` 20 / 21, `RefAB` 44 / 45;
- block encoding 9,749 (M 2,524), controlled 9,750; walk 9,793; controlled walk 9,795; T_Toffoli 1,118; A 8,675 / 8,677; M + A 11,199 / 11,201.

Also: the QPE rows at t = 1, 2, 3, 5, with the rounds summing to the total; `truncated_unary` and `onehot`; `unary` equals `spikes/prototypes.unary_toffolis` on 200 random cases with n ≤ 8; `cheapest_block_bits` returns a brute-force minimizer of `walk(p).toffoli` at `FEMOCO54`, `H2_STO3G_TEST`, `MID_NAPHTHALENE` and a shape with `X_o = 5` (the last two with `X_o` not a multiple of `K1`), and equals the block bits of `FEMOCO54`, `(2, 4, 4, 2)`, and of `MID_NAPHTHALENE`, `(1, 2, 3, 1)`; the outer erase covers the read's loaded set (`(X_o, k1, ke1) = (5, 2, 0)` gives T = 17, and a grid of nonaligned shapes; change 9); Python ints throughout; invalid ranges are rejected.

**Definition of done:** `14 passed`; the docstring of every function cites its spec 9.24 row; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_costs.py`:

```python
"""dfthc.costs against every row of the spec 9.24 evaluation table at FeMoco-54."""

import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from dfthc import costs
from dfthc.costs import Cost
from dfthc.params import FEMOCO54, H2_STO3G_TEST, MID_NAPHTHALENE, lg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spikes"))
from prototypes import unary_toffolis  # noqa: E402

P = FEMOCO54


def test_unary_iteration_rows():
    assert costs.unary(0, 81, 7, controlled=False) == 84  # outer read
    assert costs.unary_nested((0, 270, 9), (0, 2, 1), controlled=False) == 543  # 273 + 270 * 1
    assert costs.unary(0, 270, 9, controlled=False) == 273
    assert costs.unary(0, 21, 5, controlled=False) == 22  # outer erase high part
    assert costs.unary(0, 68, 7, controlled=False) == 70  # inner erase high part
    assert costs.unary(270, 324, 9, controlled=False) == 61  # tail range
    assert costs.unary(0, 10, 4, True) + 10 * costs.unary(0, 27, 5, True) == 291  # SF
    assert costs.unary(0, 1, 0, controlled=True) == 0


def test_unary_equals_the_prototype_on_a_grid():
    """200 random (lo, hi, n, c) with n <= 8 against spikes/prototypes.unary_toffolis (Task 0.7)."""
    rng = np.random.default_rng(924)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        lo, hi = sorted(int(v) for v in rng.choice(2**n + 1, 2, replace=False))
        controlled = bool(rng.integers(2))
        assert costs.unary(lo, hi, n, controlled) == unary_toffolis(lo, hi, n, controlled)


def test_truncated_unary_and_onehot():
    assert [costs.truncated_unary(L, c) for L, c in [(81, False), (81, True), (2, False)]] == [
        79, 80, 0]
    assert [costs.onehot(ob) for ob in (0, 1, 2, 4, 7)] == [0, 0, 2, 14, 126]


def test_uniform_preparation_rows():
    assert costs.flagged_uniform_prep(324, 7) == Cost(30)
    assert costs.flagged_uniform_unprep(324, 7) == Cost(24)
    assert costs.flagged_uniform_prep(28, 7) == Cost(18)
    assert costs.flagged_uniform_unprep(28, 7) == Cost(16)
    assert costs.flagged_uniform_prep(8, 7) == costs.flagged_uniform_unprep(8, 7) == Cost(0)
    assert costs.flagged_uniform_prep(5, 3) == Cost(3 * 3 + 6 - 5)


def test_phase_gradient_rotation_row():
    assert costs.phase_gradient_rotation(0, 5) == Cost(0)
    assert costs.phase_gradient_rotation(1, 5) == costs.phase_gradient_rotation(31, 5) == Cost(4)


def test_read_and_erase_rows():
    assert costs.qroam_read([(0, 81, 7)], 2, 24, controlled=False) == Cost(156)
    assert costs.qroam_read([(0, 270, 9), (0, 2, 1)], 4, 26, controlled=False) == Cost(933)
    assert costs.phase_fixup([(0, 21, 5)], 4, controlled=False) == Cost(22 + 14)
    assert costs.phase_fixup([(0, 68, 7)], 7, controlled=False) == Cost(70 + 126)
    assert costs.qroam_erase([(0, 21, 5)], 2, 24, 4, controlled=False) == Cost(36, 96)
    assert costs.qroam_erase([(0, 68, 7)], 4, 26, 7, controlled=False) == Cost(196, 416)


X_O_5 = dataclasses.replace(H2_STO3G_TEST, N=3, n_plus=1, R=1, C=2, k1=2, ke1=0)
"""X_o = 5 with k1 = 2 > ke1 = 0: the outer read's last block [4, 8) holds dead positions 5 to 7."""


def test_outer_erase_covers_the_loaded_set():
    """The outer erase's fixup covers the read's loaded set [0, ceil(X_o / K1) K1), dead positions
    of the last block included (spec 9.5): at X_o = 5, k1 = 2, ke1 = 0 its high range is [0, 8),
    not [0, 5), so T = w_x + U(0, 8, 3, 0) + F^dag(5, 3) = 3 + 6 + 8 and M = K1 d1 = 4 * 5."""
    assert X_O_5.X_o == 5
    assert costs.outer_unprepare(X_O_5) == Cost(3 + 6 + 8, 4 * 5)
    for n, k1, ke1 in itertools.product(range(3, 11), range(4), range(4)):
        p = dataclasses.replace(X_O_5, N=n, k1=k1, ke1=ke1)
        loaded = -(-p.X_o // p.K1) * p.K1
        fixup = costs.unary(0, -(-loaded // 2**ke1), p.w_x - ke1, False) + costs.onehot(ke1)
        expected = p.w_x + fixup + costs.flagged_uniform_unprep(p.X_o, p.s_u).toffoli
        assert costs.outer_unprepare(p).toffoli == expected, (p.X_o, k1, ke1)


def test_block_rows():
    assert (costs.outer_prepare(P), costs.outer_unprepare(P)) == (Cost(210), Cost(69, 96))
    assert (costs.inner_prepare(P), costs.inner_unprepare(P)) == (Cost(971), Cost(217, 416))
    assert (costs.tail_angle_load(P), costs.sf_angle_load(P)) == (Cost(61), Cost(291))
    assert (costs.spin_mux(), costs.spin_swap(P)) == (Cost(1), Cost(54))
    assert (costs.givens_rotation(P), costs.givens_ladder(P)) == (Cost(28), Cost(1484))
    assert costs.majorana_gate() == Cost(5)
    assert costs.angle_erase(P) == Cost(102, 798)
    assert costs.inner_select(P) == Cost(3537, 798)


def test_reflection_rows():
    assert (costs.ref_t2(P, False), costs.ref_t2(P, True)) == (Cost(20), Cost(21))
    assert (costs.ref_ab(P, False), costs.ref_ab(P, True)) == (Cost(44), Cost(45))
    assert [costs.zero_reflection(n, False).toffoli for n in (1, 2, 3, 6, 46)] == [0, 0, 1, 4, 44]
    assert [costs.zero_reflection(n, True).toffoli for n in (1, 2, 3, 6, 46)] == [0, 1, 2, 5, 45]
    with pytest.raises(ValueError, match="n_w"):
        costs.zero_reflection(0, False)


def test_totals():
    assert costs.block_encoding(P, False) == Cost(9749, 2524)
    assert costs.block_encoding(P, True) == Cost(9750, 2524)
    assert costs.walk(P, False) == Cost(9793, 2524)
    assert costs.walk(P, True) == Cost(9795, 2524)
    assert costs.toffoli_only(P) == 18 + 20 + 72 + 780 + 216 + 12 == 1118
    assert costs.measured_and_uncomputes(P, False) == 8675
    assert costs.measured_and_uncomputes(P, True) == 8677
    assert costs.walk(P, False).measurements + costs.measured_and_uncomputes(P, False) == 11199
    assert costs.walk(P, True).measurements + costs.measured_and_uncomputes(P, True) == 11201


def test_application_rows():
    cw, w = costs.walk(P, True), costs.walk(P, False)
    assert costs.controlled_walk_power(P, 4) == Cost(4 * cw.toffoli, 4 * w.measurements)
    assert costs.qpe_round(P, 0, 3) == Cost(cw.toffoli + 1 + 2, w.measurements)
    assert costs.qpe_round(P, 2, 3) == Cost(4 * cw.toffoli, 4 * w.measurements)
    for t in (1, 2, 3, 5):
        expected = Cost((2**t - 1) * cw.toffoli + (t**3 - t) // 6, (2**t - 1) * w.measurements + t)
        assert costs.iterative_qpe(P, t) == expected
        assert sum(costs.qpe_round(P, j, t).toffoli for j in range(t)) == expected.toffoli


def test_cheapest_block_bits_is_a_brute_force_minimizer():
    """Over every (k1, k2, ke1, ke2), also at shapes whose X_o is not a multiple of K1."""
    for params in (P, H2_STO3G_TEST, MID_NAPHTHALENE, X_O_5):
        best = costs.cheapest_block_bits(params)
        grid = itertools.product(range(params.w_x + 1), range(params.w_b + 1),
                                 range(params.w_x + 1), range(lg(params.RC) + 1))
        brute = min(costs.walk(dataclasses.replace(params, k1=a, k2=b, ke1=c, ke2=d), False).toffoli
                    for a, b, c, d in grid)
        chosen = dataclasses.replace(params, k1=best[0], k2=best[1], ke1=best[2], ke2=best[3])
        assert costs.walk(chosen, False).toffoli == brute
    assert costs.cheapest_block_bits(P) == (2, 4, 4, 2)  # the FEMOCO54 preset's block bits
    assert costs.cheapest_block_bits(MID_NAPHTHALENE) == (1, 2, 3, 1)  # the preset's, dfthcbe's


def test_cost_values_are_python_ints():
    for value in dataclasses.astuple(costs.walk(P, True)) + (costs.toffoli_only(P),):
        assert type(value) is int


def test_invalid_ranges_are_rejected():
    with pytest.raises(ValueError, match="range"):
        costs.unary(4, 4, 3, controlled=False)
    with pytest.raises(ValueError, match="range"):
        costs.unary(0, 9, 3, controlled=False)
    with pytest.raises(ValueError, match="one or two"):
        costs.qroam_read([], 1, 2, controlled=False)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/test_costs.py -q`
Expected: FAIL at collection, `1 error`: `ImportError: cannot import name 'costs' from 'dfthc'`.

- [ ] **Step 3: Write `dfthc.costs`**

Create `src/dfthc/costs.py`:

```python
"""Closed-form costs of our construction: spec section 9.24, implemented verbatim.

``T`` counts ``Toffoli`` plus ``TemporaryAND``; ``M`` counts ``MidCircuitMeasure``, the explicit
erase measurements. Measured AND uncomputes (``Adjoint(TemporaryAND)``) are in neither; their
count ``A`` is :func:`measured_and_uncomputes`. Every T4 test asserts a compiled count against one
of these functions; each docstring names its spec 9.24 row. ``p`` is a
:class:`~dfthc.params.WalkParams`, and ``s = s_u``.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from dfthc.params import WalkParams, lg, odd_part

Range = tuple[int, int, int]
"""``(lo, hi, n)``: the range ``[lo, hi)`` of an ``n``-bit address register."""


@dataclass(frozen=True)
class Cost:
    """Toffoli count ``T`` and explicit measurement count ``M`` of one operator."""

    toffoli: int
    measurements: int = 0

    def __add__(self, other: Cost) -> Cost:
        return Cost(self.toffoli + other.toffoli, self.measurements + other.measurements)

    def __rmul__(self, factor: int) -> Cost:
        return Cost(factor * self.toffoli, factor * self.measurements)


# --- building blocks ---------------------------------------------------------------------------


def unary(lo: int, hi: int, n: int, controlled: bool) -> int:
    """Exact unary iteration over ``[lo, hi)`` of an ``n``-bit register (spec 9.24, "Exact unary
    iteration"): ``sum_{m=1}^{n} (floor((hi-1)/2^m) - floor(lo/2^m) + 1) - [c = 0]``; 0 if n = 0.
    """
    if not 0 <= lo < hi <= 2**n:
        raise ValueError(f"range [{lo}, {hi}) is not a nonempty range of {n}-bit values")
    if n == 0:
        return 0
    nodes = sum((hi - 1) // 2**m - lo // 2**m + 1 for m in range(1, n + 1))
    return nodes - (0 if controlled else 1)


def unary_nested(outer: Range, inner: Range, controlled: bool) -> int:
    """Nested exact iteration (spec 9.24): ``U_2 = U(lo1, hi1, n1, c) + (hi1 - lo1) U(lo2, hi2,
    n2, 1)``, the inner register iterated under every outer leaf."""
    lo1, hi1, n1 = outer
    return unary(lo1, hi1, n1, controlled) + (hi1 - lo1) * unary(*inner, controlled=True)


def truncated_unary(L: int, controlled: bool) -> int:
    """Truncated iteration of Qualtran and dfthcbe over ``[0, L)``, for comparison only (spec
    9.24): ``L + c - 2``."""
    return L + int(controlled) - 2


def onehot(ob: int) -> int:
    """One-hot register over ``ob`` bits (spec 9.24): ``O(ob) = 2**ob - 2`` for ob >= 1, else 0."""
    return 2**ob - 2 if ob >= 1 else 0


def _iteration(ranges: Sequence[Range], controlled: bool) -> int:
    if len(ranges) == 1:
        return unary(*ranges[0], controlled=controlled)
    if len(ranges) == 2:
        return unary_nested(ranges[0], ranges[1], controlled)
    raise ValueError(f"an address has one or two registers; got {len(ranges)}")


# --- L1 primitives -----------------------------------------------------------------------------


def phase_gradient_rotation(k: int, m: int) -> Cost:
    """``PhaseGradientRotation(k, m)`` (spec 9.24 row): ``T = (m - 1) [k != 0]``, ``M = 0``."""
    return Cost((m - 1) * int(k != 0))


def _uniform_bits(L: int) -> int:
    return lg(odd_part(L))


def flagged_uniform_prep(L: int, s: int) -> Cost:
    """``FlaggedUniformPrep(L, s)`` (spec 9.24 row): with ``L = 2**eta l``, ``l`` odd and
    ``n = lg(l)``, ``T = 3 n + 2 s - 5`` if ``l > 1``, else 0."""
    n = _uniform_bits(L)
    return Cost(3 * n + 2 * s - 5 if n else 0)


def flagged_uniform_unprep(L: int, s: int) -> Cost:
    """Adjoint of ``FlaggedUniformPrep(L, s)`` (spec 9.24 row): ``T = 2 n + 2 s - 4`` if ``l > 1``,
    else 0."""
    n = _uniform_bits(L)
    return Cost(2 * n + 2 * s - 4 if n else 0)


def qroam_read(ranges: Sequence[Range], block_bits: int, width: int, controlled: bool) -> Cost:
    """``QROAMRead`` (spec 9.24 row): ``U`` or ``U_2`` over the visited block ranges ``ranges``
    plus ``(2**block_bits - 1) width`` swap-network CSWAPs."""
    return Cost(_iteration(ranges, controlled) + (2**block_bits - 1) * width)


def phase_fixup(high_ranges: Sequence[Range], onehot_bits: int, controlled: bool) -> Cost:
    """``PhaseFixup`` (spec 9.24 row): ``U`` or ``U_2`` over the covered high ranges plus
    ``O(onehot_bits)``."""
    return Cost(_iteration(high_ranges, controlled) + onehot(onehot_bits))


def qroam_erase(ranges: Sequence[Range], block_bits: int, width: int, onehot_bits: int,
                controlled: bool) -> Cost:
    """``QROAMErase`` (spec 9.24 row): ``T`` of the ``PhaseFixup`` over the read's loaded set,
    whose high ranges are ``ranges``; ``M = 2**block_bits width``."""
    fixup = phase_fixup(ranges, onehot_bits, controlled)
    return Cost(fixup.toffoli, 2**block_bits * width)


def zero_reflection(n_w: int, controlled: bool) -> Cost:
    """``ZeroReflection`` on ``n_w`` wires (spec 9.24 row): ``T = max(n_w - 2, 0)``, controlled
    ``n_w - 1``."""
    if n_w < 1:
        raise ValueError(f"ZeroReflection needs n_w >= 1; got {n_w}")
    return Cost(n_w - 1 if controlled else max(n_w - 2, 0))


# --- L2 and L3 blocks ----------------------------------------------------------------------------


def _outer_read_ranges(p: WalkParams) -> list[Range]:
    """Outer read: ``[0, ceil(X_o / K1))`` on ``w_x - k1`` bits (spec 9.24)."""
    return [(0, -(-p.X_o // p.K1), p.w_x - p.k1)]


def _inner_read_ranges(p: WalkParams) -> list[Range]:
    """Inner read: rows ``[0, RC)`` on ``w_x`` bits, then blocks ``[0, ceil((B+1)/K2))`` on
    ``w_b - k2`` bits (spec 9.24)."""
    return [(0, p.RC, p.w_x), (0, -(-(p.B + 1) // p.K2), p.w_b - p.k2)]


def _outer_erase_ranges(p: WalkParams) -> list[Range]:
    """Outer erase: the high part of the outer read's loaded set ``[0, ceil(X_o / K1) K1)``,
    ``[0, ceil(ceil(X_o / K1) K1 / 2**ke1))`` on ``w_x - ke1`` bits (spec 9.24). It equals
    ``ceil(X_o / 2**ke1)`` when ``ke1 >= k1`` or ``K1`` divides ``X_o``."""
    loaded = -(-p.X_o // p.K1) * p.K1
    return [(0, -(-loaded // 2**p.ke1), p.w_x - p.ke1)]


def _inner_erase_ranges(p: WalkParams) -> list[Range]:
    """Inner erase: high part ``[0, ceil(RC / 2**ke2))`` on ``w_x - ke2`` bits (spec 9.24)."""
    return [(0, -(-p.RC // 2**p.ke2), p.w_x - p.ke2)]


def outer_prepare(p: WalkParams) -> Cost:
    """``OuterPrepare`` (spec 9.24 row): ``F(X_o) + T_read,out + b_k1 + w_x``, ``M = 0``."""
    read = qroam_read(_outer_read_ranges(p), p.k1, p.d1, controlled=False)
    return flagged_uniform_prep(p.X_o, p.s_u) + read + Cost(p.b_k1 + p.w_x)


def outer_unprepare(p: WalkParams) -> Cost:
    """``Adjoint(OuterPrepare)`` (spec 9.24 row): ``w_x + T_erase,out + F^dag(X_o)``,
    ``M = K1 d1``."""
    erase = qroam_erase(_outer_erase_ranges(p), p.k1, p.d1, p.ke1, controlled=False)
    return Cost(p.w_x) + erase + flagged_uniform_unprep(p.X_o, p.s_u)


def inner_prepare(p: WalkParams) -> Cost:
    """``InnerPrepare`` (spec 9.24 row): ``F(B+1) + T_read,in + b_k2 + w_b``, ``M = 0``."""
    read = qroam_read(_inner_read_ranges(p), p.k2, p.d2, controlled=False)
    return flagged_uniform_prep(p.B + 1, p.s_u) + read + Cost(p.b_k2 + p.w_b)


def inner_unprepare(p: WalkParams) -> Cost:
    """``Adjoint(InnerPrepare)`` (spec 9.24 row): ``w_b + T_erase,in + F^dag(B+1)``,
    ``M = K2 d2``; the erase one-hot spans the low ``ke2`` bits of ``x_o`` and all of ``b``."""
    erase = qroam_erase(_inner_erase_ranges(p), p.k2, p.d2, p.ke2 + p.w_b, controlled=False)
    return Cost(p.w_b) + erase + flagged_uniform_unprep(p.B + 1, p.s_u)


def tail_angle_load(p: WalkParams) -> Cost:
    """``TailAngleLoad`` (spec 9.24 row): ``U(RC, X_o, w_x, 0)``."""
    return Cost(unary(p.RC, p.X_o, p.w_x, controlled=False))


def sf_angle_load(p: WalkParams) -> Cost:
    """``SFAngleLoad`` (spec 9.24 row): ``U(0, R, w_r, 1) + R U(0, B, w_b, 1)``."""
    return Cost(unary(0, p.R, p.w_r, True) + p.R * unary(0, p.B, p.w_b, True))


def spin_mux() -> Cost:
    """``SpinMux`` (spec 9.24 row): ``T = 1``."""
    return Cost(1)


def spin_swap(p: WalkParams) -> Cost:
    """``SpinSwap`` (spec 9.24 row): ``T = N``."""
    return Cost(p.N)


def givens_rotation(p: WalkParams) -> Cost:
    """``GivensRotation`` and its adjoint (spec 9.24 row): ``T = 2 (b_rot - 1)``."""
    return Cost(2 * (p.b_rot - 1))


def givens_ladder(p: WalkParams) -> Cost:
    """``GivensLadder`` and its adjoint (spec 9.24 row): ``T = 2 (b_rot - 1)(N - 1)``."""
    return Cost(2 * (p.b_rot - 1) * (p.N - 1))


def majorana_gate() -> Cost:
    """``MajoranaGate`` and its adjoint (spec 9.24 row): ``T = 5``."""
    return Cost(5)


def angle_erase(p: WalkParams) -> Cost:
    """``AngleErase`` (spec 9.24 row): ``U(0, R, w_r, 1) + O(w_b) + U(RC, X_o, w_x, 0)``,
    ``M = (N - 1) b_rot + 3``."""
    toffoli = (unary(0, p.R, p.w_r, True) + onehot(p.w_b)
               + unary(p.RC, p.X_o, p.w_x, controlled=False))
    return Cost(toffoli, (p.N - 1) * p.b_rot + 3)


def inner_select(p: WalkParams) -> Cost:
    """``InnerSelect`` and its adjoint (spec 9.24 row): ``T_tail + T_SF + 2 + 2N +
    4 (b_rot - 1)(N - 1) + 5 + T_angle erase``, ``M = (N - 1) b_rot + 3``."""
    return (tail_angle_load(p) + sf_angle_load(p) + 2 * spin_mux() + 2 * spin_swap(p)
            + 2 * givens_ladder(p) + majorana_gate() + angle_erase(p))


def ref_t2(p: WalkParams, controlled: bool) -> Cost:
    """``RefT2`` (spec 9.24 row): ``ZeroReflection`` with ``n_w = w_b + b_k2 + 2``."""
    return zero_reflection(p.w_b + p.b_k2 + 2, controlled)


def ref_ab(p: WalkParams, controlled: bool) -> Cost:
    """``RefAB`` (spec 9.24 row): ``ZeroReflection`` with ``n_w = w_x + b_k1 + w_b + b_k2 + 2``."""
    return zero_reflection(p.w_x + p.b_k1 + p.w_b + p.b_k2 + 2, controlled)


def block_encoding(p: WalkParams, controlled: bool) -> Cost:
    """``DFTHCBlockEncoding`` (spec 9.24 row): ``T_out + T_out^dag + 2 (T_in + T_sel + T_in^dag)
    + T_RefT2`` (``InnerSigns`` costs 0), ``M = K1 d1 + 2 (K2 d2 + (N - 1) b_rot + 3)``; the
    controlled version controls only ``RefT2``."""
    inner = inner_prepare(p) + inner_select(p) + inner_unprepare(p)
    return outer_prepare(p) + outer_unprepare(p) + 2 * inner + ref_t2(p, controlled)


def walk(p: WalkParams, controlled: bool) -> Cost:
    """``DFTHCWalk`` and the controlled walk (spec 9.24 row): ``T_BE + T_RefAB``, controlled
    ``+2``; ``M = M_BE``."""
    return block_encoding(p, controlled) + ref_ab(p, controlled)


# --- L4 and totals -----------------------------------------------------------------------------


def controlled_walk_power(p: WalkParams, power: int) -> Cost:
    """``ControlledWalkPower(p)`` (spec 9.24 row): ``T = p T_C(W)``, ``M = p M_W``."""
    return Cost(power * walk(p, True).toffoli, power * walk(p, False).measurements)


def qpe_round(p: WalkParams, j: int, t: int) -> Cost:
    """``QPERound(j)`` of a ``t``-bit QPE, worst case (spec 9.24 row): ``T = 2**j T_C(W) +
    sum_{l=j+1}^{t-1} (l - j)``, ``M = 2**j M_W``."""
    corrections = sum(bit - j for bit in range(j + 1, t))
    return controlled_walk_power(p, 2**j) + Cost(corrections)


def iterative_qpe(p: WalkParams, t: int) -> Cost:
    """Iterative QPE with ``t`` bits, worst case (spec 9.24 row): ``T = (2**t - 1) T_C(W) +
    (t**3 - t) / 6``, ``M = (2**t - 1) M_W + t``."""
    return controlled_walk_power(p, 2**t - 1) + Cost((t**3 - t) // 6, t)


def toffoli_only(p: WalkParams) -> int:
    """``T_Toffoli`` per walk and controlled walk (spec 9.24, "Measured AND uncomputes"):
    ``2 w_x + 4 w_b + (K1 - 1) d1 + 2 (K2 - 1) d2 + 4 N + 12``."""
    return (2 * p.w_x + 4 * p.w_b + (p.K1 - 1) * p.d1 + 2 * (p.K2 - 1) * p.d2 + 4 * p.N + 12)


def measured_and_uncomputes(p: WalkParams, controlled: bool) -> int:
    """``A = T - T_Toffoli`` per walk or controlled walk (spec 9.24, "Measured AND uncomputes")."""
    return walk(p, controlled).toffoli - toffoli_only(p)


def cheapest_block_bits(p: WalkParams) -> tuple[int, int, int, int]:
    """``(k1, k2, ke1, ke2)`` minimizing ``walk(p).toffoli`` (spec 8.3), smallest on ties.

    The walk cost is a sum of separate terms: the outer read and erase depend on ``(k1, ke1)``
    (the erase covers the read's loaded set), the inner reads on ``k2`` and the inner erases on
    ``ke2``. So ``(k1, ke1)`` is minimized jointly, in lexicographic order, and ``k2`` and ``ke2``
    each on their own.
    """

    def cost(**changes) -> int:
        return walk(dataclasses.replace(p, **changes), False).toffoli

    k1, ke1 = min(itertools.product(range(p.w_x + 1), repeat=2),
                  key=lambda pair: cost(k1=pair[0], ke1=pair[1]))
    k2 = min(range(p.w_b + 1), key=lambda k: cost(k2=k))
    ke2 = min(range(lg(p.RC) + 1), key=lambda k: cost(ke2=k))
    return k1, k2, ke1, ke2
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/test_costs.py -q`
Expected: `14 passed`.

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `290 passed` (wave 1-a merged: 181 + 41 + 31 + 23, plus 14).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/costs.py tests/test_costs.py
git commit -m "Add dfthc.costs: the closed forms of spec 9.24"
```
