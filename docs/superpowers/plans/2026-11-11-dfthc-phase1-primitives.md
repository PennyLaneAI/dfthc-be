# DFTHC Phase 1, Workstream C: L1 Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build layer L1 of spec section 5.1: the exact unary-iteration helper, `PhaseGradientRotation`, `FlaggedUniformPrep`, the paper's `QROAMRead`, the erase phase fixup (`PhaseFixup` and `emit_phase_fixup`), the measured `QROAMErase` with its outcome certificates, and `ZeroReflection`, each an `Operator2` with one pinned rule per key, exact Toffoli and measurement counts against `dfthc.costs`, and tests T2 to T6.

**Architecture:** One module per block family under `src/dfthc/ops/` (spec 5.2): `unary.py` (helper), `phase_gradient.py`, `uniform.py`, `qroam.py` (read, fixup, erase), `reflections.py`, plus `_validate.py` with the constructor checks they share. Rules that loop over compilable data (unary iteration, QROAM loads and fix-ups, the phase-gradient fan-out) emit their gates through `dfthc.ops.unary.Emitter`; the same body run in counting mode is the rule's resource function, so declared resources equal emitted operations by construction. Straight-line rules whose children are composite keys (`FlaggedUniformPrep`, `ZeroReflection`) may declare their resources by hand; their T2 (`assert_valid_decomp_rule`, which compares exact declarations with the emitted operations) then runs at a size in every branch of the rule. Registers and table entries are big-endian (spec 6.7): wire `i` of a `width`-wire register holds bit `width - 1 - i`, and `target ‖ junk` of a read is one big-endian integer, `QROAMRead.loaded_bits`. Semantics are checked on `lightning.qubit` where the program has at most about 20 wires, and exhaustively on the sparse simulator of Workstream B beyond that.

**Tech Stack:** Python 3.12, PennyLane `Operator2` and Catalyst (`qp.qjit(capture=True)`, `qp.decompose`, `qp.analyze`), NumPy, JAX; the sparse simulator `dfthc.testing.sparse_sim` and `dfthc.testing.tape` (Task 1B.2); pytest with pytest-xdist; ruff.

This file holds Tasks 1C.1 to 1C.6 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 7). Workstreams A and B are in `2026-11-11-dfthc-phase1-data.md` and `2026-11-11-dfthc-phase1-simulator.md`.

**Code provenance.** Every code block below was executed on 2026-11-12 in a scratch replica of the repository (`.scratch/plan-check/repo-phase1c`, not committed), seeded from local `main` at `f770ca2` (`git archive`) and resynced to `main` at `aeb417e` for revision 1 (Tasks 0.9b and 0.4b: spike files, the canary `tests/canary/test_l26_unsigned_arguments.py` and documents). The Phase 1 A and B source files this workstream imports come from that plan pair's dry-run replica: `src/dfthc/params.py` (1A.1) and `src/dfthc/costs.py` (1A.7) as amended by the Phase 1 A plan's revision 1 (its changes 9 and 10), and `src/dfthc/data/*.py` (1A.2 to 1A.5) and `src/dfthc/testing/sparse_sim.py` and `tape.py` (1B.1, 1B.2) unchanged. No other file of that replica was copied: `compile.py`, `testing/resources.py`, `testing/canary.py` and `tests/conftest.py` are `main`'s. The replica is not a git repository, so the dry-run ran pytest the way `scripts/test.sh` does: same environment, at most 2 workers, `taskpolicy -c utility`, `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` exported. The steps below use `scripts/test.sh` as usual. Each task was replayed in order in a fresh copy of the replica with only the earlier tasks' files present (`.scratch/plan-check/stages/run_stages_1c.sh`, log `stages/1c-stages.log`; for revision 1 serially, under `run_watched`, in 290 s at a peak of 520 MB): every "Expected" line below is that replay's output, and the final files are byte-identical to the replica's. One test of `main` fails in the replica and only there: `tests/test_infrastructure.py::test_runner_fails_closed_without_lowered_priority` creates a directory named `venv` under `TMPDIR`, which the sandbox refuses inside the workspace, where the replica's `TMPDIR` lives; it passes in the worktree lanes, and the counts below count it as passing.

**Revision 1 (2026-11-12)** answers the GPT-5.6 plan review (`.superpowers/sdd/phase1c-plan-review-1.md`, 5 Important findings) with the orchestrator's decisions: `MID_NAPHTHALENE` compiled counts for `PhaseFixup` and `QROAMErase` (I1); the warning finding restricted to isolated probes (I2); the certificates' mutation check kept as a script and a log (I3); the outer-erase range fixed in Task 1A.7 as a prerequisite, with a cross-check against `QROAMErase.fixup_high_ranges` (I4); the remedy-1 data contract (I5); `MID_NAPHTHALENE` with dfthcbe's naphthalene block bits; and the per-commit base of `main` at `aeb417e`.

**Revision 2 (2026-11-12, Task G0.1)** brings in the G0 decision "K-1 at G0" (ledger): Task 1C.3's `QROAMRead` loads each unary leaf with one `qp.for_loop` over a dynamic `int64` array of per-leaf words ("Appendix: per-leaf read loads"), with a compiled read check at K = 16 and width 40; Tasks 1C.4 and 1C.5 change only where their tests build a read; the Global Constraints' rules 3 to 7 follow the amended spec 7.4; the per-commit base is `main`'s 225 tests at G0. Revision 2 resynced the replica's `main`-owned files to `main` at `e0d3afb` (Task 0.2c's `compile.py` and canaries, the Task 0.9c to 0.9e spikes) and replayed Tasks 1C.3 to 1C.5, RED then GREEN, with the files of Tasks 1C.1, 1C.6 and 1C.2 present but their unchanged tests not re-run (`.scratch/plan-check/stages/g0-1/run_stages_g01.sh`, log `stages/g0-1/stages-g01.log`; one worker, under the watchdog, 277 s at a peak of 526 MB). Every Workstream C per-commit test then ran once on the resynced replica (log `stages/g0-1/ops-per-commit-rev2.log`). Review round 1 (GPT-5.6) added `rtol=0` to the 17 tolerance checks of the Workstream C tests, so every `np.allclose` states both tolerances (HANDOFF section 6, test tolerances), and made `test_compiled_read_on_lightning[3]` assert the flagless read's X bound; the eight affected test files then passed serially under the watchdog (194 per-commit tests in 284 s at a peak of 493 MB, and the 7 slow certificates; log `stages/g0-1/review1/rtol-rerun.log`).

**Revision 3 (2026-11-12, Task G0.2)**: the certificates of Task 1C.5 assert, on the tape they certify, that it has fix-up conditionals and that `nonlinear_conditionals` (Task 1B.2's helper in `dfthc.testing.sparse_sim`) flags none of them, so every conditional is a `PauliZ` or `CZ` with a GF(2)-linear predicate, the premise of spec 12.5's conclusion; they start from a seeded random complex amplitude on every value of the address and control wires instead of `phased_prepare`'s four amplitudes, so any permutation or relative phase of the addresses is caught. The three planned `QROAMRead` docstrings (Tasks 1C.3 to 1C.5) say that the flagless leaf's loads are X gates (G0.1 re-check, Minor 1). The replica's `sparse_sim.py` was synced from `main` (Task 1B.2, `8758739`); the smallest certificate, `sto3g-outer`, passed with it (one worker, under the watchdog, 27 s at a peak of 372 MB; log `.scratch/plan-check/stages/g0-2/certificate-dry-run.log`), and a probe confirmed that its tape holds 4 conditional `CZ`s, none flagged, and that a conditional `PauliX` or a `CZ` on the AND of two outcomes is flagged (`stages/g0-2/certificate-premise-probe.log`). The orchestrator then ran all seven certificates (7 passed, serially under the watchdog, 2.6 s at a peak of 183 MB) and re-ran the mutation check against them, with the same failing cases as in revision 1 (log `stages/g0-2/1c-erase-mutants-rev3.log`). No test count changes.

**Revision 4 (2026-11-12, after Task 1C.4's review)**: `emit_phase_fixup` in the `qroam.py` blocks of Tasks 1C.4 and 1C.5 rejects an `onehot_bits` outside `[0, sum(address_widths)]` (an out-of-range value sliced the address from a negative index and silently dropped phases) and lists `low` and `onehot` once (a traced one-hot array was indexed once per covered address; Phase 2's `AngleErase` calls the helper directly). These are the lines merged with Task 1C.4 (`be4b378`); the replica's two `qroam.py` sources carry them. The merged tests of Tasks 1C.2, 1C.3, 1C.4 and 1C.6 also hold their review fixes, which this plan's test blocks predate; the test counts are unchanged. Task 1C.5's block is otherwise unchanged.

**Revision 5 (2026-11-12, after Task 1C.5's reviews)**: the merged tests of Task 1C.5 (`debffc3`) hold the reviews' count-neutral changes, which this plan's test blocks predate: a zeroed block in T2, T6, the lightning smoke run and both linearity tables (the zero-mask path of the counting twin and the rule), and three certificate cases replaced by `outer-k2-ob0`, `inner-6-2-enable1` and `inner-6-2-ob5-enable0` (still 19 and 7 tests); `stages/mutate_erase_1c.py` needs these ids if re-run. The merged `qroam.py` docstrings define `M` and name the paper's equations (PRX (B50), (B53); arXiv:2502.15882v1 (123), (126)).

**Load.** The dry-run ran when lane-a signalled a quiet machine; the `slow` compiles of revision 1 ran alone, each under `dfthc.testing.watchdog.run_watched`. Timings are from the development laptop and vary with its load.

---

## Changes from the master plan

The exact old and new texts of revision 1 are in `phase1c-doc-edits.md` (written and checked against `main` at `aeb417e` by `.scratch/plan-check/stages/make_phase1c_doc_edits.py`), applied on `main` in `a1ffbd6`; the edit ids of changes 1 to 6 refer to that file. Revision 2's edits are in `g0-1-doc-edits.md` (written and checked against `main` at `e0d3afb` by `.scratch/plan-check/stages/make_g0_1_doc_edits.py`); their ids start with G.

1. **Tasks 1C.1, 1C.2 and 1C.3 gain dependencies, and 1C.2 moves from wave 1-d to wave 1-e** (M5 to M7).
   - 1C.2's T3 test compares the failure weight with `dfthc.data.encoded.uniform_failure_probability` (Task 1A.5, wave 1-d), as the master plan's acceptance test says, and runs its `s = 7` cases (29 wires) on the sparse simulator (Task 1B.2, wave 1-b). 1C.2 is off the critical path; in wave 1-e it follows 1C.4 and may slip to wave 1-f next to 1C.5.
   - 1C.1's exhaustive flag test runs every range of up to 4 bits on `SparseSimulator` (Task 1B.1). 1C.3's exhaustive failed-address tests reach 155 wires (the `(28, 4)` inner shape), beyond `lightning.qubit`, and use `decompose_for_simulation` (Task 1B.2). Both dependencies sit in waves 1-a and 1-b, so neither the waves of 1C.1 and 1C.3 nor the critical path change.
2. **Section 1** lists this file and its provenance (M1 to M3).
3. **Section 3 gains items 15 to 17** (M4), after `main`'s item 14 (L-26):
   - 15: the "unable to determine the branch" warning is one flag per compiled module; isolated probes, observed with the pinned toolchain, show one `TemporaryAND` or one `qp.measure(..., reset=True)` raising it on its own (`.scratch/plan-check/stages/branch-warning-probes.log`), so only exact expected counts can show that conditionals are hinted;
   - 16: controlled compiles cost two to four times their uncontrolled ones;
   - 17: the outer erase covers the outer read's loaded set (the Task 1A.7 fix).
4. **Task 1A.1's `MID_NAPHTHALENE` takes dfthcbe's naphthalene block bits** `k1 = 1`, `k2 = 2`, `ke1 = 3`, `ke2 = 1` (were `2, 1, 2, 1`; M8, S3, and Task 3.4's comparison with the stand-in's `mid10` in M12). dfthcbe's README example (`DFTHCWalk.from_factors` with default block sizes) chooses them by its Toffoli minimizers: outer read block bits 1, outer erase 3, inner read `(0, 2)`, inner erase `(1, 0)` (`.scratch/plan-check/probes/p1c_dfthcbe_naphthalene_blocks.py`, run in the scratch Qualtran environment: 15 s, 184 MB). Our `dfthc.costs.cheapest_block_bits` returns the same `(1, 2, 3, 1)`. The Phase 1 A plan carries the code change (its change 10).
5. **Task 1A.7's outer erase covers the outer read's loaded set, and `cheapest_block_bits` minimizes `(k1, ke1)` jointly** (M9, M10, S7; the Phase 1 A plan's change 9). This is a prerequisite of Workstream C: Task 1C.5 adds `test_fixup_high_ranges_agree_with_the_cost_model`, which compares `outer_unprepare` and `inner_unprepare` with `QROAMErase.fixup_high_ranges`, the implementation oracle, on the presets and on 128 shapes with `X_o` from 5 to 12 and every `k1`, `ke1` up to 3, nonaligned ones among them.
6. **Section 7, Workstream C, is replaced as a whole** (M11). What changes, task by task:
   - **1C.1:** a fifth file `src/dfthc/ops/_validate.py` (the constructor checks every operator shares; spec 9.0); the `Emitter` class and `as_resources`; `iterate` and `iterate_nested` take a keyword `emit`; the negative-control choice (control values, X conjugation only where no AND exists); `PhaseGradientRotation`'s `phase_grad` is a minimum size; T3 is exhaustive on the sparse simulator plus three compiled lightning cases, and T4 counts every range with n <= 6 without compiling plus three compiled cases.
   - **1C.2:** the dependencies of change 1; `amplitude_angles` returns `(k, k)`, because dfthcbe uses one angle for both rotations; T3 runs on the sparse simulator for all eight `(L, s)` pairs, plus two compiled lightning cases.
   - **1C.3:** the static helpers `visited_ranges`, `loaded_ranges` and `loaded_contents`; `loaded_set(address_widths, live_ranges, block_bits)`; the table is canonicalized (sorted tuples); `control_value` is `None` with a 0-wire `control` register; the exhaustive failed-address tests also cover outer reads with k = 0 to 3 and an enabled read, and include k >= 2 (decision "Task 0.7 review decisions", item 4).
   - **1C.4:** in tape mode `PhaseFixup`'s parities must be `MeasurementValue`s, so both modes emit one conditional per covered address (decision "K-10 consequences", item 3); `high_ranges` is static; `emit_phase_fixup` skips Python `False` entries (callers decide them from compilable data) and takes `emit`; `test_erase_conditionals_are_hinted` asserts exact expected counts instead of the absence of the warning (section 3, item 15), and a separate test shows the warning absent for a fixup without any AND; `MID_NAPHTHALENE` compiled counts (`slow`; review I1).
   - **1C.5:** `extra_onehot` is always an argument (0 wires unless `K width < 2**ob`), because an `Operator2` has one signature; the static `fixup_high_ranges` and the cross-check of change 5; the certificates add a k = 2 outer shape, a `(6, 2)` inner shape and its enabled form to the four preset shapes; a test that the erase has no adjoint or controlled path; `MID_NAPHTHALENE` compiled counts (`slow`; review I1).
   - **1C.6:** `work` is a minimum size (the controlled rule needs one more wire); `C(ZeroReflection)` is defined for one control wire with control value 1; the exact matrices come from one compiled run each (reference-register construction), and each T-count program applies both pattern values.
7. **Revision 2 (Task G0.1): per-leaf read loads, and the explicit-class design around them** (for this workstream the master-plan edits GM1, GM3, GM7 and GM8 and the spec edits GS10, GS11, GS16, GS18 to GS21 and GS51; the other G edits carry lever X into Phases 2 to 4).
   - **1C.3** is written for per-leaf loads (appendix): dynamic `leaf_words`, the builder `leaf_words_from_table`, `exact=False`, no auto-adjoint test, the compiled read check at K = 16 and width 40; 44 per-commit tests (was 35). The master plan's Task 1C.3 entry follows (GM8).
   - **1C.4 and 1C.5** are unchanged except their expected read counts and the tests that build a read.
   - **Global Constraints** rules 3 to 7 follow the amended spec 7.4 (rule 4: loops in `QROAMRead` and `ControlledWalkPower` only, both never adjointed or controlled; rule 6: explicit adjoint and controlled classes; rule 7: every class's base rule is pinned). This file and the master plan include `templates/_global_constraints_g01.md`; the Phase 0 and Phase 1 A and B plans keep the shared fragment unchanged.
   - **1C.6** is unchanged: `C(ZeroReflection)` stays a tested primitive; Phase 3's `ControlledRefT2` and `ControlledRefAB` emit the multi-controlled Z with its helper instead of requesting `C(ZeroReflection)` (spec 7.4 rule 6).
8. **Revision 3 (Task G0.2): the certificates' premise check** (master-plan edit G2M1, spec edit G2S2 in `g0-2-doc-edits.md`). **1C.5**'s certificates assert `not nonlinear_conditionals(operations)` and that the tape has conditionals, and start from a random-amplitude state of the address and control wires; their count is unchanged.

## Findings for other plans and documents

1. **The Phase 1 A and B code passes against `main`; their expected counts are stale.** In a scratch copy of this replica with their tests and `registers.py`, `fixtures.py` and `scripts/make_dfthcbe_fixtures.py` added, `246 passed` against `main` at `f770ca2` (their 223 per-commit tests then, and `main`'s 23 vendored-data tests). Their per-commit numbers assume 181 tests at G0; `main` at `aeb417e` has 221 (Task 0.2b added 37 tests, Task 0.4 two more than planned, Task 0.4b one canary), so their shift rule moves every number by +40. Revision 1 adds one test to Task 1A.7 (224 A and B per-commit tests); the A plan's numbers include it, and the B plan's Task 1B.3 count (`333 passed` with waves 1-a and 1-b merged) is one higher as well.
2. **Spec and HANDOFF texts** follow from this plan: the warning (S1, S2, H1, worded strictly from the probes and qualified as observed with the pinned toolchain), the T2 `check_matrix` note (S8), `amplitude_angles` (S4), the tape-mode parity contract (S5), `extra_onehot` (S6), the outer-erase range (S7) and the `MID_NAPHTHALENE` row (S3).

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
3. Every argument that determines circuit structure is in `compilable_argnames`; `static_argnames` are not used; `dynamic_argnames` only for runtime data (`PhaseFixup` parities, `QPERound` earlier bits, `QROAMRead`'s per-leaf table words).
4. Bodies of auto-adjointed operators contain no `qp.for_loop` or `qp.cond`; `qp.cond` on mid-circuit measurement outcomes only in erase operators and the QPE wrapper; `qp.for_loop` only in `QROAMRead` (one loop per unary leaf, with `qp.cond` on table bits) and `ControlledWalkPower`, both never adjointed or controlled.
5. Measurement-based uncompute lives only in `QROAMErase`, `AngleErase` and the base rules of `OuterUnprepare` and `InnerUnprepare`; an operator that contains a measurement is never placed under `qp.ctrl`.
6. Adjoint and controlled forms of composites are explicit operator classes (`OuterUnprepare`, `InnerUnprepare`, `InnerSelectDagger`, `GivensLadderDagger`, `ControlledRefT2`, `ControlledRefAB`, `ControlledDFTHCBlockEncoding`, `ControlledDFTHCWalk` and `ControlledToyWalk`); no rule or program calls `qp.adjoint` or `qp.ctrl` on a dfthc composite; the one `C(Op)` rule left is `C(ZeroReflection)`, with signature `(base, control_wires, control_values, work_wires, work_wire_type)`.
7. One base rule per operator; every class's base rule is pinned, the adjoint and controlled classes of rule 6 included, and so are the remaining symbolic rules (§7.5).
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
| `src/dfthc/ops/__init__.py` | 1C.1 | package docstring only (never re-exports) |
| `src/dfthc/ops/_validate.py` | 1C.1 | `check_int`, `check_wires`, `canonical_ints`: the constructor checks of spec 9.0 |
| `src/dfthc/ops/unary.py` | 1C.1 | `Emitter`, `as_resources`, `iterate`, `iterate_nested`, `work_required` |
| `src/dfthc/ops/phase_gradient.py` | 1C.1 | `PhaseGradientRotation` |
| `tests/ops/test_unary.py`, `tests/ops/test_phase_gradient.py` | 1C.1 | flags on every address, counts; the phase and its sign |
| `src/dfthc/ops/uniform.py` | 1C.2 | `FlaggedUniformPrep`, `amplitude_angles` |
| `tests/ops/test_uniform.py` | 1C.2 | uniformity, failure weight, round trips, counts |
| `src/dfthc/ops/qroam.py` | 1C.3, extended by 1C.4 and 1C.5 | `QROAMEmitter`, address helpers, `QROAMRead`; then `PhaseFixup`, `emit_phase_fixup`; then `QROAMErase` |
| `tests/ops/test_qroam_read.py` | 1C.3 | every computational address, the K = 16 compiled check, counts, the leaf-word contract, helpers |
| `tests/ops/test_phase_fixup.py` | 1C.4 | phases on the covered set, hints, counts |
| `tests/ops/test_qroam_erase.py` | 1C.5 | counts, rule selection, smoke runs, `mcm_parities` linearity |
| `tests/ops/test_certificates_qroam.py` | 1C.5 | spec 12.5 certificates and their premise check (`slow`) |
| `src/dfthc/ops/reflections.py` | 1C.6 | `ZeroReflection` (Phase 3 adds `RefT2`, `RefAB`) |
| `tests/ops/test_zero_reflection.py` | 1C.6 | exact matrices, controlled version, counts |

Within each wave of master plan section 5 (as amended by change 1), the tasks' file lists are disjoint: wave 1-d holds 1C.3 (`qroam.py`) and 1C.6 (`reflections.py`) next to 1A.5; wave 1-e holds 1C.2 (`uniform.py`) and 1C.4 (`qroam.py`) next to 1A.6. Only Task 1C.1 creates and edits `src/dfthc/ops/__init__.py`; `tests/ops/` gets no `__init__.py`, and test modules have unique base names. Tasks 1C.4 and 1C.5 replace `qroam.py` as a whole with the version they give; the earlier tasks' code in it is unchanged (only the module docstring, the imports and the emitter's methods grow).

**Shell conventions.** Run commands from the worktree root of your lane. `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` locates the shared environment, as in the earlier plans.

**Expected test counts.** Each task states its targeted count exactly. The per-commit tier (`scripts/test.sh -m "not slow and not release" -q`) counts **225 tests on `main` at G0** (`b2b8c95`: revision 1's 221 at `aeb417e` plus Task 0.2c's four). Each task's expected per-commit number is that base, plus the per-commit tests of every Phase 1 A and B task of the earlier waves (master plan section 5 as amended by change 1: wave 1-a 95 = 41 + 31 + 23, wave 1-b 58 = 17 + 14 + 27 with Task 1A.7's revision-1 test, wave 1-c 49 = 49 + 0, wave 1-d 15, wave 1-e 7), plus the Workstream C tasks of the earlier waves, plus the task's own. If `main` or a merged task has a different count, shift by the difference; the pass criterion is "no failures, no errors, no warnings summary", and the number is a cross-check.

| Task | Wave | Base: earlier waves | Own per-commit | Expected per-commit tier | Slow tests |
| --- | --- | --- | ---: | ---: | ---: |
| 1C.1 | 1-c | 225 + 95 + 58 = 378 | 36 | 414 | 0 |
| 1C.3 | 1-d | 378 + 49 (1A.4) + 36 (1C.1) = 463 | 44 | 507 | 2 |
| 1C.6 | 1-d | 463 | 32 | 495 | 1 |
| 1C.2 | 1-e | 463 + 15 (1A.5) + 44 (1C.3) + 32 (1C.6) = 554 | 40 | 594 | 0 |
| 1C.4 | 1-e | 554 | 25 | 579 | 2 |
| 1C.5 | 1-f | 554 + 7 (1A.6) + 40 (1C.2) + 25 (1C.4) = 626 | 17 | 643 | 9 |

With all of Phase 1 merged the per-commit tier counts 643 tests (225 + 224 for Workstreams A and B + 194 for Workstream C), and Workstream C adds 14 nightly tests (revision 1: 630 = 221 + 224 + 185). Dry-run of the per-commit tier on the replica with `main`'s 221 tests at `aeb417e` and all of Workstream C (revision 1): 406 tests (405 passed, plus the replica-only failure described under "Code provenance") in 516 s with lane-a quiet (log `.scratch/plan-check/stages/1c-per-commit-tier-rev1.log`); before revision 1 the tier took 686 s on a loaded machine, and `main`'s 220 tests at `f770ca2` alone 351 s. Workstreams A and B's 224 per-commit tests add about 10 s. Revision 2 ran Workstream C's 194 per-commit tests once on the resynced replica with one worker (log `stages/g0-1/ops-per-commit-rev2.log`). Every Workstream C test runs in under 8 s in a warm worker, and every process tree measured stays below 530 MB (watchdog footprint).

**Design notes the tasks share.**
- *Counting twin.* A rule body takes an emitter: `Emitter()` emits PennyLane operations, `Emitter(counting=True)` counts them by name, and the resource function runs the body on dummy integer wires and maps the counts to keys with `as_resources`. A body therefore decides which gates to emit from compilable data only (spec 7.4 rules 2 and 3).
- *Exact unary iteration* (spec 9.24): one `TemporaryAND` and one measured `Adjoint(TemporaryAND)` per visited internal node that has a flag; the uncontrolled root uses the top address bit itself. Its cost is `dfthc.costs.unary`; nested iteration is `unary_nested`.
- *Address model* (spec 9.3): one or two registers, the first most significant in the concatenated integer address; blocks are the low `k` bits of the last register. `QROAMRead.visited_ranges` and `PhaseFixup.high_ranges` are the `ranges` arguments of `dfthc.costs.qroam_read`, `phase_fixup` and `qroam_erase`, so T4 tests call the closed forms with the operator's own ranges and assert the spec 9.24 instantiation separately at FeMoco-54.
- *FeMoco-54* values are asserted on the closed forms; no FeMoco table is compiled (master plan section 12.9). Table-free operators at FeMoco widths (`FlaggedUniformPrep(324, 7)`, a 46-wire `ZeroReflection`) are compiled per commit.

---

## Appendix: per-leaf read loads (adopted at G0; revision 2)

This appendix sits before the tasks so that the task-brief extraction of Task 1C.6 does not include it. The G0 decision "K-1 at G0" adopted two compile-memory levers that Task 0.9d measured and one FeMoco-54 stand-in compile confirmed (the controlled walk with both levers: 2,648 MB in 51 s against 2,566 MB projected, exact counts): per-leaf read loads in `QROAMRead` (lever P, spec 9.3) and explicit operator classes for the adjoint and controlled composites (lever X, spec 7.4 rule 6, which concerns Phases 2 to 4). Task 1C.3 below is written for lever P; this appendix records what changed against revision 1 and why. The measured read is `ProtoTableReadP` in `spikes/k1_whole_program.py` on `main` (`p_leaf_words`, `_p_leaf_loader`, `_proto_table_read_p`; Task 0.9d).

**Unchanged:** Tasks 1C.1, 1C.2 and 1C.6 (no tables); the unary helper, the address model, every T, M and A count and closed form; the erase, the fixup and the certificates' method.

**Task 1C.3, `QROAMRead`.**
- *Interface.* The compilable `table` is gone. The dynamic `leaf_words` is an `int64` array of shape `leaf_words_shape(...) = (num_leaves, K, ceil(width / 32))`: row `l` belongs to the `l`-th unary leaf in visiting order (`leaf_blocks`), position `p` to the address `(leaf_blocks[l] << k) | p`. Callers build it from their compilable table with `QROAMRead.leaf_words_from_table`, which validates the table as revision 1's constructor did. `required_wires` drops its unused `table` argument; `loaded_bits` and `loaded_contents` keep the table form, because the erase builds its masks from the table.
- *Bit order.* Spec 6.7 holds on the wires: wire `i` of a register holds entry bit `width - 1 - i`. The array builder encodes it, so bit `s` of word `w` is the bit for wire `32 w + s`, and the loop body is the one Task 0.9d measured (`p = t // width`, `i = t % width`, word `i // 32`, shift `i % 32`; 77 captured equations per leaf), with no index reversal under capture. The convention lives in one classical helper, checked by hand (`test_leaf_words_by_hand`) and on every address against `loaded_bits` (the T3 tests). The spike stored natural bit order; production does not take its order from the prototype (decision "Task 0.7 review decisions", item 3).
- *Data contract.* Concrete words (Python or NumPy integers, nested sequences or arrays) must be integers in `[0, 2**32)`, booleans and floats rejected, checked on the Python values before the `int64` cast, so that no word wraps around before the check. Every padding and dead position, and every bit beyond `width`, must be zero: a nonzero dead position would load data that the erase's masks do not cover. Tracers and abstract arrays (operators rebuilt by Catalyst, resource keys) are checked for their shape and an integer dtype only. Under capture the array becomes a JAX array (L-23). Violations raise `ValueError`.
- *Rule body.* The unary iteration and the swap network stay unrolled in Python. At each leaf one `qp.for_loop(0, K * width, 1)` applies `qp.cond(bit == 1, CNOT)(flag, register[p * width + i])`, with `register = target ‖ junk` as one JAX array under capture; no conditional surrounds the loop. A leaf without a flag (the whole address low and no enable, revision 1's `None` flag) uses `qp.cond(bit == 1, qp.X)` instead: `test_compiled_read_on_lightning[3]` compiles it and asserts its counts, and `test_every_address_loads_loaded_bits[outer-k3]` checks it on every address.
- *Resources.* The counting twin declares `K width` CNOTs (X gates for the flagless leaf) per leaf, every position and bit with padding included, so the rule is registered with `qp.register_resources(..., exact=False)`; T2's `assert_valid_decomp_rule` then checks that every emitted gate is declared. T, M and A come from the unrolled iteration and swap network and do not depend on the data, so T4 asserts them exactly against `dfthc.costs.qroam_read`, as before. T4 also asserts that the analyzed CNOT count equals the declared one plus two per `CSWAP`: `qp.analyze` counts every unhinted load conditional, so the compiled program and the declared upper bound agree. For the flagless leaf the bound is on X gates: `test_compiled_read_on_lightning[3]` asserts T, M and A and the analyzed X count equal to the declared `K width` (review round 1; a twin that declares one X fewer passes T2 but fails this check). The loops have static bounds, so the counts stay concrete.
- *Adjoint and control.* `QROAMRead` is never adjointed or controlled (spec 7.4 rules 4 and 6). `test_auto_adjoint_is_a_re_read` and spec 9.3's "auto adjoint is a test reference" are dropped: the exhaustive T3 test establishes the read, and the 1C.5 certificates establish that the erase undoes it. Under lever X no adjoint or controlled form of a composite that contains the read is ever requested, so Catalyst synthesizes no `Adjoint(QROAMRead)` or `C(QROAMRead)` candidate; Phase 3's inventory test checks that per program.
- *Compiled read check at K = 16 and width > 32* (Task 0.9d, Opus finding M-2). `test_compiled_multiword_read_with_sixteen_positions` compiles the rule once on lightning for a 6-bit address over `[0, 40)` with `k = 4` and `width = 40`: three leaves, two words per entry, dead positions 40 to 47 and unvisited addresses 48 to 63, with the address and `leaf_words` as runtime arguments. The 640 target and junk wires are aliased onto 7 lightning wires (7 divides no `2**b * 40`, so every `CSWAP` acts on distinct wires; the rule is called directly, because an operator rejects repeated wires, as in the Task 0.9b Opus probe). Every address of the domain must give the folded image of the spec's loads and swap network, and a word-order and a bit-order mutant of the array must both fail. Revision-2 dry run: 4.2 s at 264 MB under the watchdog.
- *Tests.* 46 (44 per commit, 2 slow; revision 1: 37, 35 and 2): the auto-adjoint test is dropped; added are the flagless compiled case, the K = 16 check, `test_leaf_words_by_hand`, revision 1's four table violations (now raised by the builder) and seven leaf-word violations.

**Task 1C.4, `PhaseFixup`.** Unchanged. Its staged `qroam.py` carries the revision-2 read, and `test_qroam_read.py -m "not slow"` now gives 44.

**Task 1C.5, `QROAMErase`.** The erase keeps its compilable table and is unchanged: the read and its erase hold the same table once as dynamic per-leaf words and once as compilable data, the form Task 0.9d measured (its arms P and PX kept arm F's erases). The tests that run a read (the lightning smoke run and the certificates) build its `leaf_words` from the same table. Dynamic erase masks (revision 1's "full remedy") are not adopted: no measurement asks for them, and the masks are still derived inside the rule from the compilable table (decision "K-10 consequences", item 2).

**Counts.** Task 1C.3's per-commit tests rise from 35 to 44, and the G0 base is 225; the expected tiers are in "Expected test counts".

---

### Task 1C.1: Unary iteration helper and `PhaseGradientRotation`

**Task id:** `1C.1` · **Branch:** `phase1/1C.1` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 1A.7, 1B.1 (change 1)

**Files:**
- Create: `src/dfthc/ops/__init__.py`, `src/dfthc/ops/_validate.py`, `src/dfthc/ops/unary.py`, `src/dfthc/ops/phase_gradient.py`, `tests/ops/test_unary.py`, `tests/ops/test_phase_gradient.py`

**Interfaces:**
- Consumes: `dfthc.compile` (`GATE_SET`, `FIXED_DECOMPS`, `compiled_resources`, `register_rule`, `resource_key`; Task 0.2); `dfthc.costs` (`unary`, `unary_nested`, `phase_gradient_rotation`; Task 1A.7); `dfthc.testing.sparse_sim.SparseSimulator`, `SparseState` (Task 1B.1); PennyLane's `TemporaryAND`, `SemiAdder` and `PhaseGradientStatePrep`.
- Produces:
  - `dfthc.ops._validate`: `check_int(name, value, low=None, high=None) -> int`, `check_wires(op, wires, required, *, at_least=())`, `canonical_ints(name, value) -> tuple | int`; each raises `ValueError` naming the expected value.
  - `dfthc.ops.unary`:
    - `class Emitter(counting: bool = False)` with `gate(name, build)` (emit `build()`, or count one gate `name`), `x(wire)`, `cnot(control, target)`, `and_(a, b, target, control_values=(1, 1))`, `unand(a, b, target, control_values=(1, 1))` and the attribute `counts: Counter | None`; subclasses add gates through `gate`;
    - `as_resources(counts) -> dict`, the resource keys of `PauliX`, `PauliZ`, `Hadamard`, `CNOT`, `CZ`, `CSWAP`, `TemporaryAND`, `Adjoint(TemporaryAND)` and `MidMeasure` (with reset);
    - `work_required(widths, controlled) -> int`;
    - `iterate(address, lo, hi, work, leaf, control=None, control_value=1, *, emit=None) -> None`: calls `leaf(flag, value)` for every `value` in `[lo, hi)` in increasing order, with a wire `flag` that is 1 exactly when the big-endian `address` holds `value` and `control` holds `control_value`; `flag` is `None` only for a zero-width address without control; `leaf` uses `flag` only as a control; `work` needs `work_required([len(address)], control is not None)` wires in `|0>` and gets them back;
    - `iterate_nested(registers, work, leaf, control=None, control_value=1, *, emit=None) -> None`: `registers` is a sequence of `(wires, lo, hi)`, outermost first, each inner register iterated under every leaf of the one before; `leaf(flag, values)` receives the tuple of register values;
    - both raise `ValueError` for an empty or out-of-range range, too few work wires, or a control value other than 0 or 1.
  - `dfthc.ops.phase_gradient.PhaseGradientRotation(k, m, target, phase_grad, scratch, work)`, compilable `k`, `m`; `required_wires(k, m) -> {"target": 1, "phase_grad": m, "scratch": m, "work": m - 1}` (`phase_grad` a minimum); the rule `_phase_gradient_rotation`; adjoint auto-derived.

**Acceptance tests** (36 tests):
- `tests/ops/test_unary.py` (15), with the test operator `UnaryIterationProbe`, whose rule copies each leaf flag onto its own wire:
  - T3, exhaustive: every range `[lo, hi)` of every register of 0 to 4 bits, every address, uncontrolled and with control values 0 and 1 (control in `|+>`), raises exactly its flag, returns `work` to `|0>` and keeps amplitude +1/sqrt N (sparse simulator; 3 tests);
  - T4 without compiling: the counted ANDs and measured uncomputes equal `costs.unary` on every range with n <= 6, controlled and not, and `costs.unary_nested` on 300 random nested cases;
  - `work_required` values, and the rejected arguments;
  - three cases (a 3-bit range, a controlled single value, nested 2 + 2 bits with control value 0) compile on `lightning.qubit` to the exact state, and on `null.qubit` to T = A = the closed form (6 tests);
  - T2 at two sizes.
- `tests/ops/test_phase_gradient.py` (21):
  - spec 6.5: `SemiAdder` adding 1 into PennyLane's phase gradient multiplies it by e^{+2πi/2^b};
  - T3: k = 1 and k = 2^m − 1 for m = 1 to 5 against an (m+1)-wire gradient, exact states (5 tests); the adjoint gives the conjugate phase;
  - T4: T = (m − 1)[k != 0] and A = T for (k, m) = (1, 1), (5, 3), adjoint (5, 3), (31, 5) and (0, 4) (5 tests);
  - T5 at `level="device"`; T2 at two sizes; `required_wires`; five constructor violations.

**Definition of done:** `36 passed`; the per-commit tier passes; `ruff check .` clean; the spec 9.0 docstring sections in order; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_unary.py`:

```python
"""Exact unary iteration (spec 9.3 and 9.24): flags on every address, counts, work.

``UnaryIterationProbe`` is a test operator whose rule copies every leaf flag of an exact
iteration onto its own wire, so the flags can be checked on every basis address.
"""

import itertools
import math

import numpy as np
import pennylane as qp
import pytest
from pennylane.core import Operator2

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources, register_rule
from dfthc.ops.unary import Emitter, as_resources, iterate, iterate_nested, work_required
from dfthc.testing.sparse_sim import SparseSimulator, SparseState


class UnaryIterationProbe(Operator2):
    """Copies the flag of leaf ``i`` (in iteration order) onto ``flags[i]``."""

    compilable_argnames = ("widths", "ranges", "control_value")
    wire_argnames = ("address", "flags", "work", "control")

    def __init__(self, widths, ranges, control_value, address, flags, work, control):
        super().__init__(tuple(widths), tuple(tuple(r) for r in ranges), control_value, address,
                         flags, work, control)


def _probe_body(emit, widths, ranges, control_value, address, flags, work, control):
    registers, start = [], 0
    for n, (lo, hi) in zip(widths, ranges, strict=True):
        registers.append((address[start:start + n], lo, hi))
        start += n
    leaves = iter(range(len(flags)))

    def leaf(flag, values):
        target = flags[next(leaves)]
        if flag is None:
            emit.x(target)
        else:
            emit.cnot(flag, target)

    ctrl = None if control_value is None else control[0]
    iterate_nested(registers, work, leaf, ctrl, 1 if control_value is None else control_value,
                   emit=emit)


def _probe_resources(widths, ranges, control_value, address, flags, work, control):
    counter = Emitter(counting=True)
    _probe_body(counter, widths, ranges, control_value, range(len(address)), range(len(flags)),
                range(len(work)), range(len(control)))
    return as_resources(counter.counts)


@qp.register_resources(_probe_resources)
def _unary_iteration_probe(widths, ranges, control_value, address, flags, work, control):
    _probe_body(Emitter(), widths, ranges, control_value, address, flags, work, control)


register_rule(UnaryIterationProbe, _unary_iteration_probe)


def _layout(widths, ranges, control_value):
    """Wire registers of a probe: address, flags, work, then the control."""
    num_flags = math.prod(hi - lo for lo, hi in ranges)
    sizes = [sum(widths), num_flags, work_required(widths, control_value is not None),
             int(control_value is not None)]
    starts = np.cumsum([0] + sizes)
    return [list(range(starts[i], starts[i + 1])) for i in range(4)], int(starts[-1])


def _expected_flags(widths, ranges, control_value, address_value, control_bit):
    """Index of the raised flag for one basis input, or None."""
    if control_value is not None and control_bit != control_value:
        return None
    values, shift = [], sum(widths)
    for n in widths:
        shift -= n
        values.append((address_value >> shift) & (2**n - 1))
    if not all(lo <= v < hi for v, (lo, hi) in zip(values, ranges, strict=True)):
        return None
    index = 0
    for v, (lo, hi) in zip(values, ranges, strict=True):
        index = index * (hi - lo) + (v - lo)
    return index


def _expected_state(widths, ranges, control_value):
    """The probe's output on H|0> on the address and the control, as a dense vector."""
    (address, flags, _, control), num_wires = _layout(widths, ranges, control_value)
    num_address = sum(widths)
    state = np.zeros(2**num_wires, dtype=complex)
    amplitude = 1 / np.sqrt(2 ** (num_address + len(control)))
    for a, c in itertools.product(range(2**num_address), range(2 ** len(control))):
        ones = [address[i] for i in range(num_address) if (a >> (num_address - 1 - i)) & 1]
        ones += control if c else []
        raised = _expected_flags(widths, ranges, control_value, a, c)
        if raised is not None:
            ones.append(flags[raised])
        state[sum(1 << (num_wires - 1 - w) for w in ones)] += amplitude
    return state


def _probe_program(widths, ranges, control_value):
    (address, flags, work, control), num_wires = _layout(widths, ranges, control_value)

    def program():
        for wire in address + control:
            qp.H(wire)
        UnaryIterationProbe(widths, ranges, control_value, address, flags, work, control)

    return program, num_wires


CASES = [
    ((3,), ((1, 7),), None),
    ((4,), ((5, 6),), 1),
    ((2, 2), ((1, 4), (0, 3)), 0),
]


def _simulate_iterate(n, lo, hi, control_value):
    """Record iterate() directly with integer wires and run it on H|0> in the sparse simulator."""
    address = list(range(n))
    flags = list(range(n, n + hi - lo))
    control = [] if control_value is None else [n + hi - lo]
    work = list(range(n + hi - lo + len(control),
                      n + hi - lo + len(control) + work_required([n], bool(control))))

    def leaf(flag, value):
        if flag is None:
            qp.X(flags[value - lo])
        else:
            qp.CNOT([flag, flags[value - lo]])

    with qp.queuing.AnnotatedQueue() as queue:
        for wire in address + control:
            qp.H(wire)
        iterate(address, lo, hi, work, leaf, control[0] if control else None,
                1 if control_value is None else control_value)
    num_wires = n + hi - lo + len(control) + len(work)
    final = SparseSimulator(num_wires).run(queue.queue, SparseState.basis(num_wires, 0))
    return final, address, flags, control, work


@pytest.mark.parametrize("control_value", [None, 0, 1])
def test_every_basis_address_raises_exactly_its_flag(control_value):
    """T3, exhaustive for n <= 4: every range [lo, hi), every address, sparse simulator."""
    for n in range(5):
        for lo, hi in itertools.combinations(range(2**n + 1), 2):
            final, address, flags, control, work = _simulate_iterate(n, lo, hi, control_value)
            assert len(final.amps) == 2 ** (n + len(control))
            assert np.allclose(final.amps, 1 / np.sqrt(len(final.amps)), rtol=0, atol=1e-12)
            for row in range(len(final.amps)):
                a = sum(int(final.bits(w)[row]) << (n - 1 - i) for i, w in enumerate(address))
                c = int(final.bits(control[0])[row]) if control else 0
                raised = [i for i, w in enumerate(flags) if final.bits(w)[row]]
                expected = _expected_flags((n,), ((lo, hi),), control_value, a, c)
                assert raised == ([] if expected is None else [expected]), (n, lo, hi, a, c)
                assert not any(final.bits(w)[row] for w in work), (n, lo, hi, a)


def test_counts_equal_the_closed_form_on_the_grid():
    """T4 without compiling: every range with n <= 6, controlled and not, against costs.unary."""
    for n in range(7):
        for (lo, hi), controlled in itertools.product(
                itertools.combinations(range(2**n + 1), 2), (False, True)):
            counter = Emitter(counting=True)
            iterate(range(n), lo, hi, range(n), lambda flag, value: None,
                    control=-1 if controlled else None, emit=counter)
            expected = costs.unary(lo, hi, n, controlled)
            assert counter.counts["TemporaryAND"] == expected, (lo, hi, n, controlled)
            assert counter.counts["Adjoint(TemporaryAND)"] == expected


def test_nested_counts_equal_the_closed_form():
    rng = np.random.default_rng(93)
    for _ in range(300):
        n1, n2 = int(rng.integers(1, 5)), int(rng.integers(0, 5))
        lo1, hi1 = sorted(int(v) for v in rng.choice(2**n1 + 1, 2, replace=False))
        lo2, hi2 = sorted(int(v) for v in rng.choice(2**n2 + 1, 2, replace=False))
        controlled = bool(rng.integers(2))
        counter = Emitter(counting=True)
        iterate_nested([(range(n1), lo1, hi1), (range(n1, n1 + n2), lo2, hi2)], range(12),
                       lambda flag, values: None, control=-1 if controlled else None,
                       emit=counter)
        expected = costs.unary_nested((lo1, hi1, n1), (lo2, hi2, n2), controlled)
        assert counter.counts["TemporaryAND"] == expected


def test_work_required():
    assert [work_required([n], False) for n in range(5)] == [0, 0, 1, 2, 3]
    assert [work_required([n], True) for n in range(5)] == [0, 1, 2, 3, 4]
    assert work_required([9, 1], False) == 8 + 1  # the FeMoco inner read: rows, then blocks
    assert work_required([0, 3], False) == 2 and work_required([0, 3], True) == 3
    assert work_required([4, 0], True) == 4


def test_invalid_arguments_are_rejected():
    def leaf(flag, value):
        return None

    with pytest.raises(ValueError, match="nonempty range"):
        iterate(range(3), 4, 4, range(3), leaf)
    with pytest.raises(ValueError, match="nonempty range"):
        iterate(range(3), 0, 9, range(3), leaf)
    with pytest.raises(ValueError, match="needs 3 work wires, got 2"):
        iterate(range(3), 0, 8, range(2), leaf, control=-1)
    with pytest.raises(ValueError, match="control_value must be 0 or 1"):
        iterate(range(3), 0, 8, range(3), leaf, control=-1, control_value=2)


@pytest.mark.compiled
@pytest.mark.parametrize(("widths", "ranges", "control_value"), CASES)
def test_compiled_flags_on_lightning(widths, ranges, control_value):
    """T3 on the compiled path: the probe's exact output state, including amplitudes."""
    program, num_wires = _probe_program(widths, ranges, control_value)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        program()
        return qp.state()

    expected = _expected_state(widths, ranges, control_value)
    assert np.allclose(np.asarray(circuit()), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize(("widths", "ranges", "control_value"), CASES)
def test_compiled_counts_equal_the_closed_form(widths, ranges, control_value):
    """T4: compiled T equals U (single) or U_2 (nested), and every AND is uncomputed measured."""
    program, num_wires = _probe_program(widths, ranges, control_value)
    resources = compiled_resources(program, num_wires=num_wires)
    triples = [(lo, hi, n) for n, (lo, hi) in zip(widths, ranges, strict=True)]
    controlled = control_value is not None
    expected = (costs.unary(*triples[0], controlled) if len(triples) == 1
                else costs.unary_nested(triples[0], triples[1], controlled))
    assert resources.toffoli == resources.measured_and_uncomputes == expected
    assert resources.measurements == 0


@pytest.mark.parametrize(("widths", "ranges", "control_value"), [CASES[0], CASES[2]])
def test_rule_is_valid(widths, ranges, control_value):
    """T2: declared resources equal the emitted operations (tape mode)."""
    (address, flags, work, control), _ = _layout(widths, ranges, control_value)
    op = UnaryIterationProbe(widths, ranges, control_value, address, flags, work, control)
    qp.testing.assert_valid_decomp_rule(op, _unary_iteration_probe, check_matrix=False)
```

Create `tests/ops/test_phase_gradient.py`:

```python
"""PhaseGradientRotation (spec 9.1) and the phase-gradient sign convention (spec 6.5)."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.phase_gradient import PhaseGradientRotation, _phase_gradient_rotation


def _gradient(b):
    """PhaseGradientStatePrep's state on b wires: sum_y exp(-2 pi i y / 2^b) |y> / sqrt(2^b)."""
    return np.exp(-2j * np.pi * np.arange(2**b) / 2**b) / np.sqrt(2**b)


def _lightning_state(num_wires, body):
    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return qp.state()

    return np.asarray(circuit())


@pytest.mark.compiled
def test_semiadder_adding_one_kicks_back_the_positive_phase():
    """Spec 6.5: adding 1 into PennyLane's phase gradient multiplies it by exp(+2 pi i / 2^b)."""
    b = 4
    grad, x = [0, 1, 2, 3], [4, 5, 6, 7]

    def body():
        qp.PhaseGradientStatePrep(wires=grad)
        qp.X(x[-1])
        qp.SemiAdder(x, grad, [8, 9, 10])

    one = np.zeros(2**4)
    one[1] = 1.0
    expected = np.exp(2j * np.pi / 2**b) * np.kron(np.kron(_gradient(b), one), np.eye(8)[0])
    assert np.allclose(_lightning_state(11, body), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_rotation_applies_its_phase(m):
    """T3: k = 1 and k = 2^m - 1 on two |+> targets, against an (m+1)-wire phase gradient whose
    top m wires the rotation uses; exact state, so the sign of spec 6.5 is pinned."""
    b = m + 1
    t1, t2 = 0, 1
    grad = list(range(2, 2 + b))
    scratch = list(range(2 + b, 2 + b + m))
    work = list(range(2 + b + m, 2 + b + 2 * m - 1))
    num_wires = 2 + b + 2 * m - 1

    def body():
        qp.PhaseGradientStatePrep(wires=grad)
        qp.H(t1)
        qp.H(t2)
        PhaseGradientRotation(1, m, [t1], grad, scratch, work)
        PhaseGradientRotation(2**m - 1, m, [t2], grad, scratch, work)

    omega = np.exp(2j * np.pi / 2**m)
    targets = np.kron([1, omega], [1, omega ** (2**m - 1)]) / 2
    expected = np.kron(np.kron(targets, _gradient(b)), np.eye(2 ** (2 * m - 1))[0])
    assert np.allclose(_lightning_state(num_wires, body), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
def test_adjoint_applies_the_conjugate_phase():
    """The auto-derived adjoint subtracts: exp(-2 pi i k / 2^m) on |1>."""
    k, m = 3, 3
    grad, scratch, work = [1, 2, 3], [4, 5, 6], [7, 8]

    def body():
        qp.PhaseGradientStatePrep(wires=grad)
        qp.H(0)
        qp.adjoint(PhaseGradientRotation(k, m, [0], grad, scratch, work))

    target = np.array([1, np.exp(-2j * np.pi * k / 2**m)]) / np.sqrt(2)
    expected = np.kron(np.kron(target, _gradient(m)), np.eye(2**5)[0])
    assert np.allclose(_lightning_state(9, body), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize(("k", "m", "adjoint"),
                         [(1, 1, False), (5, 3, False), (5, 3, True), (31, 5, False),
                          (0, 4, False)])
def test_toffoli_count_matches_the_closed_form(k, m, adjoint):
    """T4: T = (m - 1)[k != 0] for the rotation and its adjoint; M = 0."""
    b = m

    def program():
        op = PhaseGradientRotation(k, m, [0], list(range(1, 1 + b)),
                                   list(range(1 + b, 1 + b + m)),
                                   list(range(1 + b + m, 2 * m + b)))
        if adjoint:
            qp.adjoint(op)

    resources = compiled_resources(program, num_wires=2 * m + b)
    expected = costs.phase_gradient_rotation(k, m)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    """T5 at level="device" (L-20)."""

    def program():
        PhaseGradientRotation(5, 3, [0], [1, 2, 3], [4, 5, 6], [7, 8])

    compiled_resources(program, num_wires=9, level="device")


@pytest.mark.parametrize(("k", "m"), [(1, 1), (6, 3)])
def test_rule_is_valid(k, m):
    """T2 at two sizes (tape mode)."""
    op = PhaseGradientRotation(k, m, [0], list(range(1, 1 + m)), list(range(1 + m, 1 + 2 * m)),
                               list(range(1 + 2 * m, 3 * m)))
    qp.testing.assert_valid_decomp_rule(op, _phase_gradient_rotation, check_matrix=False)


def test_required_wires():
    assert PhaseGradientRotation.required_wires(5, 3) == {
        "target": 1, "phase_grad": 3, "scratch": 3, "work": 2}


@pytest.mark.parametrize(
    ("k", "m", "grad", "scratch", "match"),
    [(8, 3, 3, 3, r"k must lie in \[0, 7\]"), (1, 0, 3, 0, r"m must lie in \[1, None\]"),
     (1, 3, 2, 3, "'phase_grad' has 2 wires; expected at least 3"),
     (1, 3, 3, 2, "'scratch' has 2 wires; expected 3"), (1.0, 3, 3, 3, "k must be an int")],
)
def test_constructor_validates(k, m, grad, scratch, match):
    with pytest.raises(ValueError, match=match):
        PhaseGradientRotation(k, m, [0], list(range(1, 1 + grad)),
                              list(range(10, 10 + scratch)), list(range(20, 20 + max(m - 1, 0))))
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_unary.py tests/ops/test_phase_gradient.py -q`
Expected: FAIL at collection, `2 errors`: `ModuleNotFoundError: No module named 'dfthc.ops'`.

- [ ] **Step 3: Write the package, the checks, the helper and the rotation**

Create `src/dfthc/ops/__init__.py`:

```python
"""Operators of layers L1 and L2 (spec section 5.2), one module per block family.

Import operators from their modules, for example ``from dfthc.ops.qroam import QROAMRead``; this
package re-exports nothing.
"""
```

Create `src/dfthc/ops/_validate.py`:

```python
"""Constructor checks shared by the dfthc operators.

Spec 9.0: every constructor validates its wire-register lengths against ``required_wires(...)`` and
its compilable arguments against their documented domains, and raises ``ValueError`` with the
expected value. The checks use ``len()`` only, so they accept wire lists, ``Wires``, the arrays
that rules see and the abstract wires of resource keys.
"""

from __future__ import annotations

import numbers
from collections.abc import Collection, Mapping


def check_int(name: str, value, low: int | None = None, high: int | None = None) -> int:
    """Return ``value`` as a Python ``int``.

    Raises ``ValueError`` unless it is an integer (not a ``bool``) in ``[low, high]``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an int, got {value!r}")
    value = int(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} must lie in [{low}, {high}], got {value}")
    return value


def check_wires(op: str, wires: Mapping[str, object], required: Mapping[str, int], *,
                at_least: Collection[str] = ()) -> None:
    """Compare each register's length with ``required``; names in ``at_least`` may be longer."""
    for name, size in required.items():
        actual = len(wires[name])
        if name in at_least and actual < size:
            raise ValueError(f"{op}: register {name!r} has {actual} wires; expected at least "
                             f"{size} (required_wires)")
        if name not in at_least and actual != size:
            raise ValueError(f"{op}: register {name!r} has {actual} wires; expected {size} "
                             "(required_wires)")


def canonical_ints(name: str, value) -> tuple | int:
    """Nested lists or tuples of integers as nested tuples of Python ints.

    Constructors canonicalize their compilable sequences this way, so equal arguments give one
    rule variant whatever sequence type the caller used (L-21).
    """
    if isinstance(value, (list, tuple)):
        return tuple(canonical_ints(name, item) for item in value)
    return check_int(name, value)
```

Create `src/dfthc/ops/unary.py`:

```python
"""Exact unary iteration, the helper that operator rules call to emit their AND ladders.

Spec sections 9.3 (address model, exact iteration) and 9.24 (the cost ``U``); user decision Q-4.
It is a Python helper used inside rules, not an ``Operator2``.

**Construction.** The iteration walks the binary tree of a big-endian address register (spec
6.7), most significant bit first. Every visited internal node holds a flag wire that is 1 exactly
when the address prefix matches the node and the control is on, and computes its child flag with
one ``TemporaryAND`` on a work wire:

- two visited children: ``child = flag AND NOT bit`` (control values ``(1, 0)``), then the left
  subtree, ``CNOT(flag, child)`` (now ``flag AND bit``), the right subtree, and the measured
  uncompute ``Adjoint(TemporaryAND)``;
- one visited child: one ``TemporaryAND`` with that child's control value, its subtree, and the
  measured uncompute with the same control values.

Negative controls use ``TemporaryAND``'s ``control_values``, a dynamic argument, so they keep a
single resource key. Two places have no AND to carry a control value and use X conjugation: an
uncontrolled root, whose children use the most significant address bit itself as their flag
(X-conjugated for the left child), and ``control_value=0``, which X-conjugates the control wire
around the whole iteration.

The iteration is exact: an address outside ``[lo, hi)`` raises no flag. Qualtran's and dfthcbe's
truncated iteration, which routes some out-of-range addresses onto live leaves, is a later
optimization (spec 4.4, item 8).

**Cost** (spec 9.24): one ``TemporaryAND`` and one ``Adjoint(TemporaryAND)`` per visited internal
node that has a flag, ``U(lo, hi, n, c) = sum_{m=1}^{n} (floor((hi-1)/2^m) - floor(lo/2^m) + 1)
- [c = 0]``, implemented by :func:`dfthc.costs.unary` and :func:`dfthc.costs.unary_nested`.

**Counting.** Every gate goes through an :class:`Emitter`. ``Emitter()`` emits PennyLane
operations; ``Emitter(counting=True)`` only counts them by name, so a rule and its resource
function run the same code, and :func:`as_resources` turns the counts into resource keys.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

import pennylane as qp
from pennylane.typing import Wire

from dfthc.compile import resource_key


class Emitter:
    """Emits gates with PennyLane, or only counts them (``counting=True``).

    Subclasses add gates with :meth:`gate`, which keeps emission and counting in one method.
    """

    def __init__(self, counting: bool = False):
        self.counts: Counter[str] | None = Counter() if counting else None

    def gate(self, name: str, build: Callable[[], object]):
        """Run ``build`` (emit), or count one gate called ``name``."""
        if self.counts is None:
            return build()
        self.counts[name] += 1
        return None

    def x(self, wire):
        self.gate("PauliX", lambda: qp.X(wire))

    def cnot(self, control, target):
        self.gate("CNOT", lambda: qp.CNOT([control, target]))

    def and_(self, a, b, target, control_values=(1, 1)):
        self.gate("TemporaryAND",
                  lambda: qp.TemporaryAND([a, b, target], control_values=control_values))

    def unand(self, a, b, target, control_values=(1, 1)):
        self.gate("Adjoint(TemporaryAND)", lambda: qp.adjoint(
            qp.TemporaryAND([a, b, target], control_values=control_values)))


def as_resources(counts: Counter) -> dict:
    """Resource keys for the gate names an :class:`Emitter` counts (zero counts dropped)."""
    keys = {
        "PauliX": lambda: qp.X, "PauliZ": lambda: qp.Z, "Hadamard": lambda: qp.H,
        "CNOT": lambda: qp.CNOT, "CZ": lambda: qp.CZ, "CSWAP": lambda: qp.CSWAP,
        "TemporaryAND": lambda: resource_key(lambda: qp.TemporaryAND(Wire[3])),
        "Adjoint(TemporaryAND)": lambda: resource_key(
            lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))),
        "MidMeasure": lambda: resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True)),
    }
    return {keys[name](): count for name, count in counts.items() if count}


def work_required(widths: Sequence[int], controlled: bool) -> int:
    """Work wires of a nested iteration over registers of ``widths``, outermost first.

    A register of ``n > 0`` bits needs ``n - 1`` wires, plus one if a flag (the control, or an
    outer leaf) arrives at its root; the paths of all registers are live at an innermost leaf.
    """
    total, flagged = 0, controlled
    for n in widths:
        if n:
            total += n - 1 + int(flagged)
            flagged = True
    return total


def iterate(address, lo: int, hi: int, work, leaf: Callable, control=None, control_value: int = 1,
            *, emit: Emitter | None = None) -> None:
    """Call ``leaf(flag, value)`` for every ``value`` in ``[lo, hi)``, in increasing order.

    ``flag`` is a wire that is 1 exactly when the big-endian register ``address`` holds ``value``
    and the optional ``control`` wire holds ``control_value``. It is ``None`` only for a
    zero-width address without control, where the condition always holds. ``work`` needs at
    least ``work_required([len(address)], control is not None)`` wires in ``|0>``; they are
    returned to ``|0>``. A ``leaf`` may use ``flag`` only as a control.
    """
    iterate_nested([(address, lo, hi)], work, lambda flag, values: leaf(flag, values[0]),
                   control, control_value, emit=emit)


def iterate_nested(registers: Sequence, work, leaf: Callable, control=None,
                   control_value: int = 1, *, emit: Emitter | None = None) -> None:
    """Nested exact iteration over ``registers``, a sequence of ``(wires, lo, hi)``.

    The first register is outermost; under each of its leaves the next register is iterated with
    that leaf's flag as its control. ``leaf(flag, values)`` receives the tuple of register values.
    Work: :func:`work_required` of the register widths.
    """
    emit = Emitter() if emit is None else emit
    widths = [len(wires) for wires, _, _ in registers]
    for (_, lo, hi), n in zip(registers, widths, strict=True):
        if not 0 <= lo < hi <= 2**n:
            raise ValueError(f"range [{lo}, {hi}) is not a nonempty range of {n}-bit values")
    if control_value not in (0, 1):
        raise ValueError(f"control_value must be 0 or 1, got {control_value!r}")
    needed = work_required(widths, control is not None)
    if len(work) < needed:
        raise ValueError(f"the iteration needs {needed} work wires, got {len(work)}")
    if control is not None and control_value == 0:
        emit.x(control)
    _nest(emit, registers, 0, control, (), work, leaf)
    if control is not None and control_value == 0:
        emit.x(control)


def _nest(emit, registers, index, flag, values, work, leaf):
    wires, lo, hi = registers[index]
    n = len(wires)
    depth = n - 1 + int(flag is not None) if n else 0

    def at_leaf(leaf_flag, value):
        if index + 1 == len(registers):
            leaf(leaf_flag, values + (value,))
        else:
            _nest(emit, registers, index + 1, leaf_flag, values + (value,), work[depth:], leaf)

    _node(emit, wires, lo, hi, work[:depth], at_leaf, flag, 0, n)


def _node(emit, wires, lo, hi, work, at_leaf, flag, base, height):
    """Visit the subtree of values ``[base, base + 2**height)`` whose flag wire is ``flag``."""
    if height == 0:
        at_leaf(flag, base)
        return
    bit = wires[len(wires) - height]
    half = 1 << (height - 1)
    left = lo < base + half and base < hi
    right = lo < base + 2 * half and base + half < hi
    if flag is None:
        if left:
            emit.x(bit)
            _node(emit, wires, lo, hi, work, at_leaf, bit, base, height - 1)
            emit.x(bit)
        if right:
            _node(emit, wires, lo, hi, work, at_leaf, bit, base + half, height - 1)
        return
    child, rest = work[0], work[1:]
    if left and right:
        emit.and_(flag, bit, child, (1, 0))
        _node(emit, wires, lo, hi, rest, at_leaf, child, base, height - 1)
        emit.cnot(flag, child)
        _node(emit, wires, lo, hi, rest, at_leaf, child, base + half, height - 1)
        emit.unand(flag, bit, child)
    elif left:
        emit.and_(flag, bit, child, (1, 0))
        _node(emit, wires, lo, hi, rest, at_leaf, child, base, height - 1)
        emit.unand(flag, bit, child, (1, 0))
    else:
        emit.and_(flag, bit, child)
        _node(emit, wires, lo, hi, rest, at_leaf, child, base + half, height - 1)
        emit.unand(flag, bit, child)
```

Create `src/dfthc/ops/phase_gradient.py`:

```python
"""``PhaseGradientRotation``: a Z rotation by phase-gradient addition (spec section 9.1)."""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_int, check_wires


class PhaseGradientRotation(Operator2):
    r"""Multiply ``|t>`` of one qubit by :math:`e^{2\pi i k t/2^m}`: a phase-gradient addition.

    **Action.** :math:`|t\rangle \mapsto e^{2\pi i k t/2^m}|t\rangle` on ``target``; every other
    register is returned unchanged. So ``k = 2**m - 1`` gives :math:`e^{-2\pi i/2^m}` on
    :math:`|1\rangle`.

    **Registers.**

    ============== ================ ========================== ===========================
    name           width            input                      output
    ============== ================ ========================== ===========================
    ``target``     1                any                        phased by ``k t``
    ``phase_grad`` at least ``m``   phase-gradient state       unchanged (catalysed)
    ``scratch``    ``m``            ``|0>``                    ``|0>``
    ``work``       ``m - 1``        ``|0>``                    ``|0>``
    ============== ================ ========================== ===========================

    ``phase_grad`` is the walk's register (:math:`b_{grad}` wires) or ``qpe_grad``; the rotation
    adds into its first (most significant) ``m`` wires, which hold an ``m``-bit phase gradient.

    **Compilable arguments.** ``k`` (int, ``0 <= k < 2**m``) and ``m`` (int, ``m >= 1``).

    **Pre- and postconditions.** ``phase_grad`` holds the state that ``PhaseGradientStatePrep``
    prepares; ``scratch`` and ``work`` start and end in ``|0>``.

    **Decomposition.** CNOTs from ``target`` onto the ``scratch`` bits where ``k`` has ones
    (``scratch`` is big-endian, so it then holds ``k t``), ``SemiAdder(scratch, phase_grad[:m],
    work)``, and the same CNOTs again. Adding ``k t`` into the top ``m`` qubits multiplies the
    phase gradient by :math:`e^{+2\pi i k t/2^m}` (spec 6.5), and the phase kicks back onto
    ``target``. For ``k = 0`` the rule emits nothing.

    **Adjoint and control.** The adjoint is auto-derived: the body has no loops or conditionals,
    and the adjoint ``SemiAdder`` subtracts (K-7). Never controlled.

    **Cost.** :func:`dfthc.costs.phase_gradient_rotation`: :math:`T = (m - 1)[k \ne 0]`, one
    ``m``-bit ``SemiAdder``; ``M = 0``.

    **Paper and dfthcbe.** The phase-gradient rotation of Sanders et al. (PRX Quantum 1, 020312);
    in the paper it realizes the rotations of steps 1, 5, 22 and 27 inside the uniform
    preparations. dfthcbe uses Qualtran's ``ZPowConstViaPhaseGradient``.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.phase_gradient import PhaseGradientRotation

        def program():
            qp.PhaseGradientStatePrep(wires=[1, 2, 3])
            PhaseGradientRotation(5, 3, target=[0], phase_grad=[1, 2, 3], scratch=[4, 5, 6],
                                  work=[7, 8])

        compiled_resources(program, num_wires=9).toffoli  # 2
    """

    compilable_argnames = ("k", "m")
    wire_argnames = ("target", "phase_grad", "scratch", "work")

    def __init__(self, k, m, target, phase_grad, scratch, work):
        m = check_int("m", m, low=1)
        k = check_int("k", k, low=0, high=2**m - 1)
        check_wires("PhaseGradientRotation",
                    {"target": target, "phase_grad": phase_grad, "scratch": scratch,
                     "work": work},
                    self.required_wires(k, m), at_least=("phase_grad",))
        super().__init__(k, m, target, phase_grad, scratch, work)

    @staticmethod
    def required_wires(k: int, m: int) -> dict[str, int]:
        """Register sizes; ``phase_grad`` is a minimum, every other size is exact."""
        return {"target": 1, "phase_grad": m, "scratch": m, "work": m - 1}


def _fan_out(k, m, target, scratch):
    for j in range(m):
        if (k >> (m - 1 - j)) & 1:
            qp.CNOT([target[0], scratch[j]])


def _phase_gradient_rotation_resources(k, m, target, phase_grad, scratch, work):
    if k == 0:
        return {}
    adder = resource_key(lambda: qp.SemiAdder(Wire[m], Wire[m], Wire[m - 1] if m > 1 else None))
    return {qp.CNOT: 2 * bin(k).count("1"), adder: 1}


@qp.register_resources(_phase_gradient_rotation_resources)
def _phase_gradient_rotation(k, m, target, phase_grad, scratch, work):
    if k == 0:
        return
    _fan_out(k, m, target, scratch)
    qp.SemiAdder(scratch, phase_grad[:m], work if m > 1 else None)
    _fan_out(k, m, target, scratch)


register_rule(PhaseGradientRotation, _phase_gradient_rotation)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_unary.py tests/ops/test_phase_gradient.py -q`
Expected: `36 passed` (17 to 36 s, depending on the machine's load; every test under 2 s on a quiet machine).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `414 passed` (378 + 36; see "Expected test counts").

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/__init__.py src/dfthc/ops/_validate.py src/dfthc/ops/unary.py \
        src/dfthc/ops/phase_gradient.py tests/ops/test_unary.py tests/ops/test_phase_gradient.py
git commit -m "Add the exact unary-iteration helper and PhaseGradientRotation"
```

---

### Task 1C.2: `FlaggedUniformPrep`

**Task id:** `1C.2` · **Branch:** `phase1/1C.2` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 1C.1, 1A.5, 1B.2 (change 1)

**Files:**
- Create: `src/dfthc/ops/uniform.py`, `tests/ops/test_uniform.py`

**Interfaces:**
- Consumes: `PhaseGradientRotation` and `dfthc.ops._validate` (Task 1C.1); `dfthc.params.lg`, `odd_part` (1A.1); `dfthc.costs.flagged_uniform_prep`, `flagged_uniform_unprep` (1A.7); `dfthc.data.encoded.uniform_failure_probability` (1A.5); `SparseSimulator` and `decompose_for_simulation` (1B.2); PennyLane's `LeftClassicalComparator` (K-7: its adjoint compiles and is exact).
- Produces, in `dfthc.ops.uniform`:
  - `amplitude_angles(L, s) -> tuple[int, int]`: `(k, k)` with `k = floor(acos(1 - 2**floor(log2 l) / l) / (2 pi) 2**s) mod 2**s` for the odd part `l` of `L` (the integer `uniform_failure_probability` uses), `(0, 0)` for `l = 1`;
  - `FlaggedUniformPrep(L, s, target, succ, succ_work, phase_grad, work)`, compilable `L >= 1` and `s >= 3`; `required_wires(L, s) -> {"target": lg(L), "succ": 1, "succ_work": max(n - 1, 0), "phase_grad": s, "work": n + 2s - 1 if n else 0}` with `n = lg(odd_part(L))` (`phase_grad` a minimum); the rule `_flagged_uniform_prep`; adjoint auto-derived.

**Acceptance tests** (40 tests in `tests/ops/test_uniform.py`):
- T3 on the sparse simulator for L ∈ {3, 5, 6, 7} and s ∈ {3, 7} (8 tests; the s = 7 programs have up to 29 wires): with the catalysed gradient factor divided out, succ = [t < L] on every term, the L amplitudes with t < L are equal, the weight on t >= L equals `uniform_failure_probability(L, s)` to 1e-12, and the norm is 1; `work` returns to `|0>`.
- The prepare followed by its auto-derived adjoint returns every register to `|0>` (5 cases on the sparse simulator, one compiled on lightning).
- Compiled on lightning (s = 3, L = 5 and 6): P(t, succ) has the same structure and failure weight.
- T4: T of the prepare and of the adjoint at L = 324 and 28 (s = 7, a 15-wire gradient: FeMoco's widths, 30 / 24 and 18 / 16), and L = 5, 7, 12 and 8 (12 tests); the FeMoco values on the closed forms; a prepare/unprepare pair has A = T.
- T5 at `level="device"`; T2 at L = 5, 12 and 8; `amplitude_angles` and `required_wires` values; four constructor violations.

**Definition of done:** `40 passed`; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_uniform.py`:

```python
"""FlaggedUniformPrep (spec 9.2): exact uniformity, the failure weight, round trips and counts."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.encoded import uniform_failure_probability
from dfthc.ops.uniform import FlaggedUniformPrep, _flagged_uniform_prep, amplitude_angles
from dfthc.params import lg, odd_part
from dfthc.testing.sparse_sim import SparseSimulator
from dfthc.testing.tape import decompose_for_simulation


def _layout(L, s, grad=None):
    """Registers target, succ, succ_work, phase_grad, work, in that order of wires."""
    sizes = FlaggedUniformPrep.required_wires(L, s)
    sizes["phase_grad"] = s if grad is None else grad
    names = ["target", "succ", "succ_work", "phase_grad", "work"]
    wires, start = {}, 0
    for name in names:
        wires[name] = list(range(start, start + sizes[name]))
        start += sizes[name]
    return wires, start


def _prep(L, s, wires):
    return FlaggedUniformPrep(L, s, wires["target"], wires["succ"], wires["succ_work"],
                              wires["phase_grad"], wires["work"])


def _simulate(L, s, round_trip=False):
    wires, num_wires = _layout(L, s)

    def qfunc():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        _prep(L, s, wires)
        if round_trip:
            qp.adjoint(_prep(L, s, wires))

    tape = decompose_for_simulation(qfunc)
    return SparseSimulator(num_wires).run(tape.operations), wires


def _value(state, register, row):
    return sum(int(state.bits(w)[row]) << (len(register) - 1 - i) for i, w in enumerate(register))


def _without_gradient(state, wires):
    """Amplitudes with the catalysed phase-gradient factor divided out, keyed by the other wires.

    Asserts that the state is (something) x (the phase-gradient state) with work in |0>.
    """
    grad = wires["phase_grad"]
    b = len(grad)
    others = wires["target"] + wires["succ"] + wires["succ_work"]
    amplitudes = {}
    for row, amp in enumerate(state.amps):
        assert not any(state.bits(w)[row] for w in wires["work"])
        y = _value(state, grad, row)
        key = _value(state, others, row)
        amplitudes.setdefault(key, []).append(amp * np.sqrt(2**b) * np.exp(2j * np.pi * y / 2**b))
    for key, values in amplitudes.items():
        assert len(values) == 2**b and np.allclose(values, values[0], rtol=0, atol=1e-12), key
    return {key: values[0] for key, values in amplitudes.items()}, len(others)


@pytest.mark.parametrize("s", [3, 7])
@pytest.mark.parametrize("L", [3, 5, 6, 7])
def test_succ_part_is_exactly_uniform(L, s):
    """T3 (sparse simulator): succ = 1 exactly on t < L with equal amplitudes, succ = 0 only on
    t >= L, and the failure weight equals uniform_failure_probability (spec 8.5)."""
    state, wires = _simulate(L, s)
    amplitudes, num_others = _without_gradient(state, wires)
    shift = num_others - lg(L)
    succ_shift = shift - 1
    good, failure = [], 0.0
    for key, amp in amplitudes.items():
        t, succ = key >> shift, (key >> succ_shift) & 1
        assert succ == int(t < L), (t, succ)
        if t < L:
            good.append(amp)
        else:
            failure += abs(amp) ** 2
    assert len(good) == L and np.allclose(good, good[0], rtol=0, atol=1e-12)
    assert abs(failure - uniform_failure_probability(L, s)) < 1e-12
    assert abs(L * abs(good[0]) ** 2 + failure - 1) < 1e-12


@pytest.mark.parametrize(("L", "s"), [(3, 3), (5, 7), (6, 3), (7, 7), (8, 3)])
def test_adjoint_round_trip_restores_every_register(L, s):
    """The prepare followed by its auto-derived adjoint returns every register to |0>."""
    state, wires = _simulate(L, s, round_trip=True)
    amplitudes, _ = _without_gradient(state, wires)
    assert list(amplitudes) == [0] and abs(amplitudes[0] - 1) < 1e-12


def _lightning(num_wires, body, returns):
    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return returns()

    return np.asarray(circuit())


@pytest.mark.compiled
@pytest.mark.parametrize("L", [5, 6])
def test_compiled_distribution_on_lightning(L):
    """T3 on the compiled path (s = 3): P(t, succ) on lightning.qubit."""
    s = 3
    wires, num_wires = _layout(L, s)

    def body():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        _prep(L, s, wires)

    probs = _lightning(num_wires, body,
                       lambda: qp.probs(wires=wires["target"] + wires["succ"]))
    probs = probs.reshape(2 ** lg(L), 2)
    failure = uniform_failure_probability(L, s)
    for t in range(2 ** lg(L)):
        if t < L:
            assert abs(probs[t, 1] - (1 - failure) / L) < 1e-10 and probs[t, 0] < 1e-10
        else:
            assert probs[t, 1] < 1e-10
    assert abs(probs[L:, 0].sum() - failure) < 1e-10


@pytest.mark.compiled
def test_compiled_round_trip_on_lightning():
    L, s = 7, 3
    wires, num_wires = _layout(L, s)

    def body():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        _prep(L, s, wires)
        qp.adjoint(_prep(L, s, wires))

    grad = np.exp(-2j * np.pi * np.arange(2**s) / 2**s) / np.sqrt(2**s)
    before = 2 ** (lg(L) + 1 + 2)
    after = 2 ** len(wires["work"])
    expected = np.kron(np.kron(np.eye(before)[0], grad), np.eye(after)[0])
    assert np.allclose(_lightning(num_wires, body, qp.state), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize(("L", "s", "grad"), [(324, 7, 15), (28, 7, 15), (5, 7, 7), (7, 7, 7),
                                              (12, 7, 7), (8, 7, 7)])
@pytest.mark.parametrize("adjoint", [False, True], ids=["prepare", "unprepare"])
def test_toffoli_count_matches_the_closed_form(L, s, grad, adjoint):
    """T4: 3n + 2s - 5 and 2n + 2s - 4 for l > 1, 0 for l = 1; FeMoco widths are table-free."""
    wires, num_wires = _layout(L, s, grad)

    def program():
        op = _prep(L, s, wires)
        if adjoint:
            qp.adjoint(op)

    resources = compiled_resources(program, num_wires=num_wires)
    expected = (costs.flagged_uniform_unprep if adjoint else costs.flagged_uniform_prep)(L, s)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)


def test_femoco_values_of_the_closed_form():
    assert (costs.flagged_uniform_prep(324, 7).toffoli,
            costs.flagged_uniform_unprep(324, 7).toffoli) == (30, 24)
    assert (costs.flagged_uniform_prep(28, 7).toffoli,
            costs.flagged_uniform_unprep(28, 7).toffoli) == (18, 16)


@pytest.mark.compiled
def test_prepare_and_unprepare_uncompute_every_and():
    """A prepare/unprepare pair measures every AND it computes (spec 9.24, A = T per pair)."""
    L, s = 12, 4
    wires, num_wires = _layout(L, s)

    def program():
        _prep(L, s, wires)
        qp.adjoint(_prep(L, s, wires))

    resources = compiled_resources(program, num_wires=num_wires)
    expected = costs.flagged_uniform_prep(L, s) + costs.flagged_uniform_unprep(L, s)
    assert resources.toffoli == resources.measured_and_uncomputes == expected.toffoli


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    wires, num_wires = _layout(6, 3)
    compiled_resources(lambda: _prep(6, 3, wires), num_wires=num_wires, level="device")


@pytest.mark.parametrize(("L", "s"), [(5, 3), (12, 4), (8, 3)])
def test_rule_is_valid(L, s):
    wires, _ = _layout(L, s)
    qp.testing.assert_valid_decomp_rule(_prep(L, s, wires), _flagged_uniform_prep,
                                        check_matrix=False)


def test_amplitude_angles():
    assert amplitude_angles(324, 7) == (27, 27)
    assert amplitude_angles(28, 7) == (22, 22)
    assert amplitude_angles(5, 3) == (1, 1)
    assert amplitude_angles(8, 7) == amplitude_angles(1, 3) == (0, 0)


def test_required_wires():
    assert FlaggedUniformPrep.required_wires(324, 7) == {
        "target": 9, "succ": 1, "succ_work": 6, "phase_grad": 7, "work": 20}
    assert FlaggedUniformPrep.required_wires(8, 3) == {
        "target": 3, "succ": 1, "succ_work": 0, "phase_grad": 3, "work": 0}
    assert odd_part(324) == 81 and lg(81) == 7


@pytest.mark.parametrize(("L", "s", "change", "match"), [
    (0, 3, {}, r"L must lie in \[1, None\]"),
    (5, 2, {}, r"s must lie in \[3, None\]"),
    (5, 3, {"phase_grad": 2}, "'phase_grad' has 2 wires; expected at least 3"),
    (5, 3, {"succ_work": 1}, "'succ_work' has 1 wires; expected 2"),
])
def test_constructor_validates(L, s, change, match):
    sizes = FlaggedUniformPrep.required_wires(max(L, 1), max(s, 3)) | change
    wires = {name: list(range(100 * i, 100 * i + size))
             for i, (name, size) in enumerate(sizes.items())}
    with pytest.raises(ValueError, match=match):
        FlaggedUniformPrep(L, s, **wires)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_uniform.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.ops.uniform'`.

- [ ] **Step 3: Write `FlaggedUniformPrep`**

Create `src/dfthc/ops/uniform.py`:

```python
"""``FlaggedUniformPrep``: a uniform superposition over ``[0, L)`` and its flag (spec 9.2)."""

from __future__ import annotations

import math

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_int, check_wires
from dfthc.ops.phase_gradient import PhaseGradientRotation
from dfthc.params import lg, odd_part


def amplitude_angles(L: int, s: int) -> tuple[int, int]:
    r"""``(k1, k2)``: the ``s``-bit phase-gradient integers of the two amplification rotations.

    With ``L = 2**eta l`` and ``l`` odd, both rotations use dfthcbe's angle
    :math:`\theta = \arccos(1 - 2^{\lfloor \log_2 l \rfloor}/l)`, loaded as
    :math:`k = \lfloor \theta/(2\pi)\, 2^s \rfloor \bmod 2^s` (Qualtran's
    ``QFxp(s, s).to_fixed_width_int`` truncates; the same ``k`` as
    :func:`dfthc.data.encoded.uniform_failure_probability`). So ``k1 == k2``; the pair keeps the
    two rotation steps of spec 9.2 apart. ``(0, 0)`` if ``l = 1``.
    """
    odd = odd_part(L)
    if odd == 1:
        return 0, 0
    angle = math.acos(1 - 2 ** math.floor(math.log2(odd)) / odd)
    k = math.floor(angle / (2 * math.pi) * 2**s) % 2**s
    return k, k


class FlaggedUniformPrep(Operator2):
    r"""Prepare a uniform superposition over ``[0, L)`` and flag success (dfthcbe's
    ``UniformViaPhaseGradient``).

    **Action.** From :math:`|0\rangle_t|0\rangle_{succ}`, prepare a state whose ``succ = 1``
    part is exactly uniform over :math:`t \in [0, L)` and whose ``succ = 0`` part is supported on
    :math:`t \ge L`. The weight of the ``succ = 0`` part is
    :func:`dfthc.data.encoded.uniform_failure_probability` ``(L, s)``.

    **Registers.** Write :math:`L = 2^\eta l` with ``l`` odd and :math:`n = \lg l`.

    ============== ============================ ====================== =========================
    name           width                        input                  output
    ============== ============================ ====================== =========================
    ``target``     :math:`\lg L`                ``|0>``                prepared
    ``succ``       1                            ``|0>``                ``[t < L]``
    ``succ_work``  :math:`\max(n - 1, 0)`       ``|0>``                dirty until the adjoint
    ``phase_grad`` at least ``s``               phase-gradient state   unchanged (catalysed)
    ``work``       ``n + 2s - 1`` (0 if l = 1)  ``|0>``                ``|0>``
    ============== ============================ ====================== =========================

    **Compilable arguments.** ``L`` (int, ``L >= 1``) and ``s`` (int, ``s >= 3``): the bits of
    the two rotations, which add into the top ``s`` wires of ``phase_grad``.

    **Pre- and postconditions.** As in the table; ``succ_work`` holds the success comparator's
    intermediate ANDs from the prepare until its adjoint uncomputes them.

    **Decomposition.** If ``l = 1``: ``H`` on every ``target`` wire and ``X(succ)``. Otherwise,
    with ``top = target[:n]`` (spec 9.2, the order of dfthcbe's construction):

    1. ``H`` on every ``target`` wire;
    2. ``LeftClassicalComparator(top, l, "<")`` into ``work[0]``, its work wires ``work[1:n]``;
    3. ``PhaseGradientRotation(k1, s)`` on ``work[0]``, scratch ``work[n:n+s]``, work
       ``work[n+s:]``;
    4. the comparator's adjoint (measured uncompute);
    5. ``H`` on ``top``;
    6. a ``TemporaryAND`` ladder with control values 0 computing ``[top == 0]`` into
       ``work[n-2]``, on ``work[0:n-1]``;
    7. ``PhaseGradientRotation(k2, s)`` on ``work[n-2]``;
    8. the ladder's measured uncompute;
    9. ``H`` on ``top``;
    10. ``LeftClassicalComparator(top, l, "<")`` into ``succ``, its work wires ``succ_work``.

    ``(k1, k2) = amplitude_angles(L, s)``. The low :math:`\eta` qubits keep their Hadamards.

    **Adjoint and control.** The adjoint is auto-derived: the body is unrolled in Python, and the
    comparator adjoints compile (K-7). The adjoint uncomputes ``succ_work``. Never controlled.

    **Cost.** :func:`dfthc.costs.flagged_uniform_prep` and
    :func:`dfthc.costs.flagged_uniform_unprep`: for ``l > 1`` the prepare costs
    :math:`T = 3n + 2s - 5` (two comparators, two rotations, one ladder) and the adjoint
    :math:`T = 2n + 2s - 4` (one comparator, two rotations, one ladder); both 0 if ``l = 1``;
    ``M = 0``. At FeMoco-54: :math:`X_o = 324` gives 30 / 24, ``B + 1 = 28`` gives 18 / 16.

    **Paper and dfthcbe.** Steps 1 and 5 of the circuit (adjoint: 22 and 27); the uniform
    preparation of Babbush et al. (PRX 8, 041015, Fig. 12) with phase-gradient rotations.
    dfthcbe: ``UniformViaPhaseGradient`` (``dfthcbe/prepare.py``).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.uniform import FlaggedUniformPrep

        def program():
            qp.PhaseGradientStatePrep(wires=[4, 5, 6])
            FlaggedUniformPrep(5, 3, target=[0, 1, 2], succ=[3], succ_work=[7, 8],
                               phase_grad=[4, 5, 6], work=list(range(9, 17)))

        compiled_resources(program, num_wires=17).toffoli  # 3 * 3 + 2 * 3 - 5 = 10
    """

    compilable_argnames = ("L", "s")
    wire_argnames = ("target", "succ", "succ_work", "phase_grad", "work")

    def __init__(self, L, s, target, succ, succ_work, phase_grad, work):
        L = check_int("L", L, low=1)
        s = check_int("s", s, low=3)
        check_wires("FlaggedUniformPrep",
                    {"target": target, "succ": succ, "succ_work": succ_work,
                     "phase_grad": phase_grad, "work": work},
                    self.required_wires(L, s), at_least=("phase_grad",))
        super().__init__(L, s, target, succ, succ_work, phase_grad, work)

    @staticmethod
    def required_wires(L: int, s: int) -> dict[str, int]:
        """Register sizes; ``phase_grad`` is a minimum, every other size is exact."""
        n = lg(odd_part(L))
        return {"target": lg(L), "succ": 1, "succ_work": max(n - 1, 0), "phase_grad": s,
                "work": n + 2 * s - 1 if n else 0}


def _comparator(top, odd, target, work):
    return qp.LeftClassicalComparator(top, odd, target, work, "<")


def _rotation(k, s, target, phase_grad, work, n):
    PhaseGradientRotation(k, s, [target], phase_grad, work[n:n + s], work[n + s:n + 2 * s - 1])


def _flagged_uniform_prep_resources(L, s, target, succ, succ_work, phase_grad, work):
    odd, n = odd_part(L), lg(odd_part(L))
    if n == 0:
        return {qp.H: lg(L), qp.X: 1}
    k, _ = amplitude_angles(L, s)
    grad = len(phase_grad)
    comparator = resource_key(
        lambda: qp.LeftClassicalComparator(Wire[n], odd, Wire[1], Wire[n - 1], "<"))
    return {
        qp.H: lg(L) + 2 * n,
        comparator: 2,
        resource_key(lambda: qp.adjoint(
            qp.LeftClassicalComparator(Wire[n], odd, Wire[1], Wire[n - 1], "<"))): 1,
        resource_key(lambda: PhaseGradientRotation(
            k, s, Wire[1], Wire[grad], Wire[s], Wire[s - 1])): 2,
        resource_key(lambda: qp.TemporaryAND(Wire[3])): n - 1,
        resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))): n - 1,
    }


@qp.register_resources(_flagged_uniform_prep_resources)
def _flagged_uniform_prep(L, s, target, succ, succ_work, phase_grad, work):
    odd, n = odd_part(L), lg(odd_part(L))
    for wire in target:
        qp.H(wire)
    if n == 0:
        qp.X(succ[0])
        return
    k1, k2 = amplitude_angles(L, s)
    top = target[:n]
    _comparator(top, odd, work[0:1], work[1:n])
    _rotation(k1, s, work[0], phase_grad, work, n)
    qp.adjoint(_comparator(top, odd, work[0:1], work[1:n]))
    for wire in top:
        qp.H(wire)
    ladder = [(top[0], top[1], work[0], (0, 0))]
    ladder += [(work[j - 2], top[j], work[j - 1], (1, 0)) for j in range(2, n)]
    for a, b, out, values in ladder:
        qp.TemporaryAND([a, b, out], control_values=values)
    _rotation(k2, s, work[n - 2], phase_grad, work, n)
    for a, b, out, values in reversed(ladder):
        qp.adjoint(qp.TemporaryAND([a, b, out], control_values=values))
    for wire in top:
        qp.H(wire)
    _comparator(top, odd, succ, succ_work)


register_rule(FlaggedUniformPrep, _flagged_uniform_prep)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_uniform.py -q`
Expected: `40 passed` (22 to 43 s; the slowest test, the FeMoco-width adjoint, takes 2.7 s and 294 MB on a quiet machine).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `594 passed` (554 + 40).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/uniform.py tests/ops/test_uniform.py
git commit -m "Add FlaggedUniformPrep"
```

---

### Task 1C.3: `QROAMRead`

**Task id:** `1C.3` · **Branch:** `phase1/1C.3` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 1C.1, 1B.2 (change 1). The G0 decision "K-1 at G0" adopted per-leaf read loads, and this task is written for them (revision 2); the "Appendix: per-leaf read loads" before Task 1C.1 lists what changed and why.

**Files:**
- Create: `src/dfthc/ops/qroam.py` (the `QROAMRead` part), `tests/ops/test_qroam_read.py`

**Interfaces:**
- Consumes: `Emitter`, `as_resources`, `iterate_nested`, `work_required` and `dfthc.ops._validate` (1C.1); `dfthc.costs.qroam_read` (1A.7); the presets of `dfthc.params` (1A.1); `SparseSimulator`, `decompose_for_simulation` (1B.2).
- Produces, in `dfthc.ops.qroam`:
  - `WORD = 32`, `WORD_DTYPE = np.int64` and `QROAMEmitter(Emitter)` with `cswap(control, a, b)`, `wire_array(target, junk)` and `load(flag, leaf_words, leaf, register, positions, width)` (one `qp.for_loop` per leaf; counting: `positions * width` gates, the upper bound);
  - `QROAMRead(leaf_words, address_widths, live_ranges, width, block_bits, control_value, address, target, junk, work, control)`:
    - dynamic `leaf_words`: an `int64` array of shape `leaf_words_shape(address_widths, live_ranges, width, block_bits) = (num_leaves, 2**block_bits, ceil(width / 32))`; row `l` is the `l`-th leaf of `leaf_blocks`, position `p` the address `(leaf_blocks[l] << block_bits) | p`; 32 data bits per word in wire order (bit `s` of word `w` is the bit for wire `32 w + s`, entry bit `width - 1 - (32 w + s)`, spec 6.7); data contract: concrete words must be integers in [0, 2**32), booleans and floats rejected, checked before the `int64` cast, and zero at padding and dead positions and beyond `width`; tracers and abstract arrays are checked for shape and an integer dtype only; `ValueError` otherwise;
    - compilable: `address_widths` (one or two widths, at least one wire, the first of two nonempty), `live_ranges` (one `(lo, hi)` per register), `width >= 1`, `0 <= block_bits <= address_widths[-1]`, `control_value` (`None` with a 0-wire `control`, 0 or 1);
  - static methods: `required_wires(address_widths, live_ranges, width, block_bits, control_value) -> dict`, `visited_ranges(address_widths, live_ranges, block_bits) -> list[tuple[int, int, int]]`, `loaded_ranges(address_widths, live_ranges, block_bits) -> tuple[tuple[int, int], ...]`, `loaded_set(address_widths, live_ranges, block_bits) -> list[int]`, `junk_layout(block_bits, low) -> list[int]`, `leaf_blocks(address_widths, live_ranges, block_bits) -> list[int]` (the leaves in visiting order), `leaf_words_shape(...) -> tuple[int, int, int]`, `leaf_words_from_table(table, *, address_widths, live_ranges, width, block_bits) -> np.ndarray` (`table`: `(address, words)` pairs, each address live, each entry below `2**width` in `ceil(width / 32)` words in [0, 2**32), least significant first), `loaded_bits(table, address, *, address_widths, live_ranges, width, block_bits) -> int`, `loaded_contents(table, addresses, *, ...) -> list[int]`;
  - the rule `_qroam_read`, registered with `exact=False`. It is never adjointed or controlled.

**Acceptance tests** (46 tests in `tests/ops/test_qroam_read.py`, 2 of them `slow`):
- T3, exhaustive failed-address tests item 1 of spec 12.1 on the sparse simulator, with `H` on the whole address register (9 tests): inner shapes with rows [0, 2) of a 3-bit `x_o` (`X_o = 5`, so x_o 2 to 4 are one-body rows and 5 to 7 failed) and (B+1, k2) ∈ {(3,1), (5,1), (6,2), (7,0), (28,4)}, widths `w_b + 4`; outer shapes `X_o = 5` with k = 0, 1, 2, 3 (k = 3: the whole address is low, one flagless leaf). Every computational address ends with `target ‖ junk == loaded_bits`, `work` in `|0>`, amplitude +1/sqrt N (no phase), and every address occurs.
- An enable in `|+>` with control value 0 or 1 loads only on its value (2 tests).
- Compiled on lightning at k = 1, 2 and 3: the exact output state; k = 3 compiles the flagless leaf's conditional X loads and also asserts T, M and A and the analyzed X count equal to the declared upper bound, `2**k * width` (3 tests).
- The compiled read check at K = 16 and width 40 (appendix; Task 0.9d, Opus finding M-2): every address of the domain on 7 aliased lightning wires equals the folded image of the spec's loads and swap network; a word-order and a bit-order mutant of the array fail.
- T4: T = U(visited) + (2^k − 1) width, M = 0, A = U, and the analyzed CNOT count equal to the declared upper bound plus two per `CSWAP` (`exact=False`), at `H2_STO3G_TEST` and `H2_631G_TEST`, outer and inner (4 tests), with an enable, and at `MID_NAPHTHALENE` outer and inner (`slow`); the FeMoco ranges `[(0, 81, 7)]` and `[(0, 270, 9), (0, 2, 1)]`, the closed forms 156 and 933, `required_wires` at FeMoco (work 6 and 9), and the leaf-array shapes of the four reads of a FeMoco walk, (81, 4, 1), (540, 16, 1), (54, 1, 25) and (270, 1, 25): 945 leaves.
- T5 at `level="device"`; T2 at two sizes (one enabled; with `exact=False` every emitted gate must be declared); `junk_layout`, `loaded_bits`, `leaf_words_from_table` (wire order, one- and two-word entries) and the range and leaf helpers by hand; four table violations raised by the builder; twelve constructor violations, five structural and seven of the leaf-word contract.

**Definition of done:** `46 passed` targeted (44 in the per-commit tier); the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_qroam_read.py`:

```python
"""QROAMRead (spec 9.3): every computational address, no phase, counts, helpers."""

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from pennylane.typing import AbstractArray

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.qroam import QROAMEmitter, QROAMRead, _qroam_read, _read_body
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, lg
from dfthc.testing.sparse_sim import SparseSimulator
from dfthc.testing.tape import decompose_for_simulation


def random_table(widths, live_ranges, width, seed=7):
    """An entry of ``width`` random bits at every live address, as 32-bit words."""
    rng = np.random.default_rng(seed)
    (lo1, hi1), *rest = live_ranges
    rows = [(v,) for v in range(lo1, hi1)]
    if rest:
        rows = [(v, u) for v in range(lo1, hi1) for u in range(*rest[0])]
    table = []
    for values in rows:
        address = values[0] if len(values) == 1 else (values[0] << widths[1]) | values[1]
        value = int(rng.integers(0, 2**width))
        table.append((address, tuple((value >> (32 * w)) & (2**32 - 1)
                                     for w in range(-(-width // 32)))))
    return tuple(table)


def outer_shape(X_o, k, width):
    """The outer read: x_o over [0, X_o) on lg(X_o) bits."""
    return (lg(X_o),), ((0, X_o),), width, k


def inner_shape(RC, X_o, B1, k, width):
    """The inner read: rows x_o in [0, RC) of lg(X_o) bits, then b in [0, B + 1)."""
    return (lg(X_o), lg(B1)), ((0, RC), (0, B1)), width, k


def layout(widths, live_ranges, width, k, control_value=None):
    sizes = QROAMRead.required_wires(widths, live_ranges, width, k, control_value)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def leaf_words(table, widths, live_ranges, width, k):
    return QROAMRead.leaf_words_from_table(table, address_widths=widths, live_ranges=live_ranges,
                                           width=width, block_bits=k)


def make_read(table, widths, live_ranges, width, k, control_value, wires):
    return QROAMRead(leaf_words(table, widths, live_ranges, width, k), widths, live_ranges,
                     width, k, control_value, wires["address"], wires["target"], wires["junk"],
                     wires["work"], wires["control"])


def value_of(state, register, row):
    return sum(int(state.bits(w)[row]) << (len(register) - 1 - i) for i, w in enumerate(register))


# Exhaustive failed-address shapes (spec 12.1, T3 item 1): X_o = 5 (failed x_o = 5..7), RC = 2
# (one-body rows 2..4), and (B + 1, k2) as listed; widths w_b + 4 as d2 with b_k2 = 2, w_r = 0.
INNER = [inner_shape(2, 5, B1, k, lg(B1) + 4) for B1, k in [(3, 1), (5, 1), (6, 2), (7, 0),
                                                            (28, 4)]]
OUTER = [outer_shape(5, k, 5) for k in (0, 1, 2, 3)]


@pytest.mark.parametrize("shape", INNER + OUTER,
                         ids=[f"inner-{s[1][1][1]}-{s[3]}" for s in INNER]
                         + [f"outer-k{s[3]}" for s in OUTER])
def test_every_address_loads_loaded_bits(shape):
    """T3, exhaustive (sparse simulator): on H|0> over the whole address register, every
    computational address ends with target ‖ junk == loaded_bits, work in |0>, amplitude +1/sqrt N.
    Includes failed x_o >= X_o, failed b >= B + 1, one-body rows, dead block positions, and
    (outer-k3) the flagless leaf of a fully low address."""
    widths, live, width, k = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k)

    def qfunc():
        for wire in wires["address"]:
            qp.H(wire)
        make_read(table, widths, live, width, k, None, wires)

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    size = 2 ** sum(widths)
    assert len(state.amps) == size
    assert np.allclose(state.amps, 1 / np.sqrt(size), rtol=0, atol=1e-12)
    contents = wires["target"] + wires["junk"]
    seen = set()
    for row in range(size):
        a = value_of(state, wires["address"], row)
        seen.add(a)
        expected = QROAMRead.loaded_bits(table, a, address_widths=widths, live_ranges=live,
                                         width=width, block_bits=k)
        assert value_of(state, contents, row) == expected, a
        assert value_of(state, wires["work"], row) == 0
    assert seen == set(range(size))


@pytest.mark.parametrize("control_value", [0, 1])
def test_enable_gates_the_read(control_value):
    """With the enable wire in |+>, only its control_value branch loads anything."""
    widths, live, width, k = INNER[2]
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k, control_value)

    def qfunc():
        for wire in wires["address"] + wires["control"]:
            qp.H(wire)
        make_read(table, widths, live, width, k, control_value, wires)

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    assert len(state.amps) == 2 ** (sum(widths) + 1)
    for row in range(len(state.amps)):
        a = value_of(state, wires["address"], row)
        on = value_of(state, wires["control"], row) == control_value
        expected = QROAMRead.loaded_bits(table, a, address_widths=widths, live_ranges=live,
                                         width=width, block_bits=k) if on else 0
        assert value_of(state, wires["target"] + wires["junk"], row) == expected
        assert value_of(state, wires["work"], row) == 0


def _lightning_state(num_wires, body):
    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return qp.state()

    return np.asarray(circuit())


def _loaded_state(table, widths, live, width, k, wires, num_wires):
    """sum_a |a>|loaded_bits(a)>|0> / sqrt(N) as a dense vector."""
    state = np.zeros(2**num_wires, dtype=complex)
    n = sum(widths)
    rest = num_wires - n
    contents = len(wires["target"]) + len(wires["junk"])
    for a in range(2**n):
        loaded = QROAMRead.loaded_bits(table, a, address_widths=widths, live_ranges=live,
                                       width=width, block_bits=k)
        state[(a << rest) | (loaded << (rest - contents))] = 1 / np.sqrt(2**n)
    return state


@pytest.mark.compiled
@pytest.mark.parametrize("k", [1, 2, 3])
def test_compiled_read_on_lightning(k):
    """T3 on the compiled path: the exact output state, so the read adds no phase. At k = 3 the
    whole 3-bit address is low: one flagless leaf, whose loads are conditional X gates; there T4
    also holds, with the analyzed X count equal to the declared upper bound 2**k * width."""
    widths, live, width, _ = outer_shape(5, k, 2)
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k)

    def body():
        for wire in wires["address"]:
            qp.H(wire)
        make_read(table, widths, live, width, k, None, wires)

    expected = _loaded_state(table, widths, live, width, k, wires, num_wires)
    assert np.allclose(_lightning_state(num_wires, body), expected, rtol=0, atol=1e-10)
    if k == 3:
        resources, declared = _assert_compiled_counts((widths, live, width, k))
        assert resources.counts["PauliX"] == declared["PauliX"] == 2**k * width


# The compiled read check at FeMoco's inner block size and with two-word entries (Task 0.9d,
# Opus M-2): K = 16 positions of width 40 > 32, so every load loop crosses a word boundary.
# Its 640 target and junk wires are aliased onto ALIAS lightning wires: virtual wire j is
# physical wire j mod ALIAS, so each physical wire ends with the XOR of its virtual wires' bits,
# then permuted by the swap network. 7 divides no 2**b * 40, so every CSWAP acts on distinct
# wires. The rule is called directly, because an operator rejects repeated wires.
MULTIWORD = ((6,), ((0, 40),), 40, 4)
ALIAS = 7


def _aliased_expected(table, address):
    """The physical register after the read at ``address``, as a big-endian integer: the
    spec's loads (wire i of position p holds entry bit width - 1 - i) and swap network, with
    every virtual wire folded onto its physical one."""
    widths, live, width, k = MULTIWORD
    entries = {a: sum(w << (32 * i) for i, w in enumerate(words)) for a, words in table}
    bits = [0] * ALIAS
    block, low = address >> k, address & (2**k - 1)
    if block in QROAMRead.leaf_blocks(widths, live, k):
        for p in range(2**k):
            value = entries.get((block << k) | p, 0)
            for i in range(width):
                bits[(p * width + i) % ALIAS] ^= (value >> (width - 1 - i)) & 1
    for b in reversed(range(k)):
        if (low >> b) & 1:
            for r in range(2**b):
                for i in range(width):
                    x, y = (r * width + i) % ALIAS, ((r + 2**b) * width + i) % ALIAS
                    bits[x], bits[y] = bits[y], bits[x]
    return sum(bit << (ALIAS - 1 - j) for j, bit in enumerate(bits))


@pytest.mark.compiled
def test_compiled_multiword_read_with_sixteen_positions():
    """One lightning qjit, the address and leaf_words as runtime arguments, every address of
    the domain (three leaves, dead positions 40 to 47, unvisited 48 to 63): the aliased register
    equals the spec's loads. A word-order and a bit-order mutant of the array both fail."""
    widths, live, width, k = MULTIWORD
    table = random_table(widths, live, width, seed=40)
    words = leaf_words(table, widths, live, width, k)
    assert words.shape == (3, 16, 2)
    n = sum(widths)
    address = list(range(n))
    work = [n]
    physical = list(range(n + 1, n + 1 + ALIAS))
    register = [physical[j % ALIAS] for j in range(2**k * width)]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=n + 1 + ALIAS))
    def circuit(a, array):
        for position, wire in enumerate(address):
            qp.cond(((a >> (n - 1 - position)) & 1) == 1, qp.X)(wire)
        _qroam_read(array, widths, live, width, k, None, address, register[:width],
                    register[width:], work, [])
        return qp.probs(wires=physical)

    def outcomes(array):
        return [np.asarray(circuit(a, jnp.asarray(array))) for a in range(2**n)]

    probabilities = outcomes(words)
    assert all(abs(p.max() - 1) < 1e-9 for p in probabilities)
    assert [int(np.argmax(p)) for p in probabilities] == [
        _aliased_expected(table, a) for a in range(2**n)]
    natural = np.zeros_like(words)
    for leaf, block in enumerate(QROAMRead.leaf_blocks(widths, live, k)):
        for p in range(2**k):
            entry = dict(table).get((block << k) | p, (0, 0))
            natural[leaf, p] = entry
    for mutant in (words[:, :, ::-1], natural):
        assert [int(np.argmax(p)) for p in outcomes(mutant)] != [
            _aliased_expected(table, a) for a in range(2**n)]


def preset_shapes(p):
    return [outer_shape(p.X_o, p.k1, p.d1), inner_shape(p.RC, p.X_o, p.B + 1, p.k2, p.d2)]


def _declared_counts(widths, live, width, k, control_value):
    """The rule's declared gate counts by name (its counting twin)."""
    counter = QROAMEmitter(counting=True)
    wires, _ = layout(widths, live, width, k, control_value)
    _read_body(counter, None, widths, live, width, k, control_value, **wires)
    return counter.counts


def _assert_compiled_counts(shape, control_value=None):
    """T, M and A exact; CNOT equal to the declared upper bound plus two per CSWAP, because
    qp.analyze counts every unhinted load conditional. Returns the analyzed resources and the
    declared counts."""
    widths, live, width, k = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k, control_value)
    resources = compiled_resources(
        lambda: make_read(table, widths, live, width, k, control_value, wires),
        num_wires=num_wires)
    ranges = QROAMRead.visited_ranges(widths, live, k)
    expected = costs.qroam_read(ranges, k, width, control_value is not None)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    swaps = (2**k - 1) * width
    assert resources.measured_and_uncomputes == expected.toffoli - swaps
    declared = _declared_counts(widths, live, width, k, control_value)
    assert declared["CSWAP"] == swaps
    assert resources.counts["CNOT"] == declared["CNOT"] + 2 * swaps
    return resources, declared


@pytest.mark.compiled
@pytest.mark.parametrize("shape", preset_shapes(H2_STO3G_TEST) + preset_shapes(H2_631G_TEST),
                         ids=["sto3g-outer", "sto3g-inner", "631g-outer", "631g-inner"])
def test_toffoli_count_at_the_test_presets(shape):
    """T4 at H2_STO3G_TEST and H2_631G_TEST: T = U(visited) + (2^k - 1) width; A = U."""
    _assert_compiled_counts(shape)


@pytest.mark.compiled
def test_toffoli_count_with_enable():
    _assert_compiled_counts(INNER[2], control_value=0)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("shape", preset_shapes(MID_NAPHTHALENE), ids=["outer", "inner"])
def test_toffoli_count_at_mid_naphthalene(shape):
    """T4 at MID_NAPHTHALENE (nightly tier)."""
    _assert_compiled_counts(shape)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 is asserted on the closed forms; no FeMoco table is compiled (12.9). The four
    reads of a walk have 81 + 540 + 54 + 270 = 945 leaves (Task 0.9d)."""
    p = FEMOCO54
    outer, inner = preset_shapes(p)
    assert QROAMRead.visited_ranges(*outer[:2], outer[3]) == [(0, 81, 7)]
    assert QROAMRead.visited_ranges(*inner[:2], inner[3]) == [(0, 270, 9), (0, 2, 1)]
    assert costs.qroam_read([(0, 81, 7)], p.k1, p.d1, False).toffoli == 84 + 72 == 156
    assert costs.qroam_read([(0, 270, 9), (0, 2, 1)], p.k2, p.d2, False).toffoli == 933
    assert QROAMRead.required_wires(*outer[:2], p.d1, p.k1, None) == {
        "address": 9, "target": 24, "junk": 72, "work": 6, "control": 0}
    assert QROAMRead.required_wires(*inner[:2], p.d2, p.k2, None) == {
        "address": 14, "target": 26, "junk": 390, "work": 9, "control": 0}
    tail = ((p.w_x,), ((p.RC, p.X_o),), (p.N - 1) * p.b_rot + 2, 0)
    sf = ((p.w_r, p.w_b), ((0, p.R), (0, p.B)), (p.N - 1) * p.b_rot + 1, 0)
    shapes = [QROAMRead.leaf_words_shape(*s) for s in (outer, inner, tail, sf)]
    assert shapes == [(81, 4, 1), (540, 16, 1), (54, 1, 25), (270, 1, 25)]
    assert sum(s[0] for s in shapes) == 945


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    widths, live, width, k = INNER[0]
    table = random_table(widths, live, width)
    wires, num_wires = layout(widths, live, width, k)
    compiled_resources(lambda: make_read(table, widths, live, width, k, None, wires),
                       num_wires=num_wires, level="device")


@pytest.mark.parametrize(("shape", "control_value"), [(OUTER[1], None), (INNER[2], 0)])
def test_rule_is_valid(shape, control_value):
    """T2: with exact=False, every emitted gate must be declared."""
    widths, live, width, k = shape
    table = random_table(widths, live, width)
    wires, _ = layout(widths, live, width, k, control_value)
    qp.testing.assert_valid_decomp_rule(make_read(table, widths, live, width, k, control_value,
                                                  wires), _qroam_read, check_matrix=False)


def test_junk_layout():
    assert [QROAMRead.junk_layout(2, low) for low in range(4)] == [
        [1, 2, 3], [0, 2, 3], [3, 0, 1], [2, 0, 1]]
    assert QROAMRead.junk_layout(0, 0) == []


def test_loaded_bits_by_hand():
    """X_o = 5, k = 1, width 2: blocks {0,1}, {2,3}, {4,5 (dead)}; addresses 6, 7 not visited."""
    table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
    args = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
    loaded = [QROAMRead.loaded_bits(table, a, **args) for a in range(8)]
    assert loaded == [(1 << 2) | 2, (2 << 2) | 1, (3 << 2) | 1, (1 << 2) | 3,
                      (2 << 2) | 0, (0 << 2) | 2, 0, 0]


def test_leaf_words_by_hand():
    """Wire order (spec 6.7 on the wires): bit i of the word stream is entry bit width - 1 - i.
    X_o = 5, k = 1, width 2: entries 1 = 0b01 -> 0b10, 2 -> 0b01, 3 -> 0b11; address 5 is dead.
    Width 33: the top entry bit lands in word 0, bit 0, and bit 0 in word 1, bit 0."""
    table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
    args = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
    words = QROAMRead.leaf_words_from_table(table, **args)
    assert words.dtype == np.int64
    assert words.tolist() == [[[2], [1]], [[3], [2]], [[1], [0]]]
    wide = {"address_widths": (1,), "live_ranges": ((0, 2),), "width": 33, "block_bits": 1}
    words = QROAMRead.leaf_words_from_table(((0, (0, 1)), (1, (1, 0))), **wide)
    assert words.tolist() == [[[1, 0], [0, 1]]]


def test_loaded_and_visited_sets():
    assert QROAMRead.visited_ranges((3,), ((0, 5),), 1) == [(0, 3, 2)]
    assert QROAMRead.visited_ranges((3, 3), ((0, 2), (0, 5)), 3) == [(0, 2, 3)]
    assert QROAMRead.loaded_ranges((3, 3), ((0, 2), (0, 5)), 2) == ((0, 2), (0, 8))
    assert QROAMRead.loaded_set((3,), ((0, 5),), 1) == [0, 1, 2, 3, 4, 5]
    assert QROAMRead.loaded_set((2, 2), ((1, 2), (0, 3)), 1) == [4, 5, 6, 7]
    assert QROAMRead.leaf_blocks((3, 3), ((1, 3), (0, 5)), 2) == [2, 3, 4, 5]
    assert QROAMRead.leaf_blocks((3,), ((0, 5),), 3) == [0]
    assert QROAMRead.leaf_words_shape((3, 3), ((1, 3), (0, 5)), 33, 2) == (4, 4, 2)


@pytest.mark.parametrize(("table", "match"), [
    (((5, (1,)),), "outside the live ranges"),
    (((0, (4,)),), "does not fit in 2 bits"),
    (((0, (1, 0)),), "must be 1 words"),
    (((0, (1,)), (0, (2,))), "appears twice"),
])
def test_leaf_words_from_table_validates(table, match):
    with pytest.raises(ValueError, match=match):
        QROAMRead.leaf_words_from_table(table, address_widths=(3,), live_ranges=((0, 5),),
                                        width=2, block_bits=1)


def _words(**changes):
    """Valid leaf_words of the X_o = 5, k = 1, width-2 read, with entries changed."""
    words = np.zeros((3, 2, 1), dtype=object)
    for (leaf, position), word in changes.get("set", {}).items():
        words[leaf, position, 0] = word
    return words


@pytest.mark.parametrize(("change", "match"), [
    ({"block_bits": 4}, r"block_bits must lie in \[0, 3\]"),
    ({"address_widths": (0, 3), "live_ranges": ((0, 1), (0, 5))}, "nonempty first register"),
    ({"live_ranges": ((0, 9),)}, "not a nonempty range of 3-bit values"),
    ({"control_value": 2}, r"control_value must lie in \[0, 1\]"),
    ({"junk": [9]}, "'junk' has 1 wires; expected 2"),
    ({"leaf_words": np.zeros((2, 2, 1), dtype=np.int64)}, r"shape \(3, 2, 1\)"),
    ({"leaf_words": AbstractArray((3, 2, 2), np.int64)}, r"shape \(3, 2, 1\)"),
    ({"leaf_words": _words(set={(0, 0): 2**32 + 1})}, r"integers in \[0, 2\*\*32\)"),
    ({"leaf_words": _words(set={(0, 0): True})}, r"integers in \[0, 2\*\*32\)"),
    ({"leaf_words": _words(set={(0, 0): 1.0})}, r"integers in \[0, 2\*\*32\)"),
    ({"leaf_words": _words(set={(2, 1): 1})}, "padding or dead position"),
    ({"leaf_words": _words(set={(0, 0): 4})}, "bits beyond width 2"),
])
def test_constructor_validates(change, match):
    args = {"leaf_words": np.zeros((3, 2, 1), dtype=np.int64), "address_widths": (3,),
            "live_ranges": ((0, 5),), "width": 2, "block_bits": 1, "control_value": None,
            "address": [0, 1, 2], "target": [3, 4], "junk": [5, 6], "work": [7],
            "control": []} | change
    with pytest.raises(ValueError, match=match):
        QROAMRead(**args)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_qroam_read.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.ops.qroam'`.

- [ ] **Step 3: Write `QROAMRead`**

Create `src/dfthc/ops/qroam.py`:

```python
"""The paper's QROAM read (spec 9.3).

**Address model** (spec 9.3). The address is one or two big-endian registers (spec 6.7),
``address_widths = (w_1,)`` or ``(w_1, w_2)``, concatenated with the first register most
significant: the integer address is ``(a_1 << w_2) | a_2``. Each register has a range
``(lo, hi)``. A table maps integer addresses to entries of ``width`` bits, stored as tuples of
32-bit words, least significant word first (L-24).

**Bit order.** Every register holds an entry big-endian: wire ``i`` of a ``width``-wire register
holds bit ``width - 1 - i``. ``target ‖ junk`` is read as one big-endian integer of
``2**block_bits * width`` bits (``target`` most significant, then the junk slots in order); that
integer is :meth:`QROAMRead.loaded_bits`. The read's dynamic ``leaf_words`` hold the entries in
wire order instead: bit ``s`` of word ``w`` of block position ``p`` is the bit for wire
``32 w + s`` of that position's register, which is entry bit ``width - 1 - (32 w + s)``
(:meth:`QROAMRead.leaf_words_from_table` converts a table).
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import AbstractArray

from dfthc.compile import register_rule
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.unary import Emitter, as_resources, iterate_nested, work_required

WORD = 32
"""Data bits per table word (spec 8.4, L-24)."""

WORD_DTYPE = np.int64
"""dtype of ``leaf_words``: signed, because Catalyst builds no rule for an operator with an
unsigned dynamic argument (L-26); 32 data bits per word keep every word non-negative."""


class QROAMEmitter(Emitter):
    """:class:`~dfthc.ops.unary.Emitter` with the gates of the read."""

    def cswap(self, control, a, b):
        self.gate("CSWAP", lambda: qp.CSWAP([control, a, b]))

    def wire_array(self, target, junk):
        """``target ‖ junk`` as one register that a traced loop index can address: a JAX array
        under capture, a list in tape mode; ``None`` when counting."""
        if self.counts is not None:
            return None
        if not qp.capture.enabled():
            return list(target) + list(junk)
        if len(junk) == 0:
            return jnp.asarray(target)
        return jnp.concatenate([jnp.asarray(target), jnp.asarray(junk)])

    def load(self, flag, leaf_words, leaf, register, positions, width):
        """The loads of unary leaf ``leaf``: one ``qp.for_loop`` over its ``positions * width``
        bits. Iteration ``t`` takes position ``p = t // width`` and bit ``i = t % width`` and
        applies, conditioned on that bit of ``leaf_words``, a CNOT from ``flag`` (an ``X`` for a
        ``None`` flag) onto ``register[p * width + i]``. No conditional surrounds the loop.
        Counting: ``positions * width`` gates, the upper bound."""
        name = "PauliX" if flag is None else "CNOT"
        if self.counts is not None:
            self.counts[name] += positions * width
            return

        @qp.for_loop(0, positions * width, 1)
        def bit_loop(t):
            p, i = t // width, t % width
            bit = (leaf_words[leaf, p, i // WORD] >> (i % WORD)) & 1
            if flag is None:
                qp.cond(bit == 1, qp.X)(register[p * width + i])
            else:
                qp.cond(bit == 1, qp.CNOT)([flag, register[p * width + i]])

        bit_loop()


# --------------------------------------------------------------------------- address helpers


def _check_address(widths, ranges) -> tuple[tuple, tuple]:
    """Canonical ``(address_widths, ranges)``; ``ValueError`` outside the address model."""
    widths = canonical_ints("address_widths", widths)
    ranges = canonical_ints("ranges", ranges)
    if not isinstance(widths, tuple) or len(widths) not in (1, 2):
        raise ValueError(f"an address has one or two registers, got address_widths={widths!r}")
    if any(n < 0 for n in widths) or sum(widths) < 1 or (len(widths) == 2 and widths[0] < 1):
        raise ValueError("address registers need nonnegative widths, at least one wire, and a "
                         f"nonempty first register of two; got {widths}")
    if not isinstance(ranges, tuple) or len(ranges) != len(widths):
        raise ValueError(f"need one (lo, hi) per address register, got {ranges!r}")
    for n, pair in zip(widths, ranges, strict=True):
        if not isinstance(pair, tuple) or len(pair) != 2 or not 0 <= pair[0] < pair[1] <= 2**n:
            raise ValueError(f"range {pair!r} is not a nonempty range of {n}-bit values")
    return widths, ranges


def _check_control_value(control_value):
    if control_value is None:
        return None
    return check_int("control_value", control_value, low=0, high=1)


def _control(control, control_value) -> dict:
    """``iterate_nested``'s control arguments for a 0- or 1-wire ``control`` register."""
    if control_value is None:
        return {"control": None, "control_value": 1}
    return {"control": control[0], "control_value": control_value}


def _high_specs(widths, ranges, low_bits) -> list[tuple[int, int, int, int]]:
    """``(start, n, lo, hi)`` of every register that keeps bits once the low ``low_bits`` bits of
    the concatenated address are taken off; a register split by the boundary keeps its top bits
    and the shifted range. A fully low address gives one zero-width register ``(0, 0, 0, 1)``."""
    starts = list(itertools.accumulate((0, *widths)))
    specs, remaining = [], low_bits
    for index in reversed(range(len(widths))):
        n, (lo, hi) = widths[index], ranges[index]
        if remaining >= n:
            remaining -= n
            continue
        specs.append((starts[index], n - remaining, lo >> remaining,
                      ((hi - 1) >> remaining) + 1))
        remaining = 0
    return list(reversed(specs)) or [(0, 0, 0, 1)]


def _registers(address, specs) -> list:
    return [(address[start:start + n], lo, hi) for start, n, lo, hi in specs]


def _concatenate(values, widths) -> int:
    result = 0
    for value, n in zip(values, widths, strict=True):
        result = (result << n) | value
    return result


def _dummy(sizes: dict[str, int]) -> dict[str, range]:
    """Consecutive integer wires for every register: what resource functions count on."""
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = range(start, start + size)
        start += size
    return wires


# --------------------------------------------------------------------------- QROAMRead


def _check_table(table, widths, live_ranges, width) -> tuple:
    """Canonical table: ``(address, words)`` pairs sorted by address, each address live."""
    table = canonical_ints("table", table)
    words = -(-width // WORD)
    seen = set()
    for pair in table:
        if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[1], tuple):
            raise ValueError(f"table entries are (address, words) pairs, got {pair!r}")
        address, entry = pair
        if address in seen:
            raise ValueError(f"address {address} appears twice in the table")
        seen.add(address)
        if not _is_live(address, widths, live_ranges):
            raise ValueError(f"table entry at address {address} lies outside the live ranges "
                             f"{live_ranges}")
        if len(entry) != words or any(not 0 <= word < 2**WORD for word in entry):
            raise ValueError(f"the entry at address {address} must be {words} words in "
                             f"[0, 2**{WORD}), got {entry}")
        if _entry_value(entry) >> width:
            raise ValueError(f"the entry at address {address} does not fit in {width} bits")
    return tuple(sorted(table))


def _register_values(address, widths) -> list[int]:
    values, shift = [], sum(widths)
    for n in widths:
        shift -= n
        values.append((address >> shift) & (2**n - 1))
    return values


def _is_live(address, widths, ranges) -> bool:
    return 0 <= address < 2 ** sum(widths) and all(
        lo <= value < hi for value, (lo, hi) in zip(_register_values(address, widths), ranges,
                                                    strict=True))


def _entry_value(words) -> int:
    return sum(word << (WORD * i) for i, word in enumerate(words))


def _entries(table) -> dict[int, int]:
    return {address: _entry_value(words) for address, words in table}


def _wire_order(value, width) -> int:
    """``value`` with its ``width`` bits reversed: bit ``i`` of the result is the bit for wire
    ``i`` of a big-endian register (spec 6.7)."""
    return int(format(value, f"0{width}b")[::-1], 2)


def _live_positions(widths, live_ranges, block_bits) -> np.ndarray:
    """``(num_leaves, K)`` booleans: which block positions of each leaf hold a live address."""
    blocks = QROAMRead.leaf_blocks(widths, live_ranges, block_bits)
    return np.array([[_is_live((block << block_bits) | p, widths, live_ranges)
                      for p in range(2**block_bits)] for block in blocks], dtype=bool)


def _leaf_word_array(leaf_words, widths, live_ranges, width, block_bits):
    """Canonicalize the dynamic ``leaf_words`` once (L-23).

    Concrete words (Python or NumPy integers, nested sequences or arrays) must be integers in
    ``[0, 2**32)``, booleans and floats rejected, checked on the Python values before the
    ``int64`` cast, so that no word can wrap around before the check; every padding and dead
    position must be zero, and so must every bit beyond ``width`` (wire order). Tracers and
    abstract arrays (operators rebuilt by Catalyst, resource keys) are checked for their shape
    and an integer dtype only. Raises ``ValueError``.
    """
    shape = QROAMRead.leaf_words_shape(widths, live_ranges, width, block_bits)
    if isinstance(leaf_words, AbstractArray) or qp.math.is_tracer(leaf_words):
        if tuple(leaf_words.shape) != shape or not np.issubdtype(np.dtype(leaf_words.dtype),
                                                                 np.integer):
            raise ValueError(f"leaf_words must be an integer array of shape {shape}, got "
                             f"{tuple(leaf_words.shape)} of {leaf_words.dtype}")
        return leaf_words
    values = np.asarray(leaf_words, dtype=object)
    if values.shape != shape:
        raise ValueError(f"leaf_words must have shape {shape}, got {values.shape}")
    for word in values.flat:
        if isinstance(word, bool) or not isinstance(word, (int, np.integer)) or not (
                0 <= word < 2**WORD):
            raise ValueError(f"leaf words must be integers in [0, 2**{WORD}), got {word!r}")
    array = values.astype(WORD_DTYPE)
    if np.any(array[~_live_positions(widths, live_ranges, block_bits)]):
        raise ValueError("a padding or dead position of leaf_words holds a nonzero word")
    spare = WORD * shape[2] - width
    if spare and np.any(array[..., -1] >> (WORD - spare)):
        raise ValueError(f"leaf_words holds bits beyond width {width}")
    return jnp.asarray(array) if qp.capture.enabled() else array


def _swap_order(block_bits, low) -> list[int]:
    """The block position each register ``target, junk slot 0, ...`` holds after the swap
    network for the low address bits ``low``."""
    held = list(range(2**block_bits))
    for b in reversed(range(block_bits)):
        if (low >> b) & 1:
            for r in range(2**b):
                held[r], held[r + 2**b] = held[r + 2**b], held[r]
    return held


def _loaded_contents(entries, widths, live_ranges, width, block_bits, addresses) -> list[int]:
    """``target ‖ junk`` (big-endian) after an enabled read, for each address."""
    loaded = QROAMRead.loaded_ranges(widths, live_ranges, block_bits)
    size = 2**block_bits
    contents = []
    for address in addresses:
        if not _is_live(address, widths, loaded):
            contents.append(0)
            continue
        base, value = address - (address % size), 0
        for position in _swap_order(block_bits, address % size):
            value = (value << width) | entries.get(base | position, 0)
        contents.append(value)
    return contents


class QROAMRead(Operator2):
    r"""The paper's QROAM read: the requested entry into ``target``, the rest of its block into
    ``junk``, which stays until the measured erase (:class:`QROAMErase`).

    **Action** on every computational address ``a`` (spec 9.3), with :math:`K = 2^k` and
    ``k = block_bits``; a *block* is the set of ``K`` addresses that differ only in the low ``k``
    bits of the last register, and a block is *visited* if it meets the live ranges:

    ============================================== ================ =========================
    case                                           ``target``       ``junk``
    ============================================== ================ =========================
    visited block, live position, enable satisfied :math:`T[a]`     the block's other entries
    visited block, dead position (padding)         0                the block's other entries
    block not visited (e.g. one-body rows,         0                0
    failed :math:`x_o \ge X_o`)
    enable not satisfied                           0                0
    ============================================== ================ =========================

    ``T`` is the table that :meth:`leaf_words_from_table` packed into ``leaf_words``; entries
    missing from it are zero. ``junk`` slot ``r`` holds the block position
    ``junk_layout(k, a mod K)[r]``. :meth:`loaded_bits` gives ``target ‖ junk`` for every address.
    The read adds no phase, and the address is unchanged.

    **Registers.**

    =========== ================================= ======== ===============================
    name        width                             input    output
    =========== ================================= ======== ===============================
    ``address`` ``sum(address_widths)``           any      unchanged
    ``target``  ``width``                         ``|0>``  the entry
    ``junk``    ``(K - 1) width``                 ``|0>``  the block's other entries
    ``work``    unary-iteration ancillas          ``|0>``  ``|0>``
    ``control`` 0 wires, or 1 (the enable)        any      unchanged
    =========== ================================= ======== ===============================

    **Arguments.** Dynamic: ``leaf_words``, an ``int64`` array of shape :meth:`leaf_words_shape`,
    ``(num_leaves, K, ceil(width / 32))``: row ``l`` belongs to the ``l``-th unary leaf in
    visiting order (:meth:`leaf_blocks`), position ``p`` to the address
    ``(leaf_blocks[l] << k) | p``; 32 data bits per word in wire order (module docstring); zero
    at padding and dead positions. Callers build it from their compilable table with
    :meth:`leaf_words_from_table`. Compilable: ``address_widths`` (one or two widths);
    ``live_ranges`` (one ``(lo, hi)`` per register); ``width`` (int >= 1); ``block_bits``
    (``0 <= k <= address_widths[-1]``); ``control_value`` (``None`` without an enable wire, else
    the enabling value 0 or 1).

    **Pre- and postconditions.** ``target``, ``junk`` and ``work`` start in ``|0>``; ``work``
    ends in ``|0>``.

    **Decomposition.** (1) Nested exact unary iteration (:mod:`dfthc.ops.unary`) over the visited
    blocks: the first register over its live range, then the top ``w_p - k`` bits of the last
    register over the blocks that meet its live range, rooted at the enable. (2) At each block
    leaf ``l``, one ``qp.for_loop`` over its ``K width`` bits: iteration ``t`` takes position
    ``p = t // width`` and bit ``i = t % width`` and applies
    ``qp.cond(bit == 1, CNOT)(leaf flag, (target ‖ junk)[p width + i])`` with
    ``bit = (leaf_words[l, p, i // 32] >> (i % 32)) & 1`` (an ``X`` instead of the CNOT when the
    whole address is low and there is no enable, so no flag exists). No conditional surrounds a
    loop. (3) A swap-with-zero network on the low ``k`` address bits, most significant first:
    for bit ``b``, ``CSWAP`` register ``r`` with register ``r + 2**b`` for ``r < 2**b``, which
    moves the selected entry into ``target`` with ``(K - 1) width`` exact ``CSWAP``\s.

    **Adjoint and control.** Never adjointed or controlled: its body has loops and data
    conditionals (spec 7.4 rule 4; L-6), and no program wraps a composite that contains it in an
    adjoint or a control (spec 7.4 rule 6). The walk erases with :class:`QROAMErase`; use the
    enable instead of a control.

    **Cost.** :func:`dfthc.costs.qroam_read` with ``ranges = visited_ranges(...)``:
    :math:`T = U` (or :math:`U_2`) over the visited block ranges, controlled if there is an
    enable, plus :math:`(2^k - 1)` ``width``; ``M = 0``; ``A = U``. At FeMoco-54: outer
    84 + 72 = 156, inner 543 + 390 = 933. The rule declares ``num_leaves K width`` load CNOTs,
    or X gates for the flagless leaf (every position and bit, padding included), and is
    registered with ``exact=False``: the CNOT or X count is an upper bound, which ``qp.analyze``
    reports as well, because it counts every unhinted load conditional; T, M and A do not depend
    on the data and are exact.

    **Paper and dfthcbe.** Steps 2 and 6 (and the engine of 10 and 11); the QROAM of Berry et al.
    (Quantum 3, 208) with the junk kept until a measured erase. dfthcbe: the reads in
    ``OuterPrepare``/``InnerPrepare``; its truncated iteration and flat ``r`` target are
    deferred refinements (spec 9.3, "Known deltas").

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import QROAMRead

        table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
        shape = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
        words = QROAMRead.leaf_words_from_table(table, **shape)  # shape (3, 2, 1)
        program = lambda: QROAMRead(words, *shape.values(), None, address=[0, 1, 2],
                                    target=[3, 4], junk=[5, 6], work=[7], control=[])
        compiled_resources(program, num_wires=8).toffoli  # U(0, 3, 2, 0) + 2 = 2 + 2
    """

    dynamic_argnames = ("leaf_words",)
    compilable_argnames = ("address_widths", "live_ranges", "width", "block_bits",
                           "control_value")
    wire_argnames = ("address", "target", "junk", "work", "control")

    def __init__(self, leaf_words, address_widths, live_ranges, width, block_bits, control_value,
                 address, target, junk, work, control):
        address_widths, live_ranges = _check_address(address_widths, live_ranges)
        width = check_int("width", width, low=1)
        block_bits = check_int("block_bits", block_bits, low=0, high=address_widths[-1])
        control_value = _check_control_value(control_value)
        leaf_words = _leaf_word_array(leaf_words, address_widths, live_ranges, width, block_bits)
        check_wires("QROAMRead",
                    {"address": address, "target": target, "junk": junk, "work": work,
                     "control": control},
                    self.required_wires(address_widths, live_ranges, width, block_bits,
                                        control_value))
        super().__init__(leaf_words, address_widths, live_ranges, width, block_bits,
                         control_value, address, target, junk, work, control)

    @staticmethod
    def required_wires(address_widths, live_ranges, width, block_bits,
                       control_value) -> dict[str, int]:
        """Exact register sizes."""
        ranges = QROAMRead.visited_ranges(address_widths, live_ranges, block_bits)
        return {"address": sum(address_widths), "target": width,
                "junk": (2**block_bits - 1) * width,
                "work": work_required([n for _, _, n in ranges], control_value is not None),
                "control": int(control_value is not None)}

    @staticmethod
    def visited_ranges(address_widths, live_ranges, block_bits) -> list[tuple[int, int, int]]:
        """``(lo, hi, n)`` of each iterated register: the ``ranges`` of
        :func:`dfthc.costs.qroam_read`."""
        widths, ranges = _check_address(address_widths, live_ranges)
        return [(lo, hi, n) for _, n, lo, hi in _high_specs(widths, ranges, block_bits)]

    @staticmethod
    def loaded_ranges(address_widths, live_ranges, block_bits) -> tuple[tuple[int, int], ...]:
        """The loaded set (every address of a visited block) as one ``(lo, hi)`` per register:
        the ``covered`` argument of :class:`PhaseFixup` for this read's erase."""
        widths, ranges = _check_address(address_widths, live_ranges)
        (*outer, (lo, hi)) = ranges
        return (*outer, ((lo >> block_bits) << block_bits,
                         (((hi - 1) >> block_bits) + 1) << block_bits))

    @staticmethod
    def loaded_set(address_widths, live_ranges, block_bits) -> list[int]:
        """Every address of a visited block, in increasing order."""
        widths, _ = _check_address(address_widths, live_ranges)
        loaded = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
        return [_concatenate(values, widths)
                for values in itertools.product(*(range(lo, hi) for lo, hi in loaded))]

    @staticmethod
    def junk_layout(block_bits, low) -> list[int]:
        """The block position held by each junk slot, for low address bits ``low``."""
        return _swap_order(block_bits, low)[1:]

    @staticmethod
    def leaf_blocks(address_widths, live_ranges, block_bits) -> list[int]:
        """The high part of every unary leaf (its block), in the order the iteration visits the
        leaves: outermost register first, values increasing. Row ``l`` of ``leaf_words``."""
        widths, ranges = _check_address(address_widths, live_ranges)
        specs = _high_specs(widths, ranges, block_bits)
        return [_concatenate(values, [n for _, n, _, _ in specs])
                for values in itertools.product(*(range(lo, hi) for _, _, lo, hi in specs))]

    @staticmethod
    def leaf_words_shape(address_widths, live_ranges, width, block_bits) -> tuple[int, int, int]:
        """``(num_leaves, K, ceil(width / 32))``, the shape of ``leaf_words``."""
        return (len(QROAMRead.leaf_blocks(address_widths, live_ranges, block_bits)),
                2**block_bits, -(-width // WORD))

    @staticmethod
    def leaf_words_from_table(table, *, address_widths, live_ranges, width,
                              block_bits) -> np.ndarray:
        """The ``leaf_words`` of ``table`` (``(address, words)`` pairs, every address live, each
        entry below ``2**width`` in 32-bit words, least significant first): each entry in wire
        order at its leaf and position, zero at padding and dead positions. ``ValueError`` for an
        invalid table."""
        widths, ranges = _check_address(address_widths, live_ranges)
        entries = _entries(_check_table(table, widths, ranges, width))
        words = np.zeros(QROAMRead.leaf_words_shape(widths, ranges, width, block_bits),
                         dtype=WORD_DTYPE)
        for leaf, block in enumerate(QROAMRead.leaf_blocks(widths, ranges, block_bits)):
            for position in range(2**block_bits):
                stream = _wire_order(entries.get((block << block_bits) | position, 0), width)
                for w in range(words.shape[2]):
                    words[leaf, position, w] = (stream >> (WORD * w)) & (2**WORD - 1)
        return words

    @staticmethod
    def loaded_bits(table, address, *, address_widths, live_ranges, width, block_bits) -> int:
        """``target ‖ junk`` as one big-endian integer after an enabled read at ``address``;
        0 outside the loaded set."""
        return QROAMRead.loaded_contents(table, [address], address_widths=address_widths,
                                         live_ranges=live_ranges, width=width,
                                         block_bits=block_bits)[0]

    @staticmethod
    def loaded_contents(table, addresses, *, address_widths, live_ranges, width,
                        block_bits) -> list[int]:
        """:meth:`loaded_bits` of each address in ``addresses``, validating the table once."""
        widths, ranges = _check_address(address_widths, live_ranges)
        table = _check_table(table, widths, ranges, width)
        return _loaded_contents(_entries(table), widths, ranges, width, block_bits, addresses)


def _read_body(emit, leaf_words, address_widths, live_ranges, width, block_bits, control_value,
               address, target, junk, work, control):
    registers = [target] + [junk[r * width:(r + 1) * width] for r in range(2**block_bits - 1)]
    register = emit.wire_array(target, junk)
    leaves = itertools.count()

    def leaf(flag, values):
        emit.load(flag, leaf_words, next(leaves), register, 2**block_bits, width)

    iterate_nested(_registers(address, _high_specs(address_widths, live_ranges, block_bits)),
                   work, leaf, emit=emit, **_control(control, control_value))
    low = address[len(address) - block_bits:]
    for b in reversed(range(block_bits)):
        for r in range(2**b):
            for i in range(width):
                emit.cswap(low[block_bits - 1 - b], registers[r][i], registers[r + 2**b][i])


def _qroam_read_resources(leaf_words, address_widths, live_ranges, width, block_bits,
                          control_value, address, target, junk, work, control):
    counter = QROAMEmitter(counting=True)
    wires = _dummy(QROAMRead.required_wires(address_widths, live_ranges, width, block_bits,
                                            control_value))
    _read_body(counter, leaf_words, address_widths, live_ranges, width, block_bits,
               control_value, **wires)
    return as_resources(counter.counts)


@qp.register_resources(_qroam_read_resources, exact=False)
def _qroam_read(leaf_words, address_widths, live_ranges, width, block_bits, control_value,
                address, target, junk, work, control):
    _read_body(QROAMEmitter(), leaf_words, address_widths, live_ranges, width, block_bits,
               control_value, address, target, junk, work, control)


register_rule(QROAMRead, _qroam_read)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_qroam_read.py -q`
Expected: `46 passed` (20 s in the revision-2 replay with one worker, including the two `MID_NAPHTHALENE` compiles, 2.8 s and 3.5 s when each ran alone under `run_watched`, at peaks of 270 and 304 MB, log `.scratch/plan-check/stages/g0-1/mid-read-compiles.log`; not heavy).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `507 passed` (463 + 44).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/qroam.py tests/ops/test_qroam_read.py
git commit -m "Add QROAMRead with exhaustive failed-address tests"
```

---

### Task 1C.4: `PhaseFixup` and `emit_phase_fixup`

**Task id:** `1C.4` · **Branch:** `phase1/1C.4` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 1C.3 (same module; its `loaded_ranges` give the covered sets)

**Files:**
- Modify: `src/dfthc/ops/qroam.py` (replace it with the version below: the 1C.3 code is unchanged; the module docstring, the imports and `QROAMEmitter` grow, and the fixup section is new)
- Create: `tests/ops/test_phase_fixup.py`

**Interfaces:**
- Consumes: Task 1C.3's `qroam.py`; `dfthc.costs.phase_fixup` (1A.7); `SparseSimulator`, `ForcedPolicy`, `decompose_for_simulation` (1B.2).
- Produces, in `dfthc.ops.qroam`:
  - `QROAMEmitter.cond_cz(parity, a, b)` and `cond_z(parity, wire)`: `qp.cond(parity, branch)()` with the branch wrapped in `qp.hint({"branch-prob": 0.5})`;
  - `PhaseFixup(parities, address_widths, covered, onehot_bits, control_value, address, onehot, work, control)`: dynamic `parities` (one per covered address in `address_order`; a bool array under capture, `MeasurementValue`s in tape mode), compilable `address_widths`, `covered` (one `(lo, hi)` per register), `0 <= onehot_bits <= sum(address_widths)`, `control_value` (`None`, 0 or 1);
  - static `required_wires(address_widths, covered, onehot_bits, control_value) -> dict` (`onehot`: `2**ob` wires, 0 for ob = 0), `address_order(address_widths, covered) -> list[int]`, `high_ranges(address_widths, covered, onehot_bits) -> list[tuple[int, int, int]]`;
  - `emit_phase_fixup(parities, *, address, address_widths, covered, onehot, onehot_bits, work, control=None, control_value=1, emit=None) -> None`: the one-hot tree over the low ob bits (copy the top low bit and its negation, then one `TemporaryAND` and one CNOT per split), exact iteration over the high part with a hinted conditional `CZ` (or `Z` for ob = 0) per covered address whose parity is not the Python `False`, and the measured uncompute;
  - the rule `_phase_fixup`.

**Acceptance tests** (27 tests in `tests/ops/test_phase_fixup.py`, 2 of them `slow`):
- T3 (risk K-2), compiled on lightning with the parities as a runtime argument, so one compile serves the random, all-ones and single-bit tables: exactly (−1)^{f(a)} on the covered set and +1 elsewhere, ancillas in `|0>`, for the outer covered set of `X_o = 5`, k = 1 (ob = 0, 1, 2 with an enable of value 1) and the inner covered set of rows [0, 2), (B+1, k2) = (3, 1) (ob = 0, 3 and 2 with an enable of value 0) (6 tests); the ob = 0 cases are the fallback and give the same states.
- The tape path: parities from deterministic measurements on the sparse simulator.
- T4: T = U(covered high ranges, c) + O(ob), M = 0 and A = T for the outer and inner erase shapes of both test presets, and an enabled inner one (5 tests); the FeMoco ranges `[(0, 21, 5)]` and `[(0, 68, 7)]` and the closed forms 22 + 14 and 70 + 126.
- `test_toffoli_count_at_mid_naphthalene` (`slow`, 2 tests): the same at the erase shapes of `MID_NAPHTHALENE`, the outer covered set `[(0, 50)]` with ob = 3 and the inner `[(0, 40), (0, 8)]` with ob = 4 (`compiled_resources` rejects symbolic counts). Each compiled alone under `run_watched`: T = A = 12 = 6 + 6 in 2.1 s at a peak of 272 MB (outer), T = A = 34 = 20 + 14 in 3.7 s at 317 MB (inner); test call times, log `.scratch/plan-check/stages/1c-mid-compiles.log`.
- `test_erase_conditionals_are_hinted` (2 tests): 8 conditional CZs (ob = 3) and 8 conditional Zs (ob = 0) count exactly 4. `test_and_free_fixup_compiles_without_the_unresolved_branch_warning`: a fixup without any AND shows no warning. The warning is one flag per compiled module, and one `TemporaryAND` raises it on its own (master plan section 3, item 15; isolated probes in `.scratch/plan-check/stages/branch-warning-probes.log`), so only these exact counts can show that the conditionals are hinted.
- T5; T2 in tape mode at three sizes; tape mode rejects a bool array; `address_order`, `high_ranges` and `required_wires`; three constructor violations.

**Definition of done:** `27 passed` targeted (25 in the per-commit tier); `scripts/test.sh tests/ops/test_qroam_read.py -m "not slow" -q` still gives `44 passed`; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_phase_fixup.py`:

```python
"""PhaseFixup and emit_phase_fixup (spec 9.4): exact phases on the covered set (risk K-2)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.qroam import PhaseFixup, QROAMRead, _phase_fixup
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator
from dfthc.testing.tape import decompose_for_simulation

# Covered sets of real reads (QROAMRead.loaded_ranges): the outer read X_o = 5, k = 1, and the
# inner read with rows [0, 2) of 3 bits and (B + 1, k2) = (3, 1).
OUTER = ((3,), QROAMRead.loaded_ranges((3,), ((0, 5),), 1))
INNER = ((3, 2), QROAMRead.loaded_ranges((3, 2), ((0, 2), (0, 3)), 1))


def layout(widths, covered, ob, control_value):
    sizes = PhaseFixup.required_wires(widths, covered, ob, control_value)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def make_fixup(parities, widths, covered, ob, control_value, wires):
    return PhaseFixup(parities, widths, covered, ob, control_value, wires["address"],
                      wires["onehot"], wires["work"], wires["control"])


def expected_state(widths, covered, control_value, parities, num_wires):
    """(-1)^f(a) on covered addresses with the enable on, over H|0> on address and control;
    the address wires come first, then the control wire."""
    order = PhaseFixup.address_order(widths, covered)
    phase = dict(zip(order, (-1) ** np.asarray(parities, dtype=int), strict=True))
    n, c = sum(widths), int(control_value is not None)
    state = np.zeros(2**num_wires, dtype=complex)
    for a in range(2**n):
        for bit in range(2**c):
            on = control_value is None or bit == control_value
            index = (a << (num_wires - n)) | (bit << (num_wires - n - 1))
            state[index] = (phase.get(a, 1) if on else 1) / np.sqrt(2 ** (n + c))
    return state


def tables(size, seed=4):
    """Random, all-ones and single-bit parity tables."""
    rng = np.random.default_rng(seed)
    single = np.zeros(size, dtype=bool)
    single[size // 2] = True
    return [rng.integers(0, 2, size).astype(bool), np.ones(size, dtype=bool), single]


CONFIGS = [(OUTER, 1, None), (OUTER, 2, 1), (OUTER, 0, None), (INNER, 3, None), (INNER, 2, 0),
           (INNER, 0, None)]
IDS = ["outer-ob1", "outer-ob2-enable1", "outer-ob0", "inner-ob3", "inner-ob2-enable0",
       "inner-ob0"]


def _move_control_first(wires):
    """Put the control wire right after the address, so expected_state can index it."""
    order = wires["address"] + wires["control"] + wires["onehot"] + wires["work"]
    relabel = {old: new for new, old in enumerate(order)}
    return {name: [relabel[w] for w in register] for name, register in wires.items()}


@pytest.mark.compiled
@pytest.mark.parametrize(("shape", "ob", "control_value"), CONFIGS, ids=IDS)
def test_fixup_applies_exactly_the_parity_table(shape, ob, control_value):
    """T3 (risk K-2): random, all-ones and single-bit tables give exactly (-1)^f(a) on the
    covered set and nothing elsewhere, on lightning.qubit; parities are a runtime argument, so
    one compile serves all three tables. The ob = 0 rows are the fallback of spec 9.4."""
    widths, covered = shape
    size = len(PhaseFixup.address_order(widths, covered))
    wires, num_wires = layout(widths, covered, ob, control_value)
    wires = _move_control_first(wires)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit(parities):
        for wire in wires["address"] + wires["control"]:
            qp.H(wire)
        make_fixup(parities, widths, covered, ob, control_value, wires)
        return qp.state()

    for table in tables(size):
        state = np.asarray(circuit(jnp.asarray(table)))
        expected = expected_state(widths, covered, control_value, table, num_wires)
        assert np.allclose(state, expected, rtol=0, atol=1e-10), table


def test_tape_path_with_measured_parities():
    """The tape path (sparse simulator): parities from deterministic measurements m0 = 1, m1 = 0,
    so the covered addresses at even positions of address_order get the sign."""
    widths, covered = INNER
    ob, size = 3, len(PhaseFixup.address_order(*INNER))
    wires, num_wires = layout(widths, covered, ob, None)
    source = [num_wires, num_wires + 1]

    def qfunc():
        for wire in wires["address"]:
            qp.H(wire)
        qp.X(source[0])
        measured = [qp.measure(source[0]), qp.measure(source[1])]
        make_fixup([measured[i % 2] for i in range(size)], widths, covered, ob, None, wires)

    tape = decompose_for_simulation(qfunc)
    simulator = SparseSimulator(num_wires + 2, policy=ForcedPolicy([1, 0]), strict=False)
    state = simulator.run(tape.operations)
    order = PhaseFixup.address_order(widths, covered)
    sign = {a: -1 if i % 2 == 0 else 1 for i, a in enumerate(order)}
    assert len(state.amps) == 2 ** sum(widths)
    for row, amp in enumerate(state.amps):
        a = sum(int(state.bits(w)[row]) << (4 - i) for i, w in enumerate(wires["address"]))
        assert abs(amp - sign.get(a, 1) / np.sqrt(32)) < 1e-12, a
        assert not any(state.bits(w)[row] for w in wires["onehot"] + wires["work"])


def erase_shapes(p):
    """The covered sets and one-hot bits of the outer and inner erases of preset p (spec 9.24)."""
    outer = ((p.w_x,), QROAMRead.loaded_ranges((p.w_x,), ((0, p.X_o),), p.k1), p.ke1)
    inner_widths = (p.w_x, p.w_b)
    inner = (inner_widths, QROAMRead.loaded_ranges(inner_widths, ((0, p.RC), (0, p.B + 1)),
                                                   p.k2), p.ke2 + p.w_b)
    return [outer, inner]


@pytest.mark.compiled
@pytest.mark.parametrize(("shape", "control_value"),
                         [(s, None) for s in erase_shapes(H2_STO3G_TEST) + erase_shapes(
                             H2_631G_TEST)] + [(erase_shapes(H2_631G_TEST)[1], 1)],
                         ids=["sto3g-outer", "sto3g-inner", "631g-outer", "631g-inner",
                              "631g-inner-enable"])
def test_toffoli_count_matches_the_closed_form(shape, control_value):
    """T4: T = U(covered high ranges, c) + O(ob); every AND is uncomputed by measurement."""
    _assert_compiled_counts(shape, control_value)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("shape", erase_shapes(MID_NAPHTHALENE), ids=["outer", "inner"])
def test_toffoli_count_at_mid_naphthalene(shape):
    """T4 at MID_NAPHTHALENE (nightly tier); compiled_resources rejects symbolic counts."""
    _assert_compiled_counts(shape, None)


def _assert_compiled_counts(shape, control_value):
    widths, covered, ob = shape
    size = len(PhaseFixup.address_order(widths, covered))
    wires, num_wires = layout(widths, covered, ob, control_value)
    resources = compiled_resources(
        lambda: make_fixup(np.ones(size, dtype=bool), widths, covered, ob, control_value, wires),
        num_wires=num_wires)
    expected = costs.phase_fixup(PhaseFixup.high_ranges(widths, covered, ob), ob,
                                 control_value is not None)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli


def test_femoco_counts_on_the_closed_form():
    outer, inner = erase_shapes(FEMOCO54)
    assert PhaseFixup.high_ranges(*outer) == [(0, 21, 5)]
    assert PhaseFixup.high_ranges(*inner) == [(0, 68, 7)]
    assert costs.phase_fixup([(0, 21, 5)], 4, False).toffoli == 22 + 14
    assert costs.phase_fixup([(0, 68, 7)], 7, False).toffoli == 70 + 126


def _compile_recording(widths, covered, ob):
    size = len(PhaseFixup.address_order(widths, covered))
    wires, num_wires = layout(widths, covered, ob, None)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        resources = compiled_resources(
            lambda: make_fixup(np.ones(size, dtype=bool), widths, covered, ob, None, wires),
            num_wires=num_wires)
    unresolved = [r for r in records if "unable to determine the branch" in str(r.message)]
    return size, resources, unresolved


@pytest.mark.compiled
@pytest.mark.parametrize(("shape", "ob", "gate"), [(INNER, 3, "CZ"), (INNER, 0, "PauliZ")])
def test_erase_conditionals_are_hinted(shape, ob, gate):
    """Every conditional carries branch-prob 0.5, so the n conditional gates count n / 2 (L-17:
    expected counts, summed and rounded up once). With n even, a single unhinted conditional
    would count n / 2 + 1. qp.analyze's "unable to determine the branch" warning cannot serve
    here: it fires once per compile if any function of the module branches unhinted, and the
    rules of TemporaryAND that Catalyst lowers into every module with an AND do."""
    size, resources, _ = _compile_recording(*shape, ob)
    assert size % 2 == 0
    assert resources.counts[gate] == size // 2


@pytest.mark.compiled
def test_and_free_fixup_compiles_without_the_unresolved_branch_warning():
    """A fixup without any TemporaryAND (one high bit, uncontrolled, ob = 1): no warning."""
    size, resources, unresolved = _compile_recording((2,), ((0, 4),), 1)
    assert resources.toffoli == 0 and resources.counts["CZ"] == size // 2
    assert not unresolved


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    widths, covered = INNER
    size = len(PhaseFixup.address_order(widths, covered))
    wires, num_wires = layout(widths, covered, 3, None)
    compiled_resources(lambda: make_fixup(np.ones(size, dtype=bool), widths, covered, 3, None,
                                          wires), num_wires=num_wires, level="device")


def _measured(count):
    with qp.queuing.AnnotatedQueue():
        measured = [qp.measure(w) for w in range(100, 103)]
    return [measured[i % 3] for i in range(count)]


@pytest.mark.parametrize(("shape", "ob", "control_value"),
                         [(OUTER, 1, None), (INNER, 3, 0), (INNER, 0, None)])
def test_rule_is_valid_in_tape_mode(shape, ob, control_value):
    """T2 in tape mode, with MeasurementValue parities."""
    widths, covered = shape
    size = len(PhaseFixup.address_order(widths, covered))
    wires, _ = layout(widths, covered, ob, control_value)
    op = make_fixup(_measured(size), widths, covered, ob, control_value, wires)
    qp.testing.assert_valid_decomp_rule(op, _phase_fixup, check_matrix=False)


def test_tape_mode_requires_measurement_values():
    widths, covered = OUTER
    wires, _ = layout(widths, covered, 1, None)
    with pytest.raises(ValueError, match="every parity must be a MeasurementValue; entry 0"):
        make_fixup(np.ones(6, dtype=bool), widths, covered, 1, None, wires)


def test_address_order_and_high_ranges():
    assert PhaseFixup.address_order((3, 2), ((1, 3), (0, 2))) == [4, 5, 8, 9]
    assert PhaseFixup.high_ranges((3, 2), ((1, 3), (0, 2)), 1) == [(1, 3, 3), (0, 1, 1)]
    assert PhaseFixup.high_ranges((3, 2), ((1, 3), (0, 2)), 3) == [(0, 2, 2)]
    assert PhaseFixup.high_ranges((3,), ((0, 6),), 3) == [(0, 1, 0)]
    assert PhaseFixup.required_wires((3, 2), ((0, 2), (0, 4)), 3, 1) == {
        "address": 5, "onehot": 8, "work": 2, "control": 1}


@pytest.mark.parametrize(("change", "match"), [
    ({"parities": np.ones(5, dtype=bool)}, r"one parity per covered address \(6\), got 5"),
    ({"onehot_bits": 4}, r"onehot_bits must lie in \[0, 3\]"),
    ({"onehot": [3]}, "'onehot' has 1 wires; expected 2"),
])
def test_constructor_validates(change, match):
    args = {"parities": _measured(6), "address_widths": (3,), "covered": ((0, 6),),
            "onehot_bits": 1, "control_value": None, "address": [0, 1, 2], "onehot": [3, 4],
            "work": [5], "control": []} | change
    with pytest.raises(ValueError, match=match):
        PhaseFixup(**args)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_phase_fixup.py -q`
Expected: FAIL at collection, `1 error`: `ImportError: cannot import name 'PhaseFixup' from 'dfthc.ops.qroam'`.

- [ ] **Step 3: Add the fixup to `qroam.py`**

Replace `src/dfthc/ops/qroam.py` with:

```python
"""The paper's QROAM read and the erase phase fixup (spec 9.3 and 9.4).

**Address model** (spec 9.3). The address is one or two big-endian registers (spec 6.7),
``address_widths = (w_1,)`` or ``(w_1, w_2)``, concatenated with the first register most
significant: the integer address is ``(a_1 << w_2) | a_2``. Each register has a range
``(lo, hi)``. A table maps integer addresses to entries of ``width`` bits, stored as tuples of
32-bit words, least significant word first (L-24).

**Bit order.** Every register holds an entry big-endian: wire ``i`` of a ``width``-wire register
holds bit ``width - 1 - i``. ``target ‖ junk`` is read as one big-endian integer of
``2**block_bits * width`` bits (``target`` most significant, then the junk slots in order); that
integer is :meth:`QROAMRead.loaded_bits`. The read's dynamic ``leaf_words`` hold the entries in
wire order instead: bit ``s`` of word ``w`` of block position ``p`` is the bit for wire
``32 w + s`` of that position's register, which is entry bit ``width - 1 - (32 w + s)``
(:meth:`QROAMRead.leaf_words_from_table` converts a table).
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.ops import MeasurementValue
from pennylane.typing import AbstractArray

from dfthc.compile import register_rule
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.unary import Emitter, as_resources, iterate_nested, work_required

WORD = 32
"""Data bits per table word (spec 8.4, L-24)."""

WORD_DTYPE = np.int64
"""dtype of ``leaf_words``: signed, because Catalyst builds no rule for an operator with an
unsigned dynamic argument (L-26); 32 data bits per word keep every word non-negative."""


class QROAMEmitter(Emitter):
    """:class:`~dfthc.ops.unary.Emitter` with the gates of the read and the fixup."""

    def cswap(self, control, a, b):
        self.gate("CSWAP", lambda: qp.CSWAP([control, a, b]))

    def wire_array(self, target, junk):
        """``target ‖ junk`` as one register that a traced loop index can address: a JAX array
        under capture, a list in tape mode; ``None`` when counting."""
        if self.counts is not None:
            return None
        if not qp.capture.enabled():
            return list(target) + list(junk)
        if len(junk) == 0:
            return jnp.asarray(target)
        return jnp.concatenate([jnp.asarray(target), jnp.asarray(junk)])

    def load(self, flag, leaf_words, leaf, register, positions, width):
        """The loads of unary leaf ``leaf``: one ``qp.for_loop`` over its ``positions * width``
        bits. Iteration ``t`` takes position ``p = t // width`` and bit ``i = t % width`` and
        applies, conditioned on that bit of ``leaf_words``, a CNOT from ``flag`` (an ``X`` for a
        ``None`` flag) onto ``register[p * width + i]``. No conditional surrounds the loop.
        Counting: ``positions * width`` gates, the upper bound."""
        name = "PauliX" if flag is None else "CNOT"
        if self.counts is not None:
            self.counts[name] += positions * width
            return

        @qp.for_loop(0, positions * width, 1)
        def bit_loop(t):
            p, i = t // width, t % width
            bit = (leaf_words[leaf, p, i // WORD] >> (i % WORD)) & 1
            if flag is None:
                qp.cond(bit == 1, qp.X)(register[p * width + i])
            else:
                qp.cond(bit == 1, qp.CNOT)([flag, register[p * width + i]])

        bit_loop()

    def cond_cz(self, parity, a, b):
        def branch():
            qp.CZ([a, b])

        self.gate("CZ", lambda: qp.cond(parity, qp.hint({"branch-prob": 0.5})(branch))())

    def cond_z(self, parity, wire):
        def branch():
            qp.Z(wire)

        self.gate("PauliZ", lambda: qp.cond(parity, qp.hint({"branch-prob": 0.5})(branch))())


# --------------------------------------------------------------------------- address helpers


def _check_address(widths, ranges) -> tuple[tuple, tuple]:
    """Canonical ``(address_widths, ranges)``; ``ValueError`` outside the address model."""
    widths = canonical_ints("address_widths", widths)
    ranges = canonical_ints("ranges", ranges)
    if not isinstance(widths, tuple) or len(widths) not in (1, 2):
        raise ValueError(f"an address has one or two registers, got address_widths={widths!r}")
    if any(n < 0 for n in widths) or sum(widths) < 1 or (len(widths) == 2 and widths[0] < 1):
        raise ValueError("address registers need nonnegative widths, at least one wire, and a "
                         f"nonempty first register of two; got {widths}")
    if not isinstance(ranges, tuple) or len(ranges) != len(widths):
        raise ValueError(f"need one (lo, hi) per address register, got {ranges!r}")
    for n, pair in zip(widths, ranges, strict=True):
        if not isinstance(pair, tuple) or len(pair) != 2 or not 0 <= pair[0] < pair[1] <= 2**n:
            raise ValueError(f"range {pair!r} is not a nonempty range of {n}-bit values")
    return widths, ranges


def _check_control_value(control_value):
    if control_value is None:
        return None
    return check_int("control_value", control_value, low=0, high=1)


def _control(control, control_value) -> dict:
    """``iterate_nested``'s control arguments for a 0- or 1-wire ``control`` register."""
    if control_value is None:
        return {"control": None, "control_value": 1}
    return {"control": control[0], "control_value": control_value}


def _high_specs(widths, ranges, low_bits) -> list[tuple[int, int, int, int]]:
    """``(start, n, lo, hi)`` of every register that keeps bits once the low ``low_bits`` bits of
    the concatenated address are taken off; a register split by the boundary keeps its top bits
    and the shifted range. A fully low address gives one zero-width register ``(0, 0, 0, 1)``."""
    starts = list(itertools.accumulate((0, *widths)))
    specs, remaining = [], low_bits
    for index in reversed(range(len(widths))):
        n, (lo, hi) = widths[index], ranges[index]
        if remaining >= n:
            remaining -= n
            continue
        specs.append((starts[index], n - remaining, lo >> remaining,
                      ((hi - 1) >> remaining) + 1))
        remaining = 0
    return list(reversed(specs)) or [(0, 0, 0, 1)]


def _registers(address, specs) -> list:
    return [(address[start:start + n], lo, hi) for start, n, lo, hi in specs]


def _concatenate(values, widths) -> int:
    result = 0
    for value, n in zip(values, widths, strict=True):
        result = (result << n) | value
    return result


def _dummy(sizes: dict[str, int]) -> dict[str, range]:
    """Consecutive integer wires for every register: what resource functions count on."""
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = range(start, start + size)
        start += size
    return wires


# --------------------------------------------------------------------------- QROAMRead


def _check_table(table, widths, live_ranges, width) -> tuple:
    """Canonical table: ``(address, words)`` pairs sorted by address, each address live."""
    table = canonical_ints("table", table)
    words = -(-width // WORD)
    seen = set()
    for pair in table:
        if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[1], tuple):
            raise ValueError(f"table entries are (address, words) pairs, got {pair!r}")
        address, entry = pair
        if address in seen:
            raise ValueError(f"address {address} appears twice in the table")
        seen.add(address)
        if not _is_live(address, widths, live_ranges):
            raise ValueError(f"table entry at address {address} lies outside the live ranges "
                             f"{live_ranges}")
        if len(entry) != words or any(not 0 <= word < 2**WORD for word in entry):
            raise ValueError(f"the entry at address {address} must be {words} words in "
                             f"[0, 2**{WORD}), got {entry}")
        if _entry_value(entry) >> width:
            raise ValueError(f"the entry at address {address} does not fit in {width} bits")
    return tuple(sorted(table))


def _register_values(address, widths) -> list[int]:
    values, shift = [], sum(widths)
    for n in widths:
        shift -= n
        values.append((address >> shift) & (2**n - 1))
    return values


def _is_live(address, widths, ranges) -> bool:
    return 0 <= address < 2 ** sum(widths) and all(
        lo <= value < hi for value, (lo, hi) in zip(_register_values(address, widths), ranges,
                                                    strict=True))


def _entry_value(words) -> int:
    return sum(word << (WORD * i) for i, word in enumerate(words))


def _entries(table) -> dict[int, int]:
    return {address: _entry_value(words) for address, words in table}


def _wire_order(value, width) -> int:
    """``value`` with its ``width`` bits reversed: bit ``i`` of the result is the bit for wire
    ``i`` of a big-endian register (spec 6.7)."""
    return int(format(value, f"0{width}b")[::-1], 2)


def _live_positions(widths, live_ranges, block_bits) -> np.ndarray:
    """``(num_leaves, K)`` booleans: which block positions of each leaf hold a live address."""
    blocks = QROAMRead.leaf_blocks(widths, live_ranges, block_bits)
    return np.array([[_is_live((block << block_bits) | p, widths, live_ranges)
                      for p in range(2**block_bits)] for block in blocks], dtype=bool)


def _leaf_word_array(leaf_words, widths, live_ranges, width, block_bits):
    """Canonicalize the dynamic ``leaf_words`` once (L-23).

    Concrete words (Python or NumPy integers, nested sequences or arrays) must be integers in
    ``[0, 2**32)``, booleans and floats rejected, checked on the Python values before the
    ``int64`` cast, so that no word can wrap around before the check; every padding and dead
    position must be zero, and so must every bit beyond ``width`` (wire order). Tracers and
    abstract arrays (operators rebuilt by Catalyst, resource keys) are checked for their shape
    and an integer dtype only. Raises ``ValueError``.
    """
    shape = QROAMRead.leaf_words_shape(widths, live_ranges, width, block_bits)
    if isinstance(leaf_words, AbstractArray) or qp.math.is_tracer(leaf_words):
        if tuple(leaf_words.shape) != shape or not np.issubdtype(np.dtype(leaf_words.dtype),
                                                                 np.integer):
            raise ValueError(f"leaf_words must be an integer array of shape {shape}, got "
                             f"{tuple(leaf_words.shape)} of {leaf_words.dtype}")
        return leaf_words
    values = np.asarray(leaf_words, dtype=object)
    if values.shape != shape:
        raise ValueError(f"leaf_words must have shape {shape}, got {values.shape}")
    for word in values.flat:
        if isinstance(word, bool) or not isinstance(word, (int, np.integer)) or not (
                0 <= word < 2**WORD):
            raise ValueError(f"leaf words must be integers in [0, 2**{WORD}), got {word!r}")
    array = values.astype(WORD_DTYPE)
    if np.any(array[~_live_positions(widths, live_ranges, block_bits)]):
        raise ValueError("a padding or dead position of leaf_words holds a nonzero word")
    spare = WORD * shape[2] - width
    if spare and np.any(array[..., -1] >> (WORD - spare)):
        raise ValueError(f"leaf_words holds bits beyond width {width}")
    return jnp.asarray(array) if qp.capture.enabled() else array


def _swap_order(block_bits, low) -> list[int]:
    """The block position each register ``target, junk slot 0, ...`` holds after the swap
    network for the low address bits ``low``."""
    held = list(range(2**block_bits))
    for b in reversed(range(block_bits)):
        if (low >> b) & 1:
            for r in range(2**b):
                held[r], held[r + 2**b] = held[r + 2**b], held[r]
    return held


def _loaded_contents(entries, widths, live_ranges, width, block_bits, addresses) -> list[int]:
    """``target ‖ junk`` (big-endian) after an enabled read, for each address."""
    loaded = QROAMRead.loaded_ranges(widths, live_ranges, block_bits)
    size = 2**block_bits
    contents = []
    for address in addresses:
        if not _is_live(address, widths, loaded):
            contents.append(0)
            continue
        base, value = address - (address % size), 0
        for position in _swap_order(block_bits, address % size):
            value = (value << width) | entries.get(base | position, 0)
        contents.append(value)
    return contents


class QROAMRead(Operator2):
    r"""The paper's QROAM read: the requested entry into ``target``, the rest of its block into
    ``junk``, which stays until the measured erase (:class:`QROAMErase`).

    **Action** on every computational address ``a`` (spec 9.3), with :math:`K = 2^k` and
    ``k = block_bits``; a *block* is the set of ``K`` addresses that differ only in the low ``k``
    bits of the last register, and a block is *visited* if it meets the live ranges:

    ============================================== ================ =========================
    case                                           ``target``       ``junk``
    ============================================== ================ =========================
    visited block, live position, enable satisfied :math:`T[a]`     the block's other entries
    visited block, dead position (padding)         0                the block's other entries
    block not visited (e.g. one-body rows,         0                0
    failed :math:`x_o \ge X_o`)
    enable not satisfied                           0                0
    ============================================== ================ =========================

    ``T`` is the table that :meth:`leaf_words_from_table` packed into ``leaf_words``; entries
    missing from it are zero. ``junk`` slot ``r`` holds the block position
    ``junk_layout(k, a mod K)[r]``. :meth:`loaded_bits` gives ``target ‖ junk`` for every address.
    The read adds no phase, and the address is unchanged.

    **Registers.**

    =========== ================================= ======== ===============================
    name        width                             input    output
    =========== ================================= ======== ===============================
    ``address`` ``sum(address_widths)``           any      unchanged
    ``target``  ``width``                         ``|0>``  the entry
    ``junk``    ``(K - 1) width``                 ``|0>``  the block's other entries
    ``work``    unary-iteration ancillas          ``|0>``  ``|0>``
    ``control`` 0 wires, or 1 (the enable)        any      unchanged
    =========== ================================= ======== ===============================

    **Arguments.** Dynamic: ``leaf_words``, an ``int64`` array of shape :meth:`leaf_words_shape`,
    ``(num_leaves, K, ceil(width / 32))``: row ``l`` belongs to the ``l``-th unary leaf in
    visiting order (:meth:`leaf_blocks`), position ``p`` to the address
    ``(leaf_blocks[l] << k) | p``; 32 data bits per word in wire order (module docstring); zero
    at padding and dead positions. Callers build it from their compilable table with
    :meth:`leaf_words_from_table`. Compilable: ``address_widths`` (one or two widths);
    ``live_ranges`` (one ``(lo, hi)`` per register); ``width`` (int >= 1); ``block_bits``
    (``0 <= k <= address_widths[-1]``); ``control_value`` (``None`` without an enable wire, else
    the enabling value 0 or 1).

    **Pre- and postconditions.** ``target``, ``junk`` and ``work`` start in ``|0>``; ``work``
    ends in ``|0>``.

    **Decomposition.** (1) Nested exact unary iteration (:mod:`dfthc.ops.unary`) over the visited
    blocks: the first register over its live range, then the top ``w_p - k`` bits of the last
    register over the blocks that meet its live range, rooted at the enable. (2) At each block
    leaf ``l``, one ``qp.for_loop`` over its ``K width`` bits: iteration ``t`` takes position
    ``p = t // width`` and bit ``i = t % width`` and applies
    ``qp.cond(bit == 1, CNOT)(leaf flag, (target ‖ junk)[p width + i])`` with
    ``bit = (leaf_words[l, p, i // 32] >> (i % 32)) & 1`` (an ``X`` instead of the CNOT when the
    whole address is low and there is no enable, so no flag exists). No conditional surrounds a
    loop. (3) A swap-with-zero network on the low ``k`` address bits, most significant first:
    for bit ``b``, ``CSWAP`` register ``r`` with register ``r + 2**b`` for ``r < 2**b``, which
    moves the selected entry into ``target`` with ``(K - 1) width`` exact ``CSWAP``\s.

    **Adjoint and control.** Never adjointed or controlled: its body has loops and data
    conditionals (spec 7.4 rule 4; L-6), and no program wraps a composite that contains it in an
    adjoint or a control (spec 7.4 rule 6). The walk erases with :class:`QROAMErase`; use the
    enable instead of a control.

    **Cost.** :func:`dfthc.costs.qroam_read` with ``ranges = visited_ranges(...)``:
    :math:`T = U` (or :math:`U_2`) over the visited block ranges, controlled if there is an
    enable, plus :math:`(2^k - 1)` ``width``; ``M = 0``; ``A = U``. At FeMoco-54: outer
    84 + 72 = 156, inner 543 + 390 = 933. The rule declares ``num_leaves K width`` load CNOTs,
    or X gates for the flagless leaf (every position and bit, padding included), and is
    registered with ``exact=False``: the CNOT or X count is an upper bound, which ``qp.analyze``
    reports as well, because it counts every unhinted load conditional; T, M and A do not depend
    on the data and are exact.

    **Paper and dfthcbe.** Steps 2 and 6 (and the engine of 10 and 11); the QROAM of Berry et al.
    (Quantum 3, 208) with the junk kept until a measured erase. dfthcbe: the reads in
    ``OuterPrepare``/``InnerPrepare``; its truncated iteration and flat ``r`` target are
    deferred refinements (spec 9.3, "Known deltas").

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import QROAMRead

        table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
        shape = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
        words = QROAMRead.leaf_words_from_table(table, **shape)  # shape (3, 2, 1)
        program = lambda: QROAMRead(words, *shape.values(), None, address=[0, 1, 2],
                                    target=[3, 4], junk=[5, 6], work=[7], control=[])
        compiled_resources(program, num_wires=8).toffoli  # U(0, 3, 2, 0) + 2 = 2 + 2
    """

    dynamic_argnames = ("leaf_words",)
    compilable_argnames = ("address_widths", "live_ranges", "width", "block_bits",
                           "control_value")
    wire_argnames = ("address", "target", "junk", "work", "control")

    def __init__(self, leaf_words, address_widths, live_ranges, width, block_bits, control_value,
                 address, target, junk, work, control):
        address_widths, live_ranges = _check_address(address_widths, live_ranges)
        width = check_int("width", width, low=1)
        block_bits = check_int("block_bits", block_bits, low=0, high=address_widths[-1])
        control_value = _check_control_value(control_value)
        leaf_words = _leaf_word_array(leaf_words, address_widths, live_ranges, width, block_bits)
        check_wires("QROAMRead",
                    {"address": address, "target": target, "junk": junk, "work": work,
                     "control": control},
                    self.required_wires(address_widths, live_ranges, width, block_bits,
                                        control_value))
        super().__init__(leaf_words, address_widths, live_ranges, width, block_bits,
                         control_value, address, target, junk, work, control)

    @staticmethod
    def required_wires(address_widths, live_ranges, width, block_bits,
                       control_value) -> dict[str, int]:
        """Exact register sizes."""
        ranges = QROAMRead.visited_ranges(address_widths, live_ranges, block_bits)
        return {"address": sum(address_widths), "target": width,
                "junk": (2**block_bits - 1) * width,
                "work": work_required([n for _, _, n in ranges], control_value is not None),
                "control": int(control_value is not None)}

    @staticmethod
    def visited_ranges(address_widths, live_ranges, block_bits) -> list[tuple[int, int, int]]:
        """``(lo, hi, n)`` of each iterated register: the ``ranges`` of
        :func:`dfthc.costs.qroam_read`."""
        widths, ranges = _check_address(address_widths, live_ranges)
        return [(lo, hi, n) for _, n, lo, hi in _high_specs(widths, ranges, block_bits)]

    @staticmethod
    def loaded_ranges(address_widths, live_ranges, block_bits) -> tuple[tuple[int, int], ...]:
        """The loaded set (every address of a visited block) as one ``(lo, hi)`` per register:
        the ``covered`` argument of :class:`PhaseFixup` for this read's erase."""
        widths, ranges = _check_address(address_widths, live_ranges)
        (*outer, (lo, hi)) = ranges
        return (*outer, ((lo >> block_bits) << block_bits,
                         (((hi - 1) >> block_bits) + 1) << block_bits))

    @staticmethod
    def loaded_set(address_widths, live_ranges, block_bits) -> list[int]:
        """Every address of a visited block, in increasing order."""
        widths, _ = _check_address(address_widths, live_ranges)
        loaded = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
        return [_concatenate(values, widths)
                for values in itertools.product(*(range(lo, hi) for lo, hi in loaded))]

    @staticmethod
    def junk_layout(block_bits, low) -> list[int]:
        """The block position held by each junk slot, for low address bits ``low``."""
        return _swap_order(block_bits, low)[1:]

    @staticmethod
    def leaf_blocks(address_widths, live_ranges, block_bits) -> list[int]:
        """The high part of every unary leaf (its block), in the order the iteration visits the
        leaves: outermost register first, values increasing. Row ``l`` of ``leaf_words``."""
        widths, ranges = _check_address(address_widths, live_ranges)
        specs = _high_specs(widths, ranges, block_bits)
        return [_concatenate(values, [n for _, n, _, _ in specs])
                for values in itertools.product(*(range(lo, hi) for _, _, lo, hi in specs))]

    @staticmethod
    def leaf_words_shape(address_widths, live_ranges, width, block_bits) -> tuple[int, int, int]:
        """``(num_leaves, K, ceil(width / 32))``, the shape of ``leaf_words``."""
        return (len(QROAMRead.leaf_blocks(address_widths, live_ranges, block_bits)),
                2**block_bits, -(-width // WORD))

    @staticmethod
    def leaf_words_from_table(table, *, address_widths, live_ranges, width,
                              block_bits) -> np.ndarray:
        """The ``leaf_words`` of ``table`` (``(address, words)`` pairs, every address live, each
        entry below ``2**width`` in 32-bit words, least significant first): each entry in wire
        order at its leaf and position, zero at padding and dead positions. ``ValueError`` for an
        invalid table."""
        widths, ranges = _check_address(address_widths, live_ranges)
        entries = _entries(_check_table(table, widths, ranges, width))
        words = np.zeros(QROAMRead.leaf_words_shape(widths, ranges, width, block_bits),
                         dtype=WORD_DTYPE)
        for leaf, block in enumerate(QROAMRead.leaf_blocks(widths, ranges, block_bits)):
            for position in range(2**block_bits):
                stream = _wire_order(entries.get((block << block_bits) | position, 0), width)
                for w in range(words.shape[2]):
                    words[leaf, position, w] = (stream >> (WORD * w)) & (2**WORD - 1)
        return words

    @staticmethod
    def loaded_bits(table, address, *, address_widths, live_ranges, width, block_bits) -> int:
        """``target ‖ junk`` as one big-endian integer after an enabled read at ``address``;
        0 outside the loaded set."""
        return QROAMRead.loaded_contents(table, [address], address_widths=address_widths,
                                         live_ranges=live_ranges, width=width,
                                         block_bits=block_bits)[0]

    @staticmethod
    def loaded_contents(table, addresses, *, address_widths, live_ranges, width,
                        block_bits) -> list[int]:
        """:meth:`loaded_bits` of each address in ``addresses``, validating the table once."""
        widths, ranges = _check_address(address_widths, live_ranges)
        table = _check_table(table, widths, ranges, width)
        return _loaded_contents(_entries(table), widths, ranges, width, block_bits, addresses)


def _read_body(emit, leaf_words, address_widths, live_ranges, width, block_bits, control_value,
               address, target, junk, work, control):
    registers = [target] + [junk[r * width:(r + 1) * width] for r in range(2**block_bits - 1)]
    register = emit.wire_array(target, junk)
    leaves = itertools.count()

    def leaf(flag, values):
        emit.load(flag, leaf_words, next(leaves), register, 2**block_bits, width)

    iterate_nested(_registers(address, _high_specs(address_widths, live_ranges, block_bits)),
                   work, leaf, emit=emit, **_control(control, control_value))
    low = address[len(address) - block_bits:]
    for b in reversed(range(block_bits)):
        for r in range(2**b):
            for i in range(width):
                emit.cswap(low[block_bits - 1 - b], registers[r][i], registers[r + 2**b][i])


def _qroam_read_resources(leaf_words, address_widths, live_ranges, width, block_bits,
                          control_value, address, target, junk, work, control):
    counter = QROAMEmitter(counting=True)
    wires = _dummy(QROAMRead.required_wires(address_widths, live_ranges, width, block_bits,
                                            control_value))
    _read_body(counter, leaf_words, address_widths, live_ranges, width, block_bits,
               control_value, **wires)
    return as_resources(counter.counts)


@qp.register_resources(_qroam_read_resources, exact=False)
def _qroam_read(leaf_words, address_widths, live_ranges, width, block_bits, control_value,
                address, target, junk, work, control):
    _read_body(QROAMEmitter(), leaf_words, address_widths, live_ranges, width, block_bits,
               control_value, address, target, junk, work, control)


register_rule(QROAMRead, _qroam_read)


# --------------------------------------------------------------------------- PhaseFixup


def _parity_array(parities, size):
    """Canonicalize the dynamic ``parities`` once (L-23): a bool array under capture; in tape
    mode an object array of ``MeasurementValue``s, so both modes emit one conditional per covered
    address (K-10). Abstract arrays and tracers pass unchanged."""
    if isinstance(parities, AbstractArray) or qp.math.is_tracer(parities):
        if tuple(parities.shape) != (size,):
            raise ValueError(f"parities must have shape ({size},), got {tuple(parities.shape)}")
        return parities
    values = list(parities)
    if len(values) != size:
        raise ValueError(f"need one parity per covered address ({size}), got {len(values)}")
    if qp.capture.enabled():
        return jnp.stack([jnp.asarray(value, dtype=bool) for value in values])
    for i, value in enumerate(values):
        if not isinstance(value, MeasurementValue):
            raise ValueError(f"in tape mode every parity must be a MeasurementValue; entry {i} "
                             f"is {value!r}")
    return np.asarray(values, dtype=object)


class PhaseFixup(Operator2):
    r"""Multiply each address of the covered set by :math:`(-1)^{f(a)}`, with ``f`` a dynamic
    array; the standalone form of :func:`emit_phase_fixup`.

    **Action.** :math:`|a\rangle \mapsto (-1)^{f(a)}|a\rangle` for ``a`` in the covered set, and
    nothing elsewhere; no other phase (with an enable: only where the enable is satisfied). The
    covered set is the product of the ``covered`` ranges; for an erase it is the read's loaded
    set, :meth:`QROAMRead.loaded_ranges`.

    **Registers.**

    =========== ========================================= ======== ========================
    name        width                                     input    output
    =========== ========================================= ======== ========================
    ``address`` ``sum(address_widths)``                   any      phased
    ``onehot``  ``2**onehot_bits`` (0 if ``onehot_bits``  ``|0>``  ``|0>``
                is 0)
    ``work``    unary-iteration ancillas of the high part ``|0>``  ``|0>``
    ``control`` 0 wires, or 1 (the enable)                any      unchanged
    =========== ========================================= ======== ========================

    **Arguments.** Compilable: ``address_widths`` and ``covered`` (one ``(lo, hi)`` per register,
    the address model of :class:`QROAMRead`), ``onehot_bits`` (ob, ``0 <= ob <=
    sum(address_widths)``), ``control_value`` (``None``, 0 or 1). Dynamic: ``parities``, one
    boolean per covered address in the order of :meth:`address_order` (increasing addresses).
    Under capture it becomes a bool array; in tape mode every entry must be a
    ``MeasurementValue``, so that both modes emit a conditional for every covered address.

    **Pre- and postconditions.** ``onehot`` and ``work`` start and end in ``|0>``.

    **Decomposition** (spec 9.4). (1) The one-hot encoding of the low ob address bits: copy the
    top low bit and its negation into two flags, then split every flag ``f`` on each further bit
    ``a_j`` into ``f AND a_j`` (one ``TemporaryAND``) and ``f AND NOT a_j`` (one CNOT), so that
    ``onehot[j] = [low bits == j]``. (2) Exact unary iteration over the high part, visiting only
    high values with a covered address (:meth:`high_ranges`); at leaf ``h``, for each covered
    ``a = (h << ob) | j``, ``qp.cond(parities[a], CZ)(leaf flag, onehot[j])`` with
    ``qp.hint({"branch-prob": 0.5})``. (3) The measured uncompute of the tree. With
    ``onehot_bits = 0`` it is plain iteration with a conditional ``Z`` on every leaf flag.

    **Adjoint and control.** Never adjointed or controlled; use the enable.

    **Cost.** :func:`dfthc.costs.phase_fixup` with ``high_ranges(...)``: :math:`T = U` (or
    :math:`U_2`) over the covered high ranges, controlled if there is an enable, plus
    :math:`O(ob) = 2^{ob} - 2` (0 for ob = 0); ``M = 0``. At FeMoco-54: outer 22 + 14, inner
    70 + 126.

    **Paper and dfthcbe.** The phase fixup of the measured QROAM erase (Berry et al., Quantum 3,
    208, App. C), with ``CZ`` gates applied directly instead of a parity load into a
    :math:`|-\rangle` ancilla (spec 4.4, item 2). dfthcbe: the erase fixups of its unprepares.

    **Example.**

    .. code-block:: python

        import numpy as np
        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import PhaseFixup

        program = lambda: PhaseFixup(np.ones(6, dtype=bool), (3,), ((0, 6),), 1, None,
                                     address=[0, 1, 2], onehot=[3, 4], work=[5], control=[])
        compiled_resources(program, num_wires=6).toffoli  # U(0, 3, 2, 0) + O(1) = 2
    """

    dynamic_argnames = ("parities",)
    compilable_argnames = ("address_widths", "covered", "onehot_bits", "control_value")
    wire_argnames = ("address", "onehot", "work", "control")

    def __init__(self, parities, address_widths, covered, onehot_bits, control_value, address,
                 onehot, work, control):
        address_widths, covered = _check_address(address_widths, covered)
        onehot_bits = check_int("onehot_bits", onehot_bits, low=0, high=sum(address_widths))
        control_value = _check_control_value(control_value)
        parities = _parity_array(parities, len(self.address_order(address_widths, covered)))
        check_wires("PhaseFixup",
                    {"address": address, "onehot": onehot, "work": work, "control": control},
                    self.required_wires(address_widths, covered, onehot_bits, control_value))
        super().__init__(parities, address_widths, covered, onehot_bits, control_value, address,
                         onehot, work, control)

    @staticmethod
    def required_wires(address_widths, covered, onehot_bits, control_value) -> dict[str, int]:
        """Exact register sizes."""
        ranges = PhaseFixup.high_ranges(address_widths, covered, onehot_bits)
        return {"address": sum(address_widths), "onehot": 2**onehot_bits if onehot_bits else 0,
                "work": work_required([n for _, _, n in ranges], control_value is not None),
                "control": int(control_value is not None)}

    @staticmethod
    def address_order(address_widths, covered) -> list[int]:
        """The covered addresses in the canonical order of ``parities``: increasing."""
        widths, covered = _check_address(address_widths, covered)
        return [_concatenate(values, widths)
                for values in itertools.product(*(range(lo, hi) for lo, hi in covered))]

    @staticmethod
    def high_ranges(address_widths, covered, onehot_bits) -> list[tuple[int, int, int]]:
        """``(lo, hi, n)`` of each iterated high register: the ``high_ranges`` of
        :func:`dfthc.costs.phase_fixup`."""
        widths, covered = _check_address(address_widths, covered)
        return [(lo, hi, n) for _, n, lo, hi in _high_specs(widths, covered, onehot_bits)]


def _onehot_compute(emit, low, onehot):
    size = 2 ** len(low)
    emit.cnot(low[0], onehot[size // 2])
    emit.x(onehot[0])
    emit.cnot(low[0], onehot[0])
    for t in range(1, len(low)):
        step = size >> t
        for s in range(0, size, step):
            emit.and_(onehot[s], low[t], onehot[s + step // 2])
            emit.cnot(onehot[s + step // 2], onehot[s])


def _onehot_uncompute(emit, low, onehot):
    size = 2 ** len(low)
    for t in reversed(range(1, len(low))):
        step = size >> t
        for s in reversed(range(0, size, step)):
            emit.cnot(onehot[s + step // 2], onehot[s])
            emit.unand(onehot[s], low[t], onehot[s + step // 2])
    emit.cnot(low[0], onehot[0])
    emit.x(onehot[0])
    emit.cnot(low[0], onehot[size // 2])


def emit_phase_fixup(parities, *, address, address_widths, covered, onehot, onehot_bits, work,
                     control=None, control_value=1, emit=None) -> None:
    r"""Emit the gates of :class:`PhaseFixup` inside a rule (spec 9.4).

    ``parities`` holds one entry per covered address in :meth:`PhaseFixup.address_order`: a
    predicate for ``qp.cond``, or the Python ``False``, which emits no conditional. Callers decide
    the ``False`` entries from compilable data only (the erase: its zero masks), so both modes
    emit the same gates. ``control`` is the enable wire or ``None``; ``onehot`` has exactly
    ``2**onehot_bits`` wires (none for 0) in ``|0>``; ``work`` has
    ``PhaseFixup.required_wires(...)["work"]`` wires in ``|0>``. ``emit`` is a
    :class:`QROAMEmitter` (default: emitting).
    """
    emit = QROAMEmitter() if emit is None else emit
    widths, covered = _check_address(address_widths, covered)
    if not 0 <= onehot_bits <= sum(widths):
        raise ValueError(f"onehot_bits must be in [0, {sum(widths)}], got {onehot_bits}")
    order = PhaseFixup.address_order(widths, covered)
    if len(parities) != len(order):
        raise ValueError(f"need one parity per covered address ({len(order)}), "
                         f"got {len(parities)}")
    if len(onehot) != (2**onehot_bits if onehot_bits else 0):
        raise ValueError(f"onehot needs {2**onehot_bits if onehot_bits else 0} wires, "
                         f"got {len(onehot)}")
    index = {a: i for i, a in enumerate(order)}
    low = list(address[len(address) - onehot_bits:])
    onehot = list(onehot)
    specs = _high_specs(widths, covered, onehot_bits)
    high_widths = [n for _, n, _, _ in specs]

    def leaf(flag, values):
        block = _concatenate(values, high_widths)
        for j in range(2**onehot_bits):
            i = index.get((block << onehot_bits) | j)
            if i is None or parities[i] is False:
                continue
            if onehot_bits == 0:
                emit.cond_z(parities[i], flag)
            elif flag is None:
                emit.cond_z(parities[i], onehot[j])
            else:
                emit.cond_cz(parities[i], flag, onehot[j])

    if onehot_bits:
        _onehot_compute(emit, low, onehot)
    iterate_nested(_registers(address, specs), work, leaf, control, control_value, emit=emit)
    if onehot_bits:
        _onehot_uncompute(emit, low, onehot)


def _phase_fixup_body(emit, parities, address_widths, covered, onehot_bits, control_value,
                      address, onehot, work, control):
    emit_phase_fixup(parities, address=address, address_widths=address_widths, covered=covered,
                     onehot=onehot, onehot_bits=onehot_bits, work=work, emit=emit,
                     **_control(control, control_value))


def _phase_fixup_resources(parities, address_widths, covered, onehot_bits, control_value,
                           address, onehot, work, control):
    counter = QROAMEmitter(counting=True)
    size = len(PhaseFixup.address_order(address_widths, covered))
    wires = _dummy(PhaseFixup.required_wires(address_widths, covered, onehot_bits, control_value))
    _phase_fixup_body(counter, [True] * size, address_widths, covered, onehot_bits, control_value,
                      **wires)
    return as_resources(counter.counts)


@qp.register_resources(_phase_fixup_resources)
def _phase_fixup(parities, address_widths, covered, onehot_bits, control_value, address, onehot,
                 work, control):
    size = len(PhaseFixup.address_order(address_widths, covered))
    _phase_fixup_body(QROAMEmitter(), [parities[i] for i in range(size)], address_widths,
                      covered, onehot_bits, control_value, address, onehot, work, control)


register_rule(PhaseFixup, _phase_fixup)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_phase_fixup.py -q`
Expected: `27 passed` (27 s in the serial replay, including the two `MID_NAPHTHALENE` compiles).

Run: `scripts/test.sh tests/ops/test_qroam_read.py -m "not slow" -q`
Expected: `44 passed`.

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `579 passed` (554 + 25).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/qroam.py tests/ops/test_phase_fixup.py
git commit -m "Add PhaseFixup and emit_phase_fixup"
```

---

### Task 1C.5: `QROAMErase` and round-trip certificates

**Task id:** `1C.5` · **Branch:** `phase1/1C.5` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 1C.4 (and through it 1C.3), 1B.2

**Files:**
- Modify: `src/dfthc/ops/qroam.py` (replace it with the version below: the 1C.3 and 1C.4 code is unchanged; the docstring, the imports and `QROAMEmitter` grow, and the erase section is new)
- Create: `tests/ops/test_qroam_erase.py`, `tests/ops/test_certificates_qroam.py` (marked `slow`)

**Interfaces:**
- Consumes: Task 1C.4's `qroam.py`; `dfthc.compile.mcm_parities`, `pack_masks`, `packed_parities` (Task 0.2; K-3 passed, so they stay mode-agnostic); `dfthc.costs.qroam_erase`, `qroam_read`, `outer_unprepare`, `inner_unprepare`, `flagged_uniform_unprep` (1A.7 with its revision 1); `dfthc.testing.resources.verbose_program`, `chosen_rules`, `classify_rule_warning` (0.2, 0.2b); `SparseSimulator`, `ForcedPolicy`, `SparseState`, `nonlinear_conditionals`, `decompose_for_simulation` (1B.2).
- Produces, in `dfthc.ops.qroam`:
  - `QROAMEmitter.measure_x(wire)` (`H`, then `qp.measure(wire, reset=True)`) and `parities(outcomes, masks)` (`mcm_parities`; counting: `False` exactly for zero masks);
  - `QROAMErase(table, address_widths, live_ranges, width, block_bits, control_value, onehot_bits, address, target, junk, work, control, extra_onehot)`: the read's compilable arguments plus `0 <= onehot_bits <= sum(address_widths)`;
  - static `required_wires(...) -> dict` (`work`: the fixup's; `extra_onehot`: `max(2**ob - 2**k width, 0)`) and `fixup_high_ranges(address_widths, live_ranges, block_bits, onehot_bits) -> list[tuple[int, int, int]]`;
  - the rule `_qroam_erase`. It is never adjointed or controlled.

**Acceptance tests** (19 tests in `tests/ops/test_qroam_erase.py`, 2 of them `slow`; 7 `slow` tests in `tests/ops/test_certificates_qroam.py`):
- T4: the FeMoco ranges and closed forms `Cost(36, 96)` and `Cost(196, 416)` and `required_wires` at FeMoco (work 4 and 6, no extra one-hot); compiled T and M equal the closed forms for the outer and inner erases of both test presets, with M = K width and A = T (4 tests); an enabled erase whose one-hot needs 4 extra wires. `test_counts_at_mid_naphthalene` (`slow`, 2 tests): T, M and A exact at the outer erase of `MID_NAPHTHALENE` (k = 1, ob = 3, width 21, M = 42) and its inner erase (k = 2, ob = 4, width 23, M = 92). Each compiled alone under `run_watched`: T = A = 12, M = 42 in 2.2 s at a peak of 276 MB (outer), T = A = 34, M = 92 in 3.6 s at 317 MB (inner); log `.scratch/plan-check/stages/1c-mid-compiles.log`.
- `test_fixup_high_ranges_agree_with_the_cost_model`: `costs.outer_unprepare` and `inner_unprepare` equal `w + qroam_erase(fixup_high_ranges(...)) + F^dag` at the four presets and at 128 shapes with `X_o` from 5 to 12 and every `k1`, `ke1` up to 3. `fixup_high_ranges` is the oracle (change 5): against Task 1A.7's code before its revision 1 the test fails at `X_o = 5`, `k1 = 2`, `ke1 = 0` (T 16 against 17; log `.scratch/plan-check/stages/1A-rev1/xcheck_against_old_costs.log`).
- T6: the erase's T differs from a re-read's; its conditional CZs count `ceil(nonzero masks / 2)`; the verbose listing names `_qroam_erase`. `qp.adjoint` and `qp.ctrl` of the erase fail with `CompileError` "Decomposition rule not found", after Catalyst's "expected" measuring-rule warning (2 tests).
- Lightning smoke runs: read then erase (k = 2, ob = 2) restore a phased address state for seeds 0 to 4; the read's `leaf_words` come from the erase's table (`QROAMRead.leaf_words_from_table`).
- `mcm_parities` linearity: the capture core at the FeMoco-54 inner-erase shape (8,640 masks from a random FeMoco-shape inner table, 416 outcomes, 100 random pairs; classical, nothing compiled) and the tape implementation at the `H2_631G_TEST` inner erase (zero masks give `False`).
- T5; T2 in tape mode at two sizes (one enabled); `required_wires` and two constructor violations.
- Certificates (spec 12.5; `slow`, 7 tests, 3 to 7 s together): for the outer and inner erases of both test presets, an outer shape with k = 2 and `X_o = 5`, an inner shape with (B+1, k2) = (6, 2), and that shape enabled with control value 0, run read then erase on the sparse simulator in strict mode under the zero schedule and every unit schedule (M + 1 runs; M = 8 to 20). Each test first asserts, on the decomposed tape it certifies, that the tape has conditionals and that `nonlinear_conditionals(operations)` is empty: every fix-up conditional is a `PauliZ` or `CZ` with a GF(2)-linear predicate (spec 12.5, premise check). Each run starts from a seeded random complex amplitude on every value of the address and control wires (`random_input_state`), must consume exactly M forced outcomes and must return exactly that state with every ancilla in `|0>`. Mutation check (`.scratch/plan-check/stages/mutate_erase_1c.py`, mutant sources in `stages/mutants/`; logs `stages/1c-erase-mutants.log` for revision 1's phased input and `stages/g0-2/1c-erase-mutants-rev3.log` for the random input, with the same failing cases, each naming every failing case): outcomes passed in wire order instead of reversed fail 7 of 7 shapes; the fixup over the live instead of the loaded set fails the 5 shapes with dead positions (`sto3g-outer` and `631g-outer`, which have none, pass); the first conditional dropped fails 7 of 7.

**Definition of done:** `19 passed` targeted (17 in the per-commit tier), and the certificates `7 passed` run targeted; the 1C.3 and 1C.4 tests still pass; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_qroam_erase.py`:

```python
"""QROAMErase (spec 9.5): counts, rule selection, lightning smoke runs, mcm_parities linearity."""

import dataclasses
import math
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc import costs
from dfthc.compile import (
    FIXED_DECOMPS,
    GATE_SET,
    compiled_resources,
    mcm_parities,
    pack_masks,
    packed_parities,
)
from dfthc.ops.qroam import QROAMErase, QROAMRead, _qroam_erase
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program


def random_table(widths, live_ranges, width, seed=11):
    """An entry of ``width`` random bits at every live address (one or two registers)."""
    rng = np.random.default_rng(seed)
    (lo1, hi1), *rest = live_ranges
    addresses = ([(v << widths[1]) | u for v in range(lo1, hi1) for u in range(*rest[0])]
                 if rest else list(range(lo1, hi1)))
    table = []
    for address in addresses:
        value = int(rng.integers(0, 2**width))
        table.append((address, tuple((value >> (32 * w)) & (2**32 - 1)
                                     for w in range(-(-width // 32)))))
    return tuple(table)


def erase_shapes(p):
    """(widths, live ranges, width, block bits, one-hot bits) of the outer and inner erases."""
    return [((p.w_x,), ((0, p.X_o),), p.d1, p.k1, p.ke1),
            ((p.w_x, p.w_b), ((0, p.RC), (0, p.B + 1)), p.d2, p.k2, p.ke2 + p.w_b)]


def layout(shape, control_value=None):
    widths, live, width, k, ob = shape
    erase = QROAMErase.required_wires((), widths, live, width, k, control_value, ob)
    read_work = QROAMRead.required_wires(widths, live, width, k, control_value)["work"]
    sizes = dict(erase, work=max(erase["work"], read_work))
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def make_read(table, shape, control_value, wires):
    widths, live, width, k, _ = shape
    work = wires["work"][:QROAMRead.required_wires(widths, live, width, k,
                                                   control_value)["work"]]
    words = QROAMRead.leaf_words_from_table(table, address_widths=widths, live_ranges=live,
                                            width=width, block_bits=k)
    return QROAMRead(words, widths, live, width, k, control_value, wires["address"],
                     wires["target"], wires["junk"], work, wires["control"])


def make_erase(table, shape, control_value, wires):
    widths, live, width, k, ob = shape
    work = wires["work"][:QROAMErase.required_wires((), widths, live, width, k, control_value,
                                                    ob)["work"]]
    return QROAMErase(table, widths, live, width, k, control_value, ob, wires["address"],
                      wires["target"], wires["junk"], work, wires["control"],
                      wires["extra_onehot"])


def expected_cost(shape, control_value):
    widths, live, width, k, ob = shape
    return costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width, ob,
                             control_value is not None)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 on the closed forms: outer T = 22 + 14, M = 96; inner T = 70 + 126, M = 416."""
    outer, inner = erase_shapes(FEMOCO54)
    assert QROAMErase.fixup_high_ranges(*outer[:2], outer[3], outer[4]) == [(0, 21, 5)]
    assert QROAMErase.fixup_high_ranges(*inner[:2], inner[3], inner[4]) == [(0, 68, 7)]
    assert expected_cost(outer, None) == costs.Cost(36, 96)
    assert expected_cost(inner, None) == costs.Cost(196, 416)
    assert QROAMErase.required_wires((), *outer[:3], outer[3], None, outer[4]) == {
        "address": 9, "target": 24, "junk": 72, "work": 4, "control": 0, "extra_onehot": 0}
    assert QROAMErase.required_wires((), *inner[:3], inner[3], None, inner[4]) == {
        "address": 14, "target": 26, "junk": 390, "work": 6, "control": 0, "extra_onehot": 0}


SMALL = erase_shapes(H2_STO3G_TEST) + erase_shapes(H2_631G_TEST)
SMALL_IDS = ["sto3g-outer", "sto3g-inner", "631g-outer", "631g-inner"]


@pytest.mark.compiled
@pytest.mark.parametrize("shape", SMALL, ids=SMALL_IDS)
def test_counts_match_the_closed_form(shape):
    """T4 and the T6 signature: M = K width, T = the erase closed form, A = T (every AND of the
    fixup is uncomputed by measurement, and the erase has no plain Toffoli)."""
    _assert_compiled_counts(shape)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("shape", erase_shapes(MID_NAPHTHALENE), ids=["outer", "inner"])
def test_counts_at_mid_naphthalene(shape):
    """T4 at MID_NAPHTHALENE (nightly tier); compiled_resources rejects symbolic counts."""
    _assert_compiled_counts(shape)


def _assert_compiled_counts(shape):
    widths, live, width, k, ob = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(shape)
    resources = compiled_resources(lambda: make_erase(table, shape, None, wires),
                                   num_wires=num_wires)
    expected = expected_cost(shape, None)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measurements == 2**k * width
    assert resources.measured_and_uncomputes == resources.toffoli


def test_fixup_high_ranges_agree_with_the_cost_model():
    """QROAMErase.fixup_high_ranges is the oracle of dfthc.costs' erase ranges: the outer and inner
    unprepare costs equal w + qroam_erase(fixup_high_ranges) + F^dag, also where ke1 < k1 and K1
    does not divide X_o, so the outer erase covers dead positions beyond X_o."""
    shapes = [FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE] + [
        dataclasses.replace(H2_STO3G_TEST, N=n, n_plus=1, R=1, C=2, k1=k1, ke1=ke1)
        for n in range(3, 11) for k1 in range(4) for ke1 in range(4)]
    for p in shapes:
        for shape, unprepare, w, L in zip(erase_shapes(p),
                                          (costs.outer_unprepare, costs.inner_unprepare),
                                          (p.w_x, p.w_b), (p.X_o, p.B + 1), strict=True):
            widths, live, width, k, ob = shape
            erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width,
                                      ob, False)
            expected = costs.Cost(w) + erase + costs.flagged_uniform_unprep(L, p.s_u)
            assert unprepare(p) == expected, (p.X_o, p.k1, p.ke1)


@pytest.mark.compiled
def test_erase_is_not_a_second_read_and_its_conditionals_are_hinted(capfd):
    """T6: the chosen rule is the pinned measured erase, its T differs from a re-read, and its
    conditional CZs count their hinted expectation (one per nonzero mask)."""
    shape = SMALL[3]
    widths, live, width, k, ob = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(shape)
    resources = compiled_resources(lambda: make_erase(table, shape, None, wires),
                                   num_wires=num_wires)
    read = costs.qroam_read(QROAMRead.visited_ranges(widths, live, k), k, width, False)
    assert resources.toffoli != read.toffoli
    masks = QROAMRead.loaded_contents(table, QROAMRead.loaded_set(widths, live, k),
                                      address_widths=widths, live_ranges=live, width=width,
                                      block_bits=k)
    assert resources.counts["CZ"] == math.ceil(sum(1 for m in masks if m) / 2)
    qp.analyze(verbose_program(lambda: make_erase(table, shape, None, wires), num_wires),
               level="user")()
    captured = capfd.readouterr()
    assert "_qroam_erase" in chosen_rules(captured.out + captured.err)


@pytest.mark.compiled
def test_counts_with_enable_and_extra_onehot():
    """An enabled erase (c = 1 in U) whose one-hot needs 2**3 - 4 extra wires."""
    shape = ((3, 2), ((0, 2), (0, 3)), 1, 2, 3)
    widths, live, width, k, ob = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(shape, control_value=1)
    assert len(wires["extra_onehot"]) == 4
    resources = compiled_resources(lambda: make_erase(table, shape, 1, wires),
                                   num_wires=num_wires)
    expected = expected_cost(shape, 1)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 4)


@pytest.mark.compiled
@pytest.mark.parametrize("wrap", [qp.adjoint, lambda op: qp.ctrl(op, control=20)],
                         ids=["adjoint", "control"])
def test_erase_has_no_adjoint_or_controlled_path(wrap):
    """Spec 7.4 rule 5: never adjointed or controlled. Catalyst drops the measuring candidate
    with the expected warning (rule 12), and without another rule the compile fails loudly."""
    shape = SMALL[0]
    widths, live, width, _, _ = shape
    table = random_table(widths, live, width)
    wires, _ = layout(shape)
    key = "Adjoint(QROAMErase)" if wrap is qp.adjoint else "C(QROAMErase)"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(lambda: wrap(make_erase(table, shape, None, wires)), num_wires=21)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the _qroam_erase")]
    assert skipped and {classify_rule_warning(m) for m in skipped} == {"expected"}


def _phased_prepare(wires):
    """H on each wire, S on even positions, CZ on consecutive pairs (non-uniform phases)."""
    for wire in wires:
        qp.H(wire)
    for position, wire in enumerate(wires):
        if position % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


@pytest.mark.compiled
def test_lightning_smoke_runs_with_five_seeds():
    """Compiled read then erase (k = 2, one-hot bits 2) restores a phased address state and
    clears every ancilla for five measurement seeds."""
    shape = ((3,), ((0, 5),), 2, 2, 2)
    widths, live, width, _, _ = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(shape)
    address = wires["address"]

    def prepared():
        _phased_prepare(address)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def reference():
        prepared()
        return qp.state()

    expected = np.asarray(reference())
    for seed in range(5):
        @qp.qjit(capture=True, seed=seed)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def round_trip():
            prepared()
            make_read(table, shape, None, wires)
            make_erase(table, shape, None, wires)
            return qp.state()

        assert np.allclose(np.asarray(round_trip()), expected, rtol=0, atol=1e-10), seed


def _inner_masks(p, seed=3):
    """The inner erase's masks over its loaded set, for a random table of preset p's shape."""
    widths, live, width, k, _ = erase_shapes(p)[1]
    table = random_table(widths, live, width, seed)
    masks = QROAMRead.loaded_contents(table, QROAMRead.loaded_set(widths, live, k),
                                      address_widths=widths, live_ranges=live, width=width,
                                      block_bits=k)
    return masks, 2**k * width


def test_mcm_parities_capture_core_is_linear_at_the_femoco_inner_erase():
    """Spec 12.5 linearity at the FeMoco-54 inner-erase shape (8,640 masks over 416 outcomes;
    a classical computation, nothing is compiled): f_0 = 0 and f_(x ^ y) = f_x ^ f_y."""
    masks, num_outcomes = _inner_masks(FEMOCO54)
    assert (len(masks), num_outcomes) == (270 * 32, 416)
    packed = pack_masks(masks, num_outcomes)
    rng = np.random.default_rng(12)
    assert not np.asarray(packed_parities(jnp.zeros(num_outcomes, dtype=jnp.uint32), packed)).any()
    for _ in range(100):
        x, y = rng.integers(0, 2, size=(2, num_outcomes))
        fx, fy = (np.asarray(packed_parities(v, packed)) for v in (x, y))
        assert np.array_equal(np.asarray(packed_parities(x ^ y, packed)), fx ^ fy)


def test_mcm_parities_tape_mode_is_linear_at_a_test_inner_erase():
    """The tape implementation (flat MeasurementValues) at the H2_631G_TEST inner erase."""
    masks, num_outcomes = _inner_masks(H2_631G_TEST)
    with qp.queuing.AnnotatedQueue():
        outcomes = [qp.measure(w) for w in range(num_outcomes)]
        parities = mcm_parities(outcomes, masks)

    def evaluate(bits):
        record = {outcomes[i].measurements[0]: int(bits[i]) for i in range(num_outcomes)}
        return np.array([0 if p is False else int(p.concretize(record)) for p in parities])

    assert all(p is False for p, m in zip(parities, masks, strict=True) if m == 0)
    rng = np.random.default_rng(13)
    assert not evaluate(np.zeros(num_outcomes, dtype=int)).any()
    for _ in range(100):
        x, y = rng.integers(0, 2, size=(2, num_outcomes))
        assert np.array_equal(evaluate(x ^ y), evaluate(x) ^ evaluate(y))


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    shape = SMALL[1]
    widths, live, width, _, _ = shape
    table = random_table(widths, live, width)
    wires, num_wires = layout(shape)
    compiled_resources(lambda: make_erase(table, shape, None, wires), num_wires=num_wires,
                       level="device")


@pytest.mark.parametrize(("shape", "control_value"),
                         [(SMALL[0], None), (((3, 3), ((0, 2), (0, 6)), 3, 2, 3), 1)])
def test_rule_is_valid_in_tape_mode(shape, control_value):
    """T2 in tape mode: the measured outcomes and their parities are MeasurementValues."""
    widths, live, width, _, _ = shape
    table = random_table(widths, live, width)
    wires, _ = layout(shape, control_value)
    qp.testing.assert_valid_decomp_rule(make_erase(table, shape, control_value, wires),
                                        _qroam_erase, check_matrix=False)


def test_required_wires_and_validation():
    assert QROAMErase.required_wires((), (3,), ((0, 5),), 1, 0, None, 2) == {
        "address": 3, "target": 1, "junk": 0, "work": 0, "control": 0, "extra_onehot": 3}
    args = {"table": (), "address_widths": (3,), "live_ranges": ((0, 5),), "width": 1,
            "block_bits": 0, "control_value": None, "onehot_bits": 2, "address": [0, 1, 2],
            "target": [3], "junk": [], "work": [], "control": [], "extra_onehot": [4, 5, 6]}
    QROAMErase(**args)
    with pytest.raises(ValueError, match=r"onehot_bits must lie in \[0, 3\]"):
        QROAMErase(**(args | {"onehot_bits": 4}))
    with pytest.raises(ValueError, match="'extra_onehot' has 2 wires; expected 3"):
        QROAMErase(**(args | {"extra_onehot": [4, 5]}))
```

Create `tests/ops/test_certificates_qroam.py`:

```python
"""Outcome certificates of spec 12.5 for QROAMRead -> QROAMErase round trips (nightly tier).

The phase an erase leaves on address ``a`` is ``(-1)^(xi . g(a))`` and its fixup applies
``(-1)^f_xi(a)``, both GF(2)-linear in the outcome vector ``xi`` (test_qroam_erase checks
``mcm_parities``). So the residual is linear, and exact restoration under the zero vector and
under every unit vector, M + 1 forced runs, certifies all 2**M outcomes. That conclusion also
needs every conditional of the certified tape to be diagonal with a GF(2)-linear predicate,
which ``nonlinear_conditionals`` checks on the tape itself. Each run starts from a random
complex amplitude on every value of the address and control wires (so any permutation or
relative phase of the addresses is caught) and must end in exactly that state with every
ancilla in |0>; strict mode also asserts that every erase measurement has probability 1/2.
"""

import numpy as np
import pytest
from pennylane.ops import Conditional

from dfthc.ops.qroam import QROAMErase, QROAMRead
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST
from dfthc.testing.sparse_sim import (
    ForcedPolicy,
    SparseSimulator,
    SparseState,
    nonlinear_conditionals,
)
from dfthc.testing.tape import decompose_for_simulation

pytestmark = pytest.mark.slow


def random_table(widths, live_ranges, width, seed=5):
    rng = np.random.default_rng(seed)
    (lo1, hi1), *rest = live_ranges
    addresses = ([(v << widths[1]) | u for v in range(lo1, hi1) for u in range(*rest[0])]
                 if rest else list(range(lo1, hi1)))
    return tuple((a, (int(rng.integers(0, 2**width)),)) for a in addresses)


def erase_shapes(p):
    return [((p.w_x,), ((0, p.X_o),), p.d1, p.k1, p.ke1),
            ((p.w_x, p.w_b), ((0, p.RC), (0, p.B + 1)), p.d2, p.k2, p.ke2 + p.w_b)]


# The outer and inner shapes of both test presets (k = 1), and two k >= 2 shapes with dead
# positions (X_o = 5 with k = 2; rows [0, 2) of 3 bits with (B + 1, k2) = (6, 2)), one enabled.
SHAPES = erase_shapes(H2_STO3G_TEST) + erase_shapes(H2_631G_TEST) + [
    ((3,), ((0, 5),), 3, 2, 2), ((3, 3), ((0, 2), (0, 6)), 5, 2, 4)]
CASES = [(shape, None) for shape in SHAPES] + [(SHAPES[-1], 0)]
IDS = ["sto3g-outer", "sto3g-inner", "631g-outer", "631g-inner", "outer-k2", "inner-6-2",
       "inner-6-2-enable0"]


def layout(shape, control_value):
    widths, live, width, k, ob = shape
    erase = QROAMErase.required_wires((), widths, live, width, k, control_value, ob)
    read = QROAMRead.required_wires(widths, live, width, k, control_value)
    sizes = dict(erase, work=max(erase["work"], read["work"]))
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start, read["work"], erase["work"]


def random_input_state(num_wires, inputs, seed=11):
    """A random complex amplitude on every basis value of ``inputs``, every other wire |0>."""
    rng = np.random.default_rng(seed)
    size = 2 ** len(inputs)
    amps = rng.normal(size=size) + 1j * rng.normal(size=size)
    keys = []
    for v in range(size):
        value = sum(((v >> (len(inputs) - 1 - i)) & 1) << (num_wires - 1 - wire)
                    for i, wire in enumerate(inputs))
        keys.append(SparseState.basis(num_wires, value).keys[0])
    return SparseState(num_wires, np.array(keys), amps / np.linalg.norm(amps))


def sorted_terms(state):
    order = np.lexsort(state.keys.T[::-1])
    return state.keys[order], state.amps[order]


@pytest.mark.parametrize(("shape", "control_value"), CASES, ids=IDS)
def test_read_erase_round_trip_certificate(shape, control_value):
    widths, live, width, k, ob = shape
    table = random_table(widths, live, width)
    wires, num_wires, read_work, erase_work = layout(shape, control_value)
    words = QROAMRead.leaf_words_from_table(table, address_widths=widths, live_ranges=live,
                                            width=width, block_bits=k)

    def round_trip():
        QROAMRead(words, widths, live, width, k, control_value, wires["address"],
                  wires["target"], wires["junk"], wires["work"][:read_work], wires["control"])
        QROAMErase(table, widths, live, width, k, control_value, ob, wires["address"],
                   wires["target"], wires["junk"], wires["work"][:erase_work], wires["control"],
                   wires["extra_onehot"])

    operations = decompose_for_simulation(round_trip).operations
    assert any(isinstance(op, Conditional) for op in operations)
    assert not nonlinear_conditionals(operations)
    initial = random_input_state(num_wires, wires["address"] + wires["control"])
    expected_keys, expected_amps = sorted_terms(initial)
    num_measured = 2**k * width
    schedules = [[0] * num_measured] + [
        [int(i == j) for j in range(num_measured)] for i in range(num_measured)]
    for schedule in schedules:
        simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
        keys, amps = sorted_terms(simulator.run(operations, initial_state=initial))
        assert simulator.policy.position == num_measured
        assert np.array_equal(keys, expected_keys), schedule
        assert np.allclose(amps, expected_amps, rtol=0, atol=1e-12), schedule
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_qroam_erase.py tests/ops/test_certificates_qroam.py -q`
Expected: FAIL at collection, `2 errors`: `ImportError: cannot import name 'QROAMErase' from 'dfthc.ops.qroam'`.

- [ ] **Step 3: Add the erase to `qroam.py`**

Replace `src/dfthc/ops/qroam.py` with:

```python
"""The paper's QROAM read, the erase phase fixup and the measured erase (spec 9.3 to 9.5).

**Address model** (spec 9.3). The address is one or two big-endian registers (spec 6.7),
``address_widths = (w_1,)`` or ``(w_1, w_2)``, concatenated with the first register most
significant: the integer address is ``(a_1 << w_2) | a_2``. Each register has a range
``(lo, hi)``. A table maps integer addresses to entries of ``width`` bits, stored as tuples of
32-bit words, least significant word first (L-24).

**Bit order.** Every register holds an entry big-endian: wire ``i`` of a ``width``-wire register
holds bit ``width - 1 - i``. ``target ‖ junk`` is read as one big-endian integer of
``2**block_bits * width`` bits (``target`` most significant, then the junk slots in order); that
integer is :meth:`QROAMRead.loaded_bits`. The read's dynamic ``leaf_words`` hold the entries in
wire order instead: bit ``s`` of word ``w`` of block position ``p`` is the bit for wire
``32 w + s`` of that position's register, which is entry bit ``width - 1 - (32 w + s)``
(:meth:`QROAMRead.leaf_words_from_table` converts a table).
Bit ``i`` of ``loaded_bits`` is the wire ``M - 1 - i`` of ``target ‖ junk``, so the erase passes
its outcomes to :func:`dfthc.compile.mcm_parities` in reverse wire order.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.ops import MeasurementValue
from pennylane.typing import AbstractArray

from dfthc.compile import mcm_parities, register_rule
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.unary import Emitter, as_resources, iterate_nested, work_required

WORD = 32
"""Data bits per table word (spec 8.4, L-24)."""

WORD_DTYPE = np.int64
"""dtype of ``leaf_words``: signed, because Catalyst builds no rule for an operator with an
unsigned dynamic argument (L-26); 32 data bits per word keep every word non-negative."""


class QROAMEmitter(Emitter):
    """:class:`~dfthc.ops.unary.Emitter` with the gates of the read, the fixup and the erase."""

    def cswap(self, control, a, b):
        self.gate("CSWAP", lambda: qp.CSWAP([control, a, b]))

    def wire_array(self, target, junk):
        """``target ‖ junk`` as one register that a traced loop index can address: a JAX array
        under capture, a list in tape mode; ``None`` when counting."""
        if self.counts is not None:
            return None
        if not qp.capture.enabled():
            return list(target) + list(junk)
        if len(junk) == 0:
            return jnp.asarray(target)
        return jnp.concatenate([jnp.asarray(target), jnp.asarray(junk)])

    def load(self, flag, leaf_words, leaf, register, positions, width):
        """The loads of unary leaf ``leaf``: one ``qp.for_loop`` over its ``positions * width``
        bits. Iteration ``t`` takes position ``p = t // width`` and bit ``i = t % width`` and
        applies, conditioned on that bit of ``leaf_words``, a CNOT from ``flag`` (an ``X`` for a
        ``None`` flag) onto ``register[p * width + i]``. No conditional surrounds the loop.
        Counting: ``positions * width`` gates, the upper bound."""
        name = "PauliX" if flag is None else "CNOT"
        if self.counts is not None:
            self.counts[name] += positions * width
            return

        @qp.for_loop(0, positions * width, 1)
        def bit_loop(t):
            p, i = t // width, t % width
            bit = (leaf_words[leaf, p, i // WORD] >> (i % WORD)) & 1
            if flag is None:
                qp.cond(bit == 1, qp.X)(register[p * width + i])
            else:
                qp.cond(bit == 1, qp.CNOT)([flag, register[p * width + i]])

        bit_loop()

    def cond_cz(self, parity, a, b):
        def branch():
            qp.CZ([a, b])

        self.gate("CZ", lambda: qp.cond(parity, qp.hint({"branch-prob": 0.5})(branch))())

    def cond_z(self, parity, wire):
        def branch():
            qp.Z(wire)

        self.gate("PauliZ", lambda: qp.cond(parity, qp.hint({"branch-prob": 0.5})(branch))())

    def measure_x(self, wire):
        """``H`` then ``qp.measure(wire, reset=True)``; returns the outcome (``None`` counting)."""
        self.gate("Hadamard", lambda: qp.H(wire))
        return self.gate("MidMeasure", lambda: qp.measure(wire, reset=True))

    def parities(self, outcomes, masks):
        """:func:`dfthc.compile.mcm_parities`; counting, ``False`` exactly for the zero masks."""
        if self.counts is None:
            return mcm_parities(outcomes, masks)
        return [False if mask == 0 else True for mask in masks]


# --------------------------------------------------------------------------- address helpers


def _check_address(widths, ranges) -> tuple[tuple, tuple]:
    """Canonical ``(address_widths, ranges)``; ``ValueError`` outside the address model."""
    widths = canonical_ints("address_widths", widths)
    ranges = canonical_ints("ranges", ranges)
    if not isinstance(widths, tuple) or len(widths) not in (1, 2):
        raise ValueError(f"an address has one or two registers, got address_widths={widths!r}")
    if any(n < 0 for n in widths) or sum(widths) < 1 or (len(widths) == 2 and widths[0] < 1):
        raise ValueError("address registers need nonnegative widths, at least one wire, and a "
                         f"nonempty first register of two; got {widths}")
    if not isinstance(ranges, tuple) or len(ranges) != len(widths):
        raise ValueError(f"need one (lo, hi) per address register, got {ranges!r}")
    for n, pair in zip(widths, ranges, strict=True):
        if not isinstance(pair, tuple) or len(pair) != 2 or not 0 <= pair[0] < pair[1] <= 2**n:
            raise ValueError(f"range {pair!r} is not a nonempty range of {n}-bit values")
    return widths, ranges


def _check_control_value(control_value):
    if control_value is None:
        return None
    return check_int("control_value", control_value, low=0, high=1)


def _control(control, control_value) -> dict:
    """``iterate_nested``'s control arguments for a 0- or 1-wire ``control`` register."""
    if control_value is None:
        return {"control": None, "control_value": 1}
    return {"control": control[0], "control_value": control_value}


def _high_specs(widths, ranges, low_bits) -> list[tuple[int, int, int, int]]:
    """``(start, n, lo, hi)`` of every register that keeps bits once the low ``low_bits`` bits of
    the concatenated address are taken off; a register split by the boundary keeps its top bits
    and the shifted range. A fully low address gives one zero-width register ``(0, 0, 0, 1)``."""
    starts = list(itertools.accumulate((0, *widths)))
    specs, remaining = [], low_bits
    for index in reversed(range(len(widths))):
        n, (lo, hi) = widths[index], ranges[index]
        if remaining >= n:
            remaining -= n
            continue
        specs.append((starts[index], n - remaining, lo >> remaining,
                      ((hi - 1) >> remaining) + 1))
        remaining = 0
    return list(reversed(specs)) or [(0, 0, 0, 1)]


def _registers(address, specs) -> list:
    return [(address[start:start + n], lo, hi) for start, n, lo, hi in specs]


def _concatenate(values, widths) -> int:
    result = 0
    for value, n in zip(values, widths, strict=True):
        result = (result << n) | value
    return result


def _dummy(sizes: dict[str, int]) -> dict[str, range]:
    """Consecutive integer wires for every register: what resource functions count on."""
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = range(start, start + size)
        start += size
    return wires


# --------------------------------------------------------------------------- QROAMRead


def _check_table(table, widths, live_ranges, width) -> tuple:
    """Canonical table: ``(address, words)`` pairs sorted by address, each address live."""
    table = canonical_ints("table", table)
    words = -(-width // WORD)
    seen = set()
    for pair in table:
        if not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[1], tuple):
            raise ValueError(f"table entries are (address, words) pairs, got {pair!r}")
        address, entry = pair
        if address in seen:
            raise ValueError(f"address {address} appears twice in the table")
        seen.add(address)
        if not _is_live(address, widths, live_ranges):
            raise ValueError(f"table entry at address {address} lies outside the live ranges "
                             f"{live_ranges}")
        if len(entry) != words or any(not 0 <= word < 2**WORD for word in entry):
            raise ValueError(f"the entry at address {address} must be {words} words in "
                             f"[0, 2**{WORD}), got {entry}")
        if _entry_value(entry) >> width:
            raise ValueError(f"the entry at address {address} does not fit in {width} bits")
    return tuple(sorted(table))


def _register_values(address, widths) -> list[int]:
    values, shift = [], sum(widths)
    for n in widths:
        shift -= n
        values.append((address >> shift) & (2**n - 1))
    return values


def _is_live(address, widths, ranges) -> bool:
    return 0 <= address < 2 ** sum(widths) and all(
        lo <= value < hi for value, (lo, hi) in zip(_register_values(address, widths), ranges,
                                                    strict=True))


def _entry_value(words) -> int:
    return sum(word << (WORD * i) for i, word in enumerate(words))


def _entries(table) -> dict[int, int]:
    return {address: _entry_value(words) for address, words in table}


def _wire_order(value, width) -> int:
    """``value`` with its ``width`` bits reversed: bit ``i`` of the result is the bit for wire
    ``i`` of a big-endian register (spec 6.7)."""
    return int(format(value, f"0{width}b")[::-1], 2)


def _live_positions(widths, live_ranges, block_bits) -> np.ndarray:
    """``(num_leaves, K)`` booleans: which block positions of each leaf hold a live address."""
    blocks = QROAMRead.leaf_blocks(widths, live_ranges, block_bits)
    return np.array([[_is_live((block << block_bits) | p, widths, live_ranges)
                      for p in range(2**block_bits)] for block in blocks], dtype=bool)


def _leaf_word_array(leaf_words, widths, live_ranges, width, block_bits):
    """Canonicalize the dynamic ``leaf_words`` once (L-23).

    Concrete words (Python or NumPy integers, nested sequences or arrays) must be integers in
    ``[0, 2**32)``, booleans and floats rejected, checked on the Python values before the
    ``int64`` cast, so that no word can wrap around before the check; every padding and dead
    position must be zero, and so must every bit beyond ``width`` (wire order). Tracers and
    abstract arrays (operators rebuilt by Catalyst, resource keys) are checked for their shape
    and an integer dtype only. Raises ``ValueError``.
    """
    shape = QROAMRead.leaf_words_shape(widths, live_ranges, width, block_bits)
    if isinstance(leaf_words, AbstractArray) or qp.math.is_tracer(leaf_words):
        if tuple(leaf_words.shape) != shape or not np.issubdtype(np.dtype(leaf_words.dtype),
                                                                 np.integer):
            raise ValueError(f"leaf_words must be an integer array of shape {shape}, got "
                             f"{tuple(leaf_words.shape)} of {leaf_words.dtype}")
        return leaf_words
    values = np.asarray(leaf_words, dtype=object)
    if values.shape != shape:
        raise ValueError(f"leaf_words must have shape {shape}, got {values.shape}")
    for word in values.flat:
        if isinstance(word, bool) or not isinstance(word, (int, np.integer)) or not (
                0 <= word < 2**WORD):
            raise ValueError(f"leaf words must be integers in [0, 2**{WORD}), got {word!r}")
    array = values.astype(WORD_DTYPE)
    if np.any(array[~_live_positions(widths, live_ranges, block_bits)]):
        raise ValueError("a padding or dead position of leaf_words holds a nonzero word")
    spare = WORD * shape[2] - width
    if spare and np.any(array[..., -1] >> (WORD - spare)):
        raise ValueError(f"leaf_words holds bits beyond width {width}")
    return jnp.asarray(array) if qp.capture.enabled() else array


def _swap_order(block_bits, low) -> list[int]:
    """The block position each register ``target, junk slot 0, ...`` holds after the swap
    network for the low address bits ``low``."""
    held = list(range(2**block_bits))
    for b in reversed(range(block_bits)):
        if (low >> b) & 1:
            for r in range(2**b):
                held[r], held[r + 2**b] = held[r + 2**b], held[r]
    return held


def _loaded_contents(entries, widths, live_ranges, width, block_bits, addresses) -> list[int]:
    """``target ‖ junk`` (big-endian) after an enabled read, for each address."""
    loaded = QROAMRead.loaded_ranges(widths, live_ranges, block_bits)
    size = 2**block_bits
    contents = []
    for address in addresses:
        if not _is_live(address, widths, loaded):
            contents.append(0)
            continue
        base, value = address - (address % size), 0
        for position in _swap_order(block_bits, address % size):
            value = (value << width) | entries.get(base | position, 0)
        contents.append(value)
    return contents


class QROAMRead(Operator2):
    r"""The paper's QROAM read: the requested entry into ``target``, the rest of its block into
    ``junk``, which stays until the measured erase (:class:`QROAMErase`).

    **Action** on every computational address ``a`` (spec 9.3), with :math:`K = 2^k` and
    ``k = block_bits``; a *block* is the set of ``K`` addresses that differ only in the low ``k``
    bits of the last register, and a block is *visited* if it meets the live ranges:

    ============================================== ================ =========================
    case                                           ``target``       ``junk``
    ============================================== ================ =========================
    visited block, live position, enable satisfied :math:`T[a]`     the block's other entries
    visited block, dead position (padding)         0                the block's other entries
    block not visited (e.g. one-body rows,         0                0
    failed :math:`x_o \ge X_o`)
    enable not satisfied                           0                0
    ============================================== ================ =========================

    ``T`` is the table that :meth:`leaf_words_from_table` packed into ``leaf_words``; entries
    missing from it are zero. ``junk`` slot ``r`` holds the block position
    ``junk_layout(k, a mod K)[r]``. :meth:`loaded_bits` gives ``target ‖ junk`` for every address.
    The read adds no phase, and the address is unchanged.

    **Registers.**

    =========== ================================= ======== ===============================
    name        width                             input    output
    =========== ================================= ======== ===============================
    ``address`` ``sum(address_widths)``           any      unchanged
    ``target``  ``width``                         ``|0>``  the entry
    ``junk``    ``(K - 1) width``                 ``|0>``  the block's other entries
    ``work``    unary-iteration ancillas          ``|0>``  ``|0>``
    ``control`` 0 wires, or 1 (the enable)        any      unchanged
    =========== ================================= ======== ===============================

    **Arguments.** Dynamic: ``leaf_words``, an ``int64`` array of shape :meth:`leaf_words_shape`,
    ``(num_leaves, K, ceil(width / 32))``: row ``l`` belongs to the ``l``-th unary leaf in
    visiting order (:meth:`leaf_blocks`), position ``p`` to the address
    ``(leaf_blocks[l] << k) | p``; 32 data bits per word in wire order (module docstring); zero
    at padding and dead positions. Callers build it from their compilable table with
    :meth:`leaf_words_from_table`. Compilable: ``address_widths`` (one or two widths);
    ``live_ranges`` (one ``(lo, hi)`` per register); ``width`` (int >= 1); ``block_bits``
    (``0 <= k <= address_widths[-1]``); ``control_value`` (``None`` without an enable wire, else
    the enabling value 0 or 1).

    **Pre- and postconditions.** ``target``, ``junk`` and ``work`` start in ``|0>``; ``work``
    ends in ``|0>``.

    **Decomposition.** (1) Nested exact unary iteration (:mod:`dfthc.ops.unary`) over the visited
    blocks: the first register over its live range, then the top ``w_p - k`` bits of the last
    register over the blocks that meet its live range, rooted at the enable. (2) At each block
    leaf ``l``, one ``qp.for_loop`` over its ``K width`` bits: iteration ``t`` takes position
    ``p = t // width`` and bit ``i = t % width`` and applies
    ``qp.cond(bit == 1, CNOT)(leaf flag, (target ‖ junk)[p width + i])`` with
    ``bit = (leaf_words[l, p, i // 32] >> (i % 32)) & 1`` (an ``X`` instead of the CNOT when the
    whole address is low and there is no enable, so no flag exists). No conditional surrounds a
    loop. (3) A swap-with-zero network on the low ``k`` address bits, most significant first:
    for bit ``b``, ``CSWAP`` register ``r`` with register ``r + 2**b`` for ``r < 2**b``, which
    moves the selected entry into ``target`` with ``(K - 1) width`` exact ``CSWAP``\s.

    **Adjoint and control.** Never adjointed or controlled: its body has loops and data
    conditionals (spec 7.4 rule 4; L-6), and no program wraps a composite that contains it in an
    adjoint or a control (spec 7.4 rule 6). The walk erases with :class:`QROAMErase`; use the
    enable instead of a control.

    **Cost.** :func:`dfthc.costs.qroam_read` with ``ranges = visited_ranges(...)``:
    :math:`T = U` (or :math:`U_2`) over the visited block ranges, controlled if there is an
    enable, plus :math:`(2^k - 1)` ``width``; ``M = 0``; ``A = U``. At FeMoco-54: outer
    84 + 72 = 156, inner 543 + 390 = 933. The rule declares ``num_leaves K width`` load CNOTs,
    or X gates for the flagless leaf (every position and bit, padding included), and is
    registered with ``exact=False``: the CNOT or X count is an upper bound, which ``qp.analyze``
    reports as well, because it counts every unhinted load conditional; T, M and A do not depend
    on the data and are exact.

    **Paper and dfthcbe.** Steps 2 and 6 (and the engine of 10 and 11); the QROAM of Berry et al.
    (Quantum 3, 208) with the junk kept until a measured erase. dfthcbe: the reads in
    ``OuterPrepare``/``InnerPrepare``; its truncated iteration and flat ``r`` target are
    deferred refinements (spec 9.3, "Known deltas").

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import QROAMRead

        table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
        shape = {"address_widths": (3,), "live_ranges": ((0, 5),), "width": 2, "block_bits": 1}
        words = QROAMRead.leaf_words_from_table(table, **shape)  # shape (3, 2, 1)
        program = lambda: QROAMRead(words, *shape.values(), None, address=[0, 1, 2],
                                    target=[3, 4], junk=[5, 6], work=[7], control=[])
        compiled_resources(program, num_wires=8).toffoli  # U(0, 3, 2, 0) + 2 = 2 + 2
    """

    dynamic_argnames = ("leaf_words",)
    compilable_argnames = ("address_widths", "live_ranges", "width", "block_bits",
                           "control_value")
    wire_argnames = ("address", "target", "junk", "work", "control")

    def __init__(self, leaf_words, address_widths, live_ranges, width, block_bits, control_value,
                 address, target, junk, work, control):
        address_widths, live_ranges = _check_address(address_widths, live_ranges)
        width = check_int("width", width, low=1)
        block_bits = check_int("block_bits", block_bits, low=0, high=address_widths[-1])
        control_value = _check_control_value(control_value)
        leaf_words = _leaf_word_array(leaf_words, address_widths, live_ranges, width, block_bits)
        check_wires("QROAMRead",
                    {"address": address, "target": target, "junk": junk, "work": work,
                     "control": control},
                    self.required_wires(address_widths, live_ranges, width, block_bits,
                                        control_value))
        super().__init__(leaf_words, address_widths, live_ranges, width, block_bits,
                         control_value, address, target, junk, work, control)

    @staticmethod
    def required_wires(address_widths, live_ranges, width, block_bits,
                       control_value) -> dict[str, int]:
        """Exact register sizes."""
        ranges = QROAMRead.visited_ranges(address_widths, live_ranges, block_bits)
        return {"address": sum(address_widths), "target": width,
                "junk": (2**block_bits - 1) * width,
                "work": work_required([n for _, _, n in ranges], control_value is not None),
                "control": int(control_value is not None)}

    @staticmethod
    def visited_ranges(address_widths, live_ranges, block_bits) -> list[tuple[int, int, int]]:
        """``(lo, hi, n)`` of each iterated register: the ``ranges`` of
        :func:`dfthc.costs.qroam_read`."""
        widths, ranges = _check_address(address_widths, live_ranges)
        return [(lo, hi, n) for _, n, lo, hi in _high_specs(widths, ranges, block_bits)]

    @staticmethod
    def loaded_ranges(address_widths, live_ranges, block_bits) -> tuple[tuple[int, int], ...]:
        """The loaded set (every address of a visited block) as one ``(lo, hi)`` per register:
        the ``covered`` argument of :class:`PhaseFixup` for this read's erase."""
        widths, ranges = _check_address(address_widths, live_ranges)
        (*outer, (lo, hi)) = ranges
        return (*outer, ((lo >> block_bits) << block_bits,
                         (((hi - 1) >> block_bits) + 1) << block_bits))

    @staticmethod
    def loaded_set(address_widths, live_ranges, block_bits) -> list[int]:
        """Every address of a visited block, in increasing order."""
        widths, _ = _check_address(address_widths, live_ranges)
        loaded = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
        return [_concatenate(values, widths)
                for values in itertools.product(*(range(lo, hi) for lo, hi in loaded))]

    @staticmethod
    def junk_layout(block_bits, low) -> list[int]:
        """The block position held by each junk slot, for low address bits ``low``."""
        return _swap_order(block_bits, low)[1:]

    @staticmethod
    def leaf_blocks(address_widths, live_ranges, block_bits) -> list[int]:
        """The high part of every unary leaf (its block), in the order the iteration visits the
        leaves: outermost register first, values increasing. Row ``l`` of ``leaf_words``."""
        widths, ranges = _check_address(address_widths, live_ranges)
        specs = _high_specs(widths, ranges, block_bits)
        return [_concatenate(values, [n for _, n, _, _ in specs])
                for values in itertools.product(*(range(lo, hi) for _, _, lo, hi in specs))]

    @staticmethod
    def leaf_words_shape(address_widths, live_ranges, width, block_bits) -> tuple[int, int, int]:
        """``(num_leaves, K, ceil(width / 32))``, the shape of ``leaf_words``."""
        return (len(QROAMRead.leaf_blocks(address_widths, live_ranges, block_bits)),
                2**block_bits, -(-width // WORD))

    @staticmethod
    def leaf_words_from_table(table, *, address_widths, live_ranges, width,
                              block_bits) -> np.ndarray:
        """The ``leaf_words`` of ``table`` (``(address, words)`` pairs, every address live, each
        entry below ``2**width`` in 32-bit words, least significant first): each entry in wire
        order at its leaf and position, zero at padding and dead positions. ``ValueError`` for an
        invalid table."""
        widths, ranges = _check_address(address_widths, live_ranges)
        entries = _entries(_check_table(table, widths, ranges, width))
        words = np.zeros(QROAMRead.leaf_words_shape(widths, ranges, width, block_bits),
                         dtype=WORD_DTYPE)
        for leaf, block in enumerate(QROAMRead.leaf_blocks(widths, ranges, block_bits)):
            for position in range(2**block_bits):
                stream = _wire_order(entries.get((block << block_bits) | position, 0), width)
                for w in range(words.shape[2]):
                    words[leaf, position, w] = (stream >> (WORD * w)) & (2**WORD - 1)
        return words

    @staticmethod
    def loaded_bits(table, address, *, address_widths, live_ranges, width, block_bits) -> int:
        """``target ‖ junk`` as one big-endian integer after an enabled read at ``address``;
        0 outside the loaded set."""
        return QROAMRead.loaded_contents(table, [address], address_widths=address_widths,
                                         live_ranges=live_ranges, width=width,
                                         block_bits=block_bits)[0]

    @staticmethod
    def loaded_contents(table, addresses, *, address_widths, live_ranges, width,
                        block_bits) -> list[int]:
        """:meth:`loaded_bits` of each address in ``addresses``, validating the table once."""
        widths, ranges = _check_address(address_widths, live_ranges)
        table = _check_table(table, widths, ranges, width)
        return _loaded_contents(_entries(table), widths, ranges, width, block_bits, addresses)


def _read_body(emit, leaf_words, address_widths, live_ranges, width, block_bits, control_value,
               address, target, junk, work, control):
    registers = [target] + [junk[r * width:(r + 1) * width] for r in range(2**block_bits - 1)]
    register = emit.wire_array(target, junk)
    leaves = itertools.count()

    def leaf(flag, values):
        emit.load(flag, leaf_words, next(leaves), register, 2**block_bits, width)

    iterate_nested(_registers(address, _high_specs(address_widths, live_ranges, block_bits)),
                   work, leaf, emit=emit, **_control(control, control_value))
    low = address[len(address) - block_bits:]
    for b in reversed(range(block_bits)):
        for r in range(2**b):
            for i in range(width):
                emit.cswap(low[block_bits - 1 - b], registers[r][i], registers[r + 2**b][i])


def _qroam_read_resources(leaf_words, address_widths, live_ranges, width, block_bits,
                          control_value, address, target, junk, work, control):
    counter = QROAMEmitter(counting=True)
    wires = _dummy(QROAMRead.required_wires(address_widths, live_ranges, width, block_bits,
                                            control_value))
    _read_body(counter, leaf_words, address_widths, live_ranges, width, block_bits,
               control_value, **wires)
    return as_resources(counter.counts)


@qp.register_resources(_qroam_read_resources, exact=False)
def _qroam_read(leaf_words, address_widths, live_ranges, width, block_bits, control_value,
                address, target, junk, work, control):
    _read_body(QROAMEmitter(), leaf_words, address_widths, live_ranges, width, block_bits,
               control_value, address, target, junk, work, control)


register_rule(QROAMRead, _qroam_read)


# --------------------------------------------------------------------------- PhaseFixup


def _parity_array(parities, size):
    """Canonicalize the dynamic ``parities`` once (L-23): a bool array under capture; in tape
    mode an object array of ``MeasurementValue``s, so both modes emit one conditional per covered
    address (K-10). Abstract arrays and tracers pass unchanged."""
    if isinstance(parities, AbstractArray) or qp.math.is_tracer(parities):
        if tuple(parities.shape) != (size,):
            raise ValueError(f"parities must have shape ({size},), got {tuple(parities.shape)}")
        return parities
    values = list(parities)
    if len(values) != size:
        raise ValueError(f"need one parity per covered address ({size}), got {len(values)}")
    if qp.capture.enabled():
        return jnp.stack([jnp.asarray(value, dtype=bool) for value in values])
    for i, value in enumerate(values):
        if not isinstance(value, MeasurementValue):
            raise ValueError(f"in tape mode every parity must be a MeasurementValue; entry {i} "
                             f"is {value!r}")
    return np.asarray(values, dtype=object)


class PhaseFixup(Operator2):
    r"""Multiply each address of the covered set by :math:`(-1)^{f(a)}`, with ``f`` a dynamic
    array; the standalone form of :func:`emit_phase_fixup`.

    **Action.** :math:`|a\rangle \mapsto (-1)^{f(a)}|a\rangle` for ``a`` in the covered set, and
    nothing elsewhere; no other phase (with an enable: only where the enable is satisfied). The
    covered set is the product of the ``covered`` ranges; for an erase it is the read's loaded
    set, :meth:`QROAMRead.loaded_ranges`.

    **Registers.**

    =========== ========================================= ======== ========================
    name        width                                     input    output
    =========== ========================================= ======== ========================
    ``address`` ``sum(address_widths)``                   any      phased
    ``onehot``  ``2**onehot_bits`` (0 if ``onehot_bits``  ``|0>``  ``|0>``
                is 0)
    ``work``    unary-iteration ancillas of the high part ``|0>``  ``|0>``
    ``control`` 0 wires, or 1 (the enable)                any      unchanged
    =========== ========================================= ======== ========================

    **Arguments.** Compilable: ``address_widths`` and ``covered`` (one ``(lo, hi)`` per register,
    the address model of :class:`QROAMRead`), ``onehot_bits`` (ob, ``0 <= ob <=
    sum(address_widths)``), ``control_value`` (``None``, 0 or 1). Dynamic: ``parities``, one
    boolean per covered address in the order of :meth:`address_order` (increasing addresses).
    Under capture it becomes a bool array; in tape mode every entry must be a
    ``MeasurementValue``, so that both modes emit a conditional for every covered address.

    **Pre- and postconditions.** ``onehot`` and ``work`` start and end in ``|0>``.

    **Decomposition** (spec 9.4). (1) The one-hot encoding of the low ob address bits: copy the
    top low bit and its negation into two flags, then split every flag ``f`` on each further bit
    ``a_j`` into ``f AND a_j`` (one ``TemporaryAND``) and ``f AND NOT a_j`` (one CNOT), so that
    ``onehot[j] = [low bits == j]``. (2) Exact unary iteration over the high part, visiting only
    high values with a covered address (:meth:`high_ranges`); at leaf ``h``, for each covered
    ``a = (h << ob) | j``, ``qp.cond(parities[a], CZ)(leaf flag, onehot[j])`` with
    ``qp.hint({"branch-prob": 0.5})``. (3) The measured uncompute of the tree. With
    ``onehot_bits = 0`` it is plain iteration with a conditional ``Z`` on every leaf flag.

    **Adjoint and control.** Never adjointed or controlled; use the enable.

    **Cost.** :func:`dfthc.costs.phase_fixup` with ``high_ranges(...)``: :math:`T = U` (or
    :math:`U_2`) over the covered high ranges, controlled if there is an enable, plus
    :math:`O(ob) = 2^{ob} - 2` (0 for ob = 0); ``M = 0``. At FeMoco-54: outer 22 + 14, inner
    70 + 126.

    **Paper and dfthcbe.** The phase fixup of the measured QROAM erase (Berry et al., Quantum 3,
    208, App. C), with ``CZ`` gates applied directly instead of a parity load into a
    :math:`|-\rangle` ancilla (spec 4.4, item 2). dfthcbe: the erase fixups of its unprepares.

    **Example.**

    .. code-block:: python

        import numpy as np
        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import PhaseFixup

        program = lambda: PhaseFixup(np.ones(6, dtype=bool), (3,), ((0, 6),), 1, None,
                                     address=[0, 1, 2], onehot=[3, 4], work=[5], control=[])
        compiled_resources(program, num_wires=6).toffoli  # U(0, 3, 2, 0) + O(1) = 2
    """

    dynamic_argnames = ("parities",)
    compilable_argnames = ("address_widths", "covered", "onehot_bits", "control_value")
    wire_argnames = ("address", "onehot", "work", "control")

    def __init__(self, parities, address_widths, covered, onehot_bits, control_value, address,
                 onehot, work, control):
        address_widths, covered = _check_address(address_widths, covered)
        onehot_bits = check_int("onehot_bits", onehot_bits, low=0, high=sum(address_widths))
        control_value = _check_control_value(control_value)
        parities = _parity_array(parities, len(self.address_order(address_widths, covered)))
        check_wires("PhaseFixup",
                    {"address": address, "onehot": onehot, "work": work, "control": control},
                    self.required_wires(address_widths, covered, onehot_bits, control_value))
        super().__init__(parities, address_widths, covered, onehot_bits, control_value, address,
                         onehot, work, control)

    @staticmethod
    def required_wires(address_widths, covered, onehot_bits, control_value) -> dict[str, int]:
        """Exact register sizes."""
        ranges = PhaseFixup.high_ranges(address_widths, covered, onehot_bits)
        return {"address": sum(address_widths), "onehot": 2**onehot_bits if onehot_bits else 0,
                "work": work_required([n for _, _, n in ranges], control_value is not None),
                "control": int(control_value is not None)}

    @staticmethod
    def address_order(address_widths, covered) -> list[int]:
        """The covered addresses in the canonical order of ``parities``: increasing."""
        widths, covered = _check_address(address_widths, covered)
        return [_concatenate(values, widths)
                for values in itertools.product(*(range(lo, hi) for lo, hi in covered))]

    @staticmethod
    def high_ranges(address_widths, covered, onehot_bits) -> list[tuple[int, int, int]]:
        """``(lo, hi, n)`` of each iterated high register: the ``high_ranges`` of
        :func:`dfthc.costs.phase_fixup`."""
        widths, covered = _check_address(address_widths, covered)
        return [(lo, hi, n) for _, n, lo, hi in _high_specs(widths, covered, onehot_bits)]


def _onehot_compute(emit, low, onehot):
    size = 2 ** len(low)
    emit.cnot(low[0], onehot[size // 2])
    emit.x(onehot[0])
    emit.cnot(low[0], onehot[0])
    for t in range(1, len(low)):
        step = size >> t
        for s in range(0, size, step):
            emit.and_(onehot[s], low[t], onehot[s + step // 2])
            emit.cnot(onehot[s + step // 2], onehot[s])


def _onehot_uncompute(emit, low, onehot):
    size = 2 ** len(low)
    for t in reversed(range(1, len(low))):
        step = size >> t
        for s in reversed(range(0, size, step)):
            emit.cnot(onehot[s + step // 2], onehot[s])
            emit.unand(onehot[s], low[t], onehot[s + step // 2])
    emit.cnot(low[0], onehot[0])
    emit.x(onehot[0])
    emit.cnot(low[0], onehot[size // 2])


def emit_phase_fixup(parities, *, address, address_widths, covered, onehot, onehot_bits, work,
                     control=None, control_value=1, emit=None) -> None:
    r"""Emit the gates of :class:`PhaseFixup` inside a rule (spec 9.4).

    ``parities`` holds one entry per covered address in :meth:`PhaseFixup.address_order`: a
    predicate for ``qp.cond``, or the Python ``False``, which emits no conditional. Callers decide
    the ``False`` entries from compilable data only (the erase: its zero masks), so both modes
    emit the same gates. ``control`` is the enable wire or ``None``; ``onehot`` has exactly
    ``2**onehot_bits`` wires (none for 0) in ``|0>``; ``work`` has
    ``PhaseFixup.required_wires(...)["work"]`` wires in ``|0>``. ``emit`` is a
    :class:`QROAMEmitter` (default: emitting).
    """
    emit = QROAMEmitter() if emit is None else emit
    widths, covered = _check_address(address_widths, covered)
    if not 0 <= onehot_bits <= sum(widths):
        raise ValueError(f"onehot_bits must be in [0, {sum(widths)}], got {onehot_bits}")
    order = PhaseFixup.address_order(widths, covered)
    if len(parities) != len(order):
        raise ValueError(f"need one parity per covered address ({len(order)}), "
                         f"got {len(parities)}")
    if len(onehot) != (2**onehot_bits if onehot_bits else 0):
        raise ValueError(f"onehot needs {2**onehot_bits if onehot_bits else 0} wires, "
                         f"got {len(onehot)}")
    index = {a: i for i, a in enumerate(order)}
    low = list(address[len(address) - onehot_bits:])
    onehot = list(onehot)
    specs = _high_specs(widths, covered, onehot_bits)
    high_widths = [n for _, n, _, _ in specs]

    def leaf(flag, values):
        block = _concatenate(values, high_widths)
        for j in range(2**onehot_bits):
            i = index.get((block << onehot_bits) | j)
            if i is None or parities[i] is False:
                continue
            if onehot_bits == 0:
                emit.cond_z(parities[i], flag)
            elif flag is None:
                emit.cond_z(parities[i], onehot[j])
            else:
                emit.cond_cz(parities[i], flag, onehot[j])

    if onehot_bits:
        _onehot_compute(emit, low, onehot)
    iterate_nested(_registers(address, specs), work, leaf, control, control_value, emit=emit)
    if onehot_bits:
        _onehot_uncompute(emit, low, onehot)


def _phase_fixup_body(emit, parities, address_widths, covered, onehot_bits, control_value,
                      address, onehot, work, control):
    emit_phase_fixup(parities, address=address, address_widths=address_widths, covered=covered,
                     onehot=onehot, onehot_bits=onehot_bits, work=work, emit=emit,
                     **_control(control, control_value))


def _phase_fixup_resources(parities, address_widths, covered, onehot_bits, control_value,
                           address, onehot, work, control):
    counter = QROAMEmitter(counting=True)
    size = len(PhaseFixup.address_order(address_widths, covered))
    wires = _dummy(PhaseFixup.required_wires(address_widths, covered, onehot_bits, control_value))
    _phase_fixup_body(counter, [True] * size, address_widths, covered, onehot_bits, control_value,
                      **wires)
    return as_resources(counter.counts)


@qp.register_resources(_phase_fixup_resources)
def _phase_fixup(parities, address_widths, covered, onehot_bits, control_value, address, onehot,
                 work, control):
    size = len(PhaseFixup.address_order(address_widths, covered))
    _phase_fixup_body(QROAMEmitter(), [parities[i] for i in range(size)], address_widths,
                      covered, onehot_bits, control_value, address, onehot, work, control)


register_rule(PhaseFixup, _phase_fixup)


# --------------------------------------------------------------------------- QROAMErase


class QROAMErase(Operator2):
    r"""Erase what the matching :class:`QROAMRead` loaded, by X measurement and a phase fixup.

    **Action.** Returns ``target``, ``junk`` and ``work`` to :math:`|0\rangle` and leaves the
    address state exactly as it was before the read, for every measurement outcome.

    **Registers.**

    ================ =========================================== ============= ==============
    name             width                                       input         output
    ================ =========================================== ============= ==============
    ``address``      ``sum(address_widths)``                     any           unchanged
    ``target``       ``width``                                   the read's    ``|0>``
    ``junk``         ``(K - 1) width``                           the read's    ``|0>``
    ``work``         unary-iteration ancillas of the fixup        ``|0>``       ``|0>``
    ``control``      0 wires, or 1 (the read's enable)           any           unchanged
    ``extra_onehot`` ``max(2**onehot_bits - K width, 0)``        ``|0>``       ``|0>``
    ================ =========================================== ============= ==============

    **Compilable arguments.** The matching read's ``table``, ``address_widths``,
    ``live_ranges``, ``width``, ``block_bits`` and ``control_value``, plus the fixup's
    ``onehot_bits`` (``0 <= ob <= sum(address_widths)``).

    **Pre- and postconditions.** ``target ‖ junk`` hold exactly what the matching read produced
    for the current address (operations in between may act on the address only in ways that
    preserve this). Afterwards ``target``, ``junk``, ``work`` and ``extra_onehot`` are
    :math:`|0\rangle`, and the address state equals its pre-read state exactly.

    **Decomposition** (spec 9.5). (1) ``H`` and ``qp.measure(q, reset=True)`` on every qubit of
    ``target ‖ junk``, giving the outcomes :math:`\xi`. (2) The parities
    :math:`f(a) = \xi \cdot g(a) \bmod 2`, with :math:`g(a) =` :meth:`QROAMRead.loaded_bits`, by
    :func:`dfthc.compile.mcm_parities` from masks built here from the compilable table, for every
    address of the read's loaded set (dead positions included: their junk is not zero). (3)
    :func:`emit_phase_fixup` over the loaded set with the freshly reset ``target ‖ junk`` (and
    ``extra_onehot``) as its one-hot register; a zero mask emits no conditional.

    **Adjoint and control.** Never: its rule measures (spec 7.4, rule 5).

    **Cost.** :func:`dfthc.costs.qroam_erase` with the fixup's ``PhaseFixup.high_ranges`` of
    ``QROAMRead.loaded_ranges``: :math:`T` = the fixup's, :math:`M = 2^k` ``width``. At
    FeMoco-54: outer (ob = 4) T = 22 + 14 = 36, M = 96; inner (ob = 7) T = 70 + 126 = 196,
    M = 416.

    **Paper and dfthcbe.** Steps 21 and 26: the measured erase of Berry et al. (Quantum 3, 208,
    App. C). dfthcbe: the erases inside its unprepares.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.qroam import QROAMErase

        table = tuple((a, (v,)) for a, v in enumerate([1, 2, 3, 1, 2]))
        program = lambda: QROAMErase(table, (3,), ((0, 5),), 2, 1, None, 1, address=[0, 1, 2],
                                     target=[3, 4], junk=[5, 6], work=[7], control=[],
                                     extra_onehot=[])
        compiled_resources(program, num_wires=8).measurements  # 4
    """

    compilable_argnames = ("table", "address_widths", "live_ranges", "width", "block_bits",
                           "control_value", "onehot_bits")
    wire_argnames = ("address", "target", "junk", "work", "control", "extra_onehot")

    def __init__(self, table, address_widths, live_ranges, width, block_bits, control_value,
                 onehot_bits, address, target, junk, work, control, extra_onehot):
        address_widths, live_ranges = _check_address(address_widths, live_ranges)
        width = check_int("width", width, low=1)
        block_bits = check_int("block_bits", block_bits, low=0, high=address_widths[-1])
        control_value = _check_control_value(control_value)
        onehot_bits = check_int("onehot_bits", onehot_bits, low=0, high=sum(address_widths))
        table = _check_table(table, address_widths, live_ranges, width)
        check_wires("QROAMErase",
                    {"address": address, "target": target, "junk": junk, "work": work,
                     "control": control, "extra_onehot": extra_onehot},
                    self.required_wires(table, address_widths, live_ranges, width, block_bits,
                                        control_value, onehot_bits))
        super().__init__(table, address_widths, live_ranges, width, block_bits, control_value,
                         onehot_bits, address, target, junk, work, control, extra_onehot)

    @staticmethod
    def required_wires(table, address_widths, live_ranges, width, block_bits, control_value,
                       onehot_bits) -> dict[str, int]:
        """Exact register sizes."""
        covered = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
        fixup = PhaseFixup.required_wires(address_widths, covered, onehot_bits, control_value)
        return {"address": sum(address_widths), "target": width,
                "junk": (2**block_bits - 1) * width, "work": fixup["work"],
                "control": int(control_value is not None),
                "extra_onehot": max(fixup["onehot"] - 2**block_bits * width, 0)}

    @staticmethod
    def fixup_high_ranges(address_widths, live_ranges, block_bits,
                          onehot_bits) -> list[tuple[int, int, int]]:
        """The fixup's iterated ranges: the ``ranges`` of :func:`dfthc.costs.qroam_erase`."""
        covered = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
        return PhaseFixup.high_ranges(address_widths, covered, onehot_bits)


def _erase_body(emit, table, address_widths, live_ranges, width, block_bits, control_value,
                onehot_bits, address, target, junk, work, control, extra_onehot):
    qubits = list(target) + list(junk)
    outcomes = [emit.measure_x(qubit) for qubit in qubits]
    covered = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
    order = PhaseFixup.address_order(address_widths, covered)
    masks = _loaded_contents(_entries(table), address_widths, live_ranges, width, block_bits,
                             order)
    parities = emit.parities(outcomes[::-1], masks)
    onehot = (qubits + list(extra_onehot))[:2**onehot_bits] if onehot_bits else []
    emit_phase_fixup(parities, address=address, address_widths=address_widths, covered=covered,
                     onehot=onehot, onehot_bits=onehot_bits, work=work, emit=emit,
                     **_control(control, control_value))


def _qroam_erase_resources(table, address_widths, live_ranges, width, block_bits, control_value,
                           onehot_bits, address, target, junk, work, control, extra_onehot):
    counter = QROAMEmitter(counting=True)
    wires = _dummy(QROAMErase.required_wires(table, address_widths, live_ranges, width,
                                             block_bits, control_value, onehot_bits))
    _erase_body(counter, table, address_widths, live_ranges, width, block_bits, control_value,
                onehot_bits, **wires)
    return as_resources(counter.counts)


@qp.register_resources(_qroam_erase_resources)
def _qroam_erase(table, address_widths, live_ranges, width, block_bits, control_value,
                 onehot_bits, address, target, junk, work, control, extra_onehot):
    _erase_body(QROAMEmitter(), table, address_widths, live_ranges, width, block_bits,
                control_value, onehot_bits, address, target, junk, work, control, extra_onehot)


register_rule(QROAMErase, _qroam_erase)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_qroam_erase.py -q`
Expected: `19 passed` (27 s in the serial replay, including the two `MID_NAPHTHALENE` compiles; the five-seed smoke run takes 7 to 13 s and 306 MB).

Run: `scripts/test.sh tests/ops/test_certificates_qroam.py -q`
Expected: `7 passed` (3 to 7 s; nightly tier, run targeted here).

Run: `scripts/test.sh tests/ops/test_qroam_read.py tests/ops/test_phase_fixup.py -m "not slow" -q`
Expected: `69 passed`.

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `643 passed` (626 + 17).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/qroam.py tests/ops/test_qroam_erase.py tests/ops/test_certificates_qroam.py
git commit -m "Add QROAMErase and the read-erase outcome certificates"
```

---

### Task 1C.6: `ZeroReflection`

**Task id:** `1C.6` · **Branch:** `phase1/1C.6` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 1C.1 (which creates `src/dfthc/ops/__init__.py` and `_validate.py`; `dfthc.costs` from 1A.7 comes with it)

**Files:**
- Create: `src/dfthc/ops/reflections.py` (the `ZeroReflection` part; Task 3.2 adds `RefT2` and `RefAB`), `tests/ops/test_zero_reflection.py`

**Interfaces:**
- Consumes: `dfthc.ops._validate.check_wires` (1C.1); `dfthc.compile.register_rule`, `resource_key`, `compiled_resources` (0.2); `dfthc.costs.zero_reflection` (1A.7); PennyLane's `MultiControlledX` with zeroed work wires (its cost is pinned by `tests/test_primitive_costs.py`).
- Produces, in `dfthc.ops.reflections`:
  - `ZeroReflection(zero_wires, one_wires, work)`: no compilable arguments; `n_w = len(zero_wires) + len(one_wires) >= 1`; `work` has at least `max(n_w - 3, 0)` wires;
  - `ZeroReflection.required_wires(n_zero, n_one, controlled=False) -> {"zero_wires": n_zero, "one_wires": n_one, "work": max(n_w - 2, 0) if controlled else max(n_w - 3, 0)}`;
  - the rules `_zero_reflection` (base), `_adjoint_zero_reflection` (pinned as `"Adjoint(ZeroReflection)"`: the operator itself) and `_c_zero_reflection` (pinned as `"C(ZeroReflection)"`: one control wire with control value 1; `ValueError` if `work` is shorter than the controlled size).

**Acceptance tests** (33 tests in `tests/ops/test_zero_reflection.py`, 1 of them `slow`):
- The exact compiled matrix (one lightning run each: every system wire entangled with a reference wire, so the output amplitudes are the matrix entries, and `work` must return to `|0>`): 2Π − I with the global sign for n_w = 1, 2, 3, 6 and both pattern values of the target wire (8 tests); |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ (2Π − I) for the controlled version (8 tests); the adjoint at n_w = 3.
- T4: T = max(n_w − 2, 0), controlled n_w − 1, for n_w = 1, 2, 3, 6 and 46 (FeMoco `RefAB`, table-free), each program applying both pattern values (10 tests; the controlled 46-wire compile takes 8 to 18 s and is marked `slow`), with one plain `Toffoli` per `MultiControlledX` of two or more controls.
- T5 plain and controlled; T2 for the three rules at two sizes; the controlled rule's work check; `required_wires`, `ValueError` for n_w = 0 and for too little work.

**Definition of done:** `33 passed` targeted (32 in the per-commit tier); the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_zero_reflection.py`:

```python
"""ZeroReflection (spec 9.6): exact matrices with the global sign, controlled version, counts."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.reflections import (
    ZeroReflection,
    _adjoint_zero_reflection,
    _c_zero_reflection,
    _zero_reflection,
)


def split(n_w, target_value):
    """Pattern wires 0 .. n_w - 1: zeros, then the last wire with ``target_value``."""
    wires = list(range(n_w))
    return (wires, []) if target_value == 0 else (wires[:-1], wires[-1:])


def choi_matrix(num_system, apply, num_work):
    """The matrix of ``apply`` on wires 0 .. num_system - 1, from one compiled lightning run.

    Each system wire is entangled with a reference wire (H, CNOT), so the output amplitude of
    |y>_system |x>_reference |0>_work is U[y, x] / sqrt(2^n); work must come back to |0>.
    """
    system = list(range(num_system))
    reference = list(range(num_system, 2 * num_system))
    work = list(range(2 * num_system, 2 * num_system + num_work))

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=2 * num_system + num_work))
    def circuit():
        for s, r in zip(system, reference, strict=True):
            qp.H(r)
            qp.CNOT([r, s])
        apply(system, work)
        return qp.state()

    state = np.asarray(circuit()).reshape(2**num_system, 2**num_system, 2**num_work)
    assert np.allclose(state[:, :, 1:], 0, rtol=0, atol=1e-10)
    return state[:, :, 0] * np.sqrt(2**num_system)


def reflection_diagonal(n_w, target_value):
    pattern = 1 if target_value == 1 else 0  # the integer of the pattern, wire 0 most significant
    return np.where(np.arange(2**n_w) == pattern, 1.0, -1.0)


@pytest.mark.compiled
@pytest.mark.parametrize("target_value", [0, 1])
@pytest.mark.parametrize("n_w", [1, 2, 3, 6])
def test_exact_matrix_with_global_sign(n_w, target_value):
    """T3: the compiled matrix equals 2 Pi - I exactly, global sign included."""
    zero, one = split(n_w, target_value)
    work = ZeroReflection.required_wires(len(zero), len(one))["work"]

    def apply(system, work_wires):
        ZeroReflection([system[w] for w in zero], [system[w] for w in one], work_wires)

    matrix = choi_matrix(n_w, apply, work)
    assert np.allclose(matrix, np.diag(reflection_diagonal(n_w, target_value)), rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize("target_value", [0, 1])
@pytest.mark.parametrize("n_w", [1, 2, 3, 6])
def test_exact_controlled_matrix(n_w, target_value):
    """T3: C(ZeroReflection) equals |0><0| (x) I + |1><1| (x) (2 Pi - I); a wrong global sign of
    the uncontrolled reflection would show here as a relative phase."""
    zero, one = split(n_w, target_value)
    work = ZeroReflection.required_wires(len(zero), len(one), controlled=True)["work"]

    def apply(system, work_wires):
        control, pattern = system[0], system[1:]
        qp.ctrl(ZeroReflection([pattern[w] for w in zero], [pattern[w] for w in one],
                               work_wires), control=control)

    matrix = choi_matrix(n_w + 1, apply, work)
    expected = np.concatenate([np.ones(2**n_w), reflection_diagonal(n_w, target_value)])
    assert np.allclose(matrix, np.diag(expected), rtol=0, atol=1e-10)


@pytest.mark.compiled
def test_adjoint_is_the_reflection_itself():
    zero, one = split(3, 1)

    def apply(system, work_wires):
        qp.adjoint(ZeroReflection([system[w] for w in zero], [system[w] for w in one],
                                  work_wires))

    assert np.allclose(choi_matrix(3, apply, 0), np.diag(reflection_diagonal(3, 1)), rtol=0,
                       atol=1e-10)


COUNT_CASES = [pytest.param(n_w, controlled, id=f"{n_w}-{'controlled' if controlled else 'plain'}")
               for n_w in (1, 2, 3, 6, 46) for controlled in (False, True)
               if (n_w, controlled) != (46, True)]
COUNT_CASES.append(pytest.param(46, True, id="46-controlled", marks=pytest.mark.slow))


@pytest.mark.compiled
@pytest.mark.parametrize(("n_w", "controlled"), COUNT_CASES)
def test_toffoli_count_matches_the_closed_form(n_w, controlled):
    """T4: T = max(n_w - 2, 0), controlled n_w - 1, for both pattern values of the target wire
    (one program applies both, so it counts twice); n_w = 46 is FeMoco's RefAB (table-free).
    The controlled 46-wire compile takes 8 to 18 s, so it runs in the nightly tier."""
    work = list(range(n_w, n_w + ZeroReflection.required_wires(n_w, 0, controlled)["work"]))
    control = n_w + len(work)

    def program():
        for target_value in (0, 1):
            zero, one = split(n_w, target_value)
            if controlled:
                qp.ctrl(ZeroReflection(zero, one, work), control=control)
            else:
                ZeroReflection(zero, one, work)

    resources = compiled_resources(program, num_wires=control + 1)
    expected = 2 * costs.zero_reflection(n_w, controlled)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.counts.get("Toffoli", 0) == 2 * int(n_w + controlled >= 3)


@pytest.mark.compiled
@pytest.mark.parametrize("controlled", [False, True], ids=["plain", "controlled"])
def test_no_dfthc_operator_survives_at_device_level(controlled):
    def program():
        op = ZeroReflection([0, 1, 2], [3], [4, 5])
        if controlled:
            qp.ctrl(op, control=6)

    compiled_resources(program, num_wires=7, level="device")


@pytest.mark.parametrize(("zero", "one", "work"), [([0], [], []), ([0, 1, 2], [3], [4])])
def test_rules_are_valid(zero, one, work):
    """T2 for the base, the adjoint and the controlled rule (tape mode)."""
    op = ZeroReflection(zero, one, work + [9])
    qp.testing.assert_valid_decomp_rule(op, _zero_reflection, check_matrix=False)
    qp.testing.assert_valid_decomp_rule(qp.adjoint(op), _adjoint_zero_reflection,
                                        check_matrix=False)
    qp.testing.assert_valid_decomp_rule(qp.ctrl(op, control=10), _c_zero_reflection,
                                        check_matrix=False)


def test_controlled_rule_needs_one_more_work_wire():
    op = ZeroReflection([0, 1, 2], [3], [4])
    with pytest.raises(ValueError, match="C\\(ZeroReflection\\) needs 2 work wires, got 1"):
        qp.testing.assert_valid_decomp_rule(qp.ctrl(op, control=10), _c_zero_reflection,
                                            check_matrix=False)


def test_required_wires_and_validation():
    assert ZeroReflection.required_wires(46, 0) == {"zero_wires": 46, "one_wires": 0, "work": 43}
    assert ZeroReflection.required_wires(5, 1, controlled=True)["work"] == 4
    assert ZeroReflection.required_wires(0, 1)["work"] == 0
    with pytest.raises(ValueError, match="n_w >= 1"):
        ZeroReflection([], [], [])
    with pytest.raises(ValueError, match="'work' has 1 wires; expected at least 2"):
        ZeroReflection([0, 1, 2, 3], [4], [5])
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_zero_reflection.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.ops.reflections'`.

- [ ] **Step 3: Write `ZeroReflection`**

Create `src/dfthc/ops/reflections.py`:

```python
"""Reflections about a basis pattern, exact including the global sign (spec 9.6, 9.19, 9.20)."""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_wires


class ZeroReflection(Operator2):
    r""":math:`2\Pi - I` with :math:`\Pi` the projector onto zeros on ``zero_wires`` and ones on
    ``one_wires``.

    **Action.** :math:`2\Pi - I`, :math:`\Pi = |0\cdots0\rangle\langle 0\cdots0|_{zero} \otimes
    |1\cdots1\rangle\langle 1\cdots1|_{one}`, exactly, including the global sign (spec 6.6).

    **Registers.** With :math:`n_w` = ``len(zero_wires) + len(one_wires)``:

    ============== ================================= ======== =========
    name           width                             input    output
    ============== ================================= ======== =========
    ``zero_wires`` any (pattern value 0)             any      reflected
    ``one_wires``  any (pattern value 1)             any      reflected
    ``work``       at least :math:`\max(n_w - 3, 0)` ``|0>``  ``|0>``
                   (controlled: :math:`\max(n_w - 2,
                   0)`)
    ============== ================================= ======== =========

    **Compilable arguments.** None; the sizes follow from the wires. :math:`n_w \ge 1`
    (``ValueError`` otherwise).

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition.** The last pattern wire is the target. For :math:`n_w \ge 2`: ``X`` on the
    target if its pattern value is 0, ``H`` on the target, ``MultiControlledX`` with the other
    :math:`n_w - 1` wires as controls (their pattern values as control values) and
    :math:`\max(n_w - 3, 0)` zeroed work wires, ``H``, the ``X`` again: that is
    :math:`I - 2\Pi`. Then ``GlobalPhase(pi)``, which multiplies by -1. For :math:`n_w = 1` a
    ``Z`` on the pattern wire (X-conjugated for pattern value 0), because PennyLane rejects a
    one-wire ``MultiControlledX``.

    **Adjoint and control.** ``Adjoint(ZeroReflection)`` is pinned: the operator is
    self-inverse. ``C(ZeroReflection)`` is pinned, for one control wire with control value 1: the
    multi-controlled Z over the control and the pattern, then ``Z`` on the control, which is
    exactly :math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes (2\Pi - I)`; for
    :math:`n_w = 1` a ``CZ``. The controlled rule uses :math:`\max(n_w - 2, 0)` of ``work``
    (``ValueError`` if ``work`` is shorter); generic ``C(GlobalPhase)`` is never used.

    **Cost.** :func:`dfthc.costs.zero_reflection`: :math:`T = \max(n_w - 2, 0)`, controlled
    :math:`n_w - 1` (one ``Toffoli`` and a ``TemporaryAND`` ladder with measured uncompute);
    ``M = 0``.

    **Paper and dfthcbe.** The reflections of steps 23 and 28 (:math:`\mathrm{REF}_{T_2}`,
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}`); dfthcbe: Qualtran's ``Reflection``-based
    ``RefT2`` and ``RefAB``.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import ZeroReflection

        program = lambda: ZeroReflection(zero_wires=[0, 1, 2], one_wires=[3], work=[4])
        compiled_resources(program, num_wires=5).toffoli  # 2
    """

    wire_argnames = ("zero_wires", "one_wires", "work")

    def __init__(self, zero_wires, one_wires, work):
        n_zero, n_one = len(zero_wires), len(one_wires)
        if n_zero + n_one < 1:
            raise ValueError("ZeroReflection needs n_w >= 1 pattern wires, got 0")
        check_wires("ZeroReflection",
                    {"zero_wires": zero_wires, "one_wires": one_wires, "work": work},
                    self.required_wires(n_zero, n_one), at_least=("work",))
        super().__init__(zero_wires, one_wires, work)

    @staticmethod
    def required_wires(n_zero: int, n_one: int, controlled: bool = False) -> dict[str, int]:
        """Register sizes; the controlled version needs one more work wire."""
        n_w = n_zero + n_one
        return {"zero_wires": n_zero, "one_wires": n_one,
                "work": max(n_w - 2, 0) if controlled else max(n_w - 3, 0)}


def _pattern(zero_wires, one_wires):
    """Controls, their values, the target wire and the target's pattern value."""
    wires = list(zero_wires) + list(one_wires)
    values = [0] * len(zero_wires) + [1] * len(one_wires)
    return wires[:-1], values[:-1], wires[-1], values[-1]


def _mcx_key(num_controls, values):
    num_work = max(num_controls - 2, 0)
    return resource_key(lambda: qp.MultiControlledX(
        Wire[num_controls + 1], control_values=list(values),
        work_wires=Wire[num_work] if num_work else None, work_wire_type="zeroed"))


def _mcz(controls, values, target, target_value, work):
    """Multi-controlled Z on ``controls`` (``values``) and ``target`` (``target_value``)."""
    num_work = max(len(controls) - 2, 0)
    if target_value == 0:
        qp.X(target)
    qp.H(target)
    qp.MultiControlledX(wires=list(controls) + [target], control_values=list(values),
                        work_wires=list(work[:num_work]) if num_work else None,
                        work_wire_type="zeroed")
    qp.H(target)
    if target_value == 0:
        qp.X(target)


def _zero_reflection_resources(zero_wires, one_wires, work):
    controls, values, _, last = _pattern(zero_wires, one_wires)
    if not controls:
        return {qp.Z: 1, qp.X: 2 * (last == 0), qp.GlobalPhase: 1}
    return {_mcx_key(len(controls), values): 1, qp.H: 2, qp.X: 2 * (last == 0),
            qp.GlobalPhase: 1}


@qp.register_resources(_zero_reflection_resources)
def _zero_reflection(zero_wires, one_wires, work):
    controls, values, target, last = _pattern(zero_wires, one_wires)
    if controls:
        _mcz(controls, values, target, last, work)
    else:
        if last == 0:
            qp.X(target)
        qp.Z(target)
        if last == 0:
            qp.X(target)
    qp.GlobalPhase(np.pi)


def _adjoint_zero_reflection_resources(base):
    return {resource_key(lambda: ZeroReflection(Wire[len(base.zero_wires)],
                                                Wire[len(base.one_wires)],
                                                Wire[len(base.work)])): 1}


@qp.register_resources(_adjoint_zero_reflection_resources)
def _adjoint_zero_reflection(base):
    ZeroReflection(base.zero_wires, base.one_wires, base.work)


def _c_zero_reflection_resources(base, control_wires, control_values, work_wires,
                                 work_wire_type):
    controls, values, _, last = _pattern(base.zero_wires, base.one_wires)
    if not controls:
        return {qp.CZ: 1, qp.X: 2 * (last == 0), qp.Z: 1}
    return {_mcx_key(len(controls) + 1, [1] + values): 1, qp.H: 2, qp.X: 2 * (last == 0),
            qp.Z: 1}


@qp.register_resources(_c_zero_reflection_resources)
def _c_zero_reflection(base, control_wires, control_values, work_wires, work_wire_type):
    controls, values, target, last = _pattern(base.zero_wires, base.one_wires)
    control = control_wires[0]
    needed = ZeroReflection.required_wires(len(base.zero_wires), len(base.one_wires), True)
    if len(base.work) < needed["work"]:
        raise ValueError(f"C(ZeroReflection) needs {needed['work']} work wires, got "
                         f"{len(base.work)}")
    if controls:
        _mcz([control, *controls], [1, *values], target, last, base.work)
    else:
        if last == 0:
            qp.X(target)
        qp.CZ([control, target])
        if last == 0:
            qp.X(target)
    qp.Z(control)


register_rule(ZeroReflection, _zero_reflection)
register_rule("Adjoint(ZeroReflection)", _adjoint_zero_reflection)
register_rule("C(ZeroReflection)", _c_zero_reflection)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_zero_reflection.py -q`
Expected: `33 passed` (35 to 85 s with 2 workers, depending on the machine's load; the `slow` controlled 46-wire count takes 8 s in a warm worker, 18 s in a fresh process, 524 MB).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `495 passed` (463 + 32).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/reflections.py tests/ops/test_zero_reflection.py
git commit -m "Add ZeroReflection with pinned adjoint and controlled rules"
```
