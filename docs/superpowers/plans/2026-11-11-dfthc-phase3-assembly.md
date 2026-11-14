# DFTHC Phase 3, Tasks 3.3 to 3.5: Block Encoding, Walk and End-to-End Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the deliverable of spec 9.21 and 9.22: the DFTHC block encoding `DFTHCBlockEncoding` (steps 1 to 27), the qubitized walk `DFTHCWalk` (steps 1 to 28) and their explicit controlled classes `ControlledDFTHCBlockEncoding` and `ControlledDFTHCWalk`, which control only the two reflections; count them exactly against `dfthc.costs` as whole programs at the test presets and `MID_NAPHTHALENE`; prove the inventory of captured targets; and check end to end (E-1 to E-4) that the block encoding encodes H̃_SA/Λ̃ − I (D-2).

**Architecture:** `src/dfthc/ops/walk.py` holds the four `Operator2` classes. Each takes `WalkLayout`'s groups and the compilable `params = WalkParams.as_compilable()` and `tables = WalkTables.as_compilable()`; its pinned rule rebuilds both with `from_compilable`, splits the groups once with `WalkLayout.split_group`, and emits its children from one time-ordered list (`_children` for the block encodings, `_walk_children` for the walks), each child on the registers its `wire_argnames` name and on the prefix of the shared `work` pool of exactly its need. The same list declares the resources, so declared and emitted children cannot differ. The controlled block encoding swaps `RefT2` for `ControlledRefT2`, the controlled walk adds `ControlledRefAB`; nothing is adjointed or controlled generically. A new test module, `dfthc.testing.walk_sim`, runs the classes on the tape path in the strict sparse simulator and reads their block; `dfthc.testing.resources.captured_targets` ports Task 0.9d's capture instrument for the inventory test.

**Tech Stack:** Python 3.12, PennyLane `Operator2` and Catalyst (`qp.qjit(capture=True)`, `qp.decompose`, `qp.analyze`), NumPy; the sparse simulator `dfthc.testing.sparse_sim` and `dfthc.testing.tape`; the encoded-Hamiltonian model `dfthc.data.encoded`; `dfthc.testing.watchdog.run_watched`; pytest with pytest-xdist; ruff.

This file holds Tasks 3.3 to 3.5 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 9). They run one after the other in wave 3 (master plan section 5), after Tasks 3.1 and 3.2 of `2026-11-11-dfthc-phase3-select.md` have merged; Tasks 3.3 and 3.4 both edit `src/dfthc/ops/walk.py` and `tests/test_work_pool.py`. The last section, "Interfaces for Phase 4", states what the QPE driver consumes.

**Code provenance.** Every code block below was executed on 2026-11-13 in a scratch replica of the repository (`.scratch/plan-check/repo-phase3a`, not committed), seeded from local `main` at `7777bc1` with `git archive`, plus the two files of Task 2E.3 from `phase2/2E.3` at `14866f3` and the four files of Tasks 3.1 and 3.2 from the select plan's replica `repo-phase3s`; all six are byte-identical to code blocks of the merged E and select plans (`stages/3A/check_select_blocks.py`, `select-blocks.log`), and this plan treats their interfaces as final. The replica is not a git repository (the sandbox forbids creating one), so the dry run ran pytest the way `scripts/test.sh` does (the same environment, 2 workers, `taskpolicy -c utility`, `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` exported) through `.scratch/plan-check/p3atest.sh`, which runs every session under the watchdog with a 3 GB cap (3,000 MB in revision 2). The three tasks were replayed in order, each RED then GREEN, in one fresh copy of `main` with 2E.3's and the select plan's files (`stages/3A/run_stages_3a.sh`, log `stages/3A/3a-stages.log`); Task 3.3's versions of the two files Task 3.4 replaces are derived from the final ones by exact replacements (`make_stage_33.py`). Every "Expected" line below is that replay's output, and the replayed files are byte-identical to the replica's. Further evidence, each with its log under `stages/3A/`: the per-commit tier on the replica with all three tasks (`3a-per-commit-tier.log`); the `MID_NAPHTHALENE` compiles, each test alone under `run_watched` (`run_mid_watched_3a.py`, `3a-mid-compiles.log`); one input of every nightly and release-only E check, each alone under the watchdog (`time_e_checks.py`, `3a-e-timings.log`); a mutation check of the new tests (`mutate_3a.py`, `3a-mutants.log`); every docstring example run against its stated value and the nine spec 9.0 sections checked in order (`check_doc_examples_3a.py`, `doc-examples.log`); the captured equations of the walk programs and their FeMoco-54 projection (`.scratch/plan-check/probes/p3a/capture_walk.py`, `capture-walk.log`, `probes/p3a/bound_wires.py`, `capture-projection.log`); the interface script of the last section (`interfaces_probe_3a.py`, `interfaces-probe-3a.out`); Task 3.5's three RED steps on the stage copy, their patch blocks generated from exact replacements (`make_red_patches.py`, `red/*.sh`, run by `e_checks_red.sh`, `3a-e-red.log`); every new per-commit test in one one-worker session (`3a-per-test-evidence.log`, summed per test in `3a-per-test-evidence.summary`); the random work-pool test against a drifted inverse class (`work_pool_drift_probe.py`, `work-pool-drift-probe.log`); Task 3.4's Step 6 command (`3a-step6-mid.log`); and the document edits (`make_doc_edits_3a.py`, `doc-edits.md`). The mutation check ran on the final code; one docstring sentence of `walk.py` and one of `walk_sim.py` were corrected afterwards (no code change), and revision 2 changed one test of `tests/test_work_pool.py` (both versions; no code under test); the stage replay ran again after each, the docstring check after the first. Probes that sized the simulations and checked the design before the tests existed are under `.scratch/plan-check/probes/p3a/`, each run under the watchdog.

**Load.** At most 2 pytest workers ran at any time, one pytest session at a time (the mutation runs, the per-commit tier and the stage replay ran alone, in the background while only files were edited), except for one overlap in revision 2 (below), and nothing FeMoco-shaped was compiled: FeMoco-54 values are asserted on the closed forms, and `tests/test_work_pool.py` builds the FeMoco-54 rules' calls in tape mode without compiling. Every process that compiled, simulated or ran pytest ran under the watchdog (3 GB cap, 3,000 MB in revision 2, 4 GB for the `MID_NAPHTHALENE` subprocess compiles); the drivers that start such processes (the stage, timing, mutation and `MID_NAPHTHALENE` scripts) and the classical document and stage-derivation scripts ran directly and stayed light. The largest compile, the `MID_NAPHTHALENE` walk, peaked at 615 MB; the largest simulation, one `H2_631G_TEST` input, at 185 MB; the largest pytest session, the per-commit tier, at 1,949 MB. The full per-commit tier ran once. The per-commit load gate of about 1 GB and a few seconds is read per test: it bounds what one per-commit test may cost, not the per-commit process tree, which runs 2 workers and is held to its tier budget of 15 minutes and 4 GB (master plan 12.9; the per-commit tiers measured at the gates so far peaked at about 1.9 to 2.0 GB). Every new per-commit test of Tasks 3.3 and 3.4, 72 tests, ran in one session with one worker under the watchdog with `--durations=0`: `72 passed, 4 deselected in 98.93s` at a session peak of 485 MB, which bounds every test's peak; the longest test took 8.5 s, five more 5.9 to 6.7 s, the other 66 at most 4.2 s, none above 15 s (`3a-per-test-evidence.log`, `3a-per-test-evidence.summary`). The longest, `test_controlled_walk_after_a_walk_step` (about 12 s in the two-worker tier, where the workers share the machine), stays per commit: it is the only per-commit test that sees an uncontrolled `RefAB` semantically (section "Mutation evidence"). No load-limit event occurred in revision 1. In revision 2 one command started the stage replay twice (cause not determined), and the two copies overlapped for at most about two minutes, so up to 4 pytest workers ran in two sessions, each under its own 3,000 MB watchdog; neither fired, and the surviving copy's sessions peaked at most 1,612 MB. Its log was discarded; the replay and the RED script now take a lock directory, so a second copy exits at once, and the replay ran again alone.

**Revision 2** (2026-11-13, after the GPT-5.6 plan review `.superpowers/sdd/phase3a-plan-review-1.md`, Needs fixes, 0 Critical, 2 Important, 1 Minor). I-1 (the per-commit load gate) clarified, not redesigned, by the orchestrator's decision: the gate is per test, and the paragraph "Load", the section "Simulation cost and the tier budgets", risk 3 and Task 3.5's report step now say so, with the one-worker per-test evidence above; no test changes tier. I-2 (the random work-pool test omitted children) applied: `test_work_size_covers_every_child_at_random_shapes` takes every child from the rules' own lists (`_children`, plain and controlled, and from Task 3.4 also `_walk_children`; tables of zeros in each random shape's sizes), checks that each is given exactly its own class's `required_wires(...)["work"]`, `OuterUnprepare`, `InnerUnprepare` and `InnerSelectDagger` included, and that the pool covers it, and that every class of the rules was seen; a watched probe gives each of those three classes one more work wire than its forward form, and the test fails for each (`work_pool_drift_probe.py`; revision 1's random test never consulted them). M-1 (Task 3.5's RED step proved only E-1 and E-2) applied: Task 3.5 has three RED steps, each a narrow diff of `walk.py` applied with `git apply` and restored with `git apply -R` of the same diff instead of `git checkout`: the first round's select as its dagger (E-1 and E-2: `7 failed`), the controlled walk's `RefAB` left uncontrolled (E-3: `1 failed`, at its comparison after one walk step), and the outer unprepare moved before the inner one (E-4: `2 failed`, both schedules); each run selected only its E checks, with 2 workers under the watchdog, and each restored `walk.py` byte for byte (`e_checks_red.sh`, `3a-e-red.log`; the failure messages in `3a-e-red-reasons.log`); the later steps are renumbered. The test counts are unchanged (973, 1017, 1045, 1045 per commit; Task 3.5 adds 10 nightly and 32 release tests), and so are the master-plan and spec edits.

**Revision 3** (2026-11-14, after Task 3.1's reviews). Task 3.1 gained two per-commit tests (the adjoint and controlled compile test for `InnerSelectDagger`), so every per-commit count here is two higher: 975 on `main` with Tasks 3.1 and 3.2, 1019 after Task 3.3, 1047 after Tasks 3.4 and 3.5. The measured dry runs quoted (`1045 passed`) predate this.

**Revision 4** (2026-11-14, after Task 3.3's Opus review `.superpowers/sdd/task-3.3-review-opus.md`, Approved, 0 Critical, 0 Important, 4 Minor). `tests/ops/test_block_encoding.py` gained six per-commit tests: an `outer-keep` and a `tail-angle` constructor case for each class (the only tables the outer prepares' and the selects' checks reject alone), a control-width test for the controlled class, and `test_harness_sees_phases_outcomes_and_ancillas`, a self-test of `walk_sim`'s schedules, state comparison, projection and register report (an all-zero schedule, a comparison by modulus or an empty register report each blinded the T3 without failing it). The T3's norm assertion is removed: the strict simulator renormalizes the kept branch at every measurement, so it could not fail. `walk.py`'s "Paper and dfthcbe" section (both Task 3.3's and Task 3.4's block) cites arXiv v1 Eq. (11), Fig. 1, App. B.1 Eq. (87), B.2 Eq. (88) and Table 3, gives dfthcbe's call-graph split and the full -202 breakdown. Every per-commit count from Task 3.3 on is six higher: 1025 after Task 3.3, 1053 after Tasks 3.4 and 3.5; Task 3.3's targeted run is `52 passed` (50 per commit), Task 3.4's four-file run `82 passed`. The measured dry runs quoted predate this.

---

## Changes from the master plan, the spec and HANDOFF

The changes that touch a document are applied by the exact edits listed after this list (ids A-M for the master plan, A-S for the spec; HANDOFF needs none). No template was edited; every A-M old text also occurs exactly once in `.scratch/plan-check/templates/2026-11-11-dfthc-master-plan.md`, so the orchestrator mirrors A-M1 to A-M9 there when it applies them.

1. **Plan file and provenance** (A-M1, A-M2). Section 1 lists this file and its dry run.
2. **A shared simulation harness, `dfthc.testing.walk_sim`** (A-M4, A-S1). The three tasks need the same tape-path machinery: decompose a program, start from system amplitudes with every ancilla in |0⟩, run strictly under a forced schedule, read the block (ancillas |0⟩, the phase gradient contracted), compare whole states. Task 3.3 creates it with the Phase 3 test shapes (`CASES`, `FAILS_C3`), so Tasks 3.4 and 3.5 only consume it. Spec 5.2 lists it under `testing/`.
3. **`FAILS_C3`, a shape the test presets cannot be** (A-M4, A-M7, A-M9, A-S4, A-S6). Both H2 presets have `X_o` a power of two, `C <= 2` and the last inner block covering `2**w_b`, so E-1 there sees neither the failed uniform preparation of x_o (ledger row 1A.5) nor a rank that is no bit field of x_o nor dfthcbe's differing failure signs (ledger row 2D.2). `FAILS_C3` (N = 2, R = C = 3, B = 2, k2 = 0, b_k1 = 1, b_k2 = 2; 59 wires with the control, so one key word) has all three; it runs the T3 tests per commit and E-1 and E-2 nightly, always against our `encoded_block`. The block equals the model there to 1e-15.
4. **Interfaces, made precise** (A-M4, A-S2, A-S3). The assembly classes' `required_wires(params, tables)` is the layout's group sizes (`work` = `WalkLayout.work_size`), so a caller passes `WalkLayout`'s groups as they are; the constructor checks, besides the children's domains, that the pool covers every child of the walk (a `ValueError` names the child otherwise). The master plan's "every operator call ... receives exactly `required_wires(...)`" holds for every register whose size `required_wires` fixes; the four prepare classes' `phase_grad` is a minimum there (`at_least`), and they get the walk's whole `phase_grad` register, as master plan Task 2D.1 and ledger row 2D.2 prescribe (they use its top `s_u` wires).
5. **Files** (A-M4, A-M5, A-M6, A-M9). Task 3.3 also creates `src/dfthc/testing/walk_sim.py` (change 2). Task 3.4 also changes `src/dfthc/testing/resources.py` (`captured_targets` and the inventory helpers `LEAF_ADJOINTS`, `TEMPLATE_ADJOINTS`, `modified_targets`, `inventory`, so that Task 4.2 can reuse them for the QPE program) and replaces `tests/test_work_pool.py` (the walks' calls added). Task 3.5 has no `tests/integration/conftest.py`: its schedules and harness are `walk_sim`'s, and its inputs are three constants of the test file.
6. **More tests than the master plan lists** (A-M4, A-M7, A-M8). The common paragraph asks for T2 to T6 for every class, the controlled ones included, with T3 for a controlled class in three control states. Per commit: T3 of all four classes on the tape path (the block against the model, the walk as `RefAB` after the block encoding, the controlled classes in |0⟩, |1⟩ and |+⟩), the E-4 precursor under two schedules, the children's order and registers, the failing `qp.adjoint`/`qp.ctrl` of the block encoding, and a negative control of the inventory.
7. **The controlled walk is also checked on the walk's own states** (A-M7, A-M9, A-S5). The mutation check found that an uncontrolled `RefAB` inside `ControlledDFTHCWalk` passes every check that starts with the ancillas in |0⟩: there the control-0 branch keeps them |0⟩, which `RefAB` fixes. Iterative QPE does not start a round there (the walk register carries the qubitized state between rounds), so Task 3.4's T3 and the nightly E-3 also apply the controlled walk with the control in |+⟩ after one uncontrolled walk step and compare with (|0⟩W|ψ⟩ + |1⟩W²|ψ⟩)/√2. The counts and the inventory caught the mutant too, but only these checks see its semantics.
8. **E-4's schedules** (A-M9, A-S6). Spec 12.6 gives none; E-4 runs under the alternating and a seeded schedule, because the all-zeros schedule fires no fix-up and the all-ones one sees only each address's mask parity (ledger row 2D.2).
9. **`MID_NAPHTHALENE` walk compiles go through `run_watched` in the test itself** (master plan Task 3.4: "through `run_watched`"): `tests/integration/test_walk_counts.py` compiles each walk program in a subprocess with a 4 GB cap and prints the watched time and peak on a `MID_COMPILE` line, so every nightly run records them.
10. **Master plan section 3** (A-M3): a new item 23 with this dry run's compile, capture, inventory and simulation findings.
11. **Not added:** no `C(DFTHCBlockEncoding)`, `C(DFTHCWalk)`, `Adjoint(...)` or delegating rule (spec 7.4 rule 6); `test_no_adjoint_or_controlled_path` (Task 3.3) and `test_controlled_walk_has_no_generic_controlled_rule` (Task 3.4) pin the loud failures. No change to `dfthc.costs`: the block encoding's A is T − (`toffoli_only(p)` − 1), the walk's `toffoli_only` less `RefAB`'s one plain Toffoli, which the tests state.

Exact edits (master plan and spec, against local `main` at `7777bc1`):

Each edit replaces one exact text of the file on local `main` at `7777bc1`, where it occurs exactly once (checked by `.scratch/plan-check/stages/3A/make_doc_edits_3a.py`, which also applies them in order); every master-plan old text also occurs once in `.scratch/plan-check/templates/2026-11-11-dfthc-master-plan.md`, where the orchestrator mirrors the edits.

**A-M1** (2026-11-11-dfthc-master-plan.md, section 1, the file table). Old:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase3-select.md` | Tasks 3.1, 3.2 |
````

New:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase3-select.md` | Tasks 3.1, 3.2 |
| `docs/superpowers/plans/2026-11-11-dfthc-phase3-assembly.md` | Tasks 3.3 to 3.5 |
````

**A-M2** (2026-11-11-dfthc-master-plan.md, section 1, code provenance). Old:

````text
against `main` at `b34ed56` with Task 2E.3's two files (`14866f3`); its preamble lists its changes to this plan.
````

New:

````text
against `main` at `b34ed56` with Task 2E.3's two files (`14866f3`); its preamble lists its changes to this plan. The Phase 3 assembly file (`phase3-assembly`, Tasks 3.3 to 3.5) was executed the same way on 2026-11-13, against `main` at `7777bc1` with Task 2E.3's two files and the select file's code; its preamble lists its changes to this plan.
````

**A-M3** (2026-11-11-dfthc-master-plan.md, section 3, a new item 23 after item 22 (the end of item 22 is shown)). Old:

````text
The select and the reflections take their registers by name (spec 6.7, 9.18 to 9.20).
````

New:

````text
The select and the reflections take their registers by name (spec 6.7, 9.18 to 9.20).
23. **The block encoding and the walk as built** (Phase 3 assembly dry run, 2026-11-13). The four assembly classes take `WalkLayout`'s groups (`required_wires(params, tables)` is their sizes), split them once and give each child its registers by name and a prefix of the work pool. They compile to `costs.block_encoding` and `costs.walk` exactly at `H2_STO3G_TEST`, `H2_631G_TEST` and `MID_NAPHTHALENE` (walk T = 158, 240, 1,913, controlled + 2; M = 50, 68, 502; A = T − `toffoli_only`), at the test presets in 2 to 4 s below 450 MB. At `MID_NAPHTHALENE` the walk compiles in 7.8 s at 615 MB and the controlled walk in 8.1 s at 588 MB (Task 0.9's stand-in at `mid10`: 77.8 s at 1,977 MB and 155.4 s at 3,571 MB; Task 0.9d's PX arm: 11.2 s at 802 MB and 11.3 s at 806 MB), capturing 58,663 and 58,310 equations (PX arm 94,224 and 94,337). The assembly targets cost 1.66 (block encoding) and 2.71 (walk) captured equations per wire their rules bind; projected to FeMoco-54, about 7,100 and 4,500 equations against the PX stand-in's 62,832 and 8,230, about 59,500 fewer for the controlled walk (15% of its 390,949). The inventory (`captured_targets`): the only modified dfthc targets of the walk programs are the leaf adjoints of `FlaggedUniformPrep`, `PhaseGradientRotation`, `GivensRotation` and `MajoranaGate`, no `C(...)` target is captured at all, and `qp.ctrl(DFTHCWalk(...))` fails with a `CompileError` naming `C(DFTHCWalk)`. After the block encoding and after every walk step only the six index registers of 𝒜 ∪ ℬ differ from |0⟩. On the sparse simulator an `H2_STO3G_TEST` input costs 0.35 to 0.45 s per block encoding or walk (64 wires with a control, one key word) and an `H2_631G_TEST` input 35 to 47 s (80 wires, two key words); the block equals `encoded_block` to 1e-15 and E-2 holds at k = 2, also on `FAILS_C3`, where the uniform preparation of x_o fails and dfthcbe's failure signs differ from ours. An uncontrolled `RefAB` inside the controlled walk passes every check that starts with the ancillas in |0⟩; the walk's own states (after one uncontrolled step) detect it.
````

**A-M4** (2026-11-11-dfthc-master-plan.md, Task 3.3, files, interfaces and acceptance tests). Old:

````text
- **Files:** `src/dfthc/ops/walk.py` (block encoding), `tests/ops/test_block_encoding.py`, `tests/test_work_pool.py`.
- **Interfaces produced:** `DFTHCBlockEncoding(params, tables, outer, inner, select, system, phase_grad, work)`, with compilable `params = WalkParams.as_compilable()` and `tables = WalkTables.as_compilable()`; the rules rebuild both with `from_compilable` and split the wire groups with `WalkLayout.split_group`. `ControlledDFTHCBlockEncoding(params, tables, control, outer, inner, select, system, phase_grad, work)` controls only the T2 reflection, through `ControlledRefT2`. Both rules call `OuterUnprepare`, `InnerUnprepare` and `InnerSelectDagger`, and pass each child only the registers it needs (spec 9.0).
- **Acceptance tests:**
  - T4: the closed forms give T = 9,749 (controlled +1) and M = 2,524 at FeMoco, asserted without compiling.
  - The compiled counts equal the closed forms at the test presets (per commit) and at `MID_NAPHTHALENE` (nightly).
  - `test_work_pool`: every operator call in the walk receives exactly `required_wires(...)`, and `max(work)` is at most `len(layout.work)`.
  - The prepare/unprepare part (E-4 precursor) passes on the sparse simulator.
````

New:

````text
- **Files:** `src/dfthc/ops/walk.py` (block encoding), `src/dfthc/testing/walk_sim.py` (the sparse-simulator harness of the assembly operators and the Phase 3 test shapes, `FAILS_C3` included), `tests/ops/test_block_encoding.py`, `tests/test_work_pool.py`.
- **Interfaces produced:** `DFTHCBlockEncoding(params, tables, outer, inner, select, system, phase_grad, work)`, with compilable `params = WalkParams.as_compilable()` and `tables = WalkTables.as_compilable()`; the rules rebuild both with `from_compilable` and split the wire groups with `WalkLayout.split_group`. Static `required_wires(params, tables)`: the layout's group sizes (`work` is `WalkLayout.work_size`, which the constructor checks covers every child of the walk). `ControlledDFTHCBlockEncoding(params, tables, control, outer, inner, select, system, phase_grad, work)` controls only the T2 reflection, through `ControlledRefT2`. Both rules call `OuterUnprepare`, `InnerUnprepare` and `InnerSelectDagger`, and pass each child exactly the registers its `wire_argnames` name (spec 9.0) and the prefix of `work` of its need; the prepare classes get the whole `phase_grad`. `dfthc.testing.walk_sim`: `WalkCase`, `case(name)`, `program`, `prepare_unprepare`, `input_state`, `simulate`, `schedule`, `project`, `assert_same_state`, `reflect_ab`, `set_control`, `superpose`.
- **Acceptance tests:**
  - T4: the closed forms give T = 9,749 (controlled +1), M = 2,524 and A = 8,632 (controlled 8,633) at FeMoco, asserted without compiling.
  - The compiled counts, T, M and A = T − (`toffoli_only` − 1), equal the closed forms at the test presets and `FAILS_C3` (per commit) and at `MID_NAPHTHALENE` (nightly).
  - T3 on the tape path (strict sparse simulator) at `H2_STO3G_TEST` and `FAILS_C3` (the failed uniform preparation of x_o, dfthcbe's differing failure signs, C = 3): the block equals `encoded_block`'s H̃_SA/Λ̃ − I and outside it only the index registers of 𝒜 ∪ ℬ are nonzero; the controlled class with the control in |0⟩, |1⟩ and |+⟩.
  - `test_work_pool`: every operator call in the block encodings receives exactly `required_wires(...)` (the prepares' `phase_grad` is a minimum: they get the whole register), its `work` is the pool's prefix, disjoint from its other registers and from the dirty comparator work registers, and `max(work)` is at most `len(layout.work)`, at the test presets, `MID_NAPHTHALENE` and FeMoco-54 (tape mode, nothing compiled); `WalkLayout.work_size` covers every child at 300 random shapes.
  - The prepare/unprepare part (E-4 precursor) passes on the sparse simulator under the alternating and a seeded schedule.
````

**A-M5** (2026-11-11-dfthc-master-plan.md, Task 3.4, files). Old:

````text
- **Files:** `src/dfthc/ops/walk.py` (walk), `tests/ops/test_walk.py`, `tests/integration/test_walk_counts.py` (the `MID_NAPHTHALENE` cases marked `slow`).
````

New:

````text
- **Files:** `src/dfthc/ops/walk.py` (walk), `src/dfthc/testing/resources.py` (`captured_targets` and the inventory helpers), `tests/ops/test_walk.py`, `tests/integration/test_walk_counts.py` (the `MID_NAPHTHALENE` cases marked `slow`), `tests/test_work_pool.py` (the walks' calls added).
````

**A-M6** (2026-11-11-dfthc-master-plan.md, Task 3.4, interfaces (the end of the `captured_targets` sentence)). Old:

````text
a port of Task 0.9d's capture instrument (it wraps a private frontend function, so it asserts that it saw the program's own top-level target).
````

New:

````text
a port of Task 0.9d's capture instrument (it wraps a private frontend function, so its callers assert that it saw the program's own top-level target), with `LEAF_ADJOINTS`, `TEMPLATE_ADJOINTS` and `inventory(names)`, the modified targets the inventory test compares (Task 4.2 reuses them for the QPE program).
````

**A-M7** (2026-11-11-dfthc-master-plan.md, Task 3.4, acceptance tests: a T3 item after the FeMoco item). Old:

````text
  - At FeMoco-54 the closed forms give T = 9,793 and 9,795, M = 2,524, A = 8,675 and 8,677, asserted without compiling. The FeMoco compile is Task 4.5's (D-3).
````

New:

````text
  - At FeMoco-54 the closed forms give T = 9,793 and 9,795, M = 2,524, A = 8,675 and 8,677, asserted without compiling. The FeMoco compile is Task 4.5's (D-3).
  - T3 on the tape path at `H2_STO3G_TEST` and `FAILS_C3`: the walk's whole output is `RefAB`'s exact action on `DFTHCBlockEncoding`'s; the controlled walk with the control in |0⟩, |1⟩ and |+⟩, and with the control in |+⟩ after one uncontrolled walk step (the index registers of 𝒜 ∪ ℬ no longer |0⟩, as between QPE rounds).
````

**A-M8** (2026-11-11-dfthc-master-plan.md, Task 3.4, acceptance tests: the inventory item (its end)). Old:

````text
capture no `Adjoint(...)` or `C(...)` target whose base is a dfthc composite or `QROAMRead`; the only modified dfthc targets are the leaf adjoints of spec 7.4 rule 6.
````

New:

````text
capture no `Adjoint(...)` or `C(...)` target whose base is a dfthc composite or `QROAMRead`; the only modified dfthc targets are the leaf adjoints of spec 7.4 rule 6 (`Adjoint(FlaggedUniformPrep)`, `Adjoint(GivensRotation)`, `Adjoint(MajoranaGate)`, `Adjoint(PhaseGradientRotation)`), and the templates they and the unprepares adjoint give `Adjoint(SemiAdder)`, `Adjoint(LeftQuantumComparator)` and `Adjoint(LeftClassicalComparator)`. A stray `qp.adjoint` of a dfthc operator shows in the inventory (negative control).
````

**A-M9** (2026-11-11-dfthc-master-plan.md, Task 3.5, files, interfaces and E-1 to E-4). Old:

````text
- **Files:** `tests/integration/test_end_to_end.py`, `tests/integration/conftest.py` (schedules and inputs).
- **Interfaces:** consumes `encoded_block`, the sparse simulator, `decompose_for_simulation` and the walk.
- **Acceptance tests (spec 12.4, schedules per spec 12.6):**
  - **E-1:** ⟨0|BE|0⟩ = H̃_SA/Λ̃ − I to 1e-10 (16 inputs under S0 and the 4 fixed inputs under S1, nightly; all 16 under S1, and one seeded schedule on the 4 fixed inputs, in the release tier; `H2_631G_TEST`'s 28 two-electron inputs, S0, in the release tier).
  - **E-2:** k = 1, 2, against the Chebyshev polynomials.
  - **E-3:** the three control states, with |+⟩ detecting global-phase errors.
  - **E-4:** U†U on all 16 inputs.
````

New:

````text
- **Files:** `tests/integration/test_end_to_end.py` (the schedules and the simulation harness are `dfthc.testing.walk_sim`'s, from Task 3.3, so there is no `conftest.py`).
- **Interfaces:** consumes `encoded_block`, `dfthc.testing.walk_sim` (the sparse simulator and `decompose_for_simulation` through it) and the walk classes.
- **Acceptance tests (spec 12.4, schedules per spec 12.6):**
  - **E-1:** ⟨0|BE|0⟩ = H̃_SA/Λ̃ − I to 1e-10 (16 inputs under S0 and the 4 fixed inputs under S1, nightly, and `FAILS_C3`'s 16 inputs under S0, nightly; all 16 under S1, and one seeded schedule on the 4 fixed inputs, in the release tier; `H2_631G_TEST`'s 28 two-electron inputs, S0, in the release tier, one test each).
  - **E-2:** k = 1, 2, against the Chebyshev polynomials, under S0 on the 4 fixed inputs of `H2_STO3G_TEST` and of `FAILS_C3` (nightly).
  - **E-3:** the three control states, with |+⟩ detecting global-phase errors, and the control in |+⟩ after one uncontrolled walk step (nightly: S0 on the 4 fixed inputs; release: S1 on all 16 and a seeded schedule on the 4 fixed).
  - **E-4:** U†U on all 16 inputs, under the alternating and a seeded schedule (nightly).
````

**A-S1** (2026-11-11-dfthc-pennylane-design.md, 5.2, the testing modules). Old:

````text
      tape.py                 tape-path decomposition for simulation
````

New:

````text
      tape.py                 tape-path decomposition for simulation
      walk_sim.py             sparse-simulator harness of the block encoding and walk, Phase 3 test shapes (§12.4)
````

**A-S2** (2026-11-11-dfthc-pennylane-design.md, 9.21, Wires). Old:

````text
| Wires | Grouped: `outer`, `inner`, `select`, `system`, `phase_grad`, `work` (§10) |
````

New:

````text
| Wires | Grouped: `outer`, `inner`, `select`, `system`, `phase_grad`, `work`, exactly `WalkLayout`'s groups (§10; `required_wires(params, tables)` gives their sizes). The rule splits them once with `WalkLayout.split_group` and gives each child the registers its §9 table names and the prefix of `work` of its need; the prepares take the whole `phase_grad` |
````

**A-S3** (2026-11-11-dfthc-pennylane-design.md, 9.21, Compilable). Old:

````text
| Compilable | `WalkParams` fields and `WalkTables.as_compilable()` |
````

New:

````text
| Compilable | `params` = `WalkParams.as_compilable()` and `tables` = `WalkTables.as_compilable()`, rebuilt in the rules with `from_compilable`; the constructor checks them against the children's domains and that `WalkLayout.work_size` covers every child of the walk |
````

**A-S4** (2026-11-11-dfthc-pennylane-design.md, 12.4, the paragraph before the table). Old:

````text
All checks run on `H2_STO3G_TEST` with all ancillas in $\lvert 0\rangle$. Which inputs and schedules each tier uses is set in §12.6. `H2_631G_TEST` runs E-1 in the release tier, on its 28 two-electron inputs.
````

New:

````text
All checks run on `H2_STO3G_TEST` with all ancillas in $\lvert 0\rangle$. Which inputs and schedules each tier uses is set in §12.6. `H2_631G_TEST` runs E-1 in the release tier, on its 28 two-electron inputs. E-1 and E-2 also run on the shape `FAILS_C3` of `dfthc.testing.walk_sim` ($X_o = 11$ and $B + 1 = 3$ both fail their uniform preparation, $k_2 = 0$ makes dfthcbe's failure signs differ from ours, $C = 3$), which the test presets cannot show; there too the oracle is our `encoded_block`.
````

**A-S5** (2026-11-11-dfthc-pennylane-design.md, 12.4, the E-3 row (its end)). Old:

````text
The last case detects global-phase errors in the controlled reflections. |
````

New:

````text
The last case detects global-phase errors in the controlled reflections. After one uncontrolled step $W$, control $\lvert +\rangle$ gives $(\lvert 0\rangle W\lvert\psi\rangle + \lvert 1\rangle W^2\lvert\psi\rangle)/\sqrt2$; there the index registers of $\mathcal A\cup\mathcal B$ are not $\lvert 0\rangle$, as between QPE rounds, so an uncontrolled $\mathrm{REF}_{\mathcal A\cup\mathcal B}$ fails it. |
````

**A-S6** (2026-11-11-dfthc-pennylane-design.md, 12.6, the nightly row (its E-4 item)). Old:

````text
E-4 on all 16 inputs;
````

New:

````text
E-4 on all 16 inputs under the alternating and a seeded schedule; E-1 under S0 on all 16 inputs and E-2 under S0 on the 4 fixed inputs of `FAILS_C3`;
````

## Shared edits

- **`src/dfthc/testing/resources.py`** (Task 3.4): one import line widened by `compiled_resources`, and `captured_targets`, `LEAF_ADJOINTS`, `TEMPLATE_ADJOINTS`, `modified_targets` and `inventory` appended after `chosen_rules`; nothing else changes. Its private Catalyst module is imported inside `captured_targets`, so a renamed module fails only the inventory tests, not the conftest import every test needs. The exact diff against `main`:

```text
--- main:src/dfthc/testing/resources.py
+++ Task 3.4:src/dfthc/testing/resources.py
@@ -13,7 +13,13 @@
 from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
 from catalyst.passes import graph_decomposition
 
-from dfthc.compile import FIXED_DECOMPS, GATE_SET, CompiledResources, dfthc_operator_names
+from dfthc.compile import (
+    FIXED_DECOMPS,
+    GATE_SET,
+    CompiledResources,
+    compiled_resources,
+    dfthc_operator_names,
+)
 
 _EXCLUDED_RULE = (
     r"(?:_(?:\w*_)?ppm|_toffoli_elbow|_cphase_to_single_phase|\w*_workers?\w*"
@@ -109,6 +115,66 @@
     return [line.split(prefix, 1)[1].strip() for line in listing.splitlines() if prefix in line]
 
 
+def captured_targets(qfunc: Callable[[], None], num_wires: int) -> list[str]:
+    """The rule targets the Catalyst frontend converts while :func:`compiled_resources` compiles
+    ``qfunc`` once, one name per converted target, in order: the operator name with its modifiers
+    (``Adjoint(MajoranaGate)``, ``C(RefT2)``), without the attribute dump that starts at the first
+    ``{`` (spec 12.1, T6: the inventory test).
+
+    A port of Task 0.9d's capture instrument (``spikes/k1_whole_program.capture_instrument``): it
+    wraps Catalyst's private ``catalyst.from_plxpr.qfunc_interpreter._convert_decomp_target_spec``
+    for this one compile and restores it afterwards. The frontend converts every target it builds
+    rules for, chosen or not, so the list holds every distributed ``Adjoint``/``C`` candidate the
+    compile captured. The hook is private, which makes this a compatibility canary at every
+    nightly bump: a renamed hook raises ``AttributeError``, and callers assert that the program's
+    own top-level target is in the list, so a hook that no longer sees targets fails instead of
+    passing. The private module is imported here, so a renamed module fails only the callers.
+    """
+    import catalyst.from_plxpr.qfunc_interpreter as qfunc_interpreter
+
+    convert = qfunc_interpreter._convert_decomp_target_spec
+    names: list[str] = []
+
+    def recording(interpreter, target_spec):
+        names.append(str(target_spec.target_id).split("{", 1)[0])
+        return convert(interpreter, target_spec)
+
+    qfunc_interpreter._convert_decomp_target_spec = recording
+    try:
+        compiled_resources(qfunc, num_wires)
+    finally:
+        qfunc_interpreter._convert_decomp_target_spec = convert
+    return names
+
+
+LEAF_ADJOINTS = frozenset({"Adjoint(FlaggedUniformPrep)", "Adjoint(GivensRotation)",
+                           "Adjoint(MajoranaGate)", "Adjoint(PhaseGradientRotation)"})
+"""The only modified dfthc targets a walk program may capture: the auto-derived leaf adjoints of
+spec 7.4 rule 6 (the uniform preparation's and, inside it, its rotations'; the Givens rung's; the
+Majorana gate's)."""
+
+TEMPLATE_ADJOINTS = frozenset({"Adjoint(SemiAdder)", "Adjoint(LeftQuantumComparator)",
+                               "Adjoint(LeftClassicalComparator)"})
+"""The PennyLane templates those leaves and the unprepares adjoint: the rotations' and Givens
+rungs' adders and the comparators (spec 7.4 rule 4, K-7)."""
+
+
+def modified_targets(names: Iterable[str], bases: Iterable[str]) -> set[str]:
+    """The ``Adjoint(...)`` and ``C(...)`` targets of ``names`` (any nesting, ``2C(...)``
+    included) whose operator is one of ``bases``."""
+    bases = set(bases)
+    return {name for name in names if "(" in name and dfthc_operator_names([name]) & bases}
+
+
+def inventory(names: Iterable[str]) -> set[str]:
+    """Spec 12.1 T6: the modified targets of ``names`` (from :func:`captured_targets`) whose
+    operator is a dfthc operator (every name with a pinned rule: the composites, ``QROAMRead`` and
+    the leaves) or a template of :data:`TEMPLATE_ADJOINTS`. For a walk, controlled-walk or QPE
+    program it must equal ``LEAF_ADJOINTS | TEMPLATE_ADJOINTS``: no modified composite or read."""
+    templates = {name[len("Adjoint("):-1] for name in TEMPLATE_ADJOINTS}
+    return modified_targets(names, dfthc_operator_names() | templates)
+
+
 def assert_resources(
     resources: CompiledResources, *, toffoli: int, measurements: int,
     measured_and_uncomputes: int | None = None,
```

- **`src/dfthc/testing/walk_sim.py`** is a new module (Task 3.3), not an edit.
- No other merged module changes. `dfthc.costs` already has `block_encoding`, `walk`, `toffoli_only` and `measured_and_uncomputes` (spec 9.24 rows); `WalkLayout.work_size`, called provisional in `registers.py`, covers every child at every preset and at 300 random shapes (`test_work_pool.py`; 3,000 in the probe `probes/p3a/work_sweep.py`), so it stays; `split_group` takes lists as well as arrays. `walk.py` reuses the merged validators `prepare._check_outer`, `prepare._check_inner` and `select._check`, as `select.py` reuses the loads' (a public validator is a Task 5.1 cleanup).

## How this plan meets the binding ledger rows

| Ledger row or decision | What this plan does |
| --- | --- |
| Decision "K-1 at G0" (PX: explicit classes, per-leaf loads) | `ControlledDFTHCBlockEncoding` and `ControlledDFTHCWalk` are classes with pinned base rules that control only `RefT2` and `RefAB` through `ControlledRefT2` and `ControlledRefAB`; the only adjoints in the new rules are the explicit classes `InnerUnprepare`, `InnerSelectDagger`, `OuterUnprepare`. The inventory test (Task 3.4) shows the walk programs capture no modified composite and no `C(...)` target at all; `qp.ctrl(DFTHCWalk(...))` fails with a `CompileError` naming `C(DFTHCWalk)` after only "expected" and "noise" warnings (the Task G0.1 probe, now a test). At `MID_NAPHTHALENE` the walk compiles in 7.8 s at 615 MB, the controlled walk in 8.1 s at 588 MB, against Task 0.9d's PX arm at `mid10` (802 and 806 MB). |
| Decision "Phase 3 plans" (2026-11-13: named registers for the select and reflections; table-free T4 exception) | The assembly classes split their groups once with `WalkLayout.split_group` and pass every child its registers by name (`_registers`, `_emit`); `test_children_order_and_registers` and `test_work_pool` assert it. Nothing table-bearing compiles at FeMoco-54; the walks compile at the test presets per commit and at `MID_NAPHTHALENE` nightly. |
| 2D.2 notes 2 for Phase 3: `InnerSigns`' position; dirty `succ_work_b` and `cmp_work_b` never in the select's `work` prefix; the prepares' `work` prefix; the whole `phase_grad`; failed `b` | `InnerSigns` follows `InnerPrepare` in round 1 and `InnerSelectDagger` in round 2 (spec 9.21; the children test fails both moved-signs mutants). `test_work_pool` checks for every call of the four rules, at the test presets, `MID_NAPHTHALENE` and FeMoco-54: `work` is the pool's prefix of exactly the need and disjoint from every other register of the call; no other register holds a pool wire; `succ_work_*` and `cmp_work_*` go only to their own prepare and unprepare classes under their own names. The prepare classes get the whole `phase_grad`. Failed `b` on SF rows is in every T3 and E check at both shapes (B + 1 = 3 fails). |
| 2D.2 M-1: an all-ones schedule sees only each address's mask parity | Every per-commit T3 and the E-4 precursor run under the alternating and a seeded schedule; E-4 nightly likewise; E-1 and E-3 use S0, S1 and a seeded schedule as spec 12.6 places them. |
| 2D.2 notes 3 for Task 3.5: compare against our `encoded_block`, not dfthcbe's failure signs | Every block comparison uses `WalkCase.block()`, `encoded_block`'s H̃_SA/Λ̃ − I; `FAILS_C3` (dfthcbe's failure signs [−1, 1, −1, −1, −1, −1, −1, −1, −0.5] there, ours +1) passes E-1 against it to 1e-15. |
| 1A.5: the x_o failure is untested end to end | `FAILS_C3` (X_o = 11, P_fail = 0.11) runs the T3 tests per commit and E-1 and E-2 nightly; there REF_T2 is −I and Λ̃ = Λ/(1 − P_fail). |
| 1A.5 Minor 3: take Λ̃ from `encoded_block` | The E checks and T3 tests take both H̃_SA and Λ̃ from `encoded_block` (`WalkCase.block`). |
| 1A.5: E-1 cannot see the Givens order or a tail-direction error | Unchanged: both E shapes have N = 2 (one rung); Task 3.1's select tests carry those checks at N = 3 and 4. |
| 2E.4 and 2E.5: the inventory's expected modified keys, `Adjoint(MajoranaGate)` listed with `_majorana_gate` | `test_inventory_of_captured_targets` asserts the modified targets of dfthc operators and of the templates they adjoint are exactly `Adjoint(FlaggedUniformPrep)`, `Adjoint(GivensRotation)`, `Adjoint(MajoranaGate)`, `Adjoint(PhaseGradientRotation)`, `Adjoint(SemiAdder)`, `Adjoint(LeftQuantumComparator)` and `Adjoint(LeftClassicalComparator)`; `test_pinned_rules_are_chosen` (Task 3.4) reads the listing operator by operator and finds `Adjoint(MajoranaGate)` lowered with a copy of `_majorana_gate` (`_majorana_gate_28` today). |
| 2E.5 test-design lesson: a tape-path sparse-simulator check for every operator with hand-written AND pairs | The assembly rules have no AND pairs of their own; every T3 and E check runs the decomposed classes on the tape path in strict mode, whose handlers assert every child's AND preconditions. |
| 1B.1 Minor (`np.unique(axis=0)` and two-word keys) and 1B.3's note (keep the 6-31G E-1 in the release tier until Task 3.5 measures the real walk) | Measured on the real walk: one `H2_STO3G_TEST` input 0.35 to 0.45 s (64 wires, one word), one `H2_631G_TEST` input 35 to 47 s (80 wires, two words). The 6-31G E-1 stays in the release tier, one test per input; no spec 12.6 mitigation is needed (projections below). |
| Decision "Reduced release tier" (2026-11-11) | E-1 to E-4 are placed exactly as spec 12.6 says, plus the `FAILS_C3` additions (change 3) and E-3 after a walk step (change 7); release-only checks are timed, not run, and projected. |
| Task 0.9d's PX figures at `mid10` (walk 802 MB, controlled walk 806 MB) | Compared in "Compile cost and captured equations". |
| 2E.2: the SF fix-up refinement ob = `w_b` − 1 (to decide at Task 4.3) | Unchanged: `_children` passes `fixup_bits = w_b`, which `costs.block_encoding` assumes; the refinement would be that one argument. |
| 2D.1 and 2D.2: retaining the swaps' ANDs (deferred, R-7) | Not adopted; no construction is tuned. |
| 2E.4 (project-wide): no operator checks wire disjointness | `test_work_pool` checks the pool's disjointness for every call of the assembly rules; the constructors still do not (Task 5.1). |
| Heavy-runs log: the select writer's two breaches | Every process of this dry run that compiled, simulated or ran pytest ran under the watchdog with a cap, one pytest session at a time; no load-limit event (see "Load"). |

## FeMoco-54 numbers, compiled counts, dfthcbe and the paper

`dfthc.costs` on `main` gives exactly the master plan's numbers, and the four rules emit exactly the closed forms at every compiled size (T / M / A; A of the block encodings is T − (`toffoli_only` − 1)):

| Class | `H2_STO3G_TEST` | `H2_631G_TEST` | `FAILS_C3` | `MID_NAPHTHALENE` | FeMoco-54 (closed form) |
| --- | --- | --- | --- | --- | --- |
| `DFTHCBlockEncoding` | 150 / 50 / 101 | 231 / 68 / 171 | 276 / 35 / 241 | 1,874 / 502 / 1,640 | 9,749 / 2,524 / 8,632 |
| `ControlledDFTHCBlockEncoding` | 151 / 50 / 102 | 232 / 68 / 172 | 277 / 35 / 242 | 1,875 / 502 / 1,641 | 9,750 / 2,524 / 8,633 |
| `DFTHCWalk` | 158 / 50 / 108 | 240 / 68 / 179 | 285 / 35 / 249 | 1,913 / 502 / 1,678 | 9,793 / 2,524 / 8,675 |
| `ControlledDFTHCWalk` | 160 / 50 / 110 | 242 / 68 / 181 | 287 / 35 / 251 | 1,915 / 502 / 1,680 | 9,795 / 2,524 / 8,677 |

At FeMoco-54: T_BE = 279 + 2 · 4,725 + 20 = 9,749 (outer pair 210 + 69; inner round 971 + 3,537 + 217; RefT2 20), M = 96 + 2 · (416 + 798) = 2,524, T_Toffoli = 18 + 20 + 72 + 780 + 216 + 12 = 1,118, so A = 9,793 − 1,118 = 8,675 and M + A = 11,199 per walk (controlled 9,795, 8,677, 11,201). The layout has 1,564 wires with a 44-wire pool, `ControlledRefAB`'s need. The 2E.3 and select tasks' figures enter unchanged (select 3,537 / 798; `RefT2` 20 / 21; `RefAB` 44 / 45).

Against dfthcbe (`tests/fixtures/dfthcbe/femoco_block_toffolis.json`, its symbolic call graph at FeMoco-54) and the paper's Table III (spec 13.1):

| Program | Ours | dfthcbe | Paper | Ours − dfthcbe | Ours − paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| Block encoding | 9,749 | 9,596 | 9,951 ¹ | +153 | −202 |
| Controlled block encoding | 9,750 | 9,597 | | +153 | |
| Walk | 9,793 | 9,641 | 9,997 | +152 | −204 |
| Controlled walk | 9,795 | 9,642 | | +153 | |

¹ Spec 13.1's paper column without its REF_𝒜∪ℬ row (46); Table III states the walk. dfthcbe's block encoding is its blocks' sum, 212 + 94 + 2 (913 + 3,502 + 220) + 20: ours is −2 and −25 on the outer pair, +58, +35 and −3 per inner round (spec 13.1: `flat-r`, `exact-iter`, `onehot-pad`, `cmp-free`), +153 in all; dfthcbe's `RefAB` costs 45 with and without its control, ours 44 and 45, hence +152 for the walk and +153 for the controlled walk. Ours − paper is spec 13.1's −204 (−424 from `rz-addsub` in the Givens ladders, +186 from the 26-bit inner entries, +80 from exact boundaries and padding, −46 from free comparator adjoints, 0 net from the rest). No construction was tuned (R-7).

## Compile cost and captured equations

Each `MID_NAPHTHALENE` test alone under `run_watched` (`3a-mid-compiles.log`; 4 GB cap, kill at 3,840 MB), compared with Task 0.9's stand-in (`spikes/results/k1_mid10.json`; two compiles per process then, before L-27) and Task 0.9d's PX arm (`k1d_PX_mid10.json`) at `mid10`, the same sizes (the stand-in's block bits are k1 = 2, k2 = 1, the preset's k1 = 1, k2 = 2):

| Program at `MID_NAPHTHALENE` | Compile | Peak | Task 0.9 stand-in | Task 0.9d PX arm |
| --- | --- | --- | --- | --- |
| `DFTHCBlockEncoding` (test process) | 5.9 s | 613 MB | | |
| `ControlledDFTHCBlockEncoding` (test process) | 6.9 s | 597 MB | | |
| `DFTHCWalk` (subprocess; 9.4 s with start-up) | 7.8 s | 615 MB | 77.8 s, 1,977 MB | 11.2 s, 802 MB |
| `ControlledDFTHCWalk` (subprocess; 9.7 s) | 8.1 s | 588 MB | 155.4 s, 3,571 MB | 11.3 s, 806 MB |

Captured equations (Task 0.9d's instrument, `probes/p3a/capture_walk.py`, `capture-walk.log`): the walk and controlled walk programs capture 11,470 and 11,475 equations at `H2_STO3G_TEST`, 13,121 and 13,064 at `H2_631G_TEST`, and 58,663 and 58,310 at `MID_NAPHTHALENE`, against the PX arm's 94,224 and 94,337 at `mid10` (−38%). The assembly targets cost 1.661 equations per wire their rules bind (block encoding; residual at most 4 over three presets) and 2.707 (walk; residual 1). The FeMoco-54 rules bind 4,301 and 1,656 wires (controlled), so the targets project to about 7,100 and 4,500 equations, against the PX stand-in's FeMoco-54 controlled walk's 62,832 and 8,230 (`k1d_PX_femoco_controlled_walk.json`): about 59,500 fewer, 15% of its 390,949. With the select plan's 42,000 (master plan section 3, item 22), the production controlled walk should capture about a quarter fewer equations than the stand-in that compiled in 2,648 MB; Task 4.5 measures it.

## Simulation cost and the tier budgets

One input of each E check, alone under the watchdog (`time_e_checks.py`, `3a-e-timings.log`; decomposition once, then one two-electron input; every peak below 190 MB):

| E check (tier) | Decompose | One input | Inputs | Projected |
| --- | ---: | ---: | ---: | ---: |
| E-1 `H2_STO3G_TEST` S0 (nightly) | 0.22 s | 0.40 s | 16 | 6.6 s |
| E-1 `H2_STO3G_TEST` S1 (nightly) | 0.22 s | 0.45 s | 4 | 2.0 s |
| E-1 `FAILS_C3` S0 (nightly) | 0.30 s | 1.34 s | 16 | 21.7 s |
| E-2 `H2_STO3G_TEST` k = 1, 2 (nightly) | 0.25, 0.45 s | 0.45, 1.62 s | 4 each | 9.0 s |
| E-2 `FAILS_C3` k = 1, 2 (nightly) | 0.31, 0.60 s | 1.36, 4.27 s | 4 each | 23.4 s |
| E-3 S0, three control states and after a step (nightly) | 1.39 s | 7.69 s | 4 | 32.2 s |
| E-4 alternating, seeded (nightly) | 0.24 s each | 0.35, 0.36 s | 16 each | 11.8 s |
| E-1 `H2_STO3G_TEST` S1, all (release) | 0.25 s | 0.45 s | 16 | 7.5 s |
| E-1 `H2_STO3G_TEST` seeded (release) | 0.25 s | 0.44 s | 4 | 2.0 s |
| E-1 `H2_631G_TEST` S0, one test per input (release) | 0.39 s | 34.8 s (47.0 s for input `10001000`) | 28 | 986 to 1,327 s |
| E-3 S1, all (release) | 1.39 s | 7.67 s | 16 | 124 s |
| E-3 seeded (release) | 1.45 s | 7.58 s | 4 | 31.8 s |

**Projection method.** A test's time is its programs' decomposition plus its inputs times the one-input time; a tier's wall time is the sum over its tests divided by the 2 workers, plus start-up. One input's time depends on the input (at `FAILS_C3` 0.6 to 1.9 s, at 6-31G 35 to 47 s for the two inputs measured), so the projections carry that range. The pre-existing slow tests are taken from their recorded alone times (the Phase 1 to 3 plans' logs: the Task 1B.3 benchmark's two subprocesses 79 s and 26 s, the L-21 compile-time canary about 30 s of compiles, the 22 certificates and the 15 `MID_NAPHTHALENE` compiles each a few seconds) and from the G1 nightly run (348 s for every test but the release ones, per-commit included).

- **Per commit** (measured): the tier, 2 workers, against its budget of 15 minutes and 4 GB for the process tree: `1045 passed in 432.98s`, 7.2 min, peak 1,949 MB (`3a-per-commit-tier.log`); the base, 973 tests on the select replica, took 353 s. Each test, against the per-test gate of about 1 GB and a few seconds: the 72 new per-commit tests in one one-worker session peaked at 485 MB together, so each below that; the longest took 8.5 s (`test_controlled_walk_after_a_walk_step`), five more 5.9 to 6.7 s (the pinned-rule listings of both walks and of the block encoding, the walk-is-`RefAB`-after-the-block-encoding check and the block check at `FAILS_C3`), the other 66 at most 4.2 s (`3a-per-test-evidence.summary`). In the two-worker tier the same tests take up to about 12 s (the after-a-step check: 12.35 s), because the two workers share the machine; their own cost is the one-worker figure.
- **Nightly** (`-m "slow and not release"`, 61 tests): this plan's 14 add about 140 s of CPU time (the E checks 107 s by the table, the four `MID_NAPHTHALENE` compiles 32 s), measured as `10 passed in 77.35s` for the E checks with 2 workers in the replay; the 47 others about 260 s. Projection: about 4 to 6 minutes with 2 workers, below 8 minutes even on the G1 bound (348 s plus the slow tests added since, halved by the workers), against 4 h. The orchestrator runs it at G3, alone, under the watchdog.
- **Release** (`scripts/test.sh`, everything): the per-commit and nightly content above plus this plan's 32 release-only tests, 1,150 to 1,490 s of CPU time (the 6-31G E-1 dominates), 10 to 13 minutes with 2 workers. Projection: about 25 minutes, against 8 h; Phase 4's report tests and the FeMoco cache check (no compile) come on top. No escalation is needed.

## Mutation evidence

`stages/3A/mutate_3a.py` (log `3a-mutants.log`, 15 mutants, each a wrong construction applied to a copy of the replica, run against the per-commit tests of the four new test files): every mutant fails at least one test. The T3 tests on the tape path catch the select and its dagger exchanged (at both shapes since `FAILS_C3`'s inputs are |01⟩|01⟩ and |11⟩|00⟩: with both electrons of an input in one spin, a D1 or Q1 row's a†a and aa† average to the same value over the spin), the dagger in both rounds, the second `InnerSigns` dropped, the first moved before its prepare, the outer unprepare before the second round, the block encoding reflecting about 𝒜 ∪ ℬ, the controlled block encoding with an uncontrolled `RefT2`, the walk reflecting first, and the `phase_grad` register reversed. The controlled walk with an uncontrolled `RefAB` fails only `test_controlled_walk_after_a_walk_step` semantically, and structurally the counts, the children, the listing, the inventory and the pool test (change 7). T4 catches the select's erase at `fixup_bits = 0`; `test_work_pool` and the children test catch work taken from the pool's end; T2 catches the repeated prepares declared once; the constructor test catches the pool check removed; the inventory tests catch a hook that records nothing (the top-level target missing) and the negative control a stray modifier.

## Risks and open questions

1. **The inventory hook is private.** `captured_targets` wraps `catalyst.from_plxpr.qfunc_interpreter._convert_decomp_target_spec`. It stays in the per-commit tier, which the nightly bump procedure runs, and the inventory test's assertion that the program's own top-level target was seen turns a silent hook into a failure (master plan section 9). The pinned PennyLane-side keys (`TEMPLATE_ADJOINTS`) are the templates our leaves adjoint; a bump that changes their decompositions changes the set, which the test then reports.
2. **Two-word keys are slow.** Programs above 64 wires run about 25 to 30 times slower per input than comparable ones within 64 (the probes: 20.6 and 35.5 s at 68 wires against 0.7 and 1.4 s at 60), so `H2_631G_TEST` (80 wires) takes 35 to 47 s per input against 0.4 s at `H2_STO3G_TEST` (`np.unique(axis=0)`, ledger row 1B.1). The release projection carries it; spec 12.6's mitigation 2 (a byte-view unique) would cut it if a later tier needs room.
3. **Per-commit cost.** The three tasks add 72 per-commit tests and about 80 s to the tier (353 s to 433 s), each inside the per-test gate (one-worker session peak 485 MB, at most 8.5 s per test). The slowest, the controlled walk after a walk step (8.5 s with one worker, about 12 s in the two-worker tier), stays per commit: it is the only per-commit check that sees an uncontrolled `RefAB` semantically.
4. **The FeMoco-54 capture projection extrapolates** the assembly targets' linear fit from 1,351 to 4,301 bound wires (residual at most 4 equations over three presets), as the select plan did for its targets; Task 4.5 measures it.
5. **Cross-module private validators.** `walk.py` calls `prepare._check_outer`, `prepare._check_inner` and `select._check`, and the tests and `walk_sim` use `walk._children`, `_emit`, `_registers`; public helpers would be a Task 5.1 cleanup.
6. **No open question blocks the plan.** The decisions above are the obvious choices; change 7 strengthens a check the spec already names (E-3) on the states QPE uses.

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
4. Bodies of auto-adjointed operators contain no `qp.for_loop` or `qp.cond`; `qp.cond` on mid-circuit measurement outcomes only in erase operators and the QPE wrapper; `qp.for_loop` only in `QROAMRead` (one loop per unary leaf, with `qp.cond` on table bits), `ControlledWalkPower` and `ToyControlledWalkPower`, all three never adjointed or controlled.
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
| `src/dfthc/ops/walk.py` | 3.3 (block encodings), 3.4 (replaced: walks added) | the four assembly classes: validation, the time-ordered children lists, the one emitter and the one declaration, the pinned rules |
| `src/dfthc/testing/walk_sim.py` | 3.3 | the tape-path harness of the assembly classes (programs, inputs, strict runs, the block, whole-state comparison, `RefAB`'s action, control superpositions) and the Phase 3 test shapes |
| `tests/ops/test_block_encoding.py` | 3.3 | T2 to T6 of the block encodings, the E-4 precursor, the children, validation |
| `tests/test_work_pool.py` | 3.3, 3.4 (replaced: the walks' calls added) | the work pool's discipline for every call of the assembly rules, and `WalkLayout.work_size` |
| `src/dfthc/testing/resources.py` | 3.4 (shared edit) | `captured_targets` and the inventory helpers |
| `tests/ops/test_walk.py` | 3.4 | T2, T3, T5, T6 of the walks, the inventory test, the failing `qp.ctrl(DFTHCWalk)`, validation |
| `tests/integration/test_walk_counts.py` | 3.4 | whole-program T4 of the walks at the test presets, `MID_NAPHTHALENE` (`slow`, through `run_watched`) and FeMoco-54's closed forms |
| `tests/integration/test_end_to_end.py` | 3.5 | E-1 to E-4 in the nightly and release tiers |

`tests/integration/` is a new directory without `__init__.py`, as `tests/ops/`; test modules keep unique base names.

**Shell conventions.** Run commands from the worktree root of your lane. `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` locates the shared environment, as in the earlier plans.

**Expected test counts.** Each task states its targeted count exactly. The per-commit tier (`scripts/test.sh -m "not slow and not release" -q`) counts **975 tests on `main` with Tasks 2E.3, 3.1 and 3.2 merged** (measured on the stage copy: `973/1021 tests collected (48 deselected)`; 867 on `main` at `7777bc1`, plus 2E.3's 24, 3.1's 37 and 3.2's 45; Task 3.1's review added 2, revision 3). The tasks run in order, so each starts from the previous one's count. The pass criterion is "no failures, no errors, no warnings summary"; the number is a cross-check, and the orchestrator shifts the base if other tasks merge first.

| Task | Base | Own per-commit | Expected per-commit tier | Own slow and release tests |
| --- | ---: | ---: | ---: | --- |
| 3.3 | 975 | 50 (`test_block_encoding.py` 41 of 43, `test_work_pool.py` 9) | 1025 = 975 + 50 | 2 `slow` (the `MID_NAPHTHALENE` counts) |
| 3.4 | 1025 | 28 (`test_walk.py` 23, `test_walk_counts.py` 5 of 7; `test_work_pool.py` replaced, still 9) | 1053 = 1025 + 28 | 2 `slow` (the `MID_NAPHTHALENE` compiles) |
| 3.5 | 1053 | 0 | 1053 | 10 `slow`, 32 `release` (`test_end_to_end.py`) |

Dry run on the replica with all three tasks: `1045 passed in 432.98s`, peak 1,949 MB (`stages/3A/3a-per-commit-tier.log`; run with `TMPDIR` outside the workspace, so `tests/test_infrastructure.py` passes in the replica too).

**Design notes the tasks share.**
- *One list per rule.* `_children(p, t, controlled)` is the block encoding's time order (spec 9.21) with each child's compilable arguments and work need; `_walk_children` is the walk's. `_emit` runs a list (each child on the registers its `wire_argnames` name, `work` the pool's prefix of its need), `_resources` declares it (each distinct child once with its count, `Wire[n]` of exactly the sizes `_emit` passes), and the tests and `walk_sim.prepare_unprepare` read the same list. A wrong order or a missing child therefore shows in the semantics, not as a declaration mismatch.
- *Registers by name, unpacked once.* `_registers` turns every group into a list once and splits it with `WalkLayout.split_group`; the walk passes its block encoding the group lists, the block encoding passes each child named lists (spec 9.0; 2 captured equations per bound wire instead of 4).
- *The tape-path harness* (`walk_sim`). A program prepares the phase gradient itself (`PhaseGradientStatePrep`), so an input is system amplitudes with every other wire in |0⟩; the control wire sits after the layout (64 wires at `H2_STO3G_TEST`: one key word). `project` reads the block: the amplitude with every ancilla |0⟩ and `phase_grad` contracted with the gradient state. `assert_same_state` compares whole outputs term by term, which is how the controlled classes are checked against the plain ones (control |1⟩: the plain output with the control set; |+⟩: the superposition). Every run is strict (each erase outcome has probability 1/2; every AND precondition holds) and consumes exactly the forced schedule. The expected blocks come from `encoded_block`, never from the circuit.
- *Tolerances.* Every comparison states both tolerances (HANDOFF section 6): states and per-commit blocks `rtol=0, atol=1e-12`, the E checks `rtol=0, atol=1e-10` (spec 12.4).
- *FeMoco-54* values are asserted on the closed forms only; `test_work_pool` constructs the FeMoco-54 rules' children in tape mode, which compiles nothing (master plan section 12.9).

---

### Task 3.3: `DFTHCBlockEncoding` and `ControlledDFTHCBlockEncoding`

**Task id:** `3.3` · **Branch:** `phase3/3.3` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 3.1, 3.2

**Files:**
- Create: `src/dfthc/ops/walk.py`, `src/dfthc/testing/walk_sim.py`, `tests/ops/test_block_encoding.py`, `tests/test_work_pool.py`

**Interfaces:**
- Consumes: `dfthc.ops.prepare` (`OuterPrepare`, `OuterUnprepare`, `InnerPrepare`, `InnerUnprepare`, `InnerSigns` with their `required_wires`, and the validators `_check_outer`, `_check_inner`; Tasks 2D.1, 2D.2); `dfthc.ops.select` (`InnerSelect`, `InnerSelectDagger`, `required_wires`, `_check`; Task 3.1); `dfthc.ops.reflections` (`RefT2`, `ControlledRefT2`, `RefAB`, `ControlledRefAB` with `required_wires`; Task 3.2); `dfthc.ops._validate.check_wires`; `dfthc.compile` (`register_rule`, `resource_key`, `compiled_resources`, `FIXED_DECOMPS`); `dfthc.costs` (`block_encoding`, `toffoli_only`, `outer_prepare`, `outer_unprepare`, `inner_prepare`, `inner_select`, `inner_unprepare`, `ref_t2`); `dfthc.params` (`WalkParams`, the presets); `dfthc.registers` (`GROUPS`, `WalkLayout`: `from_params`, `register_sizes`, `work_size`, `split_group`); `dfthc.data` (`WalkTables`, `load_factors`, `random_factors`, `encoded_block`); `SparseSimulator`, `SparseState`, `ForcedPolicy`, `decompose_for_simulation` (Task 1B.2); `chosen_rules`, `verbose_program`, `classify_rule_warning` (Task 0.2).
- Produces, in `dfthc.ops.walk`:
  - `DFTHCBlockEncoding(params, tables, outer, inner, select, system, phase_grad, work)`: compilable `params` (13 ints, `WalkParams.as_compilable()`) and `tables` (`WalkTables.as_compilable()` of tables of `params`' shapes; the constructor runs the prepares' and the select's checks, the one-hot fits and the pool check, `ValueError` otherwise, and stores both canonically); the six `WalkLayout` groups; static `required_wires(params, tables) -> dict` (the layout's group sizes); the rule `_block_encoding`;
  - `ControlledDFTHCBlockEncoding(params, tables, control, outer, inner, select, system, phase_grad, work)`: the same with a one-wire `control` first; `required_wires` adds `"control": 1`; the rule `_controlled_block_encoding` (`ControlledRefT2` in place of `RefT2`, every other child the same);
  - module-private helpers that Task 3.4 and the tests use: `_rebuild`, `_children(p, t, controlled) -> list[(class, compilable args, work need)]`, `_group_sizes`, `_sizes`, `_check`, `_validate`, `_registers(p, outer, inner, select, system, phase_grad, control=())`, `_emit(children, registers, work)`, `_resources(children, p)`. Neither class is ever adjointed or controlled.
- Produces, in `dfthc.testing.walk_sim`: `WalkCase` (`build(params, factors)`, `control`, `num_wires`, `system`, `compilable`, `groups(controlled)`, `block()`), `FAILS_C3`, `CASES`, `case(name)`, `operations(case, *calls)`, `program(case, cls, controlled=False, repeat=1)`, `prepare_unprepare(case)`, `num_measurements`, `schedule(kind, num, seed=0)` (`"S0"`, `"S1"`, `"alternating"`, `"seeded"`), `input_state(case, system, control=None)`, `basis`, `superposition`, `simulate(case, ops, initial, outcomes)`, `gradient`, `project(case, state) -> (2, 4**N)`, `nonzero_registers`, `reflect_ab`, `set_control`, `superpose`, `assert_same_state`, `AB_REGISTERS`, `ATOL`.

**Acceptance tests** (52 tests, 2 of them `slow`), on the shapes `sto3g`, `631g` (the test presets with their vendored factors) and `fails-c3` (`walk_sim.FAILS_C3`, random factors):
- T3 on the tape path (5 tests): at `sto3g` and `fails-c3`, from a superposition of two two-electron system states with every ancilla in |0⟩, the block of `DFTHCBlockEncoding` is `encoded_block`'s H̃_SA/Λ̃ − I applied to it, the control-1 part is zero, only the six index registers of 𝒜 ∪ ℬ are nonzero outside the block, under the alternating and a seeded schedule; `ControlledDFTHCBlockEncoding` at `sto3g` with the control in |0⟩ (the whole input), |1⟩ (the plain class's whole output) and |+⟩ (their superposition); the E-4 precursor, the rule's children without `RefT2` (U then U†), returns a superposition of two states and every ancilla exactly at both shapes, under the alternating and a seeded schedule.
- The children (2 tests, `sto3g`, `fails-c3`): both rules queue, in tape mode, spec 9.21's eleven children in order, each on the layout's registers of its own names (`work` the pool's prefix, `phase_grad` the whole register); the two rules differ only in the reflection, `ControlledRefT2` on the control with one more work wire; both `InnerPrepare` and both `InnerUnprepare` calls are equal, the selects' arguments are equal with `fixup_bits = w_b`, the outer prepare and unprepare get the outer tables.
- T4 (9 tests, 2 `slow`): compiled, each class alone, at `sto3g`, `631g` and `fails-c3`: T and M against `costs.block_encoding`, the plain `Toffoli` count `toffoli_only − 1` and A = T − (`toffoli_only` − 1); at `MID_NAPHTHALENE` (`slow`; 1,874 / 1,875, M = 502); FeMoco-54 on the closed forms (279 + 2 · 4,725 + 20 = 9,749, controlled 9,750, M = 2,524, A = 8,632 and 8,633) and `required_wires` there (143, 455, 799, 108, 15, 44; 1,564 wires).
- T5 and T6 (4 tests): the verbose listing of a program holding both classes names `_block_encoding`, `_controlled_block_encoding`, `_ref_t2` and `_controlled_ref_t2` and not `_c_zero_reflection`; nothing dfthc survives at `level="device"`; `qp.adjoint` and `qp.ctrl` of `DFTHCBlockEncoding` fail with `CompileError` "Decomposition rule not found for operator 'Adjoint(DFTHCBlockEncoding)'" (`'C(DFTHCBlockEncoding)'`) after the expected measuring-rule warnings only.
- T2 (3 tests): both rules at the three shapes, each pinned and the only rule under its class name.
- `required_wires` by hand (1 test); constructor validation (18 tests: eight violations for each class, each a `ValueError`, among them an outer keep entry only the outer prepares reject and a tail angle only the selects reject; the pool check with `WalkLayout.work_size` shrunk; a two-wire `control`); the harness self-test (1 test: both schedule kinds force both outcomes, `assert_same_state` rejects a global and a relative −1, `project` reads an input exactly and 0 once an ancilla is flipped, `nonzero_registers` is empty and then `{"x_o"}`).
- `tests/test_work_pool.py` (9 tests): at `sto3g`, `631g`, `mid` and `femoco` (tape mode), every call of both block-encoding rules gets exactly `required_wires` (`phase_grad` of the prepare classes the whole register), its `work` is the pool's prefix and disjoint from its other registers and the dirty comparator registers, no other register holds a pool wire, the dirty registers go only to their prepare and unprepare classes; the pool equals `ControlledRefAB`'s need and bounds every other; at 300 random valid shapes every child in `_children` (plain and controlled), `OuterUnprepare`, `InnerUnprepare` and `InnerSelectDagger` included, is given exactly its own class's `required_wires` work, and `WalkLayout.work_size` covers it and the walk's `RefAB` and `ControlledRefAB`.

**Definition of done:** `52 passed` targeted (50 in the per-commit tier); the per-commit tier passes; `ruff check .` clean; the spec 9.0 docstring sections in order; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_block_encoding.py`:

```python
"""DFTHCBlockEncoding and ControlledDFTHCBlockEncoding (spec 9.21): the block against the encoded
model on the tape path, the three control states, U then U-dagger, the harness, the children's
order and registers, counts, rule selection, validation."""

import dataclasses
import re
import warnings

import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, compiled_resources
from dfthc.data.factors import load_factors
from dfthc.ops.walk import (
    ControlledDFTHCBlockEncoding,
    DFTHCBlockEncoding,
    _block_encoding,
    _controlled_block_encoding,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.registers import WalkLayout
from dfthc.testing import walk_sim
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program

CLASSES = {"plain": (DFTHCBlockEncoding, _block_encoding),
           "controlled": (ControlledDFTHCBlockEncoding, _controlled_block_encoding)}
case = walk_sim.case
"""The test shapes: ``sto3g``, ``631g`` and ``fails-c3`` (walk_sim.FAILS_C3: the failed uniform
preparation of x_o, dfthcbe's differing failure signs, C = 3)."""


SUPPORT = {"sto3g": (5, 10), "fails-c3": (5, 12)}
"""Two-electron system inputs (psi_up's bits first): |01>|01> and |10>|10> at sto3g, |01>|01>
and |11>|00> at fails-c3 (about 1.1 and 0.9 s each there). Not both electrons of every input in
one spin: there a D1 or Q1 row's a-dagger a and a a-dagger average to the same value over the
spin, which hides the select exchanged with its dagger."""


# --------------------------------------------------------------------------- semantics


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_block_against_the_encoded_model(name):
    """T3 on the tape path (strict sparse simulator: every erase outcome has probability 1/2 and
    every AND precondition holds): from a superposition of two system states with every ancilla
    in |0>, the output's part with every ancilla in |0> and phase_grad in the gradient state is
    (H~_SA / Lambda~ - I) applied to the input (encoded_block, spec 8.5), under the alternating
    and a seeded schedule of the erases. fails-c3 has the failed uniform preparation of x_o (so
    Lambda~ != Lambda) and the dfthcbe failure-sign difference; sto3g is the H2 test preset.
    Outside the block, only the six index registers of A u B are nonzero (the postcondition that
    makes RefAB the walk's reflection). The output's norm is not asserted: the simulator
    renormalizes the kept branch of every measurement to norm 1."""
    c = case(name)
    ops = walk_sim.program(c, DFTHCBlockEncoding)
    psi = walk_sim.superposition(c, SUPPORT[name], seed=3)
    for kind in ("alternating", "seeded"):
        out = walk_sim.simulate(c, ops, walk_sim.input_state(c, psi),
                                walk_sim.schedule(kind, walk_sim.num_measurements(ops)))
        block = walk_sim.project(c, out)
        assert np.allclose(block[0], c.block() @ psi, rtol=0, atol=1e-12)
        assert np.allclose(block[1], 0, rtol=0, atol=1e-12)
        assert walk_sim.nonzero_registers(c, out) <= set(walk_sim.AB_REGISTERS)


def test_controlled_block_encoding_in_three_control_states():
    """T3 of the controlled class at H2_STO3G_TEST: control |0> returns the whole input
    (U-dagger U = I), control |1> gives DFTHCBlockEncoding's whole output, and control |+> their
    superposition with weight 1/sqrt 2 each, which fails if the controlled reflection had a wrong
    global phase (spec 3.3). The controlled reflection is table-free, so one shape suffices."""
    c = case("sto3g")
    psi = walk_sim.basis(c, 10)
    plain = walk_sim.program(c, DFTHCBlockEncoding)
    controlled = walk_sim.program(c, ControlledDFTHCBlockEncoding, controlled=True)
    outcomes = walk_sim.schedule("seeded", walk_sim.num_measurements(plain), seed=4)
    identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    applied = walk_sim.set_control(c, walk_sim.simulate(c, plain, walk_sim.input_state(c, psi),
                                                        outcomes))
    plus = 2**-0.5
    expected = {"0": identity, "1": applied,
                "+": walk_sim.superpose((plus, identity), (plus, applied))}
    for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", [plus, plus])):
        out = walk_sim.simulate(c, controlled, walk_sim.input_state(c, psi, control), outcomes)
        walk_sim.assert_same_state(out, expected[label])
    assert np.allclose(walk_sim.project(c, applied)[1], c.block() @ psi, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_prepare_then_unprepare_is_the_identity(name):
    """The E-4 precursor (spec 12.4): the rule's children without the reflection, U then
    U-dagger, return a superposition of two system states and every ancilla exactly,
    under the alternating and a seeded schedule of the five erases (an all-ones schedule sees only
    each address's mask parity; ledger row 2D.2)."""
    c = case(name)
    ops = walk_sim.prepare_unprepare(c)
    psi = walk_sim.superposition(c, SUPPORT[name], seed=5)
    identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    for kind in ("alternating", "seeded"):
        out = walk_sim.simulate(c, ops, walk_sim.input_state(c, psi),
                                walk_sim.schedule(kind, walk_sim.num_measurements(ops), seed=6))
        walk_sim.assert_same_state(out, identity)


def test_harness_sees_phases_outcomes_and_ancillas():
    """The harness the semantic tests rely on (and Tasks 3.4 and 3.5): the alternating and seeded
    schedules force both outcomes, assert_same_state compares complex amplitudes (a -1 on the
    whole state or on one term fails it), project returns an input exactly and drops every term
    with a nonzero ancilla, and nonzero_registers names that ancilla's register."""
    c = case("sto3g")
    for kind in ("alternating", "seeded"):
        assert set(walk_sim.schedule(kind, 8)) == {0, 1}
    psi = walk_sim.superposition(c, SUPPORT["sto3g"], seed=1)
    state = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    assert np.allclose(walk_sim.project(c, state)[0], psi, rtol=0, atol=1e-12)
    relative = state.copy()
    relative.amps = np.where(relative.bits(c.system[0]), -1, 1) * relative.amps
    for wrong in (walk_sim.superpose((-1, state)), relative):
        with pytest.raises(AssertionError):
            walk_sim.assert_same_state(wrong, state)
    assert walk_sim.nonzero_registers(c, state) == set()
    dirty = state.copy()
    dirty.flip(c.layout.x_o[0])
    assert np.allclose(walk_sim.project(c, dirty), 0, rtol=0, atol=1e-12)
    assert walk_sim.nonzero_registers(c, dirty) == {"x_o"}


def children(rule, c, controlled):
    """The rule's operators in tape mode."""
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(*c.compilable, **c.groups(controlled))
    return list(queue.queue)


def registers(op):
    op = getattr(op, "base", op)
    return {name: list(getattr(op, name)) for name in op.wire_argnames}


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_children_order_and_registers(name):
    """Spec 9.21's order, every child on the layout's registers of its own names (spec 9.0):
    the work pool's prefix, the whole phase_grad; InnerSigns after InnerPrepare in the first
    round and after InnerSelectDagger in the second; the two rules differ only in the reflection,
    ControlledRefT2 on the control; the two InnerPrepare and InnerUnprepare calls are equal, and
    the selects have equal arguments with fixup_bits = w_b, so each shares one target."""
    c = case(name)
    p, layout = c.params, c.layout
    names = ["OuterPrepare", "InnerPrepare", "InnerSigns", "InnerSelect", "InnerUnprepare",
             "RefT2", "InnerPrepare", "InnerSelectDagger", "InnerSigns", "InnerUnprepare",
             "OuterUnprepare"]
    plain, controlled = children(_block_encoding, c, False), children(
        _controlled_block_encoding, c, True)
    assert [op.name for op in plain] == names
    assert [op.name for op in controlled] == [
        "ControlledRefT2" if n == "RefT2" else n for n in names]
    for op in plain + controlled:
        for register, wires in registers(op).items():
            if register == "work":
                assert wires == list(layout.work)[:len(wires)]
            elif register == "control":
                assert wires == c.control
            else:
                assert wires == list(getattr(layout, register)), (op.name, register)
    for a, b in zip(plain, controlled, strict=True):
        if a.name != "RefT2":
            assert (a.hyperparameters, registers(a)) == (b.hyperparameters, registers(b))
    reflection, controlled_reflection = registers(plain[5]), registers(controlled[5])
    assert controlled_reflection.pop("control") == c.control
    assert len(controlled_reflection.pop("work")) == len(reflection.pop("work")) + 1
    assert controlled_reflection == reflection
    assert plain[1].hyperparameters == plain[6].hyperparameters
    assert plain[4].hyperparameters == plain[9].hyperparameters
    assert plain[3].hyperparameters == plain[7].hyperparameters
    assert plain[3].hyperparameters["fixup_bits"] == p.w_b
    assert plain[0].hyperparameters == plain[10].hyperparameters == {
        "X_o": p.X_o, "alt": c.tables.outer_alt, "keep": c.tables.outer_keep, "b_k1": p.b_k1,
        "k1": p.k1, "ke1": p.ke1, "s_u": p.s_u}


# --------------------------------------------------------------------------- resources


def assert_counts(resources, p, controlled):
    """T4 and the T6 signature: T and M from costs.block_encoding; the plain Toffolis are the
    walk's T_Toffoli less RefAB's one, so A = T - (T_Toffoli - 1) (spec 9.24)."""
    expected = costs.block_encoding(p, controlled)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.counts["Toffoli"] == costs.toffoli_only(p) - 1
    assert resources.measured_and_uncomputes == resources.toffoli - (costs.toffoli_only(p) - 1)


def compile_class(c, kind):
    cls, _ = CLASSES[kind]
    controlled = kind == "controlled"

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        cls(*c.compilable, **c.groups(controlled))

    return compiled_resources(program, c.num_wires)


@pytest.mark.compiled
@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize("name", ["sto3g", "631g", "fails-c3"])
def test_counts(name, kind):
    """T4 compiled, each class alone: T, M, A at both test presets (T = 150 / 151 and 231 / 232,
    M = 50 and 68) and at fails-c3."""
    c = case(name)
    assert_counts(compile_class(c, kind), c.params, kind == "controlled")


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("kind", CLASSES)
def test_counts_at_mid_naphthalene(kind):
    """T4 at MID_NAPHTHALENE (nightly tier): T = 1,874 / 1,875, M = 502, A = 1,640 / 1,641."""
    c = walk_sim.WalkCase.build(MID_NAPHTHALENE, load_factors("naphthalene"))
    assert_counts(compile_class(c, kind), c.params, kind == "controlled")


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 (spec 9.24), without compiling: T = 279 + 2 * 4,725 + 20 = 9,749 (controlled
    9,750), M = 96 + 2 * (416 + 798) = 2,524, A = T - (1,118 - 1) = 8,632 (8,633); the groups are
    143, 455, 799, 108, 15 and 44 wires, 1,564 in all."""
    p = FEMOCO54
    outer = costs.outer_prepare(p) + costs.outer_unprepare(p)
    inner = costs.inner_prepare(p) + costs.inner_select(p) + costs.inner_unprepare(p)
    assert (outer, inner, costs.ref_t2(p, False)) == (costs.Cost(279, 96), costs.Cost(4725, 1214),
                                                      costs.Cost(20))
    assert costs.block_encoding(p, False) == costs.Cost(9749, 2524)
    assert costs.block_encoding(p, True) == costs.Cost(9750, 2524)
    assert costs.toffoli_only(p) == 1118
    assert [costs.block_encoding(p, c).toffoli - (costs.toffoli_only(p) - 1)
            for c in (False, True)] == [8632, 8633]
    sizes = DFTHCBlockEncoding.required_wires(p.as_compilable(), ())
    assert sizes == {"outer": 143, "inner": 455, "select": 799, "system": 108, "phase_grad": 15,
                     "work": 44}
    assert sum(sizes.values()) == WalkLayout.from_params(p).num_wires == 1564


def both_classes():
    c = case("sto3g")

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        DFTHCBlockEncoding(*c.compilable, **c.groups())
        ControlledDFTHCBlockEncoding(*c.compilable, **c.groups(True))

    return program, c.num_wires


@pytest.mark.compiled
def test_pinned_rules_are_chosen(capfd):
    """T6 at H2_STO3G_TEST: the verbose listing of a program holding both classes names both
    pinned rules and the two reflections', and not C(ZeroReflection)'s."""
    program, num_wires = both_classes()
    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_block_encoding", "_controlled_block_encoding", "_ref_t2",
            "_controlled_ref_t2"} <= rules
    assert "_c_zero_reflection" not in rules


@pytest.mark.compiled
def test_nothing_survives_at_device_level():
    """T5 at level="device" (L-20): no dfthc operator survives in a program holding both
    classes."""
    program, num_wires = both_classes()
    compiled_resources(program, num_wires, level="device")


@pytest.mark.compiled
@pytest.mark.parametrize("wrap", ["adjoint", "control"])
def test_no_adjoint_or_controlled_path(wrap):
    """Spec 7.4 rules 5 and 6: nothing adjoints or controls the block encoding (the controlled
    form is ControlledDFTHCBlockEncoding). Wrapped anyway, Catalyst drops the measuring candidates
    with the expected warning (rule 12), and the compile fails loudly."""
    c = case("sto3g")
    op = DFTHCBlockEncoding(*c.compilable, **c.groups())
    key = "Adjoint(DFTHCBlockEncoding)" if wrap == "adjoint" else "C(DFTHCBlockEncoding)"
    program = ((lambda: qp.adjoint(op)) if wrap == "adjoint"
               else (lambda: qp.ctrl(op, control=c.control[0])))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(program, c.num_wires)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the ")]
    named = [message for message in skipped if classify_rule_warning(message) != "noise"]
    assert named and {classify_rule_warning(message) for message in named} == {"expected"}


@pytest.mark.parametrize("name", ["sto3g", "631g", "fails-c3"])
def test_rules_are_valid(name):
    """T2 (tape mode) for both rules at three sizes: the declared children equal the emitted ones
    (compilable arguments and register sizes, the repeated prepares counted twice); each rule is
    pinned and the only one under its class name (spec 7.4 rules 7 and 8)."""
    c = case(name)
    for kind, (cls, rule) in CLASSES.items():
        assert FIXED_DECOMPS[cls.__name__] is rule
        assert list(qp.list_decomps(cls)) == [rule]
        op = cls(*c.compilable, **c.groups(kind == "controlled"))
        qp.testing.assert_valid_decomp_rule(op, rule, check_matrix=False)


def test_required_wires():
    """By hand at H2_STO3G_TEST: the layout's groups (work: RefAB's controlled need, 8)."""
    p = H2_STO3G_TEST
    sizes = {"outer": 16, "inner": 23, "select": 8, "system": 4, "phase_grad": 4, "work": 8}
    assert DFTHCBlockEncoding.required_wires(p.as_compilable(), ()) == sizes
    assert ControlledDFTHCBlockEncoding.required_wires(p.as_compilable(), ()) == {
        "control": 1} | sizes


def bad_outer_keep(c):
    """The case's tables with the first outer keep value 2**b_k1, outside the outer prepares'
    range; every other table stays valid."""
    keep = (2**c.params.b_k1, *c.tables.outer_keep[1:])
    return dataclasses.replace(c.tables, outer_keep=keep).as_compilable()


def bad_tail_angle(c):
    """The case's tables with the first tail angle 2**(b_rot - 1), outside the selects' range."""
    rows = c.tables.tail_angles
    rows = ((2 ** (c.params.b_rot - 1), *rows[0][1:]), *rows[1:])
    return dataclasses.replace(c.tables, tail_angles=rows).as_compilable()


@pytest.mark.parametrize(("name", "change", "match"), [
    ("sto3g", {"params": 7}, "params and tables must be"),
    ("sto3g", {"params": H2_631G_TEST.as_compilable()}, r"alt must be a table of ints of shape"),
    ("fails-c3", {"params": walk_sim.FAILS_C3.as_compilable()[:-1] + (3,)},
     r"the erase's one-hot over ke2 \+ lg\(B \+ 1\) = 5 bits needs 32 wires"),
    ("sto3g", {"tables": ()}, "WalkTables needs 8 fields; got 0"),
    ("sto3g", {"tables": bad_outer_keep}, r"keep entries must lie in \[0, 4\), got 4"),
    ("sto3g", {"tables": bad_tail_angle}, r"tail_angles angles must lie in \[0, 2\*\*\(b_rot"),
    ("sto3g", {"work": list(range(7))}, "'work' has 7 wires; expected 8"),
    ("sto3g", {"system": [0]}, "'system' has 1 wires; expected 4"),
], ids=["params-type", "params-shape", "onehot", "tables", "outer-keep", "tail-angle", "work",
        "system"])
@pytest.mark.parametrize("kind", CLASSES)
def test_constructor_validates(kind, name, change, match):
    """Spec 9.0: a wrong argument type, tables of another shape, a one-hot register that does
    not fit (ke2 = 3 needs 2**(3 + w_b) = 32 wires of the inner erase's 8), missing tables, an
    outer keep value only the outer prepares check, a tail angle only the selects check, and
    wrong group widths each raise ValueError."""
    c = case(name)
    cls, _ = CLASSES[kind]
    params, tables = c.compilable
    change = {key: value(c) if callable(value) else value for key, value in change.items()}
    arguments = {"params": params, "tables": tables} | c.groups(kind == "controlled") | change
    with pytest.raises(ValueError, match=match):
        cls(**arguments)


def test_controlled_constructor_checks_the_control_width():
    """The controlled class checks its one-wire control like the groups (spec 9.0)."""
    c = case("sto3g")
    with pytest.raises(ValueError, match="'control' has 2 wires; expected 1"):
        ControlledDFTHCBlockEncoding(*c.compilable,
                                     **c.groups(True) | {"control": [*c.control, c.num_wires]})


def test_constructor_checks_the_work_pool(monkeypatch):
    """The layout's pool must cover every child of the walk: with WalkLayout.work_size shrunk to
    7 at H2_STO3G_TEST, ControlledRefAB's 8 wires do not fit (spec 10.1)."""
    c = case("sto3g")
    monkeypatch.setattr(WalkLayout, "work_size", staticmethod(lambda params: 7))
    with pytest.raises(ValueError, match=r"too small for \{'ControlledRefAB': 8\}"):
        DFTHCBlockEncoding(*c.compilable, **c.groups())
```

Create `tests/test_work_pool.py`:

```python
"""The walk's shared work pool (spec 10.1 and 10.2): every call of the block encoding's rules gets
exactly its required_wires, its work is a prefix of the pool, and neither the pool nor the
comparators' dirty work registers appear where they do not belong; WalkLayout.work_size covers
every child of the walk."""

import functools
import inspect
import random

import pennylane as qp
import pytest

from dfthc.data.factors import load_factors, random_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefAB, RefT2
from dfthc.ops.walk import _block_encoding, _children, _controlled_block_encoding
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams, lg
from dfthc.registers import WalkLayout

PRESETS = {
    "sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "mid": lambda: (MID_NAPHTHALENE, load_factors("naphthalene")),
    "femoco": lambda: (FEMOCO54, random_factors(54, 10, 27, 27, n_plus=27, seed=0)),
}
"""FeMoco-54 is classical here: the rules run in tape mode and construct their children (the
constructors check every register); nothing is compiled."""

DIRTY = {"succ_work_x_o": {"OuterPrepare", "OuterUnprepare"},
         "cmp_work_x_o": {"OuterPrepare", "OuterUnprepare"},
         "succ_work_b": {"InnerPrepare", "InnerUnprepare"},
         "cmp_work_b": {"InnerPrepare", "InnerUnprepare"}}
"""The comparators' work registers, dirty from a prepare to its unprepare (spec 10.2), and the
only classes that may take them."""

AT_LEAST = {"OuterPrepare", "OuterUnprepare", "InnerPrepare", "InnerUnprepare"}
"""Their ``phase_grad`` is a minimum (``s_u``); the walk passes the whole register."""

RULES = {"block encoding": (_block_encoding, False, 11),
         "controlled block encoding": (_controlled_block_encoding, True, 11)}
"""Each assembly rule, whether it is controlled, and its number of calls."""


@functools.cache
def preset(name):
    p, factors = PRESETS[name]()
    return p, WalkTables.from_factors(factors, p), WalkLayout.from_params(p, qpe_bits=1)


def required(op) -> dict:
    """``required_wires`` of the call: from its compilable arguments, or, for the reflections,
    whose sizes follow from their wires, from the widths of its pattern registers."""
    cls = type(op)
    if cls in (RefT2, ControlledRefT2):
        return cls.required_wires(len(op.b), len(op.m_b))
    if cls in (RefAB, ControlledRefAB):
        return cls.required_wires(len(op.x_o), len(op.m_x_o), len(op.b), len(op.m_b))
    names = inspect.signature(cls.required_wires).parameters
    return cls.required_wires(**{k: v for k, v in op.hyperparameters.items() if k in names})


def calls(rule, p, t, layout, controlled):
    """The rule's children in tape mode, on the layout's groups (``qpe_control`` as control)."""
    groups = {name: getattr(layout, name)
              for name in ("outer", "inner", "select", "system", "phase_grad", "work")}
    if controlled:
        groups = {"control": layout.qpe_control} | groups
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(p.as_compilable(), t.as_compilable(), **groups)
    return list(queue.queue)


def check_call(op, layout) -> int:
    """Assert the pool discipline for one call and return its work need."""
    sizes = required(op)
    pool = set(layout.work)
    registers = {name: list(getattr(op, name)) for name in op.wire_argnames}
    assert set(registers) == set(sizes), op.name
    for name, wires in registers.items():
        if name == "phase_grad" and op.name in AT_LEAST:
            assert wires == list(layout.phase_grad) and len(wires) >= sizes[name], op.name
        else:
            assert len(wires) == sizes[name], (op.name, name)
        if name == "work":
            assert wires == list(layout.work)[:len(wires)], op.name
        else:
            assert not pool & set(wires), (op.name, name)
        if name in DIRTY:
            assert op.name in DIRTY[name] and wires == list(getattr(layout, name))
    work = set(registers.get("work", ()))
    others = {wire for name, wires in registers.items() if name != "work" for wire in wires}
    dirty = {wire for name in DIRTY for wire in getattr(layout, name)}
    assert not work & others and not work & dirty, op.name
    return sizes.get("work", 0)


@pytest.mark.parametrize("name", PRESETS)
def test_every_call_gets_exactly_its_required_wires(name):
    """Spec 10.2 and master plan Task 3.3: in both block-encoding rules every child receives
    exactly required_wires (phase_grad of the prepare classes is a minimum: they get the whole
    register), its work is the pool's prefix of that size and disjoint from its other registers,
    no other register holds a pool wire, the dirty comparator work registers go only to their
    prepare and unprepare (never into a work prefix), and every need fits the pool."""
    p, t, layout = preset(name)
    for rule, controlled, count in RULES.values():
        needs = [check_call(op, layout) for op in calls(rule, p, t, layout, controlled)]
        assert len(needs) == count and max(needs) <= len(layout.work)


@pytest.mark.parametrize("name", PRESETS)
def test_the_pool_is_the_controlled_walks_largest_need(name):
    """WalkLayout.work_size is the largest need of the walk's operators: the controlled RefAB's
    w_x + b_k1 + w_b + b_k2 at every preset (44 at FeMoco-54, spec 10.3), which the walk (Task
    3.4) will call; the uncontrolled RefAB needs one wire less, the block encodings' children at
    most that."""
    p, t, layout = preset(name)
    needs = {}
    for rule, controlled, _ in RULES.values():
        for op in calls(rule, p, t, layout, controlled):
            needs[op.name] = max(needs.get(op.name, 0), check_call(op, layout))
    for cls in (RefAB, ControlledRefAB):
        needs[cls.__name__] = cls.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)["work"]
    assert max(needs.values()) == needs["ControlledRefAB"] == len(layout.work)
    assert needs["ControlledRefAB"] == WalkLayout.work_size(p) == p.w_x + p.b_k1 + p.w_b + p.b_k2
    assert needs["RefAB"] == needs["ControlledRefAB"] - 1


def own_work(cls, args, p) -> int:
    """A child's own work need: its class's ``required_wires(...)["work"]`` from the compilable
    arguments the rule gives it, or, for the reflections (sizes from their wires), from the
    pattern widths; 0 without a work register."""
    if cls in (RefT2, ControlledRefT2):
        return cls.required_wires(p.w_b, p.b_k2)["work"]
    if cls in (RefAB, ControlledRefAB):
        return cls.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)["work"]
    arguments = dict(zip(cls.compilable_argnames, args, strict=True))
    names = inspect.signature(cls.required_wires).parameters
    return cls.required_wires(**{k: v for k, v in arguments.items() if k in names}).get("work", 0)


def zero_tables(p) -> WalkTables:
    """Tables of ``p``'s shapes with every entry 0: the work needs depend on the shapes only."""
    def rows(n, m):
        return tuple((0,) * m for _ in range(n))

    inner = rows(p.RC, p.B + 1)
    return WalkTables(outer_alt=(0,) * p.X_o, outer_keep=(0,) * p.X_o, inner_alt=inner,
                      inner_keep=inner, inner_sgn=inner, inner_d_sgn=inner,
                      tail_angles=rows(p.N, p.N - 1),
                      sf_angles=tuple(rows(p.B, p.N - 1) for _ in range(p.R)))


CHILDREN = {"OuterPrepare", "InnerPrepare", "InnerSigns", "InnerSelect", "InnerUnprepare",
            "RefT2", "ControlledRefT2", "InnerSelectDagger", "OuterUnprepare", "RefAB",
            "ControlledRefAB"}
"""Every class the two block-encoding rules call, and the walk's reflections (Task 3.4)."""


def test_work_size_covers_every_child_at_random_shapes():
    """At 300 random valid shapes, every child of both block-encoding rules, taken from their own
    child list (``_children``, plain and controlled), is given exactly its class's own
    ``required_wires`` work, so no explicit inverse or controlled class (``OuterUnprepare``,
    ``InnerUnprepare``, ``InnerSelectDagger``, ``ControlledRefT2``) can drift from its forward
    form unnoticed; and ``WalkLayout.work_size`` (spec 10.1, provisional until this test) covers
    every need, the walk's ``RefAB`` and ``ControlledRefAB`` (Task 3.4) included."""
    rng = random.Random(0)
    checked, seen = 0, set()
    while checked < 300:
        N, R, B, C = rng.randint(2, 12), rng.randint(1, 10), rng.randint(1, 30), rng.randint(1, 30)
        b_rot = rng.randint(3, 16)
        values = dict(N=N, R=R, B=B, C=C, n_plus=rng.randint(0, N), b_rot=b_rot,
                      b_k1=rng.randint(1, 16), b_k2=rng.randint(1, 16), s_u=rng.randint(3, b_rot))
        values |= dict(k1=rng.randint(0, lg(N + R * C)), k2=rng.randint(0, lg(B + 1)),
                       ke1=rng.randint(0, lg(N + R * C)), ke2=rng.randint(0, lg(R * C)))
        p = WalkParams(**values)
        if (2**p.ke1 > p.K1 * p.d1 or 2**(p.ke2 + p.w_b) > p.K2 * p.d2
                or 2**p.w_b > (p.N - 1) * p.b_rot + 1):
            continue
        t = zero_tables(p)
        for controlled in (False, True):
            for cls, args, need in _children(p, t, controlled):
                assert own_work(cls, args, p) == need, (p, cls.__name__)
                assert need <= WalkLayout.work_size(p), (p, cls.__name__, need)
                seen.add(cls.__name__)
        for cls in (RefAB, ControlledRefAB):
            assert own_work(cls, (), p) <= WalkLayout.work_size(p), (p, cls.__name__)
            seen.add(cls.__name__)
        checked += 1
    assert seen == CHILDREN
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_block_encoding.py tests/test_work_pool.py -q`
Expected: FAIL at collection, `2 errors`, each `ModuleNotFoundError: No module named 'dfthc.ops.walk'`.

- [ ] **Step 3: Write the block encoding**

Create `src/dfthc/ops/walk.py`:

```python
r"""The block encoding of the sum-of-squares Hamiltonian and its controlled class (spec 9.21;
steps 1 to 27).

**The circuit** (spec 3.3 and 3.5). The block encoding is :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U`
with :math:`U` = ``OuterPrepare``, ``InnerPrepare``, ``InnerSigns``, ``InnerSelect``,
``InnerUnprepare`` in time order and :math:`U^\dagger` = ``InnerPrepare``, ``InnerSelectDagger``,
``InnerSigns``, ``InnerUnprepare``, ``OuterUnprepare``; its block on the ancillas in
:math:`|0\rangle` is :math:`\tilde H_{SA}/\tilde\Lambda - I` (spec 8.5). Every adjoint in it is an
explicit class (``InnerUnprepare``, ``InnerSelectDagger``, ``OuterUnprepare``; spec 7.4 rule 6).
The controlled class controls only the reflection: on states whose ancillas start in
:math:`|0\rangle`, :math:`U^\dagger U = I` (spec 3.3). There is no ``C(...)`` rule of either
operator.

**Wires.** Each class takes the groups of :class:`~dfthc.registers.WalkLayout` (``outer``,
``inner``, ``select``, ``system``, ``phase_grad``, ``work``; a controlled class ``control``
first). Its rule splits them once with :meth:`WalkLayout.split_group`, unpacks every register into
a list once, and passes each child the registers its ``wire_argnames`` name, with the prefix of the
``work`` group of exactly the child's need (spec 9.0, wire passing). ``work`` is the walk's shared
pool: every child returns it in :math:`|0\rangle`; the comparators' dirty work registers belong to
the ``outer`` and ``inner`` groups, never to ``work`` (spec 10.2).

**Arguments.** ``params`` is :meth:`WalkParams.as_compilable` and ``tables`` is
:meth:`WalkTables.as_compilable`; the rules rebuild both with ``from_compilable``. One list,
:func:`_children`, gives the rule its children in time order with their compilable arguments and
work needs, and the resource function its declaration, so the two cannot disagree.
"""

from __future__ import annotations

from collections import Counter

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.data.tables import WalkTables
from dfthc.ops._validate import check_wires
from dfthc.ops.prepare import (
    InnerPrepare,
    InnerSigns,
    InnerUnprepare,
    OuterPrepare,
    OuterUnprepare,
    _check_inner,
    _check_outer,
)
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefT2
from dfthc.ops.select import InnerSelect, InnerSelectDagger
from dfthc.ops.select import _check as _check_select
from dfthc.params import WalkParams
from dfthc.registers import GROUPS, WalkLayout

_GROUPS = ("outer", "inner", "select", "system", "phase_grad", "work")
"""The wire arguments of the assembly operators, after ``control`` in the controlled ones."""


def _rebuild(params, tables) -> tuple[WalkParams, WalkTables]:
    return WalkParams.from_compilable(params), WalkTables.from_compilable(tables)


def _children(p: WalkParams, t: WalkTables, controlled: bool) -> list[tuple[type, tuple, int]]:
    """The block encoding's children in time order (spec 9.21), each ``(class, compilable
    arguments, work wires)``; the reflection is ``ControlledRefT2`` if ``controlled``."""
    outer = (p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    inner = (p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn, p.b_k2,
             p.k2, p.ke2, p.s_u)
    select = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    reflection = ControlledRefT2 if controlled else RefT2
    outer_work = OuterPrepare.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)["work"]
    inner_work = InnerPrepare.required_wires(p.N, p.R, p.B, p.C, p.b_k2, p.k2, p.ke2,
                                             p.s_u)["work"]
    select_work = InnerSelect.required_wires(*select)["work"]
    reflection_work = reflection.required_wires(p.w_b, p.b_k2)["work"]
    return [
        (OuterPrepare, outer, outer_work),
        (InnerPrepare, inner, inner_work),
        (InnerSigns, (), 0),
        (InnerSelect, select, select_work),
        (InnerUnprepare, inner, inner_work),
        (reflection, (), reflection_work),
        (InnerPrepare, inner, inner_work),
        (InnerSelectDagger, select, select_work),
        (InnerSigns, (), 0),
        (InnerUnprepare, inner, inner_work),
        (OuterUnprepare, outer, outer_work),
    ]


def _group_sizes(p: WalkParams) -> dict[str, int]:
    """The width of every group: the layout's (spec 10.1)."""
    sizes = WalkLayout.register_sizes(p)
    return {group: sum(sizes[name] for name in GROUPS[group]) for group in _GROUPS}


def _sizes(p: WalkParams) -> dict[str, int]:
    """The width of every register and group a child of an assembly rule can take."""
    return WalkLayout.register_sizes(p) | _group_sizes(p) | {"control": 1}


def _check(params, tables) -> tuple[WalkParams, WalkTables]:
    """``params`` and ``tables`` rebuilt and checked against the children's domains (spec 9.0):
    the tables have the shapes of ``params`` and lie in the children's ranges, the erases'
    one-hot registers fit (``2**ke1 <= K1 d1``, ``2**(ke2 + w_b) <= K2 d2`` and
    ``2**w_b <= (N - 1) b_rot + 1``), and the layout's ``work`` pool covers every child of the walk
    and the controlled walk. ``ValueError`` otherwise."""
    if not isinstance(params, (tuple, list)) or not isinstance(tables, (tuple, list)):
        raise ValueError("params and tables must be WalkParams.as_compilable() and "
                         "WalkTables.as_compilable()")
    p, t = _rebuild(params, tables)
    _check_outer(p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    _check_inner(p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn,
                 p.b_k2, p.k2, p.ke2, p.s_u)
    _check_select(t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    needs = {cls.__name__: work for cls, _, work in _children(p, t, controlled=True)}
    needs["ControlledRefAB"] = ControlledRefAB.required_wires(p.w_x, p.b_k1, p.w_b,
                                                              p.b_k2)["work"]
    pool = WalkLayout.work_size(p)
    over = {name: need for name, need in needs.items() if need > pool}
    if over:
        raise ValueError(f"the work pool of {pool} wires (WalkLayout.work_size) is too small for "
                         f"{over}")
    return p, t


def _validate(name: str, wire_argnames: tuple, params, tables, wires: tuple) -> tuple:
    """The canonical compilable arguments, after checking them and the wire groups."""
    p, t = _check(params, tables)
    required = ({"control": 1} if "control" in wire_argnames else {}) | _group_sizes(p)
    check_wires(name, dict(zip(wire_argnames, wires, strict=True)), required)
    return p.as_compilable(), t.as_compilable()


def _registers(p: WalkParams, outer, inner, select, system, phase_grad, control=()) -> dict:
    """Every group and every register by name, each unpacked into a list once (spec 9.0)."""
    registers = {"outer": list(outer), "inner": list(inner), "select": list(select),
                 "system": list(system), "phase_grad": list(phase_grad),
                 "control": list(control)}
    for group in ("outer", "inner", "select", "system"):
        registers |= WalkLayout.split_group(p, group, registers[group])
    return registers


def _emit(children: list, registers: dict, work) -> None:
    """Apply ``children`` in order, each on the registers its ``wire_argnames`` name and on the
    prefix of ``work`` of exactly its need."""
    work = list(work)
    for cls, args, need in children:
        cls(*args, **{name: work[:need] if name == "work" else registers[name]
                      for name in cls.wire_argnames})


def _resources(children: list, p: WalkParams) -> dict:
    """The declaration of ``children``: each distinct child once, with its count, its compilable
    arguments and exactly the register sizes :func:`_emit` passes (spec 7.4 rule 2)."""
    sizes = _sizes(p)

    def key(cls, args, need):
        return resource_key(lambda: cls(*args, **{
            name: Wire[need if name == "work" else sizes[name]] for name in cls.wire_argnames}))

    return {key(*child): count for child, count in Counter(children).items()}


class DFTHCBlockEncoding(Operator2):
    r"""The DFTHC block encoding :math:`\mathrm{BE} = U^\dagger\,\mathrm{REF}_{T_2}\,U` (steps 1
    to 27): :math:`\langle 0|\mathrm{BE}|0\rangle = \tilde H_{SA}/\tilde\Lambda - I`.

    **Action.** With every group but ``system`` and ``phase_grad`` in :math:`|0\rangle`, the part
    of the output with every ancilla back in :math:`|0\rangle` is
    :math:`(\tilde H_{SA}/\tilde\Lambda - I)|\psi\rangle` for the system input :math:`|\psi\rangle`,
    with :math:`(\tilde H_{SA}, \tilde\Lambda)` = ``encoded_block(params, tables, factors)``
    (spec 8.5): the alias probabilities of the tables, the sign columns, the directions of the
    rounded angles, the identity term ``b = B``, the failed uniform preparation of ``b`` (acting as
    :math:`+I`) and that of ``x_o`` (where :math:`\mathrm{REF}_{T_2}` is :math:`-I`, which gives
    :math:`\tilde\Lambda = \Lambda/(1 - P_{fail}(x_o))`). :math:`\mathrm{BE}` is Hermitian and
    self-inverse. The rest of the output has a nonzero index register of
    :math:`\mathcal{A}\cup\mathcal{B}`. The output is the same for every outcome of the erase
    measurements.

    **Registers.** The groups of :class:`~dfthc.registers.WalkLayout` (spec 10.1), each a
    concatenation of named registers in layout order.

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``outer``      the ``outer`` group (143)      ``|0>``               see the action
    ``inner``      the ``inner`` group (455)      ``|0>``               see the action
    ``select``     ``(N - 1) b_rot + 4`` (799)    ``|0>``               ``|0>``
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** ``params``: :meth:`WalkParams.as_compilable`, the 13 fields; the
    erases' one-hot bits must fit their reset registers (:math:`2^{k_{e1}} \le K_1 d_1`,
    :math:`2^{k_{e2} + w_b} \le K_2 d_2`, :math:`2^{w_b} \le (N - 1) b_{rot} + 1`) and
    ``WalkLayout.work_size`` must cover every child. ``tables``:
    :meth:`WalkTables.as_compilable` of tables with the shapes of ``params``, in the ranges the
    prepares and the select accept. The constructor checks both (``ValueError``).

    **Pre- and postconditions.** The groups are pairwise disjoint (unchecked). ``phase_grad``
    holds the state ``PhaseGradientStatePrep`` prepares and keeps it. The block needs the ancilla
    groups in ``|0>``; in the walk the input is also any state the walk reaches from there, on
    which every erase's precondition holds (spec 3.3). Afterwards every register but the six
    index registers of :math:`\mathcal{A}\cup\mathcal{B}` (``x_o``, ``m_x_o``, ``sigma_1``,
    ``b``, ``m_b``, ``sigma_0``) is back in ``|0>``: the erases reset the loaded data, the
    measured uncomputes clean the comparators, and the adjoint uniform preparations clear the
    success flags. So ``RefAB``'s reflection about those six is the walk's reflection about every
    ancilla.

    **Decomposition** (spec 9.21). The groups are split with ``WalkLayout.split_group``; each
    child gets its registers by name and the prefix of ``work`` of its need: ``OuterPrepare``;
    ``InnerPrepare``, ``InnerSigns``, ``InnerSelect``, ``InnerUnprepare``; ``RefT2``;
    ``InnerPrepare``, ``InnerSelectDagger``, ``InnerSigns``, ``InnerUnprepare``;
    ``OuterUnprepare``. The prepare classes take the whole ``phase_grad`` (they use its top
    ``s_u`` wires), the selects exactly its ``b_rot`` wires; both selects get
    ``fixup_bits = w_b``. ``InnerSigns`` is diagonal and touches no register of the selects, so
    its place on either side of them is exact; this is spec 9.21's.

    **Adjoint and control.** Never adjointed: its rule measures (spec 7.4 rule 5), and the
    operator is its own inverse anyway. Never controlled: the controlled form is the class
    :class:`ControlledDFTHCBlockEncoding`, which controls only ``RefT2``; there is no
    ``C(DFTHCBlockEncoding)`` rule, so ``qp.ctrl`` of this operator does not compile (spec 7.4
    rule 6).

    **Cost.** :func:`dfthc.costs.block_encoding` (uncontrolled):
    :math:`T = T_{out} + T_{out}^\dagger + 2(T_{in} + T_{sel} + T_{in}^\dagger) + T_{RefT2}`,
    :math:`M = K_1 d_1 + 2(K_2 d_2 + (N - 1) b_{rot} + 3)`; ``A = T - (T_Toffoli - 1)`` with
    ``T_Toffoli`` = :func:`dfthc.costs.toffoli_only` (the walk's plain Toffolis less ``RefAB``'s
    one). At FeMoco-54: :math:`279 + 2 \cdot 4725 + 20 = 9749`, ``M = 2524``, ``A = 8632``.
    ``CNOT`` and ``PauliX`` counts are upper bounds (spec 7.6).

    **Paper and dfthcbe.** Steps 1 to 27 of spec 3.5: the sum-of-squares block encoding
    :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U` (PRX 15, 041016; arXiv:2502.15882v1 Eq. (11) and
    Fig. 1), built in App. B: the prepare over :math:`x_o` (B.1, Eq. (87)) and the step of
    oblivious amplitude amplification, whose reflection is controlled on the success of
    :math:`x_o` (B.2, Eq. (88)); the costs are its Table 3 (PRX Table III). dfthcbe:
    ``DFTHCBlockEncoding`` (``dfthcbe/block_encoding.py``), 9,596 Toffolis at FeMoco-54 by its
    call graph, :math:`212 + 2 (913 + 220) + 2 \cdot 3502 + 20 + 94`; ours is +153 (the inner
    prepare +58 and select +35 per round, the inner unprepare -3 per round, the outer prepare -2
    and unprepare -25; spec 13.1). The paper gives 9,997 per walk, 46 of them for
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}`; against the other 9,951 ours is -202: the
    Givens ladders -424 (``rz-addsub``), the 26-bit inner entries against its 20-bit ones +186,
    exact boundaries and one-hot padding +80 (``exact-iter``, ``no-prune``, ``onehot-pad``), the
    free comparator adjoints -46 (``cmp-free``) and the rest +2 (spec 13.1).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import DFTHCBlockEncoding
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p)
        groups = {name: getattr(layout, name) for name in DFTHCBlockEncoding.wire_argnames}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            DFTHCBlockEncoding(p.as_compilable(), t.as_compilable(), **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 150
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = _GROUPS

    def __init__(self, params, tables, outer, inner, select, system, phase_grad, work):
        wires = (outer, inner, select, system, phase_grad, work)
        compilable = _validate("DFTHCBlockEncoding", self.wire_argnames, params, tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """The layout's group sizes (spec 10.1); ``tables`` does not change them."""
        return _group_sizes(WalkParams.from_compilable(params))


def _block_encoding_resources(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_children(p, t, controlled=False), p)


@qp.register_resources(_block_encoding_resources)
def _block_encoding(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    _emit(_children(p, t, controlled=False),
          _registers(p, outer, inner, select, system, phase_grad), work)


register_rule(DFTHCBlockEncoding, _block_encoding)


class ControlledDFTHCBlockEncoding(Operator2):
    r"""The controlled block encoding: the explicit controlled class of
    :class:`DFTHCBlockEncoding` (spec 7.4 rule 6), :math:`U^\dagger\,C(\mathrm{REF}_{T_2})\,U`.

    **Action.** :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle 1|_{control}
    \otimes \mathrm{BE}` on inputs whose ancilla groups are in :math:`|0\rangle`, with
    :math:`\mathrm{BE}` the action of :class:`DFTHCBlockEncoding` for the same arguments: control
    ``|0>`` returns the whole input (:math:`U^\dagger U = I`), control ``|1>`` gives
    :class:`DFTHCBlockEncoding`'s whole output, global sign included. The output is the same for
    every outcome of the erase measurements.

    **Registers.** Those of :class:`DFTHCBlockEncoding` after a one-wire ``control``, which
    stays a control (only phases on it).

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``control``    1                              any                   a control
    ``outer``      the ``outer`` group (143)      ``|0>``               see the action
    ``inner``      the ``inner`` group (455)      ``|0>``               see the action
    ``select``     ``(N - 1) b_rot + 4`` (799)    ``|0>``               ``|0>``
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCBlockEncoding`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCBlockEncoding`, on both control
    branches; ``control`` is disjoint from the groups (unchecked).

    **Decomposition** (spec 9.21). :class:`DFTHCBlockEncoding`'s eleven children with
    ``ControlledRefT2(control, succ_x_o, b, m_b, sigma_0, work)`` in place of ``RefT2``: every
    other child is uncontrolled, the same operator with the same compilable arguments and
    register sizes as in :class:`DFTHCBlockEncoding`, so the measured unprepares are never
    controlled (spec 7.4 rule 5).

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0).

    **Cost.** :func:`dfthc.costs.block_encoding` (controlled): :class:`DFTHCBlockEncoding`'s
    ``T + 1`` (the controlled reflection's extra ``TemporaryAND``), the same ``M``,
    ``A = T - (T_Toffoli - 1)``. At FeMoco-54: ``T = 9750``, ``M = 2524``, ``A = 8633``.

    **Paper and dfthcbe.** The controlled block encoding inside the controlled walk
    :math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,U^\dagger\,
    C(\mathrm{REF}_{T_2})\,U` (spec 3.3). dfthcbe: ``DFTHCBlockEncoding(is_controlled=True)``,
    9,597 Toffolis at FeMoco-54, which also controls only ``ref_t2``.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import ControlledDFTHCBlockEncoding
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p, qpe_bits=1)
        groups = {name: getattr(layout, name)
                  for name in ControlledDFTHCBlockEncoding.wire_argnames[1:]}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            ControlledDFTHCBlockEncoding(p.as_compilable(), t.as_compilable(),
                                         layout.qpe_control, **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 150 + 1 = 151
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = ("control", *_GROUPS)

    def __init__(self, params, tables, control, outer, inner, select, system, phase_grad, work):
        wires = (control, outer, inner, select, system, phase_grad, work)
        compilable = _validate("ControlledDFTHCBlockEncoding", self.wire_argnames, params,
                               tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """``control`` (1 wire), then the layout's group sizes."""
        return {"control": 1} | _group_sizes(WalkParams.from_compilable(params))


def _controlled_block_encoding_resources(params, tables, control, outer, inner, select, system,
                                         phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_children(p, t, controlled=True), p)


@qp.register_resources(_controlled_block_encoding_resources)
def _controlled_block_encoding(params, tables, control, outer, inner, select, system, phase_grad,
                               work):
    p, t = _rebuild(params, tables)
    _emit(_children(p, t, controlled=True),
          _registers(p, outer, inner, select, system, phase_grad, control), work)


register_rule(ControlledDFTHCBlockEncoding, _controlled_block_encoding)
```

- [ ] **Step 4: Write the simulation harness**

Create `src/dfthc/testing/walk_sim.py`:

```python
r"""Sparse-simulator harness of the block encoding and the walk (spec 12.3 and 12.4).

A :class:`WalkCase` holds one walk's parameters, tables, factors and wires: the ``WalkLayout`` of
the parameters plus one more wire, ``control``, for the controlled classes (the uncontrolled
programs leave it in ``|0>``). A program (:func:`program`, :func:`operations`) prepares the
phase-gradient state on ``phase_grad`` and then applies the assembly operators; it is decomposed
once to gate-set operations on the tape path. :func:`prepare_unprepare` is the block encoding
without its reflection, :math:`U^\dagger U` (spec 12.4, E-4). An input (:func:`input_state`) is a
vector of amplitudes on the system ``psi_up (x) psi_dn`` (:func:`basis`, :func:`superposition`),
optionally times a control state, with every other wire in ``|0>``.

:func:`project` reads the block of an output: for each control value and system basis state, the
amplitude with every other wire in ``|0>`` and ``phase_grad`` contracted with the gradient state,
which is ``<0|BE|0>`` applied to the input (spec 12.4). :func:`assert_same_state` compares two
whole outputs, ancillas included, so a dirty ancilla, a wrong sign outside the block or a wrong
global phase under control shows. The expected blocks come from the encoded model
(:func:`dfthc.data.encoded.encoded_block`), never from the circuit. Every run is strict: each
erase measurement must have probability 1/2, and every AND precondition is asserted (spec 12.3).

:data:`CASES` names the shapes of the Phase 3 tests: the two test presets and
:data:`FAILS_C3`. At most 64 wires make a key one 64-bit word; wider programs (``H2_631G_TEST``:
80 wires) are correct but about 25 times slower per input (``np.unique`` over two-word keys).
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
from dfthc.ops.reflections import RefT2
from dfthc.ops.walk import _children, _emit, _registers
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, WalkParams
from dfthc.registers import GROUPS, WalkLayout
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator, SparseState
from dfthc.testing.tape import decompose_for_simulation

ATOL = 1e-12
"""Absolute tolerance of the state comparisons (HANDOFF section 6)."""

AB_REGISTERS = ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0")
r"""The registers of :math:`\mathcal{A} \cup \mathcal{B}` that ``RefAB`` reflects (spec 3.3)."""


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
        """The control wire of the controlled classes, right after the layout."""
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
        """The wire groups of an assembly operator, ``control`` first if ``controlled``."""
        groups = {name: list(getattr(self.layout, name)) for name in
                  ("outer", "inner", "select", "system", "phase_grad", "work")}
        return {"control": self.control} | groups if controlled else groups

    def block(self) -> np.ndarray:
        """The encoded block ``H~_SA / Lambda~ - I`` (spec 8.5)."""
        h_tilde, lambda_tilde = encoded_block(self.params, self.tables, self.factors)
        return h_tilde / lambda_tilde - np.eye(len(h_tilde))


FAILS_C3 = WalkParams(N=2, R=3, B=2, C=3, n_plus=1, b_rot=4, b_k1=1, b_k2=2, s_u=3, k1=0, k2=0,
                      ke1=1, ke2=1)
"""A shape the test presets cannot be (ledger rows 1A.5 and 2D.2), with ``random_factors(2, 3, 2,
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
"""The Phase 3 test shapes: ``H2_STO3G_TEST`` (64 wires with the control), ``H2_631G_TEST``
(80) and :data:`FAILS_C3` (59), each with its factors."""


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


def program(case: WalkCase, cls: type, controlled: bool = False, repeat: int = 1) -> list:
    """The decomposed program: the phase gradient, then ``cls`` (an assembly class) on the case's
    groups, ``repeat`` times."""
    def call():
        cls(*case.compilable, **case.groups(controlled))

    return operations(case, *[call] * repeat)


def prepare_unprepare(case: WalkCase) -> list:
    r"""The decomposed :math:`U^\dagger U`: the block encoding's children (``dfthc.ops.walk``'s
    own list, in its order and on its registers) without ``RefT2`` (spec 12.4, E-4)."""
    children = [child for child in _children(case.params, case.tables, controlled=False)
                if child[0] is not RefT2]
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
    """``RefAB``'s exact action on ``state``: -1 on every term with a nonzero
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
```

- [ ] **Step 5: Run the tests**

Run: `scripts/test.sh tests/ops/test_block_encoding.py tests/test_work_pool.py -q`
Expected: `52 passed` (about 49 s with 2 workers, the 2 `slow` compiles included; alone under `run_watched` the `MID_NAPHTHALENE` counts took 5.9 and 6.9 s at 613 and 597 MB).

Run: `scripts/test.sh tests/test_tolerance_hygiene.py tests/test_compile.py -q`
Expected: `67 passed` (the tolerance guard covers `walk_sim.py`; the pins of the new keys).

- [ ] **Step 6: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1025 passed` (975 + 50; see "Expected test counts").

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/dfthc/ops/walk.py src/dfthc/testing/walk_sim.py tests/ops/test_block_encoding.py tests/test_work_pool.py
git commit -m "Add DFTHCBlockEncoding and ControlledDFTHCBlockEncoding with the walk simulation harness"
```

---

### Task 3.4: `DFTHCWalk`, `ControlledDFTHCWalk`, whole-program counts and the inventory test

**Task id:** `3.4` · **Branch:** `phase3/3.4` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 3.3

**Files:**
- Modify: `src/dfthc/ops/walk.py` (replace it with the version below: Task 3.3's file with the module docstring's title and two paragraphs extended to the walk, `RefAB` imported, and the section from `_walk_children` on appended; `stages/3A/make_stage_33.py` derives Task 3.3's file from this one by exactly these replacements)
- Modify: `src/dfthc/testing/resources.py` (replace it with the version below; the diff against `main` is under "Shared edits")
- Modify: `tests/test_work_pool.py` (replace it with the version below: the walks' rules added to `RULES`, the pool test reading `ControlledRefAB`'s need from the controlled walk's rule, the random-shape test taking the walks' children from `_walk_children`)
- Create: `tests/ops/test_walk.py`, `tests/integration/test_walk_counts.py`

**Interfaces:**
- Consumes: Task 3.3's `dfthc.ops.walk` (`DFTHCBlockEncoding`, `ControlledDFTHCBlockEncoding`, `_rebuild`, `_group_sizes`, `_validate`, `_registers`, `_emit`, `_resources`) and `dfthc.testing.walk_sim`; `dfthc.ops.reflections` (`RefAB`, `ControlledRefAB`); `dfthc.costs` (`walk`, `measured_and_uncomputes`, `toffoli_only`, `ref_ab`); `dfthc.compile` (`compiled_resources`, `dfthc_operator_names`, `FIXED_DECOMPS`); `dfthc.testing.watchdog` (`run_watched`, `effective_limit_mb`); `dfthc.ops.prepare.InnerSigns` (the inventory's negative control).
- Produces, in `dfthc.ops.walk`:
  - `DFTHCWalk(params, tables, outer, inner, select, system, phase_grad, work)`: the block encoding's arguments, checks and `required_wires`; the rule `_walk` (`DFTHCBlockEncoding` on the whole groups and pool, then `RefAB(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work[:n_w - 3])`);
  - `ControlledDFTHCWalk(params, tables, control, outer, inner, select, system, phase_grad, work)`: the rule `_controlled_walk` (`ControlledDFTHCBlockEncoding`, then `ControlledRefAB(control, ..., work[:n_w - 2])`); there is no `C(DFTHCWalk)` rule;
  - `_walk_children(p, t, controlled)`. Neither class is ever adjointed or controlled.
- Produces, in `dfthc.testing.resources`: `captured_targets(qfunc, num_wires) -> list[str]` (every converted rule target of one `compiled_resources` compile, in order, with its modifiers and without the attribute dump); `LEAF_ADJOINTS`, `TEMPLATE_ADJOINTS`; `modified_targets(names, bases) -> set[str]`; `inventory(names) -> set[str]` (the modified targets of dfthc operators and of the templates of `TEMPLATE_ADJOINTS`).

**Acceptance tests** (30 new tests, 2 of them `slow`, and the replaced `test_work_pool.py`'s 9):
- T3 on the tape path (4 tests): at `sto3g` and `fails-c3` the walk's whole output equals `DFTHCBlockEncoding`'s with `RefAB`'s exact action applied, and its block is H̃_SA/Λ̃ − I; `ControlledDFTHCWalk` at `sto3g` with the control in |0⟩, |1⟩ and |+⟩; after one uncontrolled walk step (the index registers no longer |0⟩), the control in |+⟩ gives (|0⟩W|ψ⟩ + |1⟩W²|ψ⟩)/√2.
- The children (2 tests): the (controlled) block encoding on the whole groups, the control and the whole pool; the (controlled) `RefAB` on the six registers and the pool's prefix of n_w − 3 (n_w − 2).
- T6, T5 and the inventory (6 tests): the listing pairs `DFTHCWalk`/`ControlledDFTHCWalk`, the block encodings, the reflections and the selects with their pinned rules, `Adjoint(MajoranaGate)` with a copy of `_majorana_gate`, and has no `C(ZeroReflection)`; nothing dfthc survives at `level="device"` in a program holding both walks; the inventory of each walk program holds its own top-level target once and every Phase 3 class of the program, and its modified dfthc and template targets are exactly `LEAF_ADJOINTS | TEMPLATE_ADJOINTS`; a program with `qp.adjoint(InnerSigns(...))` shows `Adjoint(InnerSigns)` (negative control).
- The failing generic control (1 test): `qp.ctrl(DFTHCWalk(...))` fails with `CompileError` "Decomposition rule not found for operator 'C(DFTHCWalk)'" after the expected measuring-rule warnings only (the Task G0.1 probe).
- T2 (3 tests), `required_wires` (1 test), constructor validation (6 tests).
- `tests/integration/test_walk_counts.py` (7 tests, 2 `slow`): the walk and controlled walk compiled each as one program on `null.qubit` at both test presets, T, M and A against `costs.walk` and `costs.measured_and_uncomputes`, the plain `Toffoli` count `toffoli_only`, non-symbolic and closed (4); at `MID_NAPHTHALENE` (`slow`, `subprocess`): each compile in a subprocess under `run_watched` with a 4 GB cap, the same equalities, the watched time and peak on a `MID_COMPILE` line (2); FeMoco-54 on the closed forms: 9,793 and 9,795, M = 2,524, T_Toffoli = 1,118, A = 8,675 and 8,677, M + A = 11,199 and 11,201 (1).
- `tests/test_work_pool.py` (9 tests): as in Task 3.3, for all four rules; the pool is `ControlledRefAB`'s need, read from the controlled walk's rule, and `RefAB`'s is one less; the random-shape test also takes the walks' children from `_walk_children`.

**Definition of done:** `82 passed` for the four files targeted (28 new per-commit tests); the slow `MID_NAPHTHALENE` cases pass and their `MID_COMPILE` numbers are in the report (each about 10 s and 0.6 GB; the orchestrator copies them into the ledger); the per-commit tier passes; `ruff check .` clean; the spec 9.0 docstring sections in order; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_walk.py`:

```python
"""DFTHCWalk and ControlledDFTHCWalk (spec 9.22): the walk against the block encoding and the
encoded model on the tape path, the three control states, the children, rule selection, the
inventory of captured targets, the failing qp.ctrl(DFTHCWalk), validation."""

import re
import warnings

import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc.compile import FIXED_DECOMPS, compiled_resources
from dfthc.ops.prepare import InnerSigns
from dfthc.ops.walk import (
    ControlledDFTHCBlockEncoding,
    ControlledDFTHCWalk,
    DFTHCBlockEncoding,
    DFTHCWalk,
    _controlled_walk,
    _walk,
)
from dfthc.params import H2_STO3G_TEST
from dfthc.testing import walk_sim
from dfthc.testing.resources import (
    LEAF_ADJOINTS,
    TEMPLATE_ADJOINTS,
    captured_targets,
    classify_rule_warning,
    inventory,
    verbose_program,
)

CLASSES = {"plain": (DFTHCWalk, _walk), "controlled": (ControlledDFTHCWalk, _controlled_walk)}
SUPPORT = {"sto3g": (5, 10), "fails-c3": (5, 12)}
"""Two-electron system inputs (psi_up's bits first), as in test_block_encoding.py."""


# --------------------------------------------------------------------------- semantics


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_walk_is_ref_ab_after_the_block_encoding(name):
    """T3 on the tape path (strict sparse simulator): the walk's whole output equals
    DFTHCBlockEncoding's whole output with RefAB's exact action applied (-1 on every term with a
    nonzero x_o, m_x_o, sigma_1, b, m_b or sigma_0; spec 3.3), so its block is
    H~_SA / Lambda~ - I (spec 12.4, E-2 at k = 1), under a seeded schedule."""
    c = walk_sim.case(name)
    psi = walk_sim.superposition(c, SUPPORT[name], seed=7)
    walk_ops, be_ops = walk_sim.program(c, DFTHCWalk), walk_sim.program(c, DFTHCBlockEncoding)
    outcomes = walk_sim.schedule("seeded", walk_sim.num_measurements(walk_ops), seed=8)
    walked = walk_sim.simulate(c, walk_ops, walk_sim.input_state(c, psi), outcomes)
    encoded = walk_sim.simulate(c, be_ops, walk_sim.input_state(c, psi), outcomes)
    walk_sim.assert_same_state(walked, walk_sim.reflect_ab(c, encoded))
    assert np.allclose(walk_sim.project(c, walked)[0], c.block() @ psi, rtol=0, atol=1e-12)


def test_controlled_walk_in_three_control_states():
    """T3 of the controlled class at H2_STO3G_TEST: control |0> returns the whole input, control
    |1> gives DFTHCWalk's whole output, and control |+> their superposition with weight 1/sqrt 2
    each; a wrong global phase of either controlled reflection fails the last (spec 3.3)."""
    c = walk_sim.case("sto3g")
    psi = walk_sim.basis(c, 10)
    plain = walk_sim.program(c, DFTHCWalk)
    controlled = walk_sim.program(c, ControlledDFTHCWalk, controlled=True)
    outcomes = walk_sim.schedule("alternating", walk_sim.num_measurements(plain))
    identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    applied = walk_sim.set_control(c, walk_sim.simulate(c, plain, walk_sim.input_state(c, psi),
                                                        outcomes))
    plus = 2**-0.5
    expected = {"0": identity, "1": applied,
                "+": walk_sim.superpose((plus, identity), (plus, applied))}
    for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", [plus, plus])):
        out = walk_sim.simulate(c, controlled, walk_sim.input_state(c, psi, control), outcomes)
        walk_sim.assert_same_state(out, expected[label])


def test_controlled_walk_after_a_walk_step():
    """T3 of the controlled class on the walk's own states, as iterative QPE uses it (spec 11.1):
    after one DFTHCWalk step the index registers of A u B are not |0>, and the controlled walk
    with the control in |+> gives (|0> W|psi> + |1> W^2|psi>)/sqrt 2. From ancillas in |0> an
    uncontrolled RefAB would act as the identity on the control-0 branch; here it would not."""
    c = walk_sim.case("sto3g")
    psi = walk_sim.basis(c, 10)
    once = walk_sim.program(c, DFTHCWalk)
    twice = walk_sim.program(c, DFTHCWalk, repeat=2)
    m = walk_sim.num_measurements(once)
    then_controlled = walk_sim.operations(
        c, lambda: DFTHCWalk(*c.compilable, **c.groups()),
        lambda: ControlledDFTHCWalk(*c.compilable, **c.groups(True)))
    outcomes = walk_sim.schedule("seeded", 2 * m, seed=9)
    stepped = walk_sim.simulate(c, once, walk_sim.input_state(c, psi), outcomes[:m])
    stepped_twice = walk_sim.simulate(c, twice, walk_sim.input_state(c, psi), outcomes)
    plus = 2**-0.5
    out = walk_sim.simulate(c, then_controlled, walk_sim.input_state(c, psi, [plus, plus]),
                            outcomes)
    assert walk_sim.nonzero_registers(c, stepped)
    walk_sim.assert_same_state(out, walk_sim.superpose(
        (plus, stepped), (plus, walk_sim.set_control(c, stepped_twice))))


def registers(op):
    return {name: list(getattr(op, name)) for name in op.wire_argnames}


@pytest.mark.parametrize("kind", CLASSES)
def test_children(kind):
    """Spec 9.22: the (controlled) block encoding on the whole groups and pool, then the
    (controlled) RefAB on x_o, m_x_o, sigma_1, b, m_b, sigma_0 and the pool's prefix of exactly
    its need; the control goes to both children and nowhere else."""
    c = walk_sim.case("sto3g")
    controlled = kind == "controlled"
    _, rule = CLASSES[kind]
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(*c.compilable, **c.groups(controlled))
    be, ref = queue.queue
    layout = c.layout
    assert (be.name, ref.name) == (("ControlledDFTHCBlockEncoding", "ControlledRefAB")
                                   if controlled else ("DFTHCBlockEncoding", "RefAB"))
    assert registers(be) == c.groups(controlled)
    assert be.hyperparameters == {"params": c.compilable[0], "tables": c.compilable[1]}
    ab = {name: list(getattr(layout, name)) for name in walk_sim.AB_REGISTERS}
    work = c.params.w_x + c.params.b_k1 + c.params.w_b + c.params.b_k2 - (0 if controlled else 1)
    assert registers(ref) == ({"control": c.control} if controlled else {}) | ab | {
        "work": list(layout.work)[:work]}


# --------------------------------------------------------------------------- rules, inventory


def chosen_by_operator(listing: str) -> dict[str, set[str]]:
    """``Operator: <name>{...}`` and the ``Chosen Rule: __builtin_<rule>`` line after it, as
    ``{name: {rule, ...}}`` (a verbose listing names each operator signature once)."""
    chosen, operator = {}, None
    for line in listing.splitlines():
        line = line.strip()
        if line.startswith("Operator: "):
            operator = line[len("Operator: "):].split("{", 1)[0]
        elif line.startswith("Chosen Rule: __builtin_") and operator is not None:
            chosen.setdefault(operator, set()).add(line[len("Chosen Rule: __builtin_"):])
            operator = None
    return chosen


@pytest.mark.compiled
@pytest.mark.parametrize("kind", CLASSES)
def test_pinned_rules_are_chosen(kind, capfd):
    """T6 at H2_STO3G_TEST: the listing pairs each assembly operator with its pinned rule, the
    reflections with theirs (no C(ZeroReflection)), and Adjoint(MajoranaGate), the select
    dagger's leaf adjoint, with a lowered copy of the base rule _majorana_gate (ledger row
    2E.5)."""
    c = walk_sim.case("sto3g")
    controlled = kind == "controlled"
    cls, _ = CLASSES[kind]

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        cls(*c.compilable, **c.groups(controlled))

    qp.analyze(verbose_program(program, c.num_wires), level="user")()
    captured = capfd.readouterr()
    chosen = chosen_by_operator(captured.out + captured.err)
    prefix = "Controlled" if controlled else ""
    expected = {f"{prefix}DFTHCWalk": "_controlled_walk" if controlled else "_walk",
                f"{prefix}DFTHCBlockEncoding": ("_controlled_block_encoding" if controlled
                                                else "_block_encoding"),
                f"{prefix}RefT2": "_controlled_ref_t2" if controlled else "_ref_t2",
                f"{prefix}RefAB": "_controlled_ref_ab" if controlled else "_ref_ab",
                "InnerSelect": "_inner_select", "InnerSelectDagger": "_inner_select_dagger"}
    assert {name: chosen[name] for name in expected} == {
        name: {rule} for name, rule in expected.items()}
    assert all(re.fullmatch(r"_majorana_gate(_\d+)?", rule)
               for rule in chosen["Adjoint(MajoranaGate)"])
    assert "C(ZeroReflection)" not in chosen


@pytest.mark.compiled
def test_nothing_survives_at_device_level():
    """T5 at level="device" for the H2 walk (L-20): no dfthc operator survives in a program
    holding the walk and the controlled walk."""
    c = walk_sim.case("sto3g")

    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        DFTHCWalk(*c.compilable, **c.groups())
        ControlledDFTHCWalk(*c.compilable, **c.groups(True))

    compiled_resources(program, c.num_wires, level="device")


@pytest.mark.compiled
@pytest.mark.parametrize("kind", CLASSES)
def test_inventory_of_captured_targets(kind):
    """The inventory test (spec 7.4 rule 6, 12.1 T6; master plan section 3, item 20): compiling
    the walk or the controlled walk at H2_STO3G_TEST, the frontend converts the program's own
    top-level target (so the hook sees targets) and every Phase 3 class of the program, and no
    Adjoint(...) or C(...) target of a dfthc composite or QROAMRead: the modified targets of dfthc
    operators and of the templates they adjoint are exactly the leaf adjoints and the template
    adjoints."""
    c = walk_sim.case("sto3g")
    controlled = kind == "controlled"
    cls, _ = CLASSES[kind]
    names = captured_targets(lambda: (qp.PhaseGradientStatePrep(c.layout.phase_grad),
                                      cls(*c.compilable, **c.groups(controlled))), c.num_wires)
    prefix = "Controlled" if controlled else ""
    assert {f"{prefix}DFTHCWalk", f"{prefix}DFTHCBlockEncoding", "InnerSelect",
            "InnerSelectDagger", f"{prefix}RefT2", f"{prefix}RefAB"} <= set(names)
    assert names.count(f"{prefix}DFTHCWalk") == 1
    assert inventory(names) == LEAF_ADJOINTS | TEMPLATE_ADJOINTS


@pytest.mark.compiled
def test_inventory_flags_a_stray_modifier():
    """Negative control: one qp.adjoint of a dfthc operator outside the leaves (InnerSigns, whose
    pinned self-adjoint rule compiles) appears as Adjoint(InnerSigns) in the inventory."""
    names = captured_targets(lambda: qp.adjoint(InnerSigns(sgn=[0], d_sgn=[1],
                                                           less_than_b=[2])), 3)
    assert inventory(names) == {"Adjoint(InnerSigns)"}


@pytest.mark.compiled
def test_controlled_walk_has_no_generic_controlled_rule():
    """Spec 7.4 rule 6 and 9.22, the Task G0.1 probe: qp.ctrl(DFTHCWalk(...)) at H2_STO3G_TEST
    fails to compile with a CompileError naming C(DFTHCWalk), after Catalyst's expected
    "Skipped ... mid-circuit measurement" warnings for the erases (rule 12)."""
    c = walk_sim.case("sto3g")
    op = DFTHCWalk(*c.compilable, **c.groups())
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                "Decomposition rule not found for operator 'C(DFTHCWalk)")):
            compiled_resources(lambda: qp.ctrl(op, control=c.control[0]), c.num_wires)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the ")]
    named = [message for message in skipped if classify_rule_warning(message) != "noise"]
    assert named and {classify_rule_warning(message) for message in named} == {"expected"}


@pytest.mark.parametrize("name", ["sto3g", "631g", "fails-c3"])
def test_rules_are_valid(name):
    """T2 (tape mode) for both rules at three sizes; each pinned and the only rule under its
    class name (spec 7.4 rules 7 and 8)."""
    c = walk_sim.case(name)
    for kind, (cls, rule) in CLASSES.items():
        assert FIXED_DECOMPS[cls.__name__] is rule
        assert list(qp.list_decomps(cls)) == [rule]
        op = cls(*c.compilable, **c.groups(kind == "controlled"))
        qp.testing.assert_valid_decomp_rule(op, rule, check_matrix=False)


def test_required_wires():
    """The block encoding's groups: the walk adds no register (RefAB's work is in the pool)."""
    p = H2_STO3G_TEST
    for plain, controlled in ((DFTHCWalk, ControlledDFTHCWalk),
                              (DFTHCBlockEncoding, ControlledDFTHCBlockEncoding)):
        sizes = plain.required_wires(p.as_compilable(), ())
        assert sizes == {"outer": 16, "inner": 23, "select": 8, "system": 4, "phase_grad": 4,
                         "work": 8}
        assert controlled.required_wires(p.as_compilable(), ()) == {"control": 1} | sizes


@pytest.mark.parametrize(("change", "match"), [
    ({"params": 7}, "params and tables must be"),
    ({"tables": ()}, "WalkTables needs 8 fields; got 0"),
    ({"outer": [0]}, "'outer' has 1 wires; expected 16"),
], ids=["params-type", "tables", "outer"])
@pytest.mark.parametrize("kind", CLASSES)
def test_constructor_validates(kind, change, match):
    """Spec 9.0: the walks run the block encoding's checks (test_block_encoding.py has the full
    set) and raise ValueError."""
    c = walk_sim.case("sto3g")
    cls, _ = CLASSES[kind]
    params, tables = c.compilable
    arguments = {"params": params, "tables": tables} | c.groups(kind == "controlled") | change
    with pytest.raises(ValueError, match=match):
        cls(**arguments)
```

Create `tests/integration/test_walk_counts.py` (a new directory, no `__init__.py`):

```python
"""Whole-program counts of the walk and the controlled walk (spec 9.22 and 9.24; master plan Task
3.4): each compiles as one program, its T, M and A equal the closed forms exactly at the test
presets (per commit) and at MID_NAPHTHALENE (nightly, each compile in a subprocess under
run_watched, which records its time and memory), and the FeMoco-54 values on the closed forms.

``python tests/integration/test_walk_counts.py PROGRAM`` compiles one MID_NAPHTHALENE program
(``walk`` or ``controlled_walk``) and prints its counts as JSON on a line prefixed ``RESULT``.
"""

import json
import os
import sys
import time
from pathlib import Path

import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import compiled_resources
from dfthc.data.factors import load_factors
from dfthc.ops.walk import ControlledDFTHCWalk, DFTHCWalk
from dfthc.params import FEMOCO54, MID_NAPHTHALENE
from dfthc.testing import walk_sim
from dfthc.testing.watchdog import effective_limit_mb, run_watched

REPO = Path(__file__).resolve().parents[2]
PROGRAMS = {"walk": (DFTHCWalk, False), "controlled_walk": (ControlledDFTHCWalk, True)}
CAP_MB = 4096
"""The MID_NAPHTHALENE compiles' cap: the watchdog kills at 3,840 MB (master plan Task 3.4)."""
PREFIX = "RESULT "


def compile_program(c, program):
    cls, controlled = PROGRAMS[program]

    def qfunc():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        cls(*c.compilable, **c.groups(controlled))

    return compiled_resources(qfunc, c.num_wires)


def assert_counts(resources, p, controlled):
    """T, M and A against costs.walk and costs.measured_and_uncomputes, the plain Toffolis
    against costs.toffoli_only (spec 9.24); compiled_resources has asserted non-symbolic counts
    and gate-set closure."""
    expected = costs.walk(p, controlled)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measured_and_uncomputes == costs.measured_and_uncomputes(p, controlled)
    assert resources.counts["Toffoli"] == costs.toffoli_only(p)


@pytest.mark.compiled
@pytest.mark.parametrize("program", PROGRAMS)
@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_counts_at_the_test_presets(name, program):
    """T = 158 / 160 and 240 / 242, M = 50 and 68, A = 108 / 110 and 179 / 181."""
    c = walk_sim.case(name)
    assert_counts(compile_program(c, program), c.params, PROGRAMS[program][1])


def compile_mid(program: str) -> dict:
    """One MID_NAPHTHALENE compile (subprocess side): its counts and compile time."""
    c = walk_sim.WalkCase.build(MID_NAPHTHALENE, load_factors("naphthalene"))
    start = time.perf_counter()
    resources = compile_program(c, program)
    return {"program": program, "seconds": round(time.perf_counter() - start, 1),
            "toffoli": resources.toffoli, "measurements": resources.measurements,
            "measured_and_uncomputes": resources.measured_and_uncomputes,
            "plain_toffoli": resources.counts["Toffoli"], "num_wires": c.num_wires}


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize("program", PROGRAMS)
def test_counts_at_mid_naphthalene(program):
    """T4 at MID_NAPHTHALENE (nightly), each program compiled once in a subprocess under
    run_watched with a 4 GB cap: T = 1,913 / 1,915, M = 502, A = 1,678 / 1,680. The MID_COMPILE
    line records the watched wall time and peak memory, for comparison with the K-1 stand-in at
    the same sizes (spikes/results/k1_mid10.json and k1d_PX_mid10.json)."""
    command = [sys.executable, str(Path(__file__).resolve()), program]
    env = os.environ | {"PYTHONPATH": str(REPO / "src"),
                        "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "0"}
    run = run_watched(command, cap_mb=CAP_MB, timeout_s=900, cwd=REPO, env=env)
    lines = [line for line in run.stdout.splitlines() if line.startswith(PREFIX)]
    result = json.loads(lines[0][len(PREFIX):]) if len(lines) == 1 else None
    print("MID_COMPILE " + json.dumps({"watchdog": run.summary(), "result": result}))
    assert run.killed is None and not run.survivors and run.returncode == 0, run.stderr[-2000:]
    assert run.peak_mb <= effective_limit_mb(CAP_MB)
    assert result is not None, run.stdout[-2000:]
    controlled = PROGRAMS[program][1]
    expected = costs.walk(MID_NAPHTHALENE, controlled)
    assert (result["toffoli"], result["measurements"], result["measured_and_uncomputes"],
            result["plain_toffoli"]) == (
        expected.toffoli, expected.measurements,
        costs.measured_and_uncomputes(MID_NAPHTHALENE, controlled),
        costs.toffoli_only(MID_NAPHTHALENE))


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 (spec 9.22 and 9.24), without compiling (master plan section 12.9; the FeMoco
    compile is Task 4.5's): T = 9,749 + 44 = 9,793, controlled 9,750 + 45 = 9,795; M = 2,524;
    T_Toffoli = 18 + 20 + 72 + 780 + 216 + 12 = 1,118; A = 8,675 and 8,677; M + A = 11,199 and
    11,201."""
    p = FEMOCO54
    assert [costs.ref_ab(p, c).toffoli for c in (False, True)] == [44, 45]
    assert costs.walk(p, False) == costs.Cost(9793, 2524)
    assert costs.walk(p, True) == costs.Cost(9795, 2524)
    assert [2 * p.w_x, 4 * p.w_b, (p.K1 - 1) * p.d1, 2 * (p.K2 - 1) * p.d2, 4 * p.N, 12] == [
        18, 20, 72, 780, 216, 12]
    assert costs.toffoli_only(p) == 1118
    assert [costs.measured_and_uncomputes(p, c) for c in (False, True)] == [8675, 8677]
    assert [costs.walk(p, c).measurements + costs.measured_and_uncomputes(p, c)
            for c in (False, True)] == [11199, 11201]


if __name__ == "__main__":
    print(PREFIX + json.dumps(compile_mid(sys.argv[1])), flush=True)
```

Replace `tests/test_work_pool.py` with:

```python
"""The walk's shared work pool (spec 10.1 and 10.2): every call of the four assembly rules (the
walks' and the block encodings') gets exactly its required_wires, its work is a prefix of the
pool, and neither the pool nor the comparators' dirty work registers appear where they do not
belong; WalkLayout.work_size is the largest need, the controlled RefAB's."""

import functools
import inspect
import random

import pennylane as qp
import pytest

from dfthc.data.factors import load_factors, random_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefAB, RefT2
from dfthc.ops.walk import (
    _block_encoding,
    _children,
    _controlled_block_encoding,
    _controlled_walk,
    _walk,
    _walk_children,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams, lg
from dfthc.registers import WalkLayout

PRESETS = {
    "sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "mid": lambda: (MID_NAPHTHALENE, load_factors("naphthalene")),
    "femoco": lambda: (FEMOCO54, random_factors(54, 10, 27, 27, n_plus=27, seed=0)),
}
"""FeMoco-54 is classical here: the rules run in tape mode and construct their children (the
constructors check every register); nothing is compiled."""

DIRTY = {"succ_work_x_o": {"OuterPrepare", "OuterUnprepare"},
         "cmp_work_x_o": {"OuterPrepare", "OuterUnprepare"},
         "succ_work_b": {"InnerPrepare", "InnerUnprepare"},
         "cmp_work_b": {"InnerPrepare", "InnerUnprepare"}}
"""The comparators' work registers, dirty from a prepare to its unprepare (spec 10.2), and the
only classes that may take them."""

AT_LEAST = {"OuterPrepare", "OuterUnprepare", "InnerPrepare", "InnerUnprepare"}
"""Their ``phase_grad`` is a minimum (``s_u``); the walk passes the whole register."""

RULES = {"block encoding": (_block_encoding, False, 11),
         "controlled block encoding": (_controlled_block_encoding, True, 11),
         "walk": (_walk, False, 2), "controlled walk": (_controlled_walk, True, 2)}
"""Each assembly rule, whether it is controlled, and its number of calls."""


@functools.cache
def preset(name):
    p, factors = PRESETS[name]()
    return p, WalkTables.from_factors(factors, p), WalkLayout.from_params(p, qpe_bits=1)


def required(op) -> dict:
    """``required_wires`` of the call: from its compilable arguments, or, for the reflections,
    whose sizes follow from their wires, from the widths of its pattern registers."""
    cls = type(op)
    if cls in (RefT2, ControlledRefT2):
        return cls.required_wires(len(op.b), len(op.m_b))
    if cls in (RefAB, ControlledRefAB):
        return cls.required_wires(len(op.x_o), len(op.m_x_o), len(op.b), len(op.m_b))
    names = inspect.signature(cls.required_wires).parameters
    return cls.required_wires(**{k: v for k, v in op.hyperparameters.items() if k in names})


def calls(rule, p, t, layout, controlled):
    """The rule's children in tape mode, on the layout's groups (``qpe_control`` as control)."""
    groups = {name: getattr(layout, name)
              for name in ("outer", "inner", "select", "system", "phase_grad", "work")}
    if controlled:
        groups = {"control": layout.qpe_control} | groups
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(p.as_compilable(), t.as_compilable(), **groups)
    return list(queue.queue)


def check_call(op, layout) -> int:
    """Assert the pool discipline for one call and return its work need."""
    sizes = required(op)
    pool = set(layout.work)
    registers = {name: list(getattr(op, name)) for name in op.wire_argnames}
    assert set(registers) == set(sizes), op.name
    for name, wires in registers.items():
        if name == "phase_grad" and op.name in AT_LEAST:
            assert wires == list(layout.phase_grad) and len(wires) >= sizes[name], op.name
        else:
            assert len(wires) == sizes[name], (op.name, name)
        if name == "work":
            assert wires == list(layout.work)[:len(wires)], op.name
        else:
            assert not pool & set(wires), (op.name, name)
        if name in DIRTY:
            assert op.name in DIRTY[name] and wires == list(getattr(layout, name))
    work = set(registers.get("work", ()))
    others = {wire for name, wires in registers.items() if name != "work" for wire in wires}
    dirty = {wire for name in DIRTY for wire in getattr(layout, name)}
    assert not work & others and not work & dirty, op.name
    return sizes.get("work", 0)


@pytest.mark.parametrize("name", PRESETS)
def test_every_call_gets_exactly_its_required_wires(name):
    """Spec 10.2 and master plan Tasks 3.3 and 3.4: in the four assembly rules every child
    receives exactly required_wires (phase_grad of the prepare classes is a minimum: they get the
    whole register; the walks' block encoding gets the whole groups and pool), its work is the
    pool's prefix of that size and disjoint from its other registers, no other register holds a
    pool wire, the dirty comparator work registers go only to their prepare and unprepare (never
    into a work prefix), and every need fits the pool."""
    p, t, layout = preset(name)
    for rule, controlled, count in RULES.values():
        needs = [check_call(op, layout) for op in calls(rule, p, t, layout, controlled)]
        assert len(needs) == count and max(needs) <= len(layout.work)


@pytest.mark.parametrize("name", PRESETS)
def test_the_pool_is_the_controlled_walks_largest_need(name):
    """WalkLayout.work_size is the largest need of the controlled walk's operators below the
    block encoding: the controlled RefAB's w_x + b_k1 + w_b + b_k2 at every preset (44 at
    FeMoco-54, spec 10.3); the uncontrolled walk's RefAB needs one wire less."""
    p, t, layout = preset(name)
    needs = {}
    for rule, controlled, _ in RULES.values():
        for op in calls(rule, p, t, layout, controlled):
            if op.name not in ("DFTHCBlockEncoding", "ControlledDFTHCBlockEncoding"):
                needs[op.name] = max(needs.get(op.name, 0), check_call(op, layout))
    assert max(needs.values()) == needs["ControlledRefAB"] == len(layout.work)
    assert needs["ControlledRefAB"] == WalkLayout.work_size(p) == p.w_x + p.b_k1 + p.w_b + p.b_k2
    assert needs["RefAB"] == needs["ControlledRefAB"] - 1


def own_work(cls, args, p) -> int:
    """A child's own work need: its class's ``required_wires(...)["work"]`` from the compilable
    arguments the rule gives it, or, for the reflections (sizes from their wires), from the
    pattern widths; 0 without a work register."""
    if cls in (RefT2, ControlledRefT2):
        return cls.required_wires(p.w_b, p.b_k2)["work"]
    if cls in (RefAB, ControlledRefAB):
        return cls.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)["work"]
    arguments = dict(zip(cls.compilable_argnames, args, strict=True))
    names = inspect.signature(cls.required_wires).parameters
    return cls.required_wires(**{k: v for k, v in arguments.items() if k in names}).get("work", 0)


def zero_tables(p) -> WalkTables:
    """Tables of ``p``'s shapes with every entry 0: the work needs depend on the shapes only."""
    def rows(n, m):
        return tuple((0,) * m for _ in range(n))

    inner = rows(p.RC, p.B + 1)
    return WalkTables(outer_alt=(0,) * p.X_o, outer_keep=(0,) * p.X_o, inner_alt=inner,
                      inner_keep=inner, inner_sgn=inner, inner_d_sgn=inner,
                      tail_angles=rows(p.N, p.N - 1),
                      sf_angles=tuple(rows(p.B, p.N - 1) for _ in range(p.R)))


CHILDREN = {"OuterPrepare", "InnerPrepare", "InnerSigns", "InnerSelect", "InnerUnprepare",
            "RefT2", "ControlledRefT2", "InnerSelectDagger", "OuterUnprepare",
            "DFTHCBlockEncoding", "ControlledDFTHCBlockEncoding", "RefAB", "ControlledRefAB"}
"""Every class the four assembly rules call."""


def test_work_size_covers_every_child_at_random_shapes():
    """At 300 random valid shapes, every child of the four assembly rules, taken from the rules'
    own child lists (``_children`` and ``_walk_children``, plain and controlled), is given exactly
    its class's own ``required_wires`` work, so no explicit inverse or controlled class
    (``OuterUnprepare``, ``InnerUnprepare``, ``InnerSelectDagger``, ``ControlledRefT2``,
    ``ControlledRefAB``) can drift from its forward form unnoticed; and ``WalkLayout.work_size``
    (spec 10.1, provisional until this test) covers every need."""
    rng = random.Random(0)
    checked, seen = 0, set()
    while checked < 300:
        N, R, B, C = rng.randint(2, 12), rng.randint(1, 10), rng.randint(1, 30), rng.randint(1, 30)
        b_rot = rng.randint(3, 16)
        values = dict(N=N, R=R, B=B, C=C, n_plus=rng.randint(0, N), b_rot=b_rot,
                      b_k1=rng.randint(1, 16), b_k2=rng.randint(1, 16), s_u=rng.randint(3, b_rot))
        values |= dict(k1=rng.randint(0, lg(N + R * C)), k2=rng.randint(0, lg(B + 1)),
                       ke1=rng.randint(0, lg(N + R * C)), ke2=rng.randint(0, lg(R * C)))
        p = WalkParams(**values)
        if (2**p.ke1 > p.K1 * p.d1 or 2**(p.ke2 + p.w_b) > p.K2 * p.d2
                or 2**p.w_b > (p.N - 1) * p.b_rot + 1):
            continue
        t = zero_tables(p)
        for controlled in (False, True):
            for cls, args, need in _children(p, t, controlled) + _walk_children(p, t, controlled):
                assert own_work(cls, args, p) == need, (p, cls.__name__)
                assert need <= WalkLayout.work_size(p), (p, cls.__name__, need)
                seen.add(cls.__name__)
        checked += 1
    assert seen == CHILDREN
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_walk.py tests/integration/test_walk_counts.py tests/test_work_pool.py -q`
Expected: FAIL at collection, `3 errors`: `ImportError: cannot import name 'ControlledDFTHCWalk' from 'dfthc.ops.walk'` (two files) and `cannot import name '_controlled_walk'` (`test_work_pool.py`).

- [ ] **Step 3: Write the walks**

Replace `src/dfthc/ops/walk.py` with:

```python
r"""The block encoding of the sum-of-squares Hamiltonian, the qubitized walk and their controlled
classes (spec 9.21 and 9.22; steps 1 to 28).

**The circuit** (spec 3.3 and 3.5). The block encoding is :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U`
with :math:`U` = ``OuterPrepare``, ``InnerPrepare``, ``InnerSigns``, ``InnerSelect``,
``InnerUnprepare`` in time order and :math:`U^\dagger` = ``InnerPrepare``, ``InnerSelectDagger``,
``InnerSigns``, ``InnerUnprepare``, ``OuterUnprepare``; its block on the ancillas in
:math:`|0\rangle` is :math:`\tilde H_{SA}/\tilde\Lambda - I` (spec 8.5). Every adjoint in it is an
explicit class (``InnerUnprepare``, ``InnerSelectDagger``, ``OuterUnprepare``; spec 7.4 rule 6).
The walk is :math:`W = \mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}\,\mathrm{BE}`: the block
encoding, then ``RefAB``. The controlled classes control only the reflections: on states whose
ancillas start in :math:`|0\rangle`, :math:`U^\dagger U = I`, so
:math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,U^\dagger\,C(\mathrm{REF}_{T_2})\,U`
(spec 3.3). There is no ``C(...)`` rule of any of these operators.

**Wires.** Each class takes the groups of :class:`~dfthc.registers.WalkLayout` (``outer``,
``inner``, ``select``, ``system``, ``phase_grad``, ``work``; a controlled class ``control``
first). Its rule splits them once with :meth:`WalkLayout.split_group`, unpacks every register into
a list once, and passes each child the registers its ``wire_argnames`` name, with the prefix of the
``work`` group of exactly the child's need (spec 9.0, wire passing). ``work`` is the walk's shared
pool: every child returns it in :math:`|0\rangle`; the comparators' dirty work registers belong to
the ``outer`` and ``inner`` groups, never to ``work`` (spec 10.2).

**Arguments.** ``params`` is :meth:`WalkParams.as_compilable` and ``tables`` is
:meth:`WalkTables.as_compilable`; the rules rebuild both with ``from_compilable``. One list,
:func:`_children`, gives the rule its children in time order with their compilable arguments and
work needs, and the resource function its declaration, so the two cannot disagree;
:func:`_walk_children` does the same for the walks.
"""

from __future__ import annotations

from collections import Counter

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.data.tables import WalkTables
from dfthc.ops._validate import check_wires
from dfthc.ops.prepare import (
    InnerPrepare,
    InnerSigns,
    InnerUnprepare,
    OuterPrepare,
    OuterUnprepare,
    _check_inner,
    _check_outer,
)
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefAB, RefT2
from dfthc.ops.select import InnerSelect, InnerSelectDagger
from dfthc.ops.select import _check as _check_select
from dfthc.params import WalkParams
from dfthc.registers import GROUPS, WalkLayout

_GROUPS = ("outer", "inner", "select", "system", "phase_grad", "work")
"""The wire arguments of the assembly operators, after ``control`` in the controlled ones."""


def _rebuild(params, tables) -> tuple[WalkParams, WalkTables]:
    return WalkParams.from_compilable(params), WalkTables.from_compilable(tables)


def _children(p: WalkParams, t: WalkTables, controlled: bool) -> list[tuple[type, tuple, int]]:
    """The block encoding's children in time order (spec 9.21), each ``(class, compilable
    arguments, work wires)``; the reflection is ``ControlledRefT2`` if ``controlled``."""
    outer = (p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    inner = (p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn, p.b_k2,
             p.k2, p.ke2, p.s_u)
    select = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    reflection = ControlledRefT2 if controlled else RefT2
    outer_work = OuterPrepare.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)["work"]
    inner_work = InnerPrepare.required_wires(p.N, p.R, p.B, p.C, p.b_k2, p.k2, p.ke2,
                                             p.s_u)["work"]
    select_work = InnerSelect.required_wires(*select)["work"]
    reflection_work = reflection.required_wires(p.w_b, p.b_k2)["work"]
    return [
        (OuterPrepare, outer, outer_work),
        (InnerPrepare, inner, inner_work),
        (InnerSigns, (), 0),
        (InnerSelect, select, select_work),
        (InnerUnprepare, inner, inner_work),
        (reflection, (), reflection_work),
        (InnerPrepare, inner, inner_work),
        (InnerSelectDagger, select, select_work),
        (InnerSigns, (), 0),
        (InnerUnprepare, inner, inner_work),
        (OuterUnprepare, outer, outer_work),
    ]


def _group_sizes(p: WalkParams) -> dict[str, int]:
    """The width of every group: the layout's (spec 10.1)."""
    sizes = WalkLayout.register_sizes(p)
    return {group: sum(sizes[name] for name in GROUPS[group]) for group in _GROUPS}


def _sizes(p: WalkParams) -> dict[str, int]:
    """The width of every register and group a child of an assembly rule can take."""
    return WalkLayout.register_sizes(p) | _group_sizes(p) | {"control": 1}


def _check(params, tables) -> tuple[WalkParams, WalkTables]:
    """``params`` and ``tables`` rebuilt and checked against the children's domains (spec 9.0):
    the tables have the shapes of ``params`` and lie in the children's ranges, the erases'
    one-hot registers fit (``2**ke1 <= K1 d1``, ``2**(ke2 + w_b) <= K2 d2`` and
    ``2**w_b <= (N - 1) b_rot + 1``), and the layout's ``work`` pool covers every child of the walk
    and the controlled walk. ``ValueError`` otherwise."""
    if not isinstance(params, (tuple, list)) or not isinstance(tables, (tuple, list)):
        raise ValueError("params and tables must be WalkParams.as_compilable() and "
                         "WalkTables.as_compilable()")
    p, t = _rebuild(params, tables)
    _check_outer(p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    _check_inner(p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn,
                 p.b_k2, p.k2, p.ke2, p.s_u)
    _check_select(t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    needs = {cls.__name__: work for cls, _, work in _children(p, t, controlled=True)}
    needs["ControlledRefAB"] = ControlledRefAB.required_wires(p.w_x, p.b_k1, p.w_b,
                                                              p.b_k2)["work"]
    pool = WalkLayout.work_size(p)
    over = {name: need for name, need in needs.items() if need > pool}
    if over:
        raise ValueError(f"the work pool of {pool} wires (WalkLayout.work_size) is too small for "
                         f"{over}")
    return p, t


def _validate(name: str, wire_argnames: tuple, params, tables, wires: tuple) -> tuple:
    """The canonical compilable arguments, after checking them and the wire groups."""
    p, t = _check(params, tables)
    required = ({"control": 1} if "control" in wire_argnames else {}) | _group_sizes(p)
    check_wires(name, dict(zip(wire_argnames, wires, strict=True)), required)
    return p.as_compilable(), t.as_compilable()


def _registers(p: WalkParams, outer, inner, select, system, phase_grad, control=()) -> dict:
    """Every group and every register by name, each unpacked into a list once (spec 9.0)."""
    registers = {"outer": list(outer), "inner": list(inner), "select": list(select),
                 "system": list(system), "phase_grad": list(phase_grad),
                 "control": list(control)}
    for group in ("outer", "inner", "select", "system"):
        registers |= WalkLayout.split_group(p, group, registers[group])
    return registers


def _emit(children: list, registers: dict, work) -> None:
    """Apply ``children`` in order, each on the registers its ``wire_argnames`` name and on the
    prefix of ``work`` of exactly its need."""
    work = list(work)
    for cls, args, need in children:
        cls(*args, **{name: work[:need] if name == "work" else registers[name]
                      for name in cls.wire_argnames})


def _resources(children: list, p: WalkParams) -> dict:
    """The declaration of ``children``: each distinct child once, with its count, its compilable
    arguments and exactly the register sizes :func:`_emit` passes (spec 7.4 rule 2)."""
    sizes = _sizes(p)

    def key(cls, args, need):
        return resource_key(lambda: cls(*args, **{
            name: Wire[need if name == "work" else sizes[name]] for name in cls.wire_argnames}))

    return {key(*child): count for child, count in Counter(children).items()}


class DFTHCBlockEncoding(Operator2):
    r"""The DFTHC block encoding :math:`\mathrm{BE} = U^\dagger\,\mathrm{REF}_{T_2}\,U` (steps 1
    to 27): :math:`\langle 0|\mathrm{BE}|0\rangle = \tilde H_{SA}/\tilde\Lambda - I`.

    **Action.** With every group but ``system`` and ``phase_grad`` in :math:`|0\rangle`, the part
    of the output with every ancilla back in :math:`|0\rangle` is
    :math:`(\tilde H_{SA}/\tilde\Lambda - I)|\psi\rangle` for the system input :math:`|\psi\rangle`,
    with :math:`(\tilde H_{SA}, \tilde\Lambda)` = ``encoded_block(params, tables, factors)``
    (spec 8.5): the alias probabilities of the tables, the sign columns, the directions of the
    rounded angles, the identity term ``b = B``, the failed uniform preparation of ``b`` (acting as
    :math:`+I`) and that of ``x_o`` (where :math:`\mathrm{REF}_{T_2}` is :math:`-I`, which gives
    :math:`\tilde\Lambda = \Lambda/(1 - P_{fail}(x_o))`). :math:`\mathrm{BE}` is Hermitian and
    self-inverse. The rest of the output has a nonzero index register of
    :math:`\mathcal{A}\cup\mathcal{B}`. The output is the same for every outcome of the erase
    measurements.

    **Registers.** The groups of :class:`~dfthc.registers.WalkLayout` (spec 10.1), each a
    concatenation of named registers in layout order.

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``outer``      the ``outer`` group (143)      ``|0>``               see the action
    ``inner``      the ``inner`` group (455)      ``|0>``               see the action
    ``select``     ``(N - 1) b_rot + 4`` (799)    ``|0>``               ``|0>``
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** ``params``: :meth:`WalkParams.as_compilable`, the 13 fields; the
    erases' one-hot bits must fit their reset registers (:math:`2^{k_{e1}} \le K_1 d_1`,
    :math:`2^{k_{e2} + w_b} \le K_2 d_2`, :math:`2^{w_b} \le (N - 1) b_{rot} + 1`) and
    ``WalkLayout.work_size`` must cover every child. ``tables``:
    :meth:`WalkTables.as_compilable` of tables with the shapes of ``params``, in the ranges the
    prepares and the select accept. The constructor checks both (``ValueError``).

    **Pre- and postconditions.** The groups are pairwise disjoint (unchecked). ``phase_grad``
    holds the state ``PhaseGradientStatePrep`` prepares and keeps it. The block needs the ancilla
    groups in ``|0>``; in the walk the input is also any state the walk reaches from there, on
    which every erase's precondition holds (spec 3.3). Afterwards every register but the six
    index registers of :math:`\mathcal{A}\cup\mathcal{B}` (``x_o``, ``m_x_o``, ``sigma_1``,
    ``b``, ``m_b``, ``sigma_0``) is back in ``|0>``: the erases reset the loaded data, the
    measured uncomputes clean the comparators, and the adjoint uniform preparations clear the
    success flags. So ``RefAB``'s reflection about those six is the walk's reflection about every
    ancilla.

    **Decomposition** (spec 9.21). The groups are split with ``WalkLayout.split_group``; each
    child gets its registers by name and the prefix of ``work`` of its need: ``OuterPrepare``;
    ``InnerPrepare``, ``InnerSigns``, ``InnerSelect``, ``InnerUnprepare``; ``RefT2``;
    ``InnerPrepare``, ``InnerSelectDagger``, ``InnerSigns``, ``InnerUnprepare``;
    ``OuterUnprepare``. The prepare classes take the whole ``phase_grad`` (they use its top
    ``s_u`` wires), the selects exactly its ``b_rot`` wires; both selects get
    ``fixup_bits = w_b``. ``InnerSigns`` is diagonal and touches no register of the selects, so
    its place on either side of them is exact; this is spec 9.21's.

    **Adjoint and control.** Never adjointed: its rule measures (spec 7.4 rule 5), and the
    operator is its own inverse anyway. Never controlled: the controlled form is the class
    :class:`ControlledDFTHCBlockEncoding`, which controls only ``RefT2``; there is no
    ``C(DFTHCBlockEncoding)`` rule, so ``qp.ctrl`` of this operator does not compile (spec 7.4
    rule 6).

    **Cost.** :func:`dfthc.costs.block_encoding` (uncontrolled):
    :math:`T = T_{out} + T_{out}^\dagger + 2(T_{in} + T_{sel} + T_{in}^\dagger) + T_{RefT2}`,
    :math:`M = K_1 d_1 + 2(K_2 d_2 + (N - 1) b_{rot} + 3)`; ``A = T - (T_Toffoli - 1)`` with
    ``T_Toffoli`` = :func:`dfthc.costs.toffoli_only` (the walk's plain Toffolis less ``RefAB``'s
    one). At FeMoco-54: :math:`279 + 2 \cdot 4725 + 20 = 9749`, ``M = 2524``, ``A = 8632``.
    ``CNOT`` and ``PauliX`` counts are upper bounds (spec 7.6).

    **Paper and dfthcbe.** Steps 1 to 27 of spec 3.5: the sum-of-squares block encoding
    :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U` (PRX 15, 041016; arXiv:2502.15882v1 Eq. (11) and
    Fig. 1), built in App. B: the prepare over :math:`x_o` (B.1, Eq. (87)) and the step of
    oblivious amplitude amplification, whose reflection is controlled on the success of
    :math:`x_o` (B.2, Eq. (88)); the costs are its Table 3 (PRX Table III). dfthcbe:
    ``DFTHCBlockEncoding`` (``dfthcbe/block_encoding.py``), 9,596 Toffolis at FeMoco-54 by its
    call graph, :math:`212 + 2 (913 + 220) + 2 \cdot 3502 + 20 + 94`; ours is +153 (the inner
    prepare +58 and select +35 per round, the inner unprepare -3 per round, the outer prepare -2
    and unprepare -25; spec 13.1). The paper gives 9,997 per walk, 46 of them for
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}`; against the other 9,951 ours is -202: the
    Givens ladders -424 (``rz-addsub``), the 26-bit inner entries against its 20-bit ones +186,
    exact boundaries and one-hot padding +80 (``exact-iter``, ``no-prune``, ``onehot-pad``), the
    free comparator adjoints -46 (``cmp-free``) and the rest +2 (spec 13.1).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import DFTHCBlockEncoding
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p)
        groups = {name: getattr(layout, name) for name in DFTHCBlockEncoding.wire_argnames}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            DFTHCBlockEncoding(p.as_compilable(), t.as_compilable(), **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 150
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = _GROUPS

    def __init__(self, params, tables, outer, inner, select, system, phase_grad, work):
        wires = (outer, inner, select, system, phase_grad, work)
        compilable = _validate("DFTHCBlockEncoding", self.wire_argnames, params, tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """The layout's group sizes (spec 10.1); ``tables`` does not change them."""
        return _group_sizes(WalkParams.from_compilable(params))


def _block_encoding_resources(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_children(p, t, controlled=False), p)


@qp.register_resources(_block_encoding_resources)
def _block_encoding(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    _emit(_children(p, t, controlled=False),
          _registers(p, outer, inner, select, system, phase_grad), work)


register_rule(DFTHCBlockEncoding, _block_encoding)


class ControlledDFTHCBlockEncoding(Operator2):
    r"""The controlled block encoding: the explicit controlled class of
    :class:`DFTHCBlockEncoding` (spec 7.4 rule 6), :math:`U^\dagger\,C(\mathrm{REF}_{T_2})\,U`.

    **Action.** :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle 1|_{control}
    \otimes \mathrm{BE}` on inputs whose ancilla groups are in :math:`|0\rangle`, with
    :math:`\mathrm{BE}` the action of :class:`DFTHCBlockEncoding` for the same arguments: control
    ``|0>`` returns the whole input (:math:`U^\dagger U = I`), control ``|1>`` gives
    :class:`DFTHCBlockEncoding`'s whole output, global sign included. The output is the same for
    every outcome of the erase measurements.

    **Registers.** Those of :class:`DFTHCBlockEncoding` after a one-wire ``control``, which
    stays a control (only phases on it).

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``control``    1                              any                   a control
    ``outer``      the ``outer`` group (143)      ``|0>``               see the action
    ``inner``      the ``inner`` group (455)      ``|0>``               see the action
    ``select``     ``(N - 1) b_rot + 4`` (799)    ``|0>``               ``|0>``
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCBlockEncoding`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCBlockEncoding`, on both control
    branches; ``control`` is disjoint from the groups (unchecked).

    **Decomposition** (spec 9.21). :class:`DFTHCBlockEncoding`'s eleven children with
    ``ControlledRefT2(control, succ_x_o, b, m_b, sigma_0, work)`` in place of ``RefT2``: every
    other child is uncontrolled, the same operator with the same compilable arguments and
    register sizes as in :class:`DFTHCBlockEncoding`, so the measured unprepares are never
    controlled (spec 7.4 rule 5).

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0).

    **Cost.** :func:`dfthc.costs.block_encoding` (controlled): :class:`DFTHCBlockEncoding`'s
    ``T + 1`` (the controlled reflection's extra ``TemporaryAND``), the same ``M``,
    ``A = T - (T_Toffoli - 1)``. At FeMoco-54: ``T = 9750``, ``M = 2524``, ``A = 8633``.

    **Paper and dfthcbe.** The controlled block encoding inside the controlled walk
    :math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,U^\dagger\,
    C(\mathrm{REF}_{T_2})\,U` (spec 3.3). dfthcbe: ``DFTHCBlockEncoding(is_controlled=True)``,
    9,597 Toffolis at FeMoco-54, which also controls only ``ref_t2``.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import ControlledDFTHCBlockEncoding
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p, qpe_bits=1)
        groups = {name: getattr(layout, name)
                  for name in ControlledDFTHCBlockEncoding.wire_argnames[1:]}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            ControlledDFTHCBlockEncoding(p.as_compilable(), t.as_compilable(),
                                         layout.qpe_control, **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 150 + 1 = 151
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = ("control", *_GROUPS)

    def __init__(self, params, tables, control, outer, inner, select, system, phase_grad, work):
        wires = (control, outer, inner, select, system, phase_grad, work)
        compilable = _validate("ControlledDFTHCBlockEncoding", self.wire_argnames, params,
                               tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """``control`` (1 wire), then the layout's group sizes."""
        return {"control": 1} | _group_sizes(WalkParams.from_compilable(params))


def _controlled_block_encoding_resources(params, tables, control, outer, inner, select, system,
                                         phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_children(p, t, controlled=True), p)


@qp.register_resources(_controlled_block_encoding_resources)
def _controlled_block_encoding(params, tables, control, outer, inner, select, system, phase_grad,
                               work):
    p, t = _rebuild(params, tables)
    _emit(_children(p, t, controlled=True),
          _registers(p, outer, inner, select, system, phase_grad, control), work)


register_rule(ControlledDFTHCBlockEncoding, _controlled_block_encoding)


def _walk_children(p: WalkParams, t: WalkTables, controlled: bool) -> list[tuple[type, tuple, int]]:
    """The walk's children (spec 9.22), each ``(class, compilable arguments, work wires)``: the
    block encoding on the whole groups and pool, then ``RefAB``; both controlled if
    ``controlled``."""
    block_encoding = ControlledDFTHCBlockEncoding if controlled else DFTHCBlockEncoding
    reflection = ControlledRefAB if controlled else RefAB
    return [(block_encoding, (p.as_compilable(), t.as_compilable()), WalkLayout.work_size(p)),
            (reflection, (), reflection.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)["work"])]


class DFTHCWalk(Operator2):
    r"""The qubitized walk :math:`W = \mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}\,\mathrm{BE}`
    (steps 1 to 28) of the DFTHC block encoding.

    **Action.** :class:`DFTHCBlockEncoding`, then
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}} = 2|0\rangle\langle 0| - I` on
    ``x_o``, ``m_x_o``, ``sigma_1``, ``b``, ``m_b`` and ``sigma_0`` (spec 3.3). With the ancilla
    groups in :math:`|0\rangle`, :math:`\langle 0|W^k|0\rangle = T_k(\tilde H_{SA}/\tilde\Lambda
    - I)` for the Chebyshev polynomials :math:`T_k` (spec 12.4, E-2), and :math:`W` has the
    eigenphases :math:`\pm\arccos(E_j/\tilde\Lambda - 1)` for the eigenvalues :math:`E_j` of
    :math:`\tilde H_{SA}` (spec 3.3, 8.5). The output is the same for every outcome of the erase
    measurements.

    **Registers.** Those of :class:`DFTHCBlockEncoding`, with the same input and output states.

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``outer``      the ``outer`` group (143)      ``|0>``               see the action
    ``inner``      the ``inner`` group (455)      ``|0>``               see the action
    ``select``     ``(N - 1) b_rot + 4`` (799)    ``|0>``               ``|0>``
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCBlockEncoding`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCBlockEncoding`. After every step only
    the six index registers of :math:`\mathcal{A}\cup\mathcal{B}` may differ from ``|0>``;
    they carry the walk's state to the next step.

    **Decomposition** (spec 9.22). ``DFTHCBlockEncoding(params, tables, outer, inner, select,
    system, phase_grad, work)`` on the whole groups, then ``RefAB(x_o, m_x_o, sigma_1, b, m_b,
    sigma_0, work[:n_w - 3])`` on the registers ``WalkLayout.split_group`` names.

    **Adjoint and control.** Never adjointed: its rule measures. Never controlled: the
    controlled walk is the class :class:`ControlledDFTHCWalk`. There is no ``C(DFTHCWalk)``
    rule, so ``qp.ctrl(DFTHCWalk(...))`` fails to compile with a ``CompileError`` (the erases
    cannot be controlled); a delegating rule would compile but bring back the cascade of
    distributed candidates (spec 7.4 rule 6, Task G0.1).

    **Cost.** :func:`dfthc.costs.walk` (uncontrolled): :math:`T = T_{BE} + T_{RefAB}`,
    :math:`M = M_{BE}`; ``A`` = :func:`dfthc.costs.measured_and_uncomputes` (uncontrolled),
    ``T - T_Toffoli``. At FeMoco-54: :math:`9749 + 44 = 9793`, ``M = 2524``, ``A = 8675``, so
    ``M + A = 11199`` physical measurements (spec 9.24). ``CNOT`` and ``PauliX`` counts are
    upper bounds (spec 7.6).

    **Paper and dfthcbe.** Steps 1 to 28 of spec 3.5, the walk of the paper's App. B and Table
    III (9,997 Toffolis at FeMoco-54; ours is -204, spec 13.1). dfthcbe: ``DFTHCWalk``
    (``dfthcbe/block_encoding.py``), 9,641; ours is +152 (spec 13.1).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import DFTHCWalk
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p)
        groups = {name: getattr(layout, name) for name in DFTHCWalk.wire_argnames}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            DFTHCWalk(p.as_compilable(), t.as_compilable(), **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 150 + 8 = 158
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = _GROUPS

    def __init__(self, params, tables, outer, inner, select, system, phase_grad, work):
        wires = (outer, inner, select, system, phase_grad, work)
        compilable = _validate("DFTHCWalk", self.wire_argnames, params, tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """The layout's group sizes (spec 10.1); ``tables`` does not change them."""
        return _group_sizes(WalkParams.from_compilable(params))


def _walk_resources(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    return _resources(_walk_children(p, t, controlled=False), p)


@qp.register_resources(_walk_resources)
def _walk(params, tables, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    _emit(_walk_children(p, t, controlled=False),
          _registers(p, outer, inner, select, system, phase_grad), work)


register_rule(DFTHCWalk, _walk)


class ControlledDFTHCWalk(Operator2):
    r"""The controlled walk: the explicit controlled class of :class:`DFTHCWalk` (spec 7.4 rule
    6), :math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,U^\dagger\,
    C(\mathrm{REF}_{T_2})\,U`.

    **Action.** :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle 1|_{control}
    \otimes W` on inputs whose ancilla groups are in :math:`|0\rangle`, with :math:`W` the action
    of :class:`DFTHCWalk` for the same arguments: control ``|0>`` returns the whole input,
    control ``|1>`` gives :class:`DFTHCWalk`'s whole output, global sign included, and control
    ``|+>`` their superposition (spec 12.4, E-3). The output is the same for every outcome of the
    erase measurements.

    **Registers.** Those of :class:`DFTHCWalk` after a one-wire ``control``, which stays a
    control (only phases on it).

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``control``    1                              any                   a control
    ``outer``      the ``outer`` group (143)      ``|0>``               see the action
    ``inner``      the ``inner`` group (455)      ``|0>``               see the action
    ``select``     ``(N - 1) b_rot + 4`` (799)    ``|0>``               ``|0>``
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCWalk`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCWalk`, on both control branches;
    ``control`` is disjoint from the groups (unchecked).

    **Decomposition** (spec 9.22). ``ControlledDFTHCBlockEncoding(params, tables, control, outer,
    inner, select, system, phase_grad, work)``, then ``ControlledRefAB(control, x_o, m_x_o,
    sigma_1, b, m_b, sigma_0, work[:n_w - 2])``: only the two reflections are controlled; the
    prepares, selects and measured unprepares run uncontrolled (spec 7.4 rule 5).

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0);
    ``ControlledWalkPower`` (spec 9.23) repeats it.

    **Cost.** :func:`dfthc.costs.walk` (controlled): :class:`DFTHCWalk`'s ``T + 2``, the same
    ``M``; ``A`` = :func:`dfthc.costs.measured_and_uncomputes` (controlled). At FeMoco-54:
    ``T = 9795``, ``M = 2524``, ``A = 8677``, ``M + A = 11201``.

    **Paper and dfthcbe.** The controlled walk of spec 3.3, the unit of the paper's phase
    estimation. dfthcbe: ``DFTHCWalk(is_controlled=True)``, 9,642 Toffolis at FeMoco-54 (its
    controlled ``ref_ab`` costs the same as the uncontrolled one); ours is +153.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.walk import ControlledDFTHCWalk
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p, qpe_bits=1)
        groups = {name: getattr(layout, name) for name in ControlledDFTHCWalk.wire_argnames[1:]}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            ControlledDFTHCWalk(p.as_compilable(), t.as_compilable(), layout.qpe_control,
                                **groups)

        compiled_resources(program, num_wires=layout.num_wires).toffoli  # 158 + 2 = 160
    """

    compilable_argnames = ("params", "tables")
    wire_argnames = ("control", *_GROUPS)

    def __init__(self, params, tables, control, outer, inner, select, system, phase_grad, work):
        wires = (control, outer, inner, select, system, phase_grad, work)
        compilable = _validate("ControlledDFTHCWalk", self.wire_argnames, params, tables, wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(params, tables) -> dict[str, int]:
        """``control`` (1 wire), then the layout's group sizes."""
        return {"control": 1} | _group_sizes(WalkParams.from_compilable(params))


def _controlled_walk_resources(params, tables, control, outer, inner, select, system, phase_grad,
                               work):
    p, t = _rebuild(params, tables)
    return _resources(_walk_children(p, t, controlled=True), p)


@qp.register_resources(_controlled_walk_resources)
def _controlled_walk(params, tables, control, outer, inner, select, system, phase_grad, work):
    p, t = _rebuild(params, tables)
    _emit(_walk_children(p, t, controlled=True),
          _registers(p, outer, inner, select, system, phase_grad, control), work)


register_rule(ControlledDFTHCWalk, _controlled_walk)
```

- [ ] **Step 4: Add the inventory instrument**

Replace `src/dfthc/testing/resources.py` with:

```python
"""Assertions on compiled resources and the rule-lowering warning policy.

Spec sections 7.1 (warning noise), 7.4 rule 12 (warning policy) and 7.6 (accounting).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable

import pennylane as qp
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.passes import graph_decomposition

from dfthc.compile import (
    FIXED_DECOMPS,
    GATE_SET,
    CompiledResources,
    compiled_resources,
    dfthc_operator_names,
)

_EXCLUDED_RULE = (
    r"(?:_(?:\w*_)?ppm|_toffoli_elbow|_cphase_to_single_phase|\w*_workers?\w*"
    r"|use_allocation\(\w+\))"
)
NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"^Excluded the {_EXCLUDED_RULE} decomposition rule for \S+ since "
        r"--decompose-lowering cannot work with multiple registers yet$"
    ),
    re.compile(
        r"^Skipped the _adjoint_temporary_and\w* (?:decomposition )?rule for "
        r"C\(Adjoint\(TemporaryAND\)\)"
    ),
)
"""RuleLoweringWarning texts that PennyLane's own rules print on every compile (spec 7.1)."""

UNRESOLVED_BRANCH_MESSAGE = (
    r"Resource analysis was unable to determine the branch of a conditional or switch statement"
)
"""qp.analyze UserWarning for an unhinted conditional. PennyLane's SemiAdder rule triggers it on
every compile, so it is filtered; exact T4 counts and the erase-hint tests guard our own code."""


_EXPECTED_SHAPE = re.compile(
    r"^Skipped the \S+ decomposition rule for "
    r"(?P<wrapped>(?P<open>(?:(?:Adjoint|C)\()+)(?P<name>\w+)(?P<close>\)+))"
    r"(?:\{[^{}]*\})*"
    r": it contains a mid-circuit measurement, which is not supported with adjoint or control "
    r"regions\.\Z"
)
"""Catalyst's drop of a measuring rule for a modified operator (spec 7.4 rule 12). Nested
``Adjoint(...)``/``C(...)`` wrappers around one operator name, then only attribute dumps; the
parentheses are balanced if the wrapper opens equal the closes (checked in the classifier)."""


def classify_rule_warning(message: str, names: Iterable[str] | None = None) -> str:
    """Return ``"expected"``, ``"dfthc"``, ``"noise"`` or ``"other"`` for a RuleLoweringWarning.

    ``"expected"`` is exactly Catalyst's "Skipped the <rule> decomposition rule for <wrapped op>:
    it contains a mid-circuit measurement, which is not supported with adjoint or control
    regions." where ``<wrapped op>`` is nested ``Adjoint(...)``/``C(...)`` around an operator in
    ``names`` and that operator is the only name of ``names`` in the message. Every other message
    naming a dfthc operator is ``"dfthc"``, even if it also matches a noise pattern. ``names``
    defaults to the dfthc operator names of the pinned rules; a bare ``str`` raises ``TypeError``.
    """
    if isinstance(names, str):
        raise TypeError(f"names must be an iterable of operator names, not the string {names!r}")
    names = dfthc_operator_names() if names is None else frozenset(names)
    named = {name for name in names if re.search(rf"\b{re.escape(name)}\b", message)}
    match = _EXPECTED_SHAPE.match(message)
    if (
        match
        and named == {match["name"]}
        and match["open"].count("(") == len(match["close"])
    ):
        return "expected"
    if named:
        return "dfthc"
    if any(pattern.search(message) for pattern in NOISE_PATTERNS):
        return "noise"
    return "other"


def install_warning_filters() -> None:
    """Show every RuleLoweringWarning; the test fixture drops the known PennyLane noise after
    classification."""
    warnings.filterwarnings("always", category=RuleLoweringWarning)
    warnings.filterwarnings("ignore", message=UNRESOLVED_BRANCH_MESSAGE, category=UserWarning)


def verbose_program(qfunc: Callable[[], None], num_wires: int):
    """The compile stack of spec 7.2 with the solver's rule listing forwarded to stderr (T6).

    Only ``qjit(verbose=True)`` forwards the compiler output. Run the returned program (or
    ``qp.analyze`` it) under pytest's ``capfd`` and pass the captured text to :func:`chosen_rules`.
    """

    @qp.qjit(capture=True, verbose=True)
    @graph_decomposition(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS) or None,
                         verbose=True)
    @qp.qnode(qp.device("null.qubit", wires=num_wires))
    def program():
        qfunc()
        return qp.probs(wires=[0])

    return program


def chosen_rules(listing: str) -> list[str]:
    """Rule names from the lines ``Chosen Rule: __builtin_<rule>`` of a verbose listing."""
    prefix = "Chosen Rule: __builtin_"
    return [line.split(prefix, 1)[1].strip() for line in listing.splitlines() if prefix in line]


def captured_targets(qfunc: Callable[[], None], num_wires: int) -> list[str]:
    """The rule targets the Catalyst frontend converts while :func:`compiled_resources` compiles
    ``qfunc`` once, one name per converted target, in order: the operator name with its modifiers
    (``Adjoint(MajoranaGate)``, ``C(RefT2)``), without the attribute dump that starts at the first
    ``{`` (spec 12.1, T6: the inventory test).

    A port of Task 0.9d's capture instrument (``spikes/k1_whole_program.capture_instrument``): it
    wraps Catalyst's private ``catalyst.from_plxpr.qfunc_interpreter._convert_decomp_target_spec``
    for this one compile and restores it afterwards. The frontend converts every target it builds
    rules for, chosen or not, so the list holds every distributed ``Adjoint``/``C`` candidate the
    compile captured. The hook is private, which makes this a compatibility canary at every
    nightly bump: a renamed hook raises ``AttributeError``, and callers assert that the program's
    own top-level target is in the list, so a hook that no longer sees targets fails instead of
    passing. The private module is imported here, so a renamed module fails only the callers.
    """
    import catalyst.from_plxpr.qfunc_interpreter as qfunc_interpreter

    convert = qfunc_interpreter._convert_decomp_target_spec
    names: list[str] = []

    def recording(interpreter, target_spec):
        names.append(str(target_spec.target_id).split("{", 1)[0])
        return convert(interpreter, target_spec)

    qfunc_interpreter._convert_decomp_target_spec = recording
    try:
        compiled_resources(qfunc, num_wires)
    finally:
        qfunc_interpreter._convert_decomp_target_spec = convert
    return names


LEAF_ADJOINTS = frozenset({"Adjoint(FlaggedUniformPrep)", "Adjoint(GivensRotation)",
                           "Adjoint(MajoranaGate)", "Adjoint(PhaseGradientRotation)"})
"""The only modified dfthc targets a walk program may capture: the auto-derived leaf adjoints of
spec 7.4 rule 6 (the uniform preparation's and, inside it, its rotations'; the Givens rung's; the
Majorana gate's)."""

TEMPLATE_ADJOINTS = frozenset({"Adjoint(SemiAdder)", "Adjoint(LeftQuantumComparator)",
                               "Adjoint(LeftClassicalComparator)"})
"""The PennyLane templates those leaves and the unprepares adjoint: the rotations' and Givens
rungs' adders and the comparators (spec 7.4 rule 4, K-7)."""


def modified_targets(names: Iterable[str], bases: Iterable[str]) -> set[str]:
    """The ``Adjoint(...)`` and ``C(...)`` targets of ``names`` (any nesting, ``2C(...)``
    included) whose operator is one of ``bases``."""
    bases = set(bases)
    return {name for name in names if "(" in name and dfthc_operator_names([name]) & bases}


def inventory(names: Iterable[str]) -> set[str]:
    """Spec 12.1 T6: the modified targets of ``names`` (from :func:`captured_targets`) whose
    operator is a dfthc operator (every name with a pinned rule: the composites, ``QROAMRead`` and
    the leaves) or a template of :data:`TEMPLATE_ADJOINTS`. For a walk, controlled-walk or QPE
    program it must equal ``LEAF_ADJOINTS | TEMPLATE_ADJOINTS``: no modified composite or read."""
    templates = {name[len("Adjoint("):-1] for name in TEMPLATE_ADJOINTS}
    return modified_targets(names, dfthc_operator_names() | templates)


def assert_resources(
    resources: CompiledResources, *, toffoli: int, measurements: int,
    measured_and_uncomputes: int | None = None,
) -> None:
    """Assert exact T, M and optionally A, with every mismatch in one message.

    T, M and A exist only in a ``level="user"`` record; any other record fails the assertion.
    """
    if resources.level != "user":
        raise AssertionError(
            f"assert_resources needs a level='user' record, got level={resources.level!r}"
        )
    expected = {"toffoli": toffoli, "measurements": measurements}
    if measured_and_uncomputes is not None:
        expected["measured_and_uncomputes"] = measured_and_uncomputes
    actual = {name: getattr(resources, name) for name in expected}
    if actual != expected:
        raise AssertionError(
            f"resources differ: expected {expected}, got {actual}; counts {resources.counts}"
        )
```

- [ ] **Step 5: Run the tests**

Run: `scripts/test.sh tests/ops/test_walk.py tests/integration/test_walk_counts.py tests/test_work_pool.py tests/ops/test_block_encoding.py -q`
Expected: `82 passed` (about 110 s with 2 workers, the 4 `slow` compiles included).

Run: `scripts/test.sh tests/test_tolerance_hygiene.py tests/test_compile.py tests/test_canary_helper.py -q`
Expected: `71 passed`.

- [ ] **Step 6: Record the `MID_NAPHTHALENE` compiles**

Run: `DFTHC_TEST_WORKERS=0 scripts/test.sh tests/integration/test_walk_counts.py -m slow -q -s`
Expected: `2 passed, 5 deselected`, after two `MID_COMPILE {...}` lines (the second follows the first test's progress dot, `.MID_COMPILE {...}`) whose `result` holds T, M, A = 1913, 502, 1678 (walk) and 1915, 502, 1680 (controlled walk) with the compile time, and whose `watchdog` holds the subprocess's wall time and peak. Dry run of this command: compile 7.3 and 7.8 s, subprocess 8.9 and 9.5 s, peaks 615 and 589 MB (`.scratch/plan-check/stages/3A/3a-step6-mid.log`; each test alone under `run_watched`: 7.8 and 8.1 s, 615 and 588 MB, `3a-mid-compiles.log`). Copy both lines into the report.

- [ ] **Step 7: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1053 passed` (1025 + 28).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add src/dfthc/ops/walk.py src/dfthc/testing/resources.py tests/ops/test_walk.py tests/integration/test_walk_counts.py tests/test_work_pool.py
git commit -m "Add DFTHCWalk, ControlledDFTHCWalk, whole-program counts and the inventory test"
```

---

### Task 3.5: End-to-end checks E-1 to E-4 and tier budgets

**Task id:** `3.5` · **Branch:** `phase3/3.5` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 3.4

**Files:**
- Create: `tests/integration/test_end_to_end.py`

**Interfaces:**
- Consumes: `dfthc.testing.walk_sim` (Task 3.3: `case`, `program`, `operations`, `prepare_unprepare`, `schedule`, `num_measurements`, `input_state`, `basis`, `simulate`, `project`, `set_control`, `superpose`, `assert_same_state`, and `WalkCase.block`, `encoded_block`'s H̃_SA/Λ̃ − I); `DFTHCBlockEncoding`, `DFTHCWalk`, `ControlledDFTHCWalk` (Tasks 3.3, 3.4).
- Produces: the E checks of spec 12.4 in the tiers of spec 12.6, and the tier projections recorded in the ledger. Nothing for later tasks to import.

**Acceptance tests** (42 tests: 10 `slow`, 32 `release`, none per commit), each comparison to 1e-10 (spec 12.4), every run strict:
- **E-1** (`test_e1_block_encoding`): ⟨0|BE|0⟩ = H̃_SA/Λ̃ − I, built column by column from `DFTHCBlockEncoding`'s outputs. Nightly: `H2_STO3G_TEST`'s 16 inputs under S0 and the 4 fixed inputs under S1; `FAILS_C3`'s 16 inputs under S0. Release: the 16 under S1, the 4 fixed under a seeded schedule, and `H2_631G_TEST`'s 28 two-electron inputs under S0, one test each.
- **E-2** (`test_e2_walk_powers`, nightly): ⟨0|W^k|0⟩ = T_k(H̃_SA/Λ̃ − I) for k = 1, 2 (T_1(x) = x, T_2(x) = 2x² − 1), from k `DFTHCWalk` steps on the 4 fixed inputs under S0, at `H2_STO3G_TEST` and `FAILS_C3`.
- **E-3** (`test_e3_controlled_walk`): `ControlledDFTHCWalk` with the control in |0⟩ (the whole input), |1⟩ (`DFTHCWalk`'s whole output) and |+⟩ (their superposition, which detects a global-phase error), and with the control in |+⟩ after one uncontrolled walk step ((|0⟩W|ψ⟩ + |1⟩W²|ψ⟩)/√2). Nightly: the 4 fixed inputs under S0. Release: the 16 under S1, the 4 fixed under a seeded schedule.
- **E-4** (`test_e4_prepare_then_unprepare`, nightly): U then U† (the block encoding's children without `RefT2`) returns every ancilla to |0⟩ and the system unchanged on all 16 inputs, under the alternating and a seeded schedule.

**Definition of done:** the placement collected as stated; the 10 nightly checks pass targeted; one release-only input timed alone; the per-commit tier passes unchanged; `ruff check .` clean; one commit; the report states the timings, and the orchestrator records the budgets and the release projection in the ledger and runs the nightly tier at G3 (D-2 under the nightly schedules; the release schedules run once, in Task 5.2). This task checks code that exists, so its RED steps are three deliberately broken walks (Steps 3 to 5), one for E-1 and E-2, one for E-3 and one for E-4, each a narrow diff of `walk.py` applied with `git apply` and restored at once with `git apply -R` of the same diff.

- [ ] **Step 1: Write the checks**

Create `tests/integration/test_end_to_end.py`:

```python
"""End-to-end checks E-1 to E-4 (spec 12.4) on the sparse simulator, placed in the tiers of spec
12.6: the nightly tier (``slow``) and the release tier (``release``); none runs per commit, where
the operators' own T3 tests stand in (tests/ops/test_block_encoding.py, tests/ops/test_walk.py).

Every run starts with every ancilla in |0> and ``phase_grad`` in the gradient state the program
prepares, decomposes the production classes on the tape path, and is strict (every erase outcome
has probability 1/2, every AND precondition holds). The expected values come from the encoded
model, ``encoded_block`` (spec 8.5): its ``Lambda~`` (ledger row 1A.5) and our failure signs, not
dfthcbe's (ledger row 2D.2). Schedules (spec 12.5): S0 every erase outcome 0, S1 every outcome 1,
and seeded random outcomes; the certificates of spec 12.5 prove every outcome, so these are smoke
tests. Inputs are system basis states, psi_up's bits first; FIXED_INPUTS are spec 12.6's four.

The shape fails-c3 (``walk_sim.FAILS_C3``) is added to E-1 and E-2: the test presets cannot show
the failed uniform preparation of x_o, dfthcbe's differing failure signs, or C = 3.
"""

import numpy as np
import pytest

from dfthc.ops.walk import ControlledDFTHCWalk, DFTHCBlockEncoding, DFTHCWalk
from dfthc.testing import walk_sim

ALL_INPUTS = tuple(range(16))
FIXED_INPUTS = (0b0000, 0b1000, 0b1010, 0b1111)
"""Spec 12.6: |00>|00>, |10>|00>, |10>|10>, |11>|11> in psi_up (x) psi_dn."""
TWO_ELECTRON_631G = tuple(v for v in range(256) if bin(v).count("1") == 2)
"""The 28 two-electron basis states of H2_631G_TEST (N = 4, 8 spin orbitals)."""
ATOL = 1e-10
"""Spec 12.4: E-1 and E-2 to 1e-10."""

nightly, release = pytest.mark.slow, pytest.mark.release


def columns(c, ops, inputs, kind, control=None, row=0, seed=0):
    """The projected output (``row``: control value) of every basis input, as columns, each run
    with the forced schedule ``kind`` of exactly the program's measurements."""
    outcomes = walk_sim.schedule(kind, walk_sim.num_measurements(ops), seed=seed)
    return np.array([walk_sim.project(c, walk_sim.simulate(
        c, ops, walk_sim.input_state(c, walk_sim.basis(c, v), control), outcomes))[row]
        for v in inputs]).T


E1 = [pytest.param("sto3g", "S0", ALL_INPUTS, marks=nightly, id="sto3g-S0-all"),
      pytest.param("sto3g", "S1", FIXED_INPUTS, marks=nightly, id="sto3g-S1-fixed"),
      pytest.param("fails-c3", "S0", ALL_INPUTS, marks=nightly, id="fails-c3-S0-all"),
      pytest.param("sto3g", "S1", ALL_INPUTS, marks=release, id="sto3g-S1-all"),
      pytest.param("sto3g", "seeded", FIXED_INPUTS, marks=release, id="sto3g-seeded-fixed")]
E1 += [pytest.param("631g", "S0", (v,), marks=release, id=f"631g-S0-{v:08b}")
       for v in TWO_ELECTRON_631G]


@pytest.mark.parametrize(("name", "kind", "inputs"), E1)
def test_e1_block_encoding(name, kind, inputs):
    """E-1: <0_anc|BE|0_anc> = H~_SA / Lambda~ - I to 1e-10, built column by column from the
    simulated outputs of DFTHCBlockEncoding (nightly: all 16 inputs under S0 and the 4 fixed
    under S1 at H2_STO3G_TEST, all 16 under S0 at fails-c3; release: all 16 under S1, the 4 fixed
    under a seeded schedule, and H2_631G_TEST's 28 two-electron inputs under S0, one per test)."""
    c = walk_sim.case(name)
    ops = walk_sim.program(c, DFTHCBlockEncoding)
    measured = columns(c, ops, inputs, kind, seed=12)
    assert np.allclose(measured, c.block()[:, list(inputs)], rtol=0, atol=ATOL)


@nightly
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_e2_walk_powers(name, k):
    """E-2: <0|W^k|0> = T_k(H~_SA / Lambda~ - I) for the Chebyshev polynomials T_1(x) = x and
    T_2(x) = 2x^2 - 1, from k DFTHCWalk steps on the 4 fixed inputs under S0 (nightly)."""
    c = walk_sim.case(name)
    x = c.block()
    chebyshev = {1: x, 2: 2 * x @ x - np.eye(len(x))}[k]
    ops = walk_sim.program(c, DFTHCWalk, repeat=k)
    measured = columns(c, ops, FIXED_INPUTS, "S0")
    assert np.allclose(measured, chebyshev[:, list(FIXED_INPUTS)], rtol=0, atol=ATOL)


E3 = [pytest.param("S0", FIXED_INPUTS, marks=nightly, id="S0-fixed"),
      pytest.param("S1", ALL_INPUTS, marks=release, id="S1-all"),
      pytest.param("seeded", FIXED_INPUTS, marks=release, id="seeded-fixed")]


@pytest.mark.parametrize(("kind", "inputs"), E3)
def test_e3_controlled_walk(kind, inputs):
    """E-3 at H2_STO3G_TEST: ControlledDFTHCWalk with the control in |0> returns the whole input,
    in |1> gives DFTHCWalk's whole output, and in |+> their superposition with weight 1/sqrt 2
    each, which detects a global-phase error of either controlled reflection; and, after one
    uncontrolled DFTHCWalk step (index registers of A u B no longer |0>, as between the rounds of
    iterative QPE), the control in |+> gives (|0> W|psi> + |1> W^2|psi>)/sqrt 2, which also
    needs RefAB controlled (nightly: the 4 fixed inputs under S0; release: all 16 under S1 and
    the 4 fixed under a seeded schedule)."""
    c = walk_sim.case("sto3g")
    once = walk_sim.program(c, DFTHCWalk)
    twice = walk_sim.program(c, DFTHCWalk, repeat=2)
    controlled = walk_sim.program(c, ControlledDFTHCWalk, controlled=True)
    then_controlled = walk_sim.operations(
        c, lambda: DFTHCWalk(*c.compilable, **c.groups()),
        lambda: ControlledDFTHCWalk(*c.compilable, **c.groups(True)))
    m = walk_sim.num_measurements(once)
    outcomes = walk_sim.schedule(kind, 2 * m, seed=13)
    plus = 2**-0.5
    for v in inputs:
        psi = walk_sim.basis(c, v)
        identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
        stepped = walk_sim.simulate(c, once, walk_sim.input_state(c, psi), outcomes[:m])
        applied = walk_sim.set_control(c, stepped)
        expected = {"0": identity, "1": applied,
                    "+": walk_sim.superpose((plus, identity), (plus, applied))}
        for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", [plus, plus])):
            out = walk_sim.simulate(c, controlled, walk_sim.input_state(c, psi, control),
                                    outcomes[:m])
            walk_sim.assert_same_state(out, expected[label])
        stepped_twice = walk_sim.simulate(c, twice, walk_sim.input_state(c, psi), outcomes)
        out = walk_sim.simulate(c, then_controlled, walk_sim.input_state(c, psi, [plus, plus]),
                                outcomes)
        walk_sim.assert_same_state(out, walk_sim.superpose(
            (plus, stepped), (plus, walk_sim.set_control(c, stepped_twice))))


@nightly
@pytest.mark.parametrize("kind", ["alternating", "seeded"])
def test_e4_prepare_then_unprepare(kind):
    """E-4 at H2_STO3G_TEST: U then U-dagger (the block encoding's children without RefT2)
    returns every ancilla to |0> and the system unchanged, on all 16 inputs, under the
    alternating and a seeded schedule of the five erases (an all-ones or all-zeros schedule sees
    less of the erases' masks; ledger row 2D.2)."""
    c = walk_sim.case("sto3g")
    ops = walk_sim.prepare_unprepare(c)
    outcomes = walk_sim.schedule(kind, walk_sim.num_measurements(ops), seed=14)
    for v in ALL_INPUTS:
        psi = walk_sim.basis(c, v)
        identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
        walk_sim.assert_same_state(walk_sim.simulate(c, ops, walk_sim.input_state(c, psi),
                                                     outcomes), identity)
```

- [ ] **Step 2: Check the tier placement**

Run: `scripts/test.sh tests/integration/test_end_to_end.py --collect-only -q -m "not slow and not release"`
Expected: `no tests collected (42 deselected)`.

Run: `scripts/test.sh tests/integration/test_end_to_end.py --collect-only -q -m "slow and not release"`
Expected: `10/42 tests collected (32 deselected)`.

Run: `scripts/test.sh tests/integration/test_end_to_end.py --collect-only -q -m "release"`
Expected: `32/42 tests collected (10 deselected)`.

- [ ] **Step 3: See E-1 and E-2 fail on a broken block encoding, then restore it**

Make the first round's select the dagger, with this narrow patch of `src/dfthc/ops/walk.py` (run from the worktree's root):

```bash
git apply <<'EOF'
--- a/src/dfthc/ops/walk.py
+++ b/src/dfthc/ops/walk.py
@@ -81,3 +81,3 @@
         (InnerSigns, (), 0),
-        (InnerSelect, select, select_work),
+        (InnerSelectDagger, select, select_work),
         (InnerUnprepare, inner, inner_work),
EOF
```

Run: `scripts/test.sh tests/integration/test_end_to_end.py -q -m "slow and not release" -k "e1 or e2"`
Expected: FAIL, `7 failed`: every E-1 and E-2 check (the block becomes Σ O O† instead of Σ O† O).

Restore `walk.py` with the reverse of the same patch:

```bash
git apply -R <<'EOF'
--- a/src/dfthc/ops/walk.py
+++ b/src/dfthc/ops/walk.py
@@ -81,3 +81,3 @@
         (InnerSigns, (), 0),
-        (InnerSelect, select, select_work),
+        (InnerSelectDagger, select, select_work),
         (InnerUnprepare, inner, inner_work),
EOF
```

Run: `git status --short`
Expected: only `?? tests/integration/test_end_to_end.py` (the dry run, in a replica without git, compared the restored `walk.py` byte for byte with the original).

- [ ] **Step 4: See E-3 fail on an uncontrolled reflection, then restore it**

Leave the controlled walk's `RefAB` uncontrolled:

```bash
git apply <<'EOF'
--- a/src/dfthc/ops/walk.py
+++ b/src/dfthc/ops/walk.py
@@ -406,3 +406,3 @@
     block_encoding = ControlledDFTHCBlockEncoding if controlled else DFTHCBlockEncoding
-    reflection = ControlledRefAB if controlled else RefAB
+    reflection = RefAB
     return [(block_encoding, (p.as_compilable(), t.as_compilable()), WalkLayout.work_size(p)),
EOF
```

Run: `scripts/test.sh tests/integration/test_end_to_end.py -q -m "slow and not release" -k e3`
Expected: FAIL, `1 failed`: `test_e3_controlled_walk[S0-fixed]`, at the comparison after one walk step (`AssertionError: 0.21496583928721302` from `assert_same_state`, about 0.21 away). The three control states before it pass: from ancillas in |0⟩ an uncontrolled `RefAB` on the control-|0⟩ branch acts as the identity, and only the state after a walk step has ancillas for it to reflect.

Restore `walk.py`:

```bash
git apply -R <<'EOF'
--- a/src/dfthc/ops/walk.py
+++ b/src/dfthc/ops/walk.py
@@ -406,3 +406,3 @@
     block_encoding = ControlledDFTHCBlockEncoding if controlled else DFTHCBlockEncoding
-    reflection = ControlledRefAB if controlled else RefAB
+    reflection = RefAB
     return [(block_encoding, (p.as_compilable(), t.as_compilable()), WalkLayout.work_size(p)),
EOF
```

Run: `git status --short`
Expected: only `?? tests/integration/test_end_to_end.py`.

- [ ] **Step 5: See E-4 fail on a disordered unprepare, then restore it**

Run the outer unprepare before the inner one:

```bash
git apply <<'EOF'
--- a/src/dfthc/ops/walk.py
+++ b/src/dfthc/ops/walk.py
@@ -87,4 +87,4 @@
         (InnerSigns, (), 0),
+        (OuterUnprepare, outer, outer_work),
         (InnerUnprepare, inner, inner_work),
-        (OuterUnprepare, outer, outer_work),
     ]
EOF
```

Run: `scripts/test.sh tests/integration/test_end_to_end.py -q -m "slow and not release" -k e4`
Expected: FAIL, `2 failed`: `test_e4_prepare_then_unprepare[alternating]` and `[seeded]`, each at the comparison with the input (`AssertionError: 0.1631672471881693` and `0.22634238599178114`): the inner unprepare, whose tables `x_o` addresses, runs after the outer unprepare has already uncomputed `x_o`, so the ancillas do not return to |0⟩.

Restore `walk.py`:

```bash
git apply -R <<'EOF'
--- a/src/dfthc/ops/walk.py
+++ b/src/dfthc/ops/walk.py
@@ -87,4 +87,4 @@
         (InnerSigns, (), 0),
+        (OuterUnprepare, outer, outer_work),
         (InnerUnprepare, inner, inner_work),
-        (OuterUnprepare, outer, outer_work),
     ]
EOF
```

Run: `git status --short`
Expected: only `?? tests/integration/test_end_to_end.py`.

- [ ] **Step 6: Run the nightly checks**

Run: `scripts/test.sh tests/integration/test_end_to_end.py -q -m "slow and not release"`
Expected: `10 passed` (about 78 s with 2 workers; peak about 0.6 GB).

- [ ] **Step 7: Time one release-only input**

Run: `DFTHC_TEST_WORKERS=0 scripts/test.sh "tests/integration/test_end_to_end.py::test_e1_block_encoding[631g-S0-00110000]" -q --durations=1`
Expected: `1 passed` (about 40 s; one input of the 28 of the release tier's 6-31G E-1). Do not run the release tier.

- [ ] **Step 8: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1053 passed` (unchanged: the checks are nightly or release).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 9: Commit**

```bash
git add tests/integration/test_end_to_end.py
git commit -m "Add the end-to-end checks E-1 to E-4 in the nightly and release tiers"
```

- [ ] **Step 10: Report the budgets (the orchestrator records them)**

In the report give: the per-commit tier's time and peak from Step 8 against its budget of 15 min and 4 GB, which binds the two-worker process tree (the load gate of about 1 GB and a few seconds binds each test, not the tree: this plan's one-worker session puts every new per-commit test of Tasks 3.3 and 3.4 below 485 MB and at most 8.5 s, and Task 3.5 adds none); the nightly E checks' time from Step 6 and the one-input time from Step 7; and the projection of the section "Simulation cost and the tier budgets" of this plan (nightly about 4 to 6 min against 4 h; release about 25 min against 8 h), with any difference to the dry run's numbers. The orchestrator runs the nightly tier at G3, alone, under the watchdog (`PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 300 -- scripts/test.sh -m "slow and not release"`), records it as a heavy run, and escalates only if a projection exceeds its budget.

---

## Interfaces for Phase 4

What the QPE task (4.1: `ControlledWalkPower`, `QPERound`, the iterative driver) and its counts and report (4.2, 4.3, 4.5) consume from this file, exactly as built.

**Constructors and sizes.** `stages/3A/interfaces_probe_3a.py` below is the interface as executed in the dry run: it lays out the walk with QPE registers (`WalkLayout.from_params(p, qpe_bits=3)`), flattens the controlled walk's groups the way `QPERound`'s `walk_wires` will carry them, splits them back, and compiles C(W)² (two `ControlledDFTHCWalk`s on `layout.qpe_control`, what `ControlledWalkPower(2, ...)` will loop) against `costs.controlled_walk_power`.

```python
"""Interfaces for Phase 4: how the QPE driver (Task 4.1) calls the controlled walk.

Executed by the Phase 3 assembly plan's dry run, from the replica root with PYTHONPATH=src. It lays
out the walk with QPE registers, flattens the controlled walk's groups in wire_argnames order (the
form of QPERound's walk_wires with walk_wire_sizes), splits them back, and compiles C(W)^2 on
layout.qpe_control at H2_STO3G_TEST against costs.controlled_walk_power.
"""

import pennylane as qp

from dfthc import costs
from dfthc.compile import compiled_resources
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.walk import ControlledDFTHCWalk
from dfthc.params import H2_STO3G_TEST as p
from dfthc.registers import WalkLayout


def main():
    t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
    params, tables = p.as_compilable(), t.as_compilable()
    layout = WalkLayout.from_params(p, qpe_bits=3)
    names = ControlledDFTHCWalk.wire_argnames[1:]
    sizes = ControlledDFTHCWalk.required_wires(params, tables)
    print("required_wires:", sizes)
    walk_wire_sizes = tuple(sizes[name] for name in names)
    walk_wires = [wire for name in names for wire in getattr(layout, name)]
    print("walk_wire_sizes, in wire_argnames order", names, ":", walk_wire_sizes)
    print("layout order starts with system:", list(layout.system), "; qpe_control",
          list(layout.qpe_control), "qpe_grad", list(layout.qpe_grad), "; wires",
          layout.num_wires)
    groups, start = {}, 0
    for name, size in zip(names, walk_wire_sizes, strict=True):
        groups[name], start = walk_wires[start:start + size], start + size
    assert groups == {name: list(getattr(layout, name)) for name in names}

    def program():
        qp.PhaseGradientStatePrep(layout.phase_grad)
        for _ in range(2):
            ControlledDFTHCWalk(params, tables, layout.qpe_control, **groups)

    resources = compiled_resources(program, layout.num_wires)
    expected = costs.controlled_walk_power(p, 2)
    print("C(W)^2 at H2_STO3G_TEST: T, M, A =", resources.toffoli, resources.measurements,
          resources.measured_and_uncomputes, "; costs.controlled_walk_power(p, 2) =", expected,
          "; 2 * measured_and_uncomputes(p, True) =", 2 * costs.measured_and_uncomputes(p, True))
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measured_and_uncomputes == 2 * costs.measured_and_uncomputes(p, True)


if __name__ == "__main__":
    main()
```

Its output (`stages/3A/interfaces-probe-3a.out`):

````text
required_wires: {'control': 1, 'outer': 16, 'inner': 23, 'select': 8, 'system': 4, 'phase_grad': 4, 'work': 8}
walk_wire_sizes, in wire_argnames order ('outer', 'inner', 'select', 'system', 'phase_grad', 'work') : (16, 23, 8, 4, 4, 8)
layout order starts with system: [0, 1, 2, 3] ; qpe_control [63] qpe_grad [64, 65, 66] ; wires 67
C(W)^2 at H2_STO3G_TEST: T, M, A = 320 100 220 ; costs.controlled_walk_power(p, 2) = Cost(toffoli=320, measurements=100) ; 2 * measured_and_uncomputes(p, True) = 220
````

**What Task 4.1 must respect.**
- **Signatures.** `DFTHCWalk(params, tables, outer, inner, select, system, phase_grad, work)` and `ControlledDFTHCWalk(params, tables, control, outer, inner, select, system, phase_grad, work)`; compilable `params = WalkParams.as_compilable()` and `tables = WalkTables.as_compilable()` (both stored canonically, so every call with the same `WalkParams` and `WalkTables` is one rule variant); `required_wires(params, tables)` = `{"control": 1}` (controlled only) then the layout's group sizes in the order `outer`, `inner`, `select`, `system`, `phase_grad`, `work`.
- **Wires.** Pass `WalkLayout`'s groups unchanged (`getattr(layout, name)`), with `layout.qpe_control` as `control`. `wire_argnames` order is not layout order (the layout starts with `system`): a flat `walk_wires` register for `QPERound` must concatenate the groups in `ControlledDFTHCWalk.wire_argnames[1:]` order, with `walk_wire_sizes` from `required_wires`, as the probe does. `WalkLayout`'s `work` group is the whole pool: the controlled walk needs all of it (`ControlledRefAB`), the plain walk one wire less.
- **Never adjoint or control the walks.** `ControlledWalkPower(power, params, tables, control, ...)` loops `ControlledDFTHCWalk` with `qp.for_loop` and a static bound, each iteration with identical compilable arguments and wires (spec 9.23); `qp.ctrl(DFTHCWalk(...))` does not compile, and `qp.adjoint` of either walk is never needed (QPE uses powers only). The inventory helpers `captured_targets` and `inventory` apply to the QPE program as they are (Task 4.2): its modified targets must again be `LEAF_ADJOINTS | TEMPLATE_ADJOINTS`.
- **State between rounds.** The phase gradient is prepared once, outside the walk, and returned unchanged. After every walk step every register but the six index registers of 𝒜 ∪ ℬ is |0⟩, and those carry the walk's state into the next step and the next QPE round; do not reset them between rounds. The controlled walk is correct there (E-3 after a step), not only from |0⟩.
- **Costs.** `costs.walk(p, True)` per controlled step (T_C(W) = 9,795, M_W = 2,524 at FeMoco-54; A = `costs.measured_and_uncomputes(p, True)` = 8,677); `costs.controlled_walk_power(p, power)`, `costs.qpe_round` and `costs.iterative_qpe` already build on them. The probe's C(W)² at `H2_STO3G_TEST`: T = 320 = 2 · 160, M = 100, A = 220.
- **Simulation.** `dfthc.testing.walk_sim` puts its own control wire after the layout (64 wires at `H2_STO3G_TEST`); QPE tests that use `WalkLayout.from_params(p, qpe_bits=t)` have 64 + t wires and two-word keys, about 25 to 30 times slower per input, which favours the toy walk (spec 11.4) for semantic QPE tests, as planned.
- **Compile cost for Task 4.5.** At `MID_NAPHTHALENE` the controlled walk compiles in 8.1 s at 588 MB and captures 58,310 equations; the assembly targets project to about 7,100 and 4,500 equations at FeMoco-54 (section "Compile cost and captured equations"), about 59,500 fewer than the PX stand-in whose FeMoco-54 controlled walk compiled in 2,648 MB.
