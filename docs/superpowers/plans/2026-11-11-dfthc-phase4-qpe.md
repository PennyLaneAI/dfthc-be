# DFTHC Phase 4, Tasks 4.1 and 4.2: QPE Operators, Driver, Toy Walk and QPE Counts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the iterative phase estimation of spec 11 on the controlled walk, the first half of D-4: the walk power `ControlledWalkPower` (spec 9.23), the round `QPERound` with its feed-forward phase corrections, the plain-function driver `iterative_phase_estimation`, the energy helpers, and the toy walk of spec 11.4 that checks the estimation exactly; then, the second half of D-4, compile the QPE program at the test presets and match `dfthc.costs.iterative_qpe` exactly.

**Architecture:** `src/dfthc/qpe.py` holds `ControlledWalkPower` (a `qp.for_loop` with the static bound `power` over the explicit `ControlledDFTHCWalk`), `QPERound` (an `Operator2` with dynamic `earlier_bits` of shape (t,), whose rule applies `H`, the walk power named by a string in the registry `WALK_POWERS`, one unhinted `PhaseGradientRotation` per earlier bit under `qp.cond` on wires it slices from the walk's `work` pool, and `H`), the driver, `walk_power_spec`, `spare_wires_needed` and `qpe_program` (the QPE program at a preset), and `phase_from_bits`, `energy_from_phase`, `physical_energy`. `src/dfthc/testing/toy_walk.py` holds the toy walk of H = (Z + X)/2, its explicit controlled class, which controls its two reflections, and its walk power, with dense references built from spec 11.4's definition. The QPE is checked exactly on the sparse simulator (the toy at t = 3, and the DFTHC walk itself at the two smallest walk shapes), statistically on `lightning.qubit`, and counted exactly when compiled.

**Tech Stack:** Python 3.12, PennyLane `Operator2` and Catalyst (`qp.qjit(capture=True)`, `qp.decompose`, `qp.analyze`, `qp.for_loop`, `qp.cond`), NumPy; the sparse simulator `dfthc.testing.sparse_sim` (`BranchPolicy`, `run_branches`) and `dfthc.testing.tape`; `dfthc.testing.walk_sim` (Task 3.3); `dfthc.data.encoded`; `dfthc.testing.watchdog.run_watched`; pytest with pytest-xdist; ruff.

This file holds Tasks 4.1 and 4.2 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 10). The master plan's one Phase 4 file (`phase4-application`) is split, as Phase 3's was (ledger decision "Phase 4 plan split", 2026-11-14): this file, `phase4-qpe`, holds Tasks 4.1 and 4.2, and `phase4-report` will hold Tasks 4.3 to 4.5, written once the walk classes are merged. Task 4.1 starts after G3; Task 4.2 follows it (wave 4, master plan section 5). The last section, "Interfaces for Tasks 4.3 to 4.5", states what the resource report consumes.

**Revision 2** (2026-11-14, after the GPT-5.6 plan review `.superpowers/sdd/phase4q-plan-review-1.md`, Needs fixes, 0 Critical, 4 Important, 1 Minor), with the orchestrator's decisions:
- **I-1** (`ControlledWalkPower`'s wires overcounted by one) applied: the interface script's `power_program` builds the operator on `WalkLayout.from_params(p)` with one control wire after it, 64 wires at `H2_STO3G_TEST` and 1,565 at FeMoco-54 (revision 1: 65 and 1,566, with an unused `qpe_grad` wire), and asserts that the count equals `sum(ControlledWalkPower.required_wires(...).values())`; the class's docstring example and the last section say the same.
- **I-2** (the correction register was not forced) applied with the review's option 1, which worked: every wire of a round is bound once. `QPERound` drops `correction_wires`; its flat `walk_wires` hold the walk power's groups and, only where needed, spare wires, and its rule slices the corrections' 2t − 1 wires from the first wire of the walk power's scratch group on: `work` for `ControlledWalkPower`, declared by the new class attribute `scratch_argname`; the toy walk declares none and gets 2t − 1 spare wires appended once. The QPE program at the H2 presets now has exactly `WalkLayout`'s wires with `qpe_bits = t` (64 + t at `H2_STO3G_TEST`, 80 + t at `H2_631G_TEST`), and its compiled T, M and A still equal `costs.iterative_qpe` at both presets for t = 1, 2, 3, and at `H2_STO3G_TEST` for t = 5, whose nine correction wires are the 8-wire pool and one spare wire. The toy distribution and its eigenvector check pass. `ControlledWalkPower`'s T3 and every branch of QPE of the walk (nightly and release, rerun) now assert that the work pool is back in |0⟩. At FeMoco-54 the 44-wire pool holds the corrections up to t = 22 (2t − 1 ≤ 44): the program has 1,565 + t wires (1,568 at t = 3; 1,585 at t = 20, 39 fewer than revision 1's 1,624), and from t = 23 on `qpe_program` appends 2t − 45 spare wires. Revision 1's failed reuse is preserved (`stages/4Q/alias-failure-rev1.log`: revision 1's round with `correction_wires` = `work[:3]` gives `'qref.operator' op qubit operands must refer to distinct qubits`). Change 4, Q-M3, Q-M7, Q-M8, Q-S4, Q-S5, Q-S7, Q-H2, the qubit tables and the last section are rewritten to match.
- **I-3** (the copied Global Constraints named two looping operators) applied: the new master-plan edit Q-M13 names `QROAMRead`, `ControlledWalkPower` and `ToyControlledWalkPower`, all three never adjointed or controlled. Its old text occurs once in the master plan on `main` and once in the shared fragment `templates/_global_constraints_g01.md`, which is where the master-plan template holds it: the template only includes the fragment, so the old text cannot occur in the template file itself. This plan includes a corrected copy, `stages/4Q/global-constraints-4q.md` (the fragment with Q-M13 applied, written by the edit script), instead of the fragment. Applying Q-M13 to the fragment would change six other reassembled plans by that one line (section "Shared edits"); the orchestrator decides. A new spec edit, Q-S10, corrects the same statement in spec 4.4 item 5.
- **I-4** (Task 4.2 observed GREEN before RED) applied: after writing the test, Step 2 applies the hinted-correction patch, Step 3 runs the two t = 3 count cases (`2 failed`), Step 4 restores `qpe.py` byte for byte, and Step 5 runs the unmodified file GREEN; the compile-recording step follows unchanged.
- **M-1** applied: the recorded field is `expected_clifford_totals`, the deterministic `CZ` and `Z` of `InnerSigns` and of the reflections plus the erase fix-ups' hinted conditionals at their expectation.
- **Count base:** the assembly plan's revision 4 (Task 3.3's six review tests) gives 1053 per commit with Tasks 3.3 to 3.5. Task 4.1 adds 54 per-commit tests (revision 1: 51; plus the registry and spare-wire tests, the t = 5 wiring case and the driver's aliasing case, less the round's correction-register case), so 1107, and 1108 after Task 4.2. The eight Phase 3 files were rechecked against revision 4's blocks, the replay rerun on `main` at `e447e53` (Task 3.3 merged), and the mutation check rerun for the six mutants whose killing tests changed and for five new ones.

**Code provenance.** Every code block below was executed on 2026-11-14 in a scratch replica of the repository (`.scratch/plan-check/repo-phase4q`, not committed), seeded from local `main` at `0c966ea` with `git archive`, plus the eight files that Tasks 3.3 to 3.5 create or change, from the assembly plan's replica `repo-phase3a`. `main` has since moved to `e447e53`: it merged Task 3.3, whose four files the replica's overlay supersedes, and changed only the ledger and the assembly plan otherwise, so the replica's code equals `main`'s with the overlay. Each of the eight is byte-identical to the last code block the assembly plan on `main` (revision 4) gives for it (`stages/4Q/check_phase3_blocks.py`, `phase3-blocks.log`); this plan treats their interfaces as final (section "Phase 3 interfaces this plan depends on"). The replica is not a git repository (the sandbox forbids creating one), so the dry run ran pytest the way `scripts/test.sh` does (the same environment, 2 workers, `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` exported, lowered priority) through `.scratch/plan-check/p4qtest.sh`, which runs every session under the watchdog with a 3,000 MB cap and a session lock. The two tasks were replayed in order, RED then GREEN, in one fresh copy of `main` (`e447e53`) with the eight Phase 3 files (`stages/4Q/run_stages_4q.sh`, log `stages/4Q/4q-stages.log`); every "Expected" line below is that replay's output, and the replayed files are byte-identical to the replica's. Further evidence, each with its log under `stages/4Q/`: the per-commit tier on the replica with both tasks (`4q-per-commit-tier.log`); every new per-commit test in one one-worker session (`4q-per-test-evidence.log`); Task 4.1's nightly checks (`qpe-nightly-4q.log`) and its release check (`qpe-walk-release-4q.log`); the probes that sized QPE of the walk (`.scratch/plan-check/probes/p4q/probe_real_qpe.py`; revision 2's `probe-real-qpe-r2.log`, revision 1's `probe-real-qpe-t2.log` and `probe-real-qpe-fails-c3.log`); the corrections' wires in tape mode and the t = 5 compile, whose corrections need one spare wire (`probes/p4q/probe_reuse.py`, `reuse-probe.log`); revision 1's failed reuse (`probes/p4q/probe_alias_rev1.py`, `alias-failure-rev1.log`); the QPE compiles with their watched time and memory (`qpe-compiles.log`); a mutation check of Task 4.1's tests (`mutate_4q.py`; revision 1's 15 mutants in `4q-mutants.log`, revision 2's 11 in `4q-mutants-r2.log`, and every mutant's patch checked against revision 2's code in `4q-mutants-check.log`); every docstring example run against its stated value and the nine spec 9.0 sections checked in order (`check_doc_examples_4q.py`, `doc-examples.log`); the interface script of the last section (`interfaces_probe_4q.py`, `interfaces-probe-4q.out`); Task 4.2's RED patch, generated from the replica's `qpe.py` (`make_red_patch_4q.py`, `red/*.sh`); and the document edits with the corrected Global Constraints and the shared-fragment report (`make_doc_edits_4q.py`, `doc-edits.md`, `global-constraints-4q.md`, `shared-fragment.md`).

**Load.** At most 2 pytest workers ran at any time, one pytest session at a time (the mutation check, the stage replay and the release check ran alone, in the background while only files were edited, each pytest session behind `p4qtest.sh`'s lock), and nothing FeMoco-shaped was compiled: FeMoco-54 values are asserted on the closed forms. Every process that compiled, simulated or ran pytest ran under the watchdog with a cap (3,000 MB; 4,096 MB for Task 4.2's subprocess compiles, as Task 3.4's): the probes, the docstring and interface scripts, every pytest session. In revision 2 the largest pytest session, the per-commit tier, peaked at 1,871 MB; the largest single simulation at 368 MB (the release check, QPE of the walk at `H2_STO3G_TEST` with t = 2, 487 s), the largest compile at 379 MB. The full per-commit tier ran once per revision. Every new per-commit test, 55 tests, ran in one session with one worker under the watchdog with `--durations=0`: `55 passed, 15 deselected in 33.85s` at a session peak of 398 MB, which bounds every test's peak; the longest took 5.1 s (the toy QPE program's rule listing and device-level closure, two compiles), four more 3.4 to 3.8 s (the walk-power and round compiles), the lightning test 2.5 s, the other 49 at most 1.4 s (`4q-per-test-evidence.log`). The checks that would exceed the per-test gate of about 1 GB and a few seconds run nightly or in the release tier: `ControlledWalkPower`'s three control states on the tape path (20 s), the verbose listing of the H2 QPE program (16 s), QPE of the walk (12 to 144 s nightly, 487 s release). No load-limit event occurred.

---

## Changes from the master plan, the spec and HANDOFF

The changes that touch a document are applied by the exact edits listed after this list (ids Q-M for the master plan, Q-S for the spec, Q-H for HANDOFF). No template was edited. Every old text of Q-M1 to Q-M12 also occurs exactly once in `.scratch/plan-check/templates/2026-11-11-dfthc-master-plan.md`, so the orchestrator mirrors them there when it applies them; Q-M13's occurs once in the shared fragment `templates/_global_constraints_g01.md` that the template includes (section "Shared edits").

1. **Plan files** (Q-M1, Q-M4, Q-M5). Section 1's file table and section 4's Plan column name `phase4-qpe` for Tasks 4.1 and 4.2 and `phase4-report` for Tasks 4.3 to 4.5.
2. **Provenance and findings** (Q-M2, Q-M3). Section 1 lists this file's dry run; section 3 gains item 24 with its compile, simulation and design findings.
3. **`QPERound`'s final form, from K-10** (Q-M7, Q-S4). K-10 passed (Task 0.8; ledger decision "K-10 consequences"), so `QPERound` is an `Operator2` with dynamic `earlier_bits` of shape (t,) in every round, and its rule holds spec 11.1's steps 1 to 4; the fallback (corrections in the driver) is not needed. The section "QPERound's final form" below states it. One check is added: `walk_wire_sizes` must equal the walk power's `required_wires` after `control` (`ValueError` otherwise).
4. **The corrections reuse the walk's `work` pool through one binding of every wire** (Q-M3, Q-M7, Q-M8, Q-S4, Q-S5, Q-S7, Q-H2). Spec 11.1 lets the correction wires reuse the walk's `work` pool, which is `|0⟩` whenever they run, but the master plan's `QPERound` takes `correction_wires` and `walk_wires` as two registers, and two registers of one operator that share a qubit do not compile (`'qref.operator' op qubit operands must refer to distinct qubits`; revision 1's attempt, `stages/4Q/alias-failure-rev1.log`). So `QPERound` drops `correction_wires`: its rule slices the corrections' `2t − 1` wires from `walk_wires`, from the first wire of the walk power's scratch group on. A walk-power class names that group in a class attribute, `scratch_argname`: its last wire group, which it takes and returns in `|0⟩` (`"work"` for `ControlledWalkPower`; `None` for `ToyControlledWalkPower`), and `register_walk_power` refuses any other group. Where the group holds fewer than `2t − 1` wires, `walk_wires` continues with spare wires after the walk's groups, `spare_wires_needed(walk_op, walk_wire_sizes, t)` of them, which the driver takes from its new argument `spare_wires` (default none) and `qpe_program` places after `WalkLayout`'s wires; the toy, without a scratch group, takes `2t − 1`. The driver checks that no wire is bound twice. The QPE program's wires are `WalkLayout`'s with `qpe_bits = t`, 64 + t at `H2_STO3G_TEST` (up to t = 4; its pool has 8 wires) and 1,565 + t at FeMoco-54 (up to t = 22; 44 wires), and `WalkLayout` is unchanged.
5. **`ControlledToyWalk` controls the toy's two reflections** (Q-M9, Q-S3, Q-S8). Spec 11.4 says "control only the toy reflection". The toy walk W = REF_anc PREP† SEL PREP has one reflection outside its U = PREP, so controlling REF_anc alone applies PREP† SEL PREP ≠ I on the control's |0⟩: not a controlled walk. Its SELECT is itself a reflection (a Hermitian unitary), the role `RefT2` plays in the DFTHC walk, so C(W) = C(REF_anc) PREP† C(SEL) PREP controls exactly the two reflections, as `ControlledDFTHCWalk` controls `RefAB` and `RefT2`. C(SEL) costs two Toffolis. The mutation check confirms the difference: controlling REF_anc alone fails 10 tests, among them the three control states and the exact distribution.
6. **New interfaces** (Q-M8, Q-M9, Q-S1, Q-S6). In `dfthc.qpe`: `WALK_POWERS` and `register_walk_power` (the registry the master plan names), the class attribute `scratch_argname` of a walk power, `spare_wires_needed(walk_op, walk_wire_sizes, num_bits)`, `walk_power_spec(params, tables, layout)`, `qpe_program(params, tables, num_bits)` (the QPE program at a preset, for Tasks 4.2 and 4.3) and `phase_from_bits(bits)`. The driver's last argument is `spare_wires=()` instead of `correction_wires`; it prepares the phase gradient on `qpe_grad` and also raises `ValueError` for fewer spare wires than the corrections need and for a wire bound twice. In `dfthc.testing.toy_walk`: `toy_walk_matrix`, `controlled_toy_walk_matrix`, `toy_walk_cost`, `toy_walk_power_cost`, `toy_walk_power_spec`, `toy_qpe_wires`, `toy_qpe`. The toy's costs live in the toy module, not in `dfthc.costs`: the toy is test scaffolding, not a spec 9.24 block.
7. **QPE of the DFTHC walk itself** (Q-M6, Q-M10, Q-S9). The master plan simulates QPE on the toy walk only, and allows a check on the real walk if it can be sized. Probes sized it under the watchdog: at `FAILS_C3` (58 wires plus the QPE registers: 60 wires at t = 1, 61 at t = 2, one key word) t = 1 takes 5 s and t = 2 88 s; at `H2_STO3G_TEST` (65 and 66 wires, two key words) t = 1 takes 24 s and t = 2 487 s (the release test itself), all below 370 MB. A new file, `tests/integration/test_qpe_walk.py`, runs them on the tape path (strict, `BranchPolicy` on the control, the erasures forced through `otherwise`) against the exact distribution of the encoded block, and checks that every branch ends with only the index registers of A ∪ B nonzero (the work pool, which the corrections borrow, back in |0⟩): `FAILS_C3` t = 1 under two schedules and t = 2, and `H2_STO3G_TEST` t = 1, nightly; `H2_STO3G_TEST` t = 2 in the release tier. They match to below 1e-15. Only these checks see the walk's state carried between rounds and the corrections on the real walk's pool. R-8 ("QPE is simulated only at toy sizes") holds: these are the two smallest walk shapes, the E checks' shapes.
8. **More tests than the master plan lists** (Q-M10). Every branch of the exact toy distribution must leave the toy wires in an eigenvector of W with eigenvalue e^{2πiφ}: the distribution alone is symmetric under φ → 1 − φ, so it cannot see W† in place of W or flipped corrections (both mutants pass the distribution and the lightning test and fail this check). `ControlledWalkPower` gets T2 to T6 (T3 with the control in |0⟩, |1⟩ and |+⟩ at power 2, nightly, which also asserts the clean work pool), `QPERound` T2, T4, T5 and T6; the toy QPE program's counts and inventory run per commit; the corrections' wires (`test_qpe_program_wires` at t = 3 and 5), the registry's check of `scratch_argname` and `spare_wires_needed` run per commit.
9. **Task 4.2** (Q-M11, Q-M12). "The hinted expected counts are recorded" is read as spec 7.6 and 13.3 use "hinted": each compile's `QPE_COMPILE` line records `expected_clifford_totals`, the expected `CZ` and `PauliZ` totals: the deterministic gates of `InnerSigns` and of the reflections plus the erase fix-ups' hinted conditionals, which `qp.analyze` counts at their expectation. The corrections are deliberately unhinted, so the compiled T is the worst case of spec 9.24 (a hint would halve the correction term; Task 4.2's RED step shows that it breaks the equality). A and the wire count are asserted too. The FeMoco-54 closed form runs per commit (it compiles nothing), the compiles nightly.
10. **`qp.for_loop` in three operators** (Q-M13, Q-S2, Q-S10, Q-H1). `ToyControlledWalkPower` loops as `ControlledWalkPower` does (master plan Task 4.1, spec 11.4); the master plan's Global Constraints rule 4, spec 4.4 item 5, spec 7.4 rule 4 and HANDOFF rule 4 said two operators. All four now name `QROAMRead`, `ControlledWalkPower` and `ToyControlledWalkPower`, all three never adjointed or controlled, and so does this plan's copy of the Global Constraints.
11. **Not added or changed:** no change to `dfthc.costs` (`controlled_walk_power`, `qpe_round` and `iterative_qpe` are spec 9.24's rows and match every compile); no `C(...)` or `Adjoint(...)` rule of any new class, and no `qp.ctrl` or `qp.adjoint` in any new rule or program; no hint on the corrections; the QPE program is not compiled at `MID_NAPHTHALENE` (ledger decision "Mid size for compiled T4": the QPE at the H2 presets only), while `ControlledWalkPower`, an L4 operator, gets its third compiled size there (nightly); no change to `WalkLayout`, `walk_sim` or `dfthc.testing.resources`.

Exact edits (master plan, spec and HANDOFF, against local `main`):

Each edit replaces one exact text of the file on local `main` at `e447e53`, where it occurs exactly once (checked by `.scratch/plan-check/stages/4Q/make_doc_edits_4q.py`, which also applies them in order). Every master-plan old text also occurs once where the orchestrator mirrors the edits: in `.scratch/plan-check/templates/2026-11-11-dfthc-master-plan.md`, except Q-M13's, which occurs once in the shared fragment `templates/_global_constraints_g01.md` that the template includes (see "Shared edits").

**Q-M1** (2026-11-11-dfthc-master-plan.md, section 1, the file table). Old:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase3-assembly.md` | Tasks 3.3 to 3.5 |
````

New:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase3-assembly.md` | Tasks 3.3 to 3.5 |
| `docs/superpowers/plans/2026-11-11-dfthc-phase4-qpe.md` | Tasks 4.1, 4.2 |
| `docs/superpowers/plans/2026-11-11-dfthc-phase4-report.md` | Tasks 4.3 to 4.5 (written when the walk classes are merged) |
````

**Q-M2** (2026-11-11-dfthc-master-plan.md, section 1, code provenance). Old:

````text
against `main` at `7777bc1` with Task 2E.3's two files and the select file's code; its preamble lists its changes to this plan.
````

New:

````text
against `main` at `7777bc1` with Task 2E.3's two files and the select file's code; its preamble lists its changes to this plan. The Phase 4 QPE file (`phase4-qpe`, Tasks 4.1 and 4.2) was executed the same way on 2026-11-14 (revision 2: against `main` at `e447e53`, with Task 3.3 merged, and the assembly file's revision 4 code for Tasks 3.3 to 3.5); its preamble lists its changes to this plan.
````

**Q-M3** (2026-11-11-dfthc-master-plan.md, section 3, a new item 24 after item 23 (the end of item 23 is shown)). Old:

````text
An uncontrolled `RefAB` inside the controlled walk passes every check that starts with the ancillas in |0⟩; the walk's own states (after one uncontrolled step) detect it.
````

New:

````text
An uncontrolled `RefAB` inside the controlled walk passes every check that starts with the ancillas in |0⟩; the walk's own states (after one uncontrolled step) detect it.
24. **The QPE as built** (Phase 4 QPE dry run, 2026-11-14, revision 2). K-10 passed, so `QPERound` is an `Operator2` whose rule holds spec 11.1's steps 1 to 4, with dynamic `earlier_bits` of shape (t,) in every round (both modes); it names the walk power by a string in `dfthc.qpe.WALK_POWERS` (`ControlledWalkPower`, the toy's `ToyControlledWalkPower`) and takes its compilable arguments and its groups, flattened in `wire_argnames` order (`WalkPowerSpec`). The corrections reuse the walk's `work` pool, as spec 11.1 allows, through one register bound once: `QPERound`'s `walk_wires` hold the walk's groups (then spare wires where the pool is too small), and its rule slices the corrections' 2t − 1 wires from the pool's first wire on. Binding the pool's wires again as a second register of the round does not compile (`'qref.operator' op qubit operands must refer to distinct qubits`), so revision 1 of the plan had given the corrections 2t − 1 wires of their own. So the QPE adds only `qpe_control` and `qpe_grad` to the walk: 64 + t wires at `H2_STO3G_TEST`, 1,565 + t at FeMoco-54, whose 44-wire pool holds the corrections up to t = 22. The corrections are unhinted, so `qp.analyze` counts them at their maximum branch and the compiled QPE program equals `costs.iterative_qpe` exactly at both test presets for t = 1, 2, 3 (`H2_STO3G_TEST`: T = 160, 481, 1,124; M = 51, 152, 353) and at `H2_STO3G_TEST` with t = 5 (9 correction wires: the 8 of the pool and one spare wire), each compile 3.8 to 4.7 s below 380 MB (5.9 s at t = 5), with expected `CZ` and `PauliZ` totals (the deterministic gates plus the hinted fix-ups) at (2^t − 1) times the controlled walk's. `ControlledToyWalk` controls the toy's two reflections, REF_anc and SELECT: the toy walk has one reflection outside its U = PREP, so controlling REF_anc alone leaves PREP† SEL PREP ≠ I on the control's |0⟩. The toy QPE gives spec 11.4's distribution exactly on the sparse simulator, each branch leaving an eigenvector of W with eigenvalue e^{2πiφ}, and within five standard deviations on `lightning.qubit` (2,000 shots). QPE of the DFTHC walk itself on the sparse simulator equals the encoded block's exact distribution to 1e-15, every branch ending with the work pool in |0⟩: `FAILS_C3` t = 1 and 2 in 5 and 88 s (60 and 61 wires, one key word), `H2_STO3G_TEST` t = 1 in 24 s and t = 2 in 487 s (65 and 66 wires, two key words).
````

**Q-M4** (2026-11-11-dfthc-master-plan.md, section 4, the Plan column of Tasks 4.1 to 4.4). Old:

````text
| 4.1 | QPE: `ControlledWalkPower`, `QPERound`, driver, energies, `ToyWalk`, `ControlledToyWalk` | A | G3 | phase4-application |
| 4.2 | QPE resource counts (compiled at the test presets) | B | 4.1 | phase4-application |
| 4.3 | Resource report and `dfthc.costs_reference` | B | 4.1 | phase4-application |
| 4.4 | Documentation set and docstring audit | B | 4.1, 4.2, 4.3 | phase4-application |
````

New:

````text
| 4.1 | QPE: `ControlledWalkPower`, `QPERound`, driver, energies, `ToyWalk`, `ControlledToyWalk` | A | G3 | phase4-qpe |
| 4.2 | QPE resource counts (compiled at the test presets) | B | 4.1 | phase4-qpe |
| 4.3 | Resource report and `dfthc.costs_reference` | B | 4.1 | phase4-report |
| 4.4 | Documentation set and docstring audit | B | 4.1, 4.2, 4.3 | phase4-report |
````

**Q-M5** (2026-11-11-dfthc-master-plan.md, section 4, the Plan column of Task 4.5). Old:

````text
| 4.5 | The single FeMoco walk and controlled-walk compile; report finalization | B | 4.4, the Q-1 decision, Q1b if chosen | phase4-application |
````

New:

````text
| 4.5 | The single FeMoco walk and controlled-walk compile; report finalization | B | 4.4, the Q-1 decision, Q1b if chosen | phase4-report |
````

**Q-M6** (2026-11-11-dfthc-master-plan.md, Task 4.1, files). Old:

````text
- **Files:** `src/dfthc/qpe.py`, `src/dfthc/testing/toy_walk.py`, `tests/integration/test_qpe_toy.py`, `tests/test_qpe.py`.
````

New:

````text
- **Files:** `src/dfthc/qpe.py`, `src/dfthc/testing/toy_walk.py`, `tests/integration/test_qpe_toy.py`, `tests/test_qpe.py`, `tests/integration/test_qpe_walk.py` (QPE of the walk itself on the sparse simulator; nightly and release).
````

**Q-M7** (2026-11-11-dfthc-master-plan.md, Task 4.1, interfaces: `QPERound`'s final form). Old:

````text
  - `QPERound(j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes, control, qpe_grad, correction_wires, walk_wires)`, with dynamic `earlier_bits` of shape (t,) (spec 11.1). An `Operator2` cannot take a Python callable, and it cannot mix compilable with hybrid (nested) wire arguments. So `walk_wires` is one flat register split by `walk_wire_sizes`, and the rule looks up `walk_op` in a registry of walk-power classes and calls it with `(2**j, *walk_compilable, control, *groups)`. The final form is fixed by the K-10 result in the Phase 4 plan.
````

New:

````text
  - `QPERound(j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes, control, qpe_grad, walk_wires)`, with dynamic `earlier_bits` of shape (t,) (spec 11.1). An `Operator2` cannot take a Python callable, and it cannot mix compilable with hybrid (nested) wire arguments. So `walk_wires` is one flat register split by `walk_wire_sizes`, and the rule looks up `walk_op` in a registry of walk-power classes and calls it with `(2**j, *walk_compilable, control, *groups)`. This is the final form (K-10 passed; Phase 4 QPE plan): the registry is `dfthc.qpe.WALK_POWERS` (filled by `register_walk_power`), `walk_wire_sizes` must equal the walk power's `required_wires` after `control`, and every round takes `qpe_grad` of t wires. The corrections' 2t − 1 wires are no register of their own: the rule slices them from `walk_wires`, from the first wire of the walk power's scratch group (its last group, named by its class attribute `scratch_argname`: `ControlledWalkPower`'s `work` pool, which it returns to |0⟩) on, into the spare wires that `walk_wires` holds after the walk's groups where that group is too small (`spare_wires_needed`; the toy walk lends none). So every wire is bound once: a second register on the same qubits does not compile.
````

**Q-M8** (2026-11-11-dfthc-master-plan.md, Task 4.1, interfaces: the driver and the energy helpers). Old:

````text
  - `iterative_phase_estimation(walk_power_factory, control, num_bits, qpe_grad, correction_wires) -> list`, the measured bits m_{t−1} … m_0, with `walk_power_factory() -> WalkPowerSpec`. It is a plain quantum function (Q-3) and raises `ValueError` if `num_bits > len(qpe_grad)`.
  - `energy_from_phase(phi, Lambda, shift=1.0) -> float` and `physical_energy(E_SA, e_sos_phys, core_energy=0.0) -> float`.
````

New:

````text
  - `iterative_phase_estimation(walk_power_factory, control, num_bits, qpe_grad, spare_wires=()) -> list`, the measured bits m_{t−1} … m_0, with `walk_power_factory() -> WalkPowerSpec`. It is a plain quantum function (Q-3), prepares the phase gradient on `qpe_grad`, and raises `ValueError` if `num_bits > len(qpe_grad)`, if `spare_wires` has fewer than `spare_wires_needed(walk_op, walk_wire_sizes, num_bits)` wires (2t − 1 less the walk power's scratch group, at least 0), or if a wire is bound twice. `walk_power_spec(params, tables, layout) -> WalkPowerSpec` flattens `ControlledWalkPower`'s groups in its `wire_argnames` order, and `qpe_program(params, tables, num_bits) -> (qfunc, num_wires)` is the QPE program at a preset (Tasks 4.2 and 4.3): `WalkLayout`'s wires with `qpe_bits = num_bits` (64 + t at `H2_STO3G_TEST`, 1,565 + t at FeMoco-54), and spare wires after them only where 2t − 1 exceeds the work pool (from t = 5 at `H2_STO3G_TEST`, from t = 23 at FeMoco-54).
  - `phase_from_bits(bits) -> float`, `energy_from_phase(phi, Lambda, shift=1.0) -> float` and `physical_energy(E_SA, e_sos_phys, core_energy=0.0) -> float`.
````

**Q-M9** (2026-11-11-dfthc-master-plan.md, Task 4.1, interfaces: the toy walk). Old:

````text
  - `ToyWalk(system, ancilla)`, `ControlledToyWalk(control, system, ancilla)` (control only the toy reflection; spec 11.4) and `ToyControlledWalkPower`, which loops over `ControlledToyWalk`.
````

New:

````text
  - `ToyWalk(system, ancilla)`, `ControlledToyWalk(control, system, ancilla)` (control only the toy's two reflections, REF_anc and SELECT; spec 11.4) and `ToyControlledWalkPower(power, control, system, ancilla)`, which loops over `ControlledToyWalk` and is registered in `WALK_POWERS` with no scratch group (its corrections take 2t − 1 spare wires); with the dense references `toy_walk_matrix()` and `controlled_toy_walk_matrix()`, the costs `toy_walk_cost(controlled)` and `toy_walk_power_cost(power)`, `toy_walk_power_spec(system, ancilla)`, `toy_qpe_wires(num_bits)` and `toy_qpe(num_bits)`.
````

**Q-M10** (2026-11-11-dfthc-master-plan.md, Task 4.1, acceptance tests (the last item and the definition of done)). Old:

````text
  - Energy helpers match spec 11.1.
- **Definition of done:** D-4, first half.
````

New:

````text
  - Energy helpers match spec 11.1.
  - Each branch of the exact toy distribution leaves the toy wires in an eigenvector of W with eigenvalue e^{2πiφ} (spec 11.1's phase convention), the control and the spare wires in |0⟩.
  - The corrections' wires: `qpe_program` at `H2_STO3G_TEST` has exactly `WalkLayout`'s wires at t = 3 (the corrections in the work pool) and one spare wire more at t = 5; round 0's rotations take their scratch and work wires from the pool's first wire on.
  - `ControlledToyWalk` and `ToyControlledWalkPower` (p = 3): the compiled matrix is |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ W^p, applied with the control in |0⟩, |1⟩ and |+⟩; `ControlledWalkPower` (p = 2) likewise on the tape path against two `DFTHCWalk` steps, every output with the work pool in |0⟩ (nightly).
  - T4: `ControlledWalkPower` (p = 2) at both test presets (per commit) and `MID_NAPHTHALENE` (nightly); round 0 of a 2-bit `QPERound` at `H2_STO3G_TEST` against `costs.qpe_round`; the toy QPE program for t = 1, 2, 3; FeMoco-54 on the closed forms.
  - QPE of the DFTHC walk itself on the sparse simulator (`tests/integration/test_qpe_walk.py`) equals the encoded block's exact distribution, at `H2_STO3G_TEST` (t = 1 nightly, t = 2 release) and `FAILS_C3` (t = 1 under two schedules and t = 2, nightly), the erasures forced through `otherwise`, every branch ending with the work pool in |0⟩.
- **Definition of done:** D-4, first half.
````

**Q-M11** (2026-11-11-dfthc-master-plan.md, Task 4.2, files). Old:

````text
- **Files:** `tests/integration/test_qpe_counts.py` (marked `slow`).
````

New:

````text
- **Files:** `tests/integration/test_qpe_counts.py` (the compiles marked `slow`; the FeMoco-54 closed form per commit).
````

**Q-M12** (2026-11-11-dfthc-master-plan.md, Task 4.2, acceptance tests (the last three items)). Old:

````text
  - The hinted expected counts are recorded.
  - The inventory test of Task 3.4 passes for the QPE program at `H2_STO3G_TEST` with t = 2.
  - It is not compiled at FeMoco-54 shape (spec 11.3). There the report evaluates the closed form with the controlled-walk counts of Task 4.5.
````

New:

````text
  - The hinted expected counts are recorded: every compile runs in a subprocess under `run_watched` (4 GB cap), and its `QPE_COMPILE` line holds the watched time and peak memory and the expected `CZ` and `PauliZ` totals (`expected_clifford_totals`: the deterministic gates of `InnerSigns` and the reflections plus the erase fix-ups' hinted conditionals, which `qp.analyze` counts at their expectation; spec 7.6). The corrections are not hinted, so T and A are the worst case; A = (2^t − 1) A_C(W) + (t³ − t)/6 is asserted too, and the wire count is `WalkLayout`'s with `qpe_bits = t`.
  - The inventory test of Task 3.4 passes for the QPE program at `H2_STO3G_TEST` with t = 2.
  - It is not compiled at FeMoco-54 shape (spec 11.3). There the report evaluates the closed form with the controlled-walk counts of Task 4.5. A per-commit test asserts that closed form for t = 1, 2, 3 (T = 9,795, 29,386, 68,569; M = 2,525, 7,574, 17,671).
````

**Q-M13** (2026-11-11-dfthc-master-plan.md, Global Constraints, rule 4 (its last clause; in the template, the shared fragment `templates/_global_constraints_g01.md`)). Old:

````text
`qp.for_loop` only in `QROAMRead` (one loop per unary leaf, with `qp.cond` on table bits) and `ControlledWalkPower`, both never adjointed or controlled.
````

New:

````text
`qp.for_loop` only in `QROAMRead` (one loop per unary leaf, with `qp.cond` on table bits), `ControlledWalkPower` and `ToyControlledWalkPower`, all three never adjointed or controlled.
````

**Q-S1** (2026-11-11-dfthc-pennylane-design.md, 5.2, the `qpe.py` line). Old:

````text
    qpe.py                    ControlledWalkPower, QPERound, iterative_phase_estimation(), energy_from_phase(), physical_energy()
````

New:

````text
    qpe.py                    ControlledWalkPower, QPERound, WalkPowerSpec, spare_wires_needed(), iterative_phase_estimation(), qpe_program(), phase_from_bits(), energy_from_phase(), physical_energy()
````

**Q-S2** (2026-11-11-dfthc-pennylane-design.md, 7.4 rule 4, the `qp.for_loop` item). Old:

````text
   - `qp.for_loop` appears in two operators, both never adjointed and never controlled: `QROAMRead`, whose per-leaf loads also carry `qp.cond` on table bits (§9.3, K-1 lever P), and `ControlledWalkPower`.
````

New:

````text
   - `qp.for_loop` appears in three operators, all never adjointed and never controlled: `QROAMRead`, whose per-leaf loads also carry `qp.cond` on table bits (§9.3, K-1 lever P), `ControlledWalkPower` and its toy twin `ToyControlledWalkPower` (§11.4).
````

**Q-S3** (2026-11-11-dfthc-pennylane-design.md, 7.5, the `ControlledToyWalk` row). Old:

````text
| `ControlledToyWalk` | control only the toy reflection (§11.4) | `C(ToyWalk)` |
````

New:

````text
| `ControlledToyWalk` | control only the toy's two reflections, $\mathrm{REF}_{\mathrm{anc}}$ and SELECT (§11.4) | `C(ToyWalk)` |
````

**Q-S4** (2026-11-11-dfthc-pennylane-design.md, 11.1, the K-10 paragraph). Old:

````text
Whether such measurement-valued arrays work as dynamic arguments in both capture and tape mode is decided by the Phase 0 probe K-10. If they do not, steps 1–4 stay in the driver, with the same gates and counts; `ControlledWalkPower` remains the operator.
````

New:

````text
K-10 passed (Task 0.8): such arrays work as dynamic arguments in both modes, so `QPERound` holds steps 1–4. Its final signature is `QPERound(j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes, control, qpe_grad, walk_wires)`. An `Operator2` takes no Python callable and cannot mix compilable with hybrid (nested) wire arguments, so the walk power is named: `walk_op` is a class name in the registry `dfthc.qpe.WALK_POWERS` (`ControlledWalkPower`, and the toy's `ToyControlledWalkPower`), `walk_compilable` its compilable arguments after `power`, and `walk_wires` its wire groups after `control`, concatenated in its `wire_argnames` order and split by `walk_wire_sizes` (its `required_wires` after `control`), then the spare wires of the correction register below; the driver takes them as a `WalkPowerSpec` from `walk_power_factory()`. Every round takes `qpe_grad` of $t$ wires. The corrections are not hinted, so `qp.analyze` counts each at its maximum branch (§11.3).
````

**Q-S5** (2026-11-11-dfthc-pennylane-design.md, 11.1, the correction register (its last sentence)). Old:

````text
The correction scratch and work wires (`correction_wires`, $2t - 1$ wires) may reuse the walk's `work` pool, because corrections never overlap a walk power.
````

New:

````text
The correction scratch and work wires ($2t - 1$ wires) reuse the walk's `work` pool, because corrections never overlap a walk power and the controlled walk returns the pool to $\lvert 0\rangle$. They are no register of their own, because two wire registers of one operator that share a qubit do not compile (`'qref.operator' op qubit operands must refer to distinct qubits`): `QPERound` binds every wire once, and its rule slices the corrections' wires from `walk_wires`, from the first wire of the walk power's scratch group (its last group, named by the class attribute `scratch_argname`: `work` for `ControlledWalkPower`) on, into spare wires that `walk_wires` holds after the walk's groups where the pool has fewer than $2t - 1$ wires (`spare_wires_needed`: from $t = 5$ at `H2_STO3G_TEST`, whose pool has 8 wires, and from $t = 23$ at FeMoco-54, 44). The toy walk lends no scratch group, so its corrections take $2t - 1$ spare wires. The driver prepares `qpe_grad` and checks that no wire is bound twice; `dfthc.qpe.qpe_program` places the spare wires, if any, after `WalkLayout`'s wires.
````

**Q-S6** (2026-11-11-dfthc-pennylane-design.md, 11.1, energies (the phase)). Old:

````text
**Energies.** The estimated phase is $\varphi = \sum_j m_j 2^{-(j+1)}$, and $\theta = 2\pi\varphi$.
````

New:

````text
**Energies.** The estimated phase is $\varphi = \sum_j m_j 2^{-(j+1)}$, and $\theta = 2\pi\varphi$ (`phase_from_bits`, with the bits in the driver's order $m_{t-1}, \dots, m_0$). The bits estimate the eigenvalue $e^{2\pi i\varphi}$ of $W$ itself; an exact $t$-bit phase is returned with certainty.
````

**Q-S7** (2026-11-11-dfthc-pennylane-design.md, 11.3 (the end of the paragraph)). Old:

````text
(§13.3), for $t \le 3$ and for larger $t$.
````

New:

````text
(§13.3), for $t \le 3$ and for larger $t$. The program's wires are `WalkLayout`'s with `qpe_bits` $= t$, the corrections in the `work` pool (§11.1): $64 + t$ at `H2_STO3G_TEST` for $t \le 4$, $1{,}565 + t$ at FeMoco-54 for $t \le 22$; beyond, the $2t - 45$ spare wires of §11.1 follow at FeMoco-54.
````

**Q-S8** (2026-11-11-dfthc-pennylane-design.md, 11.4, the controlled toy walk). Old:

````text
- Its controlled form is the class `ControlledToyWalk` (control only the toy reflection; §7.4 rule 6), which `ToyControlledWalkPower` loops over.
````

New:

````text
- Its controlled form is the class `ControlledToyWalk` (§7.4 rule 6), which `ToyControlledWalkPower` loops over. It controls only the toy's two reflections, $\mathrm{REF}_{\mathrm{anc}}$ and SELECT (itself a reflection: a Hermitian unitary): $C(W) = C(\mathrm{REF}_{\mathrm{anc}})\,\mathrm{PREP}^\dagger\,C(\mathrm{SEL})\,\mathrm{PREP}$, as `ControlledDFTHCWalk` controls `RefAB` and `RefT2`. Controlling $\mathrm{REF}_{\mathrm{anc}}$ alone would leave $\mathrm{PREP}^\dagger\,\mathrm{SEL}\,\mathrm{PREP} \ne I$ on the control's $\lvert 0\rangle$: the toy walk has one reflection outside $U = \mathrm{PREP}$, the DFTHC walk two.
````

**Q-S9** (2026-11-11-dfthc-pennylane-design.md, 11.4, a third check after the compiled smoke test). Old:

````text
- **Compiled smoke test:** `lightning.qubit` with 2,000 shots, comparing within five standard deviations. It uses no postselection and no measurement inside `qp.cond` (§14, L-15).
````

New:

````text
- **Compiled smoke test:** `lightning.qubit` with 2,000 shots, comparing within five standard deviations. It uses no postselection and no measurement inside `qp.cond` (§14, L-15).
- **The walk itself:** QPE of the DFTHC walk also runs on the sparse simulator, at the two smallest walk shapes, `H2_STO3G_TEST` ($t = 1$ nightly, $t = 2$ release) and `FAILS_C3` of §12.4 ($t = 1, 2$, nightly), with the erase measurements forced through the branch policy's `otherwise`. The control bits' distribution equals the exact distribution of the encoded block: each eigenvalue $a$ of $\tilde H_{\mathrm{SA}}/\tilde\Lambda - I$ splits its weight equally between the phases $\pm\arccos(a)/2\pi$. These are the toy sizes of R-8.
````

**Q-S10** (2026-11-11-dfthc-pennylane-design.md, 4.4, item 5 (its last sentence)). Old:

````text
Since G0 the two operators with loops, `QROAMRead` (item 12) and `ControlledWalkPower`, are never adjointed or controlled, by construction (item 13).
````

New:

````text
Since G0 the operators with loops, `QROAMRead` (item 12), `ControlledWalkPower` and its toy twin `ToyControlledWalkPower` (§11.4), are never adjointed or controlled, by construction (item 13).
````

**Q-H1** (HANDOFF.md, section 5, rule 4). Old:

````text
   inside erase operators and the QPE wrapper. `qp.for_loop` appears only in `QROAMRead` (one
   loop per unary leaf, with `qp.cond` on table bits; spec 9.3) and `ControlledWalkPower`, both
   never adjointed or controlled (rule 6).
````

New:

````text
   inside erase operators and the QPE wrapper (`QPERound`'s corrections). `qp.for_loop`
   appears only in `QROAMRead` (one loop per unary leaf, with `qp.cond` on table bits;
   spec 9.3), `ControlledWalkPower` and its toy twin `ToyControlledWalkPower`, all never
   adjointed or controlled (rule 6).
````

**Q-H2** (HANDOFF.md, section 3.2, a pitfall after the wire-register one). Old:

````text
- **Inside a rule, wire registers are arrays.** Never test them for truth (`work[:n] or None`
  raises); compute lengths from compilable arguments and slice explicitly.
````

New:

````text
- **Inside a rule, wire registers are arrays.** Never test them for truth (`work[:n] or None`
  raises); compute lengths from compilable arguments and slice explicitly.
- **The wires of one operator must be distinct qubits.** Registers that share a wire fail to
  compile (`'qref.operator' op qubit operands must refer to distinct qubits`), even when the
  shared wires are idle in the other register. To reuse a pool, bind it once and slice it
  inside the rule: `QPERound`'s corrections take the walk's `work` pool from its `walk_wires`.
````

## Shared edits

No merged module changes, and Tasks 3.3 to 3.5's files are used as the assembly plan builds them. `dfthc.qpe` imports `dfthc.ops.walk._validate`, the walk classes' private validator (as `walk.py` imports `prepare._check_outer`); a public validator is a Task 5.1 cleanup (risk 5).

One plan fragment is shared: the Global Constraints in `templates/_global_constraints_g01.md`, whose rule 4 is Q-M13's old text. This plan did not edit it; it includes `stages/4Q/global-constraints-4q.md` instead, the fragment with Q-M13 applied (the only difference). The templates that include the fragment, and their plans on `main` (`stages/4Q/shared-fragment.md`):

`templates/_global_constraints_g01.md` holds Q-M13's old text once. Templates that include it, and the count of the old text in their plan on `main` at `e447e53`:

- `2026-11-11-dfthc-master-plan.md`: 1 on `main`
- `2026-11-11-dfthc-phase1-primitives.md`: 1 on `main`
- `2026-11-11-dfthc-phase2-blocks.md`: 1 on `main`
- `2026-11-11-dfthc-phase2-prepare.md`: 1 on `main`
- `2026-11-11-dfthc-phase3-assembly.md`: 1 on `main`
- `2026-11-11-dfthc-phase3-select.md`: 1 on `main`

If the orchestrator applies Q-M13 to the fragment, the reassembled master plan, phase1-primitives, phase2-blocks, phase2-prepare, phase3-select and phase3-assembly each change by exactly that one line, and each of their files on `main` holds the old text once, so the same edit keeps them byte-identical to their reassembly; this plan's template can then include the fragment again with no change to its output. The older fragment `templates/_global_constraints.md` (Phase 0 plans, phase1-data, phase1-simulator) has the pre-G0 rule 4 (`qp.for_loop` only in `ControlledWalkPower`) and is left as those finished tasks' record.

## QPERound's final form

K-10 passed in both modes (Task 0.8: `earlier_bits` of shape (t,), exact in tape mode with deferred measurements and under capture; the Opus review's stale-bits mutation failed), so spec 11.1's preferred form holds, as built and dry-run here:

- `QPERound(j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes, control, qpe_grad, walk_wires)`: compilable `j`, `num_bits`, `walk_op` (a class name in `dfthc.qpe.WALK_POWERS`), `walk_compilable` (the walk power's compilable arguments after `power`, a tuple) and `walk_wire_sizes` (the walk power's `required_wires` after `control`); dynamic `earlier_bits`; wires `control` (1), `qpe_grad` (`num_bits`) and `walk_wires`: the walk power's groups after `control`, concatenated in its `wire_argnames` order, then `spare_wires_needed(walk_op, walk_wire_sizes, num_bits)` spare wires (none for `ControlledWalkPower` while `2 num_bits − 1` fits its `work` pool; `2 num_bits − 1` for the toy). Every round of one estimation has the same registers and the same `earlier_bits` shape, and every wire is bound once.
- `earlier_bits[l]` is m_l for l > j and `False` for l ≤ j; the rule reads only l > j. The constructor canonicalizes it (L-23): a boolean `jax` array under capture (the driver sets entry j with `.at[j].set`), an object array of `MeasurementValue`s and `False` in tape mode.
- The rule: `groups` split from `walk_wires` by `walk_wire_sizes`; `corrections` = the `2 num_bits − 1` wires of `walk_wires` from the first wire of the walk power's scratch group on (its last group if its class's `scratch_argname` names it, else the first spare wire); `H(control)`; `WALK_POWERS[walk_op](2**j, *walk_compilable, control, *groups)`; for l = j + 1, …, t − 1 with d = l − j + 1, `qp.cond(earlier_bits[l], correction)()` whose branch builds `PhaseGradientRotation(2**d − 1, d, control, qpe_grad[:d], corrections[:d], corrections[d:2d − 1])`, the phase e^{−2πi/2^d} on |1⟩ (the top d wires of a t-bit gradient are a d-bit gradient); `H(control)`. The corrections run after the walk power, whose scratch group is then |0⟩, and return it to |0⟩. They are not hinted, so `qp.analyze` counts each at its maximum branch: T = 2^j T_C(W) + Σ_{l>j}(l − j), `costs.qpe_round`.
- The driver measures the control with reset after each round, so the control starts every round in |0⟩, and returns the bits m_{t−1}, …, m_0; φ = Σ_j m_j 2^{−(j+1)} (`phase_from_bits`), and the bits estimate the eigenvalue e^{2πiφ} of W itself.

## How this plan meets the binding ledger rows

| Ledger row or decision | What this plan does |
| --- | --- |
| Decision "K-1 at G0" (PX: explicit classes) and master plan section 3, item 20 | `ControlledWalkPower` loops over the explicit `ControlledDFTHCWalk`; `ControlledToyWalk` is an explicit controlled class with its own T2 to T6 (control in |0⟩, |1⟩ and |+⟩ on `lightning.qubit`, the whole matrix); `ToyControlledWalkPower` loops over it. No new rule or program calls `qp.ctrl` or `qp.adjoint`. The inventory of the toy QPE program (per commit) has no `C(...)` target and no modified dfthc or template target; that of the QPE program at `H2_STO3G_TEST` with t = 2 (Task 4.2) has no `C(...)` target and exactly the walk's `LEAF_ADJOINTS | TEMPLATE_ADJOINTS`. `qp.ctrl` of a walk failing loudly stays pinned by Task 3.4. |
| Decision "K-10 consequences" (2026-11-12): `QPERound` takes a dynamic `earlier_bits` of shape (t,); masks derived inside rules; Q-1 stays open to G4, preferring variant (b) | `QPERound`'s final form above; `test_earlier_bits_have_shape_t_in_every_round` checks shape (t,) in every round, round t − 1 included, in tape mode (entry l is the very `MeasurementValue` of round l) and under capture. The QPE passes no masks (the erases inside the walk keep deriving theirs). Nothing here touches the unprepares, so Q-1 stays open; `qpe_program` builds on `WalkLayout`, so a variant (b) layout change carries over. |
| Q-3 (the driver is a plain quantum function) | `iterative_phase_estimation` is a function returning the measured bits; its per-round content is `QPERound` and the walk power, both `Operator2`s. |
| Decision "Phase 4 plan split" (2026-11-14) | Q-M1, Q-M4, Q-M5. |
| 1A.3 notes for Task 4.1 (`.superpowers/sdd/notes-for-1B.2.md`): `physical_energy` only for η-electron eigenstates, tested at STO-3G, `e_sos_walk` rejected in its place; never compare QPE outputs with `e_gap` at inexact instances | `test_physical_energy_is_the_physical_frame` (two-electron sector of `h2_sto3g_R2_B2_C1`: E_SA + E_SoS^phys = e_upper to 1e-10, the walk frame off by Δ_η = −1.3209547); every walk phase and QPE distribution is compared with the encoded block (`encoded_block`, `WalkCase.block`), never with `summary.json`. |
| 1A.5 Minor 3 (take Λ̃ from `encoded_block`) | `test_energy_from_phase_inverts_the_walk_phases` and the QPE-of-the-walk checks take H̃_SA and Λ̃ from `encoded_block`. |
| G0.2 and 1B.2 (Opus): `run_branches` with `strict=True` where erasures go to `otherwise`; never `otherwise=SeededPolicy` where independent samples per branch are wanted; the toy has no erasures | The toy distribution uses `BranchPolicy` without `otherwise`, so any measurement off the control would raise; the QPE-of-the-walk checks force the erasures with `otherwise=ForcedPolicy(...)` (seeded and alternating schedules), strict. |
| 1C.1 (spec 11.1's correction form, `cond_probe.py`) | The corrections are `qp.cond(earlier_bits[l], correction)()` with a branch function that builds the rotation and returns nothing. |
| 3.2 notes for Tasks 3.3 to 3.5: keep the controlled-walk check after an uncontrolled step | `ControlledWalkPower`'s T3 at power 2 runs its second step from the walk's own state; QPE of the walk carries the walk's state between rounds (spec 11.1), from round t − 1 to round 0. |
| Test-design lessons 2D.2 (all-ones schedules see only mask parities), 2E.5 (tape-path checks where AND pairs matter), 3.1 (branch classes as the prepares leave them) | The QPE-of-the-walk checks use seeded and alternating erase schedules. The corrections' `SemiAdder` ANDs run on the tape path in strict mode in the exact toy distribution (`Conditional(TemporaryAND)` and its measured uncompute, preconditions asserted); the toy walks have no AND pairs (plain `Toffoli`s). QPE of the walk runs the whole walk from the states the prepares leave, failed b and failed x_o included at `FAILS_C3`. |
| Master plan section 3, item 15 and spec 7.6 (hints) | The corrections are unhinted on purpose (worst case = closed form); the expected `CZ` and `PauliZ` totals, deterministic gates plus the erase fix-ups' hinted conditionals, are recorded (Task 4.2). |
| Heavy-runs log: the earlier writers' breaches | Every process that compiled, simulated or ran pytest ran under the watchdog with a cap, one pytest session at a time behind a lock; the real-walk QPE was sized on the smallest case first. No load-limit event. |

## Counts, closed forms and qubits

Compiled at the test presets (T / M / A; `null.qubit`, each equal to the closed form); FeMoco-54 on the closed forms only:

| Program | `H2_STO3G_TEST` | `H2_631G_TEST` | `MID_NAPHTHALENE` | FeMoco-54 (closed form) |
| --- | --- | --- | --- | --- |
| `ControlledDFTHCWalk` (T_C(W), M_W, A_C(W); Task 3.4) | 160 / 50 / 110 | 242 / 68 / 181 | 1,915 / 502 / 1,680 | 9,795 / 2,524 / 8,677 |
| `ControlledWalkPower`, p = 2 | 320 / 100 / 220 | 484 / 136 / 362 | 3,830 / 1,004 / 3,360 (nightly) | 19,590 / 5,048 / 17,354 |
| `QPERound`, round 0 of 2 bits | 161 / 50 / 111 | | | 9,796 / 2,524 / 8,678 |
| QPE program, t = 1 | 160 / 51 / 110 | 242 / 69 / 181 | | 9,795 / 2,525 / 8,677 |
| QPE program, t = 2 | 481 / 152 / 331 | 727 / 206 / 544 | | 29,386 / 7,574 / 26,032 |
| QPE program, t = 3 | 1,124 / 353 / 774 | 1,698 / 479 / 1,271 | | 68,569 / 17,671 / 60,743 |
| `ControlledWalkPower` wires | 64 | 80 | 427 | 1,565 |
| QPE wires, t = 1, 2, 3 | 65, 66, 67 | 81, 82, 83 | | 1,566, 1,567, 1,568 |

The QPE program is T = (2^t − 1) T_C(W) + (t³ − t)/6, M = (2^t − 1) M_W + t, A = (2^t − 1) A_C(W) + (t³ − t)/6 (`costs.iterative_qpe`; the corrections' A is their `SemiAdder`s' measured uncomputes, d − 1 per correction, at the maximum branch). Its expected `CZ` and `PauliZ` totals (deterministic gates plus the hinted erase fix-ups) are `CZ` 16, 48, 112 and `PauliZ` 6, 18, 42 at `H2_STO3G_TEST` (t = 1, 2, 3; the controlled walk 16 and 6), `CZ` 26, 78, 182 and `PauliZ` 8, 24, 56 at `H2_631G_TEST`: (2^t − 1) times the controlled walk's. The toy: `ToyWalk` T = 0; `ControlledToyWalk` T = 2 (two plain `Toffoli`s, A = 0); `ToyControlledWalkPower(p)` T = 2p; the toy QPE program T = 2, 7, 18, M = 1, 2, 3, A = 0, 1, 4 for t = 1, 2, 3, on 3t + 2 wires (its 2t − 1 spare wires included).

**Qubits.** A walk power needs the walk's groups and its control: `WalkLayout.from_params(p).num_wires + 1`, which equals `sum(ControlledWalkPower.required_wires(...).values())`. The QPE program needs `WalkLayout.from_params(p, qpe_bits=t).num_wires` = the walk's wires + 1 + t (`qpe_control`, `qpe_grad`), plus `spare_wires_needed(...)` = max(0, 2t − 1 − `WalkLayout.work_size(p)`) spare wires, because the corrections borrow the controlled walk's `work` pool. The pools are 8 wires at `H2_STO3G_TEST`, 9 at `H2_631G_TEST` and 44 at FeMoco-54, so no spare wire is needed up to t = 4, 5 and 22. At FeMoco-54:

| t | 1 | 2 | 3 | 20 | 22 | 23 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QPE wires | 1,566 | 1,567 | 1,568 | 1,585 | 1,587 | 1,589 | 1,595 |
| of which spare | 0 | 0 | 0 | 0 | 0 | 1 | 5 |

Revision 1's own correction register cost 2t − 1 more: 1,573 at t = 3 and 1,624 at t = 20.

Against the paper and dfthcbe: dfthcbe builds no phase estimation, and the paper costs it as its number of controlled-walk queries (about λ_eff/δE, spec 3.3) times the controlled walk's 9,795 (ours; the paper's Table III walk is 9,997). The per-round corrections add (t³ − t)/6 Toffolis in the worst case (half of it expected), 4 at t = 3 and 1,330 at t = 20, against (2^t − 1) controlled walks; the QPE adds t + 1 qubits to `WalkLayout`'s 1,564 up to t = 22 (change 4).

## Compile and simulation cost

Each QPE compile alone in a subprocess under `run_watched` (Task 4.2's Step 6, one worker; `qpe-compiles.log`, while another lane ran its tests): every one 3.8 to 4.7 s of compile and 6.8 to 7.9 s of subprocess, below 380 MB, nearly the controlled walk's own (3.8 and 4.2 s, 348 and 355 MB): the loops compile their body once, and `ControlledDFTHCWalk` with identical arguments is one rule variant in every round. The t = 5 program at `H2_STO3G_TEST` (31 controlled walks, its corrections on the pool and one spare wire) compiled in 5.9 s at 377 MB (`reuse-probe.log`).

| Program | `H2_STO3G_TEST` compile, peak | `H2_631G_TEST` compile, peak |
| --- | --- | --- |
| controlled walk | 3.8 s, 348 MB | 4.2 s, 355 MB |
| QPE t = 1 | 4.2 s, 355 MB | 4.5 s, 355 MB |
| QPE t = 2 | 4.2 s, 357 MB | 4.6 s, 371 MB |
| QPE t = 3 | 4.5 s, 364 MB | 4.7 s, 379 MB |

QPE on the sparse simulator (tape path, strict; revision 2's probes, each alone under the watchdog, `probe-real-qpe-r2.log`, and for `H2_STO3G_TEST` t = 2 the release test alone, `qpe-walk-release-4q.log`):

| Program | Wires (key words) | Operations, measurements | Terms per branch | Time | Peak |
| --- | --- | --- | --- | --- | --- |
| toy, t = 3 | 11 (1) | 110, 3 | | 0.04 s | below 250 MB (with a compile) |
| walk at `FAILS_C3`, t = 1 | 60 (1) | 1,883, 36 | 100,352 | 5.3 s | 212 MB |
| walk at `FAILS_C3`, t = 2 | 61 (1) | 5,651, 107 | 200,704 | 88.4 s | 349 MB |
| walk at `H2_STO3G_TEST`, t = 1 | 65 (2) | 1,280, 51 | 24,576 | 24.0 s | 183 MB |
| walk at `H2_STO3G_TEST`, t = 2 | 66 (2) | 3,842, 152 | 51,200 | 487 s (test) | 368 MB (session) |

Revision 1's probes, with its correction register (2t − 1 more wires; the same operations, measurements and terms), took 5.7 s, 95.7 s, 44.5 s and 677.5 s.

**Tier budgets.** Per commit: Task 4.1 adds 54 tests and Task 4.2 one; together 55 tests in 33.9 s with one worker at 398 MB ("Load" above); the two-worker tier with both tasks took 554 s at 1,871 MB ("Expected test counts" below), against 15 minutes and 4 GB. Nightly: Task 4.1 adds 7 tests (`7 passed in 201.15s` with 2 workers, 1,288 MB, `qpe-nightly-4q.log`): `test_qpe_walk.py`'s four (`[fails-c3-2-seeded]` 144 s, `[sto3g-1-seeded]` 50 s, the two `fails-c3-1` cases 12 s each) and `test_qpe.py`'s three (the three control states 20 s, the `MID_NAPHTHALENE` power 19 s, the listing 16 s); Task 4.2 adds 7 (`7 passed` in 64 s with one worker); together about 5 minutes with 2 workers on top of Phase 3's 4 to 6 minutes, against 4 h. Release: the `H2_STO3G_TEST` t = 2 check, about 8 minutes (`1 passed, 5 deselected in 487.43s`, alone, 368 MB) on top of the assembly plan's projection of about 25 minutes, against 8 h. No escalation is needed.

## Mutation evidence

`stages/4Q/mutate_4q.py` (revision 1's log `4q-mutants.log`, 15 mutants, each a wrong construction in `qpe.py` or `toy_walk.py` with its declaration changed where its emitted operators change, applied to a copy of the replica, run against the per-commit tests of Task 4.1's three files): every mutant fails at least one test. The controlled toy walk with only REF_anc controlled fails 10 (the three control states, the counts, the exact distribution, the lightning test). Two mutants keep the distribution symmetric and pass both distribution checks: the controlled toy walk applying W† (REF first) and the corrections with the wrong sign (phase +2π/2^d, which maps φ to 1 − φ); the eigenvector check of the exact distribution test catches both, the first also the three control states. The stale `earlier_bits` (every round all `False`), the rounds in increasing j, the walk power 2^{j+1}, the control measured without reset, a correction one bit too wide and the corrections on the gradient's low wires each fail the exact distribution and the lightning test, and the counts or the `earlier_bits` test where they apply. One extra walk step in `ControlledWalkPower` fails its T4 at both presets and the round's T4. The walk's groups flattened in layout order fail `test_walk_power_spec_flattens_the_groups_in_wire_argnames_order` (and would fail QPE of the walk, nightly). The energy without its shift, without the core energy and the bits read in reverse each fail their helper test; a missing shape check fails the validation test. The section "Task 4.2" has its own RED step (hinted corrections).

Revision 2 changed the round's wiring and some of these tests, so the check was rerun (`4q-mutants-r2.log`) for the six mutants whose killing tests changed (the flipped sign, the correction one bit too wide, the low gradient wires, the increasing rounds, the doubled walk power, the unchecked shape) and for five new ones; all eleven fail at least one test. The corrections sliced from the first walk wire (the `outer` group, which holds the walk's state; for the toy its own wires) fail the exact distribution, the lightning test and both `test_qpe_program_wires` cases. `ControlledWalkPower` without a scratch group (revision 1's qubit cost back) fails the wiring, registry, spare-wire, round T2 and round T4 tests. One spare wire too few fails 15 tests and 6 more error (the toy's rotations get too few work wires). The registry without its `scratch_argname` check fails the registry test, the driver without its aliasing check its `aliased` case. The other nine mutants' patches still apply to revision 2's code unchanged (`4q-mutants-check.log`), and for each at least one test that killed it in revision 1 has unchanged assertions.

## Risks and open questions

1. **The hinted counts' reading.** Change 9 reads "the hinted expected counts are recorded" as the expected `CZ` and `PauliZ` totals, in which the erase fix-ups' hinted conditionals count at their expectation (beside the deterministic gates; fix-up-only values are not separated). If the master plan meant the QPE's expected T (the correction term halved), it is (t³ − t)/12 from the closed form, never compiled: hinting the corrections would make the compiled T differ from the worst case that the same task must match.
2. **The corrections rely on the walk power's scratch group being |0⟩** (change 4). `ControlledWalkPower` declares `work`, which the controlled walk returns to |0⟩ on every erase outcome (Task 3.4's T3 compares whole states; here `ControlledWalkPower`'s T3 and every branch of QPE of the walk assert it, and the t = 2 checks would fail if a correction started from a dirty pool). A future walk power whose last group is not returned clean must declare `scratch_argname = None` and pay 2t − 1 spare wires; `register_walk_power` refuses a scratch group that is not the last group, but cannot check cleanliness. If a later walk change made `work` smaller than 2t − 1, `qpe_program` would add spare wires by itself (`spare_wires_needed`).
3. **QPE of the walk at two key words is slow.** `H2_STO3G_TEST` t = 2 takes 487 s (release only); spec 12.6 mitigation 2 (a byte-view `np.unique`, ledger row 1B.1) would cut it if the release tier needs room. `FAILS_C3` fits one key word up to t = 2 and carries the two-round check nightly.
4. **The lightning test is statistical** with a fixed seed (`qp.qjit(seed=2026)`): deterministic for one toolchain; a nightly bump that changes the random stream still passes each five-standard-deviation bound with probability above 1 − 10⁻⁶. Its mutation-killing power rests on the exact test, not on it.
5. **Cross-module private names.** `dfthc.qpe` imports `dfthc.ops.walk._validate`, and the tests build on `walk_sim`, which uses `walk.py`'s private helpers. Public helpers are a Task 5.1 cleanup.
6. **Phase 3 under review.** This plan builds on the assembly plan's code for Tasks 3.3 to 3.5 (section "Phase 3 interfaces this plan depends on"). Task 3.3 is merged (`main` at `e447e53`) with its review's changes, which the assembly plan's revision 4 carries; if Task 3.4's review changes those interfaces, the orchestrator reconciles this plan's code before dispatching Task 4.1.
7. **No open question blocks the plan.** Change 4 follows spec 11.1's reuse with the one interface that compiles (the review's option 1, chosen by the orchestrator), change 5 is forced by the toy walk's algebra, and change 7 adds a check that only tightens D-4.

## Phase 3 interfaces this plan depends on

From the assembly plan's code (Tasks 3.3 to 3.5), exactly as built:
- `dfthc.ops.walk.ControlledDFTHCWalk(params, tables, control, outer, inner, select, system, phase_grad, work)`, its `wire_argnames` (`control` first, then `outer`, `inner`, `select`, `system`, `phase_grad`, `work`), its static `required_wires(params, tables)` (`{"control": 1}` then the layout's group sizes), the constructor's checks through the private `_validate(name, wire_argnames, params, tables, wires)`, which returns the canonical `(params, tables)`, and its pinned rule `_controlled_walk`; `DFTHCWalk` (the reference of `ControlledWalkPower`'s T3).
- `dfthc.testing.walk_sim`: `case`, `WalkCase` (`build`, `params`, `tables`, `factors`, `layout`, `compilable`, `groups(controlled)`, `control`, `num_wires`, `block()`), `program`, `operations`, `schedule`, `num_measurements`, `input_state`, `basis`, `simulate`, `set_control`, `superpose`, `assert_same_state`, `nonzero_registers` and `AB_REGISTERS` (the clean-pool checks; `nonzero_registers` reads the walk's registers by wire, which the QPE layout keeps, its QPE registers coming after `work`), and the shapes `sto3g`, `631g`, `fails-c3`.
- `dfthc.testing.resources`: `captured_targets`, `inventory`, `LEAF_ADJOINTS`, `TEMPLATE_ADJOINTS`, `chosen_rules`, `verbose_program`.
- The costs `costs.walk`, `costs.measured_and_uncomputes`, `costs.toffoli_only` and the compiled counts of the controlled walk at the test presets (160 / 50 / 110 and 242 / 68 / 181).

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
| `src/dfthc/qpe.py` | 4.1 | `ControlledWalkPower`, `QPERound`, the registry `WALK_POWERS` with `register_walk_power`, `spare_wires_needed`, `WalkPowerSpec`, `walk_power_spec`, the driver `iterative_phase_estimation`, `qpe_program`, `phase_from_bits`, `energy_from_phase`, `physical_energy` |
| `src/dfthc/testing/toy_walk.py` | 4.1 | `ToyWalk`, `ControlledToyWalk`, `ToyControlledWalkPower` (registered as a walk power, no scratch group), the dense references, the toy's costs, `toy_walk_power_spec`, `toy_qpe_wires`, `toy_qpe` |
| `tests/test_qpe.py` | 4.1 | T2 to T6 of `ControlledWalkPower`, T2, T4, T5, T6 and validation of `QPERound`, the registry's checks and `spare_wires_needed`, the driver's checks, `walk_power_spec` and `qpe_program`'s wires (the corrections in the work pool), the energy helpers, FeMoco-54 closed forms |
| `tests/integration/test_qpe_toy.py` | 4.1 | T2 to T6 of the three toy classes; the exact t = 3 distribution (sparse simulator), the lightning smoke test, `earlier_bits` in both modes, the toy QPE program's counts and inventory |
| `tests/integration/test_qpe_walk.py` | 4.1 | QPE of the DFTHC walk on the sparse simulator against the encoded block's exact distribution (nightly and release), and the reference formula against the toy (per commit) |
| `tests/integration/test_qpe_counts.py` | 4.2 | the QPE program's compiled counts at both test presets for t = 1, 2, 3 (`slow`, each compile through `run_watched`), its inventory, FeMoco-54's closed form |

**Shell conventions.** Run commands from the worktree root of your lane. `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` locates the shared environment, as in the earlier plans.

**Expected test counts.** Each task states its targeted count exactly. The per-commit tier (`scripts/test.sh -m "not slow and not release" -q`) counts **1053 tests on `main` with Tasks 3.3 to 3.5 merged** (measured on the stage copy: `1053/1147 tests collected (94 deselected)`; the assembly plan's revision 4 gives the same), and the nightly tier (`-m "slow and not release"`) 61. The tasks run in order, so Task 4.2 starts from Task 4.1's count; the stage copy collects `1107/1209` per commit after Task 4.1, and `1108/1217` per commit and `75/1217` nightly after Task 4.2. The pass criterion is "no failures, no errors, no warnings summary"; the number is a cross-check, and the orchestrator shifts the base if other tasks merge first.

| Task | Base | Own per-commit | Expected per-commit tier | Own nightly and release tests |
| --- | ---: | ---: | ---: | --- |
| 4.1 | 1053 | 54 (`test_qpe.py` 32 of 35, `test_qpe_toy.py` 21, `test_qpe_walk.py` 1 of 6) | 1107 = 1053 + 54 | 7 `slow` (`test_qpe.py` 3, `test_qpe_walk.py` 4), 1 `release` (`test_qpe_walk.py`) |
| 4.2 | 1107 | 1 (`test_qpe_counts.py` 1 of 8) | 1108 = 1107 + 1 | 7 `slow` |

Dry run on the replica with both tasks: `1108 passed in 553.71s`, peak 1,871 MB (`stages/4Q/4q-per-commit-tier.log`; run with `TMPDIR` outside the workspace, so `tests/test_infrastructure.py` passes in the replica too; another lane ran its tests part of the time), within the tier's budget of 15 minutes and 4 GB for the two-worker process tree. The per-test gate of about 1 GB and a few seconds binds each test: see "Load".

**Design notes the tasks share.**
- *One walk-power interface.* A walk power is an `Operator2` whose arguments are `power`, its compilable arguments, a one-wire `control` and its wire groups; it registers its class with `dfthc.qpe.register_walk_power`. `QPERound` and the driver know it only through a `WalkPowerSpec` (`op_name`, `compilable`, `wire_sizes`, `wires`), so the toy walk and the DFTHC walk run through the same round, rule and driver.
- *Every wire is bound once.* `QPERound` binds `control`, `qpe_grad` and `walk_wires` as the operands of one operator, and Catalyst rejects operands that share a qubit; so the corrections borrow the walk's `work` pool by slicing `walk_wires` inside the rule, never as a register of their own (change 4). The driver checks that its wires are distinct before it queues anything.
- *The phase convention.* The bits estimate the eigenvalue e^{2πiφ} of W itself (spec 11.1): the toy test checks each branch's state against W's eigenvalue, not just the symmetric distribution.
- *Tolerances.* Every comparison states both tolerances (HANDOFF section 6): states, matrices and distributions `rtol=0, atol=1e-12`; the physical frame `atol=1e-10` (the factorization error of the spec 3.2.1 check); the lightning test five standard deviations.
- *FeMoco-54* values are asserted on the closed forms only (master plan section 12.9).

---

### Task 4.1: QPE operators, driver, energies and the toy walk

**Task id:** `4.1` · **Branch:** `phase4/4.1` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** G3

**Files:**
- Create: `src/dfthc/qpe.py`, `src/dfthc/testing/toy_walk.py`, `tests/test_qpe.py`, `tests/integration/test_qpe_toy.py`, `tests/integration/test_qpe_walk.py`

**Interfaces:**
- Consumes: Task 3.4's `dfthc.ops.walk` (`ControlledDFTHCWalk` with `required_wires`, `wire_argnames`, the private validator `_validate`, and the controlled walk's clean `work` pool on return; `DFTHCWalk`); Task 3.3's `dfthc.testing.walk_sim` (see "Phase 3 interfaces this plan depends on"); Task 3.4's `dfthc.testing.resources` (`captured_targets`, `inventory`) and `chosen_rules`, `verbose_program` (Task 0.2); `dfthc.ops.phase_gradient.PhaseGradientRotation` (Task 1C.1); `dfthc.ops._validate` (`canonical_ints`, `check_int`, `check_wires`); `dfthc.compile` (`register_rule`, `resource_key`, `compiled_resources`, `FIXED_DECOMPS`, `GATE_SET`); `dfthc.costs` (`Cost`, `controlled_walk_power`, `qpe_round`, `measured_and_uncomputes`, `toffoli_only`); `dfthc.params`, `dfthc.registers.WalkLayout`, `dfthc.data` (`WalkTables`, `encoded_block`, `load_factors`, `load_summary`, `load_integrals`, `sector_projector`, `h_sa_dense`, `e_sos_phys`, `e_sos_walk`, `frame_shift`); the sparse simulator (`SparseSimulator`, `SparseState`, `BranchPolicy`, `ForcedPolicy`) and `decompose_for_simulation`.
- Produces, in `dfthc.qpe`:
  - `WALK_POWERS: dict[str, type]` and `register_walk_power(cls) -> cls` (which requires the class attribute `scratch_argname`: `None` or the class's last wire group);
  - `spare_wires_needed(walk_op: str, walk_wire_sizes: Sequence[int], num_bits: int) -> int` (max(0, 2 num_bits − 1 − the scratch group's size));
  - `WalkPowerSpec(op_name: str, compilable: tuple, wire_sizes: tuple[int, ...], wires: Wires)` (a named tuple);
  - `ControlledWalkPower(power, params, tables, control, outer, inner, select, system, phase_grad, work)`: compilable `power` (at least 1), `params`, `tables`; `scratch_argname = "work"`; static `required_wires(power, params, tables)` (`ControlledDFTHCWalk`'s); the rule `_controlled_walk_power`, one `qp.for_loop(power)` over `ControlledDFTHCWalk`; registered as a walk power;
  - `walk_power_spec(params: WalkParams, tables: WalkTables, layout: WalkLayout) -> WalkPowerSpec`;
  - `QPERound(j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes, control, qpe_grad, walk_wires)` as in "QPERound's final form"; static `required_wires(j, num_bits, walk_op, walk_compilable, walk_wire_sizes)` (`control` 1, `qpe_grad` `num_bits`, `walk_wires` the sum of `walk_wire_sizes` plus `spare_wires_needed`); the rule `_qpe_round`;
  - `iterative_phase_estimation(walk_power_factory, control, num_bits, qpe_grad, spare_wires=()) -> list` (the bits m_{t−1}, …, m_0);
  - `qpe_program(params, tables, num_bits) -> tuple[Callable[[], list], int]` (the program and its wire count: `WalkLayout.from_params(params, qpe_bits=num_bits)`'s wires, then `spare_wires_needed` spare wires, none up to t = 4 at `H2_STO3G_TEST`);
  - `phase_from_bits(bits) -> float`, `energy_from_phase(phi, Lambda, shift=1.0) -> float`, `physical_energy(E_SA, e_sos_phys, core_energy=0.0) -> float`.
- Produces, in `dfthc.testing.toy_walk`: `ToyWalk(system, ancilla)`, `ControlledToyWalk(control, system, ancilla)`, `ToyControlledWalkPower(power, control, system, ancilla)` (`scratch_argname = None`; rules `_toy_walk`, `_controlled_toy_walk`, `_toy_controlled_walk_power`), `toy_walk_matrix()`, `controlled_toy_walk_matrix()`, `toy_walk_cost(controlled) -> Cost`, `toy_walk_power_cost(power) -> Cost`, `toy_walk_power_spec(system, ancilla) -> WalkPowerSpec`, `toy_qpe_wires(num_bits) -> dict[str, list[int]]` (keys `control`, `qpe_grad`, `spare_wires`, `system`, `ancilla`), `toy_qpe(num_bits) -> list`. No new class is ever adjointed or controlled.

**Acceptance tests** (62 tests: 54 per commit, 7 `slow`, 1 `release`):
- `tests/integration/test_qpe_toy.py` (21, per commit). The reference W has the spec's block (Z + X)/2 and eigenphases ±π/4, ±3π/4. T3 on `lightning.qubit`: the whole compiled matrix (one Bell-pair run) of `ToyWalk` is W, of `ControlledToyWalk` and `ToyControlledWalkPower(3)` |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ W^p, applied with the control in |0⟩, |1⟩ and |+⟩. T4: T = 0, 2 and 2p, A = 0, M = 0. T5 and T6 in one program (the toy QPE and a `ToyWalk`): the listing names `_toy_walk`, `_controlled_toy_walk`, `_toy_controlled_walk_power`, `_qpe_round` and `_phase_gradient_rotation`, nothing dfthc survives at `level="device"`. T2 and pins of the three rules; three constructor checks. The exact t = 3 distribution (strict sparse simulator, `BranchPolicy` on the control, no `otherwise`): φ ∈ {1/8, 7/8} with cos²(π/8)/2 each, {3/8, 5/8} with sin²(π/8)/2 each, nothing else, to 1e-12, every branch leaving the control and the spare wires in |0⟩, `qpe_grad` in the gradient state and the toy wires in an eigenvector of W with eigenvalue e^{2πiφ}. The lightning smoke test with 2,000 one-shot samples within five standard deviations, no other phase. `earlier_bits` of shape (t,) in every round in tape mode (False up to j, the earlier rounds' own `MeasurementValue`s after) and under capture (boolean). The toy QPE program's T, M, A for t = 1, 2, 3 (2 / 1 / 0, 7 / 2 / 1, 18 / 3 / 4). Its inventory: the rounds, walk powers, controlled toy walk and corrections converted, no `C(...)` target, no modified dfthc or template target.
- `tests/test_qpe.py` (35: 32 per commit, 3 `slow`). `ControlledWalkPower`: T4 at p = 2 at `H2_STO3G_TEST` and `H2_631G_TEST` (per commit) and `MID_NAPHTHALENE` (`slow`), T, M against `costs.controlled_walk_power`, A = 2 A_C(W), the plain `Toffoli`s 2 `toffoli_only`; T3 on the tape path at p = 2 with the control in |0⟩, |1⟩ and |+⟩ against two `DFTHCWalk` steps, every output with only the index registers of A ∪ B nonzero (`slow`); T2 at three shapes and powers with the pin; validation and `required_wires`; FeMoco-54 closed forms. `QPERound`: T4 of round 0 of 2 bits at `H2_STO3G_TEST` (161 / 50 / 111, a measured bit as m_1); T2 on the toy (j = 0, 1, 2 of 3 bits) and the walk; T6, the listing of the 2-bit QPE program at `H2_STO3G_TEST` (`slow`); T5 at `level="device"` for that program; seven constructor checks. The registry: `scratch_argname` is `"work"` and `None`, a second registration, a scratch group that is not the last group and a class without the attribute are refused. `spare_wires_needed`: the toy 1, 3, 5; `H2_STO3G_TEST` 0 up to t = 4, then 1 and 3; FeMoco-54 0 up to t = 22, 1 at 23, 5 at 25. The driver's four checks (nothing queued before they raise; one is a wire bound twice). `walk_power_spec` flattens in `wire_argnames` order (not the layout's); `qpe_program`'s wires at t = 3 (67 wires, no spare) and t = 5 (70, one spare wire): every round's registers, and round 0's rotations' scratch and work wires sliced from the work pool on into the spare wire. `phase_from_bits`; `energy_from_phase` returns every eigenvalue of H̃_SA from both walk phases (shift 1) and the toy's ±1/√2 (shift 0); `physical_energy` is the physical frame.
- `tests/integration/test_qpe_walk.py` (6: 1 per commit, 4 `slow`, 1 `release`). The reference formula reproduces the toy's distribution (per commit). QPE of the walk: `FAILS_C3` t = 1 (alternating and seeded schedules) and t = 2, `H2_STO3G_TEST` t = 1 (nightly), `H2_STO3G_TEST` t = 2 (release): the control bits' probabilities equal the encoded block's exact distribution to 1e-12, and every branch ends with only the index registers of A ∪ B nonzero.

**Definition of done:** `54 passed` per commit and `7 passed` nightly, targeted; the release check collected and not run; the per-commit tier passes; `ruff check .` clean; the spec 9.0 docstring sections in order for the five classes; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_qpe_toy.py`:

```python
"""The toy walk of spec 11.4 and iterative phase estimation on it (spec 11.1): ToyWalk,
ControlledToyWalk and ToyControlledWalkPower (T2 to T6, the action against the dense reference
with the control in |0>, |1> and |+>), the exact t = 3 distribution on the sparse simulator, the
lightning.qubit smoke test, earlier_bits of shape (t,) in every round in both modes, the toy QPE
program's counts and its inventory of captured targets."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import qpe
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.qpe import phase_from_bits
from dfthc.testing.resources import captured_targets, chosen_rules, inventory, verbose_program
from dfthc.testing.sparse_sim import BranchPolicy, SparseSimulator
from dfthc.testing.tape import decompose_for_simulation
from dfthc.testing.toy_walk import (
    ControlledToyWalk,
    ToyControlledWalkPower,
    ToyWalk,
    _controlled_toy_walk,
    _toy_controlled_walk_power,
    _toy_walk,
    controlled_toy_walk_matrix,
    toy_qpe,
    toy_qpe_wires,
    toy_walk_cost,
    toy_walk_matrix,
    toy_walk_power_cost,
)

T = 3
NUM_WIRES = 3 * T + 2
COS2, SIN2 = np.cos(np.pi / 8) ** 2, np.sin(np.pi / 8) ** 2
EXACT = {1 / 8: COS2 / 2, 3 / 8: SIN2 / 2, 5 / 8: SIN2 / 2, 7 / 8: COS2 / 2}
"""Spec 11.4: phi in {1/8, 7/8} with total probability cos^2(pi/8), {3/8, 5/8} with sin^2(pi/8);
each eigenvalue of H splits equally between the walk phases +-theta."""
PLUS = np.array([1, 1]) / np.sqrt(2)


# --------------------------------------------------------------------------- the toy walks


def test_reference_walk_has_the_spec_eigenphases():
    """Spec 11.4: W is unitary, its block on the ancilla's |0> is H = (Z + X)/2, and its
    eigenphases are -3 pi/4, -pi/4, pi/4, 3 pi/4."""
    walk = toy_walk_matrix()
    assert np.allclose(walk.conj().T @ walk, np.eye(4), rtol=0, atol=1e-12)
    block = walk[np.ix_([0, 2], [0, 2])]
    assert np.allclose(block, np.array([[1, 1], [1, -1]]) / 2, rtol=0, atol=1e-12)
    phases = np.sort(np.angle(np.linalg.eigvals(walk)))
    assert np.allclose(phases, np.pi * np.array([-3, -1, 1, 3]) / 4, rtol=0, atol=1e-12)


def choi_matrix(build, n: int) -> np.ndarray:
    """The matrix of the n-qubit program ``build(data_wires)`` from one compiled lightning run:
    Bell pairs between reference wires 0..n-1 and data wires n..2n-1, then the program on the
    data wires; the output amplitude [reference x, data y] is U[y, x] / sqrt(2**n)."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=2 * n))
    def circuit():
        for wire in range(n):
            qp.H(wire)
            qp.CNOT([wire, n + wire])
        build(list(range(n, 2 * n)))
        return qp.state()

    return np.sqrt(2**n) * np.asarray(circuit()).reshape(2**n, 2**n).T


@pytest.mark.compiled
def test_toy_walk_acts_as_the_reference():
    """T3: the compiled ToyWalk's whole matrix, global phase included, is spec 11.4's W."""
    matrix = choi_matrix(lambda w: ToyWalk(system=w[:1], ancilla=w[1:]), 2)
    assert np.allclose(matrix, toy_walk_matrix(), rtol=0, atol=1e-12)


@pytest.mark.compiled
@pytest.mark.parametrize("power", [1, 3])
def test_controlled_toy_walks_in_three_control_states(power):
    """T3 of the controlled classes: the compiled matrix of ControlledToyWalk (power 1) and of
    ToyControlledWalkPower(3) is |0><0| (x) I + |1><1| (x) W^p exactly, so control |0> keeps the
    input, control |1> applies W^p and control |+> gives their superposition with the right
    relative phase (spec 3.3: a wrong global phase of a controlled reflection shows only there)."""
    def build(w):
        if power == 1:
            ControlledToyWalk(control=w[:1], system=w[1:2], ancilla=w[2:])
        else:
            ToyControlledWalkPower(power, control=w[:1], system=w[1:2], ancilla=w[2:])

    matrix = choi_matrix(build, 3)
    walked = np.linalg.matrix_power(toy_walk_matrix(), power)
    if power == 1:
        assert np.allclose(controlled_toy_walk_matrix(), np.kron(np.diag([1, 0]), np.eye(4))
                           + np.kron(np.diag([0, 1]), walked), rtol=0, atol=1e-12)
    rng = np.random.default_rng(power)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    expected = {"0": np.kron([1, 0], psi), "1": np.kron([0, 1], walked @ psi),
                "+": (np.kron([1, 0], psi) + np.kron([0, 1], walked @ psi)) / np.sqrt(2)}
    for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", PLUS)):
        out = matrix @ np.kron(control, psi)
        assert np.allclose(out, expected[label], rtol=0, atol=1e-12), label


def toy_program(name, power=1):
    if name == "ToyWalk":
        return lambda: ToyWalk(system=[0], ancilla=[1])
    if name == "ControlledToyWalk":
        return lambda: ControlledToyWalk(control=[0], system=[1], ancilla=[2])
    return lambda: ToyControlledWalkPower(power, control=[0], system=[1], ancilla=[2])


@pytest.mark.compiled
@pytest.mark.parametrize(("name", "power"), [("ToyWalk", None), ("ControlledToyWalk", None),
                                             ("ToyControlledWalkPower", 1),
                                             ("ToyControlledWalkPower", 4)])
def test_counts(name, power):
    """T4: ToyWalk T = M = A = 0; ControlledToyWalk two plain Toffolis (T = 2, A = 0, M = 0);
    the power p of it T = 2 p, all against toy_walk_cost and toy_walk_power_cost."""
    resources = compiled_resources(toy_program(name, power), 2 if name == "ToyWalk" else 3)
    expected = (toy_walk_cost(False) if name == "ToyWalk" else toy_walk_cost(True)
                if name == "ControlledToyWalk" else toy_walk_power_cost(power))
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.counts.get("Toffoli", 0) == expected.toffoli
    assert resources.measured_and_uncomputes == 0


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing of the toy QPE program and a ToyWalk names the pinned rules of
    the three toy classes, of QPERound and of the corrections' PhaseGradientRotation; T5: at
    level="device" no dfthc operator survives (compiled_resources asserts it)."""
    wires = toy_qpe_wires(T)

    def program():
        toy_qpe(T)
        ToyWalk(system=wires["system"], ancilla=wires["ancilla"])

    qp.analyze(verbose_program(program, NUM_WIRES), level="user")()
    captured = capfd.readouterr()
    assert {"_toy_walk", "_controlled_toy_walk", "_toy_controlled_walk_power", "_qpe_round",
            "_phase_gradient_rotation"} <= set(chosen_rules(captured.out + captured.err))
    compiled_resources(program, NUM_WIRES, level="device")


def test_rules_are_valid():
    """T2 (tape mode) of the three rules, the power at two values; each pinned and the only rule
    under its class name (spec 7.4 rules 7 and 8)."""
    for name, rule, powers in (("ToyWalk", _toy_walk, [None]),
                               ("ControlledToyWalk", _controlled_toy_walk, [None]),
                               ("ToyControlledWalkPower", _toy_controlled_walk_power, [1, 3])):
        assert FIXED_DECOMPS[name] is rule
        for power in powers:
            with qp.queuing.AnnotatedQueue():
                op = toy_program(name, power)()
            assert list(qp.list_decomps(type(op))) == [rule]
            qp.testing.assert_valid_decomp_rule(op, rule, check_matrix=False)


@pytest.mark.parametrize(("build", "match"), [
    (lambda: ToyWalk(system=[0, 1], ancilla=[2]), "'system' has 2 wires; expected 1"),
    (lambda: ControlledToyWalk(control=[], system=[1], ancilla=[2]),
     "'control' has 0 wires; expected 1"),
    (lambda: ToyControlledWalkPower(0, control=[0], system=[1], ancilla=[2]),
     r"power must lie in \[1, None\], got 0"),
], ids=["system", "control", "power"])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()


# --------------------------------------------------------------------------- phase estimation


def control_measurements(ops):
    wires = toy_qpe_wires(T)
    measures = [op for op in ops if op.name == "MidMeasure"]
    assert [int(op.wires[0]) for op in measures] == wires["control"] * T
    return measures


def test_exact_distribution_at_t_3():
    """Spec 11.4 on the tape path (strict sparse simulator, BranchPolicy on the control; the toy
    has no erase measurements, so the policy has no `otherwise` and any other measurement would
    raise): phi in {1/8, 7/8} with cos^2(pi/8) and {3/8, 5/8} with sin^2(pi/8), half each, and
    nothing else. Every branch leaves the control and the spare wires in |0>, qpe_grad in
    the gradient state and the toy wires in an eigenvector of W with eigenvalue e^{2 pi i phi}:
    the bits read the eigenphase of W itself (spec 11.1's convention), not of W^dagger."""
    ops = decompose_for_simulation(toy_qpe, T).operations
    measures = control_measurements(ops)
    simulator = SparseSimulator(NUM_WIRES, policy=BranchPolicy(toy_qpe_wires(T)["control"]),
                                strict=True)
    gradient = SparseSimulator(T).run([qp.PhaseGradientStatePrep(range(T))]).to_dense()
    walk, found = toy_walk_matrix(), {}
    for branch in simulator.run_branches(ops):
        phi = phase_from_bits([branch.record[op] for op in measures])
        found[phi] = found.get(phi, 0.0) + branch.probability
        state = branch.state.to_dense().reshape(2, 2**T, 2 ** (2 * T - 1), 4)
        toy = gradient.conj() @ state[0, :, 0, :]
        assert np.allclose(state[0, :, 0, :], np.outer(gradient, toy), rtol=0, atol=1e-12)
        assert np.isclose(np.linalg.norm(toy), 1, rtol=0, atol=1e-12)
        assert np.allclose(walk @ toy, np.exp(2j * np.pi * phi) * toy, rtol=0, atol=1e-12)
    assert sorted(found) == sorted(EXACT)
    for phi, probability in EXACT.items():
        assert np.isclose(found[phi], probability, rtol=0, atol=1e-12), phi
    assert np.isclose(found[1 / 8] + found[7 / 8], COS2, rtol=0, atol=1e-12)
    assert np.isclose(found[3 / 8] + found[5 / 8], SIN2, rtol=0, atol=1e-12)


SHOTS = 2000


@pytest.mark.compiled
def test_lightning_smoke_with_2000_shots():
    """Spec 11.4: the compiled program on lightning.qubit, 2,000 one-shot samples (no
    postselection, no measurement inside qp.cond; L-15): each of the four phases within five
    standard deviations of its exact probability, the two pairs too, and no other phase."""

    @qp.qjit(capture=True, seed=2026)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.set_shots(SHOTS)
    @qp.qnode(qp.device("lightning.qubit", wires=NUM_WIRES), mcm_method="one-shot")
    def circuit():
        return [qp.sample(op=bit) for bit in toy_qpe(T)]

    samples = np.asarray(circuit()).reshape(T, SHOTS)
    phases = [phase_from_bits(samples[:, shot]) for shot in range(SHOTS)]
    values, counts = np.unique(phases, return_counts=True)
    frequency = dict(zip(values.tolist(), (counts / SHOTS).tolist(), strict=True))
    assert set(frequency) <= set(EXACT)
    pairs = {(1 / 8, 7 / 8): COS2, (3 / 8, 5 / 8): SIN2}
    checks = {(phi,): p for phi, p in EXACT.items()} | pairs
    for group, p in checks.items():
        observed = sum(frequency.get(phi, 0.0) for phi in group)
        assert abs(observed - p) <= 5 * np.sqrt(p * (1 - p) / SHOTS), (group, observed, p)


@pytest.mark.parametrize("mode", ["tape", pytest.param("capture", marks=pytest.mark.compiled)])
def test_earlier_bits_have_shape_t_in_every_round(mode, monkeypatch):
    """Spec 11.1, master plan Task 4.1: every QPERound of the driver gets earlier_bits of shape
    (t,), round t - 1 included (no bit set yet). In tape mode entry l of round j is False for
    l <= j and, for l > j, the very MeasurementValue that round l's measurement returned; under
    capture it is a boolean array of the same shape, the measured bits set with .at[j].set."""
    rounds, original = [], qpe.QPERound

    def recording(j, num_bits, earlier_bits, *rest):
        rounds.append((j, earlier_bits))
        return original(j, num_bits, earlier_bits, *rest)

    monkeypatch.setattr(qpe, "QPERound", recording)
    if mode == "tape":
        with qp.capture.pause(), qp.queuing.AnnotatedQueue():
            measured = toy_qpe(T)
    else:
        compiled_resources(lambda: toy_qpe(T), NUM_WIRES)
    assert [j for j, _ in rounds] == [2, 1, 0]
    for j, bits in rounds:
        assert tuple(bits.shape) == (T,)
        if mode == "tape":
            assert all(bits[later] is False for later in range(j + 1))
            assert all(bits[later] is measured[T - 1 - later] for later in range(j + 1, T))
        else:
            assert bits.dtype == bool


@pytest.mark.compiled
@pytest.mark.parametrize(("t", "toffoli", "measurements", "uncomputes"),
                         [(1, 2, 1, 0), (2, 3 * 2 + 1, 2, 1), (3, 7 * 2 + 4, 3, 4)])
def test_toy_qpe_counts(t, toffoli, measurements, uncomputes):
    """T4 of the QPE program on the toy walk, worst case (spec 9.24 with T_C(W) = 2, M_W = 0):
    T = (2^t - 1) 2 + (t^3 - t)/6, M = t, and A = (t^3 - t)/6, the corrections' measured AND
    uncomputes, every unhinted correction counted at its maximum branch."""
    assert toffoli == toy_walk_power_cost(2**t - 1).toffoli + (t**3 - t) // 6
    resources = compiled_resources(lambda: toy_qpe(t), 3 * t + 2)
    assert (resources.toffoli, resources.measurements,
            resources.measured_and_uncomputes) == (toffoli, measurements, uncomputes)


@pytest.mark.compiled
def test_inventory_of_the_toy_qpe_program():
    """The inventory (spec 12.1 T6) of the toy QPE program at t = 3: the frontend converts the
    rounds, the walk powers and the controlled toy walk, no C(...) target at all, and no modified
    dfthc or template target (the corrections are forward rotations)."""
    names = captured_targets(lambda: toy_qpe(T), NUM_WIRES)
    assert {"QPERound", "ToyControlledWalkPower", "ControlledToyWalk",
            "PhaseGradientRotation"} <= set(names)
    assert not [name for name in names if name.startswith("C(")]
    assert inventory(names) == set()
```

Create `tests/test_qpe.py`:

```python
"""dfthc.qpe (spec 9.23 and 11): ControlledWalkPower (T2 to T6), QPERound (T2, T4, T5, T6),
the walk-power registry and the corrections' wires, the driver's checks, WalkPowerSpec and
qpe_program, and the energy helpers of spec 11.1."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, compiled_resources
from dfthc.data.encoded import encoded_block
from dfthc.data.factors import load_factors, load_integrals, load_summary
from dfthc.data.fermion import sector_projector
from dfthc.data.sos import e_sos_phys, e_sos_walk, frame_shift, h_sa_dense
from dfthc.ops.walk import DFTHCWalk
from dfthc.params import FEMOCO54, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.qpe import (
    WALK_POWERS,
    ControlledWalkPower,
    QPERound,
    WalkPowerSpec,
    _controlled_walk_power,
    _qpe_round,
    energy_from_phase,
    iterative_phase_estimation,
    phase_from_bits,
    physical_energy,
    qpe_program,
    register_walk_power,
    spare_wires_needed,
    walk_power_spec,
)
from dfthc.registers import WalkLayout
from dfthc.testing import walk_sim
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.toy_walk import toy_walk_matrix, toy_walk_power_spec

PLUS = 2**-0.5


def power_call(c, power):
    return lambda: ControlledWalkPower(power, *c.compilable, **c.groups(controlled=True))


def compile_power(c, power, level="user"):
    def program():
        qp.PhaseGradientStatePrep(c.layout.phase_grad)
        power_call(c, power)()

    return compiled_resources(program, c.num_wires, level=level)


def assert_power_counts(resources, p, power):
    """T, M against costs.controlled_walk_power; A and the plain Toffolis power times the
    controlled walk's (spec 9.23 and 9.24)."""
    expected = costs.controlled_walk_power(p, power)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measured_and_uncomputes == power * costs.measured_and_uncomputes(p, True)
    assert resources.counts["Toffoli"] == power * costs.toffoli_only(p)


# --------------------------------------------------------------------------- ControlledWalkPower


@pytest.mark.compiled
@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_controlled_walk_power_counts(name):
    """T4 at the test presets, power 2 (one loop body): T = 2 T_C(W) = 320 and 484, M = 2 M_W =
    100 and 136, A = 2 A_C(W) = 220 and 362."""
    c = walk_sim.case(name)
    assert_power_counts(compile_power(c, 2), c.params, 2)


@pytest.mark.compiled
@pytest.mark.slow
def test_controlled_walk_power_counts_at_mid_naphthalene():
    """T4 at MID_NAPHTHALENE (nightly), power 2: T = 3,830, M = 1,004, A = 3,360."""
    c = walk_sim.WalkCase.build(MID_NAPHTHALENE, load_factors("naphthalene"))
    assert_power_counts(compile_power(c, 2), c.params, 2)


@pytest.mark.slow
def test_controlled_walk_power_in_three_control_states():
    """T3 on the tape path (strict sparse simulator) at H2_STO3G_TEST, power 2 (nightly: about
    15 s, eight walk steps from a walked state): control |0> returns the whole input, control |1>
    gives two DFTHCWalk steps' whole output, control |+> their superposition; the second step
    starts from the walk's own state (spec 12.4, E-3). Every output has only the index registers
    of A u B nonzero: the walk power returns its work pool, which QPERound's corrections borrow,
    to |0>."""
    c = walk_sim.case("sto3g")
    psi = walk_sim.basis(c, 10)
    plain = walk_sim.program(c, DFTHCWalk, repeat=2)
    powered = walk_sim.operations(c, power_call(c, 2))
    outcomes = walk_sim.schedule("seeded", walk_sim.num_measurements(plain), seed=4)
    identity = walk_sim.simulate(c, walk_sim.operations(c), walk_sim.input_state(c, psi), [])
    applied = walk_sim.set_control(c, walk_sim.simulate(c, plain, walk_sim.input_state(c, psi),
                                                        outcomes))
    expected = {"0": identity, "1": applied,
                "+": walk_sim.superpose((PLUS, identity), (PLUS, applied))}
    for label, control in (("0", [1, 0]), ("1", [0, 1]), ("+", [PLUS, PLUS])):
        out = walk_sim.simulate(c, powered, walk_sim.input_state(c, psi, control), outcomes)
        walk_sim.assert_same_state(out, expected[label])
        assert walk_sim.nonzero_registers(c, out) <= set(walk_sim.AB_REGISTERS), label


@pytest.mark.parametrize(("name", "power"), [("sto3g", 1), ("sto3g", 3), ("fails-c3", 2)])
def test_controlled_walk_power_rule_is_valid(name, power):
    """T2 (tape mode; the loop unrolls) at two shapes and three powers; pinned and the only
    rule under its class name."""
    c = walk_sim.case(name)
    assert FIXED_DECOMPS["ControlledWalkPower"] is _controlled_walk_power
    assert list(qp.list_decomps(ControlledWalkPower)) == [_controlled_walk_power]
    op = ControlledWalkPower(power, *c.compilable, **c.groups(controlled=True))
    qp.testing.assert_valid_decomp_rule(op, _controlled_walk_power, check_matrix=False)


def test_controlled_walk_power_validates():
    """Spec 9.0: power at least 1; the controlled walk's checks (``ValueError``)."""
    c = walk_sim.case("sto3g")
    with pytest.raises(ValueError, match=r"power must lie in \[1, None\], got 0"):
        ControlledWalkPower(0, *c.compilable, **c.groups(controlled=True))
    with pytest.raises(ValueError, match="'outer' has 1 wires; expected 16"):
        ControlledWalkPower(1, *c.compilable, **(c.groups(controlled=True) | {"outer": [0]}))
    assert ControlledWalkPower.required_wires(5, *c.compilable) == {
        "control": 1, "outer": 16, "inner": 23, "select": 8, "system": 4, "phase_grad": 4,
        "work": 8}


def test_closed_forms_at_femoco():
    """Spec 9.23 and 9.24 at FeMoco-54, without compiling: C(W)^2 is 19,590 Toffolis and 5,048
    measurements; round 0 of a 3-bit estimation adds 1 + 2 correction Toffolis to one step."""
    assert costs.controlled_walk_power(FEMOCO54, 2) == costs.Cost(2 * 9795, 2 * 2524)
    assert costs.qpe_round(FEMOCO54, 0, 3) == costs.Cost(9795 + 1 + 2, 2524)
    assert costs.qpe_round(FEMOCO54, 2, 3) == costs.Cost(4 * 9795, 4 * 2524)


# --------------------------------------------------------------------------- QPERound


def walk_round(layout, p, t, j, earlier_bits):
    spec = walk_power_spec(p, walk_sim.case("sto3g").tables, layout)
    return QPERound(j, t, earlier_bits, spec.op_name, spec.compilable, spec.wire_sizes,
                    layout.qpe_control, layout.qpe_grad, spec.wires)


@pytest.mark.compiled
def test_qpe_round_counts():
    """T4 of round 0 of a 2-bit estimation at H2_STO3G_TEST, worst case (spec 9.24): T =
    T_C(W) + 1 = 161 (the d = 2 correction, unhinted, at its maximum branch), M = M_W = 50 plus
    the measurement that gives m_1, A = A_C(W) + 1. The correction borrows the work pool."""
    p, t = H2_STO3G_TEST, 2
    layout = WalkLayout.from_params(p, qpe_bits=t)
    extra = layout.num_wires

    def program():
        qp.PhaseGradientStatePrep(layout.phase_grad)
        qp.PhaseGradientStatePrep(layout.qpe_grad)
        qp.H(extra)
        m_1 = qp.measure(extra)
        walk_round(layout, p, t, 0, [False, m_1])

    resources = compiled_resources(program, extra + 1)
    expected = costs.qpe_round(p, 0, t)
    assert (resources.toffoli, resources.measurements - 1) == (expected.toffoli,
                                                               expected.measurements)
    assert expected == costs.Cost(161, 50)
    assert resources.measured_and_uncomputes == costs.measured_and_uncomputes(p, True) + 1


def measured(count):
    with qp.queuing.AnnotatedQueue():
        return [qp.measure(wire) for wire in range(100, 100 + count)]


@pytest.mark.parametrize("j", [0, 1, 2])
def test_qpe_round_rule_is_valid_on_the_toy(j):
    """T2 (tape mode) of the three rounds of a 3-bit estimation of the toy walk, the earlier bits
    MeasurementValues (the corrections are conditionals on them); pinned and the only rule."""
    assert FIXED_DECOMPS["QPERound"] is _qpe_round
    assert list(qp.list_decomps(QPERound)) == [_qpe_round]
    bits = [False] * (j + 1) + measured(2 - j)
    spec = toy_walk_power_spec(system=[9], ancilla=[10])
    op = QPERound(j, 3, bits, spec.op_name, spec.compilable, spec.wire_sizes, control=[0],
                  qpe_grad=[1, 2, 3], walk_wires=[9, 10, 4, 5, 6, 7, 8])
    qp.testing.assert_valid_decomp_rule(op, _qpe_round, check_matrix=False)


def test_qpe_round_rule_is_valid_on_the_walk():
    """T2 (tape mode) of round 0 of a 2-bit estimation of the walk at H2_STO3G_TEST."""
    layout = WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=2)
    op = walk_round(layout, H2_STO3G_TEST, 2, 0, [False, *measured(1)])
    qp.testing.assert_valid_decomp_rule(op, _qpe_round, check_matrix=False)


@pytest.mark.compiled
@pytest.mark.slow
def test_pinned_rules_are_chosen(capfd):
    """T6 (nightly: a verbose compile of about 12 s): the listing of the 2-bit QPE program at
    H2_STO3G_TEST names the rounds', the walk powers', the controlled walk's and the
    correction's pinned rules. The per-commit listing is the toy QPE program's
    (tests/integration/test_qpe_toy.py), and the exact counts below are the per-commit count
    signature of these rules."""
    program, num_wires = qpe_program(H2_STO3G_TEST, walk_sim.case("sto3g").tables, 2)
    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    assert {"_qpe_round", "_controlled_walk_power", "_controlled_walk",
            "_phase_gradient_rotation"} <= set(chosen_rules(captured.out + captured.err))


@pytest.mark.compiled
def test_nothing_survives_at_device_level():
    """T5 at level="device" (L-20): no dfthc operator survives in the 2-bit QPE program."""
    program, num_wires = qpe_program(H2_STO3G_TEST, walk_sim.case("sto3g").tables, 2)
    compiled_resources(program, num_wires, level="device")


@pytest.mark.parametrize(("change", "match"), [
    ({"j": 3}, r"j must lie in \[0, 2\], got 3"),
    ({"walk_op": "DFTHCWalk"}, "walk_op must be one of"),
    ({"walk_compilable": []}, "walk_compilable must be a tuple"),
    ({"walk_wire_sizes": (2,), "walk_wires": [9, 10, 4, 5, 6, 7, 8]},
     r"walk_wire_sizes must be ToyControlledWalkPower's required_wires after control, \(1, 1\)"),
    ({"earlier_bits": [False, False]}, r"earlier_bits must have shape \(3,\); got \(2,\)"),
    ({"qpe_grad": [1, 2]}, "'qpe_grad' has 2 wires; expected 3"),
    ({"walk_wires": [9, 10, 4, 5, 6, 7]}, "'walk_wires' has 6 wires; expected 7"),
], ids=["j", "walk_op", "walk_compilable", "walk_wire_sizes", "earlier_bits", "qpe_grad",
        "spare"])
def test_qpe_round_validates(change, match):
    """Spec 9.0: QPERound checks its compilable arguments, the shape (t,) of earlier_bits and
    every register size (``ValueError``); the toy walk lends no scratch group, so walk_wires
    holds its two wires and the corrections' five spare wires."""
    spec = toy_walk_power_spec(system=[9], ancilla=[10])
    arguments = {"j": 0, "num_bits": 3, "earlier_bits": [False] * 3, "walk_op": spec.op_name,
                 "walk_compilable": spec.compilable, "walk_wire_sizes": spec.wire_sizes,
                 "control": [0], "qpe_grad": [1, 2, 3], "walk_wires": [9, 10, 4, 5, 6, 7, 8]
                 } | change
    with pytest.raises(ValueError, match=match):
        QPERound(**arguments)


def test_register_walk_power_checks_the_class():
    """A walk power registers once, and its scratch_argname is None or its last wire group
    (the group QPERound's corrections borrow); a class without the attribute is refused."""
    assert WALK_POWERS["ControlledWalkPower"].scratch_argname == "work"
    assert WALK_POWERS["ToyControlledWalkPower"].scratch_argname is None
    with pytest.raises(ValueError, match="ControlledWalkPower is already a registered"):
        register_walk_power(ControlledWalkPower)
    wires = {"wire_argnames": ("control", "system", "work")}
    with pytest.raises(ValueError, match=r"scratch_argname must be None or its last wire group "
                                         r"'work'; got 'system'"):
        register_walk_power(type("Middle", (), wires | {"scratch_argname": "system"}))
    with pytest.raises(ValueError, match="got ''"):
        register_walk_power(type("Undeclared", (), wires))
    assert "Middle" not in WALK_POWERS and "Undeclared" not in WALK_POWERS


def test_spare_wires_needed():
    """The corrections' 2t - 1 wires less the walk power's scratch group: the toy lends none
    (1, 3, 5 spare wires for t = 1, 2, 3); ControlledWalkPower lends its work pool, 8 wires at
    H2_STO3G_TEST (spare wires only from t = 5) and 44 at FeMoco-54 (only from t = 23).
    QPERound's walk_wires are the walk's groups plus these."""
    toy = toy_walk_power_spec(system=[0], ancilla=[1])
    assert [spare_wires_needed(toy.op_name, toy.wire_sizes, t) for t in (1, 2, 3)] == [1, 3, 5]
    sto3g = walk_power_spec(H2_STO3G_TEST, walk_sim.case("sto3g").tables,
                            WalkLayout.from_params(H2_STO3G_TEST))
    assert [spare_wires_needed(sto3g.op_name, sto3g.wire_sizes, t) for t in range(1, 7)] == [
        0, 0, 0, 0, 1, 3]
    assert [QPERound.required_wires(0, t, sto3g.op_name, sto3g.compilable, sto3g.wire_sizes)[
        "walk_wires"] for t in (4, 5)] == [63, 64]
    femoco = WalkLayout.from_params(FEMOCO54)
    sizes = tuple(len(getattr(femoco, name)) for name in ControlledWalkPower.wire_argnames[1:])
    assert sizes[-1] == WalkLayout.work_size(FEMOCO54) == 44 and sum(sizes) == 1564
    assert [spare_wires_needed("ControlledWalkPower", sizes, t) for t in (3, 20, 22, 23, 25)] == [
        0, 0, 0, 1, 5]


# --------------------------------------------------------------------------- driver and wiring


@pytest.mark.parametrize(("arguments", "match"), [
    ((0, [1, 2, 3], [4, 5, 6, 7, 8]), r"num_bits must lie in \[1, None\], got 0"),
    ((3, [1, 2], [4, 5, 6, 7, 8]), r"num_bits = 3 exceeds len\(qpe_grad\) = 2"),
    ((3, [1, 2, 3], [4, 5, 6, 7]),
     "spare_wires needs 5 wires for num_bits = 3 with ToyControlledWalkPower; got 4"),
    ((3, [1, 2, 3], [4, 5, 6, 7, 9]), "must be distinct"),
], ids=["zero-bits", "qpe_grad", "spare", "aliased"])
def test_driver_validates(arguments, match):
    """Spec 11.1: the driver raises ValueError if num_bits > len(qpe_grad), for no bits, for too
    few spare wires (the toy's corrections need 2t - 1) and for a wire bound twice (here the
    toy's system wire as a spare wire), before it queues anything."""
    num_bits, qpe_grad, spare = arguments
    with qp.queuing.AnnotatedQueue() as queue, pytest.raises(ValueError, match=match):
        iterative_phase_estimation(lambda: toy_walk_power_spec([9], [10]), [0], num_bits,
                                   qpe_grad, spare)
    assert not queue.queue


def test_walk_power_spec_flattens_the_groups_in_wire_argnames_order():
    """The flat walk_wires of QPERound hold ControlledWalkPower's groups in its wire_argnames
    order (outer, inner, select, system, phase_grad, work), not the layout's order, which starts
    with system; split by wire_sizes they are the layout's groups again."""
    c = walk_sim.case("sto3g")
    layout = WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=2)
    spec = walk_power_spec(c.params, c.tables, layout)
    assert isinstance(spec, WalkPowerSpec) and WALK_POWERS[spec.op_name] is ControlledWalkPower
    assert spec.compilable == c.compilable
    assert spec.wire_sizes == (16, 23, 8, 4, 4, 8)
    wires, start = list(spec.wires), 0
    for name, size in zip(ControlledWalkPower.wire_argnames[1:], spec.wire_sizes, strict=True):
        assert wires[start:start + size] == list(getattr(layout, name)), name
        start += size
    assert wires[:4] != list(layout.system) and start == len(wires) == 63


@pytest.mark.parametrize(("t", "spare"), [(3, []), (5, [69])])
def test_qpe_program_wires(t, spare):
    """qpe_program at H2_STO3G_TEST: the layout with qpe_bits = t (control 63, qpe_grad from
    64), then the spare wires: none at t = 3, whose 5 correction wires fit the 8-wire work pool
    (55 to 62), one (69) at t = 5, whose 9 do not. Every round binds each wire once, the same
    registers in every round: control, qpe_grad and walk_wires (the walk's 63 wires, then the
    spare ones). Round 0's corrections borrow the pool from its first wire on, into the spare
    wire: for d = l + 1, the scratch wires corrections[:d] and the work wires
    corrections[d:2d - 1]."""
    c = walk_sim.case("sto3g")
    program, num_wires = qpe_program(c.params, c.tables, t)
    layout = WalkLayout.from_params(c.params, qpe_bits=t)
    assert num_wires == layout.num_wires + len(spare) == 63 + 1 + t + len(spare)
    spec = walk_power_spec(c.params, c.tables, layout)
    walk_wires = [*spec.wires, *spare]
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        program()
    rounds = [op for op in queue.queue if isinstance(op, QPERound)]
    assert [op.hyperparameters["j"] for op in rounds] == list(reversed(range(t)))
    for op in rounds:
        assert list(op.control) == [63] and list(op.qpe_grad) == list(range(64, 64 + t))
        assert list(op.walk_wires) == walk_wires and sorted(walk_wires[:63]) == list(range(63))
    corrections = [*layout.work, *spare][:2 * t - 1]
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as inner:
        _qpe_round(0, t, [False, *measured(t - 1)], spec.op_name, spec.compilable,
                   spec.wire_sizes, [63], list(range(64, 64 + t)), walk_wires)
    rotations = [op.base for op in inner.queue if isinstance(op, qp.ops.Conditional)]
    assert len(rotations) == t - 1
    for d, rotation in enumerate(rotations, start=2):
        assert list(rotation.target) == [63] and list(rotation.phase_grad) == list(
            range(64, 64 + d))
        assert list(rotation.scratch) == corrections[:d]
        assert list(rotation.work) == corrections[d:2 * d - 1]


# --------------------------------------------------------------------------- energies


def test_phase_from_bits():
    """phi = sum_j m_j 2^-(j+1), the bits in the driver's order m_{t-1}, ..., m_0 (spec 11.1)."""
    assert phase_from_bits([1, 0, 0]) == 1 / 8
    assert phase_from_bits([0, 0, 1]) == 1 / 2
    assert phase_from_bits([1, 1, 1]) == 7 / 8
    assert phase_from_bits([1]) == 1 / 2 and phase_from_bits([0, 1]) == 1 / 2


def test_energy_from_phase_inverts_the_walk_phases():
    """Spec 11.1: for every eigenvalue E of H~_SA at H2_STO3G_TEST, both walk phases +-theta/2 pi
    with cos theta = E / Lambda~ - 1 give E back (shift 1); for the toy walk (shift 0, Lambda 1)
    the phases 1/8, 7/8 give 1/sqrt 2 and 3/8, 5/8 give -1/sqrt 2, the eigenvalues of
    (Z + X)/2 (spec 11.4)."""
    c = walk_sim.case("sto3g")
    h_tilde, lambda_tilde = encoded_block(c.params, c.tables, c.factors)
    for energy in np.linalg.eigvalsh(h_tilde):
        theta = np.arccos(energy / lambda_tilde - 1)
        for phi in (theta / (2 * np.pi), 1 - theta / (2 * np.pi)):
            assert np.isclose(energy_from_phase(phi, lambda_tilde), energy, rtol=0, atol=1e-12)
    toy = np.sort(np.linalg.eigvalsh(toy_walk_matrix()[np.ix_([0, 2], [0, 2])]))
    assert np.allclose(toy, [-2**-0.5, 2**-0.5], rtol=0, atol=1e-12)
    for phi, energy in ((1 / 8, toy[1]), (7 / 8, toy[1]), (3 / 8, toy[0]), (5 / 8, toy[0])):
        assert np.isclose(energy_from_phase(phi, 1.0, shift=0.0), energy, rtol=0, atol=1e-12)


def test_physical_energy_is_the_physical_frame():
    """Spec 3.2.1, 3.4 and 11.1 for h2_sto3g_R2_B2_C1 (exactly factorized): the lowest
    eigenvalue of H_SA on the two-electron sector plus E_SoS^phys is the physical ground energy
    e_upper, and plus core_energy the total; the walk frame's offset would be off by
    Delta_eta = -1.3209547."""
    system = "h2_sto3g_R2_B2_C1"
    factors, summary = load_factors(system), load_summary(system)
    keep = np.flatnonzero(np.diag(sector_projector(factors.N, factors.eta)).real)
    e_sa = np.linalg.eigvalsh(h_sa_dense(factors)[np.ix_(keep, keep)]).min()
    assert np.isclose(physical_energy(e_sa, e_sos_phys(summary)), summary["e_upper"], rtol=0,
                      atol=1e-10)
    core = float(load_integrals(system)["core_energy"])
    assert np.isclose(physical_energy(e_sa, e_sos_phys(summary), core),
                      summary["e_upper"] + core, rtol=0, atol=1e-10)
    walk_frame = e_sa + e_sos_walk(factors)
    assert np.isclose(physical_energy(e_sa, e_sos_phys(summary)) - walk_frame,
                      frame_shift(factors, summary), rtol=0, atol=1e-12)
    assert np.isclose(frame_shift(factors, summary), -1.3209547, rtol=0, atol=1e-7)
```

Create `tests/integration/test_qpe_walk.py`:

```python
"""Iterative phase estimation of the DFTHC walk itself on the sparse simulator (spec 11.1; master
plan Task 4.1): the branch distribution of the control bits equals the exact distribution the
encoded model predicts, at H2_STO3G_TEST (t = 1 nightly, t = 2 release) and at FAILS_C3 (t = 1
and t = 2 nightly), with the erase measurements forced by the BranchPolicy's `otherwise` and
every run strict. The reference formula is checked against the toy walk per commit."""

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
    A u B nonzero (the work pool back in |0>). BranchPolicy follows both outcomes of each control
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
    psi = walk_sim.basis(c, INPUTS[name])
    assert np.allclose(found, qpe_distribution(c.block(), psi, num_bits), rtol=0, atol=1e-12)
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/test_qpe.py tests/integration/test_qpe_toy.py tests/integration/test_qpe_walk.py -q`
Expected: FAIL at collection, `3 errors`: `ModuleNotFoundError: No module named 'dfthc.qpe'` (`tests/test_qpe.py`, `tests/integration/test_qpe_walk.py`) and `ImportError: cannot import name 'qpe' from 'dfthc'` (`tests/integration/test_qpe_toy.py`).

- [ ] **Step 3: Write the QPE module**

Create `src/dfthc/qpe.py`:

```python
r"""Iterative phase estimation on the controlled walk (spec 9.23 and 11).

**The rounds** (spec 11.1). With :math:`t` = ``num_bits``, rounds run for
:math:`j = t-1, \dots, 0`. Round :math:`j` is one :class:`QPERound`: ``H`` on the control, the
controlled walk to the power :math:`2^j`, a phase correction
:math:`e^{-2\pi i/2^{l-j+1}}` on the control's :math:`|1\rangle` for every earlier bit
:math:`m_l = 1` (:math:`l > j`), ``H``; then the driver measures the control, :math:`m_j`, with
reset. The estimated phase is :math:`\varphi = \sum_j m_j 2^{-(j+1)}`: the bits give the
eigenvalue :math:`e^{2\pi i\varphi}` of the walk (an exact :math:`t`-bit phase is returned with
certainty).

**The walk power.** :class:`ControlledWalkPower` repeats ``ControlledDFTHCWalk`` with a
``qp.for_loop`` of static bound ``power`` (spec 9.23); ``dfthc.testing.toy_walk``'s
``ToyControlledWalkPower`` does the same for the toy walk (spec 11.4). :class:`QPERound` cannot
take the class (an ``Operator2`` takes no Python callable) or nested wire groups (it cannot mix
compilable with hybrid arguments), so it takes the class's name, its compilable arguments after
``power``, its wire groups after ``control`` as one flat register and their sizes: a
:class:`WalkPowerSpec`. Its rule looks the name up in :data:`WALK_POWERS`, where each walk-power
class registers itself with :func:`register_walk_power`.

**The correction wires** (spec 11.1). A round binds every wire once: its corrections' ``2t - 1``
wires are a slice of the round's flat register, taken inside the rule, not a register of their
own. The slice starts at the walk power's scratch group (``scratch_argname``: its last group,
which it takes and returns in ``|0>``; ``work`` for :class:`ControlledWalkPower`) and runs on
into spare wires after the walk's groups where that group is too small
(:func:`spare_wires_needed`); the toy walk has no scratch group, so its corrections take spare
wires only. A second register on the same qubits would not compile: the wire registers of one
operator must be distinct qubits.

**The driver** :func:`iterative_phase_estimation` is a plain quantum function, not an
``Operator2``, because it returns the measured bits (spec 2.1 R-4, 16 Q-3). It passes every
round the earlier bits as one array of shape :math:`(t,)`, ``False`` where no bit is known yet,
so every round has one argument signature (spec 11.1; K-10 part 3 passed in both modes).

**Energies** (spec 3.4 and 11.1): :func:`phase_from_bits`, :func:`energy_from_phase` and
:func:`physical_energy`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import AbstractArray, Wire
from pennylane.wires import Wires

from dfthc.compile import register_rule, resource_key
from dfthc.data.tables import WalkTables
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.phase_gradient import PhaseGradientRotation
from dfthc.ops.walk import ControlledDFTHCWalk, _validate
from dfthc.params import WalkParams
from dfthc.registers import WalkLayout

WALK_POWERS: dict[str, type[Operator2]] = {}
"""The walk-power classes :class:`QPERound` can call, by class name."""


def register_walk_power(cls: type[Operator2]) -> type[Operator2]:
    """Add ``cls`` to :data:`WALK_POWERS` (once). A walk-power class takes ``power`` first, then
    its compilable arguments, then a one-wire ``control`` and its wire groups. Its class
    attribute ``scratch_argname`` names its last wire group if it takes that group in ``|0>`` and
    returns it in ``|0>`` (:class:`QPERound`'s corrections borrow it), and is ``None``
    otherwise."""
    if cls.__name__ in WALK_POWERS:
        raise ValueError(f"{cls.__name__} is already a registered walk power")
    scratch = getattr(cls, "scratch_argname", "")
    if scratch not in (None, cls.wire_argnames[-1]):
        raise ValueError(f"{cls.__name__}.scratch_argname must be None or its last wire group "
                         f"{cls.wire_argnames[-1]!r}; got {scratch!r}")
    WALK_POWERS[cls.__name__] = cls
    return cls


def _walk_power(walk_op: str) -> type[Operator2]:
    if walk_op not in WALK_POWERS:
        raise ValueError(f"walk_op must be one of {sorted(WALK_POWERS)}; got {walk_op!r}")
    return WALK_POWERS[walk_op]


def spare_wires_needed(walk_op: str, walk_wire_sizes: Sequence[int], num_bits: int) -> int:
    """The wires a ``num_bits``-bit estimation of the walk power ``walk_op`` (its groups after
    ``control`` of sizes ``walk_wire_sizes``) needs after the walk's groups: the
    ``2 num_bits - 1`` correction wires less the walk power's scratch group, at least 0.
    ``ControlledWalkPower`` lends its ``work`` pool (``WalkLayout.work_size``: 8 wires at
    ``H2_STO3G_TEST``, 44 at FeMoco-54, so no spare wire up to 4 and 22 bits); the toy walk
    lends none."""
    scratch = walk_wire_sizes[-1] if _walk_power(walk_op).scratch_argname else 0
    return max(0, 2 * num_bits - 1 - scratch)


class WalkPowerSpec(NamedTuple):
    """A walk-power operator as :class:`QPERound` takes it (spec 11.1)."""

    op_name: str
    """The class name in :data:`WALK_POWERS`."""
    compilable: tuple
    """Its compilable arguments after ``power``."""
    wire_sizes: tuple[int, ...]
    """The sizes of its wire groups after ``control``, in ``wire_argnames`` order."""
    wires: Wires
    """Those groups concatenated in the same order."""


class ControlledWalkPower(Operator2):
    r"""The controlled walk to a power, :math:`C(W)^p = C(W^p)` (spec 9.23): ``power`` steps of
    :class:`~dfthc.ops.walk.ControlledDFTHCWalk`.

    **Action.** :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle 1|_{control}
    \otimes W^p` on inputs whose ancilla groups are in :math:`|0\rangle` or in a state the walk
    reaches from there, with :math:`W` the walk of
    :class:`~dfthc.ops.walk.DFTHCWalk` for the same ``params`` and ``tables``: each step is the
    controlled walk, which is correct on the walk's own states (spec 12.4, E-3), so the steps
    compose. The output is the same for every outcome of the erase measurements.

    **Registers.** Those of :class:`~dfthc.ops.walk.ControlledDFTHCWalk`.

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``control``    1                              any                   a control
    ``outer``      the ``outer`` group (143)      walk state            walk state
    ``inner``      the ``inner`` group (455)      walk state            walk state
    ``select``     ``(N - 1) b_rot + 4`` (799)    ``|0>``               ``|0>``
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    "Walk state": ``|0>`` before the first step; between steps only the six index registers of
    :math:`\mathcal{A}\cup\mathcal{B}` (``x_o``, ``m_x_o``, ``sigma_1`` in ``outer``; ``b``,
    ``m_b``, ``sigma_0`` in ``inner``) may differ from ``|0>``. ``work`` is its scratch group
    (``scratch_argname``): :class:`QPERound`'s corrections borrow it between walk powers.

    **Compilable arguments.** ``power`` (int, at least 1), then ``params`` and ``tables`` as in
    :class:`~dfthc.ops.walk.ControlledDFTHCWalk`, with the same checks (``ValueError``).

    **Pre- and postconditions.** Those of :class:`~dfthc.ops.walk.ControlledDFTHCWalk`, on both
    control branches.

    **Decomposition** (spec 9.23). One ``qp.for_loop`` with the static bound ``power`` whose body
    is ``ControlledDFTHCWalk(params, tables, control, outer, inner, select, system, phase_grad,
    work)``, the groups unpacked into lists once before the loop (spec 9.0).

    **Adjoint and control.** Never adjointed or controlled (spec 7.4 rules 4 and 6): its body
    holds a loop and measured erases. :class:`QPERound` calls it.

    **Cost.** :func:`dfthc.costs.controlled_walk_power`: :math:`T = p\,T_{C(W)}`,
    :math:`M = p\,M_W`; ``A`` = ``p`` times :func:`dfthc.costs.measured_and_uncomputes`
    (controlled). At FeMoco-54 and ``p = 1``: 9,795, 2,524 and 8,677.

    **Paper and dfthcbe.** The walk powers :math:`W^{2^j}` of phase estimation on the
    controlled walk, whose query count the paper estimates (PRX 15, 041016; arXiv:2502.15882v1;
    spec 3.3 and 3.4). dfthcbe builds no phase estimation; its controlled walk is
    ``DFTHCWalk(is_controlled=True)``.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.params import H2_STO3G_TEST as p
        from dfthc.qpe import ControlledWalkPower
        from dfthc.registers import WalkLayout

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        layout = WalkLayout.from_params(p)
        groups = {name: getattr(layout, name) for name in ControlledWalkPower.wire_argnames[1:]}

        def program():
            qp.PhaseGradientStatePrep(layout.phase_grad)
            ControlledWalkPower(2, p.as_compilable(), t.as_compilable(), [layout.num_wires],
                                **groups)

        compiled_resources(program, num_wires=layout.num_wires + 1).toffoli  # 2 * 160 = 320
    """

    compilable_argnames = ("power", "params", "tables")
    wire_argnames = ControlledDFTHCWalk.wire_argnames
    scratch_argname = "work"

    def __init__(self, power, params, tables, control, outer, inner, select, system, phase_grad,
                 work):
        power = check_int("power", power, low=1)
        wires = (control, outer, inner, select, system, phase_grad, work)
        compilable = _validate("ControlledWalkPower", self.wire_argnames, params, tables, wires)
        super().__init__(power, *compilable, *wires)

    @staticmethod
    def required_wires(power, params, tables) -> dict[str, int]:
        """Those of :class:`~dfthc.ops.walk.ControlledDFTHCWalk`; ``power`` changes none."""
        return ControlledDFTHCWalk.required_wires(params, tables)


def _controlled_walk_power_resources(power, params, tables, control, outer, inner, select,
                                     system, phase_grad, work):
    sizes = ControlledDFTHCWalk.required_wires(params, tables)
    step = resource_key(lambda: ControlledDFTHCWalk(params, tables, **{
        name: Wire[sizes[name]] for name in ControlledDFTHCWalk.wire_argnames}))
    return {step: power}


@qp.register_resources(_controlled_walk_power_resources)
def _controlled_walk_power(power, params, tables, control, outer, inner, select, system,
                           phase_grad, work):
    groups = [list(group) for group in (control, outer, inner, select, system, phase_grad, work)]

    @qp.for_loop(power)
    def step(_):
        ControlledDFTHCWalk(params, tables, *groups)

    step()


register_rule(ControlledWalkPower, _controlled_walk_power)
register_walk_power(ControlledWalkPower)


def walk_power_spec(params: WalkParams, tables: WalkTables, layout: WalkLayout) -> WalkPowerSpec:
    """The :class:`WalkPowerSpec` of :class:`ControlledWalkPower` on ``layout``'s groups, in
    ``wire_argnames`` order (which is not the layout's order: the layout starts with
    ``system``)."""
    names = ControlledWalkPower.wire_argnames[1:]
    compilable = (params.as_compilable(), tables.as_compilable())
    sizes = ControlledWalkPower.required_wires(1, *compilable)
    return WalkPowerSpec("ControlledWalkPower", compilable, tuple(sizes[name] for name in names),
                         Wires.all_wires([getattr(layout, name) for name in names]))


def _bits_array(earlier_bits, num_bits: int):
    """``earlier_bits`` as one array of shape ``(num_bits,)`` (L-23): a boolean ``jax`` array
    under capture, an object array of ``MeasurementValue`` s and ``False`` in tape mode."""
    if isinstance(earlier_bits, AbstractArray):
        bits = earlier_bits
    elif qp.capture.enabled():
        bits = jnp.stack([jnp.asarray(bit, dtype=bool) for bit in earlier_bits])
    else:
        bits = np.empty(len(earlier_bits), dtype=object)
        bits[:] = list(earlier_bits)
    if tuple(bits.shape) != (num_bits,):
        raise ValueError(f"earlier_bits must have shape ({num_bits},); got {tuple(bits.shape)}")
    return bits


class QPERound(Operator2):
    r"""Round :math:`j` of iterative phase estimation, up to the control's measurement (spec
    11.1): ``H``, the walk power :math:`2^j`, the phase corrections of the earlier bits, ``H``.

    **Action.** On the control in :math:`|0\rangle` and a walk eigenstate with eigenvalue
    :math:`e^{2\pi i\varphi}`, it leaves the control in
    :math:`\big((1 + e^{i\omega})|0\rangle + (1 - e^{i\omega})|1\rangle\big)/2` with
    :math:`\omega = 2\pi\big(2^j\varphi - \sum_{l > j} m_l\,2^{-(l-j+1)}\big)`, where
    :math:`m_l` = ``earlier_bits[l]``; measuring it gives :math:`m_j`. For a :math:`t`-bit
    :math:`\varphi` with the earlier bits right, :math:`\omega = \pi m_j` exactly.

    **Registers.**

    ============== ================================== ===================== ==================
    name           width                              input                 output
    ============== ================================== ===================== ==================
    ``control``    1                                  ``|0>``               see the action
    ``qpe_grad``   ``num_bits``                       phase-gradient state  unchanged
    ``walk_wires`` ``sum(walk_wire_sizes) + spare``   see below             see below
    ============== ================================== ===================== ==================

    ``walk_wires`` are the walk power's wire groups after its ``control``, concatenated in its
    ``wire_argnames`` order (the walk's input and output), then ``spare`` =
    :func:`spare_wires_needed` wires in ``|0>``. The corrections use the ``2 num_bits - 1``
    wires of ``walk_wires`` from the first wire of the walk power's scratch group (its last
    group, ``|0>`` whenever corrections run; ``work`` for ``ControlledWalkPower``) on, into the
    spare wires where the group is too small; without a scratch group (the toy walk) they are
    the spare wires. So each wire is bound once (the wire registers of one operator must be
    distinct qubits).

    **Compilable arguments.** ``j`` (int, ``0 <= j < num_bits``), ``num_bits`` (int, at least
    1), ``walk_op`` (a name in :data:`WALK_POWERS`), ``walk_compilable`` (the walk power's
    compilable arguments after ``power``, a tuple) and ``walk_wire_sizes`` (the walk power's
    ``required_wires`` after ``control``, in order).
    **Dynamic:** ``earlier_bits``, shape ``(num_bits,)``: ``m_l`` at every ``l > j``, ``False``
    elsewhere; the rule reads only ``l > j``.

    **Pre- and postconditions.** ``qpe_grad`` holds the ``num_bits``-bit phase-gradient state;
    the walk power's scratch group and the spare wires start and end in ``|0>``; the walk
    power's own conditions hold on its groups.

    **Decomposition** (spec 11.1). ``H(control)``; ``WALK_POWERS[walk_op](2**j,
    *walk_compilable, control, *groups)`` with ``groups`` split from ``walk_wires`` by
    ``walk_wire_sizes``; ``corrections`` the ``2 num_bits - 1`` correction wires above; for each
    ``l = j + 1, ..., num_bits - 1`` with ``d = l - j + 1``, ``qp.cond(earlier_bits[l],
    correction)()``, whose branch builds ``PhaseGradientRotation(2**d - 1, d, control,
    qpe_grad[:d], corrections[:d], corrections[d:2 d - 1])``, the phase
    :math:`e^{-2\pi i/2^d}` on :math:`|1\rangle` (the top ``d`` wires of a ``num_bits``-bit
    gradient are a ``d``-bit gradient); ``H(control)``. The corrections are not hinted, so
    ``qp.analyze`` counts each at its maximum branch.

    **Adjoint and control.** Never adjointed or controlled (spec 7.4 rules 4 to 6): it holds the
    walk power's loop, conditionals on measured bits and measured erases.

    **Cost.** :func:`dfthc.costs.qpe_round`, worst case: :math:`T = 2^j T_{C(W)} +
    \sum_{l=j+1}^{t-1}(l - j)`, :math:`M = 2^j M_W` (a ``d``-bit rotation costs ``d - 1``
    Toffolis, spec 9.1). With ``walk_op = "ControlledWalkPower"``.

    **Paper and dfthcbe.** The semiclassical (iterative) phase estimation with feed-forward
    rotations (Griffiths and Niu, PRL 76, 3228 (1996)) on the controlled walk; the paper's
    estimate costs and dfthcbe have no circuit for it. The corrections are phase-gradient
    additions (Sanders et al., PRX Quantum 1, 020312), so no inverse QFT is needed (spec 11.2).

    **Example.**

    .. code-block:: python

        import numpy as np
        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.qpe import QPERound
        from dfthc.testing.toy_walk import toy_walk_power_spec

        spec = toy_walk_power_spec(system=[3], ancilla=[4])

        def program():
            qp.PhaseGradientStatePrep(wires=[1, 2])
            QPERound(0, 2, np.array([False, True]), spec.op_name, spec.compilable,
                     spec.wire_sizes, control=[0], qpe_grad=[1, 2], walk_wires=[3, 4, 5, 6, 7])

        compiled_resources(program, num_wires=8).toffoli  # 2 + 1 = 3
    """

    dynamic_argnames = ("earlier_bits",)
    compilable_argnames = ("j", "num_bits", "walk_op", "walk_compilable", "walk_wire_sizes")
    wire_argnames = ("control", "qpe_grad", "walk_wires")

    def __init__(self, j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes,
                 control, qpe_grad, walk_wires):
        num_bits = check_int("num_bits", num_bits, low=1)
        j = check_int("j", j, low=0, high=num_bits - 1)
        cls = _walk_power(walk_op)
        if not isinstance(walk_compilable, tuple):
            raise ValueError(f"walk_compilable must be a tuple; got {walk_compilable!r}")
        walk_wire_sizes = canonical_ints("walk_wire_sizes", tuple(walk_wire_sizes))
        required = cls.required_wires(2**j, *walk_compilable)
        groups = tuple(size for name, size in required.items() if name != "control")
        if walk_wire_sizes != groups:
            raise ValueError(f"walk_wire_sizes must be {walk_op}'s required_wires after control, "
                             f"{groups}; got {walk_wire_sizes}")
        bits = _bits_array(earlier_bits, num_bits)
        check_wires("QPERound", {"control": control, "qpe_grad": qpe_grad,
                                 "walk_wires": walk_wires},
                    self.required_wires(j, num_bits, walk_op, walk_compilable, walk_wire_sizes))
        super().__init__(j, num_bits, bits, walk_op, walk_compilable, walk_wire_sizes, control,
                         qpe_grad, walk_wires)

    @staticmethod
    def required_wires(j, num_bits, walk_op, walk_compilable, walk_wire_sizes) -> dict[str, int]:
        """Exact sizes; every round of a ``num_bits``-bit estimation has the same ones."""
        return {"control": 1, "qpe_grad": num_bits,
                "walk_wires": sum(walk_wire_sizes)
                + spare_wires_needed(walk_op, walk_wire_sizes, num_bits)}


def _corrections(j: int, num_bits: int) -> list[int]:
    """The correction widths ``d = l - j + 1`` of round ``j``, for ``l = j + 1 .. num_bits - 1``."""
    return [later - j + 1 for later in range(j + 1, num_bits)]


def _qpe_round_resources(j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes,
                         control, qpe_grad, walk_wires):
    cls = WALK_POWERS[walk_op]
    walk = resource_key(lambda: cls(2**j, *walk_compilable, Wire[1],
                                    *[Wire[size] for size in walk_wire_sizes]))
    resources = {qp.H: 2, walk: 1}
    for d in _corrections(j, num_bits):
        rotation = resource_key(lambda d=d: PhaseGradientRotation(
            2**d - 1, d, Wire[1], Wire[d], Wire[d], Wire[d - 1]))
        resources[rotation] = 1
    return resources


@qp.register_resources(_qpe_round_resources)
def _qpe_round(j, num_bits, earlier_bits, walk_op, walk_compilable, walk_wire_sizes, control,
               qpe_grad, walk_wires):
    control, qpe_grad, walk_wires = list(control), list(qpe_grad), list(walk_wires)
    groups, start = [], 0
    for size in walk_wire_sizes:
        groups.append(walk_wires[start:start + size])
        start += size
    if WALK_POWERS[walk_op].scratch_argname:
        start -= walk_wire_sizes[-1]
    corrections = walk_wires[start:start + 2 * num_bits - 1]
    qp.H(control)
    WALK_POWERS[walk_op](2**j, *walk_compilable, control, *groups)
    for later, d in zip(range(j + 1, num_bits), _corrections(j, num_bits), strict=True):
        def correction(d=d):
            PhaseGradientRotation(2**d - 1, d, control, qpe_grad[:d], corrections[:d],
                                  corrections[d:2 * d - 1])

        qp.cond(earlier_bits[later], correction)()
    qp.H(control)


register_rule(QPERound, _qpe_round)


def _no_bits(num_bits: int):
    if qp.capture.enabled():
        return jnp.zeros(num_bits, dtype=bool)
    return np.full(num_bits, False, dtype=object)


def _with_bit(bits, j: int, bit):
    if qp.capture.enabled():
        return bits.at[j].set(jnp.asarray(bit, dtype=bool))
    bits = bits.copy()
    bits[j] = bit
    return bits


def iterative_phase_estimation(walk_power_factory: Callable[[], WalkPowerSpec], control,
                               num_bits: int, qpe_grad, spare_wires=()) -> list:
    r"""Iterative phase estimation of the walk power that ``walk_power_factory()`` names (spec
    11.1): the measured bits :math:`m_{t-1}, \dots, m_0` in measurement order, with
    :math:`t` = ``num_bits`` and :math:`\varphi = \sum_j m_j 2^{-(j+1)}` (:func:`phase_from_bits`).

    A plain quantum function (spec 2.1 R-4, 16 Q-3): it prepares the ``num_bits``-bit phase
    gradient on ``qpe_grad[:num_bits]`` (``PhaseGradientStatePrep``), then for
    :math:`j = t-1, \dots, 0` applies :class:`QPERound` with the earlier bits as one array of
    shape :math:`(t,)` and measures ``control`` with reset, so it ends in :math:`|0\rangle`.
    ``control`` (one wire) starts in :math:`|0\rangle`. The corrections borrow the walk power's
    scratch group and, where it is too small, the first :func:`spare_wires_needed` wires of
    ``spare_wires`` (in :math:`|0\rangle`, disjoint from the walk's wires; the rest unused).
    The walk's own registers keep the walk's state between rounds; the caller prepares the
    walk's ``phase_grad``. Returns ``MeasurementValue`` s in tape mode and traced booleans under
    capture. Raises ``ValueError`` if ``num_bits`` is not a positive int, exceeds
    ``len(qpe_grad)``, if ``spare_wires`` is shorter than the corrections need, or if the wires
    the rounds use are not distinct.
    """
    num_bits = check_int("num_bits", num_bits, low=1)
    if num_bits > len(qpe_grad):
        raise ValueError(f"num_bits = {num_bits} exceeds len(qpe_grad) = {len(qpe_grad)}")
    spec = walk_power_factory()
    spare = spare_wires_needed(spec.op_name, spec.wire_sizes, num_bits)
    if len(spare_wires) < spare:
        raise ValueError(f"spare_wires needs {spare} wires for num_bits = {num_bits} with "
                         f"{spec.op_name}; got {len(spare_wires)}")
    control, qpe_grad = list(control), list(qpe_grad)[:num_bits]
    walk_wires = [*spec.wires, *list(spare_wires)[:spare]]
    if len({*control, *qpe_grad, *walk_wires}) != len(control) + num_bits + len(walk_wires):
        raise ValueError("control, qpe_grad, the walk's wires and the spare wires must be "
                         "distinct")
    qp.PhaseGradientStatePrep(qpe_grad)
    earlier, measured = _no_bits(num_bits), []
    for j in reversed(range(num_bits)):
        QPERound(j, num_bits, earlier, spec.op_name, spec.compilable, spec.wire_sizes, control,
                 qpe_grad, walk_wires)
        bit = qp.measure(control[0], reset=True)
        measured.append(bit)
        earlier = _with_bit(earlier, j, bit)
    return measured


def phase_from_bits(bits: Sequence[int]) -> float:
    r""":math:`\varphi = \sum_j m_j 2^{-(j+1)}` from the bits in the driver's order
    :math:`m_{t-1}, \dots, m_0` (spec 11.1)."""
    t = len(bits)
    return sum(int(bit) * 2.0 ** -(t - position) for position, bit in enumerate(bits))


def energy_from_phase(phi: float, Lambda: float, shift: float = 1.0) -> float:
    r"""The eigenvalue :math:`E = \Lambda(\mathrm{shift} + \cos 2\pi\varphi)` that the phase
    :math:`\varphi` of the walk estimates (spec 11.1): with ``Lambda`` the encoded
    :math:`\tilde\Lambda` and ``shift = 1`` the sum-of-squares eigenvalue :math:`E_{SA}` of
    :math:`\tilde H_{SA}` for the DFTHC walk (block :math:`\tilde H_{SA}/\tilde\Lambda - I`),
    ``shift = 0`` for standard qubitization (block :math:`H/\lambda`). The phases
    :math:`\pm\varphi` give the same energy."""
    return float(Lambda * (shift + np.cos(2 * np.pi * phi)))


def physical_energy(E_SA: float, e_sos_phys: float, core_energy: float = 0.0) -> float:
    r""":math:`E_{SA} + E_{SoS}^{phys}` + ``core_energy``: the physical electronic energy of an
    :math:`\eta`-electron eigenstate, plus the core energy where it is given (spec 3.4 and 11.1).
    ``e_sos_phys`` is ``summary.json``'s ``e_sos`` (:func:`dfthc.data.sos.e_sos_phys`), not the
    walk frame's offset."""
    return float(E_SA + e_sos_phys + core_energy)


def qpe_program(params: WalkParams, tables: WalkTables,
                num_bits: int) -> tuple[Callable[[], list], int]:
    """The QPE program of spec 11.3 at ``params`` and its wire count: the walk's phase gradient,
    then :func:`iterative_phase_estimation` of :class:`ControlledWalkPower` on the groups of
    ``WalkLayout.from_params(params, qpe_bits=num_bits)``, its ``qpe_control`` and ``qpe_grad``.
    The corrections borrow the walk's ``work`` pool; only where ``2 num_bits - 1`` exceeds it
    (from 5 bits at ``H2_STO3G_TEST``, from 23 at FeMoco-54) do spare wires follow the
    layout."""
    layout = WalkLayout.from_params(params, qpe_bits=num_bits)
    spec = walk_power_spec(params, tables, layout)
    spare = spare_wires_needed(spec.op_name, spec.wire_sizes, num_bits)
    spare_wires = list(range(layout.num_wires, layout.num_wires + spare))

    def program():
        qp.PhaseGradientStatePrep(layout.phase_grad)
        return iterative_phase_estimation(lambda: spec, layout.qpe_control, num_bits,
                                          layout.qpe_grad, spare_wires)

    return program, layout.num_wires + spare
```

- [ ] **Step 4: Write the toy walk**

Create `src/dfthc/testing/toy_walk.py`:

```python
r"""The toy qubitization walk of :math:`H = (Z + X)/2` for phase-estimation tests (spec 11.4).

One system qubit and one LCU ancilla: PREP is a Hadamard on the ancilla, SELECT is a zero-controlled
``Z`` plus a one-controlled ``X`` on the system, and the walk is
:math:`W = \mathrm{REF}_{anc}\,\mathrm{PREP}^\dagger\,\mathrm{SEL}\,\mathrm{PREP}` with
:math:`\mathrm{REF}_{anc} = 2|0\rangle\langle 0| - I = Z`. Its block on the ancilla's
:math:`|0\rangle` is :math:`H` (:math:`\lambda = 1`), so its eigenphases are
:math:`\pm\arccos(\pm 1/\sqrt 2) = \pm\pi/4, \pm 3\pi/4`, exact 3-bit phases
:math:`\varphi \in \{1/8, 3/8, 5/8, 7/8\}`.

The walk has the DFTHC walk's form :math:`\mathrm{REF}\,U^\dagger R\,U` with :math:`U` = PREP and
:math:`R` = SELECT, itself a reflection (a Hermitian unitary). So :class:`ControlledToyWalk`
controls exactly its two reflections, as ``ControlledDFTHCWalk`` controls ``RefAB`` and ``RefT2``:
:math:`C(W) = C(\mathrm{REF}_{anc})\,\mathrm{PREP}^\dagger\,C(\mathrm{SEL})\,\mathrm{PREP}`, and
with the control in :math:`|0\rangle` the rest is :math:`\mathrm{PREP}^\dagger\mathrm{PREP} = I`.
Controlling the ancilla reflection alone would leave :math:`\mathrm{PREP}^\dagger\,\mathrm{SEL}\,
\mathrm{PREP} \ne I` on the control's :math:`|0\rangle` (the toy has one reflection outside
:math:`U`, the DFTHC walk two). :class:`ToyControlledWalkPower` repeats the controlled walk with a
``qp.for_loop``, as ``ControlledWalkPower`` repeats ``ControlledDFTHCWalk``; it is registered as a
walk power of :class:`dfthc.qpe.QPERound`.

The dense references (:func:`toy_walk_matrix`, :func:`controlled_toy_walk_matrix`) are built from
the definition above, never from the circuit. The costs (:func:`toy_walk_cost`) live here because
the toy is test scaffolding, not a spec 9.24 block.
"""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire
from pennylane.wires import Wires

from dfthc.compile import register_rule, resource_key
from dfthc.costs import Cost
from dfthc.ops._validate import check_int, check_wires
from dfthc.qpe import WalkPowerSpec, iterative_phase_estimation, register_walk_power

_I2 = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_P0, _P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)


def toy_walk_matrix() -> np.ndarray:
    r"""The 4 x 4 matrix of :math:`W` on (``system``, ``ancilla``), ``system`` the more
    significant qubit, from spec 11.4's definition."""
    prep = np.kron(_I2, _H)
    select = np.kron(_Z, _P0) + np.kron(_X, _P1)
    reflection = np.kron(_I2, 2 * _P0 - _I2)
    return reflection @ prep.conj().T @ select @ prep


def controlled_toy_walk_matrix() -> np.ndarray:
    r""":math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes W` on (``control``,
    ``system``, ``ancilla``)."""
    return np.kron(_P0, np.eye(4)) + np.kron(_P1, toy_walk_matrix())


def toy_walk_cost(controlled: bool) -> Cost:
    """``ToyWalk``: no Toffoli; ``ControlledToyWalk``: the two Toffolis of the controlled SELECT.
    No measurement either way."""
    return Cost(2 if controlled else 0)


def toy_walk_power_cost(power: int) -> Cost:
    """``ToyControlledWalkPower(power)``: ``power`` controlled toy walks."""
    return power * toy_walk_cost(True)


def _check(name: str, wires: dict) -> None:
    check_wires(name, wires, {wire: 1 for wire in wires})


class ToyWalk(Operator2):
    r"""The toy qubitization walk :math:`W = \mathrm{REF}_{anc}\,\mathrm{PREP}^\dagger\,
    \mathrm{SEL}\,\mathrm{PREP}` of :math:`H = (Z + X)/2` (spec 11.4).

    **Action.** :func:`toy_walk_matrix` on (``system``, ``ancilla``): eigenphases
    :math:`\pm\pi/4` (the eigenvalue :math:`+1/\sqrt 2` of :math:`H`) and :math:`\pm 3\pi/4`
    (:math:`-1/\sqrt 2`); its block on the ancilla's :math:`|0\rangle` is :math:`H`.

    **Registers.**

    ============ ===== ===== ==============
    name         width input output
    ============ ===== ===== ==============
    ``system``   1     any   see the action
    ``ancilla``  1     any   see the action
    ============ ===== ===== ==============

    **Compilable arguments.** None.

    **Pre- and postconditions.** None: it is unitary on every input.

    **Decomposition.** PREP: ``H(ancilla)``. SELECT: ``X(ancilla)``, ``CZ(ancilla, system)``,
    ``X(ancilla)`` (the zero-controlled ``Z``), ``CNOT(ancilla, system)`` (the one-controlled
    ``X``). PREP†: ``H(ancilla)``. :math:`\mathrm{REF}_{anc}`: ``Z(ancilla)``, exactly
    :math:`2|0\rangle\langle 0| - I`.

    **Adjoint and control.** Never adjointed or controlled: the controlled form is the class
    :class:`ControlledToyWalk` (spec 7.4 rule 6); there is no ``C(ToyWalk)`` rule.

    **Cost.** :func:`toy_walk_cost` (uncontrolled): ``T = 0``, ``M = 0``.

    **Paper and dfthcbe.** The qubitization walk of Low and Chuang (Quantum 3, 163 (2019)) for a
    two-term LCU; test scaffolding of spec 11.4 with no counterpart in the paper or dfthcbe.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.testing.toy_walk import ToyWalk

        compiled_resources(lambda: ToyWalk(system=[0], ancilla=[1]), num_wires=2).toffoli  # 0
    """

    wire_argnames = ("system", "ancilla")

    def __init__(self, system, ancilla):
        _check("ToyWalk", {"system": system, "ancilla": ancilla})
        super().__init__(system, ancilla)

    @staticmethod
    def required_wires() -> dict[str, int]:
        return {"system": 1, "ancilla": 1}


def _toy_walk_resources(system, ancilla):
    return {qp.H: 2, qp.X: 2, qp.CZ: 1, qp.CNOT: 1, qp.Z: 1}


@qp.register_resources(_toy_walk_resources)
def _toy_walk(system, ancilla):
    qp.H(ancilla)
    qp.X(ancilla)
    qp.CZ([ancilla[0], system[0]])
    qp.X(ancilla)
    qp.CNOT([ancilla[0], system[0]])
    qp.H(ancilla)
    qp.Z(ancilla)


register_rule(ToyWalk, _toy_walk)


class ControlledToyWalk(Operator2):
    r"""The controlled toy walk, the explicit controlled class of :class:`ToyWalk` (spec 7.4 rule
    6): :math:`C(W) = C(\mathrm{REF}_{anc})\,\mathrm{PREP}^\dagger\,C(\mathrm{SEL})\,
    \mathrm{PREP}`, which controls only the toy's two reflections.

    **Action.** :func:`controlled_toy_walk_matrix`, exactly :math:`|0\rangle\langle 0| \otimes I
    + |1\rangle\langle 1| \otimes W` on every input, global phase included: control ``|0>`` keeps
    the input (:math:`\mathrm{PREP}^\dagger\mathrm{PREP} = I`), control ``|1>`` gives
    :class:`ToyWalk`'s output, and control ``|+>`` their superposition.

    **Registers.** Those of :class:`ToyWalk` after a one-wire ``control``.

    ============ ===== ===== ==============
    name         width input output
    ============ ===== ===== ==============
    ``control``  1     any   a control
    ``system``   1     any   see the action
    ``ancilla``  1     any   see the action
    ============ ===== ===== ==============

    **Compilable arguments.** None.

    **Pre- and postconditions.** None: it is unitary on every input.

    **Decomposition.** PREP: ``H(ancilla)``. The controlled SELECT, two ``Toffoli`` s:
    ``X(ancilla)``, ``H(system)``, ``Toffoli(control, ancilla, system)``, ``H(system)``,
    ``X(ancilla)`` (``Z`` on the system where the control is 1 and the ancilla 0), then
    ``Toffoli(control, ancilla, system)`` (``X`` where both are 1). PREP†: ``H(ancilla)``. The
    controlled :math:`\mathrm{REF}_{anc}`: ``CZ(control, ancilla)``. SELECT is the reflection
    inside :math:`U^\dagger R\,U` (here :math:`U` = PREP), the role ``RefT2`` plays in the DFTHC
    walk; controlling only :math:`\mathrm{REF}_{anc}` would apply
    :math:`\mathrm{PREP}^\dagger\,\mathrm{SEL}\,\mathrm{PREP}` on the control's ``|0>``.

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0);
    :class:`ToyControlledWalkPower` repeats it.

    **Cost.** :func:`toy_walk_cost` (controlled): ``T = 2`` (two ``Toffoli``, ``A = 0``),
    ``M = 0``.

    **Paper and dfthcbe.** The controlled walk of spec 3.3 in toy form (control only the
    reflections); test scaffolding of spec 11.4.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.testing.toy_walk import ControlledToyWalk

        program = lambda: ControlledToyWalk(control=[0], system=[1], ancilla=[2])
        compiled_resources(program, num_wires=3).toffoli  # 2
    """

    wire_argnames = ("control", "system", "ancilla")

    def __init__(self, control, system, ancilla):
        _check("ControlledToyWalk", {"control": control, "system": system, "ancilla": ancilla})
        super().__init__(control, system, ancilla)

    @staticmethod
    def required_wires() -> dict[str, int]:
        return {"control": 1, "system": 1, "ancilla": 1}


def _controlled_toy_walk_resources(control, system, ancilla):
    return {qp.H: 4, qp.X: 2, qp.Toffoli: 2, qp.CZ: 1}


@qp.register_resources(_controlled_toy_walk_resources)
def _controlled_toy_walk(control, system, ancilla):
    qp.H(ancilla)
    qp.X(ancilla)
    qp.H(system)
    qp.Toffoli([control[0], ancilla[0], system[0]])
    qp.H(system)
    qp.X(ancilla)
    qp.Toffoli([control[0], ancilla[0], system[0]])
    qp.H(ancilla)
    qp.CZ([control[0], ancilla[0]])


register_rule(ControlledToyWalk, _controlled_toy_walk)


class ToyControlledWalkPower(Operator2):
    r"""The controlled toy walk to a power, :math:`C(W^p)`: ``power`` steps of
    :class:`ControlledToyWalk` (the toy twin of ``dfthc.qpe.ControlledWalkPower``, spec 11.4).

    **Action.** :math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes W^p` on
    (``control``, ``system``, ``ancilla``).

    **Registers.** Those of :class:`ControlledToyWalk`.

    ============ ===== ===== ==============
    name         width input output
    ============ ===== ===== ==============
    ``control``  1     any   a control
    ``system``   1     any   see the action
    ``ancilla``  1     any   see the action
    ============ ===== ===== ==============

    **Compilable arguments.** ``power`` (int, at least 1).

    **Pre- and postconditions.** None: it is unitary on every input.

    **Decomposition.** One ``qp.for_loop`` with the static bound ``power`` whose body is
    ``ControlledToyWalk(control, system, ancilla)``.

    **Adjoint and control.** Never adjointed or controlled (spec 7.4 rules 4 and 6); registered
    in :data:`dfthc.qpe.WALK_POWERS` for :class:`dfthc.qpe.QPERound`, with no scratch group
    (``scratch_argname = None``), so the round's corrections take spare wires.

    **Cost.** :func:`toy_walk_power_cost`: ``T = 2 power``, ``M = 0``.

    **Paper and dfthcbe.** Test scaffolding of spec 11.4.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.testing.toy_walk import ToyControlledWalkPower

        program = lambda: ToyControlledWalkPower(4, control=[0], system=[1], ancilla=[2])
        compiled_resources(program, num_wires=3).toffoli  # 8
    """

    compilable_argnames = ("power",)
    wire_argnames = ("control", "system", "ancilla")
    scratch_argname = None

    def __init__(self, power, control, system, ancilla):
        power = check_int("power", power, low=1)
        _check("ToyControlledWalkPower", {"control": control, "system": system,
                                          "ancilla": ancilla})
        super().__init__(power, control, system, ancilla)

    @staticmethod
    def required_wires(power) -> dict[str, int]:
        return {"control": 1, "system": 1, "ancilla": 1}


def _toy_controlled_walk_power_resources(power, control, system, ancilla):
    step = resource_key(lambda: ControlledToyWalk(Wire[1], Wire[1], Wire[1]))
    return {step: power}


@qp.register_resources(_toy_controlled_walk_power_resources)
def _toy_controlled_walk_power(power, control, system, ancilla):
    control, system, ancilla = list(control), list(system), list(ancilla)

    @qp.for_loop(power)
    def step(_):
        ControlledToyWalk(control, system, ancilla)

    step()


register_rule(ToyControlledWalkPower, _toy_controlled_walk_power)
register_walk_power(ToyControlledWalkPower)


def toy_walk_power_spec(system, ancilla) -> WalkPowerSpec:
    """The :class:`dfthc.qpe.WalkPowerSpec` of :class:`ToyControlledWalkPower` on ``system``
    and ``ancilla`` (no compilable arguments after ``power``)."""
    return WalkPowerSpec("ToyControlledWalkPower", (), (1, 1), Wires.all_wires([system, ancilla]))


def toy_qpe_wires(num_bits: int) -> dict[str, list[int]]:
    """The wires of :func:`toy_qpe`: ``control`` 0, ``qpe_grad`` (``num_bits``), the corrections'
    ``spare_wires`` (``2 num_bits - 1``: the toy walk lends no scratch group), then ``system`` and
    ``ancilla``; ``3 num_bits + 2`` wires in all."""
    t = num_bits
    return {"control": [0], "qpe_grad": list(range(1, t + 1)),
            "spare_wires": list(range(t + 1, 3 * t)), "system": [3 * t], "ancilla": [3 * t + 1]}


def toy_qpe(num_bits: int) -> list:
    """Iterative phase estimation of the toy walk with ``num_bits`` bits on
    :func:`toy_qpe_wires` (system and ancilla start in the caller's state, ``|0>`` by default):
    the measured bits, as :func:`dfthc.qpe.iterative_phase_estimation` returns them."""
    wires = toy_qpe_wires(num_bits)
    return iterative_phase_estimation(
        lambda: toy_walk_power_spec(wires["system"], wires["ancilla"]), wires["control"],
        num_bits, wires["qpe_grad"], wires["spare_wires"])
```

- [ ] **Step 5: Run the per-commit tests**

Run: `scripts/test.sh tests/test_qpe.py tests/integration/test_qpe_toy.py tests/integration/test_qpe_walk.py -q -m "not slow and not release"`
Expected: `54 passed` (dry run 28.3 s with 2 workers).

- [ ] **Step 6: Run the nightly checks**

Run: `scripts/test.sh tests/test_qpe.py tests/integration/test_qpe_toy.py tests/integration/test_qpe_walk.py -q -m "slow and not release"`
Expected: `7 passed` (dry run 163.5 s with 2 workers, while another lane ran its tests; in a run with `--durations` the four walk checks took `test_qpe_of_the_walk_is_exact[fails-c3-2-seeded]` 144 s, `[sto3g-1-seeded]` 50 s, the two `fails-c3-1` cases 12 s each).

- [ ] **Step 7: Check the release placement (do not run it)**

Run: `scripts/test.sh tests/test_qpe.py tests/integration/test_qpe_toy.py tests/integration/test_qpe_walk.py --collect-only -q -m release`
Expected: `1/62 tests collected (61 deselected)` (`test_qpe_of_the_walk_is_exact[sto3g-2-seeded]`, about 8 minutes: it runs in the release tier, which only the orchestrator starts).

- [ ] **Step 8: Run the guards**

Run: `scripts/test.sh tests/test_tolerance_hygiene.py tests/test_compile.py tests/test_canary_helper.py -q`
Expected: `71 passed` (the tolerance guard covers `toy_walk.py`; the pins of the five new keys are asserted in the new files).

- [ ] **Step 9: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1107 passed` (1053 + 54; see "Expected test counts").

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 10: Commit**

```bash
git add src/dfthc/qpe.py src/dfthc/testing/toy_walk.py tests/test_qpe.py tests/integration/test_qpe_toy.py tests/integration/test_qpe_walk.py
git commit -m "Add ControlledWalkPower, QPERound, the iterative QPE driver, energy helpers and the toy walk"
```

- [ ] **Step 11: Report the tiers (the orchestrator records them)**

In the report give: the per-commit tier's time and peak from Step 9; the nightly checks' time from Step 6 (dry run `7 passed in 163.47s` with 2 workers); and the release-only check's dry-run time (487 s alone, `H2_STO3G_TEST` t = 2, `stages/4Q/qpe-walk-release-4q.log`), with any difference from the dry run's numbers.

---

### Task 4.2: QPE resource counts

**Task id:** `4.2` · **Branch:** `phase4/4.2` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 4.1

**Files:**
- Create: `tests/integration/test_qpe_counts.py`

**Interfaces:**
- Consumes: Task 4.1's `dfthc.qpe.qpe_program(params, tables, num_bits) -> (qfunc, num_wires)`; Task 3.4's `ControlledDFTHCWalk` and `dfthc.testing.resources` (`captured_targets`, `inventory`, `LEAF_ADJOINTS`, `TEMPLATE_ADJOINTS`); Task 3.3's `dfthc.testing.walk_sim.case`; `dfthc.costs` (`walk`, `measured_and_uncomputes`, `iterative_qpe`, `Cost`); `dfthc.compile.compiled_resources`; `dfthc.registers.WalkLayout`; `dfthc.testing.watchdog` (`run_watched`, `effective_limit_mb`).
- Produces: the counts of spec 11.3 at the test presets and their `QPE_COMPILE` lines (the orchestrator copies them into the ledger; Task 4.3 reads the numbers). Nothing for later tasks to import.

**What the task does, in plain terms.** The QPE program of Task 4.1 is compiled six times (two presets, 1 to 3 bits) and its exact Toffoli (T), measurement (M) and measured-uncompute (A) counts are compared with two things: the closed form `dfthc.costs.iterative_qpe(p, t)`, and the formula built from the controlled walk compiled at the same preset: T = (2^t − 1) T_C(W) + (t³ − t)/6, M = (2^t − 1) M_W + t, A = (2^t − 1) A_C(W) + (t³ − t)/6. Every compile runs in a separate Python process under the watchdog (`run_watched`, 4 GB cap), which records its time and memory; the test prints them on one `QPE_COMPILE` line per compile. Those lines also record `expected_clifford_totals`, the program's `CZ` and `PauliZ` totals: the deterministic `CZ` and `Z` gates of `InnerSigns` and of the reflections, plus the erase fix-ups' conditional `CZ` and `Z`, which `qp.analyze` counts at their expected value (each fires with probability 1/2, a hint); the task records these totals without asserting them (spec 7.6). The QPE's own corrections are not hinted, so T is the worst case. The file also runs the inventory check of Task 3.4 on the 2-bit program at `H2_STO3G_TEST`, and checks the FeMoco-54 closed form without compiling anything at FeMoco shape. The test is seen failing first: Step 2 hints the corrections in `qpe.py`, Step 3 runs the two cases where that changes T, and Step 4 restores the file before the first passing run.

**Acceptance tests** (8 tests: 7 `slow`, 1 per commit):
- `test_qpe_counts[name-t]` (6, `slow`, `compiled`, `subprocess`): for `name` in `sto3g`, `631g` and t in 1, 2, 3, the controlled walk and the QPE program each compile in a subprocess under `run_watched` (cap 4,096 MB, so the watchdog kills at 3,840 MB; 900 s), each printing `QPE_COMPILE {"watchdog": ..., "result": ...}`; the controlled walk equals `costs.walk(p, True)` and `costs.measured_and_uncomputes(p, True)`; the QPE program's T, M and A equal the formula from the compiled controlled walk and `costs.iterative_qpe(p, t)`; its wire count is `WalkLayout.from_params(p, qpe_bits=t).num_wires` (the corrections in the work pool). Values: `H2_STO3G_TEST` T = 160, 481, 1,124, M = 51, 152, 353, A = 110, 331, 774; `H2_631G_TEST` T = 242, 727, 1,698, M = 69, 206, 479, A = 181, 544, 1,271.
- `test_inventory_of_the_qpe_program` (`slow`, `compiled`): at `H2_STO3G_TEST` with t = 2, `captured_targets` sees both rounds and both walk powers (two of each), the controlled walk's explicit children and the correction; no target name contains `C(`; `inventory(names) == LEAF_ADJOINTS | TEMPLATE_ADJOINTS`.
- `test_qpe_closed_forms_at_femoco` (per commit): `costs.iterative_qpe(FEMOCO54, t)` is (9,795, 2,525), (29,386, 7,574), (68,569, 17,671) for t = 1, 2, 3.

**Definition of done:** the RED step seen first and `qpe.py` restored; then `8 passed` targeted; the `QPE_COMPILE` lines in the report; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the test file**

Create `tests/integration/test_qpe_counts.py` (the directory exists since Task 3.4; no `__init__.py`):

```python
"""QPE resource counts (spec 11.3; master plan Task 4.2): the iterative QPE program compiles at
H2_STO3G_TEST and H2_631G_TEST for t = 1, 2, 3, and its worst-case counts equal
dfthc.costs.iterative_qpe exactly: T = (2^t - 1) T_C(W) + (t^3 - t)/6 and M = (2^t - 1) M_W + t,
with T_C(W) and M_W compiled at the same preset (A likewise: (2^t - 1) A_C(W) + (t^3 - t)/6).
Every compile runs in a subprocess under run_watched, and its QPE_COMPILE line records the
watched time and peak memory and the expected CZ and PauliZ totals. The inventory test of Task
3.4 passes for the program at H2_STO3G_TEST with t = 2. Nothing is compiled at FeMoco-54 shape:
there the closed form stands (spec 11.3).

``python tests/integration/test_qpe_counts.py PRESET PROGRAM`` compiles one program at ``sto3g``
or ``631g``: ``walk`` (the controlled walk) or ``qpe1``, ``qpe2``, ``qpe3`` (the QPE program with
1, 2, 3 bits), and prints its counts as JSON on a line prefixed ``RESULT``.
"""

import functools
import json
import os
import sys
import time
from pathlib import Path

import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import compiled_resources
from dfthc.ops.walk import ControlledDFTHCWalk
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST
from dfthc.qpe import qpe_program
from dfthc.registers import WalkLayout
from dfthc.testing import walk_sim
from dfthc.testing.resources import LEAF_ADJOINTS, TEMPLATE_ADJOINTS, captured_targets, inventory
from dfthc.testing.watchdog import effective_limit_mb, run_watched

REPO = Path(__file__).resolve().parents[2]
PRESETS = {"sto3g": H2_STO3G_TEST, "631g": H2_631G_TEST}
CAP_MB = 4096
"""Each compile's cap: the watchdog kills at 3,840 MB (master plan 12.9)."""
PREFIX = "RESULT "
CLIFFORDS = ("CZ", "PauliZ")
"""Recorded, not asserted, as expected totals: the deterministic CZ and Z of InnerSigns and of
the reflections plus the erase fix-ups' hinted conditionals, which qp.analyze counts at their
expectation (spec 7.6)."""


def build(name: str, program: str):
    """``(qfunc, num_wires)``: the controlled walk on walk_sim's case, or qpe_program."""
    c = walk_sim.case(name)
    if program == "walk":
        def qfunc():
            qp.PhaseGradientStatePrep(c.layout.phase_grad)
            ControlledDFTHCWalk(*c.compilable, **c.groups(controlled=True))

        return qfunc, c.num_wires
    return qpe_program(c.params, c.tables, int(program.removeprefix("qpe")))


def compile_counts(name: str, program: str) -> dict:
    """One compile (subprocess side): its counts, the expected CZ and PauliZ totals and the
    compile time."""
    qfunc, num_wires = build(name, program)
    start = time.perf_counter()
    resources = compiled_resources(qfunc, num_wires)
    return {"preset": name, "program": program,
            "seconds": round(time.perf_counter() - start, 1), "toffoli": resources.toffoli,
            "measurements": resources.measurements,
            "measured_and_uncomputes": resources.measured_and_uncomputes,
            "expected_clifford_totals": {gate: resources.counts.get(gate, 0)
                                         for gate in CLIFFORDS},
            "num_wires": num_wires}


@functools.cache
def watched(name: str, program: str) -> dict:
    """``compile_counts`` in a subprocess under run_watched (4 GB cap); prints the QPE_COMPILE
    line and returns the counts."""
    command = [sys.executable, str(Path(__file__).resolve()), name, program]
    env = os.environ | {"PYTHONPATH": str(REPO / "src"),
                        "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "0"}
    run = run_watched(command, cap_mb=CAP_MB, timeout_s=900, cwd=REPO, env=env)
    lines = [line for line in run.stdout.splitlines() if line.startswith(PREFIX)]
    result = json.loads(lines[0][len(PREFIX):]) if len(lines) == 1 else None
    print("QPE_COMPILE " + json.dumps({"watchdog": run.summary(), "result": result}))
    assert run.killed is None and not run.survivors and run.returncode == 0, run.stderr[-2000:]
    assert run.peak_mb <= effective_limit_mb(CAP_MB)
    assert result is not None, run.stdout[-2000:]
    return result


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.subprocess
@pytest.mark.parametrize("t", [1, 2, 3])
@pytest.mark.parametrize("name", PRESETS)
def test_qpe_counts(name, t):
    """Worst case, every unhinted correction at its maximum branch (spec 9.24 and 11.3).
    H2_STO3G_TEST (T_C(W) = 160, M_W = 50, A_C(W) = 110): T = 160, 481, 1,124; M = 51, 152,
    353; A = 110, 331, 774. H2_631G_TEST (242, 68, 181): T = 242, 727, 1,698; M = 69, 206, 479;
    A = 181, 544, 1,271. The program has exactly WalkLayout's wires with qpe_bits = t: the
    corrections' 2t - 1 wires are the work pool's (8 and 9 wires)."""
    p = PRESETS[name]
    walk, qpe = watched(name, "walk"), watched(name, f"qpe{t}")
    assert (walk["toffoli"], walk["measurements"], walk["measured_and_uncomputes"]) == (
        costs.walk(p, True).toffoli, costs.walk(p, False).measurements,
        costs.measured_and_uncomputes(p, True))
    corrections = (t**3 - t) // 6
    assert qpe["toffoli"] == (2**t - 1) * walk["toffoli"] + corrections
    assert qpe["measurements"] == (2**t - 1) * walk["measurements"] + t
    assert qpe["measured_and_uncomputes"] == (2**t - 1) * walk["measured_and_uncomputes"] \
        + corrections
    expected = costs.iterative_qpe(p, t)
    assert (qpe["toffoli"], qpe["measurements"]) == (expected.toffoli, expected.measurements)
    assert qpe["num_wires"] == WalkLayout.from_params(p, qpe_bits=t).num_wires


@pytest.mark.compiled
@pytest.mark.slow
def test_inventory_of_the_qpe_program():
    """The inventory test of Task 3.4 (spec 7.4 rule 6, 12.1 T6) for the QPE program at
    H2_STO3G_TEST with t = 2: the frontend converts both rounds, both walk powers, the controlled
    walk with its explicit controlled children and the correction, no C(...) target at all, and
    the modified dfthc and template targets are exactly the walk's leaf and template adjoints."""
    c = walk_sim.case("sto3g")
    program, num_wires = qpe_program(c.params, c.tables, 2)
    names = captured_targets(program, num_wires)
    assert {"QPERound", "ControlledWalkPower", "ControlledDFTHCWalk",
            "ControlledDFTHCBlockEncoding", "ControlledRefT2", "ControlledRefAB",
            "PhaseGradientRotation"} <= set(names)
    assert names.count("QPERound") == 2 and names.count("ControlledWalkPower") == 2
    assert not [name for name in names if "C(" in name]
    assert inventory(names) == LEAF_ADJOINTS | TEMPLATE_ADJOINTS


def test_qpe_closed_forms_at_femoco():
    """Spec 11.3: no compile at FeMoco-54 shape; the closed form with T_C(W) = 9,795 and
    M_W = 2,524 (Task 4.5 compiles those two) gives T = 9,795, 29,386, 68,569 and
    M = 2,525, 7,574, 17,671 for t = 1, 2, 3."""
    assert [costs.iterative_qpe(FEMOCO54, t) for t in (1, 2, 3)] == [
        costs.Cost(9795, 2525), costs.Cost(29386, 7574), costs.Cost(68569, 17671)]


if __name__ == "__main__":
    print(PREFIX + json.dumps(compile_counts(sys.argv[1], sys.argv[2])), flush=True)
```

- [ ] **Step 2: Break the code under test: hint the corrections**

The code under test exists (Task 4.1), so the new assertions would pass at once; their first run must show that they can fail. Hint the QPE corrections, which turns their worst-case Toffoli count into its expectation, with this narrow patch of `src/dfthc/qpe.py` (run from the worktree's root; the blank line before the conditional is removed and added back, so no context line of the patch is blank):

```bash
git apply <<'EOF'
--- a/src/dfthc/qpe.py
+++ b/src/dfthc/qpe.py
@@ -401,4 +401,4 @@
                                   corrections[d:2 * d - 1])
-
-        qp.cond(earlier_bits[later], correction)()
+
+        qp.cond(earlier_bits[later], qp.hint({"branch-prob": 0.5})(correction))()
     qp.H(control)
EOF
```

Run: `git status --short`
Expected: ` M src/dfthc/qpe.py` and `?? tests/integration/test_qpe_counts.py` (the dry run, in a replica without git, checked that the patch changed `qpe.py`).

- [ ] **Step 3: Run the t = 3 count cases to see them fail**

Run: `scripts/test.sh "tests/integration/test_qpe_counts.py::test_qpe_counts[sto3g-3]" "tests/integration/test_qpe_counts.py::test_qpe_counts[631g-3]" -q`
Expected: FAIL, `2 failed`: `test_qpe_counts[sto3g-3]` and `test_qpe_counts[631g-3]`, at the T comparison (`assert 1122 == ...` and `assert 1696 == ...`, against 1,124 and 1,698): hinted, the three corrections' four Toffolis count 2. At t = 2 the one correction's single Toffoli counts 0.5, and `qp.analyze` rounds the program's Toffoli total up once (master plan section 3, item 6), so t = 1 and t = 2 would still pass; the worst case the test pins differs from the expectation from t = 3 on, which is why these two cases are the RED run (dry run 23.0 s with 2 workers).

- [ ] **Step 4: Restore `qpe.py`**

Restore it with the reverse of the same patch:

```bash
git apply -R <<'EOF'
--- a/src/dfthc/qpe.py
+++ b/src/dfthc/qpe.py
@@ -401,4 +401,4 @@
                                   corrections[d:2 * d - 1])
-
-        qp.cond(earlier_bits[later], correction)()
+
+        qp.cond(earlier_bits[later], qp.hint({"branch-prob": 0.5})(correction))()
     qp.H(control)
EOF
```

Run: `git status --short`
Expected: only `?? tests/integration/test_qpe_counts.py` (the dry run, in a replica without git, compared the restored `qpe.py` byte for byte with the original).

- [ ] **Step 5: Run the unmodified file**

Run: `scripts/test.sh tests/integration/test_qpe_counts.py -q`
Expected: `8 passed` (dry run 59.5 s with 2 workers: eight compiles in subprocesses and the inventory compile).

- [ ] **Step 6: Record the compiles**

Run: `DFTHC_TEST_WORKERS=0 scripts/test.sh tests/integration/test_qpe_counts.py -m slow -q -s`
Expected: `7 passed, 1 deselected`, after eight `QPE_COMPILE {...}` lines, one per compile (the controlled walk and t = 1, 2, 3 at each preset; most follow a test's progress dot, `.QPE_COMPILE {...}`).

Copy the eight `QPE_COMPILE` lines into the report. Each holds the compile's `seconds`, its counts, `expected_clifford_totals` and `num_wires` in `result`, and the subprocess's wall time and peak in `watchdog`. Dry run: every compile 3.8 to 4.7 s, every subprocess 6.8 to 7.9 s below 380 MB (`stages/4Q/qpe-compiles.log`):

| Program | `H2_STO3G_TEST` T / M / A, expected `CZ` / `PauliZ` totals, wires, compile, peak | `H2_631G_TEST` |
| --- | --- | --- |
| controlled walk | 160 / 50 / 110, 16 / 6, 64, 3.8 s, 348 MB | 242 / 68 / 181, 26 / 8, 80, 4.2 s, 355 MB |
| QPE t = 1 | 160 / 51 / 110, 16 / 6, 65, 4.2 s, 355 MB | 242 / 69 / 181, 26 / 8, 81, 4.5 s, 355 MB |
| QPE t = 2 | 481 / 152 / 331, 48 / 18, 66, 4.2 s, 357 MB | 727 / 206 / 544, 78 / 24, 82, 4.6 s, 371 MB |
| QPE t = 3 | 1,124 / 353 / 774, 112 / 42, 67, 4.5 s, 364 MB | 1,698 / 479 / 1,271, 182 / 56, 83, 4.7 s, 379 MB |

- [ ] **Step 7: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1108 passed` (1107 + 1).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 8: Commit**

```bash
git add tests/integration/test_qpe_counts.py
git commit -m "Add the QPE program's compiled counts at the test presets"
```

- [ ] **Step 9: Report**

In the report give the RED output of Step 3, the eight `QPE_COMPILE` lines of Step 6, and the per-commit tier's time and peak from Step 7.

---

## Interfaces for Tasks 4.3 to 4.5

What the resource report (Task 4.3: the L4 rows `ControlledWalkPower` with p = 1 and 2 and the QPE program with t = 1, 2, 3, at the H2 systems; spec 13.3), the documentation (4.4) and the FeMoco finalization (4.5) consume from this file, exactly as built.

**Building the programs at a preset.** `stages/4Q/interfaces_probe_4q.py` below is the interface as executed in the dry run: at `H2_STO3G_TEST` it builds `ControlledWalkPower` with p = 1 and 2 on `WalkLayout.from_params(p)`'s groups with one control wire after them (64 wires, asserted equal to `sum(ControlledWalkPower.required_wires(...).values())`) and the QPE program with `qpe_program` (its wire count asserted equal to the sum of `QPERound.required_wires`), compiles each with `compiled_resources`, and checks T, M and A; at FeMoco-54 it prints the closed forms and wire counts without compiling.

```python
"""Interfaces for Tasks 4.3 to 4.5: the L4 programs of the resource report, built at a preset.

Executed by the phase4-qpe plan's dry run, from the replica root with PYTHONPATH=src, under the
watchdog. At H2_STO3G_TEST it compiles ControlledWalkPower with p = 1 and 2 (on WalkLayout's
groups plus one control wire) and the QPE program with t = 1, 2, 3 (dfthc.qpe.qpe_program),
checks T, M and A against the closed forms, and checks each wire count against the operators'
required_wires; at FeMoco-54 it prints the closed forms and the wire counts without compiling
(master plan 12.9).
"""

import time

import pennylane as qp

from dfthc import costs
from dfthc.compile import compiled_resources
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.params import FEMOCO54, H2_STO3G_TEST
from dfthc.qpe import ControlledWalkPower, QPERound, qpe_program, spare_wires_needed
from dfthc.registers import WalkLayout


def power_program(p, t, power):
    """ControlledWalkPower on WalkLayout's groups and one control wire after them."""
    layout = WalkLayout.from_params(p)
    groups = {name: getattr(layout, name) for name in ControlledWalkPower.wire_argnames[1:]}

    def program():
        qp.PhaseGradientStatePrep(layout.phase_grad)
        ControlledWalkPower(power, p.as_compilable(), t.as_compilable(), [layout.num_wires],
                            **groups)

    return program, layout.num_wires + 1


def qpe_a(p, t):
    """A of the t-bit QPE program, worst case: every correction's d - 1 measured uncomputes."""
    return (2**t - 1) * costs.measured_and_uncomputes(p, True) + (t**3 - t) // 6


def qpe_wires(p, t):
    """The QPE program's wires without building it: WalkLayout's with qpe_bits = t, then the
    spare wires the corrections need beyond the work pool."""
    layout = WalkLayout.from_params(p)
    sizes = tuple(len(getattr(layout, name)) for name in ControlledWalkPower.wire_argnames[1:])
    return (WalkLayout.from_params(p, qpe_bits=t).num_wires
            + spare_wires_needed("ControlledWalkPower", sizes, t))


def main():
    p = H2_STO3G_TEST
    t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
    compilable = (p.as_compilable(), t.as_compilable())
    for power in (1, 2):
        program, num_wires = power_program(p, t, power)
        assert num_wires == sum(ControlledWalkPower.required_wires(power, *compilable).values())
        start = time.perf_counter()
        r = compiled_resources(program, num_wires)
        expected = costs.controlled_walk_power(p, power)
        a = power * costs.measured_and_uncomputes(p, True)
        print(f"H2_STO3G_TEST ControlledWalkPower p={power}: T, M, A = {r.toffoli}, "
              f"{r.measurements}, {r.measured_and_uncomputes}; closed forms {expected.toffoli}, "
              f"{expected.measurements}, {a}; {num_wires} wires (its required_wires); "
              f"{time.perf_counter() - start:.1f} s")
        assert (r.toffoli, r.measurements, r.measured_and_uncomputes) == (
            expected.toffoli, expected.measurements, a)
    sizes = tuple(ControlledWalkPower.required_wires(1, *compilable).values())[1:]
    for bits in (1, 2, 3):
        program, num_wires = qpe_program(p, t, bits)
        round_wires = QPERound.required_wires(0, bits, "ControlledWalkPower", compilable, sizes)
        assert num_wires == sum(round_wires.values()) == qpe_wires(p, bits)
        start = time.perf_counter()
        r = compiled_resources(program, num_wires)
        expected = costs.iterative_qpe(p, bits)
        print(f"H2_STO3G_TEST QPE t={bits}: T, M, A = {r.toffoli}, {r.measurements}, "
              f"{r.measured_and_uncomputes}; closed forms {expected.toffoli}, "
              f"{expected.measurements}, {qpe_a(p, bits)}; {num_wires} wires (QPERound's "
              f"required_wires); {time.perf_counter() - start:.1f} s")
        assert (r.toffoli, r.measurements, r.measured_and_uncomputes) == (
            expected.toffoli, expected.measurements, qpe_a(p, bits))
    f = FEMOCO54
    print("FeMoco-54 (closed forms, not compiled): T_C(W), M_W, A_C(W) =",
          costs.walk(f, True).toffoli, costs.walk(f, False).measurements,
          costs.measured_and_uncomputes(f, True), "; work pool", WalkLayout.work_size(f))
    for power in (1, 2):
        c = costs.controlled_walk_power(f, power)
        print(f"  ControlledWalkPower p={power}: T, M, A = {c.toffoli}, {c.measurements}, "
              f"{power * costs.measured_and_uncomputes(f, True)}; wires "
              f"{WalkLayout.from_params(f).num_wires + 1}")
    for bits in (1, 2, 3):
        c = costs.iterative_qpe(f, bits)
        print(f"  QPE t={bits}: T, M, A = {c.toffoli}, {c.measurements}, {qpe_a(f, bits)}; wires "
              f"{qpe_wires(f, bits)}")
    print("  QPE wires, t = 20, 22, 23, 25:", [qpe_wires(f, bits) for bits in (20, 22, 23, 25)])


if __name__ == "__main__":
    main()
```

Its output (`stages/4Q/interfaces-probe-4q.out`):

````text
H2_STO3G_TEST ControlledWalkPower p=1: T, M, A = 160, 50, 110; closed forms 160, 50, 110; 64 wires (its required_wires); 4.7 s
H2_STO3G_TEST ControlledWalkPower p=2: T, M, A = 320, 100, 220; closed forms 320, 100, 220; 64 wires (its required_wires); 3.3 s
H2_STO3G_TEST QPE t=1: T, M, A = 160, 51, 110; closed forms 160, 51, 110; 65 wires (QPERound's required_wires); 3.3 s
H2_STO3G_TEST QPE t=2: T, M, A = 481, 152, 331; closed forms 481, 152, 331; 66 wires (QPERound's required_wires); 3.3 s
H2_STO3G_TEST QPE t=3: T, M, A = 1124, 353, 774; closed forms 1124, 353, 774; 67 wires (QPERound's required_wires); 3.6 s
FeMoco-54 (closed forms, not compiled): T_C(W), M_W, A_C(W) = 9795 2524 8677 ; work pool 44
  ControlledWalkPower p=1: T, M, A = 9795, 2524, 8677; wires 1565
  ControlledWalkPower p=2: T, M, A = 19590, 5048, 17354; wires 1565
  QPE t=1: T, M, A = 9795, 2525, 8677; wires 1566
  QPE t=2: T, M, A = 29386, 7574, 26032; wires 1567
  QPE t=3: T, M, A = 68569, 17671, 60743; wires 1568
  QPE wires, t = 20, 22, 23, 25: [1585, 1587, 1589, 1595]
WATCHDOG {"command": ["/Users/david.wierichs/repos/dfthc-be/.venv/bin/python", "../stages/4Q/interfaces_probe_4q.py"], "pid": 36140, "returncode": 0, "seconds": 20.8, "peak_mb": 364.7, "samples": 393, "killed": null, "survivors": [], "cap_mb": 3000.0, "timeout_s": 1200.0, "priority": "taskpolicy -c utility", "heavy": false}
````

**What Tasks 4.3 to 4.5 must respect.**
- **The QPE program** is `qpe_program(params, tables, num_bits)`, which returns `(qfunc, num_wires)`: the walk's phase gradient, then `iterative_phase_estimation` of `ControlledWalkPower` on `WalkLayout.from_params(params, qpe_bits=num_bits)`'s groups, `qpe_control` and `qpe_grad`; the corrections borrow the walk's `work` pool. Pass `num_wires` to `compiled_resources` as it is. Its qubits: `WalkLayout.from_params(p, qpe_bits=t).num_wires + max(0, 2t − 1 − WalkLayout.work_size(p))` (the probe's `qpe_wires`), that is spec 10.3's count plus t + 1 up to t = 22 at FeMoco-54: 1,566, 1,567, 1,568 for t = 1, 2, 3 and 1,585 at t = 20; from t = 23 on, 2t − 45 spare wires more (1,589 at t = 23). At the test presets 64 + t and 80 + t (no spare wire up to t = 4 and 5).
- **`ControlledWalkPower`** takes `(power, params, tables, control, outer, inner, select, system, phase_grad, work)` with `params = WalkParams.as_compilable()`, `tables = WalkTables.as_compilable()` and `WalkLayout.from_params(p)`'s groups as they are, and one control wire after them: the probe's `power_program`. Its qubits are `sum(ControlledWalkPower.required_wires(power, params, tables).values())` = `WalkLayout.from_params(p).num_wires + 1`: 64 at `H2_STO3G_TEST`, 80 at `H2_631G_TEST`, 427 at `MID_NAPHTHALENE` and 1,565 at FeMoco-54, for every power.
- **Closed forms** (all in `dfthc.costs`, T and M): `controlled_walk_power(p, power)` (A = power · `measured_and_uncomputes(p, True)`), `qpe_round(p, j, t)`, `iterative_qpe(p, t)` (A = (2^t − 1) · `measured_and_uncomputes(p, True)` + (t³ − t)/6, the corrections' measured uncomputes, which the probe's `qpe_a` computes; there is no `costs` function for it). At FeMoco-54 the report evaluates `iterative_qpe` with Task 4.5's compiled T_C(W) and M_W (spec 11.3), also for t > 3. Every compiled count must equal its closed form (Task 4.2 checks it at both test presets).
- **Clifford counts.** In the QPE program, `CZ` and `PauliZ` are expected totals (deterministic gates plus the hinted erase fix-ups at their expectation, spec 7.6; Task 4.2's `expected_clifford_totals`); `PauliX` and `CNOT` are upper bounds (the resets' and the reads' unhinted conditionals, and the corrections' fan-out under the unhinted `qp.cond`). Label them as spec 13.3 says.
- **No FeMoco-shaped compile** of the QPE program or of `ControlledWalkPower` (spec 11.3; master plan 12.9): only the walk and the controlled walk compile at FeMoco-54, once, in Task 4.5.
- **Compile cost.** Each QPE compile at the test presets costs about the controlled walk's compile (3.8 to 4.7 s, below 380 MB): the loops compile their body once, and `ControlledDFTHCWalk` with identical arguments is one rule variant in every round. The report's `naphthalene` row of `ControlledWalkPower` (p = 1, 2) costs about the `MID_NAPHTHALENE` controlled walk's compile (8 s, 0.6 GB in the assembly plan); `test_qpe.py`'s nightly p = 2 compile there took 19 s with 2 workers.
- **Docstrings (Task 4.4).** The five classes have the nine spec 9.0 sections in order; `ControlledWalkPower`'s and `QPERound`'s cost sections name `dfthc.costs.controlled_walk_power` and `dfthc.costs.qpe_round`; the toy classes' name the toy module's `toy_walk_cost` and `toy_walk_power_cost` (they live in `dfthc.testing`, outside the audit's `dfthc.ops`).
