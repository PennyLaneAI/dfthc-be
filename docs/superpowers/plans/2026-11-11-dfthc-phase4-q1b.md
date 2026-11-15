# DFTHC Phase 4, Task Q1b: Early Junk Measurement (Q-1 Variant (b)) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spec 16, Q-1 variant (b): the block encodings X-measure each read's junk register right after its prepare and pass the outcomes to two new operators, `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`, which measure only the read's target and fix the phases from both outcome vectors; `WalkLayout` then gives the junk registers' wires to the angle registers. T, A and the total M stay as they are (9,793 and 9,795; 8,675 and 8,677; 2,524), and the FeMoco-54 walk needs 1,102 qubits instead of 1,564 (the paper: 1,131). The report and the documents (D-3, D-6) are brought up to date before Task 4.5.

**Architecture:** `dfthc.ops.qroam.emit_erase_from_outcomes` is `QROAMErase`'s body with the junk outcomes given: it X-measures the target and runs the same fix-up on the outcomes target ‖ junk, its masks built in the rule from the compilable table. `dfthc.ops.prepare` adds the two unprepares from outcomes (the prepare's wires except the junk, then `extra_onehot`; the prepare's compilable arguments; the dynamic `junk_outcomes`), their pinned rules and `abstract_junk_outcomes`, the mode-dependent declaration of an outcome array. `dfthc.registers.WalkLayout` drops the junk registers from the `outer` and `inner` groups and makes each the first wires of `angles ‖ junk_overflow`, the start of `select`. The block encodings' rules in `dfthc.ops.walk` measure the junk after each prepare, keep the outcomes, and call the unprepares from outcomes, lending each its `extra_onehot` from the start of the measured junk wires. The report gains the new rows and the qubit comparison by group.

**Tech Stack:** Python 3.12, PennyLane `Operator2` and Catalyst (`qp.qjit(capture=True)`, `qp.decompose` with the graph and `FIXED_DECOMPS` pins, `qp.analyze` through `dfthc.compile.compiled_resources`), mid-circuit measurements (`qp.measure(reset=True)`, `MeasurementValue`, `qp.cond`), NumPy, the sparse simulator `dfthc.testing.walk_sim`, `lightning.qubit`; pytest with pytest-xdist; ruff.

This file holds the conditional Task Q1b of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 10), as two sub-tasks under the id Q1b: **Q1b.1** (the unprepares from outcomes, unused by the walk) and then **Q1b.2** (the walk measures the junk early; the layout, the report and the documents follow). It was written at the Q-1 working choice (ledger decision "Q-1 working choice: variant (b)", 2026-11-15), which the user may override before Q1b runs; it starts after Task 4.4 and comes before Task 4.5 (wave 4, master plan section 5). The report file (`2026-11-11-dfthc-phase4-report.md`, revision 7) and the QPE file (revision 4) are taken as final.

**Revision 1** (2026-11-15). First version.

**Revision 2** (2026-11-15), after the GPT-5.6 review `.superpowers/sdd/q1b-plan-review-1.md` (Needs fixes: 0 Critical, 2 Important, 1 Minor; the design, the aliasing, the outcome order, both modes, the counts and the projections verified correct); revision 1's assembled plan is kept as `.scratch/plan-check/rev1-q1b/2026-11-11-dfthc-phase4-q1b.md`.

- **I-1, the master-template mirroring was not executable: fixed.** Section "Shared edits" now gives the orchestrator's procedure at approval: apply the document edits to `docs/`, apply B-M1 to B-M14 and B-M18 to the master-plan template, copy this plan's Global Constraints fragment to `templates/_global_constraints_q1b.md`, point only the master-plan template's include and this plan's include at it (the exact line changes are in the section), assemble, and assert that the assembled master plan equals the edited `docs/` master plan. Revision 1's sentence "No existing plan template changes" is gone. The procedure was replayed on copies of the templates (`stages/Q1B/verify_mirroring_q1b.py`, `mirroring-check.log`): the assembled master plan equals `main`'s with all 18 B-M edits, this plan equals its assembly in `out-q1b/`, and the other 12 plans equal `main`.
- **I-2, the prepare and read contracts forbade the new junk lifetime: fixed in the code blocks.** The docstrings now allow both paths, (a) the junk kept until the baseline unprepare or `QROAMErase`, or (b) X-measured with reset right after the prepare or read, its outcomes kept in wire order (`QROAMRead.junk_layout`) and given to the matching unprepare from outcomes or `emit_erase_from_outcomes`: `OuterPrepare` and `InnerPrepare` (pre- and postconditions, the inverse), `QROAMRead` (summary, pre- and postconditions, adjoint and control, paper), `QROAMErase` (pre- and postconditions), the module docstrings of `dfthc.ops.qroam` (the erase's outcomes) and `dfthc.ops.prepare` (the forced order, and "no walk instance needs `extra_onehot`", now false), and `docs/architecture.md` (the emitters of the fix-up). A search of every docstring and comment in `src/dfthc/` and of the documents this plan touches found no other statement of the old lifetime. Nine spec edits join (B-S30 to B-S38: section 3.5's steps 19–22 and 24–27 and its two inner rounds, 4.2, 4.3's decision, 9.3's introduction, 9.4's argument that no operator boundary carries measurement data, and 9.5's wires and precondition rows), so the plan has 61 edits. Docstrings only: the code hash stays `5e3eed1e7a95...`, so the regenerated report stays current. Replayed on `main` at `9f4c7eb` (`stages/Q1B/run_rev2_q1b.sh`, `q1b-rev2.log`): Q1b.1's RED and GREEN unchanged, the tests of the touched modules `266 passed` on both trees (section "Expected test counts").
- **M-1, the report plan still states Task 4.5's pre-Q1b count: fixed by a dispatch rule.** Section "Expected test counts" states that the orchestrator revises the report plan's Task 4.5 counts at Q1b's merge (report plan revision 8: base 1221 → 1268, expected 1222 → 1269) and that Task 4.5's dispatch states the Q1b counts.
- **Also redone:** `main` moved to `9f4c7eb` (a ledger edit since `1a3e062`): all 61 edits apply exactly once there. In the mirroring replay the 12 other plans equal `main` byte for byte; separately, the assembled master plan equals the edited `docs/` master plan, and this plan equals its `out-q1b/` assembly.

**Revision 3** (2026-11-15), after the GPT-5.6 re-check `.superpowers/sdd/q1b-plan-review-2.md` (Needs fixes: 0 Critical, 1 Important, 1 Minor; the mirroring, the counts, the replay equality, the code hash and the prepare contracts verified).

- **I-1, the spec and the read's docstring still misstated the walk's erase path and measurement point: fixed.** New edit B-S39: spec 9.3's adjoint-and-control row no longer ends "The walk erases with `QROAMErase`." but names the three erase paths (`QROAMErase` for a read whose junk is kept, the unprepares from outcomes through `emit_erase_from_outcomes` for the walk's alias reads, `AngleErase` for the angle reads). The generic path (a caller may X-measure the junk right after the read) is now kept apart from this walk's, which X-measures it right after the enclosing prepare (steps 4 and 8), before any angle load; the comparator and the swaps in between do not touch the junk, so the two placements are equivalent. Changed: B-S34 (4.3), B-S35 (9.3's introduction), B-S38 (9.5's precondition row) and B-S33 (4.2; the paper measures "early, in the prepare", not claimed to be right after the read). New edit B-S40: spec 16's Q-1 "What it is" said "right after its swap network". In the code (Q1b.1's snapshot): `QROAMRead`'s summary, path (b) of its pre- and postconditions (after the read and before anything else acts on the junk) and its paper paragraph; `QROAMErase`'s path (b) pointer; the paper sections of `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`, which said the junk is "measured at step 2" (and step 6). A new search of `src/dfthc/`, the documents this plan touches and the edited spec, master plan and HANDOFF for where or when the walk measures or erases the junk found every other statement already saying "right after its prepare". The plan now has 63 edits (40 to the spec). Docstrings only: the code hash stays `5e3eed1e7a95...`. Replayed again (`run_rev2_q1b.sh`, `q1b-rev3.log`), with the same counts as revision 2.
- **M-1, the Revision 2 note overcounted the plans equal to `main`: fixed** in its last bullet.

**Revision 4** (2026-11-15, by the orchestrator, after Task Q1b.1's reviews: GPT-5.6 `.superpowers/sdd/task-Q1b.1-review-1.md`, Approved, 0 Critical, 0 Important, 0 Minor, 10 of 10 mutants killed; Opus `.superpowers/sdd/task-Q1b.1-review-opus.md`, Approved, 0 Critical, 0 Important, 2 Minor, 29 of 29 mutants killed per commit). m-1: the compiled path of the junk outcomes (the capture branch of `_junk_outcomes`) had only sampled checks, so a reversed or stale outcome array died only in the two seeded lightning tests (missed with probability about 1/64 and 1/4096). The test file now has `test_lightning_round_trip_with_postselected_junk[outer-sto3g, inner-small]`, which postselects the junk on 1, 0, ..., 0 and kills both for every seed; `measure_x` and `round_trip` take an optional `postselect`, and the replica `stages/Q1B/q1b1` holds the reviewed file. m-2: the acceptance list named the no-adjoint test's cases `[outer, inner]`; they are `[adjoint, control]` on the inner class (the outer class fails the same way, by the review's probe). No hashed file changed. Q1b.1 has 44 per-commit tests, so every expected per-commit count from Q1b.1 on rises by 2: 1265 after Q1b.1, 1270 after Q1b.2, 1271 for Task 4.5 (report plan revision 8 at Q1b's merge); the nightly (86) and release (34) counts are unchanged. Measured on the rebased Q1b.1 branch with the fix: Step 7's targeted run `199 passed in 50.28s`, the per-commit tier `1265 passed in 432.42s (435 s, 2,038 MB)`, no warnings summary. The master plan's two Q1b count sentences change accordingly.

**Revision 5** (2026-11-15, by the orchestrator, after Task Q1b.2's reviews: GPT-5.6 `.superpowers/sdd/task-Q1b.2-review-1.md`, Approved, 0 Critical, 0 Important, 0 Minor, 10 of 10 mutants killed; Opus `.superpowers/sdd/task-Q1b.2-review-opus.md`, Approved, 0 Critical, 0 Important, 3 Minor, 26 of 26 mutants killed per commit, the walk equal to the baseline walk state for state at `H2_STO3G_TEST`, `H2_631G_TEST` and `FAILS_C3`). m-1: the prepares' FeMoco tests (`tests/ops/test_outer_prepare.py`, `tests/ops/test_inner_prepare.py`, outside this plan's blocks) named 143 and 455 wires as spec 10.3's groups; their docstrings and assertions now give the groups (71, 65) and the junk aliases (72, 390) from `WalkLayout` (tests only, no count change, code hash unchanged). m-2 (deferred to the next change of `scripts/resource_report.py`, which is hashed): the report's sentence counts the overlapping junk registers' widths, 462, as wires that share the angles; they lie on the first 390 angle wires. m-3: section "Interfaces for Task 4.5" still gave revision 3's count; it now gives 1271 = 1270 + 1. The GPT-5.6 review's Focus 7 found that Task 4.5's release test `test_femoco_cache` sums only the partition rows, whose M at FeMoco-54 is 1,672 under Q1b; the block encodings' own 852 junk measurements make 2,524, as `resource_report.cross_check` adds them. The report plan's revision 8 adds them to that test; the interface below says so.

**Code provenance.** Every code block below was executed on 2026-11-15 in a scratch replica of the repository (`.scratch/plan-check/repo-phase4b`, not committed): `git archive` of local `main` at `9f4c7eb` (Task 4.4 merged at `ce5cc0b` with its review fixes, report file revision 7; since `1a3e062` only the ledger changed), plus this plan's files. The dry run began on `main` at `c157211` with Task 4.4's six files from the report file's revision 6 (`stages/Q1B/overlay_task44.py`); when Task 4.4 merged with the review's changes to `tests/test_docstrings.py` and four documents (nothing hashed, nothing under `src/`), this plan's edits of `tests/test_docstrings.py`, `docs/architecture.md`, `docs/conventions.md` and `docs/verification.md` were ported onto the merged files by exact replacements (`stages/Q1B/port_task44_files.py`, each old text once), and the replica was rebuilt on `1a3e062`; every other file is unchanged by the rebase. The replica is not a git repository, so the dry run ran pytest the way `scripts/test.sh` does (the same environment, 2 workers, `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0`, lowered priority) through `.scratch/plan-check/p4btest.sh`, which runs every session under the watchdog with a 3,000 MB cap and a session lock; every script ran through `p4brun.sh` (same cap, same lock). Both sub-tasks were replayed in order, RED then GREEN, in a fresh copy of the base: from `c157211` with the per-commit tier after each sub-task (`stages/Q1B/run_stages_q1b.sh`, log `q1b-stages.log`), and again from `1a3e062` with every step that runs `tests/test_docstrings.py`, that test on the Q1b.2 tree, ruff and each tier's collection count (`run_rebase_q1b.sh`, log `q1b-rebase.log`); the report plan's revision 7 rule (a tree that differs from a tested one only in `docs/` carries the tier result after `tests/test_docstrings.py` passes on it) covers the rest, since the two trees differ only in `docs/` and that test. Revision 2's docstring changes were replayed from `9f4c7eb` (`run_rev2_q1b.sh`, log `q1b-rev2.log`): Q1b.1's RED and GREEN steps, `tests/test_docstrings.py` and the tests of the modules that cover `dfthc.ops.qroam` and `dfthc.ops.prepare` on both trees, the code hash, ruff and each tier's collection count; revision 3's by the same script (log `q1b-rev3.log`). All four replays end byte-identical to the replica of their time, the last to the final one. Every "Expected" line below is a replay's output; the step says which. Further evidence under `stages/Q1B/`: the layout sizes at every preset and system (`probe_layout_sizes.py`, `probe-layout-sizes.log`), the operators' first compiles and T2 checks, among them a caller inside another rule (`probe_from_outcomes.py`, `probe-from-outcomes.log`), the report's rows at every system (`probe_report_rows.py`, `probe-report-rows.log`), the regenerated report of the H2 systems and `naphthalene` (`report-small-systems.log`, summarized by `report_summary_q1b.py` in `report-summary.md`), back-to-back compiles against the baseline code (`compare_compiles.py`, `compare-compiles.raw`), the projections (`project_large_q1b.py`, four `projections-*.log`), seven mutants (`mutate_q1b.py`, `q1b-mutants.log`), the nightly tier (`q1b-nightly-tier.raw`), three release-tier inputs (`release-sample.log`), Q1b.2's RED failures by name (`q1b2-red-detail.log`), Task 4.5's per-commit test on the Q1b tree (`task45-test-on-q1b.log`) and the document edits (`make_doc_edits_q1b.py`, `doc-edits.md`).

**Load.** At most 2 pytest workers ran at any time and one process that compiled or ran pytest at a time (the shared lock), each under the watchdog with a 3,000 MB cap. The per-commit tier ran twice (once per sub-task, in the first replay) and the report regeneration once, never beside a pytest session; the largest process tree was the Q1b.2 per-commit tier, 1,975 MB, and the largest single compile the `MID_NAPHTHALENE` walk in the compile comparison, 630.9 MB. Nothing above `naphthalene` was compiled: `fe2s2`, `fe4s4` and FeMoco-54 were built in tape mode only (their rows and gates). Revisions 2 and 3 ran targeted sessions only (the longest 50 s, at most 1,827 MB), no per-commit tier and no report. No load-limit event occurred.

---

## Why two sub-tasks

Q1b changes four layers at once: two operators, the walk's rule, the layout, and the report with its tests and documents. Q1b.1 adds the operators and their tests without touching the walk: it is a complete review unit (the operators' semantics, the order of the junk outcomes, a caller inside another rule, the certificates, the docstrings), every tier stays green, and a reviewer of Q1b.2 can take the operators as given. Q1b.2 cannot be split further without a broken or wasted intermediate commit: the layout's aliasing of the junk with the angles is wrong while the walk still keeps the junk to the erase (the angle loads would overwrite live junk), and switching the walk without the layout would regenerate the report (11 minutes plus the orchestrator's two `--large` reruns) for an intermediate state whose qubit count nobody uses. Each sub-task is one commit, has its own review, and Q1b is complete when both are merged; the ledger keeps the umbrella row Q1b and adds rows Q1b.1 and Q1b.2.

## Changes from the master plan, the spec and HANDOFF

The changes that touch a document are applied by the exact edits listed after this list (ids B-M for the master plan, B-S for the spec, B-H for HANDOFF).

1. **Two sub-tasks** (B-M6, B-M7, B-M10). Section "Why two sub-tasks". The master plan's Task Q1b entry is rewritten with both scopes, their counts and the definition of done; the overview and wave rows name them.
2. **`extra_onehot`, a register spec 16 does not list** (B-S24, B-S15, B-M5). Spec 16 gives the unprepares from outcomes "the prepare's wires except junk". The fix-up's one-hot register needs 2^ob reset wires (spec 9.5), and the baseline erase finds them in the read's reset target ‖ junk; without the junk, the target alone is too small where 2^ob exceeds the entry width d: at FeMoco-54 the inner fix-up has ob = 7 (128 wires) and the inner target 26 wires, so 102 more are needed. That is forced, not chosen: `_check_onehot` guarantees only that 2^ob fits target ‖ junk. So each unprepare from outcomes takes `extra_onehot`, max(2^ob − d, 0) wires in |0⟩, and the block encoding lends them from the start of the measured junk wires, which are |0⟩ again at the unprepare (the select's angle erase has run). Values: outer 0 everywhere; inner 1 at `H2_STO3G_TEST` and `H2_631G_TEST` (so the lending path runs per commit), 0 at `MID_NAPHTHALENE`, 102 at FeMoco-54. Ledger row 1C.5's note "`extra_onehot` empty (0 wires at every preset)" was about `QROAMErase`'s register in the walk, which the walk no longer calls.
3. **The junk sits on `angles ‖ junk_overflow`** (B-S14, B-S15, B-S16, B-M4, B-M18, B-H5). Spec 16 says "`WalkLayout` then lets the angle registers reuse the freed junk wires". Each junk register is the first wires of `angles`, followed by a new register `junk_overflow` of max(junk sizes, angles) − angles wires, at the start of `select`: 0 wires at every vendored system and FeMoco-54, 3 at `H2_STO3G_TEST`, whose 7 inner junk wires exceed its 4 angle wires. Aliasing onto the whole `select` group instead would save those 3 wires at `H2_STO3G_TEST` and nothing elsewhere (`probe-layout-sizes.log`); the plan keeps the spec's wording. The walk's groups: `outer` 143 → 71 and `inner` 455 → 65 at FeMoco-54, 16 → 12 and 23 → 16 at `H2_STO3G_TEST` (`select` 8 → 11 there).
4. **The walk calls no `QROAMErase`** (B-M9, B-S20). The baseline unprepares, which measure target and junk together through `QROAMErase`, stay in the package with their tests (they are spec 9.7 and 9.8's operators and the reference the certificates compare the measurement split against); the walk calls the unprepares from outcomes instead. The report's rows follow the walk's calls, so the two `QROAMErase@...` rows go and the unprepares from outcomes and their adjoint uniform preparations come in: 46 rows at `H2_STO3G_TEST` and at each H2 system (was 48), 44 at `naphthalene`, `fe2s2` and `fe4s4` (was 46), 47 at FeMoco-54 (was 49). The `PhaseFixup@...` rows take the unprepares' erase arguments.
5. **M is attributed differently, not changed** (B-S11, B-S12, B-S13, B-S19, B-S20, B-M5). Each unprepare from outcomes measures its target only (`costs.outer_unprepare_from_outcomes`, `costs.inner_unprepare_from_outcomes`: the baseline unprepares' T, M = d1 and d2), and the block encoding's rule measures the junk (`costs.junk_measurements`: (K1 − 1) d1 + 2 (K2 − 1) d2, 852 = 72 + 2 · 390 at FeMoco-54). The partition rows times their calls plus the junk measurements give the walk's M; the report's cross-check adds them. T and A are those of the baseline unprepares (the same CSWAPs, comparator adjoint, fix-up and adjoint uniform preparation).
6. **The qubit comparison by group** (B-S16, B-S18, B-S21, B-M13). The report compares `WalkLayout`'s groups with the paper's Table 3 under our group names (`costs_reference.PAPER_QUBIT_GROUPS_FEMOCO`, summing to 1,131): section "Qubits against the paper". The reason code `junk-qubits` no longer applies and leaves spec 13.3; the report's deferred-refinements list drops the early junk measurement.
7. **Outcome arrays have a mode-dependent dtype** (B-S3, B-H4). Under capture the junk outcomes are a traced bool array; in tape mode, an object array of `MeasurementValue`s. A resource key built with the wrong one fails: a bool key in tape mode fails T2 (declared and emitted keys differ), and an object key under capture makes Catalyst drop the rule ("Decomposition rule not found"). `prepare.abstract_junk_outcomes(size)` gives the mode's `AbstractArray`, and a caller must compute it before `resource_key` pauses capture (the dry run's first probe computed it inside, which failed under capture). Spec 7.4 rule 3 and a new HANDOFF pitfall say so.
8. **Where measurements and conditionals live** (B-S4, B-S5, B-M15, B-M16, B-M17, B-H1 to B-H3). The block encodings' rules now contain mid-circuit measurements (the junk) and the unprepares from outcomes contain `qp.cond`; spec 7.4 rules 4 and 5, the Global Constraints' rules 3 to 5 and HANDOFF section 5 name them. The block encodings are never controlled as a whole (the controlled block encoding is its own class whose rule measures the junk the same way), so rule 5's "never placed under `qp.ctrl`" still holds.
9. **QPE wires** (B-M8). `qpe_program`'s wires become 56 + t at `H2_STO3G_TEST` and 1,103 + t at FeMoco-54; spare wires appear where 2t − 1 exceeds the work pool, as before (t ≥ 5 at `H2_STO3G_TEST`, t ≥ 23 at FeMoco-54, where the pool is 44).
10. **Scope and records** (B-S1, B-S2, B-S6, B-S29, B-S22, B-S23, B-S25 to B-S28, B-M1 to B-M3, B-M11, B-M12, B-M14). Spec section 1 drops early junk measurement from the deferred optimizations and states 1,102 qubits; 4.4 item 7 records variant (b); 9.0's class bullet names the two classes among those with `dynamic_argnames`; section 16 records Q-1's working choice and the operators as built (signature, rule step 4, pins and use, catalog and tests). The master plan gains this file, the provenance sentence, the decision and open-question rows, item 26 (variant (b) as built) and Task 4.5's projections after Q1b.
11. **The junk's lifetime in the contracts** (B-S30 to B-S40; the docstrings of Q1b.1's `qroam.py` and `prepare.py`; `docs/architecture.md`). Every statement that the junk stays until the erase now names both paths: (a) the junk kept until `QROAMErase` or the baseline unprepare, or (b) the junk X-measured with reset after the read and before anything else acts on it, its outcomes kept in wire order (`QROAMRead.junk_layout`) and given to `emit_erase_from_outcomes` or the matching unprepare from outcomes, its wires free in between. Where they say when, the documents keep the generic path (a caller may measure right after the read) apart from this walk's: right after the enclosing prepare (steps 4 and 8), before any angle load, which is equivalent because the comparator and the swaps in between do not touch the junk. In the code: `OuterPrepare`'s and `InnerPrepare`'s pre- and postconditions (`junk_x_o` and `junk_b` leave the list of registers nothing may write and get the two paths) and their inverse; `QROAMRead`'s summary, pre- and postconditions (the two paths), adjoint and control note ("The walk erases with `QROAMErase`" becomes the three erasers) and paper note (the paper measures its prepares' junk early, the walk right after each prepare); `QROAMErase`'s pre- and postconditions (the path (b) pointer); the paper sections of the two unprepares from outcomes (the walk measures after steps 4 and 8, not at the reads of steps 2 and 6); the module docstrings of `dfthc.ops.qroam` (whose outcomes the erase uses) and `dfthc.ops.prepare` (the forced order for an unprepare from outcomes; "no walk instance needs `extra_onehot`" replaced). In the spec: the step table and inner rounds of 3.5, the paper's junk in 4.2 and 4.3's decision, 9.3's introduction and adjoint-and-control row (the three erase paths), 9.4's callers of `emit_phase_fixup` (the junk outcomes are now the one operator boundary that carries measurement data, K-10 part 2), 9.5's `extra_onehot` and precondition rows, and 16's description of Q-1. `docs/architecture.md` adds the unprepares from outcomes to the fix-up's emitters.

Exact edits (master plan, spec and HANDOFF, against local `main`):

Each edit replaces one exact text of the file on local `main` at `9f4c7eb`, where it occurs exactly once (checked by `.scratch/plan-check/stages/Q1B/make_doc_edits_q1b.py`, which also applies them in order). Every master-plan old text also occurs exactly once where the orchestrator mirrors the edits: in `.scratch/plan-check/templates/2026-11-11-dfthc-master-plan.md`, except those of B-M15, B-M16 and B-M17, which occur once in the shared fragment `templates/_global_constraints_g01.md` that the template includes (see "Shared edits").

**B-M1** (2026-11-11-dfthc-master-plan.md, section 1, the file table (a new row after the report file's)). Old:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase4-report.md` | Tasks 4.3 to 4.5 |
````

New:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase4-report.md` | Tasks 4.3 to 4.5 |
| `docs/superpowers/plans/2026-11-11-dfthc-phase4-q1b.md` | Task Q1b (sub-tasks Q1b.1 and Q1b.2), written at the Q-1 working choice |
````

**B-M2** (2026-11-11-dfthc-master-plan.md, section 1, code provenance (the end of the paragraph)). Old:

````text
Its large compiles (`fe2s2`, `fe4s4`, FeMoco-54) are projected there and run by the orchestrator.
````

New:

````text
Its large compiles (`fe2s2`, `fe4s4`, FeMoco-54) are projected there and run by the orchestrator. The Q1b file (`phase4-q1b`, Task Q1b) was executed the same way on 2026-11-15 (revision 1), against `main` at `1a3e062` (Task 4.4 merged, report file revision 7); its preamble lists its changes to this plan. The code-hash change makes the orchestrator rerun `fe2s2` and `fe4s4`; those reruns and Task 4.5's FeMoco-54 compiles are projected there.
````

**B-M3** (2026-11-11-dfthc-master-plan.md, section 2, the Q-1 decision row). Old:

````text
| Q-1 early junk measurement | **Open.** Decided by the user after the K-10 spike (Task 0.8) and the Phase 4 resource report, preferring variant (b) if K-10 passes. | the Q-1 decision before G4; conditional Task Q1b |
````

New:

````text
| Q-1 early junk measurement | **Variant (b), working choice** (decision "Q-1 working choice: variant (b)", 2026-11-15): K-10 passed, and the user prefers variant (b) if it does; the orchestrator adopted it under the standing directive, and the user may override before Task Q1b runs | Task Q1b (`phase4-q1b`): the block encodings measure each junk register right after its prepare, and the junk shares the angle wires (1,102 qubits at FeMoco-54) |
````

**B-M4** (2026-11-11-dfthc-master-plan.md, section 2, the baseline's qubits). Old:

````text
- Qubits: `WalkLayout` gives 1,564 with a 44-wire work pool (spec estimate about 1,570).
````

New:

````text
- Qubits: `WalkLayout` gives 1,564 with a 44-wire work pool (spec estimate about 1,570) while the junk persists to the erase; after Task Q1b (Q-1 variant (b)) the junk registers share the angle wires and it gives 1,102, 462 fewer (the paper: 1,131).
````

**B-M5** (2026-11-11-dfthc-master-plan.md, section 3, a new item 26 after item 25 (the end of item 25 is shown)). Old:

````text
All 32 `Operator2` classes of `dfthc.ops` and `dfthc.qpe` have the nine spec 9.0 sections in order.
````

New:

````text
All 32 `Operator2` classes of `dfthc.ops` and `dfthc.qpe` have the nine spec 9.0 sections in order.
26. **Q-1 variant (b) as built** (Q1b plan dry run, 2026-11-15). `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes` take the prepare's wires except the junk, then `extra_onehot`, the prepare's compilable arguments and the dynamic `junk_outcomes` in the junk register's wire order (`QROAMRead.junk_layout`'s slots); their rules undo the swap and the comparator, X-measure the target, compute the parities of the target and junk outcomes against masks built in the rule from the compilable tables (`emit_erase_from_outcomes`, never masks as data) and undo the uniform preparation. `extra_onehot` is forced by the fix-up's one-hot register: at FeMoco-54 the inner one-hot needs 2^7 = 128 reset wires and the inner target has 26, so the block encoding lends the other 102 from the start of the measured junk wires (0 for the outer one, 2^4 = 16 <= 24). Under capture the outcomes are a traced bool array, in tape mode an object array of `MeasurementValue`s, so a rule that declares an unprepare from outcomes builds the key with the mode's dtype (`abstract_junk_outcomes`) before `resource_key` pauses capture; with a bool key in tape mode T2 fails, and with an object key under capture Catalyst drops the rule. `WalkLayout` drops `junk_x_o` and `junk_b` from the groups: each is the first wires of `angles ‖ junk_overflow`, the start of `select` (`junk_overflow` 0 at FeMoco-54, 3 at `H2_STO3G_TEST`); the block encoding's rule X-measures each junk register right after its prepare, before any angle is loaded. FeMoco-54 needs 1,102 wires (outer 71, inner 65, select 799; 1,564 - 72 - 390), `MID_NAPHTHALENE` 336, `H2_STO3G_TEST` 55; QPE 1,103 + t through t = 22. T, A and the total M are unchanged (9,793 and 9,795; 2,524; 8,675 and 8,677): 852 of the M are the rule's own junk measurements (`costs.junk_measurements`), the unprepares measure only their targets (24 and 26). Compiled at `H2_STO3G_TEST`, `H2_631G_TEST`, `FAILS_C3` and `MID_NAPHTHALENE`, the walks equal the closed forms; back to back against the junk-to-erase code, the junk-outcome arrays cost no compile time and 3 to 5% memory (the `MID_NAPHTHALENE` walk 6.8 s at 610 MB against 6.8 s at 631 MB, the controlled walk 579 MB against 604 MB). The report's walk calls no `QROAMErase` (46 rows at `H2_STO3G_TEST`, 44 at `naphthalene`, `fe2s2` and `fe4s4`, 47 at FeMoco-54); the `PhaseFixup` rows take the unprepares' erase arguments, and the cross-check adds the junk measurements to M. Its FeMoco-54 projections stay within 7.5 GB and 60 min: from the H2 systems and `naphthalene`, the walk 198 s at 3.9 GB and the controlled walk 196 s at 4.4 GB; with the earlier `fe2s2` and `fe4s4` measurements scaled by the measured `naphthalene` ratio, 40 s at 1.6 GB and 34 s at 1.7 GB.
````

**B-M6** (2026-11-11-dfthc-master-plan.md, section 4, the Q1b row). Old:

````text
| Q1b | (conditional) Q-1 variant (b): unprepare-from-outcomes operators | A | the Q-1 decision after 4.4 | written at the decision if chosen |
````

New:

````text
| Q1b | (conditional) Q-1 variant (b): unprepare-from-outcomes operators (sub-tasks Q1b.1, Q1b.2) | A | the Q-1 decision after 4.4 | phase4-q1b |
````

**B-M7** (2026-11-11-dfthc-master-plan.md, section 5, wave 4 (the start of the row)). Old:

````text
| 4 | 4.1, then 4.2 and 4.3, then 4.4; then the Q-1 decision, and Q1b if variant (b) is chosen; then 4.5 |
````

New:

````text
| 4 | 4.1, then 4.2 and 4.3, then 4.4; then the Q-1 decision, and Q1b (Q1b.1, then Q1b.2) if variant (b) is chosen; then 4.5 |
````

**B-M8** (2026-11-11-dfthc-master-plan.md, Task 4.1, interfaces (`qpe_program`'s wires)). Old:

````text
(64 + t at `H2_STO3G_TEST`, 1,565 + t at FeMoco-54), and spare wires after them only where 2t − 1 exceeds the work pool
````

New:

````text
(64 + t at `H2_STO3G_TEST`, 1,565 + t at FeMoco-54; 56 + t and 1,103 + t after Task Q1b), and spare wires after them only where 2t − 1 exceeds the work pool
````

**B-M9** (2026-11-11-dfthc-master-plan.md, Task 4.3, programs (the row count)). Old:

````text
are programs the walk does not call. 48 rows at `H2_STO3G_TEST`.
````

New:

````text
are programs the walk does not call. 48 rows at `H2_STO3G_TEST` (46 after Task Q1b, whose walk calls no `QROAMErase`: the `PhaseFixup` rows then take the unprepares' erase arguments).
````

**B-M10** (2026-11-11-dfthc-master-plan.md, Task Q1b, the whole entry). Old:

````text
#### Task Q1b (conditional): Q-1 variant (b)
- **Opened** at the Q-1 decision after Task 4.4, only if the user chooses variant (b) and K-10 passed. G4 completes only after Q1b, because Q1b changes artifacts that D-3 and D-6 rest on. **Tier:** A. **Reviewers:** GPT-5.6, Opus. **Depends on:** the Q-1 decision after 4.4. **Branch:** `phase4/Q1b`.
- **Scope (spec 16, Q-1 variant (b)):**
  - `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`: the prepare's wires except junk, the prepare's compilable arguments, and dynamic `junk_outcomes` in `QROAMRead.junk_layout` order;
  - their pinned rules;
  - the block-encoding rule measuring junk after each prepare;
  - `WalkLayout` aliasing the freed junk wires with the angle registers;
  - spec section 9 catalog entries and the spec 10.3 qubit budget;
  - T2 to T6 and the spec 12.5 certificates over target and junk outcomes;
  - the D-3 and D-6 artifacts brought up to date:
    - regenerate `docs/resources.md` and `docs/resources.json` with `scripts/resource_report.py` (the orchestrator reruns the `--large` systems, because the code hash changed); Task 4.5 then performs the single FeMoco compile;
    - update `docs/architecture.md` (blocks and layout), `docs/verification.md` (the new certificates) and the operator docstrings.
- **Acceptance tests:**
  - The closed forms keep T (9,793; controlled 9,795) and the total M (2,524); compiled counts equal the closed forms at the test presets and `MID_NAPHTHALENE`; qubits are reduced by about 460 (exact value from `WalkLayout`).
  - E-1 to E-4 still pass under the nightly schedules.
  - The report and documentation tests pass again: `tests/report/test_report.py` (including `test_report_mid` in the nightly tier) and `tests/test_docstrings.py`.
- **Definition of done:** the code-level plan is written at the Q-1 decision and executed; the regenerated report and updated documents are committed; then Task 4.5 runs, and G4 completes after it.
````

New:

````text
#### Task Q1b (conditional): Q-1 variant (b)
- **Opened** at the Q-1 decision after Task 4.4, only if the user chooses variant (b) and K-10 passed; the working choice of 2026-11-15 is variant (b) (section 2), and the user may override before Q1b runs. G4 completes only after Q1b, because Q1b changes artifacts that D-3 and D-6 rest on. **Tier:** A. **Reviewers:** GPT-5.6, Opus. **Depends on:** the Q-1 decision after 4.4. Two sub-tasks under the id Q1b, each one commit with every tier green and its own review: **Q1b.1** (branch `phase4/Q1b.1`) and then **Q1b.2** (branch `phase4/Q1b.2`, from `main` after Q1b.1's merge). Plan: `phase4-q1b`.
- **Q1b.1, the unprepares from outcomes:** `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes` in `src/dfthc/ops/prepare.py` (the prepare's wires except junk, then `extra_onehot`: max(2^ob - d, 0) wires lent for the fix-up's one-hot register, 102 for the inner one at FeMoco-54; the prepare's compilable arguments; dynamic `junk_outcomes` in `QROAMRead.junk_layout` order, an object array of `MeasurementValue`s in tape mode; `junk_size(...)`, `required_wires(...)`), their pinned rules through `dfthc.ops.qroam.emit_erase_from_outcomes` (masks built in the rule from the compilable tables), `prepare.abstract_junk_outcomes(size)` for declarations, `costs.outer_unprepare_from_outcomes` and `costs.inner_unprepare_from_outcomes` (the unprepares' T, M = d1 and d2); never adjointed or controlled. Tests: `tests/ops/test_unprepare_from_outcomes.py` (T2 to T6, the order of the junk outcomes, a caller inside another operator's rule, decision "K-10 consequences" (3)), the spec 12.5 certificates in `tests/ops/test_certificates_prepare.py` (outcome vectors over junk and target), the docstring audit at 34 classes; `docs/architecture.md` and `docs/conventions.md` name the classes. Per commit 1,221 + 42 = 1,263, nightly 76 + 10 = 86.
- **Q1b.2, early junk measurement in the walk:** the block encodings' rules (`src/dfthc/ops/walk.py`) X-measure each junk register right after its prepare and call the unprepares from outcomes; `WalkLayout` (`src/dfthc/registers.py`) drops the junk registers from the groups and makes them the first wires of `angles ‖ junk_overflow` (`JUNK`, `junk_sizes`, `split_junk`); `costs.junk_measurements`; the report (`scripts/resource_report.py`: the unprepares' rows, `PhaseFixup` rows from their erase arguments, the junk measurements in the cross-check, the qubit section against the paper's Table 3 groups, `costs_reference.PAPER_QUBIT_GROUPS_FEMOCO`); the walk and QPE docstrings, `dfthc.testing.walk_sim`'s wire counts; the changed literals of `tests/test_registers.py`, `tests/ops/test_block_encoding.py`, `tests/ops/test_walk.py`, `tests/test_work_pool.py`, `tests/test_qpe.py` and `tests/report/test_report.py`; `docs/architecture.md`, `docs/verification.md`, `docs/conventions.md`; the regenerated `docs/resources.md` and `docs/resources.json` of the H2 systems and `naphthalene`. Per commit 1,263 + 5 = 1,268, nightly 86.
- **Acceptance tests:**
  - The closed forms keep T (9,793; controlled 9,795), A (8,675; 8,677) and the total M (2,524), 852 of it the block encoding's junk measurements; compiled counts equal the closed forms at the test presets, `FAILS_C3` and `MID_NAPHTHALENE`; qubits from `WalkLayout`: 1,102 at FeMoco-54, 462 fewer than 1,564 (the paper: 1,131, compared by group in the report).
  - E-1 to E-4 still pass under the nightly schedules; the inventory tests stay exact.
  - The report and documentation tests pass again: `tests/report/test_report.py` (including `test_report_mid` in the nightly tier) and `tests/test_docstrings.py`.
- **Definition of done:** Q1b.1 merged; Q1b.2's implementer commits the code, tests, documents and the report of the H2 systems and `naphthalene`, and stops; the orchestrator reruns `--large fe2s2` and `--large fe4s4`, each alone under the watchdog, and commits the report on the branch; after the merge Task 4.5 runs, and G4 completes after it.
````

**B-M11** (2026-11-11-dfthc-master-plan.md, Task 4.5, procedure (the projection)). Old:

````text
the `fe2s2` and `fe4s4` points refine the projection.
````

New:

````text
the `fe2s2` and `fe4s4` points refine the projection. After Task Q1b, which changes the code hash, the Q1b plan's dry run projects them from its regenerated small-system report to 198 s and 3.9 GB (walk) and 196 s and 4.4 GB (controlled walk), and, with the earlier `fe2s2` and `fe4s4` measurements scaled by the measured `naphthalene` ratio as an estimate of the orchestrator's reruns, to 40 s and 1.6 GB and 34 s and 1.7 GB; all within 7.5 GB and 60 min.
````

**B-M12** (2026-11-11-dfthc-master-plan.md, section 13, the Q-1 row). Old:

````text
| Q-1 early junk measurement | K-10 verdict (Task 0.8) at G0, then the user's decision after Task 4.4, from the resource report (Task 4.3) | Task Q1b (with the report and documentation regenerated) before G4 completes, or the baseline unchanged |
````

New:

````text
| Q-1 early junk measurement | K-10 verdict (Task 0.8) at G0, then the working choice of variant (b) on 2026-11-15 (the user's stated preference; the user may override before Task Q1b runs) | Task Q1b (`phase4-q1b`, with the report and documentation regenerated) before G4 completes |
````

**B-M13** (2026-11-11-dfthc-master-plan.md, section 13, the K-9 row). Old:

````text
| K-9 qubits above the paper's | Task 4.3 report; Q-1 | documentation |
````

New:

````text
| K-9 qubits above the paper's | Task 4.3 report; Task Q1b (1,102 against the paper's 1,131 after the early junk measurement, compared by group in the report) | documentation |
````

**B-M14** (2026-11-11-dfthc-master-plan.md, section 14, the row of spec section 10). Old:

````text
| 10 layout and budget | 1A.1, 3.3 (work pool), 4.3 (qubit report) |
````

New:

````text
| 10 layout and budget | 1A.1, 3.3 (work pool), 4.3 (qubit report), Q1b (the junk on the angle wires, 10.2 and 10.3) |
````

**B-M15** (2026-11-11-dfthc-master-plan.md, Global Constraints, rule 3 (in the template, the shared fragment `templates/_global_constraints_g01.md`)). Old:

````text
3. Every argument that determines circuit structure is in `compilable_argnames`; `static_argnames` are not used; `dynamic_argnames` only for runtime data (`PhaseFixup` parities, `QPERound` earlier bits, `QROAMRead`'s per-leaf table words).
````

New:

````text
3. Every argument that determines circuit structure is in `compilable_argnames`; `static_argnames` are not used; `dynamic_argnames` only for runtime data (`PhaseFixup` parities, `QPERound` earlier bits, `QROAMRead`'s per-leaf table words, the junk outcomes of `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`).
````

**B-M16** (2026-11-11-dfthc-master-plan.md, Global Constraints, rule 4 (its second clause; in the shared fragment)). Old:

````text
`qp.cond` on mid-circuit measurement outcomes only in erase operators and the QPE wrapper;
````

New:

````text
`qp.cond` on mid-circuit measurement outcomes only in erase operators, the unprepares from outcomes and the QPE wrapper;
````

**B-M17** (2026-11-11-dfthc-master-plan.md, Global Constraints, rule 5 (in the shared fragment)). Old:

````text
5. Measurement-based uncompute lives only in `QROAMErase`, `AngleErase` and the base rules of `OuterUnprepare` and `InnerUnprepare`; an operator that contains a measurement is never placed under `qp.ctrl`.
````

New:

````text
5. Measurement-based uncompute lives only in `QROAMErase`, `AngleErase`, the base rules of `OuterUnprepare`, `InnerUnprepare`, `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`, and the junk measurements of the block encodings' rules (spec 16, Q-1 variant (b)); an operator that contains a measurement is never placed under `qp.ctrl`.
````

**B-M18** (2026-11-11-dfthc-master-plan.md, Task 1A.1, acceptance tests (the wire counts; Task 1A.1 is merged, so the Q1b values are added in parentheses)). Old:

````text
  - Group sizes at FEMOCO54: system 108, phase_grad 15, outer 143, inner 455, select 799, work 44; `num_wires == 1564`.
  - `H2_STO3G_TEST.num_wires == 63`, and work = 8.
````

New:

````text
  - Group sizes at FEMOCO54: system 108, phase_grad 15, outer 143, inner 455, select 799, work 44; `num_wires == 1564` (after Task Q1b, whose junk registers share the angle wires: outer 71, inner 65, `num_wires == 1102`).
  - `H2_STO3G_TEST.num_wires == 63` (55 after Task Q1b), and work = 8.
````

**B-S1** (2026-11-11-dfthc-pennylane-design.md, 1, the qubit line of the summary). Old:

````text
- about 1,570 qubits, against the paper's 1,131.
````

New:

````text
- 1,102 qubits, with each read's junk measured right after its prepare (Q-1 variant (b), §16; 1,564 if the junk is kept to the erase), against the paper's 1,131.
````

**B-S2** (2026-11-11-dfthc-pennylane-design.md, 4.4, item 7). Old:

````text
7. **Junk persists from read to erase.** The qubit count therefore exceeds the paper's (§10.3). An alternative with no Toffoli penalty stays open (Q-1, §16). The user decided to settle it after the Phase 0 probe K-10 and the Phase 4 resource report, preferring the argument variant if it works.
````

New:

````text
7. **Junk persists from read to erase in the baseline; Q-1 variant (b) measures it early.** With the junk kept, the qubit count exceeds the paper's (§10.3). Q-1 variant (b) (§16; the working choice of 2026-11-15, after K-10 passed, as the user preferred) X-measures each read's junk right after its prepare and passes the outcomes to the unprepares from outcomes, with no Toffoli penalty; the junk then shares the angle wires: 1,102 qubits at FeMoco-54, against the paper's 1,131 (§10.3).
````

**B-S3** (2026-11-11-dfthc-pennylane-design.md, 7.4, rule 3 (the dynamic arguments)). Old:

````text
   - `dynamic_argnames` are used only for runtime data: the parity bits of `PhaseFixup`, the earlier measured bits of `QPERound`, and the per-leaf table words of `QROAMRead` (§9.3), which its callers build from their compilable tables.
````

New:

````text
   - `dynamic_argnames` are used only for runtime data: the parity bits of `PhaseFixup`, the earlier measured bits of `QPERound`, the per-leaf table words of `QROAMRead` (§9.3), which its callers build from their compilable tables, and the junk outcomes of `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes` (§9.7, §9.8; §16, Q-1 variant (b)). Measurement outcomes are a bool array under capture and an object array of `MeasurementValue`s in tape mode, so a resource key that declares them takes its dtype from the mode, computed before `resource_key` pauses capture (`dfthc.ops.prepare.abstract_junk_outcomes`).
````

**B-S4** (2026-11-11-dfthc-pennylane-design.md, 7.4, rule 4 (the conditionals on outcomes)). Old:

````text
   - `qp.cond` on mid-circuit measurement outcomes appears only inside erase operators and the QPE wrapper.
````

New:

````text
   - `qp.cond` on mid-circuit measurement outcomes appears only inside erase operators, the unprepares from outcomes (§9.7, §9.8) and the QPE wrapper.
````

**B-S5** (2026-11-11-dfthc-pennylane-design.md, 7.4, rule 5 (where measurements live)). Old:

````text
   - It lives only in dedicated erase operators (`QROAMErase`, `AngleErase`) and in the base rules of the unprepare classes `OuterUnprepare` and `InnerUnprepare` (§9.7, §9.8), which call `QROAMErase`. Their resource functions declare the measurements.
````

New:

````text
   - It lives only in dedicated erase operators (`QROAMErase`, `AngleErase`), in the base rules of the unprepare classes `OuterUnprepare` and `InnerUnprepare` (§9.7, §9.8), which call `QROAMErase`, in the base rules of their siblings `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`, which measure the read's target and apply the fix-up themselves, and in the block encodings' rules, which X-measure each read's junk right after its prepare (§9.21; §16, Q-1 variant (b)). Their resource functions declare the measurements.
````

**B-S6** (2026-11-11-dfthc-pennylane-design.md, 9.0, the class bullet). Old:

````text
only for `PhaseFixup`, `QPERound` and `QROAMRead`, `dynamic_argnames`.
````

New:

````text
only for `PhaseFixup`, `QPERound`, `QROAMRead`, `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`, `dynamic_argnames`.
````

**B-S7** (2026-11-11-dfthc-pennylane-design.md, 9.7, a new row for `OuterUnprepareFromOutcomes` and the cost row). Old:

````text
| `OuterUnprepare` (own class, pinned base rule; same wires and compilable arguments as `OuterPrepare`) | `CSWAP`s (self-inverse), adjoint comparator (measured uncompute; also cleans `cmp_work_x_o`), `QROAMErase`, `H` on `m_x_o` and `sigma_1`, adjoint `FlaggedUniformPrep` |
| Cost | Prepare $T = F(X_o) + T_{\mathrm{read,out}} + b_{k1} + w_x$; unprepare $T = w_x + T_{\mathrm{erase,out}} + F^\dagger(X_o)$, $M = K_1 d_1$ (§9.24). At FeMoco-54: prepare $30 + 156 + 15 + 9 = 210$; unprepare $9 + 36 + 24 = 69$, $M = 96$. |
````

New:

````text
| `OuterUnprepare` (own class, pinned base rule; same wires and compilable arguments as `OuterPrepare`) | `CSWAP`s (self-inverse), adjoint comparator (measured uncompute; also cleans `cmp_work_x_o`), `QROAMErase`, `H` on `m_x_o` and `sigma_1`, adjoint `FlaggedUniformPrep` |
| `OuterUnprepareFromOutcomes` (own class, pinned base rule; §16, Q-1 variant (b); the walk's outer unprepare) | Wires: `OuterPrepare`'s except `junk_x_o`, then `extra_onehot` ($\max(2^{k_{e1}} - d_1, 0)$ wires in $\lvert 0\rangle$, which the block encoding lends from the start of the measured junk wires; 0 at FeMoco-54). Compilable: `OuterPrepare`'s. Dynamic: `junk_outcomes`, the $(K_1 - 1) d_1$ outcomes of the junk measurement right after the prepare, in the junk register's wire order (the slots of `QROAMRead.junk_layout`). Base rule: the `CSWAP`s, the adjoint comparator, `H` and `qp.measure(q, reset=True)` on `alt_x_o ‖ keep_x_o`, the parities of the target and junk outcomes against the loaded bits (masks built in the rule from the compilable tables) and `emit_phase_fixup` over the loaded set with the reset target ‖ `extra_onehot` as its one-hot register (`dfthc.ops.qroam.emit_erase_from_outcomes`), `H` on `m_x_o` and `sigma_1`, the adjoint `FlaggedUniformPrep`. Never adjointed or controlled |
| Cost | Prepare $T = F(X_o) + T_{\mathrm{read,out}} + b_{k1} + w_x$; unprepare $T = w_x + T_{\mathrm{erase,out}} + F^\dagger(X_o)$, $M = K_1 d_1$ (§9.24). At FeMoco-54: prepare $30 + 156 + 15 + 9 = 210$; unprepare $9 + 36 + 24 = 69$, $M = 96$; `OuterUnprepareFromOutcomes` the same $T$, $M = d_1 = 24$ (the block encoding measures the other $(K_1 - 1) d_1 = 72$). |
````

**B-S8** (2026-11-11-dfthc-pennylane-design.md, 9.8, a new row for `InnerUnprepareFromOutcomes` and the cost row). Old:

````text
| `InnerUnprepare` (own class, pinned base rule; same wires and compilable arguments as `InnerPrepare`) | `CSWAP`s, adjoint comparator (measured uncompute), `QROAMErase` (one-hot over the low $k_{e2}$ bits of $x_o$ and all bits of $b$), `H`s, adjoint `FlaggedUniformPrep` |
| Cost | Prepare $T = F(B+1) + T_{\mathrm{read,in}} + b_{k2} + w_b$; unprepare $T = w_b + T_{\mathrm{erase,in}} + F^\dagger(B+1)$, $M = K_2 d_2$ (§9.24). At FeMoco-54: prepare $18 + 933 + 15 + 5 = 971$; unprepare $5 + 196 + 16 = 217$, $M = 416$. |
````

New:

````text
| `InnerUnprepare` (own class, pinned base rule; same wires and compilable arguments as `InnerPrepare`) | `CSWAP`s, adjoint comparator (measured uncompute), `QROAMErase` (one-hot over the low $k_{e2}$ bits of $x_o$ and all bits of $b$), `H`s, adjoint `FlaggedUniformPrep` |
| `InnerUnprepareFromOutcomes` (own class, pinned base rule; §16, Q-1 variant (b); the walk's inner unprepare) | Wires: `InnerPrepare`'s except `junk_b`, then `extra_onehot` ($\max(2^{k_{e2} + w_b} - d_2, 0)$ wires in $\lvert 0\rangle$, which the block encoding lends from the start of the measured junk wires; 102 at FeMoco-54, whose one-hot needs 128 wires and whose target has 26). Compilable: `InnerPrepare`'s. Dynamic: `junk_outcomes`, the $(K_2 - 1) d_2$ outcomes of the junk measurement right after the prepare, in the junk register's wire order. Base rule: as `OuterUnprepareFromOutcomes`'s on the inner registers (`H` and `qp.measure(q, reset=True)` on `alt_b ‖ keep_b ‖ sgn ‖ d_sgn ‖ r`; one-hot over the low $k_{e2}$ bits of $x_o$ and all bits of $b$). Never adjointed or controlled |
| Cost | Prepare $T = F(B+1) + T_{\mathrm{read,in}} + b_{k2} + w_b$; unprepare $T = w_b + T_{\mathrm{erase,in}} + F^\dagger(B+1)$, $M = K_2 d_2$ (§9.24). At FeMoco-54: prepare $18 + 933 + 15 + 5 = 971$; unprepare $5 + 196 + 16 = 217$, $M = 416$; `InnerUnprepareFromOutcomes` the same $T$, $M = d_2 = 26$ (the block encoding measures the other $(K_2 - 1) d_2 = 390$). |
````

**B-S9** (2026-11-11-dfthc-pennylane-design.md, 9.21, the wires row). Old:

````text
| Wires | Grouped: `outer`, `inner`, `select`, `system`, `phase_grad`, `work`, exactly `WalkLayout`'s groups (§10; `required_wires(params, tables)` gives their sizes). The rule splits them once with `WalkLayout.split_group` and gives each child the registers its §9 table names and the prefix of `work` of its need; the prepares take the whole `phase_grad` |
````

New:

````text
| Wires | Grouped: `outer`, `inner`, `select`, `system`, `phase_grad`, `work`, exactly `WalkLayout`'s groups (§10; `required_wires(params, tables)` gives their sizes); the junk registers are the start of `select` (§10.2). The rule splits them once with `WalkLayout.split_group` and `WalkLayout.split_junk` and gives each child the registers its §9 table names and the prefix of `work` of its need; the prepares take the whole `phase_grad`; each unprepare from outcomes gets the first `extra_onehot` wires of its junk register |
````

**B-S10** (2026-11-11-dfthc-pennylane-design.md, 9.21, the base rule row). Old:

````text
| Base rule | `OuterPrepare`; [`InnerPrepare`, `InnerSigns`, `InnerSelect`, `InnerUnprepare`]; `RefT2`; [`InnerPrepare`, `InnerSelectDagger`, `InnerSigns`, `InnerUnprepare`]; `OuterUnprepare` |
````

New:

````text
| Base rule | `OuterPrepare`, X measurement of `junk_x_o`; [`InnerPrepare`, X measurement of `junk_b`, `InnerSigns`, `InnerSelect`, `InnerUnprepareFromOutcomes`]; `RefT2`; [`InnerPrepare`, X measurement of `junk_b`, `InnerSelectDagger`, `InnerSigns`, `InnerUnprepareFromOutcomes`]; `OuterUnprepareFromOutcomes` (Q-1 variant (b), §16: an X measurement is `H` and `qp.measure(q, reset=True)` on every junk wire; each unprepare from outcomes takes the outcomes of the latest measurement of its junk register) |
````

**B-S11** (2026-11-11-dfthc-pennylane-design.md, 9.21, the cost row (its end)). Old:

````text
$M = 2524$ explicit erase measurements. |
````

New:

````text
$M = 2524$ explicit erase measurements, $852 = 72 + 2 \cdot 390$ of them the rule's own junk measurements (`dfthc.costs.junk_measurements`). |
````

**B-S12** (2026-11-11-dfthc-pennylane-design.md, 9.24, the operator table (two rows after the inner pair's)). Old:

````text
| `InnerPrepare` / `InnerUnprepare` | $F(B+1) + T_{\mathrm{read,in}} + b_{k2} + w_b$ / $w_b + T_{\mathrm{erase,in}} + F^\dagger(B+1)$ | 0 / $K_2 d_2$ |
````

New:

````text
| `InnerPrepare` / `InnerUnprepare` | $F(B+1) + T_{\mathrm{read,in}} + b_{k2} + w_b$ / $w_b + T_{\mathrm{erase,in}} + F^\dagger(B+1)$ | 0 / $K_2 d_2$ |
| `OuterUnprepareFromOutcomes` / `InnerUnprepareFromOutcomes` | those of `OuterUnprepare` / `InnerUnprepare` | $d_1$ / $d_2$ |
| junk measurements in `DFTHCBlockEncoding`'s rule (Q-1 variant (b)) | 0 | $(K_1 - 1)d_1 + 2(K_2 - 1)d_2$ |
````

**B-S13** (2026-11-11-dfthc-pennylane-design.md, 9.24, the FeMoco-54 evaluation (a row after the prepares')). Old:

````text
| `OuterPrepare` / `OuterUnprepare`; `InnerPrepare` / `InnerUnprepare` | 210 / 69; 971 / 217 |
````

New:

````text
| `OuterPrepare` / `OuterUnprepare`; `InnerPrepare` / `InnerUnprepare` | 210 / 69; 971 / 217 |
| `OuterUnprepareFromOutcomes` / `InnerUnprepareFromOutcomes` ($M$); junk measurements per walk | 69 / 217 ($M = 24$ / 26); 852 ($72 + 2 \cdot 390$) |
````

**B-S14** (2026-11-11-dfthc-pennylane-design.md, 10.1, the outer, inner and select rows). Old:

````text
| `outer` | `x_o`, `succ_x_o`, `succ_work_x_o`, `sigma_1`, `m_x_o`, `less_than_x_o`, `cmp_work_x_o`, `alt_x_o`, `keep_x_o`, `junk_x_o` |
| `inner` | `b`, `succ_b`, `succ_work_b`, `sigma_0`, `m_b`, `less_than_b`, `cmp_work_b`, `alt_b`, `keep_b`, `sgn`, `d_sgn`, `r`, `junk_b` |
| `select` | `angles`, `G0`, `G1`, `b_neq_B`, `sg` |
````

New:

````text
| `outer` | `x_o`, `succ_x_o`, `succ_work_x_o`, `sigma_1`, `m_x_o`, `less_than_x_o`, `cmp_work_x_o`, `alt_x_o`, `keep_x_o` |
| `inner` | `b`, `succ_b`, `succ_work_b`, `sigma_0`, `m_b`, `less_than_b`, `cmp_work_b`, `alt_b`, `keep_b`, `sgn`, `d_sgn`, `r` |
| `select` | `angles`, `junk_overflow`, `G0`, `G1`, `b_neq_B`, `sg`; the junk registers `junk_x_o` and `junk_b` are the first wires of `angles ‖ junk_overflow` (§10.2) |
````

**B-S15** (2026-11-11-dfthc-pennylane-design.md, 10.2, the one-hot and aliasing rules). Old:

````text
- `QROAMErase` and `AngleErase` reuse their own freshly reset target, junk and angle wires as the one-hot register.
- No other aliasing exists. Tests check that groups do not overlap and that every operator call receives exactly `required_wires(...)`.
````

New:

````text
- `QROAMErase` and `AngleErase` reuse their own freshly reset target, junk and angle wires as the one-hot register; the unprepares from outcomes their reset target and `extra_onehot`, which the block encoding lends from the start of the measured junk wires.
- The reads' junk registers `junk_x_o` and `junk_b` have no wires of their own (Q-1 variant (b), §16): each is the first wires of `angles ‖ junk_overflow`, the start of the `select` group, with `junk_overflow` = $\max((K_1 - 1)d_1, (K_2 - 1)d_2, (N - 1)b_{\mathrm{rot}}) - (N - 1)b_{\mathrm{rot}}$ wires (0 at FeMoco-54, 3 at `H2_STO3G_TEST`). The block encoding X-measures each junk register right after its prepare, before any angle is loaded, so the junk and the angles never hold data at the same time, and no operator takes both.
- No other aliasing exists. Tests check that the groups do not overlap, that the junk registers are the start of the angle wires, and that every operator call receives exactly `required_wires(...)`.
````

**B-S16** (2026-11-11-dfthc-pennylane-design.md, 10.3, the budget). Old:

````text
At FeMoco-54 (estimate; `WalkLayout` computes the exact number and the report prints it):

| Group | Wires |
| --- | --- |
| `system` | 108 |
| `phase_grad` | 15 |
| `outer` (51 index and data + 72 junk + 14 `cmp_work` + 6 `succ_work`) | 143 |
| `inner` (49 index and data + 390 junk + 14 `cmp_work` + 2 `succ_work`) | 455 |
| `select` (795 angles + 4 flags) | 799 |
| `work` (the largest need is `RefAB`'s 43 zeroed `MultiControlledX` work wires, 44 when controlled; the Givens adders need 14) | ≈ 50 |
| **Total** | **≈ 1,570** |

The paper reports 1,131. Its inner-read temporaries (313 qubits) share the 795 angle qubits, which requires measuring the inner junk before the angle load. In our design the junk persists from read to erase (§4.3), and the angle registers are live in between, so the two cannot share. Early junk measurement would recover about 460 qubits with no Toffoli penalty, in one of two variants (§16, Q-1).

The `H2_STO3G_TEST` configuration needs about 60 wires, exactly as computed by `WalkLayout`.
````

New:

````text
At FeMoco-54 `WalkLayout` gives these wires (the report prints the same comparison):

| Group | Wires | The paper (its Table 3) |
| --- | --- | --- |
| `system` | 108 | 108 |
| `phase_grad` | 15 | 15 (SEL's $b_{\mathrm{rot}}$) |
| `outer` (51 index and data + 14 `cmp_work` + 6 `succ_work`) | 71 | 50 |
| `inner` (49 index and data + 14 `cmp_work` + 2 `succ_work`) | 65 | 53 |
| `select` (795 angles, which the 72 outer and 390 inner junk wires share, + 4 flags; `junk_overflow` 0) | 799 | 795 (RPREP) |
| `work` (the largest need is the controlled `RefAB`'s 44 zeroed `MultiControlledX` work wires; the Givens adders need 14) | 44 | 110 (the outer prepare's temporaries) |
| **Total** | **1,102** | **1,131** |

The junk registers have no wires of their own (§10.2; Q-1 variant (b), §16): with them the walk would need 1,564. The paper shares its inner-read temporaries (313 qubits) with its 795 angle qubits, which requires measuring the inner junk before the angle load, and adds the outer prepare's 110 temporaries to its 1,021 persistent qubits; ours shares both junk registers with the angles, and the work pool takes the place of the temporaries. The rest of the difference, −29 in all, is our dirty comparator and uniform-preparation work registers (20 outer, 16 inner; they buy the free comparator adjoints, §4.4 item 10), the keep test's `less_than_x_o`, the inner entry registers (`less_than_b`, `sgn`, `d_sgn`, `r`: 7 wires against the paper's 11 for G, r and c) and the four select flags.

The `H2_STO3G_TEST` configuration needs 55 wires (its 7 inner junk wires exceed its 4 angle wires, so `junk_overflow` has 3), exactly as computed by `WalkLayout`.
````

**B-S17** (2026-11-11-dfthc-pennylane-design.md, 12.5, the certified blocks (the variant (b) item)). Old:

````text
  - if Q-1 variant (b) is adopted, `OuterPrepare` → junk measurement → `OuterUnprepareFromOutcomes`, and likewise for the inner pair, with $\xi$ ranging over the concatenated target and junk outcomes.
````

New:

````text
  - Q-1 variant (b) (the working choice, Task Q1b): `OuterPrepare` → junk measurement → `OuterUnprepareFromOutcomes`, and likewise for the inner pair, with $\xi$ ranging over the junk outcomes and the target outcomes together, at the shapes of the two certificates above; and `InnerPrepare` → junk measurement → `InnerUnprepareFromOutcomes` inside another operator's rule, as the block encoding calls them.
````

**B-S18** (2026-11-11-dfthc-pennylane-design.md, 13.2). Old:

````text
See §10.3: about 1,570 against the paper's 1,131.
````

New:

````text
See §10.3: 1,102 with Q-1 variant (b) (1,564 with the junk kept to the erase) against the paper's 1,131, compared by group.
````

**B-S19** (2026-11-11-dfthc-pennylane-design.md, 13.3, step 4). Old:

````text
4. Checks exactly, and exits non-zero on any difference: every compiled count equals its closed form, and at FeMoco-54 the sum over the blocks of calls per walk times the closed form equals the compiled walk and controlled-walk totals of $T$, $M$ and $A$.
````

New:

````text
4. Checks exactly, and exits non-zero on any difference: every compiled count equals its closed form, and at FeMoco-54 the sum over the blocks of calls per walk times the closed form, plus in $M$ the block encoding's own junk measurements (Q-1 variant (b)), equals the compiled walk and controlled-walk totals of $T$, $M$ and $A$.
````

**B-S20** (2026-11-11-dfthc-pennylane-design.md, 13.3, the report rows (the partition sentence)). Old:

````text
The partition rows, the children of the walks, block encodings and selects, times their calls per walk give the walk exactly.
````

New:

````text
The partition rows, the children of the walks, block encodings and selects, times their calls per walk give the walk exactly, with the block encoding's own junk measurements in $M$; the walk's unprepares from outcomes erase without `QROAMErase`, so the `PhaseFixup` rows take their erase arguments and `QROAMErase` has no row.
````

**B-S21** (2026-11-11-dfthc-pennylane-design.md, 13.3, the reason codes (`junk-qubits` removed)). Old:

````text
| `attrib` | The paper attributes the cost to a different step |
| `junk-qubits` | Junk persists until the erase |
````

New:

````text
| `attrib` | The paper attributes the cost to a different step |
````

**B-S22** (2026-11-11-dfthc-pennylane-design.md, 16, questions for the user (the Q-1 sentence)). Old:

````text
Q-1 stays open and does not block Phase 0.
````

New:

````text
Q-1 has a working choice, variant (b), which the user may override.
````

**B-S23** (2026-11-11-dfthc-pennylane-design.md, 16, Q-1 (its heading)). Old:

````text
- **Q-1: Early junk measurement. Open; decided after K-10 and the Phase 4 report.**
````

New:

````text
- **Q-1: Early junk measurement. Working choice: variant (b) (2026-11-15).**
````

**B-S24** (2026-11-11-dfthc-pennylane-design.md, 16, Q-1 variant (b), signature). Old:

````text
    - *Signature.* Wires: the matching prepare's wires except its junk register. Compilable: the prepare's compilable arguments. Dynamic: `junk_outcomes`, a boolean array with one entry per junk qubit ($(K_1 - 1)d_1$ outer, $(K_2 - 1)d_2$ inner), in the order of `QROAMRead.junk_layout`.
````

New:

````text
    - *Signature.* Wires: the matching prepare's wires except its junk register, then `extra_onehot`, $\max(2^{\mathrm{ob}} - d, 0)$ wires in $\lvert 0\rangle$ for the fix-up's one-hot register beyond the reset target ($d$ the read's entry width; 0 outer and 102 inner at FeMoco-54, whose inner one-hot needs 128 wires and whose inner target has 26: without the junk the reset target alone cannot hold it, and the block encoding lends these wires from the start of the measured junk wires). Compilable: the prepare's compilable arguments. Dynamic: `junk_outcomes`, a boolean array with one entry per junk qubit ($(K_1 - 1)d_1$ outer, $(K_2 - 1)d_2$ inner), in the order of `QROAMRead.junk_layout` (the junk register's wire order); in tape mode an object array of `MeasurementValue`s.
````

**B-S25** (2026-11-11-dfthc-pennylane-design.md, 16, Q-1 variant (b), rule step 4). Old:

````text
      4. compute the parities with `mcm_parities` from the concatenated (target ‖ `junk_outcomes`) outcomes, and apply `emit_phase_fixup` over the read's loaded set;
````

New:

````text
      4. compute the parities with `mcm_parities` from the concatenated (target ‖ `junk_outcomes`) outcomes, and apply `emit_phase_fixup` over the read's loaded set with the reset target ‖ `extra_onehot` as its one-hot register;
````

**B-S26** (2026-11-11-dfthc-pennylane-design.md, 16, Q-1 variant (b), pins and use (its last sentence)). Old:

````text
`WalkLayout` then lets the angle registers reuse the freed junk wires.
````

New:

````text
`WalkLayout` then lets the angle registers reuse the freed junk wires: each junk register is the first wires of `angles ‖ junk_overflow` (§10.2).
````

**B-S27** (2026-11-11-dfthc-pennylane-design.md, 16, Q-1 variant (b), catalog and tests). Old:

````text
    - *Catalog and tests.* If adopted, both operators get §9 entries, T2–T6 tests and §12.5 certificates (outcome vectors over target and junk). `OuterUnprepare` and `InnerUnprepare` remain for the baseline. Block boundaries stay as they are. Feasibility is decided by K-10.
````

New:

````text
    - *Catalog and tests.* Both operators have §9 entries (§9.7, §9.8), T2–T6 tests and §12.5 certificates (outcome vectors over target and junk), and an `InnerUnprepareFromOutcomes` test calls it from inside another operator's rule. `OuterUnprepare` and `InnerUnprepare` remain for the baseline. Block boundaries stay as they are. K-10 passed.
````

**B-S28** (2026-11-11-dfthc-pennylane-design.md, 16, Q-1 status). Old:

````text
  - *Status.* Open. The user decides after K-10 (Phase 0) and the Phase 4 resource report, preferring variant (b) if K-10 passes. The baseline keeps the junk until the erase.
````

New:

````text
  - *Status.* Variant (b) is the working choice (2026-11-15): K-10 passed and the user prefers variant (b) if it does, so the orchestrator adopted it under the user's standing directive; the user may override before Task Q1b runs. The FeMoco-54 walk needs 1,102 qubits with it (1,564 with the junk kept to the erase); T, A and the total M are unchanged (§9.24, §10.3).
````

**B-S29** (2026-11-11-dfthc-pennylane-design.md, 1, not in scope (the deferred optimizations)). Old:

````text
- deferred cost and qubit optimizations: early junk measurement (Q-1, open) and truncated unary iteration (Q-4, decided as a later optimization);
````

New:

````text
- deferred cost optimizations: truncated unary iteration (Q-4, decided as a later optimization); the early junk measurement of Q-1 is in scope as variant (b) (§16);
````

**B-S30** (2026-11-11-dfthc-pennylane-design.md, 3.5, the row of steps 19–22). Old:

````text
| 19–22 | Undo 5–8; erase inner data by measurement | `InnerUnprepare` |
````

New:

````text
| 19–22 | Undo 5–8; erase inner data by measurement (the junk was X-measured right after step 8) | `InnerUnprepareFromOutcomes`; `InnerUnprepare` measures the junk here instead (§16, Q-1 variant (b)) |
````

**B-S31** (2026-11-11-dfthc-pennylane-design.md, 3.5, the row of steps 24–27). Old:

````text
| 24–27 | Undo 1–4; erase outer data by measurement | `OuterUnprepare` |
````

New:

````text
| 24–27 | Undo 1–4; erase outer data by measurement (the junk was X-measured right after step 4) | `OuterUnprepareFromOutcomes`; `OuterUnprepare` measures the junk here instead (§16, Q-1 variant (b)) |
````

**B-S32** (2026-11-11-dfthc-pennylane-design.md, 3.5, the two inner rounds). Old:

````text
1. Before $\mathrm{REF}_{T_2}$: `InnerPrepare`, `InnerSigns`, `InnerSelect`, `InnerUnprepare`.
2. After it: `InnerPrepare`, `InnerSelectDagger`, `InnerSigns`, `InnerUnprepare`.
````

New:

````text
1. Before $\mathrm{REF}_{T_2}$: `InnerPrepare`, the X measurement of `junk_b`, `InnerSigns`, `InnerSelect`, `InnerUnprepareFromOutcomes`.
2. After it: `InnerPrepare`, the X measurement of `junk_b`, `InnerSelectDagger`, `InnerSigns`, `InnerUnprepareFromOutcomes`.

The block encoding X-measures `junk_x_o` right after step 4 and `junk_b` right after each step 8, as the paper does (App. B.7.2, steps 2 and 4), and passes each round's outcomes to its unprepare from outcomes (§16, Q-1 variant (b)). The baseline `OuterUnprepare` and `InnerUnprepare` keep the junk to their erase; the walk does not call them.
````

**B-S33** (2026-11-11-dfthc-pennylane-design.md, 4.2, the paper's junk (Option 2's table-read bullet)). Old:

````text
  - The paper instead keeps the swap junk until a measured erase.
````

New:

````text
  - The paper instead erases the swap junk by measurement (X measurements, which its prepares take early, in the prepare, and a phase fix-up later), so it never undoes the swap network.
````

**B-S34** (2026-11-11-dfthc-pennylane-design.md, 4.3, the decision's first bullet). Old:

````text
- The read keeps its swap junk on caller-provided wires until the measured erase.
````

New:

````text
- The read leaves its swap junk on caller-provided wires for a measured erase and never undoes the swap network: the caller keeps the junk until `QROAMErase`, or X-measures it with reset after the read, before anything else acts on the junk wires, and passes the outcomes, in wire order, to the erase of the target (Q-1 variant (b), §16). The walk measures each alias read's junk right after the enclosing prepare (steps 4 and 8), before any angle load; the comparator and the swaps between the read and that point do not touch the junk, so this equals measuring right after the read.
````

**B-S35** (2026-11-11-dfthc-pennylane-design.md, 9.3, the introduction's junk bullet). Old:

````text
- In the paper's variant, the $K - 1$ entries that were not requested stay on extra qubits as *junk*. A later measurement-based erase clears them, so the swap network never has to be undone.
````

New:

````text
- In the paper's variant, the $K - 1$ entries that were not requested stay on extra qubits as *junk*. A measurement-based erase clears them, so the swap network never has to be undone: X measurements, either at the erase or early, then a phase fix-up at the erase. A caller may measure the junk right after the read; the paper's prepares measure it early, in the prepare, and the walk, under Q-1 variant (b) (§16), right after each enclosing prepare (steps 4 and 8), before any angle load.
````

**B-S36** (2026-11-11-dfthc-pennylane-design.md, 9.4, the helper's callers). Old:

````text
`QROAMErase` and `AngleErase` call that helper directly with their measurement-derived parities. They do so on architectural grounds: the parities never leave the rule that measured them, so no operator boundary carries measurement data, and the erase is a single node in the graph and in the resource report.
````

New:

````text
`QROAMErase`, `AngleErase` and the unprepares from outcomes (through `emit_erase_from_outcomes`, §16) call that helper directly with their measurement-derived parities. They do so on architectural grounds: the parities never leave the rule that computes them, and the erase is a single node in the graph and in the resource report. The one operator boundary that carries measurement data is the junk outcomes' dynamic array, from the block encoding's rule into an unprepare from outcomes (K-10 part 2; §16, Q-1 variant (b)); the parities are still computed in the rule that applies the fix-up.
````

**B-S37** (2026-11-11-dfthc-pennylane-design.md, 9.5, the wires row (its end)). Old:

````text
$, 0)$ wires; 0 in every walk instance) |
````

New:

````text
$, 0)$ wires; 0 wherever the baseline unprepares call it, whose constructors check that the one-hot fits; the walk calls no `QROAMErase` under Q-1 variant (b), §16) |
````

**B-S38** (2026-11-11-dfthc-pennylane-design.md, 9.5, the precondition row (its end)). Old:

````text
In the walk, they act on the address only as controls or as permutations that are undone before the erase. |
````

New:

````text
In the baseline unprepares (§9.7, §9.8), they act on the address only as controls or as permutations that are undone before the erase. A caller that X-measured the junk with reset after the read, before anything else acted on it, erases the target with `emit_erase_from_outcomes` and the junk's outcomes in wire order instead (§16, Q-1 variant (b)). The walk does so for its alias reads, measuring right after each enclosing prepare (steps 4 and 8); the comparator and the swaps in between do not touch the junk. |
````

**B-S39** (2026-11-11-dfthc-pennylane-design.md, 9.3, the adjoint-and-control row (its end)). Old:

````text
(§7.4 rule 6). The walk erases with `QROAMErase`. |
````

New:

````text
(§7.4 rule 6). Its data is erased by `QROAMErase` for a read whose junk is kept (the baseline unprepares, §9.7 and §9.8), by the unprepares from outcomes through `emit_erase_from_outcomes` for the walk's alias reads (§16, Q-1 variant (b)), and by `AngleErase` for the angle reads (§9.17). |
````

**B-S40** (2026-11-11-dfthc-pennylane-design.md, 16, Q-1, what it is). Old:

````text
  - *What it is.* Each read's junk is X-measured right after its swap network.
````

New:

````text
  - *What it is.* Each alias read's junk is X-measured early instead of at its erase: a caller may measure it right after the read's swap network, and the walk measures it right after the enclosing prepare (steps 4 and 8), before any angle load (the comparator and the swaps in between do not touch the junk).
````

**B-H1** (HANDOFF.md, section 5, rule 3). Old:

````text
   `dynamic_argnames` only for runtime data: `PhaseFixup` parities, `QPERound` earlier bits, and
   `QROAMRead`'s per-leaf table words (`int64`, built by the caller from its compilable table).
   Convert list-valued dynamic arguments to arrays in `__init__` (L-23).
````

New:

````text
   `dynamic_argnames` only for runtime data: `PhaseFixup` parities, `QPERound` earlier bits,
   `QROAMRead`'s per-leaf table words (`int64`, built by the caller from its compilable table)
   and the junk outcomes of `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`.
   Convert list-valued dynamic arguments to arrays in `__init__` (L-23).
````

**B-H2** (HANDOFF.md, section 5, rule 4). Old:

````text
   unroll loops in Python with static bounds. `qp.cond` on measurement outcomes appears only
   inside erase operators and the QPE wrapper (`QPERound`'s corrections).
````

New:

````text
   unroll loops in Python with static bounds. `qp.cond` on measurement outcomes appears only
   inside erase operators, the unprepares from outcomes and the QPE wrapper (`QPERound`'s
   corrections).
````

**B-H3** (HANDOFF.md, section 5, rule 5). Old:

````text
5. **Measurement-based uncompute** lives only in `QROAMErase`, `AngleErase` and the base rules
   of `OuterUnprepare` and `InnerUnprepare`. Never place a measuring operator under `qp.ctrl`.
````

New:

````text
5. **Measurement-based uncompute** lives only in `QROAMErase`, `AngleErase`, the base rules
   of `OuterUnprepare`, `InnerUnprepare`, `OuterUnprepareFromOutcomes` and
   `InnerUnprepareFromOutcomes`, and the block encodings' junk measurements (spec 16, Q-1
   variant (b)). Never place a measuring operator under `qp.ctrl`.
````

**B-H4** (HANDOFF.md, section 3.2, a pitfall after the wire-register one). Old:

````text
  inside the rule: `QPERound`'s corrections take the walk's `work` pool from its `walk_wires`.
````

New:

````text
  inside the rule: `QPERound`'s corrections take the walk's `work` pool from its `walk_wires`.
- **Measurement outcomes passed as a dynamic argument have a mode-specific type.** Under
  capture they are a bool array, in tape mode an object array of `MeasurementValue`s, so a rule
  that declares such an operator builds the key's abstract array with the mode's dtype
  (`dfthc.ops.prepare.abstract_junk_outcomes`) before calling `resource_key`, which pauses
  capture. With the wrong dtype Catalyst drops the declaring rule ("Decomposition rule not
  found") or tape-mode T2 reports a resource mismatch (Q1b plan dry run).
````

**B-H5** (HANDOFF.md, section 6, wires). Old:

````text
  `WalkLayout.split_group`; the blocks they call, `InnerSelect`, `InnerSelectDagger` and the
  reflections included, take the registers their spec 9 tables list by name (spec 9.0, wire
  passing).
````

New:

````text
  `WalkLayout.split_group`, and the junk registers, the start of `select`, through
  `WalkLayout.split_junk` (spec 10.2); the blocks they call, `InnerSelect`,
  `InnerSelectDagger` and the reflections included, take the registers their spec 9 tables
  list by name (spec 9.0, wire passing).
````

## Shared edits

Modified merged modules, each in a sub-task's file list: `src/dfthc/ops/qroam.py` (Task 1C.5's: `_erase_body` delegates its fix-up to a new `_fixup_from_outcomes`, which `emit_erase_from_outcomes` shares; `QROAMErase`'s emitted operators are unchanged, and its tests pass untouched), `src/dfthc/ops/prepare.py` (Tasks 2D.1 and 2D.2's: new classes and helpers; the existing classes change only in docstrings: their siblings, and both junk paths in the prepares' contracts), `src/dfthc/costs.py` (three new functions), `src/dfthc/costs_reference.py` (a new constant), `src/dfthc/registers.py` (Task 1A.1's), `src/dfthc/ops/walk.py` (Tasks 3.3 and 3.4's), `src/dfthc/qpe.py` and `src/dfthc/testing/walk_sim.py` (docstrings only), `scripts/resource_report.py` (Task 4.3's), and Task 4.4's documents and audit. Changed test literals are listed in section "Changed tests".

**Mirroring at approval (the orchestrator).** The plan writer edits no existing template; the orchestrator mirrors this plan's edits when it approves the plan, in this order (paths under `.scratch/plan-check/` unless they start with `docs/`):

1. Apply the 63 edits above to `docs/` (B-M1 to B-M18 to the master plan, B-S1 to B-S40 to the spec, B-H1 to B-H5 to HANDOFF), each old text exactly once (`stages/Q1B/make_doc_edits_q1b.py` checks and applies them; its output is `tmp/doc-edits-q1b/`).
2. Apply B-M1 to B-M14 and B-M18 to `templates/2026-11-11-dfthc-master-plan.md`, where each old text occurs exactly once (the script checks that too). B-M15 to B-M17 are not applied to any template: their old texts live in the shared fragment `templates/_global_constraints_g01.md`, which seven plans include and the QPE plan holds a copy of (`shared-fragment.md` below), and it stays unchanged so that the executed plans keep the constraints they ran under.
3. Copy `stages/Q1B/global-constraints-q1b.md` (the shared fragment with B-M15 to B-M17 applied) to `templates/_global_constraints_q1b.md`.
4. Change exactly two include lines, each a whole line of its template; only the path changes:

| Template | Line | Before | After |
| --- | ---: | --- | --- |
| `templates/2026-11-11-dfthc-master-plan.md` | 18 | \{\{include:templates/_global_constraints_g01.md\}\} | \{\{include:templates/_global_constraints_q1b.md\}\} |
| `templates/2026-11-11-dfthc-phase4-q1b.md` (this plan) | 229 | \{\{include:stages/Q1B/global-constraints-q1b.md\}\} | \{\{include:templates/_global_constraints_q1b.md\}\} |

5. Assemble (`.venv/bin/python assemble.py <out>/`) and assert that the assembled master plan equals the edited `docs/superpowers/plans/2026-11-11-dfthc-master-plan.md` byte for byte and that every other assembled plan, this one included, equals its file under `docs/superpowers/plans/`.

Replayed on copies of the templates (`stages/Q1B/verify_mirroring_q1b.py`, which applies steps 2 to 4 in `tmp/q1b-mirror/` and assembles there; `mirroring-check.log`):

```text
applied to the master-plan template: B-M1, B-M2, B-M3, B-M4, B-M5, B-M6, B-M7, B-M8, B-M9, B-M10, B-M11, B-M12, B-M13, B-M14, B-M18
2026-11-11-dfthc-master-plan.md: the include of templates/_global_constraints_g01.md now names templates/_global_constraints_q1b.md
2026-11-11-dfthc-phase4-q1b.md: the include of stages/Q1B/global-constraints-q1b.md now names templates/_global_constraints_q1b.md
identical: 2026-11-11-dfthc-master-plan.md (against main + B-M1 to B-M18)
identical: 2026-11-11-dfthc-phase0-canaries.md (against main)
identical: 2026-11-11-dfthc-phase0-infrastructure.md (against main)
identical: 2026-11-11-dfthc-phase0-spikes.md (against main)
identical: 2026-11-11-dfthc-phase1-data.md (against main)
identical: 2026-11-11-dfthc-phase1-primitives.md (against main)
identical: 2026-11-11-dfthc-phase1-simulator.md (against main)
identical: 2026-11-11-dfthc-phase2-blocks.md (against main)
identical: 2026-11-11-dfthc-phase2-prepare.md (against main)
identical: 2026-11-11-dfthc-phase3-assembly.md (against main)
identical: 2026-11-11-dfthc-phase3-select.md (against main)
identical: 2026-11-11-dfthc-phase4-q1b.md (against out-q1b)
identical: 2026-11-11-dfthc-phase4-qpe.md (against main)
identical: 2026-11-11-dfthc-phase4-report.md (against main)
```

`templates/_global_constraints_g01.md` holds the old texts of B-M15, B-M16, B-M17 once each. Every template that includes it (or a stage copy of it), and the count of each old text in its plan on `main` at `9f4c7eb`:

- `2026-11-11-dfthc-master-plan.md` (includes `templates/_global_constraints_g01.md`): B-M15 1, B-M16 1, B-M17 1 on `main`
- `2026-11-11-dfthc-phase1-primitives.md` (includes `templates/_global_constraints_g01.md`): B-M15 1, B-M16 1, B-M17 1 on `main`
- `2026-11-11-dfthc-phase2-blocks.md` (includes `templates/_global_constraints_g01.md`): B-M15 1, B-M16 1, B-M17 1 on `main`
- `2026-11-11-dfthc-phase2-prepare.md` (includes `templates/_global_constraints_g01.md`): B-M15 1, B-M16 1, B-M17 1 on `main`
- `2026-11-11-dfthc-phase3-assembly.md` (includes `templates/_global_constraints_g01.md`): B-M15 1, B-M16 1, B-M17 1 on `main`
- `2026-11-11-dfthc-phase3-select.md` (includes `templates/_global_constraints_g01.md`): B-M15 1, B-M16 1, B-M17 1 on `main`
- `2026-11-11-dfthc-phase4-qpe.md` (includes `stages/4Q/global-constraints-4q.md`): B-M15 1, B-M16 1, B-M17 1 on `main`
- `2026-11-11-dfthc-phase4-report.md` (includes `templates/_global_constraints_g01.md`): B-M15 1, B-M16 1, B-M17 1 on `main`

## How this plan meets the binding ledger rows

| Ledger row or decision | What this plan does |
| --- | --- |
| Decision "Q-1 working choice: variant (b)" (2026-11-15): keep masks derived inside rules; project Task 4.5's FeMoco compiles with the junk-outcome arrays | Masks: both rules build them from the compilable tables through `QROAMRead.loaded_bits` of `OuterPrepare.read_table` and `InnerPrepare.read_table` (`emit_erase_from_outcomes`); only outcomes are dynamic. Projections: section "Projections", made from the regenerated report, whose walks carry the junk-outcome arrays: FeMoco-54 walk 198.2 s and 3,892.5 MB, controlled walk 195.9 s and 4,399.9 MB from the small systems; 40.0 s and 1,628.3 MB, 34.3 s and 1,728.8 MB with the earlier `fe2s2` and `fe4s4` points scaled; all within 60 minutes and 7.5 GB. Back to back at `MID_NAPHTHALENE` the junk-outcome walk compiles in the same time at 3 to 5% more memory (section "Compile time and memory"). |
| Decision "Q-1 early junk measurement" (2026-11-11: decided after K-10 and the Phase 4 report; prefer variant (b)) and K-10 (row "K-10 measurement arrays": part 2 passes from inside another operator's rule) | Variant (b) as spec 16 describes it, with change 2's `extra_onehot`. No blocker for (b) was found. |
| Decision "K-10 consequences" (1) to (3) | (1) Q-1 is decided as the working choice; `QPERound`'s `earlier_bits` is unchanged. (2) Masks stay derived inside rules (above); no compile with masks as data was run. (3) `tests/ops/test_unprepare_from_outcomes.py` has a test operator, `InnerRoundTripProbe`, whose pinned rule holds `InnerPrepare`, the junk measurement and `InnerUnprepareFromOutcomes`: T2 and compiled counts at two shapes, the round trip under zero and unit schedules on the sparse simulator, and on `lightning.qubit`; the block encodings are a second such caller in Q1b.2. The fix-up's conditionals come from the compilable table in both modes (`PhaseFixup`'s rule, unchanged). |
| Decision "FeMoco compile in Phase 4" (Task 4.5 after Q1b; per-operator FeMoco rows closed forms validated at compiled sizes) | Q1b changes no FeMoco-54 count of the walk; the new rows' closed forms equal the compiled programs at the four small systems (182 programs) and at the test presets and `MID_NAPHTHALENE` (T4). Task 4.5 compiles the Q1b walk once. |
| 4.4 "For Q1b" (the review's ten forward notes, Focus 6) | Audited count 32 → 34 in Q1b.1, both classes with the nine sections. `architecture.md`: the L2 list (Q1b.1); the call tree with the junk measurements, the register groups with exact totals (H2 55, FeMoco-54 1,102), the aliasing of the junk with `angles` and `junk_overflow`, the explicit-class list (Q1b.2); the module inventory is unchanged (no new module) and the report wording ("one row per distinct operator call, plus explicit auxiliary and L4 program rows") still holds. `conventions.md`: `junk_outcomes` in `QROAMRead.junk_layout` order and their dtype (Q1b.1); the junk lifetime and `junk_overflow` (Q1b.2); the documents never said "junk persists read to erase" (spec 4.4 item 7 did; B-S2). `verification.md`: the two certificates, the rule-crossing test and the simulator with earlier measurements. Docstrings: both new classes, `OuterUnprepare`, `InnerUnprepare`, the block encodings, the walks, `WalkLayout`, `ControlledWalkPower`, `walk_sim`; the junk lifetime in `OuterPrepare`, `InnerPrepare`, `QROAMRead`, `QROAMErase` and both modules' docstrings (change 11). The regenerated report (Q1b.2) and the orchestrator's `--large` reruns before Task 4.5. Every 1,564 and 63 figure: the documents, the spec (B-S1, B-S16, B-S18), the master plan (B-M4, B-M5, B-M18), QPE totals and spare-wire boundaries (B-M8; the report tests). T, A and M on compiled evidence (section "Compiled counts"); the M attribution is documented (change 5). The Q-1 wording: the documents, the report (its measurement sentence and qubit section; no deferred-refinements bullet), the spec (B-S2, B-S22 to B-S40) and HANDOFF. Base 1,221 and the document edits against `1a3e062`: this revision. |
| 4.4 process row (per-commit tests read `docs/`) | Q1b.2's Step 12 runs `tests/test_docstrings.py` after the report is regenerated (`resources.md` is one of the linked documents), and the rebased replay ran it on both trees. |
| 2D.1 and 2D.2 qubit rows ("Persistent outer qubits 143 / 123 / 50", "Persistent inner qubits 455 / 379 / 53"; the paper measures the junk at steps 2 and 4) | The junk is measured at those steps now: outer 71 against the paper's 50, inner 65 against 53, with the causes in section "Qubits against the paper" and the report's "By group" paragraph. |
| 2D.2 and 3.1 notes for Phase 3 (the select must not write `junk_b` before `InnerUnprepare`; never pass the select `junk_b`) | The invariant changes with the junk's lifetime: `junk_b` is measured before the select, its wires are the angles' from then on, and the unprepare from outcomes takes the outcomes, not the wires. `test_junk_is_measured_right_after_each_prepare` checks the order of the rule's operations; `test_children_order_and_registers` checks which registers each child receives. The other invariants of those rows (`succ_work_b`, `cmp_work_b` dirty from prepare to unprepare, the prepares' `work` returned clean) hold unchanged and stay tested by `tests/test_work_pool.py`. |
| 1C.5 Phase 2 notes (the same compilable table and entry layout for the erase as for the read; the swap undone first; no enable in the walk; bit i of a mask belongs to `outcomes[i]`) | The unprepares from outcomes take the prepare's table and build the read's loaded contents from it; the CSWAPs come first; `emit_erase_from_outcomes` has no enable; the outcome order (target, then junk, reversed against the big-endian masks as in `QROAMErase`) is the shared `_fixup_from_outcomes`, and two mutants on it (junk before target, junk reversed) are killed. `extra_onehot`: change 2. |
| 1C.5 Minor 3 (one comparison per nonzero mask in `_capture_parities`) and 0.9d Opus M-4 (2.5 MB per bind of compilable arguments) | Unchanged: the fix-ups have the same masks and the same number of parities, and each unprepare from outcomes binds the same tables as the baseline unprepare it replaces; the outcomes are dynamic. The projections leave room (above). |
| K-1 PX design (explicit classes; the inventory test) | Both new classes are explicit siblings of the baseline unprepares, never adjointed or controlled (`test_no_adjoint_or_controlled_path` expects `CompileError`); the inventory tests pass unchanged in both per-commit tiers. |
| 4.3 notes for Task 4.5 (full-cache projections; Task 4.4 consumes the report as committed) | Q1b regenerates the report (a code change), so the orchestrator reruns `fe2s2` and `fe4s4`; section "Projections" gives their projections and Task 4.5's, from the small-system report and from the scaled earlier `fe2s2` and `fe4s4` points. |
| Heavy-runs table | The `--large fe2s2` and `--large fe4s4` reruns are orchestrator runs with ledger rows (large-compile count 0); this plan's dry run started no heavy process. |

## The design as built

- **`OuterUnprepareFromOutcomes`** (spec 9.7 row, B-S7). Wires: `OuterPrepare`'s without `junk_x_o` (`x_o`, `succ_x_o`, `succ_work_x_o`, `sigma_1`, `m_x_o`, `less_than_x_o`, `cmp_work_x_o`, `alt_x_o`, `keep_x_o`, `phase_grad`, `work`), then `extra_onehot`. Compilable: `X_o`, `alt`, `keep`, `b_k1`, `k1`, `ke1`, `s_u`. Dynamic: `junk_outcomes`, (K1 − 1) d1 outcomes in the order of the `junk_x_o` wires (slot r of `QROAMRead.junk_layout` gives outcomes r d1 to (r + 1) d1 − 1). Rule: the w_x CSWAPs; the measured comparator adjoint; X-measure `alt_x_o ‖ keep_x_o`; the parities of target ‖ junk outcomes against the masks of the read's loaded set; the fix-up with `alt_x_o ‖ keep_x_o ‖ extra_onehot` as its one-hot register; H on `m_x_o` and `sigma_1`; the adjoint uniform preparation. `required_wires(X_o, b_k1, k1, ke1, s_u)`, `junk_size(X_o, b_k1, k1)`.
- **`InnerUnprepareFromOutcomes`** (spec 9.8 row, B-S8). The same for `InnerPrepare` without `junk_b`: compilable `N`, `R`, `B`, `C`, `alt`, `keep`, `sgn_table`, `d_sgn_table`, `b_k2`, `k2`, `ke2`, `s_u`; (K2 − 1) d2 junk outcomes. `required_wires(N, R, B, C, b_k2, k2, ke2, s_u)`, `junk_size(N, R, B, C, b_k2, k2)`.
- **Never adjointed or controlled** (spec 7.4 rule 5): they measure. Their docstrings have spec 9.0's nine sections, cite the paper's App. B.7.2 steps 2 and 4 and dfthcbe's steps, and compile an example.
- **The block encoding's rule** (spec 9.21, B-S9, B-S10): `OuterPrepare`; H and `qp.measure(q, reset=True)` on every `junk_x_o` qubit; `InnerPrepare`; the same on `junk_b`; `InnerSigns`, `InnerSelect`, `InnerUnprepareFromOutcomes` (round 1's outcomes); `RefT2`; `InnerPrepare`, the measurement of `junk_b` again; `InnerSelectDagger`, `InnerSigns`, `InnerUnprepareFromOutcomes` (round 2's outcomes); `OuterUnprepareFromOutcomes`. `_children` returns 14 steps (`walk.Step(op, args, work, extra_onehot)`); a junk step's op is `walk.JUNK_MEASUREMENT`. The controlled block encoding has the same rule with `ControlledRefT2`.
- **`WalkLayout`** (spec 10.1, 10.2): `GROUPS["outer"]` and `GROUPS["inner"]` without the junk; `GROUPS["select"] = ("angles", "junk_overflow", "G0", "G1", "b_neq_B", "sg")`; `registers.JUNK = ("junk_x_o", "junk_b")`; `WalkLayout.junk_sizes(params)`; `WalkLayout.split_junk(params, select)` returns the junk slices of a `select` group; the attributes `junk_x_o` and `junk_b` are those slices.

## Qubits against the paper

`WalkLayout` gives these walks (`probe-layout-sizes.log`; the report prints the same):

| System | Before | Junk wires (outer + inner) | After | Notes |
| --- | ---: | ---: | ---: | --- |
| `H2_STO3G_TEST` | 63 | 4 + 7 | 55 | 7 > 4 angle wires: `junk_overflow` 3, so 63 − 11 + 3 |
| `H2_631G_TEST` | 79 | 5 + 7 | 67 | |
| `FAILS_C3` | 58 | 0 + 0 | 58 | k1 = k2 = 0: no junk |
| `MID_NAPHTHALENE`, `naphthalene` | 426 | 21 + 69 | 336 | |
| `h2_sto3g_R2_B2_C1`, `h2_631g_R2_B2_C2`, `h2_631g_R2_B3_C3` | 178, 215, 219 | 0 | unchanged | k1 = k2 = 0 |
| `fe2s2` | 714 | 22 + 175 | 517 | |
| `fe4s4` | 1,004 | 23 + 189 | 792 | |
| FeMoco-54 | 1,564 | 72 + 390 | **1,102** | 72 = 3 · 24, 390 = 15 · 26 |

At FeMoco-54 against the paper's 1,131 (its Table 3, under our group names; `PAPER_QUBIT_GROUPS_FEMOCO`): `system` 108 against 108; `outer` 71 against 50, +21: the dirty comparator and uniform-preparation work registers `cmp_work_x_o` (14) and `succ_work_x_o` (6), which make the comparator adjoint free (`cmp-free`, 15 Toffolis per outer unprepare; ledger row 2D.1), and the keep test's `less_than_x_o` (1); `inner` 65 against 53, +12: `cmp_work_b` (14) and `succ_work_b` (2), and 7 wires for `less_than_b`, `sgn`, `d_sgn` and `r` against the paper's 11 for G, r and c (−4); `select` 799 against 795, +4: the flags `G0`, `G1`, `b_neq_B` and `sg` beside the 795 angle wires; `phase_grad` 15 against 15; `work` 44 against 110, −66: our shared pool's largest need (the controlled `RefAB`'s 44) against the paper's outer-prepare temporaries, which it adds because only its inner junk shares the angle qubits, while ours both do. Total −29. The report's qubit section gives this table and paragraph.

## Compiled counts

Every compiled program equals its closed form:

- **The operators** (`probe-from-outcomes.log`; T4 in the tests): at `H2_STO3G_TEST`'s shapes the outer unprepare from outcomes T 2, M 4, A 2 (closed form T 2, M 4) and the inner one T 14, M 7, A 15 (T 14, M 7); a prepare, the junk measurement and the unprepare in one program T 10, M 8 (outer) and T 35, M 14 (inner); the rule-crossing operator T 35, M 14, A 24, equal to the inner pair's closed form. `test_counts_match_the_closed_forms` covers six shapes per commit, `test_counts_at_mid_naphthalene` the two at `MID_NAPHTHALENE` (nightly).
- **The walks** (back to back in `compare-compiles.raw`, and the tests): `H2_STO3G_TEST` walk 158 / 50 / 108 and controlled walk 160 / 50 / 110 (T / M / A); `H2_631G_TEST` 240 / 68 / 179 and 242 / 68 / 181; `MID_NAPHTHALENE` 1,913 / 502 / 1,678 and 1,915 / 502 / 1,680: the baseline's counts exactly, on 8, 12 and 90 fewer wires. `FAILS_C3` (no junk) is unchanged.
- **The report**: all 182 programs of the four small systems are measured and equal their closed forms in T, M and A (`report-small-systems.log`); the partition rows plus the junk measurements give the walks exactly at every system; the reason codes sum to every delta.
- **FeMoco-54 on the closed forms** (per commit): walk 9,793 / 2,524 / 8,675, controlled walk 9,795 / 2,524 / 8,677; the outer unprepare from outcomes T 69, M 24; the inner T 217, M 26, `extra_onehot` 102; junk 72 and 390 wires, 852 junk measurements per walk.

## Compile time and memory

Back to back on the same machine state, the baseline code (`repo-phase4b-base`, the junk kept to the erase) against Q1b's (the junk measured in the rule, its outcomes passed as arrays), one program per watched subprocess (`compare-compiles.raw`; compile seconds, peak MB):

| Program | Baseline | Q1b |
| --- | --- | --- |
| `H2_STO3G_TEST` walk | 3.02 s, 340.8 MB | 2.43 s, 358.6 MB |
| `H2_STO3G_TEST` controlled walk | 2.31 s, 328.5 MB | 2.36 s, 341.4 MB |
| `H2_631G_TEST` walk | 2.51 s, 353.5 MB | 2.57 s, 365.8 MB |
| `H2_631G_TEST` controlled walk | 2.48 s, 338.1 MB | 2.51 s, 350.5 MB |
| `MID_NAPHTHALENE` walk | 6.76 s, 610.3 MB | 6.77 s, 630.9 MB |
| `MID_NAPHTHALENE` controlled walk | 6.63 s, 579.1 MB | 6.66 s, 603.5 MB |
| `MID_NAPHTHALENE` inner unprepare (baseline) / from outcomes | 2.07 s, 299.0 MB | 1.99 s, 286.9 MB |

The junk-outcome walk compiles in the same time (within 0.1 s, except the session's first compile, the baseline `H2_STO3G_TEST` walk; the script alternates which replica compiles first) at 3 to 5% more memory (+20.6 MB and +24.4 MB at `MID_NAPHTHALENE`); the unprepare from outcomes alone is lighter than the baseline unprepare, which measures 92 qubits there against 23. In the regenerated report the `naphthalene` walk took 7.36 s at 612.5 MB against 7.71 s at 614.9 MB on `main`, the controlled walk 7.26 s at 608.8 MB against 8.29 s at 585.8 MB (section "Dry-run results").

## Dry-run results

The report of the H2 systems and `naphthalene`, regenerated under Q1b's code hash (`report-small-systems.log`: 182 programs, all measured, exit 0, 664 s, process tree at most 795 MB):

| System | Rows (main, Q1b) | Walk qubits (main, Q1b) | Junk wires | Walk T / M / A measured | Walk s, MB (main) | Walk s, MB (Q1b) | Controlled walk s, MB (main) | Controlled walk s, MB (Q1b) |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| `h2_sto3g_R2_B2_C1` | 48, 46 | 178, 178 | 0 | 342 / 93 / 310 | 8.14, 468.2 | 4.71, 471.9 | 7.55, 447.7 | 4.12, 444.4 |
| `h2_631g_R2_B2_C2` | 48, 46 | 215, 215 | 0 | 611 / 154 / 569 | 4.93, 486.0 | 4.38, 484.5 | 4.67, 460.0 | 4.04, 458.5 |
| `h2_631g_R2_B3_C3` | 48, 46 | 219, 219 | 0 | 636 / 155 / 592 | 5.0, 480.4 | 4.51, 476.1 | 4.56, 464.1 | 4.24, 456.5 |
| `naphthalene` | 46, 44 | 426, 336 | 90 | 1,913 / 502 / 1,678 | 7.71, 614.9 | 7.36, 612.5 | 8.29, 585.8 | 7.26, 608.8 |
| `fe2s2` | 46, 44 | 714, 517 | 197 | 4,414 / 1,020 / 3,916 (closed form) | 17.91, 993.1 | — | 17.92, 1011.7 | — |
| `fe4s4` | 46, 44 | 1,004, 792 | 212 | 6,534 / 1,534 / 5,945 (closed form) | 20.47, 1015.5 | — | 19.08, 994.0 | — |
| `femoco54` | 49, 47 | 1,564, 1,102 | 462 | 9,793 / 2,524 / 8,675 (closed form) | — | — | — | — |

At the four small systems all 182 programs are measured and equal their closed forms (T, M and A), in 0.68 to 7.45 s and at most 612.5 MB each; code hash `5e3eed1e7a95...` (main: `2e5482adb928...`). The `fe2s2` and `fe4s4` rows show their closed forms until the orchestrator's reruns; their earlier measurements stay in the json as history (282 entries of another code hash).

FeMoco-54: 1,102 wires (system 108, outer 71, inner 65, select 799, phase_grad 15, work 44), junk 462 wires, 852 junk measurements per walk; QPE wires t = 1, 2, 3, 10, 20, 23: 1,104, 1,105, 1,106, 1,113, 1,123, 1,127.

The FeMoco-54 summary in `docs/resources.md` states: "852 of the M are the block encoding's own junk measurements, right after each prepare (spec 16, Q-1 variant (b)); the partition rows hold the rest", the qubit table of section "Qubits against the paper" with its "By group" paragraph, and QPE qubits 1,104, 1,105, 1,106, 1,113, 1,123 and 1,127 at t = 1, 2, 3, 10, 20, 23. Its deferred refinements are truncated iteration (Q-4), the retained swap ANDs and the 4-bit SF one-hot.

## Projections

Projected through the report's own `gates`, `family` and `project_compile`, from the regenerated report's entries, with nothing compiled at these sizes (`project_large_q1b.py`):

| Calibration | `fe2s2` (44 programs) | `fe4s4` (44 programs) | FeMoco-54 walk | FeMoco-54 controlled walk |
| --- | --- | --- | --- | --- |
| `main`, small systems only (190 entries; `projections-baseline-small-only.log`) | all allowed; at most 1,424 MB, 66.5 s | all allowed; at most 2,303 MB, 127.0 s | 209.3 s, 3,950.2 MB | 224.9 s, 3,724.8 MB |
| `main`, with the measured `fe2s2` and `fe4s4` (282 entries; `projections-baseline-full.log`, the ledger's 42.2 s and 1,635.9 MB, 39.4 s and 1,641.6 MB) | all allowed; at most 1,046 MB, 22.3 s | the same | 42.2 s, 1,635.9 MB | 39.4 s, 1,641.6 MB |
| **Q1b, small systems only** (182 entries; `projections-q1b-small-only.log`) | **all allowed; largest peak the controlled walk 1,528.8 MB, longest `ControlledWalkPower(p=1)` 57.3 s; 11.2 min in total** | **all allowed; 2,527.0 MB, 109.4 s; 20.7 min** | **198.2 s, 3,892.5 MB** | **195.9 s, 4,399.9 MB** |
| Q1b, estimate of the reruns (the earlier `fe2s2` and `fe4s4` measurements scaled row by row by the measured `naphthalene` ratio, Q1b over `main`; 84 scaled entries; `projections-q1b-scaled-large.log`) | all allowed; at most 1,051 MB, 21.1 s; 5.0 min | all allowed; the same; 4.8 min | 40.0 s, 1,628.3 MB | 34.3 s, 1,728.8 MB |

- **The orchestrator's `fe2s2` and `fe4s4` reruns** project at most 1.53 GB and 57 s per program at `fe2s2` and 2.53 GB and 109 s at `fe4s4` (the conservative row), within 7.5 GB and 60 minutes; the estimate is about 1.05 GB and 21 s per program. On `main` they measured 286 s and 1,175 MB (`fe2s2`) and 320 s and 1,216 MB (`fe4s4`) for 46 programs each. Expected results: 44 rows each, all `compiled` and equal to their closed forms; walks T / M / A 4,414 / 1,020 / 3,916 (`fe2s2`) and 6,534 / 1,534 / 5,945 (`fe4s4`), unchanged; controlled walks + 2 T and + 2 A; walk qubits 517 and 792; 372 and 401 junk measurements per walk.
- **Task 4.5's FeMoco-54 compiles** project to 198 s and 3.9 GB (walk) and 196 s and 4.4 GB (controlled walk) from the small systems alone, and to about 40 s and 1.6 GB and 34 s and 1.7 GB once the reruns' points join; both within 60 minutes and 7.5 GB. The small-only row moves little from `main`'s (209.3 s and 3,950.2 MB, 224.9 s and 3,724.8 MB), so Q1b's 3 to 5% memory shows within the projection's noise; Task 4.5 projects again from the committed report after the reruns, as its procedure says.
- The four rows of the unprepares from outcomes have no earlier `fe2s2` or `fe4s4` counterpart to scale (the family is new), so the estimate leaves them out; the small-only projection includes them.

## E-1 to E-4 and the certificates with earlier measurements

Nothing in the simulator or the certificates changes; both already handle measurements anywhere in the program:

- **Schedules.** `walk_sim`'s forced schedules (all zeros, all ones, alternating, seeded) assign outcomes in program order; the junk outcomes now come right after each prepare, the target outcomes in the unprepare. The total number of measurements per walk is unchanged (2,524 at FeMoco-54; 50 at `H2_STO3G_TEST`), so each schedule keeps its meaning. Strict mode asserts probability 1/2 for every X measurement, junk ones included.
- **Conditionals.** A fix-up's predicate is a parity of `MeasurementValue`s of both measurements; the simulator evaluates it on the recorded outcomes when the conditional runs, as it did for `QROAMErase`'s own outcomes.
- **Certificates (spec 12.5, B-S17).** The GF(2) certificate checks that every conditional is diagonal with a linear predicate (the premise check), then runs the zero schedule and the unit schedules. The new certificates `test_outer_unprepare_from_outcomes_certificate` and `test_inner_unprepare_from_outcomes_certificate` (4 shapes each, nightly) run a prepare, the X measurement of its junk and the unprepare from outcomes, with ξ over the junk outcomes and the target outcomes together, from the same random inputs as the baseline pair's certificates.
- **Evidence.** E-1 to E-4 pass on the Q1b walk (`10 passed`, 70 s, in the first replay; the nightly tier `86 passed`, 443 s, 1,872 MB). Three release-tier inputs, run once on the replica: E-1 under S1 on all 16 `sto3g` inputs (6 s), E-1 on one `H2_631G_TEST` input (26 s, the preset where the inner unprepare lends a wire) and E-3 under S1 on all 16 (80 s): `3 passed` (`release-sample.log`, 619 MB).

## Docstring audit

`tests/test_docstrings.py` audits 34 classes after Q1b.1 (32 on `main`): both new classes have a summary line and spec 9.0's nine sections in order, their cost sections name `dfthc.costs.outer_unprepare_from_outcomes` and `dfthc.costs.inner_unprepare_from_outcomes`, their section 3 is titled "Arguments" (they take a dynamic argument, as `QROAMRead` and `PhaseFixup` do; the test's `SECTIONS` docstring says so), and their examples call `compiled_resources`. The architecture names both (`test_architecture_names_every_operator`).

## Mutants

Seven targeted mutants of the production code, each run against the tests that should catch it (`q1b-mutants.log`), all killed: the junk outcomes before the target outcomes (`qroam.py`); the outer junk outcomes read in reversed order (`prepare.py`); the second inner round reusing the first round's junk outcomes (`walk.py`); `extra_onehot` lent from the end of `select` (`walk.py`); no `junk_overflow` register (`registers.py`); the junk measurements of one inner round only (`costs.py`); the target measurements left out of the declaration (`prepare.py`).

## Changed tests

Q1b.1 changes one existing test file's literals and adds tests:

- `tests/test_docstrings.py`: the audited count 32 → 34 and both class names (the new classes); the `SECTIONS` docstring names the classes that title section 3 "Arguments".
- `tests/ops/test_certificates_prepare.py`: a `from_outcomes` helper and two parametrized certificate tests (8 nightly); the existing certificates are unchanged.
- `tests/ops/test_unprepare_from_outcomes.py` (new): 42 per commit, 2 nightly.

Q1b.2 changes literals where the layout, the rule or the report changed:

- `tests/test_registers.py`: group sizes at FeMoco-54 (`outer` 143 → 71, `inner` 455 → 65, `num_wires` 1,564 → 1,102) and `H2_STO3G_TEST` (63 → 55); the error messages ("outer needs 12 wires; got 11", "select needs 11 wires; got 10"); the QPE register positions (55, then 56 to 58); new `test_junk_registers_alias_the_start_of_the_angle_wires` (4 presets); the two `split_group` tests renamed `test_split_helpers_*`, covering `split_junk` too.
- `tests/ops/test_block_encoding.py`: the children (the junk measurements and the unprepares from outcomes; `NAMES`, `JUNK_OF`); the required wires (12, 16, 11, 4, 4, 8 at `H2_STO3G_TEST`); the FeMoco-54 closed forms with the junk measurements (852) and the group sizes 71, 65, 799; new `test_junk_is_measured_right_after_each_prepare` (`sto3g`, `631g`).
- `tests/ops/test_walk.py`: the walk's group sizes and the error message ("expected 12").
- `tests/test_work_pool.py`: the steps are `walk.Step` tuples and a junk step names a junk register; every call still gets exactly its required wires (the unprepares from outcomes no junk register); the lent `extra_onehot` wires are the start of the matching junk register (`LENDS`), and each step's `extra_onehot` is no more than that register holds.
- `tests/test_qpe.py`: group sizes 12, 16, 11; `walk_wires` totals 55 and 56; FeMoco-54 total 1,102; the spare wire at t = 5 is 61 (was 69); control 55 and `qpe_grad` from 56.
- `tests/report/test_report.py`: `test_report_h2` has 46 rows (the two `QROAMErase@...` rows go, the unprepares from outcomes and their adjoint uniform preparations come in: 48 → 46 parametrizations); the partition names; the walk wires 55, 56, 59; the steps and Clifford kinds of the new rows; 44 rows at `naphthalene`, `fe2s2` and `fe4s4`; QPE wires 59, 60, 62 at t = 3, 4, 5 and 1,124, 1,125, 1,127 at t = 21, 22, 23; new `test_qubit_summary_at_femoco`.

## Expected test counts

The per-commit tier (`scripts/test.sh -m "not slow and not release" -q`) counts **1221 tests on `main` at `9f4c7eb`** (measured: `1221/1331 tests collected (110 deselected)`), the nightly tier 76, the release tier 34. The pass criterion is "no failures, no errors, no warnings summary"; the number is a cross-check, and the orchestrator shifts the base if other tasks merge first.

| Task | Base | Own per-commit | Expected per-commit tier | Own nightly and release tests |
| --- | ---: | ---: | ---: | --- |
| Q1b.1 | 1221 | 44 (`test_unprepare_from_outcomes.py` 42, `test_docstrings.py` + 2) | 1265 = 1221 + 44 | 10 nightly (`test_certificates_prepare.py` + 8, `test_unprepare_from_outcomes.py` 2): 86 = 76 + 10 |
| Q1b.2 | 1265 | 5 (`test_registers.py` + 4, `test_block_encoding.py` + 2, `test_report.py` − 1: `test_report_h2` 48 → 46 and `test_qubit_summary_at_femoco` + 1) | 1270 = 1265 + 5 | none: nightly 86, release 34 |
| 4.5 (report file) | 1270 | 1 | 1271 = 1270 + 1 (was 1222) | 1 release |

**Task 4.5's counts after Q1b (the orchestrator).** The report plan (revision 7) states Task 4.5's per-commit base as 1221 (its "Expected test counts" table) and its expectation as `1222 passed` (the table, and Task 4.5's Step 3, "Run the per-commit tier and the linter, commit, and stop": "Expected: `1222 passed` (1221 + 1)"). At Q1b's merge the orchestrator revises the report plan as its revision 8, changing only those numbers: base 1221 → 1270 and expected per-commit tier 1222 → 1271, in the table and in Step 3 ("Expected: `1271 passed` (1270 + 1)"), with the nightly and release counts unchanged (nightly 86 after Q1b; release 34 + 1, `test_femoco_cache`). Task 4.5's dispatch states the Q1b counts (base 1270, expected `1271 passed`) whatever the report plan's revision at that time.

Measured in the replays (before revision 4's two tests): collection `1263/1383 tests collected (120 deselected)` after Q1b.1 and `1268/1388 tests collected (120 deselected)` after Q1b.2 (nightly 86, release 34; `q1b-rebase.log`, and again on `9f4c7eb` in `q1b-rev2.log` and `q1b-rev3.log`). Revision 3's replay on `9f4c7eb` (the final files; revision 2's gave the same counts): Q1b.1's RED `1 error` and `1 failed, 45 passed`, GREEN `197 passed` (36 s, 1,622 MB) and nightly `25 passed`; the tests of the modules that cover `dfthc.ops.qroam` and `dfthc.ops.prepare` (`test_docstrings.py`, `test_unprepare_from_outcomes.py`, `test_outer_prepare.py`, `test_inner_prepare.py`, `test_qroam_read.py`, `test_qroam_erase.py`, `test_phase_fixup.py`, per commit) `266 passed` on the Q1b.1 tree (49 s, 1,816 MB) and on the Q1b.2 tree (50 s, 1,827 MB); ruff clean on both; the Q1b.2 tree's code hash `5e3eed1e7a95...`, the regenerated report's. The per-commit tiers ran in the first replay, on `c157211` with revision 6's `tests/test_docstrings.py` (4 tests fewer, the review's link tests): `1259 passed in 371.25s` (373 s, 1,859 MB) after Q1b.1 and `1264 passed in 482.98s` (486 s, 1,975 MB) after Q1b.2, both without a warnings summary, inside the 15-minute budget; `tests/test_docstrings.py` then passed on both rebased trees (`48 passed`). Task 4.5's per-commit test passes on the Q1b tree (`task45-test-on-q1b.log`). The nightly tier, run once on the Q1b.2 replica: `86 passed in 442.72s` (1,872 MB), against its 4-hour budget.

## Risks and open questions

1. **The large reruns are projected, not measured.** If a `fe2s2` or `fe4s4` compile is killed or fails, `--large` stops that system's run, records the rest as skipped, and the orchestrator escalates before committing anything; the rows stay their closed forms.
2. **Task 4.5's FeMoco-54 projection from the small systems alone is 4.4 GB** for the controlled walk, against the full-cache projection's 1.6 to 1.7 GB; both are within 7.5 GB, and the full-cache value applies once the reruns are committed (ledger row 4.3, notes for Task 4.5).
3. **The user may override the working choice** before Q1b runs (decision "Q-1 working choice: variant (b)"). Then the ledger sets Q1b to `not needed`, and none of this plan's edits applies.
4. **The base moves.** The plan is against `main` at `9f4c7eb`; if other tasks merge first, the orchestrator shifts the counts and reapplies the edits (each old text was checked once at `9f4c7eb`).
5. **The release tier is not run here** (master plan section 12.9; Task 5.2 runs it once). Three of its inputs pass on the Q1b walk (section "E-1 to E-4"); the rest run in Task 5.2.
6. **No open question blocks the plan.**

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
3. Every argument that determines circuit structure is in `compilable_argnames`; `static_argnames` are not used; `dynamic_argnames` only for runtime data (`PhaseFixup` parities, `QPERound` earlier bits, `QROAMRead`'s per-leaf table words, the junk outcomes of `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`).
4. Bodies of auto-adjointed operators contain no `qp.for_loop` or `qp.cond`; `qp.cond` on mid-circuit measurement outcomes only in erase operators, the unprepares from outcomes and the QPE wrapper; `qp.for_loop` only in `QROAMRead` (one loop per unary leaf, with `qp.cond` on table bits), `ControlledWalkPower` and `ToyControlledWalkPower`, all three never adjointed or controlled.
5. Measurement-based uncompute lives only in `QROAMErase`, `AngleErase`, the base rules of `OuterUnprepare`, `InnerUnprepare`, `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`, and the junk measurements of the block encodings' rules (spec 16, Q-1 variant (b)); an operator that contains a measurement is never placed under `qp.ctrl`.
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
| `src/dfthc/ops/qroam.py` | Q1b.1 | `emit_erase_from_outcomes`, `_fixup_from_outcomes` (shared with `QROAMErase`'s body) |
| `src/dfthc/ops/prepare.py` | Q1b.1 | `OuterUnprepareFromOutcomes`, `InnerUnprepareFromOutcomes`, their rules, `abstract_junk_outcomes`, `_junk_outcomes` |
| `src/dfthc/costs.py` | Q1b.1, Q1b.2 | `outer_unprepare_from_outcomes`, `inner_unprepare_from_outcomes`; `junk_measurements` |
| `tests/ops/test_unprepare_from_outcomes.py` | Q1b.1 | the operators' tests (42 per commit, 2 nightly) |
| `tests/ops/test_certificates_prepare.py` | Q1b.1 | the two certificates (8 nightly) |
| `tests/test_docstrings.py` | Q1b.1 | the audit at 34 classes |
| `src/dfthc/registers.py` | Q1b.2 | `GROUPS` without junk, `junk_overflow`, `JUNK`, `junk_sizes`, `split_junk` |
| `src/dfthc/ops/walk.py` | Q1b.2 | the block encodings' rules with the junk measurements; `Step`, `JUNK_MEASUREMENT`, `FROM_OUTCOMES` |
| `src/dfthc/costs_reference.py` | Q1b.2 | `PAPER_QUBIT_GROUPS_FEMOCO` |
| `src/dfthc/qpe.py`, `src/dfthc/testing/walk_sim.py` | Q1b.2 | docstrings (wire counts) |
| `scripts/resource_report.py` | Q1b.2 | the new rows, `PhaseFixup` rows from them, the junk measurements in the cross-check, the qubit section |
| `tests/test_registers.py`, `tests/ops/test_block_encoding.py`, `tests/ops/test_walk.py`, `tests/test_work_pool.py`, `tests/test_qpe.py`, `tests/report/test_report.py` | Q1b.2 | changed literals and three new tests |
| `docs/architecture.md`, `docs/conventions.md` | Q1b.1, Q1b.2 | the classes; the walk, the layout, the junk |
| `docs/verification.md` | Q1b.2 | the certificates and the simulator |
| `docs/resources.md`, `docs/resources.json` | Q1b.2 | generated |

**Shell conventions.** Run commands from the worktree root of your lane. `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` locates the shared environment, as in the earlier plans.

---

### Task Q1b.1: The unprepares from outcomes

**Task id:** `Q1b.1` · **Branch:** `phase4/Q1b.1` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 4.4 and the Q-1 decision (the working choice, variant (b))

**Files:**
- Create: `tests/ops/test_unprepare_from_outcomes.py`
- Modify: `src/dfthc/ops/qroam.py`, `src/dfthc/ops/prepare.py`, `src/dfthc/costs.py`, `tests/ops/test_certificates_prepare.py`, `tests/test_docstrings.py`, `docs/architecture.md`, `docs/conventions.md`

**Interfaces:**
- Consumes: `OuterPrepare`, `InnerPrepare` (`required_wires`, `read_shape`, `read_table`), `QROAMRead.junk_layout` and `loaded_bits`, `QROAMEmitter`, `dfthc.compile.mcm_parities` and `emit_phase_fixup` (through `qroam`), `dfthc.costs.outer_unprepare` and `inner_unprepare`.
- Produces: `OuterUnprepareFromOutcomes(junk_outcomes, X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, phase_grad, work, extra_onehot)` and `InnerUnprepareFromOutcomes(junk_outcomes, N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, phase_grad, work, extra_onehot)`, each with static `required_wires(...)` and `junk_size(...)`; `prepare.abstract_junk_outcomes(size) -> AbstractArray`; `qroam.emit_erase_from_outcomes(junk_outcomes, *, table, address_widths, live_ranges, width, block_bits, onehot_bits, address, target, work, extra_onehot, emit=None)`; `costs.outer_unprepare_from_outcomes(p)`, `costs.inner_unprepare_from_outcomes(p)`. Q1b.2 calls them from the block encodings.

**Acceptance tests** (44 per commit, 10 nightly):
- `test_round_trip_after_the_junk_measurement[...]` (9): prepare, H and measure on the junk, unprepare from outcomes, on the sparse simulator in strict mode at four outer and five inner shapes (among them `inner-small`: N = 2, R = 1, B = 1, C = 1) under the alternating and a seeded schedule: the inputs' random state comes back, every other register is |0⟩ and the phase gradient unchanged; where the junk has more than one qubit, the outcomes passed in reversed order fail under the zero or some unit schedule (`basis_schedules`), so their order is tested.
- `test_lightning_round_trip_with_two_seeds`: the same on `lightning.qubit` with two seeds.
- `test_lightning_round_trip_with_postselected_junk[outer-sto3g, inner-small]`: the compiled round trip with the junk measurements postselected on 1, 0, ..., 0, so the capture path of the junk outcomes is checked for every seed (revision 4).
- `test_caller_inside_another_rule_is_valid_and_counted[inner-sto3g, inner-small]`, `test_caller_inside_another_rule_restores_every_outcome` (zero and unit schedules at `small`), `test_caller_inside_another_rule_on_lightning`: `InnerRoundTripProbe`, a test operator whose pinned rule holds the inner prepare, the junk measurement and the unprepare from outcomes (decision "K-10 consequences" (3)).
- `test_rules_are_valid[...]` (6, T2 in tape mode with `MeasurementValue` outcomes), `test_counts_match_the_closed_forms[...]` (6, T4), `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` (T5, T6), `test_no_adjoint_or_controlled_path[adjoint, control]` (`CompileError`, on the inner class), `test_required_wires_and_junk_sizes_by_hand`, `test_constructors_validate[...]` (9: too few outcomes, booleans in tape mode, an abstract array of the wrong shape, a wrong `extra_onehot` or `r` width, a one-hot too large for its reset wires, an out-of-domain `ke2`), `test_femoco_counts_on_the_closed_forms` (69 / 24 and 217 / 26, `extra_onehot` 0 and 102, junk 72 and 390).
- Nightly: `test_counts_at_mid_naphthalene[outer, inner]`; `test_outer_unprepare_from_outcomes_certificate[...]`, `test_inner_unprepare_from_outcomes_certificate[...]` (4 each).
- `tests/test_docstrings.py`: 48 tests (34 classes).

**Definition of done:** RED seen; the targeted tests pass; the task's nightly tests pass; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_unprepare_from_outcomes.py`:

```python
"""OuterUnprepareFromOutcomes and InnerUnprepareFromOutcomes (spec 9.7, 9.8 and 16, Q-1 variant
(b)): the round trip after the junk measurement, the order of the junk outcomes, a caller inside
another operator's rule, counts, rule selection, checks.

A pair is a prepare, ``H`` and ``qp.measure(q, reset=True)`` on every junk qubit right after it,
and the unprepare from those outcomes; the unprepare's ``extra_onehot`` is the first wires of the
measured junk register, as the block encoding lends them. The tables are random. The certificates
over every outcome are in tests/ops/test_certificates_prepare.py (nightly).
"""

import re
import warnings
from typing import NamedTuple

import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError
from pennylane.core import Operator2
from pennylane.ops import Conditional
from pennylane.typing import AbstractArray, Wire

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources, register_rule, resource_key
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.prepare import (
    InnerPrepare,
    InnerUnprepareFromOutcomes,
    OuterPrepare,
    OuterUnprepareFromOutcomes,
    _inner_unprepare_from_outcomes,
    _outer_unprepare_from_outcomes,
    abstract_junk_outcomes,
)
from dfthc.ops.qroam import QROAMErase
from dfthc.params import (
    FEMOCO54,
    H2_631G_TEST,
    H2_STO3G_TEST,
    MID_NAPHTHALENE,
    WalkParams,
    lg,
    odd_part,
)
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program
from dfthc.testing.sparse_sim import (
    ForcedPolicy,
    SparseSimulator,
    SparseState,
    nonlinear_conditionals,
)
from dfthc.testing.tape import decompose_for_simulation


def shape(N, R, B, C, b_k1, b_k2, k1, k2, ke1, ke2):
    return WalkParams(N=N, R=R, B=B, C=C, n_plus=1, b_rot=4, b_k1=b_k1, b_k2=b_k2, s_u=3, k1=k1,
                      k2=k2, ke1=ke1, ke2=ke2)


OUTER = {"sto3g": H2_STO3G_TEST, "631g": H2_631G_TEST,
         "x-o-5": shape(2, 3, 1, 1, 3, 2, 1, 1, 1, 1),
         "x-o-7-k0-b1": shape(3, 2, 1, 2, 1, 2, 0, 1, 2, 1)}
"""Both test presets (extra_onehot 0); X_o = 5 (l = 5: the uniform preparation fails, with a
phase); X_o = 7 with k1 = 0 (no junk: an empty outcome array) and b_k1 = 1 (no comparator
work)."""
INNER = {"sto3g": H2_STO3G_TEST, "631g": H2_631G_TEST,
         "6-2": shape(3, 2, 5, 1, 2, 2, 1, 2, 1, 1),
         "7-0": shape(3, 2, 6, 1, 2, 2, 1, 0, 1, 0),
         "small": WalkParams(N=2, R=1, B=1, C=1, n_plus=1, b_rot=3, b_k1=1, b_k2=1, s_u=3, k1=0,
                             k2=1, ke1=0, ke2=0)}
"""Both test presets (extra_onehot 1); rows [0, 2) of a 3-bit x_o with (B + 1, k2) = (6, 2)
(dead positions, failed b, l = 3); (B + 1, k2) = (7, 0) (no junk, l = 7); the small shape of the
caller-inside-a-rule tests (R = 1: no rank bits; 20 wires)."""


class Pair(NamedTuple):
    """A prepare and its unprepare from outcomes on one wire layout."""

    params: WalkParams
    prepare: type
    unprepare: type
    args: tuple
    prepare_wires: dict
    unprepare_wires: dict
    junk: str
    """The prepare's junk register."""
    measured: int
    """K d: every qubit of the read's target and junk."""
    inputs: list
    """The read-only input wires (``x_o`` of the inner pair)."""
    num_wires: int


def layout(sizes):
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def lend(unprepare, prepare_wires, junk, structure):
    """The unprepare's registers: the prepare's, and ``extra_onehot`` from the start of the
    measured junk register."""
    sizes = unprepare.required_wires(*structure)
    wires = {name: prepare_wires[name] for name in sizes if name != "extra_onehot"}
    return wires | {"extra_onehot": prepare_wires[junk][:sizes["extra_onehot"]]}


def outer_structure(p):
    return (p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)


def inner_structure(p):
    return (p.N, p.R, p.B, p.C, p.b_k2, p.k2, p.ke2, p.s_u)


def outer(p, seed=5):
    rng = np.random.default_rng(seed)
    args = (p.X_o, tuple(int(a) for a in rng.integers(0, p.X_o, p.X_o)),
            tuple(int(k) for k in rng.integers(0, 2**p.b_k1, p.X_o)), p.b_k1, p.k1, p.ke1, p.s_u)
    wires, num_wires = layout(OuterPrepare.required_wires(*outer_structure(p)))
    return Pair(p, OuterPrepare, OuterUnprepareFromOutcomes, args, wires,
                lend(OuterUnprepareFromOutcomes, wires, "junk_x_o", outer_structure(p)),
                "junk_x_o", p.K1 * p.d1, [], num_wires)


def inner(p, seed=5):
    rng = np.random.default_rng(seed)
    size = (p.RC, p.B + 1)
    columns = [rng.integers(0, p.B + 1, size), rng.integers(0, 2**p.b_k2, size),
               rng.integers(0, 2, size), rng.integers(0, 2, size)]
    tables = tuple(tuple(tuple(int(v) for v in row) for row in column) for column in columns)
    args = (p.N, p.R, p.B, p.C, *tables, p.b_k2, p.k2, p.ke2, p.s_u)
    wires, num_wires = layout(InnerPrepare.required_wires(*inner_structure(p)))
    return Pair(p, InnerPrepare, InnerUnprepareFromOutcomes, args, wires,
                lend(InnerUnprepareFromOutcomes, wires, "junk_b", inner_structure(p)), "junk_b",
                p.K2 * p.d2, wires["x_o"], num_wires)


PAIRS = {f"outer-{name}": (outer, p) for name, p in OUTER.items()} | {
    f"inner-{name}": (inner, p) for name, p in INNER.items()}


def pair(key, seed=5):
    build, p = PAIRS[key]
    return build(p, seed)


def measure_x(wires, postselect=None):
    """``H`` and ``qp.measure(q, reset=True)`` on every wire, postselected on the bits of
    ``postselect`` if it is given."""
    outcomes = []
    for i, wire in enumerate(wires):
        qp.H(wire)
        outcomes.append(qp.measure(wire, reset=True,
                                   postselect=None if postselect is None else postselect[i]))
    return outcomes


def round_trip(pr, order=lambda outcomes: outcomes, postselect=None):
    """The phase gradient, the prepare, the junk measurement (postselected on ``postselect`` if
    it is given) and the unprepare from the outcomes, passed in the order ``order`` gives."""
    def qfunc():
        qp.PhaseGradientStatePrep(pr.prepare_wires["phase_grad"])
        pr.prepare(*pr.args, **pr.prepare_wires)
        outcomes = measure_x(pr.prepare_wires[pr.junk], postselect)
        pr.unprepare(order(outcomes), *pr.args, **pr.unprepare_wires)

    return qfunc


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


def restores(num_wires, gradient_wires, inputs, operations, schedule) -> bool:
    """Whether ``operations`` return a random complex state of ``inputs``, every other register
    |0> and the phase gradient unchanged, under the forced ``schedule``, all of it consumed
    (strict mode: every measurement has probability 1/2, every AND precondition holds)."""
    initial = random_input_state(num_wires, inputs)
    gradient = decompose_for_simulation(
        lambda: qp.PhaseGradientStatePrep(gradient_wires)).operations
    expected = sorted_terms(SparseSimulator(num_wires).run(gradient, initial_state=initial))
    simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
    keys, amps = sorted_terms(simulator.run(operations, initial_state=initial))
    assert simulator.policy.position == len(schedule)
    return (np.array_equal(keys, expected[0])
            and np.allclose(amps, expected[1], rtol=0, atol=1e-12))


def schedules(num, seed=3):
    """The alternating and a seeded schedule of ``num`` outcomes."""
    return {"alternating": [i % 2 for i in range(num)],
            "seeded": np.random.default_rng(seed).integers(0, 2, num).tolist()}


def basis_schedules(num):
    """The zero schedule and every unit schedule of ``num`` outcomes (spec 12.5)."""
    return [[0] * num] + [[int(i == j) for j in range(num)] for i in range(num)]


@pytest.mark.parametrize("key", list(PAIRS))
def test_round_trip_after_the_junk_measurement(key):
    """T3 (sparse simulator, strict): prepare, X measurement of the junk, unprepare from the
    outcomes return a random state of the inputs (x_o for the inner pair), every other register
    |0> and the phase gradient unchanged, under the alternating and a seeded schedule of all K d
    measurements, junk first; an all-ones schedule would see only each address's mask parity
    (ledger row 2D.2). With more than one junk outcome, the outcomes passed in reversed order fail
    the round trip under the zero or some unit schedule: the fixup reads them in the junk
    register's wire order, the order of QROAMRead.junk_layout's slots."""
    pr = pair(key)
    operations = decompose_for_simulation(round_trip(pr)).operations
    assert sum(op.name == "MidMeasure" for op in operations) == pr.measured
    grad = pr.prepare_wires["phase_grad"]
    for kind, outcomes in schedules(pr.measured).items():
        assert restores(pr.num_wires, grad, pr.inputs, operations, outcomes), kind
    if len(pr.prepare_wires[pr.junk]) > 1:
        wrong = decompose_for_simulation(round_trip(pr, lambda o: o[::-1])).operations
        assert not all(restores(pr.num_wires, grad, pr.inputs, wrong, outcomes)
                       for outcomes in basis_schedules(pr.measured))


def lightning(num_wires, body, seed):
    @qp.qjit(capture=True, seed=seed)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return qp.state()

    return np.asarray(circuit())


def expected_state(num_wires, gradient_wires, inputs=()):
    """|+>^len(inputs) on ``inputs``, the phase-gradient state, |0> elsewhere."""
    state = np.ones(1)
    for wire in range(num_wires):
        if wire == gradient_wires[0]:
            b = len(gradient_wires)
            state = np.kron(state, np.exp(-2j * np.pi * np.arange(2**b) / 2**b) / np.sqrt(2**b))
        elif wire in gradient_wires[1:]:
            continue
        else:
            state = np.kron(state, [2**-0.5, 2**-0.5] if wire in inputs else [1, 0])
    return state


@pytest.mark.compiled
def test_lightning_round_trip_with_two_seeds():
    """The compiled round trip at H2_STO3G_TEST's outer shape (19 wires): every register returns
    to |0> and the phase gradient is unchanged, for two measurement seeds (smoke runs: the
    certificates prove every outcome on the tape path, and the postselected test below fixes the
    junk outcomes of the compiled path)."""
    pr = pair("outer-sto3g")
    expected = expected_state(pr.num_wires, pr.prepare_wires["phase_grad"])
    for seed in range(2):
        state = lightning(pr.num_wires, round_trip(pr), seed)
        assert np.allclose(state, expected, rtol=0, atol=1e-10), seed


@pytest.mark.compiled
@pytest.mark.parametrize("key", ["outer-sto3g", "inner-small"])
def test_lightning_round_trip_with_postselected_junk(key):
    """The compiled path of the junk outcomes, deterministically: the junk measurements are
    postselected on the unit pattern 1, 0, ..., 0 (analytic lightning.qubit honours postselection
    under capture and renormalizes), so the unprepare receives a traced bool array of known
    content. The inputs (x_o of the inner pair in |+>) come back, every other register |0> and the
    phase gradient unchanged. The pattern is neither zero nor a palindrome, so junk outcomes that
    the capture branch of the canonicalization reverses or drops leave a phase, which the sampled
    lightning tests see only for some seeds."""
    pr = pair(key)
    pattern = [1] + [0] * (len(pr.prepare_wires[pr.junk]) - 1)

    def body():
        for wire in pr.inputs:
            qp.H(wire)
        round_trip(pr, postselect=pattern)()

    expected = expected_state(pr.num_wires, pr.prepare_wires["phase_grad"], pr.inputs)
    assert np.allclose(lightning(pr.num_wires, body, 0), expected, rtol=0, atol=1e-10)


# ------------------------------------------------- a caller inside another operator's rule


class InnerRoundTripProbe(Operator2):
    """Test operator (ledger decision "K-10 consequences" (3)): its rule calls ``InnerPrepare``,
    X-measures ``junk_b`` and calls ``InnerUnprepareFromOutcomes`` with the outcomes, as the block
    encoding's rule does. ``wires`` is ``InnerPrepare``'s registers, concatenated in its
    ``wire_argnames`` order."""

    compilable_argnames = ("args",)
    wire_argnames = ("wires",)

    def __init__(self, args, wires):
        super().__init__(args, wires)


def _probe_registers(args, wires) -> dict:
    sizes = InnerPrepare.required_wires(*args[:4], *args[8:])
    registers, start = {}, 0
    for name, size in sizes.items():
        registers[name] = wires[start:start + size]
        start += size
    return registers


def _probe_resources(args, wires):
    prepare = InnerPrepare.required_wires(*args[:4], *args[8:])
    unprepare = InnerUnprepareFromOutcomes.required_wires(*args[:4], *args[8:])
    junk = prepare["junk_b"]
    outcomes = abstract_junk_outcomes(junk)
    return {
        resource_key(lambda: InnerPrepare(*args, **{k: Wire[n] for k, n in prepare.items()})): 1,
        qp.H: junk, resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True)): junk,
        resource_key(lambda: InnerUnprepareFromOutcomes(
            outcomes, *args, **{k: Wire[n] for k, n in unprepare.items()})): 1}


@qp.register_resources(_probe_resources)
def _inner_round_trip_probe(args, wires):
    registers = _probe_registers(args, wires)
    InnerPrepare(*args, **registers)
    outcomes = measure_x(registers["junk_b"])
    InnerUnprepareFromOutcomes(outcomes, *args, **lend(InnerUnprepareFromOutcomes, registers,
                                                       "junk_b", args[:4] + args[8:]))


register_rule(InnerRoundTripProbe, _inner_round_trip_probe)


def probe(key):
    pr = pair(key)
    flat = [wire for register in pr.prepare_wires.values() for wire in register]
    return pr, flat


@pytest.mark.compiled
@pytest.mark.parametrize("key", ["inner-sto3g", "inner-small"])
def test_caller_inside_another_rule_is_valid_and_counted(key):
    """The junk outcomes cross an operator boundary inside another rule (K-10 part 2 in
    production form): T2 of the probe's rule in tape mode, where the outcomes are an object array
    of MeasurementValues, and its compiled T, M and A, where they are a traced bool array: T of
    the prepare and the unprepare, all K2 d2 measurements, and A the prepare's uniform-preparation
    and read ANDs plus the unprepare's."""
    pr, flat = probe(key)
    op = InnerRoundTripProbe(pr.args, flat)
    qp.testing.assert_valid_decomp_rule(op, _inner_round_trip_probe, check_matrix=False)
    resources = compiled_resources(lambda: InnerRoundTripProbe(pr.args, flat), pr.num_wires)
    p = pr.params
    prepare = costs.inner_prepare(p).toffoli
    expected = prepare + costs.inner_unprepare_from_outcomes(p).toffoli
    assert (resources.toffoli, resources.measurements) == (expected, p.K2 * p.d2)
    prepare_and = prepare - (p.K2 - 1) * p.d2 - p.b_k2 - p.w_b - _dirty(p.B + 1)
    assert resources.measured_and_uncomputes == prepare_and + expected_and_uncomputes(pr)


def test_caller_inside_another_rule_restores_every_outcome():
    """The probe on the tape path (strict sparse simulator) at the small shape restores a random
    complex state of x_o, every other register |0> and the phase gradient, under the zero and
    every unit schedule of its K2 d2 = 8 outcomes: by GF(2) linearity, every outcome (spec 12.5);
    every conditional is a PauliZ or CZ on a linear predicate of the outcomes."""
    pr, flat = probe("inner-small")

    def qfunc():
        qp.PhaseGradientStatePrep(pr.prepare_wires["phase_grad"])
        InnerRoundTripProbe(pr.args, flat)

    operations = decompose_for_simulation(qfunc).operations
    assert any(isinstance(op, Conditional) for op in operations)
    assert not nonlinear_conditionals(operations)
    for schedule in basis_schedules(pr.measured):
        assert restores(pr.num_wires, pr.prepare_wires["phase_grad"], pr.inputs, operations,
                        schedule), schedule


@pytest.mark.compiled
def test_caller_inside_another_rule_on_lightning():
    """The compiled probe at the small shape (20 wires), x_o in |++>: x_o comes back in |++>,
    every other register in |0> and the phase gradient unchanged, for two measurement seeds."""
    pr, flat = probe("inner-small")

    def body():
        for wire in pr.inputs:
            qp.H(wire)
        qp.PhaseGradientStatePrep(pr.prepare_wires["phase_grad"])
        InnerRoundTripProbe(pr.args, flat)

    expected = expected_state(pr.num_wires, pr.prepare_wires["phase_grad"], pr.inputs)
    for seed in range(2):
        assert np.allclose(lightning(pr.num_wires, body, seed), expected, rtol=0,
                           atol=1e-10), seed


# --------------------------------------------------------------------------- resources


def _dirty(L):
    n = lg(odd_part(L))
    return n - 1 if n else 0


def expected_and_uncomputes(pr) -> int:
    """A of the unprepare from outcomes, as the baseline unprepare's (spec 9.24): the
    comparator's b_k, the fixup's T and the adjoint uniform preparation's 3(n - 1) + 2(s_u - 1)
    (n = lg l, none if l = 1)."""
    p = pr.params
    if pr.prepare is OuterPrepare:
        L, b_k, ob = p.X_o, p.b_k1, p.ke1
        read = OuterPrepare.read_shape(p.X_o, p.b_k1, p.k1)
    else:
        L, b_k, ob = p.B + 1, p.b_k2, p.ke2 + p.w_b
        read = InnerPrepare.read_shape(p.N, p.R, p.B, p.C, p.b_k2, p.k2)
    widths, live, width, k = read.values()
    fixup = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width, ob,
                              False)
    n = lg(odd_part(L))
    return b_k + fixup.toffoli + (3 * (n - 1) + 2 * (p.s_u - 1) if n else 0)


def closed_form(pr) -> costs.Cost:
    if pr.prepare is OuterPrepare:
        return costs.outer_unprepare_from_outcomes(pr.params)
    return costs.inner_unprepare_from_outcomes(pr.params)


def alone(pr):
    """The unprepare alone, with constant junk outcomes (a bool array under capture)."""
    junk = np.zeros(len(pr.prepare_wires[pr.junk]), dtype=bool)
    return lambda: pr.unprepare(junk, *pr.args, **pr.unprepare_wires)


def assert_counts(pr):
    resources = compiled_resources(alone(pr), pr.num_wires)
    expected = closed_form(pr)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measurements == pr.measured // (pr.params.K1 if pr.prepare is OuterPrepare
                                                     else pr.params.K2)
    assert resources.measured_and_uncomputes == expected_and_uncomputes(pr)


@pytest.mark.compiled
@pytest.mark.parametrize("key", ["outer-sto3g", "outer-631g", "outer-x-o-5", "inner-sto3g",
                                 "inner-631g", "inner-6-2"])
def test_counts_match_the_closed_forms(key):
    """T4 and the T6 signature, each class alone: T equals the baseline unprepare's (dfthc.costs),
    M is the target's d (not the K d of the baseline: the junk was measured by the caller), and A
    is the baseline unprepare's."""
    assert_counts(pair(key))


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("build", [outer, inner], ids=["outer", "inner"])
def test_counts_at_mid_naphthalene(build):
    """T4 at MID_NAPHTHALENE with the naphthalene tables (nightly tier): outer T 38, M 21; inner
    T 53, M 23."""
    p = MID_NAPHTHALENE
    pr = build(p)
    t = WalkTables.from_factors(load_factors("naphthalene"), p)
    args = ((p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u) if build is outer
            else (p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn,
                  p.b_k2, p.k2, p.ke2, p.s_u))
    assert_counts(pr._replace(args=args))


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 on the closed forms, nothing compiled: T 69 and 217 as the baseline unprepares,
    M 24 and 26 (the targets; the block encoding measures the 72 and 390 junk qubits); the
    registers are the prepares' without the junk, with extra_onehot 0 (2**4 <= 24) and 102
    (2**7 - 26) wires."""
    p = FEMOCO54
    assert costs.outer_unprepare_from_outcomes(p) == costs.Cost(69, 24)
    assert costs.inner_unprepare_from_outcomes(p) == costs.Cost(217, 26)
    assert costs.outer_unprepare(p).measurements == 24 + 72
    assert costs.inner_unprepare(p).measurements == 26 + 390
    assert OuterUnprepareFromOutcomes.junk_size(p.X_o, p.b_k1, p.k1) == 72
    assert InnerUnprepareFromOutcomes.junk_size(p.N, p.R, p.B, p.C, p.b_k2, p.k2) == 390
    outer_sizes = OuterUnprepareFromOutcomes.required_wires(*outer_structure(p))
    assert outer_sizes == {"x_o": 9, "succ_x_o": 1, "succ_work_x_o": 6, "sigma_1": 1,
                           "m_x_o": 15, "less_than_x_o": 1, "cmp_work_x_o": 14, "alt_x_o": 9,
                           "keep_x_o": 15, "phase_grad": 7, "work": 20, "extra_onehot": 0}
    inner_sizes = InnerUnprepareFromOutcomes.required_wires(*inner_structure(p))
    assert inner_sizes == {"x_o": 9, "b": 5, "succ_b": 1, "succ_work_b": 2, "sigma_0": 1,
                           "m_b": 15, "less_than_b": 1, "cmp_work_b": 14, "alt_b": 5,
                           "keep_b": 15, "sgn": 1, "d_sgn": 1, "r": 4, "phase_grad": 7,
                           "work": 16, "extra_onehot": 102}


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: in a program with both classes alone, the solver names both pinned rules and no
    QROAMErase rule (the erase is split, spec 16); T5 at level="device" (L-20)."""
    outer_pair, inner_pair = pair("outer-x-o-5"), pair("inner-sto3g")

    def program():
        alone(outer_pair)()
        alone(inner_pair)()

    num_wires = max(outer_pair.num_wires, inner_pair.num_wires)
    qp.analyze(verbose_program(program, num_wires=num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_outer_unprepare_from_outcomes", "_inner_unprepare_from_outcomes"} <= rules, rules
    assert "_qroam_erase" not in rules
    compiled_resources(program, num_wires=num_wires, level="device")


@pytest.mark.parametrize("key", ["outer-sto3g", "outer-x-o-5", "outer-x-o-7-k0-b1",
                                 "inner-sto3g", "inner-6-2", "inner-7-0"])
def test_rules_are_valid(key):
    """T2 in tape mode, with junk outcomes measured before the operator, in every branch of the
    hand-written resources: l = 1 (sto3g), l > 1, no junk and no comparator work (k = 0,
    b_k1 = 1), extra_onehot of 0 (outer) and 1 or more (inner) wires."""
    pr = pair(key)
    with qp.queuing.AnnotatedQueue():
        outcomes = measure_x(pr.prepare_wires[pr.junk])
    rule = (_outer_unprepare_from_outcomes if pr.unprepare is OuterUnprepareFromOutcomes
            else _inner_unprepare_from_outcomes)
    qp.testing.assert_valid_decomp_rule(pr.unprepare(outcomes, *pr.args, **pr.unprepare_wires),
                                        rule, check_matrix=False)


def test_required_wires_and_junk_sizes_by_hand():
    """At H2_STO3G_TEST: the prepares' registers without the junk; extra_onehot 0 for the outer
    one-hot (2**1 <= d1 = 4) and 1 for the inner (2**(1 + 2) - d2 = 8 - 7); 4 and 7 outcomes."""
    p = H2_STO3G_TEST
    assert OuterUnprepareFromOutcomes.required_wires(*outer_structure(p)) == {
        "x_o": 2, "succ_x_o": 1, "succ_work_x_o": 0, "sigma_1": 1, "m_x_o": 2,
        "less_than_x_o": 1, "cmp_work_x_o": 1, "alt_x_o": 2, "keep_x_o": 2, "phase_grad": 3,
        "work": 0, "extra_onehot": 0}
    assert InnerUnprepareFromOutcomes.required_wires(*inner_structure(p)) == {
        "x_o": 2, "b": 2, "succ_b": 1, "succ_work_b": 1, "sigma_0": 1, "m_b": 2,
        "less_than_b": 1, "cmp_work_b": 1, "alt_b": 2, "keep_b": 2, "sgn": 1, "d_sgn": 1,
        "r": 1, "phase_grad": 3, "work": 7, "extra_onehot": 1}
    assert OuterUnprepareFromOutcomes.junk_size(p.X_o, p.b_k1, p.k1) == 4
    assert InnerUnprepareFromOutcomes.junk_size(p.N, p.R, p.B, p.C, p.b_k2, p.k2) == 7


@pytest.mark.parametrize(("key", "change", "match"), [
    ("outer-x-o-5", {"junk_outcomes": "short"}, r"need one outcome per junk qubit \(6\), got 5"),
    ("outer-x-o-5", {"junk_outcomes": "booleans"},
     "in tape mode every junk outcome must be a MeasurementValue; entry 0 is"),
    ("outer-x-o-5", {"junk_outcomes": "abstract"}, r"junk_outcomes must have shape \(6,\), got"),
    ("outer-x-o-5", {"extra_onehot": [99]}, "'extra_onehot' has 1 wires; expected 0"),
    ("outer-x-o-5", {"ke1": 3, "k1": 0}, "needs 8 wires, more than the 6 target and junk wires"),
    ("inner-sto3g", {"junk_outcomes": "short"}, r"need one outcome per junk qubit \(7\), got 6"),
    ("inner-sto3g", {"extra_onehot": []}, "'extra_onehot' has 0 wires; expected 1"),
    ("inner-sto3g", {"r": []}, "'r' has 0 wires; expected 1"),
    ("inner-sto3g", {"ke2": 2}, r"ke2 must lie in \[0, 1\]"),
], ids=["outer-short", "outer-booleans", "outer-abstract", "outer-extra-onehot", "outer-onehot",
        "inner-short", "inner-extra-onehot", "inner-r", "inner-ke2"])
def test_constructors_validate(key, change, match):
    """Spec 9.0: the junk outcomes must be one MeasurementValue per junk qubit in tape mode (or an
    array of their shape), the registers exactly required_wires, and the compilable arguments in
    the prepare's domains, the one-hot fitting in the reset target and junk."""
    pr = pair(key)
    names = pr.unprepare.compilable_argnames
    arguments = dict(zip(names, pr.args, strict=True)) | pr.unprepare_wires
    with qp.queuing.AnnotatedQueue():
        outcomes = measure_x(pr.prepare_wires[pr.junk])
    junk = {"short": outcomes[:-1], "booleans": [False] * len(outcomes),
            "abstract": AbstractArray((len(outcomes) - 1,), bool), None: outcomes}
    arguments |= {name: value for name, value in change.items() if name != "junk_outcomes"}
    with pytest.raises(ValueError, match=match):
        pr.unprepare(junk[change.get("junk_outcomes")], **arguments)


@pytest.mark.compiled
@pytest.mark.parametrize("wrap", ["adjoint", "control"])
def test_no_adjoint_or_controlled_path(wrap):
    """Spec 7.4 rules 5 and 6: nothing adjoints or controls an unprepare from outcomes. Wrapped
    anyway, Catalyst drops the measuring candidates with the expected warning (rule 12), and the
    compile fails loudly."""
    pr = pair("inner-sto3g")
    key = ("Adjoint(InnerUnprepareFromOutcomes)" if wrap == "adjoint"
           else "C(InnerUnprepareFromOutcomes)")

    def program():
        op = alone(pr)()
        return qp.adjoint(op) if wrap == "adjoint" else qp.ctrl(op, control=pr.num_wires)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(program, pr.num_wires + 1)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the ")]
    named = [message for message in skipped if classify_rule_warning(message) != "noise"]
    assert named and {classify_rule_warning(message) for message in named} == {"expected"}
```

Replace `tests/ops/test_certificates_prepare.py` with (new: the `from_outcomes` helper and the two certificates at the end):

```python
"""Outcome certificates of spec 12.5 for the composite unprepares (nightly tier).

A prepare followed by its unprepare must return every register to its input for every outcome of
the unprepare's erase. The phase the erase leaves on the read's address and the phase of its
fix-up are both GF(2)-linear in the outcome vector ``xi``, so the residual is linear, and exact
restoration under the zero vector and under every unit vector, M + 1 forced runs, certifies all
2**M outcomes. The premise is checked on the certified tape itself: it has conditionals, and each
is a ``PauliZ`` or ``CZ`` with a GF(2)-linear predicate (``nonlinear_conditionals``); the measured
uncomputes of the comparators and of the uniform preparations are ``Adjoint(TemporaryAND)``
gates, whose preconditions the simulator checks on every term, not conditionals. Strict mode
also asserts that every erase measurement has probability 1/2. Each run must consume exactly M
outcomes and end in exactly the input state times the phase-gradient state.

OuterPrepare -> OuterUnprepare: the pair has no input register and no enable (the walk uses
none), so every register in |0> is its only legal pre-state, and each run starts from it; an
outcome-dependent phase, a wrong amplitude or a leftover ancilla shows as a difference from it
(the uniform preparation's adjoint moves any relative phase on ``x_o`` away from |0>). This fixed
input does not instantiate spec 12.5's general premise of random complex address and control
amplitudes: the read and erase carry that premise through Task 1C.5's QROAMRead -> QROAMErase
certificates (tests/ops/test_certificates_qroam.py), and this certificate composes them with the
pair's other children on its one legal input. The tables are random, the shapes include a
nontrivial uniform preparation (failure branches and its phase), dead positions in the read's
last block and a partly covered fix-up leaf; one shape has a ``phase_grad`` longer than ``s_u``,
whose extra wires must end unchanged.

InnerPrepare -> InnerUnprepare: ``x_o`` is the read-only address, so each run starts from a
random complex amplitude on every value of ``x_o``, SF rows, one-body rows and failed values
alike; any permutation or relative phase of the rows is caught. The tables are random, with a
zero row (so zero erase masks) where the rank column is zero too; the shapes include dead
positions, failed ``b`` and partly covered fix-up leaves.

Prepare -> junk measurement -> unprepare from outcomes (spec 16, Q-1 variant (b)): the same
shapes, with ``H`` and ``qp.measure(q, reset=True)`` on every junk qubit right after the prepare
and the unprepare from those outcomes, its ``extra_onehot`` lent from the start of the measured
junk register. The outcome vector ``xi`` ranges over the junk outcomes, measured first, and the
target outcomes together: M = K d forced outcomes per run, M + 1 runs.
"""

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional

from dfthc.ops.prepare import (
    InnerPrepare,
    InnerUnprepare,
    InnerUnprepareFromOutcomes,
    OuterPrepare,
    OuterUnprepare,
    OuterUnprepareFromOutcomes,
)
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, lg
from dfthc.testing.sparse_sim import (
    ForcedPolicy,
    SparseSimulator,
    SparseState,
    nonlinear_conditionals,
)
from dfthc.testing.tape import decompose_for_simulation

pytestmark = pytest.mark.slow


def layout(sizes):
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


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


def certify(prepare, unprepare, wires, num_wires, inputs, num_measured):
    """Run ``prepare`` then ``unprepare`` (after the phase-gradient state) under the zero and every
    unit schedule, from a random complex state of the ``inputs`` wires."""
    gradient = lambda: qp.PhaseGradientStatePrep(wires=wires["phase_grad"])  # noqa: E731

    def round_trip():
        gradient()
        prepare()
        unprepare()

    operations = decompose_for_simulation(round_trip).operations
    assert any(isinstance(op, Conditional) for op in operations)
    assert not nonlinear_conditionals(operations)
    assert sum(op.name == "MidMeasure" for op in operations) == num_measured
    initial = random_input_state(num_wires, inputs)
    reference = decompose_for_simulation(gradient).operations
    expected_keys, expected_amps = sorted_terms(
        SparseSimulator(num_wires).run(reference, initial_state=initial))
    schedules = [[0] * num_measured] + [
        [int(i == j) for j in range(num_measured)] for i in range(num_measured)]
    for schedule in schedules:
        simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
        keys, amps = sorted_terms(simulator.run(operations, initial_state=initial))
        assert simulator.policy.position == num_measured
        assert np.array_equal(keys, expected_keys), schedule
        assert np.allclose(amps, expected_amps, rtol=0, atol=1e-12), schedule


# (X_o, b_k1, k1, ke1, s_u): both test presets (X_o = 4 and 8, trivial uniform preparations);
# X_o = 5 with ke1 = 2 (the fix-up leaf [4, 8) covers only 4 and 5 of the loaded [0, 6); the
# uniform preparation fails with weight 0.15 and a phase; phase_grad has two wires more than s_u,
# as when Phase 3 passes the whole register); X_o = 6 with k1 = 2 (dead positions 6 and 7, l = 3).
OUTER = {
    "sto3g": (4, 2, 1, 1, 3),
    "631g": (8, 2, 1, 1, 3),
    "x-o-5-ke2": (5, 3, 1, 2, 3),
    "x-o-6-k2": (6, 2, 2, 1, 3),
}
assert OUTER["sto3g"] == tuple(getattr(H2_STO3G_TEST, f) for f in ("X_o", "b_k1", "k1", "ke1",
                                                                     "s_u"))
assert OUTER["631g"] == tuple(getattr(H2_631G_TEST, f) for f in ("X_o", "b_k1", "k1", "ke1",
                                                                  "s_u"))


def random_outer_tables(X_o, b_k1, seed=5):
    rng = np.random.default_rng(seed)
    return (tuple(int(a) for a in rng.integers(0, X_o, X_o)),
            tuple(int(k) for k in rng.integers(0, 2**b_k1, X_o)))


@pytest.mark.parametrize("name", list(OUTER))
def test_outer_prepare_unprepare_certificate(name):
    X_o, b_k1, k1, ke1, s_u = OUTER[name]
    args = (X_o, *random_outer_tables(X_o, b_k1), b_k1, k1, ke1, s_u)
    sizes = OuterPrepare.required_wires(X_o, b_k1, k1, ke1, s_u)
    sizes["phase_grad"] += 2 if name == "x-o-5-ke2" else 0
    wires, num_wires = layout(sizes)
    certify(lambda: OuterPrepare(*args, **wires), lambda: OuterUnprepare(*args, **wires), wires,
            num_wires, inputs=[], num_measured=2**k1 * (lg(X_o) + b_k1))


# (N, R, B, C, b_k2, k2, ke2, s_u): both test presets ((B + 1, k2) = (3, 1), l_b = 3); rows
# [0, 3) of a 3-bit x_o with (B + 1, k2) = (6, 2), row 0 zero (rank 0 as well, so its masks are
# zero) and the last fix-up leaf (x_o in {2, 3}) partly covered; rows [0, 2) with
# (B + 1, k2) = (5, 1): b = 5 is a dead position, and every fix-up leaf covers b in [0, 6) only.
INNER = {
    "sto3g": (2, 2, 2, 1, 2, 1, 1, 3),
    "631g": (4, 2, 2, 2, 2, 1, 1, 3),
    "inner-6-2-zero-row": (2, 3, 5, 1, 2, 2, 1, 3),
    "inner-5-1": (3, 2, 4, 1, 2, 1, 1, 3),
}
assert INNER["sto3g"] == tuple(getattr(H2_STO3G_TEST, f) for f in ("N", "R", "B", "C", "b_k2",
                                                                     "k2", "ke2", "s_u"))
assert INNER["631g"] == tuple(getattr(H2_631G_TEST, f) for f in ("N", "R", "B", "C", "b_k2",
                                                                  "k2", "ke2", "s_u"))


def random_inner_tables(R, B, C, b_k2, seed=5, zero_rows=()):
    """Random alt, keep, sgn and d_sgn of shape (R C, B + 1), zero on the rows ``zero_rows``."""
    rng = np.random.default_rng(seed)
    shape = (R * C, B + 1)
    columns = [rng.integers(0, B + 1, shape), rng.integers(0, 2**b_k2, shape),
               rng.integers(0, 2, shape), rng.integers(0, 2, shape)]
    return tuple(tuple(tuple(0 if x in zero_rows else int(v) for v in row)
                       for x, row in enumerate(column)) for column in columns)


@pytest.mark.parametrize("name", list(INNER))
def test_inner_prepare_unprepare_certificate(name):
    N, R, B, C, b_k2, k2, ke2, s_u = INNER[name]
    zero_rows = (0,) if name.endswith("zero-row") else ()
    args = (N, R, B, C, *random_inner_tables(R, B, C, b_k2, zero_rows=zero_rows), b_k2, k2, ke2,
            s_u)
    wires, num_wires = layout(InnerPrepare.required_wires(N, R, B, C, b_k2, k2, ke2, s_u))
    certify(lambda: InnerPrepare(*args, **wires), lambda: InnerUnprepare(*args, **wires), wires,
            num_wires, inputs=wires["x_o"],
            num_measured=2**k2 * (lg(B + 1) + b_k2 + 2 + lg(R)))


def from_outcomes(prepare, unprepare, args, wires, junk, structure):
    """``(prepare and junk measurement, unprepare from those outcomes)`` as two callables of one
    round trip; the unprepare's ``extra_onehot`` is the start of the measured junk register."""
    sizes = unprepare.required_wires(*structure)
    registers = {name: wires[name] for name in sizes if name != "extra_onehot"}
    registers["extra_onehot"] = wires[junk][:sizes["extra_onehot"]]
    outcomes = []

    def measure():
        prepare(*args, **wires)
        for wire in wires[junk]:
            qp.H(wire)
            outcomes.append(qp.measure(wire, reset=True))

    return measure, lambda: unprepare(outcomes, *args, **registers)


@pytest.mark.parametrize("name", list(OUTER))
def test_outer_unprepare_from_outcomes_certificate(name):
    X_o, b_k1, k1, ke1, s_u = OUTER[name]
    args = (X_o, *random_outer_tables(X_o, b_k1), b_k1, k1, ke1, s_u)
    sizes = OuterPrepare.required_wires(X_o, b_k1, k1, ke1, s_u)
    sizes["phase_grad"] += 2 if name == "x-o-5-ke2" else 0
    wires, num_wires = layout(sizes)
    certify(*from_outcomes(OuterPrepare, OuterUnprepareFromOutcomes, args, wires, "junk_x_o",
                           (X_o, b_k1, k1, ke1, s_u)),
            wires, num_wires, inputs=[], num_measured=2**k1 * (lg(X_o) + b_k1))


@pytest.mark.parametrize("name", list(INNER))
def test_inner_unprepare_from_outcomes_certificate(name):
    N, R, B, C, b_k2, k2, ke2, s_u = INNER[name]
    zero_rows = (0,) if name.endswith("zero-row") else ()
    args = (N, R, B, C, *random_inner_tables(R, B, C, b_k2, zero_rows=zero_rows), b_k2, k2, ke2,
            s_u)
    structure = (N, R, B, C, b_k2, k2, ke2, s_u)
    wires, num_wires = layout(InnerPrepare.required_wires(*structure))
    certify(*from_outcomes(InnerPrepare, InnerUnprepareFromOutcomes, args, wires, "junk_b",
                           structure),
            wires, num_wires, inputs=wires["x_o"],
            num_measured=2**k2 * (lg(B + 1) + b_k2 + 2 + lg(R)))
```

Replace `tests/test_docstrings.py` with (changed: the `SECTIONS` docstring and `test_every_operator_is_audited`):

```python
"""The documentation set (spec 2.3 D-6, 5.2) and the operator docstrings (spec 9.0; master plan
Task 4.4).

Every ``Operator2`` subclass of ``dfthc.ops`` and ``dfthc.qpe`` has the nine docstring sections
of spec 9.0 in order, and its cost section names an existing ``dfthc.costs`` function. The
documentation files of spec 5.2 exist, their relative links resolve, the limitations name only
existing canaries and every limitation of spec 14, and the architecture names every operator.
"""

import importlib
import inspect
import pkgutil
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import pytest
from pennylane.core import Operator2

import dfthc.ops
from dfthc import costs

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SECTIONS = (
    ("purpose and mathematical action", r"\*\*Action\b"),
    ("register table", r"\*\*Registers\.\*\*"),
    ("compilable arguments", r"\*\*(?:Compilable a|A)rguments\.\*\*"),
    ("pre- and postconditions", r"\*\*Pre- and postconditions\.\*\*"),
    ("decomposition outline", r"\*\*Decomposition\b"),
    ("adjoint and control behaviour", r"\*\*Adjoint and control\.\*\*"),
    ("cost formula", r"\*\*Cost\.\*\*"),
    ("paper equations and the dfthcbe bloq", r"\*\*Paper and dfthcbe\.\*\*"),
    ("a minimal compiled example", r"\*\*Example\.\*\*"),
)
"""Spec 9.0's sections in order, each a bold heading at the start of a docstring line. QROAMRead,
PhaseFixup and the two unprepares from outcomes, which also take dynamic arguments, title
section 3 "Arguments"."""
DOCUMENTS = ("architecture.md", "conventions.md", "environment.md", "limitations.md",
             "verification.md", "resources.md", "resources.json", "agents/HANDOFF.md",
             "agents/LEDGER.md")
"""The documentation set of spec 5.2 under ``docs/``."""
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def operator_classes() -> dict[str, type]:
    """Every Operator2 subclass reachable from dfthc.ops (recursively) or dfthc.qpe."""
    modules = [dfthc.ops]
    modules.extend(importlib.import_module(info.name)
                   for info in pkgutil.walk_packages(dfthc.ops.__path__, prefix="dfthc.ops."))
    modules.append(importlib.import_module("dfthc.qpe"))
    classes = {cls for module in modules for _, cls in inspect.getmembers(module, inspect.isclass)
               if cls is not Operator2 and issubclass(cls, Operator2)}
    return {f"{cls.__module__}.{cls.__qualname__}": cls for cls in classes}


CLASSES = operator_classes()


def test_every_operator_is_audited():
    """The 32 operators of dfthc.ops (spec 9.1 to 9.22, the two unprepares from outcomes of
    spec 16, Q-1 variant (b), included) and the two of dfthc.qpe (9.23, 11.1)."""
    assert len(CLASSES) == 34
    names = {cls.__name__ for cls in CLASSES.values()}
    assert {"PhaseGradientRotation", "QROAMRead", "DFTHCWalk", "ControlledDFTHCWalk",
            "OuterUnprepareFromOutcomes", "InnerUnprepareFromOutcomes", "ControlledWalkPower",
            "QPERound"} <= names


@pytest.mark.parametrize("qualified_name", sorted(CLASSES))
def test_docstring_has_the_nine_sections_in_order(qualified_name):
    """Spec 9.0: a summary line, then the nine sections in order, each once; the cost section
    names at least one dfthc.costs function and only existing ones; the example calls
    compiled_resources."""
    cls = CLASSES[qualified_name]
    name = cls.__name__
    doc = inspect.getdoc(cls) or ""
    lines = doc.splitlines()
    assert lines and lines[0].strip(), f"{name} has no summary line"
    starts = []
    for title, pattern in SECTIONS:
        found = [i for i, line in enumerate(lines) if re.match(pattern, line.strip())]
        assert len(found) == 1, f"{name}: section '{title}' found {len(found)} times"
        starts.append(found[0])
    assert starts == sorted(starts), f"{name}: sections out of order"
    assert starts[0] > 0, f"{name}: the summary must precede the Action section"
    cost = "\n".join(lines[starts[6]:starts[7]])
    named = re.findall(r"dfthc\.costs\.(\w+)", cost)
    assert named, f"{name}: the cost section names no dfthc.costs function"
    missing = [function for function in named if not callable(getattr(costs, function, None))]
    assert not missing, f"{name}: dfthc.costs has no {missing}"
    assert "compiled_resources(" in "\n".join(lines[starts[8]:]), f"{name}: no compiled example"


def test_the_documentation_set_exists():
    """Spec 5.2's documentation files exist (D-6)."""
    missing = [name for name in DOCUMENTS if not (DOCS / name).is_file()]
    assert not missing


def markdown_anchors(text: str) -> set[str]:
    """GitHub-style anchors for the Markdown headings used by this repository."""
    anchors, counts = set(), Counter()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
        heading = re.sub(r"`([^`]*)`", r"\1", heading)
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        slug = re.sub(r"[^\w\s-]", "", heading.lower())
        slug = re.sub(r"\s+", "-", slug.strip())
        suffix = counts[slug]
        counts[slug] += 1
        anchors.add(f"{slug}-{suffix}" if suffix else slug)
    return anchors


DOCUMENT_PATHS = tuple(sorted(path.relative_to(DOCS) for path in DOCS.rglob("*.md")
                              if "superpowers" not in path.relative_to(DOCS).parts))


@pytest.mark.parametrize("document", DOCUMENT_PATHS, ids=lambda path: path.as_posix())
def test_documentation_links_resolve(document):
    """Every relative link in every Markdown document resolves, including its anchor."""
    path = DOCS / document
    broken = []
    for target in LINK.findall(path.read_text()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        relative, _, anchor = target.partition("#")
        target_path = path if not relative else path.parent / unquote(relative)
        try:
            target_path.resolve().relative_to(DOCS.resolve())
        except ValueError:
            broken.append(target)
            continue
        if not target_path.is_file():
            broken.append(target)
            continue
        if anchor and unquote(anchor) not in markdown_anchors(target_path.read_text()):
            broken.append(target)
    assert not broken, f"{document}: {broken}"


def test_limitations_match_the_canaries_and_spec_14():
    """docs/limitations.md has an entry for every limitation L-1 to L-27 of spec 14, and every
    canary it names is a test function in tests/canary."""
    text = (DOCS / "limitations.md").read_text()
    numbers = [int(n) for n in re.findall(r"\*\*L-(\d+)\b", text)]
    assert Counter(numbers) == Counter(range(1, 28))
    canaries = set()
    for path in (REPO / "tests" / "canary").glob("test_*.py"):
        canaries |= set(re.findall(r"^def (test_\w+)", path.read_text(), re.M))
    named = set(re.findall(r"`(test_\w+)`", text))
    assert named and named <= canaries, sorted(named - canaries)


def test_architecture_names_every_operator():
    """docs/architecture.md names every audited operator class and every dfthc module."""
    text = (DOCS / "architecture.md").read_text()
    assert [cls.__name__ for cls in CLASSES.values() if f"`{cls.__name__}`" not in text] == []
    modules = [path.relative_to(REPO / "src").with_suffix("").as_posix().replace("/", ".")
               for path in (REPO / "src" / "dfthc").rglob("*.py") if path.name != "__init__.py"]
    assert [module for module in modules if f"`{module}`" not in text] == []
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_unprepare_from_outcomes.py -q`
Expected: FAIL, `1 error`: `ImportError` while importing the module (`InnerUnprepareFromOutcomes` is not in `dfthc.ops.prepare`).

Run: `scripts/test.sh tests/test_docstrings.py -q`
Expected: FAIL, `1 failed, 45 passed`: `test_every_operator_is_audited` (`assert len(CLASSES) == 34`; 32 classes).

- [ ] **Step 3: Add the erase from outcomes**

Replace `src/dfthc/ops/qroam.py` with (new: `_fixup_from_outcomes`, which `_erase_body` now calls, and `emit_erase_from_outcomes` at the end; changed docstrings, for the two junk paths: the module's last paragraph, `QROAMRead`'s summary, pre- and postconditions, adjoint and control, and paper sections, `QROAMErase`'s pre- and postconditions):

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
With ``M = 2**block_bits * width``, the qubits of ``target ‖ junk`` whose outcomes the erase
uses, bit ``i`` of ``loaded_bits`` is the wire ``M - 1 - i`` of ``target ‖ junk``, so the erase
passes its outcomes to :func:`dfthc.compile.mcm_parities` in reverse wire order.
:class:`QROAMErase` measures all of them; :func:`emit_erase_from_outcomes` measures the target and
takes the junk's outcomes, in wire order, from a caller that X-measured the junk after the read.
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
    ``junk``, for a measured erase: the junk stays until :class:`QROAMErase`, or the caller
    X-measures it early and erases with :func:`emit_erase_from_outcomes`.

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
    ends in ``|0>``. The loaded data is erased by measurement on one of two paths: (a) nothing
    writes ``target ‖ junk`` until :class:`QROAMErase` measures both; or (b) the caller applies
    ``H`` and ``qp.measure(q, reset=True)`` to every ``junk`` qubit after the read and before
    anything else acts on the junk (right after the read, or later: the walk measures right
    after each enclosing prepare, whose comparator and swaps do not touch the junk), keeps the
    outcomes in wire order (slot ``r`` of :meth:`junk_layout` is outcomes ``r width`` to
    ``(r + 1) width - 1``), may then use the junk wires for other registers, and later erases
    the unchanged ``target`` with :func:`emit_erase_from_outcomes` (spec 16, Q-1 variant (b)).
    On both paths, operations in between may act on the address only in ways that keep what the
    read loaded for the current address.

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
    adjoint or a control (spec 7.4 rule 6). Its data is erased by :class:`QROAMErase` (the
    baseline unprepares), by :func:`emit_erase_from_outcomes` (the unprepares from outcomes,
    which the walk calls) or by :class:`~dfthc.ops.angle_data.AngleErase` (the angle loads); use
    the enable instead of a control.

    **Cost.** :func:`dfthc.costs.qroam_read` with ``ranges = visited_ranges(...)``:
    :math:`T = U` (or :math:`U_2`) over the visited block ranges, controlled if there is an
    enable, plus :math:`(2^k - 1)` ``width``; ``M = 0``; ``A = U``. At FeMoco-54: outer
    84 + 72 = 156, inner 543 + 390 = 933. The rule declares ``num_leaves K width`` load CNOTs,
    or X gates for the flagless leaf (every position and bit, padding included), and is
    registered with ``exact=False``: the CNOT or X count is an upper bound, which ``qp.analyze``
    reports as well, because it counts every unhinted load conditional; T, M and A do not depend
    on the data and are exact.

    **Paper and dfthcbe.** Steps 2 and 6 (and the engine of 10 and 11); the QROAM of Berry et al.
    (Quantum 3, 208) with a measured erase. The paper X-measures its prepares' junk early, in
    the prepare (arXiv:2502.15882v1 App. B.7.2, steps 2 and 4); the walk measures it right after
    each enclosing prepare, before any angle load, which equals measuring right after the read
    because the comparator and the swaps in between do not touch the junk; dfthcbe keeps it
    until the erase. dfthcbe: the reads in ``OuterPrepare``/``InnerPrepare``; its truncated
    iteration and flat ``r`` target are deferred refinements (spec 9.3, "Known deltas").

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
    :math:`|0\rangle`, and the address state equals its pre-read state exactly. A caller that
    X-measured the junk after the read, before anything else acted on it, erases the target
    with :func:`emit_erase_from_outcomes` instead (spec 16, Q-1 variant (b)).

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
    App. C). Its fixups are the phase-fixup terms of PRX Eqs. (B50) (inner) and (B53) (outer),
    whose totals also undo the swaps and the inequality test; in arXiv:2502.15882v1 these are
    Eqs. (123) and (126), and the inner fixup term alone is Eq. (122). dfthcbe: the erases inside
    its unprepares.

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


def _fixup_from_outcomes(emit, outcomes, onehot_wires, table, address_widths, live_ranges, width,
                         block_bits, control_value, onehot_bits, address, work, control):
    """Steps 2 and 3 of the erase (spec 9.5): the parities of ``outcomes``, one per qubit of
    ``target ‖ junk`` in wire order, against the loaded bits of every address of the read's loaded
    set (masks built here from the compilable table), then the fixup over that set with the first
    ``2**onehot_bits`` wires of ``onehot_wires``, reset wires, as its one-hot register."""
    covered = QROAMRead.loaded_ranges(address_widths, live_ranges, block_bits)
    order = PhaseFixup.address_order(address_widths, covered)
    masks = _loaded_contents(_entries(table), address_widths, live_ranges, width, block_bits,
                             order)
    parities = emit.parities(outcomes[::-1], masks)
    onehot = list(onehot_wires)[:2**onehot_bits] if onehot_bits else []
    emit_phase_fixup(parities, address=address, address_widths=address_widths, covered=covered,
                     onehot=onehot, onehot_bits=onehot_bits, work=work, emit=emit,
                     **_control(control, control_value))


def _erase_body(emit, table, address_widths, live_ranges, width, block_bits, control_value,
                onehot_bits, address, target, junk, work, control, extra_onehot):
    qubits = list(target) + list(junk)
    outcomes = [emit.measure_x(qubit) for qubit in qubits]
    _fixup_from_outcomes(emit, outcomes, qubits + list(extra_onehot), table, address_widths,
                         live_ranges, width, block_bits, control_value, onehot_bits, address,
                         work, control)


def emit_erase_from_outcomes(junk_outcomes, *, table, address_widths, live_ranges, width,
                             block_bits, onehot_bits, address, target, work, extra_onehot,
                             emit=None) -> None:
    r"""Emit the erase of a :class:`QROAMRead` without an enable whose junk was X-measured
    earlier (spec 16, Q-1 variant (b)), inside a rule.

    ``H`` and ``qp.measure(q, reset=True)`` on every ``target`` qubit, then the parities of the
    concatenated outcomes ``target ‖ junk_outcomes`` and the fixup over the read's loaded set,
    exactly as :class:`QROAMErase` does once it has measured ``target ‖ junk``; the one-hot
    register is the reset ``target ‖ extra_onehot``. ``junk_outcomes`` holds one outcome per junk
    qubit, ``(2**block_bits - 1) width`` of them, in the junk register's wire order: slot ``r`` of
    :meth:`QROAMRead.junk_layout` holds outcomes ``r width`` to ``(r + 1) width - 1``.
    ``extra_onehot`` has ``max(2**onehot_bits - width, 0)`` wires in ``|0>``; ``work`` the fixup's
    ``PhaseFixup.required_wires(...)["work"]``. ``emit`` is a :class:`QROAMEmitter` (default:
    emitting); counting, the outcomes are not read.
    """
    emit = QROAMEmitter() if emit is None else emit
    outcomes = [emit.measure_x(qubit) for qubit in target] + list(junk_outcomes)
    _fixup_from_outcomes(emit, outcomes, list(target) + list(extra_onehot), table,
                         address_widths, live_ranges, width, block_bits, None, onehot_bits,
                         address, work, [])


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

- [ ] **Step 4: Add the unprepares from outcomes**

Replace `src/dfthc/ops/prepare.py` with (new: the module docstring's paragraph, `_junk_outcomes`, `abstract_junk_outcomes`, `_AliasSampling.extra_onehot`, `unprepare_from_outcomes` and its resources, the two classes after their baseline siblings, whose docstrings now name them; changed docstrings, for the two junk paths: the module docstring's forced-order and table paragraphs, `OuterPrepare`'s and `InnerPrepare`'s pre- and postconditions and their inverse):

```python
"""The alias-sampling prepares and their measured unprepares (spec 9.7 and 9.8).

**Alias sampling** (spec 6.4). Each prepare samples an index ``i`` over ``[0, L)`` with the
probabilities ``alias_probabilities(alt, keep, b_k)`` of its alias tables, in five steps (dfthcbe's
``OuterPrepare`` and ``InnerPrepare``):

1. ``FlaggedUniformPrep(L, s_u)`` on the index register, its flag ``succ`` and ``succ_work``;
2. ``H`` on the ``b_k`` wires of ``m`` and on the spin wire;
3. ``QROAMRead`` of the alias entry at the address into ``alt ‖ keep ‖ ...`` (the target) and the
   block's other entries (the junk);
4. ``LeftQuantumComparator(keep, m, less_than, cmp_work, "<=")``: ``less_than = [keep <= m]``;
5. one ``CSWAP`` per index wire swaps ``alt`` and the index, controlled by ``less_than``.

The unprepare runs the swaps again, the comparator's adjoint (a measured uncompute, which also
cleans ``cmp_work``), ``QROAMErase`` of the read, the Hadamards, and
``qp.adjoint(FlaggedUniformPrep(...))`` on the same index, ``succ`` and ``succ_work`` wires. The
order is forced: the erase needs ``target ‖ junk`` to hold what the read loaded at the current
address (for an unprepare from outcomes, the target: the junk was X-measured right after the
prepare), so the swap is undone first, and the comparator reads ``keep``, so its adjoint comes
before the erase; nothing between a prepare and its unprepare may write ``succ``, ``succ_work`` or
``cmp_work``.

**Tables.** A read entry is one big-endian integer (spec 6.7) of ``width`` bits, stored as 32-bit
words, least significant first (L-24): ``(alt << b_k1) | keep`` for the outer read
(:meth:`OuterPrepare.read_table`). A prepare packs its compilable tables into the read's dynamic
``leaf_words`` inside its rule (:meth:`dfthc.ops.qroam.QROAMRead.leaf_words_from_table`); its
unprepare passes the same table, in the same entry layout, to
:class:`~dfthc.ops.qroam.QROAMErase`, whose masks are therefore exactly what the read loaded. The
erase's one-hot register is the reset ``target ‖ junk``, so ``2**onehot_bits`` must not exceed
``2**block_bits * width`` (the constructors check it, so ``QROAMErase`` needs no
``extra_onehot``; an unprepare from outcomes, whose junk was measured earlier, takes the wires
beyond its target as ``extra_onehot``).

**Inner read.** Its address is ``x_o ‖ b``, rows ``[0, R C)`` and ``b`` in ``[0, B + 1)`` with the
block bits on ``b``, and its entry ``alt ‖ keep ‖ sgn ‖ d_sgn ‖ r`` with ``r = x_o // C``
(:meth:`InnerPrepare.read_table`). At one-body rows, failed ``x_o`` and failed ``b`` the selected
target entry is zero (spec 8.5; the junk may hold other entries of a visited block): ``keep = 0``,
so ``less_than = 1``, the swap moves ``alt = 0`` into ``b``, and ``sgn = d_sgn = r = 0``.

**Signs** (spec 6.1). :class:`InnerSigns` applies ``Z(sgn) CZ(less_than_b, d_sgn)`` after
``InnerPrepare``: with ``sgn = [w_b < 0]`` and ``d_sgn = sgn XOR sgn[alt]`` loaded for the drawn
``b``, every branch carries the sign of the weight that survives the swap, and every branch whose
selected entry is zero carries +1.

**Rules.** The rules are straight-line and their children are composite keys, so they declare
their resources by hand; T2 runs them at a size in every branch.

**Unprepares from outcomes** (spec 16, Q-1 variant (b)). :class:`OuterUnprepareFromOutcomes` and
:class:`InnerUnprepareFromOutcomes` are the unprepares for a caller that X-measured the junk
right after the prepare (the block encoding does, so the junk wires are free for the angle
registers). They take the prepare's wires except the junk, the prepare's compilable arguments and
the junk outcomes as a dynamic array, and run the unprepare with the erase split: the swaps, the
comparator's adjoint, X measurements of the target only, the parities of the target and junk
outcomes together and the fixup (:func:`~dfthc.ops.qroam.emit_erase_from_outcomes`), the
Hadamards and the adjoint uniform preparation. The fixup's one-hot register is the reset target
and ``extra_onehot``, wires the caller lends in ``|0>``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.ops import MeasurementValue
from pennylane.typing import AbstractArray, Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.qroam import (
    WORD,
    WORD_DTYPE,
    QROAMEmitter,
    QROAMErase,
    QROAMRead,
    emit_erase_from_outcomes,
)
from dfthc.ops.unary import as_resources
from dfthc.ops.uniform import FlaggedUniformPrep
from dfthc.params import lg

COMPARATOR = "<="
"""The keep test writes ``less_than = [keep <= m]``; a set flag swaps in ``alt`` (spec 6.4)."""


# --------------------------------------------------------------------------- shared skeleton


def _words(value: int, width: int) -> tuple[int, ...]:
    """``value`` as ``ceil(width / 32)`` words, least significant first (L-24)."""
    return tuple((value >> (WORD * w)) & (2**WORD - 1) for w in range(-(-width // WORD)))


def _join(*registers) -> list:
    """The wires of ``registers``, concatenated in order."""
    return [wire for register in registers for wire in register]


def _check_table(name: str, values, shape: tuple[int, ...], bound: int) -> tuple:
    """``values`` as nested tuples of Python ints of ``shape`` (one or two sizes), every entry in
    ``[0, bound)``; ``ValueError`` otherwise."""
    values = canonical_ints(name, values)
    rows = [values] if len(shape) == 1 else values
    if (not isinstance(values, tuple) or len(values) != shape[0]
            or any(not isinstance(row, tuple) or len(row) != shape[-1] for row in rows)
            or any(not isinstance(entry, int) for row in rows for entry in row)):
        raise ValueError(f"{name} must be a table of ints of shape {shape}")
    bad = [entry for row in rows for entry in row if not 0 <= entry < bound]
    if bad:
        raise ValueError(f"{name} entries must lie in [0, {bound}), got {bad[0]}")
    return values


def _check_onehot(name: str, onehot_bits: int, read: dict) -> None:
    """``ValueError`` unless the erase's one-hot register fits in the reset ``target ‖ junk``."""
    wires = 2 ** read["block_bits"] * read["width"]
    if 2**onehot_bits > wires:
        raise ValueError(f"the erase's one-hot over {name} = {onehot_bits} bits needs "
                         f"{2**onehot_bits} wires, more than the {wires} target and junk wires")


def _junk_outcomes(junk_outcomes, size: int):
    """Canonicalize the dynamic ``junk_outcomes`` once (L-23), one outcome per junk qubit: under
    capture a bool array of shape ``(size,)`` (empty for ``size = 0``); in tape mode an object
    array whose every entry is a ``MeasurementValue``, so that both modes emit the same fixup
    conditionals (K-10). Abstract arrays and tracers pass with their shape checked.
    ``ValueError`` otherwise."""
    if isinstance(junk_outcomes, AbstractArray) or qp.math.is_tracer(junk_outcomes):
        if tuple(junk_outcomes.shape) != (size,):
            raise ValueError(f"junk_outcomes must have shape ({size},), got "
                             f"{tuple(junk_outcomes.shape)}")
        return junk_outcomes
    values = list(junk_outcomes)
    if len(values) != size:
        raise ValueError(f"need one outcome per junk qubit ({size}), got {len(values)}")
    if qp.capture.enabled():
        if not values:
            return jnp.zeros((0,), dtype=bool)
        return jnp.stack([jnp.asarray(value, dtype=bool) for value in values])
    for i, value in enumerate(values):
        if not isinstance(value, MeasurementValue):
            raise ValueError(f"in tape mode every junk outcome must be a MeasurementValue; "
                             f"entry {i} is {value!r}")
    return np.asarray(values, dtype=object)


def abstract_junk_outcomes(size: int) -> AbstractArray:
    """The ``junk_outcomes`` of a resource key that declares an unprepare from outcomes: a bool
    array under capture, an object array of ``MeasurementValue``s in tape mode, as
    :func:`_junk_outcomes` canonicalizes them (spec 7.4 rule 9: the declarations' only
    mode-specific code). Call it before :func:`~dfthc.compile.resource_key`, which pauses
    capture."""
    return AbstractArray((size,), bool if qp.capture.enabled() else object)


@dataclass(frozen=True)
class _AliasSampling:
    """The alias-sampling rule skeleton of both prepares and unprepares (module docstring).

    ``L`` values of the sampled index, ``s_u``-bit uniform rotations, ``b_k`` keep bits, the
    read's shape ``read`` (``address_widths``, ``live_ranges``, ``width``, ``block_bits``) and the
    erase's ``onehot_bits``.
    """

    L: int
    s_u: int
    b_k: int
    read: dict
    onehot_bits: int

    def work_sizes(self) -> tuple[int, int, int]:
        """Work wires of the uniform preparation, the read and the erase."""
        return (FlaggedUniformPrep.required_wires(self.L, self.s_u)["work"],
                QROAMRead.required_wires(**self.read, control_value=None)["work"],
                QROAMErase.required_wires((), **self.read, control_value=None,
                                          onehot_bits=self.onehot_bits)["work"])

    def sizes(self) -> dict[str, int]:
        """Sizes of the skeleton's registers; ``work`` is the largest of the three needs."""
        uniform = FlaggedUniformPrep.required_wires(self.L, self.s_u)
        return {"index": lg(self.L), "succ": 1, "succ_work": uniform["succ_work"],
                "m": self.b_k, "less_than": 1, "cmp_work": self.b_k - 1, "alt": lg(self.L),
                "keep": self.b_k,
                "junk": (2 ** self.read["block_bits"] - 1) * self.read["width"],
                "phase_grad": self.s_u, "work": max(self.work_sizes())}

    def extra_onehot(self) -> int:
        """Wires of the erase's one-hot register beyond the reset target, when the junk was
        measured earlier: ``max(2**onehot_bits - width, 0)`` (0 without a one-hot)."""
        return max(2**self.onehot_bits - self.read["width"], 0) if self.onehot_bits else 0

    def prepare(self, table, *, index, succ, succ_work, hadamards, m, less_than, cmp_work, alt,
                keep, address, target, junk, phase_grad, work) -> None:
        """Steps 1 to 5; the read's ``leaf_words`` come from ``table``."""
        uniform, read, _ = self.work_sizes()
        FlaggedUniformPrep(self.L, self.s_u, index, succ, succ_work, phase_grad[:self.s_u],
                           work[:uniform])
        for wire in hadamards:
            qp.H(wire)
        QROAMRead(QROAMRead.leaf_words_from_table(table, **self.read), **self.read,
                  control_value=None, address=address, target=target, junk=junk,
                  work=work[:read], control=[])
        qp.LeftQuantumComparator(keep, m, less_than, cmp_work, COMPARATOR)
        for a, i in zip(alt, index, strict=True):
            qp.CSWAP([less_than[0], a, i])

    def unprepare(self, table, *, index, succ, succ_work, hadamards, m, less_than, cmp_work,
                  alt, keep, address, target, junk, phase_grad, work) -> None:
        """Steps 5 to 1 undone, with the measured comparator uncompute and the measured erase."""
        uniform, _, erase = self.work_sizes()
        for a, i in zip(alt, index, strict=True):
            qp.CSWAP([less_than[0], a, i])
        qp.adjoint(qp.LeftQuantumComparator(keep, m, less_than, cmp_work, COMPARATOR))
        QROAMErase(table, **self.read, control_value=None, onehot_bits=self.onehot_bits,
                   address=address, target=target, junk=junk, work=work[:erase], control=[],
                   extra_onehot=[])
        for wire in hadamards:
            qp.H(wire)
        qp.adjoint(FlaggedUniformPrep(self.L, self.s_u, index, succ, succ_work,
                                      phase_grad[:self.s_u], work[:uniform]))

    def unprepare_from_outcomes(self, junk_outcomes, table, *, index, succ, succ_work, hadamards,
                                m, less_than, cmp_work, alt, keep, address, target, phase_grad,
                                work, extra_onehot) -> None:
        """:meth:`unprepare` with the erase split (spec 16, Q-1 variant (b)): the caller measured
        the junk; the target is measured here and the fixup reads both outcome vectors."""
        uniform, _, erase = self.work_sizes()
        for a, i in zip(alt, index, strict=True):
            qp.CSWAP([less_than[0], a, i])
        qp.adjoint(qp.LeftQuantumComparator(keep, m, less_than, cmp_work, COMPARATOR))
        emit_erase_from_outcomes(junk_outcomes, table=table, **self.read,
                                 onehot_bits=self.onehot_bits, address=address, target=target,
                                 work=work[:erase], extra_onehot=extra_onehot)
        for wire in hadamards:
            qp.H(wire)
        qp.adjoint(FlaggedUniformPrep(self.L, self.s_u, index, succ, succ_work,
                                      phase_grad[:self.s_u], work[:uniform]))

    def _uniform_key(self, adjoint: bool):
        sizes = FlaggedUniformPrep.required_wires(self.L, self.s_u)

        def build():
            op = FlaggedUniformPrep(self.L, self.s_u, Wire[sizes["target"]], Wire[1],
                                    Wire[sizes["succ_work"]], Wire[self.s_u], Wire[sizes["work"]])
            return qp.adjoint(op) if adjoint else op

        return resource_key(build)

    def _comparator_key(self, adjoint: bool):
        n = self.b_k

        def build():
            op = qp.LeftQuantumComparator(Wire[n], Wire[n], Wire[1], Wire[n - 1], COMPARATOR)
            return qp.adjoint(op) if adjoint else op

        return resource_key(build)

    def prepare_resources(self) -> dict:
        words = AbstractArray(QROAMRead.leaf_words_shape(**self.read), WORD_DTYPE)
        sizes = QROAMRead.required_wires(**self.read, control_value=None)
        read = resource_key(lambda: QROAMRead(words, **self.read, control_value=None,
                                              **{name: Wire[n] for name, n in sizes.items()}))
        return {self._uniform_key(False): 1, qp.H: self.b_k + 1, read: 1,
                self._comparator_key(False): 1, qp.CSWAP: lg(self.L)}

    def unprepare_resources(self, table) -> dict:
        sizes = QROAMErase.required_wires(table, **self.read, control_value=None,
                                          onehot_bits=self.onehot_bits)
        erase = resource_key(lambda: QROAMErase(table, **self.read, control_value=None,
                                                onehot_bits=self.onehot_bits,
                                                **{name: Wire[n] for name, n in sizes.items()}))
        return {qp.CSWAP: lg(self.L), self._comparator_key(True): 1, erase: 1,
                qp.H: self.b_k + 1, self._uniform_key(True): 1}

    def unprepare_from_outcomes_resources(self, table) -> dict:
        """The declaration of :meth:`unprepare_from_outcomes`: the swaps, the comparator's
        adjoint, the Hadamards and the adjoint uniform preparation as keys, and the split erase
        gate by gate, counted by running it."""
        _, _, erase = self.work_sizes()
        counter = QROAMEmitter(counting=True)
        emit_erase_from_outcomes(
            [None] * self.sizes()["junk"], table=table, **self.read, onehot_bits=self.onehot_bits,
            address=range(sum(self.read["address_widths"])), target=range(self.read["width"]),
            work=range(erase), extra_onehot=range(self.extra_onehot()), emit=counter)
        counts = Counter({qp.CSWAP: lg(self.L), self._comparator_key(True): 1, qp.H: self.b_k + 1,
                          self._uniform_key(True): 1})
        counts.update(as_resources(counter.counts))
        return dict(counts)


# --------------------------------------------------------------------------- OuterPrepare


def _check_outer(X_o, alt, keep, b_k1, k1, ke1, s_u) -> tuple:
    X_o = check_int("X_o", X_o, low=2)
    b_k1 = check_int("b_k1", b_k1, low=1)
    k1 = check_int("k1", k1, low=0, high=lg(X_o))
    ke1 = check_int("ke1", ke1, low=0, high=lg(X_o))
    s_u = check_int("s_u", s_u, low=3)
    alt = _check_table("alt", alt, (X_o,), X_o)
    keep = _check_table("keep", keep, (X_o,), 2**b_k1)
    _check_onehot("ke1", ke1, OuterPrepare.read_shape(X_o, b_k1, k1))
    return X_o, alt, keep, b_k1, k1, ke1, s_u


def _outer(X_o, b_k1, k1, ke1, s_u) -> _AliasSampling:
    return _AliasSampling(X_o, s_u, b_k1, OuterPrepare.read_shape(X_o, b_k1, k1), ke1)


def _outer_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                     alt_x_o, keep_x_o, junk_x_o, phase_grad, work) -> dict:
    """The skeleton's registers for the outer prepare: the read's address is ``x_o``."""
    return {"index": x_o, "succ": succ_x_o, "succ_work": succ_work_x_o,
            "hadamards": _join(m_x_o, sigma_1), "m": m_x_o, "less_than": less_than_x_o,
            "cmp_work": cmp_work_x_o, "alt": alt_x_o, "keep": keep_x_o, "address": x_o,
            "target": _join(alt_x_o, keep_x_o), "junk": junk_x_o, "phase_grad": phase_grad,
            "work": work}


_OUTER_WIRES = ("x_o", "succ_x_o", "succ_work_x_o", "sigma_1", "m_x_o", "less_than_x_o",
                "cmp_work_x_o", "alt_x_o", "keep_x_o", "junk_x_o", "phase_grad", "work")


class OuterPrepare(Operator2):
    r"""Prepare the outer index ``x_o`` with the alias distribution of the outer weights
    (steps 1 to 4).

    **Action.** From ``|0>`` on every register, with ``phase_grad`` holding the phase-gradient
    state, let :math:`x_0` be the value the uniform preparation draws on ``x_o``,
    :math:`\tilde p` = ``alias_probabilities(alt, keep, b_k1)`` (spec 6.4) and
    :math:`P_f` = ``uniform_failure_probability(X_o, s_u)``. The output is a sum of basis terms,
    one per :math:`x_0`, value of ``m_x_o`` and value of ``sigma_1``. A term with
    :math:`x_0 < X_o` has ``succ_x_o = 1`` and the amplitude
    :math:`e^{i\varphi}\sqrt{(1 - P_f)/X_o}\,/\sqrt{2^{b_{k1} + 1}}`, where :math:`\varphi` is the
    one phase that :class:`~dfthc.ops.uniform.FlaggedUniformPrep` gives all its ``succ = 1``
    amplitudes (0 when ``X_o`` is a power of two): steps 2 to 5 permute basis states and add no
    phase. So the ``succ_x_o = 1`` part is
    :math:`e^{i\varphi}\sqrt{1 - P_f}\sum_x \sqrt{\tilde p(x)}\,|x\rangle|\mathrm{garb}_x\rangle
    \otimes |+\rangle_{\sigma_1}` with normalized garbage states, and ``x_o`` holds :math:`x` with
    probability :math:`(1 - P_f)\tilde p(x)`. The failure part, of weight :math:`P_f`, comes from
    :math:`x_0 \ge X_o`, where the selected target entry ``alt_x_o ‖ keep_x_o`` is zero (the junk
    may hold other entries of a visited block): ``less_than = 1``, and the swap leaves ``x_o = 0``
    and :math:`x_0` in ``alt_x_o``, with ``succ_x_o = 0``.

    **Registers.** ``w_x = lg(X_o)``, ``l_x`` the odd part of ``X_o``, ``K1 = 2**k1``.

    ================= ============================= ============== ===============================
    name              width                         input          output
    ================= ============================= ============== ===============================
    ``x_o``           ``w_x``                       ``|0>``        the sampled index
    ``succ_x_o``      1                             ``|0>``        ``[x_0 < X_o]``
    ``succ_work_x_o`` ``max(lg(l_x) - 1, 0)``       ``|0>``        dirty until the unprepare
    ``sigma_1``       1                             ``|0>``        ``|+>``
    ``m_x_o``         ``b_k1``                      ``|0>``        uniform over ``[0, 2**b_k1)``
    ``less_than_x_o`` 1                             ``|0>``        ``[keep[x_0] <= m]``
    ``cmp_work_x_o``  ``b_k1 - 1``                  ``|0>``        dirty until the unprepare
    ``alt_x_o``       ``w_x``                       ``|0>``        ``alt[x_0]``, or ``x_0`` if
                                                                   swapped
    ``keep_x_o``      ``b_k1``                      ``|0>``        ``keep[x_0]``
    ``junk_x_o``      ``(K1 - 1)(w_x + b_k1)``      ``|0>``        the read block's other entries
    ``phase_grad``    at least ``s_u``              gradient state unchanged (catalysed)
    ``work``          :meth:`required_wires`        ``|0>``        ``|0>``
    ================= ============================= ============== ===============================

    **Compilable arguments.** ``X_o`` (int >= 2, the values of ``x_o``); ``alt`` (``X_o`` ints in
    ``[0, X_o)``) and ``keep`` (``X_o`` ints in ``[0, 2**b_k1)``), the outer alias tables
    (``WalkTables.outer_alt``, ``outer_keep``); ``b_k1`` (int >= 1); ``k1`` (the read's block
    bits, ``0 <= k1 <= w_x``); ``ke1`` (the erase's one-hot bits, ``0 <= ke1 <= w_x``, with
    ``2**ke1 <= K1 (w_x + b_k1)``; used by :class:`OuterUnprepare`, checked here so that one
    argument tuple serves both); ``s_u`` (int >= 3, the uniform preparation's rotation bits).

    **Pre- and postconditions.** As in the table. Until the unprepare, nothing may write
    ``succ_x_o``, ``succ_work_x_o``, ``less_than_x_o``, ``cmp_work_x_o``, ``m_x_o``, ``alt_x_o``
    or ``keep_x_o``, and ``x_o`` may be used only as a control or changed by operations that are
    undone before the unprepare. ``junk_x_o`` takes one of two paths: (a) nothing writes it until
    :class:`OuterUnprepare`; or (b) the caller applies ``H`` and ``qp.measure(q, reset=True)`` to
    each of its qubits right after the prepare, keeps the outcomes in wire order (the slots of
    :meth:`QROAMRead.junk_layout <dfthc.ops.qroam.QROAMRead.junk_layout>`), may use its wires for
    other registers in between, and calls :class:`OuterUnprepareFromOutcomes` with the outcomes
    (spec 16, Q-1 variant (b); the block encoding does).

    **Decomposition** (spec 9.7). ``FlaggedUniformPrep(X_o, s_u)`` on ``x_o``, ``succ_x_o``,
    ``succ_work_x_o``, ``phase_grad[:s_u]``; ``H`` on ``m_x_o`` and ``sigma_1``; ``QROAMRead`` of
    :meth:`read_table` at ``x_o`` (shape :meth:`read_shape`, no enable) into
    ``alt_x_o ‖ keep_x_o`` and ``junk_x_o``, its ``leaf_words`` built here from the table;
    ``LeftQuantumComparator(keep_x_o, m_x_o, less_than_x_o, cmp_work_x_o, "<=")``; ``w_x``
    ``CSWAP``\s of ``alt_x_o`` and ``x_o`` controlled by ``less_than_x_o``. Each child gets the
    start of ``work`` it needs.

    **Adjoint and control.** Never adjointed or controlled: the read has loops and data
    conditionals (spec 7.4 rules 4 and 6). The inverse is the operator :class:`OuterUnprepare`
    (path (a) above) or :class:`OuterUnprepareFromOutcomes` (path (b)).

    **Cost.** :func:`dfthc.costs.outer_prepare`: :math:`T = F(X_o) + T_{read,out} + b_{k1} +
    w_x` (uniform preparation, read, comparator, swaps), ``M = 0``; ``A`` is the uniform
    preparation's :math:`2(n - 1) + 2(s_u - 1)` (0 if ``l_x = 1``; ``n = lg(l_x)``) plus the
    read's iteration count ``U``. At FeMoco-54: :math:`30 + 156 + 15 + 9 = 210`. The ``CNOT`` and
    ``PauliX`` counts are upper bounds: the read declares a load gate for every bit
    (``exact=False``, spec 9.3).

    **Paper and dfthcbe.** Steps 1 to 4 of dfthcbe's README circuit, its ``OuterPrepare``
    (``dfthcbe/prepare.py``), with PennyLane's ``LeftQuantumComparator`` in place of
    ``LinearDepthGreaterThan`` and its ``X``; the outer superposition of arXiv:2502.15882v1
    App. B.1 and B.7.2, steps 1 and 2, Eqs. (97) and (100) to (102) (Table 7).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import OuterPrepare

        sizes = OuterPrepare.required_wires(5, 2, 1, 1, 3)  # X_o, b_k1, k1, ke1, s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        program = lambda: OuterPrepare(5, (1, 0, 0, 2, 3), (3, 2, 1, 0, 2), 2, 1, 1, 3, **wires)
        compiled_resources(program, num_wires=start).toffoli  # F 10 + read 2 + 5, + 2 + 3 = 22
    """

    compilable_argnames = ("X_o", "alt", "keep", "b_k1", "k1", "ke1", "s_u")
    wire_argnames = _OUTER_WIRES

    def __init__(self, X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o,
                 sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o,
                 phase_grad, work):
        args = _check_outer(X_o, alt, keep, b_k1, k1, ke1, s_u)
        wires = (x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                 alt_x_o, keep_x_o, junk_x_o, phase_grad, work)
        check_wires("OuterPrepare", dict(zip(_OUTER_WIRES, wires, strict=True)),
                    self.required_wires(args[0], *args[3:]), at_least=("phase_grad",))
        super().__init__(*args, *wires)

    @staticmethod
    def required_wires(X_o, b_k1, k1, ke1, s_u) -> dict[str, int]:
        """Register sizes; ``phase_grad`` is a minimum, every other size is exact. ``work`` is
        the largest need of the uniform preparation, the read and :class:`OuterUnprepare`'s
        erase, so that both classes take the same wires."""
        sizes = _outer(X_o, b_k1, k1, ke1, s_u).sizes()
        return {"x_o": sizes["index"], "succ_x_o": 1, "succ_work_x_o": sizes["succ_work"],
                "sigma_1": 1, "m_x_o": b_k1, "less_than_x_o": 1, "cmp_work_x_o": b_k1 - 1,
                "alt_x_o": sizes["alt"], "keep_x_o": b_k1, "junk_x_o": sizes["junk"],
                "phase_grad": s_u, "work": sizes["work"]}

    @staticmethod
    def read_shape(X_o, b_k1, k1) -> dict:
        """The outer read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``:
        ``x_o`` over ``[0, X_o)`` on ``lg(X_o)`` bits, entries of ``d1 = lg(X_o) + b_k1`` bits."""
        return {"address_widths": (lg(X_o),), "live_ranges": ((0, X_o),),
                "width": lg(X_o) + b_k1, "block_bits": k1}

    @staticmethod
    def read_table(X_o, alt, keep, b_k1) -> tuple:
        """The outer read's table: ``(x, words)`` with entry ``(alt[x] << b_k1) | keep[x]`` for
        every ``x < X_o``, the target ``alt_x_o ‖ keep_x_o`` big-endian."""
        width = lg(X_o) + b_k1
        return tuple((x, _words((alt[x] << b_k1) | keep[x], width)) for x in range(X_o))


def _outer_prepare_resources(X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o,
                             sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o,
                             junk_x_o, phase_grad, work):
    return _outer(X_o, b_k1, k1, ke1, s_u).prepare_resources()


@qp.register_resources(_outer_prepare_resources)
def _outer_prepare(X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o, sigma_1,
                   m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o, phase_grad,
                   work):
    _outer(X_o, b_k1, k1, ke1, s_u).prepare(
        OuterPrepare.read_table(X_o, alt, keep, b_k1),
        **_outer_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                           cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o, phase_grad, work))


register_rule(OuterPrepare, _outer_prepare)


# --------------------------------------------------------------------------- OuterUnprepare


class OuterUnprepare(Operator2):
    r"""The measured unprepare of :class:`OuterPrepare` (steps 24 to 27): its explicit adjoint
    class (spec 7.4 rule 6).

    **Action.** ``OuterUnprepare`` after ``OuterPrepare`` with the same arguments is the
    identity on the prepare's input (every register ``|0>``, ``phase_grad`` unchanged), for every
    measurement outcome; operations in between must respect :class:`OuterPrepare`'s
    postconditions.

    **Registers.** Those of :class:`OuterPrepare`; the input is the prepare's output.

    ================= ============================= ============================== ==============
    name              width                         input                          output
    ================= ============================= ============================== ==============
    ``x_o``           ``w_x``                       the sampled index              ``|0>``
    ``succ_x_o``      1                             ``[x_0 < X_o]``                ``|0>``
    ``succ_work_x_o`` ``max(lg(l_x) - 1, 0)``       the prepare's dirty wires      ``|0>``
    ``sigma_1``       1                             ``|+>``                        ``|0>``
    ``m_x_o``         ``b_k1``                      uniform                        ``|0>``
    ``less_than_x_o`` 1                             ``[keep[x_0] <= m]``           ``|0>``
    ``cmp_work_x_o``  ``b_k1 - 1``                  the prepare's dirty wires      ``|0>``
    ``alt_x_o``       ``w_x``                       ``alt[x_0]``, or ``x_0``       ``|0>``
    ``keep_x_o``      ``b_k1``                      ``keep[x_0]``                  ``|0>``
    ``junk_x_o``      ``(K1 - 1)(w_x + b_k1)``      the read block's other entries ``|0>``
    ``phase_grad``    at least ``s_u``              gradient state                 unchanged
    ``work``          :meth:`required_wires`        ``|0>``                        ``|0>``
    ================= ============================= ============================== ==============

    **Compilable arguments.** Those of :class:`OuterPrepare`, with the same domains; ``ke1`` sets
    the erase's one-hot bits (spec 9.5).

    **Pre- and postconditions.** The registers hold what :class:`OuterPrepare` produced, up to
    operations that use ``x_o`` only as a control or are undone; ``work`` is ``|0>``. Afterwards
    every register but ``phase_grad`` is ``|0>``.

    **Decomposition** (spec 9.7). The ``w_x`` ``CSWAP``\s (self-inverse);
    ``qp.adjoint(LeftQuantumComparator(keep_x_o, m_x_o, less_than_x_o, cmp_work_x_o, "<="))``,
    whose rule is a measured uncompute (it also cleans ``cmp_work_x_o``); ``QROAMErase`` of
    :meth:`OuterPrepare.read_table` with ``onehot_bits = ke1``, no enable and no
    ``extra_onehot``; ``H`` on ``m_x_o`` and ``sigma_1``; ``qp.adjoint(FlaggedUniformPrep(X_o,
    s_u, ...))`` on the prepare's wires, the auto-derived leaf adjoint.

    **Adjoint and control.** Never adjointed or controlled: it measures (spec 7.4 rule 5), and it
    is itself the adjoint class of :class:`OuterPrepare`. The walk's block encoding measures the
    junk right after the prepare and calls the sibling :class:`OuterUnprepareFromOutcomes`
    instead (spec 16, Q-1 variant (b)); this class serves callers that keep the junk.

    **Cost.** :func:`dfthc.costs.outer_unprepare`: :math:`T = w_x + T_{erase,out} +
    F^\dagger(X_o)`, :math:`M = K_1 d_1` (every target and junk qubit); ``A`` is the comparator's
    ``b_k1`` plus the erase's ``T`` plus the adjoint uniform preparation's
    :math:`3(n - 1) + 2(s_u - 1)` (0 if ``l_x = 1``). At FeMoco-54: :math:`9 + 36 + 24 = 69`,
    ``M = 96``. The ``PauliX`` count is an upper bound: each measurement's reset is counted at
    its maximum (spec 7.6).

    **Paper and dfthcbe.** Steps 24 to 27 of dfthcbe's README circuit, the adjoint of its
    ``OuterPrepare`` with the measured erase; arXiv:2502.15882v1 App. B.7.2, steps 10 and 11,
    Eqs. (126) and (127) (Table 7). Our comparator adjoint costs no Toffoli (the paper's and
    dfthcbe's ``b_k1`` term): the comparator's ``b_k1 - 1`` work wires stay dirty until here,
    the saving the paper notes after Eq. (127).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import OuterUnprepare

        sizes = OuterUnprepare.required_wires(5, 2, 1, 1, 3)  # X_o, b_k1, k1, ke1, s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        program = lambda: OuterUnprepare(5, (1, 0, 0, 2, 3), (3, 2, 1, 0, 2), 2, 1, 1, 3,
                                         **wires)
        compiled_resources(program, num_wires=start).measurements  # K1 d1 = 2 * 5 = 10
    """

    compilable_argnames = ("X_o", "alt", "keep", "b_k1", "k1", "ke1", "s_u")
    wire_argnames = _OUTER_WIRES

    def __init__(self, X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o,
                 sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o,
                 phase_grad, work):
        args = _check_outer(X_o, alt, keep, b_k1, k1, ke1, s_u)
        wires = (x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                 alt_x_o, keep_x_o, junk_x_o, phase_grad, work)
        check_wires("OuterUnprepare", dict(zip(_OUTER_WIRES, wires, strict=True)),
                    self.required_wires(args[0], *args[3:]), at_least=("phase_grad",))
        super().__init__(*args, *wires)

    @staticmethod
    def required_wires(X_o, b_k1, k1, ke1, s_u) -> dict[str, int]:
        """:meth:`OuterPrepare.required_wires`."""
        return OuterPrepare.required_wires(X_o, b_k1, k1, ke1, s_u)


def _outer_unprepare_resources(X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o,
                               succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                               alt_x_o, keep_x_o, junk_x_o, phase_grad, work):
    return _outer(X_o, b_k1, k1, ke1, s_u).unprepare_resources(
        OuterPrepare.read_table(X_o, alt, keep, b_k1))


@qp.register_resources(_outer_unprepare_resources)
def _outer_unprepare(X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o, sigma_1,
                     m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o,
                     phase_grad, work):
    _outer(X_o, b_k1, k1, ke1, s_u).unprepare(
        OuterPrepare.read_table(X_o, alt, keep, b_k1),
        **_outer_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                           cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o, phase_grad, work))


register_rule(OuterUnprepare, _outer_unprepare)


# ----------------------------------------------------------------------- OuterUnprepareFromOutcomes

_OUTER_FROM_OUTCOMES_WIRES = ("x_o", "succ_x_o", "succ_work_x_o", "sigma_1", "m_x_o",
                              "less_than_x_o", "cmp_work_x_o", "alt_x_o", "keep_x_o",
                              "phase_grad", "work", "extra_onehot")


class OuterUnprepareFromOutcomes(Operator2):
    r"""The measured unprepare of :class:`OuterPrepare` (steps 24 to 27) for a caller that
    X-measured ``junk_x_o`` right after the prepare (spec 16, Q-1 variant (b)).

    **Action.** After ``OuterPrepare``, ``H`` and ``qp.measure(q, reset=True)`` on every qubit
    of ``junk_x_o`` (the outcomes :math:`\xi_{junk}`) and operations that respect
    :class:`OuterPrepare`'s postconditions, ``OuterUnprepareFromOutcomes`` with the outcomes
    :math:`\xi_{junk}` and the prepare's compilable arguments is the identity on the prepare's
    input (every register ``|0>``, ``phase_grad`` unchanged), for every outcome of the junk and
    target measurements: :class:`OuterUnprepare` with its erase's junk measurements moved to the
    caller. The junk wires are free from their measurement on; the block encoding gives them to
    the angle registers (spec 10.2).

    **Registers.** Those of :class:`OuterPrepare` without ``junk_x_o``, then ``extra_onehot``.

    ================= ============================= ============================== ==============
    name              width                         input                          output
    ================= ============================= ============================== ==============
    ``x_o``           ``w_x``                       the sampled index              ``|0>``
    ``succ_x_o``      1                             ``[x_0 < X_o]``                ``|0>``
    ``succ_work_x_o`` ``max(lg(l_x) - 1, 0)``       the prepare's dirty wires      ``|0>``
    ``sigma_1``       1                             ``|+>``                        ``|0>``
    ``m_x_o``         ``b_k1``                      uniform                        ``|0>``
    ``less_than_x_o`` 1                             ``[keep[x_0] <= m]``           ``|0>``
    ``cmp_work_x_o``  ``b_k1 - 1``                  the prepare's dirty wires      ``|0>``
    ``alt_x_o``       ``w_x``                       ``alt[x_0]``, or ``x_0``       ``|0>``
    ``keep_x_o``      ``b_k1``                      ``keep[x_0]``                  ``|0>``
    ``phase_grad``    at least ``s_u``              gradient state                 unchanged
    ``work``          :meth:`required_wires`        ``|0>``                        ``|0>``
    ``extra_onehot``  ``max(2**ke1 - d1, 0)``       ``|0>``                        ``|0>``
    ================= ============================= ============================== ==============

    **Arguments.** Dynamic: ``junk_outcomes``, the :math:`(K_1 - 1) d_1` outcomes of the junk
    measurements in the order of the ``junk_x_o`` wires: slot ``r`` of
    :meth:`QROAMRead.junk_layout <dfthc.ops.qroam.QROAMRead.junk_layout>` gives outcomes
    ``r d1`` to ``(r + 1) d1 - 1`` (``d1 = w_x + b_k1``). Under capture it becomes a bool array;
    in tape mode every entry must be a ``MeasurementValue``, so both modes emit the same fixup
    (K-10). Compilable: those of :class:`OuterPrepare`, with the same domains; ``ke1`` sets the
    fixup's one-hot bits (spec 9.5).

    **Pre- and postconditions.** The registers hold what :class:`OuterPrepare` produced, up to
    operations that use ``x_o`` only as a control or are undone; ``junk_x_o`` was X-measured with
    reset right after the prepare, ``junk_outcomes`` are its outcomes in wire order, and its
    wires may serve other registers in between; ``work`` and ``extra_onehot`` are ``|0>``.
    Afterwards every register but ``phase_grad`` is ``|0>``.

    **Decomposition** (spec 16, Q-1 variant (b)). The ``w_x`` ``CSWAP``\s;
    ``qp.adjoint(LeftQuantumComparator(keep_x_o, m_x_o, less_than_x_o, cmp_work_x_o, "<="))``
    (a measured uncompute); ``H`` and ``qp.measure(q, reset=True)`` on every qubit of
    ``alt_x_o ‖ keep_x_o``; the parities :math:`f(a) = \xi \cdot g(a)` of the outcomes
    :math:`\xi` = target ‖ ``junk_outcomes`` with :math:`g(a)` =
    :meth:`QROAMRead.loaded_bits <dfthc.ops.qroam.QROAMRead.loaded_bits>` of
    :meth:`OuterPrepare.read_table` over the read's loaded set (masks built here from the
    compilable tables, never passed as data), by :func:`dfthc.compile.mcm_parities`; the fixup
    over that set with the reset ``alt_x_o ‖ keep_x_o ‖ extra_onehot`` as its one-hot register
    (:func:`dfthc.ops.qroam.emit_erase_from_outcomes`); ``H`` on ``m_x_o`` and ``sigma_1``;
    ``qp.adjoint(FlaggedUniformPrep(X_o, s_u, ...))``, the auto-derived leaf adjoint.

    **Adjoint and control.** Never adjointed or controlled: it measures (spec 7.4 rule 5). It is
    a sibling of :class:`OuterUnprepare`, the same inverse of :class:`OuterPrepare` with another
    signature, not a modified form of a composite (spec 7.4 rule 6).

    **Cost.** :func:`dfthc.costs.outer_unprepare_from_outcomes`: the ``T`` of
    :func:`dfthc.costs.outer_unprepare`, :math:`w_x + T_{erase,out} + F^\dagger(X_o)`, and
    ``M = d1`` (the target qubits; the caller's :math:`(K_1 - 1) d_1` junk measurements make up
    :class:`OuterUnprepare`'s :math:`K_1 d_1`); ``A`` as :class:`OuterUnprepare`'s, the
    comparator's ``b_k1`` plus the fixup's ``T`` plus the adjoint uniform preparation's
    :math:`3(n - 1) + 2(s_u - 1)`. At FeMoco-54: :math:`9 + 36 + 24 = 69`, ``M = 24`` (and 72
    junk measurements in the block encoding). The ``PauliX`` count is an upper bound (spec 7.6).

    **Paper and dfthcbe.** Steps 24 to 27 of dfthcbe's README circuit with the junk measured
    early, as the paper does: arXiv:2502.15882v1 App. B.7.2, step 2 (":math:`(2^{k_1} - 1)
    b_1` qubits are erased (via an X measurement and a later phase fixup)", after Eq. (104)) and
    steps 10 and 11, Eqs. (126) and (127) (Table 7). The walk measures the junk right after
    step 4, the end of the prepare, before any angle load; the comparator and the swaps of steps
    3 and 4 do not touch it, so this equals measuring right after the read (step 2). dfthcbe's
    adjoint ``OuterPrepare`` measures target and junk together at the erase, as
    :class:`OuterUnprepare` does.

    **Example.**

    .. code-block:: python

        import numpy as np
        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import OuterUnprepareFromOutcomes

        sizes = OuterUnprepareFromOutcomes.required_wires(5, 2, 1, 1, 3)  # X_o b_k1 k1 ke1 s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        junk = np.zeros(OuterUnprepareFromOutcomes.junk_size(5, 2, 1), dtype=bool)
        program = lambda: OuterUnprepareFromOutcomes(junk, 5, (1, 0, 0, 2, 3), (3, 2, 1, 0, 2),
                                                     2, 1, 1, 3, **wires)
        compiled_resources(program, num_wires=start).measurements  # d1 = 5
    """

    dynamic_argnames = ("junk_outcomes",)
    compilable_argnames = ("X_o", "alt", "keep", "b_k1", "k1", "ke1", "s_u")
    wire_argnames = _OUTER_FROM_OUTCOMES_WIRES

    def __init__(self, junk_outcomes, X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o,
                 succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o,
                 phase_grad, work, extra_onehot):
        args = _check_outer(X_o, alt, keep, b_k1, k1, ke1, s_u)
        junk_outcomes = _junk_outcomes(junk_outcomes, self.junk_size(args[0], args[3], args[4]))
        wires = (x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o,
                 alt_x_o, keep_x_o, phase_grad, work, extra_onehot)
        check_wires("OuterUnprepareFromOutcomes",
                    dict(zip(_OUTER_FROM_OUTCOMES_WIRES, wires, strict=True)),
                    self.required_wires(args[0], *args[3:]), at_least=("phase_grad",))
        super().__init__(junk_outcomes, *args, *wires)

    @staticmethod
    def required_wires(X_o, b_k1, k1, ke1, s_u) -> dict[str, int]:
        """:meth:`OuterPrepare.required_wires` without ``junk_x_o``, then ``extra_onehot``:
        ``max(2**ke1 - d1, 0)`` wires (0 for ``ke1 = 0``), the one-hot beyond the reset target."""
        sizes = OuterPrepare.required_wires(X_o, b_k1, k1, ke1, s_u)
        del sizes["junk_x_o"]
        return sizes | {"extra_onehot": _outer(X_o, b_k1, k1, ke1, s_u).extra_onehot()}

    @staticmethod
    def junk_size(X_o, b_k1, k1) -> int:
        """The number of junk outcomes: ``(K1 - 1) d1``, the width of ``junk_x_o``."""
        return (2**k1 - 1) * (lg(X_o) + b_k1)


def _outer_from_outcomes_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                                   cmp_work_x_o, alt_x_o, keep_x_o, phase_grad, work,
                                   extra_onehot) -> dict:
    registers = _outer_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                                 cmp_work_x_o, alt_x_o, keep_x_o, (), phase_grad, work)
    del registers["junk"]
    return registers | {"extra_onehot": extra_onehot}


def _outer_unprepare_from_outcomes_resources(junk_outcomes, X_o, alt, keep, b_k1, k1, ke1, s_u,
                                             x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o,
                                             less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o,
                                             phase_grad, work, extra_onehot):
    return _outer(X_o, b_k1, k1, ke1, s_u).unprepare_from_outcomes_resources(
        OuterPrepare.read_table(X_o, alt, keep, b_k1))


@qp.register_resources(_outer_unprepare_from_outcomes_resources)
def _outer_unprepare_from_outcomes(junk_outcomes, X_o, alt, keep, b_k1, k1, ke1, s_u, x_o,
                                   succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o,
                                   cmp_work_x_o, alt_x_o, keep_x_o, phase_grad, work,
                                   extra_onehot):
    size = OuterUnprepareFromOutcomes.junk_size(X_o, b_k1, k1)
    _outer(X_o, b_k1, k1, ke1, s_u).unprepare_from_outcomes(
        [junk_outcomes[i] for i in range(size)], OuterPrepare.read_table(X_o, alt, keep, b_k1),
        **_outer_from_outcomes_registers(x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o,
                                         less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o,
                                         phase_grad, work, extra_onehot))


register_rule(OuterUnprepareFromOutcomes, _outer_unprepare_from_outcomes)


# --------------------------------------------------------------------------- InnerPrepare


def _check_inner(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u) -> tuple:
    N = check_int("N", N, low=1)
    R = check_int("R", R, low=1)
    B = check_int("B", B, low=1)
    C = check_int("C", C, low=1)
    b_k2 = check_int("b_k2", b_k2, low=1)
    k2 = check_int("k2", k2, low=0, high=lg(B + 1))
    ke2 = check_int("ke2", ke2, low=0, high=lg(R * C))
    s_u = check_int("s_u", s_u, low=3)
    shape = (R * C, B + 1)
    alt = _check_table("alt", alt, shape, B + 1)
    keep = _check_table("keep", keep, shape, 2**b_k2)
    sgn_table = _check_table("sgn_table", sgn_table, shape, 2)
    d_sgn_table = _check_table("d_sgn_table", d_sgn_table, shape, 2)
    _check_onehot("ke2 + lg(B + 1)", ke2 + lg(B + 1), InnerPrepare.read_shape(N, R, B, C, b_k2, k2))
    return N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u


def _inner(N, R, B, C, b_k2, k2, ke2, s_u) -> _AliasSampling:
    return _AliasSampling(B + 1, s_u, b_k2, InnerPrepare.read_shape(N, R, B, C, b_k2, k2),
                          ke2 + lg(B + 1))


def _inner_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b,
                     keep_b, sgn, d_sgn, r, junk_b, phase_grad, work) -> dict:
    """The skeleton's registers for the inner prepare: the read's address is ``x_o ‖ b``."""
    return {"index": b, "succ": succ_b, "succ_work": succ_work_b,
            "hadamards": _join(sigma_0, m_b), "m": m_b, "less_than": less_than_b,
            "cmp_work": cmp_work_b, "alt": alt_b, "keep": keep_b, "address": _join(x_o, b),
            "target": _join(alt_b, keep_b, sgn, d_sgn, r), "junk": junk_b,
            "phase_grad": phase_grad, "work": work}


_INNER_WIRES = ("x_o", "b", "succ_b", "succ_work_b", "sigma_0", "m_b", "less_than_b",
                "cmp_work_b", "alt_b", "keep_b", "sgn", "d_sgn", "r", "junk_b", "phase_grad",
                "work")


class InnerPrepare(Operator2):
    r"""Prepare the inner index ``b`` with the alias distribution of row ``x_o`` of the inner
    weights, and load the signs and the rank (steps 5 to 8).

    **Action.** ``x_o`` is a read-only address; every other register starts in ``|0>``, with
    ``phase_grad`` holding the phase-gradient state. Let :math:`b_0` be the value the uniform
    preparation draws on ``b`` and :math:`P_f` = ``uniform_failure_probability(B + 1, s_u)``. The
    output is a sum of basis terms, one per :math:`b_0`, value of ``m_b`` and value of
    ``sigma_0``, each with the amplitude of :math:`b_0` in
    :class:`~dfthc.ops.uniform.FlaggedUniformPrep` over :math:`\sqrt{2^{b_{k2} + 1}}` (one common
    success phase :math:`e^{i\varphi}` on :math:`b_0 \le B`): steps 2 to 5 permute basis states
    and add no phase. On an SF row ``x_o = x < R C`` the ``succ_b = 1`` part is
    :math:`e^{i\varphi}\sqrt{1 - P_f}\sum_b \sqrt{\tilde q_x(b)}\,|b\rangle|\mathrm{garb}_{x,b}
    \rangle \otimes |+\rangle_{\sigma_0}` with :math:`\tilde q_x` =
    ``alias_probabilities(alt[x], keep[x], b_k2)``, and ``sgn``, ``d_sgn`` and ``r = x // C``
    hold the drawn entry's columns. Where the selected target entry is zero (one-body rows
    :math:`R C \le x < N + R C`, failed ``x_o``, and failed :math:`b_0 > B`, which have
    ``succ_b = 0``; the junk may hold other entries of a visited block): ``keep = 0``, so
    ``less_than_b = 1``, the swap leaves ``b = 0`` and :math:`b_0` in ``alt_b``, and
    ``sgn = d_sgn = r = 0`` (spec 8.5). :class:`InnerSigns` then applies the signs.

    **Registers.** ``w_x = lg(N + R C)``, ``w_b = lg(B + 1)``, ``w_r = lg(R)``, ``l_b`` the odd
    part of ``B + 1``, ``K2 = 2**k2``, ``d2 = w_b + b_k2 + 2 + w_r``.

    =============== =========================== ============== =================================
    name            width                       input          output
    =============== =========================== ============== =================================
    ``x_o``         ``w_x``                     any            unchanged
    ``b``           ``w_b``                     ``|0>``        the sampled index
    ``succ_b``      1                           ``|0>``        ``[b_0 <= B]``
    ``succ_work_b`` ``max(lg(l_b) - 1, 0)``     ``|0>``        dirty until the unprepare
    ``sigma_0``     1                           ``|0>``        ``|+>``
    ``m_b``         ``b_k2``                    ``|0>``        uniform over ``[0, 2**b_k2)``
    ``less_than_b`` 1                           ``|0>``        ``[keep <= m_b]``
    ``cmp_work_b``  ``b_k2 - 1``                ``|0>``        dirty until the unprepare
    ``alt_b``       ``w_b``                     ``|0>``        the loaded ``alt``, or ``b_0`` if
                                                               swapped
    ``keep_b``      ``b_k2``                    ``|0>``        the loaded ``keep``
    ``sgn``         1                           ``|0>``        the loaded ``sgn``
    ``d_sgn``       1                           ``|0>``        the loaded ``d_sgn``
    ``r``           ``w_r``                     ``|0>``        the loaded rank, ``x_o // C``
    ``junk_b``      ``(K2 - 1) d2``             ``|0>``        the read block's other entries
    ``phase_grad``  at least ``s_u``            gradient state unchanged (catalysed)
    ``work``        :meth:`required_wires`      ``|0>``        ``|0>``
    =============== =========================== ============== =================================

    **Compilable arguments.** ``N``, ``R``, ``B``, ``C`` (ints >= 1; ``x_o`` takes ``N + R C``
    values); ``alt`` (ints in ``[0, B + 1)``), ``keep`` (ints in ``[0, 2**b_k2)``),
    ``sgn_table`` and ``d_sgn_table`` (bits), each of shape ``(R C, B + 1)``: the inner tables
    ``WalkTables.inner_alt``, ``inner_keep``, ``inner_sgn``, ``inner_d_sgn``, named
    ``sgn_table`` and ``d_sgn_table`` because ``sgn`` and ``d_sgn`` name the wire registers;
    ``b_k2`` (int >= 1); ``k2`` (the read's block bits on ``b``, ``0 <= k2 <= w_b``); ``ke2`` (the
    erase's one-hot bits of ``x_o``, ``0 <= ke2 <= lg(R C)``; the one-hot also spans all ``w_b``
    bits of ``b``, and ``2**(ke2 + w_b) <= K2 d2``; used by :class:`InnerUnprepare`, checked here
    so that one argument tuple serves both); ``s_u`` (int >= 3).

    **Pre- and postconditions.** As in the table. Until the unprepare, nothing may write ``x_o``,
    ``succ_b``, ``succ_work_b``, ``less_than_b``, ``cmp_work_b``, ``m_b``, ``alt_b``, ``keep_b``,
    ``sgn``, ``d_sgn`` or ``r``, and ``b`` may be used only as a control or changed by
    operations that are undone before the unprepare. ``junk_b`` takes one of two paths: (a)
    nothing writes it until :class:`InnerUnprepare`; or (b) the caller applies ``H`` and
    ``qp.measure(q, reset=True)`` to each of its qubits right after the prepare, keeps the
    outcomes in wire order (the slots of
    :meth:`QROAMRead.junk_layout <dfthc.ops.qroam.QROAMRead.junk_layout>`), may use its wires for
    other registers in between, and calls :class:`InnerUnprepareFromOutcomes` with the outcomes
    (spec 16, Q-1 variant (b); the block encoding does, once per inner round).

    **Decomposition** (spec 9.8). ``FlaggedUniformPrep(B + 1, s_u)`` on ``b``, ``succ_b``,
    ``succ_work_b``, ``phase_grad[:s_u]`` (unconditionally, as in dfthcbe); ``H`` on ``sigma_0``
    and ``m_b``; ``QROAMRead`` of :meth:`read_table` at ``x_o ‖ b`` (shape :meth:`read_shape`, no
    enable) into ``alt_b ‖ keep_b ‖ sgn ‖ d_sgn ‖ r`` and ``junk_b``, its ``leaf_words`` built
    here from the tables; ``LeftQuantumComparator(keep_b, m_b, less_than_b, cmp_work_b, "<=")``;
    ``w_b`` ``CSWAP``\s of ``alt_b`` and ``b`` controlled by ``less_than_b``.

    **Adjoint and control.** Never adjointed or controlled: the read has loops and data
    conditionals (spec 7.4 rules 4 and 6). The inverse is the operator :class:`InnerUnprepare`
    (path (a) above) or :class:`InnerUnprepareFromOutcomes` (path (b)).

    **Cost.** :func:`dfthc.costs.inner_prepare`: :math:`T = F(B + 1) + T_{read,in} + b_{k2} +
    w_b`, ``M = 0``; ``A`` is the uniform preparation's :math:`2(n - 1) + 2(s_u - 1)` (0 if
    ``l_b = 1``; ``n = lg(l_b)``) plus the read's iteration count ``U``. At FeMoco-54:
    :math:`18 + 933 + 15 + 5 = 971`, where the read carries ``r`` through its swap network (+60
    against dfthcbe's flat ``r``, spec 9.3). The ``CNOT`` and ``PauliX`` counts are upper bounds:
    the read declares a load gate for every bit (``exact=False``).

    **Paper and dfthcbe.** Steps 5 to 8 of dfthcbe's README circuit, its ``InnerPrepare``
    (``dfthcbe/prepare.py``), with PennyLane's ``LeftQuantumComparator`` and with ``r`` in every
    entry; the superposition over ``b`` of arXiv:2502.15882v1 App. B.4 and B.7.2, steps 3 and 4,
    Eqs. (105) and (108) to (110) (Table 8).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import InnerPrepare

        sizes = InnerPrepare.required_wires(2, 2, 2, 1, 2, 1, 1, 3)  # N R B C b_k2 k2 ke2 s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        tables = (((2, 0, 0), (1, 1, 0)), ((3, 1, 2), (0, 2, 3)), ((0, 1, 0), (1, 0, 0)),
                  ((1, 1, 0), (1, 1, 1)))
        program = lambda: InnerPrepare(2, 2, 2, 1, *tables, 2, 1, 1, 3, **wires)
        compiled_resources(program, num_wires=start).toffoli  # 7 + (3 + 7) + 2 + 2 = 21
    """

    compilable_argnames = ("N", "R", "B", "C", "alt", "keep", "sgn_table", "d_sgn_table", "b_k2",
                           "k2", "ke2", "s_u")
    wire_argnames = _INNER_WIRES

    def __init__(self, N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b,
                 succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b, sgn,
                 d_sgn, r, junk_b, phase_grad, work):
        args = _check_inner(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u)
        wires = (x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b,
                 keep_b, sgn, d_sgn, r, junk_b, phase_grad, work)
        check_wires("InnerPrepare", dict(zip(_INNER_WIRES, wires, strict=True)),
                    self.required_wires(*args[:4], *args[8:]), at_least=("phase_grad",))
        super().__init__(*args, *wires)

    @staticmethod
    def required_wires(N, R, B, C, b_k2, k2, ke2, s_u) -> dict[str, int]:
        """Register sizes; ``phase_grad`` is a minimum, every other size is exact. ``work`` is
        the largest need of the uniform preparation, the read and :class:`InnerUnprepare`'s
        erase, so that both classes take the same wires."""
        sizes = _inner(N, R, B, C, b_k2, k2, ke2, s_u).sizes()
        return {"x_o": lg(N + R * C), "b": sizes["index"], "succ_b": 1,
                "succ_work_b": sizes["succ_work"], "sigma_0": 1, "m_b": b_k2,
                "less_than_b": 1, "cmp_work_b": b_k2 - 1, "alt_b": sizes["alt"],
                "keep_b": b_k2, "sgn": 1, "d_sgn": 1, "r": lg(R), "junk_b": sizes["junk"],
                "phase_grad": s_u, "work": sizes["work"]}

    @staticmethod
    def read_shape(N, R, B, C, b_k2, k2) -> dict:
        """The inner read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``:
        rows ``x_o`` in ``[0, R C)`` of ``lg(N + R C)`` bits, then ``b`` in ``[0, B + 1)`` of
        ``lg(B + 1)`` bits, entries of ``d2 = lg(B + 1) + b_k2 + 2 + lg(R)`` bits."""
        return {"address_widths": (lg(N + R * C), lg(B + 1)),
                "live_ranges": ((0, R * C), (0, B + 1)),
                "width": lg(B + 1) + b_k2 + 2 + lg(R), "block_bits": k2}

    @staticmethod
    def read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2) -> tuple:
        """The inner read's table: ``(x << lg(B + 1) | b, words)`` for every SF row ``x < R C``
        and every ``b <= B``, with the entry ``alt ‖ keep ‖ sgn ‖ d_sgn ‖ r`` (``r = x // C``)
        big-endian, as the target ``alt_b ‖ keep_b ‖ sgn ‖ d_sgn ‖ r`` holds it."""
        w_b, w_r = lg(B + 1), lg(R)
        width = w_b + b_k2 + 2 + w_r
        table = []
        for x in range(R * C):
            for b in range(B + 1):
                entry = (alt[x][b] << b_k2) | keep[x][b]
                entry = (entry << 2) | (sgn_table[x][b] << 1) | d_sgn_table[x][b]
                table.append(((x << w_b) | b, _words((entry << w_r) | (x // C), width)))
        return tuple(table)


def _inner_prepare_resources(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u,
                             x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                             alt_b, keep_b, sgn, d_sgn, r, junk_b, phase_grad, work):
    return _inner(N, R, B, C, b_k2, k2, ke2, s_u).prepare_resources()


@qp.register_resources(_inner_prepare_resources)
def _inner_prepare(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b,
                   succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b, sgn,
                   d_sgn, r, junk_b, phase_grad, work):
    _inner(N, R, B, C, b_k2, k2, ke2, s_u).prepare(
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2),
        **_inner_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                           alt_b, keep_b, sgn, d_sgn, r, junk_b, phase_grad, work))


register_rule(InnerPrepare, _inner_prepare)


# --------------------------------------------------------------------------- InnerUnprepare


class InnerUnprepare(Operator2):
    r"""The measured unprepare of :class:`InnerPrepare` (steps 19 to 22): its explicit adjoint
    class (spec 7.4 rule 6).

    **Action.** ``InnerUnprepare`` after ``InnerPrepare`` with the same arguments is the
    identity on the prepare's input (``x_o`` unchanged, every other register ``|0>``,
    ``phase_grad`` unchanged), for every measurement outcome; operations in between must respect
    :class:`InnerPrepare`'s postconditions (:class:`InnerSigns` does: it only applies phases).

    **Registers.** Those of :class:`InnerPrepare`; the input is the prepare's output.

    =============== =========================== ============================== =================
    name            width                       input                          output
    =============== =========================== ============================== =================
    ``x_o``         ``w_x``                     any                            unchanged
    ``b``           ``w_b``                     the sampled index              ``|0>``
    ``succ_b``      1                           ``[b_0 <= B]``                 ``|0>``
    ``succ_work_b`` ``max(lg(l_b) - 1, 0)``     the prepare's dirty wires      ``|0>``
    ``sigma_0``     1                           ``|+>``                        ``|0>``
    ``m_b``         ``b_k2``                    uniform                        ``|0>``
    ``less_than_b`` 1                           ``[keep <= m_b]``              ``|0>``
    ``cmp_work_b``  ``b_k2 - 1``                the prepare's dirty wires      ``|0>``
    ``alt_b``       ``w_b``                     the loaded ``alt``, or ``b_0`` ``|0>``
    ``keep_b``      ``b_k2``                    the loaded ``keep``            ``|0>``
    ``sgn``         1                           the loaded ``sgn``             ``|0>``
    ``d_sgn``       1                           the loaded ``d_sgn``           ``|0>``
    ``r``           ``w_r``                     the loaded rank                ``|0>``
    ``junk_b``      ``(K2 - 1) d2``             the read block's other entries ``|0>``
    ``phase_grad``  at least ``s_u``            gradient state                 unchanged
    ``work``        :meth:`required_wires`      ``|0>``                        ``|0>``
    =============== =========================== ============================== =================

    **Compilable arguments.** Those of :class:`InnerPrepare`, with the same domains; ``ke2`` sets
    the erase's one-hot bits, ``ke2 + lg(B + 1)`` (spec 9.5).

    **Pre- and postconditions.** The registers hold what :class:`InnerPrepare` produced, up to
    phases and operations that use ``b`` only as a control or are undone; ``work`` is ``|0>``.
    Afterwards every register but ``x_o`` and ``phase_grad`` is ``|0>``.

    **Decomposition** (spec 9.8). The ``w_b`` ``CSWAP``\s (self-inverse);
    ``qp.adjoint(LeftQuantumComparator(keep_b, m_b, less_than_b, cmp_work_b, "<="))``, whose rule
    is a measured uncompute (it also cleans ``cmp_work_b``); ``QROAMErase`` of
    :meth:`InnerPrepare.read_table` with ``onehot_bits = ke2 + lg(B + 1)`` (the low ``ke2`` bits
    of ``x_o`` and all of ``b``), no enable and no ``extra_onehot``; ``H`` on ``sigma_0`` and
    ``m_b``; ``qp.adjoint(FlaggedUniformPrep(B + 1, s_u, ...))`` on the prepare's wires, the
    auto-derived leaf adjoint.

    **Adjoint and control.** Never adjointed or controlled: it measures (spec 7.4 rule 5), and it
    is itself the adjoint class of :class:`InnerPrepare`. The walk's block encoding measures the
    junk right after the prepare and calls the sibling :class:`InnerUnprepareFromOutcomes`
    instead (spec 16, Q-1 variant (b)); this class serves callers that keep the junk.

    **Cost.** :func:`dfthc.costs.inner_unprepare`: :math:`T = w_b + T_{erase,in} +
    F^\dagger(B + 1)`, :math:`M = K_2 d_2` (every target and junk qubit); ``A`` is the
    comparator's ``b_k2`` plus the erase's ``T`` plus the adjoint uniform preparation's
    :math:`3(n - 1) + 2(s_u - 1)` (0 if ``l_b = 1``). At FeMoco-54: :math:`5 + 196 + 16 = 217`,
    ``M = 416``. The ``PauliX`` count is an upper bound: each measurement's reset is counted at
    its maximum (spec 7.6).

    **Paper and dfthcbe.** Steps 19 to 22 of dfthcbe's README circuit, the adjoint of its
    ``InnerPrepare`` with the measured erase; arXiv:2502.15882v1 App. B.7.2, steps 8 and 9,
    Eqs. (121) to (123) and (125) (Table 8). Eq. (123)'s ``N`` term, the fix-up of the tail
    rows, is :class:`~dfthc.ops.angle_data.AngleErase`'s here (spec 13.1). Our comparator
    adjoint costs no Toffoli (the paper's and dfthcbe's ``b_k2`` term): the comparator's
    ``b_k2 - 1`` work wires stay dirty until here, the saving the paper notes after Eq. (127).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import InnerUnprepare

        sizes = InnerUnprepare.required_wires(2, 2, 2, 1, 2, 1, 1, 3)  # N R B C b_k2 k2 ke2 s_u
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        tables = (((2, 0, 0), (1, 1, 0)), ((3, 1, 2), (0, 2, 3)), ((0, 1, 0), (1, 0, 0)),
                  ((1, 1, 0), (1, 1, 1)))
        program = lambda: InnerUnprepare(2, 2, 2, 1, *tables, 2, 1, 1, 3, **wires)
        compiled_resources(program, num_wires=start).measurements  # K2 d2 = 2 * 7 = 14
    """

    compilable_argnames = ("N", "R", "B", "C", "alt", "keep", "sgn_table", "d_sgn_table", "b_k2",
                           "k2", "ke2", "s_u")
    wire_argnames = _INNER_WIRES

    def __init__(self, N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b,
                 succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b, sgn,
                 d_sgn, r, junk_b, phase_grad, work):
        args = _check_inner(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u)
        wires = (x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b,
                 keep_b, sgn, d_sgn, r, junk_b, phase_grad, work)
        check_wires("InnerUnprepare", dict(zip(_INNER_WIRES, wires, strict=True)),
                    self.required_wires(*args[:4], *args[8:]), at_least=("phase_grad",))
        super().__init__(*args, *wires)

    @staticmethod
    def required_wires(N, R, B, C, b_k2, k2, ke2, s_u) -> dict[str, int]:
        """:meth:`InnerPrepare.required_wires`."""
        return InnerPrepare.required_wires(N, R, B, C, b_k2, k2, ke2, s_u)


def _inner_unprepare_resources(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2,
                               s_u, x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b,
                               cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, junk_b, phase_grad,
                               work):
    return _inner(N, R, B, C, b_k2, k2, ke2, s_u).unprepare_resources(
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2))


@qp.register_resources(_inner_unprepare_resources)
def _inner_unprepare(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b,
                     succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b,
                     sgn, d_sgn, r, junk_b, phase_grad, work):
    _inner(N, R, B, C, b_k2, k2, ke2, s_u).unprepare(
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2),
        **_inner_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                           alt_b, keep_b, sgn, d_sgn, r, junk_b, phase_grad, work))


register_rule(InnerUnprepare, _inner_unprepare)


# ----------------------------------------------------------------------- InnerUnprepareFromOutcomes

_INNER_FROM_OUTCOMES_WIRES = ("x_o", "b", "succ_b", "succ_work_b", "sigma_0", "m_b",
                              "less_than_b", "cmp_work_b", "alt_b", "keep_b", "sgn", "d_sgn", "r",
                              "phase_grad", "work", "extra_onehot")


class InnerUnprepareFromOutcomes(Operator2):
    r"""The measured unprepare of :class:`InnerPrepare` (steps 19 to 22) for a caller that
    X-measured ``junk_b`` right after the prepare (spec 16, Q-1 variant (b)).

    **Action.** After ``InnerPrepare``, ``H`` and ``qp.measure(q, reset=True)`` on every qubit
    of ``junk_b`` (the outcomes :math:`\xi_{junk}`) and operations that respect
    :class:`InnerPrepare`'s postconditions (:class:`InnerSigns` and the selects do),
    ``InnerUnprepareFromOutcomes`` with the outcomes :math:`\xi_{junk}` and the prepare's
    compilable arguments is the identity on the prepare's input (``x_o`` unchanged, every other
    register ``|0>``, ``phase_grad`` unchanged), for every outcome of the junk and target
    measurements: :class:`InnerUnprepare` with its erase's junk measurements moved to the caller.
    The junk wires are free from their measurement on; the block encoding gives them to the
    angle registers, which the selects load and erase in between (spec 10.2).

    **Registers.** Those of :class:`InnerPrepare` without ``junk_b``, then ``extra_onehot``.

    ================ ================================ ============================== ==========
    name             width                            input                          output
    ================ ================================ ============================== ==========
    ``x_o``          ``w_x``                          any                            unchanged
    ``b``            ``w_b``                          the sampled index              ``|0>``
    ``succ_b``       1                                ``[b_0 <= B]``                 ``|0>``
    ``succ_work_b``  ``max(lg(l_b) - 1, 0)``          the prepare's dirty wires      ``|0>``
    ``sigma_0``      1                                ``|+>``                        ``|0>``
    ``m_b``          ``b_k2``                         uniform                        ``|0>``
    ``less_than_b``  1                                ``[keep <= m_b]``              ``|0>``
    ``cmp_work_b``   ``b_k2 - 1``                     the prepare's dirty wires      ``|0>``
    ``alt_b``        ``w_b``                          the loaded ``alt``, or ``b_0`` ``|0>``
    ``keep_b``       ``b_k2``                         the loaded ``keep``            ``|0>``
    ``sgn``          1                                the loaded ``sgn``             ``|0>``
    ``d_sgn``        1                                the loaded ``d_sgn``           ``|0>``
    ``r``            ``w_r``                          the loaded rank                ``|0>``
    ``phase_grad``   at least ``s_u``                 gradient state                 unchanged
    ``work``         :meth:`required_wires`           ``|0>``                        ``|0>``
    ``extra_onehot`` ``max(2**(ke2 + w_b) - d2, 0)``  ``|0>``                        ``|0>``
    ================ ================================ ============================== ==========

    **Arguments.** Dynamic: ``junk_outcomes``, the :math:`(K_2 - 1) d_2` outcomes of the junk
    measurements in the order of the ``junk_b`` wires: slot ``r`` of
    :meth:`QROAMRead.junk_layout <dfthc.ops.qroam.QROAMRead.junk_layout>` gives outcomes
    ``r d2`` to ``(r + 1) d2 - 1`` (``d2 = w_b + b_k2 + 2 + w_r``). Under capture it becomes a
    bool array; in tape mode every entry must be a ``MeasurementValue``, so both modes emit the
    same fixup (K-10). Compilable: those of :class:`InnerPrepare`, with the same domains; ``ke2``
    sets the fixup's one-hot bits, ``ke2 + lg(B + 1)`` (spec 9.5).

    **Pre- and postconditions.** The registers hold what :class:`InnerPrepare` produced, up to
    phases and operations that use ``b`` only as a control or are undone; ``junk_b`` was
    X-measured with reset right after the prepare, ``junk_outcomes`` are its outcomes in wire
    order, and its wires may serve other registers in between; ``work`` and ``extra_onehot`` are
    ``|0>``. Afterwards every register but ``x_o`` and ``phase_grad`` is ``|0>``.

    **Decomposition** (spec 16, Q-1 variant (b)). The ``w_b`` ``CSWAP``\s;
    ``qp.adjoint(LeftQuantumComparator(keep_b, m_b, less_than_b, cmp_work_b, "<="))`` (a
    measured uncompute); ``H`` and ``qp.measure(q, reset=True)`` on every qubit of
    ``alt_b ‖ keep_b ‖ sgn ‖ d_sgn ‖ r``; the parities :math:`f(a) = \xi \cdot g(a)` of the
    outcomes :math:`\xi` = target ‖ ``junk_outcomes`` with :math:`g(a)` =
    :meth:`QROAMRead.loaded_bits <dfthc.ops.qroam.QROAMRead.loaded_bits>` of
    :meth:`InnerPrepare.read_table` over the read's loaded set (masks built here from the
    compilable tables, never passed as data), by :func:`dfthc.compile.mcm_parities`; the fixup
    over that set (one-hot over the low ``ke2`` bits of ``x_o`` and all of ``b``) with the reset
    target ‖ ``extra_onehot`` as its one-hot register
    (:func:`dfthc.ops.qroam.emit_erase_from_outcomes`); ``H`` on ``sigma_0`` and ``m_b``;
    ``qp.adjoint(FlaggedUniformPrep(B + 1, s_u, ...))``, the auto-derived leaf adjoint.

    **Adjoint and control.** Never adjointed or controlled: it measures (spec 7.4 rule 5). It is
    a sibling of :class:`InnerUnprepare`, the same inverse of :class:`InnerPrepare` with another
    signature, not a modified form of a composite (spec 7.4 rule 6).

    **Cost.** :func:`dfthc.costs.inner_unprepare_from_outcomes`: the ``T`` of
    :func:`dfthc.costs.inner_unprepare`, :math:`w_b + T_{erase,in} + F^\dagger(B + 1)`, and
    ``M = d2`` (the target qubits; the caller's :math:`(K_2 - 1) d_2` junk measurements make up
    :class:`InnerUnprepare`'s :math:`K_2 d_2`); ``A`` as :class:`InnerUnprepare`'s. At
    FeMoco-54: :math:`5 + 196 + 16 = 217`, ``M = 26`` (and 390 junk measurements in the block
    encoding per round), ``extra_onehot`` 102 wires. The ``PauliX`` count is an upper bound
    (spec 7.6).

    **Paper and dfthcbe.** Steps 19 to 22 of dfthcbe's README circuit with the junk measured
    early, as the paper does: its :math:`(2^{k_2} - 1) b_2` QROM output qubits are temporary
    and share the angle qubits (arXiv:2502.15882v1 App. B.7.2, step 4, after Eq. (113), and the
    caption of Table 3), and steps 8 and 9, Eqs. (121) to (123) and (125) (Table 8). The walk
    measures the junk right after step 8, the end of the prepare, before any angle load; the
    comparator and the swaps of steps 7 and 8 do not touch it, so this equals measuring right
    after the read (step 6). dfthcbe's adjoint ``InnerPrepare`` measures target and junk
    together at the erase, as :class:`InnerUnprepare` does.

    **Example.**

    .. code-block:: python

        import numpy as np
        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import InnerUnprepareFromOutcomes

        structure = (2, 2, 2, 1, 2, 1, 1, 3)  # N R B C b_k2 k2 ke2 s_u
        sizes = InnerUnprepareFromOutcomes.required_wires(*structure)
        wires, start = {}, 0
        for name, size in sizes.items():
            wires[name], start = list(range(start, start + size)), start + size
        tables = (((2, 0, 0), (1, 1, 0)), ((3, 1, 2), (0, 2, 3)), ((0, 1, 0), (1, 0, 0)),
                  ((1, 1, 0), (1, 1, 1)))
        junk = np.zeros(InnerUnprepareFromOutcomes.junk_size(2, 2, 2, 1, 2, 1), dtype=bool)
        program = lambda: InnerUnprepareFromOutcomes(junk, 2, 2, 2, 1, *tables, 2, 1, 1, 3,
                                                     **wires)
        compiled_resources(program, num_wires=start).measurements  # d2 = 7
    """

    dynamic_argnames = ("junk_outcomes",)
    compilable_argnames = ("N", "R", "B", "C", "alt", "keep", "sgn_table", "d_sgn_table", "b_k2",
                           "k2", "ke2", "s_u")
    wire_argnames = _INNER_FROM_OUTCOMES_WIRES

    def __init__(self, junk_outcomes, N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2,
                 ke2, s_u, x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                 alt_b, keep_b, sgn, d_sgn, r, phase_grad, work, extra_onehot):
        args = _check_inner(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u)
        junk_outcomes = _junk_outcomes(junk_outcomes,
                                       self.junk_size(*args[:4], args[8], args[9]))
        wires = (x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b,
                 keep_b, sgn, d_sgn, r, phase_grad, work, extra_onehot)
        check_wires("InnerUnprepareFromOutcomes",
                    dict(zip(_INNER_FROM_OUTCOMES_WIRES, wires, strict=True)),
                    self.required_wires(*args[:4], *args[8:]), at_least=("phase_grad",))
        super().__init__(junk_outcomes, *args, *wires)

    @staticmethod
    def required_wires(N, R, B, C, b_k2, k2, ke2, s_u) -> dict[str, int]:
        """:meth:`InnerPrepare.required_wires` without ``junk_b``, then ``extra_onehot``:
        ``max(2**(ke2 + w_b) - d2, 0)`` wires, the one-hot beyond the reset target."""
        sizes = InnerPrepare.required_wires(N, R, B, C, b_k2, k2, ke2, s_u)
        del sizes["junk_b"]
        return sizes | {"extra_onehot": _inner(N, R, B, C, b_k2, k2, ke2, s_u).extra_onehot()}

    @staticmethod
    def junk_size(N, R, B, C, b_k2, k2) -> int:
        """The number of junk outcomes: ``(K2 - 1) d2``, the width of ``junk_b``."""
        return (2**k2 - 1) * (lg(B + 1) + b_k2 + 2 + lg(R))


def _inner_from_outcomes_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b,
                                   cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, phase_grad, work,
                                   extra_onehot) -> dict:
    registers = _inner_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b,
                                 cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, (), phase_grad, work)
    del registers["junk"]
    return registers | {"extra_onehot": extra_onehot}


def _inner_unprepare_from_outcomes_resources(junk_outcomes, N, R, B, C, alt, keep, sgn_table,
                                             d_sgn_table, b_k2, k2, ke2, s_u, x_o, b, succ_b,
                                             succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b,
                                             alt_b, keep_b, sgn, d_sgn, r, phase_grad, work,
                                             extra_onehot):
    return _inner(N, R, B, C, b_k2, k2, ke2, s_u).unprepare_from_outcomes_resources(
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2))


@qp.register_resources(_inner_unprepare_from_outcomes_resources)
def _inner_unprepare_from_outcomes(junk_outcomes, N, R, B, C, alt, keep, sgn_table, d_sgn_table,
                                   b_k2, k2, ke2, s_u, x_o, b, succ_b, succ_work_b, sigma_0, m_b,
                                   less_than_b, cmp_work_b, alt_b, keep_b, sgn, d_sgn, r,
                                   phase_grad, work, extra_onehot):
    size = InnerUnprepareFromOutcomes.junk_size(N, R, B, C, b_k2, k2)
    _inner(N, R, B, C, b_k2, k2, ke2, s_u).unprepare_from_outcomes(
        [junk_outcomes[i] for i in range(size)],
        InnerPrepare.read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2),
        **_inner_from_outcomes_registers(x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b,
                                         cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, phase_grad,
                                         work, extra_onehot))


register_rule(InnerUnprepareFromOutcomes, _inner_unprepare_from_outcomes)


# --------------------------------------------------------------------------- InnerSigns


class InnerSigns(Operator2):
    r"""Give every branch of the inner prepare the sign of the weight it selected (step 9).

    **Action.** :math:`Z(\mathrm{sgn})\,CZ(\mathrm{less\_than}_b, \mathrm{d\_sgn})`, the phase
    :math:`(-1)^{\mathrm{sgn} \oplus (\mathrm{less\_than}_b \wedge \mathrm{d\_sgn})}` on every
    basis state. After :class:`InnerPrepare`, ``sgn = [w_{b_0} < 0]`` and
    ``d_sgn = sgn XOR sgn[alt[b_0]]`` of the drawn :math:`b_0`, so the phase is the sign of the
    weight of the ``b`` that survives the swap (spec 6.1); branches whose selected entry is zero
    (failed ``b``, one-body and failed rows) get +1 (spec 8.5).

    **Registers.**

    =============== ===== ===== =========
    name            width input output
    =============== ===== ===== =========
    ``sgn``         1     any   unchanged
    ``d_sgn``       1     any   unchanged
    ``less_than_b`` 1     any   unchanged
    =============== ===== ===== =========

    **Compilable arguments.** None.

    **Pre- and postconditions.** None: it acts on every input as stated (it only applies
    phases).

    **Decomposition** (spec 9.9). ``Z(sgn)``, then ``CZ(less_than_b, d_sgn)``.

    **Adjoint and control.** Self-adjoint: ``Adjoint(InnerSigns)`` is pinned to the operator
    itself (spec 7.5); no walk program requests it. Never controlled.

    **Cost.** :func:`dfthc.costs.inner_signs`: ``T = 0``, ``M = 0``.

    **Paper and dfthcbe.** Step 9 of dfthcbe's README circuit, its ``InnerSigns``
    (``dfthcbe/prepare.py``); the paper applies the signs inside its iteration over ``b``
    (arXiv:2502.15882v1 App. B.5).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.prepare import InnerSigns

        compiled_resources(lambda: InnerSigns([0], [1], [2]), num_wires=3).counts
        # {"CZ": 1, "PauliZ": 1}
    """

    wire_argnames = ("sgn", "d_sgn", "less_than_b")

    def __init__(self, sgn, d_sgn, less_than_b):
        check_wires("InnerSigns", {"sgn": sgn, "d_sgn": d_sgn, "less_than_b": less_than_b},
                    self.required_wires())
        super().__init__(sgn, d_sgn, less_than_b)

    @staticmethod
    def required_wires() -> dict[str, int]:
        """Exact register sizes: one wire each."""
        return {"sgn": 1, "d_sgn": 1, "less_than_b": 1}


def _inner_signs_resources(sgn, d_sgn, less_than_b):
    return {qp.Z: 1, qp.CZ: 1}


@qp.register_resources(_inner_signs_resources)
def _inner_signs(sgn, d_sgn, less_than_b):
    qp.Z(sgn[0])
    qp.CZ([less_than_b[0], d_sgn[0]])


def _adjoint_inner_signs_resources(base):
    return {resource_key(lambda: InnerSigns(Wire[1], Wire[1], Wire[1])): 1}


@qp.register_resources(_adjoint_inner_signs_resources)
def _adjoint_inner_signs(base):
    InnerSigns(base.sgn, base.d_sgn, base.less_than_b)


register_rule(InnerSigns, _inner_signs)
register_rule("Adjoint(InnerSigns)", _adjoint_inner_signs)
```

- [ ] **Step 5: Add their cost functions**

Replace `src/dfthc/costs.py` with (new: `outer_unprepare_from_outcomes` and `inner_unprepare_from_outcomes`):

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
    n2, c2)``, the inner register iterated under every outer leaf, whose flag controls it:
    ``c2 = [n1 > 0 or c]``, since a zero-width outer register without control has no flag."""
    lo1, hi1, n1 = outer
    inner_controlled = n1 > 0 or controlled
    return (unary(lo1, hi1, n1, controlled)
            + (hi1 - lo1) * unary(*inner, controlled=inner_controlled))


def truncated_unary(L: int, controlled: bool) -> int:
    """Truncated iteration of Qualtran and dfthcbe over ``[0, L)``, for comparison only (spec
    9.24, "Truncated iteration"): ``max(L + c - 2, 0)`` for ``L >= 1``. A single leaf needs no
    AND: the root control, if any, is its flag. ``L < 1`` raises ``ValueError``."""
    if L < 1:
        raise ValueError(f"truncated iteration needs L >= 1 leaves; got {L}")
    return max(L + int(controlled) - 2, 0)


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


def outer_unprepare_from_outcomes(p: WalkParams) -> Cost:
    """``OuterUnprepareFromOutcomes`` (spec 9.24 row; spec 16, Q-1 variant (b)): the ``T`` of
    :func:`outer_unprepare`, ``M = d1`` (the target; the caller measures the ``(K1 - 1) d1``
    junk qubits)."""
    return Cost(outer_unprepare(p).toffoli, p.d1)


def inner_unprepare_from_outcomes(p: WalkParams) -> Cost:
    """``InnerUnprepareFromOutcomes`` (spec 9.24 row; spec 16, Q-1 variant (b)): the ``T`` of
    :func:`inner_unprepare`, ``M = d2`` (the target; the caller measures the ``(K2 - 1) d2``
    junk qubits)."""
    return Cost(inner_unprepare(p).toffoli, p.d2)


def inner_signs() -> Cost:
    """``InnerSigns`` (spec 9.24 row): ``T = 0``, ``M = 0``."""
    return Cost(0)


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

- [ ] **Step 6: Name the classes in the documents**

Replace `docs/architecture.md` with (changed: the L2 row and the emitters of the fix-up):

````markdown
# Architecture

This package builds the DFTHC block encoding of Low et al. (PRX 15, 041016 (2025);
arXiv:2502.15882), its qubitized walk, the controlled walk and an iterative phase estimation,
as PennyLane `Operator2` building blocks. Programs compile with `qp.qjit(capture=True)` and
`qp.decompose` to a Toffoli-level gate set, and `qp.analyze` counts their resources exactly. The
design is fixed by the specification
([`superpowers/specs/2026-11-11-dfthc-pennylane-design.md`](superpowers/specs/2026-11-11-dfthc-pennylane-design.md),
cited as "spec N.M"); this page maps it onto the code.

## Layers

Each layer depends only on the layers below it (spec 5.1).

| Layer | Content | Modules |
| --- | --- | --- |
| L0, classical data | factors, parameters, alias and angle tables, the encoded Hamiltonian, fermion matrices; NumPy only | `dfthc.params`, `dfthc.registers`, `dfthc.data.factors`, `dfthc.data.sos`, `dfthc.data.alias`, `dfthc.data.angles`, `dfthc.data.tables`, `dfthc.data.encoded`, `dfthc.data.fermion` |
| L1, primitives | `PhaseGradientRotation`, `FlaggedUniformPrep`, `QROAMRead`, `PhaseFixup`, `QROAMErase`, `ZeroReflection`; the unary-iteration helper | `dfthc.ops.phase_gradient`, `dfthc.ops.uniform`, `dfthc.ops.qroam`, `dfthc.ops.reflections`, `dfthc.ops.unary`, `dfthc.ops._validate` |
| L2, paper blocks | `OuterPrepare`, `OuterUnprepare`, `OuterUnprepareFromOutcomes`, `InnerPrepare`, `InnerUnprepare`, `InnerUnprepareFromOutcomes`, `InnerSigns`, `TailAngleLoad`, `SFAngleLoad`, `AngleErase`, `SpinMux`, `SpinSwap`, `GivensRotation`, `GivensLadder`, `GivensLadderDagger`, `MajoranaGate`, `InnerSelect`, `InnerSelectDagger`, `RefT2`, `RefAB`, `ControlledRefT2`, `ControlledRefAB` | `dfthc.ops.prepare`, `dfthc.ops.angle_data`, `dfthc.ops.spin`, `dfthc.ops.givens`, `dfthc.ops.majorana`, `dfthc.ops.select`, `dfthc.ops.reflections` |
| L3, assembly | `DFTHCBlockEncoding`, `ControlledDFTHCBlockEncoding`, `DFTHCWalk`, `ControlledDFTHCWalk`; `WalkLayout` | `dfthc.ops.walk`, `dfthc.registers` |
| L4, application | `ControlledWalkPower`, `QPERound`, the driver `iterative_phase_estimation` (a plain quantum function, spec 2.1 R-4), `qpe_program`, the energy helpers | `dfthc.qpe` |
| Tooling | gate set, pins and the compile helper; closed-form costs; the paper's and dfthcbe's costs; test support | `dfthc.compile`, `dfthc.costs`, `dfthc.costs_reference`, `dfthc.testing.resources`, `dfthc.testing.sparse_sim`, `dfthc.testing.tape`, `dfthc.testing.walk_sim`, `dfthc.testing.toy_walk`, `dfthc.testing.fixtures`, `dfthc.testing.canary`, `dfthc.testing.watchdog`, `dfthc.testing.provenance` |

## The walk, call by call

The walk `W = REF_{A u B} BE` is one program. The tree below is the pinned rules' call structure
(spec 3.5 gives the steps); the resource report has one row per distinct operator call, plus
explicit auxiliary and L4 program rows ([`resources.md`](resources.md)).

```text
DFTHCWalk                                steps 1-28
  DFTHCBlockEncoding                     1-27: U^dag REF_T2 U
    OuterPrepare                         1-4: FlaggedUniformPrep, QROAMRead, keep test, CSWAPs
    InnerPrepare (round 1)               5-8: FlaggedUniformPrep (two PhaseGradientRotations), QROAMRead
    InnerSigns                           9
    InnerSelect                          10-18
      TailAngleLoad, SFAngleLoad         10, 11 (each a QROAMRead)
      SpinMux, SpinSwap                  12, 13
      GivensLadder                       14 (GivensRotation per rung)
      MajoranaGate                       15
      GivensLadderDagger                 16 (adjoint GivensRotation per rung)
      SpinSwap, SpinMux                  17
      AngleErase                         18
    InnerUnprepare                       19-22: CSWAPs, comparator adjoint, QROAMErase, adjoint FlaggedUniformPrep
    RefT2                                23 (ZeroReflection)
    InnerPrepare (round 2), InnerSelectDagger (adjoint MajoranaGate), InnerSigns, InnerUnprepare
    OuterUnprepare                       24-27: CSWAPs, comparator adjoint, QROAMErase, adjoint FlaggedUniformPrep
  RefAB                                  28 (ZeroReflection)
```

The controlled walk `ControlledDFTHCWalk` is `ControlledDFTHCBlockEncoding`, which controls only
the T2 reflection (`ControlledRefT2`), followed by `ControlledRefAB`: on ancillas in `|0>` the
rest is `U^dag U = I` (spec 3.3). `ControlledWalkPower` repeats it with one `qp.for_loop`, and each
`QPERound` applies `H`, the walk power `2**j` times and the feed-forward phase corrections
(`PhaseGradientRotation` under `qp.cond`) on wires borrowed from the walk's `work` pool
(spec 11.1). `PhaseFixup` is the operator form of the erase fix-up that `QROAMErase`,
`AngleErase` and the unprepares from outcomes (through
`dfthc.ops.qroam.emit_erase_from_outcomes`) emit through `dfthc.ops.qroam.emit_phase_fixup`; the
walk does not call it as an operator. The toy walk of spec 11.4 (`ToyWalk`, `ControlledToyWalk`,
`ToyControlledWalkPower` in `dfthc.testing.toy_walk`) checks the phase estimation exactly.

## Registers

`dfthc.registers.WalkLayout.from_params(params, qpe_bits=0)` gives every register named wires in
the groups `system`, `outer`, `inner`, `select`, `phase_grad` and `work` (spec 10.1). The assembly
operators take the groups and split them once with `WalkLayout.split_group`; every block they call
takes only the registers its spec 9 table names (spec 9.0, wire passing).

| Group | `H2_STO3G_TEST` | FeMoco-54 |
| --- | ---: | ---: |
| `system` (`psi_up`, `psi_dn`) | 4 | 108 |
| `outer` | 16 | 143 |
| `inner` | 23 | 455 |
| `select` (`angles`, `G0`, `G1`, `b_neq_B`, `sg`) | 8 | 799 |
| `phase_grad` | 4 | 15 |
| `work` (the shared pool) | 8 | 44 |
| **walk** | **63** | **1,564** |

The controlled walk adds one control wire; the QPE adds `qpe_control` and `qpe_grad` (t wires),
its corrections borrow the work pool, and spare wires follow only where 2t - 1 exceeds the pool
(spec 11.3). The comparators' work registers stay dirty from a prepare to its unprepare and belong
to their group, never to `work` (spec 10.2).

## Compilation

`dfthc.compile.compiled_resources(qfunc, num_wires)` compiles a program once (spec 7.2):
`qp.qjit(capture=True)` around `qp.decompose(gate_set=GATE_SET, fixed_decomps=FIXED_DECOMPS)`
around a `null.qubit` QNode, analyzed by `qp.analyze`. It returns T (Toffoli plus
`TemporaryAND`), M (`MidCircuitMeasure`) and A (`Adjoint(TemporaryAND)`), and fails if the counts
are symbolic or an operation outside the gate set remains. Four rules carry the design
(spec 7.4):

- every dfthc operator has exactly one decomposition rule per key, registered and pinned with
  `dfthc.compile.register_rule`;
- every ancilla is an explicit wire argument, and every operator states
  `required_wires(**compilable_args)`;
- the adjoint and controlled forms of the composites are explicit classes (`OuterUnprepare`,
  `InnerUnprepare`, `InnerSelectDagger`, `GivensLadderDagger`, `ControlledRefT2`,
  `ControlledRefAB`, `ControlledDFTHCBlockEncoding`, `ControlledDFTHCWalk`); no program calls
  `qp.adjoint` or `qp.ctrl` on a composite, because Catalyst would capture the generic candidate
  of its whole subtree (the K-1 memory finding), and an inventory test checks it;
- `QROAMRead` loads each unary leaf with one `qp.for_loop` over a dynamic `int64` table array
  (spec 9.3), so the FeMoco-54 walk compiles in a few GB.

## Costs and the resource report

`dfthc.costs` implements the closed forms of spec 9.24; every compiled count in the tests equals
one of them exactly. `dfthc.costs_reference` evaluates the paper's Table III formulas and ports
dfthcbe's per-block formulas, checked against dfthcbe's own call graph.
`scripts/resource_report.py` writes [`resources.md`](resources.md) and
[`resources.json`](resources.json): one row per program, measured where compiled, with the
closed form, dfthcbe's and the paper's values, the deltas and the reason codes that explain them,
each with its contribution (`dfthc.costs_reference.delta_causes`; spec 13.3). Its compiles are
cached with `dfthc.testing.provenance.code_hash()`, which covers `src/dfthc` outside
`dfthc.testing`, the report script and `requirements-nightly.lock`, while ignoring Python
docstrings and comments. After Task 4.5, the FeMoco-54 walk and controlled-walk compiles can be
reused while that hash matches.

## Further reading

- [`conventions.md`](conventions.md): indices, signs, fermions, angles, registers.
- [`verification.md`](verification.md): what each test establishes and how to run the tiers.
- [`limitations.md`](limitations.md): the PennyLane and Catalyst limitations and their canaries.
- [`environment.md`](environment.md): the environment, the load limits and the commands.
- [`agents/HANDOFF.md`](agents/HANDOFF.md): the shared context of the development agents.
- [`agents/LEDGER.md`](agents/LEDGER.md): task status, decisions and heavy-run evidence.
````

Replace `docs/conventions.md` with (changed: the dynamic-arguments bullet):

```markdown
# Conventions

The conventions follow dfthcbe at commit `930255020cec30172e0a7d17c0071d9476a687d1` unless stated
otherwise, so blocks compare one to one (spec 6). Symbols are those of spec 3.1:
`X_o = N + RC`, `w_x = lg X_o`, `w_b = lg(B + 1)`, `w_r = lg R`, with `lg y = ceil(log2 y)`.

## Indices and flags (spec 6.1)

- **The generator index `x_o`.** `x_o < RC` is a sum-of-squares (SF) generator, with
  `x_o = r C + c`; the last N values are the one-body tail, its `N+` D1 rows first, then the Q1
  rows.
- **Families.** `G0 = 0` is SF and `G0 = 1` one-body; `G1 = 0` is D1 and `G1 = 1` Q1.
- **The inner index `b`.** `b = B` is the identity term; `r = floor(x_o / C)`.
- **Success flags.** `succ_x_o` and `succ_b` are 1 where the flagged uniform preparations succeed;
  a failed `x_o` makes `REF_T2` act as `-I`, a failed `b` makes the select act as `+I`
  (spec 8.5).

## Signs and spins (spec 6.1)

- `InnerPrepare` loads `sgn = [w_b < 0]` and `d_sgn = sgn XOR sgn[alt]`; `InnerSigns` applies
  `Z(sgn)` and `CZ(less_than_b, d_sgn)`.
- `sigma_0` is the inner spin (the Majorana flavour on one-body rows), `sigma_1` the outer spin;
  `sg = sigma_{G0}` controls the spin swap.

## Fermions (spec 6.2)

- Jordan-Wigner per spin register: `gamma_{j,0} = Z_0 ... Z_{j-1} X_j` and
  `gamma_{j,1} = Z_0 ... Z_{j-1} Y_j`, qubit 0 the most significant, `|1>` occupied; the system is
  `psi_up` (x) `psi_dn`.
- The Majorana gate acts on `psi_dn[0]`: SF applies `-Z` (dfthcbe's convention, `i gamma_0
  gamma_1`; the paper's Fig. 4 shows `+Z`), D1 `a = (X + iY)/2`, Q1 `a^dag = (X - iY)/2`.

## Givens angles (spec 6.3)

Rung k acts on qubits `(q_k, q_{k+1})`. An angle is stored as
`x_k = round(theta_k / (2 pi) * 2^(b_rot - 1)) mod 2^(b_rot - 1)`, and the rung realizes
`theta = 4 pi x / 2^b_rot` exactly, with the rung matrix `[[cos, -sin], [sin, cos]]` on
`{|01>, |10>}`. Rung N - 2 comes first in time and rung 0 last.

## Alias sampling (spec 6.4)

`less_than = [keep <= m]`; a set flag swaps `alt` into the index register. Keep values have
`b_k1` (outer) and `b_k2` (inner) bits.

## Phase gradient (spec 6.5)

`PhaseGradientStatePrep` prepares the gradient once, outside the walk, which returns it
unchanged. Adding `k` into the gradient register multiplies the state by `e^{+2 pi i k / 2^b}`;
`PhaseGradientRotation(k, m)` applies `e^{2 pi i k t / 2^m}` on `|t>`.

## Reflections (spec 6.6)

Reflections are exact, global sign included: `2|0><0| - I` is a multi-controlled Z with
`GlobalPhase(pi)` when uncontrolled and `Z` on the control when controlled. No generic
`C(GlobalPhase)` is used.

## Wires and registers (spec 6.7)

- Integers in registers are big-endian: the first wire is the most significant bit.
- Core register names follow dfthcbe: `x_o`, `succ_x_o`, `sigma_1`, `m_x_o`, `less_than_x_o`,
  `alt_x_o`, `keep_x_o`, `junk_x_o`, `b`, `succ_b`, `sigma_0`, `m_b`, `less_than_b`, `alt_b`,
  `keep_b`, `sgn`, `d_sgn`, `r`, `junk_b`, `angles`, `G0`, `G1`, `b_neq_B`, `sg`, `psi_up`,
  `psi_dn`, `phase_grad`, `work`. `WalkLayout` also names the persistent comparator registers
  `succ_work_x_o`, `cmp_work_x_o`, `succ_work_b`, `cmp_work_b`; QPE adds `qpe_control` and
  `qpe_grad`.
- The L3 assembly operators and `ControlledWalkPower` take `WalkLayout`'s groups. `QPERound`
  takes `control`, `qpe_grad` and the walk groups once as flat `walk_wires`; L1 and L2 operators
  take their spec 9 registers by name. A composite passes each child only the registers it needs.
- One operator's registers are distinct qubits: two registers that share a wire do not compile.

## Arguments and tables

- Everything that determines the circuit's structure is a compilable argument; dynamic arguments
  carry runtime data only (`QROAMRead`'s per-leaf table words, `PhaseFixup`'s parities,
  `QPERound`'s earlier bits, the junk outcomes of `OuterUnprepareFromOutcomes` and
  `InnerUnprepareFromOutcomes`, one per junk wire in `QROAMRead.junk_layout` order). Measurement
  outcomes are a bool array under capture and an object array of `MeasurementValue`s in tape
  mode, so a resource key that declares them takes its dtype from the mode
  (`dfthc.ops.prepare.abstract_junk_outcomes`).
- Compilable integers fit in signed 64 bits, so wide table entries are tuples of 32-bit words,
  least significant first (L-24); dynamic table words are `int64` holding 32 data bits (L-26).
- Adjoints use the instance form `qp.adjoint(Op(...))` (L-25).

## Numbers in tests

Every comparison states both tolerances: `rtol=0` with an absolute `atol` (1e-12 for dense and
simulator comparisons unless a test names another), or an explicit relative bound with `atol=0`;
`tests/test_tolerance_hygiene.py` enforces it.
```

- [ ] **Step 7: Run the targeted tests**

Run: `scripts/test.sh tests/ops/test_unprepare_from_outcomes.py tests/test_docstrings.py tests/ops/test_outer_prepare.py tests/ops/test_inner_prepare.py tests/ops/test_qroam_erase.py -m "not slow and not release" -q`
Expected: `199 passed` (dry run 33 s, 1,686 MB, before revision 4's two tests).

Run: `scripts/test.sh tests/ops/test_unprepare_from_outcomes.py tests/ops/test_certificates_prepare.py tests/ops/test_certificates_qroam.py -m "slow and not release" -q`
Expected: `25 passed` (dry run 6 s; the first replay).

- [ ] **Step 8: Run the per-commit tier and the linter, and commit**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1265 passed` (1221 + 44), no warnings summary.

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

```bash
git add src/dfthc/ops/qroam.py src/dfthc/ops/prepare.py src/dfthc/costs.py \
    tests/ops/test_unprepare_from_outcomes.py tests/ops/test_certificates_prepare.py \
    tests/test_docstrings.py docs/architecture.md docs/conventions.md
git commit -m "Add OuterUnprepareFromOutcomes and InnerUnprepareFromOutcomes"
```

- [ ] **Step 9: Report**

Report the RED outputs of Step 2 and the counts of Steps 7 and 8.

---

### Task Q1b.2: Early junk measurement in the block encodings

**Task id:** `Q1b.2` · **Branch:** `phase4/Q1b.2` (from `main` after Q1b.1's merge) · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** Q1b.1

**Files:**
- Modify: `src/dfthc/registers.py`, `src/dfthc/ops/walk.py`, `src/dfthc/costs.py`, `src/dfthc/costs_reference.py`, `src/dfthc/qpe.py`, `src/dfthc/testing/walk_sim.py`, `scripts/resource_report.py`, `tests/test_registers.py`, `tests/ops/test_block_encoding.py`, `tests/ops/test_walk.py`, `tests/test_work_pool.py`, `tests/test_qpe.py`, `tests/report/test_report.py`, `docs/architecture.md`, `docs/conventions.md`, `docs/verification.md`
- Regenerate: `docs/resources.md`, `docs/resources.json`

**Interfaces:**
- Consumes: Q1b.1's classes, `abstract_junk_outcomes` and cost functions.
- Produces: `registers.JUNK`, `WalkLayout.junk_sizes(params)`, `WalkLayout.split_junk(params, select)`, the register `junk_overflow`; `walk.Step`, `walk.JUNK_MEASUREMENT`, `walk.FROM_OUTCOMES`; `costs.junk_measurements(p)`; `costs_reference.PAPER_QUBIT_GROUPS_FEMOCO`; the walk at 55 wires (`H2_STO3G_TEST`) and 1,102 (FeMoco-54). Task 4.5 compiles this walk.

**Acceptance tests** (5 more per commit; the changed literals of section "Changed tests"):
- `test_junk_is_measured_right_after_each_prepare[sto3g, 631g]`: in the block encoding's rule every junk qubit is X-measured with reset right after its prepare and before the next operator, and each unprepare from outcomes receives the outcomes of its own round.
- `test_junk_registers_alias_the_start_of_the_angle_wires[...]` (4 presets): each junk register is the first wires of `angles ‖ junk_overflow`.
- `test_qubit_summary_at_femoco`: the report's FeMoco-54 qubit table (1,102 against 1,131 by group) and its measurement sentence.
- E-1 to E-4 (nightly, unchanged files) pass on the new walk.

**Definition of done:** RED seen; the targeted tests and E-1 to E-4 pass; the report of the H2 systems and `naphthalene` regenerated; the documentation tests pass on it; the per-commit tier passes; `ruff check .` clean; one commit; then stop for the orchestrator's `--large` reruns.

- [ ] **Step 1: Write the changed tests**

Replace `tests/test_registers.py` with:

```python
"""WalkLayout: named wires, groups, the junk aliases, the work pool and the split helpers (spec
section 10)."""

import jax.numpy as jnp
import pytest
from pennylane.wires import Wires

from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.registers import GROUPS, JUNK, WalkLayout

PRESETS = [FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE]


def test_femoco_group_sizes():
    """Spec 10.3 with the junk registers on the angle wires (spec 16, Q-1 variant (b)): 1,102
    wires, 462 fewer than the 1,564 of separate junk registers (72 outer, 390 inner)."""
    layout = WalkLayout.from_params(FEMOCO54)
    sizes = {name: len(getattr(layout, name)) for name in
             ("system", "phase_grad", "outer", "inner", "select", "work")}
    assert sizes == {"system": 108, "phase_grad": 15, "outer": 71, "inner": 65,
                     "select": 799, "work": 44}
    assert layout.num_wires == 1102 == 1564 - 72 - 390


def test_femoco_register_sizes_follow_spec_9_7_and_9_8():
    sizes = WalkLayout.register_sizes(FEMOCO54)
    assert (sizes["succ_work_x_o"], sizes["cmp_work_x_o"]) == (6, 14)
    assert (sizes["succ_work_b"], sizes["cmp_work_b"]) == (2, 14)
    assert (sizes["x_o"], sizes["b"], sizes["r"], sizes["angles"]) == (9, 5, 4, 795)
    assert WalkLayout.junk_sizes(FEMOCO54) == {"junk_x_o": 72, "junk_b": 390}
    assert sizes["junk_overflow"] == 0 and not set(JUNK) & set(sizes)


def test_h2_sto3g_layout():
    """55 wires: the 4 and 7 junk wires share the 4 angle wires and 3 overflow wires."""
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    assert layout.num_wires == 55
    assert len(layout.work) == 8 == WalkLayout.work_size(H2_STO3G_TEST)
    assert len(layout.succ_work_x_o) == 0  # X_o = 4 has odd part 1: a zero-width register
    assert (len(layout.angles), len(layout.junk_overflow)) == (4, 3)


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
def test_junk_registers_alias_the_start_of_the_angle_wires(params):
    """Spec 10.2: each junk register is the first wires of angles ‖ junk_overflow, the start of
    the select group, and junk_overflow holds exactly the junk wires beyond the angles (none at
    FeMoco-54, H2_631G_TEST and MID_NAPHTHALENE; 3 at H2_STO3G_TEST); no group register holds a
    junk wire elsewhere."""
    layout = WalkLayout.from_params(params)
    shared = list(layout.angles) + list(layout.junk_overflow)
    junk = WalkLayout.junk_sizes(params)
    assert len(layout.junk_overflow) == max(*junk.values(), len(layout.angles)) - len(
        layout.angles)
    for name, size in junk.items():
        assert list(getattr(layout, name)) == shared[:size] == list(layout.select)[:size]
    others = {wire for group in GROUPS.values() for name in group
              if name not in ("angles", "junk_overflow") for wire in getattr(layout, name)}
    assert not others & set(shared)


@pytest.mark.parametrize("params", PRESETS)
def test_split_group_inverts_the_layout(params):
    layout = WalkLayout.from_params(params)
    for group, names in GROUPS.items():
        wires = getattr(layout, group)
        parts = WalkLayout.split_group(params, group, wires)
        assert list(parts) == list(names)
        assert all(parts[name] == getattr(layout, name) for name in names)
    assert WalkLayout.split_junk(params, layout.select) == {
        name: getattr(layout, name) for name in JUNK}


def test_split_helpers_slice_arrays_as_rules_see_them():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    parts = WalkLayout.split_group(H2_STO3G_TEST, "inner", jnp.asarray(layout.inner.tolist()))
    assert [p.tolist() for p in parts.values()] == [w.tolist() for w in
                                                   (getattr(layout, n) for n in GROUPS["inner"])]
    junk = WalkLayout.split_junk(H2_STO3G_TEST, jnp.asarray(layout.select.tolist()))
    assert [j.tolist() for j in junk.values()] == [layout.junk_x_o.tolist(),
                                                   layout.junk_b.tolist()]


def test_split_helpers_reject_a_wrong_length():
    layout = WalkLayout.from_params(H2_STO3G_TEST)
    with pytest.raises(ValueError, match="outer needs 12 wires; got 11"):
        WalkLayout.split_group(H2_STO3G_TEST, "outer", layout.outer[:-1])
    with pytest.raises(ValueError, match="unknown group"):
        WalkLayout.split_group(H2_STO3G_TEST, "angles", layout.angles)
    with pytest.raises(ValueError, match="select needs 11 wires; got 10"):
        WalkLayout.split_junk(H2_STO3G_TEST, layout.select[:-1])


def test_angle_register_is_a_slice_of_angles():
    layout = WalkLayout.from_params(FEMOCO54)
    for k in (0, 1, 52):
        assert layout.angle_register(k) == layout.angles[15 * k:15 * (k + 1)]
    with pytest.raises(ValueError, match="rung 53"):
        layout.angle_register(53)


def test_qpe_registers_only_when_requested():
    layout = WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=3)
    assert layout.num_wires == 55 + 1 + 3
    assert layout.qpe_control == Wires([55]) and layout.qpe_grad == Wires([56, 57, 58])
    with pytest.raises(AttributeError):
        _ = WalkLayout.from_params(H2_STO3G_TEST).qpe_control
    with pytest.raises(ValueError, match="qpe_bits"):
        WalkLayout.from_params(H2_STO3G_TEST, qpe_bits=-1)
```

Replace `tests/ops/test_block_encoding.py` with:

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
    """The harness the semantic tests rely on (and Tasks 3.4 and 3.5): S0 and S1 force every
    outcome 0 and every outcome 1, the alternating and seeded schedules both outcomes,
    assert_same_state compares complex amplitudes (a -1 on the whole state or on one term fails
    it), project returns an input exactly and drops every term with a nonzero ancilla, and
    nonzero_registers names that ancilla's register."""
    c = case("sto3g")
    assert (walk_sim.schedule("S0", 8), walk_sim.schedule("S1", 8)) == ([0] * 8, [1] * 8)
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


def queued(rule, c, controlled):
    """Everything the rule queues in tape mode: the children and the junk measurements."""
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(*c.compilable, **c.groups(controlled))
    return list(queue.queue)


def children(rule, c, controlled):
    """The rule's operators in tape mode."""
    return [op for op in queued(rule, c, controlled) if op.name not in ("Hadamard", "MidMeasure")]


def registers(op):
    op = getattr(op, "base", op)
    return {name: list(getattr(op, name)) for name in op.wire_argnames}


NAMES = ["OuterPrepare", "InnerPrepare", "InnerSigns", "InnerSelect", "InnerUnprepareFromOutcomes",
         "RefT2", "InnerPrepare", "InnerSelectDagger", "InnerSigns", "InnerUnprepareFromOutcomes",
         "OuterUnprepareFromOutcomes"]
"""The block encoding's children in time order (spec 9.21; spec 16, Q-1 variant (b))."""
JUNK_OF = {"OuterPrepare": "junk_x_o", "InnerPrepare": "junk_b",
           "OuterUnprepareFromOutcomes": "junk_x_o", "InnerUnprepareFromOutcomes": "junk_b"}


@pytest.mark.parametrize("name", ["sto3g", "fails-c3"])
def test_children_order_and_registers(name):
    """Spec 9.21's order, every child on the layout's registers of its own names (spec 9.0):
    the work pool's prefix, the whole phase_grad, the junk registers on the start of the angle
    wires, an unprepare's extra_onehot on the start of its junk register; InnerSigns after
    InnerPrepare in the first round and after InnerSelectDagger in the second; the two rules
    differ only in the reflection, ControlledRefT2 on the control; the two InnerPrepare and
    InnerUnprepareFromOutcomes calls are equal, and the selects have equal arguments with
    fixup_bits = w_b, so each shares one target."""
    c = case(name)
    p, layout = c.params, c.layout
    plain, controlled = children(_block_encoding, c, False), children(
        _controlled_block_encoding, c, True)
    assert [op.name for op in plain] == NAMES
    assert [op.name for op in controlled] == [
        "ControlledRefT2" if n == "RefT2" else n for n in NAMES]
    for op in plain + controlled:
        for register, wires in registers(op).items():
            if register == "work":
                assert wires == list(layout.work)[:len(wires)]
            elif register == "control":
                assert wires == c.control
            elif register == "extra_onehot":
                assert wires == list(getattr(layout, JUNK_OF[op.name]))[:len(wires)]
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


@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_junk_is_measured_right_after_each_prepare(name):
    """Spec 16, Q-1 variant (b): right after each prepare the rule applies H and a reset
    measurement to every wire of its junk register, in wire order, and nothing else; each
    unprepare from outcomes receives exactly the outcomes of the latest measurement of its junk
    register, in that order (the two inner rounds measure separately)."""
    c = case(name)
    ops = queued(_block_encoding, c, False)
    measured, latest, seen = {}, {}, []
    i = 0
    while i < len(ops):
        op = ops[i]
        if op.name in ("OuterPrepare", "InnerPrepare"):
            junk = list(getattr(c.layout, JUNK_OF[op.name]))
            pairs = ops[i + 1:i + 1 + 2 * len(junk)]
            assert [g.name for g in pairs] == ["Hadamard", "MidMeasure"] * len(junk)
            assert [list(g.wires) for g in pairs] == [[w] for w in junk for _ in range(2)]
            assert all(g.reset for g in pairs[1::2])
            latest[JUNK_OF[op.name]] = pairs[1::2]
            i += 1 + 2 * len(junk)
            seen.append(op.name)
            continue
        if op.name in ("OuterUnprepareFromOutcomes", "InnerUnprepareFromOutcomes"):
            outcomes = [mv.measurements[0] for mv in op.junk_outcomes]
            assert outcomes == latest[JUNK_OF[op.name]], op.name
            measured[op.name] = measured.get(op.name, 0) + 1
        assert op.name not in ("Hadamard", "MidMeasure"), i
        i += 1
    assert seen == ["OuterPrepare", "InnerPrepare", "InnerPrepare"]
    assert measured == {"InnerUnprepareFromOutcomes": 2, "OuterUnprepareFromOutcomes": 1}
    junk = costs.junk_measurements(c.params).measurements
    assert sum(op.name == "MidMeasure" for op in ops) == junk == (
        len(c.layout.junk_x_o) + 2 * len(c.layout.junk_b))


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
    9,750), M = 24 + 2 * (26 + 798) + 852 = 2,524, the 852 junk measurements (72 + 2 * 390) in the
    rule itself (spec 16, Q-1 variant (b)), as spec 9.24's 96 + 2 * (416 + 798); A = T - (1,118 -
    1) = 8,632 (8,633); the groups are 71, 65, 799, 108, 15 and 44 wires, 1,102 in all."""
    p = FEMOCO54
    outer = costs.outer_prepare(p) + costs.outer_unprepare_from_outcomes(p)
    inner = (costs.inner_prepare(p) + costs.inner_select(p)
             + costs.inner_unprepare_from_outcomes(p))
    junk = costs.junk_measurements(p)
    assert (outer, inner, junk, costs.ref_t2(p, False)) == (
        costs.Cost(279, 24), costs.Cost(4725, 824), costs.Cost(0, 852), costs.Cost(20))
    for controlled in (False, True):
        assert costs.block_encoding(p, controlled) == (
            outer + 2 * inner + junk + costs.ref_t2(p, controlled))
    assert costs.block_encoding(p, False) == costs.Cost(9749, 2524)
    assert costs.block_encoding(p, True) == costs.Cost(9750, 2524)
    assert costs.toffoli_only(p) == 1118
    assert [costs.block_encoding(p, c).toffoli - (costs.toffoli_only(p) - 1)
            for c in (False, True)] == [8632, 8633]
    sizes = DFTHCBlockEncoding.required_wires(p.as_compilable(), ())
    assert sizes == {"outer": 71, "inner": 65, "select": 799, "system": 108, "phase_grad": 15,
                     "work": 44}
    assert sum(sizes.values()) == WalkLayout.from_params(p).num_wires == 1102


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
    """By hand at H2_STO3G_TEST: the layout's groups (select: 4 angle wires and 3 overflow wires
    for the 7 inner junk wires, and 4 flags; work: RefAB's controlled need, 8)."""
    p = H2_STO3G_TEST
    sizes = {"outer": 12, "inner": 16, "select": 11, "system": 4, "phase_grad": 4, "work": 8}
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

Replace `tests/ops/test_walk.py` with (changed: the group sizes in `test_required_wires` and the message in `test_constructor_validates`):

```python
"""DFTHCWalk and ControlledDFTHCWalk (spec 9.22): the walk against the block encoding and the
encoded model on the tape path, the three control states, the children, rule selection, the
inventory of captured targets and its instrument, the failing qp.ctrl(DFTHCWalk), validation."""

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
def test_captured_targets_records_one_compile_and_restores_the_hook():
    """captured_targets wraps Catalyst's private hook for one compile only: a second call of the
    same program gives the same list and another program of the same width its own (nothing
    accumulates or is served from an earlier call), and afterwards, also after a program that
    raises, the module holds the original function again."""
    import catalyst.from_plxpr.qfunc_interpreter as qfunc_interpreter

    original = qfunc_interpreter._convert_decomp_target_spec
    registers = {"sgn": [0], "d_sgn": [1], "less_than_b": [2]}
    adjointed = list(captured_targets(lambda: qp.adjoint(InnerSigns(**registers)), 3))
    assert qfunc_interpreter._convert_decomp_target_spec is original
    assert captured_targets(lambda: qp.adjoint(InnerSigns(**registers)), 3) == adjointed
    plain = captured_targets(lambda: InnerSigns(**registers), 3)
    assert "Adjoint(InnerSigns)" in adjointed and "InnerSigns" in plain
    assert "Adjoint(InnerSigns)" not in plain

    def failing():
        raise RuntimeError("the program fails to trace")

    with pytest.raises(RuntimeError, match="fails to trace"):
        captured_targets(failing, 3)
    assert qfunc_interpreter._convert_decomp_target_spec is original


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
        assert sizes == {"outer": 12, "inner": 16, "select": 11, "system": 4, "phase_grad": 4,
                         "work": 8}
        assert controlled.required_wires(p.as_compilable(), ()) == {"control": 1} | sizes


@pytest.mark.parametrize(("change", "match"), [
    ({"params": 7}, "params and tables must be"),
    ({"tables": ()}, "WalkTables needs 8 fields; got 0"),
    ({"outer": [0]}, "'outer' has 1 wires; expected 12"),
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


def test_controlled_walk_checks_the_control_width():
    """ControlledDFTHCWalk checks its one-wire control like the groups (spec 9.0; its _validate
    call passes the class's own wire_argnames, so the block encoding's test does not cover it)."""
    c = walk_sim.case("sto3g")
    with pytest.raises(ValueError, match="'control' has 2 wires; expected 1"):
        ControlledDFTHCWalk(*c.compilable,
                            **c.groups(True) | {"control": [*c.control, c.num_wires]})
```

Replace `tests/test_work_pool.py` with:

```python
"""The walk's shared work pool (spec 10.1 and 10.2): every call of the four assembly rules (the
walks' and the block encodings') gets exactly its required_wires, its work is a prefix of the
pool, and neither the pool nor the comparators' dirty work registers appear where they do not
belong; the junk registers and the unprepares' extra_onehot wires are the start of the angle
wires; WalkLayout.work_size is the largest need, the controlled RefAB's."""

import functools
import inspect
import random

import pennylane as qp
import pytest

from dfthc.compile import dfthc_operator_names
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefAB, RefT2
from dfthc.ops.walk import (
    FROM_OUTCOMES,
    JUNK_MEASUREMENT,
    _block_encoding,
    _children,
    _controlled_block_encoding,
    _controlled_walk,
    _walk,
    _walk_children,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams, lg
from dfthc.registers import JUNK, WalkLayout

PRESETS = {
    "sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "mid": lambda: (MID_NAPHTHALENE, load_factors("naphthalene")),
    "femoco": lambda: (FEMOCO54, random_factors(54, 10, 27, 27, n_plus=27, seed=0)),
}
"""FeMoco-54 is classical here: the rules run in tape mode and construct their children (the
constructors check every register); nothing is compiled."""

DIRTY = {"succ_work_x_o": {"OuterPrepare", "OuterUnprepareFromOutcomes"},
         "cmp_work_x_o": {"OuterPrepare", "OuterUnprepareFromOutcomes"},
         "succ_work_b": {"InnerPrepare", "InnerUnprepareFromOutcomes"},
         "cmp_work_b": {"InnerPrepare", "InnerUnprepareFromOutcomes"}}
"""The comparators' work registers, dirty from a prepare to its unprepare (spec 10.2), and the
only classes that may take them."""

AT_LEAST = {"OuterPrepare", "OuterUnprepareFromOutcomes", "InnerPrepare",
            "InnerUnprepareFromOutcomes"}
"""Their ``phase_grad`` is a minimum (``s_u``); the walk passes the whole register."""

LENDS = {cls.__name__: junk for cls, junk in FROM_OUTCOMES.items()}
"""The unprepares from outcomes and the junk register whose start is their ``extra_onehot``."""

RULES = {"block encoding": (_block_encoding, False, 11),
         "controlled block encoding": (_controlled_block_encoding, True, 11),
         "walk": (_walk, False, 2), "controlled walk": (_controlled_walk, True, 2)}
"""Each assembly rule, whether it is controlled, and its number of operator calls."""


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
    """The rule's operator calls in tape mode, on the layout's groups (``qpe_control`` as
    control); the junk measurements' gates are left out."""
    groups = {name: getattr(layout, name)
              for name in ("outer", "inner", "select", "system", "phase_grad", "work")}
    if controlled:
        groups = {"control": layout.qpe_control} | groups
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(p.as_compilable(), t.as_compilable(), **groups)
    return [op for op in queue.queue if op.name in dfthc_operator_names()]


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
        if name in JUNK:
            assert wires == list(getattr(layout, name)), (op.name, name)
        if name == "extra_onehot":
            assert wires == list(getattr(layout, LENDS[op.name]))[:len(wires)], op.name
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
    into a work prefix), the junk registers and the unprepares' extra_onehot are the start of
    the angle wires, and every need fits the pool."""
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


def own_sizes(cls, args, p) -> dict:
    """A child's own ``required_wires`` from the compilable arguments the rule gives it, or, for
    the reflections (sizes from their wires), from the pattern widths."""
    if cls in (RefT2, ControlledRefT2):
        return cls.required_wires(p.w_b, p.b_k2)
    if cls in (RefAB, ControlledRefAB):
        return cls.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)
    arguments = dict(zip(cls.compilable_argnames, args, strict=True))
    names = inspect.signature(cls.required_wires).parameters
    return cls.required_wires(**{k: v for k, v in arguments.items() if k in names})


def zero_tables(p) -> WalkTables:
    """Tables of ``p``'s shapes with every entry 0: the work needs depend on the shapes only."""
    def rows(n, m):
        return tuple((0,) * m for _ in range(n))

    inner = rows(p.RC, p.B + 1)
    return WalkTables(outer_alt=(0,) * p.X_o, outer_keep=(0,) * p.X_o, inner_alt=inner,
                      inner_keep=inner, inner_sgn=inner, inner_d_sgn=inner,
                      tail_angles=rows(p.N, p.N - 1),
                      sf_angles=tuple(rows(p.B, p.N - 1) for _ in range(p.R)))


CHILDREN = {"OuterPrepare", "InnerPrepare", "InnerSigns", "InnerSelect",
            "InnerUnprepareFromOutcomes", "RefT2", "ControlledRefT2", "InnerSelectDagger",
            "OuterUnprepareFromOutcomes", "DFTHCBlockEncoding", "ControlledDFTHCBlockEncoding",
            "RefAB", "ControlledRefAB"}
"""Every class the four assembly rules call."""


def test_work_size_covers_every_child_at_random_shapes():
    """At 300 random valid shapes, every child of the four assembly rules, taken from the rules'
    own step lists (``_children`` and ``_walk_children``, plain and controlled), is given exactly
    its class's own ``required_wires`` work (and an unprepare from outcomes its own
    ``extra_onehot``, no more than its junk register holds), so no explicit inverse or controlled
    class (``OuterUnprepareFromOutcomes``, ``InnerUnprepareFromOutcomes``, ``InnerSelectDagger``,
    ``ControlledRefT2``, ``ControlledRefAB``) can drift from its forward form unnoticed;
    ``WalkLayout.work_size`` (spec 10.1, provisional until this test) covers every need; and each
    junk register fits in the angle wires and the overflow wires."""
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
        sizes = WalkLayout.register_sizes(p)
        junk = WalkLayout.junk_sizes(p)
        assert max(junk.values()) <= sizes["angles"] + sizes["junk_overflow"], p
        for controlled in (False, True):
            for step in _children(p, t, controlled) + _walk_children(p, t, controlled):
                if step.op == JUNK_MEASUREMENT:
                    assert step.args in JUNK
                    continue
                own = own_sizes(step.op, step.args, p)
                assert own.get("work", 0) == step.work, (p, step.op.__name__)
                assert own.get("extra_onehot", 0) == step.extra_onehot, (p, step.op.__name__)
                if step.op in FROM_OUTCOMES:
                    assert step.extra_onehot <= junk[FROM_OUTCOMES[step.op]], p
                assert step.work <= WalkLayout.work_size(p), (p, step.op.__name__, step.work)
                seen.add(step.op.__name__)
        checked += 1
    assert seen == CHILDREN
```

Replace `tests/test_qpe.py` with (changed: the wire counts and positions):

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
    with pytest.raises(ValueError, match="'outer' has 1 wires; expected 12"):
        ControlledWalkPower(1, *c.compilable, **(c.groups(controlled=True) | {"outer": [0]}))
    assert ControlledWalkPower.required_wires(5, *c.compilable) == {
        "control": 1, "outer": 12, "inner": 16, "select": 11, "system": 4, "phase_grad": 4,
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
        "walk_wires"] for t in (4, 5)] == [55, 56]
    femoco = WalkLayout.from_params(FEMOCO54)
    sizes = tuple(len(getattr(femoco, name)) for name in ControlledWalkPower.wire_argnames[1:])
    assert sizes[-1] == WalkLayout.work_size(FEMOCO54) == 44 and sum(sizes) == 1102
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
    assert spec.wire_sizes == (12, 16, 11, 4, 4, 8)
    wires, start = list(spec.wires), 0
    for name, size in zip(ControlledWalkPower.wire_argnames[1:], spec.wire_sizes, strict=True):
        assert wires[start:start + size] == list(getattr(layout, name)), name
        start += size
    assert wires[:4] != list(layout.system) and start == len(wires) == 55


@pytest.mark.parametrize(("t", "spare"), [(3, []), (5, [61])])
def test_qpe_program_wires(t, spare):
    """qpe_program at H2_STO3G_TEST: the layout with qpe_bits = t (control 55, qpe_grad from
    56), then the spare wires: none at t = 3, whose 5 correction wires fit the 8-wire work pool
    (47 to 54), one (61) at t = 5, whose 9 do not. Every round binds each wire once, the same
    registers in every round: control, qpe_grad and walk_wires (the walk's 55 wires, then the
    spare ones). Round 0's corrections borrow the pool from its first wire on, into the spare
    wire: for d = l + 1, the scratch wires corrections[:d] and the work wires
    corrections[d:2d - 1]."""
    c = walk_sim.case("sto3g")
    program, num_wires = qpe_program(c.params, c.tables, t)
    layout = WalkLayout.from_params(c.params, qpe_bits=t)
    assert num_wires == layout.num_wires + len(spare) == 55 + 1 + t + len(spare)
    spec = walk_power_spec(c.params, c.tables, layout)
    walk_wires = [*spec.wires, *spare]
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        program()
    rounds = [op for op in queue.queue if isinstance(op, QPERound)]
    assert [op.hyperparameters["j"] for op in rounds] == list(reversed(range(t)))
    for op in rounds:
        assert list(op.control) == [55] and list(op.qpe_grad) == list(range(56, 56 + t))
        assert list(op.walk_wires) == walk_wires and sorted(walk_wires[:55]) == list(range(55))
    corrections = [*layout.work, *spare][:2 * t - 1]
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as inner:
        _qpe_round(0, t, [False, *measured(t - 1)], spec.op_name, spec.compilable,
                   spec.wire_sizes, [55], list(range(56, 56 + t)), walk_wires)
    rotations = [op.base for op in inner.queue if isinstance(op, qp.ops.Conditional)]
    assert len(rotations) == t - 1
    for d, rotation in enumerate(rotations, start=2):
        assert list(rotation.target) == [55] and list(rotation.phase_grad) == list(
            range(56, 56 + d))
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

Replace `tests/report/test_report.py` with:

```python
"""The resource report (spec 13.3; master plan Task 4.3): ``scripts/resource_report.py``,
``dfthc.costs_reference`` and ``dfthc.testing.provenance``.

Per commit: every row and column at ``H2_STO3G_TEST`` (one compile per row, in this process), the
two failing checks, the watched and projection-gated ``--large`` path on a stand-in with its stop
policy, a projection for every large row (also after ``fe2s2``'s entries join), the code-hash
cache, the reason codes and the QPE summary at FeMoco-54, dfthcbe's call graph against the port,
Table III, and the projection method on monotone and non-monotone points. Nightly: the whole
report at ``naphthalene``, every program in its own watched subprocess.
"""

import dataclasses
import functools
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from dfthc import costs, costs_reference
from dfthc.params import FEMOCO54, H2_STO3G_TEST, WalkParams
from dfthc.testing import provenance
from dfthc.testing.fixtures import load_dfthcbe_fixture

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "resource_report.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("resource_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["resource_report"] = module
    spec.loader.exec_module(module)
    return module


report = _load_script()

H2_ROWS = [
    "DFTHCWalk", "DFTHCBlockEncoding", "OuterPrepare", "FlaggedUniformPrep@OuterPrepare",
    "QROAMRead@OuterPrepare", "InnerPrepare", "FlaggedUniformPrep@InnerPrepare",
    "PhaseGradientRotation", "QROAMRead@InnerPrepare", "InnerSigns", "InnerSelect",
    "TailAngleLoad", "QROAMRead@TailAngleLoad", "SFAngleLoad", "QROAMRead@SFAngleLoad", "SpinMux",
    "SpinSwap", "GivensLadder", "GivensRotation", "MajoranaGate", "GivensLadderDagger",
    "Adjoint(GivensRotation)", "AngleErase", "InnerUnprepareFromOutcomes",
    "Adjoint(FlaggedUniformPrep)@InnerUnprepareFromOutcomes", "RefT2", "ZeroReflection@RefT2",
    "InnerSelectDagger", "Adjoint(MajoranaGate)", "OuterUnprepareFromOutcomes",
    "Adjoint(FlaggedUniformPrep)@OuterUnprepareFromOutcomes", "RefAB", "ZeroReflection@RefAB",
    "ControlledDFTHCWalk", "ControlledDFTHCBlockEncoding", "ControlledRefT2", "ControlledRefAB",
    "PhaseFixup@InnerUnprepareFromOutcomes", "C(ZeroReflection)@RefT2",
    "PhaseFixup@OuterUnprepareFromOutcomes", "C(ZeroReflection)@RefAB",
    "ControlledWalkPower(p=1)", "ControlledWalkPower(p=2)", "QPE(t=1)", "QPE(t=2)", "QPE(t=3)",
]
"""Every program of the report at H2_STO3G_TEST, in report order: the walk calls no QROAMErase
(the unprepares from outcomes erase themselves, spec 16, Q-1 variant (b)), so the fix-up rows
take their arguments."""
PARTITION = {"OuterPrepare": (1, 1), "InnerPrepare": (2, 2), "InnerSigns": (2, 2),
             "TailAngleLoad": (2, 2), "SFAngleLoad": (2, 2), "SpinMux": (4, 4),
             "SpinSwap": (4, 4), "GivensLadder": (2, 2), "MajoranaGate": (1, 1),
             "GivensLadderDagger": (2, 2), "AngleErase": (2, 2),
             "InnerUnprepareFromOutcomes": (2, 2), "RefT2": (1, 0),
             "Adjoint(MajoranaGate)": (1, 1), "OuterUnprepareFromOutcomes": (1, 1),
             "RefAB": (1, 0), "ControlledRefT2": (0, 1), "ControlledRefAB": (0, 1)}
"""The partition rows and their calls per walk and per controlled walk (spec 3.5)."""


@functools.cache
def h2():
    s = report.build_system("H2_STO3G_TEST")
    return s, {row.id: row for row in report.build_rows(s)}


def _counts(record: dict, which: str) -> tuple:
    values = record[which]
    return values and (values["toffoli"], values["measurements"],
                       values["measured_and_uncomputes"])


def test_report_h2_rows_and_columns(tmp_path):
    """Spec 13.3 at H2_STO3G_TEST without compiling: the rows, their calls, levels, steps,
    closed forms, the dfthcbe and paper values with their deltas and reason codes, the Clifford
    labels and the qubits; the partition sums to the walks; main() writes both files with every
    row and column."""
    s, rows = h2()
    assert list(rows) == H2_ROWS
    assert {r.id: (r.calls, r.controlled_calls) for r in rows.values() if r.partition} == PARTITION
    walk, cwalk = rows["DFTHCWalk"], rows["ControlledDFTHCWalk"]
    assert (walk.closed, walk.uncomputes) == (costs.walk(H2_STO3G_TEST, False), 108)
    assert (cwalk.closed, cwalk.uncomputes) == (costs.walk(H2_STO3G_TEST, True), 110)
    assert (walk.wires, cwalk.wires, rows["QPE(t=3)"].wires) == (55, 56, 59)
    assert rows["QROAMRead@InnerPrepare"].level == "L1" and rows["SpinSwap"].level == "L2"
    assert rows["RefAB"].level == "L3" and rows["QPE(t=1)"].level == "L4"
    assert rows["AngleErase"].steps == "18" and rows["InnerUnprepareFromOutcomes"].steps == "19-22"
    assert rows["PhaseFixup@InnerUnprepareFromOutcomes"].steps == "21"
    assert report.clifford_kind(rows["SpinSwap"]) == "exact"
    assert report.clifford_kind(rows["InnerUnprepareFromOutcomes"]) == (
        "upper bound (CNOT, PauliX); expected, hinted (CZ, PauliZ)")
    assert report.clifford_kind(rows["PhaseFixup@OuterUnprepareFromOutcomes"]) == (
        "expected, hinted (CZ, PauliZ)")

    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "resources.json").read_text())
    records = {r["id"]: r for r in record["systems"]["H2_STO3G_TEST"]["rows"]}
    assert list(records) == H2_ROWS
    assert {r["status"] for r in records.values()} == {"pending"}
    keys = {"id", "level", "steps", "calls", "controlled_calls", "partition", "status", "measured",
            "closed", "dfthcbe", "paper", "paper_step18", "delta_dfthcbe", "delta_paper",
            "codes_dfthcbe", "codes_paper", "cliffords", "clifford_kind", "wires", "seconds",
            "peak_mb", "gates", "projection"}
    assert all(set(r) == keys for r in records.values())
    w = records["DFTHCWalk"]
    assert (_counts(w, "closed"), w["dfthcbe"], w["paper"]) == ((158, 50, 108), 167, 205)
    assert (w["delta_dfthcbe"], w["delta_paper"]) == (-9, -47)
    assert w["codes_dfthcbe"]["ref-target"] == -1 and w["codes_paper"]["qpe-ctrl"] == -1
    assert records["SpinMux"]["codes_dfthcbe"] == {"mux": -1}
    assert records["ControlledRefT2"]["codes_paper"] == {"ref-unif": -1, "ctrl-ref-t2": 1}
    assert records["AngleErase"]["paper_step18"] == records["AngleErase"]["paper"] + 2
    for r in records.values():
        for side in ("dfthcbe", "paper"):
            delta, codes = r[f"delta_{side}"], r[f"codes_{side}"]
            assert codes == {} if delta is None else sum(codes.values()) == delta, (r["id"], side)
            assert 0 not in codes.values() and set(codes) <= set(report.REASON_CODES)
    text = (tmp_path / "resources.md").read_text()
    assert all(f"`{row_id}`" in text for row_id in H2_ROWS)
    assert "| Row | Level | Steps | Calls (W, C(W)) | T / M / A measured |" in text


@pytest.mark.compiled
@pytest.mark.parametrize("row_id", H2_ROWS)
def test_report_h2(row_id):
    """Every program at H2_STO3G_TEST compiles to exactly its closed form (T, M and A), on
    exactly the qubits the report gives it; compiled_resources checks the gate-set closure."""
    s, rows = h2()
    row = rows[row_id]
    counts = report.compile_counts(s, row)
    assert (counts["toffoli"], counts["measurements"], counts["measured_and_uncomputes"]) == (
        row.closed.toffoli, row.closed.measurements, row.uncomputes)
    assert counts["num_wires"] == row.wires


@pytest.mark.slow
@pytest.mark.compiled
@pytest.mark.subprocess
def test_report_mid(tmp_path):
    """The report at naphthalene (MID_NAPHTHALENE's parameters): every program compiles in its
    own watched subprocess to its closed form, and the partition gives the measured walks."""
    assert report.main(["--systems", "naphthalene", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "resources.json").read_text())["systems"]["naphthalene"]
    assert record["params"] == {"N": 10, "R": 8, "B": 6, "C": 5, "n_plus": 5, "b_rot": 15,
                                "b_k1": 15, "b_k2": 15, "s_u": 7, "k1": 1, "k2": 2, "ke1": 3,
                                "ke2": 1}
    rows = {r["id"]: r for r in record["rows"]}
    assert {r["status"] for r in rows.values()} == {"measured"}
    assert _counts(rows["DFTHCWalk"], "measured") == (1913, 502, 1678)
    assert _counts(rows["ControlledDFTHCWalk"], "measured") == (1915, 502, 1680)
    assert all(_counts(r, "measured") == _counts(r, "closed") for r in rows.values())


@pytest.mark.compiled
@pytest.mark.subprocess
def test_report_fails_on_mismatch(tmp_path, monkeypatch, capsys):
    """Spec 13.3 step 4: a closed form that differs from the compiled count, and a nonzero delta
    without a reason code, each make the script exit 1 (after writing both files)."""
    monkeypatch.setattr(costs, "spin_swap", lambda p: costs.Cost(p.N + 1))
    out = tmp_path / "mismatch"
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "SpinSwap", "--out",
                        str(out)]) == 1
    assert "H2_STO3G_TEST/SpinSwap: measured" in capsys.readouterr().err
    assert (out / "resources.md").exists()
    monkeypatch.undo()
    causes = costs_reference.delta_causes

    def without_mux(p):
        return causes(p) | {"spin_mux": ({}, {})}

    monkeypatch.setattr(costs_reference, "delta_causes", without_mux)
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "--out",
                        str(tmp_path / "codes")]) == 1
    assert "SpinMux: delta -1 against dfthcbe != its reason codes {} (sum 0)" in (
        capsys.readouterr().err)


@pytest.mark.subprocess
def test_large_compile_is_watched(tmp_path, monkeypatch):
    """--large on a stand-in: a program that allocates past a 200 MB cap is killed by the
    watchdog and recorded as killed, not as a measurement; a program projected above the limits
    is refused, with its projection, without compiling."""
    standin = [sys.executable, "-c", "import time\nx = bytearray(400 * 2**20)\n"
               "x[::4096] = b'1' * len(x[::4096])\ntime.sleep(20)"]
    monkeypatch.setattr(report, "child_command", lambda s, row: standin)
    monkeypatch.setattr(report, "LARGE_CAP_MB", 200)
    monkeypatch.setattr(report, "calibration", lambda *args: [(1, 0.1, 10.0), (2, 0.2, 10.0)])
    argv = ["--systems", "H2_STO3G_TEST", "--large", "H2_STO3G_TEST", "--rows", "SpinSwap",
            "--out", str(tmp_path)]
    assert report.main(argv) == 0
    record = json.loads((tmp_path / "resources.json").read_text())
    entry = record["compiled"]["H2_STO3G_TEST/SpinSwap"][-1]
    assert (entry["status"], entry["reason"]) == ("killed", "memory")
    assert entry["watchdog"]["cap_mb"] == 200 and entry["watchdog"]["survivors"] == []
    assert entry["watchdog"]["timeout_s"] == report.LARGE_TIMEOUT_S
    row = next(r for r in record["systems"]["H2_STO3G_TEST"]["rows"] if r["id"] == "SpinSwap")
    assert (row["status"], row["measured"]) == ("killed", None)

    monkeypatch.setattr(report, "LARGE_CAP_MB", report.MEMORY_CAP_MB)
    monkeypatch.setattr(report, "calibration", lambda *args: [(1, 1.0, 100.0), (2, 2.0, 5000.0)])

    def no_compile(*args):
        raise AssertionError("a refused program must not be compiled")

    monkeypatch.setattr(report, "measure", no_compile)
    assert report.main(argv) == 0
    record = json.loads((tmp_path / "resources.json").read_text())
    entries = record["compiled"]["H2_STO3G_TEST/SpinSwap"]
    assert [e["status"] for e in entries] == ["killed", "refused"]
    projection = entries[-1]["projection"]
    assert not projection["allowed"] and "GB > 7.5 GB" in projection["reason"]
    row = next(r for r in record["systems"]["H2_STO3G_TEST"]["rows"] if r["id"] == "SpinSwap")
    assert (row["status"], row["measured"]) == ("refused", None)


def _standin(name: str):
    """H2_STO3G_TEST's system and rows under another system's name (the stop policy keys on the
    name; nothing is compiled)."""
    s = dataclasses.replace(report.build_system("H2_STO3G_TEST"), name=name)
    return s, report.build_rows(s)


def test_large_stops_after_a_refusal_at_femoco(monkeypatch):
    """Under --large at FeMoco-54, a refused first program stops the run: the controlled walk is
    never measured and is recorded as skipped, and the log names the escalation."""
    s, rows = _standin("femoco54")
    monkeypatch.setattr(report, "projection", lambda *args: {"allowed": False,
                                                              "reason": "8.0 GB > 7.5 GB"})
    monkeypatch.setattr(report, "measure", lambda *args: pytest.fail("nothing is compiled"))
    cache = {}
    log = report.run_system(s, rows, cache, "1" * 64, projected=True, only=None)
    assert [cache[f"femoco54/{p}"][-1]["status"] for p in report.FEMOCO_PROGRAMS] == [
        "refused", "skipped"]
    assert set(cache) == {"femoco54/DFTHCWalk", "femoco54/ControlledDFTHCWalk"}
    assert any("stopped; escalate to the user" in line for line in log)


@pytest.mark.parametrize("status", ["killed", "failed"])
def test_large_stops_after_a_failed_or_killed_compile(monkeypatch, status):
    """Under --large, the first killed or failed compile stops the system's run: the next
    program is never measured and is recorded as skipped, and the log names the escalation."""
    s, rows = _standin("fe2s2")
    monkeypatch.setattr(report, "projection", lambda *args: {"allowed": True,
                                                              "reason": "within the limits"})
    measured = []

    def stops_once(system, row, *args):
        assert not measured, f"the run continued after a {status} compile"
        measured.append(row.id)
        return {"code_hash": "1" * 64, "environment": {}, "status": status,
                "reason": "memory" if status == "killed" else "compile",
                "watchdog": {"seconds": 1.0, "peak_mb": 7700.0}}

    monkeypatch.setattr(report, "measure", stops_once)
    cache = {}
    log = report.run_system(s, rows, cache, "1" * 64, projected=True,
                            only={"SpinMux", "SpinSwap"})
    assert measured == ["SpinMux"]
    assert [cache[f"fe2s2/{r}"][-1]["status"] for r in ("SpinMux", "SpinSwap")] == [
        status, "skipped"]
    assert any(f"stopped; the orchestrator reports the {status} compile" in line for line in log)


@functools.cache
def _compiled_rows(name: str) -> tuple[list, dict[str, int]]:
    """The rows the report compiles at a system and their gates (tape mode, no compile)."""
    s = report.build_system(name)
    rows = [row for row in report.build_rows(s) if report.compiled_here(s, row)]
    return rows, report.gates(rows)


def _synthetic_entries(cache: dict, name: str, hash_: str) -> None:
    """Current entries for every sized row of the system, with made-up seconds and MB."""
    rows, sizes = _compiled_rows(name)
    for row in rows:
        if row.id in sizes:
            size = sizes[row.id]
            cache[f"{name}/{row.id}"] = [{
                "code_hash": hash_, "status": "compiled", "gates": size,
                "family": report.family(row),
                "watchdog": {"seconds": 1.0 + size / 20_000, "peak_mb": 200.0 + 3 * size / 1000}}]


def test_every_large_row_projects():
    """Every program the report compiles at fe2s2 and fe4s4 has a size measure (gates from its
    operator's resource functions) and an allowed projection from a calibration built like the
    report's: the real rows and gates of the H2 systems and naphthalene, with made-up seconds and
    MB (a synthetic cache, so that a code-hash change does not break this test). Rows no larger
    than a measured member of their family take the family's largest measurement; the others are
    extrapolated (nothing is compiled)."""
    hash_ = "1" * 64
    cache = {}
    for name in report.SMALL:
        _synthetic_entries(cache, name, hash_)
    methods = set()
    for name in ("fe2s2", "fe4s4"):
        rows, sizes = _compiled_rows(name)
        assert len(rows) == len(sizes) == 44, name
        for row in rows:
            estimate = report.projection(cache, row, hash_, sizes[row.id])
            assert estimate["allowed"], (name, row.id, estimate["reason"])
            methods.add(estimate["method"])
    assert methods == {"bounded", "k1"}


def test_large_sequence_projects_fe4s4_after_fe2s2():
    """The orchestrator's order: after `--large fe2s2`, its entries calibrate `--large fe4s4`.
    Every fe4s4 program projects from its family's entries at the small systems and at fe2s2: an
    extrapolated one through the fe2s2 point, a bounded one as the family's largest time and peak
    over all of them, so an expensive fe2s2 entry of a constant-size block (here a made-up
    3,500 s, 7,000 MB SpinMux) sets the fe4s4 projection although other entries are cheaper."""
    hash_ = "1" * 64
    cache = {}
    for name in (*report.SMALL, "fe2s2"):
        _synthetic_entries(cache, name, hash_)
    cache["fe2s2/SpinMux"][0]["watchdog"] = {"seconds": 3500.0, "peak_mb": 7000.0}
    _, fe2s2_sizes = _compiled_rows("fe2s2")
    rows, sizes = _compiled_rows("fe4s4")
    for row in rows:
        estimate = report.projection(cache, row, hash_, sizes[row.id])
        points = report.calibration(cache, report.family(row), hash_)
        assert estimate["allowed"], (row.id, estimate["reason"])
        assert fe2s2_sizes[row.id] in {g for g, _, _ in points}, row.id
        if estimate["method"] == "bounded":
            assert (estimate["seconds"], estimate["peak_mb"]) == (
                max(s for _, s, _ in points), max(m for _, _, m in points)), row.id
        else:
            assert max(g for g, _, _ in estimate["points"]) == fe2s2_sizes[row.id], row.id
    spin_mux = next(row for row in rows if row.id == "SpinMux")
    assert report.projection(cache, spin_mux, hash_, sizes["SpinMux"])["peak_mb"] == 7000.0


def test_bounded_projection_takes_every_measurement():
    """A target no larger than some measured member of its family projects to the family's
    largest time and largest peak, whatever their gate counts; gate count only decides between
    this and extrapolation. From (100 gates, 3,500 s, 7,000 MB) and (200 gates, 10 s, 100 MB), a
    150-gate target projects 3,500 s and 7,000 MB, not the larger member's 10 s and 100 MB: within
    60 minutes and 7.5 GB, refused under the small systems' 4 GB cap, and refused under 60 minutes
    once the smaller member took 3,700 s."""
    points = [(100, 3500.0, 7000.0), (200, 10.0, 100.0)]
    estimate = provenance.project_compile(points, 150)
    assert (estimate["method"], estimate["seconds"], estimate["peak_mb"]) == (
        "bounded", 3500.0, 7000.0)
    assert estimate["allowed"]
    assert provenance.project_compile(points, 150, cap_mb=report.SMALL_CAP_MB)["reason"] == (
        "the family's largest measurement; 6.8 GB > 3.8 GB")
    slow = provenance.project_compile([(100, 3700.0, 7000.0), (200, 10.0, 100.0)], 150)
    assert not slow["allowed"] and slow["reason"] == (
        "the family's largest measurement; 62 min > 60 min")


def _copy_package(root: Path) -> Path:
    shutil.copytree(REPO / "src" / "dfthc", root / "src" / "dfthc",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "scripts").mkdir()
    shutil.copy(SCRIPT, root / provenance.REPORT_SCRIPT)
    shutil.copy(REPO / provenance.LOCK_FILE, root / provenance.LOCK_FILE)
    return root


def test_cache_needs_matching_code_hash(tmp_path, monkeypatch):
    """code_hash() ignores docstrings, comments and dfthc/testing/ and sees the package's code,
    the report script's code (its program builders) and the lock file; a cached entry of another
    code hash is kept as history and never shown as a current measurement, while an entry of the
    current hash is reused without compiling."""
    root = _copy_package(tmp_path / "copy")
    base = provenance.code_hash(root)
    path = root / "src" / "dfthc" / "costs.py"
    text = path.read_text()
    path.write_text(text.replace('"""Toffoli count ``T`` and explicit measurement count',
                                 '"""Toffoli count ``T`` and the explicit measurement count')
                    .replace("# --- building blocks", "# --- the building blocks"))
    assert path.read_text() != text and provenance.code_hash(root) == base
    (root / "src" / "dfthc" / "testing" / "fixtures.py").write_text("X = 1\n")
    assert provenance.code_hash(root) == base
    script = root / provenance.REPORT_SCRIPT
    source = script.read_text()
    script.write_text(source.replace('"""The walk or controlled walk with its phase gradient',
                                     '"""The walk, or the controlled walk, with its phase gradient')
                      .replace('sys.path.insert(0, str(REPO / "src"))',
                               'sys.path.insert(0, str(REPO / "src"))  # the package'))
    assert script.read_text() != source and provenance.code_hash(root) == base
    script.write_text(source.replace("return qfunc, s.layout.num_wires + int(controlled)",
                                     "return qfunc, s.layout.num_wires + 1"))
    assert provenance.code_hash(root) != base
    script.write_text(source)
    path.write_text(text.replace("return 2**ob - 2 if ob >= 1 else 0",
                                 "return 2**ob - 3 if ob >= 1 else 0"))
    assert provenance.code_hash(root) != base

    out = tmp_path / "out"
    out.mkdir()
    measured = {"toffoli": 2, "measurements": 0, "measured_and_uncomputes": 0,
                "counts": {"Toffoli": 2}, "num_wires": 5, "seconds": 1.0,
                "watchdog": {"seconds": 3.0, "peak_mb": 200.0}, "gates": 10}
    stale = {"code_hash": "0" * 64, "environment": {}, "status": "compiled"} | measured
    (out / "resources.json").write_text(json.dumps(
        {"compiled": {"H2_STO3G_TEST/SpinSwap": [stale]}}))
    monkeypatch.setattr(report, "measure", lambda *args: pytest.fail("nothing is compiled"))
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "--out", str(out)]) == 0
    record = json.loads((out / "resources.json").read_text())
    row = next(r for r in record["systems"]["H2_STO3G_TEST"]["rows"] if r["id"] == "SpinSwap")
    assert (row["status"], row["measured"]) == ("pending", None)
    assert record["compiled"]["H2_STO3G_TEST/SpinSwap"] == [stale]
    assert "kept as history, never shown as current measurements: 1." in (
        out / "resources.md").read_text()

    fresh = stale | {"code_hash": provenance.code_hash()}
    record["compiled"]["H2_STO3G_TEST/SpinSwap"].append(fresh)
    (out / "resources.json").write_text(json.dumps(record))
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "SpinSwap", "--out",
                        str(out)]) == 0
    record = json.loads((out / "resources.json").read_text())
    row = next(r for r in record["systems"]["H2_STO3G_TEST"]["rows"] if r["id"] == "SpinSwap")
    assert (row["status"], _counts(row, "measured")) == ("measured", (2, 0, 0))


BE_DFTHCBE = {"cmp-free": -93, "rot-add": 12, "exact-iter": 74, "no-prune": 2, "flat-r": 120,
              "mux": -4, "sf-enable": 2, "onehot-pad": 40}
BE_PAPER = {"cmp-free": -93, "rot-add": 12, "unif-odd": 6, "exact-iter": 74, "iter-const": -26,
            "onehot-const": -8, "flat-r": 120, "sign-bits": 60, "no-prune": 20, "prefix-stops": 16,
            "mux": 4, "rz-addsub": -424, "maj-and": -4, "sf-enable": 2, "onehot-pad": 40,
            "ref-unif": -1}
"""The FeMoco-54 block encoding's contributions (the plan's decomposition table)."""


@pytest.fixture(scope="module")
def femoco_report(tmp_path_factory):
    """The report's FeMoco-54 record and markdown without compiling (`main --rows`)."""
    out = tmp_path_factory.mktemp("femoco")
    assert report.main(["--systems", "femoco54", "--rows", "--out", str(out)]) == 0
    return (json.loads((out / "resources.json").read_text())["systems"]["femoco54"],
            (out / "resources.md").read_text())


def test_reason_codes_at_femoco(femoco_report):
    """The reason codes of the angle loads and erase, the reflections, block encodings, walks and
    walk powers at FeMoco-54, against dfthcbe and the paper, are exactly the causes with a nonzero
    contribution, each with its contribution (the plan's decomposition table; nothing is
    compiled). The angle rows split exact against truncated iteration (`exact-iter`) from no
    aligned-block pruning (`no-prune`) at the intermediate of `iteration_counts`: the angle
    erase's +7 against dfthcbe is +2 over `r` and +5 over the tail rows (ledger row 2E.2). The
    tail walks reach the paper through dfthcbe's: its `s(RC)` = 4 stops on the dead SF prefix
    (`prefix-stops`) and its truncated iteration's -2 (`iter-const`). The QPE control is a cause
    only where our program lacks it, and the controlled REF_T2 shows the two causes that cancel
    against the paper."""
    rows = {r["id"]: r for r in femoco_report[0]["rows"]}
    cwalk = BE_PAPER | {"ctrl-ref-t2": 1, "ref-target": -1}
    expected = {
        "TailAngleLoad": ({"no-prune": 5}, {"no-prune": 5, "prefix-stops": 4, "iter-const": -2}),
        "SFAngleLoad": ({"exact-iter": 22, "no-prune": -2}, {"exact-iter": 22, "iter-const": -1}),
        "AngleErase": ({"exact-iter": 2, "no-prune": 5, "onehot-pad": 4, "sf-enable": 1},
                       {"exact-iter": 2, "no-prune": 5, "prefix-stops": 4, "iter-const": -4,
                        "onehot-pad": 4, "onehot-const": -1, "sf-enable": 1, "attrib": 54}),
        "RefT2": ({}, {"ref-unif": -1}),
        "ControlledRefT2": ({}, {"ref-unif": -1, "ctrl-ref-t2": 1}),
        "RefAB": ({"ref-target": -1}, {"ref-target": -1, "qpe-ctrl": -1}),
        "ControlledRefAB": ({}, {"ref-target": -1}),
        "DFTHCBlockEncoding": (BE_DFTHCBE, BE_PAPER),
        "ControlledDFTHCBlockEncoding": (BE_DFTHCBE, BE_PAPER | {"ctrl-ref-t2": 1}),
        "DFTHCWalk": (BE_DFTHCBE | {"ref-target": -1}, BE_PAPER | {"ref-target": -1,
                                                                    "qpe-ctrl": -1}),
        "ControlledDFTHCWalk": (BE_DFTHCBE, cwalk),
        "ControlledWalkPower(p=1)": ({}, cwalk),
        "ControlledWalkPower(p=2)": ({}, {code: 2 * value for code, value in cwalk.items()}),
    }
    assert {row_id: (rows[row_id]["codes_dfthcbe"], rows[row_id]["codes_paper"])
            for row_id in expected} == expected
    assert [rows[row_id]["delta_paper"] for row_id in ("DFTHCWalk", "ControlledDFTHCWalk",
                                                       "ControlledWalkPower(p=2)")] == [
        -204, -202, -404]


def test_qpe_summary_and_wire_boundaries(femoco_report):
    """The FeMoco-54 QPE summary gives T, M and A from the controlled walk, A = (2^t - 1) A_C(W)
    + (t^3 - t)/6 (the phase4-qpe plan's interface), in its record, its basis sentence and its
    table; and the QPE wires step by two where the phase register grows a spare wire:
    `H2_STO3G_TEST` 59, 60, 62 at t = 3, 4, 5 and FeMoco-54 1,124, 1,125, 1,127 at t = 21, 22,
    23 (nothing is compiled)."""
    record, markdown = femoco_report
    cwalk = next(r for r in record["rows"] if r["id"] == "ControlledDFTHCWalk")["closed"]
    assert cwalk["measured_and_uncomputes"] == 8677
    for bits in (1, 2, 3, 10, 20, 23):
        qpe = record["qpe"][str(bits)]
        assert (qpe["toffoli"], qpe["measurements"], qpe["measured_and_uncomputes"]) == (
            (2**bits - 1) * cwalk["toffoli"] + (bits**3 - bits) // 6,
            (2**bits - 1) * cwalk["measurements"] + bits,
            (2**bits - 1) * cwalk["measured_and_uncomputes"] + (bits**3 - bits) // 6)
    assert [record["qpe"][str(bits)]["measured_and_uncomputes"] for bits in (1, 2, 3)] == [
        8677, 26032, 60743]
    assert [record["qpe"][str(bits)]["wires"] for bits in (20, 23)] == [1123, 1127]
    assert "A = (2^t - 1) A_C(W) + (t^3 - t)/6" in record["qpe_basis"]
    assert "| t | T | M | A | Qubits |" in markdown
    h2, femoco = report.build_system("H2_STO3G_TEST"), report.build_system("femoco54")
    assert [report.qpe_wires(h2, bits) for bits in (3, 4, 5)] == [59, 60, 62]
    assert [report.qpe_wires(femoco, bits) for bits in (21, 22, 23)] == [1124, 1125, 1127]


def test_qubit_summary_at_femoco(femoco_report):
    """The FeMoco-54 qubit section (spec 10.3, 13.2) after Q-1 variant (b): WalkLayout's 1,102
    wires by group against the paper's Table 3 groups, 1,131 in all (-29: outer +21, inner +12,
    select +4, work -66); the junk registers' 462 wires share the angle wires and their 852
    measurements per walk are the block encoding's own; the early junk measurement is no longer
    an open question or a deferred refinement (nothing is compiled)."""
    record, markdown = femoco_report
    assert record["wires"] == 1102 and record["qubit_groups"] == {
        "system": 108, "outer": 71, "inner": 65, "select": 799, "phase_grad": 15, "work": 44}
    assert (record["junk_wires"], record["junk_measurements"]) == (462, 852)
    paper = costs_reference.PAPER_QUBIT_GROUPS_FEMOCO
    assert sum(paper.values()) == costs_reference.PAPER_QUBITS_FEMOCO == 1131
    assert {g: record["qubit_groups"][g] - paper[g] for g in paper} == {
        "system": 0, "outer": 21, "inner": 12, "select": 4, "phase_grad": 0, "work": -66}
    assert "| **total** | **1,102** | **1,131** | **-29** |" in markdown
    assert "852 of the M are the block encoding's own junk measurements" in markdown
    assert "Q-1, open" not in markdown and "Early junk measurement" not in markdown
    assert "`junk-qubits`" not in markdown


def test_dfthcbe_blocks_equal_the_call_graph():
    """costs_reference.dfthcbe_blocks equals dfthcbe's own call graph (Qualtran, .venv-ref;
    scripts/make_report_fixture.py) at every parameter set of the report, and the 12 FeMoco-54
    block counts of Task 1A.6's fixture. The call graph's tail walk is its N leaves and its s(RC)
    stops on the dead SF prefix, truncated (-2), which is how iteration_causes takes the tail to
    the paper; and the iteration intermediate is dfthcbe's walk where it is a valid circuit."""
    fixture = load_dfthcbe_fixture("report_toffolis.json")
    for name, values in fixture["toffolis"].items():
        params = WalkParams(**fixture["parameters"][name])
        assert params == (FEMOCO54 if name == "femoco54" else report.build_system(name).params)
        port = costs_reference.dfthcbe_blocks(params)
        assert {key: port[key] for key in values} == values, name
        tail = costs_reference.iteration_causes(params)["tail"][1]
        assert (tail["prefix-stops"], tail["iter-const"]) == (
            costs_reference.popcount(params.RC), -2), name
        assert values["tail_angle_load"] == params.N + tail["prefix-stops"] + tail["iter-const"]
        counts = costs_reference.iteration_counts(params)
        assert all(counts[key][1] == counts[key][2]
                   for key in ("outer_read", "outer_erase", "sf_fixup")), name
    femoco = load_dfthcbe_fixture("femoco_block_toffolis.json")
    port = costs_reference.dfthcbe_blocks(FEMOCO54)
    ours = {"inner_select_adjoint": "inner_select_dagger"}
    assert {key: port[ours.get(key, key)] for key in femoco} == femoco


def test_paper_table3_reproduces_table_iii():
    """paper_table3 gives Table III's FeMoco-54 values per call (spec 13.1): 9,997 per walk,
    with the tail fix-up under the inner erase (Table III) or at step 18."""
    paper = costs_reference.paper_table3(FEMOCO54)
    assert {key: paper[key] for key in (
        "uniform_x_o", "outer_prepare", "outer_unprepare", "uniform_b", "inner_prepare",
        "inner_unprepare", "inner_unprepare_step18", "tail_angle_load", "sf_angle_load",
        "spin_mux", "spin_mux_app_b3", "spin_swap", "givens_ladder", "majorana_gate",
        "angle_erase", "angle_erase_step18", "inner_select", "inner_select_step18", "ref_t2",
        "ref_ab", "outer_erase", "inner_erase", "block_encoding", "walk")} == {
        "uniform_x_o": 36, "outer_prepare": 213, "outer_unprepare": 97, "uniform_b": 20,
        "inner_prepare": 880, "inner_unprepare": 274, "inner_unprepare_step18": 220,
        "tail_angle_load": 54, "sf_angle_load": 270, "spin_mux": 0, "spin_mux_app_b3": 1,
        "spin_swap": 54, "givens_ladder": 1590, "majorana_gate": 7, "angle_erase": 37,
        "angle_erase_step18": 91, "inner_select": 3656, "inner_select_step18": 3710,
        "ref_t2": 21, "ref_ab": 46, "outer_erase": 37, "inner_erase": 180,
        "block_encoding": 9951, "walk": 9997}


def test_project_compile_is_the_k1_method():
    """Task 0.9's projection beyond the measured sizes: a power law in time through the two
    largest points (exponent at least 1) and a least-squares line in memory (slope at least 0);
    within them, the largest of all the measurements; refused above 60 min or above
    the 7.5 GB kill point of the 8 GB cap, so a 7.8 GB projection is refused."""
    points = [(1000, 10.0, 500.0), (2000, 20.0, 600.0), (4000, 80.0, 700.0)]
    result = provenance.project_compile(points, 8000)
    slope, intercept = np.polyfit([1000, 2000, 4000], [500, 600, 700], 1)
    assert result["allowed"] and result["time_exponent"] == 2.0
    assert result["seconds"] == pytest.approx(320.0, rel=0, abs=1e-9)
    assert result["peak_mb"] == pytest.approx(intercept + slope * 8000, rel=0, abs=0.05)
    assert provenance.project_compile([(1000, 10.0, 500.0), (2000, 11.0, 500.0)],
                                      4000)["seconds"] == pytest.approx(22.0, rel=0, abs=1e-9)
    refused = provenance.project_compile([(1000, 1.0, 1000.0), (2000, 2.0, 2000.0)], 7800)
    assert refused["peak_mb"] == pytest.approx(7800.0, rel=0, abs=0.05)
    assert not refused["allowed"] and refused["reason"] == "7.6 GB > 7.5 GB"
    assert not provenance.project_compile([(1000, 1.0, 1.0)], 2000)["allowed"]
    bounded = provenance.project_compile([(50, 9.0, 280.0), (100, 5.0, 300.0),
                                          (200, 4.0, 310.0)], 80)
    assert (bounded["method"], bounded["seconds"], bounded["peak_mb"]) == ("bounded", 9.0, 310.0)
    assert bounded["allowed"] and result["method"] == "k1"
    slow = provenance.project_compile([(100, 4000.0, 300.0)], 50)
    assert not slow["allowed"] and slow["reason"] == (
        "the family's largest measurement; 67 min > 60 min")
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/test_registers.py tests/test_work_pool.py -q`
Expected: FAIL, `2 errors`: `ImportError` while importing each module (`JUNK` is not in `dfthc.registers`; `FROM_OUTCOMES` is not in `dfthc.ops.walk`).

Run: `scripts/test.sh tests/ops/test_block_encoding.py tests/ops/test_walk.py tests/test_qpe.py tests/report/test_report.py -m "not slow and not release" -q`
Expected: FAIL, `24 failed, 138 passed` (`q1b2-red-detail.log`): in `test_block_encoding.py` `test_required_wires`, `test_children_order_and_registers[sto3g]`, `[fails-c3]`, `test_junk_is_measured_right_after_each_prepare[sto3g]`, `[631g]`, `test_femoco_counts_on_the_closed_forms`; in `test_walk.py` `test_required_wires`, `test_constructor_validates[plain-outer]`, `[controlled-outer]`; in `test_qpe.py` `test_controlled_walk_power_validates`, `test_qpe_program_wires[3-spare0]`, `[5-spare1]`, `test_spare_wires_needed`, `test_walk_power_spec_flattens_the_groups_in_wire_argnames_order`; in `test_report.py` `test_report_h2_rows_and_columns`, `test_report_h2[...]` for the six new rows (`InnerUnprepareFromOutcomes`, `OuterUnprepareFromOutcomes`, their `Adjoint(FlaggedUniformPrep)@...` and `PhaseFixup@...` rows), `test_every_large_row_projects`, `test_qpe_summary_and_wire_boundaries`, `test_qubit_summary_at_femoco`.

- [ ] **Step 3: Move the junk onto the angle wires**

Replace `src/dfthc/registers.py` with:

```python
"""Named wires of the walk: ``WalkLayout`` (spec section 10)."""

from __future__ import annotations

from pennylane.wires import Wires

from dfthc.params import WalkParams, lg, odd_part

GROUPS: dict[str, tuple[str, ...]] = {
    "system": ("psi_up", "psi_dn"),
    "outer": ("x_o", "succ_x_o", "succ_work_x_o", "sigma_1", "m_x_o", "less_than_x_o",
              "cmp_work_x_o", "alt_x_o", "keep_x_o"),
    "inner": ("b", "succ_b", "succ_work_b", "sigma_0", "m_b", "less_than_b", "cmp_work_b",
              "alt_b", "keep_b", "sgn", "d_sgn", "r"),
    "select": ("angles", "junk_overflow", "G0", "G1", "b_neq_B", "sg"),
    "phase_grad": ("phase_grad",),
    "work": ("work",),
}
"""Groups of spec 10.1 and their registers, in layout order."""

JUNK = ("junk_x_o", "junk_b")
"""The reads' junk registers, which have no wires of their own (spec 10.2; spec 16, Q-1 variant
(b)): each is the first wires of ``angles ‖ junk_overflow``, the start of the ``select`` group.
The block encoding X-measures each junk register right after its prepare, before any angle is
loaded, so the junk and the angles never hold data at the same time."""


class WalkLayout:
    """Named wires for every register of the walk (spec 10.1 and 10.2).

    Build it with :meth:`from_params`. Every register of :data:`GROUPS` is an attribute holding
    its ``Wires``; a register of width 0 is an empty ``Wires``. With ``qpe_bits > 0`` the layout
    also has ``qpe_control`` (1 wire) and ``qpe_grad`` (``qpe_bits`` wires) after ``work``.
    Wires are consecutive integers from 0 in group order: ``system``, ``outer``, ``inner``,
    ``select``, ``phase_grad``, ``work``. The registers of :data:`GROUPS` never overlap; the
    junk registers of :data:`JUNK` are attributes too, aliases of the start of ``select``
    (:meth:`split_junk`).

    The register sizes follow spec 9.7 and 9.8: ``succ_work_* = max(lg(l) - 1, 0)`` with ``l``
    the odd part of ``X_o`` or ``B + 1``, ``cmp_work_* = b_k - 1``, ``junk_x_o = (K1 - 1) d1``
    and ``junk_b = (K2 - 1) d2`` (:meth:`junk_sizes`); ``junk_overflow`` holds the junk wires
    beyond ``angles``, ``max(junk_x_o, junk_b, (N - 1) b_rot) - (N - 1) b_rot`` (0 at FeMoco-54,
    3 at ``H2_STO3G_TEST``). The work pool size is :meth:`work_size`. That formula is
    provisional: Task 3.3 asserts that every operator's ``required_wires()["work"]`` fits it.
    """

    def __init__(self, params: WalkParams, qpe_bits: int, registers: dict[str, Wires]):
        self.params = params
        self.qpe_bits = qpe_bits
        self._registers = dict(registers)
        for name, wires in registers.items():
            setattr(self, name, wires)
        for name, wires in self.split_junk(params, self.select).items():
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
        angles = (p.N - 1) * p.b_rot
        return {
            "psi_up": p.N, "psi_dn": p.N,
            "x_o": p.w_x, "succ_x_o": 1, "succ_work_x_o": max(lg(odd_part(p.X_o)) - 1, 0),
            "sigma_1": 1, "m_x_o": p.b_k1, "less_than_x_o": 1, "cmp_work_x_o": p.b_k1 - 1,
            "alt_x_o": p.w_x, "keep_x_o": p.b_k1,
            "b": p.w_b, "succ_b": 1, "succ_work_b": max(lg(odd_part(p.B + 1)) - 1, 0),
            "sigma_0": 1, "m_b": p.b_k2, "less_than_b": 1, "cmp_work_b": p.b_k2 - 1,
            "alt_b": p.w_b, "keep_b": p.b_k2, "sgn": 1, "d_sgn": 1, "r": p.w_r,
            "angles": angles,
            "junk_overflow": max(*WalkLayout.junk_sizes(p).values(), angles) - angles,
            "G0": 1, "G1": 1, "b_neq_B": 1, "sg": 1,
            "phase_grad": p.b_grad,
            "work": WalkLayout.work_size(p),
        }

    @staticmethod
    def junk_sizes(params: WalkParams) -> dict[str, int]:
        """Width of every junk register of :data:`JUNK`: ``(K1 - 1) d1`` and ``(K2 - 1) d2``."""
        p = params
        return {"junk_x_o": (p.K1 - 1) * p.d1, "junk_b": (p.K2 - 1) * p.d2}

    @staticmethod
    def split_junk(params: WalkParams, select) -> dict:
        """The junk registers as the first wires of the ``select`` group, which starts with
        ``angles ‖ junk_overflow`` (spec 10.2): slices of the same type, as in
        :meth:`split_group`. Raises ``ValueError`` for a wrong length."""
        sizes = WalkLayout.register_sizes(params)
        total = sum(sizes[name] for name in GROUPS["select"])
        if len(select) != total:
            raise ValueError(f"select needs {total} wires; got {len(select)}")
        return {name: select[:size] for name, size in WalkLayout.junk_sizes(params).items()}

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

- [ ] **Step 4: Measure the junk in the block encodings**

Replace `src/dfthc/ops/walk.py` with:

```python
r"""The block encoding of the sum-of-squares Hamiltonian, the qubitized walk and their controlled
classes (spec 9.21 and 9.22; steps 1 to 28).

**The circuit** (spec 3.3 and 3.5). The block encoding is :math:`U^\dagger\,\mathrm{REF}_{T_2}\,U`
with :math:`U` = ``OuterPrepare``, ``InnerPrepare``, ``InnerSigns``, ``InnerSelect``,
``InnerUnprepareFromOutcomes`` in time order and :math:`U^\dagger` = ``InnerPrepare``,
``InnerSelectDagger``, ``InnerSigns``, ``InnerUnprepareFromOutcomes``,
``OuterUnprepareFromOutcomes``; its block on the ancillas in :math:`|0\rangle` is
:math:`\tilde H_{SA}/\tilde\Lambda - I` (spec 8.5). Every adjoint in it is an explicit class
(``InnerUnprepareFromOutcomes``, ``InnerSelectDagger``, ``OuterUnprepareFromOutcomes``; spec 7.4
rule 6). The walk is :math:`W = \mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}\,\mathrm{BE}`: the block
encoding, then ``RefAB``. The controlled classes control only the reflections: on states whose
ancillas start in :math:`|0\rangle`, :math:`U^\dagger U = I`, so
:math:`C(W) = C(\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}})\,U^\dagger\,C(\mathrm{REF}_{T_2})\,U`
(spec 3.3). There is no ``C(...)`` rule of any of these operators.

**Early junk measurement** (spec 16, Q-1 variant (b)). Right after each prepare the block
encoding's rule X-measures the prepare's junk register (``H`` and ``qp.measure(q, reset=True)``
on every qubit) and later passes the outcomes to the matching unprepare from outcomes, which
measures only the read's target and fixes the phases from both outcome vectors. The junk wires
are free from their measurement on, so the layout gives them to the angle registers
(:data:`~dfthc.registers.JUNK`): the angles are loaded only after the junk is measured, and the
unprepares borrow their ``extra_onehot`` wires from the start of the same, by then reset, wires.
T and A are those of the baseline unprepares, and the total M is unchanged: the junk's share
moves into this rule (:func:`dfthc.costs.junk_measurements`).

**Wires.** Each class takes the groups of :class:`~dfthc.registers.WalkLayout` (``outer``,
``inner``, ``select``, ``system``, ``phase_grad``, ``work``; a controlled class ``control``
first). Its rule splits them once with :meth:`WalkLayout.split_group` and
:meth:`WalkLayout.split_junk`, unpacks every register into a list once, and passes each child the
registers its ``wire_argnames`` name, with the prefix of the ``work`` group of exactly the child's
need (spec 9.0, wire passing). ``work`` is the walk's shared pool: every child returns it in
:math:`|0\rangle`; the comparators' dirty work registers belong to the ``outer`` and ``inner``
groups, never to ``work`` (spec 10.2).

**Arguments.** ``params`` is :meth:`WalkParams.as_compilable` and ``tables`` is
:meth:`WalkTables.as_compilable`; the rules rebuild both with ``from_compilable``. One list,
:func:`_children`, gives the rule its steps in time order (the children with their compilable
arguments, work needs and lent one-hot wires, and the junk measurements), and the resource
function its declaration, so the two cannot disagree; :func:`_walk_children` does the same for
the walks.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.data.tables import WalkTables
from dfthc.ops._validate import check_wires
from dfthc.ops.prepare import (
    InnerPrepare,
    InnerSigns,
    InnerUnprepareFromOutcomes,
    OuterPrepare,
    OuterUnprepareFromOutcomes,
    _check_inner,
    _check_outer,
    abstract_junk_outcomes,
)
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefAB, RefT2
from dfthc.ops.select import InnerSelect, InnerSelectDagger
from dfthc.ops.select import _check as _check_select
from dfthc.params import WalkParams
from dfthc.registers import GROUPS, WalkLayout

_GROUPS = ("outer", "inner", "select", "system", "phase_grad", "work")
"""The wire arguments of the assembly operators, after ``control`` in the controlled ones."""

JUNK_MEASUREMENT = "junk measurement"
"""The step of :func:`_children` that X-measures the junk register its ``args`` names."""

FROM_OUTCOMES = {OuterUnprepareFromOutcomes: "junk_x_o", InnerUnprepareFromOutcomes: "junk_b"}
"""Each unprepare from outcomes and the junk register whose outcomes it takes and whose first
wires it borrows as ``extra_onehot``."""


class Step(NamedTuple):
    """One step of an assembly rule: a child with its compilable arguments, the prefix of the
    ``work`` pool it takes and its ``extra_onehot`` wires (for an unprepare from outcomes), or,
    with ``op`` = :data:`JUNK_MEASUREMENT`, the X measurement of the junk register ``args``."""

    op: type | str
    args: tuple | str
    work: int
    extra_onehot: int = 0


def _rebuild(params, tables) -> tuple[WalkParams, WalkTables]:
    return WalkParams.from_compilable(params), WalkTables.from_compilable(tables)


def _children(p: WalkParams, t: WalkTables, controlled: bool) -> list[Step]:
    """The block encoding's steps in time order (spec 9.21; spec 16, Q-1 variant (b)): the
    children, each junk register's X measurement right after its prepare, and the unprepares from
    those outcomes; the reflection is ``ControlledRefT2`` if ``controlled``."""
    outer = (p.X_o, t.outer_alt, t.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)
    inner = (p.N, p.R, p.B, p.C, t.inner_alt, t.inner_keep, t.inner_sgn, t.inner_d_sgn, p.b_k2,
             p.k2, p.ke2, p.s_u)
    select = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    reflection = ControlledRefT2 if controlled else RefT2
    outer_sizes = OuterUnprepareFromOutcomes.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)
    inner_sizes = InnerUnprepareFromOutcomes.required_wires(p.N, p.R, p.B, p.C, p.b_k2, p.k2,
                                                            p.ke2, p.s_u)
    outer_work, inner_work = outer_sizes["work"], inner_sizes["work"]
    select_work = InnerSelect.required_wires(*select)["work"]
    reflection_work = reflection.required_wires(p.w_b, p.b_k2)["work"]
    return [
        Step(OuterPrepare, outer, outer_work),
        Step(JUNK_MEASUREMENT, "junk_x_o", 0),
        Step(InnerPrepare, inner, inner_work),
        Step(JUNK_MEASUREMENT, "junk_b", 0),
        Step(InnerSigns, (), 0),
        Step(InnerSelect, select, select_work),
        Step(InnerUnprepareFromOutcomes, inner, inner_work, inner_sizes["extra_onehot"]),
        Step(reflection, (), reflection_work),
        Step(InnerPrepare, inner, inner_work),
        Step(JUNK_MEASUREMENT, "junk_b", 0),
        Step(InnerSelectDagger, select, select_work),
        Step(InnerSigns, (), 0),
        Step(InnerUnprepareFromOutcomes, inner, inner_work, inner_sizes["extra_onehot"]),
        Step(OuterUnprepareFromOutcomes, outer, outer_work, outer_sizes["extra_onehot"]),
    ]


def _group_sizes(p: WalkParams) -> dict[str, int]:
    """The width of every group: the layout's (spec 10.1)."""
    sizes = WalkLayout.register_sizes(p)
    return {group: sum(sizes[name] for name in GROUPS[group]) for group in _GROUPS}


def _sizes(p: WalkParams) -> dict[str, int]:
    """The width of every register, junk register and group a step of an assembly rule can
    take."""
    return (WalkLayout.register_sizes(p) | WalkLayout.junk_sizes(p) | _group_sizes(p)
            | {"control": 1})


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
    needs = {step.op.__name__: step.work for step in _children(p, t, controlled=True)
             if step.op != JUNK_MEASUREMENT}
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
    """Every group, every register and both junk registers (aliases of the start of ``select``)
    by name, each unpacked into a list once (spec 9.0)."""
    registers = {"outer": list(outer), "inner": list(inner), "select": list(select),
                 "system": list(system), "phase_grad": list(phase_grad),
                 "control": list(control)}
    for group in ("outer", "inner", "select", "system"):
        registers |= WalkLayout.split_group(p, group, registers[group])
    return registers | WalkLayout.split_junk(p, registers["select"])


def _measure_x(wire):
    qp.H(wire)
    return qp.measure(wire, reset=True)


def _emit(steps: list, registers: dict, work) -> None:
    """Apply ``steps`` in order: each child on the registers its ``wire_argnames`` name and on
    the prefix of ``work`` of exactly its need; each junk measurement on every wire of its
    register, keeping the outcomes; an unprepare from outcomes with the latest outcomes of its
    junk register and that register's first ``extra_onehot`` wires (reset by the measurement)."""
    work, outcomes = list(work), {}
    for step in steps:
        if step.op == JUNK_MEASUREMENT:
            outcomes[step.args] = [_measure_x(wire) for wire in registers[step.args]]
            continue
        wires = {name: work[:step.work] if name == "work" else registers[name]
                 for name in step.op.wire_argnames if name != "extra_onehot"}
        if step.op in FROM_OUTCOMES:
            junk = FROM_OUTCOMES[step.op]
            step.op(outcomes[junk], *step.args, **wires,
                    extra_onehot=registers[junk][:step.extra_onehot])
        else:
            step.op(*step.args, **wires)


def _resources(steps: list, p: WalkParams) -> dict:
    """The declaration of ``steps``: each distinct child once, with its count, its compilable
    arguments and exactly the register sizes :func:`_emit` passes, and the junk measurements'
    ``Hadamard`` and ``MidMeasure`` gates (spec 7.4 rule 2)."""
    sizes = _sizes(p)
    measure = resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True))
    resources = Counter()
    for step, count in Counter(steps).items():
        if step.op == JUNK_MEASUREMENT:
            resources[qp.H] += count * sizes[step.args]
            resources[measure] += count * sizes[step.args]
            continue
        dynamic = ((abstract_junk_outcomes(sizes[FROM_OUTCOMES[step.op]]),)
                   if step.op in FROM_OUTCOMES else ())

        def build(step=step, dynamic=dynamic):
            return step.op(*dynamic, *step.args, **{
                name: Wire[step.work if name == "work" else step.extra_onehot
                           if name == "extra_onehot" else sizes[name]]
                for name in step.op.wire_argnames})

        resources[resource_key(build)] += count
    return dict(resources)


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
    concatenation of named registers in layout order; the junk registers ``junk_x_o`` and
    ``junk_b`` are the first wires of ``select`` (spec 10.2).

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``outer``      the ``outer`` group (71)       ``|0>``               see the action
    ``inner``      the ``inner`` group (65)       ``|0>``               see the action
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
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
    ``b``, ``m_b``, ``sigma_0``) is back in ``|0>``: the junk and target measurements and the
    angle erase reset the loaded data, the measured uncomputes clean the comparators, and the
    adjoint uniform preparations clear the success flags. So ``RefAB``'s reflection about those
    six is the walk's reflection about every ancilla.

    **Decomposition** (spec 9.21; spec 16, Q-1 variant (b)). The groups are split with
    ``WalkLayout.split_group`` and ``WalkLayout.split_junk``; each child gets its registers by
    name and the prefix of ``work`` of its need: ``OuterPrepare``, the X measurement of
    ``junk_x_o``; ``InnerPrepare``, the X measurement of ``junk_b``, ``InnerSigns``,
    ``InnerSelect``, ``InnerUnprepareFromOutcomes``; ``RefT2``; ``InnerPrepare``, the X
    measurement of ``junk_b``, ``InnerSelectDagger``, ``InnerSigns``,
    ``InnerUnprepareFromOutcomes``; ``OuterUnprepareFromOutcomes``. An X measurement is ``H`` and
    ``qp.measure(q, reset=True)`` on every junk wire; each unprepare from outcomes gets the
    outcomes of the latest measurement of its junk register and that register's first
    ``extra_onehot`` wires, reset by then (the select loads and erases the angles on the same
    wires in between). The prepare classes take the whole ``phase_grad`` (they use its top
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
    one). Of ``M``, :math:`(K_1 - 1) d_1 + 2 (K_2 - 1) d_2` are this rule's own junk
    measurements (:func:`dfthc.costs.junk_measurements`) and the unprepares from outcomes
    measure the targets, :math:`d_1 + 2 d_2`. At FeMoco-54:
    :math:`279 + 2 \cdot 4725 + 20 = 9749`, ``M = 2524`` (852 of them junk), ``A = 8632``.
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
    free comparator adjoints -46 (``cmp-free``) and the rest +2 (spec 13.1). As the paper does
    (App. B.7.2, steps 2 and 4, and Table 3, whose inner QROM temporaries share the angle
    qubits), the rule measures each read's junk right after its prepare; dfthcbe keeps the junk
    until the erase.

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
    ``outer``      the ``outer`` group (71)       ``|0>``               see the action
    ``inner``      the ``inner`` group (65)       ``|0>``               see the action
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
    ``system``     ``2N`` (108)                   any                   see the action
    ``phase_grad`` ``b_rot`` (15)                 phase-gradient state  unchanged
    ``work``       ``WalkLayout.work_size`` (44)  ``|0>``               ``|0>``
    ============== ============================== ===================== ======================

    **Compilable arguments.** Those of :class:`DFTHCBlockEncoding`, with the same checks.

    **Pre- and postconditions.** Those of :class:`DFTHCBlockEncoding`, on both control
    branches; ``control`` is disjoint from the groups (unchecked).

    **Decomposition** (spec 9.21). :class:`DFTHCBlockEncoding`'s eleven children and three junk
    measurements with ``ControlledRefT2(control, succ_x_o, b, m_b, sigma_0, work)`` in place of
    ``RefT2``: every other step is uncontrolled, the same operator with the same compilable
    arguments and register sizes as in :class:`DFTHCBlockEncoding`, so the measurements and the
    measured unprepares are never controlled (spec 7.4 rule 5).

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


def _walk_children(p: WalkParams, t: WalkTables, controlled: bool) -> list[Step]:
    """The walk's children (spec 9.22): the block encoding on the whole groups and pool, then
    ``RefAB``; both controlled if ``controlled``."""
    block_encoding = ControlledDFTHCBlockEncoding if controlled else DFTHCBlockEncoding
    reflection = ControlledRefAB if controlled else RefAB
    return [Step(block_encoding, (p.as_compilable(), t.as_compilable()), WalkLayout.work_size(p)),
            Step(reflection, (), reflection.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)["work"])]


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
    ``outer``      the ``outer`` group (71)       ``|0>``               see the action
    ``inner``      the ``inner`` group (65)       ``|0>``               see the action
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
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

    **Paper and dfthcbe.** Steps 1 to 28 of spec 3.5: the walk
    :math:`W = \mathrm{Ref}_{\mathrm{a}\mathcal{B}}\,U^\dagger\,\mathrm{Ref}_{\mathcal{B}}\,U`
    with eigenphases :math:`\pm\arccos(E_k/\Lambda - 1)` (PRX 15, 041016; arXiv:2502.15882v1
    Eq. (10), App. A.2 Eqs. (76) and (77), :math:`\mathrm{Ref} = 2|0\rangle\langle 0| - I` as in
    App. A.1), whose reflection covers every ancilla of the linear combination and of the
    qubitization signal (App. B); its Table 3 (PRX Table III) gives 9,997 Toffolis at FeMoco-54,
    46 of them for :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` with the phase-estimation
    control (App. B.7, Eqs. (96) and (128)). Ours is -204: the block encoding's -202 and
    ``RefAB``'s 44 against 46 (spec 13.1). dfthcbe: ``DFTHCWalk``
    (``dfthcbe/block_encoding.py``), the same two children with the same sign (Qualtran's
    ``reflection_around_zero`` with ``global_phase=-1``), :math:`9596 + 45 = 9641` by its call
    graph; ours is +152: the block encoding's +153 and ``RefAB``'s 44 against 45 (dfthcbe's
    uncontrolled reflection adds a phase target).

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
    \otimes W` on inputs whose ancilla groups are in :math:`|0\rangle` and on every state the walk
    reaches from there, where only the six index registers of :math:`\mathcal{A}\cup\mathcal{B}`
    are nonzero (as between QPE rounds; the prepares and unprepares run on both branches and
    :math:`U^\dagger U = I` there too), with :math:`W` the action of :class:`DFTHCWalk` for the
    same arguments: control ``|0>`` returns the whole input,
    control ``|1>`` gives :class:`DFTHCWalk`'s whole output, global sign included, and control
    ``|+>`` their superposition (spec 12.4, E-3). The output is the same for every outcome of the
    erase measurements.

    **Registers.** Those of :class:`DFTHCWalk` after a one-wire ``control``, which stays a
    control (only phases on it).

    ============== ============================== ===================== ======================
    name           width (FeMoco-54)              input                 output
    ============== ============================== ===================== ======================
    ``control``    1                              any                   a control
    ``outer``      the ``outer`` group (71)       ``|0>``               see the action
    ``inner``      the ``inner`` group (65)       ``|0>``               see the action
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
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
    estimation (arXiv:2502.15882v1 Sec. II, after Eq. (11)). The paper counts the phase-estimation
    control on :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` only (App. B.7.2, Eq. (128)),
    which gives controlled even powers of :math:`W` because the block encoding is self-inverse;
    this class also controls :math:`\mathrm{REF}_{T_2}` (+1), so every power works. dfthcbe:
    ``DFTHCWalk(is_controlled=True)``, which also controls only the two reflections,
    :math:`9597 + 45 = 9642` Toffolis at FeMoco-54 by its call graph (its controlled ``ref_ab``
    uses the control as the phase target, so it costs the same as the uncontrolled one); ours is
    +153.

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

- [ ] **Step 5: Add the junk measurements' cost and the paper's qubit groups**

Replace `src/dfthc/costs.py` with (new: `junk_measurements`):

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
    n2, c2)``, the inner register iterated under every outer leaf, whose flag controls it:
    ``c2 = [n1 > 0 or c]``, since a zero-width outer register without control has no flag."""
    lo1, hi1, n1 = outer
    inner_controlled = n1 > 0 or controlled
    return (unary(lo1, hi1, n1, controlled)
            + (hi1 - lo1) * unary(*inner, controlled=inner_controlled))


def truncated_unary(L: int, controlled: bool) -> int:
    """Truncated iteration of Qualtran and dfthcbe over ``[0, L)``, for comparison only (spec
    9.24, "Truncated iteration"): ``max(L + c - 2, 0)`` for ``L >= 1``. A single leaf needs no
    AND: the root control, if any, is its flag. ``L < 1`` raises ``ValueError``."""
    if L < 1:
        raise ValueError(f"truncated iteration needs L >= 1 leaves; got {L}")
    return max(L + int(controlled) - 2, 0)


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


def outer_unprepare_from_outcomes(p: WalkParams) -> Cost:
    """``OuterUnprepareFromOutcomes`` (spec 9.24 row; spec 16, Q-1 variant (b)): the ``T`` of
    :func:`outer_unprepare`, ``M = d1`` (the target; the caller measures the ``(K1 - 1) d1``
    junk qubits)."""
    return Cost(outer_unprepare(p).toffoli, p.d1)


def inner_unprepare_from_outcomes(p: WalkParams) -> Cost:
    """``InnerUnprepareFromOutcomes`` (spec 9.24 row; spec 16, Q-1 variant (b)): the ``T`` of
    :func:`inner_unprepare`, ``M = d2`` (the target; the caller measures the ``(K2 - 1) d2``
    junk qubits)."""
    return Cost(inner_unprepare(p).toffoli, p.d2)


def inner_signs() -> Cost:
    """``InnerSigns`` (spec 9.24 row): ``T = 0``, ``M = 0``."""
    return Cost(0)


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


def junk_measurements(p: WalkParams) -> Cost:
    """The block encoding's own X measurements of the reads' junk (spec 9.24 row; spec 16, Q-1
    variant (b)): ``(K1 - 1) d1`` after ``OuterPrepare`` and ``(K2 - 1) d2`` after each
    ``InnerPrepare``, ``T = 0``. With the unprepares from outcomes they make up
    :func:`block_encoding`'s ``M``."""
    return Cost(0, (p.K1 - 1) * p.d1 + 2 * (p.K2 - 1) * p.d2)


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

Replace `src/dfthc/costs_reference.py` with (new: `PAPER_QUBIT_GROUPS_FEMOCO`):

```python
"""Reference costs for the resource report (spec 13.1 and 13.3): the paper's Table III formulas,
dfthcbe's per-block Toffoli counts, and the decomposition of our deltas against both into reason
codes.

All three are closed forms in pure Python, so the report never imports Qualtran in the main
environment. :func:`paper_table3` evaluates the equations of Low et al. (PRX 15, 041016;
arXiv:2502.15882v1, App. B and Table III) as Table III does: the inner entry has
``b_2 = w_b + b_k2`` bits (Table III's 20 at FeMoco-54; dfthcbe's README notes that B36 also counts
a sign bit, which Table III's numbers leave out) and the erase one-hots are ``K_5 = 2**ke1`` and
``K_4 = 2**ke2 (B + 1)``. :func:`dfthcbe_blocks` ports the per-step formulas of dfthcbe's README
("Comparison with [1]") at commit 930255020cec30172e0a7d17c0071d9476a687d1, with Qualtran's
truncated unary iteration (``max(L + c - 2, 0)``, over the live leaves and the dead stops of the
README's zero rows together) wherever the README writes ``L - 2``. The README's formulas are
piecewise, so the port is guaranteed only where it is pinned: at the eight parameter sets of the
report (the six vendored systems at their default precisions, FeMoco-54 and ``H2_STO3G_TEST``),
where ``tests/report/test_report.py::test_dfthcbe_blocks_equal_the_call_graph`` checks it against
dfthcbe's own call graph, evaluated in ``.venv-ref`` by ``scripts/make_report_fixture.py``. A new
parameter set needs a fixture entry before its dfthcbe values are trusted. Values are Toffolis per
call (``Toffoli + CSwap + And + T/4`` in Qualtran's ``QECGatesCost``); dfthcbe's Qualtran
"measurement" count is our ``A``, not our ``M``, and is not ported.
"""

from __future__ import annotations

import dataclasses

from dfthc import costs
from dfthc.params import WalkParams, lg, odd_part

DEFAULT_PRECISIONS = {"b_rot": 15, "b_k1": 15, "b_k2": 15, "s_u": 7}
"""The precisions of the paper's Table III, which the report gives every vendored system
(spec 8.3: FeMoco-54's and ``MID_NAPHTHALENE``'s)."""

PAPER_QUBITS_FEMOCO = 1131
"""The paper's logical qubits at FeMoco-54 (spec 10.3, 13.2)."""

PAPER_QUBIT_GROUPS_FEMOCO = {"system": 108, "outer": 50, "inner": 53, "select": 795,
                             "phase_grad": 15, "work": 110}
"""The paper's FeMoco-54 qubits (arXiv:2502.15882v1 Table 3) under the names of our groups: the
system, the persistent ancillas of the outer prepare, the inner prepare, RPREP (the angles) and
SEL (the rotations' b_rot), and its largest temporary ancilla count, the outer prepare's, which
it adds; the inner prepare's 313 temporaries share RPREP's 795 qubits. They sum to
:data:`PAPER_QUBITS_FEMOCO`."""


def default_params(N: int, R: int, B: int, C: int, n_plus: int) -> WalkParams:
    """A system's report parameters: :data:`DEFAULT_PRECISIONS` and the block bits that minimize
    our closed-form walk Toffolis (:func:`dfthc.costs.cheapest_block_bits`, spec 8.3)."""
    p = WalkParams(N=N, R=R, B=B, C=C, n_plus=n_plus, **DEFAULT_PRECISIONS, k1=0, k2=0, ke1=0,
                   ke2=0)
    k1, k2, ke1, ke2 = costs.cheapest_block_bits(p)
    return dataclasses.replace(p, k1=k1, k2=k2, ke1=ke1, ke2=ke2)


def popcount(value: int) -> int:
    """``s(D)`` of dfthcbe's README: the number of ones of ``value`` in binary."""
    return bin(value).count("1")


def aligned_blocks(lo: int, hi: int) -> int:
    """``beta(lo, hi)`` of dfthcbe's README: the aligned blocks ``[j 2^h, (j + 1) 2^h)`` that
    ``[lo, hi)`` splits into, each as large as alignment allows."""
    count = 0
    while lo < hi:
        size = lo & -lo if lo else 1 << (hi.bit_length() - 1)
        while lo + size > hi:
            size //= 2
        lo += size
        count += 1
    return count


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def paper_table3(p: WalkParams) -> dict[str, int]:
    """Toffolis per call of the paper's blocks (Table III; App. B equations), keyed by our row
    names. Table III charges the tail-angle fix-up ``N`` (Eq. (123), B50) to the inner erase; the
    keys ending in ``_step18`` move it to the angle erase, where dfthcbe and we apply it (spec 13.1,
    footnote 1). ``spin_mux`` is Table III's 0; ``spin_mux_app_b3`` the 1 per application that
    App. B.3 and B.7.1 count (``4(N + 1)`` for the four swaps). The paper's QPE step controls only
    ``REF_{A u B}`` (App. B.7.2, Eq. (128)), so its controlled reflection ``REF_T2``, controlled
    block encoding and controlled walk are its uncontrolled ones (keys ``controlled_ref_t2``,
    ``controlled_block_encoding``, ``controlled_walk``). The paper does not split a read from its
    keep test and swap, nor give a standalone rotation, so those rows have no key."""
    b1, b2 = p.w_x + p.b_k1, p.w_b + p.b_k2
    K5, K4 = 2**p.ke1, 2**p.ke2 * (p.B + 1)
    unif_x, unif_b = 4 * lg(p.X_o), 4 * lg(p.B + 1)                       # B24, B54; B32, B52
    outer_erase = _ceil(p.X_o, K5) + K5                                    # B53, the fix-up
    inner_erase = _ceil(p.RC * (p.B + 1), K4) + K4                         # B50, the fix-up
    rows = {
        "uniform_x_o": unif_x, "uniform_x_o_adjoint": unif_x,
        "uniform_b": unif_b, "uniform_b_adjoint": unif_b,
        "outer_erase": outer_erase, "inner_erase": inner_erase,
        "outer_prepare": unif_x + p.M1 + p.K1 * b1,                        # B24, B29
        "outer_unprepare": b1 + outer_erase + unif_x,                      # B53, B54
        "inner_prepare": unif_b + p.M2 + p.K2 * b2,                        # B32, B35, B37
        "inner_unprepare": b2 + inner_erase + p.N + unif_b,                # B50 (with N), B52
        "inner_signs": 0,                                                  # B19
        "tail_angle_load": p.N,                                            # B41
        "sf_angle_load": p.R * p.B,                                        # Table III
        "spin_mux": 0, "spin_mux_app_b3": 1,
        "spin_swap": p.N,                                                  # Table III, 2N each
        "givens_rotation": 2 * p.b_rot, "givens_rotation_adjoint": 2 * p.b_rot,
        "givens_ladder": 2 * p.b_rot * (p.N - 1),                          # Table III
        "givens_ladder_dagger": 2 * p.b_rot * (p.N - 1),
        "majorana_gate": 7, "majorana_gate_adjoint": 7,                    # Fig. 4
        "angle_erase": p.R + p.B,                                          # B47
        "ref_t2": b2 + 1,                                                  # B11
        "controlled_ref_t2": b2 + 1,                                       # B11 uncontrolled
        "ref_ab": b1 + b2 + 2,                                             # B55, QPE control
        "controlled_ref_ab": b1 + b2 + 2,                                  # B55 is controlled
    }
    rows["inner_unprepare_step18"] = rows["inner_unprepare"] - p.N
    rows["angle_erase_step18"] = rows["angle_erase"] + p.N
    select = (rows["tail_angle_load"] + rows["sf_angle_load"] + 2 * rows["spin_mux"]
              + 2 * rows["spin_swap"] + 2 * rows["givens_ladder"] + rows["majorana_gate"]
              + rows["angle_erase"])
    rows["inner_select"] = rows["inner_select_dagger"] = select
    rows["inner_select_step18"] = rows["inner_select_dagger_step18"] = select + p.N
    rows["block_encoding"] = (rows["outer_prepare"] + rows["outer_unprepare"]
                              + 2 * (rows["inner_prepare"] + select + rows["inner_unprepare"])
                              + rows["ref_t2"])
    rows["controlled_block_encoding"] = rows["block_encoding"]
    rows["walk"] = rows["block_encoding"] + rows["ref_ab"]
    rows["controlled_walk"] = rows["walk"]
    return rows


def dfthcbe_blocks(p: WalkParams) -> dict[str, int]:
    """dfthcbe's Toffolis per call, keyed by our row names: the README's per-step formulas at the
    parameters of ``p`` (its block bits included). ``spin_mux`` is half of dfthcbe's
    ``_build_sigma`` pair (4 per select); ``givens_rotation`` one rung of its ladder;
    ``rotation`` one Qualtran ``ZPowConstViaPhaseGradient`` on ``s_u`` bits, ``s_u - 2``. The
    controlled block encoding controls only ``ref_t2`` (+1); the controlled walk's ``ref_ab``
    reuses the control as its phase target (``b_1 + b_2 + 1``, as uncontrolled). dfthcbe builds
    no phase estimation, so the L4 rows have no key."""
    b1, b2 = p.w_x + p.b_k1, p.w_b + p.b_k2
    Ke1, Ke2 = 2**p.ke1, 2**p.ke2
    b_hat = _ceil(p.B + 1, p.K2)
    n_x, n_b = lg(odd_part(p.X_o)), lg(odd_part(p.B + 1))

    def uniform(n: int) -> int:
        return 4 * n + 2 * p.s_u - 5 if n else 0

    def onehot(K: int) -> int:
        return max(K - 2, 0)

    rows = {
        "rotation": p.s_u - 2,
        "uniform_x_o": uniform(n_x), "uniform_x_o_adjoint": uniform(n_x),
        "uniform_b": uniform(n_b), "uniform_b_adjoint": uniform(n_b),
        "outer_read": costs.truncated_unary(p.M1, False) + (p.K1 - 1) * b1,
        "inner_read": (costs.truncated_unary(p.M2 + aligned_blocks(p.RC, p.X_o)
                                             * popcount(b_hat), False)
                       + (p.K2 - 1) * (b2 + 2)),
        "outer_erase": costs.truncated_unary(_ceil(p.X_o, Ke1), False) + onehot(Ke1),
        "inner_erase": (costs.truncated_unary(_ceil(p.RC, Ke2) + aligned_blocks(
            _ceil(p.RC, Ke2), _ceil(p.X_o, Ke2)), False) + onehot(Ke2 * (p.B + 1))),
        "inner_signs": 0,
        "tail_angle_load": costs.truncated_unary(p.N + popcount(p.RC), False),
        "sf_angle_load": costs.truncated_unary(p.R * p.B + popcount(p.R), True),
        "spin_mux": 2,
        "spin_swap": p.N,
        "givens_rotation": 2 * (p.b_rot - 1), "givens_rotation_adjoint": 2 * (p.b_rot - 1),
        "givens_ladder": 2 * (p.N - 1) * (p.b_rot - 1),
        "givens_ladder_dagger": 2 * (p.N - 1) * (p.b_rot - 1),
        "majorana_gate": 5, "majorana_gate_adjoint": 5,
        "angle_erase": (p.R + p.B - 3) + costs.truncated_unary(p.N + popcount(p.RC), False),
        "ref_t2": b2, "controlled_ref_t2": b2 + 1,
        "ref_ab": b1 + b2 + 1, "controlled_ref_ab": b1 + b2 + 1,
    }
    rows["outer_prepare"] = rows["uniform_x_o"] + rows["outer_read"] + b1
    rows["outer_unprepare"] = b1 + rows["outer_erase"] + rows["uniform_x_o_adjoint"]
    rows["inner_prepare"] = rows["uniform_b"] + rows["inner_read"] + b2
    rows["inner_unprepare"] = b2 + rows["inner_erase"] + rows["uniform_b_adjoint"]
    select = (rows["tail_angle_load"] + rows["sf_angle_load"] + 2 * rows["spin_mux"]
              + 2 * rows["spin_swap"] + 2 * rows["givens_ladder"] + rows["majorana_gate"]
              + rows["angle_erase"])
    rows["inner_select"] = rows["inner_select_dagger"] = select
    rows["block_encoding"] = (rows["outer_prepare"] + rows["outer_unprepare"]
                              + 2 * (rows["inner_prepare"] + select + rows["inner_unprepare"])
                              + rows["ref_t2"])
    rows["controlled_block_encoding"] = rows["block_encoding"] - rows["ref_t2"] + rows[
        "controlled_ref_t2"]
    rows["walk"] = rows["block_encoding"] + rows["ref_ab"]
    rows["controlled_walk"] = rows["controlled_block_encoding"] + rows["controlled_ref_ab"]
    return rows


def _nonzero(causes: dict[str, int]) -> dict[str, int]:
    return {code: value for code, value in causes.items() if value}


def _add(*parts: tuple[dict[str, int], int]) -> dict[str, int]:
    """The sum of ``(causes, calls)`` parts, code by code."""
    total: dict[str, int] = {}
    for causes, calls in parts:
        for code, value in causes.items():
            total[code] = total.get(code, 0) + calls * value
    return total


def iteration_counts(p: WalkParams) -> dict[str, tuple[int, int, int, int]]:
    """The unary iterations of the blocks as ``(ours, intermediate, dfthcbe, paper)`` Toffolis per
    call: the counts at which :func:`iteration_causes` splits each iteration difference by
    mechanism.

    ``ours`` is exact unary iteration over the live range (spec 9.24). dfthcbe's walk truncates
    and prunes: Qualtran's truncated iteration (``L + c - 2``) over its live leaves and one stop on
    each aligned block of a dead interval that the register can hold (its README, "Zero rows").
    The paper counts one Toffoli per live leaf. The intermediate is an accounting convention, a
    cost model and not a circuit: on the walks from address 0 it is truncated iteration over the
    live leaves alone, with no stops (:func:`dfthc.costs.truncated_unary`, nested walks nested),
    and on the tail rows ``[RC, X_o)`` it is our exact count. It coincides with a valid
    construction only on the outer read, the outer erase's fix-up and the SF fix-up, where
    dfthcbe's walk has no stops and the intermediate is that walk (equal at the report's eight
    parameter sets). It does not elsewhere: truncated iteration routes a dead address that carries
    amplitude onto a live leaf, so dfthcbe stops on the zero D1/Q1 rows in the inner read, on the
    dead tail rows in the inner erase's fix-up and on the identity row ``b = B`` in the SF load,
    and the intermediate leaves those stops out. On the tail rows the dead SF prefix ``[0, RC)``
    carries amplitude and no truncated walk without its stops is valid (truncated iteration over
    ``[RC, X_o)`` assumes an in-range selection), so the convention takes our exact count there:
    the whole difference to dfthcbe is ``no-prune``, and the paper is reached through dfthcbe's
    walk (:func:`iteration_causes`).

    Keys: ``outer_read``, ``inner_read`` (nested over ``x_o`` and the inner block address),
    ``outer_erase`` and ``inner_erase`` (their fix-ups' high address), ``tail`` (the tail angle
    load and the angle erase's tail fix-up), ``sf_load`` (nested over ``r`` and ``b``, enabled on
    ``G0``) and ``sf_fixup`` (the angle erase's walk over ``r``, without its enable, which is
    ``sf-enable``)."""
    N, R, B, w_x = p.N, p.R, p.B, p.w_x
    trunc = costs.truncated_unary
    Ke1, Ke2, K4 = 2**p.ke1, 2**p.ke2, 2**p.ke2 * (B + 1)
    inner_blocks = _ceil(B + 1, p.K2)
    outer_fix, inner_fix = _ceil(p.M1 * p.K1, Ke1), _ceil(p.RC, Ke2)
    tail = costs.unary(p.RC, p.X_o, w_x, False)
    return {
        "outer_read": (costs.unary(0, p.M1, w_x - p.k1, False), trunc(p.M1, False),
                       trunc(p.M1, False), p.M1),
        "inner_read": (
            costs.unary_nested((0, p.RC, w_x), (0, inner_blocks, p.w_b - p.k2), False),
            trunc(p.RC, False) + p.RC * trunc(inner_blocks, w_x > 0),
            trunc(p.M2 + aligned_blocks(p.RC, p.X_o) * popcount(inner_blocks), False), p.M2),
        "outer_erase": (costs.unary(0, outer_fix, w_x - p.ke1, False), trunc(outer_fix, False),
                        trunc(_ceil(p.X_o, Ke1), False), _ceil(p.X_o, Ke1)),
        "inner_erase": (
            costs.unary(0, inner_fix, w_x - p.ke2, False), trunc(inner_fix, False),
            trunc(inner_fix + aligned_blocks(inner_fix, _ceil(p.X_o, Ke2)), False),
            _ceil(p.RC * (B + 1), K4)),
        "tail": (tail, tail, trunc(N + popcount(p.RC), False), N),
        "sf_load": (costs.unary(0, R, p.w_r, True) + R * costs.unary(0, B, p.w_b, True),
                    trunc(R, True) + R * trunc(B, True), trunc(R * B + popcount(R), True), R * B),
        "sf_fixup": (costs.unary(0, R, p.w_r, False), trunc(R, False), R - 2, R),
    }


def iteration_causes(p: WalkParams) -> dict[str, tuple[dict[str, int], dict[str, int]]]:
    """For every unary iteration of :func:`iteration_counts`: the signed contribution of each
    reason code to ours minus dfthcbe's and to ours minus the paper's, per call, zeros included.

    Against dfthcbe, ``exact-iter`` is ours minus the intermediate (exact against truncated
    iteration) and ``no-prune`` the intermediate minus dfthcbe (no aligned-block pruning, signed:
    negative where dfthcbe's stops cost more than the intermediate). Against the paper, on the
    walks from address 0, ``exact-iter`` again and ``iter-const``, the intermediate minus the
    paper (truncated iteration's ``c - 2``). The tail walk goes to the paper through dfthcbe's:
    ours minus dfthcbe as above, plus dfthcbe minus the paper by dfthcbe's mechanisms, its
    ``s(RC)`` stops on the dead SF prefix ``[0, RC)`` (``prefix-stops``) and its truncated
    iteration's constant over the ``N + s(RC)`` leaves and stops (``iter-const``)."""
    causes = {}
    for key, (ours, mid, ref, paper) in iteration_counts(p).items():
        against_ref = {"exact-iter": ours - mid, "no-prune": mid - ref}
        if key == "tail":
            stops = popcount(p.RC)
            against_paper = against_ref | {"prefix-stops": stops,
                                           "iter-const": ref - stops - paper}
        else:
            against_paper = {"exact-iter": ours - mid, "iter-const": mid - paper}
        causes[key] = (against_ref, against_paper)
    return causes


def delta_causes(p: WalkParams) -> dict[str, tuple[dict[str, int], dict[str, int]]]:
    """For every row key: the signed contribution of each reason code to ours minus dfthcbe's and
    to ours minus the paper's (Table III's attribution), per call; codes with a zero contribution
    are left out, and two different codes may cancel (for example ``ref-unif`` -1 and
    ``ctrl-ref-t2`` +1 in the controlled ``REF_T2`` against the paper).

    Each block is split into parts that the three constructions share, and each part's difference
    goes to the code that names its mechanism: the uniform preparation's comparators (``cmp-free``)
    and rotations (``rot-add``) against dfthcbe, and dfthcbe's uniform-preparation formula against
    the paper's ``4 lg L`` (``unif-odd``); every unary iteration split by
    :func:`iteration_causes` into exact against truncated iteration (``exact-iter``), no
    aligned-block pruning (``no-prune``) and, against the paper, truncated iteration's constants
    (``iter-const``) and, on the tail walks, dfthcbe's stops on the dead SF prefix
    (``prefix-stops``); the inner entry's ``r`` (``flat-r``) and sign bits (``sign-bits``); the
    one-hots' padding (``onehot-pad``) and constants (``onehot-const``); the SF fix-up's enable
    (``sf-enable``); the tail fix-up the paper charges to the inner erase (``attrib``); the spin
    multiplexer (``mux``), the Givens rotations (``rz-addsub``) and the Majorana gate
    (``maj-and``); the reflections' target (``ref-target``), the paper's extra reflected qubit
    (``ref-unif``) and QPE control (``qpe-ctrl``), and our controlled ``REF_T2``
    (``ctrl-ref-t2``). The report checks at every system that each row's contributions sum to its
    deltas exactly."""
    ref, paper = dfthcbe_blocks(p), paper_table3(p)
    s, N, R, B = p.s_u, p.N, p.R, p.B
    Ke1, K4 = 2**p.ke1, 2**p.ke2 * (B + 1)

    def uniform(L: int, adjoint: bool) -> tuple[dict, dict]:
        n = lg(odd_part(L))
        ours_rest = (2 * n - 2 if adjoint else 3 * n - 3) if n else 0
        ours_rot, ref_rot, ref_rest = (2 * (s - 1), 2 * (s - 2), 4 * n - 1) if n else (0, 0, 0)
        against_ref = {"cmp-free": ours_rest - ref_rest, "rot-add": ours_rot - ref_rot}
        return against_ref, against_ref | {"unif-odd": ref_rot + ref_rest - 4 * lg(L)}

    x_prep, x_unprep = uniform(p.X_o, False), uniform(p.X_o, True)
    b_prep, b_unprep = uniform(B + 1, False), uniform(B + 1, True)
    iteration = iteration_causes(p)
    inner_onehot = costs.onehot(p.ke2 + p.w_b)
    causes: dict[str, tuple[dict, dict]] = {}
    causes["outer_read"] = (iteration["outer_read"][0], {})
    causes["inner_read"] = (_add((iteration["inner_read"][0], 1),
                                 ({"flat-r": (p.K2 - 1) * p.w_r}, 1)), {})
    causes["outer_erase"] = (iteration["outer_erase"][0],
                             _add((iteration["outer_erase"][1], 1),
                                  ({"onehot-const": costs.onehot(p.ke1) - Ke1}, 1)))
    pad = {"onehot-pad": inner_onehot - max(K4 - 2, 0)}
    causes["inner_erase"] = (
        _add((iteration["inner_erase"][0], 1), (pad, 1)),
        _add((iteration["inner_erase"][1], 1), (pad, 1),
             ({"onehot-const": max(K4 - 2, 0) - K4}, 1)))
    causes["uniform_x_o"], causes["uniform_x_o_adjoint"] = x_prep, x_unprep
    causes["uniform_b"], causes["uniform_b_adjoint"] = b_prep, b_unprep
    causes["rotation"] = ({"rot-add": 1}, {})
    causes["outer_prepare"] = (_add((x_prep[0], 1), (iteration["outer_read"][0], 1)),
                               _add((x_prep[1], 1), (iteration["outer_read"][1], 1)))
    causes["outer_unprepare"] = (
        _add((x_unprep[0], 1), ({"cmp-free": -p.b_k1}, 1), (causes["outer_erase"][0], 1)),
        _add((x_unprep[1], 1), ({"cmp-free": -p.b_k1}, 1), (causes["outer_erase"][1], 1)))
    causes["inner_prepare"] = (
        _add((b_prep[0], 1), (causes["inner_read"][0], 1)),
        _add((b_prep[1], 1), (iteration["inner_read"][1], 1),
             ({"flat-r": (p.K2 - 1) * p.w_r, "sign-bits": 2 * (p.K2 - 1)}, 1)))
    causes["inner_unprepare"] = (
        _add((b_unprep[0], 1), ({"cmp-free": -p.b_k2}, 1), (causes["inner_erase"][0], 1)),
        _add((b_unprep[1], 1), ({"cmp-free": -p.b_k2}, 1), (causes["inner_erase"][1], 1),
             ({"attrib": -N}, 1)))
    causes["inner_signs"] = ({}, {})
    causes["tail_angle_load"] = iteration["tail"]
    causes["sf_angle_load"] = iteration["sf_load"]
    causes["spin_mux"] = ({"mux": 1 - ref["spin_mux"]}, {"mux": 1 - paper["spin_mux"]})
    causes["spin_swap"] = ({}, {})
    for key in ("givens_rotation", "givens_rotation_adjoint"):
        causes[key] = ({}, {"rz-addsub": 2 * (p.b_rot - 1) - paper[key]})
    for key in ("givens_ladder", "givens_ladder_dagger"):
        causes[key] = ({}, {"rz-addsub": 2 * (p.b_rot - 1) * (N - 1) - paper[key]})
    for key in ("majorana_gate", "majorana_gate_adjoint"):
        causes[key] = ({}, {"maj-and": 5 - paper[key]})
    fixup = {"sf-enable": costs.unary(0, R, p.w_r, True) - costs.unary(0, R, p.w_r, False),
             "onehot-pad": costs.onehot(p.w_b) - (B - 1)}
    causes["angle_erase"] = (
        _add((fixup, 1), (iteration["sf_fixup"][0], 1), (iteration["tail"][0], 1)),
        _add((fixup, 1), ({"onehot-const": -1, "attrib": N}, 1), (iteration["sf_fixup"][1], 1),
             (iteration["tail"][1], 1)))
    select_parts = ("tail_angle_load", "sf_angle_load", "spin_mux", "spin_mux", "spin_swap",
                    "spin_swap", "givens_ladder", "givens_ladder_dagger", "majorana_gate",
                    "angle_erase")
    causes["inner_select"] = tuple(_add(*((causes[k][i], 1) for k in select_parts))
                                   for i in (0, 1))
    causes["inner_select_dagger"] = causes["inner_select"]
    causes["ref_t2"] = ({}, {"ref-unif": -1})
    causes["controlled_ref_t2"] = ({}, {"ref-unif": -1, "ctrl-ref-t2": 1})
    causes["ref_ab"] = ({"ref-target": -1}, {"ref-target": -1, "qpe-ctrl": -1})
    causes["controlled_ref_ab"] = ({}, {"ref-target": -1})
    inner_round = ("inner_prepare", "inner_select", "inner_unprepare", "inner_signs")
    for name, reflection in (("block_encoding", "ref_t2"),
                             ("controlled_block_encoding", "controlled_ref_t2")):
        causes[name] = tuple(_add((causes["outer_prepare"][i], 1),
                                  (causes["outer_unprepare"][i], 1),
                                  *((causes[k][i], 2) for k in inner_round),
                                  (causes[reflection][i], 1)) for i in (0, 1))
    causes["walk"] = tuple(_add((causes["block_encoding"][i], 1), (causes["ref_ab"][i], 1))
                           for i in (0, 1))
    causes["controlled_walk"] = tuple(
        _add((causes["controlled_block_encoding"][i], 1), (causes["controlled_ref_ab"][i], 1))
        for i in (0, 1))
    return {key: (_nonzero(against_ref), _nonzero(against_paper))
            for key, (against_ref, against_paper) in causes.items()}
```

- [ ] **Step 6: Update the wire counts in two docstrings**

Replace `src/dfthc/qpe.py` with (changed: `ControlledWalkPower`'s register table):

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
    ``outer``      the ``outer`` group (71)       walk state            walk state
    ``inner``      the ``inner`` group (65)       walk state            walk state
    ``select``     ``(N - 1) b_rot`` +            ``|0>``               ``|0>``
                   ``junk_overflow`` + 4 (799)
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
    controlled walk, whose query count the paper estimates (PRX 15, 041016; arXiv:2502.15882v1
    Sec. II, after Eq. (11); spec 3.3 and 3.4). The paper's phase-estimation step controls only
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` (App. B.7.2, Eq. (128)), which gives
    :math:`C(W^p)` only for even ``p`` because the block encoding is self-inverse. Each step here
    also controls :math:`\mathrm{REF}_{T_2}` (spec 9.23), so every ``p`` works, round
    :math:`j = 0` included, at ``p`` Toffolis more than that model: at FeMoco-54 ``9795 p``
    against ``9794 p``, and the paper's Table III step is 9,997. dfthcbe builds no phase
    estimation; its controlled walk is ``DFTHCWalk(is_controlled=True)``.

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

Replace `src/dfthc/testing/walk_sim.py` with (changed: the wire counts in its docstrings):

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
68 wires) are correct but about 25 times slower per input (``np.unique`` over two-word keys).
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
"""The Phase 3 test shapes: ``H2_STO3G_TEST`` (56 wires with the control), ``H2_631G_TEST``
(68) and :data:`FAILS_C3` (59), each with its factors."""


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

- [ ] **Step 7: Update the report**

Replace `scripts/resource_report.py` with:

```python
"""The resource report: ``docs/resources.md`` and ``docs/resources.json`` (spec 13.3; master plan
Tasks 4.3 and 4.5).

Usage, from the repository root (``PYTHONPATH=src``)::

    python scripts/resource_report.py [--systems NAME ...] [--large SYSTEM] [--rows [ID ...]]
                                      [--out DIR]

``--rows`` compiles only the rows named (none if it names none; the tests use it); the other rows
show their closed form, marked pending.

Every row is one program, the operator exactly as the walk calls it: the rows are the distinct
dfthc operator calls in the walk and the controlled walk (found by running each composite's pinned
rule in tape mode, nothing compiled), two programs the walk does not call (``PhaseFixup`` with each
erase's fix-up arguments, and ``C(ZeroReflection)`` on each reflection's pattern) and the L4
programs (``ControlledWalkPower`` with p = 1, 2; the QPE program with t = 1, 2, 3 at the H2
systems).
Each program is compiled once, in its own subprocess under ``run_watched``, with
``compiled_resources`` on ``null.qubit``; its counts go into the ``"compiled"`` section of the json
with :func:`~dfthc.testing.provenance.code_hash`, the environment, the compile time, the watched
peak memory and the gate count from the resource functions. An entry of the current code hash is
reused; an entry of another hash stays as history and is never shown as a current measurement.

The H2 systems and ``naphthalene`` compile every program that has no current entry. ``fe2s2``,
``fe4s4`` and FeMoco-54 compile only with ``--large SYSTEM``, run by the orchestrator alone
(master plan 12.9): each program is first projected with
:func:`~dfthc.testing.provenance.project_compile` from its current entries at smaller systems and
compiled only if the projection is within 7.5 GB and 60 min, under the 8 GB cap; a refused or
killed program is recorded with its projection and is not a measurement. At FeMoco-54 only the walk
and the controlled walk compile (Task 4.5); every other FeMoco-54 row is its closed form,
validated by the compiles at the other systems and by the cross-check.

The script exits with status 1 after writing both files if a check fails: a measured T, M or A
that differs from its closed form, a nonzero delta against dfthcbe or the paper without a reason
code, or, wherever the walks are measured, a cross-check (the sum over the partition rows of calls
per walk times the closed form, plus in M the block encoding's own junk measurements, against the
measured walk and controlled walk). The walk's unprepares take the junk outcomes the block
encoding measured (spec 16, Q-1 variant (b)): their programs pass constant outcomes, and the
``PhaseFixup`` rows take their erase arguments; the walk calls no ``QROAMErase``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pennylane as qp  # noqa: E402
from pennylane.decomposition import DecompositionGraph  # noqa: E402
from pennylane.typing import AbstractArray  # noqa: E402

from dfthc import costs, costs_reference  # noqa: E402
from dfthc.compile import (  # noqa: E402
    FIXED_DECOMPS,
    GATE_SET,
    compiled_resources,
    surviving_dfthc_operators,
)
from dfthc.data.factors import load_factors, random_factors  # noqa: E402
from dfthc.data.tables import WalkTables  # noqa: E402
from dfthc.ops.prepare import InnerPrepare, OuterPrepare  # noqa: E402
from dfthc.ops.qroam import PhaseFixup, QROAMErase, QROAMRead  # noqa: E402
from dfthc.ops.reflections import ZeroReflection  # noqa: E402
from dfthc.ops.walk import ControlledDFTHCWalk, DFTHCWalk  # noqa: E402
from dfthc.params import FEMOCO54, H2_STO3G_TEST, WalkParams, lg, odd_part  # noqa: E402
from dfthc.qpe import ControlledWalkPower, qpe_program  # noqa: E402
from dfthc.registers import WalkLayout  # noqa: E402
from dfthc.testing.provenance import code_hash, environment, project_compile  # noqa: E402
from dfthc.testing.watchdog import MEMORY_CAP_MB, run_watched  # noqa: E402

SMALL = ("h2_sto3g_R2_B2_C1", "h2_631g_R2_B2_C2", "h2_631g_R2_B3_C3", "naphthalene")
LARGE = ("fe2s2", "fe4s4", "femoco54")
SYSTEMS = SMALL + LARGE
"""The report's systems (spec 13.3 step 1), smallest first."""
TEST_SYSTEMS = ("H2_STO3G_TEST",)
"""Accepted by ``--systems`` for the tests; never part of the default report."""
QPE_SYSTEMS = ("h2_sto3g_R2_B2_C1", "h2_631g_R2_B2_C2", "h2_631g_R2_B3_C3", "H2_STO3G_TEST")
"""Where the QPE program is compiled (spec 11.3, 13.3 step 2)."""
FEMOCO_PROGRAMS = ("DFTHCWalk", "ControlledDFTHCWalk")
"""The only programs compiled at FeMoco-54 (spec 13.3 step 2; Task 4.5)."""
FEMOCO_SEED = 0
"""``random_factors(54, 10, 27, 27, n_plus=27, seed=0)``, as the work-pool test uses."""
QPE_BITS = (1, 2, 3)
POWERS = (1, 2)
SMALL_CAP_MB, SMALL_TIMEOUT_S = 4096, 900
LARGE_CAP_MB, LARGE_TIMEOUT_S = MEMORY_CAP_MB, 3600
JSON_NAME, MD_NAME = "resources.json", "resources.md"
PREFIX = "RESULT "
CLIFFORDS = ("PauliX", "PauliY", "PauliZ", "Hadamard", "S", "Adjoint(S)", "CNOT", "CZ", "SWAP",
             "GlobalPhase")

REASON_CODES = {
    "flat-r": "r is carried through the inner swap; heterogeneous targets are deferred",
    "no-prune": "no aligned-block pruning of unary-iteration boundaries: dfthcbe stops once on "
                "each aligned block of a dead interval, and a walk over a sub-range [lo, hi) can "
                "be truncated only that way (costs_reference.iteration_counts)",
    "exact-iter": "exact unary iteration: out-of-range addresses raise no flag, which costs at "
                  "most one Toffoli per address bit at each range end; dfthcbe's truncated "
                  "iteration is cheaper (costs_reference.iteration_counts)",
    "iter-const": "truncated iteration over L leaves costs L + c - 2 Toffolis with c controls; the "
                  "paper counts L, one per leaf",
    "prefix-stops": "dfthcbe's walks over the tail rows [RC, X_o) (the tail load and the angle "
                    "erase's tail fix-up) stop once on each aligned block of the dead SF prefix "
                    "[0, RC), s(RC) stops; the paper's (B41, and B50's fix-up N) count their N "
                    "leaves alone (costs_reference.iteration_causes)",
    "onehot-pad": "the fixup one-hot is padded to a power of two",
    "adder-const": "SemiAdder constants differ from Qualtran's",
    "mux": "spin-multiplexer construction",
    "attrib": "the paper attributes the cost to a different step",
    "rz-addsub": "each Givens R_z is Qualtran's ControlledAddOrSubtract, b_rot - 1 Toffolis, as in "
                 "dfthcbe; the paper counts b_rot",
    "cmp-free": "PennyLane's comparators uncompute by measurement, so their adjoints cost no "
                "Toffoli",
    "maj-and": "the Majorana gate folds the success flags into two ANDs with G0 (2 + 2 + 1); the "
               "paper's Fig. 4 controls its four gates on one AND of both flags (1 + 2 + 4)",
    "sf-enable": "the SF fix-up of AngleErase is enabled on G0, one Toffoli more than dfthcbe's",
    "rot-add": "PhaseGradientRotation adds with one m-bit SemiAdder, m - 1 Toffolis; Qualtran's "
               "ZPowConstViaPhaseGradient (dfthcbe's rotation) costs m - 2",
    "unif-odd": "dfthcbe's uniform preparation, 4 n + 2 s_u - 5 on the n = lg l qubits of the odd "
                "part l of L (as ours), against the paper's 4 lg L over all lg L qubits",
    "sign-bits": "the inner entry carries both sign bits, sgn and d_sgn; Table III evaluates its "
                 "inner swaps with b_2 = w_b + b_k2 bits",
    "onehot-const": "a one-hot over K values costs K - 2 Toffolis; the paper's B50 and B53 count K "
                    "and its B47 counts B for the B + 1 values of b",
    "ref-target": "the reflection's multi-controlled Z targets one of its n_w pattern qubits "
                  "(n_w - 2 Toffolis); dfthcbe's uncontrolled REF_{A u B} and the paper's B55 add "
                  "a phase target (n_w - 1)",
    "ref-unif": "the paper's REF_T2 (B11) also reflects the rotated qubit of its Unif(b), which "
                "the flagged uniform preparation does not have",
    "qpe-ctrl": "the paper's B55 books one Toffoli for the unary iteration on the QPE control in "
                "the walk (App. B.7.2, Eq. (128)); our walk has no QPE control",
    "ctrl-ref-t2": "the controlled walk also controls REF_T2 (+1 per step), so C(W)^p is exact for "
                   "every p; the paper's QPE step controls only REF_{A u B} (App. B.7.2, "
                   "Eq. (128))",
}
"""Spec 13.3's reason codes, with the twelve this task adds (master plan Task 4.3). A row lists
the codes whose contribution to its delta is nonzero (``costs_reference.delta_causes``)."""

LEAVES = frozenset({"QROAMRead", "QROAMErase", "PhaseFixup", "AngleErase", "ZeroReflection",
                    "SpinMux", "SpinSwap", "MajoranaGate", "InnerSigns", "PhaseGradientRotation",
                    "GivensRotation"})
"""Operators whose rules call no dfthc operator; their rules are not run to find rows."""
COMPOSITES = frozenset({"DFTHCWalk", "ControlledDFTHCWalk", "DFTHCBlockEncoding",
                        "ControlledDFTHCBlockEncoding", "InnerSelect", "InnerSelectDagger"})
"""The partition rows are the children of these that are not in :data:`INNER_COMPOSITES`."""
INNER_COMPOSITES = frozenset({"DFTHCBlockEncoding", "ControlledDFTHCBlockEncoding",
                              "InnerSelect", "InnerSelectDagger"})
LEVELS = {"L1": {"PhaseGradientRotation", "FlaggedUniformPrep", "Adjoint(FlaggedUniformPrep)",
                 "QROAMRead", "PhaseFixup", "QROAMErase", "ZeroReflection", "C(ZeroReflection)"},
          "L3": {"RefT2", "RefAB", "ControlledRefT2", "ControlledRefAB", "DFTHCBlockEncoding",
                 "ControlledDFTHCBlockEncoding", "DFTHCWalk", "ControlledDFTHCWalk"},
          "L4": {"ControlledWalkPower", "QPE"}}
FROM_OUTCOMES = frozenset({"OuterUnprepareFromOutcomes", "InnerUnprepareFromOutcomes"})
"""The unprepares the walk calls (spec 16, Q-1 variant (b)): they take the junk outcomes the
block encoding measured, so their programs pass constant outcomes, and they emit the erase's
fix-up themselves, so the ``PhaseFixup`` rows take their erase arguments."""
NAME_KEYS = {
    "OuterPrepare": "outer_prepare", "OuterUnprepareFromOutcomes": "outer_unprepare",
    "InnerPrepare": "inner_prepare", "InnerUnprepareFromOutcomes": "inner_unprepare",
    "InnerSigns": "inner_signs", "TailAngleLoad": "tail_angle_load",
    "SFAngleLoad": "sf_angle_load", "SpinMux": "spin_mux", "SpinSwap": "spin_swap",
    "GivensRotation": "givens_rotation", "Adjoint(GivensRotation)": "givens_rotation_adjoint",
    "GivensLadder": "givens_ladder", "GivensLadderDagger": "givens_ladder_dagger",
    "MajoranaGate": "majorana_gate", "Adjoint(MajoranaGate)": "majorana_gate_adjoint",
    "AngleErase": "angle_erase", "InnerSelect": "inner_select",
    "InnerSelectDagger": "inner_select_dagger", "RefT2": "ref_t2",
    "ControlledRefT2": "controlled_ref_t2", "RefAB": "ref_ab",
    "ControlledRefAB": "controlled_ref_ab",
    "DFTHCBlockEncoding": "block_encoding",
    "ControlledDFTHCBlockEncoding": "controlled_block_encoding", "DFTHCWalk": "walk",
    "ControlledDFTHCWalk": "controlled_walk",
}
"""The ``costs_reference`` key of every row whose class appears with one signature."""
PARENT_KEYS = {
    ("FlaggedUniformPrep", "OuterPrepare"): "uniform_x_o",
    ("FlaggedUniformPrep", "InnerPrepare"): "uniform_b",
    ("Adjoint(FlaggedUniformPrep)", "OuterUnprepareFromOutcomes"): "uniform_x_o_adjoint",
    ("Adjoint(FlaggedUniformPrep)", "InnerUnprepareFromOutcomes"): "uniform_b_adjoint",
    ("PhaseGradientRotation", "FlaggedUniformPrep"): "rotation",
    ("QROAMRead", "OuterPrepare"): "outer_read", ("QROAMRead", "InnerPrepare"): "inner_read",
    ("QROAMRead", "TailAngleLoad"): "tail_angle_load",
    ("QROAMRead", "SFAngleLoad"): "sf_angle_load",
    ("PhaseFixup", "OuterUnprepareFromOutcomes"): "outer_erase",
    ("PhaseFixup", "InnerUnprepareFromOutcomes"): "inner_erase",
    ("ZeroReflection", "RefT2"): "ref_t2", ("ZeroReflection", "RefAB"): "ref_ab",
    ("C(ZeroReflection)", "RefT2"): "controlled_ref_t2",
    ("C(ZeroReflection)", "RefAB"): "controlled_ref_ab",
}
"""The ``costs_reference`` key of the L1 rows, by class and parent class."""
STEPS = {"OuterPrepare": "1-4", "InnerPrepare": "5-8", "InnerSigns": "9", "InnerSelect": "10-18",
         "TailAngleLoad": "10", "SFAngleLoad": "11", "SpinMux": "12, 17", "SpinSwap": "13, 17",
         "GivensLadder": "14", "GivensRotation": "14", "MajoranaGate": "15",
         "Adjoint(MajoranaGate)": "15", "GivensLadderDagger": "16",
         "Adjoint(GivensRotation)": "16", "AngleErase": "18", "InnerSelectDagger": "10-18",
         "InnerUnprepareFromOutcomes": "19-22", "RefT2": "23", "ControlledRefT2": "23",
         "OuterUnprepareFromOutcomes": "24-27", "RefAB": "28", "ControlledRefAB": "28",
         "DFTHCBlockEncoding": "1-27", "ControlledDFTHCBlockEncoding": "1-27",
         "DFTHCWalk": "1-28", "ControlledDFTHCWalk": "1-28", "ControlledWalkPower": "QPE",
         "QPE": "QPE"}
PARENT_STEPS = {("FlaggedUniformPrep", "OuterPrepare"): "1",
                ("FlaggedUniformPrep", "InnerPrepare"): "5",
                ("Adjoint(FlaggedUniformPrep)", "OuterUnprepareFromOutcomes"): "27",
                ("Adjoint(FlaggedUniformPrep)", "InnerUnprepareFromOutcomes"): "22",
                ("PhaseGradientRotation", "FlaggedUniformPrep"): "1, 5",
                ("QROAMRead", "OuterPrepare"): "2", ("QROAMRead", "InnerPrepare"): "6",
                ("QROAMRead", "TailAngleLoad"): "10", ("QROAMRead", "SFAngleLoad"): "11",
                ("PhaseFixup", "OuterUnprepareFromOutcomes"): "26",
                ("PhaseFixup", "InnerUnprepareFromOutcomes"): "21",
                ("ZeroReflection", "RefT2"): "23", ("ZeroReflection", "RefAB"): "28",
                ("C(ZeroReflection)", "RefT2"): "23", ("C(ZeroReflection)", "RefAB"): "28"}


@dataclasses.dataclass
class System:
    """One report system: its parameters, tables and layout."""

    name: str
    params: WalkParams
    tables: WalkTables
    layout: WalkLayout


@dataclasses.dataclass
class Row:
    """One program of the report at one system."""

    id: str
    name: str
    parent: str | None
    level: str
    steps: str
    calls: int
    controlled_calls: int
    partition: bool
    closed: costs.Cost
    uncomputes: int
    """The closed form of A, the measured AND uncomputes (:func:`closed_form`)."""
    wires: int
    program: Callable[[], tuple[Callable[[], None], int]]
    key: str | None
    descendants: frozenset[str]
    op: object = None
    size_op: Callable[[], object] | None = None
    """Builds, in tape mode, the operator whose resource functions give the row's gates."""


def build_system(name: str) -> System:
    """FeMoco-54 from ``random_factors``; the test preset; a vendored system at the report's
    default precisions (``costs_reference.default_params``)."""
    if name == "femoco54":
        p, factors = FEMOCO54, random_factors(54, 10, 27, 27, n_plus=27, seed=FEMOCO_SEED)
    elif name == "H2_STO3G_TEST":
        p, factors = H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")
    elif name in SYSTEMS:
        factors = load_factors(name)
        p = costs_reference.default_params(factors.N, factors.R, factors.B, factors.C,
                                           factors.n_plus)
    else:
        raise ValueError(f"unknown system {name!r}; expected one of {SYSTEMS + TEST_SYSTEMS}")
    return System(name, p, WalkTables.from_factors(factors, p), WalkLayout.from_params(p))


def _groups(s: System) -> dict:
    return {name: getattr(s.layout, name) for name in DFTHCWalk.wire_argnames}


def _walk_op(s: System, controlled: bool):
    p, t = s.params.as_compilable(), s.tables.as_compilable()
    if controlled:
        return ControlledDFTHCWalk(p, t, [s.layout.num_wires], **_groups(s))
    return DFTHCWalk(p, t, **_groups(s))


def walk_program(s: System, controlled: bool):
    """The walk or controlled walk with its phase gradient: the programs of Task 3.4."""

    def qfunc():
        qp.PhaseGradientStatePrep(s.layout.phase_grad)
        _walk_op(s, controlled)

    return qfunc, s.layout.num_wires + int(controlled)


def _power_op(s: System, power: int):
    groups = {name: getattr(s.layout, name) for name in ControlledWalkPower.wire_argnames[1:]}
    return ControlledWalkPower(power, s.params.as_compilable(), s.tables.as_compilable(),
                               [s.layout.num_wires], **groups)


def power_program(s: System, power: int):
    """``ControlledWalkPower`` on ``WalkLayout``'s groups and one control wire after them (the
    phase4-qpe plan's interface)."""
    groups = {name: getattr(s.layout, name) for name in ControlledWalkPower.wire_argnames[1:]}

    def qfunc():
        qp.PhaseGradientStatePrep(s.layout.phase_grad)
        ControlledWalkPower(power, s.params.as_compilable(), s.tables.as_compilable(),
                            [s.layout.num_wires], **groups)

    return qfunc, s.layout.num_wires + 1


def _children(op) -> list:
    """The dfthc operators the pinned rule of ``op`` calls (tape mode, nothing compiled)."""
    if op.name in LEAVES or op.name not in FIXED_DECOMPS:
        return []
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        FIXED_DECOMPS[op.name](**op.arguments)
    return [child for child in queue.queue if surviving_dfthc_operators([child.name])]


def _target(op):
    return op.base if op.name.startswith("Adjoint(") else op


def _signature(op) -> tuple:
    target = _target(op)
    return (op.name, tuple(sorted(target.compilable_args.items())),
            tuple((name, len(getattr(target, name))) for name in target.wire_argnames))


def _rebuilt(op, index: dict):
    """``op`` constructed again on the wires ``index`` maps its wires to (under capture this
    queues it); an unprepare from outcomes gets constant junk outcomes, a bool array, since its
    program has no junk measurement (its counts do not depend on the outcomes)."""
    if op.name.startswith("Adjoint("):
        return qp.adjoint(_rebuilt(op.base, index))
    arguments = dict(op.arguments)
    for name in op.wire_argnames:
        arguments[name] = [index[wire] for wire in getattr(op, name)]
    if op.name in FROM_OUTCOMES:
        arguments["junk_outcomes"] = np.zeros(len(arguments["junk_outcomes"]), dtype=bool)
    return type(op)(**arguments)


def _call_program(op):
    def program():
        index = {wire: i for i, wire in enumerate(op.wires)}
        return (lambda: _rebuilt(op, index)), len(op.wires)

    return program


def _erase_arguments(op) -> dict:
    """The erase an unprepare from outcomes performs: its read's ``address_widths``,
    ``live_ranges``, ``width`` and ``block_bits`` (no enable) and the fix-up's ``onehot_bits``
    (spec 9.7, 9.8)."""
    a = op.arguments
    if op.name == "OuterUnprepareFromOutcomes":
        shape, onehot_bits = OuterPrepare.read_shape(a["X_o"], a["b_k1"], a["k1"]), a["ke1"]
    else:
        shape = InnerPrepare.read_shape(a["N"], a["R"], a["B"], a["C"], a["b_k2"], a["k2"])
        onehot_bits = a["ke2"] + lg(a["B"] + 1)
    return shape | {"control_value": None, "onehot_bits": onehot_bits}


def _fixup_program(a: dict):
    """``PhaseFixup`` with an erase's fix-up arguments ``a`` (:func:`_erase_arguments`): its
    covered set, one-hot bits, enable and work, on a one-hot of ``2**onehot_bits`` wires; the
    program's parities are all ``False``, and the size operator (tape mode) takes an abstract
    parity array. Returns ``(program, size_op, wires)``."""
    covered = QROAMRead.loaded_ranges(a["address_widths"], a["live_ranges"], a["block_bits"])
    sizes = PhaseFixup.required_wires(a["address_widths"], covered, a["onehot_bits"],
                                      a["control_value"])
    size = len(PhaseFixup.address_order(a["address_widths"], covered))
    wires, start = {}, 0
    for name in ("address", "onehot", "work", "control"):
        wires[name] = list(range(start, start + sizes[name]))
        start += sizes[name]

    def build(parities):
        return PhaseFixup(parities, a["address_widths"], covered, a["onehot_bits"],
                          a["control_value"], **wires)

    def program():
        return (lambda: build(np.zeros(size, dtype=bool))), start

    return program, lambda: build(AbstractArray((size,), bool)), start


def _controlled_reflection_program(reflection):
    """``qp.ctrl`` of the reflection's ``ZeroReflection`` with one more work wire. Returns
    ``(program, size_op, wires)``."""
    zero, one = len(reflection.zero_wires), len(reflection.one_wires)
    work = ZeroReflection.required_wires(zero, one, True)["work"]
    n = zero + one + work + 1
    wires = list(range(n))

    def build():
        return qp.ctrl(ZeroReflection(wires[:zero], wires[zero:zero + one],
                                      wires[zero + one:n - 1]), control=wires[n - 1])

    return (lambda: (build, n)), build, n


def _dirty(L: int) -> int:
    """ANDs a flagged uniform preparation of ``L`` leaves for its adjoint to uncompute: its
    classical comparator's ``n - 1`` (``n = lg l``; none if ``l = 1``, spec 9.24)."""
    n = lg(odd_part(L))
    return n - 1 if n else 0


def closed_form(name: str, op, p: WalkParams) -> tuple[costs.Cost, int]:
    """The closed form (``dfthc.costs``) of one call: ``(Cost(T, M), A)``; for ``PhaseFixup``,
    ``op`` is the erase's arguments (:func:`_erase_arguments`). A counts the measured AND
    uncomputes inside the call: a prepare leaves its comparators' ANDs (``b_k`` and the uniform
    preparation's :func:`_dirty`) for its unprepare to measure (spec 9.24, 10.2), so per call A
    is not ``T`` minus the plain ``Toffoli`` gates; over the walk it is
    (``measured_and_uncomputes``)."""
    a = op if isinstance(op, dict) else _target(op).arguments if op is not None else {}
    if name == "PhaseGradientRotation":
        cost = costs.phase_gradient_rotation(a["k"], a["m"])
        return cost, cost.toffoli
    if name == "FlaggedUniformPrep":
        cost = costs.flagged_uniform_prep(a["L"], a["s"])
        return cost, cost.toffoli - _dirty(a["L"])
    if name == "Adjoint(FlaggedUniformPrep)":
        cost = costs.flagged_uniform_unprep(a["L"], a["s"])
        return cost, cost.toffoli + _dirty(a["L"])
    if name == "QROAMRead":
        ranges = QROAMRead.visited_ranges(a["address_widths"], a["live_ranges"], a["block_bits"])
        cost = costs.qroam_read(ranges, a["block_bits"], a["width"],
                                a["control_value"] is not None)
        return cost, cost.toffoli - (2**a["block_bits"] - 1) * a["width"]
    if name == "PhaseFixup":
        ranges = QROAMErase.fixup_high_ranges(a["address_widths"], a["live_ranges"],
                                              a["block_bits"], a["onehot_bits"])
        cost = costs.phase_fixup(ranges, a["onehot_bits"], a["control_value"] is not None)
        return cost, cost.toffoli
    if name in ("ZeroReflection", "C(ZeroReflection)"):
        n_w = len(op.zero_wires) + len(op.one_wires)
        controlled = name.startswith("C(")
        cost = costs.zero_reflection(n_w, controlled)
        return cost, cost.toffoli - int(n_w >= (2 if controlled else 3))
    walk_a = costs.measured_and_uncomputes(p, False)
    table = {
        "OuterPrepare": (costs.outer_prepare(p),
                         -_dirty(p.X_o) - (p.K1 - 1) * p.d1 - p.b_k1 - p.w_x),
        "OuterUnprepareFromOutcomes": (costs.outer_unprepare_from_outcomes(p),
                                       -p.w_x + p.b_k1 + _dirty(p.X_o)),
        "InnerPrepare": (costs.inner_prepare(p),
                         -_dirty(p.B + 1) - (p.K2 - 1) * p.d2 - p.b_k2 - p.w_b),
        "InnerUnprepareFromOutcomes": (costs.inner_unprepare_from_outcomes(p),
                                       -p.w_b + p.b_k2 + _dirty(p.B + 1)),
        "InnerSigns": (costs.inner_signs(), 0),
        "TailAngleLoad": (costs.tail_angle_load(p), 0),
        "SFAngleLoad": (costs.sf_angle_load(p), 0),
        "SpinMux": (costs.spin_mux(), -1), "SpinSwap": (costs.spin_swap(p), -p.N),
        "GivensRotation": (costs.givens_rotation(p), 0),
        "Adjoint(GivensRotation)": (costs.givens_rotation(p), 0),
        "GivensLadder": (costs.givens_ladder(p), 0),
        "GivensLadderDagger": (costs.givens_ladder(p), 0),
        "MajoranaGate": (costs.majorana_gate(), -3),
        "Adjoint(MajoranaGate)": (costs.majorana_gate(), -3),
        "AngleErase": (costs.angle_erase(p), 0),
        "InnerSelect": (costs.inner_select(p), -(2 * p.N + 5)),
        "InnerSelectDagger": (costs.inner_select(p), -(2 * p.N + 5)),
        "RefT2": (costs.ref_t2(p, False), -1), "ControlledRefT2": (costs.ref_t2(p, True), -1),
        "RefAB": (costs.ref_ab(p, False), -1), "ControlledRefAB": (costs.ref_ab(p, True), -1),
        "DFTHCBlockEncoding": (costs.block_encoding(p, False), 1 - costs.toffoli_only(p)),
        "ControlledDFTHCBlockEncoding": (costs.block_encoding(p, True),
                                         1 - costs.toffoli_only(p)),
        "DFTHCWalk": (costs.walk(p, False), walk_a - costs.walk(p, False).toffoli),
        "ControlledDFTHCWalk": (costs.walk(p, True), costs.measured_and_uncomputes(p, True)
                                - costs.walk(p, True).toffoli),
    }
    cost, offset = table[name]
    return cost, cost.toffoli + offset


def _level(name: str) -> str:
    return next((level for level, names in LEVELS.items() if name in names), "L2")


def build_rows(s: System) -> list[Row]:
    """Every row of ``s``: the distinct calls in the walk, then the controlled walk's own, in the
    order of first appearance; then the two programs the walk does not call, and L4. A row's id
    is its class name if the class appears with one signature, else ``name@parent id``, with
    ``#2``, ``#3`` for further calls of the class under the same parent."""
    p = s.params
    with qp.capture.pause(), qp.queuing.AnnotatedQueue():
        roots = [_walk_op(s, False), _walk_op(s, True)]
    order, calls, first, below = [], [{}, {}], {}, {}
    for tree, root in enumerate(roots):
        def visit(op, parent, tree=tree):
            sig = _signature(op)
            if sig not in first:
                first[sig] = (op, parent)
                order.append(sig)
            calls[tree][sig] = calls[tree].get(sig, 0) + 1
            names = {op.name}
            for child in _children(op):
                names |= visit(child, op)
            below[sig] = frozenset(names)
            return names

        visit(root, None)
    signatures_of = {}
    for sig in order:
        signatures_of[sig[0]] = signatures_of.get(sig[0], 0) + 1
    ids, used, rows = {}, {}, []
    for sig in order:
        op, parent = first[sig]
        base = sig[0] if signatures_of[sig[0]] == 1 else f"{sig[0]}@{ids[_signature(parent)]}"
        used[base] = used.get(base, 0) + 1
        ids[sig] = base if used[base] == 1 else f"{base}#{used[base]}"
        parent_name = parent.name if parent is not None else None
        closed, uncomputes = closed_form(op.name, op, p)
        rows.append(Row(
            id=ids[sig], name=op.name, parent=parent_name, level=_level(op.name),
            steps=PARENT_STEPS.get((op.name, parent_name), STEPS.get(op.name, "")),
            calls=calls[0].get(sig, 0), controlled_calls=calls[1].get(sig, 0),
            partition=parent_name in COMPOSITES and op.name not in INNER_COMPOSITES,
            closed=closed, uncomputes=uncomputes, wires=len(op.wires),
            program=(_walk_rows_program(s, op.name) if op.name in FEMOCO_PROGRAMS
                     else _call_program(op)),
            key=PARENT_KEYS.get((op.name, parent_name), NAME_KEYS.get(op.name)),
            descendants=below[sig], op=op, size_op=lambda op=op: op))
    rows += _extra_rows(s, rows)
    return rows


def _walk_rows_program(s: System, name: str):
    return lambda: walk_program(s, name == "ControlledDFTHCWalk")


def _extra_rows(s: System, rows: list[Row]) -> list[Row]:
    p, extra = s.params, []
    for row in rows:
        if row.name in FROM_OUTCOMES:
            erase = _erase_arguments(row.op)
            program, size_op, wires = _fixup_program(erase)
            closed, uncomputes = closed_form("PhaseFixup", erase, p)
            extra.append(dataclasses.replace(
                row, id=f"PhaseFixup@{row.id}", name="PhaseFixup", parent=row.name, level="L1",
                steps=PARENT_STEPS[("PhaseFixup", row.name)], calls=0, controlled_calls=0,
                partition=False, closed=closed, uncomputes=uncomputes, wires=wires, program=program,
                key=PARENT_KEYS[("PhaseFixup", row.name)], descendants=frozenset({
                    "PhaseFixup"}), size_op=size_op))
        if row.name == "ZeroReflection":
            program, size_op, wires = _controlled_reflection_program(row.op)
            closed, uncomputes = closed_form("C(ZeroReflection)", row.op, p)
            extra.append(dataclasses.replace(
                row, id=f"C(ZeroReflection)@{row.parent}", name="C(ZeroReflection)",
                steps=PARENT_STEPS[("C(ZeroReflection)", row.parent)], calls=0,
                controlled_calls=0, partition=False, closed=closed, uncomputes=uncomputes,
                wires=wires, program=program, key=PARENT_KEYS[("C(ZeroReflection)", row.parent)],
                descendants=frozenset({"C(ZeroReflection)"}), size_op=size_op))
    walk = next(row for row in rows if row.name == "ControlledDFTHCWalk")
    for power in POWERS:
        extra.append(Row(
            id=f"ControlledWalkPower(p={power})", name="ControlledWalkPower", parent=None,
            level="L4", steps=STEPS["ControlledWalkPower"], calls=0, controlled_calls=0,
            partition=False, closed=costs.controlled_walk_power(p, power),
            uncomputes=power * costs.measured_and_uncomputes(p, True),
            wires=s.layout.num_wires + 1,
            program=(lambda power=power: power_program(s, power)), key="controlled_walk_power",
            descendants=walk.descendants | {"ControlledWalkPower"},
            size_op=lambda power=power: _power_op(s, power)))
    if s.name in QPE_SYSTEMS or s.name == "femoco54":
        for bits in QPE_BITS:
            extra.append(Row(
                id=f"QPE(t={bits})", name="QPE", parent=None, level="L4", steps=STEPS["QPE"],
                calls=0, controlled_calls=0, partition=False,
                closed=costs.iterative_qpe(p, bits),
                uncomputes=(2**bits - 1) * costs.measured_and_uncomputes(p, True)
                + (bits**3 - bits) // 6, wires=qpe_wires(s, bits),
                program=(lambda bits=bits: qpe_program(s.params, s.tables, bits)), key=None,
                descendants=walk.descendants | {"ControlledWalkPower", "QPERound"}))
    return extra


def qpe_wires(s: System, bits: int) -> int:
    """The QPE program's wires without building it (the phase4-qpe plan's interface)."""
    from dfthc.qpe import spare_wires_needed

    sizes = tuple(len(getattr(s.layout, name)) for name in ControlledWalkPower.wire_argnames[1:])
    return (WalkLayout.from_params(s.params, qpe_bits=bits).num_wires
            + spare_wires_needed("ControlledWalkPower", sizes, bits))


def clifford_kind(row: Row) -> str:
    """How the row's Clifford total is to be read (spec 7.6): reads and resets are unhinted
    conditionals, counted at their maximum branch; erase fix-ups are hinted, counted at their
    expectation."""
    names = row.descendants
    unhinted = names & ({"QROAMRead", "QROAMErase", "AngleErase", "QPERound"} | FROM_OUTCOMES)
    hinted = names & ({"QROAMErase", "AngleErase", "PhaseFixup"} | FROM_OUTCOMES)
    parts = (["upper bound (CNOT, PauliX)"] if unhinted else []) + (
        ["expected, hinted (CZ, PauliZ)"] if hinted else [])
    return "; ".join(parts) or "exact"


def gates(rows: list[Row]) -> dict[str, int]:
    """Each row's emitted gates from the resource functions of its size operator (no compile):
    the size measure of the projection. One decomposition graph serves all rows; the QPE rows,
    which the large systems never compile, have none."""
    with qp.capture.pause(), qp.queuing.AnnotatedQueue():
        ops = {row.id: row.size_op() for row in rows if row.size_op is not None}
    solution = DecompositionGraph(operations=list(ops.values()), gate_set=GATE_SET,
                                  fixed_decomps=FIXED_DECOMPS).solve()
    return {row_id: int(solution.resource_estimate(op).num_gates) for row_id, op in ops.items()}


def compile_counts(s: System, row: Row) -> dict:
    """Compile one row's program in this process; its counts and compile time."""
    qfunc, num_wires = row.program()
    start = time.perf_counter()
    resources = compiled_resources(qfunc, num_wires)
    return {"toffoli": resources.toffoli, "measurements": resources.measurements,
            "measured_and_uncomputes": resources.measured_and_uncomputes,
            "counts": resources.counts, "num_wires": num_wires,
            "seconds": round(time.perf_counter() - start, 2)}


def _find_row(s: System, row_id: str) -> Row:
    if row_id in FEMOCO_PROGRAMS:
        controlled = row_id == "ControlledDFTHCWalk"
        closed, uncomputes = closed_form(row_id, None, s.params)
        return Row(id=row_id, name=row_id, parent=None, level="L3", steps=STEPS[row_id],
                   calls=1 - controlled, controlled_calls=int(controlled), partition=False,
                   closed=closed, uncomputes=uncomputes, wires=s.layout.num_wires + controlled,
                   program=_walk_rows_program(s, row_id), key=NAME_KEYS[row_id],
                   descendants=frozenset())
    return next(row for row in build_rows(s) if row.id == row_id)


def child_command(s: System, row: Row) -> list[str]:
    """The subprocess that compiles one row (``--compile-one``)."""
    return [sys.executable, str(Path(__file__).resolve()), "--compile-one", s.name, row.id]


def measure(s: System, row: Row, hash_: str, cap_mb: float, timeout_s: float) -> dict:
    """Compile one row in a subprocess under ``run_watched``; the cache entry. A killed or
    failed compile is recorded with its status and is never a measurement."""
    command = child_command(s, row)
    env = os.environ | {"PYTHONPATH": str(REPO / "src"),
                        "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "0"}
    run = run_watched(command, cap_mb=cap_mb, timeout_s=timeout_s, cwd=REPO, env=env)
    lines = [line for line in run.stdout.splitlines() if line.startswith(PREFIX)]
    summary = run.summary()
    summary["command"] = summary["command"][-2:]
    entry = {"code_hash": hash_, "environment": environment(), "watchdog": summary}
    if run.killed or run.survivors:
        return entry | {"status": "killed", "reason": run.killed}
    if run.returncode != 0 or len(lines) != 1:
        return entry | {"status": "failed", "reason": run.stderr[-2000:]}
    return entry | {"status": "compiled"} | json.loads(lines[0][len(PREFIX):])


def current(cache: dict, key: str, hash_: str) -> dict | None:
    """The newest compiled entry of ``key`` with code hash ``hash_``."""
    entries = [e for e in cache.get(key, []) if e["code_hash"] == hash_
               and e["status"] == "compiled"]
    return entries[-1] if entries else None


def family(row: Row) -> str:
    """The calibration family of a row: its class and comparison key, which name the same block
    at every system although a row's id can differ between systems (``PhaseGradientRotation`` has
    one call shape at the H2 systems and two elsewhere); an L4 row's id."""
    return row.id if row.level == "L4" else f"{row.name}:{row.key or row.id}"


def calibration(cache: dict, family_: str, hash_: str) -> list[tuple]:
    """``(gates, seconds, peak_mb)`` of every current entry of the family, at every system."""
    points = []
    for key, entries in cache.items():
        entry = current({key: entries}, key, hash_)
        if entry and entry.get("family") == family_ and entry.get("gates"):
            points.append((entry["gates"], entry["watchdog"]["seconds"],
                           entry["watchdog"]["peak_mb"]))
    return points


def projection(cache: dict, row: Row, hash_: str, size: int | None) -> dict:
    """The row's projection to ``size`` gates from the current entries of its family at every
    system, earlier large systems included (:func:`~dfthc.testing.provenance.project_compile`
    under the 8 GB cap: the family's largest measurement if some entry has at least ``size``
    gates, else extrapolated)."""
    if size is None:
        return {"allowed": False, "reason": "no size measure", "gates": None, "points": []}
    return project_compile(calibration(cache, family(row), hash_), size, cap_mb=LARGE_CAP_MB)


def compiled_here(s: System, row: Row) -> bool:
    """Whether the report compiles this row at this system at all (spec 13.3 step 2)."""
    if s.name == "femoco54":
        return row.id in FEMOCO_PROGRAMS
    return not (row.name == "QPE" and s.name not in QPE_SYSTEMS)


def run_system(s: System, rows: list[Row], cache: dict, hash_: str, *, projected: bool,
               only: set[str] | None) -> list[str]:
    """Compile the rows of ``s`` that this run compiles and record them in ``cache``; the log.

    Rows with a current entry are reused. ``projected`` (``--large``) first projects each row
    from its current entries (:func:`projection`) and compiles it under the 8 GB cap only if the
    projection allows; a refused row is recorded with its projection. Under ``projected`` the
    first ``killed`` or ``failed`` row stops the system's run, and so does the first ``refused``
    row at FeMoco-54: every remaining candidate is recorded as ``skipped``, never a measurement,
    and the log names the escalation (master plan 12.6, 12.9; Task 4.5). The large systems compile
    nothing without ``projected``; the others compile under a 4 GB cap."""
    log, stopped = [], None
    if s.name in LARGE and not projected:
        return log
    candidates = [row for row in rows if compiled_here(s, row)
                  and (only is None or row.id in only)
                  and not current(cache, f"{s.name}/{row.id}", hash_)]
    sizes = gates(candidates) if candidates else {}
    for row in candidates:
        key = f"{s.name}/{row.id}"
        if stopped:
            cache.setdefault(key, []).append(
                {"code_hash": hash_, "environment": environment(), "status": "skipped",
                 "reason": f"not compiled after {stopped}"})
            log.append(f"{key}: skipped (after {stopped})")
            continue
        if projected:
            estimate = projection(cache, row, hash_, sizes.get(row.id))
            if not estimate["allowed"]:
                cache.setdefault(key, []).append(
                    {"code_hash": hash_, "environment": environment(), "status": "refused",
                     "gates": sizes.get(row.id), "projection": estimate})
                log.append(f"{key}: refused ({estimate['reason']})")
                if s.name == "femoco54":
                    stopped = f"{key} refused"
                    log.append(f"{s.name}: stopped; escalate to the user with the projection "
                               "(master plan 12.6, \"Where compiles above the cap run\")")
                continue
            entry = measure(s, row, hash_, LARGE_CAP_MB, LARGE_TIMEOUT_S)
            entry["projection"] = estimate
        else:
            entry = measure(s, row, hash_, SMALL_CAP_MB, SMALL_TIMEOUT_S)
        entry["gates"], entry["family"] = sizes.get(row.id), family(row)
        cache.setdefault(key, []).append(entry)
        log.append(f"{key}: {entry['status']} ({entry['watchdog']['seconds']} s, "
                   f"{entry['watchdog']['peak_mb']} MB)")
        if projected and entry["status"] in ("killed", "failed"):
            stopped = f"{key} {entry['status']}"
            log.append(f"{s.name}: stopped; the orchestrator reports the {entry['status']} "
                       "compile and escalates (master plan 12.6, 12.9)")
    return log


def report_system(s: System, rows: list[Row], cache: dict, hash_: str) -> tuple[dict, list]:
    """The json record of ``s`` and the failed checks."""
    p, errors, records = s.params, [], []
    dfthcbe, paper = costs_reference.dfthcbe_blocks(p), costs_reference.paper_table3(p)
    causes = costs_reference.delta_causes(p)
    for row in rows:
        entry = current(cache, f"{s.name}/{row.id}", hash_)
        history = cache.get(f"{s.name}/{row.id}", [])
        closed = {"toffoli": row.closed.toffoli, "measurements": row.closed.measurements,
                  "measured_and_uncomputes": row.uncomputes}
        if entry:
            status = "measured"
            measured = {name: entry[name] for name in closed}
            if measured != closed:
                errors.append(f"{s.name}/{row.id}: measured {measured} != closed form {closed}")
        else:
            measured = None
            last = history[-1] if history else None
            status = ("closed form" if not compiled_here(s, row)
                      else last["status"] if last and last["code_hash"] == hash_
                      else "pending")
        ref_d = dfthcbe.get(row.key) if row.key else None
        ref_p = paper.get(row.key) if row.key else None
        codes_d, codes_p = causes.get(row.key, ({}, {}))
        if row.key == "controlled_walk_power":
            power = int(row.id.split("=")[1].rstrip(")"))
            ref_d, ref_p = None, power * paper["controlled_walk"]
            codes_d = {}
            codes_p = {code: power * value for code, value in causes["controlled_walk"][1].items()}
        delta_d = None if ref_d is None else row.closed.toffoli - ref_d
        delta_p = None if ref_p is None else row.closed.toffoli - ref_p
        for delta, codes, who in ((delta_d, codes_d, "dfthcbe"), (delta_p, codes_p, "paper")):
            if delta is None:
                continue
            unknown = sorted(set(codes) - set(REASON_CODES))
            if unknown:
                errors.append(f"{s.name}/{row.id}: unknown reason codes {unknown}")
            if sum(codes.values()) != delta:
                errors.append(f"{s.name}/{row.id}: delta {delta} against {who} != its reason "
                              f"codes {codes} (sum {sum(codes.values())})")
        counts = entry["counts"] if entry else {}
        records.append({
            "id": row.id, "level": row.level, "steps": row.steps, "calls": row.calls,
            "controlled_calls": row.controlled_calls, "partition": row.partition,
            "status": status, "measured": measured, "closed": closed,
            "dfthcbe": ref_d, "paper": ref_p,
            "paper_step18": paper.get(f"{row.key}_step18") if row.key else None,
            "delta_dfthcbe": delta_d, "delta_paper": delta_p,
            "codes_dfthcbe": dict(codes_d) if delta_d is not None else {},
            "codes_paper": dict(codes_p) if delta_p is not None else {},
            "cliffords": sum(counts.get(name, 0) for name in CLIFFORDS) if entry else None,
            "clifford_kind": clifford_kind(row), "wires": row.wires,
            "seconds": entry["seconds"] if entry else None,
            "peak_mb": entry["watchdog"]["peak_mb"] if entry else None,
            "gates": entry.get("gates") if entry else None,
            "projection": history[-1].get("projection") if history else None,
        })
    errors += cross_check(s, records)
    record = {"params": dataclasses.asdict(p), "wires": s.layout.num_wires,
              "qubit_groups": {g: len(getattr(s.layout, g)) for g in
                               ("system", "outer", "inner", "select", "phase_grad", "work")},
              "junk_wires": len(s.layout.junk_x_o) + len(s.layout.junk_b),
              "junk_measurements": costs.junk_measurements(p).measurements, "rows": records}
    if s.name == "femoco54":
        cwalk = next(r for r in records if r["id"] == "ControlledDFTHCWalk")
        basis = cwalk["measured"] or cwalk["closed"]
        formulas = ("T = (2^t - 1) T_C(W) + (t^3 - t)/6, M = (2^t - 1) M_W + t and "
                    "A = (2^t - 1) A_C(W) + (t^3 - t)/6, every correction at its maximum branch")
        record["qpe_basis"] = (
            f"T, M and A from the compiled controlled walk (Task 4.5): {formulas}."
            if cwalk["measured"] else
            f"T, M and A from the closed form of the controlled walk: {formulas}; Task 4.5 "
            "replaces them by the compiled T_C(W), M_W and A_C(W).")
        record["qpe"] = {str(bits): {
            "toffoli": (2**bits - 1) * basis["toffoli"] + (bits**3 - bits) // 6,
            "measurements": (2**bits - 1) * basis["measurements"] + bits,
            "measured_and_uncomputes": (2**bits - 1) * basis["measured_and_uncomputes"]
            + (bits**3 - bits) // 6,
            "wires": qpe_wires(s, bits)} for bits in (1, 2, 3, 10, 20, 23)}
    return record, errors


def cross_check(s: System, records: list[dict]) -> list[str]:
    """Spec 13.3 step 4: the partition rows times their calls, plus in M the block encoding's own
    junk measurements (:func:`dfthc.costs.junk_measurements`; spec 16, Q-1 variant (b)), give the
    walk and the controlled walk, in closed form always and against the measured walks where they
    are measured; and the partition sums of dfthcbe's and the paper's values give their walks."""
    errors, by_id = [], {r["id"]: r for r in records}
    junk = costs.junk_measurements(s.params).measurements
    for walk, calls in (("DFTHCWalk", "calls"), ("ControlledDFTHCWalk", "controlled_calls")):
        parts = [r for r in records if r["partition"] and r[calls]]
        total = {name: sum(r[calls] * r["closed"][name] for r in parts)
                 for name in ("toffoli", "measurements", "measured_and_uncomputes")}
        total["measurements"] += junk
        if total != by_id[walk]["closed"]:
            errors.append(f"{s.name}: partition {total} != closed {walk} {by_id[walk]['closed']}")
        if by_id[walk]["measured"] is not None and total != by_id[walk]["measured"]:
            errors.append(f"{s.name}: partition {total} != measured {walk}")
        for ref in ("dfthcbe", "paper"):
            if by_id[walk][ref] is not None and all(r[ref] is not None for r in parts):
                if sum(r[calls] * r[ref] for r in parts) != by_id[walk][ref]:
                    errors.append(f"{s.name}: {ref} partition != {ref} {walk}")
    return errors


def _cell(value) -> str:
    return "—" if value is None else f"{value:,}" if isinstance(value, int) else str(value)


def _tma(counts: dict | None) -> str:
    if counts is None:
        return "—"
    return " / ".join(f"{counts[n]:,}" for n in ("toffoli", "measurements",
                                                 "measured_and_uncomputes"))


def render(report: dict) -> str:
    """``docs/resources.md`` from the json record."""
    lines = ["# Resource report", "",
             "Generated by `scripts/resource_report.py` (spec 13.3). Every row is one program, "
             "the operator as the walk calls it, compiled on its own with `compiled_resources` "
             "on `null.qubit` where the status says measured; T, M and A are the Toffoli, "
             "explicit-measurement and measured-AND-uncompute counts (spec 7.6). A measured "
             "value always equals the closed form of `dfthc.costs`, or the script fails. "
             "dfthcbe's and the paper's values are Toffolis per call "
             "(`dfthc.costs_reference`); dfthcbe's Qualtran \"measurement\" count is our A, not "
             "our M, and the paper itemizes neither, so they are not shown. Calls are per walk "
             "and per controlled walk; the partition rows (P) sum to the walk exactly.", "",
             f"Code hash `{report['code_hash']}`; environment: "
             + ", ".join(f"{k} {v}" for k, v in report["environment"].items()) + ".", ""]
    for name, system in report["systems"].items():
        p = system["params"]
        lines += [f"## {name}", "",
                  "Parameters: " + ", ".join(f"{k} = {v}" for k, v in p.items())
                  + f". Walk qubits (`WalkLayout`): {system['wires']:,} ("
                  + ", ".join(f"{g} {n}" for g, n in system["qubit_groups"].items()) + ").", "",
                  "| Row | Level | Steps | Calls (W, C(W)) | T / M / A measured | "
                  "T / M / A closed form | Status | dfthcbe | Paper | Δ dfthcbe | Δ paper | "
                  "Reason codes | Cliffords | Qubits | Compile s | Peak MB |",
                  "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- "
                  "| --- | ---: | ---: | ---: |"]
        for r in system["rows"]:
            calls = f"{r['calls']}, {r['controlled_calls']}" + (" (P)" if r["partition"] else "")
            paper = _cell(r["paper"]) + (f" ({r['paper_step18']:,} at step 18)"
                                         if r["paper_step18"] is not None else "")
            codes = "; ".join(", ".join(f"`{c}` {v:+,}" for c, v in r[side].items()) or "—"
                              for side in ("codes_dfthcbe", "codes_paper"))
            cliffords = ("—" if r["cliffords"] is None
                         else f"{r['cliffords']:,} ({r['clifford_kind']})")
            lines.append(
                f"| `{r['id']}` | {r['level']} | {r['steps']} | {calls} | {_tma(r['measured'])} "
                f"| {_tma(r['closed'])} | {r['status']} | {_cell(r['dfthcbe'])} | {paper} | "
                f"{_cell(r['delta_dfthcbe'])} | {_cell(r['delta_paper'])} | {codes} | "
                f"{cliffords} | {r['wires']:,} | {_cell(r['seconds'])} | "
                f"{_cell(r['peak_mb'])} |")
        lines.append("")
    if "femoco54" in report["systems"]:
        lines += femoco_summary(report["systems"]["femoco54"])
    lines += large_compiles(report)
    lines += ["## Reason codes", "", "The codes of a row are the causes with a nonzero "
              "contribution to its delta, with that contribution (against dfthcbe; against the "
              "paper); the contributions sum to the delta exactly, and two different causes may "
              "cancel (`dfthc.costs_reference.delta_causes`).", "",
              "| Code | Meaning |", "| --- | --- |"]
    lines += [f"| `{code}` | {meaning} |" for code, meaning in REASON_CODES.items()]
    return "\n".join(lines) + "\n"


def femoco_summary(system: dict) -> list[str]:
    """Spec 13.1 at FeMoco-54 from the partition rows: per walk, ours, dfthcbe's and the paper's
    (Table III's attribution), the qubits, the QPE and the deferred refinements."""
    rows = {r["id"]: r for r in system["rows"]}
    parts = [r for r in system["rows"] if r["partition"] and r["calls"]]
    lines = ["## FeMoco-54 summary (spec 13.1)", "",
             "Toffolis per walk: each partition row's calls per walk times its value. The walk "
             "and the controlled walk are compiled once, in Task 4.5; until then they show their "
             "closed form, marked in their rows above.", "",
             "| Row | Calls | Ours | dfthcbe | Paper |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in parts:
        lines.append(f"| `{r['id']}` | {r['calls']} | {r['calls'] * r['closed']['toffoli']:,} "
                     f"| {_cell(None if r['dfthcbe'] is None else r['calls'] * r['dfthcbe'])} "
                     f"| {_cell(None if r['paper'] is None else r['calls'] * r['paper'])} |")
    walk, cwalk = rows["DFTHCWalk"], rows["ControlledDFTHCWalk"]
    lines.append(f"| **walk** | 1 | **{walk['closed']['toffoli']:,}** | **{walk['dfthcbe']:,}** | "
                 f"**{walk['paper']:,}** |")
    m, a = walk["closed"]["measurements"], walk["closed"]["measured_and_uncomputes"]
    ca = cwalk["closed"]["measured_and_uncomputes"]
    junk, junk_m = system["junk_wires"], system["junk_measurements"]
    ours, paper = system["qubit_groups"], costs_reference.PAPER_QUBIT_GROUPS_FEMOCO
    qubit_rows = [f"| `{group}` | {ours[group]:,} | {paper[group]:,} | "
                  f"{ours[group] - paper[group]:+,} |" for group in ours]
    lines += ["", f"- Ours minus dfthcbe: walk {walk['delta_dfthcbe']:+,}, controlled walk "
              f"{cwalk['delta_dfthcbe']:+,}. Ours minus the paper: walk {walk['delta_paper']:+,}. "
              f"The paper's {walk['paper']:,} is its QPE step, with the control on "
              f"REF_{{A u B}} only (App. B.7.2, Eq. (128)); against our controlled walk, "
              f"{cwalk['closed']['toffoli']:,}, the difference is {cwalk['delta_paper']:+,}: the "
              "block encoding's, with `ctrl-ref-t2` +1 (our controlled walk also controls "
              "REF_T2) and `ref-target` -1 (the controlled REF_{A u B} against B55), and no "
              "`qpe-ctrl`, since both carry the QPE control.",
              f"- Measurements per walk: M = {m:,}, A = {a:,} (controlled {ca:,}), so "
              f"M + A = {m + a:,} (controlled {m + ca:,}) physical measurements. {junk_m:,} of "
              "the M are the block encoding's own junk measurements, right after each prepare "
              "(spec 16, Q-1 variant (b)); the partition rows hold the rest.",
              "- The paper's Table III charges the tail-angle fix-up (N) to the inner erase; "
              "the rows show its value and, in parentheses, the value with the fix-up at step 18, "
              "where dfthcbe and we apply it. The spin multiplexer is 0 in Table III and 1 per "
              "application in App. B.3 and B.7.1 (4(N + 1) for the four swaps).",
              "", "### Qubits (spec 10.3, 13.2)", "",
              f"Ours: {system['wires']:,} wires from `WalkLayout`; the paper: "
              f"{costs_reference.PAPER_QUBITS_FEMOCO:,} (its Table 3, under the names of our "
              f"groups: the persistent ancillas of each subroutine and, as `work`, its largest "
              f"temporary count, the outer prepare's). The junk registers' {junk:,} wires are "
              "X-measured right after each prepare and share the angle register's wires (spec 16, "
              "Q-1 variant (b)); the paper shares only its inner junk (313 temporaries) with its "
              "angles and adds the outer prepare's 110 temporaries.", "",
              "| Group | Ours | Paper | Ours - paper |", "| --- | ---: | ---: | ---: |",
              *qubit_rows,
              f"| **total** | **{system['wires']:,}** | **{sum(paper.values()):,}** | "
              f"**{system['wires'] - sum(paper.values()):+,}** |", "",
              "By group: `outer` holds the dirty comparator and uniform-preparation work "
              "registers (20 wires, which make the comparator adjoint free, `cmp-free`) and the "
              "keep test's `less_than_x_o`; `inner` holds the dirty work registers (16 wires), "
              "and its `less_than_b`, `sgn`, `d_sgn` and `r` take 7 wires against the paper's "
              "11 for G, r and c; `select` adds the flags `G0`, `G1`, `b_neq_B` and `sg` to the "
              "angles; the shared `work` pool (the controlled `RefAB`'s 44 wires, the largest "
              "need) takes the place of the paper's 110 outer temporaries, because our outer "
              "junk shares the angle register like the inner junk.", "",
              "### QPE at FeMoco-54 (spec 11.3)", "",
              "| t | T | M | A | Qubits |", "| ---: | ---: | ---: | ---: | ---: |"]
    for bits in (1, 2, 3, 10, 20, 23):
        qpe = system["qpe"][str(bits)]
        lines.append(f"| {bits} | {qpe['toffoli']:,} | {qpe['measurements']:,} | "
                     f"{qpe['measured_and_uncomputes']:,} | {qpe['wires']:,} |")
    lines += ["", system["qpe_basis"], "", "### Deferred refinements (decided by the user)", "",
              "- Truncated or pruned unary iteration (Q-4): up to 100 Toffolis per walk "
              "(spec 13.1).",
              "- Keeping each alias swap's AND (the paper's remark after Eq. (127)): the swaps' "
              "undo becomes Clifford plus a measured uncompute, w_x + 2 w_b = 19 Toffolis less per "
              "walk for 14 more dirty qubits (ledger rows 2D.1 and 2D.2).",
              "- An SF fix-up one-hot of ob = w_b - 1 = 4 bits in `AngleErase`: 35 instead of 41 "
              "Toffolis per erase, 12 less per walk and per controlled walk, no extra qubits "
              "(ledger row 2E.2).", ""]
    return lines


def large_compiles(report: dict) -> list[str]:
    """The large-compile log: every refused, killed or failed entry of the current code hash
    with its projection, and the count of entries kept as history (other code hashes)."""
    hash_, lines, history = report["code_hash"], [], 0
    for key, entries in report["compiled"].items():
        history += sum(e["code_hash"] != hash_ for e in entries)
        for e in entries:
            if e["code_hash"] == hash_ and e["status"] != "compiled":
                projection = e.get("projection") or {}
                lines.append(f"| `{key}` | {e['status']} | {projection.get('seconds', '—')} | "
                             f"{projection.get('peak_mb', '—')} | "
                             f"{projection.get('reason', e.get('reason', ''))} |")
    out = ["## Large compiles", "", f"Entries of other code hashes kept as history, never shown "
           f"as current measurements: {history}.", ""]
    if lines:
        out += ["| Program | Status | Projected s | Projected MB | Reason |",
                "| --- | --- | ---: | ---: | --- |", *lines, ""]
    return out


def load(out: Path) -> dict:
    path = out / JSON_NAME
    return json.loads(path.read_text()) if path.exists() else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--large", choices=LARGE + TEST_SYSTEMS + SMALL)
    parser.add_argument("--rows", nargs="*")
    parser.add_argument("--out", type=Path, default=REPO / "docs")
    parser.add_argument("--compile-one", nargs=2, metavar=("SYSTEM", "ROW"))
    args = parser.parse_args(argv)
    if args.compile_one:
        s = build_system(args.compile_one[0])
        print(PREFIX + json.dumps(compile_counts(s, _find_row(s, args.compile_one[1]))),
              flush=True)
        return 0
    hash_ = code_hash()
    old = load(args.out)
    cache = old.get("compiled", {})
    systems = list(dict.fromkeys(args.systems + ([args.large] if args.large else [])))
    report = {"code_hash": hash_, "environment": environment(), "systems": {}, "log": []}
    errors = []
    for name in sorted(systems, key=lambda n: (SYSTEMS + TEST_SYSTEMS).index(n)):
        s = build_system(name)
        rows = build_rows(s)
        report["log"] += run_system(s, rows, cache, hash_, projected=name == args.large,
                                    only=None if args.rows is None else set(args.rows))
        record, failed = report_system(s, rows, cache, hash_)
        report["systems"][name] = record
        errors += failed
    report["compiled"] = cache
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / JSON_NAME).write_text(json.dumps(report, indent=1, sort_keys=False) + "\n")
    (args.out / MD_NAME).write_text(render(report))
    for line in report["log"]:
        print(line)
    for error in errors:
        print("CHECK FAILED:", error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: Update the documents**

Replace `docs/architecture.md` with (changed: the call tree and the paragraph after it, section "Registers", the explicit-class bullet):

````markdown
# Architecture

This package builds the DFTHC block encoding of Low et al. (PRX 15, 041016 (2025);
arXiv:2502.15882), its qubitized walk, the controlled walk and an iterative phase estimation,
as PennyLane `Operator2` building blocks. Programs compile with `qp.qjit(capture=True)` and
`qp.decompose` to a Toffoli-level gate set, and `qp.analyze` counts their resources exactly. The
design is fixed by the specification
([`superpowers/specs/2026-11-11-dfthc-pennylane-design.md`](superpowers/specs/2026-11-11-dfthc-pennylane-design.md),
cited as "spec N.M"); this page maps it onto the code.

## Layers

Each layer depends only on the layers below it (spec 5.1).

| Layer | Content | Modules |
| --- | --- | --- |
| L0, classical data | factors, parameters, alias and angle tables, the encoded Hamiltonian, fermion matrices; NumPy only | `dfthc.params`, `dfthc.registers`, `dfthc.data.factors`, `dfthc.data.sos`, `dfthc.data.alias`, `dfthc.data.angles`, `dfthc.data.tables`, `dfthc.data.encoded`, `dfthc.data.fermion` |
| L1, primitives | `PhaseGradientRotation`, `FlaggedUniformPrep`, `QROAMRead`, `PhaseFixup`, `QROAMErase`, `ZeroReflection`; the unary-iteration helper | `dfthc.ops.phase_gradient`, `dfthc.ops.uniform`, `dfthc.ops.qroam`, `dfthc.ops.reflections`, `dfthc.ops.unary`, `dfthc.ops._validate` |
| L2, paper blocks | `OuterPrepare`, `OuterUnprepare`, `OuterUnprepareFromOutcomes`, `InnerPrepare`, `InnerUnprepare`, `InnerUnprepareFromOutcomes`, `InnerSigns`, `TailAngleLoad`, `SFAngleLoad`, `AngleErase`, `SpinMux`, `SpinSwap`, `GivensRotation`, `GivensLadder`, `GivensLadderDagger`, `MajoranaGate`, `InnerSelect`, `InnerSelectDagger`, `RefT2`, `RefAB`, `ControlledRefT2`, `ControlledRefAB` | `dfthc.ops.prepare`, `dfthc.ops.angle_data`, `dfthc.ops.spin`, `dfthc.ops.givens`, `dfthc.ops.majorana`, `dfthc.ops.select`, `dfthc.ops.reflections` |
| L3, assembly | `DFTHCBlockEncoding`, `ControlledDFTHCBlockEncoding`, `DFTHCWalk`, `ControlledDFTHCWalk`; `WalkLayout` | `dfthc.ops.walk`, `dfthc.registers` |
| L4, application | `ControlledWalkPower`, `QPERound`, the driver `iterative_phase_estimation` (a plain quantum function, spec 2.1 R-4), `qpe_program`, the energy helpers | `dfthc.qpe` |
| Tooling | gate set, pins and the compile helper; closed-form costs; the paper's and dfthcbe's costs; test support | `dfthc.compile`, `dfthc.costs`, `dfthc.costs_reference`, `dfthc.testing.resources`, `dfthc.testing.sparse_sim`, `dfthc.testing.tape`, `dfthc.testing.walk_sim`, `dfthc.testing.toy_walk`, `dfthc.testing.fixtures`, `dfthc.testing.canary`, `dfthc.testing.watchdog`, `dfthc.testing.provenance` |

## The walk, call by call

The walk `W = REF_{A u B} BE` is one program. The tree below is the pinned rules' call structure
(spec 3.5 gives the steps); the resource report has one row per distinct operator call, plus
explicit auxiliary and L4 program rows ([`resources.md`](resources.md)).

```text
DFTHCWalk                                steps 1-28
  DFTHCBlockEncoding                     1-27: U^dag REF_T2 U
    OuterPrepare                         1-4: FlaggedUniformPrep, QROAMRead, keep test, CSWAPs
    (X measurement of junk_x_o)          the outer junk, right after the prepare
    InnerPrepare (round 1)               5-8: FlaggedUniformPrep (two PhaseGradientRotations), QROAMRead
    (X measurement of junk_b)            the inner junk, right after the prepare
    InnerSigns                           9
    InnerSelect                          10-18
      TailAngleLoad, SFAngleLoad         10, 11 (each a QROAMRead)
      SpinMux, SpinSwap                  12, 13
      GivensLadder                       14 (GivensRotation per rung)
      MajoranaGate                       15
      GivensLadderDagger                 16 (adjoint GivensRotation per rung)
      SpinSwap, SpinMux                  17
      AngleErase                         18
    InnerUnprepareFromOutcomes           19-22: CSWAPs, comparator adjoint, target measurement and fix-up, adjoint FlaggedUniformPrep
    RefT2                                23 (ZeroReflection)
    InnerPrepare (round 2), X measurement of junk_b, InnerSelectDagger (adjoint MajoranaGate), InnerSigns, InnerUnprepareFromOutcomes
    OuterUnprepareFromOutcomes           24-27: CSWAPs, comparator adjoint, target measurement and fix-up, adjoint FlaggedUniformPrep
  RefAB                                  28 (ZeroReflection)
```

The block encoding's rule X-measures each read's junk right after its prepare and gives the
outcomes to the matching unprepare from outcomes, which measures only the read's target and
fixes the phases from both outcome vectors (spec 16, Q-1 variant (b)); the baseline
`OuterUnprepare` and `InnerUnprepare`, which measure target and junk together through
`QROAMErase`, stay in the package with their tests but are not called by the walk.

The controlled walk `ControlledDFTHCWalk` is `ControlledDFTHCBlockEncoding`, which controls only
the T2 reflection (`ControlledRefT2`), followed by `ControlledRefAB`: on ancillas in `|0>` the
rest is `U^dag U = I` (spec 3.3). `ControlledWalkPower` repeats it with one `qp.for_loop`, and each
`QPERound` applies `H`, the walk power `2**j` times and the feed-forward phase corrections
(`PhaseGradientRotation` under `qp.cond`) on wires borrowed from the walk's `work` pool
(spec 11.1). `PhaseFixup` is the operator form of the erase fix-up that `QROAMErase`,
`AngleErase` and the unprepares from outcomes (through
`dfthc.ops.qroam.emit_erase_from_outcomes`) emit through `dfthc.ops.qroam.emit_phase_fixup`; the
walk does not call it as an operator. The toy walk of spec 11.4 (`ToyWalk`, `ControlledToyWalk`,
`ToyControlledWalkPower` in `dfthc.testing.toy_walk`) checks the phase estimation exactly.

## Registers

`dfthc.registers.WalkLayout.from_params(params, qpe_bits=0)` gives every register named wires in
the groups `system`, `outer`, `inner`, `select`, `phase_grad` and `work` (spec 10.1). The reads'
junk registers `junk_x_o` and `junk_b` have no wires of their own: each is the first wires of
`angles` (and of `junk_overflow` where a junk register is longer than the angles), because the
block encoding measures the junk right after each prepare, before any angle is loaded
(`dfthc.registers.JUNK`, spec 10.2). The assembly operators take the groups and split them once
with `WalkLayout.split_group` and `WalkLayout.split_junk`; every block they call takes only the
registers its spec 9 table names (spec 9.0, wire passing).

| Group | `H2_STO3G_TEST` | FeMoco-54 |
| --- | ---: | ---: |
| `system` (`psi_up`, `psi_dn`) | 4 | 108 |
| `outer` | 12 | 71 |
| `inner` | 16 | 65 |
| `select` (`angles`, `junk_overflow`, `G0`, `G1`, `b_neq_B`, `sg`) | 11 | 799 |
| `phase_grad` | 4 | 15 |
| `work` (the shared pool) | 8 | 44 |
| **walk** | **55** | **1,102** |

The junk shares 72 + 390 wires with the 795 angle wires at FeMoco-54, so the walk needs 462 wires
fewer than with junk registers of its own; at `H2_STO3G_TEST` the 7 inner junk wires exceed the 4
angle wires, so `junk_overflow` has 3. The paper's FeMoco-54 count is 1,131 (its Table 3): the
resource report compares the two by group ([`resources.md`](resources.md)).

The controlled walk adds one control wire; the QPE adds `qpe_control` and `qpe_grad` (t wires),
its corrections borrow the work pool, and spare wires follow only where 2t - 1 exceeds the pool
(spec 11.3). The comparators' work registers stay dirty from a prepare to its unprepare and belong
to their group, never to `work` (spec 10.2).

## Compilation

`dfthc.compile.compiled_resources(qfunc, num_wires)` compiles a program once (spec 7.2):
`qp.qjit(capture=True)` around `qp.decompose(gate_set=GATE_SET, fixed_decomps=FIXED_DECOMPS)`
around a `null.qubit` QNode, analyzed by `qp.analyze`. It returns T (Toffoli plus
`TemporaryAND`), M (`MidCircuitMeasure`) and A (`Adjoint(TemporaryAND)`), and fails if the counts
are symbolic or an operation outside the gate set remains. Four rules carry the design
(spec 7.4):

- every dfthc operator has exactly one decomposition rule per key, registered and pinned with
  `dfthc.compile.register_rule`;
- every ancilla is an explicit wire argument, and every operator states
  `required_wires(**compilable_args)`;
- the adjoint and controlled forms of the composites are explicit classes (`OuterUnprepare`,
  `InnerUnprepare`, `InnerSelectDagger`, `GivensLadderDagger`, `ControlledRefT2`,
  `ControlledRefAB`, `ControlledDFTHCBlockEncoding`, `ControlledDFTHCWalk`, and the walk's
  unprepares `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`, which take the junk
  outcomes as a dynamic array); no program calls `qp.adjoint` or `qp.ctrl` on a composite,
  because Catalyst would capture the generic candidate of its whole subtree (the K-1 memory
  finding), and an inventory test checks it;
- `QROAMRead` loads each unary leaf with one `qp.for_loop` over a dynamic `int64` table array
  (spec 9.3), so the FeMoco-54 walk compiles in a few GB.

## Costs and the resource report

`dfthc.costs` implements the closed forms of spec 9.24; every compiled count in the tests equals
one of them exactly. `dfthc.costs_reference` evaluates the paper's Table III formulas and ports
dfthcbe's per-block formulas, checked against dfthcbe's own call graph.
`scripts/resource_report.py` writes [`resources.md`](resources.md) and
[`resources.json`](resources.json): one row per program, measured where compiled, with the
closed form, dfthcbe's and the paper's values, the deltas and the reason codes that explain them,
each with its contribution (`dfthc.costs_reference.delta_causes`; spec 13.3). Its compiles are
cached with `dfthc.testing.provenance.code_hash()`, which covers `src/dfthc` outside
`dfthc.testing`, the report script and `requirements-nightly.lock`, while ignoring Python
docstrings and comments. After Task 4.5, the FeMoco-54 walk and controlled-walk compiles can be
reused while that hash matches.

## Further reading

- [`conventions.md`](conventions.md): indices, signs, fermions, angles, registers.
- [`verification.md`](verification.md): what each test establishes and how to run the tiers.
- [`limitations.md`](limitations.md): the PennyLane and Catalyst limitations and their canaries.
- [`environment.md`](environment.md): the environment, the load limits and the commands.
- [`agents/HANDOFF.md`](agents/HANDOFF.md): the shared context of the development agents.
- [`agents/LEDGER.md`](agents/LEDGER.md): task status, decisions and heavy-run evidence.
````

Replace `docs/conventions.md` with (changed: the register-name bullet; new: the junk bullet):

```markdown
# Conventions

The conventions follow dfthcbe at commit `930255020cec30172e0a7d17c0071d9476a687d1` unless stated
otherwise, so blocks compare one to one (spec 6). Symbols are those of spec 3.1:
`X_o = N + RC`, `w_x = lg X_o`, `w_b = lg(B + 1)`, `w_r = lg R`, with `lg y = ceil(log2 y)`.

## Indices and flags (spec 6.1)

- **The generator index `x_o`.** `x_o < RC` is a sum-of-squares (SF) generator, with
  `x_o = r C + c`; the last N values are the one-body tail, its `N+` D1 rows first, then the Q1
  rows.
- **Families.** `G0 = 0` is SF and `G0 = 1` one-body; `G1 = 0` is D1 and `G1 = 1` Q1.
- **The inner index `b`.** `b = B` is the identity term; `r = floor(x_o / C)`.
- **Success flags.** `succ_x_o` and `succ_b` are 1 where the flagged uniform preparations succeed;
  a failed `x_o` makes `REF_T2` act as `-I`, a failed `b` makes the select act as `+I`
  (spec 8.5).

## Signs and spins (spec 6.1)

- `InnerPrepare` loads `sgn = [w_b < 0]` and `d_sgn = sgn XOR sgn[alt]`; `InnerSigns` applies
  `Z(sgn)` and `CZ(less_than_b, d_sgn)`.
- `sigma_0` is the inner spin (the Majorana flavour on one-body rows), `sigma_1` the outer spin;
  `sg = sigma_{G0}` controls the spin swap.

## Fermions (spec 6.2)

- Jordan-Wigner per spin register: `gamma_{j,0} = Z_0 ... Z_{j-1} X_j` and
  `gamma_{j,1} = Z_0 ... Z_{j-1} Y_j`, qubit 0 the most significant, `|1>` occupied; the system is
  `psi_up` (x) `psi_dn`.
- The Majorana gate acts on `psi_dn[0]`: SF applies `-Z` (dfthcbe's convention, `i gamma_0
  gamma_1`; the paper's Fig. 4 shows `+Z`), D1 `a = (X + iY)/2`, Q1 `a^dag = (X - iY)/2`.

## Givens angles (spec 6.3)

Rung k acts on qubits `(q_k, q_{k+1})`. An angle is stored as
`x_k = round(theta_k / (2 pi) * 2^(b_rot - 1)) mod 2^(b_rot - 1)`, and the rung realizes
`theta = 4 pi x / 2^b_rot` exactly, with the rung matrix `[[cos, -sin], [sin, cos]]` on
`{|01>, |10>}`. Rung N - 2 comes first in time and rung 0 last.

## Alias sampling (spec 6.4)

`less_than = [keep <= m]`; a set flag swaps `alt` into the index register. Keep values have
`b_k1` (outer) and `b_k2` (inner) bits.

## Phase gradient (spec 6.5)

`PhaseGradientStatePrep` prepares the gradient once, outside the walk, which returns it
unchanged. Adding `k` into the gradient register multiplies the state by `e^{+2 pi i k / 2^b}`;
`PhaseGradientRotation(k, m)` applies `e^{2 pi i k t / 2^m}` on `|t>`.

## Reflections (spec 6.6)

Reflections are exact, global sign included: `2|0><0| - I` is a multi-controlled Z with
`GlobalPhase(pi)` when uncontrolled and `Z` on the control when controlled. No generic
`C(GlobalPhase)` is used.

## Wires and registers (spec 6.7)

- Integers in registers are big-endian: the first wire is the most significant bit.
- Core register names follow dfthcbe: `x_o`, `succ_x_o`, `sigma_1`, `m_x_o`, `less_than_x_o`,
  `alt_x_o`, `keep_x_o`, `junk_x_o`, `b`, `succ_b`, `sigma_0`, `m_b`, `less_than_b`, `alt_b`,
  `keep_b`, `sgn`, `d_sgn`, `r`, `junk_b`, `angles`, `G0`, `G1`, `b_neq_B`, `sg`, `psi_up`,
  `psi_dn`, `phase_grad`, `work`. `WalkLayout` also names the persistent comparator registers
  `succ_work_x_o`, `cmp_work_x_o`, `succ_work_b`, `cmp_work_b`, and `junk_overflow`, the junk wires
  beyond the angles; QPE adds `qpe_control` and `qpe_grad`.
- The L3 assembly operators and `ControlledWalkPower` take `WalkLayout`'s groups. `QPERound`
  takes `control`, `qpe_grad` and the walk groups once as flat `walk_wires`; L1 and L2 operators
  take their spec 9 registers by name. A composite passes each child only the registers it needs.
- One operator's registers are distinct qubits: two registers that share a wire do not compile.
- The reads' junk registers `junk_x_o` and `junk_b` share the first wires of `angles` (and of
  `junk_overflow`) in `WalkLayout`: the block encoding measures each junk register right after
  its prepare, so no operator holds a junk register and the angles at once (spec 10.2; spec 16,
  Q-1 variant (b)).

## Arguments and tables

- Everything that determines the circuit's structure is a compilable argument; dynamic arguments
  carry runtime data only (`QROAMRead`'s per-leaf table words, `PhaseFixup`'s parities,
  `QPERound`'s earlier bits, the junk outcomes of `OuterUnprepareFromOutcomes` and
  `InnerUnprepareFromOutcomes`, one per junk wire in `QROAMRead.junk_layout` order). Measurement
  outcomes are a bool array under capture and an object array of `MeasurementValue`s in tape
  mode, so a resource key that declares them takes its dtype from the mode
  (`dfthc.ops.prepare.abstract_junk_outcomes`).
- Compilable integers fit in signed 64 bits, so wide table entries are tuples of 32-bit words,
  least significant first (L-24); dynamic table words are `int64` holding 32 data bits (L-26).
- Adjoints use the instance form `qp.adjoint(Op(...))` (L-25).

## Numbers in tests

Every comparison states both tolerances: `rtol=0` with an absolute `atol` (1e-12 for dense and
simulator comparisons unless a test names another), or an explicit relative bound with `atol=0`;
`tests/test_tolerance_hygiene.py` enforces it.
```

Replace `docs/verification.md` with (changed: the certified blocks; new: the paragraph on the simulator):

```markdown
# Verification

What the tests establish, where they are, and how to run them (spec 12). Correctness rests on
three independent checks: per-block semantics on `lightning.qubit` and the sparse simulator, the
end-to-end checks of the whole walk on the sparse simulator, and GF(2) certificates for every
measured erase. Resources are asserted exactly against closed forms.

## Test taxonomy (spec 12.1)

| Code | What it establishes | Where |
| --- | --- | --- |
| T1 | classical data: tables, normalizations, both offset frames, the encoded model, agreement with dfthcbe's fixtures | `tests/data/` |
| T2 | rule validity: declared resources equal emitted operators (`qp.testing.assert_valid_decomp_rule`) | `tests/ops/`, `tests/test_qpe.py` |
| T3 | semantics against NumPy references, compiled on `lightning.qubit` or on the sparse simulator | `tests/ops/`, `tests/integration/` |
| T4 | exact T, M and A against `dfthc.costs`: table-bearing blocks at the two H2 presets (per commit) and `MID_NAPHTHALENE` (nightly), table-free blocks over representative widths, the QPE round and program at H2 sizes, and FeMoco-54 on closed forms | `tests/ops/`, `tests/test_qpe.py`, `tests/integration/test_walk_counts.py`, `tests/integration/test_qpe_counts.py`, `tests/test_costs.py` |
| T5 | gate-set closure at `level="user"` and no dfthc operator left at `level="device"` | `tests/ops/`, `tests/test_qpe.py` |
| T6 | the pinned rules are chosen (count signatures, verbose rule listings) and the inventory test: no modified dfthc composite is captured | `tests/ops/`, `tests/test_qpe.py`, `tests/ops/test_walk.py` |
| T7 | canaries: one test per claim of every known limitation | `tests/canary/` |
| T8 | end-to-end checks E-1 to E-4 | `tests/integration/test_end_to_end.py` |

Supporting tests: `tests/test_work_pool.py` (every call gets exactly its `required_wires`),
`tests/sim/` (the simulator's differential tests), `tests/test_compile.py` and
`tests/test_primitive_costs.py` (the gate set and PennyLane's primitive costs),
`tests/report/` (the resource report), `tests/test_docstrings.py` (the documentation).

## The sparse simulator (spec 12.3)

`dfthc.testing.tape.decompose_for_simulation` decomposes a program on the tape path with the
pinned rules, and `dfthc.testing.sparse_sim` applies it to a sparse state with one handler per
gate-set member. `TemporaryAND` and its measured uncompute assert their preconditions on every
term. In strict mode every measurement that is neither a branch wire nor postselected must have
probability 1/2. Measurement policies: forced schedules (the certificates), seeded random
outcomes, and branching on listed wires (the QPE control). `dfthc.testing.walk_sim` builds the
walk cases (`sto3g`, `631g`, `fails-c3`) and their schedules.

## Outcome certificates (spec 12.5)

An erase X-measures M qubits; its fix-up must cancel the phase `(-1)^{xi . g(a)}` for every
outcome vector `xi`. The residual is GF(2)-linear in `xi`, so the zero vector and the M unit
vectors prove it for all `2^M` outcomes: each block runs M + 1 forced schedules on the simulator,
after checking that every conditional is diagonal with a linear predicate. Certified blocks:
`QROAMRead` with `QROAMErase`, the angle loads with `AngleErase`, each prepare with its
unprepare, and each prepare with the X measurement of its junk and its unprepare from outcomes,
where `xi` ranges over the junk outcomes and the target outcomes together (spec 16, Q-1 variant
(b); `tests/ops/test_certificates_qroam.py`, `test_certificates_angles.py`,
`test_certificates_prepare.py`). A test operator whose rule holds the inner prepare, the junk
measurement and the unprepare from outcomes certifies the junk outcomes crossing an operator
boundary inside another rule, as in the block encoding
(`tests/ops/test_unprepare_from_outcomes.py`).

The walk measures each read's junk right after its prepare, in the block encoding's rule, and the
target in the unprepare. Nothing in the simulator changes for that: a forced schedule assigns
outcomes in program order, so the junk outcomes come first; a fix-up's predicate is a parity of
`MeasurementValue`s of both measurements, evaluated on the recorded outcomes when the
conditional runs; strict mode asserts probability 1/2 for each of them. The end-to-end schedules
(all zeros, all ones, alternating, seeded) keep their meaning, and the total number of
measurements is unchanged.

## End-to-end checks (spec 12.4)

On `H2_STO3G_TEST` (and E-1, E-2 on `FAILS_C3`), with every ancilla in `|0>`:

| Check | Statement |
| --- | --- |
| E-1 | `<0|BE|0> = H_SA/Lambda - I` to 1e-10, column by column |
| E-2 | `<0|W^k|0> = T_k(H_SA/Lambda - I)` for k = 1, 2 |
| E-3 | the controlled walk with the control in `|0>`, `|1>`, `|+>`, and after one uncontrolled step |
| E-4 | prepare then unprepare returns every ancilla to `|0>` |

The oracle is `dfthc.data.encoded.encoded_block`. QPE of the walk itself is checked on the
simulator against the encoded block's exact phase distribution
(`tests/integration/test_qpe_walk.py`), and the toy walk's QPE exactly and on `lightning.qubit`
(`tests/integration/test_qpe_toy.py`).

## Resources

Every compiled T, M and A count equals `dfthc.costs`. Table-bearing blocks use the two H2 presets
per commit and `MID_NAPHTHALENE` nightly; table-free blocks use representative register widths,
including FeMoco-54 widths where inexpensive. The QPE tests compile the round and full program at
H2 sizes and `ControlledWalkPower` at `MID_NAPHTHALENE`. The FeMoco-54 values (walk 9,793
Toffolis, controlled walk 9,795, M = 2,524, A = 8,675 and 8,677) are asserted on the closed forms
per commit. The resource report compiles every program of the walk at the H2 systems and
`naphthalene` (and `fe2s2`, `fe4s4` as heavy runs), checks each against its closed form, and
checks that the partition rows times their calls give the compiled walk. Task 4.5 will compile
the FeMoco-54 walk and controlled walk once and cache them in [`resources.json`](resources.json);
`tests/report/test_femoco_cache.py` (release tier) will assert the cached counts while the code
hash matches.

## Tiers (spec 12.6)

| Tier | Command | Content | Budget |
| --- | --- | --- | --- |
| per commit | `scripts/test.sh -m "not slow and not release"` | T1 to T6 at small sizes, canaries, simulator tests, the report at `H2_STO3G_TEST` | 15 min |
| nightly (phase gates) | `scripts/test.sh -m "slow and not release"` | certificates, E-1 to E-4 under the nightly schedules, T4 at `MID_NAPHTHALENE`, the report at `naphthalene`, subprocess canaries | 4 h |
| release (once) | `scripts/test.sh` | everything, with E-1 and E-3 under more schedules and the cached FeMoco-54 counts | 8 h |

`scripts/test.sh` runs two workers at lowered priority (`DFTHC_TEST_WORKERS` overrides). One
test: `scripts/test.sh tests/ops/test_walk.py::test_name -q`. The nightly and release tiers and
every heavy run are started by the orchestrator under the watchdog (see
[`environment.md`](environment.md)).
```

- [ ] **Step 9: Run the targeted tests**

Run: `scripts/test.sh tests/test_registers.py tests/test_work_pool.py tests/ops/test_block_encoding.py tests/ops/test_walk.py tests/test_qpe.py tests/report/test_report.py tests/integration/test_walk_counts.py tests/integration/test_qpe_counts.py -m "not slow and not release" -q`
Expected: `197 passed` (dry run 139 s, 1,266 MB).

- [ ] **Step 10: Run E-1 to E-4**

Run: `scripts/test.sh tests/integration/test_end_to_end.py -m "slow and not release" -q`
Expected: `10 passed` (dry run 70 s).

- [ ] **Step 11: Regenerate the report of the H2 systems and `naphthalene`**

Run: `PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --cap-mb 3000 --timeout-min 40 -- "$VENV/bin/python" scripts/resource_report.py`
Expected: 182 lines `<system>/<row>: compiled (<s> s, <MB> MB)` (46 for each H2 system, 44 for `naphthalene`), no `CHECK FAILED` line, a `WATCHDOG` line with `"returncode": 0` and `"killed": null`; about 11 minutes (dry run 664 s, process tree 795 MB). `fe2s2`, `fe4s4` and `femoco54` are built in tape mode, not compiled: their rows show closed forms, their earlier measurements stay in the json as history. Check the key rows against section "Dry-run results": every row `measured` at the four small systems, the `naphthalene` walk `1,913 / 502 / 1,678` on 336 qubits, the FeMoco-54 qubit table totalling 1,102 against 1,131.

- [ ] **Step 12: Run the documentation tests on the regenerated report**

Run: `scripts/test.sh tests/test_docstrings.py -q`
Expected: `48 passed`.

- [ ] **Step 13: Run the per-commit tier and the linter, and commit**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1270 passed` (1265 + 5), no warnings summary.

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

```bash
git add src/dfthc/registers.py src/dfthc/ops/walk.py src/dfthc/costs.py src/dfthc/costs_reference.py \
    src/dfthc/qpe.py src/dfthc/testing/walk_sim.py scripts/resource_report.py \
    tests/test_registers.py tests/ops/test_block_encoding.py tests/ops/test_walk.py \
    tests/test_work_pool.py tests/test_qpe.py tests/report/test_report.py \
    docs/architecture.md docs/conventions.md docs/verification.md docs/resources.md docs/resources.json
git commit -m "Measure the junk right after each prepare and share its wires with the angles"
```

- [ ] **Step 14: Report and stop for the large reruns**

Report the RED outputs of Step 2, the counts of Steps 9, 10, 12 and 13, Step 11's time and its largest compile, and stop. The code hash changed, so the orchestrator then runs, from the lane, alone (master plan 12.9), each under the watchdog with a ledger row (large-compile count 0: not FeMoco-shaped):

```bash
PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 120 -- \
    "$VENV/bin/python" scripts/resource_report.py --large fe2s2
PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 120 -- \
    "$VENV/bin/python" scripts/resource_report.py --large fe4s4
```

Expected for each: the small systems reused (no compile line), then one line per program of the large system, `compiled (...)` for all 44 (section "Projections": every one allowed, at most 1.53 GB and 57 s at `fe2s2` and 2.53 GB and 109 s at `fe4s4`, about 11 and 21 minutes of compiles by that projection, about 5 minutes each by the estimate), no `CHECK FAILED`, exit 0, a `WATCHDOG` line below the 8 GB cap. The walks T / M / A 4,414 / 1,020 / 3,916 and 6,534 / 1,534 / 5,945, the controlled walks + 2 T and + 2 A, walk qubits 517 and 792. A `refused` line is recorded and the run continues; a `killed` or `failed` line stops that system's run, and the orchestrator reports it and escalates before committing anything. Then the orchestrator runs `scripts/test.sh tests/test_docstrings.py tests/report/test_report.py -m "not slow and not release" -q` (expected `110 passed`: 48 + 62) and commits the regenerated files on the branch:

```bash
git add docs/resources.md docs/resources.json
git commit -m "Report: the fe2s2 and fe4s4 compiles after Q1b"
```

---

## Interfaces for Task 4.5

- **The walk to compile** is Q1b's: 1,102 wires at FeMoco-54 (1,103 with the controlled walk's control), the same T, M and A as before (9,793 / 2,524 / 8,675 and 9,795 / 2,524 / 8,677), so `tests/report/test_femoco_cache.py`'s `EXPECTED` needs no change; its QPE wires come from `report.qpe_wires` (1,104 at t = 1). Its per-commit test passes on the Q1b tree, and its expected per-commit count is 1271 = 1270 + 1 (revision 5). Its release test `test_femoco_cache` sums the partition rows and must add the block encodings' own junk measurements in M (`costs.junk_measurements(FEMOCO54).measurements`, 852; the partition alone gives 1,672), as `resource_report.cross_check` does; the report plan's revision 8 makes that change.
- **The projection** (Task 4.5, Step 1 of its procedure) uses the committed report after the orchestrator's reruns; this plan's estimate is 40.0 s and 1,628.3 MB (walk) and 34.3 s and 1,728.8 MB (controlled walk), and 198.2 s and 3,892.5 MB and 195.9 s and 4,399.9 MB from the small systems alone (B-M11).
- **The report's rows at FeMoco-54** are 47: the walk's distinct calls (the unprepares from outcomes instead of `QROAMErase`), the controlled walk's own, `PhaseFixup@...` and `C(ZeroReflection)@...`, `ControlledWalkPower(p=1)`, `(p=2)`, and `QPE(t=1)` to `(t=3)`.
