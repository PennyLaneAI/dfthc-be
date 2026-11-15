# DFTHC Phase 4, Tasks 4.3 to 4.5: Resource Report, Documentation and the FeMoco Compile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver D-3 and D-6: the resource report of spec 13.3 (every program of the walk compiled and checked against its closed form, next to the paper's and dfthcbe's counts with a reason code for every delta), the documentation set of spec 5.2 with an audit of every operator docstring, and the single FeMoco-54 compile of the walk and the controlled walk, cached with a code hash.

**Architecture:** `src/dfthc/costs_reference.py` evaluates the paper's Table III formulas and ports dfthcbe's README formulas in pure Python (checked against dfthcbe's own call graph, evaluated once in `.venv-ref` by `scripts/make_report_fixture.py`). `src/dfthc/testing/provenance.py` gives the docstring-insensitive code hash, the environment record and Task 0.9's projection. `scripts/resource_report.py` finds every operator call of the walk and the controlled walk by running the composites' pinned rules in tape mode, compiles each as its own program in a watched subprocess, checks it against `dfthc.costs`, checks that the partition rows sum to the walk, and writes `docs/resources.md` and `docs/resources.json`; `--large SYSTEM` projects each program first and compiles under the 8 GB cap. Task 4.4 adds `docs/architecture.md`, `docs/conventions.md`, `docs/verification.md`, refreshes `docs/limitations.md` and `docs/environment.md`, and tests the docstrings and the documents. Task 4.5 adds the cache test; the orchestrator compiles FeMoco-54.

**Tech Stack:** Python 3.12, PennyLane `Operator2` and Catalyst (`qp.qjit(capture=True)`, `qp.decompose`, `qp.analyze`), `pennylane.decomposition.DecompositionGraph` (gates from the resource functions), NumPy, `ast` and `hashlib`; `dfthc.testing.watchdog.run_watched`; Qualtran and dfthcbe only in `.venv-ref`; pytest with pytest-xdist; ruff.

This file holds Tasks 4.3, 4.4 and 4.5 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 10). Task 4.3 starts after Task 4.1 and runs beside Task 4.2; Task 4.4 follows Tasks 4.1 to 4.3; then the user's Q-1 decision; then Task 4.5 (wave 4, master plan section 5). The conditional Task Q1b is not in this file: it is written at the Q-1 decision if the user chooses variant (b). The QPE plan (`2026-11-11-dfthc-phase4-qpe.md`, revision 3) is taken as final, with its section "Interfaces for Tasks 4.3 to 4.5".

**Revision 2** (2026-11-14), after the GPT-5.6 review `.superpowers/sdd/phase4r-plan-review-1.md` (0 Critical, 4 Important, 2 Minor); revision 1's assembled plan is kept as `.scratch/plan-check/rev1-p4r/2026-11-11-dfthc-phase4-report.md`.

- **I-1, `--large` refused 14 of the 46 rows at `fe2s2` and `fe4s4`: fixed.** Every row now has a size measure, the gates of its own operator from the resource functions: `PhaseFixup` built in tape mode with an abstract parity array, `C(ZeroReflection)` through its controlled rule, `ControlledWalkPower(p)` through its resource function (p times the controlled walk's gates); one decomposition graph serves all rows of a system (`gates(rows)`). Rows calibrate by family, their class and comparison key (`family`), because a row's id can differ between systems (`PhaseGradientRotation` has one call shape at the H2 systems and two elsewhere, which the new test found). `project_compile` bounded a target no larger than a measured point by the largest time and peak over the points with at least as many gates (`method: "bounded"`, the blocks whose size does not grow; revision 3 takes all of the family's points) and keeps Task 0.9's two-point extrapolation otherwise (`method: "k1"`). The new per-commit test `test_every_large_row_projects` builds a synthetic cache (the real rows and gates of the four small systems, made-up seconds and MB, so a code-hash change cannot break it) and asserts an allowed projection for all 46 rows at each of `fe2s2` and `fe4s4`. `projections-large.log` now covers those 92 rows and the two FeMoco-54 programs, all allowed, none refused (largest `ControlledWalkPower(p=1)`: 1.6 GB at `fe2s2`, 2.6 GB at `fe4s4`); section "Projections", R-M4 (item 25), R-M9 and Task 4.3's Step 12 are corrected.
- **I-2, the report script was not hashed: fixed.** `code_hash` also covers the docstring-free syntax tree of `scripts/resource_report.py` (`provenance.REPORT_SCRIPT`). `test_cache_needs_matching_code_hash` copies the script into its root and checks that a docstring and a comment edit of the script keep the hash and a program-builder edit changes it. R-M7, the new R-M16 (the cache test) and the new R-S13 (spec 13.3, step 5) state the scope; risk 3 is rewritten. The dry-run report was regenerated once under the new hash.
- **I-3, no stop after a refusal or kill: fixed.** Under `--large`, the first `killed` or `failed` compile stops the system's run, and at FeMoco-54 so does the first `refused` projection; every remaining candidate is recorded with the new status `skipped`, never a measurement, and the log names the escalation. Two new per-commit tests, which compile nothing: `test_large_stops_after_a_refusal_at_femoco` and `test_large_stops_after_a_kill`. Task 4.5's Step 4, risk 1, section "The report as built", R-M10, R-M15 and the new R-S11 and R-S12 (spec 13.3, step 2) state the policy.
- **I-4, `qpe-ctrl` on the controlled rows: fixed.** The static code table is gone. `costs_reference.delta_causes(p)` splits every block into parts that ours, dfthcbe's and the paper's constructions share and gives each part's difference to the code that names its mechanism; a row lists the codes with a nonzero contribution, each with its contribution, and the report checks at every system that they sum to the delta exactly (section "Reason-code decomposition at FeMoco-54", the table the orchestrator asked for). `qpe-ctrl` now appears only on `RefAB` and the walk; `ControlledRefAB` and the controlled walk carry `ref-target` −1 against B55, and `ControlledRefT2` against the paper lists both `ref-unif` −1 and `ctrl-ref-t2` +1, which cancel. The paper's controlled `REF_T2` and controlled block encoding are its uncontrolled ones (its QPE step controls only `REF_{A u B}`), so those rows now have a paper value. The new per-commit test `test_reason_codes_at_femoco` asserts the exact codes and contributions of `RefT2`, `ControlledRefT2`, `RefAB`, `ControlledRefAB`, both block encodings, both walks and `ControlledWalkPower(p=1)` and `(p=2)`, against dfthcbe and the paper. R-S1 states the listing rule; R-S2 and the new R-S10 gave the angle erase and the angle loads `no-prune` alone (revision 3 replaces this location-based split by a mechanism-based one); R-S6 gives the controlled walk's −202 by cause; R-S5 already agrees.
- **M-1, duplicate heading: fixed.** The template's own `## Global Constraints` line is removed; the shared fragment supplies the heading.
- **M-2, the port's guarantee: narrowed.** `costs_reference`'s module docstring, change 4 and R-M6 guarantee `dfthcbe_blocks` only at the report's eight fixture-pinned parameter sets, as risk 5 says.
- **Also redone:** `main` moved to `5fe4d65` (Task 3.5 merged at `329ede8` with the assembly plan's revision 6; the later commits touch only the ledger), so the replica and the replay take Task 3.5's files from `main` and overlay only Tasks 4.1 and 4.2; the counts chain, the replay, the one-worker evidence, the document edits, the byte identity and the per-commit tier were rerun (sections below).

**Revision 3** (2026-11-14), after the GPT-5.6 re-check `.superpowers/sdd/phase4r-plan-review-2.md` (0 Critical, 3 Important, 1 Minor; revision 1's six findings resolved, and the paper's controlled `REF_T2` and controlled block encoding confirmed equal to its uncontrolled ones); revision 2's assembled plan is kept as `.scratch/plan-check/rev1-p4r/2026-11-11-dfthc-phase4-report.rev2.md`.

- **I-1, the bounded projection was not conservative: fixed.** A target no larger than some measured member of its family now projects to the largest time and the largest peak over *all* the family's current entries, whatever their gate counts; gate count only decides between this and extrapolation, since it is not a monotone measure of compile cost (`project_compile`, method `bounded`, reason "the family's largest measurement"). New per-commit tests: `test_bounded_projection_takes_every_measurement` (the review's points, 100 gates at 3,500 s and 7,000 MB and 200 gates at 10 s and 100 MB: a 150-gate target projects 3,500 s and 7,000 MB) and `test_large_sequence_projects_fe4s4_after_fe2s2` (synthetic `fe2s2` entries in the cache, then every `fe4s4` program projected; an expensive made-up `fe2s2` `SpinMux` entry sets the `fe4s4` `SpinMux` projection; nothing compiled). The review's example is allowed under the default limits, since 3,500 s is 58 minutes and 7,000 MB is below 7.5 GB, so the test shows the refusals under the small systems' 4 GB cap and with the dearer point at 3,700 s, refused under 60 minutes. Change 8, risk 1, section "The report as built", R-M7, R-M4 and R-S11 now say exactly this; the projection log and tables were redone.
- **I-2, the FeMoco QPE summary omitted A: fixed.** Every FeMoco QPE record has `measured_and_uncomputes` = (2^t − 1) A_C(W) + (t^3 − t)/6 from the controlled walk; the summary renders `T | M | A | Qubits`, `qpe_basis` names A's formula, and `test_femoco_cache` asserts T, M, A and wires of the summary for t = 1, 2, 3, 10, 20, 23 against the cached controlled walk (R-M17). The new per-commit `test_qpe_summary_and_wire_boundaries` checks the summary from the closed form (A = 8,677, 26,032, 60,743 at t = 1, 2, 3) and the wire boundaries: `H2_STO3G_TEST` 67, 68, 70 at t = 3, 4, 5 and FeMoco-54 1,586, 1,587, 1,589 at t = 21, 22, 23.
- **I-3, the angle reason codes named locations, not mechanisms: fixed.** The codes follow spec 13.3's definitions again. `costs_reference.iteration_counts(p)` gives every unary iteration as ours (exact), an intermediate (the same walk with truncated iteration and no aligned-block pruning), dfthcbe's and the paper's count. `exact-iter` is ours minus the intermediate; `no-prune` is the intermediate minus dfthcbe, whose walk stops once on each aligned block of a dead interval; a walk over a sub-range `[lo, hi)`, `lo > 0`, can be truncated only by that pruning, so its intermediate is our exact walk. Against the paper's one Toffoli per leaf a third mechanism appears, truncated iteration's `c − 2`, for which a new code `iter-const` joins the ten (R-S1). At FeMoco-54 `AngleErase` gets `exact-iter` +2 (over `r`) and `no-prune` +5 (over the tail rows) per call, with `onehot-pad` +4 and `sf-enable` +1 (ledger row 2E.2). The angle loads get `exact-iter` +22 and `no-prune` +3 per select (ledger row 3.1's +25 under both codes), and the alias rows keep what the derivation gives: inner read `exact-iter` +5 and `no-prune` −4 (spec 13.1's "+5 per read, less dfthcbe's 4 stops"), inner erase +4 and −3, outer read and erase `exact-iter` alone. Edits: R-S2 restores both codes with the per-walk split (+4 and +10), R-S10 is rewritten from the derivation, R-S5 splits the outer read, and the new R-S14 and R-S15 name the codes of the inner-unprepare and inner-read rows. `test_reason_codes_at_femoco` asserts `TailAngleLoad`, `SFAngleLoad` and `AngleErase` exactly, against dfthcbe and the paper.
- **M-1, the template statement: fixed.** Change list and section "Shared edits" now say that this plan's template changed only to drop its redundant `## Global Constraints` heading (revision 2) and that the shared fragment did not change.
- **Also redone:** the small-system report under revision 3's code hash, the projections, the counts chain, the replay, the one-worker evidence, the document edits (against `main` at `5fe4d65`), the byte identity of the other 12 plans and the per-commit tier (sections below). Task 4.1 (`a0c9171`) is in review; its files in the replica are still the QPE plan's revision 3 blocks.

**Revision 4** (2026-11-14), after the GPT-5.6 re-check `.superpowers/sdd/phase4r-plan-review-3.md` (0 Critical, 1 Important, 0 Minor; everything else resolved, including the alias-row splits, the SF load and fix-up, `iter-const` against `onehot-const`, the bounded projections, QPE A and the document edits); revision 3's assembled plan is kept as `.scratch/plan-check/rev1-p4r/2026-11-11-dfthc-phase4-report.rev3.md`.

- **I-1, the tail intermediate was not the stated construction: fixed by the review's option 2.** `iteration_counts`' intermediate is now documented as an accounting convention, a cost model and not a circuit: truncated iteration over the live leaves without stops on the walks from address 0, our exact count on the tail rows `[RC, X_o)`. It is a valid circuit only on the outer read, the outer erase's fix-up and the SF fix-up, where it equals dfthcbe's walk at all eight parameter sets. On the inner read, the inner erase's fix-up and the SF load it leaves out dfthcbe's stops on dead addresses that carry amplitude. On the tail rows no truncated walk without stops on the dead SF prefix is valid (the review's truncated call graph on `[270, 324)`, 53 Toffolis, assumes an in-range selection), so there the convention's count only fixes ours against dfthcbe as `no-prune`. The tail walks now reach the paper transitively, in the new public `costs_reference.iteration_causes(p)`: ours against dfthcbe (`no-prune` +5 at FeMoco-54, as before), plus dfthcbe against the paper split by dfthcbe's mechanisms, its s(RC) = 4 stops on the dead SF prefix and its truncated iteration's −2 (`iter-const`). Spec 13.3 has no code for those stops (`no-prune` is our missing pruning, not dfthcbe's extra stops), so a twelfth code joins it in R-S1: `prefix-stops`. At FeMoco-54, against the paper, `TailAngleLoad` is `no-prune` +5, `prefix-stops` +4, `iter-const` −2 (was `no-prune` +7); `AngleErase`, whose tail fix-up is split the same way, is `exact-iter` +2, `no-prune` +5, `prefix-stops` +4, `iter-const` −4, `onehot-pad` +4, `onehot-const` −1, `sf-enable` +1, `attrib` +54 (was `no-prune` +7, `iter-const` −2 and the rest unchanged); the walk's paper column moves from `no-prune` +28 and `iter-const` −18 to `no-prune` +20, `prefix-stops` +16 and `iter-const` −26. Every dfthcbe column and every row without a tail walk keeps its causes. `test_dfthcbe_blocks_equal_the_call_graph` gains an independent check: at all eight parameter sets dfthcbe's own call graph (the fixture) gives the tail walk exactly N + s(RC) − 2, the split's N leaves, `prefix-stops` and `iter-const`, and the intermediate equals dfthcbe's walk on the three walks where it is a valid circuit. `test_reason_codes_at_femoco` asserts the new `TailAngleLoad`, `AngleErase` and block-encoding causes in both columns. Edits: R-S1 gains the `prefix-stops` row and states the convention and the transitive tail in its listing rule (its label now says twelve rows, not ten); R-M6 names `iteration_causes`. R-S2 and R-S10 give only causes against dfthcbe, which did not change, and spec 13.1's "+80" bullet against the paper names no codes and keeps its total, so none of them changes.
- **Also redone:** the small-system report (the code hash changed with `costs_reference` and the report's code table; the counts are revision 3's, 1,253 s on a busier machine against revision 3's 808 s) and the projections from it, which therefore rose: every program is still allowed, at most 1.6 GB and 123 s at `fe2s2`, 2.6 GB and 234 s at `fe4s4`, 4.5 GB and 408 s for the FeMoco-54 walk (R-M4, R-M9 and R-M15 carry the new numbers, and R-M2 now names revision 4); the replay of the three tasks; the one-worker evidence; the per-commit tier. `main` moved to `7e0a472` during this revision (at `d61b162` Task 4.1 merged with its review fixes, the QPE plan's revision 4 and one line of spec 11.1; then a ledger commit): all 35 document edits apply exactly once there, and the other 12 assembled plans equal it byte for byte. The replica and the replay stay at `5fe4d65` with the QPE plan's revision 3 files of Tasks 4.1 and 4.2, which the orchestrator mirrors: the merged `qpe.py` differs from them only in `ControlledWalkPower`'s docstring (the same code hash, so the report's cache stays current), and Task 4.1's two review tests raise the base to 1112 (the QPE plan's revision 4), so every count of this plan's chain then rises by 2. No test was added or removed here. The nightly `test_report_mid` was not rerun: it runs `main --systems naphthalene` with the report's checks, which the regenerated report passed.

**Revision 5** (2026-11-15, by the orchestrator, at approval after the GPT-5.6 re-check `.superpowers/sdd/phase4r-plan-review-4.md`: Approved, 0 Critical, 0 Important, 0 Minor). The replica now holds Task 4.1's merged files (the QPE plan's revision 4): `src/dfthc/qpe.py`, whose only change is `ControlledWalkPower`'s docstring, and `tests/integration/test_qpe_toy.py` and `tests/integration/test_qpe_walk.py` with the two review tests. The code hash is unchanged (`2e5482ad...` before and after), so the regenerated report stays current. The replica collects `1217/1328 tests collected (111 deselected)` per commit, and `tests/integration/test_qpe_toy.py` with `test_reason_codes_at_femoco` and `test_dfthcbe_blocks_equal_the_call_graph` passes there (`25 passed`, one worker). Every expected per-commit count rises by 2: the base is 1112 (the QPE plan's revision 4, after Task 4.2), then 1174, 1216 and 1217. The measured runs quoted below (the replay, the one-worker evidence, the tier's `1215 passed`) predate this and are kept as measured.

**Revision 6** (2026-11-15, by the orchestrator, after Task 4.3's review `.superpowers/sdd/task-4.3-review-1.md`: GPT-5.6, Needs fixes, 0 Critical, 2 Important, 0 Minor; the report data and the production code were correct). Both findings were test gaps in the large-run safety rules, and the review's test-only fixes are applied: `test_large_compile_is_watched` asserts that the watched run gets `LARGE_TIMEOUT_S` unchanged (I-1: a doubled timeout survived), and the stop test, renamed `test_large_stops_after_a_failed_or_killed_compile`, runs for a `killed` and a `failed` first row (I-2: a stop on `killed` only survived), + 1 per commit. No hashed file changed, so the committed report stays current (code hash `2e5482ad...`). Every expected per-commit count from Task 4.3 on rises by 1: 63 for Task 4.3, then 1175, 1217 and 1218. The measured runs quoted below predate this.

**Revision 7** (2026-11-15, by the orchestrator, after Task 4.4's review `.superpowers/sdd/task-4.4-review-1.md`: GPT-5.6, Needs fixes, 0 Critical, 2 Important, 4 Minor; 3 of 13 targeted mutants survived the audit). The replica holds the reviewed `tests/test_docstrings.py` and four reviewed documents (`environment.md` unchanged). The audit now finds `Operator2` subclasses recursively in `dfthc.ops` (re-exports included) and in `dfthc.qpe`, keyed by qualified name, and requires the summary line before the Action section (I-1); the link test covers every Markdown document under `docs/` outside `superpowers/` (10) and checks anchors, and an L-number may appear only once (I-2). The documents' factual fixes: the verification guide's test locations and resource coverage, and Task 4.5 in the future tense (M-1); the architecture's report rows, the exact code-hash inputs and a direct ledger link (M-2); the conventions' comparator and QPE registers and `QPERound`'s flat `walk_wires` (M-3); the limitations' date and provenance sentence (M-4). No hashed file changed, so the committed report stays current. Task 4.4 has 46 tests (RED `2 failed, 41 passed`, measured on `main` with only the test file), so every expected per-commit count from Task 4.4 on rises by 4: 1221 for Task 4.4 and 1222 for Task 4.5. Since this revision, per-commit tests read `docs/` (including `agents/LEDGER.md`): a tree that differs from a tested one only in `docs/` needs `tests/test_docstrings.py` rerun before its tier result carries.

**Code provenance.** Every code block below was executed on 2026-11-14 in a scratch replica of the repository (`.scratch/plan-check/repo-phase4r`, not committed): `git archive` of local `main` at `5fe4d65` (Task 3.5 merged at `329ede8`, the assembly plan at revision 6; the later commits up to `5fe4d65` touch only the ledger), plus the six files of Tasks 4.1 and 4.2 (QPE plan, revision 3), each written from the last code block that plan on `main` gives (`stages/4R/check_overlay_blocks.py`, `overlay-blocks.log`: `repo-phase4q` equals those blocks, `tests/test_qpe.py` up to one trailing blank line that the plan assembly strips; the eight files of Tasks 3.3 to 3.5 on `main` equal the assembly plan's blocks). The replica is not a git repository, so the dry run ran pytest the way `scripts/test.sh` does (the same environment, 2 workers, `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0`, lowered priority) through `.scratch/plan-check/p4rtest.sh`, which runs every session under the watchdog with a 3,000 MB cap and a session lock; every script ran through `p4rrun.sh` (same cap, same lock), and the `.venv-ref` scripts through `ref_run.sh` (same cap, same lock). The three tasks were replayed in order, RED then GREEN, in a fresh copy of `main` at `5fe4d65` with the same overlay (`stages/4R/run_stages_4r.sh`, log `4r-stages.log`); every "Expected" line below is that replay's output, and its files are byte-identical to the replica's. Further evidence under `stages/4R/`: the per-commit tier with all three tasks (`4r-per-commit-tier.log`), every per-commit test of the three tasks in one one-worker session (`4r-per-test-evidence.log`), the nightly report test (`4r-report-mid.log`), the dry-run report of the H2 systems and `naphthalene` (`report-small-systems.log`, summarized in `report-summary.md`), the `--large` path on a stand-in (`probes/p4r/standin_large.sh`, `large-standin.log`), the projections of every `fe2s2`, `fe4s4` and FeMoco-54 program (`probes/p4r/project_large.py`, `projections-large.log`), the size measures of the extra rows (`probes/p4r/probe_size_ops.py`), the QPE wire counts and the 4 GB cap's effective limit (`probes/p4r/probe_qpe_wires.py`), the FeMoco-54 decomposition and iteration counts (`decomposition_4r.py`, `decomposition.md`, `iteration-counts.md`), the gate counts (`probe-gates.log`), the first walk compiles at the default precisions (`probe-walk-compiles.log`), dfthcbe's rotation cost in `.venv-ref` (`probe-zpow-ref.log`), the docstring audit and its mutants (`docstring-audit.log`, `mutate_docstrings_4r.py`, `docstring-mutants.log`) and the document edits (`make_doc_edits_4r.py`, `doc-edits.md`). Revision 2's runs waited 12 minutes for the orchestrator's hold on the lock (`lock-waits.log`); revisions 3 and 4 did not wait.

**Load.** At most 2 pytest workers ran at any time and one process that compiled or ran pytest at a time (the shared lock), each under the watchdog with a 3,000 MB cap; the largest process tree was the per-commit tier, 2,232 MB, and the largest single compile was the `naphthalene` walk, 634 MB. Nothing above `naphthalene` was compiled: `fe2s2`, `fe4s4` and FeMoco-54 were built in tape mode only (their rows, 46 and 49, and their gates from the resource functions, at most 177 MB). The full per-commit tier ran once in each revision. No load-limit event occurred.

---

## Changes from the master plan, the spec and HANDOFF

The changes that touch a document are applied by the exact edits listed after this list (ids R-M for the master plan, R-S for the spec, R-H for HANDOFF). Of the plan templates only this plan's own changed, in revision 2 and only to drop its redundant `## Global Constraints` heading, which the shared fragment supplies; the shared fragment `templates/_global_constraints_g01.md` did not change, and no edit touches it.

1. **The rows are the walk's operator calls** (R-M8, R-S7). The master plan lists the programs by class; a class is called at several shapes (`QROAMRead` at four: the outer and inner alias reads, the tail and SF angle loads), and each shape has its own closed form. So each row is one distinct call: the report runs each composite's pinned rule in tape mode, collects the dfthc operators it calls, and identifies a call by its class, compilable arguments and register sizes. Rows are named by class where the class has one call shape (`SpinSwap`), else `class@parent` (`QROAMRead@InnerPrepare`). `PhaseFixup` (the operator form of the erase fix-up, which the erases emit through `emit_phase_fixup`) and `C(ZeroReflection)` are not called by the walk; they get rows with their callers' arguments. 48 rows at `H2_STO3G_TEST`, 46 at each vendored system, 49 at FeMoco-54 (with its three closed-form QPE rows).
2. **Calls per walk and the partition** (R-S7). The partition rows are the children of the walks, block encodings and selects (18 rows: `OuterPrepare` once, `InnerPrepare` twice, ..., `RefAB` once in the walk, `ControlledRefT2` and `ControlledRefAB` once in the controlled walk); their calls per walk times their closed forms give the walk and the controlled walk exactly, in T, M and A. The report checks this at every system against the closed form and, wherever the walks are compiled, against the compiled totals: Task 4.5's cross-check, exercised per commit at `H2_STO3G_TEST`. dfthcbe's and the paper's values partition the same way (9,641 and 9,997 at FeMoco-54), which the report also checks.
3. **A per call** (no edit). Spec 9.24 gives A per walk as T minus the plain Toffolis. Per call that is wrong for the prepares and unprepares: a prepare leaves its comparators' ANDs (`b_k` and the uniform preparation's `lg l − 1`) for its unprepare to measure (spec 10.2). The report's closed form of A per call follows the merged tests (`expected_and_uncomputes` in `tests/ops/test_outer_prepare.py`); over the partition it sums to `costs.measured_and_uncomputes`. The first dry run used T minus the plain Toffolis and failed on exactly the six prepare, unprepare and uniform rows (`probe-rows-h2-compile.log` predates the fix; the replay passes).
4. **dfthcbe's values: a formula port, checked against dfthcbe's call graph** (R-M5, R-M6, R-S8, R-H1). `costs_reference.dfthcbe_blocks(p)` ports the per-step formulas of dfthcbe's README, with Qualtran's truncated iteration over the live leaves and the README's dead stops together (one call graph detail the README's `L − 2` hides: at `H2_STO3G_TEST` the inner erase iterates one live leaf and one stop, 4 Toffolis, not 5). Chosen over vendored values because the report needs dfthcbe's counts at every system and every row without Qualtran in the main environment. The README's formulas are piecewise, so the port is guaranteed only where the fixture pins it: at the report's eight parameter sets; a new set needs a fixture entry first. The check is a fixture: `scripts/make_report_fixture.py` evaluates dfthcbe's own symbolic call graph in `.venv-ref` at the eight parameter sets of the report (the six vendored systems, FeMoco-54, `H2_STO3G_TEST`) and writes `tests/fixtures/dfthcbe/report_toffolis.json`; the port equals it exactly (29 blocks at each set), and equals Task 1A.6's 12 FeMoco-54 counts. Two new files for Task 4.3: the script and the fixture.
5. **Default precisions** (R-S9). "Every vendored system with default precisions" means Table III's precisions (b_rot = b_k1 = b_k2 = 15, s_u = 7, as `FEMOCO54` and `MID_NAPHTHALENE`) with the block bits of `costs.cheapest_block_bits` (spec 8.3): `costs_reference.default_params`. At `naphthalene` they are `MID_NAPHTHALENE`'s parameters exactly.
6. **`test_report_h2` per commit within the per-test gate** (R-M10). Compiling all 48 programs takes about 45 s in one process, above the per-commit gate of a few seconds per test; so `test_report_h2` is one test per program (48 in-process compiles, each at most 4.5 s and below 1 GB in a one-worker session), and `test_report_h2_rows_and_columns` checks every row and column without compiling, through `main`. Further per-commit tests, none of which compiles, check the dfthcbe port against the fixture, Table III, the projection method, a projection for every large row from a synthetic calibration, the stop policy of `--large` (two tests), and the exact reason codes at FeMoco-54.
7. **Reason codes** (R-S1, R-S2, R-S4 to R-S6, R-S10, R-S14, R-S15). Twelve codes join spec 13.3, each for a cause the ledger names without a code: `maj-and` (the Majorana gate's −2 against the paper), `sf-enable` (the SF fix-up enable, +1 against dfthcbe), `rot-add` (our rotation costs s − 1 against Qualtran's s − 2, confirmed in `.venv-ref`: `probe-zpow-ref.log`, closing ledger row 1C.1), `unif-odd` (dfthcbe's uniform-preparation formula against the paper's 4 lg L), `sign-bits` (Table III's 20-bit inner entries carry no sign bit), `onehot-const` (a one-hot over K values costs K − 2, the paper counts K), `iter-const` (truncated iteration over L leaves costs L + c − 2, the paper counts L), `prefix-stops` (dfthcbe's s(RC) stops on the dead SF prefix of its tail walks, which the paper's B41 and B50 leave out), `ref-target` (the reflections' phase target), `ref-unif` (`REF_T2`'s −1 against the paper), `qpe-ctrl` (the paper books its QPE control in `REF_{A u B}`, which our plain walk lacks) and `ctrl-ref-t2` (the controlled walk's +1 per step against the paper's QPE control model, ledger decision "Controlled walk powers keep C(W)^p"). The codes of a row are derived, not listed by hand: `costs_reference.delta_causes(p)` decomposes each delta by cause (section "Reason-code decomposition at FeMoco-54"), a row lists the causes with a nonzero contribution, two different causes that cancel are both listed, and a row whose contributions do not sum to its delta fails the report. Every unary-iteration difference is split by mechanism (`costs_reference.iteration_causes`) at the intermediate of `costs_reference.iteration_counts`, an accounting convention and not a circuit (truncated iteration without stops on the walks from address 0, our exact count on the tail rows): `exact-iter` and `no-prune` keep spec 13.3's meanings, and `iter-const` takes the truncated count against the paper's; the tail walks reach the paper through dfthcbe's walk, adding its stops on the dead SF prefix (`prefix-stops`) and its truncated constant (`iter-const`). So spec 13.1's angle-erase reason becomes the Task 2E.2 review's (+14: `exact-iter` +4 over `r`, `no-prune` +10 over the tail rows; +8 `onehot-pad`; +2 `sf-enable`), its angle-loads row gives its split (+44 `exact-iter`, +10 and −4 `no-prune`), its inner-read and inner-unprepare rows name the codes of their exact boundaries and dfthcbe's stops, its outer-unprepare row cites the paper's remark after Eq. (127), its uniform-preparation row names `rot-add`, its remaining terms get their codes, and its last paragraph compares the paper's 9,997 (its QPE step) with our controlled walk (−202, by cause).
8. **What compiles where** (R-M9, R-M15, R-S11, R-S12). At `fe2s2` and `fe4s4` all 46 programs compile through `--large` (they have no QPE rows), `ControlledWalkPower` included (an operator; it costs about the controlled walk's compile). Each is projected first from the current entries of its family at every system: a program no larger than some measured member of its family projects to the largest time and the largest peak over all the family's measurements, whatever their gate counts (gate count only decides between this and extrapolation), and a larger one is extrapolated by Task 0.9's method. All 92 project within the limits, far below the K-1 stand-in's 4 to 9 GB (section "Projections"); master plan section 3, item 9's statement is kept as history, and Task 4.3's entry now gives the production projections. A refused program is recorded with its projection and shows its closed form marked refused. The first killed or failed compile stops the system's run, and at FeMoco-54 so does the first refusal: the remaining programs are recorded as skipped and the orchestrator escalates.
9. **Interfaces** (R-M6, R-M7, R-M16, R-S13). `code_hash(root=REPO)` and `environment(root=REPO)` take the repository root, so the cache test hashes a copied package and report script; the hash covers `src/dfthc/` without `testing/`, `scripts/resource_report.py` and the lock file. `project_compile(points, gates, *, cap_mb, time_limit_s)` takes the largest time and peak of all the points where some point has at least `gates` gates and otherwise extrapolates, refusing with fewer than two calibration points. `costs_reference.delta_causes(p)` gives each row key's contributions by code, splitting iteration by `costs_reference.iteration_causes(p)` at the counts of `costs_reference.iteration_counts(p)`. `resource_report.main(argv)` has `--systems`, `--large SYSTEM`, `--rows [ID ...]` (compile only these rows; the tests use it) and `--out DIR`, and the internal `--compile-one SYSTEM ROW` of its subprocesses.
10. **Task 4.3's definition of done** (R-M11). The implementer commits the code, the fixture and the report of the H2 systems and `naphthalene` (13 to 21 minutes of single-core compiles, depending on the machine's load, below 1 GB: not heavy) and stops; the orchestrator runs `--large fe2s2` and `--large fe4s4` alone and commits the regenerated report on the branch.
11. **Task 4.4** (R-M12, R-M13). The audit covers `dfthc.qpe` too (`ControlledWalkPower`, `QPERound`: spec 9.23 and 11.1 operators whose cost sections name `dfthc.costs` functions); the toy classes live in `dfthc.testing` and name the toy module's costs, so they are not audited. Every one of the 32 classes passes the audit as merged, so Task 4.4 edits no docstring (section "Docstring audit"). The documentation tests also check that the documentation set of spec 5.2 exists, that `docs/limitations.md` has an entry for every limitation and names only existing canaries, and that `docs/architecture.md` names every operator and module.
12. **Task 4.5** (R-M14, R-M15). A per-commit companion, `test_stale_femoco_entry_names_the_recompile`, checks the stale-entry failure without compiling; the release test `test_femoco_cache` checks the cached counts, their provenance and the cross-check. The procedure states the projection, the stop after a refusal, kill or failure, and the escalation.
13. **Documents** (R-M1 to R-M4, R-H2, R-H3). The file table drops "(written when the walk classes are merged)"; the provenance paragraph names this file's dry run; section 3, item 23's captured-equation projection is corrected to the reviews' 7,449 and 4,795 (ledger rows 3.3 and 3.4: "correct A-M3 ... with the next master-plan edit"); a new item 25 records the report as built; HANDOFF names the report and its commands.

Exact edits (master plan, spec and HANDOFF, against local `main`):

Each edit replaces one exact text of the file on local `main` at `7e0a472`, where it occurs exactly once (checked by `.scratch/plan-check/stages/4R/make_doc_edits_4r.py`, which also applies them in order). Every master-plan old text also occurs exactly once in `.scratch/plan-check/templates/2026-11-11-dfthc-master-plan.md`, where the orchestrator mirrors the edits, and none occurs in the shared Global Constraints fragment.

**R-M1** (2026-11-11-dfthc-master-plan.md, section 1, the file table). Old:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase4-report.md` | Tasks 4.3 to 4.5 (written when the walk classes are merged) |
````

New:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase4-report.md` | Tasks 4.3 to 4.5 |
````

**R-M2** (2026-11-11-dfthc-master-plan.md, section 1, code provenance (the end of the paragraph)). Old:

````text
(revision 2: against `main` at `e447e53`, with Task 3.3 merged, and the assembly file's revision 4 code for Tasks 3.3 to 3.5); its preamble lists its changes to this plan.
````

New:

````text
(revision 2: against `main` at `e447e53`, with Task 3.3 merged, and the assembly file's revision 4 code for Tasks 3.3 to 3.5); its preamble lists its changes to this plan. The Phase 4 report file (`phase4-report`, Tasks 4.3 to 4.5) was executed the same way on 2026-11-14 (revision 4), against `main` at `5fe4d65`, with Task 3.5 merged, and Tasks 4.1 and 4.2's files from the QPE file (revision 3); its preamble lists its changes to this plan. Its large compiles (`fe2s2`, `fe4s4`, FeMoco-54) are projected there and run by the orchestrator.
````

**R-M3** (2026-11-11-dfthc-master-plan.md, section 3, item 23 (the captured-equation projection)). Old:

````text
projected to FeMoco-54, about 7,100 and 4,500 equations against the PX stand-in's 62,832 and 8,230, about 59,500 fewer for the controlled walk (15% of its 390,949).
````

New:

````text
projected to FeMoco-54, 7,449 and 4,795 equations by the Task 3.3 and 3.4 reviews' fits (the plan had estimated about 7,100 and 4,500) against the PX stand-in's 62,832 and 8,230, 58,818 fewer for the controlled walk (15.0% of its 390,949).
````

**R-M4** (2026-11-11-dfthc-master-plan.md, section 3, a new item 25 after item 24 (the end of item 24 is shown)). Old:

````text
`H2_STO3G_TEST` t = 1 in 24 s and t = 2 in 487 s (65 and 66 wires, two key words).
````

New:

````text
`H2_STO3G_TEST` t = 1 in 24 s and t = 2 in 487 s (65 and 66 wires, two key words).
25. **The resource report as built** (Phase 4 report dry run, 2026-11-14). Every row of `scripts/resource_report.py` is one program: an operator exactly as the walk or the controlled walk calls it, found by running each composite's pinned rule in tape mode (48 rows at `H2_STO3G_TEST`, 46 at each vendored system, 49 at FeMoco-54 with its QPE rows; `QROAMRead` has four rows, one per call shape), plus `PhaseFixup` and `C(ZeroReflection)` with their callers' arguments and the L4 programs. The 18 partition rows times their calls per walk give the walk and the controlled walk exactly (T, M and A), which the report checks at every system. Per call, A is not T minus the plain Toffolis: a prepare leaves its comparators' ANDs for its unprepare to measure (spec 9.24, 10.2). At the H2 systems and `naphthalene`, at the report's default precisions (b_rot = b_k1 = b_k2 = 15, s_u = 7, `cheapest_block_bits`), all 190 programs compile, each in its own watched subprocess, to their closed forms, in 0.8 to 16.3 s and at most 634 MB each (the `naphthalene` walk 15.5 s at 634 MB), 1,253 s for the four systems on a busy machine (808 s on a quieter one). `dfthc.costs_reference.dfthcbe_blocks`, a port of dfthcbe's README formulas with Qualtran's truncated iteration over the live leaves and the dead stops together, equals dfthcbe's own call graph (`.venv-ref`) at all eight parameter sets of the report; `paper_table3` gives Table III's 9,997. Qualtran's `ZPowConstViaPhaseGradient` costs s − 2 Toffolis against our s − 1 (reason code `rot-add`). Each row's reason codes come from an exact decomposition of its two deltas by cause (`costs_reference.delta_causes`), which the report checks at every system. Every program at `fe2s2` and `fe4s4` (46 each) has a size, the gates of its operator from the resource functions (walk 296,917 at `fe2s2`, 566,743 at `fe4s4`, 1,073,044 at FeMoco-54), and is projected from the current compiles of its family at the four small systems: where one is at least as large (8 and 10 programs whose size does not grow), the largest time and peak over all the family's compiles, whatever their gate counts, else Task 0.9's method. All 92 project within 7.5 GB and 60 minutes, at most 1.6 GB and 123 s at `fe2s2` and 2.6 GB and 234 s at `fe4s4` (the walks have the largest peaks, `ControlledWalkPower(p=2)` the longest times); the FeMoco-54 walk projects to 4.5 GB and 408 s and its controlled walk to 3.8 GB and 430 s; the captured-equation model of ledger rows 2E.4, 3.1, 3.3 and 3.4 (about 274,000 equations against the PX stand-in's 390,949 at 2,648 MB) puts the FeMoco-54 compile near 1.9 GB. Under `--large` the first killed or failed compile stops the system's run, and at FeMoco-54 so does the first refusal; the remaining programs are recorded as skipped. `code_hash` covers the report script too. All 32 `Operator2` classes of `dfthc.ops` and `dfthc.qpe` have the nine spec 9.0 sections in order.
````

**R-M5** (2026-11-11-dfthc-master-plan.md, Task 4.3, files). Old:

````text
- **Files:** `src/dfthc/costs_reference.py`, `src/dfthc/testing/provenance.py`, `scripts/resource_report.py`, `docs/resources.md`, `docs/resources.json`, `tests/report/test_report.py`.
````

New:

````text
- **Files:** `src/dfthc/costs_reference.py`, `src/dfthc/testing/provenance.py`, `scripts/resource_report.py`, `scripts/make_report_fixture.py` (dfthcbe's call graph at the report's parameter sets, run in `.venv-ref`), `tests/fixtures/dfthcbe/report_toffolis.json`, `docs/resources.md`, `docs/resources.json`, `tests/report/test_report.py`.
````

**R-M6** (2026-11-11-dfthc-master-plan.md, Task 4.3, interfaces: `costs_reference`). Old:

````text
  - `costs_reference.paper_table3(p) -> dict[str, int]` and `costs_reference.dfthcbe_blocks(p) -> dict[str, int]`;
````

New:

````text
  - `costs_reference.paper_table3(p) -> dict[str, int]` (Table III's formulas with its attribution, and `_step18` keys with the tail fix-up at step 18) and `costs_reference.dfthcbe_blocks(p) -> dict[str, int]` (a port of dfthcbe's README formulas, guaranteed only at the report's eight parameter sets, where it equals dfthcbe's call graph: `test_dfthcbe_blocks_equal_the_call_graph`; a new parameter set needs a fixture entry first), both Toffolis per call keyed by our row names; `delta_causes(p) -> dict[str, tuple[dict[str, int], dict[str, int]]]`, per row key the signed contribution of each reason code to ours minus dfthcbe's and to ours minus the paper's, nonzero only, summing to the deltas exactly; `iteration_counts(p)`, each unary iteration as ours, the intermediate (an accounting convention, not a circuit: truncated iteration without stops on the walks from address 0, our exact count on the tail rows), dfthcbe's and the paper's count, and `iteration_causes(p)`, each iteration's split into `exact-iter`, `no-prune` and `iter-const`, the tail walks reaching the paper through dfthcbe's (`prefix-stops`); `default_params(N, R, B, C, n_plus)`, the report's parameters of a vendored system (Table III's precisions, `cheapest_block_bits`); `PAPER_QUBITS_FEMOCO = 1131`;
````

**R-M7** (2026-11-11-dfthc-master-plan.md, Task 4.3, interfaces: `provenance`). Old:

````text
  - `dfthc.testing.provenance.code_hash() -> str`: SHA-256 over the syntax trees of `src/dfthc/**/*.py` without docstrings and comments (`testing/` excluded) and the lock file, so documentation edits keep the hash; `environment() -> dict` (Python, platform, package versions, lock digest); `project_compile(points, gates) -> dict`, the K-1 projection method of Task 0.9 (power law in time, least-squares line in memory, over measured `(gates, seconds, peak_mb)` points), with gates from `DecompositionGraph.resource_estimate` and the verdict from `within_effective_limit` (7.5 GB under the 8 GB cap);
````

New:

````text
  - `dfthc.testing.provenance.code_hash(root=REPO) -> str`: SHA-256 over the syntax trees, without docstrings and comments, of `src/dfthc/**/*.py` (`testing/` excluded) and of `scripts/resource_report.py` (which builds the cached programs and selects the cache's entries), and the lock file, so documentation edits keep the hash and any code edit of the package or the script changes it; `environment(root=REPO) -> dict` (Python, platform, package versions, lock digest); `project_compile(points, gates, *, cap_mb=MEMORY_CAP_MB, time_limit_s=3600) -> dict` over measured `(gates, seconds, peak_mb)` points of the same program family: where some point has at least `gates` gates (blocks whose size does not grow), the largest time and the largest peak over all the family's points, whatever their gate counts (gate count decides only between this and extrapolation), else the K-1 projection method of Task 0.9 (power law in time through the two largest points, exponent at least 1; least-squares line in memory, slope at least 0; fewer than two points refuse), with every program's gates from its operator's `DecompositionGraph.resource_estimate` and the verdict from `within_effective_limit` (7.5 GB under the 8 GB cap);
````

**R-M8** (2026-11-11-dfthc-master-plan.md, Task 4.3, programs (the L4 item and a new item)). Old:

````text
    - L4: `ControlledWalkPower` with p = 1 and 2, and the QPE program with t = 1, 2, 3.
````

New:

````text
    - L4: `ControlledWalkPower` with p = 1 and 2, and the QPE program with t = 1, 2, 3.
    - One row per distinct call: the rows are the operator calls of the walk and the controlled walk, found by running each composite's pinned rule in tape mode, so an L1 class has one row per call shape (`QROAMRead@OuterPrepare`, `@InnerPrepare`, `@TailAngleLoad`, `@SFAngleLoad`); `PhaseFixup` (with each erase's fix-up arguments) and `C(ZeroReflection)` (on each reflection's pattern) are programs the walk does not call. 48 rows at `H2_STO3G_TEST`.
````

**R-M9** (2026-11-11-dfthc-master-plan.md, Task 4.3, what is compiled where (`fe2s2` and `fe4s4`)). Old:

````text
    - `fe2s2` and `fe4s4`: every operator, the walk and the controlled walk through `--large`, run by the orchestrator, alone. These are heavy: the K-1 stand-in projects 4 to 6 GB for the `fe2s2` walks and 6 to 9 GB for the `fe4s4` walks (section 3, item 9). A program projected above 7.5 GB is not compiled on this machine; it goes where the G0 decision sends FeMoco-size compiles, or its row stays the closed form, marked pending;
````

New:

````text
    - `fe2s2` and `fe4s4`: every program except the QPE through `--large`, run by the orchestrator, alone. The K-1 stand-in had projected 4 to 6 GB for the `fe2s2` walks and 6 to 9 GB for the `fe4s4` walks (section 3, item 9); the production compiles of all 46 programs at each project within the limits, at most 1.6 GB and 123 s at `fe2s2` and 2.6 GB and 234 s at `fe4s4` (section 3, item 25), so none is expected to be refused. A program projected above 7.5 GB is not compiled on this machine; it goes where the G0 decision sends FeMoco-size compiles, or its row stays the closed form, marked refused. The first killed or failed compile stops the system's run: the remaining programs are recorded as skipped, never as measurements, and the orchestrator escalates;
````

**R-M10** (2026-11-11-dfthc-master-plan.md, Task 4.3, tests (`test_report_h2`)). Old:

````text
    - `test_report_h2` (per commit): runs on `H2_STO3G_TEST` only and checks every program and column above, with measured equal to closed form.
````

New:

````text
    - `test_report_h2` (per commit): runs on `H2_STO3G_TEST` only and checks every program and column above, with measured equal to closed form: one test per program (48, each one in-process compile of at most a few seconds), and `test_report_h2_rows_and_columns`, every row and column without compiling, through `main`.
    - Also per commit, none compiling: `test_dfthcbe_blocks_equal_the_call_graph` (the port against `report_toffolis.json` and Task 1A.6's FeMoco fixture), `test_paper_table3_reproduces_table_iii` (9,997), `test_project_compile_is_the_k1_method` (a 7.8 GB projection is refused; the bounded case), `test_every_large_row_projects` (from a synthetic calibration with the small systems' real rows and gates, every `fe2s2` and `fe4s4` program has an allowed projection), `test_large_sequence_projects_fe4s4_after_fe2s2` (with `fe2s2`'s entries in the calibration), `test_bounded_projection_takes_every_measurement` (non-monotone points), `test_large_stops_after_a_refusal_at_femoco` and `test_large_stops_after_a_kill` (the next program is recorded skipped, never measured), `test_reason_codes_at_femoco` (the exact codes and contributions of the angle loads and erase, the reflections, block encodings, walks and walk powers) and `test_qpe_summary_and_wire_boundaries` (the FeMoco-54 QPE summary's T, M and A; the QPE wires at t = 4, 5 and 22, 23).
````

**R-M11** (2026-11-11-dfthc-master-plan.md, Task 4.3, definition of done). Old:

````text
- **Definition of done:** `docs/resources.md` and `docs/resources.json` generated with the `fe2s2` and `fe4s4` compiles (one ledger entry per heavy run) and the FeMoco walk totals pending, and committed. Task 4.5 completes D-3.
````

New:

````text
- **Definition of done:** the implementer commits the code, the fixture and the report of the H2 systems and `naphthalene`, and stops; the orchestrator runs `--large fe2s2` and `--large fe4s4`, each alone under the watchdog with a ledger entry, and commits the regenerated `docs/resources.md` and `docs/resources.json` on the branch, with the FeMoco walk totals pending. Task 4.5 completes D-3.
````

**R-M16** (2026-11-11-dfthc-master-plan.md, Task 4.3, tests (`test_cache_needs_matching_code_hash`)). Old:

````text
    - `test_cache_needs_matching_code_hash`: an entry with another code hash is not reused, and a docstring edit keeps `code_hash()`.
````

New:

````text
    - `test_cache_needs_matching_code_hash`: an entry with another code hash is not reused; a docstring or comment edit, in the package or in `scripts/resource_report.py`, keeps `code_hash()`, and a code edit of either, such as a program builder of the script, changes it.
````

**R-M12** (2026-11-11-dfthc-master-plan.md, Task 4.4, files). Old:

````text
- **Files:** `docs/architecture.md`, `docs/conventions.md`, `docs/verification.md`, `docs/limitations.md` (consolidated), `docs/environment.md` (refreshed), `tests/test_docstrings.py`.
````

New:

````text
- **Files:** `docs/architecture.md`, `docs/conventions.md`, `docs/verification.md`, `docs/limitations.md` (consolidated), `docs/environment.md` (refreshed), `tests/test_docstrings.py`. No docstring needs an edit: the audit passes on every class as merged (Phase 4 report dry run).
````

**R-M13** (2026-11-11-dfthc-master-plan.md, Task 4.4, acceptance tests). Old:

````text
  - `test_docstrings`: every `Operator2` subclass in `dfthc.ops` has the nine spec 9.0 sections in order, and its cost section names an existing `dfthc.costs` function.
  - The documentation links resolve to existing files.
````

New:

````text
  - `test_docstrings`: every `Operator2` subclass in `dfthc.ops` and `dfthc.qpe` (32 classes) has the nine spec 9.0 sections in order, and its cost section names an existing `dfthc.costs` function, and its example calls `compiled_resources`.
  - The documentation links resolve to existing files.
  - The documentation set of spec 5.2 exists; `docs/limitations.md` has every limitation L-1 to L-27 and names only existing canaries; `docs/architecture.md` names every operator class and every module.
````

**R-M14** (2026-11-11-dfthc-master-plan.md, Task 4.5, files). Old:

````text
- **Files:** `docs/resources.md` and `docs/resources.json` (regenerated, with the cached compile), `tests/report/test_femoco_cache.py` (marked `release`, `femoco`).
````

New:

````text
- **Files:** `docs/resources.md` and `docs/resources.json` (regenerated, with the cached compile), `tests/report/test_femoco_cache.py` (`test_femoco_cache` marked `release`, `femoco`; `test_stale_femoco_entry_names_the_recompile` per commit, which compiles nothing).
````

**R-M15** (2026-11-11-dfthc-master-plan.md, Task 4.5, procedure (its last sentence)). Old:

````text
These are two of the six FeMoco-scale compiles, each with a ledger entry.
````

New:

````text
These are two of the six FeMoco-scale compiles, each with a ledger entry. The Phase 4 report dry run projects them, from the H2 systems and `naphthalene` alone, to 4.5 GB and 408 s (walk) and 3.8 GB and 430 s (controlled walk), and the captured-equation model to about 1.9 GB (section 3, item 25); the `fe2s2` and `fe4s4` points refine the projection. If the walk's projection still exceeds 7.5 GB or 60 min, or its compile is killed or fails, the run stops before the controlled walk, which is recorded as skipped and never compiled, and the orchestrator escalates to the user with the projection (section 2, "Where compiles above the cap run": a machine with at least 32 GB, or an approved idle-time window with a higher cap).
````

**R-M17** (2026-11-11-dfthc-master-plan.md, Task 4.5, tests (`test_femoco_cache`)). Old:

````text
it asserts the cached counts and the cross-check without compiling
````

New:

````text
it asserts the cached counts, the cross-check and the report's FeMoco-54 QPE summary (T, M and A = (2^t - 1) A_C(W) + (t^3 - t)/6 from the cached controlled walk, and its wires) without compiling
````

**R-S1** (2026-11-11-dfthc-pennylane-design.md, 13.3, the reason codes (twelve rows after `cmp-free`, and the listing rule)). Old:

````text
| `cmp-free` | PennyLane's comparators uncompute by measurement, so their adjoints cost no Toffoli |

The report never adjusts measured numbers.
````

New:

````text
| `cmp-free` | PennyLane's comparators uncompute by measurement, so their adjoints cost no Toffoli |
| `maj-and` | The Majorana gate folds the success flags into two ANDs with $G_0$ ($2 + 2 + 1$ Toffolis); the paper's Fig. 4 controls its four gates on one AND of both flags ($1 + 2 + 4$) (§9.16) |
| `sf-enable` | The SF fix-up of `AngleErase` is enabled on $G_0$, one Toffoli more than dfthcbe's (§9.17) |
| `rot-add` | `PhaseGradientRotation` adds with one $m$-bit `SemiAdder`, $m - 1$ Toffolis; Qualtran's `ZPowConstViaPhaseGradient`, dfthcbe's rotation, costs $m - 2$ (§9.1) |
| `unif-odd` | dfthcbe's uniform preparation, $4n + 2s_u - 5$ on the $n = \lg l$ qubits of the odd part $l$ of $L$ (as ours), against the paper's $4\lg L$ (B24, B32, B52, B54) over all $\lg L$ qubits (§9.2) |
| `sign-bits` | The inner entry carries both sign bits, `sgn` and `d_sgn`; Table III evaluates its inner swaps with $b_2 = w_b + b_{k2}$ bits |
| `onehot-const` | A one-hot over $K$ values costs $K - 2$ Toffolis (§9.24); the paper's B50 and B53 count $K$, and its B47 counts $B$ for the $B + 1$ values of $b$ |
| `iter-const` | Truncated unary iteration over $L$ leaves with $c$ controls costs $L + c - 2$ Toffolis; the paper counts $L$, one per leaf (dfthcbe's README, "Constants") |
| `prefix-stops` | dfthcbe's walks over the tail rows $[RC, X_o)$ (the tail angle load and the angle erase's tail fix-up) stop once on each aligned block of the dead SF prefix $[0, RC)$, $s(RC)$ stops, $s$ counting the ones in binary (its README, "Zero rows"); the paper's (B41, and the fix-up $N$ of B50) count their $N$ leaves alone |
| `ref-target` | The reflection's multi-controlled $Z$ targets one of its $n_w$ pattern qubits ($n_w - 2$ Toffolis, §9.6); dfthcbe's uncontrolled $\mathrm{REF}_{\mathcal A\cup\mathcal B}$ and the paper's B55 add a phase target ($n_w - 1$) |
| `ref-unif` | The paper's $\mathrm{REF}_{T_2}$ (B11) also reflects the rotated qubit of its $\mathrm{Unif}(b)$, which the flagged uniform preparation does not have |
| `qpe-ctrl` | The paper's B55 books one Toffoli for the unary iteration on the QPE control in the walk (App. B.7.2, Eq. (128)); our walk has no QPE control (our controlled walk has it, so the controlled rows do not carry this code) |
| `ctrl-ref-t2` | The controlled walk also controls $\mathrm{REF}_{T_2}$ (+1 per step), so $C(W)^p$ is exact for every $p$; the paper's QPE step controls only $\mathrm{REF}_{\mathcal A\cup\mathcal B}$, which gives controlled even powers because $\mathrm{BE}^2 = I$ (App. B.7.2, Eq. (128); §9.23) |

A row's reason codes are the causes with a nonzero contribution to its delta, each with its contribution (`dfthc.costs_reference.delta_causes`): every block is split into parts that ours, dfthcbe's and the paper's constructions share, each part's difference goes to the code that names its mechanism, and a composite's contributions are its children's times their calls. A code whose contributions cancel within a row is not listed (the paper's tail fix-up, `attrib`, moves between two rows of the block encoding and nets to zero there); two different codes that cancel within a row are both listed (`ref-unif` $-1$ and `ctrl-ref-t2` $+1$ in the controlled $\mathrm{REF}_{T_2}$ against the paper). Each unary-iteration difference is split by mechanism (`dfthc.costs_reference.iteration_causes`) at an intermediate count (`iteration_counts`), an accounting convention and not a circuit: on the walks from address 0, Qualtran's truncated iteration ($L + c - 2$) over the live leaves alone, without stops on dead addresses; on the tail rows $[RC, X_o)$, our exact count. It is a valid circuit only where dfthcbe's walk has no stops (the outer read, the outer erase's fix-up and the SF fix-up), where it is that walk. Elsewhere dead addresses carry amplitude, which truncated iteration would route onto live leaves, and on the tail rows no truncated walk without stops on the dead SF prefix is valid. `exact-iter` is ours minus the intermediate, `no-prune` the intermediate minus dfthcbe's walk, which stops once on each aligned block of a dead interval (signed: negative where those stops cost more than the intermediate). Against the paper's one Toffoli per leaf, the intermediate minus the paper is `iter-const`, except on the tail walks, which reach the paper through dfthcbe's walk: ours minus dfthcbe as above, then dfthcbe's $s(RC)$ stops on the dead SF prefix (`prefix-stops`) and its truncated iteration's constant (`iter-const`). The report checks at every system that each row's contributions sum to its deltas exactly and fails otherwise. The report never adjusts measured numbers.
````

**R-S2** (2026-11-11-dfthc-pennylane-design.md, 13.1, the angle-erase row). Old:

````text
| Angle erase (×2) | 74 ¹ | 180 | 204 | same, for the fixups |
````

New:

````text
| Angle erase (×2) | 74 ¹ | 180 | 204 | exact boundaries in the fix-up walks (+14: `exact-iter` +4 over $r$, `no-prune` +10 over the tail rows), one-hot over 32 positions instead of 28 (+8, `onehot-pad`), the SF fix-up enabled on $G_0$ (+2, `sf-enable`) |
````

**R-S3** (2026-11-11-dfthc-pennylane-design.md, 13.1, the outer-unprepare row). Old:

````text
| Outer unprepare and erase | 61 | 57 | 45 | free comparator adjoint (`cmp-free`); exact boundaries (+3) |
````

New:

````text
| Outer unprepare and erase | 61 | 57 | 45 | free comparator adjoint (`cmp-free`; the saving the paper notes after its Eq. (127)); exact boundaries (+3) |
````

**R-S4** (2026-11-11-dfthc-pennylane-design.md, 13.1, the uniform-preparation row (its reason)). Old:

````text
measured comparator uncomputes and PennyLane's $(n-1)$-Toffoli classical comparator, less one Toffoli per rotation ($s - 1$ against Qualtran's $s - 2$) (`cmp-free`) |
````

New:

````text
measured comparator uncomputes and PennyLane's $(n-1)$-Toffoli classical comparator (`cmp-free`), offset by one Toffoli more per rotation ($s - 1$ against Qualtran's $s - 2$, `rot-add`) |
````

**R-S5** (2026-11-11-dfthc-pennylane-design.md, 13.1, the remaining terms). Old:

````text
- **Remaining terms: net 0.** Outer read +3, spin multiplexer +4, Majorana gate −4, reflections −3.
````

New:

````text
- **Remaining terms: net 0.** Outer read +3 (`exact-iter` +5, `iter-const` −2), spin multiplexer +4 (`mux`; Table III has no row, while App. B.3 and B.7.1 count 2 per select), Majorana gate −4 (`maj-and`), reflections −3 (`ref-unif` −1 on $\mathrm{REF}_{T_2}$; `ref-target` −1 and `qpe-ctrl` −1 on $\mathrm{REF}_{\mathcal A\cup\mathcal B}$).
````

**R-S6** (2026-11-11-dfthc-pennylane-design.md, 13.1, the last paragraph (its last sentence)). Old:

````text
The controlled walk costs two more Toffolis than the walk.
````

New:

````text
The controlled walk costs two more Toffolis than the walk. The paper's 9,997 is its QPE step, with the control on $\mathrm{REF}_{\mathcal A\cup\mathcal B}$ only (App. B.7.2, Eq. (128)): against our controlled walk, 9,795, the difference is −202, the block encoding's −202 with +1 for controlling $\mathrm{REF}_{T_2}$ too (`ctrl-ref-t2`, §9.23) and −1 for the controlled $\mathrm{REF}_{\mathcal A\cup\mathcal B}$ against B55 (`ref-target`); both carry the QPE control, so `qpe-ctrl` does not apply.
````

**R-S7** (2026-11-11-dfthc-pennylane-design.md, 13.3, the report rows). Old:

````text
Each report row contains:
- block and paper steps;
````

New:

````text
Each row is one program: an operator exactly as the walk or the controlled walk calls it (one row per distinct call, so `QROAMRead` has four), `PhaseFixup` and `C(ZeroReflection)` with their callers' arguments, `ControlledWalkPower` ($p = 1, 2$) and the QPE program. The partition rows, the children of the walks, block encodings and selects, times their calls per walk give the walk exactly. Each report row contains:
- block and paper steps;
````

**R-S8** (2026-11-11-dfthc-pennylane-design.md, 5.2, the scripts). Old:

````text
    make_dfthcbe_fixtures.py  generates reference fixtures from dfthcbe in a separate environment (§8.6)
````

New:

````text
    make_dfthcbe_fixtures.py  generates reference fixtures from dfthcbe in a separate environment (§8.6)
    make_report_fixture.py    dfthcbe's per-block Toffolis at the report's parameter sets, from its call graph (§13.3)
````

**R-S9** (2026-11-11-dfthc-pennylane-design.md, 8.3, block-bit defaults). Old:

````text
- **Block-bit defaults** come from `dfthc.costs.cheapest_block_bits`, which minimizes our closed-form Toffoli count, as dfthcbe does.
````

New:

````text
- **Block-bit defaults** come from `dfthc.costs.cheapest_block_bits`, which minimizes our closed-form Toffoli count, as dfthcbe does. The resource report gives every vendored system Table III's precisions ($b_{\mathrm{rot}} = b_{k1} = b_{k2} = 15$, $s_u = 7$) with these block bits (`dfthc.costs_reference.default_params`), its "default precisions" (§13.3).
````

**R-S10** (2026-11-11-dfthc-pennylane-design.md, 13.1, the angle-loads row). Old:

````text
| Angle loads (×2) | 648 | 654 | 704 | exact boundaries instead of aligned-block pruning (`exact-iter`, `no-prune`) |
````

New:

````text
| Angle loads (×2) | 648 | 654 | 704 | exact boundaries in the SF loads (+44, `exact-iter`); the tail loads without aligned-block pruning (+10, `no-prune`), less dfthcbe's $s(R) = 2$ stops per SF load for the identity row $b = B$ (−4, `no-prune`) |
````

**R-S11** (2026-11-11-dfthc-pennylane-design.md, 13.3, step 2, `fe2s2` and `fe4s4` (the end of the item)). Old:

````text
the others follow the G0 decision of §16, K-1, or keep their closed form, marked pending;
````

New:

````text
the others follow the G0 decision of §16, K-1, or keep their closed form, marked refused. A program's size is its gates from its operator's resource functions; a program no larger than some measured one of its kind takes the largest time and peak over all measurements of its kind, and a larger one is extrapolated (K-1's method). The first killed or failed compile stops the system's run: the remaining programs are recorded as skipped, never as measurements, and the orchestrator escalates;
````

**R-S12** (2026-11-11-dfthc-pennylane-design.md, 13.3, step 2, FeMoco-54). Old:

````text
   - at FeMoco-54: only the walk and the controlled walk, once each, as the last step of Phase 4. Every per-operator FeMoco-54 row is the closed form, validated exactly by the compiles above.
````

New:

````text
   - at FeMoco-54: only the walk and the controlled walk, once each, as the last step of Phase 4. Every per-operator FeMoco-54 row is the closed form, validated exactly by the compiles above. If the walk is refused, killed or fails, the run stops before the controlled walk, which is recorded as skipped, and the orchestrator escalates to the user (§16, "Where compiles above the cap run").
````

**R-S13** (2026-11-11-dfthc-pennylane-design.md, 13.3, step 5, the code hash). Old:

````text
a hash of the code that ignores docstrings and comments
````

New:

````text
a hash of the code (the package outside `dfthc.testing`, `scripts/resource_report.py`, which builds the programs and selects the cache's entries, and the lock file) that ignores docstrings and comments
````

**R-S14** (2026-11-11-dfthc-pennylane-design.md, 13.1, the inner-unprepare row (its fix-up)). Old:

````text
exact iteration in the fix-up (+1 per erase, `exact-iter`)
````

New:

````text
exact boundaries in the fix-up (+4 per erase, `exact-iter`), less dfthcbe's 3 stops on the dead tail rows, which ours does not visit (−3 per erase, `no-prune`)
````

**R-S15** (2026-11-11-dfthc-pennylane-design.md, 13.1, the inner-read row (its iteration)). Old:

````text
exact boundaries +5 per read, less dfthcbe's 4 stops on the zero D1/Q1 rows, which ours does not visit (net +1 per read)
````

New:

````text
exact boundaries +5 per read (`exact-iter`), less dfthcbe's 4 stops on the zero D1/Q1 rows, which ours does not visit (`no-prune`; net +1 per read)
````

**R-H1** (HANDOFF.md, section 3.4, who uses `.venv-ref`). Old:

````text
Only Task 1A.6 (and later refreshes) uses it, to run `scripts/make_dfthcbe_fixtures.py`.
````

New:

````text
Only Tasks 1A.6 and 4.3 (and later refreshes) use it, to run `scripts/make_dfthcbe_fixtures.py` and `scripts/make_report_fixture.py`.
````

**R-H2** (HANDOFF.md, section 2, where things are). Old:

````text
| Scratch (never committed) | `.scratch/` |
````

New:

````text
| Resource report | `scripts/resource_report.py` writes `docs/resources.md` and `docs/resources.json` (spec 13.3) |
| Scratch (never committed) | `.scratch/` |
````

**R-H3** (HANDOFF.md, section 4, commands). Old:

````text
| Spike | `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/<script>.py` |
````

New:

````text
| Spike | `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/<script>.py` |
| Resource report (H2 systems and `naphthalene`; reuses current compiles) | `PYTHONPATH=src "$VENV/bin/python" scripts/resource_report.py` |
| Large compiles of the report (orchestrator only, alone) | `PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 120 -- "$VENV/bin/python" scripts/resource_report.py --large fe2s2` (then `fe4s4`; `femoco54` in Task 4.5) |
````

## Shared edits

No merged module changes. Task 4.3 adds one file to Task 1A.6's fixture directory, `tests/fixtures/dfthcbe/report_toffolis.json`, and touches neither `scripts/make_dfthcbe_fixtures.py` nor its test (whose fixture list is exact). Task 4.4 edits `docs/limitations.md` (Tasks 0.3 and 0.4's) and `docs/environment.md` (Task 0.1's), both in its file list. No other plan template changes, and the shared Global Constraints fragment does not change (this plan's template only dropped its redundant heading in revision 2): the other plans assembled into `.scratch/plan-check/out-p4r/` with this template added are byte-identical to their counterparts on `main`.

## How this plan meets the binding ledger rows

| Ledger row or decision | What this plan does |
| --- | --- |
| Per-block FeMoco comparisons 2D.1, 2D.2, 2E.x (2E.2, 2E.3, 2E.4, 2E.5), 3.1, 3.2, 3.3, 3.4 ("input to Task 4.3") | Every block of those rows is a report row with dfthcbe's and the paper's values per call, both from `costs_reference`, at every system. At FeMoco-54 they reproduce the rows' numbers: outer pair 210 / 69 against 212 / 94 and 213 / 97; inner pair 971 / 217 against 913 / 220 and 880 / 274 (220 at step 18); select 3,537 against 3,502 and 3,656 (3,710); tail and SF loads 61, 291 against 56, 271 and 54, 270; angle erase 102 against 90 and 37 (91); ladder 1,484 against 1,484 and 1,590; Majorana 5 against 5 and 7; `RefT2` 20 / 21 against 20 / 21 and 21; `RefAB` 44 / 45 against 45 / 45 and 46; block encoding 9,749 against 9,596 and 9,951; walk 9,793 against 9,641 and 9,997 (`test_dfthcbe_blocks_equal_the_call_graph`, `test_paper_table3_reproduces_table_iii`). Each row's causes are its reason codes with their contributions (section "Reason-code decomposition at FeMoco-54"); the FeMoco-54 summary of the report gives the per-walk totals. |
| Reason codes spec 13.3 lacks: Majorana −2, SF fix-up enable +1, REF_T2 −1, reflections −1 / −2, the angle-erase reason, `maj-and`, controlled walk +1 per step | Change 7: `maj-and`, `sf-enable`, `ref-unif`, `ref-target`, `qpe-ctrl`, `ctrl-ref-t2` and six more (R-S1); the angle-erase reason with the review's split, `exact-iter` +4 and `no-prune` +10 per walk (R-S2); every delta of every row at every system equals the sum of its codes' contributions, or the script fails (`test_report_fails_on_mismatch` removes one and sees the failure); the FeMoco-54 codes of the reflections, block encodings, walks and powers are asserted exactly (`test_reason_codes_at_femoco`). |
| M versus A (dfthcbe's Qualtran `measurement` is our A, not M; rows 2E.4, 2E.5, 3.1, 3.2, 3.3) | The report shows our M and A and dfthcbe's and the paper's T only; its header says that dfthcbe's "measurement" count is our A and that the paper itemizes neither, so no zero is shown for a missing split. |
| Both paper mux figures (2E.3) | `paper_table3` has `spin_mux` (Table III: 0) and `spin_mux_app_b3` (App. B.3 and B.7.1: 1 per application); the report's FeMoco-54 summary states both, R-S5 too. |
| `ob = w_b − 1` (2E.2) and retained swap ANDs (2D.1, 2D.2) | Not adopted (R-7: refinements of our own construction, which the user decides with the deferred items): the report's section "Deferred refinements (decided by the user)" lists them with their savings (12 and 19 Toffolis per walk) beside truncated iteration (up to 100) and Q-1 (about 460 qubits), for the Q-1 decision at G4. |
| The paper's 9,997 as its QPE step against our controlled walk (3.4) | The FeMoco-54 summary and R-S6: walk −204, controlled walk −202 (the block encoding's −202, `ctrl-ref-t2` +1, `ref-target` −1; no `qpe-ctrl`, since both carry the QPE control); `ControlledWalkPower(p)` rows compare with p × 9,997. |
| K-1 rows (captured-equation projections; A-M3's correction to 7,449 and 4,795) | R-M3 corrects item 23; the FeMoco-54 projection (section "Projections") uses both the report's gate model and the captured-equation model. |
| 1C.1 (dfthcbe's rotation s − 2, "Task 4.3 checks it and names the term") | Checked in `.venv-ref`: `ZPowConstViaPhaseGradient` costs s − 2 at s = 3, 4, 7, 15; named `rot-add` (R-S1, R-S4). |
| 1C.2, 1C.3, 1C.4, 1C.5, 2D.1, 2D.2 (uniform preparations, reads, erases, measured qubits, persistent qubits) | Rows `FlaggedUniformPrep@...`, `QROAMRead@...`, `QROAMErase@...`, `PhaseFixup@...` at every system; the FeMoco-54 summary's qubit section gives the groups (outer 143, inner 455), the junk wires (462) and the paper's 1,131 with `junk-qubits`, `cmp-free` and `flat-r`. |
| 1C.3 (Clifford counts of reads are upper bounds) | Each row's Clifford total carries its kind: "upper bound (CNOT, PauliX)" where reads, resets or QPE corrections occur, "expected, hinted (CZ, PauliZ)" where erase fix-ups occur, else "exact" (spec 7.6). |
| 1A.6 (Task 4.3 reads `femoco_block_toffolis.json`) | `test_dfthcbe_blocks_equal_the_call_graph` compares the port with it. |
| 2E.4 (the FeMoco-width rows can be compiled for the report) | The table-free rows compile at FeMoco-54 width only inside `--large femoco54`'s walks; per spec 13.3 step 2 FeMoco-54 compiles only the walks, and every per-operator FeMoco row is its closed form. |
| 0.9d Opus M-4 (2.5 MB per bind; "noted for Task 4.5's projection") and 1C.5 Minor 3 (9,000 equations, "Task 4.5 if compile memory gets tight") | Task 4.5's projection (4.5 GB by gates, about 1.9 GB by equations) leaves room; neither change is needed. |
| Heavy-runs table | Task 4.3's `--large fe2s2` and `--large fe4s4` and Task 4.5's `--large femoco54` are orchestrator runs, each with a ledger row (command, wall time, peak, killed, large-compile count: 0, 0 and 2); this plan's dry run started no heavy process. |
| Decision "Where compiles above the cap run" and "K-1 at G0" | `--large` refuses anything projected above 7.5 GB or 60 min, and stops a system's run at its first killed or failed compile (at FeMoco-54 also at its first refusal), recording the rest as skipped; Task 4.5 then escalates to the user (a machine with at least 32 GB, or an approved window). |
| D-3, D-6 (master plan section 14) | D-3: Task 4.3's report with every delta explained, completed by Task 4.5's compile and cache. D-6: Task 4.4's documents and tests. |
| Q-1 and K-9 | The report's qubit section feeds the Q-1 decision after Task 4.4; Q1b, if chosen, regenerates the report (a code change changes the hash) before Task 4.5. |

## The report as built

- **Systems** (spec 13.3 step 1): `h2_sto3g_R2_B2_C1`, `h2_631g_R2_B2_C2`, `h2_631g_R2_B3_C3`, `naphthalene` (compiled in every run that lacks a current entry), `fe2s2`, `fe4s4` (with `--large`), `femoco54` (`random_factors(54, 10, 27, 27, n_plus=27, seed=0)`; the walks with `--large femoco54`). `H2_STO3G_TEST` is accepted by `--systems` for the tests only.
- **Rows** (changes 1 and 2): the distinct calls of the walk, then the controlled walk's own (`ControlledDFTHCWalk`, `ControlledDFTHCBlockEncoding`, `ControlledRefT2`, `ControlledRefAB`), then `PhaseFixup@...` and `C(ZeroReflection)@...`, then `ControlledWalkPower(p=1)`, `(p=2)` and, at the H2 systems and FeMoco-54, `QPE(t=1)` to `(t=3)`. Each row's program rebuilds its call on renumbered wires; the walks are the programs of Task 3.4 (with the phase gradient), the L4 programs those of the QPE plan's interface section.
- **Columns** (spec 13.3): level, paper steps, calls per walk and per controlled walk (P marks the partition), T / M / A measured, T / M / A closed form, status (`measured`, `pending`, `refused`, `killed`, `failed`, `skipped`, or `closed form` where the report does not compile the row), dfthcbe, paper (with the step-18 attribution where it differs), the two deltas, the reason codes with their contributions, Cliffords with their kind, qubits (the wires the call binds; the walks' are `WalkLayout`'s), compile seconds, peak MB.
- **Checks** (spec 13.3 step 4): measured T, M, A equal the closed form; each delta equals the sum of its codes' contributions (`delta_causes`); the partition sums to the walks (closed form; measured where measured); dfthcbe's and the paper's partitions sum to their walks. Failures print `CHECK FAILED` lines and exit 1 after both files are written.
- **Cache** (spec 13.3 step 5): `docs/resources.json` keeps `"compiled"`: per `system/row`, a list of entries (code hash, environment, status, counts, compile seconds, the watched run's summary, gates, family, and for `--large` the projection). The newest entry of the current hash with status `compiled` is the measurement; every other entry is history, counted in the report and never shown as a measurement.
- **Large compiles**: `--large SYSTEM` sizes each program by its gates (`gates`, from the resource functions of its operator) and projects it from the current entries of its family at every system (`projection`, `project_compile`: where some entry of the family is at least as large, the largest time and peak over all its entries, else Task 0.9's method), compiles only within 7.5 GB and 60 minutes, one subprocess each under `run_watched` with the 8 GB cap, and records refusals and kills. The first `killed` or `failed` result stops the system's run, and at FeMoco-54 so does the first `refused`: the remaining programs are recorded as `skipped` and the log line names the escalation.
- **Sections of `docs/resources.md`**: one table per system; the FeMoco-54 summary (per-walk totals against dfthcbe and the paper, M + A, the attribution and mux notes, qubits and Q-1, the QPE at t = 1, 2, 3, 10, 20, 23 from the controlled walk, with T, M, A and qubits, the deferred refinements); the large-compile log; the reason codes.

## Reason-code decomposition at FeMoco-54

`costs_reference.delta_causes(p)` splits each block into parts that the three constructions share and gives each part's difference to one code: the uniform preparation's comparators (`cmp-free`) and rotations (`rot-add`) against dfthcbe, and dfthcbe's uniform formula against the paper's 4 lg L (`unif-odd`); every unary iteration split by `iteration_causes` at the intermediate of `iteration_counts` (below) into exact against truncated iteration (`exact-iter`), no aligned-block pruning (`no-prune`) and, against the paper, truncated iteration's constants (`iter-const`) and, on the tail walks, dfthcbe's stops on the dead SF prefix (`prefix-stops`); the inner entry's `r` (`flat-r`) and sign bits (`sign-bits`); the one-hots' padding (`onehot-pad`) and constants (`onehot-const`); the SF fix-up's enable (`sf-enable`); the tail fix-up the paper charges to the inner erase (`attrib`); the multiplexer (`mux`), the Givens rotations (`rz-addsub`), the Majorana gate (`maj-and`); the reflections' target (`ref-target`), the paper's extra reflected qubit (`ref-unif`) and QPE control (`qpe-ctrl`), and our controlled `REF_T2` (`ctrl-ref-t2`). A composite's contributions are its children's, times their calls. **Rule:** a row lists every code with a nonzero contribution and no other; a code whose contributions cancel within a row is not listed (`attrib`: −54 twice in the inner unprepares, +54 twice in the selects, so 0 in the block encoding), while two different codes that cancel are both listed (`ControlledRefT2` against the paper: `ref-unif` −1, `ctrl-ref-t2` +1). The report checks at every system that each row's contributions sum to its deltas exactly; the dry run passes at all eight parameter sets.

**The iteration intermediate, an accounting convention.** `costs_reference.iteration_counts(p)` gives every unary iteration as ours (exact, spec 9.24), an intermediate, dfthcbe's walk and the paper's count; `costs_reference.iteration_causes(p)` splits the differences. dfthcbe's walk truncates and prunes: Qualtran's truncated iteration (`truncated_unary`: L + c − 2) over its live leaves and one stop on each aligned block of a dead interval that the register can hold (its README, "Zero rows"). The intermediate is a cost model, not a circuit: on the walks from address 0, truncated iteration over the live leaves alone (nested walks nested), without stops; on the tail rows `[RC, X_o)`, our exact count. It coincides with a valid construction only where dfthcbe's walk has no stops: on the outer read, the outer erase's fix-up and the SF fix-up it is that walk (equal at all eight parameter sets, `test_dfthcbe_blocks_equal_the_call_graph`). It does not elsewhere, because truncated iteration routes a dead address that carries amplitude onto a live leaf: the intermediate leaves out dfthcbe's 4 stops on the zero D1/Q1 rows in the inner read, its 3 on the dead tail rows in the inner erase's fix-up and its s(R) = 2 for the identity row b = B in the SF load, which makes `no-prune` negative there. On the tail rows the dead SF prefix `[0, RC)` carries amplitude and no truncated walk without stops on it is valid (Qualtran's truncated call graph on `[270, 324)` at FeMoco-54, 53 Toffolis by the review's count, assumes an in-range selection), so the convention takes our exact count: `exact-iter` is 0 and the whole difference to dfthcbe is `no-prune`. Then `exact-iter` = ours − intermediate and `no-prune` = intermediate − dfthcbe; against the paper, `iter-const` = intermediate − paper on the walks from address 0, while the tail walks reach the paper through dfthcbe's walk: ours − dfthcbe as above, plus dfthcbe − paper by dfthcbe's mechanisms, its s(RC) stops on the dead SF prefix (`prefix-stops`) and its truncated constant over the N + s(RC) leaves and stops (`iter-const`). dfthcbe's own call graph gives its tail walk as exactly N + s(RC) − 2 at all eight parameter sets (the fixture; `test_dfthcbe_blocks_equal_the_call_graph`), so that split is pinned, not fitted. At FeMoco-54 (per call; generated with the table below):

| Unary iteration | Ours (exact) | Intermediate | dfthcbe | Paper | Intermediate as a circuit | `exact-iter` | `no-prune` | Against the paper |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| outer read (`QROAMRead@OuterPrepare`) | 84 | 79 | 79 | 81 | dfthcbe's walk | +5 | +0 | `exact-iter` +5, `iter-const` -2 |
| inner read, nested over `x_o` and the block address | 543 | 538 | 542 | 540 | none: leaves out dfthcbe's stops on the zero D1/Q1 rows | +5 | -4 | `exact-iter` +5, `iter-const` -2 |
| outer erase fix-up (high address) | 22 | 19 | 19 | 21 | dfthcbe's walk | +3 | +0 | `exact-iter` +3, `iter-const` -2 |
| inner erase fix-up (high address) | 70 | 66 | 69 | 68 | none: leaves out dfthcbe's stops on the dead tail rows | +4 | -3 | `exact-iter` +4, `iter-const` -2 |
| tail rows `[RC, X_o)`: tail load, angle erase's tail fix-up | 61 | 61 | 56 | 54 | none valid without the prefix stops; our exact count by convention | +0 | +5 | `no-prune` +5, `prefix-stops` +4, `iter-const` -2 |
| SF load, nested over `r` and `b`, enabled on `G0` | 291 | 269 | 271 | 270 | none: leaves out dfthcbe's stops for the identity row `b = B` | +22 | -2 | `exact-iter` +22, `iter-const` -1 |
| angle erase's walk over `r` (without its enable) | 10 | 8 | 8 | 10 | dfthcbe's walk | +2 | +0 | `exact-iter` +2, `iter-const` -2 |

The table (per call; "Calls" is per walk and per controlled walk for the 18 partition rows; generated by `stages/4R/decomposition_4r.py`, which checks every sum):

| Row | Calls (W, C(W)) | Ours | dfthcbe | Paper | Δ dfthcbe | Δ paper | Codes against dfthcbe | Codes against the paper |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `OuterPrepare` | 1, 1 | 210 | 212 | 213 | -2 | -3 | `cmp-free` -9, `rot-add` +2, `exact-iter` +5 | `cmp-free` -9, `rot-add` +2, `unif-odd` +1, `exact-iter` +5, `iter-const` -2 |
| `OuterUnprepare` | 1, 1 | 69 | 94 | 97 | -25 | -28 | `cmp-free` -30, `rot-add` +2, `exact-iter` +3 | `cmp-free` -30, `rot-add` +2, `unif-odd` +1, `exact-iter` +3, `iter-const` -2, `onehot-const` -2 |
| `InnerPrepare` | 2, 2 | 971 | 913 | 880 | 58 | 91 | `cmp-free` -5, `rot-add` +2, `exact-iter` +5, `no-prune` -4, `flat-r` +60 | `cmp-free` -5, `rot-add` +2, `unif-odd` +1, `exact-iter` +5, `iter-const` -2, `flat-r` +60, `sign-bits` +30 |
| `InnerUnprepare` | 2, 2 | 217 | 220 | 274 | -3 | -57 | `cmp-free` -22, `rot-add` +2, `exact-iter` +4, `no-prune` -3, `onehot-pad` +16 | `cmp-free` -22, `rot-add` +2, `unif-odd` +1, `exact-iter` +4, `iter-const` -2, `onehot-pad` +16, `onehot-const` -2, `attrib` -54 |
| `InnerSigns` | 2, 2 | 0 | 0 | 0 | 0 | 0 | — | — |
| `TailAngleLoad` | 2, 2 | 61 | 56 | 54 | 5 | 7 | `no-prune` +5 | `no-prune` +5, `prefix-stops` +4, `iter-const` -2 |
| `SFAngleLoad` | 2, 2 | 291 | 271 | 270 | 20 | 21 | `exact-iter` +22, `no-prune` -2 | `exact-iter` +22, `iter-const` -1 |
| `SpinMux` | 4, 4 | 1 | 2 | 0 | -1 | 1 | `mux` -1 | `mux` +1 |
| `SpinSwap` | 4, 4 | 54 | 54 | 54 | 0 | 0 | — | — |
| `GivensLadder` | 2, 2 | 1,484 | 1,484 | 1,590 | 0 | -106 | — | `rz-addsub` -106 |
| `GivensLadderDagger` | 2, 2 | 1,484 | 1,484 | 1,590 | 0 | -106 | — | `rz-addsub` -106 |
| `MajoranaGate` | 1, 1 | 5 | 5 | 7 | 0 | -2 | — | `maj-and` -2 |
| `Adjoint(MajoranaGate)` | 1, 1 | 5 | 5 | 7 | 0 | -2 | — | `maj-and` -2 |
| `AngleErase` | 2, 2 | 102 | 90 | 37 | 12 | 65 | `sf-enable` +1, `onehot-pad` +4, `exact-iter` +2, `no-prune` +5 | `sf-enable` +1, `onehot-pad` +4, `onehot-const` -1, `attrib` +54, `exact-iter` +2, `iter-const` -4, `no-prune` +5, `prefix-stops` +4 |
| `RefT2` | 1, 0 | 20 | 20 | 21 | 0 | -1 | — | `ref-unif` -1 |
| `ControlledRefT2` | 0, 1 | 21 | 21 | 21 | 0 | 0 | — | `ref-unif` -1, `ctrl-ref-t2` +1 |
| `RefAB` | 1, 0 | 44 | 45 | 46 | -1 | -2 | `ref-target` -1 | `ref-target` -1, `qpe-ctrl` -1 |
| `ControlledRefAB` | 0, 1 | 45 | 45 | 46 | 0 | -1 | — | `ref-target` -1 |
| `InnerSelect` |  | 3,537 | 3,502 | 3,656 | 35 | -119 | `exact-iter` +24, `no-prune` +8, `mux` -2, `sf-enable` +1, `onehot-pad` +4 | `exact-iter` +24, `no-prune` +10, `prefix-stops` +8, `iter-const` -7, `mux` +2, `rz-addsub` -212, `maj-and` -2, `sf-enable` +1, `onehot-pad` +4, `onehot-const` -1, `attrib` +54 |
| `InnerSelectDagger` |  | 3,537 | 3,502 | 3,656 | 35 | -119 | `exact-iter` +24, `no-prune` +8, `mux` -2, `sf-enable` +1, `onehot-pad` +4 | `exact-iter` +24, `no-prune` +10, `prefix-stops` +8, `iter-const` -7, `mux` +2, `rz-addsub` -212, `maj-and` -2, `sf-enable` +1, `onehot-pad` +4, `onehot-const` -1, `attrib` +54 |
| `DFTHCBlockEncoding` |  | 9,749 | 9,596 | 9,951 | 153 | -202 | `cmp-free` -93, `rot-add` +12, `exact-iter` +74, `no-prune` +2, `flat-r` +120, `mux` -4, `sf-enable` +2, `onehot-pad` +40 | `cmp-free` -93, `rot-add` +12, `unif-odd` +6, `exact-iter` +74, `iter-const` -26, `onehot-const` -8, `flat-r` +120, `sign-bits` +60, `no-prune` +20, `prefix-stops` +16, `mux` +4, `rz-addsub` -424, `maj-and` -4, `sf-enable` +2, `onehot-pad` +40, `ref-unif` -1 |
| `ControlledDFTHCBlockEncoding` |  | 9,750 | 9,597 | 9,951 | 153 | -201 | `cmp-free` -93, `rot-add` +12, `exact-iter` +74, `no-prune` +2, `flat-r` +120, `mux` -4, `sf-enable` +2, `onehot-pad` +40 | `cmp-free` -93, `rot-add` +12, `unif-odd` +6, `exact-iter` +74, `iter-const` -26, `onehot-const` -8, `flat-r` +120, `sign-bits` +60, `no-prune` +20, `prefix-stops` +16, `mux` +4, `rz-addsub` -424, `maj-and` -4, `sf-enable` +2, `onehot-pad` +40, `ref-unif` -1, `ctrl-ref-t2` +1 |
| `DFTHCWalk` |  | 9,793 | 9,641 | 9,997 | 152 | -204 | `cmp-free` -93, `rot-add` +12, `exact-iter` +74, `no-prune` +2, `flat-r` +120, `mux` -4, `sf-enable` +2, `onehot-pad` +40, `ref-target` -1 | `cmp-free` -93, `rot-add` +12, `unif-odd` +6, `exact-iter` +74, `iter-const` -26, `onehot-const` -8, `flat-r` +120, `sign-bits` +60, `no-prune` +20, `prefix-stops` +16, `mux` +4, `rz-addsub` -424, `maj-and` -4, `sf-enable` +2, `onehot-pad` +40, `ref-unif` -1, `ref-target` -1, `qpe-ctrl` -1 |
| `ControlledDFTHCWalk` |  | 9,795 | 9,642 | 9,997 | 153 | -202 | `cmp-free` -93, `rot-add` +12, `exact-iter` +74, `no-prune` +2, `flat-r` +120, `mux` -4, `sf-enable` +2, `onehot-pad` +40 | `cmp-free` -93, `rot-add` +12, `unif-odd` +6, `exact-iter` +74, `iter-const` -26, `onehot-const` -8, `flat-r` +120, `sign-bits` +60, `no-prune` +20, `prefix-stops` +16, `mux` +4, `rz-addsub` -424, `maj-and` -4, `sf-enable` +2, `onehot-pad` +40, `ref-unif` -1, `ctrl-ref-t2` +1, `ref-target` -1 |
| `ControlledWalkPower(p=1)` |  | 9,795 | — | 9,997 | — | -202 | — | `cmp-free` -93, `rot-add` +12, `unif-odd` +6, `exact-iter` +74, `iter-const` -26, `onehot-const` -8, `flat-r` +120, `sign-bits` +60, `no-prune` +20, `prefix-stops` +16, `mux` +4, `rz-addsub` -424, `maj-and` -4, `sf-enable` +2, `onehot-pad` +40, `ref-unif` -1, `ctrl-ref-t2` +1, `ref-target` -1 |
| `ControlledWalkPower(p=2)` |  | 19,590 | — | 19,994 | — | -404 | — | `cmp-free` -186, `rot-add` +24, `unif-odd` +12, `exact-iter` +148, `iter-const` -52, `onehot-const` -16, `flat-r` +240, `sign-bits` +120, `no-prune` +40, `prefix-stops` +32, `mux` +8, `rz-addsub` -848, `maj-and` -8, `sf-enable` +4, `onehot-pad` +80, `ref-unif` -2, `ctrl-ref-t2` +2, `ref-target` -2 |

Against the paper, the walk's −204 is the block encoding's −202 with `ref-target` −1 and `qpe-ctrl` −1 from `RefAB`; the controlled walk's −202 is the controlled block encoding's −201 (the block encoding's −202 and `ctrl-ref-t2` +1) with `ref-target` −1 from `ControlledRefAB`, and no `qpe-ctrl`, because both sides carry the QPE control. Against dfthcbe, the controlled reflections cost the same (`ControlledRefT2` 21, `ControlledRefAB` 45), so the controlled walk's +153 is the block encoding's.

## Dry-run results

The report of the H2 systems and `naphthalene`, regenerated under revision 4's code hash (`report-small-systems.log`: 190 programs, all measured, exit 0, 1,253 s, process tree at most 765 MB, on a busier machine than revision 3's 808 s; the counts equal revision 3's, and only the reason codes of the rows with a tail walk changed, against the paper):

Key rows of the dry-run report (every row measured and equal to its closed form; T / M / A; dfthcbe's and the paper's Toffolis per call; the compile time inside the subprocess and the watched peak of the subprocess):

| System | Row | T / M / A | dfthcbe | Paper | Δ dfthcbe | Δ paper | Compile s | Peak MB |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `h2_sto3g_R2_B2_C1` | `DFTHCWalk` | 342 / 93 / 310 | 392 | 400 | -50 | -58 | 7.27 | 462.3 |
| `h2_sto3g_R2_B2_C1` | `ControlledDFTHCWalk` | 344 / 93 / 312 | 393 | 400 | -49 | -56 | 8.05 | 449.1 |
| `h2_sto3g_R2_B2_C1` | `DFTHCBlockEncoding` | 308 / 93 / 277 | 357 | 364 | -49 | -56 | 4.52 | 386.8 |
| `h2_sto3g_R2_B2_C1` | `InnerPrepare` | 39 / 0 / 21 | 40 | 31 | -1 | 8 | 3.25 | 276.9 |
| `h2_sto3g_R2_B2_C1` | `QROAMRead@InnerPrepare` | 7 / 0 / 7 | 6 | — | 1 | — | 2.53 | 251.6 |
| `h2_sto3g_R2_B2_C1` | `InnerSelect` | 77 / 18 / 68 | 76 | 81 | 1 | -4 | 3.86 | 296.3 |
| `h2_sto3g_R2_B2_C1` | `AngleErase` | 4 / 18 / 4 | 2 | 4 | 2 | 0 | 1.9 | 237.8 |
| `h2_sto3g_R2_B2_C1` | `RefAB` | 34 / 0 / 33 | 35 | 36 | -1 | -2 | 2.09 | 275.0 |
| `h2_sto3g_R2_B2_C1` | `ControlledWalkPower(p=2)` | 688 / 186 / 624 | — | 800 | — | -112 | 10.67 | 447.5 |
| `h2_sto3g_R2_B2_C1` | `QPE(t=3)` | 2,412 / 654 / 2,188 | — | — | — | — | 11.43 | 468.4 |
| `h2_631g_R2_B2_C2` | `DFTHCWalk` | 611 / 154 / 569 | 657 | 689 | -46 | -78 | 11.33 | 480.0 |
| `h2_631g_R2_B2_C2` | `ControlledDFTHCWalk` | 613 / 154 / 571 | 658 | 689 | -45 | -76 | 10.76 | 466.1 |
| `h2_631g_R2_B2_C2` | `DFTHCBlockEncoding` | 576 / 154 / 535 | 621 | 652 | -45 | -76 | 7.96 | 402.6 |
| `h2_631g_R2_B2_C2` | `InnerPrepare` | 47 / 0 / 29 | 46 | 37 | 1 | 10 | 4.55 | 269.2 |
| `h2_631g_R2_B2_C2` | `QROAMRead@InnerPrepare` | 15 / 0 / 15 | 12 | — | 3 | — | 3.04 | 246.0 |
| `h2_631g_R2_B2_C2` | `InnerSelect` | 197 / 48 / 184 | 196 | 207 | 1 | -10 | 4.52 | 300.3 |
| `h2_631g_R2_B2_C2` | `AngleErase` | 6 / 48 / 6 | 4 | 4 | 2 | 2 | 3.02 | 244.9 |
| `h2_631g_R2_B2_C2` | `RefAB` | 35 / 0 / 34 | 36 | 37 | -1 | -2 | 3.74 | 270.0 |
| `h2_631g_R2_B2_C2` | `ControlledWalkPower(p=2)` | 1,226 / 308 / 1,142 | — | 1,378 | — | -152 | 9.42 | 472.8 |
| `h2_631g_R2_B2_C2` | `QPE(t=3)` | 4,295 / 1,081 / 4,001 | — | — | — | — | 9.86 | 485.2 |
| `h2_631g_R2_B3_C3` | `DFTHCWalk` | 636 / 155 / 592 | 679 | 739 | -43 | -103 | 9.86 | 490.5 |
| `h2_631g_R2_B3_C3` | `ControlledDFTHCWalk` | 638 / 155 / 594 | 680 | 739 | -42 | -101 | 5.05 | 456.8 |
| `h2_631g_R2_B3_C3` | `DFTHCBlockEncoding` | 600 / 155 / 557 | 642 | 701 | -42 | -101 | 7.51 | 412.5 |
| `h2_631g_R2_B3_C3` | `InnerPrepare` | 41 / 0 / 24 | 41 | 49 | 0 | -8 | 3.05 | 261.4 |
| `h2_631g_R2_B3_C3` | `QROAMRead@InnerPrepare` | 24 / 0 / 24 | 24 | — | 0 | — | 1.79 | 245.2 |
| `h2_631g_R2_B3_C3` | `InnerSelect` | 205 / 48 / 192 | 201 | 210 | 4 | -5 | 2.45 | 294.3 |
| `h2_631g_R2_B3_C3` | `AngleErase` | 9 / 48 / 9 | 6 | 5 | 3 | 4 | 1.32 | 240.2 |
| `h2_631g_R2_B3_C3` | `RefAB` | 36 / 0 / 35 | 37 | 38 | -1 | -2 | 1.58 | 274.2 |
| `h2_631g_R2_B3_C3` | `ControlledWalkPower(p=2)` | 1,276 / 310 / 1,188 | — | 1,478 | — | -202 | 9.89 | 471.1 |
| `h2_631g_R2_B3_C3` | `QPE(t=3)` | 4,470 / 1,088 / 4,162 | — | — | — | — | 10.99 | 498.6 |
| `naphthalene` | `DFTHCWalk` | 1,913 / 502 / 1,678 | 1,937 | 1,965 | -24 | -52 | 15.47 | 633.6 |
| `naphthalene` | `ControlledDFTHCWalk` | 1,915 / 502 / 1,680 | 1,938 | 1,965 | -23 | -50 | 16.0 | 589.7 |
| `naphthalene` | `DFTHCBlockEncoding` | 1,874 / 502 / 1,640 | 1,897 | 1,924 | -23 | -50 | 12.9 | 525.6 |
| `naphthalene` | `InnerPrepare` | 185 / 0 / 96 | 179 | 164 | 6 | 21 | 3.25 | 297.4 |
| `naphthalene` | `QROAMRead@InnerPrepare` | 149 / 0 / 80 | 140 | — | 9 | — | 2.99 | 281.9 |
| `naphthalene` | `InnerSelect` | 625 / 138 / 600 | 612 | 639 | 13 | -14 | 4.82 | 349.6 |
| `naphthalene` | `AngleErase` | 26 / 138 / 26 | 21 | 14 | 5 | 12 | 2.93 | 261.7 |
| `naphthalene` | `RefAB` | 39 / 0 / 38 | 40 | 41 | -1 | -2 | 3.19 | 275.4 |
| `naphthalene` | `ControlledWalkPower(p=2)` | 3,830 / 1,004 / 3,360 | — | 3,930 | — | -100 | 16.34 | 601.3 |

| System | Rows | Status | Walk qubits | Compile s | Peak MB |
| --- | ---: | --- | ---: | --- | --- |
| `h2_sto3g_R2_B2_C1` | 48 | measured | 178 | 0.94 to 11.43 | 205.0 to 468.4 |
| `h2_631g_R2_B2_C2` | 48 | measured | 215 | 1.69 to 11.33 | 209.5 to 485.2 |
| `h2_631g_R2_B3_C3` | 48 | measured | 219 | 0.78 to 10.99 | 199.6 to 498.6 |
| `naphthalene` | 46 | measured | 426 | 1.0 to 16.34 | 200.3 to 633.6 |

`--large` on a stand-in (`probes/p4r/standin_large.sh`: `h2_631g_R2_B3_C3` as the large system; the entries of its walk, controlled walk and `InnerSelect` removed there and at `naphthalene`, so they extrapolate from the two smaller H2 systems, and its `SpinMux` entry removed, so it is bounded by the other systems' measurements), each compiled under the 8 GB cap:

| Row | Gates | Method | Projected s | Projected MB | Measured s (watched) | Measured MB | Cap MB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `DFTHCWalk` | 7,766 | k1 | 8.2 | 474.5 | 7.6 | 479.4 | 8192 |
| `ControlledDFTHCWalk` | 7,772 | k1 | 7.0 | 464.0 | 7.2 | 451.7 | 8192 |
| `InnerSelect` | 2,492 | k1 | 4.6 | 299.8 | 4.3 | 301.6 | 8192 |
| `SpinMux` | 4 | bounded | 3.0 | 212.2 | 2.9 | 206.4 | 8192 |

## Projections

Every program the report compiles at `fe2s2` and `fe4s4` (46 each) and FeMoco-54 (the two walks), projected from the dry-run report's 190 current entries through the report's own `gates`, `family` and `projection` (gates from each operator's resource functions; the largest time and peak over all the family's entries where one of them is at least as large, else Task 0.9's method), with nothing compiled at these sizes (`projections-large.log`; "pts" is the number of calibration points, "s" the projected watched seconds):

```text
calibration: 190 cached entries, 0 of another code hash

fe2s2: 46 programs
  row                                              gates  method pts       s       MB  verdict
  DFTHCWalk                                      296,917      k1   4   112.8   1561.8  allowed (within the limits)
  DFTHCBlockEncoding                             296,784      k1   4    99.4   1270.7  allowed (within the limits)
  OuterPrepare                                     3,013      k1   4    10.5    317.7  allowed (within the limits)
  FlaggedUniformPrep@OuterPrepare                    237      k1   4     4.3    263.1  allowed (within the limits)
  PhaseGradientRotation@FlaggedUniformPrep@OuterPrepare        53 bounded   5     6.0    244.7  allowed (the family's largest measurement)
  QROAMRead@OuterPrepare                           2,651      k1   4    10.0    256.3  allowed (within the limits)
  InnerPrepare                                    73,913      k1   4    50.3    539.7  allowed (within the limits)
  FlaggedUniformPrep@InnerPrepare                    187      k1   4     4.9    267.9  allowed (within the limits)
  PhaseGradientRotation@FlaggedUniformPrep@InnerPrepare        51 bounded   5     6.0    244.7  allowed (the family's largest measurement)
  QROAMRead@InnerPrepare                          73,607      k1   4    49.0    580.8  allowed (within the limits)
  InnerSigns                                           2 bounded   4     4.9    211.6  allowed (the family's largest measurement)
  InnerSelect                                     68,646      k1   4    35.3    583.3  allowed (within the limits)
  TailAngleLoad                                    5,807      k1   4    18.2    285.1  allowed (within the limits)
  QROAMRead@TailAngleLoad                          5,807      k1   4    18.2    243.8  allowed (within the limits)
  SFAngleLoad                                     50,311      k1   4    33.9    365.2  allowed (within the limits)
  QROAMRead@SFAngleLoad                           50,311      k1   4    33.9    407.6  allowed (within the limits)
  SpinMux                                              4 bounded   4     4.9    216.4  allowed (the family's largest measurement)
  SpinSwap                                            60      k1   4     6.0    217.3  allowed (within the limits)
  GivensLadder                                     5,700      k1   4     6.8    271.0  allowed (within the limits)
  GivensRotation                                     300 bounded   4     6.2    253.2  allowed (the family's largest measurement)
  MajoranaGate                                        18 bounded   4     5.5    242.1  allowed (the family's largest measurement)
  GivensLadderDagger                               5,700      k1   4    11.6    256.4  allowed (within the limits)
  Adjoint(GivensRotation)                            300 bounded   4     6.1    255.6  allowed (the family's largest measurement)
  AngleErase                                         982      k1   4    13.5    298.9  allowed (within the limits)
  InnerUnprepare                                   3,977      k1   4    33.1    524.1  allowed (within the limits)
  QROAMErase@InnerUnprepare                        3,671      k1   4    40.5    462.1  allowed (within the limits)
  Adjoint(FlaggedUniformPrep)@InnerUnprepare         187      k1   4     6.5    268.5  allowed (within the limits)
  RefT2                                               65      k1   4     6.4    264.1  allowed (within the limits)
  ZeroReflection@RefT2                                65      k1   4     6.6    261.1  allowed (within the limits)
  InnerSelectDagger                               68,646      k1   4    44.1    575.9  allowed (within the limits)
  Adjoint(MajoranaGate)                               18 bounded   4     6.0    234.3  allowed (the family's largest measurement)
  OuterUnprepare                                     630      k1   4    11.3    293.0  allowed (within the limits)
  QROAMErase@OuterUnprepare                          268      k1   4     8.6    263.1  allowed (within the limits)
  Adjoint(FlaggedUniformPrep)@OuterUnprepare         237      k1   4     9.7    264.0  allowed (within the limits)
  RefAB                                              133      k1   4    11.5    276.3  allowed (within the limits)
  ZeroReflection@RefAB                               133      k1   4    12.6    263.9  allowed (within the limits)
  ControlledDFTHCWalk                            296,923      k1   4   118.9   1372.1  allowed (within the limits)
  ControlledDFTHCBlockEncoding                   296,787      k1   4   102.5   1370.4  allowed (within the limits)
  ControlledRefT2                                     68      k1   4     7.0    259.9  allowed (within the limits)
  ControlledRefAB                                    136      k1   4    13.0    257.6  allowed (within the limits)
  PhaseFixup@InnerUnprepare                        3,271      k1   4    54.8    374.5  allowed (within the limits)
  C(ZeroReflection)@RefT2                             68      k1   4    13.3    354.1  allowed (within the limits)
  PhaseFixup@OuterUnprepare                          180      k1   4    11.8    267.8  allowed (within the limits)
  C(ZeroReflection)@RefAB                            136      k1   4    24.8    383.9  allowed (within the limits)
  ControlledWalkPower(p=1)                       296,923      k1   4   120.8   1460.7  allowed (within the limits)
  ControlledWalkPower(p=2)                       593,846      k1   4   122.6   1420.9  allowed (within the limits)
  fe2s2: 46 of 46 allowed (8 bounded, 38 extrapolated), refused: []; largest peak DFTHCWalk 1561.8 MB, longest ControlledWalkPower(p=2) 122.6 s, total 22.4 min

fe4s4: 46 programs
  row                                              gates  method pts       s       MB  verdict
  DFTHCWalk                                      566,743      k1   4   215.2   2568.0  allowed (within the limits)
  DFTHCBlockEncoding                             566,610      k1   4   189.8   2078.3  allowed (within the limits)
  OuterPrepare                                     3,815      k1   4    13.3    335.1  allowed (within the limits)
  FlaggedUniformPrep@OuterPrepare                    282      k1   4     5.1    273.6  allowed (within the limits)
  PhaseGradientRotation@FlaggedUniformPrep@OuterPrepare        53 bounded   5     6.0    244.7  allowed (the family's largest measurement)
  QROAMRead@OuterPrepare                           3,405      k1   4    12.9    259.6  allowed (within the limits)
  InnerPrepare                                    44,194      k1   4    30.1    430.2  allowed (within the limits)
  FlaggedUniformPrep@InnerPrepare                    142 bounded   4     6.3    258.9  allowed (the family's largest measurement)
  PhaseGradientRotation@FlaggedUniformPrep@InnerPrepare        51 bounded   5     6.0    244.7  allowed (the family's largest measurement)
  QROAMRead@InnerPrepare                          43,936      k1   4    29.3    445.9  allowed (within the limits)
  InnerSigns                                           2 bounded   4     4.9    211.6  allowed (the family's largest measurement)
  InnerSelect                                    234,251      k1   4   120.5   1293.2  allowed (within the limits)
  TailAngleLoad                                   19,094      k1   4    59.8    377.0  allowed (within the limits)
  QROAMRead@TailAngleLoad                         19,094      k1   4    59.8    243.8  allowed (within the limits)
  SFAngleLoad                                    192,168      k1   4   129.3    711.9  allowed (within the limits)
  QROAMRead@SFAngleLoad                          192,168      k1   4   129.3    875.9  allowed (within the limits)
  SpinMux                                              4 bounded   4     4.9    216.4  allowed (the family's largest measurement)
  SpinSwap                                           108      k1   4    10.8    217.3  allowed (within the limits)
  GivensLadder                                    10,500      k1   4    12.4    290.9  allowed (within the limits)
  GivensRotation                                     300 bounded   4     6.2    253.2  allowed (the family's largest measurement)
  MajoranaGate                                        18 bounded   4     5.5    242.1  allowed (the family's largest measurement)
  GivensLadderDagger                              10,500      k1   4    21.4    256.4  allowed (within the limits)
  Adjoint(GivensRotation)                            300 bounded   4     6.1    255.6  allowed (the family's largest measurement)
  AngleErase                                       1,747      k1   4    24.0    349.8  allowed (within the limits)
  InnerUnprepare                                   2,555      k1   4    21.2    423.1  allowed (within the limits)
  QROAMErase@InnerUnprepare                        2,297      k1   4    25.3    377.6  allowed (within the limits)
  Adjoint(FlaggedUniformPrep)@InnerUnprepare         142 bounded   4     6.1    257.9  allowed (the family's largest measurement)
  RefT2                                               62      k1   4     6.1    264.1  allowed (within the limits)
  ZeroReflection@RefT2                                62      k1   4     6.3    260.7  allowed (within the limits)
  InnerSelectDagger                              234,251      k1   4   150.6   1273.8  allowed (within the limits)
  Adjoint(MajoranaGate)                               18 bounded   4     6.0    234.3  allowed (the family's largest measurement)
  OuterUnprepare                                     729      k1   4    15.7    302.0  allowed (within the limits)
  QROAMErase@OuterUnprepare                          319      k1   4    10.3    271.0  allowed (within the limits)
  Adjoint(FlaggedUniformPrep)@OuterUnprepare         282      k1   4    17.4    274.0  allowed (within the limits)
  RefAB                                              133      k1   4    11.5    276.3  allowed (within the limits)
  ZeroReflection@RefAB                               133      k1   4    12.6    263.9  allowed (within the limits)
  ControlledDFTHCWalk                            566,749      k1   4   227.0   2221.0  allowed (within the limits)
  ControlledDFTHCBlockEncoding                   566,613      k1   4   195.7   2282.1  allowed (within the limits)
  ControlledRefT2                                     65      k1   4     6.7    257.6  allowed (within the limits)
  ControlledRefAB                                    136      k1   4    13.0    257.6  allowed (within the limits)
  PhaseFixup@InnerUnprepare                        1,865      k1   4    31.2    315.8  allowed (within the limits)
  C(ZeroReflection)@RefT2                             65      k1   4    12.6    354.1  allowed (within the limits)
  PhaseFixup@OuterUnprepare                          227      k1   4    14.9    279.0  allowed (within the limits)
  C(ZeroReflection)@RefAB                            136      k1   4    24.8    383.9  allowed (within the limits)
  ControlledWalkPower(p=1)                       566,749      k1   4   230.5   2390.8  allowed (within the limits)
  ControlledWalkPower(p=2)                     1,133,498      k1   4   234.0   2309.3  allowed (within the limits)
  fe4s4: 46 of 46 allowed (10 bounded, 36 extrapolated), refused: []; largest peak DFTHCWalk 2568.0 MB, longest ControlledWalkPower(p=2) 234.0 s, total 39.8 min

femoco54: 2 programs
  row                                              gates  method pts       s       MB  verdict
  DFTHCWalk                                    1,073,044      k1   4   407.5   4456.1  allowed (within the limits)
  ControlledDFTHCWalk                          1,073,050      k1   4   429.7   3813.8  allowed (within the limits)
  femoco54: 2 of 2 allowed (0 bounded, 2 extrapolated), refused: []; largest peak DFTHCWalk 4456.1 MB, longest ControlledDFTHCWalk 429.7 s, total 14.0 min
WATCHDOG {"command": ["env", "PYTHONPATH=src", "/Users/david.wierichs/repos/dfthc-be/.venv/bin/python", "../probes/p4r/project_large.py", "../tmp/p4r/out-report/resources.json"], "pid": 6574, "returncode": 0, "seconds": 14.8, "peak_mb": 177.0, "samples": 280, "killed": null, "survivors": [], "cap_mb": 3000.0, "timeout_s": 600.0, "priority": "taskpolicy -c utility", "heavy": false}
```

- **`fe2s2` and `fe4s4`: all 92 programs project within the limits; none is refused.** At `fe2s2`, 8 take their family's largest measurement, blocks whose size does not grow with the system (the two `PhaseGradientRotation` rows, `InnerSigns`, `SpinMux`, `GivensRotation`, `MajoranaGate` and their adjoints), and 38 extrapolated; at `fe4s4`, 10 and 36 (also the inner uniform preparation and its adjoint: 142 gates, against 163 at `naphthalene`). The largest peaks are the walks', 1.6 GB (113 s) at `fe2s2` and 2.6 GB (215 s) at `fe4s4`, and the longest times `ControlledWalkPower(p=2)`'s (an operator looping over the controlled walk), 123 s (1.4 GB) and 234 s (2.3 GB); the controlled walks 1.4 and 2.2 GB. These come from the report regenerated under revision 4's code hash on a busy machine (1,253 s against revision 3's 808 s), so the times are about 1.7 times revision 3's (at most 1.4 GB and 71 s, and 2.3 GB and 135 s), and the memory slopes moved with the noisier small-system peaks (the `naphthalene` walk 634 MB against 613 MB); the bounded rows take at most 6.3 s and 259 MB. The K-1 stand-in had projected 4 to 6 GB and 6 to 9 GB (master plan section 3, item 9): the production walks are leaner (per-leaf loads, explicit classes, registers passed by name). The projected watched seconds sum to 22 minutes at `fe2s2` and 40 minutes at `fe4s4`. A bounded projection carries the measurements' noise: the stand-in's `SpinMux` compiled at 2.9 s and 206 MB against its bound in revision 3's cache, 3.0 s and 212 MB, while in revision 2's run it came out 5 MB and 0.2 s above its then bound.
- **FeMoco-54** (Task 4.5): from the four small systems the walk projects to 4.5 GB and 408 s and the controlled walk to 3.8 GB and 430 s, within 7.5 GB and 60 minutes (from revision 3's quieter timings: 3.9 GB and 241 s, 3.9 GB and 245 s). The gate measure overstates the compile: with per-leaf loads the compile grows with the captured equations, not with the loaded bits that the gates count. The captured-equation model of ledger rows 2E.4, 3.1, 3.3 and 3.4 (the PX stand-in compiled at FeMoco-54 in 51 s at 2,648 MB with 390,949 equations; the production walk saves about 58,800 in the assembly targets, 42,000 in the selects and 15,800 in the ladders, about 274,000 left) gives about 1.9 GB. Task 4.5's projection adds the `fe2s2` and `fe4s4` points. If it still exceeded 7.5 GB or 60 minutes, `--large femoco54` would refuse the walk, skip the controlled walk, and the orchestrator would escalate to the user (section 2 of the master plan, "Where compiles above the cap run").

## Docstring audit

`tests/test_docstrings.py` on the replica audits the 32 `Operator2` classes of `dfthc.ops` (30) and `dfthc.qpe` (2): every class has a summary line and the nine sections of spec 9.0 in order, each once (`**Action**`, `**Registers.**`, `**Compilable arguments.**` or, for `QROAMRead` and `PhaseFixup`, `**Arguments.**`, `**Pre- and postconditions.**`, `**Decomposition**`, `**Adjoint and control.**`, `**Cost.**`, `**Paper and dfthcbe.**`, `**Example.**`), its cost section names only existing `dfthc.costs` functions, at least one, and its example calls `compiled_resources`. No class fails (`docstring-audit.log`), so no docstring changes. Seven mutants (a section renamed, one out of order, a cost function renamed, an example without its compile, a broken link, an operator missing from the architecture, a limitation without an entry) each fail the tests (`docstring-mutants.log`).

## Expected test counts

The per-commit tier (`scripts/test.sh -m "not slow and not release" -q`) counts **1112 tests on `main` with Tasks 4.1 and 4.2** (the QPE plan's revision 4: 1111 on `main` at `7e0a472` with Task 4.1 merged, plus Task 4.2's one; revision 4's replay at `5fe4d65`, with the QPE plan's revision 3 files, measured `1110/1219 tests collected (109 deselected)`), the nightly tier 75. The pass criterion is "no failures, no errors, no warnings summary"; the number is a cross-check, and the orchestrator shifts the base if other tasks merge first.

| Task | Base | Own per-commit | Expected per-commit tier | Own nightly and release tests |
| --- | ---: | ---: | ---: | --- |
| 4.3 | 1112 | 63 (`test_report.py` 63 of 64) | 1175 = 1112 + 63 | 1 `slow` (`test_report_mid`) |
| 4.4 | 1175 | 46 (`test_docstrings.py`) | 1221 = 1175 + 46 | none |
| 4.5 | 1221 | 1 (`test_femoco_cache.py` 1 of 2) | 1222 = 1221 + 1 | 1 `release` (`test_femoco_cache`) |

Revision 2 added four per-commit tests to Task 4.3 (`test_every_large_row_projects`, `test_large_stops_after_a_refusal_at_femoco`, `test_large_stops_after_a_kill`, `test_reason_codes_at_femoco`) and revision 3 three more (`test_large_sequence_projects_fe4s4_after_fe2s2`, `test_bounded_projection_takes_every_measurement`, `test_qpe_summary_and_wire_boundaries`), so every later count is 7 above revision 1's; revision 4 changed the assertions of two tests and added none. Dry run with all three tasks: `1215 passed in 467.97s`, peak 2,232 MB, no warnings summary, 432 s inside the 15-minute budget (revision 3: 627.16 s with the reviewer's and an implementer's tests running in parallel; revision 2: 462.53 s; `4r-per-commit-tier.log`, with `TMPDIR` outside the workspace so that `tests/test_infrastructure.py` passes). Every per-commit test of the three tasks, 105, ran in one session with one worker under the watchdog with `--durations`: `105 passed, 2 deselected in 53.70s` at a session peak of 567 MB, which bounds every test's peak. The longest was `test_every_large_row_projects`, 3.6 s (the rows and gate graphs of six systems, 3.2 s of work in a fresh process: `probes/p4r/profile_projects.py`; `test_large_sequence_projects_fe4s4_after_fe2s2` reuses them within a worker); then `test_report_fails_on_mismatch` (3.1 s: two runs of `main`, one compile in a subprocess), the `C(ZeroReflection)` compiles (2.7 and 2.6 s), the walk, QPE and L4 compiles (1.5 to 2.0 s) and the FeMoco-54 `main --rows` that two tests share (1.4 s, once per worker); every other phase (setup, call or teardown) of every test, the 42 documentation tests among them, is below 1.0 s (`4r-per-test-evidence.log`; revision 3's session: 74.42 s, 608 MB). The nightly test: `test_report_mid` alone: `1 passed in 226.61s`, process tree 1,181 MB (`4r-report-mid.log`); the nightly tier grows by about 4 minutes against its 4-hour budget.

## Risks and open questions

1. **The large compiles are projected, not measured.** The projections rest on four small systems; the H2 points barely move memory, so the extrapolated slopes come from `naphthalene`, and a bounded row takes the largest time and peak over its family's measurements, which assumes that a program no larger than some measured member of its family compiles no heavier than the dearest member; gate count only chooses between that and extrapolation. If a `fe2s2` or `fe4s4` compile is killed at 7.5 GB, the run stops there: the remaining programs of that system are recorded as skipped, the orchestrator reports the kill and escalates, and the rows stay their closed forms, marked killed or skipped.
2. **Timings are machine-dependent.** The committed json holds the timings of the run that made it; reuse keeps them while the code hash matches. A projection that uses timings from a loaded machine is pessimistic, not optimistic.
3. **Any code edit of the report script invalidates the cache.** The hash covers `scripts/resource_report.py` (review finding I-2), so even a code change that cannot change a count, such as a log message, makes every cached entry history: the next run recompiles the small systems (13 to 21 minutes) and the large rows become pending until the orchestrator reruns them. Accepted: a cached count is current only for the code that built it; docstring and comment edits keep the hash.
4. **The base moves.** Task 3.5 is merged on `main` (`329ede8`; the replica is at `5fe4d65`); Tasks 4.1 and 4.2 are in this plan's replica from the QPE plan's revision 3. The orchestrator shifts the counts if other tasks merge first.
5. **dfthcbe's port is exact at the report's parameter sets only.** Its README formulas are piecewise (truncation, dead stops); the fixture pins all eight sets, and a new parameter set needs a fixture row before its dfthcbe values are trusted (`make_report_fixture.py` takes them from `default_params`).
6. **No open question blocks the plan.** The user's next decision is Q-1, after Task 4.4, with the report's qubit section and its deferred refinements.

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
| `src/dfthc/costs_reference.py` | 4.3 | `paper_table3`, `dfthcbe_blocks`, `delta_causes`, `iteration_counts`, `iteration_causes`, `default_params`, `popcount`, `aligned_blocks`, `DEFAULT_PRECISIONS`, `PAPER_QUBITS_FEMOCO` |
| `src/dfthc/testing/provenance.py` | 4.3 | `code_hash`, `hashed_files`, `environment`, `project_compile`, `REPORT_SCRIPT` |
| `scripts/make_report_fixture.py` | 4.3 | dfthcbe's call graph at the report's parameter sets (`.venv-ref` only) |
| `tests/fixtures/dfthcbe/report_toffolis.json` | 4.3 | its output, read by the port's test |
| `scripts/resource_report.py` | 4.3 | the report: rows, compiles, checks, cache, `--large`, `docs/resources.md` and `.json` |
| `tests/report/test_report.py` | 4.3 | the report's tests (63 per commit, 1 nightly) |
| `docs/resources.md`, `docs/resources.json` | 4.3, 4.5 | generated |
| `docs/architecture.md`, `docs/conventions.md`, `docs/verification.md` | 4.4 | new documents |
| `docs/limitations.md`, `docs/environment.md` | 4.4 | consolidated, refreshed |
| `tests/test_docstrings.py` | 4.4 | the docstring audit and the documentation tests (46 per commit) |
| `tests/report/test_femoco_cache.py` | 4.5 | the cached FeMoco-54 compile (1 per commit, 1 release) |

**Shell conventions.** Run commands from the worktree root of your lane. `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` locates the shared environment and `REF="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv-ref"` the reference environment, as in the earlier plans.

---

### Task 4.3: Resource report and `dfthc.costs_reference`

**Task id:** `4.3` · **Branch:** `phase4/4.3` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 4.1

**Files:**
- Create: `src/dfthc/costs_reference.py`, `src/dfthc/testing/provenance.py`, `scripts/make_report_fixture.py`, `tests/fixtures/dfthcbe/report_toffolis.json`, `scripts/resource_report.py`, `tests/report/test_report.py`, `docs/resources.md`, `docs/resources.json` (the last two generated)

**Interfaces:**
- Consumes: `dfthc.costs` (every spec 9.24 function, `cheapest_block_bits`, `truncated_unary`, `measured_and_uncomputes`, `toffoli_only`); `dfthc.compile` (`FIXED_DECOMPS`, `GATE_SET`, `compiled_resources`, `surviving_dfthc_operators`, `ANALYZE_NAMES`); `dfthc.data.factors` (`load_factors`, `random_factors`, `SYSTEMS`); `dfthc.data.tables.WalkTables`; `dfthc.ops.qroam` (`QROAMRead.visited_ranges`, `QROAMRead.loaded_ranges`, `QROAMErase.fixup_high_ranges`, `PhaseFixup.required_wires`, `PhaseFixup.address_order`); `dfthc.ops.reflections.ZeroReflection`; `dfthc.ops.walk` (`DFTHCWalk`, `ControlledDFTHCWalk`); `dfthc.params` (`FEMOCO54`, `H2_STO3G_TEST`, `WalkParams`, `lg`, `odd_part`); Task 4.1's `dfthc.qpe` (`ControlledWalkPower`, `qpe_program`, `spare_wires_needed`); `dfthc.registers.WalkLayout`; `dfthc.testing.watchdog` (`run_watched`, `MEMORY_CAP_MB`, `effective_limit_mb`, `within_effective_limit`); `dfthc.testing.fixtures.load_dfthcbe_fixture`; Task 1A.6's `tests/fixtures/dfthcbe/femoco_block_toffolis.json`.
- Produces: `dfthc.costs_reference.paper_table3(p) -> dict[str, int]`, `dfthcbe_blocks(p) -> dict[str, int]`, `delta_causes(p) -> dict[str, tuple[dict[str, int], dict[str, int]]]` (per row key, the contribution of each code against dfthcbe and against the paper), `iteration_counts(p) -> dict[str, tuple[int, int, int, int]]` (each unary iteration as ours, the intermediate, dfthcbe's and the paper's count), `iteration_causes(p) -> dict[str, tuple[dict[str, int], dict[str, int]]]` (each unary iteration's contributions by code against dfthcbe and against the paper, zeros included), `default_params(N, R, B, C, n_plus) -> WalkParams`, `popcount`, `aligned_blocks`, `DEFAULT_PRECISIONS`, `PAPER_QUBITS_FEMOCO`; `dfthc.testing.provenance.code_hash(root=REPO) -> str`, `hashed_files(root=REPO)`, `environment(root=REPO) -> dict`, `project_compile(points, gates, *, cap_mb=MEMORY_CAP_MB, time_limit_s=3600) -> dict` (keys `allowed`, `reason`, `method` (`bounded` or `k1`), `gates`, `seconds`, `peak_mb`, `limits`, `points`, and for `k1` also `time_exponent` and `mb_per_1000_gates`), `LOCK_FILE`, `REPORT_SCRIPT`; `scripts/resource_report.py` with `main(argv) -> int` and, for Task 4.5's test, `build_system(name) -> System`, `build_rows(s) -> list[Row]` (`Row.id`, `.calls`, `.controlled_calls`, `.partition`, `.closed` (`Cost`), `.uncomputes`, `.wires`), `current(cache, key, hash_) -> dict | None`, `compile_counts`, `measure`, `FEMOCO_PROGRAMS`; and `gates(rows) -> dict[str, int]`, `family(row) -> str`, `projection(cache, row, hash_, size) -> dict`, `run_system(...)`.

**Acceptance tests** (64 tests: 63 per commit, 1 `slow`):
- `test_report_h2_rows_and_columns` (per commit): the 48 rows at `H2_STO3G_TEST` in order, the partition and its calls, the walks' closed forms and qubits (63, 64; QPE t = 3: 67), levels, steps, Clifford kinds; `main --rows` (no compile) writes both files, every row `pending` with all 23 json keys, the walk's dfthcbe 167 and paper 205 with deltas −9 and −47 and their codes, every row's codes nonzero and summing to its deltas (`ControlledRefT2` against the paper: `ref-unif` −1, `ctrl-ref-t2` +1 for a delta of 0), and every row in the markdown table.
- `test_report_h2[row]` (48, per commit, `compiled`): each program compiles in this process to exactly its closed T, M and A on exactly its qubits.
- `test_report_mid` (`slow`, `compiled`, `subprocess`): `main --systems naphthalene`, every program in its own watched subprocess, exit 0, every row measured and equal to its closed form (walk 1,913 / 502 / 1,678, controlled 1,915 / 502 / 1,680), `MID_NAPHTHALENE`'s parameters.
- `test_report_fails_on_mismatch` (`compiled`, `subprocess`): a monkeypatched `costs.spin_swap` makes `main` exit 1 with the mismatch named; a removed reason contribution makes it exit 1 with the unexplained delta named.
- `test_large_compile_is_watched` (`subprocess`): `--large` on a stand-in that allocates 400 MB under a 200 MB cap is killed (`memory`, no survivors) and recorded as killed, not as a measurement, and the watched run gets `LARGE_TIMEOUT_S` unchanged; a projection of tens of GB is refused with its projection and no compile.
- `test_large_stops_after_a_refusal_at_femoco`, `test_large_stops_after_a_failed_or_killed_compile` (2: `killed`, `failed`; no compile; the projection and the watched run monkeypatched, on `H2_STO3G_TEST`'s rows under the names `femoco54` and `fe2s2`): a refused `DFTHCWalk` at FeMoco-54, or a killed or failed first row at a large system, stops the run; the next row is never measured, is recorded `skipped`, and the log names the escalation.
- `test_every_large_row_projects` (no compile): from a synthetic cache with the real rows and gates of the four small systems and made-up seconds and MB, all 46 rows at each of `fe2s2` and `fe4s4` have a size measure and an allowed projection, by both methods.
- `test_large_sequence_projects_fe4s4_after_fe2s2` (no compile): with synthetic `fe2s2` entries added, every `fe4s4` program projects from its family's entries including `fe2s2`'s (the extrapolated ones through the `fe2s2` point, the bounded ones as the family's maximum, which a made-up 7,000 MB `fe2s2` `SpinMux` entry sets for the `fe4s4` `SpinMux`).
- `test_bounded_projection_takes_every_measurement`: non-monotone points (100 gates, 3,500 s, 7,000 MB) and (200 gates, 10 s, 100 MB) project a 150-gate target to 3,500 s and 7,000 MB; refused under the 4 GB cap, and under 60 minutes when the dearer point took 3,700 s.
- `test_cache_needs_matching_code_hash`: a docstring, a comment and a `testing/` edit keep `code_hash`, and so do a docstring and a comment edit of the report script; a code edit of the package or of the script's program builders changes it; a stale entry stays as history and the row is `pending`; an entry of the current hash is reused without compiling.
- `test_reason_codes_at_femoco` (no compile): the exact codes and contributions of `TailAngleLoad`, `SFAngleLoad`, `AngleErase`, `RefT2`, `ControlledRefT2`, `RefAB`, `ControlledRefAB`, both block encodings, both walks and `ControlledWalkPower(p=1)` and `(p=2)` at FeMoco-54, against dfthcbe and the paper (section "Reason-code decomposition at FeMoco-54").
- `test_qpe_summary_and_wire_boundaries` (no compile): the FeMoco-54 QPE summary's T, M and A from the controlled walk (A = 8,677, 26,032, 60,743 at t = 1, 2, 3), A in the basis sentence and the table; the QPE wires `H2_STO3G_TEST` 67, 68, 70 at t = 3, 4, 5 and FeMoco-54 1,586, 1,587, 1,589 at t = 21, 22, 23.
- `test_dfthcbe_blocks_equal_the_call_graph` (the port against the fixture at the eight parameter sets and Task 1A.6's counts; the fixture's tail walk is N + s(RC) − 2, the split of `prefix-stops` and `iter-const`, and the intermediate is dfthcbe's walk on the outer read, the outer erase's fix-up and the SF fix-up), `test_paper_table3_reproduces_table_iii`, `test_project_compile_is_the_k1_method` (the K-1 extrapolation and the bounded case, the maximum over all points).

**Definition of done:** RED seen; `63 passed` per commit and `1 passed` nightly, targeted; the report of the H2 systems and `naphthalene` generated (exit 0); the per-commit tier passes; `ruff check .` clean; one commit; then the implementer stops for the orchestrator's `--large fe2s2` and `--large fe4s4` (Step 12), after which the regenerated report is committed on the branch.

- [ ] **Step 1: Write the failing test file**

Create `tests/report/test_report.py` (a new directory, no `__init__.py`):

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
    "Adjoint(GivensRotation)", "AngleErase", "InnerUnprepare", "QROAMErase@InnerUnprepare",
    "Adjoint(FlaggedUniformPrep)@InnerUnprepare", "RefT2", "ZeroReflection@RefT2",
    "InnerSelectDagger", "Adjoint(MajoranaGate)", "OuterUnprepare", "QROAMErase@OuterUnprepare",
    "Adjoint(FlaggedUniformPrep)@OuterUnprepare", "RefAB", "ZeroReflection@RefAB",
    "ControlledDFTHCWalk", "ControlledDFTHCBlockEncoding", "ControlledRefT2", "ControlledRefAB",
    "PhaseFixup@InnerUnprepare", "C(ZeroReflection)@RefT2", "PhaseFixup@OuterUnprepare",
    "C(ZeroReflection)@RefAB", "ControlledWalkPower(p=1)", "ControlledWalkPower(p=2)", "QPE(t=1)",
    "QPE(t=2)", "QPE(t=3)",
]
"""Every program of the report at H2_STO3G_TEST, in report order."""
PARTITION = {"OuterPrepare": (1, 1), "InnerPrepare": (2, 2), "InnerSigns": (2, 2),
             "TailAngleLoad": (2, 2), "SFAngleLoad": (2, 2), "SpinMux": (4, 4),
             "SpinSwap": (4, 4), "GivensLadder": (2, 2), "MajoranaGate": (1, 1),
             "GivensLadderDagger": (2, 2), "AngleErase": (2, 2), "InnerUnprepare": (2, 2),
             "RefT2": (1, 0), "Adjoint(MajoranaGate)": (1, 1), "OuterUnprepare": (1, 1),
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
    assert (walk.wires, cwalk.wires, rows["QPE(t=3)"].wires) == (63, 64, 67)
    assert rows["QROAMRead@InnerPrepare"].level == "L1" and rows["SpinSwap"].level == "L2"
    assert rows["RefAB"].level == "L3" and rows["QPE(t=1)"].level == "L4"
    assert rows["AngleErase"].steps == "18" and rows["QROAMErase@InnerUnprepare"].steps == "21"
    assert report.clifford_kind(rows["SpinSwap"]) == "exact"
    assert report.clifford_kind(rows["QROAMErase@InnerUnprepare"]) == (
        "upper bound (CNOT, PauliX); expected, hinted (CZ, PauliZ)")
    assert report.clifford_kind(rows["PhaseFixup@OuterUnprepare"]) == (
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
        assert len(rows) == len(sizes) == 46, name
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
    `H2_STO3G_TEST` 67, 68, 70 at t = 3, 4, 5 and FeMoco-54 1,586, 1,587, 1,589 at t = 21, 22,
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
    assert [record["qpe"][str(bits)]["wires"] for bits in (20, 23)] == [1585, 1589]
    assert "A = (2^t - 1) A_C(W) + (t^3 - t)/6" in record["qpe_basis"]
    assert "| t | T | M | A | Qubits |" in markdown
    h2, femoco = report.build_system("H2_STO3G_TEST"), report.build_system("femoco54")
    assert [report.qpe_wires(h2, bits) for bits in (3, 4, 5)] == [67, 68, 70]
    assert [report.qpe_wires(femoco, bits) for bits in (21, 22, 23)] == [1586, 1587, 1589]


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

- [ ] **Step 2: Run it to confirm it fails**

Run: `scripts/test.sh tests/report/test_report.py -q -m "not slow and not release"`
Expected: FAIL at collection, `1 error`: `ImportError: cannot import name 'costs_reference' from 'dfthc'`.

- [ ] **Step 3: Write `dfthc.costs_reference`**

Create `src/dfthc/costs_reference.py`:

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

- [ ] **Step 4: Write `dfthc.testing.provenance`**

Create `src/dfthc/testing/provenance.py`:

```python
"""Code hash, environment record and compile projection for cached compiles (spec 13.3; master
plan Task 4.3 and section 12.9).

:func:`code_hash` decides whether a compile cached in ``docs/resources.json`` is a current
measurement: it hashes the syntax trees, without docstrings (comments are not in a syntax tree),
of the package's modules and of ``scripts/resource_report.py``, which builds the cached programs
and selects the cache's entries, and the lock file. So a documentation edit keeps the hash and any
code or toolchain change gives a new one. ``dfthc/testing/`` is excluded: no compiled program
depends on it. :func:`project_compile` is Task 0.9's K-1 projection method, replaced by the
largest of the measurements where the target is no larger than a measured point.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import math
import platform
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dfthc.testing.watchdog import MEMORY_CAP_MB, effective_limit_mb, within_effective_limit

REPO = Path(__file__).resolve().parents[3]
LOCK_FILE = "requirements-nightly.lock"
REPORT_SCRIPT = "scripts/resource_report.py"
DISTRIBUTIONS = ("pennylane", "pennylane_catalyst", "pennylane_lightning", "jax", "jaxlib", "numpy")
TIME_LIMIT_S = 3600.0
"""The compile time a large program may be projected to (60 min, master plan 12.9)."""


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove the docstring of the module and of every class and function."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def hashed_files(root: Path = REPO) -> list[Path]:
    """The modules :func:`code_hash` covers: ``src/dfthc/**/*.py`` without ``dfthc/testing/``,
    then the report script."""
    package = root / "src" / "dfthc"
    return sorted(path for path in package.rglob("*.py")
                  if "testing" not in path.relative_to(package).parts) + [root / REPORT_SCRIPT]


def code_hash(root: Path = REPO) -> str:
    """SHA-256 over the docstring-free syntax trees of :func:`hashed_files` (each preceded by its
    path relative to ``root``) and the bytes of the lock file."""
    digest = hashlib.sha256()
    for path in hashed_files(root):
        tree = _strip_docstrings(ast.parse(path.read_text(), filename=str(path)))
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(ast.dump(tree, annotate_fields=True, include_attributes=False).encode())
    digest.update((root / LOCK_FILE).read_bytes())
    return digest.hexdigest()


def environment(root: Path = REPO) -> dict[str, str]:
    """Python, platform, the toolchain's package versions and the lock file's SHA-256."""
    record = {"python": platform.python_version(), "platform": platform.platform()}
    for name in DISTRIBUTIONS:
        try:
            record[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record[name] = "not installed"
    lock = root / LOCK_FILE
    record["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() \
        else "missing"
    return record


def _over(seconds: float, peak_mb: float, cap_mb: float, time_limit_s: float) -> list[str]:
    """Why a projection exceeds the limits, if it does."""
    reasons = []
    if seconds > time_limit_s:
        reasons.append(f"{seconds / 60:.0f} min > {time_limit_s / 60:.0f} min")
    if not within_effective_limit(peak_mb, cap_mb):
        reasons.append(f"{peak_mb / 1024:.1f} GB > {effective_limit_mb(cap_mb) / 1024:.1f} GB")
    return reasons


def project_compile(points: Sequence[tuple[int, float, float]], gates: int, *,
                    cap_mb: float = MEMORY_CAP_MB, time_limit_s: float = TIME_LIMIT_S) -> dict:
    """Project one program to ``gates`` gates from measured ``(gates, seconds, peak_mb)`` points
    of the same program family at other sizes.

    If some point has at least ``gates`` gates (a block whose size does not grow, such as the spin
    multiplexer), the projection is the largest time and the largest peak over all points,
    whatever their gate counts (``method: "bounded"``): gate count is not a monotone measure of
    compile cost, and it only decides between this and extrapolation. Otherwise it extrapolates
    by Task 0.9's K-1 method (``method: "k1"``): time as a power law through the two points with
    the most gates, exponent at least 1; memory as a least-squares line through all points, slope
    at least 0; fewer than two points with distinct gate counts give ``allowed: False`` and no
    projection. ``allowed`` is true only if time and memory stay within ``time_limit_s`` and
    :func:`~dfthc.testing.watchdog.effective_limit_mb` of ``cap_mb`` (7,680 MB under the 8 GB
    cap).
    """
    usable = sorted({(int(g), float(s), float(m)) for g, s, m in points})
    limits = {"seconds": time_limit_s, "mb": effective_limit_mb(cap_mb), "cap_mb": cap_mb}
    if any(g >= gates for g, _, _ in usable):
        seconds, peak_mb = max(s for _, s, _ in usable), max(m for _, _, m in usable)
        reasons = _over(seconds, peak_mb, cap_mb, time_limit_s)
        return {"allowed": not reasons,
                "reason": "; ".join(["the family's largest measurement", *reasons]),
                "method": "bounded",
                "gates": gates, "seconds": seconds, "peak_mb": peak_mb, "limits": limits,
                "points": [list(point) for point in usable]}
    if len({g for g, _, _ in usable}) < 2:
        return {"allowed": False, "reason": "fewer than two calibration points", "gates": gates,
                "points": [list(point) for point in usable]}
    (g1, s1, _), (g2, s2, _) = sorted({g: (g, s, m) for g, s, m in usable}.values())[-2:]
    exponent = max(1.0, math.log(s2 / s1) / math.log(g2 / g1)) if s1 > 0 and s2 > 0 else 1.0
    seconds = s2 * (gates / g2) ** exponent
    xs = np.array([g for g, _, _ in usable], float)
    ys = np.array([m for _, _, m in usable], float)
    slope = max(0.0, float(np.polyfit(xs, ys, 1)[0]))
    peak_mb = float(ys.mean() - slope * xs.mean()) + slope * gates
    reasons = _over(seconds, peak_mb, cap_mb, time_limit_s)
    return {"allowed": not reasons, "reason": "; ".join(reasons) or "within the limits",
            "method": "k1", "gates": gates, "seconds": round(seconds, 1),
            "time_exponent": round(exponent, 2), "peak_mb": round(peak_mb, 1),
            "mb_per_1000_gates": round(1000 * slope, 3), "limits": limits,
            "points": [list(point) for point in usable]}
```

- [ ] **Step 5: Write the fixture script**

Create `scripts/make_report_fixture.py`:

```python
"""Generate ``tests/fixtures/dfthcbe/report_toffolis.json``: dfthcbe's Toffolis per block at every
parameter set of the resource report, from its own call graph. Runs only in ``.venv-ref``.

Usage, from the repository root (environment: docs/agents/HANDOFF.md, section 3.4)::

    .venv-ref/bin/python scripts/make_report_fixture.py [--out tests/fixtures/dfthcbe]

For each report system (``dfthc.costs_reference.default_params`` of a vendored system's shape),
``H2_STO3G_TEST`` and ``FEMOCO54``, it builds ``DFTHCWalk.symbolic`` (no table) and evaluates the
Toffolis (``Toffoli + CSwap + And + T/4`` of Qualtran's ``QECGatesCost``, as dfthcbe's tests and
``scripts/make_dfthcbe_fixtures.py`` count them) of every bloq that has a row in the report, keyed
as :func:`dfthc.costs_reference.dfthcbe_blocks` keys them. The test
``tests/report/test_report.py::test_dfthcbe_blocks_equal_the_call_graph`` compares that port with
this file. Qualtran and dfthcbe are imported inside the functions, so the main environment can
import this module's constants.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dfthc.costs_reference import default_params  # noqa: E402
from dfthc.data.factors import SYSTEMS, load_factors  # noqa: E402
from dfthc.params import FEMOCO54, H2_STO3G_TEST  # noqa: E402

FIXTURE = "report_toffolis.json"
DFTHCBE_COMMIT = "930255020cec30172e0a7d17c0071d9476a687d1"


def report_parameters() -> dict:
    """The report's parameter sets by system name (spec 13.3 step 1), and the test preset."""
    sets = {}
    for system in SYSTEMS:
        f = load_factors(system)
        sets[system] = default_params(f.N, f.R, f.B, f.C, f.n_plus)
    sets["femoco54"] = FEMOCO54
    sets["H2_STO3G_TEST"] = H2_STO3G_TEST
    return sets


def call_graph_toffolis(p) -> dict[str, int]:
    """dfthcbe's Toffolis per bloq of the symbolic walk at ``p``."""
    import sympy
    from dfthcbe import DFTHCWalk
    from qualtran.bloqs.mcmt import MultiControlX
    from qualtran.bloqs.rotations.zpow_via_phase_gradient import ZPowConstViaPhaseGradient
    from qualtran.resource_counting import QECGatesCost, get_cost_value

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
    outer, inner, select = be.outer, be.inner, be.select
    bloqs = {
        "rotation": ZPowConstViaPhaseGradient(exponent=sympy.Rational(1, 3),
                                              phase_grad_bitsize=p.s_u),
        "uniform_x_o": outer.uniform, "uniform_x_o_adjoint": outer.uniform.adjoint(),
        "uniform_b": inner.uniform, "uniform_b_adjoint": inner.uniform.adjoint(),
        "outer_read": outer.qroam, "outer_erase": outer.qroam.adjoint(),
        "inner_read": inner.qroam, "inner_erase": inner.qroam.adjoint(),
        "tail_angle_load": select.tail_qrom, "sf_angle_load": select.sf_qrom,
        "spin_swap": select.uswap,
        "givens_ladder": select.ladder, "givens_ladder_dagger": select.ladder.adjoint(),
        "majorana_gate": select.maj, "majorana_gate_adjoint": select.maj.adjoint(),
        "angle_erase": select.eraser,
        "outer_prepare": outer, "outer_unprepare": outer.adjoint(),
        "inner_prepare": inner, "inner_unprepare": inner.adjoint(),
        "inner_select": select, "inner_select_dagger": select.adjoint(),
        "ref_t2": be.ref_t2, "ref_ab": walk.ref_ab,
        "block_encoding": be, "controlled_block_encoding": be.controlled(),
        "walk": walk, "controlled_walk": walk.controlled(),
    }
    counts = {name: toffoli(bloq) for name, bloq in bloqs.items()}
    counts["spin_mux"] = (toffoli(MultiControlX(cvs=(0, 1)))
                          + toffoli(MultiControlX(cvs=(1, 1))))
    counts["controlled_ref_t2"] = (counts["controlled_block_encoding"]
                                   - counts["block_encoding"] + counts["ref_t2"])
    counts["controlled_ref_ab"] = counts["controlled_walk"] - counts["controlled_block_encoding"]
    return dict(sorted(counts.items()))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "tests" / "fixtures" / "dfthcbe")
    out = parser.parse_args(argv).out
    out.mkdir(parents=True, exist_ok=True)
    sets = report_parameters()
    fixture = {
        "dfthcbe_commit": DFTHCBE_COMMIT,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("dfthcbe", "qualtran", "sympy")},
        "method": "DFTHCWalk.symbolic call graph evaluated at each parameter set; Toffolis = "
                  "Toffoli + CSwap + And + T/4 of QECGatesCost; spin_mux = one MultiControlX "
                  "(0, 1) plus one (1, 1); controlled_ref_t2 and controlled_ref_ab are the "
                  "controlled block encoding's and controlled walk's increments",
        "parameters": {name: dataclasses.asdict(p) for name, p in sets.items()},
        "toffolis": {name: call_graph_toffolis(p) for name, p in sets.items()},
    }
    (out / FIXTURE).write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out / FIXTURE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write the fixture and check it against dfthcbe**

Create `tests/fixtures/dfthcbe/report_toffolis.json` with exactly this content (the dry run's output of Step 5's script; 405 lines):

```json
{
  "dfthcbe_commit": "930255020cec30172e0a7d17c0071d9476a687d1",
  "versions": {
    "dfthcbe": "1.0",
    "qualtran": "0.7.0",
    "sympy": "1.14.0"
  },
  "method": "DFTHCWalk.symbolic call graph evaluated at each parameter set; Toffolis = Toffoli + CSwap + And + T/4 of QECGatesCost; spin_mux = one MultiControlX (0, 1) plus one (1, 1); controlled_ref_t2 and controlled_ref_ab are the controlled block encoding's and controlled walk's increments",
  "parameters": {
    "h2_sto3g_R2_B2_C1": {
      "N": 2,
      "R": 2,
      "B": 2,
      "C": 1,
      "n_plus": 1,
      "b_rot": 15,
      "b_k1": 15,
      "b_k2": 15,
      "s_u": 7,
      "k1": 0,
      "k2": 0,
      "ke1": 1,
      "ke2": 0
    },
    "h2_631g_R2_B2_C2": {
      "N": 4,
      "R": 2,
      "B": 2,
      "C": 2,
      "n_plus": 3,
      "b_rot": 15,
      "b_k1": 15,
      "b_k2": 15,
      "s_u": 7,
      "k1": 0,
      "k2": 0,
      "ke1": 1,
      "ke2": 0
    },
    "h2_631g_R2_B3_C3": {
      "N": 4,
      "R": 2,
      "B": 3,
      "C": 3,
      "n_plus": 3,
      "b_rot": 15,
      "b_k1": 15,
      "b_k2": 15,
      "s_u": 7,
      "k1": 0,
      "k2": 0,
      "ke1": 2,
      "ke2": 0
    },
    "naphthalene": {
      "N": 10,
      "R": 8,
      "B": 6,
      "C": 5,
      "n_plus": 5,
      "b_rot": 15,
      "b_k1": 15,
      "b_k2": 15,
      "s_u": 7,
      "k1": 1,
      "k2": 2,
      "ke1": 3,
      "ke2": 1
    },
    "fe2s2": {
      "N": 20,
      "R": 6,
      "B": 29,
      "C": 15,
      "n_plus": 5,
      "b_rot": 15,
      "b_k1": 15,
      "b_k2": 15,
      "s_u": 7,
      "k1": 1,
      "k2": 3,
      "ke1": 3,
      "ke2": 1
    },
    "fe4s4": {
      "N": 36,
      "R": 33,
      "B": 11,
      "C": 3,
      "n_plus": 10,
      "b_rot": 15,
      "b_k1": 15,
      "b_k2": 15,
      "s_u": 7,
      "k1": 1,
      "k2": 3,
      "ke1": 4,
      "ke2": 1
    },
    "femoco54": {
      "N": 54,
      "R": 10,
      "B": 27,
      "C": 27,
      "n_plus": 27,
      "b_rot": 15,
      "b_k1": 15,
      "b_k2": 15,
      "s_u": 7,
      "k1": 2,
      "k2": 4,
      "ke1": 4,
      "ke2": 2
    },
    "H2_STO3G_TEST": {
      "N": 2,
      "R": 2,
      "B": 2,
      "C": 1,
      "n_plus": 1,
      "b_rot": 4,
      "b_k1": 2,
      "b_k2": 2,
      "s_u": 3,
      "k1": 1,
      "k2": 1,
      "ke1": 1,
      "ke2": 1
    }
  },
  "toffolis": {
    "h2_sto3g_R2_B2_C1": {
      "angle_erase": 2,
      "block_encoding": 357,
      "controlled_block_encoding": 358,
      "controlled_ref_ab": 35,
      "controlled_ref_t2": 18,
      "controlled_walk": 393,
      "givens_ladder": 28,
      "givens_ladder_dagger": 28,
      "inner_erase": 2,
      "inner_prepare": 40,
      "inner_read": 6,
      "inner_select": 76,
      "inner_select_dagger": 76,
      "inner_unprepare": 36,
      "majorana_gate": 5,
      "majorana_gate_adjoint": 5,
      "outer_erase": 0,
      "outer_prepare": 19,
      "outer_read": 2,
      "outer_unprepare": 17,
      "ref_ab": 35,
      "ref_t2": 17,
      "rotation": 5,
      "sf_angle_load": 4,
      "spin_mux": 2,
      "spin_swap": 2,
      "tail_angle_load": 1,
      "uniform_b": 17,
      "uniform_b_adjoint": 17,
      "uniform_x_o": 0,
      "uniform_x_o_adjoint": 0,
      "walk": 392
    },
    "h2_631g_R2_B2_C2": {
      "angle_erase": 4,
      "block_encoding": 621,
      "controlled_block_encoding": 622,
      "controlled_ref_ab": 36,
      "controlled_ref_t2": 18,
      "controlled_walk": 658,
      "givens_ladder": 84,
      "givens_ladder_dagger": 84,
      "inner_erase": 4,
      "inner_prepare": 46,
      "inner_read": 12,
      "inner_select": 196,
      "inner_select_dagger": 196,
      "inner_unprepare": 38,
      "majorana_gate": 5,
      "majorana_gate_adjoint": 5,
      "outer_erase": 2,
      "outer_prepare": 24,
      "outer_read": 6,
      "outer_unprepare": 20,
      "ref_ab": 36,
      "ref_t2": 17,
      "rotation": 5,
      "sf_angle_load": 4,
      "spin_mux": 2,
      "spin_swap": 4,
      "tail_angle_load": 3,
      "uniform_b": 17,
      "uniform_b_adjoint": 17,
      "uniform_x_o": 0,
      "uniform_x_o_adjoint": 0,
      "walk": 657
    },
    "h2_631g_R2_B3_C3": {
      "angle_erase": 6,
      "block_encoding": 642,
      "controlled_block_encoding": 643,
      "controlled_ref_ab": 37,
      "controlled_ref_t2": 18,
      "controlled_walk": 680,
      "givens_ladder": 84,
      "givens_ladder_dagger": 84,
      "inner_erase": 8,
      "inner_prepare": 41,
      "inner_read": 24,
      "inner_select": 201,
      "inner_select_dagger": 201,
      "inner_unprepare": 25,
      "majorana_gate": 5,
      "majorana_gate_adjoint": 5,
      "outer_erase": 3,
      "outer_prepare": 48,
      "outer_read": 8,
      "outer_unprepare": 43,
      "ref_ab": 37,
      "ref_t2": 17,
      "rotation": 5,
      "sf_angle_load": 6,
      "spin_mux": 2,
      "spin_swap": 4,
      "tail_angle_load": 4,
      "uniform_b": 0,
      "uniform_b_adjoint": 0,
      "uniform_x_o": 21,
      "uniform_x_o_adjoint": 21,
      "walk": 679
    },
    "naphthalene": {
      "angle_erase": 21,
      "block_encoding": 1897,
      "controlled_block_encoding": 1898,
      "controlled_ref_ab": 40,
      "controlled_ref_t2": 19,
      "controlled_walk": 1938,
      "givens_ladder": 252,
      "givens_ladder_dagger": 252,
      "inner_erase": 32,
      "inner_prepare": 179,
      "inner_read": 140,
      "inner_select": 612,
      "inner_select_dagger": 612,
      "inner_unprepare": 71,
      "majorana_gate": 5,
      "majorana_gate_adjoint": 5,
      "outer_erase": 11,
      "outer_prepare": 94,
      "outer_read": 44,
      "outer_unprepare": 61,
      "ref_ab": 40,
      "ref_t2": 18,
      "rotation": 5,
      "sf_angle_load": 48,
      "spin_mux": 2,
      "spin_swap": 10,
      "tail_angle_load": 10,
      "uniform_b": 21,
      "uniform_b_adjoint": 21,
      "uniform_x_o": 29,
      "uniform_x_o_adjoint": 29,
      "walk": 1937
    },
    "fe2s2": {
      "angle_erase": 54,
      "block_encoding": 4377,
      "controlled_block_encoding": 4378,
      "controlled_ref_ab": 43,
      "controlled_ref_t2": 21,
      "controlled_walk": 4421,
      "givens_ladder": 532,
      "givens_ladder_dagger": 532,
      "inner_erase": 106,
      "inner_prepare": 562,
      "inner_read": 517,
      "inner_select": 1364,
      "inner_select_dagger": 1364,
      "inner_unprepare": 151,
      "majorana_gate": 5,
      "majorana_gate_adjoint": 5,
      "outer_erase": 18,
      "outer_prepare": 130,
      "outer_read": 75,
      "outer_unprepare": 73,
      "ref_ab": 43,
      "ref_t2": 20,
      "rotation": 5,
      "sf_angle_load": 175,
      "spin_mux": 2,
      "spin_swap": 20,
      "tail_angle_load": 22,
      "uniform_b": 25,
      "uniform_b_adjoint": 25,
      "uniform_x_o": 33,
      "uniform_x_o_adjoint": 33,
      "walk": 4420
    },
    "fe4s4": {
      "angle_erase": 79,
      "block_encoding": 6293,
      "controlled_block_encoding": 6294,
      "controlled_ref_ab": 43,
      "controlled_ref_t2": 20,
      "controlled_walk": 6337,
      "givens_ladder": 980,
      "givens_ladder_dagger": 980,
      "inner_erase": 74,
      "inner_prepare": 386,
      "inner_read": 350,
      "inner_select": 2522,
      "inner_select_dagger": 2522,
      "inner_unprepare": 110,
      "majorana_gate": 5,
      "majorana_gate_adjoint": 5,
      "outer_erase": 21,
      "outer_prepare": 153,
      "outer_read": 89,
      "outer_unprepare": 85,
      "ref_ab": 43,
      "ref_t2": 19,
      "rotation": 5,
      "sf_angle_load": 364,
      "spin_mux": 2,
      "spin_swap": 36,
      "tail_angle_load": 38,
      "uniform_b": 17,
      "uniform_b_adjoint": 17,
      "uniform_x_o": 41,
      "uniform_x_o_adjoint": 41,
      "walk": 6336
    },
    "femoco54": {
      "angle_erase": 90,
      "block_encoding": 9596,
      "controlled_block_encoding": 9597,
      "controlled_ref_ab": 45,
      "controlled_ref_t2": 21,
      "controlled_walk": 9642,
      "givens_ladder": 1484,
      "givens_ladder_dagger": 1484,
      "inner_erase": 179,
      "inner_prepare": 913,
      "inner_read": 872,
      "inner_select": 3502,
      "inner_select_dagger": 3502,
      "inner_unprepare": 220,
      "majorana_gate": 5,
      "majorana_gate_adjoint": 5,
      "outer_erase": 33,
      "outer_prepare": 212,
      "outer_read": 151,
      "outer_unprepare": 94,
      "ref_ab": 45,
      "ref_t2": 20,
      "rotation": 5,
      "sf_angle_load": 271,
      "spin_mux": 2,
      "spin_swap": 54,
      "tail_angle_load": 56,
      "uniform_b": 21,
      "uniform_b_adjoint": 21,
      "uniform_x_o": 37,
      "uniform_x_o_adjoint": 37,
      "walk": 9641
    },
    "H2_STO3G_TEST": {
      "angle_erase": 2,
      "block_encoding": 158,
      "controlled_block_encoding": 159,
      "controlled_ref_ab": 9,
      "controlled_ref_t2": 5,
      "controlled_walk": 168,
      "givens_ladder": 6,
      "givens_ladder_dagger": 6,
      "inner_erase": 4,
      "inner_prepare": 22,
      "inner_read": 9,
      "inner_select": 32,
      "inner_select_dagger": 32,
      "inner_unprepare": 17,
      "majorana_gate": 5,
      "majorana_gate_adjoint": 5,
      "outer_erase": 0,
      "outer_prepare": 8,
      "outer_read": 4,
      "outer_unprepare": 4,
      "ref_ab": 9,
      "ref_t2": 4,
      "rotation": 1,
      "sf_angle_load": 4,
      "spin_mux": 2,
      "spin_swap": 2,
      "tail_angle_load": 1,
      "uniform_b": 9,
      "uniform_b_adjoint": 9,
      "uniform_x_o": 0,
      "uniform_x_o_adjoint": 0,
      "walk": 167
    }
  }
}
```

Then regenerate it from dfthcbe's call graph in `.venv-ref` into a scratch directory (about 5 s, below 200 MB; `MPLCONFIGDIR` keeps Matplotlib's cache out of your home directory) and compare:

```bash
mkdir -p .tmp-fixture
MPLCONFIGDIR="$PWD/.tmp-fixture" PYTHONPATH= "$REF/bin/python" scripts/make_report_fixture.py --out .tmp-fixture
cmp .tmp-fixture/report_toffolis.json tests/fixtures/dfthcbe/report_toffolis.json && echo identical
rm -rf .tmp-fixture
```

Expected: `wrote .tmp-fixture/report_toffolis.json`, then `identical`. If `.venv-ref` is missing, keep the plan's content and say so in the report; if the files differ, stop and report the difference (BLOCKED).

- [ ] **Step 7: Write the report script**

Create `scripts/resource_report.py`:

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
per walk times the closed form against the measured walk and controlled walk).
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
    "junk-qubits": "junk persists until the erase",
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
NAME_KEYS = {
    "OuterPrepare": "outer_prepare", "OuterUnprepare": "outer_unprepare",
    "InnerPrepare": "inner_prepare", "InnerUnprepare": "inner_unprepare",
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
    ("Adjoint(FlaggedUniformPrep)", "OuterUnprepare"): "uniform_x_o_adjoint",
    ("Adjoint(FlaggedUniformPrep)", "InnerUnprepare"): "uniform_b_adjoint",
    ("PhaseGradientRotation", "FlaggedUniformPrep"): "rotation",
    ("QROAMRead", "OuterPrepare"): "outer_read", ("QROAMRead", "InnerPrepare"): "inner_read",
    ("QROAMRead", "TailAngleLoad"): "tail_angle_load",
    ("QROAMRead", "SFAngleLoad"): "sf_angle_load",
    ("QROAMErase", "OuterUnprepare"): "outer_erase",
    ("QROAMErase", "InnerUnprepare"): "inner_erase",
    ("PhaseFixup", "OuterUnprepare"): "outer_erase",
    ("PhaseFixup", "InnerUnprepare"): "inner_erase",
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
         "InnerUnprepare": "19-22", "RefT2": "23", "ControlledRefT2": "23",
         "OuterUnprepare": "24-27", "RefAB": "28", "ControlledRefAB": "28",
         "DFTHCBlockEncoding": "1-27", "ControlledDFTHCBlockEncoding": "1-27",
         "DFTHCWalk": "1-28", "ControlledDFTHCWalk": "1-28", "ControlledWalkPower": "QPE",
         "QPE": "QPE"}
PARENT_STEPS = {("FlaggedUniformPrep", "OuterPrepare"): "1",
                ("FlaggedUniformPrep", "InnerPrepare"): "5",
                ("Adjoint(FlaggedUniformPrep)", "OuterUnprepare"): "27",
                ("Adjoint(FlaggedUniformPrep)", "InnerUnprepare"): "22",
                ("PhaseGradientRotation", "FlaggedUniformPrep"): "1, 5",
                ("QROAMRead", "OuterPrepare"): "2", ("QROAMRead", "InnerPrepare"): "6",
                ("QROAMRead", "TailAngleLoad"): "10", ("QROAMRead", "SFAngleLoad"): "11",
                ("QROAMErase", "OuterUnprepare"): "26", ("QROAMErase", "InnerUnprepare"): "21",
                ("PhaseFixup", "OuterUnprepare"): "26", ("PhaseFixup", "InnerUnprepare"): "21",
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
    queues it)."""
    if op.name.startswith("Adjoint("):
        return qp.adjoint(_rebuilt(op.base, index))
    arguments = dict(op.arguments)
    for name in op.wire_argnames:
        arguments[name] = [index[wire] for wire in getattr(op, name)]
    return type(op)(**arguments)


def _call_program(op):
    def program():
        index = {wire: i for i, wire in enumerate(op.wires)}
        return (lambda: _rebuilt(op, index)), len(op.wires)

    return program


def _fixup_program(erase):
    """``PhaseFixup`` with the erase's fix-up arguments: its covered set, one-hot bits, enable
    and work, on a one-hot of ``2**onehot_bits`` wires; the program's parities are all ``False``,
    and the size operator (tape mode) takes an abstract parity array. Returns ``(program,
    size_op, wires)``."""
    a = erase.arguments
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
    """The closed form (``dfthc.costs``) of one call: ``(Cost(T, M), A)``. A counts the measured
    AND uncomputes inside the call: a prepare leaves its comparators' ANDs (``b_k`` and the
    uniform preparation's :func:`_dirty`) for its unprepare to measure (spec 9.24, 10.2), so per
    call A is not ``T`` minus the plain ``Toffoli`` gates; over the walk it is
    (``measured_and_uncomputes``)."""
    a = _target(op).arguments if op is not None else {}
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
    if name in ("QROAMErase", "PhaseFixup"):
        ranges = QROAMErase.fixup_high_ranges(a["address_widths"], a["live_ranges"],
                                              a["block_bits"], a["onehot_bits"])
        cost = costs.qroam_erase(ranges, a["block_bits"], a["width"], a["onehot_bits"],
                                 a["control_value"] is not None)
        cost = cost if name == "QROAMErase" else costs.Cost(cost.toffoli)
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
        "OuterUnprepare": (costs.outer_unprepare(p), -p.w_x + p.b_k1 + _dirty(p.X_o)),
        "InnerPrepare": (costs.inner_prepare(p),
                         -_dirty(p.B + 1) - (p.K2 - 1) * p.d2 - p.b_k2 - p.w_b),
        "InnerUnprepare": (costs.inner_unprepare(p), -p.w_b + p.b_k2 + _dirty(p.B + 1)),
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
        if row.name == "QROAMErase":
            program, size_op, wires = _fixup_program(row.op)
            closed, uncomputes = closed_form("PhaseFixup", row.op, p)
            extra.append(dataclasses.replace(
                row, id=f"PhaseFixup@{row.parent}", name="PhaseFixup", level="L1",
                steps=PARENT_STEPS[("PhaseFixup", row.parent)], calls=0, controlled_calls=0,
                partition=False, closed=closed, uncomputes=uncomputes, wires=wires, program=program,
                key=PARENT_KEYS[("PhaseFixup", row.parent)], descendants=frozenset({
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
    unhinted = names & {"QROAMRead", "QROAMErase", "AngleErase", "QPERound"}
    hinted = names & {"QROAMErase", "AngleErase", "PhaseFixup"}
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
              "junk_wires": len(s.layout.junk_x_o) + len(s.layout.junk_b), "rows": records}
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
    """Spec 13.3 step 4: the partition rows times their calls give the walk and the controlled
    walk, in closed form always and against the measured walks where they are measured; and the
    partition sums of dfthcbe's and the paper's values give their walks."""
    errors, by_id = [], {r["id"]: r for r in records}
    for walk, calls in (("DFTHCWalk", "calls"), ("ControlledDFTHCWalk", "controlled_calls")):
        parts = [r for r in records if r["partition"] and r[calls]]
        total = {name: sum(r[calls] * r["closed"][name] for r in parts)
                 for name in ("toffoli", "measurements", "measured_and_uncomputes")}
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
    junk = system["junk_wires"]
    lines += ["", f"- Ours minus dfthcbe: walk {walk['delta_dfthcbe']:+,}, controlled walk "
              f"{cwalk['delta_dfthcbe']:+,}. Ours minus the paper: walk {walk['delta_paper']:+,}. "
              f"The paper's {walk['paper']:,} is its QPE step, with the control on "
              f"REF_{{A u B}} only (App. B.7.2, Eq. (128)); against our controlled walk, "
              f"{cwalk['closed']['toffoli']:,}, the difference is {cwalk['delta_paper']:+,}: the "
              "block encoding's, with `ctrl-ref-t2` +1 (our controlled walk also controls "
              "REF_T2) and `ref-target` -1 (the controlled REF_{A u B} against B55), and no "
              "`qpe-ctrl`, since both carry the QPE control.",
              f"- Measurements per walk: M = {m:,}, A = {a:,} (controlled {ca:,}), so "
              f"M + A = {m + a:,} (controlled {m + ca:,}) physical measurements.",
              "- The paper's Table III charges the tail-angle fix-up (N) to the inner erase; "
              "the rows show its value and, in parentheses, the value with the fix-up at step 18, "
              "where dfthcbe and we apply it. The spin multiplexer is 0 in Table III and 1 per "
              "application in App. B.3 and B.7.1 (4(N + 1) for the four swaps).",
              "", "### Qubits (spec 10.3, 13.2)", "",
              f"Ours: {system['wires']:,} wires from `WalkLayout` ("
              + ", ".join(f"{g} {n}" for g, n in system["qubit_groups"].items())
              + f"); the paper: {costs_reference.PAPER_QUBITS_FEMOCO:,}. The junk registers hold "
              f"{junk:,} wires (`junk-qubits`): the paper measures the junk right after each "
              "read; early junk measurement (Q-1, open) would let the angle registers reuse them. "
              "The dirty comparator work registers (`cmp-free`) and r in the inner junk "
              "(`flat-r`) account for the rest of the outer and inner groups' excess.", "",
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
              "(ledger row 2E.2).",
              "- Early junk measurement (Q-1): about 460 qubits (the junk wires above), no Toffoli "
              "penalty.", ""]
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

- [ ] **Step 8: Run the per-commit tests**

Run: `scripts/test.sh tests/report/test_report.py -q -m "not slow and not release"`
Expected: `63 passed` (dry run 79 s with 2 workers on a busy machine, 1,145 MB for the session, before the review's test).

- [ ] **Step 9: Run the nightly test**

It compiles the 46 programs of `naphthalene`, each in its own watched subprocess (not heavy: one core, each below 700 MB).

Run: `scripts/test.sh tests/report/test_report.py -q -m "slow and not release"`
Expected: `1 passed` (dry run: 227 s, 1,181 MB for the process tree).

- [ ] **Step 10: Generate the report of the H2 systems and `naphthalene`**

Run: `PYTHONPATH=src "$VENV/bin/python" scripts/resource_report.py`
Expected: 190 lines `<system>/<row>: compiled (<s> s, <MB> MB)` (48 for each H2 system, 46 for `naphthalene`), no `CHECK FAILED` line, exit status 0 (`echo $?` prints `0`); 13 to 21 minutes. `fe2s2`, `fe4s4` and `femoco54` are built in tape mode, not compiled: their rows are `pending` (the walks at FeMoco-54) or `closed form`. Check the key rows against section "Dry-run results" of this plan: every row `measured` at the four small systems, the `naphthalene` walk `1,913 / 502 / 1,678` with dfthcbe 1,937 and paper 1,965.

- [ ] **Step 11: Run the per-commit tier and the linter, and commit**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1175 passed` (1112 + 63).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

```bash
git add src/dfthc/costs_reference.py src/dfthc/testing/provenance.py scripts/make_report_fixture.py \
    scripts/resource_report.py tests/fixtures/dfthcbe/report_toffolis.json tests/report/test_report.py \
    docs/resources.md docs/resources.json
git commit -m "Add the resource report, dfthc.costs_reference and the compile provenance"
```

- [ ] **Step 12: Report and stop for the large compiles**

Report the RED output, the counts of Steps 8 to 11, Step 10's time and its largest compile, and stop. The orchestrator then runs, from the lane, alone (master plan 12.9), each under the watchdog with a ledger row (large-compile count 0: not FeMoco-shaped):

```bash
PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 120 -- \
    "$VENV/bin/python" scripts/resource_report.py --large fe2s2
PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 120 -- \
    "$VENV/bin/python" scripts/resource_report.py --large fe4s4
```

Expected for each: the small systems reused (no compile line), then one line per program of the large system, `compiled (...)` for all 46 (section "Projections": every one allowed, at most 1.6 GB at `fe2s2` and 2.6 GB at `fe4s4`, about 22 and 40 minutes of compiles; `--large fe4s4` also calibrates on the `fe2s2` entries), no `CHECK FAILED`, exit 0, a `WATCHDOG` line below the 8 GB cap. A `refused` line is recorded and the run continues; a `killed` or `failed` line stops that system's run: every remaining program gets a `skipped (after ...)` line and the log line `<system>: stopped; the orchestrator reports the killed compile and escalates (master plan 12.6, 12.9)`, and the orchestrator does so before committing anything. Then the orchestrator commits the regenerated files on the branch:

```bash
git add docs/resources.md docs/resources.json
git commit -m "Report: the fe2s2 and fe4s4 compiles"
```

---

### Task 4.4: Documentation set and docstring audit

**Task id:** `4.4` · **Branch:** `phase4/4.4` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 4.1, 4.2, 4.3

**Files:**
- Create: `docs/architecture.md`, `docs/conventions.md`, `docs/verification.md`, `tests/test_docstrings.py`
- Modify: `docs/limitations.md` (consolidated), `docs/environment.md` (refreshed)

**Interfaces:**
- Consumes: every `Operator2` class of `dfthc.ops` and `dfthc.qpe` and its docstring; `dfthc.costs`; Task 4.3's `docs/resources.md` and `docs/resources.json` (linked from the new documents); `tests/canary/test_*.py` (the canary names).
- Produces: the documentation set of spec 5.2 (D-6); nothing for later tasks to import.

**Acceptance tests** (46 tests, all per commit):
- `test_every_operator_is_audited`: 32 classes, found recursively in `dfthc.ops` (re-exports included) and in `dfthc.qpe`, among them `PhaseGradientRotation`, `QROAMRead`, `DFTHCWalk`, `ControlledDFTHCWalk`, `ControlledWalkPower`, `QPERound`.
- `test_docstring_has_the_nine_sections_in_order[qualified_name]` (32): a summary line before the Action section; the nine sections of spec 9.0 in order, each once; the cost section names at least one `dfthc.costs` function and only existing ones; the example calls `compiled_resources`.
- `test_the_documentation_set_exists`; `test_documentation_links_resolve[document]` (10: every Markdown document under `docs/` outside `superpowers/`: `agents/HANDOFF.md`, `agents/LEDGER.md`, `architecture.md`, `conventions.md`, `environment.md`, `limitations.md`, `resources.md`, the two `upstream/` drafts, `verification.md`; each relative link resolves inside `docs/`, with its anchor); `test_limitations_match_the_canaries_and_spec_14` (each L-number once); `test_architecture_names_every_operator`.

**Definition of done:** RED seen; `46 passed` targeted; the per-commit tier passes; `ruff check .` clean; one commit. D-6 met.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_docstrings.py`:

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
"""Spec 9.0's sections in order, each a bold heading at the start of a docstring line. QROAMRead
and PhaseFixup, which also take dynamic arguments, title section 3 "Arguments"."""
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
    """The 30 operators of dfthc.ops (spec 9.1 to 9.22) and the two of dfthc.qpe (9.23, 11.1)."""
    assert len(CLASSES) == 32
    names = {cls.__name__ for cls in CLASSES.values()}
    assert {"PhaseGradientRotation", "QROAMRead", "DFTHCWalk", "ControlledDFTHCWalk",
            "ControlledWalkPower", "QPERound"} <= names


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

- [ ] **Step 2: Run it to confirm it fails**

Run: `scripts/test.sh tests/test_docstrings.py -q`
Expected: FAIL, `2 failed, 41 passed`: `test_the_documentation_set_exists` (`architecture.md`, `conventions.md`, `verification.md` missing) and `test_architecture_names_every_operator` (`FileNotFoundError`). The 32 docstring tests pass already: every class has the nine sections as merged (section "Docstring audit").

- [ ] **Step 3: Write the architecture**

Create `docs/architecture.md`:

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
| L2, paper blocks | `OuterPrepare`, `OuterUnprepare`, `InnerPrepare`, `InnerUnprepare`, `InnerSigns`, `TailAngleLoad`, `SFAngleLoad`, `AngleErase`, `SpinMux`, `SpinSwap`, `GivensRotation`, `GivensLadder`, `GivensLadderDagger`, `MajoranaGate`, `InnerSelect`, `InnerSelectDagger`, `RefT2`, `RefAB`, `ControlledRefT2`, `ControlledRefAB` | `dfthc.ops.prepare`, `dfthc.ops.angle_data`, `dfthc.ops.spin`, `dfthc.ops.givens`, `dfthc.ops.majorana`, `dfthc.ops.select`, `dfthc.ops.reflections` |
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
(spec 11.1). `PhaseFixup` is the operator form of the erase fix-up that `QROAMErase` and
`AngleErase` emit through `dfthc.ops.qroam.emit_phase_fixup`; the walk does not call it as an
operator. The toy walk of spec 11.4 (`ToyWalk`, `ControlledToyWalk`, `ToyControlledWalkPower` in
`dfthc.testing.toy_walk`) checks the phase estimation exactly.

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

- [ ] **Step 4: Write the conventions**

Create `docs/conventions.md`:

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
  `QPERound`'s earlier bits).
- Compilable integers fit in signed 64 bits, so wide table entries are tuples of 32-bit words,
  least significant first (L-24); dynamic table words are `int64` holding 32 data bits (L-26).
- Adjoints use the instance form `qp.adjoint(Op(...))` (L-25).

## Numbers in tests

Every comparison states both tolerances: `rtol=0` with an absolute `atol` (1e-12 for dense and
simulator comparisons unless a test names another), or an explicit relative bound with `atol=0`;
`tests/test_tolerance_hygiene.py` enforces it.
```

- [ ] **Step 5: Write the verification guide**

Create `docs/verification.md`:

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
`QROAMRead` with `QROAMErase`, the angle loads with `AngleErase`, and each prepare with its
unprepare (`tests/ops/test_certificates_qroam.py`, `test_certificates_angles.py`,
`test_certificates_prepare.py`).

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

- [ ] **Step 6: Consolidate the limitations**

Replace `docs/limitations.md` with (two paragraphs change: the introduction, which states that spec section 14 matches entry for entry, and L-6's mitigation, which names the third looping operator):

```markdown
# Known limitations

One entry per limitation of spec section 14, with the behaviour the canary in `tests/canary/`
pins today. A flipped canary is a signal to revisit the mitigation, not a failure to hide. All
observations are from 2026-11-11 (PennyLane 0.46.0.dev151, Catalyst 0.16.0.dev140) unless an entry
gives its own date. Entries marked **changed** differ from the spec's original wording. As of Task
4.4 (2026-11-15), this file consolidates L-1 to L-27, including L-26 (Task 0.4b), L-27 (Task
0.2c) and the ledger's later canary-state updates; `tests/test_docstrings.py` checks that every
limitation has an entry and that every canary named here exists.

## Decomposition rules, pins, control and templates

- **L-1, rules that allocate are excluded** (catalyst#3243). A rule declaring `work_wires=` is
  excluded even when pinned; with the pin there is no path. Canary:
  `test_allocating_rule_is_excluded`. Mitigation: explicit ancilla arguments (spec 7.4 rule 1).
- **L-2, measuring adjoint rules under control** (pennylane#9629), **changed**. Inside a control
  region the measuring `Adjoint(Op)` rule is skipped even when pinned, and the solver silently
  controls the unitary inverse of `Op`'s base rule instead (one Toffoli and four PauliX for a
  one-CNOT base rule, no measurement). It fails only when no unitary path exists. Canary:
  `test_measuring_adjoint_rule_has_no_controlled_path`. Mitigation: spec 7.4 rules 5 and 6; never
  place a measuring operator under `qp.ctrl`. Since G0 no program wraps a dfthc composite in an
  adjoint or a control (spec 7.4 rule 6), so no measuring rule is wrapped by design.
- **L-3, rule selection is cost-based.** With unweighted counts the solver picks the auto-derived
  inverse (two CNOTs) over the measured adjoint rule. Canary:
  `test_unweighted_solver_prefers_unitary_inverse`. Mitigation: weighted gate set, pins, T6.
- **L-4, pins apply to nested same-name operators.** Pinning `QROM` to `_select_swap` leaves its
  nested `QROM` without a path. Canary: `test_pin_applies_to_nested_instances`. Mitigation: pin dfthc
  names only (`dfthc.compile.register_rule`).
- **L-5, rules are compiled against abstract inputs** (catalyst#3172). A rule that branches on a
  dynamic argument sees a tracer and never specializes. Canary:
  `test_rule_branch_on_dynamic_value_is_not_specialized`. Mitigation: structure-determining data
  in `compilable_argnames` (spec 7.4 rule 3).
- **L-6, adjoint lowering with nested control flow** (catalyst#3217, open), **changed**: not
  reproduced; regression sentinel. The canary's reproducer (an auto-adjointed rule with nested
  `for_loop`, `cond` and a nested operator) compiles to `{"PauliX": "4"}`; the IRMapping crash was
  not reproduced. The canary fails if the crash returns; it cannot detect a fix. Canary:
  `test_auto_adjoint_with_nested_control_flow`. Mitigation kept: no control flow in auto-adjointed
  bodies (spec 7.4 rule 4). Since G0 the looped `QROAMRead`, `ControlledWalkPower` and
  `ToyControlledWalkPower` are never adjointed or controlled by construction (spec 7.4 rule 6),
  which the inventory test checks.
- **L-7, `C(ChangeOpBasis)` loses its saving under capture** (pennylane#10357). Controlling
  `change_op_basis(CNOT, Z, CNOT)` gives two Toffolis instead of zero. Canary:
  `test_controlled_change_op_basis_controls_everything`. Mitigation: the explicit controlled
  classes `ControlledDFTHCWalk` and `ControlledDFTHCBlockEncoding` (spec 7.4 rule 6).
- **L-8, generic controlled wrapping adds PauliX** (pennylane#10080). Generic control of a one-CNOT
  rule compiles to one Toffoli plus four PauliX. Canary: `test_generic_ctrl_adds_paulix_per_control`.
  Mitigation: explicit controlled classes (spec 7.4 rule 6).
- **L-9, function calls inside control regions** (catalyst#3330, catalyst#3384), **changed**.
  Controlling a `qp.capture.subroutine` leaves `C(PauliX)` undecomposed at `level="user"` (outside
  the gate set), and compiling for execution fails. Canary:
  `test_function_call_in_ctrl_region_raises`. Mitigation: no `Subroutine`; explicit controlled
  classes (spec 7.4 rule 6).
- **L-10, compilation can depend on gate order** (catalyst#3431). Canary:
  `test_order_dependent_compile`, built from the issue's reproducer in
  `tests/canary/reproducers/catalyst_3431.py`, with the observed outcome per order recorded in
  `catalyst_3431.json`. Mitigation: an occurrence is a compiler blocker; preserve a reproducer
  and escalate (spec 14, L-10).
- **L-11, resource functions run with capture enabled** (pennylane#10162). Canary:
  `test_resource_function_runs_with_capture_enabled`. Mitigation: `dfthc.compile.resource_key`
  builds keys with capture paused.
- **L-12, select-swap `QROM` is not self-inverse and counts symbolically.** A dirty 16-entry
  `QROM` (select-swap chosen) applied twice has overlap 0.8125 with the input; its counts are
  symbolic. Canaries: `test_select_swap_qrom_twice_is_not_identity`,
  `test_select_swap_qrom_counts_are_symbolic`. Mitigation: `QROAMRead` with exact CSWAPs and a
  `PhaseFixup` that applies `CZ`/`Z` directly.
- **L-13, `Select` with shared target wires crashes the compiler** (`'qref.operator' op qubit
  operands must refer to distinct qubits`). Canary: `test_select_with_shared_target_crashes`.
  Mitigation: our own unary-iteration helper.
- **L-14, `MultiControlledX` without work wires has no path.** Canary:
  `test_mcx_without_work_wires_has_no_path`. Mitigation: explicit work wires.
- **L-19, `UniformPrep` and `AliasSampling` need arbitrary rotations**, **changed**. For a
  non-power-of-two size (L = 5) neither has a path to the gate set. For a power-of-two size
  `AliasSampling` compiles, because its uniform preparation is then Hadamards only. Canary:
  `test_template_has_no_toffoli_level_path` (two cases). Mitigation: `FlaggedUniformPrep`.

## Operator arguments

- **L-23, list-valued dynamic arguments are not canonicalized** (pennylane#10060, open). The base
  class keeps a list as a list. Canary: `test_dynamic_list_argument_not_canonicalized`.
  Mitigation: dfthc operators with dynamic arguments convert lists to arrays in `__init__`.
- **L-24, compilable integers must fit in signed 64 bits** (new, found by the Phase 0 dry-run).
  `2**63 - 1` compiles as a compilable argument; `2**63 + 1` does not (`CompileError`, MLIR "Large
  integer attributes currently not supported"). Canary: `test_compilable_int_must_fit_int64`.
  Mitigation: tables store wide entries as tuples of 32-bit words, least significant word first.
- **L-25, `qp.adjoint(OpClass)(*args)` traces positional arguments** (new). Under capture a
  compilable argument passed positionally is rejected ("must be a concrete, compile-time
  constant"); keyword arguments, `qp.adjoint(Op(...))` and `qp.adjoint(lambda: Op(...))()` work.
  Canary: `test_adjoint_of_class_traces_positional_arguments`. Mitigation: always use the
  instance form `qp.adjoint(Op(...))`.
- **L-26, unsigned dynamic arguments get no decomposition rule** (new, found by Task 0.9b).
  Catalyst's `_MLIR_DTYPES_TO_PY_DTYPES` (`catalyst/decomposition/type_utils.py`) maps only `i1`,
  `i8`, `i16`, `i32` and `i64` (and floats and complex numbers). On 2026-11-12 an `Operator2` whose
  dynamic argument is a `uint32` array warns "Failed to synthesize the distributed decomposition
  rules for <Op>: 'ui32'" and then fails with `CompileError` "Decomposition rule not found for
  operator '<Op>'"; the same operator with `int32` or `int64` data compiles. Canary:
  `test_unsigned_dynamic_argument_has_no_rule`. Mitigation: compilable table words are `int64`
  holding 32 data bits each (ledger decision "Table words are `int64`").

## Measurements, analysis, device level, warnings and compile time

- **L-15, measurement-related bugs.**
  - Hardware-like postselection (pennylane#8379), **changed**: `qp.qjit(capture=True)` now rejects
    `postselect_mode="hw-like"` with a `CompileError`. Canary:
    `test_hw_like_postselection_wrong_under_capture`.
  - Measurements nested in `qp.cond` (catalyst#2382), **changed**: under one-shot execution the
    canary's nested measurement runs on every shot (mean 0.5 within five standard deviations);
    the bug was not reproduced. This is a regression sentinel: it fails if the bug returns, but
    because the reproducer is not affected it cannot detect a fix. Canary:
    `test_mcm_inside_cond_missed`.
  - `qp.pauli_measure(postselect=...)` is ignored (catalyst#3338): the postselected outcome
    appears on about half the shots. Canary: `test_pauli_measure_postselect_ignored`.
  - Mitigation for all three: spec 12.2 (no hardware-like postselection, no measurements inside
    `qp.cond`, no `pauli_measure` postselection).
- **L-16, execution decomposes again to the device gate set** (catalyst#3275). At
  `level="device"` a `TemporaryAND` becomes a Toffoli; a decomposed `Operator2`
  leaves no named operation. Canaries: `test_device_pass_runs`,
  `test_device_level_has_no_operator_op`. Mitigation: counts at `level="user"`; closure at
  `level="device"` (`compiled_resources(level="device")`).
- **L-17, `qp.analyze` behaviour.** It accepts only qjit-compiled QNodes (`ValueError`); an
  unhinted conditional counts its most expensive branch and warns; branch-probability hints give
  expected counts, summed per gate type over the program and rounded up once (four one-X
  conditionals at probability 1/2 count 2); a runtime loop bound gives symbolic counts; a
  `num-iters` hint makes them concrete. Canaries: `test_analyze_requires_qjit`,
  `test_unhinted_cond_counts_max_branch_with_warning`, `test_branch_prob_hint_gives_expected_counts`,
  `test_dynamic_loop_counts_are_symbolic`, `test_num_iters_hint_makes_loop_concrete`.
  Mitigation: `compiled_resources()`, hints, static loop bounds.
- **L-18, rule-lowering warnings are silenced by default** (catalyst#3190), and a solver failure
  reports "the rules file could not be parsed" (catalyst#2767); the message also names the
  operator without a rule. Canaries: `test_warnings_silenced_without_env`,
  `test_warnings_visible_with_env`, `test_unreachable_op_reports_missing_rule`. Mitigation: spec
  7.1 and 7.4 rule 12.
- **L-20, undecomposed `Operator2` at runtime** (catalyst#3430), **changed**. An `Operator2`
  listed in the gate set stays undecomposed at `level="user"`; with a rule, the device pass
  decomposes it and the result is correct; without a rule, compiling for execution fails. It no
  longer reaches runtime as a bare named gate. Canary:
  `test_undecomposed_operator_op_reaches_runtime`. Mitigation kept: the `level="device"` closure
  test of every composite operator (T5).
- **L-21, distinct tables add rule traces and compile cost grows with table size**
  (catalyst#3173). Canary: `test_each_distinct_table_is_a_separate_rule_compile` (per commit).
  On 2026-11-12 each distinct table traces its rule exactly once per compile, and identical
  tables in one program share that trace: compile time grows with table size, and each distinct
  table adds its own rule trace. (Task 0.4 read "twice" on 2026-11-11; the second trace was the
  decoration-time compile of L-27. The canary now keeps that compile off and pins the doubled
  count separately.)
  Benchmark: `test_compile_time_scaling` (nightly) compiles 1, 2 and 4 distinct 64-entry tables
  and single tables of 64, 1,024 and 8,640 entries, each in a fresh process under the watchdog.
  It fails at more than twice `tests/canary/compile_baseline.json` in CPU time (across the compile
  only: the process plus the Catalyst compiler subprocess; and more than 2 s above it, because
  sub-second cases are noise-dominated) or peak memory (`peak_mb`: the summed physical footprint
  of the process tree, as the watchdog measures it), or if the 8,640-entry case costs less than 5
  times the CPU time of the 64-entry case. Wall time is recorded for information.
  The benchmark programs disable the decoration-time compile (L-27). On the development laptop
  (2026-11-12, after Task 0.2c) the ratio was 55, and one 8,640-entry table took 25 s wall, 25 s
  CPU and 973 MB (Task 0.4, with the decoration-time compile and under memory pressure: ratio 84,
  37 s wall, 36 s CPU and 1.5 GB). Mitigation: few distinct table operators
  per walk; the K-1 budget.
- **L-22, unresolved loops give symbolic counts.** Shared with L-12 and L-17
  (`test_select_swap_qrom_counts_are_symbolic`, `test_dynamic_loop_counts_are_symbolic`).
  Mitigation: loops unrolled in Python or with static bounds; T4 asserts `is_symbolic == False`.
- **L-27, `qp.qjit` compiles a fully annotated function ahead of time at decoration** (new, found
  by the Opus review of Task 0.9c; zero arguments are the case `compiled_resources` hits).
  `qp.qjit` AOT-compiles at decoration when every parameter has a Catalyst-supported type
  annotation (`params_are_annotated` in `catalyst/tracing/type_signatures.py`); a zero-argument
  function satisfies this rule vacuously. With `enable_aot_compilation=True` (the default,
  `catalyst/jit.py`), `QJIT.__init__` compiles a linked binary. `qp.analyze` creates a separate
  `QJIT` with AOT disabled and compiles through `mlir_opt`, so an analyze-only fully annotated
  program is compiled twice, the binary is never used, and the first capture stays in memory
  during the second compile. On the K-1 stand-in the mid10 controlled walk fell from 2,552 MB and
  73 s to 1,536 MB and 26 s, with identical counts, when the first compile was switched off.
  Canaries: `test_zero_argument_qjit_compiles_at_decoration`,
  `test_fully_annotated_qjit_compiles_at_decoration`,
  `test_unannotated_parameter_is_not_compiled_at_decoration`. Mitigation: use
  `@qp.qjit(capture=True, enable_aot_compilation=False)` for analyze-only programs
  (`compiled_resources` does). D-3 needs only the analyze compile; a linked binary, if wanted, is
  compiled in a separate process. Programs that are executed keep the default, because the call
  reuses the AOT binary.
```

- [ ] **Step 7: Refresh the environment page**

Replace `docs/environment.md` with (new: the load-limit item on the report's compiles and the sections "The reference environment `.venv-ref`" and "The resource report"):

````markdown
# Environment

The canonical environment is `.venv` in the main checkout, CPython 3.12 (spec section 7.1).
Every git worktree uses it through `scripts/test.sh`; nothing is installed in editable mode.

## Snapshot

| Package | Version |
| --- | --- |
| pennylane | 0.46.0.dev151 |
| pennylane_catalyst | 0.16.0.dev140 |
| pennylane_lightning | 0.46.0.dev34 |
| jax, jaxlib | 0.7.1 |
| pytest / pytest-xdist / ruff | 9.2.0 / 3.8.0 / 0.17.4 |

`requirements-nightly.lock` is the complete `pip freeze`; `tests/test_infrastructure.py` checks
that it pins the installed PennyLane, Catalyst and Lightning versions.

## Recipe

`scripts/setup_env.sh` runs exactly the commands of spec section 7.1 in a fresh `.venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane-catalyst --upgrade
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane --upgrade
pip install pytest pytest-xdist ruff
pip freeze > requirements-nightly.lock
```

## Pitfalls (observed on 2026-11-11)

- Always use plain `pip`, never `uv pip`: uv resolves `pennylane-catalyst` from PyPI and installs
  the stable release.
- The second install prints `ERROR: ... pennylane-catalyst 0.16.0.dev140 requires
  pennylane==0.46.0-dev149, but you have pennylane 0.46.0.dev151`. The override is intended.
- Every compile prints `ld: warning: object file ... was built for newer 'macOS' version (28.0)
  than being linked (27.0)`. It is benign.
- `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` must be exported before Python starts; Catalyst
  reads it once at import. `scripts/test.sh` exports it, and `tests/conftest.py` re-enables the
  warnings as a second line of defence.
- `--pre` with the TestPyPI index also pulls dev builds of dependencies (`termcolor`, `urllib3`);
  the lock file records them.
- A missing decomposition path surfaces as `failed to load built-in decomposition rules ...: the
  rules file could not be parsed`; the real cause, `Decomposition rule not found for operator
  '<op>'`, is in the same error message.

## Load limits

The development laptop has 10 cores and 16 GB, and the user works on it while agents run
(master plan section 12.9).

- `scripts/test.sh` runs 2 xdist workers (`DFTHC_TEST_WORKERS` overrides; `0` runs serially) at
  lowered priority: `nice -n 10`, or, in the agent sandbox, where setpriority() is forbidden, the
  utility QoS clamp of `taskpolicy`.
- Each process tree has a sampled 8 GB memory cap: the watchdog kills it at the first sample above
  7.5 GB, so the overshoot past 7.5 GB is bounded by 50 ms of growth. A heavy run (above 4 GB, more than 2 busy cores,
  longer than 30 minutes, or any FeMoco-shaped compile) is started only by the orchestrator, alone,
  under `PYTHONPATH=src .venv/bin/python -m dfthc.testing.watchdog -- <command>`, and logged in the
  ledger with its peak memory and wall time.
- Every compile of the resource report runs in its own process under `run_watched`: a 4 GB cap
  at the small systems, the 8 GB cap for `--large`, whose programs are projected first
  (`dfthc.testing.provenance.project_compile`) and compiled only within 7.5 GB and 60 minutes.
  The first killed or failed compile of a `--large` run stops it, and so does the first refusal
  at FeMoco-54; the remaining programs are recorded as skipped, and the orchestrator escalates.

## The reference environment `.venv-ref`

`.venv-ref` in the main checkout holds Qualtran 0.7.0 and dfthcbe at commit
`930255020cec30172e0a7d17c0071d9476a687d1` (spec 8.6, Q-2). It only generates the reference
fixtures under `tests/fixtures/dfthcbe/`: `scripts/make_dfthcbe_fixtures.py` (Task 1A.6) and
`scripts/make_report_fixture.py` (Task 4.3). The recipe is Task 1A.6, Step 7, of the Phase 1 data
plan; nothing in the main environment imports Qualtran.

```bash
.venv-ref/bin/python scripts/make_report_fixture.py --out tests/fixtures/dfthcbe
```

## The resource report

`scripts/resource_report.py` writes `docs/resources.md` and `docs/resources.json` (spec 13.3). It
compiles every program of the walk at the H2 systems and `naphthalene`, each in its own process
under the watchdog, and reuses every compile cached under the current code hash:

```bash
PYTHONPATH=src .venv/bin/python scripts/resource_report.py
```

The large systems compile only with `--large fe2s2`, `--large fe4s4` or `--large femoco54`,
which project each program first and are heavy runs: the orchestrator starts them alone, under
the watchdog, and logs each in the ledger.

## Bumping the nightlies

Bumps are orchestrator work, done at a phase gate.

1. Recreate `.venv` with the recipe (`rm -rf .venv && scripts/setup_env.sh`).
2. Run the per-commit tier, which holds most canaries: `scripts/test.sh -m "not slow and not release"`.
   The gate's nightly-tier run covers the slow canaries; the release tier is not run for a bump.
3. Update spec section 14 and `docs/limitations.md` for every canary whose state changed.
4. Refresh the lock file, this snapshot and `tests/canary/compile_baseline.json`
   (`DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 scripts/test.sh tests/canary/test_l21_compile_time.py`).
5. Commit on a branch.
````

- [ ] **Step 8: Run the tests**

Run: `scripts/test.sh tests/test_docstrings.py -q`
Expected: `46 passed` (2.1 s, after the review's fixes).

- [ ] **Step 9: Run the per-commit tier and the linter, and commit**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1221 passed` (1175 + 46).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

```bash
git add docs/architecture.md docs/conventions.md docs/verification.md docs/limitations.md \
    docs/environment.md tests/test_docstrings.py
git commit -m "Add the architecture, conventions and verification documents and the docstring audit"
```

- [ ] **Step 10: Report**

Report the RED output of Step 2 and the counts of Steps 8 and 9. The orchestrator then presents the report (its FeMoco-54 summary, qubit section and deferred refinements) and the K-10 result to the user for the Q-1 decision (master plan 12.6, G4).

---

### Task 4.5: The single FeMoco walk and controlled-walk compile, and report finalization

**Task id:** `4.5` · **Branch:** `phase4/4.5` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 4.4, the Q-1 decision, and Q1b if chosen

**Files:**
- Create: `tests/report/test_femoco_cache.py`
- Modify: `docs/resources.md`, `docs/resources.json` (regenerated by the orchestrator's compile)

**Interfaces:**
- Consumes: Task 4.3's `scripts/resource_report.py` (`build_system`, `build_rows`, `current`, `compile_counts`, `measure`), `dfthc.testing.provenance.code_hash`, `dfthc.costs.iterative_qpe`, `dfthc.compile.ANALYZE_NAMES`.
- Produces: the cached FeMoco-54 compile (D-3); Task 5.2 reuses it while the code hash matches.

**Acceptance tests** (2 tests: 1 per commit, 1 `release` and `femoco`):
- `test_stale_femoco_entry_names_the_recompile` (per commit): an entry of another code hash is no measurement; the check names `scripts/resource_report.py --large femoco54`.
- `test_femoco_cache` (`release`, `femoco`): the current entries of `femoco54/DFTHCWalk` and `femoco54/ControlledDFTHCWalk` hold T = 9,793, M = 2,524, A = 8,675 and T = 9,795, M = 2,524, A = 8,677, gates within the gate set, the environment, the compile time and the peak memory; the partition rows times their calls per walk equal those totals exactly; the report's FeMoco-54 QPE summary gives T, M, A and wires from the cached controlled walk for t = 1, 2, 3, 10, 20, 23 (A = (2^t − 1) A_C(W) + (t^3 − t)/6), and `costs.iterative_qpe` agrees for t = 1, 2, 3. Nothing is compiled.

**Definition of done:** the RED run of Step 2; the orchestrator's compile (Step 4) with two ledger rows; `test_femoco_cache` passes (Step 5); the regenerated report committed. D-3 met; then G4 can complete.

- [ ] **Step 1: Write the test file**

Create `tests/report/test_femoco_cache.py`:

```python
"""The cached FeMoco-54 compile of the walk and the controlled walk (spec 2.3 D-3, 13.3; master
plan Task 4.5).

Nothing here compiles. The release-tier test reads ``docs/resources.json``: the entry of the
current code hash for each program holds the counts of spec 9.24 (walk T = 9,793, M = 2,524,
A = 8,675; controlled walk T = 9,795, M = 2,524, A = 8,677), the code hash, the environment, the
wall time and the peak memory, and the partition rows of the report (tape mode, no compile) times
their calls per walk give exactly the cached T, M and A. A missing or stale entry fails and names
the recompile, which only the orchestrator runs.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dfthc import costs
from dfthc.compile import ANALYZE_NAMES
from dfthc.params import FEMOCO54
from dfthc.testing.provenance import code_hash

REPO = Path(__file__).resolve().parents[2]
EXPECTED = {"DFTHCWalk": (9793, 2524, 8675), "ControlledDFTHCWalk": (9795, 2524, 8677)}
RECOMPILE = ("scripts/resource_report.py --large femoco54 (the orchestrator, alone, under the "
             "watchdog; master plan 12.9 and Task 4.5)")


def _load_script():
    spec = importlib.util.spec_from_file_location("resource_report",
                                                  REPO / "scripts" / "resource_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["resource_report"] = module
    spec.loader.exec_module(module)
    return module


report = _load_script()


def femoco_cache_problems(record: dict, hash_: str) -> list[str]:
    """What is wrong with the cached FeMoco-54 compiles of ``record`` under code hash ``hash_``:
    a missing entry (naming the recompile), counts other than spec 9.24's, or an entry without
    its provenance."""
    problems, cache = [], record.get("compiled", {})
    for program, expected in EXPECTED.items():
        key = f"femoco54/{program}"
        entry = report.current(cache, key, hash_)
        if entry is None:
            others = sorted({e["code_hash"][:12] for e in cache.get(key, [])})
            problems.append(f"{key}: no compile for code hash {hash_[:12]} (cached: "
                            f"{others or 'none'}); recompile with {RECOMPILE}")
            continue
        counts = (entry["toffoli"], entry["measurements"], entry["measured_and_uncomputes"])
        if counts != expected:
            problems.append(f"{key}: cached T, M, A {counts} != {expected}")
        if not set(entry["counts"]) <= ANALYZE_NAMES:
            problems.append(f"{key}: gates outside the gate set {set(entry['counts'])}")
        for field in ("environment", "seconds"):
            if not entry.get(field):
                problems.append(f"{key}: no {field}")
        if not entry.get("watchdog", {}).get("peak_mb"):
            problems.append(f"{key}: no peak memory")
    return problems


def test_stale_femoco_entry_names_the_recompile():
    """An entry of another code hash is not a current measurement: the check fails and names
    the recompile (per commit; nothing is compiled)."""
    stale = {"code_hash": "0" * 64, "status": "compiled", "toffoli": 9793, "measurements": 2524,
             "measured_and_uncomputes": 8675, "counts": {}, "environment": {"python": "3.12"},
             "seconds": 1.0, "watchdog": {"peak_mb": 1.0}}
    problems = femoco_cache_problems({"compiled": {"femoco54/DFTHCWalk": [stale]}}, "1" * 64)
    assert len(problems) == 2
    assert problems[0].startswith("femoco54/DFTHCWalk: no compile for code hash 111111111111 "
                                  "(cached: ['000000000000']); recompile with")
    assert "--large femoco54" in problems[1]


@pytest.mark.release
@pytest.mark.femoco
def test_femoco_cache(monkeypatch):
    """D-3 from the cache: the current entries hold spec 9.24's counts with their provenance,
    and the sum over the partition rows of calls per walk times the closed form equals the cached
    walk and controlled-walk totals of T, M and A; the report's FeMoco-54 QPE summary gives T, M
    and A from the cached controlled walk (spec 11.3; A = (2^t - 1) A_C(W) + (t^3 - t)/6).
    Nothing is compiled."""
    monkeypatch.setattr(report, "compile_counts", lambda *args: pytest.fail("no compile"))
    monkeypatch.setattr(report, "measure", lambda *args: pytest.fail("no compile"))
    record = json.loads((REPO / "docs" / "resources.json").read_text())
    hash_ = code_hash()
    assert femoco_cache_problems(record, hash_) == []
    s = report.build_system("femoco54")
    rows = report.build_rows(s)
    cache = record["compiled"]
    for program, calls in (("DFTHCWalk", "calls"), ("ControlledDFTHCWalk", "controlled_calls")):
        entry = report.current(cache, f"femoco54/{program}", hash_)
        parts = [row for row in rows if row.partition and getattr(row, calls)]
        total = (sum(getattr(r, calls) * r.closed.toffoli for r in parts),
                 sum(getattr(r, calls) * r.closed.measurements for r in parts),
                 sum(getattr(r, calls) * r.uncomputes for r in parts))
        assert total == (entry["toffoli"], entry["measurements"],
                         entry["measured_and_uncomputes"]) == EXPECTED[program]
    cwalk = report.current(cache, "femoco54/ControlledDFTHCWalk", hash_)
    summary = record["systems"]["femoco54"]
    assert summary["qpe_basis"].startswith("T, M and A from the compiled controlled walk")
    for bits in (1, 2, 3, 10, 20, 23):
        expected = {"toffoli": (2**bits - 1) * cwalk["toffoli"] + (bits**3 - bits) // 6,
                    "measurements": (2**bits - 1) * cwalk["measurements"] + bits,
                    "measured_and_uncomputes": (2**bits - 1) * cwalk["measured_and_uncomputes"]
                    + (bits**3 - bits) // 6, "wires": report.qpe_wires(s, bits)}
        assert summary["qpe"][str(bits)] == expected, bits
        if bits <= 3:
            qpe = costs.iterative_qpe(FEMOCO54, bits)
            assert (qpe.toffoli, qpe.measurements) == (expected["toffoli"],
                                                       expected["measurements"])
```

- [ ] **Step 2: Run it: the release test fails until FeMoco-54 is compiled**

Run: `scripts/test.sh tests/report/test_femoco_cache.py -q -m "not slow and not release"`
Expected: `1 passed` (the stale-entry check).

Run: `scripts/test.sh tests/report/test_femoco_cache.py -q -m release`
Expected: FAIL, `1 failed`: `femoco_cache_problems` lists `femoco54/DFTHCWalk: no compile for code hash ...; recompile with scripts/resource_report.py --large femoco54 ...` (and the same for the controlled walk).

- [ ] **Step 3: Run the per-commit tier and the linter, commit, and stop**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `1222 passed` (1221 + 1).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

```bash
git add tests/report/test_femoco_cache.py
git commit -m "Add the cached FeMoco-54 compile check"
```

Report the two runs of Step 2 and stop: the compile is the orchestrator's.

- [ ] **Step 4: The FeMoco-54 compile (orchestrator only, alone, under the watchdog)**

From the lane, with no implementer active and nothing else running (master plan 12.9, rule 3), and with a ledger row per program (large-compile count 1 each: two of the six):

```bash
PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 150 -- \
    "$VENV/bin/python" scripts/resource_report.py --large femoco54
```

The script first projects `DFTHCWalk` and `ControlledDFTHCWalk` from their current entries at the H2 systems, `naphthalene`, `fe2s2` and `fe4s4` (expected near or below the dry run's 4.5 and 3.8 GB, which used the four small systems only; about 1.9 GB by the captured-equation model), then compiles each under the 8 GB cap, killed at 7.5 GB. Expected: `femoco54/DFTHCWalk: compiled (...)` and `femoco54/ControlledDFTHCWalk: compiled (...)`, no `CHECK FAILED` (the cross-check of the partition against the compiled walks runs here), exit 0. If `DFTHCWalk` is `refused` (projection above 7.5 GB or 60 min), `killed` or `failed`, the script stops: `ControlledDFTHCWalk` gets the line `femoco54/ControlledDFTHCWalk: skipped (after femoco54/DFTHCWalk ...)` and is never compiled, and the log names the escalation (`test_large_stops_after_a_refusal_at_femoco`); the same holds if the controlled walk itself is refused, killed or fails. The orchestrator then records the run in the ledger and escalates to the user with the projection (a machine with at least 32 GB, or an approved idle-time window with a higher cap; master plan section 2, "Where compiles above the cap run"); a mismatch against the closed form goes to a Tier A fix dispatch.

- [ ] **Step 5: Check the cache and commit the report**

Run: `scripts/test.sh tests/report/test_femoco_cache.py -q -m release`
Expected: `1 passed`.

```bash
git add docs/resources.md docs/resources.json
git commit -m "Report: the FeMoco-54 walk and controlled walk, compiled once"
```

The report's FeMoco-54 rows of the walks now read `measured` with `9,793 / 2,524 / 8,675` and `9,795 / 2,524 / 8,677`, its QPE section says "T, M and A from the compiled controlled walk", and its summary gives walk −204 and controlled walk −202 against the paper and +152 and +153 against dfthcbe.

---

## Interfaces for later tasks

- **Regenerating the report** (Q1b, Task 5.2): `PYTHONPATH=src "$VENV/bin/python" scripts/resource_report.py` recompiles every program of the small systems whose entry has another code hash and keeps the others; a code change (not a docstring or comment) changes `code_hash()`, so the large systems' rows become `pending` until the orchestrator reruns `--large` (master plan section 11, Task 5.2: `fe2s2` and `fe4s4` are not recompiled in Phase 5; their earlier measurements stay as history).
- **The cache check** (Task 5.2's release tier): `tests/report/test_femoco_cache.py::test_femoco_cache` passes while the FeMoco-54 entries' hash matches; after a code change it fails and names the recompile, the budget's contingency pair (master plan 12.9, rule 5).
- **Q1b** changes `WalkLayout` and the block-encoding rule: the report's rows follow the walk's calls, so new operators appear as rows by themselves; their closed forms need entries in `closed_form`, their comparisons keys in `NAME_KEYS` and parts in `costs_reference.delta_causes` (and `iteration_counts` for a new walk), or the script fails with a `KeyError` or an unexplained delta.
