# DFTHC Phase 5: Final Review, Fix Wave, Release Tier and Final Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Review the whole project once more against the spec's definition of done (D-1 to D-7), fix what the review finds, run the release tier once, record the evidence of every D criterion in the ledger, and merge into local `main`; nothing is pushed.

**Architecture:** Task 5.1 is read-only. The orchestrator cuts the project's review package (`993b8fb..main`, 10.4 MB) into four area packages; Opus and GPT-5.6 review each area in parallel, one pair at a time; the orchestrator merges the eight reviews into one findings list, `docs/agents/final-review.md`, with a decision per finding and a disposition per open ledger row. Task 5.2 gives that list to one Tier A fixer on `phase5/5.2`, re-reviewed by GPT-5.6 and Opus. A fix of a hashed file makes the orchestrator regenerate the small-system report and recompile the FeMoco-54 pair once. After the user's go-ahead, asked together with the confirmation of Q-1 variant (b), the orchestrator runs the release tier alone under the watchdog, checks D-1 to D-7, merges, and closes the ledger.

**Tech Stack:** git (local branches, the two worktree lanes), the superpowers review package and subagent workflow, pytest with pytest-xdist through `scripts/test.sh`, `dfthc.testing.watchdog`, `dfthc.testing.provenance.code_hash`, `scripts/resource_report.py` (orchestrator only), Python 3.12 in the main checkout's `.venv`.

This file holds Tasks 5.1 and 5.2 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 11). It was written at G4 under the rolling-wave rule (master plan 12.7) against the tree of `c443c51`, the head of Task 4.5's branch `phase4/4.5`. Task 4.5 merged while it was written (`d433c4e`; ledger and report plan revision 9 at `99c5ffb`): the merge brought one review fix, in `tests/report/test_femoco_cache.py` only (the same two tests, stronger checks, no hashed file), so the collection counts, the test ids, the code hash, the report's state, the D-3 and D-7 evidence and the documentation tests were measured again on the tree of `99c5ffb` and are unchanged except the D-7 ledger check, which now finds Task 4.5 merged (`stages/P5/main-*.log`). The orchestrator's G4 checks then closed three ledger rows and corrected `docs/verification.md` (`1c15419`, documents only); the open ledger rows below are those of `1c15419`. G4 completes when the orchestrator has checked D-3, D-4 and D-6 on the final artifacts and this plan is approved; then Task 5.1 starts.

**Revision 1** (2026-11-15). First version.

**Revision 2** (2026-11-15), after the GPT-5.6 review `.superpowers/sdd/phase5-plan-review-1.md` (Needs fixes: 0 Critical, 5 Important, 4 Minor; the area split, the counts, the hashed-fix procedure, the 17 edit anchors and the 72-row table verified). Revision 1's assembled plan is kept as `.scratch/plan-check/rev1-p5/2026-11-11-dfthc-phase5-final.md`. All nine findings are applied as the review proposes:
- **I-1, the Q-1 move left G4 and the spec contradictory: fixed.** New edit P5-M11 rewrites master plan 12.6's G4 row in the review's wording, adding that this working choice is the "Q-1 decision" that sections 4 and 5 and Tasks Q1b and 4.5 refer to, so their dependency labels stay true. P5-M9 and P5-S1 lose "the user may override before Task Q1b runs" (P5-S1 now has the review's text). The same stale timing sat in two more master-plan places, so two edits join: P5-M12 (Task Q1b's "Opened", which also said "only if the user chooses variant (b)", while the working choice opened it) and P5-M13 (section 2's Q-1 decision row). Checked and left unchanged, because neither names a time and both agree with the Task 5.2 confirmation: master plan 12.6's introduction ("asked only about escalations, the Q-1 decision and the go-ahead for the release run") and spec 16's "Q-1 has a working choice, variant (b), which the user may override" (line 1647); Task 4.5's "Depends on: ... the Q-1 decision" (line 988) is now defined by P5-M11. The ledger's decision rows are records and stay. The plan has 20 edits (13 master plan, 2 spec, 1 HANDOFF, 4 ledger): all old texts occur exactly once on `main` at `1c15419`, every master-plan old text once in the template and never in `_global_constraints_q1b.md`, and the mirroring replay passes (section "Mirroring at approval").
- **I-2, the Q-1 override misstated the FeMoco budget: fixed** at all three places (P5-M8, the user message of Task 5.2, Step 7, and risk 5), in the review's words: when the answer is taken the contingency pair has not run, so a baseline pair would be compiles 4 and 5 of about 6; only compiles beyond about six need approval. Step 10's second-wave case now says the same (compiles 6 and 7).
- **I-3, D-7 overclaimed: fixed.** Task 5.1, Step 1 snapshots the local remote configuration and the remote-tracking refs (`git-remote-config.before`, `remote-refs.before`; their content on `1c15419` is quoted); Task 5.2, Step 12 snapshots them again after the merge and compares with `cmp`. D-7 (P5-L4's row, the checklist of Step 11, Step 13) is worded as a process-control assertion: every Phase 5 prompt forbade push, fetch and remote changes (the two re-review prompts and the fix dispatches now say so too), and the local evidence is unchanged; without contacting the remote it is not a forensic proof that no push occurred.
- **I-4, the accepted load-only failure could not satisfy Step 11 and G5: fixed** with the review's conditional evidence form. Step 10: a known load-sensitive failure must pass twice alone (`5.2-rerun-<k>.log`); then the release result is accepted with a documented load-only exception. Step 11 checks that no D-1 to D-4 test failed, records the canary's node and serial passes under D-5, and does not claim an all-passed summary; the D-1 and D-5 rows and the G5 row template carry both forms (`<1393 + N> passed`, or `<f> failed, <1393 + N − f> passed` with the exceptions named).
- **I-5, the release estimate was not fully backed by P5 evidence: fixed.** New `stages/P5/release-estimate-evidence.log`, written by `extract_release_evidence.py` (standard library only), quotes every source path with the exact result, "s call" and `WATCHDOG` lines the projection uses and recomputes the arithmetic (25.2 minutes, at most 33.3 if the QPE test starts last, 75.8 under load; largest tier peak 2,776.1 MB). The QPE row is a 487 s planning allowance whose post-Q1b runtime is expected, but not proved, to be lower; the 25 to 35 and 80 minutes are labelled a projection everywhere (P5-M7, the estimate section, Steps 7 and 10).
- **M-1: fixed.** D-2 counts E-1, E-2, E-3 and E-4 separately, `33`, `4`, `3`, `2`, with 42 as the cross-check (checked against the parametrizations of `tests/integration/test_end_to_end.py` at `1c15419`).
- **M-2: fixed.** P5-M7 and P5-L3 use 1,272 + N_pc, 86 + N_n, 35 + N_r and 1,393 + N_pc + N_n + N_r; the plan defines N as their sum where it writes 1,393 + N.
- **M-3: fixed.** "At most one hashed change" is now "at most one report-invalidating hashed fix wave before the release run", with the review's added sentence, in the changes list, section "Hashed and unhashed fixes" and P5-M7.
- **M-4: fixed.** Mirroring step 4 starts on its own line.

Revision 2 ran no pytest and no Python that imports `dfthc`, PennyLane, Catalyst or JAX (the orchestrator's G4 gate tier was running): only text edits, standard-library scripts and git reads. The staged documentation test of revision 1 (`49 passed` with the edited HANDOFF and ledger and a `final-review.md`) was not rerun; revision 2 changes no HANDOFF text and only the D-7 wording and the Task 5.2 row in the ledger, and its edits add no Markdown link (the edit script checks that); the orchestrator runs the documentation tests at approval (mirroring step 4).

**Code provenance.** Phase 5 adds no code before the final review, so this plan has no code to dry-run. Everything it does contain was executed on 2026-11-15, in a scratch replica of the tree of `c443c51` (`.scratch/plan-check/repo-phase5`, from `git -C .worktrees/lane-a archive c443c51`; not a git repository) through a capped, locked pytest wrapper (`.scratch/plan-check/p5test.sh`: one xdist worker, every session under the watchdog with a 3,000 MB cap and a 10-minute limit, lock `tmp/p5test.lock`), or read-only against the main checkout's git:
- the collection count of every tier (`--collect-only`; `stages/P5/collect-*.log`);
- the documentation tests, on the replica (`doc-tests.log`) and on a staged copy that holds the edited HANDOFF and ledger and a `docs/agents/final-review.md` in this plan's format (`repo-phase5-stage`; `stage-doc-tests.log`, `stage-collect-per-commit.log`, `stage-collect-all.log`);
- three release tests, each alone (`release-e1-631g-00000011.log`, `release-e1-sto3g-S1-all.log`, `release-e3-seeded-fixed.log`), and the `-rA` output format (`rA-probe.log`);
- the whole review package and its four areas (`review-993b8fb..c443c51.diff`, `review_areas.sh`, `areas/`);
- the report's state, the hashed files and the evidence commands of D-3, D-7 and Step 8 (`probe_report_state.py`, `report-state.log`, `hashed-files.txt`, `check_d3.py`, `check-d3.log`, `check_d7_ledger.py`, `check-d7-ledger.log`, `report_statuses.py`, `report-statuses.log`);
- the open ledger rows and their mapping to later ledger lines (`open-rows.md`, `map_rows.py`, `map-rows-edited.log`; moved from the ledger at `a221d97` to the ones at `99c5ffb` and `1c15419` by `rebase_rows.py`, `rebase-rows.log`);
- after Task 4.5's merge, the same counts, report state, D-3 and D-7 checks and documentation tests on a replica of `main` at `99c5ffb` (`repo-phase5-main`, from `git archive 99c5ffb`; `main-collect-*.log`, `main-report-statuses.log`, `main-check-d3.log`, `main-check-d7-ledger.log`, `main-doc-tests.log`);
- the document edits and their mirroring into the master-plan template (`make_doc_edits_p5.py`, `doc-edits.md`, `verify_mirroring_p5.py`, `mirroring-check.log`).

- the release-tier projection's inputs, quoted from their logs with the arithmetic recomputed, by a standard-library extractor (`extract_release_evidence.py`, `release-estimate-evidence.log`).

All paths under `stages/P5/` are in `.scratch/plan-check/`. Every "Expected" line that quotes an output is one of these runs; where a count differs from the run (+ 1 for `docs/agents/final-review.md`, + N_pc, N_n or N_r for the fix wave), the step gives the arithmetic.

**Load.** One process at a time, one pytest worker, each under the watchdog; the largest process tree was 415 MB (one release test, 30 s). No tier ran, no `scripts/resource_report.py` in any form, nothing compiled above the H2 test presets, nothing FeMoco-shaped (the release test `test_femoco_cache` was not run). No load-limit event occurred.

---

## Changes from the master plan, the spec, HANDOFF and the ledger

The changes that touch a document are applied by the exact edits listed after this list (ids P5-M for the master plan, P5-S for the spec, P5-H for HANDOFF, P5-L for the ledger).

1. **The review package is split by area** (P5-M3, P5-M4, P5-M6). The master plan's input is one package, `review-package <first project commit> main`. For the tree of `c443c51` it is 10,439,159 bytes (275,854 lines, 322 commits), too large for one reader, and 88% of its lines (241,927) are not production text: the spikes and their results (140,740 lines added), the plan files (50,631) and the generated `docs/resources.json` (50,556). The four area packages hold every other file except the vendored data, the reference fixtures and the ledger, at 397,152 to 488,607 bytes each, below what reviewers already read for single tasks (Q1b.2's package: 1,167,904 bytes). Opus and GPT-5.6 review each area in parallel, one pair at a time (so at most two pytest sessions run), and the orchestrator keeps one consolidated list. Section "The review package and its split".
2. **The reviews may compile at small sizes** (P5-M5). The master plan says the reviews "compile nothing". Here they may run focused tests that compile at `H2_STO3G_TEST`, `H2_631G_TEST`, `FAILS_C3`, the toy walk and table-free operators, one worker, under the watchdog, as every task review did (each such compile takes seconds and at most a few hundred MB); nothing at `MID_NAPHTHALENE`, a vendored system or FeMoco-54, and never `scripts/resource_report.py`. The FeMoco-54 and large-system counts come from `docs/resources.json`, as the master plan says.
3. **`docs/agents/final-review.md` adds one per-commit test** (P5-M5, P5-M7, P5-L2, P5-L3, P5-H1). `tests/test_docstrings.py` collects one link check per Markdown file under `docs/` outside `superpowers/`, so the committed findings list makes the per-commit tier 1,272 (staged copy: `1272/1393 tests collected (121 deselected)`; `tests/test_docstrings.py` `49 passed`). The file holds no Markdown link, so its check passes by construction.
4. **One status value more in the ledger** (P5-L1). Task 5.1 has no branch and no merge; its row ends as `done`, with the commit of `docs/agents/final-review.md` on `main`.
5. **The user is asked once in Phase 5, after the fix wave's re-review** (P5-M7 to P5-M13, P5-S1, P5-S2). The master plan asks for the release go-ahead in Task 5.2 and had the Q-1 decision at G4. Q-1 variant (b) was adopted as the working choice under the user's standing directive and built in Task Q1b, but the user has not confirmed it explicitly. Both answers block the same steps: the FeMoco-54 recompile, if a fix changed a hashed file, and the release run, whose results rest on variant (b). So both are asked together, right before the first of those steps, and not earlier (the user's directive: "Delay input from my side if it is easily doable"). The G4 row (P5-M11) records variant (b) as the working choice, which is the "Q-1 decision" that sections 4 and 5 and Tasks Q1b and 4.5 depend on, and defers the explicit confirmation to Task 5.2; the stale "the user may override before Task Q1b runs" goes from section 2 (P5-M13), Task Q1b (P5-M12), section 13 (P5-M9) and spec 16 (P5-S1); master plan 12.9 rule 7 and spec 15 ("asked at gates only") gain the Task 5.2 sentence (P5-M10, P5-S2). So no rule is left contradictory. Two places stay as they are: master plan 12.6's introduction ("asked only about escalations, the Q-1 decision and the go-ahead for the release run") and spec 16's "Q-1 has a working choice, variant (b), which the user may override", because neither names a time and both agree with the confirmation in Task 5.2; the ledger's decision rows are records and are not rewritten.
6. **At most one report-invalidating hashed fix wave before the release run** (P5-M7). Master plan 12.9 keeps one contingency pair of FeMoco-54 compiles for a Phase 5 recompile. A fix of a hashed file is made only for a Critical or Important finding; hashed Minor findings ride along only with one of them, as the ledger rows of Q1b.2's m-2 and of Task 4.5's summary wording already wait for "the next hashed change". Any number of hashed Critical or Important fixes may land in that wave; the report and the FeMoco pair are regenerated once, at the final reviewed hash. A later hashed fix after that pair is a second wave and requires the release-failure escalation and any budget approval it triggers. Section "Hashed and unhashed fixes".
7. **A D-1 to D-7 checklist in the ledger** (P5-L4): a new section with one row per criterion, filled in Task 5.2, Step 11. D-7 is worded as what the local evidence can show: every Phase 5 prompt forbade push, fetch and remote changes, and the local remote configuration and remote-tracking refs are unchanged between Task 5.1, Step 1 and the merge. Without contacting the remote, that is a process-control assertion, not a forensic proof that no push occurred.
8. **This file joins the master plan** (P5-M1, P5-M2).

Exact edits (master plan, spec, HANDOFF and ledger, against local `main`):

Each edit replaces one exact text of the file on local `main` at `1c15419`, where it occurs exactly once (checked by `.scratch/plan-check/stages/P5/make_doc_edits_p5.py`, which also applies them in order). Every master-plan old text also occurs exactly once in `.scratch/plan-check/templates/2026-11-11-dfthc-master-plan.md` and never in the shared fragment `templates/_global_constraints_q1b.md` that it includes, so the orchestrator mirrors P5-M1 to P5-M13 into the template and no fragment changes (section "Mirroring at approval").

**P5-M1** (2026-11-11-dfthc-master-plan.md, section 1, the file table (a new row after the Q1b file's)). Old:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase4-q1b.md` | Task Q1b (sub-tasks Q1b.1 and Q1b.2), written at the Q-1 working choice |
````

New:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase4-q1b.md` | Task Q1b (sub-tasks Q1b.1 and Q1b.2), written at the Q-1 working choice |
| `docs/superpowers/plans/2026-11-11-dfthc-phase5-final.md` | Tasks 5.1 and 5.2: the final review in four area packages, the fix wave, the release tier, the D-1 to D-7 checklist and the final merge |
````

**P5-M2** (2026-11-11-dfthc-master-plan.md, section 1, code provenance (the end of the paragraph)). Old:

````text
The code-hash change makes the orchestrator rerun `fe2s2` and `fe4s4`; those reruns and Task 4.5's FeMoco-54 compiles are projected there.
````

New:

````text
The code-hash change makes the orchestrator rerun `fe2s2` and `fe4s4`; those reruns and Task 4.5's FeMoco-54 compiles are projected there. The Phase 5 file (`phase5-final`, Tasks 5.1 and 5.2) adds no code before the final review; its commands were checked on 2026-11-15 against the tree of `c443c51` (Task 4.5's branch) and, after Task 4.5's merge, of `99c5ffb`: the collection counts of every tier, the documentation tests, three single release tests, the size of the review package and the evidence commands of D-3 and D-7; its preamble lists its changes to this plan.
````

**P5-M3** (2026-11-11-dfthc-master-plan.md, Task 5.1, tier and reviewers). Old:

````text
- **Tier:** A. Reviewers are Opus (`inherit`, the final whole-branch review per superpowers requesting-code-review) and GPT-5.6, in parallel. **Depends on:** G4. **Branch:** none (read-only).
````

New:

````text
- **Tier:** A. Reviewers are Opus (`inherit`, the final whole-branch review per superpowers requesting-code-review) and GPT-5.6, in parallel on each of four area packages, one pair at a time. **Depends on:** G4. **Branch:** none (read-only).
````

**P5-M4** (2026-11-11-dfthc-master-plan.md, Task 5.1, inputs). Old:

````text
- **Inputs:** the review package `review-package <first project commit> main`; the ledger's Minor-findings list; spec D-1 to D-7.
````

New:

````text
- **Inputs:** the review package of `993b8fb..main` (`993b8fb` is the first project commit). For the tree of `c443c51` it is 10.4 MB, too large for one reader, so it is split into four area packages of 0.40 to 0.49 MB (operators; operator tests; data, report and tooling; documents), which leave out the spikes, the plan files, the vendored data, the reference fixtures, the generated `docs/resources.json` and the ledger (plan `phase5-final`); the ledger's Minor-findings list with the plan's proposed disposition of each open row; spec D-1 to D-7.
````

**P5-M5** (2026-11-11-dfthc-master-plan.md, Task 5.1, output). Old:

````text
- **Output:** one consolidated findings list (Critical, Important, Minor) in `docs/agents/final-review.md`. The reviews read the counts cached in `docs/resources.json` and compile nothing.
````

New:

````text
- **Output:** one consolidated findings list (Critical, Important, Minor; each with where, evidence, fix and whether the fix touches a hashed file) in `docs/agents/final-review.md`, merged by the orchestrator from the eight area reviews and committed on `main`; its link check adds one per-commit test. The reviews read the counts cached in `docs/resources.json` and compile nothing beyond the H2 test presets, `FAILS_C3`, the toy walk and table-free operators (one pytest worker, under the watchdog).
````

**P5-M6** (2026-11-11-dfthc-master-plan.md, Task 5.1, definition of done). Old:

````text
- **Definition of done:** both reviews are in and merged into one list.
````

New:

````text
- **Definition of done:** all eight area reviews are in and merged into one list.
````

**P5-M7** (2026-11-11-dfthc-master-plan.md, Task 5.2, a bullet before the definition of done). Old:

````text
- **Definition of done:** release tier green; every D criterion checked with evidence; merged into local `main`; nothing pushed.
````

New:

````text
- **Counts and the user's questions** (plan `phase5-final`): per commit 1,272 + N_pc (1,271 measured on the trees of `c443c51` and `99c5ffb`, + 1 for the link check of `docs/agents/final-review.md`), nightly 86 + N_n, release-only 35 + N_r, so `scripts/test.sh` runs 1,393 + N_pc + N_n + N_r, where N_pc, N_n and N_r are the per-commit, nightly and release-only tests the fix wave adds. After the fix wave's re-review the orchestrator asks the user once: the confirmation of Q-1 variant (b) (the working choice of 2026-11-15, built in Task Q1b) and the go-ahead for the release tier with its projection (about 25 to 35 minutes on a quiet machine, up to about 80 under load; peak about 2 to 3 GB). A fix that changes a hashed file is made only for a Critical or Important finding, and hashed Minor findings ride along only with one of them, so Phase 5 has at most one report-invalidating hashed fix wave before the release run. Any number of hashed Critical or Important fixes may land in that wave; the report and the FeMoco-54 pair are regenerated once, at the final reviewed hash. A later hashed fix after that pair is a second wave and requires the release-failure escalation and any budget approval it triggers.
- **Definition of done:** release tier green; every D criterion checked with evidence; merged into local `main`; nothing pushed.
````

**P5-M8** (2026-11-11-dfthc-master-plan.md, section 12.6, the G5 row). Old:

````text
| **G5** (Phase 5) | D-1 to D-7 met with evidence; release tier green (run once, after the user's go-ahead); ledger complete. | Unresolved Critical or Important findings block the final merge. |
````

New:

````text
| **G5** (Phase 5) | D-1 to D-7 met with evidence; release tier green (run once, after the user's go-ahead, asked together with the user's confirmation of Q-1 variant (b)); ledger complete. | Unresolved Critical or Important findings block the final merge. If the user overrides Q-1, Phase 5 stops and the orchestrator escalates with a plan to restore the baseline: it needs a new task, report regeneration and a FeMoco-54 pair. Because the current contingency pair has not run when this answer is taken, that pair would be compiles 4 and 5 of about 6; the new plan must account for it, and any later compile beyond about six needs the user's approval. |
````

**P5-M9** (2026-11-11-dfthc-master-plan.md, section 13, the Q-1 row (its first two cells)). Old:

````text
| Q-1 early junk measurement | K-10 verdict (Task 0.8) at G0, then the working choice of variant (b) on 2026-11-15 (the user's stated preference; the user may override before Task Q1b runs) |
````

New:

````text
| Q-1 early junk measurement | K-10 verdict (Task 0.8) at G0, then the working choice of variant (b) on 2026-11-15 (the user's stated preference, adopted under the standing directive); Task Q1b merged; the user's explicit confirmation is asked in Task 5.2, with the release go-ahead |
````

**P5-M10** (2026-11-11-dfthc-master-plan.md, section 12.9, rule 7 (its first sentence)). Old:

````text
7. **Questions to the user.** Asked at gates only: the G0 escalation if K-1 projects above the limits (where compiles above the cap run), the Q-1 decision, the go-ahead for the release run, and any compile beyond the six.
````

New:

````text
7. **Questions to the user.** Asked at gates only: the G0 escalation if K-1 projects above the limits (where compiles above the cap run), the Q-1 decision, the go-ahead for the release run, and any compile beyond the six. The release go-ahead and the confirmation of the Q-1 working choice are asked together in Task 5.2, after the fix wave's re-review and before the first step that depends on them (the FeMoco-54 recompile, if a fix changed a hashed file, or the release run).
````

**P5-M11** (2026-11-11-dfthc-master-plan.md, section 12.6, the G4 row). Old:

````text
| **G4** (Phase 4) | Tasks 4.1 to 4.5 merged; resource report committed with the FeMoco totals of Task 4.5. **Q-1 decision** (between 4.4 and 4.5): present the report and the K-10 result; the user chooses variant (b) or the baseline. If variant (b): Task Q1b, including the regenerated report and updated documentation, is merged before Task 4.5. Then D-3, D-4 and D-6 are checked on the final artifacts, and the Phase 5 plan is approved. | The Q-1 decision is the one planned user question of this gate. If the G0 decision left the FeMoco compile to another machine or a later window, G4 waits for it. |
````

New:

````text
| **G4** (Phase 4) | Tasks 4.1 to 4.5 merged; resource report committed with the FeMoco totals of Task 4.5; variant (b) is the working Q-1 choice under the user's standing directive (the Q-1 decision that sections 4 and 5 and Tasks Q1b and 4.5 refer to; Task Q1b, with the regenerated report and updated documentation, was merged before Task 4.5), with explicit confirmation deferred to Task 5.2 together with the release go-ahead. D-3, D-4 and D-6 are checked on the final artifacts, and the Phase 5 plan is approved. | If the G0 decision left the FeMoco compile to another machine or a later window, G4 waits for it. |
````

**P5-M12** (2026-11-11-dfthc-master-plan.md, Task Q1b, "Opened" (its first sentence after the opening point)). Old:

````text
only if the user chooses variant (b) and K-10 passed; the working choice of 2026-11-15 is variant (b) (section 2), and the user may override before Q1b runs.
````

New:

````text
only if variant (b) is chosen and K-10 passed; the working choice of 2026-11-15, variant (b) (section 2), adopted under the user's standing directive, opened it; Q1b is merged, and the user's explicit confirmation is asked in Task 5.2, with the release go-ahead.
````

**P5-M13** (2026-11-11-dfthc-master-plan.md, section 2, the Q-1 decision row (the end of its second cell)). Old:

````text
the orchestrator adopted it under the standing directive, and the user may override before Task Q1b runs |
````

New:

````text
the orchestrator adopted it under the standing directive; Task Q1b and Task 4.5 are merged, and Task 5.2 asks the user for explicit confirmation together with the release go-ahead |
````

**P5-S1** (2026-11-11-dfthc-pennylane-design.md, 16, Q-1 status (the end of its first sentence)). Old:

````text
so the orchestrator adopted it under the user's standing directive; the user may override before Task Q1b runs.
````

New:

````text
so the orchestrator adopted it under the user's standing directive. Task Q1b and Task 4.5 are merged; Task 5.2 asks the user for explicit confirmation together with the release go-ahead.
````

**P5-S2** (2026-11-11-dfthc-pennylane-design.md, 15, load policy, questions to the user). Old:

````text
- *Questions to the user* are asked at gates only: the K-1 escalation (where compiles above the cap run), the Q-1 decision, the go-ahead for the release run, and any compile beyond the six.
````

New:

````text
- *Questions to the user* are asked at gates only: the K-1 escalation (where compiles above the cap run), the Q-1 decision, the go-ahead for the release run, and any compile beyond the six. The release go-ahead and the confirmation of the Q-1 working choice are asked together in Phase 5, right before the release run (or the FeMoco-54 recompile a code fix may need).
````

**P5-H1** (HANDOFF.md, section 2, a row after the ledger's). Old:

````text
| Ledger (status, spikes, decisions, gates) | `docs/agents/LEDGER.md` |
````

New:

````text
| Ledger (status, spikes, decisions, gates) | `docs/agents/LEDGER.md` |
| Final review findings (Task 5.1; the fix wave's list) | `docs/agents/final-review.md` |
````

**P5-L1** (LEDGER.md, task status, the status values). Old:

````text
- **Status values:** `not started`, `in progress`, `in review`, `fixes`, `merged`, `blocked`,
  `cancelled`, `not needed`.
````

New:

````text
- **Status values:** `not started`, `in progress`, `in review`, `fixes`, `merged`, `blocked`,
  `cancelled`, `not needed`, and `done` for Task 5.1 (read-only: its output, `docs/agents/final-review.md`,
  is committed on `main`).
````

**P5-L2** (LEDGER.md, Phase 5, the row of Task 5.1). Old:

````text
| 5.1 | Whole-branch final review | A | not started | none | | | / | read-only: no branch, no merge |
````

New:

````text
| 5.1 | Whole-branch final review | A | not started | none | | | / | read-only: no branch, no merge; plan `phase5-final`: four area packages of `993b8fb..main`, Opus and GPT-5.6 on each (one pair at a time); output `docs/agents/final-review.md`, committed on `main` (+ 1 per commit: its link check) |
````

**P5-L3** (LEDGER.md, Phase 5, the row of Task 5.2). Old:

````text
| 5.2 | Fix wave, release tier, final merge | A | not started | `phase5/5.2` | | | / | the release tier runs once, after the user's go-ahead |
````

New:

````text
| 5.2 | Fix wave, release tier, final merge | A | not started | `phase5/5.2` | | | / | the release tier runs once, after the user's go-ahead; plan `phase5-final`: per commit 1,272 + N_pc, nightly 86 + N_n, release-only 35 + N_r, release tier 1,393 + N_pc + N_n + N_r (the tests the fix wave adds to each tier); the user is asked once, after the re-review (the Q-1 confirmation and the release go-ahead); a fix of a hashed file makes the orchestrator regenerate the small-system report and recompile the FeMoco-54 pair (the contingency pair), with `fe2s2` and `fe4s4` pending |
````

**P5-L4** (LEDGER.md, a new section before "Canary state changes" (the D-1 to D-7 checklist)). Old:

````text
## Canary state changes
````

New:

````text
## Definition of done (spec 2.3; Task 5.2)

Filled in Task 5.2 (plan `phase5-final`, Step 11), each criterion with its evidence command and key output.

| D | Criterion | Evidence (command and key output) | Checked |
| --- | --- | --- | --- |
| D-1 | every spec 9 operator documented per spec 9.0; T2 to T6 green | | not checked |
| D-2 | the spec 12.5 certificates; E-1 to E-4 under the release schedules | | not checked |
| D-3 | FeMoco-54 walk and controlled walk compiled once, counts equal to the closed forms, cached with the code hash, every delta explained | | not checked |
| D-4 | QPE: the toy walk's exact phases; compiled and resource-counted for a DFTHC instance | | not checked |
| D-5 | the canary suite runs and records every limitation | | not checked |
| D-6 | the documentation set exists and matches the code | | not checked |
| D-7 | merged into local `main` after review gates; every Phase 5 prompt forbade push, fetch and remote changes; the local remote configuration and remote-tracking refs are unchanged (a process-control assertion: without contacting the remote it is no forensic proof that no push occurred) | | not checked |

## Canary state changes
````

## Mirroring at approval (the orchestrator)

The plan writer edits no existing template. At approval the orchestrator, from `.scratch/plan-check/`:

1. Runs `/Users/david.wierichs/repos/dfthc-be/.venv/bin/python stages/P5/make_doc_edits_p5.py`. It checks that each old text occurs exactly once in the four files on the current `main` and once in the master-plan template (never in the shared fragment), applies the 20 edits in order to copies in `tmp/doc-edits-p5/`, and prints `20 edits (13 master plan, 2 spec, 1 HANDOFF, 4 ledger) checked against main (<main>) and applied in tmp/doc-edits-p5; listing stages/P5/doc-edits.md`. An old text that no longer occurs once stops the approval (the orchestrator then rebases the edit on the changed text and reruns). The orchestrator copies the four files from `tmp/doc-edits-p5/` over `docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md`, `docs/agents/HANDOFF.md` and `docs/agents/LEDGER.md`.
2. Applies P5-M1 to P5-M13 to `templates/2026-11-11-dfthc-master-plan.md`, each old text exactly once there. No Global Constraints fragment changes: this plan includes `templates/_global_constraints_q1b.md` unchanged, because nothing in Phase 5 changes a global rule (its release-tier, load and git rules are the ones Phase 5 follows).
3. Assembles (`/Users/david.wierichs/repos/dfthc-be/.venv/bin/python assemble.py <out>/`) and checks that the assembled master plan equals the edited master plan in `docs/` byte for byte, and that every other assembled plan equals its file under `docs/superpowers/plans/`. This plan is committed as assembled: if step 1 ran against a `main` newer than `1c15419`, the edit listing above names that `main`, and the replay (`stages/P5/verify_mirroring_p5.py`) is rerun first.
4. Commits the plan and the four documents on `main` and runs `scripts/test.sh tests/test_docstrings.py -q -p no:cacheprovider` there (the ledger and HANDOFF keep valid links; no edit adds one). Expected: `48 passed`.

Replayed on copies of the templates (`stages/P5/verify_mirroring_p5.py`, which applies step 2 in `tmp/p5-mirror/`, assembles there and compares):

```text
applied to the master-plan template: P5-M1, P5-M2, P5-M3, P5-M4, P5-M5, P5-M6, P5-M7, P5-M8, P5-M9, P5-M10, P5-M11, P5-M12, P5-M13
identical: 2026-11-11-dfthc-master-plan.md (against main (1c15419) + P5-M1 to P5-M13)
identical: 2026-11-11-dfthc-phase0-canaries.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase0-infrastructure.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase0-spikes.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase1-data.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase1-primitives.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase1-simulator.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase2-blocks.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase2-prepare.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase3-assembly.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase3-select.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase4-q1b.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase4-qpe.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase4-report.md (against main (1c15419))
identical: 2026-11-11-dfthc-phase5-final.md (against out-p5)
```

Without step 2, every existing template assembles byte-identically to its plan on `main` at `1c15419`: 14 of 14 (`out-p5/`, `stages/P5/check_out_identity.py`, `out-p5-identity.log`), the report plan's revision 9 (`99c5ffb`) included, which the orchestrator mirrored into its template while this plan was written. The replay script reports a plan that differs from `main` but equals its `a221d97` version as `PRE-EXISTING` (a change on `main` not yet in its template, not this plan's); there is none now.

## Measured bases and expected counts

Collected on the replica of `c443c51` with one worker (`DFTHC_TEST_WORKERS=1 .scratch/plan-check/p5test.sh <selection> --collect-only -q`), and again on the replica of `99c5ffb` (Task 4.5 merged), which gives the same lines and the same 1,392 test ids (`main-collect-*.log`); in a lane or the main checkout the same selection runs as `scripts/test.sh <selection> --collect-only -q -p no:cacheprovider`.

| Tier | Selection | On the tree of `c443c51` | With `docs/agents/final-review.md` (staged copy) |
| --- | --- | --- | --- |
| Per commit | `-m "not slow and not release"` | `1271/1392 tests collected (121 deselected)` | `1272/1393 tests collected (121 deselected)` |
| Nightly | `-m "slow and not release"` | `86/1392 tests collected (1306 deselected)` | 86 (no nightly test reads `docs/`) |
| Release only | `-m release` | `35/1392 tests collected (1357 deselected)` | 35 |
| Gate run | `-m "not release"` | `1357/1392 tests collected (35 deselected)` | 1,358 |
| Release tier, `scripts/test.sh` | none | `1392 tests collected` | `1393 tests collected` |
| FeMoco cache | `-m femoco` | `1/1392 tests collected (1391 deselected)` | 1 |
| Documentation | `tests/test_docstrings.py tests/test_tolerance_hygiene.py` | `50 passed` (48 + 2) | `tests/test_docstrings.py` alone: `49 passed` |

1,271 + 86 + 35 = 1,392 and 1,272 + 86 + 35 = 1,393. With N_pc, N_n and N_r the per-commit, nightly and release-only tests the fix wave adds (from the fixer's collection counts, checked by the orchestrator):
- after Task 5.1, on `main`: per commit 1,271 + 1 = 1,272; nightly 86; release-only 35; release tier 1,392 + 1 = 1,393;
- after the fix wave, on `phase5/5.2`: per commit 1,272 + N_pc; nightly 86 + N_n; release-only 35 + N_r; release tier 1,393 + N_pc + N_n + N_r.

Below, N alone always means N_pc + N_n + N_r, the fix wave's tests over all tiers, as in "1,393 + N"; a single tier always names its own delta.

Per-file counts the D-1 to D-7 evidence uses (from `stages/P5/collect-all.log`; the fix wave adds its own tests to them): `tests/test_docstrings.py` 48 (34 docstring audits, 10 link checks; 49 with `final-review.md`), `tests/ops/test_certificates_*.py` 30 (all nightly: prepare 16, QROAM 7, angles 7), `tests/integration/test_end_to_end.py` 42 (E-1 33, E-2 4, E-3 3, E-4 2, from the file's parametrizations: E-1 five schedules plus the 28 two-electron 6-31G inputs, E-2 two shapes times k = 1, 2, E-3 three schedules, E-4 two schedules; 10 nightly, 32 release), `tests/integration/test_qpe_toy.py` 23, `tests/integration/test_qpe_counts.py` 8, `tests/integration/test_qpe_walk.py` 6, `tests/canary/` 41 (40 per commit, 1 nightly), `tests/report/test_femoco_cache.py` 2.

## The review package and its split

`review-package 993b8fb c443c51 <file>` (the superpowers script; `993b8fb` "Add .gitignore" is the first project commit) wrote `10439159 bytes` for 322 commits. By path (lines added): `spikes/` 140,740; `docs/superpowers/plans/` 50,631; `docs/resources.json` 50,556; `tests/ops/` 6,896; `src/dfthc/ops/` 5,901; the rest of `src/dfthc/` 4,683; the spec 1,681; everything else below 1,600 per directory. The split keeps every production and test file and every document a reviewer must judge, in four areas; it leaves out what is generated, vendored or process record:

| Area | Paths | Package on `c443c51` |
| --- | --- | --- |
| 1 operators | `src/dfthc/ops/`, `src/dfthc/qpe.py`, `src/dfthc/registers.py`, `src/dfthc/params.py`, `src/dfthc/costs.py`, `src/dfthc/compile.py`, `src/dfthc/__init__.py` | 397,152 bytes, 7,547 lines |
| 2 operator tests | `tests/ops/`, `tests/integration/`, `tests/test_qpe.py`, `tests/test_work_pool.py`, `tests/test_registers.py`, `tests/test_costs.py`, `tests/test_compile.py`, `tests/test_primitive_capabilities.py`, `tests/test_primitive_costs.py` | 482,051 bytes, 9,853 lines |
| 3 data, report and tooling | `src/dfthc/data/`, `src/dfthc/costs_reference.py`, `src/dfthc/testing/`, `scripts/`, `tests/data/`, `tests/sim/`, `tests/report/`, `tests/canary/`, `tests/test_watchdog.py`, `tests/test_infrastructure.py`, `tests/test_tolerance_hygiene.py`, `tests/test_docstrings.py`, `tests/test_canary_helper.py`, `tests/conftest.py`, `pyproject.toml`, `requirements-nightly.lock`, `.gitignore` | 488,607 bytes, 10,466 lines |
| 4 documents | `docs/architecture.md`, `docs/conventions.md`, `docs/environment.md`, `docs/limitations.md`, `docs/verification.md`, `docs/resources.md`, `docs/upstream/`, `docs/agents/HANDOFF.md`, `docs/superpowers/specs/` | 419,089 bytes, 4,047 lines |
| left out | `spikes/` (stand-ins and results, ledger "Spike results"), `docs/superpowers/plans/` (the requirements, read as such), `docs/resources.json` (generated; area 3 reads it directly), `docs/agents/LEDGER.md` (the record; its open rows are the plan's table below), `data/dfthc/` (vendored, digests checked by `tests/data/test_vendored.py`), `tests/fixtures/dfthcbe/` (generated in `.venv-ref`; provenance in its `provenance.json`) | 86 files |

The script below writes the four packages and the list of the left-out files. It is the review-package layout (commits, files changed, the diff with ten lines of context) restricted to each area's paths; it runs from the main checkout with bash 3.2 (macOS):

```bash
#!/usr/bin/env bash
# Dry run of the Phase 5 plan's Task 5.1 Step 2 (the area review packages); bash 3.2 compatible.
# Usage, from the main checkout: review_areas.sh BASE HEAD OUTDIR. Writes OUTDIR/area-<name>.diff
# (the review-package layout restricted to the area's paths) and OUTDIR/excluded.txt (the files no
# area holds); prints one size line per file.
set -euo pipefail
base=$1 head=$2 out=$3
mkdir -p "$out"
AREAS="1-operators 2-operator-tests 3-data-report-tooling 4-documents"
area_paths() {
  case $1 in
    1-operators) echo "src/dfthc/ops src/dfthc/qpe.py src/dfthc/registers.py src/dfthc/params.py src/dfthc/costs.py src/dfthc/compile.py src/dfthc/__init__.py" ;;
    2-operator-tests) echo "tests/ops tests/integration tests/test_qpe.py tests/test_work_pool.py tests/test_registers.py tests/test_costs.py tests/test_compile.py tests/test_primitive_capabilities.py tests/test_primitive_costs.py" ;;
    3-data-report-tooling) echo "src/dfthc/data src/dfthc/costs_reference.py src/dfthc/testing scripts tests/data tests/sim tests/report tests/canary tests/test_watchdog.py tests/test_infrastructure.py tests/test_tolerance_hygiene.py tests/test_docstrings.py tests/test_canary_helper.py tests/conftest.py pyproject.toml requirements-nightly.lock .gitignore" ;;
    4-documents) echo "docs/architecture.md docs/conventions.md docs/environment.md docs/limitations.md docs/verification.md docs/resources.md docs/upstream docs/agents/HANDOFF.md docs/superpowers/specs" ;;
  esac
}
excludes=""
for name in $AREAS; do
  paths=$(area_paths "$name")
  file="$out/area-$name.diff"
  {
    echo "# Review package (area $name): ${base}..${head}"
    echo
    echo "## Commits touching the area"
    git log --oneline "${base}..${head}" -- $paths
    echo
    echo "## Files changed"
    git diff --stat=200 "${base}..${head}" -- $paths
    echo
    echo "## Diff"
    git diff -U10 "${base}..${head}" -- $paths
  } > "$file"
  echo "$file: $(wc -c < "$file" | tr -d ' ') bytes, $(wc -l < "$file" | tr -d ' ') lines"
  for p in $paths; do excludes="$excludes :(exclude)$p"; done
done
git diff --stat=200 "${base}..${head}" -- . $excludes > "$out/excluded.txt"
echo "$out/excluded.txt: $(tail -1 "$out/excluded.txt")"
```

Its output on `c443c51` (`bash .scratch/plan-check/stages/P5/review_areas.sh 993b8fb c443c51 .scratch/plan-check/stages/P5/areas`):

```text
.scratch/plan-check/stages/P5/areas/area-1-operators.diff: 397152 bytes, 7547 lines
.scratch/plan-check/stages/P5/areas/area-2-operator-tests.diff: 482051 bytes, 9853 lines
.scratch/plan-check/stages/P5/areas/area-3-data-report-tooling.diff: 488607 bytes, 10466 lines
.scratch/plan-check/stages/P5/areas/area-4-documents.diff: 419089 bytes, 4047 lines
.scratch/plan-check/stages/P5/areas/excluded.txt:  86 files changed, 243260 insertions(+)
```

Why four areas and not two: an operator package with its tests would be about 880 KB, more than one reader holds next to the spec; the areas also match the reviewers' focus (operators against spec 9; tests against spec 12; the report, data and tooling against spec 8, 13 and 15; documents against D-6). Cross-area risks are covered three ways: a reviewer may open files outside its area for a risk it names; area 4 reads the spec whole; and the orchestrator's merge looks for findings that touch two areas.

## Hashed and unhashed fixes

`code_hash()` (`src/dfthc/testing/provenance.py`) is the SHA-256 over the docstring-free syntax trees of 30 files, `src/dfthc/**/*.py` outside `dfthc/testing/` and `scripts/resource_report.py` (`stages/P5/hashed-files.txt`), and over the bytes of `requirements-nightly.lock`. On the tree of `c443c51` it is `5e3eed1e7a9535765299d6c90afe643f96a5155e7eb43d43e039864133fbba8e`, the hash of every current entry of `docs/resources.json`: 272 current entries (the four small systems 182, `fe2s2` 44, `fe4s4` 44, FeMoco-54's walk and controlled walk 2) and 282 entries of hash `2e5482adb928...` kept as history (`stages/P5/report-state.log`).

- **Unhashed fix:** tests, `src/dfthc/testing/`, documents, the spec, and docstrings or comments anywhere. The hash stays, the committed report stays current, nothing is rerun, and the release tier's `test_femoco_cache` reads the cached FeMoco-54 entries.
- **Hashed fix:** any other change of a hashed file, a string literal of `scripts/resource_report.py` included. Every cached entry becomes history. The orchestrator then (Task 5.2, Step 8, case H): regenerates the small-system report (182 programs, about 11 to 14 minutes below 0.8 GB, not heavy); recompiles the FeMoco-54 walk and controlled walk once with `--large femoco54`, alone, the budget's contingency pair (FeMoco-scale compiles 4 and 5 of about 6; 3 are used: K-1's controlled walk and Task 4.5's pair); does not recompile `fe2s2` and `fe4s4`, whose 88 rows keep their measurements as history and show their closed forms, marked pending (ledger decision "Stale large-system rows in Phase 5"); and commits the report on the branch.

**Which findings are fixed:**
- Critical and Important: always, hashed or not.
- Minor, unhashed: fixed, unless the orchestrator records a reason to decline (the fix would add more than 10 s to the per-commit tier, needs a heavy run, or conflicts with the spec or a plan).
- Minor, hashed: fixed only if the fix list holds at least one hashed Critical or Important finding; otherwise deferred, recorded as "deferred: hashed Minor without a hashed Critical or Important fix". So Phase 5 has at most one report-invalidating hashed fix wave before the release run. Any number of hashed Critical or Important fixes may land in that wave; the report and the FeMoco-54 pair are regenerated once, at the final reviewed hash. A later hashed fix after that pair is a second wave and requires the release-failure escalation and any budget approval it triggers (Task 5.2, Step 10).
- Rows marked optional in the ledger (L246, L247, L251, L258, L266, L338): declined unless a reviewer raises one to Important with evidence; the hashed ones (L251, L258, L266, L338) may also ride along with a hashed Critical or Important fix, like a hashed Minor, at the orchestrator's choice in Task 5.1, Step 8 (the ledger already says so for L266: "optional with the next hashed change").
- Never: a change that moves T, M or A of any operator at any preset (it would contradict spec 9.24 and the cached FeMoco-54 counts: escalate to the user), or a change toward the paper's or dfthcbe's counts (R-7 and the user's directive: resource matching is a separate later step; such a finding is declined and recorded).

Why: a hashed fix wave costs about 12 minutes of small-system compiles and two of the about six FeMoco-scale compiles (4 and 5), and turns 88 measured `fe2s2` and `fe4s4` rows into pending closed forms; a second wave would need compiles 6 and 7, past about six, which needs the user's approval (master plan 12.9 rule 7).

## Open ledger rows and their dispositions

The ledger's "Minor findings (input to Task 5.1)" on `main` at `1c15419` has 72 rows whose status is open or accepted. At `a221d97` there were 75; Task 4.5's merge (`99c5ffb`) closed Q1b.2's notes for Task 4.5 and opened one row of its own (L289); the orchestrator's G4 checks (`1c15419`) closed three more with the closures this table had proposed (0.9d, 4.3 and 4.2: Task 4.5's measured compiles and report) and restated L266 as "open: Task 5.1, optional with the next hashed change" (`stages/P5/rebase-rows.log`). Ten of them name Task 5.1 (L266, L278, L283, L289, L293, L302, L314, L332, L338, L339). Every row is an input of Task 5.1 and gets one disposition:
- **R:** the area's reviewers rate it at the code (Critical, Important, Minor, or decline, with evidence); the orchestrator's merge applies section "Hashed and unhashed fixes"; Task 5.2 closes it as fixed (with the commit) or declined or deferred (with the reason).
- **C:** closed by merged work; the evidence is in the table; the reviewers confirm or object with evidence; Task 5.2 closes it with that evidence.
- **D:** a refinement that changes the Toffoli count, deferred by the user's directive (resource matching is a separate later step) and listed in `docs/resources.md` under "Deferred refinements"; closed as deferred.
- **A:** accepted earlier; stays accepted unless a reviewer shows it is now wrong.
- **W:** closed at G5 if the test passed in every Phase 5 run; a failure makes it R.
- **P:** a standing procedure; closed at G5, after Phase 5 has followed it.
- **T:** closed by the release tier's run.

Counts: R 23 (10 hashed, 13 unhashed; 6 of them optional), C 41, D 3, A 2, W 1, P 1, T 1. Line numbers are those of `main` at `1c15419`; later ledger commits shift them (P5-L4 alone by 15 lines), so Task 5.1, Step 1 maps each to its current line by its exact text with this script, which also checks that the table names exactly the open and accepted rows at `1c15419` (on `main` at `1c15419`: `72 rows, equal to the open and accepted rows at 1c15419; dispositions {'A': 2, 'C': 41, 'D': 3, 'P': 1, 'R': 23, 'T': 1, 'W': 1}; 0 moved, 0 changed in /Users/david.wierichs/repos/dfthc-be/docs/agents/LEDGER.md`). If the orchestrator closes a row of the table before Task 5.1, the script reports it as "changed" and the merge in Step 9 records that closure:

```python
"""Map the open Minor-findings rows of the Phase 5 plan (their ledger lines on `1c15419`) to their
lines in a later ledger, by the exact text of each row.

The plan's table "Open ledger rows and their dispositions" (stages/P5/open-rows.md) names each row
by its line in docs/agents/LEDGER.md at 1c15419. The ledger's later commits insert rows above the
Minor-findings list, so the lines shift; this prints, for every row of the table, its line in the
given ledger, or "changed" if no line of that ledger equals the row's text at 1c15419. It also
checks that the table names exactly the open and accepted rows of the ledger at 1c15419.
Run from the main checkout: <venv>/bin/python <this file> [LEDGER.md] (default: the working tree's).
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TABLE = HERE / "open-rows.md"
OLD = subprocess.run(["git", "-C", str(REPO), "show", "1c15419:docs/agents/LEDGER.md"], check=True,
                     capture_output=True, text=True).stdout.split("\n")
new_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "agents" / "LEDGER.md"
NEW = new_path.read_text().split("\n")

table = {}
for line in TABLE.read_text().splitlines():
    match = re.match(r"^\| L(\d+) \| ([^|]+) \| ([^|]+) \| ([1-4]) \| ([^|]+) \| ([A-Z]) ", line)
    if match:
        table[int(match[1])] = (match[2].strip(), match[4], match[6])

start = next(i for i, line in enumerate(OLD) if line.startswith("## Minor findings"))
expected = {}
for i, line in enumerate(OLD[start:], start=start + 1):
    if not line.startswith("| ") or line.startswith("| Task ") or line.startswith("| ---"):
        continue
    cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
    if "open" in cells[-1].lower() or "accepted" in cells[-1].lower():
        expected[i] = cells[0]
assert set(table) == set(expected), (sorted(set(expected) - set(table)),
                                     sorted(set(table) - set(expected)))
for number, (task, _, _) in table.items():
    assert task == expected[number], (number, task, expected[number])

positions = {}
for i, line in enumerate(NEW, start=1):
    positions.setdefault(line, []).append(i)
moved = changed = 0
for number in sorted(table):
    task, area, disposition = table[number]
    found = positions.get(OLD[number - 1], [])
    where = f"L{found[0]}" if len(found) == 1 else "changed" if not found else f"ambiguous {found}"
    moved += where not in ("changed", f"L{number}")
    changed += where == "changed"
    print(f"L{number} ({task}, area {area}, {disposition}) -> {where}")
counts = {}
for _, area, disposition in table.values():
    counts[disposition] = counts.get(disposition, 0) + 1
print(f"{len(table)} rows, equal to the open and accepted rows at 1c15419; dispositions "
      f"{dict(sorted(counts.items()))}; {moved} moved, {changed} changed in {new_path}")
```

The table (`stages/P5/open-rows.md`; "Hashed fix" says whether a fix of the row would change the code hash, "—" where nothing is to be fixed):

| Ledger line (`1c15419`) | Task | Finding, short | Area | Hashed fix | Disposition | Proposed closure, or what the reviewers rate |
| --- | --- | --- | --- | --- | --- | --- |
| L239 | 0.2 | the device-level closure of `dfthc.compile` checks only dfthc names, narrower than spec 7.2 | 1 | yes | R | rate; a fix is hashed (`src/dfthc/compile.py`) |
| L251 | 1C.1 | `FlaggedUniformPrep` could pass `phase_grad[:s]` to its rotations (optional) | 1 | yes | R | optional: declined unless a reviewer raises it to Important (count-neutral; the FeMoco-54 compiles took 1.7 GB of 7.5) |
| L252 | 1C.1 | contracts Tasks 1C.3 to 1C.5 rely on (leaf order, flags, nested work) | 1 | — | C | Tasks 1C.3 to 1C.5 merged with tests of those contracts |
| L258 | 1C.3 | the abstract `leaf_words` check accepts unsigned dtypes (optional `np.signedinteger`) | 1 | yes | R | optional: declined unless raised (keys and arrays use `np.int64`; canary L-26 pins the failure) |
| L264 | 1C.4 | constant parity tables compiled into `PhaseFixup` are never checked for their phases | 1 | — | A | stays accepted while no caller compiles constant tables into `PhaseFixup` (reviewers confirm) |
| L266 | 1C.5 | `_capture_parities` compares once per nonzero mask (about 2% of the FeMoco compile) | 1 | yes | R | optional: rides along with a hashed Critical or Important fix if there is one (the ledger: "optional with the next hashed change"), else declined; its first condition, "if compile memory gets tight", did not occur (1.7 GB) |
| L273 | 2D.1 | notes for Task 2D.2 (`_AliasSampling` unhashable; `work_sizes()` passes an empty table) | 1 | — | C | Task 2D.2 merged; reviewers confirm that `QROAMErase.required_wires` still ignores its table |
| L277 | 2E.2 | notes for Task 3.1 (one shared `AngleErase` target) | 1 | — | C | Task 3.1 merged (ledger row 3.1, K-1: every child converts once) |
| L278 | 2E.2 | `angle_data._dummy_wires` duplicates `qroam._dummy`; `AngleErase.required_wires` takes tables it does not use | 1 | yes | R | rate (a Task 5.1 row) |
| L284 | 2D.2 | notes for Task 3.1 (the registers the select must not write; the dirty work registers) | 1 | — | C | Task 3.1 merged; its children and work-pool tests |
| L287 | 2E.3 | notes for Task 3.1 (the order of the spin multiplexer and swaps) | 1 | — | C | Task 3.1 merged (`test_children_order_and_wires`) |
| L331 | 3.1 | notes for Tasks 3.3 and 3.4 (the select's registers and its prefix of the pool) | 1 | — | C | Tasks 3.3 and 3.4 merged (their children and work-pool tests) |
| L332 | 3.1 | `select._check` repeats `AngleErase`'s validation; the select checks lengths, not disjointness | 1 | yes | R | rate (a Task 5.1 row; with L339) |
| L334 | 3.2 | notes for Tasks 3.3 to 3.5 (named registers; the controlled classes) | 1 | — | C | Tasks 3.3 to 3.5 merged; E-3 runs after one walk step |
| L338 | 2E.4 | the R_z CNOT fans could skip `phase_grad[0]`, 848 CNOTs per walk less (optional) | 1 | yes | R | optional: declined unless raised (the construction mirrors Qualtran's; Clifford counts are reported, not compared) |
| L339 | 2E.4 | no dfthc operator validates wire disjointness (`check_wires` checks lengths) | 1 | yes | R | rate (a Task 5.1 row) |
| L340 | 2E.4 | notes for Tasks 3.1 and 3.4 (ladder order, work wires) | 1 | — | C | Tasks 3.1 and 3.4 merged |
| L344 | 2E.5 | notes for Phase 3 (the Majorana gate's wiring) | 1 | — | C | Tasks 3.1 and 3.4 merged |
| L254 | 1A.5 | E-1 at `H2_STO3G_TEST` cannot see a tail-direction error | 2 | — | C | `tests/ops/test_inner_select.py::test_action_against_the_encoded_model` checks every tail row, with both values of `sigma_1`, against the encoded model at `random-C3` on the strict tape path; Task 2E.4 pins the ladder orders |
| L255 | 1A.5 | the walk's normalization is Λ̃ from `encoded_block`, not `normalization(encoded_factors(...))` | 2 | — | C | the E checks and the real-walk QPE compare with `walk_sim.case(...).block()`, the encoded block (Λ̃) |
| L283 | 2D.2 | the outer round trip in `tests/ops/test_outer_prepare.py` runs only an all-ones schedule | 2 | no | R | rate (a Task 5.1 row; the spec 12.5 certificates already prove every outcome) |
| L285 | 2D.2 | notes for Task 3.5 (shapes that show the failure-sign difference) | 2 | — | C | Task 3.5 added `FAILS_C3` (C = 3, k2 = 0) with `encoded_block` as the oracle |
| L311 | 4.1 | the QPE's counts, wires and K-1 for Task 4.2 and the report | 2 | — | C | `tests/integration/test_qpe_counts.py` asserts worst-case T, M and A and the inventory with its non-vacuity checks; the report gives A and the wire formulas |
| L313 | 4.1 | lesson: a mode-specific branch needs a deterministic test in that mode | 2 | — | C | applied in Task 4.2 and in Q1b.1 (the postselected compiled round trip, `128eb16`) |
| L314 | 4.1 | for Task 5.1: a spread input for the release check at `sto3g`, t = 2; group-swap mutants die only through `631g`; the toy inventory's `startswith("C(")` | 2 | no | R | rate (a Task 5.1 row) |
| L322 | 3.4 | lesson: an instrument that patches global state needs its own tests; the inventory for Task 4.2 | 2 | — | C | Task 4.2 merged (`test_inventory_of_the_qpe_program`) |
| L326 | 3.3 | lessons for Tasks 3.4 and 3.5 (vacuous norm checks; harness self-tests) | 2 | — | C | Tasks 3.4 and 3.5 merged (the harness self-test pins S0 and S1) |
| L330 | 3.1 | lesson: branch classes use the values after the swap | 2 | — | C | E-1 covers it at both test presets, as the row says |
| L343 | 2E.5 | lesson: hand-written AND pairs need a tape-path test on the strict simulator | 2 | — | C | `test_action_against_the_encoded_model` and `test_dagger_inverts_the_select_on_every_branch` for the selects, the E checks for the walk |
| L230 | 0.1 | the worker test checks only the current invocation, not unset, empty and `0` | 3 | no | R | rate |
| L231 | 0.1 | the lock test checks three packages, not the whole lock against a normalized `pip freeze` | 3 | no | R | rate |
| L233 | 0.2 | watchdog: `proc.poll()` reaps the root early; a later scan can adopt a reused pid | 3 | no | R | rate |
| L234 | 0.2 | watchdog: a run returns about 10 s later than its docstring says | 3 | no | R | rate |
| L235 | 0.2 | watchdog: `proc.kill()` in `_final_output` can raise `PermissionError` inside `finally` | 3 | no | R | rate |
| L236 | 0.2 | watchdog `main()`: a command that cannot be executed ends in a traceback without a `WATCHDOG` line | 3 | no | R | rate |
| L237 | 0.2 | the watchdog's loop-path test has about 0.8 s of margin | 3 | no | R | rate |
| L238 | 0.2 | `dfthc.testing.resources` ignores "unable to determine the branch" for every caller | 3 | no | R | rate |
| L245 | 1H.1 | the watchdog's `ESCAPE` test can fail in a window of microseconds | 3 | no | W | closed at G5 if it passed in every Phase 5 run; a failure makes it R |
| L246 | 1B.1 | sparse simulator `_merge`: two-word keys about 6 times slower (optional speed-up) | 3 | no | R | optional: declined unless raised (the release tier projects at 25 to 80 minutes against 8 hours) |
| L247 | 1B.1 | the precondition messages could show the offending amplitude (optional) | 3 | no | R | optional: declined unless raised |
| L248 | 1B.1 | a measurement can grow an error by up to sqrt(2); log the final norm in Task 1B.3's benchmark | 3 | — | C | superseded: the E checks compare whole states, ancillas included, at 1e-10 against the encoded model, which fails on any norm growth (reviewers confirm) |
| L250 | 1C.1 | `PhaseGradientRotation` costs s − 1 Toffolis against Qualtran's s − 2 | 3 | — | C | reason code `rot-add` (spec 13.3, `REASON_CODES`), + 12 per walk against dfthcbe in `docs/resources.md` |
| L256 | 1C.3 | FeMoco read costs against dfthcbe | 3 | — | C | the report's FeMoco-54 rows `QROAMRead@...` with dfthcbe's and the paper's values and their codes |
| L257 | 1C.3 | the reads' Clifford counts are upper bounds and must be labelled | 3 | — | C | `clifford_kind`: "upper bound (CNOT, PauliX)" in `docs/resources.md` |
| L261 | 1C.2 | uniform-preparation costs against dfthcbe | 3 | — | C | the report's rows with `cmp-free`, `rot-add` and `unif-odd` |
| L262 | 1C.4 | erase fix-up costs against dfthcbe and the paper | 3 | — | C | the report's `PhaseFixup@...` rows with `exact-iter` and `onehot-pad` |
| L265 | 1A.6 | the `X_O_FAILS` factors come from our `random_factors` on both sides of the differential test | 3 | — | A | stays accepted (dfthcbe computes the oracle; the provenance records the input digests) |
| L268 | 1C.5 | X-measured qubits per erase against dfthcbe and the paper | 3 | — | C | the report's M column; after Q1b the block encodings' 852 junk measurements |
| L270 | 2D.1 | FeMoco outer pair costs | 3 | — | C | the report's FeMoco-54 rows of the outer prepare and unprepare with their codes |
| L271 | 2D.1 | keeping each alias swap's AND: 19 Toffolis per walk less for 14 dirty qubits | 3 | — | D | deferred: listed in `docs/resources.md`, "Deferred refinements" |
| L274 | 2E.2 | FeMoco angle-erase costs | 3 | — | C | the report's `AngleErase` rows with their codes |
| L276 | 2E.2 | an SF fix-up one-hot of ob = w_b − 1: 12 Toffolis per walk less | 3 | — | D | deferred: listed in `docs/resources.md`, "Deferred refinements" |
| L279 | 2D.2 | FeMoco inner pair costs | 3 | — | C | the report's FeMoco-54 rows of the inner prepare and unprepare with their codes |
| L280 | 2D.2 | persistent inner qubits against dfthcbe and the paper | 3 | — | C | the report's qubit comparison by group (`costs_reference.PAPER_QUBIT_GROUPS_FEMOCO`, Task Q1b) |
| L282 | 2D.2 | the swap undo with kept ANDs: 10 Toffolis per walk less | 3 | — | D | deferred, with L271 |
| L286 | 2E.3 | spin plumbing; state both of the paper's figures | 3 | — | C | `docs/resources.md`, FeMoco-54 summary: "The spin multiplexer is 0 in Table III and 1 per application in App. B.3 and B.7.1" |
| L293 | Q1b.2 | the report's "The junk registers' 462 wires ... share the angle register's wires" counts overlapping aliases | 3 | yes | R | rate (a Task 5.1 row); the sentence is a string in `scripts/resource_report.py`, so it changes only with another hashed fix |
| L289 | 4.5 | the FeMoco-54 summary still says "until then they show their closed form" although both walks are measured (`femoco_summary`, rendered in `docs/resources.md`) | 3 | yes | R | rate (a Task 5.1 row); a string in `scripts/resource_report.py`, so it changes only with another hashed fix, like L293 |
| L317 | 3.5 | tier budgets; "Task 5.2 runs the release tier" | 3 | — | T | closed by Task 5.2's release tier, with its wall time and peak |
| L318 | 3.5 | G3 runs `-m "not release"` | 3 | — | C | the G3 gate run (heavy-runs row of 2026-11-14: 1,116 passed) |
| L319 | 3.4 | FeMoco walk costs; reason codes for the reflections | 3 | — | C | spec 13.3's `ref-target`, `ref-unif`, `qpe-ctrl` and `ctrl-ref-t2`; the walk's codes sum to −204 against the paper and + 152 against dfthcbe |
| L323 | 3.3 | FeMoco block-encoding costs; a code for `REF_T2`'s −1 | 3 | — | C | `ref-unif` |
| L327 | 3.1 | FeMoco select costs; codes for the Majorana gate's −2 and the SF enable's + 1 | 3 | — | C | `maj-and` and `sf-enable` |
| L333 | 3.2 | reflection costs; report M and A separately, without zeros for the paper's missing splits | 3 | — | C | the report's M and A columns; its header says the paper itemizes neither |
| L335 | 2E.4 | FeMoco Givens costs | 3 | — | C | `rz-addsub` in the report's ladder rows |
| L341 | 2E.5 | FeMoco Majorana-gate costs; a reason code | 3 | — | C | `maj-and` |
| L272 | 2D.1 | spec 13.1's "Outer unprepare and erase" row could cite the remark after Eq. (127) | 4 | — | C | the row cites it ("the saving the paper notes after its Eq. (127)") |
| L275 | 2E.2 | spec 13.1's "Angle erase (x2)" reason | 4 | — | C | the row reads as the finding asks (+14 `exact-iter` and `no-prune`, +8 `onehot-pad`, +2 `sf-enable`) |
| L301 | 4.4 | standing procedure: a tree that differs from a tested one only in `docs/` carries the tier after `tests/test_docstrings.py` passes on it | 4 | — | P | closed at G5, after Phase 5 has followed it |
| L302 | 4.4 | spec 14's L-6 mitigation omits `ToyControlledWalkPower` | 4 | no | R | rate (a Task 5.1 row; a spec edit) |
| L321 | 3.4 | correct the master plan's "about 7,100 and 4,500" captured equations | 4 | — | C | master plan section 3, item 23 gives 7,449 and 4,795 |
| L325 | 3.3 | the same correction (7,449) | 4 | — | C | as L321 |

## The release tier estimate

The release tier is `scripts/test.sh` with no marker: 1,393 + N tests with 2 workers (spec 12.6, reduced; budget 8 hours). What follows is a projection for planning and for the user's go-ahead, not a bound. Every number it uses is quoted, with its source path and line, in `stages/P5/release-estimate-evidence.log`, written by a standard-library extractor that also recomputes the arithmetic (shown in full after the projection). Each release-only test:

| Release test (35) | Evidence | Time each (measured, or an allowance) | Peak |
| --- | --- | ---: | ---: |
| `test_e1_block_encoding[631g-S0-*]`, 28 inputs, one per test | `00000011` alone on the replica: `26.12s call`, 30.1 s wall, 414.7 MB (`release-e1-631g-00000011.log`); the Q1b writer's sample, 2 workers: 25.89 s (`stages/Q1B/release-sample.log`); Task 3.5's GPT review under heavy concurrent load: 56.84 s and 79.14 s, 413.8 and 435.3 MB | 26 s (up to 80 s loaded) | 0.44 GB |
| `test_e1_block_encoding[sto3g-S1-all]` | alone: `6.14s call`, 384.8 MB (`release-e1-sto3g-S1-all.log`); Q1b sample 6.01 s | 6 s | 0.39 GB |
| `test_e1_block_encoding[sto3g-seeded-fixed]` | not run: the same block encoding on 4 of those 16 inputs | 6 s allowance | 0.39 GB |
| `test_e3_controlled_walk[S1-all]` | Q1b sample (2 workers, Q1b tree): `80.30s call`; the three-test session's peak 618.7 MB | 80 s | 0.62 GB |
| `test_e3_controlled_walk[seeded-fixed]` | alone: `22.14s call`, 414.4 MB (`release-e3-seeded-fixed.log`) | 22 s | 0.42 GB |
| `test_qpe_of_the_walk_is_exact[sto3g-2-seeded]` | not run (over 5 minutes before Q1b): `1 passed ... in 487.43s`, peak 367.8 MB on 66 wires, two 64-bit key words (`stages/4Q/qpe-walk-release-4q.log`); since Q1b the program has 58 wires (56 + t), one key word, which `dfthc.testing.walk_sim`'s docstring puts at about 25 times faster per state | 487 s planning allowance from the measured pre-Q1b program; the post-Q1b runtime is unmeasured and is expected, but not proved, to be lower | 0.37 GB |
| `test_femoco_cache` | not run (FeMoco-shaped data, though it compiles nothing): it builds the FeMoco-54 rows in tape mode like the per-commit fixture `femoco_report`, whose setup took 1.43 s in a one-worker session with a 566.9 MB peak (`stages/4R/4r-per-test-evidence.log`) | 10 s allowance | 0.6 GB |
| `test_release_marked_probe` | an empty body | 0 s | — |

Release-only occupancy: 28 × 26.12 + 612.01 = 1,343 s (22 minutes), where 612.01 s is the other seven (6.14 + 6 + 80.30 + 22.14 + 487.43 + 10); under load (the 6-31G inputs at 79.14 s, the other seven tripled) 4,052 s. The rest of the tier, from earlier runs with 2 workers on trees with the same tests: the per-commit part, `1271 passed in 397.15s`, peak 1,949.0 MB (Task 4.5's tier, `.scratch/g1/4.5-tier.log`); the nightly part, `86 passed in 442.72s`, peak 1,871.7 MB (the Q1b writer's nightly tier on the Q1b tree, `stages/Q1B/q1b-nightly-tier.raw`). Tier occupancy 2 × 397.15 + 2 × 442.72 + 1,343 = 3,023 s, so a wall time of about 1,512 s (25.2 minutes) with 2 workers, at most about 1,999 s (33.3 minutes) if the QPE test starts last; under load, the 6-31G inputs at 79.14 s and everything else tripled, about 4,546 s (75.8 minutes). Peak: the per-commit tiers of Tasks 4.4 and 4.5 reached 2,201.1 and 1,949.0 MB and the G2 and G3 gate runs 2,776.1 and 2,480.7 MB; no release-only session exceeded 618.7 MB; so about 2 to 3 GB, under the watchdog's 7.5 GB kill point. Projection for the user: **about 25 to 35 minutes on a quiet machine, up to about 80 under load; peak about 2 to 3 GB.** It is a projection: two inputs are allowances (the QPE test's 487 s and `test_femoco_cache`'s 10 s), and the load factor is one observation's. When the G4 gate run (`-m "not release"`, 1,357 tests on the merged tree) has its heavy-runs row, the orchestrator replaces the per-commit and nightly parts by its wall time and peak.

The extractor (`stages/P5/extract_release_evidence.py`, standard library only) and its output (`stages/P5/release-estimate-evidence.log`):

```python
"""The evidence of the Phase 5 plan's release-tier projection (section "The release tier
estimate"), extracted from the logs it rests on, and the projection's arithmetic.

Standard library only (it imports neither dfthc nor PennyLane). For each source it prints the path
and every line, with its line number, that matches the source's patterns: the pytest result and
"s call" lines and the watchdog's `WATCHDOG {json}` line. Then it reads the numbers the projection
uses from those lines and prints the sums. Run from anywhere:
    python3 .scratch/plan-check/stages/P5/extract_release_evidence.py > .scratch/plan-check/stages/P5/release-estimate-evidence.log
"""

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
P5 = ".scratch/plan-check/stages/P5"
SOURCES = {
    "e1_631g": (f"{P5}/release-e1-631g-00000011.log", r"s call|passed|^WATCHDOG"),
    "e1_sto3g_s1": (f"{P5}/release-e1-sto3g-S1-all.log", r"s call|passed|^WATCHDOG"),
    "e3_seeded": (f"{P5}/release-e3-seeded-fixed.log", r"s call|passed|^WATCHDOG"),
    "q1b_sample": (".scratch/plan-check/stages/Q1B/release-sample.log",
                   r"s call|passed|^WATCHDOG"),
    "loaded_631g": (".superpowers/sdd/task-3.5-review-1.md", r"s call"),
    "qpe_pre_q1b": (".scratch/plan-check/stages/4Q/qpe-walk-release-4q.log",
                    r"s call|passed|^WATCHDOG"),
    "femoco_rows": (".scratch/plan-check/stages/4R/4r-per-test-evidence.log",
                    r"test_reason_codes_at_femoco|passed|^WATCHDOG"),
    "per_commit": (".scratch/g1/4.5-tier.log", r"passed|^WATCHDOG"),
    "nightly": (".scratch/plan-check/stages/Q1B/q1b-nightly-tier.raw", r"passed|^WATCHDOG"),
    "per_commit_4.4": (".scratch/g1/4.4-merge-tier.log", r"passed|^WATCHDOG"),
    "gate_g2": (".scratch/g2/nightly.log", r"passed|^WATCHDOG"),
    "gate_g3": (".scratch/g3/nightly.log", r"passed|^WATCHDOG"),
}

lines = {}
for name, (path, pattern) in SOURCES.items():
    text = (REPO / path).read_text().splitlines()
    found = [(i, line) for i, line in enumerate(text, start=1) if re.search(pattern, line)]
    lines[name] = found
    print(f"== {name}: {path}")
    for i, line in found:
        print(f"{i}: {line}")
    print()


def number(name, pattern):
    """The first group of the first extracted line of `name` that matches `pattern`."""
    for _, line in lines[name]:
        match = re.search(pattern, line)
        if match:
            return float(match[1].replace(",", ""))
    raise SystemExit(f"{name}: no line matches {pattern}")


def peak(name):
    for _, line in lines[name]:
        if line.startswith("WATCHDOG "):
            return json.loads(line[len("WATCHDOG "):])["peak_mb"]
    raise SystemExit(f"{name}: no WATCHDOG line")


e1_631g = number("e1_631g", r"([\d.]+)s call")
e1_631g_loaded = max(number("loaded_631g", r"`10001000`: ([\d.]+) s call"),
                     number("loaded_631g", r"`00000011`: ([\d.]+) s call"))
e1_sto3g_s1 = number("e1_sto3g_s1", r"([\d.]+)s call")
e1_seeded_allowance = 6.0
e3_s1 = number("q1b_sample", r"([\d.]+)s call\s+\S+test_e3_controlled_walk\[S1-all\]")
e3_seeded = number("e3_seeded", r"([\d.]+)s call")
qpe_allowance = number("qpe_pre_q1b", r"in ([\d.]+)s")
femoco_rows = number("femoco_rows", r"([\d.]+)s setup\s+tests/report/test_report.py::test_reason_codes")
femoco_allowance = 10.0
per_commit = number("per_commit", r"^1271 passed in ([\d.]+)s")
nightly = number("nightly", r"^86 passed in ([\d.]+)s")
others = e1_sto3g_s1 + e1_seeded_allowance + e3_s1 + e3_seeded + qpe_allowance + femoco_allowance
release_only = 28 * e1_631g + others
release_only_loaded = 28 * e1_631g_loaded + 3 * others
occupancy = 2 * per_commit + 2 * nightly + release_only
occupancy_loaded = 3 * (2 * per_commit + 2 * nightly) + release_only_loaded
print("== projection (2 workers; occupancy is the summed test time of both workers)")
print(f"inputs: 6-31G E-1 input {e1_631g} s (loaded {e1_631g_loaded} s); sto3g S1-all "
      f"{e1_sto3g_s1} s; sto3g seeded-fixed allowance {e1_seeded_allowance} s; E-3 S1-all "
      f"{e3_s1} s; E-3 seeded-fixed {e3_seeded} s; QPE t = 2 planning allowance {qpe_allowance} s "
      f"(pre-Q1b measurement); test_femoco_cache allowance {femoco_allowance} s (the same rows "
      f"built in {femoco_rows} s); per-commit part {per_commit} s; nightly part {nightly} s")
print(f"release-only occupancy: 28 x {e1_631g} + {others:.2f} = {release_only:.0f} s "
      f"(loaded: 28 x {e1_631g_loaded} + 3 x {others:.2f} = {release_only_loaded:.0f} s)")
print(f"tier occupancy: 2 x {per_commit} + 2 x {nightly} + {release_only:.0f} = {occupancy:.0f} s;"
      f" wall with 2 workers {occupancy / 2:.0f} s ({occupancy / 120:.1f} min), at most "
      f"{occupancy / 2 + qpe_allowance:.0f} s ({(occupancy / 2 + qpe_allowance) / 60:.1f} min) if "
      f"the QPE test starts last")
print(f"under load (the 6-31G inputs at {e1_631g_loaded} s, everything else tripled): occupancy "
      f"{occupancy_loaded:.0f} s, wall {occupancy_loaded / 2:.0f} s "
      f"({occupancy_loaded / 120:.1f} min)")
peaks = {name: peak(name) for name in ("per_commit", "per_commit_4.4", "nightly", "gate_g2",
                                       "gate_g3", "e1_631g", "e1_sto3g_s1", "e3_seeded",
                                       "q1b_sample", "qpe_pre_q1b", "femoco_rows")}
print("peaks (MB): " + ", ".join(f"{name} {value}" for name, value in peaks.items()))
print(f"largest tier peak {max(peaks[n] for n in ('per_commit', 'per_commit_4.4', 'nightly', 'gate_g2', 'gate_g3'))} MB; "
      f"largest release-only session peak "
      f"{max(peaks[n] for n in ('e1_631g', 'e1_sto3g_s1', 'e3_seeded', 'q1b_sample', 'qpe_pre_q1b', 'femoco_rows'))} MB")
```

```text
== e1_631g: .scratch/plan-check/stages/P5/release-e1-631g-00000011.log
6: 26.12s call     tests/integration/test_end_to_end.py::test_e1_block_encoding[631g-S0-00000011]
9: 1 passed in 28.14s
10: WATCHDOG {"command": ["/Users/david.wierichs/repos/dfthc-be/.venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider", "-n", "1", "tests/integration/test_end_to_end.py::test_e1_block_encoding[631g-S0-00000011]", "-q", "--durations=0"], "pid": 12955, "returncode": 0, "seconds": 30.1, "peak_mb": 414.7, "samples": 571, "killed": null, "survivors": [], "cap_mb": 3000.0, "timeout_s": 600.0, "priority": "taskpolicy -c utility", "heavy": false}

== e1_sto3g_s1: .scratch/plan-check/stages/P5/release-e1-sto3g-S1-all.log
6: 6.14s call     tests/integration/test_end_to_end.py::test_e1_block_encoding[sto3g-S1-all]
9: 1 passed in 8.31s
10: WATCHDOG {"command": ["/Users/david.wierichs/repos/dfthc-be/.venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider", "-n", "1", "tests/integration/test_end_to_end.py::test_e1_block_encoding[sto3g-S1-all]", "-q", "--durations=0"], "pid": 13155, "returncode": 0, "seconds": 10.5, "peak_mb": 384.8, "samples": 199, "killed": null, "survivors": [], "cap_mb": 3000.0, "timeout_s": 600.0, "priority": "taskpolicy -c utility", "heavy": false}

== e3_seeded: .scratch/plan-check/stages/P5/release-e3-seeded-fixed.log
6: 22.14s call     tests/integration/test_end_to_end.py::test_e3_controlled_walk[seeded-fixed]
9: 1 passed in 24.32s
10: WATCHDOG {"command": ["/Users/david.wierichs/repos/dfthc-be/.venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider", "-n", "1", "tests/integration/test_end_to_end.py::test_e3_controlled_walk[seeded-fixed]", "-q", "--durations=0"], "pid": 13309, "returncode": 0, "seconds": 26.5, "peak_mb": 414.4, "samples": 501, "killed": null, "survivors": [], "cap_mb": 3000.0, "timeout_s": 600.0, "priority": "taskpolicy -c utility", "heavy": false}

== q1b_sample: .scratch/plan-check/stages/Q1B/release-sample.log
1: 80.30s call     tests/integration/test_end_to_end.py::test_e3_controlled_walk[S1-all]
2: 25.89s call     tests/integration/test_end_to_end.py::test_e1_block_encoding[631g-S0-00000011]
3: 6.01s call     tests/integration/test_end_to_end.py::test_e1_block_encoding[sto3g-S1-all]
4: 3 passed in 88.48s (0:01:28)
5: WATCHDOG {"command": ["/Users/david.wierichs/repos/dfthc-be/.venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider", "-n", "2", "tests/integration/test_end_to_end.py::test_e1_block_encoding[sto3g-S1-all]", "tests/integration/test_end_to_end.py::test_e1_block_encoding[631g-S0-00000011]", "tests/integration/test_end_to_end.py::test_e3_controlled_walk[S1-all]", "-m", "release", "-q", "--durations=5"], "pid": 55977, "returncode": 0, "seconds": 90.6, "peak_mb": 618.7, "samples": 1716, "killed": null, "survivors": [], "cap_mb": 3000.0, "timeout_s": 1200.0, "priority": "taskpolicy -c utility", "heavy": false}

== loaded_631g: .superpowers/sdd/task-3.5-review-1.md
146:   - `10001000`: 79.14 s call, 435.3 MB process-tree peak.
147:   - `00000011`: 56.84 s call, 413.8 MB process-tree peak.

== qpe_pre_q1b: .scratch/plan-check/stages/4Q/qpe-walk-release-4q.log
3: 487.37s call     tests/integration/test_qpe_walk.py::test_qpe_of_the_walk_is_exact[sto3g-2-seeded]
5: 1 passed, 5 deselected in 487.43s (0:08:07)
6: WATCHDOG {"command": ["/Users/david.wierichs/repos/dfthc-be/.venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider", "-n", "0", "tests/integration/test_qpe_walk.py", "-q", "-m", "release", "--durations=3"], "pid": 28658, "returncode": 0, "seconds": 490.7, "peak_mb": 367.8, "samples": 9287, "killed": null, "survivors": [], "cap_mb": 3000.0, "timeout_s": 1500.0, "priority": "taskpolicy -c utility", "heavy": false}

== femoco_rows: .scratch/plan-check/stages/4R/4r-per-test-evidence.log
16: 1.43s setup    tests/report/test_report.py::test_reason_codes_at_femoco
19: 105 passed, 2 deselected in 53.70s
20: WATCHDOG {"command": ["/Users/david.wierichs/repos/dfthc-be/.venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider", "-n", "0", "tests/report/test_report.py", "tests/test_docstrings.py", "tests/report/test_femoco_cache.py", "-m", "not slow and not release", "-q", "--durations=15"], "pid": 8975, "returncode": 0, "seconds": 55.7, "peak_mb": 566.9, "samples": 1055, "killed": null, "survivors": [], "cap_mb": 3000.0, "timeout_s": 1200.0, "priority": "taskpolicy -c utility", "heavy": false}

== per_commit: .scratch/g1/4.5-tier.log
22: 1271 passed in 397.15s (0:06:37)
23: WATCHDOG {"command": ["scripts/test.sh", "-m", "not slow and not release", "-q", "-p", "no:cacheprovider"], "pid": 8181, "returncode": 0, "seconds": 399.3, "peak_mb": 1949.0, "samples": 7547, "killed": null, "survivors": [], "cap_mb": 8192, "timeout_s": 1800.0, "priority": "taskpolicy -c utility", "heavy": false}

== nightly: .scratch/plan-check/stages/Q1B/q1b-nightly-tier.raw
6: 86 passed in 442.72s (0:07:22)
7: WATCHDOG {"command": ["/Users/david.wierichs/repos/dfthc-be/.venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider", "-n", "2", "-m", "slow and not release", "-q"], "pid": 51097, "returncode": 0, "seconds": 445.5, "peak_mb": 1871.7, "samples": 8424, "killed": null, "survivors": [], "cap_mb": 3000.0, "timeout_s": 1800.0, "priority": "taskpolicy -c utility", "heavy": false}

== per_commit_4.4: .scratch/g1/4.4-merge-tier.log
21: 1221 passed in 455.31s (0:07:35)
22: WATCHDOG {"command": ["scripts/test.sh", "-m", "not slow and not release", "-q", "-p", "no:cacheprovider"], "pid": 36287, "returncode": 0, "seconds": 457.8, "peak_mb": 2201.1, "samples": 8652, "killed": null, "survivors": [], "cap_mb": 8192, "timeout_s": 1800.0, "priority": "taskpolicy -c utility", "heavy": false}

== gate_g2: .scratch/g2/nightly.log
17: 934 passed in 327.94s (0:05:27)
18: WATCHDOG {"command": ["scripts/test.sh", "-m", "not release", "-q", "-p", "no:cacheprovider"], "pid": 81141, "returncode": 0, "seconds": 330.5, "peak_mb": 2776.1, "samples": 6238, "killed": null, "survivors": [], "cap_mb": 8192, "timeout_s": 3600.0, "priority": "taskpolicy -c utility", "heavy": false}

== gate_g3: .scratch/g3/nightly.log
36: 1116 passed in 680.82s (0:11:20)
37: WATCHDOG {"command": ["scripts/test.sh", "-m", "not release", "-q", "-p", "no:cacheprovider", "--durations=15"], "pid": 68607, "returncode": 0, "seconds": 683.4, "peak_mb": 2480.7, "samples": 12909, "killed": null, "survivors": [], "cap_mb": 8192, "timeout_s": 3600.0, "priority": "taskpolicy -c utility", "heavy": false}

== projection (2 workers; occupancy is the summed test time of both workers)
inputs: 6-31G E-1 input 26.12 s (loaded 79.14 s); sto3g S1-all 6.14 s; sto3g seeded-fixed allowance 6.0 s; E-3 S1-all 80.3 s; E-3 seeded-fixed 22.14 s; QPE t = 2 planning allowance 487.43 s (pre-Q1b measurement); test_femoco_cache allowance 10.0 s (the same rows built in 1.43 s); per-commit part 397.15 s; nightly part 442.72 s
release-only occupancy: 28 x 26.12 + 612.01 = 1343 s (loaded: 28 x 79.14 + 3 x 612.01 = 4052 s)
tier occupancy: 2 x 397.15 + 2 x 442.72 + 1343 = 3023 s; wall with 2 workers 1512 s (25.2 min), at most 1999 s (33.3 min) if the QPE test starts last
under load (the 6-31G inputs at 79.14 s, everything else tripled): occupancy 9091 s, wall 4546 s (75.8 min)
peaks (MB): per_commit 1949.0, per_commit_4.4 2201.1, nightly 1871.7, gate_g2 2776.1, gate_g3 2480.7, e1_631g 414.7, e1_sto3g_s1 384.8, e3_seeded 414.4, q1b_sample 618.7, qpe_pre_q1b 367.8, femoco_rows 566.9
largest tier peak 2776.1 MB; largest release-only session peak 618.7 MB
```

## Risks and open questions

1. **The QPE release test's time since Q1b is not measured.** The projection carries it at its pre-Q1b 487 s as a planning allowance; the changed program is expected, but not proved, to be faster, and the test is one item of one worker, so a slower run moves the wall time by about as much as it exceeds 487 s.
2. **Load-sensitive tests in the release tier.** `tests/canary/test_l21_compile_time.py::test_compile_time_scaling` failed once under load (ledger row L-21) and the watchdog's timing tests race under parallel load (rows L237, L245). Step 10 reruns such a test alone before treating it as a defect.
3. **A Critical finding that needs a design change** (one that changes counts or the spec) cannot be fixed in Task 5.2: it goes to the user (master plan 12.6, "Always escalate"), and the final merge waits.
4. **The area split can hide a defect that spans two areas.** The mitigations are in section "The review package and its split".
5. **The user may override Q-1 at the end.** Phase 5 then stops; restoring the baseline needs a new task, report regeneration and a FeMoco-54 pair. Because the current contingency pair has not run when this answer is taken, that pair would be compiles 4 and 5 of about 6; the new plan must account for it, and any later compile beyond about six needs the user's approval (P5-M8).
6. **The contingency recompile may be refused, killed or fail.** The run stops before the controlled walk; D-3 is unmet until the user decides where the compile runs (master plan section 2, "Where compiles above the cap run"). The projection from the small systems alone (the `fe2s2` and `fe4s4` entries become history) is the Q1b dry run's: walk 198.2 s at 3,892.5 MB, controlled walk 195.9 s at 4,399.9 MB (`stages/Q1B/projections-q1b-small-only.log`), within 7.5 GB and 60 minutes; Task 4.5 measured 34.6 s at 1,692 MB and 36.0 s at 1,675 MB.
7. **`main` may move during Phase 5** (ledger commits only, by rule). Each task measures its base counts in its Step 1; the ledger-row mapping (`map_rows.py`) runs on the current ledger.

Open questions for the user: two, asked together in Task 5.2, Step 7 (the Q-1 confirmation and the release go-ahead). Nothing else needs the user unless an escalation arises.

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

## File structure

| File | Task | Created or changed by | Responsibility |
| --- | --- | --- | --- |
| `.superpowers/sdd/final-review/` (git-ignored) | 5.1 | orchestrator, reviewers | the four area packages, `excluded.txt`, `hashed-files.txt`, `ledger-rows.txt`, the eight reviews |
| `docs/agents/final-review.md` | 5.1 | orchestrator, on `main` | the consolidated findings list (format in Task 5.1, Step 9) |
| files named by the findings | 5.2 | the fixer, on `phase5/5.2` | the fixes and their tests (never `docs/resources.*`, the ledger or `final-review.md`) |
| `docs/resources.md`, `docs/resources.json` | 5.2 | orchestrator, on `phase5/5.2`, only after a hashed fix | the regenerated report with the recompiled FeMoco-54 pair |
| `docs/agents/LEDGER.md` | 5.1, 5.2 | orchestrator, on `main` | task rows, heavy runs, the D checklist, the Minor-findings closures, decisions, gate G5 |
| `.scratch/g5/` (git-ignored) | 5.2 | orchestrator | the logs of the report runs and the release tier |

---

### Task 5.1: Whole-branch final review

**Files:** Create `docs/agents/final-review.md` (orchestrator, on `main`); modify `docs/agents/LEDGER.md` (row 5.1). Work files under `.superpowers/sdd/final-review/`. No branch.

**Interfaces:**
- Consumes: local `main` at G4 (Task 4.5 merged, this plan approved), the scripts `stages/P5/review_areas.sh` and `stages/P5/map_rows.py` in `.scratch/plan-check/` (shown in full in this plan), this plan's table "Open ledger rows and their dispositions".
- Produces: `docs/agents/final-review.md` in the format of Step 9: findings C-n, I-n and m-n, each with **Where**, **Evidence**, **Fix**, **Hashed**, **D criterion** and **Ledger row**, **Raised by**, **Decision** (`fix`, `declined` or `deferred`, with the reason); the table of the 72 open rows with their final dispositions; the T2 to T6 table of the 34 audited classes. Task 5.2's fixer takes every finding whose decision is `fix`.

Throughout Task 5.1: `<main>` is `/Users/david.wierichs/repos/dfthc-be`; the orchestrator commits nothing on `main` from Step 1 until Step 10, and runs no implementer and no heavy process while reviewers run.

- [ ] **Step 1: Check the preconditions and fix the reviewed tree**

```bash
cd /Users/david.wierichs/repos/dfthc-be
git status --short --branch
git branch --no-merged main
REVIEW_HEAD=$(git rev-parse main); echo "$REVIEW_HEAD"
PYTHONPATH=src .venv/bin/python -c "from dfthc.testing.provenance import code_hash; print(code_hash())"
scripts/test.sh -m "not slow and not release" --collect-only -q -p no:cacheprovider | tail -1
mkdir -p .superpowers/sdd/final-review
.venv/bin/python .scratch/plan-check/stages/P5/map_rows.py > .superpowers/sdd/final-review/ledger-rows.txt
tail -1 .superpowers/sdd/final-review/ledger-rows.txt
PYTHONPATH=src .venv/bin/python -c "from dfthc.testing.provenance import hashed_files, REPO; print('\n'.join(str(p.relative_to(REPO)) for p in hashed_files()))" > .superpowers/sdd/final-review/hashed-files.txt
wc -l < .superpowers/sdd/final-review/hashed-files.txt
git config --local --get-regexp '^(remote\.|branch\..*\.(remote|merge))' > .superpowers/sdd/final-review/git-remote-config.before
git for-each-ref --format='%(refname) %(objectname)' refs/remotes > .superpowers/sdd/final-review/remote-refs.before
cat .superpowers/sdd/final-review/git-remote-config.before
wc -c < .superpowers/sdd/final-review/remote-refs.before
```

The last two snapshots are D-7's baseline; Task 5.2, Step 12 takes them again after the merge and compares with `cmp`.

Expected: `## main...origin/main [gone]` and nothing else (the configured upstream was never fetched; no remote-tracking ref exists); no output from `git branch --no-merged main` (Task 4.5 merged); the 40-digit head; `5e3eed1e7a9535765299d6c90afe643f96a5155e7eb43d43e039864133fbba8e` (Task 4.5's merge and the ledger and plan commits change no hashed file); `1271/1392 tests collected (121 deselected) in ...`; the mapping's last line `72 rows, equal to the open and accepted rows at 1c15419; dispositions {'A': 2, 'C': 41, 'D': 3, 'P': 1, 'R': 23, 'T': 1, 'W': 1}; <m> moved, <c> changed in /Users/david.wierichs/repos/dfthc-be/docs/agents/LEDGER.md`, with m = 72 − c once P5-L4 is applied (it inserts 15 lines above the list; on the edited copy of `1c15419`'s ledger: `72 moved, 0 changed`, `stages/P5/map-rows-edited.log`) and c = 0 unless the orchestrator edited a row since `1c15419`; `30`; the four configuration lines below (as on `1c15419`, `stages/P5/git-remote-config.now`) and `0` (no remote-tracking ref):

```text
remote.origin.url git@github.com:PennyLaneAI/dfthc-be.git
remote.origin.fetch +refs/heads/*:refs/remotes/origin/*
branch.main.remote origin
branch.main.merge refs/heads/main
```

A different hash means a hashed file changed after Task 4.5's compile: stop and find out why before reviewing (`test_femoco_cache` would fail). A "changed" row means the orchestrator edited that row after `a221d97`: the reviewers then take the row from the current ledger by its task and finding text.

- [ ] **Step 2: Write the four area packages**

```bash
cd /Users/david.wierichs/repos/dfthc-be
bash .scratch/plan-check/stages/P5/review_areas.sh 993b8fb "$REVIEW_HEAD" .superpowers/sdd/final-review
grep -v -E '^ (spikes/|docs/superpowers/plans/|data/dfthc/|tests/fixtures/dfthcbe/|docs/resources\.json|docs/agents/LEDGER\.md)' .superpowers/sdd/final-review/excluded.txt
```

Expected: four lines `.superpowers/sdd/final-review/area-<k>-<name>.diff: <bytes> bytes, <lines> lines`, each within 1% of the sizes measured on `c443c51` (397,152; 482,051; 488,607; 419,089 bytes; area 4 grows by the few hundred bytes of P5-S1, P5-S2 and P5-H1), then `.superpowers/sdd/final-review/excluded.txt:  <f> files changed, ...` with f = 86 plus the plan files this plan and the later commits add; the `grep -v` prints only the last line of `excluded.txt` (its summary): no production, test or document file is left out.

- [ ] **Step 3: Fill the reviewer prompts**

The two prompts below are complete except for the slots in braces, which the orchestrator fills from the area table that follows them and from these values: `{repo}` = `/Users/david.wierichs/repos/dfthc-be`; `{review_head}` = Step 1's head; `{out_dir}` = `/Users/david.wierichs/repos/dfthc-be/.superpowers/sdd/final-review`; `{scratch}` = `/Users/david.wierichs/repos/dfthc-be/.scratch/final-review-area-{k}-<opus or gpt>`; `{package_bytes}` = Step 2's size of the area's package.

Opus (deep correctness):

```text
description: "Final review, area {k} ({area_name}), Opus"
model: inherit
prompt: |
  You are one of the eight reviewers of the DFTHC project's whole-branch final review (master
  plan Task 5.1; superpowers requesting-code-review). The project implements Low et al., PRX 15,
  041016 (2025), arXiv:2502.15882, in PennyLane: the DFTHC block encoding, the qubitization walk,
  the controlled walk and a small iterative QPE, as Operator2 building blocks compiled with
  Catalyst qp.qjit(capture=True) and graph decomposition, resources from qp.analyze, compared with
  the paper and with the Qualtran implementation dfthcbe, documented and never tuned. Opus (you:
  deep correctness) and GPT-5.6 (spec compliance and quality) each review the four areas, in
  parallel per area; you have area {k}, {area_name}. Do not pre-judge severities; report what you
  find, with evidence. Acknowledge what is done well before the issues.

  Read first, in this order:
  1. {repo}/docs/agents/HANDOFF.md, sections 1, 2, 3.5, 5, 6, 7 and 11.
  2. {repo}/docs/superpowers/plans/2026-11-11-dfthc-phase5-final.md: sections "Hashed and
     unhashed fixes", "Open ledger rows and their dispositions" (the rows of area {k}; their
     current ledger lines are in {out_dir}/ledger-rows.txt) and "Task 5.1", Step 4 (the layout
     of your review).
  3. {repo}/docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md: section 2.3 (D-1 to
     D-7) and sections {spec_sections}.
  4. Your package, {out_dir}/area-{area_slug}.diff ({package_bytes} bytes): the commits, the
     file list and the full diff of 993b8fb..{review_head} restricted to the area's paths. It is
     your view of the area. Open files outside it only for a risk you can name, and name it.

  The reviewed tree is local main at {review_head}; nothing changes main while you review.
  Read-only: do not modify the working tree, the index, HEAD, any branch, or any file outside
  {out_dir} and {scratch}. Never commit, push, fetch or touch remotes.

  Check independently, with file:line evidence:
  {opus_focus}
  D criteria to evidence from this area: {d_list}.
  The ledger rows of area {k}: rate every R row at the code (Critical, Important, Minor, or
  decline, with evidence); for every C, A, D, W, P or T row, confirm the proposed closure or
  object with evidence.

  Allowed runs (master plan 12.9: the orchestrator may run heavy work on this laptop, and the
  other reviewer of your area runs beside you):
  - One process at a time, one pytest worker, every run under the watchdog, from {repo}:
      DFTHC_TEST_WORKERS=1 PYTHONPATH=src .venv/bin/python -m dfthc.testing.watchdog \
          --cap-mb 3000 --timeout-min 15 -- scripts/test.sh <node ids> -q -p no:cacheprovider
    Leave TMPDIR unset when the node ids include tests/test_infrastructure.py.
  - Per-commit tests; of the slow tests only tests/integration/test_end_to_end.py,
    tests/ops/test_certificates_*.py, the cases fails-c3 and sto3g-1 of
    tests/integration/test_qpe_walk.py, and tests/report/test_femoco_cache.py (it compiles
    nothing).
  - Compiles only at H2_STO3G_TEST, H2_631G_TEST and FAILS_C3, of the toy walk, and of
    table-free operators. Nothing at MID_NAPHTHALENE, at a vendored system or at FeMoco-54.
    Never run scripts/resource_report.py and never call its main(): the counts of fe2s2, fe4s4
    and FeMoco-54 are in docs/resources.json and docs/resources.md; read them.
  - Mutants: at most 10, checked with per-commit tests only, in your own copy, never in {repo}:
      mkdir -p {scratch} && git -C {repo} archive {review_head} | tar -x -C {scratch}
      cd {scratch} && CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src \
          {repo}/.venv/bin/python -m dfthc.testing.watchdog --cap-mb 3000 --timeout-min 15 -- \
          {repo}/.venv/bin/python -m pytest -p no:cacheprovider -n 1 <node ids> -q
    (scripts/test.sh inside .scratch resolves to the main checkout through git: never use it
    there.)
  - Stop any run that would go beyond these limits, and say so in your review.

  Hashed files: {out_dir}/hashed-files.txt (30 files) and requirements-nightly.lock. A change of
  their code, not of docstrings or comments, changes code_hash(); the orchestrator must then
  regenerate the report and recompile the FeMoco-54 walk and controlled walk. Mark every fix you
  propose "Hashed: yes" or "Hashed: no" by that rule.

  Write your review to {out_dir}/area-{k}-opus.md in the layout of the plan's Task 5.1, Step 4.
  Reply in at most 12 lines: the verdict, the count of findings per severity, the ledger rows you
  object to, the file path.
```

GPT-5.6 (spec compliance and quality):

```text
description: "Final review, area {k} ({area_name}), GPT-5.6"
model: gpt-5.6-sol-high-fast
prompt: |
  You are one of the eight reviewers of the DFTHC project's whole-branch final review (master
  plan Task 5.1; superpowers requesting-code-review). The project implements Low et al., PRX 15,
  041016 (2025), arXiv:2502.15882, in PennyLane: the DFTHC block encoding, the qubitization walk,
  the controlled walk and a small iterative QPE, as Operator2 building blocks compiled with
  Catalyst qp.qjit(capture=True) and graph decomposition, resources from qp.analyze, compared with
  the paper and with the Qualtran implementation dfthcbe, documented and never tuned. GPT-5.6
  (you: spec compliance and quality) and Opus (deep correctness) each review the four areas, in
  parallel per area; you have area {k}, {area_name}. Categorize by actual severity; acknowledge
  what is done well before the issues; give a clear verdict.

  Read first, in this order:
  1. {repo}/docs/agents/HANDOFF.md, sections 1, 2, 3.5, 5, 6, 7 and 11.
  2. {repo}/docs/superpowers/plans/2026-11-11-dfthc-phase5-final.md: sections "Hashed and
     unhashed fixes", "Open ledger rows and their dispositions" (the rows of area {k}; their
     current ledger lines are in {out_dir}/ledger-rows.txt) and "Task 5.1", Step 4 (the layout
     of your review).
  3. {repo}/docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md: section 2.3 (D-1 to
     D-7) and sections {spec_sections}.
  4. Your package, {out_dir}/area-{area_slug}.diff ({package_bytes} bytes): the commits, the
     file list and the full diff of 993b8fb..{review_head} restricted to the area's paths. It is
     your view of the area. Open files outside it only for a risk you can name, and name it.

  The reviewed tree is local main at {review_head}; nothing changes main while you review.
  Read-only: do not modify the working tree, the index, HEAD, any branch, or any file outside
  {out_dir} and {scratch}. Never commit, push, fetch or touch remotes.

  Part 1, spec compliance, and Part 2, code quality, with file:line evidence:
  {gpt_focus}
  D criteria to evidence from this area: {d_list}.
  The ledger rows of area {k}: rate every R row at the code (Critical, Important, Minor, or
  decline, with evidence); for every C, A, D, W, P or T row, confirm the proposed closure or
  object with evidence.

  Allowed runs (master plan 12.9: the orchestrator may run heavy work on this laptop, and the
  other reviewer of your area runs beside you):
  - One process at a time, one pytest worker, every run under the watchdog, from {repo}:
      DFTHC_TEST_WORKERS=1 PYTHONPATH=src .venv/bin/python -m dfthc.testing.watchdog \
          --cap-mb 3000 --timeout-min 15 -- scripts/test.sh <node ids> -q -p no:cacheprovider
    Leave TMPDIR unset when the node ids include tests/test_infrastructure.py.
  - Per-commit tests; of the slow tests only tests/integration/test_end_to_end.py,
    tests/ops/test_certificates_*.py, the cases fails-c3 and sto3g-1 of
    tests/integration/test_qpe_walk.py, and tests/report/test_femoco_cache.py (it compiles
    nothing).
  - Compiles only at H2_STO3G_TEST, H2_631G_TEST and FAILS_C3, of the toy walk, and of
    table-free operators. Nothing at MID_NAPHTHALENE, at a vendored system or at FeMoco-54.
    Never run scripts/resource_report.py and never call its main(): the counts of fe2s2, fe4s4
    and FeMoco-54 are in docs/resources.json and docs/resources.md; read them.
  - Mutants: at most 10, checked with per-commit tests only, in your own copy, never in {repo}:
      mkdir -p {scratch} && git -C {repo} archive {review_head} | tar -x -C {scratch}
      cd {scratch} && CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src \
          {repo}/.venv/bin/python -m dfthc.testing.watchdog --cap-mb 3000 --timeout-min 15 -- \
          {repo}/.venv/bin/python -m pytest -p no:cacheprovider -n 1 <node ids> -q
    (scripts/test.sh inside .scratch resolves to the main checkout through git: never use it
    there.)
  - Stop any run that would go beyond these limits, and say so in your review.

  Hashed files: {out_dir}/hashed-files.txt (30 files) and requirements-nightly.lock. A change of
  their code, not of docstrings or comments, changes code_hash(); the orchestrator must then
  regenerate the report and recompile the FeMoco-54 walk and controlled walk. Mark every fix you
  propose "Hashed: yes" or "Hashed: no" by that rule.

  Write your review to {out_dir}/area-{k}-gpt.md in the layout of the plan's Task 5.1, Step 4.
  Reply in at most 12 lines: the verdict, the count of findings per severity, the ledger rows you
  object to, the file path.
```

The slots per area:

| Slot | Area 1 | Area 2 | Area 3 | Area 4 |
| --- | --- | --- | --- | --- |
| `{area_name}` | operators | operator tests | data, report and tooling | documents |
| `{area_slug}` | `1-operators` | `2-operator-tests` | `3-data-report-tooling` | `4-documents` |
| `{spec_sections}` | 3.5, 5.1 to 5.3, 6, 7.2 to 7.6, 8.4, 8.5, 9.0 to 9.24, 10, 11, 16 (Q-1) | 9.0 (T2 to T6), 9.24, 11.4, 12.1 to 12.6 | 7.1, 8.1 to 8.6, 12.3, 12.6, 13, 14, 15 | 2, 5.2, 14, and the whole spec for internal consistency |
| `{d_list}` | D-1 (the operators and their documentation), D-4 (the QPE code) | D-1 (T2 to T6 for every class: give the table of Step 4), D-2, D-4 (the tests) | D-3, D-5 | D-6 |

`{opus_focus}` per area:
- Area 1: phases and signs (spec 6.2, 6.5, 6.6; global phases under control); register widths, endianness and wire aliasing (6.7, 10.2: the junk on the angle wires, the lent `extra_onehot`); measurement semantics, resets and hints, and that every outcome restores the state (12.5); failure branches (8.5) and the loaded set (9.3); the adjoint and controlled classes, their pins, and nothing measuring under control (7.4 rules 5 and 6, 7.5); declared resource keys equal to the emitted operators; the QPE round's corrections and their single binding of the work pool (11.1); re-derive two counts by hand from the code at FeMoco-54, one alias read's T and the walk's M, and compare with `dfthc.costs`.
- Area 2: give the T2 to T6 table of the 34 classes `tests/test_docstrings.py` audits (class, test module, one test name per T); tests that assert behaviour (vacuous assertions, an oracle that is half of its own harness; use your mutants where a test looks weak); exact counts asserted against `dfthc.costs` and never a closed form against itself; the certificates' premises and that every erase block of spec 12.5 is certified; E-1 to E-4: inputs, schedules, the known blind spots (ledger row 3.5, E-check coverage) and the release schedules; the QPE tests (spec 11.4; the real-walk QPE).
- Area 3: the report (D-3): read `docs/resources.json` and `docs/resources.md` directly: every measured row equals its closed form, the cross-check at FeMoco-54 (plus the 852 junk measurements in M), every delta's reason codes sum to it, the cache semantics (current, history, pending) and the large-run stop policy; `costs_reference` against `tests/fixtures/dfthcbe/report_toffolis.json`; the scope of `provenance.code_hash` (what a change it does not see could invalidate); the watchdog's fail-closed behaviour, which guards the release tier; `scripts/test.sh`; the sparse simulator's handlers and strict mode; the encoded model and the dfthcbe fixtures' provenance.
- Area 4: each document against the code: names, signatures, register groups and wire totals, the walk's call order, the report's wording; the spec against itself after Q1b and Task 4.5 (stale statements, numbers that disagree between sections, spec 9.24 against `dfthc.costs`); spec 14 against `docs/limitations.md` and the canaries' recorded states; HANDOFF against the workflow as run; `docs/environment.md` against `requirements-nightly.lock`.

`{gpt_focus}` per area:
- Area 1: for every operator class, its spec 9 entry (wires, compilable and dynamic arguments, pre- and postconditions, decomposition, cost); the nine spec 9.0 docstring sections; the forbidden constructs of spec 7.4 and the Global Constraints' rules 1 to 12; `required_wires` and constructor validation with `ValueError`; pins through `register_rule`; code quality: duplication (`_dummy_wires`, `select._check`), file sizes and responsibilities, error handling, wire disjointness (rows L332, L339).
- Area 2: every class has T2 to T6 (give the same table as Opus, independently); every test states both tolerances (HANDOFF 6); tier placement against the 15-minute per-commit budget; tests named for what they check; mutants where coverage looks thin; the release tests' inputs (row L314).
- Area 3: the report script and `costs_reference` against spec 13.3 (rows, columns, reason codes, the large-run policy, the cache); the canaries against spec 14 (D-5: one canary per claim, each recording today's state); `pyproject.toml`'s markers and `tests/conftest.py`'s `release` implies `slow`; the watchdog and test runner (rows L230 to L238); the fixtures' provenance; code quality of `dfthc.testing`.
- Area 4: D-6: the documentation set of spec 5.2 exists and every claim in it matches the code; every limitation L-1 to L-27 has its entry and canary; the spec's cross-references and section numbers; the L-6 drift (row L302); HANDOFF's commands as they run today.

- [ ] **Step 4: The layout of each review** (part of every prompt by reference)

````markdown
# Final review, area <k> (<area name>), <Opus or GPT-5.6>

Reviewed tree: main at <review head>; package <package path> (<bytes> bytes).
Verdict: Ready to merge | With fixes | Not ready. Findings: Critical <n>, Important <n>, Minor <n>.

## Strengths
- <specific, with file:line>

## Findings

### <C, I or M>-<n>: <one-line title>
- Where: `<path>:<line>` (at the reviewed tree)
- Evidence: <a quoted line, or the command and its key output line>
- Fix: <the concrete change, and the test that fails before it and passes after it>
- Hashed: <yes or no>
- D criterion: <D-n or none>
- Ledger row: <its line on main at 1c15419, as in the plan's table, or none>

## Ledger rows of area <k>

| Row | Proposed | Verdict (confirm, object, or the rating of an R row) | Evidence |
| --- | --- | --- | --- |

## D-criteria evidence
<what in this area shows each criterion of the prompt's list; area 2: the table
| Class | Test module | T2 | T3 | T4 | T5 | T6 | with one test name per cell>

## Runs
- `<command>`: <result line>; WATCHDOG <seconds> s, <peak_mb> MB
````

The file must contain no Markdown link and no `](` sequence (the orchestrator copies parts of it into `docs/agents/final-review.md`, whose links `tests/test_docstrings.py` checks).

- [ ] **Step 5: Run the four review rounds, one pair at a time**

For k = 1, 2, 3, 4, in this order: dispatch the Opus and the GPT-5.6 prompt of area k in parallel, with the model passed explicitly (`inherit` and `gpt-5.6-sol-high-fast`); wait for both replies; check both files exist and hold the headings of Step 4:

```bash
cd /Users/david.wierichs/repos/dfthc-be/.superpowers/sdd/final-review
for f in area-<k>-opus.md area-<k>-gpt.md; do echo "$f: $(grep -c -E '^## (Strengths|Findings|Ledger rows of area|D-criteria evidence|Runs)' "$f") of 5 headings, $(grep -c '](' "$f") link sequences"; done
```

Expected: `area-<k>-opus.md: 5 of 5 headings, 0 link sequences` and the same for `area-<k>-gpt.md`. A missing heading or a link sequence: resume that reviewer once with the defect named. A reviewer that reports a limit it had to stop at: note it for Step 9. Then the next area.

- [ ] **Step 6: Resolve what the reviewers could not verify**

For every finding or ledger-row objection marked as unverifiable, or whose evidence is missing: read the cited code; if a run is needed, run it within the reviewers' limits (Step 3, "Allowed runs"). A finding still without evidence after one resumed request to its reviewer is declined in Step 9 with that reason.

- [ ] **Step 7: Check every Critical and Important finding yourself**

Read the cited lines at `$REVIEW_HEAD` and, where the evidence is a run, rerun it within the same limits. Classify each fix as hashed or not by `hashed-files.txt` (a docstring or comment edit is not hashed). A finding that contradicts the spec or a plan goes to the user as a plan contradiction (master plan 12.2, step 8); it blocks only its own fix. A finding that asks to change a count toward the paper's or dfthcbe's is declined (R-7; the user's directive that resource matching is a separate later step).

- [ ] **Step 8: Decide the fix list**

Apply section "Hashed and unhashed fixes" to every finding: Critical and Important `fix`; unhashed Minor `fix` unless declined with a reason; hashed Minor `fix` only if some hashed Critical or Important is `fix`, else `deferred`; the optional rows `declined` unless raised to Important, the hashed ones `fix` or `declined` at your choice when a hashed Critical or Important is `fix`. Duplicates (the same defect at the same place from two reviewers, or from two areas) become one finding with both raisers and the higher severity, unless Step 7 showed the lower one right (record why).

- [ ] **Step 9: Write `docs/agents/final-review.md`**

Number the findings C-1, C-2, ..., I-1, ..., m-1, ... in area order. Fill this format, every field of every finding, the open-rows table in the order of this plan's table (one row each, 72), and the T2 to T6 table (34 rows, from the two area-2 reviews; a disagreement between them is checked in the tests and resolved in the table):

```markdown
# Final review (Task 5.1)

Reviewed tree: local `main` at `<REVIEW_HEAD, 40 hex digits>` (`993b8fb..<REVIEW_HEAD, 7 digits>`, <number> commits), in four area packages: 1 operators, 2 operator tests, 3 data, report and tooling, 4 documents (plan `phase5-final`, Task 5.1). Each area was reviewed by Opus (`inherit`) and GPT-5.6 (`gpt-5.6-sol-high-fast`) in parallel; the eight reviews are `.superpowers/sdd/final-review/area-<1 to 4>-<opus or gpt>.md` (not committed). Merged by the orchestrator on <date>.

Code hash at review: `<code_hash(), 64 hex digits>`. A fix is hashed when it changes code (not only docstrings or comments) in a file of `dfthc.testing.provenance.hashed_files()` (`src/dfthc/**/*.py` outside `dfthc/testing/`, and `scripts/resource_report.py`) or changes `requirements-nightly.lock`.

## Summary

| Severity | Findings | Fix | Hashed fix | Declined or deferred |
| --- | ---: | ---: | ---: | ---: |
| Critical | <n> | <n> | <n> | <n> |
| Important | <n> | <n> | <n> | <n> |
| Minor | <n> | <n> | <n> | <n> |

Hashed change in the fix wave: <yes or no>. With yes, the orchestrator regenerates the small-system report and recompiles the FeMoco-54 walk and controlled walk once (the contingency pair); `fe2s2` and `fe4s4` keep their measurements as history and show their closed forms, marked pending.

## Critical

### C-1: <one-line title>

- **Where:** `<path>:<line>` at the reviewed tree.
- **Evidence:** <what was read or run: a quoted line, or the command and its key output line>.
- **Fix:** <the concrete change, and the test that shows it, RED before the fix>.
- **Hashed:** <yes or no>.
- **D criterion:** <D-n or none>. **Ledger row:** <its line on `1c15419`, as in the plan's table, or none>.
- **Raised by:** <area-k-opus, area-k-gpt>, as <the severities they gave>.
- **Decision:** <fix, or declined or deferred with the reason>.

## Important

### I-1: <one-line title>

(the same seven fields)

## Minor

### m-1: <one-line title>

(the same seven fields)

## Open ledger rows

One row per row of the plan's table "Open ledger rows and their dispositions", in its order.

| Ledger row | Task | Disposition | Reviewers | Closure, or the finding that carries it |
| --- | --- | --- | --- | --- |
| L229 | 0.1 | R | <confirmed or rated> | <m-n, or the closure text> |

## Declined and corrected findings

### <area-k-model>, <its finding id>: <title>

- **Why:** <the evidence that the finding is wrong, or the rule that declines it (R-7: no tuning toward the paper's or dfthcbe's counts; a hashed Minor finding without a hashed Critical or Important fix)>.

## T2 to T6 per operator (D-1)

From the area-2 reviews, one row per class audited by `tests/test_docstrings.py` (34).

| Class | Test module | T2 | T3 | T4 | T5 | T6 |
| --- | --- | --- | --- | --- | --- | --- |
| `<class>` | `tests/ops/<module>.py` | `<test>` | `<test>` | `<test>` | `<test>` | `<test>` |
```

Check it:

```bash
cd /Users/david.wierichs/repos/dfthc-be
grep -c '](' docs/agents/final-review.md
grep -c -E '^\| L[0-9]+ \|' docs/agents/final-review.md
grep -c -E '^\| `[A-Za-z]+` \| `tests/' docs/agents/final-review.md
grep -c -E '^- \*\*Decision:\*\* ' docs/agents/final-review.md
grep -c -E '^### (C|I|m)-[0-9]+: ' docs/agents/final-review.md
```

Expected: `0`; `72`; `34`; and the last two equal (one decision per finding).

- [ ] **Step 10: Commit the findings list and the ledger row on `main`**

Update ledger row 5.1: status `done`, branch `none`, merge commit the commit below, test evidence (the package sizes, the eight review files, the counts per severity and how many are `fix`, whether a hashed fix is in the list), reviews (the eight verdicts), open issues (the user questions waiting for Task 5.2, Step 7). Then:

```bash
cd /Users/david.wierichs/repos/dfthc-be
git add docs/agents/final-review.md docs/agents/LEDGER.md
git commit -m "Final review (Task 5.1): the consolidated findings list"
scripts/test.sh tests/test_docstrings.py -q -p no:cacheprovider
scripts/test.sh -m "not slow and not release" --collect-only -q -p no:cacheprovider | tail -1
```

Expected: `49 passed` (48 + the link check of `final-review.md`); `1272/1393 tests collected (121 deselected) in ...` (1,271 + 1). Append `Task 5.1: done (final-review.md <sha7>, eight reviews)` to `.superpowers/sdd/progress.md`.

**Definition of done (Task 5.1):** all eight area reviews are in and merged into `docs/agents/final-review.md`, committed on `main`, with a decision for every finding and a disposition for each of the 72 open ledger rows; `tests/test_docstrings.py` passes on `main` (49).

---

### Task 5.2: Fix wave, release tier and final merge

**Files:** Modify the files the findings with decision `fix` name (the fixer, on `phase5/5.2`; never `docs/resources.md`, `docs/resources.json`, `docs/agents/LEDGER.md` or `docs/agents/final-review.md`); after a hashed fix, `docs/resources.md` and `docs/resources.json` (the orchestrator, on the branch); `docs/agents/LEDGER.md` (the orchestrator, on `main`).

**Interfaces:**
- Consumes: `docs/agents/final-review.md` on `main` (Task 5.1); the code hash `5e3eed1e7a9535765299d6c90afe643f96a5155e7eb43d43e039864133fbba8e` at the branch's base; this plan's scripts `check_d3.py`, `check_d7_ledger.py` and `report_statuses.py` in `.scratch/plan-check/stages/P5/` (shown in full below).
- Produces: `phase5/5.2` merged into local `main` with `--no-ff`; the ledger closed: rows 5.1 and 5.2, the heavy runs, the D-1 to D-7 checklist, the 72 open Minor-findings rows, the Q-1 decision row, gate G5.

Throughout Task 5.2: every dispatch (the fixer, the re-reviews, and every fix dispatch of master plan 12.5.5 in Steps 6 and 10, which gets the added line "Never push, fetch, or add, change or remove remotes.") forbids push, fetch and remote changes, so D-7's process-control assertion holds; `<main>` is `/Users/david.wierichs/repos/dfthc-be`; `<lane>` is the lane of Step 1; in the lane, `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` (HANDOFF 3.3).

- [ ] **Step 1: Create the branch and measure the base**

A lane is free when `git -C <main>/.worktrees/<lane> status --short --branch` shows `## HEAD (no branch)` and nothing else; lane-a is freed by Task 4.5's merge.

```bash
git -C /Users/david.wierichs/repos/dfthc-be/.worktrees/<lane> switch -c phase5/5.2 main
BASE=$(git -C /Users/david.wierichs/repos/dfthc-be rev-parse main); echo "$BASE"
cd /Users/david.wierichs/repos/dfthc-be/.worktrees/<lane>
VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"
PYTHONPATH=src "$VENV/bin/python" -c "from dfthc.testing.provenance import code_hash; print(code_hash())"
scripts/test.sh -m "not slow and not release" --collect-only -q -p no:cacheprovider | tail -1
scripts/test.sh -m "slow and not release" --collect-only -q -p no:cacheprovider | tail -1
scripts/test.sh -m release --collect-only -q -p no:cacheprovider | tail -1
```

Expected: `Switched to a new branch 'phase5/5.2'`; the head of `main`; `5e3eed1e7a9535765299d6c90afe643f96a5155e7eb43d43e039864133fbba8e`; `1272/1393 tests collected (121 deselected) in ...`; `86/1393 tests collected (1307 deselected) in ...`; `35/1393 tests collected (1358 deselected) in ...`. If the fix list is empty, skip to Step 7.

- [ ] **Step 2: Extract the brief**

```bash
cd /Users/david.wierichs/repos/dfthc-be
/Users/david.wierichs/.cursor/plugins/cache/cursor-public/superpowers/d884ae04edebef577e82ff7c4e143debd0bbec99/skills/subagent-driven-development/scripts/task-brief docs/superpowers/plans/2026-11-11-dfthc-phase5-final.md 5.2
```

Expected: the brief's path (`.superpowers/sdd/task-5.2-brief.md`); the report path is the brief path with `-brief` replaced by `-report`. The ids of the hashed findings with decision `fix`, for the slot `{hashed_fix_ids}` (a test on a four-finding sample printed exactly its two hashed `fix` ids):

```bash
awk '/^### (C|I|m)-[0-9]+: /{id=$2; sub(":","",id)} /^- \*\*Hashed:\*\* yes/{h[id]=1} /^- \*\*Decision:\*\* fix/{if (h[id]) print id}' docs/agents/final-review.md
```

- [ ] **Step 3: Dispatch the fixer** (Tier A, `inherit`)

```text
description: "Implement Task 5.2: the final review's fix wave"
model: inherit
prompt: |
  You are the fixer of the DFTHC project's final review (master plan Task 5.2): one Tier A fixer
  with the complete findings list. The project implements Low et al., PRX 15, 041016 (2025), in
  PennyLane: the DFTHC block encoding, walk, controlled walk and iterative QPE as Operator2 blocks
  compiled with Catalyst, resources from qp.analyze, compared with the paper and dfthcbe without
  tuning.

  Read first, in this order:
  1. {repo}/docs/agents/HANDOFF.md: environment, rules, conventions, limitations, commands, git,
     self-review checklist, report format.
  2. {brief_path}: Task 5.2 of docs/superpowers/plans/2026-11-11-dfthc-phase5-final.md. Its
     Step 4 is your procedure, and that file's "Global Constraints" and "Hashed and unhashed
     fixes" bind you.
  3. {repo}/docs/agents/final-review.md: the findings list. Fix every finding whose Decision is
     "fix", in the order Critical, Important, Minor, and nothing else.

  Work only in the lane {lane} (branch phase5/5.2, created there from local main at {base_sha}).
  Never switch the lane to another branch; never run git worktree add/remove, git clone, or
  anything that writes .git/config or hooks; never push, merge, fetch or touch remotes.

  The code hash at {base_sha} is 5e3eed1e7a9535765299d6c90afe643f96a5155e7eb43d43e039864133fbba8e.
  The findings marked "Hashed: yes" that you fix: {hashed_fix_ids}. Every other fix must leave
  the code hash unchanged.

  If anything is unclear, ask now (status NEEDS_CONTEXT) before writing code.

  Stop and report BLOCKED if a fix would change T, M or A of any operator at any preset (in
  dfthc.costs or in a compiled count), would need docs/resources.md, docs/resources.json,
  docs/agents/LEDGER.md or docs/agents/final-review.md, would need a heavy run or a compile above
  the limits of Step 4, contradicts the spec or a plan, or if a compiler crash or an
  order-dependent compile appears (L-10: keep a minimal reproducer under tests/canary/reproducers/
  and stop).

  Write the full report to {report_path}; reply in at most 15 lines: Status (DONE |
  DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT), commits, one-line test summary, the code hash
  after the last commit, N_pc, N_n and N_r, concerns, the report path.
```

Slots: `{repo}` = `/Users/david.wierichs/repos/dfthc-be`; `{lane}` = `/Users/david.wierichs/repos/dfthc-be/.worktrees/<lane>`; `{base_sha}` = Step 1's `BASE`; `{brief_path}` and `{report_path}` from Step 2; `{hashed_fix_ids}` from Step 2 (`none` if empty).

- [ ] **Step 4: The fixer's procedure** (the brief carries it)

1. For each finding with decision `fix`, in the order of `docs/agents/final-review.md`:
   - If it changes behaviour (code, or what a test asserts): write the failing test its **Fix** field names; run it, `scripts/test.sh <node id> -q -p no:cacheprovider`, and record the failing line (RED); make the fix; run the finding's covering tests (GREEN).
   - If it only edits a document, a docstring or a comment: no new test; run `scripts/test.sh tests/test_docstrings.py -q -p no:cacheprovider`, expected `49 passed` plus the documentation tests the fix wave added.
   - Commit: `git add <files> && git commit -m "Final review <id>: <title>"`, one commit per finding (for example `Final review I-2: <title>`).
   - Hash check: `PYTHONPATH=src "$VENV/bin/python" -c "from dfthc.testing.provenance import code_hash; print(code_hash())"`; an unhashed finding leaves `5e3eed1e7a9535765299d6c90afe643f96a5155e7eb43d43e039864133fbba8e`; if it does not, stop (BLOCKED).
2. Targeted runs: the tests of the files you touched, one worker (`DFTHC_TEST_WORKERS=1 scripts/test.sh <paths or node ids> -q -p no:cacheprovider`), per-commit and nightly tests; never `tests/report/test_report.py::test_report_mid`, `tests/sim/test_benchmark.py`, `tests/canary/test_l21_compile_time.py` or a `release` test (the orchestrator runs them), never `scripts/resource_report.py`, and nothing FeMoco-shaped.
3. After the last commit, once each:

```bash
scripts/test.sh -m "not slow and not release" -q -p no:cacheprovider
scripts/test.sh -m "not slow and not release" --collect-only -q -p no:cacheprovider | tail -1
scripts/test.sh -m "slow and not release" --collect-only -q -p no:cacheprovider | tail -1
scripts/test.sh -m release --collect-only -q -p no:cacheprovider | tail -1
"$VENV/bin/ruff" check .
PYTHONPATH=src "$VENV/bin/python" -c "from dfthc.testing.provenance import code_hash; print(code_hash())"
```

   Expected: `<1272 + N_pc> passed in ...` with no warnings summary; `<1272 + N_pc>/<1393 + N> tests collected (121 deselected)`; `<86 + N_n>/<1393 + N> tests collected (...)`; `<35 + N_r>/<1393 + N> tests collected (...)`, N = N_pc + N_n + N_r; `All checks passed!`; the base hash if no hashed finding was fixed, else a new hash.
4. The self-review checklist (HANDOFF 9) and the report (HANDOFF 10), with a table: finding id, commit, files, hashed (yes or no), covering tests, the RED and GREEN lines.

- [ ] **Step 5: Check the fixer's result** (orchestrator)

```bash
cd /Users/david.wierichs/repos/dfthc-be/.worktrees/<lane>
git log --oneline "$BASE"..HEAD
git diff --stat "$BASE"..HEAD -- docs/resources.md docs/resources.json docs/agents/LEDGER.md docs/agents/final-review.md
PYTHONPATH=src "$VENV/bin/python" -c "from dfthc.testing.provenance import code_hash; print(code_hash())"
```

Expected: one commit per finding with decision `fix`, each `Final review <id>: <title>`; no output from the `git diff --stat` (those four files untouched); the hash as the report states, equal to the base hash exactly when `{hashed_fix_ids}` was `none`. Rerun the three collection commands of Step 4 and compare N_pc, N_n and N_r with the report.

- [ ] **Step 6: Re-review, and the fix loop**

```bash
cd /Users/david.wierichs/repos/dfthc-be
/Users/david.wierichs/.cursor/plugins/cache/cursor-public/superpowers/d884ae04edebef577e82ff7c4e143debd0bbec99/skills/subagent-driven-development/scripts/review-package "$BASE" phase5/5.2
```

Expected: `wrote /Users/david.wierichs/repos/dfthc-be/.superpowers/sdd/review-<base7>..<head7>.diff: <n> commit(s), <bytes> bytes`. Dispatch both reviews in parallel:

```text
description: "Review Task 5.2 (the final review's fix wave)"
model: gpt-5.6-sol-high-fast
prompt: |
  You are reviewing the fix wave of the DFTHC project's final review: first whether every
  finding whose Decision is "fix" in {repo}/docs/agents/final-review.md is fixed, with a covering
  test, and nothing else changed; then whether the fixes are well-built.

  What was requested: {brief_path} (Task 5.2) and the findings list.
  Global constraints that bind this task (verbatim from the plan): {global_constraints_block}
  What the fixer claims: {report_path}. Treat it as unverified.
  Diff: base {base_sha}, head {head_sha}, file {diff_path}. Read it once; it is your view of the
  change. Inspect code outside the diff only for a risk you can name, and name it.
  Read-only: do not modify the working tree, the index, HEAD or branches; never commit, push,
  fetch or touch remotes. Run tests only as Task 5.1, Step 3 of
  docs/superpowers/plans/2026-11-11-dfthc-phase5-final.md allows ("Allowed runs"), in the lane
  {lane} at {head_sha}.

  Part 1, the findings: for each finding id with Decision "fix", fixed or not (file:line); does
  its covering test fail without the fix (say how you know); is "Hashed" right (compute
  code_hash() in the lane: PYTHONPATH=src {repo}/.venv/bin/python -c "from
  dfthc.testing.provenance import code_hash; print(code_hash())"; the base's is
  5e3eed1e7a9535765299d6c90afe643f96a5155e7eb43d43e039864133fbba8e); no T, M or A changed
  (tests/test_costs.py and the T4 tests of every touched operator pass); the counts N_pc, N_n,
  N_r (collect-only).
  Part 2, quality: the fixes follow HANDOFF section 5 and the spec; tests verify behaviour and
  state both tolerances; no file grew beyond its responsibility; nothing outside the findings.

  Output, starting directly with the verdict:
  ### Findings fixed (one line per finding id: ✅ | ❌ with file:line)
  ### Spec Compliance (✅ | ❌ with file:line | ⚠️ cannot verify from diff)
  ### Issues (Critical | Important | Minor; each with file:line, why, how to fix)
  ### Assessment: Approved | Needs fixes, with a 1-2 sentence reason
```

```text
description: "Opus correctness review of Task 5.2"
model: inherit
prompt: |
  Review the correctness of the fix wave of the DFTHC project's final review (Task 5.2) beyond
  its compliance; the GPT-5.6 review covers the rest. Inputs: the findings list
  {repo}/docs/agents/final-review.md (the findings whose Decision is "fix"), brief {brief_path},
  report {report_path}, diff {diff_path} (base {base_sha}, head {head_sha}). Read-only, and never
  commit, push, fetch or touch remotes; run tests only as Task 5.1, Step 3 of
  docs/superpowers/plans/2026-11-11-dfthc-phase5-final.md allows ("Allowed runs"), in the lane
  {lane} at {head_sha}.

  Check independently, with file:line evidence:
  - each fix removes the defect its finding names, without a new one (phases and signs, register
    widths and endianness, wire aliasing, measurement semantics, resets and hints, failure
    branches, adjoint and controlled classes, pins);
  - every new test fails without its fix (mutate the fix back in a copy, as Task 5.1's mutants:
    at most 10);
  - no count moved: re-derive by hand one count that a touched operator contributes and compare
    with dfthc.costs;
  - the hashed classification of each commit, and that nothing hashed changed where the findings
    list says "Hashed: no".
  Same output format as the GPT-5.6 review. Do not pre-judge severities; report what you find.
```

Slots: `{head_sha}` = `git -C <main> rev-parse phase5/5.2`; `{diff_path}` = the package's path; `{global_constraints_block}` = this plan's section "Global Constraints" verbatim; the others as in Step 3. The reviews go to `.superpowers/sdd/task-5.2-review-1.md` (GPT-5.6) and `.superpowers/sdd/task-5.2-review-opus.md` (Opus). Critical and Important findings of either review go to a fix dispatch (master plan 12.5.5, model `inherit`, the covering tests named); then a fresh package (`review-package <previous head> phase5/5.2`) and a re-review, until both approve. Their Minor findings are fixed in the same loop if unhashed, else recorded in the ledger's Minor-findings list as deferred.

- [ ] **Step 7: Ask the user, once**

Send this message, filled, and wait for both answers:

```text
Phase 5: the final review is done and its fixes are in and re-reviewed (<k> findings fixed,
<j> declined or deferred; <hashed: "one fix changed hashed code" | "no fix changed hashed code">).
Two questions before the next step:

1. Q-1, early junk measurement: variant (b) has been the working choice since 2026-11-15 and is
   built (Task Q1b). The FeMoco-54 walk needs 1,102 qubits instead of 1,564 (the paper: 1,131),
   with T, M and A unchanged (walk 9,793 / 2,524 / 8,675; controlled walk 9,795 / 2,524 / 8,677).
   Do you confirm variant (b)? If not, Phase 5 stops: restoring the baseline needs a new task,
   report regeneration and a FeMoco-54 pair. Because the current contingency pair has not run
   when this answer is taken, that pair would be compiles 4 and 5 of about 6; the new plan must
   account for it, and any later compile beyond about six needs your approval.
2. May I run the release tier now? scripts/test.sh with 2 workers, alone, under the watchdog:
   <1393 + N> tests; projected (not a bound) at about 25 to 35 minutes on a quiet machine, up
   to about 80 under load; peak about 2 to 3 GB (the watchdog kills at 7.5 GB; limit 8 hours).
   <only after a hashed fix: Before it I recompile the FeMoco-54 walk and controlled walk once,
   the planned contingency pair (FeMoco-scale compiles 4 and 5 of about 6), projected from the
   small systems at about 3.9 and 4.4 GB and about 3.3 minutes each (Task 4.5 measured 35 s
   and 1.7 GB each).>
```

While waiting, after a hashed fix, the orchestrator may run Step 8, case H, part 1 (it is not heavy and is needed unless the user overrides Q-1). If the user overrides Q-1: stop Phase 5, record the decision in the ledger, and escalate with a plan to restore the baseline. If the user declines the release run: record it and wait.

- [ ] **Step 8: Keep the report current**

Find the case:

```bash
cd /Users/david.wierichs/repos/dfthc-be/.worktrees/<lane>
PYTHONPATH=src "$VENV/bin/python" -c "from dfthc.testing.provenance import code_hash; print(code_hash())"
```

The status script (`stages/P5/report_statuses.py`), used in both cases:

```python
"""The row statuses of docs/resources.json per system, and whether its code hash is the tree's
(the Phase 5 plan, Task 5.2 Step 8). Reads files only. Run from the root of the tree it checks,
with PYTHONPATH=src.
"""

import json
from collections import Counter
from pathlib import Path

from dfthc.testing.provenance import code_hash

record = json.loads(Path("docs/resources.json").read_text())
current = code_hash()
print(f"report hash {record['code_hash'][:12]}, tree hash {current[:12]}, "
      f"equal {record['code_hash'] == current}")
for name, system in record["systems"].items():
    print(f"{name}: {dict(sorted(Counter(row['status'] for row in system['rows']).items()))}")
entries = [e for es in record["compiled"].values() for e in es]
print(f"entries: {sum(e['code_hash'] == current for e in entries)} of the tree's hash, "
      f"{sum(e['code_hash'] != current for e in entries)} history")
```

**Case U** (the hash is `5e3eed1e7a9535765299d6c90afe643f96a5155e7eb43d43e039864133fbba8e`): nothing is rerun. Check:

```bash
PYTHONPATH=src "$VENV/bin/python" /Users/david.wierichs/repos/dfthc-be/.scratch/plan-check/stages/P5/report_statuses.py
scripts/test.sh tests/report/test_femoco_cache.py -q -p no:cacheprovider
```

Expected (as on the replica, `stages/P5/report-statuses.log`):

```text
report hash 5e3eed1e7a95, tree hash 5e3eed1e7a95, equal True
h2_sto3g_R2_B2_C1: {'measured': 46}
h2_631g_R2_B2_C2: {'measured': 46}
h2_631g_R2_B3_C3: {'measured': 46}
naphthalene: {'measured': 44}
fe2s2: {'measured': 44}
fe4s4: {'measured': 44}
femoco54: {'closed form': 45, 'measured': 2}
entries: 272 of the tree's hash, 282 history
```

and `2 passed in ...`.

**Case H** (another hash), by the orchestrator, alone, with no implementer active:

Part 1, the small-system report:

```bash
mkdir -p /Users/david.wierichs/repos/dfthc-be/.scratch/g5
PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --cap-mb 3000 --timeout-min 40 -- \
    "$VENV/bin/python" scripts/resource_report.py \
    > /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-report-small.log 2>&1; echo "exit $?"
grep -c ': compiled (' /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-report-small.log
grep -c -E 'CHECK FAILED|: (killed|failed) ' /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-report-small.log
tail -1 /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-report-small.log
PYTHONPATH=src "$VENV/bin/python" /Users/david.wierichs/repos/dfthc-be/.scratch/plan-check/stages/P5/report_statuses.py
```

Expected: `exit 0`; `182` (46 + 46 + 46 + 44 programs); `0`; a `WATCHDOG` line with `"killed": null` and about 660 to 830 s at about 0.8 GB (Task Q1b.2's run: 657.5 s, 772.1 MB; Task 4.3's: 823 s, 774 MB); and:

```text
report hash <new12>, tree hash <new12>, equal True
h2_sto3g_R2_B2_C1: {'measured': 46}
h2_631g_R2_B2_C2: {'measured': 46}
h2_631g_R2_B3_C3: {'measured': 46}
naphthalene: {'measured': 44}
fe2s2: {'pending': 44}
fe4s4: {'pending': 44}
femoco54: {'closed form': 45, 'pending': 2}
entries: 182 of the tree's hash, 554 history
```

A `CHECK FAILED` line means a measured count differs from its closed form: stop, it contradicts the fix wave's rule; fix or revert on the branch (a fix dispatch), then rerun part 1.

Part 2, the contingency pair (a heavy run: FeMoco-shaped), after the user's go-ahead (Step 7):

```bash
PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 150 -- \
    "$VENV/bin/python" scripts/resource_report.py --large femoco54 \
    > /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-large-femoco54.log 2>&1; echo "exit $?"
grep -E '^femoco54/' /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-large-femoco54.log
grep -v '^WATCHDOG' /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-large-femoco54.log | grep -c -E 'CHECK FAILED|: (refused|killed|failed|skipped) '
tail -1 /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-large-femoco54.log
PYTHONPATH=src "$VENV/bin/python" /Users/david.wierichs/repos/dfthc-be/.scratch/plan-check/stages/P5/report_statuses.py
PYTHONPATH=src "$VENV/bin/python" /Users/david.wierichs/repos/dfthc-be/.scratch/plan-check/stages/P5/check_d3.py
scripts/test.sh tests/report/test_femoco_cache.py -q -p no:cacheprovider
```

Expected: `exit 0`; `femoco54/DFTHCWalk: compiled (<s> s, <MB> MB)` then `femoco54/ControlledDFTHCWalk: compiled (<s> s, <MB> MB)`, each about 35 to 200 s below 4.4 GB (Task 4.5: 36.5 s at 1,692.1 MB and 38.0 s at 1,675.1 MB; the projection from the small systems alone: 198.2 s at 3,892.5 MB and 195.9 s at 4,399.9 MB); `0`; a `WATCHDOG` line with `"killed": null`; the statuses as after part 1 except `femoco54: {'closed form': 45, 'measured': 2}` and `entries: 184 of the tree's hash, 554 history`; `check_d3.py` as in Step 11, D-3; `2 passed in ...`. Add two heavy-run rows to the ledger (the command, wall time, peak, killed no, "Large compile" 1 each: "Phase 5 contingency pair 1 of 2 (4 of about 6 overall)" and "2 of 2 (5 of about 6 overall)"). If the walk is refused, killed or fails, the script records the controlled walk as skipped and compiles nothing more: record the run, keep the branch, and escalate to the user (master plan section 2, "Where compiles above the cap run"); Phase 5 waits.

Part 3, commit the report on the branch:

```bash
git add docs/resources.md docs/resources.json
git commit -m "Report: regenerated after the final review's fixes; the FeMoco-54 walk and controlled walk recompiled (contingency pair)"
```

- [ ] **Step 9: Rebase onto `main`**

From here to Step 12 the orchestrator commits nothing on `main`; the rows of Steps 10 to 12 go into one ledger commit in Step 13.

```bash
PRE=$(git -C /Users/david.wierichs/repos/dfthc-be/.worktrees/<lane> rev-parse HEAD)
git -C /Users/david.wierichs/repos/dfthc-be/.worktrees/<lane> rebase main
git -C /Users/david.wierichs/repos/dfthc-be/.worktrees/<lane> diff --stat "$PRE" HEAD -- . ':(exclude)docs/agents/LEDGER.md'
```

Expected: `Successfully rebased and updated refs/heads/phase5/5.2.` or `Current branch phase5/5.2 is up to date.`; no output from the `diff --stat` (only the ledger came from `main`).

- [ ] **Step 10: Run the release tier** (heavy: once, alone, under the watchdog, after the user's go-ahead)

```bash
cd /Users/david.wierichs/repos/dfthc-be/.worktrees/<lane>
mkdir -p /Users/david.wierichs/repos/dfthc-be/.scratch/g5
PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 480 -- \
    scripts/test.sh -q -rA --durations=60 -p no:cacheprovider \
    > /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-release-tier.log 2>&1; echo "exit $?"
LOG=/Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-release-tier.log
grep -E '^([0-9]+ failed, )?[0-9]+ passed' "$LOG"
grep -c -E '^(FAILED|ERROR) ' "$LOG"
grep -c 'warnings summary' "$LOG"
tail -1 "$LOG"
```

Expected: `exit 0`; `<1393 + N> passed in <s>s (<h:mm:ss>)`, projected at about 1,500 to 2,000 s on a quiet machine (section "The release tier estimate"; a projection, not a bound); `0`; `0`; a `WATCHDOG` line with `"returncode": 0`, `"killed": null`, `"survivors": []` and a peak of about 2,000 to 3,000 MB. `TMPDIR` stays unset (`tests/test_infrastructure.py` creates a directory named `venv` under it, which the sandbox refuses inside the workspace). The 8 GB cap is the watchdog's default (`--cap-mb` 8192, killed at 7.5 GB). Note the wall time, the peak and the 60 slowest tests for the ledger.

On a failure (any `FAILED` or `ERROR` line, or a non-zero exit):
- **A known load-sensitive test** (`tests/canary/test_l21_compile_time.py::test_compile_time_scaling`, ledger row L-21; a test of `tests/test_watchdog.py`, rows L237 and L245): rerun it alone twice, serially, each under the watchdog: `DFTHC_TEST_WORKERS=0 PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 30 -- scripts/test.sh <node id> -q -p no:cacheprovider > /Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-rerun-<k>.log 2>&1` for k = 1, 2, each expected `1 passed`. If every load-sensitive failure passes twice alone, the release result is accepted with a documented load-only exception: Step 11 checks that no D-1 to D-4 test failed, records the original failed node and both serial passes under D-5 (for the canary) or in row 5.2's evidence (for a watchdog test), and does not claim an all-passed summary; the G5 row quotes the summary as printed (`<f> failed, <1393 + N − f> passed`) and names each exception. The exit status is then non-zero by pytest's rule, and that is expected. Record the exception in row 5.2 and in a Minor-findings row, as for L-21 at G1. If a serial rerun fails, treat the test as below; a failure of the `ESCAPE` test turns row L245 into an R row to fix.
- **Any other failing test** is a defect, rated Critical (a release check gives a wrong result): a fix dispatch (master plan 12.5.5, `inherit`, the failing test named) on `phase5/5.2`, Step 6's re-review, the failing test alone and its file, then Step 8 if the fix is hashed. The final tree needs a green release run, so ask the user for one rerun of the release tier (the same projection). If the fix is hashed after the contingency pair has run, it is a second hashed wave: its FeMoco-54 pair would be compiles 6 and 7, past about six, so ask for that approval in the same message.
- **The watchdog killed the run or it timed out:** stop, no rerun; record the heavy run (killed yes) and escalate to the user with the log's last lines and the peak.
- **`test_femoco_cache` fails naming the recompile:** a hashed change came in after Step 8; find it (`git diff <the commit of Step 8 part 3 or BASE>..HEAD -- $(cat /Users/david.wierichs/repos/dfthc-be/.superpowers/sdd/final-review/hashed-files.txt) requirements-nightly.lock`), and escalate before any further compile.

- [ ] **Step 11: Check D-1 to D-7, with evidence** (in the lane at the tested head; `LOG` as in Step 10)

```bash
cd /Users/david.wierichs/repos/dfthc-be/.worktrees/<lane>
LOG=/Users/david.wierichs/repos/dfthc-be/.scratch/g5/5.2-release-tier.log
grep -c '^PASSED tests/test_docstrings.py::test_docstring_has_the_nine_sections_in_order' "$LOG"
grep -c '^PASSED tests/ops/test_certificates_' "$LOG"
for e in test_e1_block_encoding test_e2_walk_powers test_e3_controlled_walk test_e4_prepare_then_unprepare; do grep -c "^PASSED tests/integration/test_end_to_end.py::$e\[" "$LOG"; done
grep -c -E '^PASSED tests/(test_compile.py::test_mcm_parities_capture_core_is_linear|ops/test_qroam_erase.py::test_mcm_parities_(capture_core|tape_mode)_is_linear|ops/test_angle_erase.py::test_no_mask_is_zero_and_parities_are_linear)' "$LOG"
grep -c '^PASSED tests/report/test_femoco_cache.py::' "$LOG"
PYTHONPATH=src "$VENV/bin/python" /Users/david.wierichs/repos/dfthc-be/.scratch/plan-check/stages/P5/check_d3.py
grep -c '^PASSED tests/integration/test_qpe_toy.py::' "$LOG"
grep -c '^PASSED tests/integration/test_qpe_counts.py::' "$LOG"
grep -c '^PASSED tests/integration/test_qpe_walk.py::' "$LOG"
grep -c '^PASSED tests/canary/' "$LOG"
grep -c '^PASSED tests/test_docstrings.py::test_limitations_match_the_canaries_and_spec_14$' "$LOG"
grep -c '^PASSED tests/test_docstrings.py::' "$LOG"
grep -E '^(FAILED|ERROR) ' "$LOG"
```

Expected, each plus the tests the fix wave added to that file (from the fixer's report): `34`; `30`; `33`, `4`, `3` and `2` for E-1 to E-4 (42 in all, the cross-check); `4`; `2`; the D-3 output below; `23`; `8`; `6`; `41`, or `40` with an accepted load-only exception of the L-21 canary (Step 10); `1`; `49`; and no `FAILED` or `ERROR` line, or only the nodes Step 10 accepted as load-only exceptions, none of them a D-1 to D-4 test (the counts above are then exact). The D-3 script (`stages/P5/check_d3.py`) and its output on the replica of `c443c51` (`check-d3.log`), unchanged after the contingency pair because no fix may move a count:

```python
"""D-3 evidence (the Phase 5 plan, Task 5.2 Step 11): the committed report's code hash is the
tree's, and its FeMoco-54 walk and controlled walk are measured, with their deltas against the
paper and dfthcbe and the reason codes that explain them. Reads files only. Run from the root of
the tree it checks (a lane, or a replica), with PYTHONPATH=src, under the watchdog.
"""

import json
from pathlib import Path

from dfthc.testing.provenance import code_hash

record = json.loads(Path("docs/resources.json").read_text())
rows = {row["id"]: row for row in record["systems"]["femoco54"]["rows"]}
print("hash current:", record["code_hash"] == code_hash())
for name in ("DFTHCWalk", "ControlledDFTHCWalk"):
    row = rows[name]
    measured = row["measured"] or {}
    explained = (sum(row["codes_paper"].values()) == row["delta_paper"]
                 and sum(row["codes_dfthcbe"].values()) == row["delta_dfthcbe"])
    print(name, row["status"], measured.get("toffoli"), measured.get("measurements"),
          measured.get("measured_and_uncomputes"), "delta paper", row["delta_paper"],
          "delta dfthcbe", row["delta_dfthcbe"], "explained", explained)
```

```text
hash current: True
DFTHCWalk measured 9793 2524 8675 delta paper -204 delta dfthcbe 152 explained True
ControlledDFTHCWalk measured 9795 2524 8677 delta paper -202 delta dfthcbe 153 explained True
```

The checklist rows (the ledger section "Definition of done"; each cell "Evidence" holds the command's short form and its output, each "Checked" the date):

| D | Evidence | Expected |
| --- | --- | --- |
| D-1 | the docstring audits in the release log; no `FAILED` or `ERROR` line other than an accepted load-only exception of Step 10, and none in a D-1 to D-4 test; the T2 to T6 table of `docs/agents/final-review.md` with a test in every cell | 34; none, or only the exceptions; 34 complete rows |
| D-2 | the certificates; E-1, E-2, E-3 and E-4 counted separately (nightly and release schedules), with their sum as a cross-check; the four linearity tests; all in the release log | 30; 33, 4, 3, 2 (42); 4 |
| D-3 | `test_femoco_cache` and the per-commit stale-entry test in the release log; `check_d3.py`; `docs/resources.md` holds the FeMoco-54 summary with the deltas and reason codes (the report exits 0 only if every delta's codes sum to it) | 2; the three lines above |
| D-4 | the toy QPE, the compiled QPE counts at the test presets, and the real-walk QPE in the release log | 23; 8; 6 |
| D-5 | the canaries and the limitations check in the release log, and an accepted load-only canary failure, if any, with its node, its release-log line and its two serial passes (Step 10); no canary flipped in Phase 5 (no new row in "Canary state changes") | 41, or 40 and the exception with `1 passed` twice; 1 |
| D-6 | `tests/test_docstrings.py` in the release log; no Critical or Important finding of area 4 left unfixed | 49 |
| D-7 | Step 12's commands after the merge: the merge commit; the remote-configuration and remote-tracking-ref snapshots of Task 5.1, Step 1 and after the merge, compared with `cmp`; no unmerged branch; every Phase 5 prompt forbade push, fetch and remote changes. A process-control assertion: without contacting the remote, it is no forensic proof that no push occurred | no output from either `cmp` and from `git branch --no-merged main` |

- [ ] **Step 12: Merge into local `main`; never push**

```bash
cd /Users/david.wierichs/repos/dfthc-be
git log --oneline -1 main
git merge --no-ff phase5/5.2 -m "Merge phase5/5.2: Fix wave, release tier, final merge"
git -C .worktrees/<lane> switch --detach main
git branch --no-merged main
git config --local --get-regexp '^(remote\.|branch\..*\.(remote|merge))' > .superpowers/sdd/final-review/git-remote-config.after
git for-each-ref --format='%(refname) %(objectname)' refs/remotes > .superpowers/sdd/final-review/remote-refs.after
cmp .superpowers/sdd/final-review/git-remote-config.before .superpowers/sdd/final-review/git-remote-config.after
cmp .superpowers/sdd/final-review/remote-refs.before .superpowers/sdd/final-review/remote-refs.after
git status --short --branch
```

Expected: `main`'s head is the one Step 9 rebased onto (nothing was committed on `main` since); the merge succeeds (`Merge made by the 'ort' strategy.`); the lane is detached at `main`; no output from `git branch --no-merged main` and from either `cmp` (the local remote configuration and the remote-tracking refs are as Task 5.1, Step 1 recorded them: the four configuration lines and no remote-tracking ref); `## main...origin/main [gone]`. Never run `git push`, `git fetch` or any `git remote` command. What this shows is a process control, stated as such in D-7: every Phase 5 prompt forbade push, fetch and remote changes, and the local evidence is unchanged; without contacting the remote it is not a forensic proof that no push occurred (a push need not leave a local trace, and a configuration could be changed and restored).

- [ ] **Step 13: Close the ledger, on `main`**

In one commit on `main`:
- Row 5.2: status `merged`, branch `phase5/5.2`, the merge commit, test evidence (the fixer's tier, the case and its runs, the release tier's result line, wall time and peak), reviews (the GPT-5.6 and Opus verdicts and files); row 5.1 as Task 5.1, Step 10 left it.
- "Heavy runs": the release tier (command, wall time, peak, killed no, "Large compile" 0); in case H the two rows of Step 8, if not yet added.
- "Definition of done": the seven rows of Step 11, with D-7's evidence: the merge commit; `git branch --no-merged main` and both `cmp` commands of Step 12 with no output; `check_d7_ledger.py`'s line below; and the sentence "a process-control assertion: every Phase 5 prompt forbade push, fetch and remote changes, and the local remote configuration and remote-tracking refs are unchanged; without contacting the remote it is no forensic proof that no push occurred".
- "Minor findings (input to Task 5.1)": each of the 72 rows of this plan's table gets its final status, from `docs/agents/final-review.md`'s open-rows table: R fixed, "fixed in Task 5.2 (`<commit>`; final review `<id>`)"; R declined or deferred, "declined (final review `<id>`: `<reason>`)" or "deferred: hashed Minor without a hashed Critical or Important fix (final review `<id>`)"; C, "closed: `<the closure text>` (confirmed in Task 5.1)"; D, "deferred (resource matching is a separate later step, by the user's directive; `docs/resources.md`, Deferred refinements)"; A, "accepted (confirmed in Task 5.1)"; W, "closed: passed in every Phase 5 run" or its fix; P, "closed: followed through Phase 5"; T, "closed: release tier `<wall time>`, `<peak>` (Task 5.2)". New findings of the re-review that were deferred get rows of their own.
- "Decision log": a row "Q-1 confirmed: variant (b)" (date, the user's answer, by user); a row for the release go-ahead.
- "Gate log": the G5 row:

```text
| G5 | <date> | yes: Task 5.1 done (`docs/agents/final-review.md` <sha7>: <c> Critical, <i> Important, <m> Minor; <k> fixed, <j> declined or deferred) and Task 5.2 merged (<merge sha7>); release tier on the merged code (<s> s, <MB> MB, alone, under the watchdog): <either `<1393 + N> passed`, or `<f> failed, <1393 + N − f> passed` with: <f> known load-sensitive test(s) (<node ids>) failed under two-worker load and passed twice serially (`5.2-rerun-*.log`)>; D-1 to D-7 checked with evidence (section "Definition of done"; D-7 as a process-control assertion); <case U: the code hash unchanged, the report as committed | case H: the small-system report regenerated, the FeMoco-54 pair recompiled (the contingency pair), `fe2s2` and `fe4s4` pending>; ledger complete (`check_d7_ledger.py`: 57 task rows, 0 not closed) | <none, or the escalations> | n/a (the last gate) | yes (<date>) |
```

Then:

```bash
cd /Users/david.wierichs/repos/dfthc-be
.venv/bin/python .scratch/plan-check/stages/P5/check_d7_ledger.py
git add docs/agents/LEDGER.md
git commit -m "Ledger: Phase 5 complete, gate G5"
scripts/test.sh tests/test_docstrings.py -q -p no:cacheprovider
```

Expected: `57 task rows, 0 not closed` (on `1c15419` and `99c5ffb` the script prints `5.1: status not started`, `5.2: status not started` and `57 task rows, 2 not closed`, `stages/P5/main-check-d7-ledger.log`; on `a221d97` also `4.5: status in review` and `3 not closed`, `check-d7-ledger.log`); the commit; `49 passed` plus the documentation tests the fix wave added (the merged tree differs from the release-tested one only in `docs/agents/LEDGER.md`, so the release result carries, ledger row L301's procedure). Append `Task 5.2: complete (commits <base7>..<head7>, review clean)` to `.superpowers/sdd/progress.md`.

The script (`stages/P5/check_d7_ledger.py`):

```python
"""D-7 evidence from the ledger (the Phase 5 plan, Task 5.2 Step 11): every task row of
docs/agents/LEDGER.md that is not `merged`, `done` or `not needed`, and every `merged` row without
a merge commit or a review verdict. Prints one line per such row and a summary line. Run from the
root of a checkout: <venv>/bin/python <this file> [path to LEDGER.md].
"""

import re
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/agents/LEDGER.md")
text = path.read_text().split("## Spike results")[0]
rows = [line for line in text.splitlines() if re.match(r"^\| [0-9QG][^|]* \| ", line)
        and not line.startswith("| Id ")]
problems = 0
for line in rows:
    cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
    task, status, merge, reviews = cells[0], cells[3], cells[5], cells[7]
    if status not in ("merged", "done", "not needed"):
        print(f"{task}: status {status}")
        problems += 1
    elif status == "merged" and (not merge or reviews.strip(" /") == ""):
        print(f"{task}: merged without a merge commit or a review verdict")
        problems += 1
print(f"{len(rows)} task rows, {problems} not closed")
```

- [ ] **Step 14: Tell the user**

A short G5 summary: the review (findings per severity, fixed, declined or deferred), the case (U or H, and the recompile's numbers), the release tier's result, wall time and peak, the D-1 to D-7 evidence in one line each, the final merge commit, and that nothing was pushed.

**Definition of done (Task 5.2):** every finding with decision `fix` fixed and re-reviewed (both reviews approve); the report current under the merged code hash (case U, or case H with the contingency pair); the release tier green on the merged code; every D criterion checked with evidence in the ledger; `phase5/5.2` merged into local `main` with `--no-ff`; the ledger closed (G5); nothing pushed.

---

## After Phase 5

The project's plan ends here. The deferred items stay recorded where the user will find them: the resource refinements (truncated or pruned unary iteration, the kept swap ANDs, the smaller SF fix-up one-hot) in `docs/resources.md` under "Deferred refinements", each a separate later step by the user's directive; the declined and deferred findings in `docs/agents/final-review.md` and the ledger; the upstream drafts in `docs/upstream/`, which agents never file.
