# Ledger

The durable record of the implementation. The orchestrator maintains it on local `main` and
commits it after every merge, spike and gate (master plan, section 12.8). After any loss of
context, trust this file and `git log` over memory: a task marked `merged` is never re-dispatched.

## Task status

Row format: `Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues`.

- **Status values:** `not started`, `in progress`, `in review`, `fixes`, `merged`, `blocked`,
  `cancelled`, `not needed`. A conditional task starts as `not started`; its condition is in
  "Open issues", and it is set to `not needed` once the condition is ruled out.
- **Test evidence:** the command and the key output line, for example `scripts/test.sh tests/x.py: 23 passed`.
- **Reviews:** the verdict and the review file, for example `Approved (review-ab12..cd34)`; Opus is `n/a` for Tier B.
- **Open issues:** ordering constraints, conditions and pointers to the Minor-findings list.

Every task of the master plan has a row from the start (44 planned, 2 conditional), so a task
marked `merged` is never re-dispatched after a loss of context.

### Phase 0: foundation and feasibility

| Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | Scaffold, environment, lock, test runner | B | not started | `phase0/0.1` | | | / n/a | |
| 0.2 | `dfthc.compile`, resource assertions, canary runner | A | not started | `phase0/0.2` | | | / | |
| 0.3 | Canary suite A (L-1 to L-14, L-19, L-23 to L-25) | B | not started | `phase0/0.3` | | | / n/a | needs the catalyst#3431 reproducer (scout) |
| 0.4 | Canary suite B (L-15 to L-18, L-20 to L-22), spec section 14 | B | not started | `phase0/0.4` | | | / n/a | after 0.3 (shared `docs/limitations.md`); spec edit approved by the orchestrator at merge |
| 0.5 | Vendored dfthcbe data and license | B | not started | `phase0/0.5` | | | / n/a | |
| 0.6 | Spike K-7: comparator and adder adjoints | B | not started | `phase0/0.6` | | | / n/a | |
| 0.7 | Spike K-3: mode-agnostic measured erase; prototypes | A | not started | `phase0/0.7` | | | / | |
| 0.8 | Spike K-10: measurement arrays as dynamic arguments | A | not started | `phase0/0.8` | | | / | |
| 0.9 | Spike K-1: small and mid compiles, FeMoco projection | A | not started | `phase0/0.9` | | | / | the FeMoco driver runs only if the projection allows it, started by the orchestrator, alone (heavy run) |
| 0.9b | (conditional) K-1 remedy 1 in the stand-in | A | not started | `phase0/0.9b` | | | / | conditional: opened at G0 only if 0.9's projection or FeMoco run exceeds 8 GB or 60 min; otherwise set to `not needed` |

### Phase 1: parallel workstreams

| Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1A.1 | `WalkParams`, presets and `WalkLayout` | B | not started | `phase1/1A.1` | | | / n/a | |
| 1A.2 | `DFTHCFactors`, loaders, `random_factors`; fermion matrices | B | not started | `phase1/1A.2` | | | / n/a | |
| 1A.3 | Sum-of-squares quantities, both offset frames, spec 8.6 checks 1 to 3 | A | not started | `phase1/1A.3` | | | / | |
| 1A.4 | Alias tables, Givens angles, `WalkTables` | B | not started | `phase1/1A.4` | | | / n/a | |
| 1A.5 | Encoded-Hamiltonian model and failure signs | A | not started | `phase1/1A.5` | | | / | |
| 1A.6 | dfthcbe reference fixtures (`.venv-ref`) | B | not started | `phase1/1A.6` | | | / n/a | |
| 1A.7 | `dfthc.costs`: closed forms of spec 9.24 | B | not started | `phase1/1A.7` | | | / n/a | |
| 1B.1 | Sparse simulator core and differential tests | A | not started | `phase1/1B.1` | | | / | |
| 1B.2 | Measurements, conditionals, policies, tape front end | A | not started | `phase1/1B.2` | | | / | |
| 1B.3 | Simulator benchmark gate | B | not started | `phase1/1B.3` | | | / n/a | re-dispatch to Tier A (with an Opus review) if a spec 12.6 mitigation is needed |
| 1C.1 | Unary iteration helper and `PhaseGradientRotation` | A | not started | `phase1/1C.1` | | | / | |
| 1C.2 | `FlaggedUniformPrep` | A | not started | `phase1/1C.2` | | | / | |
| 1C.3 | `QROAMRead` | A | not started | `phase1/1C.3` | | | / | |
| 1C.4 | `PhaseFixup` and `emit_phase_fixup` | A | not started | `phase1/1C.4` | | | / | after 1C.3 (same module `qroam.py`) |
| 1C.5 | `QROAMErase` and round-trip certificates | A | not started | `phase1/1C.5` | | | / | after 1C.4 (same module `qroam.py`) |
| 1C.6 | `ZeroReflection` | B | not started | `phase1/1C.6` | | | / n/a | after 1C.1 (which creates `dfthc/ops/__init__.py`) |

### Phase 2: paper blocks

| Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2D.1 | `OuterPrepare` and its pinned adjoint | A | not started | `phase2/2D.1` | | | / | |
| 2D.2 | `InnerPrepare`, its pinned adjoint, `InnerSigns` | A | not started | `phase2/2D.2` | | | / | after 2D.1 (same module `prepare.py`) |
| 2E.1 | `TailAngleLoad`, `SFAngleLoad` | B | not started | `phase2/2E.1` | | | / n/a | |
| 2E.2 | `AngleErase` | A | not started | `phase2/2E.2` | | | / | after 2E.1 (same module `angle_data.py`) |
| 2E.3 | `SpinMux`, `SpinSwap` | B | not started | `phase2/2E.3` | | | / n/a | |
| 2E.4 | `GivensRotation`, `GivensLadder` | A | not started | `phase2/2E.4` | | | / | |
| 2E.5 | `MajoranaGate` | A | not started | `phase2/2E.5` | | | / | |

### Phase 3: assembly and end-to-end checks

| Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.1 | `InnerSelect` and its pinned adjoint | A | not started | `phase3/3.1` | | | / | |
| 3.2 | `RefT2`, `RefAB` and controlled versions | B | not started | `phase3/3.2` | | | / n/a | |
| 3.3 | `DFTHCBlockEncoding` and `C(DFTHCBlockEncoding)` | A | not started | `phase3/3.3` | | | / | |
| 3.4 | `DFTHCWalk`, `C(DFTHCWalk)`, whole-program counts at the test and mid presets | A | not started | `phase3/3.4` | | | / | |
| 3.5 | End-to-end checks E-1 to E-4 and tier budgets | A | not started | `phase3/3.5` | | | / | |

### Phase 4: application, report and documentation

| Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1 | QPE: `ControlledWalkPower`, `QPERound`, driver, energies, `ToyWalk` | A | not started | `phase4/4.1` | | | / | |
| 4.2 | QPE resource counts (compiled at the test presets) | B | not started | `phase4/4.2` | | | / n/a | |
| 4.3 | Resource report and `dfthc.costs_reference` | B | not started | `phase4/4.3` | | | / n/a | the `fe2s2` and `fe4s4` compiles are heavy runs by the orchestrator; programs projected above 8 GB follow the G0 decision |
| 4.4 | Documentation set and docstring audit | B | not started | `phase4/4.4` | | | / n/a | |
| Q1b | (conditional) Q-1 variant (b): unprepare-from-outcomes operators | A | not started | `phase4/Q1b` | | | / | conditional: opened only if the user chooses Q-1 variant (b) after 4.4; regenerates the report and documentation before Task 4.5; otherwise set to `not needed` |
| 4.5 | The single FeMoco walk and controlled-walk compile; report finalization | B | not started | `phase4/4.5` | | | / n/a | after 4.4, the Q-1 decision and Q1b if chosen; the compile is two of the six FeMoco-scale compiles, run by the orchestrator where the G0 decision allows |

### Phase 5: final review

| Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1 | Whole-branch final review | A | not started | none | | | / | read-only: no branch, no merge |
| 5.2 | Fix wave, release tier, final merge | A | not started | `phase5/5.2` | | | / | the release tier runs once, after the user's go-ahead |

## Spike results

| Spike | Task | Verdict | Measurements | Consequence |
| --- | --- | --- | --- | --- |
| T6 verbose listing | 0.2 | not run | plan dry-run 2026-11-11: capturable only with `qp.qjit(verbose=True)`; lines `Chosen Rule: __builtin_<rule>` | T6 tests may assert rule names |
| K-7 comparator adjoints | 0.6 | not run | plan dry-run: 6/6 passed | |
| K-3 mode-agnostic erase | 0.7 | not run | plan dry-run: pass; T = 9, M = 4; tape rules valid; tape path exact | |
| K-10 measurement arrays | 0.8 | not run | plan dry-run: pass at small size. FeMoco-shape compiles while writing the plan: 199 to 274 s (packed parities) and 1,517 s (masks as compilable data), other compiles running, peak memory not recorded; quoted from the spikes plan at commit `994f7aa`, not reproduced (large-compile policy). Round-3 calibration at 544 and 1,088 addresses: 13.2 / 11.6 s and 30.9 / 23.2 s, peaks 1.1 and 1.75 GB | |
| K-1 whole program | 0.9 | not run | plan dry-runs: small shape walk T 236 (9 s, 0.5 GB footprint, 0.6 GB getrusage sum), controlled walk T 238 (17 s, 0.8 GB). Round-3 calibration (sampled resident size): `mid10` walk 60 s, 1.6 GB, controlled walk 124 s, 2.3 GB; `mid14` walk 128 to 146 s, 1.7 to 2.3 GB. Projection by the K-1 method (gates from the resource functions: FeMoco walk 627,557): FeMoco walk 31 to 57 min, 12.5 to 17.6 GB sampled peak (upper bound 21 GB); controlled walk about 2 h, 17 GB (upper bound 28 GB; its `mid14` point estimated). Same method: `fe2s2` walks 4 to 6 GB, `fe4s4` walks 6 to 9 GB. FeMoco not compiled | the projection exceeds the 8 GB cap: expect Task 0.9 to escalate at G0 (decision row "K-1 under the load policy") |

Dry-run values are indicative. The task runs are authoritative and replace them.

## Decision log

| Date | Item | Decision | By |
| --- | --- | --- | --- |
| 2026-11-11 | Design spec | approved and merged into `main` (`5fb077c`) | user |
| 2026-11-11 | Planning | rolling-wave: master plan plus Phase 0 code-level plans; later phases at their gates | user |
| 2026-11-11 | Q-1 early junk measurement | open; decided after K-10 and the Phase 4 report; prefer variant (b) if it works | user |
| 2026-11-11 | Q-2 reference fixtures | `.venv-ref` with Qualtran and dfthcbe approved | user |
| 2026-11-11 | Q-3 QPE driver | plain quantum function, exempt from R-4; `QPERound`, `ControlledWalkPower` stay `Operator2` | user |
| 2026-11-11 | Q-4 unary iteration | exact now; truncated iteration and pruning are later optimizations | user |
| 2026-11-11 | Q-5 Givens rotations | dfthcbe's exact construction, 2(b_rot − 1) Toffolis per rung | user |
| 2026-11-11 | Models | Tier A `claude-opus-5-5-high-fast`; Tier B `claude-sonnet-5-5-high`; review gate `gpt-5.6-sol-high-fast`; scout `grok-4.7-high-fast` (superseded by "Model switch" below) | user |
| 2026-11-11 | Heavy-process and large-compile policy | "Keep large-scale compilations to a minimum"; "Try to not run heavy processes throughout. Delay input from my side if it is easily doable." No process tree above 8 GB on the 10-core, 16 GB laptop; heavy runs only by the orchestrator, one at a time, under the watchdog, each logged in "Heavy runs"; `scripts/test.sh` with 2 workers at lowered priority; targeted tests during TDD, the per-commit tier at commit time, the nightly tier at gates only, the release tier once after the user's go-ahead; at most two implementers at once; about six FeMoco-scale compiles in total; none in the per-commit or nightly tier (master plan 12.9; spec 15) | user |
| 2026-11-11 | Model switch | Tier A implementers and the Opus reviewer use `inherit`, the orchestrator's own model, passed explicitly in every dispatch, because `claude-opus-5-5-high-fast` is no longer offered; Tier B, the review gate and the scout are unchanged | user |
| 2026-11-11 | L-21 wording (plan review round 2) | accepted: identical tables in one program share their rule compiles, while each distinct table compiles separately. Task 0.4 carries the matching spec section 14 edits for L-21 (item 7) and L-5 (item 8) | orchestrator |
| 2026-11-11 | Heavy-process definition | by load, not by wall time alone: process-tree peak above 4 GB, more than 2 busy cores, any FeMoco-shaped compile, or more than 30 min. A per-commit run with 2 workers is not heavy and keeps its 15-min budget; measured on the Phase 0 replica (164 tests): 117 s wall, 175 s CPU, about 1.5 busy cores | orchestrator |
| 2026-11-11 | Watchdog | `dfthc.testing.watchdog`, macOS `libproc` through ctypes (no new dependency; the sandbox forbids `ps`). Gate metric: the sampled peak of the process tree's summed physical footprint, which counts compressed and swapped pages (resident size read 190 to 220 MB for 300 MB under memory pressure). The getrusage sum is recorded as an upper bound. Priority: `nice -n 10`, or `taskpolicy -c utility` where the sandbox forbids setpriority() | orchestrator |
| 2026-11-11 | K-1 under the load policy | Task 0.9 compiles the small and mid shapes and projects FeMoco-54 (gates from the resource functions; power law in time; least-squares line in memory). The FeMoco compile runs once, alone, under the watchdog, only if the projection is within 8 GB and 60 min; remedies are evaluated at mid size first. Otherwise the projection goes to the user at G0, and that one answer decides where every compile above the cap runs (larger machine, or approved idle-time window with a higher cap), including Task 4.5's and Task 4.3's `fe4s4` walks if projected above 8 GB. Meanwhile Phase 1 runs except Tasks 1C.3 to 1C.5 | orchestrator |
| 2026-11-11 | K-10 FeMoco timings | stored as what they are: a dry-run measurement taken while writing the plan (199 to 274 s and 1,517 s), quoted from the spikes plan at commit `994f7aa` in `spikes/results/k10_femoco_dryrun.json`, not reproduced; Task 0.8 compiles 544 and 1,088 addresses only | orchestrator |
| 2026-11-11 | FeMoco compile in Phase 4 | a separate Task 4.5 after 4.4, the Q-1 decision and Q1b, so the walk and controlled walk compile once on final code; per-operator FeMoco rows are closed forms validated exactly at the compiled sizes and cross-checked exactly against the compiled totals; counts cached in `docs/resources.json` with a docstring-insensitive code hash | orchestrator |
| 2026-11-11 | Mid size for compiled T4 | preset `MID_NAPHTHALENE` (the `naphthalene` factors, FeMoco precisions), the third compiled size of every operator, nightly; the QPE program compiles at the H2 presets only | orchestrator |
| 2026-11-11 | Watchdog hardening (plan review round 3b) | fails closed. It refuses before launch off macOS, and unless libproc reads its own process. ESRCH is the only "gone": any other libproc error kills the tree and ends the run as a watchdog error. It samples at once, then every 50 ms (`POLL_SECONDS`), and kills at 7.5 GB, 15/16 of the 8 GB cap (`SAFETY_FRACTION`). It tracks processes by pid and start time, so descendants that leave the tree or the process group stay measured and killed, and reused pids are left alone. Kill: stop the tree, re-scan until no new process appears, kill all, re-check; cleanup however the run ends. Remaining gap: a descendant that leaves both the tree and the group within 50 ms of its creation | orchestrator |
| 2026-11-11 | K-1 gate hardening (plan review round 3b) | `code_hash()` covers the driver, the prototypes, `dfthc.compile`, the watchdog and `environment()` (Python, platform, the PennyLane, Catalyst, Lightning, JAX, jaxlib and NumPy versions, the lock file's SHA-256). FeMoco is allowed only if the projected sampled peak and the projected getrusage upper bound both fit 8 GB, and the time 60 min. Each FeMoco program is checkpointed by hash; a completed one is reused, a failed one is retried only with `--allow-retry` after the user's approval. A FeMoco `--program` run without the running driver's token is refused | orchestrator |
| 2026-11-11 | Stale large-system rows in Phase 5 | after a code change, `fe2s2` and `fe4s4` are not recompiled: their earlier measurements stay in `docs/resources.json` as history with their hash, and their current report rows show the closed form, marked pending, never as current measured validation | orchestrator |
| 2026-11-11 | Reduced release tier | 8 h with 2 workers (was 12 h with every core): E-1 and E-3 under S1 on all 16 inputs, one seeded schedule on the 4 fixed inputs instead of two on all 16, `H2_631G_TEST` E-1 unchanged, FeMoco counts from the cache. The nightly tier gains E-1 under S1 on the 4 fixed inputs, so a release-only failure is unlikely to surface first in Phase 5. Phase 3 projects the release run; D-2's release schedules run once, in Task 5.2 | orchestrator |

## Heavy runs

Every heavy run (master plan section 12.9) gets one row, started by the orchestrator only. The
watchdog's `WATCHDOG {json}` line gives the wall time, peak and kill status. "Large compile" counts
FeMoco-scale compiles against the budget of about six; more need the user's approval.

| Date | Run (command) | Started by | Wall time | Peak memory (tree footprint; getrusage sum) | Killed | Large compile |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-11 | K-10 parts 1 and 2 at the FeMoco-54 inner-erase shape (plan dry-run, before the policy) | plan author | 199 to 274 s and 1,517 s | not recorded | no | not counted (before the policy) |

## Gate log

| Gate | Date | Exit criteria met | Escalations | Next plan approved | User summary sent |
| --- | --- | --- | --- | --- | --- |
| G0 | | | | | |
| G1 | | | | | |
| G2 | | | | | |
| G3 | | | | | |
| G4 (Q-1 decision) | | | | | |
| G5 | | | | | |

## Canary state changes

| Date | Canaries | Change | Recorded in |
| --- | --- | --- | --- |
| 2026-11-11 | L-2, L-6, L-9, L-15 (two items), L-19, L-20 | today's behaviour differs from the spec's original claim (plan dry-run) | Task 0.4: `docs/limitations.md`, spec section 14 |
| 2026-11-11 | L-24, L-25 | new limitations found by the plan dry-run | Tasks 0.3 and 0.4 |

## Minor findings (input to Task 5.1)

| Task | Finding (file:line) | Review | Status |
| --- | --- | --- | --- |
