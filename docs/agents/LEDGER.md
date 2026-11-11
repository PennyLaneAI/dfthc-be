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

Every task of the master plan has a row from the start (43 planned, 2 conditional), so a task
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
| 0.9 | Spike K-1: whole-program FeMoco-shaped compile | A | not started | `phase0/0.9` | | | / | |
| 0.9b | (conditional) K-1 remedy 1 in the stand-in | A | not started | `phase0/0.9b` | | | / | conditional: opened at G0 only if 0.9 exceeds its budget; otherwise set to `not needed` |

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
| 3.4 | `DFTHCWalk`, `C(DFTHCWalk)`, FeMoco whole-program counts | A | not started | `phase3/3.4` | | | / | |
| 3.5 | End-to-end checks E-1 to E-4 and tier budgets | A | not started | `phase3/3.5` | | | / | |

### Phase 4: application, report and documentation

| Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.1 | QPE: `ControlledWalkPower`, `QPERound`, driver, energies, `ToyWalk` | A | not started | `phase4/4.1` | | | / | |
| 4.2 | QPE at FeMoco shape | B | not started | `phase4/4.2` | | | / n/a | |
| 4.3 | Resource report and `dfthc.costs_reference` | B | not started | `phase4/4.3` | | | / n/a | |
| 4.4 | Documentation set and docstring audit | B | not started | `phase4/4.4` | | | / n/a | |
| Q1b | (conditional) Q-1 variant (b): unprepare-from-outcomes operators | A | not started | `phase4/Q1b` | | | / | conditional: opened only if the user chooses Q-1 variant (b) after 4.4; regenerates the report and documentation before G4; otherwise set to `not needed` |

### Phase 5: final review

| Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1 | Whole-branch final review | A | not started | none | | | / | read-only: no branch, no merge |
| 5.2 | Fix wave, release tier, final merge | A | not started | `phase5/5.2` | | | / | |

## Spike results

| Spike | Task | Verdict | Measurements | Consequence |
| --- | --- | --- | --- | --- |
| T6 verbose listing | 0.2 | not run | plan dry-run 2026-11-11: capturable only with `qp.qjit(verbose=True)`; lines `Chosen Rule: __builtin_<rule>` | T6 tests may assert rule names |
| K-7 comparator adjoints | 0.6 | not run | plan dry-run: 6/6 passed | |
| K-3 mode-agnostic erase | 0.7 | not run | plan dry-run: pass; T = 9, M = 4; tape rules valid; tape path exact | |
| K-10 measurement arrays | 0.8 | not run | plan dry-run: pass; FeMoco compiles 274 s (packed parities) and 1,517 s (masks as compilable data) | |
| K-1 whole program | 0.9 | not run | plan dry-run, small shape: walk T 236 (10 s), controlled walk T 238 (19 s); FeMoco not run | |

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
| 2026-11-11 | Models | Tier A `claude-opus-5-5-high-fast`; Tier B `claude-sonnet-5-5-high`; review gate `gpt-5.6-sol-high-fast`; scout `grok-4.7-high-fast` | user |

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
