# Ledger

The durable record of the implementation. The orchestrator maintains it on local `main` and
commits it after every merge, spike and gate (master plan, section 12.8). After any loss of
context, trust this file and `git log` over memory: a task marked `merged` is never re-dispatched.

## Task status

Row format: `Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues`.

- **Status values:** `not started`, `in progress`, `in review`, `fixes`, `merged`, `blocked`,
  `cancelled`, `not needed` (conditional tasks).
- **Test evidence:** the command and the key output line, for example `scripts/test.sh tests/x.py: 23 passed`.
- **Reviews:** the verdict and the review file, for example `Approved (review-ab12..cd34)`; Opus is `n/a` for Tier B.
- **Open issues:** Minor findings go to the list at the end, so write a pointer here.

### Phase 0: foundation and feasibility

| Id | Title | Tier | Status | Branch | Merge commit | Test evidence | Reviews (GPT-5.6 / Opus) | Open issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | Scaffold, environment, lock, test runner | B | not started | `phase0/0.1` | | | / n/a | |
| 0.2 | `dfthc.compile`, resource assertions, canary runner | A | not started | `phase0/0.2` | | | / | |
| 0.3 | Canary suite A (L-1 to L-14, L-19, L-23 to L-25) | B | not started | `phase0/0.3` | | | / n/a | needs the catalyst#3431 reproducer (scout) |
| 0.4 | Canary suite B (L-15 to L-18, L-20 to L-22), spec section 14 | B | not started | `phase0/0.4` | | | / n/a | spec edit approved by the orchestrator at merge |
| 0.5 | Vendored dfthcbe data and license | B | not started | `phase0/0.5` | | | / n/a | |
| 0.6 | Spike K-7: comparator and adder adjoints | B | not started | `phase0/0.6` | | | / n/a | |
| 0.7 | Spike K-3: mode-agnostic measured erase; prototypes | A | not started | `phase0/0.7` | | | / | |
| 0.8 | Spike K-10: measurement arrays as dynamic arguments | A | not started | `phase0/0.8` | | | / | |
| 0.9 | Spike K-1: whole-program FeMoco-shaped compile | A | not started | `phase0/0.9` | | | / | |
| 0.9b | (conditional) K-1 remedy 1 in the stand-in | A | not needed unless 0.9 fails | `phase0/0.9b` | | | / | |

### Phases 1 to 5

Rows are added when each phase's code-level plan is approved at its gate, with the ids of master
plan section 4: 1A.1 to 1A.7, 1B.1 to 1B.3, 1C.1 to 1C.6, 2D.1, 2D.2, 2E.1 to 2E.5, 3.1 to 3.5,
4.1 to 4.4, Q1b (conditional), 5.1, 5.2.

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
