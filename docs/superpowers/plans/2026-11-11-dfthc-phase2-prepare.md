# DFTHC Phase 2, Workstream D: Alias-Sampling Prepares Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the outer and inner alias-sampling prepares of the block encoding (steps 1 to 4 and 5 to 8), their measured unprepares as explicit adjoint classes (steps 24 to 27 and 19 to 22) and the inner signs (step 9): `Operator2`s with one pinned rule each, exact Toffoli, measurement and measured-uncompute counts against `dfthc.costs`, tests T2 to T6, the failed-address tests of spec 12.1 and the outcome certificates of spec 12.5.

**Architecture:** One module, `src/dfthc/ops/prepare.py` (spec 5.2). A private frozen dataclass `_AliasSampling` holds the five alias-sampling steps, their measured inverse and both resource declarations; each prepare maps its named registers onto it, so the two prepares share one rule skeleton. Each prepare packs its compilable tables into the read's dynamic `leaf_words` inside its rule (`read_table`, then `QROAMRead.leaf_words_from_table`), and its unprepare hands the same table to `QROAMErase`. The rules are straight-line and their children are composite keys, so they declare their resources by hand (the Phase 1 C resource-function rule), and T2 runs them at a size in every branch. Semantics are checked term by term on the sparse simulator, and on `lightning.qubit` where a program fits (the outer prepare at `H2_STO3G_TEST`, 19 wires).

**Tech Stack:** Python 3.12, PennyLane `Operator2` and Catalyst (`qp.qjit(capture=True)`, `qp.decompose`, `qp.analyze`), NumPy; the sparse simulator `dfthc.testing.sparse_sim` and `dfthc.testing.tape`; pytest with pytest-xdist; ruff.

This file holds Tasks 2D.1 and 2D.2 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 8). Workstream E (Tasks 2E.1 to 2E.5) has its own plan file; the two workstreams share no file except possibly `src/dfthc/costs.py` (section "Shared edits").

**Code provenance.** Every code block below was executed on 2026-11-12 in a scratch replica of the repository (`.scratch/plan-check/repo-phase2d`, not committed), seeded from local `main` at `9ce0a74` (`git archive`) and resynced to `main` at `f7acf97` (Task 1H.1: the import-aware tolerance guard `tests/test_tolerance_hygiene.py`, the watchdog test, HANDOFF section 6); the replica then differs from `main` only by this plan's files. The replica is not a git repository, so the dry-run ran pytest the way `scripts/test.sh` does (same environment, at most 2 workers, `taskpolicy -c utility`, `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` exported, `TMPDIR` under `.scratch/`); the steps below use `scripts/test.sh`. Both tasks were replayed in order in a fresh copy holding only `main`'s files plus the earlier task's (`.scratch/plan-check/stages/2D/run_stages_2d.sh`, log `stages/2D/2d-stages.log`; 1.5 to 4 minutes, depending on whether another lane's tests were running): every "Expected" line below is that replay's output, and the replayed files are byte-identical to the replica's (Task 2D.1's `prepare.py` and certificate file are derived from the final ones by `stages/2D/derive_stage_2d1.py`). Supporting evidence, all under `.scratch/plan-check/`: the `MID_NAPHTHALENE` compiles, each alone under `run_watched` (`stages/2D/run_mid_watched_2d.py`, log `stages/2D/2d-mid-compiles.log`); a mutation check of eleven wrong `prepare.py` versions against the three test files (`stages/2D/mutate_prepare_2d.py`, log `stages/2D/2d-mutants.log`: every mutant fails at least one test); dfthcbe's per-piece FeMoco counts from its call graph in `.venv-ref` (`probes/p2d_dfthcbe_blocks.py`, log `stages/2D/dfthcbe-blocks.log`); the one-hot constraint at every preset and vendored system (`probes/p2d_onehot_constraint.py`, log `stages/2D/onehot-constraint.log`); the API probes behind the rules (`probes/p2d_api.py`: a 1-bit `LeftQuantumComparator` compiles with T = 1, empty registers and `Wire[0]` keys match inside a rule); and one per-commit tier on the final replica (log `stages/2D/2d-per-commit-tier.log`).

**Load.** Nothing here is heavy and nothing is FeMoco-shaped. The four `MID_NAPHTHALENE` compiles take 1.4 to 2.3 s each at peaks of 284 to 320 MB (watchdog footprint, a 4 GB cap); every per-commit test takes under 10 s in a warm worker.

---

## Changes from the master plan, the spec and HANDOFF

The exact old and new texts are in `phase2d-doc-edits.md` (written and checked against `main` at `f7acf97` by `.scratch/plan-check/stages/2D/make_phase2d_doc_edits.py`: every old text occurs exactly once, and the edits apply in order). 16 edits; no template was edited, so the orchestrator mirrors D-M1 to D-M10 into the master plan's template when it applies them. D-M1 and D-M3 touch the same table rows as the Workstream E plan's edits.

1. **Plan file** (D-M1 to D-M3). Tasks 2D.1 and 2D.2 move from `phase2-blocks` to this file, `phase2-prepare`; section 1 lists it and its provenance.
2. **Interfaces gain their helpers and one constraint** (D-M4, D-M8).
   - Static methods on each prepare: `required_wires` (shared by the prepare and its unprepare: `work` is the largest need of the uniform preparation, the read and the erase, so both classes take the same wires), `read_shape` and `read_table` (the read's entry layout, which the erase reuses); the module constant `COMPARATOR = "<="`.
   - The constructors require that the erase's one-hot fits in the reset target and junk (`2**ke1 <= K1 d1`; `2**(ke2 + w_b) <= K2 d2`), so `QROAMErase` gets an empty `extra_onehot` (Task 1C.5's note: 0 wires at every preset). It holds at every preset and at every vendored system with the block bits `cheapest_block_bits` chooses, `fe2s2` and `fe4s4` included (`stages/2D/onehot-constraint.log`).
   - `phase_grad` is a minimum of `s_u` wires (Phase 3 passes the whole register), and the rules pass the uniform preparation `phase_grad[:s_u]`; one T2 case per prepare and one outer certificate use a register two wires longer (D-M5, D-M9).
   - Validation domains: `X_o >= 2`; `N, R, B, C >= 1` for the inner prepare (the operator only needs `N + RC >= 2`, which `B, C, R >= 1` and `N >= 1` give); `b_k >= 1` (a 1-bit comparator compiles: T = 1, its adjoint A = 1); `s_u >= 3`; tables of the documented shapes and ranges.
3. **T3 states the success phase and factor** (D-M5, D-M9; ledger row 1C.2). Every amplitude is compared, phase included, with the uniform preparation's amplitude recomputed from spec 9.2 (`uniform_amplitudes` in the tests) times `1 / sqrt(2**(b_k + 1))`, so the `sqrt(1 - P_fail)` factor and the phase `e^{i phi}` are both checked. The outer T3 runs on the sparse simulator at five shapes (both presets have `X_o` a power of two, so three more shapes give the uniform preparation a failure branch and a phase, dead positions, `k1 = 0` and `b_k1 = 1`), and on lightning at `H2_STO3G_TEST` (the master plan's lightning case). The inner programs have at least 33 wires, so the inner T3 runs on the sparse simulator only.
4. **The encoded model in T3.** The outer success weight equals `Lambda / Lambda~` of `encoded_block`, and its one-body part `(1 - P_fail) w~_pm / Lambda` of `encoded_factors`. The inner per-row signed weights equal `inner_weights(encoded_factors(...)) / sqrt(4 Lambda p~_rc)`, the model's `a_b`, with the failed branches as the identity term (spec 8.5).
5. **`dfthc.costs.inner_signs()`** (D-M7; "Shared edits"): the spec 9.24 row `InnerSigns 0 / 0` gets its closed form, like `spin_mux()` and `majorana_gate()`, so the docstring's cost section names a `dfthc.costs` function and T4 asserts against it.
6. **Certificate shapes** (D-M6, D-M10): listed in the master plan, with their zero row and partly covered fix-up leaves. D-M6 also states the outer certificate's scope: its fixed all-`|0>` input is the pair's only legal pre-state, and spec 12.5's general premise of random complex address and control amplitudes rests on Task 1C.5's `QROAMRead` → `QROAMErase` certificates.
7. **Spec 9.7 and 9.8** (D-S2 to D-S5): the one-hot constraint in both Compilable rows and the table-argument names `sgn_table`, `d_sgn_table`; 9.7's Post row states the success phase and factor and what the failure part looks like (its "Failure branches behave as in dfthcbe" is replaced; ledger row 1C.2); 9.8 gains the matching Post row.
8. **Grouped wires** (D-S1, D-H1). Spec 6.7 and HANDOFF section 6 say that "composite operators take grouped wire arguments", while spec 9.7, 9.8 and the master plan give the prepares named registers, as spec 9.0's wire-passing rule requires. The edits restrict the grouped form to `InnerSelect` and `InnerSelectDagger` (spec 9.18) and the assembly operators of spec 9.21 to 9.23, and say that the blocks they call take the registers their spec 9 tables list. These sentences concern Workstream E's blocks too; the orchestrator may coordinate them with that plan.
9. **Not added:** a test that `qp.adjoint` or `qp.ctrl` of an unprepare fails to compile; this plan does not pin a direct `OuterUnprepare` or `InnerUnprepare` modifier failure. The safety nets are Task 1C.5, which pins the loud adjoint and control failure of `QROAMErase`, the unprepares' only measuring child, and Task 3.4, whose inventory test checks that the walk and the controlled walk capture no `Adjoint(...)` or `C(...)` target whose base is a dfthc composite.

## How this plan answers the binding ledger rows

| Ledger row | What the plan does |
| --- | --- |
| Decision "K-1 at G0" (per-leaf loads, explicit classes) | `_AliasSampling.prepare` builds the read's `leaf_words` with `QROAMRead.leaf_words_from_table` from the compilable table inside the rule; `OuterUnprepare` and `InnerUnprepare` are classes of their own with pinned base rules; the only adjoints in a rule are the leaf adjoints `qp.adjoint(FlaggedUniformPrep(...))` and `qp.adjoint(qp.LeftQuantumComparator(...))` (a PennyLane primitive, K-7); nothing calls `qp.adjoint` or `qp.ctrl` on a dfthc composite. |
| Decision "Measuring-rule warnings" | No test wraps a measuring operator, so the expected "Skipped ... mid-circuit measurement" warning never appears; every other warning naming a dfthc operator fails a test (the autouse fixture). |
| Decision "K-10 consequences" | The erase masks stay derived inside `QROAMErase`'s rule from the compilable table, which the unprepares pass unchanged (item 2). `InnerUnprepareFromOutcomes` belongs to the conditional Task Q1b. |
| Decision "Table words are `int64`" | `leaf_words` come from `leaf_words_from_table` (`WORD_DTYPE = np.int64`); the read's resource key uses `AbstractArray(shape, WORD_DTYPE)`. |
| 1C.2: success phase and factor | Changes 3 and 7: every T3 amplitude is compared with phase and factor; spec 9.7's Post is reworded (D-S3), 9.8 gets a Post row (D-S5), the master plan's 2D.1 T3 is rewritten (D-M5). |
| 1C.2: unprepare contract | `_AliasSampling.unprepare` applies `qp.adjoint(FlaggedUniformPrep(...))` on the prepare's index, `succ`, `succ_work`, `phase_grad[:s_u]` and work prefix; the docstrings' pre- and postconditions forbid writing `succ`, `succ_work`, `cmp_work` and the alias registers in between; the swap is undone first. The certificates assert that a pair's only measurements are the erase's (`K width` `MidMeasure`s) and that its only conditionals are the fix-ups (premise check). |
| 1C.5: erase contract | Same compilable table and entry layout for read and erase (`read_table`); `work` the larger need (`required_wires`); `extra_onehot` empty (constructor check, change 2); swap undone before the erase; no enable. |
| 1C.5: Phase 2 certificates | Zero row in the inner `(6, 2)` table (with rank 0, so its masks are zero); partly covered fix-up leaves (outer `X_o = 5` with `ke1 = 2`; inner `(6, 2)` last leaf, every `(5, 1)` leaf); a random complex input on every value of `x_o` for the inner pair; the outer pair runs from its only legal pre-state, every register in `|0>` (it has no input register, and the walk has no enable), so any outcome-dependent phase shows against that fixed input. That fixed input does not instantiate spec 12.5's general premise of random complex address and control amplitudes: Task 1C.5's merged `QROAMRead` → `QROAMErase` certificates carry it for the read and erase, and the outer certificate composes them with the pair's other children. Premise check; tape path only. |
| 1C.3: Clifford counts are upper bounds | Tests compare T, M and A exactly and assert no `CNOT` or `PauliX` count; the four docstrings label those counts as bounds (the read's loads, `exact=False`; the erase's reset `PauliX`). |
| 1C.3: `int64` resource keys | As for "Table words are `int64`"; the optional `np.signedinteger` check in `qroam.py` is not made (no shared edit needed). |
| 1C.1: new emitter gate names | None: the rules use no emitter. |
| 1A.5: `normalization(encoded_factors(...))` is not the walk's | T3 uses `Lambda~` from `encoded_block` and the input `Lambda` from `normalization(factors)`. |
| 1A.5: the `x_o` failure is untested end to end | The outer T3 at `X_o = 5`, 6 and 7 checks the failure branch's amplitudes and weight against `encoded_block`'s `Lambda~`; the end-to-end check stays with Task 3.5. |
| Cost comparison rows (1C.1 to 1C.5) | Background for T4: the per-block comparison with dfthcbe below names every term. |
| 1A.5 (Givens order), 1C.4 (`AngleErase`) | Workstream E's; not in this plan. |

## Findings for other plans and documents

1. **FeMoco-54 numbers.** `dfthc.costs` on `main` at `f7acf97` gives exactly the master plan's: `OuterPrepare` 210 = F(324) 30 + read 156 + `b_k1` 15 + `w_x` 9; `OuterUnprepare` 69 = 9 + erase 36 + F†(324) 24 with M = 96; `InnerPrepare` 971 = 18 + 933 + 15 + 5; `InnerUnprepare` 217 = 5 + 196 + 16 with M = 416; `InnerSigns` 0. The measured AND uncomputes, derived per operator ("Design notes"): `OuterPrepare` A = 108 = 24 + 84 (uniform preparation, read iteration), `OuterUnprepare` 81 = 15 + 36 + 30 (comparator, erase fix-up, adjoint uniform preparation), `InnerPrepare` 559 = 16 + 543, `InnerUnprepare` 229 = 15 + 196 + 18, `InnerSigns` 0. The tests also rebuild every piece from the operators' own read and erase shapes (`read_shape`, `QROAMRead.visited_ranges`, `QROAMErase.fixup_high_ranges`). In `dfthc.costs.measured_and_uncomputes` (A = T − `toffoli_only`; the walk's 8,675 of spec 9.24's FeMoco table, which Task 3.4 asserts), Workstream D's blocks contribute 108 + 81 + 2 (559 + 229) = 1,765, which is their T, 210 + 69 + 2 (971 + 217) = 2,655, less their Toffoli-only part 2 w_x + 4 w_b + (K1 − 1) d1 + 2 (K2 − 1) d2 = 18 + 20 + 72 + 780 = 890. The other 8,675 − 1,765 = 6,910 = 2 · 3,537 + 20 + 44 − (4N + 12 = 228) belongs to the two inner selects (Workstream E's blocks, assembled in Task 3.1) and the two reflections (Task 3.2). The compiled counts equal the closed forms at `H2_STO3G_TEST` (outer 8 / 2 with M = 8, inner 21 / 14 with M = 14), `H2_631G_TEST` (12 / 5, M 10; 25 / 15, M 14), the `X_o = 5` shape (24 / 13, M 12) and `MID_NAPHTHALENE` (92 / 38, M 42; 185 / 53, M 92).
2. **Against dfthcbe** (FeMoco-54, its call graph in `.venv-ref`, `stages/2D/dfthcbe-blocks.log`; uniform preparation + read or erase + comparator + swaps):

   | Block | Ours | dfthcbe | Difference and causes |
   | --- | ---: | ---: | --- |
   | `OuterPrepare` | 30 + 156 + 15 + 9 = 210 | 37 + 151 + 15 + 9 = 212 | −2: uniform preparation −7 (PennyLane's comparators, s − 1 rotations), read +5 (exact iteration) |
   | `OuterUnprepare` | 24 + 36 + 0 + 9 = 69 (M 96) | 37 + 33 + 15 + 9 = 94 (M 96) | −25: comparator adjoint −15 (measured uncompute), erase +3 (exact iteration), uniform adjoint −13 |
   | `InnerPrepare` | 18 + 933 + 15 + 5 = 971 | 21 + 872 + 15 + 5 = 913 | +58: uniform −3, read +61 (`r` through the swap +60, exact iteration +1) |
   | `InnerUnprepare` | 16 + 196 + 0 + 5 = 217 (M 416) | 21 + 179 + 15 + 5 = 220 (M 356) | −3: comparator −15, erase +17 (one-hot over 128 positions +16, exact iteration +1), uniform adjoint −5 |

   Every term is one of spec 13.1's documented causes (`cmp-free`, `exact-iter`, `flat-r`, `onehot-pad`).
3. **For Phase 3** (Task 3.3): call `OuterPrepare` and `OuterUnprepare` with `(p.X_o, tables.outer_alt, tables.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)` and `InnerPrepare` and `InnerUnprepare` with `(p.N, p.R, p.B, p.C, tables.inner_alt, tables.inner_keep, tables.inner_sgn, tables.inner_d_sgn, p.b_k2, p.k2, p.ke2, p.s_u)`, plus each group's registers by name (`WalkLayout.split_group`), `phase_grad` whole (the rules slice `phase_grad[:s_u]`) and `work[:required_wires(...)["work"]]`. At FeMoco the work needs are 20 (outer) and 16 (inner), within `WalkLayout.work_size` = 44. `InnerSigns` sits between `InnerPrepare` and the select (spec 9.21); its phases leave the unprepare's preconditions intact.
4. **For Workstream E:** `_AliasSampling` is private to this module; nothing in it is meant for reuse outside `prepare.py`.

## Revision 2

After the GPT-5.6 plan review (`.superpowers/sdd/phase2d-plan-review-1.md`: 0 Critical, 3 Important, 1 Minor; all findings accepted):

1. **FeMoco A (I-1).** Both FeMoco tests assert the per-operator A (outer 108 / 81, inner 559 / 229) and that each pair's A is its T less its Toffoli-only part; `InnerSigns`' A = 0 is its compiled count, the same at every size. D-M5 and D-M9 carry the values; finding 1 places them in `measured_and_uncomputes`' 8,675.
2. **Failed-read wording (I-2).** D-S3, D-S5, the "Failure branches" design note and the docstrings (module, `OuterPrepare`, `InnerPrepare`, `InnerSigns`) say that the selected target entry is zero while the junk may hold other entries of a visited block. D-S3 and D-S5 write `P_fail(X_o, s_u)` and `P_fail(B + 1, s_u)`.
3. **Oversized `phase_grad` (I-3).** The outer T2 at `X_O_FAILS`, the inner T2 at the `H2_STO3G_TEST` shape and the outer certificate at `X_o = 5` use a `phase_grad` two wires longer than `s_u`; no test is added (44 and 48 per commit, 700 and 748). Mutant 11, `phase_grad_unsliced`, fails the four T2 cases.
4. **Outer certificate scope (M-1).** D-M6, the ledger row "1C.5: Phase 2 certificates", change 6 and the certificate file's docstring qualify the fixed all-`|0>` input and cross-reference Task 1C.5's certificates; change 9 names Task 1C.5 and Task 3.4 as the safety nets.
5. **Grouped wires.** D-S1, D-H1 and change 8 add `InnerSelect` and `InnerSelectDagger` (spec 9.18) to the assembly operators.

## Revision 3

After Task 2D.1's code reviews (2026-11-13): GPT-5.6, `.superpowers/sdd/task-2D.1-review-1.md` (0 Critical, 2 Important, 0 Minor), and Opus, `.superpowers/sdd/task-2D.1-review-opus.md` (0 Critical, 1 Important, 3 Minor). Task 2D.1 merged as planned (`b6fcd42`) plus a tests-and-docstring fix round. The Task 2D.1 block now records what merges, and Task 2D.2's block follows it:

1. **Which gradient wires the rotations use** (GPT-5.6 finding 1, Opus I-1). T2 and the certificate pin only the size of the uniform preparation's slice, so a rule that took the bottom `s_u` wires passed every test. T3 now gives `X_O_FAILS` (outer) and the (5, 1) shape (inner) a `phase_grad` two wires longer than `s_u`, so the amplitudes pin the top wires. The two bottom-slice mutants (both calls; the prepare only) fail the inner T3 at (5, 1) and pass every per-commit test of revision 2's inner file (`stages/2D/mutate_rev3_2d.py`, log `stages/2D/2d-mutants-rev3.log`).
2. **Unprepare semantics per commit** (Opus M-1). Both T3s end with a round trip under the all-ones outcome schedule in strict mode: every register returns to `|0>`, the inner `x_o` to its uniform superposition. The inner one adds about 0.7 s to its file's 20 s with one worker.
3. **The erase's work need** (Opus M-2). The outer by-hand test asserts `required_wires(4, 2, 1, 0, 3)["work"] == 1`, a `ke1 = 0` shape where the erase needs the most work.
4. **The free comparator adjoint** (Opus M-3). The `OuterUnprepare` and `InnerUnprepare` docstrings cite the paper's remark after Eq. (127): the comparator's `b_k1 - 1` and `b_k2 - 1` work wires stay dirty until the unprepare.

No count changes (44 and 48 per commit, 700 and 748). Re-run for revision 3 with one worker (log `stages/2D/2d-rev3-replay.log`): Task 2D.2's per-commit inner tests (`48 passed`), the inner certificates (`4 passed`) and Task 2D.1's T3 (`5 passed`). Task 2D.1's whole outer file with these tests passed in the Opus review (`48 passed` with the certificates, without the two `MID_NAPHTHALENE` compiles). The other "Expected" lines are revision 2's replay; revision 3 changes only the two T3 tests, one by-hand assertion and two docstrings. GPT-5.6's finding 2, the sampled premise check in `nonlinear_conditionals`, is fixed in `dfthc.testing.sparse_sim` by the Task 2D.1 fix round; the call `nonlinear_conditionals(operations)` in the certificates is unchanged.

## Revision 4

After Task 2D.2's code reviews (2026-11-13): GPT-5.6, `.superpowers/sdd/task-2D.2-review-1.md` (Approved, no findings), and Opus, `.superpowers/sdd/task-2D.2-review-opus.md` (0 Critical, 1 Important, 2 Minor). Applied on `phase2/2D.2` and here:

1. **The rank at a `C` that is not a power of two** (Opus I-1). Every T3 shape and certificate has `C` = 1 or 2, where `x_o // C` equals `x_o >> lg(C)`, so a rule that took `r` from the top bits of `x_o` (the paper's shortcut for a power-of-two `C`) or truncated it passed every test; FeMoco has `C` = 27. The inner by-hand test asserts `read_table` at `R = C = 3`, which every T3 term is tied to through `loaded_bits`.
2. **Mask-bit permutations per commit** (Opus M-1). The all-ones schedule sees only each address's mask parity, so an erase table built with `sgn_table` and `d_sgn_table` swapped passed every per-commit test. The inner T3 round trip also runs the alternating schedule.
3. **The paper's tail fix-up** (Opus M-2). The `InnerUnprepare` docstring says that Eq. (123)'s `N` term is `AngleErase`'s here.

No count changes (48 per commit). Task 2D.2's inner file and the certificates passed with these edits (`58 passed`, two workers). Task 2D.1's outer round trip has the same all-ones blind spot; its certificates cover it.

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
| `src/dfthc/ops/prepare.py` | 2D.1, replaced whole by 2D.2 | `COMPARATOR`, the skeleton `_AliasSampling`, `OuterPrepare`, `OuterUnprepare`; then `InnerPrepare`, `InnerUnprepare`, `InnerSigns` |
| `tests/ops/test_outer_prepare.py` | 2D.1 | the alias state term by term, the encoded model, counts, rule selection, validation |
| `tests/ops/test_inner_prepare.py` | 2D.2 | the spec 12.1 sign and failure-sign tests, the encoded model, counts, `InnerSigns`, validation |
| `tests/ops/test_certificates_prepare.py` | 2D.1, replaced whole by 2D.2 | the spec 12.5 certificates of both pairs (`slow`) |
| `src/dfthc/costs.py` | 2D.2 (one function) | `inner_signs()` |

Task 2D.2 replaces `prepare.py` with a version that contains every line of Task 2D.1's, in order: it only inserts the module docstring's "Inner read" and "Signs" paragraphs and the three inner sections at the end (checked with `difflib`: insertions only, 482 lines become 915). It replaces the certificate file with a version whose outer part is unchanged; the docstring gains a paragraph, the import line names the inner classes, and the inner certificate is appended. In wave 2-a, Task 2D.1's files are disjoint from Workstream E's; in wave 2-b, Task 2D.2 edits `costs.py`, which no Workstream E task of the master plan lists.

**Shell conventions.** Run commands from the worktree root of your lane. `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` locates the shared environment, as in the earlier plans.

## Shared edits

One edit outside `prepare.py`, in Task 2D.2: `inner_signs` in `src/dfthc/costs.py`, right after `inner_unprepare` (the spec 9.24 row `InnerSigns 0 / 0`). It is count-neutral: `block_encoding` keeps its formula, which already counts `InnerSigns` as 0. The exact diff against `main` at `f7acf97`:

```text
--- a/src/dfthc/costs.py
+++ b/src/dfthc/costs.py
@@ -188,6 +188,11 @@
     return Cost(p.w_b) + erase + flagged_uniform_unprep(p.B + 1, p.s_u)
 
 
+def inner_signs() -> Cost:
+    """``InnerSigns`` (spec 9.24 row): ``T = 0``, ``M = 0``."""
+    return Cost(0)
+
+
 def tail_angle_load(p: WalkParams) -> Cost:
     """``TailAngleLoad`` (spec 9.24 row): ``U(RC, X_o, w_x, 0)``."""
     return Cost(unary(p.RC, p.X_o, p.w_x, controlled=False))
```

## Expected test counts

Each task states its targeted counts exactly. The per-commit tier (`scripts/test.sh -m "not slow and not release" -q`) counts **656 tests on `main` at `f7acf97`** (measured on the replica: `656/674 tests collected (18 deselected)`; Task 1H.1 added the guard's 2 tests to the 654 of `9ce0a74`). Each task's expected number is the count of `main` when its branch is created plus its own per-commit tests. Workstream E tasks may merge between 2D.1 and 2D.2; then shift by their tests. The pass criterion is "no failures, no errors, no warnings summary"; the number is a cross-check.

| Task | Base | Own per-commit | Expected per-commit tier | Own slow tests |
| --- | ---: | ---: | ---: | ---: |
| 2D.1 | 656 | 44 (`test_outer_prepare.py`) | 700 = 656 + 44 | 6 (2 `MID_NAPHTHALENE` counts, 4 certificates) |
| 2D.2 | 700 | 48 (`test_inner_prepare.py`) | 748 = 700 + 48 | 6 (2 `MID_NAPHTHALENE` counts, 4 more certificates) |

With both tasks merged, Workstream D adds 92 per-commit tests and 12 nightly tests. Dry run of the per-commit tier on the final replica: 748 tests, `747 passed` and one failure that occurs only in the replica, `tests/test_infrastructure.py::test_runner_fails_closed_without_lowered_priority` (it creates a directory named `venv` under `TMPDIR`, which the sandbox refuses inside the workspace, where the replica's `TMPDIR` lives; it passes in the worktree lanes, as the Phase 1 C plan recorded), in 289 s with 2 workers (log `stages/2D/2d-per-commit-tier.log`; it ran on the first revision before its last, docstring-only edit, the register tables of the two unprepare classes; that edit and revision 2 changed only Workstream D's files, and the replay re-ran every Workstream D test after both). Workstream D's own per-commit tests take about 30 s with 2 workers.

**Design notes the tasks share.**
- *The skeleton.* `_AliasSampling(L, s_u, b_k, read, onehot_bits)` owns the order of the five steps and of their inverse (module docstring), the work sizes of its three children, and both resource declarations. `OuterPrepare` maps `x_o` to the index and the read's address and `alt_x_o ‖ keep_x_o` to the target; `InnerPrepare` maps `b` to the index, `x_o ‖ b` to the address and `alt_b ‖ keep_b ‖ sgn ‖ d_sgn ‖ r` to the target. Each child gets the start of `work` it needs, and the uniform preparation gets `phase_grad[:s_u]` (fewer captured equations, as in Task 1C.2). One T2 case per prepare passes a `phase_grad` two wires longer, so a rule that hands on the whole register fails its declared resources (mutant `phase_grad_unsliced`); the outer certificate at `X_o = 5` checks that the extra wires end unchanged. T3 at `X_O_FAILS` and at the inner (5, 1) shape also use a longer register, so the amplitudes pin which wires the rotations add into: a rule that takes the bottom `s_u` wires fails T3 (revision 3).
- *The unprepare's order is forced.* The swap is undone first (the erase's precondition is that `target ‖ junk` holds what the read loaded at the current address), the comparator's adjoint comes next (it reads `keep`, which the erase clears), then the erase, the Hadamards and the adjoint uniform preparation, on the prepare's own wires. Two mutants of the mutation check break this order (the swap left in place, the erase before the comparator's adjoint); all eight certificates fail for each.
- *Failure branches* (spec 8.5). Where the selected target entry is zero (`x_0 >= X_o`; one-body rows, failed `x_o` and failed `b` for the inner read; the junk may hold other entries of a visited block, as at the dead positions of the `X_o = 5`, `(5, 1)` and `(6, 2)` shapes), `keep = 0`, so `less_than = 1` and the swap leaves the index register at 0 and the drawn value in `alt`; for the inner read `sgn = d_sgn = r = 0`, so `InnerSigns` applies +1.
- *Resources and A.* T and M come from `dfthc.costs`; A is derived per operator: the prepare uncomputes the uniform preparation's `2(n - 1) + 2(s_u - 1)` ANDs (none if the odd part is 1) and the read's iteration ANDs `U`; the unprepare the comparator's `b_k`, the erase fix-up's T and the adjoint uniform preparation's `3(n - 1) + 2(s_u - 1)`. A prepare and its unprepare together uncompute every AND they compute (spec 9.24), so the pair's A is its T less its swaps and swap network, `2 lg(L) + (K - 1) d`; the FeMoco tests assert both (finding 1).
- *T2 branches.* The hand-written declarations depend on zero-size registers only: `succ_work` and the uniform work (odd part 1), `junk` (`k = 0`), `cmp_work` (`b_k = 1`), `r` (`R = 1`). T2 runs at shapes covering each.
- *References in the tests.* `uniform_amplitudes` recomputes spec 9.2's state with its own `k`, independent of `amplitude_angles`; `QROAMRead.loaded_bits` of `read_table` gives what each term must hold; `alias_probabilities`, `encoded_factors`, `encoded_block`, `our_failure_signs` and `dfthcbe_failure_signs` are Task 1A.4's and 1A.5's.
- *FeMoco-54* values are asserted on the closed forms only; nothing FeMoco-shaped is compiled (master plan section 12.9).

---

### Task 2D.1: `OuterPrepare` and `OuterUnprepare`

**Task id:** `2D.1` · **Branch:** `phase2/2D.1` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** G1

**Files:**
- Create: `src/dfthc/ops/prepare.py` (the skeleton and the outer classes), `tests/ops/test_outer_prepare.py`, `tests/ops/test_certificates_prepare.py` (the outer certificate, marked `slow`)

**Interfaces:**
- Consumes: `FlaggedUniformPrep` (1C.2); `QROAMRead` with `leaf_words_from_table`, `leaf_words_shape`, `required_wires`, `visited_ranges`, `loaded_bits`, and `QROAMErase` with `required_wires`, `fixup_high_ranges`, `WORD`, `WORD_DTYPE` (1C.3 to 1C.5); `check_int`, `check_wires`, `canonical_ints` (1C.1); `register_rule`, `resource_key`, `compiled_resources`, `FIXED_DECOMPS`, `GATE_SET` (0.2); `dfthc.costs` `Cost`, `outer_prepare`, `outer_unprepare`, `qroam_read`, `qroam_erase`, `flagged_uniform_prep`, `flagged_uniform_unprep` (1A.7); `lg`, `odd_part`, `WalkParams` and the presets (1A.1); `alias_probabilities`, `WalkTables`, `load_factors`, `random_factors`, `normalization`, `encoded_block`, `encoded_factors`, `uniform_failure_probability` (1A.2 to 1A.5); `SparseSimulator`, `SparseState`, `ForcedPolicy`, `nonlinear_conditionals`, `decompose_for_simulation`, `chosen_rules`, `verbose_program` (0.2, 1B.1, 1B.2); PennyLane's `LeftQuantumComparator` (K-7: its adjoint compiles and is exact), `CSWAP`, `H`.
- Produces, in `dfthc.ops.prepare`:
  - `COMPARATOR = "<="`, and the private skeleton `_AliasSampling(L, s_u, b_k, read, onehot_bits)` (frozen dataclass) with `work_sizes()`, `sizes()`, `prepare(table, **registers)`, `unprepare(table, **registers)`, `prepare_resources()`, `unprepare_resources(table)`, which Task 2D.2 reuses, and the helpers `_words`, `_join`, `_check_table`, `_check_onehot`;
  - `OuterPrepare(X_o, alt, keep, b_k1, k1, ke1, s_u, x_o, succ_x_o, succ_work_x_o, sigma_1, m_x_o, less_than_x_o, cmp_work_x_o, alt_x_o, keep_x_o, junk_x_o, phase_grad, work)`: compilable `X_o >= 2`, `alt` (`X_o` ints in `[0, X_o)`), `keep` (`X_o` ints in `[0, 2**b_k1)`), `b_k1 >= 1`, `0 <= k1 <= lg(X_o)`, `0 <= ke1 <= lg(X_o)` with `2**ke1 <= 2**k1 (lg(X_o) + b_k1)`, `s_u >= 3`; static `required_wires(X_o, b_k1, k1, ke1, s_u) -> dict[str, int]` (`phase_grad` a minimum of `s_u`, `work` the largest need of the uniform preparation, the read and the erase), `read_shape(X_o, b_k1, k1) -> dict` and `read_table(X_o, alt, keep, b_k1) -> tuple` (`(x, words)` with entry `(alt[x] << b_k1) | keep[x]`); the rule `_outer_prepare`;
  - `OuterUnprepare` with the same arguments and `required_wires`; the rule `_outer_unprepare` (the measured unprepare of spec 9.7). Neither class is adjointed or controlled.

**Acceptance tests** (46 tests in `tests/ops/test_outer_prepare.py`, 2 of them `slow`; 4 `slow` tests in `tests/ops/test_certificates_prepare.py`):
- T3 on the sparse simulator (5 tests): both test presets with their tables, `X_O_FAILS` (`X_o = 5`, `b_k1 = 3`), `X_o = 6` with `k1 = 2` (dead positions 6 and 7) and `X_o = 7` with `k1 = 0`, `b_k1 = 1`, tables from `WalkTables.from_factors`. Every term decodes to its drawn `x_0`: the loaded `keep` and junk (`QROAMRead.loaded_bits` of `read_table`), `less_than = [keep <= m]`, the swap, `succ_x_o = [x_0 < X_o]`, `work` in `|0>`; its amplitude, with the catalysed gradient divided out, equals the uniform preparation's for `x_0` over `sqrt(2**(b_k1 + 1))` to 1e-12, phase and `sqrt(1 - P_fail)` included; every `(x_0, m, sigma_1)` occurs; P(x_o = x, succ = 1) equals `(1 - P_fail) alias_probabilities(...)`; the success weight equals `Lambda / Lambda~` of `encoded_block` and the one-body part `(1 - P_fail) w~_pm / Lambda` of `encoded_factors` (all with `rtol=0, atol=1e-12`). At `X_O_FAILS` `phase_grad` has two wires more than `s_u`, as Phase 3 passes it, so the amplitudes pin which wires the rotations add into. Then `OuterUnprepare` under the all-ones outcome schedule (every fix-up applied, strict mode) returns every register to `|0>` with amplitude 1.
- Compiled on lightning at `H2_STO3G_TEST` (19 wires): P(x_o, succ) is the alias distribution and `succ = 0` never occurs; prepare then unprepare restores every register and the gradient for two measurement seeds (2 tests).
- T4 (6 per commit, 2 `slow`): T and M equal `costs.outer_prepare` and `costs.outer_unprepare`, M = `K1 d1` for the unprepare, A as derived ("Design notes"), at both test presets and `X_O_FAILS`, and at `MID_NAPHTHALENE` with the naphthalene tables (`slow`; 92 and 38 with M = 42, each compile under 2 s and 290 MB); FeMoco on the closed forms: 210 = 30 + 156 + 15 + 9 with A = 108 = 24 + 84, and 69 = 9 + 36 + 24 with M = 96 and A = 81 = 15 + 36 + 30 (the pair's A is its T less `2 w_x + (K1 - 1) d1` = 90), the pieces rebuilt from the operator's own shapes, and `required_wires` at FeMoco summing to spec 10.3's 143-wire outer group (1 test).
- T6 and T5 (1 test): the verbose listing names `_outer_prepare`, `_outer_unprepare`, `_qroam_read` and `_qroam_erase`; `level="device"` leaves no dfthc operator.
- T2 (6 tests): both rules at `H2_STO3G_TEST` (odd part 1), `X_O_FAILS` (with a `phase_grad` two wires longer than `s_u`, as Phase 3 passes it) and the `k1 = 0`, `b_k1 = 1` shape.
- `read_shape`, `read_table` (one- and two-word entries) and `required_wires` by hand, including a `ke1 = 0` shape where the erase's work need is the largest (1 test); 11 constructor violations for each class (22 tests).
- Certificates (spec 12.5; `slow`, 4 tests, under 2 s together): both presets' outer shapes, `X_o = 5` with `ke1 = 2` (the fix-up leaf `[4, 8)` covers only 4 and 5; `phase_grad` two wires longer than `s_u`, whose extra wires must end unchanged) and `X_o = 6` with `k1 = 2`, random tables. Each run starts from every register in `|0>`, the pair's only legal pre-state; spec 12.5's general premise of random complex address and control amplitudes is carried by Task 1C.5's `QROAMRead` → `QROAMErase` certificates, which this certificate composes with the pair's other children. Each asserts on the decomposed tape that it has conditionals, that `nonlinear_conditionals` flags none, and that it measures `K1 d1` qubits; then it runs the zero and every unit schedule in strict mode, each consuming exactly `K1 d1` outcomes and ending in exactly the gradient state.

**Definition of done:** `46 passed` targeted (44 in the per-commit tier) and the certificates `4 passed`; the per-commit tier passes; `ruff check .` clean; the spec 9.0 docstring sections in order for both classes; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_outer_prepare.py`:

```python
"""OuterPrepare and OuterUnprepare (spec 9.7): the alias state, counts, rule selection, checks."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.alias import alias_probabilities
from dfthc.data.encoded import encoded_block, encoded_factors, uniform_failure_probability
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.sos import normalization
from dfthc.data.tables import WalkTables
from dfthc.ops.prepare import OuterPrepare, OuterUnprepare, _outer_prepare, _outer_unprepare
from dfthc.ops.qroam import QROAMErase, QROAMRead
from dfthc.params import (
    FEMOCO54,
    H2_631G_TEST,
    H2_STO3G_TEST,
    MID_NAPHTHALENE,
    WalkParams,
    lg,
    odd_part,
)
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator
from dfthc.testing.tape import decompose_for_simulation

X_O_FAILS = WalkParams(N=2, R=3, B=1, C=1, n_plus=1, b_rot=6, b_k1=3, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
"""X_o = 5 (l = 5): the uniform preparation of x_o fails, with weight 0.15 and a phase (the
shape of tests/data/test_encoded.py)."""
X_O_6_K2 = WalkParams(N=2, R=2, B=1, C=2, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=2, k2=1,
                      ke1=1, ke2=1)
"""X_o = 6 (l = 3) with K1 = 4: dead positions 6 and 7 in the read's last block."""
X_O_7_K0_B1 = WalkParams(N=3, R=2, B=1, C=2, n_plus=2, b_rot=4, b_k1=1, b_k2=2, s_u=3, k1=0,
                         k2=1, ke1=2, ke2=1)
"""X_o = 7 (l = 7) with k1 = 0 (no junk) and b_k1 = 1 (no comparator work wires)."""

CASES = {
    "sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "x-o-5": lambda: (X_O_FAILS, random_factors(2, 3, 1, 1, n_plus=1, seed=3)),
    "x-o-6-k2": lambda: (X_O_6_K2, random_factors(2, 2, 1, 2, n_plus=1, seed=4)),
    "x-o-7-k0-b1": lambda: (X_O_7_K0_B1, random_factors(3, 2, 1, 2, n_plus=2, seed=5)),
}


def case(name):
    p, factors = CASES[name]()
    return p, factors, WalkTables.from_factors(factors, p)


def outer_args(p, tables):
    return (p.X_o, tables.outer_alt, tables.outer_keep, p.b_k1, p.k1, p.ke1, p.s_u)


def layout(p, extra_grad=0):
    """Consecutive wires for every register, in the order of ``OuterPrepare.wire_argnames``;
    ``phase_grad`` gets ``extra_grad`` wires more than ``s_u``."""
    sizes = OuterPrepare.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)
    sizes["phase_grad"] += extra_grad
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def uniform_amplitudes(L, s):
    """FlaggedUniformPrep's amplitudes on t in [0, 2**lg(L)), from spec 9.2 with k recomputed
    from the spec formula (H on target, phase on top < l, H on top, phase on top = 0, H on top):
    every t < L carries the success amplitude, every t >= L the failure amplitude."""
    odd, n = odd_part(L), lg(odd_part(L))
    if n == 0:
        return np.full(L, 1 / np.sqrt(L), dtype=complex)
    angle = np.arccos(1 - 2 ** (odd.bit_length() - 1) / odd)
    k = int(np.floor(angle / (2 * np.pi) * 2**s)) % 2**s
    phase = np.exp(2j * np.pi * k / 2**s)
    h = np.ones((1, 1))
    for _ in range(n):
        h = np.kron(h, [[1, 1], [1, -1]]) / np.sqrt(2)
    top = h @ (np.where(np.arange(2**n) < odd, phase, 1) / np.sqrt(2**n))
    top[0] *= phase
    eta = lg(L) - n
    return np.repeat(h @ top, 2**eta) / np.sqrt(2**eta)


def register_values(state, register):
    """The big-endian value of ``register`` on every term of a sparse state."""
    values = np.zeros(len(state.amps), dtype=np.int64)
    for wire in register:
        values = (values << 1) | state.bits(wire).astype(np.int64)
    return values


def terms_without_gradient(state, wires):
    """``(names, {register values: amplitude})`` over every register but ``phase_grad``, with the
    catalysed phase-gradient factor divided out. Asserts that the state is a product with the
    phase-gradient state."""
    grad = wires["phase_grad"]
    names = [name for name in wires if name != "phase_grad"]
    columns = np.stack([register_values(state, wires[name]) for name in names], axis=1)
    phases = np.exp(2j * np.pi * register_values(state, grad) / 2 ** len(grad))
    groups = {}
    for key, amp in zip(map(tuple, columns.tolist()), state.amps * phases, strict=True):
        groups.setdefault(key, []).append(amp * np.sqrt(2 ** len(grad)))
    for key, amps in groups.items():
        assert len(amps) == 2 ** len(grad), key
        assert np.allclose(amps, amps[0], rtol=0, atol=1e-12), key
    return names, {key: amps[0] for key, amps in groups.items()}


@pytest.mark.parametrize("name", list(CASES))
def test_prepared_state_is_the_alias_distribution(name):
    """T3 (sparse simulator). Every term decodes to the drawn x_0: keep and junk as the read
    loaded them, less_than = [keep <= m], the alias swap, succ = [x_0 < X_o], work |0>, and the
    amplitude of the uniform preparation over sqrt(2**(b_k1 + 1)), its success phase and factor
    sqrt(1 - P_fail) included; every (x_0, m, sigma_1) occurs. The succ = 1 marginal is
    (1 - P_fail) alias_probabilities, and against the encoded model (spec 8.5) its total is
    Lambda / Lambda~ and its one-body part (1 - P_fail) w~_pm / Lambda. At x-o-5 ``phase_grad``
    has two wires more than ``s_u``, as Phase 3 passes it, so the amplitudes pin which wires the
    rotations add into. Then OuterUnprepare under the all-ones outcome schedule (every fix-up
    applied) returns every register to |0> with amplitude 1 (the certificates prove every
    outcome)."""
    p, factors, tables = case(name)
    wires, num_wires = layout(p, extra_grad=2 if name == "x-o-5" else 0)

    def qfunc():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        OuterPrepare(*outer_args(p, tables), **wires)

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    names, terms = terms_without_gradient(state, wires)
    shape = OuterPrepare.read_shape(p.X_o, p.b_k1, p.k1)
    table = OuterPrepare.read_table(p.X_o, tables.outer_alt, tables.outer_keep, p.b_k1)
    junk_bits = (p.K1 - 1) * p.d1
    amplitudes = uniform_amplitudes(p.X_o, p.s_u)
    scale = 1 / np.sqrt(2 ** (p.b_k1 + 1))
    marginal, drawn = np.zeros(p.X_o), set()
    for key, amp in terms.items():
        v = dict(zip(names, key, strict=True))
        swapped = v["less_than_x_o"]
        x0 = v["alt_x_o"] if swapped else v["x_o"]
        loaded = QROAMRead.loaded_bits(table, x0, **shape)
        target, junk = loaded >> junk_bits, loaded & (2**junk_bits - 1)
        alt0, keep0 = target >> p.b_k1, target & (2**p.b_k1 - 1)
        assert (v["keep_x_o"], v["junk_x_o"], v["work"]) == (keep0, junk, 0)
        assert swapped == int(keep0 <= v["m_x_o"])
        assert (v["x_o"], v["alt_x_o"]) == ((alt0, x0) if swapped else (x0, alt0))
        assert v["succ_x_o"] == int(x0 < p.X_o)
        assert abs(amp - amplitudes[x0] * scale) < 1e-12, (x0, amp)
        drawn.add((x0, v["m_x_o"], v["sigma_1"]))
        if x0 < p.X_o:
            marginal[v["x_o"]] += abs(amp) ** 2
    nonzero = np.count_nonzero(np.abs(amplitudes) > 1e-12)
    assert len(terms) == len(drawn) == nonzero * 2 ** (p.b_k1 + 1)
    fail = uniform_failure_probability(p.X_o, p.s_u)
    assert np.allclose(marginal,
                       (1 - fail) * alias_probabilities(tables.outer_alt, tables.outer_keep,
                                                        p.b_k1), rtol=0, atol=1e-12)
    lam = normalization(factors)
    _, lambda_tilde = encoded_block(p, tables, factors)
    enc = encoded_factors(p, tables, factors)
    assert abs(marginal.sum() - lam / lambda_tilde) < 1e-12
    assert np.allclose(marginal[p.RC:],
                       (1 - fail) * np.concatenate([enc.w_plus, enc.w_minus]) / lam,
                       rtol=0, atol=1e-12)

    def round_trip():
        qfunc()
        OuterUnprepare(*outer_args(p, tables), **wires)

    policy = ForcedPolicy([1] * (p.K1 * p.d1))
    final = SparseSimulator(num_wires, policy=policy, strict=True).run(
        decompose_for_simulation(round_trip).operations)
    _, restored = terms_without_gradient(final, wires)
    zero = (0,) * len(names)
    assert list(restored) == [zero] and abs(restored[zero] - 1) < 1e-12


def _lightning(num_wires, body, returns, seed=None):
    @qp.qjit(capture=True, seed=seed)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        body()
        return returns()

    return np.asarray(circuit())


@pytest.mark.compiled
def test_compiled_distribution_on_lightning():
    """T3 on the compiled path at H2_STO3G_TEST (19 wires): P(x_o = x, succ_x_o = 1) is the
    alias distribution, and succ_x_o = 0 never occurs (X_o = 4 is a power of two)."""
    p, _, tables = case("sto3g")
    wires, num_wires = layout(p)

    def body():
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        OuterPrepare(*outer_args(p, tables), **wires)

    probs = _lightning(num_wires, body,
                       lambda: qp.probs(wires=wires["x_o"] + wires["succ_x_o"]))
    probs = probs.reshape(2 ** lg(p.X_o), 2)
    expected = alias_probabilities(tables.outer_alt, tables.outer_keep, p.b_k1)
    assert np.allclose(probs[:, 1], expected, rtol=0, atol=1e-10)
    assert np.allclose(probs[:, 0], 0, rtol=0, atol=1e-10)


@pytest.mark.compiled
def test_lightning_round_trip_with_two_seeds():
    """Compiled OuterPrepare then OuterUnprepare at H2_STO3G_TEST: every register returns to |0>
    and the phase gradient is unchanged, for two measurement seeds (smoke runs; the certificates
    in test_certificates_prepare.py prove every outcome)."""
    p, _, tables = case("sto3g")
    wires, num_wires = layout(p)
    grad = wires["phase_grad"]

    def body():
        qp.PhaseGradientStatePrep(wires=grad)
        OuterPrepare(*outer_args(p, tables), **wires)
        OuterUnprepare(*outer_args(p, tables), **wires)

    gradient = np.exp(-2j * np.pi * np.arange(2 ** len(grad)) / 2 ** len(grad))
    expected = np.kron(np.kron(np.eye(2 ** grad[0])[0], gradient / np.sqrt(2 ** len(grad))),
                       np.eye(2 ** (num_wires - grad[-1] - 1))[0])
    for seed in range(2):
        state = _lightning(num_wires, body, qp.state, seed=seed)
        assert np.allclose(state, expected, rtol=0, atol=1e-10), seed


def expected_and_uncomputes(p, unprepare):
    """A: every TemporaryAND of the pair is uncomputed by measurement inside it (spec 9.24). The
    prepare uncomputes the uniform preparation's 2(n - 1) + 2(s_u - 1) ANDs and the read's
    iteration ANDs U; the unprepare the comparator's b_k1, the erase fix-up's T and the adjoint
    uniform preparation's 3(n - 1) + 2(s_u - 1) (n = lg(l_x); none of the uniform terms if
    l_x = 1)."""
    n, s = lg(odd_part(p.X_o)), p.s_u
    shape = OuterPrepare.read_shape(p.X_o, p.b_k1, p.k1)
    widths, live, width, k = shape.values()
    if unprepare:
        erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, p.ke1), k, width,
                                  p.ke1, False)
        return p.b_k1 + erase.toffoli + (3 * (n - 1) + 2 * (s - 1) if n else 0)
    read = costs.qroam_read(QROAMRead.visited_ranges(widths, live, k), k, width, False)
    return (2 * (n - 1) + 2 * (s - 1) if n else 0) + read.toffoli - (2**k - 1) * width


def _assert_counts(p, tables, unprepare):
    wires, num_wires = layout(p)
    op_class = OuterUnprepare if unprepare else OuterPrepare
    resources = compiled_resources(lambda: op_class(*outer_args(p, tables), **wires),
                                   num_wires=num_wires)
    expected = (costs.outer_unprepare if unprepare else costs.outer_prepare)(p)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measurements == (p.K1 * p.d1 if unprepare else 0)
    assert resources.measured_and_uncomputes == expected_and_uncomputes(p, unprepare)


@pytest.mark.compiled
@pytest.mark.parametrize("unprepare", [False, True], ids=["prepare", "unprepare"])
@pytest.mark.parametrize("name", ["sto3g", "631g", "x-o-5"])
def test_counts_match_the_closed_forms(name, unprepare):
    """T4 and the T6 signature: T and M equal dfthc.costs.outer_prepare / outer_unprepare, the
    unprepare measures every target and junk qubit (K1 d1), and A is as derived."""
    p, _, tables = case(name)
    _assert_counts(p, tables, unprepare)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("unprepare", [False, True], ids=["prepare", "unprepare"])
def test_counts_at_mid_naphthalene(unprepare):
    """T4 at MID_NAPHTHALENE with the naphthalene tables (nightly tier)."""
    p = MID_NAPHTHALENE
    _assert_counts(p, WalkTables.from_factors(load_factors("naphthalene"), p), unprepare)


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 on the closed forms, nothing compiled: prepare 30 + 156 + 15 + 9 = 210 with
    A = 24 + 84 = 108, unprepare 9 + 36 + 24 = 69 with M = 96 and A = 15 + 36 + 30 = 81; the pair
    uncomputes all it computes, so its A is its T less its Toffoli-only part 2 w_x + (K1 - 1) d1,
    as dfthc.costs.measured_and_uncomputes counts it. The operator's own read and erase shapes
    give the same pieces, and its registers make spec 10.3's 143-wire outer group."""
    p = FEMOCO54
    assert costs.outer_prepare(p) == costs.Cost(210)
    assert costs.outer_unprepare(p) == costs.Cost(69, 96)
    uncomputes = expected_and_uncomputes(p, False), expected_and_uncomputes(p, True)
    assert uncomputes == (108, 81)
    assert sum(uncomputes) == 210 + 69 - (2 * p.w_x + (p.K1 - 1) * p.d1)
    widths, live, width, k = OuterPrepare.read_shape(p.X_o, p.b_k1, p.k1).values()
    read = costs.qroam_read(QROAMRead.visited_ranges(widths, live, k), k, width, False)
    erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, p.ke1), k, width,
                              p.ke1, False)
    assert (read, erase) == (costs.Cost(156), costs.Cost(36, 96))
    uniform = costs.flagged_uniform_prep(p.X_o, p.s_u), costs.flagged_uniform_unprep(p.X_o, p.s_u)
    assert uniform == (costs.Cost(30), costs.Cost(24)) and (p.b_k1, p.w_x) == (15, 9)
    assert uniform[0].toffoli + read.toffoli + p.b_k1 + p.w_x == 210
    assert p.w_x + erase.toffoli + uniform[1].toffoli == 69
    sizes = OuterPrepare.required_wires(p.X_o, p.b_k1, p.k1, p.ke1, p.s_u)
    assert sizes == {"x_o": 9, "succ_x_o": 1, "succ_work_x_o": 6, "sigma_1": 1, "m_x_o": 15,
                     "less_than_x_o": 1, "cmp_work_x_o": 14, "alt_x_o": 9, "keep_x_o": 15,
                     "junk_x_o": 72, "phase_grad": 7, "work": 20}
    assert sum(sizes.values()) - sizes["phase_grad"] - sizes["work"] == 143


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the solver names the pinned rules of both classes and of the read and the erase; T5
    at level="device" (L-20)."""
    p, _, tables = case("x-o-5")
    wires, num_wires = layout(p)

    def program():
        OuterPrepare(*outer_args(p, tables), **wires)
        OuterUnprepare(*outer_args(p, tables), **wires)

    qp.analyze(verbose_program(program, num_wires=num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_outer_prepare", "_outer_unprepare", "_qroam_read", "_qroam_erase"} <= rules, rules
    compiled_resources(program, num_wires=num_wires, level="device")


@pytest.mark.parametrize("rule", ["prepare", "unprepare"])
@pytest.mark.parametrize("name", ["sto3g", "x-o-5", "x-o-7-k0-b1"])
def test_rules_are_valid(name, rule):
    """T2 in every branch of the hand-written resources: l_x = 1 (sto3g; no uniform work), l_x > 1
    (x-o-5), and k1 = 0 with b_k1 = 1 (no junk, no comparator work). At x-o-5 ``phase_grad`` has
    two wires more than ``s_u``, as when Phase 3 passes the whole register, so the declared
    resources hold only if the rule passes the uniform preparation ``phase_grad[:s_u]``."""
    p, _, tables = case(name)
    wires, _ = layout(p, extra_grad=2 if name == "x-o-5" else 0)
    op_class, rule_fn = ((OuterUnprepare, _outer_unprepare) if rule == "unprepare"
                         else (OuterPrepare, _outer_prepare))
    qp.testing.assert_valid_decomp_rule(op_class(*outer_args(p, tables), **wires), rule_fn,
                                        check_matrix=False)


def test_read_shape_and_table_by_hand():
    assert OuterPrepare.read_shape(5, 3, 1) == {"address_widths": (3,), "live_ranges": ((0, 5),),
                                                "width": 6, "block_bits": 1}
    assert OuterPrepare.read_table(5, (1, 0, 4, 2, 3), (7, 0, 5, 1, 2), 3) == (
        (0, (15,)), (1, (0,)), (2, (37,)), (3, (17,)), (4, (26,)))
    assert OuterPrepare.read_table(3, (2, 0, 1), (2**31, 5, 0), 32) == (
        (0, (2**31, 2)), (1, (5, 0)), (2, (0, 1)))
    assert OuterPrepare.required_wires(4, 2, 1, 1, 3) == {
        "x_o": 2, "succ_x_o": 1, "succ_work_x_o": 0, "sigma_1": 1, "m_x_o": 2,
        "less_than_x_o": 1, "cmp_work_x_o": 1, "alt_x_o": 2, "keep_x_o": 2, "junk_x_o": 4,
        "phase_grad": 3, "work": 0}
    assert OuterUnprepare.required_wires(4, 2, 1, 1, 3) == OuterPrepare.required_wires(4, 2, 1, 1,
                                                                                       3)
    # ke1 = 0: the erase's fix-up iterates both address bits, so its work need (1) is the largest
    assert OuterPrepare.required_wires(4, 2, 1, 0, 3)["work"] == 1


BASE = {"X_o": 5, "alt": (1, 0, 4, 2, 3), "keep": (7, 0, 5, 1, 2), "b_k1": 3, "k1": 1, "ke1": 1,
        "s_u": 3}


@pytest.mark.parametrize(("change", "match"), [
    ({"X_o": 1}, r"X_o must lie in \[2, None\]"),
    ({"b_k1": 0}, r"b_k1 must lie in \[1, None\]"),
    ({"k1": 4}, r"k1 must lie in \[0, 3\]"),
    ({"ke1": 4}, r"ke1 must lie in \[0, 3\]"),
    ({"k1": 0, "ke1": 3}, "needs 8 wires, more than the 6 target and junk wires"),
    ({"s_u": 2}, r"s_u must lie in \[3, None\]"),
    ({"alt": (1, 0, 5, 2, 3)}, r"alt entries must lie in \[0, 5\), got 5"),
    ({"keep": (8, 0, 5, 1, 2)}, r"keep entries must lie in \[0, 8\), got 8"),
    ({"alt": (1, 0, 4, 2)}, r"alt must be a table of ints of shape \(5,\)"),
    ({"junk_x_o": 5}, "'junk_x_o' has 5 wires; expected 6"),
    ({"phase_grad": 2}, "'phase_grad' has 2 wires; expected at least 3"),
])
@pytest.mark.parametrize("op_class", [OuterPrepare, OuterUnprepare])
def test_constructors_validate(op_class, change, match):
    sizes = OuterPrepare.required_wires(5, 3, 1, 1, 3)
    sizes |= {name: value for name, value in change.items() if name in sizes}
    wires = {name: list(range(100 * i, 100 * i + size))
             for i, (name, size) in enumerate(sizes.items())}
    args = BASE | {name: value for name, value in change.items() if name in BASE}
    with pytest.raises(ValueError, match=match):
        op_class(**args, **wires)
```

Create `tests/ops/test_certificates_prepare.py`:

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
"""

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional

from dfthc.ops.prepare import OuterPrepare, OuterUnprepare
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
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_outer_prepare.py tests/ops/test_certificates_prepare.py -q`
Expected: FAIL at collection, `2 errors`: `ModuleNotFoundError: No module named 'dfthc.ops.prepare'`.

- [ ] **Step 3: Write the skeleton and the outer classes**

Create `src/dfthc/ops/prepare.py`:

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
address, so the swap is undone first, and the comparator reads ``keep``, so its adjoint comes
before the erase; nothing between a prepare and its unprepare may write ``succ``, ``succ_work`` or
``cmp_work``.

**Tables.** A read entry is one big-endian integer (spec 6.7) of ``width`` bits, stored as 32-bit
words, least significant first (L-24): ``(alt << b_k1) | keep`` for the outer read
(:meth:`OuterPrepare.read_table`). A prepare packs its compilable tables into the read's dynamic
``leaf_words`` inside its rule (:meth:`dfthc.ops.qroam.QROAMRead.leaf_words_from_table`); its
unprepare passes the same table, in the same entry layout, to
:class:`~dfthc.ops.qroam.QROAMErase`, whose masks are therefore exactly what the read loaded. The
erase's one-hot register is the reset ``target ‖ junk``, so ``2**onehot_bits`` must not exceed
``2**block_bits * width`` (the constructors check it: no walk instance needs ``extra_onehot``).

**Rules.** The rules are straight-line and their children are composite keys, so they declare
their resources by hand; T2 runs them at a size in every branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import AbstractArray, Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.qroam import WORD, WORD_DTYPE, QROAMErase, QROAMRead
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

    **Pre- and postconditions.** As in the table. Until :class:`OuterUnprepare`, nothing may
    write ``succ_x_o``, ``succ_work_x_o``, ``less_than_x_o``, ``cmp_work_x_o``, ``m_x_o``,
    ``alt_x_o``, ``keep_x_o`` or ``junk_x_o``, and ``x_o`` may be used only as a control or
    changed by operations that are undone before the unprepare.

    **Decomposition** (spec 9.7). ``FlaggedUniformPrep(X_o, s_u)`` on ``x_o``, ``succ_x_o``,
    ``succ_work_x_o``, ``phase_grad[:s_u]``; ``H`` on ``m_x_o`` and ``sigma_1``; ``QROAMRead`` of
    :meth:`read_table` at ``x_o`` (shape :meth:`read_shape`, no enable) into
    ``alt_x_o ‖ keep_x_o`` and ``junk_x_o``, its ``leaf_words`` built here from the table;
    ``LeftQuantumComparator(keep_x_o, m_x_o, less_than_x_o, cmp_work_x_o, "<=")``; ``w_x``
    ``CSWAP``\s of ``alt_x_o`` and ``x_o`` controlled by ``less_than_x_o``. Each child gets the
    start of ``work`` it needs.

    **Adjoint and control.** Never adjointed or controlled: the read has loops and data
    conditionals (spec 7.4 rules 4 and 6). The inverse is the operator :class:`OuterUnprepare`.

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
    is itself the adjoint class of :class:`OuterPrepare`.

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
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_outer_prepare.py -q`
Expected: `46 passed` (16 to 40 s in the replays with 2 workers, depending on the machine's load, including the two `MID_NAPHTHALENE` compiles; the slowest per-commit tests are the two-seed lightning round trip, about 6 s, and the verbose and device-level compiles, about 8 s).

Run: `scripts/test.sh tests/ops/test_certificates_prepare.py -q`
Expected: `4 passed` (nightly tier, run targeted here).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `700 passed` (656 + 44; see "Expected test counts").

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/prepare.py tests/ops/test_outer_prepare.py \
        tests/ops/test_certificates_prepare.py
git commit -m "Add OuterPrepare and OuterUnprepare with the outer certificates"
```

---

### Task 2D.2: `InnerPrepare`, `InnerUnprepare` and `InnerSigns`

**Task id:** `2D.2` · **Branch:** `phase2/2D.2` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 2D.1 (same module and certificate file; reuses its skeleton `_AliasSampling`)

**Files:**
- Modify: `src/dfthc/ops/prepare.py` (replace it with the version below; it contains every line of Task 2D.1's version, in order, and adds the module docstring's "Inner read" and "Signs" paragraphs and the `InnerPrepare`, `InnerUnprepare` and `InnerSigns` sections)
- Modify: `tests/ops/test_certificates_prepare.py` (replace it with the version below: the outer certificate is unchanged; a docstring paragraph, the inner classes in the import and the inner certificate are new)
- Modify: `src/dfthc/costs.py` (add `inner_signs` after `inner_unprepare`)
- Create: `tests/ops/test_inner_prepare.py`

**Interfaces:**
- Consumes: Task 2D.1's `prepare.py` (`_AliasSampling`, `_join`, `_words`, `_check_table`, `_check_onehot`); everything Task 2D.1 consumes; `dfthc.costs.inner_prepare`, `inner_unprepare`; `dfthc.data.encoded.dfthcbe_failure_signs`, `our_failure_signs`; `dfthc.data.sos.inner_weights`.
- Produces:
  - `dfthc.costs.inner_signs() -> Cost` (T = M = 0);
  - in `dfthc.ops.prepare`: `InnerPrepare(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2, k2, ke2, s_u, x_o, b, succ_b, succ_work_b, sigma_0, m_b, less_than_b, cmp_work_b, alt_b, keep_b, sgn, d_sgn, r, junk_b, phase_grad, work)`: compilable `N, R, B, C >= 1`; the four tables of shape `(R C, B + 1)` with `alt` in `[0, B + 1)`, `keep` in `[0, 2**b_k2)`, `sgn_table` and `d_sgn_table` bits (named so because `sgn` and `d_sgn` are wire registers); `b_k2 >= 1`; `0 <= k2 <= lg(B + 1)`; `0 <= ke2 <= lg(R C)` with `2**(ke2 + lg(B + 1)) <= 2**k2 d2`; `s_u >= 3`; static `required_wires(N, R, B, C, b_k2, k2, ke2, s_u) -> dict[str, int]`, `read_shape(N, R, B, C, b_k2, k2) -> dict` and `read_table(N, R, B, C, alt, keep, sgn_table, d_sgn_table, b_k2) -> tuple` (`(x << lg(B + 1) | b, words)` with entry `alt ‖ keep ‖ sgn ‖ d_sgn ‖ r`, `r = x // C`); the rule `_inner_prepare`;
  - `InnerUnprepare` with the same arguments and `required_wires`; the rule `_inner_unprepare` (spec 9.8, the erase's one-hot over `ke2 + lg(B + 1)` bits);
  - `InnerSigns(sgn, d_sgn, less_than_b)` with `required_wires()`, the rule `_inner_signs` and the pinned `"Adjoint(InnerSigns)"` rule `_adjoint_inner_signs` (the operator itself).

**Acceptance tests** (50 tests in `tests/ops/test_inner_prepare.py`, 2 of them `slow`; 4 more `slow` tests in `tests/ops/test_certificates_prepare.py`):
- Spec 12.1 T3 items 2 and 3 on the sparse simulator (6 tests): (B+1, k2) = (3, 1), (5, 1), (6, 2), (7, 0) with a 3-bit `x_o` (SF rows [0, 2), one-body rows 2 to 4, failed values 5 to 7; `ke2 = 0` at (7, 0), where 16 one-hot wires would exceed the 8 target wires), and both test presets, tables from `WalkTables.from_factors`. With `H` on `x_o`, the uniform preparation drawing every `b_0` and the Hadamards on `m_b`, every basis input (x_o, b_0, m_b) appears once per `sigma_0`, with `succ_b = [b_0 <= B]`. After the read, keep test, swap and `InnerSigns`, each term decodes as the read loaded it (`r = x_o // C` where a live entry was loaded, 0 elsewhere), and its amplitude equals the uniform preparation's for `b_0` over `sqrt(2**(w_x + b_k2 + 1))` times the sign of the selected weight (from the factors) for a valid `b_0` on an SF row, and +1 otherwise. The per-row signed weights equal the encoded model's `a_b` (`inner_weights(encoded_factors(...)) / sqrt(4 Lambda p~_rc)`, the failed weight on `b = B`); the failure signs equal `our_failure_signs` (+1), and equal `dfthcbe_failure_signs` where `2**w_b <= ceil((B + 1) / K2) K2` ((3, 1), (6, 2), presets) and differ from them by more than 1e-2 at (5, 1) and (7, 0). At (5, 1) `phase_grad` has two wires more than `s_u`, as Phase 3 passes it, so the amplitudes pin which wires the rotations add into. Then `InnerPrepare` and `InnerUnprepare` under the all-ones outcome schedule (every fix-up applied, strict mode) and the alternating one (which also sees a permutation of the erase masks' bits) leave `x_o` in its uniform superposition and every other register in `|0>`.
- T4 (4 per commit, 2 `slow`): T and M equal `costs.inner_prepare` and `costs.inner_unprepare`, M = `K2 d2`, A as derived, at both test presets and at `MID_NAPHTHALENE` (`slow`; 185 and 53 with M = 92, each compile under 2.3 s and 320 MB); FeMoco on the closed forms: 971 = 18 + 933 + 15 + 5 with A = 559 = 16 + 543, 217 = 5 + 196 + 16 with M = 416 and A = 229 = 15 + 196 + 18 (the pair's A is its T less `2 w_b + (K2 - 1) d2` = 400), `inner_signs()` = 0 (its A = 0 is the compiled `InnerSigns` count, the same at every size), `required_wires` summing to spec 10.3's 455-wire inner group (1 test).
- T6 and T5 (1 test): the verbose listing of `InnerPrepare`, `InnerSigns`, `InnerUnprepare` names `_inner_prepare`, `_inner_signs`, `_inner_unprepare`, `_qroam_read`, `_qroam_erase`; `level="device"` leaves no dfthc operator.
- T2 (7 tests): both rules at the `H2_STO3G_TEST` shape (with a `phase_grad` two wires longer than `s_u`), the (7, 0) shape (`k2 = 0`) and `R = 1`, `B + 1 = 4`, `b_k2 = 1` (empty `r`, odd part 1, no comparator work); `InnerSigns`' base and adjoint rules.
- `InnerSigns` (3 tests): the phase `(-1)**(sgn XOR (less_than_b AND d_sgn))` on every basis input on lightning, plain and through the adjoint; T = M = A = 0 against `costs.inner_signs()` with one `PauliZ` and one `CZ` per application, both rules chosen, `level="device"` closure.
- `read_shape`, `read_table` and `required_wires` by hand, including `read_table` at `C = 3`, where `r = x_o // C` is not a bit field of `x_o` (1 test); 12 constructor violations for each prepare class (24 tests) and one for `InnerSigns`.
- Certificates (`slow`, 4 tests, under 4 s together): both presets' inner shapes, rows [0, 3) with (B+1, k2) = (6, 2) and row 0 zero (rank 0, so its masks are zero; the last fix-up leaf, `x_o` in {2, 3}, is partly covered), and rows [0, 2) with (B+1, k2) = (5, 1) (dead position `b = 5`; every fix-up leaf covers `b < 6` only); random tables; each run starts from a random complex amplitude on every value of `x_o`. Same assertions as Task 2D.1's.

**Definition of done:** `50 passed` targeted (48 in the per-commit tier), the certificates `8 passed`, Task 2D.1's per-commit tests still `44 passed`; the per-commit tier passes; `ruff check .` clean; the spec 9.0 docstring sections in order for the three classes; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_inner_prepare.py`:

```python
"""InnerPrepare, InnerUnprepare and InnerSigns (spec 9.8, 9.9): the sign of every basis input,
the encoded model, failure signs, counts, rule selection, checks."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.alias import alias_probabilities
from dfthc.data.encoded import (
    dfthcbe_failure_signs,
    encoded_factors,
    our_failure_signs,
    uniform_failure_probability,
)
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.sos import inner_weights, normalization
from dfthc.data.tables import WalkTables
from dfthc.ops.prepare import (
    InnerPrepare,
    InnerSigns,
    InnerUnprepare,
    _adjoint_inner_signs,
    _inner_prepare,
    _inner_signs,
    _inner_unprepare,
)
from dfthc.ops.qroam import QROAMErase, QROAMRead
from dfthc.params import (
    FEMOCO54,
    H2_631G_TEST,
    H2_STO3G_TEST,
    MID_NAPHTHALENE,
    WalkParams,
    lg,
    odd_part,
)
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator
from dfthc.testing.tape import decompose_for_simulation

FAILURE_SIGN_MARGIN = 1e-2
"""dfthcbe's failure signs differ from ours by more than this where they differ at all
(tests/data/test_encoded.py: every nonzero difference is a multiple of 1/6 at (B + 1, k2) =
(5, 1) and of 1/2 at (7, 0))."""


def failed_address_case(B, k2, ke2=1, seed=5):
    """A spec 12.1 T3 shape: x_o of 3 bits with SF rows [0, 2) (R = 2, C = 1), one-body rows 2 to
    4 (N = 3) and failed values 5 to 7; (B + 1, k2) as given; b_k2 = 2 and s_u = 3."""
    params = WalkParams(N=3, R=2, B=B, C=1, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=1,
                        k2=k2, ke1=1, ke2=ke2)
    return params, random_factors(3, 2, B, 1, n_plus=1, seed=seed)


# Both test presets, and the failed-address shapes (B + 1, k2) = (3, 1), (5, 1), (6, 2), (7, 0)
# of spec 12.1 (the (28, 4) shape is for the read alone). At (7, 0) the one-hot over ke2 = 1 and
# all three bits of b would need 16 wires, more than the 8 of target and junk, so ke2 = 0.
CASES = {
    "sto3g": lambda: (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
    "631g": lambda: (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
    "3-1": lambda: failed_address_case(2, 1),
    "5-1": lambda: failed_address_case(4, 1),
    "6-2": lambda: failed_address_case(5, 2),
    "7-0": lambda: failed_address_case(6, 0, ke2=0),
}


def case(name):
    p, factors = CASES[name]()
    return p, factors, WalkTables.from_factors(factors, p)


def inner_args(p, tables):
    return (p.N, p.R, p.B, p.C, tables.inner_alt, tables.inner_keep, tables.inner_sgn,
            tables.inner_d_sgn, p.b_k2, p.k2, p.ke2, p.s_u)


def structure(p):
    return (p.N, p.R, p.B, p.C, p.b_k2, p.k2, p.ke2, p.s_u)


def layout(sizes):
    """Consecutive wires for every register, in the order of ``sizes``."""
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def uniform_amplitudes(L, s):
    """FlaggedUniformPrep's amplitudes on t in [0, 2**lg(L)), from spec 9.2 with k recomputed
    from the spec formula (H on target, phase on top < l, H on top, phase on top = 0, H on top):
    every t < L carries the success amplitude, every t >= L the failure amplitude."""
    odd, n = odd_part(L), lg(odd_part(L))
    if n == 0:
        return np.full(L, 1 / np.sqrt(L), dtype=complex)
    angle = np.arccos(1 - 2 ** (odd.bit_length() - 1) / odd)
    k = int(np.floor(angle / (2 * np.pi) * 2**s)) % 2**s
    phase = np.exp(2j * np.pi * k / 2**s)
    h = np.ones((1, 1))
    for _ in range(n):
        h = np.kron(h, [[1, 1], [1, -1]]) / np.sqrt(2)
    top = h @ (np.where(np.arange(2**n) < odd, phase, 1) / np.sqrt(2**n))
    top[0] *= phase
    eta = lg(L) - n
    return np.repeat(h @ top, 2**eta) / np.sqrt(2**eta)


def register_values(state, register):
    """The big-endian value of ``register`` on every term of a sparse state."""
    values = np.zeros(len(state.amps), dtype=np.int64)
    for wire in register:
        values = (values << 1) | state.bits(wire).astype(np.int64)
    return values


def terms_without_gradient(state, wires):
    """``(names, {register values: amplitude})`` over every register but ``phase_grad``, with the
    catalysed phase-gradient factor divided out. Asserts that the state is a product with the
    phase-gradient state."""
    grad = wires["phase_grad"]
    names = [name for name in wires if name != "phase_grad"]
    columns = np.stack([register_values(state, wires[name]) for name in names], axis=1)
    phases = np.exp(2j * np.pi * register_values(state, grad) / 2 ** len(grad))
    groups = {}
    for key, amp in zip(map(tuple, columns.tolist()), state.amps * phases, strict=True):
        groups.setdefault(key, []).append(amp * np.sqrt(2 ** len(grad)))
    for key, amps in groups.items():
        assert len(amps) == 2 ** len(grad), key
        assert np.allclose(amps, amps[0], rtol=0, atol=1e-12), key
    return names, {key: amps[0] for key, amps in groups.items()}


@pytest.mark.parametrize("name", list(CASES))
def test_every_basis_input_gets_the_sign_of_its_selected_weight(name):
    """Spec 12.1 T3 items 2 and 3 (sparse simulator). With H on x_o, the uniform preparation of
    b and the Hadamards on m_b, every basis input (x_o, b_0, m_b) occurs once per sigma_0, with
    succ_b = [b_0 <= B]. After the read, keep test, alias swap and InnerSigns, each term decodes
    as the read loaded it, and its amplitude is the uniform preparation's for b_0 (success phase
    and factor included) over sqrt(2**(w_x + b_k2 + 1)), times the sign of the selected weight
    for a valid b_0 on an SF row and +1 otherwise. Summed per row, the signed weights equal the
    encoded model's inner amplitudes (spec 8.5), the failure signs equal ours (+1), and they
    equal dfthcbe's exactly when 2**w_b <= ceil((B + 1) / K2) K2, else differ as documented. At
    5-1 ``phase_grad`` has two wires more than ``s_u``, as Phase 3 passes it, so the amplitudes
    pin which wires the rotations add into. Then InnerPrepare and InnerUnprepare under the
    all-ones outcome schedule (every fix-up applied) and the alternating one (which also sees a
    permutation of the erase masks' bits, to which the all-ones parity is blind) leave ``x_o`` in
    its uniform superposition and every other register in |0> (the certificates prove every
    outcome)."""
    p, factors, tables = case(name)
    sizes = InnerPrepare.required_wires(*structure(p))
    sizes["phase_grad"] += 2 if name == "5-1" else 0
    wires, num_wires = layout(sizes)

    def prepare():
        for wire in wires["x_o"]:
            qp.H(wire)
        qp.PhaseGradientStatePrep(wires=wires["phase_grad"])
        InnerPrepare(*inner_args(p, tables), **wires)

    def qfunc():
        prepare()
        InnerSigns(wires["sgn"], wires["d_sgn"], wires["less_than_b"])

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    names, terms = terms_without_gradient(state, wires)
    shape = InnerPrepare.read_shape(p.N, p.R, p.B, p.C, p.b_k2, p.k2)
    table = InnerPrepare.read_table(*inner_args(p, tables)[:9])
    junk_bits = (p.K2 - 1) * p.d2
    amplitudes = uniform_amplitudes(p.B + 1, p.s_u)
    scale = 1 / np.sqrt(2 ** (p.w_x + p.b_k2 + 1))
    signs = np.sign(inner_weights(factors))
    selected, failed, drawn = np.zeros((p.RC, p.B + 1)), np.zeros(p.RC), set()
    for key, amp in terms.items():
        v = dict(zip(names, key, strict=True))
        x, swapped = v["x_o"], v["less_than_b"]
        b0 = v["alt_b"] if swapped else v["b"]
        valid = x < p.RC and b0 <= p.B
        loaded = QROAMRead.loaded_bits(table, (x << p.w_b) | b0, **shape)
        target, junk = loaded >> junk_bits, loaded & (2**junk_bits - 1)
        alt, keep = target >> (p.b_k2 + 2 + p.w_r), (target >> (2 + p.w_r)) % 2**p.b_k2
        columns = ((target >> (1 + p.w_r)) & 1, (target >> p.w_r) & 1, target % 2**p.w_r)
        assert (v["keep_b"], v["sgn"], v["d_sgn"], v["r"]) == (keep, *columns)
        assert (v["junk_b"], v["work"]) == (junk, 0)
        assert v["r"] == (x // p.C if valid else 0)
        assert swapped == int(keep <= v["m_b"])
        assert (v["b"], v["alt_b"]) == ((alt, b0) if swapped else (b0, alt))
        assert v["succ_b"] == int(b0 <= p.B)
        sign = (amp / (amplitudes[b0] * scale)).real
        assert abs(amp - (signs[x, v["b"]] if valid else 1) * amplitudes[b0] * scale) < 1e-12
        drawn.add((x, b0, v["m_b"], v["sigma_0"]))
        if valid:
            selected[x, v["b"]] += sign * abs(amp) ** 2 * 2**p.w_x
        elif x < p.RC:
            failed[x] += sign * abs(amp) ** 2 * 2**p.w_x
    nonzero = np.count_nonzero(np.abs(amplitudes) > 1e-12)
    assert len(terms) == len(drawn) == 2**p.w_x * nonzero * 2 ** (p.b_k2 + 1)
    fail = uniform_failure_probability(p.B + 1, p.s_u)
    assert fail > 0
    outer = alias_probabilities(tables.outer_alt, tables.outer_keep, p.b_k1)[:p.RC]
    model = (inner_weights(encoded_factors(p, tables, factors))
             / np.sqrt(4 * normalization(factors) * outer)[:, None])
    selected[:, p.B] += failed
    assert np.allclose(selected, model, rtol=0, atol=1e-12)
    assert np.allclose(failed / fail, our_failure_signs(p, tables), rtol=0, atol=1e-12)
    port = dfthcbe_failure_signs(p, tables)
    if 2**p.w_b <= -(-(p.B + 1) // p.K2) * p.K2:
        assert np.allclose(port, failed / fail, rtol=0, atol=1e-12)
    else:
        assert np.abs(port - failed / fail).max() > FAILURE_SIGN_MARGIN

    def round_trip():
        prepare()
        InnerUnprepare(*inner_args(p, tables), **wires)

    operations = decompose_for_simulation(round_trip).operations
    measured, rest = p.K2 * p.d2, (0,) * (len(names) - 1)
    for schedule in ([1] * measured, [i % 2 for i in range(measured)]):
        final = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True).run(
            operations)
        _, restored = terms_without_gradient(final, wires)
        assert names[0] == "x_o" and sorted(restored) == [(x, *rest) for x in range(2**p.w_x)]
        assert all(abs(amp - 2 ** (-p.w_x / 2)) < 1e-12 for amp in restored.values())


def expected_and_uncomputes(p, unprepare):
    """A: every TemporaryAND of the pair is uncomputed by measurement inside it (spec 9.24). The
    prepare uncomputes the uniform preparation's 2(n - 1) + 2(s_u - 1) ANDs and the read's
    iteration ANDs U; the unprepare the comparator's b_k2, the erase fix-up's T and the adjoint
    uniform preparation's 3(n - 1) + 2(s_u - 1) (n = lg(l_b); none of the uniform terms if
    l_b = 1)."""
    n, s = lg(odd_part(p.B + 1)), p.s_u
    widths, live, width, k = InnerPrepare.read_shape(p.N, p.R, p.B, p.C, p.b_k2, p.k2).values()
    if unprepare:
        ob = p.ke2 + p.w_b
        erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width,
                                  ob, False)
        return p.b_k2 + erase.toffoli + (3 * (n - 1) + 2 * (s - 1) if n else 0)
    read = costs.qroam_read(QROAMRead.visited_ranges(widths, live, k), k, width, False)
    return (2 * (n - 1) + 2 * (s - 1) if n else 0) + read.toffoli - (2**k - 1) * width


def _assert_counts(p, tables, unprepare):
    wires, num_wires = layout(InnerPrepare.required_wires(*structure(p)))
    op_class = InnerUnprepare if unprepare else InnerPrepare
    resources = compiled_resources(lambda: op_class(*inner_args(p, tables), **wires),
                                   num_wires=num_wires)
    expected = (costs.inner_unprepare if unprepare else costs.inner_prepare)(p)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measurements == (p.K2 * p.d2 if unprepare else 0)
    assert resources.measured_and_uncomputes == expected_and_uncomputes(p, unprepare)


@pytest.mark.compiled
@pytest.mark.parametrize("unprepare", [False, True], ids=["prepare", "unprepare"])
@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_counts_match_the_closed_forms(name, unprepare):
    """T4 and the T6 signature: T and M equal dfthc.costs.inner_prepare / inner_unprepare, the
    unprepare measures every target and junk qubit (K2 d2), and A is as derived."""
    p, _, tables = case(name)
    _assert_counts(p, tables, unprepare)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("unprepare", [False, True], ids=["prepare", "unprepare"])
def test_counts_at_mid_naphthalene(unprepare):
    """T4 at MID_NAPHTHALENE with the naphthalene tables (nightly tier)."""
    p = MID_NAPHTHALENE
    _assert_counts(p, WalkTables.from_factors(load_factors("naphthalene"), p), unprepare)


def test_femoco_counts_on_the_closed_forms():
    """FeMoco-54 on the closed forms, nothing compiled: prepare 18 + 933 + 15 + 5 = 971 with
    A = 16 + 543 = 559, unprepare 5 + 196 + 16 = 217 with M = 416 and A = 15 + 196 + 18 = 229;
    the pair's A is its T less its Toffoli-only part 2 w_b + (K2 - 1) d2, as
    dfthc.costs.measured_and_uncomputes counts it. InnerSigns costs 0 (its A = 0 is the compiled
    count of test_inner_signs_counts_rules_and_device_level: it has three wires at every size).
    The operator's own read and erase shapes give the same pieces, and its registers make spec
    10.3's 455-wire inner group."""
    p = FEMOCO54
    assert costs.inner_prepare(p) == costs.Cost(971)
    assert costs.inner_unprepare(p) == costs.Cost(217, 416)
    assert costs.inner_signs() == costs.Cost(0)
    uncomputes = expected_and_uncomputes(p, False), expected_and_uncomputes(p, True)
    assert uncomputes == (559, 229)
    assert sum(uncomputes) == 971 + 217 - (2 * p.w_b + (p.K2 - 1) * p.d2)
    widths, live, width, k = InnerPrepare.read_shape(p.N, p.R, p.B, p.C, p.b_k2, p.k2).values()
    ob = p.ke2 + p.w_b
    read = costs.qroam_read(QROAMRead.visited_ranges(widths, live, k), k, width, False)
    erase = costs.qroam_erase(QROAMErase.fixup_high_ranges(widths, live, k, ob), k, width, ob,
                              False)
    assert (read, erase) == (costs.Cost(933), costs.Cost(196, 416))
    uniform = costs.flagged_uniform_prep(p.B + 1, p.s_u), costs.flagged_uniform_unprep(p.B + 1,
                                                                                       p.s_u)
    assert uniform == (costs.Cost(18), costs.Cost(16)) and (p.b_k2, p.w_b) == (15, 5)
    assert uniform[0].toffoli + read.toffoli + p.b_k2 + p.w_b == 971
    assert p.w_b + erase.toffoli + uniform[1].toffoli == 217
    sizes = InnerPrepare.required_wires(*structure(p))
    assert sizes == {"x_o": 9, "b": 5, "succ_b": 1, "succ_work_b": 2, "sigma_0": 1, "m_b": 15,
                     "less_than_b": 1, "cmp_work_b": 14, "alt_b": 5, "keep_b": 15, "sgn": 1,
                     "d_sgn": 1, "r": 4, "junk_b": 390, "phase_grad": 7, "work": 16}
    assert sum(sizes.values()) - sizes["x_o"] - sizes["phase_grad"] - sizes["work"] == 455


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the solver names the pinned rules of the three classes and of the read and the erase;
    T5 at level="device" (L-20)."""
    p, _, tables = case("sto3g")
    wires, num_wires = layout(InnerPrepare.required_wires(*structure(p)))

    def program():
        InnerPrepare(*inner_args(p, tables), **wires)
        InnerSigns(wires["sgn"], wires["d_sgn"], wires["less_than_b"])
        InnerUnprepare(*inner_args(p, tables), **wires)

    qp.analyze(verbose_program(program, num_wires=num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_inner_prepare", "_inner_signs", "_inner_unprepare", "_qroam_read",
            "_qroam_erase"} <= rules, rules
    compiled_resources(program, num_wires=num_wires, level="device")


def random_inner_args(N, R, B, C, b_k2, k2, ke2, s_u, seed=7):
    rng = np.random.default_rng(seed)
    shape = (R * C, B + 1)
    columns = [rng.integers(0, B + 1, shape), rng.integers(0, 2**b_k2, shape),
               rng.integers(0, 2, shape), rng.integers(0, 2, shape)]
    tables = [tuple(tuple(int(v) for v in row) for row in column) for column in columns]
    return (N, R, B, C, *tables, b_k2, k2, ke2, s_u)


# T2 shapes: the sto3g shape (l_b = 3, k2 = 1; with two phase_grad wires more than s_u); the
# (7, 0) shape (k2 = 0: no junk; ke2 = 0); and R = 1, B + 1 = 4, b_k2 = 1 (an empty r register,
# l_b = 1: no uniform work, and no comparator work).
RULE_SHAPES = {"sto3g": (2, 2, 2, 1, 2, 1, 1, 3), "7-0": (3, 2, 6, 1, 2, 0, 0, 3),
               "r1-l1-b1": (2, 1, 3, 2, 1, 2, 1, 3)}


@pytest.mark.parametrize("rule", ["prepare", "unprepare"])
@pytest.mark.parametrize("name", list(RULE_SHAPES))
def test_rules_are_valid(name, rule):
    """T2 in every branch of the hand-written resources. At the sto3g shape ``phase_grad`` has two
    wires more than ``s_u``, as when Phase 3 passes the whole register, so the declared resources
    hold only if the rule passes the uniform preparation ``phase_grad[:s_u]``."""
    args = random_inner_args(*RULE_SHAPES[name])
    sizes = InnerPrepare.required_wires(*args[:4], *args[8:])
    sizes["phase_grad"] += 2 if name == "sto3g" else 0
    wires, _ = layout(sizes)
    op_class, rule_fn = ((InnerUnprepare, _inner_unprepare) if rule == "unprepare"
                         else (InnerPrepare, _inner_prepare))
    qp.testing.assert_valid_decomp_rule(op_class(*args, **wires), rule_fn, check_matrix=False)


def test_inner_signs_rules_are_valid():
    """T2 for InnerSigns' base rule and its pinned adjoint rule (tape mode)."""
    op = InnerSigns([0], [1], [2])
    qp.testing.assert_valid_decomp_rule(op, _inner_signs, check_matrix=False)
    qp.testing.assert_valid_decomp_rule(qp.adjoint(op), _adjoint_inner_signs, check_matrix=False)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["plain", "adjoint"])
def test_inner_signs_phase_on_every_basis_input(adjoint):
    """T3 on lightning: on H|000> of (sgn, d_sgn, less_than_b), the phase
    (-1)**(sgn XOR (less_than_b AND d_sgn)) on every basis input; the adjoint is the same."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    def circuit():
        for wire in range(3):
            qp.H(wire)
        op = InnerSigns([0], [1], [2])
        if adjoint:
            qp.adjoint(op)
        return qp.state()

    expected = [(-1) ** (sgn ^ (less_than & d_sgn)) / np.sqrt(8)
                for sgn in (0, 1) for d_sgn in (0, 1) for less_than in (0, 1)]
    assert np.allclose(np.asarray(circuit()), expected, rtol=0, atol=1e-12)


@pytest.mark.compiled
def test_inner_signs_counts_rules_and_device_level(capfd):
    """T4: T = M = A = 0 (dfthc.costs.inner_signs), one PauliZ and one CZ per application; T6:
    the base and the pinned adjoint rule are chosen; T5 at level="device"."""

    def program():
        InnerSigns([0], [1], [2])
        qp.adjoint(InnerSigns([0], [1], [2]))

    resources = compiled_resources(program, num_wires=3)
    expected = costs.inner_signs()
    assert (resources.toffoli, resources.measurements, resources.measured_and_uncomputes) == (
        2 * expected.toffoli, 2 * expected.measurements, 0)
    assert resources.counts == {"CZ": 2, "PauliZ": 2}
    qp.analyze(verbose_program(program, num_wires=3), level="user")()
    captured = capfd.readouterr()
    assert {"_inner_signs", "_adjoint_inner_signs"} <= set(chosen_rules(captured.out
                                                                        + captured.err))
    compiled_resources(program, num_wires=3, level="device")


def test_read_shape_table_and_required_wires():
    assert InnerPrepare.read_shape(2, 2, 2, 1, 2, 1) == {
        "address_widths": (2, 2), "live_ranges": ((0, 2), (0, 3)), "width": 7, "block_bits": 1}
    assert InnerPrepare.read_table(1, 2, 1, 1, ((1, 0), (0, 1)), ((2, 3), (0, 1)),
                                   ((1, 0), (0, 1)), ((1, 1), (0, 0)), 2) == (
        (0, (54,)), (1, (26,)), (2, (1,)), (3, (45,)))
    # C = 3 (FeMoco has C = 27): r = x // C is not a bit field of x_o, unlike at C = 1 and 2
    zeros = ((0, 0),) * 9
    assert InnerPrepare.read_table(1, 3, 1, 3, zeros, zeros, zeros, zeros, 1) == tuple(
        ((x << 1) | b, (x // 3,)) for x in range(9) for b in range(2))
    assert InnerPrepare.required_wires(2, 2, 2, 1, 2, 1, 1, 3) == {
        "x_o": 2, "b": 2, "succ_b": 1, "succ_work_b": 1, "sigma_0": 1, "m_b": 2,
        "less_than_b": 1, "cmp_work_b": 1, "alt_b": 2, "keep_b": 2, "sgn": 1, "d_sgn": 1, "r": 1,
        "junk_b": 7, "phase_grad": 3, "work": 7}
    assert InnerUnprepare.required_wires(2, 2, 2, 1, 2, 1, 1, 3) == InnerPrepare.required_wires(
        2, 2, 2, 1, 2, 1, 1, 3)
    assert InnerSigns.required_wires() == {"sgn": 1, "d_sgn": 1, "less_than_b": 1}


ROWS = ((0, 1, 2), (2, 1, 0))
BITS = ((0, 1, 0), (1, 1, 0))
BASE = {"N": 3, "R": 2, "B": 2, "C": 1, "alt": ROWS, "keep": ((3, 0, 1), (2, 2, 0)),
        "sgn_table": BITS, "d_sgn_table": BITS, "b_k2": 2, "k2": 1, "ke2": 1, "s_u": 3}


@pytest.mark.parametrize(("change", "match"), [
    ({"N": 0}, r"N must lie in \[1, None\]"),
    ({"B": 0}, r"B must lie in \[1, None\]"),
    ({"k2": 3}, r"k2 must lie in \[0, 2\]"),
    ({"ke2": 2}, r"ke2 must lie in \[0, 1\]"),
    ({"k2": 0}, "needs 8 wires, more than the 7 target and junk wires"),
    ({"s_u": 2}, r"s_u must lie in \[3, None\]"),
    ({"alt": ((0, 1, 3), (2, 1, 0))}, r"alt entries must lie in \[0, 3\), got 3"),
    ({"keep": ((4, 0, 1), (2, 2, 0))}, r"keep entries must lie in \[0, 4\), got 4"),
    ({"sgn_table": ((0, 2, 0), (1, 1, 0))}, r"sgn_table entries must lie in \[0, 2\), got 2"),
    ({"d_sgn_table": ((0, 1), (1, 1))}, r"d_sgn_table must be a table of ints of shape \(2, 3\)"),
    ({"junk_b": 6}, "'junk_b' has 6 wires; expected 7"),
    ({"phase_grad": 2}, "'phase_grad' has 2 wires; expected at least 3"),
])
@pytest.mark.parametrize("op_class", [InnerPrepare, InnerUnprepare])
def test_constructors_validate(op_class, change, match):
    sizes = InnerPrepare.required_wires(3, 2, 2, 1, 2, 1, 1, 3)
    sizes |= {name: value for name, value in change.items() if name in sizes}
    wires = {name: list(range(100 * i, 100 * i + size))
             for i, (name, size) in enumerate(sizes.items())}
    args = BASE | {name: value for name, value in change.items() if name in BASE}
    with pytest.raises(ValueError, match=match):
        op_class(**args, **wires)


def test_inner_signs_constructor_validates():
    with pytest.raises(ValueError, match="'d_sgn' has 2 wires; expected 1"):
        InnerSigns([0], [1, 2], [3])
```

Replace `tests/ops/test_certificates_prepare.py` with:

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
"""

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional

from dfthc.ops.prepare import InnerPrepare, InnerUnprepare, OuterPrepare, OuterUnprepare
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
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_inner_prepare.py tests/ops/test_certificates_prepare.py -q`
Expected: FAIL at collection, `2 errors`: `ImportError: cannot import name 'InnerPrepare' from 'dfthc.ops.prepare'`.

- [ ] **Step 3: Add `inner_signs` to `dfthc.costs`**

In `src/dfthc/costs.py`, insert the function after `inner_unprepare`, as in this diff:

```text
--- a/src/dfthc/costs.py
+++ b/src/dfthc/costs.py
@@ -188,6 +188,11 @@
     return Cost(p.w_b) + erase + flagged_uniform_unprep(p.B + 1, p.s_u)
 
 
+def inner_signs() -> Cost:
+    """``InnerSigns`` (spec 9.24 row): ``T = 0``, ``M = 0``."""
+    return Cost(0)
+
+
 def tail_angle_load(p: WalkParams) -> Cost:
     """``TailAngleLoad`` (spec 9.24 row): ``U(RC, X_o, w_x, 0)``."""
     return Cost(unary(p.RC, p.X_o, p.w_x, controlled=False))
```

- [ ] **Step 4: Add the inner classes to `prepare.py`**

Replace `src/dfthc/ops/prepare.py` with:

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
address, so the swap is undone first, and the comparator reads ``keep``, so its adjoint comes
before the erase; nothing between a prepare and its unprepare may write ``succ``, ``succ_work`` or
``cmp_work``.

**Tables.** A read entry is one big-endian integer (spec 6.7) of ``width`` bits, stored as 32-bit
words, least significant first (L-24): ``(alt << b_k1) | keep`` for the outer read
(:meth:`OuterPrepare.read_table`). A prepare packs its compilable tables into the read's dynamic
``leaf_words`` inside its rule (:meth:`dfthc.ops.qroam.QROAMRead.leaf_words_from_table`); its
unprepare passes the same table, in the same entry layout, to
:class:`~dfthc.ops.qroam.QROAMErase`, whose masks are therefore exactly what the read loaded. The
erase's one-hot register is the reset ``target ‖ junk``, so ``2**onehot_bits`` must not exceed
``2**block_bits * width`` (the constructors check it: no walk instance needs ``extra_onehot``).

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
"""

from __future__ import annotations

from dataclasses import dataclass

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import AbstractArray, Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.qroam import WORD, WORD_DTYPE, QROAMErase, QROAMRead
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

    **Pre- and postconditions.** As in the table. Until :class:`OuterUnprepare`, nothing may
    write ``succ_x_o``, ``succ_work_x_o``, ``less_than_x_o``, ``cmp_work_x_o``, ``m_x_o``,
    ``alt_x_o``, ``keep_x_o`` or ``junk_x_o``, and ``x_o`` may be used only as a control or
    changed by operations that are undone before the unprepare.

    **Decomposition** (spec 9.7). ``FlaggedUniformPrep(X_o, s_u)`` on ``x_o``, ``succ_x_o``,
    ``succ_work_x_o``, ``phase_grad[:s_u]``; ``H`` on ``m_x_o`` and ``sigma_1``; ``QROAMRead`` of
    :meth:`read_table` at ``x_o`` (shape :meth:`read_shape`, no enable) into
    ``alt_x_o ‖ keep_x_o`` and ``junk_x_o``, its ``leaf_words`` built here from the table;
    ``LeftQuantumComparator(keep_x_o, m_x_o, less_than_x_o, cmp_work_x_o, "<=")``; ``w_x``
    ``CSWAP``\s of ``alt_x_o`` and ``x_o`` controlled by ``less_than_x_o``. Each child gets the
    start of ``work`` it needs.

    **Adjoint and control.** Never adjointed or controlled: the read has loops and data
    conditionals (spec 7.4 rules 4 and 6). The inverse is the operator :class:`OuterUnprepare`.

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
    is itself the adjoint class of :class:`OuterPrepare`.

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

    **Pre- and postconditions.** As in the table. Until :class:`InnerUnprepare`, nothing may
    write ``x_o``, ``succ_b``, ``succ_work_b``, ``less_than_b``, ``cmp_work_b``, ``m_b``,
    ``alt_b``, ``keep_b``, ``sgn``, ``d_sgn``, ``r`` or ``junk_b``, and ``b`` may be used only as
    a control or changed by operations that are undone before the unprepare.

    **Decomposition** (spec 9.8). ``FlaggedUniformPrep(B + 1, s_u)`` on ``b``, ``succ_b``,
    ``succ_work_b``, ``phase_grad[:s_u]`` (unconditionally, as in dfthcbe); ``H`` on ``sigma_0``
    and ``m_b``; ``QROAMRead`` of :meth:`read_table` at ``x_o ‖ b`` (shape :meth:`read_shape`, no
    enable) into ``alt_b ‖ keep_b ‖ sgn ‖ d_sgn ‖ r`` and ``junk_b``, its ``leaf_words`` built
    here from the tables; ``LeftQuantumComparator(keep_b, m_b, less_than_b, cmp_work_b, "<=")``;
    ``w_b`` ``CSWAP``\s of ``alt_b`` and ``b`` controlled by ``less_than_b``.

    **Adjoint and control.** Never adjointed or controlled: the read has loops and data
    conditionals (spec 7.4 rules 4 and 6). The inverse is the operator :class:`InnerUnprepare`.

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
    is itself the adjoint class of :class:`InnerPrepare`.

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

- [ ] **Step 5: Run the tests**

Run: `scripts/test.sh tests/ops/test_inner_prepare.py -q`
Expected: `50 passed` (15 to 36 s in the replays with 2 workers, depending on the machine's load, including the two `MID_NAPHTHALENE` compiles).

Run: `scripts/test.sh tests/ops/test_certificates_prepare.py -q`
Expected: `8 passed` (nightly tier, run targeted here).

Run: `scripts/test.sh tests/ops/test_outer_prepare.py -m "not slow" -q`
Expected: `44 passed`.

Run: `scripts/test.sh tests/test_costs.py tests/test_tolerance_hygiene.py -q`
Expected: `16 passed`.

- [ ] **Step 6: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `748 passed` (700 + 48; see "Expected test counts").

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add src/dfthc/ops/prepare.py src/dfthc/costs.py tests/ops/test_inner_prepare.py \
        tests/ops/test_certificates_prepare.py
git commit -m "Add InnerPrepare, InnerUnprepare and InnerSigns with the inner certificates"
```
