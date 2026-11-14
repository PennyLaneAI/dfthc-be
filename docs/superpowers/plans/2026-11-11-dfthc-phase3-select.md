# DFTHC Phase 3, Tasks 3.1 and 3.2: Inner Select and Reflections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the inner select of the block encoding (steps 10 to 18 of spec 3.5), `InnerSelect`, with its explicit adjoint class `InnerSelectDagger`, and the walk's reflections `RefT2` and `RefAB` with their explicit controlled classes `ControlledRefT2` and `ControlledRefAB`: `Operator2`s with one pinned rule each, exact Toffoli, measurement and measured-uncompute counts against `dfthc.costs`, and tests T2 to T6.

**Architecture:** `src/dfthc/ops/select.py` composes every Workstream E block in spec 9.18's order inside one straight-line rule per class; the two rules differ only in the Majorana gate (`qp.adjoint(MajoranaGate(...))` in the dagger), so every other child, `AngleErase` included, gets identical compilable arguments and register sizes. The reflections are appended to `src/dfthc/ops/reflections.py`: the plain ones emit `ZeroReflection`, the controlled ones emit the multi-controlled Z and the control's `Z` with `C(ZeroReflection)`'s helper, so no `C(...)` key is requested. Every class takes its registers by name and passes each child only the registers it needs, unpacked into lists once (spec 9.0).

**Tech Stack:** Python 3.12, PennyLane `Operator2` and Catalyst (`qp.qjit(capture=True)`, `qp.decompose`, `qp.analyze`), NumPy; the sparse simulator `dfthc.testing.sparse_sim` and `dfthc.testing.tape`; the encoded-Hamiltonian model `dfthc.data.encoded` and the fermion matrices `dfthc.data.fermion`; pytest with pytest-xdist; ruff.

This file holds Tasks 3.1 and 3.2 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 9), which run in parallel in wave 3 (master plan section 5) and touch disjoint files. Tasks 3.3 to 3.5 (block encoding, walk, controlled walk, end-to-end checks) stay in `2026-11-11-dfthc-phase3-assembly.md`, planned on top of this file's code; the last section, "Interfaces for Tasks 3.3 to 3.5", states what they consume.

**Code provenance.** Every code block below was executed on 2026-11-13 in a scratch replica of the repository (`.scratch/plan-check/repo-phase3s`, not committed), seeded from local `main` at `b34ed56` (Task 2E.5 merged) with `git archive`, plus the two files of Task 2E.3 from `phase2/2E.3` at `14866f3` (`src/dfthc/ops/spin.py`, `tests/ops/test_spin.py`, byte-identical to the E plan's blocks); this plan treats 2E.3's interfaces as final. The replica is not a git repository (the sandbox forbids creating one), so the dry run ran pytest the way `scripts/test.sh` does (the same environment, 2 workers, `taskpolicy -c utility`, `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` exported) through `.scratch/plan-check/p3stest.sh`, which also runs every session under the watchdog with a 3 GB cap. Each task was replayed RED then GREEN in a fresh copy of `main` plus 2E.3's files, as the two tasks run in parallel lanes, and then both together (`.scratch/plan-check/stages/3S/run_stages_3s.sh`, log `stages/3S/3s-stages.log`); every "Expected" line below is that replay's output, and the replayed files are byte-identical to the replica's. Further evidence, each with its log under `stages/3S/`: the per-commit tier on the replica with both tasks (`3s-per-commit-tier.log`); the `MID_NAPHTHALENE` compiles and the nightly T3, each alone under `run_watched` (`run_mid_watched_3s.py`, `3s-mid-compiles.log`); a mutation check of both test files (`mutate_3s.py`, `3s-mutants.log`); every docstring example run against its stated value and the nine spec 9.0 sections checked in order (`check_doc_examples.py`, `doc-examples.log`); the captured-equation measurements and their FeMoco-54 projection (`.scratch/plan-check/probes/p3s/capture_cost.py`, `capture-presets.log`, `capture-scaling.log`, `project_capture.py`, `capture-projection.log`); the check that Task 3.2's `reflections.py` is `main`'s with one import changed and a section appended (`check_reflections_diff.py`, `reflections-diff.log`); the interface script of the last section, which calls every class from split `WalkLayout` groups (`interfaces_probe.py`, `interfaces-probe.log`); which forced schedules see a parity-preserving erase mistake (`schedule_probe.py`, `schedule-probe.log`); and the document edits (`make_doc_edits_3s.py`, `doc-edits.md`).

**Load.** At most 2 pytest workers ran at a time, except during one 40-second targeted run that overlapped the mutation check (4 workers), and nothing FeMoco-shaped was compiled: FeMoco-54 values of the select are asserted on the closed forms, and the table-free reflections are compiled at FeMoco-54's register widths (1.2 to 1.7 s, below 290 MB). Every compile measured alone stayed below 360 MB; the largest pytest session, the per-commit tier, peaked at 1,976 MB. The full per-commit tier ran once. One scratch micro-benchmark of the sparse simulator, written while sizing the T3 tests and not part of this plan, put Hadamards on random wires of a random 100,000-term state and grew to about 14 GB in about two minutes before it was killed; every later probe ran under the watchdog.

**Revision 2** (2026-11-13, after the GPT-5.6 plan review `.superpowers/sdd/phase3s-plan-review-1.md`, Needs fixes, 0 Critical, 1 Important, 1 Minor). M-1 applied: the rank-from-bits mutant fails the valid-branch tests only at `random-C3`, and the all-branch round trip also at `sto3g` (its load and the unchanged erase disagree on unreachable addresses); the ledger table and the mutation evidence say so. I-1 (compiled reflection counts at `MID_NAPHTHALENE` and FeMoco-54's widths per commit) not applied, by the orchestrator's decision: the reflections are table-free, and the master plan already compiles table-free blocks at FeMoco's widths per commit (Task 1C.2's `FlaggedUniformPrep` at L = 324 and 28, Task 2E.4's ladders; the merged `test_givens.py` and `test_spin.py` do so at `MID_NAPHTHALENE` and FeMoco's widths), each case 1.2 to 1.7 s and below 290 MB, with nothing table-bearing compiled at FeMoco. Spec 12.1's T4 row states the stricter rule for every operator; the orchestrator amends it to name the table-free exception in the next spec edit batch. No code or count changes.

**Revision 3** (2026-11-14, after Task 3.1's reviews: GPT-5.6 `.superpowers/sdd/task-3.1-review-1.md`, Needs fixes, 0 Critical, 0 Important, 1 Minor; Opus `.superpowers/sdd/task-3.1-review-opus.md`, Approved, 0 Critical, 0 Important, 3 Minor; applied on `phase3/3.1` and here). The T3 adds the reachable failed-`b` class: the inner read loads a zero entry at a failed `b`, so `(b, r) = (0, 0)` on the last SF row, where the SF load writes `sf_angles[0][0]` and `b_neq_B = 1` and only the Majorana gate's `succ_b` control keeps the branch the identity; the former class (`b = 2**w_b - 1`, nothing loaded) stays. Gating the SF term on `b_neq_B` instead of `succ_b` passed every test before (Opus M-1, no count change). The adjoint and controlled compile test runs for both classes (GPT M-1, + 2). The select's docstring cites App. B.3, B.5 with Eqs. (93) and (94), B.6 and Table 9, and completes the causes of the erase's + 12 against dfthcbe and + 11 against the paper, the multiplexer's two paper figures and the Majorana gate's - 2 (Opus M-2). Opus M-3 (`_check` repeats `AngleErase`'s validation) waits for Task 5.1. Task 3.1 counts 43 tests, 39 per commit; the per-commit tier with both tasks is 975.

---

## Changes from the master plan, the spec and HANDOFF

The changes that touch a document are applied by the exact edits listed after this list (ids S-M for the master plan, S-S for the spec, S-H for HANDOFF). No template was edited; every S-M old text also occurs exactly once in `.scratch/plan-check/templates/2026-11-11-dfthc-master-plan.md`, so the orchestrator mirrors S-M1 to S-M8 there when it applies them.

1. **Plan file** (S-M1, S-M2, S-M4). Tasks 3.1 and 3.2 move from `phase3-assembly` to this file, `phase3-select`; section 1 lists it and its provenance. Tasks 3.3 to 3.5 keep `phase3-assembly`.
2. **`InnerSelect` and `InnerSelectDagger` take their registers by name** (S-M5, S-S1, S-S2, S-S3, S-H1). The master plan's `InnerSelect(..., select, outer, inner, system, phase_grad, work)` cannot be implemented as written: `WalkLayout.split_group(params, group, wires)` needs all thirteen `WalkParams` fields, and the sizes of the `outer` and `inner` groups depend on `b_k1`, `b_k2`, `k1`, `k2` and `s_u`, which are not among the select's compilable arguments. Passing whole groups would also bind 598 wires at FeMoco-54 to an operator that uses 22 of them, against spec 9.0's wire-passing rule, which spec 9.18 itself follows ("the needed registers of `outer` and `inner`"). So both classes take the sixteen registers they use (`x_o`, `succ_x_o`, `sigma_1`, `b`, `succ_b`, `sigma_0`, `r`, `angles`, `G0`, `G1`, `b_neq_B`, `sg`, `psi_up`, `psi_dn`, `phase_grad`, `work`) by name, in `WalkLayout` order; the select is an L2 block (spec 5.1), and grouped wires stay with the assembly operators of spec 9.21 to 9.23, which split their groups with `WalkLayout.split_group` and pass the registers on. The compilable arguments are the master plan's, `fixup_bits` being `AngleErase`'s `sf_onehot_bits` (`w_b` in the walk; any value `AngleErase` accepts, so the ledger's ob = `w_b - 1` refinement would be a one-argument change in Task 3.3). `required_wires` takes all nine compilable arguments, as `AngleErase.required_wires` does (spec 7.4 rule 1).
3. **The reflections take their registers by name** (S-M7, S-S4). The master plan's `RefT2(outer, inner, work)` has no compilable argument from which to split the groups, and would bind about 600 wires to use 22 (`RefT2`) or 46 (`RefAB`). The classes are `RefT2(succ_x_o, b, m_b, sigma_0, work)`, `RefAB(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work)` and the controlled ones with `control` first (spec 9.0); their sizes follow from the wires, so `required_wires(w_b, b_k2)` and `required_wires(w_x, b_k1, w_b, b_k2)` take the widths, as `ZeroReflection.required_wires(n_zero, n_one)` does on `main`. `work` is exact (n_w − 3, controlled n_w − 2), so Task 3.3's `test_work_pool` can require every call to receive exactly `required_wires`; the reflections pass `ZeroReflection` exactly its need.
4. **More tests than the master plan lists** (S-M6, S-M8). The master plan's common paragraph asks for T2 to T6 for every class, the dagger and controlled classes included; its 3.1 and 3.2 entries name only the FeMoco T4 and the semantic checks. The tests below add, for Task 3.1: T3 on the tape path at `H2_STO3G_TEST`, at a `C = 3` shape (`random-C3`: N = 3, a Q1 row, r = x_o // 3 off every bit field of `x_o`) and at `H2_631G_TEST` (`slow`, about 200,000 terms) against the encoded model, each under three forced schedules; the round trip on every branch under two schedules; the children's order and registers; compiled T4 at four shapes and at `MID_NAPHTHALENE` (`slow`, as spec 12.6 places the composites' mid-size T4); the failing `qp.adjoint` and `qp.ctrl` of `InnerSelect`. For Task 3.2: the exact action at both test presets with the pinned adjoints, the controlled classes with the control in |0⟩, |1⟩ and |+⟩, and compiled T4 at both test presets, `MID_NAPHTHALENE` and FeMoco-54's register widths per commit (table-free, master plan section 12.9).
5. **Master plan section 3** (S-M3): a new item 22 with this dry run's compile and capture findings.
6. **Not added:** no `Adjoint(InnerSelect)`, `C(InnerSelect)`, `C(RefT2)`, `C(RefAB)`, `Adjoint(ControlledRefT2)` or `Adjoint(ControlledRefAB)` rule. A delegating rule would compile but bring back the cascade of distributed candidates (spec 7.4 rule 6, the `C(DFTHCWalk)` argument); `test_no_adjoint_or_controlled_path` pins the loud failure for the select, and Task 3.4's inventory test catches a stray modifier anywhere.

Exact edits (master plan, spec and HANDOFF, against local `main` at `b34ed56`):

Each edit replaces one exact text of the file on local `main` at `b34ed56`, where it occurs exactly once (checked by `.scratch/plan-check/stages/3S/make_doc_edits_3s.py`, which also applies them in order).

**S-M1** (2026-11-11-dfthc-master-plan.md, section 1, the file table). Old:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase2-blocks.md` | Tasks 2E.1 to 2E.5 |
````

New:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase2-blocks.md` | Tasks 2E.1 to 2E.5 |
| `docs/superpowers/plans/2026-11-11-dfthc-phase3-select.md` | Tasks 3.1, 3.2 |
````

**S-M2** (2026-11-11-dfthc-master-plan.md, section 1, code provenance). Old:

````text
The Phase 2 E file (`phase2-blocks`) was executed the same way on 2026-11-12, against `main` at `f7acf97`; its preamble lists its changes to this plan.
````

New:

````text
The Phase 2 E file (`phase2-blocks`) was executed the same way on 2026-11-12, against `main` at `f7acf97`; its preamble lists its changes to this plan. The Phase 3 select file (`phase3-select`, Tasks 3.1 and 3.2) was executed the same way on 2026-11-13, against `main` at `b34ed56` with Task 2E.3's two files (`14866f3`); its preamble lists its changes to this plan.
````

**S-M3** (2026-11-11-dfthc-master-plan.md, section 3, a new item 22 after item 21 (the end of item 21 is shown)). Old:

````text
with an effective matrix comparison; their compiled path is unchanged.
````

New:

````text
with an effective matrix comparison; their compiled path is unchanged.
22. **`InnerSelect` and the reflections as built** (Phase 3 select dry run, 2026-11-13). `InnerSelect` and `InnerSelectDagger` compile to `costs.inner_select` at `H2_STO3G_TEST`, `H2_631G_TEST` and `MID_NAPHTHALENE` (T = 33, 65, 625; M = 7, 15, 138; A = T − (2N + 5)), the mid select alone in 2.5 to 2.9 s at 358 MB; the four reflections compile to `costs.ref_t2` and `costs.ref_ab` up to FeMoco-54's register widths in about 1.4 s below 290 MB, the controlled classes as fast as the plain ones (they request no `C(...)` key, so item 16's overhead is gone). Captured equations (Task 0.9d's instrument): the select target costs 1.43 equations per wire its rule binds to its children when the rule unpacks every register into a list once, against 5.0 when it passes arrays on (`MID_NAPHTHALENE`: 1,293 against 4,326); projected to FeMoco-54's 4,444 bound wires, 6,408 equations per select target against the K-1 stand-in's 27,537, about 42,000 fewer for the two (11% of the stand-in compile's 390,949). A program holding both selects captures every child target once except `MajoranaGate` and its adjoint. The select and the reflections take their registers by name (spec 6.7, 9.18 to 9.20).
````

**S-M4** (2026-11-11-dfthc-master-plan.md, section 4, the rows of Tasks 3.1 and 3.2). Old:

````text
| 3.1 | `InnerSelect` and `InnerSelectDagger` | A | G2 | phase3-assembly |
| 3.2 | `RefT2`, `RefAB`, `ControlledRefT2`, `ControlledRefAB` | B | G2 | phase3-assembly |
````

New:

````text
| 3.1 | `InnerSelect` and `InnerSelectDagger` | A | G2 | phase3-select |
| 3.2 | `RefT2`, `RefAB`, `ControlledRefT2`, `ControlledRefAB` | B | G2 | phase3-select |
````

**S-M5** (2026-11-11-dfthc-master-plan.md, Task 3.1, interfaces). Old:

````text
- **Interfaces produced:** `InnerSelect(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, select, outer, inner, system, phase_grad, work)`, with grouped wires split through `WalkLayout.split_group`. `InnerSelectDagger`, with the same arguments and a pinned base rule, uses the adjoint Majorana gate; both call `GivensLadderDagger`.
````

New:

````text
- **Interfaces produced:** `InnerSelect(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up, psi_dn, phase_grad, work)`: the registers by name, in `WalkLayout` order, so the select receives only the registers it reads or writes (spec 9.0; the block encoding splits its groups with `WalkLayout.split_group` and passes them on); `fixup_bits` is `AngleErase`'s `sf_onehot_bits` (`w_b` in the walk). Static `required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)`: exact sizes, `phase_grad` `b_rot` wires, `work` the largest need of the children (14 at FeMoco-54). `InnerSelectDagger`, with the same arguments and a pinned base rule, uses `qp.adjoint(MajoranaGate(...))` and still loads first and erases last; both call `GivensLadderDagger` and pass every child but the Majorana gate identical compilable arguments and register sizes.
````

**S-M6** (2026-11-11-dfthc-master-plan.md, Task 3.1, acceptance tests). Old:

````text
  - T4: T = 3,537, M = 798 at FeMoco.
  - T3 on valid branches: V† Maj V for the H2 test shape (sparse simulator).
  - `InnerSelectDagger` loads first and erases last, and on valid branches it is the inverse of `InnerSelect` (sparse simulator).
````

New:

````text
  - T4: T = 3,537, M = 798, A = 3,424 at FeMoco on the closed forms; compiled T, M and A = T − (2N + 5) at both test presets (33 and 65, M = 7 and 15), at a `C = 3` shape with the erase's `ob = 0` fallback, at R = 1 and at `MID_NAPHTHALENE` (`slow`; 625, M = 138), with the erase's hinted conditionals at their expectation.
  - T3 on the tape path (sparse simulator, which asserts every AND precondition): on the branches the prepares leave, with `sigma_0` in |+⟩ and the phase-gradient state, every term equals the encoded model's generator piece (iγ̃₀γ̃₁ on SF; with `sigma_0` projected onto |+⟩, a_ũ on D1 and a_ũ† on Q1; the identity elsewhere), and `InnerSelectDagger`'s the adjoint, under the all-ones, an alternating and a seeded schedule of the erase; at `H2_STO3G_TEST`, at a `C = 3` shape (N = 3, a Q1 row, rank selection off the bit fields of `x_o`) and at `H2_631G_TEST` (`slow`).
  - `InnerSelectDagger` after `InnerSelect` is the identity on every branch, unreachable ones included, under two forced schedules (sparse simulator). The children run in spec 9.18's order on exactly their registers: `InnerSelectDagger` loads first and erases last, with `Adjoint(MajoranaGate)` and the explicit `GivensLadderDagger`, and the two selects' children differ only in the Majorana gate.
  - T2 at four sizes, T5, T6 (the rule listing; `qp.adjoint` and `qp.ctrl` of `InnerSelect` fail to compile after the expected measuring-rule warning).
````

**S-M7** (2026-11-11-dfthc-master-plan.md, Task 3.2, files and interfaces (the start of the interfaces is shown)). Old:

````text
- **Files:** `src/dfthc/ops/reflections.py` (`RefT2`, `RefAB`), `tests/ops/test_ref_t2_ab.py`.
- **Interfaces produced:** `RefT2(outer, inner, work)` and `RefAB(outer, inner, work)`, with pinned self-adjoints, and the classes `ControlledRefT2(control, outer, inner, work)` and `ControlledRefAB(control, outer, inner, work)`, whose base rules
````

New:

````text
- **Files:** `src/dfthc/ops/reflections.py` (`RefT2`, `RefAB`, `ControlledRefT2`, `ControlledRefAB`), `tests/ops/test_ref_t2_ab.py`.
- **Interfaces produced:** `RefT2(succ_x_o, b, m_b, sigma_0, work)` and `RefAB(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work)`, the pattern registers by name (spec 9.0), with pinned self-adjoints and `required_wires(w_b, b_k2)` and `required_wires(w_x, b_k1, w_b, b_k2)` (`work` exactly n_w − 3), and the classes `ControlledRefT2(control, succ_x_o, b, m_b, sigma_0, work)` and `ControlledRefAB(control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work)` (`work` n_w − 2; `ControlledRefAB`'s 44 wires at FeMoco-54 are the walk's largest need), whose base rules
````

**S-M8** (2026-11-11-dfthc-master-plan.md, Task 3.2, acceptance tests). Old:

````text
  - T4 at FeMoco: 20 / 21 and 44 / 45.
  - Exact action, including the global sign, at the H2 shape: controlled with the control in |+⟩.
````

New:

````text
  - T4 at FeMoco on the closed forms: 20 / 21 and 44 / 45; compiled T, M = 0 and A = T − 1 at both test presets, `MID_NAPHTHALENE` and FeMoco-54's register widths (table-free, per commit).
  - Exact action, including the global sign, at the walk's register widths of both test presets (lightning, on a phased state of the pattern registers), the pinned adjoints included; the controlled classes with the control in |0⟩, |1⟩ and |+⟩.
  - T2 at two sizes for the six rules, T5, T6 (the listing names the six rules and not `C(ZeroReflection)`'s).
````

**S-S1** (2026-11-11-dfthc-pennylane-design.md, 6.7, the grouped-wires item). Old:

````text
- `InnerSelect` and `InnerSelectDagger` (§9.18) and the assembly operators (§9.21 to §9.23) take grouped wire arguments (`outer`, `inner`, `select`, `system`, `phase_grad`, `work`). `dfthc.registers.WalkLayout` defines the order of the registers inside each group, and every operator splits groups only through `WalkLayout` helpers. The blocks they call take the registers their §9 tables list (the prepares of §9.7 and §9.8 take named registers), so that every composite passes each child only the registers it needs (§9.0).
````

New:

````text
- The assembly operators (§9.21 to §9.23) take grouped wire arguments (`outer`, `inner`, `select`, `system`, `phase_grad`, `work`). `dfthc.registers.WalkLayout` defines the order of the registers inside each group, and every operator splits groups only through `WalkLayout` helpers. The blocks they call take the registers their §9 tables list, by name (the prepares of §9.7 and §9.8, `InnerSelect` and `InnerSelectDagger` of §9.18 and the reflections of §9.19 and §9.20 included), so that every composite passes each child only the registers it needs (§9.0).
````

**S-S2** (2026-11-11-dfthc-pennylane-design.md, 9.18, Wires). Old:

````text
| Wires | Grouped: `select` (`angles`, `G0`, `G1`, `b_neq_B`, `sg`), the needed registers of `outer` and `inner` (`succ_x_o`, `x_o`, `sigma_1`, `succ_b`, `b`, `r`, `sigma_0`), `system`, `phase_grad`, `work` |
````

New:

````text
| Wires | By name, in `WalkLayout` order: `x_o`, `succ_x_o`, `sigma_1` (outer), `b`, `succ_b`, `sigma_0`, `r` (inner), `angles`, `G0`, `G1`, `b_neq_B`, `sg` (select), `psi_up`, `psi_dn`, `phase_grad` ($b_{\mathrm{rot}}$), `work` (the largest need of its children: the ladders' $b_{\mathrm{rot}} - 1$ at FeMoco-54) |
````

**S-S3** (2026-11-11-dfthc-pennylane-design.md, 9.18, Compilable). Old:

````text
| Compilable | `tail_angles`, `sf_angles`, $N, R, B, C, N_+, b_{\mathrm{rot}}$, fixup bits |
````

New:

````text
| Compilable | `tail_angles`, `sf_angles`, $N, R, B, C, N_+, b_{\mathrm{rot}}$, `fixup_bits` (`AngleErase`'s SF one-hot bits, §9.17; $w_b$ in the walk) |
````

**S-S4** (2026-11-11-dfthc-pennylane-design.md, 9.19 and 9.20, a new Wires row after Action (the Action row is shown)). Old:

````text
| Action | $2\Pi_{T_2} - I$ | $2\lvert 0\rangle\langle 0\rvert_{\mathcal A\cup\mathcal B} - I$ |
````

New:

````text
| Action | $2\Pi_{T_2} - I$ | $2\lvert 0\rangle\langle 0\rvert_{\mathcal A\cup\mathcal B} - I$ |
| Wires | `succ_x_o`, `b` ($w_b$), `m_b` ($b_{k2}$), `sigma_0`, `work` ($n_w - 3$); `ControlledRefT2`: `control` first, `work` $n_w - 2$ | `x_o`, `m_x_o`, `sigma_1`, `b`, `m_b`, `sigma_0`, `work` ($n_w - 3$); `ControlledRefAB`: `control` first, `work` $n_w - 2$ |
````

**S-H1** (HANDOFF.md, section 6, Wires). Old:

````text
  dfthcbe (Global Constraints). `InnerSelect` and `InnerSelectDagger` (spec 9.18) and the
  assembly operators (spec 9.21 to 9.23) take grouped wires (`outer`, `inner`, `select`,
  `system`, `phase_grad`, `work`) and split them only through `WalkLayout.split_group`; the
  blocks they call take the registers their spec 9 tables list (spec 9.0, wire passing).
````

New:

````text
  dfthcbe (Global Constraints). The assembly operators (spec 9.21 to 9.23) take grouped wires
  (`outer`, `inner`, `select`, `system`, `phase_grad`, `work`) and split them only through
  `WalkLayout.split_group`; the blocks they call, `InnerSelect`, `InnerSelectDagger` and the
  reflections included, take the registers their spec 9 tables list by name (spec 9.0, wire
  passing).
````

## Shared edits

None. `src/dfthc/ops/reflections.py` is Task 3.2's file in the master plan; it keeps `main`'s 182 lines except one import line (`check_int` added) and gains the `RefT2`/`RefAB` section at the end (`stages/3S/check_reflections_diff.py`, log `reflections-diff.log`). No other merged module changes: `dfthc.costs` already has `inner_select`, `ref_t2` and `ref_ab` (spec 9.24 rows), `WalkLayout` needs no helper (change 2), the rules use no emitter (so no `as_resources` key), and `dfthc.ops._validate` already has `canonical_ints`, `check_int` and `check_wires`. `select.py` reuses the loads' validators `SFAngleLoad._check` and `TailAngleLoad._check`, as `AngleErase` does.

## How this plan meets the binding ledger rows

| Ledger row or decision | What this plan does |
| --- | --- |
| Decision "K-1 at G0" (explicit classes; per-leaf loads) | `InnerSelectDagger`, `ControlledRefT2` and `ControlledRefAB` are classes with pinned base rules. The only adjoints in the new rules are the leaf adjoint `qp.adjoint(MajoranaGate(...))` in instance form (spec 7.4 rules 6 and 13) and the pinned self-adjoints `Adjoint(RefT2)`, `Adjoint(RefAB)`; nothing calls `qp.adjoint` or `qp.ctrl` on a composite. `test_children_order_and_wires` fails on an adjointed ladder (`GivensLadderDagger` expected by name), the T6 listing of Task 3.2 fails if a controlled reflection requests `C(ZeroReflection)`, and the mutation check shows both are the only tests that see these (identical semantics and counts). The loads still build `QROAMRead`'s per-leaf words inside their own rules (Task 2E.1). |
| Decision "Measuring-rule warnings" | `test_no_adjoint_or_controlled_path` wraps `InnerSelect` in `qp.adjoint` and `qp.ctrl` on purpose: the compile fails with `CompileError` "Decomposition rule not found for operator 'Adjoint(InnerSelect)'" (or `'C(InnerSelect)'`) after Catalyst's "Skipped the _angle_erase ..." warnings, which `classify_rule_warning` returns as `"expected"`; PennyLane's own `_adjoint_temporary_and` skips are `"noise"` and filtered. |
| K-1 = PX: resources equal emitted gates, pins, hinted erase conditionals | Both select rules and the six reflection rules declare their children by hand (straight-line rules with composite keys), and T2 (`assert_valid_decomp_rule`) runs them at four and two sizes covering every branch of the declarations (the dagger's adjoint Majorana key, the erase's `ob` at `w_b`, 1 and 0, R = 1; both reflection widths); each rule is registered with `register_rule` and asserted the only one under its key. `assert_counts` checks the erase's hinted conditionals inside the select at their expectation, `ceil(R B / 2)` `CZ` and `ceil(N / 2)` `Z` (`ob >= 1`) or `ceil((R B + N) / 2)` `Z` (`ob = 0`); nothing else in the select emits `CZ` or `Z`. |
| 2E.5: a tape-path sparse-simulator test for every operator with hand-written AND pairs (lightning runs `TemporaryAND` and its measured uncompute both as `Toffoli`) | The select's T3 and round trip run the tape path on the sparse simulator in strict mode, whose handlers assert every AND precondition, the Majorana gate's included, forward and adjoint. The reflections have no hand-written AND pairs (they emit PennyLane's `MultiControlledX`), so their T3 runs compiled on lightning, the path the walk takes. |
| 2D.2 (M-1): an all-ones schedule sees only each address's mask parity | The select's T3 runs the erase under the all-ones, the alternating and a seeded forced schedule, the round trip both erases under the alternating and a seeded one. The mutant `erase_with_the_rungs_reversed` (the erase's tables with every entry's rung registers reversed, a parity-preserving permutation of each mask) passes the all-ones schedule and fails the other two (`stages/3S/schedule-probe.log`); `erase_with_n_plus_0` (wrong `G1` mask bits) fails all three. |
| 2D.2 (I-1): a rank-selection test at a `C` that is not a power of two | `random-C3` (C = 3) in the T3 and the round trip; the mutant `select_rank_from_x_o_bits` (the SF load addressed by low bits of `x_o`) passes every valid-branch semantic test at `C = 1` and 2 and fails them only at `random-C3`; the all-branch round trip also fails at `sto3g`, because the mutant changes the load's address but not the erase's, which disagree on unreachable addresses. |
| 2D.2, for Task 3.1: read-only and dirty registers, `phase_grad` and `work`, `InnerSigns`' position, failed `b`, ungated D1/Q1 | The select takes none of `less_than_b`, `sgn`, `d_sgn`, `alt_b`, `keep_b`, `m_b`, `junk_b`, `succ_work_b` or `cmp_work_b`, so it cannot write them; it reads `x_o`, `succ_x_o`, `sigma_1`, `b`, `succ_b`, `r` only as controls and `sigma_0` diagonally (T3 asserts the index registers unchanged and every ancilla clean). `phase_grad` is exactly `b_rot` wires, the walk's whole register; `work` is the largest child need (14 at FeMoco-54) and each child gets its prefix. The T3 classes include failed `b` on an SF row (`succ_b = 0`: the identity) and a D1 row with `succ_b = 0` (still `a_ũ`: the Majorana gate's one-body path is not gated on `succ_b`). `InnerSigns`' position is Task 3.3's ("Interfaces for Tasks 3.3 to 3.5"). |
| 2E.2, for Task 3.1: one shared `AngleErase` target; load/erase register discipline; the dagger loads first and erases last; hinted counts; shared `work` | Both rules pass `AngleErase` (and the loads, ladders and spin operators) identical compilable arguments and sizes: `test_children_order_and_wires` compares every child's hyperparameters and registers between the two rules, and the capture probe at `MID_NAPHTHALENE` shows one `AngleErase` target (2,605 equations) in a program holding both selects. Between the loads and the erase `angles`, `G0`, `G1`, `b_neq_B` are only controls or the adders' addend (docstring; T3 under three schedules). The children test asserts the order, loads first and erase last in both classes. Work at FeMoco-54: tail load 8, SF load 9, ladders 14, Majorana gate 2, erase 8 (`test_femoco_counts_on_the_closed_form` asserts the select's 14). |
| 2E.4, for Task 3.1: `GivensLadder`, `MajoranaGate`, `GivensLadderDagger`; never `qp.adjoint(GivensLadder)`; exact registers; the wire-passing pattern | `test_children_order_and_wires` asserts the order and that the ladders get `system = psi_dn`, the whole `angles`, exactly `phase_grad` (`b_rot` wires) and exactly `work[:b_rot - 1]`; the mutants `select_ladders_exchanged`, `select_adjointed_ladder_in_rule_and_resources` and `select_ladder_registers_reversed` fail. The pattern was measured: unpacking every register into a list once costs 1.43 captured equations per bound wire against 5.0 for arrays passed on (section "Captured equations"). |
| 2E.5, for Task 3.1: the Majorana wiring, `qp.adjoint(MajoranaGate(...))` in instance form, `sigma_0` projected onto \|+⟩, no `succ_b` on the one-body path | `target = psi_dn[0]`, `flavour = sigma_0`, `work[:2]`, asserted by the children test (mutants `select_majorana_on_the_last_mode`, `select_majorana_flavour_sigma_1` fail). The T3 puts `sigma_0` in \|+⟩ and expects the flavour components a + a† and a − a† on D1 (a + a†, a† − a on Q1), whose ⟨+\| projection is a_ũ (a_ũ†); the dagger expects their adjoints (mutants `dagger_with_the_forward_majorana` and `select_with_the_adjoint_majorana` fail). 2E.5's own tests stay unchanged. |
| 1A.5: E-1 at `H2_STO3G_TEST` cannot see the ladder order or a tail-direction error | The select's T3 compares every tail row's action with the encoded model's own direction and D1/Q1 assignment at `random-C3` (two D1 rows and a Q1 row, two rungs, per commit) and `H2_631G_TEST` (three D1 rows and a Q1 row, three rungs, nightly), and every SF entry's iγ̃₀γ̃₁. |
| 1C.3: Clifford counts are upper bounds | T4 asserts T, M, A, the plain `Toffoli` count and the hinted `CZ` and `Z` expectations, never a `CNOT` or `PauliX` count; the docstrings label those as bounds. |
| 1C.6: `C(ZeroReflection)` with too little work fails at rule time | The controlled classes check `work` exactly in their constructors (`ValueError`, spec 9.0) and pass the multi-controlled Z exactly n_w − 2 wires. |
| 2E.4 (project-wide): no operator checks wire disjointness | Documented as a precondition of every new class ("pairwise disjoint, unchecked"); the select's children receive disjoint sub-registers of disjoint registers, as the children test shows; the check stays with Task 5.1. |
| HANDOFF section 6, tolerances | Every comparison states both tolerances (`rtol=0` with `atol=1e-12` on the simulator, `1e-10` on lightning); `tests/test_tolerance_hygiene.py` passes with the new files. |

Rows for later tasks: 1A.5's normalization row and 2D.2's notes for E-1 (`C <= 2` at the presets) are Task 3.5's; the inventory keys of 2E.4 and 2E.5 are Task 3.4's ("Interfaces for Tasks 3.3 to 3.5" lists the keys these classes add).

## FeMoco-54 numbers, compiled counts, dfthcbe and the paper

The master plan's FeMoco-54 numbers hold: `dfthc.costs` on `main` gives `InnerSelect` and `InnerSelectDagger` T = 3,537, M = 798 (A = 3,424), `RefT2` 20 and 21 controlled, `RefAB` 44 and 45 controlled. The rules emit exactly the closed forms at every compiled size:

| Operator | `H2_STO3G_TEST` | `H2_631G_TEST` | `MID_NAPHTHALENE` | FeMoco-54 |
| --- | --- | --- | --- | --- |
| `InnerSelect`, `InnerSelectDagger` (T / M / A) | 33 / 7 / 24 | 65 / 15 / 52 | 625 / 138 / 600 | 3,537 / 798 / 3,424 (closed form) |
| `RefT2` / `ControlledRefT2` (T; M = 0, A = T − 1) | 4 / 5 | 4 / 5 | 18 / 19 | 20 / 21 (compiled at FeMoco's widths) |
| `RefAB` / `ControlledRefAB` | 8 / 9 | 9 / 10 | 39 / 40 | 44 / 45 (compiled at FeMoco's widths) |

Against dfthcbe (its symbolic call graph at FeMoco-54, `tests/fixtures/dfthcbe/femoco_block_toffolis.json`; the controlled reflections from its controlled block-encoding and walk totals) and the paper's Table III (spec 13.1):

| Block | Ours | dfthcbe | Paper | Causes |
| --- | ---: | ---: | ---: | --- |
| `InnerSelect` (and the dagger) | 3,537 | 3,502 | 3,710 ¹ | dfthcbe +35: tail load +5, SF load +20 (exact iteration), erase +12 (one-hot over 32 positions, exact iteration), multiplexer −2. Paper −173: Givens ladders −212 (`rz-addsub`), loads +28 and erase +11 (exact iteration, one-hot padding), multiplexer +2, Majorana gate −2 |
| `RefT2` / controlled | 20 / 21 | 20 / 21 | 21 | none against dfthcbe |
| `RefAB` / controlled | 44 / 45 | 45 / 45 | 46 | `constants`: our multi-controlled Z targets the last pattern wire; dfthcbe's reflection costs the same with and without its control |

¹ With the tail-angle fix-up (N = 54) that the paper's Eq. (B50) charges to the inner erase and spec 13.1 moves to step 18; without it, 3,656. Per walk the select's share is unchanged from spec 13.1, and the walk stays at 9,793 (controlled 9,795).

## Captured equations and the wire-passing pattern

Task 0.9d's instrument (`probes/p3s/capture_cost.py`: every converted rule target's nested equations, through Catalyst's `_convert_decomp_target_spec`), one compile per program on `null.qubit`, each under the watchdog (2.8 to 8.0 s, at most 519 MB):

| Program (both selects) | Wires bound by one select rule | `InnerSelect` target, lists (this plan) | the same, arrays passed on |
| --- | ---: | ---: | ---: |
| `H2_STO3G_TEST` | 87 | 169 | 465 |
| `H2_631G_TEST` | 143 | 251 | 745 |
| `MID_NAPHTHALENE` | 859 | 1,293 | 4,326 |
| random N = 10, 20, 30 (b_rot = 15) | 842, 1,656, 2,466 | 1,266, 2,422, 3,572 | n/a, n/a, 12,361 |

`InnerSelectDagger`'s target costs the same. The fit is 1.430 equations per bound wire plus 53 (largest residual 12) with lists, against 5.000 plus 30 with arrays (`capture-projection.log`). At FeMoco-54 one select rule binds 4,444 wires (tail load 814, SF load 815, two multiplexers 8, two swaps 218, two ladders 1,756, Majorana gate 9, erase 824), so each select target projects to 6,408 equations with lists and 22,252 with arrays, against 27,537 for each of the K-1 stand-in's select targets. The two selects therefore capture about 42,000 equations fewer than the stand-in assumed, 11% of the 390,949 nested equations of its 2,648 MB FeMoco-54 controlled-walk compile (at 5.6 to 6.8 KB per equation, about 240 to 290 MB at most); passing arrays would have saved only about 10,600. A program holding both selects captures each child target once except `MajoranaGate` and its adjoint (86 equations): at `MID_NAPHTHALENE` the select alone captures 17,886 equations and both 19,265 = 17,886 + 1,293 + 86. Compile time and memory: the `MID_NAPHTHALENE` select compiles alone in 2.5 s (dagger 2.9 s) at 358 MB; the reflections at `MID_NAPHTHALENE` and at FeMoco-54's widths in 1.2 to 1.4 s below 290 MB, the controlled ones as fast as the plain ones (`3s-mid-compiles.log`).

## Mutation evidence

`stages/3S/mutate_3s.py` (log `3s-mutants.log`, 18 mutants, each a wrong construction with the rule and its declaration changed together where needed, each run against its test file's per-commit tests): all 18 fail at least one test. The select's T3 catches the exchanged ladders, the Majorana gate on the last mode or with flavour `sigma_1`, a forward Majorana gate in the dagger and an adjoint one in the select, a multiplexer not undone, the erase before the un-multiplexing, the reversed angle-register order (only at `random-C3`), the rank from bits of `x_o` (the valid-branch tests only at `random-C3`; the all-branch round trip also at `sto3g`, since the mutant's load and the unchanged erase disagree on unreachable addresses) and the erase with `n_plus = 0`. The erase given the loads' tables with every entry's rung registers reversed permutes each mask's bits, so every address keeps its mask parity: the all-ones schedule passes it and the alternating and seeded schedules fail it (`stages/3S/schedule_probe.py`, log `schedule-probe.log`), and the T3, the round trip and the children test at `random-C3` fail. T4 catches the erase at `fixup_bits = 0` (a semantically valid one-hot width with other counts); only the children test catches `qp.adjoint(GivensLadder(...))` declared consistently (same semantics and counts). For the reflections, `RefT2` with `sigma_0` as its one-valued wire and `RefAB` with `sigma_1` moved fail the exact-action tests, the controlled rules without the control's `Z` or on control value 0 fail the three-control-state test, and a `ControlledRefAB` built with `qp.ctrl(ZeroReflection(...))` fails only the T6 listing.

## Risks and open questions

1. **The named-register interfaces need the orchestrator's approval** (changes 2 and 3). They are the only interface changes; Task 3.3's writer must split the groups with `WalkLayout.split_group` and pass the registers on. If the grouped form were wanted after all, the select would need the thirteen `WalkParams` fields as a compilable argument (and still should not take the whole `outer` and `inner` groups).
2. **The capture projection extrapolates** from 2,466 to 4,444 bound wires. The fit is linear to within 12 equations over 87 to 2,466 wires and the cost is a per-bind constant per wire (spec 9.0), so the projection is safe; the FeMoco compile of Task 4.5 measures it.
3. **Per-commit time.** The two files add 84 per-commit tests: the select file's 43 run in 35 to 40 s and the reflections' 45 in 30 to 34 s with 2 workers; the slowest per-commit tests, each file's verbose and device-level compile and the failing controlled select compile, take 5 to 7 s. The replica's per-commit tier with both took 353 s.
4. **Cross-module use of private validators.** `select.py` calls `SFAngleLoad._check` and `TailAngleLoad._check`; a public validator in `angle_data.py` would be a later cleanup (Task 5.1), not done here to keep the merged modules untouched.
5. **No open question blocks the plan.** The decisions above are the obvious choices: change 2 follows spec 9.18's own "needed registers" and spec 5.1's layering, change 3 the same rule.

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
| `src/dfthc/ops/select.py` | 3.1 | `InnerSelect`, `InnerSelectDagger`, their shared validation, child sizes, rule body and resource declaration |
| `tests/ops/test_inner_select.py` | 3.1 | the action against the encoded model, the round trip, the children's order and registers, counts, rule selection, validation |
| `src/dfthc/ops/reflections.py` | 3.2 (replaces `main`'s file) | `main`'s `ZeroReflection` unchanged, plus `RefT2`, `RefAB`, `ControlledRefT2`, `ControlledRefAB` and the pinned `Adjoint(RefT2)`, `Adjoint(RefAB)` |
| `tests/ops/test_ref_t2_ab.py` | 3.2 | the exact action with the global sign, three control states, counts, rule selection, validation |

Within wave 3 (master plan section 5) the two tasks' file lists are disjoint. `tests/ops/` gets no `__init__.py`, and test modules have unique base names.

**Shell conventions.** Run commands from the worktree root of your lane. `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` locates the shared environment, as in the earlier plans.

**Expected test counts.** Each task states its targeted count exactly. The per-commit tier (`scripts/test.sh -m "not slow and not release" -q`) counts **891 tests on `main` at `b34ed56` plus Task 2E.3** (measured on the replica: `891/935 tests collected (44 deselected)`; 867 on `main` and 2E.3's 24, all per commit: 867 + 24 = 891). Both tasks branch from the same `main`; whichever merges second rebases onto the first and shifts by its tests. The pass criterion is "no failures, no errors, no warnings summary"; the number is a cross-check.

| Task | Base | Own per-commit | Expected per-commit tier | Own slow tests |
| --- | ---: | ---: | ---: | ---: |
| 3.1 | 891 | 39 (`test_inner_select.py`, 43 tests) | 930 = 891 + 39 | 4 (the T3 at `H2_631G_TEST`, both classes; the `MID_NAPHTHALENE` counts, both classes) |
| 3.2 | 891 | 45 (`test_ref_t2_ab.py`) | 936 = 891 + 45 | 0 |

With both tasks merged the per-commit tier counts 891 + 39 + 45 = 975. Dry run on the replica with both (before revision 3's two tests): `973 passed in 352.66s`, peak 1,976 MB (`stages/3S/3s-per-commit-tier.log`; run with `TMPDIR` outside the workspace, so `tests/test_infrastructure.py` passes in the replica too).

**Design notes the tasks share.**
- *Registers by name, unpacked once.* Every class takes the registers it uses, in `WalkLayout` order (changes 2 and 3). The select's rule turns every register into a list once and hands each child lists of wires and slices of lists, so a bind costs 2 captured equations per wire instead of 4 (spec 9.0; section "Captured equations"); the reflections splat their registers into the pattern lists.
- *Resource declarations.* All eight rules are straight-line with composite or template keys, so they declare their children by hand; each key carries the child's compilable arguments and exactly its register sizes (`Wire[n]`), and T2 runs every branch of every declaration. `select.py`'s `_child_sizes` is the single source of the children's sizes for the rule, the declaration, `required_wires` and the tests' expectations.
- *The select's tape-path harness* (`test_inner_select.py`). Every program has at most 64 wires, so a basis key is one 64-bit word (wire 0 the top bit, the simulator's layout); `key_values` builds the input keys from register values, `with_gradient` multiplies in the phase-gradient state that the simulator itself prepares (so the decomposed program starts with the gradient in place), and `assert_same_state` compares the final state with the expected terms over the union of their keys, so a dirty ancilla or a changed index register shows as a nonzero difference. The expected states come from `dfthc.data.fermion` and the encoded model, never from the circuit.
- *Tolerances.* Every comparison states both tolerances (HANDOFF section 6): simulator states `rtol=0, atol=1e-12`, compiled lightning states `rtol=0, atol=1e-10`.
- *FeMoco-54* values of the select are asserted on the closed forms only; the table-free reflections are also compiled at FeMoco's widths (master plan section 12.9).

---

### Task 3.1: `InnerSelect` and `InnerSelectDagger`

**Task id:** `3.1` · **Branch:** `phase3/3.1` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** G2

**Files:**
- Create: `src/dfthc/ops/select.py`, `tests/ops/test_inner_select.py`

**Interfaces:**
- Consumes: `dfthc.ops.angle_data` (`TailAngleLoad`, `SFAngleLoad`, `AngleErase` with their `required_wires`, `AngleErase.sf_fixup_high_ranges`, and the validators `SFAngleLoad._check`, `TailAngleLoad._check`; Tasks 2E.1, 2E.2); `dfthc.ops.spin` (`SpinMux`, `SpinSwap`; Task 2E.3); `dfthc.ops.givens` (`GivensLadder`, `GivensLadderDagger`; Task 2E.4); `dfthc.ops.majorana.MajoranaGate` (Task 2E.5); `dfthc.ops._validate` (`canonical_ints`, `check_int`, `check_wires`); `dfthc.compile` (`register_rule`, `resource_key`, `compiled_resources`, `FIXED_DECOMPS`); `dfthc.costs` (`inner_select`, `phase_fixup`, `tail_angle_load`, `sf_angle_load`, `spin_mux`, `spin_swap`, `givens_ladder`, `majorana_gate`, `angle_erase`); `dfthc.params` (`lg`, `WalkParams`, `FEMOCO54`, `H2_STO3G_TEST`, `H2_631G_TEST`, `MID_NAPHTHALENE`); `dfthc.data` (`WalkTables`, `load_factors`, `encoded_factors`, `vector_from_angles`, `rotated_majorana`, `annihilation`); `SparseSimulator`, `SparseState`, `ForcedPolicy`, `decompose_for_simulation` (Task 1B.2); `chosen_rules`, `verbose_program`, `classify_rule_warning` (Task 0.2).
- Produces, in `dfthc.ops.select`:
  - `InnerSelect(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up, psi_dn, phase_grad, work)`: compilable the first nine (the loads' tables as nested tuples of ints in `[0, 2**(b_rot - 1))`, `N >= 2`, `R, B, C >= 1`, `0 <= n_plus <= N`, `b_rot >= 2`, `0 <= fixup_bits <= w_b` with `2**fixup_bits <= (N - 1) b_rot + 1`), the sixteen registers by name; static `required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits) -> dict` (`x_o`: `lg(N + RC)`, `b`: `lg(B + 1)`, `r`: `lg(R)`, `angles`: `(N - 1) b_rot`, `psi_up`, `psi_dn`: `N`, `phase_grad`: `b_rot`, `work`: the largest child need, every other register 1); the rule `_inner_select`;
  - `InnerSelectDagger(...)` with the same arguments, `required_wires` and domains; the rule `_inner_select_dagger` (spec 9.18 with `qp.adjoint(MajoranaGate(...))`);
  - module-private helpers: `_check`, `_child_sizes`, `_required_wires`, `_validate`, `_select_body(dagger, ...)`, `_select_resources(dagger, ...)`. Neither class is ever adjointed or controlled.

**Acceptance tests** (43 tests in `tests/ops/test_inner_select.py`, 4 of them `slow`), on the shapes `sto3g` and `631g` (the test presets with their vendored tables and encoded factors), `random-C3` (N = 3, R = 2, B = 2, C = 3, N+ = 2, b_rot = 3; random tables) and `random-R1` (N = 2, R = 1, B = 2, C = 2; `r` has no wires):
- T3 on the tape path (6 tests, 2 `slow`): at `sto3g`, `random-C3` and `631g` (`slow`), for both classes, the sparse simulator in strict mode runs the decomposed class on a state over the branch classes the prepares leave (every SF row with every `b <= B`; failed `b`, both unloaded (`b = 2**w_b - 1`) and as the prepares leave it (`(b, r) = (0, 0)` on the last SF row: the SF load writes `sf_angles[0][0]`, and only the Majorana gate's `succ_b` control keeps the identity); `sigma_1 = 1` on an SF row; every tail row with `sigma_1 = 0` and 1; a D1 row with `succ_b = 0`; failed `x_o` on SF, tail and, at `random-C3`, `x_o = X_o`), with `sigma_0` in |+⟩, a random system state per class and the phase-gradient state; the final state equals the encoded model's action exactly (iγ̃₀γ̃₁ on the register `sigma_0` selects for SF with `b < B`; on the register `sigma_1` selects, flavour components a ± a† for D1 and a + a†, a† − a for Q1, whose ⟨+\| projection is a_ũ and a_ũ†; the identity elsewhere; adjoints for the dagger; ũ from `encoded_factors` at the presets), with every ancilla in |0⟩, under the all-ones, the alternating and a seeded forced schedule of the erase, each consuming exactly M outcomes.
- The round trip (2 tests): `InnerSelect` then `InnerSelectDagger` returns a random complex state over every value of the index registers and `sigma_0` (times all 16 system states at `sto3g`, 2 random ones at `random-C3`) exactly, under the alternating and a seeded schedule of both erases.
- The children (2 tests, `random-C3`, `random-R1`): both rules queue, in tape mode, exactly `TailAngleLoad`, `SFAngleLoad`, `SpinMux`, `SpinSwap`, `GivensLadder`, `MajoranaGate` (dagger: `Adjoint(MajoranaGate)`), `GivensLadderDagger`, `SpinSwap`, `SpinMux`, `AngleErase`, each on exactly its registers with the prefix of `work` of exactly its size; the two rules' children have identical hyperparameters and registers except the Majorana gate's wrapper; the tail load's and the erase's compilable arguments are the select's.
- T4 (7 tests, 2 `slow`): compiled, each class alone: T, M against `costs.inner_select` at both test presets, against the closed form with the erase's `ob = 0` at `random-C3` and at `random-R1`; the plain `Toffoli` count 2N + 5 and A = T − (2N + 5); the hinted `CZ` and `Z` at their expectation; `MID_NAPHTHALENE` (`slow`, both classes: 625, M = 138, A = 600); FeMoco-54 on the closed forms (61 + 291 + 2 + 108 + 2,968 + 5 + 102 = 3,537, M = 798, A = 3,424) and `required_wires` at FeMoco (work 14).
- T5 and T6 (5 tests): the verbose listing names `_inner_select` and `_inner_select_dagger`, and no dfthc operator survives at `level="device"` in a program holding both (`sto3g`); `qp.adjoint` and `qp.ctrl` of each class fail with `CompileError` "Decomposition rule not found" after the expected measuring-rule warning.
- T2 (4 tests): both rules at `sto3g` (ob = `w_b`), `631g` (ob = 1), `random-C3` (ob = 0) and `random-R1`, each pinned and the only rule under its class name; `required_wires` by hand (1 test); eight constructor violations for each class (16 tests), each a `ValueError`.

**Definition of done:** `43 passed` targeted (39 in the per-commit tier); the per-commit tier passes; `ruff check .` clean; the spec 9.0 docstring sections in order; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_inner_select.py`:

```python
"""InnerSelect and InnerSelectDagger (spec 9.18): the action against the encoded model on the tape
path, the inverse under forced schedules, the order and wire passing, counts, rule selection."""

import math
import re
import warnings

import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, compiled_resources
from dfthc.data.angles import vector_from_angles
from dfthc.data.encoded import encoded_factors
from dfthc.data.factors import load_factors
from dfthc.data.fermion import annihilation, rotated_majorana
from dfthc.data.tables import WalkTables
from dfthc.ops.angle_data import AngleErase, SFAngleLoad, TailAngleLoad
from dfthc.ops.givens import GivensLadder
from dfthc.ops.select import InnerSelect, InnerSelectDagger, _inner_select, _inner_select_dagger
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator, SparseState
from dfthc.testing.tape import decompose_for_simulation

CLASSES = {"select": (InnerSelect, _inner_select), "dagger": (InnerSelectDagger,
                                                              _inner_select_dagger)}


def vendored(system, p):
    """The preset's angle tables and its encoded factors (spec 8.5)."""
    factors = load_factors(system)
    tables = WalkTables.from_factors(factors, p)
    return tables.tail_angles, tables.sf_angles, encoded_factors(p, tables, factors)


def random_tables(p, seed):
    """Random angle tables of p's shape, every angle below 2**(b_rot - 1)."""
    rng = np.random.default_rng(seed)
    bound = 2 ** (p.b_rot - 1)
    tail = tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.N))
    sf = tuple(tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.B))
               for _ in range(p.R))
    return tail, sf, None


# C = 3: r = x_o // 3 is no bit field of x_o, and the tail rows [6, 9) start off a power of two;
# N = 3 gives the ladder two rungs (their order and the register order matter), N+ = 2 a Q1 row.
RANDOM_C3 = WalkParams(N=3, R=2, B=2, C=3, n_plus=2, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
# R = 1: r has no wires and the SF address is b alone.
RANDOM_R1 = WalkParams(N=2, R=1, B=2, C=2, n_plus=2, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
SHAPES = {
    "sto3g": (H2_STO3G_TEST, *vendored("h2_sto3g_R2_B2_C1", H2_STO3G_TEST)),
    "631g": (H2_631G_TEST, *vendored("h2_631g_R2_B2_C2", H2_631G_TEST)),
    "random-C3": (RANDOM_C3, *random_tables(RANDOM_C3, seed=5)),
    "random-R1": (RANDOM_R1, *random_tables(RANDOM_R1, seed=0)),
}


def compilable(name, ob=None):
    p, tail, sf, _ = SHAPES[name]
    return (tail, sf, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b if ob is None else ob)


def layout(args):
    """Consecutive wires for every register, in the order of required_wires."""
    wires, start = {}, 0
    for name, size in InnerSelect.required_wires(*args).items():
        wires[name], start = list(range(start, start + size)), start + size
    return wires, start


# --------------------------------------------------------------------------- the reference


def directions(name):
    """u~ of every SF entry (r, b) and tail row j: the encoded model's unit vectors at the
    vendored presets (encoded_factors), vector_from_angles of the tables otherwise (spec 6.3)."""
    p, tail, sf, enc = SHAPES[name]
    if enc is not None:
        return ({(r, b): enc.U[r, b] for r in range(p.R) for b in range(p.B)},
                [*enc.u_plus, *enc.u_minus])
    return ({(r, b): vector_from_angles(sf[r][b], p.b_rot) for r in range(p.R)
             for b in range(p.B)}, [vector_from_angles(row, p.b_rot) for row in tail])


def branch_classes(p):
    """(x_o, succ_x_o, sigma_1, b, succ_b, r) of the branches the T3 runs, as the prepares leave
    them: every SF row with every b <= B (both flags set, r = x_o // C), a failed b (succ_b = 0)
    and sigma_1 = 1 on SF rows, every tail row with sigma_1 = 0 and 1, a D1 row with succ_b = 0,
    and failed x_o (succ_x_o = 0) on an SF row, a tail row and, if it exists, x_o = X_o. The
    inner read loads a zero entry at a failed b, so the alias swap leaves (b, r) = (0, 0) there:
    the SF load then writes sf_angles[0][0] and b_neq_B = 1, and only the Majorana gate's succ_b
    control keeps the branch the identity (spec 8.5, ledger row 2D.2). That class sits on the
    last SF row, whose own rank is not 0; a failed b at b = 2**w_b - 1 (nothing loaded) stays."""
    classes = [(x, 1, 0, b, 1, x // p.C) for x in range(p.RC) for b in range(p.B + 1)]
    classes += [(0, 1, 0, 2**p.w_b - 1, 0, 0), (p.RC - 1, 1, 0, 0, 0, 0),
                (p.RC - 1, 1, 1, 0, 1, (p.RC - 1) // p.C)]
    classes += [(p.RC + j, 1, s, 0, 1, 0) for j in range(p.N) for s in (0, 1)]
    classes += [(p.RC, 1, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0), (p.RC, 0, 1, 1, 1, 0)]
    if p.X_o < 2**p.w_x:
        classes.append((p.X_o, 0, 0, 0, 1, 0))
    return classes


def flavour_actions(name, branch):
    """The select's action on psi_up (x) psi_dn for sigma_0 = 0 and 1, from the encoded model.

    SF with b < B and both flags: i gamma~_0 gamma~_1 on the register sigma_0 selects (psi_dn for
    0, which is spin 1 of dfthc.data.fermion). D1/Q1 with succ_x_o: on the register sigma_1
    selects, with a = sum_j u~_j a_j, D1 gives a + a^dag and a - a^dag, Q1 a + a^dag and
    a^dag - a, whose <+| projection on sigma_0 is a (D1) and a^dag (Q1). The identity elsewhere."""
    p = SHAPES[name][0]
    sf_u, tail_u = directions(name)
    x_o, succ_x_o, sigma_1, b, succ_b, r = branch
    identity = np.eye(4**p.N)
    if not succ_x_o or (x_o < p.RC and not (succ_b and b < p.B)):
        return [identity, identity]
    if x_o < p.RC:
        u = sf_u[r, b]
        return [1j * rotated_majorana(u, 0, 1 - f, p.N) @ rotated_majorana(u, 1, 1 - f, p.N)
                for f in (0, 1)]
    j, spin = x_o - p.RC, 1 - sigma_1
    lower = sum(tail_u[j][k] * annihilation(k, spin, p.N) for k in range(p.N))
    upper = lower.conj().T
    return [lower + upper, lower - upper] if j < p.n_plus else [lower + upper, upper - lower]


def key_values(num_wires, registers, values):
    """Basis keys of the simulator (one 64-bit word for at most 64 wires, wire 0 the top bit)
    with ``values[name]`` (arrays of ints) big-endian in ``registers[name]``; registers not named
    are 0."""
    assert num_wires <= 64
    total = np.zeros(len(next(iter(values.values()))), dtype=np.uint64)
    one = np.uint64(1)
    for name, column in values.items():
        wires, column = registers[name], np.asarray(column, dtype=np.uint64)
        for i, wire in enumerate(wires):
            total |= ((column >> np.uint64(len(wires) - 1 - i)) & one) << np.uint64(63 - wire)
    return total[:, None]


def gradient_state(b_rot):
    """The phase-gradient state the simulator prepares on b_rot wires."""
    return SparseSimulator(b_rot).run([qp.PhaseGradientStatePrep(range(b_rot))]).to_dense()


def with_gradient(wires, keys, amps, b_rot):
    """``keys``, ``amps`` (phase_grad in |0>) times the phase-gradient state on phase_grad."""
    gradient = gradient_state(b_rot)
    g = np.arange(2**b_rot)
    shift = key_values(64, {"g": wires["phase_grad"]}, {"g": g})[:, 0]
    return ((keys[:, 0][:, None] | shift[None, :]).ravel()[:, None],
            (amps[:, None] * gradient[None, :]).ravel())


def assert_same_state(state, keys, amps):
    """The simulated state equals the terms (keys, amps) exactly: every amplitude to 1e-12, every
    other basis state (dirty ancillas, changed index registers) zero."""
    all_keys = np.concatenate([state.keys[:, 0], keys[:, 0]])
    unique, inverse = np.unique(all_keys, return_inverse=True)
    diff = np.concatenate([state.amps, -amps])
    total = (np.bincount(inverse.ravel(), diff.real, len(unique))
             + 1j * np.bincount(inverse.ravel(), diff.imag, len(unique)))
    assert np.allclose(total, 0, rtol=0, atol=1e-12), np.abs(total).max()


def schedules(num, rng):
    """The all-ones, the alternating and a seeded forced schedule of ``num`` outcomes (an
    all-ones schedule sees only each address's mask parity, ledger row 2D.2)."""
    return {"ones": [1] * num, "alternating": [i % 2 for i in range(num)],
            "seeded": rng.integers(0, 2, num).tolist()}


def simulate(operations, num_wires, keys, amps, schedule):
    simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
    state = simulator.run(operations, initial_state=SparseState(num_wires, keys, amps))
    assert simulator.policy.position == len(schedule)
    return state


# --------------------------------------------------------------------------- semantics


@pytest.mark.parametrize("kind", CLASSES)
@pytest.mark.parametrize("name", ["sto3g", "random-C3",
                                  pytest.param("631g", marks=pytest.mark.slow)])
def test_action_against_the_encoded_model(name, kind):
    """T3 on the tape path (spec 7.4 rule 9), as the sparse simulator runs it: on the branch
    classes of branch_classes, with sigma_0 in |+>, a random system state per class and the
    phase-gradient state on phase_grad, every
    term equals the encoded model's generator piece (flavour_actions; for InnerSelectDagger its
    adjoint), every ancilla is |0> and the index registers are unchanged, under the all-ones, the
    alternating and a seeded forced schedule of the erase (strict mode). The simulator asserts
    every AND precondition, which lightning cannot see (ledger row 2E.5). sto3g is the H2 test
    shape; random-C3 has a ladder of two rungs, a Q1 row and C = 3 (rank selection); 631g (N = 4,
    three D1 rows and a Q1 row, about 200,000 terms) runs in the nightly tier."""
    p = SHAPES[name][0]
    cls, _ = CLASSES[kind]
    args = compilable(name)
    wires, num_wires = layout(args)
    rng = np.random.default_rng(7)
    classes = branch_classes(p)
    dim = 4**p.N
    states = rng.normal(size=(len(classes), dim)) + 1j * rng.normal(size=(len(classes), dim))
    states /= np.linalg.norm(states) * math.sqrt(2)  # the 1 / sqrt 2 of sigma_0 = |+>
    rows = [(k, f, x) for k in range(len(classes)) for f in (0, 1) for x in range(dim)]
    k, f, x = (np.array(column) for column in zip(*rows, strict=True))
    index = np.array(classes)[k]
    values = dict(zip(("x_o", "succ_x_o", "sigma_1", "b", "succ_b", "r"), index.T, strict=True))
    registers = wires | {"system": wires["psi_up"] + wires["psi_dn"]}
    keys = key_values(num_wires, registers, values | {"sigma_0": f, "system": x})
    actions = [flavour_actions(name, branch) for branch in classes]
    if kind == "dagger":
        actions = [[action.conj().T for action in pair] for pair in actions]
    outputs = np.array([[actions[c][s] @ states[c] for s in (0, 1)] for c in range(len(classes))])
    initial = with_gradient(wires, keys, states[k, x], p.b_rot)
    expected = with_gradient(wires, keys, outputs[k, f, x], p.b_rot)

    operations = decompose_for_simulation(lambda: cls(*args, **wires)).operations
    for schedule in schedules((p.N - 1) * p.b_rot + 3, rng).values():
        assert_same_state(simulate(operations, num_wires, *initial, schedule), *expected)


@pytest.mark.parametrize(("name", "num_system"), [("sto3g", 16), ("random-C3", 2)])
def test_dagger_inverts_the_select_on_every_branch(name, num_system):
    """InnerSelectDagger after InnerSelect is the identity on every branch, the ones no prepare
    reaches included (failed x_o >= X_o, r != x_o // C, sigma_0 not in |+>):
    from a random complex state over every value of the index registers and sigma_0, times
    num_system random system basis states, the sparse simulator returns that state exactly with
    every ancilla in |0>, under the alternating and a seeded schedule of both erases."""
    p = SHAPES[name][0]
    args = compilable(name)
    wires, num_wires = layout(args)
    rng = np.random.default_rng(11)
    index = ("x_o", "succ_x_o", "sigma_1", "b", "succ_b", "sigma_0", "r")
    sizes = [len(wires[reg]) for reg in index]
    system = rng.choice(4**p.N, num_system, replace=False)
    grid = np.array(np.meshgrid(*[np.arange(2**n) for n in sizes], system,
                                indexing="ij")).reshape(len(sizes) + 1, -1)
    registers = wires | {"system": wires["psi_up"] + wires["psi_dn"]}
    keys = key_values(num_wires, registers, dict(zip((*index, "system"), grid, strict=True)))
    amps = rng.normal(size=len(keys)) + 1j * rng.normal(size=len(keys))
    initial = with_gradient(wires, keys, amps / np.linalg.norm(amps), p.b_rot)

    def program():
        InnerSelect(*args, **wires)
        InnerSelectDagger(*args, **wires)

    operations = decompose_for_simulation(program).operations
    outcomes = schedules(2 * ((p.N - 1) * p.b_rot + 3), rng)
    for schedule in (outcomes["alternating"], outcomes["seeded"]):
        assert_same_state(simulate(operations, num_wires, *initial, schedule), *initial)


def children(rule, args, wires):
    """The rule's operators in tape mode, with their registers."""
    with qp.capture.pause(), qp.queuing.AnnotatedQueue() as queue:
        rule(*args, **wires)
    return list(queue.queue)


def registers(op):
    op = getattr(op, "base", op)
    return {name: list(getattr(op, name)) for name in op.wire_argnames}


@pytest.mark.parametrize("name", ["random-C3", "random-R1"])
def test_children_order_and_wires(name):
    """Spec 9.18's order, with each child on exactly its registers (spec 9.0): the loads, the
    spin multiplexer and swap, the ladder, the Majorana gate on psi_dn[0] with flavour sigma_0
    (InnerSelectDagger: Adjoint(MajoranaGate), the instance form), the explicit
    GivensLadderDagger (never an adjointed ladder), the swap, the multiplexer, the erase; each
    child's work is the prefix of exactly its required size. Both selects load first and erase
    last, and their children differ only in the Majorana gate, so they share every other target
    (identical compilable arguments and register sizes)."""
    p, tail, sf, _ = SHAPES[name]
    args = compilable(name)
    w, _ = layout(args)
    tail_work = TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot)["work"]
    sf_work = SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot)["work"]
    erase_work = AngleErase.required_wires(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot,
                                           p.w_b)["work"]
    mux = {"G0": w["G0"], "sigma_0": w["sigma_0"], "sigma_1": w["sigma_1"], "sg": w["sg"]}
    swap = {"sg": w["sg"], "psi_up": w["psi_up"], "psi_dn": w["psi_dn"]}
    ladder = {"system": w["psi_dn"], "angles": w["angles"], "phase_grad": w["phase_grad"],
              "work": w["work"][:p.b_rot - 1]}
    majorana = {"succ_x_o": w["succ_x_o"], "succ_b": w["succ_b"], "G0": w["G0"], "G1": w["G1"],
                "b_neq_B": w["b_neq_B"], "flavour": w["sigma_0"], "target": w["psi_dn"][:1],
                "work": w["work"][:2]}
    expected = [
        ("TailAngleLoad", {"x_o": w["x_o"], "angles": w["angles"], "G0": w["G0"], "G1": w["G1"],
                           "work": w["work"][:tail_work]}),
        ("SFAngleLoad", {"r": w["r"], "b": w["b"], "G0": w["G0"], "angles": w["angles"],
                         "b_neq_B": w["b_neq_B"], "work": w["work"][:sf_work]}),
        ("SpinMux", mux), ("SpinSwap", swap), ("GivensLadder", ladder), ("MajoranaGate", majorana),
        ("GivensLadderDagger", ladder), ("SpinSwap", swap), ("SpinMux", mux),
        ("AngleErase", {"angles": w["angles"], "b_neq_B": w["b_neq_B"], "G0": w["G0"],
                        "G1": w["G1"], "x_o": w["x_o"], "r": w["r"], "b": w["b"],
                        "work": w["work"][:erase_work]}),
    ]
    forward, backward = (children(rule, args, w) for _, rule in CLASSES.values())
    assert [(op.name, registers(op)) for op in forward] == expected
    assert [op.name for op in backward] == [
        "Adjoint(MajoranaGate)" if name == "MajoranaGate" else name for name, _ in expected]
    assert [registers(op) for op in backward] == [regs for _, regs in expected]
    assert [op.hyperparameters for op in forward] == [
        getattr(op, "base", op).hyperparameters for op in backward]
    assert forward[0].hyperparameters == {"tail_angles": tail, "n_plus": p.n_plus, "RC": p.RC,
                                          "X_o": p.X_o, "b_rot": p.b_rot}
    assert forward[-1].hyperparameters == {
        "tail_angles": tail, "sf_angles": sf, "n_plus": p.n_plus, "N": p.N, "R": p.R, "B": p.B,
        "C": p.C, "b_rot": p.b_rot, "sf_onehot_bits": p.w_b}


# --------------------------------------------------------------------------- resources


def expected_cost(p, ob):
    """costs.inner_select with the erase's SF fix-up at one-hot width ob (spec 9.17); the walk's
    ob = w_b gives costs.inner_select itself."""
    def sf(bits):
        return costs.phase_fixup(AngleErase.sf_fixup_high_ranges(p.R, p.B, bits), bits, True)

    walk = costs.inner_select(p)
    return costs.Cost(walk.toffoli + sf(ob).toffoli - sf(p.w_b).toffoli, walk.measurements)


def assert_counts(resources, p, ob):
    """T4 and the T6 signature: T and M exact; the plain Toffolis are the 2N + 5 of the spin
    operators and the Majorana gate, so A = T - (2N + 5); the erase's hinted conditionals count
    their expectation (nothing else in the select emits CZ or Z): R B conditional CZ (ob >= 1) or
    Z (ob = 0) on the SF fix-up and N conditional Z on the tail fix-up, summed per gate type and
    rounded up once (L-17)."""
    expected = expected_cost(p, ob)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.counts["Toffoli"] == 2 * p.N + 5
    assert resources.measured_and_uncomputes == resources.toffoli - (2 * p.N + 5)
    hinted = ({"CZ": math.ceil(p.R * p.B / 2), "PauliZ": math.ceil(p.N / 2)} if ob
              else {"CZ": 0, "PauliZ": math.ceil((p.R * p.B + p.N) / 2)})
    assert {gate: resources.counts.get(gate, 0) for gate in hinted} == hinted


@pytest.mark.compiled
@pytest.mark.parametrize(("name", "ob"), [("sto3g", None), ("631g", None), ("random-C3", 0),
                                          ("random-R1", None)],
                         ids=["sto3g", "631g", "C3-ob0", "R1"])
def test_counts(name, ob):
    """T4 compiled, each class alone: costs.inner_select at both test presets (T = 33 and 65,
    M = 7 and 15), the ob = 0 fallback of the erase's SF fix-up at random-C3, and R = 1."""
    p = SHAPES[name][0]
    args = compilable(name, ob)
    wires, num_wires = layout(args)
    for cls, _ in CLASSES.values():
        assert_counts(compiled_resources(lambda cls=cls: cls(*args, **wires), num_wires), p,
                      p.w_b if ob is None else ob)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("kind", CLASSES)
def test_counts_at_mid_naphthalene(kind):
    """T4 at MID_NAPHTHALENE (naphthalene tables, nightly tier): T = 625, M = 138, A = 600."""
    p = MID_NAPHTHALENE
    tail, sf, _ = vendored("naphthalene", p)
    args = (tail, sf, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    wires, num_wires = layout(args)
    cls, _ = CLASSES[kind]
    assert_counts(compiled_resources(lambda: cls(*args, **wires), num_wires), p, p.w_b)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 (spec 9.24): 61 + 291 + 2 + 108 + 2,968 + 5 + 102 = 3,537 Toffolis and
    M = 798 for each select, A = 3,537 - 113 = 3,424; the select's work is the ladder's 14 wires,
    within the walk's pool of 44."""
    p = FEMOCO54
    parts = [costs.tail_angle_load(p), costs.sf_angle_load(p), 2 * costs.spin_mux(),
             2 * costs.spin_swap(p), 2 * costs.givens_ladder(p), costs.majorana_gate(),
             costs.angle_erase(p)]
    assert [part.toffoli for part in parts] == [61, 291, 2, 108, 2968, 5, 102]
    assert costs.inner_select(p) == costs.Cost(3537, 798)
    assert costs.inner_select(p).toffoli - (2 * p.N + 5) == 3424
    tail = tuple((0,) * 53 for _ in range(54))
    sf = tuple(tuple((0,) * 53 for _ in range(27)) for _ in range(10))
    sizes = InnerSelect.required_wires(tail, sf, 54, 10, 27, 27, 27, 15, 5)
    assert sizes == {"x_o": 9, "succ_x_o": 1, "sigma_1": 1, "b": 5, "succ_b": 1, "sigma_0": 1,
                     "r": 4, "angles": 795, "G0": 1, "G1": 1, "b_neq_B": 1, "sg": 1,
                     "psi_up": 54, "psi_dn": 54, "phase_grad": 15, "work": 14}
    assert GivensLadder.required_wires(15, 54)["work"] == 14


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6 and T5 at H2_STO3G_TEST: the verbose listing names both pinned rules; at
    level="device" no dfthc operator survives in a program holding both selects (L-20)."""
    args = compilable("sto3g")
    wires, num_wires = layout(args)

    def both():
        InnerSelect(*args, **wires)
        InnerSelectDagger(*args, **wires)

    qp.analyze(verbose_program(both, num_wires), level="user")()
    captured = capfd.readouterr()
    assert {"_inner_select", "_inner_select_dagger"} <= set(chosen_rules(captured.out
                                                                         + captured.err))
    compiled_resources(both, num_wires, level="device")


@pytest.mark.compiled
@pytest.mark.parametrize("cls", [InnerSelect, InnerSelectDagger],
                         ids=["select", "dagger"])
@pytest.mark.parametrize("wrap", [qp.adjoint, lambda op: qp.ctrl(op, control=40)],
                         ids=["adjoint", "control"])
def test_no_adjoint_or_controlled_path(wrap, cls):
    """Spec 7.4 rules 5 and 6: the adjoint of the select is the class InnerSelectDagger, and
    neither class is ever adjointed or controlled. Wrapped anyway, Catalyst drops the measuring
    candidates with the expected warning (rule 12), and the compile fails loudly."""
    args = compilable("sto3g")
    wires, _ = layout(args)
    key = f"Adjoint({cls.__name__})" if wrap is qp.adjoint else f"C({cls.__name__})"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(lambda: wrap(cls(*args, **wires)), num_wires=41)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the ")]
    named = [message for message in skipped if classify_rule_warning(message) != "noise"]
    assert named and {classify_rule_warning(message) for message in named} == {"expected"}


@pytest.mark.parametrize(("name", "ob"), [("sto3g", None), ("631g", 1), ("random-C3", 0),
                                          ("random-R1", None)],
                         ids=["sto3g", "631g-ob1", "C3-ob0", "R1"])
def test_rules_are_valid(name, ob):
    """T2 (tape mode) for both rules at four sizes, every branch of the hand-written resources
    (the dagger's adjoint Majorana key, the erase at three one-hot widths, R = 1); each rule is
    pinned and the only one registered under its class name (spec 7.4 rules 7 and 8)."""
    args = compilable(name, ob)
    wires, _ = layout(args)
    for cls, rule in CLASSES.values():
        assert FIXED_DECOMPS[cls.__name__] is rule
        assert list(qp.list_decomps(cls)) == [rule]
        qp.testing.assert_valid_decomp_rule(cls(*args, **wires), rule, check_matrix=False)


def test_required_wires():
    """By hand at H2_STO3G_TEST: work is the largest child need, max(1, 3, 3, 2, 2) = 3 (the SF
    load's w_r + w_b and the ladder's b_rot - 1)."""
    assert InnerSelect.required_wires(*compilable("sto3g")) == {
        "x_o": 2, "succ_x_o": 1, "sigma_1": 1, "b": 2, "succ_b": 1, "sigma_0": 1, "r": 1,
        "angles": 4, "G0": 1, "G1": 1, "b_neq_B": 1, "sg": 1, "psi_up": 2, "psi_dn": 2,
        "phase_grad": 4, "work": 3}
    assert InnerSelectDagger.required_wires(*compilable("random-R1"))["r"] == 0


@pytest.mark.parametrize(("change", "match"), [
    ({"fixup_bits": 3}, r"fixup_bits must lie in \[0, 2\]"),
    ({"N": 3}, "tail_angles must have N = 3 rows"),
    ({"tail_angles": ((1,), (8,))}, r"angles must lie in \[0, 2\*\*\(b_rot - 1\)\)"),
    ({"n_plus": 3}, r"n_plus must lie in \[0, 2\]"),
    ({"C": 0}, r"C must lie in \[1, None\]"),
    ({"phase_grad": [0, 1, 2]}, "'phase_grad' has 3 wires; expected 4"),
    ({"psi_dn": [0]}, "'psi_dn' has 1 wires; expected 2"),
    ({"work": [0, 1]}, "'work' has 2 wires; expected 3"),
])
@pytest.mark.parametrize("cls", [InnerSelect, InnerSelectDagger], ids=["select", "dagger"])
def test_constructor_validates(cls, change, match):
    args = dict(zip(InnerSelect.compilable_argnames, compilable("sto3g"), strict=True))
    wires, _ = layout(compilable("sto3g"))
    with pytest.raises(ValueError, match=match):
        cls(**(args | wires | change))
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_inner_select.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.ops.select'`.

- [ ] **Step 3: Write the select**

Create `src/dfthc/ops/select.py`:

```python
r"""The inner select of the block encoding and its explicit adjoint (spec 9.18; steps 10 to 18).

**The circuit** (spec 3.5 and 9.18). For every branch of the index registers ``InnerSelect``
loads the branch's Givens angles and family flags (:class:`~dfthc.ops.angle_data.TailAngleLoad`,
then :class:`~dfthc.ops.angle_data.SFAngleLoad`), routes the spin register that ``sg`` selects into
``psi_dn`` (:class:`~dfthc.ops.spin.SpinMux`, :class:`~dfthc.ops.spin.SpinSwap`), applies
:math:`U^\dagger\,\mathrm{Maj}\,U` there (:class:`~dfthc.ops.givens.GivensLadder`,
:class:`~dfthc.ops.majorana.MajoranaGate`, :class:`~dfthc.ops.givens.GivensLadderDagger`), undoes
the routing in reverse order, and erases the loaded data by measurement
(:class:`~dfthc.ops.angle_data.AngleErase`). :class:`InnerSelectDagger` is the same sequence with
``qp.adjoint(MajoranaGate(...))``: it also loads first and erases last, and it inverts
:class:`InnerSelect` on every branch, because the select is block diagonal in the index
registers and only the Majorana gate is not Hermitian.

**Wires.** Both classes take the index registers they read (``x_o``, ``succ_x_o``, ``sigma_1``
of the outer group; ``b``, ``succ_b``, ``sigma_0``, ``r`` of the inner group) and the ``select``
and ``system`` registers by name, in ``WalkLayout`` order, plus ``phase_grad`` and ``work``. The
rule unpacks every register into a list once and passes each child only the registers it needs,
with the prefix of ``work`` it needs (spec 9.0, wire passing). Both rules pass ``AngleErase``,
the loads, the ladders and the spin operators identical compilable arguments and register sizes,
so a program holding both selects captures each of those targets once.
"""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.angle_data import AngleErase, SFAngleLoad, TailAngleLoad
from dfthc.ops.givens import GivensLadder, GivensLadderDagger
from dfthc.ops.majorana import MajoranaGate
from dfthc.ops.spin import SpinMux, SpinSwap
from dfthc.params import lg

_WIRES = ("x_o", "succ_x_o", "sigma_1", "b", "succ_b", "sigma_0", "r", "angles", "G0", "G1",
          "b_neq_B", "sg", "psi_up", "psi_dn", "phase_grad", "work")


def _check(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits) -> tuple:
    """The compilable arguments as canonical tuples and ints, each checked against its domain
    (spec 9.0); ``ValueError`` otherwise. The domains are those of ``AngleErase``."""
    N = check_int("N", N, low=2)
    C = check_int("C", C, low=1)
    sf_angles, R, B, b_rot = SFAngleLoad._check(sf_angles, R, B, b_rot)
    rows = canonical_ints("tail_angles", tail_angles)
    if not isinstance(rows, tuple) or len(rows) != N or len(sf_angles[0][0]) != N - 1:
        raise ValueError(f"tail_angles must have N = {N} rows and sf_angles N - 1 = {N - 1} "
                         "angles per entry")
    tail_angles, n_plus, _, _, _ = TailAngleLoad._check(rows, n_plus, R * C, N + R * C, b_rot)
    fixup_bits = check_int("fixup_bits", fixup_bits, low=0, high=lg(B + 1))
    if fixup_bits and 2**fixup_bits > (N - 1) * b_rot + 1:
        raise ValueError(f"the SF one-hot over fixup_bits = {fixup_bits} bits needs "
                         f"{2**fixup_bits} wires, more than the {(N - 1) * b_rot + 1} reset angle "
                         "and b_neq_B wires")
    return tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits


def _child_sizes(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits) -> dict:
    """``required_wires`` of every child, keyed by child; each child gets a prefix of ``work``
    of exactly its size."""
    return {
        "tail": TailAngleLoad.required_wires(tail_angles, n_plus, R * C, N + R * C, b_rot),
        "sf": SFAngleLoad.required_wires(sf_angles, R, B, b_rot),
        "mux": SpinMux.required_wires(),
        "swap": SpinSwap.required_wires(N),
        "ladder": GivensLadder.required_wires(b_rot, N),
        "majorana": MajoranaGate.required_wires(),
        "erase": AngleErase.required_wires(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                                           fixup_bits),
    }


def _required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits) -> dict:
    children = _child_sizes(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)
    return {"x_o": lg(N + R * C), "succ_x_o": 1, "sigma_1": 1, "b": lg(B + 1), "succ_b": 1,
            "sigma_0": 1, "r": lg(R), "angles": (N - 1) * b_rot, "G0": 1, "G1": 1, "b_neq_B": 1,
            "sg": 1, "psi_up": N, "psi_dn": N, "phase_grad": b_rot,
            "work": max(sizes["work"] for sizes in children.values() if "work" in sizes)}


def _validate(name: str, compilable: tuple, wires: tuple) -> tuple:
    compilable = _check(*compilable)
    check_wires(name, dict(zip(_WIRES, wires, strict=True)), _required_wires(*compilable))
    return compilable


class InnerSelect(Operator2):
    r"""The inner select (steps 10 to 18): on every branch, :math:`U^\dagger\,\mathrm{Maj}\,U`
    with the branch's Givens ladder :math:`U` on the spin register that ``sg`` selects.

    **Action.** Block diagonal in the index registers ``x_o``, ``succ_x_o``, ``sigma_1``, ``b``,
    ``succ_b``, ``r`` and diagonal in ``sigma_0``; every ancilla starts and ends in
    :math:`|0\rangle`. On a basis state of the index registers let
    :math:`G_0 = [RC \le x_o < X_o]`, :math:`G_1 = G_0 \wedge [x_o - RC \ge N_+]`, ``x`` the
    angle integers ``tail_angles[x_o - RC]`` if :math:`G_0`, else ``sf_angles[r][b]`` if
    :math:`(r, b) \in [0, R) \times [0, B)`, else zeros, and
    :math:`\tilde u` = ``vector_from_angles(x, b_rot)`` (spec 6.3). The spin register is
    selected by :math:`\sigma = \sigma_{G_0}`: ``psi_dn`` for :math:`\sigma = 0`, ``psi_up`` for
    :math:`\sigma = 1`. With :math:`U` = ``ladder_matrix(x, b_rot, N)`` on that register and
    ``Maj`` the :class:`~dfthc.ops.majorana.MajoranaGate` on (``succ_x_o``, ``succ_b``,
    :math:`G_0`, :math:`G_1`, ``b_neq_B`` = :math:`\neg G_0 \wedge [(r, b) \in [0, R) \times
    [0, B)]`, flavour ``sigma_0``, target the register's qubit 0), the system and ``sigma_0`` get
    :math:`U^\dagger\,\mathrm{Maj}\,U`. On
    the valid branches of the walk this is a generator piece of spec 3.2, with
    :math:`\tilde\gamma_f = \sum_j \tilde u_j \gamma_{j,f}` on the selected spin register:

    - SF (``x_o = rC + c < RC``, ``b < B``, both success flags set):
      :math:`i\tilde\gamma_0\tilde\gamma_1`, on the register ``sigma_0`` selects;
    - D1 (``RC <= x_o < RC + N+``) and Q1 (the other tail rows), ``succ_x_o`` set: with
      ``sigma_0`` projected onto :math:`|+\rangle` before and after,
      :math:`a_{\tilde u} = (\tilde\gamma_0 + i\tilde\gamma_1)/2` (D1) or
      :math:`a_{\tilde u}^\dagger` (Q1) on the register ``sigma_1`` selects, whatever
      ``succ_b`` (spec 9.16);
    - the identity on ``b = B``, on SF rows with ``succ_b = 0`` and wherever ``succ_x_o = 0``.

    **Registers.** :math:`w_x = \lg(N + RC)`, :math:`w_b = \lg(B + 1)`, :math:`w_r = \lg R`.

    ============== ======================= ============================ =======================
    name           width                   input                        output
    ============== ======================= ============================ =======================
    ``x_o``        :math:`w_x`             any (read only)              unchanged
    ``succ_x_o``   1                       any (read only)              unchanged
    ``sigma_1``    1                       any (read only)              unchanged
    ``b``          :math:`w_b`             any (read only)              unchanged
    ``succ_b``     1                       any (read only)              unchanged
    ``sigma_0``    1                       any                          a control (diagonal)
    ``r``          :math:`w_r` (0 if R=1)  any (read only)              unchanged
    ``angles``     :math:`(N - 1) b_{rot}` ``|0>``                      ``|0>``
    ``G0``         1                       ``|0>``                      ``|0>``
    ``G1``         1                       ``|0>``                      ``|0>``
    ``b_neq_B``    1                       ``|0>``                      ``|0>``
    ``sg``         1                       ``|0>``                      ``|0>``
    ``psi_up``     ``N``                   any                          the action above
    ``psi_dn``     ``N``                   any                          the action above
    ``phase_grad`` :math:`b_{rot}`         phase-gradient state         unchanged (catalysed)
    ``work``       :meth:`required_wires`  ``|0>``                      ``|0>``
    ============== ======================= ============================ =======================

    **Compilable arguments.** ``tail_angles`` (``N x (N - 1)``, D1 rows first) and
    ``sf_angles`` (``R x B x (N - 1)``), ints in ``[0, 2**(b_rot - 1))``
    (``WalkTables.tail_angles`` and ``sf_angles``); ``N >= 2``; ``R``, ``B``, ``C >= 1``;
    ``n_plus`` (``0 <= N+ <= N``); ``b_rot >= 2``; ``fixup_bits``, the SF fix-up's one-hot bits
    of ``AngleErase`` (``0 <= fixup_bits <= w_b`` with
    ``2**fixup_bits <= (N - 1) b_rot + 1``; the walk passes ``w_b``).

    **Pre- and postconditions.** The registers are pairwise disjoint (unchecked); the ancillas
    and ``work`` start in ``|0>`` and end there for every measurement outcome of the erase;
    ``phase_grad`` holds the state ``PhaseGradientStatePrep`` prepares. In the walk the index
    registers hold what ``InnerPrepare`` and ``OuterPrepare`` wrote, and the select writes none
    of them (``sigma_0`` gets only a phase), so ``InnerUnprepare``'s preconditions hold
    afterwards.

    **Decomposition** (spec 9.18), each child on its own registers and a prefix of ``work``:
    ``TailAngleLoad(x_o; angles, G0, G1)``; ``SFAngleLoad(r, b, G0; angles, b_neq_B)``, enabled
    on ``G0 = 0``; ``SpinMux(G0, sigma_0, sigma_1; sg)``; ``SpinSwap(sg; psi_up, psi_dn)``;
    ``GivensLadder(psi_dn, angles, phase_grad)``; ``MajoranaGate(succ_x_o, succ_b, G0, G1,
    b_neq_B, flavour=sigma_0, target=psi_dn[0])``; ``GivensLadderDagger(psi_dn, angles,
    phase_grad)``; ``SpinSwap``; ``SpinMux``; ``AngleErase(angles, b_neq_B, G0, G1; x_o, r, b)``
    with ``sf_onehot_bits = fixup_bits``. Between the loads and the erase, ``angles``, ``G0``,
    ``G1`` and ``b_neq_B`` are only controls or the adders' unchanged addend, so the erase's
    precondition holds.

    **Adjoint and control.** Never adjointed or controlled: its rule contains the measured
    ``AngleErase`` (spec 7.4 rules 5 and 6). The adjoint is the class :class:`InnerSelectDagger`.

    **Cost.** :func:`dfthc.costs.inner_select` for ``fixup_bits = w_b``:
    :math:`T = T_{tail} + T_{SF} + 2 + 2N + 4(b_{rot} - 1)(N - 1) + 5 + T_{angle\,erase}`,
    :math:`M = (N - 1) b_{rot} + 3`; ``A = T - (2N + 5)``, the plain ``Toffoli`` gates being the
    two ``SpinMux`` Toffolis, the ``2N`` ``SpinSwap`` CSWAPs and three in the Majorana gate. At
    FeMoco-54: :math:`61 + 291 + 2 + 108 + 2968 + 5 + 102 = 3537`, ``M = 798``, ``A = 3424``. For
    another ``fixup_bits`` the erase's SF term is ``costs.phase_fixup`` over
    ``AngleErase.sf_fixup_high_ranges``. ``CNOT`` and ``PauliX`` counts are upper bounds (the
    loads' reads, the erase's resets).

    **Paper and dfthcbe.** Steps 10 to 18 (spec 3.5): the paper's inner SELECT,
    :math:`U_{Rot}^\dagger\,\mathrm{Maj}\,U_{Rot}` with its spin swap, data loads and erase (PRX
    App. B; arXiv:2502.15882v1 App. B.3 for the spin selection, B.5 and Eqs. (93), (94) for the
    rotations, B.6 and Fig. 4 for the Majorana gate, Table 9 for the loads and the erase).
    dfthcbe: ``InnerSelect`` (``dfthcbe/select.py``), 3,502 Toffolis at FeMoco-54; ours is +35:
    the tail load +5 and the SF load +20 (exact iteration), the erase +12 (one-hot over 32
    positions +4, the SF fix-up enabled on :math:`G_0` +1, exact iteration +7), the spin
    multiplexer -2 (one Toffoli per application where dfthcbe has two ``MultiControlX``). The
    paper's Table III gives 3,710 per select with the tail fix-up its Eq. (B50) (v1 Eq. (123))
    charges to the inner erase; ours is -173: the Givens ladders -212 (``rz-addsub``), the loads
    +28 (exact iteration), the erase +11 (exact iteration, the one-hot over 32 positions), the
    multiplexer +2 (Table III omits it; App. B.3 and B.7.1 count two per select, as we do),
    the Majorana gate -2 (the success flags folded into its two ANDs, as in dfthcbe; spec
    13.1).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.select import InnerSelect
        from dfthc.params import H2_STO3G_TEST as p

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        args = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
        wires, start = {}, 0
        for name, size in InnerSelect.required_wires(*args).items():
            wires[name], start = list(range(start, start + size)), start + size

        def program():
            qp.PhaseGradientStatePrep(wires["phase_grad"])
            InnerSelect(*args, **wires)

        compiled_resources(program, num_wires=start).toffoli  # 33
    """

    compilable_argnames = ("tail_angles", "sf_angles", "N", "R", "B", "C", "n_plus", "b_rot",
                           "fixup_bits")
    wire_argnames = _WIRES

    def __init__(self, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work):
        wires = (x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg,
                 psi_up, psi_dn, phase_grad, work)
        compilable = _validate("InnerSelect", (tail_angles, sf_angles, N, R, B, C, n_plus,
                                               b_rot, fixup_bits), wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                       fixup_bits) -> dict[str, int]:
        """Exact register sizes; ``work`` is the largest need among the children."""
        return _required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)


def _select_body(dagger, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work) -> None:
    """Steps 10 to 18, with ``qp.adjoint(MajoranaGate(...))`` if ``dagger``. Every register is
    unpacked into a list once; each child gets lists (spec 9.0)."""
    sizes = _child_sizes(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)
    (x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up, psi_dn,
     phase_grad, work) = (list(register) for register in (
         x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
         psi_dn, phase_grad, work))
    TailAngleLoad(tail_angles, n_plus, R * C, N + R * C, b_rot, x_o, angles, G0, G1,
                  work[:sizes["tail"]["work"]])
    SFAngleLoad(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work[:sizes["sf"]["work"]])
    SpinMux(G0, sigma_0, sigma_1, sg)
    SpinSwap(sg, psi_up, psi_dn)
    ladder_work = work[:sizes["ladder"]["work"]]
    GivensLadder(b_rot, psi_dn, angles, phase_grad, ladder_work)
    majorana = (succ_x_o, succ_b, G0, G1, b_neq_B, sigma_0, psi_dn[:1],
                work[:sizes["majorana"]["work"]])
    if dagger:
        qp.adjoint(MajoranaGate(*majorana))
    else:
        MajoranaGate(*majorana)
    GivensLadderDagger(b_rot, psi_dn, angles, phase_grad, ladder_work)
    SpinSwap(sg, psi_up, psi_dn)
    SpinMux(G0, sigma_0, sigma_1, sg)
    AngleErase(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, fixup_bits, angles, b_neq_B, G0,
               G1, x_o, r, b, work[:sizes["erase"]["work"]])


def _select_resources(dagger, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                      fixup_bits) -> dict:
    """The rule's children, declared by hand (a straight-line rule with composite keys; T2 pins
    it): each key has the child's compilable arguments and exactly its register sizes."""
    sizes = _child_sizes(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)
    wires = {child: {name: Wire[n] for name, n in regs.items()} for child, regs in sizes.items()}

    def majorana():
        op = MajoranaGate(**wires["majorana"])
        return qp.adjoint(op) if dagger else op

    return {
        resource_key(lambda: TailAngleLoad(tail_angles, n_plus, R * C, N + R * C, b_rot,
                                           **wires["tail"])): 1,
        resource_key(lambda: SFAngleLoad(sf_angles, R, B, b_rot, **wires["sf"])): 1,
        resource_key(lambda: SpinMux(**wires["mux"])): 2,
        resource_key(lambda: SpinSwap(**wires["swap"])): 2,
        resource_key(lambda: GivensLadder(b_rot, **wires["ladder"])): 1,
        resource_key(majorana): 1,
        resource_key(lambda: GivensLadderDagger(b_rot, **wires["ladder"])): 1,
        resource_key(lambda: AngleErase(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                                        fixup_bits, **wires["erase"])): 1,
    }


def _inner_select_resources(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits,
                            x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1,
                            b_neq_B, sg, psi_up, psi_dn, phase_grad, work):
    return _select_resources(False, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                             fixup_bits)


@qp.register_resources(_inner_select_resources)
def _inner_select(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o, succ_x_o,
                  sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up, psi_dn,
                  phase_grad, work):
    _select_body(False, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work)


register_rule(InnerSelect, _inner_select)


class InnerSelectDagger(Operator2):
    r"""The adjoint of :class:`InnerSelect`: its explicit adjoint class (spec 7.4 rule 6).

    **Action.** :math:`U^\dagger\,\mathrm{Maj}^\dagger\,U` on every branch, with :math:`U` and
    ``Maj`` as in :class:`InnerSelect` for the same arguments: ``InnerSelectDagger`` after
    ``InnerSelect`` is the identity on every input with the ancillas in :math:`|0\rangle`, for
    every measurement outcome of both erases. On the valid branches of the walk it applies the
    adjoint generator pieces: :math:`i\tilde\gamma_0\tilde\gamma_1` (Hermitian) on SF, and with
    ``sigma_0`` projected onto :math:`|+\rangle`, :math:`a_{\tilde u}^\dagger` on D1 and
    :math:`a_{\tilde u}` on Q1; the identity where :class:`InnerSelect` acts as the identity.

    **Registers.** Those of :class:`InnerSelect`, with the same input and output states.

    ============== ======================= ============================ =======================
    name           width                   input                        output
    ============== ======================= ============================ =======================
    ``x_o``        :math:`w_x`             any (read only)              unchanged
    ``succ_x_o``   1                       any (read only)              unchanged
    ``sigma_1``    1                       any (read only)              unchanged
    ``b``          :math:`w_b`             any (read only)              unchanged
    ``succ_b``     1                       any (read only)              unchanged
    ``sigma_0``    1                       any                          a control (diagonal)
    ``r``          :math:`w_r` (0 if R=1)  any (read only)              unchanged
    ``angles``     :math:`(N - 1) b_{rot}` ``|0>``                      ``|0>``
    ``G0``         1                       ``|0>``                      ``|0>``
    ``G1``         1                       ``|0>``                      ``|0>``
    ``b_neq_B``    1                       ``|0>``                      ``|0>``
    ``sg``         1                       ``|0>``                      ``|0>``
    ``psi_up``     ``N``                   any                          the action above
    ``psi_dn``     ``N``                   any                          the action above
    ``phase_grad`` :math:`b_{rot}`         phase-gradient state         unchanged (catalysed)
    ``work``       :meth:`required_wires`  ``|0>``                      ``|0>``
    ============== ======================= ============================ =======================

    **Compilable arguments.** Those of :class:`InnerSelect`, with the same domains.

    **Pre- and postconditions.** Those of :class:`InnerSelect`. In the walk it runs after the
    second ``InnerPrepare``, before ``InnerSigns`` (spec 9.21).

    **Decomposition** (spec 9.18). The sequence of :class:`InnerSelect` with
    ``qp.adjoint(MajoranaGate(...))``, the auto-derived leaf adjoint, in place of the Majorana
    gate: it loads first and erases last (reversing ``InnerSelect`` would run the measured erase
    backwards). Every other child has the same compilable arguments and register sizes as in
    :class:`InnerSelect`, so both selects share those targets.

    **Adjoint and control.** It is the explicit adjoint class of :class:`InnerSelect`; it is
    never adjointed or controlled itself (its rule measures; spec 7.4 rules 5 and 6).

    **Cost.** :func:`dfthc.costs.inner_select`, as for :class:`InnerSelect`: ``T`` and ``M``
    unchanged, ``A = T - (2N + 5)``. At FeMoco-54: ``T = 3537``, ``M = 798``, ``A = 3424``.

    **Paper and dfthcbe.** :math:`\mathrm{SEL}^\dagger` of the second round of steps 5 to 22
    (spec 3.5); dfthcbe: ``InnerSelect(..., is_adjoint=True)``, ``InnerSelect.adjoint()``, the
    same circuit with ``Maj().adjoint()`` (3,502 at FeMoco-54).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.data.factors import load_factors
        from dfthc.data.tables import WalkTables
        from dfthc.ops.select import InnerSelectDagger
        from dfthc.params import H2_STO3G_TEST as p

        t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
        args = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
        wires, start = {}, 0
        for name, size in InnerSelectDagger.required_wires(*args).items():
            wires[name], start = list(range(start, start + size)), start + size

        def program():
            qp.PhaseGradientStatePrep(wires["phase_grad"])
            InnerSelectDagger(*args, **wires)

        compiled_resources(program, num_wires=start).measurements  # (N - 1) b_rot + 3 = 7
    """

    compilable_argnames = ("tail_angles", "sf_angles", "N", "R", "B", "C", "n_plus", "b_rot",
                           "fixup_bits")
    wire_argnames = _WIRES

    def __init__(self, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work):
        wires = (x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg,
                 psi_up, psi_dn, phase_grad, work)
        compilable = _validate("InnerSelectDagger", (tail_angles, sf_angles, N, R, B, C, n_plus,
                                                     b_rot, fixup_bits), wires)
        super().__init__(*compilable, *wires)

    @staticmethod
    def required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                       fixup_bits) -> dict[str, int]:
        """:meth:`InnerSelect.required_wires`."""
        return _required_wires(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits)


def _inner_select_dagger_resources(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                                   fixup_bits, x_o, succ_x_o, sigma_1, b, succ_b, sigma_0, r,
                                   angles, G0, G1, b_neq_B, sg, psi_up, psi_dn, phase_grad,
                                   work):
    return _select_resources(True, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot,
                             fixup_bits)


@qp.register_resources(_inner_select_dagger_resources)
def _inner_select_dagger(tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                         succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg,
                         psi_up, psi_dn, phase_grad, work):
    _select_body(True, tail_angles, sf_angles, N, R, B, C, n_plus, b_rot, fixup_bits, x_o,
                 succ_x_o, sigma_1, b, succ_b, sigma_0, r, angles, G0, G1, b_neq_B, sg, psi_up,
                 psi_dn, phase_grad, work)


register_rule(InnerSelectDagger, _inner_select_dagger)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_inner_select.py -q`
Expected: `43 passed` (35 to 40 s with 2 workers, the 4 `slow` tests included; alone under `run_watched` the `MID_NAPHTHALENE` counts took 2.5 and 2.9 s at 358 MB and the `H2_631G_TEST` T3 7.9 and 8.2 s at 256 MB, log `.scratch/plan-check/stages/3S/3s-mid-compiles.log`).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `930 passed` (891 + 39; see "Expected test counts").

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/select.py tests/ops/test_inner_select.py
git commit -m "Add InnerSelect and InnerSelectDagger"
```

---

### Task 3.2: `RefT2`, `RefAB`, `ControlledRefT2`, `ControlledRefAB`

**Task id:** `3.2` · **Branch:** `phase3/3.2` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** G2

**Files:**
- Modify: `src/dfthc/ops/reflections.py` (replace it with the version below: `main`'s file with the import of `dfthc.ops._validate` extended by `check_int` and the section "RefT2 and RefAB" appended after `register_rule("C(ZeroReflection)", _c_zero_reflection)`; every other line of `main`'s file is unchanged, checked by `.scratch/plan-check/stages/3S/check_reflections_diff.py`)
- Create: `tests/ops/test_ref_t2_ab.py`

**Interfaces:**
- Consumes: `main`'s `dfthc.ops.reflections` (`ZeroReflection` with `required_wires(n_zero, n_one, controlled)`, and the `C(ZeroReflection)` rule's helpers `_pattern`, `_mcz`, `_mcx_key`; Task 1C.6); `dfthc.ops._validate` (`check_int`, `check_wires`); `dfthc.compile` (`register_rule`, `resource_key`, `compiled_resources`, `GATE_SET`, `FIXED_DECOMPS`); `dfthc.costs` (`ref_t2`, `ref_ab`); `dfthc.registers.WalkLayout` (`register_sizes`, `work_size`) and the presets of `dfthc.params`; `verbose_program`, `chosen_rules`.
- Produces, in `dfthc.ops.reflections`:
  - `RefT2(succ_x_o, b, m_b, sigma_0, work)`: 2Π_T2 − I with Π_T2 = \|0⟩⟨0\|_{b, m_b, σ0} ⊗ \|1⟩⟨1\|_{succ_x_o}, exact with the global sign; `required_wires(w_b, b_k2) -> {"succ_x_o": 1, "b": w_b, "m_b": b_k2, "sigma_0": 1, "work": w_b + b_k2 - 1}` (`w_b, b_k2 >= 1`); the rules `_ref_t2` (emits `ZeroReflection(b ‖ m_b ‖ sigma_0, succ_x_o, work)`) and `_adjoint_ref_t2` (pinned as `"Adjoint(RefT2)"`: the operator itself);
  - `RefAB(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work)`: 2\|0⟩⟨0\| − I on the six registers; `required_wires(w_x, b_k1, w_b, b_k2)` (`work`: n_w − 3 with n_w = w_x + b_k1 + w_b + b_k2 + 2); the rules `_ref_ab` and `_adjoint_ref_ab`;
  - `ControlledRefT2(control, succ_x_o, b, m_b, sigma_0, work)` and `ControlledRefAB(control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work)`: \|0⟩⟨0\| ⊗ I + \|1⟩⟨1\| ⊗ (2Π − I) exactly; `required_wires` as the plain classes' plus `control: 1`, with `work` n_w − 2; the rules `_controlled_ref_t2` and `_controlled_ref_ab` (the multi-controlled Z over the control, value 1, and the pattern, then `Z` on the control; no `C(...)` key). Never adjointed or controlled.

**Acceptance tests** (45 tests in `tests/ops/test_ref_t2_ab.py`), at the walk's register widths of each preset (`WalkLayout.register_sizes`):
- T3 compiled on lightning (12 tests): `RefT2` and `RefAB` at both test presets, plain and through their pinned adjoints, map a phased state of the pattern registers (H on each wire, S on even positions, CZ on consecutive pairs: every amplitude nonzero, phases not uniform) to exactly (2Π − I) of it, global sign included, with `work` back in \|0⟩ (8 tests); `ControlledRefT2` and `ControlledRefAB` at both test presets, one program each with the control's state a runtime argument: the control in \|0⟩ leaves the state unchanged, in \|1⟩ applies 2Π − I, and in \|+⟩ gives (\|0⟩ψ + \|1⟩(2Π − I)ψ)/√2 (4 tests).
- T4 compiled (16 tests): each class at both test presets, `MID_NAPHTHALENE` and FeMoco-54's register widths (table-free, so per commit; about 1.2 to 1.7 s each): T against `costs.ref_t2` and `costs.ref_ab` (n_w − 2, controlled n_w − 1), M = 0, one plain `Toffoli`, A = T − 1; FeMoco-54 on the closed forms (20 / 21, 44 / 45) and the `work` sizes there (19, 43, 20, 44; `ControlledRefAB`'s equals `WalkLayout.work_size`) (1 test).
- T5 and T6 (1 test): the verbose listing names the six pinned rules and `_zero_reflection` but not `_c_zero_reflection`, and no dfthc operator survives at `level="device"`.
- T2 (8 tests): the four base rules at both presets' widths and the two pinned adjoints, each the only rule under its key; `required_wires` by hand (1 test); six constructor violations, each a `ValueError` (1 parametrized test, 6 cases).

**Definition of done:** `45 passed`; `tests/ops/test_zero_reflection.py` still passes; the per-commit tier passes; `ruff check .` clean; the spec 9.0 docstring sections in order; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_ref_t2_ab.py`:

```python
"""RefT2, RefAB, ControlledRefT2 and ControlledRefAB (spec 9.19, 9.20): the exact action with the
global sign, the control in |0>, |1> and |+>, counts, rule selection, validation."""

import math

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.reflections import (
    ControlledRefAB,
    ControlledRefT2,
    RefAB,
    RefT2,
    _adjoint_ref_ab,
    _adjoint_ref_t2,
    _controlled_ref_ab,
    _controlled_ref_t2,
    _ref_ab,
    _ref_t2,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.registers import WalkLayout
from dfthc.testing.resources import chosen_rules, verbose_program

PRESETS = {"sto3g": H2_STO3G_TEST, "631g": H2_631G_TEST, "mid": MID_NAPHTHALENE,
           "femoco-width": FEMOCO54}
# class, its pattern registers in wire order (control and work excluded), the registers with
# pattern value 1, the closed form, whether it is controlled, its rule
CLASSES = {
    "RefT2": (RefT2, ("succ_x_o", "b", "m_b", "sigma_0"), {"succ_x_o"}, costs.ref_t2, False,
              _ref_t2),
    "RefAB": (RefAB, ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0"), set(), costs.ref_ab,
              False, _ref_ab),
    "ControlledRefT2": (ControlledRefT2, ("succ_x_o", "b", "m_b", "sigma_0"), {"succ_x_o"},
                        costs.ref_t2, True, _controlled_ref_t2),
    "ControlledRefAB": (ControlledRefAB, ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0"),
                        set(), costs.ref_ab, True, _controlled_ref_ab),
}


def sizes(kind, p):
    """The class's required_wires for the walk's register widths at p (WalkLayout's)."""
    cls, pattern, *_ = CLASSES[kind]
    widths = WalkLayout.register_sizes(p)
    args = ([widths["b"], widths["m_b"]] if "x_o" not in pattern
            else [widths["x_o"], widths["m_x_o"], widths["b"], widths["m_b"]])
    return cls.required_wires(*args)


def layout(register_sizes):
    wires, start = {}, 0
    for name, size in register_sizes.items():
        wires[name], start = list(range(start, start + size)), start + size
    return wires, start


def phased_prepare(wires):
    """H on each wire, S on even positions, CZ on consecutive pairs (non-uniform phases)."""
    for wire in wires:
        qp.H(wire)
    for position, wire in enumerate(wires):
        if position % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


def phased_amplitudes(n):
    """The state phased_prepare makes on n wires, wire 0 the most significant bit."""
    bits = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    return (1j ** bits[:, ::2].sum(axis=1) * (-1.0) ** (bits[:, :-1] * bits[:, 1:]).sum(axis=1)
            / math.sqrt(2**n))


def reflection_diagonal(pattern_sizes, ones):
    """Diagonal of 2 Pi - I over the pattern registers (wire order, big-endian): +1 on the one
    basis state with the registers in ``ones`` all ones and the others all zeros, -1 elsewhere."""
    target = "".join(("1" if name in ones else "0") * n for name, n in pattern_sizes.items())
    return np.where(np.arange(2 ** len(target)) == int(target, 2), 1.0, -1.0)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["plain", "adjoint"])
@pytest.mark.parametrize("preset", ["sto3g", "631g"])
@pytest.mark.parametrize("kind", ["RefT2", "RefAB"])
def test_exact_action_with_the_global_sign(kind, preset, adjoint):
    """T3 on the compiled path (lightning) at the walk's register widths of both test presets: on
    a phased state of every pattern register (every amplitude nonzero, phases not uniform) the
    output is exactly (2 Pi - I) applied, global sign included, with work back in |0>; the pinned
    adjoint (the reflection itself) gives the same."""
    cls, pattern, ones, *_ = CLASSES[kind]
    register_sizes = sizes(kind, PRESETS[preset])
    wires, num_wires = layout(register_sizes)
    pattern_wires = [wire for name in pattern for wire in wires[name]]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        phased_prepare(pattern_wires)
        op = cls(**wires)
        if adjoint:
            qp.adjoint(op)
        return qp.state()

    psi = phased_amplitudes(len(pattern_wires))
    diagonal = reflection_diagonal({name: register_sizes[name] for name in pattern}, ones)
    expected = np.kron(diagonal * psi, np.eye(2 ** register_sizes["work"])[0])
    assert np.allclose(np.asarray(circuit()), expected, rtol=0, atol=1e-10)


@pytest.mark.compiled
@pytest.mark.parametrize("preset", ["sto3g", "631g"])
@pytest.mark.parametrize("kind", ["ControlledRefT2", "ControlledRefAB"])
def test_controlled_action_for_three_control_states(kind, preset):
    """T3 on the compiled path for the controlled classes, one program with the control's state
    as a runtime argument: with the control in |0> the phased pattern state is unchanged, in |1>
    it gets 2 Pi - I, and in |+> the output is (|0> psi + |1> (2 Pi - I) psi) / sqrt 2, which
    detects a wrong global phase on the controlled branch (spec 3.3); work ends in |0>."""
    cls, pattern, ones, *_ = CLASSES[kind]
    register_sizes = sizes(kind, PRESETS[preset])
    wires, num_wires = layout(register_sizes)
    pattern_wires = [wire for name in pattern for wire in wires[name]]
    control = wires["control"][0]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit(mode):
        qp.cond(mode == 1, qp.X)(control)
        qp.cond(mode == 2, qp.H)(control)
        phased_prepare(pattern_wires)
        cls(**wires)
        return qp.state()

    psi = phased_amplitudes(len(pattern_wires))
    reflected = reflection_diagonal({name: register_sizes[name] for name in pattern}, ones) * psi
    clean = np.eye(2 ** register_sizes["work"])[0]
    zero, one = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    expected = {0: np.kron(zero, psi), 1: np.kron(one, reflected),
                2: (np.kron(zero, psi) + np.kron(one, reflected)) / math.sqrt(2)}
    for mode, state in expected.items():
        assert np.allclose(np.asarray(circuit(mode)), np.kron(state, clean), rtol=0,
                           atol=1e-10), mode


COUNT_CASES = [pytest.param(kind, preset, id=f"{kind}-{preset}")
               for kind in CLASSES for preset in PRESETS]


@pytest.mark.compiled
@pytest.mark.parametrize(("kind", "preset"), COUNT_CASES)
def test_counts(kind, preset):
    """T4 compiled at both test presets, MID_NAPHTHALENE and FeMoco-54's register widths (the
    reflections are table-free, so per commit): T = n_w - 2 (controlled n_w - 1) against
    dfthc.costs.ref_t2 and ref_ab, M = 0, one plain Toffoli, so A = T - 1."""
    cls, _, _, cost, controlled, _ = CLASSES[kind]
    p = PRESETS[preset]
    wires, num_wires = layout(sizes(kind, p))
    resources = compiled_resources(lambda: cls(**wires), num_wires)
    assert (resources.toffoli, resources.measurements) == (cost(p, controlled).toffoli, 0)
    assert resources.counts["Toffoli"] == 1
    assert resources.measured_and_uncomputes == resources.toffoli - 1


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 (spec 9.24): RefT2 20 and 21 controlled, RefAB 44 and 45 controlled; the
    controlled RefAB's 44 work wires are the walk's largest need, WalkLayout.work_size."""
    p = FEMOCO54
    assert [costs.ref_t2(p, False), costs.ref_t2(p, True)] == [costs.Cost(20), costs.Cost(21)]
    assert [costs.ref_ab(p, False), costs.ref_ab(p, True)] == [costs.Cost(44), costs.Cost(45)]
    assert [sizes(kind, p)["work"] for kind in CLASSES] == [19, 43, 20, 44]
    assert sizes("ControlledRefAB", p)["work"] == WalkLayout.work_size(p) == 44


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names the six pinned rules and ZeroReflection's, and not
    C(ZeroReflection)'s: the controlled classes request no C(...) key (spec 7.4 rule 6). T5 at
    level="device" for all four classes and both adjoints (L-20)."""
    p = H2_STO3G_TEST

    def program():
        for kind, (cls, *_) in CLASSES.items():
            wires, _ = layout(sizes(kind, p))
            cls(**wires)
            if kind in ("RefT2", "RefAB"):
                qp.adjoint(cls(**wires))

    num_wires = max(layout(sizes(kind, p))[1] for kind in CLASSES)
    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_ref_t2", "_adjoint_ref_t2", "_controlled_ref_t2", "_ref_ab", "_adjoint_ref_ab",
            "_controlled_ref_ab", "_zero_reflection"} <= rules
    assert "_c_zero_reflection" not in rules
    compiled_resources(program, num_wires, level="device")


@pytest.mark.parametrize("preset", ["sto3g", "631g"])
@pytest.mark.parametrize("kind", CLASSES)
def test_rules_are_valid(kind, preset):
    """T2 (tape mode) at two sizes for the four base rules and the two pinned adjoints; each rule
    is the only one registered under its key (spec 7.4 rules 7 and 8)."""
    cls, _, _, _, controlled, rule = CLASSES[kind]
    wires, _ = layout(sizes(kind, PRESETS[preset]))
    op = cls(**wires)
    assert FIXED_DECOMPS[kind] is rule
    assert list(qp.list_decomps(cls)) == [rule]
    qp.testing.assert_valid_decomp_rule(op, rule, check_matrix=False)
    if not controlled:
        adjoint_rule = {"RefT2": _adjoint_ref_t2, "RefAB": _adjoint_ref_ab}[kind]
        assert list(qp.list_decomps(f"Adjoint({kind})")) == [adjoint_rule]
        qp.testing.assert_valid_decomp_rule(qp.adjoint(op), adjoint_rule, check_matrix=False)


def test_required_wires():
    """By hand at H2_STO3G_TEST (w_x = w_b = b_k1 = b_k2 = 2): n_w = 6 and 10."""
    assert RefT2.required_wires(2, 2) == {"succ_x_o": 1, "b": 2, "m_b": 2, "sigma_0": 1,
                                          "work": 3}
    assert ControlledRefT2.required_wires(2, 2) == {"control": 1, "succ_x_o": 1, "b": 2,
                                                    "m_b": 2, "sigma_0": 1, "work": 4}
    assert RefAB.required_wires(2, 2, 2, 2) == {"x_o": 2, "m_x_o": 2, "sigma_1": 1, "b": 2,
                                                "m_b": 2, "sigma_0": 1, "work": 7}
    assert ControlledRefAB.required_wires(2, 2, 2, 2)["work"] == 8


@pytest.mark.parametrize(("build", "match"), [
    (lambda: RefT2([0], [], [1], [2], []), r"RefT2: len\(b\) must lie in \[1, None\]"),
    (lambda: RefT2([0, 9], [1], [2], [3], [4]), "'succ_x_o' has 2 wires; expected 1"),
    (lambda: RefT2([0], [1], [2], [3], []), "'work' has 0 wires; expected 1"),
    (lambda: ControlledRefT2([], [0], [1], [2], [3], [4, 5]), "'control' has 0 wires"),
    (lambda: RefAB([0], [1], [2], [3], [], [4], [5, 6, 7]), r"len\(m_b\) must lie in \[1, None\]"),
    (lambda: ControlledRefAB([9], [0], [1], [2], [3], [4], [5], [6, 7]),
     "'work' has 2 wires; expected 4"),
])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_ref_t2_ab.py -q`
Expected: FAIL at collection, `1 error`: `ImportError: cannot import name 'ControlledRefAB' from 'dfthc.ops.reflections'`.

- [ ] **Step 3: Add the reflections to `reflections.py`**

Replace `src/dfthc/ops/reflections.py` with:

```python
"""Reflections about a basis pattern, exact including the global sign (spec 9.6, 9.19, 9.20)."""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_int, check_wires


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


# --------------------------------------------------------------------------- RefT2 and RefAB


def _widths(op: str, **registers) -> tuple[int, ...]:
    """The lengths of the named pattern registers, each at least 1 (``ValueError`` otherwise)."""
    return tuple(check_int(f"{op}: len({name})", len(register), low=1)
                 for name, register in registers.items())


def _reflection_key(n_zero: int, n_one: int):
    """The ``ZeroReflection`` that ``RefT2`` and ``RefAB`` emit, with exactly its work."""
    work = ZeroReflection.required_wires(n_zero, n_one)["work"]
    return resource_key(lambda: ZeroReflection(Wire[n_zero], Wire[n_one], Wire[work]))


def _controlled_reflection(control, zero_wires, one_wires, work) -> None:
    r""":math:`|0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes (2\Pi - I)` for at
    least two pattern wires: the multi-controlled Z over the control and the pattern, then ``Z``
    on the control, as the ``C(ZeroReflection)`` rule builds it (spec 9.6)."""
    controls, values, target, last = _pattern(zero_wires, one_wires)
    _mcz([control, *controls], [1, *values], target, last, work)
    qp.Z(control)


def _controlled_reflection_resources(n_zero: int, n_one: int) -> dict:
    controls, values, _, last = _pattern(range(n_zero), range(n_zero, n_zero + n_one))
    return {_mcx_key(len(controls) + 1, [1, *values]): 1, qp.H: 2, qp.X: 2 * (last == 0),
            qp.Z: 1}


class RefT2(Operator2):
    r""":math:`\mathrm{REF}_{T_2} = 2\Pi_{T_2} - I` (step 23), exact including the global sign.

    **Action.** :math:`\Pi_{T_2} = |0\rangle\langle 0|_{b, m_b, \sigma_0} \otimes
    |1\rangle\langle 1|_{\mathrm{succ}_{x_o}}` (spec 3.3): the phase +1 on the one basis state
    with ``b = 0``, ``m_b = 0``, ``sigma_0 = 0`` and ``succ_x_o = 1``, and -1 on every other, so
    the operator is :math:`-I` wherever the uniform preparation of ``x_o`` failed
    (``succ_x_o = 0``; spec 8.5). Every register is unchanged up to that phase.

    **Registers.** :math:`n_w = w_b + b_{k2} + 2` pattern wires.

    ============ ============================ ======== =========
    name         width                        input    output
    ============ ============================ ======== =========
    ``succ_x_o`` 1                            any      phased
    ``b``        :math:`w_b` (at least 1)     any      phased
    ``m_b``      :math:`b_{k2}` (at least 1)  any      phased
    ``sigma_0``  1                            any      phased
    ``work``     :math:`n_w - 3`              ``|0>``  ``|0>``
    ============ ============================ ======== =========

    **Compilable arguments.** None; :math:`w_b` and :math:`b_{k2}` follow from the wires.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.19). ``ZeroReflection(zero_wires=b ‖ m_b ‖ sigma_0,
    one_wires=succ_x_o, work)``: a ``MultiControlledX`` on ``succ_x_o`` with the other
    :math:`n_w - 1` pattern wires as zero-valued controls, ``H``-conjugated, then
    ``GlobalPhase(pi)``; :math:`n_w = w_b + b_{k2} + 2`.

    **Adjoint and control.** ``Adjoint(RefT2)`` is pinned to the operator itself (it is
    self-inverse; spec 7.5). The controlled form is the class :class:`ControlledRefT2`; nothing
    applies ``qp.ctrl`` to this operator (spec 7.4 rule 6).

    **Cost.** :func:`dfthc.costs.ref_t2` (uncontrolled): :math:`T = n_w - 2 = w_b + b_{k2}`, one
    ``Toffoli`` and :math:`w_b + b_{k2} - 1` ``TemporaryAND`` with measured uncomputes
    (``A = T - 1``); ``M = 0``. At FeMoco-54: 20, as dfthcbe's; the paper's Table III counts 21.

    **Paper and dfthcbe.** Step 23, :math:`\mathrm{REF}_{T_2}` of spec 3.3 (PRX App. B; dfthcbe
    cites the paper's (B11) for its multi-controlled Z). dfthcbe: ``DFTHCBlockEncoding.ref_t2``,
    a Qualtran ``ReflectionUsingPrepare.reflection_around_zero`` on ``b``, ``m_b``, ``sigma_0``
    controlled by ``succ_x_o``, with ``GlobalPhase(exponent=1)`` (``dfthcbe/block_encoding.py``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import RefT2

        program = lambda: RefT2(succ_x_o=[0], b=[1, 2], m_b=[3, 4], sigma_0=[5], work=[6, 7, 8])
        compiled_resources(program, num_wires=9).toffoli  # w_b + b_k2 = 4
    """

    wire_argnames = ("succ_x_o", "b", "m_b", "sigma_0", "work")

    def __init__(self, succ_x_o, b, m_b, sigma_0, work):
        w_b, b_k2 = _widths("RefT2", b=b, m_b=m_b)
        check_wires("RefT2", {"succ_x_o": succ_x_o, "b": b, "m_b": m_b, "sigma_0": sigma_0,
                              "work": work}, self.required_wires(w_b, b_k2))
        super().__init__(succ_x_o, b, m_b, sigma_0, work)

    @staticmethod
    def required_wires(w_b: int, b_k2: int) -> dict[str, int]:
        """Exact register sizes; ``work`` is the reflection's :math:`n_w - 3`."""
        return {"succ_x_o": 1, "b": w_b, "m_b": b_k2, "sigma_0": 1,
                "work": ZeroReflection.required_wires(w_b + b_k2 + 1, 1)["work"]}


def _ref_t2_resources(succ_x_o, b, m_b, sigma_0, work):
    return {_reflection_key(len(b) + len(m_b) + 1, 1): 1}


@qp.register_resources(_ref_t2_resources)
def _ref_t2(succ_x_o, b, m_b, sigma_0, work):
    ZeroReflection([*b, *m_b, *sigma_0], [*succ_x_o], work)


def _adjoint_ref_t2_resources(base):
    sizes = RefT2.required_wires(len(base.b), len(base.m_b))
    return {resource_key(lambda: RefT2(**{name: Wire[n] for name, n in sizes.items()})): 1}


@qp.register_resources(_adjoint_ref_t2_resources)
def _adjoint_ref_t2(base):
    RefT2(base.succ_x_o, base.b, base.m_b, base.sigma_0, base.work)


register_rule(RefT2, _ref_t2)
register_rule("Adjoint(RefT2)", _adjoint_ref_t2)


class ControlledRefT2(Operator2):
    r"""The controlled :math:`\mathrm{REF}_{T_2}` of the controlled block encoding: the explicit
    controlled class of :class:`RefT2` (spec 7.4 rule 6).

    **Action.** Exactly :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle
    1|_{control} \otimes (2\Pi_{T_2} - I)`, with :math:`\Pi_{T_2}` as in :class:`RefT2`: the
    identity where ``control = 0`` and :class:`RefT2`, global sign included, where
    ``control = 1``. The sign matters here: control turns :class:`RefT2`'s global -1 into a
    relative phase (spec 3.3).

    **Registers.** :math:`n_w = w_b + b_{k2} + 2` pattern wires.

    ============ ============================ ======== =========
    name         width                        input    output
    ============ ============================ ======== =========
    ``control``  1                            any      phased
    ``succ_x_o`` 1                            any      phased
    ``b``        :math:`w_b` (at least 1)     any      phased
    ``m_b``      :math:`b_{k2}` (at least 1)  any      phased
    ``sigma_0``  1                            any      phased
    ``work``     :math:`n_w - 2`              ``|0>``  ``|0>``
    ============ ============================ ======== =========

    **Compilable arguments.** None; the sizes follow from the wires.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.19). The ``MultiControlledX`` on ``succ_x_o`` with ``control``
    (value 1) and ``b ‖ m_b ‖ sigma_0`` (value 0) as controls, ``H``-conjugated, which is
    :math:`I - 2|1\rangle\langle 1|_{control} \otimes \Pi_{T_2}`; then ``Z`` on ``control``. It is
    the construction of the ``C(ZeroReflection)`` rule (spec 9.6), emitted directly with that
    rule's helper, so no ``C(...)`` key is requested.

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0).

    **Cost.** :func:`dfthc.costs.ref_t2` (controlled): :math:`T = n_w - 1 = w_b + b_{k2} + 1`,
    one ``Toffoli`` and :math:`w_b + b_{k2}` ``TemporaryAND`` (``A = T - 1``); ``M = 0``. At
    FeMoco-54: 21, as dfthcbe's.

    **Paper and dfthcbe.** The controlled walk's controlled :math:`\mathrm{REF}_{T_2}` (spec
    3.3); dfthcbe: ``DFTHCBlockEncoding.ref_t2`` of ``is_controlled=True``, the reflection's
    ``.controlled()`` with a ``ZGate`` on ``ctrl`` (``dfthcbe/block_encoding.py``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import ControlledRefT2

        program = lambda: ControlledRefT2(control=[0], succ_x_o=[1], b=[2, 3], m_b=[4, 5],
                                          sigma_0=[6], work=[7, 8, 9, 10])
        compiled_resources(program, num_wires=11).toffoli  # w_b + b_k2 + 1 = 5
    """

    wire_argnames = ("control", "succ_x_o", "b", "m_b", "sigma_0", "work")

    def __init__(self, control, succ_x_o, b, m_b, sigma_0, work):
        w_b, b_k2 = _widths("ControlledRefT2", b=b, m_b=m_b)
        check_wires("ControlledRefT2", {"control": control, "succ_x_o": succ_x_o, "b": b,
                                        "m_b": m_b, "sigma_0": sigma_0, "work": work},
                    self.required_wires(w_b, b_k2))
        super().__init__(control, succ_x_o, b, m_b, sigma_0, work)

    @staticmethod
    def required_wires(w_b: int, b_k2: int) -> dict[str, int]:
        """Exact register sizes; ``work`` is the controlled reflection's :math:`n_w - 2`."""
        return {"control": 1, "succ_x_o": 1, "b": w_b, "m_b": b_k2, "sigma_0": 1,
                "work": ZeroReflection.required_wires(w_b + b_k2 + 1, 1, controlled=True)["work"]}


def _controlled_ref_t2_resources(control, succ_x_o, b, m_b, sigma_0, work):
    return _controlled_reflection_resources(len(b) + len(m_b) + 1, 1)


@qp.register_resources(_controlled_ref_t2_resources)
def _controlled_ref_t2(control, succ_x_o, b, m_b, sigma_0, work):
    _controlled_reflection(control[0], [*b, *m_b, *sigma_0], [*succ_x_o], work)


register_rule(ControlledRefT2, _controlled_ref_t2)


class RefAB(Operator2):
    r""":math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}} = 2|0\rangle\langle 0| - I` on the outer
    and inner index registers (step 28), exact including the global sign.

    **Action.** The phase +1 on the one basis state with ``x_o``, ``m_x_o``, ``sigma_1``, ``b``,
    ``m_b`` and ``sigma_0`` all zero, and -1 on every other (spec 3.3:
    :math:`\mathcal{A} = (x_o, m_{x_o}, \sigma_1)`, :math:`\mathcal{B} = (b, m_b, \sigma_0)`).
    Every register is unchanged up to that phase.

    **Registers.** :math:`n_w = w_x + b_{k1} + w_b + b_{k2} + 2` pattern wires.

    =========== ============================ ======== =========
    name        width                        input    output
    =========== ============================ ======== =========
    ``x_o``     :math:`w_x` (at least 1)     any      phased
    ``m_x_o``   :math:`b_{k1}` (at least 1)  any      phased
    ``sigma_1`` 1                            any      phased
    ``b``       :math:`w_b` (at least 1)     any      phased
    ``m_b``     :math:`b_{k2}` (at least 1)  any      phased
    ``sigma_0`` 1                            any      phased
    ``work``    :math:`n_w - 3`              ``|0>``  ``|0>``
    =========== ============================ ======== =========

    **Compilable arguments.** None; the sizes follow from the wires.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.20). ``ZeroReflection(zero_wires=x_o ‖ m_x_o ‖ sigma_1 ‖ b ‖ m_b ‖
    sigma_0, one_wires=[], work)``: a ``MultiControlledX`` on ``sigma_0`` with the other
    :math:`n_w - 1` wires as zero-valued controls, conjugated by ``X`` and ``H``, then
    ``GlobalPhase(pi)``; :math:`n_w = w_x + b_{k1} + w_b + b_{k2} + 2`.

    **Adjoint and control.** ``Adjoint(RefAB)`` is pinned to the operator itself (spec 7.5). The
    controlled form is the class :class:`ControlledRefAB`; nothing applies ``qp.ctrl`` to this
    operator (spec 7.4 rule 6).

    **Cost.** :func:`dfthc.costs.ref_ab` (uncontrolled):
    :math:`T = n_w - 2 = w_x + b_{k1} + w_b + b_{k2}`, one ``Toffoli`` (``A = T - 1``);
    ``M = 0``. At FeMoco-54: 44; dfthcbe's costs 45 and the paper's Table III 46 (spec 13.1,
    reason ``constants``).

    **Paper and dfthcbe.** Step 28, :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}`, which makes
    :math:`W` a qubitization walk (spec 3.3); dfthcbe: ``DFTHCWalk.ref_ab``, Qualtran's
    ``ReflectionUsingPrepare.reflection_around_zero`` on the six registers
    (``dfthcbe/block_encoding.py``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import RefAB

        program = lambda: RefAB(x_o=[0, 1], m_x_o=[2, 3], sigma_1=[4], b=[5, 6], m_b=[7, 8],
                                sigma_0=[9], work=list(range(10, 17)))
        compiled_resources(program, num_wires=17).toffoli  # w_x + b_k1 + w_b + b_k2 = 8
    """

    wire_argnames = ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0", "work")

    def __init__(self, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
        widths = _widths("RefAB", x_o=x_o, m_x_o=m_x_o, b=b, m_b=m_b)
        check_wires("RefAB", {"x_o": x_o, "m_x_o": m_x_o, "sigma_1": sigma_1, "b": b,
                              "m_b": m_b, "sigma_0": sigma_0, "work": work},
                    self.required_wires(*widths))
        super().__init__(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work)

    @staticmethod
    def required_wires(w_x: int, b_k1: int, w_b: int, b_k2: int) -> dict[str, int]:
        """Exact register sizes; ``work`` is the reflection's :math:`n_w - 3`."""
        return {"x_o": w_x, "m_x_o": b_k1, "sigma_1": 1, "b": w_b, "m_b": b_k2, "sigma_0": 1,
                "work": ZeroReflection.required_wires(w_x + b_k1 + w_b + b_k2 + 2, 0)["work"]}


def _ref_ab_resources(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
    return {_reflection_key(len(x_o) + len(m_x_o) + len(b) + len(m_b) + 2, 0): 1}


@qp.register_resources(_ref_ab_resources)
def _ref_ab(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
    ZeroReflection([*x_o, *m_x_o, *sigma_1, *b, *m_b, *sigma_0], [], work)


def _adjoint_ref_ab_resources(base):
    sizes = RefAB.required_wires(len(base.x_o), len(base.m_x_o), len(base.b), len(base.m_b))
    return {resource_key(lambda: RefAB(**{name: Wire[n] for name, n in sizes.items()})): 1}


@qp.register_resources(_adjoint_ref_ab_resources)
def _adjoint_ref_ab(base):
    RefAB(base.x_o, base.m_x_o, base.sigma_1, base.b, base.m_b, base.sigma_0, base.work)


register_rule(RefAB, _ref_ab)
register_rule("Adjoint(RefAB)", _adjoint_ref_ab)


class ControlledRefAB(Operator2):
    r"""The controlled :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` of the controlled walk:
    the explicit controlled class of :class:`RefAB` (spec 7.4 rule 6).

    **Action.** Exactly :math:`|0\rangle\langle 0|_{control} \otimes I + |1\rangle\langle
    1|_{control} \otimes (2|0\rangle\langle 0|_{\mathcal{A}\cup\mathcal{B}} - I)`, global sign
    included (spec 3.3).

    **Registers.** :math:`n_w = w_x + b_{k1} + w_b + b_{k2} + 2` pattern wires.

    =========== ============================ ======== =========
    name        width                        input    output
    =========== ============================ ======== =========
    ``control`` 1                            any      phased
    ``x_o``     :math:`w_x` (at least 1)     any      phased
    ``m_x_o``   :math:`b_{k1}` (at least 1)  any      phased
    ``sigma_1`` 1                            any      phased
    ``b``       :math:`w_b` (at least 1)     any      phased
    ``m_b``     :math:`b_{k2}` (at least 1)  any      phased
    ``sigma_0`` 1                            any      phased
    ``work``    :math:`n_w - 2`              ``|0>``  ``|0>``
    =========== ============================ ======== =========

    **Compilable arguments.** None; the sizes follow from the wires.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.20). The ``MultiControlledX`` on ``sigma_0`` with ``control``
    (value 1) and the other pattern wires (value 0) as controls, conjugated by ``X`` and ``H``,
    then ``Z`` on ``control``: the ``C(ZeroReflection)`` construction (spec 9.6), emitted with
    that rule's helper, so no ``C(...)`` key is requested.

    **Adjoint and control.** Never adjointed or controlled itself (spec 9.0).

    **Cost.** :func:`dfthc.costs.ref_ab` (controlled):
    :math:`T = n_w - 1 = w_x + b_{k1} + w_b + b_{k2} + 1` (``A = T - 1``); ``M = 0``. At
    FeMoco-54: 45; dfthcbe's controlled reflection also costs 45. Its work, 44 wires at
    FeMoco-54, is the walk's largest need, the size of ``WalkLayout.work`` (spec 10.3).

    **Paper and dfthcbe.** The controlled walk's controlled
    :math:`\mathrm{REF}_{\mathcal{A}\cup\mathcal{B}}` (spec 3.3); dfthcbe: ``DFTHCWalk.ref_ab``
    of ``is_controlled=True`` (``control_val=1``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.reflections import ControlledRefAB

        program = lambda: ControlledRefAB(control=[0], x_o=[1, 2], m_x_o=[3, 4], sigma_1=[5],
                                          b=[6, 7], m_b=[8, 9], sigma_0=[10],
                                          work=list(range(11, 19)))
        compiled_resources(program, num_wires=19).toffoli  # w_x + b_k1 + w_b + b_k2 + 1 = 9
    """

    wire_argnames = ("control", "x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0", "work")

    def __init__(self, control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
        widths = _widths("ControlledRefAB", x_o=x_o, m_x_o=m_x_o, b=b, m_b=m_b)
        check_wires("ControlledRefAB", {"control": control, "x_o": x_o, "m_x_o": m_x_o,
                                        "sigma_1": sigma_1, "b": b, "m_b": m_b,
                                        "sigma_0": sigma_0, "work": work},
                    self.required_wires(*widths))
        super().__init__(control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work)

    @staticmethod
    def required_wires(w_x: int, b_k1: int, w_b: int, b_k2: int) -> dict[str, int]:
        """Exact register sizes; ``work`` is the controlled reflection's :math:`n_w - 2`."""
        return {"control": 1, "x_o": w_x, "m_x_o": b_k1, "sigma_1": 1, "b": w_b, "m_b": b_k2,
                "sigma_0": 1,
                "work": ZeroReflection.required_wires(w_x + b_k1 + w_b + b_k2 + 2, 0,
                                                      controlled=True)["work"]}


def _controlled_ref_ab_resources(control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
    return _controlled_reflection_resources(len(x_o) + len(m_x_o) + len(b) + len(m_b) + 2, 0)


@qp.register_resources(_controlled_ref_ab_resources)
def _controlled_ref_ab(control, x_o, m_x_o, sigma_1, b, m_b, sigma_0, work):
    _controlled_reflection(control[0], [*x_o, *m_x_o, *sigma_1, *b, *m_b, *sigma_0], [], work)


register_rule(ControlledRefAB, _controlled_ref_ab)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_ref_t2_ab.py -q`
Expected: `45 passed` (30 to 34 s with 2 workers; the slowest test, the verbose and device-level compile, about 6 s).

Run: `scripts/test.sh tests/ops/test_zero_reflection.py -q`
Expected: `33 passed` (Task 1C.6's tests, its one `slow` test included).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `936 passed` (891 + 45).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/reflections.py tests/ops/test_ref_t2_ab.py
git commit -m "Add RefT2, RefAB, ControlledRefT2 and ControlledRefAB"
```

---

## Interfaces for Tasks 3.3 to 3.5

What the block-encoding, walk and end-to-end tasks consume from this file, exactly as built.

**Constructors and sizes.** `stages/3S/interfaces_probe.py` below is the interface as executed in the dry run: it splits the `WalkLayout` groups with `WalkLayout.split_group`, as the block encoding's rule will, calls each class on exactly its registers and a prefix of the work pool, builds the resource keys an assembly rule declares (`keys`: each class with `Wire[n]` of exactly its register sizes), compiles the `H2_STO3G_TEST` program once against `dfthc.costs`, and checks the work sizes at every preset (FeMoco-54 from `random_factors`, classical only). The controlled reflections take `layout.qpe_control` here only because the walk layout has no other one-wire register; the controlled walk passes its own `control`.

```python
"""Interfaces for Tasks 3.3 to 3.5: how an assembly rule calls the select and the reflections.

Executed by the Phase 3 select plan's dry run, from the replica root with PYTHONPATH=src. It splits
the WalkLayout groups with WalkLayout.split_group (as the block encoding's rule will), calls
InnerSelect, InnerSelectDagger and the four reflections on exactly their registers and prefixes of
the work pool, builds their resource keys, compiles the H2_STO3G_TEST program once on null.qubit
against dfthc.costs, and checks the work sizes at every preset, FeMoco-54 included (classical:
nothing FeMoco-shaped is compiled).
"""

from pennylane.typing import Wire

from dfthc import costs
from dfthc.compile import compiled_resources, resource_key
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.reflections import ControlledRefAB, ControlledRefT2, RefAB, RefT2
from dfthc.ops.select import InnerSelect, InnerSelectDagger
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.registers import WalkLayout


def calls(p, t, layout):
    """Every call of the select and the reflections, as (class, compilable args, registers)."""
    regs = {name: wires for group in ("outer", "inner", "select", "system")
            for name, wires in WalkLayout.split_group(p, group, getattr(layout, group)).items()}
    work, control = list(layout.work), list(layout.qpe_control)
    select_args = (t.tail_angles, t.sf_angles, p.N, p.R, p.B, p.C, p.n_plus, p.b_rot, p.w_b)
    select_sizes = InnerSelect.required_wires(*select_args)
    select = {name: regs[name] for name in InnerSelect.wire_argnames[:-2]}
    select |= {"phase_grad": layout.phase_grad, "work": work[:select_sizes["work"]]}
    t2 = {name: regs[name] for name in ("succ_x_o", "b", "m_b", "sigma_0")}
    ab = {name: regs[name] for name in ("x_o", "m_x_o", "sigma_1", "b", "m_b", "sigma_0")}
    t2_work = RefT2.required_wires(p.w_b, p.b_k2)["work"]
    ab_work = RefAB.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)["work"]
    t2_controlled = ControlledRefT2.required_wires(p.w_b, p.b_k2)["work"]
    ab_controlled = ControlledRefAB.required_wires(p.w_x, p.b_k1, p.w_b, p.b_k2)["work"]
    return [
        (InnerSelect, select_args, select),
        (RefT2, (), t2 | {"work": work[:t2_work]}),
        (InnerSelectDagger, select_args, select),
        (RefAB, (), ab | {"work": work[:ab_work]}),
        (ControlledRefT2, (), {"control": control} | t2 | {"work": work[:t2_controlled]}),
        (ControlledRefAB, (), {"control": control} | ab | {"work": work[:ab_controlled]}),
    ]


def keys(entries):
    """The resource keys an assembly rule declares for these calls (exact register sizes)."""
    return [resource_key(lambda cls=cls, args=args, regs=regs: cls(
        *args, **{name: Wire[len(wires)] for name, wires in regs.items()}))
        for cls, args, regs in entries]


def main():
    p = H2_STO3G_TEST
    t = WalkTables.from_factors(load_factors("h2_sto3g_R2_B2_C1"), p)
    layout = WalkLayout.from_params(p, qpe_bits=1)
    entries = calls(p, t, layout)
    print("keys:", [key.name for key in keys(entries)])

    def program():
        for cls, args, regs in entries:
            cls(*args, **regs)

    resources = compiled_resources(program, layout.num_wires)
    expected = (2 * costs.inner_select(p) + costs.ref_t2(p, False) + costs.ref_ab(p, False)
                + costs.ref_t2(p, True) + costs.ref_ab(p, True))
    print("H2_STO3G_TEST: T, M, A =", resources.toffoli, resources.measurements,
          resources.measured_and_uncomputes, "; closed forms T, M =", expected.toffoli,
          expected.measurements)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, expected.measurements)
    systems = {"H2_STO3G_TEST": (H2_STO3G_TEST, load_factors("h2_sto3g_R2_B2_C1")),
               "H2_631G_TEST": (H2_631G_TEST, load_factors("h2_631g_R2_B2_C2")),
               "MID_NAPHTHALENE": (MID_NAPHTHALENE, load_factors("naphthalene")),
               "FEMOCO54": (FEMOCO54, random_factors(54, 10, 27, 27, 27, seed=0))}
    for name, (p, factors) in systems.items():
        t = WalkTables.from_factors(factors, p)
        layout = WalkLayout.from_params(p, qpe_bits=1)
        sizes = {cls.__name__: len(regs["work"]) for cls, _, regs in calls(p, t, layout)}
        print(f"{name}: work {sizes}; WalkLayout.work_size {WalkLayout.work_size(p)}")
        assert sizes["ControlledRefAB"] == WalkLayout.work_size(p) == max(sizes.values())


if __name__ == "__main__":
    main()
```

Its output (`stages/3S/interfaces-probe.out`; 12.8 s, 318 MB under the watchdog, `interfaces-probe.log`):

````text
keys: ['InnerSelect', 'RefT2', 'InnerSelectDagger', 'RefAB', 'ControlledRefT2', 'ControlledRefAB']
H2_STO3G_TEST: T, M, A = 92 14 70 ; closed forms T, M = 92 14
H2_STO3G_TEST: work {'InnerSelect': 3, 'RefT2': 3, 'InnerSelectDagger': 3, 'RefAB': 7, 'ControlledRefT2': 4, 'ControlledRefAB': 8}; WalkLayout.work_size 8
H2_631G_TEST: work {'InnerSelect': 3, 'RefT2': 3, 'InnerSelectDagger': 3, 'RefAB': 8, 'ControlledRefT2': 4, 'ControlledRefAB': 9}; WalkLayout.work_size 9
MID_NAPHTHALENE: work {'InnerSelect': 14, 'RefT2': 17, 'InnerSelectDagger': 14, 'RefAB': 38, 'ControlledRefT2': 18, 'ControlledRefAB': 39}; WalkLayout.work_size 39
FEMOCO54: work {'InnerSelect': 14, 'RefT2': 19, 'InnerSelectDagger': 14, 'RefAB': 43, 'ControlledRefT2': 20, 'ControlledRefAB': 44}; WalkLayout.work_size 44
````

So `ControlledRefAB`'s work is `WalkLayout.work_size(p)`, the pool's largest need, at every preset; the select needs the ladders' `b_rot - 1` from `MID_NAPHTHALENE` on. Pass `fixup_bits = p.w_b`: `costs.inner_select` and `costs.block_encoding` assume it. The tables are `WalkTables`' tuples; both selects get the same compilable arguments and register sizes, so they share the `AngleErase`, load, ladder and spin targets (one target each in the capture probe).

**Registers each class reads, writes and leaves.**

| Class | Reads only (controls) | Writes | Ancillas, clean in and out | Leaves dirty |
| --- | --- | --- | --- | --- |
| `InnerSelect`, `InnerSelectDagger` | `x_o`, `succ_x_o`, `sigma_1`, `b`, `succ_b`, `r`; `sigma_0` diagonally (a phase) | `psi_up`, `psi_dn` | `angles`, `G0`, `G1`, `b_neq_B`, `sg`, `work`; `phase_grad` catalysed (exactly `b_rot` wires: the walk's whole `phase_grad` group) | nothing |
| `RefT2`, `ControlledRefT2` | (phases only) `succ_x_o`, `b`, `m_b`, `sigma_0`, and `control` | none (diagonal) | `work` | nothing |
| `RefAB`, `ControlledRefAB` | (phases only) `x_o`, `m_x_o`, `sigma_1`, `b`, `m_b`, `sigma_0`, and `control` | none (diagonal) | `work` | nothing |

None of the classes takes `less_than_b`, `sgn`, `d_sgn`, `alt_b`, `keep_b`, `junk_b`, `succ_work_b`, `cmp_work_b` or any outer register except `x_o`, `succ_x_o`, `sigma_1`, `m_x_o`, so the prepares' postconditions hold across them.

**Order in the block encoding** (spec 9.21): `OuterPrepare`; `InnerPrepare`, `InnerSigns`, `InnerSelect`, `InnerUnprepare`; `RefT2` (controlled block encoding: `ControlledRefT2`); `InnerPrepare`, `InnerSelectDagger`, `InnerSigns`, `InnerUnprepare`; `OuterUnprepare`. The walk adds `RefAB` (controlled walk: `ControlledRefAB`). `InnerSigns` is diagonal and the select touches none of its registers, so its position next to the select is exact either way; keep spec 9.21's. The selects are never controlled: the controlled block encoding controls only `RefT2`. Never apply `qp.adjoint` to `InnerSelect` (use `InnerSelectDagger`) or `qp.ctrl` to any of these classes (use the controlled classes).

**Costs** (`dfthc.costs`): `inner_select(p)` for each select (M = (N − 1) b_rot + 3, A = T − (2N + 5)); `ref_t2(p, controlled)` and `ref_ab(p, controlled)` (A = T − 1, M = 0). `block_encoding`, `walk` and `measured_and_uncomputes` already sum them; `toffoli_only(p)` counts the selects' 2(2N + 5) and the reflections' 2 plain Toffolis per walk.

**What Task 3.4's inventory test will see from these classes**: the leaf adjoint `Adjoint(MajoranaGate)` (from `InnerSelectDagger`, 86 equations; its verbose listing names the base rule `_majorana_gate`), together with Task 2E.4's `Adjoint(GivensRotation)` and `Adjoint(SemiAdder)`; no `C(...)` target from the controlled reflections (Task 3.2's T6 shows `_c_zero_reflection` is not chosen); `Adjoint(RefT2)` and `Adjoint(RefAB)` are registered but no walk program requests them.

**Captured equations** (for Task 3.4's comparison with Task 0.9 and Task 4.5's projection): each select target 1.43 per bound wire + 53 (1,293 at `MID_NAPHTHALENE`, projected 6,408 at FeMoco-54 against the stand-in's 27,537).

**Test helpers Task 3.5 may copy** (they are local to `tests/ops/test_inner_select.py`, not a shared module): `key_values`, `with_gradient`, `gradient_state`, `assert_same_state` (one-word keys, at most 64 wires; the walk at `H2_STO3G_TEST` has more, so E-1 keeps Task 1B.3's harness), and `flavour_actions`, the per-branch generator pieces from the encoded model.
