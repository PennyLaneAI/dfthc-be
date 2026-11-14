# DFTHC Phase 2, Workstream E: Select Blocks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the blocks of the inner select (steps 10 to 18 of spec 3.5): the angle loads `TailAngleLoad` and `SFAngleLoad`, their measured erase `AngleErase` with its outcome certificates, the spin plumbing `SpinMux` and `SpinSwap`, the Givens rotation and ladders `GivensRotation`, `GivensLadder` and `GivensLadderDagger`, and the `MajoranaGate`, each an `Operator2` with one pinned rule per key, exact Toffoli and measurement counts against `dfthc.costs`, and tests T2 to T6.

**Architecture:** One module per block family under `src/dfthc/ops/` (spec 5.2): `angle_data.py` (the two loads and the erase), `spin.py`, `givens.py`, `majorana.py`. The loads are straight-line rules around one `QROAMRead` without a swap network, whose per-leaf words the rule builds from the compilable angle table (spec 9.3); `AngleErase` loops over its tables and emits through `QROAMEmitter`, with the counting twin as its resource function, and calls `emit_phase_fixup` twice (spec 9.17). The other rules are straight-line and declare their resources by hand, pinned by T2. The angle register is big-endian (spec 6.7): register `k` drives rung `k`, register 0 most significant.

**Tech Stack:** Python 3.12, PennyLane `Operator2` and Catalyst (`qp.qjit(capture=True)`, `qp.decompose`, `qp.analyze`), NumPy, JAX; the sparse simulator `dfthc.testing.sparse_sim` and `dfthc.testing.tape`; pytest with pytest-xdist; ruff.

This file holds Tasks 2E.1 to 2E.5 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`, section 8). Workstream D (Tasks 2D.1 and 2D.2) is in `2026-11-11-dfthc-phase2-prepare.md`. Task 3.1 (`InnerSelect`) composes all five tasks of this file.

**Code provenance.** Every code block below was executed on 2026-11-12 in a scratch replica of the repository (`.scratch/plan-check/repo-phase2e`, not committed), seeded from local `main` at `f7acf97` (Task 1H.1 merged) with `git archive`. The replica is not a git repository (the sandbox forbids creating one), so the dry run ran pytest the way `scripts/test.sh` does: the same environment, 2 workers, `taskpolicy -c utility`, `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` exported, `TMPDIR` under `.scratch/`. The steps below use `scripts/test.sh` as usual. Each task was replayed RED then GREEN in a fresh copy of `main` (`.scratch/plan-check/stages/2E/run_stages_2e.sh`, log `stages/2E/2e-stages.log`): 2E.1 and then 2E.2 in one copy, and 2E.3, 2E.4 and 2E.5 each in a copy with no other Workstream E file, as they run in parallel lanes; then all five together. Every "Expected" line below is that replay's output, and the final files are byte-identical to the replica's. The 2E.1 version of `angle_data.py` is written from the final one by `stages/2E/make_stage_2e1.py`, which also checks that the 2E.2 block contains every line of it. Further evidence, each with its log under `stages/2E/`: the per-commit tier on the replica with all five tasks (`2e-per-commit-tier.log`); the `MID_NAPHTHALENE` compiles, each alone under `run_watched` (`2e-mid-compiles.log`); a mutation check of the tests (`2e-mutants.log`); the ten slowest per-commit Workstream E tests, from the replay's last run (`2e-durations.log`); every docstring example run against its stated value (`doc-examples.log`); an `InnerSelect`-shaped composition of the five tasks (`select_composition_probe.py`, log `composition.log`); and dfthcbe's per-block FeMoco-54 Toffolis from Qualtran's symbolic call graph (`.scratch/plan-check/probes/2e/dfthcbe_select_blocks.py`, 24 s, nothing simulated). One test of `main` fails in the replica and only there: `tests/test_infrastructure.py::test_runner_fails_closed_without_lowered_priority` creates a directory named `venv` under `TMPDIR`, which the sandbox refuses inside the workspace; it passes in the worktree lanes, and the counts below count it as passing.

**Load.** At most 2 pytest workers ran at a time, and nothing FeMoco-shaped was compiled. Every compile measured alone under the watchdog stayed below 340 MB: the largest were the `MID_NAPHTHALENE` select composition (3.6 to 4.3 s, 336 MB) and the table-free FeMoco-width ladder (1.6 s, 276 MB). The full per-commit tier ran once per review round on the replica (its log holds revision 2's run); the stage replay, which runs only Workstream E's tests, ran three times in round 1 and once in revision 2, on the final files (the timings below span these runs; a busy machine doubled them, and revision 2's ran beside the Workstream D dry run). The mutation check ran all 18 round-1 mutants once; revision 2 reran the nine `angle_data.py` mutants on the final code and added two, and the seven Givens, Majorana and spin mutants stand from round 1, as neither their modules nor their tests changed.

**Revision 2** (2026-11-12, after the GPT-5.6 review `.superpowers/sdd/phase2e-plan-review-1.md`; all five findings and its accept and modify judgements adopted):
1. *R = 1 under capture* (Important 1). `test_compiled_loads_on_lightning` also compiles both loads at `random-R1` (the SF address `b` alone) with `qp.qjit(capture=True)`, runs them on lightning and compares the full state, `b = B` and the failed `b` included; it stays one pytest item. `random-R1`'s load tables now use seed 0, whose SF entries are distinct and nonzero (asserted; seed 3 gave all zeros, which would hide a wrong `b`). The compiled T4 gains the R = 1 SF load (one test, renamed `test_toffoli_count_compiled`). Change 2, E-M5 and E-S3 name this evidence.
2. *`TailAngleLoad` constructor error* (Important 2). A scalar `tail_angles` raised `TypeError` from `len`; `_check` now canonicalizes, checks for a table, and only then takes `len`, so it raises `ValueError` (spec 9.0). The load tests add a scalar and a flat `tail_angles` and a scalar `sf_angles`, and the erase tests a scalar `tail_angles` (four tests); `SFAngleLoad` and `AngleErase` already raised `ValueError` for these.
3. *Certificate premises* (Minor 1). The certificates' docstring, Task 2E.2's acceptance text and the ledger rows 1C.4 and 1C.5 say that the suite includes partly covered SF leaves and that ob = 0 exercises the fallback; `test_certificate_cases_cover_the_branches` asserts the zero tail row and SF entry of `R3-zero-blocks` and a partly covered leaf in every case with ob > 0.
4. *Edit wording* (Minor 2). E-M3 quotes the measured 3.6 to 4.3 s; E-M6 adds `2**sf_onehot_bits <= (N - 1) * b_rot + 1` (E-S4 already states it; E-S5 states no domain); E-M7 says "three additional shapes, including `ob = 0`, `ob = 1` and `R = 1`".
5. *Slowest tests* (Minor 3). The unsupported "slowest single test" sentence is replaced by the replay's `--durations=10` record (`stages/2E/2e-durations.log`).

Counts move from 34 to 38 tests for 2E.1 (36 per commit) and from 27 to 28 for 2E.2 (23 per commit): the per-commit tier with all of Workstream E is 788 instead of 783 (run once in revision 2: 787 passed plus the replica-only failure). Two mutants are new, an R = 1 SF load that keeps the two-register address and a tail constructor that takes `len` first (20 in all, all caught). The doc edits were regenerated against `main` at `f7acf97`, the stages replayed (every "Expected" line below is from that replay) and the plan reassembled.

**Revision 3** (2026-11-13, after Task 2E.1's GPT-5.6 review `.superpowers/sdd/task-2E.1-review-1.md`, 0 Critical, 0 Important, 2 Minor; applied on `phase2/2E.1` and here): the module docstring of `angle_data.py` states the composed load (every `G0 = 0` branch, failed `x_o` included, holds the SF entry where `(r, b)` is in range), and `TailAngleLoad`'s docstring says that `AngleErase` erases `angles`, `b_neq_B`, `G0` and `G1`. Docstrings only, in the 2E.1 and 2E.2 blocks of `angle_data.py`; no count changes.

**Revision 4** (2026-11-13, after Task 2E.2's reviews: GPT-5.6 `.superpowers/sdd/task-2E.2-review-1.md`, Approved, no findings; Opus `.superpowers/sdd/task-2E.2-review-opus.md`, Approved, 0 Critical, 0 Important, 4 Minor; applied on `phase2/2E.2` and here). Four mutants (the SF mask without `b_neq_B`, the flags measured `G1` first, a stray `b_neq_B` bit in the tail masks, the SF address reversed) survived the per-commit tier and died only in the nightly certificates, so `test_angle_erase.py` gains a per-commit forced-schedule round trip at three shapes (M-1, + 3), compares the masks the rule builds with the independently built ones (M-2), and checks the SF rung count in the constructor (M-3, + 1); the `AngleErase` docstring names dfthcbe's one-hot over `B + 1` positions among the differences (M-4). Task 2E.2 counts 32 tests (27 per commit); the per-commit tier with all of Workstream E is 792. The timings and the dry-run figures below are revision 2's.

**Revision 5** (2026-11-13, after Task 2E.4's reviews: GPT-5.6 `.superpowers/sdd/task-2E.4-review-1.md`, Approved, no findings; Opus `.superpowers/sdd/task-2E.4-review-opus.md`, Approved, 0 Critical, 0 Important, 3 Minor; applied on `phase2/2E.4` and here). Deleting `GivensLadderDagger`'s wire check passed every test, so `test_givens.py` gains a seventh constructor violation, an oversized `angles` register on the dagger (M-1, + 1); the N = 3 ladder angles become `[1, 3]`, whose `u` has a nonzero odd component, so the γ₀ identity also sees a flipped rung sign (M-3, no count change); the `GivensRotation` and `GivensLadder` docstrings state the paper's Table III cost, 30 per rung and 1,590 per ladder at FeMoco-54 (M-2). Task 2E.4 counts 36 tests, all per commit; the per-commit tier with all of Workstream E is 793.

**Revision 6** (2026-11-13, after Task 2E.5's reviews: GPT-5.6 `.superpowers/sdd/task-2E.5-review-1.md`, Approved, no findings; Opus `.superpowers/sdd/task-2E.5-review-opus.md`, Needs fixes, tests only, 0 Critical, 1 Important, 2 Minor; applied on `phase2/2E.5` and here). On lightning the device pass runs `TemporaryAND` and its measured uncompute both as a `Toffoli`, and T and A are symmetric under exchanging them, so a rule (or a hand-written adjoint) with the two AND forms exchanged passed every test. `test_majorana.py` gains a tape-path run on the sparse simulator, gate and adjoint on all 128 inputs, whose handlers assert each AND's precondition (I-1, + 2), and asserts the rule's pin in `FIXED_DECOMPS` (M-1); the `MajoranaGate` docstring cites arXiv:2502.15882v1 App. B.6 and Eqs. (29) and (30), and says that D1 and Q1 stay on where `succ_b = 0`, unlike Fig. 4 (M-2). Task 2E.5 counts 16 tests, all per commit; the per-commit tier with all of Workstream E is 795.

**Revision 7** (2026-11-14, after Task 2E.3's GPT-5.6 review `.superpowers/sdd/task-2E.3-review-1.md`, Needs fixes, 0 Critical, 0 Important, 1 Minor; applied on `phase2/2E.3` and here). The `SpinMux` docstring cites App. B.3 ("Selecting between the spins") for computing $\sigma_{G_0}$, not App. B.2, the qubitization step (M-1). Docstring only; Task 2E.3 still counts 24 tests.

---

## Changes from the master plan and the spec

The changes that touch a document are applied by the exact edits listed after this list (ids E-M for the master plan, E-S for the spec).

1. **The angle loads take `b_rot`** (E-M4, E-S1, E-S2). The master plan's interfaces `TailAngleLoad(tail_angles, n_plus, RC, X_o, x_o, ...)` and `SFAngleLoad(sf_angles, R, B, r, b, ...)` have no argument that sizes the angle registers, yet `required_wires(**compilable_args)` must give `len(angles) = (N - 1) b_rot` and the tables must be checked against `x < 2**(b_rot - 1)` (spec 6.3). `b_rot` is a structure-determining argument, so it is compilable (spec 7.4 rules 1 and 3): `TailAngleLoad(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work)` and `SFAngleLoad(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work)`. `AngleErase` already has `b_rot`.
2. **`SFAngleLoad` with R = 1** (E-M4, E-M5, E-S3). Spec 8.3 allows R = 1, where `w_r = 0` and `QROAMRead` rejects a two-register address with an empty first register. The SF address is then `b` alone; the cost formula is unchanged (U(0, 1, 0, 1) = 0). The `random-R1` shapes test it in tape mode, certificates included, and on the capture path: `test_compiled_loads_on_lightning` compiles both loads at `random-R1` with `qp.qjit(capture=True)`, runs them on lightning and compares the full state, `b = B` and the failed `b` included, and the compiled T4 counts the R = 1 SF load.
3. **`AngleErase`'s `sf_onehot_bits` domain** (E-M6, E-S4, E-S5). Spec 9.17 names the walk's value, `w_b`. The operator accepts `0 <= ob <= w_b` with `2**ob <= (N - 1) b_rot + 1`, because the one-hot lives on the reset `angles ‖ b_neq_B` wires (spec 10.2). ob = 0 is spec 9.4's fallback ("Every caller can switch to it with one argument"); for ob < `w_b` the SF term is `costs.phase_fixup` over `AngleErase.sf_fixup_high_ranges`, which T4 asserts at three shapes. `costs.angle_erase` stays the walk's ob = `w_b` closed form, unchanged.
4. **`SpinMux` and `SpinSwap` define `compute_matrix`** (E-M8, E-S6). The master plan asks for T2 with `check_matrix=True`; spec 12.1 says dfthc operators define no matrix, in which case PennyLane skips the comparison silently (`assert_valid_decomp_rule` checks `op.has_matrix`). Both operators are permutations, so they define their permutation matrices; the tests assert `has_matrix`, and the mutation check shows that a wrong matrix fails T2. The compiled path is unchanged (both compile, at user and device level, with the pinned rules).
5. **Size-only `required_wires`.** `GivensRotation.required_wires(b_rot)`, `GivensLadder.required_wires(b_rot, N)`, `GivensLadderDagger.required_wires(b_rot, N)`, `SpinSwap.required_wires(N)`, `SpinMux.required_wires()` and `MajoranaGate.required_wires()` take the sizes that their wires determine, as `ZeroReflection.required_wires(n_zero, n_one)` does on `main`: their sizes follow from their wires, not (or not only) from compilable arguments (spec 9.0, "If it has no Compilable row ... its sizes follow from its wire lengths"). No document edit.
6. **More compiled sizes than the master plan lists** (E-M5, E-M7, E-M9). The loads' and the erase's T4 also compile at `MID_NAPHTHALENE` (`slow`, nightly tier, as spec 12.6 places the composites' mid-size T4); the table-free `GivensLadder`, `GivensLadderDagger` and `SpinSwap` compile per commit at `MID_NAPHTHALENE` and at FeMoco-54's widths (master plan section 7 allows table-free FeMoco-width compiles in any tier; 1.6 s and 276 MB for the 53-rung ladder). FeMoco-54 values of the table operators are asserted on the closed forms only.
7. **The ladder tests at N = 4 use b_rot = 3** (E-M9). At b_rot = 4 the N = 4 ladder needs 23 lightning wires and took 50 s per test; at b_rot = 3 it needs 18 wires and 1.6 s. The rung test covers b_rot = 2, 3 and 4 for every angle, so the N = 4 test's job is the order and the register mapping: its angles (1, 3, 2) are asserted to give different unitaries for the opposite rung order and for the reversed register order.
8. **Master plan section 1 and section 3** (E-M1 to E-M3): this file's row and provenance, and a new section 3 item 21 with this dry run's composition and compile-cost findings.
9. **Fixed-size operators have one size.** The master plan's "T2 to T6 at two or more sizes" cannot hold for `SpinMux` and `MajoranaGate`, whose registers are all one wire wide (and two work wires); their T2 runs at two wire labelings instead. No document edit.

Exact edits (master plan and spec, against local `main` at `f7acf97`):

Each edit replaces one exact text of the file on local `main` at `f7acf97`, where it occurs exactly once (checked by `.scratch/plan-check/stages/2E/make_doc_edits_2e.py`, which also applies them).

**E-M1** (2026-11-11-dfthc-master-plan.md, section 1, the file table). Old:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase1-primitives.md` | Tasks 1C.1 to 1C.6 |
````

New:

````text
| `docs/superpowers/plans/2026-11-11-dfthc-phase1-primitives.md` | Tasks 1C.1 to 1C.6 |
| `docs/superpowers/plans/2026-11-11-dfthc-phase2-blocks.md` | Tasks 2E.1 to 2E.5 |
````

**E-M2** (2026-11-11-dfthc-master-plan.md, section 1, code provenance). Old:

````text
with the Phase 1 A and B code it imports; its preamble lists its changes to this plan.
````

New:

````text
with the Phase 1 A and B code it imports; its preamble lists its changes to this plan. The Phase 2 E file (`phase2-blocks`) was executed the same way on 2026-11-12, against `main` at `f7acf97`; its preamble lists its changes to this plan.
````

**E-M3** (2026-11-11-dfthc-master-plan.md, section 3, a new item 21 after item 20 (the end of item 20 is shown)). Old:

````text
are siblings of `OuterUnprepare` and `InnerUnprepare`.
````

New:

````text
are siblings of `OuterUnprepare` and `InnerUnprepare`.
21. **The Workstream E blocks compose as Task 3.1 needs them** (Phase 2 E dry run, 2026-11-12). In `InnerSelect`'s order (spec 9.18), each block passed only its registers and a prefix of the walk's work pool, they compile to `costs.inner_select` exactly at `H2_STO3G_TEST`, `H2_631G_TEST` and `MID_NAPHTHALENE` (T = 33, 65, 625; M = 7, 15, 138), forward and with the adjoint Majorana gate. The `MID_NAPHTHALENE` select compiles in 3.6 to 4.3 s at 336 MB, each E block alone in 1 to 2 s below 270 MB, and the table-free FeMoco-width `GivensLadder` and `GivensLadderDagger` (878 wires) in 1.6 s at 276 MB. `SpinMux` and `SpinSwap` define `compute_matrix`, so their T2 passes `check_matrix=True` with an effective matrix comparison; their compiled path is unchanged.
````

**E-M4** (2026-11-11-dfthc-master-plan.md, Task 2E.1, interfaces). Old:

````text
`TailAngleLoad(tail_angles, n_plus, RC, X_o, x_o, angles, G0, G1, work)` and `SFAngleLoad(sf_angles, R, B, r, b, G0, angles, b_neq_B, work)`, where the SF load is enabled on G0 = 0.
````

New:

````text
`TailAngleLoad(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work)` and `SFAngleLoad(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work)`, where the SF load is enabled on G0 = 0 and its address is `b` alone when R = 1; `b_rot` sizes the angle registers (spec 7.4 rules 1 and 3).
````

**E-M5** (2026-11-11-dfthc-master-plan.md, Task 2E.1, acceptance tests). Old:

````text
  - T2, T5.
````

New:

````text
  - T3 compiled on lightning at `H2_STO3G_TEST` and at an R = 1 shape (the SF address `b` alone, with `b = B` and a failed `b` left untouched): the full state equals the exclusive-branch table, so the one-register address lowers under capture.
  - T4 compiled at both test presets, for the SF load at the R = 1 shape, and at `MID_NAPHTHALENE` (`slow`): T, M and A exact; the reads' CNOT counts equal their declared upper bounds.
  - T2, T5, T6.
````

**E-M6** (2026-11-11-dfthc-master-plan.md, Task 2E.2, interfaces). Old:

````text
`AngleErase(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits, angles, b_neq_B, G0, G1, x_o, r, b, work)`, following the four-step order of spec 9.17.
````

New:

````text
`AngleErase(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits, angles, b_neq_B, G0, G1, x_o, r, b, work)`, following the four-step order of spec 9.17, with `0 <= sf_onehot_bits <= w_b` and `2**sf_onehot_bits <= (N - 1) * b_rot + 1` (the one-hot lives on the reset `angles ‖ b_neq_B` wires; the walk passes `w_b`; 0 is the fallback of spec 9.4).
````

**E-M7** (2026-11-11-dfthc-master-plan.md, Task 2E.2, acceptance tests). Old:

````text
  - T4: T = 102, M = 798 at FeMoco; two small sizes.
````

New:

````text
  - T4: T = 102, M = 798 at FeMoco; both test presets compiled, `MID_NAPHTHALENE` (`slow`), and three additional shapes, including `ob = 0`, `ob = 1` and `R = 1`.
````

**E-M8** (2026-11-11-dfthc-master-plan.md, Task 2E.3, acceptance tests). Old:

````text
  - T2 with `check_matrix=True`.
````

New:

````text
  - T2 with `check_matrix=True`: both classes define `compute_matrix` (their permutation matrices), so the check compares the decomposition with it.
````

**E-M9** (2026-11-11-dfthc-master-plan.md, Task 2E.4, acceptance tests). Old:

````text
  - The ladder equals `ladder_matrix` for N ≤ 4.
  - T4: 2(b_rot − 1) per rung; 1,484 per ladder at FeMoco.
````

New:

````text
  - The ladder equals `ladder_matrix` for N ≤ 4 (b_rot = 4 for N = 2, 3 and b_rot = 3 for N = 4, at most 18 lightning wires); for N ≥ 3 the opposite rung order and the reversed angle-register order give different unitaries at the test's angles.
  - T4: 2(b_rot − 1) per rung; 1,484 per ladder at FeMoco, on the closed form and compiled at FeMoco's widths (table-free).
````

**E-S1** (2026-11-11-dfthc-pennylane-design.md, 9.10, Compilable). Old:

````text
| Compilable | `tail_angles` ($N \times (N-1)$, D1 rows first), $N_+$, $RC$, $X_o$ |
````

New:

````text
| Compilable | `tail_angles` ($N \times (N-1)$, D1 rows first), $N_+$, $RC$, $X_o$, $b_{\mathrm{rot}}$ (it sizes `angles`) |
````

**E-S2** (2026-11-11-dfthc-pennylane-design.md, 9.11, Compilable). Old:

````text
| Compilable | `sf_angles` ($R \times B \times (N-1)$), $R$, $B$ |
````

New:

````text
| Compilable | `sf_angles` ($R \times B \times (N-1)$), $R$, $B$, $b_{\mathrm{rot}}$ (it sizes `angles`) |
````

**E-S3** (2026-11-11-dfthc-pennylane-design.md, 9.11, Rule). Old:

````text
no swap, enable on $G_0 = 0$, and entries (angles, $[b \ne B] = 1$) |
````

New:

````text
no swap, enable on $G_0 = 0$, and entries (angles, $[b \ne B] = 1$); for $R = 1$ ($w_r = 0$) the address is $b$ alone, and T3 compiles and executes this case under capture, with $b = B$ and a failed $b$ |
````

**E-S4** (2026-11-11-dfthc-pennylane-design.md, 9.17, Compilable). Old:

````text
| Compilable | `tail_angles`, `sf_angles`, $N_+$, sizes, and the SF fixup's one-hot bits ($w_b$) |
````

New:

````text
| Compilable | `tail_angles`, `sf_angles`, $N_+$, sizes, and the SF fixup's one-hot bits ob: $w_b$ in the walk; any $0 \le \mathrm{ob} \le w_b$ with $2^{\mathrm{ob}} \le (N-1)b_{\mathrm{rot}} + 1$ (the one-hot lives on the reset angle and `b_neq_B` wires), ob $= 0$ being the fallback of §9.4 |
````

**E-S5** (2026-11-11-dfthc-pennylane-design.md, 9.17, Cost). Old:

````text
$M = (N-1)b_{\mathrm{rot}} + 3$ (§9.24). At FeMoco-54: $T = (11 + 30) + 61 = 102$, $M = 798$. |
````

New:

````text
$M = (N-1)b_{\mathrm{rot}} + 3$ (§9.24), for ob $= w_b$; for a smaller ob the bracket is the `PhaseFixup` cost of §9.24 over the SF fix-up's high ranges. At FeMoco-54: $T = (11 + 30) + 61 = 102$, $M = 798$. |
````

**E-S6** (2026-11-11-dfthc-pennylane-design.md, 12.1, T2). Old:

````text
dfthc operators define none, so the tests pass `check_matrix=False`, and their T3 tests check the action exactly.
````

New:

````text
of the dfthc operators only `SpinMux` and `SpinSwap` define one (`compute_matrix`, a permutation matrix), and their T2 passes `check_matrix=True`; the others pass `check_matrix=False`, and their T3 tests check the action exactly.
````

The Workstream D plan edits the same master plan and spec: its own Task 2D entries, the "Plan" column of 2D.1 and 2D.2 in section 4, and probably section 1's file table and provenance paragraph, where E-M1 and E-M2 add one row and one sentence. The orchestrator applies both sets in sequence; no edit above touches a 2D entry or spec 9.7 to 9.9.

## Shared edits

None. No merged module outside `angle_data.py`, `spin.py`, `givens.py` and `majorana.py` changes: the erase's emitter uses only gate names that `as_resources` already maps (`Hadamard`, `MidMeasure`, `CZ`, `PauliZ`, `PauliX`, `CNOT`, `TemporaryAND`, `Adjoint(TemporaryAND)`; ledger row 1C.1), `emit_phase_fixup` already rejects an out-of-range `onehot_bits`, and every cost comes from `dfthc.costs` on `main`. Two small helpers are local to `angle_data.py` and may duplicate Workstream D's: `_words(value, width)` (an int as 32-bit words) and `_join(*registers)`; a later shared helper in `qroam.py` could replace both. The optional `np.signedinteger` check in `QROAMRead`'s abstract-array path (ledger row 1C.3) is not adopted here, because it would edit `qroam.py`; the loads declare their keys with `WORD_DTYPE` (`np.int64`).

## How this plan meets the binding ledger rows

| Ledger row or decision | What this plan does |
| --- | --- |
| Decision "K-1 at G0" (per-leaf read loads, explicit classes) | The loads build `QROAMRead`'s dynamic per-leaf `leaf_words` in their rules from their compilable tables with `QROAMRead.leaf_words_from_table`; the erase keeps the compilable tables and derives its masks in its rule. `GivensLadderDagger` is the explicit adjoint class; nothing calls `qp.adjoint` or `qp.ctrl` on a dfthc composite. The only leaf adjoints are `qp.adjoint(GivensRotation(...))` inside `GivensLadderDagger` and, in Task 3.1, `qp.adjoint(MajoranaGate(...))` (spec 7.4 rule 6). |
| Decision "Measuring-rule warnings" | `test_erase_has_no_adjoint_or_controlled_path` wraps `AngleErase` in `qp.adjoint` and `qp.ctrl` on purpose: the compile fails with `CompileError` "Decomposition rule not found", after Catalyst's "Skipped the _angle_erase ..." warning, which `classify_rule_warning` returns as `"expected"`. |
| Decision "K-10 consequences" (conditionals decided by compilable data in both modes; masks never compilable data) | `AngleErase` builds its masks inside the rule from the compilable tables; no mask is ever zero (spec 9.17), so both modes emit one hinted conditional per covered address, R B + N in all, and the counting twin declares exactly that (T2 in tape mode at three sizes; T6 counts them compiled). |
| Decision "Table words are `int64`" and ledger row 1C.3 (`WORD_DTYPE` keys) | The loads' resource keys use `AbstractArray(QROAMRead.leaf_words_shape(...), WORD_DTYPE)`; `leaf_words_from_table` returns `int64` words. |
| Ledger row 1C.3: Clifford counts of reads are upper bounds | The loads' T4 asserts T, M and A exactly and the CNOT count equal to the read's declared upper bound (its counting twin), labelled as a bound; the erase's Clifford counts are not asserted except the hinted `CZ` and `PauliZ` expectations of T6. |
| Ledger row 1C.4: `AngleErase` calls `emit_phase_fixup` directly (enable value 0 with an unaligned end inside the one-hot; a nonzero lower bound; branch types) | The SF fix-up runs enabled on `G0 = 0` (control value 0). For ob >= 1 its covered range `[0, B)` ends inside a one-hot leaf, the one holding `b = B`, which is never covered (certificate cases with ob = `w_b`: `sto3g`, `631g`, `R3-zero-blocks`, `R1`; with ob = 1: `R3-ob1`); ob = 0 (`R3-ob0`, and in T2 and T4) exercises the fallback, which has no one-hot leaf; the tail fix-up iterates `[RC, X_o)`, a nonzero lower bound; the branches are `CZ` (SF, ob >= 1) and `Z` (tail, SF ob = 0), the root X-conjugated for control value 0. `sf_onehot_bits` is validated to `[0, w_b]` before the helper's own range check. |
| Ledger row 1C.5: `AngleErase` never has a zero mask; reuse `QROAMEmitter.measure_x` and `.parities`; reversed outcomes | The erase uses both emitter methods; `test_no_mask_is_zero_and_parities_are_linear` asserts every mask nonzero (the `b_neq_B` and `G0` bits); the outcomes go to `mcm_parities` reversed, and the mutation check's "outcomes in wire order" mutant fails the certificates. |
| Ledger row 1C.5: certificate requirements (premise check, random complex input on every address and control value, partly covered fix-up leaf, zero block, tape path only) | `test_certificates_angles.py`: six cases; each asserts the tape has conditionals, all `PauliZ` or `CZ`, and `not nonlinear_conditionals(operations)`; random complex amplitude on every value of `x_o`, `r`, `b`; the suite includes partly covered SF leaves (every case with ob > 0 has one, as `b = B` and failed `b` are never covered; `R3-ob0` is the fallback); a zero angle entry in the STO-3G SF table and a zero tail row and SF entry in `R3-zero-blocks` (zero masks cannot occur: the flag bits); `test_certificate_cases_cover_the_branches` asserts the partly covered leaves and the zero blocks; the sparse simulator runs the tape path of `mcm_parities` only. |
| Ledger row 1A.5: E-1 at `H2_STO3G_TEST` cannot see the ladder order or a tail-direction error | `test_ladder_equals_ladder_matrix_and_the_opposite_order_fails` pins the ladder against `ladder_matrix` (Task 1A.4) for N = 2, 3, 4 and against the defining identity U† γ₀ U = Σ u_j γ_j; for N >= 3 it asserts that the opposite rung order and the reversed register order differ at its angles (both fail as mutants). The tail direction is pinned by the tail load's T3 on every address (each row's own angles, `G1` per row; the "G1 inverted" mutant fails); Task 3.1's `InnerSelect` tests carry the end-to-end check. |
| Ledger row 1C.1: new emitter gate names go into `as_resources` | No new names (see "Shared edits"). |
| Ledger rows 1C.3, 1C.4, 1C.5 (dfthcbe comparisons) and 1C.1 (s - 1 against s - 2) | Background for the next section and for Task 4.3; the uniform-preparation rotations are not in this workstream. |
| HANDOFF section 6, test tolerances | Every comparison states both tolerances (`rtol=0` with an absolute `atol`); `tests/test_tolerance_hygiene.py` passes with the new files. |

Ledger rows 1C.2 (prepares and unprepares) and 1A.5's normalization row concern Workstream D and Task 3.5.

## FeMoco-54 numbers, dfthcbe, and the `InnerSelect` sum

Ours are `dfthc.costs` on `main`, asserted without compiling (the table-free ladder and spin swap are also compiled at FeMoco's widths); every closed form equals the compiled count at both test presets and `MID_NAPHTHALENE`. dfthcbe's are its symbolic call graph at FeMoco-54 (`.venv-ref`, Qualtran 0.7.0).

| Block (per `InnerSelect`) | Master plan | Ours, `dfthc.costs` | dfthcbe | Ours − dfthcbe, cause |
| --- | ---: | ---: | ---: | --- |
| `TailAngleLoad` | 61 | 61 | 56 | +5: exact iteration over `[270, 324)` against the pruned walk past the zero SF prefix (spec 9.3, "Known deltas") |
| `SFAngleLoad` | 291 | 291 = 11 + 10 · 28 | 271 | +20: exact nested iteration with `r` outer (11 + 10 · 28), against dfthcbe's truncated walk with `b` outer and the identity row pruned (RB + s(R) − 1) |
| `SpinMux` (×2) | 1 each | 2 | 4 | −2: one Toffoli per application, against two `MultiControlX` |
| `SpinSwap` (×2) | N each | 108 | 108 | 0 |
| `GivensLadder` and `GivensLadderDagger` | 1,484 each | 2,968 | 2,968 | 0 (Q-5) |
| `MajoranaGate` | 5 | 5 | 5 | 0 |
| `AngleErase` | 102, M = 798 | 102 = (11 + 30) + 61, M = 798 | 90, M = 798 | +12: one-hot over 32 positions instead of 28 (+4), exact controlled iteration over `r` instead of truncated (+3), the tail fix-up as for the load (+5) |
| **`InnerSelect`** | 3,537 | 3,537 | 3,502 | +35 per select, +70 per walk (spec 13.1's angle-load, erase and multiplexer rows) |

The per-block costs add up: 61 + 291 + 2 + 108 + 2,968 + 5 + 102 = 3,537 = `costs.inner_select(FEMOCO54)`, and with Workstream D's closed forms and the reflections the walk is 210 + 69 + 2 (971 + 3,537 + 217) + 20 + 44 = 9,793 = `costs.walk(FEMOCO54)` (controlled 9,795); M = 96 + 2 (416 + 798) = 2,524. The E blocks hold 2N + 5 = 113 plain `Toffoli` gates per select (the spin swaps and multiplexers and the Majorana gate), the 4N + 10 of `costs.toffoli_only`'s per-walk terms. Compiled, the five tasks in `InnerSelect`'s order give T = 33, 65 and 625 and M = 7, 15 and 138 at `H2_STO3G_TEST`, `H2_631G_TEST` and `MID_NAPHTHALENE`, equal to `costs.inner_select`, forward and with the adjoint Majorana gate (`stages/2E/select_composition_probe.py`; at the mid preset 3.6 to 4.3 s and 336 MB).

## Findings for Task 3.1 and other plans

1. **Composition.** The five tasks compose as Task 3.1 needs them, each passed only its registers (spec 9.0) and a prefix of `WalkLayout.work`; the work needs are the tail load's `w_x - 1`, the SF load's `w_r + w_b`, the ladder's `b_rot - 1`, the Majorana gate's 2 and the erase's `max(w_r, w_x - 1)` at ob = `w_b`, all within `WalkLayout.work_size`. `InnerSelectDagger` differs only by `qp.adjoint(MajoranaGate(...))`; `T - A = 2N + 5` per select in both.
2. **The erase's precondition in the select.** Between the loads and `AngleErase`, `SpinMux`, the ladders and the Majorana gate use `angles`, `G0`, `G1` and `b_neq_B` only as controls or read-only inputs (`G0` controls `SpinMux` and the Majorana gate; the ladders' `SemiAdder`s read `angles`; `G1` and `b_neq_B` are controls of the Majorana gate), so the erase's precondition holds.
3. **Mutation evidence** (`stages/2E/2e-mutants.log`, 20 mutants, each one wrong line or block, all caught): for the erase, flags measured before the SF fix-up, the SF fix-up not enabled on `G0 = 0`, the SF mask without the `b_neq_B` bit, the tail mask without the `G0` bit, outcomes in wire order and the tail fix-up one row short each fail every certificate; the loads' G1 inverted, the SF load enabled on `G0 = 1`, little-endian angle registers, an R = 1 SF load that keeps the two-register address (it fails every `random-R1` load test, the compiled T3 and T4 included) and a tail constructor that takes `len` before checking the table's type (the scalar case raises `TypeError`) fail the load tests; the round-1 review named one plausible wrong implementation that the tests then missed, an R = 1 SF load that passes in tape mode but fails under capture, which the compiled T3 at `random-R1` now covers; the ladder's opposite order, the dagger in the ladder's order, a flipped rung sign and reversed rung registers fail the Givens tests; `+Z` on SF, D1 and Q1 exchanged and SF firing on a failed `b` fail the Majorana tests; the multiplexer following `sigma_1` on SF and an identity `SpinSwap` matrix fail the spin tests (the latter through T2's matrix check).

## Risks and open questions

1. **`compute_matrix` on `SpinMux` and `SpinSwap`.** A matrix-bearing `Operator2` compiles unchanged here (user and device level, pinned rules chosen). If a later toolchain lowered matrix-bearing operators differently (for example to `QubitUnitary`), T5 and T6 of `test_spin.py` would fail at the nightly bump; the fallback is to drop `compute_matrix` and pass `check_matrix=False`, as every other dfthc operator does.
2. **Rule-name collisions are checked per worker.** `tests/test_compile.py::test_registered_names_have_only_our_rules` sees only the modules its xdist worker imported; each new test file therefore asserts that its pinned rules are the only ones registered under its names (none collide on `main`'s toolchain).
3. **No open question blocks the plan** (revision 2 fixes both Important review findings). The decisions taken here (items 1 to 9 of the changes) are the obvious choices. Item 6 can be reverted by moving test parameters into the `slow` tier; reverting item 4 removes the two `compute_matrix` methods and turns `check_matrix` off in two tests.

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
| `src/dfthc/ops/angle_data.py` | 2E.1, replaced by 2E.2 | `angle_entry`, `TailAngleLoad`, `SFAngleLoad`; then `AngleErase` |
| `tests/ops/test_angle_loads.py` | 2E.1 | every address of four shapes, exclusive branches, compiled states (R = 1 included), counts, tables |
| `tests/ops/test_angle_erase.py` | 2E.2 | counts, rule selection, hints, smoke runs, parity linearity, validation |
| `tests/ops/test_certificates_angles.py` | 2E.2 | spec 12.5 certificates and their premise check (`slow`) |
| `src/dfthc/ops/spin.py` | 2E.3 | `SpinMux`, `SpinSwap` and their pinned self-adjoints |
| `tests/ops/test_spin.py` | 2E.3 | every basis input, matrices, counts |
| `src/dfthc/ops/givens.py` | 2E.4 | `GivensRotation`, `GivensLadder`, `GivensLadderDagger` |
| `tests/ops/test_givens.py` | 2E.4 | the rung for every angle, the ladder order, the dagger, counts |
| `src/dfthc/ops/majorana.py` | 2E.5 | `MajoranaGate` |
| `tests/ops/test_majorana.py` | 2E.5 | every branch, the flavour convention, counts |

Within wave 2-a (master plan section 5) the file lists of 2E.1, 2E.3, 2E.4 and 2E.5 are disjoint from each other and from Workstream D's; 2E.2 follows 2E.1 in wave 2-b (same module). Task 2E.2 replaces `angle_data.py` as a whole with the version it gives; Task 2E.1's code in it is unchanged (only the import of `dfthc.ops.qroam` grows and one import is added). `tests/ops/` gets no `__init__.py`, and test modules have unique base names.

**Shell conventions.** Run commands from the worktree root of your lane. `VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"` locates the shared environment, as in the earlier plans.

**Expected test counts.** Each task states its targeted count exactly. The per-commit tier (`scripts/test.sh -m "not slow and not release" -q`) counts **656 tests on `main` at `f7acf97`** (no Phase 2 task merged). A task's expected per-commit count is the per-commit count of `main` at its branch point plus its own per-commit tests; the table assumes the dry run's base and, for 2E.2, Task 2E.1 merged. Workstream D's tasks, and the other E tasks of wave 2-a, merge in an order the orchestrator chooses: shift by the difference. The pass criterion is "no failures, no errors, no warnings summary"; the number is a cross-check.

| Task | Wave | Base | Own per-commit | Expected per-commit tier | Slow tests |
| --- | --- | --- | ---: | ---: | ---: |
| 2E.1 | 2-a | 656 | 36 | 692 | 2 |
| 2E.3 | 2-a | 656 | 24 | 680 | 0 |
| 2E.4 | 2-a | 656 | 36 | 692 | 0 |
| 2E.5 | 2-a | 656 | 16 | 672 | 0 |
| 2E.2 | 2-b | 656 + 36 (2E.1) = 692 | 27 | 719 | 12 |

With all five tasks merged and nothing else, the per-commit tier counts 656 + 36 + 23 + 24 + 35 + 14 = 788 tests, and Workstream E adds 14 nightly tests. Dry run on the replica with all five tasks (revision 2): 788 collected, 787 passed plus the replica-only failure described under "Code provenance", in a pytest session of 303 s with 2 workers (log `stages/2E/2e-per-commit-tier.log`; round 1's 783 tests took 423 s). The process took 21 min of wall time to exit, against 425 s in round 1, on a machine shared with the Workstream D dry run; the session time is the comparable figure. Workstream E's 132 per-commit tests took 62 s together with 2 workers in the final replay. Its ten slowest (`stages/2E/2e-durations.log`, `--durations=10` of that run) took 2.98 to 4.90 s each: the verbose-listing and device-level compiles of `test_givens.py`, `test_angle_loads.py`, `test_spin.py` and `test_majorana.py` (4.90, 4.55, 4.39 and 4.04 s), the erase's hinted-conditional test (4.31 s), the compiled loads at their two shapes (4.25 s), and four Givens ladder and dagger tests (2.98 to 3.98 s). No E test needs the `slow` tier.

**Design notes the tasks share.**
- *Resource declarations.* `AngleErase` loops over compilable data, so its rule emits through `QROAMEmitter` and its resource function runs the same body in counting mode (the C plan's counting twin). The loads, the spin operators, the Givens classes and the Majorana gate are straight-line; they declare their resources by hand (the loads: one `QROAMRead` key), and T2 (`assert_valid_decomp_rule`, which compares the declaration with the emitted operations exactly) runs at a size in every branch of each rule.
- *Angle register and entries* (spec 6.7). `angle_entry(angles, b_rot)` is the big-endian integer of the `N - 1` angle registers, register 0 most significant. The tail load's entry is `angles ‖ G0 ‖ G1` and the SF load's `angles ‖ b_neq_B`, each one big-endian integer stored as 32-bit words, least significant first (L-24); the erase measures `angles ‖ b_neq_B` and then `G0 ‖ G1` and builds its masks from the same entries.
- *FeMoco-54* values of the table operators are asserted on the closed forms; nothing FeMoco-shaped is compiled (master plan section 12.9). The table-free ladders and the spin swap are compiled at FeMoco's widths.
- *Tolerances.* Every comparison states both tolerances (HANDOFF section 6); dense and simulator comparisons use `rtol=0` with `atol=1e-12`, compiled lightning states `atol=1e-10`.

---

### Task 2E.1: `TailAngleLoad`, `SFAngleLoad`

**Task id:** `2E.1` · **Branch:** `phase2/2E.1` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** G1

**Files:**
- Create: `src/dfthc/ops/angle_data.py` (the loads; Task 2E.2 replaces the module with a version that adds `AngleErase`), `tests/ops/test_angle_loads.py`

**Interfaces:**
- Consumes: `dfthc.ops.qroam` (`WORD`, `WORD_DTYPE`, `QROAMRead` with its static `leaf_words_from_table`, `leaf_words_shape`, `required_wires` and `visited_ranges`; in the tests also `QROAMEmitter` and `_read_body`; Task 1C.3); `dfthc.ops._validate` (`canonical_ints`, `check_int`, `check_wires`; Task 1C.1); `dfthc.compile` (`register_rule`, `resource_key`, `compiled_resources`, `GATE_SET`, `FIXED_DECOMPS`); `dfthc.params` (`lg`, `WalkParams`, `FEMOCO54`, `H2_STO3G_TEST`, `H2_631G_TEST`, `MID_NAPHTHALENE`); `dfthc.costs` (`tail_angle_load`, `sf_angle_load`, `qroam_read`); `dfthc.data.tables.WalkTables` and `dfthc.data.factors.load_factors` (Tasks 1A.2, 1A.4); `SparseSimulator`, `decompose_for_simulation` (Task 1B.2); `verbose_program`, `chosen_rules` (Task 0.2).
- Produces, in `dfthc.ops.angle_data`:
  - `angle_entry(angles, b_rot) -> int`: the big-endian integer of the angle registers holding `angles`, register 0 most significant (spec 6.7);
  - `TailAngleLoad(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work)`, all five leading arguments compilable: `tail_angles` an `N x (N - 1)` table of ints in `[0, 2**(b_rot - 1))` (D1 rows first, `N >= 2`), `0 <= n_plus <= N`, `RC >= 1`, `X_o = RC + N`, `b_rot >= 2`; static `read_shape(N, RC, X_o, b_rot) -> dict` (`address_widths`, `live_ranges`, `width`, `block_bits`), `table(tail_angles, n_plus, RC, b_rot) -> tuple` (`(address, words)` pairs with the entry `angles ‖ G0 ‖ G1`), `required_wires(tail_angles, n_plus, RC, X_o, b_rot) -> dict` (`x_o`: `lg(X_o)`, `angles`: `(N - 1) b_rot`, `G0`, `G1`: 1, `work`: `lg(X_o) - 1`); the rule `_tail_angle_load`;
  - `SFAngleLoad(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work)`, the four leading arguments compilable: `sf_angles` an `R x B x (N - 1)` table of ints in `[0, 2**(b_rot - 1))`, `R, B >= 1`, `b_rot >= 2`; enabled on `G0 = 0`; static `read_shape(N, R, B, b_rot)`, `table(sf_angles, B, b_rot)` (the entry `angles ‖ b_neq_B` at address `(r << w_b) | b`), `required_wires(sf_angles, R, B, b_rot)` (`r`: `lg(R)`, `b`: `lg(B + 1)`, `G0`, `b_neq_B`: 1, `angles`: `(N - 1) b_rot`, `work`: the read's); the rule `_sf_angle_load`;
  - module-private helpers Task 2E.2 builds on: `_tail_entries(tail_angles, n_plus, b_rot)`, `_sf_entries(sf_angles, b_rot)`, `_sf_address(R, B)`, `_words(value, width)`, `_join(*registers)`, `_check_angle_table(...)`, `TailAngleLoad._check(...)`, `SFAngleLoad._check(...)`.

**Acceptance tests** (38 tests in `tests/ops/test_angle_loads.py`, 2 of them `slow`), on four shapes: both test presets with their vendored tables (`h2_sto3g_R2_B2_C1`, `h2_631g_R2_B2_C2`), `random-R3` (N = 3, R = 3, B = 3: failed `x_o` 6 and 7, `r = 3` unvisited, `b = 3` the identity term) and `random-R1` (R = 1, the SF address `b` alone):
- T3 on the sparse simulator, every address (12 tests): the tail load on `H|x_o>` writes each tail row's angles, `G0 = 1` and `G1 = [x_o - RC >= N+]` and leaves SF and failed rows untouched; the SF load on `H` over `r`, `b` and the enable writes `(r, b)`'s angles and `b_neq_B = 1` exactly where `G0 = 0` and `(r, b)` is in range; after both loads every branch holds one family's data (the SF load leaves tail branches untouched and vice versa). Every term keeps amplitude `+1/sqrt N` and `work` returns to `|0>`. Expected values are computed bit by bit in the test, independently of `angle_entry`.
- T3 on the compiled path: both loads at `H2_STO3G_TEST` (15 lightning wires) and at `random-R1` (12 wires; the SF address `b` alone, with `b = B = 2` and the failed `b = 3`) give exactly the exclusive-branch state, compared in full; `random-R1`'s SF entries are distinct and nonzero (asserted), so a wrong `b` shows. This is the capture-path evidence for the one-register address (change 2, E-S3).
- T4: T, M = 0 and A = T against `costs.tail_angle_load` and `costs.sf_angle_load` at both test presets and for the SF load at `random-R1` (5 tests) and at `MID_NAPHTHALENE` (`slow`, 2 tests; tail 13, SF 55), with the CNOT count equal to the read's declared upper bound (`QROAMRead` has `exact=False`); FeMoco-54 on the closed forms (61; 11 + 10 · 28 = 291), the read shapes (797 and 796 bits, 54 and 270 leaves of 25 words) and work sizes (8 and 9).
- T5 at `level="device"` and T6 (the verbose listing names `_tail_angle_load`, `_sf_angle_load` and `_qroam_read`); T2 for both rules at three sizes, each rule the only one registered under its name; `angle_entry` and both tables by hand (a two-word entry included); `read_shape` and `required_wires`; eleven constructor violations, each a `ValueError` (spec 9.0), among them a scalar and a flat `tail_angles` and a scalar `sf_angles`.

**Definition of done:** `38 passed` targeted (36 in the per-commit tier); the per-commit tier passes; `ruff check .` clean; the spec 9.0 docstring sections in order; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_angle_loads.py`:

```python
"""TailAngleLoad and SFAngleLoad (spec 9.10, 9.11): every address, exclusive branches, counts."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.angle_data import (
    SFAngleLoad,
    TailAngleLoad,
    _sf_angle_load,
    _tail_angle_load,
    angle_entry,
)
from dfthc.ops.qroam import QROAMEmitter, QROAMRead, _read_body
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.sparse_sim import SparseSimulator
from dfthc.testing.tape import decompose_for_simulation


def random_tables(p, seed=3):
    """Random angle tables of p's shape, every angle below 2**(b_rot - 1)."""
    rng = np.random.default_rng(seed)
    bound = 2 ** (p.b_rot - 1)
    tail = tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.N))
    sf = tuple(tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.B))
               for _ in range(p.R))
    return tail, sf


def register_value(angles, b_rot):
    """The angle registers' big-endian integer when register k holds angles[k] (spec 6.7), written
    out bit by bit, independently of the module's angle_entry."""
    return int("".join(format(angle, f"0{b_rot}b") for angle in angles), 2)


def vendored_tables(system, p):
    tables = WalkTables.from_factors(load_factors(system), p)
    return tables.tail_angles, tables.sf_angles


# R = 3 leaves r = 3 unvisited, X_o = 6 leaves x_o = 6, 7 failed, B = 3 makes b = 3 the identity
# term; R = 1 gives an SF address of b alone (r has no wires).
RANDOM_R3 = WalkParams(N=3, R=3, B=3, C=1, n_plus=1, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
RANDOM_R1 = WalkParams(N=2, R=1, B=2, C=2, n_plus=2, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
SHAPES = {
    "sto3g": (H2_STO3G_TEST, vendored_tables("h2_sto3g_R2_B2_C1", H2_STO3G_TEST)),
    "631g": (H2_631G_TEST, vendored_tables("h2_631g_R2_B2_C2", H2_631G_TEST)),
    "random-R3": (RANDOM_R3, random_tables(RANDOM_R3)),
    "random-R1": (RANDOM_R1, random_tables(RANDOM_R1, seed=0)),
}


def layout(p, tail, sf):
    """Wires: x_o, r, b, angles, G0, G1, b_neq_B, then work for the larger load."""
    sizes = {"x_o": p.w_x, "r": p.w_r, "b": p.w_b, "angles": (p.N - 1) * p.b_rot, "G0": 1,
             "G1": 1, "b_neq_B": 1}
    tail_work = TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot)["work"]
    sf_work = SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot)["work"]
    sizes["work"] = max(tail_work, sf_work)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start, tail_work, sf_work


def tail_load(p, tail, wires, work):
    TailAngleLoad(tail, p.n_plus, p.RC, p.X_o, p.b_rot, wires["x_o"], wires["angles"],
                  wires["G0"], wires["G1"], wires["work"][:work])


def sf_load(p, sf, wires, work):
    SFAngleLoad(sf, p.R, p.B, p.b_rot, wires["r"], wires["b"], wires["G0"], wires["angles"],
                wires["b_neq_B"], wires["work"][:work])


def value_of(state, register, row):
    return sum(int(state.bits(w)[row]) << (len(register) - 1 - i) for i, w in enumerate(register))


def simulate(num_wires, inputs, body):
    """H on every input wire, then ``body``, on the sparse simulator; every basis value of the
    inputs must occur once, with amplitude +1/sqrt(2**len(inputs)) (no phase)."""

    def qfunc():
        for wire in inputs:
            qp.H(wire)
        body()

    state = SparseSimulator(num_wires).run(decompose_for_simulation(qfunc).operations)
    assert len(state.amps) == 2 ** len(inputs)
    assert np.allclose(state.amps, 2 ** (-len(inputs) / 2), rtol=0, atol=1e-12)
    return state


@pytest.mark.parametrize("name", SHAPES)
def test_tail_load_on_every_address(name):
    """T3 (sparse simulator, H on x_o): tail rows x_o in [RC, X_o) get their angles, G0 = 1 and
    G1 = [x_o - RC >= N+]; SF rows and failed x_o >= X_o are untouched; b_neq_B, r and b are not
    the tail load's wires and stay |0>."""
    p, (tail, sf) = SHAPES[name]
    wires, num_wires, tail_work, _ = layout(p, tail, sf)
    state = simulate(num_wires, wires["x_o"], lambda: tail_load(p, tail, wires, tail_work))
    for row in range(len(state.amps)):
        x_o = value_of(state, wires["x_o"], row)
        is_tail = p.RC <= x_o < p.X_o
        expected = (register_value(tail[x_o - p.RC], p.b_rot) if is_tail else 0,
                    int(is_tail), int(is_tail and x_o - p.RC >= p.n_plus), 0, 0)
        actual = tuple(value_of(state, wires[k], row)
                       for k in ("angles", "G0", "G1", "b_neq_B", "work"))
        assert actual == expected, x_o


@pytest.mark.parametrize("name", SHAPES)
def test_sf_load_on_every_address(name):
    """T3 (sparse simulator, H on r, b and the enable G0): where G0 = 0 and (r, b) lies in
    [0, R) x [0, B), the angles of (r, b) and b_neq_B = 1; nothing on G0 = 1, b = B, failed b or
    r >= R; G1 is not the SF load's wire and stays |0>."""
    p, (tail, sf) = SHAPES[name]
    wires, num_wires, _, sf_work = layout(p, tail, sf)
    state = simulate(num_wires, wires["r"] + wires["b"] + wires["G0"],
                     lambda: sf_load(p, sf, wires, sf_work))
    for row in range(len(state.amps)):
        r, b, g0 = (value_of(state, wires[k], row) for k in ("r", "b", "G0"))
        on = g0 == 0 and r < p.R and b < p.B
        expected = (register_value(sf[r][b], p.b_rot) if on else 0, int(on), 0, 0)
        actual = tuple(value_of(state, wires[k], row) for k in ("angles", "b_neq_B", "G1", "work"))
        assert actual == expected, (r, b, g0)


@pytest.mark.parametrize("name", SHAPES)
def test_loads_write_exclusive_branches(name):
    """T3 (sparse simulator, H on x_o, r and b): after both loads every branch holds one
    family's data: a tail row its angles, G0 = 1, G1 and b_neq_B = 0 whatever (r, b) is (the SF
    load leaves it untouched); any other x_o the SF entry of (r, b) and b_neq_B = 1 if (r, b) is in
    range, else zeros, with G0 = G1 = 0 (the tail load left it untouched)."""
    p, (tail, sf) = SHAPES[name]
    wires, num_wires, tail_work, sf_work = layout(p, tail, sf)

    def body():
        tail_load(p, tail, wires, tail_work)
        sf_load(p, sf, wires, sf_work)

    state = simulate(num_wires, wires["x_o"] + wires["r"] + wires["b"], body)
    for row in range(len(state.amps)):
        x_o, r, b = (value_of(state, wires[k], row) for k in ("x_o", "r", "b"))
        if p.RC <= x_o < p.X_o:
            expected = (register_value(tail[x_o - p.RC], p.b_rot), 1,
                        int(x_o - p.RC >= p.n_plus), 0)
        else:
            on = r < p.R and b < p.B
            expected = (register_value(sf[r][b], p.b_rot) if on else 0, 0, 0, int(on))
        actual = tuple(value_of(state, wires[k], row) for k in ("angles", "G0", "G1", "b_neq_B"))
        assert actual == expected, (x_o, r, b)
        assert value_of(state, wires["work"], row) == 0


def compiled_loads_state(name):
    """Both loads on H over x_o, r and b, compiled and run on lightning, and the state the
    exclusive-branch table predicts."""
    p, (tail, sf) = SHAPES[name]
    wires, num_wires, tail_work, sf_work = layout(p, tail, sf)
    inputs = wires["x_o"] + wires["r"] + wires["b"]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit():
        for wire in inputs:
            qp.H(wire)
        tail_load(p, tail, wires, tail_work)
        sf_load(p, sf, wires, sf_work)
        return qp.state()

    expected = np.zeros(2**num_wires)
    for value in range(2 ** len(inputs)):
        bits = dict(zip(inputs, ((value >> (len(inputs) - 1 - i)) & 1
                                 for i in range(len(inputs))), strict=True))
        x_o = sum(bits[w] << (p.w_x - 1 - i) for i, w in enumerate(wires["x_o"]))
        r = sum(bits[w] << (p.w_r - 1 - i) for i, w in enumerate(wires["r"]))
        b = sum(bits[w] << (p.w_b - 1 - i) for i, w in enumerate(wires["b"]))
        if p.RC <= x_o < p.X_o:
            registers = (register_value(tail[x_o - p.RC], p.b_rot), 1,
                         int(x_o - p.RC >= p.n_plus), 0)
        else:
            on = r < p.R and b < p.B
            registers = (register_value(sf[r][b], p.b_rot) if on else 0, 0, 0, int(on))
        index = value
        for register, content in zip(("angles", "G0", "G1", "b_neq_B"), registers, strict=True):
            index = (index << len(wires[register])) | content
        expected[index << len(wires["work"])] = 2 ** (-len(inputs) / 2)
    return np.asarray(circuit()), expected


@pytest.mark.compiled
def test_compiled_loads_on_lightning():
    """T3 on the compiled path at H2_STO3G_TEST (15 wires) and at random-R1 (12 wires; the SF
    address is b alone, and b = B = 2 and the failed b = 3 must stay untouched): both loads give
    exactly the state of the exclusive-branch table, so the rules' leaf words are built and
    loaded correctly under capture, without phases, and the one-register SF address lowers. The
    R1 SF entries are distinct and nonzero, so a wrong b shows."""
    _, (_, sf) = SHAPES["random-R1"]
    assert len(set(sf[0])) == len(sf[0]) and all(any(entry) for entry in sf[0])
    for name in ("sto3g", "random-R1"):
        state, expected = compiled_loads_state(name)
        assert np.allclose(state, expected, rtol=0, atol=1e-10), name


def declared_cnots(read, control_value):
    """The read's declared CNOT upper bound (its counting twin): load CNOTs for every bit of
    every leaf plus the iteration's CNOTs."""
    counter = QROAMEmitter(counting=True)
    sizes = QROAMRead.required_wires(**read, control_value=control_value)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = range(start, start + size)
        start += size
    _read_body(counter, None, **read, control_value=control_value, **wires)
    return counter.counts["CNOT"]


def assert_load_counts(p, tail, sf, which):
    """T4: T, M and A exact against dfthc.costs; the CNOT count, an upper bound for the data
    (QROAMRead has exact=False), equals the declared bound."""
    wires, num_wires, tail_work, sf_work = layout(p, tail, sf)
    if which == "tail":
        resources = compiled_resources(lambda: tail_load(p, tail, wires, tail_work), num_wires)
        expected = costs.tail_angle_load(p)
        read, control_value = TailAngleLoad.read_shape(p.N, p.RC, p.X_o, p.b_rot), None
    else:
        resources = compiled_resources(lambda: sf_load(p, sf, wires, sf_work), num_wires)
        expected = costs.sf_angle_load(p)
        read, control_value = SFAngleLoad.read_shape(p.N, p.R, p.B, p.b_rot), 0
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli
    assert resources.counts["CNOT"] == declared_cnots(read, control_value)


@pytest.mark.compiled
@pytest.mark.parametrize(("name", "which"), [("sto3g", "tail"), ("sto3g", "sf"), ("631g", "tail"),
                                             ("631g", "sf"), ("random-R1", "sf")])
def test_toffoli_count_compiled(name, which):
    """T4 at H2_STO3G_TEST and H2_631G_TEST with the vendored tables, and the SF load at
    random-R1 (the one-register address)."""
    p, (tail, sf) = SHAPES[name]
    assert_load_counts(p, tail, sf, which)


@pytest.mark.compiled
@pytest.mark.slow
@pytest.mark.parametrize("which", ["tail", "sf"])
def test_toffoli_count_at_mid_naphthalene(which):
    """T4 at MID_NAPHTHALENE with the naphthalene tables (nightly tier): tail 13, SF 55."""
    assert_load_counts(MID_NAPHTHALENE, *vendored_tables("naphthalene", MID_NAPHTHALENE), which)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 on the closed forms (no FeMoco table is compiled, master plan 12.9): tail 61 and
    SF 11 + 10 * 28 = 291; the reads have 54 and 270 leaves of 25 words (797 and 796 bits)."""
    p = FEMOCO54
    assert costs.tail_angle_load(p) == costs.Cost(61)
    assert costs.sf_angle_load(p) == costs.Cost(11 + 10 * 28) == costs.Cost(291)
    tail = TailAngleLoad.read_shape(p.N, p.RC, p.X_o, p.b_rot)
    sf = SFAngleLoad.read_shape(p.N, p.R, p.B, p.b_rot)
    assert tail == {"address_widths": (9,), "live_ranges": ((270, 324),), "width": 797,
                    "block_bits": 0}
    assert sf == {"address_widths": (4, 5), "live_ranges": ((0, 10), (0, 27)), "width": 796,
                  "block_bits": 0}
    assert QROAMRead.leaf_words_shape(**tail) == (54, 1, 25)
    assert QROAMRead.leaf_words_shape(**sf) == (270, 1, 25)
    assert costs.qroam_read(QROAMRead.visited_ranges(tail["address_widths"], tail["live_ranges"],
                                                     0), 0, 797, False) == costs.Cost(61)
    assert costs.qroam_read(QROAMRead.visited_ranges(sf["address_widths"], sf["live_ranges"], 0),
                            0, 796, True) == costs.Cost(291)
    assert QROAMRead.required_wires(**tail, control_value=None)["work"] == 8
    assert QROAMRead.required_wires(**sf, control_value=0)["work"] == 9


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names both pinned rules and the read's; T5 at level="device"."""
    p, (tail, sf) = SHAPES["sto3g"]
    wires, num_wires, tail_work, sf_work = layout(p, tail, sf)

    def program():
        tail_load(p, tail, wires, tail_work)
        sf_load(p, sf, wires, sf_work)

    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    assert {"_tail_angle_load", "_sf_angle_load", "_qroam_read"} <= set(
        chosen_rules(captured.out + captured.err))
    compiled_resources(program, num_wires=num_wires, level="device")


@pytest.mark.parametrize("name", ["sto3g", "random-R3", "random-R1"])
def test_rules_are_valid(name):
    """T2 (tape mode) for both rules at three sizes, one with a one-register SF address; each
    pinned rule is the only rule registered under its class name (spec 7.4 rule 8)."""
    assert list(qp.list_decomps(TailAngleLoad)) == [_tail_angle_load]
    assert list(qp.list_decomps(SFAngleLoad)) == [_sf_angle_load]
    p, (tail, sf) = SHAPES[name]
    wires, _, tail_work, sf_work = layout(p, tail, sf)
    qp.testing.assert_valid_decomp_rule(
        TailAngleLoad(tail, p.n_plus, p.RC, p.X_o, p.b_rot, wires["x_o"], wires["angles"],
                      wires["G0"], wires["G1"], wires["work"][:tail_work]),
        _tail_angle_load, check_matrix=False)
    qp.testing.assert_valid_decomp_rule(
        SFAngleLoad(sf, p.R, p.B, p.b_rot, wires["r"], wires["b"], wires["G0"],
                    wires["angles"], wires["b_neq_B"], wires["work"][:sf_work]),
        _sf_angle_load, check_matrix=False)


def test_entries_and_tables_by_hand():
    """angle_entry puts register 0 first (spec 6.7); the tail entry is angles ‖ G0 ‖ G1 and the
    SF entry angles ‖ b_neq_B, in 32-bit words, least significant first."""
    assert angle_entry((1, 2, 3), 3) == 0b001_010_011
    assert TailAngleLoad.table(((1,), (2,), (3,)), 2, 5, 3) == (
        (5, (0b001_1_0,)), (6, (0b010_1_0,)), (7, (0b011_1_1,)))
    assert SFAngleLoad.table((((1,), (2,), (3,)), ((0,), (1,), (2,))), 3, 3) == (
        (0, (0b001_1,)), (1, (0b010_1,)), (2, (0b011_1,)),
        (4, (0b000_1,)), (5, (0b001_1,)), (6, (0b010_1,)))
    # 47 bits in two words: ((16383 << 30) | 1) << 2 | 0b10 = (16383 << 32) | 6.
    wide = TailAngleLoad.table(((2**14 - 1, 0, 1),) + ((0, 0, 0),) * 3, 4, 1, 15)
    assert wide[0] == (1, (6, 2**14 - 1))


def test_read_shapes_and_required_wires():
    p, (tail, sf) = SHAPES["random-R1"]
    assert SFAngleLoad.read_shape(p.N, p.R, p.B, p.b_rot) == {
        "address_widths": (2,), "live_ranges": ((0, 2),), "width": 4, "block_bits": 0}
    assert SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot) == {
        "r": 0, "b": 2, "G0": 1, "angles": 3, "b_neq_B": 1, "work": 2}
    p, (tail, sf) = SHAPES["631g"]
    assert TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot) == {
        "x_o": 3, "angles": 12, "G0": 1, "G1": 1, "work": 2}


TAIL = ((1,), (2,))
SF = (((1,), (2,)), ((3,), (0,)))


def tail_args(**changes):
    """Valid TailAngleLoad arguments (N = 2, RC = 2, b_rot = 3), with ``changes``."""
    return {"tail_angles": TAIL, "n_plus": 1, "RC": 2, "X_o": 4, "b_rot": 3, "x_o": [0, 1],
            "angles": [2, 3, 4], "G0": [5], "G1": [6], "work": [7]} | changes


def sf_args(**changes):
    """Valid SFAngleLoad arguments (R = B = 2, N = 2, b_rot = 3), with ``changes``."""
    return {"sf_angles": SF, "R": 2, "B": 2, "b_rot": 3, "r": [0], "b": [1, 2], "G0": [3],
            "angles": [4, 5, 6], "b_neq_B": [7], "work": [8, 9, 10]} | changes


@pytest.mark.parametrize(("build", "match"), [
    (lambda: TailAngleLoad(**tail_args(tail_angles=((4,), (2,)))),
     r"angles must lie in \[0, 2\*\*\(b_rot - 1\)\) = \[0, 4\), got 4"),
    (lambda: TailAngleLoad(**tail_args(tail_angles=((1, 2), (2, 1)))), r"shape \(2, 1\)"),
    (lambda: TailAngleLoad(**tail_args(tail_angles=3)),
     r"tail_angles must be a table of ints of shape \(N, N - 1\)"),
    (lambda: TailAngleLoad(**tail_args(tail_angles=(1, 2))),
     r"tail_angles must be a table of ints of shape \(2, 1\)"),
    (lambda: TailAngleLoad(**tail_args(X_o=5)), r"X_o must lie in \[4, 4\]"),
    (lambda: TailAngleLoad(**tail_args(n_plus=3)), r"n_plus must lie in \[0, 2\]"),
    (lambda: TailAngleLoad(**tail_args(work=[])), "'work' has 0 wires; expected 1"),
    (lambda: SFAngleLoad(**sf_args(R=3)), r"shape \(3, 2, 1\)"),
    (lambda: SFAngleLoad(**sf_args(sf_angles=3)), r"sf_angles must be a table of shape \(R, B,"),
    (lambda: SFAngleLoad(**sf_args(b=[1])), "'b' has 1 wires; expected 2"),
    (lambda: SFAngleLoad(**sf_args(b_rot=1)), r"b_rot must lie in \[2,"),
])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_angle_loads.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.ops.angle_data'`.

- [ ] **Step 3: Write the loads**

Create `src/dfthc/ops/angle_data.py`:

```python
"""The angle registers of the inner select: loads and measured erase (spec 9.10, 9.11, 9.17).

**The shared angle register** (spec 6.7). ``angles`` is the concatenation of ``N - 1`` registers
of ``b_rot`` wires; register ``k`` drives rung ``k`` of the Givens ladder and holds the angle
integer ``x_k`` big-endian, with ``x_k < 2**(b_rot - 1)`` (spec 6.3). :func:`angle_entry` is the
big-endian integer of the whole register, register 0 most significant.

**Two exclusive loads.** :class:`TailAngleLoad` writes the one-body rows ``x_o`` in
``[RC, X_o)``: their angles, ``G0 = 1`` and ``G1 = [x_o - RC >= N+]``. :class:`SFAngleLoad` then
writes, where ``G0 = 0``, the angles of ``(r, b)`` in ``[0, R) x [0, B)`` and ``b_neq_B = 1``.
Each is a :class:`~dfthc.ops.qroam.QROAMRead` without a swap network (``block_bits = 0``) whose
entry is one big-endian integer over its target (``angles ‖ G0 ‖ G1``, ``angles ‖ b_neq_B``) and
whose ``leaf_words`` the rule builds from the compilable angle table (spec 9.3). After both
loads, tail rows hold tail data on ``G0 = 1``; every ``G0 = 0`` branch, failed ``x_o``
included, holds the SF entry of ``(r, b)`` where it is in range, and zeros elsewhere
(``b = B``, failed ``b``, unvisited ``r``). :class:`AngleErase` (Task 2E.2) erases what they
write.
"""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import AbstractArray, Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.qroam import WORD, WORD_DTYPE, QROAMRead
from dfthc.params import lg


def angle_entry(angles, b_rot: int) -> int:
    """The big-endian integer of the angle registers holding ``angles``: register ``k`` holds
    ``angles[k]``, register 0 most significant (spec 6.7)."""
    value = 0
    for angle in angles:
        value = (value << b_rot) | angle
    return value


def _words(value: int, width: int) -> tuple[int, ...]:
    """``value`` as ``ceil(width / 32)`` words, least significant first (L-24)."""
    return tuple((value >> (WORD * w)) & (2**WORD - 1) for w in range(-(-width // WORD)))


def _check_angle_table(name: str, table, shape: tuple[int, ...], b_rot: int) -> tuple:
    """``table`` as nested tuples of Python ints of ``shape``, every angle in
    ``[0, 2**(b_rot - 1))`` (spec 6.3); ``ValueError`` otherwise."""
    table = canonical_ints(name, table)

    def check(value, dims):
        if not dims:
            if not isinstance(value, int):
                raise ValueError(f"{name} must be a table of ints of shape {shape}")
            if not 0 <= value < 2 ** (b_rot - 1):
                raise ValueError(f"{name} angles must lie in [0, 2**(b_rot - 1)) = "
                                 f"[0, {2 ** (b_rot - 1)}), got {value}")
            return
        if not isinstance(value, tuple) or len(value) != dims[0]:
            raise ValueError(f"{name} must be a table of ints of shape {shape}")
        for item in value:
            check(item, dims[1:])

    check(table, shape)
    return table


def _join(*registers) -> list:
    """The wires of ``registers``, concatenated in order."""
    return [wire for register in registers for wire in register]


def _read_key(read: dict, control_value):
    """The resource key of the loads' ``QROAMRead`` for the read shape ``read``."""
    words = AbstractArray(QROAMRead.leaf_words_shape(**read), WORD_DTYPE)
    sizes = QROAMRead.required_wires(**read, control_value=control_value)
    return resource_key(lambda: QROAMRead(words, **read, control_value=control_value,
                                          **{name: Wire[n] for name, n in sizes.items()}))


# --------------------------------------------------------------------------- TailAngleLoad


def _tail_entries(tail_angles, n_plus: int, b_rot: int) -> list[int]:
    """Entry of tail row ``j``: ``angles ‖ G0 ‖ G1`` big-endian, ``G0 = 1``, ``G1 = [j >= N+]``."""
    return [(angle_entry(row, b_rot) << 2) | 2 | int(j >= n_plus)
            for j, row in enumerate(tail_angles)]


class TailAngleLoad(Operator2):
    r"""Load the one-body rows' Givens angles and the family flags ``G0``, ``G1`` (step 10).

    **Action.** For ``x_o`` in ``[RC, X_o)``, with ``j = x_o - RC``: the angle registers get
    ``tail_angles[j]`` (XOR), ``G0`` gets 1 and ``G1`` gets :math:`[j \ge N_+]` (D1 rows first,
    then Q1; spec 6.1). Every other ``x_o`` (the SF rows ``x_o < RC`` and the failed
    ``x_o >= X_o``) is untouched. ``x_o`` is unchanged and no phase is added.

    **Registers.** With ``N = len(tail_angles)`` and :math:`w_x = \lg X_o`:

    ========== ======================= ======== =========================================
    name       width                   input    output
    ========== ======================= ======== =========================================
    ``x_o``    :math:`w_x`             any      unchanged
    ``angles`` :math:`(N - 1) b_{rot}` ``|0>``  ``tail_angles[x_o - RC]`` on tail rows
    ``G0``     1                       ``|0>``  :math:`[RC \le x_o < X_o]`
    ``G1``     1                       ``|0>``  :math:`[x_o - RC \ge N_+]` on tail rows
    ``work``   :math:`w_x - 1`         ``|0>``  ``|0>``
    ========== ======================= ======== =========================================

    **Compilable arguments.** ``tail_angles``: ``N x (N - 1)`` ints in
    ``[0, 2**(b_rot - 1))``, the ``N+`` D1 rows first (``WalkTables.tail_angles``), ``N >= 2``;
    ``n_plus`` (``0 <= N+ <= N``); ``RC`` (int >= 1); ``X_o`` (``RC + N``); ``b_rot``
    (int >= 2). ``b_rot`` sizes the angle registers; it is not in the master plan's interface
    but determines the structure (spec 7.4 rules 1 and 3).

    **Pre- and postconditions.** ``angles``, ``G0``, ``G1`` and ``work`` start in ``|0>``;
    ``work`` ends in ``|0>``. After both loads, :class:`AngleErase` erases the shared
    ``angles``, ``b_neq_B``, ``G0`` and ``G1`` registers.

    **Decomposition.** One :class:`~dfthc.ops.qroam.QROAMRead` of :meth:`table` with address
    ``x_o`` over ``[RC, X_o)``, no swap network (``block_bits = 0``), no enable, and target
    ``angles ‖ G0 ‖ G1`` (:meth:`read_shape`). The rule builds the read's ``leaf_words`` from
    the compilable ``tail_angles`` with ``QROAMRead.leaf_words_from_table`` (spec 9.3).

    **Adjoint and control.** Never adjointed or controlled (its read has loops; spec 7.4 rules
    4 and 6). :class:`AngleErase` undoes it by measurement.

    **Cost.** :func:`dfthc.costs.tail_angle_load`: :math:`T = U(RC, X_o, w_x, 0)` (``A = T``),
    ``M = 0``. The read's CNOT count is an upper bound (``exact=False`` in ``QROAMRead``). At
    FeMoco-54: 61, with 54 leaves of 25 words each.

    **Paper and dfthcbe.** Step 10, the D1/Q1 angle load (PRX Eq. (B41)); dfthcbe:
    ``tail_rotation_qrom`` (``dfthcbe/rot_data.py``), which prunes the SF prefix to aligned
    blocks (56 at FeMoco-54) where ours iterates exactly (spec 9.3, "Known deltas").

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.angle_data import TailAngleLoad

        tail = ((1,), (2,))  # N = 2, b_rot = 3
        program = lambda: TailAngleLoad(tail, 1, 2, 4, 3, x_o=[0, 1], angles=[2, 3, 4], G0=[5],
                                        G1=[6], work=[7])
        compiled_resources(program, num_wires=8).toffoli  # U(2, 4, 2, 0) = 1
    """

    compilable_argnames = ("tail_angles", "n_plus", "RC", "X_o", "b_rot")
    wire_argnames = ("x_o", "angles", "G0", "G1", "work")

    def __init__(self, tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
        tail_angles, n_plus, RC, X_o, b_rot = self._check(tail_angles, n_plus, RC, X_o, b_rot)
        check_wires("TailAngleLoad",
                    {"x_o": x_o, "angles": angles, "G0": G0, "G1": G1, "work": work},
                    self.required_wires(tail_angles, n_plus, RC, X_o, b_rot))
        super().__init__(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work)

    @staticmethod
    def _check(tail_angles, n_plus, RC, X_o, b_rot) -> tuple:
        b_rot = check_int("b_rot", b_rot, low=2)
        rows = canonical_ints("tail_angles", tail_angles)
        if not isinstance(rows, tuple):
            raise ValueError("tail_angles must be a table of ints of shape (N, N - 1)")
        num_rows = len(rows)
        if num_rows < 2:
            raise ValueError(f"tail_angles needs N >= 2 rows, got {num_rows}")
        tail_angles = _check_angle_table("tail_angles", rows, (num_rows, num_rows - 1), b_rot)
        n_plus = check_int("n_plus", n_plus, low=0, high=num_rows)
        RC = check_int("RC", RC, low=1)
        X_o = check_int("X_o", X_o, low=RC + num_rows, high=RC + num_rows)
        return tail_angles, n_plus, RC, X_o, b_rot

    @staticmethod
    def read_shape(N: int, RC: int, X_o: int, b_rot: int) -> dict:
        """The read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``."""
        return {"address_widths": (lg(X_o),), "live_ranges": ((RC, X_o),),
                "width": (N - 1) * b_rot + 2, "block_bits": 0}

    @staticmethod
    def table(tail_angles, n_plus: int, RC: int, b_rot: int) -> tuple:
        """The read's table: ``(RC + j, words)`` with the entry ``angles ‖ G0 ‖ G1`` of row ``j``
        in 32-bit words, least significant first."""
        width = (len(tail_angles) - 1) * b_rot + 2
        return tuple((RC + j, _words(entry, width))
                     for j, entry in enumerate(_tail_entries(tail_angles, n_plus, b_rot)))

    @staticmethod
    def required_wires(tail_angles, n_plus, RC, X_o, b_rot) -> dict[str, int]:
        """Exact register sizes."""
        read = TailAngleLoad.read_shape(len(tail_angles), RC, X_o, b_rot)
        return {"x_o": lg(X_o), "angles": (len(tail_angles) - 1) * b_rot, "G0": 1, "G1": 1,
                "work": QROAMRead.required_wires(**read, control_value=None)["work"]}


def _tail_angle_load_resources(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
    return {_read_key(TailAngleLoad.read_shape(len(tail_angles), RC, X_o, b_rot), None): 1}


@qp.register_resources(_tail_angle_load_resources)
def _tail_angle_load(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
    read = TailAngleLoad.read_shape(len(tail_angles), RC, X_o, b_rot)
    words = QROAMRead.leaf_words_from_table(TailAngleLoad.table(tail_angles, n_plus, RC, b_rot),
                                            **read)
    QROAMRead(words, **read, control_value=None, address=x_o, target=_join(angles, G0, G1),
              junk=[], work=work, control=[])


register_rule(TailAngleLoad, _tail_angle_load)


# --------------------------------------------------------------------------- SFAngleLoad


def _sf_address(R: int, B: int) -> tuple[tuple, tuple]:
    """``(address_widths, ranges)`` of ``r ‖ b`` over ``[0, R) x [0, B)``; only ``b`` if
    ``R = 1`` (then ``r`` has no wires)."""
    if lg(R) == 0:
        return (lg(B + 1),), ((0, B),)
    return (lg(R), lg(B + 1)), ((0, R), (0, B))


def _sf_entries(sf_angles, b_rot: int) -> list[list[int]]:
    """Entry of ``(r, b)``: ``angles ‖ b_neq_B`` big-endian, ``b_neq_B = 1``."""
    return [[(angle_entry(entry, b_rot) << 1) | 1 for entry in plane] for plane in sf_angles]


class SFAngleLoad(Operator2):
    r"""Load the SF Givens angles and ``b_neq_B`` at ``(r, b)``, enabled on ``G0 = 0`` (step 11).

    **Action.** Where ``G0 = 0`` and ``(r, b)`` lies in ``[0, R) x [0, B)``: the angle registers
    get ``sf_angles[r][b]`` (XOR) and ``b_neq_B`` gets 1. Everywhere else (``G0 = 1``, the
    identity term ``b = B``, failed ``b > B``, ``r >= R``) nothing changes. ``r``, ``b`` and
    ``G0`` are unchanged and no phase is added. In the walk it follows :class:`TailAngleLoad`,
    which sets ``G0`` on the one-body rows, so the two loads never write the same branch.

    **Registers.** With :math:`w_r = \lg R` and :math:`w_b = \lg(B + 1)`:

    =========== ======================= ======== =========================================
    name        width                   input    output
    =========== ======================= ======== =========================================
    ``r``       :math:`w_r` (0 if R = 1) any     unchanged
    ``b``       :math:`w_b`             any      unchanged
    ``G0``      1                       any      unchanged (the enable, value 0)
    ``angles``  :math:`(N - 1) b_{rot}` ``|0>``  ``sf_angles[r][b]`` where enabled and in range
    ``b_neq_B`` 1                       ``|0>``  1 where enabled and in range
    ``work``    :math:`w_r + w_b`       ``|0>``  ``|0>``
    =========== ======================= ======== =========================================

    **Compilable arguments.** ``sf_angles``: ``R x B x (N - 1)`` ints in
    ``[0, 2**(b_rot - 1))`` (``WalkTables.sf_angles``), ``N >= 2``; ``R`` and ``B`` (ints >= 1,
    its first two sizes); ``b_rot`` (int >= 2), which sizes the angle registers (not in the
    master plan's interface; spec 7.4 rules 1 and 3).

    **Pre- and postconditions.** On enabled branches ``angles`` and ``b_neq_B`` start in
    ``|0>`` (they hold the tail load's data only where ``G0 = 1``); ``work`` starts and ends in
    ``|0>``. :class:`AngleErase` erases ``angles`` and ``b_neq_B``.

    **Decomposition.** One :class:`~dfthc.ops.qroam.QROAMRead` of :meth:`table` with address
    ``r ‖ b`` (``r`` outer over ``[0, R)``, ``b`` inner over ``[0, B)``; ``b`` alone if
    ``R = 1``), no swap network, the enable ``G0`` with control value 0, and target
    ``angles ‖ b_neq_B`` (:meth:`read_shape`). The rule builds the read's ``leaf_words`` from
    the compilable ``sf_angles`` (spec 9.3).

    **Adjoint and control.** Never adjointed or controlled (spec 7.4 rules 4 and 6).

    **Cost.** :func:`dfthc.costs.sf_angle_load`:
    :math:`T = U(0, R, w_r, 1) + R\,U(0, B, w_b, 1)` (``A = T``), ``M = 0``; the read's CNOT
    count is an upper bound. At FeMoco-54: :math:`11 + 10 \cdot 28 = 291`, with 270 leaves of
    25 words each.

    **Paper and dfthcbe.** Step 11, the SF angle load (PRX Table III, :math:`RB`); dfthcbe:
    ``sf_rotation_qrom`` (``dfthcbe/rot_data.py``), nested ``b`` outer and ``r`` inner with the
    identity row pruned (271 at FeMoco-54); ours nests ``r`` outer and iterates exactly.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.angle_data import SFAngleLoad

        sf = (((1,), (2,)), ((3,), (0,)))  # R = 2, B = 2, N = 2, b_rot = 3
        program = lambda: SFAngleLoad(sf, 2, 2, 3, r=[0], b=[1, 2], G0=[3], angles=[4, 5, 6],
                                      b_neq_B=[7], work=[8, 9, 10])
        compiled_resources(program, num_wires=11).toffoli  # U(0, 2, 1, 1) + 2 U(0, 2, 2, 1) = 5
    """

    compilable_argnames = ("sf_angles", "R", "B", "b_rot")
    wire_argnames = ("r", "b", "G0", "angles", "b_neq_B", "work")

    def __init__(self, sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
        sf_angles, R, B, b_rot = self._check(sf_angles, R, B, b_rot)
        check_wires("SFAngleLoad",
                    {"r": r, "b": b, "G0": G0, "angles": angles, "b_neq_B": b_neq_B,
                     "work": work},
                    self.required_wires(sf_angles, R, B, b_rot))
        super().__init__(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work)

    @staticmethod
    def _check(sf_angles, R, B, b_rot) -> tuple:
        b_rot = check_int("b_rot", b_rot, low=2)
        R, B = check_int("R", R, low=1), check_int("B", B, low=1)
        planes = canonical_ints("sf_angles", sf_angles)
        try:
            num_rungs = len(planes[0][0])
        except (IndexError, TypeError):
            raise ValueError(f"sf_angles must be a table of shape (R, B, N - 1) = ({R}, {B}, "
                             "N - 1)") from None
        if num_rungs < 1:
            raise ValueError("sf_angles needs N - 1 >= 1 angles per entry")
        return _check_angle_table("sf_angles", planes, (R, B, num_rungs), b_rot), R, B, b_rot

    @staticmethod
    def read_shape(N: int, R: int, B: int, b_rot: int) -> dict:
        """The read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``."""
        widths, ranges = _sf_address(R, B)
        return {"address_widths": widths, "live_ranges": ranges, "width": (N - 1) * b_rot + 1,
                "block_bits": 0}

    @staticmethod
    def table(sf_angles, B: int, b_rot: int) -> tuple:
        """The read's table: ``((r << w_b) | b, words)`` with the entry ``angles ‖ b_neq_B`` of
        ``(r, b)`` in 32-bit words, least significant first."""
        width = len(sf_angles[0][0]) * b_rot + 1
        return tuple(((r << lg(B + 1)) | b, _words(entry, width))
                     for r, plane in enumerate(_sf_entries(sf_angles, b_rot))
                     for b, entry in enumerate(plane))

    @staticmethod
    def required_wires(sf_angles, R, B, b_rot) -> dict[str, int]:
        """Exact register sizes."""
        num_modes = len(sf_angles[0][0]) + 1
        read = SFAngleLoad.read_shape(num_modes, R, B, b_rot)
        return {"r": lg(R), "b": lg(B + 1), "G0": 1, "angles": (num_modes - 1) * b_rot,
                "b_neq_B": 1, "work": QROAMRead.required_wires(**read, control_value=0)["work"]}


def _sf_angle_load_resources(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
    return {_read_key(SFAngleLoad.read_shape(len(sf_angles[0][0]) + 1, R, B, b_rot), 0): 1}


@qp.register_resources(_sf_angle_load_resources)
def _sf_angle_load(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
    read = SFAngleLoad.read_shape(len(sf_angles[0][0]) + 1, R, B, b_rot)
    words = QROAMRead.leaf_words_from_table(SFAngleLoad.table(sf_angles, B, b_rot), **read)
    QROAMRead(words, **read, control_value=0, address=_join(r, b),
              target=_join(angles, b_neq_B), junk=[], work=work, control=G0)


register_rule(SFAngleLoad, _sf_angle_load)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_angle_loads.py -q`
Expected: `38 passed` (11 to 21 s with 2 workers, depending on the machine's load; the two `MID_NAPHTHALENE` compiles took 1.1 and 1.3 s at 255 and 266 MB, each alone under `run_watched`, log `.scratch/plan-check/stages/2E/2e-mid-compiles.log`).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `692 passed` (656 + 36; see "Expected test counts").

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/angle_data.py tests/ops/test_angle_loads.py
git commit -m "Add TailAngleLoad and SFAngleLoad"
```

---

### Task 2E.2: `AngleErase`

**Task id:** `2E.2` · **Branch:** `phase2/2E.2` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 2E.1 (same module, `angle_data.py`)

**Files:**
- Modify: `src/dfthc/ops/angle_data.py` (replace it with the version below. It contains every line of Task 2E.1's version, in the same order, except the `dfthc.ops.qroam` import, which gains `PhaseFixup`, `QROAMEmitter` and `emit_phase_fixup`; it adds the import of `dfthc.ops.unary.as_resources` and the `AngleErase` section at the end; checked by `.scratch/plan-check/stages/2E/make_stage_2e1.py`)
- Create: `tests/ops/test_angle_erase.py`, `tests/ops/test_certificates_angles.py` (marked `slow`)

**Interfaces:**
- Consumes: Task 2E.1's `angle_data.py` (`TailAngleLoad`, `SFAngleLoad` and the helpers listed there); `dfthc.ops.qroam` (`PhaseFixup` with `address_order`, `high_ranges` and `required_wires`; `QROAMEmitter` with `measure_x`, `parities`, `cond_cz`, `cond_z`; `emit_phase_fixup`; Tasks 1C.4, 1C.5); `dfthc.ops.unary.as_resources`; `dfthc.compile` (`mcm_parities`, `pack_masks`, `packed_parities`, `compiled_resources`); `dfthc.costs` (`angle_erase`, `phase_fixup`, `tail_angle_load`, `sf_angle_load`); `ForcedPolicy`, `SparseSimulator`, `SparseState`, `nonlinear_conditionals`, `decompose_for_simulation` (Task 1B.2); `classify_rule_warning`, `verbose_program`, `chosen_rules`.
- Produces, in `dfthc.ops.angle_data`:
  - `AngleErase(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits, angles, b_neq_B, G0, G1, x_o, r, b, work)`: compilable the loads' tables and `n_plus`, `N >= 2`, `R, B, C >= 1`, `b_rot >= 2`, and `0 <= sf_onehot_bits <= w_b` with `2**sf_onehot_bits <= (N - 1) b_rot + 1` (the walk passes `w_b`); the four-step rule of spec 9.17;
  - static `required_wires(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits) -> dict` (`work`: the larger of the two fix-ups' needs), `sf_fixup_high_ranges(R, B, sf_onehot_bits)` and `tail_fixup_high_ranges(N, R, C)` (the `high_ranges` of `costs.phase_fixup` for the two fix-ups);
  - the rule `_angle_erase`. It is never adjointed or controlled.

**Acceptance tests** (32 tests in `tests/ops/test_angle_erase.py`, 5 of them `slow`; 7 `slow` tests in `tests/ops/test_certificates_angles.py`):
- T4: T and M against `costs.angle_erase` at both test presets (vendored tables; M = 7 and 15) and at `MID_NAPHTHALENE` (`slow`; T = 26, M = 138), A = T; three shapes off the walk's ob = `w_b` against the fix-ups' closed forms (the ob = 0 fallback, a split one-hot ob = 1, and R = 1); `costs.angle_erase` equals the two fix-ups' closed forms over `sf_fixup_high_ranges` and `tail_fixup_high_ranges` at the four presets and on 576 shapes (N 2 to 5, R 1 to 6, B 1 to 8, C 1 to 3); FeMoco-54 on the closed form, `Cost(102, 798)`, and `required_wires` at FeMoco (work 8).
- T6: M counts every erased qubit, T differs from re-running both loads, the R B conditional `CZ`s and N conditional `Z`s count their hinted expectations `ceil(R B / 2)` and `ceil(N / 2)` (ledger "K-10 consequences": no zero mask, so every covered address has a conditional), the verbose listing names `_angle_erase`; `qp.adjoint` and `qp.ctrl` of the erase fail with `CompileError` "Decomposition rule not found" after the expected measuring-rule warning (2 tests).
- Lightning smoke runs: loads then erase at `H2_STO3G_TEST` restore a phased state of `x_o`, `r`, `b` for seed 0 (per commit) and seeds 1 to 4 (`slow`).
- Forced-schedule round trips (3 tests, per commit): loads then erase on the sparse simulator in strict mode at the T2 shapes `sto3g` (ob = 2), `random-R3` (ob = 1, failed `x_o` and `r >= R`) and `random-R1` (ob = 0), from a random complex state of `x_o`, `r` and `b`, under the all-ones schedule and two seeded random ones; each consumes exactly M outcomes and restores the state exactly.
- `mcm_parities` on the erase's masks: the masks the rule's body passes to `mcm_parities` (recorded by a counting emitter, ob = `w_b`) equal the independently built ones at FeMoco-54 and `H2_631G_TEST`, every mask is nonzero, and the capture core at the FeMoco-54 shapes (270 SF masks over 796 outcomes, 54 tail masks over 798; random tables with zero angle entries; classical) and the tape implementation at `H2_631G_TEST` are GF(2)-linear (100 random pairs each).
- T5 at `level="device"`; T2 in tape mode at three sizes (ob = 2, 1, 0), the rule the only one registered under its name; `required_wires`; seven constructor violations, each a `ValueError`, a scalar `tail_angles` and SF entries without exactly `N - 1` angles among them.
- Certificates (spec 12.5; `slow`, 6 cases and a check of the case design, 3 to 7 s together): `sto3g` and `631g` (vendored tables, ob = `w_b`), `R3-zero-blocks` (a zero tail row and a zero SF entry), `R3-ob0` (the fallback), `R3-ob1` (split one-hot) and `R1` (SF address `b` alone). Each run is `TailAngleLoad`, `SFAngleLoad`, `AngleErase` on the sparse simulator in strict mode under the zero schedule and every unit schedule (M + 1 runs, M = (N - 1) b_rot + 3 = 7 to 15), from a seeded random complex amplitude on every value of `x_o`, `r` and `b`; it must consume exactly M forced outcomes and return exactly that state with every ancilla in `|0>`. Each test first asserts, on the decomposed tape it certifies, that it has conditionals, all `PauliZ` or `CZ`, and that `nonlinear_conditionals(operations)` is empty (the premise of spec 12.5). The suite includes partly covered SF fix-up leaves: every case with ob > 0 has a leaf holding both a covered `b < B` and the never covered `b = B` or a failed `b`, while `R3-ob0` exercises the fallback, which has no one-hot leaf; `test_certificate_cases_cover_the_branches` asserts this, the all-zero tail row and SF entry of `R3-zero-blocks`, and the branch types. Mutation check: `.scratch/plan-check/stages/2E/mutate_2e.py` (log `stages/2E/2e-mutants.log`); each of the six erase mutants (flags measured before the SF fix-up, the SF fix-up not enabled, the SF mask without `b_neq_B`, the tail mask without `G0`, outcomes in wire order, the tail fix-up one row short) fails all six certificates.

**Definition of done:** `32 passed` targeted (27 in the per-commit tier) and the certificates `7 passed` run targeted; the 2E.1 tests still pass; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_angle_erase.py`:

```python
"""AngleErase (spec 9.17): counts, rule selection, smoke runs, parity linearity, validation."""

import itertools
import math
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.utils.exceptions import CompileError

from dfthc import costs
from dfthc.compile import (
    FIXED_DECOMPS,
    GATE_SET,
    compiled_resources,
    mcm_parities,
    pack_masks,
    packed_parities,
)
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.angle_data import (
    AngleErase,
    SFAngleLoad,
    TailAngleLoad,
    _angle_erase,
    _erase_body,
)
from dfthc.ops.qroam import QROAMEmitter
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE, WalkParams
from dfthc.testing.resources import chosen_rules, classify_rule_warning, verbose_program
from dfthc.testing.sparse_sim import ForcedPolicy, SparseSimulator, SparseState
from dfthc.testing.tape import decompose_for_simulation


def random_tables(p, seed=3):
    """Random angle tables of p's shape, every angle below 2**(b_rot - 1)."""
    rng = np.random.default_rng(seed)
    bound = 2 ** (p.b_rot - 1)
    tail = tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.N))
    sf = tuple(tuple(tuple(int(v) for v in rng.integers(0, bound, p.N - 1)) for _ in range(p.B))
               for _ in range(p.R))
    return tail, sf


def register_value(angles, b_rot):
    """The angle registers' big-endian integer when register k holds angles[k] (spec 6.7), written
    out bit by bit, independently of the module's angle_entry."""
    return int("".join(format(angle, f"0{b_rot}b") for angle in angles), 2)


def vendored_tables(system, p):
    tables = WalkTables.from_factors(load_factors(system), p)
    return tables.tail_angles, tables.sf_angles


RANDOM_R3 = WalkParams(N=3, R=3, B=3, C=1, n_plus=1, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
RANDOM_R1 = WalkParams(N=2, R=1, B=2, C=2, n_plus=2, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
SHAPES = {
    "sto3g": (H2_STO3G_TEST, vendored_tables("h2_sto3g_R2_B2_C1", H2_STO3G_TEST)),
    "631g": (H2_631G_TEST, vendored_tables("h2_631g_R2_B2_C2", H2_631G_TEST)),
    "random-R3": (RANDOM_R3, random_tables(RANDOM_R3)),
    "random-R1": (RANDOM_R1, random_tables(RANDOM_R1)),
}


def layout(p, tail, sf, ob):
    """Wires: x_o, r, b, angles, G0, G1, b_neq_B, then work for the largest of the three."""
    sizes = {"x_o": p.w_x, "r": p.w_r, "b": p.w_b, "angles": (p.N - 1) * p.b_rot, "G0": 1,
             "G1": 1, "b_neq_B": 1}
    works = (TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot)["work"],
             SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot)["work"],
             AngleErase.required_wires(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot,
                                       ob)["work"])
    sizes["work"] = max(works)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start, works


def loads(p, tail, sf, wires, works):
    TailAngleLoad(tail, p.n_plus, p.RC, p.X_o, p.b_rot, wires["x_o"], wires["angles"],
                  wires["G0"], wires["G1"], wires["work"][:works[0]])
    SFAngleLoad(sf, p.R, p.B, p.b_rot, wires["r"], wires["b"], wires["G0"], wires["angles"],
                wires["b_neq_B"], wires["work"][:works[1]])


def make_erase(p, tail, sf, ob, wires, works):
    return AngleErase(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot, ob, wires["angles"],
                      wires["b_neq_B"], wires["G0"], wires["G1"], wires["x_o"], wires["r"],
                      wires["b"], wires["work"][:works[2]])


def expected_cost(p, ob):
    """The erase's closed form for any SF one-hot width ob: the SF fix-up (enabled) over
    sf_fixup_high_ranges plus the tail fix-up (no one-hot, no enable); M = (N - 1) b_rot + 3."""
    sf = costs.phase_fixup(AngleErase.sf_fixup_high_ranges(p.R, p.B, ob), ob, True)
    tail = costs.phase_fixup(AngleErase.tail_fixup_high_ranges(p.N, p.R, p.C), 0, False)
    return costs.Cost(sf.toffoli + tail.toffoli, (p.N - 1) * p.b_rot + 3)


def compiled_erase(p, tail, sf, ob):
    wires, num_wires, works = layout(p, tail, sf, ob)
    return compiled_resources(lambda: make_erase(p, tail, sf, ob, wires, works), num_wires)


def assert_counts(resources, expected):
    """T4 and the T6 signature: T and M exact, A = T (every AND of both fix-ups is uncomputed by
    measurement; the erase has no plain Toffoli)."""
    assert (resources.toffoli, resources.measurements) == (expected.toffoli,
                                                           expected.measurements)
    assert resources.measured_and_uncomputes == resources.toffoli


@pytest.mark.compiled
@pytest.mark.parametrize("name", ["sto3g", "631g"])
def test_counts_at_the_test_presets(name):
    """T4 at H2_STO3G_TEST and H2_631G_TEST (vendored tables, ob = w_b): dfthc.costs.angle_erase,
    M = (N - 1) b_rot + 3 (7 and 15)."""
    p, (tail, sf) = SHAPES[name]
    assert_counts(compiled_erase(p, tail, sf, p.w_b), costs.angle_erase(p))


@pytest.mark.compiled
@pytest.mark.slow
def test_counts_at_mid_naphthalene():
    """T4 at MID_NAPHTHALENE (naphthalene tables, nightly tier): T = 26, M = 138."""
    p = MID_NAPHTHALENE
    assert_counts(compiled_erase(p, *vendored_tables("naphthalene", p), p.w_b),
                  costs.angle_erase(p))


@pytest.mark.compiled
@pytest.mark.parametrize(("name", "ob"), [("random-R3", 0), ("random-R3", 1), ("random-R1", 2)],
                         ids=["R3-ob0", "R3-ob1", "R1-ob2"])
def test_counts_with_other_one_hot_widths(name, ob):
    """T4 off the walk's ob = w_b: the ob = 0 fallback (conditional Z per SF address), a split
    one-hot (ob = 1: r and the high bit of b iterated), and an SF address of b alone (R = 1)."""
    p, (tail, sf) = SHAPES[name]
    assert_counts(compiled_erase(p, tail, sf, ob), expected_cost(p, ob))


def test_fixup_high_ranges_agree_with_the_cost_model():
    """The erase's fix-up ranges are the oracle of dfthc.costs.angle_erase: at ob = w_b it equals
    the two fix-ups' closed forms at the presets, FeMoco-54 included, and on a grid of shapes with
    R from 1 to 6, B from 1 to 8, N from 2 to 5, C from 1 to 3."""
    assert AngleErase.sf_fixup_high_ranges(10, 27, 5) == [(0, 10, 4)]
    assert AngleErase.tail_fixup_high_ranges(54, 10, 27) == [(270, 324, 9)]
    grid = [WalkParams(N=n, R=r, B=b, C=c, n_plus=1, b_rot=4, b_k1=2, b_k2=2, s_u=3, k1=0, k2=0,
                       ke1=0, ke2=0)
            for n, r, b, c in itertools.product(range(2, 6), range(1, 7), range(1, 9), (1, 2, 3))]
    for p in [FEMOCO54, H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE, *grid]:
        assert costs.angle_erase(p) == expected_cost(p, p.w_b), p


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 on the closed forms: T = (11 + 30) + 61 = 102, M = 53 * 15 + 3 = 798; work is the
    tail fix-up's 8 wires (the SF fix-up needs 4)."""
    p = FEMOCO54
    assert costs.angle_erase(p) == costs.Cost(11 + 30 + 61, 798) == costs.Cost(102, 798)
    tail = tuple((0,) * 53 for _ in range(54))
    sf = tuple(tuple((0,) * 53 for _ in range(27)) for _ in range(10))
    assert AngleErase.required_wires(tail, sf, 27, 54, 10, 27, 27, 15, 5) == {
        "angles": 795, "b_neq_B": 1, "G0": 1, "G1": 1, "x_o": 9, "r": 4, "b": 5, "work": 8}


@pytest.mark.compiled
def test_erase_is_not_a_second_load_and_its_conditionals_are_hinted(capfd):
    """T6: the chosen rule is the pinned measured erase, M counts every erased qubit, its T
    differs from re-running both loads, and its conditionals count their hinted expectation:
    R B conditional CZs (one per SF address; no mask is zero) and N conditional Zs (one per tail
    row) give ceil(R B / 2) and ceil(N / 2)."""
    p, (tail, sf) = SHAPES["631g"]
    wires, num_wires, works = layout(p, tail, sf, p.w_b)
    resources = compiled_resources(lambda: make_erase(p, tail, sf, p.w_b, wires, works),
                                   num_wires)
    assert resources.measurements == len(wires["angles"]) + 3
    assert resources.toffoli != (costs.tail_angle_load(p) + costs.sf_angle_load(p)).toffoli
    assert resources.counts["CZ"] == math.ceil(p.R * p.B / 2)
    assert resources.counts["PauliZ"] == math.ceil(p.N / 2)
    qp.analyze(verbose_program(lambda: make_erase(p, tail, sf, p.w_b, wires, works), num_wires),
               level="user")()
    captured = capfd.readouterr()
    assert "_angle_erase" in chosen_rules(captured.out + captured.err)


@pytest.mark.compiled
@pytest.mark.parametrize("wrap", [qp.adjoint, lambda op: qp.ctrl(op, control=40)],
                         ids=["adjoint", "control"])
def test_erase_has_no_adjoint_or_controlled_path(wrap):
    """Spec 7.4 rule 5: never adjointed or controlled. Catalyst drops the measuring candidate
    with the expected warning (rule 12), and without another rule the compile fails loudly."""
    p, (tail, sf) = SHAPES["sto3g"]
    wires, _, works = layout(p, tail, sf, p.w_b)
    key = "Adjoint(AngleErase)" if wrap is qp.adjoint else "C(AngleErase)"
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(CompileError, match=re.escape(
                f"Decomposition rule not found for operator '{key}")):
            compiled_resources(lambda: wrap(make_erase(p, tail, sf, p.w_b, wires, works)),
                               num_wires=41)
    skipped = [str(r.message) for r in records if issubclass(r.category, RuleLoweringWarning)
               and str(r.message).startswith("Skipped the _angle_erase")]
    assert skipped and {classify_rule_warning(m) for m in skipped} == {"expected"}


def phased_prepare(wires):
    """H on each wire, S on even positions, CZ on consecutive pairs (non-uniform phases)."""
    for wire in wires:
        qp.H(wire)
    for position, wire in enumerate(wires):
        if position % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


SEEDS = [0, *(pytest.param(seed, marks=pytest.mark.slow) for seed in range(1, 5))]


@pytest.mark.compiled
@pytest.mark.parametrize("seed", SEEDS)
def test_lightning_smoke_run(seed):
    """Compiled loads then erase at H2_STO3G_TEST (15 wires) restore a phased state of x_o, r and
    b and clear every ancilla, for five measurement seeds (seeds 1 to 4 in the nightly tier;
    the certificates prove every outcome, this runs the compiled path)."""
    p, (tail, sf) = SHAPES["sto3g"]
    wires, num_wires, works = layout(p, tail, sf, p.w_b)
    inputs = wires["x_o"] + wires["r"] + wires["b"]

    def program(round_trip):
        @qp.qjit(capture=True, seed=seed)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            phased_prepare(inputs)
            if round_trip:
                loads(p, tail, sf, wires, works)
                make_erase(p, tail, sf, p.w_b, wires, works)
            return qp.state()

        return np.asarray(circuit())

    assert np.allclose(program(True), program(False), rtol=0, atol=1e-10)


def sorted_terms(state):
    order = np.lexsort(state.keys.T[::-1])
    return state.keys[order], state.amps[order]


@pytest.mark.parametrize(("name", "ob"), [("sto3g", 2), ("random-R3", 1), ("random-R1", 0)])
def test_round_trip_under_forced_schedules(name, ob):
    """Per-commit semantics (the nightly certificates prove every outcome): loads then erase on
    the sparse simulator in strict mode restore a random complex state of x_o, r and b exactly,
    every ancilla in |0>, under the all-ones schedule and two seeded random schedules. The shapes
    hold D1 and Q1 rows, b = B and a failed b (sto3g, random-R1), failed x_o and r >= R
    (random-R3), at ob = w_b, a split one-hot and the fallback."""
    p, (tail, sf) = SHAPES[name]
    wires, num_wires, works = layout(p, tail, sf, ob)

    def round_trip():
        loads(p, tail, sf, wires, works)
        make_erase(p, tail, sf, ob, wires, works)

    operations = decompose_for_simulation(round_trip).operations
    inputs = wires["x_o"] + wires["r"] + wires["b"]
    rng = np.random.default_rng(17)
    size = 2 ** len(inputs)
    keys = np.concatenate([SparseState.basis(num_wires, sum(
        ((v >> (len(inputs) - 1 - i)) & 1) << (num_wires - 1 - wire)
        for i, wire in enumerate(inputs))).keys for v in range(size)])
    amps = rng.normal(size=size) + 1j * rng.normal(size=size)
    initial = SparseState(num_wires, keys, amps / np.linalg.norm(amps))
    expected_keys, expected_amps = sorted_terms(initial)
    num_measured = (p.N - 1) * p.b_rot + 3
    for schedule in [[1] * num_measured, *rng.integers(0, 2, (2, num_measured)).tolist()]:
        simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
        keys, amps = sorted_terms(simulator.run(operations, initial_state=initial))
        assert simulator.policy.position == num_measured
        assert np.array_equal(keys, expected_keys), schedule
        assert np.allclose(amps, expected_amps, rtol=0, atol=1e-12), schedule


def erase_masks(p, tail, sf):
    """The two fix-ups' masks as the erase builds them: SF over angles ‖ b_neq_B, tail over
    angles ‖ b_neq_B ‖ G0 ‖ G1 (big-endian)."""
    sf_masks = [(register_value(sf[r][b], p.b_rot) << 1) | 1
                for r in range(p.R) for b in range(p.B)]
    tail_masks = [(register_value(row, p.b_rot) << 3) | 2 | int(j >= p.n_plus)
                  for j, row in enumerate(tail)]
    return [(sf_masks, (p.N - 1) * p.b_rot + 1), (tail_masks, (p.N - 1) * p.b_rot + 3)]


class MaskRecorder(QROAMEmitter):
    """A counting emitter that records the masks and the outcome count of every parities call."""

    def __init__(self):
        super().__init__(counting=True)
        self.calls = []

    def parities(self, outcomes, masks):
        self.calls.append((list(masks), len(outcomes)))
        return super().parities(outcomes, masks)


def rule_masks(p, tail, sf):
    """The masks the rule's body passes to mcm_parities at ob = w_b (classical, counting)."""
    recorder = MaskRecorder()
    sizes = AngleErase.required_wires(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot, p.w_b)
    starts = itertools.accumulate(sizes.values(), initial=0)
    wires = {name: range(start, start + n)
             for (name, n), start in zip(sizes.items(), starts, strict=False)}
    _erase_body(recorder, tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot, p.w_b, **wires)
    return recorder.calls


def test_no_mask_is_zero_and_parities_are_linear():
    """Spec 12.5's linearity for the erase's masks, which are never zero (the b_neq_B and G0
    bits): the masks the rule builds equal erase_masks, then the capture core at the FeMoco-54
    shapes (270 SF masks over 796 outcomes, 54 tail masks over 798, random tables with a zero
    angle entry in each; classical, nothing compiled) and the tape implementation at
    H2_631G_TEST, both with f(0) = 0 and f(x ^ y) = f(x) ^ f(y) for 100 random pairs."""
    p = FEMOCO54
    tail, sf = random_tables(p, seed=8)
    tail = ((0,) * 53,) + tail[1:]
    sf = (((0,) * 53,) + sf[0][1:],) + sf[1:]
    assert rule_masks(p, tail, sf) == erase_masks(p, tail, sf)
    assert rule_masks(H2_631G_TEST, *SHAPES["631g"][1]) == erase_masks(H2_631G_TEST,
                                                                         *SHAPES["631g"][1])
    rng = np.random.default_rng(12)
    for masks, num_outcomes in erase_masks(p, tail, sf):
        assert all(masks) and len(masks) in (270, 54)
        packed = pack_masks(masks, num_outcomes)
        assert not np.asarray(packed_parities(jnp.zeros(num_outcomes, dtype=jnp.uint32),
                                              packed)).any()
        for _ in range(100):
            x, y = rng.integers(0, 2, size=(2, num_outcomes))
            fx, fy = (np.asarray(packed_parities(v, packed)) for v in (x, y))
            assert np.array_equal(np.asarray(packed_parities(x ^ y, packed)), fx ^ fy)
    p, (tail, sf) = SHAPES["631g"]
    for masks, num_outcomes in erase_masks(p, tail, sf):
        with qp.queuing.AnnotatedQueue():
            outcomes = [qp.measure(w) for w in range(num_outcomes)]
            parities = mcm_parities(outcomes[::-1], masks)

        def evaluate(bits, outcomes=outcomes, parities=parities):
            record = {o.measurements[0]: int(b) for o, b in zip(outcomes, bits, strict=True)}
            return np.array([int(f.concretize(record)) for f in parities])

        assert not evaluate(np.zeros(num_outcomes, dtype=int)).any()
        for _ in range(100):
            x, y = rng.integers(0, 2, size=(2, num_outcomes))
            assert np.array_equal(evaluate(x ^ y), evaluate(x) ^ evaluate(y))


@pytest.mark.compiled
def test_no_dfthc_operator_survives_at_device_level():
    p, (tail, sf) = SHAPES["sto3g"]
    wires, num_wires, works = layout(p, tail, sf, p.w_b)
    compiled_resources(lambda: make_erase(p, tail, sf, p.w_b, wires, works), num_wires,
                       level="device")


@pytest.mark.parametrize(("name", "ob"), [("sto3g", 2), ("random-R3", 1), ("random-R1", 0)])
def test_rule_is_valid_in_tape_mode(name, ob):
    """T2 in tape mode at three sizes: the outcomes and their parities are MeasurementValues,
    and the counting twin declares exactly the emitted gates. The pinned rule is the only rule
    registered under the class name (spec 7.4 rule 8)."""
    assert list(qp.list_decomps(AngleErase)) == [_angle_erase]
    p, (tail, sf) = SHAPES[name]
    wires, _, works = layout(p, tail, sf, ob)
    qp.testing.assert_valid_decomp_rule(make_erase(p, tail, sf, ob, wires, works), _angle_erase,
                                        check_matrix=False)


def erase_args(**changes):
    """Valid AngleErase arguments (N = 2, R = B = 2, C = 1, b_rot = 3, ob = 2), with changes."""
    return {"tail_angles": ((1,), (2,)), "sf_angles": (((1,), (2,)), ((3,), (0,))), "n_plus": 1,
            "N": 2, "R": 2, "B": 2, "C": 1, "b_rot": 3, "sf_onehot_bits": 2,
            "angles": [0, 1, 2], "b_neq_B": [3], "G0": [4], "G1": [5], "x_o": [6, 7], "r": [8],
            "b": [9, 10], "work": [11]} | changes


def test_required_wires():
    args = erase_args()
    assert AngleErase.required_wires(*list(args.values())[:9]) == {
        "angles": 3, "b_neq_B": 1, "G0": 1, "G1": 1, "x_o": 2, "r": 1, "b": 2, "work": 1}
    AngleErase(**args)


@pytest.mark.parametrize(("change", "match"), [
    ({"sf_onehot_bits": 3}, r"sf_onehot_bits must lie in \[0, 2\]"),
    ({"b_rot": 2, "angles": [0, 1], "tail_angles": ((1,), (0,)),
      "sf_angles": (((1,), (0,)), ((0,), (1,)))}, "needs 4 wires, more than the 3 reset angle"),
    ({"N": 3}, "tail_angles must have N = 3 rows"),
    ({"tail_angles": 3}, "tail_angles must have N = 2 rows"),
    ({"sf_angles": (((1, 1), (2, 2)), ((3, 3), (0, 0)))}, "sf_angles N - 1 = 1 angles per entry"),
    ({"sf_angles": (((1,), (4,)), ((3,), (0,)))}, r"angles must lie in \[0, 2\*\*\(b_rot - 1\)"),
    ({"work": []}, "'work' has 0 wires; expected 1"),
])
def test_constructor_validates(change, match):
    with pytest.raises(ValueError, match=match):
        AngleErase(**erase_args(**change))
```

Create `tests/ops/test_certificates_angles.py`:

```python
"""Outcome certificates of spec 12.5 for TailAngleLoad, SFAngleLoad -> AngleErase (nightly tier).

The erase X-measures ``M = (N - 1) b_rot + 3`` qubits. The phase it leaves on a branch is
``(-1)^(xi . g)`` with ``g`` the branch's loaded content, which the load tests establish as a
function of ``(x_o, r, b)``, and its two fix-ups apply ``(-1)^f_xi`` with ``f`` from
``mcm_parities``, GF(2)-linear (test_angle_erase checks it on the erase's masks). So the residual
is linear in ``xi``, and exact restoration under the zero vector and every unit vector, M + 1
forced runs, certifies all 2**M outcomes. The conclusion also needs every conditional of the
certified tape to be diagonal with a GF(2)-linear predicate, which ``nonlinear_conditionals``
checks on the tape itself. Each run starts from a random complex amplitude on every value of
x_o, r and b (failed x_o, b = B, failed b and r >= R included) and must end in exactly that state
with every ancilla in |0>; strict mode also asserts that every erase measurement has
probability 1/2.

The suite includes partly covered SF fix-up leaves (a leaf is one value of r and the high bits
of b, with a one-hot over the low ob bits of b), as at FeMoco-54: every case with ob > 0 has a
leaf holding both a covered b < B and the never covered b = B or a failed b; the ob = 0 case
exercises the fallback, which has no one-hot leaf. A zero angle entry occurs in the STO-3G SF
table and is set in one tail row and one SF entry of the zero-block case; a zero mask cannot
occur, because every SF mask holds the b_neq_B bit and every tail mask the G0 bit (spec 9.17).
test_certificate_cases_cover_the_branches asserts these premises of the case design.
"""

import numpy as np
import pytest
from pennylane.ops import Conditional

from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.angle_data import AngleErase, SFAngleLoad, TailAngleLoad
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, WalkParams
from dfthc.testing.sparse_sim import (
    ForcedPolicy,
    SparseSimulator,
    SparseState,
    nonlinear_conditionals,
)
from dfthc.testing.tape import decompose_for_simulation

pytestmark = pytest.mark.slow


def random_tables(p, seed, zero_tail=(), zero_sf=()):
    """Random angle tables of p's shape (angles below 2**(b_rot - 1)), with zero entries at the
    tail rows ``zero_tail`` and the SF addresses ``zero_sf``."""
    rng = np.random.default_rng(seed)
    bound = 2 ** (p.b_rot - 1)
    tail = tuple(tuple(0 if j in zero_tail else int(v) for v in rng.integers(0, bound, p.N - 1))
                 for j in range(p.N))
    sf = tuple(tuple(tuple(0 if (r, b) in zero_sf else int(v)
                           for v in rng.integers(0, bound, p.N - 1)) for b in range(p.B))
               for r in range(p.R))
    return tail, sf


def vendored_tables(system, p):
    tables = WalkTables.from_factors(load_factors(system), p)
    return tables.tail_angles, tables.sf_angles


RANDOM_R3 = WalkParams(N=3, R=3, B=3, C=1, n_plus=1, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
RANDOM_R1 = WalkParams(N=2, R=1, B=2, C=2, n_plus=2, b_rot=3, b_k1=2, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
# (params, tables, SF one-hot bits): both test presets with their vendored tables (ob = w_b, the
# walk's); R = 3 (r = 3 unvisited, failed x_o = 6, 7) with a zero tail row and a zero SF entry, on
# the ob = 0 fallback and with a split one-hot (ob = 1: r and the high bit of b iterated); R = 1
# (the SF address is b alone).
CASES = {
    "sto3g": (H2_STO3G_TEST, vendored_tables("h2_sto3g_R2_B2_C1", H2_STO3G_TEST), 2),
    "631g": (H2_631G_TEST, vendored_tables("h2_631g_R2_B2_C2", H2_631G_TEST), 2),
    "R3-zero-blocks": (RANDOM_R3, random_tables(RANDOM_R3, 5, (1,), ((2, 0),)), 2),
    "R3-ob0": (RANDOM_R3, random_tables(RANDOM_R3, 6), 0),
    "R3-ob1": (RANDOM_R3, random_tables(RANDOM_R3, 7), 1),
    "R1": (RANDOM_R1, random_tables(RANDOM_R1, 9), 2),
}


def layout(p, tail, sf, ob):
    sizes = {"x_o": p.w_x, "r": p.w_r, "b": p.w_b, "angles": (p.N - 1) * p.b_rot, "G0": 1,
             "G1": 1, "b_neq_B": 1}
    works = (TailAngleLoad.required_wires(tail, p.n_plus, p.RC, p.X_o, p.b_rot)["work"],
             SFAngleLoad.required_wires(sf, p.R, p.B, p.b_rot)["work"],
             AngleErase.required_wires(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot,
                                       ob)["work"])
    sizes["work"] = max(works)
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start, works


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


@pytest.mark.parametrize("name", CASES)
def test_loads_erase_round_trip_certificate(name):
    p, (tail, sf), ob = CASES[name]
    wires, num_wires, works = layout(p, tail, sf, ob)

    def round_trip():
        TailAngleLoad(tail, p.n_plus, p.RC, p.X_o, p.b_rot, wires["x_o"], wires["angles"],
                      wires["G0"], wires["G1"], wires["work"][:works[0]])
        SFAngleLoad(sf, p.R, p.B, p.b_rot, wires["r"], wires["b"], wires["G0"], wires["angles"],
                    wires["b_neq_B"], wires["work"][:works[1]])
        AngleErase(tail, sf, p.n_plus, p.N, p.R, p.B, p.C, p.b_rot, ob, wires["angles"],
                   wires["b_neq_B"], wires["G0"], wires["G1"], wires["x_o"], wires["r"],
                   wires["b"], wires["work"][:works[2]])

    operations = decompose_for_simulation(round_trip).operations
    conditionals = [op for op in operations if isinstance(op, Conditional)]
    assert conditionals and {op.base.name for op in conditionals} <= {"PauliZ", "CZ"}
    assert not nonlinear_conditionals(operations)
    initial = random_input_state(num_wires, wires["x_o"] + wires["r"] + wires["b"])
    expected_keys, expected_amps = sorted_terms(initial)
    num_measured = (p.N - 1) * p.b_rot + 3
    schedules = [[0] * num_measured] + [
        [int(i == j) for j in range(num_measured)] for i in range(num_measured)]
    for schedule in schedules:
        simulator = SparseSimulator(num_wires, policy=ForcedPolicy(schedule), strict=True)
        keys, amps = sorted_terms(simulator.run(operations, initial_state=initial))
        assert simulator.policy.position == num_measured
        assert np.array_equal(keys, expected_keys), schedule
        assert np.allclose(amps, expected_amps, rtol=0, atol=1e-12), schedule


def test_certificate_cases_cover_the_branches():
    """The cases reach every branch type the erase must restore: tail rows of both families
    (D1 and Q1, except R1, which is all D1), SF rows with b < B and b = B, failed x_o and failed b
    where the registers allow them, and r >= R (R3). The zero-block case has an all-zero tail row
    and SF entry; every case with ob > 0 has a partly covered SF leaf (a value of b >> ob whose
    low positions hold both a covered b < B and an uncovered b >= B); R3-ob0 is the fallback."""
    for name, (p, _, _) in CASES.items():
        assert 0 < p.RC < p.X_o
        assert p.n_plus >= 1 and (p.n_plus < p.N or name == "R1")
    assert any(2**p.w_x > p.X_o for p, _, _ in CASES.values())
    assert any(2**p.w_b > p.B + 1 for p, _, _ in CASES.values())
    assert 2**RANDOM_R3.w_r > RANDOM_R3.R
    _, (tail, sf), _ = CASES["R3-zero-blocks"]
    assert any(not any(row) for row in tail)
    assert any(not any(entry) for plane in sf for entry in plane)
    for name, (p, _, ob) in CASES.items():
        if ob:
            leaves = {}
            for b in range(2**p.w_b):
                leaves.setdefault(b >> ob, set()).add(b < p.B)
            assert {True, False} in leaves.values(), name
    assert CASES["R3-ob0"][2] == 0
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_angle_erase.py tests/ops/test_certificates_angles.py -q`
Expected: FAIL at collection, `2 errors`: `ImportError: cannot import name 'AngleErase' from 'dfthc.ops.angle_data'`.

- [ ] **Step 3: Add the erase to `angle_data.py`**

Replace `src/dfthc/ops/angle_data.py` with:

```python
"""The angle registers of the inner select: loads and measured erase (spec 9.10, 9.11, 9.17).

**The shared angle register** (spec 6.7). ``angles`` is the concatenation of ``N - 1`` registers
of ``b_rot`` wires; register ``k`` drives rung ``k`` of the Givens ladder and holds the angle
integer ``x_k`` big-endian, with ``x_k < 2**(b_rot - 1)`` (spec 6.3). :func:`angle_entry` is the
big-endian integer of the whole register, register 0 most significant.

**Two exclusive loads.** :class:`TailAngleLoad` writes the one-body rows ``x_o`` in
``[RC, X_o)``: their angles, ``G0 = 1`` and ``G1 = [x_o - RC >= N+]``. :class:`SFAngleLoad` then
writes, where ``G0 = 0``, the angles of ``(r, b)`` in ``[0, R) x [0, B)`` and ``b_neq_B = 1``.
Each is a :class:`~dfthc.ops.qroam.QROAMRead` without a swap network (``block_bits = 0``) whose
entry is one big-endian integer over its target (``angles ‖ G0 ‖ G1``, ``angles ‖ b_neq_B``) and
whose ``leaf_words`` the rule builds from the compilable angle table (spec 9.3). After both
loads, tail rows hold tail data on ``G0 = 1``; every ``G0 = 0`` branch, failed ``x_o``
included, holds the SF entry of ``(r, b)`` where it is in range, and zeros elsewhere
(``b = B``, failed ``b``, unvisited ``r``). :class:`AngleErase` (Task 2E.2) erases what they
write.
"""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import AbstractArray, Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import canonical_ints, check_int, check_wires
from dfthc.ops.qroam import (
    WORD,
    WORD_DTYPE,
    PhaseFixup,
    QROAMEmitter,
    QROAMRead,
    emit_phase_fixup,
)
from dfthc.ops.unary import as_resources
from dfthc.params import lg


def angle_entry(angles, b_rot: int) -> int:
    """The big-endian integer of the angle registers holding ``angles``: register ``k`` holds
    ``angles[k]``, register 0 most significant (spec 6.7)."""
    value = 0
    for angle in angles:
        value = (value << b_rot) | angle
    return value


def _words(value: int, width: int) -> tuple[int, ...]:
    """``value`` as ``ceil(width / 32)`` words, least significant first (L-24)."""
    return tuple((value >> (WORD * w)) & (2**WORD - 1) for w in range(-(-width // WORD)))


def _check_angle_table(name: str, table, shape: tuple[int, ...], b_rot: int) -> tuple:
    """``table`` as nested tuples of Python ints of ``shape``, every angle in
    ``[0, 2**(b_rot - 1))`` (spec 6.3); ``ValueError`` otherwise."""
    table = canonical_ints(name, table)

    def check(value, dims):
        if not dims:
            if not isinstance(value, int):
                raise ValueError(f"{name} must be a table of ints of shape {shape}")
            if not 0 <= value < 2 ** (b_rot - 1):
                raise ValueError(f"{name} angles must lie in [0, 2**(b_rot - 1)) = "
                                 f"[0, {2 ** (b_rot - 1)}), got {value}")
            return
        if not isinstance(value, tuple) or len(value) != dims[0]:
            raise ValueError(f"{name} must be a table of ints of shape {shape}")
        for item in value:
            check(item, dims[1:])

    check(table, shape)
    return table


def _join(*registers) -> list:
    """The wires of ``registers``, concatenated in order."""
    return [wire for register in registers for wire in register]


def _read_key(read: dict, control_value):
    """The resource key of the loads' ``QROAMRead`` for the read shape ``read``."""
    words = AbstractArray(QROAMRead.leaf_words_shape(**read), WORD_DTYPE)
    sizes = QROAMRead.required_wires(**read, control_value=control_value)
    return resource_key(lambda: QROAMRead(words, **read, control_value=control_value,
                                          **{name: Wire[n] for name, n in sizes.items()}))


# --------------------------------------------------------------------------- TailAngleLoad


def _tail_entries(tail_angles, n_plus: int, b_rot: int) -> list[int]:
    """Entry of tail row ``j``: ``angles ‖ G0 ‖ G1`` big-endian, ``G0 = 1``, ``G1 = [j >= N+]``."""
    return [(angle_entry(row, b_rot) << 2) | 2 | int(j >= n_plus)
            for j, row in enumerate(tail_angles)]


class TailAngleLoad(Operator2):
    r"""Load the one-body rows' Givens angles and the family flags ``G0``, ``G1`` (step 10).

    **Action.** For ``x_o`` in ``[RC, X_o)``, with ``j = x_o - RC``: the angle registers get
    ``tail_angles[j]`` (XOR), ``G0`` gets 1 and ``G1`` gets :math:`[j \ge N_+]` (D1 rows first,
    then Q1; spec 6.1). Every other ``x_o`` (the SF rows ``x_o < RC`` and the failed
    ``x_o >= X_o``) is untouched. ``x_o`` is unchanged and no phase is added.

    **Registers.** With ``N = len(tail_angles)`` and :math:`w_x = \lg X_o`:

    ========== ======================= ======== =========================================
    name       width                   input    output
    ========== ======================= ======== =========================================
    ``x_o``    :math:`w_x`             any      unchanged
    ``angles`` :math:`(N - 1) b_{rot}` ``|0>``  ``tail_angles[x_o - RC]`` on tail rows
    ``G0``     1                       ``|0>``  :math:`[RC \le x_o < X_o]`
    ``G1``     1                       ``|0>``  :math:`[x_o - RC \ge N_+]` on tail rows
    ``work``   :math:`w_x - 1`         ``|0>``  ``|0>``
    ========== ======================= ======== =========================================

    **Compilable arguments.** ``tail_angles``: ``N x (N - 1)`` ints in
    ``[0, 2**(b_rot - 1))``, the ``N+`` D1 rows first (``WalkTables.tail_angles``), ``N >= 2``;
    ``n_plus`` (``0 <= N+ <= N``); ``RC`` (int >= 1); ``X_o`` (``RC + N``); ``b_rot``
    (int >= 2). ``b_rot`` sizes the angle registers; it is not in the master plan's interface
    but determines the structure (spec 7.4 rules 1 and 3).

    **Pre- and postconditions.** ``angles``, ``G0``, ``G1`` and ``work`` start in ``|0>``;
    ``work`` ends in ``|0>``. After both loads, :class:`AngleErase` erases the shared
    ``angles``, ``b_neq_B``, ``G0`` and ``G1`` registers.

    **Decomposition.** One :class:`~dfthc.ops.qroam.QROAMRead` of :meth:`table` with address
    ``x_o`` over ``[RC, X_o)``, no swap network (``block_bits = 0``), no enable, and target
    ``angles ‖ G0 ‖ G1`` (:meth:`read_shape`). The rule builds the read's ``leaf_words`` from
    the compilable ``tail_angles`` with ``QROAMRead.leaf_words_from_table`` (spec 9.3).

    **Adjoint and control.** Never adjointed or controlled (its read has loops; spec 7.4 rules
    4 and 6). :class:`AngleErase` undoes it by measurement.

    **Cost.** :func:`dfthc.costs.tail_angle_load`: :math:`T = U(RC, X_o, w_x, 0)` (``A = T``),
    ``M = 0``. The read's CNOT count is an upper bound (``exact=False`` in ``QROAMRead``). At
    FeMoco-54: 61, with 54 leaves of 25 words each.

    **Paper and dfthcbe.** Step 10, the D1/Q1 angle load (PRX Eq. (B41)); dfthcbe:
    ``tail_rotation_qrom`` (``dfthcbe/rot_data.py``), which prunes the SF prefix to aligned
    blocks (56 at FeMoco-54) where ours iterates exactly (spec 9.3, "Known deltas").

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.angle_data import TailAngleLoad

        tail = ((1,), (2,))  # N = 2, b_rot = 3
        program = lambda: TailAngleLoad(tail, 1, 2, 4, 3, x_o=[0, 1], angles=[2, 3, 4], G0=[5],
                                        G1=[6], work=[7])
        compiled_resources(program, num_wires=8).toffoli  # U(2, 4, 2, 0) = 1
    """

    compilable_argnames = ("tail_angles", "n_plus", "RC", "X_o", "b_rot")
    wire_argnames = ("x_o", "angles", "G0", "G1", "work")

    def __init__(self, tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
        tail_angles, n_plus, RC, X_o, b_rot = self._check(tail_angles, n_plus, RC, X_o, b_rot)
        check_wires("TailAngleLoad",
                    {"x_o": x_o, "angles": angles, "G0": G0, "G1": G1, "work": work},
                    self.required_wires(tail_angles, n_plus, RC, X_o, b_rot))
        super().__init__(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work)

    @staticmethod
    def _check(tail_angles, n_plus, RC, X_o, b_rot) -> tuple:
        b_rot = check_int("b_rot", b_rot, low=2)
        rows = canonical_ints("tail_angles", tail_angles)
        if not isinstance(rows, tuple):
            raise ValueError("tail_angles must be a table of ints of shape (N, N - 1)")
        num_rows = len(rows)
        if num_rows < 2:
            raise ValueError(f"tail_angles needs N >= 2 rows, got {num_rows}")
        tail_angles = _check_angle_table("tail_angles", rows, (num_rows, num_rows - 1), b_rot)
        n_plus = check_int("n_plus", n_plus, low=0, high=num_rows)
        RC = check_int("RC", RC, low=1)
        X_o = check_int("X_o", X_o, low=RC + num_rows, high=RC + num_rows)
        return tail_angles, n_plus, RC, X_o, b_rot

    @staticmethod
    def read_shape(N: int, RC: int, X_o: int, b_rot: int) -> dict:
        """The read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``."""
        return {"address_widths": (lg(X_o),), "live_ranges": ((RC, X_o),),
                "width": (N - 1) * b_rot + 2, "block_bits": 0}

    @staticmethod
    def table(tail_angles, n_plus: int, RC: int, b_rot: int) -> tuple:
        """The read's table: ``(RC + j, words)`` with the entry ``angles ‖ G0 ‖ G1`` of row ``j``
        in 32-bit words, least significant first."""
        width = (len(tail_angles) - 1) * b_rot + 2
        return tuple((RC + j, _words(entry, width))
                     for j, entry in enumerate(_tail_entries(tail_angles, n_plus, b_rot)))

    @staticmethod
    def required_wires(tail_angles, n_plus, RC, X_o, b_rot) -> dict[str, int]:
        """Exact register sizes."""
        read = TailAngleLoad.read_shape(len(tail_angles), RC, X_o, b_rot)
        return {"x_o": lg(X_o), "angles": (len(tail_angles) - 1) * b_rot, "G0": 1, "G1": 1,
                "work": QROAMRead.required_wires(**read, control_value=None)["work"]}


def _tail_angle_load_resources(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
    return {_read_key(TailAngleLoad.read_shape(len(tail_angles), RC, X_o, b_rot), None): 1}


@qp.register_resources(_tail_angle_load_resources)
def _tail_angle_load(tail_angles, n_plus, RC, X_o, b_rot, x_o, angles, G0, G1, work):
    read = TailAngleLoad.read_shape(len(tail_angles), RC, X_o, b_rot)
    words = QROAMRead.leaf_words_from_table(TailAngleLoad.table(tail_angles, n_plus, RC, b_rot),
                                            **read)
    QROAMRead(words, **read, control_value=None, address=x_o, target=_join(angles, G0, G1),
              junk=[], work=work, control=[])


register_rule(TailAngleLoad, _tail_angle_load)


# --------------------------------------------------------------------------- SFAngleLoad


def _sf_address(R: int, B: int) -> tuple[tuple, tuple]:
    """``(address_widths, ranges)`` of ``r ‖ b`` over ``[0, R) x [0, B)``; only ``b`` if
    ``R = 1`` (then ``r`` has no wires)."""
    if lg(R) == 0:
        return (lg(B + 1),), ((0, B),)
    return (lg(R), lg(B + 1)), ((0, R), (0, B))


def _sf_entries(sf_angles, b_rot: int) -> list[list[int]]:
    """Entry of ``(r, b)``: ``angles ‖ b_neq_B`` big-endian, ``b_neq_B = 1``."""
    return [[(angle_entry(entry, b_rot) << 1) | 1 for entry in plane] for plane in sf_angles]


class SFAngleLoad(Operator2):
    r"""Load the SF Givens angles and ``b_neq_B`` at ``(r, b)``, enabled on ``G0 = 0`` (step 11).

    **Action.** Where ``G0 = 0`` and ``(r, b)`` lies in ``[0, R) x [0, B)``: the angle registers
    get ``sf_angles[r][b]`` (XOR) and ``b_neq_B`` gets 1. Everywhere else (``G0 = 1``, the
    identity term ``b = B``, failed ``b > B``, ``r >= R``) nothing changes. ``r``, ``b`` and
    ``G0`` are unchanged and no phase is added. In the walk it follows :class:`TailAngleLoad`,
    which sets ``G0`` on the one-body rows, so the two loads never write the same branch.

    **Registers.** With :math:`w_r = \lg R` and :math:`w_b = \lg(B + 1)`:

    =========== ======================= ======== =========================================
    name        width                   input    output
    =========== ======================= ======== =========================================
    ``r``       :math:`w_r` (0 if R = 1) any     unchanged
    ``b``       :math:`w_b`             any      unchanged
    ``G0``      1                       any      unchanged (the enable, value 0)
    ``angles``  :math:`(N - 1) b_{rot}` ``|0>``  ``sf_angles[r][b]`` where enabled and in range
    ``b_neq_B`` 1                       ``|0>``  1 where enabled and in range
    ``work``    :math:`w_r + w_b`       ``|0>``  ``|0>``
    =========== ======================= ======== =========================================

    **Compilable arguments.** ``sf_angles``: ``R x B x (N - 1)`` ints in
    ``[0, 2**(b_rot - 1))`` (``WalkTables.sf_angles``), ``N >= 2``; ``R`` and ``B`` (ints >= 1,
    its first two sizes); ``b_rot`` (int >= 2), which sizes the angle registers (not in the
    master plan's interface; spec 7.4 rules 1 and 3).

    **Pre- and postconditions.** On enabled branches ``angles`` and ``b_neq_B`` start in
    ``|0>`` (they hold the tail load's data only where ``G0 = 1``); ``work`` starts and ends in
    ``|0>``. :class:`AngleErase` erases ``angles`` and ``b_neq_B``.

    **Decomposition.** One :class:`~dfthc.ops.qroam.QROAMRead` of :meth:`table` with address
    ``r ‖ b`` (``r`` outer over ``[0, R)``, ``b`` inner over ``[0, B)``; ``b`` alone if
    ``R = 1``), no swap network, the enable ``G0`` with control value 0, and target
    ``angles ‖ b_neq_B`` (:meth:`read_shape`). The rule builds the read's ``leaf_words`` from
    the compilable ``sf_angles`` (spec 9.3).

    **Adjoint and control.** Never adjointed or controlled (spec 7.4 rules 4 and 6).

    **Cost.** :func:`dfthc.costs.sf_angle_load`:
    :math:`T = U(0, R, w_r, 1) + R\,U(0, B, w_b, 1)` (``A = T``), ``M = 0``; the read's CNOT
    count is an upper bound. At FeMoco-54: :math:`11 + 10 \cdot 28 = 291`, with 270 leaves of
    25 words each.

    **Paper and dfthcbe.** Step 11, the SF angle load (PRX Table III, :math:`RB`); dfthcbe:
    ``sf_rotation_qrom`` (``dfthcbe/rot_data.py``), nested ``b`` outer and ``r`` inner with the
    identity row pruned (271 at FeMoco-54); ours nests ``r`` outer and iterates exactly.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.angle_data import SFAngleLoad

        sf = (((1,), (2,)), ((3,), (0,)))  # R = 2, B = 2, N = 2, b_rot = 3
        program = lambda: SFAngleLoad(sf, 2, 2, 3, r=[0], b=[1, 2], G0=[3], angles=[4, 5, 6],
                                      b_neq_B=[7], work=[8, 9, 10])
        compiled_resources(program, num_wires=11).toffoli  # U(0, 2, 1, 1) + 2 U(0, 2, 2, 1) = 5
    """

    compilable_argnames = ("sf_angles", "R", "B", "b_rot")
    wire_argnames = ("r", "b", "G0", "angles", "b_neq_B", "work")

    def __init__(self, sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
        sf_angles, R, B, b_rot = self._check(sf_angles, R, B, b_rot)
        check_wires("SFAngleLoad",
                    {"r": r, "b": b, "G0": G0, "angles": angles, "b_neq_B": b_neq_B,
                     "work": work},
                    self.required_wires(sf_angles, R, B, b_rot))
        super().__init__(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work)

    @staticmethod
    def _check(sf_angles, R, B, b_rot) -> tuple:
        b_rot = check_int("b_rot", b_rot, low=2)
        R, B = check_int("R", R, low=1), check_int("B", B, low=1)
        planes = canonical_ints("sf_angles", sf_angles)
        try:
            num_rungs = len(planes[0][0])
        except (IndexError, TypeError):
            raise ValueError(f"sf_angles must be a table of shape (R, B, N - 1) = ({R}, {B}, "
                             "N - 1)") from None
        if num_rungs < 1:
            raise ValueError("sf_angles needs N - 1 >= 1 angles per entry")
        return _check_angle_table("sf_angles", planes, (R, B, num_rungs), b_rot), R, B, b_rot

    @staticmethod
    def read_shape(N: int, R: int, B: int, b_rot: int) -> dict:
        """The read's ``address_widths``, ``live_ranges``, ``width`` and ``block_bits``."""
        widths, ranges = _sf_address(R, B)
        return {"address_widths": widths, "live_ranges": ranges, "width": (N - 1) * b_rot + 1,
                "block_bits": 0}

    @staticmethod
    def table(sf_angles, B: int, b_rot: int) -> tuple:
        """The read's table: ``((r << w_b) | b, words)`` with the entry ``angles ‖ b_neq_B`` of
        ``(r, b)`` in 32-bit words, least significant first."""
        width = len(sf_angles[0][0]) * b_rot + 1
        return tuple(((r << lg(B + 1)) | b, _words(entry, width))
                     for r, plane in enumerate(_sf_entries(sf_angles, b_rot))
                     for b, entry in enumerate(plane))

    @staticmethod
    def required_wires(sf_angles, R, B, b_rot) -> dict[str, int]:
        """Exact register sizes."""
        num_modes = len(sf_angles[0][0]) + 1
        read = SFAngleLoad.read_shape(num_modes, R, B, b_rot)
        return {"r": lg(R), "b": lg(B + 1), "G0": 1, "angles": (num_modes - 1) * b_rot,
                "b_neq_B": 1, "work": QROAMRead.required_wires(**read, control_value=0)["work"]}


def _sf_angle_load_resources(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
    return {_read_key(SFAngleLoad.read_shape(len(sf_angles[0][0]) + 1, R, B, b_rot), 0): 1}


@qp.register_resources(_sf_angle_load_resources)
def _sf_angle_load(sf_angles, R, B, b_rot, r, b, G0, angles, b_neq_B, work):
    read = SFAngleLoad.read_shape(len(sf_angles[0][0]) + 1, R, B, b_rot)
    words = QROAMRead.leaf_words_from_table(SFAngleLoad.table(sf_angles, B, b_rot), **read)
    QROAMRead(words, **read, control_value=0, address=_join(r, b),
              target=_join(angles, b_neq_B), junk=[], work=work, control=G0)


register_rule(SFAngleLoad, _sf_angle_load)


# --------------------------------------------------------------------------- AngleErase


def _sf_fixup(R: int, B: int, onehot_bits: int) -> dict:
    """The SF fix-up's address model: the SF load's live set, enabled on ``G0 = 0``."""
    widths, covered = _sf_address(R, B)
    return {"address_widths": widths, "covered": covered, "onehot_bits": onehot_bits,
            "control_value": 0}


def _tail_fixup(N: int, R: int, C: int) -> dict:
    """The tail fix-up's address model: ``x_o`` over ``[RC, X_o)``, no one-hot, no enable."""
    return {"address_widths": (lg(N + R * C),), "covered": ((R * C, N + R * C),),
            "onehot_bits": 0, "control_value": None}


def _dummy_wires(sizes: dict[str, int]) -> dict[str, range]:
    """Consecutive integer wires for every register: what the resource function counts on."""
    wires, start = {}, 0
    for name, size in sizes.items():
        wires[name] = range(start, start + size)
        start += size
    return wires


class AngleErase(Operator2):
    r"""Erase what the two angle loads wrote, by X measurement and two phase fix-ups (step 18).

    **Action.** Returns ``angles``, ``b_neq_B``, ``G0``, ``G1`` and ``work`` to
    :math:`|0\rangle` and leaves the state of ``x_o``, ``r`` and ``b`` exactly as it was before
    :class:`TailAngleLoad` and :class:`SFAngleLoad`, for every measurement outcome.

    **Registers.** With :math:`w_x = \lg X_o`, :math:`w_r = \lg R`, :math:`w_b = \lg(B+1)`:

    =========== ======================= ============== ==============
    name        width                   input          output
    =========== ======================= ============== ==============
    ``angles``  :math:`(N - 1) b_{rot}` the loads'     ``|0>``
    ``b_neq_B`` 1                       the SF load's  ``|0>``
    ``G0``      1                       the tail's     ``|0>``
    ``G1``      1                       the tail's     ``|0>``
    ``x_o``     :math:`w_x`             any            unchanged
    ``r``       :math:`w_r` (0 if R=1)  any            unchanged
    ``b``       :math:`w_b`             any            unchanged
    ``work``    :meth:`required_wires`  ``|0>``        ``|0>``
    =========== ======================= ============== ==============

    **Compilable arguments.** The loads' tables ``tail_angles`` (``N x (N - 1)``) and
    ``sf_angles`` (``R x B x (N - 1)``), ``n_plus``, ``N``, ``R``, ``B``, ``C`` (ints >= 1,
    ``N >= 2``), ``b_rot`` (int >= 2) and ``sf_onehot_bits`` (ob, ``0 <= ob <= w_b``; the walk
    uses :math:`w_b`, spec 9.17), whose one-hot register needs ``2**ob <= (N - 1) b_rot + 1``.

    **Pre- and postconditions.** ``angles``, ``b_neq_B``, ``G0`` and ``G1`` hold exactly what
    the two loads wrote for the current ``x_o``, ``r``, ``b`` (operations in between may act on
    them only in ways that preserve this; in ``InnerSelect`` they are controls). Afterwards
    they and ``work`` are :math:`|0\rangle`, with exact phases on every branch.

    **Decomposition** (spec 9.17; the order matters). With :math:`\xi_a` the outcomes of
    ``angles ‖ b_neq_B`` and :math:`\xi_g` those of ``G0 ‖ G1``:

    1. ``H`` and ``qp.measure(q, reset=True)`` on every qubit of ``angles ‖ b_neq_B``.
    2. SF fix-up: :func:`~dfthc.ops.qroam.emit_phase_fixup` over ``(r, b)`` in
       ``[0, R) x [0, B)``, enabled on ``G0 = 0``, with the parities
       :math:`f(r, b) = \xi_a \cdot (\mathrm{sf\_angles}[r][b] \Vert 1)` and a one-hot over the
       low ob bits of ``b`` on the reset ``angles ‖ b_neq_B`` wires.
    3. ``H`` and ``qp.measure(q, reset=True)`` on ``G0`` and ``G1``.
    4. Tail fix-up: ``emit_phase_fixup`` over ``x_o`` in ``[RC, X_o)``, no one-hot, no enable,
       with :math:`f(x_o) = \xi_a \cdot (\mathrm{tail\_angles}[x_o - RC] \Vert 0) \oplus
       \xi_g \cdot (1, [x_o - RC \ge N_+])`.

    The SF fix-up runs while ``G0`` still separates SF from tail branches: a tail branch's
    ``(r, b)`` may be anything, and its angle phase is paid by the tail fix-up. The parities
    come from :func:`dfthc.compile.mcm_parities` with masks built here from the compilable
    tables, through the loads' own entries (:meth:`TailAngleLoad.table`,
    :meth:`SFAngleLoad.table`); bit ``i`` of a big-endian mask belongs to the ``i``-th outcome
    from the end, so the outcomes are passed reversed. No mask is zero: every SF mask has the
    ``b_neq_B`` bit and every tail mask the ``G0`` bit. Every conditional is a hinted ``CZ`` (or
    ``Z`` for ob = 0, and on the tail) on a GF(2)-linear predicate (spec 12.5).

    **Adjoint and control.** Never: its rule measures (spec 7.4 rule 5).

    **Cost.** :func:`dfthc.costs.angle_erase` for ob = :math:`w_b`:
    :math:`T = U(0, R, w_r, 1) + O(w_b) + U(RC, X_o, w_x, 0)` (``A = T``),
    :math:`M = (N - 1) b_{rot} + 3`. For other ob, the SF term is
    ``costs.phase_fixup(sf_fixup_high_ranges(R, B, ob), ob, True)``. At FeMoco-54:
    :math:`T = (11 + 30) + 61 = 102`, :math:`M = 798`.

    **Paper and dfthcbe.** Step 18; the fix-ups of the paper's (B47) (the ``(b, r)`` table) and
    the :math:`N` term of (B50) (the D1/Q1 angles), PRX App. B, as dfthcbe's README attributes
    them. dfthcbe: ``RotDataErase`` (``dfthcbe/rot_data.py``, 90 at FeMoco-54), whose SF fix-up is
    not enabled (a ``Z`` on ``G0`` removes the tail branches' ``(0, 0)`` phase), whose iterations
    are truncated and pruned, and whose one-hot over ``b`` has ``B + 1`` positions where ours has
    ``2**w_b`` (FeMoco-54: 26 against 30).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.angle_data import AngleErase

        tail, sf = ((1,), (2,)), (((1,), (2,)), ((3,), (0,)))  # N = 2, R = B = 2, C = 1
        program = lambda: AngleErase(tail, sf, 1, 2, 2, 2, 1, 3, 2, angles=[0, 1, 2],
                                     b_neq_B=[3], G0=[4], G1=[5], x_o=[6, 7], r=[8], b=[9, 10],
                                     work=[11])
        compiled_resources(program, num_wires=12).measurements  # (N - 1) b_rot + 3 = 6
    """

    compilable_argnames = ("tail_angles", "sf_angles", "n_plus", "N", "R", "B", "C", "b_rot",
                           "sf_onehot_bits")
    wire_argnames = ("angles", "b_neq_B", "G0", "G1", "x_o", "r", "b", "work")

    def __init__(self, tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                 angles, b_neq_B, G0, G1, x_o, r, b, work):
        N = check_int("N", N, low=2)
        C = check_int("C", C, low=1)
        sf_angles, R, B, b_rot = SFAngleLoad._check(sf_angles, R, B, b_rot)
        rows = canonical_ints("tail_angles", tail_angles)
        if not isinstance(rows, tuple) or len(rows) != N or len(sf_angles[0][0]) != N - 1:
            raise ValueError(f"tail_angles must have N = {N} rows and sf_angles N - 1 = {N - 1} "
                             "angles per entry")
        tail_angles, n_plus, _, _, _ = TailAngleLoad._check(rows, n_plus, R * C, N + R * C,
                                                            b_rot)
        sf_onehot_bits = check_int("sf_onehot_bits", sf_onehot_bits, low=0, high=lg(B + 1))
        if sf_onehot_bits and 2**sf_onehot_bits > (N - 1) * b_rot + 1:
            raise ValueError(f"the SF one-hot over {sf_onehot_bits} bits needs "
                             f"{2**sf_onehot_bits} wires, more than the {(N - 1) * b_rot + 1} "
                             "reset angle and b_neq_B wires")
        check_wires("AngleErase",
                    {"angles": angles, "b_neq_B": b_neq_B, "G0": G0, "G1": G1, "x_o": x_o,
                     "r": r, "b": b, "work": work},
                    self.required_wires(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                                        sf_onehot_bits))
        super().__init__(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                         angles, b_neq_B, G0, G1, x_o, r, b, work)

    @staticmethod
    def required_wires(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                       sf_onehot_bits) -> dict[str, int]:
        """Exact register sizes; ``work`` is the larger of the two fix-ups' needs."""
        sf, tail = _sf_fixup(R, B, sf_onehot_bits), _tail_fixup(N, R, C)
        work = max(PhaseFixup.required_wires(**sf)["work"],
                   PhaseFixup.required_wires(**tail)["work"])
        return {"angles": (N - 1) * b_rot, "b_neq_B": 1, "G0": 1, "G1": 1,
                "x_o": lg(N + R * C), "r": lg(R), "b": lg(B + 1), "work": work}

    @staticmethod
    def sf_fixup_high_ranges(R, B, sf_onehot_bits) -> list[tuple[int, int, int]]:
        """The SF fix-up's iterated ranges: the ``high_ranges`` of
        :func:`dfthc.costs.phase_fixup` (controlled)."""
        sf = _sf_fixup(R, B, sf_onehot_bits)
        return PhaseFixup.high_ranges(sf["address_widths"], sf["covered"], sf_onehot_bits)

    @staticmethod
    def tail_fixup_high_ranges(N, R, C) -> list[tuple[int, int, int]]:
        """The tail fix-up's iterated range ``[(RC, X_o, w_x)]`` (uncontrolled, no one-hot)."""
        tail = _tail_fixup(N, R, C)
        return PhaseFixup.high_ranges(tail["address_widths"], tail["covered"], 0)


def _erase_body(emit, tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                angles, b_neq_B, G0, G1, x_o, r, b, work):
    data = _join(angles, b_neq_B)
    data_outcomes = [emit.measure_x(qubit) for qubit in data]
    sf, w_b = _sf_fixup(R, B, sf_onehot_bits), lg(B + 1)
    entries = _sf_entries(sf_angles, b_rot)
    masks = [entries[address >> w_b][address & (2**w_b - 1)]  # angles ‖ b_neq_B = 1
             for address in PhaseFixup.address_order(sf["address_widths"], sf["covered"])]
    sf_work = PhaseFixup.required_wires(**sf)["work"]
    emit_phase_fixup(emit.parities(data_outcomes[::-1], masks), address=_join(r, b),
                     address_widths=sf["address_widths"], covered=sf["covered"],
                     onehot=data[:2**sf_onehot_bits] if sf_onehot_bits else [],
                     onehot_bits=sf_onehot_bits, work=list(work)[:sf_work], control=G0[0],
                     control_value=0, emit=emit)
    flag_outcomes = [emit.measure_x(G0[0]), emit.measure_x(G1[0])]
    tail = _tail_fixup(N, R, C)
    masks = [((entry >> 2) << 3) | (entry & 3)  # angles ‖ b_neq_B = 0 ‖ G0 = 1 ‖ G1
             for entry in _tail_entries(tail_angles, n_plus, b_rot)]
    tail_work = PhaseFixup.required_wires(**tail)["work"]
    emit_phase_fixup(emit.parities((data_outcomes + flag_outcomes)[::-1], masks),
                     address=x_o, address_widths=tail["address_widths"],
                     covered=tail["covered"], onehot=[], onehot_bits=0,
                     work=list(work)[:tail_work], emit=emit)


def _angle_erase_resources(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                           angles, b_neq_B, G0, G1, x_o, r, b, work):
    counter = QROAMEmitter(counting=True)
    wires = _dummy_wires(AngleErase.required_wires(tail_angles, sf_angles, n_plus, N, R, B, C,
                                                   b_rot, sf_onehot_bits))
    _erase_body(counter, tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits,
                **wires)
    return as_resources(counter.counts)


@qp.register_resources(_angle_erase_resources)
def _angle_erase(tail_angles, sf_angles, n_plus, N, R, B, C, b_rot, sf_onehot_bits, angles,
                 b_neq_B, G0, G1, x_o, r, b, work):
    _erase_body(QROAMEmitter(), tail_angles, sf_angles, n_plus, N, R, B, C, b_rot,
                sf_onehot_bits, angles, b_neq_B, G0, G1, x_o, r, b, work)


register_rule(AngleErase, _angle_erase)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_angle_erase.py -q`
Expected: `32 passed` (18 to 32 s with 2 workers, including the five-seed smoke runs, 3.5 to 4 s each, and the `MID_NAPHTHALENE` compile, 1.4 s and 266 MB alone under `run_watched`).

Run: `scripts/test.sh tests/ops/test_certificates_angles.py -q`
Expected: `7 passed` (3 to 7 s; nightly tier, run targeted here).

Run: `scripts/test.sh tests/ops/test_angle_loads.py -q`
Expected: `38 passed`.

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `719 passed` (692 + 27).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/angle_data.py tests/ops/test_angle_erase.py tests/ops/test_certificates_angles.py
git commit -m "Add AngleErase and the angle load-erase outcome certificates"
```

---

### Task 2E.3: `SpinMux`, `SpinSwap`

**Task id:** `2E.3` · **Branch:** `phase2/2E.3` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** G1

**Files:**
- Create: `src/dfthc/ops/spin.py`, `tests/ops/test_spin.py`

**Interfaces:**
- Consumes: `dfthc.ops._validate.check_wires`; `dfthc.compile` (`register_rule`, `resource_key`, `compiled_resources`, `GATE_SET`, `FIXED_DECOMPS`); `dfthc.costs` (`spin_mux`, `spin_swap`); the presets of `dfthc.params`; `verbose_program`, `chosen_rules`.
- Produces, in `dfthc.ops.spin`:
  - `SpinMux(G0, sigma_0, sigma_1, sg)`: one wire each, `sg ^= sigma_1 if G0 else sigma_0`; `required_wires() -> dict`; `compute_matrix(G0, sigma_0, sigma_1, sg)`, the 16 x 16 permutation; the rules `_spin_mux` and `_adjoint_spin_mux` (pinned as `"Adjoint(SpinMux)"`: the operator itself);
  - `SpinSwap(sg, psi_up, psi_dn)`: `N >= 1` exact `CSWAP`s controlled by `sg`; `required_wires(N) -> dict`; `compute_matrix(sg, psi_up, psi_dn)`; the rules `_spin_swap` and `_adjoint_spin_swap` (pinned as `"Adjoint(SpinSwap)"`).

**Acceptance tests** (24 tests in `tests/ops/test_spin.py`):
- T3 on the compiled path, one lightning program per case with the basis input as a runtime argument: `SpinMux` on all 16 inputs, and its pinned adjoint; `SpinSwap` on every input at N = 2 and 3, and its adjoint at N = 2 (5 tests); each output is one basis state with amplitude exactly 1.
- `compute_matrix` equals the spec permutations (`SpinMux`; `SpinSwap` at N = 1, 2, 3), and both classes have a matrix, so T2's matrix check is not skipped.
- T2 with `check_matrix=True` for the base and the adjoint rule: `SpinMux` at two wire labelings, `SpinSwap` at N = 2 and 3 (4 tests), each rule the only one registered under its key.
- T4: `SpinMux` T = 1 (one plain `Toffoli`, A = 0) and its adjoint; `SpinSwap` T = N at both test presets, `MID_NAPHTHALENE` and FeMoco-54's width (54), and its adjoint (5 tests); FeMoco on the closed forms (1, 54; 110 per `InnerSelect`).
- T5 at `level="device"` and T6 (the verbose listing names `_spin_mux`, `_adjoint_spin_mux`, `_spin_swap`, `_adjoint_spin_swap`); `required_wires`; four constructor violations.

**Definition of done:** `24 passed`; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_spin.py`:

```python
"""SpinMux and SpinSwap (spec 9.12, 9.13): every basis input, matrices, counts."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.ops.spin import (
    SpinMux,
    SpinSwap,
    _adjoint_spin_mux,
    _adjoint_spin_swap,
    _spin_mux,
    _spin_swap,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.testing.resources import chosen_rules, verbose_program


def basis_outputs(num_wires, apply):
    """``basis -> output basis index`` from one compiled lightning program; asserts that each
    output is one basis state with amplitude exactly 1 (a permutation, no phase)."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit(basis):
        for wire in range(num_wires):
            qp.cond(((basis >> (num_wires - 1 - wire)) & 1) == 1, qp.X)(wire)
        apply()
        return qp.state()

    def output(basis):
        state = np.asarray(circuit(basis))
        index = int(np.argmax(np.abs(state)))
        assert np.allclose(state, np.eye(2**num_wires)[index], rtol=0, atol=1e-12), basis
        return index

    return output


def mux_image(index):
    """Spec 9.12 on |G0, sigma_0, sigma_1, sg>: sg ^= sigma_1 if G0 else sigma_0."""
    g0, s0, s1, sg = ((index >> (3 - i)) & 1 for i in range(4))
    return (g0 << 3) | (s0 << 2) | (s1 << 1) | (sg ^ (s1 if g0 else s0))


def swap_image(index, num_modes):
    """Spec 9.13 on |sg, psi_up, psi_dn>: the two N-bit registers swapped if sg."""
    sg = index >> (2 * num_modes)
    up, dn = (index >> num_modes) & (2**num_modes - 1), index & (2**num_modes - 1)
    return index if not sg else (sg << (2 * num_modes)) | (dn << num_modes) | up


def permutation(images):
    """The matrix with a 1 at (images[b], b)."""
    matrix = np.zeros((len(images), len(images)))
    matrix[images, range(len(images))] = 1
    return matrix


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["mux", "adjoint"])
def test_spin_mux_on_every_basis_input(adjoint):
    """T3 on the compiled path: on each of the 16 basis inputs, sg ^= sigma_{G0} and nothing else
    changes; the pinned adjoint is the same permutation."""

    def apply():
        op = SpinMux([0], [1], [2], [3])
        if adjoint:
            qp.adjoint(op)

    output = basis_outputs(4, apply)
    assert [output(b) for b in range(16)] == [mux_image(b) for b in range(16)]


@pytest.mark.compiled
@pytest.mark.parametrize(("num_modes", "adjoint"), [(2, False), (3, False), (2, True)],
                         ids=["N2", "N3", "N2-adjoint"])
def test_spin_swap_on_every_basis_input(num_modes, adjoint):
    """T3 on the compiled path: on every basis input, psi_up and psi_dn are swapped exactly
    when sg is 1; the pinned adjoint is the same permutation."""
    up = list(range(1, 1 + num_modes))
    dn = list(range(1 + num_modes, 1 + 2 * num_modes))

    def apply():
        op = SpinSwap([0], up, dn)
        if adjoint:
            qp.adjoint(op)

    output = basis_outputs(1 + 2 * num_modes, apply)
    size = 2 ** (1 + 2 * num_modes)
    assert [output(b) for b in range(size)] == [swap_image(b, num_modes) for b in range(size)]


def test_matrices_are_the_spec_permutations():
    """compute_matrix, which T2 compares with the decompositions, is the action of spec 9.12
    and 9.13."""
    assert SpinMux.has_matrix and SpinSwap.has_matrix
    assert np.array_equal(SpinMux.compute_matrix([0], [1], [2], [3]),
                          permutation([mux_image(b) for b in range(16)]))
    for num_modes in (1, 2, 3):
        size = 2 ** (1 + 2 * num_modes)
        assert np.array_equal(
            SpinSwap.compute_matrix([0], list(range(num_modes)), list(range(num_modes))),
            permutation([swap_image(b, num_modes) for b in range(size)]))


@pytest.mark.parametrize("wires", [[0, 1, 2, 3], [3, 0, 2, 1]], ids=["ordered", "permuted"])
def test_spin_mux_rules_are_valid_with_matrices(wires):
    """T2 with check_matrix=True for the base and the pinned adjoint rule (tape mode): the
    declared resources equal the emitted gates and the decomposition's matrix equals
    compute_matrix. SpinMux has one size; the second case relabels the wires. Each pinned rule is
    the only rule registered under its key (spec 7.4 rule 8)."""
    assert list(qp.list_decomps(SpinMux)) == [_spin_mux]
    assert list(qp.list_decomps("Adjoint(SpinMux)")) == [_adjoint_spin_mux]
    op = SpinMux(*([w] for w in wires))
    qp.testing.assert_valid_decomp_rule(op, _spin_mux, check_matrix=True)
    qp.testing.assert_valid_decomp_rule(qp.adjoint(op), _adjoint_spin_mux, check_matrix=True)


@pytest.mark.parametrize("num_modes", [2, 3])
def test_spin_swap_rules_are_valid_with_matrices(num_modes):
    """T2 with check_matrix=True for the base and the pinned adjoint rule at two sizes; each
    pinned rule is the only rule registered under its key (spec 7.4 rule 8)."""
    assert list(qp.list_decomps(SpinSwap)) == [_spin_swap]
    assert list(qp.list_decomps("Adjoint(SpinSwap)")) == [_adjoint_spin_swap]
    op = SpinSwap([0], list(range(1, 1 + num_modes)),
                  list(range(1 + num_modes, 1 + 2 * num_modes)))
    qp.testing.assert_valid_decomp_rule(op, _spin_swap, check_matrix=True)
    qp.testing.assert_valid_decomp_rule(qp.adjoint(op), _adjoint_spin_swap, check_matrix=True)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["mux", "adjoint"])
def test_spin_mux_toffoli_count(adjoint):
    """T4: T = 1, one plain Toffoli (A = 0), M = 0, for the operator and its pinned adjoint."""

    def program():
        op = SpinMux([0], [1], [2], [3])
        if adjoint:
            qp.adjoint(op)

    resources = compiled_resources(program, num_wires=4)
    assert (resources.toffoli, resources.measurements) == (costs.spin_mux().toffoli, 0)
    assert (resources.counts["Toffoli"], resources.measured_and_uncomputes) == (1, 0)


@pytest.mark.compiled
@pytest.mark.parametrize(("p", "adjoint"), [(H2_STO3G_TEST, False), (H2_631G_TEST, False),
                                            (MID_NAPHTHALENE, False), (FEMOCO54, False),
                                            (H2_631G_TEST, True)],
                         ids=["sto3g", "631g", "mid", "femoco-width", "631g-adjoint"])
def test_spin_swap_toffoli_count(p, adjoint):
    """T4 at both test presets, MID_NAPHTHALENE and FeMoco-54's width (table-free, so per
    commit): T = N plain Toffolis from the CSWAPs (A = 0), M = 0; the pinned adjoint the same."""
    up, dn = list(range(1, 1 + p.N)), list(range(1 + p.N, 1 + 2 * p.N))

    def program():
        op = SpinSwap([0], up, dn)
        if adjoint:
            qp.adjoint(op)

    resources = compiled_resources(program, num_wires=1 + 2 * p.N)
    assert (resources.toffoli, resources.measurements) == (costs.spin_swap(p).toffoli, 0)
    assert (resources.counts["Toffoli"], resources.measured_and_uncomputes) == (p.N, 0)


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 (spec 9.24): 1 per SpinMux, 54 per SpinSwap; 2 + 108 per InnerSelect."""
    assert costs.spin_mux() == costs.Cost(1)
    assert costs.spin_swap(FEMOCO54) == costs.Cost(54)
    assert 2 * costs.spin_mux() + 2 * costs.spin_swap(FEMOCO54) == costs.Cost(110)


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names the four pinned rules; T5 at level="device" (L-20)."""

    def program():
        SpinMux([0], [1], [2], [3])
        qp.adjoint(SpinMux([0], [1], [2], [3]))
        SpinSwap([3], [4, 5], [6, 7])
        qp.adjoint(SpinSwap([3], [4, 5], [6, 7]))

    qp.analyze(verbose_program(program, num_wires=8), level="user")()
    captured = capfd.readouterr()
    assert {"_spin_mux", "_adjoint_spin_mux", "_spin_swap", "_adjoint_spin_swap"} <= set(
        chosen_rules(captured.out + captured.err))
    compiled_resources(program, num_wires=8, level="device")


def test_required_wires():
    assert SpinMux.required_wires() == {"G0": 1, "sigma_0": 1, "sigma_1": 1, "sg": 1}
    assert SpinSwap.required_wires(54) == {"sg": 1, "psi_up": 54, "psi_dn": 54}


@pytest.mark.parametrize(("build", "match"), [
    (lambda: SpinMux([0, 4], [1], [2], [3]), "'G0' has 2 wires; expected 1"),
    (lambda: SpinMux([0], [1], [2], []), "'sg' has 0 wires; expected 1"),
    (lambda: SpinSwap([0], [1, 2], [3]), "'psi_dn' has 1 wires; expected 2"),
    (lambda: SpinSwap([0], [], []), "N >= 1"),
])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_spin.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.ops.spin'`.

- [ ] **Step 3: Write the spin operators**

Create `src/dfthc/ops/spin.py`:

```python
"""The spin multiplexer and the controlled spin swap of the inner select (spec 9.12 and 9.13).

Both are permutations of the computational basis and their own inverses. They define
``compute_matrix`` (small sizes only: ``SpinSwap`` has ``2**(2N + 1)`` rows), so their T2 tests
compare the decomposition's matrix with the operator's (``check_matrix=True``, spec 12.1).
"""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_wires


class SpinMux(Operator2):
    r"""``sg ^= sigma_{G0}``: the spin that controls the spin swap (steps 12 and 17).

    **Action.** :math:`\mathrm{sg} \mathrel{\oplus}= \sigma_1` if :math:`G_0 = 1`, else
    :math:`\sigma_0`; that is, :math:`\mathrm{sg} \mathrel{\oplus}= \sigma_0 \oplus
    G_0(\sigma_0 \oplus \sigma_1)`. ``G0``, ``sigma_0`` and ``sigma_1`` are unchanged. On SF
    branches (:math:`G_0 = 0`) the swap follows the inner spin :math:`\sigma_0`; on D1/Q1 branches
    the outer spin :math:`\sigma_1` (spec 6.1).

    **Registers.**

    =========== ===== ======= ======================================
    name        width input   output
    =========== ===== ======= ======================================
    ``G0``      1     any     unchanged
    ``sigma_0`` 1     any     unchanged
    ``sigma_1`` 1     any     unchanged
    ``sg``      1     any     :math:`\mathrm{sg} \oplus \sigma_{G_0}`
    =========== ===== ======= ======================================

    **Compilable arguments.** None.

    **Pre- and postconditions.** None beyond the action: it is a permutation of the basis. In
    the walk ``sg`` starts in ``|0>``, and the second application (step 17) returns it to
    ``|0>``.

    **Decomposition** (spec 9.12). ``CNOT(sigma_0, sg)``, ``CNOT(sigma_0, sigma_1)``,
    ``Toffoli(G0, sigma_1, sg)``, ``CNOT(sigma_0, sigma_1)``.

    **Adjoint and control.** Self-adjoint: ``Adjoint(SpinMux)`` is pinned to the operator
    itself (spec 7.5). Never controlled. The walk applies it twice instead of adjointing it.

    **Cost.** :func:`dfthc.costs.spin_mux`: :math:`T = 1`, one plain ``Toffoli`` (``A = 0``);
    ``M = 0``. Two per ``InnerSelect``.

    **Paper and dfthcbe.** Steps 12 and 17; the paper computes :math:`\sigma_{G_0}` in App. B.3
    but has no Table III row for it. dfthcbe: ``InnerSelect._build_sigma``, two
    ``MultiControlX`` gates (2 Toffolis per application).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.spin import SpinMux

        program = lambda: SpinMux(G0=[0], sigma_0=[1], sigma_1=[2], sg=[3])
        compiled_resources(program, num_wires=4).toffoli  # 1
    """

    wire_argnames = ("G0", "sigma_0", "sigma_1", "sg")

    def __init__(self, G0, sigma_0, sigma_1, sg):
        check_wires("SpinMux", {"G0": G0, "sigma_0": sigma_0, "sigma_1": sigma_1, "sg": sg},
                    self.required_wires())
        super().__init__(G0, sigma_0, sigma_1, sg)

    @staticmethod
    def required_wires() -> dict[str, int]:
        """Exact register sizes."""
        return {"G0": 1, "sigma_0": 1, "sigma_1": 1, "sg": 1}

    @staticmethod
    def compute_matrix(G0, sigma_0, sigma_1, sg) -> np.ndarray:
        """The 16 x 16 permutation matrix in the wire order ``G0, sigma_0, sigma_1, sg``."""
        matrix = np.zeros((16, 16))
        for index in range(16):
            g0, s0, s1 = ((index >> (3 - i)) & 1 for i in range(3))
            matrix[index ^ (s1 if g0 else s0), index] = 1
        return matrix


def _spin_mux_resources(G0, sigma_0, sigma_1, sg):
    return {qp.CNOT: 3, qp.Toffoli: 1}


@qp.register_resources(_spin_mux_resources)
def _spin_mux(G0, sigma_0, sigma_1, sg):
    qp.CNOT([sigma_0[0], sg[0]])
    qp.CNOT([sigma_0[0], sigma_1[0]])
    qp.Toffoli([G0[0], sigma_1[0], sg[0]])
    qp.CNOT([sigma_0[0], sigma_1[0]])


def _adjoint_spin_mux_resources(base):
    return {resource_key(lambda: SpinMux(Wire[1], Wire[1], Wire[1], Wire[1])): 1}


@qp.register_resources(_adjoint_spin_mux_resources)
def _adjoint_spin_mux(base):
    SpinMux(base.G0, base.sigma_0, base.sigma_1, base.sg)


register_rule(SpinMux, _spin_mux)
register_rule("Adjoint(SpinMux)", _adjoint_spin_mux)


class SpinSwap(Operator2):
    r"""Swap ``psi_up`` and ``psi_dn`` if ``sg`` is 1 (steps 13 and 17).

    **Action.** :math:`|s\rangle|u\rangle|d\rangle \mapsto |s\rangle|u\rangle|d\rangle` for
    :math:`s = 0` and :math:`|1\rangle|d\rangle|u\rangle` for :math:`s = 1`, with ``u`` and ``d``
    the ``N``-qubit contents of ``psi_up`` and ``psi_dn``. It routes the active spin register
    into ``psi_dn``, where the Givens ladder and the Majorana gate act.

    **Registers.** With ``N = len(psi_up)``:

    ========== ===== ======= =============================================
    name       width input   output
    ========== ===== ======= =============================================
    ``sg``     1     any     unchanged
    ``psi_up`` ``N`` any     ``psi_dn``'s content if ``sg``, else unchanged
    ``psi_dn`` ``N`` any     ``psi_up``'s content if ``sg``, else unchanged
    ========== ===== ======= =============================================

    **Compilable arguments.** None; ``N >= 1`` follows from the wires.

    **Pre- and postconditions.** None: it is a permutation of the basis.

    **Decomposition** (spec 9.13). ``N`` exact ``CSWAP`` gates ``(sg, psi_up[k], psi_dn[k])``.

    **Adjoint and control.** Self-adjoint: ``Adjoint(SpinSwap)`` is pinned to the operator
    itself (spec 7.5). Never controlled. The walk applies it twice instead of adjointing it.

    **Cost.** :func:`dfthc.costs.spin_swap`: :math:`T = N`, ``N`` plain ``Toffoli`` gates from
    the ``CSWAP`` (``A = 0``); ``M = 0``. Two per ``InnerSelect``; 54 at FeMoco-54.

    **Paper and dfthcbe.** Steps 13 and 17, the paper's :math:`U_{Swap}` (Table III); dfthcbe:
    ``InnerSelect.uswap``, a ``CSwap`` of ``N`` qubits.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.spin import SpinSwap

        program = lambda: SpinSwap(sg=[0], psi_up=[1, 2], psi_dn=[3, 4])
        compiled_resources(program, num_wires=5).toffoli  # 2
    """

    wire_argnames = ("sg", "psi_up", "psi_dn")

    def __init__(self, sg, psi_up, psi_dn):
        num_modes = len(psi_up)
        if num_modes < 1:
            raise ValueError("SpinSwap: psi_up needs N >= 1 wires, got 0")
        check_wires("SpinSwap", {"sg": sg, "psi_up": psi_up, "psi_dn": psi_dn},
                    self.required_wires(num_modes))
        super().__init__(sg, psi_up, psi_dn)

    @staticmethod
    def required_wires(N: int) -> dict[str, int]:
        """Exact register sizes for ``N`` modes per spin register."""
        return {"sg": 1, "psi_up": N, "psi_dn": N}

    @staticmethod
    def compute_matrix(sg, psi_up, psi_dn) -> np.ndarray:
        """The permutation matrix in the wire order ``sg, psi_up, psi_dn``."""
        num_modes = len(psi_up)
        size, mask = 2 ** (2 * num_modes + 1), 2**num_modes - 1
        matrix = np.zeros((size, size))
        for index in range(size):
            up, dn = (index >> num_modes) & mask, index & mask
            swapped = (index >> (2 * num_modes)) << (2 * num_modes) | (dn << num_modes) | up
            matrix[swapped if index >> (2 * num_modes) else index, index] = 1
        return matrix


def _spin_swap_resources(sg, psi_up, psi_dn):
    return {qp.CSWAP: len(psi_up)}


@qp.register_resources(_spin_swap_resources)
def _spin_swap(sg, psi_up, psi_dn):
    for up, dn in zip(list(psi_up), list(psi_dn), strict=True):
        qp.CSWAP([sg[0], up, dn])


def _adjoint_spin_swap_resources(base):
    num_modes = len(base.psi_up)
    return {resource_key(lambda: SpinSwap(Wire[1], Wire[num_modes], Wire[num_modes])): 1}


@qp.register_resources(_adjoint_spin_swap_resources)
def _adjoint_spin_swap(base):
    SpinSwap(base.sg, base.psi_up, base.psi_dn)


register_rule(SpinSwap, _spin_swap)
register_rule("Adjoint(SpinSwap)", _adjoint_spin_swap)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_spin.py -q`
Expected: `24 passed` (13 to 26 s with 2 workers).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `680 passed` (656 + 24).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/spin.py tests/ops/test_spin.py
git commit -m "Add SpinMux and SpinSwap with pinned self-adjoints"
```

---

### Task 2E.4: `GivensRotation`, `GivensLadder`, `GivensLadderDagger`

**Task id:** `2E.4` · **Branch:** `phase2/2E.4` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** G1

**Files:**
- Create: `src/dfthc/ops/givens.py`, `tests/ops/test_givens.py`

**Interfaces:**
- Consumes: `dfthc.ops._validate` (`check_int`, `check_wires`); `dfthc.compile` (`register_rule`, `resource_key`, `compiled_resources`, `GATE_SET`, `FIXED_DECOMPS`); `dfthc.costs` (`givens_rotation`, `givens_ladder`); `dfthc.data.angles` (`ladder_matrix`, `vector_from_angles`; Task 1A.4) and `dfthc.data.fermion.majorana` (Task 1A.2); the presets of `dfthc.params`; PennyLane's `SemiAdder` (its adjoint compiles, K-7) and `PhaseGradientStatePrep`; `verbose_program`, `chosen_rules`.
- Produces, in `dfthc.ops.givens`:
  - `GivensRotation(q_i, q_j, angle, phase_grad, work)`: no compilable arguments; `b_rot = len(angle) >= 2`; `required_wires(b_rot) -> {"q_i": 1, "q_j": 1, "angle": b_rot, "phase_grad": b_rot, "work": b_rot - 1}`; the rule `_givens_rotation` (dfthcbe's `GivensRotationByPhaseGradient`); adjoint auto-derived;
  - `GivensLadder(b_rot, system, angles, phase_grad, work)`: compilable `b_rot >= 2`, `N = len(system) >= 2`; rungs in time order `k = N - 2, ..., 0` on `(system[k], system[k + 1])` with angle register `angles[k b_rot:(k + 1) b_rot]`; `required_wires(b_rot, N) -> {"system": N, "angles": (N - 1) b_rot, "phase_grad": b_rot, "work": b_rot - 1}`; the rule `_givens_ladder`;
  - `GivensLadderDagger(b_rot, system, angles, phase_grad, work)`: the same arguments; rungs in time order `k = 0, ..., N - 2`, each `qp.adjoint(GivensRotation(...))`; `required_wires(b_rot, N)`; the rule `_givens_ladder_dagger`. Nothing adjoints `GivensLadder` itself.

**Acceptance tests** (36 tests in `tests/ops/test_givens.py`), dense actions from one compiled lightning program per case with the angle and the basis input as runtime arguments, each column projected onto `|x>|gradient>|0>` of the angle, phase-gradient and work registers and required to be a unit vector (no leakage):
- The rung's 4 x 4 action from all four basis inputs equals `[[cos t, -sin t], [sin t, cos t]]` on `{|01>, |10>}` and the identity on `|00>`, `|11>`, with `t = 4 pi x / 2**b_rot`, for every `x < 2**b_rot` at b_rot = 2, 3, 4 (the stored angles use `x < 2**(b_rot - 1)`), with no global phase (3 tests); the auto-derived adjoint is the transpose for every `x` at b_rot = 3, 4 (2 tests).
- The ladder equals `ladder_matrix(x, b_rot, N)` for N = 2, 3, 4 (b_rot = 4, 4, 3) and satisfies U† γ_{0,f} U = Σ_j u_j γ_{j,f} with `u = vector_from_angles(x, b_rot)` (spec 6.3, Majoranas from `dfthc.data.fermion`); for N >= 3 the opposite rung order fails both and the reversed register order differs, at the test's angles, and at N = 3 a flipped rung sign fails the identity too (3 tests). The dagger equals `ladder_matrix(x)^dag` for N = 2, 3, 4 (3 tests), and the ladder then the dagger restores a phased system state on lightning with the ancillas clean.
- T4: the rung T = 2 (b_rot - 1) and its adjoint at b_rot = 4 and 15 (4 tests); the ladder and the dagger T = 2 (b_rot - 1)(N - 1) at both test presets, `MID_NAPHTHALENE` and FeMoco-54's widths (8 tests; 1,484 at FeMoco, table-free, about 1.5 s each); A = T, M = 0; FeMoco on the closed forms (28, 1,484) and `required_wires` at FeMoco.
- T5 at `level="device"` and T6 (the verbose listing names `_givens_rotation`, `_givens_ladder`, `_givens_ladder_dagger`); T2 for the three rules at two sizes, each rule the only one registered under its name; `required_wires`; seven constructor violations, among them the dagger's own wire check.

**Definition of done:** `36 passed`; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_givens.py`:

```python
"""GivensRotation, GivensLadder and GivensLadderDagger (spec 9.14, 9.15): exact actions, the ladder
order, counts."""

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.angles import ladder_matrix, vector_from_angles
from dfthc.data.fermion import majorana
from dfthc.ops.givens import (
    GivensLadder,
    GivensLadderDagger,
    GivensRotation,
    _givens_ladder,
    _givens_ladder_dagger,
    _givens_rotation,
)
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, MID_NAPHTHALENE
from dfthc.testing.resources import chosen_rules, verbose_program


def gradient(b_rot):
    """PhaseGradientStatePrep's state on b_rot wires: sum_y exp(-2 pi i y / 2^b) |y> / sqrt(2^b)."""
    return np.exp(-2j * np.pi * np.arange(2**b_rot) / 2**b_rot) / np.sqrt(2**b_rot)


def layout(num_modes, b_rot):
    """Consecutive wires for the ladder's registers: system, angles, phase_grad, work."""
    wires, start = {}, 0
    for name, size in GivensLadder.required_wires(b_rot, num_modes).items():
        wires[name] = list(range(start, start + size))
        start += size
    return wires, start


def load_bits(value, wires):
    """X on every wire whose bit of the runtime integer ``value`` is 1 (big-endian)."""
    for position, wire in enumerate(wires):
        qp.cond(((value >> (len(wires) - 1 - position)) & 1) == 1, qp.X)(wire)


def angle_value(x, b_rot):
    """The angle registers' big-endian integer: register k (rung k) holds x[k]."""
    value = 0
    for x_k in x:
        value = (value << b_rot) | int(x_k)
    return value


def compiled_matrix(apply, num_modes, b_rot):
    """``x -> the matrix on the modes`` from one compiled lightning program. Each column runs one
    basis input with the angle registers holding ``x``, then projects the state onto
    ``|x>|gradient>|0>`` of the angle, phase-gradient and work registers; the columns are unit
    vectors exactly when nothing leaks into those registers (asserted)."""
    wires, num_wires = layout(num_modes, b_rot)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
    def circuit(value, basis):
        load_bits(basis, wires["system"])
        load_bits(value, wires["angles"])
        qp.PhaseGradientStatePrep(wires["phase_grad"])
        apply(**wires)
        return qp.state()

    def matrix(x):
        value = angle_value(x, b_rot)
        rest = np.kron(np.kron(np.eye(2 ** len(wires["angles"]))[value], gradient(b_rot)),
                       np.eye(2 ** len(wires["work"]))[0])
        columns = [np.asarray(circuit(value, basis)).reshape(2**num_modes, -1) @ rest.conj()
                   for basis in range(2**num_modes)]
        assert np.allclose([np.linalg.norm(c) for c in columns], 1, rtol=0, atol=1e-10)
        return np.column_stack(columns)

    return matrix


def rung(system, angles, phase_grad, work):
    GivensRotation(system[:1], system[1:], angles, phase_grad, work)


def adjoint_rung(system, angles, phase_grad, work):
    qp.adjoint(GivensRotation(system[:1], system[1:], angles, phase_grad, work))


def rung_matrix(x, b_rot):
    """Spec 6.3: [[cos t, -sin t], [sin t, cos t]] on {|01>, |10>}, identity on |00>, |11>."""
    c, s = np.cos(4 * np.pi * x / 2**b_rot), np.sin(4 * np.pi * x / 2**b_rot)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


@pytest.mark.compiled
@pytest.mark.parametrize("b_rot", [2, 3, 4])
def test_rung_matrix_for_every_angle(b_rot):
    """The dense 4 x 4 action on the mode qubits from all four basis inputs, the angle and
    phase-gradient registers projected out, for every x < 2**b_rot (the stored angles use
    x < 2**(b_rot - 1)): the sign of spec 6.3, no global phase and no leakage."""
    matrix = compiled_matrix(rung, 2, b_rot)
    for x in range(2**b_rot):
        assert np.allclose(matrix([x]), rung_matrix(x, b_rot), rtol=0, atol=1e-10), x
        assert np.allclose(rung_matrix(x, b_rot), ladder_matrix(np.array([x]), b_rot, 2), rtol=0,
                           atol=1e-12)


@pytest.mark.compiled
@pytest.mark.parametrize("b_rot", [3, 4])
def test_auto_derived_rung_adjoint_rotates_back(b_rot):
    """qp.adjoint(GivensRotation(...)), the leaf adjoint GivensLadderDagger uses, is the
    transposed rung (theta -> -theta) for every x."""
    matrix = compiled_matrix(adjoint_rung, 2, b_rot)
    for x in range(2**b_rot):
        assert np.allclose(matrix([x]), rung_matrix(x, b_rot).T, rtol=0, atol=1e-10), x


# (b_rot, integer angles below 2**(b_rot - 1)) per N. At b_rot = 4 the rungs turn by multiples
# of 45 degrees; N = 4 uses b_rot = 3 (90 degrees) to stay at 18 lightning wires. The angles are
# distinct, and for N >= 3 both the opposite rung order and the reversed register order give
# different unitaries (asserted below), so the test pins both. At N = 3, u has a nonzero odd
# component, so the gamma_0 identity also sees a flipped rung sign (u_j -> (-1)^j u_j).
LADDERS = {2: (4, [3]), 3: (4, [1, 3]), 4: (3, [1, 3, 2])}
B_ROT = 4


def single_register_majorana(j, flavour, num_modes):
    """gamma_{j, flavour} on one spin register: the psi_dn block of dfthc.data.fermion."""
    return majorana(j, flavour, 1, num_modes)[:2**num_modes, :2**num_modes]


def rotates_gamma_0_onto_u(matrix, x, b_rot, num_modes):
    """Spec 6.3: U^dag gamma_{0,f} U == sum_j u_j gamma_{j,f} with u = vector_from_angles."""
    u = vector_from_angles(x, b_rot)
    return all(np.allclose(
        matrix.conj().T @ single_register_majorana(0, f, num_modes) @ matrix,
        sum(u[j] * single_register_majorana(j, f, num_modes) for j in range(num_modes)),
        rtol=0, atol=1e-10) for f in (0, 1))


@pytest.mark.compiled
@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_ladder_equals_ladder_matrix_and_the_opposite_order_fails(num_modes):
    """The compiled ladder equals ladder_matrix (rung N - 2 first in time, rung 0 last) and
    rotates gamma_0 onto u = vector_from_angles(x). For N >= 3 the opposite time order (rung 0
    first) and the reversed register order (register k driving rung N - 2 - k) are different
    unitaries at these angles, so a ladder built either way fails this test; the opposite order
    also fails the gamma_0 identity there."""
    b_rot, x = LADDERS[num_modes]
    x = np.array(x)
    expected = ladder_matrix(x, b_rot, num_modes)
    matrix = compiled_matrix(
        lambda **wires: GivensLadder(b_rot, **wires), num_modes, b_rot)(x)
    assert np.allclose(matrix, expected, rtol=0, atol=1e-10)
    assert rotates_gamma_0_onto_u(matrix, x, b_rot, num_modes)
    rungs = [ladder_matrix(np.where(np.arange(num_modes - 1) == k, x, 0), b_rot, num_modes)
             for k in range(num_modes - 1)]
    opposite = np.eye(2**num_modes)
    for rung_k in rungs:
        opposite = rung_k @ opposite
    if num_modes == 2:
        assert np.allclose(opposite, expected, rtol=0, atol=1e-12)
    else:
        assert np.abs(opposite - expected).max() > 0.1
        assert not rotates_gamma_0_onto_u(opposite, x, b_rot, num_modes)
        assert np.abs(ladder_matrix(x[::-1], b_rot, num_modes) - expected).max() > 0.1


@pytest.mark.compiled
@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_dagger_is_the_inverse_of_ladder_matrix(num_modes):
    """GivensLadderDagger equals ladder_matrix(x)^dag densely."""
    b_rot, x = LADDERS[num_modes]
    x = np.array(x)
    matrix = compiled_matrix(
        lambda **wires: GivensLadderDagger(b_rot, **wires), num_modes, b_rot)(x)
    assert np.allclose(matrix, ladder_matrix(x, b_rot, num_modes).conj().T, rtol=0,
                       atol=1e-10)


def phased_prepare(wires):
    """H on each wire, S on even positions, CZ on consecutive pairs (non-uniform phases)."""
    for wire in wires:
        qp.H(wire)
    for position, wire in enumerate(wires):
        if position % 2 == 0:
            qp.S(wire)
    for first, second in zip(wires[:-1], wires[1:], strict=True):
        qp.CZ([first, second])


@pytest.mark.compiled
def test_dagger_after_the_ladder_restores_the_state():
    """On lightning: the ladder, then the dagger, on a phased system state with loaded angles
    returns the system state exactly, the angles unchanged and the ancillas clean."""
    num_modes, (_, x) = 3, LADDERS[3]
    wires, num_wires = layout(num_modes, B_ROT)

    def prepared():
        phased_prepare(wires["system"])
        load_bits(angle_value(x, B_ROT), wires["angles"])
        qp.PhaseGradientStatePrep(wires["phase_grad"])

    def program(round_trip):
        @qp.qjit(capture=True)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
        @qp.qnode(qp.device("lightning.qubit", wires=num_wires))
        def circuit():
            prepared()
            if round_trip:
                GivensLadder(B_ROT, **wires)
                GivensLadderDagger(B_ROT, **wires)
            return qp.state()

        return np.asarray(circuit())

    assert np.allclose(program(True), program(False), rtol=0, atol=1e-10)


def ladder_program(cls, num_modes, b_rot):
    wires, num_wires = layout(num_modes, b_rot)
    return (lambda: cls(b_rot, **wires)), num_wires


def rung_program(b_rot, adjoint):
    wires, num_wires = layout(2, b_rot)

    def program():
        (adjoint_rung if adjoint else rung)(**wires)

    return program, num_wires


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["rung", "adjoint"])
@pytest.mark.parametrize("p", [H2_STO3G_TEST, MID_NAPHTHALENE], ids=["b4", "b15"])
def test_rung_toffoli_count(p, adjoint):
    """T4: T = 2 (b_rot - 1) for the rung and its auto-derived adjoint; A = T, M = 0."""
    resources = compiled_resources(*rung_program(p.b_rot, adjoint))
    expected = costs.givens_rotation(p)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli


@pytest.mark.compiled
@pytest.mark.parametrize("cls", [GivensLadder, GivensLadderDagger], ids=["ladder", "dagger"])
@pytest.mark.parametrize("p", [H2_STO3G_TEST, H2_631G_TEST, MID_NAPHTHALENE, FEMOCO54],
                         ids=["sto3g", "631g", "mid", "femoco-width"])
def test_ladder_toffoli_count(p, cls):
    """T4 at both test presets, MID_NAPHTHALENE and FeMoco-54's widths (N = 54, b_rot = 15: a
    table-free 53-rung ladder on 878 wires, about 1.5 s, which master plan section 7 allows in any
    tier): T = 2 (b_rot - 1)(N - 1) for the ladder and the dagger, 1,484 at FeMoco-54; A = T,
    M = 0."""
    resources = compiled_resources(*ladder_program(cls, p.N, p.b_rot))
    expected = costs.givens_ladder(p)
    assert (resources.toffoli, resources.measurements) == (expected.toffoli, 0)
    assert resources.measured_and_uncomputes == expected.toffoli


def test_femoco_counts_on_the_closed_form():
    """FeMoco-54 on the closed forms: 28 per rung, 1,484 per ladder and per dagger (spec 9.24)."""
    assert costs.givens_rotation(FEMOCO54) == costs.Cost(28)
    assert costs.givens_ladder(FEMOCO54) == costs.Cost(2 * 14 * 53) == costs.Cost(1484)
    assert GivensLadder.required_wires(15, 54) == {"system": 54, "angles": 795,
                                                   "phase_grad": 15, "work": 14}


@pytest.mark.compiled
def test_pinned_rules_are_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names the three pinned rules; T5 at level="device" for the
    ladder, the dagger, a rung and its adjoint (L-20)."""
    wires, num_wires = layout(3, 3)

    def program():
        GivensLadder(3, **wires)
        GivensLadderDagger(3, **wires)
        rung(wires["system"][:2], wires["angles"][:3], wires["phase_grad"], wires["work"])
        adjoint_rung(wires["system"][:2], wires["angles"][:3], wires["phase_grad"],
                     wires["work"])

    qp.analyze(verbose_program(program, num_wires), level="user")()
    captured = capfd.readouterr()
    assert {"_givens_rotation", "_givens_ladder", "_givens_ladder_dagger"} <= set(
        chosen_rules(captured.out + captured.err))
    compiled_resources(program, num_wires=num_wires, level="device")


@pytest.mark.parametrize(("num_modes", "b_rot"), [(2, 2), (4, 4)])
def test_rules_are_valid(num_modes, b_rot):
    """T2 for the rung, the ladder and the dagger at two sizes (tape mode); each pinned rule is
    the only rule registered under its class name (spec 7.4 rule 8)."""
    for cls, rule in ((GivensRotation, _givens_rotation), (GivensLadder, _givens_ladder),
                      (GivensLadderDagger, _givens_ladder_dagger)):
        assert list(qp.list_decomps(cls)) == [rule]
    wires, _ = layout(num_modes, b_rot)
    qp.testing.assert_valid_decomp_rule(
        GivensRotation(wires["system"][:1], wires["system"][1:2], wires["angles"][:b_rot],
                       wires["phase_grad"], wires["work"]), _givens_rotation, check_matrix=False)
    qp.testing.assert_valid_decomp_rule(GivensLadder(b_rot, **wires), _givens_ladder,
                                        check_matrix=False)
    qp.testing.assert_valid_decomp_rule(GivensLadderDagger(b_rot, **wires),
                                        _givens_ladder_dagger, check_matrix=False)


def test_required_wires():
    assert GivensRotation.required_wires(4) == {"q_i": 1, "q_j": 1, "angle": 4, "phase_grad": 4,
                                                "work": 3}
    assert GivensLadder.required_wires(4, 2) == GivensLadderDagger.required_wires(4, 2) == {
        "system": 2, "angles": 4, "phase_grad": 4, "work": 3}


@pytest.mark.parametrize(("build", "match"), [
    (lambda: GivensRotation([0], [1], [2], [3], []), r"b_rot = len\(angle\) must lie in \[2,"),
    (lambda: GivensRotation([0], [1], [2, 3, 4], [5, 6], [7, 8]),
     "'phase_grad' has 2 wires; expected 3"),
    (lambda: GivensRotation([0, 9], [1], [2, 3], [5, 6], [7]), "'q_i' has 2 wires; expected 1"),
    (lambda: GivensLadder(3, [0], [], [1, 2, 3], [4, 5]), "system needs N >= 2"),
    (lambda: GivensLadder(3, [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]),
     "'angles' has 3 wires; expected 6"),
    (lambda: GivensLadderDagger(1, [0, 1], [2], [3], []), r"b_rot must lie in \[2,"),
    (lambda: GivensLadderDagger(2, [0, 1], [2, 3, 9], [4, 5], [6]),
     "'angles' has 3 wires; expected 2"),
])
def test_constructors_validate(build, match):
    with pytest.raises(ValueError, match=match):
        build()
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_givens.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.ops.givens'`.

- [ ] **Step 3: Write the Givens operators**

Create `src/dfthc/ops/givens.py`:

```python
"""Givens rotations by phase-gradient addition and the Givens ladders (spec 9.14 and 9.15).

**Rung.** On the mode qubits ``(q_k, q_{k+1})``, big-endian (spec 6.7), a rung fixes ``|00>``
and ``|11>`` and rotates ``{|01>, |10>}`` by ``[[cos t, -sin t], [sin t, cos t]]`` with
``t = 4 pi x / 2**b_rot``, where ``x`` is the integer in the rung's angle register (spec 6.3).
It is dfthcbe's ``GivensRotationByPhaseGradient``: a Clifford basis change around two ``Rz(t)``,
each an addition of ``x`` into the phase-gradient register or a subtraction, as the qubit says.

**Ladder.** ``GivensLadder`` applies ``U = G_0 G_1 ... G_{N-2}``: rung ``N - 2`` first in time
and rung 0 last, which realizes ``U^dag gamma_{0,f} U = sum_j u_j gamma_{j,f}`` for
``u = vector_from_angles(x, b_rot)``; :func:`dfthc.data.angles.ladder_matrix` is its dense matrix.
The opposite order fails for ``N >= 3``. ``GivensLadderDagger`` runs the inverse rungs in the
opposite time order. Nothing adjoints ``GivensLadder`` itself (spec 7.4 rule 6).
"""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_int, check_wires


class GivensRotation(Operator2):
    r"""One rung of the Givens ladder: a real rotation of ``{|01>, |10>}`` by the loaded angle.

    **Action.** On ``(q_i, q_j)``, with :math:`\theta = 4\pi x/2^{b_{rot}}` and ``x`` the
    big-endian integer in ``angle``: :math:`|00\rangle` and :math:`|11\rangle` are fixed,
    :math:`|01\rangle \mapsto \cos\theta|01\rangle + \sin\theta|10\rangle` and
    :math:`|10\rangle \mapsto -\sin\theta|01\rangle + \cos\theta|10\rangle`, exactly, with no
    global phase (spec 6.3). ``angle`` is unchanged; ``phase_grad`` is returned unchanged.

    **Registers.** With :math:`b_{rot}` = ``len(angle)``:

    ============== ========================= ====================== ====================
    name           width                     input                  output
    ============== ========================= ====================== ====================
    ``q_i``        1                         any                    rotated
    ``q_j``        1                         any                    rotated
    ``angle``      :math:`b_{rot}`           any (``x``)            unchanged
    ``phase_grad`` :math:`b_{rot}`           phase-gradient state   unchanged (catalysed)
    ``work``       :math:`b_{rot} - 1`       ``|0>``                ``|0>``
    ============== ========================= ====================== ====================

    **Compilable arguments.** None; the sizes follow from the wires. :math:`b_{rot} \ge 2`.

    **Pre- and postconditions.** ``phase_grad`` holds the state that ``PhaseGradientStatePrep``
    prepares on :math:`b_{rot}` wires; ``work`` starts and ends in ``|0>``. The stored angles of
    the walk have ``x < 2**(b_rot - 1)`` (spec 6.3), but the action holds for every ``x``.

    **Decomposition** (dfthcbe's ``GivensRotationByPhaseGradient``; spec 9.14), in time order,
    with :math:`\sqrt{X} = HSH` and :math:`\sqrt{X}^\dagger = HS^\dagger H`:

    1. :math:`\sqrt{X}^\dagger` on ``q_i``; ``S``, then :math:`\sqrt{X}^\dagger`, on ``q_j``;
    2. ``CNOT(q_i, q_j)``, then ``H`` on ``q_i``;
    3. :math:`R_z(\theta)` on ``q_i``, then on ``q_j``;
    4. ``H`` on ``q_i``, then ``CNOT(q_i, q_j)``;
    5. :math:`\sqrt{X}` on ``q_i``; :math:`\sqrt{X}`, then :math:`S^\dagger`, on ``q_j``.

    Each :math:`R_z(\theta)` on a qubit ``q`` is Qualtran's ``ControlledAddOrSubtract``:
    ``X(q)``, ``CNOT`` from ``q`` onto every ``phase_grad`` wire, ``SemiAdder(angle, phase_grad,
    work)``, the same CNOTs, ``X(q)``. It maps the gradient's value ``g`` to ``g + x`` for
    ``q = 1`` and to ``g - x`` for ``q = 0``, so ``q = 1`` picks up
    :math:`e^{+2\pi i x/2^{b_{rot}}}` and ``q = 0`` the conjugate (spec 6.5): exactly
    :math:`R_z(\theta)`.

    **Adjoint and control.** The adjoint is auto-derived (:math:`\theta \to -\theta`; the body
    has no loops or conditionals, and the adjoint ``SemiAdder`` subtracts, K-7):
    :class:`GivensLadderDagger` uses it. Never controlled.

    **Cost.** :func:`dfthc.costs.givens_rotation`: :math:`T = 2(b_{rot} - 1)`, two
    :math:`b_{rot}`-bit ``SemiAdder``\ s, all ``TemporaryAND`` with measured uncomputes
    (``A = T``); ``M = 0``. The adjoint costs the same. At FeMoco-54: 28, as dfthcbe's; the
    paper counts :math:`b_{rot}` Toffolis per :math:`R_z`, so 30 per rung (PRX Table III;
    ``rz-addsub`` in spec 13.1).

    **Paper and dfthcbe.** One rung of the paper's :math:`U_{Rot}` (PRX Eq. (B20)), steps 14 and
    16; dfthcbe's ``GivensRotationByPhaseGradient`` (``dfthcbe/givens.py``), whose two
    :math:`R_z` are Qualtran's ``RzViaPhaseGradient`` (user decision Q-5).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.givens import GivensRotation

        def program():
            qp.PhaseGradientStatePrep(wires=[5, 6, 7])
            GivensRotation(q_i=[0], q_j=[1], angle=[2, 3, 4], phase_grad=[5, 6, 7],
                           work=[8, 9])

        compiled_resources(program, num_wires=10).toffoli  # 2 (3 - 1) = 4
    """

    wire_argnames = ("q_i", "q_j", "angle", "phase_grad", "work")

    def __init__(self, q_i, q_j, angle, phase_grad, work):
        b_rot = check_int("b_rot = len(angle)", len(angle), low=2)
        check_wires("GivensRotation",
                    {"q_i": q_i, "q_j": q_j, "angle": angle, "phase_grad": phase_grad,
                     "work": work},
                    self.required_wires(b_rot))
        super().__init__(q_i, q_j, angle, phase_grad, work)

    @staticmethod
    def required_wires(b_rot: int) -> dict[str, int]:
        """Exact register sizes."""
        return {"q_i": 1, "q_j": 1, "angle": b_rot, "phase_grad": b_rot, "work": b_rot - 1}


def _sqrt_x(wire, adjoint: bool) -> None:
    qp.H(wire)
    if adjoint:
        qp.adjoint(qp.S(wire))
    else:
        qp.S(wire)
    qp.H(wire)


def _rz(qubit, angle, phase_grad, work) -> None:
    qp.X(qubit)
    for wire in phase_grad:
        qp.CNOT([qubit, wire])
    qp.SemiAdder(angle, phase_grad, work)
    for wire in phase_grad:
        qp.CNOT([qubit, wire])
    qp.X(qubit)


def _givens_rotation_resources(q_i, q_j, angle, phase_grad, work):
    b_rot = len(angle)
    return {
        qp.H: 10, qp.S: 3, resource_key(lambda: qp.adjoint(qp.S(Wire[1]))): 3,
        qp.CNOT: 4 * b_rot + 2, qp.X: 4,
        resource_key(lambda: qp.SemiAdder(Wire[b_rot], Wire[b_rot], Wire[b_rot - 1])): 2,
    }


@qp.register_resources(_givens_rotation_resources)
def _givens_rotation(q_i, q_j, angle, phase_grad, work):
    i, j = q_i[0], q_j[0]
    _sqrt_x(i, adjoint=True)
    qp.S(j)
    _sqrt_x(j, adjoint=True)
    qp.CNOT([i, j])
    qp.H(i)
    _rz(i, angle, phase_grad, work)
    _rz(j, angle, phase_grad, work)
    qp.H(i)
    qp.CNOT([i, j])
    _sqrt_x(i, adjoint=False)
    _sqrt_x(j, adjoint=False)
    qp.adjoint(qp.S(j))


register_rule(GivensRotation, _givens_rotation)


def _ladder_sizes(op: str, b_rot, system) -> tuple[int, int]:
    b_rot = check_int("b_rot", b_rot, low=2)
    num_modes = len(system)
    if num_modes < 2:
        raise ValueError(f"{op}: system needs N >= 2 mode wires, got {num_modes}")
    return b_rot, num_modes


def _ladder_required_wires(b_rot: int, num_modes: int) -> dict[str, int]:
    return {"system": num_modes, "angles": (num_modes - 1) * b_rot, "phase_grad": b_rot,
            "work": b_rot - 1}


def _rung_key(b_rot: int, adjoint: bool):
    def build():
        rung = GivensRotation(Wire[1], Wire[1], Wire[b_rot], Wire[b_rot], Wire[b_rot - 1])
        return qp.adjoint(rung) if adjoint else rung

    return resource_key(build)


def _rungs(b_rot, system, angles, phase_grad, work, order, adjoint: bool) -> None:
    """The rungs ``k`` of ``order`` on ``(system[k], system[k+1])`` with angle register ``k``.
    Registers are unpacked into lists once, so each rung is passed lists of wires (spec 9.0)."""
    system, angles = list(system), list(angles)
    phase_grad, work = list(phase_grad), list(work)
    for k in order:
        wires = ([system[k]], [system[k + 1]], angles[k * b_rot:(k + 1) * b_rot], phase_grad,
                 work)
        if adjoint:
            qp.adjoint(GivensRotation(*wires))
        else:
            GivensRotation(*wires)


class GivensLadder(Operator2):
    r"""The Givens ladder :math:`U(u)` on one spin register: rung ``N - 2`` first, rung 0 last.

    **Action.** :math:`U = G_0 G_1 \cdots G_{N-2}` on ``system``, with :math:`G_k` the
    :class:`GivensRotation` on modes ``(q_k, q_{k+1})`` driven by angle register ``k``,
    ``angles[k b_rot:(k + 1) b_rot]`` (spec 6.7). Then
    :math:`U^\dagger \gamma_{0,f} U = \sum_j u_j \gamma_{j,f}` for
    ``u = vector_from_angles(x, b_rot)`` (spec 6.3), and :math:`U` equals
    :func:`dfthc.data.angles.ladder_matrix` ``(x, b_rot, N)``. ``angles`` is unchanged.

    **Registers.** With ``N = len(system)``:

    ============== ========================= ====================== ====================
    name           width                     input                  output
    ============== ========================= ====================== ====================
    ``system``     ``N`` (at least 2)        any                    :math:`U` applied
    ``angles``     :math:`(N - 1) b_{rot}`   any                    unchanged
    ``phase_grad`` :math:`b_{rot}`           phase-gradient state   unchanged (catalysed)
    ``work``       :math:`b_{rot} - 1`       ``|0>``                ``|0>``
    ============== ========================= ====================== ====================

    **Compilable arguments.** ``b_rot`` (int, at least 2): the width of one angle register and
    of ``phase_grad``. ``N`` follows from ``system``.

    **Pre- and postconditions.** As for :class:`GivensRotation`.

    **Decomposition.** The :class:`GivensRotation` rungs in time order ``k = N - 2, ..., 0``,
    unrolled in Python, all sharing ``phase_grad`` and ``work``: the order of dfthcbe's
    ``GivensRotationLadder``, which realizes ``angles_from_vector``. The opposite time order
    gives a different unitary for ``N >= 3``.

    **Adjoint and control.** Never adjointed or controlled. The inverse is the explicit class
    :class:`GivensLadderDagger` (spec 7.4 rule 6); nothing applies ``qp.adjoint`` to this
    operator.

    **Cost.** :func:`dfthc.costs.givens_ladder`: :math:`T = 2(b_{rot} - 1)(N - 1)`, ``A = T``,
    ``M = 0``. At FeMoco-54: 1,484, as dfthcbe's; the paper's Table III gives 1,590
    (:math:`2 b_{rot}` per rung, ``rz-addsub``).

    **Paper and dfthcbe.** The paper's :math:`U_{Rot}` (PRX Eq. (B20)), step 14; dfthcbe's
    ``GivensRotationLadder`` (``dfthcbe/rotations.py``).

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.givens import GivensLadder

        def program():
            qp.PhaseGradientStatePrep(wires=[9, 10, 11])
            GivensLadder(3, system=[0, 1, 2], angles=[3, 4, 5, 6, 7, 8],
                         phase_grad=[9, 10, 11], work=[12, 13])

        compiled_resources(program, num_wires=14).toffoli  # 2 (3 - 1)(3 - 1) = 8
    """

    compilable_argnames = ("b_rot",)
    wire_argnames = ("system", "angles", "phase_grad", "work")

    def __init__(self, b_rot, system, angles, phase_grad, work):
        b_rot, num_modes = _ladder_sizes("GivensLadder", b_rot, system)
        check_wires("GivensLadder",
                    {"system": system, "angles": angles, "phase_grad": phase_grad,
                     "work": work},
                    self.required_wires(b_rot, num_modes))
        super().__init__(b_rot, system, angles, phase_grad, work)

    @staticmethod
    def required_wires(b_rot: int, N: int) -> dict[str, int]:
        """Exact register sizes for ``N`` modes."""
        return _ladder_required_wires(b_rot, N)


def _givens_ladder_resources(b_rot, system, angles, phase_grad, work):
    return {_rung_key(b_rot, adjoint=False): len(system) - 1}


@qp.register_resources(_givens_ladder_resources)
def _givens_ladder(b_rot, system, angles, phase_grad, work):
    _rungs(b_rot, system, angles, phase_grad, work, reversed(range(len(system) - 1)),
           adjoint=False)


register_rule(GivensLadder, _givens_ladder)


class GivensLadderDagger(Operator2):
    r"""The inverse of :class:`GivensLadder`: :math:`U^\dagger` on the same registers.

    **Action.** :math:`U^\dagger = G_{N-2}^\dagger \cdots G_1^\dagger G_0^\dagger` on
    ``system``, with :math:`U` the :class:`GivensLadder` of the same arguments: applied after
    the ladder it gives the identity on every input. ``angles`` is unchanged.

    **Registers.** Those of :class:`GivensLadder`:

    ============== ========================= ====================== ====================
    name           width                     input                  output
    ============== ========================= ====================== ====================
    ``system``     ``N`` (at least 2)        any                    :math:`U^\dagger`
                                                                    applied
    ``angles``     :math:`(N - 1) b_{rot}`   any                    unchanged
    ``phase_grad`` :math:`b_{rot}`           phase-gradient state   unchanged (catalysed)
    ``work``       :math:`b_{rot} - 1`       ``|0>``                ``|0>``
    ============== ========================= ====================== ====================

    **Compilable arguments.** ``b_rot``, as for :class:`GivensLadder`.

    **Pre- and postconditions.** As for :class:`GivensLadder`.

    **Decomposition.** The rungs in time order ``k = 0, ..., N - 2`` (the ladder's order
    reversed), each ``qp.adjoint(GivensRotation(...))``, the auto-derived leaf adjoint
    (:math:`\theta \to -\theta`), unrolled in Python (spec 9.15).

    **Adjoint and control.** It is the explicit adjoint class of :class:`GivensLadder` (spec 7.4
    rule 6); it is never adjointed or controlled itself.

    **Cost.** :func:`dfthc.costs.givens_ladder`, the same as the ladder:
    :math:`T = 2(b_{rot} - 1)(N - 1)`, ``A = T``, ``M = 0``. At FeMoco-54: 1,484.

    **Paper and dfthcbe.** :math:`U_{Rot}^\dagger`, step 16; dfthcbe's
    ``GivensRotationLadder(...).adjoint()``.

    **Example.**

    .. code-block:: python

        import pennylane as qp
        from dfthc.compile import compiled_resources
        from dfthc.ops.givens import GivensLadderDagger

        def program():
            qp.PhaseGradientStatePrep(wires=[9, 10, 11])
            GivensLadderDagger(3, system=[0, 1, 2], angles=[3, 4, 5, 6, 7, 8],
                               phase_grad=[9, 10, 11], work=[12, 13])

        compiled_resources(program, num_wires=14).toffoli  # 8
    """

    compilable_argnames = ("b_rot",)
    wire_argnames = ("system", "angles", "phase_grad", "work")

    def __init__(self, b_rot, system, angles, phase_grad, work):
        b_rot, num_modes = _ladder_sizes("GivensLadderDagger", b_rot, system)
        check_wires("GivensLadderDagger",
                    {"system": system, "angles": angles, "phase_grad": phase_grad,
                     "work": work},
                    self.required_wires(b_rot, num_modes))
        super().__init__(b_rot, system, angles, phase_grad, work)

    @staticmethod
    def required_wires(b_rot: int, N: int) -> dict[str, int]:
        """Exact register sizes for ``N`` modes (those of :class:`GivensLadder`)."""
        return _ladder_required_wires(b_rot, N)


def _givens_ladder_dagger_resources(b_rot, system, angles, phase_grad, work):
    return {_rung_key(b_rot, adjoint=True): len(system) - 1}


@qp.register_resources(_givens_ladder_dagger_resources)
def _givens_ladder_dagger(b_rot, system, angles, phase_grad, work):
    _rungs(b_rot, system, angles, phase_grad, work, range(len(system) - 1), adjoint=True)


register_rule(GivensLadderDagger, _givens_ladder_dagger)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_givens.py -q`
Expected: `36 passed` (29 to 50 s with 2 workers; the slowest test, the verbose and device-level compile, took 4.90 s in the durations record, `.scratch/plan-check/stages/2E/2e-durations.log`; the FeMoco-width ladder compiled in 1.6 s at 276 MB alone under the watchdog).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `692 passed` (656 + 36).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/givens.py tests/ops/test_givens.py
git commit -m "Add GivensRotation, GivensLadder and GivensLadderDagger"
```

---

### Task 2E.5: `MajoranaGate`

**Task id:** `2E.5` · **Branch:** `phase2/2E.5` · **Tier:** A (`inherit`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** G1

**Files:**
- Create: `src/dfthc/ops/majorana.py`, `tests/ops/test_majorana.py`

**Interfaces:**
- Consumes: `dfthc.ops._validate.check_wires`; `dfthc.compile` (`register_rule`, `resource_key`, `compiled_resources`, `GATE_SET`, `FIXED_DECOMPS`); `dfthc.costs.majorana_gate`; `dfthc.data.fermion` (`majorana`, `annihilation`; Task 1A.2); `verbose_program`, `chosen_rules`.
- Produces, in `dfthc.ops.majorana`:
  - `MajoranaGate(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work)`: one wire each and `work` of 2; no compilable arguments; with `a = succ_x_o AND G0` and `s = succ_x_o AND NOT G0 AND succ_b`, in order: `-Z` on `target` if `s AND b_neq_B`; `X` on `target` if `a`; `Z` on `target` if `a AND flavour`; `Z` on `flavour` if `a AND G1` (spec 9.16). In the walk `flavour = sigma_0` and `target = psi_dn[0]`;
  - `required_wires() -> dict`; the rule `_majorana_gate`; adjoint auto-derived (`InnerSelectDagger`'s `Maj^dag`, Task 3.1).

**Acceptance tests** (16 tests in `tests/ops/test_majorana.py`):
- T3: the compiled 128 x 128 matrix of the gate (one lightning run, each wire entangled with a reference wire, the work wires required back in `|00>`) equals spec 9.16's four steps on every basis input, and the auto-derived adjoint equals its transpose (2 tests).
- The same on the tape path (spec 7.4 rule 9) on the sparse simulator, gate and adjoint, whose handlers assert that every `TemporaryAND` target is 0 and every `Adjoint(TemporaryAND)` target equals its AND: lightning runs both forms as a `Toffoli`, so only this test sees a compute and a measured uncompute exchanged (2 tests).
- The flavour convention (spec 6.2): with `flavour` projected onto `|+>` before and after, on each of the 32 control branches the compiled gate acts on `target` as `i gamma_0 gamma_1 = -Z` (valid SF, `b != B`), `a = (X + iY)/2` (valid D1), `a^dag` (valid Q1) or the identity (failed `x_o`, failed SF `b`, `b = B`), with `gamma_f` and `a` from `dfthc.data.fermion`; the adjoint gives the adjoint actions (2 tests).
- T4: T = 5 with three `Toffoli` and two `TemporaryAND` (A = 2), M = 0, for the gate and its adjoint (2 tests); FeMoco on the closed form.
- T5 at `level="device"` (gate and adjoint) and T6 (the verbose listing names `_majorana_gate`); T2 at two wire labelings (the gate has one size), the rule pinned in `FIXED_DECOMPS` and the only one registered under its name; `required_wires`; three constructor violations.

**Definition of done:** `16 passed`; the per-commit tier passes; `ruff check .` clean; one commit.

- [ ] **Step 1: Write the failing tests**

Create `tests/ops/test_majorana.py`:

```python
"""MajoranaGate (spec 9.16): the action on every branch, the flavour convention, counts."""

import functools

import numpy as np
import pennylane as qp
import pytest

from dfthc import costs
from dfthc.compile import FIXED_DECOMPS, GATE_SET, compiled_resources
from dfthc.data.fermion import annihilation, majorana
from dfthc.ops.majorana import MajoranaGate, _majorana_gate
from dfthc.testing.resources import chosen_rules, verbose_program
from dfthc.testing.sparse_sim import SparseSimulator
from dfthc.testing.tape import decompose_for_simulation

# Wire order of the 7 one-wire registers, as in the operator: wire 0 is the most significant bit
# of a basis index |succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target>.
NAMES = ("succ_x_o", "succ_b", "G0", "G1", "b_neq_B", "flavour", "target")


def reference_matrix():
    """Spec 9.16 step by step, on every basis input: a = succ_x_o AND G0,
    s = succ_x_o AND NOT G0 AND succ_b; (1) -Z on target if s AND b_neq_B; (2) X on target if a;
    (3) Z on target if a AND flavour; (4) Z on flavour if a AND G1."""
    matrix = np.zeros((2**7, 2**7))
    for index in range(2**7):
        succ_x_o, succ_b, g0, g1, b_neq_b, flavour, target = (
            (index >> (6 - i)) & 1 for i in range(7))
        a, s = succ_x_o & g0, succ_x_o & (1 - g0) & succ_b
        sign = -1 if s and b_neq_b and target == 0 else 1
        target ^= a
        sign *= -1 if a and flavour and target else 1
        sign *= -1 if a and g1 and flavour else 1
        matrix[(index & ~1) | target, index] = sign
    return matrix


@functools.cache
def compiled_matrix(adjoint: bool):
    """The compiled gate's 2**7 x 2**7 matrix from one lightning run: each of the seven wires is
    entangled with a reference wire (H, CNOT), so the amplitude of |y>|x>|00>_work is
    U[y, x] / sqrt(2**7); the work wires must return to |00>."""
    system, reference, work = list(range(7)), list(range(7, 14)), [14, 15]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS))
    @qp.qnode(qp.device("lightning.qubit", wires=16))
    def circuit():
        for s, r in zip(system, reference, strict=True):
            qp.H(r)
            qp.CNOT([r, s])
        program(adjoint, system + work)()
        return qp.state()

    state = np.asarray(circuit()).reshape(2**7, 2**7, 4)
    assert np.allclose(state[:, :, 1:], 0, rtol=0, atol=1e-10)
    return state[:, :, 0] * np.sqrt(2**7)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["gate", "adjoint"])
def test_action_on_every_basis_input(adjoint):
    """T3: the compiled matrix (and its auto-derived adjoint) equals spec 9.16's four steps on
    all 128 basis inputs of the seven registers, every branch included, with clean work wires."""
    expected = reference_matrix()
    assert np.allclose(compiled_matrix(adjoint), expected.T if adjoint else expected, rtol=0,
                       atol=1e-10)


@pytest.mark.parametrize("adjoint", [False, True], ids=["gate", "adjoint"])
def test_tape_path_keeps_the_and_contracts(adjoint):
    """The tape path (spec 7.4 rule 9), as the sparse simulator runs it, on all 128 inputs at
    once (as in T3): spec 9.16's four steps with clean work wires. The simulator checks on every
    term that each TemporaryAND target is 0 and each Adjoint(TemporaryAND) target equals its AND;
    lightning runs both as a Toffoli, so T3 and T4 cannot tell a compute from a measured
    uncompute."""
    system, reference, work = list(range(7)), list(range(7, 14)), [14, 15]

    def qfunc():
        for s, r in zip(system, reference, strict=True):
            qp.H(r)
            qp.CNOT([r, s])
        program(adjoint, system + work)()

    state = SparseSimulator(16).run(decompose_for_simulation(qfunc).operations)
    amplitudes = state.to_dense().reshape(2**7, 2**7, 4)
    expected = reference_matrix()
    assert np.allclose(amplitudes[:, :, 1:], 0, rtol=0, atol=1e-10)
    assert np.allclose(amplitudes[:, :, 0] * np.sqrt(2**7), expected.T if adjoint else expected,
                       rtol=0, atol=1e-10)


PLUS = np.array([1, 1]) / np.sqrt(2)


def generator_on_target(succ_x_o, succ_b, g0, g1, b_neq_b):
    """The 2 x 2 action spec 6.2 assigns to psi_dn[0] on one branch, from dfthc.data.fermion
    (N = 1, spin down): i gamma_0 gamma_1 = -Z on a valid SF branch with b != B, a = (X + iY)/2 on
    a valid D1 branch, a^dag on a valid Q1 branch, the identity otherwise."""
    gamma = [majorana(0, f, 1, 1)[:2, :2] for f in (0, 1)]
    lower = annihilation(0, 1, 1)[:2, :2]
    if succ_x_o and g0:
        return lower if g1 == 0 else lower.conj().T
    if succ_x_o and succ_b and b_neq_b:
        return 1j * gamma[0] @ gamma[1]
    return np.eye(2)


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["gate", "adjoint"])
def test_flavour_projected_action_is_the_generator(adjoint):
    """The flavour convention (spec 6.2, 9.16): with flavour = sigma_0 projected onto |+> before
    and after, the compiled gate acts on target as -Z (SF, b != B), a (D1), a^dag (Q1) or the
    identity, on each of the 32 control branches; the adjoint gives the adjoint actions."""
    matrix = compiled_matrix(adjoint)
    project = np.kron(PLUS, np.eye(2))  # <+|_flavour (x) I_target, a 2 x 4 matrix
    for control in range(32):
        bits = [(control >> (4 - i)) & 1 for i in range(5)]
        block = matrix[4 * control:4 * control + 4, 4 * control:4 * control + 4]
        expected = generator_on_target(*bits)
        assert np.allclose(project @ block @ project.T,
                           expected.conj().T if adjoint else expected, rtol=0, atol=1e-10), bits


def program(adjoint, wires=tuple(range(9))):
    """Queues the gate (or its adjoint) on the seven one-wire registers and two work wires."""

    def queue():
        op = MajoranaGate(*([w] for w in wires[:7]), work=list(wires[7:]))
        if adjoint:
            qp.adjoint(op)

    return queue


@pytest.mark.compiled
@pytest.mark.parametrize("adjoint", [False, True], ids=["gate", "adjoint"])
def test_toffoli_count(adjoint):
    """T4: T = 5 with three Toffoli and two TemporaryAND (A = 2), M = 0; the same for the
    auto-derived adjoint."""
    resources = compiled_resources(program(adjoint), num_wires=9)
    assert (resources.toffoli, resources.measurements) == (costs.majorana_gate().toffoli, 0)
    assert (resources.counts["Toffoli"], resources.counts["TemporaryAND"]) == (3, 2)
    assert resources.measured_and_uncomputes == 2


def test_femoco_count_on_the_closed_form():
    """FeMoco-54 (spec 9.24): 5 Toffolis for the gate and for its adjoint."""
    assert costs.majorana_gate() == costs.Cost(5)


@pytest.mark.compiled
def test_pinned_rule_is_chosen_and_nothing_survives_at_device_level(capfd):
    """T6: the verbose listing names the pinned rule; T5 at level="device" for the gate and
    its auto-derived adjoint (L-20)."""

    def both():
        program(False)()
        program(True)()

    qp.analyze(verbose_program(both, num_wires=9), level="user")()
    captured = capfd.readouterr()
    assert "_majorana_gate" in chosen_rules(captured.out + captured.err)
    compiled_resources(both, num_wires=9, level="device")


@pytest.mark.parametrize("wires", [list(range(9)), [8, 3, 5, 0, 7, 1, 6, 2, 4]],
                         ids=["ordered", "permuted"])
def test_rule_is_valid(wires):
    """T2 (tape mode). The gate has one size; the second case relabels the wires. The rule is
    pinned and the only rule registered under the class name (spec 7.4 rules 7 and 8)."""
    assert FIXED_DECOMPS["MajoranaGate"] is _majorana_gate
    assert list(qp.list_decomps(MajoranaGate)) == [_majorana_gate]
    op = MajoranaGate(*([w] for w in wires[:7]), work=wires[7:])
    qp.testing.assert_valid_decomp_rule(op, _majorana_gate, check_matrix=False)


def test_required_wires():
    assert MajoranaGate.required_wires() == {name: 1 for name in NAMES} | {"work": 2}


@pytest.mark.parametrize(("change", "match"), [
    ({"work": [7]}, "'work' has 1 wires; expected 2"),
    ({"target": [6, 9]}, "'target' has 2 wires; expected 1"),
    ({"G0": []}, "'G0' has 0 wires; expected 1"),
])
def test_constructor_validates(change, match):
    args = {name: [i] for i, name in enumerate(NAMES)} | {"work": [7, 8]}
    with pytest.raises(ValueError, match=match):
        MajoranaGate(**(args | change))
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/ops/test_majorana.py -q`
Expected: FAIL at collection, `1 error`: `ModuleNotFoundError: No module named 'dfthc.ops.majorana'`.

- [ ] **Step 3: Write the Majorana gate**

Create `src/dfthc/ops/majorana.py`:

```python
"""``MajoranaGate``: the generator's one-qubit Majorana action after the Givens ladder (spec 9.16).

After the ladder the generator acts on ``psi_dn[0]`` alone, where the Jordan-Wigner string is
empty, so ``gamma_0 = X`` and ``gamma_1 = Y`` there (spec 6.2). SF branches apply
``i gamma_0 gamma_1 = -Z`` (dfthcbe's sign, not Fig. 4's ``+Z``); D1 and Q1 branches apply
``a = (X + iY)/2`` and ``a^dag = (X - iY)/2`` through the flavour ancilla ``sigma_0``, which arrives
in ``|+>`` from the inner prepare and is projected back onto ``|+>`` by the inner unprepare.
"""

from __future__ import annotations

import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._validate import check_wires

_REGISTERS = ("succ_x_o", "succ_b", "G0", "G1", "b_neq_B", "flavour", "target", "work")


class MajoranaGate(Operator2):
    r"""The Majorana gate of step 15: ``-Z`` on SF branches, ``a`` and ``a^dag`` on D1 and Q1.

    **Action.** With :math:`a = \mathrm{succ}_{x_o} \wedge G_0` (a valid one-body branch) and
    :math:`s = \mathrm{succ}_{x_o} \wedge \neg G_0 \wedge \mathrm{succ}_b` (a valid SF branch),
    in this order (spec 9.16):

    1. if :math:`s \wedge [b \ne B]`: :math:`-Z` on ``target`` (a phase -1 on ``target = 0``);
    2. if :math:`a`: ``X`` on ``target``;
    3. if :math:`a \wedge` ``flavour``: ``Z`` on ``target``;
    4. if :math:`a \wedge G_1`: ``Z`` on ``flavour``.

    Every other register is unchanged. On ``target = psi_dn[0]``, :math:`\gamma_0 = X` and
    :math:`\gamma_1 = Y` (spec 6.2), so step 1 is :math:`i\gamma_0\gamma_1 = -Z`, the SF pair
    (dfthcbe's sign; the paper's Fig. 4 applies :math:`+Z`, which would flip every :math:`w_b`
    against :math:`w_B`). **Flavour convention:** ``flavour`` is ``sigma_0``, which arrives in
    :math:`|+\rangle` on D1/Q1 branches; with it projected onto :math:`|+\rangle` before and
    after, steps 2 to 4 give :math:`(X + iY)/2 = a` on D1 (:math:`G_1 = 0`) and
    :math:`(X - iY)/2 = a^\dagger` on Q1 (:math:`G_1 = 1`). On SF branches ``flavour`` is the spin
    sum and is untouched. D1/Q1 ignores ``succ_b``; failed :math:`x_o` and failed SF :math:`b`
    branches get the identity.

    **Registers.**

    ============ ======= ================================ ==============================
    name         width   input                            output
    ============ ======= ================================ ==============================
    ``succ_x_o`` 1       any                              unchanged
    ``succ_b``   1       any                              unchanged
    ``G0``       1       any (0: SF, 1: one-body)         unchanged
    ``G1``       1       any (0: D1, 1: Q1)               unchanged
    ``b_neq_B``  1       any (:math:`[b \ne B]` on SF)    unchanged
    ``flavour``  1       any (``sigma_0``)                ``Z`` applied by step 4
    ``target``   1       any (``psi_dn[0]``)              the Majorana action
    ``work``     2       ``|0>``                          ``|0>``
    ============ ======= ================================ ==============================

    **Compilable arguments.** None; every register has a fixed width.

    **Pre- and postconditions.** ``work`` starts and ends in ``|0>``.

    **Decomposition** (dfthcbe's ``Maj``). ``work[0] = succ_x_o AND G0`` (``TemporaryAND``);
    ``CNOT(work[0], succ_x_o)``, which leaves :math:`\mathrm{succ}_{x_o} \wedge \neg G_0`;
    ``work[1] = succ_x_o AND succ_b`` (``TemporaryAND``), so ``work`` holds :math:`a` and
    :math:`s`. Step 1 is ``CCZ(work[1], target, b_neq_B)`` with ``target`` as an open control
    (``X``-conjugated); step 2 ``CNOT(work[0], target)``; step 3 ``CCZ(work[0], flavour,
    target)``; step 4 ``CCZ(work[0], G1, flavour)``; each ``CCZ`` is a ``Toffoli`` conjugated by
    ``H`` on its last wire. Then the measured uncompute of ``work[1]``, the ``CNOT`` again and
    the measured uncompute of ``work[0]``.

    **Adjoint and control.** The adjoint is auto-derived (the body has no loops or
    conditionals): it is the :math:`\mathrm{Maj}^\dagger` of ``InnerSelectDagger`` (spec 9.18),
    with ``a`` and ``a^dag`` exchanged on the D1 and Q1 branches. Never controlled.

    **Cost.** :func:`dfthc.costs.majorana_gate`: :math:`T = 5`, two ``TemporaryAND`` and three
    ``Toffoli`` (``A = 2``); ``M = 0``. The adjoint costs the same.

    **Paper and dfthcbe.** Step 15; the Majorana gate of the paper's Fig. 4 (PRX;
    arXiv:2502.15882v1 App. B.6), which applies the generators of its Eqs. (29) and (30) (v1
    numbering) on one qubit. Fig. 4 controls all four gates by one AND of both success flags (7
    Toffolis); ours, as dfthcbe's, folds the flags into the two ANDs with :math:`G_0` (5), which
    also leaves D1 and Q1 on where :math:`\mathrm{succ}_b = 0`, and takes dfthcbe's SF sign.
    dfthcbe: ``Maj`` (``dfthcbe/majorana.py``).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.majorana import MajoranaGate

        program = lambda: MajoranaGate([0], [1], [2], [3], [4], [5], [6], work=[7, 8])
        compiled_resources(program, num_wires=9).toffoli  # 5
    """

    wire_argnames = _REGISTERS

    def __init__(self, succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
        check_wires("MajoranaGate",
                    dict(zip(_REGISTERS, (succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target,
                                          work), strict=True)),
                    self.required_wires())
        super().__init__(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work)

    @staticmethod
    def required_wires() -> dict[str, int]:
        """Exact register sizes."""
        return {name: 1 for name in _REGISTERS[:-1]} | {"work": 2}


def _ccz(a, b, target) -> None:
    qp.H(target)
    qp.Toffoli([a, b, target])
    qp.H(target)


def _majorana_gate_resources(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
    return {resource_key(lambda: qp.TemporaryAND(Wire[3])): 2,
            resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3]))): 2,
            qp.Toffoli: 3, qp.CNOT: 3, qp.H: 6, qp.X: 2}


@qp.register_resources(_majorana_gate_resources)
def _majorana_gate(succ_x_o, succ_b, G0, G1, b_neq_B, flavour, target, work):
    succ, a, s, t = succ_x_o[0], work[0], work[1], target[0]
    qp.TemporaryAND([succ, G0[0], a])
    qp.CNOT([a, succ])
    qp.TemporaryAND([succ, succ_b[0], s])
    qp.X(t)
    _ccz(s, t, b_neq_B[0])
    qp.X(t)
    qp.CNOT([a, t])
    _ccz(a, flavour[0], t)
    _ccz(a, G1[0], flavour[0])
    qp.adjoint(qp.TemporaryAND([succ, succ_b[0], s]))
    qp.CNOT([a, succ])
    qp.adjoint(qp.TemporaryAND([succ, G0[0], a]))


register_rule(MajoranaGate, _majorana_gate)
```

- [ ] **Step 4: Run the tests**

Run: `scripts/test.sh tests/ops/test_majorana.py -q`
Expected: `16 passed` (11 to 16 s with 2 workers).

- [ ] **Step 5: Run the per-commit tier and the linter**

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `672 passed` (656 + 16).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/dfthc/ops/majorana.py tests/ops/test_majorana.py
git commit -m "Add MajoranaGate"
```
