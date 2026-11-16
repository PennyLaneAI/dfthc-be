# Final review (Task 5.1)

Reviewed tree: local `main` at `22d0fe5dd9247ef05793c9e56ae82ad706f96316` (`993b8fb..22d0fe5`, 329 commits), in four area packages: 1 operators, 2 operator tests, 3 data, report and tooling, 4 documents (plan `phase5-final`, Task 5.1). Each area was reviewed by Opus (`inherit`) and GPT-5.6 (`gpt-5.6-sol-high-fast`) in parallel; the eight reviews are `.superpowers/sdd/final-review/area-<1 to 4>-<opus or gpt>.md` (not committed). Merged by the orchestrator on 2026-11-16.

Code hash at review: `5e3eed1e7a9535765299d6c90afe643f96a5155e7eb43d43e039864133fbba8e`. A fix is hashed when it changes code (not only docstrings or comments) in a file of `dfthc.testing.provenance.hashed_files()` (`src/dfthc/**/*.py` outside `dfthc/testing/`, and `scripts/resource_report.py`) or changes `requirements-nightly.lock`.

## Summary

| Severity | Findings | Fix | Hashed fix | Declined or deferred |
| --- | ---: | ---: | ---: | ---: |
| Critical | 0 | 0 | 0 | 0 |
| Important | 4 | 4 | 2 | 0 |
| Minor | 32 | 30 | 6 | 2 |

Hashed change in the fix wave: yes. With yes, the orchestrator regenerates the small-system report and recompiles the FeMoco-54 walk and controlled walk once (the contingency pair); `fe2s2` and `fe4s4` keep their measurements as history and show their closed forms, marked pending.

The eight hashed fixes are I-1, I-2, m-1, m-4, m-5, m-6, m-15 and m-22. None of them may move a reported T, M, A or wire count at any reported system or preset; the fixer stops and reports if one does.

How the findings are numbered: in area order; within an area, in the order of the first review that raised them (Opus before GPT, each review in its own order), and a Minor rating that a review gave only in its ledger-row table follows that review's numbered findings. A finding raised in several areas sits in the area whose package holds the file it changes (I-4 and m-25 in area 4, m-22 in area 3). "Raised by" names each review as `area-<k>-<opus or gpt>` with its own finding id in parentheses. "The spec" is `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md`.

## Critical

None. No review raised a Critical finding.

## Important

### I-1: The pinned `C(ZeroReflection)` rule silently compiles a second control or a control value 0 to the wrong operation

- **Where:** `src/dfthc/ops/reflections.py:162-178` (the rule), `:152-158` (its resources) and the docstring at `:46-51`, at the reviewed tree.
- **Evidence:** The rule reads `control = control_wires[0]` and hard-codes the value 1 (`_mcz([control, *controls], [1, *values], ...)`, `qp.Z(control)`); it checks only the work length (`:166-168`), never the number of controls or `control_values`. The reviewer's probe `.scratch/final-review-area-1-opus/probe_c_zero_reflection.py`, rerun by the orchestrator in Step 7 under the watchdog, gives overlap 1.0 for one control with value 1, 0.625 for controls `[2, 3]` with values `[1, 1]`, and 0.5 for control value 0, with no error. No walk or QPE program requests `C(ZeroReflection)`, so no reported count is affected; the trap is in the public L1 primitive of spec 9.6.
- **Fix:** In `_c_zero_reflection`, raise `ValueError` unless there is exactly one control wire with control value 1 (the documented domain); the resource function stays as it is for that case. Test in `tests/ops/test_zero_reflection.py`: the probe's two cases (two controls; control value 0) raise that `ValueError`. It is RED today (a silent wrong state) and GREEN after; T, M and A of every walk, controlled walk and QPE program stay unchanged.
- **Hashed:** yes.
- **D criterion:** D-1. **Ledger row:** none.
- **Raised by:** area-1-opus (I-1), as Important.
- **Decision:** fix (Step 7 reproduced it).

### I-2: `QPERound` does not validate its nested walk compilable arguments

- **Where:** `src/dfthc/qpe.py:345-363`, `src/dfthc/qpe.py:202-205` and `src/dfthc/ops/walk.py:708-711`, at the reviewed tree.
- **Evidence:** `QPERound.__init__` checks only `isinstance(walk_compilable, tuple)` and calls `cls.required_wires(2**j, *walk_compilable)`; for `ControlledWalkPower` that reaches `ControlledDFTHCWalk.required_wires`, which reads `params` only. Malformed tables therefore pass the constructor and raise `ValueError` only when `_qpe_round` builds the child during decomposition (`qpe.py:402`). Spec 9.0 requires every constructor to validate every compilable argument.
- **Fix:** Add a side-effect-free compilable-argument validator to the walk-power registration protocol, implement it for `ControlledWalkPower` and `ToyControlledWalkPower`, and call it in `QPERound.__init__` before the wire sizes are computed. Test in `tests/test_qpe.py`, next to `test_qpe_round_validates`: a `QPERound` with malformed nested tables raises `ValueError` at construction. It is RED today (the object constructs) and GREEN after; the valid tape and capture paths are unchanged.
- **Hashed:** yes.
- **D criterion:** D-1 and D-4. **Ledger row:** none.
- **Raised by:** area-1-gpt (I-1), as Important.
- **Decision:** fix (Step 7 confirmed it by reading: the error is not silent, but late, a spec 9.0 contract gap).

### I-3: The canaries do not record three sub-claims of spec 14 (L-19, L-2 and L-18)

- **Where:** `tests/canary/test_l12_l13_l14_l19_templates.py:90-120`, `tests/canary/test_l01_to_l05_rules.py:60-101` and `tests/canary/test_l18_l20_warnings_runtime.py:42-63`, at the reviewed tree.
- **Evidence:** L-19's claim that a power-of-two `AliasSampling` compiles appears only in a docstring (`test_l12_l13_l14_l19_templates.py:100-104`); the parametrization (`:93-98`) holds only the two L = 5 failures. L-2's "has no path otherwise" is in the docstring of `test_l01_to_l05_rules.py:60-65`, but the test asserts only the fallback (`:101`). L-18's misleading "could not be parsed" text is deliberately not asserted (`test_l18_l20_warnings_runtime.py:42-47`). D-5 (spec 2.3) asks that the canary suite record the current state of every known limitation, so these three claims can change without a canary flipping.
- **Fix:** Add three canary cases: a power-of-two `AliasSampling` that compiles (L-19); a controlled measuring adjoint without a unitary inverse that fails to compile (L-2); and the L-18 failure message containing "could not be parsed" next to the real missing-rule cause. The claim-to-case inventory test that is RED before the three cases exist and GREEN after is added only if it runs in under 1 s.
- **Hashed:** no.
- **D criterion:** D-5. **Ledger row:** none.
- **Raised by:** area-3-gpt (I-1), as Important.
- **Decision:** fix (Step 7 confirmed the three gaps; the claim-to-case inventory is optional, only if under 1 s).

### I-4: Spec 11.3 gives the QPE program's wire totals from before Q1b (64 + t and 1,565 + t; the code gives 56 + t and 1,103 + t)

- **Where:** `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md:1235` at the reviewed tree.
- **Evidence:** The spec reads "64 + t at `H2_STO3G_TEST` for t ≤ 4, 1,565 + t at FeMoco-54 for t ≤ 22". `WalkLayout.from_params(p, qpe_bits=t).num_wires` gives 57, 58 and 59 at `H2_STO3G_TEST` for t = 1, 2, 3 (56 + t) and 1,104 at FeMoco-54 for t = 1 (1,103 + t); `tests/test_qpe.py:328` asserts `55 + 1 + t + spare`, and `docs/resources.md:421-426` lists 1,104 to 1,127 qubits. The differences, 8 and 462, are the junk wires Q1b moved onto the angle wires; the bounds t ≤ 4 and t ≤ 22 and the 2t − 45 spare wires are still right.
- **Fix:** Write "56 + t at `H2_STO3G_TEST` for t ≤ 4, 1,103 + t at FeMoco-54 for t ≤ 22". Test in `tests/test_docstrings.py`, which reads only the two totals of spec 11.3 and compares them with `WalkLayout.from_params(H2_STO3G_TEST).num_wires + 1` and `WalkLayout.from_params(FEMOCO54).num_wires + 1` (import only, no compile): RED on 64 and 1,565, GREEN on 56 and 1,103.
- **Hashed:** no.
- **D criterion:** D-6 (and D-4's documentation). **Ledger row:** none.
- **Raised by:** area-4-gpt (I-1) and area-4-opus (I-1), as Important; area-1-opus (M-3) and area-2-opus (M-4), as Minor.
- **Decision:** fix, at Important (Step 7: the higher severity; the lower rating was not shown right). The test only reads the two numbers the code computes, as step 8 allows.

## Minor

### m-1: `cheapest_block_bits` ignores the erases' one-hot fit, and `WalkParams` accepts shapes whose walk cannot be built

- **Where:** `src/dfthc/costs.py:324-340`; the constructors' checks at `src/dfthc/ops/prepare.py:117-122`, `306`, `784` and `src/dfthc/ops/select.py:53-57`; `src/dfthc/params.py:59-74`; at the reviewed tree.
- **Evidence:** The minimizer treats `(k1, ke1)`, `k2` and `ke2` as free, but the constructors require `2**ke1 <= K1 d1`, `2**(ke2 + w_b) <= K2 d2` and, for the select's one-hot, `2**w_b <= (N - 1) b_rot + 1`. At `(N, R, B, C) = (10, 1, 16, 1)` the defaults `(k1, k2, ke1, ke2) = (0, 0, 2, 0)` give 32 > K2 d2 = 22, so `InnerPrepare` and the walk raise `ValueError` although `k2 = 1` fits; a sweep of 14,040 shapes found 576 inner-fit and 1,920 select-fit violations. Every reported system and preset is feasible, and a violation is a loud `ValueError`, never a wrong count.
- **Fix:** Minimize `(k1, ke1)` subject to `2**ke1 <= K1 d1` and `(k2, ke2)` jointly subject to `2**(ke2 + w_b) <= K2 d2`; reject `2**w_b > (N - 1) b_rot + 1` in `WalkParams` (and list it in spec 8.3). Test in `tests/test_costs.py`: the defaults at `(10, 1, 16, 1)` satisfy both fits (RED today), and `costs_reference.default_params` and the defaults of every reported system and preset are unchanged.
- **Hashed:** yes.
- **D criterion:** D-1. **Ledger row:** none.
- **Raised by:** area-1-opus (M-1), as Minor.
- **Decision:** fix (a hashed Minor with the hashed wave of I-1 and I-2), with the guard step 8 requires: the test asserts that no reported default changes.

### m-2: Three docstrings say a failed `x_o` selects a zero inner entry and gets +1 from `InnerSigns`; in the walk a failed draw reaches `InnerPrepare` as `x_o = 0`

- **Where:** `src/dfthc/ops/prepare.py:36-38`, `823-828` and `1301-1303` at the reviewed tree; spec 9.8 (Post) has the same wording.
- **Evidence:** On a failed draw the outer target is zero and the swap leaves `x_o = 0` (`prepare.py:344-346`; spec 9.7), an SF row, so the walk loads row 0's data and applies row 0's signs. The result is still exact (the Majorana gate is off for `succ_x_o = 0`, `majorana.py:126-129`, and `InnerSigns` squares to the identity across the reflection), and `test_block_against_the_encoded_model[fails-c3]` passes. The statements hold only for a standalone input `x_o >= X_o`, which the walk never presents; `angle_data.py:14-16` already states the walk's behaviour.
- **Fix:** In the three docstrings and in spec 9.8, qualify "failed `x_o`" as "`x_o >= X_o`, an input the walk never presents: `OuterPrepare`'s swap maps a failed draw to `x_o = 0`, an SF row, where the inner signs cancel between the two rounds and `REF_T2` gives −I". Docstring and spec text only, so no behavioural test can be RED; `tests/test_docstrings.py` stays green.
- **Hashed:** no (docstrings and spec text).
- **D criterion:** D-1. **Ledger row:** none.
- **Raised by:** area-1-opus (M-2), as Minor.
- **Decision:** fix.

### m-3: `_capture_parities` compares once per nonzero mask (optional)

- **Where:** `src/dfthc/compile.py:251-254` at the reviewed tree.
- **Evidence:** The helper still indexes `parities[row]` and compares once per nonzero mask; one comparison on the vector would save one nested equation per nonzero mask (the ledger: about 2% of the FeMoco-54 compile). It is count-neutral.
- **Fix:** Not applied. The change would compare once on the vector (`vec = packed_parities(...) == 1`, then `vec[row] if mask else False`); a capture test counting the equations of one erase would show the saving.
- **Hashed:** yes.
- **D criterion:** none. **Ledger row:** L266.
- **Raised by:** area-1-opus, as Minor (optional) in its ledger-row table; area-1-gpt declined the row.
- **Decision:** declined: an optional row not raised to Important (plan, section "Hashed and unhashed fixes"); its condition, "if compile memory gets tight", did not occur (1.7 GB), and a performance edit in the hashed hot path adds risk to the one FeMoco-54 recompile without a benefit (step 8).

### m-4: Duplicated helpers and validation in the angle erase and the select

- **Where:** `src/dfthc/ops/qroam.py:170-176` and `src/dfthc/ops/angle_data.py:377-384` (`_dummy` and `_dummy_wires`); `src/dfthc/ops/select.py:42-58` against `src/dfthc/ops/angle_data.py:476-489`; `AngleErase.required_wires` at `angle_data.py:498-507` and `select._child_sizes` at `select.py:71-72`; at the reviewed tree.
- **Evidence:** The two dummy-wire allocators are identical line for line. `select._check` repeats `AngleErase`'s table, shape, one-hot-range and capacity checks almost verbatim, and `AngleErase.required_wires` takes `tail_angles`, `sf_angles` and `n_plus` and uses none of them. A divergence would fail loudly, because the select's rule constructs `AngleErase`, which validates again, so no wrong result follows.
- **Fix:** One dummy-wire helper; one select and erase validator in `angle_data` called by both constructors; document or narrow the size-only arguments without changing the public operator signature. The refactor changes no behaviour, so no test is RED before it; a new parity test asserts that `InnerSelect` and `AngleErase` reject the same invalid domains with the same message, and the existing constructor tests stay green.
- **Hashed:** yes.
- **D criterion:** none. **Ledger row:** L278 and L332.
- **Raised by:** area-1-gpt (M-2), as Minor; area-1-opus, as Minor in its ledger-row table (rows L278 and L332).
- **Decision:** fix (a hashed Minor with the hashed wave).

### m-5: Constructors validate wire lengths but not wire identity

- **Where:** `src/dfthc/ops/_validate.py:28-38`; `src/dfthc/ops/select.py:84-87`, `149-153`; `src/dfthc/ops/walk.py:278-281`, `650-656`; at the reviewed tree.
- **Evidence:** `check_wires` compares lengths only. The reviewer's probe constructs `SpinSwap(sg=[0], psi_up=[1, 2], psi_dn=[2, 3])` and a `MajoranaGate` whose `flavour` equals its `target` without an error; the compiled path then fails with a compiler-level error, and the tape path catches a repeat only inside one leaf gate. The walk passes disjoint registers by construction (`src/dfthc/registers.py:28-67`), and `iterative_phase_estimation` already checks distinctness (`qpe.py:457-461`).
- **Fix:** `check_wires` raises `ValueError` for a wire label repeated within or across nonempty concrete registers (skipping `Wire[n]` keys and tracers; the junk and angle aliases stay outside a single operator bind). Test: shared constructor tests in which the probe's two overlaps raise `ValueError` (RED today: they construct), plus zero-width registers and the permitted longer `phase_grad` and `work`.
- **Hashed:** yes.
- **D criterion:** D-1. **Ledger row:** L339 and L332.
- **Raised by:** area-1-gpt (M-3), as Minor; area-1-opus, as Minor in its ledger-row table (row L339).
- **Decision:** fix (a hashed Minor with the hashed wave).

### m-6: The device-level closure checks only dfthc operator names

- **Where:** `src/dfthc/compile.py:129-136` and `159-169` at the reviewed tree.
- **Evidence:** At `level="device"` the helper asks only whether a count name contains a registered dfthc class name (`compile.py:133-136`), so a surviving non-dfthc custom `Operator2` passes, which is narrower than spec 7.2's "no undecomposed operator remains". The user-level closure is total (`compile.py:129-132`), and since L-20 changed an operator without a rule fails the execution compile (`docs/limitations.md:132-137`), so no project result is wrong today.
- **Fix:** The device-level assertion rejects every residual custom operator or wrapper, not only names drawn from `FIXED_DECOMPS`. Test: a focused case with a non-dfthc custom operator that survives the device pass, RED today (it passes) and GREEN after (it raises).
- **Hashed:** yes.
- **D criterion:** D-1. **Ledger row:** L239.
- **Raised by:** area-1-gpt (M-1), as Minor; area-1-opus declined the row (no defect while the user-level closure is total).
- **Decision:** fix (a hashed Minor with the hashed wave, step 8's "any other hashed Minor"; the rating stands, because the decline does not show the check complete).

### m-7: The baseline unprepares' per-commit round trips miss mask permutations that keep each bit's position parity

- **Where:** `tests/ops/test_outer_prepare.py:174` (the all-ones schedule only); `tests/ops/test_inner_prepare.py:222` (all-ones and alternating) and its docstring at `:149-152`; at the reviewed tree.
- **Evidence:** Mutant M1 (every outer erase entry bit-reversed in `_outer_unprepare` and its resources) passes every deterministic per-commit test of `test_outer_prepare.py`; only the sampled `test_lightning_round_trip_with_two_seeds` kills it, at its second seed. Mutant M3 (bits 0 and 2 of every inner erase entry exchanged) passes all 48 per-commit tests of `test_inner_prepare.py`. The nightly certificates see every such permutation, so the gap is per commit only.
- **Fix:** Run both round trips under the all-ones, the alternating and a seeded schedule, as `tests/ops/test_unprepare_from_outcomes.py:208-211` does, and restate the inner docstring. In the reviewer's scratch copy the outer case fails under M1 at all 5 shapes and the inner case under M3 at 4 of 6 (RED), and both pass on the reviewed tree (GREEN), in under 1 s per commit.
- **Hashed:** no.
- **D criterion:** D-1. **Ledger row:** L283.
- **Raised by:** area-2-opus (M-1), as Minor; area-2-gpt declined the row (the certificate proves every outcome by linearity).
- **Decision:** fix: the mutant evidence outweighs the linearity argument, because the certificates are slow-tier (step 8); the added schedules must keep the per-commit cost under 10 s.

### m-8: `ControlledWalkPower`'s per-commit tests cannot see its groups bound to the wrong registers where the sizes agree

- **Where:** `tests/test_qpe.py:66-128` at the reviewed tree; there is no children test of `_controlled_walk_power`, and its T3 (`:83-104`) and verbose listing (`:196-208`) are nightly.
- **Evidence:** Mutant M2 (`system` and `phase_grad` swapped in the rule's group list, `src/dfthc/qpe.py:219`) passes 54 of the 55 per-commit tests of `tests/test_qpe.py` and `tests/integration/test_qpe_toy.py`; only `test_controlled_walk_power_counts[631g]` kills it, through a width check ("register 'system' has 4 wires; expected 8"). The test area-2-gpt cites, `test_walk_power_spec_flattens_the_groups_in_wire_argnames_order` (`tests/test_qpe.py:299-313`), checks the spec's flattening, not the rule's binding, and is among the 54 that M2 passes.
- **Fix:** A tape-mode children test (about 0.1 s, no compile) at sto3g with power 2 and at fails-c3 with power 1: the rule queues `power` operators named `ControlledDFTHCWalk`, each with its `wire_argnames` registers equal to `c.groups(True)`. In the reviewer's scratch copy it is RED under M2 (2 failed) and GREEN on the reviewed tree (2 passed).
- **Hashed:** no.
- **D criterion:** D-1 and D-4. **Ledger row:** L314 (its second item).
- **Raised by:** area-2-opus (M-2), as Minor; area-2-gpt declined this item.
- **Decision:** fix.

### m-9: The toy QPE inventory filters `C(` targets with `startswith`, weaker than the real-walk test

- **Where:** `tests/integration/test_qpe_toy.py:327` against `tests/integration/test_qpe_counts.py:130`, at the reviewed tree.
- **Evidence:** `name.startswith("C(")` misses `Adjoint(C(...))` and `2C(...)` targets, while the real-walk test uses `"C(" in name`. The `inventory(names) == set()` check on `:328` catches nested modifiers only of dfthc operators and the three templates (`src/dfthc/testing/resources.py:162-174`). The toy program at t = 3 captures no name containing `C(` today.
- **Fix:** `assert not [name for name in names if "C(" in name]`. It passes on the reviewed tree; an injected `Adjoint(C(...))` target passes the current filter and fails the new one (RED before, GREEN after).
- **Hashed:** no.
- **D criterion:** D-1 and D-4. **Ledger row:** L314 (its third item).
- **Raised by:** area-2-opus (M-3), as Minor; area-2-gpt declined this item (the inventory check makes it redundant).
- **Decision:** fix.

### m-10: The release real-walk QPE check uses one basis input

- **Where:** `tests/integration/test_qpe_walk.py:48-50`, `67-74` and `95-106` at the reviewed tree.
- **Evidence:** `INPUTS["sto3g"]` is the single basis value 10, and the release case at t = 2 builds `SparseState.basis` from it, so the check does not implement row L314's requested spread input. Area-2-opus shows the check is not blind: the comparison at 1e-12 sees a 0.0131 shift from the 0.021 weight off the dominant eigenvalue, and the input reaches 3 of the 16 eigenvalues.
- **Fix:** Not applied. The change would replace the input with a normalized complex superposition of two two-electron basis states, passed to `qpe_distribution` as well; a relative-phase mutant would survive the basis input and fail the spread one.
- **Hashed:** no.
- **D criterion:** D-4. **Ledger row:** L314 (its first item).
- **Raised by:** area-2-gpt (M-1), as Minor; area-2-opus declined this item.
- **Decision:** declined: the case runs only in the release tier, so a changed input cannot be checked before the single release run (the plan's decline reason "needs a heavy run"), and area-2-opus's evidence shows the basis input is already sensitive (step 8).

### m-11: The FeMoco-54 cache check's arms have no negative test; three mutants survive every tier

- **Where:** `tests/report/test_femoco_cache.py:44-77` (the arms) and `:80-90` (the only per-commit test), at the reviewed tree.
- **Evidence:** Mutants M4 (the gate-dictionary arm dropped), M5 (the wire arm dropped) and M8 (the environment and seconds arm dropped) each leave the file at "2 passed", the release test included; the per-commit test feeds only a stale entry. The gate-dictionary arm is Task 4.5's review fix for its I-1, added without a test that makes it fail.
- **Fix:** A per-commit parametrized test that copies one valid entry and injects one defect each (T, M or A other than `EXPECTED`; a gate outside `ANALYZE_NAMES`; `TemporaryAND` + 1 in `counts`; `num_wires` 1103 for the walk; no `environment`; no `seconds`; no `peak_mb`) and asserts exactly one problem naming its arm. It is RED under M4, M5 and M8 and GREEN on the reviewed tree.
- **Hashed:** no.
- **D criterion:** D-3. **Ledger row:** none (a follow-up of Task 4.5's review I-1).
- **Raised by:** area-3-opus (M-1), as Minor.
- **Decision:** fix.

### m-12: The watchdog's kill on the tree's summed memory is untested; a per-process-maximum mutant survives

- **Where:** `src/dfthc/testing/watchdog.py:330` (`peak = max(peak, sum(running.values()))`); `tests/test_watchdog.py:130-145`, `169-190` and `369-380`; at the reviewed tree.
- **Evidence:** Mutant M1 replaces the sum by the largest single process and leaves `tests/test_watchdog.py` at "21 passed": every memory-kill test has one process that alone exceeds the kill point, and `test_tree_memory_includes_grandchildren` checks `tree_memory_mb`, not `run_watched`'s decision. The release tier runs a pytest controller and two xdist workers under this watchdog, so the sum is what guards it.
- **Fix:** A test with two sibling processes of about 130 MB each under `cap_mb=200` (kill point 187.5 MB): the run is killed with "memory" while each process stays below the kill point. It is RED under M1 and GREEN on the reviewed tree, in a few seconds.
- **Hashed:** no.
- **D criterion:** none (the load policy that guards the release run). **Ledger row:** none.
- **Raised by:** area-3-opus (M-2), as Minor.
- **Decision:** fix.

### m-13: The cache's toolchain coverage rests on the lock's bytes, which nothing tests or compares with the running environment

- **Where:** `src/dfthc/testing/provenance.py:62`; `tests/report/test_report.py:360-397`; `tests/report/test_femoco_cache.py:72-74`; `tests/test_infrastructure.py:54-59`; at the reviewed tree.
- **Evidence:** `test_cache_needs_matching_code_hash` copies the lock but never changes it, so a mutant that drops `provenance.py:62` survives it; the release check requires only a non-empty `environment`; and the lock test checks 3 of the lock's 38 lines (row L231). A toolchain change made without refreshing the lock would keep a stale cached FeMoco-54 count current. Today the lock equals `pip freeze` exactly (both reviewers checked).
- **Fix:** Change a byte of the copied lock in `test_cache_needs_matching_code_hash` and assert that the hash changes (RED under the line-62 mutant); compare the entry's `environment` with `provenance.environment()` in `femoco_cache_problems` on every field except `platform`; replace the three-package lock test by a whole-lock comparison with the installed distributions, names normalized, with a unit test that injects one missing and one extra distribution (RED against the current three-package check).
- **Hashed:** no.
- **D criterion:** D-3. **Ledger row:** L231.
- **Raised by:** area-3-opus (M-3) and area-3-gpt (M-1), as Minor.
- **Decision:** fix.

### m-14: The report's exit-1 path is tested for a T and A mismatch only; the cross-check and the M arm have no negative test

- **Where:** `scripts/resource_report.py:768-769` and `:837-857`; `tests/report/test_report.py:164-183`; at the reviewed tree.
- **Evidence:** `test_report_fails_on_mismatch` patches `costs.spin_swap` to `Cost(p.N + 1)`, which moves T (and A through the row's offset) but never M. Spec 13.3's step 4 cross-check is exercised only positively, so a `cross_check` that returns `[]`, or a `closed` dict without `"measurements"`, survives the per-commit tier. The evidence is static: the reviewer did not call `main()`.
- **Fix:** Two monkeypatched `main(["--systems", "H2_STO3G_TEST", "--rows", ...])` cases: with no rows (no compile), where `closed_form` gives one partition row T + 1, expecting exit 1 and "partition ... != closed DFTHCWalk"; and `--rows SpinSwap` (one compile, about 3 s) with `costs.spin_swap` returning `Cost(p.N, 1)`, expecting exit 1 and "SpinSwap: measured". Both are RED under the two mutants named.
- **Hashed:** no (tests only).
- **D criterion:** D-3. **Ledger row:** none.
- **Raised by:** area-3-opus (M-4), as Minor.
- **Decision:** fix (about 3 s per commit, under 10 s).

### m-15: The FeMoco-54 QPE summary takes M from the controlled walk, where its own formula and spec 11.3 name M_W

- **Where:** `scripts/resource_report.py:820` and `:830`; `docs/resources.md:428`; `tests/report/test_report.py:494-495`; at the reviewed tree.
- **Evidence:** The script computes `(2**bits - 1) * basis["measurements"] + bits` with `basis = cwalk[...]`, while the printed formula "M = (2^t − 1) M_W + t", spec 9.24 (lines 1123-1125) and spec 11.3 use the walk's M; Task 4.5's review already moved `test_femoco_cache` to the walk's M (`tests/report/test_femoco_cache.py:124`). Both values are 2,524 today, so no number moves.
- **Fix:** Read M from the `DFTHCWalk` record (`(walk["measured"] or walk["closed"])["measurements"]`) and update `test_report.py:494-495` to the walk. A test with a controlled-walk record whose M differs from the walk's is RED before and GREEN after; the FeMoco-54 QPE rows are unchanged.
- **Hashed:** yes.
- **D criterion:** D-4 (the QPE's resource count) and D-3. **Ledger row:** none.
- **Raised by:** area-3-opus (M-5), as Minor.
- **Decision:** fix (a hashed Minor with the hashed wave; step 8: make code and text agree, no number moves).

### m-16: `report_toffolis.json`, the source of every dfthcbe number in the report, records no dfthcbe source digest

- **Where:** `tests/fixtures/dfthcbe/report_toffolis.json:1-8`; `scripts/make_report_fixture.py:103-112`; at the reviewed tree.
- **Evidence:** `provenance.json` records `dfthcbe_source_url`, `dfthcbe_source_sha256` and `dfthcbe_installed_sha256`, and a test pins them (`tests/data/test_dfthcbe_fixtures.py:135-140`); `report_toffolis.json` has only the commit and package versions, so nothing ties its 8 call-graph evaluations to the digest-verified dfthcbe tree that HANDOFF 3.4 describes.
- **Fix:** `make_report_fixture.py` writes the three fields, reusing `make_dfthcbe_fixtures.installed_digest`; `test_dfthcbe_blocks_equal_the_call_graph` asserts them equal to `SOURCE_SHA256` and `INSTALLED_SHA256` (RED on today's fixture, GREEN after); the orchestrator regenerates the fixture in `.venv-ref` (symbolic evaluation only, values unchanged).
- **Hashed:** no.
- **D criterion:** D-3. **Ledger row:** none.
- **Raised by:** area-3-opus (M-7), as Minor.
- **Decision:** fix: step 8's condition holds, because `.venv-ref` has the dfthcbe package installed (`dfthcbe-1.0.dist-info`), `installed_digest()` reads the imported package's files, and the source URL and digest are constants of `scripts/make_dfthcbe_fixtures.py`, so no network is needed.

### m-17: The worker-count test checks only the current invocation

- **Where:** `tests/test_infrastructure.py:30-35`; `scripts/test.sh:36`; at the reviewed tree.
- **Evidence:** With `DFTHC_TEST_WORKERS` unset the test checks `-n 2`, and under `DFTHC_TEST_WORKERS=1` it checks 1; a `test.sh` that hard-codes `-n 2` passes every default run, and the empty and `0` cases are never isolated. A non-numeric value such as `auto` passes through as `-n auto`, and the test skips it (`:32`).
- **Fix:** A subprocess test in the style of `test_runner_fails_closed_without_lowered_priority` (`:114-133`): a fake `DFTHC_VENV` python prints its argv, and `test.sh` run with the variable unset, empty, `0` and `3` must pass `-n 2`, `-n 2`, `-n 0` and `-n 3`. It is RED against a `test.sh` that hard-codes `-n 2`.
- **Hashed:** no.
- **D criterion:** none. **Ledger row:** L230.
- **Raised by:** area-3-opus (M-8), as Minor; area-3-gpt declined the row (`${DFTHC_TEST_WORKERS:-2}` is correct and the live check verifies it).
- **Decision:** fix.

### m-18: A reaped root's pid can be re-adopted by the next scan

- **Where:** `src/dfthc/testing/watchdog.py:159`, `166-168` and `342`; the docstring at `:23-25`; at the reviewed tree.
- **Evidence:** Each scan starts from `[self.root, *self.known]` (`:159`). Once the root is reaped and popped from `known`, the next scan meets its pid with no entry, so `setdefault` adopts whatever process now holds that pid, measures it and may stop and kill it. This needs pid reuse within the window (up to 5 s on the pipes-held path), which is remote on macOS. Area-3-gpt's reason to decline, that tracked pids are keyed by start time, covers the descendants but not the re-seeded root.
- **Fix:** Record the root's start time at the first scan and never re-adopt a pid whose start time differs (a set of retired pids in `_Tree`). Test with a monkeypatched `_rusage` that reports another start time for the root after its exit: the scan returns nothing for it (RED today).
- **Hashed:** no (`dfthc.testing`).
- **D criterion:** none. **Ledger row:** L233.
- **Raised by:** area-3-opus (M-9), as Minor; area-3-gpt declined the row.
- **Decision:** fix.

### m-19: A pipes-held run returns up to 10 s later than the module docstring says

- **Where:** `src/dfthc/testing/watchdog.py:25-29`, `308-311` and `365-385` at the reviewed tree.
- **Evidence:** The docstring says the run ends as a watchdog error 5 s after nothing traceable is left, but `_final_output` then waits twice more for `_KILL_SECONDS`, so the run returns at about 15 s. The test's own docstring states the true bound, `min(timeout_s, kill_seconds) + 2 * kill_seconds` (`tests/test_watchdog.py:335-336`), and its `[2-4.0]` case measured 10.11 s.
- **Fix:** When the loop already ended with `_PIPES_HELD`, close the pipes at once instead of repeating both waits, with a deterministic timing test that is RED under the three-wait path; or, if the code stays, state the true bound in both docstrings.
- **Hashed:** no.
- **D criterion:** none. **Ledger row:** L234.
- **Raised by:** area-3-opus (M-10) and area-3-gpt (M-2), as Minor.
- **Decision:** fix.

### m-20: A non-`WatchdogError` exception escapes `run_watched` and `main` without a `WATCHDOG` line

- **Where:** `src/dfthc/testing/watchdog.py:372-383` (`proc.kill()` inside `_final_output`, called from `finally`), `:303-320` (`Popen`) and `:394-412` (`main` catches only `WatchdogError`), at the reviewed tree.
- **Evidence:** When the root refuses signals, `proc.kill()` raises `PermissionError` out of `finally`, `run_watched` loses its record, and `main` ends in a traceback with status 1 instead of 3 (row L235). Row L236's stated case no longer reproduces: area-3-opus's probe with a nonexistent command printed `nice: ...: No such file or directory`, then a `WATCHDOG` line with return code 127, because the priority prefix runs and reports it. What remains of L236 is any other exception, such as an `OSError` from `Popen` itself.
- **Fix:** Wrap `proc.kill()` in `contextlib.suppress(PermissionError, ProcessLookupError)`, and let `main` catch `Exception` after `WatchdogError`, print `WATCHDOG {"killed": "watchdog error: <type>: <message>", ...}` and return 3. Tests: a fake `Popen` whose `kill` raises `PermissionError`, and a `run_watched` monkeypatched to raise; both are RED today (a traceback) and GREEN after (the line and status 3). A test with a nonexistent executable, as area-3-gpt proposed, would not be RED, so the monkeypatched test replaces it.
- **Hashed:** no.
- **D criterion:** none. **Ledger row:** L235 and L236.
- **Raised by:** area-3-opus (M-11) and area-3-gpt (M-3 for L235, M-4 for L236), all as Minor.
- **Decision:** fix.

### m-21: The pipes-held watchdog test keeps under 1 s of margin in both parametrizations

- **Where:** `tests/test_watchdog.py:321-356`, the bound at `:353`, at the reviewed tree.
- **Evidence:** Measured alone with one worker on a quiet machine, `[60-1.0-]` took 3.21 s against a bound of 4 s and `[2-4.0-...]` took 10.11 s against 11 s. The first includes Python start-up and the escaped child's readiness handshake, so it moves with load; the plan already lists this file as load-sensitive (Task 5.2, Step 10).
- **Fix:** Keep only the property the bound proves, that the run ends long before the escaped child's 15 s sleep: `elapsed < min(timeout_s, kill_seconds) + 2 * kill_seconds + 3` (6 s and 13 s). The looser bound has no RED state of its own; it removes false failures under load (a run 1 s slower than the measured one fails today's bound and passes the new one).
- **Hashed:** no.
- **D criterion:** none. **Ledger row:** L237.
- **Raised by:** area-3-opus (M-12), as Minor; area-3-gpt declined the row (the readiness protocol and the one-second allowance).
- **Decision:** fix.

### m-22: Two sentences of the FeMoco-54 report summary are stale or overcount

- **Where:** `scripts/resource_report.py:929-931` and `:815`, `966-967`, rendered at `docs/resources.md:375` and `:403`, at the reviewed tree.
- **Evidence:** Line 375 reads "The walk and the controlled walk are compiled once, in Task 4.5; until then they show their closed form, marked in their rows above", although both rows are measured (9,793 / 2,524 / 8,675 and 9,795 / 2,524 / 8,677). Line 403's "The junk registers' 462 wires ... share the angle register's wires" adds aliases that overlap: at FeMoco-54 `junk_x_o` (72 wires) and `junk_b` (390 wires) both start at `angles[0]`, so their union is the 390 wires `angles[:390]`; 462 is right as the saving, not as a number of wires.
- **Fix:** Present tense for the first sentence ("The walk and the controlled walk were compiled once each, in Task 4.5; their measured rows above equal their closed forms") and, for the second, "The junk registers (72 and 390 wires, aliased onto the first 390 angle wires) are X-measured right after each prepare". Pin both in `tests/report/test_report.py` (a FeMoco-54 summary test that rejects "until then", and `test_qubit_summary_at_femoco` the 390 wires), both RED on today's text; the report is regenerated in the hashed wave.
- **Hashed:** yes (string literals of `scripts/resource_report.py`).
- **D criterion:** D-3 (and D-6). **Ledger row:** L289 and L293.
- **Raised by:** area-3-opus (M-13 for L289, M-14 for L293), area-3-gpt (M-5 for L289, M-6 for L293) and area-4-opus (M-9, both sentences), all as Minor.
- **Decision:** fix (a hashed Minor with the hashed wave; L289 is Task 4.5's review M-2 and L293 is Q1b.2's m-2, both waiting for the next hashed change).

### m-23: Spec 14's L-6 mitigation omits `ToyControlledWalkPower`; HANDOFF 7's L-6 row is narrower still

- **Where:** `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md:1483-1485`; `docs/agents/HANDOFF.md:359`; at the reviewed tree.
- **Evidence:** Spec 14 reads "Since G0 the looped `QROAMRead` and `ControlledWalkPower` are never adjointed or controlled by construction", while spec 4.4 item 5 (line 293), spec 7.4 rule 4 (line 593), HANDOFF 5 rule 4 and `docs/limitations.md:38-40` name all three looped operators; `ToyControlledWalkPower` loops (`src/dfthc/testing/toy_walk.py:301`). HANDOFF's L-6 row names only `QROAMRead`, and only "never adjointed". No code adjoints or controls any of the three.
- **Fix:** Name `QROAMRead`, `ControlledWalkPower` and `ToyControlledWalkPower`, "never adjointed or controlled", in spec 14's L-6 and in HANDOFF 7's L-6 row. Extend the limitation consistency audit to compare this inventory, RED on today's spec and GREEN after; `rg -n 'looped'` over the spec, HANDOFF and `docs/limitations.md` then names the same three everywhere.
- **Hashed:** no.
- **D criterion:** D-6. **Ledger row:** L302.
- **Raised by:** area-4-opus (M-1) and area-4-gpt (M-2), as Minor.
- **Decision:** fix.

### m-24: Spec 14's L-23 and L-26 entries lag the code and `docs/limitations.md`

- **Where:** `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md:1546-1549` (L-23) and `:1556` (L-26), at the reviewed tree.
- **Evidence:** L-23 says the operators with a dynamic argument are "today `PhaseFixup` and `QPERound`", but five classes have dynamic arguments and convert lists in `__init__`: `QROAMRead` (`src/dfthc/ops/qroam.py:420`), `PhaseFixup` (`qroam.py:651`), `OuterUnprepareFromOutcomes` (`src/dfthc/ops/prepare.py:712`), `InnerUnprepareFromOutcomes` (`prepare.py:1234`) and `QPERound`; spec 9.0 (lines 778-781) already lists all five. L-26 says Catalyst's `_MLIR_DTYPES_TO_PY_DTYPES` maps only `i1` to `i64`; the installed map has `i1`, `i8`, `i16`, `i32` and `i64` (`catalyst/decomposition/type_utils.py:52-63`), as `docs/limitations.md:91-92` says.
- **Fix:** L-23 names the five classes; L-26 reads as `docs/limitations.md` does. Test: an introspection check that the spec's L-23 inventory equals the classes with `dynamic_argnames`, RED on today's two names and GREEN after.
- **Hashed:** no.
- **D criterion:** D-6. **Ledger row:** none.
- **Raised by:** area-4-opus (M-2, both entries) and area-4-gpt (M-3, L-23), as Minor.
- **Decision:** fix.

### m-25: Spec 12.6 and `docs/verification.md` put the subprocess canaries in the nightly tier; 17 of them run per commit

- **Where:** `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md:1357-1358`; `docs/verification.md:93-95`; `pyproject.toml:20-27`; at the reviewed tree.
- **Evidence:** Spec 12.6 lists "non-subprocess canaries" per commit and "subprocess canaries" nightly, and `docs/verification.md:95` repeats it. `scripts/test.sh tests/canary -m "not slow and not release and subprocess" --collect-only` gives `17/41 tests collected`: every subprocess canary except `test_compile_time_scaling` (`tests/canary/test_l21_compile_time.py:232-233`, marked `slow`) runs per commit. HANDOFF 4's marker rule, spec 7.1 (line 518) and `docs/environment.md:100` ("holds most canaries") match the code; coverage is larger than stated, not smaller.
- **Fix:** Spec 12.6 per commit: "every canary except the L-21 compile-time benchmark (subprocess canaries included)"; nightly: "the L-21 compile-time benchmark" in place of "subprocess canaries"; `docs/verification.md:93-95` likewise. Text only, with no marker change, so no test is RED; the collect-only command above keeps showing the 17 per-commit subprocess canaries.
- **Hashed:** no.
- **D criterion:** D-5 and D-6. **Ledger row:** none.
- **Raised by:** area-3-opus (M-6) and area-4-opus (M-3), as Minor.
- **Decision:** fix the text to today's placement; no test moves tiers (step 8).

### m-26: Three spec 7 statements that observed behaviour has superseded

- **Where:** `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md:598` (7.4 rule 5), `:511` and `:510` (7.1), at the reviewed tree.
- **Evidence:** Rule 5 gives as its reason that "Catalyst rejects measurements in control regions", while spec 14's L-2 (line 1471), rule 12 (line 613) and HANDOFF 5 rule 5 record that Catalyst skips the measuring rule and silently controls a unitary inverse. Section 7.1 says the "could not be parsed" error means there is no decomposition path, while HANDOFF 3.2 and the L-18 canary's docstring record that a successful verbose compile prints it too. The 7.1 list of what `dfthc.testing.resources` filters omits `use_allocation(...)` and the "unable to determine the branch" warning (`src/dfthc/testing/resources.py:24-26`, `40-44`, `91`).
- **Fix:** Rule 5's reason becomes "because Catalyst skips the measuring rule in a control region and silently controls a unitary inverse where one exists (L-2, rule 12)"; 7.1 says, as HANDOFF 3.2 does, that the text alone proves nothing and the cause is the "Decomposition rule not found for operator '<op>'" line; the filter list gains the two missing entries. Spec text only; no test reads it, so none is RED.
- **Hashed:** no.
- **D criterion:** D-6. **Ledger row:** none.
- **Raised by:** area-4-opus (M-4), as Minor.
- **Decision:** fix.

### m-27: Spec 5.1, 5.2, the front matter and K-9 predate Q1b and the finished project

- **Where:** `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md:6-7`, `321`, `359` and `1634` at the reviewed tree.
- **Evidence:** Spec 5.1's L2 row (line 321) and 5.2's `prepare.py` line (359) omit `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes` (`src/dfthc/ops/prepare.py:704`, `1225`; listed in `docs/architecture.md:19`); the 5.2 tree omits `src/dfthc/ops/_validate.py`, `src/dfthc/testing/canary.py` and `docs/upstream/`. The front matter still says "Status: Written for user review; the implementation plan is written after approval" and "Branch: `docs/design-spec` (local only)". K-9's title, "Qubits above the paper's count", no longer holds in total (spec 10.3: 1,102 against 1,131).
- **Fix:** Add the two classes and the three paths; set Status to "Implemented; source of truth for the code on `main`" and Branch to `main`; retitle K-9 "Qubits per group above the paper's (total below it with Q-1 variant (b))". Spec text only; no test reads it, so none is RED.
- **Hashed:** no.
- **D criterion:** D-6. **Ledger row:** none.
- **Raised by:** area-4-opus (M-5), as Minor.
- **Decision:** fix.

### m-28: `docs/architecture.md` calls L0 "NumPy only" and places `dfthc.registers` in L0 as well as L3

- **Where:** `docs/architecture.md:12-20` (the L0 row at `:17`) at the reviewed tree.
- **Evidence:** `src/dfthc/data/encoded.py:16` imports `scipy.linalg.hadamard`, and `src/dfthc/registers.py:5` imports `pennylane.wires` and implements the L3 `WalkLayout`; spec 5.1 (line 319) says "NumPy and SciPy only" and puts `WalkLayout` in L3, as `docs/architecture.md:20` also does. The duplicate contradicts the page's claim that each layer depends only on lower layers.
- **Fix:** "NumPy and SciPy only", and drop `dfthc.registers` from the L0 modules. Extend the architecture audit in `tests/test_docstrings.py` to reject a module assigned to two layers: RED on today's table, GREEN after.
- **Hashed:** no.
- **D criterion:** D-6. **Ledger row:** none.
- **Raised by:** area-4-opus (M-6) and area-4-gpt (M-1), as Minor.
- **Decision:** fix.

### m-29: HANDOFF 4's "Canaries only" command also runs the slow L-21 benchmark

- **Where:** `docs/agents/HANDOFF.md:222` at the reviewed tree.
- **Evidence:** The row gives `scripts/test.sh tests/canary -q` with no marker, and `pyproject.toml:19` adds no default deselection, so it collects `test_compile_time_scaling` (`slow`; subprocess compiles of 64- to 8,640-entry tables, about 90 s and 1.6 GB), although HANDOFF 3.5 tells agents never to run the nightly tier.
- **Fix:** `scripts/test.sh tests/canary -m "not slow and not release" -q` (40 passed on the reviewed tree); the benchmark keeps its own row (line 223). Shown by `--collect-only`: 41 tests collected by today's command, 40 by the new one.
- **Hashed:** no.
- **D criterion:** D-6. **Ledger row:** none.
- **Raised by:** area-4-opus (M-7), as Minor.
- **Decision:** fix.

### m-30: Spec 13.1 attributes the inner reads' +186 against the paper to the entry width alone; 6 of it is iteration

- **Where:** `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md:1401` (and `:1404`) at the reviewed tree.
- **Evidence:** The spec reads "Inner reads: +186. Our swapped inner entry has d2 = 26 bits ..., against the 20-bit entries of the paper's Table III", but six extra bits times 15 junk entries times two reads is 180. The report's FeMoco-54 `InnerPrepare` row (`docs/resources.md:331`) gives per read `exact-iter` +5, `iter-const` −2, `flat-r` +60 and `sign-bits` +30: 93 per read and 186 for two, of which 180 is width and 6 is iteration. The total is right; only the attribution differs from the report.
- **Fix:** "Inner reads: +186: the 26-bit swapped entry against Table III's 20 bits (+180: `flat-r` +120, `sign-bits` +60) and exact boundaries (+6: `exact-iter` +10, `iter-const` −4)". Attribution text only: no count moves, toward the paper or otherwise. No test reads it, so none is RED.
- **Hashed:** no.
- **D criterion:** D-6. **Ledger row:** none.
- **Raised by:** area-4-opus (M-8), as Minor.
- **Decision:** fix.

### m-31: The documentation test named "..._and_spec_14" never reads spec 14, so spec 14's canary list is unguarded

- **Where:** `tests/test_docstrings.py:143-153` at the reviewed tree.
- **Evidence:** `test_limitations_match_the_canaries_and_spec_14` hard-codes L-1 to L-27 and checks only that the canaries `docs/limitations.md` names exist; it opens no spec file, and `DOCUMENT_PATHS` excludes `superpowers/` (`:116-117`). The reviewer's mutant 2 (spec 14 renames `test_allocating_rule_is_excluded`) survives all 48 documentation tests, while the same rename in `docs/limitations.md` is killed at `:153`. Today the two lists agree (39 names each).
- **Fix:** In that test, parse spec 14 (between "## 14." and "## 15.") for `**L-<n>` and the backticked `test_*` names and assert that both sets equal those of `docs/limitations.md`. It is RED under mutant 2 and GREEN on the reviewed tree.
- **Hashed:** no.
- **D criterion:** D-6 (and D-5). **Ledger row:** none.
- **Raised by:** area-4-opus (M-10), as Minor.
- **Decision:** fix.

### m-32: The Catalyst issue draft gives a malformed PRX identifier

- **Where:** `docs/upstream/catalyst-unused-distributed-candidates.md:278-280` at the reviewed tree.
- **Evidence:** The draft calls the application "PRX 10.1103/pb2g-j9cw"; the spec's reference (lines 1673-1676) gives Phys. Rev. X 15, 041016 (2025), with the DOI `10.1103/pb2g-j9cw`.
- **Fix:** "Phys. Rev. X 15, 041016 (2025), DOI 10.1103/pb2g-j9cw", with a small citation consistency assertion over the upstream drafts that is RED on the current phrase and GREEN after.
- **Hashed:** no.
- **D criterion:** none. **Ledger row:** none.
- **Raised by:** area-4-gpt (M-4), as Minor.
- **Decision:** fix.

## Open ledger rows

One row per row of the plan's table "Open ledger rows and their dispositions", in its order. "Reviewers" are the Opus and GPT reviews of the row's area in that table.

| Ledger row | Task | Disposition | Reviewers | Closure, or the finding that carries it |
| --- | --- | --- | --- | --- |
| L239 | 0.2 | R | Opus declined (the user-level closure is total); GPT Minor | m-6 (fix, hashed) |
| L251 | 1C.1 | R | Opus and GPT declined: already implemented | closed as done in `63820c3` ("Add FlaggedUniformPrep"): `src/dfthc/ops/uniform.py:136-137` passes `phase_grad[:s]` to both rotations, and the prepares pass `phase_grad[:s_u]` (`src/dfthc/ops/prepare.py:199`, `222`, `239`) |
| L252 | 1C.1 | C | Opus and GPT confirmed | Tasks 1C.3 to 1C.5 merged with tests of those contracts: increasing row-major leaf order, a flag of `None` only at zero width without a control, exact nested work (`src/dfthc/ops/unary.py:95-131`, `157-168`) |
| L258 | 1C.3 | R | Opus and GPT declined (optional) | declined: optional and not raised to Important; every abstract key uses the signed `WORD_DTYPE`, concrete arrays are cast to `int64` (`src/dfthc/ops/qroam.py:251-270`), and canary L-26 pins the failure |
| L264 | 1C.4 | A | Opus and GPT confirmed | stays accepted: `PhaseFixup` is constructed in `src/dfthc` only in its docstring example (`src/dfthc/ops/qroam.py:637`); the erases call `emit_phase_fixup` with measured parities, and tape mode rejects any other parities (`qroam.py:571-575`) |
| L266 | 1C.5 | R | Opus Minor (optional); GPT declined | m-3 (declined) |
| L273 | 2D.1 | C | Opus and GPT confirmed | Task 2D.2 merged; `QROAMErase.required_wires` still ignores its table (`src/dfthc/ops/qroam.py:870-879`), so `_AliasSampling.work_sizes` passing `()` stays valid (`src/dfthc/ops/prepare.py:174-178`) |
| L277 | 2E.2 | C | Opus and GPT confirmed | Task 3.1 merged (ledger row 3.1, K-1: every child converts once); one `_select_body` serves both selects, so `AngleErase` gets identical arguments (`src/dfthc/ops/select.py:237-264`) |
| L278 | 2E.2 | R | Opus and GPT Minor | m-4 (fix, hashed) |
| L284 | 2D.2 | C | Opus and GPT confirmed | Task 3.1 merged with its children and work-pool tests; the select's `_WIRES` exclude the registers it must not write (`src/dfthc/ops/select.py:38-39`) |
| L287 | 2E.3 | C | Opus and GPT confirmed | Task 3.1 merged (`test_children_order_and_wires`): `SpinMux`, `SpinSwap`, ladder, Majorana gate, dagger, `SpinSwap`, `SpinMux` (`src/dfthc/ops/select.py:249-263`) |
| L331 | 3.1 | C | Opus and GPT confirmed | Tasks 3.3 and 3.4 merged with their children and work-pool tests; the named registers and work prefixes are in `src/dfthc/ops/walk.py:98-128`, `193-211` |
| L332 | 3.1 | R | Opus and GPT Minor | m-4 (the repeated validation) and m-5 (lengths, not disjointness); fix, hashed |
| L334 | 3.2 | C | Opus and GPT confirmed | Tasks 3.3 to 3.5 merged; E-3 runs after one walk step (`tests/ops/test_walk.py::test_controlled_walk_after_a_walk_step`), and the controlled classes control only the reflections (`src/dfthc/ops/walk.py:106`, `494-500`) |
| L338 | 2E.4 | R | Opus and GPT declined (optional) | declined: optional and not raised to Important; the fan mirrors Qualtran's construction (`src/dfthc/ops/givens.py:124-131`), and skipping `phase_grad[0]` removes CNOTs only, changing the reported Clifford counts through a hashed file |
| L339 | 2E.4 | R | Opus and GPT Minor | m-5 (fix, hashed) |
| L340 | 2E.4 | C | Opus and GPT confirmed | Tasks 3.1 and 3.4 merged: ladder rungs `N - 2, ..., 0` and the dagger `0, ..., N - 2` with leaf adjoints (`src/dfthc/ops/givens.py:279`, `361`); the select passes exact ladder work (`src/dfthc/ops/select.py:252-260`) |
| L344 | 2E.5 | C | Opus and GPT confirmed | Tasks 3.1 and 3.4 merged: `target = psi_dn[:1]`, `flavour = sigma_0` and two work wires (`src/dfthc/ops/select.py:253-259`) |
| L254 | 1A.5 | C | Opus and GPT confirmed | `tests/ops/test_inner_select.py::test_action_against_the_encoded_model` checks every tail row, with both values of `sigma_1`, against the encoded model at `random-C3` on the strict tape path; Task 2E.4 pins the ladder orders (`tests/ops/test_givens.py:143-166`) |
| L255 | 1A.5 | C | Opus and GPT confirmed | the E checks and the real-walk QPE compare with `walk_sim.case(...).block()`, the encoded block with the normalization Λ̃ (`src/dfthc/testing/walk_sim.py:90-93`; `tests/integration/test_qpe_walk.py:106`) |
| L283 | 2D.2 | R | Opus Minor; GPT declined (the certificates prove every outcome) | m-7 (fix) |
| L285 | 2D.2 | C | Opus and GPT confirmed | Task 3.5 added `FAILS_C3` (C = 3, k2 = 0) with `encoded_block` as the oracle; dfthcbe's failure signs would move the expected block by up to 0.329, so E-1 tells the two conventions apart at 1e-10 |
| L311 | 4.1 | C | Opus and GPT confirmed (Opus notes the stale spec 11.3 totals, I-4) | `tests/integration/test_qpe_counts.py:103-131` asserts worst-case T, M and A, the wire total and the inventory with its non-vacuity checks; the report gives A and the qubits per t (`docs/resources.md:418-428`) |
| L313 | 4.1 | C | Opus and GPT confirmed | applied in Task 4.2 and in Q1b.1: the postselected compiled round trip (`128eb16`, `tests/ops/test_unprepare_from_outcomes.py:278-297`) and the postselected compiled feed-forward (`tests/integration/test_qpe_toy.py:245-275`) |
| L314 | 4.1 | R | item 1: GPT Minor, Opus declined; items 2 and 3: Opus Minor, GPT declined | m-10 (item 1, declined), m-8 (item 2, fix) and m-9 (item 3, fix) |
| L322 | 3.4 | C | Opus and GPT confirmed | Task 4.2 merged (`test_inventory_of_the_qpe_program`); the instrument's self-test and negative control are `tests/ops/test_walk.py:213-245` |
| L326 | 3.3 | C | Opus and GPT confirmed | Tasks 3.4 and 3.5 merged: the harness self-test pins S0 and S1, the complex comparison and `project` (`tests/ops/test_block_encoding.py:107-129`), and the block T3 makes no vacuous norm claim |
| L330 | 3.1 | C | Opus and GPT confirmed | E-1 covers it at both test presets, as the row says; the branch classes use the values after the swap (`tests/ops/test_inner_select.py:93-103`) |
| L343 | 2E.5 | C | Opus and GPT confirmed | `test_action_against_the_encoded_model` and `test_dagger_inverts_the_select_on_every_branch` for the selects, a strict tape-path test for each hand-written AND pair (for example `tests/ops/test_majorana.py:71-91`), and the E checks for the walk |
| L230 | 0.1 | R | Opus Minor; GPT declined | m-17 (fix) |
| L231 | 0.1 | R | Opus and GPT Minor | m-13 (fix) |
| L233 | 0.2 | R | Opus Minor; GPT declined | m-18 (fix) |
| L234 | 0.2 | R | Opus and GPT Minor | m-19 (fix) |
| L235 | 0.2 | R | Opus and GPT Minor | m-20 (fix) |
| L236 | 0.2 | R | Opus Minor, narrowed (the stated case no longer reproduces); GPT Minor | m-20 (fix) |
| L237 | 0.2 | R | Opus Minor; GPT declined | m-21 (fix) |
| L238 | 0.2 | R | Opus and GPT declined | declined: the "unable to determine the branch" warning is one flag per compiled module and fires on nearly every program, so it cannot name a dfthc conditional (`src/dfthc/testing/resources.py:40-44`, `90`); the exact T, M and A asserted against `dfthc.costs` at every compiled preset catch an unhinted branch |
| L245 | 1H.1 | W | Opus and GPT confirmed | closed at G5 if it passed in every Phase 5 run; a failure makes it R (both pipes-held cases passed in area-3-opus's run) |
| L246 | 1B.1 | R | Opus and GPT declined (optional) | declined: optional and not raised to Important; the release tier projects at about 25 to 80 minutes against its 8-hour budget (`src/dfthc/testing/sparse_sim.py:57-68`) |
| L247 | 1B.1 | R | Opus and GPT declined (optional) | declined: optional and not raised to Important; the precondition errors already name the operation, the wires and the offending basis key (`src/dfthc/testing/sparse_sim.py:382-409`) |
| L248 | 1B.1 | C | Opus and GPT confirmed (Opus with a wording note) | superseded: the E checks compare whole states, ancillas included, against the independent encoded model, and the renormalization after a measurement can only enlarge an error (a false failure, never a false pass); the closure is the whole-state comparison, not "fails on any norm growth" |
| L250 | 1C.1 | C | Opus and GPT confirmed | reason code `rot-add` (spec 13.3, `REASON_CODES`): + 12 per walk against dfthcbe in `docs/resources.md` |
| L256 | 1C.3 | C | Opus and GPT confirmed | the report's FeMoco-54 rows `QROAMRead@...` with dfthcbe's and the paper's values and their codes (for example `QROAMRead@OuterPrepare` 156 against 151, `exact-iter` + 5) |
| L257 | 1C.3 | C | Opus and GPT confirmed | `clifford_kind`: "upper bound (CNOT, PauliX)" on every read and composite row (`scripts/resource_report.py:590-600`) |
| L261 | 1C.2 | C | Opus and GPT confirmed | the report's uniform-preparation rows with `cmp-free`, `rot-add` and `unif-odd` (`FlaggedUniformPrep@OuterPrepare` −7 against dfthcbe, −6 against the paper) |
| L262 | 1C.4 | C | Opus and GPT confirmed | the report's `PhaseFixup@...` rows with `exact-iter` and `onehot-pad` (+3 and +17 against dfthcbe) |
| L265 | 1A.6 | A | Opus and GPT confirmed | stays accepted: dfthcbe computes the oracle, and the test recomputes the factor digest from the arrays fed to our code (`tests/data/test_dfthcbe_fixtures.py:94-117`) |
| L268 | 1C.5 | C | Opus and GPT confirmed | the report's M column: M per walk 2,524 = 1,672 from the partition + the block encodings' 852 junk measurements, which the summary states |
| L270 | 2D.1 | C | Opus and GPT confirmed | the report's FeMoco-54 rows of the outer prepare (−2) and the outer unprepare from outcomes (−25) with their codes |
| L271 | 2D.1 | D | Opus and GPT confirmed | deferred: listed in `docs/resources.md`, "Deferred refinements" (19 Toffolis per walk for 14 dirty qubits; `scripts/resource_report.py:991-993`) |
| L274 | 2E.2 | C | Opus and GPT confirmed | the report's `AngleErase` rows (102 / 90 / 37) with their codes |
| L276 | 2E.2 | D | Opus and GPT confirmed | deferred: listed in `docs/resources.md`, "Deferred refinements" (12 Toffolis per walk; `scripts/resource_report.py:994-996`) |
| L279 | 2D.2 | C | Opus and GPT confirmed | the report's FeMoco-54 rows of the inner prepare (+58) and the inner unprepare from outcomes (−3) with their codes |
| L280 | 2D.2 | C | Opus and GPT confirmed (Opus with a scope note) | the report's qubit comparison by group (`costs_reference.PAPER_QUBIT_GROUPS_FEMOCO`, Task Q1b), as spec 13.2 asks; dfthcbe's persistent inner qubits are not in the report, and spec 13.2 does not require them |
| L282 | 2D.2 | D | Opus and GPT confirmed | deferred, with L271 (the same bullet of "Deferred refinements") |
| L286 | 2E.3 | C | Opus and GPT confirmed | `docs/resources.md`, FeMoco-54 summary: "The spin multiplexer is 0 in Table III and 1 per application in App. B.3 and B.7.1" |
| L293 | Q1b.2 | R | Opus and GPT Minor, hashed (area-4-opus too) | m-22 (fix, hashed) |
| L289 | 4.5 | R | Opus and GPT Minor, hashed (area-4-opus too) | m-22 (fix, hashed) |
| L317 | 3.5 | T | Opus and GPT confirmed | closed by Task 5.2's release tier, with its wall time and peak |
| L318 | 3.5 | C | Opus and GPT confirmed | the G3 gate run, `-m "not release"` on `18b3b8e`: 1,116 passed (the heavy-runs row of 2026-11-14) |
| L319 | 3.4 | C | Opus and GPT confirmed | spec 13.3's `ref-target`, `ref-unif`, `qpe-ctrl` and `ctrl-ref-t2`; the walk's codes sum to −204 against the paper and +152 against dfthcbe, the controlled walk's to −202 and +153 |
| L323 | 3.3 | C | Opus and GPT confirmed | `ref-unif` (−1 for `RefT2` against the paper) |
| L327 | 3.1 | C | Opus and GPT confirmed | `maj-and` (−2 per Majorana gate against the paper) and `sf-enable` (+1 per `AngleErase` against dfthcbe) |
| L333 | 3.2 | C | Opus and GPT confirmed | the report's separate M and A columns; its header says the paper itemizes neither, and the paper's missing splits are dashes, not zeros |
| L335 | 2E.4 | C | Opus and GPT confirmed | `rz-addsub` in the report's ladder rows (−106 per ladder against the paper) |
| L341 | 2E.5 | C | Opus and GPT confirmed | `maj-and`, as for L327 |
| L272 | 2D.1 | C | Opus and GPT confirmed | spec 13.1's "Outer unprepare and erase" row cites it ("the saving the paper notes after its Eq. (127)", spec line 1383) |
| L275 | 2E.2 | C | Opus and GPT confirmed | spec 13.1's "Angle erase (×2)" row reads as the finding asks (+14 `exact-iter` and `no-prune`, +8 `onehot-pad`, +2 `sf-enable`; spec line 1387) |
| L301 | 4.4 | P | Opus and GPT confirmed | closed at G5, after Phase 5 has followed it (`tests/test_docstrings.py` passed 48 tests on `22d0fe5`) |
| L302 | 4.4 | R | Opus and GPT Minor | m-23 (fix) |
| L321 | 3.4 | C | Opus and GPT confirmed | master plan section 3, item 23 gives 7,449 and 4,795 (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md:188`) |
| L325 | 3.3 | C | Opus and GPT confirmed | as for L321 (7,449) |

## Declined and corrected findings

None. No finding of the eight reviews was declined as wrong or re-rated. The two declines, m-3 (the optional row L266) and m-10 (the release-only input of L314), are declines by rule and stay numbered findings. The merges kept the higher severity: area-1-opus M-3 and area-2-opus M-4, raised as Minor, are part of I-4 at Important. Where one reviewer declined a ledger row that the other rated, the rating stands and the decline is recorded in the finding's "Raised by" (m-3, m-6, m-7, m-8, m-9, m-10, m-17, m-18, m-21).

## T2 to T6 per operator (D-1)

From the area-2 reviews, one row per class audited by `tests/test_docstrings.py` (34). A cell naming another module gives its path. The erase blocks' T3 is spec 12.5's certificate (nightly), and `ControlledWalkPower`'s T3 and T6 are nightly; m-8 adds a per-commit children test.

| Class | Test module | T2 | T3 | T4 | T5 | T6 |
| --- | --- | --- | --- | --- | --- | --- |
| `PhaseGradientRotation` | `tests/ops/test_phase_gradient.py` | `test_rule_is_valid` | `test_rotation_applies_its_phase` | `test_toffoli_count_matches_the_closed_form` | `test_pinned_rule_is_chosen_and_nothing_survives_at_device_level` | `test_pinned_rule_is_chosen_and_nothing_survives_at_device_level` |
| `FlaggedUniformPrep` | `tests/ops/test_uniform.py` | `test_rule_is_valid` | `test_succ_part_is_exactly_uniform` | `test_toffoli_count_matches_the_closed_form` | `test_pinned_rule_is_chosen_and_nothing_survives_at_device_level` | `test_pinned_rule_is_chosen_and_nothing_survives_at_device_level` |
| `QROAMRead` | `tests/ops/test_qroam_read.py` | `test_rule_is_valid` | `test_every_address_loads_loaded_bits` | `test_toffoli_count_at_the_test_presets` | `test_no_dfthc_operator_survives_at_device_level` | `tests/ops/test_angle_loads.py::test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `PhaseFixup` | `tests/ops/test_phase_fixup.py` | `test_rule_is_valid_in_tape_mode` | `test_fixup_applies_exactly_the_parity_table` | `test_toffoli_count_matches_the_closed_form` | `test_no_dfthc_operator_survives_at_device_level` | `test_erase_conditionals_are_hinted` |
| `QROAMErase` | `tests/ops/test_qroam_erase.py` | `test_rule_is_valid_in_tape_mode` | `tests/ops/test_certificates_qroam.py::test_read_erase_round_trip_certificate` | `test_counts_match_the_closed_form` | `test_no_dfthc_operator_survives_at_device_level` | `test_erase_is_not_a_second_read_and_its_conditionals_are_hinted` |
| `ZeroReflection` | `tests/ops/test_zero_reflection.py` | `test_rules_are_valid` | `test_exact_matrix_with_global_sign` | `test_toffoli_count_matches_the_closed_form` | `test_no_dfthc_operator_survives_at_device_level` | `tests/ops/test_ref_t2_ab.py::test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `OuterPrepare` | `tests/ops/test_outer_prepare.py` | `test_rules_are_valid` | `test_prepared_state_is_the_alias_distribution` | `test_counts_match_the_closed_forms` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `OuterUnprepare` | `tests/ops/test_outer_prepare.py` | `test_rules_are_valid` | `tests/ops/test_certificates_prepare.py::test_outer_prepare_unprepare_certificate` | `test_counts_match_the_closed_forms` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `OuterUnprepareFromOutcomes` | `tests/ops/test_unprepare_from_outcomes.py` | `test_rules_are_valid` | `test_round_trip_after_the_junk_measurement` | `test_counts_match_the_closed_forms` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `InnerPrepare` | `tests/ops/test_inner_prepare.py` | `test_rules_are_valid` | `test_every_basis_input_gets_the_sign_of_its_selected_weight` | `test_counts_match_the_closed_forms` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `InnerUnprepare` | `tests/ops/test_inner_prepare.py` | `test_rules_are_valid` | `tests/ops/test_certificates_prepare.py::test_inner_prepare_unprepare_certificate` | `test_counts_match_the_closed_forms` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `InnerUnprepareFromOutcomes` | `tests/ops/test_unprepare_from_outcomes.py` | `test_rules_are_valid` | `test_round_trip_after_the_junk_measurement` | `test_counts_match_the_closed_forms` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `InnerSigns` | `tests/ops/test_inner_prepare.py` | `test_inner_signs_rules_are_valid` | `test_inner_signs_phase_on_every_basis_input` | `test_inner_signs_counts_rules_and_device_level` | `test_inner_signs_counts_rules_and_device_level` | `test_inner_signs_counts_rules_and_device_level` |
| `TailAngleLoad` | `tests/ops/test_angle_loads.py` | `test_rules_are_valid` | `test_tail_load_on_every_address` | `test_toffoli_count_compiled` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `SFAngleLoad` | `tests/ops/test_angle_loads.py` | `test_rules_are_valid` | `test_sf_load_on_every_address` | `test_toffoli_count_compiled` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `SpinMux` | `tests/ops/test_spin.py` | `test_spin_mux_rules_are_valid_with_matrices` | `test_spin_mux_on_every_basis_input` | `test_spin_mux_toffoli_count` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `SpinSwap` | `tests/ops/test_spin.py` | `test_spin_swap_rules_are_valid_with_matrices` | `test_spin_swap_on_every_basis_input` | `test_spin_swap_toffoli_count` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `GivensRotation` | `tests/ops/test_givens.py` | `test_rules_are_valid` | `test_rung_matrix_for_every_angle` | `test_rung_toffoli_count` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `GivensLadder` | `tests/ops/test_givens.py` | `test_rules_are_valid` | `test_ladder_equals_ladder_matrix_and_the_opposite_order_fails` | `test_ladder_toffoli_count` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `GivensLadderDagger` | `tests/ops/test_givens.py` | `test_rules_are_valid` | `test_dagger_is_the_inverse_of_ladder_matrix` | `test_ladder_toffoli_count` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `MajoranaGate` | `tests/ops/test_majorana.py` | `test_rule_is_valid` | `test_action_on_every_basis_input` | `test_toffoli_count` | `test_pinned_rule_is_chosen_and_nothing_survives_at_device_level` | `test_pinned_rule_is_chosen_and_nothing_survives_at_device_level` |
| `AngleErase` | `tests/ops/test_angle_erase.py` | `test_rule_is_valid_in_tape_mode` | `test_round_trip_under_forced_schedules` | `test_counts_at_the_test_presets` | `test_no_dfthc_operator_survives_at_device_level` | `test_erase_is_not_a_second_load_and_its_conditionals_are_hinted` |
| `InnerSelect` | `tests/ops/test_inner_select.py` | `test_rules_are_valid` | `test_action_against_the_encoded_model` | `test_counts` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `InnerSelectDagger` | `tests/ops/test_inner_select.py` | `test_rules_are_valid` | `test_action_against_the_encoded_model` | `test_counts` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `RefT2` | `tests/ops/test_ref_t2_ab.py` | `test_rules_are_valid` | `test_exact_action_with_the_global_sign` | `test_counts` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `ControlledRefT2` | `tests/ops/test_ref_t2_ab.py` | `test_rules_are_valid` | `test_controlled_action_for_three_control_states` | `test_counts` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `RefAB` | `tests/ops/test_ref_t2_ab.py` | `test_rules_are_valid` | `test_exact_action_with_the_global_sign` | `test_counts` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `ControlledRefAB` | `tests/ops/test_ref_t2_ab.py` | `test_rules_are_valid` | `test_controlled_action_for_three_control_states` | `test_counts` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |
| `DFTHCBlockEncoding` | `tests/ops/test_block_encoding.py` | `test_rules_are_valid` | `test_block_against_the_encoded_model` | `test_counts` | `test_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen` |
| `ControlledDFTHCBlockEncoding` | `tests/ops/test_block_encoding.py` | `test_rules_are_valid` | `test_controlled_block_encoding_in_three_control_states` | `test_counts` | `test_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen` |
| `DFTHCWalk` | `tests/ops/test_walk.py` | `test_rules_are_valid` | `test_walk_is_ref_ab_after_the_block_encoding` | `tests/integration/test_walk_counts.py::test_counts_at_the_test_presets` | `test_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen` |
| `ControlledDFTHCWalk` | `tests/ops/test_walk.py` | `test_rules_are_valid` | `test_controlled_walk_in_three_control_states` | `tests/integration/test_walk_counts.py::test_counts_at_the_test_presets` | `test_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen` |
| `ControlledWalkPower` | `tests/test_qpe.py` | `test_controlled_walk_power_rule_is_valid` | `test_controlled_walk_power_in_three_control_states` | `test_controlled_walk_power_counts` | `test_nothing_survives_at_device_level` | `test_pinned_rules_are_chosen` |
| `QPERound` | `tests/test_qpe.py` | `test_qpe_round_rule_is_valid_on_the_toy` | `tests/integration/test_qpe_toy.py::test_exact_distribution_at_t_3` | `test_qpe_round_counts` | `test_nothing_survives_at_device_level` | `tests/integration/test_qpe_toy.py::test_pinned_rules_are_chosen_and_nothing_survives_at_device_level` |

Where the two reviews named different tests, the table keeps one checked in its module: for `PhaseFixup` T6 the count signature `test_erase_conditionals_are_hinted` (area-2-gpt's `test_fixup_applies_exactly_the_parity_table` is the T3 test); for `InnerSelectDagger` T3, `ControlledDFTHCWalk` T3 and `QPERound` T2, T3 and T6 both named tests that are that T.
