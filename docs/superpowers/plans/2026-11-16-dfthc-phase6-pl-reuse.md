# Phase 6: reuse PennyLane functionality (2026-11-16)

The user reviewed the PennyLane reuse audit (`.scratch/pl-audit/summary.md`, area reports
`.scratch/pl-audit/{A,B,C}/report.md` with probes) and chose five changes. Each task below is one
of them. The aim is fewer blocks of our own wherever PennyLane already has the functionality, with
exact compiled counts kept.

Lanes (existing worktrees, one implementer at a time per lane):

| Lane | Worktree | Tasks, in order |
| --- | --- | --- |
| A | `.worktrees/lane-a` | 6.1, 6.2, 6.3 |
| B | `.worktrees/lane-b` | 6.4, 6.5 |

Each task branches from the current local `main`, is reviewed, passes the per-commit tier (run by
the orchestrator) and is merged into local `main` with `--no-ff` before the lane's next task
starts. After all five tasks, the orchestrator regenerates the resource report (small systems,
fe2s2, fe4s4), recompiles the FeMoco-54 walk and controlled walk once, updates the README, the
spec's decision record and the ledger, and runs the final whole-branch review.

## Global constraints

1. **Exact counts.** T = Toffoli + TemporaryAND, M = MidCircuitMeasure, A = Adjoint(TemporaryAND),
   from `dfthc.compile.compiled_resources`. Every closed form in `dfthc.costs` keeps equal to the
   compiled counts it describes. When a change moves a count, update the closed form, its
   docstring, the tests and the dfthcbe/paper delta explanations (`dfthc.costs_reference` and the
   operator docstrings' **Cost** sections) in the same task. No workarounds whose purpose is to
   match the paper's or dfthcbe's numbers.
2. **Decompositions stay in the qjit call.** Rules are registered with `dfthc.compile.register_rule`
   and lowered by `compiled_resources` (`qjit(capture=True)`, graph `qp.decompose`, `qp.analyze`).
   No decompositions file.
3. **Operator docstrings** keep the project format (`docs/conventions.md`). `docs/architecture.md`,
   `docs/limitations.md` and the spec passages that describe a changed block are updated in the
   same task. Do not edit `docs/resources.md`, `docs/resources.json`, `README.md` or
   `docs/agents/LEDGER.md`; the orchestrator regenerates or updates them after all tasks.
4. **Code comments** state only constraints the code cannot show.
5. **Load limits.** Run only the test files your change touches, with `scripts/test.sh` (default
   two workers), never a whole tier (`-m ...`) or the whole suite; the orchestrator runs the
   per-commit tier on your branch. Every probe compile runs under the watchdog:
   `PYTHONPATH=src <main checkout>/.venv/bin/python -m dfthc.testing.watchdog --cap-mb 3000 --timeout-min 20 -- <cmd>`.
   Do not compile naphthalene, fe2s2, fe4s4 or FeMoco-54 unless your task says so. Do not run
   `scripts/resource_report.py` except with `--compile-one` where your task says so.
6. **Git.** Commit on your task branch in your lane's worktree only. Never push, fetch, run any
   `git remote` command, add or remove worktrees, or touch `main` or other branches. Never request
   elevated permissions. Scratch files go under `.scratch/phase6/` in your worktree.
7. **One code path.** Do not keep an old implementation next to the new one "for safety". Where a
   PennyLane path cannot serve a case the project needs (both tape mode and capture mode, all test
   parameter sets), stop and report instead of adding a second path, unless the task allows it.

## Task 6.1: `qp.registers` inside `WalkLayout`

**Lane A, first.** Audit: `.scratch/pl-audit/C/report.md` (row `WalkLayout`, recommendation 1).

Replace the hand-written register and group bookkeeping in `dfthc.registers.WalkLayout` with
`qp.registers` (`pennylane/registers.py`). The audit verified that
`qp.registers({group: {register: size}})` reproduces every register and group of the H2 layout.

- Build the layout's wires from one `qp.registers` call over `GROUPS` and `register_sizes`. Remove
  what it makes redundant (the `from_params` wire-numbering loop, `_group`, the per-group
  properties, `num_wires`), keeping the public interface that callers use, or updating the callers.
- Keep `register_sizes`, `junk_sizes`, `split_junk`, `work_size`, `split_group` and
  `angle_register`: PennyLane has nothing for traced slices inside rules or for the junk aliasing.
- Wire order and numbering must be unchanged: every layout used in tests (H2 STO-3G test
  parameters, the toy layouts, `walk_sim` cases) gets exactly the same wires for every register
  and group as before. Add a test that pins this for at least the H2 test layout and one layout
  with non-zero `junk_overflow`, comparing against the wire numbers from the current code.

**Acceptance.** `tests/test_registers.py` (or wherever `WalkLayout` is tested) and every test
file that imports `WalkLayout` pass; no compiled count changes.

## Task 6.2: Reflections through `qp.FlipSign`

**Lane A, after 6.1.** Audit: `.scratch/pl-audit/A/report.md` (rows `ZeroReflection` to
`ControlledRefAB`, recommendation 1), probe `.scratch/pl-audit/A/probe2_reflections.py`.

Replace the five reflection operators in `dfthc/ops/reflections.py` (`ZeroReflection`, `RefT2`,
`ControlledRefT2`, `RefAB`, `ControlledRefAB`, with their adjoint and `C(...)` rules) by two
plain helper functions built on PennyLane:

- `ref_t2(succ_x_o, b, m_b, sigma_0, work, control=None)` and
  `ref_ab(x_o, m_x_o, sigma_1, b, m_b, sigma_0, work, control=None)` (keep these argument names;
  the patterns are those of today's `RefT2` and `RefAB`).
- Uncontrolled: `qp.FlipSign(pattern, wires, work_wires=work)` then `qp.GlobalPhase(np.pi)`.
- Controlled: `qp.FlipSign([1, *pattern], [control, *wires], work_wires=work)` then
  `qp.Z(control)`. Generic `qp.ctrl(FlipSign + GlobalPhase)` fails to compile (audit), and
  `FlipSign` without work wires has no path (L-14), so always pass the work wires.
- Matching resource helpers for the block encoding's and walk's `register_resources` functions.
- The block encoding, the walk and their controlled versions call the helpers in place of the
  classes. Nothing adjoints a reflection: `FlipSign` and the global phase are self-inverse up to
  the declared phase, so drop the adjoint rules. If something in the codebase does adjoint the
  block encoding generically, report it.

Counts: T and A must stay exactly as today for every reflection, uncontrolled and controlled, and
for every block encoding and walk (audit: n_w = 8 gives 6 / 5 uncontrolled, 7 / 6 controlled).
Clifford counts may change (the audit saw PauliX 32 against 18). Find out whether that is a real
gate difference or `qp.analyze` counting conditional flips at their worst branch (L-17), and
update every test that pins X, CNOT or other Clifford counts with that explanation in the test or
docstring. The unitary must stay the same: keep or adapt the dense checks (2|s><s| - I, and the
controlled form) from `tests/ops/test_zero_reflection.py` and `tests/ops/test_ref_t2_ab.py`.

Also update: `dfthc.costs` (the `ZeroReflection` closed form becomes the reflection closed form,
same values), `dfthc.compile` mentions, `dfthc.testing.resources`, `toy_walk`, `walk_sim`,
`scripts/resource_report.py` (rows keep the names `RefT2`, `RefAB`, `ControlledRefT2`,
`ControlledRefAB` but compile the helpers; drop the `ZeroReflection` rows), the report tests,
`docs/architecture.md`, and `docs/limitations.md` (L-14's mitigation now names `FlipSign`; add the
generic-`ctrl` failure as a limitation with a canary only if the existing canary suite has the
pattern for it; otherwise mention it in L-14).

**Acceptance.** Every test file that mentioned a removed class passes (rewritten where it tested
the class), plus `tests/ops/test_walk.py`, `tests/ops/test_block_encoding.py`,
`tests/integration/test_qpe_counts.py`, `tests/report/test_report.py`, `tests/test_compile.py`,
`tests/test_work_pool.py`, `tests/test_docstrings.py`. T/M/A of every walk and block encoding
unchanged.

## Task 6.3: One `C(DFTHCWalk)` rule

**Lane A, after 6.2.** Audit: `.scratch/pl-audit/C/report.md` (row `ControlledDFTHCBlockEncoding,
ControlledDFTHCWalk`, recommendation 2), probe `.scratch/pl-audit/C/p6_ctrl_walk.py` and its log.

Replace `ControlledDFTHCBlockEncoding` and `ControlledDFTHCWalk` (`dfthc/ops/walk.py`) with one
rule registered for `"C(DFTHCWalk)"`. Users write `qp.ctrl(DFTHCWalk(...), control)`. The rule
keeps today's construction: every step uncontrolled except the two reflections, which use the
6.2 helpers with `control`. It must honour the control value (today's classes handle control
value 0 under capture, see L-28 and its tests); handle the rule's `control_wires`,
`control_values` and `work_wires` arguments, and fail loudly for more than one control wire.
Pin the rule through `compiled_resources`' fixed decompositions if the graph could otherwise pick
a generic controlled rule. Do the same for the toy walk (`ControlledToyWalk` becomes a
`C(ToyWalk)` rule) so the toy keeps mirroring the real walk.

Update every caller: `dfthc.qpe` (`ControlledWalkPower` and `WALK_POWERS`), `toy_walk`,
`walk_sim`, `dfthc.testing.resources`, `scripts/resource_report.py` (the controlled rows compile
`qp.ctrl(DFTHCWalk(...))`), tests and docs.

**Memory check (required before reporting DONE).** Decision 13 / K-1: generic controlled
candidates once made up 95-99 % of the IR at FeMoco size. The audit measured +30 % memory at H2
(450 against 346 MB). Measure the naphthalene controlled walk on the base commit and on your head,
alone, each under the watchdog with `--cap-mb 3000 --timeout-min 20`, using
`scripts/resource_report.py --compile-one naphthalene <row>` (find the controlled-walk row name).
Report wall time and peak memory of both and the counts. If the head needs more than 1.5 times
the base's peak memory, stop and report DONE_WITH_CONCERNS with the numbers.

**Acceptance.** Counts of the controlled walk and controlled block encoding unchanged (T/M/A,
wires) at every size the tests cover; the control-value-0 tests pass; QPE tests
(`tests/integration/test_qpe_counts.py`, `tests/test_qpe*.py` or equivalents), walk, toy,
report and docstring tests pass.

## Task 6.4: Alias reads and fix-ups through `qp.QROM`

**Lane B, first.** Audit: `.scratch/pl-audit/B/report.md` (rows `QROAMRead`, `QROAMEmitter`,
`PhaseFixup`, recommendations 1, 2 and 4), probes `.scratch/pl-audit/B/probe1_qrom_unary.py`,
`probe2_selectswap.py`, `probe3_compile_scale.py`, `probe4_fixup_qrom.py` and their JSON.

1. **Alias reads.** Rebuild the select stage of the outer and inner alias reads (`QROAMRead` as
   used by `OuterPrepare` and `InnerPrepare`) on `qp.QROM(clean=True)` over the block columns,
   followed by our existing CSWAP network. Pass exactly c - 1 work wires so the depth is 1 and
   only the unary rule applies. Pin `QROM` to PennyLane's `_qrom_unary_iteration` through a
   fixed-decomps entry in `compiled_resources` (not `register_rule`; that rule nests no `QROM`,
   so L-4 does not bite). Pass bool column data. The audit measured the same state, one Toffoli
   less per read, and a faster compile at 300 leaves.
   - The inner read flattens `x_o` and the top `b` bit into contiguous columns. That is exact only
     when `K2 >= 2**(w_b - 1)`. Check every parameter set the tests and the report use. If some
     need a different addressing, find the cheapest exact QROM-based form; if none exists, stop
     and report (do not keep the old read as a second path).
   - The angle loads keep using our unary iteration (ranges and nesting); keep only the
     `QROAMRead` / `QROAMEmitter` code they still need and delete what the alias reads no longer
     use (word packing, `leaf_words_from_table`, ... as far as the angle loads allow).
2. **Fix-ups.** Rebuild the leaf loop of the outer, inner and SF fix-ups as
   H . `QROM`(parity matrix) . H on the one-hot (probe 4: exact phases, T 3 against 4). The tail
   fix-up stays on `emit_phase_fixup` (its range starts at RC). First probe tape mode:
   `emit_phase_fixup` must work in both modes (spec 9.x `PhaseFixup`, K-10: in tape mode the
   parities are `MeasurementValue`s). If `QROM` cannot take them in tape mode, leave the fix-ups
   unchanged, record the probe and the reason in `docs/limitations.md`, and continue with part 1
   only.
3. **Counts and docs.** Update `dfthc.costs` (FeMoco-54 expectation from the audit: -6 Toffolis
   per walk with both parts), `dfthc.costs_reference` deltas, docstrings, `docs/architecture.md`,
   spec sections 4.2 and the `QROAMRead` / `PhaseFixup` entries, and the two `docs/limitations.md`
   corrections from the audit: L-13's mitigation (the real reasons are ranges and nesting; `QROM`'s
   unary rule never hits L-13) and L-12's framing (not self-inverse is the intended read
   behaviour; the real blockers are cost-based rule selection, L-4 and symbolic counts).
4. **Compile scale.** Measure the compile time and memory of the inner alias read alone
   (`InnerPrepare` row) for naphthalene with `scripts/resource_report.py --compile-one`, on the
   base and on your head, alone, under the watchdog. Report both.

**Acceptance.** Exact state equivalence of the new reads against the old ones (tests through the
simulator certificates the project already uses for `QROAMRead` and the prepares), exact phases
of the fix-ups in both modes, closed forms equal to compiled counts for every test parameter set,
and all prepare, unprepare, qroam, walk, report and docstring tests pass.

## Task 6.5: Angle registers of b_rot - 1 bits

**Lane B, after 6.4.** Audit: `.scratch/pl-audit/A/report.md` (recommendation 2 and its open
question).

The stored angles satisfy `x < 2**(b_rot - 1)` (`dfthc/data/givens.py:50-51` or wherever the
quantisation lives), so the top bit of each `b_rot`-bit angle register is always 0. PennyLane's
THC uses beth-bit angles with a beth+1-bit gradient, and `SemiAdder` costs `len(y_wires) - 1`
whatever the length of `x_wires`.

1. **Probe first** and write the result to the report before changing code: confirm that no
   operator reads or writes the top angle bit (loads, `GivensRotation`, `AngleErase`, spec 6.3,
   6.7), that the Givens rotation with a (b_rot - 1)-bit angle register and the unchanged gradient
   gives the same unitary (dense check at a small size) and the same T, and that every junk
   register still fits in `angles ‖ junk_overflow` (FeMoco-54: `junk_b` = 390 against 742). If
   any of this fails, stop and report.
2. Shrink each angle register to `b_rot - 1` wires (`WalkLayout.register_sizes`, `angle_register`,
   the angle loads, `AngleErase`, `SFAngleLoad`, `TailAngleLoad`, the ladders, `costs`,
   `required_wires` everywhere). Keep the angle tables and the dfthcbe data unchanged.
3. Update closed forms (qubits: N - 1 fewer, FeMoco-54 1,102 to 1,049 expected; M: N - 1 fewer
   per `AngleErase`, check the exact number), `dfthc.costs_reference` deltas against the paper's
   1,131 qubits, docstrings, `docs/architecture.md`, spec sections 6.x and 10.x on the angle
   register and the qubit count.

**Acceptance.** Dense or certificate checks of the Givens rotation, the ladders and the angle
loads and erase pass; closed forms equal compiled counts; all select, ladder, angle, registers,
walk, report and docstring tests pass.
