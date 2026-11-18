# Known limitations

One entry per limitation of spec section 14, with the behaviour the canary in `tests/canary/`
pins today. A flipped canary is a signal to revisit the mitigation, not a failure to hide. All
observations are from 2026-11-11 (PennyLane 0.46.0.dev151, Catalyst 0.16.0.dev140) unless an entry
gives its own date. Entries marked **changed** differ from the spec's original wording. As of Task
4.4 (2026-11-15), this file consolidates L-1 to L-27, including L-26 (Task 0.4b), L-27 (Task
0.2c) and the ledger's later canary-state updates; Task 6.4 (2026-11-16) added L-28 and corrected
L-12 and L-13, Task 6.2 added L-29. `tests/test_docstrings.py` checks that every limitation has an
entry and that every canary named here exists.

## Decomposition rules, pins, control and templates

- **L-1, rules that allocate are excluded** (catalyst#3243). A rule declaring `work_wires=` is
  excluded even when pinned; with the pin there is no path. Canary:
  `test_allocating_rule_is_excluded`. Mitigation: explicit ancilla arguments (spec 7.4 rule 1).
  `Toffoli`'s `_toffoli_elbow` allocates, so every dfthc `Toffoli` and `CSWAP` is a two-control
  `MultiControlledX` with an explicit zeroed work wire (spec 9.0), whose elbow rule applies when
  `Toffoli` is left out of the gate set; the reflections get one lent zeroed wire for the same
  purpose (spec 9.6). A bare `Toffoli` or `CSWAP` has no decomposition path then (the solver
  failure reads "the rules file could not be parsed", L-18).
- **L-2, measuring adjoint rules under control** (pennylane#9629), **changed**. Inside a control
  region the measuring `Adjoint(Op)` rule is skipped even when pinned, and the solver silently
  controls the unitary inverse of `Op`'s base rule instead (one Toffoli and four PauliX for a
  one-CNOT base rule, no measurement). It fails only when no unitary path exists. Canaries:
  `test_measuring_adjoint_rule_has_no_controlled_path` and
  `test_measuring_adjoint_without_unitary_inverse_has_no_path`. Mitigation: spec 7.4 rules 5 and
  6; never place a measuring operator under `qp.ctrl`. The only modified composite a program
  requests is `C(DFTHCWalk)` (the toy's `C(ToyWalk)`), whose rule is pinned and controls only the
  reflections (spec 7.4 rule 6). Catalyst still builds the generic controlled candidate of
  `C(DFTHCWalk)` one level deep: it requests `C(DFTHCBlockEncoding)`, whose measuring base rule is
  skipped with the expected warning, so the cascade stops there and no measuring rule is used
  under control. The candidate costs +26 % compile memory at naphthalene (584 to 733 MB).
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
  bodies (spec 7.4 rule 4). Since G0 the looped `QROAMRead` (through its `QROM` select),
  `TailAngleLoad`, `SFAngleLoad`, `ControlledWalkPower` and `ToyControlledWalkPower` are never
  adjointed or controlled by construction (spec 7.4 rule 6), which the inventory test checks.
- **L-7, `C(ChangeOpBasis)` loses its saving under capture** (pennylane#10357). Controlling
  `change_op_basis(CNOT, Z, CNOT)` gives two Toffolis instead of zero. Canary:
  `test_controlled_change_op_basis_controls_everything`. Mitigation: the pinned `C(DFTHCWalk)`
  rule, which controls only the walk's two reflections (spec 7.4 rule 6).
- **L-8, generic controlled wrapping adds PauliX** (pennylane#10080). Generic control of a one-CNOT
  rule compiles to one Toffoli plus four PauliX. Canary: `test_generic_ctrl_adds_paulix_per_control`.
  Mitigation: the pinned `C(DFTHCWalk)` and `C(ToyWalk)` rules (spec 7.4 rule 6), which control
  only the reflections and add `X` on the control only for control value 0.
- **L-9, function calls inside control regions** (catalyst#3330, catalyst#3384), **changed**.
  Controlling a `qp.capture.subroutine` leaves `C(PauliX)` undecomposed at `level="user"` (outside
  the gate set), and compiling for execution fails. Canary:
  `test_function_call_in_ctrl_region_raises`. Mitigation: no `Subroutine`; the pinned
  `C(DFTHCWalk)` rule, which opens no control region around the walk (spec 7.4 rule 6).
- **L-10, compilation can depend on gate order** (catalyst#3431). Canary:
  `test_order_dependent_compile`, built from the issue's reproducer in
  `tests/canary/reproducers/catalyst_3431.py`, with the observed outcome per order recorded in
  `catalyst_3431.json`. Mitigation: an occurrence is a compiler blocker; preserve a reproducer
  and escalate (spec 14, L-10).
- **L-11, resource functions run with capture enabled** (pennylane#10162). Canary:
  `test_resource_function_runs_with_capture_enabled`. Mitigation: `dfthc.compile.resource_key`
  builds keys with capture paused.
- **L-12, select-swap `QROM` is not self-inverse and counts symbolically**, **changed**
  (2026-11-16, Task 6.4). A dirty 16-entry `QROM` (select-swap chosen) applied twice has overlap
  0.8125 with the input; its counts are symbolic. Canaries:
  `test_select_swap_qrom_twice_is_not_identity`, `test_select_swap_qrom_counts_are_symbolic`.
  Not being self-inverse is the intended behaviour of a dirty read, which leaves junk for a
  measured erase as `QROAMRead` does. What keeps the select-swap rule out: rule selection is
  cost-based (L-3), so a dirty `QROM` gets whichever rule the solver prefers (at our sizes the
  unary rule, which leaves no junk at all); pinning the select-swap rule fails because it nests a
  `QROM` (L-4); and its counts are symbolic (L-22). Mitigation: `QROAMRead` runs its select stage
  as a clean `QROM` with `c - 1` work wires (depth 1), pinned to the unary-iteration rule, which
  nests no `QROM` (`dfthc.compile.PENNYLANE_PINS`), then its own exact CSWAP network; a
  `PhaseFixup` that applies `CZ`/`Z` directly.
- **L-13, `Select` with shared target wires crashes the compiler** (`'qref.operator' op qubit
  operands must refer to distinct qubits`), **changed** (2026-11-16, Task 6.4). Canary:
  `test_select_with_shared_target_crashes`. Mitigation: no `Select` of operators that share wires.
  `QROM`'s unary rule loads with `MultiX` and never hits L-13, so `QROAMRead`'s select is a
  `QROM`. Our own unary-iteration helper remains for what `QROM` does not offer: address ranges
  with a lower bound (the tail load and the tail fix-up), nested ranges (the SF load and fix-up),
  an enable on value 1, and leaves that apply phases (the fix-ups).
- **L-14, `MultiControlledX` without work wires has no path.** Canary:
  `test_mcx_without_work_wires_has_no_path`. Mitigation: explicit work wires; the reflections'
  `MultiControlledX` (`dfthc.ops.reflections`) gets `max(c - 2, 0)` zeroed work wires for `c`
  controls and, in the walk, one lent zeroed wire (the last `angles` wire).
- **L-19, `UniformPrep` and `AliasSampling` need arbitrary rotations**, **changed**. For a
  non-power-of-two size (L = 5) neither has a path to the gate set. For a power-of-two size
  `AliasSampling` compiles, because its uniform preparation is then Hadamards only. Canaries:
  `test_template_has_no_toffoli_level_path` (two cases) and
  `test_power_of_two_alias_sampling_compiles`. Mitigation: `FlaggedUniformPrep`.
- **L-28, `QROM` takes no measurement values as data in tape mode** (new, found by Task 6.4 on
  2026-11-16). `QROM`'s bitstrings, and the row of its leaf operator `MultiX`, must be bool or
  integer arrays: recording either with `MeasurementValue`s raises `ValueError`. Under capture
  the same data are traced bools, and the erase fix-up's leaf loop as `H . QROM(parities) . H`
  on the one-hot register compiles with exact phases and one Toffoli fewer per outer and inner
  fix-up (the audit's probe 4; three per FeMoco-54 walk). Canary:
  `test_qrom_rejects_measurement_values_in_tape_mode`. Mitigation: the fix-ups keep their
  conditional `CZ` leaves (`emit_phase_fixup`, spec 9.4), which work in both modes (K-10), as the
  sparse simulator, `assert_valid_decomp_rule` and the report's call finder need.
- **L-29, `FlipSign` does not scale to the walk's reflections** (new, found by Task 6.2 on
  2026-11-17). `FlipSign` declares its multi-controlled `Z` with traced control values, and its
  compile time and memory grow exponentially with the number of pattern wires: 20 s at 6 wires,
  63 s at 12, over 6 min and 2.8 GB at 14 (`FlipSign` with `GlobalPhase(pi)`, `n - 3` work
  wires; an all-ones pattern behaves the same). The walk needs 20 and 41 pattern wires at
  MID_NAPHTHALENE and 22 and 46 at FeMoco-54. Generic `qp.ctrl` of `FlipSign` and
  `GlobalPhase(pi)` also fails: `qp.ctrl(GlobalPhase)` has no Toffoli-level path
  (`CompileError`). No canary: the scaling is too heavy to pin. Mitigation: the helpers of
  `dfthc.ops.reflections` build the multi-controlled `Z` from `MultiControlledX` with static
  control values (1.7 s at 8 wires, 1.3 s at 46), and the controlled form takes the control as
  one more control of value 1, then `Z` on the control.

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
