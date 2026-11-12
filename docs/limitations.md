# Known limitations

One entry per limitation of spec section 14, with the behaviour the canary in `tests/canary/`
pins today. A flipped canary is a signal to revisit the mitigation, not a failure to hide. All
observations are from 2026-11-11 (PennyLane 0.46.0.dev151, Catalyst 0.16.0.dev140). Entries marked
**changed** differ from the spec's original wording; Task 0.4 updates spec section 14 to match,
and until then this file is authoritative where the two differ.

## Decomposition rules, pins, control and templates

- **L-1, rules that allocate are excluded** (catalyst#3243). A rule declaring `work_wires=` is
  excluded even when pinned; with the pin there is no path. Canary:
  `test_allocating_rule_is_excluded`. Mitigation: explicit ancilla arguments (spec 7.4 rule 1).
- **L-2, measuring adjoint rules under control** (pennylane#9629), **changed**. Inside a control
  region the measuring `Adjoint(Op)` rule is skipped even when pinned, and the solver silently
  controls the unitary inverse of `Op`'s base rule instead (one Toffoli and four PauliX for a
  one-CNOT base rule, no measurement). It fails only when no unitary path exists. Canary:
  `test_measuring_adjoint_rule_has_no_controlled_path`. Mitigation: spec 7.4 rules 5 and 6; never
  place a measuring operator under `qp.ctrl`.
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
  bodies (spec 7.4 rule 4).
- **L-7, `C(ChangeOpBasis)` loses its saving under capture** (pennylane#10357). Controlling
  `change_op_basis(CNOT, Z, CNOT)` gives two Toffolis instead of zero. Canary:
  `test_controlled_change_op_basis_controls_everything`. Mitigation: explicit `C(DFTHCWalk)` and
  `C(DFTHCBlockEncoding)` rules.
- **L-8, generic controlled wrapping adds PauliX** (pennylane#10080). Generic control of a one-CNOT
  rule compiles to one Toffoli plus four PauliX. Canary: `test_generic_ctrl_adds_paulix_per_control`.
  Mitigation: explicit `C(...)` rules.
- **L-9, function calls inside control regions** (catalyst#3330, catalyst#3384), **changed**.
  Controlling a `qp.capture.subroutine` leaves `C(PauliX)` undecomposed at `level="user"` (outside
  the gate set), and compiling for execution fails. Canary:
  `test_function_call_in_ctrl_region_raises`. Mitigation: no `Subroutine`; explicit `C(...)` rules.
- **L-10, compilation can depend on gate order** (catalyst#3431). Canary:
  `test_order_dependent_compile`, built from the issue's reproducer in
  `tests/canary/reproducers/catalyst_3431.py`, with the observed outcome per order recorded in
  `catalyst_3431.json`. Mitigation: an occurrence is a compiler blocker; preserve a reproducer
  and escalate (spec 14, L-10).
- **L-11, resource functions run with capture enabled** (pennylane#10162). Canary:
  `test_resource_function_runs_with_capture_enabled`. Mitigation: `dfthc.compile.resource_key`
  builds keys with capture paused.
- **L-12, select-swap `QROM` is not self-inverse and counts symbolically.** A dirty 16-entry
  `QROM` (select-swap chosen) applied twice has overlap 0.8125 with the input; its counts are
  symbolic. Canaries: `test_select_swap_qrom_twice_is_not_identity`,
  `test_select_swap_qrom_counts_are_symbolic`. Mitigation: `QROAMRead` with exact CSWAPs and a
  `PhaseFixup` that applies `CZ`/`Z` directly.
- **L-13, `Select` with shared target wires crashes the compiler** (`'qref.operator' op qubit
  operands must refer to distinct qubits`). Canary: `test_select_with_shared_target_crashes`.
  Mitigation: our own unary-iteration helper.
- **L-14, `MultiControlledX` without work wires has no path.** Canary:
  `test_mcx_without_work_wires_has_no_path`. Mitigation: explicit work wires.
- **L-19, `UniformPrep` and `AliasSampling` need arbitrary rotations**, **changed**. For a
  non-power-of-two size (L = 5) neither has a path to the gate set. For a power-of-two size
  `AliasSampling` compiles, because its uniform preparation is then Hadamards only. Canary:
  `test_template_has_no_toffoli_level_path` (two cases). Mitigation: `FlaggedUniformPrep`.

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
