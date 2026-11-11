# DFTHC Phase 0, Part 2: Canary Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin today's behaviour of every known PennyLane and Catalyst limitation of spec section 14 (L-1 to L-23) and of the two found during plan dry-runs (L-24, L-25) in focused tests, and document each in `docs/limitations.md`.

**Architecture:** One canary module per group of limitations in `tests/canary/`. Canaries that may abort the process, that exercise "no decomposition path", or that depend on import-time state run their check in a fresh interpreter through `dfthc.testing.canary.run_snippet`. Every canary asserts the behaviour observed on 2026-11-11; a flipped canary is a signal to revisit the mitigation.

**Tech Stack:** pytest, `dfthc.compile`, `dfthc.testing.canary` (Task 0.2), PennyLane and Catalyst nightlies.

This file holds Tasks 0.3 and 0.4 of the master plan. Every canary below ran on 2026-11-11 against PennyLane `0.46.0.dev151` and Catalyst `0.16.0.dev140` in the scratch replica and passed as stated, except `test_order_dependent_compile`, whose reproducer must be fetched from the issue (Task 0.3, Step 3). Canaries are characterization tests: there is no implementation step, so the "red" step of TDD is replaced by "write the canary, run it, and if it does not pass, record the observed behaviour".

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
3. Every argument that determines circuit structure is in `compilable_argnames`; `static_argnames` are not used; `dynamic_argnames` only for runtime data (`PhaseFixup` parities, `QPERound` earlier bits, K-1 remedy tables).
4. Bodies of auto-adjointed operators contain no `qp.for_loop` or `qp.cond`; `qp.cond` only on mid-circuit measurement outcomes in erase operators and the QPE wrapper; `qp.for_loop` only in `ControlledWalkPower`.
5. Measurement-based uncompute lives only in `QROAMErase`, `AngleErase` and explicit `Adjoint(Op)` rules with signature `(base)`; an operator that contains a measurement is never placed under `qp.ctrl`.
6. Controlled versions come only from explicit `C(Op)` rules with signature `(base, control_wires, control_values, work_wires, work_wire_type)`.
7. One base rule per operator; explicit adjoint and controlled rules are pinned (§7.5).
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
- Per commit: `scripts/test.sh -m "not slow and not release"` (15 min). Nightly: `scripts/test.sh -m "slow and not release"` (4 h). Release: `scripts/test.sh` (12 h).

**Git (spec 15)**
- Local branches and worktrees only. Merges go into local `main`. Never push, never modify remotes. The repo-local git identity (`dwierichs` / `david.wierichs@xanadu.ai`) is shared by all worktrees.
- Each task runs in a git worktree on a branch named `phase<N>/<task-id>` from local `main`; only the orchestrator merges.

---

## File structure

| Path | Task | Limitations |
| --- | --- | --- |
| `tests/canary/__init__.py` | none | not created: `tests/canary` is a rootdir-relative test directory without `__init__.py` |
| `tests/canary/test_l01_to_l05_rules.py` | 0.3 | L-1 to L-5 |
| `tests/canary/test_l06_to_l11_control.py` | 0.3 | L-6 to L-11 |
| `tests/canary/test_l12_l13_l14_l19_templates.py` | 0.3 | L-12, L-13, L-14, L-19 (and L-22, shared) |
| `tests/canary/test_l23_l24_l25_arguments.py` | 0.3 | L-23, L-24, L-25 |
| `tests/canary/reproducers/catalyst_3431.py`, `catalyst_3431.json` | 0.3 | L-10 reproducer and recorded outcomes |
| `tests/canary/test_l15_to_l17_measure_analyze.py` | 0.4 | L-15, L-16, L-17 (and L-22, shared) |
| `tests/canary/test_l18_l20_warnings_runtime.py` | 0.4 | L-18, L-20 |
| `tests/canary/test_l21_compile_time.py`, `compile_baseline.json` | 0.4 | L-21 benchmark |
| `docs/limitations.md` | 0.3 creates, 0.4 appends | every limitation |
| `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md` | 0.4 | section 14 (and 7.4, 8.4) updated for changed canaries |

---

### Task 0.3: Canary suite A: rules, pins, control, templates and arguments

**Task id:** `0.3` · **Branch:** `phase0/0.3` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Scout:** Grok 4.7 (`grok-4.7-high-fast`) for the L-10 reproducer · **Depends on:** 0.2

**Files:**
- Create: `tests/canary/test_l01_to_l05_rules.py`, `tests/canary/test_l06_to_l11_control.py`, `tests/canary/test_l12_l13_l14_l19_templates.py`, `tests/canary/test_l23_l24_l25_arguments.py`, `tests/canary/reproducers/catalyst_3431.py`, `tests/canary/reproducers/catalyst_3431.json`, `docs/limitations.md`

**Interfaces:**
- Consumes: `dfthc.compile.GATE_SET`; `dfthc.testing.canary.run_snippet(body, *, export_warning_variable=True, timeout=900.0)` and `outcome(result, label)`; the snippet prelude's `counts(circuit, level="user") -> dict[str, str]` and `report(label, fn)`.
- Produces: 20 canary tests with the names of spec section 14 (plus the new L-24 and L-25 canaries); `docs/limitations.md` with its first two sections; the reproducer interface for L-10: a module defining `ORDERS: tuple[str, ...]` and `build(order: str)` returning a qjit-compiled QNode, and a JSON file `{"issue": "<url>", "outcomes": {"<order>": "OK" | "ERROR"}}`.

**Acceptance tests:** all 20 tests pass, with the observations of 2026-11-11:
- L-1 `test_allocating_rule_is_excluded`: unpinned counts `{"PauliX": "1", "PauliZ": "1"}` (the plain rule); pinned to the `work_wires=` rule: `CompileError` naming `L01Op`.
- L-2 `test_measuring_adjoint_rule_has_no_controlled_path`: `C(Adjoint(L02Load))` compiles to `{"PauliX": "4", "Toffoli": "1"}`: the measuring rule is skipped.
- L-3 `test_unweighted_solver_prefers_unitary_inverse`: `{"CNOT": 2, "Hadamard": 1}`.
- L-4 `test_pin_applies_to_nested_instances`: unpinned `QROM` compiles; pinned to `_select_swap` fails naming `QROM`.
- L-5 `test_rule_branch_on_dynamic_value_is_not_specialized`: `{"PauliZ": 1}`.
- L-6 `test_auto_adjoint_with_nested_control_flow`: compiles to `{"PauliX": "4"}` (crash not reproduced).
- L-7 `test_controlled_change_op_basis_controls_everything`: two Toffolis and one CZ.
- L-8 `test_generic_ctrl_adds_paulix_per_control`: `{"PauliX": 4, "Toffoli": 1}`.
- L-9 `test_function_call_in_ctrl_region_raises`: `{"C(PauliX)": "1"}` at `level="user"`; execution fails to compile.
- L-10 `test_order_dependent_compile`: each order's outcome matches `catalyst_3431.json`.
- L-11 `test_resource_function_runs_with_capture_enabled`: every recorded call saw capture enabled.
- L-12 `test_select_swap_qrom_twice_is_not_identity` (overlap 0.8125) and `test_select_swap_qrom_counts_are_symbolic`.
- L-13 `test_select_with_shared_target_crashes`: "qubit operands must refer to distinct qubits".
- L-14 `test_mcx_without_work_wires_has_no_path`: fails naming `MultiControlledX`.
- L-19 `test_template_has_no_toffoli_level_path[UNIFORM_PREP|ALIAS_SAMPLING]` at L = 5: both fail with a missing rule.
- L-23 `test_dynamic_list_argument_not_canonicalized`, L-24 `test_compilable_int_must_fit_int64`, L-25 `test_adjoint_of_class_traces_positional_arguments`.

**Definition of done:** `scripts/test.sh tests/canary -q` gives `20 passed` (about 40 s); `docs/limitations.md` holds the "Decomposition rules, pins, control and templates" and "Operator arguments" sections; one commit.

- [ ] **Step 1: Write canaries L-1 to L-5**

Create `tests/canary/test_l01_to_l05_rules.py`:

```python
"""Canaries L-1 to L-5: rule exclusion, measuring adjoints, cost-based choice, pins, abstract rules.

Each test pins today's behaviour (spec section 14). A flipped canary is a signal to revisit the
mitigation, not a failure to hide.
"""

import jax
import pennylane as qp
import pytest
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled]


@pytest.mark.subprocess
def test_allocating_rule_is_excluded():
    """L-1: a rule that declares work_wires= is excluded even when pinned (catalyst#3243)."""
    result = run_snippet(
        """
        class L01Op(Operator2):
            wire_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires)

        @qp.register_resources({qp.X: 1}, work_wires={"zeroed": 1})
        def _l01_declares_work(wires):
            qp.X(wires[0])

        @qp.register_resources({qp.X: 1, qp.Z: 1})
        def _l01_plain(wires):
            qp.X(wires[0])
            qp.Z(wires[0])

        qp.add_decomps(L01Op, _l01_declares_work, _l01_plain)

        def build(pins):
            @qp.qjit(capture=True)
            @qp.decompose(gate_set=GATE_SET, fixed_decomps=pins)
            @qp.qnode(qp.device("null.qubit", wires=1))
            def circuit():
                L01Op([0])
                return qp.probs(wires=[0])
            return circuit

        report("UNPINNED", lambda: counts(build(None)))
        report("PINNED", lambda: counts(build({"L01Op": _l01_declares_work})))
        """
    )
    assert outcome(result, "UNPINNED") == 'OK {"PauliX": "1", "PauliZ": "1"}'
    pinned = outcome(result, "PINNED")
    assert pinned.startswith("ERROR CompileError")
    assert "Decomposition rule not found for operator 'L01Op" in pinned


@pytest.mark.subprocess
def test_measuring_adjoint_rule_has_no_controlled_path():
    """L-2: inside a control region the measuring Adjoint(Op) rule is skipped, even when pinned.

    On 2026-11-11 the solver then silently controls the unitary inverse of Op's base rule (one
    Toffoli, no measurement) instead of failing; it fails only if no unitary path exists. Hence
    spec 7.4 rule 5: an operator that contains a measurement is never placed under qp.ctrl.
    """
    result = run_snippet(
        """
        class L02Load(Operator2):
            wire_argnames = ("ctrl", "target")

            def __init__(self, ctrl, target):
                super().__init__(ctrl, target)

        @qp.register_resources({qp.CNOT: 1})
        def _l02_load(ctrl, target):
            qp.CNOT([ctrl[0], target[0]])

        def _l02_unload_resources(base):
            return {qp.H: 1, qp.ops.MidMeasure(Wire[1], reset=True): 1, qp.Z: 1}

        @qp.register_resources(_l02_unload_resources)
        def _l02_unload(base):
            qp.H(base.target[0])
            outcome = qp.measure(base.target[0], reset=True)
            qp.cond(outcome, qp.Z)(base.ctrl[0])

        qp.add_decomps(L02Load, _l02_load)
        qp.add_decomps("Adjoint(L02Load)", _l02_unload)

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=GATE_SET, fixed_decomps={"Adjoint(L02Load)": _l02_unload})
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit():
            qp.ctrl(qp.adjoint(L02Load([0], [1])), control=2)
            return qp.probs(wires=[0])

        report("CTRL_ADJOINT", lambda: counts(circuit))
        """
    )
    assert outcome(result, "CTRL_ADJOINT") == 'OK {"PauliX": "4", "Toffoli": "1"}'


class L03Load(Operator2):
    wire_argnames = ("ctrl", "target")

    def __init__(self, ctrl, target):
        super().__init__(ctrl, target)


@qp.register_resources({qp.CNOT: 1})
def _l03_load(ctrl, target):
    qp.CNOT([ctrl[0], target[0]])


def _l03_unload_resources(base):
    return {qp.H: 1, qp.ops.MidMeasure(Wire[1], reset=True): 1, qp.Z: 1}


@qp.register_resources(_l03_unload_resources)
def _l03_unload(base):
    qp.H(base.target[0])
    outcome_ = qp.measure(base.target[0], reset=True)
    qp.cond(outcome_, qp.Z)(base.ctrl[0])


qp.add_decomps(L03Load, _l03_load)
qp.add_decomps("Adjoint(L03Load)", _l03_unload)


def test_unweighted_solver_prefers_unitary_inverse():
    """L-3: with unweighted counts the solver picks the auto inverse over the measured rule."""
    unweighted = {"CNOT", "CZ", "Hadamard", "PauliZ", "PauliX", "MidMeasure", "GlobalPhase"}

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=unweighted)
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit():
        qp.H(0)
        L03Load([0], [1])
        qp.adjoint(L03Load([0], [1]))
        return qp.probs(wires=[0])

    counts = dict(qp.analyze(circuit, level="user")().resources.counts)
    assert counts == {"CNOT": 2, "Hadamard": 1}


@pytest.mark.subprocess
def test_pin_applies_to_nested_instances():
    """L-4: pinning QROM to select-swap makes the QROM nested in that rule unsatisfiable."""
    result = run_snippet(
        """
        from pennylane.templates.subroutines.qrom import _select_swap

        bits = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

        def build(pins):
            @qp.qjit(capture=True)
            @qp.decompose(gate_set=GATE_SET, fixed_decomps=pins)
            @qp.qnode(qp.device("null.qubit", wires=9))
            def circuit():
                qp.QROM(bits, control_wires=[0, 1, 2], target_wires=[3],
                        work_wires=[4, 5, 6, 7, 8], clean=True)
                return qp.probs(wires=[0])
            return circuit

        report("UNPINNED", lambda: counts(build(None)))
        report("PINNED", lambda: counts(build({"QROM": _select_swap})))
        """
    )
    assert outcome(result, "UNPINNED").startswith("OK")
    pinned = outcome(result, "PINNED")
    assert pinned.startswith("ERROR CompileError")
    assert "Decomposition rule not found for operator 'QROM" in pinned


class L05Flagged(Operator2):
    dynamic_argnames = ("flag",)
    wire_argnames = ("wires",)

    def __init__(self, flag, wires):
        super().__init__(flag, wires)


@qp.register_resources({qp.Z: 1})
def _l05_branching(flag, wires):
    if isinstance(flag, jax.core.Tracer):
        qp.Z(wires[0])
    elif flag:
        qp.X(wires[0])
    else:
        qp.Y(wires[0])


qp.add_decomps(L05Flagged, _l05_branching)


def test_rule_branch_on_dynamic_value_is_not_specialized():
    """L-5: rules see dynamic arguments as abstract values, so value branches never specialize."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set={"PauliX", "PauliY", "PauliZ"})
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        L05Flagged(1, [0])
        return qp.probs(wires=[0])

    assert dict(qp.analyze(circuit, level="user")().resources.counts) == {"PauliZ": 1}
```

- [ ] **Step 2: Write canaries L-6 to L-11**

Create `tests/canary/test_l06_to_l11_control.py`:

```python
"""Canaries L-6 to L-11: adjoint control flow, controlled wrapping, order dependence, capture.

Each test pins today's behaviour (spec section 14). L-6 and L-9 record behaviour observed on
2026-11-11 that differs from the spec's original claim; docs/limitations.md says so.
"""

import json
from pathlib import Path

import pennylane as qp
import pytest
from pennylane.core import Operator2

from dfthc.compile import GATE_SET
from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled]

REPRODUCERS = Path(__file__).parent / "reproducers"


@pytest.mark.subprocess
def test_auto_adjoint_with_nested_control_flow():
    """L-6 (catalyst#3217, open): an auto-adjointed rule with nested for_loop and cond.

    On 2026-11-11 this reproducer compiles; the IRMapping crash of the issue was not reproduced.
    The mitigation (no control flow in auto-adjointed bodies) stays while the issue is open.
    """
    result = run_snippet(
        """
        class L06Inner(Operator2):
            wire_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires)

        @qp.register_resources({qp.X: 2})
        def _l06_inner(wires):
            @qp.for_loop(2)
            def loop(i):
                qp.cond(i == 1, qp.X)(wires[0])

            loop()

        qp.add_decomps(L06Inner, _l06_inner)

        class L06Outer(Operator2):
            wire_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires)

        @qp.register_resources({L06Inner(Wire[1]): 2})
        def _l06_outer(wires):
            @qp.for_loop(2)
            def loop(i):
                L06Inner([wires[0]])

            loop()

        qp.add_decomps(L06Outer, _l06_outer)

        @qp.qjit(capture=True)
        @qp.decompose(gate_set=GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.adjoint(L06Outer([0]))
            return qp.probs(wires=[0])

        report("ADJOINT", lambda: counts(circuit))
        """
    )
    assert outcome(result, "ADJOINT") == 'OK {"PauliX": "4"}'


def test_controlled_change_op_basis_controls_everything():
    """L-7 (pennylane#10357): under capture, control of change_op_basis controls all three parts."""

    def compute():
        qp.CNOT([0, 1])

    def target():
        qp.Z(1)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("null.qubit", wires=3))
    def circuit():
        qp.ctrl(lambda: qp.change_op_basis(compute, target, compute), control=2)()
        return qp.probs(wires=[0])

    counts = dict(qp.analyze(circuit, level="user")().resources.counts)
    assert counts["Toffoli"] == 2  # the saving would leave the two CNOTs uncontrolled
    assert counts["CZ"] == 1


class L08Op(Operator2):
    wire_argnames = ("wires",)

    def __init__(self, wires):
        super().__init__(wires)


@qp.register_resources({qp.CNOT: 1})
def _l08_cnot(wires):
    qp.CNOT([wires[0], wires[1]])


qp.add_decomps(L08Op, _l08_cnot)


def test_generic_ctrl_adds_paulix_per_control():
    """L-8 (pennylane#10080): generic control of a one-CNOT rule adds PauliX gates.

    An explicit C(...) rule would emit one Toffoli and nothing else.
    """

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("null.qubit", wires=3))
    def circuit():
        qp.ctrl(L08Op([0, 1]), control=[2])
        return qp.probs(wires=[0])

    assert dict(qp.analyze(circuit, level="user")().resources.counts) == {
        "PauliX": 4, "Toffoli": 1,
    }


@pytest.mark.subprocess
def test_function_call_in_ctrl_region_raises():
    """L-9 (catalyst#3330, #3384): a function call (subroutine) inside a ctrl region.

    On 2026-11-11 the user-level pass leaves the controlled body undecomposed (C(PauliX), outside
    the gate set), and compiling for execution fails with the "no decomposition path" error.
    """
    result = run_snippet(
        """
        @qp.capture.subroutine
        def body(wires):
            qp.X(wires[0])

        def build(device):
            @qp.qjit(capture=True)
            @qp.decompose(gate_set=GATE_SET)
            @qp.qnode(qp.device(device, wires=2))
            def circuit():
                qp.ctrl(body, control=1)([0])
                return qp.probs(wires=[0])
            return circuit

        report("ANALYZE", lambda: counts(build("null.qubit")))
        report("EXECUTE", lambda: np.asarray(build("lightning.qubit")()).tolist())
        """
    )
    assert outcome(result, "ANALYZE") == 'OK {"C(PauliX)": "1"}'
    assert outcome(result, "EXECUTE").startswith("ERROR CompileError")


@pytest.mark.subprocess
def test_order_dependent_compile():
    """L-10 (catalyst#3431): whether a circuit compiles can depend on gate order.

    reproducers/catalyst_3431.py is the issue's reproducer, adapted to define ``ORDERS`` (the
    labels of the gate orders it compares) and ``build(order)``, which returns a qjit-compiled
    QNode. reproducers/catalyst_3431.json maps each order to the outcome recorded when the
    canary was written, "OK" or "ERROR"; the issue reports that they differ.
    """
    source = (REPRODUCERS / "catalyst_3431.py").read_text()
    recorded = json.loads((REPRODUCERS / "catalyst_3431.json").read_text())["outcomes"]
    loop = "\nfor order in ORDERS:\n    report(f'ORDER_{order}', lambda: counts(build(order)))\n"
    result = run_snippet(source + loop)
    for order, expected in recorded.items():
        assert outcome(result, f"ORDER_{order}").startswith(expected), order


class L11Probe(Operator2):
    wire_argnames = ("wires",)

    def __init__(self, wires):
        super().__init__(wires)


_L11_CAPTURE_STATES: list[bool] = []


def _l11_resources(wires):
    _L11_CAPTURE_STATES.append(qp.capture.enabled())
    return {qp.X: 1}


@qp.register_resources(_l11_resources)
def _l11_rule(wires):
    qp.X(wires[0])


qp.add_decomps(L11Probe, _l11_rule)


def test_resource_function_runs_with_capture_enabled():
    """L-11 (pennylane#10162): Catalyst evaluates resource functions with capture enabled."""
    _L11_CAPTURE_STATES.clear()

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        L11Probe([0])
        return qp.probs(wires=[0])

    qp.analyze(circuit, level="user")()
    assert _L11_CAPTURE_STATES and all(_L11_CAPTURE_STATES)
```

- [ ] **Step 3: Add the L-10 reproducer from catalyst#3431**

The reproducer is external input; it could not be fetched while writing this plan. Dispatch the scout (prompt template "Scout" in the master plan) with the question: "Return, verbatim, the reproducer code and the two gate orders it compares from https://github.com/PennyLaneAI/catalyst/issues/3431 (`gh issue view 3431 -R PennyLaneAI/catalyst --json title,body,comments`)." The `gh` call needs network approval in the agent sandbox.

Adapt the returned code into `tests/canary/reproducers/catalyst_3431.py` with this interface. The module is appended to the canary prelude, so `qp`, `np`, `Operator2`, `Wire`, `GATE_SET`, `counts` and `report` are already defined. It contains no top-level calls:

```python
# Reproducer of https://github.com/PennyLaneAI/catalyst/issues/3431, adapted to the canary
# interface of tests/canary/test_l06_to_l11_control.py::test_order_dependent_compile.
ORDERS = ("<label of the order that compiles in the issue>", "<label of the other order>")


def build(order):
    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("null.qubit", wires=...))  # the issue's wire count
    def circuit():
        ...  # the issue's operations, in the gate order selected by `order`
        return qp.probs(wires=[0])

    return circuit
```

Use the issue's own gate set if it differs from `GATE_SET`, and keep its operations exactly. Order labels are single words (no spaces). Then record today's outcome for each order:

```bash
"$VENV/bin/python" - <<'EOF'
import json, os, sys
os.environ["CATALYST_SILENCE_RULE_LOWERING_WARNINGS"] = "0"
sys.path.insert(0, "src")
from dfthc.testing.canary import run_snippet
source = open("tests/canary/reproducers/catalyst_3431.py").read()
loop = "\nfor order in ORDERS:\n    report(f'ORDER_{order}', lambda: counts(build(order)))\n"
result = run_snippet(source + loop)
outcomes = {line.split()[0].removeprefix("ORDER_"): line.split()[1]
            for line in result.stdout.splitlines() if line.startswith("ORDER_")}
record = {"issue": "https://github.com/PennyLaneAI/catalyst/issues/3431", "outcomes": outcomes}
with open("tests/canary/reproducers/catalyst_3431.json", "w") as handle:
    handle.write(json.dumps(record, indent=2) + "\n")
print(record)
EOF
```

Expected: a dictionary with one order `OK` and the other `ERROR`, for example `{'issue': '...', 'outcomes': {'<order a>': 'OK', '<order b>': 'ERROR'}}` (this recorder was dry-run with a stand-in reproducer). Then `scripts/test.sh tests/canary/test_l06_to_l11_control.py -k order_dependent -q` gives `1 passed`. If both are `OK`, the bug no longer reproduces: keep the JSON as recorded, mark L-10 **changed** in `docs/limitations.md` ("the issue's reproducer compiles in both orders on <date>"), and report DONE_WITH_CONCERNS so the orchestrator updates spec section 14.

- [ ] **Step 4: Write canaries L-12, L-13, L-14 and L-19**

Create `tests/canary/test_l12_l13_l14_l19_templates.py`:

```python
"""Canaries L-12, L-13, L-14 and L-19: PennyLane templates (spec section 14)."""

import numpy as np
import pennylane as qp
import pytest

from dfthc.compile import GATE_SET
from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled]

_BITS = np.random.default_rng(3).integers(0, 2, size=(16, 1))


def _dirty_qrom():
    qp.QROM(_BITS, control_wires=[0, 1, 2, 3], target_wires=[4], work_wires=[5, 6, 7], clean=False)


def test_select_swap_qrom_twice_is_not_identity():
    """L-12: the select-swap QROM (chosen for this dirty QROM) is not self-inverse."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("lightning.qubit", wires=8))
    def circuit():
        for wire in range(4):
            qp.H(wire)
        _dirty_qrom()
        _dirty_qrom()
        return qp.state()

    state = np.asarray(circuit())
    initial = np.zeros(256, dtype=complex)
    initial[[k << 4 for k in range(16)]] = 0.25
    assert abs(np.vdot(initial, state) - 0.8125) < 1e-9


def test_select_swap_qrom_counts_are_symbolic():
    """L-12 and L-22: qp.analyze counts of the select-swap QROM are symbolic."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("null.qubit", wires=8))
    def circuit():
        _dirty_qrom()
        return qp.probs(wires=[0])

    assert qp.analyze(circuit, level="user")().resources.is_symbolic


@pytest.mark.subprocess
def test_select_with_shared_target_crashes():
    """L-13: Select whose operators share a target wire crashes the compiler."""
    result = run_snippet(
        """
        @qp.qjit(capture=True)
        @qp.decompose(gate_set=GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=5))
        def circuit():
            qp.Select([qp.X(4), qp.Z(4), qp.Y(4), qp.H(4)], control=[0, 1], work_wires=[2, 3])
            return qp.probs(wires=[0])

        report("SELECT", lambda: counts(circuit))
        """
    )
    select = outcome(result, "SELECT")
    assert select.startswith("ERROR CompileError")
    assert "qubit operands must refer to distinct qubits" in select


@pytest.mark.subprocess
def test_mcx_without_work_wires_has_no_path():
    """L-14: MultiControlledX with four controls and no work wires has no path."""
    result = run_snippet(
        """
        @qp.qjit(capture=True)
        @qp.decompose(gate_set=GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=5))
        def circuit():
            qp.MultiControlledX(wires=[0, 1, 2, 3, 4])
            return qp.probs(wires=[0])

        report("MCX", lambda: counts(circuit))
        """
    )
    mcx = outcome(result, "MCX")
    assert mcx.startswith("ERROR CompileError")
    assert "Decomposition rule not found for operator 'MultiControlledX" in mcx


@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("label", "statement"),
    [
        ("UNIFORM_PREP", "qp.UniformPrep(5, target_wires=[0, 1, 2], work_wires=[3, 4, 5])"),
        ("ALIAS_SAMPLING", "qp.AliasSampling((0.1, 0.2, 0.3, 0.2, 0.2), 3, target_wires=[0, 1, 2], "
                           "temp_wires=list(range(3, 15)), work_wires=list(range(15, 21)))"),
    ],
)
def test_template_has_no_toffoli_level_path(label, statement):
    """L-19: for a non-power-of-two size L = 5, both need arbitrary rotations, so no path exists.

    For a power-of-two size (L = 4) AliasSampling compiles to the gate set (observed 2026-11-11),
    because its uniform preparation is then Hadamards only.
    """
    result = run_snippet(
        f"""
        @qp.qjit(capture=True)
        @qp.decompose(gate_set=GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=21))
        def circuit():
            {statement}
            return qp.probs(wires=[0])

        report("{label}", lambda: counts(circuit))
        """
    )
    template = outcome(result, label)
    assert template.startswith("ERROR CompileError")
    assert "Decomposition rule not found for operator" in template
```

- [ ] **Step 5: Write canaries L-23, L-24 and L-25**

Create `tests/canary/test_l23_l24_l25_arguments.py`:

```python
"""Canaries L-23, L-24 and L-25: operator arguments (spec section 14).

L-23 (pennylane#10060): the base class does not canonicalize list-valued dynamic arguments.
L-24 and L-25 were found while dry-running the Phase 0 plan on 2026-11-11.
L-24: compilable integer arguments are lowered as signed 64-bit values; larger values fail to
compile. Tables therefore store wide entries as tuples of 32-bit words (plan Task 1A.4).
L-25: under capture, qp.adjoint(OpClass)(*args) traces positional arguments, so a compilable
argument passed positionally is rejected; the instance form qp.adjoint(OpClass(...)) works.
"""

import pennylane as qp
import pytest
from pennylane.core import Operator2

from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled]


@pytest.mark.subprocess
def test_compilable_int_must_fit_int64():
    """L-24: 2**63 - 1 compiles as a compilable argument, 2**63 + 1 does not."""
    result = run_snippet(
        """
        class L24Table(Operator2):
            compilable_argnames = ("table",)
            wire_argnames = ("wires",)

            def __init__(self, table, wires):
                super().__init__(table, wires)

        @qp.register_resources({qp.X: 1})
        def _l24_rule(table, wires):
            if table[0] % 2:
                qp.X(wires[0])

        qp.add_decomps(L24Table, _l24_rule)

        def build(value):
            @qp.qjit(capture=True)
            @qp.decompose(gate_set={"PauliX"})
            @qp.qnode(qp.device("null.qubit", wires=1))
            def circuit():
                L24Table((value,), [0])
                return qp.probs(wires=[0])
            return circuit

        report("INT64_MAX", lambda: counts(build(2**63 - 1)))
        report("ABOVE", lambda: counts(build(2**63 + 1)))
        """
    )
    assert outcome(result, "INT64_MAX") == 'OK {"PauliX": "1"}'
    assert outcome(result, "ABOVE").startswith("ERROR CompileError")


class L25Repeat(Operator2):
    compilable_argnames = ("n",)
    wire_argnames = ("wires",)

    def __init__(self, n, wires):
        super().__init__(n, wires)


@qp.register_resources(lambda n, wires: {qp.X: n})
def _l25_rule(n, wires):
    for _ in range(n):
        qp.X(wires[0])


qp.add_decomps(L25Repeat, _l25_rule)


def _adjoint_program(form):
    @qp.qjit(capture=True)
    @qp.decompose(gate_set={"PauliX"})
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        if form == "positional":
            qp.adjoint(L25Repeat)(3, [0])
        else:
            qp.adjoint(L25Repeat(3, [0]))
        return qp.probs(wires=[0])

    return circuit


def test_adjoint_of_class_traces_positional_arguments():
    """L-25: the positional form is rejected; the instance form keeps the argument concrete."""
    with pytest.raises(ValueError, match="must be a concrete, compile-time constant"):
        qp.analyze(_adjoint_program("positional"))()
    counts = dict(qp.analyze(_adjoint_program("instance"))().resources.counts)
    assert counts == {"PauliX": 3}


class L23Bits(Operator2):
    dynamic_argnames = ("bits",)
    wire_argnames = ("wires",)

    def __init__(self, bits, wires):
        super().__init__(bits, wires)


def test_dynamic_list_argument_not_canonicalized():
    """L-23 (pennylane#10060): the base class keeps a list-valued dynamic argument as a list."""
    op = L23Bits([True, False, True], wires=[0])
    assert isinstance(op.arguments["bits"], list)
```

- [ ] **Step 6: Run the canaries**

Run: `scripts/test.sh tests/canary -q`
Expected: `20 passed`. A canary that fails pins behaviour that changed after 2026-11-11: do not edit its assertion silently. Record the new observation in the report, update the assertion and the `docs/limitations.md` entry (marked **changed**), and report DONE_WITH_CONCERNS.

- [ ] **Step 7: Write `docs/limitations.md`**

Create `docs/limitations.md` with exactly this content (Task 0.4 appends the remaining section):

```markdown
# Known limitations

One entry per limitation of spec section 14, with the behaviour the canary in `tests/canary/`
pins today. A flipped canary is a signal to revisit the mitigation, not a failure to hide. All
observations are from 2026-11-11 (PennyLane 0.46.0.dev151, Catalyst 0.16.0.dev140). Entries marked
**changed** differ from the spec's original wording; spec section 14 is updated to match.

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
- **L-6, adjoint lowering with nested control flow** (catalyst#3217, open), **changed**. The
  canary's reproducer (an auto-adjointed rule with nested `for_loop`, `cond` and a nested
  operator) compiles; the IRMapping crash was not reproduced. Canary:
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
```

- [ ] **Step 8: Lint and commit**

Run: `"$VENV/bin/ruff" check tests/canary` · Expected: `All checks passed!`

```bash
git add tests/canary docs/limitations.md
git commit -m "Add canaries L-1 to L-14, L-19 and L-23 to L-25"
```

---

### Task 0.4: Canary suite B: measurements, analysis, warnings and compile time

**Task id:** `0.4` · **Branch:** `phase0/0.4` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 0.3 (it appends to `docs/limitations.md`; the canary code itself needs only 0.2)

**Files:**
- Create: `tests/canary/test_l15_to_l17_measure_analyze.py`, `tests/canary/test_l18_l20_warnings_runtime.py`, `tests/canary/test_l21_compile_time.py`, `tests/canary/compile_baseline.json`
- Modify: `docs/limitations.md` (append), `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md` (sections 7.4, 8.4 and 14)

**Interfaces:**
- Consumes: as Task 0.3.
- Produces: 19 canary tests; `tests/canary/compile_baseline.json` mapping each L-21 case to `{"seconds", "cpu_seconds", "max_rss_mb"}`; the update switch `DFTHC_UPDATE_COMPILE_BASELINE=1`; spec section 14 matching the canaries.

**Acceptance tests:** 14 per-commit canaries and 5 slow benchmark cases pass:
- L-15: `test_hw_like_postselection_wrong_under_capture` ("hw-like is not supported"); `test_mcm_inside_cond_missed` (mean within 0.125 of 0.5); `test_pauli_measure_postselect_ignored` (mean below 0.75).
- L-16: `test_device_pass_runs` (`{"TemporaryAND": 1}` at user level; a Toffoli and no `TemporaryAND` at device level); `test_device_level_has_no_operator_op`.
- L-17: `test_analyze_requires_qjit`; `test_unhinted_cond_counts_max_branch_with_warning` (`{"PauliX": 2, "PauliY": 1}` and a warning); `test_branch_prob_hint_gives_expected_counts` (`{"PauliX": 1, "PauliY": 1}` and `{"PauliX": 2}`); `test_dynamic_loop_counts_are_symbolic`; `test_num_iters_hint_makes_loop_concrete` (`{"PauliZ": 7}`).
- L-18: `test_warnings_silenced_without_env` (`OK 0`); `test_warnings_visible_with_env` (more than 0); `test_unreachable_op_reports_missing_rule`.
- L-20: `test_undecomposed_operator_op_reaches_runtime` (user level `{"L20WithRule": "1"}`, device level `{"PauliX": "1"}`, result `[0.0, 1.0]`, no rule: `CompileError`).
- L-21: `test_compile_time_scaling[tables_1_size_64 | tables_2_size_64 | tables_4_size_64 | tables_1_size_1024 | tables_1_size_8640]` within twice the baseline.

**Definition of done:** `scripts/test.sh -m "not slow and not release" tests/canary -q` gives `34 passed, 5 deselected` (with Task 0.3 merged); `scripts/test.sh -m "slow and not release" tests/canary -q` gives `5 passed`; the baseline JSON is committed; `docs/limitations.md` and spec section 14 updated; one commit.

- [ ] **Step 1: Write canaries L-15 to L-17**

Create `tests/canary/test_l15_to_l17_measure_analyze.py`:

```python
"""Canaries L-15 to L-17: measurement bugs, device-level decomposition, qp.analyze behaviour.

Each test pins today's behaviour (spec section 14). Two L-15 claims behave differently on
2026-11-11 than when the spec was written; the docstrings and docs/limitations.md say how.
"""

import warnings

import numpy as np
import pennylane as qp
import pytest
from pennylane.core import Operator2

from dfthc.compile import GATE_SET
from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled]


@pytest.mark.subprocess
def test_hw_like_postselection_wrong_under_capture():
    """L-15 (pennylane#8379): hardware-like one-shot postselection under capture.

    On 2026-11-11 qjit(capture=True) rejects postselect_mode="hw-like" outright.
    """
    result = run_snippet(
        """
        @qp.qjit(capture=True)
        @qp.set_shots(100)
        @qp.qnode(qp.device("lightning.qubit", wires=2), mcm_method="one-shot",
                  postselect_mode="hw-like")
        def circuit():
            qp.H(0)
            qp.measure(0, postselect=1)
            return qp.sample(wires=[0])

        report("HW_LIKE", lambda: np.asarray(circuit()).tolist())
        """
    )
    hw_like = outcome(result, "HW_LIKE")
    assert hw_like.startswith("ERROR CompileError")
    assert "hw-like is not supported" in hw_like


def test_mcm_inside_cond_missed():
    """L-15 (catalyst#2382): a measurement nested in qp.cond under one-shot execution.

    On 2026-11-11 the nested measurement is executed on every shot (mean about 1/2), so this
    reproducer no longer shows the bug; the spec still avoids measurements inside qp.cond.
    """

    @qp.qjit(capture=True, seed=2026)
    @qp.set_shots(400)
    @qp.qnode(qp.device("lightning.qubit", wires=3), mcm_method="one-shot")
    def circuit():
        qp.X(0)
        first = qp.measure(0)

        def nested():
            qp.H(1)
            second = qp.measure(1)
            qp.cond(second, qp.X)(2)

        qp.cond(first, nested)()
        return qp.sample(wires=[2])

    mean = float(np.mean(np.asarray(circuit())))
    assert abs(mean - 0.5) < 0.125  # five standard deviations at 400 shots


def test_pauli_measure_postselect_ignored():
    """L-15 (catalyst#3338): qp.pauli_measure(postselect=...) is ignored."""

    @qp.qjit(capture=True, seed=2026)
    @qp.set_shots(400)
    @qp.qnode(qp.device("lightning.qubit", wires=1), mcm_method="one-shot")
    def circuit():
        qp.H(0)
        qp.pauli_measure("Z", wires=[0], postselect=1)
        return qp.sample(wires=[0])

    mean = float(np.mean(np.asarray(circuit())))
    assert mean < 0.75  # an honoured postselection would give 1.0


def test_device_pass_runs():
    """L-16 (catalyst#3275): executing on lightning.qubit decomposes again to device gates."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("lightning.qubit", wires=3))
    def circuit():
        qp.TemporaryAND([0, 1, 2])
        return qp.probs(wires=[0])

    user = dict(qp.analyze(circuit, level="user")().resources.counts)
    device = dict(qp.analyze(circuit, level="device")().resources.counts)
    assert user == {"TemporaryAND": 1}
    assert "TemporaryAND" not in device
    assert device.get("Toffoli") == 1


class L16Op(Operator2):
    wire_argnames = ("wires",)

    def __init__(self, wires):
        super().__init__(wires)


@qp.register_resources({qp.X: 1, qp.CNOT: 1})
def _l16_rule(wires):
    qp.X(wires[0])
    qp.CNOT([wires[0], wires[1]])


qp.add_decomps(L16Op, _l16_rule)


def test_device_level_has_no_operator_op():
    """L-16: at level="device" a decomposed Operator2 leaves no named operation behind."""

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit():
        L16Op([0, 1])
        return qp.probs(wires=[1])

    device = dict(qp.analyze(circuit, level="device")().resources.counts)
    assert "L16Op" not in device
    assert np.allclose(np.asarray(circuit()), [0.0, 1.0])


def test_analyze_requires_qjit():
    """L-17: qp.analyze accepts only qjit-compiled QNodes."""

    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        qp.X(0)
        return qp.probs(wires=[0])

    with pytest.raises(ValueError, match="can only be applied to a qjit'd QNode"):
        qp.analyze(circuit)()


def _branch_circuit(hinted):
    def true_fn():
        qp.X(0)
        qp.X(0)

    def false_fn():
        qp.Y(0)

    if hinted:
        true_fn = qp.hint({"branch-prob": 0.25})(true_fn)
        false_fn = qp.hint({"branch-prob": 0.75})(false_fn)

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit(pred: bool):
        qp.cond(pred, true_fn, false_fn)()
        return qp.probs(wires=[0])

    return circuit


def test_unhinted_cond_counts_max_branch_with_warning():
    """L-17: an unhinted conditional counts its most expensive branch and warns."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        resources = qp.analyze(_branch_circuit(hinted=False))(True).resources
    assert dict(resources.counts) == {"PauliX": 2, "PauliY": 1}
    assert any("unable to determine the branch" in str(r.message) for r in records)


def test_branch_prob_hint_gives_expected_counts():
    """L-17 (pennylane#10230): hinted conditionals count their expected gates.

    The expected count of each gate type is summed over the program and rounded up once:
    four one-X conditionals at probability 1/2 count 2 PauliX, and the 1/4-versus-3/4 branch
    pair counts ceil(0.5) = 1 PauliX and ceil(0.75) = 1 PauliY.
    """
    pair = qp.analyze(_branch_circuit(hinted=True))(True).resources
    assert dict(pair.counts) == {"PauliX": 1, "PauliY": 1}

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("null.qubit", wires=2))
    def four_conditionals(first: bool, second: bool):
        for pred, wire in ((first, 0), (second, 1), (first, 0), (second, 1)):

            @qp.hint({"branch-prob": 0.5})
            def flip(wire=wire):
                qp.X(wire)

            qp.cond(pred, flip)()
        return qp.probs(wires=[0])

    four = qp.analyze(four_conditionals)(True, False).resources
    assert dict(four.counts) == {"PauliX": 2}


def _loop_circuit(hinted):
    def body(_):
        qp.Z(0)

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit(n: int):
        loop = qp.for_loop(n)(body)
        if hinted:
            loop = qp.hint({"num-iters": 7})(loop)
        loop()
        return qp.probs(wires=[0])

    return circuit


def test_dynamic_loop_counts_are_symbolic():
    """L-17 and L-22: a loop with a runtime bound gives symbolic counts."""
    assert qp.analyze(_loop_circuit(hinted=False))(3).resources.is_symbolic


def test_num_iters_hint_makes_loop_concrete():
    """L-17: a num-iters hint makes the loop count concrete."""
    resources = qp.analyze(_loop_circuit(hinted=True))(3).resources
    assert not resources.is_symbolic
    assert dict(resources.counts) == {"PauliZ": 7}
```

- [ ] **Step 2: Write canaries L-18 and L-20**

Create `tests/canary/test_l18_l20_warnings_runtime.py`:

```python
"""Canaries L-18 and L-20: rule-lowering warnings and undecomposed operators (spec section 14)."""

import pytest

from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled, pytest.mark.subprocess]

_COUNT_RULE_WARNINGS = """
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning

@qp.qjit(capture=True)
@qp.decompose(gate_set=GATE_SET)
@qp.qnode(qp.device("null.qubit", wires=3))
def circuit():
    qp.Toffoli([0, 1, 2])
    qp.CZ([0, 1])
    return qp.probs(wires=[0])

def rule_warnings():
    with warnings.catch_warnings(record=True) as records:
        qp.analyze(circuit, level="user")()
    return sum(issubclass(r.category, RuleLoweringWarning) for r in records)

report("RULE_WARNINGS", rule_warnings)
"""


def test_warnings_silenced_without_env():
    """L-18 (catalyst#3190): without the variable, Catalyst ignores RuleLoweringWarning."""
    result = run_snippet(_COUNT_RULE_WARNINGS, export_warning_variable=False)
    assert outcome(result, "RULE_WARNINGS") == "OK 0"


def test_warnings_visible_with_env():
    """L-18: with CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 exported, the warnings appear."""
    result = run_snippet(_COUNT_RULE_WARNINGS, export_warning_variable=True)
    count = outcome(result, "RULE_WARNINGS")
    assert count.startswith("OK ") and int(count.split()[1]) > 0


def test_unreachable_op_reports_missing_rule():
    """L-18 (catalyst#2767): no path gives the misleading "could not be parsed" error.

    The real cause, "Decomposition rule not found for operator '<op>'", is in the same message.
    """
    result = run_snippet(
        """
        @qp.qjit(capture=True)
        @qp.decompose(gate_set=GATE_SET)
        @qp.qnode(qp.device("null.qubit", wires=4))
        def circuit():
            qp.ctrl(qp.adjoint(qp.TemporaryAND([0, 1, 2])), control=3)
            return qp.probs(wires=[0])

        report("UNREACHABLE", lambda: counts(circuit))
        """
    )
    unreachable = outcome(result, "UNREACHABLE")
    assert unreachable.startswith("ERROR CompileError")
    assert "the rules file could not be parsed" in unreachable
    assert "Decomposition rule not found for operator 'C(Adjoint(TemporaryAND))" in unreachable


def test_undecomposed_operator_op_reaches_runtime():
    """L-20 (catalyst#3430): an Operator2 listed in the gate set and left undecomposed.

    On 2026-11-11 it does not reach runtime as a bare named gate: with a rule, the device pass
    decomposes it and the result is correct; without a rule, compiling for execution fails.
    """
    result = run_snippet(
        """
        class L20WithRule(Operator2):
            wire_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires)

        @qp.register_resources({qp.X: 1})
        def _l20_rule(wires):
            qp.X(wires[0])

        qp.add_decomps(L20WithRule, _l20_rule)

        class L20NoRule(Operator2):
            wire_argnames = ("wires",)

            def __init__(self, wires):
                super().__init__(wires)

        def build(op_class):
            @qp.qjit(capture=True)
            @qp.decompose(gate_set=GATE_SET | {op_class.__name__: 1})
            @qp.qnode(qp.device("lightning.qubit", wires=1))
            def circuit():
                op_class([0])
                return qp.probs(wires=[0])
            return circuit

        report("WITH_RULE_USER", lambda: counts(build(L20WithRule)))
        report("WITH_RULE_DEVICE", lambda: counts(build(L20WithRule), level="device"))
        report("WITH_RULE_RUN", lambda: np.asarray(build(L20WithRule)()).tolist())
        report("NO_RULE_RUN", lambda: np.asarray(build(L20NoRule)()).tolist())
        """
    )
    assert outcome(result, "WITH_RULE_USER") == 'OK {"L20WithRule": "1"}'
    assert outcome(result, "WITH_RULE_DEVICE") == 'OK {"PauliX": "1"}'
    assert outcome(result, "WITH_RULE_RUN") == "OK [0.0, 1.0]"
    assert outcome(result, "NO_RULE_RUN").startswith("ERROR CompileError")
```

- [ ] **Step 3: Write the L-21 benchmark**

Create `tests/canary/test_l21_compile_time.py`:

```python
"""Canary and benchmark L-21: compile cost grows with distinct tables and table size.

Each case compiles in a fresh process and records wall time, CPU time (Python plus the Catalyst
compiler subprocess) and peak resident memory. The test fails if CPU time or memory exceeds
twice the stored baseline. Refresh the baseline at every nightly bump, serially:

    DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 \
        scripts/test.sh tests/canary/test_l21_compile_time.py
"""

import json
import os
from pathlib import Path

import pytest

from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled, pytest.mark.subprocess, pytest.mark.slow]

BASELINE = Path(__file__).parent / "compile_baseline.json"
CASES = {
    "tables_1_size_64": (1, 64),
    "tables_2_size_64": (2, 64),
    "tables_4_size_64": (4, 64),
    "tables_1_size_1024": (1, 1024),
    "tables_1_size_8640": (1, 8640),
}

_SNIPPET = """
import resource
import sys
import time


class L21Table(Operator2):
    compilable_argnames = ("table",)
    wire_argnames = ("wires",)

    def __init__(self, table, wires):
        super().__init__(table, wires)


def _l21_resources(table, wires):
    return {qp.CNOT: sum(bin(value).count("1") for value in table)}


@qp.register_resources(_l21_resources)
def _l21_rule(table, wires):
    for value in table:
        for bit in range(4):
            if (value >> bit) & 1:
                qp.CNOT([wires[0], wires[1 + bit]])


qp.add_decomps(L21Table, _l21_rule)
rng = np.random.default_rng(21)
tables = [tuple(int(v) for v in rng.integers(0, 16, size=SIZE)) for _ in range(COUNT)]


@qp.qjit(capture=True)
@qp.decompose(gate_set=GATE_SET)
@qp.qnode(qp.device("null.qubit", wires=5))
def circuit():
    for table in tables:
        L21Table(table, [0, 1, 2, 3, 4])
    return qp.probs(wires=[0])


def measure():
    start = time.perf_counter()
    qp.analyze(circuit, level="user")()
    wall = time.perf_counter() - start
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if sys.platform == "darwin" else 1024
    cpu = own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime
    return {
        "seconds": round(wall, 2),
        "cpu_seconds": round(cpu, 2),
        "max_rss_mb": round(max(own.ru_maxrss, children.ru_maxrss) * scale / 2**20, 1),
    }


report("MEASURE", measure)
"""


@pytest.mark.parametrize("case", sorted(CASES))
def test_compile_time_scaling(case):
    count, size = CASES[case]
    result = run_snippet(f"COUNT = {count}\nSIZE = {size}\n" + _SNIPPET, timeout=3600)
    measured = json.loads(outcome(result, "MEASURE").removeprefix("OK "))
    if os.environ.get("DFTHC_UPDATE_COMPILE_BASELINE") == "1":
        baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        baseline[case] = measured
        BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
        return
    reference = json.loads(BASELINE.read_text())[case]
    assert measured["cpu_seconds"] <= 2 * reference["cpu_seconds"], (case, measured, reference)
    assert measured["max_rss_mb"] <= 2 * reference["max_rss_mb"], (case, measured, reference)
```

- [ ] **Step 4: Run the per-commit canaries**

Run: `scripts/test.sh -m "not slow and not release" tests/canary -q`
Expected: `34 passed, 5 deselected` (20 from Task 0.3, including L-10, plus 14 here; the 5 deselected are the L-21 cases).

- [ ] **Step 5: Record the compile-time baseline, serially**

Run: `DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 scripts/test.sh tests/canary/test_l21_compile_time.py -q`
Expected: `5 passed` in about 80 s, and `tests/canary/compile_baseline.json` written. For reference, the dry-run on the development laptop recorded:

```json
{
  "tables_1_size_1024": {
    "cpu_seconds": 5.67,
    "max_rss_mb": 408.7,
    "seconds": 1.87
  },
  "tables_1_size_64": {
    "cpu_seconds": 2.11,
    "max_rss_mb": 289.7,
    "seconds": 0.42
  },
  "tables_1_size_8640": {
    "cpu_seconds": 59.02,
    "max_rss_mb": 1367.2,
    "seconds": 20.73
  },
  "tables_2_size_64": {
    "cpu_seconds": 2.33,
    "max_rss_mb": 297.0,
    "seconds": 0.49
  },
  "tables_4_size_64": {
    "cpu_seconds": 2.74,
    "max_rss_mb": 312.3,
    "seconds": 0.65
  }
}
```

Then check it: `scripts/test.sh -m "slow and not release" tests/canary -q` · Expected: `5 passed`.

- [ ] **Step 6: Append the remaining limitations**

Append exactly this to `docs/limitations.md`:

```markdown

## Measurements, analysis, device level, warnings and compile time

- **L-15, measurement-related bugs.**
  - Hardware-like postselection (pennylane#8379), **changed**: `qp.qjit(capture=True)` now rejects
    `postselect_mode="hw-like"` with a `CompileError`. Canary:
    `test_hw_like_postselection_wrong_under_capture`.
  - Measurements nested in `qp.cond` (catalyst#2382), **changed**: under one-shot execution the
    canary's nested measurement runs on every shot (mean 0.5 within five standard deviations);
    the bug was not reproduced. Canary: `test_mcm_inside_cond_missed`.
  - `qp.pauli_measure(postselect=...)` is ignored (catalyst#3338): the postselected outcome
    appears on about half the shots. Canary: `test_pauli_measure_postselect_ignored`.
  - Mitigation for all three: spec 12.2 (no hardware-like postselection, no measurements inside
    `qp.cond`, no `pauli_measure` postselection).
- **L-16, execution decomposes again to the device gate set** (catalyst#3275). At
  `level="device"` a `TemporaryAND` becomes a Toffoli and PauliX gates; a decomposed `Operator2`
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
- **L-21, compile cost grows with distinct tables and table size** (catalyst#3173). Benchmark:
  `test_compile_time_scaling` (nightly) compiles 1, 2 and 4 distinct 64-entry tables and single
  tables of 64, 1,024 and 8,640 entries, and fails above twice `tests/canary/compile_baseline.json`
  in CPU time or peak memory. On the development laptop one 8,640-entry table took 21 s wall,
  59 s CPU and 1.4 GB. Mitigation: few distinct table operators per walk; the K-1 budget.
- **L-22, unresolved loops give symbolic counts.** Shared with L-12 and L-17
  (`test_select_swap_qrom_counts_are_symbolic`, `test_dynamic_loop_counts_are_symbolic`).
  Mitigation: loops unrolled in Python or with static bounds; T4 asserts `is_symbolic == False`.
```

- [ ] **Step 7: Update the spec for the changed and new canaries**

Edit `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md`. Make only these changes (the orchestrator reviews spec edits at merge):

1. Section 14, L-2: replace the sentence "`C(Adjoint(Op))` with a measuring adjoint rule has no path." with "`C(Adjoint(Op))` skips the measuring adjoint rule; the solver then controls the unitary inverse of `Op`'s base rule if one exists, and has no path otherwise (observed 2026-11-11)."
2. Section 14, L-6: append "The canary's reproducer compiled on 2026-11-11; the issue is open, so the mitigation stays." to the entry's first paragraph.
3. Section 14, L-9: replace the title "Function calls inside control regions raise." with "Function calls inside control regions are not decomposed." and its sentence with "On 2026-11-11 a controlled `qp.capture.subroutine` leaves `C(PauliX)` undecomposed at `level=\"user\"`, and compiling for execution fails ([catalyst#3330](https://github.com/PennyLaneAI/catalyst/pull/3330), [catalyst#3384](https://github.com/PennyLaneAI/catalyst/issues/3384))."
4. Section 14, L-15: change the first bullet to "Hardware-like one-shot postselection is rejected under capture with a `CompileError` (observed 2026-11-11; [pennylane#8379](https://github.com/PennyLaneAI/pennylane/issues/8379))." and append " The canary's reproducer was not affected on 2026-11-11." to the second bullet.
5. Section 14, L-19: replace "Neither has a path to the Toffoli-level gate set." with "For a non-power-of-two size neither has a path to the Toffoli-level gate set; for a power-of-two size `AliasSampling` compiles (observed 2026-11-11)."
6. Section 14, L-20: replace the sentence starting "At runtime it ignores static data" with "On 2026-11-11 such an `Operator2` is decomposed by the device pass if it has a rule, and fails to compile for execution if it has none; it no longer reaches runtime as a bare named gate ([catalyst#3430](https://github.com/PennyLaneAI/catalyst/pull/3430))."
7. Section 14: after L-23 add
   "24. **L-24: Compilable integers must fit in signed 64 bits.** Larger values fail to compile (MLIR: large integer attributes are not supported).
       - *Mitigation:* wide table entries are tuples of 32-bit words (§8.4).
       - *Canary:* `test_compilable_int_must_fit_int64` (subprocess).
   25. **L-25: `qp.adjoint(OpClass)(*args)` traces positional arguments under capture,** so a compilable argument passed positionally is rejected.
       - *Mitigation:* apply adjoints as `qp.adjoint(Op(...))` (§7.4, rule 13).
       - *Canary:* `test_adjoint_of_class_traces_positional_arguments`."
8. Section 7.4: append rule "13. **Adjoints use the instance form** `qp.adjoint(Op(...))`, never `qp.adjoint(Op)(*positional_args)` (L-25)."
9. Section 8.4, `WalkTables` bullet: append "Integers wider than 32 bits are stored as tuples of 32-bit words, least significant word first (L-24)."

- [ ] **Step 8: Lint and commit**

Run: `"$VENV/bin/ruff" check tests/canary` · Expected: `All checks passed!`

```bash
git add tests/canary docs/limitations.md docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md
git commit -m "Add canaries L-15 to L-18, L-20 to L-22 and update spec section 14"
```
