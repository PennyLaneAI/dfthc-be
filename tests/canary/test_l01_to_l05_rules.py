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
            @qp.qjit(capture=True, enable_aot_compilation=False)
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

        @qp.qjit(capture=True, enable_aot_compilation=False)
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

    @qp.qjit(capture=True, enable_aot_compilation=False)
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
            @qp.qjit(capture=True, enable_aot_compilation=False)
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

    @qp.qjit(capture=True, enable_aot_compilation=False)
    @qp.decompose(gate_set={"PauliX", "PauliY", "PauliZ"})
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        L05Flagged(1, [0])
        return qp.probs(wires=[0])

    assert dict(qp.analyze(circuit, level="user")().resources.counts) == {"PauliZ": 1}
