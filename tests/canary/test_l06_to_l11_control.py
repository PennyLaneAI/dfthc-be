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

    Regression sentinel: the crash reported in catalyst#3217 did not reproduce on 2026-11-11 (it
    compiles to `{"PauliX": "4"}`). This test fails if the crash returns; it cannot detect a fix.
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
    canary was written: "OK" or "ERROR <ExceptionType>". Only an order that compiles next to one
    that fails demonstrates L-10; if every order fails, the reproducer or the setup is broken.
    """
    source = (REPRODUCERS / "catalyst_3431.py").read_text()
    recorded = json.loads((REPRODUCERS / "catalyst_3431.json").read_text())["outcomes"]
    loop = "\nfor order in ORDERS:\n    report(f'ORDER_{order}', lambda: counts(build(order)))\n"
    result = run_snippet(source + loop)
    observed = {order: outcome(result, f"ORDER_{order}") for order in recorded}
    assert len(observed) >= 2, observed
    assert any(text.startswith("OK") for text in observed.values()), (
        f"every order fails, so the reproducer does not demonstrate L-10: {observed}"
    )
    for order, expected in recorded.items():
        assert observed[order].startswith(expected), (order, observed[order])


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
