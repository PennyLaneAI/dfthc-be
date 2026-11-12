# Reproducer of https://github.com/PennyLaneAI/catalyst/issues/3431, adapted to the canary
# interface of tests/canary/test_l06_to_l11_control.py::test_order_dependent_compile.
#
# Rules provide resources only, no bodies: the bug is in the construction of the decomposition
# graph. C(B) leads to X, X leads to C(C(B)) (two more controls), and Y leads to X. With the
# controlled B first, X is first visited through a path on which its child is pruned (a cycle
# that adds controls) and is cached as invalid, so the later visit through Y is skipped.
#
# Adaptations of the issue's code: ``qp.decompose`` (the canary prelude's decorator) replaces
# ``catalyst.passes.graph_decomposition`` (on 2026-11-12 both reproduce the failure); the
# resources are checked with ``qp.analyze`` (through the prelude's ``counts``) instead of
# ``qp.specs``; and the rules are registered once at module level, because rules added inside
# ``local_decomps()`` are gone by the time the returned QNode compiles on its first call.
# ruff: noqa: F821  (qp, Operator2 and Wire come from the canary prelude this file is appended to)
from pennylane.decomposition import add_decomps, register_resources

ORDERS = ("Yfirst", "CBfirst")


class B(Operator2):
    def __init__(self, wires):
        super().__init__(wires=wires)


class X(Operator2):
    def __init__(self, wires):
        super().__init__(wires=wires)


class Y(Operator2):
    def __init__(self, wires):
        super().__init__(wires=wires)


@register_resources({})
def b_to_nothing(wires):
    del wires


@register_resources({X(Wire[1]): 1})
def cb_to_x(base, **_):
    del base


@register_resources({qp.ctrl(B(Wire[1]), control=[Wire[2], Wire[3]]): 1})
def x_to_ccb(wires):
    del wires


@register_resources({X(Wire[1]): 1})
def y_to_x(wires):
    del wires


add_decomps(B, b_to_nothing)
add_decomps("C(B)", cb_to_x)
add_decomps(X, x_to_ccb)
add_decomps(Y, y_to_x)


def build(order):
    @qp.qjit(capture=True, target="mlir")
    @qp.decompose(gate_set={qp.Z: 1})
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit():
        if order == "CBfirst":
            qp.ctrl(B(0), control=1)
            Y(0)
        else:
            Y(0)
            qp.ctrl(B(0), control=1)
        return qp.state()

    return circuit
