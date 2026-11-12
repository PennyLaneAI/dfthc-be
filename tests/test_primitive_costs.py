"""Pin the PennyLane primitive costs assumed by the closed forms of spec 9.24 (T4, Task 0.2)."""

import pennylane as qp
import pytest

from dfthc.compile import compiled_resources

pytestmark = pytest.mark.compiled


def _wires(start, count):
    return list(range(start, start + count))


def _maybe_adjoint(op_class, adjoint):
    return qp.adjoint(op_class) if adjoint else op_class


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 8, 15])
@pytest.mark.parametrize("adjoint", [False, True])
def test_semiadder(n, adjoint):
    def qfunc():
        _maybe_adjoint(qp.SemiAdder, adjoint)(
            x_wires=_wires(0, n), y_wires=_wires(n, n), work_wires=_wires(2 * n, n - 1) or None
        )

    resources = compiled_resources(qfunc, num_wires=3 * n)
    assert resources.counts.get("Toffoli", 0) == 0
    assert (resources.toffoli, resources.measured_and_uncomputes) == (n - 1, n - 1)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("comparator", ["<=", "<"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_left_quantum_comparator(n, comparator, adjoint):
    def qfunc():
        _maybe_adjoint(qp.LeftQuantumComparator, adjoint)(
            x_wires=_wires(0, n), y_wires=_wires(n, n), target_wire=[2 * n],
            work_wires=_wires(2 * n + 1, n - 1), comparator=comparator,
        )

    resources = compiled_resources(qfunc, num_wires=3 * n + 1)
    assert resources.counts.get("Toffoli", 0) == 0
    expected = (0, n) if adjoint else (n, 0)
    assert (resources.toffoli, resources.measured_and_uncomputes) == expected


@pytest.mark.parametrize(("n", "value"), [(3, 1), (3, 6), (5, 7), (5, 27), (7, 81), (7, 3)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_left_classical_comparator(n, value, adjoint):
    def qfunc():
        _maybe_adjoint(qp.LeftClassicalComparator, adjoint)(
            x_wires=_wires(0, n), value=value, target_wire=[n], work_wires=_wires(n + 1, n - 1),
            comparator="<",
        )

    resources = compiled_resources(qfunc, num_wires=2 * n)
    assert resources.counts.get("Toffoli", 0) == 0
    expected = (0, n - 1) if adjoint else (n - 1, 0)
    assert (resources.toffoli, resources.measured_and_uncomputes) == expected


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8, 12])
def test_multicontrolled_x_with_zeroed_work(k):
    def qfunc():
        qp.MultiControlledX(wires=_wires(0, k + 1), work_wires=_wires(k + 1, k - 2) or None,
                            work_wire_type="zeroed")

    resources = compiled_resources(qfunc, num_wires=2 * k + 1)
    assert resources.counts.get("Toffoli", 0) == 1
    assert (resources.toffoli, resources.measured_and_uncomputes) == (k - 1, k - 2)


def test_cswap():
    resources = compiled_resources(lambda: qp.CSWAP(wires=[0, 1, 2]), num_wires=3)
    assert resources.counts.get("Toffoli", 0) == 1
    assert resources.toffoli == 1
