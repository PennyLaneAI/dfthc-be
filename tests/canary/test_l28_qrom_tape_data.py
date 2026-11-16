"""Canary L-28: `qp.QROM` takes no measurement values as data in tape mode (spec section 14).

Found by Task 6.4 on 2026-11-16. The erase fix-ups' leaf loop could be `H . QROM(parities) . H`
on the one-hot register (the audit's probe 4: exact phases under capture, one Toffoli fewer per
fix-up), but in tape mode the parities are `MeasurementValue`s (K-10), and `QROM` and its leaf
operator `MultiX` accept only bool or integer arrays. The fix-ups must work in both modes (spec
9.4), so they keep their conditional `CZ` leaves.
"""

import pennylane as qp
import pytest

pytestmark = [pytest.mark.canary]


def _outcomes(n):
    return [qp.measure(wire, reset=True) for wire in range(4, 4 + n)]


@pytest.mark.parametrize("build", [
    lambda ms: qp.QROM([[ms[0], ms[1]], [ms[2], ms[3]]], control_wires=[0], target_wires=[1, 2],
                       work_wires=[], clean=True),
    lambda ms: qp.MultiX(ms[:2], wires=[1, 2]),
], ids=["QROM", "MultiX"])
def test_qrom_rejects_measurement_values_in_tape_mode(build):
    """L-28: recording QROM, or its leaf MultiX, with measurement values as data raises."""
    with qp.capture.pause(), qp.queuing.AnnotatedQueue():
        outcomes = _outcomes(4)
        with pytest.raises(ValueError):
            build(outcomes)
