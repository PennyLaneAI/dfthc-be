"""The shared constructor checks of dfthc.ops._validate (spec 9.0): wire lengths and, for
concrete wires, wire identity (final review m-5)."""

import jax
import pytest
from pennylane.typing import Wire
from pennylane.wires import Wires

from dfthc.ops._validate import check_wires
from dfthc.ops.majorana import MajoranaGate
from dfthc.ops.reflections import ZeroReflection
from dfthc.ops.spin import SpinSwap


@pytest.mark.parametrize(("build", "match"), [
    (lambda: SpinSwap(sg=[0], psi_up=[1, 2], psi_dn=[2, 3]),
     "SpinSwap: wire 2 appears in registers 'psi_up' and 'psi_dn'"),
    (lambda: MajoranaGate([0], [1], [2], [3], [4], [5], [5], work=[7, 8]),
     "MajoranaGate: wire 5 appears in registers 'flavour' and 'target'"),
    (lambda: SpinSwap(sg=[0], psi_up=[1, 1], psi_dn=[2, 3]),
     "SpinSwap: wire 1 appears twice in register 'psi_up'"),
], ids=["across-registers", "flavour-is-target", "within-a-register"])
def test_constructors_reject_a_repeated_wire(build, match):
    """The wires of one operator must be distinct qubits: a label repeated across or within
    registers raises ValueError at construction, before a compile fails on it."""
    with pytest.raises(ValueError, match=match):
        build()


def test_distinct_wires_pass_in_every_register_form():
    """Distinct concrete wires pass as lists, Wires and strings, with zero-width registers and
    the longer registers that at_least permits (phase_grad, work)."""
    check_wires("Op", {"a": [0, 1], "r": [], "phase_grad": Wires([2, 3, 4]), "work": ["w0", "w1"]},
                {"a": 2, "r": 0, "phase_grad": 2, "work": 1}, at_least=("phase_grad", "work"))
    ZeroReflection([0, 1, 2], [3], work=[4, 5, 6])


def test_abstract_wires_and_tracers_are_not_compared():
    """Resource keys hold Wire[n] and rules under capture hold tracers: neither has comparable
    labels, so identity is not checked there."""
    check_wires("Op", {"a": Wire[2], "b": Wire[2]}, {"a": 2, "b": 2})
    jax.make_jaxpr(lambda x: check_wires("Op", {"a": [x], "b": [x]}, {"a": 1, "b": 1}))(1)
