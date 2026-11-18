# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            @qp.qjit(capture=True, enable_aot_compilation=False)
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
    above = outcome(result, "ABOVE")
    assert above.startswith("ERROR CompileError")
    assert "Large integer attributes currently not supported in MLIR" in above


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
    @qp.qjit(capture=True, enable_aot_compilation=False)
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
