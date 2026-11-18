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

"""Canary L-26: unsigned dynamic arguments get no decomposition rule (spec section 14).

L-26 was found by Task 0.9b on 2026-11-12. Catalyst's `_MLIR_DTYPES_TO_PY_DTYPES` in
`catalyst/decomposition/type_utils.py` maps only `i1`, `i8`, `i16`, `i32` and `i64` (and floats
and complex numbers), so an `Operator2` with an unsigned dynamic argument (`uint32`, MLIR `ui32`)
gets no rule: Catalyst warns "Failed to synthesize the distributed decomposition rules for <Op>:
'ui32'" and the solver then fails with "Decomposition rule not found for operator '<Op>'". The
same operator with `int32` or `int64` data compiles. Compilable table words are therefore `int64`
holding 32 data bits each (ledger decision "Table words are `int64`").
"""

import pytest

from dfthc.testing.canary import outcome, run_snippet

pytestmark = [pytest.mark.canary, pytest.mark.compiled]


@pytest.mark.subprocess
def test_unsigned_dynamic_argument_has_no_rule():
    """L-26: the probe compiles for int32 and int64 data, but not for uint32 data."""
    result = run_snippet(
        """
        import jax.numpy as jnp

        from dfthc.compile import compiled_resources, register_rule

        class L26Probe(Operator2):
            dynamic_argnames = ("data",)
            wire_argnames = ("wires",)

            def __init__(self, data, wires):
                super().__init__(data, wires)

        @qp.register_resources({qp.X: 1})
        def _l26_rule(data, wires):
            qp.cond(data[0] == 1, qp.X)(wires[0])

        register_rule(L26Probe, _l26_rule)

        def compile_probe(dtype):
            def qfunc():
                L26Probe(jnp.asarray(np.array([1, 0], dtype=dtype)), [0])

            return compiled_resources(qfunc, num_wires=1).counts

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            report("UINT32", lambda: compile_probe(np.uint32))
        texts = [str(record.message) for record in records]
        report(
            "UINT32_WARNING",
            lambda: any("Failed to synthesize" in t and "'ui32'" in t for t in texts),
        )
        report("INT32", lambda: compile_probe(np.int32))
        report("INT64", lambda: compile_probe(np.int64))
        """
    )
    unsigned = outcome(result, "UINT32")
    assert unsigned.startswith("ERROR CompileError")
    # Catalyst names the operator with its argument types: 'L26Probe{data:[tensor<2xui32>]}...'.
    assert "Decomposition rule not found for operator 'L26Probe" in unsigned
    assert outcome(result, "INT32") == 'OK {"PauliX": 1}'
    assert outcome(result, "INT64") == 'OK {"PauliX": 1}'
    assert outcome(result, "UINT32_WARNING") == "OK true"
