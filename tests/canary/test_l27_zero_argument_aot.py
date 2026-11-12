"""Canary L-27: `qp.qjit` compiles a zero-argument function ahead of time at decoration.

L-27 was found by the Opus review of Task 0.9c on 2026-11-12. Catalyst's `params_are_annotated`
(`catalyst/tracing/type_signatures.py`) treats a function without parameters as fully annotated,
and `enable_aot_compilation` defaults to `True` (`catalyst/jit.py`), so the `QJIT.__init__` AOT
branch compiles the program to a linked binary at decoration. `qp.analyze` builds its own `QJIT`
with AOT disabled and compiles the program again, so an analyze-only program is compiled twice
(`dfthc.compile.compiled_resources` therefore sets `enable_aot_compilation=False`).

The observable is `QJIT.aot_compile`, which the decorator calls once in the AOT branch, plus the
state it leaves behind (`compiled_function`, `mlir_module`). The canary flips if Catalyst changes
the default or the zero-argument rule.
"""

import pennylane as qp
import pytest
from catalyst import QJIT

pytestmark = [pytest.mark.canary, pytest.mark.compiled]


def _spy_on_aot_compile(monkeypatch) -> list[str]:
    """Record the compile target of every ``QJIT.aot_compile`` call in the returned list."""
    calls = []
    original = QJIT.aot_compile

    def spy(self, *args, **kwargs):
        calls.append(self.compile_options.target)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(QJIT, "aot_compile", spy)
    return calls


def test_zero_argument_qjit_compiles_at_decoration(monkeypatch):
    """L-27: the default decorator AOT-compiles at once; `enable_aot_compilation=False` does not.

    A call then reuses the binary (no second `aot_compile`); `qp.analyze` compiles again.
    """
    calls = _spy_on_aot_compile(monkeypatch)

    def decorate(**options):
        @qp.qjit(capture=True, **options)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.X(0)
            return qp.probs(wires=[0])

        return circuit

    circuit = decorate()
    assert calls == ["binary"], "Catalyst no longer AOT-compiles a zero-argument function"
    assert circuit.compiled_function is not None and circuit.mlir_module is not None
    circuit()
    assert calls == ["binary"]
    qp.analyze(circuit, level="user")()
    assert calls == ["binary", "mlir_opt"]

    calls.clear()
    circuit = decorate(enable_aot_compilation=False)
    assert calls == []
    assert circuit.compiled_function is None and circuit.mlir_module is None
    qp.analyze(circuit, level="user")()
    assert calls == ["mlir_opt"]


def test_unannotated_parameter_is_not_compiled_at_decoration(monkeypatch):
    """L-27: the trigger is the missing parameters; an unannotated parameter compiles on call."""
    calls = _spy_on_aot_compile(monkeypatch)

    @qp.qjit(capture=True)
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit(theta):
        qp.RX(theta, 0)
        return qp.probs(wires=[0])

    assert calls == []
    assert circuit.compiled_function is None
