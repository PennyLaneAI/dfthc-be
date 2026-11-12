"""Tape-path decomposition of a quantum function for the sparse simulator (spec section 12.3)."""

from __future__ import annotations

from collections.abc import Callable

import pennylane as qp
from pennylane.transforms import _tape_decompose

from dfthc.compile import FIXED_DECOMPS, GATE_SET


def decompose_for_simulation(qfunc: Callable, *args, **kwargs) -> qp.tape.QuantumScript:
    """Record ``qfunc(*args, **kwargs)`` as a tape and decompose it to the gate set.

    Capture is paused while recording, and the graph decomposition system is enabled for
    ``qp.transforms._tape_decompose(tape, gate_set=GATE_SET, fixed_decomps=FIXED_DECOMPS)``;
    both are restored afterwards. The result holds gate-set operations, ``MidMeasure`` and
    ``Conditional`` operations with ``MeasurementValue`` expressions, plus the measurements
    ``qfunc`` queued. Our operators decompose with their pinned rules; PennyLane primitives may
    use rules the compiled path does not, which leaves the semantics unchanged (spec 12.3).
    """
    graph_was_enabled = qp.decomposition.enabled_graph()
    qp.decomposition.enable_graph()
    try:
        with qp.capture.pause():
            with qp.queuing.AnnotatedQueue() as queue:
                qfunc(*args, **kwargs)
            tape = qp.tape.QuantumScript.from_queue(queue)
            [decomposed], _ = _tape_decompose(tape, gate_set=GATE_SET,
                                              fixed_decomps=dict(FIXED_DECOMPS) or None)
    finally:
        if not graph_was_enabled:
            qp.decomposition.disable_graph()
    return decomposed
