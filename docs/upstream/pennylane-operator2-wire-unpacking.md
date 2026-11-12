# `Operator2` bind under program capture costs 4 jaxpr equations per wire for an array wire register (2 for a list): the array is unpacked twice

Draft for the user to file at <https://github.com/PennyLaneAI/pennylane/issues>. The reproducer is inlined below and saved next
to this draft as `repro_operator2_wire_unpacking.py`.

---

**Versions:** pennylane 0.46.0-dev151, jax 0.7.1, Python 3.12.7, macOS (arm64).

### Summary

Under program capture, binding an `Operator2` takes one scalar operand per wire: the bind iterates each wire register and
extends the positional operands wire by wire (`pennylane/core/operator/operator2.py:1597-1599`). Iterating a JAX array costs a
slice and a squeeze per element, so a wire register passed as a list of scalars costs 2 equations per wire. Passed as an array
slice (the natural form inside a decomposition rule, where wire registers are arrays), it costs 4: constructing the operator also
calls `_to_int_wires` (`operator2.py:2222-2237`), which iterates the array a second time (`for w in wires`) to build `Wires`,
and those per-element equations are never used by the bind.

In a large compiled program this is a fixed share of the captured IR. In a stand-in of our walk (a qubitization walk with about
1,560 wires), the composite operators, Givens ladders, read networks and erase fixups hold 77% of the projected full-size
capture, and most of their equations are this per-wire operand extraction; capture memory grows at about 6 KB per equation.
Halving the per-wire cost of array registers would remove a large part of that. Results and counts are unaffected; the cost is
capture memory and compile time.

### Reproducer (trace only, nothing compiled)

```python
"""Binding an Operator2 under program capture costs 4 jaxpr equations per wire when a wire register
is passed as an array slice, against 2 when it is passed as a list of scalars.

Trace only (jax.make_jaxpr), nothing is compiled. The equations counted are the operator's whole
bind, recursing into nested jaxprs. Usage: python repro_operator2_wire_unpacking.py
"""

import jax
import pennylane as qp
from pennylane.core import Operator2


class Child(Operator2):
    wire_argnames = ("a", "b", "c")

    def __init__(self, a, b, c):
        super().__init__(a, b, c)


def count(jaxpr) -> int:
    total = 0
    for eqn in jaxpr.eqns:
        total += 1
        for value in eqn.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    total += count(inner)
    return total


def as_lists(wires, n):
    Child([wires[i] for i in range(n)], [wires[i] for i in range(n, 2 * n)],
          [wires[i] for i in range(2 * n, 3 * n)])


def as_slices(wires, n):
    Child(wires[:n], wires[n:2 * n], wires[2 * n:3 * n])


def main():
    print(f"pennylane {qp.__version__}, jax {jax.__version__}")
    qp.capture.enable()
    try:
        for n in (5, 50, 500):
            wires = jax.numpy.arange(3 * n)
            cells = []
            for label, fn in (("list of scalars", as_lists), ("array slice", as_slices)):
                eqns = count(jax.make_jaxpr(lambda w, fn=fn, n=n: fn(w, n))(wires).jaxpr)
                cells.append(f"{label}: {eqns} equations ({eqns / (3 * n):.2f} per wire)")
            print(f"3 registers of {n} wires: " + "; ".join(cells))
    finally:
        qp.capture.disable()


if __name__ == "__main__":
    main()
```

### Output

```text
pennylane 0.46.0-dev151, jax 0.7.1
3 registers of 5 wires: list of scalars: 31 equations (2.07 per wire); array slice: 64 equations (4.27 per wire)
3 registers of 50 wires: list of scalars: 301 equations (2.01 per wire); array slice: 604 equations (4.03 per wire)
3 registers of 500 wires: list of scalars: 3001 equations (2.00 per wire); array slice: 6004 equations (4.00 per wire)
```

**Expected:** 2 equations per wire either way; ideally fewer for an array register, which could be bound as one operand.

### Suggested fix

- Reuse the scalars the bind extracts (or skip `_to_int_wires`' per-element loop) when the wire register is a traced array, so
  each wire is unpacked once.
- Longer term, let the operator primitive take an array wire register as one operand instead of one scalar per wire.
