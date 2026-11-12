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
