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

"""Catalyst captures the distributed Adjoint/C candidates of operators whose modified forms have
pinned rules, although no chosen rule uses them.

Usage: python repro_unused_distributed_candidates.py [N]

``Leaf`` has an N-CNOT rule (N even, so ``Leaf`` is the identity); it stands for any large rule
body, such as a table load. ``Part`` calls ``Leaf``; its pinned ``Adjoint(Part)`` rule does not
(two CNOTs, also the identity), as a measured uncompute does not call the load it undoes.
``Top`` calls ``Part`` and ``qp.adjoint(Part(...))`` around a ``Z``; its pinned ``C(Top)`` rule
controls only the ``Z``. Two programs with the same gates, each compiled in a child process:

- ``modifiers``: ``qp.ctrl(Top(...))`` with the four rules above pinned;
- ``explicit``: the same rule bodies as base rules of two plain classes, ``ControlledTop`` and
  ``PartDagger``, so no ``Adjoint``/``C`` form of ``Top`` or ``Part`` is ever requested.

Each child prints its analyzed counts and every rule target the frontend converted (traced)
whose name contains Leaf, Part or Top, with its nested jaxpr equations (loop bodies, cond
branches and modifier regions included). The counter wraps a private frontend function only to
make the captured targets visible; the compile is unchanged.
"""

import json
import resource
import subprocess
import sys
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 400
GATES = {"CNOT": 1, "CZ": 1, "PauliZ": 1, "Toffoli": 1}


def nested_equations(jaxpr) -> int:
    j = getattr(jaxpr, "jaxpr", jaxpr)
    total = 0
    for eqn in j.eqns:
        total += 1
        seen = set()
        for value in eqn.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                if (hasattr(item, "eqns") or hasattr(item, "jaxpr")) and id(item) not in seen:
                    seen.add(id(item))
                    total += nested_equations(item)
    return total


def child(variant: str) -> None:
    import catalyst.from_plxpr.qfunc_interpreter as interpreter
    import pennylane as qp
    from pennylane.core import Operator2
    from pennylane.typing import Wire

    qp.decomposition.enable_graph()

    def key(build):
        with qp.capture.pause():
            return build()

    class Leaf(Operator2):
        wire_argnames = ("wires",)

        def __init__(self, wires):
            super().__init__(wires)

    @qp.register_resources({qp.CNOT: N})
    def _leaf(wires):
        for _ in range(N):
            qp.CNOT(wires[:2])

    class Part(Operator2):
        wire_argnames = ("wires",)

        def __init__(self, wires):
            super().__init__(wires)

    @qp.register_resources(lambda wires: {key(lambda: Leaf(Wire[2])): 1})
    def _part(wires):
        Leaf(wires)

    def _undo_body(wires):
        qp.CNOT(wires[:2])
        qp.CNOT(wires[:2])

    @qp.register_resources(lambda base: {qp.CNOT: 2})
    def _adjoint_part(base):
        _undo_body(base.wires)

    class PartDagger(Operator2):
        wire_argnames = ("wires",)

        def __init__(self, wires):
            super().__init__(wires)

    @qp.register_resources({qp.CNOT: 2})
    def _part_dagger(wires):
        _undo_body(wires)

    class Top(Operator2):
        wire_argnames = ("wires",)

        def __init__(self, wires):
            super().__init__(wires)

    def top_keys():
        return {key(lambda: Part(Wire[2])): 1, key(lambda: qp.adjoint(Part(Wire[2]))): 1}

    @qp.register_resources(lambda wires: {**top_keys(), qp.Z: 1})
    def _top(wires):
        Part(wires)
        qp.adjoint(Part(wires))
        qp.Z(wires[0])

    @qp.register_resources(lambda base, control_wires, control_values, work_wires,
                           work_wire_type: {**top_keys(), qp.CZ: 1})
    def _c_top(base, control_wires, control_values, work_wires, work_wire_type):
        Part(base.wires)
        qp.adjoint(Part(base.wires))
        qp.CZ([control_wires[0], base.wires[0]])

    class ControlledTop(Operator2):
        wire_argnames = ("wires", "control")

        def __init__(self, wires, control):
            super().__init__(wires, control)

    @qp.register_resources(lambda wires, control: {key(lambda: Part(Wire[2])): 1,
                                                   key(lambda: PartDagger(Wire[2])): 1, qp.CZ: 1})
    def _controlled_top(wires, control):
        Part(wires)
        PartDagger(wires)
        qp.CZ([control[0], wires[0]])

    rules = {Leaf: _leaf, Part: _part, "Adjoint(Part)": _adjoint_part, PartDagger: _part_dagger,
             Top: _top, "C(Top)": _c_top, ControlledTop: _controlled_top}
    for op, rule in rules.items():
        qp.add_decomps(op, rule)

    converted = []
    convert = interpreter._convert_decomp_target_spec

    def counting(interp, spec):
        traced = convert(interp, spec)
        name = str(spec.target_id).split("{", 1)[0]
        converted.append((name, sum(nested_equations(r.closed_jaxpr) for r in traced)))
        return traced

    interpreter._convert_decomp_target_spec = counting

    @qp.qjit(capture=True, enable_aot_compilation=False)
    @qp.decompose(gate_set=GATES, fixed_decomps=rules)
    @qp.qnode(qp.device("null.qubit", wires=3))
    def circuit():
        if variant == "modifiers":
            qp.ctrl(Top([1, 2]), control=0)
        else:
            ControlledTop([1, 2], [0])
        return qp.probs(wires=[0])

    start = time.perf_counter()
    counts = qp.analyze(circuit, level="user")().resources.counts
    seconds = time.perf_counter() - start
    ours = [(name, eqs) for name, eqs in converted
            if any(s in name for s in ("Leaf", "Part", "Top"))]
    print(json.dumps({
        "variant": variant, "counts": {k: int(v) for k, v in counts.items()},
        "compile_and_analyze_s": round(seconds, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
        "targets_converted": len(converted),
        "equations_converted": sum(eqs for _, eqs in converted),
        "leaf_part_top_targets": ours}))


def main() -> None:
    import catalyst
    import jax
    import pennylane

    print(f"pennylane {pennylane.__version__}, catalyst {catalyst.__version__}, "
          f"jax {jax.__version__}, python {sys.version.split()[0]}, N = {N}")
    for variant in ("modifiers", "explicit"):
        run = subprocess.run([sys.executable, __file__, str(N), variant], capture_output=True,
                             text=True, check=True)
        record = json.loads(run.stdout.strip().splitlines()[-1])
        print(f"\n{variant}: counts {record['counts']}, "
              f"{record['compile_and_analyze_s']} s, peak RSS {record['peak_rss_mb']} MB, "
              f"{record['targets_converted']} targets / {record['equations_converted']} equations "
              "converted")
        for name, eqs in record["leaf_part_top_targets"]:
            print(f"  {name:<22} {eqs:>7} equations")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        child(sys.argv[2])
    else:
        main()
