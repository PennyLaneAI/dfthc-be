# Graph decomposition: distributed `Adjoint`/`C` candidates are synthesized, captured and lowered for operators whose modified forms have pinned rules, multiplying capture memory

Draft for the user to file at <https://github.com/PennyLaneAI/catalyst/issues>. The reproducer is inlined below and saved next
to this draft as `repro_unused_distributed_candidates.py`.

---

**Versions:** pennylane 0.46.0-dev151, pennylane-catalyst 0.16.0-dev140, jax 0.7.1, Python 3.12.7, macOS (arm64).

### Summary

When an `Operator2` has a registered and pinned (`fixed_decomps`) rule for `Adjoint(Op)` or `C(Op)`, the frontend still
synthesizes the distributed candidate for that modified state: the base rule's body wrapped in an adjoint or control region
(`build_decomp_rule_variants`, `catalyst/decomposition/decomposition_rules.py:1343-1382`; the distributed spec is appended
unconditionally at `:1376-1380`). The reachability walk then requests the same modifier of every operator in that body
(`walk_reachable_decomp_rule_sets`, `:1413-1421`), so whole subtrees are traced, captured and lowered again. The compiler pass
then applies the pinned rule and never uses them. The counts are correct; the cost is capture memory and compile time, which
grow with the number of modifier states that reach each large rule body (up to four copies: plain, `Adjoint`, `C`,
`C(Adjoint)`).

`fixed_decomps` and `exclude_decomps` reach only the MLIR pass (`catalyst/from_plxpr/from_plxpr.py:368-369`), and the
`graph_decomposition` pass documents that "excluded rules are still lowered by the frontend"
(`catalyst/passes/builtin_passes.py:1812-1813`), so a user has no supported way to keep the frontend from capturing the unused
candidates.

### Reproducer

`Leaf` has an N-CNOT rule (N even, so `Leaf` is the identity); it stands for any large rule body, such as a table load. `Part`
calls `Leaf`; its pinned `Adjoint(Part)` rule does not (two CNOTs, also the identity), the way a measured uncompute does not
call the load it undoes. `Top` calls `Part` and `qp.adjoint(Part(...))` around a `Z`; its pinned `C(Top)` rule controls only
the `Z`. The same gates are compiled twice, each in a child process:

- `modifiers`: `qp.ctrl(Top(...))` with the four rules pinned;
- `explicit`: the same rule bodies as base rules of two plain classes, `ControlledTop` and `PartDagger`, so no
  `Adjoint`/`C` form of `Top` or `Part` is ever requested.

A wrapper of the private `qfunc_interpreter._convert_decomp_target_spec` only lists the converted targets; it does not change
the compile.

```python
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
```

### Output

One run, alone, with `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` and `-W ignore` (the built-in "Excluded the ... rule"
warnings removed), on a 10-core, 16 GB laptop; the process tree peaked at 594 MB and took 14 s:

```text
pennylane 0.46.0-dev151, catalyst 0.16.0-dev140, jax 0.7.1, python 3.12.7, N = 400

modifiers: counts {'CNOT': 402, 'CZ': 1}, 5.2 s, peak RSS 480.9 MB, 204 targets / 34206 equations converted
  C(Top)                      21 equations
  C(Top)                      32 equations
  Part                        11 equations
  Adjoint(Part)               14 equations
  Adjoint(Part)               12 equations
  C(Part)                     17 equations
  C(Adjoint(Part))            20 equations
  C(Adjoint(Part))            18 equations
  Leaf                      7200 equations
  Adjoint(Leaf)             7201 equations
  C(Leaf)                   7206 equations
  C(Adjoint(Leaf))          7207 equations

explicit: counts {'CNOT': 402, 'CZ': 1}, 1.5 s, peak RSS 322.2 MB, 67 targets / 8521 equations converted
  ControlledTop               31 equations
  Part                        11 equations
  PartDagger                  36 equations
  Leaf                      7200 equations
```

Both programs give the same gates (402 CNOT, 1 CZ: the pinned rules are used). The modifier program converts `Leaf`'s body
four times (`Leaf`, `Adjoint(Leaf)`, `C(Leaf)`, `C(Adjoint(Leaf))`), although only `Leaf` is reachable from the chosen rules:
4.0 times the captured equations, 3.5 times the compile time and 159 MB more peak memory at this size. `Adjoint(Leaf)` comes
from the distributed candidate of the pinned `Adjoint(Part)`; `C(Leaf)` and `C(Adjoint(Leaf))` come from the distributed
candidate of the pinned `C(Top)`. Each listed name appears twice where both the registered and the distributed spec of that
state were converted.

**Expected:** with `Adjoint(Part)` and `C(Top)` pinned to registered rules, only the targets reachable from the chosen rules
(`C(Top)`'s pinned rule, `Part`, `Adjoint(Part)`'s pinned rule, `Leaf`) are captured, as in the explicit-class program.

### Impact in a real program

We compile a qubitization walk (a DFTHC block encoding, Phys. Rev. X 15, 041016 (2025), DOI 10.1103/pb2g-j9cw) and its controlled version as one program each,
with pinned measured-uncompute `Adjoint` rules and pinned `C` rules that control only the reflections. Measurements on a
stand-in with the walk's structure (process-tree footprint; identical analyzed counts in every case):

- **Catalyst patched to skip the distributed candidates** (a local test patch of `build_decomp_rule_variants`): the small
  controlled walk fell from 821.6 to at most 441 MB, the mid-size controlled walk from 2,559 to 1,011 MB, and the mid-size walk
  from 1,482 to 1,126 MB. The distributed candidates held 73% of the nested equations captured for the mid-size controlled walk.
  (These three measurements still included the decoration-time compile of "Related" below; the ones that follow do not.)
- **The same candidates avoided without patching**, by explicit operator classes (as in the reproducer's `explicit` program),
  at our largest mid shape (66,000 gates): the controlled walk fell from 2,333 to 1,090 MB and the walk from 1,346 to 1,105 MB;
  95–99% of the equations captured in distributed copies disappeared, and the controlled walk now costs the same as the walk.
  Memory follows captured equations at 5.6–6.1 KB per equation in every variant.
- **Projected to the full application size** (FeMoco-54, 627,557 gates per walk) from those mid-size fits: the controlled walk
  needs 12.6 GB (9.5–15.7) with Catalyst's candidates and 4.4 GB with explicit classes; with explicit classes and one further
  local change (one table-load loop per unary-iteration leaf instead of one per table entry) 2.6 GB.
- **Measured at full size** with explicit classes and that loop change: the controlled walk compiled once in 51 s at 2,648 MB,
  with exact counts. Without the explicit classes the projection exceeds the 7.5 GB we can run on a 16 GB machine.

So the explicit classes are our workaround, and they cost a design change: every adjoint and controlled form of a composite
becomes its own operator class with its own documentation and tests, and a single stray `qp.adjoint` or `qp.ctrl` of a
composite anywhere in a program brings the whole cascade back. The cost hits every Catalyst user who writes pinned modified
rules for large operators.

### Suggested fix

In order of preference:

1. **Prune what the chosen rules cannot reach.** Lower only the rule targets reachable from the rules the solver can choose
   under the pins: pass `fixed_decomps` (and `exclude_decomps`) from `qp.decompose` into the frontend capture scope, and in
   `build_decomp_rule_variants` skip the distributed spec of a modified state whose key is pinned to a registered rule. Then
   the reachability walk never requests the subtree's modified states.
2. **Prefer registered rules for modified states.** Synthesize the distributed candidate of a modified state only when no
   registered rule exists for it (or only when no registered rule is pinned for it).
3. **An opt-out.** A `qp.decompose` or `qjit` option that disables distributed synthesis for operators with registered
   modified rules, or that makes `exclude_decomps` also exclude rules from frontend lowering.

### Related

- `qp.qjit` AOT-compiles a zero-argument (vacuously fully annotated) function at decoration
  (`catalyst/tracing/type_signatures.py`, `params_are_annotated`), so the documented pattern `qp.analyze(qjit_qnode)()`
  compiles the program twice and holds both captures; `enable_aot_compilation=False` avoids it (the reproducer uses it). That
  was the other half of our memory: 2,552 to 1,536 MB for the mid-size controlled walk, same counts.
