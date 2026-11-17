# dfthc: the DFTHC block encoding in PennyLane

A fully compiled PennyLane implementation of the DFTHC block encoding of
[Low et al., PRX 15, 041016 (2025)](https://journals.aps.org/prx/abstract/10.1103/pb2g-j9cw)
([arXiv:2502.15882](https://arxiv.org/abs/2502.15882)): the block encoding, its qubitized walk, the
controlled walk and an iterative phase estimation. Every block is a modular PennyLane `Operator2`
with one pinned decomposition rule. Programs compile with `qp.qjit(capture=True)` and graph-based
`qp.decompose` down to Cliffords, Toffolis, measured AND-uncomputes, mid-circuit measurements and
a phase-gradient state, and `qp.analyze` counts their resources exactly. The Qualtran
implementation [dfthcbe](https://github.com/Covestro-HPQC/dfthcbe) serves as the reference.

## Results

Toffolis per walk step (Toffoli plus `TemporaryAND`), each compiled count equal to its closed form
in `dfthc.costs`:

| System | Walk qubits | Ours | dfthcbe | Paper |
| --- | ---: | ---: | ---: | ---: |
| H2, STO-3G | 177 | 341 | 392 | 400 |
| naphthalene | 327 | 1,910 | 1,937 | 1,965 |
| Fe2S2 | 498 | 4,411* | 4,420 | 4,506 |
| Fe4S4 | 757 | 6,533* | 6,336 | 6,556 |
| FeMoco-54 | 1,049 (paper: 1,131) | 9,790 | 9,641 | 9,997 |

\* Closed form; not recompiled since the 2026-11-17 changes (`--large fe2s2|fe4s4`).

Each difference from dfthcbe and from the paper is broken down by cause in
[`docs/resources.md`](docs/resources.md), which reports every block of every system, with
measurements, Cliffords and qubits. The FeMoco-54 walk compiles in about a minute below 2 GB.

## Circuit

The first half of the block encoding plus REF<sub>T2</sub> (paper steps 1-23), drawn at the same
granularity as the [dfthcbe figure](https://github.com/Covestro-HPQC/dfthcbe). Red circles mark the
wires and gates that differ from it; [`docs/figures/dfthcbe_comparison.md`](docs/figures/dfthcbe_comparison.md)
explains each difference step by step.

![DFTHC block encoding, steps 1-23](docs/figures/walk_circuit.png)

## Quick start

```bash
scripts/setup_env.sh                            # Python 3.12 venv with nightly PennyLane and Catalyst
scripts/test.sh -m "not slow and not release"   # per-commit tier, about 15 minutes
```

```python
import pennylane as qp
from dfthc.compile import compiled_resources
from dfthc.ops.walk import DFTHCWalk
from dfthc.testing import walk_sim

case = walk_sim.case("sto3g")  # H2 test preset: parameters, tables and wire layout

def program():
    qp.PhaseGradientStatePrep(case.layout.phase_grad)
    DFTHCWalk(*case.compilable, **case.groups())

res = compiled_resources(program, case.num_wires)  # qjit(capture=True), decompose, analyze
print(res.toffoli, res.measurements, res.measured_and_uncomputes)  # 158 48 108
```

Run it with `PYTHONPATH=src .venv/bin/python`. `qp.analyze` also warns that it could not determine
a conditional's branch; it raises that once per program, and the counts are exact.

## Verification

1,386 tests in three tiers ([`docs/verification.md`](docs/verification.md)) cover:

- per-block semantics on `lightning.qubit` and a sparse simulator;
- the block encoding and walk end to end on H2, against the encoded Hamiltonian to 1e-10;
- GF(2) certificates for every measured erase;
- exact QPE on a toy walk;
- exact T, M and A counts against closed forms;
- canary tests that pin every known PennyLane or Catalyst limitation.

## Repository layout

```text
src/dfthc/
  params.py, data/    classical data: factors, alias and angle tables, the encoded Hamiltonian
  ops/                Operator2 blocks: primitives, the paper's blocks, block encoding and walk
  qpe.py              controlled walk powers and iterative phase estimation
  registers.py        WalkLayout: the named wire groups of the walk
  compile.py          gate set, pinned rules, compiled_resources()
  costs.py            closed-form T, M and A of every block
  costs_reference.py  the paper's and dfthcbe's costs, and the causes of each difference
  testing/            sparse simulator, walk cases, memory watchdog, compile cache provenance
data/dfthc/           DFTHC factors vendored from dfthcbe (Apache-2.0)
scripts/              environment setup, test runner, resource report, fixture generators
tests/                the test suite
spikes/               early feasibility probes
docs/
  architecture.md     layers, the walk call by call, compilation rules (start here)
  resources.md        the resource report (resources.json: machine-readable, with compile cache)
  verification.md     what each test establishes, and the test tiers
  limitations.md      PennyLane and Catalyst limitations, and their canaries
  conventions.md      indices, signs, fermions, angles, registers
  environment.md      the pinned nightly toolchain, load limits, commands
  upstream/           draft issue reports with reproducers
  superpowers/        the design spec and the implementation plans
  agents/             handoff notes and the ledger of the agent-driven development
```
