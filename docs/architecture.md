# Architecture

This package builds the DFTHC block encoding of Low et al. (PRX 15, 041016 (2025);
arXiv:2502.15882), its qubitized walk, the controlled walk and an iterative phase estimation,
as PennyLane `Operator2` building blocks. Programs compile with `qp.qjit(capture=True)` and
`qp.decompose` to a Toffoli-level gate set, and `qp.analyze` counts their resources exactly. The
design is fixed by the specification
([`design.md`](design.md),
cited as "spec N.M"); this page maps it onto the code.

## Layers

Each layer depends only on the layers below it (spec 5.1).

| Layer | Content | Modules |
| --- | --- | --- |
| L0, classical data | factors, parameters, alias and angle tables, the encoded Hamiltonian, fermion matrices; NumPy and SciPy only | `dfthc.params`, `dfthc.data.factors`, `dfthc.data.sos`, `dfthc.data.alias`, `dfthc.data.angles`, `dfthc.data.tables`, `dfthc.data.encoded`, `dfthc.data.fermion` |
| L1, primitives | `PhaseGradientRotation`, `FlaggedUniformPrep`, `QROAMRead` (its select stage is PennyLane's `QROM`), `PhaseFixup`, `QROAMErase`; the unary-iteration helper (the angle loads and the fix-ups) | `dfthc.ops.phase_gradient`, `dfthc.ops.uniform`, `dfthc.ops.qroam`, `dfthc.ops.unary`, `dfthc.ops._validate`, `dfthc.ops._gates` (Toffolis with a zeroed work wire) |
| L2, paper blocks | `OuterPrepare`, `OuterUnprepare`, `OuterUnprepareFromOutcomes`, `InnerPrepare`, `InnerUnprepare`, `InnerUnprepareFromOutcomes`, `InnerSigns`, `TailAngleLoad`, `SFAngleLoad`, `AngleErase`, `SpinMux`, `SpinSwap`, `GivensRotation`, `GivensLadder`, `GivensLadderDagger`, `MajoranaGate`, `InnerSelect`, `InnerSelectDagger`; the reflection helpers `ref_t2` and `ref_ab` on PennyLane's `MultiControlledX` | `dfthc.ops.prepare`, `dfthc.ops.angle_data`, `dfthc.ops.spin`, `dfthc.ops.givens`, `dfthc.ops.majorana`, `dfthc.ops.select`, `dfthc.ops.reflections` |
| L3, assembly | `DFTHCBlockEncoding`, `DFTHCWalk` and its pinned `C(DFTHCWalk)` rule; `WalkLayout` | `dfthc.ops.walk`, `dfthc.registers` |
| L4, application | `ControlledWalkPower`, `QPERound`, the driver `iterative_phase_estimation` (a plain quantum function, spec 2.1 R-4), `qpe_program`, the energy helpers | `dfthc.qpe` |
| Tooling | gate set, pins and the compile helper; closed-form costs; the paper's and dfthcbe's costs; test support | `dfthc.compile`, `dfthc.costs`, `dfthc.costs_reference`, `dfthc.testing.resources`, `dfthc.testing.sparse_sim`, `dfthc.testing.tape`, `dfthc.testing.walk_sim`, `dfthc.testing.toy_walk`, `dfthc.testing.fixtures`, `dfthc.testing.canary`, `dfthc.testing.measure`, `dfthc.testing.provenance` |

## The walk, call by call

The walk `W = REF_{A u B} BE` is one program. The tree below is the pinned rules' call structure
(spec 3.5 gives the steps); the resource report has one row per distinct operator call, plus
explicit auxiliary and L4 program rows ([`resources.md`](resources.md)).

```text
DFTHCWalk                                steps 1-28
  DFTHCBlockEncoding                     1-27: U^dag REF_T2 U
    OuterPrepare                         1-4: FlaggedUniformPrep, QROAMRead (QROM, CSWAPs), keep test, CSWAPs
    (X measurement of junk_x_o)          the outer junk, right after the prepare
    InnerPrepare (round 1)               5-8: FlaggedUniformPrep (two PhaseGradientRotations), QROAMRead
    (X measurement of junk_b)            the inner junk, right after the prepare
    InnerSigns                           9
    InnerSelect                          10-18
      TailAngleLoad, SFAngleLoad         10, 11 (each a unary-iteration table load)
      SpinMux, SpinSwap                  12, 13
      GivensLadder                       14 (GivensRotation per rung)
      MajoranaGate                       15
      GivensLadderDagger                 16 (adjoint GivensRotation per rung)
      SpinSwap, SpinMux                  17
      AngleErase                         18
    InnerUnprepareFromOutcomes           19-22: CSWAPs, comparator adjoint, target measurement and fix-up, adjoint FlaggedUniformPrep
    ref_t2                               23: multi-controlled Z (MultiControlledX), GlobalPhase(pi)
    InnerPrepare (round 2), X measurement of junk_b, InnerSelectDagger (adjoint MajoranaGate), InnerSigns, InnerUnprepareFromOutcomes
    OuterUnprepareFromOutcomes           24-27: CSWAPs, comparator adjoint, target measurement and fix-up, adjoint FlaggedUniformPrep
  ref_ab                                 28: multi-controlled Z (MultiControlledX), GlobalPhase(pi)
```

The block encoding's rule X-measures each read's junk right after its prepare and gives the
outcomes to the matching unprepare from outcomes, which measures only the read's target and
fixes the phases from both outcome vectors (spec 16, Q-1 variant (b)); the baseline
`OuterUnprepare` and `InnerUnprepare`, which measure target and junk together through
`QROAMErase`, stay in the package with their tests but are not called by the walk.

The controlled walk is `qp.ctrl(DFTHCWalk(...), control)`. Its one rule, registered and pinned for
`"C(DFTHCWalk)"`, runs the block encoding's steps with only the T2 reflection controlled, then the
controlled `ref_ab`: on ancillas in `|0>` the rest is `U^dag U = I` (spec 3.3). A control value 0
is `X` on the control around the rule, conditioned on the (possibly traced) value. A controlled reflection takes the control as one more control of the
multi-controlled Z, then `Z` on the control (`ref_t2` and `ref_ab` with `control=`); the
reflections use `MultiControlledX` rather than `qp.FlipSign`, whose compile time grows
exponentially with the pattern width (L-29). `ControlledWalkPower` repeats it with one
`qp.for_loop`, and each
`QPERound` applies `H`, the walk power `2**j` times and the feed-forward phase corrections
(`PhaseGradientRotation` under `qp.cond`) on wires borrowed from the walk's `work` pool
(spec 11.1). `PhaseFixup` is the operator form of the erase fix-up that `QROAMErase`,
`AngleErase` and the unprepares from outcomes (through
`dfthc.ops.qroam.emit_erase_from_outcomes`) emit through `dfthc.ops.qroam.emit_phase_fixup`; the
walk does not call it as an operator. The toy walk of spec 11.4 (`ToyWalk` with its `C(ToyWalk)` rule,
`ToyControlledWalkPower` in `dfthc.testing.toy_walk`) checks the phase estimation exactly.

## Registers

`dfthc.registers.WalkLayout.from_params(params, qpe_bits=0)` gives every register named wires in
the groups `system`, `outer`, `inner`, `select`, `phase_grad` and `work` (spec 10.1), numbered by
one `qp.registers` call over `dfthc.registers.GROUPS`. The reads'
junk registers `junk_x_o` and `junk_b` have no wires of their own: each is the first wires of
`angles` (and of `junk_overflow` where a junk register is longer than the angles), because the
block encoding measures the junk right after each prepare, before any angle is loaded
(`dfthc.registers.JUNK`, spec 10.2). The assembly operators take the groups and split them once
with `WalkLayout.split_group` and `WalkLayout.split_junk`; every block they call takes only the
registers its spec 9 table names (spec 9.0, wire passing).

| Group | `H2_STO3G_TEST` | FeMoco-54 |
| --- | ---: | ---: |
| `system` (`psi_up`, `psi_dn`) | 4 | 108 |
| `outer` | 12 | 71 |
| `inner` | 16 | 65 |
| `select` (`angles`, `junk_overflow`, `G0`, `G1`, `b_neq_B`, `sg`) | 11 | 746 |
| `phase_grad` | 4 | 15 |
| `work` (the shared pool) | 8 | 44 |
| **walk** | **55** | **1,049** |

`angles` holds `N - 1` angle registers of `b_rot - 1` wires: the stored angles are below
`2**(b_rot - 1)`, so a `b_rot`-th bit would always be 0, and each Givens rung adds its
`(b_rot - 1)`-bit angle into the `b_rot`-bit `phase_grad` (spec 6.3, 6.7). The junk shares 72 + 390
wires with the 742 angle wires at FeMoco-54, so the walk needs 462 wires fewer than with junk
registers of its own; at `H2_STO3G_TEST` the 7 inner junk wires exceed the 3 angle wires, so
`junk_overflow` has 4. The paper's FeMoco-54 count is 1,131 (its Table 3): the
resource report compares the two by group ([`resources.md`](resources.md)).

The controlled walk adds one control wire; the QPE adds `qpe_control` and `qpe_grad` (t wires),
its corrections borrow the work pool, and spare wires follow only where 2t - 1 exceeds the pool
(spec 11.3). The comparators' work registers stay dirty from a prepare to its unprepare and belong
to their group, never to `work` (spec 10.2).

## Compilation

`dfthc.compile.compiled_resources(qfunc, num_wires)` compiles a program once (spec 7.2):
`qp.qjit(capture=True)` around `qp.decompose(gate_set=GATE_SET, fixed_decomps=...)` around a
`null.qubit` QNode, analyzed by `qp.analyze`; the pins are `FIXED_DECOMPS` (every dfthc rule) and
`PENNYLANE_PINS` (`QROM` to its unary-iteration rule, the select stage of `QROAMRead`; spec 7.5). It returns T (Toffoli plus
`TemporaryAND`; `GATE_SET` has no `Toffoli`, so T is the `TemporaryAND` count), M
(`MidCircuitMeasure`) and A (`Adjoint(TemporaryAND)`), and fails if the counts are symbolic or an
operation outside the gate set remains. Four rules carry the design
(spec 7.4):

- every dfthc operator has exactly one decomposition rule per key, registered and pinned with
  `dfthc.compile.register_rule`;
- every ancilla is an explicit wire argument, and every operator states
  `required_wires(**compilable_args)`;
- the adjoint forms of the composites are explicit classes (`OuterUnprepare`,
  `InnerUnprepare`, `InnerSelectDagger`, `GivensLadderDagger`, and the walk's
  unprepares `OuterUnprepareFromOutcomes` and `InnerUnprepareFromOutcomes`, which take the junk
  outcomes as a dynamic array), and the controlled walk is `qp.ctrl(DFTHCWalk)` with its pinned
  `C(DFTHCWalk)` rule, which controls only the reflections; no program calls `qp.adjoint` or
  `qp.ctrl` on any other composite, because Catalyst would capture the generic candidate of its
  whole subtree (the K-1 memory finding), and an inventory test checks it;
- the table loads keep their IR small: `QROAMRead`'s select is PennyLane's `QROM`, whose unary
  rule runs one `qp.for_loop` over the dynamic bool column array (spec 9.3), and the angle loads
  load each unary leaf with one `qp.for_loop` over a bool row array (spec 9.10, 9.11), so the
  FeMoco-54 walk compiles in a few GB.

## Costs and the resource report

`dfthc.costs` implements the closed forms of spec 9.24; every compiled count in the tests equals
one of them exactly. `dfthc.costs_reference` evaluates the paper's Table III formulas and ports
dfthcbe's per-block formulas, checked against dfthcbe's own call graph.
`scripts/resource_report.py` writes [`resources.md`](resources.md) and
[`resources.json`](resources.json): one row per program, measured where compiled, with the
closed form, dfthcbe's and the paper's values, the deltas and the reason codes that explain them,
each with its contribution (`dfthc.costs_reference.delta_causes`; spec 13.3). Its compiles are
cached with `dfthc.testing.provenance.code_hash()`, which covers `src/dfthc` outside
`dfthc.testing`, the report script and `requirements-nightly.lock`, while ignoring Python
docstrings and comments. The FeMoco-54 walk and controlled-walk compiles are
reused while that hash matches.

## Further reading

- [`conventions.md`](conventions.md): indices, signs, fermions, angles, registers.
- [`verification.md`](verification.md): what each test establishes and how to run the tiers.
- [`limitations.md`](limitations.md): the PennyLane and Catalyst limitations and their canaries.
- [`environment.md`](environment.md): the environment, the load limits and the commands.
- [`design.md`](design.md): the design specification, cited as "spec N.M".
