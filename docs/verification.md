# Verification

What the tests establish, where they are, and how to run them (spec 12). Correctness rests on
three independent checks: per-block semantics on `lightning.qubit` and the sparse simulator, the
end-to-end checks of the whole walk on the sparse simulator, and GF(2) certificates for every
measured erase. Resources are asserted exactly against closed forms.

## Test taxonomy (spec 12.1)

| Code | What it establishes | Where |
| --- | --- | --- |
| T1 | classical data: tables, normalizations, both offset frames, the encoded model, agreement with dfthcbe's fixtures | `tests/data/` |
| T2 | rule validity: declared resources equal emitted operators (`qp.testing.assert_valid_decomp_rule`) | `tests/ops/`, `tests/test_qpe.py` |
| T3 | semantics against NumPy references, compiled on `lightning.qubit` or on the sparse simulator | `tests/ops/`, `tests/integration/` |
| T4 | exact T, M and A against `dfthc.costs`: table-bearing blocks at the two H2 presets (per commit) and `MID_NAPHTHALENE` (nightly), table-free blocks over representative widths, the QPE round and program at H2 sizes, and FeMoco-54 on closed forms | `tests/ops/`, `tests/test_qpe.py`, `tests/integration/test_walk_counts.py`, `tests/integration/test_qpe_counts.py`, `tests/test_costs.py` |
| T5 | gate-set closure at `level="user"` and no dfthc operator left at `level="device"` | `tests/ops/`, `tests/test_qpe.py` |
| T6 | the pinned rules are chosen (count signatures, verbose rule listings) and the inventory test: no modified dfthc composite is captured | `tests/ops/`, `tests/test_qpe.py`, `tests/ops/test_walk.py` |
| T7 | canaries: one test per claim of every known limitation | `tests/canary/` |
| T8 | end-to-end checks E-1 to E-4 | `tests/integration/test_end_to_end.py` |

Supporting tests: `tests/test_work_pool.py` (every call gets exactly its `required_wires`),
`tests/sim/` (the simulator's differential tests), `tests/test_compile.py` and
`tests/test_primitive_costs.py` (the gate set and PennyLane's primitive costs),
`tests/report/` (the resource report), `tests/test_docstrings.py` (the documentation).

## The sparse simulator (spec 12.3)

`dfthc.testing.tape.decompose_for_simulation` decomposes a program on the tape path with the
pinned rules, and `dfthc.testing.sparse_sim` applies it to a sparse state with one handler per
gate-set member. `TemporaryAND` and its measured uncompute assert their preconditions on every
term. In strict mode every measurement that is neither a branch wire nor postselected must have
probability 1/2. Measurement policies: forced schedules (the certificates), seeded random
outcomes, and branching on listed wires (the QPE control). `dfthc.testing.walk_sim` builds the
walk cases (`sto3g`, `631g`, `fails-c3`) and their schedules.

## Outcome certificates (spec 12.5)

An erase X-measures M qubits; its fix-up must cancel the phase `(-1)^{xi . g(a)}` for every
outcome vector `xi`. The residual is GF(2)-linear in `xi`, so the zero vector and the M unit
vectors prove it for all `2^M` outcomes: each block runs M + 1 forced schedules on the simulator,
after checking that every conditional is diagonal with a linear predicate. Certified blocks:
`QROAMRead` with `QROAMErase`, the angle loads with `AngleErase`, each prepare with its
unprepare, and each prepare with the X measurement of its junk and its unprepare from outcomes,
where `xi` ranges over the junk outcomes and the target outcomes together (spec 16, Q-1 variant
(b); `tests/ops/test_certificates_qroam.py`, `test_certificates_angles.py`,
`test_certificates_prepare.py`). A test operator whose rule holds the inner prepare, the junk
measurement and the unprepare from outcomes certifies the junk outcomes crossing an operator
boundary inside another rule, as in the block encoding
(`tests/ops/test_unprepare_from_outcomes.py`).

The walk measures each read's junk right after its prepare, in the block encoding's rule, and the
target in the unprepare. Nothing in the simulator changes for that: a forced schedule assigns
outcomes in program order, so the junk outcomes come first; a fix-up's predicate is a parity of
`MeasurementValue`s of both measurements, evaluated on the recorded outcomes when the
conditional runs; strict mode asserts probability 1/2 for each of them. The end-to-end schedules
(all zeros, all ones, alternating, seeded) keep their meaning, and the total number of
measurements is unchanged.

## End-to-end checks (spec 12.4)

On `H2_STO3G_TEST` (and E-1, E-2 on `FAILS_C3`), with every ancilla in `|0>`:

| Check | Statement |
| --- | --- |
| E-1 | `<0|BE|0> = H_SA/Lambda - I` to 1e-10, column by column |
| E-2 | `<0|W^k|0> = T_k(H_SA/Lambda - I)` for k = 1, 2 |
| E-3 | the controlled walk with the control in `|0>`, `|1>`, `|+>`, and after one uncontrolled step |
| E-4 | prepare then unprepare returns every ancilla to `|0>` |

The oracle is `dfthc.data.encoded.encoded_block`. QPE of the walk itself is checked on the
simulator against the encoded block's exact phase distribution
(`tests/integration/test_qpe_walk.py`), and the toy walk's QPE exactly and on `lightning.qubit`
(`tests/integration/test_qpe_toy.py`).

## Resources

Every compiled T, M and A count equals `dfthc.costs`. Table-bearing blocks use the two H2 presets
per commit and `MID_NAPHTHALENE` nightly; table-free blocks use representative register widths,
including FeMoco-54 widths where inexpensive. The QPE tests compile the round and full program at
H2 sizes and `ControlledWalkPower` at `MID_NAPHTHALENE`. The FeMoco-54 values (walk 9,790
Toffolis, controlled walk 9,792, M = 2,524, A = 8,672 and 8,674) are asserted on the closed forms
per commit. The resource report compiles every program of the walk at the H2 systems and
`naphthalene` (and `fe2s2`, `fe4s4` as heavy runs), checks each against its closed form, and
checks that the partition rows times their calls give the compiled walk. Task 4.5 compiled the
FeMoco-54 walk and controlled walk once each and cached them in [`resources.json`](resources.json);
`tests/report/test_femoco_cache.py` (release tier) asserts, without compiling, the cached counts,
their gate dictionaries and wires, the partition sum (plus the block encodings' junk
measurements in M) and the QPE summary, while the code hash matches.

## Tiers (spec 12.6)

| Tier | Command | Content | Budget |
| --- | --- | --- | --- |
| per commit | `scripts/test.sh -m "not slow and not release"` | T1 to T6 at small sizes, every canary except the L-21 compile-time benchmark (subprocess canaries included), simulator tests, the report at `H2_STO3G_TEST` | 15 min |
| nightly (phase gates) | `scripts/test.sh -m "slow and not release"` | certificates, E-1 to E-4 under the nightly schedules, T4 at `MID_NAPHTHALENE`, the report at `naphthalene`, the L-21 compile-time benchmark | 4 h |
| release (once) | `scripts/test.sh` | everything, with E-1 and E-3 under more schedules and the cached FeMoco-54 counts | 8 h |

`scripts/test.sh` runs two workers at lowered priority (`DFTHC_TEST_WORKERS` overrides). One
test: `scripts/test.sh tests/ops/test_walk.py::test_name -q`. The nightly and release tiers and
every heavy run are started by the orchestrator under the watchdog (see
[`environment.md`](environment.md)).
