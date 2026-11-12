# Agent handoff: DFTHC block encoding in PennyLane

Every agent reads this first: implementers, reviewers and scouts. Your task brief holds your exact
requirements; this file holds the context they share. The spec is the source of truth:
`docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md` (cited below as "spec N.M").

## 1. The project in one paragraph

We implement the DFTHC sum-of-squares block encoding of Low et al., PRX 15, 041016 (2025), its
qubitized walk, the controlled walk and an iterative QPE wrapper. Every building block is a
PennyLane `Operator2` with exactly one pinned decomposition rule per key. Programs compile with
`qp.qjit(capture=True)` and `qp.decompose` to a Toffoli-level gate set, and resources are counted
exactly with `qp.analyze` and asserted against closed forms (`dfthc.costs`). Correctness rests on
three checks: per-block tests on `lightning.qubit`, a purpose-built sparse simulator end to end
on H2, and GF(2) outcome certificates for every measured erase. The reference implementation is
dfthcbe (Qualtran) at commit `930255020cec30172e0a7d17c0071d9476a687d1`. Differences from the
paper and dfthcbe are documented, never tuned away (R-7). Baseline at FeMoco-54: 9,793 Toffolis
per walk, 9,795 per controlled walk.

## 2. Where things are

| What | Where |
| --- | --- |
| Spec | `docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md` |
| Master plan (all tasks, orchestration, gates) | `docs/superpowers/plans/2026-11-11-dfthc-master-plan.md` |
| Code-level phase plans | `docs/superpowers/plans/2026-11-11-dfthc-phase<N>-*.md` |
| Ledger (status, spikes, decisions, gates) | `docs/agents/LEDGER.md` |
| Package | `src/dfthc/` (no editable install) |
| Tests | `tests/` (`data`, `ops`, `integration`, `sim`, `canary`, `report`) |
| Spikes | `spikes/` (not collected by pytest), results in `spikes/results/` |
| Vendored data | `data/dfthc/` (Apache-2.0, unmodified) |
| Scratch (never committed) | `.scratch/` |

## 3. Environment

### 3.1 The canonical environment

One environment serves every agent and worktree: `.venv` in the **main checkout**, CPython 3.12.
`scripts/setup_env.sh` creates it on a fresh machine with exactly these commands (spec 7.1):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane-catalyst --upgrade
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane --upgrade
pip install pytest pytest-xdist ruff
pip freeze > requirements-nightly.lock
```

Snapshot (2026-11-11): PennyLane `0.46.0.dev151`, Catalyst `0.16.0.dev140`, Lightning
`0.46.0.dev34`, JAX 0.7.1, pytest 9.2.0, pytest-xdist 3.8.0, ruff 0.17.4. Never recreate or
upgrade `.venv` inside a task: nightly bumps follow the procedure in `docs/environment.md` and
are orchestrator work.

### 3.2 Pitfalls

From spec 7.1:
- **Plain `pip` only, never `uv pip`.** uv installs the stable Catalyst from PyPI.
- **One expected pip error** on the second install: `ERROR: ... pennylane-catalyst 0.16.0.dev140
  requires pennylane==0.46.0-dev149, but you have pennylane 0.46.0.dev151`. It is intended.
- **A benign linker warning** on every compile: `ld: warning: object file ... was built for newer
  'macOS' version (28.0) than being linked (27.0)`.
- **Export `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` before Python starts.** Catalyst reads it
  once at import; setting it later is too late. `scripts/test.sh` exports it, and
  `tests/conftest.py` re-enables the warnings as a second line of defence.
- **Keep the lock file.** `--pre` with TestPyPI also pulls dev builds of `termcolor` and `urllib3`.
- **No editable install.** pytest's `pythonpath = ["src"]` makes every worktree test its own
  sources against the shared `.venv`.
- **Built-in warning noise is filtered** by `dfthc.testing.resources`. The autouse pytest fixture
  records every `RuleLoweringWarning`, fails the test on one that names a dfthc operator (except
  the expected measuring-rule warning below), and only then drops the noise below (Tasks 0.2,
  0.2b). Scripts outside pytest see the noise; the spikes classify it with
  `spikes/prototypes.classify_warnings`.
  - `Excluded the <rule> decomposition rule for <op> since --decompose-lowering cannot work with
    multiple registers yet` (for `_ppm`, `_*_ppm`, `_toffoli_elbow`, `_cphase_to_single_phase`,
    `use_allocation(...)` and the MCX `*_worker(s)` rules);
  - `Skipped the _adjoint_temporary_and(_ppm) decomposition rule for C(Adjoint(TemporaryAND)){...}: it contains a mid-circuit measurement, which cannot be placed in a control region.`;
  - the `qp.analyze` UserWarning "unable to determine the branch". It is one flag per compiled
    module: observed with the pinned toolchain on 2026-11-12, a program warns whose only
    branching source is one `TemporaryAND` with its measured uncompute, or one
    `qp.measure(..., reset=True)`, so nearly every program shows it. `qp.analyze` counts every
    unhinted conditional at its larger branch, so a conditional Toffoli would be counted without
    a visible signal; exact T4 counts are the backstop. Our conditionals carry hints, which exact
    expected counts check (Task 1C.4).
- **One expected warning that names a dfthc operator** (`classify_rule_warning` returns
  `"expected"`; tests record and drop it): `Skipped the <rule> decomposition rule for
  <Adjoint/C of a dfthc operator>{...}: it contains a mid-circuit measurement, which is not
  supported with adjoint or control regions.` Catalyst also lowers the generic `Adjoint(...)` and
  `C(...)` candidates of operators with pinned rules and drops the measuring ones; the pinned rules
  are used. A needed modified operator without a remaining rule fails with a `CompileError`; a
  remaining non-measuring rule is chosen silently, which exact T4 counts and T5 closure catch
  (spec 7.4 rule 12). Since G0 no walk program requests a modified dfthc composite (spec 7.4
  rule 6), so the warning appears only in tests that wrap a measuring operator on purpose
  (Tasks 1C.5 and 3.4).
- **"the rules file could not be parsed" alone proves nothing.** A successful verbose compile also
  prints it (spike T6). In a failed compile the real cause, `Decomposition rule not found for
  operator '<op>'`, is in the same error message; assert on that and the exception type.

Found while dry-running the Phase 0 plan (canaries L-24 and L-25 pin the first two):
- **Compilable integers must fit in signed 64 bits.** Store wide table entries as tuples of 32-bit
  words, least significant word first.
- **Apply adjoints as `qp.adjoint(Op(...))`.** Under capture, `qp.adjoint(Op)(*positional)`
  traces positional arguments and rejects compilable ones. Keyword arguments and
  `qp.adjoint(lambda: Op(...))()` also work.
- **Inside a rule, wire registers are arrays.** Never test them for truth (`work[:n] or None`
  raises); compute lengths from compilable arguments and slice explicitly.
- **Declared resources must match emitted operators exactly**, including compilable arguments and
  wire-register sizes. An emitted operator whose signature is not declared is left undecomposed,
  and `compiled_resources` then fails with "operations outside the gate set remain".
- **Under capture, constructing an operator already queues it.** Never `qp.apply` it again.
- **`qp.density_matrix` is not supported under capture.** Compare `qp.state()` instead.
- **The verbose rule listing needs `qp.qjit(verbose=True)`.** Use
  `dfthc.testing.resources.verbose_program` with pytest's `capfd` and `chosen_rules`.
- **Hinted conditionals count expected gates,** summed per gate type over the program and rounded
  up once.
- **Never pass bulk masks or parity data as compilable arguments.** Derive them inside the rule
  from the compilable table. In the plan's dry-run (quoted, not reproduced), one 8,640-address
  fixup compiled in about 4 min that way and in 25 min with masks as compilable data.
- **Git works in the Cursor sandbox:** branches, commits, the index, rebases and switches. Only
  git's control files (`.git/config`, hooks, new worktrees) are blocked, and you never need them.
  Do not request elevated permission for git; report a git sandbox failure to the orchestrator.
  For pip, set `PIP_NO_CACHE_DIR=1`, and request elevation only if pip still fails. Never use
  elevation to push.
- **The sandbox also forbids `ps` and setpriority()** (`nice`, `os.nice`; `nice` only warns and
  runs the command unchanged). `scripts/test.sh` and `dfthc.testing.watchdog` handle this: they
  lower the priority with `taskpolicy -c utility` and read memory through macOS `libproc`. Read a
  niceness with `os.getpriority`, never `os.nice(0)`. `pgrep` fails too ("Cannot get process
  list"), so a lane cannot see another lane's compiles: the orchestrator orders runs that need a
  quiet machine, and a brief never asks you to check for other processes.

### 3.3 Running Python and tests from a worktree lane

```bash
VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"
scripts/test.sh <pytest args>                    # always prefer this
CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" <script>
```

`scripts/test.sh` honours `DFTHC_VENV` (a different environment) and `DFTHC_TEST_WORKERS`
(default `2`; `0` runs serially), and runs pytest at lowered priority (section 3.5).
`tests/test_infrastructure.py` fails if the warning variable is not exported, the worker count
is not the requested one, or the priority is not lowered, so never run plain `pytest`.

### 3.4 `.venv-ref` for dfthcbe reference fixtures (Q-2: approved)

Only Task 1A.6 (and later refreshes) uses it, to run `scripts/make_dfthcbe_fixtures.py`. The
orchestrator builds it; nothing is cloned and no step needs elevated permission. dfthcbe comes
from the tarball of commit `930255020cec30172e0a7d17c0071d9476a687d1`, extracted into
`.scratch/dfthcbe-ref` (git-ignored) and checked by a SHA-256 digest of its sources:

```bash
mkdir -p .scratch/dfthcbe-ref && curl -sSfL \
    https://codeload.github.com/Covestro-HPQC/dfthcbe/tar.gz/930255020cec30172e0a7d17c0071d9476a687d1 \
    | tar -xz -C .scratch/dfthcbe-ref --strip-components=1
```

The complete recipe is the script in `docs/superpowers/plans/2026-11-11-dfthc-phase1-data.md`,
Task 1A.6, Step 7. It installs Qualtran 0.7.0 without its notebook dependencies and dfthcbe from a
copy of `.scratch/dfthcbe-ref` into `.venv-ref`, writes `tests/fixtures/dfthcbe/requirements-ref.lock`
and runs the fixture script in the lane. Two sandbox constraints shape it: pip verifies
certificates only with the macOS keychain's certificates (`PIP_CERT`), and the sandbox refuses any
directory named `venv` inside the workspace, so jedi (pulled in by IPython) cannot be installed and
`TMPDIR` must not point into a worktree while pytest runs. `.scratch/dfthcbe-ref` also serves
scouts and reviewers as the read-only dfthcbe source.

The fixture provenance records the commit, the codeload URL, the source and installed-package digests, the package versions (Qualtran 0.7.0) and a digest of every factor input; the lock names the codeload tarball, not the local copy. `.venv-ref/` is git-ignored.
If it cannot be created, fixture tests skip with an explicit reason.

### 3.5 Machine limits and heavy processes

The development laptop has **10 cores and 16 GB of memory**, and the user works on it while you
run. The user's directives (2026-11-11): "Keep large-scale compilations to a minimum" and "Try to
not run heavy processes throughout." The full policy is master plan section 12.9.

- **Heavy process:** any run whose process tree exceeds 4 GB of memory, keeps more than 2 cores
  busy or lasts longer than 30 minutes, and any FeMoco-shaped compile (a program that holds a
  FeMoco-54 table: the walk, the block encoding, a prepare, the select, a read, erase, fixup or
  angle operator at FeMoco-54 shape, or a spike stand-in at that shape).
- **You never start a heavy process.** If your brief needs one, it names the command; stop there
  and report it, and the orchestrator runs it alone, under the watchdog.
- **Memory cap:** 8 GB per process tree, a sampled limit (macOS enforces no memory limit).
  Compiles that could grow large run through `dfthc.testing.watchdog.run_watched`, which samples
  the tree every 50 ms, kills it at the first sample above 7.5 GB (`EFFECTIVE_LIMIT_MB`, 15/16 of
  the cap) and records the kill; the overshoot is bounded by one 50 ms interval of growth.
  Projections and budgets use 7.5 GB, not 8 GB. The watchdog fails closed: it starts nothing off
  macOS, and a libproc error ends the run as `"watchdog error: ..."` with the tree killed.
- **Test runs:** run only the targeted tests while you develop (`scripts/test.sh tests/x.py -q`),
  then the per-commit tier once at commit time. Never run the nightly or release tier: the
  orchestrator runs the nightly tier at phase gates, and the release tier once in Phase 5.
- **At most two implementers** run at once; a per-commit run with 2 workers is not heavy.
- **FeMoco values in tests** are asserted on the closed forms of `dfthc.costs`. Compiled counts
  are checked at `H2_STO3G_TEST`, `H2_631G_TEST` and `MID_NAPHTHALENE`. Table-free operators at
  FeMoco widths (`ZeroReflection` on 46 wires, a 53-rung `GivensLadder`) compile in seconds and
  are fine.
- **Measure memory as the watchdog does:** the summed physical footprint of the process tree. The
  resident size undercounts when the machine is under memory pressure.

## 4. Commands

| Purpose | Command |
| --- | --- |
| Fast suite (per commit, 15 min budget; once at commit time) | `scripts/test.sh -m "not slow and not release"` |
| Nightly tier (4 h; orchestrator only, at phase gates, under the watchdog) | `scripts/test.sh -m "slow and not release"` |
| Release tier (8 h; orchestrator only, once in Phase 5, after the user's go-ahead) | `scripts/test.sh` |
| Heavy run under the watchdog (orchestrator only) | `PYTHONPATH=src "$VENV/bin/python" -m dfthc.testing.watchdog --timeout-min 300 -- <command>` |
| One file or test | `scripts/test.sh tests/ops/test_x.py::test_name -q` |
| Serial run (debugging) | `DFTHC_TEST_WORKERS=0 scripts/test.sh ... -x` |
| Lint | `"$VENV/bin/ruff" check .` |
| Canaries only | `scripts/test.sh tests/canary -q` |
| Refresh the compile baseline (nightly bump only) | `DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 scripts/test.sh tests/canary/test_l21_compile_time.py` |
| Spike | `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/<script>.py` |

Markers: `compiled`, `slow`, `release` (implies `slow`), `canary`, `femoco`, `subprocess`. Mark
every compile test `compiled`, tests that spawn Python processes `subprocess`, and tests too slow
for the per-commit tier `slow`. `femoco` is only for the release-tier check of the cached FeMoco
counts (Task 4.5); no test compiles a FeMoco-shaped program.

## 5. Rules for operators and decomposition rules

Spec 7.4. Each rule exists because of a confirmed limitation (section 7 below).

1. **No allocation inside rules.** No `qp.allocate`, no `work_wires=` in
   `qp.register_resources`. Every ancilla is an explicit wire argument, and every operator has a
   static `required_wires(**compilable_args) -> dict[str, int]`.
2. **Resource functions.** Keys are classes with fixed signatures (`qp.CNOT`, `qp.X`, `qp.H`,
   `qp.Z`, `qp.CSWAP`, `qp.CZ`) or abstract `Operator2` instances built through
   `dfthc.compile.resource_key`, for example
   `resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3])))` or
   `resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True))`. Declare every emitted operator,
   including adjoint and controlled variants, with the exact compilable arguments and wire sizes.
   Counts are Python ints; resource functions never raise.
3. **Arguments.** Everything structural is in `compilable_argnames` (sizes, ranges, block bits,
   tables). Never use `static_argnames`, and never mix compilable and hybrid arguments.
   `dynamic_argnames` only for runtime data: `PhaseFixup` parities, `QPERound` earlier bits, and
   `QROAMRead`'s per-leaf table words (`int64`, built by the caller from its compilable table).
   Convert list-valued dynamic arguments to arrays in `__init__` (L-23).
4. **Control flow.** Bodies of auto-adjointed operators contain no `qp.for_loop` or `qp.cond`;
   unroll loops in Python with static bounds. `qp.cond` on measurement outcomes appears only
   inside erase operators and the QPE wrapper. `qp.for_loop` appears only in `QROAMRead` (one
   loop per unary leaf, with `qp.cond` on table bits; spec 9.3) and `ControlledWalkPower`, both
   never adjointed or controlled (rule 6).
5. **Measurement-based uncompute** lives only in `QROAMErase`, `AngleErase` and the base rules
   of `OuterUnprepare` and `InnerUnprepare`. Never place a measuring operator under `qp.ctrl`.
   Catalyst would silently control a unitary inverse instead (L-2).
6. **Adjoint and controlled forms of composites are explicit classes** (spec 7.4 rule 6, G0):
   `OuterUnprepare`, `InnerUnprepare`, `InnerSelectDagger`, `GivensLadderDagger`,
   `ControlledRefT2`, `ControlledRefAB`, `ControlledDFTHCBlockEncoding`, `ControlledDFTHCWalk`
   and `ControlledToyWalk`, each with one pinned base rule. Never write `qp.adjoint(...)` or
   `qp.ctrl(...)` of a dfthc composite: Catalyst would capture and lower the generic candidate
   of its whole subtree (the K-1 memory cost), and `qp.ctrl(DFTHCWalk(...))` does not compile.
   Leaf adjoints stay auto-derived (`PhaseGradientRotation`, `FlaggedUniformPrep`,
   `GivensRotation`, `MajoranaGate`). The one `C(Op)` rule left is `C(ZeroReflection)`, with
   signature `(base, control_wires, control_values, work_wires, work_wire_type)`. The inventory
   test (Task 3.4) catches a stray modifier.
7. **One rule per key, pinned.** Register every rule with
   `dfthc.compile.register_rule(OpClass | "Adjoint(Name)" | "C(Name)", rule)`; never call
   `qp.add_decomps` directly for a dfthc operator, and never pin a PennyLane operator (L-4).
8. **Unique class names.** `tests/test_compile.py` asserts that PennyLane registers no rule
   under any pinned name.
9. **Rules run in capture and tape mode.** Keep mode-specific classical code in helpers
   (`dfthc.compile.mcm_parities`).
10. **Exact swaps only** (`CSWAP`).
11. **No `Subroutine`, no `qp.pow`** on our operators.
12. **Warning policy.** Tests fail on any `RuleLoweringWarning` that names a dfthc operator,
    except Catalyst's "Skipped ... it contains a mid-circuit measurement" warning for an
    `Adjoint`/`C`-wrapped dfthc operator (section 3; spec 7.4 rule 12). A needed modified operator
    with no remaining rule fails the compile with a `CompileError`; a substitute rule is caught by
    the exact counts (T4).
13. **Adjoints use the instance form** `qp.adjoint(Op(...))` (L-25).
14. **Pass each child only the registers it needs** (spec 9.0): an operator bind costs 2
    captured equations per wire passed as a list and 4 per wire passed as an array register.

The canonical minimal example of an operator with a pinned measured adjoint rule is
`CompileTestLoad` in `tests/test_compile.py`. Production operators add the spec 9.0 docstring,
`required_wires`, constructor validation, and costs from `dfthc.costs`.

**Spec 9.0 docstring sections, in order:**
1. purpose and mathematical action;
2. register table (name, width, input state, output state);
3. compilable arguments;
4. pre- and postconditions;
5. decomposition outline;
6. adjoint and control behaviour;
7. cost formula (the `dfthc.costs` function);
8. paper equations and steps, and the dfthcbe bloq;
9. a minimal compiled example.

## 6. Conventions (spec 6)

- **Indices.** x_o < RC is SF with x_o = rC + c; the last N values are the one-body tail (N+ D1
  rows, then N- Q1 rows). G0 = 0 means SF and G0 = 1 one-body; G1 = 0 means D1 and G1 = 1 Q1.
  b = B is the identity term. r = floor(x_o / C).
- **Signs.** `InnerPrepare` loads sgn = [w_b < 0] and d_sgn = sgn ⊕ sgn[alt]. `InnerSigns`
  applies Z(sgn) and CZ(less_than_b, d_sgn).
- **Spins.** σ0 (`sigma_0`) is the inner spin or flavour, σ1 (`sigma_1`) the outer spin, and
  `sg` = σ_{G0} controls the spin swap.
- **Fermions.** Jordan-Wigner per spin register: γ_{j,0} = Z_0…Z_{j−1}X_j and
  γ_{j,1} = Z_0…Z_{j−1}Y_j, with qubit 0 the most significant and |1⟩ occupied. The system is
  ψ↑ ⊗ ψ↓. The Majorana gate acts on ψ↓[0]: SF applies −Z (dfthcbe's convention, not the paper's
  +Z), D1 applies a = (X + iY)/2, Q1 applies a† = (X − iY)/2.
- **Givens angles.** Rung k acts on (q_k, q_{k+1}). Angles are stored as
  x_k = round(θ_k/(2π) · 2^(b_rot−1)) mod 2^(b_rot−1), and the rung realizes θ = 4πx/2^b_rot
  exactly. The rung matrix on {|01⟩, |10⟩} is [[cos θ, −sin θ], [sin θ, cos θ]].
- **Alias sampling.** less_than = [keep ≤ m]; a set flag swaps in alt.
- **Phase gradient.** Prepared once by `PhaseGradientStatePrep` outside the walk and returned
  unchanged. Adding k multiplies by e^{+2πik/2^b}.
- **Reflections.** Exact including the global sign: `GlobalPhase(π)` uncontrolled, `Z` on the
  control when controlled. Never generic `C(GlobalPhase)`.
- **Wires.** Big-endian integers (the first wire is the most significant). Register names follow
  dfthcbe (Global Constraints). `InnerSelect` and `InnerSelectDagger` (spec 9.18) and the
  assembly operators (spec 9.21 to 9.23) take grouped wires (`outer`, `inner`, `select`,
  `system`, `phase_grad`, `work`) and split them only through `WalkLayout.split_group`; the
  blocks they call take the registers their spec 9 tables list (spec 9.0, wire passing).
- **Test tolerances.** Every `np.allclose`, `np.isclose`, `np.testing.assert_allclose` and
  `pytest.approx` states both tolerances: `rtol=0` (`rel=0`) with the contract's absolute `atol`
  (`abs`; 1e-12 for dense and simulator comparisons unless the brief names another), or an
  explicit `rtol` (`rel`) with `atol=0` (`abs=0`) where a relative bound is meant. NumPy's default
  `rtol=1e-5` silently loosens a 1e-12 check to about 1e-5 near amplitudes of order one, and
  `pytest.approx(x, rel=...)` keeps a default `abs=1e-12`. Plan code written before this rule often omits `rtol`; fix it
  when you copy the code and list it under deviations. `tests/test_tolerance_hygiene.py` enforces
  the rule on every call under `tests/` and `src/dfthc/testing/` (both keywords named in the call,
  `math.isclose` with `rel_tol` and `abs_tol`; no `**kwargs` forwarding), so a new comparison
  without both tolerances fails the per-commit tier.

## 7. Known limitations (spec 14, condensed)

Full entries: `docs/limitations.md`. Canaries: `tests/canary/`. **Changed** means the behaviour
differs from the spec's original claim (observed 2026-11-11).

| Id | Limitation | Mitigation |
| --- | --- | --- |
| L-1 | Rules declaring `work_wires=` are excluded, even pinned | explicit ancillas |
| L-2 | Measuring adjoint rules are skipped under control; **changed**: a unitary inverse is controlled instead | never control measuring ops; no modifier of a composite (rule 6) |
| L-3 | Unweighted solver prefers the auto inverse over the measured rule | weighted gate set, pins, T6 |
| L-4 | Pins apply to nested same-name operators | pin dfthc names only |
| L-5 | Rules see dynamic arguments as tracers | structure in compilable arguments |
| L-6 | Adjoint lowering can crash on nested control flow; **changed**: not reproduced | no control flow in auto-adjointed bodies; the looped `QROAMRead` is never adjointed (rule 6, inventory test) |
| L-7 | `C(ChangeOpBasis)` controls compute and uncompute under capture | explicit controlled classes (rule 6) |
| L-8 | Generic control adds PauliX gates | explicit controlled classes (rule 6) |
| L-9 | Controlled function calls; **changed**: left undecomposed, execution fails | no `Subroutine`; explicit controlled classes |
| L-10 | Compilation can depend on gate order | compiler blocker: reproducer and escalate |
| L-11 | Resource functions run with capture on | `resource_key` pauses capture |
| L-12 | Select-swap `QROM` is not self-inverse; symbolic counts | `QROAMRead` with CSWAPs, `PhaseFixup` |
| L-13 | `Select` with shared targets crashes | own unary iteration |
| L-14 | MCX without work wires has no path | explicit work wires |
| L-15 | hw-like postselection (**changed**: rejected); MCM in `cond` (**changed**: not reproduced); `pauli_measure` postselect ignored | spec 12.2 |
| L-16 | Execution decomposes again to device gates | counts at `level="user"`, closure at `"device"` |
| L-17 | `qp.analyze`: qjit only, max-branch, hints, symbolic loops | `compiled_resources`, hints, static loops |
| L-18 | Warnings silenced by default; misleading "could not be parsed" | export the variable; read the full message |
| L-19 | `UniformPrep`/`AliasSampling` need rotations (**changed**: power-of-two `AliasSampling` compiles) | `FlaggedUniformPrep` |
| L-20 | Undecomposed operator at runtime; **changed**: decomposed or rejected by the device pass | device-level closure test |
| L-21 | Each distinct table is a separate rule compile, while identical tables in one program share it; cost grows with table size (one 8,640-entry table: 25 s, 1.0 GB with children, one compile) | few tables; K-1 budget |
| L-22 | Unresolved loops give symbolic counts | static loops; assert non-symbolic |
| L-23 | List dynamic arguments not canonicalized | convert in `__init__` |
| L-24 | Compilable ints limited to int64 | 32-bit word tuples |
| L-25 | `qp.adjoint(Op)(*positional)` traces arguments | `qp.adjoint(Op(...))` |
| L-26 | Unsigned dynamic arguments (`uint32`, MLIR `ui32`) get no decomposition rule | signed arrays; table words `int64` with 32 data bits |
| L-27 | `qp.qjit` AOT-compiles fully annotated functions at decoration (zero arguments qualify vacuously); `qp.analyze` compiles again | `enable_aot_compilation=False` for analyze-only programs (`compiled_resources` does) |

## 8. Git rules

- Work only in the worktree lane named in your dispatch (`.worktrees/lane-a` or
  `.worktrees/lane-b` in the main checkout), on branch `phase<N>/<task-id>`. Do not touch the
  main checkout or the other lane.
- Never run `git worktree add/remove`, `git clone`, or anything that writes `.git/config` or
  hooks, and never switch the lane to another branch: the orchestrator frees the lane after the
  merge.
- Commit on your branch with the messages your brief gives. **Never push, never merge, never add,
  change or remove remotes.** Only the orchestrator merges, into local `main`.
- Identity: the repo-local `dwierichs` / `david.wierichs@xanadu.ai` (already configured).
- Never commit `.scratch/`, `.venv*/`, `.superpowers/` or generated caches.

## 9. Self-review checklist (fill in before reporting)

- [ ] Every requirement of the brief is implemented; nothing extra (YAGNI).
- [ ] TDD evidence: the failing run (command, key output) and the passing run.
- [ ] Task tests green; the fast suite green (`scripts/test.sh -m "not slow and not release"`,
      once at commit time); `ruff check .` clean; test output free of warnings.
- [ ] Every numerical comparison in the tests states both tolerances (section 6, test tolerances).
- [ ] No heavy process started (section 3.5); any heavy run the brief needs is named in the report.
- [ ] For each operator: the spec 9.0 docstring sections in order; `required_wires`; constructor
      validation with `ValueError`; T2 (`assert_valid_decomp_rule`), T3 (semantics), T4 (exact
      T, M and where relevant A against `dfthc.costs`), T5 (gate-set closure), T6 (rule
      selection) present.
- [ ] Declared resource keys equal the emitted operators (types, compilable arguments, wire
      sizes); rules registered with `register_rule` only.
- [ ] None of the forbidden constructs of section 5 (allocation, loops or conditionals in
      auto-adjointed bodies, measurements under control, `qp.adjoint` or `qp.ctrl` of a dfthc
      composite, `Subroutine`, `qp.pow`, positional `qp.adjoint(Op)(...)`, compilable ints of
      64 bits or more).
- [ ] Files match the brief's file list; no file grew beyond its responsibility.

## 10. Reporting back

Write the full report to the report file named in your dispatch, with these sections:
1. **Implemented:** what and where (files).
2. **Tests:** commands run and their key output lines (counts, pass/fail).
3. **TDD evidence:** RED (command, failing output, why it was expected); GREEN (command, output).
4. **Measurements** (spikes and compile-heavy tasks): time, peak memory of the process tree (the
   watchdog's sampled footprint, and the getrusage sum where recorded), counts; and any heavy run
   the orchestrator must still do.
5. **Self-review:** the checklist with findings.
6. **Concerns and deviations:** anything that differs from the brief, and why.

Then reply in at most 15 lines:
- **Status:** `DONE`, `DONE_WITH_CONCERNS`, `BLOCKED` or `NEEDS_CONTEXT`;
- commits (short SHA and subject);
- a one-line test summary (for example "72/72 passing, output pristine");
- concerns;
- the report path.

For `BLOCKED` or `NEEDS_CONTEXT`, put the specifics in the reply itself.

## 11. Notes for reviewers

- You review one task's diff from the review package. You are read-only: do not change the working
  tree, the index or any branch.
- Treat the implementer's report as claims to verify. Do not re-run the suite; run one focused
  test only for a specific doubt.
- Cite file:line for every finding. Severity: Critical (wrong results, data loss, broken
  contract), Important (a missed requirement, fragile behaviour, a test that asserts nothing,
  duplicated logic), Minor (polish).
- Project-specific checks: the spec 9.0 docstring; T2 to T6; resources declared equal to emitted;
  pins via `register_rule`; exact counts from `dfthc.costs`; no forbidden constructs; costs and
  semantics against the spec section the brief cites.

## 12. When to stop and escalate

- An expected output in the brief differs and you cannot see why.
- The brief and the spec disagree, or the brief cannot be followed without touching other files.
- A compiler crash, or a circuit whose compilation depends on gate order (L-10): keep a minimal
  reproducer under `tests/canary/reproducers/` and stop.
- A canary flips: record the observation and report `DONE_WITH_CONCERNS`.
- A run would be heavy (section 3.5), or a compile is killed at its memory cap: stop and report;
  the orchestrator decides.
- Anything that would require pushing, changing remotes, or recreating `.venv`.
