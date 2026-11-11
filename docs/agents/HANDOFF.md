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
- **Built-in warning noise is filtered** by `dfthc.testing.resources`:
  - `Excluded the <rule> decomposition rule for <op> since --decompose-lowering cannot work with
    multiple registers yet` (for `_*_ppm`, `_toffoli_elbow`, `_cphase_to_single_phase`,
    `use_allocation(...)` and the MCX `*_worker(s)` rules);
  - `Skipped the _adjoint_temporary_and... rule for C(Adjoint(TemporaryAND))`;
  - the `qp.analyze` UserWarning "unable to determine the branch", which PennyLane's own
    `SemiAdder` triggers.
- **"the rules file could not be parsed" means no decomposition path.** The real cause,
  `Decomposition rule not found for operator '<op>'`, is in the same error message.

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
  from the compilable table. One 8,640-address fixup compiled in about 4 min that way and in 25 min
  with masks as compilable data.
- **The Cursor sandbox may block writes** to `.git` (worktrees, commits) or `.venv` (pip). Request
  the elevated permission for that single command. Never use it to push.

### 3.3 Running Python and tests from a worktree

```bash
VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"
scripts/test.sh <pytest args>                    # always prefer this
CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" <script>
```

`scripts/test.sh` honours `DFTHC_VENV` (a different environment) and `DFTHC_TEST_WORKERS`
(default `auto`; `0` runs serially). `tests/test_infrastructure.py` fails if the warning variable
is not exported, so never run plain `pytest`.

### 3.4 `.venv-ref` for dfthcbe reference fixtures (Q-2: approved)

Only Task 1A.6 (and later refreshes) uses it, to run `scripts/make_dfthcbe_fixtures.py`:

```bash
python3.12 -m venv .venv-ref
.venv-ref/bin/pip install --upgrade pip
.venv-ref/bin/pip install "dfthcbe @ git+https://github.com/Covestro-HPQC/dfthcbe@930255020cec30172e0a7d17c0071d9476a687d1"
.venv-ref/bin/pip freeze > tests/fixtures/dfthcbe/requirements-ref.lock
.venv-ref/bin/python scripts/make_dfthcbe_fixtures.py
```

dfthcbe requires `qualtran>=0.7`. The fixture provenance records the resolved Qualtran version.
`.venv-ref/` is git-ignored. If it cannot be created, fixture tests skip with an explicit reason.

## 4. Commands

| Purpose | Command |
| --- | --- |
| Fast suite (per commit, 15 min budget) | `scripts/test.sh -m "not slow and not release"` |
| Nightly tier (4 h) | `scripts/test.sh -m "slow and not release"` |
| Release tier (12 h) | `scripts/test.sh` |
| One file or test | `scripts/test.sh tests/ops/test_x.py::test_name -q` |
| Serial run (debugging) | `DFTHC_TEST_WORKERS=0 scripts/test.sh ... -x` |
| Lint | `"$VENV/bin/ruff" check .` |
| Canaries only | `scripts/test.sh tests/canary -q` |
| Refresh the compile baseline (nightly bump only) | `DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 scripts/test.sh tests/canary/test_l21_compile_time.py` |
| Spike | `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 PYTHONPATH=src "$VENV/bin/python" spikes/<script>.py` |

Markers: `compiled`, `slow`, `release` (implies `slow`), `canary`, `femoco`, `subprocess`. Mark
every compile test `compiled`, FeMoco-size tests `slow` and `femoco`, and tests that spawn
Python processes `subprocess`.

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
   the K-1 remedy tables if activated. Convert list-valued dynamic arguments to arrays in
   `__init__` (L-23).
4. **Control flow.** Bodies of auto-adjointed operators contain no `qp.for_loop` or `qp.cond`;
   unroll loops in Python with static bounds. `qp.cond` appears only on measurement outcomes,
   inside erase operators and the QPE wrapper. `qp.for_loop` appears only in
   `ControlledWalkPower` (plus the K-1 remedy exception).
5. **Measurement-based uncompute** lives only in `QROAMErase`, `AngleErase` and explicit
   `Adjoint(Op)` rules with signature `(base)`. Never place a measuring operator under `qp.ctrl`.
   Catalyst would silently control a unitary inverse instead (L-2).
6. **Controlled versions** come only from explicit `C(Op)` rules with signature
   `(base, control_wires, control_values, work_wires, work_wire_type)`, for `ZeroReflection`,
   `RefT2`, `RefAB`, `DFTHCBlockEncoding`, `DFTHCWalk` and `ToyWalk`.
7. **One rule per key, pinned.** Register every rule with
   `dfthc.compile.register_rule(OpClass | "Adjoint(Name)" | "C(Name)", rule)`; never call
   `qp.add_decomps` directly for a dfthc operator, and never pin a PennyLane operator (L-4).
8. **Unique class names.** `tests/test_compile.py` asserts that PennyLane registers no rule
   under any pinned name.
9. **Rules run in capture and tape mode.** Keep mode-specific classical code in helpers
   (`dfthc.compile.mcm_parities`).
10. **Exact swaps only** (`CSWAP`).
11. **No `Subroutine`, no `qp.pow`** on our operators.
12. **Warning policy.** Tests fail on any `RuleLoweringWarning` that names a dfthc operator.
13. **Adjoints use the instance form** `qp.adjoint(Op(...))` (L-25).

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
  dfthcbe (Global Constraints). Composite operators take grouped wires (`outer`, `inner`,
  `select`, `system`, `phase_grad`, `work`) and split them only through
  `WalkLayout.split_group`.

## 7. Known limitations (spec 14, condensed)

Full entries: `docs/limitations.md`. Canaries: `tests/canary/`. **Changed** means the behaviour
differs from the spec's original claim (observed 2026-11-11).

| Id | Limitation | Mitigation |
| --- | --- | --- |
| L-1 | Rules declaring `work_wires=` are excluded, even pinned | explicit ancillas |
| L-2 | Measuring adjoint rules are skipped under control; **changed**: a unitary inverse is controlled instead | never control measuring ops |
| L-3 | Unweighted solver prefers the auto inverse over the measured rule | weighted gate set, pins, T6 |
| L-4 | Pins apply to nested same-name operators | pin dfthc names only |
| L-5 | Rules see dynamic arguments as tracers | structure in compilable arguments |
| L-6 | Adjoint lowering can crash on nested control flow; **changed**: not reproduced | no control flow in auto-adjointed bodies |
| L-7 | `C(ChangeOpBasis)` controls compute and uncompute under capture | explicit `C(...)` rules |
| L-8 | Generic control adds PauliX gates | explicit `C(...)` rules |
| L-9 | Controlled function calls; **changed**: left undecomposed, execution fails | no `Subroutine` |
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
| L-21 | Compile cost grows with tables (one 8,640-entry table: 21 s, 1.4 GB) | few tables; K-1 budget |
| L-22 | Unresolved loops give symbolic counts | static loops; assert non-symbolic |
| L-23 | List dynamic arguments not canonicalized | convert in `__init__` |
| L-24 | Compilable ints limited to int64 | 32-bit word tuples |
| L-25 | `qp.adjoint(Op)(*positional)` traces arguments | `qp.adjoint(Op(...))` |

## 8. Git rules

- Work only in your task's worktree (`../dfthc-be-worktrees/<task-id>`) on branch
  `phase<N>/<task-id>`. Do not touch the main checkout or other worktrees.
- Commit on your branch with the messages your brief gives. **Never push, never merge, never add,
  change or remove remotes.** Only the orchestrator merges, into local `main`.
- Identity: the repo-local `dwierichs` / `david.wierichs@xanadu.ai` (already configured).
- Never commit `.scratch/`, `.venv*/`, `.superpowers/` or generated caches.

## 9. Self-review checklist (fill in before reporting)

- [ ] Every requirement of the brief is implemented; nothing extra (YAGNI).
- [ ] TDD evidence: the failing run (command, key output) and the passing run.
- [ ] Task tests green; the fast suite green (`scripts/test.sh -m "not slow and not release"`);
      `ruff check .` clean; test output free of warnings.
- [ ] For each operator: the spec 9.0 docstring sections in order; `required_wires`; constructor
      validation with `ValueError`; T2 (`assert_valid_decomp_rule`), T3 (semantics), T4 (exact
      T, M and where relevant A against `dfthc.costs`), T5 (gate-set closure), T6 (rule
      selection) present.
- [ ] Declared resource keys equal the emitted operators (types, compilable arguments, wire
      sizes); rules registered with `register_rule` only.
- [ ] None of the forbidden constructs of section 5 (allocation, loops or conditionals in
      auto-adjointed bodies, measurements under control, `Subroutine`, `qp.pow`, positional
      `qp.adjoint(Op)(...)`, compilable ints of 64 bits or more).
- [ ] Files match the brief's file list; no file grew beyond its responsibility.

## 10. Reporting back

Write the full report to the report file named in your dispatch, with these sections:
1. **Implemented:** what and where (files).
2. **Tests:** commands run and their key output lines (counts, pass/fail).
3. **TDD evidence:** RED (command, failing output, why it was expected); GREEN (command, output).
4. **Measurements** (spikes and compile-heavy tasks): time, memory, counts.
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
- Anything that would require pushing, changing remotes, or recreating `.venv`.
