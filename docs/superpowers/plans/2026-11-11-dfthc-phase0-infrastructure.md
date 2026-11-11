# DFTHC Phase 0, Part 1: Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the repository scaffold, the canonical environment and lock, the test runner, the compile helpers (`dfthc.compile`), the resource and canary test helpers, and the vendored dfthcbe data, so that every later task compiles, counts and tests the same way.

**Architecture:** A `src`-layout package `dfthc` with no editable install; pytest finds it through `pythonpath = ["src"]`, so every git worktree runs its own sources against the shared `.venv` of the main checkout. `scripts/test.sh` exports the Catalyst warning variable before Python starts and locates that `.venv` from any worktree. `dfthc.compile` owns the gate set, the pinned rules and the exact-count compile helper that every operator test uses.

**Tech Stack:** Python 3.12, PennyLane `0.46.0.dev151` and Catalyst `0.16.0.dev140` nightlies, JAX 0.7.1, pytest 9 with pytest-xdist, ruff.

This file holds Tasks 0.1, 0.2 and 0.5 of the master plan (`docs/superpowers/plans/2026-11-11-dfthc-master-plan.md`). Tasks 0.3 and 0.4 are in `2026-11-11-dfthc-phase0-canaries.md`; Tasks 0.6 to 0.9 in `2026-11-11-dfthc-phase0-spikes.md`. Every code block below was executed on 2026-11-11 in a scratch replica of the repository (`.scratch/plan-check/repo`, not committed) and passed as stated.

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim from the
spec (`docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md`); section numbers refer to it.

**Standing requirements (spec 2.1)**
- R-1: Use the latest nightly PennyLane and Catalyst, installed as in §7.1.
- R-2: Compile with `qp.qjit(capture=True)` and `qp.decompose` (the graph decomposition system). Obtain resources with `qp.analyze`.
- R-3: Decomposition rules are lowered as part of the `qjit` call. The precompiled decomposition-rule bytecode file is not used.
- R-4: Every building block is an `Operator2` with clear documentation. The code is modular. Documented exemption, decided by the user (§16, Q-3): a top-level algorithm that returns classical measurement values, namely the iterative QPE driver of §11, is a plain quantum function. Its quantum content per round remains `Operator2`s (`QPERound`, `ControlledWalkPower`).
- R-5: Resource-aware tests accompany all development.
- R-6: Verifiability, testability and comprehensiveness: other people must be able to understand what was done and how the code works.
- R-7: No workarounds to match the paper's or dfthcbe's resource counts beyond the paper's own constructions. Deltas are documented.
- R-8: Scope: block encoding, walk, controlled walk, and an iterative QPE wrapper on the controlled walk. All are compiled and resource-counted. QPE is simulated only at toy sizes.
- R-9: Gate set: Toffoli level, with rotations done by phase-gradient addition (§7.3).
- R-10: Verification: a purpose-built sparse-state simulator checks ⟨0|BE|0⟩ = H_SA/Λ − I end to end on H2. Correctness for every erase-measurement outcome is certified at block level (§12.5). Per-block checks run on `lightning.qubit`.
- R-11: Git: local branches only. Never push and never modify remotes.

**Toolchain (spec header and 7.1)**
- Toolchain snapshot: PennyLane `0.46.0.dev151`, Catalyst `0.16.0.dev140`, Lightning `0.46.0.dev34` (nightlies of 2026-11-11).
- One canonical environment serves all agents and worktrees: `.venv` in the repository root, CPython 3.12.
- Always use plain `pip`; never `uv pip`.
- Export `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` before Python starts.
- Do not install the package in editable mode. Use pytest's `pythonpath = ["src"]`.
- Reference fixtures come from `.venv-ref`, a separate environment with Qualtran and dfthcbe at commit `930255020cec30172e0a7d17c0071d9476a687d1` (Q-2, decided: yes).

**Gate set and accounting (spec 7.3 and 7.6)**
```python
GATE_SET = {
    "PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "Adjoint(S)": 1,
    "CNOT": 1, "CZ": 1, "SWAP": 1, "GlobalPhase": 1, "MidMeasure": 1,
    "Adjoint(TemporaryAND)": 1, "PhaseGradientStatePrep": 1,
    "Toffoli": 1000, "TemporaryAND": 1000,
}
```
- Toffoli count T: number of `Toffoli` plus number of `TemporaryAND`. M: number of `MidCircuitMeasure`, the explicit erase measurements. A: number of `Adjoint(TemporaryAND)`, the measured AND uncomputes.
- What tests assert: Toffoli and measurement counts are exact and asserted against `dfthc.costs` closed forms. Clifford counts are reported, not asserted.
- Baseline at FeMoco-54 (§9.24): 9,793 Toffolis per walk and 9,795 per controlled walk; M = 2,524; A = 8,675 (controlled 8,677); M + A = 11,199 (controlled 11,201).

**Rules for operators and decomposition rules (spec 7.4, one line each)**
1. No allocation inside rules: no `qp.allocate`, no `work_wires=` in `qp.register_resources`; every ancilla is an explicit wire argument; every operator exposes `required_wires(**compilable_args) -> dict[str, int]`.
2. Resource-function keys are `Operator2` abstract instances built through `dfthc.compile.resource_key`; every emitted operator is declared; counts are Python integers; resource functions never raise.
3. Every argument that determines circuit structure is in `compilable_argnames`; `static_argnames` are not used; `dynamic_argnames` only for runtime data (`PhaseFixup` parities, `QPERound` earlier bits, K-1 remedy tables).
4. Bodies of auto-adjointed operators contain no `qp.for_loop` or `qp.cond`; `qp.cond` only on mid-circuit measurement outcomes in erase operators and the QPE wrapper; `qp.for_loop` only in `ControlledWalkPower`.
5. Measurement-based uncompute lives only in `QROAMErase`, `AngleErase` and explicit `Adjoint(Op)` rules with signature `(base)`; an operator that contains a measurement is never placed under `qp.ctrl`.
6. Controlled versions come only from explicit `C(Op)` rules with signature `(base, control_wires, control_values, work_wires, work_wire_type)`.
7. One base rule per operator; explicit adjoint and controlled rules are pinned (§7.5).
8. Unique class names; a test asserts that PennyLane registers none of our names.
9. Rules run in both capture and tape mode; mode-specific classical code is confined to helpers (`dfthc.compile.mcm_parities`).
10. Exact swaps only (`CSWAP`).
11. No `Subroutine`, and no `qp.pow` on our operators.
12. Tests fail on any `RuleLoweringWarning` that names a dfthc operator, and filter the PennyLane noise of §7.1.

**Rules added by the Phase 0 dry-run (2026-11-11; canaries L-24, L-25)**
- Compilable integers must fit in signed 64 bits; table entries are tuples of 32-bit words, least significant word first.
- Apply adjoints as `qp.adjoint(Op(...))`, never `qp.adjoint(Op)(*positional_args)`: under capture the positional form traces compilable arguments.

**Conventions (spec 6.7)**
- Integers in registers are big-endian: the first wire is the most significant bit.
- Register names follow dfthcbe: `x_o`, `succ_x_o`, `sigma_1`, `m_x_o`, `less_than_x_o`, `alt_x_o`, `keep_x_o`, `junk_x_o`, `b`, `succ_b`, `sigma_0`, `m_b`, `less_than_b`, `alt_b`, `keep_b`, `sgn`, `d_sgn`, `r`, `junk_b`, `angles`, `G0`, `G1`, `b_neq_B`, `sg`, `psi_up`, `psi_dn`, `phase_grad`, `work`.

**Tests (spec 12.6)**
- Markers: `compiled`, `slow`, `release`, `canary`, `femoco`, `subprocess`. `release` implies `slow`.
- Per commit: `scripts/test.sh -m "not slow and not release"` (15 min). Nightly: `scripts/test.sh -m "slow and not release"` (4 h). Release: `scripts/test.sh` (12 h).

**Git (spec 15)**
- Local branches and worktrees only. Merges go into local `main`. Never push, never modify remotes. The repo-local git identity (`dwierichs` / `david.wierichs@xanadu.ai`) is shared by all worktrees.
- Each task runs in a git worktree on a branch named `phase<N>/<task-id>` from local `main`; only the orchestrator merges.

---

## File structure

| Path | Task | Responsibility |
| --- | --- | --- |
| `pyproject.toml` | 0.1 | package metadata, pytest configuration and markers, ruff configuration |
| `requirements-nightly.lock` | 0.1 | `pip freeze` of the canonical environment |
| `scripts/setup_env.sh` | 0.1 | the environment recipe of spec 7.1 |
| `scripts/test.sh` | 0.1 | exports the warning variable, finds the shared `.venv`, runs pytest |
| `src/dfthc/__init__.py` | 0.1 | package root and version |
| `tests/conftest.py` | 0.1, replaced in 0.2 | warning filters, `release` implies `slow`, the rule-warning policy |
| `tests/test_infrastructure.py` | 0.1 | scaffold, environment and runner checks |
| `docs/environment.md` | 0.1 | snapshot, recipe, pitfalls, bump procedure |
| `src/dfthc/compile.py` | 0.2 | `GATE_SET`, `FIXED_DECOMPS`, `register_rule`, `resource_key`, `compiled_resources`, `mcm_parities` |
| `src/dfthc/testing/__init__.py` | 0.2 | test-support package |
| `src/dfthc/testing/resources.py` | 0.2 | warning noise and policy, `assert_resources`, verbose rule listings |
| `src/dfthc/testing/canary.py` | 0.2 | subprocess runner for canaries |
| `tests/test_compile.py` | 0.2 | compile helpers, pins, the T6 pattern, parity linearity |
| `tests/test_primitive_costs.py` | 0.2 | the PennyLane primitive costs of spec 9.24 |
| `tests/test_canary_helper.py` | 0.2 | the canary runner |
| `data/dfthc/` | 0.5 | vendored dfthcbe data, `LICENSE`, `NOTICE` |
| `tests/data/test_vendored.py` | 0.5 | file set, SHA-256 digests, license and notice |

Shell conventions used in the steps: run commands from the worktree root. Before `scripts/test.sh` exists (Task 0.1, Step 2), locate the shared environment with

```bash
VENV="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/.venv"
```

---

### Task 0.1: Repository scaffold, environment, lock and test runner

**Task id:** `0.1` · **Branch:** `phase0/0.1` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 (`gpt-5.6-sol-high-fast`) · **Depends on:** nothing

**Files:**
- Create: `pyproject.toml`, `requirements-nightly.lock`, `scripts/setup_env.sh`, `scripts/test.sh`, `src/dfthc/__init__.py`, `tests/conftest.py`, `tests/test_infrastructure.py`, `docs/environment.md`

**Interfaces:**
- Consumes: the existing `.gitignore` (already ignores `.venv/`, `.venv-ref/`, `.scratch/` and caches); the canonical `.venv` in the main checkout.
- Produces:
  - `scripts/test.sh [pytest args...]`: exports `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0`, uses `$DFTHC_VENV` or `<main checkout>/.venv`, runs `python -m pytest -n ${DFTHC_TEST_WORKERS:-auto} "$@"` from the worktree root.
  - pytest markers `compiled`, `slow`, `release`, `canary`, `femoco`, `subprocess`, with `--strict-markers`; the `release` marker adds `slow` before `-m` deselection.
  - `import dfthc` works in every worktree via `pythonpath = ["src"]`; `dfthc.__version__ == "0.1.0"`.

**Acceptance tests** (`tests/test_infrastructure.py`):
- `test_package_imports`: `dfthc.__version__ == "0.1.0"`.
- `test_warning_variable_exported_before_python_started`: the variable is `"0"` inside the test process.
- `test_rule_lowering_warnings_are_recorded`: a `RuleLoweringWarning` is recorded under the session's filters.
- `test_lock_file_pins_installed_versions[pennylane|pennylane_catalyst|pennylane_lightning]`: the lock pins the installed versions.
- `test_scripts_are_executable[...]`: both scripts are executable.
- `test_release_marker_implies_slow`: `-m "not slow"` deselects a `release`-only test.

**Definition of done:** 10 passed with `scripts/test.sh tests/test_infrastructure.py`; 9 passed with the per-commit filter (one test deselected); `ruff check .` clean; `docs/environment.md` present; one commit on `phase0/0.1`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_infrastructure.py`:

```python
"""Repository scaffold, environment and test-runner checks (plan Task 0.1)."""

import os
import re
import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning

REPO = Path(__file__).resolve().parents[1]


def test_package_imports():
    import dfthc

    assert dfthc.__version__ == "0.1.0"


def test_warning_variable_exported_before_python_started():
    assert os.environ.get("CATALYST_SILENCE_RULE_LOWERING_WARNINGS") == "0", (
        "run the tests through scripts/test.sh, which exports the variable before Python starts"
    )


def test_rule_lowering_warnings_are_recorded():
    with warnings.catch_warnings(record=True) as records:
        warnings.warn("probe", RuleLoweringWarning, stacklevel=1)
    assert [str(r.message) for r in records] == ["probe"]


@pytest.mark.parametrize("name", ["pennylane", "pennylane_catalyst", "pennylane_lightning"])
def test_lock_file_pins_installed_versions(name):
    lock = (REPO / "requirements-nightly.lock").read_text()
    pinned = re.search(rf"^{name.replace('_', '[-_]')}==(\S+)$", lock, re.MULTILINE | re.IGNORECASE)
    assert pinned is not None, f"{name} missing from requirements-nightly.lock"
    assert pinned.group(1) == version(name)


@pytest.mark.parametrize("script", ["scripts/test.sh", "scripts/setup_env.sh"])
def test_scripts_are_executable(script):
    assert os.access(REPO / script, os.X_OK)


@pytest.mark.release
def test_release_marked_probe():
    """Collected only by the release tier; the next test checks that `-m "not slow"` drops it."""


@pytest.mark.subprocess
def test_release_marker_implies_slow():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:xdist", "-m", "not slow",
         f"{Path(__file__).name}::test_release_marked_probe"],
        cwd=Path(__file__).parent, capture_output=True, text=True, check=False,
    )
    assert "test_release_marked_probe" not in result.stdout
    assert "1 deselected" in result.stdout, result.stdout + result.stderr
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0 "$VENV/bin/python" -m pytest tests/test_infrastructure.py -q -p no:cacheprovider`
Expected: FAIL. `test_package_imports` fails with `ModuleNotFoundError: No module named 'dfthc'`; the lock-file tests fail with `FileNotFoundError`; the script tests and `test_release_marker_implies_slow` fail with `AssertionError`.

- [ ] **Step 3: Write the package, configuration and scripts**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "dfthc"
version = "0.1.0"
description = "DFTHC block encoding, qubitized walk and phase estimation in PennyLane (Low et al., PRX 15, 041016 (2025))"
requires-python = ">=3.12"
# The nightly toolchain is installed by scripts/setup_env.sh and pinned in requirements-nightly.lock.
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = [
    "compiled: compiles a program with qp.qjit(capture=True)",
    "slow: not in the per-commit tier; runs nightly and in the release tier",
    "release: release tier only; the conftest hook also marks it slow",
    "canary: pins today's behaviour of a known limitation (spec section 14)",
    "femoco: builds FeMoco-54-sized programs",
    "subprocess: runs its check in a separate Python process",
]

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests", "scripts"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.ruff.lint.per-file-ignores]
"tests/data/test_vendored.py" = ["E501"]  # one SHA-256 digest per line
```

Create `src/dfthc/__init__.py`:

```python
"""DFTHC block encoding, qubitized walk and iterative phase estimation in PennyLane.

The design is docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md.
"""

__version__ = "0.1.0"
```

Create `tests/conftest.py` (Task 0.2 replaces it with the full version):

```python
"""Shared pytest configuration (spec sections 7.1 and 12.6)."""

import warnings

import catalyst  # noqa: F401  # importing Catalyst reads CATALYST_SILENCE_RULE_LOWERING_WARNINGS
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning

# Second line of defence (spec 7.1): if the variable was not exported before Python started,
# Catalyst installed an "ignore" filter for RuleLoweringWarning; this filter takes precedence.
warnings.filterwarnings("always", category=RuleLoweringWarning)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """`release` implies `slow`; runs before `-m` deselection (spec section 12.6)."""
    for item in items:
        if item.get_closest_marker("release") and not item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)
```

Create `scripts/test.sh`:

```bash
#!/usr/bin/env bash
# Run pytest with the canonical environment (spec sections 7.1 and 12.6).
#
# Catalyst reads CATALYST_SILENCE_RULE_LOWERING_WARNINGS once, when it is first imported, so the
# variable must be exported before Python starts. Works from the main checkout and from every
# git worktree: the shared .venv lives next to the main checkout's .git directory.
set -euo pipefail
export CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0

worktree_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
common_git_dir="$(git -C "$worktree_root" rev-parse --path-format=absolute --git-common-dir)"
venv="${DFTHC_VENV:-$(dirname "$common_git_dir")/.venv}"

if [[ ! -x "$venv/bin/python" ]]; then
    echo "test.sh: no Python at $venv/bin/python; run scripts/setup_env.sh in the main checkout" >&2
    exit 2
fi

cd "$worktree_root"
exec "$venv/bin/python" -m pytest -n "${DFTHC_TEST_WORKERS:-auto}" "$@"
```

Create `scripts/setup_env.sh`:

```bash
#!/usr/bin/env bash
# Create the canonical environment of spec section 7.1 in .venv at the repository root.
#
# Expected noise, both benign:
#   * the second install prints "ERROR: ... pennylane-catalyst ... requires pennylane==<dev N>,
#     but you have pennylane <dev M>"; overriding the pin is intended;
#   * every compile later prints "ld: warning: object file ... was built for newer 'macOS' version".
# Never use `uv pip`: it resolves pennylane-catalyst from PyPI and installs the stable release.
set -euo pipefail
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

if [[ -e .venv ]]; then
    echo "setup_env.sh: .venv exists; remove it first (rm -rf .venv) to recreate it" >&2
    exit 2
fi

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Catalyst nightly first (the user's `catame` alias), then the PennyLane nightly on top:
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane-catalyst --upgrade
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane --upgrade
pip install pytest pytest-xdist ruff          # from PyPI, without the TestPyPI index
pip freeze > requirements-nightly.lock
```

Then make both executable: `chmod +x scripts/test.sh scripts/setup_env.sh`.

- [ ] **Step 4: Complete the environment and write the lock**

If the main checkout has no `.venv`, run `scripts/setup_env.sh` there (it refuses to overwrite an existing `.venv`). The orchestrator's `.venv` already holds the nightlies of the spec snapshot; complete it with the recipe's last two commands, writing the lock into this worktree:

```bash
"$VENV/bin/pip" install pytest pytest-xdist ruff
"$VENV/bin/pip" freeze > requirements-nightly.lock
grep -E '^(pennylane|pennylane_catalyst|pennylane_lightning|pytest|pytest-xdist|ruff)==' requirements-nightly.lock
```

Expected output (versions as of 2026-11-11; later nightlies differ, which is fine as long as the lock matches the installed set):

```text
pennylane==0.46.0.dev151
pennylane_catalyst==0.16.0.dev140
pennylane_lightning==0.46.0.dev34
pytest==9.2.0
pytest-xdist==3.8.0
ruff==0.17.4
```

- [ ] **Step 5: Write the environment documentation**

Create `docs/environment.md`:

````markdown
# Environment

The canonical environment is `.venv` in the main checkout, CPython 3.12 (spec section 7.1).
Every git worktree uses it through `scripts/test.sh`; nothing is installed in editable mode.

## Snapshot

| Package | Version |
| --- | --- |
| pennylane | 0.46.0.dev151 |
| pennylane_catalyst | 0.16.0.dev140 |
| pennylane_lightning | 0.46.0.dev34 |
| jax, jaxlib | 0.7.1 |
| pytest / pytest-xdist / ruff | 9.2.0 / 3.8.0 / 0.17.4 |

`requirements-nightly.lock` is the complete `pip freeze`; `tests/test_infrastructure.py` checks
that it pins the installed PennyLane, Catalyst and Lightning versions.

## Recipe

`scripts/setup_env.sh` runs exactly the commands of spec section 7.1 in a fresh `.venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane-catalyst --upgrade
pip install --extra-index-url=https://test.pypi.org/simple/ --pre pennylane --upgrade
pip install pytest pytest-xdist ruff
pip freeze > requirements-nightly.lock
```

## Pitfalls (observed on 2026-11-11)

- Always use plain `pip`, never `uv pip`: uv resolves `pennylane-catalyst` from PyPI and installs
  the stable release.
- The second install prints `ERROR: ... pennylane-catalyst 0.16.0.dev140 requires
  pennylane==0.46.0-dev149, but you have pennylane 0.46.0.dev151`. The override is intended.
- Every compile prints `ld: warning: object file ... was built for newer 'macOS' version (28.0)
  than being linked (27.0)`. It is benign.
- `CATALYST_SILENCE_RULE_LOWERING_WARNINGS=0` must be exported before Python starts; Catalyst
  reads it once at import. `scripts/test.sh` exports it, and `tests/conftest.py` re-enables the
  warnings as a second line of defence.
- `--pre` with the TestPyPI index also pulls dev builds of dependencies (`termcolor`, `urllib3`);
  the lock file records them.
- A missing decomposition path surfaces as `failed to load built-in decomposition rules ...: the
  rules file could not be parsed`; the real cause, `Decomposition rule not found for operator
  '<op>'`, is in the same error message.

## Bumping the nightlies

1. Recreate `.venv` with the recipe (`rm -rf .venv && scripts/setup_env.sh`).
2. Run the full suite, including the canaries: `scripts/test.sh`.
3. Update spec section 14 and `docs/limitations.md` for every canary whose state changed.
4. Refresh the lock file, this snapshot and `tests/canary/compile_baseline.json`
   (`DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 scripts/test.sh tests/canary/test_l21_compile_time.py`).
5. Commit on a branch.
````

- [ ] **Step 6: Run the tests and the linter**

Run: `scripts/test.sh tests/test_infrastructure.py -q`
Expected: `10 passed`.

Run: `scripts/test.sh -m "not slow and not release" tests/test_infrastructure.py -q`
Expected: `9 passed`. The `release`-only marker test is deselected; with `-n auto`, pytest-xdist leaves the deselected count out of the summary (`DFTHC_TEST_WORKERS=0` shows `9 passed, 1 deselected`).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements-nightly.lock scripts/setup_env.sh scripts/test.sh \
        src/dfthc/__init__.py tests/conftest.py tests/test_infrastructure.py docs/environment.md
git commit -m "Add package scaffold, environment lock and test runner"
```

---

### Task 0.2: `dfthc.compile`, resource assertions and the canary runner

**Task id:** `0.2` · **Branch:** `phase0/0.2` · **Tier:** A (`claude-opus-5-5-high-fast`) · **Reviewers:** GPT-5.6 and Opus · **Depends on:** 0.1

**Files:**
- Create: `src/dfthc/compile.py`, `src/dfthc/testing/__init__.py`, `src/dfthc/testing/resources.py`, `src/dfthc/testing/canary.py`, `tests/test_compile.py`, `tests/test_primitive_costs.py`, `tests/test_canary_helper.py`
- Modify: `tests/conftest.py` (replace with the version below)

**Interfaces:**
- Consumes: Task 0.1's `scripts/test.sh`, markers and `tests/conftest.py`.
- Produces (every later task relies on these exact names):
  - `dfthc.compile.GATE_SET: dict[str, int]` (spec 7.3) and `ANALYZE_NAMES: frozenset[str]`.
  - `dfthc.compile.FIXED_DECOMPS: dict[str, Callable]` and `register_rule(op: type | str, rule) -> None`: adds the only rule for a dfthc class or a symbolic key (`"Adjoint(X)"`, `"C(X)"`) and pins it; raises `ValueError` on a second registration.
  - `dfthc.compile.resource_key(build: Callable[[], Operator]) -> Operator`: builds a resource key with capture paused.
  - `dfthc.compile.CompiledResources(toffoli: int, measurements: int, measured_and_uncomputes: int, counts: dict[str, int], num_wires: int)` with property `physical_measurements -> int`.
  - `dfthc.compile.compiled_resources(qfunc, num_wires, *, device="null.qubit", level="user", fixed_decomps=None) -> CompiledResources`: raises `AssertionError` on symbolic counts, on operations outside the gate set (`level="user"`), or on a surviving dfthc operator, bare or under `C(...)` or `Adjoint(...)` (other levels).
  - `dfthc.compile.dfthc_operator_names(keys=None) -> frozenset[str]`: the operator names inside rule keys (default `FIXED_DECOMPS`); `surviving_dfthc_operators(names, keys=None) -> list[str]`: the gate names that contain one of them, bare or wrapped.
  - `dfthc.compile.mcm_parities(outcomes, masks: Sequence[int]) -> list`: one entry per mask, `False` for a zero mask, else a predicate for `qp.cond`; also `pack_masks(masks, num_outcomes) -> np.ndarray` and `packed_parities(outcome_bits, packed_masks)`.
  - `dfthc.testing.resources`: `NOISE_PATTERNS`, `UNRESOLVED_BRANCH_MESSAGE`, `dfthc_operator_names` (imported from `dfthc.compile`), `classify_rule_warning(message, names=None) -> "dfthc" | "noise" | "other"`, `install_warning_filters()`, `verbose_program(qfunc, num_wires)`, `chosen_rules(listing) -> list[str]`, `assert_resources(resources, *, toffoli, measurements, measured_and_uncomputes=None)`.
  - `dfthc.testing.canary`: `PRELUDE`, `run_snippet(body, *, export_warning_variable=True, timeout=900.0) -> subprocess.CompletedProcess[str]`, `outcome(result, label) -> str`.

**Acceptance tests:**
- `tests/test_compile.py` (21 tests): the gate set equals spec 7.3; `register_rule` pins once and rejects duplicates; `qp.list_decomps(key) == [rule]` for every pinned key (PennyLane registers none of our names); **T6 pattern**: a pinned measured `Adjoint(...)` rule is chosen, with counts exactly `{"CNOT": 1, "Hadamard": 2, "MidCircuitMeasure": 1, "PauliX": 1, "PauliZ": 1}`; the verbose rule listing names both pinned rules (Phase 0 check of spec 12.1 T6: capturable only with `qp.qjit(verbose=True)`); `SemiAdder` on 3 bits gives T = 2 and A = 2; symbolic counts and operations outside the gate set are rejected; at `level="device"` a dfthc operator is rejected bare and under `C(...)` and `Adjoint(...)` (`surviving_dfthc_operators`, unit-tested on all spellings and on the device branch with `qp.analyze` stubbed), while a decomposed program passes (the old base-name check missed the wrapped spellings in the dry-run); `resource_key` pauses capture; `pack_masks` layout; out-of-range masks are rejected; `mcm_parities` equals the GF(2) reference on all 2^5 outcome vectors in tape mode, is linear on 100 random pairs of 70-bit vectors in the capture core, and drives conditional gates correctly in a compiled `lightning.qubit` circuit; `classify_rule_warning` separates dfthc, noise and other messages; `assert_resources` reports mismatches.
- `tests/test_primitive_costs.py` (53 tests), pinning spec 9.24's primitive table: `SemiAdder` and its adjoint T = A = n − 1 with no `Toffoli` (n = 1, 2, 3, 4, 6, 8, 15); `LeftQuantumComparator` T = n, A = 0 and its adjoint T = 0, A = n (n = 2, 3, 4, 6, 8; `"<="` and `"<"`); `LeftClassicalComparator` T = n − 1 and its adjoint A = n − 1 (n, value = (3, 1), (3, 6), (5, 7), (5, 27), (7, 81), (7, 3)); `MultiControlledX` with k controls and k − 2 zeroed work wires T = k − 1 with exactly one `Toffoli` and A = k − 2 (k = 2, 3, 4, 5, 8, 12); `CSWAP` T = 1.
- `tests/test_canary_helper.py` (3 tests): report lines for success and failure, the warning variable exported or unset, a clear error for a missing report line.

**Definition of done:** `scripts/test.sh tests/test_compile.py tests/test_primitive_costs.py tests/test_canary_helper.py -q` gives `77 passed`; the per-commit tier gives `86 passed` (one test deselected); `ruff check .` clean; output free of warnings.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compile.py`:

```python
"""Tests for dfthc.compile and dfthc.testing.resources (plan Task 0.2)."""

import itertools
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pennylane as qp
import pytest
from pennylane.core import Operator2
from pennylane.typing import Wire

import dfthc.compile as dc
from dfthc.compile import (
    FIXED_DECOMPS,
    GATE_SET,
    CompiledResources,
    compiled_resources,
    mcm_parities,
    pack_masks,
    packed_parities,
    register_rule,
    resource_key,
    surviving_dfthc_operators,
)
from dfthc.testing.resources import (
    assert_resources,
    chosen_rules,
    classify_rule_warning,
    verbose_program,
)


class CompileTestLoad(Operator2):
    """Copies ``ctrl`` into ``target``; its pinned adjoint erases ``target`` by measurement."""

    wire_argnames = ("ctrl", "target")

    def __init__(self, ctrl, target):
        super().__init__(ctrl, target)


@qp.register_resources({qp.CNOT: 1})
def _compile_test_load(ctrl, target):
    qp.CNOT([ctrl[0], target[0]])


def _compile_test_unload_resources(base):
    return {qp.H: 1, resource_key(lambda: qp.ops.MidMeasure(Wire[1], reset=True)): 1, qp.Z: 1}


@qp.register_resources(_compile_test_unload_resources)
def _compile_test_unload(base):
    qp.H(base.target[0])
    outcome = qp.measure(base.target[0], reset=True)
    qp.cond(outcome, qp.Z)(base.ctrl[0])


register_rule(CompileTestLoad, _compile_test_load)
register_rule("Adjoint(CompileTestLoad)", _compile_test_unload)


def test_gate_set_matches_spec():
    assert GATE_SET == {
        "PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "Adjoint(S)": 1,
        "CNOT": 1, "CZ": 1, "SWAP": 1, "GlobalPhase": 1, "MidMeasure": 1,
        "Adjoint(TemporaryAND)": 1, "PhaseGradientStatePrep": 1,
        "Toffoli": 1000, "TemporaryAND": 1000,
    }


def test_register_rule_pins_once():
    assert FIXED_DECOMPS["CompileTestLoad"] is _compile_test_load
    assert FIXED_DECOMPS["Adjoint(CompileTestLoad)"] is _compile_test_unload
    with pytest.raises(ValueError, match="already has a pinned rule"):
        register_rule(CompileTestLoad, _compile_test_load)


def test_registered_names_have_only_our_rules():
    for key, rule in FIXED_DECOMPS.items():
        assert list(qp.list_decomps(key)) == [rule], key


@pytest.mark.compiled
def test_pinned_measured_adjoint_is_selected():
    """The T6 pattern: the count signature proves the measured erase rule was chosen."""

    def qfunc():
        qp.H(0)
        CompileTestLoad([0], [1])
        qp.adjoint(CompileTestLoad([0], [1]))

    resources = compiled_resources(qfunc, num_wires=2)
    assert resources.counts == {
        "CNOT": 1, "Hadamard": 2, "MidCircuitMeasure": 1, "PauliX": 1, "PauliZ": 1,
    }
    assert_resources(resources, toffoli=0, measurements=1, measured_and_uncomputes=0)
    assert resources.physical_measurements == 1


@pytest.mark.compiled
def test_verbose_rule_listing_names_the_pinned_rules(capfd):
    """Spec 12.1 T6, checked in Phase 0: the solver's rule listing is capturable."""

    def qfunc():
        qp.H(0)
        CompileTestLoad([0], [1])
        qp.adjoint(CompileTestLoad([0], [1]))

    qp.analyze(verbose_program(qfunc, num_wires=2), level="user")()
    captured = capfd.readouterr()
    rules = set(chosen_rules(captured.out + captured.err))
    assert {"_compile_test_load", "_compile_test_unload"} <= rules


@pytest.mark.compiled
def test_compiled_resources_counts_toffoli_classes():
    resources = compiled_resources(
        lambda: qp.SemiAdder(x_wires=[0, 1, 2], y_wires=[3, 4, 5], work_wires=[6, 7]), num_wires=8
    )
    assert isinstance(resources, CompiledResources)
    assert (resources.toffoli, resources.measured_and_uncomputes) == (2, 2)


@pytest.mark.compiled
def test_compiled_resources_rejects_symbolic_counts():
    def qfunc():
        qp.H(0)
        bound = qp.measure(0)

        @qp.for_loop(bound)
        def loop(_):
            qp.X(1)

        loop()

    with pytest.raises(AssertionError, match="symbolic"):
        compiled_resources(qfunc, num_wires=2)


@pytest.mark.compiled
def test_compiled_resources_rejects_operations_outside_gate_set(monkeypatch):
    monkeypatch.setattr(dc, "GATE_SET", GATE_SET | {"RX": 1})
    with pytest.raises(AssertionError, match="outside the gate set"):
        compiled_resources(lambda: qp.RX(0.3, 0), num_wires=1)


def test_surviving_dfthc_operators_sees_wrapped_names():
    keys = ["OuterPrepare", "Adjoint(OuterPrepare)", "C(RefT2)"]
    names = ["OuterPrepare", "C(OuterPrepare)", "Adjoint(OuterPrepare)", "Adjoint(C(RefT2))",
             "OuterPrepareX", "CNOT", "C(PauliX)", "Adjoint(TemporaryAND)"]
    assert surviving_dfthc_operators(names, keys) == [
        "Adjoint(C(RefT2))", "Adjoint(OuterPrepare)", "C(OuterPrepare)", "OuterPrepare",
    ]


@pytest.mark.parametrize(
    "name", ["CompileTestLoad", "C(CompileTestLoad)", "Adjoint(CompileTestLoad)"]
)
def test_device_level_rejects_surviving_dfthc_operator(monkeypatch, name):
    """T5 at level="device" (L-20): a dfthc operator fails the check bare or wrapped."""
    reported = SimpleNamespace(is_symbolic=False, counts={name: 1, "CNOT": 2}, num_wires=2)
    monkeypatch.setattr(
        qp, "analyze", lambda circuit, level: lambda: SimpleNamespace(resources=reported)
    )
    with pytest.raises(AssertionError, match="dfthc operators reached"):
        compiled_resources(lambda: None, num_wires=2, level="device")


@pytest.mark.compiled
def test_device_level_closure_of_decomposed_program():
    def qfunc():
        qp.H(0)
        CompileTestLoad([0], [1])
        qp.adjoint(CompileTestLoad([0], [1]))

    resources = compiled_resources(qfunc, num_wires=2, level="device")
    assert resources.counts == {
        "CNOT": 1, "Hadamard": 2, "MidCircuitMeasure": 1, "PauliX": 1, "PauliZ": 1,
    }


def test_resource_key_pauses_capture():
    qp.capture.enable()
    try:
        assert resource_key(qp.capture.enabled) is False
        key = resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3])))
    finally:
        qp.capture.disable()
    assert key.name == "Adjoint(TemporaryAND)"


def test_pack_masks_layout():
    assert pack_masks([1, 2**32 + 3], 40).tolist() == [[1, 0], [3, 1]]


def test_mask_beyond_outcomes_is_rejected():
    with pytest.raises(ValueError, match="beyond the 3 outcomes"):
        mcm_parities([0, 1, 0], [8])


def _reference(masks, outcome_bits):
    value = sum(int(bit) << i for i, bit in enumerate(outcome_bits))
    return [bin(mask & value).count("1") % 2 for mask in masks]


def test_mcm_parities_tape_mode_exhaustive():
    """All 2^5 outcome vectors: f equals the GF(2) reference, so f_0 = 0 and f is linear."""
    masks = [0b00000, 0b00001, 0b10110, 0b11111, 0b01000, 0b10001, 0b00110]
    with qp.queuing.AnnotatedQueue():
        outcomes = [qp.measure(wire) for wire in range(5)]
        parities = mcm_parities(outcomes, masks)
    assert parities[0] is False
    for bits in itertools.product([0, 1], repeat=5):
        record = {outcomes[i].measurements[0]: bits[i] for i in range(5)}
        got = [0 if p is False else int(p.concretize(record)) for p in parities]
        assert got == _reference(masks, bits), bits


def test_mcm_parities_capture_core_is_linear():
    rng = np.random.default_rng(1234)
    num_outcomes = 70
    masks = [int(m) for m in rng.integers(0, 2**62, size=12)] + [0, 2**69 + 1]
    packed = pack_masks(masks, num_outcomes)
    zero = packed_parities(jnp.zeros(num_outcomes, dtype=jnp.uint32), packed)
    assert np.all(np.asarray(zero) == 0)
    for _ in range(100):
        xi, eta = rng.integers(0, 2, size=(2, num_outcomes))
        f_xi = np.asarray(packed_parities(xi, packed))
        f_eta = np.asarray(packed_parities(eta, packed))
        f_sum = np.asarray(packed_parities(xi ^ eta, packed))
        assert np.array_equal(f_sum, f_xi ^ f_eta)
        assert f_xi.tolist() == _reference(masks, xi)


@pytest.mark.compiled
def test_mcm_parities_in_compiled_circuit():
    """Deterministic outcomes (1, 0, 1) drive conditional X gates through capture-mode parities."""
    dev = qp.device("lightning.qubit", wires=7)
    masks = [0b001, 0b011, 0b101, 0b000]

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET)
    @qp.qnode(dev)
    def circuit():
        qp.X(0)
        qp.X(2)
        outcomes = [qp.measure(wire, reset=True) for wire in range(3)]
        for target, parity in zip(range(3, 7), mcm_parities(outcomes, masks), strict=True):
            if parity is not False:
                qp.cond(parity, qp.X)(target)
        return qp.probs(wires=[3, 4, 5, 6])

    probs = np.asarray(circuit())
    expected = np.zeros(16)
    expected[0b1100] = 1.0  # parities (1, 1, 0, 0) on wires 3..6, big-endian
    assert np.allclose(probs, expected)


def test_classify_rule_warning():
    names = {"OuterPrepare"}
    noise = ("Excluded the _toffoli_elbow decomposition rule for Toffoli since "
             "--decompose-lowering cannot work with multiple registers yet")
    assert classify_rule_warning(noise, names) == "noise"
    assert classify_rule_warning(
        "Excluded the use_allocation(decompose_mcx_two_controls_elbows) decomposition rule for "
        "MultiControlledX since --decompose-lowering cannot work with multiple registers yet", names
    ) == "noise"
    assert classify_rule_warning(noise.replace("Toffoli", "OuterPrepare"), names) == "dfthc"
    skipped = "Skipped the _outer rule for Adjoint(OuterPrepare)"
    assert classify_rule_warning(skipped, names) == "dfthc"
    assert classify_rule_warning("Something new about CNOT", names) == "other"


def test_assert_resources_reports_all_mismatches():
    resources = CompiledResources(toffoli=3, measurements=1, measured_and_uncomputes=2,
                                  counts={}, num_wires=4)
    assert_resources(resources, toffoli=3, measurements=1, measured_and_uncomputes=2)
    with pytest.raises(AssertionError, match="'toffoli': 4"):
        assert_resources(resources, toffoli=4, measurements=0)
```

Create `tests/test_primitive_costs.py`:

```python
"""Pin the PennyLane primitive costs assumed by the closed forms of spec 9.24 (T4, Task 0.2)."""

import pennylane as qp
import pytest

from dfthc.compile import compiled_resources

pytestmark = pytest.mark.compiled


def _wires(start, count):
    return list(range(start, start + count))


def _maybe_adjoint(op_class, adjoint):
    return qp.adjoint(op_class) if adjoint else op_class


@pytest.mark.parametrize("n", [1, 2, 3, 4, 6, 8, 15])
@pytest.mark.parametrize("adjoint", [False, True])
def test_semiadder(n, adjoint):
    def qfunc():
        _maybe_adjoint(qp.SemiAdder, adjoint)(
            x_wires=_wires(0, n), y_wires=_wires(n, n), work_wires=_wires(2 * n, n - 1) or None
        )

    resources = compiled_resources(qfunc, num_wires=3 * n)
    assert resources.counts.get("Toffoli", 0) == 0
    assert (resources.toffoli, resources.measured_and_uncomputes) == (n - 1, n - 1)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("comparator", ["<=", "<"])
@pytest.mark.parametrize("adjoint", [False, True])
def test_left_quantum_comparator(n, comparator, adjoint):
    def qfunc():
        _maybe_adjoint(qp.LeftQuantumComparator, adjoint)(
            x_wires=_wires(0, n), y_wires=_wires(n, n), target_wire=[2 * n],
            work_wires=_wires(2 * n + 1, n - 1), comparator=comparator,
        )

    resources = compiled_resources(qfunc, num_wires=3 * n + 1)
    assert resources.counts.get("Toffoli", 0) == 0
    expected = (0, n) if adjoint else (n, 0)
    assert (resources.toffoli, resources.measured_and_uncomputes) == expected


@pytest.mark.parametrize(("n", "value"), [(3, 1), (3, 6), (5, 7), (5, 27), (7, 81), (7, 3)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_left_classical_comparator(n, value, adjoint):
    def qfunc():
        _maybe_adjoint(qp.LeftClassicalComparator, adjoint)(
            x_wires=_wires(0, n), value=value, target_wire=[n], work_wires=_wires(n + 1, n - 1),
            comparator="<",
        )

    resources = compiled_resources(qfunc, num_wires=2 * n)
    assert resources.counts.get("Toffoli", 0) == 0
    expected = (0, n - 1) if adjoint else (n - 1, 0)
    assert (resources.toffoli, resources.measured_and_uncomputes) == expected


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8, 12])
def test_multicontrolled_x_with_zeroed_work(k):
    def qfunc():
        qp.MultiControlledX(wires=_wires(0, k + 1), work_wires=_wires(k + 1, k - 2) or None,
                            work_wire_type="zeroed")

    resources = compiled_resources(qfunc, num_wires=2 * k + 1)
    assert resources.counts.get("Toffoli", 0) == 1
    assert (resources.toffoli, resources.measured_and_uncomputes) == (k - 1, k - 2)


def test_cswap():
    resources = compiled_resources(lambda: qp.CSWAP(wires=[0, 1, 2]), num_wires=3)
    assert resources.counts.get("Toffoli", 0) == 1
    assert resources.toffoli == 1
```

Create `tests/test_canary_helper.py`:

```python
"""Tests for dfthc.testing.canary, the subprocess runner of the canary suite (plan Task 0.2)."""

import pytest

from dfthc.testing.canary import outcome, run_snippet

pytestmark = pytest.mark.subprocess


def test_run_snippet_reports_ok_and_error_lines():
    result = run_snippet(
        """
        report("SUM", lambda: 1 + 1)
        report("FAIL", lambda: 1 / 0)
        report("GATES", lambda: sorted(GATE_SET)[:2])
        """
    )
    assert outcome(result, "SUM") == "OK 2"
    assert outcome(result, "FAIL") == "ERROR ZeroDivisionError division by zero"
    assert outcome(result, "GATES") == 'OK ["Adjoint(S)", "Adjoint(TemporaryAND)"]'


def test_run_snippet_exports_or_unsets_the_warning_variable():
    body = """
        import os
        report("VAR", lambda: os.environ.get("CATALYST_SILENCE_RULE_LOWERING_WARNINGS"))
        """
    assert outcome(run_snippet(body), "VAR") == 'OK "0"'
    assert outcome(run_snippet(body, export_warning_variable=False), "VAR") == "OK null"


def test_outcome_explains_a_missing_report_line():
    result = run_snippet('print("nothing to see")')
    with pytest.raises(AssertionError, match="no report line for 'MISSING'"):
        outcome(result, "MISSING")
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `scripts/test.sh tests/test_compile.py tests/test_primitive_costs.py tests/test_canary_helper.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'dfthc.compile'` (and `dfthc.testing`).

- [ ] **Step 3: Write `dfthc.compile`**

Create `src/dfthc/compile.py`:

```python
"""Gate set, pinned rules and compile helpers shared by every operator and test.

Spec sections 7.2 (pipeline), 7.3 (gate set), 7.4 (rules for operators), 7.5 (pins) and
7.6 (resource accounting).
"""

from __future__ import annotations

import functools
import operator
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qp

GATE_SET: dict[str, int] = {
    "PauliX": 1, "PauliY": 1, "PauliZ": 1, "Hadamard": 1, "S": 1, "Adjoint(S)": 1,
    "CNOT": 1, "CZ": 1, "SWAP": 1, "GlobalPhase": 1, "MidMeasure": 1,
    "Adjoint(TemporaryAND)": 1, "PhaseGradientStatePrep": 1,
    "Toffoli": 1000, "TemporaryAND": 1000,
}
"""Toffoli-level target gate set with solver weights (spec section 7.3)."""

ANALYZE_NAMES: frozenset[str] = frozenset(GATE_SET) - {"MidMeasure"} | {"MidCircuitMeasure"}
"""Names ``qp.analyze`` may report at ``level="user"`` (it names MidMeasure MidCircuitMeasure)."""

FIXED_DECOMPS: dict[str, Callable] = {}
"""Pinned rule for every rule key registered through :func:`register_rule` (spec section 7.5)."""

_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")
_WRAPPERS = frozenset({"Adjoint", "C"})


def register_rule(op: type | str, rule: Callable) -> None:
    """Register ``rule`` as the only rule for a dfthc operator key and pin it.

    ``op`` is a dfthc ``Operator2`` subclass or a symbolic key such as ``"Adjoint(OuterPrepare)"``
    or ``"C(RefT2)"``. Each key gets exactly one rule (spec section 7.4, rule 7). Never call this
    for a PennyLane operator: a pin also applies to nested instances of the same name (L-4).
    """
    key = op if isinstance(op, str) else op.__name__
    if key in FIXED_DECOMPS:
        raise ValueError(f"{key} already has a pinned rule: {FIXED_DECOMPS[key]}")
    qp.add_decomps(op, rule)
    FIXED_DECOMPS[key] = rule


def resource_key(build: Callable[[], qp.operation.Operator]):
    """Build a resource-function key with program capture paused (spec section 7.4, rule 2; L-11).

    Example: ``resource_key(lambda: qp.adjoint(qp.TemporaryAND(Wire[3])))``.
    """
    with qp.capture.pause():
        return build()


@dataclass(frozen=True)
class CompiledResources:
    """Counts of one compiled program (spec section 7.6)."""

    toffoli: int
    """T: number of Toffoli plus number of TemporaryAND."""
    measurements: int
    """M: number of MidCircuitMeasure, the explicit erase measurements."""
    measured_and_uncomputes: int
    """A: number of Adjoint(TemporaryAND), the measured AND uncomputes."""
    counts: dict[str, int]
    """All gate counts reported by qp.analyze."""
    num_wires: int
    """Wire count reported by qp.analyze."""

    @property
    def physical_measurements(self) -> int:
        """M + A: every explicit measurement and every measured AND uncompute."""
        return self.measurements + self.measured_and_uncomputes


def compiled_resources(
    qfunc: Callable[[], None],
    num_wires: int,
    *,
    device: str = "null.qubit",
    level: str = "user",
    fixed_decomps: dict[str, Callable] | None = None,
) -> CompiledResources:
    """Compile ``qfunc`` with the stack of spec section 7.2 and return exact counts.

    ``qfunc`` takes no arguments and queues operations. ``fixed_decomps`` defaults to
    :data:`FIXED_DECOMPS`. Raises ``AssertionError`` if the counts are symbolic or, at
    ``level="user"``, if an operation outside the gate set remains (T5). At
    ``level="device"`` it instead asserts that no dfthc operator survives, bare or under
    ``C(...)`` and ``Adjoint(...)`` (L-20; :func:`surviving_dfthc_operators`).
    """
    pins = FIXED_DECOMPS if fixed_decomps is None else fixed_decomps
    dev = qp.device(device, wires=num_wires)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set=GATE_SET, fixed_decomps=dict(pins) or None)
    @qp.qnode(dev)
    def circuit():
        qfunc()
        return qp.probs(wires=[0])

    resources = qp.analyze(circuit, level=level)().resources
    if resources.is_symbolic:
        raise AssertionError(f"qp.analyze returned symbolic counts: {resources.counts}")
    counts = {str(name): int(value) for name, value in resources.counts.items()}
    if level == "user":
        outside = sorted(set(counts) - ANALYZE_NAMES)
        if outside:
            raise AssertionError(f"operations outside the gate set remain: {outside}")
    else:
        surviving = surviving_dfthc_operators(counts, set(FIXED_DECOMPS) | set(pins))
        if surviving:
            raise AssertionError(f"dfthc operators reached level={level!r}: {surviving}")
    return CompiledResources(
        toffoli=counts.get("Toffoli", 0) + counts.get("TemporaryAND", 0),
        measurements=counts.get("MidCircuitMeasure", 0),
        measured_and_uncomputes=counts.get("Adjoint(TemporaryAND)", 0),
        counts=counts,
        num_wires=int(resources.num_wires),
    )


def dfthc_operator_names(keys: Iterable[str] | None = None) -> frozenset[str]:
    """Operator names inside rule keys, by default those of :data:`FIXED_DECOMPS`.

    ``"C(Adjoint(OuterPrepare))"`` gives ``OuterPrepare``.
    """
    keys = FIXED_DECOMPS if keys is None else keys
    return frozenset(
        token for key in keys for token in _IDENTIFIER.findall(key) if token not in _WRAPPERS
    )


def surviving_dfthc_operators(
    names: Iterable[str], keys: Iterable[str] | None = None
) -> list[str]:
    """The gate ``names`` that contain an operator name of the rule ``keys``, bare or wrapped.

    A name matches if any identifier in it is a dfthc operator name, so ``OuterPrepare``,
    ``C(OuterPrepare)``, ``Adjoint(OuterPrepare)`` and ``Adjoint(C(OuterPrepare))`` match, while
    ``C(PauliX)``, ``Adjoint(TemporaryAND)`` and ``OuterPrepareX`` do not.
    """
    operators = dfthc_operator_names(keys)
    return sorted(name for name in names if operators.intersection(_IDENTIFIER.findall(name)))


def mcm_parities(outcomes: Sequence, masks: Sequence[int]) -> list:
    """Parities ``f(a) = xi . g(a) mod 2`` of measurement outcomes, one per address.

    ``outcomes[i]`` is the result of ``qp.measure`` for outcome bit ``i``: a ``MeasurementValue``
    in tape mode, a traced scalar under capture. ``masks[a]`` is the integer whose bit ``i`` is
    ``g(a)_i``. Returns one entry per mask: ``False`` (a Python bool) when the mask is 0, so callers
    emit no conditional; otherwise a predicate usable in ``qp.cond``. Tape mode XORs
    ``MeasurementValue``s; capture mode computes a packed-bit matrix-vector product with
    ``jax.numpy`` (spec section 7.4, rule 9). The map is GF(2)-linear in the outcomes in both modes.
    """
    num_outcomes = len(outcomes)
    for mask in masks:
        if mask < 0 or mask >> num_outcomes:
            raise ValueError(f"mask {mask} has bits beyond the {num_outcomes} outcomes")
    if qp.capture.enabled():
        return _capture_parities(outcomes, masks)
    return [_tape_parity(outcomes, mask) for mask in masks]


def _tape_parity(outcomes: Sequence, mask: int):
    bits = [outcomes[i] for i in range(len(outcomes)) if (mask >> i) & 1]
    if not bits:
        return False
    return functools.reduce(operator.xor, bits)


def pack_masks(masks: Sequence[int], num_outcomes: int) -> np.ndarray:
    """Pack masks into a ``(len(masks), ceil(num_outcomes / 32))`` uint32 array, low word first."""
    num_words = max(1, -(-num_outcomes // 32))
    packed = np.zeros((len(masks), num_words), dtype=np.uint32)
    for row, mask in enumerate(masks):
        for word in range(num_words):
            packed[row, word] = (mask >> (32 * word)) & 0xFFFFFFFF
    return packed


def packed_parities(outcome_bits, packed_masks: np.ndarray):
    """Parities of a 0/1 outcome vector against packed masks, with jax.numpy (traceable)."""
    num_words = packed_masks.shape[1]
    bits = jnp.asarray(outcome_bits, dtype=jnp.uint32)
    bits = jnp.pad(bits, (0, 32 * num_words - bits.shape[0]))
    shifts = jnp.arange(32, dtype=jnp.uint32)
    words = jnp.sum(bits.reshape(num_words, 32) << shifts, axis=1, dtype=jnp.uint32)
    overlap = jnp.asarray(packed_masks) & words
    return jnp.sum(jax.lax.population_count(overlap), axis=1) & 1


def _capture_parities(outcomes: Sequence, masks: Sequence[int]) -> list:
    bits = jnp.stack([jnp.asarray(value, dtype=jnp.uint32) for value in outcomes])
    parities = packed_parities(bits, pack_masks(masks, len(outcomes)))
    return [parities[row] == 1 if mask else False for row, mask in enumerate(masks)]
```

- [ ] **Step 4: Write the test-support package**

Create `src/dfthc/testing/__init__.py`:

```python
"""Test support: resource assertions, warning policy, and later the sparse simulator."""
```

Create `src/dfthc/testing/resources.py`:

```python
"""Assertions on compiled resources and the rule-lowering warning policy.

Spec sections 7.1 (warning noise), 7.4 rule 12 (warning policy) and 7.6 (accounting).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable

import pennylane as qp
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.passes import graph_decomposition

from dfthc.compile import FIXED_DECOMPS, GATE_SET, CompiledResources, dfthc_operator_names

_EXCLUDED_RULE = (
    r"(?:_\w*_ppm|_toffoli_elbow|_cphase_to_single_phase|\w*_workers?\w*|use_allocation\(\w+\))"
)
NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"^Excluded the {_EXCLUDED_RULE} decomposition rule for \S+ since "
        r"--decompose-lowering cannot work with multiple registers yet$"
    ),
    re.compile(r"^Skipped the _adjoint_temporary_and\w* rule for C\(Adjoint\(TemporaryAND\)\)"),
)
"""RuleLoweringWarning texts that PennyLane's own rules print on every compile (spec 7.1)."""

UNRESOLVED_BRANCH_MESSAGE = (
    r"Resource analysis was unable to determine the branch of a conditional or switch statement"
)
"""qp.analyze UserWarning for an unhinted conditional. PennyLane's SemiAdder rule triggers it on
every compile, so it is filtered; exact T4 counts and the erase-hint tests guard our own code."""


def classify_rule_warning(message: str, names: Iterable[str] | None = None) -> str:
    """Return ``"dfthc"``, ``"noise"`` or ``"other"`` for a RuleLoweringWarning message.

    A message naming a dfthc operator is ``"dfthc"`` even if it also matches a noise pattern.
    """
    names = dfthc_operator_names() if names is None else set(names)
    if any(re.search(rf"\b{re.escape(name)}\b", message) for name in names):
        return "dfthc"
    if any(pattern.search(message) for pattern in NOISE_PATTERNS):
        return "noise"
    return "other"


def install_warning_filters() -> None:
    """Show every RuleLoweringWarning except the known PennyLane noise."""
    warnings.filterwarnings("always", category=RuleLoweringWarning)
    for pattern in NOISE_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern.pattern.lstrip("^"),
                                category=RuleLoweringWarning)
    warnings.filterwarnings("ignore", message=UNRESOLVED_BRANCH_MESSAGE, category=UserWarning)


def verbose_program(qfunc: Callable[[], None], num_wires: int):
    """The compile stack of spec 7.2 with the solver's rule listing forwarded to stderr (T6).

    Only ``qjit(verbose=True)`` forwards the compiler output. Run the returned program (or
    ``qp.analyze`` it) under pytest's ``capfd`` and pass the captured text to :func:`chosen_rules`.
    """

    @qp.qjit(capture=True, verbose=True)
    @graph_decomposition(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS) or None,
                         verbose=True)
    @qp.qnode(qp.device("null.qubit", wires=num_wires))
    def program():
        qfunc()
        return qp.probs(wires=[0])

    return program


def chosen_rules(listing: str) -> list[str]:
    """Rule names from the lines ``Chosen Rule: __builtin_<rule>`` of a verbose listing."""
    prefix = "Chosen Rule: __builtin_"
    return [line.split(prefix, 1)[1].strip() for line in listing.splitlines() if prefix in line]


def assert_resources(
    resources: CompiledResources, *, toffoli: int, measurements: int,
    measured_and_uncomputes: int | None = None,
) -> None:
    """Assert exact T, M and optionally A, with every mismatch in one message."""
    expected = {"toffoli": toffoli, "measurements": measurements}
    if measured_and_uncomputes is not None:
        expected["measured_and_uncomputes"] = measured_and_uncomputes
    actual = {name: getattr(resources, name) for name in expected}
    if actual != expected:
        raise AssertionError(
            f"resources differ: expected {expected}, got {actual}; counts {resources.counts}"
        )
```

Create `src/dfthc/testing/canary.py`:

```python
"""Run canary snippets in a fresh Python process (spec section 14 and T7).

Snippets that may abort the process, that hit "no decomposition path", or that depend on
import-time state run through :func:`run_snippet`. Each snippet reports outcomes with the
prelude's ``report(label, fn)``, which prints one line ``<label> OK <json>`` or
``<label> ERROR <ExceptionType> <message lines joined by " | ">``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]

PRELUDE = textwrap.dedent(
    """
    import json
    import warnings

    import numpy as np
    import pennylane as qp
    from pennylane.core import Operator2
    from pennylane.typing import Wire

    from dfthc.compile import GATE_SET


    def counts(circuit, level="user"):
        resources = qp.analyze(circuit, level=level)().resources
        return {str(k): str(v) for k, v in resources.counts.items()}


    def report(label, fn):
        try:
            print(label, "OK", json.dumps(fn(), sort_keys=True, default=str), flush=True)
        except Exception as exc:  # noqa: BLE001
            message = " | ".join(line for line in str(exc).splitlines() if line.strip())
            print(label, "ERROR", type(exc).__name__, message, flush=True)
    """
)


def run_snippet(
    body: str, *, export_warning_variable: bool = True, timeout: float = 900.0
) -> subprocess.CompletedProcess[str]:
    """Run ``PRELUDE + body`` in a new interpreter and return the completed process.

    ``export_warning_variable=False`` unsets CATALYST_SILENCE_RULE_LOWERING_WARNINGS (L-18).
    """
    env = dict(os.environ, PYTHONPATH=str(SRC))
    if export_warning_variable:
        env["CATALYST_SILENCE_RULE_LOWERING_WARNINGS"] = "0"
    else:
        env.pop("CATALYST_SILENCE_RULE_LOWERING_WARNINGS", None)
    code = PRELUDE + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout,
        check=False,
    )


def outcome(result: subprocess.CompletedProcess[str], label: str) -> str:
    """The text after ``<label> `` on the snippet's report line, e.g. ``OK {...}``."""
    for line in result.stdout.splitlines():
        if line.startswith(label + " "):
            return line[len(label) + 1 :]
    raise AssertionError(
        f"no report line for {label!r}\nstdout:\n{result.stdout}\nstderr tail:\n"
        + "\n".join(result.stderr.splitlines()[-20:])
    )
```

- [ ] **Step 5: Replace `tests/conftest.py` with the full version**

```python
"""Shared pytest configuration (spec sections 7.1, 7.4 rule 12 and 12.6)."""

import warnings

import catalyst  # noqa: F401  # importing Catalyst reads CATALYST_SILENCE_RULE_LOWERING_WARNINGS
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning

from dfthc.testing.resources import classify_rule_warning, install_warning_filters

# Second line of defence (spec 7.1): if the variable was not exported before Python started,
# Catalyst installed an "ignore" filter for RuleLoweringWarning; these filters take precedence.
install_warning_filters()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """`release` implies `slow`; runs before `-m` deselection (spec section 12.6)."""
    for item in items:
        if item.get_closest_marker("release") and not item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _rule_lowering_warning_policy():
    """Fail on any RuleLoweringWarning that names a dfthc operator (spec 7.4, rule 12)."""
    with warnings.catch_warnings(record=True) as records:
        install_warning_filters()
        yield
    offending = []
    for record in records:
        if issubclass(record.category, RuleLoweringWarning):
            kind = classify_rule_warning(str(record.message))
            if kind == "dfthc":
                offending.append(str(record.message))
                continue
            if kind == "noise":
                continue
        warnings.warn_explicit(record.message, record.category, record.filename, record.lineno)
    if offending:
        pytest.fail("RuleLoweringWarning names a dfthc operator:\n" + "\n".join(offending))
```

- [ ] **Step 6: Run the tests**

Run: `scripts/test.sh tests/test_compile.py tests/test_primitive_costs.py tests/test_canary_helper.py -q`
Expected: `77 passed` (about 30 s with `-n auto`), no warnings summary.

Run: `scripts/test.sh -m "not slow and not release" -q`
Expected: `86 passed`. The `release`-only marker test is deselected; with `-n auto`, pytest-xdist leaves the deselected count out of the summary (`DFTHC_TEST_WORKERS=0` shows `86 passed, 1 deselected`).

Run: `"$VENV/bin/ruff" check .`
Expected: `All checks passed!`

- [ ] **Step 7: Record the T6 finding**

Append to this task's report: "Verbose rule listing (spec 12.1 T6) is capturable only when `qp.qjit(verbose=True)` forwards the compiler output; `dfthc.testing.resources.verbose_program` and `chosen_rules` implement it. Listing lines read `Chosen Rule: __builtin_<rule name>`." The orchestrator copies it into the ledger's spike section.

- [ ] **Step 8: Commit**

```bash
git add src/dfthc/compile.py src/dfthc/testing tests/conftest.py tests/test_compile.py \
        tests/test_primitive_costs.py tests/test_canary_helper.py
git commit -m "Add compile helpers, resource assertions and canary runner"
```

---

### Task 0.5: Vendored dfthcbe data and license

**Task id:** `0.5` · **Branch:** `phase0/0.5` · **Tier:** B (`claude-sonnet-5-5-high`) · **Reviewers:** GPT-5.6 · **Depends on:** 0.1

**Files:**
- Create: `data/dfthc/` (18 data files from dfthcbe), `data/dfthc/LICENSE`, `data/dfthc/NOTICE`, `tests/data/test_vendored.py`

**Interfaces:**
- Consumes: Task 0.1's runner.
- Produces: `data/dfthc/<system>/{factors.npz, integrals.npz, summary.json, best.json}` exactly as in spec 8.1, unmodified, for Task 1A.2's loaders.

**Acceptance tests** (`tests/data/test_vendored.py`, 23 tests): the file set is exactly the vendored copy plus `NOTICE`; each file matches its SHA-256 digest at commit `930255020cec30172e0a7d17c0071d9476a687d1`; `LICENSE` is Apache-2.0; `NOTICE` names the source URL, the commit and "unmodified"; `h2_sto3g_R2_B2_C1` loads with `U`, `W`, `w_B` and the summary keys `lambda_sos`, `e_sos`, `e_upper`, `e_gap`.

**Definition of done:** `23 passed`; about 350 KB added; one commit.

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_vendored.py`:

```python
"""The vendored dfthcbe data are complete, unmodified and licensed (spec section 8.1, Task 0.5)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "data" / "dfthc"
COMMIT = "930255020cec30172e0a7d17c0071d9476a687d1"

SHA256 = {
    "fe2s2/best.json": "baff823d9f67ba9f75149737dc0b6855b1b553167b2b8945c13113497c30c509",
    "fe2s2/factors.npz": "eb043ece37243e56a14ca7e2d2b4239c56af027cfdc7055bbac70209bc8c2f55",
    "fe2s2/summary.json": "376eb32239ad3205899381f2b0dcc5c3206aeeb3877edc27bac5ad4bf94a7161",
    "fe4s4/best.json": "059d7e932da55a562324df0e8c7caeab986130ad5603dd80e115da39259fbd43",
    "fe4s4/factors.npz": "a543968b05f8c4fe88e2e2535f5cc722ba6390a395d8c868f6442d6845764fc8",
    "fe4s4/summary.json": "605d93634b9bb25245e19909c33255c89dc35839f9447a290a8df540349de74b",
    "h2_631g_R2_B2_C2/factors.npz": "dcfe9adb83e380457d9a3336828c1a6f8f09a063164dd376ac26a1acab428f7f",
    "h2_631g_R2_B2_C2/integrals.npz": "91771a0379325ed329b50b68cd2ef2a79b67bd5cdb2634712354ceb94f7e8395",
    "h2_631g_R2_B2_C2/summary.json": "84f26717f7534cc81f5f9e9f47011cf9dce657ef956f0488c672189399e0e3f7",
    "h2_631g_R2_B3_C3/factors.npz": "3a80f93f00ca0e31209c02b0e3602f1185e9319fad5af1927f065e7eed414911",
    "h2_631g_R2_B3_C3/integrals.npz": "91771a0379325ed329b50b68cd2ef2a79b67bd5cdb2634712354ceb94f7e8395",
    "h2_631g_R2_B3_C3/summary.json": "6b3c775ef1b279146840cd70ad6e29305b1b9f7626d558ba1c2f1139de2d59b0",
    "h2_sto3g_R2_B2_C1/factors.npz": "7beefa76b000e0ed35b069578f6b11b75ec4511f411ae4f3205790e05849323b",
    "h2_sto3g_R2_B2_C1/integrals.npz": "fa08d4292c852bc096ef4763a3b095f4a3ce7df45dc85c2ce58ea134d0ae24cf",
    "h2_sto3g_R2_B2_C1/summary.json": "ff76a8b435a122e538a9fb10f7d24c06bec6db91f091abd8c34e1580329a8fd6",
    "naphthalene/factors.npz": "cf0ac51dfacbfb69f194b37d07107d302d6e4b5464913b2b98a1f7358eba94f7",
    "naphthalene/integrals.npz": "7a403373a46c2ed78c9fba15a5ab9ba913ed670f51c3dfabccdb3cf16c81cd6f",
    "naphthalene/summary.json": "33d4a4b762039a9d711c44fc051ea37826a5d5950af41807664a621067d0c62b",
    "LICENSE": "f5ba331216ebf35ee7079088051f95c9b8a234ffeb08f9fc0c70825a12f16a71",
}


def test_file_set_is_exactly_the_vendored_copy():
    present = {str(p.relative_to(DATA)) for p in DATA.rglob("*") if p.is_file()}
    assert present == set(SHA256) | {"NOTICE"}


@pytest.mark.parametrize("name", sorted(SHA256))
def test_file_is_unmodified(name):
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == SHA256[name]


def test_license_is_apache_2():
    text = (DATA / "LICENSE").read_text()
    assert "Apache License" in text and "Version 2.0, January 2004" in text


def test_notice_names_source_and_commit():
    notice = (DATA / "NOTICE").read_text()
    assert "https://github.com/Covestro-HPQC/dfthcbe" in notice
    assert COMMIT in notice
    assert "unmodified" in notice


def test_h2_sto3g_loads():
    factors = np.load(DATA / "h2_sto3g_R2_B2_C1" / "factors.npz")
    summary = json.loads((DATA / "h2_sto3g_R2_B2_C1" / "summary.json").read_text())
    assert {"U", "W", "w_B"} <= set(factors.files)
    assert factors["U"].shape[:2] == (2, 2)
    assert {"lambda_sos", "e_sos", "e_upper", "e_gap"} <= set(summary)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `scripts/test.sh tests/data/test_vendored.py -q`
Expected: FAIL (`FileNotFoundError` for the data files; the file-set assertion fails).

- [ ] **Step 3: Copy the data and the license at the pinned commit**

```bash
REF=/tmp/dfthcbe-ref
[ -d "$REF" ] || git clone https://github.com/Covestro-HPQC/dfthcbe "$REF"
git -C "$REF" checkout --detach 930255020cec30172e0a7d17c0071d9476a687d1
mkdir -p data/dfthc
cp -R "$REF"/data/dfthc/* data/dfthc/
cp "$REF"/LICENSE data/dfthc/LICENSE
```

Create `data/dfthc/NOTICE`:

```text
The files in this directory, except this NOTICE, are an unmodified copy of the directory
data/dfthc/ of the repository https://github.com/Covestro-HPQC/dfthcbe at commit
930255020cec30172e0a7d17c0071d9476a687d1, together with that repository's LICENSE file.

They are distributed under the Apache License, Version 2.0; see LICENSE in this directory.
Copyright notices, if any, are those of the original authors. No file has been modified.
tests/data/test_vendored.py checks every file against its SHA-256 digest at that commit.
```

- [ ] **Step 4: Run the test**

Run: `scripts/test.sh tests/data/test_vendored.py -q`
Expected: `23 passed`.

- [ ] **Step 5: Commit**

```bash
git add data/dfthc tests/data/test_vendored.py
git commit -m "Vendor dfthcbe data with license and notice"
```
