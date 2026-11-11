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

## Load limits

The development laptop has 10 cores and 16 GB, and the user works on it while agents run
(master plan section 12.9).

- `scripts/test.sh` runs 2 xdist workers (`DFTHC_TEST_WORKERS` overrides; `0` runs serially) at
  lowered priority: `nice -n 10`, or, in the agent sandbox, where setpriority() is forbidden, the
  utility QoS clamp of `taskpolicy`.
- Each process tree has a sampled 8 GB memory cap: the watchdog kills it at the first sample above
  7.5 GB, so the overshoot past 7.5 GB is bounded by 50 ms of growth. A heavy run (above 4 GB, more than 2 busy cores,
  longer than 30 minutes, or any FeMoco-shaped compile) is started only by the orchestrator, alone,
  under `PYTHONPATH=src .venv/bin/python -m dfthc.testing.watchdog -- <command>`, and logged in the
  ledger with its peak memory and wall time.

## Bumping the nightlies

Bumps are orchestrator work, done at a phase gate.

1. Recreate `.venv` with the recipe (`rm -rf .venv && scripts/setup_env.sh`).
2. Run the per-commit tier, which holds most canaries: `scripts/test.sh -m "not slow and not release"`.
   The gate's nightly-tier run covers the slow canaries; the release tier is not run for a bump.
3. Update spec section 14 and `docs/limitations.md` for every canary whose state changed.
4. Refresh the lock file, this snapshot and `tests/canary/compile_baseline.json`
   (`DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 scripts/test.sh tests/canary/test_l21_compile_time.py`).
5. Commit on a branch.
