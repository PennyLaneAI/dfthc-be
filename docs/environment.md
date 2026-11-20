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

The development laptop has 10 cores and 16 GB, and is in use while tests run (spec section 15).

- `scripts/test.sh` runs 2 xdist workers (`DFTHC_TEST_WORKERS` overrides; `0` runs serially) at
  lowered priority: `nice -n 10`, or, in the agent sandbox, where setpriority() is forbidden, the
  utility QoS clamp of `taskpolicy`.
- A heavy run (above 4 GB, more than 2 busy cores, longer than 30 minutes, or any FeMoco-shaped
  compile) is started alone. The largest compile, the FeMoco-54 walk, takes about 50 s and
  1.6 GB.

## The reference environment `.venv-ref`

`.venv-ref` in the main checkout holds Qualtran 0.7.0 and dfthcbe at commit
`930255020cec30172e0a7d17c0071d9476a687d1` (spec 8.6, Q-2). It only generates the reference
fixtures under `tests/fixtures/dfthcbe/`: `scripts/make_dfthcbe_fixtures.py` and
`scripts/make_report_fixture.py`. `scripts/make_venv_ref.sh` builds it and runs the first; nothing in the
main environment imports Qualtran.

```bash
.venv-ref/bin/python scripts/make_report_fixture.py --out tests/fixtures/dfthcbe
```

## The resource report

`scripts/resource_report.py` writes `docs/resources.md` and `docs/resources.json` (spec 13.3). It
compiles every program of the walk at the H2 systems and `naphthalene`, each in its own process
with a timeout (`dfthc.testing.measure.run_measured`, which records its wall time and peak
memory), and reuses every compile cached under the current code hash:

```bash
PYTHONPATH=src .venv/bin/python scripts/resource_report.py
```

The large systems compile only with `--large fe2s2`, `--large fe4s4` or `--large femoco54`
(at FeMoco-54 only the walk and the controlled walk), with a 60-minute timeout per program.
They are heavy runs: start them alone.

## The scaling sweep

`scripts/scaling_sweep.py` writes `docs/scaling.json`, which section 9 of
`notebooks/paper_results.ipynb` plots. It compiles the walk on random FeMoco-shaped factors
(`B = C = n_plus = N // 2`), with N from 4 to 54 at R = 10 and with R from 2 to 40 at N = 16.
Each point runs in its own process under `run_measured`. The sweep takes about 5 to 10 minutes
and peaks near 1.2 GB. It is a heavy run: start it alone.

```bash
PYTHONPATH=src .venv/bin/python scripts/scaling_sweep.py
```

`scripts/qualtran_comparison.py` writes `docs/qualtran_comparison.json`, section 9.6 of the
notebook. It times dfthcbe in `.venv-ref` on the same walks: the call-graph count, the flatten to
leaf bloqs and, for H2, the tensor-contraction simulation. It merges results by system, and each
process is killed above 8 GB or after 30 minutes. A full run takes about an hour, because the
FeMoco-54 flatten alone takes 28 minutes. It is a heavy run: start it alone.

```bash
PYTHONPATH=src .venv/bin/python scripts/qualtran_comparison.py [--systems NAME ...] [--no-sim]
```

## Bumping the nightlies

1. Recreate `.venv` with the recipe (`rm -rf .venv && scripts/setup_env.sh`).
2. Run the per-commit tier, which holds most canaries: `scripts/test.sh -m "not slow and not release"`.
   The gate's nightly-tier run covers the slow canaries; the release tier is not run for a bump.
3. Update spec section 14 and `docs/limitations.md` for every canary whose state changed.
4. Refresh the lock file, this snapshot and `tests/canary/compile_baseline.json`
   (`DFTHC_UPDATE_COMPILE_BASELINE=1 DFTHC_TEST_WORKERS=0 scripts/test.sh tests/canary/test_l21_compile_time.py`).
5. Commit on a branch.
