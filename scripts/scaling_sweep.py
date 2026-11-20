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

"""Compile-cost sweep of the walk over synthetic FeMoco-shaped systems: ``docs/scaling.json``.

Usage, from the repository root::

    PYTHONPATH=src .venv/bin/python scripts/scaling_sweep.py [--out docs/scaling.json]

Each point is a walk on random factors with ``B = C = n_plus = N // 2`` (FeMoco-54's ratios) at
the report precisions and the cheapest block bits (``costs_reference.default_params``). Two
sweeps: ``N`` from 4 to 54 at ``R = 10`` (N = 54 is FeMoco-54's shape), and ``R`` from 2 to 40 at
``N = 16``. Every point compiles with ``compiled_resources`` in its own process under
:func:`~dfthc.testing.measure.run_measured`, which records the wall time and peak memory of the
whole process (imports included); the process itself reports the time of the import, of the
classical tables and of the compile (wall and CPU, the Catalyst compiler subprocess included),
and asserts that the compiled Toffolis equal the closed form. The whole sweep takes about
5 minutes and peaks near 1.3 GB at N = 54: a heavy run (docs/environment.md), start it alone.
``python scripts/scaling_sweep.py --point N R`` measures one point in this process and prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_SWEEP = (4, 8, 12, 16, 24, 32, 40, 48, 54)
R_SWEEP = (2, 5, 10, 20, 40)
FIXED_R, FIXED_N = 10, 16


def _cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def measure_point(N: int, R: int) -> dict:
    """Compile one walk in this process and return its counts and phase timings."""
    start = time.perf_counter()
    import warnings

    import pennylane as qp

    import dfthc
    from dfthc.compile import compiled_resources
    from dfthc.costs_reference import default_params
    from dfthc.data.factors import random_factors
    from dfthc.ops.walk import DFTHCWalk
    from dfthc.registers import WalkLayout
    from dfthc.testing.resources import UNRESOLVED_BRANCH_MESSAGE

    warnings.filterwarnings("ignore", message=UNRESOLVED_BRANCH_MESSAGE, category=UserWarning)
    imported = time.perf_counter()

    B = C = n_plus = N // 2
    params = default_params(N, R, B, C, n_plus)
    tables = dfthc.WalkTables.from_factors(random_factors(N, R, B, C, n_plus=n_plus, seed=0),
                                           params)
    layout = WalkLayout.from_params(params)
    built, cpu_before = time.perf_counter(), _cpu_seconds()

    def qfunc():
        qp.PhaseGradientStatePrep(layout.phase_grad)
        DFTHCWalk(params.as_compilable(), tables.as_compilable(), **layout.groups())

    r = compiled_resources(qfunc, layout.num_wires)
    compiled = time.perf_counter()
    closed_form = dfthc.costs.walk(params, False).toffoli
    assert r.toffoli == closed_form, (N, R, r.toffoli, closed_form)
    return dict(N=N, R=R, B=B, C=C, wires=r.num_wires, toffoli=r.toffoli,
                gates=sum(r.counts.values()), measurements=r.physical_measurements,
                import_s=round(imported - start, 2), tables_s=round(built - imported, 2),
                compile_s=round(compiled - built, 2),
                compile_cpu_s=round(_cpu_seconds() - cpu_before, 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--point", nargs=2, type=int, metavar=("N", "R"))
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "scaling.json")
    args = parser.parse_args()
    if args.point:
        print("RESULT", json.dumps(measure_point(*args.point)), flush=True)
        return 0

    sys.path.insert(0, str(ROOT / "src"))
    import pennylane as qp

    from dfthc.testing.measure import run_measured

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "CATALYST_SILENCE_RULE_LOWERING_WARNINGS": "0"}
    cases = ([("N", n, FIXED_R) for n in N_SWEEP]
             + [("R", FIXED_N, r) for r in R_SWEEP if r != FIXED_R])
    points = []
    for sweep, N, R in cases:
        run = run_measured([sys.executable, __file__, "--point", str(N), str(R)],
                           timeout_s=1800, env=env, cwd=ROOT)
        line = next((s for s in run.stdout.splitlines() if s.startswith("RESULT ")), None)
        if line is None:
            raise RuntimeError(f"N={N}, R={R} failed:\n" + run.stderr[-2000:])
        point = json.loads(line.removeprefix("RESULT "))
        point.update(sweep=sweep, wall_s=round(run.seconds, 1), peak_mb=round(run.peak_mb, 1))
        print(json.dumps(point), flush=True)
        points.append(point)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "pennylane": qp.__version__,
                   "pennylane_catalyst": __import__("catalyst").__version__}
    args.out.write_text(json.dumps({"environment": environment, "points": points}, indent=1)
                        + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
