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

"""Compute cost of dfthcbe (Qualtran) on the report's walks: ``docs/qualtran_comparison.json``.

Usage, from the repository root, with ``.venv-ref`` built (``scripts/make_venv_ref.sh``)::

    PYTHONPATH=src .venv/bin/python scripts/qualtran_comparison.py [--systems NAME ...]
                                                                   [--no-sim] [--out PATH]

For each system it builds dfthcbe's ``DFTHCWalk.from_factors`` on the same factors and parameters
as the resource report (``costs_reference.default_params``; FeMoco-54 is ``FEMOCO54`` on
``random_factors(54, 10, 27, 27, n_plus=27, seed=0)``) and measures two Qualtran operations, each
in a fresh ``.venv-ref`` process:

- ``count``: ``get_cost_value(walk, QECGatesCost())``, the call-graph count (Toffolis =
  Toffoli + CSwap + And + T/4, as dfthcbe's tests count them). It evaluates each bloq's declared
  call graph and decomposes nothing.
- ``flatten``: ``walk.as_composite_bloq().flatten()``, the walk decomposed down to leaf bloqs: the
  gate-level circuit, the counterpart of our compile. dfthcbe's three measured erasures have no
  decomposition and stay as leaves.

With simulation on (the default), it also computes ``<0|BE|0>`` of H2/STO-3G at the test
precisions both ways: dfthcbe by Qualtran tensor contraction (``exact_erase=True``, as
``scripts/make_dfthcbe_fixtures.py`` does), and ours on the sparse simulator, one column per
basis state.

Every process records wall time and peak resident memory (``wait4``). A process above
``LIMIT_MB`` or ``TIMEOUT_S`` is killed and recorded as such. The ``.venv-ref`` processes import
no ``dfthc`` (its package imports JAX): this driver writes their inputs to a temporary directory.
Results are merged into ``--out`` by (system, mode), so systems can be measured in separate runs.
A full run takes about an hour, most of it the FeMoco-54 flatten reaching its timeout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF_PYTHON = ROOT / ".venv-ref" / "bin" / "python"
SYSTEMS = ("h2_sto3g_R2_B2_C1", "naphthalene", "fe2s2", "fe4s4", "femoco54")
KEYS = ("U", "W", "w_B", "w_plus", "u_plus", "w_minus", "u_minus")
LIMIT_MB, TIMEOUT_S = 8192, 1800

_CHILD = r'''
import json, resource, sys, time, warnings
from collections import Counter
warnings.filterwarnings("ignore")
inputs, system, mode = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.perf_counter()
import numpy as np
import sympy
from dfthcbe import DFTHCWalk
from qualtran.resource_counting import QECGatesCost, get_cost_value
t1 = time.perf_counter()
p = json.load(open(f"{inputs}/params.json"))[system]
with np.load(f"{inputs}/factors_{system}.npz") as d:
    f = {k: d[k] for k in d.files}
walk = DFTHCWalk.from_factors(**f, b_rot=p["b_rot"], b_k1=p["b_k1"], b_k2=p["b_k2"], k_1=p["k1"],
                              k_2=p["k2"], k_e1=p["ke1"], k_e2=p["ke2"], s=p["s_u"])
t2 = time.perf_counter()
out = dict(import_s=round(t1 - t0, 2), build_s=round(t2 - t1, 2))
if mode == "count":
    cost = get_cost_value(walk, QECGatesCost())
    out["toffoli"] = int(sympy.sympify(cost.toffoli + cost.cswap + cost.and_bloq + cost.t / 4))
    out["clifford"] = int(cost.clifford)
else:
    cbloq = walk.as_composite_bloq().flatten()
    names = Counter(type(inst.bloq).__name__ for inst in cbloq.bloq_instances)
    out["leaf_bloqs"] = sum(names.values())
    out["bookkeeping_bloqs"] = sum(names[n] for n in ("Split", "Join", "Partition", "Cast",
                                                       "Allocate", "Free"))
out["phase_s"] = round(time.perf_counter() - t2, 2)
print("RESULT", json.dumps(out), flush=True)
'''

_REF_SIM = r'''
import json, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
t = time.perf_counter()
from dfthcbe import DFTHCBlockEncoding
from dfthcbe.block_encoding import INNER_ANCILLAS, OUTER_ANCILLAS
from qualtran import BloqBuilder, QAny
from qualtran.bloqs.basic_gates import IntEffect, IntState
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState
t_imp = time.perf_counter()
with np.load(sys.argv[1]) as d:
    f = {k: np.array(d[k]) for k in d.files}
be = DFTHCBlockEncoding.from_factors(**f, b_rot=4, b_k1=2, b_k2=2, s=3, k_1=1, k_2=1, k_e1=1,
                                     k_e2=1, exact_erase=True)  # H2_STO3G_TEST
bb = BloqBuilder()
psi = {n: bb.add_register(n, QAny(be.select.num_orb)) for n in ("psi_up", "psi_dn")}
width = {n: be.signature.get_left(n).total_bits() for n in OUTER_ANCILLAS + INNER_ANCILLAS}
grad = PhaseGradientState(be.select.b_rot)
out = bb.add_d(be, **psi, phase_grad=bb.add(grad),
               **{n: bb.add(IntState(0, w)) for n, w in width.items()})
bb.add(grad.adjoint(), phase_grad=out.pop("phase_grad"))
for n, w in width.items():
    bb.add(IntEffect(0, w), val=out.pop(n))
block = bb.finalize(**out).tensor_contract()
print("RESULT", json.dumps({"import_s": round(t_imp - t, 2),
                            "phase_s": round(time.perf_counter() - t_imp, 2),
                            "shape": list(block.shape)}), flush=True)
'''

_OUR_SIM = r'''
import json, time
from dfthc.ops.walk import DFTHCBlockEncoding
from dfthc.testing import walk_sim
t = time.perf_counter()
case = walk_sim.case("sto3g")
ops = walk_sim.program(case, DFTHCBlockEncoding)
outcomes = walk_sim.schedule("S0", walk_sim.num_measurements(ops))
for v in range(16):
    walk_sim.project(case, walk_sim.simulate(case, ops,
                     walk_sim.input_state(case, walk_sim.basis(case, v)), outcomes))
print("RESULT", json.dumps({"phase_s": round(time.perf_counter() - t, 2), "shape": [16, 16]}),
      flush=True)
'''


def run_guarded(command: list[str], env=None) -> dict:
    """Run ``command``; kill it above ``LIMIT_MB`` resident or after ``TIMEOUT_S``."""
    start = time.monotonic()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True, env=env, cwd=ROOT)
    killed = None
    while not (reaped := os.wait4(proc.pid, os.WNOHANG))[0]:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True,
                             text=True).stdout.strip()
        if rss and int(rss) / 1024 > LIMIT_MB:
            killed = f"killed above {LIMIT_MB} MB"
        elif time.monotonic() - start > TIMEOUT_S:
            killed = f"killed at the {TIMEOUT_S} s timeout"
        if killed:
            os.killpg(proc.pid, signal.SIGKILL)
            reaped = os.wait4(proc.pid, 0)
            break
        time.sleep(0.5)
    _, _, usage = reaped
    unit = 1 if sys.platform == "darwin" else 1024
    record = {"wall_s": round(time.monotonic() - start, 1),
              "peak_mb": round(usage.ru_maxrss * unit / 2**20, 1)}
    line = next((s for s in proc.stdout.read().splitlines() if s.startswith("RESULT ")), None)
    if line:
        record.update(status="ok", **json.loads(line.removeprefix("RESULT ")))
    else:
        record.update(status=killed or "failed: " + proc.stderr.read()[-500:])
    return record


def write_inputs(directory: Path, systems) -> None:
    """The factors and parameters of each system, readable without dfthc."""
    import numpy as np

    from dfthc.costs_reference import default_params
    from dfthc.data.factors import load_factors, random_factors
    from dfthc.params import FEMOCO54

    params = {}
    for name in systems:
        if name == "femoco54":
            f, p = random_factors(54, 10, 27, 27, n_plus=27, seed=0), FEMOCO54
        else:
            f = load_factors(name)
            p = default_params(f.N, f.R, f.B, f.C, f.n_plus)
        np.savez(directory / f"factors_{name}.npz", **{k: getattr(f, k) for k in KEYS})
        params[name] = dataclasses.asdict(p)
    (directory / "params.json").write_text(json.dumps(params))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="*", default=list(SYSTEMS), choices=SYSTEMS)
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "qualtran_comparison.json")
    args = parser.parse_args()
    if not REF_PYTHON.exists():
        raise SystemExit(f"{REF_PYTHON} is missing: build it with scripts/make_venv_ref.sh")
    data = json.loads(args.out.read_text()) if args.out.exists() else {"records": []}
    records = {(r["system"], r["mode"]): r for r in data["records"]}

    def add(system, mode, record):
        record.update(system=system, mode=mode)
        records[(system, mode)] = record
        print(json.dumps(record), flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp)
        write_inputs(inputs, set(args.systems) | {"h2_sto3g_R2_B2_C1"})
        (inputs / "child.py").write_text(_CHILD)
        for system in args.systems:
            for mode in ("count", "flatten"):
                add(system, mode, run_guarded([str(REF_PYTHON), str(inputs / "child.py"),
                                               str(inputs), system, mode]))
        if not args.no_sim:
            (inputs / "ref_sim.py").write_text(_REF_SIM)
            add("h2_sto3g_R2_B2_C1", "simulate_block", run_guarded(
                [str(REF_PYTHON), str(inputs / "ref_sim.py"),
                 str(ROOT / "data" / "dfthc" / "h2_sto3g_R2_B2_C1" / "factors.npz")]))
            (inputs / "our_sim.py").write_text(_OUR_SIM)
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            add("h2_sto3g_R2_B2_C1", "simulate_block_ours",
                run_guarded([sys.executable, str(inputs / "our_sim.py")], env=env))

    import importlib.metadata

    versions = subprocess.run(
        [str(REF_PYTHON), "-c", "import importlib.metadata as m, json; print(json.dumps("
         "{n: m.version(n) for n in ('qualtran', 'dfthcbe', 'cirq-core', 'sympy')}))"],
        capture_output=True, text=True, check=True).stdout
    data = {"environment": {"platform": platform.platform(), "python": platform.python_version(),
                            "reference": json.loads(versions),
                            "pennylane": importlib.metadata.version("pennylane")},
            "limits": {"memory_mb": LIMIT_MB, "timeout_s": TIMEOUT_S},
            "records": sorted(records.values(), key=lambda r: (SYSTEMS.index(r["system"]),
                                                               r["mode"]))}
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
