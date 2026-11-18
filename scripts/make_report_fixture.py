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

"""Generate ``tests/fixtures/dfthcbe/report_toffolis.json``: dfthcbe's Toffolis per block at every
parameter set of the resource report, from its own call graph. Runs only in ``.venv-ref``.

Usage, from the repository root (environment: scripts/make_venv_ref.sh)::

    .venv-ref/bin/python scripts/make_report_fixture.py [--out tests/fixtures/dfthcbe]

For each report system (``dfthc.costs_reference.default_params`` of a vendored system's shape),
``H2_STO3G_TEST`` and ``FEMOCO54``, it builds ``DFTHCWalk.symbolic`` (no table) and evaluates the
Toffolis (``Toffoli + CSwap + And + T/4`` of Qualtran's ``QECGatesCost``, as dfthcbe's tests and
``scripts/make_dfthcbe_fixtures.py`` count them) of every bloq that has a row in the report, keyed
as :func:`dfthc.costs_reference.dfthcbe_blocks` keys them. The test
``tests/report/test_report.py::test_dfthcbe_blocks_equal_the_call_graph`` compares that port with
this file. Qualtran and dfthcbe are imported inside the functions, so the main environment can
import this module's constants. The fixture records the dfthcbe tree it was evaluated on as
``scripts/make_dfthcbe_fixtures.py``'s provenance does: the source URL and digest and the
installed package's digest (``installed_digest``).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from dfthc.costs_reference import default_params  # noqa: E402
from dfthc.data.factors import SYSTEMS, load_factors  # noqa: E402
from dfthc.params import FEMOCO54, H2_STO3G_TEST  # noqa: E402
from make_dfthcbe_fixtures import (  # noqa: E402
    DFTHCBE_SOURCE_SHA256,
    DFTHCBE_SOURCE_URL,
    installed_digest,
)

FIXTURE = "report_toffolis.json"
DFTHCBE_COMMIT = "930255020cec30172e0a7d17c0071d9476a687d1"


def report_parameters() -> dict:
    """The report's parameter sets by system name (spec 13.3 step 1), and the test preset."""
    sets = {}
    for system in SYSTEMS:
        f = load_factors(system)
        sets[system] = default_params(f.N, f.R, f.B, f.C, f.n_plus)
    sets["femoco54"] = FEMOCO54
    sets["H2_STO3G_TEST"] = H2_STO3G_TEST
    return sets


def call_graph_toffolis(p) -> dict[str, int]:
    """dfthcbe's Toffolis per bloq of the symbolic walk at ``p``."""
    import sympy
    from dfthcbe import DFTHCWalk
    from qualtran.bloqs.mcmt import MultiControlX
    from qualtran.bloqs.rotations.zpow_via_phase_gradient import ZPowConstViaPhaseGradient
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    symbols = sympy.symbols("N R B C b_rot b_k1 b_k2 k_1 k_2 k_e1 k_e2", positive=True,
                            integer=True)
    walk = DFTHCWalk.symbolic(*symbols, p.s_u)
    values = dict(zip(symbols, (p.N, p.R, p.B, p.C, p.b_rot, p.b_k1, p.b_k2, p.k1, p.k2, p.ke1,
                                p.ke2), strict=True))

    def toffoli(bloq) -> int:
        cost = get_cost_value(bloq, QECGatesCost())
        total = sympy.sympify(cost.toffoli + cost.cswap + cost.and_bloq + cost.t / 4)
        return int(total.subs(values))

    be = walk.block_encoding
    outer, inner, select = be.outer, be.inner, be.select
    bloqs = {
        "rotation": ZPowConstViaPhaseGradient(exponent=sympy.Rational(1, 3),
                                              phase_grad_bitsize=p.s_u),
        "uniform_x_o": outer.uniform, "uniform_x_o_adjoint": outer.uniform.adjoint(),
        "uniform_b": inner.uniform, "uniform_b_adjoint": inner.uniform.adjoint(),
        "outer_read": outer.qroam, "outer_erase": outer.qroam.adjoint(),
        "inner_read": inner.qroam, "inner_erase": inner.qroam.adjoint(),
        "tail_angle_load": select.tail_qrom, "sf_angle_load": select.sf_qrom,
        "spin_swap": select.uswap,
        "givens_ladder": select.ladder, "givens_ladder_dagger": select.ladder.adjoint(),
        "majorana_gate": select.maj, "majorana_gate_adjoint": select.maj.adjoint(),
        "angle_erase": select.eraser,
        "outer_prepare": outer, "outer_unprepare": outer.adjoint(),
        "inner_prepare": inner, "inner_unprepare": inner.adjoint(),
        "inner_select": select, "inner_select_dagger": select.adjoint(),
        "ref_t2": be.ref_t2, "ref_ab": walk.ref_ab,
        "block_encoding": be, "controlled_block_encoding": be.controlled(),
        "walk": walk, "controlled_walk": walk.controlled(),
    }
    counts = {name: toffoli(bloq) for name, bloq in bloqs.items()}
    counts["spin_mux"] = (toffoli(MultiControlX(cvs=(0, 1)))
                          + toffoli(MultiControlX(cvs=(1, 1))))
    counts["controlled_ref_t2"] = (counts["controlled_block_encoding"]
                                   - counts["block_encoding"] + counts["ref_t2"])
    counts["controlled_ref_ab"] = counts["controlled_walk"] - counts["controlled_block_encoding"]
    return dict(sorted(counts.items()))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPO / "tests" / "fixtures" / "dfthcbe")
    out = parser.parse_args(argv).out
    out.mkdir(parents=True, exist_ok=True)
    sets = report_parameters()
    fixture = {
        "dfthcbe_commit": DFTHCBE_COMMIT,
        "dfthcbe_source_url": DFTHCBE_SOURCE_URL,
        "dfthcbe_source_sha256": DFTHCBE_SOURCE_SHA256,
        "dfthcbe_installed_sha256": installed_digest(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("dfthcbe", "qualtran", "sympy")},
        "method": "DFTHCWalk.symbolic call graph evaluated at each parameter set; Toffolis = "
                  "Toffoli + CSwap + And + T/4 of QECGatesCost; spin_mux = one MultiControlX "
                  "(0, 1) plus one (1, 1); controlled_ref_t2 and controlled_ref_ab are the "
                  "controlled block encoding's and controlled walk's increments",
        "parameters": {name: dataclasses.asdict(p) for name, p in sets.items()},
        "toffolis": {name: call_graph_toffolis(p) for name, p in sets.items()},
    }
    (out / FIXTURE).write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"wrote {out / FIXTURE}")


if __name__ == "__main__":
    main()
