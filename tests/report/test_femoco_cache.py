"""The cached FeMoco-54 compile of the walk and the controlled walk (spec 2.3 D-3, 13.3; master
plan Task 4.5).

Nothing here compiles. The release-tier test reads ``docs/resources.json``: the entry of the
current code hash for each program holds the counts of spec 9.24 (walk T = 9,793, M = 2,524,
A = 8,675; controlled walk T = 9,795, M = 2,524, A = 8,677), the code hash, the environment, the
wall time and the peak memory, and the partition rows of the report (tape mode, no compile) times
their calls per walk, plus in M the block encodings' own junk measurements
(:func:`dfthc.costs.junk_measurements`; spec 16, Q-1 variant (b)), give exactly the cached T, M
and A. A missing or stale entry fails and names the recompile, which only the orchestrator runs.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dfthc import costs
from dfthc.compile import ANALYZE_NAMES
from dfthc.params import FEMOCO54
from dfthc.testing.provenance import code_hash

REPO = Path(__file__).resolve().parents[2]
EXPECTED = {"DFTHCWalk": (9793, 2524, 8675), "ControlledDFTHCWalk": (9795, 2524, 8677)}
RECOMPILE = ("scripts/resource_report.py --large femoco54 (the orchestrator, alone, under the "
             "watchdog; master plan 12.9 and Task 4.5)")


def _load_script():
    spec = importlib.util.spec_from_file_location("resource_report",
                                                  REPO / "scripts" / "resource_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["resource_report"] = module
    spec.loader.exec_module(module)
    return module


report = _load_script()


def femoco_cache_problems(record: dict, hash_: str) -> list[str]:
    """What is wrong with the cached FeMoco-54 compiles of ``record`` under code hash ``hash_``:
    a missing entry (naming the recompile), counts other than spec 9.24's, or an entry without
    its provenance."""
    problems, cache = [], record.get("compiled", {})
    for program, expected in EXPECTED.items():
        key = f"femoco54/{program}"
        entry = report.current(cache, key, hash_)
        if entry is None:
            others = sorted({e["code_hash"][:12] for e in cache.get(key, [])})
            problems.append(f"{key}: no compile for code hash {hash_[:12]} (cached: "
                            f"{others or 'none'}); recompile with {RECOMPILE}")
            continue
        counts = (entry["toffoli"], entry["measurements"], entry["measured_and_uncomputes"])
        if counts != expected:
            problems.append(f"{key}: cached T, M, A {counts} != {expected}")
        if not set(entry["counts"]) <= ANALYZE_NAMES:
            problems.append(f"{key}: gates outside the gate set {set(entry['counts'])}")
        for field in ("environment", "seconds"):
            if not entry.get(field):
                problems.append(f"{key}: no {field}")
        if not entry.get("watchdog", {}).get("peak_mb"):
            problems.append(f"{key}: no peak memory")
    return problems


def test_stale_femoco_entry_names_the_recompile():
    """An entry of another code hash is not a current measurement: the check fails and names
    the recompile (per commit; nothing is compiled)."""
    stale = {"code_hash": "0" * 64, "status": "compiled", "toffoli": 9793, "measurements": 2524,
             "measured_and_uncomputes": 8675, "counts": {}, "environment": {"python": "3.12"},
             "seconds": 1.0, "watchdog": {"peak_mb": 1.0}}
    problems = femoco_cache_problems({"compiled": {"femoco54/DFTHCWalk": [stale]}}, "1" * 64)
    assert len(problems) == 2
    assert problems[0].startswith("femoco54/DFTHCWalk: no compile for code hash 111111111111 "
                                  "(cached: ['000000000000']); recompile with")
    assert "--large femoco54" in problems[1]


@pytest.mark.release
@pytest.mark.femoco
def test_femoco_cache(monkeypatch):
    """D-3 from the cache: the current entries hold spec 9.24's counts with their provenance,
    and the sum over the partition rows of calls per walk times the closed form, plus in M the
    block encodings' junk measurements, equals the cached walk and controlled-walk totals of T, M
    and A; the report's FeMoco-54 QPE summary gives T, M and A from the cached controlled walk
    (spec 11.3; A = (2^t - 1) A_C(W) + (t^3 - t)/6). Nothing is compiled."""
    monkeypatch.setattr(report, "compile_counts", lambda *args: pytest.fail("no compile"))
    monkeypatch.setattr(report, "measure", lambda *args: pytest.fail("no compile"))
    record = json.loads((REPO / "docs" / "resources.json").read_text())
    hash_ = code_hash()
    assert femoco_cache_problems(record, hash_) == []
    s = report.build_system("femoco54")
    rows = report.build_rows(s)
    cache = record["compiled"]
    junk = costs.junk_measurements(FEMOCO54).measurements
    for program, calls in (("DFTHCWalk", "calls"), ("ControlledDFTHCWalk", "controlled_calls")):
        entry = report.current(cache, f"femoco54/{program}", hash_)
        parts = [row for row in rows if row.partition and getattr(row, calls)]
        total = (sum(getattr(r, calls) * r.closed.toffoli for r in parts),
                 sum(getattr(r, calls) * r.closed.measurements for r in parts) + junk,
                 sum(getattr(r, calls) * r.uncomputes for r in parts))
        assert total == (entry["toffoli"], entry["measurements"],
                         entry["measured_and_uncomputes"]) == EXPECTED[program]
    cwalk = report.current(cache, "femoco54/ControlledDFTHCWalk", hash_)
    summary = record["systems"]["femoco54"]
    assert summary["qpe_basis"].startswith("T, M and A from the compiled controlled walk")
    for bits in (1, 2, 3, 10, 20, 23):
        expected = {"toffoli": (2**bits - 1) * cwalk["toffoli"] + (bits**3 - bits) // 6,
                    "measurements": (2**bits - 1) * cwalk["measurements"] + bits,
                    "measured_and_uncomputes": (2**bits - 1) * cwalk["measured_and_uncomputes"]
                    + (bits**3 - bits) // 6, "wires": report.qpe_wires(s, bits)}
        assert summary["qpe"][str(bits)] == expected, bits
        if bits <= 3:
            qpe = costs.iterative_qpe(FEMOCO54, bits)
            assert (qpe.toffoli, qpe.measurements) == (expected["toffoli"],
                                                       expected["measurements"])
