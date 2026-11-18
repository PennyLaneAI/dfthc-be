"""The cached FeMoco-54 compile of the walk and the controlled walk (spec 2.3 D-3, 13.3; master
plan Task 4.5).

Nothing here compiles. The release-tier test reads ``docs/resources.json``: the entry of the
current code hash for each program holds the counts of spec 9.24 (walk T = A = 9,790,
M = 2,418; controlled walk T = A = 9,792, M = 2,418), the code hash, the environment, the
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
from dfthc.testing.provenance import code_hash, environment

REPO = Path(__file__).resolve().parents[2]
EXPECTED = {"DFTHCWalk": (9790, 2418, 9790), "ControlledDFTHCWalk": (9792, 2418, 9792)}
EXPECTED_WIRES = {"DFTHCWalk": 1049, "ControlledDFTHCWalk": 1050}
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
    a missing entry (naming the recompile), counts other than spec 9.24's, an entry without its
    provenance, or one recorded in an environment other than the running one (every field of
    ``provenance.environment()`` but the platform: Python, the toolchain and the lock's digest)."""
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
        gates = entry["counts"]
        outside = set(gates) - ANALYZE_NAMES
        if outside:
            problems.append(f"{key}: gates outside the gate set {outside}")
        analyzed = (gates.get("Toffoli", 0) + gates.get("TemporaryAND", 0),
                    gates.get("MidCircuitMeasure", 0),
                    gates.get("Adjoint(TemporaryAND)", 0))
        if analyzed != counts:
            problems.append(f"{key}: gate dictionary gives T, M, A {analyzed} != {counts}")
        if entry.get("num_wires") != EXPECTED_WIRES[program]:
            problems.append(f"{key}: cached wires {entry.get('num_wires')} != "
                            f"{EXPECTED_WIRES[program]}")
        for field in ("environment", "seconds"):
            if not entry.get(field):
                problems.append(f"{key}: no {field}")
        if entry.get("environment"):
            recorded, running = entry["environment"], environment()
            differs = sorted(field for field in (recorded.keys() | running.keys()) - {"platform"}
                             if recorded.get(field) != running.get(field))
            if differs:
                problems.append(f"{key}: environment differs from the running one in {differs}")
        if not entry.get("watchdog", {}).get("peak_mb"):
            problems.append(f"{key}: no peak memory")
    return problems


def test_stale_femoco_entry_names_the_recompile():
    """An entry of another code hash is not a current measurement: the check fails and names
    the recompile (per commit; nothing is compiled)."""
    stale = {"code_hash": "0" * 64, "status": "compiled", "toffoli": 9790, "measurements": 2418,
             "measured_and_uncomputes": 9790, "counts": {}, "environment": {"python": "3.12"},
             "seconds": 1.0, "watchdog": {"peak_mb": 1.0}}
    problems = femoco_cache_problems({"compiled": {"femoco54/DFTHCWalk": [stale]}}, "1" * 64)
    assert len(problems) == 2
    assert problems[0].startswith("femoco54/DFTHCWalk: no compile for code hash 111111111111 "
                                  "(cached: ['000000000000']); recompile with")
    assert "--large femoco54" in problems[1]


def valid_entry(program: str) -> dict:
    """A current entry of ``program`` that passes every arm of the check: spec 9.24's counts, a
    gate dictionary that gives them, the expected wires and full provenance."""
    toffoli, measurements, uncomputes = EXPECTED[program]
    return {"code_hash": "1" * 64, "status": "compiled", "toffoli": toffoli,
            "measurements": measurements, "measured_and_uncomputes": uncomputes,
            "counts": {"TemporaryAND": toffoli, "Adjoint(TemporaryAND)": uncomputes,
                       "MidCircuitMeasure": measurements, "CNOT": 1},
            "num_wires": EXPECTED_WIRES[program], "environment": environment(),
            "seconds": 36.5, "watchdog": {"peak_mb": 1692.1}}


def _more_toffolis(entry):
    entry["toffoli"] += 1
    entry["counts"]["TemporaryAND"] += 1


DEFECTS = {
    "counts": (_more_toffolis, "cached T, M, A (9791, 2418, 9790) != (9790, 2418, 9790)"),
    "gate-set": (lambda e: e["counts"].update(RX=1), "gates outside the gate set {'RX'}"),
    "dictionary": (lambda e: e["counts"].update(TemporaryAND=e["counts"]["TemporaryAND"] + 1),
                   "gate dictionary gives T, M, A (9791, 2418, 9790) != (9790, 2418, 9790)"),
    "wires": (lambda e: e.update(num_wires=1050), "cached wires 1050 != 1049"),
    "environment": (lambda e: e.pop("environment"), "no environment"),
    "toolchain": (lambda e: e["environment"].update(pennylane="0.0"),
                  "environment differs from the running one in ['pennylane']"),
    "seconds": (lambda e: e.pop("seconds"), "no seconds"),
    "peak": (lambda e: e.update(watchdog={}), "no peak memory"),
}
"""One defect per arm of ``femoco_cache_problems`` and the one problem it must report."""


@pytest.mark.parametrize("defect", DEFECTS)
def test_each_arm_of_the_cache_check_reports_its_defect(defect):
    """Final review m-11: a valid record has no problem, a platform other than the running one
    included (m-13); one defect injected into a copy of the walk's valid entry gives exactly the
    one problem of its arm (per commit; nothing is compiled)."""
    record = {"compiled": {f"femoco54/{program}": [valid_entry(program)] for program in EXPECTED}}
    record["compiled"]["femoco54/ControlledDFTHCWalk"][0]["environment"]["platform"] = "other"
    assert femoco_cache_problems(record, "1" * 64) == []
    inject, problem = DEFECTS[defect]
    inject(record["compiled"]["femoco54/DFTHCWalk"][0])
    assert femoco_cache_problems(record, "1" * 64) == [f"femoco54/DFTHCWalk: {problem}"]


@pytest.mark.release
@pytest.mark.femoco
def test_femoco_cache(monkeypatch):
    """D-3 from the cache: the current entries hold spec 9.24's counts with their provenance,
    and the sum over the partition rows of calls per walk times the closed form, plus in M the
    block encodings' junk measurements, equals the cached walk and controlled-walk totals of T, M
    and A; the report's FeMoco-54 QPE summary gives T and A from the cached controlled walk and M
    from the cached walk (spec 11.3; A = (2^t - 1) A_C(W) + (t^3 - t)/6), and its basis sentence
    names the compile (`--large femoco54`, once each per code hash), not a task. Nothing is
    compiled."""
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
    walk = report.current(cache, "femoco54/DFTHCWalk", hash_)
    cwalk = report.current(cache, "femoco54/ControlledDFTHCWalk", hash_)
    summary = record["systems"]["femoco54"]
    assert summary["qpe_basis"].startswith("Toffolis from the compiled controlled walk "
                                           "(`--large femoco54`, once per code hash): ")
    for bits in (1, 2, 3, 10, 20, 23):
        expected = {"toffoli": (2**bits - 1) * cwalk["toffoli"] + (bits**3 - bits) // 6,
                    "measurements": (2**bits - 1) * walk["measurements"] + bits,
                    "measured_and_uncomputes": (2**bits - 1) * cwalk["measured_and_uncomputes"]
                    + (bits**3 - bits) // 6, "wires": report.qpe_wires(s, bits)}
        assert summary["qpe"][str(bits)] == expected, bits
        if bits <= 3:
            qpe = costs.iterative_qpe(FEMOCO54, bits)
            assert (qpe.toffoli, qpe.measurements) == (expected["toffoli"],
                                                       expected["measurements"])
