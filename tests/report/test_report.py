"""The resource report (spec 13.3; master plan Task 4.3): ``scripts/resource_report.py``,
``dfthc.costs_reference`` and ``dfthc.testing.provenance``.

Per commit: every row and column at ``H2_STO3G_TEST`` (one compile per row, in this process), the
two failing checks, the watched and projection-gated ``--large`` path on a stand-in with its stop
policy, a projection for every large row (also after ``fe2s2``'s entries join), the code-hash
cache, the reason codes and the QPE summary at FeMoco-54, dfthcbe's call graph against the port,
Table III, and the projection method on monotone and non-monotone points. Nightly: the whole
report at ``naphthalene``, every program in its own watched subprocess.
"""

import dataclasses
import functools
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from dfthc import costs, costs_reference
from dfthc.params import FEMOCO54, H2_STO3G_TEST, WalkParams
from dfthc.testing import provenance
from dfthc.testing.fixtures import load_dfthcbe_fixture

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "resource_report.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("resource_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["resource_report"] = module
    spec.loader.exec_module(module)
    return module


report = _load_script()

H2_ROWS = [
    "DFTHCWalk", "DFTHCBlockEncoding", "OuterPrepare", "FlaggedUniformPrep@OuterPrepare",
    "QROAMRead@OuterPrepare", "InnerPrepare", "FlaggedUniformPrep@InnerPrepare",
    "PhaseGradientRotation", "QROAMRead@InnerPrepare", "InnerSigns", "InnerSelect",
    "TailAngleLoad", "SFAngleLoad", "SpinMux", "SpinSwap", "GivensLadder", "GivensRotation",
    "MajoranaGate", "GivensLadderDagger", "Adjoint(GivensRotation)", "AngleErase",
    "InnerUnprepareFromOutcomes", "Adjoint(FlaggedUniformPrep)@InnerUnprepareFromOutcomes",
    "RefT2", "InnerSelectDagger", "Adjoint(MajoranaGate)", "OuterUnprepareFromOutcomes",
    "Adjoint(FlaggedUniformPrep)@OuterUnprepareFromOutcomes", "RefAB", "ControlledDFTHCWalk",
    "ControlledDFTHCBlockEncoding", "ControlledRefT2", "ControlledRefAB",
    "PhaseFixup@InnerUnprepareFromOutcomes", "PhaseFixup@OuterUnprepareFromOutcomes",
    "ControlledWalkPower(p=1)", "ControlledWalkPower(p=2)", "QPE(t=1)", "QPE(t=2)", "QPE(t=3)",
]
"""Every program of the report at H2_STO3G_TEST, in report order: the walk calls no QROAMErase
(the unprepares from outcomes erase themselves, spec 16, Q-1 variant (b)), so the fix-up rows
take their arguments; the four reflection rows compile the helpers ref_t2 and ref_ab."""
PARTITION = {"OuterPrepare": (1, 1), "InnerPrepare": (2, 2), "InnerSigns": (2, 2),
             "TailAngleLoad": (2, 2), "SFAngleLoad": (2, 2), "SpinMux": (4, 4),
             "SpinSwap": (4, 4), "GivensLadder": (2, 2), "MajoranaGate": (1, 1),
             "GivensLadderDagger": (2, 2), "AngleErase": (2, 2),
             "InnerUnprepareFromOutcomes": (2, 2), "RefT2": (1, 0),
             "Adjoint(MajoranaGate)": (1, 1), "OuterUnprepareFromOutcomes": (1, 1),
             "RefAB": (1, 0), "ControlledRefT2": (0, 1), "ControlledRefAB": (0, 1)}
"""The partition rows and their calls per walk and per controlled walk (spec 3.5)."""


@functools.cache
def h2():
    s = report.build_system("H2_STO3G_TEST")
    return s, {row.id: row for row in report.build_rows(s)}


def _counts(record: dict, which: str) -> tuple:
    values = record[which]
    return values and (values["toffoli"], values["measurements"],
                       values["measured_and_uncomputes"])


def test_report_h2_rows_and_columns(tmp_path):
    """Spec 13.3 at H2_STO3G_TEST without compiling: the rows, their calls, levels, steps,
    closed forms, the dfthcbe and paper values with their deltas and reason codes, the Clifford
    labels and the qubits; the partition sums to the walks; main() writes both files with every
    row and column."""
    s, rows = h2()
    assert list(rows) == H2_ROWS
    assert {r.id: (r.calls, r.controlled_calls) for r in rows.values() if r.partition} == PARTITION
    walk, cwalk = rows["DFTHCWalk"], rows["ControlledDFTHCWalk"]
    assert (walk.closed, walk.uncomputes) == (costs.walk(H2_STO3G_TEST, False), 108)
    assert (cwalk.closed, cwalk.uncomputes) == (costs.walk(H2_STO3G_TEST, True), 110)
    assert (walk.wires, cwalk.wires, rows["QPE(t=3)"].wires) == (55, 56, 59)
    assert rows["QROAMRead@InnerPrepare"].level == "L1" and rows["SpinSwap"].level == "L2"
    assert rows["RefAB"].level == "L3" and rows["QPE(t=1)"].level == "L4"
    assert rows["AngleErase"].steps == "18" and rows["InnerUnprepareFromOutcomes"].steps == "19-22"
    assert rows["PhaseFixup@InnerUnprepareFromOutcomes"].steps == "21"
    assert report.clifford_kind(rows["SpinSwap"]) == "exact"
    assert report.clifford_kind(rows["InnerUnprepareFromOutcomes"]) == (
        "upper bound (CNOT, PauliX); expected, hinted (CZ, PauliZ)")
    assert report.clifford_kind(rows["PhaseFixup@OuterUnprepareFromOutcomes"]) == (
        "expected, hinted (CZ, PauliZ)")

    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "resources.json").read_text())
    records = {r["id"]: r for r in record["systems"]["H2_STO3G_TEST"]["rows"]}
    assert list(records) == H2_ROWS
    assert {r["status"] for r in records.values()} == {"pending"}
    keys = {"id", "level", "steps", "calls", "controlled_calls", "partition", "status", "measured",
            "closed", "dfthcbe", "paper", "paper_step18", "delta_dfthcbe", "delta_paper",
            "codes_dfthcbe", "codes_paper", "cliffords", "clifford_kind", "wires", "seconds",
            "peak_mb", "gates", "projection"}
    assert all(set(r) == keys for r in records.values())
    w = records["DFTHCWalk"]
    assert (_counts(w, "closed"), w["dfthcbe"], w["paper"]) == ((158, 50, 108), 167, 205)
    assert (w["delta_dfthcbe"], w["delta_paper"]) == (-9, -47)
    assert w["codes_dfthcbe"]["ref-target"] == -1 and w["codes_paper"]["qpe-ctrl"] == -1
    assert records["SpinMux"]["codes_dfthcbe"] == {"mux": -1}
    assert records["ControlledRefT2"]["codes_paper"] == {"ref-unif": -1, "ctrl-ref-t2": 1}
    assert records["AngleErase"]["paper_step18"] == records["AngleErase"]["paper"] + 2
    for r in records.values():
        for side in ("dfthcbe", "paper"):
            delta, codes = r[f"delta_{side}"], r[f"codes_{side}"]
            assert codes == {} if delta is None else sum(codes.values()) == delta, (r["id"], side)
            assert 0 not in codes.values() and set(codes) <= set(report.REASON_CODES)
    text = (tmp_path / "resources.md").read_text()
    assert all(f"`{row_id}`" in text for row_id in H2_ROWS)
    assert "| Row | Level | Steps | Calls (W, C(W)) | T / M / A measured |" in text


@pytest.mark.compiled
@pytest.mark.parametrize("row_id", H2_ROWS)
def test_report_h2(row_id):
    """Every program at H2_STO3G_TEST compiles to exactly its closed form (T, M and A), on
    exactly the qubits the report gives it; compiled_resources checks the gate-set closure."""
    s, rows = h2()
    row = rows[row_id]
    counts = report.compile_counts(s, row)
    assert (counts["toffoli"], counts["measurements"], counts["measured_and_uncomputes"]) == (
        row.closed.toffoli, row.closed.measurements, row.uncomputes)
    assert counts["num_wires"] == row.wires


@pytest.mark.slow
@pytest.mark.compiled
@pytest.mark.subprocess
def test_report_mid(tmp_path):
    """The report at naphthalene (MID_NAPHTHALENE's parameters): every program compiles in its
    own watched subprocess to its closed form, and the partition gives the measured walks."""
    assert report.main(["--systems", "naphthalene", "--out", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "resources.json").read_text())["systems"]["naphthalene"]
    assert record["params"] == {"N": 10, "R": 8, "B": 6, "C": 5, "n_plus": 5, "b_rot": 15,
                                "b_k1": 15, "b_k2": 15, "s_u": 7, "k1": 1, "k2": 2, "ke1": 3,
                                "ke2": 1}
    rows = {r["id"]: r for r in record["rows"]}
    assert {r["status"] for r in rows.values()} == {"measured"}
    assert _counts(rows["DFTHCWalk"], "measured") == (1910, 502, 1675)
    assert _counts(rows["ControlledDFTHCWalk"], "measured") == (1912, 502, 1677)
    assert all(_counts(r, "measured") == _counts(r, "closed") for r in rows.values())


@pytest.mark.compiled
@pytest.mark.subprocess
def test_report_fails_on_mismatch(tmp_path, monkeypatch, capsys):
    """Spec 13.3 step 4: a closed form that differs from the compiled count, and a nonzero delta
    without a reason code, each make the script exit 1 (after writing both files)."""
    monkeypatch.setattr(costs, "spin_swap", lambda p: costs.Cost(p.N + 1))
    out = tmp_path / "mismatch"
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "SpinSwap", "--out",
                        str(out)]) == 1
    assert "H2_STO3G_TEST/SpinSwap: measured" in capsys.readouterr().err
    assert (out / "resources.md").exists()
    monkeypatch.undo()
    causes = costs_reference.delta_causes

    def without_mux(p):
        return causes(p) | {"spin_mux": ({}, {})}

    monkeypatch.setattr(costs_reference, "delta_causes", without_mux)
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "--out",
                        str(tmp_path / "codes")]) == 1
    assert "SpinMux: delta -1 against dfthcbe != its reason codes {} (sum 0)" in (
        capsys.readouterr().err)


def test_report_fails_on_a_partition_that_misses_the_walk(tmp_path, monkeypatch, capsys):
    """Spec 13.3 step 4, the cross-check (final review m-14; no compile): a partition row whose
    closed form is one Toffoli off makes the partition sum differ from the walk's closed form,
    and the script exits 1."""
    closed_form = report.closed_form

    def shifted(name, op, p):
        cost, uncomputes = closed_form(name, op, p)
        if name == "SpinSwap":
            cost = costs.Cost(cost.toffoli + 1, cost.measurements)
        return cost, uncomputes

    monkeypatch.setattr(report, "closed_form", shifted)
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert re.search(r"CHECK FAILED: H2_STO3G_TEST: partition \{[^}]*\} != closed DFTHCWalk",
                     err), err


@pytest.mark.compiled
@pytest.mark.subprocess
def test_report_fails_on_a_measurement_mismatch(tmp_path, monkeypatch, capsys):
    """Spec 13.3 step 4, the M arm (final review m-14): a closed form whose M differs from the
    compiled count, with T unchanged, makes the script exit 1 (one SpinSwap compile)."""
    monkeypatch.setattr(costs, "spin_swap", lambda p: costs.Cost(p.N, 1))
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "SpinSwap", "--out",
                        str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "H2_STO3G_TEST/SpinSwap: measured {'toffoli': 2, 'measurements': 0," in err, err


@pytest.mark.subprocess
def test_large_compile_is_watched(tmp_path, monkeypatch):
    """--large on a stand-in: a program that allocates past a 200 MB cap is killed by the
    watchdog and recorded as killed, not as a measurement; a program projected above the limits
    is refused, with its projection, without compiling."""
    standin = [sys.executable, "-c", "import time\nx = bytearray(400 * 2**20)\n"
               "x[::4096] = b'1' * len(x[::4096])\ntime.sleep(20)"]
    monkeypatch.setattr(report, "child_command", lambda s, row: standin)
    monkeypatch.setattr(report, "LARGE_CAP_MB", 200)
    monkeypatch.setattr(report, "calibration", lambda *args: [(1, 0.1, 10.0), (2, 0.2, 10.0)])
    argv = ["--systems", "H2_STO3G_TEST", "--large", "H2_STO3G_TEST", "--rows", "SpinSwap",
            "--out", str(tmp_path)]
    assert report.main(argv) == 0
    record = json.loads((tmp_path / "resources.json").read_text())
    entry = record["compiled"]["H2_STO3G_TEST/SpinSwap"][-1]
    assert (entry["status"], entry["reason"]) == ("killed", "memory")
    assert entry["watchdog"]["cap_mb"] == 200 and entry["watchdog"]["survivors"] == []
    assert entry["watchdog"]["timeout_s"] == report.LARGE_TIMEOUT_S
    row = next(r for r in record["systems"]["H2_STO3G_TEST"]["rows"] if r["id"] == "SpinSwap")
    assert (row["status"], row["measured"]) == ("killed", None)

    monkeypatch.setattr(report, "LARGE_CAP_MB", report.MEMORY_CAP_MB)
    monkeypatch.setattr(report, "calibration", lambda *args: [(1, 1.0, 100.0), (2, 2.0, 5000.0)])

    def no_compile(*args):
        raise AssertionError("a refused program must not be compiled")

    monkeypatch.setattr(report, "measure", no_compile)
    assert report.main(argv) == 0
    record = json.loads((tmp_path / "resources.json").read_text())
    entries = record["compiled"]["H2_STO3G_TEST/SpinSwap"]
    assert [e["status"] for e in entries] == ["killed", "refused"]
    projection = entries[-1]["projection"]
    assert not projection["allowed"] and "GB > 7.5 GB" in projection["reason"]
    row = next(r for r in record["systems"]["H2_STO3G_TEST"]["rows"] if r["id"] == "SpinSwap")
    assert (row["status"], row["measured"]) == ("refused", None)


def _standin(name: str):
    """H2_STO3G_TEST's system and rows under another system's name (the stop policy keys on the
    name; nothing is compiled)."""
    s = dataclasses.replace(report.build_system("H2_STO3G_TEST"), name=name)
    return s, report.build_rows(s)


def test_large_stops_after_a_refusal_at_femoco(monkeypatch):
    """Under --large at FeMoco-54, a refused first program stops the run: the controlled walk is
    never measured and is recorded as skipped, and the log names the escalation."""
    s, rows = _standin("femoco54")
    monkeypatch.setattr(report, "projection", lambda *args: {"allowed": False,
                                                              "reason": "8.0 GB > 7.5 GB"})
    monkeypatch.setattr(report, "measure", lambda *args: pytest.fail("nothing is compiled"))
    cache = {}
    log = report.run_system(s, rows, cache, "1" * 64, projected=True, only=None)
    assert [cache[f"femoco54/{p}"][-1]["status"] for p in report.FEMOCO_PROGRAMS] == [
        "refused", "skipped"]
    assert set(cache) == {"femoco54/DFTHCWalk", "femoco54/ControlledDFTHCWalk"}
    assert any("stopped; escalate to the user" in line for line in log)


@pytest.mark.parametrize("status", ["killed", "failed"])
def test_large_stops_after_a_failed_or_killed_compile(monkeypatch, status):
    """Under --large, the first killed or failed compile stops the system's run: the next
    program is never measured and is recorded as skipped, and the log names the escalation."""
    s, rows = _standin("fe2s2")
    monkeypatch.setattr(report, "projection", lambda *args: {"allowed": True,
                                                              "reason": "within the limits"})
    measured = []

    def stops_once(system, row, *args):
        assert not measured, f"the run continued after a {status} compile"
        measured.append(row.id)
        return {"code_hash": "1" * 64, "environment": {}, "status": status,
                "reason": "memory" if status == "killed" else "compile",
                "watchdog": {"seconds": 1.0, "peak_mb": 7700.0}}

    monkeypatch.setattr(report, "measure", stops_once)
    cache = {}
    log = report.run_system(s, rows, cache, "1" * 64, projected=True,
                            only={"SpinMux", "SpinSwap"})
    assert measured == ["SpinMux"]
    assert [cache[f"fe2s2/{r}"][-1]["status"] for r in ("SpinMux", "SpinSwap")] == [
        status, "skipped"]
    assert any(f"stopped; the orchestrator reports the {status} compile" in line for line in log)


@functools.cache
def _compiled_rows(name: str) -> tuple[list, dict[str, int]]:
    """The rows the report compiles at a system and their gates (tape mode, no compile)."""
    s = report.build_system(name)
    rows = [row for row in report.build_rows(s) if report.compiled_here(s, row)]
    return rows, report.gates(rows)


def _synthetic_entries(cache: dict, name: str, hash_: str) -> None:
    """Current entries for every sized row of the system, with made-up seconds and MB."""
    rows, sizes = _compiled_rows(name)
    for row in rows:
        if row.id in sizes:
            size = sizes[row.id]
            cache[f"{name}/{row.id}"] = [{
                "code_hash": hash_, "status": "compiled", "gates": size,
                "family": report.family(row),
                "watchdog": {"seconds": 1.0 + size / 20_000, "peak_mb": 200.0 + 3 * size / 1000}}]


def test_every_large_row_projects():
    """Every program the report compiles at fe2s2 and fe4s4 has a size measure (gates from its
    operator's resource functions) and an allowed projection from a calibration built like the
    report's: the real rows and gates of the H2 systems and naphthalene, with made-up seconds and
    MB (a synthetic cache, so that a code-hash change does not break this test). Rows no larger
    than a measured member of their family take the family's largest measurement; the others are
    extrapolated (nothing is compiled)."""
    hash_ = "1" * 64
    cache = {}
    for name in report.SMALL:
        _synthetic_entries(cache, name, hash_)
    methods = set()
    for name in ("fe2s2", "fe4s4"):
        rows, sizes = _compiled_rows(name)
        assert len(rows) == len(sizes) == 38, name
        for row in rows:
            estimate = report.projection(cache, row, hash_, sizes[row.id])
            assert estimate["allowed"], (name, row.id, estimate["reason"])
            methods.add(estimate["method"])
    assert methods == {"bounded", "k1"}


def test_large_sequence_projects_fe4s4_after_fe2s2():
    """The orchestrator's order: after `--large fe2s2`, its entries calibrate `--large fe4s4`.
    Every fe4s4 program projects from its family's entries at the small systems and at fe2s2: an
    extrapolated one through the fe2s2 point, a bounded one as the family's largest time and peak
    over all of them, so an expensive fe2s2 entry of a constant-size block (here a made-up
    3,500 s, 7,000 MB SpinMux) sets the fe4s4 projection although other entries are cheaper."""
    hash_ = "1" * 64
    cache = {}
    for name in (*report.SMALL, "fe2s2"):
        _synthetic_entries(cache, name, hash_)
    cache["fe2s2/SpinMux"][0]["watchdog"] = {"seconds": 3500.0, "peak_mb": 7000.0}
    _, fe2s2_sizes = _compiled_rows("fe2s2")
    rows, sizes = _compiled_rows("fe4s4")
    for row in rows:
        estimate = report.projection(cache, row, hash_, sizes[row.id])
        points = report.calibration(cache, report.family(row), hash_)
        assert estimate["allowed"], (row.id, estimate["reason"])
        assert fe2s2_sizes[row.id] in {g for g, _, _ in points}, row.id
        if estimate["method"] == "bounded":
            assert (estimate["seconds"], estimate["peak_mb"]) == (
                max(s for _, s, _ in points), max(m for _, _, m in points)), row.id
        else:
            assert max(g for g, _, _ in estimate["points"]) == fe2s2_sizes[row.id], row.id
    spin_mux = next(row for row in rows if row.id == "SpinMux")
    assert report.projection(cache, spin_mux, hash_, sizes["SpinMux"])["peak_mb"] == 7000.0


def test_bounded_projection_takes_every_measurement():
    """A target no larger than some measured member of its family projects to the family's
    largest time and largest peak, whatever their gate counts; gate count only decides between
    this and extrapolation. From (100 gates, 3,500 s, 7,000 MB) and (200 gates, 10 s, 100 MB), a
    150-gate target projects 3,500 s and 7,000 MB, not the larger member's 10 s and 100 MB: within
    60 minutes and 7.5 GB, refused under the small systems' 4 GB cap, and refused under 60 minutes
    once the smaller member took 3,700 s."""
    points = [(100, 3500.0, 7000.0), (200, 10.0, 100.0)]
    estimate = provenance.project_compile(points, 150)
    assert (estimate["method"], estimate["seconds"], estimate["peak_mb"]) == (
        "bounded", 3500.0, 7000.0)
    assert estimate["allowed"]
    assert provenance.project_compile(points, 150, cap_mb=report.SMALL_CAP_MB)["reason"] == (
        "the family's largest measurement; 6.8 GB > 3.8 GB")
    slow = provenance.project_compile([(100, 3700.0, 7000.0), (200, 10.0, 100.0)], 150)
    assert not slow["allowed"] and slow["reason"] == (
        "the family's largest measurement; 62 min > 60 min")


def _copy_package(root: Path) -> Path:
    shutil.copytree(REPO / "src" / "dfthc", root / "src" / "dfthc",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "scripts").mkdir()
    shutil.copy(SCRIPT, root / provenance.REPORT_SCRIPT)
    shutil.copy(REPO / provenance.LOCK_FILE, root / provenance.LOCK_FILE)
    return root


def test_cache_needs_matching_code_hash(tmp_path, monkeypatch):
    """code_hash() ignores docstrings, comments and dfthc/testing/ and sees the package's code,
    the report script's code (its program builders) and the lock file (one changed byte, final
    review m-13); a cached entry of another
    code hash is kept as history and never shown as a current measurement, while an entry of the
    current hash is reused without compiling."""
    root = _copy_package(tmp_path / "copy")
    base = provenance.code_hash(root)
    path = root / "src" / "dfthc" / "costs.py"
    text = path.read_text()
    path.write_text(text.replace('"""Toffoli count ``T`` and explicit measurement count',
                                 '"""Toffoli count ``T`` and the explicit measurement count')
                    .replace("# --- building blocks", "# --- the building blocks"))
    assert path.read_text() != text and provenance.code_hash(root) == base
    (root / "src" / "dfthc" / "testing" / "fixtures.py").write_text("X = 1\n")
    assert provenance.code_hash(root) == base
    script = root / provenance.REPORT_SCRIPT
    source = script.read_text()
    script.write_text(source.replace('"""The walk or controlled walk with its phase gradient',
                                     '"""The walk, or the controlled walk, with its phase gradient')
                      .replace('sys.path.insert(0, str(REPO / "src"))',
                               'sys.path.insert(0, str(REPO / "src"))  # the package'))
    assert script.read_text() != source and provenance.code_hash(root) == base
    script.write_text(source.replace("return qfunc, s.layout.num_wires + int(controlled)",
                                     "return qfunc, s.layout.num_wires + 1"))
    assert provenance.code_hash(root) != base
    script.write_text(source)
    lock = root / provenance.LOCK_FILE
    pinned = lock.read_bytes()
    lock.write_bytes(bytes([pinned[0] ^ 1]) + pinned[1:])
    assert provenance.code_hash(root) != base
    lock.write_bytes(pinned)
    path.write_text(text.replace("return 2**ob - 2 if ob >= 1 else 0",
                                 "return 2**ob - 3 if ob >= 1 else 0"))
    assert provenance.code_hash(root) != base

    out = tmp_path / "out"
    out.mkdir()
    measured = {"toffoli": 2, "measurements": 0, "measured_and_uncomputes": 0,
                "counts": {"Toffoli": 2}, "num_wires": 5, "seconds": 1.0,
                "watchdog": {"seconds": 3.0, "peak_mb": 200.0}, "gates": 10}
    stale = {"code_hash": "0" * 64, "environment": {}, "status": "compiled"} | measured
    (out / "resources.json").write_text(json.dumps(
        {"compiled": {"H2_STO3G_TEST/SpinSwap": [stale]}}))
    monkeypatch.setattr(report, "measure", lambda *args: pytest.fail("nothing is compiled"))
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "--out", str(out)]) == 0
    record = json.loads((out / "resources.json").read_text())
    row = next(r for r in record["systems"]["H2_STO3G_TEST"]["rows"] if r["id"] == "SpinSwap")
    assert (row["status"], row["measured"]) == ("pending", None)
    assert record["compiled"]["H2_STO3G_TEST/SpinSwap"] == [stale]
    assert "kept as history, never shown as current measurements: 1." in (
        out / "resources.md").read_text()

    fresh = stale | {"code_hash": provenance.code_hash()}
    record["compiled"]["H2_STO3G_TEST/SpinSwap"].append(fresh)
    (out / "resources.json").write_text(json.dumps(record))
    assert report.main(["--systems", "H2_STO3G_TEST", "--rows", "SpinSwap", "--out",
                        str(out)]) == 0
    record = json.loads((out / "resources.json").read_text())
    row = next(r for r in record["systems"]["H2_STO3G_TEST"]["rows"] if r["id"] == "SpinSwap")
    assert (row["status"], _counts(row, "measured")) == ("measured", (2, 0, 0))


BE_DFTHCBE = {"cmp-free": -93, "rot-add": 12, "exact-iter": 71, "no-prune": 2, "flat-r": 120,
              "mux": -4, "sf-enable": 2, "onehot-pad": 40}
BE_PAPER = {"cmp-free": -93, "rot-add": 12, "unif-odd": 6, "exact-iter": 71, "iter-const": -26,
            "onehot-const": -8, "flat-r": 120, "sign-bits": 60, "no-prune": 20, "prefix-stops": 16,
            "mux": 4, "rz-addsub": -424, "maj-and": -4, "sf-enable": 2, "onehot-pad": 40,
            "ref-unif": -1}
"""The FeMoco-54 block encoding's contributions (the plan's decomposition table)."""


@pytest.fixture(scope="module")
def femoco_report(tmp_path_factory):
    """The report's FeMoco-54 record and markdown without compiling (`main --rows`)."""
    out = tmp_path_factory.mktemp("femoco")
    assert report.main(["--systems", "femoco54", "--rows", "--out", str(out)]) == 0
    return (json.loads((out / "resources.json").read_text())["systems"]["femoco54"],
            (out / "resources.md").read_text())


def test_reason_codes_at_femoco(femoco_report):
    """The reason codes of the angle loads and erase, the reflections, block encodings, walks and
    walk powers at FeMoco-54, against dfthcbe and the paper, are exactly the causes with a nonzero
    contribution, each with its contribution (the plan's decomposition table; nothing is
    compiled). The angle rows split exact against truncated iteration (`exact-iter`) from no
    aligned-block pruning (`no-prune`) at the intermediate of `iteration_counts`: the angle
    erase's +7 against dfthcbe is +2 over `r` and +5 over the tail rows (ledger row 2E.2). The
    tail walks reach the paper through dfthcbe's: its `s(RC)` = 4 stops on the dead SF prefix
    (`prefix-stops`) and its truncated iteration's -2 (`iter-const`). The QPE control is a cause
    only where our program lacks it, and the controlled REF_T2 shows the two causes that cancel
    against the paper."""
    rows = {r["id"]: r for r in femoco_report[0]["rows"]}
    cwalk = BE_PAPER | {"ctrl-ref-t2": 1, "ref-target": -1}
    expected = {
        "TailAngleLoad": ({"no-prune": 5}, {"no-prune": 5, "prefix-stops": 4, "iter-const": -2}),
        "SFAngleLoad": ({"exact-iter": 22, "no-prune": -2}, {"exact-iter": 22, "iter-const": -1}),
        "AngleErase": ({"exact-iter": 2, "no-prune": 5, "onehot-pad": 4, "sf-enable": 1},
                       {"exact-iter": 2, "no-prune": 5, "prefix-stops": 4, "iter-const": -4,
                        "onehot-pad": 4, "onehot-const": -1, "sf-enable": 1, "attrib": 54}),
        "RefT2": ({}, {"ref-unif": -1}),
        "ControlledRefT2": ({}, {"ref-unif": -1, "ctrl-ref-t2": 1}),
        "RefAB": ({"ref-target": -1}, {"ref-target": -1, "qpe-ctrl": -1}),
        "ControlledRefAB": ({}, {"ref-target": -1}),
        "DFTHCBlockEncoding": (BE_DFTHCBE, BE_PAPER),
        "ControlledDFTHCBlockEncoding": (BE_DFTHCBE, BE_PAPER | {"ctrl-ref-t2": 1}),
        "DFTHCWalk": (BE_DFTHCBE | {"ref-target": -1}, BE_PAPER | {"ref-target": -1,
                                                                    "qpe-ctrl": -1}),
        "ControlledDFTHCWalk": (BE_DFTHCBE, cwalk),
        "ControlledWalkPower(p=1)": ({}, cwalk),
        "ControlledWalkPower(p=2)": ({}, {code: 2 * value for code, value in cwalk.items()}),
    }
    assert {row_id: (rows[row_id]["codes_dfthcbe"], rows[row_id]["codes_paper"])
            for row_id in expected} == expected
    assert [rows[row_id]["delta_paper"] for row_id in ("DFTHCWalk", "ControlledDFTHCWalk",
                                                       "ControlledWalkPower(p=2)")] == [
        -207, -205, -410]


def test_qpe_summary_and_wire_boundaries(femoco_report):
    """The FeMoco-54 QPE summary gives T and A from the controlled walk and M from the walk,
    A = (2^t - 1) A_C(W) + (t^3 - t)/6 (the phase4-qpe plan's interface), in its record, its
    basis sentence and its table; and the QPE wires step by two where the phase register grows a
    spare wire: `H2_STO3G_TEST` 59, 60, 62 at t = 3, 4, 5 and FeMoco-54 1,124, 1,125, 1,127 at
    t = 21, 22, 23 (nothing is compiled)."""
    record, markdown = femoco_report
    cwalk = next(r for r in record["rows"] if r["id"] == "ControlledDFTHCWalk")["closed"]
    walk = next(r for r in record["rows"] if r["id"] == "DFTHCWalk")["closed"]
    assert cwalk["measured_and_uncomputes"] == 8674
    for bits in (1, 2, 3, 10, 20, 23):
        qpe = record["qpe"][str(bits)]
        assert (qpe["toffoli"], qpe["measurements"], qpe["measured_and_uncomputes"]) == (
            (2**bits - 1) * cwalk["toffoli"] + (bits**3 - bits) // 6,
            (2**bits - 1) * walk["measurements"] + bits,
            (2**bits - 1) * cwalk["measured_and_uncomputes"] + (bits**3 - bits) // 6)
    assert [record["qpe"][str(bits)]["measured_and_uncomputes"] for bits in (1, 2, 3)] == [
        8674, 26023, 60722]
    assert [record["qpe"][str(bits)]["wires"] for bits in (20, 23)] == [1123, 1127]
    assert "A = (2^t - 1) A_C(W) + (t^3 - t)/6" in record["qpe_basis"]
    assert record["qpe_basis"].endswith("a compile with `--large femoco54` replaces them by the "
                                        "compiled T_C(W), M_W and A_C(W).")
    assert "Task" not in record["qpe_basis"]
    assert "| t | T | M | A | Qubits |" in markdown
    h2, femoco = report.build_system("H2_STO3G_TEST"), report.build_system("femoco54")
    assert [report.qpe_wires(h2, bits) for bits in (3, 4, 5)] == [59, 60, 62]
    assert [report.qpe_wires(femoco, bits) for bits in (21, 22, 23)] == [1124, 1125, 1127]


def test_qpe_summary_takes_m_from_the_walk(tmp_path):
    """Final review m-15: the FeMoco-54 QPE summary's M is (2^t - 1) M_W + t with the walk's M
    (spec 9.24 and 11.3), T and A from the controlled walk. A cached controlled walk whose M
    differs from the walk's (2,525 against the closed 2,524) leaves the summary's M on the walk's,
    and its basis sentence says so, naming no task: the pair is recompiled per code hash (nothing
    is compiled; the cached M fails the check, exit 1)."""
    entry = {"code_hash": provenance.code_hash(), "status": "compiled", "toffoli": 9792,
             "measurements": 2525, "measured_and_uncomputes": 8674, "num_wires": 1103,
             "counts": {"Toffoli": 9792 - 8674, "TemporaryAND": 8674,
                        "Adjoint(TemporaryAND)": 8674, "MidCircuitMeasure": 2525},
             "seconds": 1.0, "watchdog": {"peak_mb": 1.0}}
    (tmp_path / "resources.json").write_text(json.dumps(
        {"compiled": {"femoco54/ControlledDFTHCWalk": [entry]}}))
    assert report.main(["--systems", "femoco54", "--rows", "--out", str(tmp_path)]) == 1
    record = json.loads((tmp_path / "resources.json").read_text())["systems"]["femoco54"]
    for bits in (1, 2, 3, 23):
        qpe = record["qpe"][str(bits)]
        assert (qpe["toffoli"], qpe["measurements"], qpe["measured_and_uncomputes"]) == (
            (2**bits - 1) * 9792 + (bits**3 - bits) // 6, (2**bits - 1) * 2524 + bits,
            (2**bits - 1) * 8674 + (bits**3 - bits) // 6), bits
    assert record["qpe_basis"].startswith("T and A from the compiled controlled walk, M from the "
                                          "walk (`--large femoco54`, once each per code hash): ")
    assert "Task" not in record["qpe_basis"]


def test_femoco_summary_says_how_the_walks_are_compiled(femoco_report, tmp_path):
    """Final review m-22: the FeMoco-54 summary states in the present tense how its walk and
    controlled walk rows are measured, with no stale "until then": without a compile under the
    current code hash they show their closed form; with both compiled (entries injected here,
    nothing is compiled) their measured rows equal their closed forms."""
    _, markdown = femoco_report
    assert "until then" not in markdown
    assert ("The walk and the controlled walk are compiled only with `--large femoco54`; without "
            "a compile under the current code hash their rows above show their closed form.") in (
        markdown)
    entries = {}
    for program, (toffoli, measurements, uncomputes, wires) in {
            "DFTHCWalk": (9790, 2524, 8672, 1102),
            "ControlledDFTHCWalk": (9792, 2524, 8674, 1103)}.items():
        entries[f"femoco54/{program}"] = [{
            "code_hash": provenance.code_hash(), "status": "compiled", "toffoli": toffoli,
            "measurements": measurements, "measured_and_uncomputes": uncomputes,
            "num_wires": wires, "seconds": 1.0, "watchdog": {"peak_mb": 1.0},
            "counts": {"Toffoli": toffoli - uncomputes, "TemporaryAND": uncomputes,
                       "Adjoint(TemporaryAND)": uncomputes, "MidCircuitMeasure": measurements}}]
    (tmp_path / "resources.json").write_text(json.dumps({"compiled": entries}))
    assert report.main(["--systems", "femoco54", "--rows", "--out", str(tmp_path)]) == 0
    measured = (tmp_path / "resources.md").read_text()
    assert "until then" not in measured
    assert ("The walk and the controlled walk are compiled with `--large femoco54`, once each "
            "per code hash; their measured rows above equal their closed forms.") in measured


def test_qubit_summary_at_femoco(femoco_report):
    """The FeMoco-54 qubit section (spec 10.3, 13.2) after Q-1 variant (b): WalkLayout's 1,102
    wires by group against the paper's Table 3 groups, 1,131 in all (-29: outer +21, inner +12,
    select +4, work -66); the junk registers, 72 and 390 wires (462 in the record), both alias
    the start of the angle register, so together they take its first 390 wires (final review
    m-22), and their 852 measurements per walk are the block encoding's own; the early junk
    measurement is no longer an open question or a deferred refinement (nothing is compiled)."""
    record, markdown = femoco_report
    assert ("The junk registers (72 and 390 wires, aliased onto the first 390 angle wires) are "
            "X-measured right after each prepare") in markdown
    assert "462 wires" not in markdown
    assert record["wires"] == 1102 and record["qubit_groups"] == {
        "system": 108, "outer": 71, "inner": 65, "select": 799, "phase_grad": 15, "work": 44}
    assert (record["junk_wires"], record["junk_measurements"]) == (462, 852)
    paper = costs_reference.PAPER_QUBIT_GROUPS_FEMOCO
    assert sum(paper.values()) == costs_reference.PAPER_QUBITS_FEMOCO == 1131
    assert {g: record["qubit_groups"][g] - paper[g] for g in paper} == {
        "system": 0, "outer": 21, "inner": 12, "select": 4, "phase_grad": 0, "work": -66}
    assert "| **total** | **1,102** | **1,131** | **-29** |" in markdown
    assert "852 of the M are the block encoding's own junk measurements" in markdown
    assert "Q-1, open" not in markdown and "Early junk measurement" not in markdown
    assert "`junk-qubits`" not in markdown


def test_dfthcbe_blocks_equal_the_call_graph():
    """costs_reference.dfthcbe_blocks equals dfthcbe's own call graph (Qualtran, .venv-ref;
    scripts/make_report_fixture.py) at every parameter set of the report, and the 12 FeMoco-54
    block counts of Task 1A.6's fixture. The call graph's tail walk is its N leaves and its s(RC)
    stops on the dead SF prefix, truncated (-2), which is how iteration_causes takes the tail to
    the paper; and the iteration intermediate is dfthcbe's walk where it is a valid circuit. The
    fixture records the dfthcbe tree it was evaluated on: the source URL and digest and the
    installed package's digest of Task 1A.6's provenance, which tests/data/test_dfthcbe_fixtures.py
    pins to SOURCE_SHA256 and INSTALLED_SHA256 (final review m-16)."""
    fixture = load_dfthcbe_fixture("report_toffolis.json")
    provenance_ = load_dfthcbe_fixture("provenance.json")
    digests = ("dfthcbe_source_url", "dfthcbe_source_sha256", "dfthcbe_installed_sha256")
    assert {key: fixture.get(key) for key in digests} == {key: provenance_[key] for key in digests}
    for name, values in fixture["toffolis"].items():
        params = WalkParams(**fixture["parameters"][name])
        assert params == (FEMOCO54 if name == "femoco54" else report.build_system(name).params)
        port = costs_reference.dfthcbe_blocks(params)
        assert {key: port[key] for key in values} == values, name
        tail = costs_reference.iteration_causes(params)["tail"][1]
        assert (tail["prefix-stops"], tail["iter-const"]) == (
            costs_reference.popcount(params.RC), -2), name
        assert values["tail_angle_load"] == params.N + tail["prefix-stops"] + tail["iter-const"]
        counts = costs_reference.iteration_counts(params)
        assert all(counts[key][1] == counts[key][2]
                   for key in ("outer_read", "outer_erase", "sf_fixup")), name
    femoco = load_dfthcbe_fixture("femoco_block_toffolis.json")
    port = costs_reference.dfthcbe_blocks(FEMOCO54)
    ours = {"inner_select_adjoint": "inner_select_dagger"}
    assert {key: port[ours.get(key, key)] for key in femoco} == femoco


def test_paper_table3_reproduces_table_iii():
    """paper_table3 gives Table III's FeMoco-54 values per call (spec 13.1): 9,997 per walk,
    with the tail fix-up under the inner erase (Table III) or at step 18."""
    paper = costs_reference.paper_table3(FEMOCO54)
    assert {key: paper[key] for key in (
        "uniform_x_o", "outer_prepare", "outer_unprepare", "uniform_b", "inner_prepare",
        "inner_unprepare", "inner_unprepare_step18", "tail_angle_load", "sf_angle_load",
        "spin_mux", "spin_mux_app_b3", "spin_swap", "givens_ladder", "majorana_gate",
        "angle_erase", "angle_erase_step18", "inner_select", "inner_select_step18", "ref_t2",
        "ref_ab", "outer_erase", "inner_erase", "block_encoding", "walk")} == {
        "uniform_x_o": 36, "outer_prepare": 213, "outer_unprepare": 97, "uniform_b": 20,
        "inner_prepare": 880, "inner_unprepare": 274, "inner_unprepare_step18": 220,
        "tail_angle_load": 54, "sf_angle_load": 270, "spin_mux": 0, "spin_mux_app_b3": 1,
        "spin_swap": 54, "givens_ladder": 1590, "majorana_gate": 7, "angle_erase": 37,
        "angle_erase_step18": 91, "inner_select": 3656, "inner_select_step18": 3710,
        "ref_t2": 21, "ref_ab": 46, "outer_erase": 37, "inner_erase": 180,
        "block_encoding": 9951, "walk": 9997}


def test_project_compile_is_the_k1_method():
    """Task 0.9's projection beyond the measured sizes: a power law in time through the two
    largest points (exponent at least 1) and a least-squares line in memory (slope at least 0);
    within them, the largest of all the measurements; refused above 60 min or above
    the 7.5 GB kill point of the 8 GB cap, so a 7.8 GB projection is refused."""
    points = [(1000, 10.0, 500.0), (2000, 20.0, 600.0), (4000, 80.0, 700.0)]
    result = provenance.project_compile(points, 8000)
    slope, intercept = np.polyfit([1000, 2000, 4000], [500, 600, 700], 1)
    assert result["allowed"] and result["time_exponent"] == 2.0
    assert result["seconds"] == pytest.approx(320.0, rel=0, abs=1e-9)
    assert result["peak_mb"] == pytest.approx(intercept + slope * 8000, rel=0, abs=0.05)
    assert provenance.project_compile([(1000, 10.0, 500.0), (2000, 11.0, 500.0)],
                                      4000)["seconds"] == pytest.approx(22.0, rel=0, abs=1e-9)
    refused = provenance.project_compile([(1000, 1.0, 1000.0), (2000, 2.0, 2000.0)], 7800)
    assert refused["peak_mb"] == pytest.approx(7800.0, rel=0, abs=0.05)
    assert not refused["allowed"] and refused["reason"] == "7.6 GB > 7.5 GB"
    assert not provenance.project_compile([(1000, 1.0, 1.0)], 2000)["allowed"]
    bounded = provenance.project_compile([(50, 9.0, 280.0), (100, 5.0, 300.0),
                                          (200, 4.0, 310.0)], 80)
    assert (bounded["method"], bounded["seconds"], bounded["peak_mb"]) == ("bounded", 9.0, 310.0)
    assert bounded["allowed"] and result["method"] == "k1"
    slow = provenance.project_compile([(100, 4000.0, 300.0)], 50)
    assert not slow["allowed"] and slow["reason"] == (
        "the family's largest measurement; 67 min > 60 min")
