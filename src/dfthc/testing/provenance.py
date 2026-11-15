"""Code hash, environment record and compile projection for cached compiles (spec 13.3; master
plan Task 4.3 and section 12.9).

:func:`code_hash` decides whether a compile cached in ``docs/resources.json`` is a current
measurement: it hashes the syntax trees, without docstrings (comments are not in a syntax tree),
of the package's modules and of ``scripts/resource_report.py``, which builds the cached programs
and selects the cache's entries, and the lock file. So a documentation edit keeps the hash and any
code or toolchain change gives a new one. ``dfthc/testing/`` is excluded: no compiled program
depends on it. :func:`project_compile` is Task 0.9's K-1 projection method, replaced by the
largest of the measurements where the target is no larger than a measured point.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import math
import platform
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dfthc.testing.watchdog import MEMORY_CAP_MB, effective_limit_mb, within_effective_limit

REPO = Path(__file__).resolve().parents[3]
LOCK_FILE = "requirements-nightly.lock"
REPORT_SCRIPT = "scripts/resource_report.py"
DISTRIBUTIONS = ("pennylane", "pennylane_catalyst", "pennylane_lightning", "jax", "jaxlib", "numpy")
TIME_LIMIT_S = 3600.0
"""The compile time a large program may be projected to (60 min, master plan 12.9)."""


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove the docstring of the module and of every class and function."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def hashed_files(root: Path = REPO) -> list[Path]:
    """The modules :func:`code_hash` covers: ``src/dfthc/**/*.py`` without ``dfthc/testing/``,
    then the report script."""
    package = root / "src" / "dfthc"
    return sorted(path for path in package.rglob("*.py")
                  if "testing" not in path.relative_to(package).parts) + [root / REPORT_SCRIPT]


def code_hash(root: Path = REPO) -> str:
    """SHA-256 over the docstring-free syntax trees of :func:`hashed_files` (each preceded by its
    path relative to ``root``) and the bytes of the lock file."""
    digest = hashlib.sha256()
    for path in hashed_files(root):
        tree = _strip_docstrings(ast.parse(path.read_text(), filename=str(path)))
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(ast.dump(tree, annotate_fields=True, include_attributes=False).encode())
    digest.update((root / LOCK_FILE).read_bytes())
    return digest.hexdigest()


def environment(root: Path = REPO) -> dict[str, str]:
    """Python, platform, the toolchain's package versions and the lock file's SHA-256."""
    record = {"python": platform.python_version(), "platform": platform.platform()}
    for name in DISTRIBUTIONS:
        try:
            record[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record[name] = "not installed"
    lock = root / LOCK_FILE
    record["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() \
        else "missing"
    return record


def _over(seconds: float, peak_mb: float, cap_mb: float, time_limit_s: float) -> list[str]:
    """Why a projection exceeds the limits, if it does."""
    reasons = []
    if seconds > time_limit_s:
        reasons.append(f"{seconds / 60:.0f} min > {time_limit_s / 60:.0f} min")
    if not within_effective_limit(peak_mb, cap_mb):
        reasons.append(f"{peak_mb / 1024:.1f} GB > {effective_limit_mb(cap_mb) / 1024:.1f} GB")
    return reasons


def project_compile(points: Sequence[tuple[int, float, float]], gates: int, *,
                    cap_mb: float = MEMORY_CAP_MB, time_limit_s: float = TIME_LIMIT_S) -> dict:
    """Project one program to ``gates`` gates from measured ``(gates, seconds, peak_mb)`` points
    of the same program family at other sizes.

    If some point has at least ``gates`` gates (a block whose size does not grow, such as the spin
    multiplexer), the projection is the largest time and the largest peak over all points,
    whatever their gate counts (``method: "bounded"``): gate count is not a monotone measure of
    compile cost, and it only decides between this and extrapolation. Otherwise it extrapolates
    by Task 0.9's K-1 method (``method: "k1"``): time as a power law through the two points with
    the most gates, exponent at least 1; memory as a least-squares line through all points, slope
    at least 0; fewer than two points with distinct gate counts give ``allowed: False`` and no
    projection. ``allowed`` is true only if time and memory stay within ``time_limit_s`` and
    :func:`~dfthc.testing.watchdog.effective_limit_mb` of ``cap_mb`` (7,680 MB under the 8 GB
    cap).
    """
    usable = sorted({(int(g), float(s), float(m)) for g, s, m in points})
    limits = {"seconds": time_limit_s, "mb": effective_limit_mb(cap_mb), "cap_mb": cap_mb}
    if any(g >= gates for g, _, _ in usable):
        seconds, peak_mb = max(s for _, s, _ in usable), max(m for _, _, m in usable)
        reasons = _over(seconds, peak_mb, cap_mb, time_limit_s)
        return {"allowed": not reasons,
                "reason": "; ".join(["the family's largest measurement", *reasons]),
                "method": "bounded",
                "gates": gates, "seconds": seconds, "peak_mb": peak_mb, "limits": limits,
                "points": [list(point) for point in usable]}
    if len({g for g, _, _ in usable}) < 2:
        return {"allowed": False, "reason": "fewer than two calibration points", "gates": gates,
                "points": [list(point) for point in usable]}
    (g1, s1, _), (g2, s2, _) = sorted({g: (g, s, m) for g, s, m in usable}.values())[-2:]
    exponent = max(1.0, math.log(s2 / s1) / math.log(g2 / g1)) if s1 > 0 and s2 > 0 else 1.0
    seconds = s2 * (gates / g2) ** exponent
    xs = np.array([g for g, _, _ in usable], float)
    ys = np.array([m for _, _, m in usable], float)
    slope = max(0.0, float(np.polyfit(xs, ys, 1)[0]))
    peak_mb = float(ys.mean() - slope * xs.mean()) + slope * gates
    reasons = _over(seconds, peak_mb, cap_mb, time_limit_s)
    return {"allowed": not reasons, "reason": "; ".join(reasons) or "within the limits",
            "method": "k1", "gates": gates, "seconds": round(seconds, 1),
            "time_exponent": round(exponent, 2), "peak_mb": round(peak_mb, 1),
            "mb_per_1000_gates": round(1000 * slope, 3), "limits": limits,
            "points": [list(point) for point in usable]}
