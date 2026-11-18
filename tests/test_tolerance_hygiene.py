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

"""Every numerical comparison in the tests states both tolerances (HANDOFF section 6).

The rule, checked on the syntax tree of every ``.py`` file under ``tests/`` and
``src/dfthc/testing/``: a call to one of the comparison functions below is an offender unless it
names both tolerance keywords itself.

| Canonical function                                  | Required keywords        |
| --------------------------------------------------- | ------------------------ |
| ``numpy.allclose``, ``numpy.isclose``               | ``rtol`` and ``atol``    |
| ``numpy.testing.assert_allclose``                   | ``rtol`` and ``atol``    |
| ``pytest.approx``                                   | ``rel`` and ``abs``      |
| ``math.isclose``                                    | ``rel_tol`` and ``abs_tol`` |

Calls are identified by import provenance, not by name. Per file, a symbol table is built from
every ``import`` and ``from ... import`` statement (with aliases, at any nesting level), and each
call target (a bare name or an attribute chain such as ``np.testing.assert_allclose``) is
resolved through it to a canonical dotted name. So ``from numpy import allclose as a``,
``import math as m`` and ``import numpy as xp`` are all recognized, while a method that merely
shares a name (``DomainObject().approx(...)``) or a receiver that is not one of these modules is
ignored. The table is flat per file: a local variable that shadows an imported name is not
tracked, and relative imports are ignored.

Tolerances passed positionally do not count, and a ``**kwargs`` forward never counts as naming a
keyword, so a helper that wraps one of these calls must spell out both keywords in the call
itself. NumPy's defaults (``rtol=1e-5`` for ``allclose`` and ``isclose``, ``atol=1e-8`` for
``allclose``) and ``pytest.approx``'s default ``abs=1e-12`` silently loosen or change a stated
contract, so no call may rely on one.
"""

import ast
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = (REPO / "tests", REPO / "src" / "dfthc" / "testing")

_NUMPY_STYLE = frozenset({"rtol", "atol"})
_REQUIRED = {
    "numpy.allclose": _NUMPY_STYLE,
    "numpy.isclose": _NUMPY_STYLE,
    "numpy.testing.assert_allclose": _NUMPY_STYLE,
    "pytest.approx": frozenset({"rel", "abs"}),
    "math.isclose": frozenset({"rel_tol", "abs_tol"}),
}


def _symbol_table(tree: ast.AST) -> dict[str, str]:
    """Map every name bound by an import statement of the file to its dotted origin."""
    table = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    table[alias.asname] = alias.name
                else:  # ``import numpy.testing`` binds ``numpy``
                    table[alias.name.split(".")[0]] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                table[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return table


def _canonical(func: ast.expr, table: dict[str, str]) -> str | None:
    """Dotted origin of a call target, or ``None`` if its root is not an imported name."""
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if not isinstance(func, ast.Name) or func.id not in table:
        return None
    return ".".join([table[func.id], *reversed(parts)])


def offenders_in_source(source: str, filename: str = "<source>") -> list[str]:
    """List ``line: function() lacks <keywords>`` for each call missing a tolerance keyword."""
    tree = ast.parse(source, filename=filename)
    table = _symbol_table(tree)
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    offenders = []
    for node in calls:
        canonical = _canonical(node.func, table)
        if canonical not in _REQUIRED:
            continue
        named = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
        missing = sorted(_REQUIRED[canonical] - named)
        if missing:
            offenders.append(f"{node.lineno}: {canonical}() lacks {', '.join(missing)}")
    return offenders


def find_offenders(roots=ROOTS, base=REPO) -> list[str]:
    """List ``file:line: call`` (paths relative to ``base``) for each call missing a keyword."""
    offenders = []
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            for offender in offenders_in_source(path.read_text(), str(path)):
                offenders.append(f"{path.relative_to(base)}:{offender}")
    return offenders


def test_every_numerical_comparison_states_both_tolerances():
    for root in ROOTS:
        assert root.is_dir(), f"guard root missing: {root}"
    offenders = find_offenders()
    assert not offenders, "comparisons without both tolerances:\n" + "\n".join(offenders)


SYNTHETIC = textwrap.dedent(
    """
    import math
    import math as m
    import numpy as np
    import numpy as xp
    import numpy.testing as npt
    import pytest
    from math import isclose as c
    from numpy import allclose as a
    from numpy import isclose
    from numpy.testing import assert_allclose as aac
    from pytest import approx


    class DomainObject:
        def approx(self, *args):
            return args

        def allclose(self, *args):
            return args


    np.allclose(1, 1)  # numpy.allclose() lacks atol, rtol
    np.allclose(1, 1, rtol=0)  # numpy.allclose() lacks atol
    np.allclose(1, 1, rtol=0, atol=0)
    np.allclose(1, 1, 0, 1e-8)  # numpy.allclose() lacks atol, rtol
    np.allclose(1, 1, **{"rtol": 0, "atol": 0})  # numpy.allclose() lacks atol, rtol
    np.isclose(1, 1, atol=0)  # numpy.isclose() lacks rtol
    isclose(1, 1)  # numpy.isclose() lacks atol, rtol
    isclose(1, 1, rtol=0, atol=0)
    xp.allclose(1, 1, atol=1e-8)  # numpy.allclose() lacks rtol
    a(1, 1)  # numpy.allclose() lacks atol, rtol
    a(1, 1, rtol=0, atol=0)
    np.testing.assert_allclose(1, 1)  # numpy.testing.assert_allclose() lacks atol, rtol
    np.testing.assert_allclose(1, 1, rtol=0, atol=0)
    npt.assert_allclose(1, 1, rtol=0)  # numpy.testing.assert_allclose() lacks atol
    aac(1, 1, atol=0)  # numpy.testing.assert_allclose() lacks rtol
    aac(1, 1, rtol=0, atol=0)
    pytest.approx(1)  # pytest.approx() lacks abs, rel
    pytest.approx(1, rel=0)  # pytest.approx() lacks abs
    pytest.approx(1, rel=0, abs=0)
    approx(1, abs=0)  # pytest.approx() lacks rel
    approx(1, rel=0, abs=0)
    math.isclose(1.0, 1.0)  # math.isclose() lacks abs_tol, rel_tol
    math.isclose(1.0, 1.0, rel_tol=0)  # math.isclose() lacks abs_tol
    math.isclose(1.0, 1.0, rel_tol=0, abs_tol=0)
    m.isclose(1.0, 1.0, abs_tol=0)  # math.isclose() lacks rel_tol
    m.isclose(1.0, 1.0, rel_tol=0, abs_tol=0)
    c(1.0, 1.0)  # math.isclose() lacks abs_tol, rel_tol
    c(1.0, 1.0, rel_tol=0, abs_tol=0)
    DomainObject().approx(1)
    DomainObject().allclose(1, 1)
    other.allclose(1, 1)
    unknown(1, 1)
    """
)


def _expected_offenders(source: str) -> list[str]:
    """The offenders a snippet declares with a trailing ``# <function>() lacks ...`` comment."""
    expected = []
    for number, line in enumerate(source.splitlines(), start=1):
        if "()" in line.partition("  # ")[2]:
            expected.append(f"{number}: {line.partition('  # ')[2]}")
    return expected


def test_the_guard_classifies_calls_by_import_provenance():
    """The classifier on synthetic source: aliases, nested modules, positional and ``**kwargs``."""
    expected = _expected_offenders(SYNTHETIC)
    assert len(expected) == 18  # guards the comment parsing itself
    assert offenders_in_source(SYNTHETIC) == expected
