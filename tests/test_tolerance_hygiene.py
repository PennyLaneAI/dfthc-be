"""Every numerical comparison in the tests states both tolerances (HANDOFF section 6).

The rule, checked on the syntax tree of every ``.py`` file under ``tests/`` and
``src/dfthc/testing/``: a call is an offender unless it names both tolerance keywords itself.

| Call (any receiver, or a bare imported name)                         | Required keywords    |
| -------------------------------------------------------------------- | -------------------- |
| ``allclose``, ``isclose`` (``np.``, ``numpy.``), ``assert_allclose`` | ``rtol`` and ``atol``  |
| ``approx`` (``pytest.approx``)                                       | ``rel`` and ``abs``    |
| ``isclose`` on the receiver ``math``                                 | ``rel_tol``, ``abs_tol`` |

The check is purely syntactic and deliberately simple: tolerances passed positionally do not
count, and a ``**kwargs`` forward never counts as naming a keyword, so a helper that wraps one of
these calls must spell out both keywords in the call itself. NumPy's defaults (``rtol=1e-5`` for
``allclose`` and ``isclose``, ``atol=1e-8`` for ``allclose``) and ``pytest.approx``'s default
``abs=1e-12`` silently loosen or change a stated contract, so no call may rely on one.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = (REPO / "tests", REPO / "src" / "dfthc" / "testing")

_NUMPY_STYLE = frozenset({"rtol", "atol"})
_REQUIRED = {
    "allclose": _NUMPY_STYLE,
    "assert_allclose": _NUMPY_STYLE,
    "isclose": _NUMPY_STYLE,
    "approx": frozenset({"rel", "abs"}),
}
_MATH_ISCLOSE = frozenset({"rel_tol", "abs_tol"})


def _callee(func: ast.expr) -> tuple[str | None, str | None]:
    """Return ``(receiver name, function name)`` of a call target, or ``(None, None)``."""
    if isinstance(func, ast.Name):
        return None, func.id
    if isinstance(func, ast.Attribute):
        receiver = func.value
        return (receiver.id if isinstance(receiver, ast.Name) else None), func.attr
    return None, None


def find_offenders(roots=ROOTS, base=REPO) -> list[str]:
    """List ``file:line: call`` (paths relative to ``base``) for each call missing a keyword."""
    offenders = []
    for root in roots:
        for path in sorted(Path(root).rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            calls = sorted(
                (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
                key=lambda node: (node.lineno, node.col_offset),
            )
            for node in calls:
                receiver, name = _callee(node.func)
                if name not in _REQUIRED:
                    continue
                required = _REQUIRED[name]
                if name == "isclose" and receiver == "math":
                    required = _MATH_ISCLOSE
                named = {keyword.arg for keyword in node.keywords if keyword.arg is not None}
                missing = sorted(required - named)
                if missing:
                    location = f"{path.relative_to(base)}:{node.lineno}"
                    offenders.append(f"{location}: {name}() lacks {', '.join(missing)}")
    return offenders


def test_every_numerical_comparison_states_both_tolerances():
    for root in ROOTS:
        assert root.is_dir(), f"guard root missing: {root}"
    offenders = find_offenders()
    assert not offenders, "comparisons without both tolerances:\n" + "\n".join(offenders)
