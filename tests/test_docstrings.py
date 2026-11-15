"""The documentation set (spec 2.3 D-6, 5.2) and the operator docstrings (spec 9.0; master plan
Task 4.4).

Every ``Operator2`` subclass of ``dfthc.ops`` and ``dfthc.qpe`` has the nine docstring sections
of spec 9.0 in order, and its cost section names an existing ``dfthc.costs`` function. The
documentation files of spec 5.2 exist, their relative links resolve, the limitations name only
existing canaries and every limitation of spec 14, and the architecture names every operator.
"""

import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import pytest
from pennylane.core import Operator2

import dfthc.ops
from dfthc import costs

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SECTIONS = (
    ("purpose and mathematical action", r"\*\*Action\b"),
    ("register table", r"\*\*Registers\.\*\*"),
    ("compilable arguments", r"\*\*(?:Compilable a|A)rguments\.\*\*"),
    ("pre- and postconditions", r"\*\*Pre- and postconditions\.\*\*"),
    ("decomposition outline", r"\*\*Decomposition\b"),
    ("adjoint and control behaviour", r"\*\*Adjoint and control\.\*\*"),
    ("cost formula", r"\*\*Cost\.\*\*"),
    ("paper equations and the dfthcbe bloq", r"\*\*Paper and dfthcbe\.\*\*"),
    ("a minimal compiled example", r"\*\*Example\.\*\*"),
)
"""Spec 9.0's sections in order, each a bold heading at the start of a docstring line. QROAMRead
and PhaseFixup, which also take dynamic arguments, title section 3 "Arguments"."""
DOCUMENTS = ("architecture.md", "conventions.md", "environment.md", "limitations.md",
             "verification.md", "resources.md", "resources.json", "agents/HANDOFF.md",
             "agents/LEDGER.md")
"""The documentation set of spec 5.2 under ``docs/``."""
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def operator_classes() -> dict[str, type]:
    """Every Operator2 subclass defined in a module of dfthc.ops or in dfthc.qpe."""
    modules = [importlib.import_module(f"dfthc.ops.{info.name}")
               for info in pkgutil.iter_modules(dfthc.ops.__path__)]
    modules.append(importlib.import_module("dfthc.qpe"))
    return {cls.__name__: cls for module in modules
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, Operator2) and cls.__module__ == module.__name__}


CLASSES = operator_classes()


def test_every_operator_is_audited():
    """The 30 operators of dfthc.ops (spec 9.1 to 9.22) and the two of dfthc.qpe (9.23, 11.1)."""
    assert len(CLASSES) == 32
    assert {"PhaseGradientRotation", "QROAMRead", "DFTHCWalk", "ControlledDFTHCWalk",
            "ControlledWalkPower", "QPERound"} <= set(CLASSES)


@pytest.mark.parametrize("name", sorted(CLASSES))
def test_docstring_has_the_nine_sections_in_order(name):
    """Spec 9.0: a summary line, then the nine sections in order, each once; the cost section
    names at least one dfthc.costs function and only existing ones; the example calls
    compiled_resources."""
    doc = inspect.getdoc(CLASSES[name]) or ""
    lines = doc.splitlines()
    assert lines and lines[0].strip(), f"{name} has no summary line"
    starts = []
    for title, pattern in SECTIONS:
        found = [i for i, line in enumerate(lines) if re.match(pattern, line.strip())]
        assert len(found) == 1, f"{name}: section '{title}' found {len(found)} times"
        starts.append(found[0])
    assert starts == sorted(starts), f"{name}: sections out of order"
    cost = "\n".join(lines[starts[6]:starts[7]])
    named = re.findall(r"dfthc\.costs\.(\w+)", cost)
    assert named, f"{name}: the cost section names no dfthc.costs function"
    missing = [function for function in named if not callable(getattr(costs, function, None))]
    assert not missing, f"{name}: dfthc.costs has no {missing}"
    assert "compiled_resources(" in "\n".join(lines[starts[8]:]), f"{name}: no compiled example"


def test_the_documentation_set_exists():
    """Spec 5.2's documentation files exist (D-6)."""
    missing = [name for name in DOCUMENTS if not (DOCS / name).is_file()]
    assert not missing


@pytest.mark.parametrize("document", sorted(p.name for p in DOCS.glob("*.md")))
def test_documentation_links_resolve(document):
    """Every relative link of a top-level document points to an existing file (anchors are
    stripped; web links are not followed)."""
    path = DOCS / document
    broken = []
    for target in LINK.findall(path.read_text()):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        if not (path.parent / target.split("#", 1)[0]).exists():
            broken.append(target)
    assert not broken, f"{document}: {broken}"


def test_limitations_match_the_canaries_and_spec_14():
    """docs/limitations.md has an entry for every limitation L-1 to L-27 of spec 14, and every
    canary it names is a test function in tests/canary."""
    text = (DOCS / "limitations.md").read_text()
    assert sorted({int(n) for n in re.findall(r"\*\*L-(\d+)\b", text)}) == list(range(1, 28))
    canaries = set()
    for path in (REPO / "tests" / "canary").glob("test_*.py"):
        canaries |= set(re.findall(r"^def (test_\w+)", path.read_text(), re.M))
    named = set(re.findall(r"`(test_\w+)`", text))
    assert named and named <= canaries, sorted(named - canaries)


def test_architecture_names_every_operator():
    """docs/architecture.md names every audited operator class and every dfthc module."""
    text = (DOCS / "architecture.md").read_text()
    assert [name for name in CLASSES if f"`{name}`" not in text] == []
    modules = [path.relative_to(REPO / "src").with_suffix("").as_posix().replace("/", ".")
               for path in (REPO / "src" / "dfthc").rglob("*.py") if path.name != "__init__.py"]
    assert [module for module in modules if f"`{module}`" not in text] == []
