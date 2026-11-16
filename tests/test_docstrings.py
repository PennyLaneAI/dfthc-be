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
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import pytest
from pennylane.core import Operator2

import dfthc.ops
from dfthc import costs
from dfthc.params import FEMOCO54, H2_STO3G_TEST
from dfthc.registers import WalkLayout
from dfthc.testing.toy_walk import ToyControlledWalkPower

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SPEC = DOCS / "superpowers" / "specs" / "2026-11-11-dfthc-pennylane-design.md"
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
"""Spec 9.0's sections in order, each a bold heading at the start of a docstring line. QROAMRead,
PhaseFixup and the two unprepares from outcomes, which also take dynamic arguments, title
section 3 "Arguments"."""
DOCUMENTS = ("architecture.md", "conventions.md", "environment.md", "limitations.md",
             "verification.md", "resources.md", "resources.json", "agents/HANDOFF.md",
             "agents/LEDGER.md")
"""The documentation set of spec 5.2 under ``docs/``."""
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def operator_classes() -> dict[str, type]:
    """Every Operator2 subclass reachable from dfthc.ops (recursively) or dfthc.qpe."""
    modules = [dfthc.ops]
    modules.extend(importlib.import_module(info.name)
                   for info in pkgutil.walk_packages(dfthc.ops.__path__, prefix="dfthc.ops."))
    modules.append(importlib.import_module("dfthc.qpe"))
    classes = {cls for module in modules for _, cls in inspect.getmembers(module, inspect.isclass)
               if cls is not Operator2 and issubclass(cls, Operator2)}
    return {f"{cls.__module__}.{cls.__qualname__}": cls for cls in classes}


CLASSES = operator_classes()


def spec_section(heading: str, next_heading: str) -> str:
    """The spec's text from the line starting with ``heading`` to the one starting with
    ``next_heading``."""
    text = SPEC.read_text()
    start = re.search(rf"^{re.escape(heading)}", text, re.M).start()
    return text[start:re.search(rf"^{re.escape(next_heading)}", text[start:], re.M).start() + start]


def test_every_operator_is_audited():
    """The 32 operators of dfthc.ops (spec 9.1 to 9.22, the two unprepares from outcomes of
    spec 16, Q-1 variant (b), included) and the two of dfthc.qpe (9.23, 11.1)."""
    assert len(CLASSES) == 34
    names = {cls.__name__ for cls in CLASSES.values()}
    assert {"PhaseGradientRotation", "QROAMRead", "DFTHCWalk", "ControlledDFTHCWalk",
            "OuterUnprepareFromOutcomes", "InnerUnprepareFromOutcomes", "ControlledWalkPower",
            "QPERound"} <= names


@pytest.mark.parametrize("qualified_name", sorted(CLASSES))
def test_docstring_has_the_nine_sections_in_order(qualified_name):
    """Spec 9.0: a summary line, then the nine sections in order, each once; the cost section
    names at least one dfthc.costs function and only existing ones; the example calls
    compiled_resources."""
    cls = CLASSES[qualified_name]
    name = cls.__name__
    doc = inspect.getdoc(cls) or ""
    lines = doc.splitlines()
    assert lines and lines[0].strip(), f"{name} has no summary line"
    starts = []
    for title, pattern in SECTIONS:
        found = [i for i, line in enumerate(lines) if re.match(pattern, line.strip())]
        assert len(found) == 1, f"{name}: section '{title}' found {len(found)} times"
        starts.append(found[0])
    assert starts == sorted(starts), f"{name}: sections out of order"
    assert starts[0] > 0, f"{name}: the summary must precede the Action section"
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


def markdown_anchors(text: str) -> set[str]:
    """GitHub-style anchors for the Markdown headings used by this repository."""
    anchors, counts = set(), Counter()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M):
        heading = re.sub(r"`([^`]*)`", r"\1", heading)
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        slug = re.sub(r"[^\w\s-]", "", heading.lower())
        slug = re.sub(r"\s+", "-", slug.strip())
        suffix = counts[slug]
        counts[slug] += 1
        anchors.add(f"{slug}-{suffix}" if suffix else slug)
    return anchors


DOCUMENT_PATHS = tuple(sorted(path.relative_to(DOCS) for path in DOCS.rglob("*.md")
                              if "superpowers" not in path.relative_to(DOCS).parts))


@pytest.mark.parametrize("document", DOCUMENT_PATHS, ids=lambda path: path.as_posix())
def test_documentation_links_resolve(document):
    """Every relative link in every Markdown document resolves, including its anchor."""
    path = DOCS / document
    broken = []
    for target in LINK.findall(path.read_text()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        relative, _, anchor = target.partition("#")
        target_path = path if not relative else path.parent / unquote(relative)
        try:
            target_path.resolve().relative_to(DOCS.resolve())
        except ValueError:
            broken.append(target)
            continue
        if not target_path.is_file():
            broken.append(target)
            continue
        if anchor and unquote(anchor) not in markdown_anchors(target_path.read_text()):
            broken.append(target)
    assert not broken, f"{document}: {broken}"


def test_limitations_match_the_canaries_and_spec_14():
    """docs/limitations.md has an entry for every limitation L-1 to L-27 of spec 14, and every
    canary it names is a test function in tests/canary."""
    text = (DOCS / "limitations.md").read_text()
    numbers = [int(n) for n in re.findall(r"\*\*L-(\d+)\b", text)]
    assert Counter(numbers) == Counter(range(1, 28))
    canaries = set()
    for path in (REPO / "tests" / "canary").glob("test_*.py"):
        canaries |= set(re.findall(r"^def (test_\w+)", path.read_text(), re.M))
    named = set(re.findall(r"`(test_\w+)`", text))
    assert named and named <= canaries, sorted(named - canaries)


def test_every_l6_mitigation_names_the_three_looped_operators():
    """Spec 7.4 rule 4's looped operators, never adjointed or controlled, are the ones that spec
    14's L-6, HANDOFF 7's L-6 row and docs/limitations.md's L-6 entry name, each with "never
    adjointed or controlled" (final review m-23)."""
    classes = {cls.__name__ for cls in CLASSES.values()} | {ToyControlledWalkPower.__name__}
    rule_4 = re.search(r"`qp\.for_loop` appears in three operators.*",
                       spec_section("### 7.4", "### 7.5")).group(0)
    looped = set(re.findall(r"`(\w+)`", rule_4)) & classes
    assert looped == {"QROAMRead", "ControlledWalkPower", "ToyControlledWalkPower"}
    entries = {
        "spec 14": re.search(r"\*\*L-6:.*?(?=\n7\. \*\*L-7)", spec_section("## 14.", "## 15."),
                             re.S).group(0),
        "HANDOFF 7": re.search(r"^\| L-6 \|.*$", (DOCS / "agents" / "HANDOFF.md").read_text(),
                               re.M).group(0),
        "limitations.md": re.search(r"\*\*L-6,.*?(?=\n- \*\*L-7)",
                                    (DOCS / "limitations.md").read_text(), re.S).group(0),
    }
    for where, entry in entries.items():
        assert set(re.findall(r"`(\w+)`", entry)) & classes == looped, where
        assert "never adjointed or controlled" in " ".join(entry.split()), where


def test_spec_14_l23_names_every_operator_with_a_dynamic_argument():
    """Spec 14's L-23 mitigation names exactly the operators that declare dynamic arguments
    and convert lists in ``__init__`` (final review m-24)."""
    entry = re.search(r"\*\*L-23:.*?(?=\n24\. \*\*L-24)", spec_section("## 14.", "## 15."),
                      re.S).group(0)
    dynamic = {cls.__name__ for cls in CLASSES.values() if getattr(cls, "dynamic_argnames", ())}
    assert dynamic == {"QROAMRead", "PhaseFixup", "OuterUnprepareFromOutcomes",
                       "InnerUnprepareFromOutcomes", "QPERound"}
    assert set(re.findall(r"`(\w+)`", entry)) & {c.__name__ for c in CLASSES.values()} == dynamic


def test_architecture_assigns_each_module_to_one_layer():
    """docs/architecture.md's layer table lists every module in one layer only (final review
    m-28), so the page's claim that each layer depends only on lower ones can hold."""
    text = (DOCS / "architecture.md").read_text()
    table = text.split("## Layers", 1)[1].split("\n## ", 1)[0]
    layers = Counter()
    for row in re.findall(r"^\| (?:L\d|Tooling)[^|]*\|[^|]*\|([^|]*)\|$", table, re.M):
        layers.update(re.findall(r"`(dfthc[\w.]*)`", row))
    assert layers and [module for module, count in layers.items() if count > 1] == []


def test_spec_11_3_gives_the_qpe_wire_totals_of_the_layout():
    """Spec 11.3's wire totals of the QPE program (final review I-4) are the walk's wires plus
    the QPE control, read from WalkLayout (no compile), plus t gradient wires: 56 + t at
    H2_STO3G_TEST and 1,103 + t at FeMoco-54."""
    text = spec_section("### 11.3", "### 11.4")

    def total(where: str) -> int:
        match = re.search(rf"\$(\d[\d{{}},]*) \+ t\$ at {where}", text)
        assert match, where
        return int(re.sub(r"\D", "", match.group(1)))

    assert total("`H2_STO3G_TEST`") == WalkLayout.from_params(H2_STO3G_TEST).num_wires + 1
    assert total("FeMoco-54") == WalkLayout.from_params(FEMOCO54).num_wires + 1


def test_architecture_names_every_operator():
    """docs/architecture.md names every audited operator class and every dfthc module."""
    text = (DOCS / "architecture.md").read_text()
    assert [cls.__name__ for cls in CLASSES.values() if f"`{cls.__name__}`" not in text] == []
    modules = [path.relative_to(REPO / "src").with_suffix("").as_posix().replace("/", ".")
               for path in (REPO / "src" / "dfthc").rglob("*.py") if path.name != "__init__.py"]
    assert [module for module in modules if f"`{module}`" not in text] == []
