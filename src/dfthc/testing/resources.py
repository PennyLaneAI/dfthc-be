"""Assertions on compiled resources and the rule-lowering warning policy.

Spec sections 7.1 (warning noise), 7.4 rule 12 (warning policy) and 7.6 (accounting).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterable

import pennylane as qp
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning
from catalyst.passes import graph_decomposition

from dfthc.compile import FIXED_DECOMPS, GATE_SET, CompiledResources, dfthc_operator_names

_EXCLUDED_RULE = (
    r"(?:_\w*_ppm|_toffoli_elbow|_cphase_to_single_phase|\w*_workers?\w*|use_allocation\(\w+\))"
)
NOISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"^Excluded the {_EXCLUDED_RULE} decomposition rule for \S+ since "
        r"--decompose-lowering cannot work with multiple registers yet$"
    ),
    re.compile(r"^Skipped the _adjoint_temporary_and\w* rule for C\(Adjoint\(TemporaryAND\)\)"),
)
"""RuleLoweringWarning texts that PennyLane's own rules print on every compile (spec 7.1)."""

UNRESOLVED_BRANCH_MESSAGE = (
    r"Resource analysis was unable to determine the branch of a conditional or switch statement"
)
"""qp.analyze UserWarning for an unhinted conditional. PennyLane's SemiAdder rule triggers it on
every compile, so it is filtered; exact T4 counts and the erase-hint tests guard our own code."""


def classify_rule_warning(message: str, names: Iterable[str] | None = None) -> str:
    """Return ``"dfthc"``, ``"noise"`` or ``"other"`` for a RuleLoweringWarning message.

    A message naming a dfthc operator is ``"dfthc"`` even if it also matches a noise pattern.
    """
    names = dfthc_operator_names() if names is None else set(names)
    if any(re.search(rf"\b{re.escape(name)}\b", message) for name in names):
        return "dfthc"
    if any(pattern.search(message) for pattern in NOISE_PATTERNS):
        return "noise"
    return "other"


def install_warning_filters() -> None:
    """Show every RuleLoweringWarning except the known PennyLane noise."""
    warnings.filterwarnings("always", category=RuleLoweringWarning)
    for pattern in NOISE_PATTERNS:
        warnings.filterwarnings("ignore", message=pattern.pattern.lstrip("^"),
                                category=RuleLoweringWarning)
    warnings.filterwarnings("ignore", message=UNRESOLVED_BRANCH_MESSAGE, category=UserWarning)


def verbose_program(qfunc: Callable[[], None], num_wires: int):
    """The compile stack of spec 7.2 with the solver's rule listing forwarded to stderr (T6).

    Only ``qjit(verbose=True)`` forwards the compiler output. Run the returned program (or
    ``qp.analyze`` it) under pytest's ``capfd`` and pass the captured text to :func:`chosen_rules`.
    """

    @qp.qjit(capture=True, verbose=True)
    @graph_decomposition(gate_set=GATE_SET, fixed_decomps=dict(FIXED_DECOMPS) or None,
                         verbose=True)
    @qp.qnode(qp.device("null.qubit", wires=num_wires))
    def program():
        qfunc()
        return qp.probs(wires=[0])

    return program


def chosen_rules(listing: str) -> list[str]:
    """Rule names from the lines ``Chosen Rule: __builtin_<rule>`` of a verbose listing."""
    prefix = "Chosen Rule: __builtin_"
    return [line.split(prefix, 1)[1].strip() for line in listing.splitlines() if prefix in line]


def assert_resources(
    resources: CompiledResources, *, toffoli: int, measurements: int,
    measured_and_uncomputes: int | None = None,
) -> None:
    """Assert exact T, M and optionally A, with every mismatch in one message."""
    expected = {"toffoli": toffoli, "measurements": measurements}
    if measured_and_uncomputes is not None:
        expected["measured_and_uncomputes"] = measured_and_uncomputes
    actual = {name: getattr(resources, name) for name in expected}
    if actual != expected:
        raise AssertionError(
            f"resources differ: expected {expected}, got {actual}; counts {resources.counts}"
        )
