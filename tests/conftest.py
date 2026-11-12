"""Shared pytest configuration (spec sections 7.1, 7.4 rule 12 and 12.6)."""

import warnings

import catalyst  # noqa: F401  # importing Catalyst reads CATALYST_SILENCE_RULE_LOWERING_WARNINGS
import pytest
from catalyst.decomposition.rule_lowering_warning import RuleLoweringWarning

from dfthc.testing.resources import classify_rule_warning, install_warning_filters

# Second line of defence (spec 7.1): if the variable was not exported before Python started,
# Catalyst installed an "ignore" filter for RuleLoweringWarning; these filters take precedence.
install_warning_filters()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """`release` implies `slow`; runs before `-m` deselection (spec section 12.6)."""
    for item in items:
        if item.get_closest_marker("release") and not item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(autouse=True)
def _rule_lowering_warning_policy():
    """Fail on any RuleLoweringWarning that names a dfthc operator, except the expected
    measuring-rule skip; drop that one and the PennyLane noise (spec 7.4, rule 12)."""
    with warnings.catch_warnings(record=True) as records:
        install_warning_filters()
        yield
    offending = []
    for record in records:
        if issubclass(record.category, RuleLoweringWarning):
            kind = classify_rule_warning(str(record.message))
            if kind == "dfthc":
                offending.append(str(record.message))
                continue
            if kind in ("noise", "expected"):
                continue
        warnings.warn_explicit(record.message, record.category, record.filename, record.lineno)
    if offending:
        pytest.fail("RuleLoweringWarning names a dfthc operator:\n" + "\n".join(offending))
