"""Our tables and encoded model against dfthcbe's reference fixtures (spec 8.6, check 4)."""

import dataclasses
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from dfthc.data.encoded import encoded_block
from dfthc.data.factors import load_factors, random_factors
from dfthc.data.tables import WalkTables
from dfthc.params import FEMOCO54, H2_631G_TEST, H2_STO3G_TEST, WalkParams
from dfthc.testing import fixtures
from dfthc.testing.fixtures import SKIP_REASON, load_dfthcbe_fixture, require_fixture

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_dfthcbe_fixtures.py"
COMMIT = "930255020cec30172e0a7d17c0071d9476a687d1"
TABLE_FIELDS = [field.name for field in dataclasses.fields(WalkTables)]
X_O_FAILS = WalkParams(N=2, R=3, B=1, C=1, n_plus=1, b_rot=6, b_k1=3, b_k2=2, s_u=3, k1=1, k2=1,
                       ke1=1, ke2=1)
"""The shape of ``X_O_FAILS`` in ``test_encoded.py``: X_o = 5, so the uniform preparation of
``x_o`` fails, which neither H2 preset exercises."""
ENCODED_CASES = [
    ("encoded_h2_sto3g_test.npz", H2_STO3G_TEST, lambda: load_factors("h2_sto3g_R2_B2_C1")),
    ("encoded_x_o_fails.npz", X_O_FAILS, lambda: random_factors(2, 3, 1, 1, n_plus=1, seed=3)),
]


def _script():
    """The fixture script's module; it imports Qualtran and dfthcbe only inside its functions."""
    spec = importlib.util.spec_from_file_location("make_dfthcbe_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_fixture_skips_with_the_explicit_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(fixtures, "FIXTURE_DIR", tmp_path)
    with pytest.raises(pytest.skip.Exception, match=re.escape(SKIP_REASON)):
        require_fixture("missing.json")
    (tmp_path / "present.json").write_text('{"a": 1}')
    assert require_fixture("present.json") == tmp_path / "present.json"
    assert load_dfthcbe_fixture("present.json") == {"a": 1}


def test_the_script_writes_the_fixtures_these_tests_read():
    script = _script()
    assert script.FIXTURES == ("tables_h2_sto3g_test.json", "tables_h2_631g_test.json",
                               "encoded_h2_sto3g_test.npz", "encoded_x_o_fails.npz",
                               "femoco_block_toffolis.json", "provenance.json")
    assert [name for name, _, _ in ENCODED_CASES] == [
        name for name in script.FIXTURES if name.startswith("encoded_")]
    assert script.DFTHCBE_COMMIT == COMMIT
    assert script.X_O_FAILS == X_O_FAILS
    assert fixtures.FIXTURE_DIR == script.REPO / "tests" / "fixtures" / "dfthcbe"


@pytest.mark.parametrize(("name", "params", "system"), [
    ("tables_h2_sto3g_test.json", H2_STO3G_TEST, "h2_sto3g_R2_B2_C1"),
    ("tables_h2_631g_test.json", H2_631G_TEST, "h2_631g_R2_B2_C2"),
])
def test_our_tables_equal_dfthcbe_exactly(name, params, system):
    fixture = load_dfthcbe_fixture(name)
    assert sorted(fixture) == sorted(TABLE_FIELDS)
    reference = WalkTables.from_compilable([fixture[field] for field in TABLE_FIELDS])
    assert WalkTables.from_factors(load_factors(system), params) == reference


def test_encoded_block_equals_dfthcbe():
    """dfthcbe's <0|BE|0> by exact contraction equals H~_SA / Lambda~ - I to 1e-10, for the H2
    preset and for the shape where the uniform preparation of x_o fails (``X_O_FAILS``)."""
    for name, params, make_factors in ENCODED_CASES:
        fixture = load_dfthcbe_fixture(name)
        assert sorted(fixture) == ["block", "h_sa", "lambda_tilde"]
        factors = make_factors()
        h_tilde, lambda_tilde = encoded_block(
            params, WalkTables.from_factors(factors, params), factors)
        dim = 4**params.N
        assert lambda_tilde == pytest.approx(float(fixture["lambda_tilde"]), rel=1e-12, abs=0), name
        assert np.allclose(h_tilde, fixture["h_sa"], rtol=0, atol=1e-10), name
        assert np.allclose(h_tilde / lambda_tilde - np.eye(dim), fixture["block"],
                           rtol=0, atol=1e-10), name


def test_femoco_block_counts_are_loadable():
    """Task 4.3 compares these with ours; here they only need to load and add up."""
    counts = load_dfthcbe_fixture("femoco_block_toffolis.json")
    assert list(counts) == list(_script().FEMOCO_BLOCKS)
    assert all(type(value) is int and value > 0 for value in counts.values())
    blocks = (counts["outer_prepare"] + counts["outer_unprepare"]
              + 2 * (counts["inner_prepare"] + counts["inner_unprepare"])
              + counts["inner_select"] + counts["inner_select_adjoint"] + counts["ref_t2"])
    assert counts["block_encoding"] == blocks
    assert counts["walk"] == counts["block_encoding"] + counts["ref_ab"]
    assert counts["controlled_walk"] == counts["walk"] + 1


def test_provenance_names_the_commit_and_parameters():
    provenance = load_dfthcbe_fixture("provenance.json")
    assert provenance["dfthcbe_commit"] == COMMIT
    assert {"dfthcbe", "qualtran", "numpy", "python"} <= set(provenance["versions"])
    for name, preset in [("H2_STO3G_TEST", H2_STO3G_TEST), ("H2_631G_TEST", H2_631G_TEST),
                         ("FEMOCO54", FEMOCO54), ("X_O_FAILS", X_O_FAILS)]:
        assert provenance["parameters"][name] == dataclasses.asdict(preset)
