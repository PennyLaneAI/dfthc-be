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

"""The vendored dfthcbe data are complete, unmodified and licensed (spec section 8.1)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parents[2] / "data" / "dfthc"
COMMIT = "930255020cec30172e0a7d17c0071d9476a687d1"

SHA256 = {
    "fe2s2/best.json": "baff823d9f67ba9f75149737dc0b6855b1b553167b2b8945c13113497c30c509",
    "fe2s2/factors.npz": "eb043ece37243e56a14ca7e2d2b4239c56af027cfdc7055bbac70209bc8c2f55",
    "fe2s2/summary.json": "376eb32239ad3205899381f2b0dcc5c3206aeeb3877edc27bac5ad4bf94a7161",
    "fe4s4/best.json": "059d7e932da55a562324df0e8c7caeab986130ad5603dd80e115da39259fbd43",
    "fe4s4/factors.npz": "a543968b05f8c4fe88e2e2535f5cc722ba6390a395d8c868f6442d6845764fc8",
    "fe4s4/summary.json": "605d93634b9bb25245e19909c33255c89dc35839f9447a290a8df540349de74b",
    "h2_631g_R2_B2_C2/factors.npz": "dcfe9adb83e380457d9a3336828c1a6f8f09a063164dd376ac26a1acab428f7f",
    "h2_631g_R2_B2_C2/integrals.npz": "91771a0379325ed329b50b68cd2ef2a79b67bd5cdb2634712354ceb94f7e8395",
    "h2_631g_R2_B2_C2/summary.json": "84f26717f7534cc81f5f9e9f47011cf9dce657ef956f0488c672189399e0e3f7",
    "h2_631g_R2_B3_C3/factors.npz": "3a80f93f00ca0e31209c02b0e3602f1185e9319fad5af1927f065e7eed414911",
    "h2_631g_R2_B3_C3/integrals.npz": "91771a0379325ed329b50b68cd2ef2a79b67bd5cdb2634712354ceb94f7e8395",
    "h2_631g_R2_B3_C3/summary.json": "6b3c775ef1b279146840cd70ad6e29305b1b9f7626d558ba1c2f1139de2d59b0",
    "h2_sto3g_R2_B2_C1/factors.npz": "7beefa76b000e0ed35b069578f6b11b75ec4511f411ae4f3205790e05849323b",
    "h2_sto3g_R2_B2_C1/integrals.npz": "fa08d4292c852bc096ef4763a3b095f4a3ce7df45dc85c2ce58ea134d0ae24cf",
    "h2_sto3g_R2_B2_C1/summary.json": "ff76a8b435a122e538a9fb10f7d24c06bec6db91f091abd8c34e1580329a8fd6",
    "naphthalene/factors.npz": "cf0ac51dfacbfb69f194b37d07107d302d6e4b5464913b2b98a1f7358eba94f7",
    "naphthalene/integrals.npz": "7a403373a46c2ed78c9fba15a5ab9ba913ed670f51c3dfabccdb3cf16c81cd6f",
    "naphthalene/summary.json": "33d4a4b762039a9d711c44fc051ea37826a5d5950af41807664a621067d0c62b",
    "LICENSE": "f5ba331216ebf35ee7079088051f95c9b8a234ffeb08f9fc0c70825a12f16a71",
}


def test_file_set_is_exactly_the_vendored_copy():
    present = {str(p.relative_to(DATA)) for p in DATA.rglob("*") if p.is_file()}
    assert present == set(SHA256) | {"NOTICE"}


@pytest.mark.parametrize("name", sorted(SHA256))
def test_file_is_unmodified(name):
    assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == SHA256[name]


def test_license_is_apache_2():
    text = (DATA / "LICENSE").read_text()
    assert "Apache License" in text and "Version 2.0, January 2004" in text


def test_notice_names_source_and_commit():
    notice = (DATA / "NOTICE").read_text()
    assert "https://github.com/Covestro-HPQC/dfthcbe" in notice
    assert COMMIT in notice
    assert "unmodified" in notice


def test_h2_sto3g_loads():
    factors = np.load(DATA / "h2_sto3g_R2_B2_C1" / "factors.npz")
    summary = json.loads((DATA / "h2_sto3g_R2_B2_C1" / "summary.json").read_text())
    assert {"U", "W", "w_B"} <= set(factors.files)
    assert factors["U"].shape[:2] == (2, 2)
    assert {"lambda_sos", "e_sos", "e_upper", "e_gap"} <= set(summary)
