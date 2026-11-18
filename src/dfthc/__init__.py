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

"""DFTHC block encoding, qubitized walk and iterative phase estimation in PennyLane.

The design is docs/design.md.
"""

from dfthc.compile import GATE_SET
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.walk import DFTHCBlockEncoding, DFTHCWalk
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, WalkParams
from dfthc.qpe import ControlledWalkPower, QPERound
from dfthc.registers import WalkLayout

__all__ = ["GATE_SET", "H2_631G_TEST", "H2_STO3G_TEST", "ControlledWalkPower",
           "DFTHCBlockEncoding", "DFTHCWalk", "QPERound", "WalkLayout", "WalkParams",
           "WalkTables", "load_factors"]

__version__ = "0.1.0"
