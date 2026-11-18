"""DFTHC block encoding, qubitized walk and iterative phase estimation in PennyLane.

The design is docs/superpowers/specs/2026-11-11-dfthc-pennylane-design.md.
"""

from dfthc.compile import GATE_SET
from dfthc.data.factors import load_factors
from dfthc.data.tables import WalkTables
from dfthc.ops.walk import DFTHCBlockEncoding, DFTHCWalk
from dfthc.params import H2_631G_TEST, H2_STO3G_TEST, WalkParams
from dfthc.qpe import ControlledWalkPower, QPERound
from dfthc.registers import WalkLayout

__version__ = "0.1.0"
