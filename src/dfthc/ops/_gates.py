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

"""Toffoli-level gates that carry one zeroed work wire.

``toffoli`` is a two-control ``MultiControlledX`` with one explicit ``work_wires`` wire of
``work_wire_type="zeroed"``, and ``cswap`` is PennyLane's ``CSWAP`` identity around it.
:data:`dfthc.compile.GATE_SET` has no ``Toffoli``, so the graph decomposes the
``MultiControlledX`` to ``TemporaryAND``, ``CNOT``, ``Adjoint(TemporaryAND)`` on the work wire
(rule ``decompose_mcx_two_controls_elbows``): one Toffoli of ``T`` and one measured uncompute of
``A``. A bare ``Toffoli`` or ``CSWAP`` has no decomposition path. The work wire must be in
``|0>``; it is returned there.
"""

from __future__ import annotations

import pennylane as qp
from pennylane.typing import Wire

from dfthc.compile import resource_key


def toffoli(c0, c1, target, work) -> None:
    """``Toffoli(c0, c1, target)`` with the zeroed wire ``work``."""
    qp.MultiControlledX(wires=[c0, c1, target], work_wires=[work], work_wire_type="zeroed")


def cswap(control, a, b, work) -> None:
    """``CSWAP(control, a, b)`` with the zeroed wire ``work``: ``CNOT(b, a)``,
    :func:`toffoli` ``(control, a, b)``, ``CNOT(b, a)``."""
    qp.CNOT([b, a])
    toffoli(control, a, b, work)
    qp.CNOT([b, a])


def toffoli_key() -> qp.operation.Operator:
    """The resource key of :func:`toffoli`."""
    return resource_key(lambda: qp.MultiControlledX(Wire[3], work_wires=Wire[1],
                                                    work_wire_type="zeroed"))


def cswap_resources(count: int) -> dict:
    """The gates of ``count`` :func:`cswap` calls (an empty dict for 0)."""
    return {qp.CNOT: 2 * count, toffoli_key(): count} if count else {}
