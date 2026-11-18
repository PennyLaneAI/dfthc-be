"""The spin multiplexer and the controlled spin swap of the inner select (spec 9.12 and 9.13).

Both are permutations of the computational basis and their own inverses. They define
``compute_matrix`` (small sizes only: ``SpinSwap`` has ``2**(2N + 1)`` rows), so their T2 tests
compare the decomposition's matrix with the operator's (``check_matrix=True``, spec 12.1).
"""

from __future__ import annotations

import numpy as np
import pennylane as qp
from pennylane.core import Operator2
from pennylane.typing import Wire

from dfthc.compile import register_rule, resource_key
from dfthc.ops._gates import cswap, cswap_resources, toffoli, toffoli_key
from dfthc.ops._validate import check_wires


class SpinMux(Operator2):
    r"""``sg ^= sigma_{G0}``: the spin that controls the spin swap (steps 12 and 17).

    **Action.** :math:`\mathrm{sg} \mathrel{\oplus}= \sigma_1` if :math:`G_0 = 1`, else
    :math:`\sigma_0`; that is, :math:`\mathrm{sg} \mathrel{\oplus}= \sigma_0 \oplus
    G_0(\sigma_0 \oplus \sigma_1)`. ``G0``, ``sigma_0`` and ``sigma_1`` are unchanged. On SF
    branches (:math:`G_0 = 0`) the swap follows the inner spin :math:`\sigma_0`; on D1/Q1 branches
    the outer spin :math:`\sigma_1` (spec 6.1).

    **Registers.**

    =========== ===== ======= ======================================
    name        width input   output
    =========== ===== ======= ======================================
    ``G0``      1     any     unchanged
    ``sigma_0`` 1     any     unchanged
    ``sigma_1`` 1     any     unchanged
    ``sg``      1     any     :math:`\mathrm{sg} \oplus \sigma_{G_0}`
    ``work``    1     ``|0>`` ``|0>``
    =========== ===== ======= ======================================

    **Compilable arguments.** None.

    **Pre- and postconditions.** None beyond the action: it is a permutation of the basis. In
    the walk ``sg`` starts in ``|0>``, and the second application (step 17) returns it to
    ``|0>``. ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.12). ``CNOT(sigma_0, sg)``, ``CNOT(sigma_0, sigma_1)``,
    ``Toffoli(G0, sigma_1, sg)`` (a two-control ``MultiControlledX`` with the zeroed work wire,
    :mod:`dfthc.ops._gates`), ``CNOT(sigma_0, sigma_1)``.

    **Adjoint and control.** Self-adjoint: ``Adjoint(SpinMux)`` is pinned to the operator
    itself (spec 7.5). Never controlled. The walk applies it twice instead of adjointing it.

    **Cost.** :func:`dfthc.costs.spin_mux`: :math:`T = 1`, one plain ``Toffoli`` (``A = 0``);
    ``M = 0``. Two per ``InnerSelect``.

    **Paper and dfthcbe.** Steps 12 and 17; the paper computes :math:`\sigma_{G_0}` in App. B.3
    but has no Table III row for it. dfthcbe: ``InnerSelect._build_sigma``, two
    ``MultiControlX`` gates (2 Toffolis per application).

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.spin import SpinMux

        program = lambda: SpinMux(G0=[0], sigma_0=[1], sigma_1=[2], sg=[3], work=[4])
        compiled_resources(program, num_wires=5).toffoli  # 1
    """

    wire_argnames = ("G0", "sigma_0", "sigma_1", "sg", "work")

    def __init__(self, G0, sigma_0, sigma_1, sg, work):
        check_wires("SpinMux", {"G0": G0, "sigma_0": sigma_0, "sigma_1": sigma_1, "sg": sg,
                                "work": work}, self.required_wires())
        super().__init__(G0, sigma_0, sigma_1, sg, work)

    @staticmethod
    def required_wires() -> dict[str, int]:
        """Exact register sizes."""
        return {"G0": 1, "sigma_0": 1, "sigma_1": 1, "sg": 1, "work": 1}

    @staticmethod
    def compute_matrix(G0, sigma_0, sigma_1, sg, work) -> np.ndarray:
        """The 32 x 32 permutation matrix in the wire order ``G0, sigma_0, sigma_1, sg, work``
        (the identity on ``work``)."""
        matrix = np.zeros((16, 16))
        for index in range(16):
            g0, s0, s1 = ((index >> (3 - i)) & 1 for i in range(3))
            matrix[index ^ (s1 if g0 else s0), index] = 1
        return np.kron(matrix, np.eye(2))


def _spin_mux_resources(G0, sigma_0, sigma_1, sg, work):
    return {qp.CNOT: 3, toffoli_key(): 1}


@qp.register_resources(_spin_mux_resources)
def _spin_mux(G0, sigma_0, sigma_1, sg, work):
    qp.CNOT([sigma_0[0], sg[0]])
    qp.CNOT([sigma_0[0], sigma_1[0]])
    toffoli(G0[0], sigma_1[0], sg[0], work[0])
    qp.CNOT([sigma_0[0], sigma_1[0]])


def _adjoint_spin_mux_resources(base):
    return {resource_key(lambda: SpinMux(Wire[1], Wire[1], Wire[1], Wire[1], Wire[1])): 1}


@qp.register_resources(_adjoint_spin_mux_resources)
def _adjoint_spin_mux(base):
    SpinMux(base.G0, base.sigma_0, base.sigma_1, base.sg, base.work)


register_rule(SpinMux, _spin_mux)
register_rule("Adjoint(SpinMux)", _adjoint_spin_mux)


class SpinSwap(Operator2):
    r"""Swap ``psi_up`` and ``psi_dn`` if ``sg`` is 1 (steps 13 and 17).

    **Action.** :math:`|s\rangle|u\rangle|d\rangle \mapsto |s\rangle|u\rangle|d\rangle` for
    :math:`s = 0` and :math:`|1\rangle|d\rangle|u\rangle` for :math:`s = 1`, with ``u`` and ``d``
    the ``N``-qubit contents of ``psi_up`` and ``psi_dn``. It routes the active spin register
    into ``psi_dn``, where the Givens ladder and the Majorana gate act.

    **Registers.** With ``N = len(psi_up)``:

    ========== ===== ======= =============================================
    name       width input   output
    ========== ===== ======= =============================================
    ``sg``     1     any     unchanged
    ``psi_up`` ``N`` any     ``psi_dn``'s content if ``sg``, else unchanged
    ``psi_dn`` ``N`` any     ``psi_up``'s content if ``sg``, else unchanged
    ``work``   1     ``|0>`` ``|0>``
    ========== ===== ======= =============================================

    **Compilable arguments.** None; ``N >= 1`` follows from the wires.

    **Pre- and postconditions.** None beyond the action: it is a permutation of the basis.
    ``work`` starts and ends in ``|0>``.

    **Decomposition** (spec 9.13). ``N`` exact ``CSWAP`` gates ``(sg, psi_up[k], psi_dn[k])``,
    each ``CNOT``, a two-control ``MultiControlledX`` with the zeroed work wire and ``CNOT``
    (:func:`dfthc.ops._gates.cswap`).

    **Adjoint and control.** Self-adjoint: ``Adjoint(SpinSwap)`` is pinned to the operator
    itself (spec 7.5). Never controlled. The walk applies it twice instead of adjointing it.

    **Cost.** :func:`dfthc.costs.spin_swap`: :math:`T = N`, ``N`` plain ``Toffoli`` gates from
    the ``CSWAP`` (``A = 0``); ``M = 0``. Two per ``InnerSelect``; 54 at FeMoco-54.

    **Paper and dfthcbe.** Steps 13 and 17, the paper's :math:`U_{Swap}` (Table III); dfthcbe:
    ``InnerSelect.uswap``, a ``CSwap`` of ``N`` qubits.

    **Example.**

    .. code-block:: python

        from dfthc.compile import compiled_resources
        from dfthc.ops.spin import SpinSwap

        program = lambda: SpinSwap(sg=[0], psi_up=[1, 2], psi_dn=[3, 4], work=[5])
        compiled_resources(program, num_wires=6).toffoli  # 2
    """

    wire_argnames = ("sg", "psi_up", "psi_dn", "work")

    def __init__(self, sg, psi_up, psi_dn, work):
        num_modes = len(psi_up)
        if num_modes < 1:
            raise ValueError("SpinSwap: psi_up needs N >= 1 wires, got 0")
        check_wires("SpinSwap", {"sg": sg, "psi_up": psi_up, "psi_dn": psi_dn, "work": work},
                    self.required_wires(num_modes))
        super().__init__(sg, psi_up, psi_dn, work)

    @staticmethod
    def required_wires(N: int) -> dict[str, int]:
        """Exact register sizes for ``N`` modes per spin register."""
        return {"sg": 1, "psi_up": N, "psi_dn": N, "work": 1}

    @staticmethod
    def compute_matrix(sg, psi_up, psi_dn, work) -> np.ndarray:
        """The permutation matrix in the wire order ``sg, psi_up, psi_dn, work`` (the identity on
        ``work``)."""
        num_modes = len(psi_up)
        size, mask = 2 ** (2 * num_modes + 1), 2**num_modes - 1
        matrix = np.zeros((size, size))
        for index in range(size):
            up, dn = (index >> num_modes) & mask, index & mask
            swapped = (index >> (2 * num_modes)) << (2 * num_modes) | (dn << num_modes) | up
            matrix[swapped if index >> (2 * num_modes) else index, index] = 1
        return np.kron(matrix, np.eye(2))


def _spin_swap_resources(sg, psi_up, psi_dn, work):
    return cswap_resources(len(psi_up))


@qp.register_resources(_spin_swap_resources)
def _spin_swap(sg, psi_up, psi_dn, work):
    for up, dn in zip(list(psi_up), list(psi_dn), strict=True):
        cswap(sg[0], up, dn, work[0])


def _adjoint_spin_swap_resources(base):
    num_modes = len(base.psi_up)
    return {resource_key(lambda: SpinSwap(Wire[1], Wire[num_modes], Wire[num_modes],
                                          Wire[1])): 1}


@qp.register_resources(_adjoint_spin_swap_resources)
def _adjoint_spin_swap(base):
    SpinSwap(base.sg, base.psi_up, base.psi_dn, base.work)


register_rule(SpinSwap, _spin_swap)
register_rule("Adjoint(SpinSwap)", _adjoint_spin_swap)
