"""Named wires of the walk: ``WalkLayout`` (spec section 10)."""

from __future__ import annotations

from pennylane.wires import Wires

from dfthc.params import WalkParams, lg, odd_part

GROUPS: dict[str, tuple[str, ...]] = {
    "system": ("psi_up", "psi_dn"),
    "outer": ("x_o", "succ_x_o", "succ_work_x_o", "sigma_1", "m_x_o", "less_than_x_o",
              "cmp_work_x_o", "alt_x_o", "keep_x_o", "junk_x_o"),
    "inner": ("b", "succ_b", "succ_work_b", "sigma_0", "m_b", "less_than_b", "cmp_work_b",
              "alt_b", "keep_b", "sgn", "d_sgn", "r", "junk_b"),
    "select": ("angles", "G0", "G1", "b_neq_B", "sg"),
    "phase_grad": ("phase_grad",),
    "work": ("work",),
}
"""Groups of spec 10.1 and their registers, in layout order."""


class WalkLayout:
    """Non-overlapping named wires for every register of the walk (spec 10.1).

    Build it with :meth:`from_params`. Every register of :data:`GROUPS` is an attribute holding
    its ``Wires``; a register of width 0 is an empty ``Wires``. With ``qpe_bits > 0`` the layout
    also has ``qpe_control`` (1 wire) and ``qpe_grad`` (``qpe_bits`` wires) after ``work``.
    Wires are consecutive integers from 0 in group order: ``system``, ``outer``, ``inner``,
    ``select``, ``phase_grad``, ``work``.

    The register sizes follow spec 9.7 and 9.8: ``succ_work_* = max(lg(l) - 1, 0)`` with ``l``
    the odd part of ``X_o`` or ``B + 1``, ``cmp_work_* = b_k - 1``, ``junk_x_o = (K1 - 1) d1``
    and ``junk_b = (K2 - 1) d2``. The work pool size is :meth:`work_size`. That formula is
    provisional: Task 3.3 asserts that every operator's ``required_wires()["work"]`` fits it.
    """

    def __init__(self, params: WalkParams, qpe_bits: int, registers: dict[str, Wires]):
        self.params = params
        self.qpe_bits = qpe_bits
        self._registers = dict(registers)
        for name, wires in registers.items():
            setattr(self, name, wires)

    @classmethod
    def from_params(cls, params: WalkParams, qpe_bits: int = 0) -> WalkLayout:
        if qpe_bits < 0:
            raise ValueError(f"qpe_bits must be >= 0; got {qpe_bits}")
        sizes = cls.register_sizes(params)
        if qpe_bits:
            sizes |= {"qpe_control": 1, "qpe_grad": qpe_bits}
        registers, start = {}, 0
        for name, size in sizes.items():
            registers[name] = Wires(range(start, start + size))
            start += size
        return cls(params, qpe_bits, registers)

    @staticmethod
    def register_sizes(params: WalkParams) -> dict[str, int]:
        """Width of every register of :data:`GROUPS`, in layout order."""
        p = params
        return {
            "psi_up": p.N, "psi_dn": p.N,
            "x_o": p.w_x, "succ_x_o": 1, "succ_work_x_o": max(lg(odd_part(p.X_o)) - 1, 0),
            "sigma_1": 1, "m_x_o": p.b_k1, "less_than_x_o": 1, "cmp_work_x_o": p.b_k1 - 1,
            "alt_x_o": p.w_x, "keep_x_o": p.b_k1, "junk_x_o": (p.K1 - 1) * p.d1,
            "b": p.w_b, "succ_b": 1, "succ_work_b": max(lg(odd_part(p.B + 1)) - 1, 0),
            "sigma_0": 1, "m_b": p.b_k2, "less_than_b": 1, "cmp_work_b": p.b_k2 - 1,
            "alt_b": p.w_b, "keep_b": p.b_k2, "sgn": 1, "d_sgn": 1, "r": p.w_r,
            "junk_b": (p.K2 - 1) * p.d2,
            "angles": (p.N - 1) * p.b_rot, "G0": 1, "G1": 1, "b_neq_B": 1, "sg": 1,
            "phase_grad": p.b_grad,
            "work": WalkLayout.work_size(p),
        }

    @staticmethod
    def work_size(params: WalkParams) -> int:
        """Shared work pool (spec 10.1), provisional:
        ``max(w_x + b_k1 + w_b + b_k2, b_rot - 1, w_x + w_b, lg(l_x) + 2 s_u - 1,
        lg(l_b) + 2 s_u - 1, 2)``, with ``l_x`` and ``l_b`` the odd parts of ``X_o`` and ``B + 1``.
        """
        p = params
        return max(p.w_x + p.b_k1 + p.w_b + p.b_k2, p.b_rot - 1, p.w_x + p.w_b,
                   lg(odd_part(p.X_o)) + 2 * p.s_u - 1, lg(odd_part(p.B + 1)) + 2 * p.s_u - 1, 2)

    @staticmethod
    def split_group(params: WalkParams, group: str, wires) -> dict:
        """Split the wires of one group into its registers: the only way operators do it.

        ``wires`` is a ``Wires`` (outside rules) or an array (inside rules, where wire registers
        are arrays); each register is the slice of the same type. Raises ``ValueError`` for an
        unknown group or a wrong length.
        """
        if group not in GROUPS:
            raise ValueError(f"unknown group {group!r}; expected one of {list(GROUPS)}")
        sizes = WalkLayout.register_sizes(params)
        total = sum(sizes[name] for name in GROUPS[group])
        if len(wires) != total:
            raise ValueError(f"{group} needs {total} wires; got {len(wires)}")
        parts, start = {}, 0
        for name in GROUPS[group]:
            parts[name] = wires[start:start + sizes[name]]
            start += sizes[name]
        return parts

    def _group(self, group: str) -> Wires:
        return Wires.all_wires([self._registers[name] for name in GROUPS[group]])

    @property
    def system(self) -> Wires:
        return self._group("system")

    @property
    def outer(self) -> Wires:
        return self._group("outer")

    @property
    def inner(self) -> Wires:
        return self._group("inner")

    @property
    def select(self) -> Wires:
        return self._group("select")

    @property
    def num_wires(self) -> int:
        return sum(len(wires) for wires in self._registers.values())

    def angle_register(self, k: int) -> Wires:
        """The ``b_rot`` wires of angle register ``k``, which drives rung ``k`` (spec 6.7)."""
        if not 0 <= k < self.params.N - 1:
            raise ValueError(f"rung {k} outside [0, {self.params.N - 1})")
        b_rot = self.params.b_rot
        return self.angles[k * b_rot:(k + 1) * b_rot]
