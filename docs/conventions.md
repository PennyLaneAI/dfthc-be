# Conventions

The conventions follow dfthcbe at commit `930255020cec30172e0a7d17c0071d9476a687d1` unless stated
otherwise, so blocks compare one to one (spec 6). Symbols are those of spec 3.1:
`X_o = N + RC`, `w_x = lg X_o`, `w_b = lg(B + 1)`, `w_r = lg R`, with `lg y = ceil(log2 y)`.

## Indices and flags (spec 6.1)

- **The generator index `x_o`.** `x_o < RC` is a sum-of-squares (SF) generator, with
  `x_o = r C + c`; the last N values are the one-body tail, its `N+` D1 rows first, then the Q1
  rows.
- **Families.** `G0 = 0` is SF and `G0 = 1` one-body; `G1 = 0` is D1 and `G1 = 1` Q1.
- **The inner index `b`.** `b = B` is the identity term; `r = floor(x_o / C)`.
- **Success flags.** `succ_x_o` and `succ_b` are 1 where the flagged uniform preparations succeed;
  a failed `x_o` makes `REF_T2` act as `-I`, a failed `b` makes the select act as `+I`
  (spec 8.5).

## Signs and spins (spec 6.1)

- `InnerPrepare` loads `sgn = [w_b < 0]` and `d_sgn = sgn XOR sgn[alt]`; `InnerSigns` applies
  `Z(sgn)` and `CZ(less_than_b, d_sgn)`.
- `sigma_0` is the inner spin (the Majorana flavour on one-body rows), `sigma_1` the outer spin;
  `sg = sigma_{G0}` controls the spin swap.

## Fermions (spec 6.2)

- Jordan-Wigner per spin register: `gamma_{j,0} = Z_0 ... Z_{j-1} X_j` and
  `gamma_{j,1} = Z_0 ... Z_{j-1} Y_j`, qubit 0 the most significant, `|1>` occupied; the system is
  `psi_up` (x) `psi_dn`.
- The Majorana gate acts on `psi_dn[0]`: SF applies `-Z` (dfthcbe's convention, `i gamma_0
  gamma_1`; the paper's Fig. 4 shows `+Z`), D1 `a = (X + iY)/2`, Q1 `a^dag = (X - iY)/2`.

## Givens angles (spec 6.3)

Rung k acts on qubits `(q_k, q_{k+1})`. An angle is stored as
`x_k = round(theta_k / (2 pi) * 2^(b_rot - 1)) mod 2^(b_rot - 1)`, and the rung realizes
`theta = 4 pi x / 2^b_rot` exactly, with the rung matrix `[[cos, -sin], [sin, cos]]` on
`{|01>, |10>}`. Rung N - 2 comes first in time and rung 0 last.

## Alias sampling (spec 6.4)

`less_than = [keep <= m]`; a set flag swaps `alt` into the index register. Keep values have
`b_k1` (outer) and `b_k2` (inner) bits.

## Phase gradient (spec 6.5)

`PhaseGradientStatePrep` prepares the gradient once, outside the walk, which returns it
unchanged. Adding `k` into the gradient register multiplies the state by `e^{+2 pi i k / 2^b}`;
`PhaseGradientRotation(k, m)` applies `e^{2 pi i k t / 2^m}` on `|t>`.

## Reflections (spec 6.6)

Reflections are exact, global sign included: `2|0><0| - I` is a multi-controlled Z with
`GlobalPhase(pi)` when uncontrolled and `Z` on the control when controlled. No generic
`C(GlobalPhase)` is used.

## Wires and registers (spec 6.7)

- Integers in registers are big-endian: the first wire is the most significant bit.
- Register names follow dfthcbe: `x_o`, `succ_x_o`, `sigma_1`, `m_x_o`, `less_than_x_o`,
  `alt_x_o`, `keep_x_o`, `junk_x_o`, `b`, `succ_b`, `sigma_0`, `m_b`, `less_than_b`, `alt_b`,
  `keep_b`, `sgn`, `d_sgn`, `r`, `junk_b`, `angles`, `G0`, `G1`, `b_neq_B`, `sg`, `psi_up`,
  `psi_dn`, `phase_grad`, `work`.
- The assembly operators take `WalkLayout`'s groups; every other operator takes the registers of
  its spec 9 table by name, and a composite passes each child only the registers it needs.
- One operator's registers are distinct qubits: two registers that share a wire do not compile.

## Arguments and tables

- Everything that determines the circuit's structure is a compilable argument; dynamic arguments
  carry runtime data only (`QROAMRead`'s per-leaf table words, `PhaseFixup`'s parities,
  `QPERound`'s earlier bits).
- Compilable integers fit in signed 64 bits, so wide table entries are tuples of 32-bit words,
  least significant first (L-24); dynamic table words are `int64` holding 32 data bits (L-26).
- Adjoints use the instance form `qp.adjoint(Op(...))` (L-25).

## Numbers in tests

Every comparison states both tolerances: `rtol=0` with an absolute `atol` (1e-12 for dense and
simulator comparisons unless a test names another), or an explicit relative bound with `atol=0`;
`tests/test_tolerance_hygiene.py` enforces it.
