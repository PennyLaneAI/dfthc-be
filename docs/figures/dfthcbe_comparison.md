# Our circuit compared with the dfthcbe figure

The dfthcbe README shows the first half of the block encoding as a register-level circuit, paper
steps 1-23. [`walk_circuit.png`](walk_circuit.png) draws our PennyLane implementation the same way,
with red circles where it differs. This note explains those differences.

In short: both implementations run the same algorithm, step for step and in the same order. They
differ in three ways:

1. **When leftover data is measured.** We measure temporary data as soon as we are done with it,
   instead of keeping it on qubits until the end.
2. **How things are undone.** Several "uncompute" steps are done by measuring instead of running
   the computation backwards, which saves Toffoli gates.
3. **A few small circuit constructions.** These change the Toffoli count by a handful each.

Below, "Toffolis" means Toffoli gates plus their cheaper "temporary AND" variant, the standard
cost measure for these circuits. All numbers are for FeMoco-54, the largest benchmark (54 spin
orbitals). Overall, one block encoding costs **9,746 Toffolis in ours against 9,596 in dfthcbe**
(+1.6%). We use **1,049 qubits** in the full walk, against the paper's 1,131.

## A few terms

- **Prepare.** Steps 1-4 and 5-8 each load a probability distribution into an index register
  (`x_o`, then `b`) by *alias sampling*:
  - prepare a uniform superposition over the indices (Unif, green);
  - look up two entries for each index, an alternative index `alt` and a threshold `keep` (blue
    "data" box);
  - compare `keep` with a random number `m` (yellow comparator);
  - swap in `alt` when the comparison says so (the × symbols).
- **Lookup (QROM read).** Reading table data addressed by a quantum index. Our version, like
  dfthcbe's, reads K table rows in parallel and then uses a network of controlled swaps to move the
  right one into the output. The other K−1 rows are left behind as **junk**.
- **Erase / fix-up.** To clean up lookup output without running the lookup backwards, measure the
  output qubits in the X basis. This leaves a known ±1 phase that depends on the index and the
  measurement results. A **phase fix-up** removes it with a cheaper lookup that applies Z gates.
  The image draws this as In<sub>Z</sub> (the lookup over the index) and ℰ (the erased register).
- **Select (steps 10-18).** Loads the rotation angles for the chosen term, rotates the
  electron-spin registers `psi_up`/`psi_dn`, applies a Majorana operator, rotates back, and erases
  the angles.

## Wires

Most wires correspond one to one; the names are dfthcbe's or close to them.

| dfthcbe wire | Ours | Comment |
| --- | --- | --- |
| A<sub>x<sub>o</sub></sub> | `alt_x_o`, `keep_x_o` | The outer table entry, split into its two fields |
| ineq<sub>x<sub>o</sub></sub>, ineq<sub>b</sub> | `less_than_x_o`, `less_than_b` | Comparator results |
| A<sub>b</sub> | `alt_b`, `keep_b`, `sgn`, `d_sgn` | The inner entry; we store two sign bits (see step 9) |
| G (2 qubits) | `G0`, `G1` | Flags from step 10 that record which kind of term was selected |
| σ | `sg` | |
| θ | `angles` | One bit narrower per angle, and also hosts the junk (see below) |

We have extra wires the image doesn't show:
- `succ_work_*` and `cmp_work_*`, 36 qubits of scratch space that make some uncomputes free (see
  "Free uncomputes" below);
- `phase_grad`, a fixed resource state that performs rotations by addition (∇ in the figure);
- a shared `work` pool, not drawn.

## The differences

### Junk is measured right away (columns J, after steps 4 and 8)

dfthcbe keeps each lookup's junk on dedicated qubits until it is erased at the end of the block
encoding. We measure the junk in the X basis immediately after each prepare and keep only the
classical outcomes. Those outcomes are used later in the unprepare's phase fix-up.

Measuring early frees the qubits. We place the junk (72 qubits for the outer lookup, 390 for the
inner one) on the `angles` wires, which are not needed until step 10. This costs no Toffolis and is
the main reason we need fewer qubits than the paper.

### Unprepare (step 21) and angle erase (step 18) measure, then fix up the phase

The image erases in a single column. We do it in explicit stages:
- **Step 21:** 21a measures the 26 qubits of the loaded inner entry. 21b applies one phase fix-up
  that accounts for both these outcomes and the junk outcomes from column J.
- **Step 18** has four stages, and their order matters:
  - 18a measures the angles;
  - 18b fixes the phase of the two-body angles, but only for that kind of term (`G0` = 0);
  - 18c measures the flags `G0` and `G1`;
  - 18d fixes the phase of the remaining angles.

The fix-up needs a scratch register with one qubit per possible index value. We borrow it from the
`angles` wires that were just measured and reset. We round its size up to a power of two (128),
which costs a few extra Toffolis.

Cost:
- step 21: 196 against 179 Toffolis (+16 of the 17 come from the rounding);
- step 18: 102 against 90 (+12).

### `r` is loaded with the rest of the inner entry (step 6)

The inner entry includes the 4-bit index `r`. We store it in the same table row as alt, keep and
the signs, so it goes through the lookup's swap network. That network moves K−1 = 15 candidate
rows, so each extra bit costs 15 Toffolis, and 4 bits cost 60. dfthcbe keeps `r` out of the swap
network; this is the separate `r` box in the image.
Cost: +60 per inner prepare, which is the largest single difference.

### Free uncomputes (steps 1, 5, 20 and 22)

A comparator writes "is keep ≤ m?" into one qubit, using scratch qubits along the way. Normally
undoing it costs as many Toffolis as computing it, 15 here. We keep the comparator's scratch qubits
(`cmp_work_*`) alive until the uncompute. Then the uncompute is just measurements plus Clifford
corrections, with no Toffolis. The uniform preparation (Unif) contains a comparison too, so its
scratch (`succ_work_*`) is kept for the same reason.

In the figure these are the ≤<sup>†</sup> boxes. The image's comparators at steps 3 and 7 are the
same operation with an extra wire for the kept scratch.

Cost: −15 for the comparator uncompute and −7 for Unif<sup>†</sup> in every inner unprepare, and
smaller savings in the prepares. The price is the 36 extra scratch qubits.

### Two sign bits (step 9)

Each inner table row needs the sign of the kept entry and the sign of the alternative. We store
the first in `sgn` and the difference between the two in `d_sgn`. Step 9 applies Z to `sgn`, and Z
to `d_sgn` only if the alternative was taken (controlled on `less_than_b`). The image does this
with one sign register. Both versions are Clifford gates, so this costs no Toffolis. It only adds
a wire.

### Smaller differences that are not circled

- **Spin multiplexer (steps 12 and 17):** the drawing is identical, but ours uses 1 Toffoli
  instead of 2 (−2 in total).
- **Angle width:** each of the 53 angles is stored with 14 bits instead of 15, because the 15th bit
  is always 0. That is 742 wires instead of 795, with no change in Toffolis.
- **Lookup address iteration (steps 2, 6, 10, 11, 18, 21):** our lookups step through the address
  range exactly. dfthcbe skips some address blocks that can't occur. This shifts each lookup by a
  few Toffolis in either direction.

## Toffolis per stage

| Stage | Ours | dfthcbe | Main reasons |
| --- | ---: | ---: | --- |
| Outer prepare (1-4) | 209 | 212 | Free Unif uncompute; address iteration |
| Inner prepare (5-8) | 970 | 913 | `r` through the swap network (+60) |
| Inner select (10-18) | 3,537 | 3,502 | Angle erase (+12); address iteration; spin multiplexer (−2) |
| Inner unprepare (19-22) | 217 | 220 | Free uncomputes (−22) against fix-up rounding (+16) |
| **Block encoding** | **9,746** | **9,596** | |

Every count difference is attributed to a cause in [`docs/resources.md`](../resources.md)
("Reason codes" section).

## Not checked

dfthcbe's comparator direction: the image writes "<", ours computes "≤". We did not confirm the
direction from dfthcbe's code. The difference doesn't affect the cost.
