"""Generate walk_circuit.tex: the DFTHC block encoding's first half plus REF_T2 (steps 1-23) of our
PennyLane implementation, in the style of the dfthcbe README figure.

Build: python walk_circuit_gen.py && pdflatex walk_circuit.tex &&
sips -s format png -Z 3600 walk_circuit.pdf --out walk_circuit.png"""

GREEN, BLUE, YELLOW, GREY, ORANGE, PINK, CAT = ("green!22", "cyan!18", "yellow!60", "gray!35",
                                                "orange!35", "red!18", "gray!12")
T = lambda s: r"\texttt{" + s.replace("_", r"\_") + "}"

ROWS = [  # (label, bundle)
    (T("succ_x_o"), None),
    (T("succ_work_x_o"), r"\lg l_x{-}1"),
    (T("x_o"), r"\lg X_o"),
    (T("m_x_o"), r"b_{k_1}"),
    (T("sigma_1"), None),
    (T("alt_x_o"), r"\lg X_o"),
    (T("keep_x_o"), r"b_{k_1}"),
    (T("less_than_x_o"), None),
    (T("cmp_work_x_o"), r"b_{k_1}{-}1"),
    (T("succ_b"), None),
    (T("succ_work_b"), r"\lg l_b{-}1"),
    (T("b"), r"\lg(B{+}1)"),
    (T("m_b"), r"b_{k_2}"),
    (T("sigma_0"), None),
    (T("alt_b"), r"\lg(B{+}1)"),
    (T("keep_b"), r"b_{k_2}"),
    (T("sgn"), None),
    (T("d_sgn"), None),
    (T("less_than_b"), None),
    (T("cmp_work_b"), r"b_{k_2}{-}1"),
    (T("r"), r"\lg R"),
    (T("G0"), None),
    (T("G1"), None),
    (T("angles") + r" $\supseteq$ " + T("junk_x_o") + ", " + T("junk_b"),
     r"(N{-}1)(b_{\mathrm{rot}}{-}1)"),
    (T("b_neq_B"), None),
    (T("sg"), None),
    (T("psi_up"), "N"),
    (T("psi_dn"), "N"),
    (T("phase_grad"), r"b_{\mathrm{rot}}"),
]
R = {name: i + 1 for i, name in enumerate(
    "succ_x_o succ_work_x_o x_o m_x_o sigma_1 alt_x_o keep_x_o less_than_x_o cmp_work_x_o "
    "succ_b succ_work_b b m_b sigma_0 alt_b keep_b sgn d_sgn less_than_b cmp_work_b r G0 G1 "
    "angles b_neq_B sg psi_up psi_dn phase_grad".split())}
NROWS = len(ROWS)


def g(label, n=1, fill=None, down=None):
    style = f",style={{fill={fill}}}" if fill else ""
    cmd = f"\\gate[{n}{style}]{{{label}}}"
    return cmd + (f"\\wire[d][{down}]{{q}}" if down else "")


def ctrl(frm, to):
    return f"\\ctrl{{{R[to] - R[frm]}}}"


def swap(a, b, c):
    """CSWAP of a and b controlled by c (c below b)."""
    return {a: f"\\swap{{{R[b] - R[a]}}}", b: r"\targX{}", c: f"\\ctrl{{{R[b] - R[c]}}}"}


METER = r"\meter{}"
CTL = r"\control{}"
OCTL = r"\ocontrol{}"
NABLA = g(r"\nabla", fill=CAT)
IN, INZ, EPS = r"\mathrm{In}", r"\mathrm{In}_Z", r"\mathcal{E}"
LE, LED = r"\le", r"\le^{\dagger}"


def unif(top, m, s, dagger=False):
    return {top: g(r"\mathrm{Unif}" + (r"^{\dagger}" if dagger else ""), 3, GREEN),
            m: g("H"), s: g("H"), "phase_grad": NABLA}


def comparator(m, keep, lt, work, dagger=False):
    label = LED if dagger else LE
    cells = {m: g(label, fill=YELLOW, down=R[work] - R[m]), keep: g(label, fill=YELLOW),
             work: g(label, fill=YELLOW)}
    cells[lt] = g(label, fill=YELLOW) if dagger else r"\targ{}"
    return cells


def mux():
    return {"sigma_1": ctrl("sigma_1", "sg"), "sigma_0": CTL, "G0": CTL,
            "sg": g(r"\sigma_{G_0}", fill=BLUE)}


def spin_swap():
    return {"sg": ctrl("sg", "psi_up"), "psi_up": r"\swap{1}", "psi_dn": r"\targX{}"}


def givens(label):
    return {"angles": ctrl("angles", "psi_dn"), "psi_dn": g(label, fill=ORANGE),
            "phase_grad": NABLA}


COLS = [  # (step label, cells)
    ("1", unif("succ_x_o", "m_x_o", "sigma_1")),
    ("2", {"x_o": g(IN, fill=GREY, down=R["angles"] - R["x_o"]),
           "alt_x_o": g(r"\mathrm{data}", 2, BLUE),
           "angles": g(r"\mathrm{junk}_{x_o}", fill=BLUE)}),
    ("3", comparator("m_x_o", "keep_x_o", "less_than_x_o", "cmp_work_x_o")),
    ("4", swap("x_o", "alt_x_o", "less_than_x_o")),
    ("J", {"angles": METER}),
    ("5", unif("succ_b", "m_b", "sigma_0")),
    ("6", {"x_o": g(IN, fill=GREY, down=R["angles"] - R["x_o"]), "b": g(IN, fill=GREY),
           "alt_b": g(r"\mathrm{data}", 4, BLUE), "r": g("r", fill=BLUE),
           "angles": g(r"\mathrm{junk}_{b}", fill=BLUE)}),
    ("7", comparator("m_b", "keep_b", "less_than_b", "cmp_work_b")),
    ("8", swap("b", "alt_b", "less_than_b")),
    ("J", {"angles": METER}),
    ("9", {"sgn": g("Z"), "d_sgn": ctrl("d_sgn", "less_than_b"), "less_than_b": CTL}),
    ("10", {"x_o": g(r"\mathrm{In}_{\geq RC}", fill=GREY, down=R["angles"] - R["x_o"]),
            "G0": g("G", 2, BLUE), "angles": g(r"\theta_{D_1Q_1}", fill=BLUE)}),
    ("11", {"b": g(IN, fill=GREY, down=R["b_neq_B"] - R["b"]), "r": g(IN, fill=GREY),
            "G0": OCTL, "angles": g(r"\theta_{SF}", fill=BLUE),
            "b_neq_B": g(r"b{\neq}B", fill=BLUE)}),
    ("12", mux()),
    ("13", spin_swap()),
    ("14", givens("R")),
    ("15", {"succ_x_o": ctrl("succ_x_o", "psi_dn"), "succ_b": CTL, "sigma_0": CTL, "G0": CTL,
            "G1": CTL, "b_neq_B": CTL, "psi_dn": g(r"\mathrm{Maj}", fill=ORANGE)}),
    ("16", givens(r"R^{\dagger}")),
    ("17", spin_swap()),
    ("17", mux()),
    ("18a", {"angles": METER, "b_neq_B": METER}),
    ("18b", {"b": g(INZ, fill=GREY, down=R["angles"] - R["b"]), "r": g(INZ, fill=GREY),
             "G0": OCTL, "angles": g(EPS, fill=GREY)}),
    ("18c", {"G0": METER, "G1": METER}),
    ("18d", {"x_o": g(INZ, fill=GREY)}),
    ("19", swap("b", "alt_b", "less_than_b")),
    ("20", comparator("m_b", "keep_b", "less_than_b", "cmp_work_b", dagger=True)),
    ("21a", {name: METER for name in ("alt_b", "keep_b", "sgn", "d_sgn", "r")}),
    ("21b", {"x_o": g(INZ, fill=GREY, down=R["angles"] - R["x_o"]), "b": g(INZ, fill=GREY),
             "alt_b": g(EPS, 4, GREY), "r": g(EPS, fill=GREY), "angles": g(EPS, fill=GREY)}),
    ("22", unif("succ_b", "m_b", "sigma_0", dagger=True)),
    ("23", {"succ_x_o": ctrl("succ_x_o", "b"), "b": g(r"\mathrm{REF}_{T_2}", 3, PINK)}),
]
SECTIONS = [  # (title, number of columns, fill)
    ("OuterPrepare", 5, "gray!16"),
    ("InnerPrepare", 5, "gray!4"),
    (r"\shortstack{Inner\\Signs}", 1, "gray!16"),
    ("InnerSelect", 13, "gray!9"),
    (r"\shortstack{InnerUnprepare\\(FromOutcomes)}", 5, "gray!4"),
    ("REF", 1, "red!10"),
]
assert sum(n for _, n, _ in SECTIONS) == len(COLS)

CIRCLED_LABELS = {"succ_work_x_o", "cmp_work_x_o", "succ_work_b", "cmp_work_b", "d_sgn",
                  "angles", "phase_grad"}
CIRCLES = [  # (column index, top row, rows, steps): spots that differ from the dfthcbe figure
    (1, "angles", 1, 1), (2, "cmp_work_x_o", 1, 1), (4, "angles", 1, 1),
    (6, "angles", 1, 1), (7, "cmp_work_b", 1, 1), (9, "angles", 1, 1),
    (10, "sgn", 3, 1),
    (20, "angles", 2, 1), (22, "G0", 2, 1), (23, "x_o", 1, 1),
    (25, "less_than_b", 2, 1),
    (26, "alt_b", 4, 1), (26, "r", 1, 1), (27, "angles", 1, 1),
]


def build() -> str:
    first_col = {}
    start = 0
    for title, n, fill in SECTIONS:
        first_col[start] = (title, n, fill)
        start += n
    lines, rings = [], []
    for row in range(1, NROWS + 1):
        label, bundle = ROWS[row - 1]
        name = [k for k, v in R.items() if v == row][0]
        if name in CIRCLED_LABELS:
            box = "ring" + "".join(chr(65 + int(ch)) if ch.isdigit() else ch
                                   for ch in name.replace("_", ""))
            rings.append(f"\\newsavebox{{\\{box}}}\\sbox{{\\{box}}}"
                         f"{{\\tikz[baseline=(l.base)]\\node[red_ring,ellipse](l){{{label}}};}}")
            label = f"\\usebox{{\\{box}}}"
        cells = [f"\\lstick{{{label}}}", f"\\qwbundle{{{bundle}}}" if bundle else "",
                 ""]
        for c, (num, col) in enumerate(COLS):
            cell = col.get(name, "")
            for cc, top, nrows, nsteps in CIRCLES:
                if cc == c and top == name:
                    cell += (f"\\gategroup[{nrows},steps={nsteps},"
                             r"style={red_ring,ellipse,inner xsep=-1pt,inner ysep=-2pt}]{}")
            if row == 1:
                cell += (r"\gategroup[1,steps=1,style={draw=none,inner ysep=0pt},"
                         r"label style={num}]{" + num + "}")
                if c in first_col:
                    title, n, fill = first_col[c]
                    cell += (f"\\gategroup[{NROWS},steps={n},style={{band={fill}}},background,"
                             f"label style={{sec}}]{{{title}}}")
            cells.append(cell)
        cells.append("")
        lines.append(" & ".join(cells))
    lines.append(r"\setwiretype{n} & & \push{\hspace{1.4cm}}" + " &" * (len(COLS) + 1))
    body = " \\\\\n".join(lines) + "\n"  # no trailing \\ (quantikz needs the last row's columns)
    head, circ = PREAMBLE.split(r"\newsavebox{\circbox}")
    return head + "\n".join(rings) + "\n" + r"\newsavebox{\circbox}" + circ + body + POSTAMBLE


PREAMBLE = r"""\documentclass[border=6pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{quantikz2,shapes.geometric}
\tikzset{
  red_ring/.style={draw=red, line width=0.5pt, solid, inner sep=0.5pt},
  band/.style={fill=#1, draw=none, inner xsep=1pt, inner ysep=14pt},
  num/.style={label position=above, anchor=south, yshift=0.02cm, font=\small},
  sec/.style={label position=above, anchor=south, yshift=0.05cm, font=\ttfamily\small,
              align=center},
}
\begin{document}
\newsavebox{\circbox}
\sbox{\circbox}{%
\begin{quantikz}[row sep={0.62cm,between origins}, column sep=0.22cm, wire types={q}]
"""

POSTAMBLE = r"""\end{quantikz}}
\begin{tabular}{@{}l@{}}
\usebox{\circbox}\\[2pt]
\parbox{\wd\circbox}{\small
Colours as in the dfthcbe figure: green flagged uniform preparation, blue data loads, yellow
comparators, grey index/erase boxes, orange rotations and Majorana gate, pink reflection.
Meters: $H$, measurement and reset (X-basis); the junk registers \texttt{junk\_x\_o}
($(K_1{-}1)d_1$) and \texttt{junk\_b} ($(K_2{-}1)d_2$) are the first wires of \texttt{angles},
X-measured in the columns J right after each prepare (Q-1 variant (b)); their outcomes feed
the fix-ups of the unprepares from outcomes (\texttt{junk\_x\_o}: steps 24--27, not drawn).
$\mathrm{In}_Z$: phase fix-up over the index from measurement parities; $\mathcal{E}$: its
one-hot on reset wires (step 21b: target $\|$ the first $2^{k_{e2}+w_b}-d_2$ wires of
\texttt{angles}). $\le^{\dagger}$: measured comparator uncompute (no Toffoli).
$\nabla$: \texttt{phase\_grad} used as catalyst. Not drawn: the shared \texttt{work} pool;
$\mathrm{REF}_{T_2}$ includes $\mathrm{GlobalPhase}(\pi)$. $X_o = N{+}RC$; $l_x$, $l_b$:
odd parts of $X_o$, $B{+}1$. Red circles: wires and gates that differ from the dfthcbe figure.}
\end{tabular}
\end{document}
"""

if __name__ == "__main__":
    with open("walk_circuit.tex", "w") as f:
        f.write(build())
