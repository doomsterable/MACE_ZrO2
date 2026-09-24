# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ase",
#   "openpyxl",
# ]
# ///
"""Build MACE_ZrO2_formation_energies.xlsx in the layout of r2scanZrO2_formation_energies.xlsx.

Sheets, row numbers and formulas mirror the r2SCAN workbook (Inputs!B4 host,
B25 mu_O, B26 mu_Zr, B28 E_VBM, B30 gap, B53 E_corr coefficient), so the two can
be read side by side. Differences, all forced by the data:

  * O-site defects are split into the 3-fold / 4-fold sites (V_O3, V_O4, ...),
    exactly as the PBE_ZrO2 defect registry builds them.
  * Only q=0 is filled: MACE-MP has no notion of charge. Charged rows are kept
    (empty E_tot) so a charge-aware model can be dropped in later.
  * E_VBM / E_CBM / gap cannot come from MACE; they are copied from the r2SCAN
    workbook and only matter for q != 0.

Only E_tot cells and reference energies are numbers; everything derived is a formula.

  uv run --script build_workbook.py
"""

import logging
import sys
from pathlib import Path

from ase.io import read
from openpyxl import Workbook
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PBE_ZrO2" / "src"))
from defects import DEFECTS, Defect, Job

log = logging.getLogger("build_workbook")

# ── styles copied from the r2SCAN workbook ──────────────────────────────────
THIN = Side(style="thin", color="FFBFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
INPUT = {"font": Font(name="Arial", color="FF0000FF"), "fill": PatternFill("solid", fgColor="FFFFF2CC"), "border": BOX}
NOTE = {"font": Font(name="Arial", italic=True, size=9, color="FF808080")}
SECTION = {"font": Font(name="Calibri", bold=True, size=12), "fill": PatternFill("solid", fgColor="FFA5A5A5"),
           "border": Border(bottom=Side(style="double"))}
HEADER = {"font": Font(name="Arial", bold=True, color="FFFFFFFF"), "fill": PatternFill("solid", fgColor="FF2F5597"),
          "border": BOX, "alignment": Alignment(horizontal="center", vertical="center", wrap_text=True)}
ENV_HEADER = HEADER | {"fill": PatternFill("solid", fgColor="FF548235")}
FINAL_HEADER = HEADER | {"fill": PatternFill("solid", fgColor="FF000000")}
TEXT = {"font": Font(name="Arial", color="FF000000"), "border": BOX}
RESULT = {"font": Font(name="Arial", color="FF008000"), "border": BOX, "number_format": "0.0000"}
STATUS = {"font": Font(name="Arial", size=9, color="FF808080"), "border": BOX}
TODO = {"font": Font(name="Arial", size=9, color="FFC00000"), "border": BOX}

# Row order of the ISIF sheets: same families and order as the r2SCAN workbook.
ORDER = ("V_O3", "V_O4", "V_Zr", "K_O3", "K_O4", "Li_O3", "Li_O4", "K_Zr", "Li_Zr", "K_i", "Li_i", "Zr_i", "O_i")
DN: dict[str, tuple[int, int, int]] = {  # (dn_Zr, dn_O, dn_dopant)
    "V_O": (0, -1, 0), "V_Zr": (-1, 0, 0), "K_O": (0, -1, 1), "Li_O": (0, -1, 1), "K_Zr": (-1, 0, 1),
    "Li_Zr": (-1, 0, 1), "K_i": (0, 0, 1), "Li_i": (0, 0, 1), "Zr_i": (1, 0, 0), "O_i": (0, 1, 0),
}
N_EF = 40  # E_F grid: rows 5..45 of Lines_vs_EF, as in the r2SCAN workbook


def style(cell: object, spec: dict[str, object] | None = None, **extra: object) -> None:
    """Apply a style dict (font/fill/border/alignment/number_format) to a cell."""
    for key, value in ((spec or {}) | extra).items():
        setattr(cell, key, value)


def put(ws: Worksheet, ref: str = "A1", value: object = None, spec: dict[str, object] | None = None,
        **extra: object) -> None:
    """Write a value and its style in one call."""
    ws[ref] = value
    style(ws[ref], spec, **extra)


def family(defect: Defect) -> str:
    """V_O3 -> V_O, K_O4 -> K_O; others unchanged."""
    return defect.key[:-1] if defect.key[-1] in "34" else defect.key


def defect_type(defect: Defect) -> str:
    """Donor/acceptor label in the r2SCAN wording."""
    kind = "donor" if max(defect.charges) > 0 else "acceptor"
    return f"{kind} (interstitial)" if defect.kind == "interstitial" else kind


def energy(path: Path) -> tuple[float, bool] | None:
    """(E_tot, converged) from a MACE extxyz, or None if the file does not exist."""
    if not path.exists():
        return None
    atoms = read(path)
    return float(atoms.get_potential_energy()), bool(atoms.info.get("converged", True))


def build_inputs(ws: Worksheet, data: Path = Path("."), model: str = "medium-mpa-0") -> None:
    """Reference energies and chemical potentials, same rows as the r2SCAN Inputs sheet."""
    e = {k: energy(data / f"{k}.xyz") for k in ("mZrO2_supercell", "tZrO2_supercell", "mZrO2_unitcell",
                                                 "tZrO2_unitcell", "aZr_unitcell", "K2ZrO3_unitcell",
                                                 "K4Zr5O12_unitcell", "Li2ZrO3_unitcell")}
    missing = [k for k, v in e.items() if v is None]
    assert not missing, f"Missing MACE reference structures: {missing}"
    tot = {k: v[0] for k, v in e.items() if v is not None}
    per_atom = {k: tot[k] / len(read(data / f"{k}.xyz")) for k in tot}
    cell = read(data / "mZrO2_supercell.xyz").cell.lengths()

    put(ws, "A1", f"Inputs & reference energies — MACE-MP ({model})", font=Font(name="Arial", bold=True, size=13))
    put(ws, "A2", "Edit any blue cell — the ISIF=3, ISIF=2 and Lines sheets all recalculate.",
        font=Font(name="Arial", italic=True, color="FF808080"))
    rows: list[tuple[int, str, object, str]] = [
        (4, "E_tot pristine, m-ZrO2 ", tot["mZrO2_supercell"], "MACE-relaxed 2x2x2 (96 atoms). Reference for all m-ZrO2 rows."),
        (5, "E_tot pristine, t-ZrO2 ", tot["tZrO2_supercell"], "MACE-relaxed 3x3x2 (108 atoms). Reference for t-ZrO2 rows."),
        (6, "mu_mZrO2 (bulk) per atom", per_atom["mZrO2_unitcell"], "MACE-relaxed unit cell (12 atoms)"),
        (7, "mu_tZrO2 (bulk) per atom", per_atom["tZrO2_unitcell"], "MACE-relaxed unit cell (6 atoms)"),
        (8, "mu_Zr (bulk)", per_atom["aZr_unitcell"], "a-Zr per atom, MACE-relaxed hcp cell"),
        (9, "mu_K (final)", "=MIN(B11,B12)", "Limiting K phase = the lower of the two equilibria below"),
        (10, "mu_Li (final)", "=B13", "Li2ZrO3 equilibrium"),
        (11, "mu_K (K2ZrO3)", f"=0.5*({tot['K2ZrO3_unitcell']:.6f}/4-B6*3-B25)",
         "μ_K in 2-phase equilibrium with K2ZrO3 (MACE cell = 4 f.u., 24 atoms)"),
        (12, "mu_K (K4Zr5O12)", f"=({tot['K4Zr5O12_unitcell']:.6f}-B6*15-B25*2)/4",
         "μ_K in 2-phase equilibrium with K4Zr5O12 (MACE cell = 1 f.u., 21 atoms)"),
        (13, "mu_Li (Li2ZrO3)", f"=0.5*({tot['Li2ZrO3_unitcell']:.6f}/4-B6*3-B25)",
         "μ_Li in 2-phase equilibrium with Li2ZrO3 (MACE cell = 4 f.u., 24 atoms)"),
        (14, "ΔG_f mZrO2 per atom", -3.814, "Materials Project mp-2858 (same value as the r2SCAN workbook)"),
        (15, "mu_O (T⁰,pO₂⁰)", "=($B6*3-$B8-$B14*3)/2", "mu_ZrO2 per atom x 3 - mu_Zr - deltaG per atom x 3"),
        (17, "k_B (eV/K)", 8.617333e-05, "Boltzmann constant"),
        (18, "S°(O₂) (eV/K)", "=205.152/96485.33212", "Entropy at standard conditions"),
        (19, "T° (K)", 298.15, "Temperature at standard conditions"),
        (20, "C_P° (eV/K)", "=B17*3.5", "Rigid diatomic: 5 translational+rotational dof + 1 extra from pV work"),
        (21, "T (K)", 635, "CHANGE THESE FOR DIFFERENT O2 POTENTIALS! (Gareth's default: 635K)"),
        (22, "log₁₀(pO₂ / atm)", -25, "CHANGE THESE FOR DIFFERENT O2 POTENTIALS! (Gareth's default: -25)"),
        (23, "Δμ_O⁰(T) thermal (eV)", "=-0.5*((B18-B20)*(B21-B19)+B20*B21*LN(B21/B19))", "Thermal term of the O potential"),
        (24, "pressure term (eV)", "=0.5*B17*B21*B22*LN(10)", "0.5 kB T ln(pO2/p⁰)"),
        (25, "mu_O (T,pO₂)", "=B15+B23+B24", "Previous two terms plus the reference oxygen"),
        (26, "mu_Zr (T,pO₂) ", "=$B6*3-$B25*2", "mu_ZrO2 minus 2 x the T/pO2-dependent mu_O"),
        (28, "E_VBM", 5.6229, "NOT FROM MACE (no electronic structure): copied from the r2SCAN workbook. Only enters q≠0 rows."),
        (29, "E_CBM", 9.254313, "As above."),
        (30, "Band gap", 3.6, "As above. Sets the E_F range on Lines_vs_EF."),
        (46, "Dielectric constant  eps", 19.7, "Monoclinic ZrO2, Zhao & Vanderbilt (~19.7)."),
        (47, "a  (Å)", float(cell[0]), "MACE-relaxed m-ZrO2 supercell"),
        (48, "b  (Å)", float(cell[1]), "MACE-relaxed m-ZrO2 supercell"),
        (49, "c  (Å)", float(cell[2]), "MACE-relaxed m-ZrO2 supercell"),
        (50, "L = (a·b·c)^(1/3)  (Å)", "=(B47*B48*B49)^(1/3)", "Effective supercell size (cube-root of cell volume)."),
        (51, "Madelung constant  alpha", 2.837297, "Cubic-lattice Madelung constant (Makov–Payne)."),
        (52, "e^2/4·pi·eps0  (eV·Å)", 14.4, ""),
        (53, "E_corr coefficient  C = alpha·(e^2/4·pi·eps0)/(2·eps·L)   (eV per q^2)", "=B51*B52/(2*B46*B50)", ""),
    ]
    for r, label, value, note in rows:
        put(ws, f"A{r}", label, font=Font(name="Arial", bold=True))
        put(ws, f"B{r}", value, INPUT, number_format="0.000000")
        put(ws, f"C{r}", note, NOTE)
    for r, title in ((3, "MACE reference potentials"), (16, "Calculating pressure/temperature dependent oxygen potential"),
                     (27, "Band edges (for charged states only)"), (32, "The formula"),
                     (44, "Finite-size correction  (Makov–Payne monopole / screened-Madelung term)")):
        put(ws, f"A{r}", title, SECTION)
    text = [
        (33, "E_form(D,q,E_F) = E_tot(defect) - E_tot(pristine) - SUM_i [ dn_i * mu_i ] + q*(E_VBM + E_F) + E_corr"),
        (35, "  dn_i    atoms of species i added(+)/removed(-):  dn_Zr, dn_O, dn_K/Li"),
        (36, "  mu_i    chemical potential (reservoir energy) of species i"),
        (37, "  q       charge state;   E_F  Fermi level measured from VBM (0 -> gap)"),
        (38, "  E_VBM   puts the electron reservoir on an absolute scale"),
        (39, "  E_corr  Makov-Payne monopole term C*q^2 (B53). Zero for every row filled here (q = 0)."),
        (41, "MACE-MP is charge-blind, so only neutral defects are computed: every line on Lines_vs_EF is flat."),
    ]
    for r, line in text:
        put(ws, f"A{r}", line, font=Font(name="Consolas", size=9))
    put(ws, "A54", "→ E_corr(q) = C · q^2   (eV)", font=Font(name="Consolas", size=9, italic=True, color="FF808080"))
    for col, width in (("A", 30), ("B", 15), ("C", 70)):
        ws.column_dimensions[col].width = width


def build_isif(ws: Worksheet, isif: int = 3, data: Path = Path("."), model: str = "medium-mpa-0") -> dict[str, list[int]]:
    """One ISIF sheet. Returns {defect key: [row, ...]} for the m-ZrO2 block (charge order)."""
    put(ws, "A1", f"ISIF = {isif}  —  MACE-MP ({model}) defect formation energies  (mu_O from Inputs!B25; E_F=0; q=0 only)",
        font=Font(name="Arial", bold=True, size=12))
    for col, head in zip("ABCDEFGHIJK", ("Phase", "Defect", "Type", "q", "dn_Zr", "dn_O", "dn_K/Li",
                                         "E_tot defect (eV)", "E_form @ E_F=0 (eV)", "E_form @ CBM (eV)", "Status")):
        put(ws, f"{col}3", head, HEADER)
    ws.row_dimensions[3].height = 31

    rows: dict[str, list[int]] = {}
    r = 4
    for host, phase in (("mZrO2", "m-ZrO2"), ("tZrO2", "t-ZrO2")):
        for key in ORDER:
            d = DEFECTS[key]
            mu = "$B$10" if d.element == "Li" else "$B$9"
            for q in d.charges:
                result = energy(data / f"{Job(d, isif, q, host).tag}.xyz")
                for col, value in zip("ABC", (phase, key, defect_type(d))):
                    put(ws, f"{col}{r}", value, TEXT)
                for col, value in zip("DEFG", (q, *DN[family(d)])):
                    put(ws, f"{col}{r}", value, INPUT)
                put(ws, f"H{r}", result[0] if result else None, INPUT, number_format="0.00000")
                put(ws, f"I{r}", f'=IF($H{r}="","",($H{r}-IF($A{r}="m-ZrO2",Inputs!$B$4,Inputs!$B$5))'
                                 f"-($E{r}*Inputs!$B$26+$F{r}*Inputs!$B$25+$G{r}*Inputs!{mu})"
                                 f"+$D{r}*Inputs!$B$28+Inputs!$B$53*$D{r}^2)", RESULT)
                put(ws, f"J{r}", f'=IF($H{r}="","",$I{r}+$D{r}*Inputs!$B$30)', RESULT)
                if result:
                    put(ws, f"K{r}", "Converged" if result[1] else "NOT converged", STATUS if result[1] else TODO)
                elif q != 0 and host == "mZrO2":
                    put(ws, f"K{r}", "Not run (MACE-MP is charge-blind)", STATUS)
                else:
                    put(ws, f"K{r}", "Not started", TODO)
                if host == "mZrO2":
                    rows.setdefault(key, []).append(r)
                r += 1
    for col, width in (("A", 8), ("B", 8), ("C", 20), ("D", 5), ("E", 7), ("F", 7), ("G", 8),
                       ("H", 17), ("I", 18), ("J", 17), ("K", 32)):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A4"
    return rows


def build_lines(ws: Worksheet, rows: dict[str, list[int]]) -> tuple[list[int], int]:
    """E_form vs E_F from the m-ZrO2 ISIF=3 rows. Returns (envelope columns, Final envelope column)."""
    put(ws, "A1", "E_form vs E_F  (each column = one charge state = a straight line; envelope = MIN)",
        font=Font(name="Arial", bold=True, size=12))
    put(ws, "A2", "Built from m-ZrO2 ISIF=3 states. MACE-MP gives q=0 only, so charged columns are blank and lines are flat.",
        font=Font(name="Arial", italic=True, color="FF808080"))
    put(ws, "A4", "E_F (eV)", HEADER)
    last = 4 + N_EF + 1
    for i in range(N_EF + 1):
        put(ws, f"A{5 + i}", f"=Inputs!$B$30*{i}/{N_EF}", TEXT, number_format="0.0000")

    col = 2
    spans: dict[str, tuple[str, str]] = {}
    for key, key_rows in rows.items():
        first = col
        for r, q in zip(key_rows, DEFECTS[key].charges):
            letter = get_column_letter(col)
            put(ws, f"{letter}4", f"{key} q{q:+d}", HEADER)
            for rr in range(5, last + 1):
                put(ws, f"{letter}{rr}", f"=IF('ISIF = 3'!$I${r}=\"\",\"\",'ISIF = 3'!$I${r}+'ISIF = 3'!$D${r}*$A{rr})",
                    TEXT, number_format="0.000")
            col += 1
        spans[key] = (get_column_letter(first), get_column_letter(col - 1))

    envelopes: list[int] = []
    for key, (a, b) in spans.items():
        letter = get_column_letter(col)
        put(ws, f"{letter}4", f"{key} envelope", ENV_HEADER)
        for rr in range(5, last + 1):
            put(ws, f"{letter}{rr}", f"=MIN({a}{rr}:{b}{rr})", TEXT, number_format="0.000", font=Font(name="Arial", bold=True))
        ws.column_dimensions[letter].width = 12
        envelopes.append(col)
        col += 1

    final = get_column_letter(col)
    first_env, last_env = get_column_letter(envelopes[0]), get_column_letter(envelopes[-1])
    put(ws, f"{final}4", "Final envelope", FINAL_HEADER)
    for rr in range(5, last + 1):
        put(ws, f"{final}{rr}", f"=MIN({first_env}{rr}:{last_env}{rr})", TEXT, number_format="0.000",
            font=Font(name="Arial", bold=True), fill=PatternFill("solid", fgColor="FFF2F2F2"))
    ws.column_dimensions[final].width = 12
    ws.row_dimensions[4].height = 46
    return envelopes, col


def envelope_chart() -> ScatterChart:
    """Per-defect envelopes + thick black Final envelope, as in the r2SCAN 'lower envelope' chart."""
    chart = ScatterChart()
    chart.title = "m-ZrO2 defect formation energies (lower envelope) vs E_F — MACE-MP"
    chart.style = 2
    chart.x_axis.title = "E_F above VBM (eV)"
    chart.y_axis.title = "E_form (eV)"
    chart.x_axis.delete = chart.y_axis.delete = False
    chart.height, chart.width = 12, 22
    return chart


def add_series(chart: ScatterChart, ws: Worksheet, columns: list[int], final: int, n_rows: int = N_EF + 1) -> None:
    """Attach one line series per envelope column; the Final envelope is thick and black."""
    x = Reference(ws, min_col=1, min_row=5, max_row=4 + n_rows)
    for col in (*columns, final):
        s = Series(Reference(ws, min_col=col, min_row=4, max_row=4 + n_rows), x, title_from_data=True)
        s.marker.symbol = "none"
        s.smooth = False
        if col == final:
            s.graphicalProperties.line.solidFill = "000000"
            s.graphicalProperties.line.width = 38100  # 3 pt
        chart.series.append(s)


def main(model: str = "medium-mpa-0") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    data = ROOT / "MACE_ZrO2" / "results" / "relaxation" / model
    out = ROOT / "MACE_ZrO2" / "results" / "MACE_ZrO2_formation_energies.xlsx"
    log.info("Reading MACE structures from %s", data)

    wb = Workbook()
    inputs = wb.active
    inputs.title = "Inputs"
    build_inputs(inputs, data, model)
    rows = build_isif(wb.create_sheet("ISIF = 3"), 3, data, model)
    build_isif(wb.create_sheet("ISIF = 2"), 2, data, model)
    lines = wb.create_sheet("Lines_vs_EF")
    envelopes, final = build_lines(lines, rows)

    for ws, anchor in ((lines, f"{get_column_letter(final + 2)}10"), (inputs, "E2")):
        chart = envelope_chart()
        add_series(chart, lines, envelopes, final)
        ws.add_chart(chart, anchor)

    wb.save(out)
    filled = sum(1 for ws in (wb["ISIF = 3"], wb["ISIF = 2"]) for c in ws["H"][3:] if c.value is not None)
    log.info("Filled %d E_tot cells; %d envelope columns + Final envelope", filled, len(envelopes))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
