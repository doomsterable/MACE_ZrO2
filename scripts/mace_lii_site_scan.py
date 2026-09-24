# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mace-torch",
#   "ase",
#   "numpy",
#   "pandas",
#   "graph-longrange",
# ]
#
# [tool.uv.sources]
# graph-longrange = { git = "https://github.com/WillBaldwin0/graph_electrostatics.git", tag = "v0.4.0" }
# ///
"""Li_i site scan in m-ZrO2 with MACE: where does Li end up from each candidate site?

Puts one Li on the MACE-relaxed 2x2x2 host (from mace_relax.py) at each candidate
interstitial site, relaxes it with the same relax() / settings as mace_relax.py,
then folds the final Li back into the primitive cell and reports which site it
landed on (under P2_1/c symmetry), how far it moved, its O coordination and the
energy relative to the lowest start.

Sites (primitive-cell frac, standard P2_1/c setting = the mp-2858 setting used by
species.py):
  X1  2c  (0, 0, 1/2)                 cavity centre, inversion centre
  X2  4e  (0.11747, 0.97376, 0.51247) cavity, off-centre
  X3  4e  (0.77435, 0.11224, 0.30292) tetrahedral side pocket
  X4  4e  (0.37105, 0.01149, 0.72367) 5 O but 1.82 A from Zr (low priority)
  DFT 4e  (0.90925, 0.05082, 0.44975) r2SCAN/PBE Li_i q=+1 ISIF=3 final position

The existing MACE Li_i relaxations from mace_relax.py (mZrO2_Lii_isif*_q0.xyz) are
analysed too, if present, so you can see where the defects.py site went.

Needs mace_relax.py (same folder) and results/relaxation/medium-mpa-0/mZrO2_supercell.xyz to exist.

  uv run --script mace_lii_site_scan.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mace_relax import build_calculator, relax, setup_logging

# ----------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------
MODEL = "medium-mpa-0"
DEVICE = "cpu"          # cpu, cuda or mps
DTYPE = "float64"
FMAX = 0.01             # eV/A, same as mace_relax.py
STEPS = 1000
ISIFS = (2, 3)          # 2 = positions only at the host cell, 3 = positions + cell
RATTLE = 0.05           # A, random kick to the Li only (needed to get X1 off the inversion centre)
SEED = 42
OVERWRITE = False       # rerun relaxations whose output already exists

DATA = HERE.parent / "results" / "relaxation" / MODEL
HOST_FILE = DATA / "mZrO2_supercell.xyz"
OUT_DIR = DATA / "Lii_site_scan"
SUPERCELL = np.array([2, 2, 2])

# Starting Li positions, 2x2x2 supercell fractional coordinates. For each site this is
# the image in the same cavity where the DFT Li_i ended up, so every run starts in the
# same neighbourhood and "moved to another site" is not just a different image.
STARTS: dict[str, tuple[float, float, float]] = {
    "X1_2c_cavity_centre":    (0.50000, 0.50000, 0.75000),
    "X2_4e_cavity_offcentre": (0.44126, 0.51312, 0.74376),
    "X3_4e_tet_pocket":       (0.38718, 0.55612, 0.65146),
    "X4_4e_5O_nearZr":        (0.31448, 0.49425, 0.63817),
    "DFT_relaxed":            (0.45462, 0.52541, 0.72488),
    "DFT_original_start":     (0.40789, 0.53806, 0.67356),
}

# One representative per site, primitive-cell fractional coordinates.
SITES: dict[str, tuple[float, float, float]] = {
    "X1": (0.00000, 0.00000, 0.50000),
    "X2": (0.11747, 0.97376, 0.51247),
    "X3": (0.77435, 0.11224, 0.30292),
    "X4": (0.37105, 0.01149, 0.72367),
    "DFT": (0.90925, 0.05082, 0.44975),
}

# P2_1/c (unique axis b, cell choice 1)
OPS = [
    (np.diag([1, 1, 1]), np.array([0.0, 0.0, 0.0])),
    (np.diag([-1, -1, -1]), np.array([0.0, 0.0, 0.0])),
    (np.diag([-1, 1, -1]), np.array([0.0, 0.5, 0.5])),
    (np.diag([1, -1, 1]), np.array([0.0, 0.5, 0.5])),
]

LI_O_CUTOFF = 2.4       # A, for the Li-O coordination number

log = logging.getLogger("lii_site_scan")


# ----------------------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------------------
def build_start(host: Atoms, frac: tuple[float, float, float], rng: np.random.Generator) -> Atoms:
    """Host supercell + one Li (appended last) at supercell frac, Li rattled by RATTLE."""
    atoms = host.copy()
    atoms.calc = None
    li = Atoms("Li", scaled_positions=[frac], cell=host.cell, pbc=True)
    if RATTLE > 0:
        kick = rng.normal(size=3)
        li.positions += RATTLE * kick / np.linalg.norm(kick)
    atoms += li
    return atoms


# ----------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------
def li_index(atoms: Atoms) -> int:
    idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Li"]
    if len(idx) != 1:
        raise ValueError(f"expected one Li, found {len(idx)}")
    return idx[0]


def fold_to_primitive(atoms: Atoms, host: Atoms) -> np.ndarray:
    """Li frac in the primitive cell, with any rigid drift of the host removed."""
    li = li_index(atoms)
    f = atoms.get_scaled_positions(wrap=False)
    others = [i for i in range(len(atoms)) if i != li]
    drift = np.zeros(3)
    if [atoms[i].symbol for i in others] == host.get_chemical_symbols():
        d = f[others] - host.get_scaled_positions(wrap=False)
        drift = ((d + 0.5) % 1 - 0.5).mean(axis=0)
    return (SUPERCELL * ((f[li] - drift) % 1)) % 1


def site_distance(f: np.ndarray, site: np.ndarray, prim_cell: np.ndarray) -> float:
    """Shortest distance (A) from f to any P2_1/c image of site."""
    best = np.inf
    for rot, t in OPS:
        g = (rot @ site + t) % 1
        d = (f - g + 0.5) % 1 - 0.5
        best = min(best, float(np.linalg.norm(d @ prim_cell)))
    return best


def analyse(name: str, isif: int, atoms: Atoms, host: Atoms, start: Atoms | None) -> dict[str, object]:
    li = li_index(atoms)
    prim = atoms.cell.array / SUPERCELL[:, None]
    f = fold_to_primitive(atoms, host)
    dists = {k: site_distance(f, np.array(v), prim) for k, v in SITES.items()}
    nearest = min(dists, key=dists.get)

    d_all = atoms.get_distances(li, range(len(atoms)), mic=True)
    sym = np.array(atoms.get_chemical_symbols())
    d_o = np.sort(d_all[sym == "O"])
    d_zr = np.sort(d_all[sym == "Zr"])

    moved = np.nan
    if start is not None:
        df = atoms.get_scaled_positions(wrap=False)[li] - start.get_scaled_positions(wrap=False)[li_index(start)]
        moved = float(np.linalg.norm(((df + 0.5) % 1 - 0.5) @ atoms.cell.array))

    return {
        "start": name,
        "isif": isif,
        "E_mace": float(atoms.get_potential_energy()),
        "prim_x": f[0], "prim_y": f[1], "prim_z": f[2],
        "nearest_site": nearest,
        "d_nearest": dists[nearest],
        **{f"d_{k}": v for k, v in dists.items()},
        "moved_A": moved,
        "CN_O": int((d_o < LI_O_CUTOFF).sum()),
        "Li_O_A": " ".join(f"{x:.2f}" for x in d_o[:6]),
        "Li_Zr_min_A": float(d_zr[0]),
        "converged": atoms.info.get("converged"),
        "steps": atoms.info.get("steps"),
    }


# ----------------------------------------------------------------------------------
def main() -> None:
    setup_logging(OUT_DIR)
    if not HOST_FILE.exists():
        raise SystemExit(f"{HOST_FILE} not found: run mace_relax.py first")
    host = read(HOST_FILE)
    host.calc = None
    log.info("Host: %s, %d atoms", HOST_FILE.name, len(host))

    rng = np.random.default_rng(SEED)
    starts = {name: build_start(host, frac, rng) for name, frac in STARTS.items()}
    (OUT_DIR / "start").mkdir(parents=True, exist_ok=True)
    for name, atoms in starts.items():
        write(OUT_DIR / "start" / f"mZrO2_222_Lii_{name}.cif", atoms)

    calc = build_calculator(MODEL, DEVICE, DTYPE)
    rows = []
    for isif in ISIFS:
        for name, atoms in starts.items():
            log.info("Li_i from %s, ISIF=%d", name, isif)
            final = relax(atoms, calc, OUT_DIR / f"mZrO2_Lii_{name}_isif{isif}.xyz",
                          charge=0, relax_cell=isif == 3, fmax=FMAX, steps=STEPS, overwrite=OVERWRITE)
            rows.append(analyse(name, isif, final, host, atoms))

        existing = DATA / f"mZrO2_Lii_isif{isif}_q0.xyz"
        if existing.exists():
            rows.append(analyse("defects.py_Li_i (existing run)", isif, read(existing), host, None))

    df = pd.DataFrame(rows)
    df["dE_eV"] = df["E_mace"] - df.groupby("isif")["E_mace"].transform("min")
    df = df.sort_values(["isif", "dE_eV"])
    df.to_csv(OUT_DIR / "lii_site_scan.csv", index=False)

    cols = ["isif", "start", "dE_eV", "nearest_site", "d_nearest", "d_DFT", "moved_A",
            "CN_O", "Li_O_A", "Li_Zr_min_A", "prim_x", "prim_y", "prim_z", "converged"]
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        log.info("\n%s", df[cols].to_string(index=False))
    log.info("Wrote %s", OUT_DIR / "lii_site_scan.csv")


if __name__ == "__main__":
    main()
