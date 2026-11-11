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
"""Relax the t-ZrO2 O_i split-dumbbell candidates (ZrO2/shared/interstitial/Oi_dumbbells) with MACE.

Each POSCAR (3x3x2 tetragonal supercell + 1 extra O) has the DFT lattice, so its
fractional coordinates are mapped onto the MACE-relaxed t-ZrO2 host supercell
(results/pbe_recreation/<model>/reference/tZrO2_supercell.xyz) before relaxing, with the
same relax() / settings as mace_relax.py:

  ISIF=2  positions only, at the MACE host cell
  ISIF=3  positions + cell

Reports the final O-O dumbbell distance, whether the dumbbell survived (the pair did not
fall apart into a regular interstitial), and E relative to the lowest candidate.
Needs mace_relax.py to have produced the tZrO2 host supercell. Output goes to
results/oi_dumbbells/<model>/ (xyz files + oi_dumbbells.csv).

  uv run --script mace_oi_dumbbells.py
  uv run --script mace_oi_dumbbells.py --replot-only   # rebuild the CSV from existing xyz
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import get_distances
from ase.io import read

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mace_relax import build_calculator, relax, setup_logging

log = logging.getLogger("mace_oi_dumbbells")

DUMBBELL_DIR = HERE.parents[1] / "ZrO2" / "shared" / "interstitial" / "Oi_dumbbells"
ISIFS = (2, 3)
DUMBBELL_MAX = 1.8  # A, O-O distance below which the pair still counts as a dumbbell


def load_start(poscar: Path, host: Atoms) -> Atoms:
    """Read a dumbbell POSCAR and put its fractional coordinates on the MACE host cell."""
    atoms = read(poscar)
    atoms.set_cell(host.cell, scale_atoms=True)
    return atoms


def oo_distance(atoms: Atoms, dumbbell_o: np.ndarray | None = None) -> float:
    """Shortest O-O distance (minimum image) in the cell."""
    idx = [a.index for a in atoms if a.symbol == "O"]
    d = get_distances(atoms.positions[idx], cell=atoms.cell, pbc=True)[1]
    return float(d[np.triu_indices(len(idx), k=1)].min())


def collect(out_dir: Path, csv: Path) -> pd.DataFrame:
    """Tabulate every relaxed dumbbell (all data needed to replot)."""
    rows = []
    for path in sorted(out_dir.glob("*.xyz")):
        atoms = read(path)
        name, isif = path.stem.removeprefix("tZrO2_Oi_").rsplit("_isif", 1)
        d_oo = oo_distance(atoms)
        rows.append({
            "dumbbell": name, "isif": int(isif.removesuffix("_q0")), "n_atoms": len(atoms),
            "E_mace": float(atoms.get_potential_energy()), "d_OO": d_oo, "is_dumbbell": d_oo < DUMBBELL_MAX,
            "volume": float(atoms.get_volume()), "steps": atoms.info.get("steps"),
            "converged": atoms.info.get("converged"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["dE_vs_lowest"] = df["E_mace"] - df.groupby("isif")["E_mace"].transform("min")
        df = df.sort_values(["isif", "dE_vs_lowest"])
    df.to_csv(csv, index=False)
    log.info("Wrote %d rows to %s\n%s", len(df), csv, df.to_string(index=False))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="medium-mpa-0", help="MACE-MP model name")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--dtype", default="float64", help="float64 or float32")
    parser.add_argument("--fmax", type=float, default=0.01, help="force tolerance (eV/A)")
    parser.add_argument("--steps", type=int, default=1000, help="max optimiser steps per relaxation")
    parser.add_argument("--overwrite", action="store_true", help="rerun even if the output exists")
    parser.add_argument("--replot-only", action="store_true", help="skip MACE, rebuild the CSV from existing files")
    args = parser.parse_args()

    root = HERE.parent / "results"
    out_dir = root / "oi_dumbbells" / args.model
    setup_logging(out_dir)
    csv = out_dir / "oi_dumbbells.csv"
    if args.replot_only:
        collect(out_dir, csv)
        return

    host = read(root / "pbe_recreation" / args.model / "reference" / "tZrO2_supercell.xyz")
    poscars = sorted(DUMBBELL_DIR.glob("*/POSCAR"))
    log.info("Found %d dumbbells in %s", len(poscars), DUMBBELL_DIR)

    calc = build_calculator(args.model, args.device, args.dtype)
    for poscar in poscars:
        start = load_start(poscar, host)
        log.info("%s: %d atoms (host %d + 1 O)", poscar.parent.name, len(start), len(host))
        for isif in ISIFS:
            relax(start, calc, out_dir / f"tZrO2_Oi_{poscar.parent.name}_isif{isif}_q0.xyz", relax_cell=isif == 3,
                  fmax=args.fmax, steps=args.steps, overwrite=args.overwrite)
    collect(out_dir, csv)


if __name__ == "__main__":
    main()
