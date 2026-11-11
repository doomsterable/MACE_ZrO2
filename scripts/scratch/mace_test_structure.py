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
"""Relax a hand-built test structure with MACE and report where each Li ends up.

Default: structures/test_structures/mZrO2supercell_2Li_iV_Zr.xyz (2 Li around a Zr vacancy).
Each Li is folded into the primitive cell and matched to the Li_i sites of
mace_lii_site_scan.py (X1-X4, DFT pocket) and to the vacated Zr site.

  uv run --script scripts/scratch/mace_test_structure.py
  uv run --script scripts/scratch/mace_test_structure.py --structure other.xyz --isif 2
"""

import argparse
import io
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from mace_lii_site_scan import LI_O_CUTOFF, SITES, SUPERCELL, site_distance
from mace_relax import build_calculator, relax, setup_logging

VAC_ZR_CART = np.array([3.055801, 5.015491, 4.183448])  # Zr removed from the 2x2x2 host

log = logging.getLogger("test_structure")


def read_positions_only(path: Path) -> Atoms:
    """Read an extxyz whose Properties header lists more columns than the atom lines hold."""
    lines = path.read_text().splitlines()
    lines[1] = re.sub(r"Properties=\S+", "Properties=species:S:1:pos:R:3", lines[1])
    atoms = read(io.StringIO("\n".join(lines) + "\n"), format="extxyz")
    atoms.calc = None
    return atoms


def analyse(final: Atoms, start: Atoms, vac: np.ndarray = VAC_ZR_CART) -> list[dict[str, object]]:
    """Per-Li: nearest site, distance to the vacated Zr site, O coordination, displacement."""
    li = [a.index for a in final if a.symbol == "Li"]
    host = [a.index for a in final if a.symbol != "Li"]
    f, f0 = final.get_scaled_positions(wrap=False), start.get_scaled_positions(wrap=False)
    drift = (((f[host] - f0[host]) + 0.5) % 1 - 0.5).mean(axis=0)
    prim = final.cell.array / SUPERCELL[:, None]
    f_vac = start.cell.scaled_positions(vac[None])[0] + drift
    sym = np.array(final.get_chemical_symbols())

    rows = []
    for i in li:
        fp = (SUPERCELL * ((f[i] - drift) % 1)) % 1
        dists = {k: site_distance(fp, np.array(v), prim) for k, v in SITES.items()}
        dv = (f[i] - f_vac + 0.5) % 1 - 0.5
        dists["V_Zr"] = float(np.linalg.norm(dv @ final.cell.array))
        nearest = min(dists, key=dists.get)
        d_all = final.get_distances(i, range(len(final)), mic=True)
        d_o = np.sort(d_all[sym == "O"])
        moved = ((f[i] - f0[i] + 0.5) % 1 - 0.5) @ final.cell.array
        rows.append({
            "Li": i,
            "nearest_site": nearest,
            "d_nearest": dists[nearest],
            **{f"d_{k}": v for k, v in dists.items()},
            "prim_x": fp[0], "prim_y": fp[1], "prim_z": fp[2],
            "moved_A": float(np.linalg.norm(moved)),
            "CN_O": int((d_o < LI_O_CUTOFF).sum()),
            "Li_O_A": " ".join(f"{x:.2f}" for x in d_o[:6]),
            "Li_Zr_min_A": float(np.sort(d_all[sym == "Zr"])[0]),
            "Li_Li_A": float(final.get_distance(li[0], li[-1], mic=True)),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", type=Path,
                   default=ROOT / "structures" / "test_structures" / "mZrO2supercell_2Li_iV_Zr.xyz")
    p.add_argument("--model", default="medium-mpa-0")
    p.add_argument("--device", default="cpu")
    p.add_argument("--isif", type=int, nargs="+", default=[2, 3], help="2 = positions, 3 = positions + cell")
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_dir = ROOT / "results" / "test_structures" / args.model
    setup_logging(out_dir)
    start = read_positions_only(args.structure)
    log.info("Start: %s, %s, %d atoms", args.structure.name, start.get_chemical_formula(), len(start))

    calc = build_calculator(args.model, args.device)
    rows = []
    for isif in args.isif:
        log.info("Relaxing ISIF=%d", isif)
        final = relax(start, calc, out_dir / f"{args.structure.stem}_isif{isif}.xyz",
                      relax_cell=isif == 3, fmax=args.fmax, steps=args.steps, overwrite=args.overwrite)
        rows += [{"isif": isif, "E_mace": float(final.get_potential_energy()),
                  "converged": final.info.get("converged"), **r} for r in analyse(final, start)]

    df = pd.DataFrame(rows)
    csv = out_dir / f"{args.structure.stem}_li_sites.csv"
    df.to_csv(csv, index=False)
    cols = ["isif", "Li", "E_mace", "nearest_site", "d_nearest", "d_V_Zr", "d_DFT", "moved_A",
            "CN_O", "Li_O_A", "Li_Zr_min_A", "Li_Li_A", "prim_x", "prim_y", "prim_z", "converged"]
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        log.info("\n%s", df[cols].to_string(index=False))
    log.info("Wrote %s", csv)


if __name__ == "__main__":
    main()
