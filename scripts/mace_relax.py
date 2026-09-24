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
"""MACE analogue of the PBE/r2SCAN relaxation stages 2-4.

Structures are built with the SAME code as the DFT repos (PBE_ZrO2/src/species.py
and defects.py, identical in r2scanZrO2), so every MACE starting geometry matches
the DFT one exactly:

  1. unit cells   all six species from the MP CIFs, relaxed at constant pressure
  2. supercells   hosts tiled (m 2x2x2, t 3x3x2) from the MACE unit cells, relaxed
  3. defects      every (defect, ISIF, charge) on the relaxed host supercell
                  ISIF=3 -> positions + cell; ISIF=2 -> positions at the host cell

MACE-MP has no notion of charge, so for it only q=0 is run; a 'polar-*' model gets
every charge state in the registry via atoms.info["charge"].

Unit cells and supercells go to results/pbe_recreation/<model>/reference/, defects to
.../defects/. Finished structures are skipped on rerun (extxyz with MACE energy/forces/stress),
and --collect-only rebuilds the CSV from those files without running MACE.

  uv run --script mace_relax.py
  uv run --script mace_relax.py --model polar-1-m --host tZrO2
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "PBE_ZrO2" / "src"))
from defects import Job, jobs, resolve_defects
from species import SPECIES, resolve

log = logging.getLogger("mace_relax")


def setup_logging(out_dir: Path = Path(".")) -> None:
    """Log to the console and to out_dir/mace_relax.log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "mace_relax.log")],
    )


def build_calculator(model: str = "medium-mpa-0", device: str = "cpu", dtype: str = "float64") -> Calculator:
    """Load a foundation model; 'polar-*' uses MACE-POLAR, anything else MACE-MP."""
    from mace.calculators import mace_mp, mace_polar

    log.info("Loading MACE model '%s' on %s (%s)", model, device, dtype)
    if model.startswith("polar"):
        return mace_polar(model=model, device=device, default_dtype=dtype)
    return mace_mp(model=model, device=device, default_dtype=dtype)


def relax(
    atoms: Atoms,
    calc: Calculator,
    path: Path = Path("out.xyz"),
    charge: int = 0,
    relax_cell: bool = True,
    fmax: float = 0.01,
    steps: int = 1000,
    overwrite: bool = False,
) -> Atoms:
    """Relax with BFGS (FrechetCellFilter if relax_cell) and save to path; reuse path if it exists."""
    if path.exists() and not overwrite:
        log.info("  %s exists, reusing", path.name)
        return read(path)

    atoms = atoms.copy()
    atoms.info.update(charge=charge, spin=1, external_field=[0.0, 0.0, 0.0])
    atoms.calc = calc
    path.parent.mkdir(parents=True, exist_ok=True)
    target = FrechetCellFilter(atoms) if relax_cell else atoms
    opt = BFGS(target, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
    atoms.info.update(steps=opt.get_number_of_steps(), converged=converged, relax_cell=relax_cell)
    write(path, atoms, format="extxyz")

    level = logging.INFO if converged else logging.WARNING
    log.log(level, "  %-30s E %12.4f eV | fmax %.4f eV/A | %4d steps | converged %s",
            path.stem, energy, float(np.linalg.norm(forces, axis=1).max()), opt.get_number_of_steps(), converged)
    return atoms


def relax_unitcells(calc: Calculator, out_dir: Path = Path("."), **kw: object) -> dict[str, Atoms]:
    """Stage 1: every species from its CIF, cell + positions."""
    log.info("Stage 1: unit cells")
    return {s.key: relax(s.atoms(), calc, out_dir / "reference" / f"{s.key}_unitcell.xyz", **kw) for s in resolve("all")}


def relax_supercells(
    calc: Calculator, unitcells: dict[str, Atoms], out_dir: Path = Path("."), **kw: object
) -> dict[str, Atoms]:
    """Stage 2: tile the relaxed host unit cells and relax again."""
    log.info("Stage 2: host supercells")
    out: dict[str, Atoms] = {}
    for s in resolve("hosts"):
        tiled = unitcells[s.key].repeat(s.host_supercell)
        log.info("  %s: %s tiling -> %d atoms", s.key, s.host_supercell, len(tiled))
        out[s.key] = relax(tiled, calc, out_dir / "reference" / f"{s.key}_supercell.xyz", **kw)
    return out


def relax_defects(
    calc: Calculator, pristine: Atoms, job_list: list[Job], out_dir: Path = Path("."), **kw: object
) -> None:
    """Stage 3: build each defect on the relaxed host and relax at ISIF 3 or 2."""
    log.info("Stage 3: %d defect relaxations", len(job_list))
    for i, job in enumerate(job_list, start=1):
        log.info("[%d/%d] %s", i, len(job_list), job.tag)
        atoms, _ = job.defect.build(pristine, job.host)
        relax(atoms, calc, out_dir / "defects" / f"{job.tag}.xyz", charge=job.charge, relax_cell=job.isif == 3, **kw)


def collect(out_dir: Path = Path("."), csv: Path = Path("mace_relax.csv")) -> pd.DataFrame:
    """Tabulate every relaxed structure in out_dir (the data needed for any later plot)."""
    rows: list[dict[str, object]] = []
    for path in sorted(out_dir.glob("*/*.xyz")):
        atoms = read(path)
        energy = float(atoms.get_potential_energy())
        rows.append({
            "file": str(path.relative_to(out_dir)),
            "formula": atoms.get_chemical_formula(),
            "n_atoms": len(atoms),
            "charge": atoms.info.get("charge", 0),
            "relax_cell": atoms.info.get("relax_cell"),
            "E_mace": energy,
            "E_mace_per_atom": energy / len(atoms),
            "volume": float(atoms.get_volume()),
            "fmax": float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
            "steps": atoms.info.get("steps"),
            "converged": atoms.info.get("converged"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    log.info("Wrote %d rows to %s", len(df), csv)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="medium-mpa-0", help="MACE-MP model name, or polar-* for MACE-POLAR")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--dtype", default="float64", help="float64 or float32")
    parser.add_argument("--host", default="mZrO2", choices=["mZrO2", "tZrO2"], help="defect host phase")
    parser.add_argument("--defects", default="all", help="group (all, vacancies, ...) or keys, e.g. V_O3,K_i")
    parser.add_argument("--fmax", type=float, default=0.01, help="force tolerance (eV/A)")
    parser.add_argument("--steps", type=int, default=1000, help="max optimiser steps per relaxation")
    parser.add_argument("--overwrite", action="store_true", help="rerun even if the output exists")
    parser.add_argument("--collect-only", action="store_true", help="skip MACE, just rebuild the CSV")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "results" / "pbe_recreation" / args.model
    setup_logging(out_dir)
    csv = out_dir / f"mace_relax_{args.model}.csv"
    if args.collect_only:
        collect(out_dir, csv)
        return

    assert args.host in SPECIES
    polar = args.model.startswith("polar")
    if not polar:
        log.info("MACE-MP ignores charge: running q=0 only")
    job_list = jobs(resolve_defects(args.defects), charges=None if polar else (0,), host=args.host)

    calc = build_calculator(args.model, args.device, args.dtype)
    kw: dict[str, object] = {"fmax": args.fmax, "steps": args.steps, "overwrite": args.overwrite}
    unitcells = relax_unitcells(calc, out_dir, **kw)
    supercells = relax_supercells(calc, unitcells, out_dir, **kw)
    relax_defects(calc, supercells[args.host], job_list, out_dir, **kw)
    collect(out_dir, csv)


if __name__ == "__main__":
    main()
