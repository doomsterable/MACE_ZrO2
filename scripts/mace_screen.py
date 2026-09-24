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
"""Single-point MACE energies for a directory of relaxed extxyz structures -> CSV.

Each input file is expected to carry the DFT reference (energy/forces) from the
relaxation, so the output holds MACE and DFT side by side for benchmarking.
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]

# e.g. mZrO2_Vo4_isif3_q+2.xyz -> polymorph m, defect Vo4, isif 3, charge +2
NAME_RE = re.compile(
    r"^(?P<polymorph>[mt])ZrO2"
    r"(?:_(?P<defect>[A-Za-z0-9]+)_isif(?P<isif>\d)(?:_q(?P<charge>[+-]?\d+))?)?"
    r"(?P<reference>unitCell|supercell)?$"
)


def parse_name(stem: str = "") -> dict[str, object]:
    """Pull polymorph / defect / isif / charge out of the file stem."""
    match = NAME_RE.match(stem)
    if match is None:
        log.warning("Could not parse name '%s'; metadata left blank", stem)
        return {"polymorph": None, "defect": None, "isif": None, "charge": None}
    parts = match.groupdict()
    return {
        "polymorph": parts["polymorph"],
        # references (unit cell / pristine supercell) carry no defect label
        "defect": parts["defect"] or parts["reference"],
        "isif": int(parts["isif"]) if parts["isif"] else None,
        # a missing _q suffix means the neutral cell
        "charge": int(parts["charge"]) if parts["charge"] else 0,
    }


def dft_reference(atoms: Atoms) -> tuple[float | None, float | None]:
    """Energy and max|force| stored in the file by the DFT relaxation, if present."""
    try:
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces())
        return energy, float(np.abs(forces).max())
    except (RuntimeError, AttributeError):
        return atoms.info.get("energy"), None


def build_calculator(model: str = "medium-mpa-0", device: str = "cpu", dtype: str = "float64"):
    """Load a foundation model; 'polar-*' uses MACE-POLAR, anything else MACE-MP."""
    from mace.calculators import mace_mp, mace_polar

    log.info("Loading MACE model '%s' on %s (%s)", model, device, dtype)
    if model.startswith("polar"):
        return mace_polar(model=model, device=device, default_dtype=dtype)
    return mace_mp(model=model, device=device, default_dtype=dtype)


def evaluate_directory(
    directory: Path = REPO / "structures" / "dft_relaxed",
    model: str = "medium-mpa-0",
    device: str = "cpu",
    dtype: str = "float64",
    use_charge: bool = False,
) -> pd.DataFrame:
    """Single-point every .xyz in *directory* and collect the results."""
    paths = sorted(Path(directory).glob("*.xyz"))
    if not paths:
        raise FileNotFoundError(f"No .xyz files in {directory}")
    log.info("Found %d structures in %s", len(paths), directory)

    calc = build_calculator(model, device, dtype)
    rows: list[dict[str, object]] = []

    for i, path in enumerate(paths, start=1):
        atoms = read(path)
        e_dft, f_dft = dft_reference(atoms)
        meta = parse_name(path.stem)

        # PolarMACE conditions on total charge; MACE-MP ignores these entirely
        atoms.info["charge"] = int(meta["charge"]) if use_charge else 0
        atoms.info["spin"] = 1
        atoms.info["external_field"] = [0.0, 0.0, 0.0]
        atoms.calc = calc
        e_mace = float(atoms.get_potential_energy())
        f_mace = float(np.abs(atoms.get_forces()).max())

        row: dict[str, object] = {"file": path.name, **meta}
        row.update(
            {
                "n_atoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
                "volume": float(atoms.get_volume()),
                "E_mace": e_mace,
                "E_mace_per_atom": e_mace / len(atoms),
                "E_dft": e_dft,
                "E_dft_per_atom": e_dft / len(atoms) if e_dft is not None else None,
                "maxF_mace": f_mace,
                "maxF_dft": f_dft,
            }
        )
        rows.append(row)
        log.info("[%3d/%3d] %-34s E_MACE %12.4f eV | E_DFT %s", i, len(paths), path.name, e_mace,
                 f"{e_dft:12.4f} eV" if e_dft is not None else "     n/a")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=REPO / "structures" / "dft_relaxed", help="directory of relaxed .xyz files")
    parser.add_argument("--model", default="medium-mpa-0", help="MACE-MP model name")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--dtype", default="float64", help="float64 or float32")
    parser.add_argument("--output", type=Path, default=REPO / "results" / "screening" / "mace_vs_dft.csv", help="output CSV path")
    parser.add_argument("--use-charge", action="store_true", help="pass the defect charge to the model (PolarMACE)")
    args = parser.parse_args()

    df = evaluate_directory(args.dir, args.model, args.device, args.dtype, args.use_charge)
    df = df.sort_values(["polymorph", "defect", "charge", "isif"], na_position="first")
    df.to_csv(args.output, index=False)
    log.info("Wrote %d rows to %s", len(df), args.output)

    paired = df.dropna(subset=["E_dft"])
    if not paired.empty:
        # constant offset between the two energy references; spread of the residual is the real signal
        residual = paired["E_mace"] - paired["E_dft"]
        log.info("Paired structures: %d", len(paired))
        log.info("E_MACE - E_DFT: mean %.4f eV, std %.4f eV", residual.mean(), residual.std())


if __name__ == "__main__":
    main()
