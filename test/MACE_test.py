# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "mace-torch",
#   "ase",
#   "numpy",
#   "graph-longrange",
# ]
# ///

from ase import Atoms
from mace.calculators import mace_polar
from mace.calculators import mace_mp
import numpy as np
from ase.io import read

atoms = read("CONTCAR")

# # Alpha-Zr: HCP structure, a=3.232 Å, c=5.147 Å
# a = 3.232
# c = 5.147

# atoms = Atoms('Zr2',
#               scaled_positions=[
#                   [1/3, 2/3, 1/4],
#                   [2/3, 1/3, 3/4],
#               ],
#               cell=[[a, 0, 0],
#                     [-a/2, a*np.sqrt(3)/2, 0],
#                     [0, 0, c]],
#               pbc=True)

# calc = mace_polar(
#     model="polar-1-m",
#     device="cpu",           # or "cuda"
#     default_dtype="float64" # use float32 for faster MD
# )

calc = mace_mp(
    model="medium-mpa-0",
    device="cpu",
    default_dtype="float64",
)

atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.info["external_field"] = [0.0, 0.0, 0.0]
atoms.calc = calc

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print("Wow! Looks like you have:")
print(f"Energy: {energy:.4f} eV")
print(f"Forces (eV/A):\n{np.round(forces, 4)}")
print(f"Stress (eV/A^3):\n{np.round(stress, 6)}")