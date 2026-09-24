# MACE_ZrO2

Benchmarking MACE machine-learned potentials against DFT for point defects in ZrO2.
Defect and species definitions are imported from the sibling repo `../PBE_ZrO2/src`.

## Layout

```
scripts/        analysis scripts (run with `uv run scripts/<name>.py`)
  scratch/      quick one-off tests
structures/
  dft_relaxed/  DFT-relaxed defect structures (extxyz, with DFT energy/forces)
results/
  screening/    MACE single-points on the DFT structures (CSV)
  relaxation/   MACE relaxations, one folder per model (xyz, CSV, logs)
  MACE_ZrO2_formation_energies.xlsx
```

## Scripts

| Script | Reads | Writes |
| --- | --- | --- |
| `mace_screen.py` | `structures/dft_relaxed/` | `results/screening/mace_vs_dft.csv` (`--output`) |
| `mace_relax.py` | `../PBE_ZrO2/src` defect list | `results/relaxation/<model>/` (`--collect-only` rebuilds the CSV only) |
| `mace_lii_site_scan.py` | `results/relaxation/medium-mpa-0/` | `results/relaxation/medium-mpa-0/Lii_site_scan/` |
| `build_workbook.py` | `results/relaxation/medium-mpa-0/` | `results/MACE_ZrO2_formation_energies.xlsx` |

Reference: arXiv:2603.05238.
