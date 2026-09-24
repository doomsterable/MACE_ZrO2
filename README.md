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
  pbe_recreation/<model>/   MACE rerun of the PBE_ZrO2 relaxation workflow
    reference/              relaxed unit cells + host supercells
    defects/                relaxed defect supercells
  lii_site_scan/<model>/    Li_i candidate-site scan (start/ holds the starting CIFs)
  MACE_ZrO2_formation_energies.xlsx
```

## Scripts

| Script | Reads | Writes |
| --- | --- | --- |
| `mace_screen.py` | `structures/dft_relaxed/` | `results/screening/mace_vs_dft.csv` (`--output`) |
| `mace_relax.py` | `../PBE_ZrO2/src` defect list | `results/pbe_recreation/<model>/` (`--collect-only` rebuilds the CSV only) |
| `mace_lii_site_scan.py` | `results/pbe_recreation/medium-mpa-0/` | `results/lii_site_scan/medium-mpa-0/` |
| `build_workbook.py` | `results/pbe_recreation/medium-mpa-0/` | `results/MACE_ZrO2_formation_energies.xlsx` |

Reference: arXiv:2603.05238.
