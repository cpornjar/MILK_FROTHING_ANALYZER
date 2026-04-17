# 🥛 Milk Frothing MD Analyzer

**Unified MD analysis tool for β-lactoglobulin (1BEB) at Air-Water Interface**  
Built from real research — COMFHA Research Group / SIMATEC Digital Chemistry R&D

---

## What this does

Running MD simulations of milk proteins is one thing.  
Knowing **what to look at** in the trajectory is another.

This tool runs 4 analyses in one command:

| # | Analysis | What it answers |
|---|----------|-----------------|
| 1 | **RMSD** | Is the protein structure stable? Which region moves most? |
| 2 | **SASA** | How much hydrophobic surface is exposed over time? |
| 3 | **Z-position** | Is the protein adsorbing to the air-water interface? |
| 4 | **Orientation** | Is the hydrophobic patch pointing toward the air phase? |

Plus a **combined dashboard plot** — one image, all analyses, ready for your report or slide.

---

## Background

β-lactoglobulin (1BEB) is a key whey protein responsible for foam stability in milk frothing.  
Understanding how it adsorbs at the air-water interface at the molecular level  
is essential for designing better dairy products and foam systems.

This tool was developed as part of ongoing MD research at COMFHA / SIMATEC.

---

## Requirements

```bash
pip install MDAnalysis matplotlib numpy freesasa
```

> `freesasa` is optional — if not installed, SASA analysis is skipped  
> but RMSD, Z-position, and Orientation still run normally.

---

## Input files needed

Generate these from GROMACS before running:

```bash
# No pre-processing needed — just your .tpr and .xtc files
# The tool handles everything from there
```

| File | Description |
|------|-------------|
| `md_1000ns.tpr` | GROMACS topology/run input |
| `traj_comp.xtc` | Compressed trajectory |

---

## Usage

```bash
# Run all 4 analyses + dashboard
python milk_frothing_analyzer.py --tpr md_1000ns.tpr --xtc traj_comp.xtc

# Run only specific analyses
python milk_frothing_analyzer.py --tpr md.tpr --xtc traj.xtc --run rmsd z orientation

# Faster run (every 5th frame) with custom output folder
python milk_frothing_analyzer.py --tpr md.tpr --xtc traj.xtc --stride 5 --outdir results/
```

---

## Output files

| File | Content |
|------|---------|
| `rmsd_analysis.png` | RMSD of backbone, beta-sheet, helix, and calyx region |
| `sasa_analysis.png` | Total, hydrophobic, hydrophilic, and calyx SASA |
| `z_position_track.png` | Protein Z-position vs air-water interface over time |
| `orientation_analysis.png` | Hydrophobic patch angle and principal axis vs Z |
| `dashboard.png` | All analyses combined in one figure |

---

## System configuration (1BEB)

The tool is pre-configured for **β-lactoglobulin (1BEB)**:

```python
# Hydrophobic calyx residues
CALYX_RESIDS = [39, 41, 56, 58, 92, 103, 105, 107, 125]

# Secondary structure regions
BETA_SHEET : resid 2:8, 16:21, 29:35, 46:52, 62:67, 75:82, 92:98, 106:113, 138:145
HELIX      : resid 130:137
```

> To use with a different protein, update `CALYX_RESIDS` and secondary structure  
> selections at the top of the script.

---

## Example interpretation

```
Z-position:  protein CoM crosses interface at t = 420 ns → adsorption event
Orientation: hydrophobic patch angle drops from ~75° → ~25° → patch faces air phase
SASA:        hydrophobic SASA decreases after adsorption → patch buried at interface
RMSD:        calyx region RMSD increases slightly → conformational adaptation
```

---

## Roadmap

- [ ] RMSF per-residue heatmap
- [ ] Hydrogen bond analysis at interface
- [ ] Radial distribution function (RDF) — protein vs water
- [ ] Multi-protein support (foam film simulation)

---

## Author

**Chalakon Pornjariyawatch**  
DPST Scholar | COMFHA Research Group / SIMATEC Digital Chemistry R&D  
[github.com/cpornjar](https://github.com/cpornjar)

---

## License

MIT — free to use, modify, and share.  
If this helped your research, a star ⭐ is appreciated.
