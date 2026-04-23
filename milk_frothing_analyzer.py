"""
milk_frothing_analyzer.py
==========================
Unified MD Analysis Tool for 1BEB β-lactoglobulin
ที่ Air-Water Interface (Milk Frothing System)

4 analyses:
  1. RMSD  — structural stability (backbone, beta-sheet, helix, calyx)
  2. SASA  — surface accessibility (total, hydrophobic, hydrophilic, calyx)
  3. Z-position — protein location relative to air-water interface
  4. Orientation — hydrophobic patch direction vs Z-axis

Author  : Chalakon Pornjariyawatch — COMFHA Research Group
GitHub  : https://github.com/[yourhandle]/milk-frothing-analyzer
License : MIT

Requirements:
    pip install MDAnalysis matplotlib numpy freesasa

Usage:
    # Run all analyses
    python milk_frothing_analyzer.py --tpr md_1000ns.tpr --xtc traj_comp.xtc

    # Run specific analyses only
    python milk_frothing_analyzer.py --tpr md.tpr --xtc traj.xtc --run rmsd sasa

    # Custom stride (faster, less detail)
    python milk_frothing_analyzer.py --tpr md.tpr --xtc traj.xtc --stride 5

    # Custom output directory
    python milk_frothing_analyzer.py --tpr md.tpr --xtc traj.xtc --outdir results/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from MDAnalysis.transformations import unwrap

# ── Try optional freesasa ──────────────────────────────────────────────────────
try:
    import freesasa
    HAS_FREESASA = True
except ImportError:
    HAS_FREESASA = False


# ══════════════════════════════════════════════════════════════════════════════
#  1BEB SYSTEM CONSTANTS  (ปรับได้ถ้า residue numbering ต่างออกไป)
# ══════════════════════════════════════════════════════════════════════════════
CALYX_RESIDS = [39, 41, 56, 58, 92, 103, 105, 107, 125]

BETA_SHEET_SEL = (
    "resid 2:8 or resid 16:21 or resid 29:35 or resid 46:52 or "
    "resid 62:67 or resid 75:82 or resid 92:98 or resid 106:113 or "
    "resid 138:145"
)
HELIX_SEL      = "resid 130:137"
HYDROPHIL_RESIDS = [1, 10, 20, 30, 60, 80, 120, 150, 160]

HYDROPHOBIC_RESNAMES = {"ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "MET", "TRP"}

ATOM_RADII = {"C": 1.70, "N": 1.55, "O": 1.52,
              "S": 1.80, "H": 1.20, "P": 1.80}

# ── Plot colours ───────────────────────────────────────────────────────────────
C_BLUE   = "#378ADD"
C_GREEN  = "#1D9E75"
C_ORANGE = "#D85A30"
C_AMBER  = "#BA7517"
C_PURPLE = "#7B5EA7"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    w = min(window, len(data))
    return np.convolve(data, np.ones(w) / w, mode="same")


def angle_with_z(vector: np.ndarray) -> float:
    """Angle (°) between vector and Z-axis, folded to 0–90°."""
    z = np.array([0.0, 0.0, 1.0])
    v = vector / (np.linalg.norm(vector) + 1e-12)
    cos_t = np.clip(np.dot(v, z), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_t))
    return min(angle, 180.0 - angle)


def principal_axis(positions: np.ndarray) -> np.ndarray:
    """Longest principal axis from inertia tensor."""
    com = positions.mean(axis=0)
    c   = positions - com
    evals, evecs = np.linalg.eigh(c.T @ c)
    return evecs[:, np.argmax(evals)]


def style_ax(ax, title: str = "", xlabel: str = "Time (ns)",
             ylabel: str = "", ylim=None):
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which="major", lw=0.4, alpha=0.4)
    ax.grid(True, which="minor", lw=0.2, alpha=0.2)
    if ylim is not None:
        ax.set_ylim(ylim)


def load_universe(tpr: str, xtc: str) -> mda.Universe:
    print(f"\n  Loading  : {xtc}")
    u = mda.Universe(tpr, xtc)
    u.trajectory.add_transformations(unwrap(u.select_atoms("protein")))
    print(f"  Frames   : {u.trajectory.n_frames} | "
          f"dt = {u.trajectory.dt:.1f} ps | "
          f"Total = {u.trajectory.totaltime/1000:.1f} ns")
    return u


def sep(label: str = ""):
    w = 60
    print("\n" + "─" * w)
    if label:
        print(f"  {label}")
        print("─" * w)


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1 — RMSD
# ══════════════════════════════════════════════════════════════════════════════
def run_rmsd(tpr: str, xtc: str, stride: int, outdir: Path):
    sep("RMSD Analysis")

    u   = load_universe(tpr, xtc)
    ref = mda.Universe(tpr, xtc)

    calyx_sel = "name CA and (" + " or ".join(
        f"resid {r}" for r in CALYX_RESIDS) + ")"

    selections = {
        "Backbone (whole protein)": "backbone",
        "Beta-sheet"              : f"backbone and ({BETA_SHEET_SEL})",
        "Alpha-helix"             : f"backbone and ({HELIX_SEL})",
        "Hydrophobic patch (Calyx)": calyx_sel,
    }
    colors = [C_BLUE, C_GREEN, C_ORANGE, C_AMBER]

    results = {}
    for label, sel in selections.items():
        print(f"  Computing RMSD: {label}")
        R = rms.RMSD(u, ref, select="backbone",
                     groupselections=[sel], ref_frame=0)
        R.run(step=stride)
        t = R.results.rmsd[:, 1] / 1000.0   # ps → ns
        r = R.results.rmsd[:, 3] / 10.0     # Å → nm
        results[label] = (t, r)

    # Summary
    print("\n  RMSD Summary:")
    print(f"  {'Region':<35} {'Mean (nm)':>10} {'Max (nm)':>10} {'Std':>8}")
    print("  " + "-" * 65)
    for label, (t, r) in results.items():
        print(f"  {label:<35} {np.mean(r):>10.3f} {np.max(r):>10.3f} {np.std(r):>8.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle("1BEB — RMSD Analysis  |  COMFHA", fontsize=13)
    axes = axes.flatten()

    for ax, (color, (label, (t, r))) in zip(axes, zip(colors, results.items())):
        ax.plot(t, r, color=color, lw=0.8, alpha=0.25)
        ax.plot(t, smooth(r), color=color, lw=2.0, label="Rolling avg")
        ax.axhline(np.mean(r), color="gray", lw=1.0, ls="--",
                   label=f"Mean = {np.mean(r):.3f} nm")
        style_ax(ax, title=label, ylabel="RMSD (nm)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = outdir / "rmsd_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2 — SASA
# ══════════════════════════════════════════════════════════════════════════════
def _sasa_frame(protein) -> np.ndarray:
    radii = np.array([
        ATOM_RADII.get(a.name[0].upper(), 1.70) for a in protein.atoms
    ])
    res = freesasa.calcCoord(
        protein.positions.flatten().tolist(), radii.tolist()
    )
    return np.array([res.atomArea(i) for i in range(len(protein.atoms))]) / 100.0


def run_sasa(tpr: str, xtc: str, stride: int, outdir: Path):
    sep("SASA Analysis")

    if not HAS_FREESASA:
        print("  freesasa not installed — skipping SASA.")
        print("  Install with: pip install freesasa")
        return None

    u       = load_universe(tpr, xtc)
    protein = u.select_atoms("protein")

    mask_hphob = np.array([a.resname in HYDROPHOBIC_RESNAMES for a in protein.atoms])
    mask_calyx = np.array([a.resid   in CALYX_RESIDS          for a in protein.atoms])

    times, s_total, s_hphob, s_hphil, s_calyx = [], [], [], [], []
    frames = list(range(0, u.trajectory.n_frames, stride))

    print(f"  Analyzing {len(frames)} frames (stride={stride}) ...")
    for i, fi in enumerate(frames):
        u.trajectory[fi]
        sa = _sasa_frame(protein)
        times.append(u.trajectory.time / 1000.0)
        s_total.append(np.sum(sa))
        s_hphob.append(np.sum(sa[mask_hphob]))
        s_hphil.append(np.sum(sa[~mask_hphob]))
        s_calyx.append(np.sum(sa[mask_calyx]))
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(frames)}] t={times[-1]:.1f} ns | "
                  f"Total={s_total[-1]:.2f} nm²  Hydrophobic={s_hphob[-1]:.2f} nm²")

    t, tot, hob, hil, cal = (np.array(x) for x in
                              [times, s_total, s_hphob, s_hphil, s_calyx])

    ratio = np.mean(hob / tot) * 100
    print(f"\n  SASA Summary:")
    print(f"  Total SASA        : {np.mean(tot):.2f} ± {np.std(tot):.2f} nm²")
    print(f"  Hydrophobic SASA  : {np.mean(hob):.2f} ± {np.std(hob):.2f} nm²")
    print(f"  Hydrophilic SASA  : {np.mean(hil):.2f} ± {np.std(hil):.2f} nm²")
    print(f"  Calyx patch SASA  : {np.mean(cal):.2f} ± {np.std(cal):.2f} nm²")
    print(f"  Hydrophobic ratio : {ratio:.1f}% of total surface")

    datasets = [
        (tot, C_BLUE,   "Total SASA",        "Entire protein"),
        (hob, C_ORANGE, "Hydrophobic SASA",  "Non-polar residues"),
        (hil, C_GREEN,  "Hydrophilic SASA",  "Polar / charged residues"),
        (cal, C_AMBER,  "Calyx patch SASA",  "1BEB hydrophobic calyx"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle("1BEB — SASA Analysis  |  COMFHA", fontsize=13)
    axes = axes.flatten()

    for ax, (data, color, title, subtitle) in zip(axes, datasets):
        ax.plot(t, data, color=color, lw=0.6, alpha=0.25)
        ax.plot(t, smooth(data, 5), color=color, lw=2.0, label="Rolling avg")
        ax.axhline(np.mean(data), color="gray", lw=1.0, ls="--",
                   label=f"Mean = {np.mean(data):.2f} nm²")
        style_ax(ax, title=f"{title}\n{subtitle}", ylabel="SASA (nm²)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = outdir / "sasa_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out}")
    return {"times": t, "total": tot, "hydrophobic": hob,
            "hydrophilic": hil, "calyx": cal}


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3 — Z-POSITION
# ══════════════════════════════════════════════════════════════════════════════
def run_z_position(tpr: str, xtc: str, stride: int, outdir: Path):
    sep("Z-Position Tracking")

    u       = load_universe(tpr, xtc)
    protein = u.select_atoms("protein")
    water   = u.select_atoms("resname SOL")

    times, z_prot, z_upper, z_lower = [], [], [], []

    print(f"  Tracking {u.trajectory.n_frames // stride} frames ...")
    for ts in u.trajectory[::stride]:
        prot_z = protein.center_of_mass()[2]

        # Water interface via percentile of oxygen Z positions
        try:
            wat_z  = water.select_atoms("name OH2 OW O").positions[:, 2]
            upper  = np.percentile(wat_z, 98)
            lower  = np.percentile(wat_z, 2)
        except Exception:
            upper = lower = np.nan

        times.append(ts.time / 1000.0)
        z_prot.append(prot_z)
        z_upper.append(upper)
        z_lower.append(lower)

    t, zp, zu, zl = (np.array(x) for x in [times, z_prot, z_upper, z_lower])
    zp_nm = zp / 10.0
    zu_nm = zu / 10.0
    zl_nm = zl / 10.0

    # Distance to nearest interface (positive = still in bulk water)
    d_upper = (zu - zp) / 10.0
    d_lower = (zp - zl) / 10.0
    dist_nm = np.where(d_upper < d_lower, d_upper, d_lower)

    adsorbed = dist_nm.min() <= 0
    print(f"\n  Z-Position Summary:")
    print(f"  Protein Z range        : {zp_nm.min():.2f} – {zp_nm.max():.2f} nm")
    print(f"  Min dist to interface  : {dist_nm.min():.3f} nm "
          f"(at t = {t[dist_nm.argmin()]:.1f} ns)")
    print(f"  Adsorption status      : {'✓ ADSORBED' if adsorbed else '✗ NOT YET ADSORBED'}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("1BEB — Z-Position Tracking  |  COMFHA", fontsize=13)

    ax1 = axes[0]
    ax1.fill_between(t, zu_nm, zl_nm, alpha=0.12, color=C_BLUE,
                     label="Water slab region")
    ax1.axhline(np.nanmean(zu_nm), color=C_BLUE, lw=0.8, ls="--", alpha=0.6)
    ax1.axhline(np.nanmean(zl_nm), color=C_BLUE, lw=0.8, ls="--", alpha=0.6,
                label="Mean interface (upper / lower)")
    ax1.plot(t, zp_nm, color=C_GREEN, lw=1.5, label="Protein CoM (Z)")
    style_ax(ax1, title="Protein CoM vs Air-Water Interface", ylabel="Z (nm)")
    ax1.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(t, dist_nm, color=C_ORANGE, lw=1.5)
    ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5,
                label="Interface boundary")
    ax2.fill_between(t, dist_nm, 0, where=(dist_nm > 0),
                     alpha=0.10, color=C_ORANGE, label="In bulk water")
    ax2.fill_between(t, dist_nm, 0, where=(dist_nm <= 0),
                     alpha=0.20, color=C_GREEN,  label="Past interface (adsorbed)")
    style_ax(ax2, title="Distance to Nearest Interface",
             ylabel="Distance (nm)", xlabel="Time (ns)")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = outdir / "z_position_track.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out}")
    return {"times": t, "z_protein": zp_nm, "dist_to_interface": dist_nm,
            "adsorbed": adsorbed}


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 4 — ORIENTATION
# ══════════════════════════════════════════════════════════════════════════════
def run_orientation(tpr: str, xtc: str, stride: int, outdir: Path):
    sep("Orientation Analysis")

    u = load_universe(tpr, xtc)
    protein   = u.select_atoms("protein")
    calyx_sel = ("protein and name CA and (" +
                 " or ".join(f"resid {r}" for r in CALYX_RESIDS) + ")")
    calyx     = u.select_atoms(calyx_sel)
    print(f"  Calyx atoms found: {len(calyx)}")

    times, z_prot, ang_hphob, ang_principal = [], [], [], []

    for ts in u.trajectory[::stride]:
        prot_com   = protein.center_of_mass()
        calyx_com  = calyx.center_of_mass()
        hphob_vec  = calyx_com - prot_com
        princ      = principal_axis(protein.positions)

        times.append(ts.time / 1000.0)
        z_prot.append(prot_com[2] / 10.0)
        ang_hphob.append(angle_with_z(hphob_vec))
        ang_principal.append(angle_with_z(princ))

    t, zp, ah, ap = (np.array(x) for x in [times, z_prot, ang_hphob, ang_principal])

    print(f"\n  Orientation Summary:")
    print(f"  Protein Z range         : {zp.min():.2f} – {zp.max():.2f} nm")
    print(f"  Hydrophobic patch angle : {np.mean(ah):.1f} ± {np.std(ah):.1f}°")
    print(f"  Principal axis angle    : {np.mean(ap):.1f} ± {np.std(ap):.1f}°")
    print(f"  Interpretation:")
    print(f"    ~0°  → patch pointing UP (toward vacuum/air) — favorable for adsorption")
    print(f"    ~90° → patch pointing SIDEWAYS — rotating or in bulk")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("1BEB — Orientation Analysis  |  COMFHA", fontsize=13)

    panels = [
        (zp, C_BLUE,   "Protein CoM — Z position",
         "Z (nm)", None),
        (ah, C_ORANGE, "Hydrophobic patch vector vs Z-axis\n"
         "(0° = pointing toward air, 90° = sideways)",
         "Angle (°)", (0, 90)),
        (ap, C_GREEN,  "Principal axis vs Z-axis\n"
         "(0° = protein standing upright, 90° = lying flat)",
         "Angle (°)", (0, 90)),
    ]

    for ax, (data, color, title, ylabel, ylim) in zip(axes, panels):
        ax.plot(t, data, color=color, lw=0.8, alpha=0.25)
        ax.plot(t, smooth(data), color=color, lw=2.0)
        ax.axhline(np.mean(data), color="gray", lw=1.0, ls="--",
                   label=f"Mean = {np.mean(data):.1f}")
        if ylim and ylim[1] == 90:
            ax.axhline(45, color=C_AMBER, lw=0.8, ls=":",
                       label="45° threshold", alpha=0.8)
        style_ax(ax, title=title, ylabel=ylabel, ylim=ylim)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time (ns)", fontsize=9)
    plt.tight_layout()
    out = outdir / "orientation_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {out}")
    return {"times": t, "z_protein": zp,
            "angle_hydrophobic": ah, "angle_principal": ap}


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY DASHBOARD — รวมทุก analysis ใน 1 หน้า
# ══════════════════════════════════════════════════════════════════════════════
def save_dashboard(rmsd_r, z_r, orient_r, sasa_r, outdir: Path):
    """Combined 1-page summary plot สำหรับใส่ใน report หรือ slide."""
    sep("Saving combined dashboard")

    has_sasa = sasa_r is not None
    n_rows   = 3 if has_sasa else 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    fig.suptitle("1BEB Milk Frothing — MD Analysis Dashboard  |  COMFHA",
                 fontsize=13, fontweight="bold")

    # RMSD — backbone
    ax = axes[0, 0]
    label = "Backbone (whole protein)"
    t, r  = rmsd_r[label]
    ax.plot(t, r, color=C_BLUE, lw=0.7, alpha=0.3)
    ax.plot(t, smooth(r), color=C_BLUE, lw=2.0)
    ax.axhline(np.mean(r), color="gray", lw=1.0, ls="--",
               label=f"Mean = {np.mean(r):.3f} nm")
    style_ax(ax, title="RMSD — Backbone", ylabel="RMSD (nm)")
    ax.legend(fontsize=8)

    # Z-position
    ax = axes[0, 1]
    ax.plot(orient_r["times"], orient_r["z_protein"], color=C_GREEN, lw=1.5,
            label="Protein CoM Z")
    style_ax(ax, title="Z-position", ylabel="Z (nm)")
    ax.legend(fontsize=8)

    # Orientation angles
    ax = axes[1, 0]
    ax.plot(orient_r["times"], orient_r["angle_hydrophobic"],
            color=C_ORANGE, lw=0.7, alpha=0.3)
    ax.plot(orient_r["times"], smooth(orient_r["angle_hydrophobic"]),
            color=C_ORANGE, lw=2.0, label="Hydrophobic patch")
    ax.plot(orient_r["times"], smooth(orient_r["angle_principal"]),
            color=C_PURPLE, lw=2.0, ls="--", label="Principal axis")
    ax.axhline(45, color=C_AMBER, lw=0.8, ls=":", alpha=0.7)
    style_ax(ax, title="Orientation vs Z-axis", ylabel="Angle (°)", ylim=(0, 90))
    ax.legend(fontsize=8)

    # Distance to interface
    ax = axes[1, 1]
    ax.plot(z_r["times"], z_r["dist_to_interface"], color=C_ORANGE, lw=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5, label="Interface")
    ax.fill_between(z_r["times"], z_r["dist_to_interface"], 0,
                    where=(z_r["dist_to_interface"] <= 0),
                    alpha=0.2, color=C_GREEN, label="Adsorbed")
    style_ax(ax, title="Distance to Interface", ylabel="Distance (nm)")
    ax.legend(fontsize=8)

    if has_sasa:
        ax = axes[2, 0]
        ax.plot(sasa_r["times"], sasa_r["total"],
                color=C_BLUE, lw=0.7, alpha=0.3)
        ax.plot(sasa_r["times"], smooth(sasa_r["total"]),
                color=C_BLUE, lw=2.0, label="Total")
        ax.plot(sasa_r["times"], smooth(sasa_r["hydrophobic"]),
                color=C_ORANGE, lw=2.0, label="Hydrophobic")
        style_ax(ax, title="SASA", ylabel="nm²")
        ax.legend(fontsize=8)

        ax = axes[2, 1]
        ax.plot(sasa_r["times"], sasa_r["calyx"],
                color=C_AMBER, lw=0.7, alpha=0.3)
        ax.plot(sasa_r["times"], smooth(sasa_r["calyx"]),
                color=C_AMBER, lw=2.0)
        ax.axhline(np.mean(sasa_r["calyx"]), color="gray", lw=1.0, ls="--",
                   label=f"Mean = {np.mean(sasa_r['calyx']):.2f} nm²")
        style_ax(ax, title="Calyx Patch SASA", ylabel="nm²")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = outdir / "dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
AVAILABLE = ["rmsd", "sasa", "z", "orientation", "dashboard"]


def build_parser():
    p = argparse.ArgumentParser(
        description="Unified 1BEB MD Analyzer — COMFHA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything
  python milk_frothing_analyzer.py --tpr md_1000ns.tpr --xtc traj_comp.xtc

  # Only RMSD and Z-position
  python milk_frothing_analyzer.py --tpr md.tpr --xtc traj.xtc --run rmsd z

  # Faster run (every 5th frame), save to results/
  python milk_frothing_analyzer.py --tpr md.tpr --xtc traj.xtc --stride 5 --outdir results/
        """
    )
    p.add_argument("--tpr",    required=True, help="GROMACS .tpr file")
    p.add_argument("--xtc",    required=True, help="GROMACS .xtc trajectory file")
    p.add_argument("--stride", type=int, default=1,
                   help="Read every N-th frame (default: 1)")
    p.add_argument("--outdir", default=".", help="Output directory (default: .)")
    p.add_argument("--run",    nargs="+", choices=AVAILABLE,
                   default=["rmsd", "sasa", "z", "orientation", "dashboard"],
                   help="Which analyses to run (default: all)")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    for path in [args.tpr, args.xtc]:
        if not Path(path).exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    run = set(args.run)

    print("\n" + "═" * 60)
    print("  1BEB Milk Frothing MD Analyzer  |  COMFHA")
    print("═" * 60)
    print(f"  TPR    : {args.tpr}")
    print(f"  XTC    : {args.xtc}")
    print(f"  Stride : {args.stride}")
    print(f"  Output : {outdir}/")
    print(f"  Run    : {', '.join(sorted(run))}")

    rmsd_r = orient_r = z_r = sasa_r = None

    if "rmsd"        in run: rmsd_r   = run_rmsd(args.tpr, args.xtc, args.stride, outdir)
    if "sasa"        in run: sasa_r   = run_sasa(args.tpr, args.xtc, max(args.stride, 5), outdir)
    if "z"           in run: z_r      = run_z_position(args.tpr, args.xtc, args.stride, outdir)
    if "orientation" in run: orient_r = run_orientation(args.tpr, args.xtc, args.stride, outdir)

    if "dashboard" in run and rmsd_r and z_r and orient_r:
        save_dashboard(rmsd_r, z_r, orient_r, sasa_r, outdir)

    elapsed = time.time() - t0
    print("\n" + "═" * 60)
    print(f"  All done in {elapsed:.1f} s  |  Results in {outdir}/")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
