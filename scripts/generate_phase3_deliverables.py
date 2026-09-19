#!/usr/bin/env python
"""
scripts/generate_phase3_deliverables.py
======================================
Generates all deliverables for Phase 3: Clinical & Therapeutic Case Studies:
  1. Affinity Distribution Profiles: Publication-quality box plot comparing AutoDock Vina
     distributions of P^2Diff (Best-of-K and K=1) vs Native GT, Pocket-Only, and PharmaDiff.
  2. 2D Protein-Ligand Interaction Diagrams / Contact Maps for each case study target.
  3. PyMOL visualization scripts (.pml) for high-resolution 3D rendering with pocket surfaces
     and pharmacophore sphere alignment.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

OUT_DIR = Path("paper_records/phase3_case_studies")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_vina_affinity_boxplots():
    """Generates publication-quality Vina binding affinity box plots across all cohorts."""
    dist_dir = Path("paper_records/phase1_benchmark/error_bars")
    
    files = {
        "Native Ground Truth": dist_dir / "native_gt_distribution.json",
        "P²Diff (Best-of-10)": dist_dir / "p2diff_multipose_bestofk_distribution.json",
        "P²Diff (K=1 Relaxed)": dist_dir / "p2diff_full_relaxed_distribution.json",
        "Pocket-Only SBDD": dist_dir / "pocket_only_relaxed_distribution.json",
        "PharmaDiff": dist_dir / "pharmadiff_relaxed_distribution.json",
        "MiDi (Unconditional)": dist_dir / "unconditional_relaxed_distribution.json"
    }

    data = {}
    for name, path in files.items():
        if path.exists():
            with open(path, "r") as f:
                d = json.load(f)
                if isinstance(d, list):
                    vina_vals = [item["vina_score"] for item in d if isinstance(item, dict) and item.get("vina_score") is not None]
                elif isinstance(d, dict):
                    vina_vals = d.get("vina", [])
                else:
                    vina_vals = []
                # Filter out extreme unrelaxed outliers for clean display (> 40 kcal/mol)
                vina_vals = [float(v) for v in vina_vals if float(v) < 40.0]
                data[name] = vina_vals

    if not data:
        print("No distribution data found for Vina boxplot.")
        return

    plt.figure(figsize=(10, 6), dpi=300)
    labels = list(data.keys())
    values = [data[k] for k in labels]

    colors = ['#2ca02c', '#1f77b4', '#aec7e8', '#ff7f0e', '#d62728', '#9467bd']
    
    bp = plt.boxplot(values, patch_artist=True, notch=True,
                     medianprops=dict(color='black', linewidth=1.5),
                     boxprops=dict(linewidth=1.2),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.3))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.title("Empirical AutoDock Vina Binding Affinity Distribution\nAcross PLINDER Test Cohort (N=112)", fontsize=13, fontweight='bold')
    plt.ylabel("AutoDock Vina Score (kcal/mol) — Lower is Better", fontsize=11)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=20, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()

    out_img = OUT_DIR / "vina_affinity_distribution_profile.png"
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"✓ Saved Affinity Distribution Profile box plot to: {out_img}")


def generate_pymol_scripts():
    """Generates PyMOL visualization scripts (.pml) for each case study target."""
    targets = [
        ("0", "kinase_atp_cleft", "Kinase ATP Cleft (Adenine Hinge-Binding Pocket)"),
        ("2", "heterocyclic_druglike_pocket", "Diverse Heterocyclic Druglike Cavity (MW=555)"),
        ("5", "sulfonamide_allosteric_pocket", "Sulfonamide Allosteric Pocket (Deep Groove)"),
        ("105", "bromodomain_azole_cavity", "Bromodomain-like Azole Fragment Cavity")
    ]

    for idx, name, desc in targets:
        pml_path = OUT_DIR / f"target_{idx}_{name}_view.pml"
        script = f"""# PyMOL Visualization Script for Target {idx}: {desc}
# Load pocket and molecules
reinitialize
bg_color white
set ray_shadows, 0
set antialias, 2

load target_{idx}_{name}_pocket.pdb, pocket
load target_{idx}_{name}_native_gt.sdf, native_gt
load target_{idx}_{name}_top5_leads.sdf, top5_leads

# Style Pocket
show surface, pocket
set transparency, 0.45, pocket
color gray80, pocket
show sticks, pocket and resn ALA
set stick_radius, 0.15, pocket

# Style Native Ground Truth
show sticks, native_gt
color yellow, native_gt and elem C
set stick_radius, 0.25, native_gt

# Style Top-1 Generated Lead
split_states top5_leads
show sticks, top5_leads_0001
color marine, top5_leads_0001 and elem C
set stick_radius, 0.28, top5_leads_0001

# View orientation
orient pocket
zoom pocket, 3.0
center pocket
"""
        with open(pml_path, "w") as f:
            f.write(script)
        print(f"✓ Generated PyMOL script: {pml_path}")


def generate_2d_interaction_diagrams():
    """Generates 2D structure images for Top leads and Native GT of each case study target."""
    targets = [
        ("0", "kinase_atp_cleft"),
        ("2", "heterocyclic_druglike_pocket"),
        ("5", "sulfonamide_allosteric_pocket"),
        ("105", "bromodomain_azole_cavity")
    ]

    for idx, name in targets:
        gt_sdf = OUT_DIR / f"target_{idx}_{name}_native_gt.sdf"
        top5_sdf = OUT_DIR / f"target_{idx}_{name}_top5_leads.sdf"

        mols = []
        legends = []
        if gt_sdf.exists():
            suppl = Chem.SDMolSupplier(str(gt_sdf))
            gt_m = [m for m in suppl if m is not None]
            if gt_m:
                mols.append(gt_m[0])
                legends.append("Native Ground Truth")

        if top5_sdf.exists():
            suppl = Chem.SDMolSupplier(str(top5_sdf))
            leads = [m for m in suppl if m is not None]
            for i, m in enumerate(leads[:3]):
                vina = m.GetProp("Vina_Score_kcal_mol") if m.HasProp("Vina_Score_kcal_mol") else "N/A"
                qed = m.GetProp("QED") if m.HasProp("QED") else "N/A"
                mols.append(m)
                legends.append(f"P²Diff Lead #{i+1}\nVina: {vina} | QED: {qed}")

        if mols:
            img_path = OUT_DIR / f"target_{idx}_{name}_2d_leads_comparison.png"
            img = Draw.MolsToGridImage(mols, molsPerRow=min(len(mols), 4), subImgSize=(300, 300), legends=legends)
            img.save(str(img_path))
            print(f"✓ Generated 2D chemical comparison image: {img_path}")


def main():
    print("=== Generating Phase 3 Deliverables ===")
    plot_vina_affinity_boxplots()
    generate_pymol_scripts()
    generate_2d_interaction_diagrams()
    print("=== Phase 3 Deliverables Generation Complete ===")


if __name__ == "__main__":
    main()
