"""
scripts/step3_analyze_multipose.py
===================================
Step 3 analysis: Multi-pose sampling evaluation with Best-of-K selection.

For each of the 112 test targets, up to K=10 generated poses are available.
This script:
  1. Loads the nested list of poses (shape: [112, K] of Mol objects).
  2. Applies MMFF94/UFF relaxation to each pose.
  3. For each target, selects the best pose by lowest empirical Vina score
     (or best PGMG pharmacophore match score as a secondary criterion).
  4. Reports full statistics with error bars on both:
     - All-poses metric (population-level quality, K*112 = up to 1,120 mols)
     - Best-of-K metric (final lead selection quality, 112 mols)

Outputs:
  - paper_records/phase1_benchmark/p2diff_multipose/multipose_analysis.json
  - paper_records/phase1_benchmark/p2diff_multipose/best_of_k_mols.pkl
  - paper_records/phase1_benchmark/error_bars/p2diff_multipose_bestofk_distribution.json
"""

import os
import json
import pickle
import time
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from pharmadiff.metrics.pocket_metrics import compute_pocket_clashes, compute_empirical_vina_score
from pharmadiff.metrics.rdkit_match_eval import calculateScore
from pharmadiff.metrics.pgmg_pharma_match_score import match_score


def get_pains_catalog():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


MAX_VALENCE = {'C': 4, 'N': 4, 'O': 2, 'F': 1, 'P': 6, 'S': 6, 'Cl': 1, 'Br': 1, 'I': 1, 'H': 1, 'B': 3}


def sanitize_and_fix_valence(mol):
    """Sanitizes molecule, resolving valence/bond order violations by pruning spurious long bonds."""
    if mol is None:
        return None, False
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        largest = max(frags, key=lambda m: m.GetNumAtoms(), default=mol)
        Chem.SanitizeMol(largest)
        return largest, True
    except Exception:
        pass

    try:
        rw = Chem.RWMol(mol)
        conf = mol.GetConformer()
        for _ in range(50):
            worst_atom = None
            worst_excess = 0
            for a in rw.GetAtoms():
                sym = a.GetSymbol()
                max_v = MAX_VALENCE.get(sym, 4)
                cur_v = sum(int(b.GetBondTypeAsDouble()) for b in a.GetBonds())
                if cur_v - max_v > worst_excess:
                    worst_excess = cur_v - max_v
                    worst_atom = a
            if worst_atom is None:
                break

            p1 = conf.GetAtomPosition(worst_atom.GetIdx())
            longest_bond = None
            max_dist = -1
            for b in worst_atom.GetBonds():
                nbr = b.GetOtherAtomIdx(worst_atom.GetIdx())
                p2 = conf.GetAtomPosition(nbr)
                d = (p1 - p2).Length()
                if d > max_dist:
                    max_dist = d
                    longest_bond = b
            if longest_bond is None:
                break

            a1, a2 = longest_bond.GetBeginAtomIdx(), longest_bond.GetEndAtomIdx()
            bt = longest_bond.GetBondType()
            if bt == Chem.rdchem.BondType.TRIPLE:
                rw.RemoveBond(a1, a2)
                rw.AddBond(a1, a2, Chem.rdchem.BondType.DOUBLE)
            elif bt == Chem.rdchem.BondType.DOUBLE:
                rw.RemoveBond(a1, a2)
                rw.AddBond(a1, a2, Chem.rdchem.BondType.SINGLE)
            else:
                rw.RemoveBond(a1, a2)

        m = rw.GetMol()
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
        largest = max(frags, default=m, key=lambda x: x.GetNumAtoms())
        Chem.SanitizeMol(largest)
        return largest, True
    except Exception:
        return None, False


def relax_molecule(mol, n_steps=200):
    if mol is None:
        return None, False
    mol, valid = sanitize_and_fix_valence(mol)
    if not valid:
        return None, False

    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
    except Exception:
        mol_h = mol

    for method in ["MMFF94", "UFF"]:
        try:
            if method == "MMFF94":
                ff = AllChem.MMFFGetMoleculeForceField(mol_h, AllChem.MMFFGetMoleculeProperties(mol_h))
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol_h)
            if ff is not None:
                ff.Minimize(maxIts=n_steps)
                mol_relaxed = Chem.RemoveHs(mol_h)
                Chem.SanitizeMol(mol_relaxed)
                return mol_relaxed, True
        except Exception:
            pass

    try:
        Chem.SanitizeMol(mol)
        return mol, True
    except Exception:
        return None, False


def compute_distribution_stats(arr):
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0,
                "ci95_low": 0.0, "ci95_high": 0.0, "median": 0.0, "n": 0}
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n))
    return {
        "mean": mean, "std": std, "sem": sem,
        "ci95_low": float(mean - 1.96 * sem),
        "ci95_high": float(mean + 1.96 * sem),
        "median": float(np.median(arr)), "n": n
    }


def evaluate_molecule_metrics(mol, ref_labels, ref_coords, pocket_pos, pains_catalog):
    """Returns a dict of metrics for a single valid molecule, or None if invalid."""
    try:
        conf = mol.GetConformer()
        lig_pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    except Exception:
        return None

    smi = Chem.MolToSmiles(mol)
    sa = float(calculateScore(mol))
    q = float(QED.qed(mol))
    pains_pass = 1.0 if not pains_catalog.HasMatch(mol) else 0.0
    r = int(rdMolDescriptors.CalcNumRings(mol))
    mw = float(Descriptors.MolWt(mol))
    lp = float(Descriptors.MolLogP(mol))

    try:
        ms = float(match_score(mol, ref_labels, ref_coords))
    except Exception:
        ms = 0.0

    clash = compute_pocket_clashes(lig_pos, pocket_pos)
    sc = 1.0 if clash["has_severe_clash"] else 0.0
    mc = 1.0 if clash["has_mild_clash"] else 0.0
    md = float(clash["min_distance"])
    cr = float(clash["contact_ratio"])

    n_rot = Descriptors.NumRotatableBonds(mol)
    lig_atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    vina = float(compute_empirical_vina_score(
        ligand_pos=lig_pos, ligand_atom_types=lig_atoms,
        pocket_pos=pocket_pos, num_rotatable_bonds=n_rot))

    return {
        "smiles": smi, "sas": sa, "qed": q, "pains_pass": pains_pass,
        "num_rings": r, "mw": mw, "logp": lp, "match_score": ms,
        "severe_clash": sc, "mild_clash": mc, "min_dist": md,
        "contact_ratio": cr, "vina_score": vina
    }


def main():
    print("Loading PLINDER test data...")
    d_dict = torch.load("data/plinder_scaled/processed/test_noh_pocket.pt", weights_only=False)
    test_data = [{"ligand": d_dict["ligand"][i],
                  "pharmacophore": d_dict["pharmacophore"][i],
                  "pocket": d_dict["pocket"][i]}
                 for i in range(len(d_dict["ligand"]))]
    print(f"Loaded {len(test_data)} test complexes.\n")

    multipose_pkl = "paper_records/phase1_benchmark/p2diff_multipose/generated_mols.pkl"
    if not os.path.exists(multipose_pkl):
        print(f"ERROR: {multipose_pkl} not found. Run step3_multipose_sampling.sh first.")
        return

    raw = pickle.load(open(multipose_pkl, "rb"))
    print(f"Loaded raw multi-pose data: {len(raw)} entries, type={type(raw[0])}")

    # Normalise shape: could be [112, K] nested or flat [112*K]
    if isinstance(raw[0], list):
        mols_per_target = raw  # [112][K]
    else:
        # flat list — reshape assuming K=10
        K = len(raw) // len(test_data)
        mols_per_target = [raw[i*K:(i+1)*K] for i in range(len(test_data))]

    pains_catalog = get_pains_catalog()
    os.makedirs("paper_records/phase1_benchmark/p2diff_multipose", exist_ok=True)

    best_of_k_mols = []
    best_of_k_records = []
    all_pose_records = []

    valid_all = 0
    total_all = 0

    for i, (poses, ref_item) in enumerate(zip(mols_per_target, test_data)):
        pocket_pos = ref_item["pocket"].pos.float()
        p_mask = ref_item["pharmacophore"].y >= 0
        ref_labels = ref_item["pharmacophore"].y[p_mask].numpy().reshape(-1, 1).astype(int)
        ref_coords = ref_item["pharmacophore"].pos[p_mask].numpy()

        candidate_metrics = []
        for raw_mol in poses:
            total_all += 1
            relaxed, ok = relax_molecule(raw_mol)
            if not ok or relaxed is None:
                continue
            valid_all += 1
            m = evaluate_molecule_metrics(relaxed, ref_labels, ref_coords, pocket_pos, pains_catalog)
            if m is not None:
                m["target_idx"] = i
                all_pose_records.append(m)
                candidate_metrics.append((relaxed, m))

        if candidate_metrics:
            # Best-of-K: lowest Vina score (most favorable binding)
            best_mol, best_m = min(candidate_metrics, key=lambda x: x[1]["vina_score"])
            best_of_k_mols.append(best_mol)
            best_of_k_records.append(best_m)
        else:
            best_of_k_mols.append(None)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_data)} targets...")

    print(f"\n--- All-Poses Summary ({total_all} poses, {valid_all} valid after relaxation) ---")
    print(f"  Validity rate: {valid_all/total_all*100:.2f}%")

    def collect(key):
        return [r[key] for r in all_pose_records if r.get(key) is not None]

    all_pose_summary = {
        "total_poses": total_all, "valid_poses": valid_all,
        "validity_pct": valid_all / total_all * 100.0 if total_all > 0 else 0.0,
        "sas": compute_distribution_stats(collect("sas")),
        "qed": compute_distribution_stats(collect("qed")),
        "pains_pass_pct": float(np.mean(collect("pains_pass")) * 100.0) if collect("pains_pass") else 0.0,
        "match_score": compute_distribution_stats(collect("match_score")),
        "severe_clash_pct": float(np.mean(collect("severe_clash")) * 100.0) if collect("severe_clash") else 0.0,
        "mild_clash_pct": float(np.mean(collect("mild_clash")) * 100.0) if collect("mild_clash") else 0.0,
        "min_dist": compute_distribution_stats(collect("min_dist")),
        "vina": compute_distribution_stats(collect("vina_score")),
    }

    def collect_bk(key):
        return [r[key] for r in best_of_k_records if r.get(key) is not None]

    vina_bk = collect_bk("vina_score")
    vina_sorted = sorted(vina_bk)
    best_of_k_summary = {
        "targets_with_valid_pose": len(best_of_k_records),
        "coverage_pct": len(best_of_k_records) / len(test_data) * 100.0,
        "sas": compute_distribution_stats(collect_bk("sas")),
        "qed": compute_distribution_stats(collect_bk("qed")),
        "pains_pass_pct": float(np.mean(collect_bk("pains_pass")) * 100.0) if collect_bk("pains_pass") else 0.0,
        "match_score": compute_distribution_stats(collect_bk("match_score")),
        "pmr_pct": float(sum(1 for m in collect_bk("match_score") if m >= 0.999) / max(1, len(collect_bk("match_score"))) * 100.0),
        "ms_above_80_pct": float(sum(1 for m in collect_bk("match_score") if m >= 0.8) / max(1, len(collect_bk("match_score"))) * 100.0),
        "severe_clash_pct": float(np.mean(collect_bk("severe_clash")) * 100.0) if collect_bk("severe_clash") else 0.0,
        "mild_clash_pct": float(np.mean(collect_bk("mild_clash")) * 100.0) if collect_bk("mild_clash") else 0.0,
        "min_dist": compute_distribution_stats(collect_bk("min_dist")),
        "contact_ratio": compute_distribution_stats(collect_bk("contact_ratio")),
        "vina": compute_distribution_stats(vina_bk),
        "vina_top1": float(vina_sorted[0]) if vina_sorted else 0.0,
        "vina_top10": float(np.mean(vina_sorted[:10])) if len(vina_sorted) >= 10 else (float(vina_sorted[0]) if vina_sorted else 0.0),
        "vina_mean": float(np.mean(vina_bk)) if vina_bk else 0.0,
    }

    print(f"\n--- Best-of-K (Best-per-target) Summary ---")
    print(f"  Coverage: {best_of_k_summary['coverage_pct']:.1f}% targets with ≥1 valid pose")
    print(f"  SAS: {best_of_k_summary['sas']['mean']:.2f} ± {best_of_k_summary['sas']['sem']:.2f}")
    print(f"  QED: {best_of_k_summary['qed']['mean']:.2f} ± {best_of_k_summary['qed']['sem']:.2f}")
    print(f"  PGMG MS: {best_of_k_summary['match_score']['mean']:.3f} ± {best_of_k_summary['match_score']['sem']:.3f}")
    print(f"  PMR: {best_of_k_summary['pmr_pct']:.1f}%  MS>=0.8: {best_of_k_summary['ms_above_80_pct']:.1f}%")
    print(f"  Severe Clash: {best_of_k_summary['severe_clash_pct']:.1f}%  "
          f"Mild Clash: {best_of_k_summary['mild_clash_pct']:.1f}%")
    print(f"  Vina Mean: {best_of_k_summary['vina_mean']:.2f}  "
          f"Top-1: {best_of_k_summary['vina_top1']:.2f}  "
          f"Top-10: {best_of_k_summary['vina_top10']:.2f}  (kcal/mol)")

    # Save everything
    out_dir = "paper_records/phase1_benchmark/p2diff_multipose"
    with open(f"{out_dir}/multipose_analysis.json", "w") as f:
        json.dump({"all_poses": all_pose_summary, "best_of_k": best_of_k_summary}, f, indent=2)

    with open(f"{out_dir}/best_of_k_mols.pkl", "wb") as f:
        pickle.dump(best_of_k_mols, f)

    with open("paper_records/phase1_benchmark/error_bars/p2diff_multipose_bestofk_distribution.json", "w") as f:
        json.dump(best_of_k_records, f, indent=2)

    print(f"\n✓ All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
