"""
scripts/compute_ground_truth_metrics.py
=======================================
Computes reference baseline metrics on the 112 native co-crystallized test ligands in PLINDER.
"""

import os
import json
import pickle
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from pharmadiff.metrics.pocket_metrics import compute_pocket_clashes, compute_empirical_vina_score
from pharmadiff.metrics.rdkit_match_eval import calculateScore, check_ring_filter, check_pains


def get_pains_catalog():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def main():
    print("Loading test data...")
    raw_test = pickle.load(open("data/plinder_scaled/raw/test_data.pickle", "rb"))
    d_dict = torch.load("data/plinder_scaled/processed/test_noh_pocket.pt", weights_only=False)
    
    pains_catalog = get_pains_catalog()
    
    sas_list = []
    qed_list = []
    pains_pass_list = []
    rings_list = []
    mw_list = []
    logp_list = []
    components_list = []
    
    severe_clashes_list = []
    mild_clashes_list = []
    min_dist_list = []
    contact_ratio_list = []
    vina_scores = []
    
    valid_count = 0

    for i in range(len(raw_test)):
        sample = raw_test[i]
        mol = sample[1][0]
        pocket_data = d_dict["pocket"][i]
        pocket_pos = pocket_data.pos.float()
        
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid_count += 1
                
                frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                components_list.append(len(frags))
                
                sas_list.append(calculateScore(mol))
                qed_list.append(QED.qed(mol))
                pains_pass_list.append(1.0 if not pains_catalog.HasMatch(mol) else 0.0)
                rings_list.append(rdMolDescriptors.CalcNumRings(mol))
                mw_list.append(Descriptors.MolWt(mol))
                logp_list.append(Descriptors.MolLogP(mol))

                # Pocket interaction
                conf = mol.GetConformer()
                lig_pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
                lig_atoms = [a.GetSymbol() for a in mol.GetAtoms()]
                n_rot = Descriptors.NumRotatableBonds(mol)
                
                clash_info = compute_pocket_clashes(lig_pos, pocket_pos)
                severe_clashes_list.append(1.0 if clash_info['has_severe_clash'] else 0.0)
                mild_clashes_list.append(1.0 if clash_info['has_mild_clash'] else 0.0)
                min_dist_list.append(clash_info['min_distance'])
                contact_ratio_list.append(clash_info['contact_ratio'])
                
                v_score = compute_empirical_vina_score(
                    ligand_pos=lig_pos,
                    ligand_atom_types=lig_atoms,
                    pocket_pos=pocket_pos,
                    num_rotatable_bonds=n_rot
                )
                vina_scores.append(v_score)
            except Exception as e:
                print(f"Error processing complex {i}: {e}")

    total = len(raw_test)
    vina_sorted = sorted(vina_scores) if vina_scores else [0.0]
    top1_vina = vina_sorted[0] if vina_sorted else 0.0
    top10_vina = np.mean(vina_sorted[:10]) if len(vina_sorted) >= 10 else top1_vina
    mean_vina = np.mean(vina_scores) if vina_scores else 0.0

    gt_summary = {
        "name": "Native Co-Crystals (Ground Truth)",
        "total_evaluated": total,
        "valid_count": valid_count,
        "validity_pct": (valid_count / total) * 100.0,
        "uniqueness_pct": 100.0,
        "mean_components": float(np.mean(components_list)) if components_list else 1.0,
        "sas_mean": float(np.mean(sas_list)) if sas_list else 0.0,
        "sas_std": float(np.std(sas_list)) if sas_list else 0.0,
        "qed_mean": float(np.mean(qed_list)) if qed_list else 0.0,
        "pains_pass_pct": float(np.mean(pains_pass_list) * 100.0) if pains_pass_list else 0.0,
        "rings_mean": float(np.mean(rings_list)) if rings_list else 0.0,
        "mw_mean": float(np.mean(mw_list)) if mw_list else 0.0,
        "logp_mean": float(np.mean(logp_list)) if logp_list else 0.0,
        "ms_mean": 1.0,
        "pmr_pct": 100.0,
        "ms_above_80_pct": 100.0,
        "rdkit_match_pct": 100.0,
        "severe_clash_pct": float(np.mean(severe_clashes_list) * 100.0) if severe_clashes_list else 0.0,
        "mild_clash_pct": float(np.mean(mild_clashes_list) * 100.0) if mild_clashes_list else 0.0,
        "min_dist_mean": float(np.mean(min_dist_list)) if min_dist_list else 0.0,
        "contact_ratio_mean": float(np.mean(contact_ratio_list) * 100.0) if contact_ratio_list else 0.0,
        "vina_top1": float(top1_vina),
        "vina_top10": float(top10_vina),
        "vina_mean": float(mean_vina),
    }

    out_file = "paper_records/phase1_benchmark/native_ground_truth_metrics.json"
    with open(out_file, "w") as f:
        json.dump(gt_summary, f, indent=2)
    print(f"\nGround truth metrics successfully saved to: {out_file}")
    for k, v in gt_summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
