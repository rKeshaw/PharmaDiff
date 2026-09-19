"""
scripts/analyze_phase1_benchmark.py
===================================
Phase 1 Core Benchmark Analysis for P^2Diff ICLR 2027 Submission.

Evaluates and compiles rigorous comparative metrics with error bars across:
1. Native Ground Truth (Co-crystallized ligands from PLINDER test set)
2. P^2Diff-NoPocket (Pharmacophore-only, 500 epochs)
3. P^2Diff Full (Joint Pocket + Pharmacophore Conditioning, 500 epochs)
4. P^2Diff-NoPharma / Pocket-Only SBDD (when available)
5. Unconditional 3D Diffusion / MiDi on PLINDER (when available)

Metrics evaluated with rigorous statistical distributions (mean, std, SEM, 95% CI):
- Chemical Quality: Validity, Uniqueness, Novelty, SAS, QED, PAINS pass rate, Mean Ring Count, MW, LogP
- Pharmacophore Fidelity: 3D Match Score (MS), Perfect Match Rate (PMR), %MS >= 0.8
- Pocket Interaction: Severe Clash (<1.5A), Mild Clash (<2.0A), Min Dist, Contact Ratio, AutoDock Vina (Top-1, Top-10, Mean)
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
from pharmadiff.metrics.rdkit_match_eval import calculateScore, check_ring_filter, check_pains, match_mol
from pharmadiff.metrics.pgmg_pharma_match_score import match_score


def get_pains_catalog():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def sanitize_and_check(mol):
    if mol is None:
        return None, False, 0
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        n_components = len(frags)
        largest = max(frags, default=mol, key=lambda m: m.GetNumAtoms())
        Chem.SanitizeMol(largest)
        return largest, True, n_components
    except Exception:
        return None, False, 0


def compute_distribution_stats(arr):
    """Computes mean, standard deviation, standard error, and 95% confidence interval."""
    if len(arr) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "median": 0.0,
            "n": 0
        }
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0
    ci95_low = float(mean - 1.96 * sem)
    ci95_high = float(mean + 1.96 * sem)
    median = float(np.median(arr))
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "median": median,
        "n": n
    }


def analyze_molecules(mols, test_data, is_ground_truth=False, name="Model", model_key="model"):
    print(f"\n--- Analyzing {name} ({len(mols)} molecules) ---")
    pains_catalog = get_pains_catalog()
    
    valid_count = 0
    unique_smiles = set()
    all_valid_smiles = []
    
    sas_list = []
    qed_list = []
    pains_pass_list = []
    rings_list = []
    mw_list = []
    logp_list = []
    components_list = []
    
    ms_list = []
    pmr_count = 0
    ms_above_80 = 0
    
    severe_clashes_list = []
    mild_clashes_list = []
    min_dist_list = []
    contact_ratio_list = []
    vina_scores = []

    # Per-complex breakdown for error bar export
    per_complex_records = []

    for i in range(len(mols)):
        raw_mol = mols[i]
        ref_item = test_data[i]
        
        pocket_pos = ref_item["pocket"].pos.float() if "pocket" in ref_item else None
        p_mask = ref_item["pharmacophore"].y >= 0
        ref_labels = ref_item["pharmacophore"].y[p_mask].numpy().reshape(-1, 1).astype(int)
        ref_coords = ref_item["pharmacophore"].pos[p_mask].numpy()
        
        valid_mol, is_valid, n_comp = sanitize_and_check(raw_mol)
        components_list.append(n_comp if n_comp > 0 else 1)
        
        record = {
            "complex_idx": i,
            "is_valid": bool(is_valid),
            "n_components": int(n_comp if n_comp > 0 else 1),
            "sas": None,
            "qed": None,
            "pains_pass": None,
            "num_rings": None,
            "mw": None,
            "logp": None,
            "match_score": None,
            "severe_clash": None,
            "mild_clash": None,
            "min_dist": None,
            "contact_ratio": None,
            "vina_score": None
        }

        if is_valid and valid_mol is not None:
            valid_count += 1
            smi = Chem.MolToSmiles(valid_mol)
            all_valid_smiles.append(smi)
            unique_smiles.add(smi)
            
            # Physicochemical properties
            sa = float(calculateScore(valid_mol))
            q = float(QED.qed(valid_mol))
            pains_pass = 1.0 if not pains_catalog.HasMatch(valid_mol) else 0.0
            r = int(rdMolDescriptors.CalcNumRings(valid_mol))
            m = float(Descriptors.MolWt(valid_mol))
            lp = float(Descriptors.MolLogP(valid_mol))
            
            sas_list.append(sa)
            qed_list.append(q)
            pains_pass_list.append(pains_pass)
            rings_list.append(r)
            mw_list.append(m)
            logp_list.append(lp)
            
            record["sas"] = sa
            record["qed"] = q
            record["pains_pass"] = pains_pass
            record["num_rings"] = r
            record["mw"] = m
            record["logp"] = lp
            
            # Pharmacophore Match
            if is_ground_truth:
                ms = 1.0
                ms_list.append(ms)
                record["match_score"] = ms
                pmr_count += 1
                ms_above_80 += 1
            else:
                try:
                    ms = float(match_score(valid_mol, ref_labels, ref_coords))
                    ms_list.append(ms)
                    record["match_score"] = ms
                    if ms >= 0.999:
                        pmr_count += 1
                    if ms >= 0.8:
                        ms_above_80 += 1
                except Exception:
                    ms_list.append(0.0)
                    record["match_score"] = 0.0
                
            # Pocket interaction
            if pocket_pos is not None:
                conf = valid_mol.GetConformer()
                lig_pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
                lig_atoms = [a.GetSymbol() for a in valid_mol.GetAtoms()]
                n_rot = Descriptors.NumRotatableBonds(valid_mol)
                
                clash_info = compute_pocket_clashes(lig_pos, pocket_pos)
                sc = 1.0 if clash_info['has_severe_clash'] else 0.0
                mc = 1.0 if clash_info['has_mild_clash'] else 0.0
                md = float(clash_info['min_distance'])
                cr = float(clash_info['contact_ratio'])
                
                severe_clashes_list.append(sc)
                mild_clashes_list.append(mc)
                min_dist_list.append(md)
                contact_ratio_list.append(cr)
                
                record["severe_clash"] = sc
                record["mild_clash"] = mc
                record["min_dist"] = md
                record["contact_ratio"] = cr
                
                v_score = float(compute_empirical_vina_score(
                    ligand_pos=lig_pos,
                    ligand_atom_types=lig_atoms,
                    pocket_pos=pocket_pos,
                    num_rotatable_bonds=n_rot
                ))
                vina_scores.append(v_score)
                record["vina_score"] = v_score
        else:
            if raw_mol is not None and pocket_pos is not None:
                try:
                    conf = raw_mol.GetConformer()
                    lig_pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
                    clash_info = compute_pocket_clashes(lig_pos, pocket_pos)
                    sc = 1.0 if clash_info['has_severe_clash'] else 0.0
                    mc = 1.0 if clash_info['has_mild_clash'] else 0.0
                    md = float(clash_info['min_distance'])
                    cr = float(clash_info['contact_ratio'])
                    
                    severe_clashes_list.append(sc)
                    mild_clashes_list.append(mc)
                    min_dist_list.append(md)
                    contact_ratio_list.append(cr)
                    
                    record["severe_clash"] = sc
                    record["mild_clash"] = mc
                    record["min_dist"] = md
                    record["contact_ratio"] = cr
                except Exception:
                    pass

        per_complex_records.append(record)

    total = len(mols)
    val_rate = (valid_count / total * 100.0) if total > 0 else 0.0
    uniq_rate = (len(unique_smiles) / max(1, valid_count) * 100.0) if valid_count > 0 else 0.0
    
    # Sort Vina scores
    vina_sorted = sorted(vina_scores) if vina_scores else [0.0]
    top1_vina = vina_sorted[0] if vina_sorted else 0.0
    top10_vina = float(np.mean(vina_sorted[:10])) if len(vina_sorted) >= 10 else top1_vina
    mean_vina = float(np.mean(vina_scores)) if vina_scores else 0.0

    summary = {
        "name": name,
        "model_key": model_key,
        "total_evaluated": total,
        "valid_count": valid_count,
        "validity_pct": val_rate,
        "uniqueness_pct": uniq_rate,
        "components": compute_distribution_stats(components_list),
        "sas": compute_distribution_stats(sas_list),
        "qed": compute_distribution_stats(qed_list),
        "pains_pass_pct": float(np.mean(pains_pass_list) * 100.0) if pains_pass_list else 0.0,
        "rings": compute_distribution_stats(rings_list),
        "mw": compute_distribution_stats(mw_list),
        "logp": compute_distribution_stats(logp_list),
        "match_score": compute_distribution_stats(ms_list),
        "pmr_pct": float(pmr_count / max(1, len(ms_list)) * 100.0) if ms_list else 0.0,
        "ms_above_80_pct": float(ms_above_80 / max(1, len(ms_list)) * 100.0) if ms_list else 0.0,
        "severe_clash_pct": float(np.mean(severe_clashes_list) * 100.0) if severe_clashes_list else 0.0,
        "mild_clash_pct": float(np.mean(mild_clashes_list) * 100.0) if mild_clashes_list else 0.0,
        "min_dist": compute_distribution_stats(min_dist_list),
        "contact_ratio": compute_distribution_stats(contact_ratio_list),
        "vina": compute_distribution_stats(vina_scores),
        "vina_top1": float(top1_vina),
        "vina_top10": float(top10_vina),
        "vina_mean": float(mean_vina),
    }

    # Save distribution records for error bars
    eb_dir = "paper_records/phase1_benchmark/error_bars"
    os.makedirs(eb_dir, exist_ok=True)
    eb_file = os.path.join(eb_dir, f"{model_key}_distribution.json")
    with open(eb_file, "w") as f:
        json.dump(per_complex_records, f, indent=2)

    print(f"Validity: {summary['validity_pct']:.2f}% ({valid_count}/{total})")
    print(f"Uniqueness: {summary['uniqueness_pct']:.2f}%")
    print(f"SAS: {summary['sas']['mean']:.2f} ± {summary['sas']['std']:.2f} (SEM: {summary['sas']['sem']:.2f})")
    print(f"QED: {summary['qed']['mean']:.2f} ± {summary['qed']['std']:.2f} (SEM: {summary['qed']['sem']:.2f})")
    print(f"PAINS Pass: {summary['pains_pass_pct']:.1f}%")
    print(f"Pharma MS: {summary['match_score']['mean']:.3f} ± {summary['match_score']['std']:.3f} | PMR: {summary['pmr_pct']:.1f}% | MS>=0.8: {summary['ms_above_80_pct']:.1f}%")
    print(f"Severe Clash: {summary['severe_clash_pct']:.1f}% | Mild Clash: {summary['mild_clash_pct']:.1f}% | Min Dist: {summary['min_dist']['mean']:.2f} ± {summary['min_dist']['std']:.2f}Å")
    print(f"Vina Top-1: {summary['vina_top1']:.2f} kcal/mol | Top-10: {summary['vina_top10']:.2f} kcal/mol | Mean: {summary['vina_mean']:.2f} ± {summary['vina']['std']:.2f} kcal/mol")
    
    return summary


def main():
    print("Loading PLINDER scaled test set...")
    d_dict = torch.load("data/plinder_scaled/processed/test_noh_pocket.pt", weights_only=False)
    test_data = []
    for i in range(len(d_dict["ligand"])):
        test_data.append({
            "ligand": d_dict["ligand"][i],
            "pharmacophore": d_dict["pharmacophore"][i],
            "pocket": d_dict["pocket"][i]
        })
    print(f"Loaded {len(test_data)} test complexes.")

    models_to_evaluate = []

    # 1. Native Ground Truth
    raw_test = pickle.load(open("data/plinder_scaled/raw/test_data.pickle", "rb"))
    gt_mols = [sample[1][0] for sample in raw_test]
    models_to_evaluate.append((gt_mols, True, "Native Ground Truth", "native_gt"))

    # 2. P^2Diff-NoPocket
    nopocket_path = "paper_records/phase1_benchmark/non_pocket/generated_mols.pkl"
    if os.path.exists(nopocket_path):
        nopocket_mols = pickle.load(open(nopocket_path, "rb"))
        models_to_evaluate.append((nopocket_mols, False, "P^2Diff-NoPocket", "p2diff_no_pocket"))

    # 3. P^2Diff (Full Pocket)
    pocket_path = "paper_records/phase1_benchmark/pocket_conditioned/generated_mols.pkl"
    if os.path.exists(pocket_path):
        pocket_mols = pickle.load(open(pocket_path, "rb"))
        models_to_evaluate.append((pocket_mols, False, "P^2Diff (Full Pocket)", "p2diff_full"))

    # 4. Pocket-Only SBDD (when generated)
    pocket_only_path = "paper_records/phase1_benchmark/pocket_only/generated_mols.pkl"
    if os.path.exists(pocket_only_path):
        po_mols = pickle.load(open(pocket_only_path, "rb"))
        models_to_evaluate.append((po_mols, False, "P^2Diff-NoPharma (Pocket-Only)", "pocket_only"))

    # 5. Unconditional Diffusion (when generated)
    uncond_path = "paper_records/phase1_benchmark/unconditional/generated_mols.pkl"
    if os.path.exists(uncond_path):
        uncond_mols = pickle.load(open(uncond_path, "rb"))
        models_to_evaluate.append((uncond_mols, False, "Unconditional 3D (MiDi)", "unconditional"))

    results = {}
    for mols, is_gt, name, key in models_to_evaluate:
        summary = analyze_molecules(mols, test_data, is_ground_truth=is_gt, name=name, model_key=key)
        results[key] = summary

    out_file = "paper_records/phase1_benchmark/benchmark_results_with_error_bars.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive benchmark results with error bars to: {out_file}")


if __name__ == "__main__":
    main()
