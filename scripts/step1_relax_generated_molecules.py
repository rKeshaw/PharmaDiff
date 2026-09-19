"""
scripts/step1_relax_generated_molecules.py
==========================================
Step 1 of the three-step P²Diff submission preparation pipeline.

Universal Force-Field Local Relaxation (MMFF94 -> UFF fallback) on all
generated molecule pickles from Phase 1 benchmark models. This is the
standard post-processing applied in:
  - TargetDiff (Guan et al. 2023)
  - DiffSBDD (Schneuing et al. 2023)
  - Pocket2Mol (Peng et al. 2022)

Applies:
  1. Fragment reassembly (largest component kept if disconnected)
  2. Hydrogen addition + MMFF94 optimization (200 steps)
  3. UFF fallback if MMFF94 fails
  4. Re-evaluate validity, and all benchmark metrics on relaxed set

Outputs per model:
  - paper_records/phase1_benchmark/<model>/generated_mols_relaxed.pkl
  - paper_records/phase1_benchmark/<model>/relaxation_stats.json
  - paper_records/phase1_benchmark/relaxed_benchmark_results.json   (final)
"""

import os
import json
import pickle
import time
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import (
    AllChem, Descriptors, QED, rdMolDescriptors, rdForceFieldHelpers
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# ── Project-local imports ─────────────────────────────────────────────────────
from pharmadiff.metrics.pocket_metrics import compute_pocket_clashes, compute_empirical_vina_score
from pharmadiff.metrics.rdkit_match_eval import calculateScore
from pharmadiff.metrics.pgmg_pharma_match_score import match_score


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_pains_catalog():
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


MAX_VALENCE = {'C': 4, 'N': 3, 'O': 2, 'F': 1, 'P': 5, 'S': 6, 'Cl': 1, 'Br': 1, 'I': 1, 'H': 1, 'B': 3}


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
    """
    Attempt MMFF94 geometry optimisation (UFF fallback).
    Applies valence correction first if raw molecule has valence violations.
    Returns (relaxed_mol, method_used, success_bool).
    """
    if mol is None:
        return None, "none", False

    mol, valid = sanitize_and_fix_valence(mol)
    if not valid:
        return None, "none", False

    # Attempt MMFF94
    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
        ff = AllChem.MMFFGetMoleculeForceField(mol_h, AllChem.MMFFGetMoleculeProperties(mol_h))
        if ff is not None:
            ff.Minimize(maxIts=n_steps)
            mol_relaxed = Chem.RemoveHs(mol_h)
            Chem.SanitizeMol(mol_relaxed)
            return mol_relaxed, "MMFF94", True
    except Exception:
        pass

    # Attempt UFF fallback
    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
        ff = AllChem.UFFGetMoleculeForceField(mol_h)
        if ff is not None:
            ff.Minimize(maxIts=n_steps)
            mol_relaxed = Chem.RemoveHs(mol_h)
            Chem.SanitizeMol(mol_relaxed)
            return mol_relaxed, "UFF", True
    except Exception:
        pass

    return mol, "unrelaxed_valid", True


def compute_distribution_stats(arr):
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0,
                "ci95_low": 0.0, "ci95_high": 0.0, "median": 0.0, "n": 0}
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = float(std / np.sqrt(n)) if n > 1 else 0.0
    return {
        "mean": mean, "std": std, "sem": sem,
        "ci95_low": float(mean - 1.96 * sem),
        "ci95_high": float(mean + 1.96 * sem),
        "median": float(np.median(arr)), "n": n
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-model evaluation (mirroring analyze_phase1_benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────

def relax_and_evaluate(raw_mols, test_data, name, model_key, out_dir,
                       is_ground_truth=False):
    print(f"\n{'='*60}")
    print(f"  Relaxing & evaluating: {name}  ({len(raw_mols)} molecules)")
    print(f"{'='*60}")
    t0 = time.time()
    pains_catalog = get_pains_catalog()
    os.makedirs(out_dir, exist_ok=True)

    relaxed_mols = []
    relax_stats = {"mmff94": 0, "uff": 0, "unrelaxed_valid": 0, "failed": 0}

    valid_count = 0
    unique_smiles = set()
    sas_list, qed_list, pains_pass_list = [], [], []
    rings_list, mw_list, logp_list = [], [], []
    ms_list = []
    pmr_count = 0
    ms_above_80 = 0
    severe_clashes_list, mild_clashes_list, min_dist_list = [], [], []
    contact_ratio_list, vina_scores = [], []
    per_complex_records = []

    for i, raw_mol in enumerate(raw_mols):
        if (i + 1) % 25 == 0 or (i + 1) == len(raw_mols):
            print(f"  [{name}] Processed {i+1}/{len(raw_mols)} (valid so far: {valid_count})...", flush=True)

        ref_item = test_data[i]
        pocket_pos = ref_item["pocket"].pos.float() if "pocket" in ref_item else None
        p_mask = ref_item["pharmacophore"].y >= 0
        ref_labels = ref_item["pharmacophore"].y[p_mask].numpy().reshape(-1, 1).astype(int)
        ref_coords = ref_item["pharmacophore"].pos[p_mask].numpy()

        relaxed, method, success = relax_molecule(raw_mol)
        relaxed_mols.append(relaxed)

        if method == "MMFF94":
            relax_stats["mmff94"] += 1
        elif method == "UFF":
            relax_stats["uff"] += 1
        elif success:
            relax_stats["unrelaxed_valid"] += 1
        else:
            relax_stats["failed"] += 1

        record = {
            "complex_idx": i,
            "relax_method": method,
            "is_valid": False,
            "sas": None, "qed": None, "pains_pass": None,
            "num_rings": None, "mw": None, "logp": None,
            "match_score": None,
            "severe_clash": None, "mild_clash": None,
            "min_dist": None, "contact_ratio": None, "vina_score": None
        }

        if success and relaxed is not None:
            valid_count += 1
            record["is_valid"] = True
            smi = Chem.MolToSmiles(relaxed)
            unique_smiles.add(smi)

            sa = float(calculateScore(relaxed))
            q = float(QED.qed(relaxed))
            pains_pass = 1.0 if not pains_catalog.HasMatch(relaxed) else 0.0
            r = int(rdMolDescriptors.CalcNumRings(relaxed))
            mw = float(Descriptors.MolWt(relaxed))
            lp = float(Descriptors.MolLogP(relaxed))

            sas_list.append(sa);   qed_list.append(q)
            pains_pass_list.append(pains_pass)
            rings_list.append(r);  mw_list.append(mw); logp_list.append(lp)

            record.update({"sas": sa, "qed": q, "pains_pass": pains_pass,
                           "num_rings": r, "mw": mw, "logp": lp})

            # Pharmacophore match
            if is_ground_truth:
                ms = 1.0
            else:
                try:
                    ms = float(match_score(relaxed, ref_labels, ref_coords))
                except Exception:
                    ms = 0.0
            ms_list.append(ms)
            record["match_score"] = ms
            if ms >= 0.999:
                pmr_count += 1
            if ms >= 0.8:
                ms_above_80 += 1

            # Pocket clash
            if pocket_pos is not None:
                try:
                    conf = relaxed.GetConformer()
                    lig_pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
                    lig_atoms = [a.GetSymbol() for a in relaxed.GetAtoms()]
                    n_rot = Descriptors.NumRotatableBonds(relaxed)

                    clash = compute_pocket_clashes(lig_pos, pocket_pos)
                    sc = 1.0 if clash["has_severe_clash"] else 0.0
                    mc = 1.0 if clash["has_mild_clash"] else 0.0
                    md = float(clash["min_distance"])
                    cr = float(clash["contact_ratio"])

                    severe_clashes_list.append(sc)
                    mild_clashes_list.append(mc)
                    min_dist_list.append(md)
                    contact_ratio_list.append(cr)
                    record.update({"severe_clash": sc, "mild_clash": mc,
                                   "min_dist": md, "contact_ratio": cr})

                    v = float(compute_empirical_vina_score(
                        ligand_pos=lig_pos, ligand_atom_types=lig_atoms,
                        pocket_pos=pocket_pos, num_rotatable_bonds=n_rot))
                    vina_scores.append(v)
                    record["vina_score"] = v
                except Exception:
                    pass

        per_complex_records.append(record)

    # Save per-complex error-bar records
    eb_file = os.path.join("paper_records/phase1_benchmark/error_bars",
                           f"{model_key}_relaxed_distribution.json")
    with open(eb_file, "w") as f:
        json.dump(per_complex_records, f, indent=2)

    # Save relaxed molecules
    relaxed_pkl = os.path.join(out_dir, "generated_mols_relaxed.pkl")
    with open(relaxed_pkl, "wb") as f:
        pickle.dump(relaxed_mols, f)

    total = len(raw_mols)
    val_rate = valid_count / total * 100.0 if total > 0 else 0.0
    uniq_rate = len(unique_smiles) / max(1, valid_count) * 100.0
    vina_sorted = sorted(vina_scores) if vina_scores else [0.0]

    summary = {
        "name": name, "model_key": model_key,
        "total_evaluated": total,
        "valid_count": valid_count,
        "validity_pct": val_rate,
        "uniqueness_pct": uniq_rate,
        "relax_stats": relax_stats,
        "sas": compute_distribution_stats(sas_list),
        "qed": compute_distribution_stats(qed_list),
        "pains_pass_pct": float(np.mean(pains_pass_list) * 100.0) if pains_pass_list else 0.0,
        "rings": compute_distribution_stats(rings_list),
        "mw": compute_distribution_stats(mw_list),
        "logp": compute_distribution_stats(logp_list),
        "match_score": compute_distribution_stats(ms_list),
        "pmr_pct": float(pmr_count / max(1, len(ms_list)) * 100.0),
        "ms_above_80_pct": float(ms_above_80 / max(1, len(ms_list)) * 100.0),
        "severe_clash_pct": float(np.mean(severe_clashes_list) * 100.0) if severe_clashes_list else 0.0,
        "mild_clash_pct": float(np.mean(mild_clashes_list) * 100.0) if mild_clashes_list else 0.0,
        "min_dist": compute_distribution_stats(min_dist_list),
        "contact_ratio": compute_distribution_stats(contact_ratio_list),
        "vina": compute_distribution_stats(vina_scores),
        "vina_top1": float(vina_sorted[0]) if vina_sorted else 0.0,
        "vina_top10": float(np.mean(vina_sorted[:10])) if len(vina_sorted) >= 10 else float(vina_sorted[0]),
        "vina_mean": float(np.mean(vina_scores)) if vina_scores else 0.0,
    }

    stats_path = os.path.join(out_dir, "relaxation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"  Validity (post-relax): {val_rate:.2f}%  ({valid_count}/{total})")
    print(f"  Uniqueness:            {uniq_rate:.2f}%")
    print(f"  Relax breakdown → MMFF94: {relax_stats['mmff94']}  UFF: {relax_stats['uff']}  "
          f"unrelaxed: {relax_stats['unrelaxed_valid']}  failed: {relax_stats['failed']}", flush=True)
    print(f"  SAS:  {summary['sas']['mean']:.2f} ± {summary['sas']['sem']:.2f}  "
          f"  QED: {summary['qed']['mean']:.2f} ± {summary['qed']['sem']:.2f}")
    print(f"  Pharma MS: {summary['match_score']['mean']:.3f} ± {summary['match_score']['sem']:.3f}"
          f"  PMR: {summary['pmr_pct']:.1f}%  MS>=0.8: {summary['ms_above_80_pct']:.1f}%")
    print(f"  Severe Clash: {summary['severe_clash_pct']:.1f}%  "
          f"Mild Clash: {summary['mild_clash_pct']:.1f}%  "
          f"Min Dist: {summary['min_dist']['mean']:.2f} ± {summary['min_dist']['sem']:.2f} Å")
    print(f"  Vina Mean: {summary['vina_mean']:.2f}  Top-1: {summary['vina_top1']:.2f}  "
          f"Top-10: {summary['vina_top10']:.2f}  (all kcal/mol)")
    print(f"  Elapsed: {elapsed:.1f} s")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading PLINDER scaled test set...")
    d_dict = torch.load("data/plinder_scaled/processed/test_noh_pocket.pt",
                        weights_only=False)
    test_data = [{"ligand": d_dict["ligand"][i],
                  "pharmacophore": d_dict["pharmacophore"][i],
                  "pocket": d_dict["pocket"][i]}
                 for i in range(len(d_dict["ligand"]))]
    print(f"Loaded {len(test_data)} test complexes.\n")

    models = [
        # (pkl_path, is_gt, display_name, key, out_dir)
        ("paper_records/phase1_benchmark/non_pocket/generated_mols.pkl",
         False, "PharmaDiff (Pharma-Only)", "pharmadiff", "paper_records/phase1_benchmark/non_pocket"),

        ("paper_records/phase1_benchmark/pocket_conditioned/generated_mols.pkl",
         False, "P²Diff (Full – Ours)", "p2diff_full", "paper_records/phase1_benchmark/pocket_conditioned"),

        ("paper_records/phase1_benchmark/pocket_only/generated_mols.pkl",
         False, "Pocket-Only SBDD", "pocket_only", "paper_records/phase1_benchmark/pocket_only"),

        ("paper_records/phase1_benchmark/unconditional/generated_mols.pkl",
         False, "MiDi (Unconditional)", "unconditional", "paper_records/phase1_benchmark/unconditional"),
    ]

    results = {}
    for pkl_path, is_gt, name, key, out_dir in models:
        if not os.path.exists(pkl_path):
            print(f"  [SKIP] {name}: file not found at {pkl_path}")
            continue
        mols = pickle.load(open(pkl_path, "rb"))
        summary = relax_and_evaluate(mols, test_data, name, key, out_dir, is_ground_truth=is_gt)
        results[key] = summary

    out_path = "paper_records/phase1_benchmark/relaxed_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Relaxed benchmark results saved to: {out_path}")


if __name__ == "__main__":
    main()
