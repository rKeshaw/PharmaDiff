import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
import warnings
from vina import Vina

def calculate_clash_score(ligand_mol, protein_pdb_path, threshold=1.5):
    """
    Returns the number of atoms in conflict with the protein.
    """
    prot = Chem.MolFromPDBFile(protein_pdb_path)
    if prot is None: return 0
    prot_conf = prot.GetConformer()
    prot_coords = prot_conf.GetPositions()

    lig_conf = ligand_mol.GetConformer()
    lig_coords = lig_conf.GetPositions()

    dists = cdist(lig_coords, prot_coords)
    
    clashes = np.sum(dists < threshold)
    return clashes

def calculate_vina_affinity(ligand_mol, protein_pdb_path, center, box_size=[20, 20, 20]):
    """
    Minimizes the ligand in the pocket and returns the Vina score.
    """

    v = Vina(sf_name='vina')
    v.set_receptor(protein_pdb_path)

    try:
        from meeko import MoleculePreparation
        preparator = MoleculePreparation()
        preparator.prepare(ligand_mol)
        ligand_pdbqt = preparator.write_pdbqt_string()

        v.set_ligand_from_string(ligand_pdbqt)

        v.compute_vina_maps(center=center, box_size=box_size)
        v.optimize()
        energy = v.score()
        return energy[0]
    except Exception as e:
        print(f"Vina Error: {e}")
        return 0.0
    
def evaluate_generation(generated_mols, protein_pdb, pocket_center):
    """
    Main evaluation loop.
    """
    results = {'affinity': [], 'clashes': []}

    for mol in generated_mols:
        clashes = calculate_clash_score(mol, protein_pdb)
        results['clashes'].append(clashes)

        if clashes < 3:
            affinity = calculate_vina_affinity(mol, protein_pdb, pocket_center)
            results['affinity'].append(affinity)
        else:
            results['affinity'].append(0.0) # Penalize for clashes

    mean_clashes = float(np.mean(results['clashes'])) if results['clashes'] else 0.0
    mean_affinity = float(np.mean(results['affinity'])) if results['affinity'] else 0.0
    print(f"Mean Clashes: {mean_clashes}")
    print(f"Mean Affinity: {mean_affinity}")

    return results

if __name__ == "__main__":
    pass
