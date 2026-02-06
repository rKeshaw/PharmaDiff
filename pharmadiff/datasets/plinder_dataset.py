import torch
from torch.utils.data import Dataset
from pinder.core import PinderSystem, get_index
from pinder.core.loader.geodata import PairedPDB
from pharmadiff.datasets.dataset_utils import mol_to_torch_geometric
from pharmadiff.datasets.pharmacophore_utils import mol_to_torch_pharmacophore
from rdkit import Chem
import numpy as np

# PharmaDiff Atom Encoder (Must match your config)
ATOM_ENCODER = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 8, 'S': 9, 'Cl': 10, 'Br': 12, 'I': 13}

class PlinderGraphDataset(Dataset):
    def __init__(self, split='train', pocket_radius=10.0, transform=None):
        self.split = split
        self.pocket_radius = pocket_radius
        self.transform = transform
        
        # Load the official Plinder Index
        full_index = get_index()
        self.index = full_index[full_index['split'] == split].reset_index(drop=True)
        
        # Filter for systems with affinity data (Crucial for our goal)
        # Note: Plinder 2024-06 has some affinity bugs, we filter for valid pKd/pIC50
        self.index = self.index[self.index['affinity_data.pKd'].notna()]
        print(f"Loaded {len(self.index)} {split} systems with affinity data.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index.iloc[idx]
        system_id = entry['id']
        
        try:
            # 1. Load System using Pinder API (Handles download/caching)
            ps = PinderSystem(system_id)
            
            # 2. Get Ligand (RDKit Mol) & Receptor (Atom Array)
            # We use 'holo' for training the structural diffusion
            ligand_mol = ps.ligand.rdkit_mol
            receptor_struct = ps.holo_receptor.atom_array
            
            if ligand_mol is None: return None
            
            # 3. Compute Pharmacophore (PharmaDiff Logic)
            # This is the "Anchor" for generation
            pharma_data = mol_to_torch_pharmacophore(ligand_mol)
            if pharma_data is None: return None

            # 4. Convert Ligand to Graph (PharmaDiff Logic)
            ligand_data, _ = mol_to_torch_geometric(ligand_mol, ATOM_ENCODER, smiles=Chem.MolToSmiles(ligand_mol))
            
            # 5. Extract Pocket (The Context)
            # Filter receptor atoms within X angstroms of ligand center of mass
            lig_centroid = ligand_mol.GetConformer().GetPositions().mean(axis=0)
            rec_coords = receptor_struct.coord
            dists = np.linalg.norm(rec_coords - lig_centroid, axis=1)
            pocket_mask = dists < self.pocket_radius
            
            pocket_coords = torch.tensor(rec_coords[pocket_mask], dtype=torch.float32)
            pocket_atoms = receptor_struct.element[pocket_mask]
            
            # Simple One-Hot encoding for protein atoms (C, N, O, S, Other)
            pocket_feats = self._encode_protein_atoms(pocket_atoms)

            # 6. Affinity Label (The Compass Target)
            affinity = torch.tensor([entry['affinity_data.pKd']], dtype=torch.float32)

            return {
                'ligand': ligand_data,
                'pharmacophore': pharma_data,
                'pocket_pos': pocket_coords,
                'pocket_feat': pocket_feats,
                'affinity': affinity
            }
            
        except Exception as e:
            print(f"Failed to load {system_id}: {e}")
            return None

    def _encode_protein_atoms(self, atoms):
        # Helper to encode protein atoms for the EGNN
        mapping = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
        feats = torch.zeros((len(atoms), 5)) # 5th is "Other"
        for i, atom in enumerate(atoms):
            idx = mapping.get(atom, 4)
            feats[i, idx] = 1.0
        return feats
