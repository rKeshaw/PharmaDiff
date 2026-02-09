import logging
import os
from typing import Any, Iterable, Optional

# =============================================================================
# PLINDER CONFIGURATION
# =============================================================================
os.environ["PLINDER_RELEASE"] = "2024-06"
os.environ["PLINDER_ITERATION"] = "v2"

logging.getLogger("plinder").setLevel(logging.WARNING)

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset

# Official PLINDER Imports
from plinder.core import PlinderSystem
from plinder.core.scores import query_index

# PharmaDiff Imports (Ensure pharmadiff is in your PYTHONPATH)
from pharmadiff.datasets.dataset_utils import mol_to_torch_geometric
from pharmadiff.datasets.pharmacophore_utils import mol_to_torch_pharmacophore

# Atom encoding for PharmaDiff (Must match your model config)
ATOM_ENCODER = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'Cl': 8, 'Br': 9, 'I': 10, 'Fe': 11}

_HYDROPHOBIC_RES = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}

class PlinderGraphDataset(Dataset):
    """PLINDER dataset loader.

    Output schema (per-sample):
        ligand: PyG Data graph for ligand atoms/bonds.
        pharmacophore: PyG Data graph for pharmacophore anchors/features.
        pocket_pos: (N_pocket, 3) pocket atom coordinates.
        pocket_feat: (N_pocket, 8) pocket interaction features.
        pocket_residue_index: (N_pocket,) residue indices for reproducibility.
        pocket_atom_index: (N_pocket,) atom indices for reproducibility.
        pocket_residue_name: list[str] residue names for interpretability.
        pocket_chain_id: list[str] chain IDs for interpretability.
        affinity: (1,) binding affinity label (pKd).
    """

    def __init__(
        self,
        split: str = 'train',
        pocket_radius: float = 10.0,
        transform=None,
        debug: bool = False,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'
            pocket_radius (float): Radius in Angstroms to crop protein context
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.split = split
        self.pocket_radius = pocket_radius
        self.transform = transform
        self.debug = debug

        print(f"--> Loading PLINDER index for split: {split}")
        # Load full index with all columns to ensure we get affinity data
        # columns=None fetches all available metadata
        full_index = query_index(columns=None, splits=["train", "val", "test"])
        
        # Filter by split
        self.index = full_index[full_index['split'] == split].reset_index(drop=True)

        # Filter for systems with valid binding affinity (pKd/pIC50)
        # The column name in 2024-06 release is 'affinity_data.pKd'
        if 'affinity_data.pKd' in self.index.columns:
            initial_len = len(self.index)
            self.index = self.index[self.index['affinity_data.pKd'].notna()].reset_index(drop=True)
            print(f"--> Filtered {initial_len} systems down to {len(self.index)} with valid 'affinity_data.pKd'")
        else:
            print("!! WARNING: 'affinity_data.pKd' column not found. Check index columns.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index.iloc[idx]
        system_id = entry.get('system_id', entry.get('id'))

        try:
            from Bio.PDB import PDBParser
            
            ps = PlinderSystem(system_id=system_id)
            
            # Load ligand from SDF
            ligand_sdfs = ps.ligand_sdfs
            if not ligand_sdfs:
                if self.debug:
                    print(f"[PlinderGraphDataset] No ligand SDF files for {system_id}")
                return None
            
            first_ligand_id = list(ligand_sdfs.keys())[0]
            ligand_sdf_path = ligand_sdfs[first_ligand_id]
            
            supplier = Chem.SDMolSupplier(ligand_sdf_path, removeHs=False)
            ligand_mol = supplier[0]
            
            if ligand_mol is None:
                if self.debug:
                    print(f"[PlinderGraphDataset] Failed to load ligand from {ligand_sdf_path}")
                return None
            
            # Load receptor with BioPython
            receptor_path = ps.receptor_pdb
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('receptor', receptor_path)
            
            # Extract receptor data
            rec_coords = []
            rec_elements = []
            rec_res_names = []
            rec_res_ids = []
            rec_atom_ids = []
            rec_chain_ids = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            rec_coords.append(atom.coord)
                            rec_elements.append(atom.element)
                            rec_res_names.append(residue.resname)
                            rec_res_ids.append(residue.id[1])
                            rec_atom_ids.append(atom.serial_number)
                            rec_chain_ids.append(chain.id)
            
            rec_coords = np.array(rec_coords)
            rec_elements = np.array(rec_elements)
            rec_res_names = np.array(rec_res_names)
            rec_res_ids = np.array(rec_res_ids, dtype=np.int64)
            rec_atom_ids = np.array(rec_atom_ids, dtype=np.int64)
            rec_chain_ids = np.array(rec_chain_ids)
            
            # Pharmacophore
            lig_pos = ligand_mol.GetConformer().GetPositions()
            pos_mean = torch.tensor(lig_pos, dtype=torch.float32).mean(dim=0)
            pharma_data = mol_to_torch_pharmacophore(ligand_mol, pos_mean, name='geom')
            
            if pharma_data is None:
                if self.debug:
                    print(f"[PlinderGraphDataset] Failed pharmacophore for {system_id}")
                return None

            ligand_data, _ = mol_to_torch_geometric(
                ligand_mol, 
                ATOM_ENCODER, 
                smiles=Chem.MolToSmiles(ligand_mol)
            )

            # Pocket extraction
            ligand_coords = np.asarray(lig_pos)
            lig_centroid = ligand_coords.mean(axis=0)
            dists = np.linalg.norm(rec_coords - lig_centroid, axis=1)
            pocket_mask = dists < self.pocket_radius

            pocket_coords = torch.tensor(rec_coords[pocket_mask], dtype=torch.float32)
            pocket_atoms = rec_elements[pocket_mask]
            residue_names = rec_res_names[pocket_mask]
            residue_ids = rec_res_ids[pocket_mask]
            atom_ids = rec_atom_ids[pocket_mask]
            chain_ids = rec_chain_ids[pocket_mask]

            pocket_min_dist = self._min_dist_to_ligand(rec_coords[pocket_mask], ligand_coords)
            pocket_feats = self._encode_protein_atoms(pocket_atoms, residue_names, pocket_min_dist)

            affinity_val = entry.get('affinity_data.pKd', 0)
            affinity = torch.tensor([affinity_val], dtype=torch.float32)

            return {
                'ligand': ligand_data,
                'pharmacophore': pharma_data,
                'pocket_pos': pocket_coords,
                'pocket_feat': pocket_feats,
                'pocket_residue_index': torch.tensor(residue_ids, dtype=torch.long),
                'pocket_atom_index': torch.tensor(atom_ids, dtype=torch.long),
                'pocket_residue_name': list(residue_names),
                'pocket_chain_id': list(chain_ids),
                'system_id': system_id,
                'affinity': affinity
            }

        except Exception as exc:
            if self.debug:
                print(f"[PlinderGraphDataset] Error for {system_id}: {exc}")
            return None
        
    def collate(self, data_list):
        """
        Custom collate to handle variable-sized pockets and PyG Batching
        """
        from torch_geometric.data import Batch

        # Filter Nones
        data_list = [d for d in data_list if d is not None]
        if not data_list:
            return None

        # Batch Ligands + Pharmacophores
        ligands = [d['ligand'] for d in data_list]
        batched_ligand = Batch.from_data_list(ligands)
        pharmacophores = [d['pharmacophore'] for d in data_list]
        batched_pharmacophore = Batch.from_data_list(pharmacophores)

        # Batch Pockets (Flattened with index vector)
        pocket_pos_list = []
        pocket_feat_list = []
        pocket_residue_index_list = []
        pocket_atom_index_list = []
        pocket_batch_list = []
        pocket_residue_name_list = []
        pocket_chain_id_list = []
        system_id_list = []
        affinities = []

        for i, data in enumerate(data_list):
            pos = data['pocket_pos']
            feat = data['pocket_feat']

            pocket_pos_list.append(pos)
            pocket_feat_list.append(feat)
            pocket_residue_index_list.append(data['pocket_residue_index'])
            pocket_atom_index_list.append(data['pocket_atom_index'])
            # Create batch index vector: [0, 0, ..., 1, 1, ...]
            pocket_batch_list.append(torch.full((pos.shape[0],), i, dtype=torch.long))
            pocket_residue_name_list.append(data['pocket_residue_name'])
            pocket_chain_id_list.append(data['pocket_chain_id'])
            system_id_list.append(data.get('system_id'))
            affinities.append(data['affinity'])

        return {
            'ligand': batched_ligand,
            'pharmacophore': batched_pharmacophore,
            'pocket_pos': torch.cat(pocket_pos_list, dim=0),
            'pocket_feat': torch.cat(pocket_feat_list, dim=0),
            'pocket_residue_index': torch.cat(pocket_residue_index_list, dim=0),
            'pocket_atom_index': torch.cat(pocket_atom_index_list, dim=0),
            'pocket_batch': torch.cat(pocket_batch_list, dim=0),
            'pocket_residue_name': pocket_residue_name_list,
            'pocket_chain_id': pocket_chain_id_list,
            'system_id': system_id_list,
            'affinity': torch.cat(affinities, dim=0)
        }

    def _encode_protein_atoms(self, atoms: Iterable[str], residue_names: Iterable[str],
                              min_distances: np.ndarray) -> torch.Tensor:
        """Encode pocket atoms with interaction-aware features.

        Feature schema (dim=8):
            0-4: Atom type one-hot (C, N, O, S, Other)
            5: Hydrophobic residue flag
            6: Contact flag (min distance < 4.5 Å)
            7: Min distance to ligand atoms (scaled)
        """
        mapping = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
        feats = torch.zeros((len(min_distances), 8))
        if residue_names is None:
            residue_names = [None] * len(atoms)
        for i, (atom, res) in enumerate(zip(atoms, residue_names)):
            idx = mapping.get(atom, 4)
            feats[i, idx] = 1.0
            if res in _HYDROPHOBIC_RES:
                feats[i, 5] = 1.0
            feats[i, 6] = 1.0 if min_distances[i] < 4.5 else 0.0
            feats[i, 7] = min(min_distances[i] / 10.0, 1.0)
        return feats

    @staticmethod
    def _min_dist_to_ligand(pocket_coords: np.ndarray, ligand_coords: np.ndarray) -> np.ndarray:
        if pocket_coords.size == 0 or ligand_coords.size == 0:
            return np.zeros((pocket_coords.shape[0],), dtype=np.float32)
        diff = pocket_coords[:, None, :] - ligand_coords[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        return dists.min(axis=1).astype(np.float32)

    @staticmethod
    def _slice_array(structure: Any,
                     names: Iterable[str],
                     mask: np.ndarray,
                     default_value: Optional[Any] = None) -> np.ndarray:
        if isinstance(names, str):
            names = (names,)
        value = None
        for name in names:
            if hasattr(structure, name):
                value = getattr(structure, name)
                break
        if value is None:
            if default_value is None:
                default_value = np.arange(len(mask))
            value = np.asarray(default_value)
        value = np.asarray(value)
        if value.shape[0] != mask.shape[0]:
            fallback = np.asarray(default_value)
            if fallback.shape[0] == mask.shape[0]:
                value = fallback
            else:
                value = np.full(mask.shape[0], default_value)
        return value[mask]

    @staticmethod
    def _coerce_indices(values: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(values, dtype=np.int64)
        except (ValueError, TypeError):
            return np.full(len(values), -1, dtype=np.int64)

    def _build_ligand_atom_mapping(self, ligand_mol, pharma_data, pocket_coords):
        ligand_pos = torch.tensor(
            ligand_mol.GetConformer().GetPositions(), dtype=torch.float32
        )
        n_atoms = ligand_pos.shape[0]
        pharma_feat = pharma_data.x.view(-1).long()
        pharma_mask = pharma_data.y.view(-1).long()

        if pocket_coords.numel() == 0:
            pocket_min_dist = torch.zeros(n_atoms, dtype=torch.float32)
        else:
            distances = torch.cdist(ligand_pos, pocket_coords)
            pocket_min_dist = distances.min(dim=1).values

        atom_index = torch.arange(n_atoms, dtype=torch.long)

        assert (
            pharma_feat.numel() == n_atoms
            and pharma_mask.numel() == n_atoms
            and pocket_min_dist.numel() == n_atoms
        ), "Ligand atom feature counts must match ligand atom count."
        assert atom_index.min() >= 0 and atom_index.max() < n_atoms, (
            "Ligand atom index mapping must be within the ligand atom range."
        )

        return {
            "atom_index": atom_index,
            "pharmacophore_feature": pharma_feat,
            "pharmacophore_mask": pharma_mask,
            "pocket_min_dist": pocket_min_dist,
        }