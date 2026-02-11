import logging
import os
import random
import hashlib
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
from torch_geometric.utils import to_dense_batch

# Official PLINDER Imports
from plinder.core import PlinderSystem
from plinder.core.scores import query_index

# PharmaDiff Imports (Ensure pharmadiff is in your PYTHONPATH)
from pharmadiff.datasets.dataset_utils import mol_to_torch_geometric
from pharmadiff.datasets.pharmacophore_utils import mol_to_torch_pharmacophore

# Atom encoding for PharmaDiff (Must match your model config)
ATOM_ENCODER = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7, 'Cl': 8, 'Br': 9, 'I': 10, 'Fe': 11}

_HYDROPHOBIC_RES = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
_PLIP_LABEL_KEYS = [
    "hydrogen_bond",
    "salt_bridge",
    "hydrophobic",
    "pi_stacking",
    "pi_cation",
    "halogen",
]

_AFFINITY_CANDIDATE_COLUMNS = [
    "affinity_data.pKd",
    "affinity_data.pki",
    "affinity_data.pIC50",
    "affinity_data.pchembl",
]

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
        contact_cutoff: float = 4.5,
        min_pocket_atoms: int = 1,
        min_affinity: Optional[float] = None,
        max_entry_resolution: Optional[float] = None,
        require_rdkit_ligand: bool = False,
        pocket_structure_mode: str = "holo_only",
        apo_pred_swap_prob: float = 0.0,
        max_collision_score: Optional[float] = None,
        require_posebusters_connected: bool = False,
        cluster_column: Optional[str] = None,
        allowed_clusters: Optional[Iterable[Any]] = None,
        use_sample_cache: bool = False,
        sample_cache_dir: str = "data/plinder/sample_cache",
        sample_cache_version: str = "v1",
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
        self.contact_cutoff = contact_cutoff
        self.min_pocket_atoms = min_pocket_atoms
        self.pocket_structure_mode = pocket_structure_mode
        self.apo_pred_swap_prob = apo_pred_swap_prob
        self.use_sample_cache = use_sample_cache
        self.cluster_column = cluster_column
        self.allowed_clusters = list(allowed_clusters) if allowed_clusters is not None else None
        self.sample_cache_dir = sample_cache_dir
        self.sample_cache_version = sample_cache_version
        if self.use_sample_cache:
            os.makedirs(self.sample_cache_dir, exist_ok=True)
        self.transform = transform
        self.debug = debug
        self.affinity_column = None

        print(f"--> Loading PLINDER index for split: {split}")
        # Load only the requested split to reduce startup latency and metadata scanning overhead.
        # columns=None fetches all available metadata for downstream filtering/labels.
        self.index = query_index(columns=None, splits=[split]).reset_index(drop=True)

        # Resolve affinity source column across PLINDER schema variants.
        for col in _AFFINITY_CANDIDATE_COLUMNS:
            if col in self.index.columns:
                self.affinity_column = col
                break

        if self.affinity_column is not None:
            initial_len = len(self.index)
            self.index = self.index[self.index[self.affinity_column].notna()].reset_index(drop=True)
            print(
                f"--> Filtered {initial_len} systems down to {len(self.index)} "
                f"with valid '{self.affinity_column}'"
            )
        else:
            print(
                "!! WARNING: no known affinity column found "
                f"(tried: {', '.join(_AFFINITY_CANDIDATE_COLUMNS)}). "
                "Continuing with affinity=0.0."
            )

        if min_affinity is not None and self.affinity_column is not None:
            before = len(self.index)
            self.index = self.index[self.index[self.affinity_column] >= min_affinity].reset_index(drop=True)
            print(f"--> Applied min_affinity>={min_affinity}: {before} -> {len(self.index)}")

        if max_entry_resolution is not None and 'entry_resolution' in self.index.columns:
            before = len(self.index)
            valid_resolution = self.index['entry_resolution'].notna() & (self.index['entry_resolution'] <= max_entry_resolution)
            self.index = self.index[valid_resolution].reset_index(drop=True)
            print(f"--> Applied entry_resolution<={max_entry_resolution}: {before} -> {len(self.index)}")

        if require_rdkit_ligand and 'ligand_is_rdkit_loadable' in self.index.columns:
            before = len(self.index)
            self.index = self.index[self.index['ligand_is_rdkit_loadable'] == True].reset_index(drop=True)
            print(f"--> Applied require_rdkit_ligand: {before} -> {len(self.index)}")

        if max_collision_score is not None and 'system_ligand_has_cofactor_collision' in self.index.columns:
            before = len(self.index)
            # Keep entries where collision score is absent or under threshold
            col = self.index['system_ligand_has_cofactor_collision']
            keep = col.isna() | (col.astype(float) <= max_collision_score)
            self.index = self.index[keep].reset_index(drop=True)
            print(f"--> Applied max_collision_score<={max_collision_score}: {before} -> {len(self.index)}")

        if require_posebusters_connected and 'ligand_posebusters_all_atoms_connected' in self.index.columns:
            before = len(self.index)
            self.index = self.index[self.index['ligand_posebusters_all_atoms_connected'] == True].reset_index(drop=True)
            print(f"--> Applied require_posebusters_connected: {before} -> {len(self.index)}")

        if self.cluster_column is not None:
            if self.cluster_column in self.index.columns:
                if self.allowed_clusters is not None:
                    before = len(self.index)
                    self.index = self.index[self.index[self.cluster_column].isin(self.allowed_clusters)].reset_index(drop=True)
                    print(f"--> Applied cluster filter on {self.cluster_column}: {before} -> {len(self.index)}")
                else:
                    print(f"--> Cluster column available: {self.cluster_column} (no filtering applied)")
            else:
                print(f"!! WARNING: cluster_column='{self.cluster_column}' not found in index; skipping cluster filtering.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index.iloc[idx]
        system_id = entry.get('system_id', entry.get('id'))

        if self.use_sample_cache:
            cache_key = f"{self.sample_cache_version}|{self.split}|{system_id}|r{self.pocket_radius}|c{self.contact_cutoff}|m{self.min_pocket_atoms}|pmode{self.pocket_structure_mode}"
            cache_hash = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
            cache_file = os.path.join(self.sample_cache_dir, f"{self.split}__{system_id}__{cache_hash}.pt")
            if os.path.exists(cache_file):
                try:
                    return torch.load(cache_file, map_location="cpu")
                except Exception:
                    pass

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
            
            # Load receptor with BioPython (holo/apo/pred selection for augmentation)
            receptor_path, pocket_source = self._select_receptor_path(ps)
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
            if pocket_coords.shape[0] < self.min_pocket_atoms:
                return None

            ligand_atom_types = ligand_data.x.long()
            ref_ifp = self._compute_reference_ifp(
                ligand_pos=ligand_data.pos,
                ligand_atom_types=ligand_atom_types,
                pocket_pos=pocket_coords,
                pocket_feat=pocket_feats,
                cutoff=self.contact_cutoff,
            )

            quality_weight = self._compute_quality_weight(entry)
            plip_labels, plip_label_mask = self._extract_plip_labels(entry)

            affinity_val = entry.get(self.affinity_column, 0.0) if self.affinity_column is not None else 0.0
            affinity = torch.tensor([affinity_val], dtype=torch.float32)

            sample = {
                'ligand': ligand_data,
                'pharmacophore': pharma_data,
                'pocket_pos': pocket_coords,
                'pocket_feat': pocket_feats,
                'pocket_residue_index': torch.tensor(residue_ids, dtype=torch.long),
                'pocket_atom_index': torch.tensor(atom_ids, dtype=torch.long),
                'pocket_residue_name': list(residue_names),
                'pocket_chain_id': list(chain_ids),
                'system_id': system_id,
                'pocket_source': pocket_source,
                'affinity': affinity,
                'reference_ifp': ref_ifp,
                'plip_labels': plip_labels,
                'plip_label_mask': plip_label_mask,
                'quality_weight': torch.tensor([quality_weight], dtype=torch.float32),
                'cluster_id': entry.get(self.cluster_column, None) if self.cluster_column is not None else None,
            }
            if self.use_sample_cache:
                try:
                    torch.save(sample, cache_file)
                except Exception:
                    pass
            return sample

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
        ifp_labels = []
        quality_weights = []
        cluster_ids = []
        pocket_source_list = []
        plip_labels_list = []
        plip_label_mask_list = []

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
            ifp_labels.append(data['reference_ifp'])
            quality_weights.append(data['quality_weight'])
            cluster_ids.append(data.get('cluster_id', None))
            pocket_source_list.append(data.get('pocket_source', 'holo'))
            plip_labels_list.append(data['plip_labels'])
            plip_label_mask_list.append(data['plip_label_mask'])

        ligand_pos_dense, ligand_mask = to_dense_batch(x=batched_ligand.pos, batch=batched_ligand.batch)
        pocket_pos_dense, pocket_mask = to_dense_batch(x=torch.cat(pocket_pos_list, dim=0), batch=torch.cat(pocket_batch_list, dim=0))
        interaction_map = torch.cdist(ligand_pos_dense, pocket_pos_dense)
        interaction_mask = ligand_mask.unsqueeze(-1) & pocket_mask.unsqueeze(1)

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
            'pocket_source': pocket_source_list,
            'affinity': torch.cat(affinities, dim=0),
            'reference_ifp': torch.stack(ifp_labels, dim=0),
            'plip_labels': torch.stack(plip_labels_list, dim=0),
            'plip_label_mask': torch.stack(plip_label_mask_list, dim=0),
            'quality_weight': torch.cat(quality_weights, dim=0),
            'cluster_id': cluster_ids,
            'interaction_map': interaction_map,
            'interaction_mask': interaction_mask,
        }

    def _select_receptor_path(self, ps: PlinderSystem):
        default_path = ps.receptor_pdb
        mode = (self.pocket_structure_mode or "holo_only").lower()
        if self.split != 'train' or mode == 'holo_only':
            return default_path, "holo"

        use_swap = random.random() < max(0.0, min(1.0, self.apo_pred_swap_prob))
        if not use_swap:
            return default_path, "holo"

        apo_candidates = [
            getattr(ps, 'apo_receptor_pdb', None),
            getattr(ps, 'apo_pdb', None),
            getattr(ps, 'apo_structure_pdb', None),
        ]
        pred_candidates = [
            getattr(ps, 'pred_receptor_pdb', None),
            getattr(ps, 'pred_pdb', None),
            getattr(ps, 'af2_receptor_pdb', None),
        ]

        candidates = []
        if mode in {'apo_pred_random', 'apo_only'}:
            candidates.extend([(p, 'apo') for p in apo_candidates if isinstance(p, str) and os.path.exists(p)])
        if mode in {'apo_pred_random', 'pred_only'}:
            candidates.extend([(p, 'pred') for p in pred_candidates if isinstance(p, str) and os.path.exists(p)])

        if not candidates:
            return default_path, "holo"
        return random.choice(candidates)

    @staticmethod
    def _extract_plip_labels(entry):
        labels = torch.zeros(len(_PLIP_LABEL_KEYS), dtype=torch.float32)
        mask = torch.zeros(len(_PLIP_LABEL_KEYS), dtype=torch.float32)

        for i, key in enumerate(_PLIP_LABEL_KEYS):
            matching_cols = [c for c in entry.index if key in c.lower() and ('num' in c.lower() or 'count' in c.lower())]
            if not matching_cols:
                matching_cols = [c for c in entry.index if key in c.lower()]
            if not matching_cols:
                continue

            values = []
            for c in matching_cols:
                val = entry.get(c)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    if isinstance(val, bool):
                        fval = 1.0 if val else 0.0
                    else:
                        continue
                if np.isnan(fval):
                    continue
                values.append(fval)
            if not values:
                continue
            labels[i] = float(max(values))
            mask[i] = 1.0

        return labels, mask

    @staticmethod
    def _compute_quality_weight(entry) -> float:
        # Start centered near 1.0 and compose bounded confidence factors.
        weight = 1.0

        resolution = entry.get('entry_resolution', None)
        if resolution is not None and not np.isnan(resolution):
            if resolution <= 1.8:
                weight *= 1.25
            elif resolution <= 2.5:
                weight *= 1.10
            elif resolution >= 3.2:
                weight *= 0.70
            elif resolution >= 2.8:
                weight *= 0.85

        collision_score = entry.get('system_pass_validation_criteria.collision_score', None)
        if collision_score is not None:
            try:
                collision_score = float(collision_score)
                if not np.isnan(collision_score):
                    if collision_score <= 0.05:
                        weight *= 1.10
                    elif collision_score >= 0.20:
                        weight *= 0.80
            except (TypeError, ValueError):
                pass

        pb_connected = entry.get('system_pass_validation_criteria.all_atoms_connected', None)
        if pb_connected is not None:
            weight *= 1.05 if bool(pb_connected) else 0.75

        rdkit_loadable = entry.get('ligand_is_rdkit_loadable', None)
        if rdkit_loadable is not None:
            weight *= 1.05 if bool(rdkit_loadable) else 0.85

        # Keep training stable: clip and softly normalize around 1.
        weight = float(np.clip(weight, 0.4, 1.6))
        return weight

    @staticmethod
    def _compute_reference_ifp(
        ligand_pos: torch.Tensor,
        ligand_atom_types: torch.Tensor,
        pocket_pos: torch.Tensor,
        pocket_feat: torch.Tensor,
        cutoff: float,
    ) -> torch.Tensor:
        n_ligand_types = len(ATOM_ENCODER)
        n_pocket_types = 5
        out = torch.zeros(n_ligand_types * n_pocket_types, dtype=torch.float32)
        if ligand_pos.numel() == 0 or pocket_pos.numel() == 0:
            return out
        dists = torch.cdist(ligand_pos, pocket_pos)
        contact_idx = torch.nonzero(dists <= cutoff, as_tuple=False)
        if contact_idx.numel() == 0:
            return out
        pocket_types = torch.argmax(pocket_feat[:, :n_pocket_types], dim=-1)
        for lig_idx, poc_idx in contact_idx:
            lig_t = int(ligand_atom_types[lig_idx].item())
            if lig_t < 0 or lig_t >= n_ligand_types:
                continue
            poc_t = int(pocket_types[poc_idx].item())
            poc_t = max(0, min(poc_t, n_pocket_types - 1))
            out[lig_t * n_pocket_types + poc_t] = 1.0
        return out

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