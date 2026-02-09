from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class PlinderPrecomputedDataset(Dataset):
    def __init__(self, split: str = "train", root: str = "data/plinder_precomputed"):
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(f"Precomputed directory not found: {self.root}")
        self.files = sorted(self.root.glob("*.pt"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        payload = torch.load(self.files[idx], map_location="cpu")
        ligand = Data(
            pos=payload["ligand_pos"],
            x=payload["ligand_x"],
            edge_index=payload["ligand_edge_index"],
            edge_attr=payload["ligand_edge_attr"],
            charges=payload["ligand_charges"],
        )
        return {
            "ligand": ligand,
            "pocket_pos": payload["pocket_pos"],
            "pocket_feat": payload["pocket_feat"],
            "affinity": payload["affinity"],
        }

    def collate(self, data_list):
        data_list = [d for d in data_list if d is not None]
        if not data_list:
            return None

        ligands = [d["ligand"] for d in data_list]
        batched_ligand = Batch.from_data_list(ligands)

        pocket_pos_list = []
        pocket_feat_list = []
        pocket_batch_list = []
        affinities = []
        for i, data in enumerate(data_list):
            pos = data["pocket_pos"]
            feat = data["pocket_feat"]
            pocket_pos_list.append(pos)
            pocket_feat_list.append(feat)
            pocket_batch_list.append(torch.full((pos.shape[0],), i, dtype=torch.long))
            affinities.append(data["affinity"])

        return {
            "ligand": batched_ligand,
            "pocket_pos": torch.cat(pocket_pos_list, dim=0),
            "pocket_feat": torch.cat(pocket_feat_list, dim=0),
            "pocket_batch": torch.cat(pocket_batch_list, dim=0),
            "affinity": torch.cat(affinities, dim=0),
        }