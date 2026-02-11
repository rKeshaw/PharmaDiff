from pathlib import Path
import random
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig

from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset, ATOM_ENCODER
from pharmadiff.datasets.abstract_dataset import AbstractDatasetInfos
from pharmadiff.metrics.metrics_utils import compute_all_statistics
from pharmadiff.datasets.pharmacophore_utils import FAMILY_MAPPING
from pharmadiff.utils import PlaceHolder


class PlinderDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if HydraConfig.initialized():
            self._original_cwd = get_original_cwd()
        else:
            # Allow use from non-Hydra entrypoints (e.g., precompute scripts).
            self._original_cwd = str(Path.cwd())
        self._sample_cache_dir = self._resolve_path(getattr(cfg.dataset, "sample_cache_dir", "data/plinder/sample_cache"))
        self._statistics_cache_path = self._resolve_path(getattr(cfg.dataset, "statistics_cache_path", "data/plinder/statistics.pt"))
        
        self.train_dataset = PlinderGraphDataset(
            split="train",
            pocket_radius=cfg.dataset.pocket_radius,
            contact_cutoff=getattr(cfg.dataset, "contact_cutoff", 4.5),
            min_pocket_atoms=getattr(cfg.dataset, "min_pocket_atoms", 1),
            min_affinity=getattr(cfg.dataset, "min_affinity", None),
            max_entry_resolution=getattr(cfg.dataset, "max_entry_resolution", None),
            require_rdkit_ligand=getattr(cfg.dataset, "require_rdkit_ligand", False),
            pocket_structure_mode=getattr(cfg.dataset, "pocket_structure_mode", "holo_only"),
            apo_pred_swap_prob=getattr(cfg.dataset, "apo_pred_swap_prob", 0.0),
            max_collision_score=getattr(cfg.dataset, "max_collision_score", None),
            require_posebusters_connected=getattr(cfg.dataset, "require_posebusters_connected", False),
            cluster_column=getattr(cfg.dataset, "cluster_column", None),
            allowed_clusters=getattr(cfg.dataset, "allowed_clusters", None),
            use_sample_cache=getattr(cfg.dataset, "use_sample_cache", False),
            sample_cache_dir=self._sample_cache_dir,
            sample_cache_version=getattr(cfg.dataset, "sample_cache_version", "v1"),
        )
        self.val_dataset = PlinderGraphDataset(
            split="val",
            pocket_radius=cfg.dataset.pocket_radius,
            contact_cutoff=getattr(cfg.dataset, "contact_cutoff", 4.5),
            min_pocket_atoms=getattr(cfg.dataset, "min_pocket_atoms", 1),
            min_affinity=getattr(cfg.dataset, "min_affinity", None),
            max_entry_resolution=getattr(cfg.dataset, "max_entry_resolution", None),
            require_rdkit_ligand=getattr(cfg.dataset, "require_rdkit_ligand", False),
            pocket_structure_mode=getattr(cfg.dataset, "pocket_structure_mode", "holo_only"),
            apo_pred_swap_prob=getattr(cfg.dataset, "apo_pred_swap_prob", 0.0),
            max_collision_score=getattr(cfg.dataset, "max_collision_score", None),
            require_posebusters_connected=getattr(cfg.dataset, "require_posebusters_connected", False),
            cluster_column=getattr(cfg.dataset, "cluster_column", None),
            allowed_clusters=getattr(cfg.dataset, "allowed_clusters", None),
            use_sample_cache=getattr(cfg.dataset, "use_sample_cache", False),
            sample_cache_dir=self._sample_cache_dir,
            sample_cache_version=getattr(cfg.dataset, "sample_cache_version", "v1"),
        )
        self.test_dataset = PlinderGraphDataset(
            split="test",
            pocket_radius=cfg.dataset.pocket_radius,
            contact_cutoff=getattr(cfg.dataset, "contact_cutoff", 4.5),
            min_pocket_atoms=getattr(cfg.dataset, "min_pocket_atoms", 1),
            min_affinity=getattr(cfg.dataset, "min_affinity", None),
            max_entry_resolution=getattr(cfg.dataset, "max_entry_resolution", None),
            require_rdkit_ligand=getattr(cfg.dataset, "require_rdkit_ligand", False),
            pocket_structure_mode=getattr(cfg.dataset, "pocket_structure_mode", "holo_only"),
            apo_pred_swap_prob=getattr(cfg.dataset, "apo_pred_swap_prob", 0.0),
            max_collision_score=getattr(cfg.dataset, "max_collision_score", None),
            require_posebusters_connected=getattr(cfg.dataset, "require_posebusters_connected", False),
            cluster_column=getattr(cfg.dataset, "cluster_column", None),
            allowed_clusters=getattr(cfg.dataset, "allowed_clusters", None),
            use_sample_cache=getattr(cfg.dataset, "use_sample_cache", False),
            sample_cache_dir=self._sample_cache_dir,
            sample_cache_version=getattr(cfg.dataset, "sample_cache_version", "v1"),
        )
        self.statistics = self._load_or_compute_statistics(self.train_dataset)

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(Path(self._original_cwd) / p)

    def _load_or_compute_statistics(self, dataset: PlinderGraphDataset):
        cache_path = self._statistics_cache_path
        force_recompute = getattr(self.cfg.dataset, "statistics_force_recompute", False)
        if cache_path and not force_recompute:
            cache_file = Path(cache_path)
            if cache_file.exists():
                stats = torch.load(cache_file, map_location="cpu", weights_only=False)
                return {"train": stats, "val": stats, "test": stats}
        stats = self._compute_statistics(dataset)
        if cache_path:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(stats["train"], cache_file)
        return stats

    def _compute_statistics(self, dataset: PlinderGraphDataset):
        max_samples = getattr(self.cfg.dataset, "statistics_max_samples", 2000)
        indices = list(range(len(dataset)))
        if max_samples is not None and len(indices) > max_samples:
            random.shuffle(indices)
            indices = indices[:max_samples]

        subset = torch.utils.data.Subset(dataset, indices)
        stats_workers = min(getattr(self.cfg.train, "num_workers", 0), getattr(self.cfg.dataset, "statistics_num_workers", 16))
        stats_batch_size = max(int(getattr(self.cfg.dataset, "statistics_batch_size", 64)), 1)
        loader = DataLoader(
            subset,
            batch_size=stats_batch_size,
            num_workers=stats_workers,
            shuffle=False,
            pin_memory=False,
            persistent_workers=stats_workers > 0,
            prefetch_factor=2 if stats_workers > 0 else None,
            collate_fn=dataset.collate,
        )

        data_list = []
        for batch in loader:
            if batch is None:
                continue
            lig_batch = batch.get("ligand", None)
            if lig_batch is None:
                continue
            data_list.extend(lig_batch.to_data_list())

        if not data_list:
            raise RuntimeError("No valid PLINDER samples available for statistics computation.")
        stats = compute_all_statistics(
            data_list,
            atom_encoder=ATOM_ENCODER,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
        )
        if not torch.is_tensor(stats.atom_types):
            stats.atom_types = torch.from_numpy(stats.atom_types)
        if not torch.is_tensor(stats.bond_types):
            stats.bond_types = torch.from_numpy(stats.bond_types)
        if not torch.is_tensor(stats.charge_types):
            stats.charge_types = torch.from_numpy(stats.charge_types)
        return {"train": stats, "val": stats, "test": stats}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=True,
            pin_memory=getattr(self.cfg.dataset, "pin_memory", False),
            persistent_workers=self.cfg.train.num_workers > 0,
            prefetch_factor=getattr(self.cfg.train, "prefetch_factor", 2) if self.cfg.train.num_workers > 0 else None,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            pin_memory=getattr(self.cfg.dataset, "pin_memory", False),
            persistent_workers=self.cfg.train.num_workers > 0,
            prefetch_factor=getattr(self.cfg.train, "prefetch_factor", 2) if self.cfg.train.num_workers > 0 else None,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            shuffle=False,
            pin_memory=getattr(self.cfg.dataset, "pin_memory", False),
            persistent_workers=self.cfg.train.num_workers > 0,
            prefetch_factor=getattr(self.cfg.train, "prefetch_factor", 2) if self.cfg.train.num_workers > 0 else None,
            collate_fn=self.test_dataset.collate,
        )


class PlinderInfos(AbstractDatasetInfos):
    def __init__(self, datamodule: PlinderDataModule, cfg):
        self.statistics = datamodule.statistics
        self.name = "plinder"
        self.atom_encoder = ATOM_ENCODER
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()
        self.need_to_strip = False

        if self.atom_encoder:
            max_index = max(self.atom_encoder.values())
            self.atom_encoder = {k: v for k, v in self.atom_encoder.items() if v <= max_index}
            if max_index + 1 != len(self.atom_encoder):
                self.atom_encoder = {k: i for i, k in enumerate(self.atom_encoder.keys())}

        super().complete_infos(datamodule.statistics, self.atom_encoder)
        self.input_dims = PlaceHolder(
            X=self.num_atom_types,
            charges=6,
            E=5,
            y=1,
            pos=3,
            pharma_feat=len(FAMILY_MAPPING),
            pharma_coord=3,
        )

        self.output_dims = PlaceHolder(
            X=self.num_atom_types,
            charges=6,
            E=5,
            y=0,
            pos=3,
            pharma_feat=len(FAMILY_MAPPING),
            pharma_coord=3,
        )

    def to_one_hot(self, X, charges, E, node_mask):
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot((charges + 2).long(), num_classes=6).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E, y=None, pos=None, pharma_coord=None, pharma_feat=None)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 2).long(), num_classes=6).float()

    def pharma_to_one_hot(self, X, pharma_mask):
        X = F.one_hot(X, num_classes=len(FAMILY_MAPPING) + 1).float()
        X = X[:, :, 1:] * pharma_mask.unsqueeze(-1)
        placeholder = PlaceHolder(X=None, charges=None, E=None, y=None, pos=None, pharma_coord=None, pharma_feat=X)
        return placeholder.pharma_feat