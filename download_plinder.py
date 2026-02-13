import hydra
import torch
import os
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from pharmadiff.datasets.plinder_datamodule import PlinderDataModule

@hydra.main(version_base='1.3', config_path='configs', config_name='config')
def main(cfg: DictConfig):

    if cfg.dataset.name != 'plinder':
        print(f"--> [Override] Switching dataset.name from '{cfg.dataset.name}' to 'plinder'")
        cfg.dataset.name = 'plinder'

    if not cfg.dataset.get('use_sample_cache', False):
        print("--> [Override] Forcing 'use_sample_cache' to True for download.")
        cfg.dataset.use_sample_cache = True

    print(f"--> Initializing DataModule with random_subset={cfg.dataset.get('random_subset', 'All')} and seed={cfg.train.get('seed', 'None')}")
    datamodule = PlinderDataModule(cfg)

    splits = {
        'train': datamodule.train_dataset,
        'val': datamodule.val_dataset,
        'test': datamodule.test_dataset
    }

    for split_name, dataset in splits.items():
        print(f"\nProcessing {split_name} split | Target Systems: {len(dataset)}")
        
        if len(dataset) == 0:
            continue

        loader = DataLoader(
            dataset,
            batch_size=64,          
            shuffle=False,          
            num_workers=cfg.train.num_workers, 
            collate_fn=dataset.collate
        )
        
        for _ in tqdm(loader, desc=f"Caching {split_name}"):
            pass

    print("\n[Success] Cache populated with the correct random subset.")

if __name__ == "__main__":
    main()