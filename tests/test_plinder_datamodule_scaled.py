"""
test_plinder_datamodule_scaled.py
=================================
Verifies that PlinderDataModule and PlinderInfos instantiate cleanly
with data/plinder_scaled (3,083 train, 153 val, 112 test) under pocket conditioning.
"""

from omegaconf import OmegaConf
from pharmadiff.datasets.plinder_dataset import PlinderDataModule, PlinderInfos


def test_scaled_datamodule_and_infos():
    cfg = OmegaConf.create({
        "dataset": {
            "name": "plinder",
            "datadir": "data/plinder_scaled",
            "remove_h": True,
            "use_pocket": True,
            "filter_dataset": False,
            "pin_memory": False,
            "adaptive_loader": True,
        },
        "model": {
            "use_pocket": True,
        },
        "train": {
            "batch_size": 32,
            "reference_batch_size": 32,
            "num_workers": 2,
        },
        "general": {
            "name": "test_scaled_plinder",
            "seed": 42,
        }
    })

    print("Instantiating PlinderDataModule with data/plinder_scaled...")
    dm = PlinderDataModule(cfg)
    dm.prepare_data()
    dm.setup(stage="fit")

    print(f"DataModule train set size: {len(dm.train_dataset)}")
    print(f"DataModule val set size:   {len(dm.val_dataset)}")
    print(f"DataModule test set size:  {len(dm.test_dataset)}")

    assert len(dm.train_dataset) == 3083
    assert len(dm.val_dataset) == 153
    assert len(dm.test_dataset) == 112

    print("Instantiating PlinderInfos...")
    infos = PlinderInfos(datamodule=dm, cfg=cfg)

    print(f"Atom types shape: {infos.atom_types.shape}")
    print(f"Output dims:      {infos.output_dims}")
    print(f"Pocket feat dim:  {infos.pocket_feat_dim}")
    print(f"Pocket present:   {infos.use_pocket}")

    assert infos.use_pocket is True
    assert infos.atom_types.shape[0] == 15  # without H
    assert infos.pocket_feat_dim == 21
    assert infos.input_dims.pocket_feat == 21

    # Test an adaptive train batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"Train batch successfully generated:")
    print(f"  Ligand batch graphs:   {batch['ligand'].num_graphs}")
    print(f"  Ligand total nodes:    {batch['ligand'].x.shape[0]}")
    print(f"  Pocket batch graphs:   {batch['pocket'].num_graphs}")
    print(f"  Pocket total nodes:    {batch['pocket'].x.shape[0]}")

    print("\nALL SCALED DATAMODULE & INFOS TESTS PASSED!")


if __name__ == "__main__":
    test_scaled_datamodule_and_infos()
