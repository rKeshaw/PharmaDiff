# Do not move these imports, the order seems to matter
import torch
import pytorch_lightning as pl

import os
import warnings
import pathlib

import hydra
import omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import Subset
from torch_geometric.loader.dataloader import DataLoader


from pharmadiff.datasets import qm9_dataset, geom_dataset
from pharmadiff.datasets.plinder_datamodule import PlinderDataModule, PlinderInfos
from pharmadiff.diffusion_model import FullDenoisingDiffusion

import random

warnings.filterwarnings("ignore", category=PossibleUserWarning)



def get_resume(cfg, dataset_infos, train_smiles, checkpoint_path, test: bool):
    name = cfg.general.name + ('_test' if test else '_resume')
    gpus = cfg.general.gpus
    model = FullDenoisingDiffusion.load_from_checkpoint(checkpoint_path, dataset_infos=dataset_infos,
                                                        train_smiles=train_smiles)
    cfg.general.gpus = gpus
    cfg.general.name = name
    return cfg, model


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: omegaconf.DictConfig):
    dataset_config = cfg.dataset
    pl.seed_everything(cfg.train.seed)

    torch.backends.cudnn.benchmark = bool(getattr(cfg.general, "cudnn_benchmark", True))
    if hasattr(torch.backends.cudnn, "deterministic"):
        torch.backends.cudnn.deterministic = bool(getattr(cfg.general, "deterministic", False))

    if dataset_config.name in ['qm9', "geom", "plinder"]:
        if dataset_config.name == 'qm9':
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)

        elif dataset_config.name == 'geom':
            datamodule = geom_dataset.GeomDataModule(cfg)
            dataset_infos = geom_dataset.GeomInfos(datamodule=datamodule, cfg=cfg)

        else:
            datamodule = PlinderDataModule(cfg)
            dataset_infos = PlinderInfos(datamodule=datamodule, cfg=cfg)
            
        train_smiles = list(datamodule.train_dataloader().dataset.smiles) if cfg.general.test_only else []

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    if cfg.general.test_only:
        cfg, _ = get_resume(cfg, dataset_infos, train_smiles, cfg.general.test_only, test=True)
        
        if cfg.general.test_sampling_num_per_graph > 1 and cfg.general.sample_condition is None:

            test_dataset = datamodule.test_dataloader().dataset
            
            total_test_size = len(test_dataset)
            
            random_indices = random.sample(range(total_test_size), cfg.general.final_model_samples_to_generate)
            
            subset_test_dataset = Subset(test_dataset, random_indices)
            
            gen_loader = DataLoader(subset_test_dataset, batch_size=cfg.train.batch_size, shuffle=False)
            
            
            datamodule.test_dataloader = lambda: gen_loader
            
            print(f"number of : {len(subset_test_dataset)}")
            print(f"number of test steps: {len(gen_loader)}")
            print(f"number of test samples: {len(datamodule.test_dataloader())}")
        
    elif cfg.general.resume is not None:
        print("Resuming from {}".format(cfg.general.resume))
        cfg, _ = get_resume(cfg, dataset_infos, train_smiles, cfg.general.resume, test=False)

    # utils.create_folders(cfg)

    model = FullDenoisingDiffusion(cfg=cfg, dataset_infos=dataset_infos, train_smiles=train_smiles)

    if getattr(cfg.general, "use_torch_compile", False):
        if hasattr(torch, "compile"):
            model.model = torch.compile(model.model)
        else:
            print("[warning] torch.compile requested but not available in this torch version.")

    callbacks = []
    params_to_ignore = ['module.model.train_smiles', 'module.model.dataset_infos']

    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, params_to_ignore)

    if cfg.train.save_model:
        checkpoint_root = getattr(cfg.general, "checkpoint_dir", "checkpoints")
        checkpoint_dir = os.path.join(checkpoint_root, cfg.general.name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                              filename='epoch-{epoch:04d}',
                                              monitor='val/epoch_NLL',
                                              save_top_k=-1,
                                              mode='min',
                                              every_n_epochs=1,
                                              save_last=True,
                                              save_on_train_epoch_end=True)
        callbacks.append(checkpoint_callback)
        print(f"[checkpoint] saving to: {os.path.abspath(checkpoint_dir)}")

    lightweight_eval_subset_size = getattr(cfg.general, "lightweight_eval_subset_size", None)
    if lightweight_eval_subset_size is not None and lightweight_eval_subset_size > 0 and not cfg.general.test_only:
        val_dataset = datamodule.val_dataloader().dataset
        val_size = len(val_dataset)
        subset_size = min(int(lightweight_eval_subset_size), val_size)
        subset_indices = list(range(subset_size))
        subset_val_dataset = Subset(val_dataset, subset_indices)
        collate_fn = getattr(getattr(datamodule, "val_dataset", None), "collate", None)
        val_loader = DataLoader(
            subset_val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.train.num_workers,
            collate_fn=collate_fn,
        )
        datamodule.val_dataloader = lambda: val_loader
        print(f"[lightweight-eval] Using validation subset size {subset_size}/{val_size}.")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()

    matmul_precision = getattr(cfg.general, "torch_matmul_precision", None)
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))

    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    
    strategy = "ddp_find_unused_parameters_true" if use_gpu and cfg.general.gpus > 1 else "auto"

    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy=strategy,
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=cfg.train.progress_bar,
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      precision=getattr(cfg.general, 'trainer_precision', '32-true'),
                      accumulate_grad_batches=getattr(cfg.general, 'accumulate_grad_batches', 1),
                      deterministic=getattr(cfg.general, 'deterministic', False),
                      )

    if not cfg.general.test_only:

        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        # if cfg.general.name not in ['debug', 'test']:
        #     trainer.test(model, datamodule=datamodule)
    else:
        for i in range(cfg.general.num_final_sampling):
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("Directory:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    print("Loading checkpoint", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
