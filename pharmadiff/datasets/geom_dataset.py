import collections
import os
import pathlib
import pickle
import copy

from rdkit import Chem, RDLogger
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler
import argparse
from torch_geometric.data import InMemoryDataset, DataLoader
from hydra.utils import get_original_cwd

from pharmadiff.utils import PlaceHolder
import pharmadiff.datasets.dataset_utils as dataset_utils
from pharmadiff.datasets.dataset_utils import load_pickle, save_pickle
from pharmadiff.datasets.abstract_dataset import AbstractDatasetInfos, AbstractAdaptiveDataModule
from pharmadiff.metrics.metrics_utils import compute_all_statistics
from pharmadiff.datasets.pharmacophore_utils import mol_to_torch_pharmacophore, FAMILY_MAPPING



full_atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                     'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15}


class GeomDrugsDataset(InMemoryDataset):
    def __init__(self, split, root, remove_h, transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.remove_h = remove_h
        self.name = 'geom'

        self.atom_encoder = full_atom_encoder
        if remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.full_data_dict = torch.load(self.processed_paths[0], weights_only=False)
        
        
        self.ligand = self.full_data_dict['ligand']
        self.pharmacophore = self.full_data_dict['pharmacophore']
        
        self.statistics = dataset_utils.Statistics(num_nodes=load_pickle(self.processed_paths[1]),
                                                   atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
                                                   bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
                                                   charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
                                                   valencies=load_pickle(self.processed_paths[5]),
                                                   bond_lengths=load_pickle(self.processed_paths[6]),
                                                   bond_angles=torch.from_numpy(np.load(self.processed_paths[7])))
        self.smiles = load_pickle(self.processed_paths[8])
        
    def __len__(self):
        return len(self.ligand)
    
    def __getitem__(self, idx):
        data1_item = self.ligand[idx]
        data2_item = self.pharmacophore[idx]
        return {'ligand': data1_item, 'pharmacophore': data2_item}


    @property
    def raw_file_names(self):
        if os.path.isdir('/Users/clementvignac/'):
            # This is my laptop
            return ['subset.pickle']
        if self.split == 'train':
            return ['train_data.pickle']
        elif self.split == 'val':
            return ['val_data.pickle']
        else:
            return ['test_data.pickle']

    @property
    def processed_file_names(self):
        h = 'noh' if self.remove_h else 'h'
        if self.split == 'train':
            return [f'train_{h}.pt', f'train_n_{h}.pickle', f'train_atom_types_{h}.npy', f'train_bond_types_{h}.npy',
                    f'train_charges_{h}.npy', f'train_valency_{h}.pickle', f'train_bond_lengths_{h}.pickle',
                    f'train_angles_{h}.npy', 'train_smiles.pickle', 'goem_train_smiles_noh.pickle']
        elif self.split == 'val':
            return [f'val_{h}.pt', f'val_n_{h}.pickle', f'val_atom_types_{h}.npy', f'val_bond_types_{h}.npy',
                    f'val_charges_{h}.npy', f'val_valency_{h}.pickle', f'val_bond_lengths_{h}.pickle',
                    f'val_angles_{h}.npy', 'val_smiles.pickle', 'goem_val_smiles_noh.pickle']
        else:
            return [f'test_{h}.pt', f'test_n_{h}.pickle', f'test_atom_types_{h}.npy', f'test_bond_types_{h}.npy',
                    f'test_charges_{h}.npy', f'test_valency_{h}.pickle', f'test_bond_lengths_{h}.pickle',
                    f'test_angles_{h}.npy', 'test_smiles.pickle', 'goem_test_smiles_noh.pickle']

    def download(self):
        from torch_geometric.data.dataset import files_exist
        if files_exist(self.processed_paths):
            return
        raise ValueError('Download and preprocessing is manual. If the data is already downloaded, '
                         f'check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}')

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        all_data = load_pickle(self.raw_paths[0])

        data_list = []
        all_smiles = []
        mols_list = []
        all_smiles_noh = []
        for i, data in enumerate(tqdm(all_data)):
            smiles, all_conformers = data
            all_smiles.append(smiles)
            for j, conformer in enumerate(all_conformers):
                if j >= 1:
                    break
                try:
                    Chem.SanitizeMol(conformer)
                    Chem.Kekulize(conformer)
                except:
                    continue
                data, pos_mean = dataset_utils.mol_to_torch_geometric(conformer, full_atom_encoder, smiles)
                pharmacophore = mol_to_torch_pharmacophore(conformer, pos_mean, name=self.name)
                if pharmacophore is None:
                    continue
                if self.remove_h:
                    data, pharmacophore  = dataset_utils.remove_hydrogens(data, pharmacophore)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append({"ligand": data, "pharmacophore": pharmacophore})
                mols_list.append(data)
                
                try:
                    mol_no_h = Chem.RemoveHs(conformer, sanitize=False)
                    if mol_no_h is not None:
                        smiles_no_h = Chem.MolToSmiles(mol_no_h)
                    if smiles_no_h is not None:
                        all_smiles_noh.append(smiles_no_h)
                except:
                    continue
                

        torch.save(self.custome_collate(self, data_list), self.processed_paths[0])

        statistics = compute_all_statistics(mols_list, self.atom_encoder, charges_dic={-2: 0, -1: 1, 0: 2,
                                                                                       1: 3, 2: 4, 3: 5})
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        save_pickle(set(all_smiles), self.processed_paths[8])
        save_pickle(set(all_smiles_noh), self.processed_paths[9])
        torch.save(self.custome_collate(self, data_list), self.processed_paths[0])
        
    @staticmethod
    def custome_collate(self, data_list):
        ligands = [item['ligand'] for item in data_list]
        pharmacophores = [item['pharmacophore'] for item in data_list]
        assert len(ligands) == len(pharmacophores)

        # Batch the ligand and pharmacophore data
        # Using Batch.from_data_list will automatically create the slices
        #ligand_batch = self.collate(ligands )
        #pharmacophore_batch = self.collate(pharmacophores)

        # Prepare the collated data
        data = {
            'ligand': ligands,
            'pharmacophore': pharmacophores
        }

        return data


class GeomDataModule(AbstractAdaptiveDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset.datadir
        try:
            cwd = pathlib.Path(get_original_cwd())
        except Exception:
            cwd = pathlib.Path.cwd()
        repo_root = cwd.parent if cwd.name == "pharmadiff" else cwd
        root_path = os.path.join(repo_root, self.datadir)

        train_dataset = GeomDrugsDataset(split='train', root=root_path, remove_h=cfg.dataset.remove_h)
        val_dataset = GeomDrugsDataset(split='val', root=root_path, remove_h=cfg.dataset.remove_h)
        test_dataset = GeomDrugsDataset(split='test', root=root_path, remove_h=cfg.dataset.remove_h)
        self.remove_h = cfg.dataset.remove_h
        self.statistics = {'train': train_dataset.statistics, 'val': val_dataset.statistics,
                           'test': test_dataset.statistics}
        super().__init__(cfg, train_dataset, val_dataset, test_dataset)


class GeomInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model
        self.statistics = datamodule.statistics
        self.name = 'geom'
        self.atom_encoder = full_atom_encoder
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()
        if self.remove_h:
            self.atom_encoder = {k: v - 1 for k, v in self.atom_encoder.items() if k != 'H'}

        super().complete_infos(datamodule.statistics, self.atom_encoder)
        self.input_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=1, pos=3, 
                                      pharma_feat = len(FAMILY_MAPPING), pharma_coord=3)
        self.output_dims = PlaceHolder(X=self.num_atom_types, charges=6, E=5, y=0, pos=3, 
                                       pharma_feat = len(FAMILY_MAPPING), pharma_coord=3)

    def to_one_hot(self, X, charges, E, node_mask):
        X = F.one_hot(X, num_classes=self.num_atom_types).float()
        E = F.one_hot(E, num_classes=5).float()
        charges = F.one_hot(charges + 2, num_classes=6).float()
        placeholder = PlaceHolder(X=X, charges=charges, E=E, y=None, pos=None, pharma_coord=None, pharma_feat=None)
        pl = placeholder.mask(node_mask)
        return pl.X, pl.charges, pl.E

    def one_hot_charges(self, charges):
        return F.one_hot((charges + 2).long(), num_classes=6).float()
    
    def pharma_to_one_hot(self, X, pharma_mask):
        X = F.one_hot(X, num_classes=len(FAMILY_MAPPING)+1).float()
        X = X[:, :, 1:] * pharma_mask.unsqueeze(-1)
        placeholder = PlaceHolder(X=None, charges=None, E=None,  y=None, pos=None, pharma_coord=None, pharma_feat=X,  
                                  pharma_atom=None, pharma_atom_pos=None)
        #pl = placeholder.mask(pharma_mask=pharma_mask)
        return placeholder.pharma_feat
    

