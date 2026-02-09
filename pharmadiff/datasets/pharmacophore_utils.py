import os.path
from collections import defaultdict
from typing import Iterable, Dict, List, Tuple
import random

import numpy as np
from rdkit import RDConfig, Chem
from rdkit.Chem import AllChem

import torch
from torch_geometric.data import Data
import torch.nn.functional as F

PHARMACOPHORE_FAMILES_TO_KEEP= ('Aromatic', 'Hydrophobe', 'PosIonizable', 'Acceptor', 'Donor', 
                                 'LumpedHydrophobe')
FAMILY_MAPPING = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3, 'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}
phar2idx = {"AROM": 1, "HYBL": 2, "POSC": 3, "HACC": 4, "HDON": 5, "LHYBL": 6, "UNKONWN": 0}
_FEATURES_FACTORY = []


def sample_probability(elment_array, plist, N):
    Psample = []
    n = len(plist)
    index = int(random.random() * n)
    mw = max(plist)
    beta = 0.0
    for i in range(N):
        beta = beta + random.random() * 2.0 * mw
        while beta > plist[index]:
            beta = beta - plist[index]
            index = (index + 1) % n
        Psample.append(elment_array[index])

    return Psample

def get_features_factory(features_names, resetPharmacophoreFactory=False):
    global _FEATURES_FACTORY, _FEATURES_NAMES
    if resetPharmacophoreFactory or (len(_FEATURES_FACTORY) > 0 and _FEATURES_FACTORY[-1] != features_names):
        _FEATURES_FACTORY.pop()
        _FEATURES_FACTORY.pop()
    if len(_FEATURES_FACTORY) == 0:
        feature_factory = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
        _FEATURES_NAMES = features_names
        if features_names is None:
            features_names = list(feature_factory.GetFeatureFamilies())

        _FEATURES_FACTORY.extend([feature_factory, features_names])
    return _FEATURES_FACTORY


def getPharamacophoreCoords(mol, features_names: Iterable[str] = PHARMACOPHORE_FAMILES_TO_KEEP, confId:int=-1) -> \
        Tuple[Dict[str, np.ndarray],  Dict[str, List[np.ndarray]]] :

    feature_factory, keep_featnames = get_features_factory(features_names)
    Chem.GetSSSR(mol)
    Chem.SanitizeMol(mol)
    rawFeats = feature_factory.GetFeaturesForMol(mol, confId=confId)

    feat_arr = np.empty(0)
    idx_arr_list = []
    coord_arr = np.empty(0)


    
    for f in rawFeats:
        if f.GetFamily() in keep_featnames:
            idx_arr = np.empty(0)
            if len(f.GetAtomIds()) > 1:
                for idx in f.GetAtomIds():
                    idx_arr = np.append(idx_arr, np.array(idx))

            else:
                idx_arr = np.append(idx_arr, np.array(list(f.GetAtomIds())))
                atom = mol.GetAtomWithIdx(f.GetAtomIds()[0])
                # neibours = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
                # idx_arr = np.append(idx_arr, np.array(neibours))
                
            
            feat_arr = np.append(feat_arr, np.array(FAMILY_MAPPING[f.GetFamily()]))
            coord_arr= np.append(coord_arr, np.array(f.GetPos(confId=f.GetActiveConformer())))
            idx_arr_list.append(idx_arr)
                
    coord_arr = coord_arr.reshape(-1, 3)
    #overlap_atoms = hydrophobes & lumped_hydrophobes_and_arom 
    
    permuted_indices = np.random.permutation(range(len(feat_arr))).astype(int)
    feat_arr = feat_arr[permuted_indices] 
    idx_arr_list = [idx_arr_list[i] for i in permuted_indices]
    coord_arr = coord_arr[permuted_indices]
    
    
    new_feat_arr = np.empty(0)
    new_idx_arr_list = []
    new_coord_arr =  np.empty((0, coord_arr.shape[1]))
    
    
    used_atoms = set()
    
    for i, idx_list in enumerate(idx_arr_list):
        if any(idx in idx_list for idx in used_atoms):
            continue    
        new_feat_arr = np.append(new_feat_arr, feat_arr[i])
        new_coord_arr = np.append(new_coord_arr, [coord_arr[i]], axis=0) 
        new_idx_arr_list.append(idx_list)
        used_atoms.update(idx_list)

    if new_feat_arr.shape[0] == 0:
        return new_feat_arr, new_idx_arr_list, new_coord_arr
    
    assert len(new_feat_arr) == len(new_idx_arr_list) == len(new_coord_arr), \
    f"Length mismatch: feat_arr({len(new_feat_arr)}), idx_arr_list({len(new_idx_arr_list)}), coord_arr({len(new_coord_arr)})"
            
    return new_feat_arr, new_idx_arr_list, new_coord_arr

def pharmacophore_to_torch(feat_arr, idx_arr_list, coord_arr, pos_mean, mol, name):
    
    n = mol.GetNumAtoms()
    
    if len(feat_arr) < 2 :
        return None
    
    if name == 'qm9':
        num = [2, 3, 4]
        num_p = [0.333, 0.334, 0.333]  # P(Number of Pharmacophore points)
        num_ = sample_probability(num, num_p, 1)
    elif name == 'geom':
        num = [3, 4, 5, 6, 7]
        num_p = [0.086, 0.0864, 0.389, 0.495, 0.0273]  # P(Number of Pharmacophore points)
        num_ = sample_probability(num, num_p, 1)
    
    
    num = num_[0]
    feat_array, coord_array, mask_array = make_pharmacophore(num, n, feat_arr, idx_arr_list, coord_arr)
            
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    
    # Check if the sum of the mask array exceeds the number of heavy atoms
    while mask_array.sum() >= num_heavy_atoms:
        #print(f"Pharmacophore points ({mask_array.sum()}) are more than or equal to the number of heavy atoms in the molecule ({num_heavy_atoms}).")
        num = num - 1
        feat_array, coord_array, mask_array = make_pharmacophore(num, n, feat_arr, idx_arr_list, coord_arr)

    coord_array = torch.Tensor(coord_array).float()

    coord_array = coord_array - pos_mean    
    pharma_feat = torch.Tensor(feat_array).long()
    mask_array = torch.Tensor(mask_array).long()
    #coord_array = coord_array - torch.mean(coord_array, dim=0, keepdim=True)
    coord_array = coord_array  * mask_array.unsqueeze(-1)

    pharma = Data(x = pharma_feat, pos = coord_array, y=mask_array)

    return pharma

def make_pharmacophore(num, n, feat_arr, idx_arr_list, coord_arr):
    if len(feat_arr) >= int(num):
        indices = np.random.choice(len(feat_arr), size=int(num), replace=False)
        feat_arr = feat_arr[indices]
        idx_arr_list = [idx_arr_list[i] for i in indices]
        coord_arr = coord_arr[indices]
         
    
    feat_array = np.zeros(n)
    coord_array = np.zeros((n, 3))
    mask_array = np.zeros(n)
    
    
    for i, idx_list in enumerate(idx_arr_list):
        for idx in idx_list:
            feat_array[int(idx)] = feat_arr[i]
            coord_array[int(idx)] = coord_arr[i]
            mask_array[int(idx)] = 1
            
    return feat_array, coord_array, mask_array


def mol_to_torch_pharmacophore(mol, pos_mean, name=None):
    feat_arr, idx_arr, coord_arr = getPharamacophoreCoords(mol)
    pharma_data =  pharmacophore_to_torch(feat_arr, idx_arr, coord_arr, pos_mean, mol, name=name)
    return pharma_data


def mol_to_torch_pharmacophore_mol_metrics(mol):
    feat_arr, idx_arr, coord_arr = getPharamacophoreCoords(mol)
    feat_arr, idx_arr, coord_arr=  pharmacophore_to_torch_mol_metrics(feat_arr, idx_arr, coord_arr)
    return feat_arr, idx_arr, coord_arr

def pharmacophore_to_torch_mol_metrics(feat_list, idx_list, coord_list,):
            
    feat_arr = np.empty(0)
    idx_arr = np.empty(0)
    coord_arr = np.empty(0)
    
    for i, idx in enumerate(idx_list):
        feat_arr = np.append(feat_arr, feat_list[i])
        idx_arr = np.append(idx_arr, idx_list[i])
        coord_arr = np.append(coord_arr, coord_list[i])


    return feat_arr, idx_arr, coord_arr


def load_phar_file(file_path):
    load_file_fn = {'.posp': load_pp_file}.get(file_path.suffix, None)

    if load_file_fn is None:
        raise ValueError(f'Invalid file path: "{file_path}"!')

    return load_file_fn(file_path)

def load_pp_file(file_path):
    node_type = []
    #node_size = []
    node_pos = []  # [(x,y,z)]

    lines = file_path.read_text().strip().split('\n')

    n_nodes = len(lines)

    assert n_nodes <= 7


    for line in lines:
        types, x, y, z = line.strip().split()
        
        tp = phar2idx.get(types, 0)

        node_type.append(tp)
        #node_size.append(size)
        node_pos.append(tuple(float(i) for i in (x, y, z)))

    node_type = np.array(node_type)
    #node_size = np.array(node_size)
    node_pos = np.array(node_pos)
    
    return node_type, node_pos


def load_ep_file(file_path):
   ## to be implemented
    return None

def prepare_pharma_data(sample_condition, n_nodes, bs, name, remove_hydrogens=True):
    node_type, node_pos = load_phar_file(sample_condition)
    
    min_nodes = torch.min(n_nodes).item()
    
    assert min_nodes < len(node_type), "Error: Pharmacophore points are more than the number of atoms in the molecule"

    if name == 'qm9':
        atom_encoder =  {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    elif name == 'geom':
        atom_encoder = {'H': 0, 'B': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7,
                        'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Br': 12, 'I': 13, 'Hg': 14, 'Bi': 15} 
        
    n = len(node_type)
    
    feat_array = np.empty(0)
    coord_array = np.empty(0)
    mask_array = np.empty(0)
    atoms_pos = np.empty(0)
    atoms_types = np.empty(0)
    atoms_charges = np.empty(0)
    
    for i in range(n):
        if node_type[i] not in [1, 6]:
            feat_array = np.append(feat_array, node_type[i])
            mask_array = np.append(mask_array, 1)
            coord_array = np.append(coord_array, node_pos[i])
            atoms_pos = np.append(atoms_pos, node_pos[i])
            atoms_charges = np.append(atoms_charges, 0)
            if node_type[i] == 2:
                # hydrophobe
                atoms_types = np.append(atoms_types, atom_encoder['C'])
    
            if node_type[i] == 3:
                # PosIonizable
                atoms_types = np.append(atoms_types, atom_encoder['N'])
                
            if node_type[i] == 4:
                # Acceptor
                atoms_types = np.append(atoms_types, random.choice([atom_encoder['O'], atom_encoder['N']]))
            
            if node_type[i] == 5:
                # Donor
                atoms_types = np.append(atoms_types, random.choice([atom_encoder['O'], atom_encoder['N']]))
        
        elif node_type[i] == 1:
            print("Aromatic")

    # Generate n random integers less than min_nodes
    random_numbers = np.random.randint(0, min_nodes, size=n)
    
    feat_array = np.zeros(min_nodes)
    coord_array = np.zeros((min_nodes, 3))
    mask_array = np.zeros(min_nodes)
    atoms_pos = torch.zeros((min_nodes, 3))
    atoms_types = torch.zeros(min_nodes)
    atoms_charges = torch.zeros(min_nodes)
    
    for i, idx in enumerate(random_numbers):
        feat_array[int(idx)] = node_type[i]
        coord_array[int(idx)] = node_pos[i]
        mask_array[int(idx)] = 1


    coord_array = torch.Tensor(coord_array).float()

    coord_array = coord_array - torch.mean(coord_array, dim=0, keepdim=True)
    pharma_feat = torch.Tensor(feat_array).long()
    mask_array = torch.tensor(mask_array).long()
    
    X = F.one_hot(pharma_feat, num_classes=len(FAMILY_MAPPING)+1).float()
    
    bs_coord_array = coord_array.repeat(bs, 1, 1)
    bs_X = X.repeat(bs, 1, 1)
    bs_mask_array = mask_array.repeat(bs, 1)
    
    print(bs_coord_array.shape, bs_X.shape, bs_mask_array.shape)
    
    return bs_coord_array, bs_X, bs_mask_array
    