import pickle

from rdkit import Chem
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def mol_to_torch_geometric(mol, atom_encoder, smiles):
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    pos_mean = torch.mean(pos, dim=0, keepdim=True)
    pos = pos - pos_mean
    atom_types = []
    all_charges = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())        # TODO: check if implicit Hs should be kept

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    data = Data(x=atom_types, edge_index=edge_index, edge_attr=edge_attr, pos=pos, charges=all_charges,
                smiles=smiles)
    return data, pos_mean


def remove_hydrogens(data: Data, pharma_data: Data, pocket_data: Data = None):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(to_keep, data.edge_index, data.edge_attr, relabel_nodes=True,
                                             num_nodes=len(to_keep))    
    mean_pos = torch.mean(data.pos[to_keep], dim=0)
    new_pos = data.pos[to_keep] - mean_pos
    mask_array = pharma_data.y[to_keep].unsqueeze(-1)
    pharma_pos = pharma_data.pos[to_keep] - mean_pos
    pharma_pos = pharma_pos * mask_array
    ligand_res = Data(x=data.x[to_keep] - 1, pos=new_pos, charges=data.charges[to_keep], edge_index=new_edge_index,
                      edge_attr=new_edge_attr)
    pharma_res = Data(x=pharma_data.x[to_keep], pos=pharma_pos, y=pharma_data.y[to_keep])
    if pocket_data is not None:
        pocket_pos = pocket_data.pos - mean_pos
        pocket_res = Data(x=pocket_data.x, pos=pocket_pos)
        return ligand_res, pharma_res, pocket_res
    return ligand_res, pharma_res


def save_pickle(array, path):
    with open(path, 'wb') as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class Statistics:
    def __init__(self, num_nodes, atom_types, bond_types, charge_types, valencies, bond_lengths, bond_angles):
        self.num_nodes = num_nodes
        print("NUM NODES IN STATISTICS", num_nodes)
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
