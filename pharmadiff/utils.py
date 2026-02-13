import os
from copy import deepcopy
from typing import Optional, Union, Dict

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops, dense_to_sparse
from torchmetrics import Metric, MeanSquaredError, MeanAbsoluteError,MetricCollection,KLDivergence
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict
import wandb


# from dgd.ggg_utils_deps import approx_small_symeig, our_small_symeig,extract_canonical_k_eigenfeat
# from dgd.ggg_utils_deps import  ensure_tensor, get_laplacian, asserts_enabled


class NoSyncMetricCollection(MetricCollection):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs) #disabling syncs since it messes up DDP sub-batching


class NoSyncMetric(Metric):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncKL(KLDivergence):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncMSE(MeanSquaredError):
    def __init__(self):
        super().__init__(sync_on_compute=False, dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching


class NoSyncMAE(MeanAbsoluteError):
    def __init__(self):
        super().__init__(sync_on_compute=False,dist_sync_on_step=False) #disabling syncs since it messes up DDP sub-batching

# Folders
def create_folders(args):
    try:
        # os.makedirs('checkpoints')
        os.makedirs('graphs', exist_ok=True)
        os.makedirs('chains', exist_ok=True)
    except OSError:
        pass

    try:
        # os.makedirs('checkpoints/' + args.general.name)
        os.makedirs('graphs/' + args.general.name, exist_ok=True)
        os.makedirs('chains/' + args.general.name, exist_ok=True)
    except OSError:
        pass


def to_dense(data, dataset_info, device=None):
    if data is None:
        return None
    if data:
        X, node_mask = to_dense_batch(x=data['ligand'].x, batch=data['ligand'].batch)
    pos, _ = to_dense_batch(x=data['ligand'].pos, batch=data['ligand'].batch)
    pos = pos.float()
    assert pos.mean(dim=1).abs().max() < 1e-3
    charges, _ = to_dense_batch(x=data['ligand'].charges, batch=data['ligand'].batch)
    max_num_nodes = X.size(1)
    edge_index, edge_attr = remove_self_loops(data['ligand'].edge_index , data['ligand'].edge_attr )
    E = to_dense_adj(edge_index=edge_index, batch=data['ligand'].batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    if E.numel() > 0:
        E = torch.maximum(E, E.transpose(1, 2))
    X, charges, E = dataset_info.to_one_hot(X, charges=charges, E=E, node_mask=node_mask)


    y = X.new_zeros((X.shape[0], 0))
    pharma_y_mask, pharma_batch_mask = to_dense_batch(x=data['pharmacophore'].y, batch=data['pharmacophore'].batch)
    #pharma_mask = pharma_y_mask * pharma_batch_mask
    pharma_mask = pharma_y_mask.bool()

    pharma_feat, _ = to_dense_batch(x=data['pharmacophore'].x, batch=data['pharmacophore'].batch)
    pharma_feat = dataset_info.pharma_to_one_hot(pharma_feat, pharma_mask)


    pharma_coord, _= to_dense_batch(x=data['pharmacophore'].pos, batch=data['pharmacophore'].batch)
    pharma_coord = pharma_coord.float() 
    

    #node_features = torch.cat((X, charges), dim=-1).clone()
    pharma_atom = X * pharma_mask.unsqueeze(-1)
    pharma_charge = charges * pharma_mask.unsqueeze(-1)
    pharma_atom_pos = pos * pharma_mask.unsqueeze(-1)

    pharma_sum = pharma_mask.sum(dim=-1)
    node_sum = node_mask.sum(dim=-1)

    # Condition: pharma_sum > 0 and pharma_sum < node_sum
    valid_rows = (pharma_sum > 0) & (pharma_sum < node_sum)
    
    pharma_E = get_frag_edges(pharma_mask, E)
    pocket_pos = data.get('pocket_pos') if isinstance(data, dict) else None
    pocket_feat = data.get('pocket_feat') if isinstance(data, dict) else None
    pocket_batch = data.get('pocket_batch') if isinstance(data, dict) else None
    plip_labels = data.get('plip_labels') if isinstance(data, dict) else None
    plip_label_mask = data.get('plip_label_mask') if isinstance(data, dict) else None
    pocket_mask = None
    if pocket_pos is not None and pocket_batch is not None:
        if pocket_pos.numel() == 0:
            pocket_pos = None
            pocket_feat = None
            pocket_mask = None
            pocket_batch = None
        else:
            pocket_pos_dense, pocket_mask = to_dense_batch(x=pocket_pos, batch=pocket_batch)
            pocket_pos = pocket_pos_dense
            if pocket_feat is not None:
                pocket_feat, _ = to_dense_batch(x=pocket_feat, batch=pocket_batch)
            pocket_batch = None

    if device is not None:
        X = X.to(device)
        charges = charges.to(device)
        E = E.to(device)
        y = y.to(device)
        pos = pos.to(device)
        node_mask = node_mask.to(device)
        pharma_coord = pharma_coord.to(device)
        pharma_feat = pharma_feat.to(device)
        pharma_mask = pharma_mask.to(device)
        pharma_atom = pharma_atom.to(device)
        pharma_charge = pharma_charge.to(device)
        pharma_atom_pos = pharma_atom_pos.to(device)
        pharma_E = pharma_E.to(device)
        if pocket_pos is not None:
            pocket_pos = pocket_pos.to(device)
        if pocket_feat is not None:
            pocket_feat = pocket_feat.to(device)
        if pocket_mask is not None:
            pocket_mask = pocket_mask.to(device)
        if plip_labels is not None:
            plip_labels = plip_labels.to(device)
        if plip_label_mask is not None:
            plip_label_mask = plip_label_mask.to(device)

    data = PlaceHolder(
        X=X[valid_rows],
        charges=charges[valid_rows],
        pos=pos[valid_rows],
        E=E[valid_rows],
        y=y[valid_rows],
        node_mask=node_mask[valid_rows],
        pharma_feat=pharma_feat[valid_rows],
        pharma_coord=pharma_coord[valid_rows],
        pharma_mask=pharma_mask[valid_rows],
        pharma_atom=pharma_atom[valid_rows],
        pharma_atom_pos=pharma_atom_pos[valid_rows],
        pharma_E=pharma_E[valid_rows],
        pharma_charge=pharma_charge[valid_rows],
        pocket_pos=pocket_pos[valid_rows] if pocket_pos is not None else None,
        pocket_feat=pocket_feat[valid_rows] if pocket_feat is not None else None,
        pocket_batch=None,
        pocket_mask=pocket_mask[valid_rows] if pocket_mask is not None else None,
        plip_labels=plip_labels[valid_rows] if plip_labels is not None else None,
        plip_label_mask=plip_label_mask[valid_rows] if plip_label_mask is not None else None,
    )

    return data.mask()


def get_frag_edges(pharma_mask, edges):

    
    p_mask = pharma_mask.unsqueeze(-1)
    square_p_mask = p_mask.unsqueeze(2) * p_mask.unsqueeze(1)


    pharma_edges = edges * square_p_mask
        
    return pharma_edges
    
        

class PlaceHolder:
    def __init__(
        self,
        pos,
        X,
        charges,
        E,
        y,
        pharma_feat=None,
        pharma_coord=None,
        t_int=None,
        t=None,
        node_mask=None,
        pharma_mask=None,
        pharma_atom=None,
        pharma_atom_pos=None,
        pharma_E=None,
        pharma_charge=None,
        pocket_pos=None,
        pocket_feat=None,
        pocket_batch=None,
        pocket_mask=None,
        ref_ligand_pos=None,
        ref_ligand_atom_types=None,
        plip_labels=None,
        plip_label_mask=None,
    ):
        self.pos = pos
        self.X = X
        self.charges = charges
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask
        self.pharma_feat = pharma_feat
        self.pharma_coord = pharma_coord
        self.pharma_mask = pharma_mask
        self.pharma_atom = pharma_atom
        self.pharma_charge = pharma_charge
        self.pharma_atom_pos = pharma_atom_pos  
        self.pharma_E = pharma_E
        self.pocket_pos = pocket_pos
        self.pocket_feat = pocket_feat
        self.pocket_batch = pocket_batch
        self.pocket_mask = pocket_mask
        self.ref_ligand_pos = ref_ligand_pos
        self.ref_ligand_atom_types = ref_ligand_atom_types
        self.plip_labels = plip_labels
        self.plip_label_mask = plip_label_mask

    def device_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.pos = self.pos.to(x.device) if self.pos is not None else None
        self.X = self.X.to(x.device) if self.X is not None else None
        self.charges = self.charges.to(x.device) if self.charges is not None else None
        self.E = self.E.to(x.device) if self.E is not None else None
        self.y = self.y.to(x.device) if self.y is not None else None
        self.pharma_feat = self.pharma_feat.to(x.device) if self.pharma_feat is not None else None 
        self.pharma_coord = self.pharma_coord.to(x.device) if self.pharma_feat is not None else None 
        self.pharma_mask = self.pharma_mask.to(x.device) if self.pharma_mask is not None else None
        self.pharma_atom = self.pharma_atom.to(x.device) if self.pharma_atom is not None else None
        self.pharma_atom_pos = self.pharma_atom_pos.to(x.device) if self.pharma_atom_pos is not None else None
        self.pharma_E = self.pharma_E.to(x.device) if self.pharma_E is not None else None
        self.pocket_pos = self.pocket_pos.to(x.device) if self.pocket_pos is not None else None
        self.pocket_feat = self.pocket_feat.to(x.device) if self.pocket_feat is not None else None
        self.pocket_batch = self.pocket_batch.to(x.device) if self.pocket_batch is not None else None
        self.pocket_mask = self.pocket_mask.to(x.device) if self.pocket_mask is not None else None
        self.ref_ligand_pos = self.ref_ligand_pos.to(x.device) if self.ref_ligand_pos is not None else None
        self.ref_ligand_atom_types = (
            self.ref_ligand_atom_types.to(x.device) if self.ref_ligand_atom_types is not None else None
        )
        self.plip_labels = self.plip_labels.to(x.device) if self.plip_labels is not None else None
        self.plip_label_mask = self.plip_label_mask.to(x.device) if self.plip_label_mask is not None else None
        return self

    def mask(self, node_mask=None):
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1
        diag_mask = ~torch.eye(n, dtype=torch.bool,
                               device=node_mask.device).unsqueeze(0).expand(bs, -1, -1).unsqueeze(-1)  # bs, n, n, 1

        if self.X is not None:
            self.X = self.X * x_mask
        if self.charges is not None:
            self.charges = self.charges * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2 * diag_mask
        if self.pos is not None:
            self.pos = self.pos * x_mask
            mean_pos = self.pos.mean(dim=1, keepdim=True)
            self.pos = self.pos - mean_pos
        #if self.pharma_coord is not None:
            #self.pharma_coord = self.pharma_coord - self.pharma_coord.mean(dim=1, keepdim=True)
            #self.pharma_coord = self.pharma_coord * x_mask
        assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


    def collapse(self, collapse_charges):
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        copy.charges = collapse_charges.to(self.charges.device)[torch.argmax(self.charges, dim=-1)]
        copy.E = torch.argmax(self.E, dim=-1)
        x_mask = self.node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        copy.X[self.node_mask == 0] = - 1
        copy.charges[self.node_mask == 0] = 1000
        copy.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        copy.pharma_feat = torch.argmax(self.pharma_feat, dim=-1)
        copy.pharma_feat[self.pharma_mask == 0] = - 1     
        copy.pharma_atom = torch.argmax(copy.pharma_atom, dim=-1)
        copy.pharma_charge = collapse_charges.to(self.charges.device)[torch.argmax(copy.pharma_charge, dim=-1)]
        copy.pharma_atom[copy.pharma_mask == 0] = - 1
        copy.pharma_charge[copy.pharma_mask == 0] = 1000
        p_mask = self.pharma_mask.unsqueeze(-1)  # bs, n, 1
        pe_mask1 = p_mask.unsqueeze(2)  # bs, n, 1, 1
        pe_mask2 = p_mask.unsqueeze(1)  # bs, 1, n, 1
        
        copy.pharma_E = torch.argmax(self.pharma_E, dim=-1)
        copy.pharma_E[(pe_mask1 * pe_mask2).squeeze(-1) == 0] = - 1
        return copy

    def __repr__(self):
        return (f"pos: {self.pos.shape if type(self.pos) == torch.Tensor else self.pos} -- " +
                f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- " +
                f"charges: {self.charges.shape if type(self.charges) == torch.Tensor else self.charges} -- " +
                f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- " +
                f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}")


    def copy(self):
        return PlaceHolder(
            X=self.X,
            charges=self.charges,
            E=self.E,
            y=self.y,
            pos=self.pos,
            t_int=self.t_int,
            t=self.t,
            node_mask=self.node_mask,
            pharma_coord=self.pharma_coord,
            pharma_feat=self.pharma_feat,
            pharma_mask=self.pharma_mask,
            pharma_atom=self.pharma_atom,
            pharma_atom_pos=self.pharma_atom_pos,
            pharma_E=self.pharma_E,
            pharma_charge=self.pharma_charge,
            pocket_pos=self.pocket_pos,
            pocket_feat=self.pocket_feat,
            pocket_batch=self.pocket_batch,
            pocket_mask=self.pocket_mask,
            ref_ligand_pos=self.ref_ligand_pos,
            ref_ligand_atom_types=self.ref_ligand_atom_types,
            plip_labels=self.plip_labels,
            plip_label_mask=self.plip_label_mask,
        )


def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': f'MolDiffusion_pharmacophore{cfg.dataset["name"]}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


def remove_mean_with_mask(x, node_mask):
    """ x: bs x n x d.
        node_mask: bs x n """
    assert node_mask.dtype == torch.bool, f"Wrong type {node_mask.dtype}"
    node_mask = node_mask.unsqueeze(-1)
    masked_max_abs_value = (x * (~node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    #if pharma_coords is not None:
    #    pharma_coords = pharma_coords - mean * pharma_mask.unsqueeze(-1)
    return x, mean
