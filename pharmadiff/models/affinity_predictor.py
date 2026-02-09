import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from torch_geometric.nn import GraphConv

class EGNNConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = GraphConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, coord: torch.Tensor = None) -> torch.Tensor:
        return self.conv(x, edge_index)

class TimeAwareAffinityPredictor(nn.Module):
    def __init__(self, ligand_in_dim=15, protein_in_dim=8, hidden_dim=64):
        super().__init__()
        
        # 1. Feature Embeddings
        self.lig_emb = nn.Linear(ligand_in_dim, hidden_dim)
        self.prot_emb = nn.Linear(protein_in_dim, hidden_dim)
        
        # 2. Time Embedding (Sinusoidal - same as Diffusion Model)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Time-Conditioned EGNN Layers
        self.conv1 = EGNNConv(hidden_dim, hidden_dim)
        self.conv2 = EGNNConv(hidden_dim, hidden_dim)
        self.conv3 = EGNNConv(hidden_dim, hidden_dim)
        
        # 4. Readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Predicts pKd
        )

    def _get_time_embedding(self, t, dim):
        # Standard Sinusoidal Embedding
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

    def _prepare_inputs(
        self,
        pos: torch.Tensor,
        feat: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pos.dim() == 3:
            if mask is None:
                raise ValueError("mask is required for dense (B, N, ...) inputs.")
            if mask.dtype != torch.bool:
                mask = mask.bool()
            batch_size, num_nodes = mask.shape
            batch_idx = torch.arange(batch_size, device=pos.device).repeat_interleave(num_nodes)
            pos = pos.reshape(-1, pos.size(-1))
            feat = feat.reshape(-1, feat.size(-1))
            mask_flat = mask.reshape(-1)
            pos = pos[mask_flat]
            feat = feat[mask_flat]
            batch_idx = batch_idx[mask_flat]
        else:
            if batch_idx is None:
                batch_idx = torch.zeros(pos.size(0), dtype=torch.long, device=pos.device)
        return pos, feat, batch_idx

    def forward(
        self,
        lig_pos: torch.Tensor,
        lig_feat: torch.Tensor,
        prot_pos: torch.Tensor,
        prot_feat: torch.Tensor,
        t: torch.Tensor,
        lig_batch: Optional[torch.Tensor] = None,
        prot_batch: Optional[torch.Tensor] = None,
        lig_mask: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            t: (Batch_Size, ) tensor of time steps
        """
        lig_pos, lig_feat, lig_batch = self._prepare_inputs(
            lig_pos, lig_feat, batch_idx=lig_batch, mask=lig_mask
        )
        prot_pos, prot_feat, prot_batch = self._prepare_inputs(
            prot_pos, prot_feat, batch_idx=prot_batch, mask=prot_mask
        )

        if t.dim() == 0:
            num_graphs = int(lig_batch.max().item()) + 1 if lig_batch.numel() > 0 else 1
            t = t.expand(num_graphs)

        # Embed Time
        t_emb = self._get_time_embedding(t, 64) # Assuming hidden_dim=64
        t_emb = self.time_mlp(t_emb)
        
        # Combine Graphs (Ligand + Protein)
        t_node = t_emb[lig_batch]  # Inject per-graph time into ligand nodes
        x_lig = self.lig_emb(lig_feat) + t_node
        x_prot = self.prot_emb(prot_feat) # Protein is clean, no time needed (or add it too)
        
        x = torch.cat([x_lig, x_prot], dim=0)
        pos = torch.cat([lig_pos, prot_pos], dim=0)
        
        # Create edges (simplified radius graph)
        batch_idx = torch.cat([lig_batch, prot_batch], dim=0)
        edge_index = self._build_edges(pos, batch_idx) 

        # Message Passing
        x = self.conv1(x, edge_index, coord=pos)
        x = self.conv2(x, edge_index, coord=pos)
        x = self.conv3(x, edge_index, coord=pos)
        
        # Global Pooling (Only pool the LIGAND nodes for prediction)
        from torch_geometric.nn import global_mean_pool
        ligand_readout = global_mean_pool(x[:lig_pos.shape[0]], lig_batch)
        
        return self.readout(ligand_readout)

    def _build_edges(self, pos, batch_idx, radius=5.0):
        from torch_cluster import radius_graph
        return radius_graph(pos, r=radius, batch=batch_idx)
