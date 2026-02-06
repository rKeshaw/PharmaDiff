import torch
import torch.nn as nn
from torch_geometric.nn import EGNNConv
import math

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

    def forward(self, lig_pos, lig_feat, prot_pos, prot_feat, t):
        """
        Args:
            t: (Batch_Size, ) tensor of time steps
        """
        # Embed Time
        t_emb = self._get_time_embedding(t, 64) # Assuming hidden_dim=64
        t_emb = self.time_mlp(t_emb)
        
        # Combine Graphs (Ligand + Protein)
        # Handle Batching
        if lig_batch is None: # Fallback for Batch Size 1
            lig_batch = torch.zeros(lig_pos.shape[0], dtype=torch.long, device=lig_pos.device)
        if prot_batch is None:
            prot_batch = torch.zeros(prot_pos.shape[0], dtype=torch.long, device=prot_pos.device)
            
        x_lig = self.lig_emb(lig_feat) + t_emb # Inject time into features
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
