from typing import Optional,Tuple

import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

import pharmadiff.utils as utils
from pharmadiff.diffusion import diffusion_utils
from pharmadiff.models.layers import Xtoy, Etoy, SE3Norm, PositionsMLP, masked_softmax, EtoX, SetNorm, GraphNorm

class ConditioningFusion(nn.Module):
    def __init__(self, cond_dim: int, fusion_type: str = "concat") -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.fusion_type = fusion_type
        self.pocket_proj = nn.Linear(8, cond_dim)
        if fusion_type == "concat":
            self.fuse_proj = nn.Linear(cond_dim * 2, cond_dim)
        elif fusion_type == "gated":
            self.gate_proj = nn.Linear(cond_dim * 2, cond_dim)
        elif fusion_type == "cross-attn":
            self.attn = nn.MultiheadAttention(cond_dim, num_heads=1, batch_first=True)
            self.attn_norm = nn.LayerNorm(cond_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def _pool_pocket(self, pocket_feat: torch.Tensor, pocket_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if pocket_mask is None:
            return pocket_feat.mean(dim=1, keepdim=True)
        masked_feat = pocket_feat * pocket_mask.unsqueeze(-1)
        denom = pocket_mask.sum(dim=1, keepdim=True).clamp(min=1).unsqueeze(-1)
        return masked_feat.sum(dim=1, keepdim=True) / denom

    def forward(
        self,
        pharma_feat: torch.Tensor,
        pocket_feat: Optional[torch.Tensor] = None,
        pocket_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pocket_feat is None:
            return pharma_feat

        pocket_feat_proj = self.pocket_proj(pocket_feat)

        if self.fusion_type == "concat":
            pocket_summary = self._pool_pocket(pocket_feat_proj, pocket_mask)
            pocket_summary = pocket_summary.expand(pharma_feat.size(0), pharma_feat.size(1), -1)
            return self.fuse_proj(torch.cat([pharma_feat, pocket_summary], dim=-1))

        if self.fusion_type == "gated":
            pocket_summary = self._pool_pocket(pocket_feat_proj, pocket_mask)
            pocket_summary = pocket_summary.expand(pharma_feat.size(0), pharma_feat.size(1), -1)
            gate = torch.sigmoid(self.gate_proj(torch.cat([pharma_feat, pocket_summary], dim=-1)))
            return gate * pharma_feat + (1 - gate) * pocket_summary

        if pocket_mask is not None:
            valid_counts = pocket_mask.sum(dim=1)
            if (valid_counts == 0).any():
                return pharma_feat
            key_padding_mask = ~pocket_mask
        else:
            key_padding_mask = None

        attn_out, _ = self.attn(pharma_feat, pocket_feat_proj, pocket_feat_proj, key_padding_mask=key_padding_mask)
        return self.attn_norm(pharma_feat + attn_out)

class PocketInteractionBlock(nn.Module):
    """Ligand-to-pocket cross-attention block for explicit pocket awareness."""

    def __init__(self, ligand_dim: int, pocket_feat_dim: int = 8, n_head: int = 4):
        super().__init__()
        heads = max(1, min(n_head, ligand_dim))
        while ligand_dim % heads != 0 and heads > 1:
            heads -= 1
        self.pocket_feat_proj = nn.Linear(pocket_feat_dim, ligand_dim)
        self.pocket_pos_proj = nn.Sequential(
            nn.Linear(3, ligand_dim),
            nn.ReLU(),
            nn.Linear(ligand_dim, ligand_dim),
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=ligand_dim, num_heads=heads, batch_first=True)
        self.out_norm = nn.LayerNorm(ligand_dim)

    def forward(
        self,
        ligand_h: torch.Tensor,
        pocket_pos: Optional[torch.Tensor],
        pocket_feat: Optional[torch.Tensor],
        pocket_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if pocket_pos is None or pocket_feat is None:
            return ligand_h
        if pocket_pos.dim() != 3 or pocket_feat.dim() != 3:
            return ligand_h
        if pocket_pos.size(1) == 0 or pocket_feat.size(1) == 0:
            return ligand_h

        pocket_h = self.pocket_feat_proj(pocket_feat) + self.pocket_pos_proj(pocket_pos)

        key_padding_mask = None
        if pocket_mask is not None:
            if pocket_mask.dim() != 2:
                return ligand_h
            if (pocket_mask.sum(dim=1) == 0).any():
                return ligand_h
            key_padding_mask = ~pocket_mask

        attn_out, _ = self.cross_attn(
            query=ligand_h,
            key=pocket_h,
            value=pocket_h,
            key_padding_mask=key_padding_mask,
        )
        return self.out_norm(ligand_h + attn_out)
    
class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None, last_layer=False) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, last_layer=last_layer)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        # self.normX1 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        # self.normX2 = SetNorm(feature_dim=dx, eps=layer_norm_eps, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.norm_pos1 = SE3Norm(eps=layer_norm_eps, **kw)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        # self.normE1 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        # self.normE2 = GraphNorm(feature_dim=de, eps=layer_norm_eps, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.last_layer = last_layer
        if not last_layer:
            self.lin_y1 = Linear(dy, dim_ffy, **kw)
            self.lin_y2 = Linear(dim_ffy, dy, **kw)
            self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
            self.dropout_y1 = Dropout(dropout)
            self.dropout_y2 = Dropout(dropout)
            self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, features: utils.PlaceHolder):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        X = features.X
        old_X = features.X
        E = features.E
        y = features.y
        pos = features.pos
        node_mask = features.node_mask
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1
        
        pharma_mask = features.pharma_mask
        p_mask = features.pharma_mask.unsqueeze(-1)
        pharma_pos = features.pharma_atom_pos
        
        pharma_feat = features.pharma_feat
        pharma_atom = features.pharma_atom
        
        
        newX, newE, new_y, vel = self.self_attn(X, E, y, pos, node_mask=node_mask, 
                                                pharma_mask=pharma_mask, pharma_pos=pharma_pos, pharma_atom=pharma_atom)

        newX_d = self.dropoutX1(newX)
        # X = self.normX1(X + newX_d, x_mask)
        X = self.normX1(X + newX_d)
        # new_pos = pos + vel
        #new_vel = pharma_pos + vel
        new_pos = self.norm_pos1(vel, x_mask) + pos
        
                
        current_pos = torch.mean((new_pos * pharma_mask.unsqueeze(-1)) , dim=1, keepdim=True)
        target_pos = torch.mean((pharma_pos * pharma_mask.unsqueeze(-1)) , dim=1, keepdim=True)
        
        new_pos = new_pos - current_pos + target_pos
        
        new_pos = new_pos * ~p_mask + pharma_pos * p_mask
        
        if torch.isnan(new_pos).any():
            raise ValueError("NaN in new_pos")

        newE_d = self.dropoutE1(newE)
        # E = self.normE1(E + newE_d, e_mask1, e_mask2)
        E = self.normE1(E + newE_d)

        if not self.last_layer:
            new_y_d = self.dropout_y1(new_y)
            y = self.norm_y1(y + new_y_d)

        # skip connection
        #X = X + pharma_atom
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        # X = self.normX2(X + ff_outputX, x_mask)
        X = self.normX2(X + ff_outputX)

        
        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        # E = self.normE2(E + ff_outputE, e_mask1, e_mask2)
        E = self.normE2(E + ff_outputE)
        E = 0.5 * (E + torch.transpose(E, 1, 2))

        if not self.last_layer:
            ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
            ff_output_y = self.dropout_y3(ff_output_y)
            y = self.norm_y2(y + ff_output_y)
        
        # X = X * ~p_mask +  pharma_atom * p_mask

        out = utils.PlaceHolder(X=X, E=E, y=y, pos=new_pos, charges=None, node_mask=node_mask, 
                                pharma_coord=None, pharma_feat=pharma_feat, 
                                pharma_mask=pharma_mask, pharma_atom=pharma_atom, 
                                pharma_atom_pos=pharma_pos).mask()

        return out


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, last_layer=False):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        self.in_E = Linear(de, de)

        # FiLM X to E
        # self.x_e_add = Linear(dx, de)
        self.x_e_mul1 = Linear(dx, de)
        self.x_e_mul2 = Linear(dx, de)

        # Distance encoding
        self.lin_dist1 = Linear(2, de)
        self.lin_norm_pos1 = Linear(1, de)
        self.lin_norm_pos2 = Linear(1, de)

        self.dist_add_e = Linear(de, de)
        self.dist_mul_e = Linear(de, de)
        # self.lin_dist2 = Linear(dx, dx)
        
        # Pharma Distance encoding
        self.p_lin_dist1 = Linear(2, de)
        self.p_lin_norm_pos1 = Linear(1, de)
        self.p_lin_norm_pos2 = Linear(1, de)

        self.p_dist_add_e = Linear(de, de)
        self.p_dist_mul_e = Linear(de, de)

        # Attention
        self.k = Linear(dx, dx)
        self.q = Linear(dx, dx)
        self.v = Linear(dx, dx)
        self.a = Linear(dx, n_head, bias=False)
        self.out = Linear(dx * n_head, dx)
        
        
        # Cross attention 
        self.k_cross = Linear(dx, dx)
        self.q_cross = Linear(dx, dx)
        self.v_cross = Linear(dx, dx)
        self.a_cross = Linear(dx, n_head, bias=False)
        self.out_cross = Linear(dx * n_head, dx)

        # Incorporate e to x
        # self.e_att_add = Linear(de, n_head)
        self.e_att_mul = Linear(de, n_head)

        self.pos_att_mul = Linear(de, n_head)

        self.e_x_mul = EtoX(de, dx)

        self.pos_x_mul = EtoX(de, dx)


        # FiLM y to E
        self.y_e_mul = Linear(dy, de)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, de)

        self.pre_softmax = Linear(de, dx)       # Unused, but needed to load old checkpoints

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.last_layer = last_layer
        if not last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)
            self.dist_y = Etoy(de, dy)
      # For EGNN v3: map to pi, pj for pharmacophores
            

        # Process_pos
        self.e_pos1 = Linear( de, de, bias=False)
        self.e_pos2 = Linear( de, 1, bias=False)          # For EGNN v3: map to pi, pj

        
        
        # Cross attention to pharma_pos
        self.Q_pos = Linear(1, de)
        self.K_pos = Linear(1, de)
        self.V_pos = Linear(1, de)
        self.a_pos = Linear(de, n_head, bias=False)
        self.out_pos = Linear(de * n_head, 1)
        self.eps = 10e-5

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(de, de)
        if not last_layer:
            self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))
            
    def forward(self, X, E, y, pos, node_mask, pharma_mask, pharma_pos, pharma_atom):
        """ :param X: bs, n, d        node features
            :param E: bs, n, n, d     edge features
            :param y: bs, dz           global features
            :param pos: bs, n, 3
            :param node_mask: bs, n
            :return: newX, newE, new_y with the same shape. """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1
        
        p_mask = pharma_mask.unsqueeze(-1)
        pe_mask1 = p_mask.unsqueeze(2)
        pe_mask2 = p_mask.unsqueeze(1)

        # 0. Create a distance matrix that can be used later
        pos = pos * x_mask
        norm_pos = torch.norm(pos, dim=-1, keepdim=True)         # bs, n, 1
        normalized_pos = pos / (norm_pos + 1e-7)                 # bs, n, 3

        pairwise_dist = torch.cdist(pos, pos).unsqueeze(-1).float()
        cosines = torch.sum(normalized_pos.unsqueeze(1) * normalized_pos.unsqueeze(2), dim=-1, keepdim=True)
        pos_info = torch.cat((pairwise_dist, cosines), dim=-1)

        norm1 = self.lin_norm_pos1(norm_pos)             # bs, n, de
        norm2 = self.lin_norm_pos2(norm_pos)             # bs, n, de
        dist1 = F.relu(self.lin_dist1(pos_info) + norm1.unsqueeze(2) + norm2.unsqueeze(1)) * e_mask1 * e_mask2
        
        
        # # 0.1 Create a distance matrix for pharmacophores
        # pharma_pos = pharma_pos * p_mask
        # p_norm_pos = torch.norm(pharma_pos, dim=-1, keepdim=True)         # bs, n, 1
        # p_normalized_pos = pharma_pos / (p_norm_pos + 1e-7)                 # bs, n, 3

        # p_pairwise_dist = torch.cdist(pharma_pos, pharma_pos).unsqueeze(-1).float()
        # p_cosines = torch.sum(p_normalized_pos.unsqueeze(1) * p_normalized_pos.unsqueeze(2), dim=-1, keepdim=True)
        # p_pos_info = torch.cat((p_pairwise_dist, p_cosines), dim=-1)

        # p_norm1 = self.p_lin_norm_pos1(p_norm_pos)             # bs, n, de
        # p_norm2 = self.p_lin_norm_pos2(norm_pos)               # bs, n, de
        # p_dist1 = F.relu(self.p_lin_dist1(p_pos_info) + p_norm1.unsqueeze(2) + p_norm2.unsqueeze(1)) * pe_mask1 * pe_mask2

        # 1. Process E
        Y = self.in_E(E)

        # 1.1 Incorporate x
        x_e_mul1 = self.x_e_mul1(X) * x_mask
        x_e_mul2 = self.x_e_mul2(X) * x_mask
        Y = Y * x_e_mul1.unsqueeze(1) * x_e_mul2.unsqueeze(2) * e_mask1 * e_mask2
        
        # # 1.2. Incorporate pharma distances
        #p_dist_add = self.p_dist_add_e(p_dist1)
        #p_dist_mul = self.p_dist_mul_e(p_dist1)
        #Y_p = (Y + p_dist_add + Y * p_dist_mul) * pe_mask1 * pe_mask2   # bs, n, n, dx
        # #Y = Y + Y_p
        
        
        # 1.2. Incorporate distances
        dist_add = self.dist_add_e(dist1)
        dist_mul = self.dist_mul_e(dist1)
        Y = (Y + dist_add + Y * dist_mul) * e_mask1 * e_mask2   # bs, n, n, dx

        # 1.3 Incorporate y to E
        y_e_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        y_e_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        E = (Y + y_e_add + Y * y_e_mul) * e_mask1 * e_mask2

        # Output E
        Eout = self.e_out(E) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(Eout, e_mask1 * e_mask2)

        # 2. Process the node features
        Q = (self.q(X) * x_mask).unsqueeze(2)          # bs, 1, n, dx
        K = (self.k(X) * x_mask).unsqueeze(1)          # bs, n, 1, dx
        prod = Q * K / math.sqrt(Y.size(-1))   # bs, n, n, dx
        a = self.a(prod) * e_mask1 * e_mask2   # bs, n, n, n_head

        # 2.1 Incorporate edge features
        e_x_mul = self.e_att_mul(E)
        a = a + e_x_mul * a

        # 2.2 Incorporate position features
        pos_x_mul = self.pos_att_mul(dist1)
        a = a + pos_x_mul * a
        a = a * e_mask1 * e_mask2
        
        # 2.3 Self-attention
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        alpha = masked_softmax(a, softmax_mask, dim=2).unsqueeze(-1)  # bs, n, n, n_head        
        V = (self.v(X) * x_mask).unsqueeze(1).unsqueeze(3)      # bs, 1, n, 1, dx
        weighted_V = alpha * V                                  # bs, n, n, n_heads, dx
        weighted_V = weighted_V.sum(dim=2)                      # bs, n, n_head, dx
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, n_head x dx        
        weighted_V = self.out(weighted_V) * x_mask              # bs, n, dx
        
        
        # 2.4 Cross attention to pharma_atom
        Q_c = (self.q_cross(weighted_V) * p_mask).unsqueeze(2)                            # bs, 1, n, dx
        if torch.isnan(Q_c).any():
            print(f"NaN detected in Q_c")
        
        K_c = (self.k_cross(pharma_atom) * p_mask).unsqueeze(1)                   # bs, n, 1, dx
        if torch.isnan(K_c).any():
            print(f"NaN detected in K_c")
        
        prod_c = Q_c * K_c / math.sqrt(Y.size(-1))                                  # bs, n, n, dx
        prod_c = prod_c - prod_c.max(dim=-1, keepdim=True)[0]
        if torch.isnan(prod_c).any():
            print(f"NaN detected in prod_c")
        
        a_c = self.a_cross(prod_c) * p_mask.unsqueeze(1) * p_mask.unsqueeze(2)    # bs, n, n, n_head
        if torch.isnan(a_c).any():
            print(f"NaN detected in a_c")
        
        softmax_mask_c = pe_mask2.expand(-1, n, -1, self.n_head)
        alpha_c = masked_softmax(a_c, softmax_mask_c, dim=2).unsqueeze(-1)        # bs, n, n, n_head
        if torch.isnan(alpha_c).any():
            print(f"NaN detected in alpha_c")
        V_c = (self.v_cross(pharma_atom) * p_mask).unsqueeze(1).unsqueeze(3)      # bs, 1, n, 1, dx
        weighted_V_c = alpha_c * V_c                                              # bs, n, n, n_heads, dx
        weighted_V_c = weighted_V_c.sum(dim=2)                                    # bs, n, n_head, dx
        weighted_V_c = weighted_V_c.flatten(start_dim=2)                          # bs, n, n_head x dx
        weighted_V_c = self.out_cross(weighted_V_c) * p_mask                      # bs, n, dx
        
        
        weighted_V = weighted_V * ~p_mask + weighted_V_c * p_mask
        
        
        # Incorporate E to X
        e_x_mul = self.e_x_mul(E, e_mask2)
        weighted_V = weighted_V + e_x_mul * weighted_V

        pos_x_mul = self.pos_x_mul(dist1, e_mask2)
        weighted_V = weighted_V + pos_x_mul * weighted_V
        
        
        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)                     # bs, 1, dx
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = weighted_V * (yx2 + 1) + yx1

        # Output X
        Xout = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(Xout, x_mask)

        # Process y based on X and E
        if self.last_layer:
            y_out = None
        else:
            y = self.y_y(y)
            e_y = self.e_y(Y, e_mask1, e_mask2)
            x_y = self.x_y(newX, x_mask)
            dist_y = self.dist_y(dist1, e_mask1, e_mask2)
            new_y = y + x_y + e_y + dist_y
            y_out = self.y_out(new_y)               # bs, dy
                
        # Update the positions
        pos1 = pos.unsqueeze(1).expand(-1, n, -1, -1)              # bs, 1, n, 3
        pos2 = pos.unsqueeze(2).expand(-1, -1, n, -1)              # bs, n, 1, 3
        delta_pos = pos2 - pos1                                    # bs, n, n, 3

        #Y_m = torch.cat((Y, Y_p), dim=-1)                    # bs, n, n, de
        messages = self.e_pos2(F.relu(self.e_pos1(Y)))       # bs, n, n, 1, 2
        vel = (messages * delta_pos).sum(dim=2) * x_mask
        
        vel, _ = utils.remove_mean_with_mask(vel, node_mask)
        return Xout, Eout, y_out, vel


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, input_dims: utils.PlaceHolder, n_layers: int, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: utils.PlaceHolder, conditioning_fusion: str = "concat",
                 use_pocket_interaction: bool = True):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims.X
        self.out_dim_E = output_dims.E
        self.out_dim_y = output_dims.y
        self.out_dim_charges = output_dims.charges
        self.input_dims = input_dims
        self.outdim = output_dims
        self.use_pocket_interaction = use_pocket_interaction
        self.conditioning_fusion = ConditioningFusion(
            cond_dim=input_dims.pharma_feat,
            fusion_type=conditioning_fusion,
        )
        self.pocket_interaction = PocketInteractionBlock(
            ligand_dim=hidden_dims['dx'],
            pocket_feat_dim=8,
            n_head=hidden_dims['n_head'],
        )

        act_fn_in = nn.ReLU()
        act_fn_out = nn.ReLU()

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims.X + input_dims.charges + input_dims.pharma_feat, 
                                                hidden_mlp_dims['X'] ), act_fn_in, 
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)
        
        self.mlp_in_pharma_atoms = nn.Sequential(nn.Linear(input_dims.X + input_dims.charges + input_dims.pharma_feat, hidden_mlp_dims['X']), act_fn_in, 
                                       nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)
        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims.E, hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims.y, hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)
        self.mlp_in_pos = PositionsMLP(hidden_mlp_dims['pos'])

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'],
                                                            last_layer=False)     # needed to load old checkpoints
                                                            # last_layer=(i == n_layers - 1))
                                        for i in range(n_layers)])
        
        

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims.X + output_dims.charges))
        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims.E))
        # self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
        #                                nn.Linear(hidden_mlp_dims['y'], output_dims.y))
        self.mlp_out_pos = PositionsMLP(hidden_mlp_dims['pos'])

    def forward(self, data: utils.PlaceHolder, samples = None):

        bs, n = data.X.shape[0], data.X.shape[1]
        node_mask = data.node_mask
        p_mask = data.pharma_mask.unsqueeze(-1)

    
        diag_mask = ~torch.eye(n, device=data.X.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)
        
        
        #data.pharma_feat = data.pharma_feat * data.pharma_mask.unsqueeze(-1)
        
        X = torch.cat((data.X, data.charges), dim= -1)
        
        
        
        #X_masked = torch.where(data.pharma_mask.unsqueeze(-1) > 0, data.pharma_atom, X)
        
        fused_pharma_feat = self.conditioning_fusion(
            data.pharma_feat,
            pocket_feat=data.pocket_feat,
            pocket_mask=data.pocket_mask,
        )

        X_all = torch.cat((X, fused_pharma_feat), dim = -1)
        
        
        
        feats_x = torch.cat((data.pharma_atom, data.pharma_charge, fused_pharma_feat), dim = -1)
        feats_atoms = self.mlp_in_pharma_atoms(feats_x) * p_mask
        


        X_to_out = X[..., :self.out_dim_X + self.out_dim_charges]
        E_to_out = data.E[..., :self.out_dim_E]
        y_to_out = data.y[..., :self.out_dim_y]
        
        pharma_pos = data.pharma_atom_pos
        
        pharma_pos = pharma_pos * data.pharma_mask.unsqueeze(-1)
        
        pos = self.mlp_in_pos(data.pos, node_mask)
        
        current_pos = torch.mean((pos * p_mask) , dim=1, keepdim=True)
        target_pos = torch.mean((pharma_pos * p_mask) , dim=1, keepdim=True)
        
        pos = pos - current_pos + target_pos
        

        pos_masked = torch.where(data.pharma_mask.unsqueeze(-1) > 0, pharma_pos, pos)    

        
        
        # tolerance = 1e-6
        square_p_mask = p_mask.unsqueeze(2) * p_mask.unsqueeze(1)
        
        
        # coordinates = data.pharma_coord
        # distances = torch.cdist(coordinates, coordinates)
        # equivalence_matrix = (distances < tolerance).unsqueeze(-1) * square_p_mask * diag_mask
        
        E_masked = data.E * ~square_p_mask + data.pharma_E * square_p_mask
        
        
        new_E = self.mlp_in_E(E_masked)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        
        new_X = self.mlp_in_X(X_all)
        if self.use_pocket_interaction:
            new_X = self.pocket_interaction(
                ligand_h=new_X,
                pocket_pos=data.pocket_pos,
                pocket_feat=data.pocket_feat,
                pocket_mask=data.pocket_mask,
            )
        
        X_masked = torch.where(data.pharma_mask.unsqueeze(-1) > 0, feats_atoms, new_X) 
        
        features = utils.PlaceHolder(X=X_masked, E=new_E, y=self.mlp_in_y(data.y), charges=None,
                                     pos=pos_masked, node_mask=node_mask,
                                     pharma_coord=data.pharma_coord, pharma_feat=data.pharma_feat,
                                     pharma_mask=data.pharma_mask, pharma_atom=feats_atoms, 
                                     pharma_atom_pos=pharma_pos).mask()    
        

        for layer in self.tf_layers:
            features = layer(features)
            
        #X_to_decode = features.X + features.pharma_atom
        X = self.mlp_out_X(features.X)
        E = self.mlp_out_E(features.E)
        # y = self.mlp_out_y(features.y)
        
        #pos = features.pos + pharma_pos
        pos = self.mlp_out_pos(features.pos, node_mask)


        X = (X + X_to_out) 
        E = (E + E_to_out) * diag_mask
        # y = y + y_to_out
        y = y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))
        final_X = X[:, :, :self.out_dim_X]
        charges = X[:, : ,self.out_dim_X:self.out_dim_X+self.out_dim_charges]
        out = utils.PlaceHolder(pos=pos, X=final_X, charges=charges, E=E, y=y, node_mask=node_mask, 
                                pharma_coord=data.pharma_coord, pharma_feat=data.pharma_feat, 
                                pharma_mask=data.pharma_mask, pharma_atom=data.pharma_atom, 
                                pharma_atom_pos=data.pharma_atom_pos, pharma_E = data.pharma_E, 
                                pharma_charge= data.pharma_charge,
                                pocket_pos=data.pocket_pos,
                                pocket_feat=data.pocket_feat,
                                pocket_mask=data.pocket_mask).mask()
            
            
            
        if samples is not None:
            outputs = []
            outputs.append(out)
            for i in range(samples):
                noise_std = 0.03
                noise_x = torch.randn_like(features.X) * noise_std * ~p_mask           
                noised_x = (features.X + noise_x) * node_mask.unsqueeze(-1)
                
                
                noise_E = torch.randn_like(features.E) * noise_std * ~square_p_mask
                noised_E = (features.E + noise_E) * diag_mask
                
                noise_pos = torch.randn_like(features.pos) * noise_std * ~p_mask
                noised_pos = (features.pos + noise_pos) * node_mask.unsqueeze(-1)
                
                X = self.mlp_out_X(noised_x)
                E = self.mlp_out_E(noised_E)
                pos = self.mlp_out_pos(noised_pos, node_mask)
                
                
                X = (X + X_to_out) 
                E = (E + E_to_out) * diag_mask

                y = y_to_out

                E = 1/2 * (E + torch.transpose(E, 1, 2))
                final_X = X[:, :, :self.out_dim_X]
                charges = X[:, : ,self.out_dim_X:self.out_dim_X+self.out_dim_charges]
                out = utils.PlaceHolder(pos=pos, X=final_X, charges=charges, E=E, y=y, node_mask=node_mask, 
                                        pharma_coord=data.pharma_coord, pharma_feat=data.pharma_feat, 
                                        pharma_mask=data.pharma_mask, pharma_atom=data.pharma_atom, 
                                        pharma_atom_pos=data.pharma_atom_pos, pharma_E = data.pharma_E, 
                                        pharma_charge=data.pharma_charge,
                                        pocket_pos=data.pocket_pos,
                                        pocket_feat=data.pocket_feat,
                                        pocket_mask=data.pocket_mask).mask()
                outputs.append(out)
            return outputs
        
        else:

            return out
