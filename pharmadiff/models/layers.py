import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class PositionsMLP(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)           # bs, n, 1
        new_norm = self.mlp(norm)                              # bs, n, 1
        new_pos = pos * new_norm / (norm + self.eps)
        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        return new_pos


class SE3Norm(nn.Module):
    def __init__(self, eps: float = 1e-5, device=None, dtype=None) -> None:
        """ Note: There is a relatively similar layer implemented by NVIDIA:
            https://catalog.ngc.nvidia.com/orgs/nvidia/resources/se3transformer_for_pytorch.
            It computes a ReLU on a mean-zero normalized norm, which I find surprising.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.normalized_shape = (1,)                   # type: ignore[arg-type]
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, pos, node_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)           # bs, n, 1
        mean_norm = torch.sum(norm, dim=1, keepdim=True) / torch.sum(node_mask, dim=1, keepdim=True)      # bs, 1, 1
        new_pos = self.weight * pos / (mean_norm + self.eps)
        return new_pos

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}'.format(**self.__dict__)


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, x_mask):
        """ X: bs, n, dx. """
        x_mask = x_mask.expand(-1, -1, X.shape[-1])
        float_imask = 1 - x_mask.float()
        m = X.sum(dim=1) / torch.sum(x_mask, dim=1)
        mi = (X + 1e5 * float_imask).min(dim=1)[0]
        ma = (X - 1e5 * float_imask).max(dim=1)[0]
        std = torch.sum(((X - m[:, None, :]) ** 2) * x_mask, dim=1) / torch.sum(x_mask, dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E, e_mask1, e_mask2):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()
        divide = torch.sum(mask, dim=(1, 2))
        m = E.sum(dim=(1, 2)) / divide
        mi = (E + 1e5 * float_imask).min(dim=2)[0].min(dim=1)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0].max(dim=1)[0]
        std = torch.sum(((E - m[:, None, None, :]) ** 2) * mask, dim=(1, 2)) / divide
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out
    
    
class Etof(nn.Module):
    def __init__(self, d, dx):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dx)

    def forward(self, E, e_mask1, e_mask2):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
             # Create the combined mask and its inverse
        mask = (e_mask1 * e_mask2).expand(-1, -1, -1, E.shape[-1])
        float_imask = 1 - mask.float()

        # Calculate the divisor (sum of the mask along the third dimension)
        divide = torch.sum(mask, dim=2, keepdim=True)

        # Mean of masked elements along the third dimension
        m = torch.sum(E * mask, dim=2) / divide

        # Minimum of masked elements along the third dimension
        mi = (E + 1e5 * float_imask).min(dim=2)[0]

        # Maximum of masked elements along the third dimension
        ma = (E - 1e5 * float_imask).max(dim=2)[0]

        # Standard deviation of masked elements along the third dimension
        std = torch.sum(((E - m.unsqueeze(2)) ** 2) * mask, dim=2) / divide

        # Concatenate the statistics along the feature dimension
        z = torch.cat((m, mi, ma, std), dim=-1)  # Shape: (batch_size, n, 4*de)

        # Apply a linear transformation to reduce back to shape (batch_size, n, de)
        out = self.lin(z)  # Assuming self.lin reduces the feature dimension back to de

        return out
    
    



class GatedGCNLayer(MessagePassing):
    def __init__(self, input_dim, output_dim, dropout, batch_norm, residual=False):
        super(GatedGCNLayer, self).__init__(aggr='add')  # Aggregation method is 'add'
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        self.lin_A = nn.Linear(input_dim, output_dim)
        self.lin_B = nn.Linear(input_dim, output_dim)
        self.lin_C = nn.Linear(input_dim, output_dim)
        self.lin_D = nn.Linear(input_dim, output_dim)
        self.lin_E = nn.Linear(input_dim, output_dim)
        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def forward(self, x, edge_index, edge_attr, h, e):
        h_in = h  # for residual connection
        e_in = e  # for residual connection
       
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Node features
        x_A = self.lin_A(x)
        x_B = self.lin_B(x)
        x_D = self.lin_D(x)
        x_E = self.lin_E(x)

        # Edge features
        edge_attr = self.lin_C(edge_attr)

        # Message passing
        out = self.propagate(edge_index, x=x, x_A=x_A, x_B=x_B, x_D=x_D, x_E=x_E, edge_attr=edge_attr)
        
        if self.batch_norm:
            out = self.bn_node_h(out)

        out = F.relu(out)
        out = F.dropout(out, self.dropout, training=self.training)

        if self.residual:
            out = x + out
        
        return out

    def message(self, x_j, x_i, x_A, x_B, x_D, x_E, edge_attr):
        # Compute edge feature
        edge_feature = x_D + x_E
        edge_feature = edge_feature + edge_attr
        sigma = torch.sigmoid(edge_feature)

        # Message computation
        return x_B * sigma

    def aggregate(self, inputs, index):
        # Aggregation (default is sum)
        return torch.sum(inputs, dim=0)

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__,
                                                            self.in_channels,
                                                            self.out_channels)



class EtoX(nn.Module):
    def __init__(self, de, dx):
        super().__init__()
        self.lin = nn.Linear(4 * de, dx)

    def forward(self, E, e_mask2):
        """ E: bs, n, n, de"""
        bs, n, _, de = E.shape
        e_mask2 = e_mask2.expand(-1, n, -1, de)
        float_imask = 1 - e_mask2.float()
        m = E.sum(dim=2) / torch.sum(e_mask2, dim=2)
        mi = (E + 1e5 * float_imask).min(dim=2)[0]
        ma = (E - 1e5 * float_imask).max(dim=2)[0]
        std = torch.sum(((E - m[:, :, None, :]) ** 2) * e_mask2, dim=2) / torch.sum(e_mask2, dim=2)
        z = torch.cat((m, mi, ma, std), dim=2)
        out = self.lin(z)
        return out


def masked_softmax(x, mask, **kwargs):
    if torch.sum(mask) == 0:
        return torch.zeros_like(x)
    x_masked = x.clone()
    x_masked[mask == 0] = -1e9
    out = torch.softmax(x_masked, **kwargs)
    out = out * mask
    return torch.nan_to_num(out, nan=0.0)



class SetNorm(nn.LayerNorm):
    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x, x_mask):
        bs, n, d = x.shape
        divide = torch.sum(x_mask, dim=1, keepdim=True) * d      # bs
        means = torch.sum(x * x_mask, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((x - means) ** 2 * x_mask, dim=[1, 2], keepdim=True) / (divide + self.eps)
        out = (x - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * x_mask
        return out


class GraphNorm(nn.LayerNorm):
    def __init__(self, feature_dim=None, **kwargs):
        super().__init__(normalized_shape=feature_dim, **kwargs)
        self.weights = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        self.biases = nn.Parameter(torch.empty(1, 1, 1, feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, E, emask1, emask2):
        bs, n, _, d = E.shape
        divide = torch.sum(emask1 * emask2, dim=[1, 2], keepdim=True) * d      # bs
        means = torch.sum(E * emask1 * emask2, dim=[1, 2], keepdim=True) / divide
        var = torch.sum((E - means) ** 2 * emask1 * emask2, dim=[1, 2], keepdim=True) / (divide + self.eps)
        out = (E - means) / (torch.sqrt(var) + self.eps)
        out = out * self.weights + self.biases
        out = out * emask1 * emask2
        return out
    

class PositionAttention(nn.Module):
    def __init__(self, hidden_size):
        super(PositionAttention, self).__init__()
        self.hidden_size = hidden_size

        # Linear layers to compute the attention scores
        self.position_score = nn.Linear(3, hidden_size)
        self.distance_score = nn.Linear(1, hidden_size)
        self.combined_score = nn.Linear(2 * hidden_size, 1)

    def forward(self, hidden, node_positions, pairwise_distances):
        """
        Args:
            hidden (torch.Tensor): current decoder hidden state, shape (batch_size, 1, hidden_size)
            node_positions (torch.Tensor): positions of the nodes, shape (batch_size, num_nodes, 3)
            pairwise_distances (torch.Tensor): conditional pairwise distances between nodes, shape (batch_size, num_nodes, num_nodes)
        Returns:
            context (torch.Tensor): weighted sum of node_positions, shape (batch_size, 1, 3)
            attention_weights (torch.Tensor): attention weights, shape (batch_size, 1, num_nodes)
        """
        batch_size, num_nodes, _ = node_positions.shape

        # Compute the position-based attention scores
        position_scores = self.position_score(node_positions)  # (batch_size, num_nodes, hidden_size)
        position_scores = torch.bmm(hidden, position_scores.transpose(1, 2))  # (batch_size, 1, num_nodes)

        # Compute the distance-based attention scores
        distance_scores = self.distance_score(pairwise_distances.unsqueeze(-1))  # (batch_size, num_nodes, num_nodes, hidden_size)
        distance_scores = torch.bmm(hidden.transpose(1, 2), distance_scores.reshape(batch_size, num_nodes, -1))  # (batch_size, 1, num_nodes)

        # Combine the position-based and distance-based scores
        combined_scores = torch.cat([position_scores, distance_scores], dim=2)  # (batch_size, 1, 2 * hidden_size)
        attention_weights = F.softmax(self.combined_score(combined_scores), dim=2)  # (batch_size, 1, num_nodes)

        # Compute the context vector
        context = torch.bmm(attention_weights, node_positions)  # (batch_size, 1, 3)

        return context, attention_weights


class PositionsAttentionMLP(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.lin_1 = nn.Linear(1, hidden_dim)
        self.lin_2 = nn.Sequential(nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.att = PositionAttention(hidden_dim)

    def forward(self, pos, node_mask, p, pharma_mask):
        norm = torch.norm(pos, dim=-1, keepdim=True)           # bs, n, 1
        new_norm = self.lin_1(norm)                              # bs, n, 1
        
        diag_mask_p = ~torch.eye(pharma_mask.size(1), device=pharma_mask.device, dtype=torch.bool).unsqueeze(0).repeat(pharma_mask.size(0), 1, 1)
        edge_mask_p = diag_mask_p & pharma_mask.unsqueeze(-1) & pharma_mask.unsqueeze(-2)
        
        
        pairwise_distances = torch.cdist(p, p).unsqueeze(-1).float() 
        pairwise_distance = pairwise_distance * edge_mask_p.unsqueeze(-1) 
        
        
        context, attention_weights = self.att(new_norm, pos, pairwise_distances)
    
        #position_input = torch.cat([context.repeat(1, pos.size(1), 1), new_norm], dim=2)
        
        new_norm = self.lin_2(new_norm)
        
        new_pos = pos * new_norm * context / (norm + self.eps)
        
        new_pos = new_pos * node_mask.unsqueeze(-1)
        new_pos = new_pos - torch.mean(new_pos, dim=1, keepdim=True)
        return new_pos
