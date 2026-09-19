"""
pharmadiff/metrics/micrometrics.py

Comprehensive micrometrics module for PharmaDiff and Pocket-PharmaDiff.
Computes fine-grained internal behavioral diagnostics across:
  1. Timestep strata (low-t, mid-t, high-t performance)
  2. Chemical topology & bond recall (atom accuracy, bond recall on existing bonds, valency deviation)
  3. 3D geometry & spatial alignment (pharmacophore anchor drift, internal steric clashes, bond length MAE, Rg error)
  4. Layer-wise transformer dynamics (embedding norms, velocity magnitudes, cross-attention entropy)
  5. Protein pocket interaction (when use_pocket=True: pocket-ligand clash rates, steric push norm, min distance)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


def compute_micrometrics(
    masked_pred,
    masked_true,
    z_t,
    layer_diagnostics=None,
    use_pocket: bool = False,
    T: int = 500
):
    """
    Computes fine-grained micrometrics on a single training or validation batch.
    All calculations are vectorized on GPU with no gradient tracking.
    """
    with torch.no_grad():
        metrics = {}
        node_mask = masked_true.node_mask                    # bs, n
        pharma_mask = masked_true.pharma_mask                # bs, n
        bs, n = node_mask.shape
        gen_mask = node_mask & (~pharma_mask)                # Generating (non-pharma) atoms

        num_nodes = node_mask.sum().item()
        if num_nodes == 0:
            return metrics

        # -----------------------------------------------------------------
        # 1. Chemical Topology & Discrete Feature Micrometrics
        # -----------------------------------------------------------------
        pred_X_classes = torch.argmax(masked_pred.X, dim=-1)     # bs, n
        true_X_classes = torch.argmax(masked_true.X, dim=-1)     # bs, n

        # Atom accuracy across all nodes
        correct_X = (pred_X_classes == true_X_classes) & node_mask
        metrics["chem/atom_acc_all"] = correct_X.sum().float() / (num_nodes + 1e-6)

        # Atom accuracy on generating (designed) atoms vs pharmacophore anchors
        num_gen = gen_mask.sum().item()
        if num_gen > 0:
            correct_gen = (pred_X_classes == true_X_classes) & gen_mask
            metrics["chem/atom_acc_generated"] = correct_gen.sum().float() / num_gen

        num_pharma = pharma_mask.sum().item()
        if num_pharma > 0:
            correct_pharma = (pred_X_classes == true_X_classes) & pharma_mask
            metrics["chem/atom_acc_pharma"] = correct_pharma.sum().float() / num_pharma

        # Bond Topology Micrometrics
        # Edge mask: non-diagonal pairs where both atoms exist
        diag_mask = ~torch.eye(n, device=node_mask.device, dtype=torch.bool).unsqueeze(0).expand(bs, -1, -1)
        pair_mask = diag_mask & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        # Upper triangular to count each undirected bond once
        triu_mask = torch.triu(torch.ones(n, n, device=node_mask.device, dtype=torch.bool), diagonal=1).unsqueeze(0).expand(bs, -1, -1)
        bond_eval_mask = pair_mask & triu_mask

        pred_E_classes = torch.argmax(masked_pred.E, dim=-1)     # bs, n, n
        true_E_classes = torch.argmax(masked_true.E, dim=-1)     # bs, n, n

        existing_bonds = bond_eval_mask & (true_E_classes > 0)
        non_bonds = bond_eval_mask & (true_E_classes == 0)
        num_existing = existing_bonds.sum().item()
        num_non_bonds = non_bonds.sum().item()

        if num_existing > 0:
            # Did the model predict ANY bond where one exists?
            detected_bonds = (pred_E_classes > 0) & existing_bonds
            metrics["chem/bond_recall_existing"] = detected_bonds.sum().float() / num_existing

            # Did the model predict the EXACT bond type?
            exact_bonds = (pred_E_classes == true_E_classes) & existing_bonds
            metrics["chem/bond_type_acc_existing"] = exact_bonds.sum().float() / num_existing

            # Bond precision
            predicted_bonds = bond_eval_mask & (pred_E_classes > 0)
            num_pred_bonds = predicted_bonds.sum().item()
            if num_pred_bonds > 0:
                true_pos_bonds = (true_E_classes > 0) & predicted_bonds
                metrics["chem/bond_precision_existing"] = true_pos_bonds.sum().float() / num_pred_bonds

        if num_non_bonds > 0:
            correct_non_bonds = (pred_E_classes == 0) & non_bonds
            metrics["chem/bond_empty_acc"] = correct_non_bonds.sum().float() / num_non_bonds

        # -----------------------------------------------------------------
        # 2. 3D Spatial & Geometric Consistency Micrometrics
        # -----------------------------------------------------------------
        pred_pos = masked_pred.pos                           # bs, n, 3
        true_pos = masked_true.pos                           # bs, n, 3

        # Pharmacophore anchor drift (Euclidean distance between predicted and true anchors)
        if num_pharma > 0:
            pharma_drift = torch.norm(pred_pos - true_pos, dim=-1)  # bs, n
            drift_vals = pharma_drift[pharma_mask]
            metrics["geom/pharma_anchor_drift_mean"] = drift_vals.mean()
            metrics["geom/pharma_anchor_drift_max"] = drift_vals.max()

        # Displacement / velocity norm from noisy pos_t to pred_pos
        if hasattr(z_t, 'pos') and z_t.pos is not None:
            vel_norm = torch.norm(pred_pos - z_t.pos, dim=-1)
            metrics["geom/velocity_norm_mean"] = vel_norm[node_mask].mean()

        # Internal steric clashes (atoms within the ligand closer than 1.0 Å)
        dist_matrix = torch.cdist(pred_pos, pred_pos)        # bs, n, n
        clash_pairs = (dist_matrix < 1.0) & bond_eval_mask
        total_eval_pairs = bond_eval_mask.sum().item()
        if total_eval_pairs > 0:
            metrics["geom/internal_clash_rate"] = clash_pairs.sum().float() / total_eval_pairs

        # Bond length MAE on real existing bonds
        if num_existing > 0:
            pred_dists = dist_matrix[existing_bonds]
            true_dists = torch.cdist(true_pos, true_pos)[existing_bonds]
            metrics["geom/bond_length_mae"] = torch.abs(pred_dists - true_dists).mean()

        # Radius of gyration error
        pred_pos_masked = pred_pos * node_mask.unsqueeze(-1)
        true_pos_masked = true_pos * node_mask.unsqueeze(-1)
        pred_center = pred_pos_masked.sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        true_center = true_pos_masked.sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
        pred_rg = torch.sqrt((((pred_pos_masked - pred_center) * node_mask.unsqueeze(-1)) ** 2).sum(dim=(1, 2)) / node_mask.sum(dim=1).clamp(min=1))
        true_rg = torch.sqrt((((true_pos_masked - true_center) * node_mask.unsqueeze(-1)) ** 2).sum(dim=(1, 2)) / node_mask.sum(dim=1).clamp(min=1))
        metrics["geom/radius_gyration_err"] = torch.abs(pred_rg - true_rg).mean()

        # -----------------------------------------------------------------
        # 3. Timestep-Stratified Micrometrics (low-t, mid-t, high-t)
        # -----------------------------------------------------------------
        if hasattr(z_t, 't_int') and z_t.t_int is not None:
            t_vals = z_t.t_int.squeeze(-1)                   # bs
            t1 = T / 3.0
            t2 = 2.0 * T / 3.0

            low_t_mask = t_vals <= t1
            mid_t_mask = (t_vals > t1) & (t_vals <= t2)
            high_t_mask = t_vals > t2

            pos_diff_sq = ((pred_pos - true_pos) ** 2).sum(dim=-1) # bs, n

            for name, mask in [("low_t", low_t_mask), ("mid_t", mid_t_mask), ("high_t", high_t_mask)]:
                if mask.any():
                    node_submask = node_mask[mask]
                    if node_submask.any():
                        metrics[f"time_strata/pos_mse_{name}"] = pos_diff_sq[mask][node_submask].mean()
                        correct_sub = (pred_X_classes[mask] == true_X_classes[mask]) & node_submask
                        metrics[f"time_strata/atom_acc_{name}"] = correct_sub.sum().float() / (node_submask.sum().float() + 1e-6)

                    existing_sub = existing_bonds[mask]
                    if existing_sub.any():
                        rec_sub = (pred_E_classes[mask] > 0) & existing_sub
                        metrics[f"time_strata/bond_rec_{name}"] = rec_sub.sum().float() / (existing_sub.sum().float() + 1e-6)

        # -----------------------------------------------------------------
        # 4. Deep Network Layer Dynamics (from layer_diagnostics)
        # -----------------------------------------------------------------
        if layer_diagnostics is not None and isinstance(layer_diagnostics, dict):
            for k, v in layer_diagnostics.items():
                if isinstance(v, torch.Tensor):
                    metrics[f"layer/{k}"] = v.detach().mean()
                elif isinstance(v, (int, float)):
                    metrics[f"layer/{k}"] = torch.tensor(v, device=node_mask.device)

        # -----------------------------------------------------------------
        # 5. Pocket Interaction Micrometrics (when use_pocket=True)
        # -----------------------------------------------------------------
        if use_pocket and getattr(masked_true, 'pocket_pos', None) is not None:
            pocket_pos = masked_true.pocket_pos              # bs, M, 3
            pocket_mask = masked_true.pocket_mask            # bs, M

            # Only consider valid pocket nodes and generating ligand atoms
            dist_lp = torch.cdist(pred_pos, pocket_pos)      # bs, n, M
            # Mask out non-pocket and non-ligand pairs by setting to infinity
            pair_valid = gen_mask.unsqueeze(-1) & pocket_mask.unsqueeze(1)
            dist_lp_valid = torch.where(pair_valid, dist_lp, torch.tensor(float('inf'), device=dist_lp.device))

            if pair_valid.any():
                min_dist = dist_lp_valid.min().item()
                metrics["pocket/ligand_pocket_min_dist"] = torch.tensor(min_dist, device=dist_lp.device)

                # Distance from each generating atom to nearest pocket atom
                nearest_pocket_dist = dist_lp_valid.min(dim=-1)[0] # bs, n
                nearest_gen_dists = nearest_pocket_dist[gen_mask]
                valid_gen_dists = nearest_gen_dists[nearest_gen_dists < float('inf')]

                if valid_gen_dists.numel() > 0:
                    metrics["pocket/mean_dist_to_pocket"] = valid_gen_dists.mean()
                    metrics["pocket/clash_rate_sub_2A"] = (valid_gen_dists < 2.0).float().mean()
                    metrics["pocket/clash_rate_sub_1_5A"] = (valid_gen_dists < 1.5).float().mean()

        # Convert all to floats/items for clean logging
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                clean_metrics[k] = v.item() if v.numel() == 1 else v.mean().item()
            else:
                clean_metrics[k] = float(v)

        return clean_metrics


class MicroMetricsTracker:
    """
    Accumulates and averages micrometrics over an epoch or interval.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, metrics_dict):
        if not metrics_dict:
            return
        for k, v in metrics_dict.items():
            if not math.isnan(v) and not math.isinf(v):
                self.totals[k] += v
                self.counts[k] += 1

    def compute(self):
        return {k: self.totals[k] / max(self.counts[k], 1) for k in self.totals}
