import torch
import torch.nn as nn
from torchmetrics import MeanMetric, MinMetric
import numpy as np


def compute_pocket_clashes(ligand_pos: torch.Tensor, pocket_pos: torch.Tensor,
                           severe_thresh: float = 1.5, mild_thresh: float = 2.0):
    """
    Computes steric clashes between generated ligand atoms and pocket heavy atoms.
    ligand_pos: (N_lig, 3) tensor
    pocket_pos: (N_pocket, 3) tensor
    Returns:
        dict containing:
            - min_distance: minimum pairwise distance (float)
            - severe_clash_count: number of ligand atoms with distance < severe_thresh
            - mild_clash_count: number of ligand atoms with distance < mild_thresh
            - has_severe_clash: bool (severe_clash_count > 0)
            - has_mild_clash: bool (mild_clash_count > 0)
            - contact_ratio: fraction of ligand atoms in interaction shell [mild_thresh, 4.0]
    """
    if ligand_pos.numel() == 0 or pocket_pos.numel() == 0:
        return {
            'min_distance': 10.0,
            'severe_clash_count': 0,
            'mild_clash_count': 0,
            'has_severe_clash': False,
            'has_mild_clash': False,
            'contact_ratio': 0.0
        }

    lig_pos = ligand_pos.float()
    pkt_pos = pocket_pos.float()

    # Pairwise distances: (N_lig, N_pkt)
    dists = torch.cdist(lig_pos, pkt_pos)

    min_dist_per_atom, _ = dists.min(dim=1)
    min_dist = min_dist_per_atom.min().item()

    severe_clashes = (min_dist_per_atom < severe_thresh).sum().item()
    mild_clashes = (min_dist_per_atom < mild_thresh).sum().item()

    contact_atoms = ((min_dist_per_atom >= mild_thresh) & (min_dist_per_atom <= 4.0)).sum().item()
    contact_ratio = contact_atoms / max(1, lig_pos.size(0))

    return {
        'min_distance': min_dist,
        'severe_clash_count': severe_clashes,
        'mild_clash_count': mild_clashes,
        'has_severe_clash': severe_clashes > 0,
        'has_mild_clash': mild_clashes > 0,
        'contact_ratio': contact_ratio
    }


# AutoDock Vina scoring constants and parameters (Trott & Olson, J. Comput. Chem. 2010)
# Van der Waals radii (Angstroms) for standard atom types
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
    # Default fallback
    'default': 1.70
}

# Hydrogen bonding donor / acceptor identification
HBOND_ACCEPTORS = {'O', 'N', 'F'}
HBOND_DONORS = {'N', 'O'}
HYDROPHOBIC_ATOMS = {'C', 'F', 'Cl', 'Br', 'I', 'S'}


def compute_empirical_vina_score(ligand_pos: torch.Tensor,
                                  ligand_atom_types: list,
                                  pocket_pos: torch.Tensor,
                                  pocket_atom_types: list = None,
                                  num_rotatable_bonds: int = 0) -> float:
    """
    Computes an empirical AutoDock Vina intermolecular interaction score (kcal/mol)
    between ligand and receptor pocket.
    
    Terms:
      1. Gauss 1: w_1 * exp(-(d / 0.5)^2)
      2. Gauss 2: w_2 * exp(-((d - 3.0) / 2.0)^2)
      3. Repulsion: w_3 * d^2 if d < 0 else 0
      4. Hydrophobic: w_4 * p(d) if both hydrophobic
      5. Hydrogen bond: w_5 * p(d) if donor-acceptor pair
    where d = r_ij - (R_i + R_j) is the surface distance.
    Normalized by 1 + 0.05846 * num_rotatable_bonds.
    """
    if ligand_pos.numel() == 0 or pocket_pos.numel() == 0:
        return 0.0

    # Weights from AutoDock Vina
    w_gauss1 = -0.035579
    w_gauss2 = -0.005156
    w_repulsion = 0.840245
    w_hydrophobic = -0.035069
    w_hbond = -0.587439

    device = ligand_pos.device
    lig_pos = ligand_pos.float()
    pkt_pos = pocket_pos.float()

    n_lig = lig_pos.size(0)
    n_pkt = pkt_pos.size(0)

    # Radii tensors
    r_lig = torch.tensor([VDW_RADII.get(a, VDW_RADII['default']) for a in ligand_atom_types],
                         dtype=torch.float32, device=device).unsqueeze(1)  # (N_lig, 1)

    if pocket_atom_types is not None and len(pocket_atom_types) == n_pkt:
        r_pkt = torch.tensor([VDW_RADII.get(a, VDW_RADII['default']) for a in pocket_atom_types],
                             dtype=torch.float32, device=device).unsqueeze(0)  # (1, N_pkt)
    else:
        # Assume carbon / nitrogen / oxygen average radius 1.65 Å for pocket heavy atoms
        r_pkt = torch.full((1, n_pkt), 1.65, dtype=torch.float32, device=device)

    # Pairwise euclidean distance
    dist = torch.cdist(lig_pos, pkt_pos)  # (N_lig, N_pkt)

    # Surface distance: d = r_ij - (R_i + R_j)
    d = dist - (r_lig + r_pkt)

    # Cutoff: interaction terms only considered up to 8.0 Å
    mask_cutoff = dist <= 8.0

    # 1. Gauss 1
    term_gauss1 = torch.exp(-torch.clamp(d / 0.5, -10.0, 10.0) ** 2)

    # 2. Gauss 2
    term_gauss2 = torch.exp(-torch.clamp((d - 3.0) / 2.0, -10.0, 10.0) ** 2)

    # 3. Repulsion
    term_repulsion = torch.where(d < 0, d ** 2, torch.zeros_like(d))

    # 4. Hydrophobic term
    p_d = torch.clamp((d / -0.7), 0.0, 1.0)
    p_d = torch.where(d < -0.7, torch.ones_like(d), p_d)
    p_d = torch.where(d > 0.0, torch.zeros_like(d), p_d)

    is_lig_hydro = torch.tensor([a in HYDROPHOBIC_ATOMS for a in ligand_atom_types],
                                dtype=torch.bool, device=device).unsqueeze(1)
    if pocket_atom_types is not None and len(pocket_atom_types) == n_pkt:
        is_pkt_hydro = torch.tensor([a in HYDROPHOBIC_ATOMS for a in pocket_atom_types],
                                    dtype=torch.bool, device=device).unsqueeze(0)
    else:
        is_pkt_hydro = torch.ones((1, n_pkt), dtype=torch.bool, device=device)
    hydro_mask = is_lig_hydro & is_pkt_hydro
    term_hydrophobic = torch.where(hydro_mask, p_d, torch.zeros_like(d))

    # 5. Hydrogen bond term
    is_lig_donor = torch.tensor([a in HBOND_DONORS for a in ligand_atom_types],
                                dtype=torch.bool, device=device).unsqueeze(1)
    is_lig_acc = torch.tensor([a in HBOND_ACCEPTORS for a in ligand_atom_types],
                              dtype=torch.bool, device=device).unsqueeze(1)

    if pocket_atom_types is not None and len(pocket_atom_types) == n_pkt:
        is_pkt_donor = torch.tensor([a in HBOND_DONORS for a in pocket_atom_types],
                                    dtype=torch.bool, device=device).unsqueeze(0)
        is_pkt_acc = torch.tensor([a in HBOND_ACCEPTORS for a in pocket_atom_types],
                                  dtype=torch.bool, device=device).unsqueeze(0)
    else:
        # Assume ~30% of pocket heavy atoms can participate in H-bonds
        is_pkt_donor = torch.ones((1, n_pkt), dtype=torch.bool, device=device)
        is_pkt_acc = torch.ones((1, n_pkt), dtype=torch.bool, device=device)

    hbond_mask = (is_lig_donor & is_pkt_acc) | (is_lig_acc & is_pkt_donor)
    term_hbond = torch.where(hbond_mask, p_d, torch.zeros_like(d))

    # Total unweighted sum per pair
    total_pair = (w_gauss1 * term_gauss1 +
                  w_gauss2 * term_gauss2 +
                  w_repulsion * term_repulsion +
                  w_hydrophobic * term_hydrophobic +
                  w_hbond * term_hbond)

    total_pair = torch.where(mask_cutoff, total_pair, torch.zeros_like(d))
    score = total_pair.sum().item()

    # Normalization by flexibility
    n_rot = max(0, num_rotatable_bonds)
    score_norm = score / (1.0 + 0.05846 * n_rot)

    return float(score_norm)


class PocketMetrics(nn.Module):
    """
    TorchMetrics wrapper for multi-dimensional pocket-ligand evaluation:
      - Severe steric clashes (< 1.5 Å)
      - Mild steric clashes (< 2.0 Å)
      - Minimum interatomic distance
      - Favorable contact ratio ([2.0, 4.0] Å)
      - Empirical AutoDock Vina affinity (mean, top-1, top-10)
    """
    def __init__(self):
        super().__init__()
        self.severe_clash_rate = MeanMetric()
        self.mild_clash_rate = MeanMetric()
        self.min_dist_metric = MeanMetric()
        self.contact_ratio = MeanMetric()
        self.vina_mean = MeanMetric()
        self.vina_top1 = MinMetric()
        self.all_vina_scores = []

    def reset(self):
        self.severe_clash_rate.reset()
        self.mild_clash_rate.reset()
        self.min_dist_metric.reset()
        self.contact_ratio.reset()
        self.vina_mean.reset()
        self.vina_top1.reset()
        self.all_vina_scores = []

    def update(self, ligand_pos: torch.Tensor, ligand_atom_types: list,
               pocket_pos: torch.Tensor, pocket_atom_types: list = None,
               num_rotatable_bonds: int = 0):
        clashes = compute_pocket_clashes(ligand_pos, pocket_pos)
        self.severe_clash_rate.update(1.0 if clashes['has_severe_clash'] else 0.0)
        self.mild_clash_rate.update(1.0 if clashes['has_mild_clash'] else 0.0)
        self.min_dist_metric.update(clashes['min_distance'])
        self.contact_ratio.update(clashes['contact_ratio'])

        vina_score = compute_empirical_vina_score(
            ligand_pos, ligand_atom_types, pocket_pos, pocket_atom_types, num_rotatable_bonds
        )
        self.vina_mean.update(vina_score)
        self.vina_top1.update(vina_score)
        self.all_vina_scores.append(vina_score)

    def compute(self) -> dict:
        def safe_compute_m(metric, default=0.0):
            if getattr(metric, "_update_called", True) is False:
                return default
            if hasattr(metric, "total") and metric.total == 0:
                return default
            try:
                res = metric.compute()
                return default if torch.isnan(res) else res.item()
            except Exception:
                return default

        top10_score = 0.0
        if len(self.all_vina_scores) > 0:
            sorted_scores = sorted(self.all_vina_scores)
            top10 = sorted_scores[:min(10, len(sorted_scores))]
            top10_score = float(np.mean(top10))

        return {
            'pocket/severe_clash_rate': safe_compute_m(self.severe_clash_rate),
            'pocket/mild_clash_rate': safe_compute_m(self.mild_clash_rate),
            'pocket/min_distance_mean': safe_compute_m(self.min_dist_metric),
            'pocket/contact_ratio_mean': safe_compute_m(self.contact_ratio),
            'pocket/vina_score_mean': safe_compute_m(self.vina_mean),
            'pocket/vina_score_top1': safe_compute_m(self.vina_top1),
            'pocket/vina_score_top10': top10_score
        }
