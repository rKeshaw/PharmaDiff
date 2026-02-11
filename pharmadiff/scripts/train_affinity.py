import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from hydra import compose, initialize_config_dir
from torch.utils.data import WeightedRandomSampler
from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset
from pharmadiff.datasets.plinder_precomputed_dataset import PlinderPrecomputedDataset
from pharmadiff.diffusion.noise_model import NoiseModel
from pharmadiff.models.affinity_predictor import TimeAwareAffinityPredictor

def train_affinity_predictor(
    epochs: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_precomputed: bool,
    precomputed_root: str,
):
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if use_precomputed:
        dataset = PlinderPrecomputedDataset(split="train", root=precomputed_root)
    else:
        dataset = PlinderGraphDataset(split='train')

    # affinities = dataset.index['affinity_data.pKd'].values
    if use_precomputed:
        sampler = None
    else:
        affinity_col = getattr(dataset, 'affinity_column', None)
        if affinity_col is not None:
            affinities = dataset.index[affinity_col].values
            weights = torch.tensor([5.0 if a > 7.0 else 1.0 for a in affinities], dtype=torch.float)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        else:
            print('Warning: affinity column missing; disabling affinity-based sampler weights.')
            sampler = None
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=dataset.collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    # 2. Models
    model = TimeAwareAffinityPredictor().to(device)
    # We need the noise scheduler to generate training samples
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=["dataset=plinder"])
    noise_scheduler = NoiseModel(cfg)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    affinity_loss_fn = torch.nn.MSELoss()
    ifp_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            if batch is None:
                print("Warning: Received None batch, skipping...")
                continue
            optimizer.zero_grad()
            
            # A. Prepare Data
            lig_pos_clean = batch['ligand'].pos.to(device)
            lig_feat = batch['ligand'].x.to(device)
            prot_pos = batch['pocket_pos'].to(device)
            prot_feat = batch['pocket_feat'].to(device)
            true_affinity = batch['affinity'].to(device).view(-1, 1)
            lig_batch_idx = batch['ligand'].batch.to(device)
            prot_batch_idx = batch['pocket_batch'].to(device)
            
            # B. Sample Timesteps (t)
            batch_size = batch['ligand'].num_graphs
            t = torch.randint(0, noise_scheduler.T, (batch_size,), device=device).long()
            
            # C. Add Noise (The Critical Step)
            # We corrupt the clean ligand positions to simulate diffusion steps
            noise = torch.randn_like(lig_pos_clean)
            alpha_bars = noise_scheduler.get_alpha_bar(t_int=t, key='p')
            sqrt_alpha = torch.sqrt(alpha_bars).clamp(min=1e-6)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bars).clamp(min=1e-6)
            node_alpha = sqrt_alpha[lig_batch_idx].unsqueeze(-1)
            node_sigma = sqrt_one_minus_alpha[lig_batch_idx].unsqueeze(-1)
            lig_pos_noisy = node_alpha * lig_pos_clean + node_sigma * noise

            num_types = model.lig_emb.in_features
            lig_feat_one_hot = F.one_hot(lig_feat, num_classes=num_types).float()
            
            # D. Predict Affinity from NOISY structure
            pred_out = model(
                lig_pos_noisy,
                lig_feat_one_hot,
                prot_pos,
                prot_feat,
                t,
                lig_batch=lig_batch_idx,
                prot_batch=prot_batch_idx,
            )
            pred_affinity = pred_out["affinity"]
            pred_ifp_logits = pred_out["ifp_logits"]
            
            # E. Loss
            # We want the model to predict the TRUE affinity even from noisy structures
            # This teaches it the "gradient direction" towards the clean, high-affinity state
            loss_affinity = affinity_loss_fn(pred_affinity, true_affinity)

            if "plip_labels" in batch and "plip_label_mask" in batch:
                true_ifp = batch["plip_labels"].to(device)
                ifp_mask = batch["plip_label_mask"].to(device)
                raw_ifp_loss = ifp_loss_fn(pred_ifp_logits, true_ifp)
                masked_ifp_loss = raw_ifp_loss * ifp_mask
                valid = ifp_mask.sum()
                loss_ifp = masked_ifp_loss.sum() / valid.clamp(min=1.0)
            else:
                loss_ifp = pred_ifp_logits.new_tensor(0.0)

            loss = loss_affinity + 0.5 * loss_ifp
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss / len(loader)}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), "affinity_compass.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train affinity predictor on PLINDER.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--use-precomputed", action="store_true")
    parser.add_argument("--precomputed-root", default="data/plinder_precomputed")
    args = parser.parse_args()
    train_affinity_predictor(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        use_precomputed=args.use_precomputed,
        precomputed_root=args.precomputed_root,
    )
