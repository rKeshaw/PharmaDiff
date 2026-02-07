from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from hydra import compose, initialize_config_dir
from torch.utils.data import WeightedRandomSampler

from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset
from pharmadiff.diffusion.noise_model import NoiseModel
from pharmadiff.models.affinity_predictor import TimeAwareAffinityPredictor

def train_affinity_predictor():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PlinderGraphDataset(split='train')

    affinities = dataset.index['affinity_data.pKd'].values
    weights = torch.tensor([5.0 if a > 7.0 else 1.0 for a in affinities], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler, collate_fn=dataset.collate)
    
    # 2. Models
    model = TimeAwareAffinityPredictor().to(device)
    # We need the noise scheduler to generate training samples
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=["dataset=plinder"])
    noise_scheduler = NoiseModel(cfg)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    # 3. Training Loop
    model.train()
    for epoch in range(100):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            
            # A. Prepare Data
            lig_pos_clean = batch['ligand'].pos.to(device)
            lig_feat = batch['ligand'].x.to(device)
            prot_pos = batch['pocket_pos'].to(device)
            prot_feat = batch['pocket_feat'].to(device)
            true_affinity = batch['affinity'].to(device)
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
            pred_affinity = model(
                lig_pos_noisy,
                lig_feat_one_hot,
                prot_pos,
                prot_feat,
                t,
                lig_batch=lig_batch_idx,
                prot_batch=prot_batch_idx,
            )
            
            # E. Loss
            # We want the model to predict the TRUE affinity even from noisy structures
            # This teaches it the "gradient direction" towards the clean, high-affinity state
            loss = loss_fn(pred_affinity, true_affinity)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch}: Loss {total_loss / len(loader)}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), "affinity_compass.pt")

if __name__ == "__main__":
    train_affinity_predictor()
