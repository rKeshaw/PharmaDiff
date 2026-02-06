import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset
from pharmadiff.models.affinity_predictor import TimeAwareAffinityPredictor
from pharmadiff.diffusion.noise_model import NoiseModel # Existing PharmaDiff Code

def train_affinity_predictor():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = PlinderGraphDataset(split='train')

    affinities = dataset.index['affinity_data.pKd'].values
    weights = torch.tensor([5.0 if a > 7.0 else 1.0 for a in affinities], dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_sampler=len(weights), replacement=True)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=sampler, collate_fn=dataset.collate)
    
    # 2. Models
    model = TimeAwareAffinityPredictor().to(device)
    # We need the noise scheduler to generate training samples
    noise_scheduler = NoiseModel() # Load default schedule
    
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
            t = torch.randint(0, 1000, (batch_size,), device=device).long()
            
            # C. Add Noise (The Critical Step)
            # We corrupt the clean ligand positions to simulate diffusion steps
            noise = torch.randn_like(lig_pos_clean)
            alpha_bars = noise_scheduler.get_alphas_bars(t) # You'll need to expose this from NoiseModel
            
            # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
            sqrt_alpha = torch.sqrt(alpha_bars)
            sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bars)

            num_types = lig_feat.shape[-1]
            uniform_dist = torch.ones_like(lig_feat) / num_types
            
            # Broadcast scalar t factors to node dimensions
            # (assuming simple batching for brevity - use torch_geometric Batch in production)
            lig_pos_noisy = (sqrt_alpha * lig_feat) + (sqrt_one_minus_alpha * uniform_dist)
            
            # D. Predict Affinity from NOISY structure
            pred_affinity = model(lig_pos_noisy, lig_feat_noisy, prot_pos, prot_feat, t, lig_batch_idx, prot_batch_idx)
            
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
