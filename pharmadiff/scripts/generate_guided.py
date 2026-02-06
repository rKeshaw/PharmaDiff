import torch
import hydra
from pharmadiff.main import PharmaDiff  # The PL Lightning Module
from pharmadiff.models.affinity_predictor import TimeAwareAffinityPredictor
from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset

@hydra.main(config_path=".", config_name="config_guided")
def generate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load PharmaDiff (The Engine)
    # This model knows how to make valid molecules from pharmacophores
    model = PharmaDiff.load_from_checkpoint(cfg.checkpoint_path)
    model.to(device)
    model.eval()

    # 2. Load The Compass (Affinity Predictor)
    # This model knows where the high-affinity regions are
    affinity_model = TimeAwareAffinityPredictor()
    affinity_model.load_state_dict(torch.load("affinity_compass.pt"))
    affinity_model.to(device)
    affinity_model.eval()
    
    # Inject compass into engine
    model.affinity_model = affinity_model

    # 3. Load a Test Case (A Pharmacophore + Pocket)
    test_set = PlinderGraphDataset(split='test')
    sample_data = test_set[0] # Grab first test case
    
    # 4. Run Guided Generation
    # We create a "Condition" object to pass the pocket info
    class Condition:
        pocket_pos = sample_data['pocket_pos'].to(device)
        pocket_feat = sample_data['pocket_feat'].to(device)
    
    print("Generating molecule with Affinity Guidance...")
    
    # The modified sample_batch (from previous step) will now use the compass
    # guidance_scale=10.0 is a strong push towards high affinity
    molecules = model.sample_batch(
        batch_size=1, 
        sample_condition=Condition(), 
        guidance_scale=10.0 
    )
    
    print("Generation Complete. Saving SDF...")
    # Save molecules...

if __name__ == "__main__":
    generate()
