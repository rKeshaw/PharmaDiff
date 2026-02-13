import torch
import hydra
from pharmadiff.main import PharmaDiff  # The PL Lightning Module
from pharmadiff.models.affinity_predictor import TimeAwareAffinityPredictor
from pharmadiff.datasets.plinder_dataset import PlinderGraphDataset

@hydra.main(config_path=".", config_name="config_guided")
def generate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PharmaDiff.load_from_checkpoint(cfg.checkpoint_path)
    model.to(device)
    model.eval()

    affinity_model = TimeAwareAffinityPredictor()
    affinity_model.load_state_dict(torch.load("affinity_compass.pt"))
    affinity_model.to(device)
    affinity_model.eval()
    
    model.affinity_model = affinity_model

    test_set = PlinderGraphDataset(split='test')
    sample_data = test_set[0] 
    
    class Condition:
        pocket_pos = sample_data['pocket_pos'].to(device)
        pocket_feat = sample_data['pocket_feat'].to(device)
    
    print("Generating molecule with Affinity Guidance...")

    molecules = model.sample_batch(
        batch_size=1, 
        sample_condition=Condition(), 
        guidance_scale=10.0 
    )
    
    print("Generation Complete. Saving SDF...")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = molecules[0]
    Chem.MolToMolFile(mol, "generated_molecule.sdf")
    print("Saved generated molecule to generated_molecule.sdf")
    
if __name__ == "__main__":
    generate()
