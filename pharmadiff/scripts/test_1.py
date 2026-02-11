import torch

import pharmadiff.utils as utils
from pharmadiff.models.transformer_model import GraphTransformer


def build_random_placeholder(bs: int = 2, n: int = 8, p: int = 5):
    node_mask = torch.ones(bs, n, dtype=torch.bool)
    pharma_mask = torch.zeros(bs, n, dtype=torch.bool)
    pharma_mask[:, :3] = True

    X = torch.nn.functional.one_hot(torch.randint(0, 12, (bs, n)), num_classes=12).float()
    charges = torch.nn.functional.one_hot(torch.randint(0, 6, (bs, n)), num_classes=6).float()
    E = torch.nn.functional.one_hot(torch.randint(0, 5, (bs, n, n)), num_classes=5).float()
    E = 0.5 * (E + E.transpose(1, 2))
    pos = torch.randn(bs, n, 3)

    pharma_feat = torch.zeros(bs, n, 8)
    pharma_feat[:, :3] = torch.nn.functional.one_hot(torch.randint(0, 8, (bs, 3)), num_classes=8).float()

    pharma_atom = X * pharma_mask.unsqueeze(-1)
    pharma_charge = charges * pharma_mask.unsqueeze(-1)
    pharma_atom_pos = pos * pharma_mask.unsqueeze(-1)
    pharma_E = E * (pharma_mask.unsqueeze(2) & pharma_mask.unsqueeze(1)).unsqueeze(-1)

    pocket_pos = torch.randn(bs, p, 3)
    pocket_feat = torch.randn(bs, p, 8)
    pocket_mask = torch.ones(bs, p, dtype=torch.bool)

    y = torch.zeros(bs, 1)

    return utils.PlaceHolder(
        X=X,
        charges=charges,
        E=E,
        y=y,
        pos=pos,
        node_mask=node_mask,
        pharma_feat=pharma_feat,
        pharma_coord=pharma_atom_pos.clone(),
        pharma_mask=pharma_mask,
        pharma_atom=pharma_atom,
        pharma_atom_pos=pharma_atom_pos,
        pharma_E=pharma_E,
        pharma_charge=pharma_charge,
        pocket_pos=pocket_pos,
        pocket_feat=pocket_feat,
        pocket_mask=pocket_mask,
    ).mask()


def main():
    input_dims = utils.PlaceHolder(X=12, charges=6, E=5, y=1, pos=3, pharma_feat=8, pharma_coord=3)
    output_dims = utils.PlaceHolder(X=12, charges=6, E=5, y=0, pos=3, pharma_feat=8, pharma_coord=3)
    hidden_mlp_dims = {'X': 64, 'E': 32, 'y': 32, 'pos': 32, 'pharma_faet': 16, 'pharma_pos': 16}
    hidden_dims = {'dx': 64, 'de': 32, 'dy': 32, 'n_head': 4, 'dim_ffX': 64, 'dim_ffE': 32, 'dim_ffy': 64}

    data = build_random_placeholder()

    model_with_pocket = GraphTransformer(
        input_dims=input_dims,
        n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        conditioning_fusion="concat",
        use_pocket_interaction=True,
    )
    out_with = model_with_pocket(data)

    model_no_pocket = GraphTransformer(
        input_dims=input_dims,
        n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        conditioning_fusion="concat",
        use_pocket_interaction=False,
    )
    out_without = model_no_pocket(data)

    assert out_with.X.shape == out_without.X.shape
    assert out_with.E.shape == out_without.E.shape
    assert out_with.pos.shape == out_without.pos.shape
    assert torch.isfinite(out_with.X).all()
    assert torch.isfinite(out_with.E).all()
    assert torch.isfinite(out_with.pos).all()
    print("Phase-1 smoke test passed.")


if __name__ == "__main__":
    main()