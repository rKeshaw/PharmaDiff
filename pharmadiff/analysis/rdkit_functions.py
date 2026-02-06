import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

allowed_bonds = {'H': {0: 1, 1: 0, -1: 0},
                 'C': {0: [3, 4], 1: 3, -1: 3},
                 'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},    # In QM9, N+ seems to be present in the form NH+ and NH2+
                 'O': {0: 2, 1: 3, -1: 1},
                 'F': {0: 1, -1: 0},
                 'B': 3, 'Al': 3, 'Si': 4,
                 'P': {0: [3, 5], 1: 4},
                 'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                 'Cl': 1, 'As': 3,
                 'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'B': 119, 'Si': 148, 'P': 144, 'As': 152, 'S': 134,
                'Cl': 127, 'Br': 141, 'I': 161},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194,
                'I': 214},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168, 'I': 222, 'P': 177},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164,
                'I': 194},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156,
                'I': 187},
          'B': {'H':  119, 'Cl': 175},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'I': 243 },
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'B': 175, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210,
                'I': 234},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222},
          'I': {'H': 161, 'C': 214, 'Si': 243, 'N': 222, 'O': 194,
                'S': 234, 'F': 187, 'I': 266},
          'As': {'H': 152}
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160},
          'N': {'C': 129, 'N': 125, 'O': 121},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150},
          'P': {'O': 150, 'S': 186},
          'S': {'P': 186, 'C': 160}}


bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113}}

margin1, margin2, margin3 = 3, 2, 1




class Molecule:
    def __init__(self, atom_types, bond_types, positions, charges, atom_decoder, pharma_coord=None, pharma_feat=None, 
                 validity =None, connected=None, match_score=None):
        """ atom_types: n      LongTensor
            charges: n         LongTensor
            bond_types: n x n  LongTensor
            positions: n x 3   FloatTensor
            atom_decoder: extracted from dataset_infos. """
        assert atom_types.dim() == 1 and atom_types.dtype == torch.long, f"shape of atoms {atom_types.shape} " \
                                                                         f"and dtype {atom_types.dtype}"
        assert bond_types.dim() == 2 and bond_types.dtype == torch.long, f"shape of bonds {bond_types.shape} --" \
                                                                         f" {bond_types.dtype}"
        assert len(atom_types.shape) == 1
        assert len(bond_types.shape) == 2
        assert len(positions.shape) == 2

        self.atom_types = atom_types.long()
        self.bond_types = bond_types.long()
        self.positions = positions
        self.charges = charges
        self.rdkit_mol = self.build_molecule(atom_decoder)
        self.valence_valid = True
        self.num_nodes = len(atom_types)
        self.num_atom_types = len(atom_decoder)
        self.pharma_coord = pharma_coord
        self.pharma_feat = pharma_feat
        
        self.validity = validity
        self.connected = connected
        self.match_score = match_score

    def build_molecule(self, atom_decoder, verbose=False):
        """ If positions is None,
        """
        if verbose:
            print("building new molecule")

        mol = Chem.RWMol()
        for atom, charge in zip(self.atom_types, self.charges):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            if charge.item() != 0:
                a.SetFormalCharge(charge.item())
            mol.AddAtom(a)
            if verbose:
                print("Atom added: ", atom.item(), atom_decoder[atom.item()])

        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
                if verbose:
                    print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                          bond_dict[edge_types[bond[0], bond[1]].item()])

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None

        # Set coordinates
        positions = self.positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        mol.AddConformer(conf)

        return mol
    
    def build_frag_molecule(self, atom_decoder, verbose=False):
        if verbose:
            print("building new frag molecule")

        frags = Chem.RWMol()
        for atom, charge in zip(self.pharma_atom_types, self.pharma_charge):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            if charge.item() != 0:
                a.SetFormalCharge(charge.item())
            frags.AddAtom(a)
            if verbose:
                print("Atom added: ", atom.item(), atom_decoder[atom.item()])

        edge_types = torch.triu(self.pharma_bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                frags.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
                if verbose:
                    print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                          bond_dict[edge_types[bond[0], bond[1]].item()])

        
        #frags = Chem.AddHs(frags)
        
        # try:
        #     frags = frags.GetMol()
        # except Chem.KekulizeException:
        #     print("Can't kekulize molecule")
        #     return None

        # Set coordinates
        positions = self.pharma_pos.double()
        frags_conf = Chem.Conformer(frags.GetNumAtoms())
        for i in range(frags.GetNumAtoms()):
            frags_conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        
        frags.RemoveAllConformers() 
        frags_conf.SetId(0)  # Set the conformer ID explicitly
        frags.AddConformer(frags_conf, assignId=True)
        
    
        return frags    
    
    
    def add_bonds_based_on_distance_cutoffs(self, mol, bond1=bonds1, bonds2=bonds2, bonds3=bonds3):
        """
        Adds bonds to an RDKit molecule based on distance cutoffs.

        Parameters:
            mol (rdkit.Chem.Mol): RDKit molecule with 3D coordinates.
            min_cutoff (float): Minimum distance for bond formation (default: 0.5 Å).
            max_cutoff (float): Maximum distance for bond formation (default: 1.6 Å).
        Returns:
            rdkit.Chem.Mol: Modified molecule with additional bonds.
        """
        # if not mol.GetNumConformers():
        #     raise ValueError("Molecule must have 3D coordinates.")
    
        # Get the 3D coordinates of the molecule
        num_atoms = mol.GetNumAtoms()
        positions = self.positions 
    
        # Compute pairwise distances
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # Shape: (num_atoms, num_atoms, 3)
        distances = torch.norm(diff, dim=-1) * 100 # Shape: (num_atoms, num_atoms)

    
        # Add bonds based on distance cutoffs
        rw_mol = Chem.RWMol(mol)
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                atom1 = mol.GetAtomWithIdx(i).GetSymbol()
                atom2 = mol.GetAtomWithIdx(j).GetSymbol()
                if mol.GetBondBetweenAtoms(i, j) is None:
                    bond_type = bond_dict[get_bond_order(atom1, atom2, distances[i, j])]
                    if bond_type:
                        #print(f"Adding bond: {atom1_symbol}-{atom2_symbol} at distance {distances[i, j].item():.2f} Å")
                        rw_mol.AddBond(i, j, order=bond_type)
                 
        return rw_mol

def get_bond_order(atom1, atom2, distance):
    distance = 100 * distance  # We change the metric

    if atom1 in bonds3 and atom2 in bonds3[atom1] and distance < bonds3[atom1][atom2] + margin3:
        return 3  # Triple

    if atom1 in bonds2 and atom2 in bonds2[atom1] and distance < bonds2[atom1][atom2] + margin2:
        return 2  # Double

    if atom1 in bonds1 and atom2 in bonds1[atom1] and distance < bonds1[atom1][atom2] + margin1:
        return 1  # Single

    return 0      # No bond


def check_stability(molecule, dataset_info, debug=False, atom_decoder=None, smiles=None):
    """ molecule: Molecule object. """
    device = molecule.atom_types.device
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    atom_types = molecule.atom_types
    edge_types = molecule.bond_types

    edge_types[edge_types == 4] = 1.5
    edge_types[edge_types < 0] = 0

    valencies = torch.sum(edge_types, dim=-1).long()

    n_stable_bonds = 0
    mol_stable = True
    for i, (atom_type, valency, charge) in enumerate(zip(atom_types, valencies, molecule.charges)):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == valency
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[charge] if charge in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == valency if type(expected_bonds) == int else valency in expected_bonds
        else:
            is_stable = valency in possible_bonds
        if not is_stable:
            mol_stable = False
        if not is_stable and debug:
            if smiles is not None:
                print(smiles)
            print(f"Invalid atom {atom_decoder[atom_type]}: valency={valency}, charge={charge}")
            print()
        n_stable_bonds += int(is_stable)

    return torch.tensor([mol_stable], dtype=torch.float, device=device),\
           torch.tensor([n_stable_bonds], dtype=torch.float, device=device),\
           len(atom_types)

