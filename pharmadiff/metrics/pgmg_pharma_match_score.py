import numpy as np
from rdkit import RDConfig, Chem, Geometry, DistanceGeometry
from rdkit.Chem import ChemicalFeatures, rdDistGeom, Draw, rdMolTransforms
from rdkit.Numerics import rdAlignment
from rdkit import RDLogger
import os
from itertools import product, permutations
from pharmadiff.datasets.pharmacophore_utils import get_features_factory
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib

__FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
__MAPPING = {'Aromatic': 0, 'Hydrophobe': 1, 'PosIonizable': 2, 'Acceptor': 3, 'Donor':4 , 'LumpedHydrophobe': 5}


def __iter_product(phco, phar_grouped):
    group_elements = [None for _ in range(len(phar_grouped))]
    n_places = []
    for i in range(len(phar_grouped)):
        group_elements[i] = list(range(len(phco[phar_grouped[i][0]])))
        l_elements = len(group_elements[i])
        l_places = len(phar_grouped[i])
        n_places.append(l_places)

        if l_elements < l_places:
            group_elements[i].extend([None] * (l_places - l_elements))

    for i in product(*[permutations(i, n) for i, n in zip(group_elements, n_places)]):
        res = [None] * len(phco)

        for g_ele, g_idx in zip(i, phar_grouped):
            for a, b in zip(g_ele, g_idx):
                res[b] = a

        yield res


def extract_info_from_lists(node_types, coordinates):
    # Generate distance dictionary
    ref_dist_list = []
    value = []
    num_nodes = len(node_types)
    
    # Assuming all nodes are connected in a complete graph
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            ref_dist_name = f"{i}{j}"  # Assuming undirected edges
            ref_dist_list.append(ref_dist_name)
            # Calculate distance between nodes i and j using coordinates
            dist_ij = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
            value.append(dist_ij)
    
    dist_dict = dict(zip(ref_dist_list, value))
    
    # Generate type list
    type_list = []
    for node_type in node_types:
        type_list.append(tuple(node_type))
    
    return dist_dict, type_list


PHARMACOPHORE_FAMILES_TO_KEEP = ('Aromatic', 'Hydrophobe', 'PosIonizable', 'Acceptor', 'Donor', 'LumpedHydrophobe')

def match_score(mol, node_types, coordinates):
    if mol is None:
        return -1
    
    unique_coords, unique_indices = np.unique(coordinates, axis=0, return_index=True)
    coordinates = coordinates[unique_indices]
    node_types = node_types[unique_indices]
    
    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        mol.UpdatePropertyCache()
        mol = Chem.AddHs(mol, addCoords=True) 
        Chem.GetSSSR(mol)
        Chem.SanitizeMol(mol)        
    except Exception as e:
        #print(f"Sanitization failed: {e}")
        return -1
    
     
    feature_factory, keep_featnames = get_features_factory(PHARMACOPHORE_FAMILES_TO_KEEP)
        
    try:
        feats = feature_factory.GetFeaturesForMol(mol, confId=-1)
    except:
        return -1
        
    if len(node_types) == 1:
        for f in feats:
            feature = f.GetFamily()
            if feature == PHARMACOPHORE_FAMILES_TO_KEEP[node_types[0, 0]]:
                return 1
        return 0


    dist, ref_type =  extract_info_from_lists(node_types, coordinates)

    all_phar_types = {i for j in ref_type for i in j}

    phar_filter = [[] for _ in range(len(ref_type))]

    phar_mapping = {i: [] for i in ref_type}
    for i in range(len(ref_type)):
        phar_mapping[ref_type[i]].append(i)
    mol_phco_candidate = []
    for f in feats:
        phar = f.GetFamily()
        phar_index = __MAPPING.setdefault(phar, 7)

        if phar_index not in all_phar_types:
            continue
        atom_index = f.GetAtomIds()
        atom_index = tuple(sorted(atom_index))
        phar_info = ((phar_index,), atom_index)
        mol_phco_candidate.append(phar_info)

    tmp_n = len(mol_phco_candidate)
    for i in range(tmp_n):
        phar_i, atom_i = mol_phco_candidate[i]
        for j in range(i + 1, tmp_n):
            phar_j, atom_j = mol_phco_candidate[j]
            if atom_i == atom_j and phar_i != phar_j:
                phars = tuple(sorted((phar_i[0], phar_j[0])))
                mol_phco_candidate.append([phars, atom_i])

    for phar, atoms in mol_phco_candidate:
        if phar in phar_mapping:
            for idx in phar_mapping[phar]:
                phar_filter[idx].append(atoms)
    try:
        match_score = max_match(mol, node_types, coordinates, phar_filter, phar_mapping.values())
    except:
        return -1
    return match_score


def cal_dist_all(mol, phco_list_i, phco_list_j):
    for phco_elment_i in phco_list_i:
        for phco_elment_j in phco_list_j:
            if phco_elment_i == phco_elment_j:
                if len(phco_list_i) == 1 and len(phco_list_j) == 1:
                    dist = 0
                else:
                    dist = max(len(phco_list_i), len(phco_list_j)) * 0.2
        if not set(phco_list_i).intersection(set(phco_list_j)):
            dist_set = []
            for atom_i in phco_list_i:
                for atom_j in phco_list_j:
                    dist_ = get_atom_distance(mol, atom_i, atom_j)
                    dist_set.append(dist_)
            min_dist = min(dist_set)
            if max(len(phco_list_i), len(phco_list_j)) == 1:
                dist = min_dist
            else:
                dist = min_dist + max(len(phco_list_i), len(phco_list_j)) * 0.2
    return dist


def get_atom_distance(mol, atom_idx1, atom_idx2):
    # Get the conformer
    conf = mol.GetConformer()

    # Get atom positions
    pos1 = conf.GetAtomPosition(atom_idx1)
    pos2 = conf.GetAtomPosition(atom_idx2)
    
    # Calculate distance using numpy's norm function
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))

    return distance


def max_match(mol, node_types, coordinates, phco, phar_mapping):
    # will modify phar_filter

    ref_dist, ref_type =  extract_info_from_lists(node_types, coordinates)

    length = len(phco)

    dist_dict = {}
    for i in range(length - 1):
        for j in range(i + 1, length):
            for elment_len1 in range(len(phco[i])):
                for elment_len2 in range(len(phco[j])):
                    if phco[i][elment_len1] is None or phco[j][elment_len2] is None:
                        dist = 100
                    else:
                        dist = cal_dist_all(mol, phco[i][elment_len1], phco[j][elment_len2])  ##

                    dist_name = (i, elment_len1, j, elment_len2)

                    dist_dict[dist_name] = dist

    match_score_max = 0
    for perm_idx, phco_elment_list in enumerate(__iter_product(phco, list(phar_mapping))):
        if perm_idx >= 2000:
            break

        error_count = 0
        correct_count = 0

        for p in range(len(phco_elment_list)):
            for q in range(p + 1, len(phco_elment_list)):

                key_ = (p, phco_elment_list[p], q, phco_elment_list[q])

                if phco_elment_list[p] is None or phco_elment_list[q] is None:
                    dist_ref_candidate = 100
                else:
                    dist_ref_candidate = abs(dist_dict[key_] - ref_dist['{}''{}'.format(p, q)])
                if dist_ref_candidate < 1.21:
                    correct_count += 1
                else:
                    error_count += 1
        match_score = correct_count / (correct_count + error_count)

        match_score_max = max(match_score, match_score_max)

        if match_score_max == 1:
            return match_score_max

    return match_score_max
