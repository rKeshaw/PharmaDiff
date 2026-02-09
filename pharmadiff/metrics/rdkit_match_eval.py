from __future__ import print_function
from rdkit import RDConfig, Chem, Geometry, DistanceGeometry
from rdkit.Chem import ChemicalFeatures, rdDistGeom, Draw, rdMolTransforms
from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Numerics import rdAlignment
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors
import pickle

import math
from collections import defaultdict


import os
import os.path as op
import torch
from pharmadiff.datasets.pharmacophore_utils import get_features_factory
from collections import Counter
import csv


RDLogger.DisableLog('rdApp.*')

__FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))


PHARMACOPHORE_FAMILES_TO_KEEP = ('Aromatic', 'Hydrophobe', 'PosIonizable', 'Acceptor', 'Donor', 'LumpedHydrophobe')
FAMILY_MAPPING = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3, 'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}
_FEATURES_FACTORY = []


def applyRadiiToBounds(radii,pcophore):
  for i in range(len(radii)):
    for j in range(i+1,len(radii)):
      sumRadii = radii[i]+radii[j]
      pcophore.setLowerBound(i,j,max(pcophore.getLowerBound(i,j)-sumRadii,0))
      pcophore.setUpperBound(i,j,pcophore.getUpperBound(i,j)+sumRadii)

def match_mol(mol, pharma_feat, pharma_coord, tolerance=1.21):
    
    try:
        mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())

        mol.UpdatePropertyCache(strict=False)
        mol = Chem.AddHs(mol, addCoords=True) 
        Chem.GetSSSR(mol)
        Chem.SanitizeMol(mol)
            
        #print("Sanitization successful")
    except Exception as e:
        #print(f"Sanitization failed: {e}")
        return -1
    
    
    Ph4Feats = []
    radii = []
    for i in range(len(pharma_feat)):
        feat = PHARMACOPHORE_FAMILES_TO_KEEP[int(pharma_feat[i])]
        g = Geometry.Point3D(pharma_coord[i, 0].item(), pharma_coord[i, 1].item(), pharma_coord[i, 2].item())
        Ph4Feats.append(ChemicalFeatures.FreeChemicalFeature(feat, g))
        radii.append(tolerance)
        
    pcophore = Pharmacophore.Pharmacophore(Ph4Feats)
    applyRadiiToBounds(radii,pcophore)
    
    try:
        # mol.UpdatePropertyCache(strict=False)
        canMatch,allMatches = EmbedLib.MatchPharmacophoreToMol(mol,__FACTORY,pcophore)
        boundsMat = rdDistGeom.GetMoleculeBoundsMatrix(mol)
        
        if canMatch:
            failed,boundsMatMatched,matched,matchDetails = EmbedLib.MatchPharmacophore(allMatches,
                                                                                       boundsMat,
                                                                                       pcophore,
                                                                                       useDownsampling=False)
            if failed == 0:
                return 1
            else:
                return 0
        else:
            return 0
    except Exception as e:
        # print(f"An error occurred: {e}")
        return -1




def check_ring_filter(linker):
    check = True
    # Get linker rings
    ssr = Chem.GetSymmSSSR(linker)
    # Check rings
    for ring in ssr:
        for atom_idx in ring:
            for bond in linker.GetAtomWithIdx(atom_idx).GetBonds():
                if bond.GetBondType() == 2 and bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring:
                    check = False
    return check


def check_pains(mol, pains_smarts):
        
    for pain in pains_smarts:
        if mol.HasSubstructMatch(pain):
            return False
    return True
        
        
        
_fscores = None


# Get the absolute path to the top-level PharamDiff_frag directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Build path to fpscores
fpscores_path = os.path.join(project_root, "resources", "fpscores")


def readFragmentScores(name=fpscores_path):
    import gzip
    global _fscores
  # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    with gzip.open('%s.pkl.gz' % name, 'rb') as f:
        _fscores = pickle.load(f)
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro

def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

  # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)  #<- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
  # ---------------------------------------
  # This differs from the paper, which defines:
  #  macrocyclePenalty = math.log10(nMacrocycles+1)
  # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

  # correction for the fingerprint density
  # not in the original publication, added in version 1.1
  # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

  # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
  # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore
        