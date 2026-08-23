




import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from typing import List, Tuple, Optional, Dict
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



BENIGNI_BOSSA_SMARTS = [
    "[N+](=O)[O-]",
    "N=N",
    "[NH2]c1ccccc1",
    "c1ccc2c(c1)ccc1ccccc12",
    "[N+]([O-])=O",
    "C(=O)Cl",
    "C1CO1",
    "[N;R1]1[C;R1][C;R1]1",
    "C=CC=O",
    "C=CC(=O)C",
    "N(C)(C)N",
    "NN",
    "[N;!R]=O",
    "N=C=O",
    "N=C=S",
    "[SH]",
    "S=C=S",
    "C#N",
    "[As]",
    "[Pb]",
    "[Cd]",
    "[Hg]",
    "[Cr]",
    "c1ccc(N)c(N)c1",
    "c1cc(N)ccc1N",
    "C(Cl)(Cl)Cl",
    "C(Br)(Br)Br",
    "[CH2]Cl",
    "[CH2]Br",
    "O=S(=O)([O-])",
    "c1ccc(O)c(O)c1",
    "C=C(C#N)C#N",
    "[N+]#[C-]",
]


EDC_SMARTS = [

    "c1ccc(O)cc1",
    "Cc1ccc(O)cc1",
    "CC(C)c1ccc(O)cc1",
    "c1ccc(O)c(O)c1",
    "c1cc(O)cc(O)c1",
    "c1ccc(O)cc1O",
    "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
    "Oc1ccc(cc1)C(c1ccc(O)cc1)(c1ccc(O)cc1)c1ccc(O)cc1",
    "Clc1ccc(O)cc1",
    "Clc1cc(Cl)c(O)cc1",


    "c1ccc2c(c1)C(=O)OC2=O",
    "O=C(O)c1ccccc1C(=O)O",
    "CCCCOC(=O)c1ccccc1C(=O)OCCCC",
    "CCCCCCCCOC(=O)c1ccccc1",
    "c1ccc(C(=O)O)c(C(=O)O)c1",


    "c1ccc(-c2ccccc2)cc1",
    "Clc1ccccc1-c1ccccc1",
    "Clc1ccc(-c2ccc(Cl)cc2)cc1",
    "Clc1cc(Cl)c(-c2ccccc2)cc1",
    "Clc1ccc(-c2ccc(Cl)cc2Cl)cc1",


    "ClC(Cl)=C(c1ccc(Cl)cc1)c1ccc(Cl)cc1",
    "ClC1C(Cl)C(Cl)C(Cl)C(Cl)C1Cl",
    "ClC(Cl)C(Cl)Cl",
    "ClC=CCl",
    "ClC(Cl)(Cl)C(Cl)(Cl)Cl",
    "c1cc(Cl)c(Cl)c(Cl)c1",
    "Clc1cccc(Cl)c1Cl",
    "OC(c1ccc(Cl)cc1)(c1ccc(Cl)cc1)C(Cl)(Cl)Cl",
    "c1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl",
    "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl",


    "Brc1ccc(Oc2ccccc2)cc1",
    "Brc1ccc(Oc2ccc(Br)cc2)cc1",
    "c1ccc(Oc2ccccc2)cc1",
    "Brc1cc(Br)c(Oc2ccccc2)cc1",
    "Brc1ccc(Oc2cc(Br)ccc2Br)cc1Br",


    "CCCCCCCCC(C)c1ccc(O)cc1",
    "CCCCCCCCc1ccc(O)cc1",
    "CCCCCCCCCC(C)c1ccc(O)cc1",
    "CC(C)(C)c1ccc(O)cc1",
    "CC(C)(C)c1ccc(O)c(C(C)(C)C)c1",


    "O=C1c2ccccc2C(=O)c2ccccc12",
    "c1ccc2[nH]c3ccccc3c2c1",
    "O=C1NC(=O)c2ccccc12",
    "Nc1ccc(N)cc1",
    "c1ccc2c(c1)ccc1ccc3cccc4ccc2c1c34",
    "S=C(N)N",
    "CC(=O)Nc1ccc(O)cc1",
    "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1",
    "CCCCCCCCCCCCCCCCC(=O)O",
    "CCCCCCCCCCCC(=O)O",
]


def get_benigni_bossa_features(mol: Chem.Mol) -> np.ndarray:









    if mol is None:
        return np.zeros(len(BENIGNI_BOSSA_SMARTS), dtype=np.float32)

    features = []
    for smarts in BENIGNI_BOSSA_SMARTS:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                has_match = mol.HasSubstructMatch(pattern)
                features.append(1.0 if has_match else 0.0)
            else:
                features.append(0.0)
        except Exception:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def get_edc_features(mol: Chem.Mol) -> np.ndarray:









    if mol is None:
        return np.zeros(len(EDC_SMARTS), dtype=np.float32)

    features = []
    for smarts in EDC_SMARTS:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None:
                has_match = mol.HasSubstructMatch(pattern)
                features.append(1.0 if has_match else 0.0)
            else:
                features.append(0.0)
        except Exception:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def get_toxprint_features(mol: Chem.Mol) -> np.ndarray:










    if mol is None:
        return np.zeros(729, dtype=np.float32)

    try:

        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
        morgan_arr = np.array(morgan_fp, dtype=np.float32)


        maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        maccs_arr = np.array(maccs_fp, dtype=np.float32)


        tox_descriptors = []


        tox_descriptors.append(Descriptors.MolWt(mol) / 500.0)

        tox_descriptors.append(Descriptors.MolLogP(mol) / 5.0)

        tox_descriptors.append(Descriptors.TPSA(mol) / 150.0)

        tox_descriptors.append(Descriptors.NumHDonors(mol) / 5.0)

        tox_descriptors.append(Descriptors.NumHAcceptors(mol) / 10.0)

        tox_descriptors.append(Descriptors.NumRotatableBonds(mol) / 10.0)

        tox_descriptors.append(Descriptors.NumAromaticRings(mol) / 5.0)

        tox_descriptors.append(Descriptors.NumHeteroatoms(mol) / 10.0)

        tox_descriptors.append(rdMolDescriptors.CalcNumHeavyAtoms(mol) / 50.0)

        tox_descriptors.append(Descriptors.RingCount(mol) / 5.0)


        atom_counts = []
        atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B',
                      'Si', 'Se', 'As', 'Hg', 'Pb', 'Cd', 'Cr', 'Ni', 'Co', 'Cu',
                      'Zn', 'Fe', 'Mn', 'Al', 'Sn', 'Sb', 'Te', 'Bi', 'Tl', 'Ba',
                      'Sr', 'Ca', 'Mg', 'Na', 'K', 'Li', 'Ag', 'Au', 'Pt', 'Pd']

        for atom_type in atom_types:
            count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == atom_type)
            atom_counts.append(count / 10.0)

        tox_descriptors.extend(atom_counts)
        tox_arr = np.array(tox_descriptors, dtype=np.float32)


        features = np.concatenate([morgan_arr, maccs_arr, tox_arr])

        return features

    except Exception as e:
        logger.warning(f"Error computing ToxPrint features: {e}")
        return np.zeros(729, dtype=np.float32)


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:









    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:

            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:









    mol = smiles_to_mol(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    return None


def get_atom_features(atom: Chem.Atom) -> np.ndarray:









    features = []


    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'other']
    symbol = atom.GetSymbol()
    for at in atom_types:
        features.append(1.0 if symbol == at else 0.0)
    if symbol not in atom_types[:-1]:
        features[-1] = 1.0


    degree = atom.GetDegree()
    for i in range(6):
        features.append(1.0 if degree == i else 0.0)


    charge = atom.GetFormalCharge()
    for c in [-2, -1, 0, 1, 2]:
        features.append(1.0 if charge == c else 0.0)


    hybridization = atom.GetHybridization()
    hyb_types = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
    for hyb in hyb_types:
        features.append(1.0 if hybridization == hyb else 0.0)


    features.append(1.0 if atom.IsInRing() else 0.0)


    features.append(1.0 if atom.GetIsAromatic() else 0.0)


    num_h = atom.GetTotalNumHs()
    for i in range(5):
        features.append(1.0 if num_h == i else 0.0)


    chiral = atom.GetChiralTag()
    features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0.0)
    features.append(1.0 if chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0.0)


    features.append(atom.GetMass() / 100.0)


    features.append(Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) / 3.0)


    features.append(Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum()) / 8.0)

    return np.array(features, dtype=np.float32)


def get_bond_features(bond: Chem.Bond) -> np.ndarray:









    features = []


    bond_types = [Chem.rdchem.BondType.SINGLE,
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE,
                  Chem.rdchem.BondType.AROMATIC]
    bt = bond.GetBondType()
    for bond_type in bond_types:
        features.append(1.0 if bt == bond_type else 0.0)


    features.append(1.0 if bond.GetIsConjugated() else 0.0)


    features.append(1.0 if bond.IsInRing() else 0.0)


    stereo = bond.GetStereo()
    stereo_types = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE,
                    Chem.rdchem.BondStereo.STEREOCIS]
    for st in stereo_types:
        features.append(1.0 if stereo == st else 0.0)

    return np.array(features, dtype=np.float32)


def set_seed(seed: int):






    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:









    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, path):






    import os
    import tempfile


    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }


    save_dir = os.path.dirname(path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)


    temp_path = path + '.tmp'
    try:
        torch.save(checkpoint, temp_path)

        os.replace(temp_path, path)
    except Exception as e:

        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def load_checkpoint(model, optimizer, path):



    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


class EarlyStopping:


    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):






        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
