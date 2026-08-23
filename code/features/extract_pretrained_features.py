









import os
import sys
import pickle
import argparse
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class FeatureExtractor:


    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.chemberta = None
        self.chemberta_tokenizer = None
        self.esm2 = None
        self.esm2_tokenizer = None

    def load_models(self):

        logger.info("Loading ChemBERTa...")
        from transformers import AutoModel, AutoTokenizer
        self.chemberta_tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.chemberta = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.chemberta = self.chemberta.to(self.device)
        self.chemberta.eval()

        logger.info("Loading ESM-2 650M...")
        self.esm2_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
        self.esm2 = AutoModel.from_pretrained('facebook/esm2_t33_650M_UR50D')
        self.esm2 = self.esm2.to(self.device)
        self.esm2.eval()

        logger.info(f"Models loaded on {self.device}")

    @torch.no_grad()
    def extract_chem_features(self, smiles_list, batch_size=64, max_length=512):

        logger.info(f"Extracting features for {len(smiles_list)} molecules...")
        features = []

        for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa"):
            batch_smiles = smiles_list[i:i+batch_size]


            inputs = self.chemberta_tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)


            outputs = self.chemberta(**inputs)


            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(cls_emb)

        features = np.vstack(features)
        logger.info(f"Chemical features shape: {features.shape}")
        return features

    @torch.no_grad()
    def extract_prot_features(self, sequences, batch_size=8, max_length=1024):

        logger.info(f"Extracting features for {len(sequences)} proteins...")
        features = []

        for i in tqdm(range(0, len(sequences), batch_size), desc="ESM-2"):
            batch_seqs = sequences[i:i+batch_size]


            inputs = self.esm2_tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)


            with torch.cuda.amp.autocast():
                outputs = self.esm2(**inputs)


            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            features.append(cls_emb)

        features = np.vstack(features)
        logger.info(f"Protein features shape: {features.shape}")
        return features


def load_data(data_path):

    df = pd.read_csv(data_path)


    chem_col = None
    prot_col = None
    seq_col = None
    smiles_col = None

    for col in df.columns:
        if col.lower() in ['chemicalname', 'chemical_name', 'chemical', 'cid', 'compound_id']:
            chem_col = col
        if col.lower() in ['uniprotkb', 'uniprot', 'uniprot_id', 'protein_id', 'pid', 'target_id']:
            prot_col = col
        if col.lower() in ['sequence', 'protein_sequence', 'prot_seq']:
            seq_col = col
        if col.lower() in ['smiles', 'canonical_smiles']:
            smiles_col = col

    logger.info(f"Data columns: chem={chem_col}, prot={prot_col}, seq={seq_col}, smiles={smiles_col}")


    if smiles_col and chem_col:
        chem_data = df[[chem_col, smiles_col]].drop_duplicates(subset=[chem_col])
        chem_data = chem_data.set_index(chem_col)[smiles_col].to_dict()
    else:
        chem_data = {}
        logger.warning("No chemical/SMILES columns found")

    if seq_col and prot_col:
        prot_data = df[[prot_col, seq_col]].drop_duplicates(subset=[prot_col])
        prot_data = prot_data.set_index(prot_col)[seq_col].to_dict()
    else:
        prot_data = {}
        logger.warning("No protein/sequence columns found")

    return chem_data, prot_data, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size_chem', type=int, default=64)
    parser.add_argument('--batch_size_prot', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = f'cuda:0'


    if args.output_dir is None:
        args.output_dir = os.environ.get(
            'TOXCPI_FEATURE_DIR',
            str(REPOSITORY_ROOT / 'runtime' / 'pretrained_features'),
        )
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Pre-trained Feature Extraction (ChemBERTa + ESM-2)")
    logger.info("=" * 60)
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Output dir: {args.output_dir}")


    extractor = FeatureExtractor(device=device)
    extractor.load_models()


    data_path = str(REPOSITORY_ROOT / 'data' / 'binding' /
                    'combined_balanced_n72045_with_seq.csv')
    logger.info(f"Loading data from {data_path}")

    chem_data, prot_data, df = load_data(data_path)
    logger.info(f"Unique chemicals: {len(chem_data)}")
    logger.info(f"Unique proteins: {len(prot_data)}")
    logger.info(f"Total samples: {len(df)}")


    chem_ids = list(chem_data.keys())
    smiles_list = [chem_data[cid] for cid in chem_ids]
    chem_features = extractor.extract_chem_features(
        smiles_list,
        batch_size=args.batch_size_chem
    )


    chem_features_dict = {cid: feat for cid, feat in zip(chem_ids, chem_features)}
    chem_path = os.path.join(args.output_dir, 'compound_features.pkl')
    with open(chem_path, 'wb') as f:
        pickle.dump(chem_features_dict, f)
    logger.info(f"Chemical features saved to {chem_path}")


    prot_ids = list(prot_data.keys())
    sequences = [prot_data[pid] for pid in prot_ids]
    prot_features = extractor.extract_prot_features(
        sequences,
        batch_size=args.batch_size_prot
    )


    prot_features_dict = {pid: feat for pid, feat in zip(prot_ids, prot_features)}
    prot_path = os.path.join(args.output_dir, 'protein_features.pkl')
    with open(prot_path, 'wb') as f:
        pickle.dump(prot_features_dict, f)
    logger.info(f"Protein features saved to {prot_path}")

    logger.info("=" * 60)
    logger.info("Feature extraction completed!")
    logger.info(f"Chemical features: {len(chem_features_dict)} x {chem_features.shape[1]}")
    logger.info(f"Protein features: {len(prot_features_dict)} x {prot_features.shape[1]}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
