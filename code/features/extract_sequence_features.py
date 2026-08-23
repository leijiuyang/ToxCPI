








import os, sys
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
        break

import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

BERMOL_DIR = Path(os.environ.get(
    "TOXCPI_BERMOL_DIR",
    Path(__file__).resolve().parents[1] / "third_party" / "BerMol",
)).expanduser().resolve()
BERMOL_MODEL_PATH = Path(os.environ.get(
    "TOXCPI_BERMOL_MODEL", BERMOL_DIR / "BerMolModel_base.pkl"
)).expanduser().resolve()
sys.path.insert(0, str(BERMOL_DIR))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'interaction_binding',
                         'combined_balanced_n72045_with_seq.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'pretrained_features')


def extract_bermol_tokens(device='cuda:0'):

    logger.info("=" * 60)
    logger.info("Extracting BerMol token-level features...")

    import dill as pkl
    with open(BERMOL_MODEL_PATH, "rb") as f:
        comp_model = pkl.load(f)
        comp_model.model.to(device)
        comp_model.model.eval()

    df = pd.read_csv(DATA_PATH, low_memory=False)
    chem_data = df[["chemical_name", "smiles"]].drop_duplicates(subset=["chemical_name"])
    chem_data = chem_data.dropna(subset=["smiles"])
    logger.info(f"Unique chemicals: {len(chem_data)}")

    comp_feat = {}
    failed = 0

    for _, row in tqdm(chem_data.iterrows(), total=len(chem_data), desc="BerMol tokens"):
        cid, smi = row["chemical_name"], row["smiles"]
        try:
            with torch.no_grad():
                output = comp_model.transform(smi, device)

                tokens_feat = output[0][0].cpu().numpy()
                comp_feat[cid] = tokens_feat.astype(np.float16)
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"Failed: {cid} ({str(e)[:80]})")

    logger.info(f"Extracted: {len(comp_feat)}, Failed: {failed}")


    lengths = [v.shape[0] for v in comp_feat.values()]
    logger.info(f"Token lengths: min={min(lengths)}, max={max(lengths)}, "
                f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.0f}")

    output_path = os.path.join(OUTPUT_DIR, 'compound_features_bermol_tokens.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(comp_feat, f)
    logger.info(f"Saved to {output_path}")


    total_bytes = sum(v.nbytes for v in comp_feat.values())
    logger.info(f"Total size: {total_bytes / 1024**2:.1f} MB")

    return comp_feat


def extract_esm2_residues(device='cuda:0'):

    logger.info("=" * 60)
    logger.info("Extracting ESM2 residue-level features...")

    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers

    df = pd.read_csv(DATA_PATH, low_memory=False)
    prot_data = df[["uniprot_id", "protein_sequence"]].drop_duplicates(subset=["uniprot_id"])
    prot_data = prot_data.dropna(subset=["protein_sequence"])
    logger.info(f"Unique proteins: {len(prot_data)}")

    prot_feat = {}
    failed = 0
    max_length = 1022

    for _, row in tqdm(prot_data.iterrows(), total=len(prot_data), desc="ESM2 residues"):
        pid, seq = row["uniprot_id"], row["protein_sequence"]
        try:
            seq_trunc = seq[:max_length]
            data = [(pid, seq_trunc)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[repr_layer])

            token_repr = results["representations"][repr_layer]

            residue_repr = token_repr[0, 1:len(seq_trunc)+1].cpu().numpy()
            prot_feat[pid] = residue_repr.astype(np.float16)
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"Failed: {pid} ({str(e)[:80]})")

    logger.info(f"Extracted: {len(prot_feat)}, Failed: {failed}")


    lengths = [v.shape[0] for v in prot_feat.values()]
    logger.info(f"Residue lengths: min={min(lengths)}, max={max(lengths)}, "
                f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.0f}")

    output_path = os.path.join(OUTPUT_DIR, 'protein_features_esm2_residues.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(prot_feat, f)
    logger.info(f"Saved to {output_path}")

    total_bytes = sum(v.nbytes for v in prot_feat.values())
    logger.info(f"Total size: {total_bytes / 1024**2:.1f} MB")

    return prot_feat


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    extract_bermol_tokens(device)


    extract_esm2_residues(device)

    logger.info("=" * 60)
    logger.info("All sequence-level features extracted!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
