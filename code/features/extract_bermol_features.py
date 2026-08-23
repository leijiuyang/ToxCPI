







import os, sys
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
        break

import pickle
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BERMOL_DIR = Path(os.environ.get(
    "TOXCPI_BERMOL_DIR",
    Path(__file__).resolve().parents[1] / "third_party" / "BerMol",
)).expanduser().resolve()
sys.path.insert(0, str(BERMOL_DIR))

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


BERMOL_MODEL_PATH = Path(os.environ.get(
    "TOXCPI_BERMOL_MODEL", BERMOL_DIR / "BerMolModel_base.pkl"
)).expanduser().resolve()
DATA_PATH = REPOSITORY_ROOT / "data" / "binding" / "combined_balanced_n72045_with_seq.csv"
OUTPUT_DIR = Path(os.environ.get(
    "TOXCPI_FEATURE_DIR", REPOSITORY_ROOT / "runtime" / "pretrained_features"
)).expanduser().resolve()


def main():
    if not BERMOL_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "BerMol weight is missing. Set TOXCPI_BERMOL_MODEL to "
            "BerMolModel_base.pkl; see DATA_AND_MODEL_ASSETS.md."
        )
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    logger.info("=" * 60)
    logger.info("BerMol Feature Extraction")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)


    logger.info(f"Loading BerMol model from {BERMOL_MODEL_PATH}...")
    try:
        import dill as pkl
    except ImportError:
        import pickle as pkl

    with open(BERMOL_MODEL_PATH, "rb") as f:
        comp_model = pkl.load(f)
        comp_model.model.to(device)
        comp_model.model.eval()
    logger.info("BerMol model loaded.")


    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    chem_data = df[["chemical_name", "smiles"]].drop_duplicates(subset=["chemical_name"])
    chem_data = chem_data.dropna(subset=["smiles"])
    logger.info(f"Unique chemicals: {len(chem_data)}")


    comp_feat = {}
    failed = 0

    for _, row in tqdm(chem_data.iterrows(), total=len(chem_data), desc="BerMol"):
        cid, smi = row["chemical_name"], row["smiles"]
        try:
            with torch.no_grad():
                output = comp_model.transform(smi, device)
                feat = output[1].cpu().numpy().reshape(-1)
                comp_feat[cid] = feat
        except Exception as e:
            failed += 1
            if failed <= 5:
                logger.warning(f"Failed: {cid} ({str(e)[:80]})")

    logger.info(f"Extracted: {len(comp_feat)}, Failed: {failed}")
    if comp_feat:
        dim = next(iter(comp_feat.values())).shape[0]
        logger.info(f"Feature dimension: {dim}")


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'compound_features_bermol.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(comp_feat, f)
    logger.info(f"Saved to {output_path}")


    chemberta_path = OUTPUT_DIR / 'compound_features.pkl'
    if chemberta_path.exists():
        with open(chemberta_path, 'rb') as f:
            chemberta_feat = pickle.load(f)
        logger.info(f"ChemBERTa features: {len(chemberta_feat)} chemicals")
        logger.info(f"BerMol features: {len(comp_feat)} chemicals")
        overlap = set(comp_feat.keys()) & set(chemberta_feat.keys())
        logger.info(f"Overlap: {len(overlap)} chemicals")

    logger.info("Done!")


if __name__ == "__main__":
    main()
