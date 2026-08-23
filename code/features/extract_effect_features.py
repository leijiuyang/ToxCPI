














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
import requests
import time




BERMOL_DIR = Path(os.environ.get(
    "TOXCPI_BERMOL_DIR",
    Path(__file__).resolve().parents[1] / "third_party" / "BerMol",
)).expanduser().resolve()
BERMOL_MODEL_PATH = Path(os.environ.get(
    "TOXCPI_BERMOL_MODEL", BERMOL_DIR / "BerMolModel_base.pkl"
)).expanduser().resolve()
sys.path.insert(0, str(BERMOL_DIR))

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = str(Path(os.environ.get(
    "TOXCPI_FEATURE_DIR", REPOSITORY_ROOT / "runtime" / "pretrained_features"
)).expanduser().resolve())
ACTIVATION_DIR = REPOSITORY_ROOT / 'data' / 'activation'
INHIBITION_DIR = REPOSITORY_ROOT / 'data' / 'inhibition'
ESM2_MODEL_PATH = Path(os.environ.get("TOXCPI_ESM2_MODEL", "")).expanduser()

BERMOL_TOKENS_PATH = os.path.join(FEATURE_DIR, 'compound_features_bermol_tokens.pkl')
ESM2_RESIDUES_PATH = os.path.join(FEATURE_DIR, 'protein_features_esm2_residues.pkl')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_effect_entities():

    act = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in sorted(ACTIVATION_DIR.glob('effect_activation_*.csv'))],
        ignore_index=True,
    )
    inh = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in sorted(INHIBITION_DIR.glob('effect_inhibition_*.csv'))],
        ignore_index=True,
    )
    all_data = pd.concat([act, inh])

    prots = all_data[['uniprot_id', 'sequence']].drop_duplicates(subset=['uniprot_id'])
    chems = all_data[['chemical_name', 'smiles']].drop_duplicates(subset=['chemical_name'])
    chems = chems.dropna(subset=['smiles'])

    return prots, chems





def fetch_uniprot_sequences(uniprot_ids, batch_size=100):

    logger.info(f"从 UniProt 获取 {len(uniprot_ids)} 个蛋白质序列...")

    results = {}
    ids_list = list(uniprot_ids)

    for i in range(0, len(ids_list), batch_size):
        batch = ids_list[i:i+batch_size]
        query = '+OR+'.join([f'accession:{uid}' for uid in batch])
        url = f'https://rest.uniprot.org/uniprotkb/search?query={query}&fields=accession,sequence&format=tsv&size=500'

        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    lines = resp.text.strip().split('\n')
                    for line in lines[1:]:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            uid, seq = parts[0], parts[1]
                            if seq:
                                results[uid] = seq
                    break
                else:
                    logger.warning(f"HTTP {resp.status_code}, retry {attempt+1}")
                    time.sleep(2)
            except Exception as e:
                logger.warning(f"Request failed: {e}, retry {attempt+1}")
                time.sleep(2)

        if (i // batch_size + 1) % 5 == 0:
            logger.info(f"  进度: {min(i+batch_size, len(ids_list))}/{len(ids_list)}, 已获取 {len(results)}")

    logger.info(f"UniProt 获取完成: {len(results)}/{len(uniprot_ids)}")
    return results


def fill_missing_sequences(prots):


    seq_map = {}
    for _, row in prots.iterrows():
        if pd.notna(row['sequence']):
            seq_map[row['uniprot_id']] = row['sequence']

    missing_ids = set(prots['uniprot_id']) - set(seq_map.keys())
    logger.info(f"蛋白质: 总 {len(prots)}, 已有序列 {len(seq_map)}, 缺失 {len(missing_ids)}")

    if missing_ids:
        fetched = fetch_uniprot_sequences(missing_ids)
        seq_map.update(fetched)
        still_missing = missing_ids - set(fetched.keys())
        if still_missing:
            logger.warning(f"仍有 {len(still_missing)} 个蛋白无法获取序列: {list(still_missing)[:10]}...")

    logger.info(f"最终有序列的蛋白: {len(seq_map)}")
    return seq_map





def extract_missing_bermol(chems, device='cuda:0'):

    logger.info("=" * 60)
    logger.info("Step 2: 提取缺失 BerMol token 级特征")


    if os.path.exists(BERMOL_TOKENS_PATH):
        existing = pickle.load(open(BERMOL_TOKENS_PATH, 'rb'))
    else:
        existing = {}
    logger.info(f"已有 BerMol 特征: {len(existing)}")


    missing = chems[~chems['chemical_name'].isin(existing.keys())]
    logger.info(f"需提取: {len(missing)}")

    if len(missing) == 0:
        logger.info("无需提取")
        return existing


    import dill as pkl
    with open(BERMOL_MODEL_PATH, "rb") as f:
        comp_model = pkl.load(f)
        comp_model.model.to(device)
        comp_model.model.eval()

    new_feat = {}
    failed = 0

    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="BerMol tokens"):
        cid, smi = row['chemical_name'], row['smiles']
        try:
            with torch.no_grad():
                output = comp_model.transform(smi, device)
                tokens_feat = output[0][0].cpu().numpy()
                new_feat[cid] = tokens_feat.astype(np.float16)
        except Exception as e:
            failed += 1
            if failed <= 10:
                logger.warning(f"Failed: {cid} ({str(e)[:80]})")

    logger.info(f"新提取: {len(new_feat)}, 失败: {failed}")


    existing.update(new_feat)
    with open(BERMOL_TOKENS_PATH, 'wb') as f:
        pickle.dump(existing, f)
    logger.info(f"已保存: {BERMOL_TOKENS_PATH} (共 {len(existing)} 个)")

    return existing





def extract_missing_esm2(seq_map, device='cuda:0'):

    logger.info("=" * 60)
    logger.info("Step 3: 提取缺失 ESM2 残基级特征")


    if os.path.exists(ESM2_RESIDUES_PATH):
        existing = pickle.load(open(ESM2_RESIDUES_PATH, 'rb'))
    else:
        existing = {}
    logger.info(f"已有 ESM2 特征: {len(existing)}")


    missing = {k: v for k, v in seq_map.items() if k not in existing}
    logger.info(f"需提取: {len(missing)}")

    if len(missing) == 0:
        logger.info("无需提取")
        return existing


    if not ESM2_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "ESM-2 weight is missing. Set TOXCPI_ESM2_MODEL to "
            "esm2_t33_650M_UR50D.pt; see DATA_AND_MODEL_ASSETS.md."
        )
    from esm.pretrained import load_model_and_alphabet_core
    model_data = torch.load(ESM2_MODEL_PATH, map_location='cpu', weights_only=False)
    model, alphabet = load_model_and_alphabet_core(
        'esm2_t33_650M_UR50D', model_data, None
    )
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers
    max_length = 1022

    new_feat = {}
    failed = 0

    for pid, seq in tqdm(missing.items(), desc="ESM2 residues"):
        try:
            seq_trunc = seq[:max_length]
            data = [(pid, seq_trunc)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[repr_layer])

            token_repr = results["representations"][repr_layer]
            residue_repr = token_repr[0, 1:len(seq_trunc)+1].cpu().numpy()
            new_feat[pid] = residue_repr.astype(np.float16)
        except Exception as e:
            failed += 1
            if failed <= 10:
                logger.warning(f"Failed: {pid} ({str(e)[:80]})")

    logger.info(f"新提取: {len(new_feat)}, 失败: {failed}")


    existing.update(new_feat)
    with open(ESM2_RESIDUES_PATH, 'wb') as f:
        pickle.dump(existing, f)
    logger.info(f"已保存: {ESM2_RESIDUES_PATH} (共 {len(existing)} 个)")

    return existing





def main():
    skip_bermol = '--skip-bermol' in sys.argv
    skip_esm2 = '--skip-esm2' in sys.argv

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")


    prots, chems = load_effect_entities()
    logger.info(f"Effect 任务实体: {len(prots)} 蛋白, {len(chems)} 化学品")


    seq_map = fill_missing_sequences(prots)


    if not skip_bermol:
        extract_missing_bermol(chems, device)
    else:
        logger.info("跳过 BerMol 提取")


    if not skip_esm2:
        extract_missing_esm2(seq_map, device)
    else:
        logger.info("跳过 ESM2 提取")


    logger.info("=" * 60)
    logger.info("最终覆盖率:")
    prot_esm = pickle.load(open(ESM2_RESIDUES_PATH, 'rb'))
    chem_bermol = pickle.load(open(BERMOL_TOKENS_PATH, 'rb'))

    prot_covered = sum(1 for p in prots['uniprot_id'] if p in prot_esm)
    chem_covered = sum(1 for c in chems['chemical_name'] if c in chem_bermol)
    logger.info(f"  蛋白: {prot_covered}/{len(prots)} ({prot_covered/len(prots)*100:.1f}%)")
    logger.info(f"  化学品: {chem_covered}/{len(chems)} ({chem_covered/len(chems)*100:.1f}%)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
