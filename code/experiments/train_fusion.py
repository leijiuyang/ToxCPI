










import glob
import os, sys
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
        break

import argparse
import ast
import logging
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from rdkit import Chem
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, matthews_corrcoef
)


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import canonicalize_smiles, smiles_to_mol, get_toxprint_features, get_benigni_bossa_features, get_edc_features
from model_fusion import BindingModelFusion


if __name__ == "__main__":
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPOSITORY_ROOT / "runtime"
FEATURE_DIR = Path(os.environ.get(
    "TOXCPI_FEATURE_DIR", RUNTIME_ROOT / "pretrained_features"
)).expanduser().resolve()


def load_partial_checkpoint(model, ckpt_path, device):

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Init checkpoint not found: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    checkpoint = {(k[7:] if k.startswith('module.') else k): v for k, v in checkpoint.items()}
    model_state = model.state_dict()
    load_state = {}
    loaded_keys = []
    skipped = []
    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            load_state[key] = value
            loaded_keys.append(key)
        else:
            reason = 'missing' if key not in model_state else f"shape {tuple(value.shape)} -> {tuple(model_state[key].shape)}"
            skipped.append((key, reason))
    model_state.update(load_state)
    model.load_state_dict(model_state, strict=False)
    param_names = {name for name, _ in model.named_parameters()}
    loaded_params = sorted(k for k in loaded_keys if k in param_names)
    logger.info(f"Warm-start loaded {len(loaded_keys)} tensors from {ckpt_path}")
    logger.info(f"Warm-start loaded trainable params: {len(loaded_params)}")
    if skipped:
        logger.info(f"Warm-start skipped {len(skipped)} tensors")
        for key, reason in skipped[:8]:
            logger.info(f"  skip {key}: {reason}")
    return set(loaded_params)


def set_params_trainable(model, param_names, trainable):

    affected = 0
    for name, param in model.named_parameters():
        if name in param_names:
            param.requires_grad = trainable
            affected += param.numel()
    state = "unfrozen" if trainable else "frozen"
    logger.info(f"Warm-start backbone {state}: {affected:,} parameters")


def load_prebuilt_splits(split_dir, split, logger):

    paths = {
        'train': os.path.join(split_dir, f'train_{split}.csv'),
        'val': os.path.join(split_dir, f'val_{split}.csv'),
        'test': os.path.join(split_dir, f'test_{split}.csv'),
    }
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Prebuilt split files not found: " + ", ".join(missing) +
            ". Provide the prebuilt split files or regenerate them with the "
            "data-preparation workflow."
        )

    train_df = pd.read_csv(paths['train'], low_memory=False)
    val_df = pd.read_csv(paths['val'], low_memory=False)
    test_df = pd.read_csv(paths['test'], low_memory=False)

    required_cols = ['smiles', 'protein_sequence', 'label', 'uniprot_id',
                     'chemical_name', 'data_source']
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        missing_cols = [col for col in required_cols if col not in split_df.columns]
        if missing_cols:
            raise ValueError(f"{name}_{split}.csv missing required columns: {missing_cols}")
        split_df['smiles_clean'] = split_df['smiles'].apply(canonicalize_smiles)
        logger.info(f"Loaded {name}_{split}.csv: {len(split_df)} samples")

    return train_df, val_df, test_df




class EMAModel:


    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):

        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}




class ContrastiveLoss(nn.Module):





    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, chem_emb, prot_emb, labels):






        chem_emb = F.normalize(chem_emb, dim=-1)
        prot_emb = F.normalize(prot_emb, dim=-1)


        sim = chem_emb @ prot_emb.T / self.temperature

        pos_mask = labels.bool()
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=chem_emb.device)


        log_prob_c2p = F.log_softmax(sim, dim=1)
        loss_c2p = -torch.diag(log_prob_c2p)[pos_mask].mean()


        log_prob_p2c = F.log_softmax(sim, dim=0)
        loss_p2c = -torch.diag(log_prob_p2c)[pos_mask].mean()

        return (loss_c2p + loss_p2c) / 2




class FusionDataset(Dataset):






    def __init__(self, df, comp_feats, prot_feats, chem_col, prot_col, label_col,
                 tox_cache_path=None, use_residue_attention=False, max_protein_length=768,
                 max_chem_tokens=64, shared_tensors=None, use_molecular_gcn=False,
                 chem_smiles_map=None, return_indices=False):
        df = df.reset_index(drop=True)
        self.use_residue_attention = use_residue_attention
        self.use_molecular_gcn = use_molecular_gcn
        self.return_indices = return_indices


        valid_mask = df.apply(
            lambda row: str(row[chem_col]) in comp_feats and str(row[prot_col]) in prot_feats,
            axis=1
        )
        df_valid = df[valid_mask].reset_index(drop=True)
        n_filtered = len(df) - len(df_valid)
        logger.info(f"Valid samples: {len(df_valid)}/{len(df)} "
                    f"(filtered {n_filtered} without pretrained features)")

        n = len(df_valid)
        self.labels = torch.zeros(n, dtype=torch.float32)

        if use_residue_attention:
            if shared_tensors is not None:

                self.chem_tensor = shared_tensors['chem_tensor']
                self.chem_lengths = shared_tensors['chem_lengths']
                self.prot_tensor = shared_tensors['prot_tensor']
                self.prot_lengths = shared_tensors['prot_lengths']
                self.chem_adj_tensor = shared_tensors.get('chem_adj_tensor')
                self.prot_site_tensor = shared_tensors.get('prot_site_tensor')
                chem_id2idx = shared_tensors['chem_id2idx']
                prot_id2idx = shared_tensors['prot_id2idx']
            else:

                tensors = FusionDataset.build_entity_tensors(
                    comp_feats, prot_feats, max_chem_tokens, max_protein_length,
                    build_chem_adj=use_molecular_gcn,
                    chem_smiles_map=chem_smiles_map)
                self.chem_tensor = tensors['chem_tensor']
                self.chem_lengths = tensors['chem_lengths']
                self.prot_tensor = tensors['prot_tensor']
                self.prot_lengths = tensors['prot_lengths']
                self.chem_adj_tensor = tensors.get('chem_adj_tensor')
                self.prot_site_tensor = tensors.get('prot_site_tensor')
                chem_id2idx = tensors['chem_id2idx']
                prot_id2idx = tensors['prot_id2idx']


            self.chem_indices = torch.zeros(n, dtype=torch.long)
            self.prot_indices = torch.zeros(n, dtype=torch.long)
            for i in range(n):
                row = df_valid.iloc[i]
                self.chem_indices[i] = chem_id2idx[str(row[chem_col])]
                self.prot_indices[i] = prot_id2idx[str(row[prot_col])]
                self.labels[i] = float(row[label_col])

            logger.info(f"Preloaded Step 2: {n} samples, "
                        f"{len(chem_id2idx)} unique chems, {len(prot_id2idx)} unique prots")
        else:

            tox_dict = self._build_tox_dict(df_valid['smiles'].unique(), tox_cache_path)
            tox_dim = 812
            comp_dim = next(iter(comp_feats.values())).shape[0]
            prot_dim = next(iter(prot_feats.values())).shape[0]

            self.tox_data = torch.zeros(n, tox_dim, dtype=torch.float32)
            self.comp_data = torch.zeros(n, comp_dim, dtype=torch.float32)
            self.prot_data = torch.zeros(n, prot_dim, dtype=torch.float32)

            for i in range(n):
                row = df_valid.iloc[i]
                smi = str(row['smiles'])
                chem = str(row[chem_col])
                prot = str(row[prot_col])
                self.tox_data[i] = torch.from_numpy(tox_dict.get(smi, np.zeros(tox_dim, dtype=np.float32)))
                self.comp_data[i] = torch.from_numpy(comp_feats[chem].astype(np.float32))
                self.prot_data[i] = torch.from_numpy(prot_feats[prot].astype(np.float32))
                self.labels[i] = float(row[label_col])

            logger.info(f"Preloaded Step 1: tox{list(self.tox_data.shape)}, "
                        f"comp{list(self.comp_data.shape)}, prot{list(self.prot_data.shape)}")

    @staticmethod
    def _build_tox_dict(unique_smiles, cache_path=None):

        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached ToxPrint from {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            tox_dict = cached['tox_features'].item()
            logger.info(f"Loaded {len(tox_dict)} cached ToxPrint features")
            return tox_dict

        logger.info(f"Computing ToxPrint for {len(unique_smiles)} unique SMILES...")
        tox_dict = {}
        computed, failed = 0, 0
        for smi in unique_smiles:
            mol = smiles_to_mol(smi)
            if mol is not None:
                tp = get_toxprint_features(mol)
                bg = get_benigni_bossa_features(mol)
                edc = get_edc_features(mol)
                tox_dict[smi] = np.concatenate([tp, bg, edc]).astype(np.float32)
                computed += 1
            else:
                tox_dict[smi] = np.zeros(812, dtype=np.float32)
                failed += 1

        logger.info(f"ToxPrint: {computed} computed, {failed} failed")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, tox_features=tox_dict)
            logger.info(f"Cached ToxPrint to {cache_path}")

        return tox_dict

    @staticmethod
    def build_entity_tensors(comp_feats, prot_feats, max_chem_tokens=64, max_protein_length=768,
                             build_chem_adj=False, chem_smiles_map=None):

        chem_dim = next(iter(comp_feats.values())).shape[1]
        prot_dim = next(iter(prot_feats.values())).shape[1]
        chem_smiles_map = chem_smiles_map or {}


        unique_chems = list(comp_feats.keys())
        chem_id2idx = {k: i for i, k in enumerate(unique_chems)}
        n_chems = len(unique_chems)
        chem_tensor = torch.zeros(n_chems, max_chem_tokens, chem_dim)
        chem_lengths = torch.zeros(n_chems, dtype=torch.long)
        chem_adj_tensor = torch.zeros(n_chems, max_chem_tokens, max_chem_tokens) if build_chem_adj else None
        n_truncated_chem = 0
        n_adj_from_rdkit = 0
        n_adj_fallback = 0
        for i, k in enumerate(unique_chems):
            orig_len = comp_feats[k].shape[0]
            feat = comp_feats[k][:max_chem_tokens]
            length = feat.shape[0]
            chem_tensor[i, :length] = torch.from_numpy(feat.astype(np.float32))
            chem_lengths[i] = length
            if orig_len > max_chem_tokens:
                n_truncated_chem += 1
            if build_chem_adj:
                smiles = chem_smiles_map.get(str(k))
                adj, ok = FusionDataset._build_chem_adjacency(smiles, length, max_chem_tokens)
                chem_adj_tensor[i] = adj
                if ok:
                    n_adj_from_rdkit += 1
                else:
                    n_adj_fallback += 1


        unique_prots = list(prot_feats.keys())
        prot_id2idx = {k: i for i, k in enumerate(unique_prots)}
        n_prots = len(unique_prots)
        prot_tensor = torch.zeros(n_prots, max_protein_length, prot_dim)
        prot_lengths = torch.zeros(n_prots, dtype=torch.long)
        n_truncated_prot = 0
        for i, k in enumerate(unique_prots):
            orig_len = prot_feats[k].shape[0]
            feat = prot_feats[k][:max_protein_length]
            length = feat.shape[0]
            prot_tensor[i, :length] = torch.from_numpy(feat.astype(np.float32))
            prot_lengths[i] = length
            if orig_len > max_protein_length:
                n_truncated_prot += 1

        logger.info(f"Entity tensors built (shared):")
        logger.info(f"  Chem: {list(chem_tensor.shape)} ({chem_tensor.nbytes / 1024**3:.1f} GB), "
                    f"lengths: mean={chem_lengths.float().mean():.1f}, "
                    f"median={chem_lengths.float().median():.0f}, "
                    f"max={chem_lengths.max().item()}")
        logger.info(f"  Prot: {list(prot_tensor.shape)} ({prot_tensor.nbytes / 1024**3:.1f} GB), "
                    f"lengths: mean={prot_lengths.float().mean():.1f}, "
                    f"median={prot_lengths.float().median():.0f}, "
                    f"max={prot_lengths.max().item()}")
        if n_truncated_chem > 0:
            logger.info(f"  Chem truncated: {n_truncated_chem}/{n_chems} "
                        f"({n_truncated_chem/n_chems*100:.1f}%)")
        if build_chem_adj:
            logger.info(f"  Chem adjacency: RDKit={n_adj_from_rdkit}, fallback_identity={n_adj_fallback}, "
                        f"shape={list(chem_adj_tensor.shape)}")
        if n_truncated_prot > 0:
            logger.info(f"  Prot truncated: {n_truncated_prot}/{n_prots} "
                        f"({n_truncated_prot/n_prots*100:.1f}%)")

        result = {
            'chem_tensor': chem_tensor, 'chem_lengths': chem_lengths,
            'chem_id2idx': chem_id2idx,
            'prot_tensor': prot_tensor, 'prot_lengths': prot_lengths,
            'prot_id2idx': prot_id2idx,
        }
        if build_chem_adj:
            result['chem_adj_tensor'] = chem_adj_tensor
        return result

    @staticmethod
    def _build_chem_adjacency(smiles, length, max_chem_tokens):

        adj = torch.zeros(max_chem_tokens, max_chem_tokens)
        valid_len = min(int(length), max_chem_tokens)
        if not smiles:
            if valid_len > 0:
                adj[:valid_len, :valid_len] = torch.eye(valid_len)
            return adj, False

        mol = smiles_to_mol(str(smiles))
        if mol is None:
            if valid_len > 0:
                adj[:valid_len, :valid_len] = torch.eye(valid_len)
            return adj, False

        try:
            rd_adj = torch.from_numpy(np.asarray(Chem.GetAdjacencyMatrix(mol), dtype=np.float32))
            n = min(valid_len, rd_adj.size(0), max_chem_tokens)
            if n > 0:
                adj[:n, :n] = rd_adj[:n, :n]
                if valid_len > n:
                    adj[n:valid_len, n:valid_len] = torch.eye(valid_len - n)
            return adj, True
        except Exception:
            if valid_len > 0:
                adj[:valid_len, :valid_len] = torch.eye(valid_len)
            return adj, False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.use_residue_attention:
            if self.return_indices:
                return {
                    'chem_idx': self.chem_indices[idx],
                    'prot_idx': self.prot_indices[idx],
                    'label': self.labels[idx],
                }

            ci = self.chem_indices[idx]
            pi = self.prot_indices[idx]
            cl = self.chem_lengths[ci].item()
            pl = self.prot_lengths[pi].item()
            item = {
                'chem_tokens_feat': self.chem_tensor[ci, :cl],
                'chem_adj': self.chem_adj_tensor[ci, :cl, :cl] if self.chem_adj_tensor is not None else None,
                'prot_residue_feat': self.prot_tensor[pi, :pl],
                'label': self.labels[idx],
            }
            if hasattr(self, 'prot_site_tensor') and self.prot_site_tensor is not None:
                item['prot_site_label'] = self.prot_site_tensor[pi, :pl]
            return item
        else:

            return {
                'tox_feat': self.tox_data[idx],
                'chemberta_feat': self.comp_data[idx],
                'esm2_feat': self.prot_data[idx],
                'label': self.labels[idx],
            }


def seq_collate_fn(batch):

    labels = torch.stack([b['label'] for b in batch])


    chem_feats = [b['chem_tokens_feat'] for b in batch]
    max_chem = max(f.size(0) for f in chem_feats)
    chem_dim = chem_feats[0].size(1)
    chem_padded = torch.zeros(len(batch), max_chem, chem_dim)
    chem_mask = torch.ones(len(batch), max_chem, dtype=torch.bool)
    for i, f in enumerate(chem_feats):
        chem_padded[i, :f.size(0)] = f
        chem_mask[i, :f.size(0)] = False

    chem_adj = None
    if batch[0].get('chem_adj') is not None:
        chem_adj = torch.zeros(len(batch), max_chem, max_chem)
        for i, b in enumerate(batch):
            adj = b['chem_adj']
            chem_adj[i, :adj.size(0), :adj.size(1)] = adj


    prot_feats = [b['prot_residue_feat'] for b in batch]
    max_prot = max(f.size(0) for f in prot_feats)
    prot_dim = prot_feats[0].size(1)
    prot_padded = torch.zeros(len(batch), max_prot, prot_dim)
    prot_mask = torch.ones(len(batch), max_prot, dtype=torch.bool)
    for i, f in enumerate(prot_feats):
        prot_padded[i, :f.size(0)] = f
        prot_mask[i, :f.size(0)] = False

    result = {
        'chem_tokens_feat': chem_padded,
        'chem_mask': chem_mask,
        'prot_residue_feat': prot_padded,
        'prot_mask': prot_mask,
        'label': labels,
    }
    if chem_adj is not None:
        result['chem_adj'] = chem_adj
    if batch[0].get('prot_site_label') is not None:
        site_padded = torch.zeros(len(batch), max_prot, dtype=torch.float32)
        for i, b in enumerate(batch):
            site = b['prot_site_label'].float()
            site_padded[i, :site.size(0)] = site
        result['prot_site_label'] = site_padded
    return result


def _parse_position_list(value):

    if isinstance(value, list):
        return [int(x) for x in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, (list, tuple, set)):
        return []
    return [int(x) for x in parsed]


def attach_site_label_tensors(shared_tensors, site_annotation_path, max_protein_length,
                              position_columns=None):






    if not site_annotation_path:
        return shared_tensors
    if not os.path.exists(site_annotation_path):
        raise FileNotFoundError(f"Site annotation CSV not found: {site_annotation_path}")

    position_columns = position_columns or [
        'filtered_binding_positions_truncated',
        'actsite_positions_truncated',
        'pdbe_strict_positions_truncated',
    ]
    ann = pd.read_csv(site_annotation_path, low_memory=False)
    if 'Entry' not in ann.columns:
        raise ValueError(f"{site_annotation_path} missing Entry column")
    ann = ann.set_index('Entry')

    prot_id2idx = shared_tensors['prot_id2idx']
    n_prots = len(prot_id2idx)
    site_tensor = torch.zeros(n_prots, max_protein_length, dtype=torch.float32)
    n_with_any = 0
    n_positions = 0

    for prot_id, prot_idx in prot_id2idx.items():
        if prot_id not in ann.index:
            continue
        row = ann.loc[prot_id]
        positions = set()
        for col in position_columns:
            if col in ann.columns:
                positions.update(_parse_position_list(row[col]))
        positions = {p for p in positions if 1 <= p <= max_protein_length}
        if not positions:
            continue
        idx = torch.tensor([p - 1 for p in sorted(positions)], dtype=torch.long)
        site_tensor[prot_idx, idx] = 1.0
        n_with_any += 1
        n_positions += len(positions)

    shared_tensors = dict(shared_tensors)
    shared_tensors['prot_site_tensor'] = site_tensor
    shared_tensors['prot_site_available'] = site_tensor.sum(dim=1) > 0
    logger.info(
        f"Attached strict site labels: {n_with_any}/{n_prots} proteins, "
        f"{n_positions} residue labels from {site_annotation_path}"
    )
    return shared_tensors


def build_or_load_entity_tensors(cache_path, comp_feats, prot_feats, max_chem_tokens,
                                 max_protein_length, build_chem_adj=False,
                                 chem_smiles_map=None, rebuild=False):

    if cache_path and os.path.exists(cache_path) and not rebuild:
        logger.info(f"Loading entity tensor cache from {cache_path}")
        try:
            return torch.load(cache_path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(cache_path, map_location='cpu')

    shared = FusionDataset.build_entity_tensors(
        comp_feats, prot_feats,
        max_chem_tokens=max_chem_tokens,
        max_protein_length=max_protein_length,
        build_chem_adj=build_chem_adj,
        chem_smiles_map=chem_smiles_map)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        logger.info(f"Saving entity tensor cache to {cache_path}")
        torch.save(shared, cache_path)
    return shared


def default_entity_cache_path(base_dir, prefix, split, max_chem_tokens, max_protein_length,
                              use_molecular_gcn=False, extra_tag=''):

    tag = f"{prefix}_{split}_chem{max_chem_tokens}_prot{max_protein_length}"
    tag += "_gcn" if use_molecular_gcn else "_nogcn"
    if extra_tag:
        digest = hashlib.sha1(extra_tag.encode('utf-8')).hexdigest()[:10]
        tag += f"_{digest}"
    return os.path.join(base_dir, 'data', 'pretrained_features', f'{tag}_entity_tensors.pt')


def make_index_collate_fn(shared_tensors):





    chem_tensor = shared_tensors['chem_tensor']
    chem_lengths = shared_tensors['chem_lengths']
    prot_tensor = shared_tensors['prot_tensor']
    prot_lengths = shared_tensors['prot_lengths']
    chem_adj_tensor = shared_tensors.get('chem_adj_tensor')
    prot_site_tensor = shared_tensors.get('prot_site_tensor')
    max_chem = chem_tensor.size(1)
    max_prot = prot_tensor.size(1)
    chem_pos = torch.arange(max_chem)
    prot_pos = torch.arange(max_prot)

    def _collate(batch):
        chem_idx = torch.stack([b['chem_idx'] for b in batch]).long()
        prot_idx = torch.stack([b['prot_idx'] for b in batch]).long()
        labels = torch.stack([b['label'] for b in batch])

        chem_len = chem_lengths[chem_idx]
        prot_len = prot_lengths[prot_idx]
        result = {
            'chem_tokens_feat': chem_tensor[chem_idx],
            'chem_mask': chem_pos.unsqueeze(0) >= chem_len.unsqueeze(1),
            'prot_residue_feat': prot_tensor[prot_idx],
            'prot_mask': prot_pos.unsqueeze(0) >= prot_len.unsqueeze(1),
            'label': labels,
        }
        if chem_adj_tensor is not None:
            result['chem_adj'] = chem_adj_tensor[chem_idx]
        if prot_site_tensor is not None:
            result['prot_site_label'] = prot_site_tensor[prot_idx]
        return result

    return _collate




def _forward_batch(model, batch, device):

    if 'chem_tokens_feat' in batch:

        logits, attn, chem_emb, prot_emb = model(
            chem_tokens_feat=batch['chem_tokens_feat'].to(device),
            chem_mask=batch['chem_mask'].to(device),
            chem_adj=batch['chem_adj'].to(device) if 'chem_adj' in batch else None,
            prot_residue_feat=batch['prot_residue_feat'].to(device),
            prot_mask=batch['prot_mask'].to(device)
        )
    else:

        logits, attn, chem_emb, prot_emb = model(
            tox_feat=batch['tox_feat'].to(device),
            chemberta_feat=batch['chemberta_feat'].to(device),
            esm2_feat=batch['esm2_feat'].to(device)
        )
    return logits, attn, chem_emb, prot_emb


def site_attention_loss(attention, chem_mask, prot_mask, site_labels, labels=None,
                        positive_only=True, eps=1e-8):






    if attention is None or site_labels is None:
        return torch.tensor(0.0, device=site_labels.device if site_labels is not None else 'cpu')
    if attention.dim() == 4:
        attention = attention.mean(dim=1)

    device = attention.device
    site_labels = site_labels.to(device).float()
    if chem_mask is None:
        chem_valid = torch.ones(attention.shape[:2], dtype=torch.bool, device=device)
    else:
        chem_valid = ~chem_mask.to(device)
    if prot_mask is None:
        prot_valid = torch.ones(attention.size(0), attention.size(-1), dtype=torch.bool, device=device)
    else:
        prot_valid = ~prot_mask.to(device)

    chem_weights = chem_valid.float()
    residue_mass = (attention * chem_weights.unsqueeze(-1)).sum(dim=1)
    residue_mass = residue_mass / chem_weights.sum(dim=1, keepdim=True).clamp(min=1.0)
    residue_mass = residue_mass.masked_fill(~prot_valid, 0.0)
    residue_mass = residue_mass / residue_mass.sum(dim=-1, keepdim=True).clamp(min=eps)

    target = site_labels.masked_fill(~prot_valid, 0.0)
    has_site = target.sum(dim=-1) > 0
    if positive_only and labels is not None:
        has_site = has_site & (labels.to(device) > 0.5)
    if has_site.sum() == 0:
        return torch.tensor(0.0, device=device)

    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1.0)
    per_sample = -(target * torch.log(residue_mass.clamp(min=eps))).sum(dim=-1)
    return per_sample[has_site].mean()


def _add_feature_noise(batch, comp_noise, prot_noise):

    if 'chem_tokens_feat' in batch:

        if comp_noise > 0:
            mask = ~batch['chem_mask']
            noise = torch.randn_like(batch['chem_tokens_feat']) * comp_noise
            batch['chem_tokens_feat'] = batch['chem_tokens_feat'] + noise * mask.unsqueeze(-1)
        if prot_noise > 0:
            mask = ~batch['prot_mask']
            noise = torch.randn_like(batch['prot_residue_feat']) * prot_noise
            batch['prot_residue_feat'] = batch['prot_residue_feat'] + noise * mask.unsqueeze(-1)
    else:

        if comp_noise > 0:
            batch['chemberta_feat'] = batch['chemberta_feat'] + torch.randn_like(batch['chemberta_feat']) * comp_noise
        if prot_noise > 0:
            batch['esm2_feat'] = batch['esm2_feat'] + torch.randn_like(batch['esm2_feat']) * prot_noise
    return batch


def train_epoch(model, loader, optimizer, criterion, device, scaler, log_interval=20,
                use_amp=True, comp_noise=0.0, prot_noise=0.0,
                label_smoothing=0.0, ema=None,
                contrastive_loss_fn=None, contrastive_weight=0.0,
                site_loss_weight=0.0, site_positive_only=True):
    model.train()
    total_loss = 0
    total_site_loss = 0.0
    n_site_batches = 0
    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        labels = batch['label'].to(device)
        hard_labels = labels.clone()


        if label_smoothing > 0:
            labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

        optimizer.zero_grad()


        if comp_noise > 0 or prot_noise > 0:
            batch = _add_feature_noise(batch, comp_noise, prot_noise)

        if use_amp:
            with autocast('cuda'):
                logits, attn, chem_emb, prot_emb = _forward_batch(model, batch, device)
                loss = criterion(logits, labels)
                site_loss_value = None
                if contrastive_loss_fn is not None and contrastive_weight > 0 and chem_emb is not None:
                    cl_loss = contrastive_loss_fn(chem_emb.float(), prot_emb.float(), hard_labels)
                    loss = loss + contrastive_weight * cl_loss
                if site_loss_weight > 0 and 'prot_site_label' in batch:
                    site_loss = site_attention_loss(
                        attn, batch.get('chem_mask'), batch.get('prot_mask'),
                        batch['prot_site_label'], hard_labels,
                        positive_only=site_positive_only
                    )
                    loss = loss + site_loss_weight * site_loss
                    site_loss_value = float(site_loss.detach().cpu())
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, attn, chem_emb, prot_emb = _forward_batch(model, batch, device)
            loss = criterion(logits, labels)
            site_loss_value = None
            if contrastive_loss_fn is not None and contrastive_weight > 0 and chem_emb is not None:
                cl_loss = contrastive_loss_fn(chem_emb, prot_emb, hard_labels)
                loss = loss + contrastive_weight * cl_loss
            if site_loss_weight > 0 and 'prot_site_label' in batch:
                site_loss = site_attention_loss(
                    attn, batch.get('chem_mask'), batch.get('prot_mask'),
                    batch['prot_site_label'], hard_labels,
                    positive_only=site_positive_only
                )
                loss = loss + site_loss_weight * site_loss
                site_loss_value = float(site_loss.detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()


        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        if site_loss_value is not None:
            total_site_loss += site_loss_value
            n_site_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            pct = 100.0 * (batch_idx + 1) / n_batches
            avg = total_loss / (batch_idx + 1)
            msg = (f"  Batch {batch_idx+1}/{n_batches} ({pct:.1f}%) | "
                   f"Loss: {loss.item():.4f} | Avg: {avg:.4f}")
            if site_loss_value is not None:
                site_avg = total_site_loss / max(1, n_site_batches)
                msg += f" | SiteLoss: {site_loss_value:.4f} | SiteAvg: {site_avg:.4f}"
            logger.info(msg)

    return total_loss / n_batches


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            labels = batch['label'].to(device)

            logits, _, _, _ = _forward_batch(model, batch, device)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)

    metrics = {
        'loss': total_loss / len(loader),
        'auroc': roc_auc_score(all_labels, all_probs),
        'auprc': average_precision_score(all_labels, all_probs),
        'f1': f1_score(all_labels, preds),
        'acc': accuracy_score(all_labels, preds),
        'mcc': matthews_corrcoef(all_labels, preds),
        'labels': all_labels,
        'probs': all_probs,
    }

    return metrics


def analyze_gate_weights(model, loader, device, n_samples=1000):

    model.eval()
    all_gates = []

    with torch.no_grad():
        count = 0
        for batch in loader:
            tox_feat = batch['tox_feat'].to(device)
            chemberta_feat = batch['chemberta_feat'].to(device)
            g = model.get_gate_weights(tox_feat, chemberta_feat)
            all_gates.extend(g.cpu().numpy().flatten())
            count += len(tox_feat)
            if count >= n_samples:
                break

    gates = np.array(all_gates[:n_samples])
    logger.info(f"Gate weights (domain_preference): "
                f"mean={gates.mean():.4f}, std={gates.std():.4f}, "
                f"min={gates.min():.4f}, max={gates.max():.4f}")
    logger.info(f"  >0.5 (prefer domain): {(gates > 0.5).mean()*100:.1f}%")
    logger.info(f"  <0.5 (prefer semantic): {(gates < 0.5).mean()*100:.1f}%")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--split', type=str, default='cold', choices=['warm', 'cold'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42,
                        help='模型初始化和训练随机种子')
    parser.add_argument('--data_seed', type=int, default=42,
                        help='数据划分固定种子 (集成训练时保持不变)')
    parser.add_argument('--split_dir', type=str,
                        default=str(REPOSITORY_ROOT / 'data' / 'binding'),
                        help='统一预建 split CSV 目录')
    parser.add_argument('--gate_min', type=float, default=0.0,
                        help='门控最小值，约束领域特征最低占比')
    parser.add_argument('--gate_max', type=float, default=1.0,
                        help='门控最大值，约束领域特征最高占比')
    parser.add_argument('--chem_feat', type=str, default='chemberta',
                        choices=['chemberta', 'bermol'],
                        help='化学品语义特征来源')
    parser.add_argument('--no_toxprint', action='store_true',
                        help='不使用ToxPrint领域特征，仅用语义特征')
    parser.add_argument('--use_residue_attention', action='store_true',
                        help='Step 2: 预提取序列特征 + BilinearAttention')
    parser.add_argument('--interaction_cnn', action='store_true',
                        help='在多头 token-residue 注意力矩阵上启用 2D CNN 交互增强')
    parser.add_argument('--interaction_cnn_channels', type=int, default=16,
                        help='InteractionCNN 第一层通道数，仅 --interaction_cnn 启用时生效')
    parser.add_argument('--molecular_gcn', action='store_true',
                        help='在 BerMol token 特征上启用 RDKit 分子图 GCN 拓扑增强')
    parser.add_argument('--gcn_layers', type=int, default=2,
                        help='Molecular GCN 层数，仅 --molecular_gcn 启用时生效')
    parser.add_argument('--max_protein_length', type=int, default=768,
                        help='蛋白质残基最大长度 (768 覆盖 74.9%%, 原 ESM2 limit=1022)')
    parser.add_argument('--max_chem_tokens', type=int, default=64,
                        help='化学品 token 最大长度 (64 覆盖 94.2%%)')
    parser.add_argument('--comp_noise', type=float, default=0.0,
                        help='训练时化学品特征高斯噪声标准差')
    parser.add_argument('--prot_noise', type=float, default=0.0,
                        help='训练时蛋白质特征高斯噪声标准差')
    parser.add_argument('--warmup_ratio', type=float, default=0.0,
                        help='学习率线性预热占总epoch比例 (0~1)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing 系数 (Cold 推荐 0.05, Warm 推荐 0.02)')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA 衰减系数 (推荐 0.999, 0=禁用)')
    parser.add_argument('--contrastive_weight', type=float, default=0.0,
                        help='对比学习损失权重 (推荐 0.1, 0=禁用)')
    parser.add_argument('--site_loss_weight', type=float, default=0.0,
                        help='Binding attention 位点弱监督损失权重，0=禁用')
    parser.add_argument('--site_annotation_path', type=str,
                        default=None,
                        help='audit_attention_site_coverage.py 生成的蛋白位点注释 CSV')
    parser.add_argument('--site_supervise_all', action='store_true',
                        help='默认只监督正样本；设置后正负样本都加 site attention loss')
    parser.add_argument('--init_from', type=str, default=None,
                        help='从指定 checkpoint 部分加载 shape 匹配权重 (warm-start)')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                        help='warm-start 后先冻结加载的旧主干多少个 epoch')
    parser.add_argument('--exp_suffix', type=str, default='',
                        help='实验后缀，用于 checkpoint 命名')
    parser.add_argument('--entity_cache', type=str, default=None,
                        help='Step 2 entity tensor cache path. Default: data/pretrained_features/*.pt')
    parser.add_argument('--rebuild_entity_cache', action='store_true',
                        help='Force rebuild Step 2 entity tensor cache')
    parser.add_argument('--prepare_data_only', action='store_true',
                        help='Only build/load cached entity tensors and exit before training')
    parser.add_argument('--fast_index_collate', action='store_true',
                        help='Use index-based fixed-length collate to reduce CPU batch preparation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers. For large shared tensors, 0 usually avoids memory blow-up')
    parser.add_argument('--torch_threads', type=int, default=4,
                        help='Limit PyTorch intra-op CPU threads per training process')
    args = parser.parse_args()
    if args.site_loss_weight > 0 and not args.use_residue_attention:
        raise ValueError("--site_loss_weight requires --use_residue_attention")


    device = torch.device('cuda:0')


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(2, args.torch_threads)))
        logger.info(f"PyTorch CPU threads: intra={torch.get_num_threads()}, "
                    f"interop={torch.get_num_interop_threads()}")

    logger.info("=" * 60)
    logger.info("ToxCPI Fusion Model Training (Step 1)")
    logger.info("=" * 60)
    logger.info(f"Split: {args.split}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    logger.info(f"Weight Decay: {args.weight_decay}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Hidden Dim: {args.hidden_dim}")
    logger.info(f"Gate Range: [{args.gate_min}, {args.gate_max}]")
    logger.info(f"Chem Feature: {args.chem_feat}")
    logger.info(f"Use ToxPrint: {not args.no_toxprint}")
    logger.info(f"Residue Attention: {args.use_residue_attention}")
    logger.info(f"Interaction CNN: {args.interaction_cnn}")
    if args.interaction_cnn:
        logger.info(f"Interaction CNN Channels: {args.interaction_cnn_channels}")
    logger.info(f"Molecular GCN: {args.molecular_gcn}")
    if args.molecular_gcn:
        logger.info(f"GCN Layers: {args.gcn_layers}")
    if args.use_residue_attention:
        logger.info(f"Max Protein Length: {args.max_protein_length}")
        logger.info(f"Max Chem Tokens: {args.max_chem_tokens}")
    logger.info(f"Comp Noise: {args.comp_noise}")
    logger.info(f"Prot Noise: {args.prot_noise}")
    logger.info(f"Warmup Ratio: {args.warmup_ratio}")
    logger.info(f"Max Epochs: {args.num_epochs}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"Label Smoothing: {args.label_smoothing}")
    logger.info(f"EMA Decay: {args.ema_decay}")
    logger.info(f"Contrastive Weight: {args.contrastive_weight}")
    logger.info(f"Site Loss Weight: {args.site_loss_weight}")
    if args.site_loss_weight > 0:
        logger.info(f"Site Annotation Path: {args.site_annotation_path}")
        logger.info(f"Site Positive Only: {not args.site_supervise_all}")
    logger.info(f"Data Seed: {args.data_seed}")
    logger.info(f"Split Dir: {args.split_dir}")
    logger.info(f"Fast Index Collate: {args.fast_index_collate}")
    logger.info(f"DataLoader Workers: {args.num_workers}")
    logger.info("=" * 60)


    base_dir = str(RUNTIME_ROOT)
    feat_dir = str(FEATURE_DIR)

    logger.info("Loading pretrained features...")
    if args.use_residue_attention:

        with open(os.path.join(feat_dir, 'compound_features_bermol_tokens.pkl'), 'rb') as f:
            comp_feats = pickle.load(f)
        with open(os.path.join(feat_dir, 'protein_features_esm2_residues.pkl'), 'rb') as f:
            prot_feats = pickle.load(f)
        sample_chem = next(iter(comp_feats.values()))
        sample_prot = next(iter(prot_feats.values()))
        logger.info(f"BerMol token features: {len(comp_feats)} chemicals, "
                    f"sample shape {sample_chem.shape}")
        logger.info(f"ESM2 residue features: {len(prot_feats)} proteins, "
                    f"sample shape {sample_prot.shape}")
    else:

        if args.chem_feat == 'bermol':
            comp_feat_file = 'compound_features_bermol.pkl'
        else:
            comp_feat_file = 'compound_features.pkl'
        with open(os.path.join(feat_dir, comp_feat_file), 'rb') as f:
            comp_feats = pickle.load(f)
        with open(os.path.join(feat_dir, 'protein_features.pkl'), 'rb') as f:
            prot_feats = pickle.load(f)
        logger.info(f"{args.chem_feat} features: {len(comp_feats)} x "
                    f"{next(iter(comp_feats.values())).shape[0]}")
        logger.info(f"ESM2 features: {len(prot_feats)} x "
                    f"{next(iter(prot_feats.values())).shape[0]}")


    data_path = str(REPOSITORY_ROOT / 'data' / 'binding' /
                    'combined_balanced_n72045_with_seq.csv')
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)

    logger.info("Canonicalizing SMILES...")
    df['smiles_clean'] = df['smiles'].apply(canonicalize_smiles)

    chem_col = 'chemical_name'
    prot_col = 'uniprot_id'
    label_col = 'label'
    chem_smiles_map = (
        df.dropna(subset=[chem_col, 'smiles_clean'])
          .drop_duplicates(chem_col)
          .set_index(chem_col)['smiles_clean']
          .astype(str)
          .to_dict()
    )

    if args.molecular_gcn:
        effect_csvs = list((REPOSITORY_ROOT / 'data' / 'activation').glob('effect_*.csv'))
        effect_csvs += list((REPOSITORY_ROOT / 'data' / 'inhibition').glob('effect_*.csv'))
        for effect_csv in sorted(map(str, effect_csvs)):
            try:
                edf = pd.read_csv(effect_csv, usecols=['chemical_name', 'smiles'], low_memory=False)
                edf = edf.dropna(subset=['chemical_name', 'smiles']).drop_duplicates('chemical_name')
                for row in edf.itertuples(index=False):
                    if row.chemical_name not in chem_smiles_map:
                        chem_smiles_map[row.chemical_name] = str(row.smiles)
            except Exception:
                pass
        logger.info(f"SMILES map: {len(chem_smiles_map)} chemicals (binding + effect)")

    logger.info(f"Loading prebuilt {args.split} splits from {args.split_dir}")
    train_df, val_df, test_df = load_prebuilt_splits(args.split_dir, args.split, logger)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


    logger.info("Creating datasets...")
    if args.use_residue_attention:

        cache_path = args.entity_cache or default_entity_cache_path(
            base_dir, 'fusion', args.split, args.max_chem_tokens,
            args.max_protein_length, args.molecular_gcn,
            extra_tag=f"data_seed={args.data_seed}")
        shared = build_or_load_entity_tensors(
            cache_path, comp_feats, prot_feats,
            max_chem_tokens=args.max_chem_tokens,
            max_protein_length=args.max_protein_length,
            build_chem_adj=args.molecular_gcn,
            chem_smiles_map=chem_smiles_map,
            rebuild=args.rebuild_entity_cache)
        if args.site_loss_weight > 0:
            shared = attach_site_label_tensors(
                shared,
                args.site_annotation_path,
                max_protein_length=args.max_protein_length
            )
        if args.prepare_data_only:
            logger.info("prepare_data_only set; exiting after entity tensor cache is ready.")
            return
        ds_kwargs = dict(use_residue_attention=True, shared_tensors=shared,
                         use_molecular_gcn=args.molecular_gcn,
                         return_indices=args.fast_index_collate)
        tox_cache = None
    else:
        ds_kwargs = dict(use_residue_attention=False)
        cache_dir = str(RUNTIME_ROOT / 'pretrained_features')
        split_cache_tag = hashlib.sha1(os.path.abspath(args.split_dir).encode('utf-8')).hexdigest()[:10]
        tox_cache = os.path.join(cache_dir, f'tox_features_cache_{args.split}_{split_cache_tag}.npz')

    train_dataset = FusionDataset(train_df, comp_feats, prot_feats,
                                   chem_col, prot_col, label_col,
                                   tox_cache_path=tox_cache, **ds_kwargs)
    val_dataset = FusionDataset(val_df, comp_feats, prot_feats,
                                 chem_col, prot_col, label_col,
                                 tox_cache_path=None, **ds_kwargs)
    test_dataset = FusionDataset(test_df, comp_feats, prot_feats,
                                  chem_col, prot_col, label_col,
                                  tox_cache_path=None, **ds_kwargs)


    if args.use_residue_attention and args.fast_index_collate:
        collate = make_index_collate_fn(shared)
    else:
        collate = seq_collate_fn if args.use_residue_attention else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers, pin_memory=True,
                               collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True,
                             collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate)


    if args.use_residue_attention:

        model = BindingModelFusion(
            chemberta_dim=768,
            esm2_dim=1280,
            hidden_dim=args.hidden_dim,
            interaction_dim=args.hidden_dim * 2,
            dropout=args.dropout,
            use_residue_attention=True,
            use_toxprint=False,
            use_interaction_cnn=args.interaction_cnn,
            interaction_cnn_channels=args.interaction_cnn_channels,
            use_molecular_gcn=args.molecular_gcn,
            gcn_layers=args.gcn_layers
        ).to(device)
    else:
        model = BindingModelFusion(
            tox_dim=812,
            chemberta_dim=768,
            esm2_dim=1280,
            hidden_dim=args.hidden_dim,
            interaction_dim=args.hidden_dim * 2,
            dropout=args.dropout,
            use_residue_attention=False,
            gate_min=args.gate_min,
            gate_max=args.gate_max,
            use_toxprint=not args.no_toxprint
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")


    loaded_param_names = set()
    if args.init_from:
        loaded_param_names = load_partial_checkpoint(model, args.init_from, device)
        if not loaded_param_names:
            raise RuntimeError("Warm-start loaded zero trainable parameters.")
        if args.freeze_backbone_epochs > 0:
            set_params_trainable(model, loaded_param_names, trainable=False)
            logger.info(f"Trainable params after freeze: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    warmup_epochs = max(1, int(args.num_epochs * args.warmup_ratio)) if args.warmup_ratio > 0 else 0
    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.num_epochs - warmup_epochs, eta_min=args.lr / 10
        )
        scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                                 milestones=[warmup_epochs])
        logger.info(f"Scheduler: LinearWarmup({warmup_epochs}ep) + CosineAnnealing({args.num_epochs - warmup_epochs}ep)")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.lr / 10
        )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler('cuda')


    best_val_auroc = 0
    no_improve = 0
    ckpt_suffix = ''
    if args.interaction_cnn:
        ckpt_suffix += '_icnn'
    if args.molecular_gcn:
        ckpt_suffix += '_gcn'
    if args.site_loss_weight > 0:
        ckpt_suffix += '_site'
    if args.exp_suffix:
        ckpt_suffix += f'_{args.exp_suffix.strip("_")}'
    ckpt_path = os.path.join(base_dir, 'checkpoints', f'fusion_{args.split}{ckpt_suffix}_best.pth')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    use_amp = not args.use_residue_attention
    if not use_amp:
        logger.info("AMP disabled for BilinearAttention (numerical stability)")


    ema = EMAModel(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        logger.info(f"EMA enabled: decay={args.ema_decay}")


    contrastive_loss_fn = ContrastiveLoss(temperature=0.07) if args.contrastive_weight > 0 else None
    if contrastive_loss_fn:
        logger.info(f"Contrastive learning enabled: weight={args.contrastive_weight}, temp=0.07")

    for epoch in range(args.num_epochs):
        if loaded_param_names and args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs:
            set_params_trainable(model, loaded_param_names, trainable=True)
            logger.info(f"Trainable params after unfreeze: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                  device, scaler, use_amp=use_amp,
                                  comp_noise=args.comp_noise, prot_noise=args.prot_noise,
                                  label_smoothing=args.label_smoothing, ema=ema,
                                  contrastive_loss_fn=contrastive_loss_fn,
                                  contrastive_weight=args.contrastive_weight,
                                  site_loss_weight=args.site_loss_weight,
                                  site_positive_only=not args.site_supervise_all)


        if ema:
            ema.apply_shadow(model)
        val_metrics = evaluate(model, val_loader, criterion, device)
        if ema:
            ema.restore(model)

        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUROC: {val_metrics['auroc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            no_improve = 0

            if ema:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), ckpt_path)
                ema.restore(model)
            else:
                torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  -> New best model! AUROC={best_val_auroc:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break


    logger.info("=" * 60)
    logger.info("Loading best model for testing...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"AUROC:    {test_metrics['auroc']:.4f}")
    logger.info(f"AUPRC:    {test_metrics['auprc']:.4f}")
    logger.info(f"F1:       {test_metrics['f1']:.4f}")
    logger.info(f"Accuracy: {test_metrics['acc']:.4f}")
    logger.info(f"MCC:      {test_metrics['mcc']:.4f}")
    logger.info("=" * 60)


    if not args.no_toxprint and not args.use_residue_attention:
        logger.info("Analyzing gate weights...")
        analyze_gate_weights(model, test_loader, device)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
