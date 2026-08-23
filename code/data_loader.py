import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
import logging
from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from config import (
    BINDING_DATA_PATH, EFFECT_DATA_PATH, NO_EFFECT_DATA_PATH,
    model_config, train_config, DEVICE
)
from utils import (
    smiles_to_mol, canonicalize_smiles,
    get_toxprint_features, get_benigni_bossa_features, get_edc_features
)
from sample_weighting import SampleWeightCalculator

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class BaseCPIDataset(Dataset):
    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"
    AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

    def __init__(
        self,
        smiles_list: List[str],
        sequences_list: List[str],
        max_smiles_length: int = 512,
        max_protein_length: int = 1024,
        precompute_features: bool = True,
        cache_path: Optional[str] = None
    ):
        self.smiles_list = smiles_list
        self.sequences_list = sequences_list
        self.max_smiles_length = max_smiles_length
        self.max_protein_length = max_protein_length

        self.toxprint_features = None
        self.benigni_features = None
        self.edc_features = None

        self.protein_tokens = None
        self.protein_masks = None

        if precompute_features:
            self._precompute_tox_features(cache_path)
            self._precompute_protein_tokens(cache_path)

    def _precompute_tox_features(self, cache_path: str = None):
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached toxicity features from {cache_path}")
            try:
                cached = np.load(cache_path)
                self.toxprint_features = cached['toxprint']
                self.benigni_features = cached['benigni']
                self.edc_features = cached['edc']

                if len(self.toxprint_features) == len(self.smiles_list):
                    logger.info(f"Loaded cached features for {len(self.smiles_list)} molecules")
                    return
                else:
                    logger.info("Cache size mismatch, recomputing...")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, recomputing...")

        logger.info(f"Precomputing toxicity features for {len(self.smiles_list)} molecules...")

        toxprint_list = []
        benigni_list = []
        edc_list = []

        from tqdm import tqdm
        for smiles in tqdm(self.smiles_list, desc="Computing tox features", file=None, disable=True):
            mol = smiles_to_mol(smiles)
            toxprint_list.append(get_toxprint_features(mol))
            benigni_list.append(get_benigni_bossa_features(mol))
            edc_list.append(get_edc_features(mol))

        self.toxprint_features = np.array(toxprint_list, dtype=np.float32)
        self.benigni_features = np.array(benigni_list, dtype=np.float32)
        self.edc_features = np.array(edc_list, dtype=np.float32)

        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.savez(cache_path,
                         toxprint=self.toxprint_features,
                         benigni=self.benigni_features,
                         edc=self.edc_features)
                logger.info(f"Cached features saved to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def _precompute_protein_tokens(self, cache_path: str = None):
        token_cache_path = cache_path.replace('.npz', '_tokens.npz') if cache_path else None
        if token_cache_path and os.path.exists(token_cache_path):
            logger.info(f"Loading cached protein tokens from {token_cache_path}")
            try:
                cached = np.load(token_cache_path)
                self.protein_tokens = cached['tokens']
                self.protein_masks = cached['masks']

                if len(self.protein_tokens) == len(self.sequences_list):
                    logger.info(f"Loaded cached tokens for {len(self.sequences_list)} proteins")
                    return
                else:
                    logger.info("Cache size mismatch, recomputing...")
            except Exception as e:
                logger.warning(f"Failed to load token cache: {e}, recomputing...")

        logger.info(f"Pre-computing protein tokens for {len(self.sequences_list)} sequences...")

        tokens_list = []
        masks_list = []

        for seq in self.sequences_list:
            if len(seq) > self.max_protein_length:
                seq = seq[:self.max_protein_length]

            tokens = np.zeros(self.max_protein_length, dtype=np.int64)
            mask = np.ones(self.max_protein_length, dtype=np.bool_)

            for j, aa in enumerate(seq):
                tokens[j] = self.AA_TO_IDX.get(aa, self.AA_TO_IDX['X']) + 1
                mask[j] = False

            tokens_list.append(tokens)
            masks_list.append(mask)

        self.protein_tokens = np.array(tokens_list, dtype=np.int64)
        self.protein_masks = np.array(masks_list, dtype=np.bool_)

        if token_cache_path:
            try:
                np.savez(token_cache_path,
                         tokens=self.protein_tokens,
                         masks=self.protein_masks)
                logger.info(f"Cached protein tokens saved to {token_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save token cache: {e}")

    def __len__(self):
        return len(self.smiles_list)

    def _get_base_item(self, idx: int) -> Dict:
        smiles = self.smiles_list[idx]
        sequence = self.sequences_list[idx]

        if len(sequence) > self.max_protein_length:
            sequence = sequence[:self.max_protein_length]

        if self.toxprint_features is not None:
            toxprint = self.toxprint_features[idx]
            benigni = self.benigni_features[idx]
            edc = self.edc_features[idx]
        else:
            mol = smiles_to_mol(smiles)
            toxprint = get_toxprint_features(mol)
            benigni = get_benigni_bossa_features(mol)
            edc = get_edc_features(mol)

        result = {
            'smiles': smiles,
            'sequence': sequence,
            'toxprint': torch.tensor(toxprint, dtype=torch.float32),
            'benigni': torch.tensor(benigni, dtype=torch.float32),
            'edc': torch.tensor(edc, dtype=torch.float32),
        }

        if self.protein_tokens is not None:
            result['protein_tokens'] = torch.tensor(self.protein_tokens[idx], dtype=torch.long)
            result['protein_mask'] = torch.tensor(self.protein_masks[idx], dtype=torch.bool)

        return result


class BindingDataset(BaseCPIDataset):
    def __init__(
        self,
        smiles_list: List[str],
        sequences_list: List[str],
        binding_labels: np.ndarray,
        **kwargs
    ):
        super().__init__(smiles_list, sequences_list, **kwargs)
        self.binding_labels = binding_labels

    def __getitem__(self, idx: int) -> Dict:
        sample = self._get_base_item(idx)
        sample['label'] = torch.tensor(self.binding_labels[idx], dtype=torch.float32)
        return sample


class EffectDataset(BaseCPIDataset):
    def __init__(
        self,
        smiles_list: List[str],
        sequences_list: List[str],
        has_effect_labels: np.ndarray,
        effect_type_labels: np.ndarray,
        **kwargs
    ):
        super().__init__(smiles_list, sequences_list, **kwargs)
        self.has_effect_labels = has_effect_labels
        self.effect_type_labels = effect_type_labels

    def __getitem__(self, idx: int) -> Dict:
        sample = self._get_base_item(idx)
        sample['has_effect'] = torch.tensor(self.has_effect_labels[idx], dtype=torch.float32)
        sample['effect_type'] = torch.tensor(self.effect_type_labels[idx], dtype=torch.float32)
        return sample


def load_binding_data(filepath: str = BINDING_DATA_PATH) -> pd.DataFrame:
    logger.info(f"Loading binding data from {filepath}")
    df = pd.read_csv(filepath)

    required_cols = ['smiles', 'protein_sequence', 'label']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df['smiles_clean'] = df['smiles'].apply(canonicalize_smiles)
    df = df.dropna(subset=['smiles_clean'])

    df['label'] = df['label'].astype(int)

    logger.info(f"Loaded {len(df)} binding samples")
    logger.info(f"  Unique chemicals: {df['smiles_clean'].nunique()}")
    logger.info(f"  Unique proteins: {df['uniprot_id'].nunique()}")
    logger.info(f"  Positive (binding=1): {(df['label'] == 1).sum()}")
    logger.info(f"  Negative (binding=0): {(df['label'] == 0).sum()}")

    return df


def load_effect_data(filepath: str = EFFECT_DATA_PATH) -> pd.DataFrame:
    logger.info(f"Loading effect data from {filepath}")
    df = pd.read_csv(filepath)

    df = df.dropna(subset=['smiles', 'sequence'])
    df['smiles_clean'] = df['smiles'].apply(canonicalize_smiles)
    df = df.dropna(subset=['smiles_clean'])

    def normalize_effect(effect):
        effect = str(effect).lower().strip()
        if effect in ['agonist', 'activator', 'activation', 'increases']:
            return 1
        elif effect in ['antagonist', 'inhibitor', 'inhibition', 'decreases']:
            return 0
        else:
            return None

    df['effect_label'] = df['effect_type'].apply(normalize_effect)
    df = df.dropna(subset=['effect_label'])
    df['effect_label'] = df['effect_label'].astype(int)

    logger.info(f"Loaded {len(df)} effect samples")
    logger.info(f"  Activation (agonist): {(df['effect_label'] == 1).sum()}")
    logger.info(f"  Inhibition (antagonist): {(df['effect_label'] == 0).sum()}")

    return df


def load_no_effect_data(filepath: str = NO_EFFECT_DATA_PATH) -> pd.DataFrame:
    logger.info(f"Loading no-effect data from {filepath}")
    df = pd.read_csv(filepath)

    df = df.rename(columns={'protein_sequence': 'sequence'})
    df = df.dropna(subset=['smiles', 'sequence'])
    df['smiles_clean'] = df['smiles'].apply(canonicalize_smiles)
    df = df.dropna(subset=['smiles_clean'])

    logger.info(f"Loaded {len(df)} no-effect samples")
    logger.info(f"  Unique proteins: {df['gene_symbol'].nunique()}")
    logger.info(f"  Unique chemicals: {df['smiles_clean'].nunique()}")

    return df


def sample_with_protein_coverage(
    df: pd.DataFrame,
    n_samples: int,
    protein_col: str = 'gene_symbol',
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)

    unique_proteins = df[protein_col].unique()
    n_proteins = len(unique_proteins)

    logger.info(f"Sampling {n_samples} samples with coverage of {n_proteins} proteins")

    if n_samples <= n_proteins:
        sampled_proteins = np.random.choice(unique_proteins, size=n_samples, replace=False)
        sampled_dfs = []
        for protein in sampled_proteins:
            protein_df = df[df[protein_col] == protein]
            sampled_dfs.append(protein_df.sample(n=1, random_state=seed))
        return pd.concat(sampled_dfs, ignore_index=True)

    sampled_dfs = []
    remaining_df = df.copy()

    for protein in unique_proteins:
        protein_df = remaining_df[remaining_df[protein_col] == protein]
        if len(protein_df) > 0:
            sampled = protein_df.sample(n=1, random_state=seed)
            sampled_dfs.append(sampled)

            remaining_df = remaining_df.drop(sampled.index)

    n_remaining = n_samples - len(sampled_dfs)

    if n_remaining > 0 and len(remaining_df) > 0:
        n_to_sample = min(n_remaining, len(remaining_df))
        additional_samples = remaining_df.sample(n=n_to_sample, random_state=seed)
        sampled_dfs.append(additional_samples)

    result_df = pd.concat(sampled_dfs, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"Sampled {len(result_df)} samples covering {result_df[protein_col].nunique()} proteins")

    return result_df


def create_effect_data_with_negatives(
    effect_df: pd.DataFrame,
    binding_df: pd.DataFrame = None,
    target_ratio: str = "1:1:1",
    seed: int = 42
) -> pd.DataFrame:
    logger.info("="*50)
    logger.info("Creating hierarchical effect dataset (combined sources)")
    logger.info(f"Target ratio (activation:inhibition:no_effect): {target_ratio}")
    logger.info("="*50)

    collected_no_effect_df = load_no_effect_data()
    collected_proteins = set(collected_no_effect_df['gene_symbol'].unique())
    logger.info(f"Collected no-effect data: {len(collected_no_effect_df)} samples, {len(collected_proteins)} proteins")

    if binding_df is None:
        binding_df = load_binding_data()

    effect_df = effect_df.copy()
    effect_df['has_effect'] = 1
    effect_df['pair_key'] = effect_df['smiles_clean'] + '|' + effect_df['gene_symbol']
    effect_pairs = set(effect_df['pair_key'].values)

    binding_positive = binding_df[binding_df['label'] == 1].copy()
    binding_positive['pair_key'] = binding_positive['smiles_clean'] + '|' + binding_positive['gene_symbol']
    no_effect_mask = ~binding_positive['pair_key'].isin(effect_pairs)
    binding_no_effect_df = binding_positive[no_effect_mask].copy()
    binding_no_effect_df = binding_no_effect_df.rename(columns={'protein_sequence': 'sequence'})

    binding_proteins = set(binding_no_effect_df['gene_symbol'].unique())
    logger.info(f"Binding-derived no-effect data: {len(binding_no_effect_df)} samples, {len(binding_proteins)} proteins")

    new_proteins = binding_proteins - collected_proteins
    logger.info(f"New proteins from binding data: {len(new_proteins)}")

    activation_df = effect_df[effect_df['effect_label'] == 1].copy()
    inhibition_df = effect_df[effect_df['effect_label'] == 0].copy()

    n_activation = len(activation_df)
    n_inhibition_total = len(inhibition_df)

    logger.info(f"\nOriginal effect data:")
    logger.info(f"  Activation samples: {n_activation}")
    logger.info(f"  Inhibition samples: {n_inhibition_total}")

    if target_ratio == "1:1:1":
        n_inhibition_target = n_activation
        n_no_effect_target = n_activation
    elif target_ratio == "1:1:2":
        n_inhibition_target = n_activation
        n_no_effect_target = n_activation * 2
    else:
        n_inhibition_target = n_activation
        n_no_effect_target = n_activation

    logger.info(f"\nTarget sampling:")
    logger.info(f"  Activation: {n_activation} (all)")
    logger.info(f"  Inhibition target: {n_inhibition_target}")
    logger.info(f"  No-effect target: {n_no_effect_target}")

    if n_inhibition_target < n_inhibition_total:
        logger.info(f"\nSampling inhibition data with protein coverage...")
        inhibition_sampled = sample_with_protein_coverage(
            inhibition_df,
            n_inhibition_target,
            protein_col='gene_symbol',
            seed=seed
        )
    else:
        inhibition_sampled = inhibition_df
        logger.info(f"Using all inhibition data")

    logger.info(f"\n--- Sampling no-effect data from combined sources ---")

    n_collected_proteins = len(collected_proteins)

    n_from_collected = min(
        max(n_collected_proteins, n_no_effect_target // 3),
        n_no_effect_target,
        len(collected_no_effect_df)
    )

    logger.info(f"Step 1: Sampling {n_from_collected} from collected data ({n_collected_proteins} proteins)")
    collected_sampled = sample_with_protein_coverage(
        collected_no_effect_df,
        n_from_collected,
        protein_col='gene_symbol',
        seed=seed
    )

    n_remaining = n_no_effect_target - len(collected_sampled)
    logger.info(f"Step 2: Need {n_remaining} more samples from binding data")

    if n_remaining > 0 and len(binding_no_effect_df) > 0:
        new_protein_df = binding_no_effect_df[binding_no_effect_df['gene_symbol'].isin(new_proteins)]
        old_protein_df = binding_no_effect_df[~binding_no_effect_df['gene_symbol'].isin(new_proteins)]

        logger.info(f"  Binding samples with new proteins: {len(new_protein_df)}")
        logger.info(f"  Binding samples with existing proteins: {len(old_protein_df)}")

        binding_sampled_parts = []

        if len(new_protein_df) > 0:
            n_from_new = min(n_remaining, len(new_protein_df))
            new_sampled = sample_with_protein_coverage(
                new_protein_df,
                n_from_new,
                protein_col='gene_symbol',
                seed=seed
            )
            binding_sampled_parts.append(new_sampled)
            n_remaining -= len(new_sampled)
            logger.info(f"  Sampled {len(new_sampled)} from new proteins, covering {new_sampled['gene_symbol'].nunique()} proteins")

        if n_remaining > 0 and len(old_protein_df) > 0:
            n_from_old = min(n_remaining, len(old_protein_df))
            old_sampled = old_protein_df.sample(n=n_from_old, random_state=seed)
            binding_sampled_parts.append(old_sampled)
            logger.info(f"  Sampled {len(old_sampled)} from existing proteins")

        if binding_sampled_parts:
            binding_sampled = pd.concat(binding_sampled_parts, ignore_index=True)
        else:
            binding_sampled = pd.DataFrame()
    else:
        binding_sampled = pd.DataFrame()

    no_effect_parts = [collected_sampled]
    if len(binding_sampled) > 0:
        no_effect_parts.append(binding_sampled)

    no_effect_combined = pd.concat(no_effect_parts, ignore_index=True)
    no_effect_combined = no_effect_combined.copy()
    no_effect_combined['has_effect'] = 0
    no_effect_combined['effect_label'] = 0

    logger.info(f"\nNo-effect data summary:")
    logger.info(f"  From collected: {len(collected_sampled)} ({collected_sampled['gene_symbol'].nunique()} proteins)")
    if len(binding_sampled) > 0:
        logger.info(f"  From binding: {len(binding_sampled)} ({binding_sampled['gene_symbol'].nunique()} proteins)")
    logger.info(f"  Total: {len(no_effect_combined)} ({no_effect_combined['gene_symbol'].nunique()} proteins)")

    logger.info(f"\n--- Final dataset composition ---")
    logger.info(f"  Activation: {n_activation}")
    logger.info(f"  Inhibition: {len(inhibition_sampled)}")
    logger.info(f"  No-effect: {len(no_effect_combined)}")

    activation_final = activation_df[['smiles_clean', 'gene_symbol', 'sequence', 'has_effect', 'effect_label']].copy()
    inhibition_final = inhibition_sampled[['smiles_clean', 'gene_symbol', 'sequence', 'has_effect', 'effect_label']].copy()
    no_effect_final = no_effect_combined[['smiles_clean', 'gene_symbol', 'sequence', 'has_effect', 'effect_label']].copy()

    combined_df = pd.concat([activation_final, inhibition_final, no_effect_final], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(f"\nCombined effect dataset: {len(combined_df)}")
    logger.info(f"  Has effect: {(combined_df['has_effect'] == 1).sum()}")
    logger.info(f"    - Activation: {((combined_df['has_effect']==1) & (combined_df['effect_label']==1)).sum()}")
    logger.info(f"    - Inhibition: {((combined_df['has_effect']==1) & (combined_df['effect_label']==0)).sum()}")
    logger.info(f"  No effect: {(combined_df['has_effect'] == 0).sum()}")

    logger.info(f"\nProtein coverage:")
    logger.info(f"  Activation unique proteins: {activation_final['gene_symbol'].nunique()}")
    logger.info(f"  Inhibition unique proteins: {inhibition_final['gene_symbol'].nunique()}")
    logger.info(f"  No-effect unique proteins: {no_effect_final['gene_symbol'].nunique()}")
    logger.info(f"  Total unique proteins: {combined_df['gene_symbol'].nunique()}")

    return combined_df


def cold_chemical_split(
    df: pd.DataFrame,
    smiles_col: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Performing Cold Chemical Split...")

    unique_smiles = df[smiles_col].unique()
    n_chemicals = len(unique_smiles)

    logger.info(f"Total unique chemicals: {n_chemicals}")

    np.random.seed(seed)
    np.random.shuffle(unique_smiles)

    n_train = int(n_chemicals * train_ratio)
    n_val = int(n_chemicals * val_ratio)

    train_chemicals = set(unique_smiles[:n_train])
    val_chemicals = set(unique_smiles[n_train:n_train + n_val])
    test_chemicals = set(unique_smiles[n_train + n_val:])

    train_df = df[df[smiles_col].isin(train_chemicals)]
    val_df = df[df[smiles_col].isin(val_chemicals)]
    test_df = df[df[smiles_col].isin(test_chemicals)]

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")

    return train_df, val_df, test_df


def random_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Performing Random Split...")

    train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed)
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=seed)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def domain_aware_split(
    df: pd.DataFrame,
    source_col: str = 'data_source',
    tox_to_valtest_ratio: float = 0.3,
    val_test_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)

    required_cols = [source_col, 'label', 'smiles_clean', 'uniprot_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    pos_df = df[df['label'] == 1].copy()
    neg_df = df[df['label'] == 0].copy()

    logger.info("="*60)
    logger.info("Source-Aware Data Split (Positive by Source, Negative Smart)")
    logger.info("="*60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"  Positive: {len(pos_df)} ({len(pos_df)/len(df)*100:.1f}%)")
    logger.info(f"  Negative: {len(neg_df)} ({len(neg_df)/len(df)*100:.1f}%)")

    pos_source_counts = pos_df[source_col].value_counts()
    logger.info(f"\nPositive samples by source:")
    for source, count in pos_source_counts.items():
        logger.info(f"  {source}: {count} ({count/len(pos_df)*100:.1f}%)")

    drug_sources = ['BindingDB']
    tox_sources = ['T3DB', 'ToxCast', 'APDB']

    pos_drug = pos_df[pos_df[source_col].isin(drug_sources)].copy()
    pos_tox = pos_df[pos_df[source_col].isin(tox_sources)].copy()

    logger.info(f"\nPositive samples分类:")
    logger.info(f"  药物正例 (BindingDB): {len(pos_drug)}")
    logger.info(f"  毒理正例 (T3DB+ToxCast+APDB): {len(pos_tox)}")

    for source in tox_sources:
        count = (pos_tox[source_col] == source).sum()
        logger.info(f"    - {source}: {count}")

    pos_drug_train = pos_drug

    n_tox_pos = len(pos_tox)
    n_tox_pos_valtest = int(n_tox_pos * tox_to_valtest_ratio)
    n_tox_pos_train = n_tox_pos - n_tox_pos_valtest
    n_tox_pos_val = int(n_tox_pos_valtest * val_test_ratio)
    n_tox_pos_test = n_tox_pos_valtest - n_tox_pos_val

    logger.info(f"\n毒理正例分配计划:")
    logger.info(f"  训练集: {n_tox_pos_train}")
    logger.info(f"  验证集: {n_tox_pos_val}")
    logger.info(f"  测试集: {n_tox_pos_test}")

    tox_pos_indices = np.random.permutation(len(pos_tox))
    pos_tox_train = pos_tox.iloc[tox_pos_indices[:n_tox_pos_train]].reset_index(drop=True)
    pos_tox_val = pos_tox.iloc[tox_pos_indices[n_tox_pos_train:n_tox_pos_train + n_tox_pos_val]].reset_index(drop=True)
    pos_tox_test = pos_tox.iloc[tox_pos_indices[n_tox_pos_train + n_tox_pos_val:]].reset_index(drop=True)

    pos_train = pd.concat([pos_drug_train, pos_tox_train], ignore_index=True)
    pos_val = pos_tox_val
    pos_test = pos_tox_test

    logger.info(f"\n正例划分结果:")
    logger.info(f"  训练集: {len(pos_train)} (药物:{len(pos_drug_train)}, 毒理:{len(pos_tox_train)})")
    logger.info(f"  验证集: {len(pos_val)} (全部毒理)")
    logger.info(f"  测试集: {len(pos_test)} (全部毒理)")

    logger.info("\n" + "="*60)
    logger.info("智能分配负例")
    logger.info("="*60)

    train_chem_counts = pos_train['smiles_clean'].value_counts()
    val_chem_counts = pos_val['smiles_clean'].value_counts()
    test_chem_counts = pos_test['smiles_clean'].value_counts()

    train_prot_counts = pos_train['uniprot_id'].value_counts()
    val_prot_counts = pos_val['uniprot_id'].value_counts()
    test_prot_counts = pos_test['uniprot_id'].value_counts()

    logger.info(f"\n正例中的化学品分布:")
    logger.info(f"  训练集: {len(train_chem_counts)} 种化学品")
    logger.info(f"  验证集: {len(val_chem_counts)} 种化学品")
    logger.info(f"  测试集: {len(test_chem_counts)} 种化学品")

    logger.info(f"\n正例中的蛋白质分布:")
    logger.info(f"  训练集: {len(train_prot_counts)} 种蛋白质")
    logger.info(f"  验证集: {len(val_prot_counts)} 种蛋白质")
    logger.info(f"  测试集: {len(test_prot_counts)} 种蛋白质")

    target_neg_train = len(pos_train)
    target_neg_val = len(pos_val)
    target_neg_test = len(pos_test)

    total_neg_needed = target_neg_train + target_neg_val + target_neg_test
    total_neg_available = len(neg_df)

    logger.info(f"\n目标负例分配:")
    logger.info(f"  训练集目标: {target_neg_train}")
    logger.info(f"  验证集目标: {target_neg_val}")
    logger.info(f"  测试集目标: {target_neg_test}")
    logger.info(f"  总需要: {total_neg_needed}, 总可用: {total_neg_available}")

    if total_neg_needed > total_neg_available:
        scale = total_neg_available / total_neg_needed
        target_neg_train = int(target_neg_train * scale)
        target_neg_val = int(target_neg_val * scale)
        target_neg_test = total_neg_available - target_neg_train - target_neg_val
        logger.info(f"\n负例不足，按比例调整:")
        logger.info(f"  训练集: {target_neg_train}")
        logger.info(f"  验证集: {target_neg_val}")
        logger.info(f"  测试集: {target_neg_test}")

    total_pos = len(pos_train) + len(pos_val) + len(pos_test)
    pos_train_ratio = len(pos_train) / total_pos
    pos_val_ratio = len(pos_val) / total_pos
    pos_test_ratio = len(pos_test) / total_pos

    target_neg_train = int(len(neg_df) * pos_train_ratio)
    target_neg_val = int(len(neg_df) * pos_val_ratio)
    target_neg_test = len(neg_df) - target_neg_train - target_neg_val

    logger.info(f"\n按正例比例分配负例:")
    logger.info(f"  正例占比 - 训练:{pos_train_ratio:.2%}, 验证:{pos_val_ratio:.2%}, 测试:{pos_test_ratio:.2%}")
    logger.info(f"  负例分配 - 训练:{target_neg_train}, 验证:{target_neg_val}, 测试:{target_neg_test}")

    neg_indices = np.random.permutation(neg_df.index.tolist())

    neg_train_indices = neg_indices[:target_neg_train].tolist()
    neg_val_indices = neg_indices[target_neg_train:target_neg_train + target_neg_val].tolist()
    neg_test_indices = neg_indices[target_neg_train + target_neg_val:].tolist()

    neg_train = neg_df.loc[neg_train_indices].reset_index(drop=True)
    neg_val = neg_df.loc[neg_val_indices].reset_index(drop=True)
    neg_test = neg_df.loc[neg_test_indices].reset_index(drop=True)

    logger.info(f"\n负例实际分配:")
    logger.info(f"  训练集: {len(neg_train)}")
    logger.info(f"  验证集: {len(neg_val)}")
    logger.info(f"  测试集: {len(neg_test)}")

    train_df = pd.concat([pos_train, neg_train], ignore_index=True)
    val_df = pd.concat([pos_val, neg_val], ignore_index=True)
    test_df = pd.concat([pos_test, neg_test], ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info("\n" + "="*60)
    logger.info("最终数据划分统计")
    logger.info("="*60)

    for name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        total = len(split_df)
        pos = (split_df['label'] == 1).sum()
        neg = (split_df['label'] == 0).sum()
        pos_ratio = pos / total * 100 if total > 0 else 0
        neg_ratio = neg / total * 100 if total > 0 else 0

        logger.info(f"{name}: {total} 样本 ({total/len(df)*100:.1f}%)")
        logger.info(f"  正例: {pos} ({pos_ratio:.1f}%)")
        logger.info(f"  负例: {neg} ({neg_ratio:.1f}%)")

    logger.info(f"\n各集合正例来源分布:")
    for name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        pos_split = split_df[split_df['label'] == 1]
        logger.info(f"  {name}:")
        for source in ['BindingDB', 'T3DB', 'ToxCast', 'APDB']:
            count = (pos_split[source_col] == source).sum()
            if count > 0:
                logger.info(f"    {source}: {count}")

    logger.info(f"\n蛋白质级别正负平衡检查 (前10个蛋白质):")
    for name, split_df, split_pos, split_neg in [
        ("训练集", train_df, pos_train, neg_train),
        ("验证集", val_df, pos_val, neg_val),
        ("测试集", test_df, pos_test, neg_test)
    ]:
        logger.info(f"  {name}:")

        prot_stats = []
        for prot in split_df['uniprot_id'].unique():
            prot_df = split_df[split_df['uniprot_id'] == prot]
            pos_count = (prot_df['label'] == 1).sum()
            neg_count = (prot_df['label'] == 0).sum()
            if pos_count > 0 or neg_count > 0:
                prot_stats.append((prot, pos_count, neg_count))

        prot_stats.sort(key=lambda x: -x[1])
        for prot, pc, nc in prot_stats[:10]:
            logger.info(f"    {prot}: 正例={pc}, 负例={nc}")

    train_chems = set(train_df['smiles_clean'].unique())
    val_chems = set(val_df['smiles_clean'].unique())
    test_chems = set(test_df['smiles_clean'].unique())

    train_prots = set(train_df['uniprot_id'].unique())
    val_prots = set(val_df['uniprot_id'].unique())
    test_prots = set(test_df['uniprot_id'].unique())

    logger.info(f"\n化学品重叠:")
    logger.info(f"  训练集/验证集: {len(train_chems & val_chems)}")
    logger.info(f"  训练集/测试集: {len(train_chems & test_chems)}")
    logger.info(f"  验证集/测试集: {len(val_chems & test_chems)}")

    logger.info(f"\n蛋白质重叠:")
    logger.info(f"  训练集/验证集: {len(train_prots & val_prots)}")
    logger.info(f"  训练集/测试集: {len(train_prots & test_prots)}")
    logger.info(f"  验证集/测试集: {len(val_prots & test_prots)}")

    return train_df, val_df, test_df


def domain_aware_cold_split(
    df: pd.DataFrame,
    source_col: str = 'data_source',
    tox_to_valtest_ratio: float = 0.3,
    val_test_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np.random.seed(seed)

    required_cols = [source_col, 'label', 'smiles_clean', 'uniprot_id']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    pos_df = df[df['label'] == 1].copy()
    neg_df = df[df['label'] == 0].copy()

    logger.info("="*60)
    logger.info("Domain-Aware Cold Chemical Split")
    logger.info("="*60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"  Positive: {len(pos_df)} ({len(pos_df)/len(df)*100:.1f}%)")
    logger.info(f"  Negative: {len(neg_df)} ({len(neg_df)/len(df)*100:.1f}%)")

    pos_source_counts = pos_df[source_col].value_counts()
    logger.info(f"\nPositive samples by source:")
    for source, count in pos_source_counts.items():
        logger.info(f"  {source}: {count} ({count/len(pos_df)*100:.1f}%)")

    drug_sources = ['BindingDB']
    tox_sources = ['T3DB', 'ToxCast', 'APDB']

    pos_drug = pos_df[pos_df[source_col].isin(drug_sources)].copy()
    pos_tox = pos_df[pos_df[source_col].isin(tox_sources)].copy()

    logger.info(f"\nPositive samples分类:")
    logger.info(f"  药物正例 (BindingDB): {len(pos_drug)}")
    logger.info(f"  毒理正例 (T3DB+ToxCast+APDB): {len(pos_tox)}")

    for source in tox_sources:
        count = (pos_tox[source_col] == source).sum()
        logger.info(f"    - {source}: {count}")

    pos_drug_train = pos_drug

    tox_unique_chemicals = pos_tox['smiles_clean'].unique().tolist()
    n_tox_chemicals = len(tox_unique_chemicals)

    logger.info(f"\n毒理正例中的唯一化学品数: {n_tox_chemicals}")

    np.random.shuffle(tox_unique_chemicals)

    n_valtest_chems = int(n_tox_chemicals * tox_to_valtest_ratio)
    n_train_chems = n_tox_chemicals - n_valtest_chems
    n_val_chems = int(n_valtest_chems * val_test_ratio)
    n_test_chems = n_valtest_chems - n_val_chems

    logger.info(f"\n化学品冷划分:")
    logger.info(f"  训练集化学品: {n_train_chems}")
    logger.info(f"  验证集化学品: {n_val_chems}")
    logger.info(f"  测试集化学品: {n_test_chems}")

    train_chemicals = set(tox_unique_chemicals[:n_train_chems])
    val_chemicals = set(tox_unique_chemicals[n_train_chems:n_train_chems + n_val_chems])
    test_chemicals = set(tox_unique_chemicals[n_train_chems + n_val_chems:])

    pos_tox_train = pos_tox[pos_tox['smiles_clean'].isin(train_chemicals)].reset_index(drop=True)
    pos_tox_val = pos_tox[pos_tox['smiles_clean'].isin(val_chemicals)].reset_index(drop=True)
    pos_tox_test = pos_tox[pos_tox['smiles_clean'].isin(test_chemicals)].reset_index(drop=True)

    logger.info(f"\n毒理正例实际分配:")
    logger.info(f"  训练集: {len(pos_tox_train)}")
    logger.info(f"  验证集: {len(pos_tox_val)}")
    logger.info(f"  测试集: {len(pos_tox_test)}")

    pos_train = pd.concat([pos_drug_train, pos_tox_train], ignore_index=True)
    pos_val = pos_tox_val
    pos_test = pos_tox_test

    logger.info(f"\n正例划分结果:")
    logger.info(f"  训练集: {len(pos_train)} (药物:{len(pos_drug_train)}, 毒理:{len(pos_tox_train)})")
    logger.info(f"  验证集: {len(pos_val)} (全部毒理)")
    logger.info(f"  测试集: {len(pos_test)} (全部毒理)")

    logger.info("\n" + "="*60)
    logger.info("智能分配负例")
    logger.info("="*60)

    all_split_chemicals = train_chemicals | val_chemicals | test_chemicals
    neg_smiles = neg_df['smiles_clean']
    neg_train_mask = neg_smiles.isin(train_chemicals) | ~neg_smiles.isin(all_split_chemicals)
    neg_val_mask = neg_smiles.isin(val_chemicals)
    neg_test_mask = neg_smiles.isin(test_chemicals)

    neg_train = neg_df[neg_train_mask].reset_index(drop=True)
    neg_val = neg_df[neg_val_mask].reset_index(drop=True)
    neg_test = neg_df[neg_test_mask].reset_index(drop=True)

    logger.info(f"\n按化学品归属分配负例:")
    logger.info(f"  训练集: {len(neg_train)} (含未出现在毒理正例化学品集合中的负例)")
    logger.info(f"  验证集: {len(neg_val)}")
    logger.info(f"  测试集: {len(neg_test)}")

    logger.info(f"\n负例实际分配:")
    logger.info(f"  训练集: {len(neg_train)}")
    logger.info(f"  验证集: {len(neg_val)}")
    logger.info(f"  测试集: {len(neg_test)}")

    train_df = pd.concat([pos_train, neg_train], ignore_index=True)
    val_df = pd.concat([pos_val, neg_val], ignore_index=True)
    test_df = pd.concat([pos_test, neg_test], ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info("\n" + "="*60)
    logger.info("最终数据划分统计")
    logger.info("="*60)

    for name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        total = len(split_df)
        pos = (split_df['label'] == 1).sum()
        neg = (split_df['label'] == 0).sum()
        pos_ratio = pos / total * 100 if total > 0 else 0
        neg_ratio = neg / total * 100 if total > 0 else 0

        logger.info(f"{name}: {total} 样本 ({total/len(df)*100:.1f}%)")
        logger.info(f"  正例: {pos} ({pos_ratio:.1f}%)")
        logger.info(f"  负例: {neg} ({neg_ratio:.1f}%)")

    logger.info(f"\n各集合正例来源分布:")
    for name, split_df in [("训练集", train_df), ("验证集", val_df), ("测试集", test_df)]:
        pos_split = split_df[split_df['label'] == 1]
        logger.info(f"  {name}:")
        for source in ['BindingDB', 'T3DB', 'ToxCast', 'APDB']:
            count = (pos_split[source_col] == source).sum()
            if count > 0:
                logger.info(f"    {source}: {count}")

    train_chems_pos = set(pos_train['smiles_clean'].unique())
    val_chems_pos = set(pos_val['smiles_clean'].unique())
    test_chems_pos = set(pos_test['smiles_clean'].unique())

    val_train_overlap = len(val_chems_pos & train_chems_pos)
    test_train_overlap = len(test_chems_pos & train_chems_pos)

    logger.info(f"\n冷启动验证（毒理正例化学品重叠）:")
    logger.info(f"  验证集/训练集化学品重叠: {val_train_overlap} (应为0)")
    logger.info(f"  测试集/训练集化学品重叠: {test_train_overlap} (应为0)")

    train_chems_all = set(train_df['smiles_clean'].unique())
    val_chems_all = set(val_df['smiles_clean'].unique())
    test_chems_all = set(test_df['smiles_clean'].unique())

    logger.info(f"\n所有化学品重叠:")
    logger.info(f"  训练集/验证集: {len(train_chems_all & val_chems_all)}")
    logger.info(f"  训练集/测试集: {len(train_chems_all & test_chems_all)}")
    logger.info(f"  验证集/测试集: {len(val_chems_all & test_chems_all)}")

    train_prots = set(train_df['uniprot_id'].unique())
    val_prots = set(val_df['uniprot_id'].unique())
    test_prots = set(test_df['uniprot_id'].unique())

    logger.info(f"\n蛋白质重叠:")
    logger.info(f"  训练集/验证集: {len(train_prots & val_prots)}")
    logger.info(f"  训练集/测试集: {len(train_prots & test_prots)}")
    logger.info(f"  验证集/测试集: {len(val_prots & test_prots)}")

    return train_df, val_df, test_df


def binding_collate_fn(batch: List[Dict]) -> Dict:
    result = {
        'smiles': [s['smiles'] for s in batch],
        'sequences': [s['sequence'] for s in batch],
        'toxprint': torch.stack([s['toxprint'] for s in batch]),
        'benigni': torch.stack([s['benigni'] for s in batch]),
        'edc': torch.stack([s['edc'] for s in batch]),
        'label': torch.stack([s['label'] for s in batch]),
    }

    if 'protein_tokens' in batch[0]:
        result['protein_tokens'] = torch.stack([s['protein_tokens'] for s in batch])
        result['protein_mask'] = torch.stack([s['protein_mask'] for s in batch])
    return result


def effect_collate_fn(batch: List[Dict]) -> Dict:
    result = {
        'smiles': [s['smiles'] for s in batch],
        'sequences': [s['sequence'] for s in batch],
        'toxprint': torch.stack([s['toxprint'] for s in batch]),
        'benigni': torch.stack([s['benigni'] for s in batch]),
        'edc': torch.stack([s['edc'] for s in batch]),
        'has_effect': torch.stack([s['has_effect'] for s in batch]),
        'effect_type': torch.stack([s['effect_type'] for s in batch]),
    }

    if 'protein_tokens' in batch[0]:
        result['protein_tokens'] = torch.stack([s['protein_tokens'] for s in batch])
        result['protein_mask'] = torch.stack([s['protein_mask'] for s in batch])
    return result


def create_binding_dataloaders(
    cold_split: bool = True,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
    balance: bool = False,
    use_sample_weights: bool = True,
    weight_config: Optional[dict] = None,
    domain_aware: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[SampleWeightCalculator]]:
    logger.info("="*50)
    logger.info("Creating Binding Task DataLoaders")
    logger.info("="*50)

    df = load_binding_data()

    if domain_aware and cold_split:
        logger.info("Using domain-aware + cold chemical split")
        train_df, val_df, test_df = domain_aware_cold_split(
            df, source_col='data_source', seed=seed
        )
    elif domain_aware:
        logger.info("Using domain-aware split (toxicology/pharmacology)")
        train_df, val_df, test_df = domain_aware_split(
            df, source_col='data_source', seed=seed
        )
    elif cold_split:
        train_df, val_df, test_df = cold_chemical_split(
            df, 'smiles_clean', train_config.train_ratio,
            train_config.val_ratio, train_config.test_ratio, seed
        )
    else:
        train_df, val_df, test_df = random_split(
            df, train_config.train_ratio, train_config.val_ratio,
            train_config.test_ratio, seed
        )

    if balance:
        def balance_dataset(df, name, seed):
            pos = df[df['label'] == 1]
            neg = df[df['label'] == 0]
            n_pos, n_neg = len(pos), len(neg)
            logger.info(f"{name} before sampling - Positive: {n_pos}, Negative: {n_neg}")

            n_sample = min(n_pos, n_neg)
            if n_pos > n_neg:
                pos_sampled = pos.sample(n=n_sample, random_state=seed)
                neg_sampled = neg
            else:
                pos_sampled = pos
                neg_sampled = neg.sample(n=n_sample, random_state=seed)

            balanced_df = pd.concat([pos_sampled, neg_sampled], ignore_index=True)
            balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            logger.info(f"{name} after sampling - Positive: {len(pos_sampled)}, Negative: {len(neg_sampled)}")
            return balanced_df

        train_df = balance_dataset(train_df, "Train set", seed)
        val_df = balance_dataset(val_df, "Val set", seed)
        test_df = balance_dataset(test_df, "Test set", seed)
    else:
        logger.info("Using original data without balance sampling")
        for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            n_pos = (split_df['label'] == 1).sum()
            n_neg = (split_df['label'] == 0).sum()
            logger.info(f"{name} - Positive: {n_pos}, Negative: {n_neg}")

    weight_calculator = None
    train_sampler = None

    if use_sample_weights:
        logger.info("\n" + "="*50)
        logger.info("Computing Sample Weights")
        logger.info("="*50)

        if weight_config is None:
            weight_config = {
                'use_protein_freq_weight': True,
                'use_chemical_freq_weight': True,
                'use_difficulty_weight': True,
                'difficulty_alpha': 0.5,
            }

        logger.info(f"Weight config: {weight_config}")

        calculator = SampleWeightCalculator(**weight_config)
        calculator.fit(train_df)
        train_weights = calculator.get_sample_weights(train_df)

        logger.info(f"Train sample weights:")
        logger.info(f"  Min: {train_weights.min():.3f}")
        logger.info(f"  Max: {train_weights.max():.3f}")
        logger.info(f"  Mean: {train_weights.mean():.3f}")
        logger.info(f"  Median: {np.median(train_weights):.3f}")

        from torch.utils.data.sampler import WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            torch.DoubleTensor(train_weights),
            num_samples=len(train_weights),
            replacement=True
        )

        weight_calculator = calculator

        weight_dict = calculator.get_weight_dict()
        logger.info("\nWeight Summary:")
        for name, config in weight_dict.items():
            if config['enabled']:
                logger.info(f"  {name}: min={config.get('min', 0):.3f}, max={config.get('max', 0):.3f}")

    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')

    datasets = []
    for name, split_df, cache_name in [
        ('train', train_df, 'binding_train'),
        ('val', val_df, 'binding_val'),
        ('test', test_df, 'binding_test')
    ]:
        ds = BindingDataset(
            smiles_list=split_df['smiles_clean'].tolist(),
            sequences_list=split_df['protein_sequence'].tolist(),
            binding_labels=split_df['label'].values,
            max_protein_length=model_config.max_protein_length,
            cache_path=os.path.join(cache_dir, f'{cache_name}_tox_features.npz')
        )
        datasets.append(ds)

    train_loader = DataLoader(
        datasets[0],
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=binding_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=binding_collate_fn,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=binding_collate_fn,
                             pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader, weight_calculator


def create_effect_dataloaders(
    cold_split: bool = True,
    batch_size: int = 32,
    target_ratio: str = "1:1:1",
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    logger.info("="*50)
    logger.info("Creating Effect Task DataLoaders (Hierarchical)")
    logger.info("="*50)

    effect_df = load_effect_data()

    combined_df = create_effect_data_with_negatives(
        effect_df=effect_df,
        target_ratio=target_ratio,
        seed=seed
    )

    if cold_split:
        train_df, val_df, test_df = cold_chemical_split(
            combined_df, 'smiles_clean', train_config.train_ratio,
            train_config.val_ratio, train_config.test_ratio, seed
        )
    else:
        train_df, val_df, test_df = random_split(
            combined_df, train_config.train_ratio, train_config.val_ratio,
            train_config.test_ratio, seed
        )

    train_activation = train_df[(train_df['has_effect'] == 1) & (train_df['effect_label'] == 1)]
    train_inhibition = train_df[(train_df['has_effect'] == 1) & (train_df['effect_label'] == 0)]
    train_no_effect = train_df[train_df['has_effect'] == 0]

    n_act = len(train_activation)
    n_inh = len(train_inhibition)
    logger.info(f"Train set before sampling - Activation: {n_act}, Inhibition: {n_inh}, No_effect: {len(train_no_effect)}")

    n_sample = min(n_act, n_inh)
    if n_act > n_inh:
        train_activation_sampled = train_activation.sample(n=n_sample, random_state=seed)
        train_inhibition_sampled = train_inhibition
    else:
        train_activation_sampled = train_activation
        train_inhibition_sampled = train_inhibition.sample(n=n_sample, random_state=seed)

    train_df = pd.concat([train_activation_sampled, train_inhibition_sampled, train_no_effect], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(f"Train set after sampling - Activation: {len(train_activation_sampled)}, Inhibition: {len(train_inhibition_sampled)}, No_effect: {len(train_no_effect)}")

    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        n_has_effect = (df['has_effect'] == 1).sum()
        n_no_effect = (df['has_effect'] == 0).sum()
        n_act = ((df['has_effect'] == 1) & (df['effect_label'] == 1)).sum()
        n_inh = ((df['has_effect'] == 1) & (df['effect_label'] == 0)).sum()
        logger.info(f"{name}: total={len(df)}, has_effect={n_has_effect} "
                   f"(act={n_act}, inh={n_inh}), no_effect={n_no_effect}")

    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')

    datasets = []
    for name, split_df, cache_name in [
        ('train', train_df, 'effect_train'),
        ('val', val_df, 'effect_val'),
        ('test', test_df, 'effect_test')
    ]:
        ds = EffectDataset(
            smiles_list=split_df['smiles_clean'].tolist(),
            sequences_list=split_df['sequence'].tolist(),
            has_effect_labels=split_df['has_effect'].values,
            effect_type_labels=split_df['effect_label'].values,
            max_protein_length=model_config.max_protein_length,
            cache_path=os.path.join(cache_dir, f'{cache_name}_tox_features.npz')
        )
        datasets.append(ds)

    train_loader = DataLoader(datasets[0], batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=effect_collate_fn,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(datasets[1], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=effect_collate_fn,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(datasets[2], batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=effect_collate_fn,
                             pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader


def prepare_binding_data(cold_split=True, batch_size=32, seed=42, balance=False, use_sample_weights=True, domain_aware=False, num_workers=0):
    return create_binding_dataloaders(
        cold_split=cold_split,
        batch_size=batch_size,
        seed=seed,
        balance=balance,
        use_sample_weights=use_sample_weights,
        domain_aware=domain_aware,
        num_workers=num_workers
    )


def prepare_effect_data(cold_split=True, batch_size=32, target_ratio="1:1:1", seed=42, num_workers=0):
    return create_effect_dataloaders(cold_split, batch_size, target_ratio, seed, num_workers)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("Testing Effect Data Loading (Hierarchical)")
    print("="*60)

    try:
        print("\nTesting with target_ratio='1:1:1':")
        train_loader, val_loader, test_loader = prepare_effect_data(
            cold_split=True, batch_size=4, target_ratio="1:1:1"
        )

        batch = next(iter(train_loader))
        print(f"\nBatch keys: {batch.keys()}")
        print(f"has_effect: {batch['has_effect']}")
        print(f"effect_type: {batch['effect_type']}")

        print("\n" + "="*60)
        print("Testing with target_ratio='1:1:2':")
        train_loader2, val_loader2, test_loader2 = prepare_effect_data(
            cold_split=True, batch_size=4, target_ratio="1:1:2"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()

    print("\nDone!")
