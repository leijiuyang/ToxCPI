import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SampleWeightCalculator:
    def __init__(
        self,
        use_protein_freq_weight: bool = True,
        use_chemical_freq_weight: bool = True,
        use_difficulty_weight: bool = True,
        protein_freq_smooth: float = 1.0,
        chemical_freq_smooth: float = 1.0,
        difficulty_alpha: float = 0.5,
        normalize: bool = True
    ):
        self.use_protein_freq_weight = use_protein_freq_weight
        self.use_chemical_freq_weight = use_chemical_freq_weight
        self.use_difficulty_weight = use_difficulty_weight
        self.protein_freq_smooth = protein_freq_smooth
        self.chemical_freq_smooth = chemical_freq_smooth
        self.difficulty_alpha = difficulty_alpha
        self.normalize = normalize

        self.protein_weights: Dict[str, float] = {}
        self.chemical_weights: Dict[str, float] = {}
        self.sample_difficulty_weights: Dict[Tuple[str, str], float] = {}

        self.stats = {}

    def _compute_protein_freq_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        protein_counts = df['uniprot_id'].value_counts()
        n_samples = len(df)
        n_proteins = len(protein_counts)

        weights = {}
        for protein, count in protein_counts.items():
            smooth_count = np.sqrt(count) * self.protein_freq_smooth

            weight = 1.0 / (smooth_count + 1)
            weights[protein] = weight

        min_w = min(weights.values())
        max_w = max(weights.values())
        if max_w > min_w:
            for protein in weights:
                weights[protein] = 0.5 + 1.5 * (weights[protein] - min_w) / (max_w - min_w)

        self.stats['protein_count_min'] = int(protein_counts.min())
        self.stats['protein_count_max'] = int(protein_counts.max())
        self.stats['protein_count_mean'] = float(protein_counts.mean())
        self.stats['protein_weight_min'] = float(min(weights.values()))
        self.stats['protein_weight_max'] = float(max(weights.values()))

        logger.info(f"Protein freq weights: min={self.stats['protein_weight_min']:.3f}, "
                   f"max={self.stats['protein_weight_max']:.3f}")

        return weights

    def _compute_chemical_freq_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        chemical_counts = df['smiles_clean'].value_counts()

        weights = {}
        for chemical, count in chemical_counts.items():
            smooth_count = np.sqrt(count) * self.chemical_freq_smooth
            weight = 1.0 / (smooth_count + 1)
            weights[chemical] = weight

        min_w = min(weights.values())
        max_w = max(weights.values())
        if max_w > min_w:
            for chemical in weights:
                weights[chemical] = 0.5 + 1.5 * (weights[chemical] - min_w) / (max_w - min_w)

        self.stats['chemical_count_min'] = int(chemical_counts.min())
        self.stats['chemical_count_max'] = int(chemical_counts.max())
        self.stats['chemical_count_mean'] = float(chemical_counts.mean())
        self.stats['chemical_weight_min'] = float(min(weights.values()))
        self.stats['chemical_weight_max'] = float(max(weights.values()))

        logger.info(f"Chemical freq weights: min={self.stats['chemical_weight_min']:.3f}, "
                   f"max={self.stats['chemical_weight_max']:.3f}")

        return weights

    def _compute_difficulty_weights(self, df: pd.DataFrame) -> Dict[Tuple[str, str], float]:
        weights = {}

        protein_pos_ratio = {}
        for protein in df['uniprot_id'].unique():
            prot_df = df[df['uniprot_id'] == protein]
            total = len(prot_df)
            pos = (prot_df['label'] == 1).sum()
            protein_pos_ratio[protein] = pos / total if total > 0 else 0.5

        chemical_pos_ratio = {}
        for chemical in df['smiles_clean'].unique():
            chem_df = df[df['smiles_clean'] == chemical]
            total = len(chem_df)
            pos = (chem_df['label'] == 1).sum()
            chemical_pos_ratio[chemical] = pos / total if total > 0 else 0.5

        for _, row in df.iterrows():
            protein = row['uniprot_id']
            chemical = row['smiles_clean']
            label = row['label']

            if label == 1:
                protein_rarity = 1 - protein_pos_ratio[protein]
                chemical_rarity = 1 - chemical_pos_ratio[chemical]
            else:
                protein_rarity = protein_pos_ratio[protein]
                chemical_rarity = chemical_pos_ratio[chemical]

            difficulty = (protein_rarity + chemical_rarity) / 2

            if difficulty < 0.5:
                weight = 1 + self.difficulty_alpha * difficulty
            else:
                weight = 1 + self.difficulty_alpha * (difficulty / (1 - difficulty + 0.1))

            weight = np.clip(weight, 0.5, 3.0)

            weights[(protein, chemical, label)] = weight

        weight_values = list(weights.values())
        self.stats['difficulty_weight_min'] = float(min(weight_values))
        self.stats['difficulty_weight_max'] = float(max(weight_values))
        self.stats['difficulty_weight_mean'] = float(np.mean(weight_values))

        logger.info(f"Difficulty weights: min={self.stats['difficulty_weight_min']:.3f}, "
                   f"max={self.stats['difficulty_weight_max']:.3f}, "
                   f"mean={self.stats['difficulty_weight_mean']:.3f}")

        return weights

    def fit(self, df: pd.DataFrame) -> 'SampleWeightCalculator':
        logger.info("Computing sample weights...")

        if self.use_protein_freq_weight:
            self.protein_weights = self._compute_protein_freq_weights(df)

        if self.use_chemical_freq_weight:
            self.chemical_weights = self._compute_chemical_freq_weights(df)

        if self.use_difficulty_weight:
            self.sample_difficulty_weights = self._compute_difficulty_weights(df)

        logger.info("Sample weights computed successfully")
        return self

    def get_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        n_samples = len(df)
        weights = np.ones(n_samples)

        for i, (idx, row) in enumerate(df.iterrows()):
            protein = row['uniprot_id']
            chemical = row['smiles_clean']
            label = row['label']

            w = 1.0

            if self.use_protein_freq_weight and protein in self.protein_weights:
                w *= self.protein_weights[protein]

            if self.use_chemical_freq_weight and chemical in self.chemical_weights:
                w *= self.chemical_weights[chemical]

            if self.use_difficulty_weight:
                key = (protein, chemical, label)
                if key in self.sample_difficulty_weights:
                    w *= self.sample_difficulty_weights[key]

            weights[i] = w

        if self.normalize:
            weights = weights / weights.mean()

        return weights

    def get_weight_dict(self) -> Dict:
        return {
            'protein_weights': {
                'enabled': self.use_protein_freq_weight,
                'count': len(self.protein_weights),
                'min': self.stats.get('protein_weight_min', 0),
                'max': self.stats.get('protein_weight_max', 0),
            },
            'chemical_weights': {
                'enabled': self.use_chemical_freq_weight,
                'count': len(self.chemical_weights),
                'min': self.stats.get('chemical_weight_min', 0),
                'max': self.stats.get('chemical_weight_max', 0),
            },
            'difficulty_weights': {
                'enabled': self.use_difficulty_weight,
                'count': len(self.sample_difficulty_weights),
                'min': self.stats.get('difficulty_weight_min', 0),
                'max': self.stats.get('difficulty_weight_max', 0),
                'mean': self.stats.get('difficulty_weight_mean', 0),
            }
        }

    def save_weights(self, filepath: str):
        import json
        data = {
            'config': {
                'use_protein_freq_weight': self.use_protein_freq_weight,
                'use_chemical_freq_weight': self.use_chemical_freq_weight,
                'use_difficulty_weight': self.use_difficulty_weight,
                'protein_freq_smooth': self.protein_freq_smooth,
                'chemical_freq_smooth': self.chemical_freq_smooth,
                'difficulty_alpha': self.difficulty_alpha,
            },
            'stats': self.stats,
            'protein_weights': self.protein_weights,
            'chemical_weights': self.chemical_weights,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Weights saved to {filepath}")

    @classmethod
    def load_weights(cls, filepath: str) -> 'SampleWeightCalculator':
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        calculator = cls(
            use_protein_freq_weight=data['config']['use_protein_freq_weight'],
            use_chemical_freq_weight=data['config']['use_chemical_freq_weight'],
            use_difficulty_weight=data['config']['use_difficulty_weight'],
            protein_freq_smooth=data['config']['protein_freq_smooth'],
            chemical_freq_smooth=data['config']['chemical_freq_smooth'],
            difficulty_alpha=data['config']['difficulty_alpha'],
        )
        calculator.protein_weights = {k: v for k, v in data['protein_weights'].items()}
        calculator.chemical_weights = {k: v for k, v in data['chemical_weights'].items()}
        calculator.stats = data['stats']

        logger.info(f"Weights loaded from {filepath}")
        return calculator


class WeightedSampler:
    def __init__(self, weights: np.ndarray, replacement: bool = True):
        self.weights = weights
        self.replacement = replacement
        self.num_samples = len(weights)

    def get_sampler(self):
        import torch
        from torch.utils.data.sampler import WeightedRandomSampler

        weights_tensor = torch.DoubleTensor(self.weights)
        return WeightedRandomSampler(
            weights_tensor,
            num_samples=self.num_samples,
            replacement=self.replacement
        )


def compute_and_apply_weights(
    df: pd.DataFrame,
    sampler_config: Optional[Dict] = None
) -> Tuple[np.ndarray, SampleWeightCalculator]:
    if sampler_config is None:
        sampler_config = {
            'use_protein_freq_weight': True,
            'use_chemical_freq_weight': True,
            'use_difficulty_weight': True,
        }

    calculator = SampleWeightCalculator(**sampler_config)
    calculator.fit(df)
    weights = calculator.get_sample_weights(df)

    return weights, calculator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    n_samples = 1000

    data = {
        'uniprot_id': ['P1'] * 400 + ['P2'] * 200 + ['P3'] * 100 + ['P4'] * 50 + ['P5'] * 50 +
                      ['P6'] * 50 + ['P7'] * 50 + ['P8'] * 50 + ['P9'] * 50,
        'smiles_clean': [f'C{i%50}' for i in range(n_samples)],
        'label': [0] * 550 + [1] * 450
    }

    df = pd.DataFrame(data)
    for i in range(len(df)):
        protein = df.loc[i, 'uniprot_id']
        if protein == 'P1':
            df.loc[i, 'label'] = 1 if np.random.rand() < 0.3 else 0
        elif protein == 'P2':
            df.loc[i, 'label'] = 1 if np.random.rand() < 0.5 else 0
        elif protein == 'P3':
            df.loc[i, 'label'] = 1 if np.random.rand() < 0.7 else 0

    print("\n" + "="*60)
    print("Testing SampleWeightCalculator")
    print("="*60)

    calculator = SampleWeightCalculator(
        use_protein_freq_weight=True,
        use_chemical_freq_weight=True,
        use_difficulty_weight=True,
        difficulty_alpha=0.5
    )

    calculator.fit(df)
    weights = calculator.get_sample_weights(df)

    print(f"\nWeight statistics:")
    print(f"  Min: {weights.min():.3f}")
    print(f"  Max: {weights.max():.3f}")
    print(f"  Mean: {weights.mean():.3f}")
    print(f"  Std: {weights.std():.3f}")

    print(f"\nSample weights by protein:")
    for protein in ['P1', 'P2', 'P3', 'P9']:
        mask = df['uniprot_id'] == protein
        print(f"  {protein}: avg weight = {weights[mask].mean():.3f}, "
              f"samples = {mask.sum()}")

    print("\n" + "="*60)
