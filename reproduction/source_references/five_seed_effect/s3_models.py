"""Isolated model variants for the controlled S3 ablation study.

This module imports the production model definitions but never modifies them.  Every
variant changes only the component named by its configuration.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from attention import BilinearAttention
from model_fusion import BindingModelFusion, EffectModelFusion, masked_mean_pool


class MaskedMeanInteraction(nn.Module):
    """Masked means followed by a capacity-matched interaction MLP."""

    def __init__(self, hidden_dim: int, interaction_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, interaction_dim),
            nn.LayerNorm(interaction_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, interaction_dim),
            nn.LayerNorm(interaction_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_chem, h_prot, chem_mask=None, prot_mask=None):
        chem = masked_mean_pool(h_chem, chem_mask)
        prot = masked_mean_pool(h_prot, prot_mask)
        interaction = self.mlp(
            torch.cat([chem, prot, chem * prot, torch.abs(chem - prot)], dim=-1)
        )
        return interaction, None, chem, prot


class BindingAblationModel(BindingModelFusion):
    """Production Binding model with an isolated interaction-module switch."""

    def __init__(self, interaction_variant: str = "full", **kwargs):
        if interaction_variant not in {"full", "mean_pooling", "single_head"}:
            raise ValueError(f"Unknown binding variant: {interaction_variant}")
        super().__init__(**kwargs)
        if not self.use_residue_attention:
            raise ValueError("S3 Binding ablations require residue attention inputs")

        self.interaction_variant = interaction_variant
        interaction_dim = self.classifier[0].in_features
        if interaction_variant == "mean_pooling":
            self.interaction = MaskedMeanInteraction(
                hidden_dim=self.hidden_dim,
                interaction_dim=interaction_dim,
                dropout=self.dropout_rate,
            )
        elif interaction_variant == "single_head":
            self.interaction = BilinearAttention(
                atom_dim=self.hidden_dim,
                residue_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                output_dim=interaction_dim,
                num_heads=1,
                dropout=self.dropout_rate,
                return_head_attention=False,
            )

    def compute_local_interaction(
        self,
        H_chem: torch.Tensor,
        H_prot: torch.Tensor,
        chem_mask: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        if self.interaction_variant == "mean_pooling":
            return self.interaction(H_chem, H_prot, chem_mask, prot_mask)
        return super().compute_local_interaction(H_chem, H_prot, chem_mask, prot_mask)


class RawGlobalOnlyModel(nn.Module):
    """Effect model containing the production raw-global branch and no local branch."""

    def __init__(
        self,
        hidden_dim: int = 256,
        interaction_dim: int = 512,
        dropout: float = 0.3,
        raw_chem_dim: int = 768,
        raw_prot_dim: int = 1280,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.raw_global_branch = nn.Sequential(
            nn.Linear(raw_chem_dim + raw_prot_dim, interaction_dim),
            nn.LayerNorm(interaction_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_dim, interaction_dim),
            nn.LayerNorm(interaction_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        tox_feat=None,
        chemberta_feat=None,
        esm2_feat=None,
        chem_tokens_feat=None,
        chem_mask=None,
        chem_adj=None,
        prot_residue_feat=None,
        prot_mask=None,
    ):
        raw_chem = masked_mean_pool(chem_tokens_feat, chem_mask)
        raw_prot = masked_mean_pool(prot_residue_feat, prot_mask)
        raw_global = self.raw_global_branch(torch.cat([raw_chem, raw_prot], dim=-1))
        logits = self.classifier(raw_global).squeeze(-1)
        return logits, None, None, None


def build_s3_model(task: str, variant: str, hidden_dim: int, dropout: float):
    """Construct one audited S3 variant from the common production dimensions."""
    interaction_dim = hidden_dim * 2
    common = dict(
        chemberta_dim=768,
        esm2_dim=1280,
        hidden_dim=hidden_dim,
        interaction_dim=interaction_dim,
        dropout=dropout,
        use_residue_attention=True,
        use_toxprint=False,
        use_multi_granularity=False,
        use_interaction_cnn=False,
        use_molecular_gcn=False,
    )
    if task == "binding":
        return BindingAblationModel(interaction_variant=variant, **common)
    if variant == "full":
        return EffectModelFusion(
            use_projected_global_branch=False,
            use_raw_global_branch=True,
            **common,
        )
    if variant == "local_only":
        return BindingModelFusion(**common)
    if variant == "global_only":
        return RawGlobalOnlyModel(
            hidden_dim=hidden_dim,
            interaction_dim=interaction_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown effect variant: {variant}")


def branch_flags(task: str, variant: str):
    if task == "binding":
        return True, False
    return {
        "full": (True, True),
        "local_only": (True, False),
        "global_only": (False, True),
    }[variant]
