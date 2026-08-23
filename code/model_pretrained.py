









import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class EncoderBlock(nn.Module):

    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class CrossAttention(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.comp_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.prot_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.comp_norm = nn.LayerNorm(hidden_dim)
        self.prot_norm = nn.LayerNorm(hidden_dim)

    def forward(self, comp_emb: torch.Tensor, prot_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        comp_seq = comp_emb.unsqueeze(1)
        prot_seq = prot_emb.unsqueeze(1)


        comp_attn_out, _ = self.comp_attn(query=comp_seq, key=prot_seq, value=prot_seq)
        comp_updated = self.comp_norm(comp_attn_out.squeeze(1) + comp_emb)

        prot_attn_out, _ = self.prot_attn(query=prot_seq, key=comp_seq, value=comp_seq)
        prot_updated = self.prot_norm(prot_attn_out.squeeze(1) + prot_emb)

        return comp_updated, prot_updated


class BindingModelPretrained(nn.Module):









    def __init__(
        self,
        comp_dim: int = 768,
        prot_dim: int = 1280,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
        comp_noise: float = 0.0,
        prot_noise: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.comp_noise = comp_noise
        self.prot_noise = prot_noise


        self.comp_proj = nn.Linear(comp_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)


        self.comp_encoder = nn.Sequential(
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout)
        )
        self.prot_encoder = nn.Sequential(
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout)
        )


        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)


        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def add_noise(self, x: torch.Tensor, noise_level: float) -> torch.Tensor:
        if self.training and noise_level > 0:
            noise = torch.randn_like(x) * noise_level
            return x + noise
        return x

    def forward(
        self,
        comp_feat: torch.Tensor,
        prot_feat: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        comp_feat = self.add_noise(comp_feat, self.comp_noise)
        prot_feat = self.add_noise(prot_feat, self.prot_noise)


        comp_emb = self.comp_proj(comp_feat)
        prot_emb = self.prot_proj(prot_feat)


        comp_enc = self.comp_encoder(comp_emb)
        prot_enc = self.prot_encoder(prot_emb)


        comp_updated, prot_updated = self.cross_attn(comp_enc, prot_enc)


        concat = torch.cat([comp_updated, prot_updated], dim=1)
        fused = self.fusion(concat)


        logits = self.classifier(fused).squeeze(-1)
        prob = torch.sigmoid(logits)

        return {
            'logits': logits.unsqueeze(-1) if logits.dim() == 1 else logits,
            'prob': prob.unsqueeze(-1) if prob.dim() == 1 else prob
        }


class EffectModelPretrained(nn.Module):





    def __init__(
        self,
        comp_dim: int = 768,
        prot_dim: int = 1280,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
        comp_noise: float = 0.0,
        prot_noise: float = 0.0
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.comp_noise = comp_noise
        self.prot_noise = prot_noise


        self.comp_proj = nn.Linear(comp_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)


        self.comp_encoder = nn.Sequential(
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout)
        )
        self.prot_encoder = nn.Sequential(
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout),
            EncoderBlock(hidden_dim, dropout)
        )


        self.cross_attn = CrossAttention(hidden_dim, num_heads, dropout)


        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )


        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        self.has_effect_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )


        self.effect_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )

    def add_noise(self, x: torch.Tensor, noise_level: float) -> torch.Tensor:
        if self.training and noise_level > 0:
            noise = torch.randn_like(x) * noise_level
            return x + noise
        return x

    def forward(
        self,
        comp_feat: torch.Tensor,
        prot_feat: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        comp_feat = self.add_noise(comp_feat, self.comp_noise)
        prot_feat = self.add_noise(prot_feat, self.prot_noise)


        comp_emb = self.comp_proj(comp_feat)
        prot_emb = self.prot_proj(prot_feat)


        comp_enc = self.comp_encoder(comp_emb)
        prot_enc = self.prot_encoder(prot_emb)


        comp_updated, prot_updated = self.cross_attn(comp_enc, prot_enc)


        concat = torch.cat([comp_updated, prot_updated], dim=1)
        fused = self.fusion(concat)


        shared_feat = self.shared_layer(fused)


        has_effect_logits = self.has_effect_head(shared_feat)
        has_effect_prob = torch.sigmoid(has_effect_logits)


        effect_type_logits = self.effect_type_head(shared_feat)
        effect_type_prob = torch.sigmoid(effect_type_logits)


        prob_activation = has_effect_prob * effect_type_prob
        prob_inhibition = has_effect_prob * (1 - effect_type_prob)
        prob_no_effect = 1 - has_effect_prob

        return {
            'has_effect_logits': has_effect_logits,
            'has_effect_prob': has_effect_prob,
            'effect_type_logits': effect_type_logits,
            'effect_type_prob': effect_type_prob,
            'prob_activation': prob_activation,
            'prob_inhibition': prob_inhibition,
            'prob_no_effect': prob_no_effect
        }


def get_pretrained_model(
    task: str,
    comp_dim: int = 768,
    prot_dim: int = 1280,
    hidden_dim: int = 256,
    num_heads: int = 8,
    dropout: float = 0.3,
    comp_noise: float = 0.0,
    prot_noise: float = 0.0
) -> nn.Module:



    if task == 'binding':
        return BindingModelPretrained(
            comp_dim=comp_dim,
            prot_dim=prot_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            comp_noise=comp_noise,
            prot_noise=prot_noise
        )
    elif task == 'effect':
        return EffectModelPretrained(
            comp_dim=comp_dim,
            prot_dim=prot_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            comp_noise=comp_noise,
            prot_noise=prot_noise
        )
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == "__main__":

    batch_size = 4
    comp_feat = torch.randn(batch_size, 768)
    prot_feat = torch.randn(batch_size, 1280)

    print("Testing BindingModelPretrained...")
    model = BindingModelPretrained()
    outputs = model(comp_feat, prot_feat)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Prob shape: {outputs['prob'].shape}")

    print("\nTesting EffectModelPretrained...")
    model = EffectModelPretrained()
    outputs = model(comp_feat, prot_feat)
    print(f"P(activation): {outputs['prob_activation'].squeeze()}")
    print(f"P(inhibition): {outputs['prob_inhibition'].squeeze()}")
    print(f"P(no_effect): {outputs['prob_no_effect'].squeeze()}")


    sum_prob = outputs['prob_activation'] + outputs['prob_inhibition'] + outputs['prob_no_effect']
    print(f"Sum of probabilities: {sum_prob.squeeze()}")

    print("\nAll tests passed!")
