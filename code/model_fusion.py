import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.3,
                 gate_min: float = 0.0, gate_max: float = 1.0):
        super().__init__()
        self.gate_min = gate_min
        self.gate_max = gate_max
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_domain: torch.Tensor, h_semantic: torch.Tensor) -> torch.Tensor:
        g_raw = self.gate(torch.cat([h_domain, h_semantic], dim=-1))

        g = self.gate_min + (self.gate_max - self.gate_min) * g_raw
        h_fused = g * h_domain + (1 - g) * h_semantic
        return self.dropout(self.layer_norm(h_fused))


class InteractionMatrixCNN(nn.Module):
    def __init__(
        self,
        num_heads: int,
        output_dim: int,
        hidden_channels: int = 16,
        dropout: float = 0.3
    ):
        super().__init__()
        mid_channels = hidden_channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(num_heads, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.ReLU()
        )
        self.proj = nn.Sequential(
            nn.Linear(mid_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        attention_maps: torch.Tensor,
        chem_mask: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if attention_maps.dim() != 4:
            raise ValueError("InteractionMatrixCNN expects attention maps shaped (B, heads, n_chem, n_prot)")

        valid_mask = None
        if chem_mask is not None or prot_mask is not None:
            batch, _, n_chem, n_prot = attention_maps.shape
            if chem_mask is None:
                chem_valid = torch.ones(batch, n_chem, dtype=torch.bool, device=attention_maps.device)
            else:
                chem_valid = ~chem_mask
            if prot_mask is None:
                prot_valid = torch.ones(batch, n_prot, dtype=torch.bool, device=attention_maps.device)
            else:
                prot_valid = ~prot_mask
            valid_mask = chem_valid.unsqueeze(-1) & prot_valid.unsqueeze(1)
            attention_maps = attention_maps.masked_fill(~valid_mask.unsqueeze(1), 0.0)

        x = self.conv(attention_maps)
        if valid_mask is not None:
            mask_float = valid_mask.unsqueeze(1).float()
            x = x * mask_float
            denom = mask_float.sum(dim=(2, 3)).clamp(min=1.0)
            pooled = x.sum(dim=(2, 3)) / denom
        else:
            pooled = x.mean(dim=(2, 3))

        return self.proj(pooled)


class MolecularGraphEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        h: torch.Tensor,
        chem_adj: torch.Tensor,
        chem_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if chem_adj is None:
            return h

        adj = chem_adj.float()
        batch, n_tokens, _ = adj.shape
        if chem_mask is None:
            valid = torch.ones(batch, n_tokens, dtype=torch.bool, device=h.device)
        else:
            valid = ~chem_mask

        valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2)
        adj = adj.masked_fill(~valid_pair, 0.0)

        eye = torch.eye(n_tokens, dtype=adj.dtype, device=adj.device).unsqueeze(0)
        adj = adj + eye * valid.unsqueeze(1).float()
        degree = adj.sum(dim=-1).clamp(min=1.0)
        norm_adj = adj / degree.unsqueeze(-1)

        out = h
        for linear, norm in zip(self.layers, self.norms):
            message = torch.bmm(norm_adj, out)
            update = self.dropout(self.activation(linear(message)))
            out = norm(out + update)

        return out.masked_fill(~valid.unsqueeze(-1), 0.0)


def masked_mean_pool(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    valid = (~mask).unsqueeze(-1).float()
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


class BindingModelFusion(nn.Module):
    def __init__(
        self,
        tox_dim: int = 812,
        chemberta_dim: int = 768,
        esm2_dim: int = 1280,
        hidden_dim: int = 128,
        interaction_dim: int = 256,
        dropout: float = 0.3,
        use_residue_attention: bool = False,
        gate_min: float = 0.0,
        gate_max: float = 1.0,
        use_toxprint: bool = True,
        use_multi_granularity: bool = False,
        use_interaction_cnn: bool = False,
        interaction_cnn_channels: int = 16,
        use_molecular_gcn: bool = False,
        gcn_layers: int = 2
    ):
        super().__init__()
        self.use_residue_attention = use_residue_attention
        self.use_toxprint = use_toxprint
        self.use_multi_granularity = use_multi_granularity
        self.use_interaction_cnn = use_interaction_cnn
        self.use_molecular_gcn = use_molecular_gcn
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        if use_toxprint:
            self.tox_encoder = nn.Sequential(
                nn.Linear(tox_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )

        self.sem_encoder = nn.Sequential(
            nn.Linear(chemberta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if use_toxprint:
            self.chem_fusion = GatedFusion(hidden_dim, dropout, gate_min, gate_max)

        if use_residue_attention:
            self.chem_token_proj = nn.Sequential(
                nn.Linear(chemberta_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            if use_molecular_gcn:
                self.molecular_gcn = MolecularGraphEncoder(
                    hidden_dim=hidden_dim,
                    num_layers=gcn_layers,
                    dropout=dropout
                )

            self.prot_residue_proj = nn.Sequential(
                nn.Linear(esm2_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            from attention import BilinearAttention
            self.interaction = BilinearAttention(
                atom_dim=hidden_dim,
                residue_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=interaction_dim,
                num_heads=4,
                dropout=dropout,
                return_head_attention=use_interaction_cnn
            )

            if use_interaction_cnn:
                self.interaction_cnn = InteractionMatrixCNN(
                    num_heads=4,
                    output_dim=interaction_dim,
                    hidden_channels=interaction_cnn_channels,
                    dropout=dropout
                )
                self.interaction_cnn_fusion = nn.Sequential(
                    nn.Linear(interaction_dim * 2, interaction_dim),
                    nn.LayerNorm(interaction_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )

            if use_multi_granularity:
                coarse_dim = hidden_dim

                self.chem_attn_pool = nn.Linear(hidden_dim, 1)
                self.prot_attn_pool = nn.Linear(hidden_dim, 1)

                self.coarse_interaction = nn.Sequential(
                    nn.Linear(hidden_dim * 3, coarse_dim),
                    nn.LayerNorm(coarse_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )

                self.coarse_proj = nn.Linear(coarse_dim, interaction_dim)

                self.granularity_gate = nn.Sequential(
                    nn.Linear(interaction_dim * 2, 1),
                    nn.Sigmoid()
                )

        else:
            self.prot_encoder = nn.Sequential(
                nn.Linear(esm2_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.interaction = nn.Sequential(
                nn.Linear(hidden_dim * 2, interaction_dim),
                nn.LayerNorm(interaction_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        tox_feat: torch.Tensor = None,
        chemberta_feat: torch.Tensor = None,
        esm2_feat: torch.Tensor = None,
        chem_tokens_feat: torch.Tensor = None,
        chem_mask: torch.Tensor = None,
        chem_adj: torch.Tensor = None,
        prot_residue_feat: torch.Tensor = None,
        prot_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.use_residue_attention and chem_tokens_feat is not None:
            H_chem, H_prot = self.encode_sequence_inputs(
                chem_tokens_feat=chem_tokens_feat,
                prot_residue_feat=prot_residue_feat,
                chem_mask=chem_mask,
                chem_adj=chem_adj
            )
            h_interact, attention, chem_emb, prot_emb = self.compute_local_interaction(
                H_chem=H_chem,
                H_prot=H_prot,
                chem_mask=chem_mask,
                prot_mask=prot_mask
            )
        else:
            h_sem = self.sem_encoder(chemberta_feat)
            if self.use_toxprint and tox_feat is not None:
                h_tox = self.tox_encoder(tox_feat)
                h_chem = self.chem_fusion(h_tox, h_sem)
            else:
                h_chem = h_sem
            h_prot = self.prot_encoder(esm2_feat)
            h_interact = self.interaction(torch.cat([h_chem, h_prot], dim=-1))
            attention = None
            chem_emb, prot_emb = None, None

        logits = self.classifier(h_interact).squeeze(-1)
        return logits, attention, chem_emb, prot_emb

    def encode_sequence_inputs(
        self,
        chem_tokens_feat: torch.Tensor,
        prot_residue_feat: torch.Tensor,
        chem_mask: Optional[torch.Tensor] = None,
        chem_adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H_chem = self.chem_token_proj(chem_tokens_feat)
        if self.use_molecular_gcn:
            H_chem = self.molecular_gcn(H_chem, chem_adj, chem_mask)
        H_prot = self.prot_residue_proj(prot_residue_feat)
        return H_chem, H_prot

    def compute_local_interaction(
        self,
        H_chem: torch.Tensor,
        H_prot: torch.Tensor,
        chem_mask: Optional[torch.Tensor] = None,
        prot_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_interact, attention, chem_emb, prot_emb = self.interaction(
            H_chem, H_prot,
            atom_mask=chem_mask, residue_mask=prot_mask
        )

        if self.use_interaction_cnn:
            z_cnn = self.interaction_cnn(attention, chem_mask, prot_mask)
            h_interact = self.interaction_cnn_fusion(
                torch.cat([h_interact, z_cnn], dim=-1)
            )
            attention = attention.mean(dim=1)

        if self.use_multi_granularity:
            chem_attn_w = self.chem_attn_pool(H_chem)
            if chem_mask is not None:
                chem_attn_w = chem_attn_w.masked_fill(chem_mask.unsqueeze(-1), float('-inf'))
            chem_attn_w = torch.softmax(chem_attn_w, dim=1).nan_to_num(0.0)
            c_global = (H_chem * chem_attn_w).sum(dim=1)

            prot_attn_w = self.prot_attn_pool(H_prot)
            if prot_mask is not None:
                prot_attn_w = prot_attn_w.masked_fill(prot_mask.unsqueeze(-1), float('-inf'))
            prot_attn_w = torch.softmax(prot_attn_w, dim=1).nan_to_num(0.0)
            p_global = (H_prot * prot_attn_w).sum(dim=1)

            z_coarse = self.coarse_interaction(
                torch.cat([c_global, p_global, c_global * p_global], dim=-1)
            )
            z_coarse = self.coarse_proj(z_coarse)

            gate = self.granularity_gate(
                torch.cat([h_interact, z_coarse], dim=-1)
            )
            h_interact = gate * h_interact + (1 - gate) * z_coarse

        return h_interact, attention, chem_emb, prot_emb

    def get_gate_weights(
        self,
        tox_feat: torch.Tensor,
        chemberta_feat: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            h_tox = self.tox_encoder(tox_feat)
            h_sem = self.sem_encoder(chemberta_feat)
            g = self.chem_fusion.gate(torch.cat([h_tox, h_sem], dim=-1))
        return g


class EffectModelFusion(BindingModelFusion):
    def __init__(
        self,
        *args,
        global_dim: Optional[int] = None,
        use_projected_global_branch: bool = True,
        use_raw_global_branch: bool = False,
        raw_global_dim: Optional[int] = None,
        raw_chem_dim: int = 768,
        raw_prot_dim: int = 1280,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not self.use_residue_attention:
            raise ValueError("EffectModelFusion requires use_residue_attention=True")

        interaction_dim = self.classifier[0].in_features
        hidden_dim = self.hidden_dim
        global_dim = global_dim or interaction_dim
        self.use_projected_global_branch = use_projected_global_branch
        self.use_raw_global_branch = use_raw_global_branch
        raw_global_dim = raw_global_dim or global_dim
        if not self.use_projected_global_branch and not self.use_raw_global_branch:
            raise ValueError("EffectModelFusion requires at least one global branch")

        if self.use_projected_global_branch:
            self.global_branch = nn.Sequential(
                nn.Linear(hidden_dim * 4, global_dim),
                nn.LayerNorm(global_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(global_dim, global_dim),
                nn.LayerNorm(global_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )

        classifier_input_dim = interaction_dim
        if self.use_projected_global_branch:
            classifier_input_dim += global_dim
        if self.use_raw_global_branch:
            self.raw_global_branch = nn.Sequential(
                nn.Linear(raw_chem_dim + raw_prot_dim, raw_global_dim),
                nn.LayerNorm(raw_global_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(raw_global_dim, raw_global_dim),
                nn.LayerNorm(raw_global_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )
            classifier_input_dim += raw_global_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        tox_feat: torch.Tensor = None,
        chemberta_feat: torch.Tensor = None,
        esm2_feat: torch.Tensor = None,
        chem_tokens_feat: torch.Tensor = None,
        chem_mask: torch.Tensor = None,
        chem_adj: torch.Tensor = None,
        prot_residue_feat: torch.Tensor = None,
        prot_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        H_chem, H_prot = self.encode_sequence_inputs(
            chem_tokens_feat=chem_tokens_feat,
            prot_residue_feat=prot_residue_feat,
            chem_mask=chem_mask,
            chem_adj=chem_adj
        )
        z_local, attention, chem_emb, prot_emb = self.compute_local_interaction(
            H_chem=H_chem,
            H_prot=H_prot,
            chem_mask=chem_mask,
            prot_mask=prot_mask
        )

        features = [z_local]
        if self.use_projected_global_branch:
            c_global = masked_mean_pool(H_chem, chem_mask)
            p_global = masked_mean_pool(H_prot, prot_mask)
            z_global = self.global_branch(
                torch.cat([
                    c_global,
                    p_global,
                    c_global * p_global,
                    torch.abs(c_global - p_global)
                ], dim=-1)
            )
            features.append(z_global)

        if self.use_raw_global_branch:
            raw_c_global = masked_mean_pool(chem_tokens_feat, chem_mask)
            raw_p_global = masked_mean_pool(prot_residue_feat, prot_mask)
            z_raw_global = self.raw_global_branch(
                torch.cat([raw_c_global, raw_p_global], dim=-1)
            )
            features.append(z_raw_global)

        logits = self.classifier(torch.cat(features, dim=-1)).squeeze(-1)
        return logits, attention, chem_emb, prot_emb


if __name__ == "__main__":
    print("Testing BindingModelFusion...")

    batch_size = 4
    tox_feat = torch.randn(batch_size, 812)
    chemberta_feat = torch.randn(batch_size, 768)
    esm2_feat = torch.randn(batch_size, 1280)

    model = BindingModelFusion(use_residue_attention=False)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Step 1 parameters: {total_params:,}")

    logits, attn, c_emb, p_emb = model(tox_feat, chemberta_feat, esm2_feat)
    print(f"Logits shape: {logits.shape}")
    print(f"Attention: {attn}")
    print(f"Chem emb: {c_emb}, Prot emb: {p_emb}")

    g = model.get_gate_weights(tox_feat, chemberta_feat)
    print(f"Gate weights: {g.squeeze()}")

    print("\nAll tests passed!")
