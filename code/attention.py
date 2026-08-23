




import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

from config import model_config


class BilinearAttention(nn.Module):













    def __init__(
        self,
        atom_dim: int = model_config.hidden_dim,
        residue_dim: int = model_config.hidden_dim,
        hidden_dim: int = model_config.bilinear_hidden_dim,
        output_dim: int = model_config.interaction_dim,
        num_heads: int = 4,
        dropout: float = 0.1,
        return_head_attention: bool = False
    ):
        super().__init__()

        self.atom_dim = atom_dim
        self.residue_dim = residue_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.return_head_attention = return_head_attention



        self.W = nn.Parameter(torch.randn(num_heads, atom_dim, residue_dim))
        nn.init.xavier_uniform_(self.W.view(num_heads, -1))


        self.atom_proj = nn.Linear(atom_dim, hidden_dim)
        self.residue_proj = nn.Linear(residue_dim, hidden_dim)


        self.head_fusion = nn.Linear(num_heads * hidden_dim, output_dim)


        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_atom: torch.Tensor,
        H_residue: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:













        batch_size = H_atom.size(0)
        n_atoms = H_atom.size(1)
        n_residues = H_residue.size(1)





        transformed = torch.einsum('bna,had->bhnd', H_atom, self.W)


        A_multi = torch.matmul(transformed, H_residue.transpose(-1, -2).unsqueeze(1))


        A_multi_masked = A_multi
        if atom_mask is not None:

            A_multi_masked = A_multi_masked.masked_fill(
                atom_mask.unsqueeze(1).unsqueeze(-1), float('-inf')
            )
        if residue_mask is not None:

            A_multi_masked = A_multi_masked.masked_fill(
                residue_mask.unsqueeze(1).unsqueeze(1), float('-inf')
            )


        A = A_multi_masked.mean(dim=1)



        A_atom = F.softmax(A / math.sqrt(self.residue_dim), dim=-1)
        A_atom = A_atom.nan_to_num(0.0)
        A_atom = self.dropout(A_atom)


        A_residue = F.softmax(A.transpose(-1, -2) / math.sqrt(self.atom_dim), dim=-1)
        A_residue = A_residue.nan_to_num(0.0)
        A_residue = self.dropout(A_residue)



        h_atom_agg = torch.bmm(A_atom, H_residue)


        h_residue_agg = torch.bmm(A_residue, H_atom)


        h_atom_proj = self.atom_proj(h_atom_agg)
        h_residue_proj = self.residue_proj(h_residue_agg)


        if atom_mask is not None:
            atom_mask_float = (~atom_mask).unsqueeze(-1).float()
            h_atom_global = (h_atom_proj * atom_mask_float).sum(dim=1) / atom_mask_float.sum(dim=1).clamp(min=1)
        else:
            h_atom_global = h_atom_proj.mean(dim=1)

        if residue_mask is not None:
            residue_mask_float = (~residue_mask).unsqueeze(-1).float()
            h_residue_global = (h_residue_proj * residue_mask_float).sum(dim=1) / residue_mask_float.sum(dim=1).clamp(min=1)
        else:
            h_residue_global = h_residue_proj.mean(dim=1)


        h_interact = self.output_proj(
            torch.cat([h_atom_global, h_residue_global], dim=-1)
        )



        if self.return_head_attention:
            A_interpretable = F.softmax(A_multi_masked, dim=-1).nan_to_num(0.0)
        else:
            A_interpretable = F.softmax(A, dim=-1).nan_to_num(0.0)

        return h_interact, A_interpretable, h_atom_global, h_residue_global

    def get_interpretability(
        self,
        A: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:













        atom_importance = A.mean(dim=-1)


        residue_importance = A.mean(dim=-2)


        if atom_mask is not None:
            atom_importance = atom_importance.masked_fill(atom_mask, 0.0)
        if residue_mask is not None:
            residue_importance = residue_importance.masked_fill(residue_mask, 0.0)

        return atom_importance, residue_importance


class CrossAttention(nn.Module):








    def __init__(
        self,
        atom_dim: int = model_config.hidden_dim,
        residue_dim: int = model_config.hidden_dim,
        hidden_dim: int = model_config.bilinear_hidden_dim,
        output_dim: int = model_config.interaction_dim,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads


        self.q_proj = nn.Linear(atom_dim, hidden_dim)
        self.k_proj = nn.Linear(residue_dim, hidden_dim)
        self.v_proj = nn.Linear(residue_dim, hidden_dim)


        self.out_proj = nn.Linear(hidden_dim, hidden_dim)


        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H_atom: torch.Tensor,
        H_residue: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:











        batch_size = H_atom.size(0)
        n_atoms = H_atom.size(1)
        n_residues = H_residue.size(1)


        Q = self.q_proj(H_atom)
        K = self.k_proj(H_residue)
        V = self.v_proj(H_residue)


        Q = Q.view(batch_size, n_atoms, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_residues, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_residues, self.num_heads, self.head_dim).transpose(1, 2)


        A = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)



        if residue_mask is not None:
            A = A.masked_fill(residue_mask.unsqueeze(1).unsqueeze(2), float('-inf'))


        A = F.softmax(A, dim=-1)
        A = self.dropout(A)


        out = torch.matmul(A, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_atoms, self.hidden_dim)
        out = self.out_proj(out)


        if atom_mask is not None:
            mask_float = (~atom_mask).unsqueeze(-1).float()
            atom_global = (out * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            atom_global = out.mean(dim=1)

        if residue_mask is not None:
            mask_float = (~residue_mask).unsqueeze(-1).float()
            residue_global = (H_residue * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            residue_global = H_residue.mean(dim=1)


        h_interact = self.fusion(torch.cat([atom_global, residue_global], dim=-1))


        A_mean = A.mean(dim=1)

        return h_interact, A_mean, atom_global, residue_global


class InteractionModule(nn.Module):





    def __init__(
        self,
        attention_type: str = 'bilinear',
        atom_dim: int = model_config.hidden_dim,
        residue_dim: int = model_config.hidden_dim,
        hidden_dim: int = model_config.bilinear_hidden_dim,
        output_dim: int = model_config.interaction_dim,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention_type = attention_type

        if attention_type == 'bilinear':
            self.attention = BilinearAttention(
                atom_dim=atom_dim,
                residue_dim=residue_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        elif attention_type == 'cross':
            self.attention = CrossAttention(
                atom_dim=atom_dim,
                residue_dim=residue_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    def forward(
        self,
        H_atom: torch.Tensor,
        H_residue: torch.Tensor,
        atom_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:













        return self.attention(H_atom, H_residue, atom_mask, residue_mask)


if __name__ == "__main__":

    print("Testing Attention Modules...")

    batch_size = 4
    n_atoms = 32
    n_residues = 128
    hidden_dim = 256


    H_atom = torch.randn(batch_size, n_atoms, hidden_dim)
    H_residue = torch.randn(batch_size, n_residues, hidden_dim)


    print("\n1. Testing BilinearAttention...")
    bilinear_attn = BilinearAttention()
    h_interact, A, chem_emb, prot_emb = bilinear_attn(H_atom, H_residue)
    print(f"Interaction shape: {h_interact.shape}")
    print(f"Attention matrix shape: {A.shape}")
    print(f"Chem embedding shape: {chem_emb.shape}")
    print(f"Prot embedding shape: {prot_emb.shape}")


    atom_imp, res_imp = bilinear_attn.get_interpretability(A)
    print(f"Atom importance shape: {atom_imp.shape}")
    print(f"Residue importance shape: {res_imp.shape}")


    print("\n2. Testing CrossAttention (for ablation)...")
    cross_attn = CrossAttention()
    h_interact_cross, A_cross, chem_emb_c, prot_emb_c = cross_attn(H_atom, H_residue)
    print(f"Interaction shape: {h_interact_cross.shape}")
    print(f"Attention matrix shape: {A_cross.shape}")
    print(f"Chem embedding shape: {chem_emb_c.shape}")
    print(f"Prot embedding shape: {prot_emb_c.shape}")


    print("\n3. Testing InteractionModule...")
    interaction_bilinear = InteractionModule(attention_type='bilinear')
    interaction_cross = InteractionModule(attention_type='cross')

    h1, _, _, _ = interaction_bilinear(H_atom, H_residue)
    h2, _, _, _ = interaction_cross(H_atom, H_residue)

    print(f"Bilinear output: {h1.shape}")
    print(f"Cross output: {h2.shape}")

    print("\nAll attention tests completed!")
