#!/usr/bin/env python3
"""Shared utilities for ToxCPI interpretability scripts.

All generated analysis artifacts should stay under test1/explanation.
The functions here import existing project code read-only.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
EXPLANATION_DIR = SCRIPT_DIR.parent
TEST1_DIR = EXPLANATION_DIR.parent
PROJECT_DIR = TEST1_DIR.parent

for path in (str(TEST1_DIR), str(TEST1_DIR / "experiments")):
    if path not in sys.path:
        sys.path.insert(0, path)


def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(name)


def resolve_path(value: Optional[str], base: Path = TEST1_DIR) -> Optional[str]:
    if value is None or value == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        if path.parts and path.parts[0] == "test1":
            path = PROJECT_DIR / path
        else:
            path = base / path
    return str(path)


def ensure_under_explanation(path: str) -> str:
    resolved = Path(path).resolve()
    root = EXPLANATION_DIR.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"Output path must be under {root}: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def resolve_output_path(value: str) -> str:
    """Resolve an output path while accepting both explanation/... and test1/explanation/... forms."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    parts = path.parts
    if parts and parts[0] == "test1":
        return str(PROJECT_DIR / path)
    return str(TEST1_DIR / path)


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return ""
    if text.lower() in {"true", "yes"}:
        return True
    if text.lower() in {"false", "no"}:
        return False
    if text.lower() in {"none", "null"}:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = _parse_scalar(value)
    return data


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        if cfg_path.parts and cfg_path.parts[0] == "test1":
            cfg_path = PROJECT_DIR / cfg_path
        else:
            cfg_path = TEST1_DIR / cfg_path
    if cfg_path.suffix.lower() == ".json":
        with cfg_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    try:
        import yaml  # type: ignore

        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a mapping: {cfg_path}")
        return loaded
    except ImportError:
        return _load_simple_yaml(cfg_path)


def update_config_from_args(config: Dict[str, Any], args: Any, keys: Iterable[str]) -> Dict[str, Any]:
    merged = dict(config)
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            merged[key] = value
    return merged


def get_device(config: Dict[str, Any]):
    import torch

    device = str(config.get("device", "auto"))
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_pickle(path: str) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def mol_from_smiles_for_bermol_mapping(smiles: str) -> Any:
    """Parse SMILES using the canonicalization step used before BerMol tokenization."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    smiles_canon = Chem.MolToSmiles(mol)
    return Chem.MolFromSmiles(smiles_canon)


def get_tokenized_atom_indices(mol: Any, radius: int = 1) -> List[int]:
    """Replay BerMol mol_to_sentence and return atoms that produce radius tokens."""
    from rdkit.Chem import AllChem

    info: Dict[Any, Any] = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    mol_atoms = [int(atom.GetIdx()) for atom in mol.GetAtoms()]
    atom_dic = {atom: {r: None for r in range(radius + 1)} for atom in mol_atoms}
    for element in info:
        for atom_idx, r in info[element]:
            atom_dic[int(atom_idx)][int(r)] = element
    return [atom for atom in mol_atoms if atom_dic[atom][radius] is not None]


def load_pretrained_features(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    feature_dir = Path(resolve_path(config.get("feature_dir"), TEST1_DIR) or (TEST1_DIR / "data/pretrained_features"))
    comp_path = feature_dir / str(config.get("compound_feature_file", "compound_features_bermol_tokens.pkl"))
    prot_path = feature_dir / str(config.get("protein_feature_file", "protein_features_esm2_residues.pkl"))
    return load_pickle(str(comp_path)), load_pickle(str(prot_path))


def read_split_dataframe(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    task = str(config.get("task", "binding"))
    split = str(config.get("split", "cold"))
    eval_split = str(config.get("eval_split", "test"))

    if task == "binding":
        if config.get("data_file"):
            df = pd.read_csv(resolve_path(config["data_file"], TEST1_DIR), low_memory=False)
        else:
            split_dir = resolve_path(
                config.get("split_dir", "data/interaction_binding/splits"),
                TEST1_DIR,
            )
            path = Path(split_dir) / f"{eval_split}_{split}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path, low_memory=False)
            logger.info("Loaded %s: %d samples", path.name, len(df))
        if bool(config.get("canonicalize_smiles", False)) and "smiles_clean" not in df.columns and "smiles" in df.columns:
            from utils import canonicalize_smiles

            df = df.copy()
            df["smiles_clean"] = df["smiles"].apply(canonicalize_smiles)
        return df.reset_index(drop=True)

    data_file = config.get("data_file")
    if data_file:
        return pd.read_csv(resolve_path(data_file, TEST1_DIR), low_memory=False).reset_index(drop=True)

    data_dir = resolve_path(
        config.get("data_dir", "data/interaction_type_1to1_targetaware"),
        TEST1_DIR,
    )
    path = Path(data_dir) / f"effect_{task}_{split}_{eval_split}.csv"
    return pd.read_csv(path, low_memory=False).reset_index(drop=True)


def filter_dataframe_for_features(
    df: pd.DataFrame,
    comp_feats: Dict[str, Any],
    prot_feats: Dict[str, Any],
    chem_col: str = "chemical_name",
    prot_col: str = "uniprot_id",
) -> pd.DataFrame:
    valid = df.apply(
        lambda row: str(row[chem_col]) in comp_feats and str(row[prot_col]) in prot_feats,
        axis=1,
    )
    return df.loc[valid].reset_index(drop=True)


def limit_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    out = df
    if bool(config.get("positive_only", False)) and "label" in out.columns:
        out = out[out["label"] == 1]
    if config.get("case_list"):
        case_path = Path(resolve_path(config["case_list"], TEST1_DIR))
        cases = pd.read_csv(case_path, low_memory=False)
        keys = [k for k in ("chemical_name", "uniprot_id", "task") if k in cases.columns and k in out.columns]
        if keys:
            out = out.merge(cases[keys].drop_duplicates(), on=keys, how="inner")
    limit = config.get("limit")
    if limit is not None:
        out = out.head(int(limit))
    return out.reset_index(drop=True)


def subset_feature_dicts(
    df: pd.DataFrame,
    comp_feats: Dict[str, Any],
    prot_feats: Dict[str, Any],
    chem_col: str = "chemical_name",
    prot_col: str = "uniprot_id",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    chem_ids = sorted({str(x) for x in df[chem_col].dropna().tolist()})
    prot_ids = sorted({str(x) for x in df[prot_col].dropna().tolist()})
    return (
        {key: comp_feats[key] for key in chem_ids if key in comp_feats},
        {key: prot_feats[key] for key in prot_ids if key in prot_feats},
    )


def build_chem_smiles_map(df: pd.DataFrame, chem_col: str = "chemical_name") -> Dict[str, str]:
    if "smiles" not in df.columns:
        return {}
    source = "smiles_clean" if "smiles_clean" in df.columns else "smiles"
    return (
        df.dropna(subset=[chem_col, source])
        .drop_duplicates(chem_col)
        .set_index(chem_col)[source]
        .astype(str)
        .to_dict()
    )


def build_shared_tensors(
    df: pd.DataFrame,
    comp_feats: Dict[str, Any],
    prot_feats: Dict[str, Any],
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    from train_fusion import FusionDataset, attach_site_label_tensors

    logger = logger or logging.getLogger(__name__)
    chem_col = str(config.get("chem_col", "chemical_name"))
    prot_col = str(config.get("prot_col", "uniprot_id"))
    comp_sub, prot_sub = subset_feature_dicts(df, comp_feats, prot_feats, chem_col, prot_col)
    logger.info("Building shared tensors for selected entities: %d chemicals, %d proteins", len(comp_sub), len(prot_sub))
    shared = FusionDataset.build_entity_tensors(
        comp_sub,
        prot_sub,
        max_chem_tokens=int(config.get("max_chem_tokens", 64)),
        max_protein_length=int(config.get("max_protein_length", 768)),
        build_chem_adj=bool(config.get("molecular_gcn", False)),
        chem_smiles_map=build_chem_smiles_map(df, chem_col),
    )
    site_path = config.get("site_annotation_path")
    if site_path:
        shared = attach_site_label_tensors(
            shared,
            resolve_path(site_path, TEST1_DIR),
            max_protein_length=int(config.get("max_protein_length", 768)),
        )
    return shared


def create_model(config: Dict[str, Any], device):
    import torch
    from model_fusion import BindingModelFusion, EffectModelFusion

    task = str(config.get("task", "binding"))
    hidden_dim = int(config.get("hidden_dim", 128))
    interaction_dim = int(config.get("interaction_dim", hidden_dim * 2))
    dropout = float(config.get("dropout", 0.3))
    model_kwargs = dict(
        chemberta_dim=int(config.get("chemberta_dim", 768)),
        esm2_dim=int(config.get("esm2_dim", 1280)),
        hidden_dim=hidden_dim,
        interaction_dim=interaction_dim,
        dropout=dropout,
        use_residue_attention=True,
        use_toxprint=False,
        use_multi_granularity=bool(config.get("multi_granularity", False)),
        use_interaction_cnn=bool(config.get("interaction_cnn", False)),
        interaction_cnn_channels=int(config.get("interaction_cnn_channels", 16)),
        use_molecular_gcn=bool(config.get("molecular_gcn", False)),
        gcn_layers=int(config.get("gcn_layers", 2)),
    )
    if task in {"activation", "inhibition"} and (
        bool(config.get("effect_dual_branch", False)) or bool(config.get("effect_raw_global_branch", False))
    ):
        model_kwargs["use_projected_global_branch"] = bool(config.get("effect_dual_branch", False))
        model_kwargs["use_raw_global_branch"] = bool(config.get("effect_raw_global_branch", False))
        model = EffectModelFusion(**model_kwargs)
    else:
        model = BindingModelFusion(**model_kwargs)

    checkpoint = resolve_path(config.get("checkpoint"), TEST1_DIR)
    if checkpoint:
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # Strict=False allows older checkpoints with harmless metadata, but real
            # architecture mismatches should still be visible to the caller.
            logging.getLogger(__name__).warning("Checkpoint load missing=%s unexpected=%s", missing, unexpected)
    model.to(device)
    model.eval()
    return model


def make_fixed_batch(
    rows: pd.DataFrame,
    shared: Dict[str, Any],
    config: Dict[str, Any],
    device,
) -> Dict[str, Any]:
    import torch

    chem_col = str(config.get("chem_col", "chemical_name"))
    prot_col = str(config.get("prot_col", "uniprot_id"))
    label_col = str(config.get("label_col", "label"))
    chem_id2idx = shared["chem_id2idx"]
    prot_id2idx = shared["prot_id2idx"]
    chem_idx = torch.tensor([chem_id2idx[str(x)] for x in rows[chem_col].tolist()], dtype=torch.long)
    prot_idx = torch.tensor([prot_id2idx[str(x)] for x in rows[prot_col].tolist()], dtype=torch.long)
    chem_len = shared["chem_lengths"][chem_idx]
    prot_len = shared["prot_lengths"][prot_idx]
    max_chem = shared["chem_tensor"].size(1)
    max_prot = shared["prot_tensor"].size(1)
    chem_pos = torch.arange(max_chem)
    prot_pos = torch.arange(max_prot)

    batch = {
        "chem_tokens_feat": shared["chem_tensor"][chem_idx].to(device),
        "chem_mask": (chem_pos.unsqueeze(0) >= chem_len.unsqueeze(1)).to(device),
        "prot_residue_feat": shared["prot_tensor"][prot_idx].to(device),
        "prot_mask": (prot_pos.unsqueeze(0) >= prot_len.unsqueeze(1)).to(device),
        "chem_idx": chem_idx,
        "prot_idx": prot_idx,
        "chem_lengths": chem_len,
        "prot_lengths": prot_len,
    }
    if label_col in rows.columns:
        batch["label"] = torch.tensor(rows[label_col].astype(float).to_numpy(), dtype=torch.float32).to(device)
    if shared.get("chem_adj_tensor") is not None:
        batch["chem_adj"] = shared["chem_adj_tensor"][chem_idx].to(device)
    if shared.get("prot_site_tensor") is not None:
        batch["prot_site_label"] = shared["prot_site_tensor"][prot_idx].to(device)
    return batch


def forward_model(model, batch: Dict[str, Any]):
    return model(
        chem_tokens_feat=batch["chem_tokens_feat"],
        chem_mask=batch["chem_mask"],
        chem_adj=batch.get("chem_adj"),
        prot_residue_feat=batch["prot_residue_feat"],
        prot_mask=batch["prot_mask"],
    )


def compute_raw_attention(model, batch: Dict[str, Any]) -> Tuple[Any, Any, Any, Any, Any]:
    """Return raw scores and bidirectional normalized attention.

    Shapes:
      raw_score: (B, n_chem, n_prot)
      token_to_residue: (B, n_chem, n_prot)
      residue_to_token: (B, n_prot, n_chem)
      residue_importance: (B, n_prot)
      token_importance: (B, n_chem)
    """
    import torch
    import torch.nn.functional as F

    h_chem, h_prot = model.encode_sequence_inputs(
        chem_tokens_feat=batch["chem_tokens_feat"],
        prot_residue_feat=batch["prot_residue_feat"],
        chem_mask=batch["chem_mask"],
        chem_adj=batch.get("chem_adj"),
    )
    interaction = model.interaction
    raw_multi = torch.matmul(
        torch.einsum("bna,had->bhnd", h_chem, interaction.W),
        h_prot.transpose(-1, -2).unsqueeze(1),
    )
    raw_multi = raw_multi.masked_fill(batch["chem_mask"].unsqueeze(1).unsqueeze(-1), float("-inf"))
    raw_multi = raw_multi.masked_fill(batch["prot_mask"].unsqueeze(1).unsqueeze(1), float("-inf"))
    raw_score = raw_multi.mean(dim=1)

    token_to_residue = F.softmax(raw_score, dim=-1).nan_to_num(0.0)
    residue_to_token = F.softmax(raw_score.transpose(-1, -2), dim=-1).nan_to_num(0.0)

    chem_valid = (~batch["chem_mask"]).float()
    prot_valid = (~batch["prot_mask"]).float()
    residue_importance = (token_to_residue * chem_valid.unsqueeze(-1)).sum(dim=1)
    residue_importance = residue_importance / chem_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    residue_importance = residue_importance * prot_valid

    token_importance = (residue_to_token * prot_valid.unsqueeze(-1)).sum(dim=1)
    token_importance = token_importance / prot_valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    token_importance = token_importance * chem_valid
    return raw_score, token_to_residue, residue_to_token, residue_importance, token_importance


def _masked_mean_pool_tensor(x: Any, mask: Optional[Any] = None) -> Any:
    if mask is None:
        return x.mean(dim=1)
    valid = (~mask).unsqueeze(-1).float()
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def compute_effect_branch_reliance(model, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Estimate local-vs-global reliance for EffectModelFusion samples.

    The effect classifier is nonlinear after branch concatenation, so this is an
    ablation-style attribution rather than an exact additive decomposition:

      local_delta  = full_logit - logit(z_local=0, z_global)
      global_delta = full_logit - logit(z_local, z_global=0)

    Larger absolute delta means the prediction is more sensitive to that branch.
    """
    if not (
        hasattr(model, "use_projected_global_branch")
        or hasattr(model, "use_raw_global_branch")
    ):
        return None

    import torch

    H_chem, H_prot = model.encode_sequence_inputs(
        chem_tokens_feat=batch["chem_tokens_feat"],
        prot_residue_feat=batch["prot_residue_feat"],
        chem_mask=batch["chem_mask"],
        chem_adj=batch.get("chem_adj"),
    )
    z_local, _, _, _ = model.compute_local_interaction(
        H_chem=H_chem,
        H_prot=H_prot,
        chem_mask=batch["chem_mask"],
        prot_mask=batch["prot_mask"],
    )

    features = [z_local]
    global_features = []
    branch_names = ["local"]

    if bool(getattr(model, "use_projected_global_branch", False)):
        c_global = _masked_mean_pool_tensor(H_chem, batch["chem_mask"])
        p_global = _masked_mean_pool_tensor(H_prot, batch["prot_mask"])
        z_global = model.global_branch(
            torch.cat(
                [
                    c_global,
                    p_global,
                    c_global * p_global,
                    torch.abs(c_global - p_global),
                ],
                dim=-1,
            )
        )
        features.append(z_global)
        global_features.append(z_global)
        branch_names.append("projected_global")

    if bool(getattr(model, "use_raw_global_branch", False)):
        raw_c_global = _masked_mean_pool_tensor(batch["chem_tokens_feat"], batch["chem_mask"])
        raw_p_global = _masked_mean_pool_tensor(batch["prot_residue_feat"], batch["prot_mask"])
        z_raw_global = model.raw_global_branch(torch.cat([raw_c_global, raw_p_global], dim=-1))
        features.append(z_raw_global)
        global_features.append(z_raw_global)
        branch_names.append("raw_global")

    if not global_features:
        return None

    def classify(parts: List[Any]) -> Any:
        return model.classifier(torch.cat(parts, dim=-1)).squeeze(-1)

    full_logit = classify(features)
    local_ablated_logit = classify([torch.zeros_like(z_local)] + features[1:])
    global_zero_features = [torch.zeros_like(x) for x in global_features]
    global_ablated_logit = classify([z_local] + global_zero_features)

    local_delta = full_logit - local_ablated_logit
    global_delta = full_logit - global_ablated_logit
    denom = local_delta.abs() + global_delta.abs() + 1e-9
    local_fraction = local_delta.abs() / denom
    global_fraction = global_delta.abs() / denom

    per_global_delta = {}
    offset = 1
    for name, feat in zip(branch_names[1:], global_features):
        parts = list(features)
        parts[offset] = torch.zeros_like(feat)
        per_global_delta[f"{name}_delta"] = full_logit - classify(parts)
        offset += 1

    return {
        "branch_names": branch_names,
        "full_logit": full_logit,
        "local_ablated_logit": local_ablated_logit,
        "global_ablated_logit": global_ablated_logit,
        "local_delta": local_delta,
        "global_delta": global_delta,
        "local_fraction": local_fraction,
        "global_fraction": global_fraction,
        **per_global_delta,
    }


def parse_position_list(value: Any) -> List[int]:
    if isinstance(value, list):
        return [int(x) for x in value]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, (list, tuple, set)):
        return []
    return [int(x) for x in parsed]


def load_site_annotation(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    resolved = resolve_path(path, TEST1_DIR)
    if not resolved or not os.path.exists(resolved):
        return None
    ann = pd.read_csv(resolved, low_memory=False)
    if "Entry" in ann.columns:
        ann = ann.set_index("Entry")
    return ann


def get_site_positions(site_ann: Optional[pd.DataFrame], uniprot_id: str, max_len: int) -> List[int]:
    if site_ann is None or uniprot_id not in site_ann.index:
        return []
    row = site_ann.loc[uniprot_id]
    cols = [
        "filtered_binding_positions_truncated",
        "actsite_positions_truncated",
        "pdbe_strict_positions_truncated",
    ]
    positions = set()
    for col in cols:
        if col in site_ann.columns:
            positions.update(parse_position_list(row[col]))
    return sorted(p for p in positions if 1 <= p <= max_len)


def top_items(values: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if values.size == 0:
        return []
    k = min(int(k), int(values.size))
    order = np.argsort(values)[::-1][:k]
    return [(int(i), float(values[i])) for i in order]


def safe_json_dump(data: Dict[str, Any], path: str) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=convert)
