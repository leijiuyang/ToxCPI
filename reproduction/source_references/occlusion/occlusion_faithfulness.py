#!/usr/bin/env python3
"""Attention-guided embedding masking for faithfulness checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from common import (
    EXPLANATION_DIR,
    build_shared_tensors,
    create_model,
    ensure_under_explanation,
    filter_dataframe_for_features,
    forward_model,
    get_device,
    load_config,
    load_pretrained_features,
    make_fixed_batch,
    resolve_output_path,
    seed_everything,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attention-guided occlusion.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--attention_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mask_ratios", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--n_random", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--positive_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def run_logit(model, batch: Dict, token_indices=None, residue_indices=None) -> float:
    import torch

    masked = dict(batch)
    masked["chem_tokens_feat"] = batch["chem_tokens_feat"].clone()
    masked["prot_residue_feat"] = batch["prot_residue_feat"].clone()
    if token_indices is not None and len(token_indices) > 0:
        token_indices = np.asarray(token_indices, dtype=np.int64).copy()
        masked["chem_tokens_feat"][0, token_indices, :] = 0.0
    if residue_indices is not None and len(residue_indices) > 0:
        residue_indices = np.asarray(residue_indices, dtype=np.int64).copy()
        masked["prot_residue_feat"][0, residue_indices, :] = 0.0
    with torch.no_grad():
        logits, _, _, _ = forward_model(model, masked)
    return float(logits[0].detach().cpu().item())


def signed_rank_pvalue(x: List[float], y: List[float]) -> float:
    if not x or not y:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(x, y).pvalue)
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    logger = setup_logger("occlusion_faithfulness")
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    config = load_config(args.config)
    attention_dir = Path(resolve_output_path(args.attention_dir))
    output_dir = args.output_dir or str(EXPLANATION_DIR / "outputs" / "occlusion" / attention_dir.name)
    output_dir = Path(ensure_under_explanation(resolve_output_path(output_dir)))
    index = pd.read_csv(attention_dir / "sample_index.csv")
    if args.positive_only and "label" in index.columns:
        index = index[index["label"] == 1]
    if args.limit:
        index = index.head(args.limit)
    if index.empty:
        raise RuntimeError("No samples selected for occlusion.")

    device = get_device(config)
    comp_feats, prot_feats = load_pretrained_features(config)
    df = index.copy()
    df = filter_dataframe_for_features(df, comp_feats, prot_feats)
    shared = build_shared_tensors(df, comp_feats, prot_feats, config, logger)
    model = create_model(config, device)

    records = []
    for row in df.itertuples(index=False):
        arrays = np.load(row.attention_npz_path)
        token_scores = arrays["token_importance"]
        residue_scores = arrays["residue_importance"]
        one = pd.DataFrame([row._asdict()])
        batch = make_fixed_batch(one, shared, config, device)
        original_logit = run_logit(model, batch)
        original_prob = sigmoid(original_logit)

        for axis, scores in [("token", token_scores), ("residue", residue_scores)]:
            n_valid = len(scores)
            high_order = np.argsort(scores)[::-1]
            low_order = np.argsort(scores)
            for ratio in args.mask_ratios:
                n_mask = max(1, min(n_valid, int(np.ceil(float(ratio) * n_valid))))
                strategies = {
                    "attention_guided": high_order[:n_mask],
                    "low_attention": low_order[:n_mask],
                }
                for strategy, indices in strategies.items():
                    if axis == "token":
                        masked_logit = run_logit(model, batch, token_indices=indices)
                    else:
                        masked_logit = run_logit(model, batch, residue_indices=indices)
                    records.append(
                        {
                            "sample_id": row.sample_id,
                            "axis": axis,
                            "strategy": strategy,
                            "mask_ratio": ratio,
                            "repeat": 0,
                            "n_mask": n_mask,
                            "original_logit": original_logit,
                            "masked_logit": masked_logit,
                            "logit_drop": original_logit - masked_logit,
                            "original_prob": original_prob,
                            "masked_prob": sigmoid(masked_logit),
                            "prob_drop": original_prob - sigmoid(masked_logit),
                        }
                    )
                for repeat in range(args.n_random):
                    indices = rng.choice(n_valid, size=n_mask, replace=False)
                    if axis == "token":
                        masked_logit = run_logit(model, batch, token_indices=indices)
                    else:
                        masked_logit = run_logit(model, batch, residue_indices=indices)
                    records.append(
                        {
                            "sample_id": row.sample_id,
                            "axis": axis,
                            "strategy": "random",
                            "mask_ratio": ratio,
                            "repeat": repeat,
                            "n_mask": n_mask,
                            "original_logit": original_logit,
                            "masked_logit": masked_logit,
                            "logit_drop": original_logit - masked_logit,
                            "original_prob": original_prob,
                            "masked_prob": sigmoid(masked_logit),
                            "prob_drop": original_prob - sigmoid(masked_logit),
                        }
                    )

    metrics = pd.DataFrame(records)
    metrics.to_csv(output_dir / "occlusion_sample_metrics.csv", index=False)

    summary_rows = []
    grouped = metrics.groupby(["axis", "mask_ratio"])
    for (axis, ratio), g in grouped:
        att = g[g["strategy"] == "attention_guided"].set_index("sample_id")["logit_drop"]
        rand = g[g["strategy"] == "random"].groupby("sample_id")["logit_drop"].mean()
        low = g[g["strategy"] == "low_attention"].set_index("sample_id")["logit_drop"]
        common = sorted(set(att.index) & set(rand.index))
        p_att_rand = signed_rank_pvalue(att.loc[common].tolist(), rand.loc[common].tolist()) if common else float("nan")
        summary_rows.append(
            {
                "axis": axis,
                "mask_ratio": ratio,
                "n_samples": len(common),
                "attention_guided_logit_drop_mean": float(att.mean()),
                "random_logit_drop_mean": float(rand.mean()),
                "low_attention_logit_drop_mean": float(low.mean()),
                "attention_vs_random_p": p_att_rand,
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "occlusion_summary.csv", index=False)
    logger.info("Occlusion complete: %d rows", len(metrics))


if __name__ == "__main__":
    main()
