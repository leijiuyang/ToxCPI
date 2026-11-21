"""Strict evaluation on fixed CSV records and explicit precomputed entity tensors.

Uses stored chemical-name / UniProt feature keys, matching FusionDataset's
eligibility rule. Does not resplit, balance, canonicalize or retrain data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_fusion import BindingModelFusion, EffectModelFusion


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(task, checkpoint, device):
    kwargs = dict(chemberta_dim=768, esm2_dim=1280, hidden_dim=128 if task == "binding" else 256,
                  interaction_dim=256 if task == "binding" else 512, dropout=0.25 if task == "activation" else 0.3,
                  use_residue_attention=True, use_toxprint=False, use_multi_granularity=False)
    if task == "binding":
        model = BindingModelFusion(**kwargs)
    else:
        model = EffectModelFusion(**kwargs, use_projected_global_branch=False, use_raw_global_branch=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["binding", "activation", "inhibition"])
    parser.add_argument("--split", required=True, choices=["warm", "cold"])
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--entity-cache", required=True, type=Path, help="Trusted torch entity tensor cache; see reproduction guide")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="0 = all eligible records; positive = first N, representative validation only")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("limit must be nonnegative and batch-size positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    frame = pd.read_csv(args.test_csv, low_memory=False)
    for col in ("chemical_name", "uniprot_id", "label"):
        if col not in frame:
            raise ValueError(f"Fixed test CSV missing {col}")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Expected fixed binary labels 0/1")
    frame.insert(0, "source_record", np.arange(len(frame)))
    cache = torch.load(args.entity_cache, map_location="cpu", mmap=True, weights_only=True)
    if cache["chem_tensor"].shape[1:] != (64, 768) or cache["prot_tensor"].shape[1:] != (768, 1280):
        raise ValueError("Expected original max64/768, BerMol768/ESM2-1280 cache contract")
    eligible = frame.chemical_name.astype(str).isin(cache["chem_id2idx"]) & frame.uniprot_id.astype(str).isin(cache["prot_id2idx"])
    evaluated = frame.loc[eligible].copy()
    n_eligible = len(evaluated)
    if args.limit:
        evaluated = evaluated.head(args.limit).copy()
    if evaluated.empty:
        raise ValueError("No feature-eligible fixed test records")
    model = load_model(args.task, args.checkpoint, device)
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(evaluated), args.batch_size):
            batch = evaluated.iloc[start:start+args.batch_size]
            ci = torch.tensor([cache["chem_id2idx"][str(key)] for key in batch.chemical_name])
            pi = torch.tensor([cache["prot_id2idx"][str(key)] for key in batch.uniprot_id])
            cmask = torch.arange(64).unsqueeze(0) >= cache["chem_lengths"][ci].unsqueeze(1)
            pmask = torch.arange(768).unsqueeze(0) >= cache["prot_lengths"][pi].unsqueeze(1)
            logits, *_ = model(chem_tokens_feat=cache["chem_tensor"][ci].to(device), chem_mask=cmask.to(device),
                prot_residue_feat=cache["prot_tensor"][pi].to(device), prot_mask=pmask.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().reshape(-1).tolist())
    evaluated["raw_probability"] = probabilities
    y = evaluated.label.to_numpy()
    p = np.array(probabilities)
    pred = p >= 0.5
    metrics = {"AUROC": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        "AUPRC": float(average_precision_score(y, p)) if y.sum() else None,
        "accuracy": float(accuracy_score(y, pred)), "F1": float(f1_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(args.output_dir/"fixed_test_predictions.csv", index=False)
    frame.loc[~eligible, ["source_record", "chemical_name", "uniprot_id", "label"]].to_csv(args.output_dir/"feature_ineligible_records.csv", index=False)
    report = {"task": args.task, "split": args.split, "strict_load": True, "n_fixed_test_records": len(frame),
        "n_feature_eligible": n_eligible, "n_evaluated": len(evaluated), "record_selection": "first N eligible in stored order" if args.limit else "all eligible in stored order",
        "validation_scope": "representative fixed-test run; not full published metrics" if len(evaluated) < n_eligible else "full feature-eligible fixed-test evaluation; comparison to paper requires provenance audit",
        "decision_threshold": 0.5, "metrics": metrics, "device": str(device), "torch": torch.__version__,
        "inputs": {str(path): sha256(path) for path in (args.test_csv, args.checkpoint, args.entity_cache)},
        "implementation_sha256": sha256(Path(__file__))}
    (args.output_dir/"evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
