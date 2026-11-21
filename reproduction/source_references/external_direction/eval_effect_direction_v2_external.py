#!/usr/bin/env python3
"""External drug-annotation effect-direction validation for ToxCPI.

This script evaluates activation-vs-inhibition direction discrimination on
external drug annotation pairs that were not used as ToxCPI effect positives.
It uses the cold activation/inhibition checkpoints, calibrates each channel on
the internal cold validation split, and reports target-wise DTIAM-style metrics.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
        break

import dill as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE = Path("/home/data/leijiuyang/test1")
OUT = BASE / "paper_draft/results_figs/task2_effect_direction_v2_external_screen"
DATA = BASE / "data/interaction_type_1to1_targetaware"
FEATURE_DIR = BASE / "data/pretrained_features"
CKPT_DIR = BASE / "checkpoints"
BERMOL_DIR = Path("/home/data/leijiuyang/tipFormer2025-main/model/BerMol")

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "experiments"))
from model_fusion import EffectModelFusion  # noqa: E402
from train_fusion import FusionDataset, seq_collate_fn  # noqa: E402

PANEL_7NR = [
    ("P10275", "AR"),
    ("P37231", "PPARG"),
    ("P04150", "NR3C1"),
    ("P03372", "ESR1"),
    ("P06401", "PGR"),
    ("Q92731", "ESR2"),
    ("P11473", "VDR"),
]

CKPTS = {
    "activation": CKPT_DIR / "effect_activation_cold_rawglobal_v3_rawglobal_local_control_1to1_targetaware_20260507_seed42_best.pth",
    "inhibition": CKPT_DIR / "effect_inhibition_cold_rawglobal_v3_rawglobal_local_control_1to1_targetaware_20260507_rerun_gpu0_seed42_best.pth",
}
DROPOUT = {"activation": 0.25, "inhibition": 0.30}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["7nr", "all56"], default="7nr")
    ap.add_argument("--min-dir", type=int, default=10, help="Minimum count in each direction for --panel all56.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="Debug limit after panel filtering.")
    ap.add_argument("--extract-missing-bermol", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    return ap.parse_args()


def output_suffix(panel_name: str) -> str:
    return "" if panel_name == "7nr" else f"_{panel_name}"


def external_feature_path(panel_name: str) -> Path:
    return OUT / ("external_bermol_tokens_7nr.pkl" if panel_name == "7nr" else "external_bermol_tokens_all56.pkl")


def load_panel(panel_name: str, min_dir: int) -> list[tuple[str, str]]:
    if panel_name == "7nr":
        return PANEL_7NR
    counts = pd.read_csv(OUT / "target_counts_pairclean_strict_dual_cold.csv")
    counts = counts[counts["min_dir"] >= min_dir].copy()
    counts = counts.sort_values(["min_dir", "total", "gene_symbol"], ascending=[False, False, True])
    panel = list(zip(counts["uniprot_id"].astype(str), counts["gene_symbol"].astype(str)))
    if not panel:
        raise RuntimeError(f"No targets passed min_dir >= {min_dir}")
    return panel


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_features(panel_name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    comp = load_pickle(FEATURE_DIR / "compound_features_bermol_tokens.pkl")
    for ext_path in [OUT / "external_bermol_tokens_7nr.pkl", OUT / "external_bermol_tokens_all56.pkl"]:
        if ext_path.exists():
            comp.update(load_pickle(ext_path))
    prot = load_pickle(FEATURE_DIR / "protein_features_esm2_residues.pkl")
    return comp, prot


def panel_external_df(panel: list[tuple[str, str]], limit: int | None = None) -> pd.DataFrame:
    path = OUT / "external_drug_effect_candidates_pairclean_strict_dual_cold.csv"
    df = pd.read_csv(path, low_memory=False)
    panel_ids = [u for u, _ in panel]
    df = df[df["uniprot_id"].isin(panel_ids)].copy()
    df = df[df["effect_type"].isin(["agonist", "antagonist"])].copy()
    df["true_direction"] = np.where(df["effect_type"] == "agonist", "activation", "inhibition")
    df["label"] = 1
    df = df.drop_duplicates(["smiles", "uniprot_id", "effect_type"]).reset_index(drop=True)
    if limit is not None:
        df = df.head(limit).copy()
    return df


def extract_missing_bermol(df: pd.DataFrame, device: torch.device, panel_name: str) -> None:
    existing = load_pickle(FEATURE_DIR / "compound_features_bermol_tokens.pkl")
    ext_path = external_feature_path(panel_name)
    external = load_pickle(ext_path) if ext_path.exists() else {}
    combined_keys = set(existing) | set(external)
    for other_path in [OUT / "external_bermol_tokens_7nr.pkl", OUT / "external_bermol_tokens_all56.pkl"]:
        if other_path.exists() and other_path != ext_path:
            combined_keys |= set(load_pickle(other_path))
    chems = df[["chemical_name", "smiles"]].dropna(subset=["chemical_name", "smiles"])
    missing = chems[~chems["chemical_name"].astype(str).isin(combined_keys)].drop_duplicates("chemical_name")
    print(f"BerMol feature coverage: need {len(missing)} new chemicals for {panel_name} external panel.")
    if len(missing) == 0:
        return

    sys.path.append(str(BERMOL_DIR))
    model_path = BERMOL_DIR / "BerMolModel_base.pkl"
    original_torch_load = torch.load
    def _torch_load_map_location(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        return original_torch_load(*args, **kwargs)
    torch.load = _torch_load_map_location
    with model_path.open("rb") as f:
        try:
            comp_model = pkl.load(f)
        finally:
            torch.load = original_torch_load
    comp_model.model.to(device)
    comp_model.model.eval()

    new_feat = {}
    failed = []
    for _, row in tqdm(missing.iterrows(), total=len(missing), desc="BerMol external"):
        cid = str(row["chemical_name"])
        smi = str(row["smiles"])
        try:
            with torch.no_grad():
                output = comp_model.transform(smi, str(device))
            new_feat[cid] = output[0][0].detach().cpu().numpy().astype(np.float16)
        except Exception as exc:
            failed.append((cid, smi, str(exc)[:160]))

    external.update(new_feat)
    save_pickle(external, ext_path)
    if failed:
        pd.DataFrame(failed, columns=["chemical_name", "smiles", "error"]).to_csv(
            OUT / f"external_bermol_failed_{panel_name}.csv", index=False
        )
    print(f"Saved external BerMol features: {len(external)} total, {len(new_feat)} new, {len(failed)} failed.")


def subset_features(df_list: Iterable[pd.DataFrame], comp: Dict[str, np.ndarray], prot: Dict[str, np.ndarray]):
    needed_chems = set()
    needed_prots = set()
    for df in df_list:
        needed_chems |= set(df["chemical_name"].astype(str))
        needed_prots |= set(df["uniprot_id"].astype(str))
    comp_sub = {k: comp[k] for k in needed_chems if k in comp}
    prot_sub = {k: prot[k] for k in needed_prots if k in prot}
    return comp_sub, prot_sub


def load_model(task: str, device: torch.device) -> EffectModelFusion:
    model = EffectModelFusion(
        chemberta_dim=768,
        esm2_dim=1280,
        hidden_dim=256,
        interaction_dim=512,
        dropout=DROPOUT[task],
        use_residue_attention=True,
        use_toxprint=False,
        use_projected_global_branch=False,
        use_raw_global_branch=True,
    ).to(device)
    model.load_state_dict(torch.load(CKPTS[task], map_location=device, weights_only=True))
    model.eval()
    return model


def make_dataset(df: pd.DataFrame, comp_feats, prot_feats, shared):
    return FusionDataset(
        df,
        comp_feats,
        prot_feats,
        "chemical_name",
        "uniprot_id",
        "label",
        use_residue_attention=True,
        shared_tensors=shared,
    )


def score_logits(model, df: pd.DataFrame, comp_feats, prot_feats, shared, device, batch_size: int) -> np.ndarray:
    before = len(df)
    ds = make_dataset(df, comp_feats, prot_feats, shared)
    if len(ds) != before:
        raise RuntimeError(f"Feature filtering changed row count: {len(ds)}/{before}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=seq_collate_fn)
    logits = []
    with torch.no_grad():
        for batch in loader:
            out, _, _, _ = model(
                chem_tokens_feat=batch["chem_tokens_feat"].to(device),
                chem_mask=batch["chem_mask"].to(device),
                prot_residue_feat=batch["prot_residue_feat"].to(device),
                prot_mask=batch["prot_mask"].to(device),
            )
            logits.append(out.detach().cpu().numpy())
    return np.concatenate(logits)


def calibration_df(task: str) -> pd.DataFrame:
    path = DATA / f"effect_{task}_cold_val.csv"
    df = pd.read_csv(path, low_memory=False).copy()
    df["label"] = df["label"].astype(int)
    return df


def fit_platt(logits: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(logits.reshape(-1, 1), labels.astype(int))
    return clf


def cal_prob(clf: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    p = clf.predict_proba(logits.reshape(-1, 1))[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


def compute_metrics(scores: pd.DataFrame, panel: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for uniprot, gene in panel:
        sub = scores[scores["uniprot_id"] == uniprot].copy()
        if sub.empty:
            continue
        y = (sub["true_direction"] == "activation").astype(int).to_numpy()
        pred = (sub["S_direction"].to_numpy() > 0).astype(int)
        n_act = int(y.sum())
        n_inh = int((1 - y).sum())
        rows.append(
            {
                "uniprot_id": uniprot,
                "gene": gene,
                "n_total": len(sub),
                "n_activation": n_act,
                "n_inhibition": n_inh,
                "accuracy": accuracy_score(y, pred),
                "activation_recall": float(((pred == 1) & (y == 1)).sum() / n_act) if n_act else np.nan,
                "inhibition_recall": float(((pred == 0) & (y == 0)).sum() / n_inh) if n_inh else np.nan,
                "direction_auroc": roc_auc_score(y, sub["S_direction"].to_numpy()) if len(set(y)) == 2 else np.nan,
                "correct": int((pred == y).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame, panel: list[tuple[str, str]], out_path: Path) -> None:
    order = [g for _, g in panel if g in set(metrics["gene"])]
    met = metrics.set_index("gene").loc[order].reset_index()
    fig_width = max(12, min(28, 0.34 * len(met) + 8))
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 8), dpi=180)
    panels = [
        ("accuracy", "Accuracy", axes[0, 0]),
        ("activation_recall", "Recall of activation", axes[0, 1]),
        ("inhibition_recall", "Recall of inhibition", axes[1, 0]),
        ("direction_auroc", "Direction AUROC", axes[1, 1]),
    ]
    colors = ["#253494", "#225aa6", "#1f82b9", "#37acc3", "#48b9c3", "#ceecb3", "#e6f5b2", "#e8edf2"]
    for key, title, ax in panels:
        vals = met[key].to_numpy()
        bars = ax.bar(met["gene"], vals, color=colors[:len(met)], width=0.72)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=60, labelsize=7 if len(met) > 20 else 9)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        for bar, val, (_, row) in zip(bars, vals, met.iterrows()):
            if key == "accuracy":
                label = f"{row['correct']}/{row['n_total']}"
            elif key == "activation_recall":
                label = f"{int(round(row[key] * row['n_activation']))}/{row['n_activation']}"
            elif key == "inhibition_recall":
                label = f"{int(round(row[key] * row['n_inhibition']))}/{row['n_inhibition']}"
            else:
                label = f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2, min(val + 0.035, 1.02), label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    panel = load_panel(args.panel, args.min_dir)
    suffix = output_suffix(args.panel)
    print(f"Panel: {args.panel} ({len(panel)} targets)")

    ext_df = panel_external_df(panel, args.limit)
    print(f"External rows after panel filtering: {len(ext_df)}")
    if args.extract_missing_bermol:
        extract_missing_bermol(ext_df, device, args.panel)
        if args.extract_only:
            return

    comp_feats, prot_feats = load_features(args.panel)
    val_act = calibration_df("activation")
    val_inh = calibration_df("inhibition")
    comp_sub, prot_sub = subset_features([ext_df, val_act, val_inh], comp_feats, prot_feats)

    missing_ext = ext_df[
        ~ext_df["chemical_name"].astype(str).isin(comp_sub)
        | ~ext_df["uniprot_id"].astype(str).isin(prot_sub)
    ]
    if len(missing_ext):
        missing_ext.to_csv(OUT / f"missing_features_external{suffix}.csv", index=False)
        raise RuntimeError(f"External rows missing features: {len(missing_ext)}. Run with --extract-missing-bermol first.")

    shared = FusionDataset.build_entity_tensors(comp_sub, prot_sub, max_chem_tokens=64, max_protein_length=768)

    model_act = load_model("activation", device)
    model_inh = load_model("inhibition", device)

    print("Scoring cold validation sets for calibration ...", flush=True)
    val_act_logits = score_logits(model_act, val_act, comp_sub, prot_sub, shared, device, args.batch_size)
    val_inh_logits = score_logits(model_inh, val_inh, comp_sub, prot_sub, shared, device, args.batch_size)
    cal_act = fit_platt(val_act_logits, val_act["label"].to_numpy())
    cal_inh = fit_platt(val_inh_logits, val_inh["label"].to_numpy())

    print(f"Scoring external {args.panel} validation set ...", flush=True)
    ext_act_logits = score_logits(model_act, ext_df, comp_sub, prot_sub, shared, device, args.batch_size)
    ext_inh_logits = score_logits(model_inh, ext_df, comp_sub, prot_sub, shared, device, args.batch_size)

    scores = ext_df.copy()
    scores["logit_activation_raw"] = ext_act_logits
    scores["logit_inhibition_raw"] = ext_inh_logits
    scores["P_activation_raw"] = 1.0 / (1.0 + np.exp(-ext_act_logits))
    scores["P_inhibition_raw"] = 1.0 / (1.0 + np.exp(-ext_inh_logits))
    scores["P_activation_cal"] = cal_prob(cal_act, ext_act_logits)
    scores["P_inhibition_cal"] = cal_prob(cal_inh, ext_inh_logits)
    scores["S_direction"] = np.log(scores["P_activation_cal"] / (1 - scores["P_activation_cal"])) - np.log(
        scores["P_inhibition_cal"] / (1 - scores["P_inhibition_cal"])
    )
    scores["pred_direction"] = np.where(scores["S_direction"] > 0, "activation", "inhibition")
    scores["correct"] = scores["pred_direction"] == scores["true_direction"]

    metrics = compute_metrics(scores, panel)
    scores.to_csv(OUT / f"effect_direction_scores_v2_external{suffix}.csv", index=False)
    metrics.to_csv(OUT / f"effect_direction_metrics_v2_external{suffix}.csv", index=False)
    with (OUT / f"platt_calibration_params_v2_external{suffix}.txt").open("w") as f:
        f.write(f"activation coef={cal_act.coef_[0,0]:.8g} intercept={cal_act.intercept_[0]:.8g}\\n")
        f.write(f"inhibition coef={cal_inh.coef_[0,0]:.8g} intercept={cal_inh.intercept_[0]:.8g}\\n")

    if not args.skip_plots:
        plot_metrics(metrics, panel, OUT / f"fig_effect_direction_v2_external{suffix}.png")

    print("\n=== External direction validation metrics ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved outputs to: {OUT}")


if __name__ == "__main__":
    main()
