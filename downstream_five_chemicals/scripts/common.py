from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


def parse_args(description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["config_path"] = str(Path(path).resolve())
    return cfg


def ensure_dirs(work: Path) -> None:
    for name in ["configs", "scripts", "tests", "manifests", "scores", "target_sets", "ora", "ctd", "metrics", "figures", "logs"]:
        (work / name).mkdir(parents=True, exist_ok=True)


def atomic_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    compression = "gzip" if path.suffix == ".gz" else None
    df.to_csv(tmp, index=False, compression=compression, **kwargs)
    os.replace(tmp, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_gmt(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                genes = {x.strip() for x in p[2:] if x.strip()}
                if genes:
                    out[p[0]] = {"name": p[1], "genes": genes}
    return out


def bh_fdr(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    if len(p) == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    q = np.empty(len(p), dtype=float)
    prev = 1.0
    for i in range(len(p) - 1, -1, -1):
        idx = order[i]
        prev = min(prev, p[idx] * len(p) / (i + 1), 1.0)
        q[idx] = prev
    return q


def ora(query: set[str], background: set[str], pathways: dict[str, dict]) -> pd.DataFrame:
    query = set(query) & background
    N, n = len(background), len(query)
    rows = []
    for pid, info in pathways.items():
        pg = set(info["genes"]) & background
        if not pg:
            continue
        ov = sorted(query & pg)
        K, k = len(pg), len(ov)
        p = float(hypergeom.sf(k - 1, N, K, n)) if k and n else 1.0
        fold = (k / n) / (K / N) if n and K else 0.0
        rows.append({"pathway_id": pid, "pathway_name": info["name"], "p_value": p,
                     "overlap_count": k, "pathway_size": K, "n_query": n,
                     "N_background": N, "fold_enrichment": fold,
                     "overlap_genes": ";".join(ov)})
    d = pd.DataFrame(rows)
    d["fdr"] = bh_fdr(d["p_value"])
    return d.sort_values(["fdr", "p_value", "overlap_count", "pathway_id"], ascending=[True, True, False, True]).reset_index(drop=True)


def overlap_metrics(pred: set[str], ref: set[str]) -> dict:
    pred, ref = set(pred), set(ref)
    ov = pred & ref
    union = pred | ref
    return {
        "n_predicted_pathways": len(pred), "n_ctd_comparable": len(ref), "n_overlap": len(ov),
        "n_model_only": len(pred - ref), "n_ctd_only": len(ref - pred),
        "CTD_support_fraction": len(ov) / len(pred) if pred else np.nan,
        "CTD_recovery_fraction": len(ov) / len(ref) if ref else np.nan,
        "Jaccard": len(ov) / len(union) if union else np.nan,
    }


def aggregate_gene_max(df: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["gene_symbol"] = d["gene_symbol"].fillna("").astype(str).str.strip()
    d = d[(d["gene_symbol"] != "") & (d["gene_symbol"].str.lower() != "nan")]
    agg = {c: "max" for c in score_cols}
    if "uniprot_id" in d:
        agg["uniprot_id"] = "nunique"
    return d.groupby("gene_symbol", as_index=False).agg(agg)


def strict_topk(df: pd.DataFrame, score: str, k: int) -> pd.DataFrame:
    return df.sort_values([score, "gene_symbol"], ascending=[False, True], kind="mergesort").head(k).copy()


def level_mask(df: pd.DataFrame, level: str) -> pd.Series:
    m = df["fdr"].lt(0.05)
    if level == "fdr_overlap2":
        m &= df["overlap_count"].ge(2)
    elif level == "fdr_overlap3":
        m &= df["overlap_count"].ge(3)
    elif level == "route1_top10":
        if "route1_top10_selected" not in df:
            raise ValueError("route1_top10_selected column missing")
        return df["route1_top10_selected"].astype(bool)
    elif level != "fdr":
        raise ValueError(level)
    return m
