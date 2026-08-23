import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import h5py
import numpy as np
import pandas as pd
import torch
from esm.pretrained import load_model_and_alphabet_core


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = REPOSITORY_ROOT / "runtime" / "human_proteome_features"
ESM2_LOCAL = Path(os.environ.get("TOXCPI_ESM2_MODEL", "")).expanduser()
SEQ_TSV = Path(os.environ.get(
    "TOXCPI_HUMAN_PROTEOME_SEQUENCES_TSV",
    RUNTIME_DIR / "inputs" / "uniprot_human_genes_with_seq.tsv",
)).expanduser()
GENE_TSV = Path(os.environ.get(
    "TOXCPI_HUMAN_PROTEOME_GENES_TSV",
    RUNTIME_DIR / "inputs" / "uniprot_human_genes.tsv",
)).expanduser()
OUT_DIR = RUNTIME_DIR
UNIVERSE = OUT_DIR / "inputs" / "human_proteome_universe.csv"
LOG = OUT_DIR / "logs" / "00_extract_esm2.log"
PKL = Path(os.environ.get(
    "TOXCPI_ESM2_RESIDUE_PKL",
    REPOSITORY_ROOT / "runtime" / "pretrained_features" / "protein_features_esm2_residues.pkl",
)).expanduser()
H5 = Path(os.environ.get(
    "TOXCPI_PROTEOME_ESM2_H5",
    REPOSITORY_ROOT / "runtime" / "pretrained_features" / "human_proteome_esm2_residues.h5",
)).expanduser()

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def valid_sequence(seq: str) -> bool:
    return bool(seq) and seq != "NAN" and len(seq) >= 30 and set(seq) <= STD_AA


def build_universe() -> pd.DataFrame:
    for source, variable in (
        (SEQ_TSV, "TOXCPI_HUMAN_PROTEOME_SEQUENCES_TSV"),
        (GENE_TSV, "TOXCPI_HUMAN_PROTEOME_GENES_TSV"),
    ):
        if not source.exists():
            raise FileNotFoundError(
                f"Required UniProt input is missing: {source}. Set {variable}."
            )
    (OUT_DIR / "inputs").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

    seq = pd.read_csv(SEQ_TSV, sep="\t")
    gene = pd.read_csv(GENE_TSV, sep="\t")
    df = seq.merge(gene, on="Entry", how="inner").rename(
        columns={
            "Entry": "uniprot_id",
            "Sequence": "sequence",
            "Gene Names (primary)": "gene_symbol",
            "Protein names": "protein_name",
        }
    )
    df = df[["uniprot_id", "gene_symbol", "protein_name", "sequence"]].copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["sequence_length"] = df["sequence"].str.len()
    df["valid_sequence"] = df["sequence"].map(valid_sequence)
    df.to_csv(UNIVERSE, index=False)

    gene_text = df["gene_symbol"].astype(str).str.strip()
    sub = df[df["valid_sequence"] & gene_text.ne("") & gene_text.ne("nan")]
    log(
        "universe "
        f"total={len(df)} valid={int(df['valid_sequence'].sum())} "
        f"to_extract(uniprot)={len(sub)} unique_gene={sub['gene_symbol'].nunique()}"
    )
    return df


def resolve_device() -> str:
    requested = os.environ.get("DEVICE", "cuda:3")
    allow_cpu = os.environ.get("ALLOW_CPU", "0") == "1"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            if allow_cpu:
                log("CUDA unavailable; ALLOW_CPU=1 so using cpu")
                return "cpu"
            raise RuntimeError(
                "CUDA is not available. Set ALLOW_CPU=1 to run on CPU, or run "
                "on a machine with a visible CUDA device."
            )
        idx = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {requested}, but only {torch.cuda.device_count()} CUDA devices are visible."
            )
    return requested


def load_esm2(device: str):
    if not ESM2_LOCAL.is_file():
        raise FileNotFoundError(
            "ESM-2 weight is missing. Set TOXCPI_ESM2_MODEL to "
            "esm2_t33_650M_UR50D.pt."
        )
    model_data = torch.load(ESM2_LOCAL, map_location="cpu", weights_only=False)
    model, alphabet = load_model_and_alphabet_core("esm2_t33_650M_UR50D", model_data, None)
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    log(f"ESM2 loaded device={device} num_layers={model.num_layers}")
    return model, batch_converter


def extract_residue_feature(seq: str, model, batch_converter, device: str) -> np.ndarray:
    clean = "".join(str(seq).split()).upper()[:1022]
    _, _, batch_tokens = batch_converter([("q", clean)])
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        result = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
    rep = result["representations"][model.num_layers]
    return rep[0, 1 : len(clean) + 1].cpu().numpy().astype(np.float16)


def h5_key_set() -> set:
    if not os.path.exists(H5):
        return set()
    with h5py.File(H5, "r") as h5:
        return set(h5.keys())


def main() -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    H5.parent.mkdir(parents=True, exist_ok=True)
    log("START Script 00 human proteome ESM2 residue cache")
    df = build_universe()

    gene_text = df["gene_symbol"].astype(str).str.strip()
    sub = df[df["valid_sequence"] & gene_text.ne("") & gene_text.ne("nan")].copy()
    sub["uniprot_id"] = sub["uniprot_id"].astype(str)
    sub["sequence"] = sub["sequence"].astype(str)
    targets = list(sub["uniprot_id"])
    seqmap = dict(zip(sub["uniprot_id"], sub["sequence"]))
    log(f"targets(uniprot)={len(targets)}")

    dry_run_limit = int(os.environ.get("DRY_RUN_LIMIT", "0") or "0")
    device = resolve_device()

    reused = 0
    new = 0
    fail = 0
    truncated_reused = 0
    truncated_new = 0

    with h5py.File(H5, "a") as h5:
        done = set(h5.keys())
        log(f"existing_h5_keys={len(done)}")

        if PKL.exists():
            with open(PKL, "rb") as f:
                pkl = pickle.load(f)
        else:
            pkl = {}
        log(f"loaded existing pkl keys={len(pkl)}")

        for uid in targets:
            if uid in done or uid not in pkl:
                continue
            arr = pkl[uid]
            if getattr(arr, "ndim", 0) != 2 or arr.shape[1] != 1280:
                log(f"SKIP_PKL_BAD_SHAPE {uid} shape={getattr(arr, 'shape', None)}")
                continue
            if arr.shape[0] > 1022:
                truncated_reused += 1
            arr = arr[:1022].astype(np.float16, copy=False)
            h5.create_dataset(uid, data=arr)
            done.add(uid)
            reused += 1
        h5.flush()
        log(f"reused from pkl={reused} truncated_reused={truncated_reused}")
        del pkl

        missing = [uid for uid in targets if uid not in done]
        missing.sort(key=lambda uid: len(seqmap[uid]))
        if dry_run_limit > 0:
            log(f"DRY_RUN_LIMIT active: extracting first {dry_run_limit} of {len(missing)} missing proteins")
            missing = missing[:dry_run_limit]
        log(f"missing to extract={len(missing)}")

        if missing:
            model, batch_converter = load_esm2(device)
            start = time.time()
            for i, uid in enumerate(missing, 1):
                try:
                    if len(seqmap[uid]) > 1022:
                        truncated_new += 1
                    arr = extract_residue_feature(seqmap[uid], model, batch_converter, device)
                    h5.create_dataset(uid, data=arr)
                    new += 1
                except Exception as exc:
                    fail += 1
                    log(f"FAIL {uid} L={len(seqmap[uid])}: {exc}")
                if i % 200 == 0:
                    h5.flush()
                    seconds_per_protein = (time.time() - start) / i
                    log(
                        f"extracted {i}/{len(missing)} "
                        f"last_L={len(seqmap[uid])} {seconds_per_protein:.3f}s/prot"
                    )
            h5.flush()
        final_keys = len(h5.keys())

    covered = h5_key_set()
    uncovered = [uid for uid in targets if uid not in covered]
    size_gb = os.path.getsize(H5) / 1e9 if os.path.exists(H5) else 0.0
    log(
        "DONE "
        f"targets={len(targets)} reused={reused} new={new} fail={fail} "
        f"truncated_new={truncated_new} final_keys={final_keys} "
        f"size={size_gb:.1f}GB uncovered={len(uncovered)}"
    )
    if uncovered:
        log("UNCOVERED(<=50): " + ", ".join(uncovered[:50]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
