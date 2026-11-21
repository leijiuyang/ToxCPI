#!/usr/bin/env python3
"""Run one controlled ToxCPI S3 ablation configuration on one GPU.

The script is intentionally separate from all production training/model files.  It
supports epoch-level resume and evidence-complete skipping.
"""

from __future__ import annotations

import os
import sys

# This must happen before importing torch.
for _i, _arg in enumerate(sys.argv):
    if _arg == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

import argparse
import fcntl
import hashlib
import json
import logging
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.amp import GradScaler
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
TEST1 = HERE.parents[1]
sys.path.insert(0, str(TEST1))
sys.path.insert(0, str(TEST1 / "experiments"))
sys.path.insert(0, str(HERE))

from s3_models import branch_flags, build_s3_model
from train_fusion import (
    EMAModel,
    FusionDataset,
    evaluate,
    make_index_collate_fn,
    train_epoch,
)


SEED = None
MAX_CHEM = 64
MAX_PROT = 768
FIXED_THRESHOLD = 0.5
PROTOCOL_VERSION = "effect-cold-five-seed-20261112-v1"
ROOT = Path('/home/data2/leijiuyang/Paper_revise/返修工作区/experiment/effect_cold_five_seed_ablation_20261112')
ORIGINAL = Path('/home/data2/leijiuyang/1_TOXCPI/leijiuyang/test1')
OUT = ROOT
FEATURE_DIR = ORIGINAL / "data" / "pretrained_features"

CONFIGS = {
    ("binding", "warm"): dict(batch_size=128, lr=1e-4, weight_decay=5e-5,
                                dropout=0.3, hidden_dim=128, epochs=80,
                                patience=15, label_smoothing=0.02, warmup_ratio=0.0),
    ("binding", "cold"): dict(batch_size=64, lr=1e-4, weight_decay=5e-5,
                                dropout=0.3, hidden_dim=128, epochs=80,
                                patience=10, label_smoothing=0.05, warmup_ratio=0.0),
    ("activation", "warm"): dict(batch_size=64, lr=1.5e-4, weight_decay=3e-5,
                                   dropout=0.25, hidden_dim=256, epochs=80,
                                   patience=20, label_smoothing=0.02, warmup_ratio=0.05),
    ("activation", "cold"): dict(batch_size=64, lr=1.5e-4, weight_decay=3e-5,
                                   dropout=0.25, hidden_dim=256, epochs=80,
                                   patience=20, label_smoothing=0.02, warmup_ratio=0.05),
    ("inhibition", "warm"): dict(batch_size=64, lr=1e-4, weight_decay=5e-5,
                                   dropout=0.3, hidden_dim=256, epochs=80,
                                   patience=20, label_smoothing=0.05, warmup_ratio=0.05),
    ("inhibition", "cold"): dict(batch_size=64, lr=1e-4, weight_decay=5e-5,
                                   dropout=0.3, hidden_dim=256, epochs=80,
                                   patience=20, label_smoothing=0.05, warmup_ratio=0.05),
}

FORMAL_CHECKPOINTS = {
    ("binding", "warm"): (TEST1 / "checkpoints" / "fusion_warm_best.pth", 62),
    ("binding", "cold"): (TEST1 / "checkpoints" / "fusion_cold_best.pth", 14),
    ("activation", "warm"): (TEST1 / "checkpoints" / "effect_activation_warm_rawglobal_v3_rawglobal_local_control_1to1_targetaware_20260507_seed42_best.pth", 22),
    ("activation", "cold"): (TEST1 / "checkpoints" / "effect_activation_cold_rawglobal_v3_rawglobal_local_control_1to1_targetaware_20260507_seed42_best.pth", 22),
    ("inhibition", "warm"): (TEST1 / "checkpoints" / "effect_inhibition_warm_rawglobal_v3_rawglobal_local_control_1to1_targetaware_20260507_rerun_gpu1_tmux_seed42_best.pth", 16),
    ("inhibition", "cold"): (TEST1 / "checkpoints" / "effect_inhibition_cold_rawglobal_v3_rawglobal_local_control_1to1_targetaware_20260507_rerun_gpu0_seed42_best.pth", 12),
}

MODEL_NAMES = {
    ("binding", "full"): "Full ToxCPI (4-head bilinear attention)",
    ("binding", "mean_pooling"): "Mean pooling + MLP",
    ("binding", "single_head"): "Single-head bilinear",
    ("activation", "full"): "Full ToxCPI",
    ("activation", "local_only"): "Local only",
    ("activation", "global_only"): "Global only",
    ("inhibition", "full"): "Full ToxCPI",
    ("inhibition", "local_only"): "Local only",
    ("inhibition", "global_only"): "Global only",
}


def experiment_id(task, split, variant):
    return f"{task}_{split}_{variant}_seed{SEED}"


def setup_logging(path):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="a")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return logging.getLogger("s3_ablation")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paths_for(task, split, variant):
    eid = experiment_id(task, split, variant)
    return {
        "id": eid,
        "log": OUT / "logs" / f"{eid}.log",
        "checkpoint": OUT / "checkpoints" / f"{eid}_best.pth",
        "resume": OUT / "checkpoints" / f"{eid}_resume.pth",
        "test_prediction": OUT / "predictions" / f"{eid}_test_predictions.csv",
        "val_prediction": OUT / "predictions" / f"{eid}_validation_predictions.csv",
        "config": OUT / "configs" / f"{eid}.json",
        "result": OUT / "results" / f"{eid}.json",
    }


def split_paths(task, split):
    if task == "binding":
        base = ORIGINAL / "data" / "interaction_binding" / "splits"
        return {part: base / f"{part}_{split}.csv" for part in ("train", "val", "test")}
    base = ORIGINAL / "data" / "interaction_type_1to1_targetaware"
    return {part: base / f"effect_{task}_{split}_{part}.csv" for part in ("train", "val", "test")}


def cache_path(task, split):
    # The production entity builder materializes the complete pretrained feature
    # dictionaries, independent of the downstream split.  Several old Effect cache
    # archives are truncated; this verified unified cache contains the same fixed
    # BerMol/ESM2 entities and is read-only for every S3 row.
    return FEATURE_DIR / "fusion_warm_chem64_prot768_nogcn_5da6d84b77_entity_tensors.pt"


def protocol_payload(task, split, variant):
    files = split_paths(task, split)
    code_files = {
        "runner": Path(__file__).resolve(),
        "ablation_models": HERE / "s3_models.py",
        "production_model": TEST1 / "model_fusion.py",
        "production_attention": TEST1 / "attention.py",
        "production_training_helpers": TEST1 / "experiments" / "train_fusion.py",
    }
    cache = cache_path(task, split)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": task,
        "split": split,
        "variant": variant,
        "seed": SEED,
        "threshold_rule": "fixed probability threshold 0.5 (formal result convention)",
        "hyperparameters": CONFIGS[(task, split)],
        "split_sha256": {part: sha256_file(path) for part, path in files.items()},
        "code_sha256": {name: sha256_file(path) for name, path in code_files.items()},
        "entity_cache": relative(cache),
        "entity_cache_size": cache.stat().st_size,
        "frozen_protocol_sha256": sha256_file(ROOT / "实验协议冻结.json"),
        "source_manifest_sha256": sha256_file(ROOT / "audit/source_before.json"),
    }


def protocol_fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def output_branch_labels(task, variant, local_present, raw_present):
    if task == "binding":
        local = {
            "full": "4-head bilinear",
            "mean_pooling": "Mean pooling + MLP",
            "single_head": "1-head bilinear",
        }[variant]
        return local, "N/A"
    return ("Present" if local_present else "Absent",
            "Present" if raw_present else "Absent")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_state(path, device):
    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location=device)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def load_data(task, split, logger):
    files = split_paths(task, split)
    frames = {}
    for part, path in files.items():
        frame = pd.read_csv(path, low_memory=False).reset_index(names="source_row_index")
        frames[part] = frame
        logger.info("%s split: %s rows | %s", part, len(frame), path)

    logger.info("Loading immutable pretrained feature dictionaries")
    with open(FEATURE_DIR / "compound_features_bermol_tokens.pkl", "rb") as handle:
        comp_feats = pickle.load(handle)
    with open(FEATURE_DIR / "protein_features_esm2_residues.pkl", "rb") as handle:
        prot_feats = pickle.load(handle)
    try:
        shared = torch.load(cache_path(task, split), map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        shared = torch.load(cache_path(task, split), map_location="cpu")

    datasets = {}
    filtered_frames = {}
    for part, frame in frames.items():
        valid = frame.apply(
            lambda row: str(row["chemical_name"]) in comp_feats and str(row["uniprot_id"]) in prot_feats,
            axis=1,
        )
        filtered = frame.loc[valid].reset_index(drop=True)
        if len(filtered) != len(frame):
            logger.warning("%s: production feature-availability filter excluded %d/%d source rows",
                           part, len(frame) - len(filtered), len(frame))
        datasets[part] = FusionDataset(
            filtered,
            comp_feats,
            prot_feats,
            "chemical_name",
            "uniprot_id",
            "label",
            use_residue_attention=True,
            shared_tensors=shared,
            return_indices=True,
        )
        filtered_frames[part] = filtered
    return files, filtered_frames, datasets, shared


def make_loaders(datasets, shared, batch_size):
    collate = make_index_collate_fn(shared)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            generator=generator, num_workers=0, pin_memory=True, collate_fn=collate),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True, collate_fn=collate),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True, collate_fn=collate),
    }
    return loaders, generator


def scheduler_for(optimizer, cfg):
    warmup = max(1, int(cfg["epochs"] * cfg["warmup_ratio"])) if cfg["warmup_ratio"] else 0
    if not warmup:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] / 10
        )
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    return SequentialLR(
        optimizer,
        [LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
         CosineAnnealingLR(optimizer, T_max=cfg["epochs"] - warmup, eta_min=cfg["lr"] / 10)],
        milestones=[warmup],
    )


def prediction_frame(frame, labels, probs, threshold, logits, task, variant):
    if len(frame) != len(labels):
        raise RuntimeError(f"Prediction count {len(labels)} != split rows {len(frame)}")
    out = pd.DataFrame({
        "record_id": [f"{task}:cold:{int(i)}" for i in frame["source_row_index"]],
        "task": task, "variant": variant, "seed": SEED,
        "source_row_index": frame["source_row_index"].astype(int),
        "smiles": frame["smiles"].astype(str),
        "actual_chemical_input_id": frame["chemical_name"].astype(str),
        "raw_logit": np.asarray(logits).astype(float),
        "chemical_name": frame["chemical_name"].astype(str),
        "uniprot_id": frame["uniprot_id"].astype(str),
        "label": np.asarray(labels).astype(int),
        "probability": np.asarray(probs).astype(float),
        "validation_threshold": threshold,
        "prediction": (np.asarray(probs) >= threshold).astype(int),
    })
    if not np.array_equal(out["label"].to_numpy(), frame["label"].astype(int).to_numpy()):
        raise RuntimeError("Prediction labels do not preserve source split order")
    return out


def relative(path):
    return str(Path(path).resolve())


def aggregate_results():
    lock_path = OUT / ".aggregate.lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = []
        for result_path in sorted((OUT / "results").glob("*.json")):
            with open(result_path, encoding="utf-8") as handle:
                records.append(json.load(handle))
        if records:
            order = {"binding": 0, "activation": 1, "inhibition": 2}
            split_order = {"warm": 0, "cold": 1}
            variant_order = {"full": 0, "mean_pooling": 1, "single_head": 2,
                             "local_only": 1, "global_only": 2}
            records.sort(key=lambda r: (order[r["task_key"]], split_order[r["split_key"]],
                                        variant_order[r["variant_key"]]))
            pd.DataFrame(records).to_csv(OUT / "ablation_results.csv", index=False)
        fcntl.flock(lock, fcntl.LOCK_UN)


def evidence_complete(paths, task, split, variant):
    needed = [paths["result"], paths["log"], paths["checkpoint"], paths["test_prediction"],
              paths["val_prediction"], paths["config"]]
    if not all(path.exists() and path.stat().st_size > 0 for path in needed):
        return False
    try:
        with open(paths["config"], encoding="utf-8") as handle:
            config = json.load(handle)
        with open(paths["result"], encoding="utf-8") as handle:
            result = json.load(handle)
        expected_payload = protocol_payload(task, split, variant)
        expected_fingerprint = protocol_fingerprint(expected_payload)
        if config.get("protocol_fingerprint") != expected_fingerprint:
            return False
        if result.get("protocol_fingerprint") != expected_fingerprint:
            return False
        if float(result.get("validation_threshold", -1)) != FIXED_THRESHOLD:
            return False
        test_predictions = pd.read_csv(paths["test_prediction"])
        val_predictions = pd.read_csv(paths["val_prediction"])
        if len(test_predictions) != int(result["test_rows"]):
            return False
        if not np.allclose(test_predictions["validation_threshold"], FIXED_THRESHOLD):
            return False
        if not np.allclose(val_predictions["validation_threshold"], FIXED_THRESHOLD):
            return False
        return result.get("split_test_sha256") == expected_payload["split_sha256"]["test"]
    except Exception:
        return False



def atomic_torch_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def state_digest(model):
    h = hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def main():
    global SEED, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=[42,1009,2027,3407,7919])
    parser.add_argument("--task", required=True, choices=["binding", "activation", "inhibition"])
    parser.add_argument("--split", required=True, choices=["warm", "cold"])
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restart-training", action="store_true",
                        help="Discard resume state; --force alone only refreshes stale evidence")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    SEED = args.seed
    if args.task == "binding" or args.split != "cold":
        parser.error("This extension is restricted to cold-chemical effect tasks")
    if args.smoke_only:
        OUT = ROOT / "smoke"

    allowed = (["full", "mean_pooling", "single_head"] if args.task == "binding"
               else ["full", "local_only", "global_only"])
    if args.variant not in allowed:
        parser.error(f"variant must be one of {allowed}")
    for subdir in ("logs", "checkpoints", "predictions", "configs", "results"):
        (OUT / subdir).mkdir(parents=True, exist_ok=True)

    paths = paths_for(args.task, args.split, args.variant)
    logger = setup_logging(paths["log"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; this experiment is GPU-only")
    device = torch.device("cuda:0")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(min(2, args.torch_threads))
    set_seed(SEED)
    cfg = dict(CONFIGS[(args.task, args.split)])
    logger.info("START %s on physical GPU %s (%s)", paths["id"], args.gpu, torch.cuda.get_device_name(0))

    if not args.force and evidence_complete(paths, args.task, args.split, args.variant):
        logger.info("SKIP protocol-matched evidence-complete configuration before feature-cache loading")
        aggregate_results()
        return

    files, frames, datasets, shared = load_data(args.task, args.split, logger)
    loaders, loader_generator = make_loaders(datasets, shared, cfg["batch_size"])
    model = build_s3_model(args.task, args.variant, cfg["hidden_dim"], cfg["dropout"]).to(device)
    initial_state_sha256 = state_digest(model)
    logger.info("INDEPENDENT INITIALIZATION seed=%d state_sha256=%s", SEED, initial_state_sha256)
    local_present, raw_present = branch_flags(args.task, args.variant)
    local_label, raw_label = output_branch_labels(
        args.task, args.variant, local_present, raw_present
    )
    protocol = protocol_payload(args.task, args.split, args.variant)
    fingerprint = protocol_fingerprint(protocol)

    config_record = {
        "independent_training": True, "initial_state_sha256": initial_state_sha256,
        "experiment_id": paths["id"], "task": args.task, "split": args.split,
        "variant": args.variant, "seed": SEED, "physical_gpu": args.gpu,
        "model_source": "experiments/ablation/s3_models.py",
        "production_model_source": "model_fusion.py", "production_attention_source": "attention.py",
        "local_interaction_branch": local_present, "raw_global_branch": raw_present,
        "local_interaction_description": local_label,
        "raw_global_description": raw_label,
        "projected_global_branch": False, "dual_control_used": False,
        "max_chem_tokens": MAX_CHEM, "max_protein_length": MAX_PROT,
        "monitor": "validation AUROC",
        "threshold_rule": "fixed probability threshold 0.5 (formal result convention)",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_fingerprint": fingerprint,
        "protocol_payload": protocol,
        "criterion": "BCEWithLogitsLoss without pos_weight", "ema_decay": 0.999,
        "amp": False, "enhancements": {"gcn": False, "interaction_cnn": False,
                                         "site_weak_supervision": False, "multi_granularity": False},
        "hyperparameters": cfg, "train_rows": len(frames["train"]),
        "validation_rows": len(frames["val"]), "test_rows": len(frames["test"]),
        "source_split_rows": {part: int(len(pd.read_csv(path, usecols=["label"])))
                              for part, path in files.items()},
        "feature_eligible_rows": {part: len(frame) for part, frame in frames.items()},
        "split_files": {part: relative(path) for part, path in files.items()},
        "split_sha256": {part: sha256_file(path) for part, path in files.items()},
        "entity_cache": relative(cache_path(args.task, args.split)),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with open(paths["config"], "w", encoding="utf-8") as handle:
        json.dump(config_record, handle, ensure_ascii=False, indent=2)

    if args.smoke_only:
        batch = next(iter(loaders["train"]))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        from train_fusion import _forward_batch
        for _ in range(3):
            model.train(); opt.zero_grad()
            z, *_ = _forward_batch(model, batch, device)
            y = batch["label"].to(device)*(1-cfg["label_smoothing"])+0.5*cfg["label_smoothing"]
            loss = nn.functional.binary_cross_entropy_with_logits(z,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        torch.cuda.synchronize()
        report = dict(task=args.task,variant=args.variant,batch_size=len(z),loss=float(loss),
                      peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                      three_steps_seconds=time.time()-t,initial_state_sha256=initial_state_sha256)
        with open(OUT/(paths["id"]+"_benchmark.json"),'w') as f: json.dump(report,f,indent=2)
        logger.info("SMOKE OK %s",report)
        return

    best_epoch = None
    best_score = float("-inf")
    no_improve = 0
    criterion = nn.BCEWithLogitsLoss()
    start_epoch = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = scheduler_for(optimizer, cfg)
    scaler = GradScaler("cuda", enabled=False)
    ema = EMAModel(model, decay=0.999)
    if paths["resume"].exists() and not args.restart_training:
        resume = torch.load(paths["resume"], map_location=device, weights_only=False)
        if resume.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch: task/variant/seed/data/features/config/code")
        model.load_state_dict(resume["model_state"])
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        ema.shadow = {key: value.to(device) for key, value in resume["ema_shadow"].items()}
        start_epoch = resume["epoch"] + 1
        best_score = resume["best_score"]
        best_epoch = resume["best_epoch"]
        no_improve = resume["no_improve"]
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in resume["cuda_rng"]])
        if "loader_rng" in resume:
            loader_generator.set_state(resume["loader_rng"].cpu())
        else:
            logger.warning("Legacy resume has no DataLoader RNG state; weights remain valid, "
                           "but the next epoch shuffle cannot exactly reproduce an uninterrupted run")
        logger.info("Resuming from epoch %d", start_epoch + 1)

    training_range = range(start_epoch, cfg["epochs"])
    if no_improve >= cfg["patience"] or start_epoch >= cfg["epochs"]:
        training_range = []
        logger.info("Resume already satisfies a terminal training condition; re-extracting metrics only")
    for epoch in training_range:
        started = time.time()
        train_loss = train_epoch(
            model, loaders["train"], optimizer, criterion, device, scaler,
            use_amp=False, label_smoothing=cfg["label_smoothing"], ema=ema,
        )
        ema.apply_shadow(model)
        val_metrics = evaluate(model, loaders["val"], criterion, device)
        ema.restore(model)
        scheduler.step()
        logger.info(
            "Epoch %d/%d train_loss=%.6f val_loss=%.6f val_AUROC=%.6f val_AUPRC=%.6f elapsed=%.1fs",
            epoch + 1, cfg["epochs"], train_loss, val_metrics["loss"],
            val_metrics["auroc"], val_metrics["auprc"], time.time() - started,
        )
        if val_metrics["auroc"] > best_score:
            best_score = float(val_metrics["auroc"])
            best_epoch = epoch + 1
            no_improve = 0
            ema.apply_shadow(model)
            atomic_torch_save({"state_dict": model.state_dict(), "best_epoch": best_epoch,
                        "best_validation_auroc": best_score, "config": config_record}, paths["checkpoint"])
            ema.restore(model)
            logger.info("NEW BEST validation AUROC %.6f", best_score)
        else:
            no_improve += 1

        atomic_torch_save({
            "protocol_fingerprint": fingerprint,
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "ema_shadow": {key: value.cpu() for key, value in ema.shadow.items()},
            "best_score": best_score, "best_epoch": best_epoch, "no_improve": no_improve,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "loader_rng": loader_generator.get_state(),
        }, paths["resume"])
        if no_improve >= cfg["patience"]:
            logger.info("EARLY STOP at epoch %d", epoch + 1)
            break
    if not paths["checkpoint"].exists():
        raise RuntimeError("No validation-best checkpoint was saved")
    model.load_state_dict(load_state(paths["checkpoint"], device), strict=True)

    # Persist any provenance added while resolving/reusing the checkpoint.
    with open(paths["config"], "w", encoding="utf-8") as handle:
        json.dump(config_record, handle, ensure_ascii=False, indent=2)

    val_metrics = evaluate(model, loaders["val"], criterion, device)
    test_metrics = evaluate(model, loaders["test"], criterion, device)
    threshold = FIXED_THRESHOLD
    val_threshold_mcc = float(matthews_corrcoef(
        val_metrics["labels"], (val_metrics["probs"] >= threshold).astype(int)
    ))
    test_preds = (test_metrics["probs"] >= threshold).astype(int)
    val_pred = prediction_frame(frames["val"], val_metrics["labels"], val_metrics["probs"], threshold, val_metrics["logits"], args.task, args.variant)
    test_pred = prediction_frame(frames["test"], test_metrics["labels"], test_metrics["probs"], threshold, test_metrics["logits"], args.task, args.variant)
    val_pred.to_csv(paths["val_prediction"], index=False)
    test_pred.to_csv(paths["test_prediction"], index=False)

    task_label = args.task.capitalize()
    split_label = "Warm-start" if args.split == "warm" else "Cold-chemical"
    config_summary = (
        f"seed={SEED}; hidden={cfg['hidden_dim']}; interaction={cfg['hidden_dim']*2}; "
        f"batch={cfg['batch_size']}; lr={cfg['lr']}; wd={cfg['weight_decay']}; "
        f"dropout={cfg['dropout']}; epochs={cfg['epochs']}; patience={cfg['patience']}; "
        f"EMA=0.999; monitor=val_AUROC; threshold=fixed_0.5"
    )
    result = {
        "Task": task_label, "Split setting": split_label,
        "Model variant": MODEL_NAMES[(args.task, args.variant)],
        "Local interaction branch": local_label,
        "Raw global branch": raw_label,
        "AUROC": float(roc_auc_score(test_metrics["labels"], test_metrics["probs"])),
        "AUPRC": float(average_precision_score(test_metrics["labels"], test_metrics["probs"])),
        "Accuracy": float(accuracy_score(test_metrics["labels"], test_preds)),
        "F1": float(f1_score(test_metrics["labels"], test_preds)),
        "MCC": float(matthews_corrcoef(test_metrics["labels"], test_preds)),
        "independent_training": True, "initial_state_sha256": initial_state_sha256,
        "seed": SEED, "validation_threshold": threshold, "best_epoch": int(best_epoch),
        "log_file": relative(paths["log"]), "checkpoint_file": relative(paths["checkpoint"]),
        "prediction_file": relative(paths["test_prediction"]),
        "validation_prediction_file": relative(paths["val_prediction"]),
        "config_file": relative(paths["config"]), "config_summary": config_summary,
        "task_key": args.task, "split_key": args.split, "variant_key": args.variant,
        "validation_AUROC": float(val_metrics["auroc"]),
        "validation_MCC_at_threshold": val_threshold_mcc,
        "test_rows": len(frames["test"]), "split_test_sha256": sha256_file(files["test"]),
        "protocol_fingerprint": fingerprint,
    }
    with open(paths["result"], "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    aggregate_results()
    logger.info("FINAL threshold=%.2f AUROC=%.6f AUPRC=%.6f ACC=%.6f F1=%.6f MCC=%.6f",
                threshold, result["AUROC"], result["AUPRC"], result["Accuracy"], result["F1"], result["MCC"])
    logger.info("COMPLETE %s", paths["id"])


if __name__ == "__main__":
    main()
