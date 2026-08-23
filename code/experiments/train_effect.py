



















import glob
import os, sys
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
        break

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from torch.amp import GradScaler


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_fusion import BindingModelFusion, EffectModelFusion
from train_fusion import (
    FusionDataset, seq_collate_fn, EMAModel,
    train_epoch, evaluate,
    build_or_load_entity_tensors, default_entity_cache_path,
    make_index_collate_fn
)


if __name__ == "__main__":
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = REPOSITORY_ROOT / "runtime"
FEATURE_DIR = Path(os.environ.get(
    "TOXCPI_FEATURE_DIR", RUNTIME_ROOT / "pretrained_features"
)).expanduser().resolve()


def load_partial_checkpoint(model, ckpt_path, device):






    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Init checkpoint not found: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    checkpoint = {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in checkpoint.items()
    }
    model_state = model.state_dict()
    load_state = {}
    loaded_keys = []
    skipped = []

    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            load_state[key] = value
            loaded_keys.append(key)
        else:
            reason = 'missing' if key not in model_state else f"shape {tuple(value.shape)} -> {tuple(model_state[key].shape)}"
            skipped.append((key, reason))

    model_state.update(load_state)
    model.load_state_dict(model_state, strict=False)

    param_names = {name for name, _ in model.named_parameters()}
    loaded_params = sorted(k for k in loaded_keys if k in param_names)
    logger.info(f"Warm-start loaded {len(loaded_keys)} tensors from {ckpt_path}")
    logger.info(f"Warm-start loaded trainable params: {len(loaded_params)}")
    if skipped:
        logger.info(f"Warm-start skipped {len(skipped)} tensors")
        for key, reason in skipped[:12]:
            logger.info(f"  skip {key}: {reason}")
        if len(skipped) > 12:
            logger.info(f"  ... {len(skipped) - 12} more skipped")
    return set(loaded_params)


def set_params_trainable(model, param_names, trainable):

    affected = 0
    for name, param in model.named_parameters():
        if name in param_names:
            param.requires_grad = trainable
            affected += param.numel()
    state = "unfrozen" if trainable else "frozen"
    logger.info(f"Warm-start backbone {state}: {affected:,} parameters")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def monitor_score(metrics, metric):
    if metric == 'loss':
        return -metrics['loss']
    return metrics[metric]


def main():
    parser = argparse.ArgumentParser(description='Effect 二分类任务训练')
    parser.add_argument('--task', type=str, required=True,
                        choices=['activation', 'inhibition'])
    parser.add_argument('--split', type=str, required=True,
                        choices=['cold', 'warm'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_protein_length', type=int, default=768)
    parser.add_argument('--max_chem_tokens', type=int, default=64)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--comp_noise', type=float, default=0.0)
    parser.add_argument('--prot_noise', type=float, default=0.0)
    parser.add_argument('--downsample', type=str, default=None,
                        help='Downsample train negatives, e.g. "1:1" or "1:3"')
    parser.add_argument('--pos_weight', type=float, default=0,
                        help='pos_weight for BCEWithLogitsLoss. 0=auto from data, -1=disabled')
    parser.add_argument('--multi_granularity', action='store_true',
                        help='Enable multi-granularity interaction (fine + coarse)')
    parser.add_argument('--interaction_cnn', action='store_true',
                        help='在多头 token-residue 注意力矩阵上启用 2D CNN 交互增强')
    parser.add_argument('--interaction_cnn_channels', type=int, default=16,
                        help='InteractionCNN 第一层通道数，仅 --interaction_cnn 启用时生效')
    parser.add_argument('--molecular_gcn', action='store_true',
                        help='在 BerMol token 特征上启用 RDKit 分子图 GCN 拓扑增强')
    parser.add_argument('--gcn_layers', type=int, default=2,
                        help='Molecular GCN 层数，仅 --molecular_gcn 启用时生效')
    parser.add_argument('--effect_dual_branch', action='store_true',
                        help='启用 Effect 专用局部交互 + 全局功能双分支模型')
    parser.add_argument('--effect_raw_global_branch', action='store_true',
                        help='在 Effect 双分支模型中启用原始 BerMol/ESM2 全局特征分支')
    parser.add_argument('--init_from', type=str, default=None,
                        help='从指定 v3/E6 checkpoint 部分加载 shape 匹配权重')
    parser.add_argument('--init_from_v3', action='store_true',
                        help='Deprecated alias; requires an explicit --init_from path')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                        help='warm-start 后先冻结加载的旧主干多少个 epoch，只训练新增模块/分类头')
    parser.add_argument('--monitor_metric', type=str, default='auroc',
                        choices=['auroc', 'auprc', 'f1', 'mcc', 'loss'],
                        help='保存 best checkpoint 和 early stopping 使用的验证集指标')
    parser.add_argument('--exp_suffix', type=str, default='',
                        help='追加到实验名和checkpoint名的后缀，用于避免覆盖旧实验')
    parser.add_argument('--entity_cache', type=str, default=None,
                        help='Step 2 entity tensor cache path. Default: data/pretrained_features/*.pt')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing effect_{task}_{split}_{train,val,test}.csv')
    parser.add_argument('--rebuild_entity_cache', action='store_true',
                        help='Force rebuild Step 2 entity tensor cache')
    parser.add_argument('--prepare_data_only', action='store_true',
                        help='Only build/load cached entity tensors and exit before training')
    parser.add_argument('--fast_index_collate', action='store_true',
                        help='Use index-based fixed-length collate to reduce CPU batch preparation')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers. With large shared tensors, 0 avoids worker memory duplication')
    parser.add_argument('--torch_threads', type=int, default=4,
                        help='Limit PyTorch intra-op CPU threads per training process')
    args = parser.parse_args()

    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(max(1, min(2, args.torch_threads)))
        logger.info(f"PyTorch CPU threads: intra={torch.get_num_threads()}, "
                    f"interop={torch.get_num_interop_threads()}")




    exp_name = f"effect_{args.task}_{args.split}"
    if args.interaction_cnn:
        exp_name = f"{exp_name}_icnn"
    if args.molecular_gcn:
        exp_name = f"{exp_name}_gcn"
    if args.effect_dual_branch:
        exp_name = f"{exp_name}_dual"
    if args.effect_raw_global_branch:
        exp_name = f"{exp_name}_rawglobal"
    if args.init_from or args.init_from_v3:
        exp_name = f"{exp_name}_v3warm"
    if args.exp_suffix:
        exp_name = f"{exp_name}_{args.exp_suffix.strip('_')}"
    logger.info("=" * 60)
    logger.info(f"Effect Binary Classification: {args.task.upper()} / {args.split.upper()}")
    logger.info("=" * 60)
    logger.info(f"GPU: {args.gpu} | Batch: {args.batch_size} | LR: {args.lr}")
    logger.info(f"Hidden: {args.hidden_dim} | Dropout: {args.dropout} | WD: {args.weight_decay}")
    logger.info(f"Label Smoothing: {args.label_smoothing} | EMA: {args.ema_decay}")
    logger.info(f"Warmup: {args.warmup_ratio} | Patience: {args.patience}")
    logger.info(f"Max Prot: {args.max_protein_length} | Max Chem: {args.max_chem_tokens}")
    logger.info(f"Multi-Granularity: {args.multi_granularity}")
    logger.info(f"Interaction CNN: {args.interaction_cnn}")
    if args.interaction_cnn:
        logger.info(f"Interaction CNN Channels: {args.interaction_cnn_channels}")
    logger.info(f"Molecular GCN: {args.molecular_gcn}")
    if args.molecular_gcn:
        logger.info(f"GCN Layers: {args.gcn_layers}")
    logger.info(f"Effect Dual Branch: {args.effect_dual_branch}")
    logger.info(f"Effect Raw Global Branch: {args.effect_raw_global_branch}")
    logger.info(f"Init From: {args.init_from}")
    logger.info(f"Init From v3: {args.init_from_v3}")
    logger.info(f"Freeze Backbone Epochs: {args.freeze_backbone_epochs}")
    logger.info(f"Monitor Metric: {args.monitor_metric}")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Fast Index Collate: {args.fast_index_collate}")
    logger.info(f"DataLoader Workers: {args.num_workers}")
    logger.info("=" * 60)




    base_dir = str(RUNTIME_ROOT)
    feat_dir = str(FEATURE_DIR)
    data_dir = args.data_dir or str(REPOSITORY_ROOT / 'data' / args.task)

    logger.info("Loading pretrained features...")
    with open(os.path.join(feat_dir, 'compound_features_bermol_tokens.pkl'), 'rb') as f:
        comp_feats = pickle.load(f)
    with open(os.path.join(feat_dir, 'protein_features_esm2_residues.pkl'), 'rb') as f:
        prot_feats = pickle.load(f)
    logger.info(f"BerMol tokens: {len(comp_feats)} chemicals")
    logger.info(f"ESM2 residues: {len(prot_feats)} proteins")




    train_file = os.path.join(data_dir, f"effect_{args.task}_{args.split}_train.csv")
    val_file = os.path.join(data_dir, f"effect_{args.task}_{args.split}_val.csv")
    test_file = os.path.join(data_dir, f"effect_{args.task}_{args.split}_test.csv")
    logger.info(f"Data dir: {data_dir}")

    train_df = pd.read_csv(train_file, low_memory=False)
    val_df = pd.read_csv(val_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)

    logger.info(f"Original Train: {len(train_df)} ({(train_df['label']==1).sum()} pos + {(train_df['label']==0).sum()} neg)")


    if args.downsample:
        ratio = int(args.downsample.split(':')[1])
        pos = train_df[train_df['label'] == 1]
        neg = train_df[train_df['label'] == 0]
        n_neg = len(pos) * ratio
        neg_sampled = neg.sample(n=min(n_neg, len(neg)), random_state=args.seed)
        train_df = pd.concat([pos, neg_sampled]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        logger.info(f"Downsampled Train (1:{ratio}): {len(train_df)} ({len(pos)} pos + {len(neg_sampled)} neg)")
        exp_name = f"{exp_name}_1to{ratio}"

    logger.info(f"Train: {len(train_df)} ({(train_df['label']==1).sum()} pos + {(train_df['label']==0).sum()} neg)")
    logger.info(f"Val:   {len(val_df)} ({(val_df['label']==1).sum()} pos + {(val_df['label']==0).sum()} neg)")
    logger.info(f"Test:  {len(test_df)} ({(test_df['label']==1).sum()} pos + {(test_df['label']==0).sum()} neg)")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    chem_smiles_map = (
        all_df.dropna(subset=['chemical_name', 'smiles'])
              .drop_duplicates('chemical_name')
              .set_index('chemical_name')['smiles']
              .astype(str)
              .to_dict()
    )

    if args.molecular_gcn:
        extra_csvs = [str(REPOSITORY_ROOT / 'data' / 'binding' / 'combined_balanced_n72045_with_seq.csv')]
        extra_csvs += sorted(glob.glob(os.path.join(data_dir, 'effect_*.csv')))
        for csv_path in extra_csvs:
            try:
                edf = pd.read_csv(csv_path, usecols=['chemical_name', 'smiles'], low_memory=False)
                edf = edf.dropna(subset=['chemical_name', 'smiles']).drop_duplicates('chemical_name')
                for row in edf.itertuples(index=False):
                    if row.chemical_name not in chem_smiles_map:
                        chem_smiles_map[row.chemical_name] = str(row.smiles)
            except Exception:
                pass
        logger.info(f"SMILES map: {len(chem_smiles_map)} chemicals (effect + binding)")




    logger.info("Building shared entity tensors...")
    cache_path = args.entity_cache or default_entity_cache_path(
        base_dir, f'effect_{args.task}', args.split, args.max_chem_tokens,
        args.max_protein_length, args.molecular_gcn
    )
    shared = build_or_load_entity_tensors(
        cache_path, comp_feats, prot_feats,
        max_chem_tokens=args.max_chem_tokens,
        max_protein_length=args.max_protein_length,
        build_chem_adj=args.molecular_gcn,
        chem_smiles_map=chem_smiles_map,
        rebuild=args.rebuild_entity_cache
    )
    if args.prepare_data_only:
        logger.info("prepare_data_only set; exiting after entity tensor cache is ready.")
        return

    chem_col = 'chemical_name'
    prot_col = 'uniprot_id'
    label_col = 'label'

    train_dataset = FusionDataset(
        train_df, comp_feats, prot_feats,
        chem_col, prot_col, label_col,
        use_residue_attention=True, shared_tensors=shared,
        use_molecular_gcn=args.molecular_gcn,
        return_indices=args.fast_index_collate
    )
    val_dataset = FusionDataset(
        val_df, comp_feats, prot_feats,
        chem_col, prot_col, label_col,
        use_residue_attention=True, shared_tensors=shared,
        use_molecular_gcn=args.molecular_gcn,
        return_indices=args.fast_index_collate
    )
    test_dataset = FusionDataset(
        test_df, comp_feats, prot_feats,
        chem_col, prot_col, label_col,
        use_residue_attention=True, shared_tensors=shared,
        use_molecular_gcn=args.molecular_gcn,
        return_indices=args.fast_index_collate
    )

    collate = make_index_collate_fn(shared) if args.fast_index_collate else seq_collate_fn
    n_workers = args.num_workers
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=n_workers, pin_memory=True,
                              persistent_workers=n_workers > 0, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=n_workers, pin_memory=True,
                            persistent_workers=n_workers > 0, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=n_workers, pin_memory=True,
                             persistent_workers=n_workers > 0, collate_fn=collate)




    model_cls = EffectModelFusion if (args.effect_dual_branch or args.effect_raw_global_branch) else BindingModelFusion
    model_kwargs = dict(
        chemberta_dim=768,
        esm2_dim=1280,
        hidden_dim=args.hidden_dim,
        interaction_dim=args.hidden_dim * 2,
        dropout=args.dropout,
        use_residue_attention=True,
        use_toxprint=False,
        use_multi_granularity=args.multi_granularity,
        use_interaction_cnn=args.interaction_cnn,
        interaction_cnn_channels=args.interaction_cnn_channels,
        use_molecular_gcn=args.molecular_gcn,
        gcn_layers=args.gcn_layers
    )
    if args.effect_dual_branch or args.effect_raw_global_branch:
        model_kwargs['use_projected_global_branch'] = args.effect_dual_branch
        model_kwargs['use_raw_global_branch'] = args.effect_raw_global_branch
    model = model_cls(**model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")




    init_path = args.init_from
    if args.init_from_v3:
        if not args.init_from:
            raise ValueError("--init_from_v3 requires an explicit --init_from checkpoint path")
    loaded_param_names = set()
    if init_path:
        loaded_param_names = load_partial_checkpoint(model, init_path, device)
        if not loaded_param_names:
            raise RuntimeError(
                "Warm-start loaded zero trainable parameters. "
                "Check hidden_dim/model flags against the v3 checkpoint."
            )




    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    warmup_epochs = max(1, int(args.num_epochs * args.warmup_ratio)) if args.warmup_ratio > 0 else 0
    if warmup_epochs > 0:
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.num_epochs - warmup_epochs, eta_min=args.lr / 10
        )
        scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                                 milestones=[warmup_epochs])
        logger.info(f"Scheduler: LinearWarmup({warmup_epochs}ep) + CosineAnnealing({args.num_epochs - warmup_epochs}ep)")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.lr / 10
        )


    if args.pos_weight == -1:
        criterion = nn.BCEWithLogitsLoss()
        logger.info("pos_weight: disabled")
    else:
        if args.pos_weight > 0:
            pw = args.pos_weight
        else:
            n_pos = (train_df['label'] == 1).sum()
            n_neg = (train_df['label'] == 0).sum()
            pw = n_neg / n_pos
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
        logger.info(f"pos_weight: {pw:.2f}")
    scaler = GradScaler('cuda')


    ema = EMAModel(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema:
        logger.info(f"EMA enabled: decay={args.ema_decay}")

    if loaded_param_names and args.freeze_backbone_epochs > 0:
        set_params_trainable(model, loaded_param_names, trainable=False)
        logger.info(f"Trainable parameters after freeze: {count_trainable_params(model):,}")




    best_score = float('-inf')
    no_improve = 0
    ckpt_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{exp_name}_seed{args.seed}_best.pth')


    use_amp = False
    logger.info("AMP disabled for BilinearAttention (numerical stability)")

    for epoch in range(args.num_epochs):
        if loaded_param_names and args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs:
            set_params_trainable(model, loaded_param_names, trainable=True)
            logger.info(f"Trainable parameters after unfreeze: {count_trainable_params(model):,}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            use_amp=use_amp,
            comp_noise=args.comp_noise,
            prot_noise=args.prot_noise,
            label_smoothing=args.label_smoothing,
            ema=ema
        )


        if ema:
            ema.apply_shadow(model)
        val_metrics = evaluate(model, val_loader, criterion, device)
        if ema:
            ema.restore(model)

        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUROC: {val_metrics['auroc']:.4f} | "
            f"Val AUPRC: {val_metrics['auprc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        current_score = monitor_score(val_metrics, args.monitor_metric)
        if current_score > best_score:
            best_score = current_score
            no_improve = 0
            if ema:
                ema.apply_shadow(model)
                torch.save(model.state_dict(), ckpt_path)
                ema.restore(model)
            else:
                torch.save(model.state_dict(), ckpt_path)
            logger.info(
                f"  -> New best! {args.monitor_metric.upper()}="
                f"{val_metrics[args.monitor_metric]:.4f}"
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break




    logger.info("=" * 60)
    logger.info("Loading best model for testing...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    best_f1_thr, best_f1 = 0.5, -1.0
    best_mcc_thr, best_mcc = 0.5, -1.0
    val_labels = val_metrics['labels']
    val_probs = val_metrics['probs']
    for thr in np.arange(0.01, 1.0, 0.01):
        val_preds = (val_probs >= thr).astype(int)
        f1 = f1_score(val_labels, val_preds)
        mcc = matthews_corrcoef(val_labels, val_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_f1_thr = float(thr)
        if mcc > best_mcc:
            best_mcc = mcc
            best_mcc_thr = float(thr)

    test_labels = test_metrics['labels']
    test_probs = test_metrics['probs']
    test_preds_f1_thr = (test_probs >= best_f1_thr).astype(int)
    test_preds_mcc_thr = (test_probs >= best_mcc_thr).astype(int)

    logger.info("=" * 60)
    logger.info(f"TEST RESULTS ({exp_name})")
    logger.info("=" * 60)
    logger.info(f"AUROC:    {test_metrics['auroc']:.4f}")
    logger.info(f"AUPRC:    {test_metrics['auprc']:.4f}")
    logger.info(f"F1:       {test_metrics['f1']:.4f}")
    logger.info(f"Accuracy: {test_metrics['acc']:.4f}")
    logger.info(f"MCC:      {test_metrics['mcc']:.4f}")
    logger.info(
        f"Val best F1 threshold: {best_f1_thr:.2f} "
        f"(F1={best_f1:.4f})"
    )
    logger.info(
        f"Test F1 @ F1-thr {best_f1_thr:.2f}: "
        f"{f1_score(test_labels, test_preds_f1_thr):.4f}"
    )
    logger.info(
        f"Test MCC @ F1-thr {best_f1_thr:.2f}: "
        f"{matthews_corrcoef(test_labels, test_preds_f1_thr):.4f}"
    )
    logger.info(
        f"Val best MCC threshold: {best_mcc_thr:.2f} "
        f"(MCC={best_mcc:.4f})"
    )
    logger.info(
        f"Test F1 @ MCC-thr {best_mcc_thr:.2f}: "
        f"{f1_score(test_labels, test_preds_mcc_thr):.4f}"
    )
    logger.info(
        f"Test MCC @ MCC-thr {best_mcc_thr:.2f}: "
        f"{matthews_corrcoef(test_labels, test_preds_mcc_thr):.4f}"
    )
    logger.info("=" * 60)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
