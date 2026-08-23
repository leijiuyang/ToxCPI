
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPOSITORY_ROOT / "code"
WORK = REPOSITORY_ROOT / "runtime" / "predict_human_proteome"
INPUT_DIR = WORK / "inputs"
PRED_DIR = WORK / "predictions"
TARGET_DIR = WORK / "target_sets"
LOG_DIR = WORK / "logs"

UNIVERSE = Path(os.environ.get(
    "TOXCPI_PROTEOME_UNIVERSE_CSV", INPUT_DIR / "human_proteome_universe.csv"
)).expanduser().resolve()
CHEMICALS = INPUT_DIR / "chemicals.csv"
H5 = Path(os.environ.get(
    "TOXCPI_PROTEOME_ESM2_H5",
    REPOSITORY_ROOT / "runtime" / "pretrained_features" / "human_proteome_esm2_residues.h5",
)).expanduser().resolve()
BERMOL_DIR = Path(os.environ.get(
    "TOXCPI_BERMOL_DIR", REPOSITORY_ROOT / "code" / "third_party" / "BerMol"
)).expanduser().resolve()
BERMOL = Path(os.environ.get(
    "TOXCPI_BERMOL_MODEL", BERMOL_DIR / "BerMolModel_base.pkl"
)).expanduser().resolve()
COMPOUND_CACHE = WORK / "features" / "compound_bermol_tokens_4chem.pkl"

MAX_CHEM_TOKENS = 64
MAX_PROTEIN_LENGTH = 768


CHEMICAL_ROWS = [
    {
        "chemical_name": "Bisphenol A",
        "casrn": "80-05-7",
        "ctd_chemical_id": "C006780",
        "smiles": "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1",
        "train_source": "binding+effect(act+inh)",
    },
    {
        "chemical_name": "Perfluorooctanoic acid",
        "casrn": "335-67-1",
        "ctd_chemical_id": "C023036",
        "smiles": "O=C(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "train_source": "binding+effect",
    },
    {
        "chemical_name": "Rotenone",
        "casrn": "83-79-4",
        "ctd_chemical_id": "D012402",
        "smiles": "C=C(C)[C@H]1Cc2c(ccc3c2O[C@@H]2COc4cc(OC)c(OC)cc4[C@@H]2C3=O)O1",
        "train_source": "binding+inhibition",
    },
    {
        "chemical_name": "2,3,7,8-Tetrachlorodibenzo-p-dioxin",
        "casrn": "1746-01-6",
        "ctd_chemical_id": "D013749",
        "smiles": "Clc1cc2c(cc1Cl)Oc1cc(Cl)c(Cl)cc1O2",
        "train_source": "binding-only",
    },
    {
        "chemical_name": "1-Hydroxypyrene",
        "casrn": "5315-79-7",
        "ctd_chemical_id": "C033146",
        "smiles": "C1=CC2=C3C(=C1)C=CC4=C(C=CC(=C43)C=C2)O",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "Benzanthrone",
        "casrn": "82-05-3",
        "ctd_chemical_id": "C012768",
        "smiles": "C1=CC=C2C(=C1)C3=CC=CC4=C3C(=CC=C4)C2=O",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "Benzo(a)pyrene",
        "casrn": "50-32-8",
        "ctd_chemical_id": "D001564",
        "smiles": "C1=CC=C2C3=C4C(=CC2=C1)C=CC5=C4C(=CC=C5)C=C3",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "Chlorpyrifos",
        "casrn": "2921-88-2",
        "ctd_chemical_id": "D004390",
        "smiles": "CCOP(=S)(OCC)OC1=NC(=C(C=C1Cl)Cl)Cl",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "DDT",
        "casrn": "50-29-3",
        "ctd_chemical_id": "D003634",
        "smiles": "C1=CC(=CC=C1C(C2=CC=C(C=C2)Cl)C(Cl)(Cl)Cl)Cl",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "p,p'-DDE",
        "casrn": "72-55-9",
        "ctd_chemical_id": "D003633",
        "smiles": "C1=CC(=CC=C1C(=C(Cl)Cl)C2=CC=C(C=C2)Cl)Cl",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "Di(2-ethylhexyl)phthalate",
        "casrn": "117-81-7",
        "ctd_chemical_id": "D004051",
        "smiles": "CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "Atrazine",
        "casrn": "1912-24-9",
        "ctd_chemical_id": "D001280",
        "smiles": "CCNC1=NC(=NC(=N1)Cl)NC(C)C",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "Vinclozolin",
        "casrn": "50471-44-8",
        "ctd_chemical_id": "C025643",
        "smiles": "CC1(C(=O)N(C(=O)O1)C2=CC(=CC(=C2)Cl)Cl)C=C",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "Dieldrin",
        "casrn": "60-57-1",
        "ctd_chemical_id": "D004026",
        "smiles": "C1C2C3C(C1C4C2O4)C5(C(=C(C3(C5(Cl)Cl)Cl)Cl)Cl)Cl",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "Tributyltin",
        "casrn": "688-73-3",
        "ctd_chemical_id": "C011559",
        "smiles": "CCCC[SnH](CCCC)CCCC",
        "train_source": "new_downstream_case",
    },
    {
        "chemical_name": "Diethylstilbestrol",
        "casrn": "56-53-1",
        "ctd_chemical_id": "D004054",
        "smiles": "CCC(=C(CC)C1=CC=C(C=C1)O)C2=CC=C(C=C2)O",
        "train_source": "new_downstream_case",
    },
]


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (LOG_DIR / "02_predict_first_chemical.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def load_chemicals_file(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Chemical input CSV not found: {path}")
    RDLogger.DisableLog("rdApp.*")
    df = pd.read_csv(path)
    required = {"chemical_name", "casrn", "ctd_chemical_id", "smiles"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Chemical input CSV is missing columns: {missing}")
    df["smiles"] = df["smiles"].map(canonical_smiles)
    return df


def load_universe(limit: int = 0) -> pd.DataFrame:
    if not UNIVERSE.exists():
        raise FileNotFoundError(
            f"Universe file not found: {UNIVERSE}. Set TOXCPI_PROTEOME_UNIVERSE_CSV; "
            "see DATA_AND_MODEL_ASSETS.md."
        )
    df = pd.read_csv(UNIVERSE)
    gene = df["gene_symbol"].fillna("").astype(str).str.strip()
    sub = df[df["valid_sequence"].astype(bool) & gene.ne("") & gene.ne("nan")].copy()
    sub["uniprot_id"] = sub["uniprot_id"].astype(str)
    sub["gene_symbol"] = sub["gene_symbol"].fillna("").astype(str).str.strip()
    sub["protein_name"] = sub["protein_name"].fillna("").astype(str)
    sub = sub.sort_values(["sequence_length", "uniprot_id"]).reset_index(drop=True)
    if limit > 0:
        sub = sub.head(limit).copy()
    return sub


def validate_h5_coverage(universe: pd.DataFrame) -> None:
    if not H5.exists():
        raise FileNotFoundError(
            f"HDF5 residue feature cache not found: {H5}. Set "
            "TOXCPI_PROTEOME_ESM2_H5; see DATA_AND_MODEL_ASSETS.md."
        )
    targets = set(universe["uniprot_id"])
    with h5py.File(H5, "r") as h5:
        keys = set(h5.keys())
    missing = sorted(targets - keys)
    log(f"HDF5 coverage keys={len(keys)} targets={len(targets)} missing={len(missing)}")
    if missing:
        raise RuntimeError(
            "HDF5 cache is incomplete; rerun Script 00 first. "
            f"Missing examples: {', '.join(missing[:20])}"
        )


def project_imports() -> None:
    project_str = str(CODE_DIR)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)


def load_models(
    device: torch.device, checkpoints: dict[str, Path]
) -> dict[str, torch.nn.Module]:
    project_imports()
    from model_fusion import BindingModelFusion, EffectModelFusion

    specs = {
        "binding": (
            BindingModelFusion(
                chemberta_dim=768,
                esm2_dim=1280,
                hidden_dim=128,
                interaction_dim=256,
                dropout=0.3,
                use_residue_attention=True,
                use_toxprint=False,
                use_multi_granularity=False,
            ),
            checkpoints["binding"],
        ),
        "activation": (
            EffectModelFusion(
                chemberta_dim=768,
                esm2_dim=1280,
                hidden_dim=256,
                interaction_dim=512,
                dropout=0.3,
                use_residue_attention=True,
                use_toxprint=False,
                use_multi_granularity=False,
                use_interaction_cnn=False,
                use_molecular_gcn=False,
                use_projected_global_branch=False,
                use_raw_global_branch=True,
            ),
            checkpoints["activation"],
        ),
        "inhibition": (
            EffectModelFusion(
                chemberta_dim=768,
                esm2_dim=1280,
                hidden_dim=256,
                interaction_dim=512,
                dropout=0.3,
                use_residue_attention=True,
                use_toxprint=False,
                use_multi_granularity=False,
                use_interaction_cnn=False,
                use_molecular_gcn=False,
                use_projected_global_branch=False,
                use_raw_global_branch=True,
            ),
            checkpoints["inhibition"],
        ),
    }
    models = {}
    for task, (model, ckpt) in specs.items():
        if not ckpt.exists():
            raise FileNotFoundError(f"{task} checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        models[task] = model
        log(f"loaded {task} checkpoint={ckpt.name}")
    return models


def load_or_extract_compound_tokens(chemical: pd.Series, device: torch.device) -> np.ndarray:
    COMPOUND_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if COMPOUND_CACHE.exists():
        with COMPOUND_CACHE.open("rb") as handle:
            cache = pickle.load(handle)
    key = str(chemical["chemical_name"])
    if key in cache:
        arr = cache[key]
    else:
        if not BERMOL.is_file():
            raise FileNotFoundError(
                "BerMol weight is missing. Set TOXCPI_BERMOL_MODEL to "
                "BerMolModel_base.pkl; see DATA_AND_MODEL_ASSETS.md."
            )
        if str(BERMOL_DIR) not in sys.path:
            sys.path.insert(0, str(BERMOL_DIR))
        try:
            import dill as pkl
        except ImportError:
            import pickle as pkl
        with BERMOL.open("rb") as handle:
            model = pkl.load(handle)
        model.model.to(device)
        model.model.eval()
        with torch.no_grad():
            sequence_output, _ = model.transform(str(chemical["smiles"]), str(device))
        arr = sequence_output.squeeze(0).detach().cpu().numpy().astype(np.float16)
        cache[key] = arr
        with COMPOUND_CACHE.open("wb") as handle:
            pickle.dump(cache, handle)
    if arr.ndim != 2 or arr.shape[1] != 768:
        raise RuntimeError(f"Unexpected BerMol token shape for {key}: {arr.shape}")
    log(f"compound tokens chemical={key} shape={arr.shape}")
    return arr[:MAX_CHEM_TOKENS].astype(np.float32, copy=False)


def train_uniprot_set(path: Path) -> set[str]:
    if not path.exists():
        log(f"WARN train file missing: {path}")
        return set()
    df = pd.read_csv(path, usecols=["uniprot_id"])
    return set(df["uniprot_id"].dropna().astype(str))


def load_train_sets() -> tuple[set[str], set[str], set[str]]:
    binding = train_uniprot_set(REPOSITORY_ROOT / "data" / "binding" / "train_cold.csv")
    activation = train_uniprot_set(REPOSITORY_ROOT / "data" / "activation" / "effect_activation_cold_train.csv")
    inhibition = train_uniprot_set(REPOSITORY_ROOT / "data" / "inhibition" / "effect_inhibition_cold_train.csv")
    log(f"train proteins binding={len(binding)} activation={len(activation)} inhibition={len(inhibition)}")
    return binding, activation, inhibition


def predict_all(
    chemical: pd.Series,
    universe: pd.DataFrame,
    models: dict[str, torch.nn.Module],
    chem_tokens: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    rows = []
    chem_one = torch.from_numpy(chem_tokens).to(device)
    chem_mask_one = torch.zeros(chem_one.size(0), dtype=torch.bool, device=device)
    start_time = time.time()
    with h5py.File(H5, "r") as h5:
        for start in range(0, len(universe), batch_size):
            batch = universe.iloc[start : start + batch_size]
            arrays = [
                h5[str(uid)][:MAX_PROTEIN_LENGTH].astype(np.float32, copy=False)
                for uid in batch["uniprot_id"]
            ]
            max_len = max(arr.shape[0] for arr in arrays)
            prot_tensor = torch.zeros(len(arrays), max_len, 1280, dtype=torch.float32, device=device)
            prot_mask = torch.ones(len(arrays), max_len, dtype=torch.bool, device=device)
            for idx, arr in enumerate(arrays):
                feat = torch.from_numpy(arr).to(device)
                prot_tensor[idx, : feat.size(0)] = feat
                prot_mask[idx, : feat.size(0)] = False

            chem_tensor = chem_one.unsqueeze(0).expand(len(arrays), -1, -1)
            chem_mask = chem_mask_one.unsqueeze(0).expand(len(arrays), -1)

            probs = {}
            with torch.no_grad():
                for task, model in models.items():
                    logits, *_ = model(
                        chem_tokens_feat=chem_tensor,
                        chem_mask=chem_mask,
                        prot_residue_feat=prot_tensor,
                        prot_mask=prot_mask,
                    )
                    probs[task] = torch.sigmoid(logits).detach().cpu().numpy()

            for i, (_, target) in enumerate(batch.iterrows()):
                rows.append(
                    {
                        "chemical_name": chemical["chemical_name"],
                        "casrn": chemical["casrn"],
                        "ctd_chemical_id": chemical["ctd_chemical_id"],
                        "smiles": chemical["smiles"],
                        "uniprot_id": target["uniprot_id"],
                        "gene_symbol": target["gene_symbol"],
                        "protein_name": target["protein_name"],
                        "sequence_length": int(target["sequence_length"]),
                        "P_binding": float(probs["binding"][i]),
                        "P_activation": float(probs["activation"][i]),
                        "P_inhibition": float(probs["inhibition"][i]),
                    }
                )
            done = min(start + batch_size, len(universe))
            if done % max(batch_size * 10, 1) == 0 or done == len(universe):
                rate = done / max(time.time() - start_time, 1e-6)
                log(f"predicted {done}/{len(universe)} proteins rate={rate:.1f}/s")
    return pd.DataFrame(rows)


def finalize_outputs(df: pd.DataFrame, binding_train: set[str], activation_train: set[str], inhibition_train: set[str]) -> pd.DataFrame:
    df["P_effect_any"] = df[["P_activation", "P_inhibition"]].max(axis=1)
    df["S_target"] = df["P_binding"] * (0.5 + 0.5 * df["P_effect_any"])
    for threshold in (0.5, 0.7):
        suffix = "" if threshold == 0.5 else "_strict"
        df[f"is_binding_positive{suffix}"] = df["P_binding"] >= threshold
        df[f"is_activation_positive{suffix}"] = df["P_activation"] >= threshold
        df[f"is_inhibition_positive{suffix}"] = df["P_inhibition"] >= threshold
        df[f"is_union_positive{suffix}"] = (
            df[f"is_binding_positive{suffix}"]
            | df[f"is_activation_positive{suffix}"]
            | df[f"is_inhibition_positive{suffix}"]
        )
    df["in_binding_train"] = df["uniprot_id"].isin(binding_train)
    df["in_activation_train"] = df["uniprot_id"].isin(activation_train)
    df["in_inhibition_train"] = df["uniprot_id"].isin(inhibition_train)
    df["effect_extrapolated"] = ~(df["in_activation_train"] | df["in_inhibition_train"])
    df["rank_binding"] = df["P_binding"].rank(ascending=False, method="min").astype(int)
    df["rank_activation"] = df["P_activation"].rank(ascending=False, method="min").astype(int)
    df["rank_inhibition"] = df["P_inhibition"].rank(ascending=False, method="min").astype(int)
    df["rank_S_target"] = df["S_target"].rank(ascending=False, method="min").astype(int)
    return df


def write_outputs(df: pd.DataFrame, chemical_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_dir = output_path.parent / "target_sets"
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = chemical_name.lower().replace(",", "").replace(" ", "_").replace("/", "_")

    df.to_csv(output_path, index=False)
    log(f"wrote {output_path}")

    diagnostics = []
    for threshold, suffix in ((0.5, ""), (0.7, "_strict")):
        diagnostics.append(
            {
                "chemical_name": chemical_name,
                "threshold": threshold,
                "n_binding_pos": int(df[f"is_binding_positive{suffix}"].sum()),
                "n_activation_pos": int(df[f"is_activation_positive{suffix}"].sum()),
                "n_inhibition_pos": int(df[f"is_inhibition_positive{suffix}"].sum()),
                "n_union_pos": int(df[f"is_union_positive{suffix}"].sum()),
                "union_fraction_of_universe": float(df[f"is_union_positive{suffix}"].mean()),
            }
        )
        union = df[df[f"is_union_positive{suffix}"]].copy()
        union_path = target_dir / f"union_positive_targets_{slug}_threshold_{threshold:.1f}.csv"
        union.to_csv(union_path, index=False)
        log(f"wrote {union_path} rows={len(union)}")
    diag_path = output_path.parent / f"positive_diagnostics_{slug}.csv"
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
    log(f"wrote {diag_path}")

    top_specs = {
        "binding": "P_binding",
        "activation": "P_activation",
        "inhibition": "P_inhibition",
        "S_target": "S_target",
    }
    for name, column in top_specs.items():
        top = df.sort_values(column, ascending=False).head(10)
        top_path = output_path.parent / f"top10_{name}_{slug}.csv"
        top.to_csv(top_path, index=False)
        log(f"wrote {top_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ToxCPI whole-proteome inference for one chemical")
    parser.add_argument("--input", required=True, type=Path,
                        help="Chemical CSV with chemical_name, casrn, ctd_chemical_id, and smiles")
    parser.add_argument("--output", required=True, type=Path,
                        help="Prediction CSV to create; parent directories are created")
    parser.add_argument("--binding-checkpoint", required=True, type=Path)
    parser.add_argument("--activation-checkpoint", required=True, type=Path)
    parser.add_argument("--inhibition-checkpoint", required=True, type=Path)
    parser.add_argument("--universe", type=Path, default=UNIVERSE,
                        help="Human-proteome universe CSV")
    parser.add_argument("--esm2-h5", type=Path, default=H5,
                        help="ESM-2 residue-feature HDF5")
    parser.add_argument("--bermol-dir", type=Path, default=BERMOL_DIR)
    parser.add_argument("--bermol-model", type=Path, default=BERMOL)
    parser.add_argument("--compound-cache", type=Path, default=COMPOUND_CACHE)
    parser.add_argument("--chemical-index", type=int, default=0, help="0-based row in --input")
    parser.add_argument(
        "--device", default="auto",
        help="Torch device (default: auto, selects cuda:0 when available, otherwise cpu)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-proteins", type=int, default=0)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    device_name = "cuda:0" if requested == "auto" and torch.cuda.is_available() else (
        "cpu" if requested == "auto" else requested
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def main() -> int:
    global UNIVERSE, H5, BERMOL_DIR, BERMOL, COMPOUND_CACHE, LOG_DIR
    args = parse_args()
    UNIVERSE = args.universe.expanduser().resolve()
    H5 = args.esm2_h5.expanduser().resolve()
    BERMOL_DIR = args.bermol_dir.expanduser().resolve()
    BERMOL = args.bermol_model.expanduser().resolve()
    COMPOUND_CACHE = args.compound_cache.expanduser().resolve()
    LOG_DIR = args.output.expanduser().resolve().parent / "logs"
    checkpoints = {
        "binding": args.binding_checkpoint.expanduser().resolve(),
        "activation": args.activation_checkpoint.expanduser().resolve(),
        "inhibition": args.inhibition_checkpoint.expanduser().resolve(),
    }
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.set_grad_enabled(False)
    log("START Script 02 first chemical full-human-proteome prediction")

    device = resolve_device(args.device)

    chemicals = load_chemicals_file(args.input.expanduser().resolve())
    if args.chemical_index < 0 or args.chemical_index >= len(chemicals):
        raise IndexError(f"--chemical-index out of range: {args.chemical_index}")
    chemical = chemicals.iloc[args.chemical_index]
    log(f"chemical index={args.chemical_index} name={chemical['chemical_name']} smiles={chemical['smiles']}")

    universe = load_universe(args.limit_proteins)
    log(f"universe targets={len(universe)} limit={args.limit_proteins}")
    validate_h5_coverage(universe)

    models = load_models(device, checkpoints)
    chem_tokens = load_or_extract_compound_tokens(chemical, device)
    binding_train, activation_train, inhibition_train = load_train_sets()
    scores = predict_all(chemical, universe, models, chem_tokens, device, args.batch_size)
    scores = finalize_outputs(scores, binding_train, activation_train, inhibition_train)
    write_outputs(scores, str(chemical["chemical_name"]), args.output.expanduser().resolve())
    log("DONE Script 02")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
