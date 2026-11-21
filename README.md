# ToxCPI

ToxCPI is a chemical–protein interaction framework with independent binding,
activation, and inhibition classifiers. This repository contains model,
feature-generation, training, inference, and CTD/AOP mapping code plus the fixed
training and processed mapping data.

## Repository scope

This repository **does not include pretrained ToxCPI checkpoints, precomputed
human-proteome predictions, CTD/AOP result tables, baseline results, manuscript
files, or paper figures**. Six original trained checkpoints and fixed downstream inputs have been prepared
as candidate release attachments, with hashes and provenance. They have not yet
been uploaded. See [publication status](reproduction/PUBLICATION_STATUS.txt), the
[reproduction guide](reproduction/REPRODUCE.txt), and the
[analysis matrix](code/manifests/analysis_matrix.csv) for the exact supported scope.
Users can also train locally and supply their own checkpoint paths explicitly.

```text
code/                          model, feature, training, inference, tests/configs
data/{binding,activation,inhibition}/  fixed training splits
data/{CTD,AOP}/                processed downstream mapping inputs
downstream_five_chemicals/     CTD/AOP mapping and table-building scripts
runtime/                       generated locally and ignored by Git
```

## Recommended workflow

```text
install environment
  -> prepare/generate external pretrained features
  -> train binding, activation, and inhibition
  -> obtain three checkpoint files
  -> run inference for chemicals against the human proteome
  -> create prediction CSV files
  -> run CTD mapping
  -> run the final unweighted AOP mechanistic-hypothesis matcher
```

### 1. Environment

```bash
conda env create -f environment.yml
conda activate toxcpi
```

This is a candidate reproduction environment, not a proven original training
lock. A fresh Python 3.9 environment was validated for final AOP matching using
`requirements-aop.txt`; the full model environment was not clean-installed.
See [environment validation](reproduction/ENVIRONMENT.txt) for actual versions,
CPU/GPU availability, the separate strict-budget environment and limitations.

### 2. External feature/model assets

BerMol and ESM-2 weights and large generated feature caches are not included.
Obtain them under their upstream terms and configure them with environment
variables such as:

```bash
export TOXCPI_BERMOL_DIR=/path/to/authorized/BerMol/source
export TOXCPI_BERMOL_MODEL=/path/to/BerMolModel_base.pkl
export TOXCPI_ESM2_MODEL=/path/to/esm2_t33_650M_UR50D.pt
export TOXCPI_FEATURE_DIR=/path/to/generated/features
```

### 3. Train the three models

Run from the repository root. Outputs are written below ignored `runtime/`.

```bash
python code/experiments/train_fusion.py \
  --split cold --split_dir data/binding \
  --chem_feat bermol --use_residue_attention --no_toxprint

python code/experiments/train_effect.py \
  --task activation --split cold --data_dir data/activation \
  --effect_raw_global_branch --hidden_dim 256 --dropout 0.25

python code/experiments/train_effect.py \
  --task inhibition --split cold --data_dir data/inhibition \
  --effect_raw_global_branch --hidden_dim 256 --dropout 0.3
```

These examples show interfaces. Complete fixed-run settings and seeds are
recorded in `code/configs/formal_runs/`; command defaults alone do not certify
the historical training trajectory. No retraining was performed in this audit.
For evaluation without retraining, see `code/inference/evaluate_fixed_checkpoint.py`.

Warm-starting is optional and always requires an explicit `--init_from
/path/to/checkpoint.pth`; no repository checkpoint path is assumed.

### 4. Inference

The chemical input CSV must contain `chemical_name`, `casrn`,
`ctd_chemical_id`, and `smiles`. The proteome universe and ESM-2 HDF5 asset are
provided explicitly through `--universe` and `--esm2-h5`.

```bash
python code/inference/predict_human_proteome.py \
  --binding-checkpoint /path/to/binding.pth \
  --activation-checkpoint /path/to/activation.pth \
  --inhibition-checkpoint /path/to/inhibition.pth \
  --input /path/to/chemicals.csv \
  --chemical-index 0 \
  --device auto \
  --universe /path/to/human_proteome_universe.csv \
  --esm2-h5 /path/to/human_proteome_esm2_residues.h5 \
  --output /path/to/predictions/all_human_proteome_scores_bisphenol_a.csv
```

Missing checkpoints are checked at runtime and reported with the exact path.
Weights are not loaded during module import. Parent output directories are
created automatically. `--device auto` is the default and selects CUDA when
available, otherwise CPU.

### 5. CTD mapping

```bash
python downstream_five_chemicals/scripts/run_ctd_mapping.py \
  --prediction-dir /path/to/predictions \
  --ctd-data-dir data/CTD \
  --chemical-list downstream_five_chemicals/config/five_chemicals.csv \
  --gmt downstream_five_chemicals/config/kegg_human.gmt \
  --background downstream_five_chemicals/config/common_background_genes.txt \
  --threshold 0.5 --fdr 0.05 --min-overlap 2 \
  --output-dir /path/to/results/route1_ctd
```

The CTD summary reports both raw significant CTD pathways and the subset
comparable to the supplied GMT. CTD recovery and Jaccard use the comparable
subset; model-support overlap is therefore evaluated on the same pathway
universe as ORA.

### 6. AOP mechanistic-hypothesis mappings

The final mapping table is `data/AOP/human_aop_proteins_multilabel.csv`.
It contains 91 protein-concept/event mappings covering 62 AOPs, including
`AOP_Title`. Each family or complex candidate group occupies one row;
semicolon-separated UniProt accessions identify its members. These correspond
to 241 member-level associations and 147 distinct human proteins.

This version uses `Use_For_AOP_Matching` and has no `Position_Weight` column.
It supports mechanistic hypotheses, without an AOP weighting or aggregate-score
method. See [the AOP data notes](data/AOP/README.md) for provenance and field
interpretation.

The original `05_run_aop_ppi_and_summary.py` entry now invokes the final matcher:

```bash
python downstream_five_chemicals/scripts/05_run_aop_ppi_and_summary.py \
  --prediction-dir runtime/predictions --aop-data-dir data/AOP \
  --chemical-list downstream_five_chemicals/config/five_chemicals.csv \
  --output-dir runtime/final_route2
```

It reads only the final mapping CSV. Every required raw task probability must be
strictly `>0.7` on the same explicitly listed member; binding is checked only when
the event labels require it. Activation and inhibition associations can both be
retained. Each mapping is one event-mapping node regardless of member count.
Outputs record all qualifying members and shared chemical–protein–task evidence;
no aggregate event/AOP/AO scores or rankings are generated.

All 455 mapping nodes for five archived prediction files were independently checked;
all 25 illustrative Table S17 associations and titles matched. This validates final
matching on archived predictions; it does not mean the full proteome predictions,
all supplementary analyses, or the model training were rerun.

Historical family-display and scoring Excel scripts are retained for provenance
and explicitly reject final-schema inputs/outputs. The original scoring entry is
available in base commit `3de0496dbfbc1b7600a52efae8a617866c8a8fde`.

## Prediction CSV contract

CTD mapping requires `chemical_name`, `gene_symbol`, `P_binding`,
`P_activation`, and `P_inhibition`. Final AOP matching requires chemical identity,
`uniprot_id`, and the raw probabilities for required task labels. Missing required
predictions cannot qualify; malformed probabilities and duplicate identities halt
with an error. A whole semicolon-delimited mapping cell is never one accession.

Historical scoring scripts additionally require their earlier rank, context and
score fields; these are not prerequisites for final Route 2 matching. Neither
calibrated external direction scores nor `S_target` replace the raw task probabilities.

## Data

| Directory | Approx. size | Purpose and source notes |
|---|---:|---|
| `data/binding` | 270 MiB | Fixed binding table/splits derived from BindingDB, T3DB, APDB, ToxCast and related sources |
| `data/activation` | 4.2 MiB | Fixed activation warm/cold splits |
| `data/inhibition` | 8.4 MiB | Fixed inhibition warm/cold splits |
| `data/CTD` | 11 MiB | Processed CTD chemical–pathway and pathway–disease inputs |
| `data/AOP` | 151 KiB | Final AOP mechanistic-hypothesis mapping table and historical source/reference tables |

The fixed scientific splits are provided without sampling or compression.
Users are responsible for complying with the terms of the upstream data
sources and external model assets.
