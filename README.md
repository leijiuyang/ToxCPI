# ToxCPI

Independent binding, activation and inhibition classifiers, fixed datasets, and
CTD/AOP analysis scripts.

## Files and weights

- `code/`: models, feature generation, training and inference.
- `data/{binding,activation,inhibition}/`: 18 fixed train/validation/test files.
- `data/{CTD,AOP}/`: downstream mapping inputs.
- `downstream_five_chemicals/`: analysis scripts and static configuration.
- `code/manifests/fixed_splits.csv`: fixed-file identities and SHA-256 hashes.
- `code/manifests/weight_identity.csv`: six primary checkpoint identities.

[Fixed source release](https://github.com/leijiuyang/ToxCPI/releases/tag/v2026.11.22-r1m26).
[Trained weights and necessary downstream inputs](https://github.com/leijiuyang/ToxCPI/releases/tag/v2026.11.21-r1m26-rc2).
Check downloaded archives with the attached `SHA256SUMS.txt`. Extract them under
`runtime/` to create `weights/`, `predictions/` and `strict_gene_budget_inputs/`.
Feature caches and upstream BerMol/ESM-2 weights are not distributed.

## Environment and features

```bash
conda env create -f environment.yml
conda activate toxcpi
```

Obtain authorized BerMol code and `BerMolModel_base.pkl`; see
`code/third_party/BerMol/README.md`. ESM-2 is
[`esm2_t33_650M_UR50D`](https://github.com/facebookresearch/esm).

```bash
export TOXCPI_BERMOL_DIR=/path/to/BerMol/source
export TOXCPI_BERMOL_MODEL=/path/to/BerMolModel_base.pkl
export TOXCPI_ESM2_MODEL=/path/to/esm2_t33_650M_UR50D.pt
export TOXCPI_FEATURE_DIR=/path/to/features
python code/features/extract_sequence_features.py
```

Effect feature generation additionally requires `TOXCPI_PROTEIN_SEQUENCE_CSV`,
a fixed sequence snapshot with `uniprot_id` and `sequence`/`protein_sequence`.
Conflicting or missing sequence identities halt extraction.

## Training and evaluation

Use the packaged fixed splits; do not regenerate or rebalance them. Settings and
seeds are in `code/configs/`. Example cold-model training commands:

```bash
python code/experiments/train_fusion.py --split cold --split_dir data/binding \
  --chem_feat bermol --use_residue_attention --no_toxprint
python code/experiments/train_effect.py --task activation --split cold \
  --data_dir data/activation --effect_raw_global_branch --hidden_dim 256 --dropout 0.25
python code/experiments/train_effect.py --task inhibition --split cold \
  --data_dir data/inhibition --effect_raw_global_branch --hidden_dim 256 --dropout 0.3
```

For evaluation, supply a checkpoint and an identity-verified feature cache:

```bash
python code/inference/evaluate_fixed_checkpoint.py --task binding --split cold \
  --test-csv data/binding/test_cold.csv \
  --checkpoint runtime/weights/binding/fusion_cold_best.pth \
  --entity-cache /path/to/entity_features.pt --output-dir runtime/evaluation \
  --limit 0 --device cpu
```

`code/features/build_entity_cache.py` builds max64/768 entity tensors from the
BerMol-token and ESM-2-residue feature dictionaries. All commands accept `--help`.

## Inference and downstream analysis

```bash
python code/inference/predict_human_proteome.py \
  --binding-checkpoint /path/to/binding.pth \
  --activation-checkpoint /path/to/activation.pth \
  --inhibition-checkpoint /path/to/inhibition.pth \
  --input /path/to/chemicals.csv --chemical-index 0 --device auto \
  --universe /path/to/human_proteome_universe.csv \
  --esm2-h5 /path/to/human_proteome_esm2_residues.h5 \
  --output /path/to/predictions/all_human_proteome_scores_bisphenol_a.csv

python downstream_five_chemicals/scripts/run_ctd_mapping.py \
  --prediction-dir runtime/predictions --ctd-data-dir data/CTD \
  --chemical-list downstream_five_chemicals/config/five_chemicals.csv \
  --gmt downstream_five_chemicals/config/kegg_human.gmt \
  --background downstream_five_chemicals/config/common_background_genes.txt \
  --threshold 0.5 --fdr 0.05 --min-overlap 2 --output-dir runtime/route1_ctd

python downstream_five_chemicals/scripts/05_run_aop_ppi_and_summary.py \
  --prediction-dir runtime/predictions --aop-data-dir data/AOP \
  --chemical-list downstream_five_chemicals/config/five_chemicals.csv \
  --output-dir runtime/route2_aop

python downstream_five_chemicals/scripts/run_strict_gene_budget.py \
  --input-dir runtime/strict_gene_budget_inputs --outdir runtime/strict_gene_budget \
  --stage run
```

Chemical input requires `chemical_name`, `casrn`, `ctd_chemical_id` and `smiles`.
Prediction CSVs use `uniprot_id`, `gene_symbol`, and raw `P_binding`,
`P_activation`, `P_inhibition` values with `chemical_name`.

The final AOP input is the unchanged 91-row table
`data/AOP/human_aop_proteins_multilabel.csv`. Every required label must have raw
probability strictly >0.7 on the same explicit member; binding is checked only
when labelled. Each mapping is one node, with qualifying members and shared
prediction evidence recorded. No aggregate event/AOP/AO scores or ranks are used.

Representative model execution and the current downstream entries were checked;
full retraining and every supplementary workflow were not re-executed.
