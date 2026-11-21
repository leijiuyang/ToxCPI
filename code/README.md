# ToxCPI model code

This directory contains model architecture, data loading, feature generation,
training, inference, experiment configuration, tests, technical documentation,
and the minimal BerMol compatibility/provenance files.

The production models are `BindingModelFusion` and `EffectModelFusion` in
`model_fusion.py`, with bidirectional bilinear attention from `attention.py`.
Binding training is in `experiments/train_fusion.py`; activation/inhibition
training is in `experiments/train_effect.py`; whole-proteome inference is in
`inference/predict_human_proteome.py`.

Pretrained checkpoints are not included. YAML and registry checkpoint fields are
`null`; pass checkpoint paths explicitly. Runtime outputs and feature caches
default to ignored `runtime/` or to paths supplied by the user. See the root
README for runnable commands and external asset environment variables.


## Reproduction candidate

See `../reproduction/REPRODUCE.txt` and `manifests/analysis_matrix.csv` for fixed
inputs, representative evaluation, current final AOP matching and remaining SI
dependencies. `inference/evaluate_fixed_checkpoint.py` strictly loads a primary
checkpoint and evaluates explicit fixed CSV records without training/resplitting.
`features/build_entity_cache.py` builds original max64/768 tensors from explicit
trusted feature dictionaries. See `manifests/weight_identity.csv` for six original
weight identities; release attachments are prepared locally, not yet public.
