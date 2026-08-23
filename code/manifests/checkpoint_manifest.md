# Checkpoint policy

No trained checkpoint is included in this repository. Training writes new
weights below `runtime/checkpoints/`. Inference requires explicit
`--binding-checkpoint`, `--activation-checkpoint`, and `--inhibition-checkpoint`
paths. Optional warm-start training requires an explicit `--init_from` path.
