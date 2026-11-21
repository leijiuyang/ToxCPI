# Checkpoint delivery status

Trained ToxCPI weights are prepared as six original-byte candidate release assets,
with original filenames, provenance and SHA-256 in `weight_identity.csv`. They are
not stored in ordinary Git history. See `reproduction/ASSETS.txt` and
`reproduction/PUBLICATION_STATUS.txt`; use the matching fixed Release for actual download availability.

All six load with `strict=True` and have passed CPU inference on 32 fixed test
records per task/split. Binding uses the full shared feature cache; historical
effect-specific cache archives are unreadable, so representative effect validation
explicitly uses the shared frozen BerMol/ESM-2 features. This does not certify
historical RNG lineage, full test metrics, or all supplementary experimental runs.

Warm BPA-AR attribution uses `fusion_warm_best.pth`. External direction evaluation
uses the cold effect models; controlled five-seed experiments use separate models.
Upstream BerMol/ESM-2 weights are not trained ToxCPI weights. No partial/non-strict
state loading is used to validate the supplied primary checkpoints.
