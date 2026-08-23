# Data and model assets

## Included

- Fixed binding, activation, and inhibition training/validation/test CSV files.
- Processed CTD pathway parquet inputs and curated AOP-Wiki mapping CSV files.
- Static KEGG GMT and common-background gene list used by portable CTD mapping.

No checkpoint, prediction, ORA, metric, validation, baseline, figure, or paper
artifact is included.

## Required external assets

| Asset | Purpose | Configuration / acquisition |
|---|---|---|
| `BerMolModel_base.pkl` | Chemical token embeddings | Obtain an authorized upstream copy; set `TOXCPI_BERMOL_MODEL` or pass `--bermol-model` |
| BerMol Python source | Load the BerMol weight | Obtain under upstream terms; set `TOXCPI_BERMOL_DIR` or pass `--bermol-dir` |
| `esm2_t33_650M_UR50D.pt` | Protein residue embeddings | Obtain through the official ESM distribution; set `TOXCPI_ESM2_MODEL` |
| `human_proteome_esm2_residues.h5` | Whole-proteome inference | Generate with `code/features/00_extract_human_proteome_esm2.py`; pass `--esm2-h5` |
| `human_proteome_universe.csv` | Protein IDs and sequence metadata | Generate with the ESM-2 extraction workflow; pass `--universe` |
| BerMol/ESM-2 feature caches | Training acceleration/input | Generate under a user-selected `TOXCPI_FEATURE_DIR` |

The historical feature cache was about 121 GB and is deliberately excluded.
Missing external assets and checkpoints produce explicit runtime errors; code
does not silently fall back to an author machine path.

Raw BindingDB, T3DB, APDB, ToxCast/InVitroDB, CTD, AOP-Wiki, KEGG and related
database snapshots are not included. See `THIRD_PARTY_NOTICES.md` for data
sources and attribution.
