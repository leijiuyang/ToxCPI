# Third-party notices

This file lists the external software and data resources used by ToxCPI.

## Code and model dependencies

- **BerMol / DTIAM** — used for chemical token representations. Relevant upstream: <https://github.com/CSUBioGroup/DTIAM>. BerMol source and `BerMolModel_base.pkl` are not included; dependency notes and the expected model checksum are retained under `code/third_party/BerMol/`.
- **tipFormer, MolTrans, TransformerCPI2.0, and PMF-CPI** — external baselines referenced by the study. Their source, checkpoints, adapters, predictions, and results are not included.
- **ESM-2** — protein residue representations; source and weights are not included. Upstream and licensing information: <https://github.com/facebookresearch/esm>.

The licenses above apply only to their respective third-party components, not to ToxCPI as a whole.

## Databases and pathway resources

- **Comparative Toxicogenomics Database (CTD)** — processed chemical–pathway and pathway–disease tables support Route 1 and Figure 5. CTD information: <https://catalystresearch.io/products/ctd>.
- **AOP-Wiki** — curated multilabel event data support Route 2. Downloads: <https://aopwiki.org/downloads>; content handbook: <https://aopwiki.org/handbooks/5>.
- **KEGG** — pathway gene sets and labels support ORA and figures. Resource information: <https://www.kegg.jp/kegg/>.
- **BindingDB, T3DB, APDB, ToxCast/InVitroDB, and UniProt** — source the model datasets and protein annotations. Raw snapshots are not distributed here. Users must obtain raw data directly under each provider's current terms.
