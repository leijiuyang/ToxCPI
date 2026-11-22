# Downstream analysis

Run commands from the repository root; see the root README.

- `scripts/run_ctd_mapping.py`: CTD pathway comparison.
- `scripts/05_run_aop_ppi_and_summary.py`: current 91-mapping AOP matching.
- `scripts/run_strict_gene_budget.py`: fixed-score gene-budget comparison.
- `config/`: chemical list, gene sets and background.

The historical family-display and scoring-table helpers reject the current AOP
schema. Use the final matcher outputs directly.
