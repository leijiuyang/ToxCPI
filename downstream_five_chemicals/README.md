# CTD/AOP downstream workflows

This directory contains scripts and static configuration needed to regenerate
downstream results from ToxCPI prediction CSV files. It contains no predictions,
ORA/metrics, CTD/AOP results, baseline scores, validation artifacts, or final
supplementary tables.

`scripts/run_ctd_mapping.py` performs thresholded target selection, KEGG ORA,
predicted-vs-CTD pathway comparison, and pathway–disease mapping. The historical
`04b_...` and `06_compare_...` commands are compatibility entry points to this
portable implementation.

The AOP data were updated on 2026-11-21 to the final 91-row, unweighted
mechanistic-hypothesis mapping table. See [the AOP data notes](../data/AOP/README.md).
The historical scripts below were retained unchanged and are not compatible
with that table's new matching flag and inline, semicolon-separated members.

`scripts/05_run_aop_ppi_and_summary.py` implements the earlier strict
label-matched AOP/AO scoring workflow. `scripts/06_build_route2_family_display.py`
creates its family-expanded event view. `scripts/build_s4_ctd_aop_tables.py`
rebuilds historical supplementary Excel tables from Route 1/Route 2 outputs.
Do not run these scripts against the updated AOP table or interpret them as
implementing the final unweighted matching approach. They require the earlier
inputs and schema; this data update did not regenerate their outputs.

Every executable accepts `--help`. Inputs, CTD/AOP data, thresholds, chemical
list, and outputs are command-line parameters. The default five-chemical list is
`config/five_chemicals.csv`; users may provide another list with the same schema.
See the repository README for commands and the exact prediction CSV contract.

`config/kegg_human.gmt` and `config/common_background_genes.txt` are the static
gene-set and background inputs used by the pathway enrichment workflow.
