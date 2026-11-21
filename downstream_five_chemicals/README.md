# CTD/AOP downstream workflows

`scripts/run_ctd_mapping.py` performs Route 1 target selection, KEGG ORA, CTD
pathway comparison and pathway–disease mapping. The historical `04b_...` and
`06_compare_...` commands remain compatibility interfaces for the portable CTD code.

`scripts/05_run_aop_ppi_and_summary.py` now implements the current unweighted Route 2
through `aop_matching.py`. Its sole formal mapping input is the unchanged final
91-row `data/AOP/human_aop_proteins_multilabel.csv`. It never reconstructs the old
169-row entry or reads historical family/manual definitions. Each required raw
task probability must be strictly >0.7 on the same explicit protein/member. Binding
is required only when labelled. Both activation and inhibition associations may
remain. Each source mapping remains one node, with all qualifying members recorded.

Outputs are `route2_event_mapping_nodes.csv`, `route2_member_label_evidence.csv`,
`route2_unique_model_evidence.csv`, `route2_match_counts.csv`, the identified input
copy and `run_metadata.json`. Counts are descriptive. No aggregate event/AOP/AO
scores, ranks, representative-member scores or PPI placeholders are generated.
Shared evidence IDs prevent repeated mapped nodes being mistaken for independent
chemical–protein–task predictions.

Run `python -m unittest discover -s downstream_five_chemicals/scripts
-p test_aop_matching.py -v` from the repository root. See the root README and
`reproduction/REPRODUCE.txt` for actual commands. `config/five_chemicals.csv` defines
the five archived cases; an explicit chemical list is required.

`scripts/run_strict_gene_budget.py` is the portable fixed-score budget adapter;
its hashed input attachment reproduces 80 original metric rows. It is separate
from Route 2. `06_build_route2_family_display.py` and `build_s4_ctd_aop_tables.py`
are historical scoring helpers and reject final mapping/output schemas. Historical
sources do not establish current manuscript table numbering.

Selected archived predictions and validation outputs are prepared as separate
release attachments, outside ordinary Git history. Their publication status and
precise validation limits are recorded under `reproduction/`.
