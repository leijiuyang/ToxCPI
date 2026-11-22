# Final AOP mapping

`human_aop_proteins_multilabel.csv` is the sole input for current AOP matching:
91 mappings, 62 AOPs and 147 distinct UniProt identifiers. The scientific table
is unchanged. Historical inventory and helper files are provenance only.

- `Use_For_AOP_Matching` controls inclusion.
- `Event_Labels` lists required raw prediction tasks.
- `Resolved_UniProt_ID` lists explicit members separated by semicolons.
- Every required probability must be strictly >0.7 on the same member.
- Binding is required only when included in `Event_Labels`.
- Each mapping remains one node; all qualifying members are recorded.
- Shared chemical/protein/task predictions are shared evidence.

Do not expand members from historical helper tables. No aggregate event/AOP/AO
scores or rankings are assigned. See the root README for the command.
