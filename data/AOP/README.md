# AOP mechanistic-hypothesis mapping data

Updated: 2026-11-21.

## Final table

`human_aop_proteins_multilabel.csv` is the authoritative final mapping table.
Its filename is retained from the project, but its contents replace the earlier
169-row input. It contains 91 mappings across 62 distinct AOPs and 85 AOP/event
nodes. One protein concept, family or complex candidate group occupies one row.
Expanding its member lists would give 241 member associations involving 147
distinct human UniProt accessions; these are not 241 independent AOP events.

The author-reviewed 135-row baseline yielded 91 final mappings by excluding
39 mappings and consolidating five additional rows in four within-event merge
groups. Protein identity corrections, task directions and member ranges follow
the accepted review decisions. Original records remain in the local review
archive; source row numbers and change references are recorded in `Mapping_Notes`.

The file is the approved `最终映射表_机制假设用_无权重评分_20261121.csv`,
including AOP titles joined by `AOP_ID` from the project's source inventory.

## Fields

- `AOP_ID` / `AOP_Title`: pathway identifier and title.
- `Event_ID` / `Event_Title` / `Event_Type`: event identity and its original role.
- `Protein_Name` / `Family_Name`: the mapped protein concept or candidate group.
- `UniProt_ID`, `Manual_UniProt_ID`, `Resolved_UniProt_ID`: effective accessions;
  multiple members are separated by semicolons. `Mapping_Notes` pairs member
  gene symbols with accessions. Do not treat an entire semicolon-separated cell
  as one protein accession or assume a group requires all its members together.
- `Original_UniProt_ID`: historical identifier retained for provenance; use the
  resolved accessions for the final mapping. A blank `Protein_PR_ID` means a new
  member/group PR identifier was not independently verified, not a missing
  UniProt mapping. A retained PR can describe the original source concept;
  it does not necessarily describe every expanded member.
- `Event_Labels`, `Expected_Effect`, `Primary_Effect`, `Has_*`: the reviewed
  binding/activation/inhibition interpretation. Functional proxies retain their
  event-specific limitations; they do not establish an entire AOP or outcome.
- `Use_For_AOP_Matching`: inclusion in the final mechanistic-hypothesis mapping;
  all 91 retained rows are `True`.
- `Binding_Context` / `Use_For_ToxCPI_Binding`: existing task-specific context
  and historical binding eligibility, distinct from overall matching inclusion.

There is no `Position_Weight` or `Use_In_AOP_Score` column. This table defines
neither a weighting scheme nor an aggregate AOP score. Group membership alone
does not validate every member as a direct chemical target or binding site.

Every effective accession was matched to the study's local UniProt human
annotation, a valid sequence and the protein lists of five existing proteome
prediction files. This establishes identifier availability, not independent
expert validation or equal effects across members/species. GABRA5 and GABRA6
use the corrected accessions P31644 and Q16445.

## Other files and compatibility

`human_aop_complete_multilabel.csv` is the historical 69-AOP source inventory.
Its chain, coverage counts and summary fields were not regenerated for the
final 62-AOP mapping subset. It is not the final mapping table.

`aop_family_expansion_map.csv`,
`aop_family_expansion_excluded_related_proteins.csv`, and
`aop_manual_uniprot_mapping.csv` retain the earlier source/helper definitions
for provenance. They were not updated to the accepted member ranges and
identifier corrections. Do not re-expand, override or supplement the final
inline mappings from those historical files.

The `05_run_aop_ppi_and_summary.py` interface has now been adapted to the final
table, using `aop_matching.py`. It reads no historical inventory, manual helper or
family file. Matching uses raw probabilities strictly >0.7 on one explicit member
for all required labels; binding is checked only if labelled. All qualifying
members are recorded, while each source mapping remains one node. Reused
chemical/protein/task predictions carry shared evidence IDs. No aggregate scores
or rankings are emitted. Historical scoring/table helpers reject final inputs.

Seven rule/interface tests passed, including the exact 0.7 boundary, missing
required predictions, cross-member versus same-member combinations, both
directions, shared evidence, and single-node counting. Five archived prediction
files produced 455 checked nodes and confirmed all 25 illustrative S17 associations.
Fresh AOP environment outputs match the model environment byte-for-byte. These
checks do not certify full model or paper regeneration. The 91-row scientific
mapping file itself is unchanged. See `reproduction/REPRODUCE.txt` and the analysis
matrix for commands and remaining dependencies.
