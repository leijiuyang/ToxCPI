from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "test1" / "explanation" / "downstream_ao_4chem"
ROUTE1 = BASE / "route1_ctd_v2"
ROUTE2 = BASE / "route2_aop"
OUT = ROOT / "test1" / "supplement" / "tables"
ROUTE2_HIT_THRESHOLD = 0.7
LABEL_PROBABILITY = {
    "binding": "P_binding",
    "activation": "P_activation",
    "inhibition": "P_inhibition",
}

CHEMICALS = [
    ("BPA", "Bisphenol A", "80-05-7"),
    ("TCDD", "2,3,7,8-Tetrachlorodibenzo-p-dioxin", "1746-01-6"),
    ("BaP", "Benzo(a)pyrene", "50-32-8"),
    ("Chlorpyrifos", "Chlorpyrifos", "2921-88-2"),
    ("DDT", "DDT", "50-29-3"),
]

def load_inputs() -> dict[str, pd.DataFrame]:
    files = {
        "overlap": ROUTE1 / "predicted_vs_ctd_pathway_overlap_summary.csv",
        "ora": ROUTE1 / "route1b_kegg_pathway_ora_ranked_threshold_0.5.csv",
        "ctd": ROUTE1 / "route1d_ctd_pathway_disease_top5_threshold_0.5.csv",
        "detail": ROUTE2 / "route2_aop_hit_detail_4chem_threshold_0.7_multilabel.csv",
        "aop": ROUTE2 / "route2_chemical_aop_ao_4chem_threshold_0.7_multilabel.csv",
        "ao": ROUTE2 / "route2_chemical_ao_summary_4chem_threshold_0.7_multilabel.csv",
        "family": ROUTE2 / "route2_family_expanded_event_hits_threshold_0.7.csv",
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)
    return {name: pd.read_csv(path, low_memory=False) for name, path in files.items()}


def validate_names(data: dict[str, pd.DataFrame]) -> None:
    expected = {actual for _, actual, _ in CHEMICALS}
    for key in ("overlap", "ctd", "detail", "aop", "ao", "family"):
        observed = set(data[key]["chemical_name"].dropna())
        missing = expected - observed
        if missing:
            raise ValueError(f"{key} is missing representative chemicals: {sorted(missing)}")


def target_display(row: pd.Series) -> str:
    if row.get("Mapping_Status") == "family_expanded_mapped":
        return str(row.get("Family_Name") or row.get("Protein_Name") or "family")
    for column in ("gene_symbol", "uniprot_id", "Protein_Name"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return "Unresolved target"


def required_event_labels(event_labels: object) -> list[str]:
    labels = str(event_labels or "").split(";")
    return [label for label in LABEL_PROBABILITY if label in labels]


def label_display(event_labels: object) -> str:
    return " + ".join(label.capitalize() for label in required_event_labels(event_labels))


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def build_hit_events(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    detail = data["detail"].copy()
    if "unique_model_evidence_id" not in detail.columns:
        raise ValueError("Route 2 detail lacks unique_model_evidence_id")
    if detail["unique_model_evidence_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Route 2 detail contains blank unique_model_evidence_id values")
    detail["is_hit"] = bool_series(detail["is_hit"])
    family = data["family"].copy()
    family["Family_Event_Hit"] = bool_series(family["Family_Event_Hit"])
    event_keys = ["chemical_name", "AOP_ID", "Event_ID", "uniprot_id", "event_labels"]
    records: list[dict[str, object]] = []
    for group_key, group in detail[detail["is_hit"]].groupby(
        event_keys, dropna=False, sort=False
    ):
        representative = group.iloc[0]
        labels = required_event_labels(representative["event_labels"])
        if not labels:
            raise ValueError(f"Hit event has no modeled labels: {event_keys}={group_key}")
        probabilities: dict[str, float] = {}
        for label in labels:
            column = LABEL_PROBABILITY[label]
            values = pd.to_numeric(group[column], errors="coerce").dropna().unique()
            if len(values) != 1:
                raise ValueError(
                    f"Event {representative['AOP_ID']}/{representative['Event_ID']} has "
                    f"non-unique {column}: {values.tolist()}"
                )
            probabilities[label] = float(values[0])
        direct_joint_score = min(probabilities.values())
        if not all(value > ROUTE2_HIT_THRESHOLD for value in probabilities.values()):
            raise ValueError(
                f"Strict-AND violation for {representative['chemical_name']}, "
                f"AOP {representative['AOP_ID']}, event {representative['Event_ID']}: "
                f"{probabilities}"
            )

        is_family = representative.get("Mapping_Status") == "family_expanded_mapped"
        if is_family:
            matching_family = family[
                family["chemical_name"].eq(representative["chemical_name"])
                & family["AOP_ID"].eq(representative["AOP_ID"])
                & family["Event_ID"].eq(representative["Event_ID"])
                & family["Family_Name"].eq(representative["Family_Name"])
            ]
            if matching_family.empty:
                raise ValueError(
                    f"Missing family event source for {representative['chemical_name']}, "
                    f"AOP {representative['AOP_ID']}, event {representative['Event_ID']}"
                )
            if not matching_family["Family_Event_Hit"].all():
                raise ValueError("Family source disagrees with Route 2 detail is_hit")
            for column in (
                "Family_Representative_Gene",
                "Family_Representative_UniProt",
                "Family_Representative_Joint_Score",
            ):
                if matching_family[column].dropna().astype(str).nunique() != 1:
                    raise ValueError(f"Family representative is not unique in {column}")
            event_score = float(matching_family["Family_Representative_Joint_Score"].iloc[0])
            scoring_gene = str(matching_family["Family_Representative_Gene"].iloc[0]).strip()
            scoring_uniprot = str(
                matching_family["Family_Representative_UniProt"].iloc[0]
            ).strip()
            if abs(event_score - direct_joint_score) > 1e-12:
                raise ValueError(
                    f"Family representative joint score mismatch: {event_score} != "
                    f"min({probabilities})={direct_joint_score}"
                )
        else:
            event_score = direct_joint_score
            scoring_gene = str(representative.get("gene_symbol") or "").strip()
            scoring_uniprot = str(representative.get("uniprot_id") or "").strip()

        if not event_score > ROUTE2_HIT_THRESHOLD:
            raise ValueError(f"Event matching score is not strictly > 0.7: {event_score}")
        record = representative.to_dict()
        record.update(
            {
                "required_prediction_labels": label_display(representative["event_labels"]),
                "event_matching_score": event_score,
                "scoring_member_gene": scoring_gene,
                "scoring_member_uniprot": scoring_uniprot,
                "is_family": is_family,
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def representative_targets(hit_events: pd.DataFrame, chemical: str, limit: int = 6) -> str:
    subset = hit_events[hit_events["chemical_name"] == chemical].copy()
    subset["target_display"] = subset.apply(target_display, axis=1)
    subset = subset.sort_values("event_matching_score", ascending=False)
    targets = subset.drop_duplicates("target_display")["target_display"].head(limit).tolist()
    return ", ".join(targets)


def clean_ao_name(value: object) -> str:
    text = str(value).strip()
    return text[5:].strip() if text.casefold().startswith("n/a,") else text


def top_outcomes(ctd: pd.DataFrame, chemical: str, limit: int = 5) -> str:
    subset = ctd[ctd["chemical_name"] == chemical].sort_values(
        "display_rank_score", ascending=False
    )
    return "; ".join(subset.drop_duplicates("disease_name")["disease_name"].head(limit))


def build_s7(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    overlap = data["overlap"]
    overlap = overlap[overlap["threshold"].round(3) == 0.5]
    rows = []
    for short, actual, cas in CHEMICALS:
        metric = overlap[overlap["chemical_name"] == actual]
        if len(metric) != 1:
            raise ValueError(f"Expected one threshold-0.5 overlap row for {actual}, got {len(metric)}")
        metric = metric.iloc[0]
        aop_rows = data["aop"][data["aop"]["chemical_name"] == actual]
        ao_rows = data["ao"][data["ao"]["chemical_name"] == actual]
        rows.append(
            {
                "Chemical": short,
                "CAS": cas,
                "CTD-supported predicted pathways (%)": 100 * metric["our_supported_by_ctd_fraction"],
                "CTD-recorded pathways recovered (%)": 100 * metric["ctd_recovered_by_our_fraction"],
                "Jaccard overlap (%)": 100 * metric["jaccard"],
                "Top CTD-associated outcome terms": top_outcomes(data["ctd"], actual),
                "Hit AOPs": int(aop_rows["AOP_ID"].nunique()),
                "Hit AOs": int(ao_rows[["AO_category", "AO_name"]].drop_duplicates().shape[0]),
                "Representative direction-matched targets": representative_targets(data["hit_events"], actual),
            }
        )
    return pd.DataFrame(rows).round(
        {
            "CTD-supported predicted pathways (%)": 1,
            "CTD-recorded pathways recovered (%)": 1,
            "Jaccard overlap (%)": 1,
        }
    )


def build_s8(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ctd = data["ctd"].copy()
    ora_columns = [
        "chemical_name",
        "pathway_id",
        "fdr",
        "overlap_count",
        "fold_enrichment",
    ]
    ora = data["ora"][ora_columns].drop_duplicates(["chemical_name", "pathway_id"])

    ora_metric_columns = ora_columns[2:]
    ctd = ctd.drop(
        columns=[column for column in ora_metric_columns if column in ctd.columns]
    )
    ctd = ctd.merge(ora, on=["chemical_name", "pathway_id"], how="left", validate="many_to_one")
    short_by_actual = {actual: short for short, actual, _ in CHEMICALS}
    selected = []
    for _, actual, _ in CHEMICALS:
        subset = ctd[ctd["chemical_name"] == actual].sort_values(
            ["display_rank_score", "pathway_fdr"], ascending=[False, True]
        )
        selected.append(subset.head(5))
    out = pd.concat(selected, ignore_index=True)
    return pd.DataFrame(
        {
            "Chemical": out["chemical_name"].map(short_by_actual),
            "KEGG pathway ID": out["pathway_id"],
            "KEGG pathway name": out["pathway_name"],
            "FDR": out["fdr"],
            "Overlap count": out["overlap_count"].astype("Int64"),
            "Fold enrichment": out["fold_enrichment"],
            "CTD disease/outcome term": out["disease_name"],
            "CTD inference genes": out["n_inference_genes"].astype("Int64"),
            "CTD support (log1p genes)": out["CTD_link_support"],
            "Display rank score": out["display_rank_score"],
        }
    )


def build_s9(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    hit_events = data["hit_events"]
    aop = data["aop"]
    rows = []
    for short, actual, _ in CHEMICALS:
        top_aops = (
            aop[aop["chemical_name"] == actual]
            .sort_values(["AOP_hit_score", "hit_coverage"], ascending=[False, False])
            .drop_duplicates("AOP_ID")
            .head(5)
        )
        for _, summary in top_aops.iterrows():
            candidates = hit_events[
                (hit_events["chemical_name"] == actual)
                & (hit_events["AOP_ID"] == summary["AOP_ID"])
            ].copy()
            if candidates.empty:
                raise ValueError(f"No hit detail for {actual}, AOP {summary['AOP_ID']}")
            representative = candidates.sort_values(
                ["event_matching_score", "Position_Weight", "Event_ID"],
                ascending=[False, False, True],
            ).iloc[0]
            rows.append(
                {
                    "Chemical": short,
                    "AOP ID": int(summary["AOP_ID"]),
                    "AOP title": summary["AOP_Title"],
                    "Event role": representative["Event_Type"],
                    "Event title": representative["Event_Title"],
                    "Required prediction label(s)": representative["required_prediction_labels"],
                    "Hit target(s)": target_display(representative),
                    "Scoring member gene": representative["scoring_member_gene"],
                    "Scoring member UniProt": representative["scoring_member_uniprot"],
                    "Event matching score": representative["event_matching_score"],
                    "AO name": clean_ao_name(summary["AO_name"]),
                    "AOP hit score": summary["AOP_hit_score"],
                    "Hit type": "Family-expanded" if representative["is_family"] else "Direct",
                }
            )
    return pd.DataFrame(rows)


def style_workbook(writer: pd.ExcelWriter) -> None:
    for sheet in writer.book.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        for idx, column in enumerate(sheet.columns, 1):
            width = min(max(len(str(c.value or "")) for c in column) + 2, 60)
            sheet.column_dimensions[get_column_letter(idx)].width = max(width, 12)


def write(path: Path, table: pd.DataFrame, title: str, notes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name="Table")
        pd.DataFrame({"Notes": [title, *notes]}).to_excel(
            writer, index=False, sheet_name="Notes"
        )
        style_workbook(writer)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route1-dir", required=True, type=Path)
    parser.add_argument("--route2-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chemical-list", required=True, type=Path,
                        help="CSV with short_name, chemical_name, and casrn columns")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global BASE, ROUTE1, ROUTE2, OUT, CHEMICALS
    ROUTE1 = args.route1_dir.expanduser().resolve()
    ROUTE2 = args.route2_dir.expanduser().resolve()
    BASE = ROUTE1.parent
    OUT = args.output_dir.expanduser().resolve()
    chemicals = pd.read_csv(args.chemical_list.expanduser().resolve())
    required = {"short_name", "chemical_name", "casrn"}
    missing = sorted(required - set(chemicals.columns))
    if missing or chemicals.empty:
        raise ValueError(f"Chemical list is empty or missing columns: {missing}")
    CHEMICALS = list(chemicals[["short_name", "chemical_name", "casrn"]].itertuples(index=False, name=None))


def main() -> None:
    args = parse_args()
    configure(args)
    data = load_inputs()
    validate_names(data)
    data["hit_events"] = build_hit_events(data)
    s7 = build_s7(data)
    s8 = build_s8(data)
    s9 = build_s9(data)
    common_boundary = (
        "CTD and AOP associations are database-supported computational links and mechanistic "
        "prioritization evidence, not experimental proof of chemical causality."
    )
    write(
        OUT / "Table_S7_ctd_aop_summary_5chem.xlsx",
        s7,
        "Table S7. Summary of CTD pathway agreement and AOP/AO associations for five representative chemicals.",
        [
            "CTD agreement uses Route 1 threshold 0.5; AOP/AO counts use Route 2 threshold 0.7 multilabel outputs.",
            "Representative targets are the six highest-probability distinct threshold-passing direct targets or curated family concepts.",
            common_boundary,
        ],
    )
    write(
        OUT / "Table_S8_ctd_pathway_disease_5chem.xlsx",
        s8,
        "Table S8. Top CTD pathway-disease associations linked to ToxCPI-enriched KEGG pathways.",
        [
            "Five rows per chemical were selected by current display_rank_score from route1_ctd_v2 threshold-0.5 output.",
            "display_rank_score = fold_enrichment x log1p(n_inference_genes); it is not a disease-risk probability.",
            common_boundary,
        ],
    )
    write(
        OUT / "Table_S9_aop_ao_direction_matched_5chem.xlsx",
        s9,
        "Table S9. Direction-matched AOP/AO associations for five representative chemicals.",
        [
            "Five highest-scoring distinct AOPs per chemical are shown; within each AOP, the event with the highest event matching score is the representative event.",
            "An event was considered a hit only when every required label-specific probability was strictly greater than 0.7. For family-level concepts, all required thresholds had to be met by the same curated member. The event matching score is the minimum required-label probability for the representative target or family member; for a single-label event, it equals the corresponding task probability.",
            "AOP hit scores and event matching scores are event-support scores, not adverse-outcome probabilities.",
            "Multiple mapped nodes can share one chemical-protein-task unique_model_evidence_id; such mappings are not independent model evidence.",
            common_boundary,
        ],
    )
    print(s7.to_string(index=False))
    print(f"Built S7={len(s7)}, S8={len(s8)}, S9={len(s9)} rows in {OUT}")


if __name__ == "__main__":
    main()
