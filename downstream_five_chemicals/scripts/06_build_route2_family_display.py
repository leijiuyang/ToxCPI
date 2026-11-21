from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORK = REPOSITORY_ROOT / "downstream_five_chemicals"
INPUT = REPOSITORY_ROOT / "data" / "AOP"
PRED = WORK / "predictions"
ROUTE2 = REPOSITORY_ROOT / "runtime" / "downstream_five_chemicals" / "route2_aop"
AOP_DIR = REPOSITORY_ROOT / "data" / "AOP"

AOP_PROTEINS_MULTILABEL = AOP_DIR / "human_aop_proteins_multilabel.csv"
FAMILY_MAP = AOP_DIR / "aop_family_expansion_map.csv"
FAMILY_EXCLUDED_RELATED = AOP_DIR / "aop_family_expansion_excluded_related_proteins.csv"
OUT = ROUTE2 / "route2_family_expanded_event_hits_threshold_0.7.csv"
THRESHOLD = 0.7

CHEMICAL_SLUGS = {
    "Bisphenol A": "bisphenol_a",
    "Perfluorooctanoic acid": "perfluorooctanoic_acid",
    "Rotenone": "rotenone",
    "2,3,7,8-Tetrachlorodibenzo-p-dioxin": "2378-tetrachlorodibenzo-p-dioxin",
    "1-Hydroxypyrene": "1-hydroxypyrene",
    "Benzanthrone": "benzanthrone",
    "Benzo(a)pyrene": "benzo(a)pyrene",
    "Chlorpyrifos": "chlorpyrifos",
    "DDT": "ddt",
    "p,p'-DDE": "pp'-dde",
    "Di(2-ethylhexyl)phthalate": "di(2-ethylhexyl)phthalate",
    "Atrazine": "atrazine",
    "Vinclozolin": "vinclozolin",
    "Dieldrin": "dieldrin",
    "Tributyltin": "tributyltin",
    "Diethylstilbestrol": "diethylstilbestrol",
}

TASK_PROBABILITY = {
    "binding": "P_binding",
    "activation": "P_activation",
    "inhibition": "P_inhibition",
}

TASK_RANK = {
    "binding": "rank_binding",
    "activation": "rank_activation",
    "inhibition": "rank_inhibition",
}

EVENT_WARNING = (
    "Family-level event requires one member to exceed the threshold for every event label; "
    "the representative member maximizes the minimum required-label probability. Not an AO probability."
)


def read_prediction(slug: str) -> pd.DataFrame:
    cols = [
        "chemical_name",
        "casrn",
        "ctd_chemical_id",
        "uniprot_id",
        "gene_symbol",
        "protein_name",
        "P_binding",
        "P_activation",
        "P_inhibition",
        "rank_binding",
        "rank_activation",
        "rank_inhibition",
    ]
    return pd.read_csv(PRED / f"all_human_proteome_scores_{slug}.csv", usecols=cols)


def required_event_labels(event_labels: object) -> list[str]:
    labels = str(event_labels or "").split(";")
    return [label for label in labels if label in TASK_PROBABILITY]


def score_family_members(
    member_scores: pd.DataFrame,
    event_labels: object,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    scored = member_scores.copy()
    labels = required_event_labels(event_labels)
    required_cols = [TASK_PROBABILITY[label] for label in labels]
    if not required_cols:
        scored["member_joint_score"] = np.nan
        scored["member_joint_hit"] = False
        return scored
    for col in required_cols:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")
    scored["member_joint_score"] = scored[required_cols].min(axis=1, skipna=False)
    scored["member_joint_hit"] = scored[required_cols].gt(threshold).all(axis=1)
    return scored


def expansion_context(task: str, binding_context: object) -> str:
    if task == "binding" or str(binding_context) == "generic_family_binding":
        return "generic_family_binding_expanded"
    return "generic_family_effect_expanded"


def build_rows() -> pd.DataFrame:
    aop = pd.read_csv(AOP_PROTEINS_MULTILABEL, encoding="utf-8-sig")
    if "Use_For_AOP_Matching" in aop.columns:
        raise ValueError("Historical family display cannot consume the final inline-member table. Run 05_run_aop_ppi_and_summary.py; qualifying members are already recorded there.")
    family = pd.read_csv(FAMILY_MAP)
    if FAMILY_EXCLUDED_RELATED.exists():
        excluded_related_df = pd.read_csv(FAMILY_EXCLUDED_RELATED)
    else:
        excluded_related_df = pd.DataFrame()

    event_cols = [
        "AOP_ID",
        "Event_ID",
        "Event_Title",
        "Protein_PR_ID",
        "Protein_Name",
        "Binding_Context",
        "Event_Labels",
        "AO_Title",
        "Mapping_Status",
    ]
    events = aop[aop["Mapping_Status"].eq("family_expanded_mapped")][event_cols].drop_duplicates(
        ["AOP_ID", "Event_ID", "Protein_PR_ID", "Protein_Name", "Event_Labels"],
        keep="first",
    )
    events = events.merge(
        family,
        on=["Protein_PR_ID", "Protein_Name"],
        how="inner",
    )

    rows: list[dict[str, object]] = []
    group_cols = event_cols + [
        "Family_Name",
        "Expansion_Source",
        "Included_Member_Rule",
        "Family_Universe_Audit_Source",
        "Use_In_AOP_Score",
    ]
    for group_key, group in events.groupby(group_cols, dropna=False):
        event = dict(zip(group_cols, group_key))
        tasks = required_event_labels(event["Event_Labels"])
        expanded = group[["Expansion_Member_Gene", "Expansion_Member_UniProt"]].drop_duplicates()
        expanded_uniprots = expanded["Expansion_Member_UniProt"].astype(str).tolist()
        expanded_members = ";".join(
            f"{r.Expansion_Member_Gene}:{r.Expansion_Member_UniProt}"
            for r in expanded.itertuples(index=False)
        )
        if excluded_related_df.empty:
            excluded_related = ""
        else:
            ex = excluded_related_df[
                excluded_related_df["Family_Name"].astype(str).eq(str(event["Family_Name"]))
                & excluded_related_df["Protein_PR_ID"].astype(str).eq(str(event["Protein_PR_ID"]))
                & excluded_related_df["Protein_Name"].astype(str).eq(str(event["Protein_Name"]))
            ]
            excluded_related = ";".join(
                f"{r.Related_Gene}:{r.Related_UniProt}({r.Exclusion_Reason})"
                for r in ex.sort_values("Related_Gene").itertuples(index=False)
            )

        for requested_chemical, slug in CHEMICAL_SLUGS.items():
            pred = read_prediction(slug)
            member_scores = expanded.merge(
                pred,
                left_on="Expansion_Member_UniProt",
                right_on="uniprot_id",
                how="left",
            )
            tested = member_scores[member_scores["uniprot_id"].notna()].copy()
            missing = sorted(set(expanded_uniprots) - set(tested["uniprot_id"].astype(str)))
            representative = None
            joint_hits = pd.DataFrame()
            if tested.empty:
                chemical_name = requested_chemical
                casrn = ""
                ctd_chemical_id = ""
            else:
                tested = score_family_members(tested, event["Event_Labels"])
                for task in tasks:
                    tested[TASK_RANK[task]] = pd.to_numeric(tested[TASK_RANK[task]], errors="coerce")
                valid_joint_scores = tested["member_joint_score"].dropna()
                if not valid_joint_scores.empty:
                    representative = tested.loc[valid_joint_scores.idxmax()]
                joint_hits = tested[tested["member_joint_hit"]].copy()
                chemical_name = (
                    str(tested["chemical_name"].dropna().iloc[0])
                    if tested["chemical_name"].notna().any()
                    else requested_chemical
                )
                casrn = str(tested["casrn"].dropna().iloc[0]) if tested["casrn"].notna().any() else ""
                ctd_chemical_id = (
                    str(tested["ctd_chemical_id"].dropna().iloc[0])
                    if tested["ctd_chemical_id"].notna().any()
                    else ""
                )

            family_event_hit = bool(not joint_hits.empty)
            hit_count = int(joint_hits.shape[0])
            hit_proteins = ";".join(
                f"{r.Expansion_Member_Gene}:{r.member_joint_score:.3f}"
                for r in joint_hits.sort_values("member_joint_score", ascending=False).itertuples(index=False)
            )
            hit_uniprots = ";".join(joint_hits.get("uniprot_id", pd.Series(dtype=str)).dropna().astype(str))
            hit_probabilities = ";".join(
                f"{value:.6g}" for value in joint_hits.get("member_joint_score", pd.Series(dtype=float)).dropna()
            )
            representative_gene = "" if representative is None else str(representative["Expansion_Member_Gene"])
            representative_uniprot = "" if representative is None else str(representative["uniprot_id"])
            representative_joint_score = (
                np.nan if representative is None else float(representative["member_joint_score"])
            )

            for task in tasks:
                rank_col = TASK_RANK[task]
                max_rank = np.nan
                if representative is not None:
                    rank_value = pd.to_numeric(representative.get(rank_col), errors="coerce")
                    max_rank = float(rank_value) if pd.notna(rank_value) else np.nan
                rows.append(
                    {
                        "chemical_name": chemical_name,
                        "casrn": casrn,
                        "ctd_chemical_id": ctd_chemical_id,
                        "AOP_ID": event["AOP_ID"],
                        "Event_ID": event["Event_ID"],
                        "Event_Title": event["Event_Title"],
                        "Event_Labels": event["Event_Labels"],
                        "AO_Title": event["AO_Title"],
                        "Protein_PR_ID": event["Protein_PR_ID"],
                        "Protein_Name": event["Protein_Name"],
                        "Binding_Context": event["Binding_Context"],
                        "Mapping_Status": "family_expanded_mapped",
                        "Expansion_Context": expansion_context(task, event["Binding_Context"]),
                        "Family_Name": event["Family_Name"],
                        "Expansion_Source": event["Expansion_Source"],
                        "Expanded_Members": expanded_members,
                        "Included_Member_Rule": event.get("Included_Member_Rule", ""),
                        "Family_Universe_Audit_Source": event.get("Family_Universe_Audit_Source", ""),
                        "Excluded_Related_Proteins_And_Reason": excluded_related,
                        "Family_Task": task,
                        "Family_Event_Hit": family_event_hit,
                        "Family_Hit_Proteins": hit_proteins,
                        "Family_Hit_UniProts": hit_uniprots,
                        "Family_Hit_Probabilities": hit_probabilities,
                        "Family_Max_Probability": representative_joint_score,
                        "Family_Max_Rank": max_rank,
                        "Family_Hit_Count": hit_count,
                        "Family_Tested_Member_Count": int(tested.shape[0]),
                        "Family_Total_Member_Count": int(expanded.shape[0]),
                        "Family_Missing_Members": ";".join(missing),
                        "Family_Required_Labels": ";".join(tasks),
                        "Family_Representative_Gene": representative_gene,
                        "Family_Representative_UniProt": representative_uniprot,
                        "Family_Representative_Joint_Score": representative_joint_score,
                        "Event_Threshold": THRESHOLD,
                        "Use_In_AOP_Score": True,
                        "Evidence_Type": "family-expanded event hit",
                        "Interpretation_Warning": EVENT_WARNING,
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build family-expanded AOP event display rows")
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--aop-data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--chemical-list", required=True, type=Path,
                        help="CSV with chemical_name and optional slug columns")
    return parser.parse_args()


def prediction_slug(name: str) -> str:
    return name.lower().replace(",", "").replace(" ", "_").replace("/", "_")


def configure(args: argparse.Namespace) -> None:
    global PRED, ROUTE2, AOP_DIR, AOP_PROTEINS_MULTILABEL
    global FAMILY_MAP, FAMILY_EXCLUDED_RELATED, OUT, THRESHOLD, CHEMICAL_SLUGS
    PRED = args.prediction_dir.expanduser().resolve()
    AOP_DIR = args.aop_data_dir.expanduser().resolve()
    OUT = args.output.expanduser().resolve()
    ROUTE2 = OUT.parent
    AOP_PROTEINS_MULTILABEL = AOP_DIR / "human_aop_proteins_multilabel.csv"
    FAMILY_MAP = AOP_DIR / "aop_family_expansion_map.csv"
    FAMILY_EXCLUDED_RELATED = AOP_DIR / "aop_family_expansion_excluded_related_proteins.csv"
    THRESHOLD = args.threshold
    chemicals = pd.read_csv(args.chemical_list.expanduser().resolve())
    if "chemical_name" not in chemicals.columns or chemicals.empty:
        raise ValueError("--chemical-list must contain a non-empty chemical_name column")
    CHEMICAL_SLUGS = {
        str(row.chemical_name): (
            str(row.slug) if "slug" in chemicals.columns and pd.notna(row.slug)
            else prediction_slug(str(row.chemical_name))
        )
        for row in chemicals.itertuples(index=False)
    }


def main() -> int:
    args = parse_args()
    configure(args)
    ROUTE2.mkdir(parents=True, exist_ok=True)
    out = build_rows()
    out = out.sort_values(
        ["chemical_name", "Family_Event_Hit", "Family_Max_Probability", "AOP_ID", "Event_ID"],
        ascending=[True, False, False, True, True],
    )
    out.to_csv(OUT, index=False)
    print(
        f"wrote {OUT} rows={len(out)} "
        f"event_hits={int(out['Family_Event_Hit'].sum())} "
        f"events={out[['AOP_ID', 'Event_ID', 'Family_Name', 'Family_Task']].drop_duplicates().shape[0]}"
    )
    if out["Family_Missing_Members"].astype(str).str.len().gt(0).any():
        missing = out.loc[out["Family_Missing_Members"].astype(str).str.len().gt(0), "Family_Missing_Members"].unique()
        print(f"missing expanded members: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
