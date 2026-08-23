from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORK = REPOSITORY_ROOT / "downstream_five_chemicals"
INPUT = REPOSITORY_ROOT / "data" / "AOP"
PRED = WORK / "predictions"
ROUTE1_V2 = WORK / "route1_ctd"
RUNTIME = REPOSITORY_ROOT / "runtime" / "downstream_five_chemicals"
ROUTE2 = RUNTIME / "route2_aop"
PPI = RUNTIME / "ppi"
SUMMARY = RUNTIME / "summary"
LOG = REPOSITORY_ROOT / "runtime" / "logs" / "05_run_aop_ppi_and_summary.log"

AOP_DIR = REPOSITORY_ROOT / "data" / "AOP"
AOP_PROTEINS_MULTILABEL = AOP_DIR / "human_aop_proteins_multilabel.csv"
AOP_PROTEINS = AOP_DIR / "human_aop_proteins.csv"
AOP_COMPLETE_MULTILABEL = AOP_DIR / "human_aop_complete_multilabel.csv"
AOP_COMPLETE = AOP_DIR / "human_aop_complete.csv"
MANUAL_MAP = INPUT / "aop_manual_uniprot_mapping.csv"
AUTHORITATIVE_MANUAL_MAP = AOP_DIR / "aop_manual_uniprot_mapping.csv"
FAMILY_MAP = AOP_DIR / "aop_family_expansion_map.csv"
FAMILY_EXCLUDED_RELATED = AOP_DIR / "aop_family_expansion_excluded_related_proteins.csv"
FAMILY_EVENT_OUT = ROUTE2 / "route2_family_expanded_event_hits_threshold_0.7.csv"

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

MANUAL_ROWS = [
    {"gene_symbol": "AHR", "uniprot_id": "P35869", "protein_name": "aryl hydrocarbon receptor"},
    {"gene_symbol": "ARNT", "uniprot_id": "P27540", "protein_name": "aryl hydrocarbon receptor nuclear translocator"},
    {"gene_symbol": "AR", "uniprot_id": "P10275", "protein_name": "androgen receptor"},
    {"gene_symbol": "HIF1A", "uniprot_id": "Q16665", "protein_name": "hypoxia-inducible factor 1-alpha"},
]

ROUTE2_HIT_THRESHOLD = 0.7
ROUTE2_OUTPUT_SUFFIX = "_multilabel"
ROUTE2_WARNING = (
    "Label-matched AOP MIE/KE hit score; every label on a multi-label event must exceed 0.7 "
    "on the same protein (strict AND). Binding-supported events use P_binding, whereas "
    "activation/inhibition events are direction-matched with P_activation/P_inhibition. "
    "Scores are not AO probabilities. "
    "Mapped AOP nodes sharing one unique_model_evidence_id are not independent model evidence. "
    "Endocrine receptor hits may be saturated and should not be used for chemical-specific ranking without recalibration."
)
FAMILY_EVENT_WARNING = (
    "Family-level event requires one member to exceed the threshold for every event label; "
    "the representative member maximizes the minimum required-label probability. Not an AO probability."
    " Mapped nodes sharing one unique_model_evidence_id are not independent model evidence."
)
LABEL_ORDER = ["binding", "activation", "inhibition", "unspecified"]
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
EVENT_HIT_GROUP_COLUMNS = [
    "chemical_name",
    "AOP_ID",
    "Event_ID",
    "uniprot_id",
    "event_labels",
]

BINDING_PATTERNS = [
    r"\bbinding\b",
    r"\bbind\b",
    r"\bbinds\b",
    r"\bbound\b",
    r"\bligand\s+binding\b",
    r"\breceptor\s+binding\b",
]
ACTIVATION_PATTERNS = [
    r"\bactivation\b",
    r"\bactivate\b",
    r"\bactivates\b",
    r"\bagonist\b",
    r"\binduction\b",
    r"\binduced\b",
    r"\bstabilization\b",
    r"\bupregulation\b",
    r"\bup-regulation\b",
]
WEAK_ACTIVATION_PATTERNS = [
    r"\bincreased\b",
    r"\bincrease\b",
    r"\boccurrence\b",
]
INHIBITION_PATTERNS = [
    r"\binhibition\b",
    r"\binhibit\b",
    r"\binhibits\b",
    r"\binhibitor\b",
    r"\bantagonist\b",
    r"\bdecreased\b",
    r"\bdecrease\b",
    r"\breduction\b",
    r"\breduced\b",
    r"\bimpaired\b",
    r"\bdisrupted\b",
    r"\bsuppression\b",
    r"\bsuppressed\b",
    r"\bloss\b",
    r"\bnot\s+increased\b",
    r"\bfunctional\s+change\b",
]
NEGATED_ACTIVATION_PATTERNS = [
    r"\b(?:decrease(?:d)?|reduction|reduced|inhibition|inhibit(?:s|ed)?|impaired|disrupted|suppression|suppressed|loss|not\s+increased)\b.*\b(?:activation|transactivation|proliferation)\b",
    r"\b(?:activation|transactivation|proliferation)\b.*\b(?:decrease(?:d)?|reduction|reduced|inhibition|inhibit(?:s|ed)?|impaired|disrupted|suppression|suppressed|loss|not\s+increased)\b",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def ensure_manual_map() -> pd.DataFrame:
    if AUTHORITATIVE_MANUAL_MAP.exists():
        return pd.read_csv(AUTHORITATIVE_MANUAL_MAP)
    raise FileNotFoundError(
        f"Manual AOP mapping not found: {AUTHORITATIVE_MANUAL_MAP}. "
        "Pass the correct directory with --aop-data-dir."
    )


def _hit_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def map_event_to_labels(event_title: str, action: str) -> list[str]:
    event_text = (event_title or "").lower()
    action_text = (action or "").lower()
    text = f"{event_text} {action_text}"
    labels = []
    has_binding = _hit_any(BINDING_PATTERNS, text)
    has_inhibition = _hit_any(INHIBITION_PATTERNS, text)
    if has_binding:
        labels.append("binding")
    activation_text = re.sub(r"\bnot\s+increased\b", " ", text)
    event_activation_text = re.sub(r"\bnot\s+increased\b", " ", event_text)
    action_activation_text = re.sub(r"\bnot\s+increased\b", " ", action_text)
    negated_activation = _hit_any(NEGATED_ACTIVATION_PATTERNS, text)
    has_activation = (
        (_hit_any(ACTIVATION_PATTERNS, activation_text) and not negated_activation)
        or (_hit_any(WEAK_ACTIVATION_PATTERNS, event_activation_text) and not has_inhibition)
        or (
            _hit_any(WEAK_ACTIVATION_PATTERNS, action_activation_text)
            and not has_binding
            and not has_inhibition
        )
    )
    if has_activation:
        labels.append("activation")
    if has_inhibition:
        labels.append("inhibition")
    if not labels:
        labels.append("unspecified")
    return [label for label in LABEL_ORDER if label in labels]


def labels_to_string(labels: list[str]) -> str:
    return ";".join(label for label in LABEL_ORDER if label in set(labels))


def required_event_labels(event_labels: object) -> list[str]:
    labels = str(event_labels or "").split(";")
    return [label for label in LABEL_ORDER if label in labels and label in TASK_PROBABILITY]


def event_joint_hit(
    probabilities: pd.Series | dict[str, object],
    event_labels: object,
    threshold: float = ROUTE2_HIT_THRESHOLD,
) -> bool:
    labels = required_event_labels(event_labels)
    if not labels:
        return False
    return all(
        pd.notna(pd.to_numeric(probabilities.get(TASK_PROBABILITY[label]), errors="coerce"))
        and float(pd.to_numeric(probabilities.get(TASK_PROBABILITY[label]), errors="coerce")) > threshold
        for label in labels
    )


def score_family_members(
    member_scores: pd.DataFrame,
    event_labels: object,
    threshold: float = ROUTE2_HIT_THRESHOLD,
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


def apply_event_joint_hits(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["_event_joint_hit"] = out.apply(
        lambda row: event_joint_hit(row, row["event_labels"]), axis=1
    )
    out["is_hit"] = (
        out.groupby(EVENT_HIT_GROUP_COLUMNS, dropna=False)["_event_joint_hit"]
        .transform("all")
        .astype(bool)
    )
    return out.drop(columns="_event_joint_hit")


def primary_effect_from_labels(labels: list[str]) -> str:
    for label in ["activation", "inhibition", "binding", "unspecified"]:
        if label in labels:
            return label
    return "unspecified"


def classify_binding_context(event_title: str, protein_name: str, pr_id: str) -> tuple[str, bool]:
    text = f"{event_title or ''} {protein_name or ''} {pr_id or ''}".lower()
    if not _hit_any(BINDING_PATTERNS, text):
        return "", False
    if "ace2" in text or "angiotensin-converting enzyme 2" in text:
        return "viral_receptor_binding", False
    if "t.f. to dna" in text or "tf/dna" in text or "transcription factor" in text or " to dna" in text:
        return "protein_dna_tf_binding", False
    if "tubulin" in text:
        return "generic_family_binding", False
    if "picrotoxin site" in text or "gabaa" in text or "igabar" in text or "chloride channel" in text:
        return "receptor_complex_binding", True
    if "inhibitor" in text or "antagonist" in text or "agonist" in text:
        return "chemical_target_binding", True
    return "chemical_target_binding", True


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def infer_event_direction(row: pd.Series) -> str:
    text = " ".join(
        str(row.get(col, ""))
        for col in ["Event_Title", "Action", "Expected_Effect", "AOP_Title", "AO_Title"]
    ).lower()

    inhibition_patterns = [
        r"\bantagonist\b",
        r"\binhibitor\b",
        r"\binhibition\b",
        r"\binhibit",
        r"\bdecreased?\b",
        r"\breduction\b",
        r"\breduced\b",
        r"\bsuppression\b",
        r"\bsuppressed\b",
        r"\bnot increased\b",
        r"\bimpair",
        r"\bloss\b",
    ]
    activation_patterns = [
        r"\bagonist\b",
        r"\bactivation\b",
        r"\bactivate",
        r"\bincreased?\b",
        r"\binduction\b",
        r"\binduced\b",
        r"\bstabilization\b",
        r"\bupregulat",
    ]
    for pattern in inhibition_patterns:
        if re.search(pattern, text):
            return "inhibition"
    for pattern in activation_patterns:
        if re.search(pattern, text):
            return "activation"
    expected = str(row.get("Expected_Effect", "")).strip().lower()
    if expected in {"activation", "inhibition"}:
        return expected
    return "unknown"


def build_aop_map() -> pd.DataFrame:
    ROUTE2.mkdir(parents=True, exist_ok=True)
    aop_path = AOP_PROTEINS_MULTILABEL
    complete_path = AOP_COMPLETE_MULTILABEL
    for required in (aop_path, complete_path, AUTHORITATIVE_MANUAL_MAP, FAMILY_MAP):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required AOP static input not found: {required}. "
                "Pass the packaged directory with --aop-data-dir."
            )
    aop = pd.read_csv(aop_path)
    if "AOP_Title" not in aop.columns:
        complete = pd.read_csv(complete_path, usecols=["AOP_ID", "AOP_Title"])
        aop = aop.merge(complete.drop_duplicates("AOP_ID"), on="AOP_ID", how="left")
    manual = ensure_manual_map()
    aop["Protein_Name_norm"] = aop["Protein_Name"].fillna("").astype(str).str.strip().str.lower()
    manual_lookup = dict(zip(manual["protein_name"].astype(str).str.strip().str.lower(), manual["uniprot_id"].astype(str)))
    missing_before = int(aop["UniProt_ID"].isna().sum())
    if "Resolved_UniProt_ID" in aop.columns:
        resolved = aop["Resolved_UniProt_ID"].fillna("").astype(str).str.strip()
        mask = aop["UniProt_ID"].isna() & resolved.ne("")
        aop.loc[mask, "UniProt_ID"] = resolved[mask]
    for protein_name_norm, uniprot_id in manual_lookup.items():
        mask = aop["UniProt_ID"].isna() & aop["Protein_Name_norm"].eq(protein_name_norm)
        aop.loc[mask, "UniProt_ID"] = uniprot_id
    missing_after = int(aop["UniProt_ID"].isna().sum())
    aop = aop.drop(columns=["Protein_Name_norm"])

    if "Event_Labels" not in aop.columns:
        label_lists = aop.apply(lambda row: map_event_to_labels(row.get("Event_Title", ""), row.get("Action", "")), axis=1)
        aop["Event_Labels"] = label_lists.apply(labels_to_string)
        aop["Has_Binding"] = label_lists.apply(lambda labels: "binding" in labels)
        aop["Has_Activation"] = label_lists.apply(lambda labels: "activation" in labels)
        aop["Has_Inhibition"] = label_lists.apply(lambda labels: "inhibition" in labels)
        aop["Has_Unspecified"] = label_lists.apply(lambda labels: "unspecified" in labels)
        aop["Primary_Effect"] = label_lists.apply(primary_effect_from_labels)
    else:
        for label, col in [
            ("binding", "Has_Binding"),
            ("activation", "Has_Activation"),
            ("inhibition", "Has_Inhibition"),
            ("unspecified", "Has_Unspecified"),
        ]:
            if col not in aop.columns:
                aop[col] = aop["Event_Labels"].fillna("").astype(str).str.split(";").apply(lambda labels: label in labels)
        if "Primary_Effect" not in aop.columns:
            aop["Primary_Effect"] = aop["Event_Labels"].fillna("").astype(str).str.split(";").apply(primary_effect_from_labels)
    if "Binding_Context" not in aop.columns or "Use_For_ToxCPI_Binding" not in aop.columns:
        binding_info = aop.apply(
            lambda row: classify_binding_context(
                row.get("Event_Title", ""),
                row.get("Protein_Name", ""),
                row.get("Protein_PR_ID", ""),
            ),
            axis=1,
        )
        aop["Binding_Context"] = binding_info.apply(lambda item: item[0])
        aop["Use_For_ToxCPI_Binding"] = binding_info.apply(lambda item: item[1])

    aop["event_direction"] = aop["Primary_Effect"]
    aop["AOP_node_type"] = aop["Event_Type"]
    aop["AO_name"] = aop["AO_Title"]
    aop["AO_category"] = aop["AO_Title"]
    aop["uniprot_id"] = aop["UniProt_ID"].astype(str)
    out = ROUTE2 / f"aop_protein_map_extended{ROUTE2_OUTPUT_SUFFIX}.csv"
    aop.to_csv(out, index=False)
    log(
        f"read {aop_path} wrote {out} rows={len(aop)} missing_before={missing_before} missing_after={missing_after} "
        f"label_counts={aop['Event_Labels'].value_counts().to_dict()}"
    )
    return aop


def load_prediction(slug: str) -> pd.DataFrame:
    path = PRED / f"all_human_proteome_scores_{slug}.csv"
    df = pd.read_csv(path)
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
        "S_target",
        "rank_binding",
        "rank_activation",
        "rank_inhibition",
        "effect_extrapolated",
    ]
    return df[cols].copy()


def expand_aop_labels(aop: pd.DataFrame) -> pd.DataFrame:
    rows = []
    aop2 = aop[aop["UniProt_ID"].notna()].copy()
    dedup_key = ["AOP_ID", "Event_ID", "Protein_PR_ID", "Protein_Name", "Event_Labels"]
    before = len(aop2)
    aop2 = aop2.drop_duplicates(dedup_key, keep="first")
    removed = before - len(aop2)
    if removed:
        log(f"removed duplicate direct AOP protein-event rows before label expansion: {removed}")
    for row in aop2.to_dict("records"):
        labels = [
            label for label in str(row.get("Event_Labels", "")).split(";")
            if label in {"binding", "activation", "inhibition"}
        ]
        for label in labels:
            if label == "binding" and not as_bool(row.get("Use_For_ToxCPI_Binding", False)):
                continue
            out = row.copy()
            out["matched_label"] = label
            out["matched_task"] = label
            out["event_labels"] = row.get("Event_Labels", "")
            rows.append(out)
    return pd.DataFrame(rows)


def build_directional_detail(aop: pd.DataFrame) -> pd.DataFrame:
    aop2 = expand_aop_labels(aop)
    detail_frames = []
    for chemical, slug in CHEMICAL_SLUGS.items():
        pred = load_prediction(slug)
        merged = aop2.merge(pred, on="uniprot_id", how="left", suffixes=("_aop", ""))
        merged["chemical_name"] = merged["chemical_name"].fillna(chemical)
        merged["matched_probability"] = np.select(
            [
                merged["matched_label"].eq("binding"),
                merged["matched_label"].eq("activation"),
                merged["matched_label"].eq("inhibition"),
            ],
            [merged["P_binding"], merged["P_activation"], merged["P_inhibition"]],
            default=np.nan,
        )
        merged["matched_rank"] = np.select(
            [
                merged["matched_label"].eq("binding"),
                merged["matched_label"].eq("activation"),
                merged["matched_label"].eq("inhibition"),
            ],
            [merged["rank_binding"], merged["rank_activation"], merged["rank_inhibition"]],
            default=np.nan,
        )
        merged["is_label_hit"] = merged["matched_probability"].gt(ROUTE2_HIT_THRESHOLD)

        merged = apply_event_joint_hits(merged)
        merged["P_dir"] = merged["matched_probability"]
        merged["rank_dir"] = merged["matched_rank"]
        merged["event_direction"] = np.where(
            merged["matched_label"].isin(["activation", "inhibition"]),
            merged["matched_label"],
            "binding-supported",
        )
        merged["direction_match_threshold"] = ROUTE2_HIT_THRESHOLD
        merged["interpretation_warning"] = ROUTE2_WARNING
        merged["mapped_node_id"] = merged.apply(
            lambda row: f"AOP:{int(row['AOP_ID'])}|Event:{int(row['Event_ID'])}", axis=1
        )
        merged["unique_model_evidence_id"] = merged.apply(
            lambda row: "|".join([
                "MODEL",
                str(row.get("chemical_name", chemical)),
                str(row.get("uniprot_id", "")),
                str(row.get("matched_label", "")),
            ]),
            axis=1,
        )
        detail_frames.append(merged)
        hit_events = merged.loc[merged["is_hit"], EVENT_HIT_GROUP_COLUMNS].drop_duplicates().shape[0]
        log(
            f"chemical={chemical} label_nodes={len(merged)} label_hits={int(merged['is_label_hit'].sum())} "
            f"joint_event_hits={hit_events} hit_rows={int(merged['is_hit'].sum())} "
            f"hit_AOPs={merged.loc[merged['is_hit'], 'AOP_ID'].nunique()}"
        )
    return pd.concat(detail_frames, ignore_index=True)


def task_probability_columns(label: str) -> tuple[str, str]:
    if label not in TASK_PROBABILITY:
        raise ValueError(f"unsupported label: {label}")
    return TASK_PROBABILITY[label], TASK_RANK[label]


def family_expansion_context(label: str, binding_context: object) -> str:
    if label == "binding" or str(binding_context) == "generic_family_binding":
        return "generic_family_binding_expanded"
    return "generic_family_effect_expanded"


def build_family_event_detail(aop: pd.DataFrame) -> pd.DataFrame:
    if not FAMILY_MAP.exists():
        return pd.DataFrame()
    family = pd.read_csv(FAMILY_MAP)
    if FAMILY_EXCLUDED_RELATED.exists():
        excluded_related_df = pd.read_csv(FAMILY_EXCLUDED_RELATED)
    else:
        excluded_related_df = pd.DataFrame()
    fam_aop = aop[aop.get("Mapping_Status", "").eq("family_expanded_mapped")].copy()
    if fam_aop.empty:
        return pd.DataFrame()
    fam_dedup_key = ["AOP_ID", "Event_ID", "Protein_PR_ID", "Protein_Name", "Event_Labels"]
    before = len(fam_aop)
    fam_aop = fam_aop.drop_duplicates(fam_dedup_key, keep="first")
    removed = before - len(fam_aop)
    if removed:
        log(f"removed duplicate family AOP protein-event rows before expansion: {removed}")
    fam_aop = fam_aop.drop(
        columns=[
            col for col in [
                "Family_Name",
                "Expansion_Source",
                "Expansion_Context",
                "Use_In_AOP_Score",
            ]
            if col in fam_aop.columns
        ]
    )
    events = fam_aop.merge(family, on=["Protein_PR_ID", "Protein_Name"], how="inner")
    if events.empty:
        return pd.DataFrame()

    rows = []
    event_rows = []
    group_cols = [
        "AOP_ID",
        "Event_Type",
        "Event_ID",
        "Event_Title",
        "Protein_PR_ID",
        "Protein_Name",
        "Action",
        "Expected_Effect",
        "Event_Labels",
        "Has_Binding",
        "Has_Activation",
        "Has_Inhibition",
        "Has_Unspecified",
        "Primary_Effect",
        "Binding_Context",
        "Use_For_ToxCPI_Binding",
        "Position_Weight",
        "UniProt_ID",
        "AO_Title",
        "AOP_Title",
        "event_direction",
        "AOP_node_type",
        "AO_name",
        "AO_category",
        "Mapping_Status",
        "Mapping_Source",
        "Mapping_Action",
        "Family_Name",
        "Expansion_Source",
        "Included_Member_Rule",
        "Family_Universe_Audit_Source",
    ]
    for group_key, group in events.groupby(group_cols, dropna=False):
        event = dict(zip(group_cols, group_key))
        labels = required_event_labels(event.get("Event_Labels", ""))
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

        for chemical, slug in CHEMICAL_SLUGS.items():
            pred = load_prediction(slug)
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
                chemical_name = chemical
                casrn = ""
                ctd_chemical_id = ""
            else:
                tested = score_family_members(tested, event.get("Event_Labels", ""))
                for label in labels:
                    _, rank_col = task_probability_columns(label)
                    tested[rank_col] = pd.to_numeric(tested[rank_col], errors="coerce")
                valid_joint_scores = tested["member_joint_score"].dropna()
                if not valid_joint_scores.empty:
                    representative = tested.loc[valid_joint_scores.idxmax()]
                joint_hits = tested[tested["member_joint_hit"]].copy()
                chemical_name = (
                    str(tested["chemical_name"].dropna().iloc[0])
                    if tested["chemical_name"].notna().any()
                    else chemical
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

            def representative_number(column: str) -> float:
                if representative is None:
                    return np.nan
                value = pd.to_numeric(representative.get(column), errors="coerce")
                return float(value) if pd.notna(value) else np.nan

            representative_probabilities = {
                task: representative_number(TASK_PROBABILITY[task]) for task in TASK_PROBABILITY
            }
            representative_ranks = {
                task: representative_number(TASK_RANK[task]) for task in TASK_RANK
            }
            family_node_id = f"FAMILY:{event['Family_Name']}"
            for label in labels:
                matched_probability = representative_probabilities[label]
                matched_rank = representative_ranks[label]
                base = event.copy()
                base.update(
                    {
                        "chemical_name": chemical_name,
                        "casrn": casrn,
                        "ctd_chemical_id": ctd_chemical_id,
                        "uniprot_id": family_node_id,
                        "matched_label": label,
                        "matched_task": label,
                        "event_labels": event.get("Event_Labels", ""),
                        "gene_symbol": str(event["Family_Name"]),
                        "protein_name": str(event["Protein_Name"]),
                        "P_binding": representative_probabilities["binding"],
                        "P_activation": representative_probabilities["activation"],
                        "P_inhibition": representative_probabilities["inhibition"],
                        "S_target": representative_joint_score,
                        "rank_binding": representative_ranks["binding"],
                        "rank_activation": representative_ranks["activation"],
                        "rank_inhibition": representative_ranks["inhibition"],
                        "effect_extrapolated": False,
                        "matched_probability": matched_probability,
                        "matched_rank": matched_rank,
                        "is_label_hit": bool(
                            pd.notna(matched_probability)
                            and matched_probability > ROUTE2_HIT_THRESHOLD
                        ),
                        "is_hit": family_event_hit,
                        "P_dir": matched_probability,
                        "rank_dir": matched_rank,
                        "event_direction": label if label in {"activation", "inhibition"} else "binding-supported",
                        "direction_match_threshold": ROUTE2_HIT_THRESHOLD,
                        "interpretation_warning": FAMILY_EVENT_WARNING,
                        "Expansion_Context": family_expansion_context(label, event.get("Binding_Context", "")),
                        "Expanded_Members": expanded_members,
                        "Included_Member_Rule": event.get("Included_Member_Rule", ""),
                        "Family_Universe_Audit_Source": event.get("Family_Universe_Audit_Source", ""),
                        "Excluded_Related_Proteins_And_Reason": excluded_related,
                        "Family_Task": label,
                        "Family_Event_Hit": family_event_hit,
                        "Family_Hit_Proteins": hit_proteins,
                        "Family_Hit_UniProts": hit_uniprots,
                        "Family_Hit_Probabilities": hit_probabilities,
                        "Family_Max_Probability": representative_joint_score,
                        "Family_Max_Rank": matched_rank,
                        "Family_Hit_Count": hit_count,
                        "Family_Tested_Member_Count": int(tested.shape[0]),
                        "Family_Total_Member_Count": int(expanded.shape[0]),
                        "Family_Missing_Members": ";".join(missing),
                        "Family_Required_Labels": labels_to_string(labels),
                        "Family_Representative_Gene": representative_gene,
                        "Family_Representative_UniProt": representative_uniprot,
                        "Family_Representative_Joint_Score": representative_joint_score,
                        "Event_Threshold": ROUTE2_HIT_THRESHOLD,
                        "Use_In_AOP_Score": True,
                        "Evidence_Type": "family-expanded event hit",
                        "mapped_node_id": f"AOP:{int(event['AOP_ID'])}|Event:{int(event['Event_ID'])}",
                        "unique_model_evidence_id": "|".join([
                            "MODEL_FAMILY",
                            chemical_name,
                            representative_uniprot or family_node_id,
                            label,
                        ]),
                    }
                )
                rows.append(base)
                event_rows.append(
                    {
                        "chemical_name": chemical_name,
                        "casrn": casrn,
                        "ctd_chemical_id": ctd_chemical_id,
                        "AOP_ID": event["AOP_ID"],
                        "Event_ID": event["Event_ID"],
                        "Event_Title": event["Event_Title"],
                        "Event_Labels": event.get("Event_Labels", ""),
                        "AO_Title": event["AO_Title"],
                        "Protein_PR_ID": event["Protein_PR_ID"],
                        "Protein_Name": event["Protein_Name"],
                        "Binding_Context": event["Binding_Context"],
                        "Mapping_Status": "family_expanded_mapped",
                        "Expansion_Context": family_expansion_context(label, event.get("Binding_Context", "")),
                        "Family_Name": event["Family_Name"],
                        "Expansion_Source": event["Expansion_Source"],
                        "Expanded_Members": expanded_members,
                        "Included_Member_Rule": base["Included_Member_Rule"],
                        "Family_Universe_Audit_Source": base["Family_Universe_Audit_Source"],
                        "Excluded_Related_Proteins_And_Reason": excluded_related,
                        "Family_Task": label,
                        "Family_Event_Hit": family_event_hit,
                        "Family_Hit_Proteins": hit_proteins,
                        "Family_Hit_UniProts": hit_uniprots,
                        "Family_Hit_Probabilities": hit_probabilities,
                        "Family_Max_Probability": representative_joint_score,
                        "Family_Max_Rank": matched_rank,
                        "Family_Hit_Count": hit_count,
                        "Family_Tested_Member_Count": int(tested.shape[0]),
                        "Family_Total_Member_Count": int(expanded.shape[0]),
                        "Family_Missing_Members": ";".join(missing),
                        "Family_Required_Labels": labels_to_string(labels),
                        "Family_Representative_Gene": representative_gene,
                        "Family_Representative_UniProt": representative_uniprot,
                        "Family_Representative_Joint_Score": representative_joint_score,
                        "Event_Threshold": ROUTE2_HIT_THRESHOLD,
                        "Use_In_AOP_Score": True,
                        "Evidence_Type": "family-expanded event hit",
                        "Interpretation_Warning": FAMILY_EVENT_WARNING,
                    }
                )

    event_df = pd.DataFrame(event_rows)
    event_df.to_csv(FAMILY_EVENT_OUT, index=False)
    log(
        f"wrote {FAMILY_EVENT_OUT} rows={len(event_df)} "
        f"event_hits={int(event_df['Family_Event_Hit'].sum()) if not event_df.empty else 0}"
    )
    if not event_df.empty and event_df["Family_Missing_Members"].fillna("").astype(str).str.len().gt(0).any():
        log("family expansion has missing members; see Family_Missing_Members")
    return pd.DataFrame(rows)


def write_route2_outputs(detail: pd.DataFrame, label_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_out = detail.copy()
    detail_out["source_target_threshold_label"] = label_threshold
    detail_path = ROUTE2 / f"route2_aop_hit_detail_4chem_threshold_{label_threshold:.1f}{ROUTE2_OUTPUT_SUFFIX}.csv"
    detail_out.to_csv(detail_path, index=False)
    log(f"wrote {detail_path} rows={len(detail_out)}")

    rows = []
    for (chemical, aop_id, aop_title, ao_name), group in detail.groupby(
        ["chemical_name", "AOP_ID", "AOP_Title", "AO_name"], dropna=False
    ):
        label_rows = group.drop_duplicates(["Event_ID", "uniprot_id", "matched_label"]).copy()
        mapped_nodes = group.drop_duplicates(["Event_ID", "uniprot_id"]).copy()
        evidence_units = group.drop_duplicates(["unique_model_evidence_id"]).copy()
        hit_label_rows = label_rows[label_rows["is_hit"]].copy()
        hit_nodes = mapped_nodes[mapped_nodes["is_hit"]].copy()
        hit_evidence = evidence_units[evidence_units["is_hit"]].copy()
        n_label_rows = int(label_rows.shape[0])
        if n_label_rows == 0 or hit_nodes.empty:
            continue
        score = float(hit_label_rows["P_dir"].sum() / n_label_rows)
        n_hit_nodes = int(hit_nodes.shape[0])
        rows.append(
            {
                "chemical_name": chemical,
                "AOP_ID": aop_id,
                "AOP_Title": aop_title,
                "AO_name": ao_name,
                "AO_category": ao_name,
                "AOP_hit_score": score,
                "mean_P_hit": float(hit_label_rows["P_dir"].mean()),
                "n_hit": n_hit_nodes,
                "n_nodes_except_AO": int(mapped_nodes.shape[0]),
                "n_mapped_nodes": int(mapped_nodes.shape[0]),
                "n_unique_evidence_units": int(evidence_units.shape[0]),
                "n_hit_mapped_nodes": n_hit_nodes,
                "n_hit_unique_evidence_units": int(hit_evidence.shape[0]),
                "hit_coverage": float(n_hit_nodes / mapped_nodes.shape[0]),
                "n_binding_supported_hits": int(hit_label_rows["matched_label"].eq("binding").sum()),
                "n_activation_direction_hits": int(hit_label_rows["matched_label"].eq("activation").sum()),
                "n_inhibition_direction_hits": int(hit_label_rows["matched_label"].eq("inhibition").sum()),
                "hit_targets": ";".join(sorted(set(hit_nodes["gene_symbol"].dropna().astype(str)))),
                "hit_uniprots": ";".join(sorted(set(hit_nodes["uniprot_id"].dropna().astype(str)))),
                "hit_unique_model_evidence_ids": ";".join(sorted(set(hit_evidence["unique_model_evidence_id"].dropna().astype(str)))),
                "hit_event_labels": ";".join(
                    sorted(set((hit_label_rows["gene_symbol"].fillna(hit_label_rows["uniprot_id"]).astype(str) + ":" + hit_label_rows["matched_label"].astype(str)).tolist()))
                ),
                "hit_event_directions": ";".join(
                    sorted(set((hit_label_rows["gene_symbol"].fillna(hit_label_rows["uniprot_id"]).astype(str) + ":" + hit_label_rows["event_direction"].astype(str)).tolist()))
                ),
                "direction_match_threshold": ROUTE2_HIT_THRESHOLD,
                "source_target_threshold_label": label_threshold,
                "interpretation_warning": ROUTE2_WARNING,
            }
        )
    aop_rows = pd.DataFrame(rows)
    if not aop_rows.empty:
        aop_rows = aop_rows.sort_values(["chemical_name", "AOP_hit_score"], ascending=[True, False])
    grouped_path = ROUTE2 / f"route2_chemical_aop_ao_4chem_threshold_{label_threshold:.1f}{ROUTE2_OUTPUT_SUFFIX}.csv"
    aop_rows.to_csv(grouped_path, index=False)
    log(f"wrote {grouped_path} rows={len(aop_rows)}")

    summary_rows = []
    if not aop_rows.empty:
        for (chemical, ao_name), group in aop_rows.groupby(["chemical_name", "AO_name"], dropna=False):
            group = group.sort_values("AOP_hit_score", ascending=False)
            top = group.iloc[0]
            summary_rows.append(
                {
                    "chemical_name": chemical,
                    "AO_category": ao_name,
                    "AO_name": ao_name,
                    "max_AOP_hit_score": float(top["AOP_hit_score"]),
                    "n_AOPs_hit": int(group["AOP_ID"].nunique()),
                    "top_AOP": top["AOP_Title"],
                    "AOP_hit_scores": "; ".join(
                        f"{r.AOP_ID}:{r.AOP_hit_score:.3f}" for r in group.itertuples()
                    ),
                    "hit_targets": ";".join(sorted(set(";".join(group["hit_targets"].dropna().astype(str)).split(";")))),
                    "direction_match_threshold": ROUTE2_HIT_THRESHOLD,
                    "source_target_threshold_label": label_threshold,
                    "interpretation_warning": ROUTE2_WARNING,
                }
            )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["chemical_name", "max_AOP_hit_score"], ascending=[True, False])
    summary_path = ROUTE2 / f"route2_chemical_ao_summary_4chem_threshold_{label_threshold:.1f}{ROUTE2_OUTPUT_SUFFIX}.csv"
    summary.to_csv(summary_path, index=False)
    log(f"wrote {summary_path} rows={len(summary)}")
    return aop_rows, summary


def write_ppi_placeholders() -> None:
    PPI.mkdir(parents=True, exist_ok=True)
    note = (
        "No local STRING/PPI source file was found under the project tree. "
        "PPI is auxiliary only and was not used as AO evidence in this run.\n"
    )
    (PPI / "PPI_NOT_RUN.txt").write_text(note, encoding="utf-8")
    for threshold in [0.5, 0.7]:
        pd.DataFrame(
            columns=[
                "chemical_name",
                "gene_symbol",
                "uniprot_id",
                "S_target",
                "degree",
                "betweenness",
                "hub_rank",
                "in_top_kegg_pathway",
                "in_aop_node",
            ]
        ).to_csv(PPI / f"ppi_hub_targets_4chem_threshold_{threshold:.1f}.csv", index=False)
        pd.DataFrame(columns=["chemical_name", "source", "target", "combined_score"]).to_csv(
            PPI / f"ppi_edges_4chem_threshold_{threshold:.1f}.csv", index=False
        )
        pd.DataFrame(columns=["chemical_name", "module_id", "genes"]).to_csv(
            PPI / f"ppi_modules_4chem_threshold_{threshold:.1f}.csv", index=False
        )
    log("wrote PPI placeholder outputs; no local STRING/PPI source found")


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = 5) -> str:
    if df is None or df.empty:
        return "_No rows._\n"
    sub = df[columns].head(n).copy().fillna("")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in sub.iterrows():
        values = [str(row[col]).replace("|", "/").replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_summary(route2_aop: pd.DataFrame, route2_summary: pd.DataFrame) -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    diag = pd.read_csv(PRED / "positive_diagnostics.csv")
    route1_pathways = read_optional_csv(ROUTE1_V2 / "route1a_kegg_pathway_evidence_threshold_0.5.csv")
    route1_brite = read_optional_csv(ROUTE1_V2 / "route1c_brite_class_summary_threshold_0.5.csv")
    route1d_chem = read_optional_csv(ROUTE1_V2 / "route1d_chemical_disease_top5_threshold_0.5.csv")

    lines = ["# Downstream AO Multi-Chemical Summary", ""]
    lines.append("> Route1 uses standard KEGG ORA FDR ranking, KEGG BRITE B-level pathway-domain summaries, and CTD TF-IDF disease/outcome context.")
    lines.append("> Route2 uses label-matched AOP MIE/KE hits with P_binding/P_activation/P_inhibition > 0.7. Multi-label events require every label on the same protein (strict AND); family events require at least one member to satisfy that joint condition. Activation/inhibition are direction-matched; binding is binding-supported, not an effect direction. Scores are prioritization/display metrics, not AO probabilities.")
    lines.append("> A mapped AOP node and a unique model evidence unit are counted separately. Multiple nodes with the same chemical-protein-task unique_model_evidence_id are repeated mappings of one prediction, not independent model evidence.")
    lines.append("> Rotenone remains included because the requested rerun covers all 16 chemicals; prior caveats about its model fit still apply.")
    lines.append("")

    for chemical, slug in CHEMICAL_SLUGS.items():
        lines.extend([f"## {chemical}", ""])
        d = diag[diag["chemical_name"].eq(chemical)]
        lines.append(markdown_table(d, ["threshold", "n_binding_pos", "n_activation_pos", "n_inhibition_pos", "n_union_pos", "union_fraction_of_universe"], 2))
        top_s = pd.read_csv(PRED / f"top10_S_target_{slug}.csv")
        lines.extend(["Top S_target:", markdown_table(top_s, ["gene_symbol", "uniprot_id", "P_binding", "P_activation", "P_inhibition", "S_target"], 10)])

        r1p = route1_pathways[route1_pathways["chemical_name"].eq(chemical)].sort_values("fdr") if not route1_pathways.empty else pd.DataFrame()
        lines.extend(["Route1 top KEGG pathways:", markdown_table(r1p, ["pathway_id", "pathway_name", "fdr", "overlap_count", "fold_enrichment", "selected_for_downstream"], 10)])

        r1b = route1_brite[route1_brite["chemical_name"].eq(chemical)].sort_values("min_fdr") if not route1_brite.empty else pd.DataFrame()
        lines.extend(["Route1 BRITE pathway domains:", markdown_table(r1b, ["brite_A_domain", "brite_B_class", "n_significant_pathways", "min_fdr", "mean_fold_enrichment"], 10)])

        if not route1d_chem.empty:
            r1d_score_col = "display_association_score" if "display_association_score" in route1d_chem.columns else "display_rank_score"
            r1d = route1d_chem[route1d_chem["chemical_name"].eq(chemical)].sort_values(r1d_score_col, ascending=False)
            r1d_cols = ["disease_name", r1d_score_col, "n_supporting_pathways", "best_pathway_name"]
        else:
            r1d = pd.DataFrame()
            r1d_cols = ["disease_name", "display_rank_score", "n_supporting_pathways", "best_pathway_name"]
        lines.extend(["Route1d CTD disease/outcome context:", markdown_table(r1d, r1d_cols, 5)])

        r2 = route2_aop[route2_aop["chemical_name"].eq(chemical)].sort_values("AOP_hit_score", ascending=False) if not route2_aop.empty else pd.DataFrame()
        lines.extend(["Route2 top AOP-AO:", markdown_table(r2, ["AOP_ID", "AOP_Title", "AO_name", "AOP_hit_score", "n_hit", "n_binding_supported_hits", "n_activation_direction_hits", "n_inhibition_direction_hits", "hit_targets"], 10)])

    md_path = SUMMARY / f"final_case_summary_4chem{ROUTE2_OUTPUT_SUFFIX}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {md_path}")

    xlsx_path = SUMMARY / f"final_case_summary_4chem{ROUTE2_OUTPUT_SUFFIX}.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            diag.to_excel(writer, sheet_name="diagnostics", index=False)
            route1_pathways.head(5000).to_excel(writer, sheet_name="route1_pathways", index=False)
            route1_brite.head(5000).to_excel(writer, sheet_name="route1_brite", index=False)
            route1d_chem.head(5000).to_excel(writer, sheet_name="route1d_ctd_top5", index=False)
            route2_aop.head(5000).to_excel(writer, sheet_name="route2_aop_ao", index=False)
            route2_summary.head(5000).to_excel(writer, sheet_name="route2_ao_summary", index=False)
        log(f"wrote {xlsx_path}")
    except Exception as exc:
        log(f"could not write xlsx: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map ToxCPI prediction CSV files to direction-matched AOP/AO events"
    )
    parser.add_argument("--prediction-dir", required=True, type=Path,
                        help="Directory containing all_human_proteome_scores_<slug>.csv")
    parser.add_argument("--aop-data-dir", required=True, type=Path,
                        help="Directory containing the packaged AOP mapping CSV files")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New output directory; it is created if needed")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Strict per-label probability threshold (default: 0.7)")
    parser.add_argument("--chemical-list", required=True, type=Path,
                        help="CSV with chemical_name and optional slug columns")
    parser.add_argument("--route1-dir", type=Path,
                        help="Optional Route 1 directory used only with --write-summary")
    parser.add_argument("--write-summary", action="store_true",
                        help="Also build the legacy combined Route 1/Route 2 summary")
    return parser.parse_args()


def prediction_slug(name: str) -> str:
    return name.lower().replace(",", "").replace(" ", "_").replace("/", "_")


def configure(args: argparse.Namespace) -> None:
    global INPUT, PRED, ROUTE1_V2, RUNTIME, ROUTE2, PPI, SUMMARY, LOG
    global AOP_DIR, AOP_PROTEINS_MULTILABEL, AOP_PROTEINS
    global AOP_COMPLETE_MULTILABEL, AOP_COMPLETE, MANUAL_MAP
    global AUTHORITATIVE_MANUAL_MAP, FAMILY_MAP, FAMILY_EXCLUDED_RELATED
    global FAMILY_EVENT_OUT, ROUTE2_HIT_THRESHOLD, CHEMICAL_SLUGS
    global ROUTE2_WARNING, FAMILY_EVENT_WARNING

    PRED = args.prediction_dir.expanduser().resolve()
    AOP_DIR = args.aop_data_dir.expanduser().resolve()
    INPUT = AOP_DIR
    RUNTIME = args.output_dir.expanduser().resolve()
    ROUTE2 = RUNTIME / "route2_aop"
    PPI = RUNTIME / "ppi"
    SUMMARY = RUNTIME / "summary"
    LOG = RUNTIME / "logs" / "run_aop_mapping.log"
    ROUTE1_V2 = (args.route1_dir.expanduser().resolve()
                 if args.route1_dir else RUNTIME / "route1_ctd")
    AOP_PROTEINS_MULTILABEL = AOP_DIR / "human_aop_proteins_multilabel.csv"
    AOP_PROTEINS = AOP_DIR / "human_aop_proteins.csv"
    AOP_COMPLETE_MULTILABEL = AOP_DIR / "human_aop_complete_multilabel.csv"
    AOP_COMPLETE = AOP_DIR / "human_aop_complete.csv"
    MANUAL_MAP = AOP_DIR / "aop_manual_uniprot_mapping.csv"
    AUTHORITATIVE_MANUAL_MAP = MANUAL_MAP
    FAMILY_MAP = AOP_DIR / "aop_family_expansion_map.csv"
    FAMILY_EXCLUDED_RELATED = AOP_DIR / "aop_family_expansion_excluded_related_proteins.csv"
    FAMILY_EVENT_OUT = ROUTE2 / f"route2_family_expanded_event_hits_threshold_{args.threshold:g}.csv"
    ROUTE2_HIT_THRESHOLD = args.threshold
    ROUTE2_WARNING = (
        f"Label-matched AOP event; every required label must be > {args.threshold:g} "
        "on the same protein. This is prioritization evidence, not an AO probability."
    )
    FAMILY_EVENT_WARNING = (
        f"Family event; one member must exceed every required label threshold > {args.threshold:g}. "
        "This is prioritization evidence, not an AO probability."
    )
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
    aop = build_aop_map()
    direct_detail = build_directional_detail(aop)
    family_detail = build_family_event_detail(aop)
    if not family_detail.empty:
        detail = pd.concat([direct_detail, family_detail], ignore_index=True, sort=False)
        log(
            f"combined Route2 matched nodes direct_rows={len(direct_detail)} "
            f"family_rows={len(family_detail)} total_rows={len(detail)}"
        )
    else:
        detail = direct_detail
    route2_aop = pd.DataFrame()
    route2_summary = pd.DataFrame()
    route2_aop, route2_summary = write_route2_outputs(detail, ROUTE2_HIT_THRESHOLD)
    write_ppi_placeholders()
    if args.write_summary:
        build_summary(route2_aop, route2_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
