"""Final unweighted Route 2 matching; inline curated members are authoritative."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

TASK_PROBABILITY = {"binding": "P_binding", "activation": "P_activation", "inhibition": "P_inhibition"}
THRESHOLD = 0.7
SCHEMA = "toxcpi-route2-unweighted-v1"
WARNING = ("Mechanistic hypothesis only; one mapping is one node regardless of member count. "
           "Shared chemical-protein-task predictions are shared evidence. No aggregate scores or rankings.")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash_stream(handle)


def _hash_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def packed(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def required_event_labels(value: object) -> list[str]:
    labels = str(value).strip().split(";")
    if not labels or any(label not in TASK_PROBABILITY for label in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"Invalid curated Event_Labels: {value!r}; no title-based inference is permitted")
    return [label for label in TASK_PROBABILITY if label in labels]


def event_joint_hit(probabilities, event_labels: object, threshold: float = THRESHOLD) -> bool:
    labels = required_event_labels(event_labels)
    for label in labels:
        value = pd.to_numeric(probabilities.get(TASK_PROBABILITY[label]), errors="coerce")
        if pd.isna(value) or not 0 <= float(value) <= 1 or not float(value) > threshold:
            return False
    return True


def load_mapping(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"AOP_ID", "AOP_Title", "Event_ID", "Event_Title", "Event_Type", "Protein_Name",
                "Event_Labels", "Resolved_UniProt_ID", "Use_For_AOP_Matching"}
    if required - set(frame):
        raise ValueError(f"Final mapping schema required; missing {sorted(required - set(frame))}")
    if {"Position_Weight", "Use_In_AOP_Score"} & set(frame):
        raise ValueError("Historical scoring schema is not accepted as final matching input")
    if frame.empty:
        raise ValueError("Empty final mapping")
    ids = []
    for row in frame.to_dict("records"):
        flag = row["Use_For_AOP_Matching"].strip().lower()
        if flag not in {"true", "false"}:
            raise ValueError("Use_For_AOP_Matching must be explicitly True or False")
        labels = required_event_labels(row["Event_Labels"])
        members = row["Resolved_UniProt_ID"].split(";")
        if any(not re.fullmatch(r"[A-Z0-9]{6}(?:[A-Z0-9]{4})?", uid) for uid in members):
            raise ValueError(f"Invalid/empty resolved member list: {members}")
        if len(set(members)) != len(members):
            raise ValueError("Repeated member within one curated mapping")
        for field in ("UniProt_ID", "Manual_UniProt_ID"):
            if row.get(field) and row[field] != row["Resolved_UniProt_ID"]:
                raise ValueError(f"{field} conflicts with curated resolved members; review input identity")
        if not all(row[key].strip() for key in ("AOP_ID", "AOP_Title", "Event_ID", "Protein_Name")):
            raise ValueError("Missing mapping identity")
        identity = [row["AOP_ID"], row["Event_ID"], row["Protein_Name"], labels, sorted(members)]
        ids.append("mapping:" + hashlib.sha256(packed(identity).encode()).hexdigest())
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate event-mapping identity; no silent deduplication")
    frame.insert(0, "mapping_id", ids)
    frame.insert(1, "source_csv_record", range(1, len(frame) + 1))
    # Keep each source record exactly once. No historical inventory/manual/family file is read.
    return frame.loc[frame.Use_For_AOP_Matching.str.strip().str.lower().eq("true")].copy()


def load_prediction(path: Path, chemical: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"uniprot_id": str})
    if "uniprot_id" not in frame or "chemical_name" not in frame:
        raise ValueError(f"{path}: uniprot_id and chemical_name are required")
    if frame.empty or frame.chemical_name.isna().any() or set(frame.chemical_name) != {chemical}:
        raise ValueError(f"{path}: expected predictions for exactly {chemical!r}")
    if frame.uniprot_id.isna().any() or frame.uniprot_id.str.strip().eq("").any() or frame.uniprot_id.duplicated().any():
        raise ValueError(f"{path}: missing or duplicate UniProt prediction identity")
    for col in TASK_PROBABILITY.values():
        if col not in frame:
            frame[col] = float("nan")
        raw = frame[col]
        numeric = pd.to_numeric(raw, errors="coerce")
        if (raw.notna() & numeric.isna()).any() or (numeric.notna() & ~numeric.between(0, 1)).any():
            raise ValueError(f"{path}: {col} must contain raw probabilities in [0,1] or missing values")
        frame[col] = numeric
    return frame.set_index("uniprot_id", verify_integrity=True)


def match_chemical(mapping: pd.DataFrame, prediction: pd.DataFrame, chemical: str):
    nodes, members, evidence = [], [], {}
    for row in mapping.to_dict("records"):
        labels = required_event_labels(row["Event_Labels"])
        uids = row["Resolved_UniProt_ID"].split(";")
        qualifiers, missing, evidence_ids, qualifying_ids = [], [], [], []
        for uid in uids:
            probs = prediction.loc[uid].to_dict() if uid in prediction.index else {}
            hit = event_joint_hit(probs, row["Event_Labels"])
            missing_labels = [label for label in labels if pd.isna(probs.get(TASK_PROBABILITY[label]))]
            if missing_labels:
                missing.append({"uniprot_id": uid, "labels": missing_labels})
            if hit:
                qualifiers.append(uid)
            for label in labels:
                probability = probs.get(TASK_PROBABILITY[label])
                value = None if pd.isna(probability) else float(probability)
                # JSON tuple is collision-free even if a chemical name contains separators.
                eid = packed([chemical, uid, label])
                evidence_ids.append(eid)
                if hit:
                    qualifying_ids.append(eid)
                item = evidence.setdefault(eid, {"model_evidence_id": eid, "chemical_name": chemical,
                    "uniprot_id": uid, "task": label, "raw_probability": value,
                    "prediction_available": value is not None, "linked_mapping_ids": [], "qualifying_mapping_ids": []})
                item["linked_mapping_ids"].append(row["mapping_id"])
                if hit:
                    item["qualifying_mapping_ids"].append(row["mapping_id"])
                members.append({"chemical_name": chemical, "mapping_id": row["mapping_id"],
                    "AOP_ID": row["AOP_ID"], "Event_ID": row["Event_ID"], "uniprot_id": uid,
                    "gene_symbol": probs.get("gene_symbol", ""), "required_label": label,
                    "raw_probability": value, "prediction_available": value is not None,
                    "label_hit": value is not None and value > THRESHOLD, "member_joint_hit": hit,
                    "model_evidence_id": eid})
        nodes.append({**row, "chemical_name": chemical, "is_hit": bool(qualifiers),
            "qualifying_members": ";".join(qualifiers), "n_qualifying_members": len(qualifiers),
            "n_explicit_members": len(uids), "missing_required_predictions": packed(missing),
            "required_model_evidence_ids": packed(evidence_ids),
            "qualifying_model_evidence_ids": packed(qualifying_ids), "interpretation_warning": WARNING})
    for item in evidence.values():
        for col in ("linked_mapping_ids", "qualifying_mapping_ids"):
            item[col] = sorted(set(item[col]))
        item["n_linked_mapping_nodes"] = len(item["linked_mapping_ids"])
        item["n_qualifying_mapping_nodes"] = len(item["qualifying_mapping_ids"])
        item["shared_evidence"] = item["n_qualifying_mapping_nodes"] > 1
        for col in ("linked_mapping_ids", "qualifying_mapping_ids"):
            item[col] = packed(item[col])
    for member in members:
        item = evidence[member["model_evidence_id"]]
        member["shared_evidence"] = item["shared_evidence"]
        member["n_qualifying_mapping_nodes"] = item["n_qualifying_mapping_nodes"]
    for node in nodes:
        node["shared_qualifying_evidence_ids"] = packed([
            eid for eid in json.loads(node["qualifying_model_evidence_ids"]) if evidence[eid]["shared_evidence"]])
    return pd.DataFrame(nodes), pd.DataFrame(members), pd.DataFrame(evidence.values())


def prediction_slug(name: str) -> str:
    return name.lower().replace(",", "").replace(" ", "_").replace("/", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Final unweighted 91-mapping Route 2 (raw probabilities strictly >0.7)")
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--aop-data-dir", required=True, type=Path)
    parser.add_argument("--chemical-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, choices=[THRESHOLD], default=THRESHOLD)
    args = parser.parse_args()
    path = args.aop_data_dir / "human_aop_proteins_multilabel.csv"
    mapping = load_mapping(path)
    chemicals = pd.read_csv(args.chemical_list, dtype=str, keep_default_na=False)
    if "chemical_name" not in chemicals or chemicals.empty or chemicals.chemical_name.str.strip().eq("").any() or chemicals.chemical_name.duplicated().any():
        raise ValueError("Chemical list must have nonempty unique chemical_name values")
    outputs = [[], [], []]
    inputs = {str(path.resolve()): sha256(path), str(args.chemical_list.resolve()): sha256(args.chemical_list)}
    for row in chemicals.to_dict("records"):
        slug = row.get("slug") or prediction_slug(row["chemical_name"])
        if Path(slug).name != slug or slug in {".", ".."}:
            raise ValueError("Chemical slug must be a filename component")
        pred_path = args.prediction_dir / f"all_human_proteome_scores_{slug}.csv"
        pred = load_prediction(pred_path, row["chemical_name"])
        inputs[str(pred_path.resolve())] = sha256(pred_path)
        result = match_chemical(mapping, pred, row["chemical_name"])
        for frames, frame in zip(outputs, result):
            frames.append(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ["route2_event_mapping_nodes.csv", "route2_member_label_evidence.csv", "route2_unique_model_evidence.csv"]
    tables = [pd.concat(frames, ignore_index=True) for frames in outputs]
    for name, frame in zip(names, tables):
        frame.to_csv(args.output_dir / name, index=False)
    mapping.to_csv(args.output_dir / "aop_mapping_input_with_ids.csv", index=False)
    counts = [{"chemical_name": name, "n_mapping_nodes": len(group), "n_hit_mapping_nodes": int(group.is_hit.sum()),
               "n_AOPs_with_hits": group.loc[group.is_hit, "AOP_ID"].nunique()}
              for name, group in tables[0].groupby("chemical_name", sort=False)]
    pd.DataFrame(counts).to_csv(args.output_dir / "route2_match_counts.csv", index=False)
    metadata = {"schema": SCHEMA, "threshold": THRESHOLD, "comparison": ">", "inputs_sha256": inputs,
        "n_mapping_nodes_per_chemical": len(mapping), "counts": counts, "interpretation_warning": WARNING,
        "implementation_sha256": sha256(Path(__file__)),
        "outputs_sha256": {p.name: sha256(p) for p in args.output_dir.glob("*.csv") if p.name in names + ["aop_mapping_input_with_ids.csv", "route2_match_counts.csv"]}}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0
