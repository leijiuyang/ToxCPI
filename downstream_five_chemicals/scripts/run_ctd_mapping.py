


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ora, read_gmt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True, type=Path,
                        help="Directory containing all_human_proteome_scores_<slug>.csv")
    parser.add_argument("--ctd-data-dir", required=True, type=Path,
                        help="Directory containing the two packaged CTD parquet files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chemical-list", required=True, type=Path,
                        help="CSV with chemical_name, ctd_chemical_id, and optional slug")
    parser.add_argument("--gmt", required=True, type=Path, help="KEGG GMT file")
    parser.add_argument("--background", required=True, type=Path,
                        help="One background gene symbol per line")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--min-overlap", type=int, default=2)
    return parser.parse_args()


def slugify(name: str) -> str:
    return name.lower().replace(",", "").replace(" ", "_").replace("/", "_")


def normalize_pathway_id(value: object) -> str:
    text = str(value).strip()
    for prefix in ("KEGG:", "path:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text


def validate_chemical_list(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"chemical_name", "ctd_chemical_id"}
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Chemical list is empty or missing columns: {missing}")
    if "slug" not in frame.columns:
        frame["slug"] = frame["chemical_name"].astype(str).map(slugify)
    return frame


def load_prediction(path: Path, threshold: float) -> tuple[set[str], pd.DataFrame]:
    required = ["chemical_name", "gene_symbol", "P_binding", "P_activation", "P_inhibition"]
    frame = pd.read_csv(path, usecols=required)
    frame["gene_symbol"] = frame["gene_symbol"].fillna("").astype(str).str.strip()
    frame = frame[frame["gene_symbol"].ne("") & frame["gene_symbol"].str.lower().ne("nan")]
    probabilities = frame[["P_binding", "P_activation", "P_inhibition"]].apply(
        pd.to_numeric, errors="coerce"
    )
    frame["score"] = probabilities.max(axis=1)
    selected = frame.loc[frame["score"].ge(threshold), "gene_symbol"]
    return set(selected), frame


def main() -> int:
    args = parse_args()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    ctd_dir = args.ctd_data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    chemicals = validate_chemical_list(args.chemical_list.expanduser().resolve())
    background = {
        line.strip() for line in args.background.expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    pathways = read_gmt(args.gmt.expanduser().resolve())
    comparable_pathway_ids = {normalize_pathway_id(pathway_id) for pathway_id in pathways}
    if not background:
        raise ValueError("Background gene list is empty")

    ora_frames: list[pd.DataFrame] = []
    for chemical in chemicals.itertuples(index=False):
        path = prediction_dir / f"all_human_proteome_scores_{chemical.slug}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Prediction CSV not found: {path}")
        targets, scores = load_prediction(path, args.threshold)
        result = ora(targets, background, pathways)
        result["chemical_name"] = str(chemical.chemical_name)
        result["ctd_chemical_id"] = str(chemical.ctd_chemical_id)
        result["threshold"] = args.threshold
        result["n_prediction_rows"] = len(scores)
        ora_frames.append(result)

    ora_all = pd.concat(ora_frames, ignore_index=True)
    ora_all["pathway_id"] = ora_all["pathway_id"].map(normalize_pathway_id)
    ora_sig = ora_all[
        ora_all["fdr"].lt(args.fdr) & ora_all["overlap_count"].ge(args.min_overlap)
    ].copy()
    ora_sig = ora_sig.sort_values(
        ["chemical_name", "fdr", "p_value", "overlap_count", "fold_enrichment"],
        ascending=[True, True, True, False, False],
    ).reset_index(drop=True)
    ora_sig["standard_ora_rank"] = ora_sig.groupby("chemical_name").cumcount() + 1
    ora_sig["selected_for_downstream"] = ora_sig["standard_ora_rank"].le(10)
    ora_sig["top10_pathway_rank"] = ora_sig["standard_ora_rank"].where(
        ora_sig["selected_for_downstream"]
    )
    ora_all.to_csv(output_dir / "kegg_ora_all.csv", index=False)
    ora_sig.to_csv(output_dir / "kegg_ora_significant.csv", index=False)

    ctd_chem = pd.read_parquet(ctd_dir / "ctd_kegg_chem_pathway_significant.parquet")
    ctd_chem["pathway_id"] = ctd_chem["pathway_id"].map(normalize_pathway_id)
    ctd_chem["ctd_chemical_id"] = ctd_chem["chemical_id_ctd"].astype(str)
    ctd_disease = pd.read_parquet(ctd_dir / "ctd_kegg_pathway_disease.parquet")
    ctd_disease["pathway_id"] = ctd_disease["pathway_id"].map(normalize_pathway_id)

    summary_rows: list[dict[str, object]] = []
    detail_frames: list[pd.DataFrame] = []
    for chemical in chemicals.itertuples(index=False):
        name, ctd_id = str(chemical.chemical_name), str(chemical.ctd_chemical_id)
        predicted = ora_sig[ora_sig["chemical_name"].eq(name)].copy()
        reference_raw = ctd_chem[ctd_chem["ctd_chemical_id"].eq(ctd_id)].copy()
        reference = reference_raw[
            reference_raw["pathway_id"].isin(comparable_pathway_ids)
        ].copy()
        predicted_ids, reference_ids = set(predicted["pathway_id"]), set(reference["pathway_id"])
        union = predicted_ids | reference_ids
        overlap = predicted_ids & reference_ids
        summary_rows.append({
            "chemical_name": name,
            "ctd_chemical_id": ctd_id,
            "threshold": args.threshold,
            "n_predicted_enriched": len(predicted_ids),
            "n_ctd_significant": len(reference_ids),
            "n_ctd_significant_raw": reference_raw["pathway_id"].nunique(),
            "n_ctd_significant_comparable": len(reference_ids),
            "n_overlap": len(overlap),
            "predicted_supported_by_ctd_fraction": len(overlap) / len(predicted_ids) if predicted_ids else 0.0,
            "ctd_recovered_fraction": len(overlap) / len(reference_ids) if reference_ids else 0.0,
            "our_supported_by_ctd_fraction": len(overlap) / len(predicted_ids) if predicted_ids else 0.0,
            "ctd_recovered_by_our_fraction": len(overlap) / len(reference_ids) if reference_ids else 0.0,
            "jaccard": len(overlap) / len(union) if union else 0.0,
        })
        merged = predicted.merge(
            reference.drop_duplicates("pathway_id"), on="pathway_id", how="outer",
            suffixes=("_predicted", "_ctd")
        )
        merged["chemical_name"] = name
        merged["ctd_chemical_id"] = ctd_id
        merged["comparison_status"] = merged["pathway_id"].map(
            lambda pid: "overlap" if pid in overlap else ("predicted_only" if pid in predicted_ids else "ctd_only")
        )
        detail_frames.append(merged)

    summary = pd.DataFrame(summary_rows)
    detail = pd.concat(detail_frames, ignore_index=True)
    summary.to_csv(output_dir / "predicted_vs_ctd_pathway_summary.csv", index=False)
    summary.to_csv(output_dir / "predicted_vs_ctd_pathway_overlap_summary.csv", index=False)
    detail.to_csv(output_dir / "predicted_vs_ctd_pathway_detail.csv", index=False)

    selected_ora = ora_sig[ora_sig["selected_for_downstream"]].copy()
    disease = selected_ora.merge(
        ctd_disease, on="pathway_id", how="inner", suffixes=("_ora", "_ctd")
    )
    if "pathway_name_ora" in disease.columns:
        disease["pathway_name"] = disease["pathway_name_ora"].fillna(disease.get("pathway_name_ctd"))
    if "n_inference_genes" in disease.columns:
        disease["CTD_link_support"] = np.log1p(
            pd.to_numeric(disease["n_inference_genes"], errors="coerce").fillna(0)
        )
        fold_column = "fold_enrichment_ora" if "fold_enrichment_ora" in disease.columns else "fold_enrichment"
        disease["display_rank_score"] = disease[fold_column] * disease["CTD_link_support"]
    disease = disease.rename(columns={
        "fdr": "pathway_fdr",
        "p_value": "pathway_p_value",
        "fold_enrichment": "chemical_pathway_association",
        "standard_ora_rank": "pathway_rank",
    })
    disease = disease.sort_values(
        ["chemical_name", "pathway_rank", "CTD_link_support", "display_rank_score"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    disease["pathway_disease_rank"] = (
        disease.groupby(["chemical_name", "pathway_id"]).cumcount() + 1
    )
    disease.to_csv(output_dir / "predicted_pathway_ctd_disease.csv", index=False)
    ora_sig.to_csv(
        output_dir / f"route1b_kegg_pathway_ora_ranked_threshold_{args.threshold:.1f}.csv",
        index=False,
    )
    disease_top5 = disease[disease["pathway_disease_rank"].le(5)].copy()
    disease_top5.to_csv(
        output_dir / f"route1d_ctd_pathway_disease_top5_threshold_{args.threshold:.1f}.csv",
        index=False,
    )
    print(f"Wrote CTD mapping outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
