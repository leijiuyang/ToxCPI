#!/usr/bin/env python3
"""R4-05: existing formal effect scores only; no model loading or inference.

Run using the interpreter recorded in environment.json. Writes only next to this
script. Each of exactly 2000 draws is retained, including undefined metrics.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import pickle
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
import rdkit
from rdkit import Chem
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = Path(__file__).resolve().parent
HOME = Path('/home/data2/leijiuyang')
ROOT = HOME / '1_TOXCPI/leijiuyang'
WORK = HOME / 'Paper_revise/返修工作区'
DOC = WORK / '15_实验结果回写/本次回复调整工作区'
BLOCKS = ['activation_warm', 'activation_cold', 'inhibition_warm', 'inhibition_cold']
MODELS = ['ToxCPI', 'RF', 'MLP']
METRICS = ['AUROC', 'AP', 'Accuracy', 'F1', 'MCC']
SEEDS = dict(zip(BLOCKS, [2026110501, 2026110502, 2026110503, 2026110504]))
B = 2000


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def js(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def canon(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        raise ValueError(f'Invalid graph key: {s}')
    # Atom maps are annotations, not molecular graph identity. None are present
    # in these inputs; removal is explicit for reproducibility of the rule.
    for atom in m.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def prepared(y, s):
    order = np.argsort(-s, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(s[order])), len(s)-1]
    return order, ends, y[order], s >= 0.5


def calculate(y, w, cache):
    """Frequency-weighted metrics, with explicit mathematical undefinedness."""
    order, ends, sy, pred = cache
    sw = w[order]
    tp = np.cumsum(sw*sy)[ends]
    fp = np.cumsum(sw*(1-sy))[ends]
    p, n = tp[-1], fp[-1]
    auc = ap = np.nan
    # Both ranking endpoints declared undefined for a single-class draw.
    if p > 0 and n > 0:
        auc = np.trapz(np.r_[0, tp], np.r_[0, fp])/(p*n)
        ap = np.sum(np.diff(np.r_[0, tp])*tp/np.maximum(tp+fp, 1))/p
    TP = np.sum(w*y*pred)
    FP = np.sum(w*(1-y)*pred)
    TN, FN = n-FP, p-TP
    fden = 2*TP+FP+FN
    mden = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    return np.array([auc, ap, (TP+TN)/(p+n),
                     2*TP/fden if fden else np.nan,
                     (TP*TN-FP*FN)/np.sqrt(mden) if mden else np.nan])


def quant(x):
    x = x[np.isfinite(x)]
    return np.quantile(x, [.025, .975], method='linear') if len(x) else [np.nan, np.nan]


def main():
    assert not (OUT/'metrics_and_paired_differences.csv').exists(), 'Refuse overwrite of completed run'
    assets = {}

    def add(p, role, block='all', model='all'):
        p = Path(p)
        if p not in assets:
            assets[p] = dict(path=str(p), role=role, block=block, model=model,
                             exists=p.is_file(), bytes=p.stat().st_size if p.is_file() else None,
                             sha256_before=sha(p) if p.is_file() else '')
        else:
            for k, v in [('role', role), ('block', block), ('model', model)]:
                old = assets[p][k].split(';')
                if v not in old:
                    assets[p][k] += ';'+v
        return p

    for f in DOC.iterdir():
        if f.is_file():
            add(f, 'protected_current_document_not_written')
    evidence = WORK/'15_实验结果回写'
    for f in ['证据索引.md', '02_证据索引与验收结果.md', '证据—表格定位索引.csv', '复现资产清单.md']:
        add(evidence/f, 'round15_evidence_entry')
    for f in ['处理说明.md', '补充材料/基线与配对统计.py', '补充材料/基线与配对统计结果.json']:
        add(WORK/'8'/f, 'prior_scope_and_archived_metrics_read_only')
    for f in ['test1/baseline_compare/effect_main_baseline_comparison_20260601.md',
              'test1/explanation0/configs/checkpoint_manifest.md',
              'test1/experiments/ablation/run_s3_ablation.py',
              'test1/baseline_compare/ECFP_ESM2_XGBRF/run_xgb_rf.py',
              'test1/baseline_compare/ECFP_ESM2_XGBRF/results/metrics_summary.json',
              'test1/baseline_compare/ECFP_ESM2_MLP/train_mlp_cached.py',
              'test1/baseline_compare/ECFP_ESM2_MLP/feature_cache/feature_cache_report.json',
              'PMF-CPI/classification/train_toxcpi_effect.py',
              'PMF-CPI/classification/prepare_toxcpi_effect.py',
              'DTIAM-main/DTIAM-main/run_toxcpi_on_dtiam.py',
              'test1/baseline_compare/CTD_PATHWAY/pmf_effect/manifests/pmf_prediction_run_summary.json',
              'test1/baseline_compare/CTD_PATHWAY/pmf_effect/manifests/pmf_prediction_manifest.csv']:
        add(ROOT/f, 'method_provenance_or_gap_audit_not_executed')
    js(OUT/'environment.json', dict(python=sys.version, executable=sys.executable,
                                   numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                                   rdkit=rdkit.__version__, platform=platform.platform()))
    # Written before generating any bootstrap result; all blocks fixed in advance.
    js(OUT/'design.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        blocks=BLOCKS, models_in_analysis=MODELS,
        reported_main_baselines=['PMF-CPI', 'RF', 'MLP'],
        separately_reported_historical_model='DTIAM; different negative pool; no test scores',
        replicates=B, seeds=SEEDS, rng='numpy.Generator(PCG64(seed))',
        group_order='lexicographically sorted canonical isomeric SMILES',
        group_key='RDKit MolFromSmiles default sanitization; remove atom maps; MolToSmiles(canonical=True,isomericSmiles=True)',
        molecular_policy='Retain stereo, isotope, formal charge, all fragments; no neutralization, tautomer normalization, parent selection, or record deduplication. Ordinary explicit hydrogens follow RDKit default molecular parsing.',
        sampling='G integers uniform in [0,G), replacement, exactly 2000 draws, all pairs of selected groups receive group frequency; one shared weight vector per draw for all models',
        invalid_draws='Do not redraw. Ranking metrics NA on single-class draws; F1 NA if 2TP+FP+FN=0; MCC NA if its denominator=0; otherwise retain mathematically defined metrics.',
        ci='Pointwise 2.5/97.5 percentiles of finite draws, numpy.quantile(method=linear); valid counts reported by metric',
        delta='ToxCPI minus comparator within each shared draw',
        classification_rule='raw probability >=0.5; equality positive for all available models; PMF historical rule >0.5 (scores missing)',
        ap='average_precision_score: non-interpolated recall-increment weighted precision, ties grouped',
        multiplicity='No adjustment, no p values',
        scope='Fixed archived checkpoints and fixed existing labels/splits. Test chemical-group sampling uncertainty only.'))
    prior = json.loads((WORK/'8/补充材料/基线与配对统计结果.json').read_text())
    rfmeta = json.loads((ROOT/'test1/baseline_compare/ECFP_ESM2_XGBRF/results/metrics_summary.json').read_text())
    mlpfeatures = json.loads((ROOT/'test1/baseline_compare/ECFP_ESM2_MLP/feature_cache/feature_cache_report.json').read_text())
    summary, denominators, audits, histories, provenance, gaps, block_status = [], [], [], [], [], [], []
    for block in BLOCKS:
        task, split = block.split('_')
        out = OUT/block
        out.mkdir(exist_ok=True)
        srcpath = add(ROOT/f'test1/data/interaction_type_1to1_targetaware/effect_{block}_test.csv', 'fixed_test_labels_and_graphs', block)
        src = pd.read_csv(srcpath)
        assert not src[['smiles', 'uniprot_id', 'label', 'chemical_name']].isna().any().any()
        assert set(src.label.unique()) == {0, 1}
        assert not src.duplicated(['smiles', 'uniprot_id']).any(), 'Never silently deduplicate ambiguous record keys'
        lookup = {k: i for i, k in enumerate(zip(src.smiles, src.uniprot_id))}
        frame = src.copy()
        frame.insert(0, 'source_row_index', np.arange(len(src)))
        frame.insert(0, 'record_id', [f'{block}:test:{i}' for i in range(len(src))])
        prediction_rows = {}
        for model in MODELS:
            if model == 'ToxCPI':
                base = ROOT/'test1/experiments/ablation'
                ppath = add(base/f'predictions/{block}_full_seed42_test_predictions.csv', 'prediction_scores_used', block, model)
                configpath = add(base/f'configs/{block}_full_seed42.json', 'formal_model_configuration', block, model)
                respath = add(base/f'results/{block}_full_seed42.json', 'archived_result_provenance', block, model)
                add(base/f'logs/{block}_full_seed42.log', 'archived_log', block, model)
                config, res = json.loads(configpath.read_text()), json.loads(respath.read_text())
                formal = add(ROOT/config['formal_checkpoint_reused'], 'formal_checkpoint_hashed_only', block, model)
                copied = add(ROOT/res['checkpoint_file'], 'reused_checkpoint_copy_hashed_only', block, model)
                assert sha(formal) == sha(copied)
                assert sha(srcpath) == res['split_test_sha256'] == config['split_sha256']['test']
                assert config['feature_eligible_rows']['test'] == config['source_split_rows']['test'] == len(src)
                p = pd.read_csv(ppath)
                ix = p.source_row_index.to_numpy(dtype=int)
                assert np.all((ix >= 0) & (ix < len(src))) and len(set(ix)) == len(ix)
                aligned = src.iloc[ix].reset_index(drop=True)
                for col in ['chemical_name', 'uniprot_id', 'label']:
                    assert aligned[col].equals(p[col])
                assert (p.validation_threshold == .5).all()
                score, saved = p.probability.to_numpy(), p.prediction.to_numpy()
                feature_filter = 'formal exported source_row_index; config confirms 0 feature exclusions'
                checkpoint = str(formal)
            else:
                family = 'ECFP_ESM2_XGBRF' if model == 'RF' else 'ECFP_ESM2_MLP'
                base = ROOT/f'test1/baseline_compare/{family}'
                ppath = add(base/f'results/{model}_{block}_test_predictions.csv', 'prediction_scores_used', block, model)
                p = pd.read_csv(ppath)
                assert not p[['smiles', 'uniprot_id', 'true_label', 'pred_prob']].isna().any().any()
                assert not p.duplicated(['smiles', 'uniprot_id']).any()
                ix = np.array([lookup[k] for k in zip(p.smiles, p.uniprot_id)], dtype=int)
                assert len(set(ix)) == len(ix)
                assert np.array_equal(src.iloc[ix].label, p.true_label)
                assert np.array_equal(src.iloc[ix].chemical_name, p.chemical_name)
                score, saved = p.pred_prob.to_numpy(), p.pred_label.to_numpy()
                if model == 'RF':
                    cp = add(base/f'checkpoints/RF_{block}.joblib', 'formal_checkpoint_hashed_only', block, model)
                    assert rfmeta[block]['RF']['failures']['test'] == {'failed_smiles': 0, 'missing_protein': 0}
                    feature_filter = 'RF archived failures.test: failed_smiles=0, missing_protein=0'
                else:
                    cp = add(base/f'results/models/MLP_{block}_best.pth', 'formal_checkpoint_hashed_only', block, model)
                    add(base/f'results/mlp_metrics_{block}.json', 'archived_result_provenance', block, model)
                    fs = mlpfeatures['effect_'+block]['test']
                    assert fs['raw_size'] == fs['effective_size'] == len(src) and fs['fail_smiles'] == fs['fail_esm2'] == 0
                    feature_filter = 'MLP cache report: raw=effective; fail_smiles=0, fail_esm2=0'
                checkpoint = str(cp)
            assert np.isfinite(score).all() and np.all((score >= 0) & (score <= 1))
            assert np.array_equal(saved, (score >= .5).astype(int))
            prediction_rows[model] = set(ix)
            arr = np.full(len(src), np.nan)
            arr[ix] = score
            frame[model+'_score'] = arr
            hist = prior['absolute'][block][model]['metrics']
            y = src.iloc[ix].label.to_numpy()
            point = calculate(y, np.ones(len(y)), prepared(y, score))
            for k, metric in enumerate(METRICS):
                old = hist['AUPRC' if metric == 'AP' else metric]
                assert abs(point[k]-old) < 1e-12
                histories.append(dict(block=block, model=model, metric=metric, archived_full_precision=old,
                                      recalculated_original_cohort=point[k], absolute_error=abs(point[k]-old),
                                      historical_pr_definition='average precision'))
            audits.append(dict(block=block, model=model, prediction_rows=len(p), duplicate_pair_rows=0,
                               duplicate_record_id_rows=0, label_mismatches=0, unmapped_predictions=0,
                               nonfinite_scores=0, outside_probability_range=0, threshold_mismatches=0,
                               scores_exactly_0_5=int(np.sum(score == .5)), matched_on='source_row_index+name+UniProt+label' if model=='ToxCPI' else 'unique raw SMILES+UniProt, then name+label checked',
                               source_order_identical=bool(np.array_equal(ix, np.arange(len(src))))))
            provenance.append(dict(block=block, model=model, model_version='formal rawglobal v3, reused full seed42 checkpoint' if model=='ToxCPI' else family,
                                   test_file=str(srcpath), prediction_file=str(ppath), checkpoint=checkpoint,
                                   score_meaning='raw sigmoid P(label=1)' if model!='RF' else 'RandomForest predict_proba class 1',
                                   positive_meaning=f'assigned {task} positive; label=0 is fixed existing negative pool',
                                   threshold_rule='score >= 0.5; equality positive', feature_availability=feature_filter))
        common = sorted(set.intersection(*prediction_rows.values()))
        assert len(common) > 0
        frame['in_common'] = frame.source_row_index.isin(common)
        frame['exclusion_reason'] = ['' if i in common else ';'.join('missing_'+m for m in MODELS if i not in prediction_rows[m]) for i in range(len(src))]
        frame.to_csv(out/'record_membership.csv.gz', index=False)
        for model in MODELS:
            denom = len(prediction_rows[model])
            denominators.append(dict(block=block, model=model, source_test_n=len(src), source_positive=int(src.label.sum()),
                source_negative=int((src.label==0).sum()), original_evaluated_n=denom,
                missing_prediction_n=len(src)-denom, historical_feature_filter_n=0, common_n=len(common),
                common_positive=int(src.iloc[common].label.sum()), common_negative=int((src.iloc[common].label==0).sum()),
                excluded_from_model_to_common=denom-len(common), excluded_from_source_to_common=len(src)-len(common),
                exclusion_reason='none' if len(common)==len(src) else 'missing a required model score'))
        df = frame.iloc[common].reset_index(drop=True).drop(columns=['in_common', 'exclusion_reason'])
        df['graph_key'] = [canon(s) for s in df.smiles]
        groups = sorted(df.graph_key.unique())
        group_ids = {s: i for i, s in enumerate(groups)}
        df['group_id'] = df.graph_key.map(group_ids)
        df.to_csv(out/'common_records.csv.gz', index=False)
        gg = df.groupby(['group_id', 'graph_key'], sort=True).agg(
            n_records=('label', 'size'), n_positive=('label', 'sum'),
            n_raw_smiles=('smiles', 'nunique'), n_chemical_names=('chemical_name', 'nunique')).reset_index()
        gg['n_negative'] = gg.n_records-gg.n_positive
        gg.to_csv(out/'chemical_groups.csv.gz', index=False)
        # Audit graph-equivalent record multiplicity; never deduplicate on graph.
        graph_pairs = df.groupby(['graph_key', 'uniprot_id']).agg(n=('label', 'size'), labels=('label', 'nunique')).reset_index()
        graph_pairs.query('n > 1').to_csv(out/'graph_equivalent_pair_multiplicity.csv', index=False)
        # PMF has the same current test cells but no test probabilities saved.
        pmfbase = ROOT/f'PMF-CPI/datasets/toxcpi_effect_{block}/ind_test'
        pmfpaths = {name: add(pmfbase/name, 'PMF_formal_test_matrix_identity_audit_only', block, 'PMF-CPI') for name in ['compounds.txt', 'proteins.txt', 'Y']}
        chems = list(json.loads(pmfpaths['compounds.txt'].read_text()).values())
        prots = list(json.loads(pmfpaths['proteins.txt'].read_text()))
        with pmfpaths['Y'].open('rb') as f:
            mat = pickle.load(f)
        ii, jj = np.where(~np.isnan(mat))
        mc = Counter((chems[i], prots[j], int(mat[i, j])) for i, j in zip(ii, jj))
        sc = Counter(zip(src.smiles, src.uniprot_id, src.label))
        assert mc == sc
        pmfcp = add(ROOT/f'PMF-CPI/classification/toxcpi_effect_{block}/best_esm2_fixed_split.pt', 'checkpoint_exists_no_inference_allowed', block, 'PMF-CPI')
        logsuffix = 'cuda0_20260510_1640' if block == 'inhibition_cold' else 'cuda2_20260510_1632'
        pmflog = add(ROOT/f'PMF-CPI/classification/toxcpi_effect_logs/{block}_{logsuffix}.log',
                     'historical_test_log_no_scores', block, 'PMF-CPI')
        gaps.append(dict(block=block, model='PMF-CPI', original_n=len(ii), original_positive=int(sum(mat[i,j] for i,j in zip(ii,jj))),
                         matched_labeled_test_records=len(src), available_test_score_n=0,
                         reason='Formal test matrix matches exactly, but training entry saves only aggregate metrics and best state_dict. No formal test-score export found. CTD five-chemical scan and external direction scores rejected as different cohorts.',
                         checkpoint=str(pmfcp), log=str(pmflog), historical_pr_definition='trapezoidal auc(recall,precision)', historical_threshold='score > 0.5; equality negative'))
        dpath = add(ROOT/f'DTIAM-main/DTIAM-main/data/moa/{task}/data_folds/{split}/test.csv', 'DTIAM_different_test_negative_pool_audit_only', block, 'DTIAM')
        dd = pd.read_csv(dpath)
        dc = Counter(zip(dd.DrugID, dd.TargetID, dd.label))
        for ext in ['csv', 'pkl']:
            add(ROOT/f'DTIAM-main/DTIAM-main/results/{block}_result.{ext}', 'DTIAM_aggregate_metrics_only_not_scores', block, 'DTIAM')
        gaps.append(dict(block=block, model='DTIAM', original_n=len(dd), original_positive=int(dd.label.sum()),
                         matched_labeled_test_records=sum((sc & dc).values()), available_test_score_n=0,
                         reason='Table S8 Part 4 historical separate cohort with different sampled negatives. Result CSV/PKL contain aggregate metrics only; no aligned test probabilities. No inference performed.',
                         checkpoint='AutoGluon saved predictor family; not loaded; no unverified single checkpoint assigned', log='',
                         historical_pr_definition='average precision', historical_threshold='score >= 0.5; equality positive'))
        y = df.label.to_numpy(dtype=int)
        scores = {m: df[m+'_score'].to_numpy() for m in MODELS}
        caches = {m: prepared(y, scores[m]) for m in MODELS}
        point = np.stack([calculate(y, np.ones(len(y)), caches[m]) for m in MODELS])
        G, inv = len(groups), df.group_id.to_numpy()
        rng = np.random.Generator(np.random.PCG64(SEEDS[block]))
        draws = rng.integers(0, G, size=(B, G), dtype=np.int32)
        np.savez_compressed(out/'bootstrap_group_draws.npz', draw_group_id=draws,
                            group_id=np.arange(G), graph_key=np.array(groups), seed=SEEDS[block])
        values = np.full((B, len(MODELS), len(METRICS)), np.nan)
        drawinfo = []
        for b, draw in enumerate(draws):
            w = np.bincount(draw, minlength=G)[inv]
            pos, neg = int(w @ y), int(w @ (1-y))
            values[b] = np.stack([calculate(y, w, caches[m]) for m in MODELS])
            drawinfo.append(dict(draw_id=b, weighted_records=int(w.sum()), positive=pos, negative=neg,
                                 single_class=pos==0 or neg==0, unique_sampled_groups=len(set(draw))))
        delta = values[:, 0:1, :]-values
        np.savez_compressed(out/'bootstrap_metrics_and_deltas.npz', metrics=values, delta_tox_minus_model=delta,
                            model=np.array(MODELS), metric=np.array(METRICS), point=point)
        pd.DataFrame(drawinfo).to_csv(out/'bootstrap_draw_status.csv.gz', index=False)
        longrows=[]
        for im, model in enumerate(MODELS):
            for ik, metric in enumerate(METRICS):
                lo, hi = quant(values[:, im, ik])
                dl, dh = quant(delta[:, im, ik])
                dpoint = point[0, ik]-point[im, ik]
                summary.append(dict(block=block, model=model, metric=metric, common_n=len(y), chemical_groups=G,
                    estimate=point[im, ik], ci95_low=lo, ci95_high=hi, delta_tox_minus_model=dpoint,
                    delta_ci95_low=dl, delta_ci95_high=dh, delta_ci_includes_zero=bool(dl <= 0 <= dh),
                    conclusion='reference' if model=='ToxCPI' else ('ToxCPI_higher' if dl>0 else ('baseline_higher' if dh<0 else 'includes_zero')),
                    requested_draws=B, valid_metric_draws=int(np.isfinite(values[:,im,ik]).sum()),
                    valid_delta_draws=int(np.isfinite(delta[:,im,ik]).sum()), multiple_comparison_adjustment='none', seed=SEEDS[block]))
                for b in range(B):
                    longrows.append(dict(draw_id=b, model=model, metric=metric, value=values[b,im,ik],
                        delta_tox_minus_model=delta[b,im,ik], valid_metric=bool(np.isfinite(values[b,im,ik])),
                        valid_delta=bool(np.isfinite(delta[b,im,ik]))))
        pd.DataFrame(longrows).to_csv(out/'bootstrap_replicates.csv.gz', index=False)
        block_status.append(dict(block=block, status='partial', completed_models=','.join(MODELS),
            missing_models='PMF-CPI;DTIAM (historical separate cohort)', source_n=len(src), common_n=len(y),
            positive=int(y.sum()), negative=int((1-y).sum()), chemical_groups=G,
            raw_pair_duplicate_rows=int(src.duplicated(['smiles','uniprot_id']).sum()),
            graph_equivalent_duplicate_pair_keys=int((graph_pairs.n>1).sum()),
            graph_equivalent_conflicting_label_keys=int((graph_pairs.labels>1).sum()),
            atom_map_records=int(sum(any(a.GetAtomMapNum() for a in Chem.MolFromSmiles(s).GetAtoms()) for s in src.smiles)),
            single_class_draws=sum(x['single_class'] for x in drawinfo)))
        print(block, 'PARTIAL; completed', MODELS, 'N', len(y), 'G', G, flush=True)
    for name, rows in [('metrics_and_paired_differences.csv', summary), ('cohort_denominators.csv', denominators),
                       ('alignment_audit.csv', audits), ('historical_metric_reconciliation.csv', histories),
                       ('model_prediction_provenance.csv', provenance), ('missing_assets.csv', gaps), ('block_status.csv', block_status)]:
        pd.DataFrame(rows).to_csv(OUT/name, index=False)
    for p, entry in assets.items():
        entry['sha256_after'] = sha(p) if p.is_file() else ''
        entry['unchanged'] = entry['sha256_before'] == entry['sha256_after']
        assert entry['unchanged']
    pd.DataFrame(assets.values()).to_csv(OUT/'input_assets_sha256.csv', index=False)
    print('All input and current-document hashes unchanged; saved',len(assets),'asset records.', flush=True)


if __name__ == '__main__':
    main()
