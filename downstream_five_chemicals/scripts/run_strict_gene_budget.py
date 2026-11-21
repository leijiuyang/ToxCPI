#!/usr/bin/env python3
"""Fixed-score strict equal-gene-budget analysis. Writes only to --outdir.

First freeze the plan: analyze.py --stage plan
Then execute it:      analyze.py --stage run
An existing plan is verified, never silently replaced.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / 'runtime/strict_gene_budget'
CHEM = {'C006780': 'BPA', 'D013749': 'TCDD', 'D001564': 'BaP',
        'D004390': 'chlorpyrifos', 'D003634': 'DDT'}
DENOM = dict(zip(CHEM, [300, 299, 297, 249, 160]))
KS = [100, 300, 500, 1000]
FULL, PMF, TIP, EFF = ('ToxCPI_full', 'PMF_CPI_effect_union_strictK',
                        'tipFormer_binding', 'ToxCPI_effect_union_strictK')
VIEWS = [FULL, PMF, TIP, EFF]
OLD = {FULL: 'M4_toxcpi_full', TIP: 'M1_tipformer_binding', EFF: 'M3_toxcpi_effect'}
COMPARISONS = [('full_minus_PMF', FULL, PMF), ('full_minus_tipFormer', FULL, TIP),
               ('effect_minus_PMF', EFF, PMF)]
METRICS = ['CTD_support', 'CTD_recovery', 'Jaccard']
INPUTS = {}


def configure_inputs(directory):
    global INPUTS
    manifest = json.loads((directory/'inputs.json').read_text())
    required = {'toxcpi_tipformer_scores', 'pmf_protein_scores', 'background', 'gmt', 'ctd_reference', 'ctd_summary', 'old_targets', 'old_metrics'}
    if set(manifest) != required:
        raise ValueError('Expected exact frozen strict-budget input roles')
    INPUTS = {role: directory/item['filename'] for role,item in manifest.items()}
    for role,path in INPUTS.items():
        if path.parent.resolve() != directory.resolve() or sha(path) != manifest[role]['sha256']:
            raise ValueError('Strict-budget input identity mismatch: '+role)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv(df, path):
    df.to_csv(path, index=False, na_rep='NA', float_format='%.17g')


def strict_bool(series):
    """Reject unknown tokens; never use bool('False')."""
    tokens = series.astype(str).str.strip().str.lower()
    assert tokens.isin(['true', 'false', '1', '0']).all(), tokens[~tokens.isin(['true', 'false', '1', '0'])].unique()
    return tokens.map({'true': True, 'false': False, '1': True, '0': False})


def input_manifest():
    return [{'role': k, 'path': str(v.resolve()), 'bytes': v.stat().st_size, 'sha256': sha(v)} for k, v in INPUTS.items()]


def protected_manifest():
    # Candidate adapter operates only on supplied frozen inputs and --outdir.
    return []


def environment():
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'statsmodels']}}


def freeze_plan(out):
    path = out / 'analysis_plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        assert plan['inputs'] == input_manifest(), 'Inputs changed since plan freeze'
        return plan
    assert not (out / 'metrics_80.csv').exists(), 'Existing analysis output requires an explicit separate directory'
    plan = {
        'created_utc': stamp(), 'analysis_label': 'strict equal-gene-budget effect-union ranking',
        'output_directory': str(out), 'chemicals': CHEM, 'K': KS, 'model_views': VIEWS,
        'primary_views': VIEWS[:3], 'auxiliary_view': EFF, 'comparisons': COMPARISONS,
        'background': {'type': 'B_common', 'expected_unique_genes': 8383, 'rank_within_background': True},
        'aggregation': 'trim existing gene_symbol, exclude blank/nan; per chemical/gene/task maximum over UniProt; reuse aggregated ToxCPI/tipFormer input',
        'percentile': '(N - rank_average_descending + 1) / N; N=8383; method=average',
        'fusion': {FULL: 'max(binding,activation,inhibition percentiles)', PMF: 'max(PMF activation,inhibition percentiles)',
                   EFF: 'max(ToxCPI activation,inhibition percentiles)', TIP: 'original binding score'},
        'selection': 'descending fusion/score, ascending gene_symbol; mergesort; first K unique genes; no threshold or direction quotas',
        'missing_policy': 'halt and report exact nonfinite/missing common-background items; no imputation or background reduction',
        'ora': {'test': 'one-sided hypergeometric sf(overlap-1)', 'pathway_size_rule': 'fixed GMT intersection with background, retain every nonempty pathway (same historical common.py); no additional size filter',
                'expected_test_family': 369, 'BH_family': 'all 369 per chemical/view/K; zero overlap p=1',
                'significant': 'FDR < 0.05 and overlap_count >= 2; all significant pathways used'},
        'ctd': {'filter': 'background_type=B_common AND strictly parsed in_background_comparable=true; unique chemical/pathway', 'expected_denominators': DENOM},
        'metrics': {'CTD_support': '|P intersection C| / |P|; NA if P empty',
                    'CTD_recovery': '|P intersection C| / |C|', 'Jaccard': '|P intersection C| / |P union C|'},
        'summaries': 'all 20 descriptive cells; by K (5), chemical (4), overall (20); paired complete-case mean with n_valid; high/same/low tolerance 1e-12; undefined counted explicitly',
        'inference': 'no statistical significance tests, bootstrap, permutations, training or new model inference',
        'reproduction': {'sets': OLD, 'metric_absolute_tolerance': 1e-12, 'no_rule_tuning': True},
        'input_interface': 'explicit frozen manifest; source scientific functions unchanged',
        'limitations': ['gene budget does not fix significant pathway count', 'full ToxCPI has three tasks; PMF effect view has two',
                        'CTD used for training annotations and knowledge comparison; not independent biological validation', 'descriptive fixed-score analysis; K cells are dependent'],
        'environment': environment(), 'inputs': input_manifest(), 'protected_artifacts_before': protected_manifest(),
        'implementation_sha256_at_freeze': sha(__file__),
    }
    write_json(plan, path)
    csv(pd.DataFrame(plan['inputs']), out / 'input_manifest.csv')
    print('PLAN FROZEN before any new ORA or CTD metric computation', flush=True)
    return plan


def read_gmt(path):
    pathways = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        parts = line.split('\t')
        if len(parts) >= 3:
            genes = {s.strip() for s in parts[2:] if s.strip()}
            if genes:
                pathways[parts[0]] = {'name': parts[1], 'genes': genes}
    return pathways


def bh(p):
    order = np.argsort(p, kind='mergesort')
    result = np.empty(len(p)); prev = 1.0
    for i in range(len(p)-1, -1, -1):
        prev = min(prev, p[order[i]] * len(p)/(i+1), 1.0)
        result[order[i]] = prev
    np.testing.assert_allclose(result, multipletests(p, method='fdr_bh')[1], atol=1e-14, rtol=1e-14)
    return result


def build_scores(out, background):
    tox = pd.read_csv(INPUTS['toxcpi_tipformer_scores'])
    pmf = pd.read_csv(INPUTS['pmf_protein_scores'])
    audits, missing, merged = [], [], []
    assert set(tox.ctd_chemical_id) == set(CHEM) == set(pmf.ctd_chemical_id)
    for cid, name in CHEM.items():
        t = tox.loc[tox.ctd_chemical_id.eq(cid)].copy()
        p = pmf.loc[pmf.ctd_chemical_id.eq(cid)].copy()
        raw_t, raw_p = len(t), len(p)
        for df in [t, p]:
            df['gene_symbol'] = df.gene_symbol.fillna('').astype(str).str.strip()
            df.drop(df.index[df.gene_symbol.eq('') | df.gene_symbol.str.lower().eq('nan')], inplace=True)
        assert not t.gene_symbol.duplicated().any(), 'Aggregated source must have one row per chemical/gene'
        assert (strict_bool(t.in_common_background) == t.gene_symbol.isin(background)).all()
        agg = p.groupby('gene_symbol', as_index=False).agg(
            P_activation_pmf=('P_activation_pmf', 'max'), P_inhibition_pmf=('P_inhibition_pmf', 'max'),
            n_uniprot_pmf=('uniprot_id', 'nunique'))
        agg.insert(0, 'ctd_chemical_id', cid); agg.insert(1, 'chemical_name', name)
        merged.append(agg)
        audits.append({'ctd_chemical_id': cid, 'chemical_name': name, 'pmf_rows_before_clean': raw_p,
            'pmf_rows_after_clean': len(p), 'pmf_unique_uniprot': int(p.uniprot_id.nunique()),
            'pmf_unique_genes_after_aggregation': len(agg), 'pmf_multi_protein_genes': int(agg.n_uniprot_pmf.gt(1).sum()),
            'pmf_multi_protein_genes_common_background': int(agg.loc[agg.gene_symbol.isin(background), 'n_uniprot_pmf'].gt(1).sum()),
            'pmf_missing_raw_activation': int(p.P_activation_pmf.isna().sum()), 'pmf_missing_raw_inhibition': int(p.P_inhibition_pmf.isna().sum()),
            'toxcpi_aggregated_rows_before_clean': raw_t, 'toxcpi_aggregated_rows_after_clean': len(t),
            'toxcpi_protein_rows_reconstructed_from_mapping_counts': int(t.n_uniprot_toxcpi.sum()),
            'toxcpi_multi_protein_genes': int(t.n_uniprot_toxcpi.gt(1).sum()),
            'tipformer_protein_rows_reconstructed_from_mapping_counts': int(t.n_uniprot_tipformer.sum()),
            'tipformer_multi_protein_genes': int(t.n_uniprot_tipformer.gt(1).sum())})
    pmf_agg = pd.concat(merged, ignore_index=True)
    csv(pmf_agg, out / 'pmf_gene_scores_aggregated_all_genes.csv.gz')
    csv(pd.DataFrame(audits), out / 'aggregation_audit.csv')
    joined = []
    score_cols = ['P_binding', 'P_activation', 'P_inhibition', 'tipformer_binding_score', 'P_activation_pmf', 'P_inhibition_pmf']
    for cid, name in CHEM.items():
        t = tox.loc[tox.ctd_chemical_id.eq(cid)].copy()
        t['gene_symbol'] = t.gene_symbol.fillna('').str.strip()
        p = pmf_agg.loc[pmf_agg.ctd_chemical_id.eq(cid)].drop(columns=['ctd_chemical_id', 'chemical_name'])
        d = pd.DataFrame({'gene_symbol': sorted(background)}).merge(t, on='gene_symbol', how='left', validate='one_to_one').merge(p, on='gene_symbol', how='left', validate='one_to_one')
        for col in score_cols:
            bad = ~np.isfinite(pd.to_numeric(d[col], errors='coerce'))
            for gene in d.loc[bad, 'gene_symbol']:
                missing.append({'ctd_chemical_id': cid, 'chemical_name': name, 'gene_symbol': gene, 'task': col})
        if missing:
            continue
        d['ctd_chemical_id'] = cid; d['chemical_name'] = name
        for col in ['P_binding', 'P_activation', 'P_inhibition', 'P_activation_pmf', 'P_inhibition_pmf']:
            suffix = col[2:]
            d['rank_average_' + suffix] = d[col].rank(method='average', ascending=False)
            d['percentile_' + suffix] = (len(background) - d['rank_average_' + suffix] + 1)/len(background)
        d[FULL + '_score'] = d[['percentile_binding', 'percentile_activation', 'percentile_inhibition']].max(axis=1)
        d[EFF + '_score'] = d[['percentile_activation', 'percentile_inhibition']].max(axis=1)
        d[PMF + '_score'] = d[['percentile_activation_pmf', 'percentile_inhibition_pmf']].max(axis=1)
        d[TIP + '_score'] = d.tipformer_binding_score
        for view in VIEWS:
            order = d.sort_values([view+'_score', 'gene_symbol'], ascending=[False, True], kind='mergesort').index
            d.loc[order, view+'_rank'] = np.arange(1, len(d)+1)
            d[view+'_rank'] = d[view+'_rank'].astype(int)
        joined.append(d)
    csv(pd.DataFrame(missing, columns=['ctd_chemical_id', 'chemical_name', 'gene_symbol', 'task']), out/'missing_common_background_scores.csv')
    assert not missing, f'Nonfinite or missing common background scores: see missing_common_background_scores.csv ({len(missing)} items)'
    scores = pd.concat(joined, ignore_index=True)
    csv(scores, out / 'common_background_scores_and_ranks.csv.gz')
    provenance = {'activation_checkpoint_sha256': sorted(pmf.activation_checkpoint_sha256.unique().tolist()),
                  'inhibition_checkpoint_sha256': sorted(pmf.inhibition_checkpoint_sha256.unique().tolist()),
                  'model_variant': sorted(pmf.model_variant.unique().tolist()),
                  'note': 'Provenance copied from fixed score file; no checkpoints loaded or inference run.'}
    write_json(provenance, out / 'pmf_fixed_score_provenance.json')
    return scores


def run(out, plan):
    assert plan['inputs'] == input_manifest()
    bg_lines = INPUTS['background'].read_text().splitlines()
    assert len(bg_lines) == len(set(bg_lines)) == 8383
    background = set(bg_lines)
    gmt = read_gmt(INPUTS['gmt'])
    pathways = {pid: {'name': v['name'], 'genes': v['genes'] & background} for pid, v in gmt.items() if v['genes'] & background}
    assert len(pathways) == 369
    scores = build_scores(out, background)
    old_targets = pd.read_csv(INPUTS['old_targets'])
    ctd = pd.read_csv(INPUTS['ctd_reference'])
    refmask = ctd.background_type.eq('B_common') & strict_bool(ctd.in_background_comparable)
    selected_ref = ctd.loc[refmask].drop_duplicates(['ctd_chemical_id', 'pathway_id'])
    assert strict_bool(selected_ref.in_gmt_comparable).all()
    refs = {cid: set(selected_ref.loc[selected_ref.ctd_chemical_id.eq(cid), 'pathway_id']) for cid in CHEM}
    ctd_summary = pd.read_csv(INPUTS['ctd_summary'])
    for cid in CHEM:
        assert len(refs[cid]) == DENOM[cid] and refs[cid] <= set(pathways)
        s = ctd_summary.loc[ctd_summary.ctd_chemical_id.eq(cid) & ctd_summary.background_type.eq('B_common')]
        assert len(s) == 1 and int(s.iloc[0].n_ctd_background_comparable) == DENOM[cid]
    csv(selected_ref, out / 'ctd_reference_used.csv.gz')
    targets, oras, rows, set_checks = [], [], [], []
    for cid, name in CHEM.items():
        d = scores.loc[scores.ctd_chemical_id.eq(cid)]
        for view in VIEWS:
            previous = set()
            for k in KS:
                z = d.loc[d[view+'_rank'].le(k)].sort_values(view+'_rank', kind='mergesort').copy()
                genes = set(z.gene_symbol)
                assert len(z) == len(genes) == k and previous < genes
                previous = genes
                z['model_view'] = view; z['K'] = k; z['final_rank'] = z[view+'_rank']; z['selection_score'] = z[view+'_score']
                z['background_type'] = 'B_common'; z['N_background'] = len(background)
                targets.append(z)
                if view in OLD:
                    old = old_targets.loc[old_targets.ctd_chemical_id.eq(cid) & old_targets.model_view.eq(OLD[view]) & old_targets.background_type.eq('B_common') & old_targets.protocol.eq('topk') & old_targets.cutoff.eq(k)]
                    old_genes = set(old.gene_symbol)
                    set_checks.append({'ctd_chemical_id': cid, 'chemical_name': name, 'model_view': view, 'old_model_view': OLD[view], 'K': k,
                        'new_gene_count': len(genes), 'old_gene_count': len(old_genes), 'only_new': ';'.join(sorted(genes-old_genes)),
                        'only_old': ';'.join(sorted(old_genes-genes)), 'identical': genes == old_genes,
                        'ordered_genes_identical': z.gene_symbol.tolist() == old.gene_symbol.tolist()})
                test_rows = []
                for pid, info in pathways.items():
                    ov = sorted(genes & info['genes']); overlap = len(ov)
                    test_rows.append({'pathway_id': pid, 'pathway_name': info['name'], 'pathway_size': len(info['genes']),
                        'overlap_count': overlap, 'overlap_genes': ';'.join(ov),
                        'p_value': float(hypergeom.sf(overlap-1, len(background), len(info['genes']), k)) if overlap else 1.0})
                a = pd.DataFrame(test_rows)
                a['FDR'] = bh(a.p_value.to_numpy())
                a['included_in_BH_family'] = True
                a['is_significant'] = a.FDR.lt(.05) & a.overlap_count.ge(2)
                a['in_CTD_reference'] = a.pathway_id.isin(refs[cid])
                for col, val in [('ctd_chemical_id', cid), ('chemical_name', name), ('model_view', view), ('K', k), ('N_background', len(background)), ('n_selected_genes', k)]:
                    a[col] = val
                oras.append(a)
                pred = set(a.loc[a.is_significant, 'pathway_id']); c = refs[cid]; overlap = len(pred & c)
                rows.append({'ctd_chemical_id': cid, 'chemical_name': name, 'model_view': view, 'K': k,
                    'background_type': 'B_common', 'N_background': len(background), 'n_selected_genes': len(genes),
                    'n_tested_pathways': len(a), 'n_predicted_pathways': len(pred), 'n_ctd_reference': len(c),
                    'n_overlap': overlap, 'n_model_only': len(pred-c), 'n_ctd_only': len(c-pred),
                    'CTD_support': overlap/len(pred) if pred else np.nan,
                    'CTD_recovery': overlap/len(c), 'Jaccard': overlap/len(pred|c)})
        print('Completed all 16 strict-K groups:', name, flush=True)
    targets = pd.concat(targets, ignore_index=True); oras = pd.concat(oras, ignore_index=True); metrics = pd.DataFrame(rows)
    assert len(targets) == 38000 and len(oras) == 80*369 and len(metrics) == 80
    assert oras.groupby(['ctd_chemical_id', 'model_view', 'K']).size().eq(369).all()
    assert oras.loc[oras.overlap_count.eq(0), 'p_value'].eq(1).all()
    csv(targets, out/'strictK_gene_sets.csv.gz'); csv(oras, out/'ORA_all_369_pathways.csv.gz')
    csv(metrics, out/'metrics_80.csv'); csv(metrics.loc[metrics.model_view.isin(VIEWS[:3])], out/'metrics_primary_60.csv')
    csv(metrics.loc[metrics.model_view.eq(EFF)], out/'metrics_auxiliary_20.csv')
    set_checks = pd.DataFrame(set_checks); csv(set_checks, out/'reproduction_gene_sets.csv')
    old_metrics = pd.read_csv(INPUTS['old_metrics'])
    metric_checks = []
    fields = {'N_background': 'N_background', 'n_selected_genes': 'n_selected_genes',
              'n_predicted_pathways': 'n_predicted_pathways', 'n_ctd_reference': 'n_ctd_comparable',
              'n_overlap': 'n_overlap', 'n_model_only': 'n_model_only', 'n_ctd_only': 'n_ctd_only',
              'CTD_support': 'CTD_support_fraction', 'CTD_recovery': 'CTD_recovery_fraction', 'Jaccard': 'Jaccard'}
    for r in metrics.loc[metrics.model_view.isin(OLD)].itertuples():
        old = old_metrics.loc[old_metrics.ctd_chemical_id.eq(r.ctd_chemical_id) & old_metrics.model_view.eq(OLD[r.model_view]) & old_metrics.cutoff.eq(r.K) & old_metrics.background_type.eq('B_common') & old_metrics.significance_level.eq('fdr_overlap2')]
        assert len(old) == 1
        o = old.iloc[0]; check = {'ctd_chemical_id': r.ctd_chemical_id, 'chemical_name': r.chemical_name, 'model_view': r.model_view, 'K': r.K}
        passes = []
        for new, historical in fields.items():
            a, b = getattr(r, new), o[historical]
            ok = bool((pd.isna(a) and pd.isna(b)) or (pd.notna(a) and pd.notna(b) and abs(a-b) <= 1e-12))
            check['new_'+new] = a; check['old_'+new] = b; check['delta_'+new] = a-b; check['pass_'+new] = ok; passes.append(ok)
        check['all_metrics_match'] = all(passes); metric_checks.append(check)
    metric_checks = pd.DataFrame(metric_checks); csv(metric_checks, out/'reproduction_metrics.csv')
    paired = []
    for label, left, right in COMPARISONS:
        for cid, name in CHEM.items():
            for k in KS:
                a = metrics.loc[metrics.ctd_chemical_id.eq(cid) & metrics.model_view.eq(left) & metrics.K.eq(k)].iloc[0]
                b = metrics.loc[metrics.ctd_chemical_id.eq(cid) & metrics.model_view.eq(right) & metrics.K.eq(k)].iloc[0]
                pair = {'comparison': label, 'left_view': left, 'right_view': right, 'ctd_chemical_id': cid, 'chemical_name': name, 'K': k}
                for metric in METRICS + ['n_predicted_pathways', 'n_overlap']:
                    pair['left_'+metric] = a[metric]; pair['right_'+metric] = b[metric]; pair['delta_'+metric] = a[metric]-b[metric]
                    if metric in METRICS:
                        pair['delta_'+metric+'_percentage_points'] = 100*(a[metric]-b[metric])
                paired.append(pair)
    paired = pd.DataFrame(paired); csv(paired, out/'paired_differences_60.csv')
    summaries = []
    for label, group in paired.groupby('comparison', sort=False):
        partitions = [('overall', 'all_20', group)]
        partitions += [('K', str(k), g) for k, g in group.groupby('K', sort=True)]
        partitions += [('chemical', CHEM[cid], g) for cid, g in group.groupby('ctd_chemical_id', sort=False)]
        for level, value, g in partitions:
            for metric in METRICS:
                delta = g['delta_'+metric]; valid = delta.dropna()
                summaries.append({'comparison': label, 'stratum_type': level, 'stratum': value, 'metric': metric,
                    'n_cells': len(g), 'left_n_valid': int(g['left_'+metric].notna().sum()), 'right_n_valid': int(g['right_'+metric].notna().sum()),
                    'paired_n_valid': len(valid), 'n_undefined_pairs': int(delta.isna().sum()),
                    'left_mean': g['left_'+metric].mean(), 'right_mean': g['right_'+metric].mean(),
                    'mean_paired_delta': valid.mean(), 'mean_delta_percentage_points': 100*valid.mean(),
                    'higher': int(valid.gt(1e-12).sum()), 'same': int(valid.abs().le(1e-12).sum()), 'lower': int(valid.lt(-1e-12).sum())})
    summary = pd.DataFrame(summaries); csv(summary, out/'comparison_summaries.csv')
    for level in ['overall', 'K', 'chemical']:
        csv(summary.loc[summary.stratum_type.eq(level)], out/f'comparison_summary_{level}.csv')
    after_inputs = input_manifest(); protected_after = protected_manifest()
    csv(pd.DataFrame([{'role': r['role'], 'path': r['path'], 'sha256_before': r['sha256'], 'sha256_after': a['sha256'], 'unchanged': r['sha256'] == a['sha256']} for r, a in zip(plan['inputs'], after_inputs)]), out/'input_hash_before_after.csv')
    write_json(protected_after, out/'protected_artifacts_after.json')
    verification = {
        'completed_utc': stamp(), 'status': 'PASS' if set_checks.identical.all() and metric_checks.all_metrics_match.all() else 'REPRODUCTION_DIFFERENCE',
        'environment': environment(), 'analysis_implementation_matches_frozen_hash': sha(__file__) == plan['implementation_sha256_at_freeze'],
        'five_chemicals': list(CHEM), 'K': KS, 'model_views': VIEWS, 'N_background': len(background),
        'pathways_in_GMT': len(gmt), 'testable_pathways': len(pathways), 'CTD_denominators': {cid: len(refs[cid]) for cid in CHEM},
        'missing_common_background_score_items': 0, 'strict_K_groups': 80, 'all_groups_exactly_K_unique_genes': True,
        'all_nested_sets_verified': True, 'metric_rows': len(metrics), 'primary_rows': 60, 'auxiliary_rows': 20,
        'target_set_rows': len(targets), 'ORA_rows': len(oras), 'BH_independent_statsmodels_check': 'all 80 groups passed',
        'zero_overlap_p_equals_one': True, 'reproduction_set_groups': len(set_checks),
        'reproduction_set_passes': int(set_checks.identical.sum()), 'reproduction_order_passes': int(set_checks.ordered_genes_identical.sum()),
        'reproduction_metric_groups': len(metric_checks), 'reproduction_metric_passes': int(metric_checks.all_metrics_match.sum()),
        'reproduction_max_abs_metric_difference': float(metric_checks.filter(regex='^delta_').abs().max().max()),
        'support_NA_groups': int(metrics.CTD_support.isna().sum()), 'all_inputs_hash_unchanged': plan['inputs'] == after_inputs,
        'protected_artifact_count': len(protected_after), 'protected_artifact_paths_and_hashes_unchanged': plan['protected_artifacts_before'] == protected_after,
        'no_training_no_inference_no_downloads': True, 'historical_scripts_executed': False,
        'known_old_tipFormer_comparison_check': summary.loc[summary.comparison.eq('full_minus_tipFormer') & summary.stratum_type.eq('overall')].to_dict('records'),
        'document_verification': 'pending; see document_verification.json after build_deliverables.py',
    }
    write_json(verification, out/'完整核查结果.json')
    assert verification['all_inputs_hash_unchanged'] and verification['protected_artifact_paths_and_hashes_unchanged']
    assert verification['status'] == 'PASS', 'Specific reproduction differences saved; do not represent as verified'
    print(summary.loc[summary.stratum_type.eq('overall')].to_string(index=False), flush=True)
    print('ANALYSIS PASS. All 60 historical gene sets and metric groups reproduced.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['plan', 'run'], required=True)
    args = parser.parse_args(); out = args.outdir.resolve(); out.mkdir(parents=True, exist_ok=True)
    configure_inputs(args.input_dir.resolve())
    plan = freeze_plan(out)
    if args.stage == 'run':
        run(out, plan)
