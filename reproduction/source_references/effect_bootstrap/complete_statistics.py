#!/usr/bin/env python3
"""Append PMF-CPI statistics to frozen three-model results, reusing all draws.

Uses toxcpi analysis environment. No inference and no repeated old-model metrics.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
OLD=OUT.parent
BLOCKS=['activation_warm','activation_cold','inhibition_warm','inhibition_cold']
MODELS=['ToxCPI','PMF-CPI','RF','MLP']
METRICS=['AUROC','AP','Accuracy','F1','MCC']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(4194304),b''):h.update(c)
    return h.hexdigest()


def js(name,o):
    (OUT/name).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def metric_cache(y,s):
    order=np.argsort(-s,kind='stable')
    end=np.r_[np.flatnonzero(np.diff(s[order])),len(s)-1]
    return order,end,y[order],s>.5


def weighted(y,w,c):
    order,end,yy,pred=c
    sw=w[order]
    tp=np.cumsum(sw*yy)[end];fp=np.cumsum(sw*(1-yy))[end]
    P,N=tp[-1],fp[-1]
    AUC=AP=np.nan
    if P>0 and N>0:
        AUC=np.trapz(np.r_[0,tp],np.r_[0,fp])/(P*N)
        AP=np.sum(np.diff(np.r_[0,tp])*tp/np.maximum(tp+fp,1))/P
    TP=np.sum(w*y*pred);FP=np.sum(w*(1-y)*pred)
    TN,FN=N-FP,P-TP
    fd=2*TP+FP+FN;md=(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    return np.array([AUC,AP,(TP+TN)/(P+N),2*TP/fd if fd else np.nan,
                     (TP*TN-FP*FN)/np.sqrt(md) if md else np.nan])


def bounds(x):
    x=x[np.isfinite(x)]
    return np.quantile(x,[.025,.975],method='linear') if len(x) else [np.nan,np.nan]


def main():
    assert not (OUT/'metrics_and_paired_differences.csv').exists(),'Refuse overwrite of statistical delivery'
    for b in BLOCKS:assert (OUT/b/'export_complete.json').exists(),b
    original_design=json.loads((OLD/'design.json').read_text())
    js('statistical_design.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        authorized_scope='PMF-CPI fixed checkpoint export plus paired bootstrap; DTIAM excluded by user',
        blocks=BLOCKS,models=MODELS,metrics=METRICS,replicates=2000,
        seeds=original_design['seeds'],group_rule=original_design['group_key'],
        molecular_policy=original_design['molecular_policy'],
        sample_indices='byte-identical copies of previous bootstrap_group_draws.npz; no additional draws',
        original_models='previous 3-model point/replicate arrays reused bit-for-bit; no re-inference or repeated bootstrap calculations',
        threshold={'ToxCPI':'>=0.5','RF':'>=0.5','MLP':'>=0.5','PMF-CPI':'>0.5'},
        threshold_equality={'PMF-CPI':'negative','others':'positive'},
        AP='average precision for all models; PMF historical trapezoidal PR area reconciled separately',
        ci='finite-replicate 2.5/97.5 percentile, numpy.quantile(method=linear)',
        invalid_draws=original_design['invalid_draws'],delta='ToxCPI minus comparator within same draw',
        multiplicity='pointwise unadjusted intervals; no p values',
        historical_replay='Retain every rounded-metric discrepancy. Newly authorized CPU export of original fixed weights is the PMF score source; do not substitute or tune scores to match historical rounded logs.',
        scope='fixed checkpoints/splits/labels; test chemical-group uncertainty'))
    summary=[];status=[];denoms=[];hashes=[];history=[];mapping=[]
    def record(p,role,block='all'):
        p=Path(p)
        hashes.append(dict(path=str(p),role=role,block=block,bytes=p.stat().st_size,sha256_before=sha(p)))
    for f in ['design.json','metrics_and_paired_differences.csv','independent_verification.json',
              'cohort_denominators.csv','model_prediction_provenance.csv','current_table_s8_reported_rows.csv']:
        record(OLD/f,'previous_frozen_analysis')
    doc=Path('/home/data2/leijiuyang/Paper_revise/返修工作区/15_实验结果回写/本次回复调整工作区')
    for p in doc.iterdir():
        if p.is_file():record(p,'protected_current_document')
    for block in BLOCKS:
        folder=OUT/block
        for f in ['common_records.csv.gz','bootstrap_metrics_and_deltas.npz','bootstrap_group_draws.npz',
                  'bootstrap_draw_status.csv.gz','chemical_groups.csv.gz']:
            record(OLD/block/f,'previous_frozen_labels_scores_or_draws',block)
        for f in ['export_complete.json','predictions.csv','native_matrix_order_logits_probabilities.npz',
                  'matrix_to_official_record_map.csv.gz','input_assets_sha256.csv','protein_feature_identity_and_hashes.csv.gz']:
            record(folder/f,'new_fixed_checkpoint_export_or_identity_audit',block)
        exp=json.loads((folder/'export_complete.json').read_text())
        # This is an authorized new export of the original fixed checkpoint,
        # not a claim to recover nonexistent historical score bytes. Preserve
        # and report rounded replay differences, without tuning scores/batches.
        assert exp['all_source_hashes_unchanged']
        common=pd.read_csv(OLD/block/'common_records.csv.gz',float_precision='round_trip')
        native=np.load(folder/'native_matrix_order_logits_probabilities.npz')
        n=len(common);indices=native['source_row_index']
        assert np.array_equal(np.sort(indices),np.arange(n))
        assert np.array_equal(common.iloc[indices].label.to_numpy(),native['label'])
        s=np.empty(n,dtype=np.float64)
        s[indices]=native['probability'].astype(np.float64)
        pred=pd.read_csv(folder/'predictions.csv',float_precision='round_trip')
        assert np.array_equal(pred.record_id,common.record_id)
        for col in ['smiles','uniprot_id','label','source_row_index']:
            assert np.array_equal(pred[col],common[col])
        assert np.allclose(pred.probability,s,rtol=0,atol=6e-8)
        assert np.array_equal(pred.prediction,(s>.5).astype(int))
        common['PMF-CPI_score']=s
        common['PMF-CPI_prediction']=(s>.5).astype(int)
        common.to_csv(folder/'common_records.csv.gz',index=False)
        membership=common.copy();membership['in_common']=True;membership['exclusion_reason']=''
        membership.to_csv(folder/'record_membership.csv.gz',index=False)
        for f in ['bootstrap_group_draws.npz','chemical_groups.csv.gz','bootstrap_draw_status.csv.gz',
                  'graph_equivalent_pair_multiplicity.csv']:
            assert not (folder/f).exists()
            shutil.copyfile(OLD/block/f,folder/f)
            assert sha(folder/f)==sha(OLD/block/f)
        z=np.load(folder/'bootstrap_group_draws.npz');draws=z['draw_group_id'];G=len(z['group_id'])
        assert draws.shape==(2000,G)
        assert common.group_id.max()==G-1
        previous=np.load(OLD/block/'bootstrap_metrics_and_deltas.npz')
        assert previous['model'].tolist()==['ToxCPI','RF','MLP']
        assert previous['metric'].tolist()==METRICS
        oldvalues=previous['metrics'];oldpoint=previous['point']
        y=common.label.to_numpy();inv=common.group_id.to_numpy()
        cache=metric_cache(y,s)
        point_pmf=weighted(y,np.ones(n),cache)
        pmf=np.stack([weighted(y,np.bincount(d,minlength=G)[inv],cache) for d in draws])
        values=np.stack([oldvalues[:,0],pmf,oldvalues[:,1],oldvalues[:,2]],axis=1)
        point=np.stack([oldpoint[0],point_pmf,oldpoint[1],oldpoint[2]])
        assert np.array_equal(values[:,[0,2,3]],oldvalues,equal_nan=True)
        assert np.array_equal(point[[0,2,3]],oldpoint,equal_nan=True)
        delta=values[:,0:1,:]-values
        np.savez_compressed(folder/'bootstrap_metrics_and_deltas.npz',metrics=values,
                            delta_tox_minus_model=delta,model=np.array(MODELS),metric=np.array(METRICS),point=point)
        long=[]
        for mi,m in enumerate(MODELS):
            for ki,k in enumerate(METRICS):
                lo,hi=bounds(values[:,mi,ki]);dl,dh=bounds(delta[:,mi,ki])
                summary.append(dict(block=block,model=m,metric=k,common_n=n,chemical_groups=G,
                    estimate=point[mi,ki],ci95_low=lo,ci95_high=hi,
                    delta_tox_minus_model=point[0,ki]-point[mi,ki],delta_ci95_low=dl,delta_ci95_high=dh,
                    delta_ci_includes_zero=bool(dl<=0<=dh),
                    conclusion='reference' if m=='ToxCPI' else ('ToxCPI_higher' if dl>0 else ('baseline_higher' if dh<0 else 'includes_zero')),
                    requested_draws=2000,valid_metric_draws=int(np.isfinite(values[:,mi,ki]).sum()),
                    valid_delta_draws=int(np.isfinite(delta[:,mi,ki]).sum()),multiple_comparison_adjustment='none',
                    seed=int(z['seed']),threshold_rule='score > 0.5' if m=='PMF-CPI' else 'score >= 0.5',
                    score_source='authorized frozen-checkpoint re-export; cpu' if m=='PMF-CPI' else 'previous frozen prediction file',
                    result_status='complete_for_requested_main_baselines'))
                for b in range(2000):
                    long.append(dict(draw_id=b,model=m,metric=k,value=values[b,mi,ki],
                                     delta_tox_minus_model=delta[b,mi,ki],
                                     valid_metric=bool(np.isfinite(values[b,mi,ki])),valid_delta=bool(np.isfinite(delta[b,mi,ki]))))
        pd.DataFrame(long).to_csv(folder/'bootstrap_replicates.csv.gz',index=False)
        for m in MODELS:
            denoms.append(dict(block=block,model=m,source_n=n,original_evaluated_n=n,
                common_n=n,positive=int(y.sum()),negative=int((1-y).sum()),
                missing_prediction_n=0,feature_filter_n=0,excluded_from_common_n=0,
                duplicates_removed_n=0,label_conflicts_n=0,many_to_many_matches_n=0,exclusion_reason='none'))
        for k,v in exp['historical_log_4decimal_reconciliation'].items():
            history.append(dict(block=block,model='PMF-CPI',metric=k,historical_4dp=v['historical'],
                                recovered_4dp=v['recovered'],matches=v['matches']))
        history.append(dict(block=block,model='PMF-CPI',metric='unified_AP_separate_from_historical_PR_area',
                            historical_4dp='not the archived PMF PR-area definition',recovered_4dp=f'{point_pmf[1]:.12f}',matches='not_applicable'))
        status.append(dict(block=block,status='complete',models=','.join(MODELS),source_n=n,
                           common_n=n,positive=int(y.sum()),negative=int((1-y).sum()),chemical_groups=G,
                           PMF_historical_metrics_match_4dp=exp['all_historical_metrics_match'],old_model_replicates_preserved_exactly=True,
                           DTIAM='outside scope by user instruction'))
        mapping.append(dict(block=block,model='PMF-CPI',official_test=str(Path('/home/data2/leijiuyang/1_TOXCPI/leijiuyang/test1/data/interaction_type_1to1_targetaware')/f'effect_{block}_test.csv'),
                            predictions=str(folder/'predictions.csv'),native_logits_and_probabilities=str(folder/'native_matrix_order_logits_probabilities.npz'),
                            checkpoint=str(Path('/home/data2/leijiuyang/1_TOXCPI/leijiuyang/PMF-CPI/classification')/f'toxcpi_effect_{block}/best_esm2_fixed_split.pt'),
                            checkpoint_sha256=exp['checkpoint_sha256'],threshold='raw softmax probability >0.5',
                            original_matrix_order_map=str(folder/'matrix_to_official_record_map.csv.gz')))
        print(block,'complete; PMF point',dict(zip(METRICS,point_pmf)),flush=True)
    pd.DataFrame(summary).to_csv(OUT/'metrics_and_paired_differences.csv',index=False)
    pd.DataFrame(denoms).to_csv(OUT/'cohort_denominators.csv',index=False)
    pd.DataFrame(status).to_csv(OUT/'block_status.csv',index=False)
    pd.DataFrame(history).to_csv(OUT/'PMF_historical_reconciliation.csv',index=False)
    pd.DataFrame(mapping).to_csv(OUT/'PMF_prediction_provenance.csv',index=False)
    for h in hashes:
        h['sha256_after']=sha(h['path']);h['unchanged']=h['sha256_before']==h['sha256_after']
        assert h['unchanged'],h['path']
    pd.DataFrame(hashes).to_csv(OUT/'statistical_input_assets_sha256.csv',index=False)
    print('Main baseline scope complete; prior files unchanged; DTIAM excluded.',flush=True)


if __name__=='__main__':main()
