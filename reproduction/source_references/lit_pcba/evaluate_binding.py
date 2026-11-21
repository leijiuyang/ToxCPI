#!/usr/bin/env python3
"""Within-assay ranking and a shared-G0, fixed-prediction multinomial bootstrap."""
import argparse,hashlib,json,math,time
import numpy as np,pandas as pd
import sys
sys.path.insert(0,'/home/data2/leijiuyang/Paper_revise/返修工作区/experiment/r3_07_decoy_evaluation/scripts')
from common import *
ROOT=__import__('pathlib').Path('/home/data2/leijiuyang/Paper_revise/返修工作区/experiment/r3_07_decoy_evaluation/binding_head_litpcba_retrieval_v1')
METRICS=['AUROC','AP','EF1pct']
class Ranking:
    def __init__(self,labels,scores,groups):
        order=np.argsort(-np.asarray(scores),kind='stable');self.y=np.asarray(labels,dtype=np.float64)[order];s=np.asarray(scores)[order];self.groups=np.asarray(groups,dtype=np.int64)[order]
        self.starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1] if len(s) else np.array([],dtype=int)
    def calculate(self,global_weights=None):
        if not len(self.y):return np.full(3,np.nan)
        w=np.ones(len(self.y)) if global_weights is None else global_weights[self.groups].astype(np.float64)
        total=np.add.reduceat(w,self.starts);p=np.add.reduceat(w*self.y,self.starts);n=total-p
        P=p.sum();Q=n.sum();N=P+Q
        if P<=0 or Q<=0:return np.full(3,np.nan)
        cp=np.cumsum(p);cn=np.cumsum(n);ct=cp+cn
        auc=np.sum(p*(Q-cn+.5*n))/(P*Q)
        positive=p>0;ap=np.sum(p[positive]*cp[positive]/ct[positive])/P
        k=max(1,int(np.ceil(.01*N)));ix=int(np.searchsorted(ct,k,side='left'));prior_n=ct[ix-1] if ix else 0.;prior_p=cp[ix-1] if ix else 0.
        hit=prior_p+(k-prior_n)/total[ix]*p[ix];ef=(hit/k)/(P/N)
        return np.asarray([auc,ap,ef])
def self_test():
    from sklearn.metrics import roc_auc_score,average_precision_score
    y=np.array([1,0,1,0,0,1,0]);s=np.array([.9,.9,.5,.5,.4,.1,.1]);g=np.arange(len(y));r=Ranking(y,s,g)
    actual=r.calculate();assert np.allclose(actual[:2],[roc_auc_score(y,s),average_precision_score(y,s)])
    assert np.isclose(actual[2],(.5)/(3/7)) # one top slot, half of a 1-active/1-control tie
    rng=np.random.default_rng(42)
    for _ in range(100):
        ix=rng.integers(0,len(y),len(y));w=np.bincount(ix,minlength=len(y));v=r.calculate(w)
        if np.unique(y[ix]).size<2:assert np.isnan(v).all();continue
        assert np.allclose(v[:2],[roc_auc_score(y[ix],s[ix]),average_precision_score(y[ix],s[ix])])
    assert np.isnan(Ranking([1],[.2],[0]).calculate()).all()
    assert np.allclose(Ranking(y,np.ones(len(y)),g).calculate(),[.5,y.mean(),1.])
    print('PASS: tied-score EF, AP/AUROC vs sklearn, integer-weight bootstrap, single-class and constant-score cases')
def main(draws=2000):
    c=json.loads((ROOT/'config/frozen_protocol.json').read_text());seed=c['bootstrap']['seed'];assert draws==c['bootstrap']['draws']
    specs=pd.read_csv(ROOT/'config/assay_decisions.csv',keep_default_na=False);specs=specs[specs.decision=='include'];data={};audit=[]
    for s in specs.to_dict('records'):
        path=ROOT/f'predictions/{s["set_id"]}.csv'
        if not path.exists():audit.append({'set_id':s['set_id'],'status':'not_predicted'});continue
        d=pd.read_csv(path,keep_default_na=False);valid=d.inference_status.eq('predicted')&pd.to_numeric(d.raw_logit,errors='coerce').notna();nfailed=int((~valid).sum());d=d[valid].copy();d.raw_logit=d.raw_logit.astype(float)
        assert not d.graph_id.duplicated().any();data[s['set_id']]=(s,d,nfailed);audit.append({'set_id':s['set_id'],'status':'complete' if not nfailed else 'complete_with_explicit_feature_exclusions','predictions_sha256':sha256(path),'feature_exclusions':nfailed})
    global_ids=sorted(set(g for _,d,_ in data.values() for g in d.graph_id));lookup={g:i for i,g in enumerate(global_ids)}
    pd.DataFrame({'group_index':range(len(global_ids)),'graph_id':global_ids}).to_csv(ROOT/'audit/bootstrap_groups.csv.gz',index=False)
    ranks=[];rows=[]
    for sid,(s,original,nfailed) in data.items():
        for view in ['primary_G0_pair_clean','parent_pair_clean_sensitivity']:
            d=original if view=='primary_G0_pair_clean' else original[original.parent_same_task_pair_exposure.eq('')]
            group=np.array([lookup[g] for g in d.graph_id]);r=Ranking(d.label.to_numpy(int),d.raw_logit.to_numpy(float),group);vals=r.calculate();ranks.append(r)
            row={k:s[k] for k in ['set_id','benchmark','target','aid','task','uniprot_id','protein_key','endpoint_tier']};n=len(d);p=int(d.label.sum());q=n-p
            row.update(view=view,n=n,positives=p,controls=q,positive_fraction=p/n if n else np.nan,top1pct_k=max(1,int(math.ceil(.01*n))) if n else 0,chemical_groups=d.graph_id.nunique(),feature_exclusions=nfailed,additional_parent_pair_exclusions=len(original)-len(d),class_status='both_classes' if p and q else 'single_class_or_empty',small_positive_flag=p<20,positive_n_lt5=p<5)
            row.update(dict(zip(METRICS,vals)));rows.append(row)
    point=pd.DataFrame(rows);B=np.full((draws,len(rows),3),np.nan);rng=np.random.default_rng(seed);G=len(global_ids);start=time.monotonic()
    for b in range(draws):
        weights=rng.multinomial(G,np.full(G,1./G))
        for j,r in enumerate(ranks):B[b,j]=r.calculate(weights)
        if (b+1)%100==0:print('bootstrap',b+1,'/',draws,'seconds',round(time.monotonic()-start,1),flush=True)
    for j in range(len(rows)):
        for k,metric in enumerate(METRICS):
            v=B[:,j,k];valid=np.isfinite(v);ci=np.percentile(v[valid],[2.5,97.5]) if valid.any() else [np.nan,np.nan]
            point.loc[j,metric+'_ci_low']=ci[0];point.loc[j,metric+'_ci_high']=ci[1];point.loc[j,metric+'_valid_bootstrap']=int(valid.sum());point.loc[j,metric+'_invalid_bootstrap']=int((~valid).sum())
    point.to_csv(ROOT/'results/per_set_metrics.csv',index=False,float_format='%.12g')
    summary=[];macro_draws={}
    for (view,task),sub in point.groupby(['view','task'],sort=True):
        for k,metric in enumerate(METRICS):
            ix=sub.index[np.isfinite(sub[metric])].to_numpy();v=B[:,ix,k];ok=np.isfinite(v).all(axis=1) if len(ix) else np.zeros(draws,dtype=bool)
            vals=v[ok].mean(axis=1) if len(ix) else np.array([]);ci=np.percentile(vals,[2.5,97.5]) if len(vals) else [np.nan,np.nan]
            summary.append(dict(view=view,task=task,metric=metric,intended_sets=int((specs.task==task).sum()),predicted_sets=len(sub),effective_sets=len(ix),ineligible_or_unpredicted_sets=int((specs.task==task).sum())-len(ix),macro_equal_weight_mean=float(sub.loc[ix,metric].mean()) if len(ix) else np.nan,ci_low=ci[0],ci_high=ci[1],valid_bootstrap=int(ok.sum()),invalid_bootstrap=int((~ok).sum())))
            full=np.full(draws,np.nan);full[ok]=vals;macro_draws[view+'__'+task+'__'+metric]=full
    pd.DataFrame(summary).to_csv(ROOT/'results/macro_metrics.csv',index=False,float_format='%.12g')
    np.savez_compressed(ROOT/'results/bootstrap_fixed_scores.npz',per_set=B,set_ids=point.set_id.to_numpy(str),views=point.view.to_numpy(str),metrics=np.asarray(METRICS),**macro_draws)
    write_json(ROOT/'audit/evaluation_complete.json',{'utc':now(),'draws':draws,'seed':seed,'generator':'numpy.random.Generator(PCG64)','global_G0_groups':G,'group_index_sha256':sha256(ROOT/'audit/bootstrap_groups.csv.gz'),'resampling':'Sample G complete chemical graph groups with replacement globally; reuse the integer group multiplicities in every assay, task and analysis view. Labels/scores stay fixed. Sets with any single-class draw yield NA for that draw; no redrawing. Macro intervals retain the fixed point-estimable set roster and reject macro draws if any member is undefined.','CI':'Unadjusted pointwise 2.5/97.5 percentiles, excludes model-retraining variability and label uncertainty','EF1pct':'k=max(1,ceil(0.01*N)); fractional allocation within the exact-score tie group crossing k; EF=(expected top-k positives/k)/(P/N). Recompute k and prevalence for each weighted bootstrap sample.','input_predictions':audit,'seconds':time.monotonic()-start})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');a=ap.parse_args();self_test() if a.self_test else main()
