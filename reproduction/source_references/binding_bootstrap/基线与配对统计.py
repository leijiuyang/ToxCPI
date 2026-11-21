#!/usr/bin/env python3
"""Read-only recovery of archived baselines; fixed-checkpoint binding bootstrap.

Run with 1_TOXCPI/leijiuyang/anaconda3/envs/toxcpi/bin/python.
No training, feature extraction, source-label reconstruction or source-file writes.
The JSON is the single numerical source used for this category's SI tables.
"""
from pathlib import Path
from collections import Counter
import json, pickle, re
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, matthews_corrcoef, precision_recall_curve, auc

ROOT = Path('/home/data2/leijiuyang/1_TOXCPI/leijiuyang')
OUT = Path(__file__).resolve().parent
METRICS = ['AUROC', 'AUPRC', 'Accuracy', 'F1', 'MCC']
B = 2000
SEED = 20261025
RDLogger.DisableLog('rdApp.*')

def read(rel):
    return pd.read_csv(ROOT / rel, low_memory=False)

def fixed(task, split, part='test'):
    rel = (f'test1/data/interaction_binding/splits/{part}_{split}.csv' if task == 'binding'
           else f'test1/data/interaction_type_1to1_targetaware/effect_{task}_{split}_{part}.csv')
    return read(rel)

def metrics(y, s, w=None):
    p = s >= .5
    return np.array([roc_auc_score(y,s,sample_weight=w), average_precision_score(y,s,sample_weight=w),
                     accuracy_score(y,p,sample_weight=w), f1_score(y,p,sample_weight=w,zero_division=0),
                     matthews_corrcoef(y,p,sample_weight=w)])

def pair_keys(df):
    return list(zip(df.smiles.astype(str),df.uniprot_id.astype(str)))

def standard(src, y, s):
    assert np.array_equal(src.label.astype(int), np.asarray(y).astype(int))
    d = src[['smiles','uniprot_id','label']].copy().reset_index(drop=True)
    d['score'] = np.asarray(s, dtype=float)
    assert np.isfinite(d.score).all() and d.score.between(0,1).all()
    return d

def ablation(task, split, variant='full'):
    rel=f'test1/experiments/ablation/predictions/{task}_{split}_{variant}_seed42_test_predictions.csv'
    p=read(rel); src=fixed(task,split).iloc[p.source_row_index.to_numpy()].reset_index(drop=True)
    assert np.array_equal(src.chemical_name.astype(str),p.chemical_name.astype(str))
    assert np.array_equal(src.uniprot_id.astype(str),p.uniprot_id.astype(str))
    return standard(src,p.label,p.probability),rel

def keyed(src, p, label, score):
    # All archived Figure-2 binding keyed files have unique raw-SMILES/UniProt keys.
    lookup={k:i for i,k in enumerate(pair_keys(src))}; assert len(lookup)==len(src)
    inds=[lookup[k] for k in pair_keys(p)]
    return standard(src.iloc[inds].reset_index(drop=True),p[label],p[score])

def binding(split):
    src=fixed('binding',split); result={}; paths={}
    result['ToxCPI'],paths['ToxCPI']=ablation('binding',split)
    # test_results.csv omits identities. Recover the deterministic source order:
    # train_test1_v2.py filter_df preserves order; test loader has shuffle=False.
    fd=ROOT/'tipFormer2025-main/features_test1_v2'
    with open(fd/'compound_features.pkl','rb') as f: ck=set(pickle.load(f))
    with open(fd/'protein_features.pkl','rb') as f: pk=set(pickle.load(f))
    valid=src.chemical_name.isin(ck)&src.uniprot_id.isin(pk)
    rel=f'tipFormer2025-main/outputs_test1_v2/{"warm_start" if split=="warm" else "cold_chemical"}/test_results.csv'
    p=read(rel); assert len(p)==int(valid.sum())
    result['tipFormer']=standard(src.loc[valid].reset_index(drop=True),p.true_label,p.pred_prob);paths['tipFormer']=rel
    rel=f'test1/baseline_compare/MolTrans_binding/results/moltrans_binding_{split}_test_predictions.csv'
    p=read(rel)
    # MolTrans export lacks UniProt ID; its converter retains source row order.
    # Identical protein sequences can belong to different IDs, so sequence alone
    # is deliberately not used as a purported unique protein identity.
    assert len(p)==len(src)
    assert np.array_equal(src.smiles.astype(str),p.smiles.astype(str))
    assert np.array_equal(src.protein_sequence.astype(str),p.protein_sequence.astype(str))
    result['MolTrans']=standard(src,p.label,p.pred_prob);paths['MolTrans']=rel
    rel=f'test1/baseline_compare/TransformerCPI2.0_binding/results/transformercpi2_binding_{split}_cached1024_bs64_test_predictions.csv'
    p=read(rel).rename(columns={'SMILES':'smiles'})
    result['TransformerCPI2.0']=keyed(src,p,'true_label','pred_prob');paths['TransformerCPI2.0']=rel
    # Use RF_XGBoost_binding, the actual Figure-2 run, not the later RF rerun.
    rel=f'test1/baseline_compare/RF_XGBoost_binding/results/RF_binding_{split}_test_predictions.csv'
    p=read(rel);result['RF']=keyed(src,p,'true_label','pred_prob');paths['RF']=rel
    rel=f'test1/baseline_compare/HyperAttentionDTI_binding/results/hyperattentiondti_binding_{split}_test_predictions.csv'
    p=read(rel).rename(columns={'protein_id':'uniprot_id'})
    result['HyperAttentionDTI']=keyed(src,p,'label','pred_prob');paths['HyperAttentionDTI']=rel
    # Graphormer export lacks IDs, but the archived evaluated pair strings retain
    # them. Verify all exported positions, sequences, and labels before lookup.
    rel=f'test1/baseline_compare/GraphormerDTI_binding/results/graphormerdti_binding_{split}_test_predictions.csv'
    p=read(rel)
    with open(ROOT/f'test1/baseline_compare/GraphormerDTI_binding/protein_data/ToxCPI_binding_{split}_pairs.pkl','rb') as f:
        pair_strings=pickle.load(f)['test']
    parts=[x.rsplit(maxsplit=4) for x in pair_strings]
    assert len(parts)==len(p) and all(len(x)==5 for x in parts)
    # Original parse_pair exports only the first whitespace token of CXSMILES.
    # Recover the intact stored string from the archived pair string, while
    # checking the exact export behavior and every sequence/label position.
    assert [x[0].split()[0] for x in parts]==p.smiles.astype(str).tolist()
    assert [x[3] for x in parts]==p.protein_sequence.astype(str).tolist()
    assert [int(x[4]) for x in parts]==p.label.astype(int).tolist()
    p['uniprot_id']=[x[2] for x in parts]
    p['smiles']=[x[0] for x in parts]
    result['GraphormerDTI']=keyed(src,p,'label','pred_prob');paths['GraphormerDTI']=rel
    return result, paths

def dedup(d):
    assert not d[['smiles','uniprot_id']].isna().any().any()
    g=d.groupby(['smiles','uniprot_id'],sort=False)
    assert g.label.nunique().max()==1
    # Do not silently average different predictions on one pair.
    assert g.score.nunique().max()==1
    return d.drop_duplicates(['smiles','uniprot_id']).set_index(['smiles','uniprot_id'])

def fast_cache(y,s):
    order=np.argsort(-s,kind='stable'); ends=np.r_[np.flatnonzero(np.diff(s[order])),len(s)-1]
    return order,ends,y[order].astype(float),s>=.5

def fast(y,w,cache):
    order,ends,ys,pred=cache;ws=w[order]
    tp=np.cumsum(ws*ys)[ends];fp=np.cumsum(ws*(1-ys))[ends]
    P=tp[-1];N=fp[-1]
    auc=np.trapz(np.r_[0,tp],np.r_[0,fp])/(P*N)
    ap=np.sum(np.diff(np.r_[0,tp])*tp/np.maximum(tp+fp,1))/P
    TP=np.sum(w*(y==1)*pred);FP=np.sum(w*(y==0)*pred)
    TN=N-FP;FN=P-TP
    den=np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return np.array([auc,ap,(TP+TN)/(P+N),2*TP/(2*TP+FP+FN) if 2*TP+FP+FN else 0,
                     (TP*TN-FP*FN)/den if den else 0])

def bootstrap(models,paths,split,scope,seed):
    ds={m:dedup(d) for m,d in models.items()}
    keys=sorted(set.intersection(*(set(d.index) for d in ds.values())))
    ys=[d.loc[keys,'label'].to_numpy(dtype=int) for d in ds.values()]
    assert all(np.array_equal(ys[0],y) for y in ys)
    y=ys[0];names=list(ds)
    scores={m:ds[m].loc[keys,'score'].to_numpy() for m in names}
    # Canonical isomeric molecular graph clusters retain stereochemistry and salts;
    # no parent-fragment selection, tautomer normalization, or stereo stripping.
    ids=[]
    for smi,_ in keys:
        mol=Chem.MolFromSmiles(smi); assert mol is not None
        ids.append(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True))
    groups,inverse=np.unique(ids,return_inverse=True);G=len(groups)
    cache={m:fast_cache(y,scores[m]) for m in names}
    point={m:metrics(y,scores[m]) for m in names}
    wcheck=(np.arange(len(y))%3+1).astype(float)
    for m in names:
        assert np.allclose(fast(y,wcheck,cache[m]),metrics(y,scores[m],wcheck),rtol=0,atol=1e-12)
    rng=np.random.default_rng(seed); draws=[];rejected=0
    while len(draws)<B:
        selected=rng.integers(0,G,size=G)
        # This weight vector represents one common multiset of sampled group indices
        # including every pair of each selected chemical; all methods share it.
        w=np.bincount(selected,minlength=G)[inverse].astype(float)
        if np.sum(w*y)==0 or np.sum(w*(1-y))==0:
            rejected+=1;continue
        draws.append(np.stack([fast(y,w,cache[m]) for m in names]))
    a=np.asarray(draws);ref=names.index('ToxCPI');delta=a[:,ref:ref+1,:]-a
    result={}
    for i,m in enumerate(names):
        result[m]={'point':dict(zip(METRICS,point[m].tolist())),
                   'ci95':dict(zip(METRICS,np.quantile(a[:,i,:],[.025,.975],axis=0).T.tolist())),
                   'delta_tox_minus_method':dict(zip(METRICS,(point['ToxCPI']-point[m]).tolist())),
                   'delta_ci95':dict(zip(METRICS,np.quantile(delta[:,i,:],[.025,.975],axis=0).T.tolist())),
                   'input_rows':len(models[m]),'unique_input_pairs':len(ds[m]),
                   'duplicates_removed':len(models[m])-len(ds[m]),'excluded_from_common':len(ds[m])-len(keys),
                   'prediction_file':paths[m]}
    print(scope,split,'common pairs',len(keys),'chemical graph clusters',G,'single-class rejected',rejected,flush=True)
    return {'split':split,'scope':scope,'n_pairs':len(keys),'n_positive':int(y.sum()),'n_negative':int(len(y)-y.sum()),
            'n_chemical_graphs':G,'resampling_seed':seed,'accepted':B,'single_class_rejected':rejected,'models':result}

def main():
    out={'protocol':{'replicates':B,'rng':'numpy.default_rng','seed_base':SEED,'ci':'pointwise equal-tailed percentile 2.5/97.5%',
                     'threshold':.5,'pair_key':['stored SMILES','UniProt ID'],
                     'cluster_key':'RDKit canonical isomeric SMILES; salts and stereochemistry retained',
                     'scope':'Fixed archived binding checkpoints only; no training or split resampling; effects remain archived point estimates.',
                     'single_class_rule':'Reject whole paired draw, count it, and redraw until 2000 valid draws.',
                     'delta':'ToxCPI minus comparator, using the same sampled chemical indices for all methods.',
                     'multiplicity':'Pointwise intervals, no multiplicity correction; no global superiority claim.'},
         'absolute':{},'paired':[],'dtiam_protocol':{}}
    for i,split in enumerate(['warm','cold']):
        models,paths=binding(split)
        out['absolute']['binding_'+split]={m:{'n':len(d),'metrics':dict(zip(METRICS,metrics(d.label.to_numpy(),d.score.to_numpy()).tolist())),
                                                            'prediction_file':paths[m]} for m,d in models.items()}
        for m,d in models.items():
            r=out['absolute']['binding_'+split][m]
            r['archived_metrics']=dict(r['metrics'])
            r['archived_pr_definition']='average precision'
            if m=='tipFormer':
                precision,recall,_=precision_recall_curve(d.label,d.score)
                r['archived_metrics']['AUPRC']=float(auc(recall,precision))
                r['archived_pr_definition']='trapezoidal PR area'
        out['paired'].append(bootstrap(models,paths,split,'baselines_common_subset',SEED+i))
        ab={'ToxCPI':models['ToxCPI']};ap={'ToxCPI':paths['ToxCPI']}
        for v in ['mean_pooling','single_head']:
            ab[v],ap[v]=ablation('binding',split,v)
        out['paired'].append(bootstrap(ab,ap,split,'binding_ablation_common_pairs',SEED+2+i))
    pmf={('activation','warm'):[.8719,.8621,.7954,.7998,.5913],('activation','cold'):[.8284,.8292,.7709,.7711,.5419],
         ('inhibition','warm'):[.9372,.9318,.8751,.8804,.7532],('inhibition','cold'):[.9272,.9246,.8489,.8553,.7005]}
    pmf_data=ROOT/'PMF-CPI/datasets'
    for task in ['activation','inhibition']:
        for split in ['warm','cold']:
            src=fixed(task,split);full,rel=ablation(task,split)
            rows={'ToxCPI':{'n':len(full),'metrics':dict(zip(METRICS,metrics(full.label.to_numpy(),full.score.to_numpy()).tolist())), 'prediction_file':rel}}
            for model,directory in [('RF','ECFP_ESM2_XGBRF'),('MLP','ECFP_ESM2_MLP')]:
                rel=f'test1/baseline_compare/{directory}/results/{model}_{task}_{split}_test_predictions.csv';p=read(rel)
                # Warm/cold effect source files have nontrivial name/identity behavior;
                # compare row multisets, retaining multiplicity, not just row counts.
                assert Counter(zip(src.smiles,src.uniprot_id,src.label))==Counter(zip(p.smiles,p.uniprot_id,p.true_label))
                rows[model]={'n':len(p),'metrics':dict(zip(METRICS,metrics(p.true_label.to_numpy(),p.pred_prob.to_numpy()).tolist())), 'prediction_file':rel}
            log=f'PMF-CPI/classification/toxcpi_effect_logs/{task}_{split}_'+('cuda0_20260510_1640.log' if (task,split)==('inhibition','cold') else 'cuda2_20260510_1632.log')
            text=(ROOT/log).read_text(errors='replace'); assert '\t'.join(f'{v:.4f}' for v in pmf[task,split][:3]) in text
            rows['PMF-CPI']={'n':len(src),'metrics':dict(zip(METRICS,pmf[task,split])),'precision':'four-decimal test log; no pair predictions archived by training entry','source':log}
            # PMF matrix construction collapses duplicate raw-SMILES/ID cells.
            # Verify actual matrices against the fixed records, not just counts.
            matrix_checks={}
            for phase,dirname in [('train','train'),('val','val'),('test','ind_test')]:
                raw=fixed(task,split,phase);part=pmf_data/f'toxcpi_effect_{task}_{split}'/dirname
                chems=list(json.load(open(part/'compounds.txt')).values());prots=list(json.load(open(part/'proteins.txt')))
                with open(part/'Y','rb') as f: yy=pickle.load(f)
                ii,jj=np.where(~np.isnan(yy));pc=Counter((chems[i],prots[j],int(yy[i,j])) for i,j in zip(ii,jj))
                rawc=Counter(zip(raw.smiles,raw.uniprot_id,raw.label))
                assert set(pc)==set(rawc) and max(pc.values())==1
                matrix_checks[phase]={'source_rows':len(raw),'matrix_pairs':len(pc),'duplicate_rows_collapsed':len(raw)-len(pc)}
            rows['PMF-CPI']['matrix_checks']=matrix_checks
            rows['PMF-CPI']['n']=matrix_checks['test']['matrix_pairs']
            for m,r in rows.items():
                r['archived_metrics']=dict(r['metrics'])
                r['archived_pr_definition']='trapezoidal PR area' if m=='PMF-CPI' else 'average precision'
            out['absolute'][task+'_'+split]=rows
            key=task+'_'+split;checks={}
            for phase in ['train','val','test']:
                a=fixed(task,split,phase);rel=f'DTIAM-main/DTIAM-main/data/moa/{task}/data_folds/{split}/{phase}.csv';d=read(rel)
                ac=Counter(zip(a.smiles,a.uniprot_id,a.label));dc=Counter(zip(d.DrugID,d.TargetID,d.label))
                checks[phase]={'current_rows':len(a),'dtiam_rows':len(d),'common_labeled_rows':sum((ac&dc).values()),
                               'current_only':sum((ac-dc).values()),'dtiam_only':sum((dc-ac).values()),
                               'current_only_positives':sum(v for k,v in (ac-dc).items() if k[2]==1),
                               'dtiam_only_positives':sum(v for k,v in (dc-ac).items() if k[2]==1),'dtiam_split_file':rel}
            rel=f'DTIAM-main/DTIAM-main/results/{key}_result.csv';p=read(rel).iloc[0]
            out['dtiam_protocol'][key]={'pair_multiset_checks':checks,'archived_metrics':{str(k):float(v) for k,v in p.items()},
                                      'accuracy':None,'source':rel,'note':'Different sampled negatives; do not rank these metrics against Figure 2.'}
    (OUT/'基线与配对统计结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('Saved',OUT/'基线与配对统计结果.json',flush=True)

if __name__=='__main__':
    main()
