"""CPU-only null for the five archived cases. Writes only beside this script.

Run with the project's anaconda3/envs/toxcpi/bin/python. Configuration was
written before computing any null result. No source script is executed.
"""
from pathlib import Path
import json, hashlib, itertools, gzip, csv
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom

OUT = Path(__file__).resolve().parent
CFG = json.loads((OUT / '空模型配置.json').read_text())
ROOT = Path(CFG['project'])
WORK = ROOT / 'explanation/downstream_ao_4chem'
BASE = ROOT / 'baseline_compare/CTD_PATHWAY'
SHORT = CFG['chemical_order']
SLUGS = ['bisphenol_a','2378-tetrachlorodibenzo-p-dioxin','benzo(a)pyrene','chlorpyrifos','ddt']
CIDS = ['C006780','D013749','D001564','D004390','D003634']

def bh(p):
    p=np.asarray(p,float); order=np.argsort(p,kind='stable')
    v=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    q=np.empty_like(p);q[order]=np.minimum(v,1);return q

def jacc(a,b):
    return len(a&b)/len(a|b) if a|b else 0.0

def clean(s):return s.fillna('').astype(str).str.strip()

def main():
    paths={k:ROOT/v for k,v in CFG['input_paths'].items()}
    background=paths['background'].read_text().splitlines()
    assert len(background)==len(set(background))==8383
    bg=set(background);gi={g:i for i,g in enumerate(background)}
    ps=[];names=[];sets=[]
    for line in paths['gmt'].read_text().splitlines():
        if not line or line.startswith('#'):continue
        p=line.split('\t');ps.append(p[0]);names.append(p[1]);sets.append(set(p[2:])&bg)
    assert len(ps)==369 and all(sets)
    pi={p:i for i,p in enumerate(ps)};N=len(bg)
    rows=[];cols=[]
    for j,gs in enumerate(sets):
        for g in gs:rows.append(gi[g]);cols.append(j)
    A=sparse.csr_matrix((np.ones(len(rows),dtype=np.int32),(rows,cols)),shape=(N,len(ps)))
    sizes=np.array([len(s) for s in sets]);degrees=np.asarray(A.sum(axis=1)).ravel()
    disease=pd.read_parquet(paths['ctd_disease'])
    assert not disease.duplicated(['pathway_id','disease_id']).any()
    # Degree is based on distinct disease IDs per inference gene in the entire
    # fixed CTD KEGG disease snapshot, independent of observed enriched paths.
    dg=defaultdict(set)
    for d in disease.itertuples():
        for g in str(d.inference_genes).split(';'):
            if g in bg:dg[g].add(d.disease_id)
    dd=np.array([len(dg[g]) for g in background])
    kb=np.digitize(degrees,[2,5,10,20]);db=np.digitize(dd,[1,10,100])
    strata=kb*4+db;groups=[np.flatnonzero(strata==s) for s in sorted(set(strata))]
    labels=np.array([f'K{k}_D{d}' for k,d in zip(kb,db)])
    pd.DataFrame({'gene_symbol':background,'KEGG_degree':degrees,'CTD_disease_degree':dd,'stratum':labels}).to_csv(OUT/'背景注释分层.csv.gz',index=False)
    links={}
    for pid,d in disease.groupby('pathway_id',sort=False):
        # stable sort retains source order on exact equal supports.
        d=d.sort_values('n_inference_genes',ascending=False,kind='stable')
        links[pid]=list(zip(d.disease_id.astype(str),np.log1p(d.n_inference_genes.astype(float))))
    refs=pd.read_csv(paths['ctd_reference'])
    refs=refs[refs.background_type.eq('B_common') & refs.in_background_comparable.astype(bool)]
    refsets=[set(refs.loc[refs.ctd_chemical_id.eq(cid),'pathway_id']) for cid in CIDS]
    masks=[];audit=[];protein_counts=[]
    universe=pd.read_csv(WORK/'inputs/human_proteome_universe.csv')
    uni_gene=clean(universe.gene_symbol)
    audit_universe={'total_proteins':len(universe),'invalid_sequence':int((~universe.valid_sequence).sum()),
                    'valid_sequence':int(universe.valid_sequence.sum()),
                    'missing_primary_gene_all':int(uni_gene.eq('').sum()),
                    'valid_sequence_missing_gene':int((universe.valid_sequence & uni_gene.eq('')).sum()),
                    'valid_unique_genes':int(uni_gene[universe.valid_sequence & uni_gene.ne('')].nunique()),
                    'maximum_genes_per_protein':int(universe.groupby('uniprot_id').gene_symbol.nunique().max()),
                    'maximum_proteins_per_gene':int(universe.groupby('gene_symbol').uniprot_id.nunique().max())}
    for short,slug,cid in zip(SHORT,SLUGS,CIDS):
        d=pd.read_csv(WORK/f'predictions/all_human_proteome_scores_{slug}.csv')
        d.gene_symbol=clean(d.gene_symbol)
        assert len(d)==20209 and d.uniprot_id.nunique()==20209
        assert np.isfinite(d[['P_binding','P_activation','P_inhibition']]).all().all()
        q=d[d[['P_binding','P_activation','P_inhibition']].max(axis=1).ge(.5)]
        genes=set(q.gene_symbol)-{''};valid=genes&bg
        mask=np.array([g in valid for g in background]);masks.append(mask)
        assert (q.S_target-q[['P_binding','P_activation','P_inhibition']].max(axis=1)).abs().max()<1e-12
        target=pd.read_csv(WORK/f'target_sets/union_positive_targets_{slug}_threshold_0.5.csv')
        assert set(target.uniprot_id)==set(q.uniprot_id)
        audit.append({'chemical':short,'n_scored_proteins':len(d),'n_union_proteins':len(q),
                      'n_mapped_unique_genes':len(genes),'protein_to_gene_collapsed':len(q)-len(genes),
                      'n_effective_background_genes':len(valid),'n_genes_outside_KEGG_background':len(genes-bg),
                      'n_selected_proteins_missing_gene':int(q.gene_symbol.eq('').sum()),
                      'duplicate_selected_UID':int(q.uniprot_id.duplicated().sum())})
        protein_counts.append({**audit[-1]})
    Q=np.asarray(masks,dtype=np.int32);ns=Q.sum(axis=1)
    assert ns.tolist()==[3688,1315,5227,4626,6476]
    targetfile=pd.DataFrame({'gene_symbol':background,**{s:Q[i].astype(bool) for i,s in enumerate(SHORT)}})
    targetfile.to_csv(OUT/'冻结目标基因集合.csv.gz',index=False)
    # Each n is fixed. Precompute every possible k's exact upper-tail p.
    maxk=int(sizes.max());lut=[]
    for n in ns:
        lut.append(hypergeom.sf(np.arange(maxk+1)[None,:]-1,N,sizes[:,None],int(n)))
    route=pd.read_csv(paths['route1_ranked'])
    original_disease=pd.read_csv(paths['route1_disease_full'])
    pairs=list(itertools.combinations(range(5),2))
    def one(q,c,retain=False):
        k=np.asarray(q@A).ravel().astype(int)
        p=lut[c][np.arange(len(ps)),k];f=bh(p)
        fold=(k/ns[c])/(sizes/N)
        # GMT order is the residual historical tie order. No ties affect
        # observed top10 boundaries; source equality is checked below.
        order=np.lexsort((np.arange(len(ps)),-fold,-k,p,f))
        sig={ps[i] for i in np.flatnonzero((f<.05)&(k>=2))}
        display=[i for i in order if f[i]<.05][:10]
        all_d=set();per5=set();global_score={};per10=[]
        for i in display:
            ll=links.get(ps[i],[])
            all_d.update(x[0] for x in ll)
            per5.update(x[0] for x in ll[:5])
            per10.extend((ps[i],d) for d,_ in ll[:10])
            for d,s in ll:global_score[d]=max(global_score.get(d,0),float(fold[i]*s))
        top20=set(sorted(global_score,key=lambda d:(-global_score[d],d))[:20])
        ref=refsets[c];ov=sig&ref
        r={'chemical':SHORT[c],'n_effective_genes':int(ns[c]),'n_significant_pathways':len(sig),
           'n_reference':len(ref),'n_ctd_overlap':len(ov),
           'CTD_support':len(ov)/len(sig) if sig else None,
           'CTD_recovery':len(ov)/len(ref) if ref else None,'CTD_Jaccard':jacc(sig,ref),
           'n_display_pathways':len(display),'n_display_pathways_without_disease':sum(ps[i] not in links for i in display),
           'n_unique_diseases_full':len(all_d),'n_diseases_top20':len(top20),
           'sig_pathways':';'.join(sorted(sig)),'display_pathways':';'.join(ps[i] for i in display),
           'top20_diseases':';'.join(sorted(top20)),'perpath_top5_unique_diseases':';'.join(sorted(per5)),
           'perpath_top10_pairs':';'.join(p+':'+d for p,d in per10)}
        if retain:
            r['display_pathway_detail']=[{'pathway_id':ps[i],'fdr':float(f[i]),'overlap_count':int(k[i]),'fold_enrichment':float(fold[i])} for i in display]
        return r,top20,per5
    observed=[];od=[];op5=[]
    for c in range(5):
        row,t,p5=one(Q[c],c,True);observed.append(row);od.append(t);op5.append(p5)
        rr=route[route.ctd_chemical_id.eq(CIDS[c])].sort_values('standard_ora_rank')
        assert set(rr.pathway_id)==set(row['sig_pathways'].split(';'))
        assert rr[rr.selected_for_downstream].pathway_id.tolist()==row['display_pathways'].split(';')
        stored=original_disease[original_disease.chemical_name.eq(rr.chemical_name.iloc[0])].sort_values('chemical_disease_rank').head(20)
        assert set(stored.disease_id)==t,('disease mismatch',SHORT[c])
    def pairrows(ds,p5):
        return [{'pair':SHORT[a]+' / '+SHORT[b], 'top20_Jaccard':jacc(ds[a],ds[b]),
                 'top5pool_Jaccard':jacc(p5[a],p5[b]), 'both_nonempty':bool(ds[a] and ds[b])} for a,b in pairs]
    obs_pairs=pairrows(od,op5);obs_mean=float(np.mean([r['top20_Jaccard'] for r in obs_pairs]))
    R=CFG['iterations_per_null'];metrics=defaultdict(list);mean_draws={};strata_info=[]
    for g in groups:
        strata_info.append({'stratum':labels[g[0]],'n_background':len(g),**{SHORT[c]:int(Q[c,g].sum()) for c in range(5)}})
    with gzip.open(OUT/'空模型逐轮完整集合.csv.gz','wt',newline='') as handle, gzip.open(OUT/'空模型逐轮疾病重叠.csv.gz','wt',newline='') as ph:
        writer=None;pw=None
        for design_index,design in enumerate(['uniform','degree_matched']):
            rng=np.random.default_rng(np.random.SeedSequence([CFG['seed'],design_index]));means=[]
            for rep in range(R):
                perm=np.arange(N)
                if design=='uniform':perm=rng.permutation(N)
                else:
                    for g in groups:perm[g]=rng.permutation(g)
                permq=Q[:,perm]
                assert np.array_equal(permq.sum(axis=1),ns)
                assert np.array_equal(permq@permq.T,Q@Q.T)
                if design=='degree_matched':
                    assert all(np.array_equal(permq[:,g].sum(axis=1),Q[:,g].sum(axis=1)) for g in groups)
                ds=[];p5s=[]
                for c in range(5):
                    rr,ds1,p51=one(permq[c],c);ds.append(ds1);p5s.append(p51)
                    rr={'null':design,'replicate':rep+1,**rr}
                    if writer is None:writer=csv.DictWriter(handle,fieldnames=list(rr));writer.writeheader()
                    writer.writerow(rr)
                    for field in ['CTD_Jaccard','CTD_support','CTD_recovery','n_significant_pathways','n_ctd_overlap','n_diseases_top20']:
                        metrics[(design,c,field)].append(rr[field])
                pr=pairrows(ds,p5s)
                v=float(np.mean([x['top20_Jaccard'] for x in pr]));means.append(v)
                for x in pr:
                    x={'null':design,'replicate':rep+1,**x}
                    if pw is None:pw=csv.DictWriter(ph,fieldnames=list(x));pw.writeheader()
                    pw.writerow(x)
                if (rep+1)%200==0:print(f'{design}: {rep+1}/{R}',flush=True)
            mean_draws[design]=means
    def summary(v,obs,test=False):
        vv=np.array([x for x in v if x is not None],float)
        rr={'observed':obs,'n_defined':len(vv),'n_missing':len(v)-len(vv)}
        if len(vv):rr.update(mean=float(vv.mean()),median=float(np.median(vv)),q025=float(np.quantile(vv,.025)),q975=float(np.quantile(vv,.975)),minimum=float(vv.min()),maximum=float(vv.max()))
        if test:
            rr['p_upper']=(1+int(np.sum(vv>=obs-1e-15)))/(len(vv)+1)
            rr['p_lower']=(1+int(np.sum(vv<=obs+1e-15)))/(len(vv)+1)
        return rr
    primary={design:summary(v,obs_mean,True) for design,v in mean_draws.items()}
    for design,q in zip(primary,bh([x['p_upper'] for x in primary.values()])):primary[design]['BH_q_upper']=float(q)
    secondary=[]
    for design in primary:
        for c in range(5):
            d={'null':design,'chemical':SHORT[c]}
            for field in ['CTD_Jaccard','CTD_support','CTD_recovery','n_significant_pathways','n_ctd_overlap','n_diseases_top20']:
                d[field]=summary(metrics[(design,c,field)],observed[c][field],field=='CTD_Jaccard')
            secondary.append(d)
    for x,q in zip(secondary,bh([x['CTD_Jaccard']['p_upper'] for x in secondary])):x['CTD_Jaccard']['BH_q_upper']=float(q)
    # Reuse all existing size-matched sensitivity rows, retaining unfavorable
    # and undefined results; do not select K or a case after null results.
    pm=pd.read_csv(BASE/'metrics/pathway_metrics_by_chemical.csv')
    top=pm[pm.protocol.eq('topk') & pm.background_type.eq('B_common') & pm.significance_level.eq('fdr_overlap2')].copy()
    top.to_csv(OUT/'既有等量topK完整结果.csv',index=False)
    # Figure 5 includes PMF, absent from the earlier navigation's model list.
    fig=pd.read_csv(BASE/'pmf_effect/metrics/figure5_three_view_metrics.csv')
    assert len(fig)==15 and fig.background_size.eq(N).all() and fig.cutoff.eq(.5).all()
    fig.to_csv(OUT/'Figure5实际比较输入.csv',index=False)
    snapshots={k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
    result={'config':CFG,'snapshot_inputs':snapshots,'universe_mapping':audit_universe,
            'mapping_audit':audit,'background_size':N,'GMT_n_pathways':len(ps),
            'GMT_unique_genes':len(set().union(*(set(line.split('\t')[2:]) for line in paths['gmt'].read_text().splitlines() if line and not line.startswith('#')))),
            'pathway_background_minmax':[int(sizes.min()),int(sizes.max())],
            'source_disease_associations':len(disease),'source_disease_pathways':int(disease.pathway_id.nunique()),
            'source_diseases':int(disease.disease_id.nunique()),
            'target_intersections':(Q@Q.T).tolist(),'strata':strata_info,'observed':observed,
            'observed_pairwise_disease_overlap':obs_pairs,'primary_null_tests':primary,'secondary_null_tests':secondary,
            'verification':'Raw union=max>=0.5 and stored target sets agree; observed ORA significant sets/top10 and chemical-level top20 disease IDs exactly reproduce current route1 outputs; target sizes and all five-set pairwise intersections checked for every draw; degree strata checked every matched draw.'}
    (OUT/'通路疾病核对与空模型结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'mapping':audit,'primary':primary},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
