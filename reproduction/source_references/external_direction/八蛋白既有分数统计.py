"""Category 9: frozen eight-protein scores only; no inference, fitting or label repair.
Run with the existing toxcpi environment. Outputs one JSON and the exact used
686 annotation records, including source fields, graph group and dual-label flag.
"""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from rdkit import Chem, rdBase
P=Path('/home/data2/leijiuyang/1_TOXCPI/leijiuyang/test1')
E=P/'paper_draft/results_figs/task2_effect_direction_v2_external_screen'
O=Path(__file__).resolve().parent
GENES=['PTGER4','GABRA1','TBXA2R','NR3C1','CHRM5','ADRA2B','ADRB2','PTGDR2']
@lru_cache(None)
def graph(s):
 m=Chem.MolFromSmiles(s)
 return Chem.MolToSmiles(m,isomericSmiles=True) if m is not None else None
s=pd.read_csv(E/'effect_direction_scores_v2_external_all56.csv')
s=s[s.gene_symbol.isin(GENES)].copy()
assert len(s)==686 and not s.duplicated(['smiles','uniprot_id','effect_type']).any()
cols=['logit_activation_raw','logit_inhibition_raw','P_activation_raw','P_inhibition_raw','P_activation_cal','P_inhibition_cal','S_direction']
assert np.isfinite(s[cols].to_numpy()).all()
for _,d in s.groupby(['smiles','uniprot_id']):
 assert all(d[c].max()-d[c].min()<1e-6 for c in cols)
 # Preserve archived per-annotation values; do not average numerical differences.
s['graph_group']=s.smiles.map(graph);assert s.graph_group.notna().all()
s['dual_direction_pair']=s.groupby(['smiles','uniprot_id']).effect_type.transform('nunique').eq(2)
s['S_raw']=s.logit_activation_raw-s.logit_inhibition_raw
# Stored parameters are rounded. Verify reproduction to their precision only.
pa=np.clip(1/(1+np.exp(-(0.46427871*s.logit_activation_raw+0.044898138))),1e-6,1-1e-6)
pi=np.clip(1/(1+np.exp(-(1.0041685*s.logit_inhibition_raw-0.56821921))),1e-6,1-1e-6)
checks={'max_within_pair_ranges':(s.groupby(['smiles','uniprot_id'])[cols].max()-s.groupby(['smiles','uniprot_id'])[cols].min()).max().to_dict(),'max_cal_p_error_from_rounded_params':float(max(abs(pa-s.P_activation_cal).max(),abs(pi-s.P_inhibition_cal).max())),
 'max_S_error_from_stored_cal_p':float(abs(np.log(s.P_activation_cal/(1-s.P_activation_cal))-np.log(s.P_inhibition_cal/(1-s.P_inhibition_cal))-s.S_direction).max()),
 'zero_direction_scores':int(s.S_direction.eq(0).sum())}
assert checks['max_cal_p_error_from_rounded_params']<1e-7 and checks['max_S_error_from_stored_cal_p']<1e-10
# Analysis remains annotation weighted: both labels on the same pair retained,
# with paired predictions kept in the same molecular-graph bootstrap cluster.
def metrics(y,z):
 pred=z>0
 ra=float(pred[y==1].mean());ri=float((~pred[y==0]).mean())
 return np.array([np.mean(pred==y),ra,ri,roc_auc_score(y,z),(ra+ri)/2])
keys=['accuracy','activation_recall','inhibition_recall','direction_auroc','balanced_accuracy']
results=[]
for gi,g in enumerate(GENES):
 d=s[s.gene_symbol.eq(g)].reset_index(drop=True);y=d.true_direction.eq('activation').to_numpy().astype(int)
 z=d.S_direction.to_numpy();zr=d.S_raw.to_numpy()
 # Constant-direction baselines fixed without fitting; majority is descriptive only.
 zs={'calibrated':z,'raw_logit_difference':zr,'always_activation':np.ones(len(d)),'always_inhibition':-np.ones(len(d))}
 point={m:metrics(y,v) for m,v in zs.items()}
 inds=[np.flatnonzero(d.graph_group.eq(k).to_numpy()) for k in sorted(d.graph_group.unique())]
 rng=np.random.default_rng(2026102509+gi);bs={m:[] for m in zs};reject=0
 while len(bs['calibrated'])<2000:
  ix=np.concatenate([inds[j] for j in rng.integers(0,len(inds),len(inds))])
  if np.unique(y[ix]).size<2:reject+=1;continue
  for m,v in zs.items():bs[m].append(metrics(y[ix],v[ix]))
 model={}
 for m in zs:
  v=np.array(bs[m]);delta=np.array(bs['calibrated'])-v
  model[m]={'point':dict(zip(keys,point[m].tolist())),
            'ci95':dict(zip(keys,np.quantile(v,[.025,.975],axis=0).T.tolist())),
            'delta_cal_minus_method':dict(zip(keys,(point['calibrated']-point[m]).tolist())),
            'delta_ci95':dict(zip(keys,np.quantile(delta,[.025,.975],axis=0).T.tolist()))}
 arch=pd.read_csv(E/'effect_direction_metrics_v2_external_showcase8.csv').set_index('gene').loc[g]
 for k in keys[:4]:assert abs(point['calibrated'][keys.index(k)]-arch[k])<1e-12
 results.append({'gene':g,'uniprot':d.uniprot_id.iloc[0],'n_annotations':len(d),'n_activation':int(y.sum()),'n_inhibition':int((1-y).sum()),'n_pairs':len(d.drop_duplicates(['smiles','uniprot_id'])), 'n_stored_chemicals':d.smiles.nunique(),'n_graph_groups':len(inds),'n_dual_pairs':int(d[d.dual_direction_pair].drop_duplicates(['smiles','uniprot_id']).shape[0]),'source_counts':d.data_source.value_counts().to_dict(),'n_excluded':0,'seed':2026102509+gi,'accepted_draws':2000,'single_class_rejected':reject,'models':model})
 print(g,len(d),len(inds),flush=True)
# Necessary identity checks restricted to this eight-protein external set.
# Include BOTH labels in each split; separate counts for positive and negative.
def info(d):
 rawpairs=set(zip(d.smiles,d.uniprot_id));rawchem=set(d.smiles)
 d=d.copy();d['graph']=d.smiles.map(graph)
 gp=set(zip(d['graph'],d.uniprot_id));gc=set(d['graph'].dropna())
 return {'n_rows':len(d),'n_unique_pairs':len(rawpairs),'n_unique_chemicals':len(rawchem),'n_graphs':len(gc),'unparseable_strings':len({v for v in rawchem if graph(v) is None}),
 'raw_pair_overlap':len(rawpairs&extp),'raw_chemical_overlap':len(rawchem&extc),'graph_pair_overlap':len(gp&extgp),'graph_chemical_overlap':len(gc&extgc)}
extp=set(zip(s.smiles,s.uniprot_id));extc=set(s.smiles);extgp=set(zip(s.graph_group,s.uniprot_id));extgc=set(s.graph_group)
overlaps=[];training={};calibration={}
for task in ['activation','inhibition','binding']:
 for split in ['cold','warm']:
  for part in ['train','val','test']:
   path=(P/'data/interaction_binding/splits'/f'{part}_{split}.csv') if task=='binding' else P/'data/interaction_type_1to1_targetaware'/f'effect_{task}_{split}_{part}.csv'
   d=pd.read_csv(path,low_memory=False)
   if part=='train' and split=='cold':
    training[task]={g:{'all':int(d.uniprot_id.eq(s[s.gene_symbol.eq(g)].uniprot_id.iloc[0]).sum()),'positive':int((d.uniprot_id.eq(s[s.gene_symbol.eq(g)].uniprot_id.iloc[0])&d.label.eq(1)).sum())} for g in GENES}
   if task!='binding' and split=='cold' and part=='val':calibration[task]={'n_rows':len(d),'classes':d.label.value_counts().to_dict(),'n_stored_chemicals':d.smiles.nunique(),'n_targets':d.uniprot_id.nunique()}
   overlaps.append({'task':task,'split':split,'part':part,'path':str(path),'all_labels':info(d),'positive':info(d[d.label.eq(1)]),'putative_negative':info(d[d.label.eq(0)])})
# Per-target graph membership and same/opposite-direction annotation coverage.
# Pair overlaps ignore direction; explicitly keep that distinction visible.
for row in overlaps:
 d=pd.read_csv(row['path'],low_memory=False)
 d['graph']=d.smiles.map(graph)
 row['per_target']={}
 for gene in GENES:
  e=s[s.gene_symbol.eq(gene)]
  m=d[d.uniprot_id.eq(e.uniprot_id.iloc[0])]
  gp=set(zip(m['graph'],m.uniprot_id))
  ep=set(zip(e.graph_group,e.uniprot_id))
  row['per_target'][gene]=len(gp&ep)
 for lab,name in [(1,'positive'),(0,'putative_negative')]:
  gpos=set(zip(d.loc[d.label.eq(lab),'graph'],d.loc[d.label.eq(lab),'uniprot_id']))
  hit=np.array([k in gpos for k in zip(s.graph_group,s.uniprot_id)])
  row[name]['external_annotation_rows_with_pair_overlap']=int(hit.sum())
  if row['task']!='binding':
   same=s.true_direction.eq(row['task']).to_numpy()
   row[name]['external_same_direction_annotations']=int((hit&same).sum())
   row[name]['external_opposite_direction_annotations']=int((hit&~same).sum())
 if row['split']=='cold':
  allgp=set(zip(d['graph'],d.uniprot_id))
  s[f"graph_pair_in_{row['task']}_cold_{row['part']}"]=[k in allgp for k in zip(s.graph_group,s.uniprot_id)]
for target in results:
 d=s[s.gene_symbol.eq(target['gene'])];y=d.true_direction.eq('activation').to_numpy().astype(int)
 target['fixed_threshold_sensitivity']={str(t):dict(zip(keys,metrics(y,d.S_direction.to_numpy()-t).tolist())) for t in [-1,0,1]}

# Background provenance membership only; no source label remapping/re-audit.
source_membership=[]
for name in ['all_cpi_clean_final_v4.csv','all_cpi_clean_final_v3_cleaned.csv']:
 path=P/'data/interaction_type'/name;d=pd.read_csv(path,low_memory=False)
 triples=set(zip(d.smiles,d.uniprot_id,d.effect_type))
 source_membership.append({'file':str(path),'n_eight_annotation_keys_present':sum(k in triples for k in zip(s.smiles,s.uniprot_id,s.effect_type))})
cand=pd.read_csv(E/'external_drug_effect_candidates_pairclean_strict_dual_cold.csv');cand=cand[cand.gene_symbol.isin(GENES)]
assert set(zip(cand.smiles,cand.uniprot_id,cand.effect_type))==set(zip(s.smiles,s.uniprot_id,s.effect_type))
out={'scope':'eight proteins only, archived annotation labels and fixed predictions; conditional descriptive statistics', 'versions':{'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'rdkit':rdBase.rdkitVersion},'input_scores':str(E/'effect_direction_scores_v2_external_all56.csv'),'calibration_file':str(E/'platt_calibration_params_v2_external_all56.txt'),'checks':checks,'denominators':{'candidate_rows':len(cand),'scored_annotation_rows':len(s),'unique_pairs':len(extp),'unique_graph_protein_pairs':len(extgp),'stored_chemicals':len(extc),'graph_groups':len(extgc),'dual_pairs':int(s[s.dual_direction_pair].drop_duplicates(['smiles','uniprot_id']).shape[0]),'source_counts':s.data_source.value_counts().to_dict(),'original_source_counts':s.original_source.value_counts().to_dict(),'exact_triple_duplicates_removed':len(cand)-len(s)},'calibration_sets':calibration,'target_training':training,'overlaps':overlaps,'source_membership':source_membership,'targets':results}
s.to_csv(O/'八蛋白实际使用逐注释分数.csv',index=False)
(O/'八蛋白统计结果.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out['denominators'],ensure_ascii=False));print(checks)
