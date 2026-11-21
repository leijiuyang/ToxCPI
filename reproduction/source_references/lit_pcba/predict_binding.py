#!/usr/bin/env python3
"""Frozen task logits on every eligible pair; no calibration, binding gate or fitting."""
import argparse,hashlib,json,time
import h5py,numpy as np,pandas as pd,torch
import sys
sys.path.insert(0,'/home/data2/leijiuyang/Paper_revise/返修工作区/experiment/r3_07_decoy_evaluation/scripts')
from common import *
from resources import model
SOURCE=__import__('pathlib').Path('/home/data2/leijiuyang/Paper_revise/返修工作区/experiment/r3_07_decoy_evaluation')
ROOT=__import__('pathlib').Path('/home/data2/leijiuyang/Paper_revise/返修工作区/experiment/r3_07_decoy_evaluation/binding_head_litpcba_retrieval_v1')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--preload-chemical-cache',action='store_true');a=ap.parse_args()
    c=verify_frozen();torch.set_num_threads(4);torch.set_grad_enabled(False)
    if a.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.40,0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lock=json.loads((ROOT/'config/assay_decisions.lock.json').read_text());assert sha256(ROOT/'config/assay_decisions.csv')==lock['assay_decisions_sha256']
    specs=pd.read_csv(ROOT/'config/assay_decisions.csv',keep_default_na=False);counts=pd.read_csv(ROOT/'results/cleaning_counts.csv')
    roster=specs[specs.decision=='include'].merge(counts[['set_id','primary_n']],on='set_id').sort_values(['primary_n','set_id'])
    chemicals=pd.read_csv(SOURCE/'features/chemical_universe.csv.gz');feature_map=dict(zip(chemicals.graph_id,chemicals.feature_row));models={};runs=[]
    proteins=json.loads((ROOT/'config/input_proteins.json').read_text())
    with h5py.File(SOURCE/'features/chemical_tokens.h5','r') as chems,h5py.File(H5,'r') as wt:
        assert chems.attrs['chemical_universe_sha256']==sha256(SOURCE/'features/chemical_universe.csv.gz')
        status=chems['status'][:];lengths=chems['length'][:]
        token_data=chems['tokens'][:] if a.preload_chemical_cache else chems['tokens']
        print('chemical cache ready; preloaded=',a.preload_chemical_cache,flush=True)
        for s in roster.to_dict('records'):
            sid=s['set_id'];inp=ROOT/f'processed/{sid}.csv.gz';out=ROOT/f'predictions/{sid}.csv';receipt=ROOT/f'audit/inference_{sid}.json'
            if out.exists():
                if receipt.exists():
                    prior=json.loads(receipt.read_text());assert prior['prediction_sha256']==sha256(out);assert prior['input_sha256']==sha256(inp);runs.append(prior);print('reused',sid,flush=True);continue
                raise FileExistsError('Unreceipted prediction '+str(out))
            d=pd.read_csv(inp,keep_default_na=False);d['raw_logit']=np.nan;d['sigmoid_score']=np.nan;d['inference_status']='pending';t=time.monotonic();begin=now()
            key=s['protein_key'];uid=s['uniprot_id'];seq=proteins[key]['sequence']
            if s['variant']=='canonical':
                z=wt[uid][:768].astype(np.float32)
            else:
                with h5py.File(SOURCE/'features/variant_proteins.h5','r') as vh:
                    assert vh[key].attrs['sequence_sha256']==proteins[key]['sha256'];z=vh[key][:768].astype(np.float32)
            assert z.shape==(min(len(seq),768),1280)
            if 'binding' not in models:models['binding']=model('binding',a.device)
            m=models['binding'];prot=torch.from_numpy(z).to(a.device)
            for start in range(0,len(d),a.batch_size):
                part=d.iloc[start:start+a.batch_size];rows=[feature_map.get(g,-1) for g in part.graph_id]
                good=[];features=[]
                for offset,i in enumerate(rows):
                    if i<0 or status[i]!=1:d.loc[part.index[offset],'inference_status']='chemical_feature_missing';continue
                    n=int(lengths[i]);features.append(token_data[i,:n].astype(np.float32));good.append(offset)
                if not good:continue
                batch=len(good);longest=max(len(z) for z in features);ct=torch.zeros(batch,longest,768,device=a.device);mask=torch.ones(batch,longest,dtype=torch.bool,device=a.device)
                for j,zchem in enumerate(features):ct[j,:len(zchem)]=torch.from_numpy(zchem).to(a.device);mask[j,:len(zchem)]=False
                kwargs=dict(chem_tokens_feat=ct,chem_mask=mask,prot_residue_feat=prot.unsqueeze(0).expand(batch,-1,-1),prot_mask=torch.zeros(batch,len(prot),dtype=torch.bool,device=a.device))
                with torch.no_grad():logits=m(**kwargs)[0].detach().cpu().numpy().reshape(-1)
                if not np.isfinite(logits).all():raise RuntimeError('Nonfinite raw logits')
                ix=part.index[np.asarray(good)];d.loc[ix,'raw_logit']=logits;d.loc[ix,'sigmoid_score']=1/(1+np.exp(-logits.astype(np.float64)));d.loc[ix,'inference_status']='predicted'
                if (start+a.batch_size)//50000!=start//50000:print('predict',sid,min(start+a.batch_size,len(d)),'/',len(d),flush=True)
            d.to_csv(out,index=False,float_format='%.17g')
            ckpt=next(x for x in c['assets'] if x['role']=='binding')
            r={'set_id':sid,'task':s['task'],'scoring_head':'binding','label_endpoint':'functional assay activity','device':a.device,'batch_size':a.batch_size,'chemical_cache_preloaded':a.preload_chemical_cache,'begin_utc':begin,'end_utc':now(),'seconds':time.monotonic()-t,'input_sha256':sha256(inp),'prediction_sha256':sha256(out),'checkpoint_sha256':ckpt['sha256'],'input_pairs':len(d),'predicted':int((d.inference_status=='predicted').sum()),'failures':d[d.inference_status!='predicted'].inference_status.value_counts().to_dict(),'score':'raw binding-head logit ranked against functional-assay labels; sigmoid exported','protein_key':key,'sequence_sha256':proteins[key]['sha256'],'no_calibration_or_binding_gate':True}
            write_json(receipt,r);runs.append(r);print('completed',sid,len(d),'seconds',round(r['seconds'],2),flush=True)
    write_json(ROOT/'audit/inference_complete.json',{'utc':now(),'runs':runs,'all_intended_sets':len(roster),'completed_sets':len(runs),'prediction_rows':sum(x['predicted'] for x in runs)})
if __name__=='__main__':main()
