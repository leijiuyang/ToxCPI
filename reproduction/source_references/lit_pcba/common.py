import hashlib, json, pathlib, datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
PROJECT=pathlib.Path('/home/data2/leijiuyang/1_TOXCPI/leijiuyang/test1')
PYTHON=PROJECT.parent/'anaconda3/envs/toxcpi/bin/python'
BERMOL=PROJECT.parent/'tipFormer2025-main/model/BerMol'
H5=PROJECT/'data/pretrained_features/human_proteome_esm2_residues.h5'
UNIVERSE=PROJECT/'explanation/downstream_ao_4chem/inputs/human_proteome_universe.csv'
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''): h.update(c)
    return h.hexdigest()
def write_json(path,obj):
    pathlib.Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def splits():
    for task in ['binding','activation','inhibition']:
        for setting in ['cold','warm']:
            for role in ['train','val','test']:
                path=PROJECT/'data/interaction_binding/splits'/f'{role}_{setting}.csv' if task=='binding' else PROJECT/'data/interaction_type_1to1_targetaware'/f'effect_{task}_{setting}_{role}.csv'
                yield task,setting,role,path
def verify_frozen():
    c=json.loads((ROOT/'config/frozen_protocol.json').read_text())
    for a in c['assets']:
        if sha256(a['path'])!=a['sha256']: raise RuntimeError('Frozen asset changed: '+a['path'])
    return c
