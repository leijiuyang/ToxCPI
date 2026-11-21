#!/usr/bin/env python3
"""Stage4 B entry: exact historical model; prefreeze; cached-row interventions.

No cohort readers, encoders, training, contact labels, or legacy output helpers.
Only worker B is writable. Completed scientific keys are never rerun.
"""
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '8'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import csv
import hashlib
import itertools
import json
import math
import pickle
import resource
import shutil
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

WORKER = Path(__file__).resolve().parents[1]
ROOT = WORKER.parents[1]
OUT = WORKER / 'outbox/B-E13-A0-0001-r001'
SRC = OUT / 'executed_source'
DEFS = ('scaled_primary', 'historical_unscaled_stable')
RATIOS = ('0.05', '0.10', '0.20')
SEMANTIC = 'original-BindingModelFusion-float32-eval-zero1280row-v1'
START = time.monotonic()


def now():
    return datetime.now(timezone.utc).isoformat()


def compact(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(x):
    return hashlib.sha256(compact(x).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size)


def verify(r):
    if ref(r['path']) != {k: r[k] for k in ('path', 'sha256', 'size_bytes')}:
        raise ValueError('frozen_hash_or_size_mismatch:' + r['path'])


def read(path):
    return json.loads(Path(path).read_text())


def safe(path):
    p = Path(path).resolve()
    if WORKER not in p.parents:
        raise ValueError('output_outside_worker_B:' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write(path, obj):
    p = safe(path)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def lines(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def append(path, obj):
    with safe(path).open('a') as f:
        f.write(compact(obj) + '\n')
        f.flush()
        os.fsync(f.fileno())


def table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with safe(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: compact(v) if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})


def status(stage, **kw):
    write(WORKER / 'status.json', dict(ticket_id='A-B-E13-A0-0001', pid=os.getpid(),
          attempt='r001', output=str(OUT), stage=stage, updated_at=now(),
          active_seconds_this_process=time.monotonic() - START, **kw))
    print(compact(dict(stage=stage, **kw)), flush=True)


def rng_info(case, namespace, definition, ratio, strategy, repeat):
    serial = compact([2026102704, case, namespace, definition, ratio, strategy, repeat])
    h = hashlib.sha256(serial.encode()).hexdigest()
    return dict(rng_serialization=serial, rng_sha256=h, derived_seed=int(h[:16], 16))


def mask_hash(indices):
    # Exact 768-bit mask, stored as 768 uint8 bytes (1=zeroed input row).
    a = bytearray(768)
    for i in indices:
        a[i] = 1
    return hashlib.sha256(a).hexdigest()


def main():
    if (OUT / 'receipt.json').exists():
        raise RuntimeError('batch_receipt_already_exists_no_more_model_operations')
    status('VERIFYING_INPUTS')
    ticket_path = ROOT / 'coordination/tickets/A-B-E13-A0-0001.json'
    ticket = read(ticket_path)
    design = read(ticket['execution_spec'])
    permit = read(ROOT / 'releases/first_batch_v0001/explanation_prediction_permission.json')
    verify(permit['ticket']); verify(permit['design']); verify(permit['model_checkpoint'])
    for k in ('model', 'case', 'coordinates', 'window_grid'):
        verify(design[k])
    case = read(design['case']['path'])
    model_spec = read(design['model']['path'])
    verify(model_spec['config_source'])
    for item in case['evidence']:
        verify(item)
    assert hashlib.sha256(case['protein_sequence'].encode()).hexdigest() == case['sequence_hash']
    assert len(case['protein_sequence']) == case['sequence_length'] == 919
    configs = dict(ticket=ticket, design=design, permit=permit, case=case, model=model_spec,
                   features=read(ROOT / 'shared/inputs/features_v0001.json'),
                   statistics_policy=read(ticket['statistics_policy']),
                   resources=read(ticket['resource_assignment']),
                   coordinate_contract=read(ROOT / 'shared/coordinates/E13_A0_contract_v0001.json'),
                   acceptance=read(ROOT / 'coordination/acceptance_contract_v0001.json'))
    code_ref = ref(__file__)
    archived = SRC / Path(__file__).name
    if not archived.exists():
        shutil.copyfile(__file__, safe(archived))
    assert sha(archived) == code_ref['sha256'], 'entry_changed_during_resume'
    for name in ('model_fusion.py', 'attention.py', 'config.py'):
        original = Path(model_spec['checkpoint']['path']).parents[1] / name
        assert sha(original) == sha(SRC / name), 'archived_model_source_mismatch'
    if not (OUT / 'actual_config.json').exists():
        write(OUT / 'actual_config.json', dict(**configs, device='cpu', cpu_threads=8,
              batch_size=1, chemistry_tensor_length=64, protein_tensor_length=768,
              precision='float32', semantic_version=SEMANTIC,
              mask_hash_encoding='768 uint8 bytes, tensor-index order; 1=zeroed row',
              input_provenance='Historical cache reuse; encoder input lineage unverified. Sequence coordinates are archived contract, not newly proven encoder lineage.',
              formal_statistics='PENDING_A_ACCEPTANCE; no population inference or significance tests',
              command=sys.argv, python_executable=sys.executable))
    else:
        assert read(OUT / 'actual_config.json')['design'] == design
    sources = [ref(p) for p in sorted(SRC.glob('*.py'))]
    if not (OUT / 'pre_run_archive.json').exists():
        write(OUT / 'pre_run_archive.json', dict(created_at=now(), code=sources,
              actual_config=ref(OUT / 'actual_config.json'),
              upstream=[ref(ticket_path), ref(ticket['execution_spec']), ref(ticket['statistics_policy']),
                        ref(ticket['resource_assignment']), ref(ROOT / 'shared/inputs/features_v0001.json'),
                        ref(ROOT / 'shared/coordinates/E13_A0_contract_v0001.json')]))
    else:
        for r in read(OUT / 'pre_run_archive.json')['code']:
            verify(r)
    import numpy as np
    import torch
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)  # initialization only, overwritten by strict checkpoint load
    sys.path.insert(0, str(SRC))
    from model_fusion import BindingModelFusion

    class CompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            return super().find_class(module.replace('numpy._core', 'numpy.core'), name)

    selected = {}
    cache_checks = []
    for item in configs['features']['cache_refs']:
        fid = item['feature_id']
        if fid not in ('tox_chem_tokens', 'tox_protein_residues'):
            continue
        status('VERIFYING_LOADING_CACHE', feature_id=fid)
        verify(item)
        path = Path(item['path'])
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        with path.open('rb') as f:
            bank = CompatibleUnpickler(f).load()
        key = case['chemical_feature_key' if fid == 'tox_chem_tokens' else 'protein_feature_key']
        if key not in bank:
            raise ValueError('required_cache_key_missing:' + key)
        array = bank[key]
        assert isinstance(array, np.ndarray) and array.ndim == 2
        assert array.shape[1] == item['expected_dimension'] and array.dtype.kind == 'f'
        assert np.isfinite(array).all()
        assert len(array) == (19 if fid == 'tox_chem_tokens' else 919), 'cached_row_sequence_length_mismatch'
        selected[fid] = array
        array_sha = hashlib.sha256(str(array.dtype).encode() + compact(list(array.shape)).encode() + array.tobytes()).hexdigest()
        assert before == (path.stat().st_size, path.stat().st_mtime_ns), 'cache_changed_during_load'
        cache_checks.append(dict(**item, key=key, shape=list(array.shape), dtype=str(array.dtype),
                                 selected_array_sha256=array_sha, finite=True, verified_full_file_sha256=True))
        del bank
    assert set(selected) == {'tox_chem_tokens', 'tox_protein_residues'}
    chem = torch.zeros(1, 64, 768, dtype=torch.float32)
    chem[0, :19] = torch.from_numpy(selected['tox_chem_tokens'].astype(np.float32))
    prot = torch.from_numpy(selected['tox_protein_residues'][:768].astype(np.float32)).unsqueeze(0)
    chem_mask = torch.arange(64).unsqueeze(0) >= 19
    prot_mask = torch.zeros(1, 768, dtype=torch.bool)
    del selected
    with Path(design['coordinates']['path']).open() as f:
        residues = list(csv.DictReader(f))
    retained = [r for r in residues if r['model_retained'] == '1']
    assert len(residues) == 919 and len(retained) == 768
    for i, r in enumerate(residues):
        assert int(r['dataset_pos1']) == i + 1 and r['aa'] == case['protein_sequence'][i]
        if i < 768:
            assert int(r['tensor_index0']) == i and int(r['absolute_block128']) == i // 128
            d = min(i, 767 - i)
            assert r['endpoint_bin'] == ('0_31' if d < 32 else '32_127' if d < 128 else 'ge128')
    with Path(design['window_grid']['path']).open() as f:
        grid = list(csv.DictReader(f))
    for row in grid:
        for k in ('length', 'start_dataset_pos1', 'end_dataset_pos1', 'first_tensor_index0',
                  'last_tensor_index0', 'construct_LBD_count', 'outside_construct_count', 'center_block128'):
            row[k] = int(row[k])
    assert Counter(g['length'] for g in grid) == {39: 93, 77: 88, 154: 78}
    for g in grid:
        s, e = g['first_tensor_index0'], g['last_tensor_index0']
        assert 0 <= s <= e < 768 and e - s + 1 == g['length']
        assert (s+1, e+1) == (g['start_dataset_pos1'], g['end_dataset_pos1'])
        counts = Counter(retained[i]['construct_marker'] for i in range(s, e+1))
        assert counts['LBD_construct_654_919'] == g['construct_LBD_count']
        assert counts['outside_construct_domain_unknown'] == g['outside_construct_count']
        center = (s + e) / 2
        d = min(center, 767 - center)
        assert g['center_block128'] == int(center // 128)
        assert g['center_endpoint_bin'] == ('0_31' if d < 32 else '32_127' if d < 128 else 'ge128')
    model = BindingModelFusion(chemberta_dim=768, esm2_dim=1280, hidden_dim=128,
             interaction_dim=256, dropout=0.3, use_residue_attention=True, use_toxprint=False,
             use_multi_granularity=False, use_interaction_cnn=False, use_molecular_gcn=False)
    ckpt = torch.load(model_spec['checkpoint']['path'], map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    ckpt = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}
    load = model.load_state_dict(ckpt, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert not any(m.training for m in model.modules())
    assert all(p.dtype == torch.float32 for p in model.parameters())
    del ckpt
    input_identity = digest(dict(case_sha256=design['case']['sha256'], cache_checks=cache_checks))
    physical = sum(x['event'] == 'START' for x in lines(OUT / 'attempts.jsonl'))
    technical = sum(x['event'] == 'START' and x['kind'] == 'technical' for x in lines(OUT / 'attempts.jsonl'))
    def forward(indices, kind, key):
        nonlocal physical, technical
        if physical >= 600 or (kind == 'technical' and technical >= 8):
            raise RuntimeError('physical_or_technical_budget_exhausted')
        if time.monotonic() - START >= 8*3600:
            raise RuntimeError('active_compute_budget_exhausted')
        physical += 1
        technical += int(kind == 'technical')
        attempt = dict(event='START', time=now(), physical_slot=physical, kind=kind, key=key,
                       mask_hash=mask_hash(indices), indices0=indices)
        append(OUT / 'attempts.jsonl', attempt)
        try:
            x = prot.clone()
            if indices:
                x[0, indices, :] = 0
            with torch.inference_mode():
                z = model(chem_tokens_feat=chem, chem_mask=chem_mask,
                          prot_residue_feat=x, prot_mask=prot_mask)[0]
            assert z.shape == (1,) and bool(torch.isfinite(z).all())
            value = float(z.item())
            append(OUT / 'attempts.jsonl', dict(**{**attempt, 'event': 'RESULT'}, time_end=now(),
                                              status='OK', logit=value))
            return value
        except Exception as exc:
            append(OUT / 'attempts.jsonl', dict(**{**attempt, 'event': 'RESULT'}, time_end=now(),
                                              status='FAILED', failure_reason=repr(exc)))
            raise

    status('STARTUP_CHECKS', peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    if (OUT / 'startup_checks.json').exists():
        checks = read(OUT / 'startup_checks.json')
        assert checks['status'] == 'PASS' and checks['input_identity_sha256'] == input_identity
        z0 = checks['baseline_logit']
    else:
        z0 = forward([], 'technical', 'baseline')
        empty = forward([], 'technical', 'empty_mask')
        checks = dict(status='PASS', at=now(), baseline_logit=z0, historical_logit=case['historical_logit'],
              absolute_baseline_error=abs(z0-case['historical_logit']), empty_mask_logit=empty,
              empty_mask_error=abs(empty-z0), strict_load_missing=list(load.missing_keys),
              strict_load_unexpected=list(load.unexpected_keys), input_identity_sha256=input_identity,
              cache_checks=cache_checks, coordinate_rows_checked=919, retained_rows=768,
              scientific_shape=[19,768], chemistry_padded_shape=[1,64,768], protein_shape=[1,768,1280],
              eval=True, BN_frozen=True, dropout=False, dtype='float32', torch_version=torch.__version__,
              numpy_version=np.__version__, python_version=sys.version,
              encoder_input_lineage='NOT_PROVEN; archived cache+coordinate contract reused without realignment')
        if checks['absolute_baseline_error'] > 1e-4 or checks['empty_mask_error'] > 1e-6:
            checks['status'] = 'FAILED'
        write(OUT / 'startup_checks.json', checks)
        assert checks['status'] == 'PASS', 'baseline_or_empty_mask_mismatch_no_substitution'
    p0 = float(1/(1+math.exp(-z0)))
    if not (OUT / 'preintervention_manifest.json').exists():
        status('EXPORTING_ORIGINAL_ATTENTION')
        # Encoder-only recomputation: no classifier/pair score; explicitly logged.
        with torch.inference_mode():
            hc, hp = model.encode_sequence_inputs(chem_tokens_feat=chem, prot_residue_feat=prot, chem_mask=chem_mask)
            multi = torch.matmul(torch.einsum('bna,had->bhnd', hc, model.interaction.W),
                                 hp.transpose(-1,-2).unsqueeze(1))[:, :, :19, :768]
            raw = multi.mean(dim=1)
            arrays = dict(raw_heads=multi[0].numpy(), raw_mean=raw[0].numpy(),
                          atom_dim=np.array(model.interaction.atom_dim), residue_dim=np.array(model.interaction.residue_dim))
            summaries = {}
            for definition in DEFS:
                scale = math.sqrt(model.interaction.residue_dim) if definition == DEFS[0] else 1
                reverse_scale = math.sqrt(model.interaction.atom_dim) if definition == DEFS[0] else 1
                fwd = torch.softmax(raw / scale, dim=-1)[0].numpy()
                rev = torch.softmax(raw.transpose(-1,-2) / reverse_scale, dim=-1)[0].numpy()
                summaries[definition] = fwd[case['atom_token_indices']].mean(axis=0)
                arrays[definition+'_forward'] = fwd
                arrays[definition+'_reverse'] = rev
                arrays[definition+'_summary'] = summaries[definition]
                assert np.isfinite(fwd).all() and np.isfinite(rev).all()
                assert np.allclose(fwd.sum(axis=-1), 1, atol=1e-6)
                assert np.allclose(rev.sum(axis=-1), 1, atol=1e-6)
        np.savez_compressed(safe(OUT/'original_attention.npz'), **arrays)
        write(OUT/'attention_export.json', dict(created_at=now(), encoder_only_recomputations=1,
              pair_score_slots=0, raw_shape=[19,768], raw_heads=4, atom_dim=128, residue_dim=128,
              primary_scale=math.sqrt(128), sensitivity_scale=1,
              summary_atom_rows=case['atom_token_indices'], special_rows_retained_in_full_arrays=[0,18]))
        roles, masks, proofs, rankings, windows = [], {}, [], [], []
        strata = defaultdict(list)
        for i, r in enumerate(retained):
            strata[(r['construct_marker'], i//128, r['endpoint_bin'])].append(i)
        strata = dict(sorted(strata.items()))
        def register(indices):
            idx = sorted(map(int, indices))
            assert len(idx) == len(set(idx)) and all(0 <= i < 768 for i in idx)
            h = mask_hash(idx)
            entry = dict(mask_hash=h, actual_indices0=idx, actual_dataset_positions1=[i+1 for i in idx], k=len(idx))
            key_parts = [model_spec['checkpoint']['sha256'], input_identity, h, 'float32', SEMANTIC]
            entry.update(scientific_key=digest(key_parts), scientific_key_parts=key_parts)
            assert h not in masks or masks[h] == entry
            masks[h] = entry
            return h
        def role(definition, ratio, strategy, repeat, indices, namespace='dispersed', **extra):
            r = dict(intervention_id=f'{namespace}|{definition}|{ratio}|{strategy}|{repeat}',
                     namespace=namespace, definition=definition, ratio=ratio,
                     k=math.ceil(float(ratio)*768), strategy=strategy, repeat=repeat,
                     mask_hash=register(indices) if indices is not None else None,
                     status='FROZEN' if indices is not None else 'NA_INSUFFICIENT',
                     rng_serialization=None, derived_seed=None, matching_status='not_applicable', **extra)
            roles.append(r)
            return r
        def sample_masks(definition, ratio, strategy, groups, total_space, info, namespace='dispersed'):
            # Enumeration is lexicographic in strata then position-combinations.
            if total_space < 20:
                combos = [list(itertools.combinations(candidates, quota)) for candidates, quota in groups]
                values = [sorted(itertools.chain.from_iterable(v)) for v in itertools.product(*combos)]
                for repeat, idx in enumerate(values):
                    r = role(definition, ratio, strategy, repeat, idx, namespace=namespace)
                    r.update(info, sampling_mode='finite_full_enumeration', feasible_space=str(total_space), proposals=0)
                return
            seen = set()
            proposals = 0
            for repeat in range(20):
                seed = rng_info(case['case_id'], namespace,
                                'shared' if strategy == 'U' else definition, ratio, strategy, repeat)
                rng = np.random.Generator(np.random.PCG64(seed['derived_seed']))
                while proposals < 10000:
                    proposals += 1
                    idx = sorted(int(i) for candidates, quota in groups
                                 for i in rng.choice(candidates, size=quota, replace=False))
                    key = tuple(idx)
                    if key not in seen:
                        seen.add(key)
                        r = role(definition, ratio, strategy, repeat, idx, namespace=namespace)
                        r.update(info, **seed, sampling_mode='distinct_uniform_random',
                                 feasible_space=str(total_space), proposals=proposals)
                        break
                else:
                    for missing in range(repeat,20):
                        r = role(definition, ratio, strategy, missing, None, namespace=namespace)
                        r.update(info, failure_reason='10000_proposal_limit', sampling_mode='partial_random')
                    break
        for definition in DEFS:
            scores = summaries[definition]
            high = sorted(range(768), key=lambda i: (-float(scores[i]), i))
            low = sorted(range(768), key=lambda i: (float(scores[i]), i))
            high_rank = {i:r+1 for r,i in enumerate(high)}
            low_rank = {i:r+1 for r,i in enumerate(low)}
            ties = Counter(float(x) for x in scores)
            for i in range(768):
                rankings.append(dict(case_id=case['case_id'], model_id=model_spec['model_id'], explainer_id=definition,
                     dataset_pos1=i+1, tensor_index0=i, aa=retained[i]['aa'], score=float(scores[i]),
                     rank=high_rank[i], low_rank=low_rank[i], tie=ties[float(scores[i])],
                     coordinate_status=retained[i]['reference_mapping_status'],
                     uniprot_pos1=retained[i]['uniprot_pos1'], construct_marker=retained[i]['construct_marker']))
            for ratio in RATIOS:
                k = math.ceil(float(ratio)*768)
                refs = {'H':high[:k], 'L':low[:k]}
                for strategy, idx in refs.items():
                    cutoff = float(scores[idx[-1]])
                    role(definition, ratio, strategy, 0, idx,
                         cutoff_score=cutoff, cutoff_tie_count=ties[cutoff],
                         cutoff_selected_tie_count=sum(float(scores[i]) == cutoff for i in idx))
                sample_masks(definition, ratio, 'U', [(list(range(768)),k)], math.comb(768,k), {})
                for strategy, reference in [('M_H',refs['H']),('M_L',refs['L'])]:
                    reference_set = set(reference)
                    evidence, groups = [], []
                    space = 1
                    for stratum, all_candidates in strata.items():
                        quota = sum(i in reference_set for i in all_candidates)
                        candidates = [i for i in all_candidates if i not in reference_set]
                        count = math.comb(len(candidates),quota) if len(candidates) >= quota else 0
                        space *= count
                        evidence.append(dict(stratum=list(stratum), quota=quota, candidate_count=len(candidates),
                            candidate_indices0=candidates, reference_indices0=[i for i in all_candidates if i in reference_set],
                            combinations=str(count), sufficient=len(candidates)>=quota))
                        groups.append((candidates,quota))
                    proof_id = f'{definition}|{ratio}|{strategy}'
                    proof = dict(proof_id=proof_id, definition=definition, ratio=ratio, strategy=strategy,
                         k=k, reference_indices0=sorted(reference), strata=evidence, feasible_space=str(space),
                         matching_status='feasible' if space else 'insufficient_joint_stratum',
                         attempted=True, rule='exact_joint_quota_excluding_reference_no_relaxation')
                    proofs.append(proof)
                    info = dict(proof_id=proof_id, strata_quota=[dict(stratum=e['stratum'],quota=e['quota']) for e in evidence],
                                candidate_count=[dict(stratum=e['stratum'],count=e['candidate_count']) for e in evidence],
                                matching_status=proof['matching_status'])
                    if space:
                        sample_masks(definition, ratio, strategy, groups, space, info)
                    else:
                        for repeat in range(20):
                            r = role(definition, ratio, strategy, repeat, None)
                            r.update(info, failure_reason='insufficient_joint_stratum', sampling_mode='not_drawn',
                                     feasible_space='0')
                # Scan and role selection use original attention only.
                candidates = [g for g in grid if g['length']==k]
                attrs = {g['window_id']:float(scores[g['first_tensor_index0']:g['last_tensor_index0']+1].mean()) for g in candidates}
                wh = min(candidates,key=lambda g:(-attrs[g['window_id']],g['start_dataset_pos1']))
                wl = min(candidates,key=lambda g:(attrs[g['window_id']],g['start_dataset_pos1']))
                for g in candidates:
                    idx=list(range(g['first_tensor_index0'],g['last_tensor_index0']+1))
                    windows.append(dict(**g, definition=definition, attribution=attrs[g['window_id']],
                                        mask_hash=register(idx), role='full_scan', matched_reference=None,
                                        overlap=None, start=g['start_dataset_pos1'],end=g['end_dataset_pos1']))
                def window_role(strategy,repeat,g,reference=None,**extra):
                    idx=list(range(g['first_tensor_index0'],g['last_tensor_index0']+1)) if g else None
                    r=role(definition,ratio,strategy,repeat,idx,namespace='window')
                    r.update(window_id=g['window_id'] if g else None, matched_reference=reference['window_id'] if reference else None,
                             overlap=(max(0,min(g['end_dataset_pos1'],reference['end_dataset_pos1'])-
                                          max(g['start_dataset_pos1'],reference['start_dataset_pos1'])+1) if g and reference else None),**extra)
                    return r
                window_role('H',0,wh); window_role('L',0,wl)
                for strategy,reference in [('U',None),('M_H',wh),('M_L',wl)]:
                    cs=candidates
                    proof_id=None
                    if reference:
                        keys=('construct_LBD_count','outside_construct_count','center_block128','center_endpoint_bin')
                        cs=[g for g in candidates if g['window_id']!=reference['window_id'] and all(g[x]==reference[x] for x in keys)]
                        proof_id=f'window|{definition}|{ratio}|{strategy}'
                        proofs.append(dict(proof_id=proof_id,definition=definition,ratio=ratio,strategy=strategy,
                          namespace='window',attempted=True,reference=reference['window_id'],
                          stratum={x:reference[x] for x in keys},candidate_count=len(cs),
                          candidate_window_ids=[g['window_id'] for g in cs],
                          matching_status='feasible' if cs else 'no_exact_window_candidates'))
                    if not cs:
                        window_role(strategy,0,None,reference,proof_id=proof_id,candidate_count=0,
                                    matching_status='no_exact_window_candidates',failure_reason='no_exact_window_candidates')
                    elif len(cs)<20:
                        for repeat,g in enumerate(cs):
                            window_role(strategy,repeat,g,reference,proof_id=proof_id,candidate_count=len(cs),
                                        matching_status='feasible',sampling_mode='finite_full_enumeration')
                    else:
                        seen=set()
                        for repeat in range(20):
                            seed=rng_info(case['case_id'],'window','shared' if strategy=='U' else definition,ratio,strategy,repeat)
                            rng=np.random.Generator(np.random.PCG64(seed['derived_seed']))
                            proposals=0
                            while True:
                                proposals+=1
                                choice=int(rng.choice(len(cs)))
                                if choice not in seen:
                                    seen.add(choice);break
                                assert proposals<10000, 'window_proposal_limit'
                            window_role(strategy,repeat,cs[choice],reference,proof_id=proof_id,
                                        candidate_count=len(cs),matching_status='feasible',
                                        sampling_mode='distinct_uniform_random',proposals=proposals,**seed)
        assert len(masks)<=572
        write(OUT/'frozen_roles.json',roles)
        write(OUT/'frozen_masks.json',list(masks.values()))
        write(OUT/'matching_proofs.json',proofs)
        table(OUT/'original_rankings.csv',rankings)
        table(OUT/'frozen_window_scan.csv',windows)
        table(OUT/'frozen_roles.csv',roles)
        freeze_paths=['frozen_roles.json','frozen_masks.json','matching_proofs.json','original_rankings.csv',
                      'frozen_window_scan.csv','frozen_roles.csv','original_attention.npz','attention_export.json',
                      'startup_checks.json','actual_config.json','pre_run_archive.json']
        write(OUT/'preintervention_manifest.json',dict(frozen_at=now(),before_first_intervention=True,
              technical_pair_scores_so_far=technical, scientific_pair_scores_so_far=0,
              code=sources, files=[ref(OUT/n) for n in freeze_paths],
              unique_scientific_masks=len(masks),role_rows=len(roles),window_scan_rows=len(windows),
              no_contact_labels_used=True, no_perturbed_scores_used=True,
              mask_hash_encoding='768 uint8 bytes; 1=zeroed row; tensor index ascending',
              rng_algorithm='NumPy Generator PCG64',
              finite_space_order='strata lexicographic, candidates ascending, combinations lexicographic',
              random_duplicate_policy='advance same repeat RNG on rejection; repeat increments only on accepted unique mask'))
        for n in freeze_paths+['preintervention_manifest.json']:
            (OUT/n).chmod(0o444)
    freeze=read(OUT/'preintervention_manifest.json')
    for item in freeze['files']+freeze['code']:
        verify(item)
    masks=read(OUT/'frozen_masks.json')
    roles=read(OUT/'frozen_roles.json')
    existing={r['scientific_key']:r for r in lines(OUT/'scientific_scores.jsonl') if r['status']=='OK'}
    terminal_failures={r['scientific_key']:r for r in lines(OUT/'scientific_scores.jsonl') if r['status']=='FAILED'}
    status('INTERVENTIONS',planned_unique=len(masks),completed=len(existing),physical_pair_slots=physical)
    for i,m in enumerate(masks):
        key=m['scientific_key']
        if key in existing or key in terminal_failures:
            continue
        row=dict(scientific_key=key,mask_hash=m['mask_hash'],original_z=z0,original_P=p0,
                 masked_z=None,masked_P=None,delta_z=None,delta_P=None,attempt=1,status='FAILED',failure_reason=None)
        try:
            zm=forward(m['actual_indices0'],'scientific',key)
            pm=float(1/(1+math.exp(-zm)))
            row.update(masked_z=zm,masked_P=pm,delta_z=z0-zm,delta_P=p0-pm,status='OK')
            existing[key]=row
        except Exception as exc:
            row['failure_reason']=repr(exc)
            terminal_failures[key]=row
        append(OUT/'scientific_scores.jsonl',row)
        if (i+1)%20==0 or i+1==len(masks):
            status('INTERVENTIONS',processed=i+1,planned_unique=len(masks),completed=len(existing),
                   failed=len(terminal_failures),physical_pair_slots=physical)
    # Frozen role rows always survive, including all planned infeasible repetitions.
    by_hash={m['mask_hash']:m for m in masks}
    exports=[]
    for r in roles:
        m=by_hash.get(r['mask_hash'])
        score=(existing.get(m['scientific_key']) or terminal_failures.get(m['scientific_key'])) if m else None
        row=dict(**r)
        row.update(case_id=case['case_id'],model_id=model_spec['model_id'])
        if m:
            row.update(m)
        if score:
            row.update(score)
        else:
            row.update(original_z=z0,original_P=p0,masked_z=None,masked_P=None,delta_z=None,delta_P=None,
                       attempt=0,status=r['status'] if not m else 'FAILED',
                       failure_reason=r.get('failure_reason','missing_scientific_result'))
        exports.append(row)
    table(OUT/'role_scores.csv',exports)
    table(OUT/'unique_scores.csv',list(existing.values())+list(terminal_failures.values()))
    with (OUT/'frozen_window_scan.csv').open() as f:
        windows=list(csv.DictReader(f))
    for row in windows:
        m=by_hash[row['mask_hash']]
        score=existing.get(m['scientific_key']) or terminal_failures.get(m['scientific_key'])
        row.update(score or dict(status='FAILED',failure_reason='missing_scientific_result'))
        row['scan_status']=row['status']
    table(OUT/'window_scan_scores.csv',windows)
    groups=defaultdict(list)
    for row in exports:
        groups[(row['namespace'],row['definition'],row['ratio'],row['strategy'])].append(row)
    counts=[]
    for key,rs in sorted(groups.items()):
        counts.append(dict(namespace=key[0],definition=key[1],ratio=key[2],strategy=key[3],
             frozen_rows=len(rs),completed_rows=sum(r['status']=='OK' for r in rs),
             NA_rows=sum(r['status'].startswith('NA') for r in rs),
             failed_rows=sum(r['status']=='FAILED' for r in rs),
             unique_scored_masks=len({r['mask_hash'] for r in rs if r['status']=='OK'}),
             sampling_modes=sorted({r.get('sampling_mode','deterministic_reference') for r in rs}),
             reasons=sorted({r['failure_reason'] for r in rs if r.get('failure_reason')})))
    table(OUT/'role_counts.csv',counts)
    status('VALIDATING_EXPORTS',completed=len(existing),physical_pair_slots=physical)
    assert len(windows)==518 and len({w['window_id'] for w in windows})==259
    assert all(r['status'] in ('OK','FAILED','NA_INSUFFICIENT') for r in exports)
    assert physical<=600 and len(existing)+len(terminal_failures)<=572
    assert len(existing)+len(terminal_failures)==len(masks)
    for row in existing.values():
        assert abs(row['delta_z']-(row['original_z']-row['masked_z']))<1e-12
        assert abs(row['delta_P']-(row['original_P']-row['masked_P']))<1e-12
    for item in freeze['files']+freeze['code']:
        verify(item)
    # Frozen method/input limitations and handoff; A owns statistics and acceptance.
    write(OUT/'handoff.json',dict(ticket_id=ticket['ticket_id'],status='EXECUTED_PENDING_A_ACCEPTANCE',
          matched_dispersed_proofs=read(OUT/'matching_proofs.json'),
          statistical_status='PENDING_A_ACCEPTANCE; frozen per-role raw data exported, no inferential statistics computed',
          limitations=['Single historically selected case and one fixed checkpoint.',
             'Input zeroing sensitivity, not deletion/mutation/reencoding/biological causality/contact truth.',
             'Historical encoder input lineage unverified; cache bytes and archived sequence coordinates checked.',
             'Overlapping windows correlated; window counts are not independent biological n.',
             'Infeasible matching remains NA with exact quota/candidate proof.'],
          coordinate_contract=configs['coordinate_contract'],central_assets_modified=False))
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
    output_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert peak<24 and output_bytes<2*1024**3
    manifest=[ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('receipt.json','artifact_manifest.json')]
    write(OUT/'artifact_manifest.json',dict(created_at=now(),files=manifest))
    receipt=dict(ticket_id=ticket['ticket_id'],worker='B',batch_id='B-E13-A0-0001-r001',
        status='COMPLETED_WITH_EXPLICIT_CONTROL_NA_PENDING_A_ACCEPTANCE' if any(r['NA_rows'] for r in counts) else 'COMPLETED_PENDING_A_ACCEPTANCE',
        completed_at=now(),acceptance='A_REVIEW_REQUIRED_NOT_SELF_CERTIFIED',
        commands=[read(OUT/'actual_config.json')['command']],python_executable=sys.executable,
        code=sources,inputs=[ref(ticket_path),ref(design['case']['path']),ref(design['model']['path']),
                            ref(ROOT/'shared/inputs/features_v0001.json')],
        actual_config=ref(OUT/'actual_config.json'),pre_run_archive=ref(OUT/'pre_run_archive.json'),
        preintervention_manifest=ref(OUT/'preintervention_manifest.json'),
        output_manifest=ref(OUT/'artifact_manifest.json'),baseline_check=checks,
        counts=dict(cases=1,checkpoints=1,trainings=0,searches=0,new_feature_extractions=0,new_vector_caches=0,
                    original_attention_exports=1,encoder_only_attention_recomputations=1,
                    physical_pair_forward_slots=physical,technical_pair_forward_slots=technical,
                    unique_scientific_masks=len(masks),successful_unique_scientific_scores=len(existing),
                    failed_unique_scientific_scores=len(terminal_failures),frozen_role_rows=len(roles),
                    role_OK_rows=sum(r['status']=='OK' for r in exports),
                    role_NA_rows=sum(r['status'].startswith('NA') for r in exports),
                    role_failed_rows=sum(r['status']=='FAILED' for r in exports),
                    window_grid_unique=259,window_definition_rows=518,
                    window_OK_rows=sum(w['status']=='OK' for w in windows),
                    negative_delta_z_unique=sum(r['delta_z']<0 for r in existing.values())),
        role_counts=counts,failures=list(terminal_failures.values()),
        control_insufficiency_proofs=ref(OUT/'matching_proofs.json'),
        resources=dict(device='cpu',model_processes=1,cpu_threads=8,GPU_processes=0,
                       peak_RSS_GiB=peak,active_seconds_this_process=time.monotonic()-START,output_bytes=output_bytes),
        scope='All frozen roles have results or explicit NA/failure. Full259 window scan exported under both definitions.',
        primary_contrast_status='See exact matching proof; unavailable controls remain NA. Statistics await A acceptance.',
        forbidden_operations_executed=[],old_assets_written=False,central_assets_written=False)
    write(OUT/'receipt.json',receipt)
    status('COMPLETED_PENDING_A_ACCEPTANCE',receipt=str(OUT/'receipt.json'),counts=receipt['counts'])


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('STOPPED_FAILURE',failure_reason=repr(exc))
        append(OUT/'entry_failures.jsonl',dict(at=now(),reason=repr(exc),traceback=traceback.format_exc()))
        raise
