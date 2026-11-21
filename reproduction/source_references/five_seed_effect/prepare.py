"""Create an isolated, auditable extension of the historical S7 runner."""
from pathlib import Path
import shutil, json, difflib

ROOT = Path(__file__).resolve().parents[1]
ORIG = Path('/home/data2/leijiuyang/1_TOXCPI/leijiuyang/test1')
DEST = ROOT/'code'/'test1'
for f in list(ORIG.glob('*.py')) + [ORIG/'experiments'/n for n in ['train_fusion.py','train_effect.py','ablation/run_s3_ablation.py','ablation/s3_models.py']]:
    d=DEST/f.relative_to(ORIG); d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,d)
runner=DEST/'experiments/ablation/run_s3_ablation.py'
old=runner.read_text(); s=old
s=s.replace('SEED = 42', 'SEED = None')
s=s.replace('PROTOCOL_VERSION = "s3-controlled-v2-fixed-threshold-0.5"', 'PROTOCOL_VERSION = "effect-cold-five-seed-20261112-v1"')
s=s.replace('OUT = HERE', 'ROOT = Path('+repr(str(ROOT))+')\nORIGINAL = Path('+repr(str(ORIG))+')\nOUT = ROOT')
s=s.replace('FEATURE_DIR = TEST1 / "data"', 'FEATURE_DIR = ORIGINAL / "data"')
s=s.replace('base = TEST1 / "data"', 'base = ORIGINAL / "data"')
s=s.replace('map_location="cpu", weights_only=False)', 'map_location="cpu", weights_only=False, mmap=True)')
s=s.replace('"entity_cache_size": cache.stat().st_size,', '"entity_cache_size": cache.stat().st_size,\n        "frozen_protocol_sha256": sha256_file(ROOT / "实验协议冻结.json"),\n        "source_manifest_sha256": sha256_file(ROOT / "audit/source_before.json"),')
s=s.replace('return str(Path(path).resolve().relative_to(TEST1.parent))', 'return str(Path(path).resolve())')
s=s.replace('def main():\n', 'def main():\n    global SEED, OUT\n')
s=s.replace('parser.add_argument("--task",', 'parser.add_argument("--seed", type=int, required=True, choices=[42,1009,2027,3407,7919])\n    parser.add_argument("--task",')
s=s.replace('    args = parser.parse_args()','    args = parser.parse_args()\n    SEED = args.seed\n    if args.task == "binding" or args.split != "cold":\n        parser.error("This extension is restricted to cold-chemical effect tasks")\n    if args.smoke_only:\n        OUT = ROOT / "smoke"')
a=s.index('    if args.variant == "full":\n',s.index('    best_epoch = None'))
b=s.index('        optimizer = torch.optim.AdamW',a)
c=s.index('    # Persist any provenance',b)
s=s[:a]+''.join(line[4:] if line.startswith('    ') else line for line in s[b:c].splitlines(True))+s[c:]
s=s.replace('        model.load_state_dict(resume["model_state"])', '        if resume.get("protocol_fingerprint") != fingerprint:\n            raise RuntimeError("Resume fingerprint mismatch: task/variant/seed/data/features/config/code")\n        model.load_state_dict(resume["model_state"])')
s=s.replace('"epoch": epoch, "model_state": model.state_dict(),','"protocol_fingerprint": fingerprint,\n            "epoch": epoch, "model_state": model.state_dict(),')
s=s.replace('torch.cuda.set_rng_state_all(resume["cuda_rng"])', 'torch.cuda.set_rng_state_all([x.cpu() for x in resume["cuda_rng"]])')
s=s.replace('loader_generator.set_state(resume["loader_rng"])','loader_generator.set_state(resume["loader_rng"].cpu())')
# Best and resume saves are atomic, preserving the last complete epoch on interruption.
s=s.replace('torch.save(', 'atomic_torch_save(')
insert='''
def atomic_torch_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def state_digest(model):
    h = hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

'''
s=s.replace('def main():\n',insert+'def main():\n')
s=s.replace('    local_present, raw_present = branch_flags', '    initial_state_sha256 = state_digest(model)\n    logger.info("INDEPENDENT INITIALIZATION seed=%d state_sha256=%s", SEED, initial_state_sha256)\n    local_present, raw_present = branch_flags')
s=s.replace('"experiment_id": paths["id"],', '"independent_training": True, "initial_state_sha256": initial_state_sha256,\n        "experiment_id": paths["id"],')
s=s.replace('"seed": SEED, "validation_threshold": threshold,', '"independent_training": True, "initial_state_sha256": initial_state_sha256,\n        "seed": SEED, "validation_threshold": threshold,')
s=s.replace('def prediction_frame(frame, labels, probs, threshold):','def prediction_frame(frame, labels, probs, threshold, logits, task, variant):')
s=s.replace('"source_row_index": frame["source_row_index"].astype(int),','"record_id": [f"{task}:cold:{int(i)}" for i in frame["source_row_index"]],\n        "task": task, "variant": variant, "seed": SEED,\n        "source_row_index": frame["source_row_index"].astype(int),\n        "smiles": frame["smiles"].astype(str),\n        "actual_chemical_input_id": frame["chemical_name"].astype(str),\n        "raw_logit": np.asarray(logits).astype(float),')
s=s.replace('val_metrics["probs"], threshold)', 'val_metrics["probs"], threshold, val_metrics["logits"], args.task, args.variant)')
s=s.replace('test_metrics["probs"], threshold)', 'test_metrics["probs"], threshold, test_metrics["logits"], args.task, args.variant)')
# Full batch forward/backward, checkpoint/RNG roundtrip without changing formal training.
a=s.index('        batch = next(iter(loaders["val"]))',s.index('    if args.smoke_only:'))
b=s.index('    best_epoch = None',a)
s=s[:a]+'''        batch = next(iter(loaders["train"]))
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        from train_fusion import _forward_batch
        for _ in range(3):
            model.train(); opt.zero_grad()
            z, *_ = _forward_batch(model, batch, device)
            y = batch["label"].to(device)*(1-cfg["label_smoothing"])+0.5*cfg["label_smoothing"]
            loss = nn.functional.binary_cross_entropy_with_logits(z,y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        torch.cuda.synchronize()
        report = dict(task=args.task,variant=args.variant,batch_size=len(z),loss=float(loss),
                      peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                      three_steps_seconds=time.time()-t,initial_state_sha256=initial_state_sha256)
        with open(OUT/(paths["id"]+"_benchmark.json"),'w') as f: json.dump(report,f,indent=2)
        logger.info("SMOKE OK %s",report)
        return

'''+s[b:]
runner.write_text(s)
helper=DEST/'experiments/train_fusion.py'; h0=helper.read_text();h=h0
# Only extend evaluation exports; all training, padding, masking, loss, EMA remain byte-equivalent.
a=h.index('def evaluate('); b=h.index('\ndef analyze_gate_weights',a)
ev=h[a:b].replace('    all_probs = []','    all_probs = []\n    all_logits = []').replace('            probs = torch.sigmoid(logits)','            all_logits.extend(logits.cpu().numpy())\n            probs = torch.sigmoid(logits)').replace("        'labels': all_labels,","        'logits': np.asarray(all_logits),\n        'labels': all_labels,")
h=h[:a]+ev+h[b:];helper.write_text(h)
(ROOT/'audit').mkdir(exist_ok=True)
(ROOT/'audit/code_changes.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(ORIG/'experiments/ablation/run_s3_ablation.py'),tofile=str(runner)))+'\n'+''.join(difflib.unified_diff(h0.splitlines(True),h.splitlines(True),fromfile='original/train_fusion.py',tofile='isolated/train_fusion.py')))
print(ROOT)
