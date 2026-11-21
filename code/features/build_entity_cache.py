"""Build original max64/768 entity tensors from trusted frozen feature dictionaries."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from train_fusion import FusionDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'inference'))
from evaluate_fixed_checkpoint import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compound-features', type=Path, required=True, help='Trusted compound_features_bermol_tokens.pkl')
    parser.add_argument('--protein-features', type=Path, required=True, help='Trusted protein_features_esm2_residues.pkl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to replace an existing feature cache: {args.output}')
    with args.compound_features.open('rb') as handle:
        compounds = pickle.load(handle)
    with args.protein_features.open('rb') as handle:
        proteins = pickle.load(handle)
    tensors = FusionDataset.build_entity_tensors(compounds, proteins, max_chem_tokens=64,
        max_protein_length=768, build_chem_adj=False, chem_smiles_map=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, args.output)
    metadata = {'inputs_sha256': {str(p):sha256(p) for p in [args.compound_features,args.protein_features]},
        'output_sha256':sha256(args.output), 'max_chem_tokens':64, 'max_protein_length':768,
        'entity_counts':{'chemicals':len(tensors['chem_id2idx']), 'proteins':len(tensors['prot_id2idx'])},
        'implementation': 'existing FusionDataset.build_entity_tensors; no resplitting, labels or parameter training'}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
