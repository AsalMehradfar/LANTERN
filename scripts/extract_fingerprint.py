import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import pandas as pd
from rdkit import RDLogger  
from argparse import ArgumentParser
from deepchem.feat import CircularFingerprint, RDKitDescriptors

from utils.io_tools import save_pickle
RDLogger.DisableLog('rdApp.*')  


def generate_rdkit(smiles):
    """Generate RDKit features for all SMILES in the dataset."""
    featurizer = RDKitDescriptors()
    features = featurizer.featurize(smiles)
    return {smiles: list(feature) for smiles, feature in zip(smiles, features)}


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices={'expert', 'circular'})
    parser.add_argument("--data_name", type=str, default='AGILE')
    parser.add_argument("--save_path", type=str, default='data/fingerprints/AGILE')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    data = pd.read_csv(f'data/{args.data_name}.csv')
    smiles_list = data.get('SMILES').tolist()
    
    if args.mode == 'expert':
        feature_extractor = RDKitDescriptors()
    elif args.mode == 'circular':
        feature_extractor = CircularFingerprint(1024, is_counts_based=True, chiral=True)
    else:
        raise ValueError(f'{args.mode} is not a valid mode!')
    fingerprints = feature_extractor.featurize(smiles_list)
    
    results = {}
    for smiles, fp in zip(smiles_list, fingerprints):
        results[smiles] = fp
    print(fp.shape)

    if not os.path.exists(f'data/fingerprints/{args.data_name}'):
        os.makedirs(f'data/fingerprints/{args.data_name}')
    save_pickle(results, f'{args.save_path}/{args.mode}.pkl')

    