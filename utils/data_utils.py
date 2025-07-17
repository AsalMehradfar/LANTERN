import os
import random
from typing import Tuple, List
import random
import pandas as pd
import numpy as np
from .scaffold import scaffold_split

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from utils.io_tools import load_pickle 

RDLogger.DisableLog("rdApp.*")


DEFAULT_LABELS_PATH = "./data/Agile.csv"


def split_data(data_name, sizes: Tuple[float, float, float] = (0.72, 0.18, 0.1), split_type='random'):

    if not os.path.exists(f'data/splits/{data_name}'):
        os.makedirs(f'data/splits/{data_name}')

    data = pd.read_csv(f'data/{data_name}.csv')
    save_path = f'data/splits/{data_name}'

    if split_type == 'Murcko_scaffold':
        train, val, test = Murcko_scaffold_split(data, sizes[1], sizes[2])
    if split_type == 'scaffold_blanaced':
        train, val, test = scaffold_split(data, sizes=sizes, balanced=True)
    else:
        indices = list(range(len(data)))
        if split_type == 'random':
            random.shuffle(indices)
        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))

        train = indices[:train_size]
        val = indices[train_size:train_val_size]
        test = indices[train_val_size:]

    splits_indexes = [train, val, test]
    splits_indexes_numpy = np.asarray(splits_indexes, dtype=object)
    path = f'{save_path}/{split_type}.npy'
    np.save(path, splits_indexes_numpy, allow_pickle=True)


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)
    smiles_list = list(dataset.get('SMILES'))

    print("About to generate scaffolds")
    for ind, smiles in enumerate(smiles_list):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    print("Number of scaffold sets: %d" % len(scaffold_sets))
    for i, scaffold_set in enumerate(scaffold_sets):
        print("Scaffold set %d: %d molecules" % (i, len(scaffold_set)))

    return scaffold_sets


def Murcko_scaffold_split(dataset, valid_size, test_size):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


def inverse_transform(labels, features, scaler):
    """
    Inverse transform labels and/or features using the provided scaler.

    Returns:
        inv_labels: shape (N, label_dim)
        inv_features: shape (N, feature_dim)
    """
    labels = np.array(labels)
    features = np.array(features)

    combined = np.hstack((features, labels))
    inversed = scaler.inverse_transform(combined)

    label_dim = labels.shape[1]
    return inversed[:, -label_dim:], inversed[:, :-label_dim]


def load_features(path, feature_names):
    """Load pickled fingerprint dictionaries by name."""
    data = {}
    for name in feature_names:
        feature_path = os.path.join(path, f"{name}.pkl")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing feature file: {feature_path}")
        data[name] = load_pickle(feature_path)
    return data


def build_feature_vectors(features_dict, labeled_data, label_column=None):
    """Construct feature vectors and labels for each SMILES."""
    if label_column is None:
        label_column = "Target"
    has_labels = label_column in labeled_data.columns
    features, labels, idxs = [], [], []

    for idx, row in labeled_data.iterrows():
        smiles = row["SMILES"]
        combined = []
        for fname, fmap in features_dict.items():
            fp = fmap.get(smiles)
            if fp is None:
                raise ValueError(f"Missing fingerprint for {smiles} in {fname}")
            combined.extend(fp)
        features.append(combined)
        idxs.append(idx)
        if has_labels:
            labels.append(float(row[label_column]))

    if not has_labels:
        labels = []

    return features, labels, idxs


def get_num_features(features):
    lengths = [len(x) for x in features]
    if len(set(lengths)) != 1:
        raise ValueError("Inconsistent feature lengths!")
    return lengths[0]


def load_dataset_bundle(features_path, labels_path, feature_names, label_column=None):
    """Top-level loader for features, labels, idxs, and feature count."""
    features_dict = load_features(features_path, feature_names)

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Label file not found at: {labels_path}")
    labeled_data = pd.read_csv(labels_path)

    features, labels, idxs = build_feature_vectors(features_dict, labeled_data, label_column=label_column)
    num_features = get_num_features(features)

    return features, labels, idxs, num_features


def check_duplicate(features):
    """
    Raise an error if any two feature vectors are exactly the same.
    """
    seen = set()
    for i, f in enumerate(features):
        f_tuple = tuple(f)  # must be hashable to use in a set
        if f_tuple in seen:
            raise ValueError(f"Duplicate feature vector found at index {i}")
        seen.add(f_tuple)