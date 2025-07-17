import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd
from argparse import ArgumentParser

from utils.io_tools import load_pickle
from utils.visual_utils import plot_tsne


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--data_name', type=str, default='AGILE')
    parser.add_argument('--features', choices=['grover', 'circular', 'expert'], type=str, default='circular')
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    if args.save_path is None:
        args.save_path = args.features
    return args


if __name__ == "__main__":
    args = get_args()

    data = pd.read_csv(f'data/{args.data_name}.csv')
    fp_dict = load_pickle(f'data/fingerprints/{args.data_name}/{args.features}.pkl')

    features = [fp_dict[row['SMILES']] for _, row in data.iterrows()]
    labels = data['Target'].values

    features = np.asarray(features)
    plot_tsne(
        features, labels,
        n_components=2,
        random_state=42,
        cmap='plasma',
        fig_size=(12, 10),
        title=None,
        xlabel='Dimension 1',
        ylabel='Dimension 2',
        show=False,
        save_path=args.save_path,
        save_format='jpg',
        point_size=40
    )