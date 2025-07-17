import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

from utils.data_utils import split_data
from argparse import ArgumentParser

AVAILABLE_MODES = ['Murcko_scaffold', 'scaffold_balanced', 'random']


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--splits",
                        type=float,
                        nargs='+', 
                        default=[0.8, 0.1, 0.1],)
    parser.add_argument("--dataset", type=str, default='AGILE')
    parser.add_argument("--mode", type=str, default='all', choices=set(AVAILABLE_MODES + ['all']))

    args = parser.parse_args()

    if sum(args.splits) != 1:
        raise ValueError('Summation of splits should be 1!')
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == 'all':
        for mode in AVAILABLE_MODES:
            split_data(args.dataset, args.splits, split_type=mode)
    else:
        split_data(args.dataset, args.splits, split_type=args.mode)
