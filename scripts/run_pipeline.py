import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
from pipeline.run_pipeline_from_config import run_pipeline_from_config
from utils.utils import seed_everything


def launch_pipeline():
    parser = argparse.ArgumentParser(description="Run the Lantern training and evaluation pipeline.")
    parser.add_argument(
        '--config', 
        default="./config/train_config.yaml", 
        help="Path to training config file"
    )
    parser.add_argument(
        '--seed', 
        default=23, 
        type=int, 
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    run_pipeline_from_config(args.config)


if __name__ == '__main__':
    launch_pipeline()

