import torch
from datetime import datetime

from utils.io_tools import load_yaml


DEFAULT_TRAIN_CONFIG_PATH = "./config/train_config.yaml"
DEFAULT_EVAL_CONFIG_PATH = "./config/eval_config.yaml"
DEFAULT_VISUAL_CONFIG_PATH = "./config/visual_config.yaml"


from datetime import datetime
import torch

def load_train_config(configpath=DEFAULT_TRAIN_CONFIG_PATH):
    train_config = load_yaml(configpath)

    # Set device
    requested_device = train_config.get("device", "cpu")
    if requested_device == "cuda":
        train_config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Compose save names
    features_str = "-".join(train_config.get("features", []))
    model_name = train_config.get("model_config", {}).get("model", "unknown")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')

    model_save_name = f"{features_str}-model-{model_name}"
    pipeline_save_name = f"{model_save_name}-{timestamp}"

    train_config["model_save_name"] = model_save_name
    train_config["pipeline_save_name"] = pipeline_save_name

    print(f"✅ Generated pipeline save name: {pipeline_save_name}")

    return train_config


def load_visual_config(configpath=DEFAULT_VISUAL_CONFIG_PATH):
    return load_yaml(configpath)


def load_eval_config(configpath=DEFAULT_EVAL_CONFIG_PATH):
    eval_config = load_yaml(configpath)

    # Set device
    requested_device = eval_config.get("device", "cpu")
    if requested_device == "cuda":
        eval_config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    return eval_config