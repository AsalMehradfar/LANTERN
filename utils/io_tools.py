import pickle
import yaml


def save_pickle(data, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        config_yaml = yaml.safe_load(file)
    return config_yaml