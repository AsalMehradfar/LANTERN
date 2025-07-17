import pandas as pd
import numpy as np
import joblib
import os
import torch
from sklearn.preprocessing import MinMaxScaler
from utils.data_utils import load_dataset_bundle
from models.neural_models import FeedforwardRegressor, TransformerRegressor
from models.wrappers import SklearnModelWrapper, PytorchModelWrapper


def extract_features_and_labels(data_path, model_name, label_column=None):
    # Load the main CSV (with ID and maybe labels)
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    fingerprint_dir = os.path.join(os.path.dirname(data_path), "fingerprints", dataset_name)

    feature_names = extract_feature_names_from_model_name(model_name)

    features, labels, _, _ = load_dataset_bundle(
        features_path=fingerprint_dir,
        labels_path=data_path,
        feature_names=feature_names,
        label_column=label_column,
    )

    return np.array(features), np.asarray(labels)
    

def scale_features(features, scaler_path=None):
    dummy_labels = torch.zeros((len(features), 1))
    tmp = np.hstack((features, dummy_labels))
    
    if scaler_path:
        scaler = joblib.load(scaler_path)
    else:
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(features)
    tmp = scaler.transform(tmp)
    return tmp[..., :-1], scaler


def load_model(model_path, input_count, model_type, config):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    if model_type == 'sklearn':
        # Sklearn model
        model = joblib.load(model_path)
        return SklearnModelWrapper(model)

    else:
        # DL model
        model_name = model_path.split("-")[-1].split(".")[0]

        if model_name == "MLP":
            model = FeedforwardRegressor(input_count=input_count, **config.get("model_config", {}))
        elif model_name == "Transformer":
            model = TransformerRegressor(input_count=input_count, **config.get("model_config", {}))
        else:
            raise ValueError(f"Unsupported DL model type: {model_name}")

        model.load_state_dict(torch.load(model_path, map_location=config.get("device", "cpu")))
        model.to(config.get("device", "cpu"))
        model.eval()
        return PytorchModelWrapper(model, config)


def predict(model, features, model_type):
    if model_type == 'sklearn':
        return model.predict(features)
    elif model_type == 'pytorch':
        with torch.no_grad():
            return model.predict(features)


def save_metrics(metrics, save_path):
    with open(save_path, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def extract_feature_names_from_model_name(model_name):
    parts = model_name.split('/')[-1].split('-')
    if 'model' not in parts:
        raise ValueError(f"'model' keyword not found in model name: {model_name}")
    model_index = parts.index('model')
    return parts[:model_index]