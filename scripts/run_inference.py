import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import argparse
import os
from utils.inference_utils import * 
from utils.config import *
from pipeline.result_saver import compute_metrics
from utils.visual_utils import generate_split_regression_plots


parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", default="data/AGILE.csv")
parser.add_argument("--model_path", default="checkpoints/circular-expert-model-MLP.pth")
parser.add_argument("--scaler_path", default="checkpoints/circular-expert-model-MLP_scaler.pkl")
parser.add_argument("--model_type", choices=["sklearn", "pytorch"], default="pytorch")
parser.add_argument("--label_column", default="Target")
parser.add_argument("--save_path", default="results/")
args = parser.parse_args()

# Load data
features, labels = extract_features_and_labels(args.csv_path, args.model_path, args.label_column)

# Scale features
scaled_features, scaler = scale_features(features, args.scaler_path)

# Load model and predict
eval_config = load_eval_config()
model = load_model(args.model_path, features.shape[1], args.model_type, eval_config)
preds = predict(model, scaled_features, args.model_type)
preds = scaler.inverse_transform(np.hstack((features, preds)))[..., -1:].flatten().tolist()

dataset_name = os.path.splitext(os.path.basename(args.csv_path))[0]
if list(labels):
    # Save predictions
    os.makedirs(os.path.join(args.save_path, dataset_name), exist_ok=True)
    output = np.column_stack((preds, labels))  # shape: (N, 2)
    np.savetxt(
        os.path.join(args.save_path, dataset_name, "predictions.csv"),
        output,
        delimiter=",",
        header="Prediction,Label",
        comments='',  # Remove the '#' comment character in the header
        fmt="%.6f"    # Format numbers to 6 decimal places
    )

    metrics = compute_metrics(preds, labels)
    save_metrics(metrics, os.path.join(args.save_path, dataset_name, "metrics.txt"))

    generate_split_regression_plots(preds, labels, 'test', os.path.join(args.save_path, dataset_name))
    
else:
    os.makedirs(os.path.join(args.save_path, dataset_name), exist_ok=True)
    np.savetxt(
        os.path.join(args.save_path, dataset_name, "predictions.csv"), 
        preds, 
        delimiter=",",
        header="Prediction",
        comments='',
        fmt="%.6f" 
    )