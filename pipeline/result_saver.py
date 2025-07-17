import os
import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_predictions_with_labels(preds_list, labels_list, idxs_list, name_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for preds, labels, idxs, name in zip(preds_list, labels_list, idxs_list, name_list):
        df = pd.DataFrame({
            'idx': idxs.flatten(),
            'true': labels.flatten(),
            'pred': preds.flatten()
        })
        df.to_csv(os.path.join(save_dir, f'{name.lower()}_results.csv'), index=False)


def compute_and_save_metrics(preds_list, labels_list, name_list, save_path, num_cls=6, num_floats=4):
    with open(save_path, 'w') as f:
        f.write("📊 Metrics Computation\n")
        for preds, labels, name in zip(preds_list, labels_list, name_list):
            f.write('\n')
            f.write(f'{name}:\n')
            metrics = compute_metrics(preds, labels, num_cls=num_cls, num_floats=num_floats)
            for key, val in metrics.items():
                f.write(f'{key}: {val}\n')


def save_all_evaluation_outputs(train_preds, val_preds, test_preds,
                                train_labels, val_labels, test_labels,
                                train_idxs, val_idxs, test_idxs,
                                config):
    results_dir = os.path.join("results", config["pipeline_save_name"])
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    compute_and_save_metrics(
        preds_list=[train_preds, val_preds, test_preds],
        labels_list=[train_labels, val_labels, test_labels],
        name_list=["Train", "Val", "Test"],
        save_path=os.path.join(results_dir, "metrics.txt")
    )

    # Save predictions and labels with indices
    save_predictions_with_labels(
        preds_list=[train_preds, val_preds, test_preds],
        labels_list=[train_labels, val_labels, test_labels],
        idxs_list=[train_idxs, val_idxs, test_idxs],
        name_list=["Train", "Val", "Test"],
        save_dir=results_dir
    )


def compute_metrics(preds, labels, num_cls=6, num_floats=4):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    results = {}

    results['R2 Score'] = round(r2_score(labels, preds), num_floats)
    results['RMSE'] = round(np.sqrt(np.mean((preds - labels) ** 2)), num_floats)
    results['MAE'] = round(np.mean(np.abs(preds - labels)), num_floats)
    results['Correlation'] = round(float(pearsonr(preds.flatten(), labels.flatten()).statistic), num_floats)
    results['Acc 0'] = round(cal_acc(preds, labels, num_cls=num_cls, tr=0), num_floats)
    results['Acc 1'] = round(cal_acc(preds, labels, num_cls=num_cls, tr=1), num_floats)

    return results


def cal_acc(preds, labels, num_cls=6, tr=1):
    s1, s2 = convert_to_class(labels, preds, num_cls)
    tmp = [1 if abs(x-y) <= tr else 0 for x,y in zip(s1, s2)]
    return round(sum(tmp) / len(tmp) * 100, 2)


def convert_to_class(labels, preds, num_classes):
    label_p_list = []
    pred_p_list = []
    for i in range(num_classes - 1):
        label_p_list.append(np.percentile(labels, 100 / num_classes * (i + 1)))
        pred_p_list.append(np.percentile(preds, 100 / num_classes * (i + 1)))
    label_cls = [sum([1 for x in label_p_list if x > label]) for label in labels]
    pred_cls = [sum([1 for x in pred_p_list if x > pred]) for pred in preds]
    return label_cls, pred_cls