import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def convert_to_class(labels, preds, num_classes):
    label_p_list = [np.percentile(labels, 100 / num_classes * (i + 1)) for i in range(num_classes - 1)]
    pred_p_list = [np.percentile(preds, 100 / num_classes * (i + 1)) for i in range(num_classes - 1)]
    label_cls = [sum([1 for x in label_p_list if x > label]) for label in labels]
    pred_cls = [sum([1 for x in pred_p_list if x > pred]) for pred in preds]
    return label_cls, pred_cls


def get_confusion_matrix(pred, target, labels, normalize=True):
    cm = confusion_matrix(target, pred, normalize='true' if normalize else None)
    return pd.DataFrame(cm, index=labels, columns=labels)


def plot_confusion_matrix(pred, target, labels, normalize=True,
                          save_path=None, fig_size=(10, 10), save_format='svg',
                          title='Confusion Matrix', x_label='Predicted Top k Percentiles', 
                          y_label='Actual Top k Percentiles', show=False, precision=3,
                          annot=True, cmap='Blues', vmax=1, num_classes=None):
    
    df_cm = get_confusion_matrix(pred, target, labels, normalize)

    if fig_size is not None:
        plt.figure(figsize=fig_size)
    sn.heatmap(df_cm, annot=annot, fmt=f'.{precision}f', cmap=cmap, vmin=0, vmax=vmax)

    plt.xlabel(x_label, labelpad=20)
    plt.ylabel(y_label, labelpad=20)
    if title:
        plt.title(title, pad=15)

    if num_classes is not None:
        tmp = np.linspace(100, 0, num_classes + 1)
        tick_labels = [f'{int(x)}%' for x in tmp[:-1]] + ['Top']
        ticks = list(range(num_classes)) + [num_classes]
        plt.xticks(ticks=ticks, labels=tick_labels)
        plt.yticks(ticks=list(range(num_classes)), labels=tick_labels[:-1])

    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}.{save_format}', format=save_format, dpi=400, bbox_inches='tight')
        print(f"✅ Saved confusion plot to: {save_path}.{save_format}")
    if show:
        plt.show()
    plt.close()


def create_and_plot_confusion_matrix(labels, preds, num_classes=6, fig_size=(13, 12), 
                                     axis_labels=None, save_path=None, show=False, 
                                     save_format='jpg', vmax=1, title='Confusion Matrix', 
                                     font_scale=3):
    sn.set_theme(style='whitegrid', context='paper', font_scale=font_scale)

    if axis_labels is None:
        axis_labels = [f'{i}' for i in range(num_classes)]

    label_cls, pred_cls = convert_to_class(labels, preds, num_classes)

    plot_confusion_matrix(
        pred_cls, label_cls, axis_labels,
        normalize=True,
        save_path=save_path,
        fig_size=fig_size,
        save_format=save_format,
        title=title,
        show=show,
        vmax=vmax,
        num_classes=num_classes
    )
