import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.manifold import TSNE

from pipeline.result_saver import compute_metrics
from utils.confusion_matrix_utils import create_and_plot_confusion_matrix


def plot_loss_curves(loss_dict, config, save=False, show=False, title="Training Loss Curve", figsize=(10, 6), font_scale=1.2):

    results_dir = os.path.join("results", config["pipeline_save_name"])
    os.makedirs(results_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    plt.title(title, fontsize=font_scale * 12)
    plt.xlabel("Epoch", fontsize=font_scale * 10)
    plt.ylabel("Loss", fontsize=font_scale * 10)

    for label, losses in loss_dict.items():
        plt.plot(losses, label=label)

    plt.legend(fontsize=font_scale * 9)
    plt.grid(True)

    plt.tight_layout()
    
    if save:
        save_path = os.path.join(results_dir, "loss_curve.jpg")
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        print(f"✅ Saved loss curve to: {save_path}")

    if show:
        plt.show()
    plt.close()


def generate_all_regression_plots(train_preds, val_preds, test_preds,
                                  train_labels, val_labels, test_labels,
                                  config,
                                  show=False,
                                  save_format="jpg",
                                  font_scale=3.2):

    save_folder = os.path.join("results", config["pipeline_save_name"])
    os.makedirs(save_folder, exist_ok=True)

    split_data = {
        "train": (train_preds, train_labels),
        "valid": (val_preds, val_labels),
        "test":  (test_preds, test_labels)
    }

    for split, (preds, labels) in split_data.items():
        generate_split_regression_plots(preds, labels, split, save_folder=save_folder,
                                    show=show,
                                    save_format=save_format,
                                    font_scale=font_scale)

    print(f"✅ Regression plots and confusion heatmaps saved to: {save_folder}")


def generate_split_regression_plots(preds, labels, split, save_folder=None,
                                    show=False,
                                    save_format="jpg",
                                    font_scale=3.2):
    """
    Generate prediction vs true plot and confusion heatmap for a single split (e.g., train, valid, test).
    """

    metrics = compute_metrics(preds, labels)
    rmse = metrics.get("RMSE", 0)
    acc_0 = metrics.get("Acc 0", 0)

    # Prediction vs Measurement Plot
    plot_prediction_vs_true(
        preds, labels,
        save_path=os.path.join(save_folder, f"prediction_vs_true_{split}"),
        fig_size=(12, 10),
        title=f"RMSE: {rmse:.3f}",
        point_size=80,
        add_color_bar=True,
        show=show,
        save_format=save_format,
        font_scale=font_scale
    )

    # Confusion-style Heatmap
    create_and_plot_confusion_matrix(
        labels, preds,
        save_path=os.path.join(save_folder, f"confusion_{split}"),
        show=show,
        save_format=save_format,
        title=f'Accuracy: {acc_0}%',
        fig_size=(13, 12),
        vmax=0.75,
        font_scale=font_scale
    )


def plot_prediction_vs_true(preds, labels, save_path=None, fig_size=(10, 10),
                                   save_format='svg', point_size=20,
                                   title='Prediction vs True Value', xy_line_color='r',
                                   x_label='True Value', y_label='Prediction',
                                   show=False, plot_line=True, cmap='viridis', 
                                   add_color_bar=False, grid=False, font_scale=3):
    
    sn.set_theme(style='whitegrid', context='paper', font_scale=font_scale)

    # Flatten arrays
    l = np.reshape(np.asarray(labels), (-1))
    p = np.reshape(np.asarray(preds), (-1))

    # Density estimation
    xy = np.vstack([l, p])
    z = gaussian_kde(xy)(xy)

    plt.figure(figsize=fig_size)
    ax = sn.scatterplot(x=l, y=p, hue=z, s=point_size, palette=cmap, legend=False)

    # Optional color bar
    if add_color_bar:
        norm = plt.Normalize(min(z), max(z))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = ax.figure.colorbar(sm, ax=ax)
        cbar.ax.get_yaxis().labelpad = 45
        cbar.ax.set_ylabel('Data Density', rotation=270)
        cbar.set_ticks([])
        cbar.ax.text(2.6, 0.97, 'High', ha='center', va='center', transform=cbar.ax.transAxes, color='black')
        cbar.ax.text(2.6, 0.02, 'Low', ha='center', va='center', transform=cbar.ax.transAxes, color='black')
        cbar.outline.set_edgecolor("black")  # Set the color to black
        cbar.outline.set_linewidth(1.5) 

    # Optional y = x line
    if plot_line:
        x = np.linspace(min(l), max(l), 1000)
        sn.lineplot(x=x, y=x, color=xy_line_color, alpha=0.8, 
                    linewidth=4, label='Perfect Prediction (y=x)')

    # Style axes
    ax = plt.gca() 
    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    plt.xlabel(x_label, labelpad=10)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title, pad=15)

    ticks = np.arange(round(min(l)) - 1, round(max(l)) + 1, 2)
    plt.xlim([round(min(l)) - 1, round(max(l)) + 1])
    plt.ylim([round(min(l)) - 1, round(max(l)) + 1])
    plt.xticks(ticks)
    plt.yticks(ticks)

    plt.grid(grid)
    if grid:
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')

    plt.tight_layout()

    if save_path:
        plt.savefig(f'{save_path}.{save_format}', format=save_format, dpi=400, bbox_inches='tight')
        print(f"✅ Saved regression plot to: {save_path}.{save_format}")

    if show:
        plt.show()
    plt.close()


def plot_tsne(features, labels, n_components=2, random_state=42, cmap='plasma', 
              fig_size=(12, 10), title=None, xlabel='Dimension 1', ylabel='Dimension 2',
              cbar_label='Transfection Efficiency', show=False, save_path=None, save_format='jpg',
              font_scale=3, point_size=20):

    sn.set_theme(style='whitegrid', context='paper', font_scale=font_scale)

    reduced = TSNE(n_components=n_components, random_state=random_state).fit_transform(features)
    df = pd.DataFrame(reduced, columns=["Dim1", "Dim2"])
    df["Label"] = labels

    plt.figure(figsize=fig_size)
    scatter = sn.scatterplot(
        data=df, x="Dim1", y="Dim2",
        hue="Label", palette=cmap,
        s=point_size, alpha=0.7, legend=False
    )

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel, labelpad=10)
    if ylabel:
        plt.ylabel(ylabel)

    norm = plt.Normalize(df["Label"].min(), df["Label"].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = scatter.figure.colorbar(sm, ax=scatter, label=cbar_label)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.outline.set_edgecolor("black")

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}.{save_format}", format=save_format, dpi=400, bbox_inches='tight')
        print(f"✅ Saved t-SNE plot to: {save_path}.{save_format}")
    if show:
        plt.show()
    plt.close()