"""Shared figure helpers so every chart in the report looks the same."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY, TEAL, TERRA, GREY = "#1F3864", "#0E7C86", "#C4622D", "#8A96A8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#5A6472", "axes.labelcolor": "#1E1E1E",
    "xtick.color": "#5A6472", "ytick.color": "#5A6472",
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def confusion(cm, classes, title, path):
    """Row-normalised confusion matrix with counts and row percentages."""
    cm = np.asarray(cm)
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.0%}", ha="center", va="center",
                    color="white" if pct[i, j] > 0.55 else "#1E1E1E", fontsize=9)
    ax.set_xticks(range(len(classes)), classes, rotation=15)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted status")
    ax.set_ylabel("Actual status")
    ax.set_title(title, fontsize=10, loc="left")
    ax.spines[:].set_visible(False)
    fig.savefig(path)
    plt.close(fig)


def roc_curves(curves, title, path):
    """curves: list of (label, fpr, tpr, auc)."""
    colors = [NAVY, TEAL, TERRA]
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for (label, fpr, tpr, auc), c in zip(curves, colors):
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{label} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def model_comparison(names, values, metric_label, target, path):
    """Horizontal bars for one metric across models, with the target line."""
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    y = np.arange(len(names))
    ax.barh(y, values, color=TEAL, height=0.55)
    for yi, v in zip(y, values):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=9)
    if target is not None:
        ax.axvline(target, color=TERRA, lw=1.5, ls="--")
        ax.text(target, -0.65, f" target {target:.2f}", color=TERRA, fontsize=8, va="bottom")
    ax.set_yticks(y, names)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric_label)
    ax.invert_yaxis()
    fig.savefig(path)
    plt.close(fig)
