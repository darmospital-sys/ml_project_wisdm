"""
Métricas y visualizaciones de evaluación para HAR.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    activity_labels: dict[int, str],
    model_name: str = "Modelo",
) -> dict[str, float]:
    """
    Calcula métricas principales y muestra reporte de clasificación.

    Returns
    -------
    dict con accuracy, f1_macro, f1_weighted
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1-macro:    {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")
    print(f"{'='*50}\n")

    labels = [activity_labels[k] for k in sorted(activity_labels)]
    print(classification_report(y_true, y_pred, target_names=labels))

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    activity_labels: dict[int, str],
    model_name: str = "Modelo",
    save: bool = True,
) -> None:
    """
    Genera y guarda la matriz de confusión normalizada.
    """
    labels = [activity_labels[k] for k in sorted(activity_labels)]
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Proporción"},
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {model_name}")
    plt.tight_layout()

    if save:
        fname = model_name.lower().replace(" ", "_")
        FIGURES_DIR.mkdir(exist_ok=True)
        plt.savefig(FIGURES_DIR / f"cm_{fname}.png", dpi=300, bbox_inches="tight")

    plt.show()
