"""
Carga de datos del dataset WISDM.

Funciones para cargar las señales crudas preprocesadas (ventanas de 90
timesteps x 3 ejes) y sus etiquetas de actividad.
"""

from pathlib import Path

import numpy as np

# Ruta por defecto
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

ACTIVITY_LABELS = {
    0: "Walking",
    1: "Jogging",
    2: "Sitting",
    3: "Standing",
    4: "Upstairs",
    5: "Downstairs",
}

# Nombres de los ejes del acelerómetro
AXIS_NAMES = ["x", "y", "z"]


def load_dataset(
    split: str = "train",
    data_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga señales y etiquetas para un split.

    Parameters
    ----------
    split : str
        'train' o 'test'.
    data_root : str o Path, opcional
        Ruta raíz donde están los archivos .npy.

    Returns
    -------
    X : np.ndarray, forma (n_samples, 90, 3)
        Ventanas de señal cruda (acelerómetro X, Y, Z).
    y : np.ndarray, forma (n_samples,)
        Etiquetas de actividad (0-5).
    """
    root = Path(data_root) if data_root else DATA_ROOT

    X = np.load(root / f"x_{split}.npy")
    y = np.load(root / f"y_{split}.npy")

    return X, y


def load_activity_labels() -> dict[int, str]:
    """
    Devuelve el mapeo de ID → nombre de actividad.

    Returns
    -------
    dict[int, str]
        {0: 'Walking', 1: 'Jogging', ...}
    """
    return ACTIVITY_LABELS.copy()


def get_dataset_info(
    data_root: str | Path | None = None,
) -> dict:
    """
    Devuelve un resumen del dataset (shapes, clases, balance).

    Returns
    -------
    dict
        Diccionario con información del dataset.
    """
    X_train, y_train = load_dataset("train", data_root)
    X_test, y_test = load_dataset("test", data_root)

    info = {
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "n_classes": len(ACTIVITY_LABELS),
        "n_timesteps": X_train.shape[1],
        "n_channels": X_train.shape[2],
        "train_class_counts": {
            ACTIVITY_LABELS[k]: int(np.sum(y_train == k))
            for k in sorted(ACTIVITY_LABELS)
        },
        "test_class_counts": {
            ACTIVITY_LABELS[k]: int(np.sum(y_test == k))
            for k in sorted(ACTIVITY_LABELS)
        },
    }
    return info
