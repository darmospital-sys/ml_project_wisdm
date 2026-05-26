"""
Modelos clásicos de ML para HAR — Random Forest y SVM.

Para modelos clásicos necesitamos primero convertir la señal cruda
(n_samples, 90, 3) a una matriz 2D de features estadísticas (n_samples, 33).
Eso lo hace src/features.py. Estos modelos trabajan sobre esa representación.
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Entrena un Random Forest sobre features estadísticas.

    Parameters
    ----------
    X_train : np.ndarray, forma (n_samples, n_features)
        Features estadísticas extraídas de la señal cruda.
    y_train : np.ndarray, forma (n_samples,)
        Etiquetas de actividad.
    n_estimators : int
        Número de árboles.

    Returns
    -------
    RandomForestClassifier entrenado
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 0.1,
    max_iter: int = 2000,
    random_state: int = 42,
) -> LinearSVC:
    """
    Entrena un SVM lineal (LinearSVC) sobre features estadísticas.

    Se usa LinearSVC en lugar de SVC(kernel='rbf') porque con ~44k
    muestras el kernel RBF tarda horas. LinearSVC es O(n) y produce
    resultados comparables con features bien normalizadas.

    Parameters
    ----------
    X_train : np.ndarray, forma (n_samples, n_features)
        Features ya normalizadas (StandardScaler).
    C : float
        Parámetro de regularización (inverso a la fuerza del regularizador).

    Returns
    -------
    LinearSVC entrenado
    """
    svm = LinearSVC(C=C, max_iter=max_iter, random_state=random_state)
    svm.fit(X_train, y_train)
    return svm


def save_model(model: object, name: str) -> Path:
    """Guarda un modelo sklearn con joblib."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str) -> object:
    """Carga un modelo sklearn guardado con joblib."""
    return joblib.load(MODELS_DIR / f"{name}.joblib")
