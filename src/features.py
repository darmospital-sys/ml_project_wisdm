"""
Extracción y transformación de features para WISDM HAR.

A diferencia de UCI HAR (que ya trae 561 features precalculadas),
WISDM trabaja con señal cruda (90 timesteps x 3 ejes). Este módulo
permite extraer features estadísticas manualmente o preparar los datos
para modelos de deep learning que consumen la señal directamente.
"""

import numpy as np
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Extracción de features estadísticas (para modelos clásicos)
# ---------------------------------------------------------------------------

def extract_statistical_features(
    signal: np.ndarray, prefix: str = "sig"
) -> dict[str, float]:
    """
    Extrae features estadísticas de una señal 1D (un eje, una ventana).

    Parameters
    ----------
    signal : np.ndarray, forma (n_timesteps,)
        Señal univariada de un eje del acelerómetro.
    prefix : str
        Prefijo para los nombres de las features.

    Returns
    -------
    dict[str, float]
        Diccionario {nombre_feature: valor}.
    """
    return {
        f"{prefix}_mean":     float(np.mean(signal)),
        f"{prefix}_std":      float(np.std(signal)),
        f"{prefix}_mad":      float(np.mean(np.abs(signal - np.mean(signal)))),
        f"{prefix}_max":      float(np.max(signal)),
        f"{prefix}_min":      float(np.min(signal)),
        f"{prefix}_range":    float(np.max(signal) - np.min(signal)),
        f"{prefix}_energy":   float(np.sum(signal ** 2)),
        f"{prefix}_iqr":      float(np.percentile(signal, 75) - np.percentile(signal, 25)),
        f"{prefix}_skewness": float(sp_stats.skew(signal)),
        f"{prefix}_kurtosis": float(sp_stats.kurtosis(signal)),
        f"{prefix}_rms":      float(np.sqrt(np.mean(signal ** 2))),
    }


def extract_features_from_windows(
    X: np.ndarray,
    axis_names: list[str] | None = None,
) -> np.ndarray:
    """
    Extrae features estadísticas de todas las ventanas del dataset.

    Transforma X de forma (n_samples, 90, 3) a una matriz 2D
    (n_samples, n_features) apta para modelos clásicos (RF, SVM, etc.).

    Parameters
    ----------
    X : np.ndarray, forma (n_samples, n_timesteps, n_channels)
        Ventanas de señal cruda.
    axis_names : list[str], opcional
        Nombres de los canales. Por defecto ['x', 'y', 'z'].

    Returns
    -------
    np.ndarray, forma (n_samples, n_channels * 11)
        Matriz de features estadísticas.
    """
    if axis_names is None:
        axis_names = ["x", "y", "z"]

    rows = []
    for i in range(X.shape[0]):
        sample_feats = {}
        for ch_idx, ch_name in enumerate(axis_names):
            signal = X[i, :, ch_idx]
            feats = extract_statistical_features(signal, prefix=f"acc_{ch_name}")
            sample_feats.update(feats)
        rows.append(list(sample_feats.values()))

    return np.array(rows, dtype=np.float32)


def get_feature_names(
    axis_names: list[str] | None = None,
) -> list[str]:
    """
    Devuelve los nombres de las features que genera extract_features_from_windows.

    Returns
    -------
    list[str]
        Lista de nombres en el mismo orden que las columnas del array.
    """
    if axis_names is None:
        axis_names = ["x", "y", "z"]

    stats = ["mean", "std", "mad", "max", "min", "range",
             "energy", "iqr", "skewness", "kurtosis", "rms"]
    return [f"acc_{ax}_{stat}" for ax in axis_names for stat in stats]


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------

def normalize_signals(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Normaliza las features estadísticas (media 0, std 1).

    El scaler se ajusta SOLO sobre train y se aplica a test.
    Esto evita data leakage.

    Parameters
    ----------
    X_train : np.ndarray, forma (n_train, n_features)
    X_test  : np.ndarray, forma (n_test, n_features)

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Selección y reducción de dimensionalidad
# ---------------------------------------------------------------------------

def select_features_anova(
    X: np.ndarray, y: np.ndarray, k: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Selecciona las k mejores features según ANOVA F-score.

    Parameters
    ----------
    X : np.ndarray, forma (n_samples, n_features)
    y : np.ndarray, forma (n_samples,)
    k : int
        Número de features a seleccionar.

    Returns
    -------
    X_selected : np.ndarray
    selected_indices : np.ndarray
    """
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector.get_support(indices=True)


def apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 20,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Aplica PCA y transforma ambos splits.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Features ya normalizadas.
    n_components : int
        Número de componentes principales.

    Returns
    -------
    X_train_pca, X_test_pca, pca_model
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


def feature_importance_rf(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 15,
) -> list[tuple[str, float]]:
    """
    Ordena y devuelve las features más importantes de un Random Forest.

    Returns
    -------
    list[tuple[str, float]]
        Lista de (nombre, importancia) ordenada de mayor a menor.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    return [(feature_names[i], float(importances[i])) for i in indices]
