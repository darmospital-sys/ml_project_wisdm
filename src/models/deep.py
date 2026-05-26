"""
Modelos de Deep Learning para HAR — CNN-1D y LSTM (PyTorch).

A diferencia de los modelos clásicos, estos consumen la SEÑAL CRUDA
(batch, 90, 3) directamente, sin necesidad de extraer features a mano.
El modelo aprende sus propias representaciones durante el entrenamiento.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


# ---------------------------------------------------------------------------
# Arquitecturas
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):
    """
    Red convolucional 1D para clasificación de series temporales.

    Recibe ventanas de señal cruda (batch, timesteps, channels) y aprende
    filtros locales que detectan patrones temporales (pasos, impactos, etc.).

    Arquitectura:
        Conv1d(3→64, k=5) → BN → ReLU → MaxPool(2)   [90 → 45]
        Conv1d(64→128, k=5) → BN → ReLU → MaxPool(2)  [45 → 22]
        Flatten → FC(2816→128) → Dropout(0.5) → FC(128→6)
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 6, n_timesteps: int = 90):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )
        # Dos MaxPool1d(2) → longitud final = n_timesteps // 4
        flat_size = 128 * (n_timesteps // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, timesteps, channels) → permute a (batch, channels, timesteps)
        # porque Conv1d espera (batch, C_in, L)
        x = x.permute(0, 2, 1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.classifier(x)


class LSTMClassifier(nn.Module):
    """
    LSTM para clasificación de series temporales.

    Procesa la señal completa de forma secuencial — tiene memoria del
    pasado, a diferencia de la CNN que solo ve ventanas locales.
    Usa el estado oculto del último timestep para clasificar.

    Arquitectura:
        LSTM(3 → 128, 2 capas, dropout=0.3)
        FC(128 → 64) → ReLU → Dropout(0.5) → FC(64 → 6)
    """

    def __init__(
        self,
        n_features: int = 3,
        n_classes: int = 6,
        hidden_size: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.3 if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, timesteps, features) — batch_first=True
        out, _ = self.lstm(x)
        # Tomamos solo el último timestep como representación de la secuencia
        return self.classifier(out[:, -1, :])


# ---------------------------------------------------------------------------
# Funciones de entrenamiento y evaluación
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Un epoch de entrenamiento. Devuelve (loss_medio, accuracy)."""
    model.train()
    total_loss, correct = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Un epoch de evaluación. Devuelve (loss_medio, accuracy)."""
    model.eval()
    total_loss, correct = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Genera predicciones sobre todo el loader. Devuelve array de etiquetas."""
    model.eval()
    preds = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds.extend(logits.argmax(1).cpu().numpy())
    return np.array(preds)


def save_model(model: nn.Module, name: str) -> Path:
    """Guarda los pesos del modelo en models/<name>.pt"""
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    return path
