# HAR — Human Activity Recognition con WISDM

Proyecto de clasificación de actividad humana a partir de señales crudas de acelerómetro (X, Y, Z). Pipeline completo: EDA → modelos clásicos (sklearn) → Deep Learning (PyTorch) → comparativa.

## Estructura del proyecto

```
wisdm-project/
├── README.md
├── requirements.txt
├── results.md
├── data/
│   ├── x_train.npy       # Señales train (43924, 90, 3)
│   ├── x_test.npy        # Señales test  (10982, 90, 3)
│   ├── y_train.npy       # Etiquetas train
│   └── y_test.npy        # Etiquetas test
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── evaluate.py
│   └── models/
│       ├── baseline.py
│       └── deep.py
├── figures/
└── models/
```

## Dataset

- **Fuente:** WISDM — Wireless Sensor Data Mining Lab, Fordham University
- **Clases (6):** Walking, Jogging, Sitting, Standing, Upstairs, Downstairs
- **Señal:** Acelerómetro triaxial (X, Y, Z) — ventanas de 90 timesteps
- **Split:** train (43,924 muestras) / test (10,982 muestras)

### Distribución de clases

| Clase | Actividad   | Train  | Test  |
|-------|-------------|--------|-------|
| 0     | Walking     | 4,005  | 1,013 |
| 1     | Jogging     | 13,620 | 3,481 |
| 2     | Sitting     | 2,400  | 596   |
| 3     | Standing    | 1,936  | 482   |
| 4     | Upstairs    | 4,972  | 1,173 |
| 5     | Downstairs  | 16,991 | 4,237 |

⚠️ **Dataset desbalanceado** — Jogging y Downstairs dominan. Métrica principal: **F1-macro**.

## Stack tecnológico

- **Python** 3.10+
- **Datos:** numpy, scipy
- **Visualización:** matplotlib, seaborn
- **Modelos clásicos:** scikit-learn, xgboost
- **Deep Learning:** PyTorch
- **Notebooks:** jupyter

## Métricas objetivo

| Modelo         | F1-macro esperado |
|----------------|-------------------|
| Random Forest  | ≥ 0.88            |
| SVM            | ≥ 0.90            |
| CNN-1D         | ≥ 0.92            |
| LSTM           | ≥ 0.91            |

La métrica **principal** es F1-macro. Reportar también accuracy y matriz de confusión.

## Diferencia clave vs UCI HAR

UCI HAR provee 561 features **precalculadas**. WISDM en esta versión provee la **señal cruda** (90 timesteps × 3 ejes). Esto implica:

1. Para modelos clásicos → extraer features estadísticas manualmente (`src/features.py`)
2. Para deep learning → alimentar la señal cruda directamente al modelo

Esta diferencia es el núcleo pedagógico del proyecto.

## Contexto del proyecto

Trabajo final de la asignatura Aprendizaje Automático — Máster en Big Data Analytics, Universidad Europea de Andalucía. El objetivo es demostrar la progresión:

1. Features manuales + modelos clásicos
2. Señal cruda + Deep Learning
3. Análisis crítico y comparativa de resultados
