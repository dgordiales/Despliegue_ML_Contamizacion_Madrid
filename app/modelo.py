from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


# Ruta del modelo, independiente del directorio desde el que se ejecute la API.
RUTA_MODELO = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "models"
    / "modelo_ridge_no2.joblib"
)


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga el pipeline entrenado y lo mantiene en memoria."""
    return joblib.load(RUTA_MODELO)


def predecir_no2(datos: dict) -> float:
    """Realiza una predicción de NO2 a partir de las variables del modelo."""
    modelo = cargar_modelo()

    # Conservamos exactamente las columnas y el orden del entrenamiento.
    columnas = list(modelo.feature_names_in_)

    faltantes = [columna for columna in columnas if columna not in datos]
    if faltantes:
        raise ValueError(f"Faltan variables: {', '.join(faltantes)}")

    entrada = pd.DataFrame([datos], columns=columnas)
    prediccion = modelo.predict(entrada)[0]

    return float(prediccion)
