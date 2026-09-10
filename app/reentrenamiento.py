"""Funciones para reentrenar el modelo de NO2."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from app.modelo import cargar_modelo


def reentrenar_modelo(datos: pd.DataFrame):
    """Entrena una copia del pipeline con un dataset preparado."""
    modelo_original = cargar_modelo()
    variables = list(modelo_original.feature_names_in_)

    columnas_necesarias = variables + ["no2"]
    faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in datos.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas: {', '.join(faltantes)}"
        )

    datos = datos.copy()

    # El target y el lag1 deben ser numéricos.
    datos["no2"] = pd.to_numeric(datos["no2"], errors="raise")
    datos["no2_lag1"] = pd.to_numeric(
        datos["no2_lag1"], errors="raise"
    )

    # Se aplica el mismo filtro utilizado en el notebook original.
    datos = datos.dropna(subset=["no2", "no2_lag1"])

    if datos.empty:
        raise ValueError("No hay filas válidas para reentrenar.")

    if not np.isfinite(datos[["no2", "no2_lag1"]].to_numpy(dtype=float)).all():
        raise ValueError("El target y el lag1 deben ser finitos.")

    X = datos[variables]
    y = datos["no2"]

    # Clone conserva la configuración, pero crea un pipeline sin entrenar.
    modelo_nuevo = clone(modelo_original)
    modelo_nuevo.fit(X, y)

    return modelo_nuevo, len(datos)
