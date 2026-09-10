from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatosPrediccion(BaseModel):
    """Variables necesarias para predecir la concentración de NO2."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    # Meteorología.
    temperatura: float | None
    humedad: float | None
    viento_vel: float | None
    precipitacion: float | None

    # Tráfico.
    intensidad_mean: float | None
    intensidad_max: float | None
    ocupacion_mean: float | None
    carga_mean: float | None
    vmed_mean: float | None
    vmed_max: float | None

    # Distancias.
    distancia_meteo_km: float | None
    distancia_trafico_km: float | None

    # Variables temporales.
    mes_sin: float
    mes_cos: float
    dia_anio_sin: float
    dia_anio_cos: float
    finde: Literal[0, 1]

    # Histórico de NO2.
    no2_lag1: float = Field(ge=0)
    no2_lag7: float | None
    no2_roll_mean_3: float | None
    no2_roll_mean_7: float | None

    # Indicadores de valores ausentes.
    viento_vel_missing: Literal[0, 1]
    precipitacion_missing: Literal[0, 1]

    # Estación.
    estacion: int = Field(gt=0)
    tipo_elem_moda: str
    estacion_anio: str
