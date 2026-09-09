"""Endpoint para reentrenar el modelo con datos preparados."""

import os
import secrets
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from app.reentrenamiento import reentrenar_modelo

router = APIRouter()
TAMANO_MAXIMO = 20 * 1024 * 1024


@router.post("/reentrenar")
def reentrenar(
    archivo: UploadFile = File(...),
    token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    """Recibe un CSV preparado y guarda un nuevo modelo por separado."""
    clave = os.getenv("REENTRENAMIENTO_TOKEN")

    if not clave or not token or not secrets.compare_digest(token, clave):
        raise HTTPException(
            status_code=403,
            detail="Reentrenamiento no autorizado.",
        )

    contenido = archivo.file.read(TAMANO_MAXIMO + 1)

    if len(contenido) > TAMANO_MAXIMO:
        raise HTTPException(
            status_code=413,
            detail="El archivo supera el tamaño máximo de 20 MB.",
        )

    try:
        datos = pd.read_csv(BytesIO(contenido))
        modelo_nuevo, filas = reentrenar_modelo(datos)
    except (ValueError, TypeError, pd.errors.ParserError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    # Se guarda una versión separada, sin sustituir el modelo original.
    ruta = (
        Path(__file__).resolve().parent.parent
        / "modelos_reentrenados"
        / "modelo_ridge_no2_reentrenado.joblib"
    )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_nuevo, ruta)

    return {
        "mensaje": "Reentrenamiento completado.",
        "filas_utilizadas": filas,
        "modelo_guardado": ruta.name,
        "modelo_activo": False,
    }
