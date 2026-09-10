"""Pruebas del endpoint de reentrenamiento."""

import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
URL_API = os.getenv("URL_API", "http://127.0.0.1:8001").rstrip("/")
TOKEN = os.getenv("REENTRENAMIENTO_TOKEN")

# Este script comprueba archivos locales, no un servidor remoto.
if urlsplit(URL_API).hostname not in {"127.0.0.1", "localhost", "::1"}:
    raise RuntimeError(
        "Este script solo debe ejecutarse contra la API local."
    )

RUTA_CSV = RAIZ / "ejemplos" / "datos_reentrenamiento_prueba.csv"
RUTA_MODELO = RAIZ / "src" / "models" / "modelo_ridge_no2.joblib"
RUTA_NUEVO = (
    RAIZ / "modelos_reentrenados"
    / "modelo_ridge_no2_reentrenado.joblib"
)

if not TOKEN:
    raise RuntimeError("Falta REENTRENAMIENTO_TOKEN.")


def calcular_hash(ruta):
    """Calcula el hash de un archivo."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def solicitar(contenido, token=None, timeout=30):
    """Envía un CSV al endpoint de reentrenamiento."""
    headers = {}
    if token is not None:
        headers["X-Admin-Token"] = token

    return requests.post(
        f"{URL_API}/reentrenar",
        files={"archivo": ("datos.csv", contenido, "text/csv")},
        headers=headers,
        timeout=timeout,
    )


hash_original = calcular_hash(RUTA_MODELO)
datos = pd.read_csv(RUTA_CSV)
filas_esperadas = len(datos.dropna(subset=["no2", "no2_lag1"]))

# 1. Petición sin clave.
respuesta = solicitar(b"no2\n1\n")
assert respuesta.status_code == 403, respuesta.text
print("Sin clave: OK (403)")

# 2. Clave incorrecta.
respuesta = solicitar(b"no2\n1\n", token="clave-incorrecta")
assert respuesta.status_code == 403, respuesta.text
print("Clave incorrecta: OK (403)")

# 3. Falta el target.
csv_sin_target = (
    datos.head(2)
    .drop(columns=["no2"])
    .to_csv(index=False)
    .encode()
)
respuesta = solicitar(csv_sin_target, token=TOKEN)
assert respuesta.status_code == 422, respuesta.text
print("Target ausente: OK (422)")

# 4. El lag1 no es numérico.
datos_invalidos = datos.head(2).copy()
datos_invalidos["no2_lag1"] = "texto"
respuesta = solicitar(
    datos_invalidos.to_csv(index=False).encode(),
    token=TOKEN,
)
assert respuesta.status_code == 422, respuesta.text
print("Lag1 no numérico: OK (422)")

# 5. Archivo demasiado grande.
respuesta = solicitar(
    b"x" * (20 * 1024 * 1024 + 1),
    token=TOKEN,
)
assert respuesta.status_code == 413, respuesta.text
print("Archivo demasiado grande: OK (413)")

# 6. Reentrenamiento válido con la muestra incluida en el repositorio.
respuesta = solicitar(
    RUTA_CSV.read_bytes(),
    token=TOKEN,
    timeout=300,
)
assert respuesta.status_code == 200, respuesta.text

resultado = respuesta.json()
assert resultado["filas_utilizadas"] == filas_esperadas
assert resultado["modelo_activo"] is False
assert RUTA_NUEVO.exists()
print(f"Reentrenamiento válido: OK ({filas_esperadas} filas)")

# 7. El modelo original debe permanecer exactamente igual.
assert calcular_hash(RUTA_MODELO) == hash_original
print("Modelo original sin modificaciones: OK")

print("\n¡Todas las pruebas de reentrenamiento han pasado!")
