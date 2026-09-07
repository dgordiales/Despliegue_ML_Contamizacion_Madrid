import json
import os
from pathlib import Path

import requests


# Por defecto probamos la API local. También permite usar la URL pública.
URL_API = os.getenv("URL_API", "http://127.0.0.1:8000").rstrip("/")
RUTA_EJEMPLO = (
    Path(__file__).resolve().parent.parent
    / "ejemplos"
    / "entrada_ejemplo.json"
)

with open(RUTA_EJEMPLO, encoding="utf-8") as archivo:
    ejemplo = json.load(archivo)


def solicitar_prediccion(datos):
    """Envía los datos mediante GET y devuelve la respuesta."""
    return requests.get(
        f"{URL_API}/predict",
        params={"datos": json.dumps(datos)},
        timeout=30,
    )


# 1. La página de inicio debe estar disponible.
respuesta = requests.get(URL_API, timeout=30)
assert respuesta.status_code == 200
print("Página de inicio: OK")


# 2. La predicción debe coincidir con la prueba del modelo original.
respuesta = solicitar_prediccion(ejemplo)
assert respuesta.status_code == 200, respuesta.text

resultado = respuesta.json()
assert abs(resultado["no2_previsto"] - 43.19) < 0.01
assert resultado["unidad"] == "µg/m³"
print("Predicción válida: OK")


# 3. La API debe rechazar entradas incorrectas.
casos = [
    ("Variable obligatoria ausente", {"no2_lag1": None}),
    ("NO2 anterior negativo", {"no2_lag1": -5}),
    ("Variable desconocida", {"variable_inventada": 123}),
    ("Temperatura no numérica", {"temperatura": "texto"}),
]

for nombre, cambios in casos:
    datos = ejemplo.copy()
    if nombre == "Variable obligatoria ausente":
        datos.pop("no2_lag1")
    else:
        datos.update(cambios)

    respuesta = solicitar_prediccion(datos)
    assert respuesta.status_code == 422, (
        f"{nombre}: {respuesta.status_code} - {respuesta.text}"
    )
    print(f"{nombre}: OK")


print("\n¡Todas las pruebas han pasado!")
