from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from app.esquemas import DatosPrediccion
from app.modelo import predecir_no2


app = FastAPI(
    title="API de Contaminación de Madrid",
    description="Predicción de la concentración diaria de NO2 mediante Machine Learning.",
)


@app.get("/", response_class=HTMLResponse)
def inicio():
    """Página de inicio con información sobre la API."""
    return """
    <h1>Predicción de NO₂ en Madrid</h1>
    <p>API para estimar la concentración diaria de NO₂ en las estaciones
    de calidad del aire de Madrid mediante un modelo Ridge entrenado.</p>

    <h2>Endpoints</h2>
    <p><strong>GET /</strong>: información sobre la API.</p>
    <p><strong>GET /predict</strong>: recibe un JSON con las 26 variables
    del modelo en el parámetro <code>datos</code> y devuelve la predicción
    en µg/m³.</p>

    <p>Los datos deben incluir información meteorológica, de tráfico,
    temporal y observaciones anteriores de NO₂. El archivo
    <code>ejemplos/entrada_ejemplo.json</code> contiene una entrada real
    que puede utilizarse para probar el servicio.</p>

    <p><a href="/docs">Consultar la documentación interactiva</a></p>
    """


@app.get("/predict")
def predecir(datos: str = Query(..., description="JSON con las 26 variables del modelo")):
    """Valida los datos y devuelve una predicción de NO2."""
    try:
        entrada = DatosPrediccion.model_validate_json(datos)
    except ValidationError as error:
        raise HTTPException(
            status_code=422,
            detail=error.errors(include_input=False, include_context=False),
        ) from error

    prediccion = predecir_no2(entrada.model_dump())

    return {
        "no2_previsto": round(prediccion, 2),
        "unidad": "µg/m³",
    }


# Tercer endpoint preparado para la demostración.
# Para activarlo, descomentar estas líneas y volver a desplegar.
#
# @app.get("/info")
# def informacion():
#     return {
#         "modelo": "Ridge",
#         "contaminante": "NO2",
#         "unidad": "µg/m³",
#     }
