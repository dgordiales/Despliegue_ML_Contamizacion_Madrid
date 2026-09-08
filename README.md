# Despliegue_ML_Contamizacion_Madrid
Repositosio del proyecto de puesta en producción de un modelo machine learning realizado por alumnos del Bootcamp online Data Science &amp; IA de The Bridge

## API de predicción de NO₂

Esta aplicación reutiliza el modelo Ridge entrenado en el proyecto anterior de contaminación de Madrid. El modelo y su preprocesamiento están guardados en `src/models/`.

La API recibe las 26 variables que necesita el modelo y devuelve la concentración diaria estimada de NO₂ en µg/m³. No realiza un nuevo entrenamiento.

### Requisitos

- Python 3.13.
- Las dependencias incluidas en `requirements.txt`.

### Instalación

Clonar el repositorio y, desde su carpeta raíz, crear un entorno virtual:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Ejecutar la API localmente

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

La aplicación estará disponible en `http://127.0.0.1:8000/` y la documentación interactiva en `http://127.0.0.1:8000/docs`.

### Endpoints

**GET /**

Muestra una página de inicio con información sobre el proyecto y los endpoints disponibles.

**GET /predict**

Recibe un parámetro de consulta llamado `datos`, que contiene un JSON con las 26 variables del modelo. Devuelve una respuesta como:

```json
{
  "no2_previsto": 43.19,
  "unidad": "µg/m³"
}
```

Los campos obligatorios deben estar presentes. Algunas variables admiten `null`, ya que el pipeline incluye imputación de valores ausentes. Una entrada incorrecta devuelve el código HTTP 422.

### Ejemplo de petición con Python

El archivo `ejemplos/entrada_ejemplo.json` contiene una entrada real del conjunto de prueba, sin incluir el valor actual del target.

```python
import json
import requests

with open("ejemplos/entrada_ejemplo.json", encoding="utf-8") as archivo:
    datos = json.load(archivo)

respuesta = requests.get(
    "http://127.0.0.1:8000/predict",
    params={"datos": json.dumps(datos)},
    timeout=30,
)

print(respuesta.status_code)
print(respuesta.json())
```

Este ejemplo devuelve aproximadamente 43,19 µg/m³ para la estación 4. El valor real de esa observación es 56,0 µg/m³ y se utiliza únicamente para comparar, no como entrada del modelo.

### Pruebas

Con el servidor local en funcionamiento, ejecutar desde otro Terminal:

```bash
python tests/test_api.py
```

El script comprueba la página inicial, la predicción válida y diferentes entradas incorrectas. También puede utilizarse con la URL pública:

```bash
URL_API=https://URL-DEL-SERVICIO python tests/test_api.py
```

### Estructura

```text
app/
    main.py                  # Endpoints y página de inicio
    modelo.py                # Carga del modelo y predicción
    esquemas.py              # Validación de las 26 variables
src/models/                 # Pipeline entrenado y metadatos
ejemplos/                   # Ejemplo real de entrada
tests/                      # Pruebas de la API
requirements.txt            # Dependencias
```

### Despliegue

El servicio está desplegado en Render (Python 3.13, fijado mediante `.python-version`) sobre la rama `feature/deploy`:

**https://despliegue-ml-contamizacion-madrid.onrender.com**

Comando de instalación:

```bash
pip install -r requirements.txt
```

Comando de inicio:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

El modelo se carga desde una ruta relativa al código, por lo que no depende de las carpetas personales del ordenador.

Render redespliega automáticamente con cada `git push` a la rama conectada, sin configuración manual de webhooks.

> Render puede dejar inactivo el servicio tras un periodo de tiempo sin actividad: la primera petición después de la inactividad puede tardar hasta un minuto en responder.

### Tercer endpoint

### Tercer endpoint

El endpoint informativo `GET /info` está preparado y comentado en `app/main.py`. Para activarlo durante la demostración basta con descomentarlo y hacer `git push`: Render lo redespliega automáticamente en menos de un minuto. El procedimiento ya se ha probado en un ensayo previo.

### Limitaciones

El modelo necesita variables meteorológicas, de tráfico, temporales y observaciones históricas de NO₂ ya preparadas. No permite obtener una predicción únicamente a partir del nombre de una ciudad y una fecha. En particular, necesita una observación anterior de NO₂ disponible.

El ejemplo incluido es histórico y se utiliza para demostrar el funcionamiento de la API. El servicio no descarga datos en tiempo real ni sustituye una predicción oficial de calidad del aire.

### Proyecto original

Modelo reutilizado de [ML_Contaminacion_Madrid](https://github.com/dgordiales/ML_Contaminacion_Madrid), realizado en el Bootcamp Data Science & IA de The Bridge.
