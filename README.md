# Cloud Fleet Challenge — Ejercicio 4 de ACI606

**Cloud Computing | Plataforma de monitoreo de flotas virtuales con Machine Learning** 

Proyecto desarrollado a partir del [repositorio inicial del profesor](https://github.com/DrOscarL/aci606-cloud-fleet-challenge), extendido para ofrecer una solución desplegada en la nube con **FastAPI, PostgreSQL (Neon), Machine Learning (scikit-learn) y un dashboard web de monitoreo**.

- **Repositorio de entrega:** https://github.com/demix16/aci606-cloud-fleet-challenge
- **Aplicación pública:** https://aci606-cloud-fleet-challenge-z5vu.onrender.com/dashboard
- **API / Swagger:** https://aci606-cloud-fleet-challenge-z5vu.onrender.com/docs
- **Salud de la API:** https://aci606-cloud-fleet-challenge-z5vu.onrender.com/health

> **Nota del despliegue:** se utiliza Render Free. Si la instancia está inactiva, la primera solicitud puede tardar mientras el servicio vuelve a iniciarse. La telemetría se conserva en Neon PostgreSQL.

## 1. Objetivo y solución

Una flota virtual de robots produce telemetría de movimiento y estado (velocidad, batería, temperatura del motor, vibración, corriente y proximidad de obstáculos). La plataforma recibe esas mediciones, las guarda en una base de datos persistente y utiliza un modelo de clasificación para estimar la condición del robot:

- `NORMAL`: funcionamiento habitual según el simulador.
- `WARNING`: condición de advertencia.
- `CRITICAL`: condición crítica.

El operador puede consultar las lecturas, predicciones y el estado general del servicio desde un dashboard accesible por internet.

## 2. Arquitectura

```text
Simulador Python (simulator/)
           |
           | POST /telemetry (JSON)
           v
Render Web Service -- FastAPI (app/main.py)
           |                   |
           |                   +--> Pipeline ML (models/model.joblib)
           |                         POST /predict
           |
           +--> Neon PostgreSQL (app/db.py)
           |        tabla telemetry, historial completo
           |
           +--> Dashboard web (dashboard/index.html)
           |        robots, estados, batería, temperatura, historial
           |
           +--> /health y /metrics (observabilidad)
```

**Responsabilidades:** Render ejecuta la API y sirve el dashboard; Neon almacena de forma persistente la telemetría; el pipeline `joblib` realiza las predicciones. GitHub conserva el código fuente, los scripts, las evidencias y el historial de desarrollo.

## 3. Tecnologías utilizadas

| Componente | Tecnología |
|---|---|
| Backend / API | Python 3.13, FastAPI, Uvicorn |
| Entrenamiento | pandas, NumPy, scikit-learn, matplotlib |
| Clasificación | Regresión logística (seleccionada) y Random Forest (comparación) |
| Persistencia | SQLAlchemy; SQLite para desarrollo, PostgreSQL en Neon para producción |
| Frontend | HTML, CSS y JavaScript en `dashboard/index.html` |
| Plataforma PaaS | Render (Web Service) |
| Control de versiones | Git y GitHub, rama de funcionalidad y Pull Request a `main` |

Las dependencias concretas se encuentran en [`requirements.txt`](requirements.txt). La versión Python del despliegue se fija en [`.python-version`](.python-version).

## 4. Estructura del proyecto

```text
aci606-cloud-fleet-challenge/
├── app/
│   ├── db.py                        # Conexión y modelos de persistencia
│   └── main.py                      # API, inferencia, métricas y dashboard
├── dashboard/
│   └── index.html                   # Interfaz de monitoreo
├── data/                            # Datos generados localmente (CSV ignorado por Git)
├── docs/
│   └── API_CONTRACT.md              # Contrato de rutas del profesor
├── ml/
│   ├── build_fleet_dataset.py       # Generación de datos por flota
│   └── train_model.py               # EDA, entrenamiento, selección y evaluación
├── models/
│   └── model.joblib                 # Modelo entrenado versionado
├── reports/                         # Gráficos, resultados y evidencias
├── simulator/
│   ├── core.py                      # Generación de estados y telemetría
│   └── fleet_simulator.py           # Cliente que envía lecturas a la API
├── tests/
│   ├── smoke_test.py
│   ├── persistence_check.py
│   ├── predict_check.py
│   ├── dashboard_metrics_check.py
│   ├── eval_unseen_fleets.py
│   └── eval_concurrency_robustness.py
├── requirements.txt
└── .python-version
```

## 5. Ejecución local en Windows con VS Code

Requisitos previos: **Python 3.13, Git y VS Code**.

Desde la raíz del repositorio, en PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para iniciar la API en **modo local de desarrollo con SQLite** (sin configurar `DATABASE_URL`):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Abrir en el navegador:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/dashboard`
- `http://127.0.0.1:8000/docs`

Para enviar telemetría desde una **segunda terminal**:

```powershell
.\.venv\Scripts\python.exe simulator/fleet_simulator.py --api http://127.0.0.1:8000 --robots 5 --steps 25 --interval 0.3 --timeout 15
```

### Utilizar Neon desde el entorno local (opcional)

Para que la API local use la misma base PostgreSQL, define `DATABASE_URL` **sin escribir su contraseña en código ni en GitHub**. Por ejemplo, tras copiar la cadena completa desde **Neon → Connect**, en PowerShell:

```powershell
$env:DATABASE_URL = (Get-Clipboard -Raw).Trim()
Set-Clipboard -Value ""
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Estas instrucciones se ejecutan en la **misma terminal**: una nueva sesión no hereda automáticamente esa variable. Nunca publicar la cadena de conexión en capturas, commits o archivos `.env` versionados.

## 6. Generación y entrenamiento de Machine Learning

El dataset es **sintético**, generado por el simulador incluido en el proyecto; no proviene de robots físicos. La semilla fija hace reproducible la generación.

```powershell
.\.venv\Scripts\python.exe -m ml.build_fleet_dataset --fleets 30 --robots 5 --steps 80 --seed 606
.\.venv\Scripts\python.exe ml/train_model.py
```

Se generan **12.000 registros** (30 flotas × 5 robots × 80 lecturas), guardados en `data/training.csv`. Este CSV se genera localmente y no se versiona en GitHub.

**Features:** `speed`, `battery`, `motor_temp`, `vibration`, `motor_current` y `obstacle_distance`.

**Etiqueta:** `status` con clases `NORMAL`, `WARNING` y `CRITICAL`.

Se realizó EDA con distribución de clases, histogramas y matriz de correlación. Para evitar fuga de información entre flotas se utiliza `GroupShuffleSplit` agrupando por `fleet_id`: desarrollo y prueba se separan por flota; dentro del desarrollo se separan entrenamiento y validación también por flota. Se compararon dos clasificadores utilizando F1 macro en validación y se evaluó el elegido en las flotas reservadas para prueba.

| Evaluación del entrenamiento | F1 macro |
|---|---:|
| Regresión logística (validación) | 0,9363 |
| Random Forest (validación) | 0,9232 |
| **Modelo elegido, regresión logística (prueba)** | **0,9154** |
| Línea base: clase mayoritaria (prueba) | 0,1718 |

El pipeline elegido incluye imputación de medianas, escalado y clasificación. Se guarda en `models/model.joblib` y se carga al iniciar FastAPI. El modelo tiene versión `v1`.

**Evidencias del entrenamiento:** [distribución de clases](reports/clases.png), [histogramas EDA](reports/eda_variables.png), [correlaciones](reports/correlacion.png), [matriz de confusión](reports/matriz_confusion.png) y [métricas JSON](reports/metricas.json).

## 7. Contrato de la API

Se conservaron las siete rutas exigidas por el proyecto inicial.

| Método | Ruta | Función |
|---|---|---|
| GET | `/health` | Estado de API, base de datos y modelo |
| POST | `/telemetry` | Validar y persistir una lectura de robot; almacenar su predicción cuando el modelo está disponible |
| GET | `/robots` | Última telemetría de robots registrados |
| GET | `/robots/{robot_id}/history` | Historial persistente del robot; admite `?limit=240` para últimas lecturas |
| POST | `/predict` | Inferencia sin necesidad de guardar telemetría |
| GET | `/metrics` | Métricas operacionales, errores, latencias y uptime |
| GET | `/dashboard` | Panel web de monitoreo |

También está disponible `GET /docs` para consultar Swagger UI.

**Ejemplo de `POST /predict`:**

```json
{
  "robot_id": "R03",
  "timestamp": 1780000000.0,
  "x": 4.21,
  "y": 7.82,
  "speed": 0.63,
  "battery": 71.0,
  "motor_temp": 48.2,
  "vibration": 0.43,
  "motor_current": 3.7,
  "obstacle_distance": 0.42
}
```

**Estructura de respuesta:**

```json
{
  "robot_id": "R03",
  "predicted_status": "WARNING",
  "confidence": 0.87,
  "model_version": "v1",
  "inference_ms": 12.4
}
```

Los valores de la respuesta anterior son **ilustrativos**, no una predicción garantizada para esa muestra. `confidence` es la probabilidad estimada de la clase predicha y **no** debe confundirse con la exactitud general del modelo.

El contrato y las reglas de validación están implementados en `app/main.py`. Las entradas faltantes, fuera de rango, con campos inesperados o con números no finitos se rechazan con HTTP 422.

## 8. Dashboard y observabilidad

El dashboard muestra robots registrados, última lectura, estado de predicción, batería, velocidad, temperatura, confianza, historial reciente y gráficos de batería/temperatura. Consulta periódicamente la API.

`GET /metrics` expone, entre otras, las solicitudes desde el arranque, errores HTTP 4xx/5xx, percentiles de latencia, inferencias realizadas y tiempo activo. Los contadores operacionales del proceso se reinician cuando Render reinicia; **los registros PostgreSQL no**.

## 9. Pruebas realizadas y resultados

### 9.1. Flotas nuevas no vistas por el modelo (Etapa 8A)

Se generaron **8 flotas nuevas**, con **6 robots cada una** y **1.680 lecturas sintéticas**; se evaluaron 320 muestras seleccionadas de forma reproducible mediante el endpoint `/predict`. Las etiquetas reales del simulador se reservaron para la evaluación; no se enviaron a la API. No se volvió a entrenar el modelo.

| Métrica | Resultado |
|---|---:|
| F1 macro | **0,9257** |
| Línea base mayoritaria | 0,1717 |
| Exactitud | **92,81 %** (297 de 320) |
| CRITICAL correctamente identificado | 110 de 111 |
| CRITICAL confundido con NORMAL | 0 |

La misma evaluación sobre Render produjo los mismos resultados predictivos. Los percentiles de latencia, medidos desde el cliente, fueron aproximadamente:

| Entorno | p50 | p95 |
|---|---:|---:|
| API local | 5,734 ms | 7,831 ms |
| API pública en Render | 163,803 ms | 175,566 ms |

**Evidencias versionadas:** [resumen 8A de la ejecución local](reports/eval_no_vista_resumen.json), [matriz de confusión 8A](reports/eval_no_vista_matriz.png) y [detalle por muestra](reports/eval_no_vista_detalle.csv). Las latencias de Render citadas arriba provienen de la ejecución posterior contra la URL pública; se deben conservar sus capturas o generar un reporte remoto separado para documentarlas.

### 9.2. Robustez y concurrencia local (Etapa 8B)

- **9/9** entradas inválidas rechazadas con HTTP 422, incluido `NaN` como literal JSON.
- **98/98** respuestas correctas a solicitudes concurrentes (1, 4 y 8 clientes).
- **0** nuevos errores HTTP 5xx registrados en esa prueba.
- El servidor siguió disponible y no se alteró la telemetría almacenada.

| Clientes concurrentes | Solicitudes correctas | p50 cliente | p95 cliente |
|---|---:|---:|---:|
| 1 | 10/10 | 3,857 ms | 11,482 ms |
| 4 | 24/24 | 12,578 ms | 14,511 ms |
| 8 | 64/64 | 22,175 ms | 28,068 ms |

**Evidencias:** [resumen 8B](reports/eval_8b_resumen.json), [gráfico de concurrencia](reports/eval_8b_concurrencia.png) y [latencias individuales](reports/eval_8b_latencias.csv).

**Límite de la evidencia:** el ensayo de concurrencia se ejecutó **localmente**; no debe presentarse como medición equivalente en Render.

### 9.3. Persistencia y despliegue

- En SQLite local se verificó un historial de **105 registros** del robot `HIST-ETAPA4` conservados tras reiniciar Uvicorn.
- En PostgreSQL (Neon) se confirmaron **105 registros** mediante SQL Editor.
- Tras reiniciar el servicio en Render, se volvió a consultar el mismo historial: **105 antes, 105 después**, con comparación `True`.
- Render publicó el servicio con estado `Live`; `/health` confirmó `database: ok`, `storage: postgresql` y `model_ready: true`.
- El smoke test público devolvió HTTP 200 en `/health`, `/telemetry`, `/robots` y `/metrics`.

Los recuentos visibles en el dashboard cambian cuando llegan nuevas muestras; no son constantes del sistema.

## 10. Cómo repetir las pruebas

Con la API local encendida, desde otra terminal:

```powershell
# Comprobación básica de API
.\.venv\Scripts\python.exe tests/smoke_test.py --api http://127.0.0.1:8000

# Predicción y validación básica
.\.venv\Scripts\python.exe tests/predict_check.py

# Dashboard y métricas
.\.venv\Scripts\python.exe tests/dashboard_metrics_check.py

# Flotas nuevas con F1 macro (requiere data/training.csv generado localmente)
.\.venv\Scripts\python.exe tests/eval_unseen_fleets.py --api http://127.0.0.1:8000

# Robustez y concurrencia
.\.venv\Scripts\python.exe tests/eval_concurrency_robustness.py --api http://127.0.0.1:8000
```

Para probar desde la URL pública, reemplazar la dirección local por:

```text
https://aci606-cloud-fleet-challenge-z5vu.onrender.com
```

**Advertencia:** `smoke_test.py` envía una lectura real a la base conectada. `eval_unseen_fleets.py` y `eval_concurrency_robustness.py` consultan `/predict` sin guardar nuevas muestras de telemetría. Las evidencias JSON/CSV de estas pruebas pueden sobrescribirse al ejecutarlas otra vez.

### Prueba de persistencia con reinicio

```powershell
# Enviar 105 lecturas de prueba
.\.venv\Scripts\python.exe tests/persistence_check.py --mode send

# Reiniciar el servidor por separado, luego comprobar
.\.venv\Scripts\python.exe tests/persistence_check.py --mode verify
```

Si se ejecuta más de una vez, se pueden acumular más de 105 lecturas. Para Render, el script también admite `--api <URL_PUBLICA>`; las credenciales de Neon permanecen exclusivamente en el servicio.

## 11. Configuración de Render y Neon

El proyecto se despliega como **Render Web Service** desde la rama `main` del fork.

- **Runtime**: Python 3.13 (archivo `.python-version`).
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Region**: Ohio (US East), próxima a la base Neon utilizada.

**Variables de entorno requeridas en Render:**

| Variable | Función |
|---|---|
| `DATABASE_URL` | URL privada PostgreSQL obtenida en Neon → Connect |
| `MODEL_PATH` | `models/model.joblib` |
| `APP_ENV` | `production` |

**Seguridad:** no subir contraseñas ni el valor real de `DATABASE_URL` a GitHub. La aplicación en producción debe usar PostgreSQL; SQLite es solo alternativa de desarrollo.

## 12. Trazabilidad de Git

El desarrollo se trabajó en `feature/ejercicio4-cloud-fleet`, con commits separados para entrenamiento, persistencia, inferencia, dashboard y pruebas; posteriormente se integró a `main` mediante el Pull Request **#1** del fork. Se conservan archivos de evidencia y scripts reproducibles.

- [Repositorio de entrega](https://github.com/demix16/aci606-cloud-fleet-challenge)
- [Pull Request #1](https://github.com/demix16/aci606-cloud-fleet-challenge/pull/1)

## 13. Consideraciones y limitaciones

1. **Datos simulados:** los resultados de Machine Learning miden generalización frente a nuevas flotas del mismo simulador, con perturbaciones moderadas de sensores; no certifican seguridad industrial ni precisión sobre robots físicos reales.
2. **Dependencia temporal:** lecturas consecutivas del mismo robot están correlacionadas. Las muestras evaluadas no equivalen a 320 robots independientes.
3. **PaaS gratuito:** Render Free puede suspender la instancia por inactividad, produciendo un arranque lento tras una pausa.
4. **Métricas por proceso:** contadores de solicitudes, errores y uptime se reinician con la instancia; el historial permanece en Neon.
5. **API demostrativa:** las rutas públicas se ofrecen con fines académicos. Para un servicio productivo se requerirían autenticación, control de acceso, cuotas y protección contra abuso.

---

**ACI606 · Ejercicio 4 · Cloud Fleet Challenge**  
Plataforma académica reproducible y desplegada públicamente a partir del repositorio inicial del profesor.
