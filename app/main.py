"""Etapa 6 ACI606: persistencia + inferencia de ML para la flota.

Mantiene las siete rutas del contrato del profesor. No altera el esquema de la
base de datos; reutiliza las columnas predicted_status y confidence de Etapa 4.
"""
from __future__ import annotations

import logging
import os
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from app.db import Base, SessionLocal, TelemetryRecord, engine

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)
Status = Literal["NORMAL", "WARNING", "CRITICAL"]
VALID_STATUSES = {"NORMAL", "WARNING", "CRITICAL"}


class Telemetry(BaseModel):
    # Campos desconocidos, NaN, Infinity o valores fuera de rango -> HTTP 422.
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    robot_id: str = Field(min_length=1, max_length=20, pattern=r"^[A-Za-z0-9_-]+$")
    timestamp: float = Field(gt=0, allow_inf_nan=False)
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    speed: float = Field(ge=0, le=1.5, allow_inf_nan=False)
    battery: float = Field(ge=0, le=100, allow_inf_nan=False)
    motor_temp: float = Field(ge=-50, le=200, allow_inf_nan=False)
    vibration: float = Field(ge=0, le=100, allow_inf_nan=False)
    motor_current: float = Field(ge=0, le=100, allow_inf_nan=False)
    obstacle_distance: float = Field(ge=0, le=10000, allow_inf_nan=False)


class PredictionRequest(Telemetry):
    pass


_model_bundle: dict | None = None
_stats_lock = Lock()
_stats = Counter()
_total_inference_ms = 0.0
_inference_count = 0
_started_at = monotonic()
# Historial acotado de latencias recientes (sin crecer indefinidamente en RAM).
_http_latencies_ms = deque(maxlen=3000)
_predict_http_latencies_ms = deque(maxlen=3000)


def model_path() -> Path:
    path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    return path if path.is_absolute() else ROOT / path


def load_model() -> dict | None:
    path = model_path()
    if not path.is_file():
        LOGGER.warning("No se encontró models/model.joblib. /predict responderá 503.")
        return None
    # joblib.load ejecuta deserialización de Python: cargar SOLO modelos propios/confiables.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise RuntimeError("El modelo guardado no tiene el formato esperado (dict).")
    needed = {"pipeline", "features", "model_version"}
    if not needed.issubset(bundle):
        raise RuntimeError("El modelo no contiene pipeline, features y model_version.")
    if not isinstance(bundle["features"], list) or not bundle["features"]:
        raise RuntimeError("El modelo no contiene una lista válida de features.")
    if not set(bundle["features"]).issubset(Telemetry.model_fields):
        raise RuntimeError("El modelo requiere features que no están en Telemetry.")
    pipeline = bundle["pipeline"]
    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
        raise RuntimeError("El pipeline debe permitir predict() y predict_proba().")
    LOGGER.info("Modelo cargado: %s", bundle["model_version"])
    return bundle


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _model_bundle
    # CREATE TABLE IF NOT EXISTS; no borra el historial existente de Neon.
    Base.metadata.create_all(bind=engine)
    _model_bundle = load_model()
    yield
    _model_bundle = None


app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.4.0-dashboard-metrics",
    description="Etapa 7: persistencia, inferencia, dashboard y observabilidad.",
    lifespan=lifespan,
)


def _percentile(values: list[float], fraction: float) -> float:
    """Percentil por interpolación lineal; 0 cuando no hay observaciones."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower), 3)


def _record_http(path: str, status_code: int, elapsed_ms: float) -> None:
    """Una sola escritura sincronizada por solicitud; mantiene métricas en RAM."""
    with _stats_lock:
        _http_latencies_ms.append(elapsed_ms)
        if path == "/predict":
            _predict_http_latencies_ms.append(elapsed_ms)
        if 400 <= status_code < 500:
            _stats["errors_4xx_since_boot"] += 1
        elif status_code >= 500:
            _stats["errors_5xx_since_boot"] += 1


@app.middleware("http")
async def count_http(request: Request, call_next):
    started = perf_counter()
    with _stats_lock:
        _stats["http_requests_since_boot"] += 1
    try:
        response = await call_next(request)
    except Exception:
        _record_http(request.url.path, 500, (perf_counter() - started) * 1000)
        raise
    _record_http(request.url.path, response.status_code, (perf_counter() - started) * 1000)
    return response


FIELDS = (
    "robot_id", "timestamp", "x", "y", "speed", "battery", "motor_temp",
    "vibration", "motor_current", "obstacle_distance",
)


def serialize(record: TelemetryRecord) -> dict:
    result = {name: getattr(record, name) for name in FIELDS}
    result["predicted_status"] = record.predicted_status
    result["confidence"] = record.confidence
    return result


def infer(sample: Telemetry) -> dict:
    global _total_inference_ms, _inference_count
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Entrenar y guardar models/model.joblib.")

    values = sample.model_dump()
    features = _model_bundle["features"]
    row = pd.DataFrame([{feature: values[feature] for feature in features}], columns=features)
    start = perf_counter()
    try:
        pipeline = _model_bundle["pipeline"]
        predicted = str(pipeline.predict(row)[0])
        probabilities = pipeline.predict_proba(row)[0]
        classes = list(pipeline.classes_)
        if predicted not in VALID_STATUSES or predicted not in classes:
            raise ValueError("Predicción fuera de clases permitidas")
        confidence = float(probabilities[classes.index(predicted)])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confianza fuera de rango")
    except Exception:
        LOGGER.exception("Error interno de inferencia")
        raise HTTPException(status_code=503, detail="No se pudo ejecutar el modelo") from None

    elapsed_ms = (perf_counter() - start) * 1000.0
    with _stats_lock:
        _inference_count += 1
        _total_inference_ms += elapsed_ms
    return {
        "robot_id": sample.robot_id,
        "predicted_status": predicted,
        "confidence": round(confidence, 6),
        "model_version": str(_model_bundle["model_version"]),
        "inference_ms": round(elapsed_ms, 3),
    }


@app.get("/health")
def health() -> dict:
    try:
        with SessionLocal() as session:
            session.execute(select(1)).scalar_one()
        db_status = "ok"
    except Exception:
        LOGGER.exception("No se pudo comprobar la base de datos")
        db_status = "unavailable"
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "database": db_status,
        "storage": engine.dialect.name,
        "model_ready": _model_bundle is not None,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    # Al guardar una nueva muestra, también se guarda el estado inferido.
    prediction = infer(sample) if _model_bundle is not None else None
    values = sample.model_dump()
    if prediction:
        values["predicted_status"] = prediction["predicted_status"]
        values["confidence"] = prediction["confidence"]
    with SessionLocal() as session:
        record = TelemetryRecord(**values)
        session.add(record)
        session.commit()
        session.refresh(record)
        record_id = record.id
    with _stats_lock:
        _stats["telemetry_ingested_since_boot"] += 1
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "id": record_id,
        "predicted_status": prediction["predicted_status"] if prediction else None,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "storage": "DATABASE",
    }


@app.get("/robots")
def list_robots() -> list[dict]:
    latest_ids = (
        select(
            TelemetryRecord.id.label("id"),
            func.row_number().over(
                partition_by=TelemetryRecord.robot_id,
                order_by=TelemetryRecord.id.desc(),
            ).label("position"),
        ).subquery()
    )
    ids = select(latest_ids.c.id).where(latest_ids.c.position == 1)
    with SessionLocal() as session:
        records = session.scalars(
            select(TelemetryRecord)
            .where(TelemetryRecord.id.in_(ids))
            .order_by(TelemetryRecord.robot_id)
        ).all()
    return [serialize(record) for record in records]


@app.get("/robots/{robot_id}/history")
def robot_history(
    robot_id: str,
    limit: int | None = Query(default=None, ge=1, le=5000),
) -> list[dict]:
    """Sin limit: historial COMPLETO (contrato). Con limit: últimas N lecturas."""
    with SessionLocal() as session:
        query = select(TelemetryRecord).where(TelemetryRecord.robot_id == robot_id)
        if limit is None:
            records = session.scalars(query.order_by(TelemetryRecord.id.asc())).all()
        else:
            # Selecciona los últimos N en SQL y los devuelve cronológicamente.
            records = list(reversed(session.scalars(
                query.order_by(TelemetryRecord.id.desc()).limit(limit)
            ).all()))
    if not records:
        raise HTTPException(status_code=404, detail="Robot not found")
    return [serialize(record) for record in records]


@app.post("/predict")
def predict(sample: PredictionRequest) -> dict:
    with _stats_lock:
        _stats["prediction_requests_since_boot"] += 1
    return infer(sample)


@app.get("/metrics")
def metrics() -> dict:
    with SessionLocal() as session:
        total_samples = session.scalar(select(func.count()).select_from(TelemetryRecord)) or 0
        total_robots = session.scalar(select(func.count(func.distinct(TelemetryRecord.robot_id)))) or 0
    with _stats_lock:
        snapshot = dict(_stats)
        count = _inference_count
        avg_ms = _total_inference_ms / count if count else 0.0
        http_latencies = list(_http_latencies_ms)
        predict_latencies = list(_predict_http_latencies_ms)
    uptime = max(0.0, monotonic() - _started_at)
    return {
        **snapshot,
        # Campos de compatibilidad con Etapa 6 conservados.
        "http_requests_since_boot": snapshot.get("http_requests_since_boot", 0),
        "errors_4xx_since_boot": snapshot.get("errors_4xx_since_boot", 0),
        "errors_5xx_since_boot": snapshot.get("errors_5xx_since_boot", 0),
        "prediction_requests_since_boot": snapshot.get("prediction_requests_since_boot", 0),
        "telemetry_ingested_since_boot": snapshot.get("telemetry_ingested_since_boot", 0),
        "average_inference_ms_since_boot": round(avg_ms, 3),
        "inference_count_since_boot": count,
        "uptime_seconds": round(uptime, 2),
        # Latencias HTTP tomadas en el servidor: NO son el tiempo desde cliente.
        "http_latency_p50_ms": _percentile(http_latencies, 0.50),
        "http_latency_p95_ms": _percentile(http_latencies, 0.95),
        "predict_http_latency_p50_ms": _percentile(predict_latencies, 0.50),
        "predict_http_latency_p95_ms": _percentile(predict_latencies, 0.95),
        "requests_per_minute_since_boot": round(snapshot.get("http_requests_since_boot", 0) * 60 / max(1.0, uptime), 2),
        "latency_sample_count": len(http_latencies),
        "latency_sample_window_max": 3000,
        "total_telemetry_persisted": int(total_samples),
        "total_robots_persisted": int(total_robots),
        "persistence": "DATABASE",
        "database": engine.dialect.name,
        "prediction_service": "READY" if _model_bundle is not None else "NOT_AVAILABLE",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
