"""Etapa 4 del ejercicio: FastAPI con telemetría persistente.

Se preservan las 7 rutas del starter. /predict seguirá con 501 hasta la
siguiente etapa, cuando se integre el pipeline entrenado.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select

from app.db import Base, SessionLocal, TelemetryRecord, engine

ROOT = Path(__file__).resolve().parents[1]
Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float = Field(gt=0, allow_inf_nan=False)
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    speed: float = Field(ge=0, allow_inf_nan=False)
    battery: float = Field(ge=0, le=100, allow_inf_nan=False)
    motor_temp: float = Field(allow_inf_nan=False)
    vibration: float = Field(ge=0, allow_inf_nan=False)
    motor_current: float = Field(ge=0, allow_inf_nan=False)
    obstacle_distance: float = Field(ge=0, allow_inf_nan=False)


class PredictionRequest(Telemetry):
    pass


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Crea la tabla al iniciar. Para cambios futuros de esquema: usar migraciones.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.2.0-persistence",
    description="Etapa 4: persistencia de la flota.",
    lifespan=lifespan,
)

# Contador de proceso: se reinicia cuando el servidor se reinicia.
REQUEST_COUNT = 0
FIELDS = [
    "robot_id", "timestamp", "x", "y", "speed", "battery", "motor_temp",
    "vibration", "motor_current", "obstacle_distance",
]


def serialize(record: TelemetryRecord) -> dict:
    return {name: getattr(record, name) for name in FIELDS}


@app.get("/health")
def health() -> dict:
    try:
        with SessionLocal() as session:
            session.execute(select(1)).scalar_one()
        database_status = "ok"
    except Exception:
        database_status = "unavailable"
    return {
        "status": "ok" if database_status == "ok" else "degraded",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "database": database_status,
        "storage": engine.dialect.name,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    global REQUEST_COUNT
    values = sample.model_dump()
    with SessionLocal() as session:
        record = TelemetryRecord(**values)
        session.add(record)
        session.commit()
        session.refresh(record)
        record_id = record.id
    REQUEST_COUNT += 1
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "id": record_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "storage": "DATABASE",
    }


@app.get("/robots")
def list_robots() -> list[dict]:
    # Último dato insertado por robot, compatible con SQLite y PostgreSQL.
    row_numbers = (
        select(
            TelemetryRecord.id.label("id"),
            func.row_number().over(
                partition_by=TelemetryRecord.robot_id,
                order_by=TelemetryRecord.id.desc(),
            ).label("num"),
        )
        .subquery()
    )
    newest_ids = select(row_numbers.c.id).where(row_numbers.c.num == 1)
    with SessionLocal() as session:
        records = session.scalars(
            select(TelemetryRecord)
            .where(TelemetryRecord.id.in_(newest_ids))
            .order_by(TelemetryRecord.robot_id)
        ).all()
        return [serialize(record) for record in records]


@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str) -> list[dict]:
    with SessionLocal() as session:
        records = session.scalars(
            select(TelemetryRecord)
            .where(TelemetryRecord.robot_id == robot_id)
            .order_by(TelemetryRecord.id.asc())
        ).all()
        if not records:
            raise HTTPException(status_code=404, detail="Robot not found")
        return [serialize(record) for record in records]


@app.post("/predict")
def predict(_: PredictionRequest) -> dict:
    # Se completará en la siguiente etapa con models/model.joblib.
    raise HTTPException(
        status_code=501,
        detail="La inferencia se integrará en la siguiente etapa.",
    )


@app.get("/metrics")
def metrics() -> dict:
    with SessionLocal() as session:
        total_samples = session.scalar(select(func.count()).select_from(TelemetryRecord)) or 0
        total_robots = session.scalar(
            select(func.count(func.distinct(TelemetryRecord.robot_id)))
        ) or 0
    return {
        "requests_received_since_boot": REQUEST_COUNT,
        "total_telemetry_persisted": int(total_samples),
        "total_robots_persisted": int(total_robots),
        "persistence": "DATABASE",
        "database": engine.dialect.name,
        "prediction_service": "NOT_IMPLEMENTED_YET",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
