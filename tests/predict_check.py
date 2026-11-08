"""Comprueba /predict, validaciones y predicción persistida en la API (Etapa 6).

Ejecutar con Uvicorn activo, en otra terminal:
    python tests/predict_check.py
"""
from __future__ import annotations

import argparse
import time

import requests


SAMPLE = {
    "robot_id": "PRED-ETAPA6",
    "timestamp": time.time(),
    "x": 1.2,
    "y": 3.4,
    "speed": 0.72,
    "battery": 81.0,
    "motor_temp": 49.0,
    "vibration": 0.42,
    "motor_current": 3.8,
    "obstacle_distance": 1.3,
}
LABELS = {"NORMAL", "WARNING", "CRITICAL"}
REQUIRED = {"robot_id", "predicted_status", "confidence", "model_version", "inference_ms"}


def check(cond: bool, text: str) -> None:
    if not cond:
        raise SystemExit(f"ERROR: {text}")
    print(f"CORRECTO: {text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.api.rstrip("/")

    health_response = requests.get(base + "/health", timeout=30)
    health_response.raise_for_status()
    health = health_response.json()
    print("Servidor:", health.get("version"), "| BD:", health.get("storage"))
    check(health.get("database") == "ok", "base de datos conectada")
    check(health.get("model_ready") is True, "modelo de Machine Learning cargado")

    r = requests.post(base + "/predict", json={**SAMPLE, "timestamp": time.time()}, timeout=30)
    print("POST /predict ->", r.status_code)
    check(r.status_code == 200, f"/predict debe devolver 200, recibido: {r.text[:200]}")
    result = r.json()
    print("Predicción:", result)
    check(REQUIRED.issubset(result.keys()), "los cinco campos obligatorios están presentes")
    check(result["robot_id"] == SAMPLE["robot_id"], "robot_id se conserva")
    check(result["predicted_status"] in LABELS, "estado válido")
    check(0.0 <= result["confidence"] <= 1.0, "confianza entre 0 y 1")
    check(result["model_version"] == "v1", "versión del modelo v1")
    check(result["inference_ms"] >= 0.0, "tiempo de inferencia no negativo")

    invalid = {**SAMPLE, "battery": 150.0, "timestamp": time.time()}
    r = requests.post(base + "/predict", json=invalid, timeout=30)
    check(400 <= r.status_code < 500, "entrada inválida se rechaza con HTTP 4xx")
    nan_sample = {**SAMPLE, "motor_temp": "NaN", "timestamp": time.time()}
    r = requests.post(base + "/predict", json=nan_sample, timeout=30)
    check(400 <= r.status_code < 500, "NaN se rechaza con HTTP 4xx")

    # Una nueva muestra enviada a /telemetry debe guardar predicted_status.
    payload = {**SAMPLE, "timestamp": time.time()}
    r = requests.post(base + "/telemetry", json=payload, timeout=30)
    check(r.status_code == 200, "telemetría aceptada")
    r = requests.get(base + f"/robots/{SAMPLE['robot_id']}/history", timeout=30)
    r.raise_for_status()
    rows = r.json()
    check(len(rows) >= 1 and rows[-1].get("predicted_status") in LABELS,
          "estado predicho almacenado con la telemetría")

    print("\nETAPA 6 APROBADA: inferencia, validación y persistencia de predicción.")


if __name__ == "__main__":
    main()
