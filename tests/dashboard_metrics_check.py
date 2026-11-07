"""Validación end-to-end de Etapa 7 contra FastAPI en ejecución.

Usa robot de prueba DASH-ETAPA7; NO borra registros. Ejecutar:
  python tests/dashboard_metrics_check.py --api http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import sys
import time

import requests


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("ERROR: " + message)
    print("CORRECTO:", message)


def request(method: str, url: str, **kwargs):
    return requests.request(method, url, timeout=35, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pruebas del dashboard y métricas de etapa 7")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base = args.api.rstrip("/")
    print("API:", base)

    health_response = request("GET", base + "/health")
    check(health_response.status_code == 200, "GET /health responde 200")
    health = health_response.json()
    print("Versión:", health.get("version"), "| Base:", health.get("storage"))
    check(health.get("database") == "ok", "base de datos disponible")
    check(health.get("model_ready") is True, "modelo entrenado cargado")

    dashboard = request("GET", base + "/dashboard")
    check(dashboard.status_code == 200, "GET /dashboard responde 200")
    check("text/html" in dashboard.headers.get("content-type", ""), "dashboard devuelve HTML")
    for text in ("Panel de flota virtual", "Robots de la flota", "Historial reciente", "Métricas de operación"):
        check(text in dashboard.text, f"dashboard contiene '{text}'")

    robot_id = "DASH-ETAPA7"
    common = {
        "robot_id": robot_id,
        "x": 1.2, "y": 3.4, "speed": 0.72, "battery": 81.0,
        "motor_temp": 49.0, "vibration": 0.42,
        "motor_current": 3.8, "obstacle_distance": 1.3,
    }
    for battery in (81.0, 79.0, 76.0):
        payload = {**common, "battery": battery, "timestamp": time.time()}
        response = request("POST", base + "/telemetry", json=payload)
        check(response.status_code == 200 and response.json().get("accepted") is True,
              f"telemetría {battery:.1f}% guardada")
        check(response.json().get("predicted_status") in {"NORMAL", "WARNING", "CRITICAL"},
              "telemetría contiene predicción real")

    r = request("GET", base + "/robots")
    check(r.status_code == 200 and isinstance(r.json(), list), "/robots devuelve lista")
    check(any(robot.get("robot_id") == robot_id for robot in r.json()), "robot de prueba visible en el panel")

    full = request("GET", base + f"/robots/{robot_id}/history")
    check(full.status_code == 200 and len(full.json()) >= 3, "historial completo sigue disponible")
    last_two = request("GET", base + f"/robots/{robot_id}/history?limit=2")
    check(last_two.status_code == 200 and len(last_two.json()) == 2, "limit=2 selecciona solo las últimas dos lecturas")
    check(last_two.json()[-1]["battery"] == 76.0, "historial limitado mantiene orden cronológico")
    invalid_limit = request("GET", base + f"/robots/{robot_id}/history?limit=0")
    check(invalid_limit.status_code == 422, "límite inválido se rechaza con HTTP 422")
    missing = request("GET", base + "/robots/ROBOT-NO-EXISTE-XYZ/history")
    check(missing.status_code == 404, "robot desconocido genera HTTP 404")

    prediction = request("POST", base + "/predict", json={**common, "timestamp": time.time()})
    check(prediction.status_code == 200 and prediction.json().get("model_version") == "v1",
          "POST /predict sigue respondiendo con el modelo entrenado")

    m = request("GET", base + "/metrics")
    check(m.status_code == 200, "GET /metrics responde 200")
    metrics = m.json()
    required = [
        "http_requests_since_boot", "errors_4xx_since_boot", "errors_5xx_since_boot",
        "http_latency_p50_ms", "http_latency_p95_ms", "predict_http_latency_p50_ms",
        "predict_http_latency_p95_ms", "uptime_seconds", "inference_count_since_boot",
        "total_telemetry_persisted", "total_robots_persisted", "prediction_service",
    ]
    for key in required:
        check(key in metrics, f"/metrics contiene {key}")
    check(metrics["errors_4xx_since_boot"] >= 2, "errores 4xx contabilizados (404 y 422)")
    check(metrics["http_requests_since_boot"] >= 12, "solicitudes contabilizadas")
    check(metrics["http_latency_p95_ms"] >= metrics["http_latency_p50_ms"] >= 0,
          "percentiles HTTP coherentes")
    check(metrics["predict_http_latency_p95_ms"] >= 0, "latencia de /predict registrada")
    check(metrics["uptime_seconds"] >= 0, "uptime válido")
    check(metrics["total_telemetry_persisted"] >= 3, "muestras persistidas")
    print("\nETAPA 7 APROBADA: dashboard, histórico completo, inferencia y observabilidad.")
    print("Nota: el panel se debe revisar visualmente en el navegador.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print("ERROR de conexión:", exc, file=sys.stderr)
        raise SystemExit(1)
