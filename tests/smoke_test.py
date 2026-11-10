"""Smoke test del Ejercicio 4 ACI606.

Modo basico (compatible con el starter):
    python tests/smoke_test.py --api http://127.0.0.1:8000

Modo final (diapositiva 15 del profesor):
    python tests/smoke_test.py --api https://TU-SERVICIO.onrender.com --final

IMPORTANTE: la prueba hace UN POST /telemetry y guarda una lectura de prueba.
No borra datos ni modifica el modelo.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import requests

LABELS = {"NORMAL", "WARNING", "CRITICAL"}
PREDICT_FIELDS = {"robot_id", "predicted_status", "confidence", "model_version", "inference_ms"}


def verify(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"CORRECTO: {message}")


def call(session: requests.Session, method: str, url: str, timeout: float, **kwargs) -> requests.Response:
    response = session.request(method, url, timeout=timeout, **kwargs)
    print(f"{method} {url.split('://', 1)[-1].split('/', 1)[-1]:<34} -> HTTP {response.status_code}")
    return response


def expect_200(response: requests.Response, endpoint: str) -> None:
    verify(response.status_code == 200,
           f"{endpoint} responde HTTP 200 (recibido {response.status_code}: {response.text[:140]})")


def json_body(response: requests.Response, endpoint: str):
    try:
        return response.json()
    except ValueError as exc:
        raise AssertionError(f"{endpoint} no devolvio JSON valido") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test ACI606 Cloud Fleet Challenge")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="URL base de FastAPI")
    parser.add_argument("--final", action="store_true", help="Verifica las siete rutas y el contrato de /predict")
    parser.add_argument("--timeout", type=float, default=90, help="Timeout por solicitud en segundos (90 por defecto)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout debe ser mayor que cero")

    base = args.api.rstrip("/")
    robot_id = "SMOKE-FINAL" if args.final else "TEST-01"
    sample = {
        "robot_id": robot_id,
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

    print(f"API: {base}")
    print(f"Modo: {'FINAL (7 rutas)' if args.final else 'BASICO (4 rutas)'}")
    print("Nota: POST /telemetry agregara una lectura; no borra registros.\n")

    with requests.Session() as session:
        # Las cuatro rutas del smoke test original.
        r = call(session, "GET", base + "/health", args.timeout)
        expect_200(r, "/health")
        health = json_body(r, "/health")
        if args.final:
            verify(health.get("database") == "ok", "base de datos disponible")
            verify(health.get("model_ready") is True, "modelo de ML cargado")

        r = call(session, "POST", base + "/telemetry", args.timeout, json=sample)
        expect_200(r, "/telemetry")
        telemetry = json_body(r, "/telemetry")
        if args.final:
            verify(telemetry.get("accepted") is True and telemetry.get("robot_id") == robot_id,
                   "telemetria aceptada para el robot de prueba")
            verify(telemetry.get("predicted_status") in LABELS,
                   "telemetria guardada con prediccion")

        r = call(session, "GET", base + "/robots", args.timeout)
        expect_200(r, "/robots")
        robots = json_body(r, "/robots")
        if args.final:
            verify(isinstance(robots, list) and any(
                isinstance(row, dict) and row.get("robot_id") == robot_id for row in robots
            ), "robot de prueba visible en /robots")

        r = call(session, "GET", base + "/metrics", args.timeout)
        expect_200(r, "/metrics")
        metrics = json_body(r, "/metrics")
        if args.final:
            expected_metrics = {"http_requests_since_boot", "errors_4xx_since_boot",
                                "errors_5xx_since_boot", "http_latency_p50_ms",
                                "http_latency_p95_ms", "uptime_seconds", "total_telemetry_persisted"}
            verify(isinstance(metrics, dict) and expected_metrics.issubset(metrics),
                   "metricas de operacion presentes")

        if not args.final:
            print("\nPRUEBA BASICA APROBADA: cuatro rutas principales responden 200.")
            return

        # Las otras tres rutas exigidas por el profesor.
        r = call(session, "GET", base + f"/robots/{robot_id}/history", args.timeout)
        expect_200(r, "/robots/{id}/history")
        rows = json_body(r, "/robots/{id}/history")
        verify(isinstance(rows, list) and any(
            isinstance(row, dict) and row.get("robot_id") == robot_id
            and math.isclose(float(row.get("timestamp", 0)), sample["timestamp"], abs_tol=0.05)
            for row in rows
        ), "lectura recien enviada recuperable desde el historial")

        r = call(session, "POST", base + "/predict", args.timeout, json=sample)
        expect_200(r, "/predict")
        prediction = json_body(r, "/predict")
        verify(isinstance(prediction, dict) and PREDICT_FIELDS.issubset(prediction),
               "/predict devuelve los cinco campos obligatorios")
        verify(prediction["robot_id"] == robot_id and prediction["predicted_status"] in LABELS,
               "identificador y estado predicho validos")
        verify(isinstance(prediction["confidence"], (int, float))
               and math.isfinite(prediction["confidence"])
               and 0 <= prediction["confidence"] <= 1,
               "confianza de prediccion entre 0 y 1")
        verify(isinstance(prediction["model_version"], str) and bool(prediction["model_version"]),
               "version del modelo presente")
        verify(isinstance(prediction["inference_ms"], (int, float))
               and math.isfinite(prediction["inference_ms"])
               and prediction["inference_ms"] >= 0,
               "tiempo de inferencia valido")

        r = call(session, "GET", base + "/dashboard", args.timeout)
        expect_200(r, "/dashboard")
        verify("text/html" in r.headers.get("content-type", "").lower(),
               "dashboard devuelve contenido HTML")

        r = call(session, "POST", base + "/predict", args.timeout,
                 json={**sample, "battery": 150.0})
        verify(400 <= r.status_code < 500,
               "bateria fuera de rango rechazada con HTTP 4xx")

    print("\nPRUEBA FINAL APROBADA: siete rutas, /predict y validacion de datos.")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, AssertionError, TypeError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
