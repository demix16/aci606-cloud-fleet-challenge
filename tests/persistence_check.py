"""Envía 105 muestras o revisa que sigan después de reiniciar la API.

Ejecutar con el servidor local activo:
python tests/persistence_check.py --mode send
# Detener y reiniciar uvicorn en otra terminal.
python tests/persistence_check.py --mode verify
"""
from __future__ import annotations

import argparse
import time
import sys

import requests

SAMPLE = {
    "robot_id": "HIST-ETAPA4",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=["send", "verify"], required=True)
    args = parser.parse_args()
    base = args.api.rstrip("/")
    robot_id = SAMPLE["robot_id"]
    path = f"/robots/{robot_id}/history"

    if args.mode == "send":
        print("Enviando 105 muestras. No cierres el servidor...")
        with requests.Session() as client:
            for i in range(105):
                payload = {**SAMPLE, "timestamp": time.time() + i}
                response = client.post(base + "/telemetry", json=payload, timeout=20)
                if response.status_code != 200:
                    print(f"Error en muestra {i+1}: HTTP {response.status_code} {response.text[:200]}")
                    sys.exit(1)
        print("105 muestras aceptadas por la API.")

    response = requests.get(base + path, timeout=30)
    response.raise_for_status()
    records = response.json()
    print(f"Historial de {robot_id}: {len(records)} registros")
    if len(records) < 105:
        print("ERROR: se esperaban 105 registros como mínimo.")
        sys.exit(1)
    print("CORRECTO: el historial contiene más de 100 registros.")
    if args.mode == "send":
        print("SIGUIENTE: detén y reinicia Uvicorn; luego ejecuta --mode verify.")
    else:
        print("CORRECTO: los registros siguen disponibles después del reinicio (si realmente reiniciaste el servidor).")


if __name__ == "__main__":
    main()
