"""ACI606, Etapa 8B: prueba de /predict, robustez y concurrencia.

Uso:
 python tests/eval_concurrency_robustness.py --api http://127.0.0.1:8000

No escribe en PostgreSQL: solo GET /health, GET /metrics y POST /predict.
Los archivos de evidencia se generan localmente en reports/.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
import time
from time import perf_counter
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports"
CLASSES = {"NORMAL", "WARNING", "CRITICAL"}
REQUIRED_FIELDS = {"robot_id", "predicted_status", "confidence", "model_version", "inference_ms"}


def sample(robot_id: str = "E8B-CHECK", index: int = 0) -> dict[str, Any]:
    return {
        "robot_id": robot_id,
        "timestamp": round(time.time() + index * 0.001, 6),
        "x": 1.2,
        "y": 3.4,
        "speed": round(0.62 + (index % 20) * 0.015, 4),
        "battery": round(90.0 - (index % 18), 3),
        "motor_temp": round(48.0 + (index % 25), 3),
        "vibration": round(0.42 + (index % 14) * 0.04, 4),
        "motor_current": round(3.8 + (index % 12) * 0.1, 4),
        "obstacle_distance": 1.3,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    rank = (len(vals) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (rank - lo), 3)


def check_prediction(response: requests.Response, expected_robot_id: str) -> tuple[bool, str]:
    if response.status_code != 200:
        return False, f"HTTP {response.status_code}: {response.text[:120]}"
    try:
        data = response.json()
    except ValueError:
        return False, "respuesta no es JSON"
    if not isinstance(data, dict):
        return False, "respuesta no es objeto JSON"
    if REQUIRED_FIELDS - data.keys():
        return False, f"faltan campos {sorted(REQUIRED_FIELDS - data.keys())}"
    if data["robot_id"] != expected_robot_id:
        return False, "robot_id no corresponde al enviado"
    if data["predicted_status"] not in CLASSES:
        return False, "estado de predicción no permitido"
    if not isinstance(data["model_version"], str) or not data["model_version"].strip():
        return False, "model_version inválida"
    try:
        confidence = float(data["confidence"])
        ms = float(data["inference_ms"])
    except (ValueError, TypeError):
        return False, "confidence/inference_ms no son números"
    if not (math.isfinite(confidence) and 0 <= confidence <= 1):
        return False, "confidence fuera de rango"
    if not (math.isfinite(ms) and ms >= 0):
        return False, "inference_ms inválido"
    return True, "correcto"


def get_json(base: str, path: str, timeout: float) -> dict:
    r = requests.get(base + path, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError(f"{path} no devuelve un objeto JSON")
    return data


def test_invalid(base: str, timeout: float) -> list[dict[str, Any]]:
    valid = sample()
    variants: list[tuple[str, dict, bool]] = [
        ("falta_battery", {k: v for k, v in valid.items() if k != "battery"}, False),
        ("battery_negativa", {**valid, "battery": -1}, False),
        ("battery_mayor_100", {**valid, "battery": 120}, False),
        ("speed_negativa", {**valid, "speed": -0.5}, False),
        ("timestamp_negativo", {**valid, "timestamp": -1}, False),
        ("robot_id_con_espacio", {**valid, "robot_id": "ROBOT INVALIDO"}, False),
        ("campo_inesperado", {**valid, "inventado": "no-permitido"}, False),
        ("NaN_no_finito", {**valid, "speed": "NaN"}, False),
    ]
    results = []
    print("\n1. ROBUSTEZ: entradas inválidas")
    for name, payload, raw in variants:
        started = perf_counter()
        try:
            r = requests.post(base + "/predict", json=payload, timeout=timeout)
            elapsed = round((perf_counter() - started) * 1000, 3)
            ok = 400 <= r.status_code < 500
            results.append({"caso": name, "http": r.status_code, "rechazado_4xx": ok,
                            "client_ms": elapsed, "detalle": r.text[:120]})
            print(f"  {name:22} -> HTTP {r.status_code} {'OK' if ok else 'ERROR'}")
        except requests.RequestException as exc:
            results.append({"caso": name, "http": None, "rechazado_4xx": False,
                            "client_ms": None, "detalle": str(exc)[:200]})
            print(f"  {name:22} -> ERROR DE CONEXIÓN: {exc}")
    # Enviar token NaN como literal de JSON para detectar el caso no estándar.
    encoded = json.dumps(valid).replace('"speed": 0.62', '"speed": NaN')
    name = "NaN_literal_json"
    started = perf_counter()
    try:
        r = requests.post(base + "/predict", data=encoded,
                          headers={"Content-Type": "application/json"}, timeout=timeout)
        elapsed = round((perf_counter() - started) * 1000, 3)
        ok = 400 <= r.status_code < 500
        results.append({"caso": name, "http": r.status_code, "rechazado_4xx": ok,
                        "client_ms": elapsed, "detalle": r.text[:120]})
        print(f"  {name:22} -> HTTP {r.status_code} {'OK' if ok else 'ERROR'}")
    except requests.RequestException as exc:
        results.append({"caso": name, "http": None, "rechazado_4xx": False,
                        "client_ms": None, "detalle": str(exc)[:200]})
        print(f"  {name:22} -> ERROR DE CONEXIÓN: {exc}")
    return results


def one_prediction(base: str, timeout: float, stage: str, index: int) -> dict[str, Any]:
    robot_id = f"E8B-{index:05d}"  # cumple máximo de 20 caracteres.
    payload = sample(robot_id, index)
    started = perf_counter()
    try:
        response = requests.post(base + "/predict", json=payload, timeout=timeout)
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        passed, message = check_prediction(response, robot_id)
        return {"fase": stage, "solicitud": index, "http": response.status_code,
                "client_ms": elapsed_ms, "correcto": passed, "detalle": message}
    except requests.RequestException as exc:
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        return {"fase": stage, "solicitud": index, "http": None,
                "client_ms": elapsed_ms, "correcto": False,
                "detalle": f"Error de conexión: {str(exc)[:180]}"}


def batch(base: str, timeout: float, workers: int, total: int, start_id: int) -> tuple[list[dict], dict]:
    name = f"{workers}_clientes"
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one_prediction, base, timeout, name, start_id + i)
                   for i in range(total)]
        # No detener inmediatamente ante un error: conservar evidencia de toda la ronda.
        data = [f.result() for f in as_completed(futures)]
    duration = perf_counter() - started
    data.sort(key=lambda x: x["solicitud"])
    latencies = [float(row["client_ms"]) for row in data if row["correcto"]]
    okay = sum(bool(row["correcto"]) for row in data)
    summary = {"clientes_concurrentes": workers, "solicitudes": total, "correctas": okay,
               "fallidas": total - okay, "duracion_s": round(duration, 3),
               "solicitudes_confirmadas_por_segundo": round(okay / max(duration, 0.001), 2),
               "latencia_cliente_p50_ms": percentile(latencies, 0.50),
               "latencia_cliente_p95_ms": percentile(latencies, 0.95),
               "latencia_cliente_max_ms": round(max(latencies), 3) if latencies else None,
               "codigos_http": dict(Counter(str(row["http"]) for row in data))}
    return data, summary


def create_chart(rounds: list[dict], filename: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("AVISO: matplotlib no está instalado, no se generará el PNG.")
        return False
    labels = [str(x["clientes_concurrentes"]) + " clientes" for x in rounds]
    p50s = [x["latencia_cliente_p50_ms"] for x in rounds]
    p95s = [x["latencia_cliente_p95_ms"] for x in rounds]
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([v - width / 2 for v in x], p50s, width, label="p50")
    ax.bar([v + width / 2 for v in x], p95s, width, label="p95")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Latencia desde el cliente (ms)")
    ax.set_title("Etapa 8B: latencias de POST /predict bajo concurrencia")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Prueba Etapa 8B: robustez, concurrencia y latencias")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="URL base de la API")
    parser.add_argument("--timeout", type=float, default=30.0, help="Máximo por solicitud, segundos")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout debe ser positivo")
    base = args.api.rstrip("/")
    print("ACI606 - Etapa 8B | API:", base)
    print("La prueba NO escribe telemetría ni altera el modelo: usa /predict.")
    try:
        health = get_json(base, "/health", args.timeout)
        if health.get("status") != "ok" or health.get("database") != "ok" or health.get("model_ready") is not True:
            raise ValueError("/health no indica servidor, DB y modelo listos")
        before = get_json(base, "/metrics", args.timeout)
    except (requests.RequestException, ValueError) as exc:
        print("ERROR: API no preparada:", exc)
        return 2
    print("Conectado. Versión:", health.get("version"), "| BD:", health.get("storage"))

    invalid = test_invalid(base, args.timeout)
    print("\n2. CALENTAMIENTO: tres predicciones válidas, no incluidas en las latencias")
    warmup = [one_prediction(base, args.timeout, "calentamiento", i) for i in range(3)]
    for row in warmup:
        print(f"  /predict -> HTTP {row['http']} | {row['client_ms']} ms | {'OK' if row['correcto'] else 'ERROR'}")

    rounds: list[dict] = []
    all_rows: list[dict] = []
    # Carga acotada para no sobrecargar una cuenta gratuita.
    print("\n3. CONCURRENCIA: rondas moderadas de 1, 4 y 8 clientes")
    for workers, total, start_id in ((1, 10, 1000), (4, 24, 2000), (8, 64, 3000)):
        data, summary = batch(base, args.timeout, workers, total, start_id)
        all_rows.extend(data)
        rounds.append(summary)
        print(f"  {workers} clientes: {summary['correctas']}/{total} correctas | "
              f"p50={summary['latencia_cliente_p50_ms']} ms | "
              f"p95={summary['latencia_cliente_p95_ms']} ms | "
              f"duración={summary['duracion_s']} s")
        if summary["fallidas"]:
            for bad in (x for x in data if not x["correcto"]):
                print("   FALLÓ", bad["solicitud"], bad["http"], bad["detalle"])

    try:
        after_health = get_json(base, "/health", args.timeout)
        after = get_json(base, "/metrics", args.timeout)
    except (requests.RequestException, ValueError) as exc:
        print("ERROR: el servidor no respondió después de la carga:", exc)
        after_health = {}
        after = {}

    errors_before = int(before.get("errors_5xx_since_boot", 0))
    errors_after = int(after.get("errors_5xx_since_boot", errors_before))
    invalid_pass = all(x["rechazado_4xx"] for x in invalid)
    warmup_pass = all(x["correcto"] for x in warmup)
    concurrent_pass = all(x["fallidas"] == 0 for x in rounds)
    available_after = after_health.get("status") == "ok" and after_health.get("model_ready") is True
    no_new_5xx = errors_after == errors_before
    before_rows = before.get("total_telemetry_persisted")
    after_rows = after.get("total_telemetry_persisted")
    persisted_equal = before_rows == after_rows if before_rows is not None and after_rows is not None else None
    passed = invalid_pass and warmup_pass and concurrent_pass and available_after and no_new_5xx

    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_file = OUTPUT / "eval_8b_latencias.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fase", "solicitud", "http", "client_ms", "correcto", "detalle"])
        writer.writeheader()
        writer.writerows(all_rows)
    graph_file = OUTPUT / "eval_8b_concurrencia.png"
    chart_exists = create_chart(rounds, graph_file)
    summary_file = OUTPUT / "eval_8b_resumen.json"
    summary_report = {
        "evaluation": "ACI606 Etapa 8B - concurrencia, robustez y latencia",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "api": base, "version": health.get("version"), "storage": health.get("storage"),
        "timeout_por_solicitud_segundos": args.timeout,
        "invalid_cases": [{k: v for k, v in item.items() if k != "detalle"} for item in invalid],
        "valid_warmup": warmup,
        "rounds": rounds,
        "errors_5xx_since_boot_before": errors_before,
        "errors_5xx_since_boot_after": errors_after,
        "health_after": {k: after_health.get(k) for k in ("status", "database", "model_ready")},
        "total_telemetry_before": before_rows,
        "total_telemetry_after": after_rows,
        "telemetry_unchanged_during_test": persisted_equal,
        "passed": passed,
        "notes": [
            "Este script NO ejecuta POST /telemetry y NO actualiza registros en PostgreSQL.",
            "Las métricas de /metrics son acumuladas desde el arranque y pueden incluir tráfico del dashboard.",
            "Los percentiles de esta evaluación se miden desde el cliente que hace las solicitudes.",
            "Si el simulador está funcionando en paralelo, los contadores y total de registros pueden variar.",
            "La prueba local NO sustituye repetirla contra la URL pública de Render.",
        ],
    }
    summary_file.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==== RESUMEN FINAL ETAPA 8B ====")
    print(f"Entradas inválidas rechazadas con 4xx: {sum(x['rechazado_4xx'] for x in invalid)}/{len(invalid)}")
    print(f"Calentamiento de /predict: {sum(x['correcto'] for x in warmup)}/{len(warmup)}")
    print(f"Concurrencia: {sum(x['correctas'] for x in rounds)}/{sum(x['solicitudes'] for x in rounds)} respuestas válidas")
    print(f"Servidor operativo después de carga: {'SÍ' if available_after else 'NO'}")
    print(f"Errores 5xx nuevos según /metrics: {errors_after - errors_before}")
    print(f"Telemetría persistida: antes={before_rows} después={after_rows} "
          f"(sin cambios={'SÍ' if persisted_equal else 'NO/NO DETERMINADO'})")
    print("Evidencias:", csv_file.relative_to(ROOT), ",", summary_file.relative_to(ROOT))
    if chart_exists:
        print("Gráfico:", graph_file.relative_to(ROOT))
    if passed:
        print("\nETAPA 8B APROBADA: robustez, concurrencia y mediciones de latencia desde el cliente.")
        return 0
    print("\nETAPA 8B REVISAR: una o más comprobaciones fallaron. Revisa las evidencias.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
