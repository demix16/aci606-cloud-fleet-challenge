"""ACI606 · Etapa 8A: evaluar /predict con flotas NUEVAS y etiquetas reales.

El generador oficial produce (telemetría, status_real). El status_real NO se
manda a la API: se reserva exclusivamente para calcular las métricas.

No se modifica el pipeline, no se reentrena y no se hacen POST /telemetry.
Requiere conservar data/training.csv y reports/metricas.json de la Etapa 3.

Ejemplo en Windows:
  python tests/eval_unseen_fleets.py --api http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.core import make_robot, step_robot  # noqa: E402

LABELS = ["NORMAL", "WARNING", "CRITICAL"]
REQUIRED_RESPONSE = {"robot_id", "predicted_status", "confidence", "model_version", "inference_ms"}


def percentile(values: list[float], frac: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * frac
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (index - low), 3)


def build_new_samples(seed: int, fleets: int, robots_per_fleet: int, steps: int) -> list[dict]:
    generated = []
    for fleet_no in range(1, fleets + 1):
        fleet_id = f"EVAL-{seed}-F{fleet_no:02d}"
        rng = random.Random(seed + 1009 * fleet_no)
        # Condiciones levemente diferentes por flota: desplazamientos de sensor
        # muy pequeños. No se modifica el estado verdadero ni el simulador base.
        temp_offset = ((fleet_no % 3) - 1) * 0.7           # -0.7, 0 o +0.7 °C
        speed_factor = 1 + ((fleet_no % 3) - 1) * 0.015    # ±1,5 %
        vibration_factor = 1 + ((fleet_no % 4) - 1.5) * 0.015  # ±2,25 %
        robots = [make_robot(i + 1, rng) for i in range(robots_per_fleet)]
        for i, robot in enumerate(robots, start=1):
            # Diferentes de R01...R05 utilizados en el dataset del profesor.
            robot.robot_id = f"EV{seed}F{fleet_no:02d}R{i:02d}"
            if len(robot.robot_id) > 20:
                raise ValueError("robot_id demasiado largo para el contrato FastAPI")
        for step in range(1, steps + 1):
            for robot in robots:
                telemetry, actual = step_robot(robot, rng)
                telemetry["motor_temp"] = round(telemetry["motor_temp"] + temp_offset, 3)
                telemetry["speed"] = round(min(1.5, max(0, telemetry["speed"] * speed_factor)), 4)
                telemetry["vibration"] = round(max(0, telemetry["vibration"] * vibration_factor), 4)
                generated.append({
                    "fleet_id": fleet_id,
                    "step": step,
                    "telemetry": telemetry,
                    "actual": actual,
                })
    return generated


def get_training_majority() -> tuple[str, int]:
    """Línea base calculada SOLO sobre las flotas de desarrollo previas."""
    metadata_file = ROOT / "reports" / "metricas.json"
    training_file = ROOT / "data" / "training.csv"
    if not metadata_file.exists() or not training_file.exists():
        raise SystemExit(
            "Falta reports/metricas.json o data/training.csv. "
            "La línea base debe usar datos de entrenamiento, no los resultados de esta prueba."
        )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    allowed_fleets = set(metadata.get("train_fleets", []))
    if not allowed_fleets:
        raise SystemExit("No hay 'train_fleets' en reports/metricas.json")
    frequencies = Counter()
    with training_file.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("fleet_id") in allowed_fleets:
                frequencies[row["status"]] += 1
    if not frequencies:
        raise SystemExit("No se encontraron flotas de entrenamiento en data/training.csv")
    return frequencies.most_common(1)[0][0], sum(frequencies.values())


def run(args: argparse.Namespace) -> int:
    if args.seed == 606:
        raise SystemExit("Usa una semilla NUEVA (la de entrenamiento fue 606)")
    if min(args.fleets, args.robots, args.steps, args.max_eval) <= 0:
        raise SystemExit("--fleets, --robots, --steps y --max-eval deben ser mayores que cero")
    base = args.api.rstrip("/")
    session = requests.Session()
    try:
        health = session.get(base + "/health", timeout=args.timeout)
        health.raise_for_status()
        health_info = health.json()
    except (requests.RequestException, ValueError) as exc:
        raise SystemExit(f"No se puede consultar /health en {base}: {exc}")
    if health_info.get("model_ready") is not True:
        raise SystemExit("El modelo no está listo. Revisa /health: model_ready debe ser true")
    print("Conexión a API confirmada. Modelo listo. Storage:", health_info.get("storage"))

    majority, n_development = get_training_majority()
    all_samples = build_new_samples(args.seed, args.fleets, args.robots, args.steps)
    selector = random.Random(args.seed + 88881)
    selected = selector.sample(all_samples, min(args.max_eval, len(all_samples)))
    if set(x["actual"] for x in selected) != set(LABELS):
        raise SystemExit("La muestra seleccionada no contiene las 3 clases. Aumenta --max-eval.")
    metadata = json.loads((ROOT / "reports" / "metricas.json").read_text(encoding="utf-8"))
    training_ids = set(metadata.get("train_fleets", []) + metadata.get("test_fleets", []))
    test_fleets = {x["fleet_id"] for x in selected}
    if training_ids.intersection(test_fleets):
        raise SystemExit("Se encontró una flota de entrenamiento en evaluación")
    print(f"Flotas generadas: {args.fleets}; robots: {args.robots} por flota")
    print(f"Muestras generadas: {len(all_samples)}; evaluadas: {len(selected)}; semilla: {args.seed}")
    print(f"Flotas nuevas representadas en evaluación: {len(test_fleets)}")
    print("Clases reales:", dict(Counter(x["actual"] for x in selected)))
    print(f"Clase mayoritaria del entrenamiento ({n_development} registros): {majority}")
    print("Solo se usa POST /predict: NO se añaden registros a Neon.")

    results: list[dict] = []
    for i, item in enumerate(selected, start=1):
        start = perf_counter()
        try:
            response = session.post(base + "/predict", json=item["telemetry"], timeout=args.timeout)
            elapsed_ms = (perf_counter() - start) * 1000
        except requests.RequestException as exc:
            raise SystemExit(f"Solicitud {i}/{len(selected)} falló: {exc}. Consulta Uvicorn.")
        if response.status_code != 200:
            raise SystemExit(
                f"Solicitud {i}/{len(selected)} devolvió HTTP {response.status_code}: "
                f"{response.text[:400]}"
            )
        try:
            predicted = response.json()
        except ValueError:
            raise SystemExit(f"Solicitud {i} no devolvió JSON válido")
        missing = REQUIRED_RESPONSE - set(predicted)
        if missing:
            raise SystemExit(f"Solicitud {i}: faltan campos {sorted(missing)}")
        label = predicted["predicted_status"]
        if label not in LABELS or predicted["robot_id"] != item["telemetry"]["robot_id"]:
            raise SystemExit(f"Solicitud {i}: robot_id o estado fuera del contrato")
        confidence = float(predicted["confidence"])
        inference_ms = float(predicted["inference_ms"])
        if not (math.isfinite(confidence) and 0 <= confidence <= 1 and
                math.isfinite(inference_ms) and inference_ms >= 0):
            raise SystemExit(f"Solicitud {i}: confianza o inferencia inválida")
        results.append({
            "fleet_id": item["fleet_id"],
            "robot_id": item["telemetry"]["robot_id"],
            "step": item["step"],
            "real": item["actual"],
            "predicho": label,
            "correcto": label == item["actual"],
            "confidence": confidence,
            "model_version": str(predicted["model_version"]),
            "inference_ms": inference_ms,
            "client_ms": round(elapsed_ms, 3),
        })
        if i % 40 == 0 or i == len(selected):
            print(f"  Predicciones consultadas: {i}/{len(selected)}")

    truth = [r["real"] for r in results]
    guesses = [r["predicho"] for r in results]
    baseline_predictions = [majority] * len(results)
    score = float(f1_score(truth, guesses, labels=LABELS, average="macro", zero_division=0))
    baseline = float(f1_score(truth, baseline_predictions, labels=LABELS, average="macro", zero_division=0))
    acc = float(accuracy_score(truth, guesses))
    cm = confusion_matrix(truth, guesses, labels=LABELS)
    report = classification_report(truth, guesses, labels=LABELS, zero_division=0, output_dict=True)
    p50 = percentile([r["client_ms"] for r in results], 0.50)
    p95 = percentile([r["client_ms"] for r in results], 0.95)

    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    plot_path = output / "eval_no_vista_matriz.png"
    png = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    png.plot(cmap="Blues", values_format="d", colorbar=True)
    png.ax_.set_title("Evaluación externa: flotas nuevas")
    png.ax_.set_xlabel("Estado predicho")
    png.ax_.set_ylabel("Estado real")
    png.figure_.tight_layout()
    png.figure_.savefig(plot_path, dpi=160)
    plt.close(png.figure_)

    csv_path = output / "eval_no_vista_detalle.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    passed = score > baseline
    summary = {
        "evaluation": "ACI606 Etapa 8A - flotas externas al entrenamiento",
        "api": base,
        "seed": args.seed,
        "fleets_generated": args.fleets,
        "fleets_evaluated": sorted(test_fleets),
        "robots_per_fleet": args.robots,
        "sensor_shift": "Temperatura ±0.7 C, velocidad ±1.5 %, vibración ±2.25 % según flota",
        "steps_per_robot": args.steps,
        "samples_generated": len(all_samples),
        "samples_evaluated": len(results),
        "training_majority_class": majority,
        "actual_class_distribution": dict(Counter(truth)),
        "predicted_class_distribution": dict(Counter(guesses)),
        "macro_f1": round(score, 6),
        "baseline_macro_f1": round(baseline, 6),
        "accuracy": round(acc, 6),
        "confusion_labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "latency_client_p50_ms": p50,
        "latency_client_p95_ms": p95,
        "model_versions": sorted(set(r["model_version"] for r in results)),
        "higher_than_baseline": passed,
        "notes": [
            "Los estados reales provienen de simulator.core.step_robot y NO se envían a /predict.",
            "El muestreo aleatorio se realizó antes de consultar el modelo.",
            "Se incluyeron perturbaciones pequeñas de sensores por flota, sin cambiar el estado real oculto.",
            "La prueba no reentrena el pipeline y no hace POST /telemetry.",
            "Los percentiles de latencia son medidos desde este cliente local.",
            "Las muestras repetidas de un mismo robot están correlacionadas temporalmente; no equivalen a observaciones independientes.",
        ],
    }
    json_path = output / "eval_no_vista_resumen.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n==== RESULTADOS DE FLOTA NO VISTA ====")
    print("F1 macro:", f"{score:.4f}")
    print("F1 macro base mayoritaria:", f"{baseline:.4f}")
    print("Exactitud:", f"{acc:.2%}")
    print("Latencia /predict desde el cliente: p50", p50, "ms; p95", p95, "ms")
    print("Matriz de confusión (filas=real, columnas=predicho):")
    print("Etiquetas:", LABELS)
    print(cm)
    print("\nInforme por clase:")
    print(classification_report(truth, guesses, labels=LABELS, zero_division=0))
    print("Evidencias generadas:")
    print("  ", json_path.relative_to(ROOT))
    print("  ", plot_path.relative_to(ROOT))
    print("  ", csv_path.relative_to(ROOT))
    if passed:
        print("\nETAPA 8A APROBADA: F1 macro superior a línea base en flotas nuevas.")
        return 0
    print("\nETAPA 8A REVISAR: el F1 macro NO supera la línea base en esta ejecución.")
    return 2


def main():
    parser = argparse.ArgumentParser(description="Evaluación de flotas nuevas sin reentrenar")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="URL base de FastAPI")
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--fleets", type=int, default=8)
    parser.add_argument("--robots", type=int, default=6)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--max-eval", type=int, default=320)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
