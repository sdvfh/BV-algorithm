"""Bernstein-Vazirani experiment runner with metrics and plotting.

Original behavior (plot generation with skip-if-exists) is preserved. Added:
- capture of predicted secrets for each run (ideal, noisy, readout-injected, real)
- computation of accuracy, precision, recall, and F1-score vs. expected secrets
- CSV summaries for per-run predictions and aggregated metrics
"""

from __future__ import annotations

from collections.abc import Iterable
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import ReadoutError
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

DEFAULT_SHOTS = 1024
DEFAULT_SEED = 42
EXCLUDED_BACKENDS = {"ibm_marrakesh"}
RESULTS_ROOT = Path("Plot_results")
HISTOGRAM_DATA_ROOT = RESULTS_ROOT / "hist_data"
READOUT_NOISE_LEVELS = {
    # Real backends (fez ≈1.0e-2, torino ≈3.0e-2) should sit inside "low".
    # Higher levels explore stress scenarios beyond current hardware error rates.
    "ultra-low": 1e-3,
    "low": 2e-2,
    "medium": 5e-2,
    "high": 1e-1,
    "very-high": 2e-1,
}
READOUT_COLORS = {
    "ultra-low": "#2ca02c",
    "low": "#1f77b4",
    "medium": "#ff7f0e",
    "high": "#d62728",
    "very-high": "#9467bd",
}
READOUT_IDEAL_COLOR = "#f1c40f"  # bright gold to stand apart from error scale
# Histogram palette: ideal (gold), emulated (blue), real (crimson)
IDEAL_OVERLAY_COLOR = READOUT_IDEAL_COLOR
EMULATED_COLOR = "#1f77b4"
REAL_OVERLAY_COLOR = "#c43c39"
SECRET_STRINGS = [
    "10000",
    # "01000",
    # "00100",
    # "00010",
    # "00001",
    # "00011",
    # "00111",
    # "01111",
    # "11111",
    # "11000",
    # "11100",
    # "10100",
    # "10001",
    # "10101",
    # "01010",
    # "01100",
    # "11011",
    # "10111",
    # "01101",
    # "10011",
    # "01001",
    # "00101",
    # "01110",
    # "11010",
    # "10110",
]
PREDICTIONS_CSV = RESULTS_ROOT / "bv_predictions.csv"
METRICS_CSV = RESULTS_ROOT / "bv_metrics.csv"


@dataclass
class RunRecord:
    """Container for a single execution result."""

    category: str
    backend: str
    secret: str
    prediction: str
    counts: dict[str, int]

    @property
    def correct(self) -> bool:
        return self.prediction == self.secret

    @property
    def total_shots(self) -> int:
        return sum(self.counts.values())

    @property
    def top_probability(self) -> float:
        if not self.counts:
            return 0.0
        top_count = max(self.counts.values())
        shots = self.total_shots
        return top_count / shots if shots else 0.0


def bernstein_vazirani_oracle(secret: str) -> QuantumCircuit:
    """Create the oracle circuit U_s for the Bernstein-Vazirani algorithm."""
    qc = QuantumCircuit(len(secret) + 1)
    for index, bit in enumerate(reversed(secret)):
        if bit == "1":
            qc.cx(index, len(secret))
    return qc


def bernstein_vazirani_circuit(secret: str) -> QuantumCircuit:
    """Build the full Bernstein-Vazirani circuit for a given secret string."""
    num_qubits = len(secret)
    circuit = QuantumCircuit(num_qubits + 1, num_qubits)

    circuit.x(num_qubits)
    circuit.h(num_qubits)

    circuit.h(range(num_qubits))

    circuit.compose(bernstein_vazirani_oracle(secret), inplace=True)

    circuit.h(range(num_qubits))

    circuit.measure(range(num_qubits), range(num_qubits))
    return circuit


def build_runtime_service() -> QiskitRuntimeService:
    """Instantiate the IBM Runtime service, honoring an optional channel env var."""
    load_dotenv()
    channel = os.getenv("IBM_QUANTUM_SERVICE")
    if channel:
        return QiskitRuntimeService(channel=channel)
    return QiskitRuntimeService()


def available_backends(service: QiskitRuntimeService):
    """Return all backends except those explicitly excluded."""
    return [backend for backend in service.backends() if backend.name not in EXCLUDED_BACKENDS]


def build_plot_path(category: str, secret: str, backend_name: str | None = None) -> Path:
    """Create the output path for a plot, ensuring a consistent naming scheme."""
    filename = f"{secret}.png" if backend_name is None else f"{secret}_{backend_name}.png"
    return RESULTS_ROOT / category / filename


def histogram_json_path(category: str, secret: str, backend_name: str | None = None, tag: str | None = None) -> Path:
    """Create the output path for serialized histogram counts."""
    suffix = f"_{backend_name}" if backend_name else ""
    tag_part = f"_{tag}" if tag else ""
    filename = f"{secret}{suffix}{tag_part}.json"
    return HISTOGRAM_DATA_ROOT / category / filename


def save_histogram_data(path: Path, counts: dict[str, int]) -> None:
    """Persist histogram counts to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)


def load_histogram_data(path: Path) -> dict[str, int] | None:
    """Load histogram counts from JSON if available."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as err:
        print(f"Failed to read histogram data from {path}: {err}")
        return None


def plot_and_save(counts, title: str, output_path: Path) -> None:
    """Plot histogram counts with a standard title and save to disk."""
    plt.clf()
    plot_histogram(counts)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def predict_from_counts(counts: dict[str, int]) -> str:
    """Return the bitstring with the highest count."""
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def record_from_counts(category: str, backend_name: str, secret: str, counts: dict[str, int]) -> RunRecord:
    """Create a RunRecord populated from measurement counts."""
    prediction = predict_from_counts(counts)
    return RunRecord(
        category=category,
        backend=backend_name,
        secret=secret,
        prediction=prediction,
        counts=counts,
    )


def run_ideal_simulation(secret: str) -> RunRecord | None:
    """Execute the Bernstein-Vazirani circuit on an ideal simulator."""
    json_path = histogram_json_path("ideal", secret)
    cached = load_histogram_data(json_path)
    if cached is not None:
        return record_from_counts("ideal", "aer_simulator_ideal", secret, cached)

    circuit = bernstein_vazirani_circuit(secret)
    backend = AerSimulator(seed_simulator=DEFAULT_SEED)
    compiled = transpile(
        circuit,
        backend,
        optimization_level=3,
        seed_transpiler=DEFAULT_SEED,
        layout_method="sabre",
        routing_method="sabre",
    )

    result = backend.run(compiled, shots=DEFAULT_SHOTS, seed_simulator=DEFAULT_SEED).result()
    counts = result.get_counts()
    save_histogram_data(json_path, counts)
    return record_from_counts("ideal", "aer_simulator_ideal", secret, counts)


def run_noisy_simulation(secret: str, backend) -> RunRecord | None:
    """Execute the circuit on a noisy simulator derived from a specific backend."""
    json_path = histogram_json_path("noisy", secret, backend_name=backend.name)
    cached = load_histogram_data(json_path)
    if cached is not None:
        return record_from_counts("noisy", backend.name, secret, cached)

    circuit = bernstein_vazirani_circuit(secret)
    noise_model = NoiseModel.from_backend(backend)
    noisy_backend = AerSimulator(noise_model=noise_model, seed_simulator=DEFAULT_SEED)

    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=noisy_backend,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=DEFAULT_SEED,
    )
    compiled = pass_manager.run(circuit)

    result = noisy_backend.run(compiled, shots=DEFAULT_SHOTS, seed_simulator=DEFAULT_SEED).result()
    counts = result.get_counts()

    save_histogram_data(json_path, counts)
    return record_from_counts("noisy", backend.name, secret, counts)


def run_real_execution(secret: str, backend) -> RunRecord | None:
    """Execute the circuit on a real backend using the runtime Sampler."""
    json_path = histogram_json_path("real", secret, backend_name=backend.name)
    cached = load_histogram_data(json_path)
    if cached is not None:
        return record_from_counts("real", backend.name, secret, cached)

    circuit = bernstein_vazirani_circuit(secret)
    compiled = transpile(
        circuit,
        backend,
        optimization_level=3,
        seed_transpiler=DEFAULT_SEED,
        layout_method="sabre",
        routing_method="sabre",
    )

    sampler = Sampler(mode=backend)
    job = sampler.run([compiled], shots=DEFAULT_SHOTS)
    result = job.result()

    bitarray = result[0].data.c
    counts = bitarray.get_counts()

    save_histogram_data(json_path, counts)
    return record_from_counts("real", backend.name, secret, counts)


def run_readout_noise_simulations(secret: str) -> list[RunRecord]:
    """Execute the circuit on an ideal simulator with injected readout error levels."""
    circuit = bernstein_vazirani_circuit(secret)
    records: list[RunRecord] = []
    counts_by_level: dict[str, dict[str, int]] = {}
    ideal_counts: dict[str, int] = {}

    # Ideal baseline for overlaying on the histogram (load or compute once).
    ideal_json = histogram_json_path("ideal", secret)
    cached_ideal = load_histogram_data(ideal_json)
    if cached_ideal is not None:
        ideal_counts = cached_ideal
    else:
        ideal_backend = AerSimulator(seed_simulator=DEFAULT_SEED)
        ideal_pm = generate_preset_pass_manager(
            optimization_level=3,
            backend=ideal_backend,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=DEFAULT_SEED,
        )
        ideal_compiled = ideal_pm.run(circuit)
        ideal_result = ideal_backend.run(ideal_compiled, shots=DEFAULT_SHOTS, seed_simulator=DEFAULT_SEED).result()
        ideal_counts = ideal_result.get_counts()
        save_histogram_data(ideal_json, ideal_counts)

    for level, probability in READOUT_NOISE_LEVELS.items():
        json_path = histogram_json_path("readout_noise", secret, tag=level)
        cached_counts = load_histogram_data(json_path)
        if cached_counts is not None:
            counts = cached_counts
        else:
            noise_model = build_readout_noise_model(probability)
            backend = AerSimulator(noise_model=noise_model, seed_simulator=DEFAULT_SEED)
            pass_manager = generate_preset_pass_manager(
                optimization_level=3,
                backend=backend,
                layout_method="sabre",
                routing_method="sabre",
                seed_transpiler=DEFAULT_SEED,
            )
            compiled = pass_manager.run(circuit)

            result = backend.run(compiled, shots=DEFAULT_SHOTS, seed_simulator=DEFAULT_SEED).result()
            counts = result.get_counts()
            save_histogram_data(json_path, counts)
        counts_by_level[level] = counts
        records.append(record_from_counts("readout_noise", f"aer_readout_{level}", secret, counts))

    plot_readout_sweep(secret, counts_by_level, ideal_counts)
    return records


def build_readout_noise_model(probability: float) -> NoiseModel:
    """Construct a noise model with symmetric readout error on all measured qubits."""
    matrix = [[1 - probability, probability], [probability, 1 - probability]]
    readout_error = ReadoutError(matrix)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def render_horizontal_histogram(
    counts_list: list[dict[int, int]],
    legends: list[str],
    colors: list[str] | None,
    title: str,
    save_path: Path,
) -> None:
    """Render a combined histogram similar to src/main.py (horizontal stacked bars)."""
    if not counts_list:
        return
    all_keys = sorted({key for counts in counts_list for key in counts})
    bit_width = max(1, max(all_keys).bit_length()) if all_keys else 1
    num_series = len(counts_list)
    height = 0.8 / max(num_series, 1)
    y_positions = list(range(len(all_keys)))

    height_dynamic = max(6.0, min(24.0, len(all_keys) * 0.35))
    fig, ax = plt.subplots(figsize=(8, height_dynamic), dpi=150)
    for idx, counts in enumerate(counts_list):
        values = [counts.get(key, 0) for key in all_keys]
        offsets = [y + (idx - (num_series - 1) / 2) * height for y in y_positions]
        kwargs = {}
        if colors and idx < len(colors):
            kwargs["color"] = colors[idx]
        ax.barh(offsets, values, height=height, label=legends[idx] if legends else None, **kwargs, zorder=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([format(key, f"0{bit_width}b") for key in all_keys])
    ax.set_xlabel("Counts")
    ax.set_ylabel("Bitstring")
    if legends and len(legends) > 1:
        ax.legend(title="Intensity")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.8, color="#999999", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    ax.invert_yaxis()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format="png", dpi=200)
    plt.close(fig)


def plot_readout_sweep(secret: str, counts_by_level: dict[str, dict[str, int]], ideal_counts: dict[str, int]) -> None:
    """Create a single figure concatenating histograms for all readout error levels plus ideal."""
    ordered_levels = list(READOUT_NOISE_LEVELS.keys())
    counts_list: list[dict[int, int]] = []
    legends: list[str] = []
    colors: list[str] = []
    if ideal_counts:
        counts_list.append({int(bitstring, 2): value for bitstring, value in ideal_counts.items()})
        legends.append("ideal")
        colors.append(READOUT_IDEAL_COLOR)
    for level in ordered_levels:
        if level in counts_by_level:
            numeric_counts = {int(bitstring, 2): value for bitstring, value in counts_by_level[level].items()}
            counts_list.append(numeric_counts)
            legends.append(level)
            colors.append(READOUT_COLORS.get(level, "#333333"))
    if not counts_list:
        return
    title = f"Readout error sweep - secret {secret}"
    save_path = RESULTS_ROOT / "readout_noise" / f"{secret}_readout_sweep.png"
    render_horizontal_histogram(counts_list, legends, colors, title, save_path)


def plot_backend_comparison(
    secret: str,
    backend_name: str,
    ideal_counts: dict[str, int],
    emulated_counts: dict[str, int],
    real_counts: dict[str, int] | None = None,
) -> None:
    """Plot ideal vs emulated (and optionally real) counts for a backend."""
    counts_list: list[dict[int, int]] = []
    legends: list[str] = []
    colors: list[str] = []

    if ideal_counts:
        counts_list.append({int(bitstring, 2): value for bitstring, value in ideal_counts.items()})
        legends.append("ideal")
        colors.append(IDEAL_OVERLAY_COLOR)

    if emulated_counts:
        counts_list.append({int(bitstring, 2): value for bitstring, value in emulated_counts.items()})
        legends.append(f"emulated-{backend_name}")
        colors.append(EMULATED_COLOR)

    if real_counts:
        counts_list.append({int(bitstring, 2): value for bitstring, value in real_counts.items()})
        legends.append(f"real-{backend_name}")
        colors.append(REAL_OVERLAY_COLOR)

    if not counts_list:
        return

    title = f"{backend_name} emulation vs ideal" if not real_counts else f"{backend_name} emulation vs ideal vs real"
    save_path = RESULTS_ROOT / "backend_comparison" / f"{secret}_{backend_name}.png"
    render_horizontal_histogram(counts_list, legends, colors, title, save_path)


def run_suite(
    secrets: Iterable[str],
    backends_list: list,
) -> list[RunRecord]:
    """Run all simulations and real executions for a sequence of secret strings."""
    records: list[RunRecord] = []
    ideal_counts_map: dict[str, dict[str, int]] = {}

    # Ideal baseline for metrics and overlays.
    for secret in secrets:
        record = run_ideal_simulation(secret)
        if record:
            records.append(record)
            ideal_counts_map[secret] = record.counts

    # Emulated backends, with overlays from ideal and optional real execution.
    for secret in secrets:
        ideal_counts = ideal_counts_map.get(secret, {})
        for backend in backends_list:
            noisy_record = run_noisy_simulation(secret, backend)
            real_record = None
            # try:
            #     real_record = run_real_execution(secret, backend)
            # except Exception as err:
            #     print(f"Real execution failed on {backend.name} for secret {secret}: {err}")
            if noisy_record:
                records.append(noisy_record)
            if real_record:
                records.append(real_record)
            if noisy_record:
                plot_backend_comparison(
                    secret,
                    backend.name,
                    ideal_counts=ideal_counts,
                    emulated_counts=noisy_record.counts,
                    real_counts=real_record.counts if real_record else None,
                )

    # Readout noise sweeps with ideal overlay.
    for secret in secrets:
        records.extend(run_readout_noise_simulations(secret))

    return records


def compute_macro_metrics(y_true: list[str], y_pred: list[str]) -> tuple[float, float, float, float]:
    """Compute accuracy, macro precision, macro recall, and macro F1-score."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0, 0.0, 0.0, 0.0

    labels = set(y_true) | set(y_pred)
    correct = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == pred)
    accuracy = correct / len(y_true)

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    f1_scores: list[float] = []

    for label in labels:
        tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == label and pred != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    macro_precision = sum(precision_scores) / len(labels)
    macro_recall = sum(recall_scores) / len(labels)
    macro_f1 = sum(f1_scores) / len(labels)
    return accuracy, macro_precision, macro_recall, macro_f1


def summarize_metrics(records: list[RunRecord]) -> list[dict[str, str | float | int]]:
    """Aggregate metrics per (category, backend) pair."""
    grouped: dict[tuple[str, str], dict[str, list[str]]] = {}
    for record in records:
        key = (record.category, record.backend)
        grouped.setdefault(key, {"y_true": [], "y_pred": []})
        grouped[key]["y_true"].append(record.secret)
        grouped[key]["y_pred"].append(record.prediction)

    summaries: list[dict[str, str | float | int]] = []
    for (category, backend), payload in grouped.items():
        accuracy, precision, recall, f1 = compute_macro_metrics(payload["y_true"], payload["y_pred"])
        summaries.append(
            {
                "category": category,
                "backend": backend,
                "samples": len(payload["y_true"]),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )
    return summaries


def save_predictions(records: list[RunRecord], path: Path = PREDICTIONS_CSV) -> None:
    """Persist individual predictions to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "backend",
        "secret",
        "expected",
        "predicted",
        "correct",
        "top_probability",
        "counts",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "category": record.category,
                    "backend": record.backend,
                    "secret": record.secret,
                    "expected": record.secret,
                    "predicted": record.prediction,
                    "correct": int(record.correct),
                    "top_probability": f"{record.top_probability:.4f}",
                    "counts": record.counts,
                }
            )
    print(f"Saved predictions to {path}")


def save_metrics(metrics: list[dict[str, str | float | int]], path: Path = METRICS_CSV) -> None:
    """Persist aggregated metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "backend", "samples", "accuracy", "precision", "recall", "f1_score"]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "category": item["category"],
                    "backend": item["backend"],
                    "samples": item["samples"],
                    "accuracy": f"{item['accuracy']:.4f}",
                    "precision": f"{item['precision']:.4f}",
                    "recall": f"{item['recall']:.4f}",
                    "f1_score": f"{item['f1_score']:.4f}",
                }
            )
    print(f"Saved metrics to {path}")


if __name__ == "__main__":
    service = build_runtime_service()
    backend_list = available_backends(service)
    all_records = run_suite(SECRET_STRINGS, backend_list)
    save_predictions(all_records)
    metric_rows = summarize_metrics(all_records)
    save_metrics(metric_rows)
