"""Bernstein-Vazirani experiment runner with metrics and plotting.

Executes a suite of secret strings across simulators, custom noise models, and
IBM hardware samplers, capturing predicted secrets, classification metrics, and
serialized histogram data suitable for analysis and reporting.
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
from qiskit_aer.noise.errors import (
    ReadoutError,
    amplitude_damping_error,
    depolarizing_error,
    phase_amplitude_damping_error,
    phase_damping_error,
    thermal_relaxation_error,
)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

DEFAULT_SHOTS = 1024
DEFAULT_SEED = 42
TARGET_BACKENDS = ("ibm_torino", "ibm_fez")
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
READOUT_LEVEL_TRANSLATIONS = {
    "ultra-low": "ultrabaixo",
    "low": "baixo",
    "medium": "medio",
    "high": "alto",
    "very-high": "muito alto",
}
LEVEL_MULTIPLIERS = {
    "ultra-low": 0.5,
    "low": 2.0,
    "medium": 10.0,
    "high": 100.0,
    "very-high": 1000.0,
}
# Baselines anchored on real hardware (fez/torino) so "low" matches typical errors.
BASE_READOUT_P = 0.02  # ~torino median
BASE_CX_P = 0.003
BASE_SX_P = 0.00035  # ~3.2e-4 sx median
BASE_DEP1_P = BASE_SX_P
BASE_DEP2_P = BASE_CX_P
BASE_T1 = 150e-6  # seconds (between fez 143us and torino 176us)
BASE_T2 = 115e-6  # seconds (between fez 99us and torino 135us)
GATE_TIME_SX = 35e-9  # representative single-qubit gate time
GATE_TIME_CX = 250e-9  # representative two-qubit gate time
CUSTOM_NOISE_KINDS = [
    "thermal_relaxation",
    "depolarizing",
    "amplitude_damping",
    "phase_damping",
    "phase_amplitude_damping",
    "cx_gate",
    "sx_gate",
    "h_gate",
    "phase_amplitude_h_gate",
]
SECRET_STRINGS = [
    "00000",
    "10000",
    "01000",
    "00100",
    "00010",
    "00001",
    "00011",
    "00111",
    "01111",
    "11111",
    "00101",
    "00110",
    "01001",
    "01010",
    "01011",
    "01100",
    "01101",
    "01110",
    "10001",
    "10010",
    "10011",
    "10100",
    "10110",
    "10111",
    "11000",
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
    channel = os.getenv("IBM_QUANTUM_SERVICE") or os.getenv("QISKIT_IBM_CHANNEL")
    if channel:
        return QiskitRuntimeService(channel=channel)
    return QiskitRuntimeService()


def available_backends(service: QiskitRuntimeService):
    """Return target backends (ibm_torino and ibm_fez) when available."""
    backend_map = {backend.name: backend for backend in service.backends()}
    missing = [name for name in TARGET_BACKENDS if name not in backend_map]
    if missing:
        print(f"Backends not available and will be skipped: {', '.join(missing)}")
    return [backend_map[name] for name in TARGET_BACKENDS if name in backend_map]


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
    plt.xlabel("Contagens")
    plt.ylabel("Bitstring")
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
        json_path = histogram_json_path("custom_noise", secret, tag=f"readout_{level}")
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
        records.append(record_from_counts("custom_noise_readout", f"readout_{level}", secret, counts))

    plot_readout_sweep(secret, counts_by_level, ideal_counts)
    return records


def build_readout_noise_model(probability: float) -> NoiseModel:
    """Construct a noise model with symmetric readout error on all measured qubits."""
    matrix = [[1 - probability, probability], [probability, 1 - probability]]
    readout_error = ReadoutError(matrix)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_readout_error(readout_error)
    return noise_model


def clamp_phase_amp_param(probability: float) -> float:
    """Ensure amplitude/phase damping parameters remain in the valid region (sum <= 1)."""
    return min(probability, 0.49)


def build_custom_noise_model(noise_kind: str, level: str) -> NoiseModel:
    """Construct custom noise models anchored to real backend error rates."""
    if level not in LEVEL_MULTIPLIERS:
        raise ValueError(f"Unknown noise level {level}")
    mult = LEVEL_MULTIPLIERS[level]
    noise_model = NoiseModel()

    if noise_kind == "readout":
        p = BASE_READOUT_P * mult
        return build_readout_noise_model(p)

    if noise_kind == "thermal_relaxation":
        t1 = max(BASE_T1 / mult, 1e-6)
        t2 = max(BASE_T2 / mult, 1e-6)
        err_1q = thermal_relaxation_error(t1, t2, GATE_TIME_SX)
        err_2q = thermal_relaxation_error(t1, t2, GATE_TIME_CX)
        add_1q_error(noise_model, err_1q)
        add_2q_error(noise_model, err_2q)
        return noise_model

    if noise_kind == "depolarizing":
        err_1q = depolarizing_error(min(BASE_DEP1_P * mult, 1.0), 1)
        err_2q = depolarizing_error(min(BASE_DEP2_P * mult, 1.0), 2)
        add_1q_error(noise_model, err_1q)
        add_2q_error(noise_model, err_2q)
        return noise_model

    if noise_kind == "amplitude_damping":
        err_1q = amplitude_damping_error(min(BASE_DEP1_P * mult, 1.0))
        err_2q = amplitude_damping_error(min(BASE_DEP2_P * mult, 1.0)).tensor(
            amplitude_damping_error(min(BASE_DEP2_P * mult, 1.0))
        )
        add_1q_error(noise_model, err_1q)
        add_2q_error(noise_model, err_2q)
        return noise_model

    if noise_kind == "phase_damping":
        err_1q = phase_damping_error(min(BASE_DEP1_P * mult, 1.0))
        err_2q = phase_damping_error(min(BASE_DEP2_P * mult, 1.0)).tensor(
            phase_damping_error(min(BASE_DEP2_P * mult, 1.0))
        )
        add_1q_error(noise_model, err_1q)
        add_2q_error(noise_model, err_2q)
        return noise_model

    if noise_kind == "phase_amplitude_damping":
        pa_param_1q = clamp_phase_amp_param(BASE_DEP1_P * mult)
        pa_param_2q = clamp_phase_amp_param(BASE_DEP2_P * mult)
        err_1q = phase_amplitude_damping_error(pa_param_1q, pa_param_1q)
        err_2q = phase_amplitude_damping_error(pa_param_2q, pa_param_2q).tensor(
            phase_amplitude_damping_error(pa_param_2q, pa_param_2q)
        )
        add_1q_error(noise_model, err_1q)
        add_2q_error(noise_model, err_2q)
        return noise_model

    if noise_kind == "cx_gate":
        err_2q = depolarizing_error(min(BASE_CX_P * mult, 1.0), 2)
        add_2q_error(noise_model, err_2q, gates=["cx"])
        return noise_model

    if noise_kind == "sx_gate":
        err_1q = depolarizing_error(min(BASE_SX_P * mult, 1.0), 1)
        add_1q_error(noise_model, err_1q, gates=["sx"])
        return noise_model

    if noise_kind == "h_gate":
        err_1q = depolarizing_error(min(BASE_SX_P * mult, 1.0), 1)
        add_1q_error(noise_model, err_1q, gates=["h"])
        return noise_model

    if noise_kind == "phase_amplitude_h_gate":
        pa_param_1q = clamp_phase_amp_param(BASE_DEP1_P * mult)
        pa_param_2q = clamp_phase_amp_param(BASE_DEP2_P * mult)
        err_1q = phase_amplitude_damping_error(pa_param_1q, pa_param_1q)
        err_2q = phase_amplitude_damping_error(pa_param_2q, pa_param_2q).tensor(
            phase_amplitude_damping_error(pa_param_2q, pa_param_2q)
        )
        base_gates_no_h = ["id", "x", "sx", "rz", "rx"]
        add_1q_error(noise_model, err_1q, gates=base_gates_no_h)
        add_2q_error(noise_model, err_2q)
        h_gate_error = depolarizing_error(min(BASE_SX_P * mult, 1.0), 1)
        combined_h_error = err_1q.compose(h_gate_error)
        add_1q_error(noise_model, combined_h_error, gates=["h"])
        return noise_model

    raise ValueError(f"Unknown noise kind {noise_kind}")


def add_1q_error(noise_model: NoiseModel, error, gates: list[str] | None = None) -> None:
    gates = gates or ["id", "x", "sx", "rz", "rx", "h"]
    for gate in gates:
        try:
            noise_model.add_all_qubit_quantum_error(error, gate)
        except Exception:
            continue


def add_2q_error(noise_model: NoiseModel, error, gates: list[str] | None = None) -> None:
    gates = gates or ["cx", "cz", "rzz"]
    for gate in gates:
        try:
            noise_model.add_all_qubit_quantum_error(error, gate)
        except Exception:
            continue


def render_horizontal_histogram(
    counts_list: list[dict[int, int]],
    legends: list[str],
    colors: list[str] | None,
    legend_title: str | None,
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
    ax.set_xlabel("Contagens")
    ax.set_ylabel("Cadeia de bits")
    if legends and len(legends) > 1:
        ax.legend(title=legend_title)
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
            legends.append(READOUT_LEVEL_TRANSLATIONS.get(level, level))
            colors.append(READOUT_COLORS.get(level, "#333333"))
    if not counts_list:
        return
    save_path = RESULTS_ROOT / "custom_noise" / "readout" / f"{secret}_readout_sweep.png"
    render_horizontal_histogram(counts_list, legends, colors, "Intensidade", save_path)


def plot_custom_noise_sweep(
    secret: str, noise_kind: str, counts_by_level: dict[str, dict[str, int]], ideal_counts: dict[str, int]
) -> None:
    """Create histogram combining ideal and multiple noise levels for a custom noise kind."""
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
            legends.append(READOUT_LEVEL_TRANSLATIONS.get(level, level))
            colors.append(READOUT_COLORS.get(level, "#333333"))
    if not counts_list:
        return
    save_path = RESULTS_ROOT / "custom_noise" / noise_kind / f"{secret}_{noise_kind}_sweep.png"
    render_horizontal_histogram(counts_list, legends, colors, "Intensidade", save_path)


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
        legends.append(backend_name)
        colors.append(EMULATED_COLOR)

    if real_counts:
        counts_list.append({int(bitstring, 2): value for bitstring, value in real_counts.items()})
        legends.append(f"real-{backend_name}")
        colors.append(REAL_OVERLAY_COLOR)

    if not counts_list:
        return

    save_path = RESULTS_ROOT / "backend_comparison" / f"{secret}_{backend_name}.png"
    render_horizontal_histogram(counts_list, legends, colors, "Emulação", save_path)


def run_custom_noise_simulations(secret: str, ideal_counts: dict[str, int]) -> list[RunRecord]:
    """Execute the circuit with multiple custom noise kinds and levels."""
    circuit = bernstein_vazirani_circuit(secret)
    records: list[RunRecord] = []
    ordered_levels = list(READOUT_NOISE_LEVELS.keys())

    for noise_kind in CUSTOM_NOISE_KINDS:
        counts_by_level: dict[str, dict[str, int]] = {}
        for level in ordered_levels:
            json_path = histogram_json_path("custom_noise", secret, tag=f"{noise_kind}_{level}")
            cached_counts = load_histogram_data(json_path)
            if cached_counts is not None:
                counts = cached_counts
            else:
                noise_model = build_custom_noise_model(noise_kind, level)
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
            records.append(record_from_counts(f"custom_noise_{noise_kind}", f"{noise_kind}_{level}", secret, counts))

        plot_custom_noise_sweep(secret, noise_kind, counts_by_level, ideal_counts)

    return records


def run_suite(
    secrets: Iterable[str],
    backends_list: list,
) -> list[RunRecord]:
    """Run simulations first (ideal, emulated, noise sweeps) then real executions."""
    records: list[RunRecord] = []
    ideal_counts_map: dict[str, dict[str, int]] = {}
    emulated_counts_map: dict[tuple[str, str], dict[str, int]] = {}

    # Ideal baseline for metrics and overlays.
    for secret in secrets:
        record = run_ideal_simulation(secret)
        if record:
            records.append(record)
            ideal_counts_map[secret] = record.counts

    # Emulated backends with overlays from ideal; no real execution yet.
    for secret in secrets:
        ideal_counts = ideal_counts_map.get(secret, {})
        for backend in backends_list:
            noisy_record = run_noisy_simulation(secret, backend)
            if noisy_record:
                records.append(noisy_record)
                emulated_counts_map[(secret, backend.name)] = noisy_record.counts
                plot_backend_comparison(
                    secret,
                    backend.name,
                    ideal_counts=ideal_counts,
                    emulated_counts=noisy_record.counts,
                    real_counts=None,
                )

    # Readout noise sweeps (now under custom_noise/readout) with ideal overlay.
    for secret in secrets:
        records.extend(run_readout_noise_simulations(secret))

    # Custom noise sweeps (depolarizing, amplitude/phase damping, thermal, gate-specific).
    for secret in secrets:
        ideal_counts = ideal_counts_map.get(secret, {})
        records.extend(run_custom_noise_simulations(secret, ideal_counts))

    # Real executions only after all simulations and plots are complete.
    for secret in secrets:
        ideal_counts = ideal_counts_map.get(secret, {})
        for backend in backends_list:
            real_record = None
            try:
                real_record = run_real_execution(secret, backend)
            except Exception as err:
                print(f"Real execution failed on {backend.name} for secret {secret}: {err}")
            if real_record:
                records.append(real_record)
                emulated_counts = emulated_counts_map.get((secret, backend.name), {})
                plot_backend_comparison(
                    secret,
                    backend.name,
                    ideal_counts=ideal_counts,
                    emulated_counts=emulated_counts,
                    real_counts=real_record.counts,
                )

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
        noise_kind, noise_level = parse_noise_info(category, backend)
        summaries.append(
            {
                "category": category,
                "backend": backend,
                "samples": len(payload["y_true"]),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "noise_kind": noise_kind or "",
                "noise_level": noise_level or "",
            }
        )
    return summaries


def save_predictions(records: list[RunRecord], path: Path = PREDICTIONS_CSV) -> None:
    """Persist individual predictions to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "category",
        "backend",
        "noise_kind",
        "noise_level",
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
            noise_kind, noise_level = parse_noise_info(record.category, record.backend)
            writer.writerow(
                {
                    "category": record.category,
                    "backend": record.backend,
                    "noise_kind": noise_kind or "",
                    "noise_level": noise_level or "",
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
    fieldnames = [
        "category",
        "backend",
        "noise_kind",
        "noise_level",
        "samples",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "category": item["category"],
                    "backend": item["backend"],
                    "noise_kind": item.get("noise_kind", ""),
                    "noise_level": item.get("noise_level", ""),
                    "samples": item["samples"],
                    "accuracy": f"{item['accuracy']:.4f}",
                    "precision": f"{item['precision']:.4f}",
                    "recall": f"{item['recall']:.4f}",
                    "f1_score": f"{item['f1_score']:.4f}",
                }
            )
    print(f"Saved metrics to {path}")


def parse_noise_info(category: str, backend: str) -> tuple[str | None, str | None]:
    """Derive noise kind and intensity level from category/backend naming."""
    if category.startswith("custom_noise_readout") and "_" in backend:
        _, level = backend.split("_", 1)
        return "readout", level
    if category.startswith("custom_noise") and "_" in backend:
        kind, level = backend.split("_", 1)
        return kind, level
    if category == "noisy":
        # Backend name is the identifier; no discrete level provided.
        return "backend_noise", ""
    return None, None


if __name__ == "__main__":
    service = build_runtime_service()
    backend_list = available_backends(service)
    all_records = run_suite(SECRET_STRINGS, backend_list)
    save_predictions(all_records)
    metric_rows = summarize_metrics(all_records)
    save_metrics(metric_rows)
