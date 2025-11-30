"""Bernstein-Vazirani experiment runner with ideal, noisy, and real executions.

This module centralizes circuit construction, simulation execution, result plotting,
and output management for the Bernstein-Vazirani algorithm. It reuses shared helpers
for plotting and backend handling, documents each step, and skips runs when a plot
already exists at the target path.
"""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path

from dotenv import load_dotenv
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

DEFAULT_SHOTS = 1024
DEFAULT_SEED = 42
EXCLUDED_BACKENDS = {"ibm_marrakesh"}
RESULTS_ROOT = Path("Plot_results")
SECRET_STRINGS = [
    "10000",
    "01000",
    "00100",
    "00010",
    "00001",
    "00011",
    "00111",
    "01111",
    "11111",
    "11000",
    "11100",
    "10100",
    "10001",
    "10101",
    "01010",
    "01100",
    "11011",
    "10111",
    "01101",
    "10011",
    "01001",
    "00101",
    "01110",
    "11010",
    "10110",
]


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


def should_skip_plot(path: Path) -> bool:
    """Determine whether a plot should be skipped because the file already exists."""
    if path.exists():
        print(f"Skipping run; plot already exists at {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    return False


def plot_and_save(counts, title: str, output_path: Path) -> None:
    """Plot histogram counts with a standard title and save to disk."""
    plt.clf()
    plot_histogram(counts)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def run_ideal_simulation(secret: str) -> None:
    """Execute the Bernstein-Vazirani circuit on an ideal simulator."""
    output_path = build_plot_path("ideal", secret)
    if should_skip_plot(output_path):
        return

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

    plot_and_save(counts, "Histogram (Ideal Simulation)", output_path)


def run_noisy_simulation(secret: str, backend) -> None:
    """Execute the circuit on a noisy simulator derived from a specific backend."""
    output_path = build_plot_path("noisy", secret, backend.name)
    if should_skip_plot(output_path):
        return

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

    plot_and_save(counts, f"Histogram (Noisy Simulation) - {backend.name}", output_path)


def run_real_execution(secret: str, backend) -> None:
    """Execute the circuit on a real backend using the runtime Sampler."""
    output_path = build_plot_path("real", secret, backend.name)
    if should_skip_plot(output_path):
        return

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

    plot_and_save(counts, f"Histogram (Real Execution) - {backend.name}", output_path)


def run_suite(
    secrets: Iterable[str],
    backends_list: list,
) -> None:
    """Run all simulations and real executions for a sequence of secret strings."""
    for secret in secrets:
        run_ideal_simulation(secret)

    for secret in secrets:
        for backend in backends_list:
            run_noisy_simulation(secret, backend)

    for secret in secrets:
        for backend in backends_list:
            run_real_execution(secret, backend)


if __name__ == "__main__":
    service = build_runtime_service()
    backend_list = available_backends(service)
    run_suite(SECRET_STRINGS, backend_list)
