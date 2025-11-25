import math
import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()


def create_results(string_c, string_m):
    return [1 if string_c[i : i + len(string_m)] == string_m else 0 for i in range(len(string_c) - len(string_m) + 1)]


def results_to_indices(results):
    return [i for i, value in enumerate(results) if value]


def create_string_match(string_c, string_m):
    if len(string_m) > len(string_c):
        raise ValueError("String match is bigger than string to compare")
    results = create_results(string_c, string_m)

    n_qubits_in = math.ceil(math.log2(len(results)))
    if n_qubits_in < 4:
        raise ValueError("Experiment with less than 5 qubits")
    print(f"Experiment with {n_qubits_in + 1} qubits")
    M = sum(results)
    N = 2**n_qubits_in

    if M == 0:
        raise ValueError("Empty oracle")

    theta = 2 * math.asin(math.sqrt(M / N))
    n_rep_without_round = (math.pi / theta - 1) / 2
    n_rep = round(n_rep_without_round)
    print("Exact number of repetitions: ", round(n_rep_without_round, 4))
    print("Number of repetitions rounded: ", n_rep)
    reg_in = QuantumRegister(n_qubits_in)
    c_reg_in = ClassicalRegister(n_qubits_in)
    reg_out = QuantumRegister(1)

    qc = QuantumCircuit(reg_in, reg_out, c_reg_in)
    qc.h(reg_in)

    qc.x(reg_out)
    qc.h(reg_out)
    for _ in range(n_rep):
        create_oracle(qc, reg_in, reg_out, results)
        create_diffusor(qc, reg_in)

    qc.measure(reg_in, c_reg_in)

    return qc


def create_oracle(qc, reg_in, reg_out, results):
    for i, result in enumerate(results):
        if result:
            qc.mcx(reg_in, reg_out, ctrl_state=i)


def create_diffusor(qc, reg_in):
    qc.h(reg_in)
    qc.x(reg_in)

    # MCZ
    qc.h(reg_in[-1])
    qc.mcx(reg_in[:-1], reg_in[-1])
    qc.h(reg_in[-1])

    qc.x(reg_in)
    qc.h(reg_in)


def run_with_backend(qc, backend, shots=2048):
    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        layout_method="sabre",
        routing_method="sabre",
    )
    optimized = pass_manager.run(qc)
    result = backend.run(optimized, shots=shots).result()
    return result.get_counts()


def extract_top_indices(counts, top_k):
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    indices = []
    for bitstring, _ in ordered:
        index = int(bitstring, 2)
        if index not in indices:
            indices.append(index)
        if len(indices) >= top_k:
            break
    return sorted(indices)


def compare_results(counts, expected_indices, backend_name, shots):
    k = len(expected_indices)
    top_indices = extract_top_indices(counts, k)
    print(f"\n[Backend: {backend_name}] Shots={shots}")
    print(f"Expected indices ({k}): {expected_indices}")
    print(f"Measured top-{k} indices: {top_indices}")


def main():
    string_comparison = "sergio encontrou uma nova estrategia para melhorar o algoritmo"
    string_match = "melhorar o algoritmo"

    expected_results = create_results(string_comparison, string_match)
    expected_indices = results_to_indices(expected_results)
    print(f"Total matches (1s) in expected vector: {len(expected_indices)}")

    qc = create_string_match(string_comparison, string_match)
    shots = 2048

    # Ideal simulation (no noise)
    ideal_backend = AerSimulator()
    ideal_counts = run_with_backend(qc, ideal_backend, shots)
    plot_histogram(ideal_counts)
    plt.title(f"Histogram for backend {ideal_backend.name}")
    plt.tight_layout()
    plt.show()
    compare_results(ideal_counts, expected_indices, ideal_backend.name, shots)

    # Noisy simulations based on real backends
    service = QiskitRuntimeService(channel=os.getenv("IBM_QUANTUM_SERVICE"))
    for backend in service.backends():
        if backend.name == "ibm_marrakesh":
            continue
        noise_model = NoiseModel.from_backend(backend)
        noisy_backend = AerSimulator(noise_model=noise_model)
        noisy_counts = run_with_backend(qc, noisy_backend, shots)
        plot_histogram(noisy_counts)
        plt.title(f"Histogram for backend {backend.name}")
        plt.tight_layout()
        plt.show()
        compare_results(noisy_counts, expected_indices, backend.name, shots)


if __name__ == "__main__":
    main()
