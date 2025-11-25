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


string_comparison = "sergio encontrou uma nova estrategia para melhorar o algoritmo"
string_match = "melhorar o algoritmo"
results = create_results(string_comparison, string_match)

qc = create_string_match(string_comparison, string_match)
service = QiskitRuntimeService(channel=os.getenv("IBM_QUANTUM_SERVICE"))

backend = AerSimulator()
pass_manager = generate_preset_pass_manager(
    optimization_level=3,
    backend=backend,
    layout_method="sabre",
    routing_method="sabre",
)
qc_best = pass_manager.run(qc)
result = backend.run(qc, shots=2048).result()
counts = result.get_counts()
plot_histogram(counts)
plt.title(f"Histogram for backend {backend.name}")
plt.tight_layout()
plt.show()

for backend in service.backends():
    if backend.name == "ibm_marrakesh":
        continue
    print("Backend: ", backend.name)
    noise_model = NoiseModel.from_backend(backend)

    backend_noisy = AerSimulator(noise_model=noise_model)
    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend_noisy,
        layout_method="sabre",
        routing_method="sabre",
    )
    qc_best = pass_manager.run(qc)
    result = backend_noisy.run(qc_best, shots=2048).result()
    counts = result.get_counts()
    plot_histogram(counts)
    plt.title(f"Histogram for backend {backend.name}")
    plt.tight_layout()
    plt.show()
