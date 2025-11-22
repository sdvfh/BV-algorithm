import math

from dotenv import load_dotenv
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator


def create_string_match(string_c, string_m):
    if len(string_m) > len(string_c):
        raise ValueError("String match is bigger than string to compare")
    results = [
        1 if string_c[i : i + len(string_m)] == string_m else 0 for i in range(len(string_c) - len(string_m) + 1)
    ]

    n_qubits_in = math.ceil(math.log2(len(results)))
    M = sum(results)
    N = 2**n_qubits_in

    if M == 0:
        raise ValueError("Empty oracle")

    theta = 2 * math.asin(math.sqrt(M / N))
    n_rep_without_round = (math.pi / theta - 1) / 2
    n_rep = round(n_rep_without_round)
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


load_dotenv()

string_comparison = "Banana!"
string_match = "Ba"

qc = create_string_match(string_comparison, string_match)
# service = QiskitRuntimeService(channel=os.getenv("IBM_QUANTUM_SERVICE"))

# backend = FakeFez()
backend = AerSimulator()
# backend.refresh(service)

pass_manager = generate_preset_pass_manager(
    optimization_level=3,
    backend=backend,
    layout_method="sabre",
    routing_method="sabre",
)
qc_best = pass_manager.run(qc)
result = backend.run(qc_best, shots=50).result()
counts = result.get_counts()
plot_histogram(counts)
plt.show()
