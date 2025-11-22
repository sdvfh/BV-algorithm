import os

from dotenv import load_dotenv
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeFez


def create_f(string_c, string_m):
    if len(string_m) > len(string_c):
        raise ValueError("String match is bigger than string to compare")
    return [1 if string_c[i : i + len(string_m)] == string_m else 0 for i in range(len(string_c) - len(string_m) + 1)]


load_dotenv()

if False:
    service = QiskitRuntimeService(channel=os.getenv("IBM_QUANTUM_SERVICE"))

    backend = FakeFez()
    backend.refresh(service)

    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        layout_method="sabre",
        routing_method="sabre",
    )

string_comparison = "Olá meu nome é Fernando."
string_match = "Fernando"

a = create_f(string_comparison, string_match)

print(a)
