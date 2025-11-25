import math
import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import amplitude_damping_error, depolarizing_error, phase_damping_error
from qiskit_ibm_runtime import QiskitRuntimeService

load_dotenv()

SHOTS = int(os.getenv("SHOTS", "2048"))
SIM_SEED = int(os.getenv("SIM_SEED", "1337"))
PLOT_HIST = bool(int(os.getenv("PLOT_HIST", "0")))
MAX_EXPERIMENTS = int(os.getenv("MAX_EXPERIMENTS", "25"))
BACKEND_TARGETS = [name.strip() for name in os.getenv("IBM_TARGET_BACKENDS", "").split(",") if name.strip()]

NOISE_LEVELS = {
    "ultra-low": 1e-4,
    "low": 1e-3,
    "medium": 5e-3,
    "high": 1e-2,
    "very-high": 5e-2,
}

EXPERIMENTS: list[tuple[str, str]] = [
    ("ola meu nome e sergio", "me"),
    ("fernando fez feitos incriveis", "fe"),
    ("bolorosos bolos de bolas de bacia", "bo"),
    (
        "meu nome completo e sergio de vasconcelos filho mas o nome completo do professor e "
        "fernando maciano de paula neto",
        "no",
    ),
    ("humberto costa cordeiro tavora e sergio de vasconcelos filho fazem a dupla de algoritmos 1", "co"),
    ("eu estava passeando em um dia muito ensolarado e veio a chuva e me levou", "m"),
    ("sergio resolveu estudar algoritmos enquanto fazia um cafe bem forte", "ori"),
    ("fernando caminhava pelo parque quando encontrou um cachorro perdido", "encontrou"),
    ("humberto revisou o codigo antigo mas nao conseguiu encontrar o erro", "humberto"),
    ("a luz do fim da tarde iluminava as montanhas no horizonte", "horizonte"),
    ("eu encontrei uma caixa misteriosa no sotao e decidi abrir", "misteriosa"),
    ("o vento frio daquela manha balancava as arvores da rua inteira", "rua"),
    ("as paginas do caderno velho estavam cheias de anotacoes importantes", "importantes"),
    ("sergio e humberto discutiam sobre quem cozinhava melhor", "melhor"),
    ("o laboratorio estava silencioso enquanto todos analisavam os dados", "dados"),
    ("eu estava sentado observando o ceu quando uma ideia surgiu de repente", "observando"),
    ("os estudantes passaram horas tentando resolver um desafio dificil", "horas"),
    ("um livro raro foi encontrado na biblioteca por acaso", "biblioteca"),
    ("a chuva fina caiu sobre a cidade deixando tudo brilhando", "chuva"),
    ("o gato preto pulou sobre a mesa e derrubou varios papeis", "mesa"),
    ("eu vi uma estrela cadente enquanto caminhava pela praia deserta", "estrela cadente"),
    ("as conversas no corredor eram sobre projetos e experimentos", "projetos e experimentos"),
    ("sergio encontrou uma nova estrategia para melhorar o algoritmo", "melhorar o algoritmo"),
    ("uma tempestade forte surgiu de repente interrompendo o passeio", "de repente interrompendo o passeio"),
    ("o som do rio corria suavemente enquanto caminhavamos pela trilha", "rio corria suavemente"),
]


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


def add_noise_to_model(noise_model, single_error, two_qubit_error=None):
    for gate in ["id", "x", "sx", "rz", "h"]:
        try:
            noise_model.add_all_qubit_quantum_error(single_error, gate)
        except Exception:
            continue
    if two_qubit_error:
        for gate in ["cx"]:
            try:
                noise_model.add_all_qubit_quantum_error(two_qubit_error, gate)
            except Exception:
                continue


def build_noise_model(noise_kind: str, level: str) -> NoiseModel:
    if level not in NOISE_LEVELS:
        raise ValueError(f"Unknown noise level {level}")
    p = NOISE_LEVELS[level]
    noise_model = NoiseModel()

    if noise_kind == "amplitude":
        single_error = amplitude_damping_error(p)
        two_qubit_error = single_error.tensor(single_error)
    elif noise_kind == "phase":
        single_error = phase_damping_error(p)
        two_qubit_error = single_error.tensor(single_error)
    elif noise_kind == "depolarizing":
        single_error = depolarizing_error(p, 1)
        two_qubit_error = depolarizing_error(p, 2)
    elif noise_kind == "combined":
        single_error = amplitude_damping_error(p).compose(phase_damping_error(p)).compose(depolarizing_error(p, 1))
        two_qubit_error = amplitude_damping_error(p).tensor(amplitude_damping_error(p))
        two_qubit_error = two_qubit_error.compose(phase_damping_error(p).tensor(phase_damping_error(p)))
        two_qubit_error = two_qubit_error.compose(depolarizing_error(p, 2))
    else:
        raise ValueError(f"Unknown noise kind {noise_kind}")

    add_noise_to_model(noise_model, single_error, two_qubit_error)
    return noise_model


def make_backend(noise_model: NoiseModel | None = None) -> AerSimulator:
    return AerSimulator(noise_model=noise_model, seed_simulator=SIM_SEED)


def run_with_backend(qc, backend, shots: int):
    pass_manager = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=SIM_SEED,
    )
    optimized = pass_manager.run(qc)
    run_kwargs = {"shots": shots}
    if isinstance(backend, AerSimulator):
        run_kwargs["seed_simulator"] = SIM_SEED
    result = backend.run(optimized, **run_kwargs).result()
    return result.get_counts()


def extract_top_indices(counts: dict[str, int], top_k: int) -> list[int]:
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    indices: list[int] = []
    for bitstring, _ in ordered:
        index = int(bitstring, 2)
        if index not in indices:
            indices.append(index)
        if len(indices) >= top_k:
            break
    return sorted(indices)


def counts_to_probabilities(counts: dict[str, int], shots: int) -> dict[int, float]:
    probs: dict[int, float] = {}
    for bitstring, count in counts.items():
        probs[int(bitstring, 2)] = count / shots
    return probs


def summarize_counts(counts: dict[str, int], expected_indices: list[int], shots: int) -> dict[str, object]:
    top_k = max(len(expected_indices), 1)
    top_indices = extract_top_indices(counts, top_k)
    probs = counts_to_probabilities(counts, shots)
    success_prob = sum(probs.get(idx, 0.0) for idx in expected_indices)
    overlap = len(set(expected_indices).intersection(top_indices))
    return {
        "top_indices": top_indices,
        "success_prob": success_prob,
        "overlap": overlap,
    }


def report(label: str, expected_indices: list[int], summary: dict[str, object]):
    print(f"\n[{label}] Expected: {sorted(expected_indices)}")
    print(
        f"Top indices: {summary['top_indices']} | Overlap: {summary['overlap']} "
        f"| Success prob: {summary['success_prob']:.4f}"
    )


def maybe_plot(title: str, counts: dict[str, int]):
    if not PLOT_HIST:
        return
    plot_histogram(counts)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def simulate_with_noise_kinds(qc, expected_indices, shots: int):
    for noise_kind in ["amplitude", "phase", "depolarizing", "combined"]:
        for level in NOISE_LEVELS:
            label = f"simulator-{noise_kind}-{level}"
            noise_model = build_noise_model(noise_kind, level)
            backend = make_backend(noise_model)
            counts = run_with_backend(qc, backend, shots)
            summary = summarize_counts(counts, expected_indices, shots)
            report(label, expected_indices, summary)
            maybe_plot(f"{label}", counts)


def simulate_with_backend_noise(qc, expected_indices, shots: int, service: QiskitRuntimeService):
    for backend in service.backends():
        if backend.name in {"ibm_marrakesh"}:  # skip if unavailable
            continue
        noise_model = NoiseModel.from_backend(backend)
        backend_noisy = make_backend(noise_model)
        counts = run_with_backend(qc, backend_noisy, shots)
        summary = summarize_counts(counts, expected_indices, shots)
        report(f"noisy-sim-{backend.name}", expected_indices, summary)
        maybe_plot(f"noisy-sim-{backend.name}", counts)


def run_on_real_backends(qc, expected_indices, shots: int, service: QiskitRuntimeService):
    for backend_name in BACKEND_TARGETS:
        backend = service.backend(backend_name)
        counts = run_with_backend(qc, backend, shots)
        summary = summarize_counts(counts, expected_indices, shots)
        report(f"real-{backend.name}", expected_indices, summary)
        maybe_plot(f"real-{backend.name}", counts)


def process_experiment(sentence: str, pattern: str, service: QiskitRuntimeService | None):
    expected_results = create_results(sentence, pattern)
    expected_indices = results_to_indices(expected_results)
    if len(expected_indices) == 0:
        print(f"\nSkipping sentence without matches: '{sentence[:40]}...'")
        return

    qc = create_string_match(sentence, pattern)

    ideal_backend = make_backend()
    counts_ideal = run_with_backend(qc, ideal_backend, SHOTS)
    summary_ideal = summarize_counts(counts_ideal, expected_indices, SHOTS)
    report("ideal-simulator", expected_indices, summary_ideal)
    maybe_plot("ideal-simulator", counts_ideal)

    simulate_with_noise_kinds(qc, expected_indices, SHOTS)

    if service:
        simulate_with_backend_noise(qc, expected_indices, SHOTS, service)
        if BACKEND_TARGETS:
            run_on_real_backends(qc, expected_indices, SHOTS, service)


def main():
    service = None
    try:
        service = QiskitRuntimeService(channel=os.getenv("QISKIT_IBM_CHANNEL"))
    except Exception:
        print("IBM runtime service not initialized; skipping hardware runs.")

    for i, (sentence, pattern) in enumerate(EXPERIMENTS[:MAX_EXPERIMENTS], start=1):
        print(f"\n=== Experiment {i}: pattern='{pattern}' ===")
        process_experiment(sentence, pattern, service)


if __name__ == "__main__":
    main()
