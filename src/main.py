import csv
import math
import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
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
RESULTS_PATH = os.getenv("RESULTS_CSV", "data/results_summary.csv")
REQUIRED_NOISE_BACKENDS = {"ibm_fez", "ibm_torino"}
HISTOGRAM_DIR = os.getenv("HISTOGRAM_DIR", "data/histograms")

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


RESULT_ROWS: list[dict[str, str]] = []


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


def record_result(
    experiment_id: int,
    sentence: str,
    pattern: str,
    backend_label: str,
    noise_kind: str,
    noise_level: str,
    expected_indices: list[int],
    summary: dict[str, object],
):
    RESULT_ROWS.append(
        {
            "experiment_id": str(experiment_id),
            "sentence": sentence,
            "pattern": pattern,
            "backend_label": backend_label,
            "noise_kind": noise_kind,
            "noise_level": noise_level,
            "expected_indices": ",".join(str(i) for i in expected_indices),
            "top_indices": ",".join(str(i) for i in summary["top_indices"]),
            "overlap": str(summary["overlap"]),
            "success_prob": f"{summary['success_prob']:.6f}",
        }
    )


def report(
    label: str,
    expected_indices: list[int],
    summary: dict[str, object],
    experiment_id: int,
    sentence: str,
    pattern: str,
    noise_kind: str,
    noise_level: str,
):
    print(f"\n[{label}] Expected: {sorted(expected_indices)}")
    print(
        f"Top indices: {summary['top_indices']} | Overlap: {summary['overlap']} "
        f"| Success prob: {summary['success_prob']:.4f}"
    )
    record_result(experiment_id, sentence, pattern, label, noise_kind, noise_level, expected_indices, summary)


def prettify_title(text: str) -> str:
    clean = text.replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() for word in clean.split())


def histogram_path(experiment_id: int, name: str) -> str:
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in name).strip("_")
    return os.path.join(HISTOGRAM_DIR, str(experiment_id), f"{safe}.pgf")


def maybe_plot(title: str, counts: dict[str, int], experiment_id: int, filename_prefix: str):
    if not PLOT_HIST:
        return
    numeric_counts = {int(bitstring, 2): value for bitstring, value in counts.items()}
    save_path = histogram_path(experiment_id, filename_prefix)
    render_horizontal_histogram([numeric_counts], [title], None, title, save_path)


def plot_noise_histogram(noise_kind: str, counts_by_level: dict[str, dict[str, int]], experiment_id: int):
    if not PLOT_HIST or not counts_by_level:
        return
    levels_ordered = list(NOISE_LEVELS.keys())
    data = []
    legends = []
    for level in levels_ordered:
        if level in counts_by_level:
            numeric_counts = {int(bitstring, 2): value for bitstring, value in counts_by_level[level].items()}
            data.append(numeric_counts)
            legends.append(level)
    colors = {
        "ultra-low": "#2ca02c",
        "low": "#1f77b4",
        "medium": "#ff7f0e",
        "high": "#d62728",
        "very-high": "#9467bd",
    }
    color_list = [colors.get(level) for level in legends]
    save_path = histogram_path(experiment_id, f"{noise_kind}_noise_sweep")
    render_horizontal_histogram(data, legends, color_list, f"{prettify_title(noise_kind)} noise sweep", save_path)


def render_horizontal_histogram(
    counts_list: list[dict[int, int]],
    legends: list[str],
    colors: list[str] | None,
    title: str,
    save_path: str | None = None,
):
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
    ax.set_title(prettify_title(title))
    ax.grid(axis="x", linestyle="--", linewidth=0.8, color="#999999", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    ax.invert_yaxis()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            fig.savefig(save_path, format="pgf")
        except Exception as err:
            fallback_path = os.path.splitext(save_path)[0] + ".png"
            print(f"PGF save failed ({err}); saving PNG fallback at {fallback_path}")
            fig.savefig(fallback_path, format="png", dpi=200)
    # plt.show()


def simulate_with_noise_kinds(qc, expected_indices, shots: int, experiment_id: int, sentence: str, pattern: str):
    for noise_kind in ["amplitude", "phase", "depolarizing", "combined"]:
        counts_by_level: dict[str, dict[str, int]] = {}
        for level in NOISE_LEVELS:
            label = f"simulator-{noise_kind}-{level}"
            noise_model = build_noise_model(noise_kind, level)
            backend = make_backend(noise_model)
            counts = run_with_backend(qc, backend, shots)
            counts_by_level[level] = counts
            summary = summarize_counts(counts, expected_indices, shots)
            report(label, expected_indices, summary, experiment_id, sentence, pattern, noise_kind, level)
        plot_noise_histogram(noise_kind, counts_by_level, experiment_id)


def simulate_with_backend_noise(
    qc, expected_indices, shots: int, service: QiskitRuntimeService, experiment_id: int, sentence: str, pattern: str
):
    backend_names = {backend.name for backend in service.backends() if backend.name not in {"ibm_marrakesh"}}
    backend_names.update(REQUIRED_NOISE_BACKENDS)
    for backend_name in backend_names:
        try:
            backend = service.backend(backend_name)
        except Exception:
            print(f"Backend '{backend_name}' unavailable; skipping noise emulation.")
            continue
        noise_model = NoiseModel.from_backend(backend)
        backend_noisy = make_backend(noise_model)
        counts = run_with_backend(qc, backend_noisy, shots)
        summary = summarize_counts(counts, expected_indices, shots)
        report(
            f"noisy-sim-{backend.name}",
            expected_indices,
            summary,
            experiment_id,
            sentence,
            pattern,
            "backend_noise",
            backend.name,
        )
        maybe_plot(f"noisy-sim-{backend.name}", counts, experiment_id, f"noisy-sim-{backend.name}")


def run_on_real_backends(
    qc, expected_indices, shots: int, service: QiskitRuntimeService, experiment_id: int, sentence: str, pattern: str
):
    for backend_name in BACKEND_TARGETS:
        backend = service.backend(backend_name)
        counts = run_with_backend(qc, backend, shots)
        summary = summarize_counts(counts, expected_indices, shots)
        report(
            f"real-{backend.name}",
            expected_indices,
            summary,
            experiment_id,
            sentence,
            pattern,
            "hardware",
            backend.name,
        )
        maybe_plot(f"real-{backend.name}", counts, experiment_id, f"real-{backend.name}")


def persist_results():
    if not RESULT_ROWS:
        print("\nNo results to save.")
        return
    os.makedirs(os.path.dirname(RESULTS_PATH) or ".", exist_ok=True)
    fieldnames = [
        "experiment_id",
        "sentence",
        "pattern",
        "backend_label",
        "noise_kind",
        "noise_level",
        "expected_indices",
        "top_indices",
        "overlap",
        "success_prob",
    ]
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(RESULT_ROWS)
    print(f"\nSaved summary table to {RESULTS_PATH} ({len(RESULT_ROWS)} rows).")


def process_experiment(experiment_id: int, sentence: str, pattern: str, service: QiskitRuntimeService | None):
    expected_results = create_results(sentence, pattern)
    expected_indices = results_to_indices(expected_results)
    if len(expected_indices) == 0:
        print(f"\nSkipping sentence without matches: '{sentence[:40]}...'")
        return

    qc = create_string_match(sentence, pattern)

    ideal_backend = make_backend()
    counts_ideal = run_with_backend(qc, ideal_backend, SHOTS)
    summary_ideal = summarize_counts(counts_ideal, expected_indices, SHOTS)
    report("ideal-simulator", expected_indices, summary_ideal, experiment_id, sentence, pattern, "ideal", "none")
    maybe_plot("ideal-simulator", counts_ideal, experiment_id, "ideal-simulator")

    simulate_with_noise_kinds(qc, expected_indices, SHOTS, experiment_id, sentence, pattern)

    if service:
        simulate_with_backend_noise(qc, expected_indices, SHOTS, service, experiment_id, sentence, pattern)
        if BACKEND_TARGETS:
            run_on_real_backends(qc, expected_indices, SHOTS, service, experiment_id, sentence, pattern)


def main():
    service = None
    try:
        service = QiskitRuntimeService(channel=os.getenv("QISKIT_IBM_CHANNEL"))
    except Exception:
        print("IBM runtime service not initialized; skipping hardware runs.")

    for i, (sentence, pattern) in enumerate(EXPERIMENTS[:MAX_EXPERIMENTS], start=1):
        print(f"\n=== Experiment {i}: pattern='{pattern}' ===")
        process_experiment(i, sentence, pattern, service)
    persist_results()


if __name__ == "__main__":
    main()
