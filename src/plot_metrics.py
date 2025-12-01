import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configurações
RESULTS_FILE = Path("Plot_results/bv_predictions.csv")
OUTPUT_DIR = Path("Plot_results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_hamming_weight(bitstring):
    # Ensure we always operate on a string to support numerically parsed columns
    # and gracefully handle missing values.
    if pd.isna(bitstring):
        return None
    return str(bitstring).count("1")


def parse_counts(raw_counts):
    """Parse a counts column that may be stored as a string in the CSV."""
    if pd.isna(raw_counts):
        return {}
    if isinstance(raw_counts, dict):
        return raw_counts
    if isinstance(raw_counts, str):
        try:
            parsed = ast.literal_eval(raw_counts)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def calculate_success_probability(secret, raw_counts):
    """Return probability of measuring the correct secret from the counts."""
    if pd.isna(secret):
        return None
    counts = parse_counts(raw_counts)
    if not counts:
        return None
    try:
        total_shots = sum(int(v) for v in counts.values())
        correct = int(counts.get(secret, 0))
    except Exception:
        return None
    return correct / total_shots if total_shots else None


def plot_hamming_trend():
    # 1. Carregar dados
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado em {RESULTS_FILE.resolve()}")

    df = pd.read_csv(
        RESULTS_FILE,
        dtype={
            "category": "string",
            "backend": "string",
            "secret": "string",
            "expected": "string",
            "predicted": "string",
            "counts": "string",
        },
    )

    # 2. Filtrar apenas dados Reais e Emulação (excluir ruídos sintéticos isolados por enquanto)
    # Ajuste os filtros conforme os nomes exatos no seu CSV ('noisy', 'real', etc)
    target_categories = ["real", "noisy"]
    df_filtered = df[df["category"].isin(target_categories)].copy()

    # 3. Calcular Peso de Hamming
    df_filtered["hamming_weight"] = df_filtered["secret"].apply(calculate_hamming_weight)

    # 4. Probabilidade de acerto baseada na contagem do bitstring correto
    df_filtered["success_probability"] = df_filtered.apply(
        lambda row: calculate_success_probability(row["secret"], row["counts"]), axis=1
    )
    df_filtered = df_filtered.dropna(subset=["hamming_weight", "success_probability"])

    # 5. Criar nome legível para a legenda
    # Ex: "ibm_torino (Real)" ou "ibm_torino (Emulação)"
    def get_label(row):
        mode = "real" if row["category"] == "real" else "emulado"
        return f"{row['backend']} ({mode})"

    df_filtered["Legenda"] = df_filtered.apply(get_label, axis=1)

    # Marcadores fixos por backend, cores fixas por categoria (real vs emulado)
    backend_markers = {"ibm_torino": "o", "ibm_fez": "s"}
    # Cores alinhadas ao bv.py: real vermelho, emulado azul
    category_colors = {"real": "#c43c39", "noisy": "#1f77b4"}

    legend_to_backend = df_filtered.set_index("Legenda")["backend"].to_dict()
    legend_to_category = df_filtered.set_index("Legenda")["category"].to_dict()
    palette = {label: category_colors.get(cat, "#555555") for label, cat in legend_to_category.items()}
    markers = {label: backend_markers.get(backend, "o") for label, backend in legend_to_backend.items()}

    # 6. Plotar
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Gráfico de linha com intervalo de confiança (se houver múltiplas execuções) ou pontos
    sns.lineplot(
        data=df_filtered,
        x="hamming_weight",
        y="success_probability",
        hue="Legenda",
        style="Legenda",
        palette=palette,
        markers=markers,
        dashes=False,
        linewidth=2.5,
        markersize=9,
    )

    max_weight = int(df_filtered["hamming_weight"].max()) if not df_filtered.empty else 0

    # plt.title('Impacto do Peso de Hamming na Fidelidade do Algoritmo', fontsize=14)
    plt.xlabel("Peso de Hamming (Número de portas CNOT)", fontsize=12)
    plt.ylabel("Probabilidade de sucesso (acerto do segredo)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(range(max_weight + 1))
    plt.legend(title="Modo")
    plt.tight_layout()

    save_path = OUTPUT_DIR / "hamming_weight_trend.png"
    plt.savefig(save_path, dpi=300)
    print(f"Gráfico salvo em: {save_path}")


if __name__ == "__main__":
    plot_hamming_trend()
