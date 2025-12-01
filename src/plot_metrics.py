import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def bootstrap_slope_confidence(x_vals: np.ndarray, y_vals: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    """Bootstrap simple linear slope confidence interval."""
    rng = np.random.default_rng(42)
    slopes: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x_vals), size=len(x_vals))
        x_sample = x_vals[idx]
        y_sample = y_vals[idx]
        if np.std(x_sample) == 0:
            continue
        slope, _ = np.polyfit(x_sample, y_sample, 1)
        slopes.append(slope)
    if not slopes:
        return float("nan"), float("nan")
    lower, upper = np.percentile(slopes, [2.5, 97.5])
    return float(lower), float(upper)


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

    # 5. Criar nome legível e ambiente
    def get_label(row):
        mode = "Real" if row["category"] == "real" else "Emulado"
        return f"{row['backend']} ({mode})"

    df_filtered["Legenda"] = df_filtered.apply(get_label, axis=1)
    df_filtered["Ambiente"] = (
        df_filtered["category"].map({"real": "Real", "noisy": "Emulação"}).fillna(df_filtered["category"])
    )

    # 6. Plotar separado por backend
    sns.set_style("whitegrid")
    for backend_name, group in df_filtered.groupby("backend"):
        if group.empty:
            continue

        plt.figure(figsize=(10, 6))
        # Gráfico de dispersão (scatter) por processador, usando marcadores ocos
        category_colors = {"Real": "#c43c39", "Emulação": "#1f77b4"}
        for ambiente, subset in group.groupby("Ambiente"):
            if subset.empty:
                continue
            color = category_colors.get(ambiente, "#555555")
            plt.scatter(
                subset["hamming_weight"],
                subset["success_probability"],
                label=ambiente,
                facecolors="none",
                edgecolors=color,
                s=120,
                linewidths=1.4,
                marker="o",
            )

        # Linha de tendência para pontos reais
        real_points = group[group["Ambiente"] == "Real"]
        if len(real_points) >= 3:
            x_vals = real_points["hamming_weight"].to_numpy(dtype=float)
            y_vals = real_points["success_probability"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_line = slope * x_line + intercept
            plt.plot(
                x_line,
                y_line,
                color="#b22222",
                linestyle="--",
                linewidth=1.6,
                label="Tendência (Real)",
            )
            corr = np.corrcoef(x_vals, y_vals)[0, 1]
            r2 = corr * corr if not np.isnan(corr) else np.nan
            ci_low, ci_high = bootstrap_slope_confidence(x_vals, y_vals)
            significance = " (significativo)" if ci_low > 0 or ci_high < 0 else ""
            print(
                f"{backend_name}: tendência Real -> slope={slope:.4f}, R^2={r2:.4f}, "
                f"IC95% slope=[{ci_low:.4f}, {ci_high:.4f}]{significance}"
            )

        max_weight = int(group["hamming_weight"].max()) if not group.empty else 0

        plt.xlabel("Peso de Hamming (número de portas CNOT)", fontsize=12)
        plt.ylabel("Probabilidade de sucesso (acerto do segredo)", fontsize=12)
        plt.ylim(0, 1.05)
        plt.xticks(range(max_weight + 1))
        plt.legend(title="Modo", loc="lower left")
        plt.tight_layout()

        save_path = OUTPUT_DIR / f"hamming_weight_trend_{backend_name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico salvo em: {save_path}")


if __name__ == "__main__":
    plot_hamming_trend()
