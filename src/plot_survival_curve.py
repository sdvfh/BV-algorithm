"""
Gera um gráfico de "curva de sobrevivência" agregando variações de ruído customizado.

Eixo X: níveis de ruído (ultra-low → very-high)
Eixo Y: probabilidade de sucesso média (ou, se disponível, a do pior caso 11111)
Linhas: uma por tipo de ruído de interesse (ex.: CX, H, Damping).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_FILE = Path("Plot_results/bv_predictions.csv")
OUTPUT_DIR = Path("Plot_results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ordem fixa dos níveis de ruído (espelha READOUT_NOISE_LEVELS em bv.py) + ideal
NOISE_LEVEL_ORDER = ["ideal", "ultra-low", "low", "medium", "high", "very-high"]
NOISE_LEVEL_TRANSLATIONS = {
    "ideal": "ideal",
    "ultra-low": "muito baixo",
    "low": "baixo",
    "medium": "medio",
    "high": "alto",
    "very-high": "muito alto",
}

# Quais tipos de ruído traçar e como nomeá-los na legenda
NOISE_KINDS_OF_INTEREST: dict[str, str] = {
    "cx_gate": "Porta CX",
    "h_gate": "Porta H",
    "sx_gate": "Porta SX",
    "amplitude_damping": "Perda de amplitude",
    "phase_damping": "Perda de fase",
    "readout": "Leitura",
}

# Se presente, usamos este segredo como "pior caso"; caso contrário, usamos a média
FOCUS_SECRET = "11111"


def parse_counts(raw_counts) -> dict[str, int]:
    """Converte a coluna counts (string) em dict."""
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


def success_probability(secret: str, raw_counts) -> float | None:
    """Probabilidade de medir o segredo correto."""
    if pd.isna(secret):
        return None
    counts = parse_counts(raw_counts)
    if not counts:
        return None
    try:
        total = sum(int(v) for v in counts.values())
        correct = int(counts.get(secret, 0))
    except Exception:
        return None
    return correct / total if total else None


def extract_noise_info(category: str, backend: str) -> tuple[str | None, str | None]:
    """
    Derive noise_kind e noise_level a partir das convenções do bv.py:
      category: custom_noise_<kind>  | custom_noise_readout
      backend:  <kind>_<level>       | readout_<level>
    """
    if not isinstance(category, str) or not isinstance(backend, str):
        return None, None
    if not category.startswith("custom_noise") or "_" not in backend:
        return None, None
    kind, level = backend.rsplit("_", 1)
    return kind, level


def prepare_noise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai colunas de ruído e filtra apenas níveis/kinds que queremos plotar."""
    df = df.copy()
    df["noise_kind"], df["noise_level"] = zip(
        *df.apply(lambda row: extract_noise_info(row["category"], row["backend"]), axis=1), strict=True
    )
    df = df.dropna(subset=["noise_kind", "noise_level"])
    df = df[df["noise_kind"].isin(NOISE_KINDS_OF_INTEREST.keys())]
    df = df[df["noise_level"].isin(NOISE_LEVEL_ORDER)]
    df["noise_level"] = pd.Categorical(df["noise_level"], categories=NOISE_LEVEL_ORDER, ordered=True)
    return df


def pick_value_for_group(group: pd.DataFrame) -> float:
    """
    Retorna a probabilidade de sucesso para o grupo:
      - Se existir a linha do segredo FOCUS_SECRET, usa essa.
      - Caso contrário, retorna a média do grupo.
    """
    focus_rows = group[group["secret"] == FOCUS_SECRET]
    if not focus_rows.empty:
        return float(focus_rows["success_probability"].mean())
    return float(group["success_probability"].mean())


def mean_confidence_interval(
    series: pd.Series, z_value: float = 1.96
) -> tuple[float | None, float | None, float | None]:
    """Retorna média e intervalo de confiança 95% (normal aproximado)."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return None, None, None
    mean_val = float(values.mean())
    if len(values) == 1:
        return mean_val, mean_val, mean_val
    std_dev = float(values.std(ddof=1))
    margin = z_value * std_dev / math.sqrt(len(values))
    lower = max(0.0, mean_val - margin)
    upper = min(1.0, mean_val + margin)
    return mean_val, lower, upper


def build_survival_dataframe(df: pd.DataFrame, baseline: float | None) -> pd.DataFrame:
    """Cria um dataframe agregado pronto para plotar (inclui ponto ideal se existir)."""
    df = prepare_noise_dataframe(df)
    if df.empty:
        raise ValueError("Nenhum dado de ruído customizado encontrado para os tipos selecionados.")

    grouped = (
        df.groupby(["noise_kind", "noise_level"], observed=True)[["secret", "success_probability"]]
        .apply(pick_value_for_group)
        .reset_index(name="success_probability")
    )

    # Inserir ponto ideal (baseline) para cada noise_kind, se existir
    if baseline is not None:
        ideal_rows = [
            {"noise_kind": nk, "noise_level": "ideal", "success_probability": baseline}
            for nk in NOISE_KINDS_OF_INTEREST
        ]
        grouped = pd.concat([pd.DataFrame(ideal_rows), grouped], ignore_index=True)

    grouped["Noise"] = grouped["noise_kind"].map(NOISE_KINDS_OF_INTEREST)
    grouped["noise_level"] = pd.Categorical(grouped["noise_level"], categories=NOISE_LEVEL_ORDER, ordered=True)
    grouped = grouped.sort_values(["noise_kind", "noise_level"])
    return grouped


def build_mean_survival_dataframe(
    df: pd.DataFrame, baseline_ci: tuple[float | None, float | None, float | None]
) -> pd.DataFrame:
    """Agrupa pela média e intervalo de confiança para todos os segredos."""
    df = prepare_noise_dataframe(df)
    if df.empty:
        raise ValueError("Nenhum dado de ruído customizado encontrado para os tipos selecionados.")

    def aggregate_mean(group: pd.DataFrame) -> pd.Series:
        mean_val, lower, upper = mean_confidence_interval(group["success_probability"])
        return pd.Series({"success_probability": mean_val, "ci_lower": lower, "ci_upper": upper})

    grouped = (
        df.groupby(["noise_kind", "noise_level"], observed=True)
        .apply(aggregate_mean)
        .dropna(subset=["success_probability"])
        .reset_index()
    )

    # Inserir ponto ideal (baseline) para cada noise_kind, se existir
    baseline_mean, baseline_lower, baseline_upper = baseline_ci
    if baseline_mean is not None:
        ideal_rows = [
            {
                "noise_kind": nk,
                "noise_level": "ideal",
                "success_probability": baseline_mean,
                "ci_lower": baseline_lower,
                "ci_upper": baseline_upper,
            }
            for nk in NOISE_KINDS_OF_INTEREST
        ]
        grouped = pd.concat([pd.DataFrame(ideal_rows), grouped], ignore_index=True)

    grouped["Noise"] = grouped["noise_kind"].map(NOISE_KINDS_OF_INTEREST)
    grouped["noise_level"] = pd.Categorical(grouped["noise_level"], categories=NOISE_LEVEL_ORDER, ordered=True)
    grouped = grouped.sort_values(["noise_kind", "noise_level"])
    return grouped


def plot_survival_curve():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Arquivo de resultados não encontrado em {RESULTS_FILE.resolve()}")

    df = pd.read_csv(
        RESULTS_FILE,
        dtype={
            "category": "string",
            "backend": "string",
            "secret": "string",
            "counts": "string",
        },
    )
    df["success_probability"] = df.apply(lambda row: success_probability(row["secret"], row["counts"]), axis=1)
    df = df.dropna(subset=["success_probability"])

    # Baseline ideal: pior caso se existir; senão, média
    ideal_df = df[df["category"] == "ideal"]
    baseline = pick_value_for_group(ideal_df) if not ideal_df.empty else None
    baseline_ci = (
        mean_confidence_interval(ideal_df["success_probability"]) if not ideal_df.empty else (None, None, None)
    )

    survival_df = build_survival_dataframe(df, baseline)

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    palette = sns.color_palette("colorblind", n_colors=survival_df["Noise"].nunique())
    markers = ["o", "s", "D", "^", "v", "P"]
    # Linhas com estilos distintos para evitar sobreposição visual
    dashes = ["", (4, 2), (2, 2), (6, 2, 2, 2), (1, 1), (5, 1, 1, 1)]  # sólido, tracejado etc.
    for idx, (noise, df_noise) in enumerate(survival_df.groupby("Noise")):
        dash = dashes[idx % len(dashes)]
        color = palette[idx % len(palette)]
        marker = markers[idx % len(markers)]
        sns.lineplot(
            data=df_noise,
            x="noise_level",
            y="success_probability",
            label=noise,
            marker=marker,
            linewidth=2.5,
            markersize=8,
            color=color,
            dashes=dash,
        )

    plt.xlabel("Nível de ruído")
    plt.ylabel("Probabilidade de sucesso (acerto do segredo)")
    plt.ylim(0, 1.05)
    plt.legend(title="Tipo de ruído")
    # Tradução dos ticks conforme bv.py
    locs, labels = plt.xticks()
    translated = [NOISE_LEVEL_TRANSLATIONS.get(lbl.get_text(), lbl.get_text()) for lbl in labels]
    plt.xticks(locs, translated)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "survival_curve_custom_noise.png"
    plt.savefig(save_path, dpi=300)
    print(f"Curva de sobrevivência salva em: {save_path}")

    # Segundo gráfico: média de todos os experimentos com faixa de confiança hachurada
    mean_df = build_mean_survival_dataframe(df, baseline_ci)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    palette = sns.color_palette("colorblind", n_colors=mean_df["Noise"].nunique())
    for idx, (noise, df_noise) in enumerate(mean_df.groupby("Noise")):
        dash = dashes[idx % len(dashes)]
        color = palette[idx % len(palette)]
        marker = markers[idx % len(markers)]
        x_positions = df_noise["noise_level"].cat.codes.to_numpy()
        y_vals = df_noise["success_probability"].to_numpy(dtype=float)
        lower = df_noise["ci_lower"].to_numpy(dtype=float)
        upper = df_noise["ci_upper"].to_numpy(dtype=float)

        band = ax.fill_between(
            x_positions,
            lower,
            upper,
            facecolor=color,
            edgecolor="none",
            alpha=0.18,
        )

        sns.lineplot(
            x=x_positions,
            y=y_vals,
            label=noise,
            marker=marker,
            linewidth=2.5,
            markersize=8,
            color=color,
            dashes=dash,
        )

    plt.xlabel("Nível de ruído")
    plt.ylabel("Probabilidade de sucesso (média de todos os segredos)")
    plt.ylim(0, 1.05)
    plt.legend(title="Tipo de ruído")
    translated = [NOISE_LEVEL_TRANSLATIONS.get(level, level) for level in NOISE_LEVEL_ORDER]
    plt.xticks(range(len(NOISE_LEVEL_ORDER)), translated)
    plt.tight_layout()

    save_path_mean = OUTPUT_DIR / "survival_curve_custom_noise_mean.png"
    plt.savefig(save_path_mean, dpi=300)
    print(f"Curva média de sobrevivência salva em: {save_path_mean}")


if __name__ == "__main__":
    plot_survival_curve()
