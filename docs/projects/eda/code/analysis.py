"""Reproduz as tabelas, figuras e verificações do projeto de EDA.

Execute a partir da raiz do repositório:

    python docs/projects/eda/code/analysis.py

Nenhum modelo preditivo é treinado. A saída é apenas exploratória e deixa o
pré-processador pronto para a futura entrega de regressão.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupShuffleSplit

try:
    import umap
except ImportError as exc:  # pragma: no cover - mensagem útil em clone limpo
    raise SystemExit("Instale as dependências com: pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = CODE_DIR.parent / "figures"
TABLE_DIR = CODE_DIR.parent / "tables"
DATA_PATH = ROOT / "data/raw/college_majors_2026.csv"
sys.path.insert(0, str(CODE_DIR))

from pipeline import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DROPPED_COLUMNS,
    GROUP_COLUMN,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
PROJECTION_SAMPLE = 3000
DATA_DICTIONARY_PATH = TABLE_DIR / "data_dictionary.md"


def validate_schema(raw: pd.DataFrame) -> None:
    """Garante que cada coluna bruta tenha um único papel documentado."""
    groups = {
        "features numéricas": set(NUMERIC_FEATURES),
        "features categóricas": set(CATEGORICAL_FEATURES),
        "alvo": {TARGET},
        "agrupamento do split": {GROUP_COLUMN},
        "colunas excluídas": set(DROPPED_COLUMNS),
    }
    labels = list(groups)
    expected_sizes = {
        "features numéricas": 26,
        "features categóricas": 7,
        "alvo": 1,
        "agrupamento do split": 1,
        "colunas excluídas": 37,
    }
    invalid_sizes = {
        label: {"esperado": expected_sizes[label], "observado": len(columns)}
        for label, columns in groups.items()
        if len(columns) != expected_sizes[label]
    }
    overlaps = {
        f"{labels[i]} × {labels[j]}": sorted(groups[labels[i]] & groups[labels[j]])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if groups[labels[i]] & groups[labels[j]]
    }
    classified = set().union(*groups.values())
    observed = set(raw.columns)
    missing = sorted(classified - observed)
    unexpected = sorted(observed - classified)
    if len(raw.columns) != 72 or invalid_sizes or overlaps or missing or unexpected:
        raise SystemExit(
            "Esquema inválido: "
            f"colunas={len(raw.columns)}, tamanhos={invalid_sizes}, sobreposições={overlaps}, "
            f"ausentes={missing}, inesperadas={unexpected}"
        )

    dictionary_text = DATA_DICTIONARY_PATH.read_text(encoding="utf-8")
    documented = re.findall(r"^\| `([^`]+)` \|", dictionary_text, flags=re.MULTILINE)
    duplicated = sorted({column for column in documented if documented.count(column) > 1})
    undocumented = sorted(observed - set(documented))
    extra_documented = sorted(set(documented) - observed)
    if len(documented) != 72 or duplicated or undocumented or extra_documented:
        raise SystemExit(
            "Dicionário inválido: "
            f"linhas={len(documented)}, duplicadas={duplicated}, "
            f"não documentadas={undocumented}, extras={extra_documented}"
        )


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_table(df: pd.DataFrame, filename: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / filename, index=True)
    (TABLE_DIR / Path(filename).with_suffix(".md")).write_text(
        df.round(3).to_markdown(), encoding="utf-8"
    )


def compact_money_axis(ax: plt.Axes, axis: str = "x") -> None:
    formatter = matplotlib.ticker.FuncFormatter(lambda x, _: f"US$ {x/1000:.0f}k")
    target_axis = ax.xaxis if axis == "x" else ax.yaxis
    target_axis.set_major_formatter(formatter)
    target_axis.set_major_locator(matplotlib.ticker.MaxNLocator(6))


def target_figure(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(data[TARGET], bins=45, kde=True, ax=axes[0], color="#2166ac")
    axes[0].axvline(data[TARGET].median(), color="#b2182b", linestyle="--", label="Mediana")
    axes[0].axvline(data[TARGET].mean(), color="#2ca25f", linestyle=":", label="Média")
    axes[0].set(title="Figura 1: Distribuição do alvo", xlabel="Renda mediana no 4º ano (US$)", ylabel="Programas")
    axes[0].legend()
    compact_money_axis(axes[0])
    sns.boxplot(x=data[TARGET], ax=axes[1], color="#67a9cf")
    axes[1].set(title="Figura 1: Cauda e valores extremos", xlabel="Renda mediana no 4º ano (US$)")
    compact_money_axis(axes[1])
    fig.suptitle("O alvo é assimétrico à direita", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig01_target.png")


def numeric_univariate_figure(train: pd.DataFrame) -> None:
    selected = [
        TARGET,
        "median_earnings_1yr_usd",
        "median_debt_usd",
        "institution_tuition_in_state_usd",
        "institution_avg_sat",
        "institution_admission_rate",
        "institution_undergrad_enrollment",
        "awards_year2",
        "occupation_median_wage_2024_usd",
        "occupation_growth_pct_2024_34",
        "occupation_annual_openings_thousands",
        "ai_software_occupation_share",
    ]
    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    abbreviate = matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:.0f}"
    )
    for ax, col in zip(axes.flat, selected):
        sns.histplot(train[col], bins=35, ax=ax, color="#2166ac")
        ax.set_title(col.replace("_", " "), fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Contagem")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.xaxis.set_major_formatter(abbreviate)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Figura 2: Distribuições numéricas selecionadas (treino)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig02_numeric_distributions.png")


def categorical_univariate_figure(train: pd.DataFrame) -> None:
    columns = CATEGORICAL_FEATURES  # as 7 categóricas que entram no modelo
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    for ax, col in zip(axes.flat, columns):
        counts = train[col].fillna("Ausente").value_counts().head(10).sort_values()
        ax.barh(counts.index.astype(str), counts.values, color="#4393c3")
        ax.set_title(col.replace("_", " "), fontsize=10)
        ax.set_xlabel("Programas no treino")
        ax.tick_params(axis="y", labelsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("Figura 3: Categorias mais frequentes (treino)", fontweight="bold", y=1.002)
    fig.tight_layout()
    save_figure(fig, "fig03_categorical_frequencies.png")


def correlation_figures(train: pd.DataFrame, corr: pd.DataFrame) -> None:
    strongest_to_target = (
        corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(11).index.tolist()
    )
    shown = strongest_to_target + [TARGET]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr.loc[shown, shown],
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title("Figura 4: correlação de Spearman entre as variáveis mais ligadas ao alvo")
    fig.tight_layout()
    save_figure(fig, "fig04_spearman_heatmap.png")

    sample = train.sample(min(5000, len(train)), random_state=RANDOM_STATE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.regplot(
        data=sample,
        x="median_earnings_1yr_usd",
        y=TARGET,
        scatter_kws={"alpha": 0.18, "s": 10},
        line_kws={"color": "#b2182b"},
        ax=axes[0],
    )
    axes[0].set(xlabel="Renda mediana no 1º ano (US$)", ylabel="Renda mediana no 4º ano (US$)")
    axes[0].annotate(
        f"ρ = {corr.loc['median_earnings_1yr_usd', TARGET]:.3f}",
        xy=(0.05, 0.92), xycoords="axes fraction", fontweight="bold",
    )
    sns.regplot(
        data=sample,
        x="occupation_median_wage_2024_usd",
        y=TARGET,
        scatter_kws={"alpha": 0.18, "s": 10},
        line_kws={"color": "#b2182b"},
        ax=axes[1],
    )
    axes[1].set(xlabel="Salário da ocupação associada (US$)", ylabel="Renda mediana no 4º ano (US$)")
    axes[1].annotate(
        f"ρ = {corr.loc['occupation_median_wage_2024_usd', TARGET]:.3f}",
        xy=(0.05, 0.92), xycoords="axes fraction", fontweight="bold",
    )
    fig.suptitle("Figura 5: Relações numéricas com o alvo", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig05_numeric_target_scatter.png")


def categorical_target_figures(train: pd.DataFrame) -> None:
    for column in ["credential_name", "institution_control", "cip_family_title", "distance_education"]:
        grouped = train.groupby(column)[TARGET].agg(["count", "median", "mean", "std"]).sort_values("median", ascending=False)
        save_table(grouped, f"target_by_{column}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    order = train.groupby("credential_name")[TARGET].median().sort_values().index
    sns.boxplot(data=train, y="credential_name", x=TARGET, order=order, showfliers=False, ax=axes[0], color="#67a9cf")
    axes[0].set(xlabel="Renda mediana no 4º ano (US$)", ylabel="Credencial", title="Por nível de credencial")
    controls = train.groupby("institution_control")[TARGET].median().sort_values().index
    sns.boxplot(data=train, y="institution_control", x=TARGET, order=controls, showfliers=False, ax=axes[1], color="#ef8a62")
    axes[1].set(xlabel="Renda mediana no 4º ano (US$)", ylabel="Controle", title="Por tipo de instituição")
    compact_money_axis(axes[0]); compact_money_axis(axes[1])
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    fig.suptitle("Figura 6: Distribuição do alvo por categorias institucionais", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig06_categories_target.png")

    top_families = train["cip_family_title"].value_counts().head(12).index
    plot_data = train[train["cip_family_title"].isin(top_families)]
    order = plot_data.groupby("cip_family_title")[TARGET].median().sort_values().index
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=plot_data, y="cip_family_title", x=TARGET, order=order, showfliers=False, ax=ax, color="#92c5de")
    ax.set(title="Figura 7: Alvo nas 12 famílias de curso mais frequentes", xlabel="Renda mediana no 4º ano (US$)", ylabel="Família CIP")
    compact_money_axis(ax)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    fig.tight_layout()
    save_figure(fig, "fig07_cip_family_target.png")


def numeric_categorical_figure(train: pd.DataFrame) -> None:
    sample = train.sample(min(12000, len(train)), random_state=RANDOM_STATE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(data=sample, y="institution_control", x="institution_tuition_in_state_usd", showfliers=False, ax=axes[0], color="#ef8a62")
    axes[0].set(xlabel="Mensalidade anual in-state (US$)", ylabel="Controle institucional")
    compact_money_axis(axes[0])
    axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    sns.boxplot(data=sample, y="credential_name", x="occupation_median_wage_2024_usd", showfliers=False, ax=axes[1], color="#67a9cf")
    axes[1].set(xlabel="Salário da ocupação associada (US$)", ylabel="Credencial")
    compact_money_axis(axes[1])
    axes[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    fig.suptitle("Figura 8: Numéricas agrupadas por categorias", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig08_numeric_categorical.png")


def clean_feature_label(name: str) -> str:
    """Remove prefixos técnicos do ColumnTransformer para exibição em figuras."""
    name = name.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        return "ausência: " + name.replace("missingindicator_", "").replace("_", " ")
    name = name.replace("_infrequent_sklearn", " (outras)")
    return name.replace("_", " ")


def projection_figures(X_sample_t: np.ndarray, y_sample: pd.Series, feature_names: np.ndarray):
    n_components = min(50, X_sample_t.shape[1], X_sample_t.shape[0])
    pca_full = PCA(n_components=n_components, random_state=RANDOM_STATE).fit(X_sample_t)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    coords = pca_full.transform(X_sample_t)[:, :2]
    pc12 = float(cumulative[1])

    loading_strength = np.sqrt(pca_full.components_[0] ** 2 + pca_full.components_[1] ** 2)
    top_idx = np.argsort(loading_strength)[-12:]
    loading_table = pd.DataFrame(
        {"feature": feature_names[top_idx], "PC1": pca_full.components_[0, top_idx], "PC2": pca_full.components_[1, top_idx], "magnitude": loading_strength[top_idx]}
    ).set_index("feature").sort_values("magnitude", ascending=False)
    save_table(loading_table, "pca_loadings.csv")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=y_sample, cmap="viridis", s=9, alpha=0.65)
    axes[0].set(xlabel="PC1", ylabel="PC2", title=f"Projeção (PC1+PC2 = {pc12:.1%})")
    fig.colorbar(sc, ax=axes[0], label="Renda 4º ano (US$)")
    axes[1].plot(np.arange(1, len(cumulative) + 1), cumulative, color="#2166ac")
    axes[1].axhline(0.80, color="#b2182b", linestyle="--", label="80%")
    axes[1].set(xlabel="Componentes", ylabel="Variância acumulada", title="Variância explicada")
    axes[1].legend()
    shown = loading_table.sort_values("magnitude")
    clean_labels = [clean_feature_label(name) for name in shown.index.astype(str)]
    axes[2].barh(clean_labels, shown["magnitude"], color="#4393c3")
    axes[2].set(xlabel="Magnitude nos dois PCs", title="Maiores loadings PC1/PC2")
    axes[2].tick_params(axis="y", labelsize=9)
    fig.suptitle("Figura 9: PCA no treino pré-processado", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig09_pca.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, perplexity in zip(axes, [30, 50]):
        emb = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=RANDOM_STATE, max_iter=1000).fit_transform(X_sample_t)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_sample, cmap="viridis", s=8, alpha=0.7)
        ax.set(title=f"perplexity = {perplexity}", xlabel="t-SNE 1", ylabel="t-SNE 2")
    fig.colorbar(sc, ax=axes, label="Renda 4º ano (US$)", shrink=0.8)
    fig.suptitle("Figura 10: t-SNE: sensibilidade à perplexidade", fontweight="bold")
    save_figure(fig, "fig10_tsne.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, neighbors in zip(axes, [15, 50]):
        emb = umap.UMAP(n_components=2, n_neighbors=neighbors, min_dist=0.1, random_state=RANDOM_STATE, n_jobs=1).fit_transform(X_sample_t)
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_sample, cmap="viridis", s=8, alpha=0.7)
        ax.set(title=f"n_neighbors = {neighbors}", xlabel="UMAP 1", ylabel="UMAP 2")
    fig.colorbar(sc, ax=axes, label="Renda 4º ano (US$)", shrink=0.8)
    fig.suptitle("Figura 11: UMAP: sensibilidade ao número de vizinhos", fontweight="bold")
    save_figure(fig, "fig11_umap.png")
    return pc12


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    if not DATA_PATH.exists():
        raise SystemExit("Dataset ausente. Execute primeiro: python download_data.py")

    raw = pd.read_csv(DATA_PATH, low_memory=False)
    validate_schema(raw)
    raw_rows, raw_cols = raw.shape
    target_available = int(raw[TARGET].notna().sum())
    target_missing = raw_rows - target_available
    full_duplicates = int(raw.duplicated().sum())
    id_duplicates = int(raw["program_id"].duplicated().sum())

    missing = pd.DataFrame({"n_ausentes": raw.isna().sum(), "percentual": raw.isna().mean() * 100}).sort_values("percentual", ascending=False)
    save_table(missing, "missing_values.csv")

    quality_issues = pd.DataFrame(
        {
            "condição": [
                "pct_above_hs_threshold_5yr > 100",
                "pct_working_in_state_5yr > 100",
                "crescimento 1º–5º ano > 200%",
                "credential_level = 99",
                "tuition in-state igual a zero",
            ],
            "quantidade": [
                int((raw["pct_above_hs_threshold_5yr"] > 100).sum()),
                int((raw["pct_working_in_state_5yr"] > 100).sum()),
                int((raw["earnings_growth_pct_1yr_to_5yr"] > 200).sum()),
                int((raw["credential_level"] == 99).sum()),
                int((raw["institution_tuition_in_state_usd"] == 0).sum()),
            ],
            "classificação": [
                "Inconsistência matemática",
                "Inconsistência matemática",
                "Valor extremo possível",
                "Categoria válida: Non-Credential Program",
                "Valor plausível",
            ],
            "decisão": [
                "Excluir: variável de 5 anos, posterior ao alvo",
                "Excluir: variável de 5 anos, posterior ao alvo",
                "Excluir: usa informação posterior ao alvo",
                "Usar credential_name; excluir apenas o código redundante",
                "Manter a feature; não corrigir como erro",
            ],
        }
    )
    save_table(quality_issues.set_index("condição"), "quality_issues.csv")

    data = raw.loc[raw[TARGET].notna() & raw["institution_control"].ne("Foreign")].copy()
    excluded_foreign = target_available - len(data)
    target_stats = data[TARGET].describe()
    target_skew = float(data[TARGET].skew())
    target_figure(data)

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(data, groups=data[GROUP_COLUMN]))
    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    split_summary = pd.DataFrame(
        {
            "programas": [len(train), len(test)],
            "instituições": [train[GROUP_COLUMN].nunique(), test[GROUP_COLUMN].nunique()],
            "média do alvo (US$)": [train[TARGET].mean(), test[TARGET].mean()],
            "mediana do alvo (US$)": [train[TARGET].median(), test[TARGET].median()],
        },
        index=["Treino", "Teste"],
    )
    save_table(split_summary, "split_target_summary.csv")

    numeric_stats = train[NUMERIC_FEATURES + [TARGET]].describe(percentiles=[0.25, 0.5, 0.75]).T
    numeric_stats["median"] = train[NUMERIC_FEATURES + [TARGET]].median()
    numeric_stats["skew"] = train[NUMERIC_FEATURES + [TARGET]].skew()
    numeric_stats["missing_pct"] = train[NUMERIC_FEATURES + [TARGET]].isna().mean() * 100
    numeric_stats = numeric_stats[["count", "missing_pct", "mean", "median", "std", "min", "25%", "75%", "max", "skew"]]
    save_table(numeric_stats, "numeric_summary_train.csv")

    categorical_source = CATEGORICAL_FEATURES + ["cip_family_title"]
    categorical_summary = pd.DataFrame(
        {
            "cardinalidade": train[categorical_source].nunique(dropna=True),
            "ausentes_pct": train[categorical_source].isna().mean() * 100,
            "categoria_mais_frequente": [train[c].mode(dropna=True).iloc[0] for c in categorical_source],
            "frequencia_max_pct": [train[c].value_counts(normalize=True, dropna=True).iloc[0] * 100 for c in categorical_source],
            "categorias_raras_lt20": [int((train[c].value_counts() < 20).sum()) for c in categorical_source],
        }
    )
    save_table(categorical_summary, "categorical_summary_train.csv")
    numeric_univariate_figure(train)
    categorical_univariate_figure(train)

    corr = train[NUMERIC_FEATURES + [TARGET]].corr(method="spearman")
    save_table(corr, "spearman_correlation_train.csv")
    arr = corr.abs().to_numpy(copy=True)
    np.fill_diagonal(arr, np.nan)
    pair_i, pair_j = np.unravel_index(np.nanargmax(arr), arr.shape)
    strongest_pair = [corr.index[pair_i], corr.columns[pair_j]]
    strongest_value = float(corr.iloc[pair_i, pair_j])
    corr_target = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    save_table(corr_target.to_frame("spearman_com_alvo"), "target_correlations_train.csv")
    correlation_figures(train, corr)
    categorical_target_figures(train)
    numeric_categorical_figure(train)

    # Quantifica a estratégia de outliers usando limites aprendidos no treino.
    num = train[NUMERIC_FEATURES]
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    active = iqr > 0
    outlier_mask = ((num.loc[:, active] < lower[active]) | (num.loc[:, active] > upper[active])).any(axis=1)
    outlier_rows = int(outlier_mask.sum())

    X_train = train[MODEL_FEATURES]
    X_test = test[MODEL_FEATURES]
    preprocess = build_preprocessor(min_category_frequency=20)
    X_train_t = preprocess.fit_transform(X_train)
    X_test_t = preprocess.transform(X_test)
    feature_names = preprocess.get_feature_names_out()
    nan_train = int(np.isnan(X_train_t.data).sum() if hasattr(X_train_t, "data") else np.isnan(X_train_t).sum())
    nan_test = int(np.isnan(X_test_t.data).sum() if hasattr(X_test_t, "data") else np.isnan(X_test_t).sum())
    pd.Series(feature_names, name="feature").to_csv(TABLE_DIR / "pipeline_feature_names.csv", index=False)

    unseen = X_test.iloc[[0]].copy()
    unseen["institution_control"] = "Categoria nunca vista"
    unseen_t = preprocess.transform(unseen)
    unseen_ok = unseen_t.shape[1] == X_train_t.shape[1]

    rng = np.random.default_rng(RANDOM_STATE)
    sample_positions = rng.choice(len(train), size=min(PROJECTION_SAMPLE, len(train)), replace=False)
    X_sample_t = X_train_t[sample_positions]
    if hasattr(X_sample_t, "toarray"):
        X_sample_t = X_sample_t.toarray()
    y_sample = train[TARGET].iloc[sample_positions].to_numpy()
    pc12 = projection_figures(X_sample_t, y_sample, feature_names)

    results = {
        "raw_shape": [raw_rows, raw_cols],
        "target_available": target_available,
        "target_missing": target_missing,
        "target_missing_pct": target_missing / raw_rows * 100,
        "modeling_rows_domestic": len(data),
        "excluded_foreign_with_target": excluded_foreign,
        "duplicates_full_row": full_duplicates,
        "duplicates_program_id": id_duplicates,
        "target_mean": float(target_stats["mean"]),
        "target_median": float(target_stats["50%"]),
        "target_std": float(target_stats["std"]),
        "target_min": float(target_stats["min"]),
        "target_max": float(target_stats["max"]),
        "target_skew": target_skew,
        "train_shape_raw": [len(train), len(MODEL_FEATURES)],
        "test_shape_raw": [len(test), len(MODEL_FEATURES)],
        "train_institutions": int(train[GROUP_COLUMN].nunique()),
        "test_institutions": int(test[GROUP_COLUMN].nunique()),
        "institution_overlap": int(len(set(train[GROUP_COLUMN]) & set(test[GROUP_COLUMN]))),
        "most_missing_column": str(missing.index[0]),
        "most_missing_count": int(missing.iloc[0]["n_ausentes"]),
        "most_missing_pct": float(missing.iloc[0]["percentual"]),
        "strongest_numeric_pair": strongest_pair,
        "strongest_numeric_pair_spearman": strongest_value,
        "strongest_target_feature": str(corr_target.index[0]),
        "strongest_target_spearman": float(corr_target.iloc[0]),
        "outlier_rows_train": outlier_rows,
        "outlier_rows_train_pct": outlier_rows / len(train) * 100,
        "pc1_pc2_explained_variance": pc12,
        "pipeline_train_shape": list(X_train_t.shape),
        "pipeline_test_shape": list(X_test_t.shape),
        "pipeline_nan_train": nan_train,
        "pipeline_nan_test": nan_test,
        "unseen_category_ok": bool(unseen_ok),
        "n_pipeline_features": len(feature_names),
    }
    (CODE_DIR.parent / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
