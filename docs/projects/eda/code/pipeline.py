"""Pipeline de pré-processamento para a futura rede neural de regressão.

O módulo não carrega dados nem treina modelos. Ele apenas expõe as listas de
features e ``build_preprocessor``, que deve ser ajustado exclusivamente no
conjunto de treino.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "median_earnings_4yr_usd"
GROUP_COLUMN = "opeid6"

NUMERIC_FEATURES = [
    "is_main_campus",
    "institution_latitude",
    "institution_longitude",
    "institution_is_hbcu",
    "institution_admission_rate",
    "institution_avg_sat",
    "institution_undergrad_enrollment",
    "institution_tuition_in_state_usd",
    "institution_tuition_out_state_usd",
    "awards_year1",
    "awards_year2",
    "outcomes_shared_across_campuses",
    "median_earnings_1yr_usd",
    "earnings_cohort_size_1yr",
    "median_debt_usd",
    "debt_borrower_count",
    "linked_occupations_count",
    "occupation_employment_2024_thousands",
    "occupation_growth_pct_2024_34",
    "occupation_growth_pct_max",
    "occupation_annual_openings_thousands",
    "occupation_median_wage_2024_usd",
    "ai_software_occupation_share",
    "expert_system_occupation_share",
    "ai_tools_max_per_occupation",
    "hot_technologies_mean_per_occupation",
]

CATEGORICAL_FEATURES = [
    "institution_control",
    "institution_state",
    "cip_title",
    "credential_name",
    "distance_education",
    "largest_linked_occupation",
    "occupation_typical_entry_education",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Identificadores, redundâncias e variáveis que usam o alvo ou informação futura.
DROPPED_COLUMNS = {
    "program_id": "identificador único",
    "unitid": "identificador institucional redundante",
    "opeid6": "identificador usado apenas para o split agrupado",
    "institution_name": "identificador de alta cardinalidade e nome inconsistente",
    "institution_city": "alta cardinalidade; estado representa a localização",
    "institution_region": "redundante com o estado",
    "cip_code_4digit": "redundante com cip_title e perde zeros à esquerda",
    "cip_family_code": "redundante com cip_title",
    "cip_family_title": "redundante com cip_title",
    "credential_level": "redundante com credential_name e contém código 99 não ordinal",
    "median_earnings_4yr_usd": "alvo",
    "earnings_cohort_size_4yr": "medido junto com o alvo",
    "national_median_earnings_4yr_usd": "agregado calculado a partir do alvo",
    "national_p25_earnings_4yr_usd": "agregado calculado a partir do alvo",
    "national_p75_earnings_4yr_usd": "agregado calculado a partir do alvo",
    "earnings_vs_national_pct": "derivado diretamente do alvo",
    "median_earnings_5yr_usd": "informação posterior ao alvo",
    "earnings_cohort_size_5yr": "informação posterior ao alvo",
    "not_working_count_5yr": "informação posterior ao alvo",
    "count_above_hs_threshold_5yr": "informação posterior ao alvo",
    "count_working_in_state_5yr": "informação posterior ao alvo",
    "median_monthly_payment_usd": "redundante com a dívida mediana",
    "earnings_growth_pct_1yr_to_5yr": "usa informação posterior ao alvo",
    "earnings_trajectory_category": "usa informação posterior ao alvo",
    "debt_to_earnings_1yr": "derivado de duas features já presentes",
    "debt_to_earnings_4yr": "derivado diretamente do alvo",
    "payment_to_income_pct_1yr": "derivado de duas features já presentes",
    "pct_above_hs_threshold_5yr": "informação posterior ao alvo",
    "pct_working_5yr": "informação posterior ao alvo",
    "pct_working_in_state_5yr": "informação posterior ao alvo",
    "linked_occupations_in_bls": "redundante com linked_occupations_count",
    "largest_linked_occupation_soc": "redundante com o nome da ocupação",
    "linked_occupations_in_onet": "redundante com linked_occupations_count",
    "occupations_using_ai_software": "redundante com a proporção de ocupações",
    "ai_tool_examples": "texto livre qualitativo, não uma feature tabular estável",
    "earnings_4yr_status": "revela diretamente se o alvo existe",
    "earnings_1yr_status": "a ausência já é capturada pelo imputador",
    "earnings_5yr_status": "informação posterior ao alvo",
    "debt_status": "a ausência já é capturada pelo imputador",
}


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsoriza cada coluna nos limites Q1 - k*IQR e Q3 + k*IQR."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.q1_ = np.nanquantile(values, 0.25, axis=0)
        self.q3_ = np.nanquantile(values, 0.75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - self.factor * iqr
        self.upper_ = self.q3_ + self.factor * iqr
        # IQR zero é comum em flags binárias e variáveis zero-infladas. Nesses
        # casos não há base para clipping; preservar a coluna evita apagar a
        # classe minoritária.
        constant_iqr = iqr == 0
        self.lower_[constant_iqr] = -np.inf
        self.upper_[constant_iqr] = np.inf
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(len(self.lower_))], dtype=object)
        return np.asarray(input_features, dtype=object)


def build_preprocessor(min_category_frequency: int = 20) -> ColumnTransformer:
    """Cria o pré-processador ainda não ajustado.

    Numéricas: clipping por IQR, mediana com indicadores de ausência e
    padronização. Categóricas: moda e one-hot, agrupando níveis raros e
    aceitando categorias inéditas no teste.
    """

    numeric = Pipeline(
        steps=[
            ("outliers", IQRClipper(factor=1.5)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
