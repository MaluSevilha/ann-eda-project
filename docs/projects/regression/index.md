---
project: regression
dataset: https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai
team:
  - Maria Luiza Sevilha
  - Vinicus Barreto Pomilio
ai_use: "OpenAI Codex e Anthropic ClaudeCode foram usados no apoio ao código e ao texto; a equipe revisou e é responsável por explicar todas as decisões."
---

# Projeto: Regressão com redes neurais

## 0. Ponto de partida

O objetivo permanece o definido na EDA: prever `median_earnings_4yr_usd`, a
renda anual mediana quatro anos depois da conclusão. O pré-processador é
importado diretamente de [`pipeline.py`](../eda/code/pipeline.py) e mantém as
estratégias justificadas na primeira entrega: clipping por IQR, imputação,
indicadores de ausência, padronização e one-hot encoding com agrupamento de
categorias raras.

A métrica principal, registrada antes de qualquer treinamento, é o **MAE em
dólares**. A rede será considerada bem-sucedida se alcançar MAE pelo menos
**15% menor que o melhor baseline obrigatório** (`DummyRegressor` ou Ridge),
isto é, `MAE_MLP ≤ 0,85 × MAE_melhor_baseline`. RMSE e R² serão métricas
secundárias.

O split externo reproduz exatamente a EDA com `GroupShuffleSplit`, agrupado por
`opeid6`, `test_size=0.20` e semente 42. A validação sai apenas do treino da
EDA, novamente agrupada por instituição, com proporção 20% e semente 42:

| Conjunto | Programas | Instituições |
|---|---:|---:|
| Treino | 36.565 | 2.969 |
| Validação | 9.758 | 743 |
| Teste bloqueado | 11.740 | 928 |

Há zero instituições compartilhadas entre quaisquer dois conjuntos. O
pré-processador é ajustado somente nas 36.565 linhas do treino interno e
transforma treino e validação em matrizes CSR `float32` com 420 features e
zero `NaN`. O teste não é transformado, inspecionado por estatísticas do alvo
nem usado para métricas nesta etapa. Identidades, hashes e invariantes estão no
[`split_manifest.json`](split_manifest.json) e podem ser conferidos com:

```bash
python -m docs.projects.regression.code.data --check
```

## 1. Baselines

Pendente — Entregável 2.

## 2. A rede

Pendente — Entregável 3.

## 3. Treinamento

Pendente — Entregável 3.

## 4. Regularização e hiperparâmetros

Pendente — Entregável 4.

## 5. Avaliação no teste

Pendente — Entregável 6. O teste permanece bloqueado até esta etapa.

## 6. Análise de erros

Pendente — Entregável 7.

## 7. Conclusão

Pendente — Entregável 8.
