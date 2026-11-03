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

Os dois baselines obrigatórios usam exatamente as mesmas 420 features geradas
pelo pipeline ajustado no treino interno. O teste continua bloqueado: todos os
números desta seção são da **validação**.

O baseline trivial é `DummyRegressor(strategy="mean")`: ele prevê para todo
programa a renda média observada no treino. Portanto, mede quanto se ganha
apenas por sair de uma constante, sem usar nenhuma feature. O baseline linear
é Ridge, que combina as features por uma função linear e penaliza coeficientes
grandes. Como o pipeline já padroniza as numéricas, comparamos a grade de
`alpha` **0,001; 0,01; 0,1; 1; 10; 100; 1.000; 10.000**, declarada antes da
execução. A escolha é feita pelo menor MAE, a métrica principal.

--8<-- "docs/projects/regression/tables/baseline_validation.md"

O melhor baseline pela regra registrada foi **Ridge com `alpha=1`**, com MAE
de **US$ 7.813,89**, RMSE de **US$ 11.795,41** e R² de **0,816399**. O
`DummyRegressor` obteve MAE de **US$ 20.023,63**, então a regressão linear
reduziu o erro absoluto em aproximadamente **61,0%**. Isso indica que uma
combinação linear das features — especialmente após o tratamento das
categorias — já explica uma parcela grande da variação e será uma referência
exigente para a rede.

Ridge com `alpha=0,1` teve RMSE marginalmente menor (US$ 11.795,13), mas não é
o vencedor porque a seleção foi fixada pelo MAE antes da execução. Para cumprir
o critério de sucesso de 15%, a futura MLP deverá alcançar na validação MAE de
no máximo **US$ 6.641,81**. Nenhum desses resultados antecipa o desempenho no
teste.

## 2. A rede

Pendente

## 3. Treinamento

Pendente

## 4. Regularização e hiperparâmetros

Pendente

## 5. Avaliação no teste

Pendente

## 6. Análise de erros

Pendente

## 7. Conclusão

Pendente
