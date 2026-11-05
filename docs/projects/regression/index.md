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

A rede de referência recebe as **420 features** do mesmo pipeline usado pelos
baselines e tem arquitetura **420 → 128 → 64 → 1**. As duas camadas ocultas
usam ReLU, o que permite aprender relações não lineares; a saída tem uma única
unidade linear, adequada à regressão porque não limita artificialmente a faixa
das previsões.

Os pesos das camadas ocultas usam inicialização de Kaiming, coerente com ReLU;
a saída usa Xavier e todos os vieses começam em zero. A rede tem **62.209
parâmetros treináveis**, equivalentes a **1,70 parâmetro por linha** nas 36.565
linhas do treino interno. É capacidade suficiente para uma referência não
linear, mas grande o bastante para exigir a análise de overfitting abaixo.

Esta primeira rede é deliberadamente **sem regularização**: não usa dropout,
BatchNorm/LayerNorm nem weight decay. Ela será o controle para as comparações
do Entregável 4, e não a configuração final.

## 3. Treinamento

O alvo recebe `log1p` para reduzir a assimetria observada na EDA e depois é
padronizado pela média (**10,9913698970**) e pelo desvio
(**0,4008667477**) aprendidos somente no treino. A rede otimiza Huber loss com
`delta=1` nessa escala; todas as previsões são invertidas para dólares antes de
calcular MAE, RMSE e R².

O treinamento usa Adam, learning rate **0,001**, batch **256**, no máximo 200
épocas e semente 42. O early stopping monitora diretamente o MAE em dólares,
com paciência 15 e melhora mínima de US$ 1, guardando em memória e restaurando
os melhores pesos. A execução canônica usa CPU e operações determinísticas.

![Curvas de loss no treino e validação e MAE de validação por época](figures/fig01_training_curves.png)

A rede aprende rapidamente nas primeiras épocas: o MAE cai de US$ 7.241,40 na
época 1 para US$ 6.584,17 na época 5. O melhor ponto pela métrica principal é a
**época 11**, com **MAE US$ 6.566,01**, **RMSE US$ 10.205,46** e **R²
0,862559**. A execução para na época **26** e restaura os pesos da época 11.

Depois da época 11, a loss de treino continua caindo, de 0,03943 para 0,02677,
enquanto a loss de validação sobe de 0,06080 para 0,06510 e o MAE oscila sem
nova melhora. Essa abertura entre as curvas é evidência de overfitting e
justifica tanto o early stopping quanto a comparação de regularização na etapa
seguinte. A menor Huber loss de validação ocorre antes do menor MAE em dólares;
isso reforça a decisão de restaurar pela métrica principal, e não pela loss
transformada.

Na validação, esta referência reduz o MAE do Ridge em **15,97%** e fica US$
**75,80** abaixo do limite preliminar de US$ 6.641,81. Isso ainda não responde
ao critério final: arquitetura e regularização serão escolhidas na validação e
o teste só será aberto na etapa 5.

## 4. Regularização e hiperparâmetros

A busca foi declarada antes da execução com **12 configurações** e manteve o
mesmo protocolo da referência: Huber, alvo em `log1p`, batch 256, no máximo
200 épocas, early stopping, CPU e semente 42. Comparamos diretamente a rede
sem regularização com dropout, weight decay e LayerNorm. A grade também variou
learning rate, largura e profundidade; todas as escolhas foram feitas somente
pelo MAE de validação.

--8<-- "docs/projects/regression/tables/hyperparameter_search.md"

![MAE das configurações e curvas das técnicas de regularização](figures/fig02_regularization_search.png)

Todas as 12 configurações terminaram sem falhas. Dropout foi a intervenção
isolada mais efetiva: com taxa 0,10, o MAE caiu de **US$ 6.566,01** para **US$
6.311,16**. Weight decay isolado teve efeito bem menor: `1e-4` chegou a **US$
6.525,00**, enquanto `1e-5` praticamente empatou com a referência (**US$
6.563,99**). LayerNorm não ajudou neste problema tabular e obteve **US$
6.645,81** sem weight decay, resultado pior que o controle.

A largura adicional também não trouxe ganho: a rede com 140.801 parâmetros
ficou em US$ 6.378,43. A versão estreita, com apenas 29.057 parâmetros, chegou
a US$ 6.342,91 e superou a versão larga, sinal de que aumentar capacidade não
era a principal necessidade. A rede profunda ficou entre elas, com US$
6.374,71. Esses resultados são coerentes com o overfitting observado na
referência, e não com underfitting por falta de parâmetros.

O vencedor provisório foi **`low_lr_regularized`**: arquitetura 420 → 128 → 64
→ 1, dropout 0,10, weight decay `1e-5` e learning rate `3e-4`. Ele alcançou
**MAE US$ 6.265,05**, **RMSE US$ 9.821,37** e **R² 0,872710** na época 31,
parando na época 46. Isso reduz o MAE da referência em **4,58%** e o do Ridge
em **19,82%**, ficando US$ **376,76** abaixo do critério preliminar.

As curvas mostram por que o learning rate menor foi útil: o vencedor convergiu
mais lentamente e atingiu sua melhor época depois das outras configurações.
Mesmo com dropout, após a época 31 a loss de treino caiu de 0,04761 para
0,04199, enquanto a de validação subiu de 0,05594 para 0,05697; portanto,
early stopping ainda é necessário. LayerNorm apresentou oscilações maiores e
começou a piorar cedo, com melhor MAE já na época 5. A configuração vencedora
é apenas uma recomendação desta busca; seu congelamento ocorre na próxima
etapa, ainda sem acessar o teste.

### Configuração congelada

A recomendação foi congelada em [`final_config.json`](final_config.json) antes
de qualquer avaliação no teste. Esse contrato fixa arquitetura, transformação
do alvo, loss, otimizador, learning rate, batch, regularização, early stopping
e a semente canônica **42**. A época 31 é somente a evidência da seleção: novos
treinamentos continuarão restaurando a melhor época de validação, em vez de
fixar a duração em 31 épocas.

O arquivo é derivado da primeira linha da busca e validado contra ela; uma
mudança no vencedor, nas métricas ou no protocolo faz a verificação falhar.
Ele registra explicitamente `test_accessed: false`, e o script de busca rejeita
objetos que exponham campos de teste. As sementes adicionais exigidas para a
avaliação final serão declaradas no protocolo da próxima etapa, antes de
liberar o conjunto bloqueado.

## 5. Avaliação no teste

Pendente

## 6. Análise de erros

Pendente

## 7. Conclusão

Pendente
