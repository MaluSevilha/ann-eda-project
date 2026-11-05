---
project: regression
dataset: https://www.kaggle.com/datasets/srisyra02/college-major-career-analytics-dataset
team:
  - Maria Luiza Sevilha
  - Vinicius Barreto Pomilio
ai_use: "OpenAI Codex e Anthropic ClaudeCode foram usados no apoio ao código e ao texto; a equipe revisou e é responsável por explicar todas as decisões."
---

# Projeto: Regressão com redes neurais

## 0. Ponto de partida

O objetivo é o definido na EDA: prever `median_earnings_4yr_usd`, a renda anual
mediana quatro anos depois da conclusão. O pré-processador é importado
diretamente de [`pipeline.py`](../eda/code/pipeline.py) e mantém as estratégias
justificadas na primeira entrega: clipping por IQR, imputação, indicadores de
ausência, padronização e one-hot encoding com agrupamento de categorias raras.

A métrica principal, registrada antes de qualquer treinamento, é o **MAE em
dólares**. A rede seria considerada bem-sucedida se alcançasse MAE pelo menos
**15% menor que o melhor baseline obrigatório** (`DummyRegressor` ou Ridge), isto
é, `MAE_MLP ≤ 0,85 × MAE_melhor_baseline`. RMSE e R² são métricas secundárias.

O split externo reproduz exatamente a EDA com `GroupShuffleSplit`, agrupado por
`opeid6`, `test_size=0.20` e seed 42. A validação sai apenas do treino da EDA,
novamente agrupada por instituição, com proporção 20% e seed 42:

| Conjunto | Programas | Instituições |
|---|---:|---:|
| Treino | 36.565 | 2.969 |
| Validação | 9.758 | 743 |
| Teste | 11.740 | 928 |

Nenhuma instituição aparece em mais de um conjunto. O pré-processador é ajustado
somente no treino e gera 420 features, sem `NaN`. Até a avaliação da seção 5, o
teste não foi transformado nem usado para estatísticas do alvo ou métricas.

A EDA terminava com 431 colunas porque ajustava o pipeline em um treino maior,
de 46.323 linhas. Com 36.565 linhas, 11 categorias ficam abaixo do mínimo de 20
ocorrências e passam ao grupo de raras.

O [`split_manifest.json`](split_manifest.json) registra a identidade de cada
conjunto, e o comando abaixo a confere:

```bash
python -m docs.projects.regression.code.data --check
```

## 1. Baselines

Os dois baselines obrigatórios usam as mesmas 420 features. Todos os números
desta seção são da **validação**.

O baseline trivial é `DummyRegressor(strategy="mean")`: ele prevê para todo
programa a renda média observada no treino e mede quanto se ganha apenas por sair
de uma constante. O baseline linear é Ridge, que combina as features por uma
função linear e penaliza coeficientes grandes. A grade de `alpha`, **0,001; 0,01;
0,1; 1; 10; 100; 1.000; 10.000**, foi declarada antes da execução, e a escolha é
feita pelo menor MAE.

--8<-- "docs/projects/regression/tables/baseline_validation.md"

O melhor baseline foi **Ridge com `alpha=1`**, com MAE de **US$ 7.814**, RMSE de
US$ 11.795 e R² de 0,816. O `DummyRegressor` teve MAE de US$ 20.024, então o Ridge
reduziu o erro em cerca de 61%. Uma combinação linear das features já explica
grande parte da variação, o que torna o Ridge uma referência exigente para a rede.

Ridge com `alpha=0,1` teve RMSE marginalmente menor, mas não venceu porque a
seleção foi fixada pelo MAE antes da execução. Na validação, o critério de 15%
corresponde a MAE de no máximo **US$ 6.642**. A decisão oficial aplica o mesmo
critério ao teste, na seção 5.

## 2. A rede

A rede de referência recebe as **420 features** e tem arquitetura **420 → 128 →
64 → 1**. As duas camadas ocultas usam ReLU, o que permite aprender relações não
lineares. A saída é uma única unidade linear, adequada à regressão porque não
limita a faixa das previsões.

As larguras 128 e 64 foram um ponto de partida escolhido antes da busca: camadas
que diminuem em direção à saída, sem ser tão estreitas a ponto de deixar a rede
quase linear nem tão largas a ponto de ter centenas de milhares de parâmetros já
no modelo de controle. A seção 4 testa versões mais estreita, mais larga e mais
profunda.

Os pesos das camadas ocultas usam inicialização de Kaiming, coerente com ReLU; a
saída usa Xavier, e todos os vieses começam em zero. A rede tem **62.209
parâmetros treináveis**, ou **1,70 parâmetro por linha** de treino. Com mais
parâmetros do que linhas, ela tem capacidade de memorizar parte dos dados, por
isso a seção 3 examina o overfitting.

Esta primeira rede é deliberadamente **sem regularização**: não usa dropout,
BatchNorm/LayerNorm nem weight decay. Ela é o controle das comparações da seção 4,
e não a configuração final.

## 3. Treinamento

O alvo recebe `log1p` para reduzir a assimetria observada na EDA e depois é
padronizado pela média (10,99) e pelo desvio (0,40) aprendidos somente no treino.
A rede otimiza Huber loss com `delta=1` nessa escala, e todas as previsões voltam
a dólares antes de calcular MAE, RMSE e R².

O treinamento usa Adam, cujas taxas adaptativas são uma referência robusta para
uma entrada que mistura numéricas padronizadas e muitas dummies esparsas. O
learning rate 0,001 é o ponto de partida; a seção 4 o compara com `3e-4`, valor
da configuração final.

O batch de **256** produz cerca de 143 atualizações por época, sem tornar cada
época excessivamente longa na CPU. O máximo de **200 épocas** é apenas um teto,
amplo o bastante para configurações com learning rate menor. Quem decide o número
efetivo de épocas é o early stopping, que monitora o MAE em dólares, com paciência
15 e melhora mínima de US$ 1, e restaura os melhores pesos ao final. A execução de
referência usa seed 42, CPU e operações determinísticas.

![Curvas de loss no treino e validação e MAE de validação por época](figures/fig01_training_curves.png)

A rede aprende rapidamente: o MAE cai de US$ 7.241 na época 1 para US$ 6.584 na
época 5. O melhor ponto é a **época 11**, com **MAE de US$ 6.566**, RMSE de US$
10.205 e R² de 0,863. A execução para na época 26 e restaura os pesos da época 11.

Depois da época 11, a loss de treino continua caindo, de 0,039 para 0,027, enquanto
a de validação sobe de 0,061 para 0,065 e o MAE oscila sem nova melhora. Essa
abertura entre as curvas é evidência de overfitting e justifica o early stopping e
a comparação de regularização da seção 4.

A menor Huber loss de validação ocorre na época 5, e o menor MAE, na época 11, com
apenas US$ 18 de diferença entre as duas. Como as medidas não apontam
necessariamente para o mesmo ponto, a restauração usa a métrica principal.

Na validação, esta referência reduz o MAE do Ridge em 15,97% e fica US$ 76 abaixo
do limite de US$ 6.642. Isso ainda não responde ao critério final, que depende do
teste.

## 4. Regularização e hiperparâmetros

A busca foi declarada antes da execução, com **12 configurações**, e manteve o
protocolo da referência: Huber, alvo em `log1p`, batch 256, no máximo 200 épocas,
early stopping, CPU e seed 42. Ela compara a rede sem regularização com dropout,
weight decay e LayerNorm, e também varia learning rate, largura e profundidade.
Todas as escolhas usam somente o MAE de validação.

--8<-- "docs/projects/regression/tables/hyperparameter_search.md"

![MAE das configurações e curvas das técnicas de regularização e da configuração vencedora](figures/fig02_regularization_search.png)

Todas as configurações terminaram sem falhas. Dropout foi a intervenção isolada
mais efetiva: com taxa 0,10, o MAE caiu de US$ 6.566 para **US$ 6.311**. Weight
decay teve efeito bem menor, e LayerNorm não ajudou: chegou a US$ 6.646 sem weight
decay, pior que a referência.

Mudar a capacidade também não trouxe ganho. As versões estreita (29.057
parâmetros), larga (140.801) e profunda (64.257) ficaram entre US$ 6.343 e US$
6.378, todas acima dos US$ 6.287 da arquitetura original com a mesma regularização.
Isso é coerente com o overfitting observado na referência, e não com falta de
parâmetros.

O vencedor foi **`low_lr_regularized`**: arquitetura 420 → 128 → 64 → 1, dropout
0,10, weight decay `1e-5` e learning rate `3e-4`. Ele alcançou **MAE de US$ 6.265**,
RMSE de US$ 9.821 e R² de 0,873 na época 31, parando na época 46. Isso reduz o MAE
da referência em 4,6% e o do Ridge em 19,8%.

A margem sobre o segundo colocado é pequena. `dropout_0.10_wd_1e-5`, que difere do
vencedor apenas no learning rate, ficou só US$ 22 acima em MAE e teve RMSE e R²
melhores. Cada configuração usou uma única seed, e o desvio entre seeds medido na
seção 5 foi de US$ 19. Além disso, uma reexecução da busca em outro ambiente manteve
o vencedor, mas moveu os MAEs em até cerca de US$ 36 e trocou o segundo e o terceiro
colocados. Portanto, posições tão próximas não indicam diferenças reais. Mantivemos
a regra registrada antes da execução, que escolhe o menor MAE.

As curvas mostram como o learning rate menor mudou o treinamento: o vencedor
convergiu mais lentamente e atingiu sua melhor época depois das outras
configurações. Mesmo com dropout, após a época 31 a loss de treino continuou caindo,
de 0,048 para 0,042, enquanto a de validação subiu de 0,056 para 0,057; portanto, o
early stopping ainda é necessário. LayerNorm oscilou mais e teve seu melhor MAE já
na época 5.

### Configuração congelada

A configuração vencedora foi registrada em [`final_config.json`](final_config.json)
antes de qualquer avaliação no teste, com arquitetura, transformação do alvo, loss,
otimizador, learning rate, batch, regularização, early stopping e seed-base 42. O
número de épocas não é fixo: cada treinamento restaura sua própria melhor época de
validação. Para o teste, a mesma configuração foi treinada com as seeds 42, 43 e 44,
sem nova seleção de hiperparâmetros.

## 5. Avaliação no teste

As três MLPs, o Ridge e o Dummy foram ajustados nas mesmas 36.565 linhas de treino,
e somente as MLPs usaram a validação para early stopping. Todos foram avaliados nos
mesmos 11.740 programas de teste.

Uma primeira avaliação apenas com a seed 42 foi produzida por engano. Para atender
à rubrica, o teste precisou ser reaberto uma segunda vez, sem alterar configuração,
pipeline ou critério após o primeiro resultado. Esse histórico está registrado em
[`final_test_results.json`](final_test_results.json) como `test_access_count: 2`.

--8<-- "docs/projects/regression/tables/final_test_metrics.md"

O resultado agregado, em média ± desvio amostral das três seeds, foi **MAE de US$
6.573 ± 19**, RMSE de US$ 11.152 ± 87 e R² de 0,847 ± 0,002. As redes pararam nas
épocas 46, 42 e 51, com melhores pesos nas épocas 31, 27 e 36.

O melhor baseline continuou sendo o Ridge, com MAE de **US$ 8.035**. O limite de 15%
era US$ 6.830: a MLP ficou US$ 256 abaixo dele e reduziu o erro do Ridge em
**18,19%**, então **o critério foi atingido**. A diferença de US$ 1.462 para o Ridge
é muito maior que o desvio de US$ 19 entre seeds. Esse desvio, porém, mede apenas a
variação entre inicializações, e não a incerteza por termos uma única amostra de
instituições no teste.

O erro no teste ficou acima do observado na validação. Na seed 42, o MAE subiu de
US$ 6.265 para US$ 6.590 (+5,2%) e o RMSE, de US$ 9.821 para US$ 11.215 (+14,2%).
Parte disso é esperada, porque a validação serviu para escolher a configuração e a
época de parada. Mas o teste também parece mais difícil por si só: o Dummy, que não
depende dessas escolhas, teve altas de 1,5% no MAE e 3,5% no RMSE. Esses dados não
permitem separar as duas causas, e a estimativa de referência é a do teste.

![Previsões da MLP comparadas aos valores reais no teste](figures/fig03_predicted_vs_actual.png)

Os gráficos usam a seed 42, fixada previamente como referência para a análise por
programa. A maior parte dos programas fica ao redor da diagonal, mas há compressão
nas rendas mais altas: vários valores reais na cauda recebem previsões menores. Isso
é compatível com regressão em direção à média e ajuda a explicar por que o RMSE é
bem maior que o MAE.

![Resíduos da MLP em função das previsões no teste](figures/fig04_residuals_vs_predicted.png)

Os resíduos se espalham mais à medida que a renda prevista cresce. Há subestimações
grandes entre os programas de maior renda, embora também existam superestimações
relevantes. Essa heterocedasticidade motiva a análise por faixa do alvo e por
categoria da seção 6.

## 6. Análise de erros

Esta seção não reabre o teste: os erros usam a **seed 42 como referência fixa** e
vêm das previsões finais já registradas. Para não escolher faixas depois de ver os
resultados, usamos os quartis do alvo no treino registrados na EDA: US$ 46.709, US$
58.878 e US$ 77.121. Grupos com menos de 100 programas são marcados e não sustentam
conclusões fortes isoladamente.

### Subgrupos e faixas do alvo

--8<-- "docs/projects/regression/tables/error_by_credential.md"

Os maiores MAEs aparecem em programas sem credencial (US$ 16.521; n=45) e em
certificados de pós-graduação (US$ 14.612; n=92), ambos grupos pequenos. No primeiro
caso há uma explicação possível: no treino, nenhum desses programas tem renda de 1
ano, o sinal mais forte do modelo. O resultado mais robusto é o de doutorado, com
**MAE de US$ 13.942 em 257 programas**. Doutorados e certificados de pós-graduação
têm rendas altas, com medianas de cerca de US$ 101 mil e US$ 88 mil no teste, e
resíduo médio positivo: a rede tende a subestimá-los.

--8<-- "docs/projects/regression/tables/error_by_institution_control.md"

Instituições privadas sem fins lucrativos têm o maior MAE: US$ 8.261, contra US$
6.168 nas públicas e US$ 4.868 nas privadas com fins lucrativos. A composição explica
parte disso, já que no treino 33% de seus programas são de pós-graduação, contra 18%
e 8%, e 35% ficam acima de US$ 77.121, contra 23% e 10%. Mas não explica tudo: o erro
delas continua maior mesmo separando pós-graduação (US$ 11.342, contra US$ 8.297 e
US$ 7.013) das demais credenciais (US$ 6.735, contra US$ 5.600 e US$ 4.522).

--8<-- "docs/projects/regression/tables/error_by_target_band.md"

A faixa acima de US$ 77.121 concentra o maior erro: **MAE de US$ 11.140** e resíduo
médio de +US$ 6.324, isto é, subestimação. Na faixa mais baixa, o resíduo médio é de
−US$ 3.138, indicando superestimação. Essa mudança de sinal é o padrão esperado de
regressão em direção à média.

![MAE da MLP por credencial, controle e faixa do alvo](figures/fig05_error_subgroups.png)

### Piores exemplos

--8<-- "docs/projects/regression/tables/worst_test_errors.md"

Os cinco maiores erros são subestimações de programas com renda real acima de US$ 303
mil e previsões entre cerca de US$ 132 mil e US$ 170 mil. Eles incluem doutorado,
mestrado e certificado profissional, em instituições públicas e privadas sem fins
lucrativos, então o padrão não parece ser de um único tipo institucional.

São casos extremos em uma cauda pouco representada: só 104 das 36.565 linhas de
treino passam de US$ 200 mil, e três desses cinco programas têm renda acima do maior
valor visto no treino e na validação (US$ 316.195). Treinar em `log1p` com Huber loss
pode contribuir para essa compressão, mas não isolamos esse efeito.

### Importância por permutação

A importância foi calculada na **validação**, com três permutações por feature
original. Cada categórica é embaralhada antes do pipeline, mantendo suas dummies como
um grupo, e cada numérica carrega junto seu indicador de ausência. Os valores mostram
quanto o MAE aumenta ao destruir a informação da feature.

![Importância agrupada das quinze principais features](figures/fig06_permutation_importance.png)

--8<-- "docs/projects/regression/tables/permutation_importance.md"

`median_earnings_1yr_usd` domina, com aumento médio de **US$ 11.670** no MAE. Depois
vêm credencial (US$ 2.049), curso (US$ 1.268), tamanho da coorte de um ano (US$ 1.169)
e salário mediano da ocupação (US$ 965). O padrão é coerente com a EDA, em que a renda
de um ano tinha Spearman ρ = 0,887 com o alvo, enquanto curso, credencial e mercado de
trabalho traziam sinais secundários. Controle institucional é bem menos importante
(US$ 175).

O tamanho da coorte é a exceção. Sua correlação com o alvo é fraca (ρ = 0,08 no
treino), mas ele fica ausente exatamente nas mesmas linhas de treino que a renda de um
ano. Ao embaralhá-lo, embaralhamos também um indicador de que essa renda foi imputada,
o que pode explicar parte da sua importância. Não testamos essa hipótese isoladamente.

### Ablação da renda de primeiro ano

--8<-- "docs/projects/regression/tables/earnings_1yr_ablation.md"

Sem `median_earnings_1yr_usd` e seu indicador de ausência, a entrada cai de 420 para
418 features e o MAE de validação sobe de US$ 6.265 para **US$ 7.868**, piora de
**25,59%**. A rede ainda aprende com curso, credencial, instituição e ocupação, mas
perde uma parte substancial da precisão.

O aumento na permutação é muito maior que o da ablação porque embaralhar a renda cria
combinações artificiais com as outras features, enquanto remover a coluna permite que
o modelo se reajuste. Assim, a ablação estima melhor o custo de operar sem essa
informação, e a permutação mede sua relevância no modelo já treinado. Nenhuma das duas
é um efeito causal.

### Controle do alvo em `log1p`

O Ridge oficial aprende em dólares com erro quadrático, enquanto a MLP aprende em
`log1p` com Huber loss. Para estimar quanto do ganho vem apenas da transformação do
alvo, repetimos a grade de `alpha` do Ridge com o mesmo `log1p` padronizado da MLP,
escolhendo pelo MAE em dólares. Esse controle foi feito depois da avaliação final, usa
somente treino e validação e não altera o baseline oficial nem o critério de sucesso.

--8<-- "docs/projects/regression/tables/target_control_validation.md"

O melhor Ridge em `log1p` reduz o MAE em US$ 155, cerca de **10%** dos US$ 1.549 que
separam o Ridge oficial da MLP na validação. Mesmo contra esse controle, a MLP tem MAE
18,2% menor. O RMSE do controle piora, de US$ 11.795 para US$ 12.022: o `log1p` ajuda
no erro típico, mas custa nos erros grandes.

O controle tem dois limites: continua usando erro quadrático, portanto não isola o
efeito da Huber loss, e a MLP comparada usa uma única seed.

## 7. Conclusão

Dentro desta tarefa, **a MLP valeu a pena em precisão preditiva**. Seu MAE médio de
teste foi de **US$ 6.573 ± 19**, contra **US$ 8.035** do Ridge: redução de **18,19%**,
superior ao critério de 15% registrado antes do treinamento. O controle da seção 6
indica que treinar em `log1p` responde por apenas cerca de 10% dessa vantagem na
validação. O restante é compatível com relações não lineares entre renda anterior,
curso, credencial, instituição e ocupação que o baseline linear não captura por
completo, mas também pode refletir a Huber loss, que o controle não isola.

Esse resultado não autoriza usar a previsão como garantia de renda individual nem como
ranking automático de cursos. A unidade do modelo é um programa e o alvo é uma mediana
de coorte; trajetórias individuais dependem de região, experiência, setor, condições
econômicas e muitas informações ausentes da base. Uma previsão errada poderia induzir
estudantes a escolher ou abandonar um curso e levar gestores a distribuir recursos de
modo inadequado. Portanto, o modelo só deve apoiar análises agregadas, acompanhado dos
dados observados, de revisão humana e de monitoramento dos erros por subgrupo.

As limitações mais importantes são:

- somente três seeds foram avaliadas, o mínimo da rubrica, então a estimativa de
  variabilidade ainda tem baixa precisão;
- a correção multiseed exigiu reabrir o teste após a primeira avaliação de seed única;
  nenhum hiperparâmetro foi alterado depois desse primeiro acesso, mas o protocolo
  ideal de acesso único não foi cumprido;
- os números exatos de desenvolvimento dependem do ambiente: em outro ambiente, os MAEs
  de validação da busca variaram até cerca de US$ 36, e a melhor época da rede de
  referência mudou, embora o vencedor tenha se mantido;
- **74,51%** dos alvos da base original estavam suprimidos, principalmente por regras
  de privacidade ligadas ao tamanho da coorte; o modelo representa programas com renda
  divulgada e pode não generalizar para programas pequenos ou suprimidos;
- a rede subestima a cauda de renda e não fornece intervalo de incerteza;
- a precisão depende fortemente da renda de primeiro ano, informação que só existe
  depois da conclusão e limita usos prospectivos no momento da matrícula;
- a avaliação usa um único recorte temporal e não mede mudanças do mercado de trabalho;
  além disso, importância e correlação são associações, não efeitos causais.

Com mais tempo, executaríamos mais seeds, adicionaríamos intervalos por regressão
quantílica ou ensembles, trataríamos explicitamente a cauda, avaliaríamos em dados
novos uma versão sem renda de primeiro ano, hoje medida só na validação, e faríamos
validação temporal externa. Antes de qualquer uso real, também seriam necessárias
auditorias periódicas de calibração e de erro por credencial, controle institucional e
faixa de renda.

### Resumo dos resultados

| # | Resumo | Valor |
|---:|---|---|
| 1 | Dataset, tarefa, alvo e métrica principal | College Majors 2026; regressão de `median_earnings_4yr_usd`; MAE em dólares |
| 2 | Treino · validação · teste (linhas) | 36.565 · 9.758 · 11.740 |
| 3 | Melhor baseline e sua métrica no teste | Ridge `alpha=1`; MAE US$ 8.035,06 |
| 4 | Rede escolhida: camadas × unidades, parâmetros e regularização | 420→128→64→1; 62.209 parâmetros; dropout 0,10 e weight decay `1e-5` |
| 5 | Otimizador, learning rate, batch e época de parada | Adam; `3e-4`; batch 256; paradas 46 · 42 · 51 nas seeds 42 · 43 · 44 |
| 6 | Métrica principal da rede no teste | MAE US$ 6.573,47 ± US$ 18,90, média ± desvio amostral de 3 seeds |
| 7 | Critério de sucesso atingido? | Sim; MAE médio 18,19% menor que o Ridge, acima dos 15% exigidos |
| 8 | Subgrupo com maior erro | Programas sem credencial; MAE US$ 16.521,33; `n=45`, grupo pequeno |
