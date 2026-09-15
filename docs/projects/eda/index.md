---
project: eda
task: regression
dataset: https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai
team:
  - PREENCHER NOME COMPLETO DO INTEGRANTE 1
  - PREENCHER NOME COMPLETO DO INTEGRANTE 2
ai_use: "OpenAI Codex foi usado na implementação do código, geração das figuras, auditoria dos resultados e primeira versão do texto; a equipe revisou e é responsável por explicar todas as decisões."
---

# Projeto — Análise Exploratória de Dados

## 0. Proposta

Este projeto prepara uma tarefa de **regressão** para estimar
`median_earnings_4yr_usd`: a renda anual mediana, em dólares, de ex-alunos de
um programa quatro anos após a matrícula. A unidade de observação é um programa
— combinação de instituição, área CIP e nível de credencial — e não uma pessoa.

O conjunto público
[College Majors 2026: Earnings, Debt, Jobs, AI](https://www.kaggle.com/datasets/kylefengkfeng209/college-majors-2026-earnings-debt-jobs-ai)
tem **227.980 × 72**. Ele é uma base pré-integrada de dados do College
Scorecard, IPEDS, BLS e O\*NET. A motivação é investigar quanto atributos do
curso, da instituição, da ocupação associada e da exposição a tecnologias
ajudam a antecipar o resultado financeiro. O primeiro risco é a disponibilidade
seletiva do alvo: **169.868 linhas (74,51%)** foram suprimidas, principalmente
por privacidade de coortes pequenas.

O escopo desta entrega termina no pré-processamento. **Nenhum modelo foi
treinado**, conforme o enunciado.

## 1. Inspeção inicial

### A — Dicionário de dados

Cada linha representa uma instituição × curso CIP de quatro dígitos × nível de
credencial. `program_id` é chave: há **227.980 valores únicos**, sem ausências.
O dicionário abaixo cobre as 72 colunas do arquivo. Algumas definições são
inferidas a partir dos nomes e fontes, pois o arquivo do Kaggle não traz um
codebook oficial completo; por isso, análises causais estão fora do escopo.

--8<-- "docs/projects/eda/tables/data_dictionary.md"

### B — Qualidade

Não há linhas totalmente duplicadas nem `program_id` repetido. Entretanto, a
ausência é extensa e estruturada: os campos de renda e dívida usam flags como
`privacy_suppressed` e `not_available`, portanto não é correto supor MCAR. A
coluna com mais ausências é `pct_working_in_state_5yr`, com **188.983 valores
ausentes (82,895%)**. As maiores taxas são:

| Coluna | Ausentes | % |
|---|---:|---:|
| `pct_working_in_state_5yr` | 188.983 | 82,895 |
| `count_working_in_state_5yr` | 188.983 | 82,895 |
| `earnings_trajectory_category` | 188.859 | 82,840 |
| `earnings_growth_pct_1yr_to_5yr` | 188.859 | 82,840 |
| `payment_to_income_pct_1yr` | 184.672 | 81,004 |
| `debt_to_earnings_1yr` | 184.672 | 81,004 |
| `debt_to_earnings_4yr` | 184.605 | 80,974 |
| `median_debt_usd` | 181.307 | 79,528 |

[Tabela completa de valores ausentes](tables/missing_values.csv).

Também foram quantificadas inconsistências: **427** percentuais acima da
referência de ensino médio e **145** percentuais de trabalho no estado excedem
100%; há **89** anuidades in-state iguais a zero, **215** crescimentos de renda
acima de 200% e **1.700** registros com código de credencial 99, que não pertence
à escala ordinal 1–8. Esses campos inconsistentes são descartados ou tratados
como categóricos, nunca usados como números contínuos sem correção.

As exclusões para a futura modelagem são:

- IDs e proxies de ID: `program_id`, `unitid`, `opeid6`, `institution_name` e
  `institution_city`; `opeid6` permanece apenas para formar os grupos do split.
- Redundâncias: códigos quando o rótulo correspondente já existe, região quando
  estado já existe, contagens duplicadas de ocupações e pagamento mensal quando
  a dívida já está presente.
- Vazamento direto: `earnings_vs_national_pct`, `debt_to_earnings_4yr`, os três
  benchmarks nacionais de renda de quatro anos e a coorte medida com o alvo.
- Informação futura: todas as variáveis de cinco anos e a trajetória 1º–5º ano.
- Texto livre: `ai_tool_examples`, que não é uma feature tabular estável.

A justificativa de cada coluna está registrada em
[`DROPPED_COLUMNS`](code/pipeline.py). Permanecem **33 features**: 26 numéricas e
7 categóricas.

### C — Alvo de regressão

Há **58.112** alvos divulgados no arquivo. Para a análise principal retiramos
49 programas estrangeiros com alvo, porque formam uma população distinta e não
têm os mesmos atributos geográficos; restam **58.063** linhas. A média é
**US$ 64.887**, a mediana **US$ 59.011**, o desvio-padrão **US$ 27.704**, e a
faixa vai de **US$ 6.917** a **US$ 336.392**. A assimetria de Fisher é **1,755**.

![Histograma e boxplot do alvo](figures/fig01_target.png)

**Conclusão da Figura 1.** O alvo tem cauda longa à direita e muitos valores
extremos; a média fica US$ 5.876 acima da mediana. Na futura rede, será sensato
comparar a escala original com `log1p(y)`, avaliando os resultados também em
dólares para preservar interpretação.

### D — Treino e teste

Foi usado `GroupShuffleSplit(test_size=0.20, random_state=42)`, agrupado por
`opeid6`. O treino contém **46.323 programas de 3.712 instituições**, e o teste,
**11.740 programas de 928 instituições**. A interseção de instituições é
**zero**. Esse critério é mais conservador do que um split aleatório por linha:
atributos repetidos de uma instituição não atravessam a fronteira de teste.
Daqui em diante, estatísticas de imputação, limites de outliers, escalas,
categorias e projeções são ajustados apenas no treino.

## 2. Análise univariada

### A — Numéricas

A tabela completa abaixo apresenta contagem, ausência, média, mediana, desvio,
quartis, extremos e assimetria para todas as variáveis numéricas mantidas e o
alvo, sempre no treino.

--8<-- "docs/projects/eda/tables/numeric_summary_train.md"

![Histogramas das variáveis numéricas](figures/fig02_numeric_distributions.png)

**Conclusão da Figura 2.** Escalas e formas são muito diferentes: SAT se
aproxima de uma distribuição unimodal, enquanto matrículas, prêmios, dívida,
coortes, vagas ocupacionais e indicadores de IA são assimétricos à direita.
`awards_year1` chega a 8.584 contra mediana 33; `institution_undergrad_enrollment`
chega a 163.164 contra mediana 5.011,5. Isso sustenta clipping robusto e
padronização, ambos aprendidos apenas no treino.

### B — Categóricas

--8<-- "docs/projects/eda/tables/categorical_summary_train.md"

`cip_title` tem **350 níveis** no treino, dos quais **142** aparecem menos de 20
vezes; `largest_linked_occupation` tem 127 níveis e 32 raros. A categoria modal
de ocupação é “Managers, all other” (13,06%), um rótulo residual que limita a
interpretação. As demais features têm cardinalidade moderada.

![Frequências das categorias](figures/fig03_categorical_frequencies.png)

**Conclusão da Figura 3.** Há forte concentração em instituições públicas,
bacharelados e cursos sem credencial totalmente online, além de caudas longas
nos títulos CIP e ocupações. O one-hot deve agrupar níveis raros e aceitar
categorias novas para não criar milhares de colunas frágeis.

## 3. Análise bivariada e multivariada

### A — Numérica × numérica

Foi usada correlação de **Spearman**, adequada às caudas, outliers e relações
monotônicas não necessariamente lineares observadas na Figura 2. A matriz
completa está em [CSV](tables/spearman_correlation_train.csv). O par numérico
mais redundante é emprego total da ocupação × aberturas anuais, com
**ρ = 0,976**. Em relação ao alvo, os maiores módulos são renda no primeiro ano
(**ρ = 0,887**), salário mediano da ocupação (0,469), dívida mediana (0,404),
anuidade out-state (0,398) e tecnologias emergentes (0,339).

![Matriz de correlação](figures/fig04_spearman_heatmap.png)

**Conclusão da Figura 4.** O sinal preditivo é dominado pelo resultado financeiro
anterior; a redundância entre emprego e vagas recomenda regularização na rede e
uma análise de ablação, sem remover automaticamente informação potencialmente
útil.

![Dispersões com o alvo](figures/fig05_numeric_target_scatter.png)

**Conclusão da Figura 5.** A renda do primeiro ano segue relação quase linear
com o alvo, enquanto o salário ocupacional tem relação positiva mais dispersa
e em faixas verticais, pois o mesmo valor de BLS é replicado entre muitos
programas ligados à mesma ocupação.

### B — Categórica × alvo

![Alvo por credencial e controle](figures/fig06_categories_target.png)

**Conclusão da Figura 6.** A posição muda fortemente entre grupos: a mediana vai
de **US$ 39.665** em certificados de graduação a **US$ 112.207** em primeiros
diplomas profissionais. Por controle, privadas sem fins lucrativos têm mediana
US$ 65.951, públicas US$ 58.592 e privadas com fins lucrativos US$ 40.451.
Essas são associações descritivas; composição de cursos e seleção de estudantes
impedem leitura causal.

![Alvo por família de curso](figures/fig07_cip_family_target.png)

**Conclusão da Figura 7.** Entre as famílias frequentes, Engenharia tem mediana
de **US$ 95.085**, seguida por Computação com **US$ 78.969**; artes, serviços
pessoais e humanidades aparecem abaixo. O campo de estudo é uma fonte central
de heterogeneidade e deve permanecer no modelo, com categorias raras agrupadas.

### C — Numérica × categórica

![Numéricas por categorias](figures/fig08_numeric_categorical.png)

**Conclusão da Figura 8.** Privadas sem fins lucrativos apresentam anuidades
in-state mais altas e dispersas. O salário da ocupação associada também muda em
posição e espalhamento conforme a credencial, sobretudo nos diplomas
profissionais. Padronizar é necessário, mas não elimina essas diferenças
estruturais entre grupos.

## 4. Pré-processamento

### A — Estratégias

As escolhas abaixo respondem diretamente à auditoria:

1. **Faltantes.** Numéricas recebem a mediana do treino e um indicador de
   ausência; categóricas recebem a moda do treino. Isso preserva informação do
   mecanismo de supressão. O alvo ausente não é imputado: essas linhas não
   pertencem à aprendizagem supervisionada.
2. **Outliers.** Cada numérica com IQR positivo é winsorizada nos limites
   Q1 − 1,5×IQR e Q3 + 1,5×IQR aprendidos no treino. A regra afeta ao menos uma
   feature em **26.006 linhas (56,14%)**; nenhuma linha é removida. Flags e
   variáveis zero-infladas com IQR zero são preservadas.
3. **Categóricas.** `OneHotEncoder(min_frequency=20,
   handle_unknown="infrequent_if_exist")` agrupa níveis raros e transforma sem
   erro uma categoria inédita criada no teste de integração.
4. **Escala.** `StandardScaler` é ajustado depois do clipping e da imputação,
   necessário porque as numéricas vão de proporções 0–1 a centenas de milhares
   de dólares ou estudantes. O destino é uma rede neural sensível à escala.

### B — Redução de dimensionalidade

As projeções usam uma amostra reproduzível de 3.000 linhas do treino já
pré-processado. A cor representa o alvo apenas para interpretação; o alvo não
entra nos algoritmos.

![PCA](figures/fig09_pca.png)

**Conclusão da Figura 9.** PC1+PC2 explicam apenas **24,92%** da variância. Os
maiores loadings pertencem sobretudo a indicadores de ausência de dados
ocupacionais/IA e de anuidade, indicando que padrões de cobertura estruturam o
espaço. A PCA 2D perde a maior parte da informação e não separa o alvo de forma
limpa.

![t-SNE](figures/fig10_tsne.png)

**Conclusão da Figura 10.** t-SNE revela ilhas locais que a PCA comprime, mas o
arranjo muda entre perplexidades 30 e 50 e as cores ainda se sobrepõem. Tamanho
de grupos e distância entre ilhas não representam frequência nem distância
global.

![UMAP](figures/fig11_umap.png)

**Conclusão da Figura 11.** UMAP também mostra componentes desconectados ligados
a combinações categóricas e padrões de ausência. A mudança de 15 para 50
vizinhos altera a geometria, mas preserva a ausência de uma fronteira simples
de renda; distâncias entre grupos não têm interpretação direta.

Em conjunto, t-SNE e UMAP evidenciam estrutura local não linear ausente na PCA,
mas não justificam tratar as ilhas como clusters reais. Para regressão, as
projeções servem ao diagnóstico; a rede receberá a matriz pré-processada
completa.

### C — Pipeline

O arquivo importável [`pipeline.py`](code/pipeline.py) implementa
`Pipeline` + `ColumnTransformer`. Ajustado só no treino, ele produz matrizes
**(46.323, 431)** e **(11.740, 431)**. Há **zero `NaN`** nas duas matrizes, os
431 nomes estão em [pipeline_feature_names.csv](tables/pipeline_feature_names.csv)
e o teste de categoria inédita preserva a mesma largura.

O script completo que reproduz split, tabelas, figuras, projeções e validações
está em [`analysis.py`](code/analysis.py).

## 5. Síntese

Os principais achados para a próxima entrega são:

- A renda de quatro anos é assimétrica (Figura 1; skew 1,755), então a futura
  modelagem deve comparar alvo original e log-transformado sem esconder erros
  absolutos em dólares.
- A renda do primeiro ano é o sinal numérico dominante (Figura 5; ρ = 0,887),
  mas falta em 18,23% do treino. Uma ablação sem essa variável medirá o quanto o
  modelo depende de um resultado financeiro anterior.
- Curso, credencial e controle institucional deslocam localização e dispersão
  do alvo (Figuras 6–8). Encoding raro e regularização serão essenciais.
- PC1+PC2 retêm só 24,92%, enquanto projeções não lineares mostram ilhas
  instáveis (Figuras 9–11). Não há suporte para reduzir a futura entrada a duas
  dimensões.

Os riscos de modelagem e planos de tratamento são: (i) **viés de seleção**, pois
74,51% dos alvos foram suprimidos — descrever o domínio como programas com
coortes divulgáveis e comparar perfis com/sem alvo; (ii) **vazamento** — manter a
lista de exclusão e o split por instituição; (iii) **outliers e cauda do alvo**
— clipping somente em X, comparação de `log1p(y)` e métricas MAE/RMSE; (iv)
**generalização a novas categorias** — agrupamento de raras e unknown-safe;
(v) **medidas ocupacionais nacionais e repetidas** — evitar interpretação causal
e validar erros por curso, credencial e controle institucional.

| # | Resumo dos resultados | Valor |
|---:|---|---|
| 1 | Dataset, tarefa e alvo | College Majors 2026; regressão; `median_earnings_4yr_usd` |
| 2 | Instâncias × features (numéricas / categóricas) | 227.980 × 72 no bruto; 58.063 linhas modeláveis; 33 features (26 / 7) |
| 3 | Coluna com mais ausentes e percentual | `pct_working_in_state_5yr`: 188.983 (82,895%) |
| 4 | Colunas descartadas e motivo | IDs/alta cardinalidade; redundâncias; vazamento do alvo; informação de 5 anos; texto livre — detalhadas em `pipeline.py` |
| 5 | Média e mediana do alvo | US$ 64.887 e US$ 59.011 |
| 6 | Tamanho de treino e teste | 46.323 e 11.740; 0 instituições em comum |
| 7 | Par numérico mais correlacionado | emprego da ocupação × aberturas anuais: Spearman ρ = 0,976 |
| 8 | Linhas afetadas pela estratégia de outlier | 26.006 no treino (56,14%); winsorizadas, não removidas |
| 9 | Variância explicada por PC1 + PC2 | 24,92% |
| 10 | `shape` após o pipeline | treino (46.323, 431); teste (11.740, 431); 0 `NaN` |

