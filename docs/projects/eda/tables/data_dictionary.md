| Coluna | Grupo | Tipo | O que significa | % ausente (treino) |
|---|---|---|---|---:|
| `median_earnings_4yr_usd` | **Alvo** | numérica (US$) | Renda anual mediana de quem se formou, medida 4 anos depois. É o que o modelo vai tentar prever. | 0 |
| `is_main_campus` | Instituição | binária (0/1) | Se a linha é o campus principal (1) ou uma unidade satélite (0). | 0 |
| `institution_latitude` | Instituição | numérica | Latitude da instituição. | 4,6 |
| `institution_longitude` | Instituição | numérica | Longitude da instituição. | 4,6 |
| `institution_is_hbcu` | Instituição | binária (0/1) | Se é uma "Historically Black College/University". | 4,6 |
| `institution_admission_rate` | Instituição | numérica (0–1) | Taxa de aceitação de candidatos. | 36,4 |
| `institution_avg_sat` | Instituição | numérica | Nota média do SAT dos aprovados. | 51,6 |
| `institution_undergrad_enrollment` | Instituição | numérica | Número de alunos de graduação matriculados. | 5,2 |
| `institution_tuition_in_state_usd` | Instituição | numérica (US$) | Mensalidade anual para quem mora no mesmo estado. | 13,5 |
| `institution_tuition_out_state_usd` | Instituição | numérica (US$) | Mensalidade anual para quem mora fora do estado. | 13,5 |
| `institution_control` | Instituição | categórica | Tipo de gestão: pública, privada sem fins lucrativos ou privada com fins lucrativos. | 0 |
| `institution_state` | Instituição | categórica | Sigla do estado (55 valores possíveis). | 4,6 |
| `awards_year1` | Curso | numérica | Diplomas concedidos pelo curso no 1º ano medido. | 6,7 |
| `awards_year2` | Curso | numérica | Diplomas concedidos pelo curso no 2º ano medido. | 8,2 |
| `outcomes_shared_across_campuses` | Curso | binária (0/1) | Se o resultado divulgado é compartilhado entre vários campi da mesma instituição, e não exclusivo desta linha. | 0 |
| `cip_title` | Curso | categórica | Nome do curso (código CIP de 4 dígitos). 350 valores possíveis, muitos raros. | 0 |
| `credential_name` | Curso | categórica | Nível do diploma: certificado, associado, bacharelado, mestrado etc. | 0 |
| `distance_education` | Curso | categórica | Modalidade: presencial, todo online ou mista. | 0 |
| `median_earnings_1yr_usd` | Financeiro | numérica (US$) | Renda mediana de quem se formou, medida **1 ano** depois (não 4). É o sinal mais forte ligado ao alvo (ρ = 0,887), mas falta em ~18% do treino. | 18,2 |
| `earnings_cohort_size_1yr` | Financeiro | numérica | Quantos ex-alunos entraram na conta de `median_earnings_1yr_usd`. | 18,2 |
| `median_debt_usd` | Financeiro | numérica (US$) | Dívida estudantil mediana dos formados. | 25,4 |
| `debt_borrower_count` | Financeiro | numérica | Quantos formados tinham empréstimo estudantil. | 20,5 |
| `linked_occupations_count` | Ocupação | numérica | Quantas ocupações (O\*NET) estão associadas a este curso. | 0,0 |
| `largest_linked_occupation` | Ocupação | categórica | Nome da ocupação mais associada ao curso. | 1,5 |
| `occupation_typical_entry_education` | Ocupação | categórica | Escolaridade normalmente exigida para entrar nessa ocupação. | 1,5 |
| `occupation_employment_2024_thousands` | Ocupação | numérica (milhares) | Total de pessoas empregadas nessa ocupação em 2024 (BLS). | 1,5 |
| `occupation_growth_pct_2024_34` | Ocupação | numérica (%) | Crescimento projetado do emprego entre 2024 e 2034. | 1,5 |
| `occupation_growth_pct_max` | Ocupação | numérica (%) | Maior crescimento entre as ocupações associadas ao curso. | 1,5 |
| `occupation_annual_openings_thousands` | Ocupação | numérica (milhares) | Vagas abertas por ano, projeção BLS. | 1,5 |
| `occupation_median_wage_2024_usd` | Ocupação | numérica (US$) | Salário mediano da ocupação em 2024 (BLS), não do curso. | 1,5 |
| `ai_software_occupation_share` | IA / Tecnologia | numérica (0–1) | Fração das ocupações ligadas ao curso que já usam software de IA no trabalho. | 4,2 |
| `expert_system_occupation_share` | IA / Tecnologia | numérica (0–1) | Fração que usa sistemas especialistas. | 4,2 |
| `ai_tools_max_per_occupation` | IA / Tecnologia | numérica | Maior número de ferramentas de IA citadas numa ocupação ligada. | 4,2 |
| `hot_technologies_mean_per_occupation` | IA / Tecnologia | numérica | Média de "tecnologias em alta" (O\*NET) por ocupação ligada. | 4,2 |
| `opeid6` | Identificador | categórica | Código da instituição. Não entra como feature: serve só para agrupar o split treino/teste, evitando que a mesma instituição apareça nos dois lados. | 0 |

**Como ler:** cada linha do dataset é um programa, a combinação de uma instituição, um curso (CIP) e um nível de credencial, não uma pessoa. "% ausente" é calculado só dentro do treino (46.323 linhas), depois de já remover os programas sem alvo.

Esta tabela cobre as 33 colunas usadas como *features* + o alvo + o identificador de agrupamento (`opeid6`). O arquivo bruto tem 72 colunas; as demais foram descartadas, ver a tabela de exclusões na seção 1-B.
