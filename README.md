# College Majors 2026 — EDA e regressão

Projetos da disciplina de Redes Neurais e Deep Learning. O objetivo é estimar
`median_earnings_4yr_usd`, a renda mediana quatro anos após a conclusão, sem usar
variáveis derivadas do alvo nem informações posteriores.

Os relatórios ficam em `docs/projects/eda/index.md` e
`docs/projects/regression/index.md`.

## Instalação e dados

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
python docs/projects/eda/code/analysis.py
```

O download grava o arquivo bruto em `data/raw/college_majors_2026.csv`, que não é
versionado por exceder 100 MB. A análise da EDA regenera suas tabelas e figuras a
partir desse arquivo.

Em Linux, o arquivo de dependências instala a distribuição CPU do PyTorch para
evitar vários gigabytes de bibliotecas CUDA. Uma instalação com GPU pode
substituí-la pela distribuição indicada para a versão de CUDA da máquina.

## Verificação da regressão

```bash
python -m docs.projects.regression.code.data --check
python -m docs.projects.regression.code.baselines --check
python -m docs.projects.regression.code.train --check
python -m docs.projects.regression.code.search --check
python -m docs.projects.regression.code.final_config --check
python -m docs.projects.regression.code.final_evaluation --check
python -m docs.projects.regression.code.error_analysis --check
python -m docs.projects.regression.code.target_control --check
pytest -q
```

| Módulo | O que confere | Retreina? |
|---|---|---|
| `data` | Splits e manifesto, sem transformar o teste | Não |
| `baselines` | Dummy e grade do Ridge, em treino e validação | Sim |
| `train` | MLP de referência, em treino e validação | Sim |
| `search` | As 12 configurações da busca, em treino e validação | Sim |
| `final_config` | Correspondência entre a configuração congelada e a busca | Não |
| `final_evaluation` | Métricas, previsões e hashes das seeds 42, 43 e 44, sem reabrir o teste | Não |
| `error_analysis` | Erros das previsões finais; permutação e ablação na validação | Sim |
| `target_control` | Ridge com alvo em `log1p`, em treino e validação | Sim |

Os módulos que retreinam podem levar alguns minutos em CPU.

### Reprodução em outros ambientes

`baselines`, `train`, `search` e `error_analysis` comparam valores exatos e podem
falhar fora do ambiente que gerou os artefatos, assim como o teste de `pytest` que
usa `baselines`. Em uma instalação limpa com Python 3.10 e as versões de
`requirements.txt`, os baselines variaram até US$ 0,12 e os MAEs da busca, até cerca
de US$ 36. A melhor época da rede de referência passou de 11 para 5, mas o vencedor
da busca se manteve.

Essa divergência não invalida os resultados de teste, que `final_evaluation --check`
confere a partir dos artefatos. Não regenere os artefatos apenas para fazer as
verificações passarem: o relatório e os resultados de teste dependem dos valores
versionados.

### Regenerar artefatos

Após uma mudança deliberada, use `--write-manifest` (`data`), `--write-results`
(`baselines`, `search` e `target_control`), `--write-reference` (`train`),
`--write-config` (`final_config`) e `--write-analysis` (`error_analysis`).

Não execute novamente `final_evaluation --write-results`: o teste já foi aberto duas
vezes, primeiro com seed única e depois na correção multiseed, e o código bloqueia
novas sobrescritas.

## Documentação

```bash
mkdocs build --strict
mkdocs serve
```
