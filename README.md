# College Majors 2026 — EDA para regressão

Projeto de Análise Exploratória de Dados da disciplina de Redes Neurais e Deep
Learning. O objetivo futuro é estimar `median_earnings_4yr_usd`, a renda mediana
quatro anos após a conclusão, sem usar variáveis derivadas do alvo nem
informações posteriores.

## Instalação e dados

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
python docs/projects/eda/code/analysis.py
```

O download grava o arquivo bruto em `data/raw/college_majors_2026.csv`. Ele não
é versionado por exceder 100 MB. A análise da EDA regenera suas tabelas e
figuras a partir desse arquivo.

## Verificação da regressão

```bash
python -m docs.projects.regression.code.data --check
python -m docs.projects.regression.code.baselines --check
python -m docs.projects.regression.code.train --check
python -m docs.projects.regression.code.search --check
python -m docs.projects.regression.code.final_config --check
python -m docs.projects.regression.code.final_evaluation --check
python -m docs.projects.regression.code.error_analysis --check
pytest -q
```

Os comandos de desenvolvimento retreinam os modelos em CPU e podem levar
alguns minutos. `final_evaluation --check` é deliberadamente offline: valida
as métricas, previsões e hashes versionados sem reabrir o teste. A análise de
erros usa esse CSV final e retreina somente em treino/validação.

Para regenerar artefatos de desenvolvimento após uma mudança deliberada, use
os comandos `--write-manifest`, `--write-results`, `--write-reference`,
`--write-config` e `--write-analysis` nos respectivos módulos. Não execute
novamente `final_evaluation --write-results`: a avaliação de teste foi única e
o código bloqueia sobrescrita.

## Documentação

```bash
mkdocs build --strict
mkdocs serve
```

Os relatórios ficam em `docs/projects/eda/index.md` e
`docs/projects/regression/index.md`.

A fundação da etapa supervisionada fica em `docs/projects/regression/`. O
comando de verificação acima reproduz os splits de desenvolvimento, confere o
manifesto versionado e mantém o conjunto de teste bloqueado para a avaliação
final.

Os baselines de regressão são reproduzidos somente sobre treino e validação.
Use `--write-results` no módulo `baselines` apenas quando quiser regenerar os
artefatos versionados; `--check` recalcula e confere os resultados existentes.

A MLP de referência também usa comandos separados: `--write-reference`
regenera o histórico e a figura em CPU, enquanto `--check` retreina com a
mesma semente e verifica a reprodução sem acessar o teste.

A busca de hiperparâmetros compara 12 configurações fixas com
`python -m docs.projects.regression.code.search --write-results`. O comando
`--check` repete a busca em CPU e confere os resultados versionados; ambos
usam somente treino e validação.

A configuração escolhida fica congelada em `final_config.json`. O comando
`python -m docs.projects.regression.code.final_config --check` confere sua
correspondência com a busca sem retreinar a rede nem acessar o teste.

A avaliação final foi executada uma única vez com a seed 42. Seus artefatos
versionados permitem verificar os resultados sem acessar novamente o teste.

A análise de erros usa somente as previsões finais já versionadas; importância
por permutação e ablação rodam na validação. Use `--write-analysis` para
regenerar seus artefatos e `--check` para reproduzir e conferir a análise sem
reabrir o teste.

Em Linux, o arquivo de dependências instala a distribuição CPU do PyTorch para
evitar vários gigabytes de bibliotecas CUDA. Uma instalação com GPU pode
substituí-la pela distribuição indicada para a versão de CUDA da máquina.
