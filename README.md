# College Majors 2026 — EDA para regressão

Projeto de Análise Exploratória de Dados da disciplina de Redes Neurais e Deep
Learning. O objetivo futuro é estimar `median_earnings_4yr_usd`, a renda mediana
quatro anos após a conclusão, sem usar variáveis derivadas do alvo nem
informações posteriores.

## Reprodução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
python docs/projects/eda/code/analysis.py
python -m docs.projects.regression.code.data --check
python -m docs.projects.regression.code.baselines --check
python -m docs.projects.regression.code.train --check
python -m docs.projects.regression.code.search --check
python -m docs.projects.regression.code.final_config --check
python -m docs.projects.regression.code.final_evaluation --check
python -m docs.projects.regression.code.error_analysis --check
pytest -q
mkdocs build --strict
mkdocs serve
```

O relatório fica em `docs/projects/eda/index.md`; figuras e tabelas de
resultados são regeneradas pelo script de análise. O dicionário, curado a
partir das fontes, é validado pelo mesmo script contra as 72 colunas do CSV.
O CSV bruto não é versionado por exceder 100 MB, mas o download público é
automatizado.

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

A avaliação final foi executada uma única vez com a seed 42. O comando
`python -m docs.projects.regression.code.final_evaluation --check` valida
offline as métricas, previsões, tabela, figuras e hashes; ele não reabre o
dataset nem permite sobrescrever os resultados finais.

A análise de erros usa somente as previsões finais já versionadas; importância
por permutação e ablação rodam na validação. Use `--write-analysis` para
regenerar seus artefatos e `--check` para reproduzir e conferir a análise sem
reabrir o teste.

Em Linux, o arquivo de dependências instala a distribuição CPU do PyTorch para
evitar vários gigabytes de bibliotecas CUDA. Uma instalação com GPU pode
substituí-la pela distribuição indicada para a versão de CUDA da máquina.
