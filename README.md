# College Majors 2026 — EDA para regressão

Projeto de Análise Exploratória de Dados da disciplina de Redes Neurais e Deep
Learning. O objetivo futuro é estimar `median_earnings_4yr_usd`, a renda mediana
quatro anos após a matrícula, sem usar variáveis derivadas do alvo nem
informações posteriores.

## Reprodução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
python docs/projects/eda/code/analysis.py
mkdocs serve
```

O relatório fica em `docs/projects/eda/index.md`; figuras e tabelas são
regeneradas pelo script de análise. O CSV bruto não é versionado por exceder
100 MB, mas o download público é automatizado.

