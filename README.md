# Boston House Prices — Professional ML Project

[![CI](https://github.com/gabrielrosalino/boston-house-prices/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrielrosalino/boston-house-prices/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/model-scikit--learn-orange)
![FastAPI](https://img.shields.io/badge/api-FastAPI-green)

Projeto profissional de Machine Learning para previsão de preços de imóveis usando o clássico Boston Housing Dataset.

Este repositório foi estruturado como um projeto de portfólio end-to-end, indo além de um notebook exploratório: ele contém pacote Python, pipeline de treino reproduzível, avaliação, inferência via CLI, API REST, testes automatizados, Docker, lint, type checking e CI/CD.

## Objetivos do projeto

- Demonstrar domínio de estrutura profissional de projetos Python/ML.
- Criar um pipeline reproduzível de treino e predição.
- Expor o modelo via CLI e API REST.
- Aplicar boas práticas de engenharia: testes, lint, typing, Docker e CI.
- Documentar decisões técnicas de forma clara para recrutadores e times técnicos.

## Stack

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3.11+ |
| ML | scikit-learn |
| Dados | pandas, numpy |
| API | FastAPI, Uvicorn |
| CLI | Typer, Rich |
| Qualidade | pytest, ruff, mypy, pre-commit |
| Empacotamento | pyproject.toml |
| Container | Docker, Docker Compose |
| CI | GitHub Actions |

## Estrutura

```text
.
├── src/boston_house_prices/
│   ├── api.py              # FastAPI app
│   ├── cli.py              # CLI commands
│   ├── config.py           # Project paths/settings
│   ├── data.py             # Dataset loading utilities
│   ├── pipeline.py         # ML pipelines and metrics
│   ├── predict.py          # Prediction service
│   ├── schemas.py          # Pydantic schemas
│   └── train.py            # Training workflow
├── tests/                  # Unit tests
├── .github/workflows/      # CI pipeline
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── README.md
```

## Arquitetura do fluxo

```text
OpenML Boston Dataset
        │
        ▼
Data loading / validation
        │
        ▼
Train-test split
        │
        ▼
Scikit-learn Pipeline
        │
        ├── Median imputation
        └── RandomForestRegressor
        │
        ▼
Evaluation metrics
        │
        ├── MAE
        ├── RMSE
        └── R²
        │
        ▼
Persisted model artifact
        │
        ├── CLI prediction
        └── FastAPI endpoint
```

## Como executar localmente

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale o projeto com dependências de desenvolvimento:

```bash
make install-dev
```

Treine o modelo:

```bash
make train
```

O treino gera:

```text
models/house_price_pipeline.joblib
reports/metrics.json
```

## Predição via CLI

```bash
boston-predict --payload '{
  "CRIM": 0.00632,
  "ZN": 18.0,
  "INDUS": 2.31,
  "CHAS": 0.0,
  "NOX": 0.538,
  "RM": 6.575,
  "AGE": 65.2,
  "DIS": 4.09,
  "RAD": 1.0,
  "TAX": 296.0,
  "PTRATIO": 15.3,
  "B": 396.9,
  "LSTAT": 4.98
}'
```

## API REST

Execute a API:

```bash
make api
```

Health check:

```bash
curl http://localhost:8000/health
```

Predição:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CRIM": 0.00632,
    "ZN": 18.0,
    "INDUS": 2.31,
    "CHAS": 0.0,
    "NOX": 0.538,
    "RM": 6.575,
    "AGE": 65.2,
    "DIS": 4.09,
    "RAD": 1.0,
    "TAX": 296.0,
    "PTRATIO": 15.3,
    "B": 396.9,
    "LSTAT": 4.98
  }'
```

Documentação interativa:

```text
http://localhost:8000/docs
```

## Docker

Build:

```bash
make docker-build
```

Run:

```bash
make docker-run
```

Ou com Docker Compose:

```bash
docker compose up --build
```

## Qualidade de código

Rodar lint:

```bash
make lint
```

Formatar:

```bash
make format
```

Type check:

```bash
make type-check
```

Testes:

```bash
make test
```

## CI/CD

O pipeline de GitHub Actions executa automaticamente:

- instalação das dependências
- lint com Ruff
- verificação de formatação
- type checking com mypy
- testes com pytest e coverage

## Observações sobre o dataset

O Boston Housing Dataset é clássico para estudos de regressão, mas possui limitações conhecidas e não deve ser usado como referência moderna para decisões reais de crédito, moradia ou avaliação imobiliária. Neste projeto, ele é utilizado apenas como material didático para demonstrar engenharia de ML.

## Próximas evoluções

- Adicionar MLflow para tracking de experimentos.
- Adicionar validação de dados com Great Expectations ou Pandera.
- Criar pipeline de batch inference.
- Publicar imagem Docker em registry.
- Fazer deploy em AWS ECS ou Render.
- Adicionar monitoramento básico de drift.
