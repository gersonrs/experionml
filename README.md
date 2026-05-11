# ExperionML

[![PyPI version](https://img.shields.io/pypi/v/experionml.svg)](https://pypi.org/project/experionml/)
[![Python versions](https://img.shields.io/pypi/pyversions/experionml.svg)](https://pypi.org/project/experionml/)
[![CI](https://github.com/gersonrs/ExperionML/actions/workflows/ci.yml/badge.svg)](https://github.com/gersonrs/ExperionML/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2a6db2.svg)](https://mypy-lang.org/)

> Camada de orquestração para experimentação em machine learning — do dado bruto à comparação estruturada de modelos em poucas linhas.

---

## 💡 Introdução

Durante a fase de exploração de um projeto de ML, o cientista de dados precisa
identificar o pipeline mais adequado para o seu problema. Isso envolve limpar
dados, engenheirar atributos, testar algoritmos e avaliar configurações.

Explorar várias combinações custa **muito código repetido**. Em um único
notebook o arquivo fica gigante; em vários, a comparação entre resultados
se perde. Refatorar a cada novo experimento consome tempo que deveria estar
indo para análise.

**ExperionML** centraliza tudo em um único objeto de orquestração:
dataset + branches + pipeline + runners de treino + plots. A API é uma
fachada única, e por trás dela o estado do experimento é explícito e
reproduzível.

## ✨ Principais recursos

- **Branches de experimento** — múltiplos estados paralelos (`main`, `scaled`,
  `balanced`, …) sem reimplementar o pipeline.
- **Pipeline estendido do sklearn** — suporta transformadores que alteram
  linhas, que operam em `X` e `y`, cache com joblib e séries temporais.
- **Limpeza, encoding, imputação, balanceamento** em chamadas de uma linha.
- **Feature engineering** via featuretools, gplearn e zoofs.
- **Treino, tuning com Optuna, bootstrap, SuccessiveHalving, TrainSizing**.
- **Tracking com MLflow** integrado, inclusive DAGsHub via `Integrator`.
- **Plots padronizados** para comparação de modelos, calibração, SHAP, etc.
- **NLP**, previsão de séries temporais (`sktime`) e mais.

## 📦 Instalação

Requer **Python 3.10, 3.11 ou 3.12**.

```bash
pip install experionml
```

Para o conjunto completo de extras opcionais (CatBoost, LightGBM, XGBoost,
Ray, Dask, Polars, Gradio, ExplainerDashboard, …):

```bash
pip install "experionml[full]"
```

## 🚀 Quickstart

```python
from sklearn.datasets import load_breast_cancer

from experionml import ExperionMLClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

# Inicia o experimento
exp = ExperionMLClassifier(X, y=y, random_state=1)

# Pré-processamento encadeado — cada chamada registra uma etapa no pipeline
exp.impute(strat_num="median", strat_cat="most_frequent")
exp.encode(strategy="Target", max_onehot=6)
exp.scale(strategy="standard")

# Treina vários modelos e compara (acronyms resolvidos via registry)
exp.run(models=["LR", "RF", "LGB"], metric="f1")

# Resultados prontos para análise
print(exp.results)
exp.plot_results()
```

## 🧭 Arquitetura em 30 segundos

```
ExperionMLClassifier / Regressor / Forecaster        (fachada)
        └── ExperionML                               (orquestrador)
             ├── BranchManager → Branches            (estado do experimento)
             ├── Pipeline                            (histórico executável)
             ├── Transformers                        (clean, encode, ...)
             ├── Runners + Trainers                  (Direct, SH, TrainSizing)
             └── Plots                               (análise visual)
```

Detalhes completos em [ARCHITECTURE_PRESENTATION.md](ARCHITECTURE_PRESENTATION.md).

## 📚 Exemplos

A pasta [examples/](examples/) contém notebooks cobrindo:
classificação binária/multiclasse/multilabel, regressão, forecasting,
NLP, SHAP, calibração, hyperparameter tuning, successive halving,
train sizing, ensembles, feature engineering e muito mais.

Comece por [examples/getting_started.ipynb](examples/getting_started.ipynb).

## 🤝 Contribuindo

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para ambiente de dev, estilo de
código, testes, conventional commits e processo de PR.

Temos também:
[Código de Conduta](CODE_OF_CONDUCT.md) · [Política de Segurança](SECURITY.md)

## 📝 Licença

[MIT](LICENSE) — © gersonrs
