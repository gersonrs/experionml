__all__ = ["show_versions"]


import importlib
import platform
import sys


# Dependências cujas versões serão exibidas
DEFAULT_DEPS = [
    "pip",
    "experionml",
    "beartype",
    "category_encoders",
    "dill",
    "featuretools",
    "gplearn",
    "imblearn",
    "ipywidgets",
    "joblib",
    "matplotlib",
    "mlflow",
    "modin",
    "nltk",
    "numpy",
    "optuna",
    "pandas",
    "plotly",
    "sklearn",
    "scipy",
    "shap",
    "sktime",
    "statsmodels",
    "zoofs",  # Has no __version__ attribute
    "botorch",
    "catboost",
    "dagshub",
    "dask",
    "explainerdashboard",
    "gradio",
    "lightgbm",
    "modin",
    "polars",
    "pyarrow",
    "pyspark",
    "ray",
    "requests",
    "sklearnex",  # Has no __version__ attribute
    "schemdraw",
    "statsforecast",
    "sweetviz",
    "wordcloud",
    "xgboost",
]


def _get_sys_info():
    """Obtém informações do sistema e da versão do Python.

    Retorna
    -------
    dict
        Informações coletadas.

    """
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "machine": platform.platform(),
    }


def _get_deps_info(deps: list[str]) -> dict[str, str | None]:
    """Retorna uma visão geral das versões instaladas das dependências principais.

    Parâmetros
    ----------
    deps: list of str
        Dependências das quais obter a versão.

    Retorna
    -------
    dict
        Informações de versão das bibliotecas em `deps`, onde as chaves
        são os nomes de importação e os valores são strings de versão
        PEP 440 conforme disponíveis no ambiente Python atual.

    """
    deps_info = {}
    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            deps_info[modname] = mod.__version__
        except (ImportError, AttributeError):  # noqa: PERF203
            deps_info[modname] = None

    return deps_info


def show_versions():
    """Exibe informações do sistema e dos pacotes.

    As seguintes informações são exibidas:

    - Versão do Python do ambiente.
    - Localização do executável do Python.
    - Versão do sistema operacional.
    - Nome de importação e versão das dependências Python selecionadas.

    """
    sys_info = _get_sys_info()
    deps_info = _get_deps_info(deps=DEFAULT_DEPS)

    print("\nSistema:")  # noqa: T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T201

    print("\nDependências Python:")  # noqa: T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T201
