from __future__ import annotations

from typing import Any, ClassVar

from experionml.basemodel import BaseModel, ClassRegModel, ForecastModel
from experionml.utils.types import Model, Predictor
from experionml.utils.utils import Goal


def create_stacking_model(**kwargs) -> BaseModel:
    """Cria um modelo de stacking.

    Esta função atribui dinamicamente a classe pai.

    Parâmetros
    ----------
    kwargs
        Argumentos adicionais passados ao construtor do modelo.

    Retorna
    -------
    Stacking
        Modelo ensemble.

    """
    base = ForecastModel if kwargs["goal"] is Goal.forecast else ClassRegModel

    class Stacking(base):  # type: ignore[valid-type]
        """Ensemble de stacking.

        Parâmetros
        ----------
        models: list of Model
            Modelos a partir dos quais construir o ensemble.

        **kwargs
            Argumentos adicionais para o construtor de BaseModel.

        """

        acronym = "Stack"
        handles_missing = False
        needs_scaling = False
        validation = None
        multiple_seasonality = False
        native_multilabel = False
        native_multioutput = False
        supports_engines = ("sklearn",)

        _estimators: ClassVar[dict[str, str]] = {
            "classification": "sklearn.ensemble.StackingClassifier",
            "regression": "sklearn.ensemble.StackingRegressor",
            "forecast": "sktime.forecasting.compose.StackingForecaster",
        }

        def __init__(self, models: list[Model], **kwargs):
            super().__init__(**kwargs)
            self._models = models

        def _get_est(self, params: dict[str, Any]) -> Predictor:
            """Obtém o estimador do modelo com os parâmetros desempacotados.

            Parâmetros
            ----------
            params: dict
                Hiperparâmetros do estimador.

            Retorna
            -------
            Predictor
                Instância do estimador.

            """
            # Usamos _est_class com get_params em vez de um dict simples
            # para também fixar os parâmetros dos modelos no ensemble
            estimator = self._est_class(
                **{
                    "estimators" if not self.task.is_forecast else "forecasters": [
                        (m.name, m.export_pipeline()[-2:] if m.scaler else m.estimator)
                        for m in self._models
                    ]
                }
            )

            # Remove os nomes dos modelos dos params, pois não são
            # parâmetros diretos do ensemble
            default = {
                k: v
                for k, v in estimator.get_params().items()
                if k not in (m.name for m in self._models)
            }

            return super()._get_est(default | params)

    return Stacking(**kwargs)


def create_voting_model(**kwargs) -> BaseModel:
    """Cria um modelo de votação.

    Esta função atribui dinamicamente a classe pai.

    Parâmetros
    ----------
    kwargs
        Argumentos adicionais passados ao construtor do modelo.

    Retorna
    -------
    Voting
        Modelo ensemble.

    """
    base = ForecastModel if kwargs["goal"] is Goal.forecast else ClassRegModel

    class Voting(base):  # type: ignore[valid-type]
        """Ensemble de votação.

        Parâmetros
        ----------
        models: list of Model
            Modelos a partir dos quais construir o ensemble.

        **kwargs
            Argumentos adicionais para o construtor de BaseModel.

        """

        acronym = "Vote"
        handles_missing = False
        needs_scaling = False
        validation = None
        multiple_seasonality = False
        native_multilabel = False
        native_multioutput = False
        supports_engines = ("sklearn",)

        _estimators: ClassVar[dict[str, str]] = {
            "classification": "sklearn.ensemble.VotingClassifier",
            "regression": "sklearn.ensemble.VotingRegressor",
            "forecast": "sktime.forecasting.compose.EnsembleForecaster",
        }

        def __init__(self, models: list[Model], **kwargs):
            super().__init__(**kwargs)
            self._models = models

        def _get_est(self, params: dict[str, Any]) -> Predictor:
            """Obtém o estimador do modelo com os parâmetros desempacotados.

            Parâmetros
            ----------
            params: dict
                Hiperparâmetros do estimador.

            Retorna
            -------
            Predictor
                Instância do estimador.

            """
            # Usamos _est_class com get_params em vez de um dict simples
            # para também fixar os parâmetros dos modelos no ensemble
            estimator = self._est_class(
                **{
                    "estimators" if not self.task.is_forecast else "forecasters": [
                        (m.name, m.export_pipeline()[-2:] if m.scaler else m.estimator)
                        for m in self._models
                    ]
                }
            )

            # Remove os nomes dos modelos dos params, pois não são
            # parâmetros diretos do ensemble
            default = {
                k: v
                for k, v in estimator.get_params().items()
                if k not in (m.name for m in self._models)
            }

            return super()._get_est(default | params)

    return Voting(**kwargs)
