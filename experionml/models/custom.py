from typing import Any

from experionml.basemodel import BaseModel, ClassRegModel, ForecastModel
from experionml.utils.types import Predictor
from experionml.utils.utils import Goal


def create_custom_model(estimator: Predictor, **kwargs) -> BaseModel:
    """Cria um modelo personalizado a partir do estimador fornecido.

    Esta função atribui dinamicamente a classe pai.

    Parâmetros
    ----------
    estimator: Predictor
        Estimador a ser usado no modelo.

    kwargs
        Argumentos adicionais passados ao construtor do modelo.

    Retorna
    -------
    CustomModel
        Modelo personalizado com o estimador fornecido.

    """
    base = ForecastModel if kwargs["goal"] is Goal.forecast else ClassRegModel

    class CustomModel(base):  # type: ignore[valid-type]
        """Modelo com estimador fornecido pelo usuário."""

        def __init__(self, **kwargs):
            # Atribui o estimador e armazena os parâmetros fornecidos
            if callable(est := kwargs.pop("estimator")):
                self._est = est
                self._params = {}
            else:
                self._est = est.__class__
                self._params = est.get_params()

            if hasattr(est, "name"):
                name = est.name
            else:
                from experionml.models import MODELS

                # Se nenhum nome for fornecido, usa o nome da classe
                name = self.fullname
                if len(n := list(filter(str.isupper, name))) >= 2 and n not in MODELS:
                    name = "".join(n)

            self.acronym = getattr(est, "acronym", name)
            if not name.startswith(self.acronym):
                raise ValueError(
                    f"O nome ({name}) e a sigla ({self.acronym}) do modelo "
                    f"{self.fullname} não coincidem. O nome deve começar com "
                    f"a sigla do modelo."
                )

            self.needs_scaling = getattr(est, "needs_scaling", False)
            self.native_multilabel = getattr(est, "native_multilabel", False)
            self.native_multioutput = getattr(est, "native_multioutput", False)
            self.validation = getattr(est, "validation", None)

            super().__init__(name=name, **kwargs)

            self._estimators = {self._goal.name: self._est_class.__name__}

        @property
        def fullname(self) -> str:
            """Retorna o nome da classe do estimador."""
            return self._est_class.__name__

        @property
        def _est_class(self) -> type[Predictor]:
            """Retorna a classe do estimador."""
            return self._est

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
            return super()._get_est(self._params | params)

    return CustomModel(estimator=estimator, **kwargs)
