from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution as Cat
from optuna.distributions import FloatDistribution as Float
from optuna.distributions import IntDistribution as Int

from experionml.basemodel import ForecastModel
from experionml.utils.types import Predictor
from experionml.utils.utils import SeasonalPeriod


class ARIMA(ForecastModel):
    """Média móvel integrada autorregressiva.

    Há suporte a modelos ARIMA sazonais e entrada exógena; portanto,
    este estimador é capaz de ajustar SARIMA, ARIMAX e SARIMAX.

    Um modelo ARIMA é uma generalização de um modelo autorregressivo de
    médias móveis (ARMA) e é ajustado a dados de séries temporais com o
    objetivo de prever pontos futuros. Modelos ARIMA podem ser
    especialmente eficazes quando os dados mostram evidências de
    não estacionariedade.

    A parte "AR" do ARIMA indica que a variável de interesse em evolução
    é regredida em seus próprios valores defasados, ou seja, observados
    anteriormente. A parte "MA" indica que o erro de regressão é, na
    prática, uma combinação linear de termos de erro cujos valores
    ocorreram contemporaneamente e em vários momentos do passado. O "I"
    de "integrated" indica que os valores dos dados foram substituídos
    pela diferença entre seus valores e os valores anteriores, e esse
    processo de diferenciação pode ter sido realizado mais de uma vez.

    Os estimadores correspondentes são:

    - [ARIMA][arimaclass] para tarefas de previsão.

    !!! note
        Os componentes sazonais são removidos do [ajuste de
        hiperparâmetros][] se nenhuma [sazonalidade][] for especificada.

    !!! warning
        O ARIMA frequentemente encontra erros numéricos ao otimizar os
        hiperparâmetros. Possíveis soluções são:

                - Use o modelo [AutoARIMA][] no lugar.
                - Use [`est_params`][directforecaster-est_params] para
                    especificar as ordens manualmente, por exemplo,
                    `#!python experionml.run("arima", n_trials=5,
                    est_params={"order": (1, 1, 0)})`.
                - Use o parâmetro `catch` em
                    [`ht_params`][directforecaster-ht_params] para evitar levantar
                    todas as exceções, por exemplo,
                    `#!python experionml.run("arima", n_trials=5,
                    ht_params={"catch": (Exception,)})`.

    Veja também
    --------
    experionml.models:AutoARIMA
    experionml.models:SARIMAX
    experionml.models:VARMAX

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    experionml = ExperionMLForecaster(X)
    experionml.run(models="ARIMA", verbose=2)
    ```

    """

    acronym = "ARIMA"
    handles_missing = True
    uses_exogenous = True
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.arima.ARIMA"}

    _order = ("p", "d", "q")
    _s_order = ("P", "D", "Q")

    def _trial_to_est(self, params: dict[str, Any]) -> dict[str, Any]:
        """Converte os hiperparâmetros do trial em parâmetros do estimador.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros do trial.

        Retorna
        -------
        dict
            Hiperparâmetros do estimador.

        """
        params = super()._trial_to_est(params)

        # Converte os parâmetros nos hiperparâmetros 'order' e 'seasonal_order'
        if all(p in params for p in self._order):
            params["order"] = [params[p] for p in self._order]
        if all(p in params for p in self._s_order) and self._config.sp.get("sp"):
            params["seasonal_order"] = [params[p] for p in self._s_order] + [self._config.sp.sp]

        # Remove os parâmetros de order e seasonal_order
        for p in self._order:
            params.pop(p, None)
        for p in self._s_order:
            params.pop(p, None)

        return params

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est({"suppress_warnings": self.warnings == "ignore"} | params)

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "p": Int(0, 2),
            "d": Int(0, 1),
            "q": Int(0, 2),
            "P": Int(0, 2),
            "D": Int(0, 1),
            "Q": Int(0, 2),
            "method": Cat(["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]),
            "maxiter": Int(50, 200, step=10),
            "with_intercept": Cat([True, False]),
        }

        # Remove os parâmetros de order se especificados pelo usuário
        if "order" in self._est_params:
            for p in self._order:
                dist.pop(p)
        if "seasonal_order" in self._est_params or not self._config.sp.get("sp"):
            # Remove os parâmetros sazonais se especificados pelo usuário ou sem sazonalidade
            for p in self._s_order:
                dist.pop(p)

        return dist


class AutoARIMA(ForecastModel):
    """Média móvel integrada autorregressiva automática.

    Implementação de [ARIMA][] que inclui ajuste automatizado dos
    hiperparâmetros de (S)ARIMA(X) (p, d, q, P, D, Q). O algoritmo
    AutoARIMA procura identificar os parâmetros mais adequados para um
    modelo ARIMA, chegando a um único modelo ARIMA ajustado. Esse
    processo se baseia na função comumente usada em R.

    O AutoARIMA funciona realizando testes de diferenciação, como
    Kwiatkowski-Phillips-Schmidt-Shin, Augmented Dickey-Fuller ou
    Phillips-Perron, para determinar a ordem de diferenciação d, e em
    seguida ajustando modelos dentro de faixas definidas. O AutoARIMA
    também busca identificar os hiperparâmetros ótimos P e Q após
    conduzir o teste de Canova-Hansen para determinar a ordem ideal da
    diferenciação sazonal.

    Observe que, devido a questões de estacionariedade, o AutoARIMA
    pode não encontrar um modelo adequado que converja. Se esse for o
    caso, uma ValueError é levantada sugerindo a adoção de medidas para
    induzir estacionariedade antes de um novo ajuste ou a seleção de um
    novo intervalo de valores de ordem.

    Os estimadores correspondentes são:

    - [AutoARIMA][autoarimaclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:ARIMA
    experionml.models:ETS
    experionml.models:SARIMAX

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    experionml = ExperionMLForecaster(X, random_state=1)
    experionml.run(models="autoarima", verbose=2)
    ```

    """

    acronym = "AutoARIMA"
    handles_missing = True
    uses_exogenous = True
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.arima.AutoARIMA"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est({"suppress_warnings": self.warnings == "ignore"} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "information_criterion": Cat(["aic", "bic", "hqic", "oob"]),
            "method": Cat(["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]),
            "maxiter": Int(50, 200, step=10),
            "with_intercept": Cat([True, False]),
        }


class AutoETS(ForecastModel):
    """Modelo ETS com capacidades de ajuste automático.

    Os modelos [ETS][] são uma família de modelos de séries temporais
    com um modelo subjacente de espaço de estados composto por um
    componente de nível, um componente de tendência (T), um componente
    sazonal (S) e um termo de erro (E). Esta implementação ajusta
    automaticamente os termos do ETS.

    Os estimadores correspondentes são:

    - [AutoETS][] para tarefas de previsão.

    Veja também
    --------
    experionml.models:AutoARIMA
    experionml.models:ETS
    experionml.models:SARIMAX

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="ETS", verbose=2)
    ```

    """

    acronym = "AutoETS"
    handles_missing = True
    uses_exogenous = False
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.ets.AutoETS"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est({"sp": self._config.sp.get("sp", 1), "auto": True} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "information_criterion": Cat(["aic", "bic", "aicc"]),
            "allow_multiplicative_trend": Cat([True, False]),
            "restrict": Cat([True, False]),
            "additive_only": Cat([True, False]),
            "ignore_inf_ic": Cat([True, False]),
        }


class BATS(ForecastModel):
    """Previsor BATS com múltiplas sazonalidades.

    BATS é a sigla para:

    - transformação Box-Cox
    - erros ARMA
    - tendência
    - componentes sazonais

    BATS foi projetado para prever séries temporais com múltiplos
    períodos sazonais. Por exemplo, dados diários podem ter um padrão
    semanal e também um padrão anual. Ou dados horários podem ter três
    períodos sazonais: um padrão diário, um semanal e um anual.

    Em BATS, uma [transformação Box-Cox][boxcox] é aplicada à série
    temporal original, e então ela é modelada como uma combinação
    linear de uma tendência suavizada exponencialmente, um componente
    sazonal e um componente ARMA. O BATS realiza algum ajuste de
    hiperparâmetros, por exemplo, decidindo quais desses componentes
    manter ou descartar, usando AIC.

    Os estimadores correspondentes são:

    - [BATS][batsclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:ARIMA
    experionml.models:AutoARIMA
    experionml.models:TBATS

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="BATS", verbose=2)
    ```

    """

    acronym = "BATS"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.bats.BATS"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est({"show_warnings": self.warnings != "ignore"} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "use_box_cox": Cat([True, False, None]),
            "use_trend": Cat([True, False, None]),
            "use_damped_trend": Cat([True, False, None]),
            "use_arma_errors": Cat([True, False]),
        }


class Croston(ForecastModel):
    """Método de Croston para previsão.

    O método de Croston é uma modificação da suavização exponencial
    tradicional para lidar com séries temporais intermitentes. Uma
    série temporal é considerada intermitente quando muitos de seus
    valores são zero e os intervalos entre entradas não nulas não são
    periódicos.

    O método de Croston prevê um valor constante para todos os tempos
    futuros, oferecendo assim outra noção do valor médio de uma série
    temporal.

    Os estimadores correspondentes são:

    - [Croston][crostonclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:ExponentialSmoothing
    experionml.models:ETS
    experionml.models:NaiveForecaster

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="Croston", verbose=2)
    ```

    """

    acronym = "Croston"
    handles_missing = False
    uses_exogenous = True
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.croston.Croston"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {"smoothing": Float(0, 1, step=0.1)}


class DynamicFactor(ForecastModel):
    """Fator dinâmico.

    O modelo DynamicFactor incorpora fatores dinâmicos para prever
    valores futuros. Nesse contexto, "fatores dinâmicos" referem-se
    a variáveis que mudam ao longo do tempo e impactam a variável que
    você está tentando prever.

    !!! warning
        DynamicFactor suporta apenas tarefas [multivariadas][].

    Os estimadores correspondentes são:

    - [DynamicFactor][dynamicfactorclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:ExponentialSmoothing
    experionml.models:STL
    experionml.models:PolynomialTrend

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    experionml = ExperionMLForecaster(X, y=(-1, -2), random_state=1)
    experionml.run(models="DF", verbose=2)
    ```

    """

    acronym = "DF"
    handles_missing = True
    uses_exogenous = True
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {
        "forecast": "sktime.forecasting.dynamic_factor.DynamicFactor"
    }

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "k_factors": Int(1, 10),
            "error_cov_type": Cat(["scalar", "diagonal", "unstructured"]),
            "error_var": Cat([True, False]),
            "enforce_stationarity": Cat([True, False]),
            "cov_type": Cat(["opg", "oim", "approx", "robust", "robust_approx", "none"]),
            "method": Cat(["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]),
            "maxiter": Int(50, 200, step=10),
        }


class ExponentialSmoothing(ForecastModel):
    """Previsor de suavização exponencial de Holt-Winters.

    ExponentialSmoothing é um modelo de previsão que estende a
    suavização exponencial simples para lidar com sazonalidade e
    tendências nos dados. Esse método é particularmente útil para
    prever dados de séries temporais com um padrão sistemático que se
    repete ao longo do tempo.

    Os estimadores correspondentes são:

    - [ExponentialSmoothing][esclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:ARIMA
    experionml.models:ETS
    experionml.models:PolynomialTrend

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="ES", verbose=2)
    ```

    """

    acronym = "ES"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {
        "forecast": "sktime.forecasting.exp_smoothing.ExponentialSmoothing"
    }

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est(
            {
                "trend": self._config.sp.get("trend_model"),
                "seasonal": self._config.sp.get("seasonal_model"),
            }
            | params
        )

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "damped_trend": Cat([True, False]),
            "use_boxcox": Cat([True, False]),
            "initialization_method": Cat(["estimated", "heuristic"]),
            "method": Cat(["L-BFGS-B", "TNC", "SLSQP", "Powell", "trust-constr", "bh", "ls"]),
            "use_brute": Cat([True, False]),
        }


class ETS(ForecastModel):
    """Modelo erro-tendência-sazonalidade.

    Os modelos ETS são uma família de modelos de séries temporais com
    um modelo subjacente em espaço de estados composto por um
    componente de nível, um componente de tendência (T), um componente
    sazonal (S) e um termo de erro (E).

    Os estimadores correspondentes são:

    - [AutoETS][] para tarefas de previsão.

    Veja também
    --------
    experionml.models:ARIMA
    experionml.models:AutoETS
    experionml.models:SARIMAX

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="ETS", verbose=2)
    ```

    """

    acronym = "ETS"
    handles_missing = True
    uses_exogenous = False
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.ets.AutoETS"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est(
            {
                "trend": self._config.sp.get("trend_model"),
                "seasonal": self._config.sp.get("seasonal_model"),
            }
            | params
        )

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "error": Cat(["add", "mul"]),
            "damped_trend": Cat([True, False]),
            "initialization_method": Cat(["estimated", "heuristic"]),
            "maxiter": Int(500, 2000, step=100),
        }


class MSTL(ForecastModel):
    """Decomposição múltipla de tendência-sazonalidade usando LOESS.

    O modelo MSTL (decomposição múltipla de tendência-sazonalidade
    usando LOESS) é um método usado para decompor uma série temporal
    em seus componentes sazonais, de tendência e residuais. Essa
    abordagem se baseia no uso de LOESS (suavização por regressão
    local) para estimar os componentes da série temporal.

    A decomposição MSTL é uma extensão do método clássico de
    decomposição sazonal-tendência, também conhecido como decomposição
    de Holt-Winters, projetado para lidar com situações em que existem
    múltiplos padrões sazonais nos dados. Isso pode ocorrer, por
    exemplo, quando uma série temporal apresenta padrões diários e
    anuais simultaneamente.

    Os estimadores correspondentes são:

    - [StatsForecastMSTL][] para tarefas de previsão.

    Veja também
    --------
    experionml.models:Prophet
    experionml.models:STL
    experionml.models:TBATS

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="MSTL", verbose=2)
    ```

    """

    acronym = "MSTL"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = True
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {
        "forecast": "sktime.forecasting.statsforecast.StatsForecastMSTL"
    }

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est({"season_length": self._config.sp.get("sp", 1)} | params)

    def _trial_to_est(self, params: dict[str, Any]) -> dict[str, Any]:
        """Converte os hiperparâmetros do trial em parâmetros do estimador.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros do trial.

        Retorna
        -------
        dict
            Hiperparâmetros do estimador.

        """
        params = super()._trial_to_est(params)

        return {"stl_kwargs": self._est_params.get("stl_kwargs", {}) | params}

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "seasonal_deg": Cat([0, 1]),
            "trend_deg": Cat([0, 1]),
            "low_pass_deg": Cat([0, 1]),
            "robust": Cat([True, False]),
        }

        # StatsForecastMSTL tem stl_kwargs, que aceita um dict de hiperparâmetros
        for p in self._est_params.get("stl_kwargs", {}):
            dist.pop(p)

        return dist


class NaiveForecaster(ForecastModel):
    """Previsor ingênuo.

    NaiveForecaster é um previsor simples que gera previsões usando
    estratégias baseadas em suposições ingênuas de continuidade das
    tendências passadas. Quando usado em tarefas [multivariadas][],
    cada coluna é prevista com a mesma estratégia.

    Os estimadores correspondentes são:

    - [NaiveForecaster][naiveforecasterclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:ExponentialSmoothing
    experionml.models:Dummy
    experionml.models:PolynomialTrend

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="NF", verbose=2)
    ```

    """

    acronym = "NF"
    handles_missing = True
    uses_exogenous = False
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.naive.NaiveForecaster"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est({"sp": self._config.sp.get("sp", 1)} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {"strategy": Cat(["last", "mean", "drift"])}


class PolynomialTrend(ForecastModel):
    """Previsor de tendência polinomial.

    Prevê dados de séries temporais com uma tendência polinomial,
    usando a classe [LinearRegression][] do sklearn para regredir os
    valores da série temporal no índice, após a extração de variáveis
    polinomiais.

    Os estimadores correspondentes são:

    - [PolynomialTrendForecaster][] para tarefas de previsão.

    Veja também
    --------
    experionml.models:ARIMA
    experionml.models:ETS
    experionml.models:NaiveForecaster

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="PT", verbose=2)
    ```

    """

    acronym = "PT"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {
        "forecast": "sktime.forecasting.trend.PolynomialTrendForecaster"
    }

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "degree": Int(1, 5),
            "with_intercept": Cat([True, False]),
        }


class Prophet(ForecastModel):
    """Previsor Prophet do Facebook.

    Prophet foi projetado para lidar com séries temporais com padrões
    sazonais fortes, feriados e dados ausentes. É particularmente útil
    em aplicações de negócio nas quais os dados podem apresentar
    irregularidades e nem sempre serem perfeitamente regulares.

    Os estimadores correspondentes são:

    - [Prophet][prophetclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:DynamicFactor
    experionml.models:MSTL
    experionml.models:VARMAX

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="Prophet", verbose=2)
    ```

    """

    acronym = "Prophet"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = True
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.fbprophet.Prophet"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        # Prophet espera uma frequência em índice DateTime
        freq = None
        if self._config.sp.get("sp"):
            try:
                freq = next(
                    n
                    for n, m in SeasonalPeriod.__members__.items()
                    if m.value == self._config.sp.sp
                )
            except StopIteration:
                # Se não estiver na tabela de mapeamento, obtém do índice
                if hasattr(self.X_train.index, "freq"):
                    freq = self.X_train.index.freq.name

        return super()._get_est(
            {"freq": freq, "seasonality_mode": self._config.sp.get("seasonal_model", "additive")}
            | params
        )

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "changepoint_prior_scale": Float(0.001, 0.5, log=True),
            "seasonality_prior_scale": Float(0.001, 10, log=True),
            "holidays_prior_scale": Float(0.001, 10, log=True),
        }


class SARIMAX(ForecastModel):
    """Média móvel integrada autorregressiva sazonal.

    SARIMAX significa média móvel integrada autorregressiva sazonal com
    fatores exógenos. Ele estende o [ARIMA][] ao incorporar componentes
    sazonais e variáveis exógenas. Observe que o modelo ARIMA também é
    capaz de ajustar um SARIMAX.

    Os estimadores correspondentes são:

    - [SARIMAX][sarimaxclass] para tarefas de previsão.

    !!! note
        Os componentes sazonais são removidos do [ajuste de
        hiperparâmetros][] se nenhuma [sazonalidade][] for especificada.

    !!! warning
        O SARIMAX frequentemente encontra erros numéricos ao otimizar os
        hiperparâmetros. Possíveis soluções são:

                - Use o modelo [AutoARIMA][] no lugar.
                - Use [`est_params`][directforecaster-est_params] para especificar
                    as ordens manualmente, por exemplo, `#!python experionml.run("sarimax",
                    n_trials=5, est_params={"order": (1, 1, 0)})`.
                - Use o parâmetro `catch` em [`ht_params`][directforecaster-ht_params]
                    para evitar levantar todas as exceções, por exemplo,
                    `#!python experionml.run("sarimax", n_trials=5,
                    ht_params={"catch": (Exception,)})`.

    Veja também
    --------
    experionml.models:ARIMA
    experionml.models:AutoARIMA
    experionml.models:VARMAX

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    experionml = ExperionMLForecaster(X)
    experionml.run(models="SARIMAX", verbose=2)
    ```

    """

    acronym = "SARIMAX"
    handles_missing = False
    uses_exogenous = True
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.sarimax.SARIMAX"}

    _order = ("p", "d", "q")
    _s_order = ("P", "D", "Q")

    def _trial_to_est(self, params: dict[str, Any]) -> dict[str, Any]:
        """Converte os hiperparâmetros do trial em parâmetros do estimador.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros do trial.

        Retorna
        -------
        dict
            Hiperparâmetros do estimador.

        """
        params = super()._trial_to_est(params)

        # Converte os parâmetros nos hiperparâmetros 'order' e 'seasonal_order'
        if all(p in params for p in self._order):
            params["order"] = [params[p] for p in self._order]
        if all(p in params for p in self._s_order) and self._config.sp.sp:
            params["seasonal_order"] = [params[p] for p in self._s_order] + [self._config.sp.sp]

        # Remove os parâmetros de order e seasonal_order
        for p in self._order:
            params.pop(p, None)
        for p in self._s_order:
            params.pop(p, None)

        return params

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "p": Int(0, 2),
            "d": Int(0, 1),
            "q": Int(0, 2),
            "P": Int(0, 2),
            "D": Int(0, 1),
            "Q": Int(0, 2),
            "trend": Cat(["n", "c", "t", "ct"]),
            "measurement_error": Cat([True, False]),
            "time_varying_regression": Cat([True, False]),
            "mle_regression": Cat([True, False]),
            "simple_differencing": Cat([True, False]),
            "enforce_stationarity": Cat([True, False]),
            "enforce_invertibility": Cat([True, False]),
            "hamilton_representation": Cat([True, False]),
            "concentrate_scale": Cat([True, False]),
            "use_exact_diffuse": Cat([True, False]),
        }

        # Remove os parâmetros de order se especificados pelo usuário
        if "order" in self._est_params:
            for p in self._order:
                dist.pop(p)
        if "seasonal_order" in self._est_params or not self._config.sp.get("sp"):
            # Remove os parâmetros sazonais se especificados pelo usuário ou sem sazonalidade
            for p in self._s_order:
                dist.pop(p)

        return dist


class STL(ForecastModel):
    """Decomposição sazonal-tendência usando LOESS.

    STL é uma técnica comumente usada para decompor dados de séries
    temporais em componentes como tendência, sazonalidade e resíduos.

    Os estimadores correspondentes são:

    - [STLForecaster][] para tarefas de previsão.

    Veja também
    --------
    experionml.models:Croston
    experionml.models:ETS
    experionml.models:Theta

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="STL", verbose=2)
    ```

    """

    acronym = "STL"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.trend.STLForecaster"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        # O parâmetro sp deve ser fornecido para STL e ser >=2
        # None só é aceito se y tiver freq no índice, mas o sktime passa array
        return super()._get_est({"sp": self._config.sp.get("sp", 2)} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "seasonal": Int(3, 11, step=2),
            "seasonal_deg": Cat([0, 1]),
            "low_pass_deg": Cat([0, 1]),
            "robust": Cat([True, False]),
        }


class TBATS(ForecastModel):
    """Previsor TBATS com múltiplas sazonalidades.

    TBATS é a sigla para:

    - sazonalidade trigonométrica
    - transformação Box-Cox
    - erros ARMA
    - tendência
    - componentes sazonais

    TBATS foi projetado para prever séries temporais com múltiplos
    períodos sazonais. Por exemplo, dados diários podem ter um padrão
    semanal e também um padrão anual. Dados horários podem ter três
    períodos sazonais: um padrão diário, um semanal e um anual.

    Em BATS, uma [transformação Box-Cox][boxcox] é aplicada à série
    temporal original, e então ela é modelada como uma combinação
    linear de uma tendência suavizada exponencialmente, um componente
    sazonal e um componente ARMA. Os componentes sazonais são modelados
    por funções trigonométricas via séries de Fourier. O TBATS realiza
    algum ajuste de hiperparâmetros, como decidir quais desses
    componentes manter ou descartar, usando AIC.

    Os estimadores correspondentes são:

    - [TBATS][tbatsclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:BATS
    experionml.models:ARIMA
    experionml.models:AutoARIMA

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="TBATS", verbose=2)
    ```

    """

    acronym = "TBATS"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = True
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.tbats.TBATS"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est({"show_warnings": self.warnings != "ignore"} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "use_box_cox": Cat([True, False, None]),
            "use_trend": Cat([True, False, None]),
            "use_damped_trend": Cat([True, False, None]),
            "use_arma_errors": Cat([True, False]),
        }


class Theta(ForecastModel):
    """Método Theta para previsão.

    O método theta é equivalente a um [ExponentialSmoothing][] simples
    com drift. A série é testada quanto à sazonalidade e, se for
    considerada sazonal, é ajustada sazonalmente com uma decomposição
    multiplicativa clássica antes de aplicar o método theta. As
    previsões resultantes são então ressaionalizadas.

    Nos casos em que ExponentialSmoothing resulta em uma previsão
    constante, o previsor theta volta a prever a constante do SES somada
    a uma tendência linear derivada dos dados de treino.

    Os intervalos de previsão são calculados usando o modelo subjacente
    em espaço de estados.

    Os estimadores correspondentes são:

    - [ThetaForecaster][] para tarefas de previsão.

    Veja também
    --------
    experionml.models:Croston
    experionml.models:ExponentialSmoothing
    experionml.models:PolynomialTrend

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    experionml = ExperionMLForecaster(y, random_state=1)
    experionml.run(models="Theta", verbose=2)
    ```

    """

    acronym = "Theta"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.theta.ThetaForecaster"}

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est({"sp": self._config.sp.get("sp", 1)} | params)

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {"deseasonalize": Cat([False, True])}


class VAR(ForecastModel):
    """Autorregressivo vetorial.

    O modelo autorregressivo vetorial (VAR) é um tipo de modelo de série
    temporal multivariada usado para analisar e prever o comportamento
    conjunto de múltiplas variáveis. Em um modelo VAR, cada variável do
    sistema é modelada como uma combinação linear de seus valores
    passados e dos valores passados de todas as outras variáveis do
    sistema. Isso permite capturar as interdependências e relações
    dinâmicas entre as variáveis ao longo do tempo.

    !!! warning
        VAR só oferece suporte a tarefas [multivariadas][].

    Os estimadores correspondentes são:

    - [VAR][varclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:MSTL
    experionml.models:Prophet
    experionml.models:VARMAX

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    experionml = ExperionMLForecaster(X, y=(-1, -2), random_state=1)
    experionml.run(models="VAR", verbose=2)
    ```

    """

    acronym = "VAR"
    handles_missing = False
    uses_exogenous = False
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.var.VAR"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "trend": Cat(["c", "ct", "ctt", "n"]),
            "ic": Cat(["aic", "fpe", "hqic", "bic"]),
        }


class VARMAX(ForecastModel):
    """Média móvel autorregressiva vetorial.

    VARMAX é uma extensão do modelo [VAR][] que incorpora não apenas
    valores defasados das variáveis endógenas, mas também variáveis
    exógenas. Isso permite que modelos VARMAX capturem tanto as
    interdependências entre múltiplas séries temporais quanto a
    influência de fatores externos.

    !!! warning
        VARMAX só oferece suporte a tarefas [multivariadas][].

    Os estimadores correspondentes são:

    - [VARMAX][varmaxclass] para tarefas de previsão.

    Veja também
    --------
    experionml.models:MSTL
    experionml.models:Prophet
    experionml.models:VAR

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_longley

    _, X = load_longley()

    experionml = ExperionMLForecaster(X, y=(-1, -2), random_state=1)
    experionml.run(models="VARMAX", verbose=2)
    ```

    """

    acronym = "VARMAX"
    handles_missing = False
    uses_exogenous = True
    multiple_seasonality = False
    native_multioutput = True
    supports_engines = ("sktime",)

    _estimators: ClassVar[dict[str, str]] = {"forecast": "sktime.forecasting.varmax.VARMAX"}

    _order = ("p", "q")

    def _trial_to_est(self, params: dict[str, Any]) -> dict[str, Any]:
        """Converte os hiperparâmetros do trial em parâmetros do estimador.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros do trial.

        Retorna
        -------
        dict
            Hiperparâmetros do estimador.

        """
        params = super()._trial_to_est(params)

        # Converte os parâmetros no hiperparâmetro 'order'
        if all(p in params for p in self._order):
            params["order"] = [params.pop(p) for p in self._order]

        # Remove os parâmetros de order
        for p in self._order:
            params.pop(p, None)

        return params

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém o estimador do modelo com os parâmetros desempacotados.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        return super()._get_est({"suppress_warnings": self.warnings == "ignore"} | params)

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "p": Int(0, 2),
            "q": Int(0, 2),
            "trend": Cat(["c", "ct", "ctt", "n"]),
            "error_cov_type": Cat(["diagonal", "unstructured"]),
            "measurement_error": Cat([True, False]),
            "enforce_stationarity": Cat([True, False]),
            "enforce_invertibility": Cat([True, False]),
            "cov_type": Cat(["opg", "oim", "approx", "robust", "robust_approx"]),
            "method": Cat(["newton", "nm", "bfgs", "lbfgs", "powell", "cg", "ncg", "basinhopping"]),
            "maxiter": Int(50, 200, step=10),
        }

        # Remove os parâmetros de order se especificados pelo usuário
        if "order" in self._est_params:
            for p in self._order:
                dist.pop(p)

        return dist
