from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution as Cat
from optuna.distributions import FloatDistribution as Float
from optuna.distributions import IntDistribution as Int
from optuna.exceptions import TrialPruned
from optuna.trial import Trial

from experionml.basemodel import ClassRegModel
from experionml.utils.types import Pandas, Predictor
from experionml.utils.utils import (
    CatBMetric,
    Goal,
    LGBMetric,
    XGBMetric,
    check_dependency,
)


class AdaBoost(ClassRegModel):
    """Boosting adaptativo.

    O AdaBoost é um metaestimador que começa ajustando um
    classificador/regressor no conjunto de dados original e, em
    seguida, ajusta cópias adicionais do algoritmo no mesmo conjunto,
    mas com os pesos das instâncias ajustados de acordo com o erro da
    predição atual.

    Os estimadores correspondentes são:

    - [AdaBoostClassifier][] para tarefas de classificação.
    - [AdaBoostRegressor][] para tarefas de regressão.

    Leia mais na [documentação][adabdocs] do sklearn.

    See Also
    --------
    experionml.models:GradientBoostingMachine
    experionml.models:RandomForest
    experionml.models:XGBoost

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="AdaB", metric="f1", verbose=2)
    ```

    """

    acronym = "AdaB"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.ensemble.AdaBoostClassifier",
        "regression": "sklearn.ensemble.AdaBoostRegressor",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "n_estimators": Int(50, 500, step=10),
            "learning_rate": Float(0.01, 10, log=True),
        }

        if self._goal is Goal.regression:
            dist["loss"] = Cat(["linear", "square", "exponential"])

        return dist


class AutomaticRelevanceDetermination(ClassRegModel):
    """Determinação automática de relevância.

    A determinação automática de relevância é muito semelhante ao
    [BayesianRidge][], mas pode levar a coeficientes mais esparsos.
    Ajusta os pesos de um modelo de regressão usando um prior ARD.
    Assume-se que os pesos do modelo de regressão seguem distribuições
    gaussianas.

    Os estimadores correspondentes são:

    - [ARDRegression][] para tarefas de regressão.

    Leia mais na [documentação][arddocs] do sklearn.

    See Also
    --------
    experionml.models:BayesianRidge
    experionml.models:GaussianProcess
    experionml.models:LeastAngleRegression

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    experionml = ExperionMLRegressor(X, y, random_state=1)
    experionml.run(models="ARD", metric="r2", verbose=2)
    ```

    """

    acronym = "ARD"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {"regression": "sklearn.linear_model.ARDRegression"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "max_iter": Int(100, 1000, step=10),
            "alpha_1": Float(1e-4, 1, log=True),
            "alpha_2": Float(1e-4, 1, log=True),
            "lambda_1": Float(1e-4, 1, log=True),
            "lambda_2": Float(1e-4, 1, log=True),
        }


class Bagging(ClassRegModel):
    """Modelo bagging com árvore de decisão como estimador base.

    Bagging usa um metaestimador em conjunto que ajusta preditores base
    em subconjuntos aleatórios do conjunto de dados original e depois
    agrega suas predições individuais, por votação ou média, para formar
    uma predição final. Esse metaestimador costuma ser usado como forma
    de reduzir a variância de um estimador caixa-preta ao introduzir
    aleatoriedade em seu processo de construção e então formar um
    ensemble a partir disso.

    Os estimadores correspondentes são:

    - [BaggingClassifier][] para tarefas de classificação.
    - [BaggingRegressor][] para tarefas de regressão.

    Leia mais na [documentação][bagdocs] do sklearn.

    See Also
    --------
    experionml.models:DecisionTree
    experionml.models:LogisticRegression
    experionml.models:RandomForest

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="Bag", metric="f1", verbose=2)
    ```

    """

    acronym = "Bag"
    handles_missing = True
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.ensemble.BaggingClassifier",
        "regression": "sklearn.ensemble.BaggingRegressor",
    }

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "n_estimators": Int(10, 500, step=10),
            "max_samples": Float(0.5, 1.0, step=0.1),
            "max_features": Float(0.5, 1.0, step=0.1),
            "bootstrap": Cat([True, False]),
            "bootstrap_features": Cat([True, False]),
        }


class BayesianRidge(ClassRegModel):
    """Regressão ridge bayesiana.

    Técnicas de regressão bayesiana podem ser usadas para incluir
    parâmetros de regularização no procedimento de estimação: o
    parâmetro de regularização não é definido rigidamente, mas ajustado
    aos dados disponíveis.

    Os estimadores correspondentes são:

    - [BayesianRidge][bayesianridgeclass] para tarefas de regressão.

    Leia mais na [documentação][brdocs] do sklearn.

    See Also
    --------
    experionml.models:AutomaticRelevanceDetermination
    experionml.models:GaussianProcess
    experionml.models:LeastAngleRegression

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    experionml = ExperionMLRegressor(X, y, random_state=1)
    experionml.run(models="BR", metric="r2", verbose=2)
    ```

    """

    acronym = "BR"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {"regression": "sklearn.linear_model.BayesianRidge"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "max_iter": Int(100, 1000, step=10),
            "alpha_1": Float(1e-4, 1, log=True),
            "alpha_2": Float(1e-4, 1, log=True),
            "lambda_1": Float(1e-4, 1, log=True),
            "lambda_2": Float(1e-4, 1, log=True),
        }


class BernoulliNB(ClassRegModel):
    """Naive Bayes de Bernoulli.

    O BernoulliNB implementa o algoritmo Naive Bayes para modelos
    multivariados de Bernoulli. Assim como o [MultinomialNB][], esse
    classificador é adequado para dados discretos. A diferença é que,
    enquanto o MNB trabalha com contagens de ocorrência, o BNB foi
    projetado para features binárias ou booleanas.

    Os estimadores correspondentes são:

    - [BernoulliNB][bernoullinbclass] para tarefas de classificação.

    Leia mais na [documentação][bnbdocs] do sklearn.

    See Also
    --------
    experionml.models:ComplementNB
    experionml.models:CategoricalNB
    experionml.models:MultinomialNB

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="BNB", metric="f1", verbose=2)
    ```

    """

    acronym = "BNB"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "cuml")

    _estimators: ClassVar[dict[str, str]] = {"classification": "sklearn.naive_bayes.BernoulliNB"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "alpha": Float(0.01, 10, log=True),
            "fit_prior": Cat([True, False]),
        }


class CatBoost(ClassRegModel):
    """Máquina de boosting CatBoost.

    CatBoost é um método de aprendizado de máquina baseado em gradient
    boosting sobre árvores de decisão. Principais vantagens do
    CatBoost:

        - Qualidade superior quando comparado a outros modelos GBDT em
            muitos conjuntos de dados.
        - Velocidade de predição entre as melhores da categoria.

    Os estimadores correspondentes são:

    - [CatBoostClassifier][] para tarefas de classificação.
    - [CatBoostRegressor][] para tarefas de regressão.

    Leia mais na [documentação][catbdocs] do CatBoost.

        !!! warning
                * O CatBoost seleciona os pesos obtidos na melhor avaliação no
                    conjunto de teste após o treinamento. Isso significa que, por
                    padrão, existe um pequeno vazamento de dados no conjunto de
                    teste. Use o parâmetro `use_best_model=False` para evitar esse
                    comportamento ou utilize um [conjunto holdout][data-sets] para
                    avaliar o estimador final.
                * A [validação durante o treinamento][] e o [pruning][] são
                    desativados quando `#!python device="gpu"`.

    !!! note
        O ExperionML usa o parâmetro `n_estimators` do CatBoost em vez
        de `iterations` para indicar o número de árvores a ajustar. Isso
        é feito para manter uma nomenclatura consistente com os modelos
        [XGBoost][] e [LightGBM][].

    See Also
    --------
    experionml.models:GradientBoostingMachine
    experionml.models:LightGBM
    experionml.models:XGBoost

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="CatB", metric="f1", verbose=2)
    ```

    """

    acronym = "CatB"
    handles_missing = True
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = "n_estimators"
    supports_engines = ("catboost",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "catboost.CatBoostClassifier",
        "regression": "catboost.CatBoostRegressor",
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Obtém os hiperparâmetros do trial.

        Parâmetros
        ----------
        trial: [Trial][]
            Trial atual.

        Retorna
        -------
        dict
            Hiperparâmetros do trial.

        """
        params = super()._get_parameters(trial)

        if self._get_param("bootstrap_type", params) == "Bernoulli":
            if "bagging_temperature" in params:
                params["bagging_temperature"] = None
        elif self._get_param("bootstrap_type", params) == "Bayesian":
            if "subsample" in params:
                params["subsample"] = None

        return params

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém a instância do estimador.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        eval_metric = None
        if getattr(self, "_metric", None) and not self._gpu:
            eval_metric = CatBMetric(self._metric[0], task=self.task)

        default = {
            "eval_metric": eval_metric,
            "train_dir": "",
            "allow_writing_files": False,
            "thread_count": self.n_jobs,
            "task_type": "GPU" if self._gpu else "CPU",
            "devices": str(self._device_id),
            "verbose": False,
        }

        return super()._get_est(default | params)

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: tuple[pd.DataFrame, Pandas],
        validation: tuple[pd.DataFrame, Pandas] | None = None,
        trial: Trial | None = None,
    ):
        """Ajusta o estimador e executa validação durante o treinamento.

        Parâmetros
        ----------
        estimator: Predictor
            Instância a ser ajustada.

        data: tuple
            Dados de treino no formato (X, y).

        validation: tuple or None
            Dados de validação no formato (X, y). Se None, nenhuma
            validação é executada.

        trial: [Trial][] or None
            Trial ativo durante o ajuste de hiperparâmetros.

        Retorna
        -------
        Predictor
            Instância ajustada.

        """
        check_dependency("optuna_integration")
        from optuna_integration import CatBoostPruningCallback

        params = self._est_params_fit.copy()

        callbacks = params.pop("callbacks", [])
        if trial and len(self._metric) == 1 and not self._gpu:
            callbacks.append(cb := CatBoostPruningCallback(trial, "CatBMetric"))

        # gpu implementation fails if callbacks!=None
        estimator.fit(*data, eval_set=validation, callbacks=callbacks or None, **params)

        if not self._gpu:
            if validation:
                # Create evals attribute with train and validation scores
                m = self._metric[0].name
                evals = estimator.evals_result_
                self._evals[f"{m}_train"] = evals["learn"]["CatBMetric"]
                self._evals[f"{m}_test"] = evals["validation"]["CatBMetric"]

            if trial and len(self._metric) == 1 and cb._pruned:
                # Add the pruned step to the output
                step = len(self.evals[f"{m}_train"])
                steps = estimator.get_params()[self.validation]
                trial.params[self.validation] = f"{step}/{steps}"

                trial.set_user_attr("estimator", estimator)
                raise TrialPruned(cb._message)

        return estimator

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "n_estimators": Int(20, 500, step=10),
            "learning_rate": Float(0.01, 1.0, log=True),
            "max_depth": Cat([None, *range(1, 17)]),
            "min_child_samples": Int(1, 30),
            "bootstrap_type": Cat(["Bayesian", "Bernoulli"]),
            "bagging_temperature": Float(0, 10),
            "subsample": Float(0.5, 1.0, step=0.1),
            "reg_lambda": Float(0.001, 100, log=True),
        }


class CategoricalNB(ClassRegModel):
    """Naive Bayes categórico.

    O Naive Bayes categórico implementa o algoritmo Naive Bayes para
    features categóricas.

    Os estimadores correspondentes são:

    - [CategoricalNB][categoricalnbclass] para tarefas de classificação.

    Leia mais na [documentação][catnbdocs] do sklearn.

    See Also
    --------
    experionml.models:BernoulliNB
    experionml.models:ComplementNB
    experionml.models:GaussianNB

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    import numpy as np

    rng = np.random.default_rng()
    X = rng.integers(5, size=(100, 100))
    y = rng.integers(2, size=100)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="CatNB", metric="f1", verbose=2)
    ```

    """

    acronym = "CatNB"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "cuml")

    _estimators: ClassVar[dict[str, str]] = {"classification": "sklearn.naive_bayes.CategoricalNB"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "alpha": Float(0.01, 10, log=True),
            "fit_prior": Cat([True, False]),
        }


class ComplementNB(ClassRegModel):
    """Naive Bayes complementar.

    O classificador Complement Naive Bayes foi projetado para corrigir
    as "suposições severas" feitas pelo classificador padrão
    [MultinomialNB][]. Ele é particularmente adequado para conjuntos de
    dados desbalanceados.

    Os estimadores correspondentes são:

    - [ComplementNB][complementnbclass] para tarefas de classificação.

    Leia mais na [documentação][cnbdocs] do sklearn.

    See Also
    --------
    experionml.models:BernoulliNB
    experionml.models:CategoricalNB
    experionml.models:MultinomialNB

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="CNB", metric="f1", verbose=2)
    ```

    """

    acronym = "CNB"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "cuml")

    _estimators: ClassVar[dict[str, str]] = {"classification": "sklearn.naive_bayes.ComplementNB"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "alpha": Float(0.01, 10, log=True),
            "fit_prior": Cat([True, False]),
            "norm": Cat([True, False]),
        }


class DecisionTree(ClassRegModel):
    """Árvore de decisão única.

    Um classificador/regressor de árvore de decisão única.

    Os estimadores correspondentes são:

    - [DecisionTreeClassifier][] para tarefas de classificação.
    - [DecisionTreeRegressor][] para tarefas de regressão.

    Leia mais na [documentação][treedocs] do sklearn.

    See Also
    --------
    experionml.models:ExtraTree
    experionml.models:ExtraTrees
    experionml.models:RandomForest

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="Tree", metric="f1", verbose=2)
    ```

    """

    acronym = "Tree"
    handles_missing = True
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.tree.DecisionTreeClassifier",
        "regression": "sklearn.tree.DecisionTreeRegressor",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        if self._goal is Goal.classification:
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error", "friedman_mse", "poisson"]

        return {
            "criterion": Cat(criterion),
            "splitter": Cat(["best", "random"]),
            "max_depth": Cat([None, *range(1, 17)]),
            "min_samples_split": Int(2, 20),
            "min_samples_leaf": Int(1, 20),
            "max_features": Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            "ccp_alpha": Float(0, 0.035, step=0.005),
        }


class Dummy(ClassRegModel):
    """Classificador/regressor dummy.

    Ao trabalhar com aprendizado supervisionado, uma verificação
    básica de sanidade consiste em comparar o estimador com regras
    simples. Os métodos de predição ignoram completamente os dados de
    entrada. Não use este modelo em problemas reais. Use-o apenas como
    uma linha de base simples para comparar com outros modelos.

    Os estimadores correspondentes são:

    - [DummyClassifier][] para tarefas de classificação.
    - [DummyRegressor][] para tarefas de regressão.

    Leia mais na [documentação][dummydocs] do sklearn.

    See Also
    --------
    experionml.models:DecisionTree
    experionml.models:ExtraTree
    experionml.models:NaiveForecaster

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="Dummy", metric="f1", verbose=2)
    ```

    """

    acronym = "Dummy"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.dummy.DummyClassifier",
        "regression": "sklearn.dummy.DummyRegressor",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "strategy": Cat(["most_frequent", "prior", "stratified", "uniform"]),
            "quantile": Float(0, 1.0, step=0.1),
        }

        if self._goal is Goal.classification:
            dist.pop("quantile")
        else:
            dist["strategy"] = Cat(["mean", "median", "quantile"])

        return dist


class ElasticNet(ClassRegModel):
    """Regressão linear com regularização elasticnet.

    Mínimos quadrados lineares com regularização l1 e l2.

    Os estimadores correspondentes são:

    - [ElasticNet][elasticnetreg] para tarefas de regressão.

    Leia mais na [documentação][endocs] do sklearn.

    See Also
    --------
    experionml.models:Lasso
    experionml.models:OrdinaryLeastSquares
    experionml.models:Ridge

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    experionml = ExperionMLRegressor(X, y, random_state=1)
    experionml.run(models="EN", metric="r2", verbose=2)
    ```

    """

    acronym = "EN"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "sklearnex", "cuml")

    _estimators: ClassVar[dict[str, str]] = {"regression": "sklearn.linear_model.ElasticNet"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "alpha": Float(1e-3, 10, log=True),
            "l1_ratio": Float(0.1, 0.9, step=0.1),
            "selection": Cat(["cyclic", "random"]),
        }


class ExtraTree(ClassRegModel):
    """Árvore extremamente aleatorizada.

    Árvores extremamente aleatorizadas diferem das árvores de decisão
    clássicas na forma como são construídas. Ao buscar a melhor divisão
    para separar as amostras de um nó em dois grupos, divisões
    aleatórias são geradas para cada uma das variáveis selecionadas
    aleatoriamente por max_features, e a melhor entre elas é escolhida.
    Quando max_features é igual a 1, isso equivale a construir uma
    árvore de decisão totalmente aleatória.

    Os estimadores correspondentes são:

    - [ExtraTreeClassifier][] para tarefas de classificação.
    - [ExtraTreeRegressor][] para tarefas de regressão.

    Leia mais na [documentação][treedocs] do sklearn.

    See Also
    --------
    experionml.models:DecisionTree
    experionml.models:ExtraTrees
    experionml.models:RandomForest

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="ETree", metric="f1", verbose=2)
    ```

    """

    acronym = "ETree"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.tree.ExtraTreeClassifier",
        "regression": "sklearn.tree.ExtraTreeRegressor",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        if self._goal is Goal.classification:
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error"]

        return {
            "criterion": Cat(criterion),
            "splitter": Cat(["random", "best"]),
            "max_depth": Cat([None, *range(1, 17)]),
            "min_samples_split": Int(2, 20),
            "min_samples_leaf": Int(1, 20),
            "max_features": Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            "ccp_alpha": Float(0, 0.035, step=0.005),
        }


class ExtraTrees(ClassRegModel):
    """Floresta extremamente aleatorizada.

    Extra-Trees usa um metaestimador que ajusta várias árvores de
    decisão aleatorizadas (também conhecidas como
    [extra-trees][extratree]) em diferentes subamostras do conjunto de
    dados e utiliza média para melhorar a acurácia preditiva e controlar
    o sobreajuste.

    Os estimadores correspondentes são:

    - [ExtraTreesClassifier][] para tarefas de classificação.
    - [ExtraTreesRegressor][] para tarefas de regressão.

    Leia mais na [documentação][etdocs] do sklearn.

    See Also
    --------
    experionml.models:DecisionTree
    experionml.models:ExtraTree
    experionml.models:RandomForest

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="ET", metric="f1", verbose=2)
    ```

    """

    acronym = "ET"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.ensemble.ExtraTreesClassifier",
        "regression": "sklearn.ensemble.ExtraTreesRegressor",
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Obtém os hiperparâmetros do trial.

        Parameters
        ----------
        trial: [Trial][]
            Trial atual.

        Returns
        -------
        dict
            Hiperparâmetros do trial.

        """
        params = super()._get_parameters(trial)

        if not self._get_param("bootstrap", params) and "max_samples" in params:
            params["max_samples"] = None

        return params

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        if self._goal is Goal.classification:
            criterion = ["gini", "entropy"]
        else:
            criterion = ["squared_error", "absolute_error"]

        return {
            "n_estimators": Int(10, 500, step=10),
            "criterion": Cat(criterion),
            "max_depth": Cat([None, *range(1, 17)]),
            "min_samples_split": Int(2, 20),
            "min_samples_leaf": Int(1, 20),
            "max_features": Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            "bootstrap": Cat([True, False]),
            "max_samples": Cat([None, 0.5, 0.6, 0.7, 0.8, 0.9]),
            "ccp_alpha": Float(0, 0.035, step=0.005),
        }


class GaussianNB(ClassRegModel):
    """Naive Bayes gaussiano.

    O Gaussian Naive Bayes implementa o algoritmo Naive Bayes para
    classificação. Assume-se que a verossimilhança das variáveis segue
    uma distribuição gaussiana.

    Os estimadores correspondentes são:

    - [GaussianNB][gaussiannbclass] para tarefas de classificação.

    Leia mais na [documentação][gnbdocs] do sklearn.

    See Also
    --------
    experionml.models:BernoulliNB
    experionml.models:CategoricalNB
    experionml.models:ComplementNB

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="GNB", metric="f1", verbose=2)
    ```

    """

    acronym = "GNB"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "cuml")

    _estimators: ClassVar[dict[str, str]] = {"classification": "sklearn.naive_bayes.GaussianNB"}


class GaussianProcess(ClassRegModel):
    """Processo gaussiano.

        Processos gaussianos são um método genérico de aprendizado
        supervisionado projetado para resolver problemas de regressão e
        classificação probabilística. As vantagens dos processos
        gaussianos são:

        * A predição interpola as observações.
        * A predição é probabilística (gaussiana), de forma que é possível
            calcular intervalos de confiança empíricos e decidir com base
            neles se a predição deve ser reajustada em alguma região de
            interesse.

        As desvantagens dos processos gaussianos incluem:

        * Eles não são esparsos, isto é, usam toda a informação de
            amostras e variáveis para realizar a previsão.
        * Eles perdem eficiência em espaços de alta dimensionalidade,
            especialmente quando o número de variáveis ultrapassa algumas
            dezenas.

        Os estimadores correspondentes são:

        - [GaussianProcessClassifier][] para tarefas de classificação.
        - [GaussianProcessRegressor][] para tarefas de regressão.

        Leia mais na [documentação][gpdocs] do sklearn.

    See Also
    --------
    experionml.models:GaussianNB
    experionml.models:LinearDiscriminantAnalysis
    experionml.models:PassiveAggressive

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="GP", metric="f1", verbose=2)
    ```

    """

    acronym = "GP"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.gaussian_process.GaussianProcessClassifier",
        "regression": "sklearn.gaussian_process.GaussianProcessRegressor",
    }


class GradientBoostingMachine(ClassRegModel):
    """Máquina de gradient boosting.

    Uma Gradient Boosting Machine constrói um modelo aditivo em estágios
    progressivos, permitindo a otimização de funções de perda
    diferenciáveis arbitrárias. Em cada estágio, árvores de regressão
    `n_classes_` são ajustadas sobre o gradiente negativo da função de
    perda, por exemplo, log loss binária ou multiclasse. A classificação
    binária é um caso especial em que apenas uma única árvore de
    regressão é induzida.

    Os estimadores correspondentes são:

    - [GradientBoostingClassifier][] para tarefas de classificação.
    - [GradientBoostingRegressor][] para tarefas de regressão.

    Leia mais na [documentação][gbmdocs] do sklearn.

    !!! tip
        [HistGradientBoosting][] é uma variante muito mais rápida desse
        algoritmo para conjuntos de dados intermediários (n_samples >= 10k).

    See Also
    --------
    experionml.models:CatBoost
    experionml.models:HistGradientBoosting
    experionml.models:LightGBM

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="GBM", metric="f1", verbose=2)
    ```

    """

    acronym = "GBM"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.ensemble.GradientBoostingClassifier",
        "regression": "sklearn.ensemble.GradientBoostingRegressor",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "loss": Cat(["log_loss", "exponential"]),
            "learning_rate": Float(0.01, 1.0, log=True),
            "n_estimators": Int(10, 500, step=10),
            "subsample": Float(0.5, 1.0, step=0.1),
            "criterion": Cat(["friedman_mse", "squared_error"]),
            "min_samples_split": Int(2, 20),
            "min_samples_leaf": Int(1, 20),
            "max_depth": Int(1, 21),
            "max_features": Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            "ccp_alpha": Float(0, 0.035, step=0.005),
        }

        # Evita usar 'task' quando a classe é inicializada sem branches
        if "_branch" in self.__dict__ and self.task.is_multiclass:
            dist.pop("loss")  # Multiclasse só aceita log_loss
        elif self._goal is Goal.regression:
            dist["loss"] = Cat(["squared_error", "absolute_error", "huber", "quantile"])
            dist["alpha"] = Float(0.1, 0.9, step=0.1)

        return dist


class HuberRegression(ClassRegModel):
    """Regressor de Huber.

    Huber é um modelo de regressão linear robusto a outliers. Ele faz
    com que a função de perda não seja fortemente influenciada pelos
    outliers sem ignorar completamente seus efeitos.

    Os estimadores correspondentes são:

    - [HuberRegressor][] para tarefas de regressão.

    Leia mais na [documentação][huberdocs] do sklearn.

    See Also
    --------
    experionml.models:AutomaticRelevanceDetermination
    experionml.models:LeastAngleRegression
    experionml.models:OrdinaryLeastSquares

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    experionml = ExperionMLRegressor(X, y, random_state=1)
    experionml.run(models="Huber", metric="r2", verbose=2)
    ```

    """

    acronym = "Huber"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {"regression": "sklearn.linear_model.HuberRegressor"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "epsilon": Float(1, 10, log=True),
            "max_iter": Int(50, 500, step=10),
            "alpha": Float(1e-4, 1, log=True),
        }


class HistGradientBoosting(ClassRegModel):
    """Máquina de gradient boosting baseada em histogramas.

    Esta Gradient Boosting Machine baseada em histogramas é muito mais
    rápida do que a [GradientBoostingMachine][] padrão para grandes
    conjuntos de dados (n_samples>=10k). Essa variação primeiro agrupa
    as amostras de entrada em bins inteiros, o que reduz muito o número
    de pontos de divisão a considerar e permite ao algoritmo aproveitar
    estruturas de dados baseadas em inteiros (histogramas) em vez de
    depender de valores contínuos ordenados ao construir as árvores.

    Os estimadores correspondentes são:

    - [HistGradientBoostingClassifier][] para tarefas de classificação.
    - [HistGradientBoostingRegressor][] para tarefas de regressão.

    Leia mais na [documentação][hgbmdocs] do sklearn.

    See Also
    --------
    experionml.models:CatBoost
    experionml.models:GradientBoostingMachine
    experionml.models:XGBoost

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="hGBM", metric="f1", verbose=2)
    ```

    """

    acronym = "hGBM"
    handles_missing = True
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.ensemble.HistGradientBoostingClassifier",
        "regression": "sklearn.ensemble.HistGradientBoostingRegressor",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "loss": Cat(["squared_error", "absolute_error", "poisson", "quantile", "gamma"]),
            "quantile": Float(0, 1, step=0.1),
            "learning_rate": Float(0.01, 1.0, log=True),
            "max_iter": Int(10, 500, step=10),
            "max_leaf_nodes": Int(10, 50),
            "max_depth": Cat([None, *range(1, 17)]),
            "min_samples_leaf": Int(10, 30),
            "l2_regularization": Float(0, 1.0, step=0.1),
        }

        if self._goal is Goal.classification:
            dist.pop("loss")
            dist.pop("quantile")

        return dist


class KNearestNeighbors(ClassRegModel):
    """K-vizinhos mais próximos.

    K-Nearest Neighbors, como o nome indica, implementa o voto dos
    k-vizinhos mais próximos. Para regressão, o alvo é previsto por
    interpolação local dos alvos associados aos vizinhos mais próximos
    no conjunto de treinamento.

    Os estimadores correspondentes são:

    - [KNeighborsClassifier][] para tarefas de classificação.
    - [KNeighborsRegressor][] para tarefas de regressão.

    Leia mais na [documentação][knndocs] do sklearn.

    See Also
    --------
    experionml.models:LinearDiscriminantAnalysis
    experionml.models:QuadraticDiscriminantAnalysis
    experionml.models:RadiusNearestNeighbors

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="KNN", metric="f1", verbose=2)
    ```

    """

    acronym = "KNN"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
    validation = None
    supports_engines = ("sklearn", "sklearnex", "cuml")

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.neighbors.KNeighborsClassifier",
        "regression": "sklearn.neighbors.KNeighborsRegressor",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "n_neighbors": Int(1, 100),
            "weights": Cat(["uniform", "distance"]),
            "algorithm": Cat(["auto", "ball_tree", "kd_tree", "brute"]),
            "leaf_size": Int(20, 40),
            "p": Int(1, 2),
        }

        if self._gpu:
            dist.pop("algorithm")  # Só 'brute' é suportado
            if self.engine.estimator == "cuml":
                dist.pop("weights")  # Só 'uniform' é suportado
                dist.pop("leaf_size")
                dist.pop("p")

        return dist


class Lasso(ClassRegModel):
    """Regressão linear com regularização lasso.

    Mínimos quadrados lineares com regularização l1.

    Os estimadores correspondentes são:

    - [Lasso][lassoreg] para tarefas de regressão.

    Leia mais na [documentação][lassodocs] do sklearn.

    See Also
    --------
    experionml.models:ElasticNet
    experionml.models:OrdinaryLeastSquares
    experionml.models:Ridge

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    experionml = ExperionMLRegressor(X, y, random_state=1)
    experionml.run(models="Lasso", metric="r2", verbose=2)
    ```

    """

    acronym = "Lasso"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "sklearnex", "cuml")

    _estimators: ClassVar[dict[str, str]] = {"regression": "sklearn.linear_model.Lasso"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "alpha": Float(1e-3, 10, log=True),
            "selection": Cat(["cyclic", "random"]),
        }


class LeastAngleRegression(ClassRegModel):
    """Regressão de mínimo ângulo.

    Least-Angle Regression é um algoritmo de regressão para dados de
    alta dimensionalidade. Lars é semelhante à regressão stepwise
    progressiva. A cada passo, encontra a variável mais correlacionada
    com o alvo. Quando há múltiplas variáveis com a mesma correlação,
    em vez de continuar na mesma variável, segue em uma direção
    equiangular entre elas.

    Os estimadores correspondentes são:

    - [Lars][] para tarefas de regressão.

    Leia mais na [documentação][larsdocs] do sklearn.

    See Also
    --------
    experionml.models:BayesianRidge
    experionml.models:HuberRegression
    experionml.models:OrdinaryLeastSquares

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    experionml = ExperionMLRegressor(X, y, random_state=1)
    experionml.run(models="Lars", metric="r2", verbose=2)
    ```

    """

    acronym = "Lars"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {"regression": "sklearn.linear_model.Lars"}


class LightGBM(ClassRegModel):
    """Máquina de gradient boosting LightGBM.

    LightGBM é um modelo de gradient boosting que usa algoritmos de
    aprendizado baseados em árvores. Ele foi projetado para ser
    distribuído e eficiente, com as seguintes vantagens:

    - Maior velocidade de treinamento e maior eficiência.
    - Menor uso de memória.
    - Melhor acurácia.
    - Capacidade de lidar com dados em grande escala.

    Os estimadores correspondentes são:

    - [LGBMClassifier][] para tarefas de classificação.
    - [LGBMRegressor][] para tarefas de regressão.

    Leia mais na [documentação][lgbdocs] do LightGBM.

    !!! info
        O uso da [aceleração por GPU][estimator-acceleration] do
        LightGBM requer [dependências adicionais de software][lgb_gpu].

    See Also
    --------
    experionml.models:CatBoost
    experionml.models:GradientBoostingMachine
    experionml.models:XGBoost

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="LGB", metric="f1", verbose=2)
    ```

    """

    acronym = "LGB"
    handles_missing = True
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = "n_estimators"
    supports_engines = ("lightgbm",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "lightgbm.sklearn.LGBMClassifier",
        "regression": "lightgbm.sklearn.LGBMRegressor",
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
        # Mapeamento customizado do lightgbm para warnings
        # PYTHONWarnings não funciona porque eles vêm do código C/C++ para stdout
        warns = {"always": 2, "default": 1, "once": 0, "error": 0, "ignore": -1}

        default = {
            "verbose": warns.get(self.warnings, -1),
            "device": "gpu" if self._gpu else "cpu",
            "gpu_device_id": self._device_id or -1,
        }

        return super()._get_est(default | params)

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: tuple[pd.DataFrame, Pandas],
        validation: tuple[pd.DataFrame, Pandas] | None = None,
        trial: Trial | None = None,
    ):
        """Ajusta o estimador e executa validação durante o treinamento.

            Parâmetros
        ----------
        estimator: Predictor
                Instância a ajustar.

        data: tuple
                Dados de treino no formato (X, y).

        validation: tuple or None
                Dados de validação no formato (X, y). Se None, nenhuma
                validação é executada.

        trial: [Trial][] or None
                Trial ativo durante o ajuste de hiperparâmetros.

            Retorna
        -------
        Predictor
                Instância ajustada.

        """
        check_dependency("optuna_integration")
        from lightgbm.callback import log_evaluation
        from optuna_integration import LightGBMPruningCallback

        m = self._metric[0].name
        params = self._est_params_fit.copy()

        callbacks = [*params.pop("callbacks", []), log_evaluation(-1)]
        if trial and len(self._metric) == 1:
            callbacks.append(LightGBMPruningCallback(trial, m, "valid_1"))

        eval_metric = None
        if getattr(self, "_metric", None):
            eval_metric = LGBMetric(self._metric[0], task=self.task)

        try:
            estimator.fit(
                *data,
                eval_set=[data, validation] if validation else None,
                eval_metric=params.pop("eval_metric", eval_metric),
                callbacks=callbacks,
                **params,
            )
        except TrialPruned as ex:
            trial = cast(Trial, trial)  # If pruned, trial can't be None

            # Adiciona o passo podado à saída
            step = str(ex).split(" ")[-1][:-1]
            steps = estimator.get_params()[self.validation]
            trial.params[self.validation] = f"{step}/{steps}"

            trial.set_user_attr("estimator", estimator)
            raise ex

        if validation:
            # Cria o atributo evals com pontuações de treino e validação
            self._evals[f"{m}_train"] = estimator.evals_result_["training"][m]
            self._evals[f"{m}_test"] = estimator.evals_result_["valid_1"][m]

        return estimator

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "n_estimators": Int(20, 500, step=10),
            "learning_rate": Float(0.01, 1.0, log=True),
            "max_depth": Int(-1, 17, step=2),
            "num_leaves": Int(20, 40),
            "min_child_weight": Float(1e-4, 100, log=True),
            "min_child_samples": Int(1, 30),
            "subsample": Float(0.5, 1.0, step=0.1),
            "colsample_bytree": Float(0.4, 1.0, step=0.1),
            "reg_alpha": Float(1e-4, 100, log=True),
            "reg_lambda": Float(1e-4, 100, log=True),
        }


class LinearDiscriminantAnalysis(ClassRegModel):
    """Análise discriminante linear.

    Linear Discriminant Analysis é um classificador com fronteira de
    decisão linear, gerada pelo ajuste de densidades condicionais por
    classe aos dados e pelo uso da regra de Bayes. O modelo ajusta uma
    densidade gaussiana a cada classe, assumindo que todas compartilham
    a mesma matriz de covariância.

    Os estimadores correspondentes são:

    - [LinearDiscriminantAnalysis][ldaclassifier] para tarefas de classificação.

    Leia mais na [documentação][ldadocs] do sklearn.

    See Also
    --------
    experionml.models:LogisticRegression
    experionml.models:RadiusNearestNeighbors
    experionml.models:QuadraticDiscriminantAnalysis

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="LDA", metric="f1", verbose=2)
    ```

    """

    acronym = "LDA"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis"
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Obtém os hiperparâmetros do trial.

        Parâmetros
        ----------
        trial: [Trial][]
            Trial atual.

        Retorna
        -------
        dict
            Hiperparâmetros do trial.

        """
        params = super()._get_parameters(trial)

        if self._get_param("solver", params) == "svd" and "shrinkage" in params:
            params["shrinkage"] = None

        return params

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "solver": Cat(["svd", "lsqr", "eigen"]),
            "shrinkage": Cat([None, "auto", 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        }


class LinearSVM(ClassRegModel):
    """Máquina de vetores de suporte linear.

    Semelhante a [SupportVectorMachine][], mas com kernel linear.
    Implementado com liblinear em vez de libsvm, o que oferece mais
    flexibilidade na escolha de penalidades e funções de perda e tende
    a escalar melhor para grandes quantidades de amostras.

    Os estimadores correspondentes são:

    - [LinearSVC][] para tarefas de classificação.
    - [LinearSVR][] para tarefas de regressão.

    Leia mais na [documentação][svmdocs] do sklearn.

    See Also
    --------
    experionml.models:KNearestNeighbors
    experionml.models:StochasticGradientDescent
    experionml.models:SupportVectorMachine

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="lSVM", metric="f1", verbose=2)
    ```

    """

    acronym = "lSVM"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "cuml")

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.svm.LinearSVC",
        "regression": "sklearn.svm.LinearSVR",
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Obtém os hiperparâmetros do trial.

        Parâmetros
        ----------
        trial: [Trial][]
            Trial atual.

        Retorna
        -------
        dict
            Hiperparâmetros do trial.

        """
        params = super()._get_parameters(trial)

        if self._goal is Goal.classification:
            if self._get_param("loss", params) == "hinge":
                # A regularização l1 não pode ser combinada com hinge
                if "penalty" in params:
                    params["penalty"] = "l2"
                # A regularização l2 não pode ser combinada com hinge quando dual=False
                if "dual" in params:
                    params["dual"] = True
            elif self._get_param("loss", params) == "squared_hinge":
                # A regularização l1 não pode ser combinada com squared_hinge quando dual=True
                if self._get_param("penalty", params) == "l1" and "dual" in params:
                    params["dual"] = False
        elif self._get_param("loss", params) == "epsilon_insensitive" and "dual" in params:
            params["dual"] = True

        return params

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém a instância do estimador.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        if self.engine.estimator == "cuml" and self._goal is Goal.classification:
            return super()._get_est({"probability": True} | params)
        else:
            return super()._get_est(params)

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist: dict[str, BaseDistribution] = {}
        if self._goal is Goal.classification:
            dist["penalty"] = Cat(["l1", "l2"])
            dist["loss"] = Cat(["hinge", "squared_hinge"])
        else:
            dist["loss"] = Cat(["epsilon_insensitive", "squared_epsilon_insensitive"])

        dist["C"] = Float(1e-3, 100, log=True)
        dist["dual"] = Cat([True, False])

        if self.engine.estimator == "cuml":
            dist.pop("dual")

        return dist


class LogisticRegression(ClassRegModel):
    """Regressão logística.

    Regressão logística, apesar do nome, é um modelo linear para
    classificação, e não para regressão. Também é conhecida na
    literatura como regressão logit, classificação de máxima entropia
    (MaxEnt) ou classificador log-linear. Nesse modelo, as
    probabilidades que descrevem os possíveis resultados de um único
    trial são modeladas usando uma função logística.

    Os estimadores correspondentes são:

    - [LogisticRegression][] para tarefas de classificação.

    Leia mais na [documentação][lrdocs] do sklearn.

    See Also
    --------
    experionml.models:GaussianProcess
    experionml.models:LinearDiscriminantAnalysis
    experionml.models:PassiveAggressive

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="RF", metric="f1", verbose=2)
    ```

    """

    acronym = "LR"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "sklearnex", "cuml")

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.linear_model.LogisticRegression"
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Obtém os hiperparâmetros do trial.

        Parâmetros
        ----------
        trial: [Trial][]
            Trial atual.

        Retorna
        -------
        dict
            Hiperparâmetros do trial.

        """
        params = super()._get_parameters(trial)

        # Limitações nas combinações entre penalty e solver
        penalty = self._get_param("penalty", params)
        solver = self._get_param("solver", params)
        cond_1 = penalty is None and solver == "liblinear"
        cond_2 = penalty == "l1" and solver not in ("liblinear", "saga")
        cond_3 = penalty == "elasticnet" and solver != "saga"

        if cond_1 or cond_2 or cond_3 and "penalty" in params:
            params["penalty"] = "l2"  # Altera para o valor padrão

        return params

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "penalty": Cat([None, "l1", "l2", "elasticnet"]),
            "C": Float(1e-3, 100, log=True),
            "solver": Cat(["lbfgs", "newton-cg", "liblinear", "sag", "saga"]),
            "max_iter": Int(100, 1000, step=10),
            "l1_ratio": Float(0, 1.0, step=0.1),
        }

        if self._gpu:
            if self.engine.estimator == "cuml":
                dist["penalty"] = Cat(["none", "l1", "l2", "elasticnet"])
                dist.pop("solver")  # Só `qn` é suportado
            elif self.engine.estimator == "sklearnex":
                dist["penalty"] = Cat(["none", "l1", "elasticnet"])
                dist["solver"] = Cat(["lbfgs", "liblinear", "sag", "saga"])
        elif self.engine.estimator == "sklearnex":
            dist["solver"] = Cat(["lbfgs", "newton-cg"])

        return dist


class MultiLayerPerceptron(ClassRegModel):
    """Perceptron multicamadas.

    Multi-layer Perceptron é um algoritmo de aprendizado supervisionado
    que aprende uma função a partir do treinamento em um conjunto de
    dados. Dado um conjunto de variáveis e um alvo, ele pode aprender
    uma função aproximadora não linear para classificação ou regressão.
    Difere da regressão logística porque, entre a camada de entrada e a
    de saída, pode haver uma ou mais camadas não lineares, chamadas de
    camadas ocultas.

    Os estimadores correspondentes são:

    - [MLPClassifier][] para tarefas de classificação.
    - [MLPRegressor][] para tarefas de regressão.

    Leia mais na [documentação][mlpdocs] do sklearn.

    See Also
    --------
    experionml.models:PassiveAggressive
    experionml.models:Perceptron
    experionml.models:StochasticGradientDescent

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="MLP", metric="f1", verbose=2)
    ```

    """

    acronym = "MLP"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = False
    validation = "max_iter"
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.neural_network.MLPClassifier",
        "regression": "sklearn.neural_network.MLPRegressor",
    }

    def _trial_to_est(self, params: dict[str, Any]) -> dict[str, Any]:
        """Converte os hiperparâmetros do trial em parâmetros do estimador.

        Parameters
        ----------
        params: dict
            Trial's hyperparameters.

        Returns
        -------
        dict
            Estimator's hyperparameters.

        """
        params = super()._trial_to_est(params)

        hidden_layer_sizes = [
            value
            for param in [p for p in sorted(params) if p.startswith("hidden_layer")]
            if (value := params.pop(param))  # Neurons should be >0
        ]

        if hidden_layer_sizes:
            params["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        return params

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        dist = {
            "hidden_layer_1": Int(10, 100),
            "hidden_layer_2": Int(0, 100),
            "hidden_layer_3": Int(0, 10),
            "activation": Cat(["identity", "logistic", "tanh", "relu"]),
            "solver": Cat(["lbfgs", "sgd", "adam"]),
            "alpha": Float(1e-4, 0.1, log=True),
            "batch_size": Cat(["auto", 8, 16, 32, 64, 128, 256]),
            "learning_rate": Cat(["constant", "invscaling", "adaptive"]),
            "learning_rate_init": Float(1e-3, 0.1, log=True),
            "power_t": Float(0.1, 0.9, step=0.1),
            "max_iter": Int(50, 500, step=10),
        }

        # Drop layers if user specifies sizes
        if "hidden_layer_sizes" in self._est_params:
            return {k: v for k, v in dist.items() if "hidden_layer" not in k}
        else:
            return dist


class MultinomialNB(ClassRegModel):
    """Naive Bayes multinomial.

    MultinomialNB implementa o algoritmo Naive Bayes para dados com
    distribuição multinomial e é uma das duas variantes clássicas de
    Naive Bayes usadas em classificação de texto, em que os dados são
    tipicamente representados como contagens de palavras, embora vetores
    tf-idf também funcionem bem na prática.

    Os estimadores correspondentes são:

    - [MultinomialNB][multinomialnbclass] para tarefas de classificação.

    Leia mais na [documentação][mnbdocs] do sklearn.

    See Also
    --------
    experionml.models:BernoulliNB
    experionml.models:ComplementNB
    experionml.models:GaussianNB

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="MNB", metric="f1", verbose=2)
    ```

    """

    acronym = "MNB"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "cuml")

    _estimators: ClassVar[dict[str, str]] = {"classification": "sklearn.naive_bayes.MultinomialNB"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "alpha": Float(0.01, 10, log=True),
            "fit_prior": Cat([True, False]),
        }


class OrdinaryLeastSquares(ClassRegModel):
    """Regressão linear.

    Ordinary Least Squares é simplesmente regressão linear sem
    regularização. Ajusta um modelo linear com coeficientes
    `w=(w1, ..., wp)` para minimizar a soma residual dos quadrados entre
    os alvos observados no conjunto de dados e os alvos previstos pela
    aproximação linear.

    Os estimadores correspondentes são:

    - [LinearRegression][] para tarefas de regressão.

    Leia mais na [documentação][olsdocs] do sklearn.

    See Also
    --------
    experionml.models:ElasticNet
    experionml.models:Lasso
    experionml.models:Ridge

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    experionml = ExperionMLRegressor(X, y, random_state=1)
    experionml.run(models="OLS", metric="r2", verbose=2)
    ```

    """

    acronym = "OLS"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "sklearnex", "cuml")

    _estimators: ClassVar[dict[str, str]] = {"regression": "sklearn.linear_model.LinearRegression"}


class OrthogonalMatchingPursuit(ClassRegModel):
    """Pursuit ortogonal de matching.

    Orthogonal Matching Pursuit implementa o algoritmo OMP para
    aproximar o ajuste de um modelo linear com restrições impostas ao
    número de coeficientes não nulos.

    Os estimadores correspondentes são:

    - [OrthogonalMatchingPursuit][] para tarefas de regressão.

    Leia mais na [documentação][ompdocs] do sklearn.

    See Also
    --------
    experionml.models:Lasso
    experionml.models:LeastAngleRegression
    experionml.models:OrdinaryLeastSquares

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    experionml = ExperionMLRegressor(X, y, random_state=1)
    experionml.run(models="OMP", metric="r2", verbose=2)
    ```

    """

    acronym = "OMP"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "regression": "sklearn.linear_model.OrthogonalMatchingPursuit"
    }


class PassiveAggressive(ClassRegModel):
    """Passivo-agressivo.

    Os algoritmos passivo-agressivos são uma família de algoritmos para
    aprendizado em grande escala. Eles são semelhantes ao
    [Perceptron][] por não exigirem taxa de aprendizado. No entanto, ao
    contrário do [Perceptron][], incluem um parâmetro de regularização
    `C`.

    Os estimadores correspondentes são:

    - [PassiveAggressiveClassifier][] para tarefas de classificação.
    - [PassiveAggressiveRegressor][] para tarefas de regressão.

    Leia mais na [documentação][padocs] do sklearn.

    See Also
    --------
    experionml.models:MultiLayerPerceptron
    experionml.models:Perceptron
    experionml.models:StochasticGradientDescent

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="PA", metric="f1", verbose=2)
    ```

    """

    acronym = "PA"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = "max_iter"
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.linear_model.PassiveAggressiveClassifier",
        "regression": "sklearn.linear_model.PassiveAggressiveRegressor",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        if self._goal is Goal.classification:
            loss = ["hinge", "squared_hinge"]
        else:
            loss = ["epsilon_insensitive", "squared_epsilon_insensitive"]

        return {
            "C": Float(1e-3, 100, log=True),
            "max_iter": Int(500, 1500, step=50),
            "loss": Cat(loss),
            "average": Cat([True, False]),
        }


class Perceptron(ClassRegModel):
    """Classificação linear com perceptron.

    O Perceptron é um algoritmo simples de classificação adequado para
    aprendizado em grande escala. Por padrão:

    * Não exige taxa de aprendizado.
    * Não é regularizado.
    * Atualiza o modelo apenas em erros.

    Essa última característica implica que o Perceptron é um pouco mais
    rápido de treinar do que [StochasticGradientDescent][] com perda
    hinge e que os modelos resultantes são mais esparsos.

    Os estimadores correspondentes são:

    - [Perceptron][percclassifier] para tarefas de classificação.

    Leia mais na [documentação][percdocs] do sklearn.

    See Also
    --------
    experionml.models:MultiLayerPerceptron
    experionml.models:PassiveAggressive
    experionml.models:StochasticGradientDescent

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="Perc", metric="f1", verbose=2)
    ```

    """

    acronym = "Perc"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = "max_iter"
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {"classification": "sklearn.linear_model.Perceptron"}

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {
            "penalty": Cat([None, "l2", "l1", "elasticnet"]),
            "alpha": Float(1e-4, 10, log=True),
            "l1_ratio": Float(0.1, 0.9, step=0.1),
            "max_iter": Int(500, 1500, step=50),
            "eta0": Float(1e-2, 10, log=True),
        }


class QuadraticDiscriminantAnalysis(ClassRegModel):
    """Análise discriminante quadrática.

    Quadratic Discriminant Analysis é um classificador com fronteira de
    decisão quadrática, gerada pelo ajuste de densidades condicionais
    por classe aos dados e pelo uso da regra de Bayes. O modelo ajusta
    uma densidade gaussiana a cada classe, assumindo que todas
    compartilham a mesma matriz de covariância.

    Os estimadores correspondentes são:

    - [QuadraticDiscriminantAnalysis][qdaclassifier] para tarefas de classificação.

    Leia mais na [documentação][ldadocs] do sklearn.

    See Also
    --------
    experionml.models:LinearDiscriminantAnalysis
    experionml.models:LogisticRegression
    experionml.models:RadiusNearestNeighbors

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="QDA", metric="f1", verbose=2)
    ```

    """

    acronym = "QDA"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = False
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis"
    }

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        return {"reg_param": Float(0, 1.0, step=0.1)}


class RadiusNearestNeighbors(ClassRegModel):
    """Vizinhos mais próximos por raio.

        Radius Nearest Neighbors implementa o voto dos vizinhos mais
        próximos, em que os vizinhos são selecionados dentro de um raio
        definido. Para regressão, o alvo é previsto por interpolação local
        dos alvos associados aos vizinhos mais próximos no conjunto de
        treinamento.

    !!! warning
                * O parâmetro `radius` deve ser ajustado aos dados em uso, caso
                    contrário o modelo terá baixo desempenho.
                * Se outliers forem detectados, o estimador levanta uma exceção,
                    a menos que `est_params={"outlier_label": "most_frequent"}` seja usado.

        Os estimadores correspondentes são:

        - [RadiusNeighborsClassifier][] para tarefas de classificação.
        - [RadiusNeighborsRegressor][] para tarefas de regressão.

        Leia mais na [documentação][knndocs] do sklearn.

    See Also
    --------
    experionml.models:KNearestNeighbors
    experionml.models:LinearDiscriminantAnalysis
    experionml.models:QuadraticDiscriminantAnalysis

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(
        models="RNN",
        metric="f1",
        est_params={"outlier_label": "most_frequent"},
        verbose=2,
    )
    ```

    """

    acronym = "RNN"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
    validation = None
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.neighbors.RadiusNeighborsClassifier",
        "regression": "sklearn.neighbors.RadiusNeighborsRegressor",
    }

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "radius": Float(1e-2, 100),
            "weights": Cat(["uniform", "distance"]),
            "algorithm": Cat(["auto", "ball_tree", "kd_tree", "brute"]),
            "leaf_size": Int(20, 40),
            "p": Int(1, 2),
        }


class RandomForest(ClassRegModel):
    """Floresta aleatória.

    Random forests são um método de aprendizado em ensemble que opera
    construindo uma grande quantidade de árvores de decisão durante o
    treinamento e produzindo como saída a classe modal das árvores
    individuais, no caso de classificação, ou a média das previsões, no
    caso de regressão. Random forests corrigem a tendência das árvores
    de decisão de sobreajustar ao conjunto de treino.

    Os estimadores correspondentes são:

    - [RandomForestClassifier][] para tarefas de classificação.
    - [RandomForestRegressor][] para tarefas de regressão.

    Leia mais na [documentação][adabdocs] do sklearn.

    !!! warning
        A implementação do cuML de [RandomForestClassifier][cumlrf] só
        suporta previsões com dtype `float32`. Converta todos os dtypes
        antes de chamar o método [run][experionmlclassifier-run] do
        experionml para evitar exceções.

    See Also
    --------
    experionml.models:DecisionTree
    experionml.models:ExtraTrees
    experionml.models:HistGradientBoosting

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="RF", metric="f1", verbose=2)
    ```

    """

    acronym = "RF"
    handles_missing = False
    needs_scaling = False
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = True
    validation = None
    supports_engines = ("sklearn", "sklearnex", "cuml")

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.ensemble.RandomForestClassifier",
        "regression": "sklearn.ensemble.RandomForestRegressor",
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Obtém os hiperparâmetros do trial.

        Parâmetros
        ----------
        trial: [Trial][]
            Trial atual.

        Retorna
        -------
        dict
            Hiperparâmetros do trial.

        """
        params = super()._get_parameters(trial)

        if not self._get_param("bootstrap", params) and "max_samples" in params:
            params["max_samples"] = None

        return params

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        if self._goal is Goal.classification:
            criterion = ["gini", "entropy"]
        else:
            if self.engine.estimator == "cuml":
                criterion = ["mse", "poisson", "gamma", "inverse_gaussian"]
            else:
                criterion = ["squared_error", "absolute_error", "poisson"]

        dist = {
            "n_estimators": Int(10, 500, step=10),
            "criterion": Cat(criterion),
            "max_depth": Cat([None, *range(1, 17)]),
            "min_samples_split": Int(2, 20),
            "min_samples_leaf": Int(1, 20),
            "max_features": Cat([None, "sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9]),
            "bootstrap": Cat([True, False]),
            "max_samples": Cat([None, 0.5, 0.6, 0.7, 0.8, 0.9]),
            "ccp_alpha": Float(0, 0.035, step=0.005),
        }

        if self.engine.estimator == "sklearnex":
            dist.pop("criterion")
            dist.pop("ccp_alpha")
        elif self.engine.estimator == "cuml":
            dist["split_criterion"] = dist.pop("criterion")
            dist["max_depth"] = Int(1, 17)
            dist["max_features"] = Cat(["sqrt", "log2", 0.5, 0.6, 0.7, 0.8, 0.9])
            dist["max_samples"] = Float(0.5, 0.9, step=0.1)
            dist.pop("ccp_alpha")

        return dist


class Ridge(ClassRegModel):
    """Mínimos quadrados lineares com regularização l2.

    Se for classificador, primeiro converte os valores do alvo para
    {-1, 1} e então trata o problema como uma tarefa de regressão.

    Os estimadores correspondentes são:

    - [RidgeClassifier][] para tarefas de classificação.
    - [Ridge][ridgeregressor] para tarefas de regressão.

    Leia mais na [documentação][ridgedocs] do sklearn.

    !!! warning
        Os engines `sklearnex` e `cuml` só estão disponíveis para
        tarefas de regressão.

    See Also
    --------
    experionml.models:BayesianRidge
    experionml.models:ElasticNet
    experionml.models:Lasso

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True)

    experionml = ExperionMLRegressor(X, y, random_state=1)
    experionml.run(models="Ridge", metric="r2", verbose=2)
    ```

    """

    acronym = "Ridge"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = True
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "sklearnex", "cuml")

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.linear_model.RidgeClassifier",
        "regression": "sklearn.linear_model.Ridge",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "alpha": Float(1e-3, 10, log=True),
            "solver": Cat(["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
        }

        if self._goal is Goal.regression:
            if self.engine.estimator == "sklearnex":
                dist.pop("solver")  # Só aceita 'auto'
            elif self.engine.estimator == "cuml":
                dist["solver"] = Cat(["eig", "svd", "cd"])

        return dist


class StochasticGradientDescent(ClassRegModel):
    """Descida de gradiente estocástica.

    Stochastic Gradient Descent é uma abordagem simples, porém muito
    eficiente, para ajustar classificadores e regressores lineares sob
    funções de perda convexas. Embora o SGD exista há muito tempo na
    comunidade de aprendizado de máquina, só recentemente recebeu muita
    atenção no contexto de aprendizado em grande escala.

    Os estimadores correspondentes são:

    - [SGDClassifier][] para tarefas de classificação.
    - [SGDRegressor][] para tarefas de regressão.

    Leia mais na [documentação][sgddocs] do sklearn.

    See Also
    --------
    experionml.models:MultiLayerPerceptron
    experionml.models:PassiveAggressive
    experionml.models:SupportVectorMachine

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="SGD", metric="f1", verbose=2)
    ```

    """

    acronym = "SGD"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = "max_iter"
    supports_engines = ("sklearn",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.linear_model.SGDClassifier",
        "regression": "sklearn.linear_model.SGDRegressor",
    }

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        loss = [
            "hinge",
            "log_loss",
            "modified_huber",
            "squared_hinge",
            "perceptron",
            "squared_error",
            "huber",
            "epsilon_insensitive",
            "squared_epsilon_insensitive",
        ]

        return {
            "loss": Cat(loss if self._goal is Goal.classification else loss[-4:]),
            "penalty": Cat([None, "l1", "l2", "elasticnet"]),
            "alpha": Float(1e-4, 1.0, log=True),
            "l1_ratio": Float(0.1, 0.9, step=0.1),
            "max_iter": Int(500, 1500, step=50),
            "epsilon": Float(1e-4, 1.0, log=True),
            "learning_rate": Cat(["constant", "invscaling", "optimal", "adaptive"]),
            "eta0": Float(1e-2, 10, log=True),
            "power_t": Float(0.1, 0.9, step=0.1),
            "average": Cat([True, False]),
        }


class SupportVectorMachine(ClassRegModel):
    """Máquina de vetores de suporte.

    A implementação de Support Vector Machine se baseia no libsvm. O
    tempo de ajuste cresce pelo menos quadraticamente com o número de
    amostras e pode se tornar impraticável acima de dezenas de milhares
    de amostras. Para conjuntos de dados grandes, considere usar um
    modelo [LinearSVM][] ou [StochasticGradientDescent][] em vez disso.

    Os estimadores correspondentes são:

    - [SVC][] para tarefas de classificação.
    - [SVR][] para tarefas de regressão.

    Leia mais na [documentação][svmdocs] do sklearn.

    See Also
    --------
    experionml.models:LinearSVM
    experionml.models:MultiLayerPerceptron
    experionml.models:StochasticGradientDescent

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="SVM", metric="f1", verbose=2)
    ```

    """

    acronym = "SVM"
    handles_missing = False
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = None
    supports_engines = ("sklearn", "sklearnex", "cuml")

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "sklearn.svm.SVC",
        "regression": "sklearn.svm.SVR",
    }

    def _get_parameters(self, trial: Trial) -> dict:
        """Obtém os hiperparâmetros do trial.

        Parâmetros
        ----------
        trial: [Trial][]
            Trial atual.

        Retorna
        -------
        dict
            Hiperparâmetros do trial.

        """
        params = super()._get_parameters(trial)

        if self._get_param("kernel", params) == "poly" and "gamma" in params:
            params["gamma"] = "scale"  # Crashes in combination with "auto"

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
        if self.engine.estimator == "cuml" and self._goal is Goal.classification:
            return super()._get_est({"probability": True} | params)
        else:
            return super()._get_est(params)

    def _get_distributions(self) -> Mapping[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Retorna
        -------
        dict
            Distribuições de hiperparâmetros.

        """
        dist = {
            "C": Float(1e-3, 100, log=True),
            "kernel": Cat(["linear", "poly", "rbf", "sigmoid"]),
            "degree": Int(2, 5),
            "gamma": Cat(["scale", "auto"]),
            "coef0": Float(-1.0, 1.0),
            "epsilon": Float(1e-3, 100, log=True),
            "shrinking": Cat([True, False]),
        }

        if self.engine.estimator == "cuml":
            dist.pop("epsilon")
            dist.pop("shrinking")
        elif self._goal is Goal.classification:
            dist.pop("epsilon")

        return dist


class XGBoost(ClassRegModel):
    """Gradient boosting extremo.

    XGBoost é um modelo distribuído de gradient boosting otimizado,
    projetado para ser altamente eficiente, flexível e portável. Ele
    fornece tree boosting paralelo que resolve muitos problemas de
    ciência de dados de forma rápida e precisa.

    Os estimadores correspondentes são:

    - [XGBClassifier][] para tarefas de classificação.
    - [XGBRegressor][] para tarefas de regressão.

    Leia mais na [documentação][xgbdocs] do XGBoost.

    See Also
    --------
    experionml.models:CatBoost
    experionml.models:GradientBoostingMachine
    experionml.models:LightGBM

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    experionml = ExperionMLClassifier(X, y, random_state=1)
    experionml.run(models="XGB", metric="f1", verbose=2)
    ```

    """

    acronym = "XGB"
    handles_missing = True
    needs_scaling = True
    accepts_sparse = True
    native_multilabel = False
    native_multioutput = False
    validation = "n_estimators"
    supports_engines = ("xgboost",)

    _estimators: ClassVar[dict[str, str]] = {
        "classification": "xgboost.XGBClassifier",
        "regression": "xgboost.XGBRegressor",
    }

    @property
    def trials(self) -> pd.DataFrame:
        """Visão geral dos resultados dos trials.

        Esta propriedade só está disponível para modelos que executaram
        [ajuste de hiperparâmetros][]. Todas as durações estão em
        segundos. As colunas incluem:

        - **[param_name]:** Valor do parâmetro usado neste trial.
        - **estimator:** Estimador usado neste trial.
        - **[metric_name]:** Pontuação da métrica no trial.
        - **[best_metric_name]:** Melhor pontuação até o momento no estudo.
        - **time_trial:** Duração do trial.
        - **time_ht:** Duração do ajuste de hiperparâmetros.
        - **state:** Estado do trial (COMPLETE, PRUNED, FAIL).

        """
        trials = super().trials

        # O XGBoost sempre minimiza a métrica, então inverte o sinal
        for met in self._metric.keys():
            trials[met] = trials.apply(
                lambda row: -row[met] if row["state"] == "PRUNED" else row[met], axis=1
            )

        return trials

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
        eval_metric = None
        if getattr(self, "_metric", None):
            eval_metric = XGBMetric(self._metric[0], task=self.task)

        default = {"eval_metric": eval_metric, "device": self.device, "verbosity": 0}
        return super()._get_est(default | params)

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: tuple[pd.DataFrame, Pandas],
        validation: tuple[pd.DataFrame, Pandas] | None = None,
        trial: Trial | None = None,
    ):
        """Ajusta o estimador e executa validação durante o treinamento.

        Parâmetros
        ----------
        estimator: Predictor
            Instância a ajustar.

        data: tuple
            Dados de treino no formato (X, y).

        validation: tuple or None
            Dados de validação no formato (X, y). Se None, nenhuma
            validação é executada.

        trial: [Trial][] or None
            Trial ativo durante o ajuste de hiperparâmetros.

        Retorna
        -------
        Predictor
            Instância ajustada.

        """
        check_dependency("optuna_integration")
        from optuna_integration import XGBoostPruningCallback

        m = self._metric[0].name
        params = self._est_params_fit.copy()

        callbacks = params.pop("callbacks", [])
        if trial and len(self._metric) == 1:
            callbacks.append(XGBoostPruningCallback(trial, f"validation_1-{m}"))

        try:
            estimator.set_params(callbacks=callbacks)
            estimator.fit(
                *data,
                eval_set=[data, validation] if validation else None,
                verbose=params.get("verbose", False),
                **params,
            )
        except TrialPruned as ex:
            trial = cast(Trial, trial)  # If pruned, trial can't be None

            # Adiciona o passo podado à saída
            step = str(ex).split(" ")[-1][:-1]
            steps = estimator.get_params()[self.validation]
            trial.params[self.validation] = f"{step}/{steps}"

            trial.set_user_attr("estimator", estimator)
            raise ex

        if validation:
            # Cria o atributo evals com pontuações de treino e validação
            # Negativo porque a função é minimizada
            results = estimator.evals_result()
            self._evals[f"{m}_train"] = np.negative(results["validation_0"][m])
            self._evals[f"{m}_test"] = np.negative(results["validation_1"][m])

        return estimator

    @staticmethod
    def _get_distributions() -> dict[str, BaseDistribution]:
        """Obtém as distribuições predefinidas de hiperparâmetros.

        Returns
        -------
        dict
            Hyperparameter distributions.

        """
        return {
            "n_estimators": Int(20, 500, step=10),
            "learning_rate": Float(0.01, 1.0, log=True),
            "max_depth": Int(1, 20),
            "gamma": Float(0, 1.0),
            "min_child_weight": Int(1, 10),
            "subsample": Float(0.5, 1.0, step=0.1),
            "colsample_bytree": Float(0.4, 1.0, step=0.1),
            "reg_alpha": Float(1e-4, 100, log=True),
            "reg_lambda": Float(1e-4, 100, log=True),
        }
