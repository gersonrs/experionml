from __future__ import annotations

from copy import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from experionml.basetrainer import BaseTrainer
from experionml.utils.types import (
    int_t,
)
from experionml.utils.utils import (
    ClassMap,
    composed,
    crash,
    lst,
    method_to_log,
)


class Direct(BaseEstimator, BaseTrainer):
    """Direct training approach.

    Fit and evaluate over the models. Contrary to SuccessiveHalving
    and TrainSizing, the direct approach only iterates once over the
    models, using the full dataset.

    See basetrainer.py for a description of the parameters.

    """

    def __init__(
        self,
        models,
        metric,
        est_params,
        n_trials,
        ht_params,
        n_bootstrap,
        parallel,
        errors,
        n_jobs,
        device,
        engine,
        backend,
        memory,
        verbose,
        warnings,
        logger,
        experiment,
        random_state,
    ):
        super().__init__(
            models,
            metric,
            est_params,
            n_trials,
            ht_params,
            n_bootstrap,
            parallel,
            errors,
            n_jobs,
            device,
            engine,
            backend,
            memory,
            verbose,
            warnings,
            logger,
            experiment,
            random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Treina e avalia os modelos.

        Leia mais no [guia do usuário][training].

        Parâmetros
        ----------
        *arrays: sequence of indexables
            Conjunto de treino e conjunto de teste. Formatos permitidos:

            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        """
        self._branches.fill(*self._get_data(arrays))
        self._prepare_parameters()

        self._log("\nTraining " + "=" * 25 + " >>", 1)
        self._log(f"Models: {', '.join(lst(self.models))}", 1)
        self._log(f"Metric: {', '.join(lst(self.metric))}", 1)

        self._core_iteration()


class SuccessiveHalving(BaseEstimator, BaseTrainer):
    """Treina e avalia os modelos no formato [successive halving][].

    See [SuccessiveHalvingClassifier][] or [SuccessiveHalvingRegressor][]
    for a description of the remaining parameters.

    """

    def __init__(
        self,
        models,
        metric,
        skip_runs,
        est_params,
        n_trials,
        ht_params,
        n_bootstrap,
        parallel,
        errors,
        n_jobs,
        device,
        engine,
        backend,
        memory,
        verbose,
        warnings,
        logger,
        experiment,
        random_state,
    ):
        self.skip_runs = skip_runs
        super().__init__(
            models,
            metric,
            est_params,
            n_trials,
            ht_params,
            n_bootstrap,
            parallel,
            errors,
            n_jobs,
            device,
            engine,
            backend,
            memory,
            verbose,
            warnings,
            logger,
            experiment,
            random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Treina e avalia os modelos.

        Leia mais no [guia do usuário][training].

        Parâmetros
        ----------
        *arrays: sequence of indexables
            Conjunto de treino e conjunto de teste. Formatos permitidos:

            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        """
        self._branches.fill(*self._get_data(arrays))
        self._prepare_parameters()

        if self.skip_runs >= len(self._models) // 2 + 1:
            raise ValueError(
                "Invalid value for the skip_runs parameter. Less than one run "
                f"remaining for this choice, got n_runs={len(self._models) // 2 + 1} "
                f"for skip_runs={self.skip_runs}."
            )

        self._log("\nTraining " + "=" * 25 + " >>", 1)
        self._log(f"Metric: {', '.join(lst(self.metric))}", 1)

        run = 0
        models = ClassMap()
        og_models = ClassMap(copy(m) for m in self._models)
        while len(self._models) > 2**self.skip_runs - 1:
            # Create the new set of models for the run
            for m in self._models:
                m._name += str(len(self._models))
                m._train_idx = len(self.train) // len(self._models)

            # Print stats for this subset of the data
            p = round(100.0 / len(self._models))
            self._log(f"\n\nRun: {run} {'=' * 27} >>", 1)
            self._log(f"Models: {', '.join(lst(self.models))}", 1)
            self._log(f"Size of training set: {len(self.train)} ({p}%)", 1)
            self._log(f"Size of test set: {len(self.test)}", 1)

            self._core_iteration()
            models.extend(self._models)

            # Select best models for halving
            best = pd.Series(
                data=[m._best_score() for m in self._models],
                index=[m._group for m in self._models],
                dtype=float,
            ).nlargest(n=len(self._models) // 2, keep="first")

            self._models = ClassMap(copy(m) for m in og_models if m.name in best.index)

            run += 1

        self._models = models  # Restore all models


class TrainSizing(BaseEstimator, BaseTrainer):
    """Treina e avalia os modelos no formato [train sizing][].

    See [TrainSizingClassifier][] or [TrainSizingRegressor][] for a
    description of the remaining parameters.

    """

    def __init__(
        self,
        models,
        metric,
        train_sizes,
        est_params,
        n_trials,
        ht_params,
        n_bootstrap,
        parallel,
        errors,
        n_jobs,
        device,
        engine,
        backend,
        memory,
        verbose,
        warnings,
        logger,
        experiment,
        random_state,
    ):
        self.train_sizes = train_sizes
        super().__init__(
            models,
            metric,
            est_params,
            n_trials,
            ht_params,
            n_bootstrap,
            parallel,
            errors,
            n_jobs,
            device,
            engine,
            backend,
            memory,
            verbose,
            warnings,
            logger,
            experiment,
            random_state,
        )

    @composed(crash, method_to_log)
    def run(self, *arrays):
        """Treina e avalia os modelos.

        Leia mais no [guia do usuário][training].

        Parâmetros
        ----------
        *arrays: sequence of indexables
            Conjunto de treino e conjunto de teste. Formatos permitidos:

            - train, test
            - X_train, X_test, y_train, y_test
            - (X_train, y_train), (X_test, y_test)

        """
        self._branches.fill(*self._get_data(arrays))
        self._prepare_parameters()

        self._log("\nTraining " + "=" * 25 + " >>", 1)
        self._log(f"Metric: {', '.join(lst(self.metric))}", 1)

        # Convert integer train_sizes to sequence
        if isinstance(self.train_sizes, int_t):
            self.train_sizes = np.linspace(1 / self.train_sizes, 1.0, self.train_sizes)

        models = ClassMap()
        og_models = ClassMap(copy(m) for m in self._models)
        for run, size in enumerate(self.train_sizes):
            # Select the fraction of the data to use in this run
            if size <= 1:
                frac = round(size, 2)
                train_idx = int(size * len(self.train))
            else:
                frac = round(size / len(self.train), 2)
                train_idx = size

            for m in self._models:
                m._name += str(frac).replace(".", "")  # Add frac to the name
                m._train_idx = train_idx

            # Print stats for this subset of the data
            p = round(train_idx * 100.0 / len(self.branch.train))
            self._log(f"\n\nRun: {run} {'=' * 27} >>", 1)
            self._log(f"Models: {', '.join(lst(self.models))}", 1)
            self._log(f"Size of training set: {train_idx} ({p}%)", 1)
            self._log(f"Size of test set: {len(self.test)}", 1)

            self._core_iteration()
            models.extend(self._models)

            # Create next models for sizing
            self._models = ClassMap(copy(m) for m in og_models)

        self._models = models  # Restore original models
