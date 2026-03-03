import logging
import time
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna import Study, Trial
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from rpmeta.config import Config
from rpmeta.model import Model

logger = logging.getLogger(__name__)


class _TransformedPredictor:
    """Stores a native regressor + inverse transform so the visualizer can call
    ``.predict()`` and get results in the original target scale."""

    def __init__(self, regressor: Any, inverse_func: Callable) -> None:
        self._regressor = regressor
        self._inverse_func = inverse_func

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return self._inverse_func(self._regressor.predict(X))


@dataclass
class TrialResult:
    model_name: str
    trial_number: int
    params: dict[str, Any]
    test_score: float
    fit_time: int


@dataclass
class BestModelResult:
    model_name: str
    model_path: Path
    r2: float
    neg_rmse: float
    neg_mae: float
    params: dict[str, Any]
    model: Any = None  # Optional: only set when needed for immediate use


class ModelTrainer(Model):
    def __init__(self, name: str, config: Config) -> None:
        super().__init__(name, config)

        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        self._model_directory = self.config.result_dir / f"{self.name}_{now}"
        self._model_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def param_space(self, trial: Trial) -> dict[str, Any]:
        """Suggest hyperparameters for Optuna trial"""
        ...

    @property
    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Fixed parameters optimized for the desired model, loaded from config"""
        ...

    def _log_model_size(self, regressor: Any) -> None:
        model_file = self._model_directory / self.native_model_filename
        size_bytes = model_file.stat().st_size
        logger.info("Model size on disk: %.2f MB", size_bytes / (1024 * 1024))

    def run_study(
        self,
        X_train: pd.DataFrame,  # noqa: N803
        X_test: pd.DataFrame,  # noqa: N803
        y_train: pd.Series,
        y_test: pd.Series,
        n_trials: int = 200,
    ) -> tuple[Study, list[TrialResult], BestModelResult]:
        """
        Run the Optuna study for hyperparameter tuning.

        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
            y_train (pd.Series): Training target
            y_test (pd.Series): Testing target
            n_trials (int): Number of trials to run

        Returns:
            tuple[Study, list[TrialResult], BestModelResult]:
                - Study object containing the results of the trials
                - List of TrialResult objects for each trial
                - BestModelResult object containing the best model and its parameters
        """
        y_train_t = self.TARGET_FUNC(y_train)

        def objective(trial: Trial) -> float:
            params = self.param_space(trial)
            regressor = self.make_regressor(params)

            start = time.time()
            regressor.fit(X_train, y_train_t)
            fit_time = time.time() - start
            trial.set_user_attr("fit_time", fit_time)

            y_pred = self.INVERSE_FUNC(regressor.predict(X_test))
            rmse = root_mean_squared_error(y_test, y_pred)
            penalty = self.compute_size_penalty(regressor, trial)
            return rmse + penalty

        study = optuna.create_study(
            direction="minimize",
            study_name=f"{self.name}_study",
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        trial_results = []
        for t in study.trials:
            trial_results.append(
                TrialResult(
                    model_name=self.name,
                    trial_number=t.number,
                    params=t.params,
                    # negative value for bigger == better
                    test_score=-t.value,
                    fit_time=int(t.user_attrs.get("fit_time", 0)),
                ),
            )

        best_regressor = self.make_regressor(study.best_trial.params)
        best_regressor.fit(X_train, y_train_t)

        y_pred = self.INVERSE_FUNC(best_regressor.predict(X_test))

        self.save_regressor(best_regressor, self._model_directory)
        self._log_model_size(best_regressor)

        best_result = BestModelResult(
            model_name=self.name,
            model_path=self._model_directory,
            r2=r2_score(y_test, y_pred),
            neg_rmse=-root_mean_squared_error(y_test, y_pred),
            neg_mae=-mean_absolute_error(y_test, y_pred),
            params=study.best_trial.params,
            model=_TransformedPredictor(best_regressor, self.INVERSE_FUNC),
        )
        return study, trial_results, best_result

    def run(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,
    ) -> Path:
        """
        Train the model on suggested hyperparameters and save the model.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target

        Returns:
            Path: Path to the saved model
        """
        regressor = self.make_regressor(self.default_params)
        y_t = self.TARGET_FUNC(y)
        logger.info("Fitting model %s with default parameters: %s", self.name, self.default_params)
        regressor.fit(X, y_t)
        logger.debug("Model fitting complete.")

        self.save_regressor(regressor, self._model_directory)
        self._log_model_size(regressor)
        return self._model_directory
