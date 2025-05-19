import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from optuna import Study, Trial
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rpmeta.config import Config
from rpmeta.constants import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from rpmeta.helpers import save_joblib

logger = logging.getLogger(__name__)


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
    model: Any
    r2: float
    neg_rmse: float
    neg_mae: float
    params: dict[str, Any]


class Model(ABC):
    """
    Defines the interface for a model to be able to perform training and hyperparameter tuning.
    """

    def __init__(
        self,
        name: str,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            name (str): Name of the model in lowercase
            n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
            random_state (int): Random state for reproducibility
        """
        self.name = name.lower()
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def create_pipeline(self, params: dict[str, int | float | str]) -> Pipeline:
        """Return a sklearn Pipeline (including preprocessing and regressor)

        Args:
            params (dict): Dictionary of parameters for the regressor

        Returns:
            Pipeline: Sklearn pipeline with preprocessor and regressor
        """
        ...

    @staticmethod
    @abstractmethod
    def param_space(trial: Trial) -> dict[str, Any]:
        """Suggest hyperparameters for Optuna trial"""
        ...

    @staticmethod
    @abstractmethod
    def default_params() -> dict[str, Any]:
        """Fixed parameters optimized for the desired model"""
        ...


class BaseModel(Model):
    def __init__(self, name: str, config: Config, use_preprocessor: bool = True) -> None:
        super().__init__(name=name)
        self.config = config
        self._use_preprocessor = use_preprocessor

        self.preprocessor = None
        if self._use_preprocessor:
            logger.debug("Creating preprocessor...")
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), NUMERICAL_FEATURES),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                ],
            )

    def create_pipeline(self, params: dict[str, int | float | str]) -> Pipeline:
        reg = self._make_regressor(params)
        ttr = TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)
        if self._use_preprocessor:
            return Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("regressor", ttr),
                ],
            )

        return ttr

    @abstractmethod
    def _make_regressor(self, params: dict[str, int | float | str]) -> Any:
        """Instantiate the base regressor (without preprocessing)"""
        ...

    @staticmethod
    @abstractmethod
    def param_space(trial: Trial) -> dict[str, Any]: ...

    @staticmethod
    @abstractmethod
    def default_params() -> dict[str, Any]: ...

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

        def objective(trial: Trial) -> float:
            params = self.param_space(trial)
            pipeline = self.create_pipeline(params)

            start = time.time()
            pipeline.fit(X_train, y_train)
            fit_time = time.time() - start
            trial.set_user_attr("fit_time", fit_time)

            y_pred = pipeline.predict(X_test)
            return root_mean_squared_error(y_test, y_pred)

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

        # refit best
        best_pipeline = self.create_pipeline(study.best_trial.params)

        best_pipeline.fit(X_train, y_train)
        y_pred = best_pipeline.predict(X_test)

        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        save_joblib(best_pipeline, self.config.result_dir, f"{self.name}_{now}")

        best_result = BestModelResult(
            model_name=self.name,
            # TODO: rather use the path to the model in the results dir, this consumes
            # a lot of memory if the model is large
            model=best_pipeline,
            r2=r2_score(y_test, y_pred),
            neg_rmse=-root_mean_squared_error(y_test, y_pred),
            neg_mae=-mean_absolute_error(y_test, y_pred),
            params=study.best_trial.params,
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
        params = self.default_params()
        pipeline = self.create_pipeline(params)
        logger.info(f"Fitting model {self.name} with default parameters: {params}")
        pipeline.fit(X, y)
        logger.debug("Model fitting complete.")

        now = time.strftime("%Y-%m-%d_%H-%M-%S")
        return save_joblib(pipeline, self.config.result_dir, f"{self.name}_{now}")
