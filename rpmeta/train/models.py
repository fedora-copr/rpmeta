# a place to play with the models and implement their interface

from typing import Any

from optuna import Trial

from rpmeta.config import Config
from rpmeta.train.base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, config: Config):
        super().__init__("xgboost", use_preprocessor=False, config=config)

    def _make_regressor(self, params: dict[str, int | float | str]):
        from xgboost import XGBRegressor

        return XGBRegressor(
            enable_categorical=True,
            tree_method="hist",
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            objective="reg:squarederror",
            **params,
        )

    @staticmethod
    def param_space(trial: Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10, log=True),
        }

    @staticmethod
    def default_params() -> dict[str, Any]:
        return {
            "n_estimators": 651,
            "learning_rate": 0.2248,
            "max_depth": 8,
            "subsample": 0.9789,
            "colsample_bytree": 0.9835,
            "reg_alpha": 0.8798,
            "reg_lambda": 5.8016,
            "min_child_weight": 1.1275,
        }


class LightGBMModel(BaseModel):
    def __init__(self, config: Config):
        super().__init__("lightgbm", use_preprocessor=False, config=config)

    def _make_regressor(self, params: dict[str, int | float | str]):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1 if self.verbose else -1,
            objective="regression",
            **params,
        )

    @staticmethod
    def param_space(trial: Trial) -> dict[str, Any]:
        max_depth = trial.suggest_int("max_depth", 4, 16)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int(
                "num_leaves",
                int((2**max_depth) * 0.4),
                int((2**max_depth) - 1),
            ),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
            "max_bin": trial.suggest_int("max_bin", 255, 320),
        }

    @staticmethod
    def default_params() -> dict[str, Any]:
        return {
            "n_estimators": 1208,
            "learning_rate": 0.2319,
            "max_depth": 10,
            "num_leaves": 849,
            "min_child_samples": 57,
            "subsample": 0.6354,
            "colsample_bytree": 0.9653,
            "lambda_l1": 0.0005,
            "lambda_l2": 0.0001,
            "max_bin": 282,
        }


def get_all_models(config: Config) -> list[BaseModel]:
    return [
        XGBoostModel(config=config),
        LightGBMModel(config=config),
    ]


def get_all_model_names(config: Config) -> list[str]:
    return [model.name.lower() for model in get_all_models(config)]
