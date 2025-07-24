# a place to play with the models and implement their interface

from typing import Any

from optuna import Trial

from rpmeta.config import Config
from rpmeta.train.base import Model


class XGBoostModel(Model):
    def __init__(self, config: Config):
        super().__init__("xgboost", use_preprocessor=False, config=config)

    def _make_regressor(self, params: dict[str, int | float | str]):
        from xgboost import XGBRegressor

        return XGBRegressor(
            enable_categorical=True,
            tree_method="hist",
            n_jobs=self.config.model.n_jobs,
            random_state=self.config.model.random_state,
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

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.config.model.xgboost.n_estimators,
            "learning_rate": self.config.model.xgboost.learning_rate,
            "max_depth": self.config.model.xgboost.max_depth,
            "subsample": self.config.model.xgboost.subsample,
            "colsample_bytree": self.config.model.xgboost.colsample_bytree,
            "reg_alpha": self.config.model.xgboost.reg_alpha,
            "reg_lambda": self.config.model.xgboost.reg_lambda,
            "min_child_weight": self.config.model.xgboost.min_child_weight,
        }


class LightGBMModel(Model):
    def __init__(self, config: Config):
        super().__init__("lightgbm", use_preprocessor=False, config=config)

    def _make_regressor(self, params: dict[str, int | float | str]):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_jobs=self.config.model.n_jobs,
            random_state=self.config.model.random_state,
            verbose=1 if self.config.model.verbose else -1,
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

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.config.model.lightgbm.n_estimators,
            "learning_rate": self.config.model.lightgbm.learning_rate,
            "max_depth": self.config.model.lightgbm.max_depth,
            "num_leaves": self.config.model.lightgbm.num_leaves,
            "min_child_samples": self.config.model.lightgbm.min_child_samples,
            "subsample": self.config.model.lightgbm.subsample,
            "colsample_bytree": self.config.model.lightgbm.colsample_bytree,
            "lambda_l1": self.config.model.lightgbm.lambda_l1,
            "lambda_l2": self.config.model.lightgbm.lambda_l2,
            "max_bin": self.config.model.lightgbm.max_bin,
        }


def get_all_models(config: Config) -> list[Model]:
    return [
        XGBoostModel(config=config),
        LightGBMModel(config=config),
    ]


def get_all_model_names() -> list[str]:
    empty_config = Config()
    return [model.name for model in get_all_models(config=empty_config)]
