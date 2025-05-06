# a place to play with the models and implement their interface

from typing import Any

from optuna import Trial

from rpmeta.train.base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__("xgboost", use_preprocessor=False)

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

    def param_space(self, trial: Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10, log=True),
        }

    def default_params(self) -> dict[str, Any]:
        return {
            "n_estimators": 1061,
            "learning_rate": 0.2793,
            "max_depth": 9,
            "subsample": 0.9962,
            "colsample_bytree": 0.591,
            "reg_alpha": 0.0001,
            "reg_lambda": 2.0521,
            "min_child_weight": 1.3283,
        }


class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__("lightgbm", use_preprocessor=False)

    def _make_regressor(self, params: dict[str, int | float | str]):
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0,
            **params,
        )

    def param_space(self, trial: Trial) -> dict[str, Any]:
        max_depth = trial.suggest_int("max_depth", 4, 16)
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": max_depth,
            # 0.8 just to prevent overfitting so it has not max leaves to not be leaf wise tree
            "num_leaves": trial.suggest_int("num_leaves", 2, int((2**max_depth) - 1) * 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_float("min_data_in_leaf", 20, 1000, log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
        }

    def default_params(self) -> dict[str, Any]:
        return {
            "n_estimators": 1162,
            "learning_rate": 0.2654,
            "max_depth": 15,
            "num_leaves": 16036,
            "min_child_samples": 20,
            "subsample": 0.6707,
            "colsample_bytree": 0.984,
            "min_data_in_leaf": 92,
            "lambda_l1": 0.0001,
            "lambda_l2": 0.2022,
        }
