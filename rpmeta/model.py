import logging
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from rpmeta.config import Config
from rpmeta.constants import ModelEnum
from rpmeta.regressor import TransformedTargetRegressor

if TYPE_CHECKING:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor


logger = logging.getLogger(__name__)


class Model(ABC):
    def __init__(self, name: str, config: Config) -> None:
        self.name = name.lower()
        self.config = config

    def create_regressor(self, params: dict[str, int | float | str]) -> TransformedTargetRegressor:
        """
        Return a regressor with the given parameters.

        Args:
            params (dict): Dictionary of parameters for the regressor

        Returns:
            A regressor with the specified parameters
        """
        return TransformedTargetRegressor(
            regressor=self._make_regressor(params),
            func=np.log1p,
            inverse_func=np.expm1,
        )

    @abstractmethod
    def _make_regressor(self, params: dict[str, int | float | str]) -> Any:
        """Instantiate the base regressor (without preprocessing)"""
        ...

    @abstractmethod
    def save_model(self, regressor: Any, path: Path) -> None:
        """
        Save the model natively with provided library (lightgbm, xgboost, etc.) to the given path

        Args:
            regressor: The Regressor to save
            path: The path to save the model
        """
        ...

    @abstractmethod
    def load_model(self, path: Path) -> Any:
        """
        Load the model natively with provided library (lightgbm, xgboost, etc.) from the given path

        Args:
            path: The path to load the model from

        Returns:
            The loaded Regressor
        """
        ...


class XGBoostModel(Model):
    __xgb: Optional[ModuleType] = None

    def __init__(self, config: Config):
        super().__init__(ModelEnum.XGBOOST, config=config)

    @property
    def xgb(self) -> ModuleType:
        # an ugly hack to be able to switch between models, not requiring all of them
        if XGBoostModel.__xgb is None:
            import xgboost

            XGBoostModel.__xgb = xgboost

        return XGBoostModel.__xgb

    def _make_regressor(self, params: dict[str, int | float | str]) -> "XGBRegressor":
        return self.xgb.XGBRegressor(
            enable_categorical=True,
            tree_method="hist",
            n_jobs=self.config.model.n_jobs,
            random_state=self.config.model.random_state,
            objective="reg:squarederror",
            **params,
        )

    def save_model(self, regressor: "XGBRegressor", path: Path) -> None:
        regressor.save_model(path)

    def load_model(self, path: Path) -> "XGBRegressor":
        regressor = self._make_regressor({})
        regressor.load_model(path)
        return regressor


class LightGBMModel(Model):
    __lgbm: Optional[ModuleType] = None

    def __init__(self, config: Config):
        super().__init__(ModelEnum.LIGHTGBM, config=config)

    @property
    def lgbm(self) -> ModuleType:
        # an ugly hack to be able to switch between models, not requiring all of them
        if LightGBMModel.__lgbm is None:
            import lightgbm

            LightGBMModel.__lgbm = lightgbm

        return LightGBMModel.__lgbm

    def _make_regressor(self, params: dict[str, int | float | str]) -> "LGBMRegressor":
        return self.lgbm.LGBMRegressor(
            n_jobs=self.config.model.n_jobs,
            random_state=self.config.model.random_state,
            verbose=1 if self.config.model.verbose else -1,
            objective="regression",
            **params,
        )

    def save_model(self, regressor: "LGBMRegressor", path: Path) -> None:
        regressor.booster_.save_model(path)

    def load_model(self, path: Path) -> "LGBMRegressor":
        return self.lgbm.Booster(model_file=path)


def get_all_models(config: Optional[Config] = None) -> list[Model]:
    if config is None:
        config = Config()

    return [
        XGBoostModel(config=config),
        LightGBMModel(config=config),
    ]
