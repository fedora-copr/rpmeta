import logging
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor

from rpmeta.config import Config
from rpmeta.constants import ModelEnum, ModelStorageBaseNames
from rpmeta.helpers import save_joblib

if TYPE_CHECKING:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor


logger = logging.getLogger(__name__)


class Model(ABC):
    def __init__(self, name: str, config: Config) -> None:
        self.name = name.lower()
        self.config = config

    def create_regressor(
        self,
        params: dict[str, int | float | str],
    ) -> TransformedTargetRegressor:
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

    def save_regressor(self, regressor: TransformedTargetRegressor, path: Path) -> None:
        """
        Save the whole regressor to the specified path, including model and transformer.

        Args:
            regressor: The TransformedTargetRegressor to save
            path: The directory to save the regressor
        """
        if not path.is_dir():
            raise ValueError(f"Provided path {path} is not a directory.")

        native_model_path = path / ModelStorageBaseNames.NATIVE_MODEL
        self.save_model(regressor.regressor_, native_model_path)
        logger.debug("Saved native model for '%s' to %s", self.name, native_model_path)

        logger.debug("Saving regressor to %s", path)

        # to not save the model twice
        regressor_instance = regressor.regressor_
        regressor.regressor_ = None

        save_joblib(regressor, path, ModelStorageBaseNames.SKELETON_NAME)

        # restore the regressor state
        regressor.regressor_ = regressor_instance

    def load_regressor(self, path: Path) -> TransformedTargetRegressor:
        """
        Load the whole regressor from the specified path, including model and transformer.

        Args:
            path: The path to the directory containing the data

        Returns:
            The loaded TransformedTargetRegressor
        """
        native_model_path = path / ModelStorageBaseNames.NATIVE_MODEL
        if not native_model_path.exists():
            raise FileNotFoundError(f"Native model file {native_model_path} does not exist.")

        skeleton_path = path / f"{ModelStorageBaseNames.SKELETON_NAME}.joblib"
        if not skeleton_path.exists():
            raise FileNotFoundError(f"Model skeleton file {skeleton_path} does not exist.")

        logger.debug("Loading model '%s' from %s", self.name, path)
        model = self.load_model(native_model_path)

        logger.debug("Loading regressor from %s", skeleton_path)
        regressor = joblib.load(skeleton_path)
        regressor.regressor_ = model

        return regressor


class XGBoostModel(Model):
    __xgb: Optional[ModuleType] = None

    def __init__(self, config: Config):
        super().__init__(ModelEnum.XGBOOST, config=config)

    @property
    def xgb(self) -> ModuleType:
        # an ugly hack to be able to switch between models, not requiring all of them
        if XGBoostModel.__xgb is None:
            try:
                import xgboost
            except ImportError:
                logger.error("XGBoost is not installed. Please install it to use XGBoostModel.")
                raise

            XGBoostModel.__xgb = xgboost

        return XGBoostModel.__xgb

    @property
    def _regressor(self) -> type["XGBRegressor"]:
        return self.xgb.XGBRegressor

    def _make_regressor(self, params: dict[str, int | float | str]) -> "XGBRegressor":
        return self._regressor(
            enable_categorical=True,
            tree_method="hist",
            n_jobs=self.config.model.n_jobs,
            random_state=self.config.model.random_state,
            objective="reg:squarederror",
            early_stopping_rounds=self.config.model.xgboost.early_stopping_rounds,
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
            try:
                import lightgbm
            except ImportError:
                logger.error("LightGBM is not installed. Please install it to use LightGBMModel.")
                raise

            LightGBMModel.__lgbm = lightgbm

        return LightGBMModel.__lgbm

    @property
    def _regressor(self) -> type["LGBMRegressor"]:
        return self.lgbm.LGBMRegressor

    def _make_regressor(self, params: dict[str, int | float | str]) -> "LGBMRegressor":
        early_stopping_rounds = 0
        if self.config.model.lightgbm.early_stopping_rounds is not None:
            early_stopping_rounds = self.config.model.lightgbm.early_stopping_rounds

        return self._regressor(
            n_jobs=self.config.model.n_jobs,
            random_state=self.config.model.random_state,
            verbose=1 if self.config.model.verbose else -1,
            objective="regression",
            early_stopping_rounds=early_stopping_rounds,
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
