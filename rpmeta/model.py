import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from rpmeta.config import Config
from rpmeta.constants import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ModelEnum,
    ModelFileExtensions,
    ModelStorageBaseNames,
)

if TYPE_CHECKING:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor


logger = logging.getLogger(__name__)


class Model(ABC):
    TARGET_FUNC: Callable = np.log1p
    INVERSE_FUNC: Callable = np.expm1

    def __init__(self, name: str, config: Config) -> None:
        self.name = name.lower()
        self.config = config
        self._native_model: Any = None

    @abstractmethod
    def make_regressor(self, params: dict[str, int | float | str]) -> Any:
        """Instantiate the native regressor with given hyperparameters."""
        ...

    @abstractmethod
    def get_native_model_extension(self) -> str:
        """
        Get the file extension for the native model file.

        Returns:
            File extension (e.g., 'txt', 'ubj')
        """
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

    @property
    def native_model_filename(self) -> str:
        """
        Get the filename for the native model file.

        Returns:
            Filename for the native model
        """
        return f"{ModelStorageBaseNames.NATIVE_MODEL}.{self.get_native_model_extension()}"

    def save_regressor(self, regressor: Any, path: Path) -> None:
        """
        Save the native model to the specified directory.

        Args:
            regressor: The native regressor to save
            path: The directory to save the model into
        """
        if not path.is_dir():
            raise ValueError(f"Provided path {path} is not a directory.")

        native_model_path = path / self.native_model_filename
        self.save_model(regressor, native_model_path)
        logger.info("Saved native model for '%s' to %s", self.name, native_model_path)

    def load_regressor(self, path: Path) -> None:
        """
        Load the native model from the specified directory.

        Args:
            path: The path to the directory containing the model files
        """
        native_model_path = path / self.native_model_filename
        if not native_model_path.exists():
            raise FileNotFoundError(f"Native model file {native_model_path} does not exist.")

        logger.debug("Loading native model '%s' from %s", self.name, native_model_path)
        self._native_model = self.load_model(native_model_path)
        logger.info("Successfully loaded model from %s", path)

    def prepare_for_prediction(
        self,
        category_maps: dict[str, list[str]],
    ) -> None:
        """
        Pre-build encoding structures

        Builds:
        - ``_cat_encoders``: ``{column: {category_string: int_code}}`` dicts
        - ``_feature_types``: ``["c", ..., "q", ...]`` list for feature types
        - ``_inverse_func``: inverse target transform
        """
        if self._native_model is None:
            raise RuntimeError("Model not loaded. Call load_regressor() first.")

        self._cat_encoders: dict[str, dict[str, int]] = {
            col: {cat: code for code, cat in enumerate(cats)} for col, cats in category_maps.items()
        }
        self._feature_types: list[str] = ["c"] * len(CATEGORICAL_FEATURES) + ["q"] * len(
            NUMERICAL_FEATURES,
        )
        self._inverse_func = self.INVERSE_FUNC

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make prediction on the given DataFrame.

        Args:
            df: DataFrame with features matching ALL_FEATURES

        Returns:
            Array of predictions with the inverse target transform already applied
        """
        if self._native_model is None:
            raise RuntimeError("Model not loaded. Call load_regressor() first.")

        return self.INVERSE_FUNC(self._native_model.predict(df))


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

    def make_regressor(self, params: dict[str, int | float | str]) -> "XGBRegressor":
        logger.debug("Creating XGBoost regressor with params: %s", params)
        return self._regressor(
            enable_categorical=True,
            tree_method="hist",
            n_jobs=self.config.model.n_jobs,
            random_state=self.config.model.random_state,
            objective="reg:squarederror",
            early_stopping_rounds=self.config.model.xgboost.early_stopping_rounds,
            **params,
        )

    def get_native_model_extension(self) -> str:
        return ModelFileExtensions.XGBOOST

    def save_model(self, regressor: "XGBRegressor", path: Path) -> None:
        regressor.save_model(path)

    def load_model(self, path: Path) -> "XGBRegressor":
        regressor = self.xgb.XGBRegressor()
        regressor.load_model(path)
        params = regressor.get_xgb_params()
        logger.debug("Loaded XGBoost model with booster params: %s", params)
        return regressor

    def prepare_for_prediction(
        self,
        category_maps: dict[str, list[str]],
    ) -> None:
        super().prepare_for_prediction(category_maps)
        self._booster = self._native_model.get_booster()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Optimized prediction that bypasses XGBoost's expensive Python-side
        categorical string serialization by encoding features to integer codes
        and constructing a DMatrix directly.
        """
        n_rows = len(df)
        n_cols = len(ALL_FEATURES)
        data = np.empty((n_rows, n_cols), dtype=np.float32)

        for i, feat in enumerate(ALL_FEATURES):
            col = df[feat]
            if feat in self._cat_encoders:
                if isinstance(col.dtype, pd.CategoricalDtype):
                    codes = col.cat.codes.to_numpy(dtype=np.float32)
                    codes[codes < 0] = np.nan
                    data[:, i] = codes
                else:
                    data[:, i] = col.map(
                        self._cat_encoders[feat],
                    ).to_numpy(dtype=np.float32)
            else:
                data[:, i] = col.to_numpy(dtype=np.float32)

        dmatrix = self.xgb.DMatrix(
            data,
            feature_names=list(ALL_FEATURES),
            feature_types=self._feature_types,
        )
        raw_pred = self._booster.predict(dmatrix)
        return self._inverse_func(raw_pred)


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

    def make_regressor(self, params: dict[str, int | float | str]) -> "LGBMRegressor":
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

    def get_native_model_extension(self) -> str:
        return ModelFileExtensions.LIGHTGBM

    def save_model(self, regressor: "LGBMRegressor", path: Path) -> None:
        regressor.booster_.save_model(str(path))

    def load_model(self, path: Path) -> "LGBMRegressor":
        booster = self.lgbm.Booster(model_file=str(path))
        regressor = self._regressor()
        regressor._Booster = booster
        regressor.fitted_ = True
        regressor._n_features = booster.num_feature()
        regressor._n_features_in = booster.num_feature()
        regressor.n_features_in_ = booster.num_feature()

        booster_params = booster.params
        if "objective" in booster_params:
            regressor._objective = booster_params["objective"]
        else:
            regressor._objective = "regression"

        logger.debug(
            "Loaded LightGBM model with %d features, objective: %s",
            regressor.n_features_in_,
            regressor._objective,
        )
        return regressor


def get_all_models(config: Optional[Config] = None) -> list[Model]:
    """
    Get instances of all available model types.

    Args:
        config: Configuration to use. If None, uses default Config.

    Returns:
        List of Model instances for each supported model type.
    """
    if config is None:
        config = Config()

    return [
        XGBoostModel(config=config),
        LightGBMModel(config=config),
    ]


def get_model_by_name(model_name: str, config: Optional[Config] = None) -> Model:
    """
    Get a specific model instance by name.

    Args:
        model_name: The name of the model (case-insensitive)
        config: Configuration to use. If None, uses default Config.

    Returns:
        The Model instance matching the given name.

    Raises:
        ValueError: If no model matches the given name.
    """
    model_name_lower = model_name.lower()
    for model in get_all_models(config):
        if model.name == model_name_lower:
            return model

    available = [m.name for m in get_all_models(config)]
    raise ValueError(
        f"Model '{model_name}' is not supported. Available models: {available}",
    )
