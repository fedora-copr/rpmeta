import logging
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor

from rpmeta.config import Config
from rpmeta.constants import ModelEnum, ModelFileExtensions, ModelStorageBaseNames

if TYPE_CHECKING:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor


logger = logging.getLogger(__name__)


class Model(ABC):
    def __init__(self, name: str, config: Config) -> None:
        self.name = name.lower()
        self.config = config
        self._loaded_regressor: TransformedTargetRegressor | None = None

    @property
    def regressor(self) -> TransformedTargetRegressor:
        """
        Get the loaded regressor.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if self._loaded_regressor is None:
            raise RuntimeError("Model not loaded. Call load_regressor() first.")

        return self._loaded_regressor

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

    @abstractmethod
    def compute_size_penalty(self, regressor: Any, trial: Any = None) -> float:
        """
        Compute the size penalty for the model.

        Args:
            regressor: The fitted regressor (may be a pipeline/wrapper)
            trial: Optional Optuna trial for logging extra info

        Returns:
            Penalty value (float, 0 if not enabled)
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

    def predict(self, data: Any) -> np.ndarray:
        """
        Perform prediction using the loaded regressor.

        Args:
            data: Input data (DataFrame or array-like)

        Returns:
            Predictions array
        """
        return self.regressor.predict(data)

    def save_regressor(self, regressor: TransformedTargetRegressor, path: Path) -> None:
        """
        Save the whole regressor to the specified path, including model and transformer.

        The regressor is split into two parts:
        1. Native model - saved using the library's native format for best compatibility
        2. Skeleton - the TransformedTargetRegressor without the model, saved via joblib

        Args:
            regressor: The TransformedTargetRegressor to save
            path: The directory to save the regressor
        """
        if not path.is_dir():
            raise ValueError(f"Provided path {path} is not a directory.")

        native_model_path = path / self.native_model_filename
        self.save_model(regressor.regressor_, native_model_path)
        logger.debug("Saved native model for '%s' to %s", self.name, native_model_path)

        logger.debug("Saving regressor skeleton to %s", path)

        # to not save the model twice
        regressor_instance = regressor.regressor_
        regressor.regressor_ = None

        skeleton_path = path / f"{ModelStorageBaseNames.SKELETON_NAME}.joblib"
        joblib.dump(regressor, skeleton_path)
        logger.debug("Saved object to %s", skeleton_path)

        # restore the regressor state
        regressor.regressor_ = regressor_instance
        logger.info("Saved regressor skeleton to %s", skeleton_path)

    def load_regressor(self, path: Path) -> "Model":
        """
        Load the whole regressor from the specified path, including model and transformer.

        Args:
            path: The path to the directory containing the data

        Returns:
            Self (Model)
        """
        native_model_path = path / self.native_model_filename
        if not native_model_path.exists():
            raise FileNotFoundError(f"Native model file {native_model_path} does not exist.")

        skeleton_path = path / f"{ModelStorageBaseNames.SKELETON_NAME}.joblib"
        if not skeleton_path.exists():
            raise FileNotFoundError(f"Model skeleton file {skeleton_path} does not exist.")

        logger.debug("Loading native model '%s' from %s", self.name, native_model_path)
        model = self.load_model(native_model_path)

        logger.debug("Loading regressor skeleton from %s", skeleton_path)
        # Use mmap_mode='r' for memory-efficient loading of large models
        regressor: TransformedTargetRegressor = joblib.load(skeleton_path, mmap_mode="r")
        regressor.regressor_ = model

        self._loaded_regressor = regressor
        logger.info("Successfully loaded regressor from %s", path)
        return self


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
        logger.debug("Creating XGBoost regressor with params: %s", params)
        return self._regressor(
            enable_categorical=True,
            tree_method="hist",
            max_depth=0,
            grow_policy="lossguide",
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

    def predict(self, data: Any) -> np.ndarray:
        """
        Perform efficient prediction using XGBoost's inplace_predict.

        Bypasses the sklearn API and DMatrix creation overhead, the Python API
        predict() creates a new DMatrix on every call, causing expensive memory copies
        (see issue #45).

        Args:
            data: Input data (DataFrame or array-like)

        Returns:
            Predictions array
        """
        booster = self.regressor.regressor_.get_booster()
        raw_pred = booster.inplace_predict(data)
        # lost the inverse transformation here because of the inplace_predict :(
        # https://xgboost.readthedocs.io/en/stable/prediction.html#in-place-prediction
        return np.expm1(raw_pred)

    def compute_size_penalty(self, regressor: Any, trial: Any = None) -> float:
        if not self.config.model.xgboost.size_penalty_enabled:
            return 0.0

        booster = regressor.regressor_.get_booster()
        n_nodes = booster.trees_to_dataframe().shape[0]
        penalty = self.config.model.xgboost.size_penalty_lambda * (n_nodes / 100000)

        if trial is not None:
            trial.set_user_attr("n_nodes", n_nodes)
            trial.set_user_attr("size_penalty", penalty)
            logger.debug(
                "XGBoost model size penalty details - n_nodes: %d, penalty: %.6f",
                n_nodes,
                penalty,
            )

        return penalty


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

    def compute_size_penalty(self, regressor: Any, trial: Any = None) -> float:
        if not self.config.model.lightgbm.size_penalty_enabled:
            return 0.0

        booster = regressor.regressor_.booster_
        n_leaves = booster.num_leaves()
        penalty = self.config.model.lightgbm.size_penalty_lambda * (n_leaves / 100000)

        if trial is not None:
            trial.set_user_attr("n_leaves", n_leaves)
            trial.set_user_attr("size_penalty", penalty)
            logger.debug(
                "LightGBM model size penalty details - n_leaves: %d, penalty: %.6f",
                n_leaves,
                penalty,
            )

        return penalty


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
