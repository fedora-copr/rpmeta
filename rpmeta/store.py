"""
Manage the hybrid model storage and loading to help with managing different model versions.
"""

import logging
from pathlib import Path

import joblib

from rpmeta.helpers import save_joblib
from rpmeta.model import Model, get_all_models
from rpmeta.regressor import TransformedTargetRegressor

logger = logging.getLogger(__name__)


class ModelStorage:
    def __init__(
        self,
        model_name: str,
        native_model_name: str = "native_model.txt",
        skeleton_name: str = "model_skeleton",
    ) -> None:
        self.model_name = model_name
        self.native_model_name = native_model_name
        self.skeleton_name = skeleton_name

    def get_model(self, path: Path) -> TransformedTargetRegressor:
        """
        Get the model from the model storage.

        Args:
            path: The path to the model directory.

        Returns:
            The loaded regressor.
        """
        native_model_path = path / self.native_model_name
        if not native_model_path.exists():
            raise FileNotFoundError(f"Native model file {native_model_path} does not exist.")

        skeleton_path = path / f"{self.skeleton_name}.joblib"
        if not skeleton_path.exists():
            raise FileNotFoundError(f"Model skeleton file {skeleton_path} does not exist.")

        logger.debug("Loading model '%s' from %s", self.model_name, path)
        model = self._model_factory()
        regressor = model.load_model(native_model_path)

        logger.debug("Loading model skeleton from %s", skeleton_path)
        regressor_skeleton = joblib.load(skeleton_path)
        regressor_skeleton.regressor = regressor
        return regressor_skeleton

    def save_model(self, model: TransformedTargetRegressor, path: Path) -> None:
        """
        Save the model to the specified path.

        Args:
            model: The model to save.
            path: The path to save the model.
        """
        if not path.is_dir():
            raise ValueError(f"Provided path {path} is not a directory.")

        regressor_instance = model.regressor

        model_instance = self._model_factory()
        native_model_path = path / self.native_model_name
        model_instance.save_model(regressor_instance, native_model_path)
        logger.debug("Saved native model for '%s' to %s", self.model_name, native_model_path)

        # Save the model skeleton
        # Remove the regressor from the model to save only the structure
        model.regressor = None
        save_joblib(model, path, self.skeleton_name)

        # Restore the model to maintain the original structure
        model.regressor = regressor_instance

    def _model_factory(self) -> Model:
        logger.debug("Looking for model implementation '%s' in registered models", self.model_name)
        for model in get_all_models():
            if model.name == self.model_name:
                logger.debug("Found model implementation for '%s'", self.model_name)
                return model

        raise ValueError(f"Model {self.model_name} not found in registered models.")
